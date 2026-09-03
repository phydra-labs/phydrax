#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, Key
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from ..._training import TrainingController, TrainingProgress
from ...linalg import FactorizationPolicy, inverse, OperatorProperties
from .._layout import StateLayout
from .._trajectory import TrajectoryData
from ._features import AbstractFeatureLibrary, FeatureEvaluation
from ._status import (
    IDENTIFICATION_INFEASIBLE,
    IDENTIFICATION_NONFINITE,
    IDENTIFICATION_SUCCESS,
)
from ._variational_checkpoint import (
    _load_variational_training_checkpoint,
    _save_variational_training_checkpoint,
)
from ._variational_kinetics import (
    _event_mask,
    _feature_pairs,
    _lagged_pair_data,
    _weighted_covariances,
    fit_vac,
    fit_vamp,
    LaggedPairWeighting,
    VACResult,
    VAMPResult,
)


class VariationalKineticTrainingPolicy(StrictModule):
    maximum_steps: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    validation_interval: int = eqx.field(static=True)
    patience: int | None = eqx.field(static=True)
    maximum_transitions: int = eqx.field(static=True)
    reversible: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int = 1000,
        learning_rate: float = 1.0e-3,
        regularization: float = 1.0e-6,
        validation_interval: int = 10,
        patience: int | None = None,
        maximum_transitions: int = 100_000,
        reversible: bool = False,
    ):
        steps = int(maximum_steps)
        interval = int(validation_interval)
        capacity = int(maximum_transitions)
        rate = float(learning_rate)
        ridge = float(regularization)
        patience_ = None if patience is None else int(patience)
        if steps < 0 or interval <= 0 or capacity <= 0:
            raise ValueError(
                "Training steps, validation interval, and capacity are invalid."
            )
        if not isfinite(rate) or rate <= 0.0 or not isfinite(ridge) or ridge <= 0.0:
            raise ValueError(
                "Learning rate and regularization must be finite and positive."
            )
        if patience_ is not None and patience_ <= 0:
            raise ValueError("patience must be positive when provided.")
        self.maximum_steps = steps
        self.learning_rate = rate
        self.regularization = ridge
        self.validation_interval = interval
        self.patience = patience_
        self.maximum_transitions = capacity
        self.reversible = bool(reversible)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "variational-kinetic-training-policy",
                "maximum_steps": steps,
                "learning_rate": rate.hex(),
                "regularization": ridge.hex(),
                "validation_interval": interval,
                "patience": patience_,
                "maximum_transitions": capacity,
                "reversible": bool(reversible),
                "execution": "bounded-full-batch-exact",
            }
        )


class ModelFeatureLibrary(AbstractFeatureLibrary):
    """One pointwise model applied after an optional fixed feature library."""

    model: AbstractArrayModel
    base: AbstractFeatureLibrary | None
    state_layout: StateLayout
    input_layout: None
    feature_names: tuple[str, ...] = eqx.field(static=True)
    library_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        state_layout: StateLayout,
        /,
        *,
        model_id: str,
        base: AbstractFeatureLibrary | None = None,
    ):
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must implement AbstractArrayModel.")
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be StateLayout.")
        if base is not None and not isinstance(base, AbstractFeatureLibrary):
            raise TypeError("base must implement AbstractFeatureLibrary or be None.")
        if base is not None and base.state_layout.layout_id != state_layout.layout_id:
            raise ValueError("Base feature library and state layout differ.")
        input_size = state_layout.size if base is None else base.num_features
        if model.in_size != input_size or not isinstance(model.out_size, int):
            raise ValueError("Model sizes do not match the kinetic feature contract.")
        identifier = str(model_id).strip()
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        self.model = model
        self.base = base
        self.state_layout = state_layout
        self.input_layout = None
        self.feature_names = tuple(f"latent:{index}" for index in range(model.out_size))
        self.library_id = canonical_fingerprint(
            {
                "kind": "model-feature-library",
                "model": identifier,
                "base": None if base is None else base.library_id,
                "layout": state_layout.layout_id,
                "output": model.out_size,
            }
        )

    def evaluate(
        self, states: ArrayLike, inputs: ArrayLike | None = None, /
    ) -> FeatureEvaluation:
        if inputs is not None:
            raise ValueError("ModelFeatureLibrary is state-only.")
        values = jnp.asarray(states)
        rank = len(self.state_layout.shape)
        if rank and tuple(values.shape[-rank:]) != self.state_layout.shape:
            raise ValueError(f"states must end in {self.state_layout.shape}.")
        if self.base is None:
            leading = values.shape if rank == 0 else values.shape[:-rank]
            model_inputs = values.reshape(leading + (self.state_layout.size,))
            source_valid = jnp.all(jnp.isfinite(model_inputs), axis=-1)
        else:
            base = self.base.evaluate(values)
            model_inputs = base.values
            source_valid = base.valid
        flat = model_inputs.reshape((-1, int(self.model.in_size)))
        encoded = jax.vmap(lambda value: self.model(value, key=None))(flat)
        encoded = encoded.reshape(source_valid.shape + (int(self.model.out_size),))
        valid = source_valid & jnp.all(jnp.isfinite(encoded), axis=-1)
        return FeatureEvaluation(
            values=jnp.where(valid[..., None], encoded, 0.0),
            valid=valid,
            feature_names=self.feature_names,
            library_id=self.library_id,
        )


class VariationalCoordinateModel(AbstractArrayModel):
    encoder: AbstractArrayModel
    mean: Array
    rotations: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        encoder: AbstractArrayModel,
        mean: ArrayLike,
        rotations: ArrayLike,
        /,
    ):
        mean_ = jnp.asarray(mean)
        rotations_ = jnp.asarray(rotations)
        if (
            not isinstance(encoder.in_size, int)
            or not isinstance(encoder.out_size, int)
            or mean_.shape != (encoder.out_size,)
            or rotations_.ndim != 2
            or rotations_.shape[0] != encoder.out_size
        ):
            raise ValueError("Encoder and canonical coordinate shapes are incompatible.")
        self.encoder = encoder
        self.mean = mean_
        self.rotations = rotations_
        self.in_size = encoder.in_size
        self.out_size = int(rotations_.shape[1])

    def __call__(self, value, /, *, key=None):
        encoded = self.encoder(value, key=key)
        return contract("i,ij->j", encoded - self.mean, self.rotations)


class VariationalKineticFitHistory(StrictModule):
    steps: Array
    training_scores: Array
    validation_scores: Array
    valid: Array


class VariationalKineticFitResult(StrictModule):
    model: AbstractArrayModel
    coordinate_model: VariationalCoordinateModel
    library: ModelFeatureLibrary
    kinetics: VAMPResult | VACResult
    history: VariationalKineticFitHistory
    progress: TrainingProgress
    valid: Array
    status: Array
    model_id: str = eqx.field(static=True)
    resumed_from_step: int = eqx.field(static=True)
    checkpoint_path: Path | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def transform(self, states: ArrayLike, /) -> Array:
        if isinstance(self.kinetics, VAMPResult):
            return self.kinetics.transform_source(states)
        return self.kinetics.transform(states)


def _training_arrays(
    data: TrajectoryData,
    library: AbstractFeatureLibrary | None,
    lag: int,
    weighting: LaggedPairWeighting,
    /,
) -> tuple[Array, Array, Array, Array]:
    if library is not None:
        source, target, valid, weights, _ = _feature_pairs(
            data, library, lag, weighting, 1.0e-8
        )
        return (
            source.values.reshape((-1, library.num_features)),
            target.values.reshape((-1, library.num_features)),
            valid,
            weights,
        )
    transitions, weights, _ = _lagged_pair_data(data, lag, weighting, 1.0e-8)
    rank = len(data.state_layout.shape)
    source = jnp.where(
        _event_mask(transitions.valid, rank), transitions.source_states, 0.0
    ).reshape((-1, data.state_layout.size))
    target = jnp.where(
        _event_mask(transitions.valid, rank), transitions.target_states, 0.0
    ).reshape((-1, data.state_layout.size))
    valid = transitions.valid.reshape((-1,))
    return source, target, valid, weights.reshape((-1,))


def _score(
    model: AbstractArrayModel,
    source: Array,
    target: Array,
    valid: Array,
    weights: Array,
    regularization: float,
    reversible: bool,
    /,
) -> tuple[Array, Array]:
    encoded_source = jax.vmap(lambda value: model(value, key=None))(source)
    encoded_target = jax.vmap(lambda value: model(value, key=None))(target)
    finite = jnp.all(jnp.isfinite(encoded_source), axis=-1) & jnp.all(
        jnp.isfinite(encoded_target), axis=-1
    )
    active = valid & finite
    _, _, c00, c11, c01 = _weighted_covariances(
        encoded_source, encoded_target, active, weights
    )
    size = int(encoded_source.shape[-1])
    identity = jnp.eye(size, dtype=encoded_source.dtype)
    properties = OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )
    policy = FactorizationPolicy("cholesky")
    if reversible:
        covariance = 0.5 * (c00 + c11) + regularization * identity
        lagged = 0.5 * (c01 + c01.T)
        inverse_covariance = inverse(covariance, policy, properties=properties)
        score = jnp.trace(
            inverse_covariance.value @ lagged @ inverse_covariance.value @ lagged.T
        )
        successful = inverse_covariance.successful
    else:
        inverse_source = inverse(
            c00 + regularization * identity, policy, properties=properties
        )
        inverse_target = inverse(
            c11 + regularization * identity, policy, properties=properties
        )
        score = jnp.trace(inverse_source.value @ c01 @ inverse_target.value @ c01.T)
        successful = inverse_source.successful & inverse_target.successful
    return score, successful & jnp.isfinite(score) & jnp.any(active)


def fit_variational_kinetic_model(
    model: AbstractArrayModel,
    data: TrajectoryData,
    key: Key[Array, ""],
    /,
    *,
    model_id: str,
    policy: VariationalKineticTrainingPolicy | None = None,
    library: AbstractFeatureLibrary | None = None,
    validation_data: TrajectoryData | None = None,
    lag: int = 1,
    n_modes: int = 2,
    weighting: LaggedPairWeighting = LaggedPairWeighting.GEOMETRIC,
    optimizer: optax.GradientTransformation | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 1,
    resume: bool = False,
) -> VariationalKineticFitResult:
    """Fit one deterministic encoder against the exact empirical VAMP-2 objective."""

    if not isinstance(model, AbstractArrayModel) or not isinstance(data, TrajectoryData):
        raise TypeError("model and data must satisfy their declared contracts.")
    policy_ = VariationalKineticTrainingPolicy() if policy is None else policy
    if not isinstance(policy_, VariationalKineticTrainingPolicy):
        raise TypeError("policy must be VariationalKineticTrainingPolicy or None.")
    identifier = str(model_id).strip()
    if not identifier:
        raise ValueError("model_id must be non-empty.")
    input_size = data.state_layout.size if library is None else library.num_features
    if model.in_size != input_size or not isinstance(model.out_size, int):
        raise ValueError("Model sizes do not match the training features.")
    if int(model.out_size) < int(n_modes) or int(n_modes) <= 0:
        raise ValueError("n_modes must be positive and no larger than model output size.")
    source, target, valid, weights = _training_arrays(data, library, lag, weighting)
    if source.shape[0] > policy_.maximum_transitions:
        raise ValueError("Training transitions exceed the exact full-batch capacity.")
    validation = data if validation_data is None else validation_data
    validation_source, validation_target, validation_valid, validation_weights = (
        _training_arrays(validation, library, lag, weighting)
    )
    if validation_source.shape[0] > policy_.maximum_transitions:
        raise ValueError("Validation transitions exceed the exact full-batch capacity.")
    optimizer_ = optax.adam(policy_.learning_rate) if optimizer is None else optimizer
    optimizer_state = optimizer_.init(eqx.filter(model, eqx.is_inexact_array))
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    cadence = int(checkpoint_every)
    if cadence <= 0:
        raise ValueError("checkpoint_every must be positive.")
    if resume and checkpoint is None:
        raise ValueError("resume requires checkpoint_path.")
    metadata = {
        "model_id": identifier,
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "policy_id": policy_.policy_id,
        "training_dataset_id": data.dataset_id,
        "validation_dataset_id": validation.dataset_id,
        "library_id": None if library is None else library.library_id,
        "lag": int(lag),
        "n_modes": int(n_modes),
        "weighting": weighting.value,
        "optimizer_type": f"{type(optimizer_).__module__}.{type(optimizer_).__qualname__}",
    }
    current = model
    controller = TrainingController(total_steps=policy_.maximum_steps, key=key)
    steps: list[int] = []
    training_scores: list[Array] = []
    validation_scores: list[Array] = []
    valid_history: list[Array] = []
    resumed_from_step = 0
    if resume:
        if checkpoint is None or not (checkpoint / "manifest.json").is_file():
            raise ValueError("resume requires an existing checkpoint manifest.")
        loaded = _load_variational_training_checkpoint(
            checkpoint,
            model,
            optimizer_state,
            model,
            TrainingProgress(),
        )
        if loaded.metadata != metadata:
            raise ValueError("Checkpoint metadata does not match this kinetic fit.")
        current = loaded.model
        optimizer_state = loaded.optimizer_state
        controller = TrainingController(
            total_steps=policy_.maximum_steps,
            key=loaded.key,
            progress=loaded.progress,
        )
        controller.best_payload = loaded.best_model
        resumed_from_step = loaded.step
        steps = [int(value) for value in loaded.history["steps"]]
        training_scores = [
            jnp.asarray(value) for value in loaded.history["training_scores"]
        ]
        validation_scores = [
            jnp.asarray(value) for value in loaded.history["validation_scores"]
        ]
        valid_history = [
            jnp.asarray(value, dtype=bool) for value in loaded.history["valid"]
        ]
    else:
        initial_score, initial_valid = _score(
            current,
            validation_source,
            validation_target,
            validation_valid,
            validation_weights,
            policy_.regularization,
            policy_.reversible,
        )
        controller.select(
            float(initial_score), current, step=0, mode="max", patience=policy_.patience
        )
        steps.append(0)
        training_scores.append(initial_score)
        validation_scores.append(initial_score)
        valid_history.append(initial_valid)

    @eqx.filter_jit
    def update(current, state):
        def objective(candidate):
            score, successful = _score(
                candidate,
                source,
                target,
                valid,
                weights,
                policy_.regularization,
                policy_.reversible,
            )
            return -score, successful

        (loss, successful), gradient = eqx.filter_value_and_grad(objective, has_aux=True)(
            current
        )
        updates, next_state = optimizer_.update(gradient, state, current)
        return eqx.apply_updates(current, updates), next_state, -loss, successful

    for step in range(resumed_from_step + 1, policy_.maximum_steps + 1):
        current, optimizer_state, training_score, successful = update(
            current, optimizer_state
        )
        controller.complete_update(step)
        should_stop = False
        if step % policy_.validation_interval == 0 or step == policy_.maximum_steps:
            validation_score, validation_successful = _score(
                current,
                validation_source,
                validation_target,
                validation_valid,
                validation_weights,
                policy_.regularization,
                policy_.reversible,
            )
            complete = successful & validation_successful
            steps.append(step)
            training_scores.append(training_score)
            validation_scores.append(validation_score)
            valid_history.append(complete)
            should_stop = not bool(complete)
            if not should_stop:
                controller.select(
                    float(validation_score),
                    current,
                    step=step,
                    mode="max",
                    patience=policy_.patience,
                )
                should_stop = controller.stop_requested
        if checkpoint is not None and (
            step % cadence == 0 or should_stop or step == policy_.maximum_steps
        ):
            _save_variational_training_checkpoint(
                checkpoint,
                current,
                optimizer_state,
                controller.selected(current),
                controller.progress,
                step=step,
                key=controller.key,
                metadata=metadata,
                history={
                    "steps": list(steps),
                    "training_scores": [float(value) for value in training_scores],
                    "validation_scores": [float(value) for value in validation_scores],
                    "valid": [bool(value) for value in valid_history],
                },
            )
        if should_stop:
            break
    if checkpoint is not None:
        _save_variational_training_checkpoint(
            checkpoint,
            current,
            optimizer_state,
            controller.selected(current),
            controller.progress,
            step=controller.progress.update_step,
            key=controller.key,
            metadata=metadata,
            history={
                "steps": list(steps),
                "training_scores": [float(value) for value in training_scores],
                "validation_scores": [float(value) for value in validation_scores],
                "valid": [bool(value) for value in valid_history],
            },
        )
    selected = controller.selected(current)
    encoded_library = ModelFeatureLibrary(
        selected,
        data.state_layout,
        model_id=identifier,
        base=library,
    )
    if policy_.reversible:
        kinetics: VAMPResult | VACResult = fit_vac(
            data,
            encoded_library,
            lag=lag,
            n_modes=n_modes,
            regularization=policy_.regularization,
            weighting=weighting,
        )
    else:
        kinetics = fit_vamp(
            data,
            encoded_library,
            lag=lag,
            n_modes=n_modes,
            regularization=policy_.regularization,
            weighting=weighting,
        )
    if isinstance(kinetics, VAMPResult):
        coordinate_model = VariationalCoordinateModel(
            selected,
            kinetics.model.source_mean,
            kinetics.model.source_rotations,
        )
    else:
        coordinate_model = VariationalCoordinateModel(
            selected,
            kinetics.mean,
            kinetics.components,
        )
    history = VariationalKineticFitHistory(
        steps=jnp.asarray(steps, dtype=jnp.int32),
        training_scores=jnp.asarray(training_scores),
        validation_scores=jnp.asarray(validation_scores),
        valid=jnp.asarray(valid_history, dtype=bool),
    )
    valid_result = kinetics.valid & jnp.all(history.valid)
    status = jnp.where(
        ~jnp.all(jnp.isfinite(history.training_scores)),
        IDENTIFICATION_NONFINITE,
        jnp.where(valid_result, IDENTIFICATION_SUCCESS, IDENTIFICATION_INFEASIBLE),
    ).astype(jnp.int32)
    result_id = canonical_fingerprint(
        {
            "kind": "variational-kinetic-fit",
            "model": identifier,
            "policy": policy_.policy_id,
            "training": data.dataset_id,
            "validation": validation.dataset_id,
            "lag": int(lag),
            "modes": int(n_modes),
            "weighting": weighting.value,
        }
    )
    return VariationalKineticFitResult(
        model=selected,
        coordinate_model=coordinate_model,
        library=encoded_library,
        kinetics=kinetics,
        history=history,
        progress=controller.progress,
        valid=valid_result,
        status=status,
        model_id=identifier,
        resumed_from_step=resumed_from_step,
        checkpoint_path=checkpoint,
        policy_id=policy_.policy_id,
        result_id=result_id,
    )


__all__ = [
    "ModelFeatureLibrary",
    "VariationalCoordinateModel",
    "VariationalKineticFitHistory",
    "VariationalKineticFitResult",
    "VariationalKineticTrainingPolicy",
    "fit_variational_kinetic_model",
]
