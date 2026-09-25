#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PyTree

from .._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import combine_parameters, ParameterOwner
from .._training_checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_metadata,
    save_training_checkpoint,
)
from .._training_kernel import (
    build_training_checkpoint,
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    run_training_attempt,
    TrainingKernelSpec,
    TrainingRejectionBudgetError,
)
from .._training_objective import _ObjectiveContribution
from ._checkpoint import (
    _runtime_versions,
    checkpoint_compatibility,
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
)
from ._posterior import PosteriorProblem
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField


_OBJECTIVE_ID = "reverse-kl"
_CHECKPOINT_FORMAT = "phydrax-uq-variational"
_FINAL_DRAWS_ADDRESS = SampleAddress(
    "uq.variational", "final-draws", role="posterior-draws"
)


def _tree_nbytes(tree: Any, /) -> int:
    return sum(leaf.nbytes for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf))


def _tree_all_finite(tree: Any, /) -> Array:
    leaves = [leaf for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf)]
    if not leaves:
        return jnp.asarray(False)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


class AbstractVariationalFamily(StrictModule, ParameterOwner):
    """Normalized distribution over unconstrained posterior coordinates.

    Unannotated inexact array leaves are the variational parameters
    (`ParameterOwner`); data held by a family is declared with `fixed_field`.
    """

    @property
    @abstractmethod
    def family_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def sample_and_log_prob(
        self,
        key: Array,
        /,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[PyTree[Array], Array]:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, value: PyTree[Any], /) -> Array:
        raise NotImplementedError


class MeanFieldGaussianFamily(AbstractVariationalFamily):
    """Reparameterized diagonal Gaussian over an arbitrary array PyTree."""

    location: PyTree[Array]
    raw_scale: PyTree[Array]
    scale_floor: float = eqx.field(static=True)

    def __init__(
        self,
        location: PyTree[Any],
        raw_scale: PyTree[Any],
        /,
        *,
        scale_floor: float = 1e-6,
    ):
        location_tree = jax.tree.map(jnp.asarray, location)
        raw_scale_tree = jax.tree.map(jnp.asarray, raw_scale)
        if jax.tree.structure(location_tree) != jax.tree.structure(raw_scale_tree):
            raise ValueError("location and raw_scale must share one PyTree structure.")
        locations = jax.tree.leaves(location_tree)
        raw_scales = jax.tree.leaves(raw_scale_tree)
        if not locations:
            raise ValueError("A variational family requires at least one array leaf.")
        for location_leaf, raw_scale_leaf in zip(
            locations,
            raw_scales,
            strict=True,
        ):
            if location_leaf.shape != raw_scale_leaf.shape:
                raise ValueError("location and raw_scale leaf shapes must agree.")
            if not jnp.issubdtype(location_leaf.dtype, jnp.floating):
                raise TypeError("Variational locations must be real floating arrays.")
            if not jnp.issubdtype(raw_scale_leaf.dtype, jnp.floating):
                raise TypeError("Variational scales must be real floating arrays.")
        floor = float(scale_floor)
        if not isfinite(floor) or floor <= 0.0:
            raise ValueError("scale_floor must be positive and finite.")
        self.location = location_tree
        self.raw_scale = raw_scale_tree
        self.scale_floor = floor

    @classmethod
    def from_position(
        cls,
        position: PyTree[Any],
        /,
        *,
        initial_scale: float = 0.1,
        scale_floor: float = 1e-6,
    ) -> "MeanFieldGaussianFamily":
        scale = float(initial_scale)
        floor = float(scale_floor)
        if not isfinite(scale) or scale <= floor:
            raise ValueError("initial_scale must be finite and exceed scale_floor.")
        location = jax.tree.map(jnp.asarray, position)
        raw_value = jnp.log(jnp.expm1(jnp.asarray(scale - floor)))
        raw_scale = jax.tree.map(
            lambda leaf: jnp.full_like(leaf, raw_value),
            location,
        )
        return cls(location, raw_scale, scale_floor=floor)

    @property
    def family_id(self) -> str:
        return "mean-field-gaussian"

    @property
    def scale(self) -> PyTree[Array]:
        return jax.tree.map(
            lambda value: jax.nn.softplus(value) + self.scale_floor,
            self.raw_scale,
        )

    def sample_and_log_prob(
        self,
        key: Array,
        /,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[PyTree[Array], Array]:
        shape = tuple(sample_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("sample_shape dimensions must be positive.")
        locations, treedef = jax.tree.flatten(self.location)
        scales = jax.tree.leaves(self.scale)
        keys = jr.split(key, len(locations))
        samples = treedef.unflatten(
            [
                location
                + scale
                * jr.normal(
                    sample_key,
                    shape + location.shape,
                    dtype=location.dtype,
                )
                for sample_key, location, scale in zip(
                    keys,
                    locations,
                    scales,
                    strict=True,
                )
            ]
        )
        return samples, self.log_prob(samples)

    def log_prob(self, value: PyTree[Any], /) -> Array:
        if jax.tree.structure(value) != jax.tree.structure(self.location):
            raise ValueError("value has an incompatible variational PyTree structure.")
        terms = []
        log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi))
        for sample, location, scale in zip(
            jax.tree.leaves(value),
            jax.tree.leaves(self.location),
            jax.tree.leaves(self.scale),
            strict=True,
        ):
            sample_array = jnp.asarray(sample)
            if location.shape and (
                sample_array.ndim < location.ndim
                or sample_array.shape[-location.ndim :] != location.shape
            ):
                raise ValueError(
                    "A variational value leaf has an invalid trailing shape."
                )
            standardized = (sample_array - location) / scale
            element = -0.5 * (
                jnp.square(standardized) + log_two_pi + 2.0 * jnp.log(scale)
            )
            axes = tuple(range(element.ndim - location.ndim, element.ndim))
            terms.append(jnp.sum(element, axis=axes) if axes else element)
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total


class VariationalConfig(StrictModule):
    """Static reverse-KL optimization and recording controls."""

    num_steps: int = eqx.field(static=True)
    samples_per_step: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    gradient_clip: float = eqx.field(static=True)
    record_every: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_steps: int = 1000,
        samples_per_step: int = 16,
        learning_rate: float = 1e-3,
        gradient_clip: float = 100.0,
        record_every: int = 10,
    ):
        steps = int(num_steps)
        samples = int(samples_per_step)
        interval = int(record_every)
        rate = float(learning_rate)
        clipping = float(gradient_clip)
        if steps < 1 or samples < 1 or interval < 1:
            raise ValueError(
                "Variational step, sample, and record counts must be positive."
            )
        if not isfinite(rate) or rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite.")
        if not isfinite(clipping) or clipping <= 0.0:
            raise ValueError("gradient_clip must be positive and finite.")
        self.num_steps = steps
        self.samples_per_step = samples
        self.learning_rate = rate
        self.gradient_clip = clipping
        self.record_every = interval

    def as_dict(self) -> dict[str, int | float]:
        return {
            "num_steps": self.num_steps,
            "samples_per_step": self.samples_per_step,
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
            "record_every": self.record_every,
        }


class VariationalDiagnostics(StrictModule):
    """Recorded ELBO, gradient norms, and finite optimization state."""

    steps: Array
    elbo: Array
    gradient_norm: Array
    finite: Array
    completed_steps: int = eqx.field(static=True)


class VariationalResult(StrictModule):
    """Fitted normalized posterior approximation and unconstrained draws."""

    problem: PosteriorProblem
    family: AbstractVariationalFamily
    samples: PyTree[Array]
    unconstrained_samples: PyTree[Array]
    log_target: Array
    log_variational: Array
    diagnostics: VariationalDiagnostics
    root_key: Array
    config: VariationalConfig
    duration_seconds: float = eqx.field(static=True)
    optimization_duration_seconds: float = eqx.field(static=True)
    sampling_duration_seconds: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)
    family_memory_bytes: int = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @property
    def num_draws(self) -> int:
        return self.log_target.shape[0]

    def predict(
        self,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        draw_dim: str = "__phydra_uq_draw",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        return predict_from_position_samples(
            self.problem,
            self.unconstrained_samples,
            *args,
            sample_dims=(draw_dim,),
            sample_sources=("epistemic",),
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )

    def sample_observations(
        self,
        key: Array,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        draw_dim: str = "__phydra_uq_draw",
        observation_dim: str = "__phydra_uq_observation",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        return sample_observations_from_position_samples(
            self.problem,
            key,
            self.unconstrained_samples,
            *args,
            sample_dims=(draw_dim,),
            sample_sources=("epistemic",),
            observation_dim=observation_dim,
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )


class _ReverseKLObjective(StrictModule):
    """Reparameterized reverse-KL estimate; the payload is the posterior problem.

    A nonfinite draw, density, or loss makes the numerator nonfinite, so the
    kernel rolls the attempt back instead of committing it.
    """

    samples_per_step: int = eqx.field(static=True)

    def __call__(self, parameters, model_state, fixed, problem, keys):
        family = combine_parameters(parameters, model_state, fixed)
        positions, log_variational = family.sample_and_log_prob(
            keys.attempt_key("reparameterization"),
            sample_shape=(self.samples_per_step,),
        )
        log_target = jax.vmap(problem.log_density)(positions)
        loss = jnp.mean(log_variational - log_target)
        finite = (
            jnp.isfinite(loss)
            & jnp.all(jnp.isfinite(log_variational))
            & jnp.all(jnp.isfinite(log_target))
            & _tree_all_finite(positions)
        )
        numerator = jnp.where(finite, loss, jnp.full_like(loss, jnp.nan))
        return _ObjectiveContribution(numerator, 1.0), model_state, loss


_CHECKPOINT_METADATA = frozenset(
    {"compatibility", "versions", "recorded_count", "history_dtype", "duration_seconds"}
)


def _history_template(count: int, dtype: Any, /) -> dict[str, Array]:
    return {
        "recorded_steps": jnp.zeros((count,), dtype=jnp.int32),
        "elbo": jnp.zeros((count,), dtype=dtype),
        "gradient_norm": jnp.zeros((count,), dtype=dtype),
        "finite": jnp.zeros((count,), dtype=jnp.bool_),
    }


def _read_variational_checkpoint(source: Path, kernel, template, /, *, compatibility):
    metadata = read_training_checkpoint_metadata(source, format=_CHECKPOINT_FORMAT)
    if set(metadata) != _CHECKPOINT_METADATA:
        raise CheckpointCorruptionError(
            "Variational checkpoint metadata must use the current canonical fields."
        )
    if metadata["versions"] != _runtime_versions():
        raise CheckpointCompatibilityError(
            "Checkpoint PhydraX or BlackJAX version does not match the runtime."
        )
    if metadata["compatibility"] != dict(compatibility):
        raise CheckpointCompatibilityError(
            "Variational checkpoint does not belong to this problem and run settings."
        )
    count = metadata["recorded_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise CheckpointCorruptionError("Variational recorded history count is invalid.")
    loaded = load_training_checkpoint(
        source,
        kernel,
        template,
        _history_template(count, jnp.dtype(str(metadata["history_dtype"]))),
        format=_CHECKPOINT_FORMAT,
    )
    return loaded.restored.state, loaded.extra, float(metadata["duration_seconds"])


def fit_variational(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    family: AbstractVariationalFamily | None = None,
    config: VariationalConfig | None = None,
    num_samples: int = 1000,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> VariationalResult:
    """Fit a normalized reverse-KL posterior in unconstrained coordinates.

    Every step is one attempt of the shared training kernel (`MODEL` root
    authority, one data-fit objective, clipped Adam). Reparameterization draws
    use the attempt-addressed key `SampleAddress("training", "reverse-kl",
    target=("reparameterization",), role="attempt")` folded with the attempt
    cursor. A nonfinite step rolls back and raises `FloatingPointError`.
    Checkpoints are shared kernel checkpoints; files from other layouts fail
    closed.
    """

    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    family_ = (
        MeanFieldGaussianFamily.from_position(problem.initial_position)
        if family is None
        else family
    )
    if not isinstance(family_, AbstractVariationalFamily):
        raise TypeError("family must implement AbstractVariationalFamily or be None.")
    config_ = VariationalConfig() if config is None else config
    if not isinstance(config_, VariationalConfig):
        raise TypeError("config must be VariationalConfig or None.")
    draws = int(num_samples)
    if draws < 1:
        raise ValueError("num_samples must be positive.")
    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else (Path(resume_from) if resume_from is not None else None)
    )
    if checkpoint_every is not None and destination is None:
        raise ValueError("checkpoint_every requires checkpoint_path or resume_from.")
    checkpoint_interval = (
        min(100, config_.num_steps) if checkpoint_every is None else int(checkpoint_every)
    )
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_every must be positive.")
    if destination is not None and (checkpoint_id is None or not str(checkpoint_id)):
        raise ValueError("checkpoint_id is required for variational checkpointing.")

    settings = {
        "algorithm": "reverse-kl-variational",
        "family_id": family_.family_id,
        "config": {
            name: value
            for name, value in config_.as_dict().items()
            if name != "num_steps"
        },
        "num_samples": draws,
        "root_key": [int(value) for value in jr.key_data(key).reshape(-1)],
    }
    compatibility = (
        checkpoint_compatibility(
            problem,
            checkpoint_id=str(checkpoint_id),
            settings=settings,
        )
        if destination is not None
        else None
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config_.gradient_clip),
        optax.adam(config_.learning_rate),
    )
    kernel = prepare_training_kernel(
        family_,
        (
            KernelObjective(
                objective_id=_OBJECTIVE_ID,
                kind=ObjectiveKind.DATA_FIT,
                route=DerivativeRoute.DIRECT,
                fn=_ReverseKLObjective(config_.samples_per_step),
            ),
        ),
        TrainingKernelSpec(
            OptaxUpdateRule(
                optimizer,
                rule_id=canonical_fingerprint(
                    {
                        "kind": "reverse-kl-clipped-adam",
                        "gradient_clip": config_.gradient_clip.hex(),
                        "learning_rate": config_.learning_rate.hex(),
                    }
                ),
            ),
            context="fit_variational",
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.MODEL,
    )
    state = kernel.init(family_, key)
    started = perf_counter()

    recorded_steps: list[Any] = []
    elbo_history: list[Any] = []
    gradient_history: list[Any] = []
    finite_history: list[Any] = []
    previous_duration = 0.0
    if resume_from is not None:
        if compatibility is None:
            raise RuntimeError("Variational resume compatibility was not initialized.")
        state, history, previous_duration = _read_variational_checkpoint(
            Path(resume_from), kernel, state, compatibility=compatibility
        )
        recorded_steps = list(history["recorded_steps"])
        elbo_history = list(history["elbo"])
        gradient_history = list(history["gradient_norm"])
        finite_history = list(history["finite"])
    completed = int(state.accepted_cursor)
    if completed > config_.num_steps:
        raise ValueError("Checkpoint exceeds the configured variational steps.")

    optimization_started = perf_counter()
    while completed < config_.num_steps:
        try:
            state, evidence = run_training_attempt(kernel, state, problem)
        except TrainingRejectionBudgetError as error:
            raise FloatingPointError(
                f"Variational optimization became nonfinite at step {completed + 1}; "
                "the family was rolled back to its last accepted state."
            ) from error
        completed += 1
        if completed % config_.record_every == 0 or completed == config_.num_steps:
            recorded_steps.append(jnp.asarray(completed, dtype=jnp.int32))
            elbo_history.append(-evidence.diagnostics[0])
            gradient_history.append(evidence.gradient_norm)
            finite_history.append(evidence.finite)
        if (
            destination is not None
            and compatibility is not None
            and (completed % checkpoint_interval == 0 or completed == config_.num_steps)
        ):
            history_dtype = evidence.diagnostics[0].dtype
            save_training_checkpoint(
                destination,
                build_training_checkpoint(kernel, state),
                {
                    "recorded_steps": jnp.asarray(recorded_steps, dtype=jnp.int32),
                    "elbo": jnp.asarray(elbo_history, dtype=history_dtype),
                    "gradient_norm": jnp.asarray(gradient_history, dtype=history_dtype),
                    "finite": jnp.asarray(finite_history, dtype=jnp.bool_),
                },
                format=_CHECKPOINT_FORMAT,
                metadata={
                    "compatibility": compatibility,
                    "versions": _runtime_versions(),
                    "recorded_count": len(recorded_steps),
                    "history_dtype": jnp.dtype(history_dtype).name,
                    "duration_seconds": previous_duration + perf_counter() - started,
                },
            )
    optimization_duration = perf_counter() - optimization_started
    fitted_family = kernel.tree(state)
    sampling_started = perf_counter()
    unconstrained_samples, log_variational = fitted_family.sample_and_log_prob(
        derive_key(key, _FINAL_DRAWS_ADDRESS),
        sample_shape=(draws,),
    )
    log_target = jax.vmap(problem.log_density)(unconstrained_samples)
    samples = problem.parameter_space.constrain(unconstrained_samples)
    jax.block_until_ready(log_target)
    sampling_duration = perf_counter() - sampling_started
    diagnostics = VariationalDiagnostics(
        steps=jnp.asarray(recorded_steps, dtype=jnp.int32),
        elbo=jnp.asarray(elbo_history),
        gradient_norm=jnp.asarray(gradient_history),
        finite=jnp.asarray(finite_history, dtype=jnp.bool_),
        completed_steps=completed,
    )
    duration = previous_duration + perf_counter() - started
    return VariationalResult(
        problem=problem,
        family=fitted_family,
        samples=samples,
        unconstrained_samples=unconstrained_samples,
        log_target=log_target,
        log_variational=log_variational,
        diagnostics=diagnostics,
        root_key=jnp.asarray(key),
        config=config_,
        duration_seconds=duration,
        optimization_duration_seconds=optimization_duration,
        sampling_duration_seconds=sampling_duration,
        sample_memory_bytes=(
            _tree_nbytes(samples)
            + _tree_nbytes(unconstrained_samples)
            + log_target.nbytes
            + log_variational.nbytes
        ),
        family_memory_bytes=_tree_nbytes(fitted_family),
        approximation_id=f"reverse-kl/{fitted_family.family_id}",
    )


__all__ = [
    "AbstractVariationalFamily",
    "fit_variational",
    "MeanFieldGaussianFamily",
    "VariationalConfig",
    "VariationalDiagnostics",
    "VariationalResult",
]
