#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, ArrayLike

from .._data_plane import EPOCH_ORDER_ALGORITHM, IndexEpochPlan
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import combine_trainable, partition_trainable
from ._diffrax_cde import solve_diffrax_cde
from ._driving_path import AbstractDifferentiableDrivingPath
from ._rough import RoughDifferentialProblem


class NeuralCDEVectorField(StrictModule):
    """Adapt a state-to-matrix callable such as an MLP or KAN to a CDE field."""

    model: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        model: Any,
        /,
        *,
        state_shape: Sequence[int],
        control_dimension: int,
    ):
        if not callable(model):
            raise TypeError("model must be callable.")
        shape = tuple(int(size) for size in state_shape)
        dimension = int(control_dimension)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("state_shape must be non-empty and positive.")
        if dimension <= 0:
            raise ValueError("control_dimension must be positive.")
        probe = jnp.zeros(shape).reshape((-1,))
        value = jnp.asarray(model(probe))
        expected_size = int(np.prod(shape)) * dimension
        if int(value.size) != expected_size:
            raise ValueError(
                "model output size must equal prod(state_shape) * control_dimension; "
                f"expected {expected_size}, got {value.size}."
            )
        self.model = model
        self.state_shape = shape
        self.control_dimension = dimension

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        del time, args
        value = jnp.asarray(self.model(jnp.asarray(state).reshape((-1,))))
        return value.reshape(self.state_shape + (self.control_dimension,))


class NeuralCDETrainingData(StrictModule):
    """Masked irregular observations paired with differentiable driving paths."""

    paths: tuple[AbstractDifferentiableDrivingPath, ...]
    initial_states: Array
    observation_times: Array
    observations: Array
    valid: Array
    observation_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_dimension: int = eqx.field(static=True)
    time_channel: int = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        paths: Sequence[AbstractDifferentiableDrivingPath],
        initial_states: ArrayLike,
        observation_times: ArrayLike,
        observations: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        time_channel: int,
        case_ids: Sequence[str] | None = None,
        data_id: str | None = None,
    ):
        path_values = tuple(paths)
        if not path_values or any(
            not isinstance(path, AbstractDifferentiableDrivingPath)
            for path in path_values
        ):
            raise TypeError("paths must be a non-empty sequence of differentiable paths.")
        initial = jnp.asarray(initial_states)
        times = jnp.asarray(observation_times, dtype=float)
        targets = jnp.asarray(observations)
        num_cases = len(path_values)
        if initial.ndim < 2 or int(initial.shape[0]) != num_cases:
            raise ValueError("initial_states must have a leading path/case axis.")
        state_shape = tuple(int(size) for size in initial.shape[1:])
        if times.ndim != 2 or int(times.shape[0]) != num_cases:
            raise ValueError("observation_times must have shape (case, observation).")
        expected_targets = times.shape + state_shape
        if targets.shape != expected_targets:
            raise ValueError(f"observations must have shape {expected_targets}.")
        mask = (
            jnp.ones(times.shape, dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if mask.shape != times.shape:
            raise ValueError("valid must have the same shape as observation_times.")
        value_shape = tuple(path_values[0].value_shape)
        if len(value_shape) != 1 or any(
            tuple(path.value_shape) != value_shape for path in path_values
        ):
            raise ValueError("All paths must share one vector-valued control shape.")
        dimension = int(value_shape[0])
        channel = int(time_channel)
        if channel < 0 or channel >= dimension:
            raise ValueError("time_channel must index the path control dimension.")

        host_times = np.asarray(jax.device_get(times))
        host_targets = np.asarray(jax.device_get(targets))
        host_initial = np.asarray(jax.device_get(initial))
        host_mask = np.asarray(jax.device_get(mask))
        if not np.all(np.isfinite(host_initial)):
            raise ValueError("initial_states must be finite.")
        selected_indices: list[tuple[int, ...]] = []
        for case_index, path in enumerate(path_values):
            indices = tuple(int(index) for index in np.flatnonzero(host_mask[case_index]))
            if not indices:
                raise ValueError(
                    "Every case must contain at least one valid observation."
                )
            selected_times = host_times[case_index, np.asarray(indices, dtype=np.int32)]
            selected_targets = host_targets[
                case_index, np.asarray(indices, dtype=np.int32)
            ]
            if not np.all(np.isfinite(selected_times)) or not np.all(
                np.isfinite(selected_targets)
            ):
                raise ValueError("Valid observations and their times must be finite.")
            if len(indices) > 1 and not np.all(np.diff(selected_times) > 0.0):
                raise ValueError(
                    "Valid observation times must be strictly increasing per case."
                )
            support = np.asarray(jax.device_get(jnp.stack(path.support)))
            if selected_times[0] < support[0] or selected_times[-1] > support[1]:
                raise ValueError("Valid observation times must lie within path support.")
            declared_time = np.asarray(
                [
                    jax.device_get(
                        path.evaluate(jnp.asarray(time, dtype=times.dtype), "right")[
                            channel
                        ]
                    )
                    for time in selected_times
                ]
            )
            if not np.allclose(
                declared_time,
                selected_times,
                rtol=1e-7,
                atol=10.0 * np.finfo(selected_times.dtype).eps,
            ):
                raise ValueError(
                    "The declared time_channel must equal physical observation time."
                )
            selected_indices.append(indices)

        resolved_case_ids = (
            tuple(f"case:{index}" for index in range(num_cases))
            if case_ids is None
            else tuple(str(value) for value in case_ids)
        )
        if len(resolved_case_ids) != num_cases or any(
            not value for value in resolved_case_ids
        ):
            raise ValueError("case_ids must contain one non-empty ID per path.")
        if len(set(resolved_case_ids)) != len(resolved_case_ids):
            raise ValueError("case_ids must be unique.")
        if data_id is None:
            array_fingerprint = array_tree_fingerprint((initial, times, targets, mask))
            identifier = canonical_fingerprint(
                {
                    "format": "phydrax-neural-cde-data-v1",
                    "path_ids": [path.path_id for path in path_values],
                    "case_ids": list(resolved_case_ids),
                    "time_channel": channel,
                    "arrays": array_fingerprint,
                }
            )
        else:
            identifier = str(data_id)
        if not identifier:
            raise ValueError("data_id must be non-empty.")

        self.paths = path_values
        self.initial_states = initial
        self.observation_times = times
        self.observations = targets
        self.valid = mask
        self.observation_indices = tuple(selected_indices)
        self.case_ids = resolved_case_ids
        self.state_shape = state_shape
        self.control_dimension = dimension
        self.time_channel = channel
        self.data_id = identifier

    @property
    def num_cases(self) -> int:
        return len(self.paths)


class NeuralCDETrainingState(StrictModule):
    """Exactly resumable Optax state at a deterministic mini-batch boundary."""

    vector_field: Any
    optimizer_state: Any
    last_loss: Array
    epoch: int = eqx.field(static=True)
    batch_index: int = eqx.field(static=True)
    update_step: int = eqx.field(static=True)
    training_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    optimizer_id: str = eqx.field(static=True)
    solver_configuration_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)
    ordering: str = eqx.field(static=True)

    def __init__(
        self,
        vector_field: Any,
        optimizer_state: Any,
        last_loss: ArrayLike,
        /,
        *,
        epoch: int,
        batch_index: int,
        update_step: int,
        training_id: str,
        data_id: str,
        optimizer_id: str,
        solver_configuration_id: str,
        dynamics_id: str,
        batch_size: int,
        seed: int,
        shuffle: bool,
    ):
        if not callable(vector_field):
            raise TypeError("vector_field must be callable.")
        if min(int(epoch), int(batch_index), int(update_step), int(seed)) < 0:
            raise ValueError("Training progress and seed must be nonnegative.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        identifiers = (
            str(training_id),
            str(data_id),
            str(optimizer_id),
            str(solver_configuration_id),
            str(dynamics_id),
        )
        if any(not identifier for identifier in identifiers):
            raise ValueError("Training provenance IDs must be non-empty.")
        loss = jnp.asarray(last_loss)
        if loss.shape != ():
            raise ValueError("last_loss must be scalar.")
        self.vector_field = vector_field
        self.optimizer_state = optimizer_state
        self.last_loss = loss
        self.epoch = int(epoch)
        self.batch_index = int(batch_index)
        self.update_step = int(update_step)
        (
            self.training_id,
            self.data_id,
            self.optimizer_id,
            self.solver_configuration_id,
            self.dynamics_id,
        ) = identifiers
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.ordering = EPOCH_ORDER_ALGORITHM


def neural_cde_loss(
    vector_field: Any,
    data: NeuralCDETrainingData,
    /,
    *,
    indices: Sequence[int] | None = None,
    drift: Any | None = None,
    args: Any = None,
    solve_options: Mapping[str, Any] | None = None,
) -> Array:
    """Return mean squared error at each case's physical irregular observation times."""
    if not callable(vector_field):
        raise TypeError("vector_field must be callable.")
    if not isinstance(data, NeuralCDETrainingData):
        raise TypeError("data must be NeuralCDETrainingData.")
    selected = (
        tuple(range(data.num_cases))
        if indices is None
        else tuple(int(index) for index in indices)
    )
    if not selected or any(index < 0 or index >= data.num_cases for index in selected):
        raise ValueError("indices must select at least one valid data case.")
    options = {} if solve_options is None else dict(solve_options)
    if "save_times" in options:
        raise ValueError("solve_options must not override case observation save_times.")
    squared_error = jnp.asarray(0.0)
    scalar_count = 0
    for case_index in selected:
        observation_indices = data.observation_indices[case_index]
        array_indices = jnp.asarray(observation_indices, dtype=jnp.int32)
        times = jnp.take(data.observation_times[case_index], array_indices, axis=0)
        targets = jnp.take(data.observations[case_index], array_indices, axis=0)
        problem = RoughDifferentialProblem(
            vector_field,
            data.initial_states[case_index],
            driver_dimension=data.control_dimension,
            drift=drift,
            args=args,
            problem_id=f"neural-cde:{data.case_ids[case_index]}",
        )
        solution = solve_diffrax_cde(
            problem,
            data.paths[case_index],
            save_times=times,
            **options,
        )
        solve_succeeded = (
            jnp.all(jnp.asarray(solution.valid, dtype=bool))
            & jnp.all(jnp.asarray(solution.successful, dtype=bool))
            & jnp.all(
                jnp.asarray(
                    solution.backend_result == dfx.RESULTS.successful,
                    dtype=bool,
                )
            )
        )
        prediction = eqx.error_if(
            solution.states,
            ~solve_succeeded,
            "Neural CDE solve failed or did not produce every requested save.",
        )
        residual = prediction - targets
        squared_error = squared_error + jnp.real(jnp.vdot(residual, residual))
        scalar_count += len(observation_indices) * int(np.prod(data.state_shape))
    return squared_error / scalar_count


def _training_identifier(
    data: NeuralCDETrainingData,
    /,
    *,
    optimizer_id: str,
    solver_configuration_id: str,
    dynamics_id: str,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> str:
    return canonical_fingerprint(
        {
            "format": "phydrax-neural-cde-training-v1",
            "data_id": data.data_id,
            "optimizer_id": optimizer_id,
            "solver_configuration_id": solver_configuration_id,
            "dynamics_id": dynamics_id,
            "batch_size": batch_size,
            "seed": seed,
            "shuffle": shuffle,
            "ordering": EPOCH_ORDER_ALGORITHM,
        }
    )


def train_neural_cde(
    data: NeuralCDETrainingData,
    /,
    *,
    optimizer: optax.GradientTransformation,
    num_steps: int,
    batch_size: int,
    vector_field: Any | None = None,
    state: NeuralCDETrainingState | None = None,
    optimizer_id: str,
    solver_configuration_id: str = "diffrax-cde-default-v1",
    dynamics_id: str = "neural-cde-dynamics-v1",
    seed: int = 0,
    shuffle: bool = True,
    drift: Any | None = None,
    args: Any = None,
    solve_options: Mapping[str, Any] | None = None,
) -> NeuralCDETrainingState:
    """Run deterministic Optax updates, resumable at every private data-plane batch."""
    if not isinstance(data, NeuralCDETrainingData):
        raise TypeError("data must be NeuralCDETrainingData.")
    if not isinstance(optimizer, optax.GradientTransformation):
        raise TypeError("optimizer must be an Optax GradientTransformation.")
    steps = int(num_steps)
    capacity = int(batch_size)
    random_seed = int(seed)
    if steps < 0:
        raise ValueError("num_steps must be nonnegative.")
    if capacity <= 0:
        raise ValueError("batch_size must be positive.")
    if random_seed < 0:
        raise ValueError("seed must be nonnegative.")
    optimizer_name = str(optimizer_id)
    solver_name = str(solver_configuration_id)
    dynamics_name = str(dynamics_id)
    if not optimizer_name or not solver_name or not dynamics_name:
        raise ValueError(
            "optimizer_id, solver_configuration_id, and dynamics_id must be non-empty."
        )
    training_id = _training_identifier(
        data,
        optimizer_id=optimizer_name,
        solver_configuration_id=solver_name,
        dynamics_id=dynamics_name,
        batch_size=capacity,
        seed=random_seed,
        shuffle=bool(shuffle),
    )

    if state is None:
        if vector_field is None or not callable(vector_field):
            raise TypeError("An initial callable vector_field is required without state.")
        parameters, fixed = partition_trainable(vector_field)
        optimizer_state = optimizer.init(parameters)
        epoch = 0
        batch_index = 0
        update_step = 0
        last_loss = jnp.asarray(jnp.nan)
    else:
        if not isinstance(state, NeuralCDETrainingState):
            raise TypeError("state must be NeuralCDETrainingState or None.")
        if vector_field is not None:
            raise ValueError("Do not pass vector_field when resuming from state.")
        if state.training_id != training_id:
            raise ValueError(
                "Training state provenance does not match this run configuration."
            )
        parameters, fixed = partition_trainable(state.vector_field)
        optimizer_state = state.optimizer_state
        epoch = state.epoch
        batch_index = state.batch_index
        update_step = state.update_step
        last_loss = state.last_loss

    def objective(trainable, fixed_tree, batch_indices):
        model = combine_trainable(trainable, fixed_tree)
        return neural_cde_loss(
            model,
            data,
            indices=batch_indices,
            drift=drift,
            args=args,
            solve_options=solve_options,
        )

    value_and_grad = eqx.filter_value_and_grad(objective)
    for _ in range(steps):
        plan = IndexEpochPlan(
            data.num_cases,
            capacity,
            bool(shuffle),
            random_seed,
            epoch,
            False,
        )
        if batch_index == plan.batch_count:
            epoch += 1
            batch_index = 0
            plan = IndexEpochPlan(
                data.num_cases,
                capacity,
                bool(shuffle),
                random_seed,
                epoch,
                False,
            )
        batch_indices = plan.batch(batch_index)
        last_loss, gradients = value_and_grad(parameters, fixed, batch_indices)
        last_loss = jax.block_until_ready(last_loss)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, parameters
        )
        parameters = eqx.apply_updates(parameters, updates)
        update_step += 1
        batch_index += 1
        if batch_index == plan.batch_count:
            epoch += 1
            batch_index = 0

    trained = combine_trainable(parameters, fixed)
    return NeuralCDETrainingState(
        trained,
        optimizer_state,
        last_loss,
        epoch=epoch,
        batch_index=batch_index,
        update_step=update_step,
        training_id=training_id,
        data_id=data.data_id,
        optimizer_id=optimizer_name,
        solver_configuration_id=solver_name,
        dynamics_id=dynamics_name,
        batch_size=capacity,
        seed=random_seed,
        shuffle=bool(shuffle),
    )


__all__ = [
    "NeuralCDETrainingData",
    "NeuralCDETrainingState",
    "NeuralCDEVectorField",
    "neural_cde_loss",
    "train_neural_cde",
]
