#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._trajectory import TrajectoryData


_KEY_POLICY_ID = "trajectory-window:case-start-depth-objective"


class _NeuralWindowBatch(eqx.Module):
    """One gathered batch of lazy parent/start trajectory windows."""

    parent_index: Array
    start_index: Array
    coordinates: Array
    states: Array
    inputs: Array | None
    prefix_valid: Array
    weights: Array
    max_horizon: int = eqx.field(static=True)

    @property
    def size(self) -> int:
        return int(self.parent_index.shape[0])


class _NeuralWindowSource:
    """Lazy index plan over flattened trajectory parents and start nodes."""

    def __init__(
        self,
        trajectory: TrajectoryData,
        /,
        *,
        max_horizon: int,
        step_size: float,
        step_rtol: float,
        step_atol: float,
    ):
        if not isinstance(trajectory, TrajectoryData):
            raise TypeError("trajectory must be a TrajectoryData.")
        horizon = int(max_horizon)
        if horizon < 1:
            raise ValueError("max_horizon must be positive.")
        if horizon >= trajectory.capacity:
            raise ValueError("max_horizon must be smaller than trajectory capacity.")
        self.trajectory = trajectory
        self.max_horizon = horizon
        self.step_size = float(step_size)
        self.step_rtol = float(step_rtol)
        self.step_atol = float(step_atol)
        coordinates = np.asarray(trajectory.coordinates)
        valid_transitions = np.asarray(trajectory.transition_valid, dtype=bool)
        intervals = coordinates[..., 1:] - coordinates[..., :-1]
        fixed = np.isfinite(intervals) & np.isclose(
            intervals,
            self.step_size,
            rtol=self.step_rtol,
            atol=self.step_atol,
        )
        if np.any(valid_transitions & ~fixed):
            raise ValueError(
                "Every valid trajectory transition must match the declared step_size."
            )
        self.starts_per_parent = trajectory.capacity - 1
        self.size = trajectory.num_cases * self.starts_per_parent
        self.fingerprint = _trajectory_content_fingerprint(
            trajectory,
            max_horizon=horizon,
            step_size=self.step_size,
            step_rtol=self.step_rtol,
            step_atol=self.step_atol,
        )

    def ordered_indices(self, epoch: int, /, *, shuffle: bool, seed: int) -> np.ndarray:
        if int(epoch) < 0:
            raise ValueError("epoch must be nonnegative.")
        indices = jnp.arange(self.size, dtype=jnp.int32)
        if shuffle:
            indices = jr.permutation(jr.fold_in(jr.key(seed), int(epoch)), indices)
        return np.asarray(indices)

    def prepare(self, indices: np.ndarray | Array, /) -> _NeuralWindowBatch:
        logical = jnp.asarray(indices, dtype=jnp.int32)
        if logical.ndim != 1 or int(logical.size) < 1:
            raise ValueError("Window indices must be a nonempty vector.")
        parents = logical // self.starts_per_parent
        starts = logical % self.starts_per_parent
        trajectory = self.trajectory
        horizon = self.max_horizon
        capacity = trajectory.capacity
        case_count = trajectory.num_cases
        node_offsets = jnp.arange(horizon + 1, dtype=jnp.int32)
        step_offsets = jnp.arange(horizon, dtype=jnp.int32)
        node_indices = starts[:, None] + node_offsets[None, :]
        step_indices = starts[:, None] + step_offsets[None, :]
        node_in_range = node_indices < capacity
        step_in_range = step_indices < capacity - 1
        clamped_nodes = jnp.clip(node_indices, 0, capacity - 1)
        clamped_steps = jnp.clip(step_indices, 0, capacity - 2)

        coordinates = trajectory.coordinates.reshape((case_count, capacity))
        states = trajectory.states.reshape(
            (case_count, capacity) + trajectory.state_layout.shape
        )
        sample_valid = trajectory.sample_valid.reshape((case_count, capacity))
        transition_valid = trajectory.transition_valid.reshape((case_count, capacity - 1))
        resets = trajectory.reset_mask.reshape((case_count, capacity - 1))
        weights = trajectory.weights.reshape((case_count, capacity))

        gathered_coordinates = coordinates[parents[:, None], clamped_nodes]
        gathered_states = states[parents[:, None], clamped_nodes]
        gathered_sample_valid = sample_valid[parents[:, None], clamped_nodes]
        gathered_transitions = transition_valid[parents[:, None], clamped_steps]
        gathered_resets = resets[parents[:, None], clamped_steps]
        gathered_weights = weights[parents[:, None], clamped_nodes]

        coordinate_finite = jnp.isfinite(gathered_coordinates)
        state_finite = _event_finite(
            gathered_states,
            len(trajectory.state_layout.shape),
        )
        node_valid = (
            node_in_range & gathered_sample_valid & coordinate_finite & state_finite
        )
        interval = gathered_coordinates[:, 1:] - gathered_coordinates[:, :-1]
        fixed_step = jnp.isclose(
            interval,
            self.step_size,
            rtol=self.step_rtol,
            atol=self.step_atol,
        )
        step_valid = (
            step_in_range
            & node_valid[:, :-1]
            & node_valid[:, 1:]
            & gathered_transitions
            & ~gathered_resets
            & fixed_step
        )

        gathered_inputs: Array | None
        if trajectory.inputs is None:
            gathered_inputs = None
        else:
            assert trajectory.input_layout is not None
            assert trajectory.input_valid is not None
            input_count = (
                capacity if trajectory.input_alignment == "samples" else capacity - 1
            )
            input_values = trajectory.inputs.reshape(
                (case_count, input_count) + trajectory.input_layout.shape
            )
            input_valid = trajectory.input_valid.reshape((case_count, input_count))
            clamped_inputs = jnp.clip(step_indices, 0, input_count - 1)
            input_in_range = step_indices < input_count
            gathered_inputs = input_values[parents[:, None], clamped_inputs]
            gathered_input_valid = input_valid[parents[:, None], clamped_inputs]
            finite_inputs = _event_finite(
                gathered_inputs,
                len(trajectory.input_layout.shape),
            )
            control_valid = input_in_range & gathered_input_valid & finite_inputs
            step_valid = step_valid & control_valid
            gathered_inputs = _sanitize_events(
                gathered_inputs,
                control_valid,
                len(trajectory.input_layout.shape),
            )

        prefix_valid = jnp.cumprod(step_valid.astype(jnp.int32), axis=1).astype(bool)
        sanitized_coordinates = jnp.where(
            node_valid,
            gathered_coordinates,
            jnp.zeros_like(gathered_coordinates),
        )
        sanitized_states = _sanitize_events(
            gathered_states,
            node_valid,
            len(trajectory.state_layout.shape),
        )
        sanitized_weights = jnp.where(
            node_valid & jnp.isfinite(gathered_weights) & (gathered_weights > 0.0),
            gathered_weights,
            jnp.zeros_like(gathered_weights),
        )
        return _NeuralWindowBatch(
            parent_index=parents,
            start_index=starts,
            coordinates=sanitized_coordinates,
            states=sanitized_states,
            inputs=gathered_inputs,
            prefix_valid=prefix_valid,
            weights=sanitized_weights,
            max_horizon=horizon,
        )


def _event_finite(values: Array, event_rank: int, /) -> Array:
    finite = jnp.isfinite(values)
    if event_rank:
        axes = tuple(range(finite.ndim - event_rank, finite.ndim))
        finite = jnp.all(finite, axis=axes)
    return finite


def _sanitize_events(values: Array, valid: Array, event_rank: int, /) -> Array:
    mask = valid
    for _ in range(event_rank):
        mask = mask[..., None]
    return jnp.where(mask, values, jnp.zeros_like(values))


def _active_window_evidence(
    batch: _NeuralWindowBatch,
    active_horizon: Array,
    /,
) -> tuple[Array, Array]:
    """Return whole-prefix eligibility and endpoint evidence at a traced horizon."""

    horizon = jnp.asarray(active_horizon, dtype=jnp.int32)
    horizon = jnp.clip(horizon, 1, batch.max_horizon)
    step_index = horizon - 1
    prefix_valid = jnp.take(batch.prefix_valid, step_index, axis=1)
    endpoint_weight = jnp.take(batch.weights, horizon, axis=1)
    evidence = jnp.sqrt(batch.weights[:, 0] * endpoint_weight)
    eligible = prefix_valid & jnp.isfinite(evidence) & (evidence > 0.0)
    return eligible, jnp.where(eligible, evidence, jnp.zeros_like(evidence))


def _semantic_window_keys(
    root_key: Array,
    parent_index: Array,
    start_index: Array,
    depth: Array,
    objective_site: int,
    /,
) -> Array:
    """Derive per-window keys from stable semantic rollout coordinates."""

    def one(parent, start):
        key = jr.fold_in(root_key, parent)
        key = jr.fold_in(key, start)
        key = jr.fold_in(key, depth)
        return jr.fold_in(key, int(objective_site))

    return jax.vmap(one)(parent_index, start_index)


def _trajectory_content_fingerprint(
    trajectory: TrajectoryData,
    /,
    *,
    max_horizon: int,
    step_size: float,
    step_rtol: float,
    step_atol: float,
) -> str:
    arrays = array_tree_fingerprint(
        {
            "coordinates": trajectory.coordinates,
            "states": trajectory.states,
            "sample_valid": trajectory.sample_valid,
            "transition_valid": trajectory.transition_valid,
            "reset_mask": trajectory.reset_mask,
            "weights": trajectory.weights,
            "inputs": trajectory.inputs,
            "input_valid": trajectory.input_valid,
        }
    )
    return canonical_fingerprint(
        {
            "arrays": arrays,
            "state_layout": trajectory.state_layout.layout_id,
            "input_layout": (
                None
                if trajectory.input_layout is None
                else trajectory.input_layout.layout_id
            ),
            "input_alignment": trajectory.input_alignment,
            "case_shape": list(trajectory.case_shape),
            "case_axes": list(trajectory.case_axes),
            "case_axis_roles": list(trajectory.case_axis_roles),
            "capacity": trajectory.capacity,
            "ordering": "flattened-parent-major:start-minor",
            "max_horizon": int(max_horizon),
            "step_size": float(step_size),
            "step_rtol": float(step_rtol),
            "step_atol": float(step_atol),
        }
    )


__all__: list[str] = []
