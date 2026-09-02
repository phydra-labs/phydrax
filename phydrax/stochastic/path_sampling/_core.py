#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-shape path state, region algebra, and dynamics-kernel contracts."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


PATH_PROPAGATION_SUCCESS = 0
PATH_PROPAGATION_KERNEL_FAILURE = 1
PATH_PROPAGATION_OVERFLOW = 2
PATH_PROPAGATION_NONFINITE = 3


def _nonempty_identity(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


class StateRegionPlan(StrictModule, NonTrainableState):
    """Composable Boolean state region with half-open primitive semantics."""

    lower: Array
    upper: Array
    children: tuple[StateRegionPlan, ...]
    predicate: Callable[[Array], Array] | None = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    region_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: str,
        lower: ArrayLike = (),
        upper: ArrayLike = (),
        children: tuple[StateRegionPlan, ...] = (),
        predicate: Callable[[Array], Array] | None = None,
        region_id: str,
    ):
        if kind not in ("half-open", "predicate", "and", "or", "xor", "not"):
            raise ValueError("Unknown state-region kind.")
        if any(not isinstance(child, StateRegionPlan) for child in children):
            raise TypeError("children must contain StateRegionPlan values.")
        expected_children = {"half-open": 0, "predicate": 0, "not": 1}.get(kind, 2)
        if len(children) != expected_children:
            raise ValueError(f"{kind!r} regions require {expected_children} children.")
        lower_ = jnp.asarray(lower)
        upper_ = jnp.asarray(upper)
        if kind == "half-open":
            lower_host, upper_host = np.asarray(lower_), np.asarray(upper_)
            if (
                lower_host.shape != upper_host.shape
                or lower_host.ndim == 0
                or np.iscomplexobj(lower_host)
                or np.iscomplexobj(upper_host)
                or not np.all(np.isfinite(lower_host))
                or not np.all(np.isfinite(upper_host))
                or not np.all(lower_host < upper_host)
            ):
                raise ValueError(
                    "Half-open region bounds must be finite, aligned, non-scalar, and ordered."
                )
            if predicate is not None:
                raise ValueError("Half-open regions do not accept a predicate.")
        elif lower_.size != 0 or upper_.size != 0:
            raise ValueError("Only half-open regions carry bounds.")
        if kind == "predicate":
            if not callable(predicate):
                raise TypeError("Predicate regions require a callable predicate.")
        elif predicate is not None:
            raise ValueError("Only predicate regions carry a predicate.")
        self.lower = lower_
        self.upper = upper_
        self.children = children
        self.predicate = predicate
        self.kind = kind
        self.region_id = _nonempty_identity(region_id, "region_id")

    @classmethod
    def half_open(
        cls,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        region_id: str | None = None,
    ) -> StateRegionPlan:
        lower_, upper_ = jnp.asarray(lower), jnp.asarray(upper)
        identity = region_id or canonical_fingerprint(
            {
                "kind": "half-open-state-region-v1",
                "lower": array_tree_fingerprint(lower_),
                "upper": array_tree_fingerprint(upper_),
            }
        )
        return cls(kind="half-open", lower=lower_, upper=upper_, region_id=identity)

    @classmethod
    def from_predicate(
        cls,
        predicate: Callable[[Array], Array],
        /,
        *,
        region_id: str,
    ) -> StateRegionPlan:
        return cls(kind="predicate", predicate=predicate, region_id=region_id)

    def contains(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if self.kind == "half-open":
            if value.shape[-self.lower.ndim :] != self.lower.shape:
                raise ValueError("State event shape does not match half-open bounds.")
            axes = tuple(range(value.ndim - self.lower.ndim, value.ndim))
            return jnp.all((value >= self.lower) & (value < self.upper), axis=axes)
        if self.kind == "predicate":
            result = jnp.asarray(self.predicate(value))
            if result.dtype != jnp.bool_:
                raise TypeError("Region predicates must return Boolean masks.")
            if result.ndim >= value.ndim:
                raise ValueError(
                    "Region predicate must reduce at least one event dimension."
                )
            return result
        if self.kind == "not":
            return ~self.children[0].contains(value)
        left = self.children[0].contains(value)
        right = self.children[1].contains(value)
        if self.kind == "and":
            return left & right
        if self.kind == "or":
            return left | right
        return left ^ right

    def _binary(self, other: StateRegionPlan, kind: str, /) -> StateRegionPlan:
        if not isinstance(other, StateRegionPlan):
            raise TypeError("Boolean region operands must be StateRegionPlan values.")
        return StateRegionPlan(
            kind=kind,
            children=(self, other),
            region_id=canonical_fingerprint(
                {
                    "kind": f"state-region-{kind}-v1",
                    "left": self.region_id,
                    "right": other.region_id,
                }
            ),
        )

    def __and__(self, other: StateRegionPlan) -> StateRegionPlan:
        return self._binary(other, "and")

    def __or__(self, other: StateRegionPlan) -> StateRegionPlan:
        return self._binary(other, "or")

    def __xor__(self, other: StateRegionPlan) -> StateRegionPlan:
        return self._binary(other, "xor")

    def __invert__(self) -> StateRegionPlan:
        return StateRegionPlan(
            kind="not",
            children=(self,),
            region_id=canonical_fingerprint(
                {"kind": "state-region-not-v1", "child": self.region_id}
            ),
        )


class PathBuffer(StrictModule, NonTrainableState):
    """Canonical fixed-capacity path with active mask and point provenance."""

    positions: Array
    times: Array
    length: Array
    mask: Array
    direction: Array
    lineage: Array

    def __init__(
        self,
        positions: ArrayLike,
        times: ArrayLike,
        length: ArrayLike,
        mask: ArrayLike,
        direction: ArrayLike,
        lineage: ArrayLike,
        /,
    ):
        positions_ = jnp.asarray(positions)
        times_ = jnp.asarray(times)
        if not jnp.issubdtype(times_.dtype, jnp.floating):
            raise TypeError("PathBuffer times must have a floating dtype.")
        length_ = jnp.asarray(length, dtype=jnp.int32)
        mask_ = jnp.asarray(mask, dtype=bool)
        direction_ = jnp.asarray(direction, dtype=jnp.int8)
        lineage_ = jnp.asarray(lineage, dtype=jnp.int32)
        if positions_.ndim < 2 or positions_.shape[0] <= 0:
            raise ValueError("positions must have shape (positive capacity, event...).")
        capacity = positions_.shape[0]
        if (
            times_.shape != (capacity,)
            or mask_.shape != (capacity,)
            or lineage_.shape != (capacity,)
            or length_.shape != ()
            or direction_.shape != ()
        ):
            raise ValueError(
                "PathBuffer times, mask, lineage, length, and direction shapes are inconsistent."
            )
        self.positions = positions_
        self.times = times_
        self.length = length_
        self.mask = mask_
        self.direction = direction_
        self.lineage = lineage_

    @classmethod
    def from_trajectory(
        cls,
        positions: ArrayLike,
        times: ArrayLike,
        /,
        *,
        capacity: int,
        direction: int = 1,
        lineage: ArrayLike | None = None,
    ) -> PathBuffer:
        values = jnp.asarray(positions)
        raw_times = jnp.asarray(times)
        time_values = (
            raw_times
            if jnp.issubdtype(raw_times.dtype, jnp.floating)
            else raw_times.astype(jnp.result_type(values.dtype, jnp.float32))
        )
        if values.ndim < 2:
            raise ValueError("positions must have shape (length, event...).")
        if time_values.ndim != 1 or time_values.shape[0] != values.shape[0]:
            raise ValueError("times must align with the trajectory length.")
        maximum = int(capacity)
        count = values.shape[0]
        if maximum <= 0 or count <= 0 or count > maximum:
            raise ValueError(
                "capacity must be positive and at least the trajectory length."
            )
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1.")
        positions_host, times_host = np.asarray(values), np.asarray(time_values)
        if not np.all(np.isfinite(positions_host)) or not np.all(np.isfinite(times_host)):
            raise ValueError("Initial trajectory values and times must be finite.")
        if count > 1 and not np.all(np.diff(times_host) > 0.0):
            raise ValueError("Trajectory times must be strictly increasing.")
        if lineage is None:
            ancestry = jnp.arange(count, dtype=jnp.int32)
        else:
            ancestry = jnp.asarray(lineage, dtype=jnp.int32)
            if ancestry.shape != (count,):
                raise ValueError("lineage must have one entry per active path point.")
            if np.any(np.asarray(ancestry) < 0):
                raise ValueError("lineage entries must be non-negative.")
        padding = ((0, maximum - count),) + ((0, 0),) * (values.ndim - 1)
        padded_values = jnp.pad(values, padding)
        padded_times = jnp.pad(time_values, ((0, maximum - count),))
        padded_lineage = jnp.pad(ancestry, ((0, maximum - count),), constant_values=-1)
        mask = jnp.arange(maximum, dtype=jnp.int32) < count
        return cls(
            padded_values,
            padded_times,
            jnp.asarray(count, dtype=jnp.int32),
            mask,
            jnp.asarray(direction, dtype=jnp.int8),
            padded_lineage,
        )

    @property
    def capacity(self) -> int:
        return self.positions.shape[0]

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.positions.shape[1:]

    def valid(self, /) -> Array:
        canonical_mask = jnp.arange(self.capacity, dtype=jnp.int32) < self.length
        event_mask = self.mask.reshape((self.capacity,) + (1,) * len(self.event_shape))
        finite_positions = jnp.all(
            jnp.where(event_mask, jnp.isfinite(self.positions), True)
        )
        finite_times = jnp.all(jnp.where(self.mask, jnp.isfinite(self.times), True))
        monotone = jnp.all(
            jnp.where(self.mask[1:], self.times[1:] > self.times[:-1], True)
        )
        canonical_padding = (
            jnp.all(jnp.where(event_mask, True, self.positions == 0))
            & jnp.all(jnp.where(self.mask, True, self.times == 0))
            & jnp.all(jnp.where(self.mask, self.lineage >= 0, self.lineage == -1))
        )
        return (
            (self.length > 0)
            & (self.length <= self.capacity)
            & jnp.all(self.mask == canonical_mask)
            & finite_positions
            & finite_times
            & monotone
            & canonical_padding
            & ((self.direction == 1) | (self.direction == -1))
        )

    def time_reversed(self, /) -> PathBuffer:
        index = jnp.arange(self.capacity, dtype=jnp.int32)
        source = jnp.clip(self.length - 1 - index, 0, self.capacity - 1)
        active = index < self.length
        event_mask = active.reshape((self.capacity,) + (1,) * len(self.event_shape))
        positions = jnp.where(
            event_mask, self.positions[source], jnp.zeros_like(self.positions)
        )
        times = jnp.where(active, self.times, jnp.zeros_like(self.times))
        lineage = jnp.where(active, self.lineage[source], -jnp.ones_like(self.lineage))
        return PathBuffer(positions, times, self.length, active, -self.direction, lineage)


def _fixed_step_time_grid_valid(path: PathBuffer, time_step: float, /) -> Array:
    interval = jnp.asarray(time_step, dtype=path.times.dtype)
    expected = jnp.arange(path.capacity, dtype=path.times.dtype) * interval
    return path.valid() & jnp.all(jnp.where(path.mask, path.times == expected, True))


def path_trajectory_id(path: PathBuffer, /) -> str:
    """Content-address every fixed-shape trajectory and point-lineage leaf."""

    if not isinstance(path, PathBuffer):
        raise TypeError("path must be PathBuffer.")
    return canonical_fingerprint(
        {
            "kind": "path-trajectory-v1",
            "positions": array_tree_fingerprint(path.positions),
            "times": array_tree_fingerprint(path.times),
            "length": int(path.length),
            "mask": array_tree_fingerprint(path.mask),
            "direction": int(path.direction),
            "lineage": array_tree_fingerprint(path.lineage),
        }
    )


class PathLineageLog(StrictModule, NonTrainableState):
    """Fixed-capacity accepted and rejected proposal ancestry."""

    parent: Array
    candidate: Array
    committed: Array
    accepted: Array
    mask: Array
    count: Array
    overflowed: Array

    def __init__(
        self,
        parent: ArrayLike,
        candidate: ArrayLike,
        committed: ArrayLike,
        accepted: ArrayLike,
        mask: ArrayLike,
        count: ArrayLike,
        overflowed: ArrayLike,
        /,
    ):
        parent_ = jnp.asarray(parent, dtype=jnp.uint32)
        candidate_ = jnp.asarray(candidate, dtype=jnp.uint32)
        committed_ = jnp.asarray(committed, dtype=jnp.uint32)
        accepted_ = jnp.asarray(accepted, dtype=bool)
        mask_ = jnp.asarray(mask, dtype=bool)
        count_ = jnp.asarray(count, dtype=jnp.int32)
        overflowed_ = jnp.asarray(overflowed, dtype=bool)
        if parent_.ndim != 1 or parent_.shape[0] <= 0:
            raise ValueError("Lineage arrays must have one positive capacity.")
        shape = parent_.shape
        if (
            candidate_.shape != shape
            or committed_.shape != shape
            or accepted_.shape != shape
            or mask_.shape != shape
            or count_.shape != ()
            or overflowed_.shape != ()
        ):
            raise ValueError("Lineage array and scalar shapes are inconsistent.")
        self.parent = parent_
        self.candidate = candidate_
        self.committed = committed_
        self.accepted = accepted_
        self.mask = mask_
        self.count = count_
        self.overflowed = overflowed_

    @classmethod
    def empty(cls, capacity: int, /) -> PathLineageLog:
        count = int(capacity)
        if count <= 0:
            raise ValueError("Lineage capacity must be positive.")
        integers = jnp.zeros((count,), dtype=jnp.uint32)
        booleans = jnp.zeros((count,), dtype=bool)
        return cls(
            integers,
            integers,
            integers,
            booleans,
            booleans,
            jnp.asarray(0, jnp.int32),
            jnp.asarray(False),
        )

    @property
    def capacity(self) -> int:
        return self.parent.shape[0]

    def valid(self, /) -> Array:
        canonical_mask = jnp.arange(self.capacity, dtype=jnp.int32) < self.count
        expected_committed = jnp.where(self.accepted, self.candidate, self.parent)
        inactive_zero = (
            jnp.all(jnp.where(self.mask, True, self.parent == 0))
            & jnp.all(jnp.where(self.mask, True, self.candidate == 0))
            & jnp.all(jnp.where(self.mask, True, self.committed == 0))
            & jnp.all(jnp.where(self.mask, True, ~self.accepted))
        )
        return (
            (self.count >= 0)
            & (self.count <= self.capacity)
            & jnp.all(self.mask == canonical_mask)
            & jnp.all(
                jnp.where(
                    self.mask,
                    self.committed == expected_committed,
                    True,
                )
            )
            & jnp.all(
                jnp.where(
                    self.mask,
                    self.candidate > self.parent,
                    True,
                )
            )
            & inactive_zero
        )

    def append(
        self,
        parent: ArrayLike,
        candidate: ArrayLike,
        accepted: ArrayLike,
        /,
    ) -> PathLineageLog:
        available = self.count < self.capacity
        index = jnp.minimum(self.count, self.capacity - 1)
        parent_ = jnp.asarray(parent, dtype=jnp.uint32)
        candidate_ = jnp.asarray(candidate, dtype=jnp.uint32)
        accepted_ = jnp.asarray(accepted, dtype=bool)
        committed_ = jnp.where(accepted_, candidate_, parent_)
        return PathLineageLog(
            self.parent.at[index].set(jnp.where(available, parent_, self.parent[index])),
            self.candidate.at[index].set(
                jnp.where(available, candidate_, self.candidate[index])
            ),
            self.committed.at[index].set(
                jnp.where(available, committed_, self.committed[index])
            ),
            self.accepted.at[index].set(
                jnp.where(available, accepted_, self.accepted[index])
            ),
            self.mask.at[index].set(jnp.where(available, True, self.mask[index])),
            self.count + available.astype(jnp.int32),
            self.overflowed | ~available,
        )


class DynamicsKernelCapabilities(StrictModule, NonTrainableState):
    """Declared path-generation and density capabilities of one dynamics kernel."""

    stochastic: bool = eqx.field(static=True)
    reversible: bool = eqx.field(static=True)
    supports_backward: bool = eqx.field(static=True)
    normalized_transition_density: bool = eqx.field(static=True)
    fixed_step: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        stochastic: bool,
        reversible: bool,
        supports_backward: bool,
        normalized_transition_density: bool,
        fixed_step: bool = True,
    ):
        values = (
            stochastic,
            reversible,
            supports_backward,
            normalized_transition_density,
            fixed_step,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("Dynamics-kernel capabilities must be booleans.")
        if stochastic and not normalized_transition_density:
            raise ValueError(
                "Stochastic path sampling requires a normalized transition density."
            )
        if reversible and not supports_backward:
            raise ValueError("Reversible kernels must support backward propagation.")
        self.stochastic = stochastic
        self.reversible = reversible
        self.supports_backward = supports_backward
        self.normalized_transition_density = normalized_transition_density
        self.fixed_step = fixed_step


class DynamicsStep(StrictModule):
    """One transition with normalized log-density and fail-closed evidence."""

    state: Array
    log_transition_density: Array
    valid: Array
    status: Array


class FunctionalDynamicsKernel(StrictModule):
    """Explicit functional path dynamics with no runtime capability discovery."""

    step_fn: Callable[[Key[Array, ""], Array, Array], DynamicsStep] = eqx.field(
        static=True
    )
    transition_log_density_fn: Callable[[Array, Array, Array], Array] = eqx.field(
        static=True
    )
    capabilities: DynamicsKernelCapabilities
    time_step: float = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        step: Callable[[Key[Array, ""], Array, Array], DynamicsStep],
        transition_log_density: Callable[[Array, Array, Array], Array],
        capabilities: DynamicsKernelCapabilities,
        /,
        *,
        time_step: float,
        kernel_id: str,
    ):
        if not callable(step) or not callable(transition_log_density):
            raise TypeError("step and transition_log_density must be callable.")
        if not isinstance(capabilities, DynamicsKernelCapabilities):
            raise TypeError("capabilities must be DynamicsKernelCapabilities.")
        interval = float(time_step)
        if not np.isfinite(interval) or interval <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        self.step_fn = step
        self.transition_log_density_fn = transition_log_density
        self.capabilities = capabilities
        self.time_step = interval
        self.kernel_id = _nonempty_identity(kernel_id, "kernel_id")

    def step(
        self, key: Key[Array, ""], state: Array, direction: Array, /
    ) -> DynamicsStep:
        result = self.step_fn(key, state, direction)
        if not isinstance(result, DynamicsStep):
            raise TypeError("A dynamics step must return DynamicsStep.")
        value = jnp.asarray(result.state)
        log_density = jnp.asarray(result.log_transition_density)
        reported_valid = jnp.asarray(result.valid)
        reported_status = jnp.asarray(result.status)
        if (
            value.shape != state.shape
            or value.dtype != state.dtype
            or log_density.shape != ()
            or reported_valid.shape != ()
            or reported_valid.dtype != jnp.bool_
            or reported_status.shape != ()
            or not jnp.issubdtype(reported_status.dtype, jnp.integer)
        ):
            raise ValueError(
                "DynamicsStep must preserve state shape/dtype and carry scalar density, validity, and status."
            )
        status_valid = (reported_status >= PATH_PROPAGATION_SUCCESS) & (
            reported_status <= PATH_PROPAGATION_NONFINITE
        )
        reported_status = jnp.where(
            status_valid,
            reported_status,
            PATH_PROPAGATION_KERNEL_FAILURE,
        ).astype(jnp.int32)
        declared_density = self.transition_log_density(state, value, direction)
        if declared_density.dtype != log_density.dtype:
            raise ValueError(
                "DynamicsStep and transition_log_density must use the same dtype."
            )
        density_tolerance = (
            32.0
            * jnp.finfo(log_density.dtype).eps
            * jnp.maximum(jnp.abs(declared_density), 1.0)
        )
        density_matches = jnp.abs(log_density - declared_density) <= density_tolerance
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.isfinite(log_density)
            & jnp.isfinite(declared_density)
        )
        valid = (
            reported_valid
            & finite
            & density_matches
            & status_valid
            & (reported_status == PATH_PROPAGATION_SUCCESS)
        )
        failed_status = jnp.where(
            reported_status == PATH_PROPAGATION_SUCCESS,
            PATH_PROPAGATION_KERNEL_FAILURE,
            reported_status,
        )
        status = jnp.where(
            valid,
            PATH_PROPAGATION_SUCCESS,
            jnp.where(finite, failed_status, PATH_PROPAGATION_NONFINITE),
        ).astype(jnp.int32)
        return DynamicsStep(value, declared_density, valid, status)

    def transition_log_density(
        self, source: Array, destination: Array, direction: Array, /
    ) -> Array:
        value = jnp.asarray(
            self.transition_log_density_fn(source, destination, direction)
        )
        if (
            value.shape != ()
            or jnp.iscomplexobj(value)
            or not jnp.issubdtype(value.dtype, jnp.floating)
        ):
            raise ValueError("Transition log density must be one real floating scalar.")
        return value


def select_path(
    current: PathBuffer, proposed: PathBuffer, choose: ArrayLike, /
) -> PathBuffer:
    """Select every dynamic path leaf with one scalar decision."""

    current_leaves = jax.tree_util.tree_leaves(current)
    proposed_leaves = jax.tree_util.tree_leaves(proposed)
    if len(current_leaves) != len(proposed_leaves) or any(
        old.shape != new.shape or old.dtype != new.dtype
        for old, new in zip(current_leaves, proposed_leaves, strict=True)
    ):
        raise ValueError("Selected paths must have identical dynamic shapes and dtypes.")
    condition = jnp.asarray(choose)
    if condition.shape != () or condition.dtype != jnp.bool_:
        raise TypeError("Path selection requires one Boolean scalar.")
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(condition, new, old),
        proposed,
        current,
    )


__all__ = [
    "DynamicsKernelCapabilities",
    "DynamicsStep",
    "FunctionalDynamicsKernel",
    "PATH_PROPAGATION_KERNEL_FAILURE",
    "PATH_PROPAGATION_NONFINITE",
    "PATH_PROPAGATION_OVERFLOW",
    "PATH_PROPAGATION_SUCCESS",
    "PathBuffer",
    "path_trajectory_id",
    "PathLineageLog",
    "StateRegionPlan",
    "select_path",
]
