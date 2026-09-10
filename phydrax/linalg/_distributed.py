#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Ownership-aware pairings, operators, and distributed Krylov execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .backends._native_krylov import _pcg_raw


class DistributedCoordinateLayout(StrictModule, NonTrainableState):
    """Exact canonical ownership ranges and fixed local capacity."""

    global_size: int = eqx.field(static=True)
    owned_ranges: tuple[tuple[int, int], ...] = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        global_size: int,
        owned_ranges: Sequence[tuple[int, int]],
        /,
        *,
        local_capacity: int | None = None,
    ) -> None:
        size = int(global_size)
        ranges = tuple((int(start), int(stop)) for start, stop in owned_ranges)
        if size <= 0 or not ranges:
            raise ValueError("distributed coordinates require positive global size")
        cursor = 0
        for start, stop in ranges:
            if start != cursor or stop <= start or stop > size:
                raise ValueError(
                    "owned_ranges must form one ordered exact partition of [0, global_size)"
                )
            cursor = stop
        if cursor != size:
            raise ValueError("owned_ranges do not cover global_size")
        capacity = (
            max(stop - start for start, stop in ranges)
            if local_capacity is None
            else int(local_capacity)
        )
        if capacity < max(stop - start for start, stop in ranges):
            raise ValueError("local_capacity is smaller than an owned range")
        self.global_size = size
        self.owned_ranges = ranges
        self.local_capacity = capacity
        self.layout_id = canonical_fingerprint(
            {
                "kind": "distributed-coordinate-layout",
                "global_size": size,
                "owned_ranges": [list(value) for value in ranges],
                "local_capacity": capacity,
            }
        )

    @property
    def partition_count(self) -> int:
        return len(self.owned_ranges)

    def owned_mask(self, partition_index: ArrayLike, /) -> Array:
        part = jnp.asarray(partition_index, dtype=jnp.int32)
        starts = jnp.asarray(tuple(start for start, _ in self.owned_ranges))
        stops = jnp.asarray(tuple(stop for _, stop in self.owned_ranges))
        count = stops[part] - starts[part]
        return jnp.arange(self.local_capacity) < count

    def pack_global(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if array.shape[0] != self.global_size:
            raise ValueError("global vector leading dimension does not match layout")
        result = jnp.zeros(
            (self.partition_count, self.local_capacity, *array.shape[1:]),
            dtype=array.dtype,
        )
        for part, (start, stop) in enumerate(self.owned_ranges):
            result = result.at[part, : stop - start].set(array[start:stop])
        return result

    def unpack_global(self, local_values: ArrayLike, /) -> Array:
        values = jnp.asarray(local_values)
        expected = (self.partition_count, self.local_capacity)
        if values.shape[:2] != expected:
            raise ValueError(f"local values must begin with shape {expected}")
        result = jnp.zeros(
            (self.global_size, *values.shape[2:]),
            dtype=values.dtype,
        )
        for part, (start, stop) in enumerate(self.owned_ranges):
            result = result.at[start:stop].set(values[part, : stop - start])
        return result


class DistributedPairing(StrictModule, NonTrainableState):
    """Owned-only weighted pairing with optional local-shard collective reduction."""

    owned_mask: Array
    weights: Array | None
    axis_name: str | None = eqx.field(static=True)
    pairing_id: str = eqx.field(static=True)

    def __init__(
        self,
        owned_mask: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        axis_name: str | None = None,
        pairing_id: str | None = None,
    ) -> None:
        mask = jnp.asarray(owned_mask, dtype=bool)
        if mask.ndim < 1:
            raise ValueError("owned_mask must have at least one dimension")
        weights_ = None if weights is None else jnp.asarray(weights)
        if weights_ is not None and weights_.shape != mask.shape:
            raise ValueError("pairing weights must match owned_mask")
        axis = None if axis_name is None else str(axis_name).strip()
        if axis_name is not None and not axis:
            raise ValueError("axis_name must be non-empty")
        if pairing_id is None and (
            not mask.is_fully_addressable
            or (weights_ is not None and not weights_.is_fully_addressable)
        ):
            raise ValueError(
                "Non-addressable distributed pairings require an explicit pairing_id."
            )
        self.owned_mask = mask
        self.weights = weights_
        self.axis_name = axis
        self.pairing_id = (
            canonical_fingerprint(
                {
                    "kind": "distributed-pairing",
                    "mask": array_tree_fingerprint(mask),
                    "weights": None
                    if weights_ is None
                    else array_tree_fingerprint(weights_),
                    "axis_name": axis,
                }
            )
            if pairing_id is None
            else str(pairing_id).strip()
        )
        if not self.pairing_id:
            raise ValueError("pairing_id must be non-empty")

    def _mask_for(self, value: Array) -> Array:
        if value.shape[: self.owned_mask.ndim] != self.owned_mask.shape:
            raise ValueError("pairing value shape does not match owned_mask")
        return self.owned_mask.reshape(
            (*self.owned_mask.shape, *(1 for _ in value.shape[self.owned_mask.ndim :])),
        )

    def inner(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if left_.shape != right_.shape:
            raise ValueError("distributed pairing arguments must have matching shapes")
        mask = self._mask_for(left_)
        weighted_right = right_
        if self.weights is not None:
            weights = self.weights.reshape(
                (*self.weights.shape, *(1 for _ in left_.shape[self.weights.ndim :])),
            )
            weighted_right = weights * weighted_right
        local = jnp.vdot(jnp.where(mask, left_, 0), jnp.where(mask, weighted_right, 0))
        if self.axis_name is None:
            return local
        return jax.lax.psum(local, self.axis_name)

    def global_all(self, value: ArrayLike, /) -> Array:
        local = jnp.all(jnp.asarray(value, dtype=bool))
        if self.axis_name is None:
            return local
        return jax.lax.pmin(local.astype(jnp.int32), self.axis_name).astype(bool)


class DistributedLinearOperator(StrictModule, NonTrainableState):
    """Local or global matrix-free action with explicit transpose identity."""

    action: Callable[[Array], Array]
    transpose_action: Callable[[Array], Array]
    source_shape: tuple[int, ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: Callable[[Array], Array],
        transpose_action: Callable[[Array], Array],
        source_shape: Sequence[int],
        target_shape: Sequence[int],
        /,
        *,
        operator_id: str,
    ) -> None:
        if not callable(action) or not callable(transpose_action):
            raise TypeError("distributed operator actions must be callable")
        source = tuple(int(size) for size in source_shape)
        target = tuple(int(size) for size in target_shape)
        if not source or not target or any(size <= 0 for size in (*source, *target)):
            raise ValueError("distributed operator shapes must be positive")
        identifier = str(operator_id).strip()
        if not identifier:
            raise ValueError("operator_id must be non-empty")
        self.action = action
        self.transpose_action = transpose_action
        self.source_shape = source
        self.target_shape = target
        self.operator_id = identifier

    def mv(self, value: ArrayLike, /) -> Array:
        vector = jnp.asarray(value)
        if vector.shape != self.source_shape:
            raise ValueError("distributed operator source shape mismatch")
        result = jnp.asarray(self.action(vector))
        if result.shape != self.target_shape:
            raise ValueError("distributed operator action changed target shape")
        return result

    def transpose_mv(self, value: ArrayLike, /) -> Array:
        vector = jnp.asarray(value)
        if vector.shape != self.target_shape:
            raise ValueError("distributed transpose source shape mismatch")
        result = jnp.asarray(self.transpose_action(vector))
        if result.shape != self.source_shape:
            raise ValueError("distributed transpose action changed target shape")
        return result


class DistributedKrylovPolicy(StrictModule, NonTrainableState):
    maximum_steps: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_steps: int,
        /,
        *,
        relative_tolerance: float = 1.0e-8,
        absolute_tolerance: float = 0.0,
    ) -> None:
        steps = int(maximum_steps)
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        if steps <= 0:
            raise ValueError("maximum_steps must be positive")
        if not np.isfinite(relative) or relative < 0:
            raise ValueError("relative_tolerance must be finite and non-negative")
        if not np.isfinite(absolute) or absolute < 0:
            raise ValueError("absolute_tolerance must be finite and non-negative")
        self.maximum_steps = steps
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.policy_id = canonical_fingerprint(
            {
                "kind": "distributed-krylov-policy",
                "maximum_steps": steps,
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
            }
        )


class DistributedKrylovResult(StrictModule):
    value: Array
    iterations: Array
    residual_norm: Array
    converged: Array
    breakdown: Array


def solve_distributed_pcg(
    operator: DistributedLinearOperator,
    right_hand_side: ArrayLike,
    pairing: DistributedPairing,
    policy: DistributedKrylovPolicy,
    /,
    *,
    initial: ArrayLike | None = None,
    preconditioner: Callable[[Array, Array], Array] | None = None,
) -> DistributedKrylovResult:
    """Run pairing-aware PCG on global arrays or inside a local shard map."""

    if operator.source_shape != operator.target_shape:
        raise ValueError("PCG requires a square distributed operator")
    rhs = jnp.asarray(right_hand_side)
    if rhs.shape != operator.source_shape:
        raise ValueError("right_hand_side shape does not match the operator")
    initial_ = jnp.zeros_like(rhs) if initial is None else jnp.asarray(initial)
    if initial_.shape != rhs.shape:
        raise ValueError("initial shape does not match right_hand_side")
    precondition = (
        (lambda residual, _: residual) if preconditioner is None else preconditioner
    )
    value, auxiliary, _ = _pcg_raw(
        operator.mv,
        rhs,
        initial_,
        pairing.inner,
        precondition,
        policy.maximum_steps,
        jnp.asarray(policy.relative_tolerance, dtype=rhs.real.dtype),
        jnp.asarray(policy.absolute_tolerance, dtype=rhs.real.dtype),
        finite_all=pairing.global_all,
    )
    iterations, residual_norm, _, _, breakdown = auxiliary
    rhs_norm = jnp.sqrt(jnp.maximum(jnp.real(pairing.inner(rhs, rhs)), 0.0))
    threshold = policy.absolute_tolerance + policy.relative_tolerance * rhs_norm
    converged = pairing.global_all(residual_norm <= threshold)
    return DistributedKrylovResult(
        value,
        iterations,
        residual_norm,
        converged,
        breakdown,
    )


__all__ = (
    "DistributedCoordinateLayout",
    "DistributedKrylovPolicy",
    "DistributedKrylovResult",
    "DistributedLinearOperator",
    "DistributedPairing",
    "solve_distributed_pcg",
)
