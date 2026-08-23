#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._high_resolution import (
    CharacteristicReconstructionPlan,
    CharacteristicSystem,
    HighResolutionMethod,
    HighResolutionReconstructionPlan,
    NonuniformWENOReconstructionPlan,
)
from ._weno import WENOOrder, WENOReconstructionPlan


DifferentiabilityClass: TypeAlias = Literal[
    "smooth_discrete",
    "almost_everywhere",
    "frozen_decision",
    "smooth_surrogate",
    "unsupported",
]


def _move_front(value: ArrayLike, axis: int, /) -> Array:
    array = jnp.asarray(value)
    if not 0 <= int(axis) < array.ndim - 1:
        raise ValueError("Reconstruction axis must be a spatial state axis.")
    return jnp.moveaxis(array, int(axis), 0)


def _restore_axis(value: Array, axis: int, /) -> Array:
    return jnp.moveaxis(value, 0, int(axis))


def _boundary_layer(value: ArrayLike, interior: Array, /) -> Array:
    layer = jnp.asarray(value)
    expected = interior.shape[1:]
    if layer.shape == () or layer.shape == (interior.shape[-1],):
        return jnp.broadcast_to(layer, expected)
    if layer.shape != expected:
        raise ValueError(f"Boundary layer must have shape {expected}.")
    return layer


class AbstractFaceReconstructionPlan(StrictModule, NonTrainableState):
    """Cell-average to directional left/right face traces."""

    formal_order: int = eqx.field(static=True)
    ghost_width: int = eqx.field(static=True)
    differentiability: DifferentiabilityClass = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def reconstruct_axis(
        self,
        state: ArrayLike,
        axis: int,
        /,
        *,
        periodic: bool,
        lower_exterior: ArrayLike | None = None,
        upper_exterior: ArrayLike | None = None,
        cell_widths: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        raise NotImplementedError


class PiecewiseConstantReconstruction(AbstractFaceReconstructionPlan):
    """First-order Godunov traces from cell averages."""

    def __init__(self):
        self.formal_order = 1
        self.ghost_width = 1
        self.differentiability = "smooth_discrete"
        self.plan_id = canonical_fingerprint({"kind": "piecewise-constant-fv"})

    def reconstruct_axis(
        self,
        state: ArrayLike,
        axis: int,
        /,
        *,
        periodic: bool,
        lower_exterior: ArrayLike | None = None,
        upper_exterior: ArrayLike | None = None,
        cell_widths: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        del cell_widths
        values = _move_front(state, axis)
        if periodic:
            return _restore_axis(jnp.roll(values, 1, axis=0), axis), jnp.asarray(state)
        if lower_exterior is None or upper_exterior is None:
            raise ValueError("Bounded reconstruction requires both exterior states.")
        lower = _boundary_layer(lower_exterior, values)[None, ...]
        upper = _boundary_layer(upper_exterior, values)[None, ...]
        left = jnp.concatenate((lower, values), axis=0)
        right = jnp.concatenate((values, upper), axis=0)
        return _restore_axis(left, axis), _restore_axis(right, axis)


class AbstractSlopeLimiter(StrictModule, NonTrainableState):
    """Two-slope nonlinear limiter."""

    limiter_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def limit(self, backward: Array, forward: Array, /) -> Array:
        raise NotImplementedError


def _same_sign_minimum(left: Array, right: Array, /) -> Array:
    same = left * right > 0.0
    return jnp.where(
        same,
        jnp.sign(left) * jnp.minimum(jnp.abs(left), jnp.abs(right)),
        0.0,
    )


class UnlimitedLimiter(AbstractSlopeLimiter):
    """Centered smooth-solution slope without nonlinear limiting."""

    def __init__(self):
        self.limiter_id = canonical_fingerprint({"kind": "unlimited-centered"})

    def limit(self, backward: Array, forward: Array, /) -> Array:
        return 0.5 * (backward + forward)


class MinmodLimiter(AbstractSlopeLimiter):
    def __init__(self):
        self.limiter_id = canonical_fingerprint({"kind": "minmod"})

    def limit(self, backward: Array, forward: Array, /) -> Array:
        return _same_sign_minimum(backward, forward)


class MCLimiter(AbstractSlopeLimiter):
    def __init__(self):
        self.limiter_id = canonical_fingerprint({"kind": "monotonized-central"})

    def limit(self, backward: Array, forward: Array, /) -> Array:
        centered = 0.5 * (backward + forward)
        return _same_sign_minimum(
            centered,
            _same_sign_minimum(2.0 * backward, 2.0 * forward),
        )


class VanLeerLimiter(AbstractSlopeLimiter):
    def __init__(self):
        self.limiter_id = canonical_fingerprint({"kind": "van-leer"})

    def limit(self, backward: Array, forward: Array, /) -> Array:
        denominator = backward + forward
        harmonic = (
            2.0 * backward * forward / jnp.where(denominator == 0.0, 1.0, denominator)
        )
        return jnp.where(backward * forward > 0.0, harmonic, 0.0)


class SuperbeeLimiter(AbstractSlopeLimiter):
    def __init__(self):
        self.limiter_id = canonical_fingerprint({"kind": "superbee"})

    def limit(self, backward: Array, forward: Array, /) -> Array:
        first = _same_sign_minimum(2.0 * backward, forward)
        second = _same_sign_minimum(backward, 2.0 * forward)
        return jnp.where(jnp.abs(first) >= jnp.abs(second), first, second)


class MUSCLReconstruction(AbstractFaceReconstructionPlan):
    """Distance-aware piecewise-linear reconstruction with a TVD limiter."""

    limiter: AbstractSlopeLimiter

    def __init__(self, limiter: AbstractSlopeLimiter | None = None, /):
        limiter_ = MCLimiter() if limiter is None else limiter
        if not isinstance(limiter_, AbstractSlopeLimiter):
            raise TypeError("MUSCL limiter must be an AbstractSlopeLimiter.")
        self.limiter = limiter_
        self.formal_order = 2
        self.ghost_width = 2
        self.differentiability = "frozen_decision"
        self.plan_id = canonical_fingerprint(
            {"kind": "muscl-fv", "limiter": limiter_.limiter_id}
        )

    def reconstruct_axis(
        self,
        state: ArrayLike,
        axis: int,
        /,
        *,
        periodic: bool,
        lower_exterior: ArrayLike | None = None,
        upper_exterior: ArrayLike | None = None,
        cell_widths: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        values = _move_front(state, axis)
        count = int(values.shape[0])
        widths = (
            jnp.ones((count,), dtype=values.dtype)
            if cell_widths is None
            else jnp.asarray(cell_widths, dtype=values.dtype).reshape((count,))
        )
        width_shape = (count,) + (1,) * (values.ndim - 1)
        widths_broadcast = widths.reshape(width_shape)
        if periodic:
            previous = jnp.roll(values, 1, axis=0)
            following = jnp.roll(values, -1, axis=0)
            backward_distance = 0.5 * (widths + jnp.roll(widths, 1))
            forward_distance = 0.5 * (widths + jnp.roll(widths, -1))
        else:
            if lower_exterior is None or upper_exterior is None:
                raise ValueError("Bounded MUSCL reconstruction requires exterior states.")
            lower = _boundary_layer(lower_exterior, values)[None, ...]
            upper = _boundary_layer(upper_exterior, values)[None, ...]
            previous = jnp.concatenate((lower, values[:-1]), axis=0)
            following = jnp.concatenate((values[1:], upper), axis=0)
            backward_distance = jnp.concatenate(
                (widths[:1], 0.5 * (widths[1:] + widths[:-1]))
            )
            forward_distance = jnp.concatenate(
                (0.5 * (widths[:-1] + widths[1:]), widths[-1:])
            )
        distance_shape = (count,) + (1,) * (values.ndim - 1)
        backward = (values - previous) / backward_distance.reshape(distance_shape)
        forward = (following - values) / forward_distance.reshape(distance_shape)
        slope = self.limiter.limit(backward, forward)
        lower_trace = values - 0.5 * widths_broadcast * slope
        upper_trace = values + 0.5 * widths_broadcast * slope
        if periodic:
            left = jnp.roll(upper_trace, 1, axis=0)
            right = lower_trace
        else:
            left = jnp.concatenate((lower, upper_trace), axis=0)
            right = jnp.concatenate((lower_trace, upper), axis=0)
        return _restore_axis(left, axis), _restore_axis(right, axis)


def reconstruct_ghosted_axis(
    reconstruction: Any,
    ghosted_state: ArrayLike,
    axis: int,
    /,
    *,
    interior_cell_count: int,
    ghost_depth: int,
    periodic: bool,
    axis_coordinates: ArrayLike,
) -> tuple[Array, Array]:
    """Reconstruct canonical faces exclusively from prepared ghosted cells."""
    values = _move_front(ghosted_state, axis)
    coordinates = jnp.asarray(axis_coordinates)
    if coordinates.shape != (values.shape[0],):
        raise ValueError("Ghosted axis coordinates must match ghosted cell count.")
    if isinstance(reconstruction, AbstractFaceReconstructionPlan):
        differences = coordinates[1:] - coordinates[:-1]
        widths = jnp.concatenate(
            (
                differences[:1],
                0.5 * (differences[:-1] + differences[1:]),
                differences[-1:],
            )
        )
        left, right = reconstruction.reconstruct_axis(
            _restore_axis(values, axis),
            axis,
            periodic=False,
            lower_exterior=values[0],
            upper_exterior=values[-1],
            cell_widths=widths,
        )
        left_front = _move_front(left, axis)
        right_front = _move_front(right, axis)
        stop = ghost_depth + interior_cell_count + (0 if periodic else 1)
        return (
            _restore_axis(left_front[ghost_depth:stop], axis),
            _restore_axis(right_front[ghost_depth:stop], axis),
        )
    reconstructed = reconstruction.reconstruct(values)
    if isinstance(reconstruction, CharacteristicReconstructionPlan):
        left_right, right_right, _ = reconstructed
    else:
        left_right, right_right = reconstructed
    start = ghost_depth - 1
    stop = start + interior_cell_count + (0 if periodic else 1)
    return (
        _restore_axis(left_right[start:stop], axis),
        _restore_axis(right_right[start:stop], axis),
    )


__all__ = [
    "AbstractFaceReconstructionPlan",
    "AbstractSlopeLimiter",
    "CharacteristicReconstructionPlan",
    "CharacteristicSystem",
    "DifferentiabilityClass",
    "HighResolutionMethod",
    "HighResolutionReconstructionPlan",
    "MCLimiter",
    "MUSCLReconstruction",
    "MinmodLimiter",
    "NonuniformWENOReconstructionPlan",
    "PiecewiseConstantReconstruction",
    "reconstruct_ghosted_axis",
    "SuperbeeLimiter",
    "UnlimitedLimiter",
    "VanLeerLimiter",
    "WENOOrder",
    "WENOReconstructionPlan",
]
