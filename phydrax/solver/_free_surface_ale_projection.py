#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._mac_ale import MACALEStageGeometry


if TYPE_CHECKING:
    from ..applications.hydrodynamics._free_surface_ale import (
        PreparedGraphSurfaceALE,
    )

FaceTuple = tuple[Array, ...]


def _tuple_add(left: FaceTuple, scale: Array, right: FaceTuple, /) -> FaceTuple:
    return tuple(a + scale * b for a, b in zip(left, right, strict=True))


class FreeSurfaceProjectionResult(StrictModule):
    momentum: FaceTuple
    velocity: FaceTuple
    pressure_head: Array
    pressure_increment: Array
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    pressure_residual_norm: Array
    hodge_residual_norm: Array
    iterations: Array
    converged: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MappedFreeSurfaceProjectionPlan(StrictModule, NonTrainableState):
    """Mixed-boundary pressure projection on one graph ALE geometry stage."""

    surface: PreparedGraphSurfaceALE
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: PreparedGraphSurfaceALE,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 200,
    ):
        from ..applications.hydrodynamics._free_surface_ale import (
            PreparedGraphSurfaceALE,
        )

        if not isinstance(surface, PreparedGraphSurfaceALE):
            raise TypeError("surface must be PreparedGraphSurfaceALE.")

        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if tolerance_ <= 0.0 or iterations <= 0:
            raise ValueError("Invalid free-surface projection tolerance or iterations.")
        self.surface = surface
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-free-surface-projection",
                "surface": surface.surface_id,
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
                "top_pressure": "dirichlet",
                "other_walls": "normal-neumann",
            }
        )

    def free_velocity_mask(self, geometry: MACALEStageGeometry, /) -> FaceTuple:
        masks = [jnp.ones_like(value) for value in geometry.face_measures]
        axes = self.surface.plan.reference.grid.structured_axes
        for axis, grid_axis in enumerate(axes):
            if grid_axis.periodic:
                continue
            lower = [slice(None)] * masks[axis].ndim
            upper = [slice(None)] * masks[axis].ndim
            lower[axis] = 0
            upper[axis] = masks[axis].shape[axis] - 1
            masks[axis] = masks[axis].at[tuple(lower)].set(0.0)
            masks[axis] = masks[axis].at[tuple(upper)].set(0.0)
        top = [slice(None)] * masks[2].ndim
        top[2] = masks[2].shape[2] - 1
        masks[2] = masks[2].at[tuple(top)].set(1.0)
        return tuple(masks)

    def surface_pressure_force(
        self,
        geometry: MACALEStageGeometry,
        pressure_head: ArrayLike,
        /,
    ) -> FaceTuple:
        head = jnp.asarray(pressure_head, dtype=geometry.cell_volumes.dtype)
        expected = self.surface.eta_shape
        if head.shape != expected:
            raise ValueError(f"Surface pressure head must have shape {expected}.")
        output = [jnp.zeros_like(value) for value in geometry.face_measures]
        location = [slice(None)] * output[2].ndim
        location[2] = output[2].shape[2] - 1
        top_area = jnp.take(geometry.face_measures[2], -1, axis=2)
        output[2] = output[2].at[tuple(location)].set(-top_area * head)
        return tuple(output)

    def _gradient_covector(
        self,
        geometry: MACALEStageGeometry,
        pressure: Array,
        mask: FaceTuple,
        /,
    ) -> FaceTuple:
        zero = tuple(jnp.zeros_like(value) for value in geometry.face_measures)

        def divergence(values):
            masked = tuple(
                value * active for value, active in zip(values, mask, strict=True)
            )
            return geometry.divergence(masked)

        cotangent = geometry.cell_volumes * pressure
        gradient = jax.linear_transpose(divergence, zero)(cotangent)[0]
        return tuple(value * active for value, active in zip(gradient, mask, strict=True))

    def project(
        self,
        geometry: MACALEStageGeometry,
        tentative_momentum: FaceTuple,
        step_size: ArrayLike,
        pressure_guess: ArrayLike | None = None,
        /,
    ) -> FreeSurfaceProjectionResult:
        if not isinstance(geometry, MACALEStageGeometry):
            raise TypeError("geometry must be MACALEStageGeometry.")
        dt = jnp.asarray(step_size, dtype=geometry.cell_volumes.dtype).reshape(())
        mask = self.free_velocity_mask(geometry)
        tentative = self.surface.inverse_hodge(
            geometry, tentative_momentum, free_mask=mask
        )
        divergence_before = geometry.divergence(tentative.velocity)
        rhs = divergence_before / dt
        pressure = (
            jnp.zeros_like(rhs)
            if pressure_guess is None
            else jnp.asarray(pressure_guess, dtype=rhs.dtype)
        )
        if pressure.shape != rhs.shape:
            raise ValueError("Pressure-head guess shape is invalid.")

        def action(value):
            gradient = self._gradient_covector(geometry, value, mask)
            inverse = self.surface.inverse_hodge(geometry, gradient, free_mask=mask)
            return geometry.divergence(inverse.velocity)

        residual = rhs - action(pressure)
        direction = residual
        norm = jnp.sum(geometry.cell_volumes * residual**2)
        threshold = self.tolerance**2 * jnp.maximum(norm, 1.0)
        active = norm > threshold
        failed = jnp.asarray(False)

        def body(_, state):
            value, residual_, direction_, norm_, active_, failed_ = state
            image = action(direction_)
            denominator = jnp.sum(geometry.cell_volumes * direction_ * image)
            valid = active_ & jnp.isfinite(denominator) & (denominator > 0.0)
            alpha = jnp.where(valid, norm_ / denominator, 0.0)
            next_value = value + alpha * direction_
            next_residual = residual_ - alpha * image
            next_norm = jnp.sum(geometry.cell_volumes * next_residual**2)
            running = valid & (next_norm > threshold)
            beta = jnp.where(running & (norm_ > 0.0), next_norm / norm_, 0.0)
            return (
                next_value,
                next_residual,
                next_residual + beta * direction_,
                next_norm,
                running,
                failed_ | (active_ & ~valid),
            )

        pressure, residual, _, norm, active, failed = jax.lax.fori_loop(
            0,
            self.maximum_iterations,
            body,
            (pressure, residual, direction, norm, active, failed),
        )
        gradient = self._gradient_covector(geometry, pressure, mask)
        corrected_momentum = _tuple_add(tentative_momentum, -dt, gradient)
        corrected = self.surface.inverse_hodge(
            geometry, corrected_momentum, free_mask=mask
        )
        divergence_after = geometry.divergence(corrected.velocity)
        pressure_residual = action(pressure) - rhs
        residual_norm = jnp.sqrt(jnp.sum(geometry.cell_volumes * pressure_residual**2))
        divergence_norm = jnp.sqrt(jnp.sum(geometry.cell_volumes * divergence_after**2))
        rhs_norm = jnp.sqrt(jnp.sum(geometry.cell_volumes * rhs**2))
        finite = (
            tentative.finite
            & corrected.finite
            & jnp.all(jnp.isfinite(pressure))
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(divergence_norm)
        )
        converged = (
            ~active
            & ~failed
            & finite
            & (divergence_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0))
        )
        return FreeSurfaceProjectionResult(
            momentum=corrected_momentum,
            velocity=corrected.velocity,
            pressure_head=pressure,
            pressure_increment=dt * pressure,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            pressure_residual=pressure_residual,
            pressure_residual_norm=residual_norm,
            hodge_residual_norm=jnp.maximum(
                tentative.residual_norm, corrected.residual_norm
            ),
            iterations=jnp.asarray(self.maximum_iterations, dtype=jnp.int32),
            converged=converged,
            finite=finite,
            successful=converged,
            plan_id=self.plan_id,
        )


__all__ = ["FreeSurfaceProjectionResult", "MappedFreeSurfaceProjectionPlan"]
