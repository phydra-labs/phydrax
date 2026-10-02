#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver._mac_ale import MACALEStageGeometry
from ...typing import checked
from ._boundary import FreeSurfaceBoundaryStage
from ._free_surface_ale import MappedHodgeSolveResult, PreparedGraphSurfaceALE


FaceTuple = tuple[Array, ...]
_CGCarry: TypeAlias = tuple[Array, Array, Array, Array, Array, Array, Array]


def _tuple_add(left: FaceTuple, scale: Array | float, right: FaceTuple, /) -> FaceTuple:
    return tuple(a + scale * b for a, b in zip(left, right, strict=True))


@final
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
    tentative_hodge: MappedHodgeSolveResult
    corrected_hodge: MappedHodgeSolveResult
    hodge_status: Array
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

    @checked
    def __init__(
        self,
        surface: PreparedGraphSurfaceALE,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_iterations: int = 200,
    ) -> None:

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
        location: list[slice | int] = [slice(None)] * output[2].ndim
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

        def divergence(values: FaceTuple) -> Array:
            return geometry.divergence(
                tuple(value * active for value, active in zip(values, mask, strict=True))
            )

        cotangent = geometry.cell_volumes * pressure
        gradient = jax.linear_transpose(divergence, zero)(cotangent)[0]
        return tuple(value * active for value, active in zip(gradient, mask, strict=True))

    @checked
    def project(
        self,
        geometry: MACALEStageGeometry,
        tentative_momentum: FaceTuple,
        boundary_stage: FreeSurfaceBoundaryStage,
        step_size: ArrayLike,
        pressure_guess: ArrayLike | None = None,
        /,
    ) -> FreeSurfaceProjectionResult:
        if boundary_stage.layout_id == "":
            raise ValueError("Free-surface boundary stage has no layout identity.")
        dt = jnp.asarray(step_size, dtype=geometry.cell_volumes.dtype).reshape(())
        mask = boundary_stage.free_velocity_mask
        lifting_momentum = self.surface.apply_hodge(
            geometry, boundary_stage.prescribed_velocity
        )
        homogeneous_momentum = _tuple_add(tentative_momentum, -1.0, lifting_momentum)
        tentative = self.surface.inverse_hodge(
            geometry, homogeneous_momentum, free_mask=mask
        )
        tentative_velocity = tuple(
            free + prescribed
            for free, prescribed in zip(
                tentative.velocity,
                boundary_stage.prescribed_velocity,
                strict=True,
            )
        )
        divergence_before = geometry.divergence(tentative_velocity)
        rhs = divergence_before / dt
        pressure = (
            jnp.zeros_like(rhs)
            if pressure_guess is None
            else jnp.asarray(pressure_guess, dtype=rhs.dtype)
        )
        if pressure.shape != rhs.shape:
            raise ValueError("Pressure-head guess shape is invalid.")

        def action(value: Array) -> tuple[Array, MappedHodgeSolveResult]:
            gradient = self._gradient_covector(geometry, value, mask)
            inverse = self.surface.inverse_hodge(geometry, gradient, free_mask=mask)
            return geometry.divergence(inverse.velocity), inverse

        initial_image, initial_hodge = action(pressure)
        residual = rhs - initial_image
        hodge_status = jnp.where(
            tentative.successful, initial_hodge.status, tentative.status
        )
        direction = residual
        norm = jnp.sum(geometry.cell_volumes * residual**2)
        threshold = self.tolerance**2 * jnp.maximum(norm, 1.0)
        active = norm > threshold
        failed = jnp.asarray(False)

        def body(_: Array, state: _CGCarry) -> _CGCarry:
            value, residual_, direction_, norm_, active_, failed_, status_ = state
            image, inverse = action(direction_)
            denominator = jnp.sum(geometry.cell_volumes * direction_ * image)
            valid = (
                active_
                & inverse.successful
                & jnp.isfinite(denominator)
                & (denominator > 0.0)
            )
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
                jnp.where(status_ == 0, inverse.status, status_),
            )

        pressure, residual, _, norm, active, failed, hodge_status = jax.lax.fori_loop(
            0,
            self.maximum_iterations,
            body,
            (pressure, residual, direction, norm, active, failed, hodge_status),
        )
        gradient = self._gradient_covector(geometry, pressure, mask)
        corrected_homogeneous = _tuple_add(homogeneous_momentum, -dt, gradient)
        corrected_free = self.surface.inverse_hodge(
            geometry, corrected_homogeneous, free_mask=mask
        )
        corrected_velocity = tuple(
            free + prescribed
            for free, prescribed in zip(
                corrected_free.velocity,
                boundary_stage.prescribed_velocity,
                strict=True,
            )
        )
        corrected_momentum = _tuple_add(corrected_homogeneous, 1.0, lifting_momentum)
        divergence_after = geometry.divergence(corrected_velocity)
        final_image, final_hodge = action(pressure)
        pressure_residual = final_image - rhs
        hodge_status = jnp.where(hodge_status == 0, corrected_free.status, hodge_status)
        hodge_status = jnp.where(hodge_status == 0, final_hodge.status, hodge_status)
        residual_norm = jnp.sqrt(jnp.sum(geometry.cell_volumes * pressure_residual**2))
        divergence_norm = jnp.sqrt(jnp.sum(geometry.cell_volumes * divergence_after**2))
        rhs_norm = jnp.sqrt(jnp.sum(geometry.cell_volumes * rhs**2))
        finite = (
            tentative.finite
            & corrected_free.finite
            & jnp.all(jnp.isfinite(pressure))
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(divergence_norm)
        )
        converged = (
            ~active
            & ~failed
            & (hodge_status == 0)
            & finite
            & (divergence_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0))
        )
        return FreeSurfaceProjectionResult(
            momentum=corrected_momentum,
            velocity=corrected_velocity,
            pressure_head=pressure,
            pressure_increment=dt * pressure,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            pressure_residual=pressure_residual,
            pressure_residual_norm=residual_norm,
            hodge_residual_norm=jnp.maximum(
                tentative.residual_norm, corrected_free.residual_norm
            ),
            tentative_hodge=tentative,
            corrected_hodge=corrected_free,
            hodge_status=hodge_status,
            iterations=jnp.asarray(self.maximum_iterations, dtype=jnp.int32),
            converged=converged,
            finite=finite,
            successful=converged,
            plan_id=self.plan_id,
        )


__all__ = ["FreeSurfaceProjectionResult", "MappedFreeSurfaceProjectionPlan"]
