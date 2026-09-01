#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import FaceVelocity
from ..discretization.flip import MACFreeSurfaceGeometryState
from ._mac_free_surface import (
    MACFreeSurfaceProjectionPlan,
    MACFreeSurfaceProjectionResult,
)


def _cell_to_face(value: Array, shape: tuple[int, ...], axis: int, /) -> Array:
    if value.shape == shape:
        return value
    if shape[axis] != value.shape[axis] + 1:
        raise ValueError("Cell interface data cannot map to this face layout.")
    result = jnp.zeros(shape, dtype=value.dtype)
    face_index = [slice(None)] * value.ndim
    cell_index = [slice(None)] * value.ndim
    face_index[axis] = slice(1, shape[axis] - 1)
    cell_index[axis] = slice(0, value.shape[axis] - 1)
    return result.at[tuple(face_index)].set(value[tuple(cell_index)])


class MACGhostFluidProjectionResult(StrictModule):
    velocity: FaceVelocity
    pressure: Array
    jump_impulse: FaceVelocity
    geometry: MACFreeSurfaceGeometryState
    projection: MACFreeSurfaceProjectionResult
    jump_work: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MACGhostFluidProjectionPlan(StrictModule, NonTrainableState):
    projection: MACFreeSurfaceProjectionPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, projection: MACFreeSurfaceProjectionPlan, /):
        if not isinstance(projection, MACFreeSurfaceProjectionPlan):
            raise TypeError("projection must be MACFreeSurfaceProjectionPlan.")
        self.projection = projection
        self.plan_id = canonical_fingerprint(
            {"kind": "mac-ghost-fluid-projection", "projection": projection.plan_id}
        )

    def project(
        self,
        velocity: FaceVelocity,
        geometry: MACFreeSurfaceGeometryState,
        step_size: ArrayLike,
        /,
        *,
        pressure_jump: ArrayLike | None = None,
        pressure: ArrayLike | None = None,
    ) -> MACGhostFluidProjectionResult:
        values = self.projection.operators.validate_velocity(velocity)
        if not isinstance(geometry, MACFreeSurfaceGeometryState):
            raise TypeError("geometry must be MACFreeSurfaceGeometryState.")
        jump = (
            jnp.zeros_like(geometry.signed_distance)
            if pressure_jump is None
            else jnp.asarray(pressure_jump, dtype=geometry.signed_distance.dtype)
        )
        if jump.shape != geometry.signed_distance.shape:
            raise ValueError("pressure_jump must match MAC cells.")
        dt = jnp.asarray(step_size, dtype=jump.dtype).reshape(())
        impulses = []
        adjusted = []
        for axis, component in enumerate(values):
            neighbor_jump = jnp.roll(jump, -1, axis=axis)
            sign = jnp.where(geometry.signed_distance <= 0.0, 1.0, -1.0)
            spacing = jnp.mean(
                self.projection.operators.discretization.grid.structured_axes[
                    axis
                ].interval_widths
            )
            cell_gradient = jnp.where(
                geometry.interface_faces[axis],
                sign
                * 0.5
                * (jump + neighbor_jump)
                / (geometry.ghost_fraction[axis] * spacing),
                0.0,
            )
            gradient = _cell_to_face(cell_gradient, component.shape, axis)
            impulse = -dt / self.projection.density * gradient
            impulses.append(impulse)
            adjusted.append(component + impulse)
        projected = self.projection.project(
            tuple(adjusted),
            geometry.liquid_mask,
            dt,
            pressure=pressure,
        )
        jump_work = sum(
            jnp.sum(impulse * (before + 0.5 * impulse))
            for impulse, before in zip(impulses, values, strict=True)
        )
        finite = (
            geometry.finite
            & jnp.isfinite(jump_work)
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in impulses))
            )
        )
        successful = geometry.successful & projected.successful & finite
        return MACGhostFluidProjectionResult(
            projected.velocity,
            projected.pressure,
            tuple(impulses),
            geometry,
            projected,
            jump_work,
            finite,
            successful,
            self.plan_id,
        )


__all__ = ["MACGhostFluidProjectionPlan", "MACGhostFluidProjectionResult"]
