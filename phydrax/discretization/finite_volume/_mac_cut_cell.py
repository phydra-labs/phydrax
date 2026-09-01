#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._incompressible import PreparedMACOperators


class MACCutCellGeometryState(StrictModule):
    cell_fluid_fraction: Array
    face_open_fraction: tuple[Array, ...]
    solid_normal: Array
    wall_velocity: tuple[Array, ...]
    swept_volume_rate: Array
    geometric_conservation_residual: Array
    small_cell_mask: Array
    finite: Array
    successful: Array
    geometry_id: str = eqx.field(static=True)


class MACCutCellGeometryPlan(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    signed_distance: Callable[[Array, Array, Any], ArrayLike] = eqx.field(static=True)
    wall_velocity_provider: Callable[[Array, Array, Any], ArrayLike] = eqx.field(
        static=True
    )
    field_id: str = eqx.field(static=True)
    interface_width: float = eqx.field(static=True)
    small_cell_fraction: float = eqx.field(static=True)
    cell_points: Array
    face_points: tuple[Array, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: PreparedMACOperators,
        signed_distance: Callable[[Array, Array, Any], ArrayLike],
        wall_velocity: Callable[[Array, Array, Any], ArrayLike],
        /,
        *,
        field_id: str,
        interface_width: float,
        small_cell_fraction: float = 1.0e-2,
    ):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        if not callable(signed_distance) or not callable(wall_velocity):
            raise TypeError("solid geometry providers must be callable.")
        width, small = float(interface_width), float(small_cell_fraction)
        if width <= 0.0 or not 0.0 < small < 1.0:
            raise ValueError("Cut-cell interface/small-cell policy is invalid.")
        grid = operators.discretization.grid

        def points(layout):
            mesh = jnp.meshgrid(*layout.coordinates_by_axis, indexing="ij")
            return jnp.stack(tuple(value for value in mesh), axis=-1)

        self.operators = operators
        self.signed_distance = signed_distance
        self.wall_velocity_provider = wall_velocity
        self.field_id = str(field_id)
        self.interface_width = width
        self.small_cell_fraction = small
        self.cell_points = points(operators.discretization.cell_layout)
        self.face_points = tuple(
            points(layout) for layout in operators.discretization.face_layouts
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-cut-cell-geometry",
                "operators": operators.prepared_id,
                "field_id": str(field_id),
                "interface_width": width,
                "small_cell_fraction": small,
            }
        )

    def evaluate(
        self,
        time: ArrayLike,
        /,
        *,
        args: Any = None,
        previous: MACCutCellGeometryState | None = None,
        step_size: ArrayLike | None = None,
    ) -> MACCutCellGeometryState:
        time_ = jnp.asarray(time)
        phi = jnp.asarray(self.signed_distance(self.cell_points, time_, args))
        if phi.shape != self.operators.discretization.cell_shape:
            raise ValueError("Solid SDF must return one value per MAC cell.")
        width = self.interface_width
        cell_fraction = jnp.clip(0.5 + phi / (2.0 * width), 0.0, 1.0)
        face_fraction = []
        wall_velocity = []
        for axis, points in enumerate(self.face_points):
            face_phi = jnp.asarray(self.signed_distance(points, time_, args))
            face_fraction.append(jnp.clip(0.5 + face_phi / (2.0 * width), 0.0, 1.0))
            velocity = jnp.asarray(self.wall_velocity_provider(points, time_, args))
            if velocity.shape != points.shape:
                raise ValueError(
                    "Wall velocity provider must return one vector per face point."
                )
            wall_velocity.append(velocity[..., axis])
        gradients = tuple(
            jnp.gradient(phi, axis=axis)
            for axis in range(len(self.operators.discretization.cell_shape))
        )
        normal = jnp.stack(gradients, axis=-1)
        norm = jnp.sqrt(jnp.sum(normal**2, axis=-1))
        normal = normal / jnp.maximum(norm[..., None], 1.0e-30)
        if previous is None or step_size is None:
            swept = jnp.zeros_like(cell_fraction)
        else:
            dt = jnp.asarray(step_size, dtype=cell_fraction.dtype).reshape(())
            swept = (cell_fraction - previous.cell_fluid_fraction) / dt
        face_flux = []
        for axis, (fraction, wall) in enumerate(
            zip(face_fraction, wall_velocity, strict=True)
        ):
            face_flux.append((1.0 - fraction) * wall)
        geometric = swept + self.operators.divergence(tuple(face_flux))
        residual = jnp.sqrt(jnp.sum(geometric**2))
        small = (cell_fraction > 0.0) & (cell_fraction < self.small_cell_fraction)
        finite = (
            jnp.all(jnp.isfinite(cell_fraction))
            & jnp.all(jnp.isfinite(normal))
            & jnp.isfinite(residual)
        )
        return MACCutCellGeometryState(
            cell_fraction,
            tuple(face_fraction),
            normal,
            tuple(wall_velocity),
            swept,
            residual,
            small,
            finite,
            finite,
            self.plan_id,
        )


__all__ = ["MACCutCellGeometryPlan", "MACCutCellGeometryState"]
