#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._incompressible import FaceVelocity, PreparedMACOperators
from ._mac_cut_cell import MACCutCellGeometryState
from ._mac_interface_state import MACFreeSurfaceGeometryState


def _cell_to_face_fraction(value: Array, shape: tuple[int, ...], axis: int, /) -> Array:
    if value.shape == shape:
        return value
    result = jnp.zeros(shape, dtype=value.dtype)
    face_index = [slice(None)] * value.ndim
    cell_index = [slice(None)] * value.ndim
    face_index[axis] = slice(1, shape[axis] - 1)
    cell_index[axis] = slice(0, value.shape[axis] - 1)
    result = result.at[tuple(face_index)].set(value[tuple(cell_index)])
    lower = [slice(None)] * value.ndim
    upper = [slice(None)] * value.ndim
    lower[axis] = 0
    upper[axis] = shape[axis] - 1
    return result.at[tuple(lower)].set(0.0).at[tuple(upper)].set(0.0)


class MACFreeSurfaceViscousMeasures(StrictModule):
    cell_fraction: Array
    face_fraction: FaceVelocity
    face_mass: FaceVelocity
    cell_viscosity: Array
    solid_face_fraction: FaceVelocity
    finite: Array
    successful: Array
    measure_id: str = eqx.field(static=True)


class MACFreeSurfaceViscousMeasurePlan(StrictModule, NonTrainableState):
    operators: PreparedMACOperators
    density: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, operators: PreparedMACOperators, density: float, /):
        if not isinstance(operators, PreparedMACOperators):
            raise TypeError("operators must be PreparedMACOperators.")
        density_ = float(density)
        if density_ <= 0.0:
            raise ValueError("density must be positive.")
        self.operators = operators
        self.density = density_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-free-surface-viscous-measures",
                "operators": operators.prepared_id,
                "density": density_,
            }
        )

    def evaluate(
        self,
        interface: MACFreeSurfaceGeometryState,
        viscosity: ArrayLike,
        /,
        *,
        solid: MACCutCellGeometryState | None = None,
    ) -> MACFreeSurfaceViscousMeasures:
        mu = jnp.asarray(viscosity, dtype=interface.cell_fraction.dtype)
        if mu.shape == ():
            mu = jnp.full_like(interface.cell_fraction, mu)
        if mu.shape != interface.cell_fraction.shape:
            raise ValueError("viscosity must be scalar or cell shaped.")
        solid_cell = (
            jnp.ones_like(interface.cell_fraction)
            if solid is None
            else solid.cell_fluid_fraction
        )
        cell_fraction = interface.cell_fraction * solid_cell
        face_fraction = []
        solid_fraction = []
        face_mass = []
        for axis, layout in enumerate(self.operators.discretization.face_layouts):
            liquid = interface.face_fraction[axis]
            solid_open = (
                jnp.ones(layout.shape, dtype=liquid.dtype)
                if solid is None
                else solid.face_open_fraction[axis]
            )
            fraction = _cell_to_face_fraction(liquid, layout.shape, axis) * solid_open
            face_fraction.append(fraction)
            solid_fraction.append(solid_open)
            face_mass.append(
                self.density * fraction * layout.measure.astype(fraction.dtype)
            )
        finite = (
            interface.finite
            & jnp.all(jnp.isfinite(mu) & (mu >= 0.0))
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in face_mass))
            )
        )
        identifier = canonical_fingerprint(
            {
                "kind": "mac-free-surface-viscous-measure-state",
                "plan": self.plan_id,
                "interface": interface.geometry_id,
                "solid": None if solid is None else solid.geometry_id,
            }
        )
        return MACFreeSurfaceViscousMeasures(
            cell_fraction,
            tuple(face_fraction),
            tuple(face_mass),
            mu,
            tuple(solid_fraction),
            finite,
            interface.successful & (solid is None or solid.successful) & finite,
            identifier,
        )


__all__ = ["MACFreeSurfaceViscousMeasurePlan", "MACFreeSurfaceViscousMeasures"]
