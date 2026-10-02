#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dynamic state of a labeled multiregion surface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ..._validation import unique_identifiers
from ...typing import Bool, checked, Dim, Float, Identifier, Size
from ._topology import MultiRegionSurfaceTopology


class _StateVertexDim(Dim, minimum=1):
    """Vertex slots of one multiregion surface state."""


class _StateSlotDim(Dim, minimum=1):
    """Region-pair slots per vertex."""


class _StateRegionDim(Dim, minimum=2):
    """Region slots."""


class _SheetFieldDim(Dim):
    """Named sheet-slot fields."""


class _RegionFieldDim(Dim):
    """Named region fields."""


def _field_names(names: Sequence[str], name: str, /) -> tuple[str, ...]:
    return unique_identifiers(names, name, allow_empty=True)


def _float_field(
    value: ArrayLike | None, shape: tuple[int, ...], dtype: np.dtype, name: str, /
) -> Array:
    if value is None:
        return jnp.zeros(shape, dtype=dtype)
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must be a floating array.")
    return array.astype(dtype)


@final
class MultiRegionSurfaceState(StrictModule):
    """Positions, velocities and extensive fields of one multiregion surface.

    Arrays are capacity-shaped; masks mark the active slots of the owning
    topology epoch. ``sheet_fields[v, s, k]`` is field ``k`` on the
    ``(vertex, region-pair)`` sheet slot ``s`` of vertex ``v`` (for example film
    liquid volume or surfactant amount); a manifold sheet is the one-slot case.
    ``region_fields[r, k]`` is an extensive field of region ``r`` (for example
    gas amount). Fields that can be transported across topology changes must
    be extensive; intensive quantities are reconstructed from them.
    """

    __strict_contract__ = True

    vertex_capacity: Size[_StateVertexDim] = eqx.field(static=True)
    slot_width: Size[_StateSlotDim] = eqx.field(static=True)
    region_capacity: Size[_StateRegionDim] = eqx.field(static=True)
    positions: Float[_StateVertexDim, Literal[3]]
    velocities: Float[_StateVertexDim, Literal[3]]
    sheet_fields: Float[_StateVertexDim, _StateSlotDim, _SheetFieldDim]
    region_fields: Float[_StateRegionDim, _RegionFieldDim]
    vertex_active: Bool[_StateVertexDim]
    slot_active: Bool[_StateVertexDim, _StateSlotDim]
    region_active: Bool[_StateRegionDim]
    sheet_field_names: tuple[str, ...] = eqx.field(static=True)
    region_field_names: tuple[str, ...] = eqx.field(static=True)
    topology_id: Identifier = eqx.field(static=True)
    epoch: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        topology: MultiRegionSurfaceTopology,
        positions: ArrayLike,
        /,
        *,
        velocities: ArrayLike | None = None,
        sheet_fields: ArrayLike | None = None,
        region_fields: ArrayLike | None = None,
        sheet_field_names: Sequence[str] = (),
        region_field_names: Sequence[str] = (),
    ) -> None:
        sheet_names = _field_names(sheet_field_names, "sheet_field_names")
        region_names = _field_names(region_field_names, "region_field_names")
        dtype = np.dtype(topology.plan.coordinate_dtype)
        vertices = topology.vertex_capacity
        slots = topology.slot_width
        regions = topology.region_capacity
        coordinates = _float_field(positions, (vertices, 3), dtype, "positions")
        motion = _float_field(velocities, (vertices, 3), dtype, "velocities")
        sheets = _float_field(
            sheet_fields, (vertices, slots, len(sheet_names)), dtype, "sheet_fields"
        )
        region_values = _float_field(
            region_fields, (regions, len(region_names)), dtype, "region_fields"
        )
        self.vertex_capacity = vertices
        self.slot_width = slots
        self.region_capacity = regions
        self.positions = coordinates
        self.velocities = motion
        self.sheet_fields = sheets
        self.region_fields = region_values
        self.vertex_active = topology.vertex_active
        self.slot_active = topology.slot_active
        self.region_active = topology.region_active
        self.sheet_field_names = sheet_names
        self.region_field_names = region_names
        self.topology_id = topology.topology_id
        self.epoch = topology.epoch

    def with_positions(self, positions: ArrayLike, /) -> MultiRegionSurfaceState:
        """Same state with new positions of identical shape and dtype."""
        values = jnp.asarray(positions)
        if values.shape != self.positions.shape:
            raise ValueError("positions must keep the state capacity shape.")
        return eqx.tree_at(
            lambda state: state.positions, self, values.astype(self.positions.dtype)
        )

    @checked
    def require_topology(self, topology: MultiRegionSurfaceTopology, /) -> None:
        """Refuse a state that belongs to another topology or epoch."""
        if self.topology_id != topology.topology_id or self.epoch != topology.epoch:
            raise ValueError("State belongs to a different multiregion topology epoch.")


__all__ = ["MultiRegionSurfaceState"]
