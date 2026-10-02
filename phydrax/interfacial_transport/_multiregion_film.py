#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Manifold B-film views of E multiregion sheet-slot storage.

A labeled multiregion surface is intentionally non-manifold.  This adapter
never presents that complex as one ``TriangleTopology``.  It prepares one
``PreparedFilmSurface`` for each region-pair sheet and keeps a fixed sparse
map between the sheet vertices and E's ``(vertex, region-pair)`` slots.
Boundary half-edges are mapped explicitly to the owning global surface edge;
a route is Plateau-border-supported exactly when that edge has three incident
faces.  Applications can therefore exchange liquid or surfactant through
explicit boundary routes without changing the manifold contract of B.
"""

from __future__ import annotations

from enum import IntEnum
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..geometry.multiregion_surface import (
    multiregion_sheet_views,
    MultiRegionSheetViews,
    MultiRegionSurfaceState,
    PreparedMultiRegionSurface,
)
from ..sparse import EdgeRelation, route_reduce
from ..typing import checked
from ._film_contracts import FilmSurfaceTopology, PreparedFilmSurface


class FilmSheetSlotStatus(IntEnum):
    """Preparation support of the B-on-E sheet adapter."""

    ACCEPTED = 0
    NONMANIFOLD_SHEET = 1
    SLOT_COVERAGE_INCOMPLETE = 2
    BOUNDARY_CAPACITY_EXCEEDED = 3


@final
class FilmSheetSlotEvidence(StrictModule):
    """Manifold, slot-coverage, and boundary-route preparation evidence."""

    status: FilmSheetSlotStatus = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    sheet_count: int = eqx.field(static=True)
    active_slot_count: int = eqx.field(static=True)
    packed_slot_capacity: int = eqx.field(static=True)
    boundary_route_count: int = eqx.field(static=True)
    boundary_route_capacity: int = eqx.field(static=True)
    plateau_border_route_count: int = eqx.field(static=True)
    unsupported_boundary_route_count: int = eqx.field(static=True)
    nonmanifold_pair_indices: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class FilmSheetSlotPreparationError(ValueError):
    """The multiregion epoch cannot implement B's manifold film contract."""

    def __init__(self, evidence: FilmSheetSlotEvidence, /) -> None:
        self.evidence = evidence
        super().__init__(
            f"Film sheet-slot preparation failed with status {evidence.status.name}."
        )


def _evidence(
    prepared: PreparedMultiRegionSurface,
    views: MultiRegionSheetViews,
    slot_indices: np.ndarray,
    boundary_count: int,
    boundary_capacity: int,
    supported_count: int,
    status: FilmSheetSlotStatus,
    /,
) -> FilmSheetSlotEvidence:
    topology = prepared.topology
    active_slots = int(np.count_nonzero(np.asarray(topology.slot_active)))
    payload = {
        "kind": "film-sheet-slot-evidence",
        "topology": topology.topology_id,
        "status": int(status),
        "views": views.views_id,
        "slots": array_tree_fingerprint(slot_indices),
        "boundary_count": boundary_count,
        "boundary_capacity": boundary_capacity,
        "supported_count": supported_count,
    }
    return FilmSheetSlotEvidence(
        status=status,
        accepted=status is FilmSheetSlotStatus.ACCEPTED,
        sheet_count=len(views.views),
        active_slot_count=active_slots,
        packed_slot_capacity=topology.vertex_capacity * topology.slot_width,
        boundary_route_count=boundary_count,
        boundary_route_capacity=boundary_capacity,
        plateau_border_route_count=supported_count,
        unsupported_boundary_route_count=boundary_count - supported_count,
        nonmanifold_pair_indices=views.nonmanifold_pair_indices,
        topology_id=topology.topology_id,
        evidence_id=canonical_fingerprint(payload),
    )


@final
class PreparedFilmSheetSlots(StrictModule):
    """Prepared per-sheet B operators and sparse E sheet-slot maps.

    ``packed_slot_indices`` begins with every active E slot exactly once,
    grouped by region-pair sheet.  The remaining rows are inert padding, so
    gather/scatter shapes stay at the declared E capacity.  Boundary routes
    are half-edges: each route identifies one sheet boundary cell and one
    global surface edge.  Positive route flux means content leaves the sheet
    cell and enters the border edge.
    """

    multiregion: PreparedMultiRegionSurface
    views: MultiRegionSheetViews
    surfaces: tuple[PreparedFilmSurface, ...]
    packed_slot_indices: Array
    packed_slot_valid: Array
    packed_vertex_area_m2: Array
    boundary_sheet_indices: Array
    boundary_local_edge_indices: Array
    boundary_endpoint_indices: Array
    boundary_slot_indices: Array
    boundary_global_edge_indices: Array
    boundary_route_valid: Array
    boundary_plateau_supported: Array
    boundary_to_slots: EdgeRelation
    boundary_to_edges: EdgeRelation
    geometry_revision: Array
    evidence: FilmSheetSlotEvidence
    sheet_offsets: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        prepared: PreparedMultiRegionSurface,
        state: MultiRegionSurfaceState,
        /,
        *,
        geometry_revision: ArrayLike = 0,
    ) -> None:
        topology = prepared.topology
        state.require_topology(topology)
        revision = jnp.asarray(geometry_revision, dtype=jnp.int32)
        if revision.shape != ():
            raise ValueError("geometry_revision must be a scalar.")
        views = multiregion_sheet_views(prepared, state)
        packed_capacity = topology.vertex_capacity * topology.slot_width
        boundary_capacity = 2 * topology.valence_width * topology.edge_capacity
        empty_slots = np.zeros((0,), dtype=np.int64)
        slot_indices = (
            np.concatenate(
                [np.asarray(view.slot_indices, dtype=np.int64) for view in views.views]
            )
            if views.views
            else empty_slots
        )
        if views.nonmanifold_pair_indices:
            evidence = _evidence(
                prepared,
                views,
                slot_indices,
                0,
                boundary_capacity,
                0,
                FilmSheetSlotStatus.NONMANIFOLD_SHEET,
            )
            raise FilmSheetSlotPreparationError(evidence)
        active_flat = np.flatnonzero(np.asarray(topology.slot_active).reshape(-1))
        if (
            slot_indices.size != active_flat.size
            or np.unique(slot_indices).size != slot_indices.size
            or not np.array_equal(np.sort(slot_indices), active_flat)
        ):
            evidence = _evidence(
                prepared,
                views,
                slot_indices,
                0,
                boundary_capacity,
                0,
                FilmSheetSlotStatus.SLOT_COVERAGE_INCOMPLETE,
            )
            raise FilmSheetSlotPreparationError(evidence)
        surfaces = tuple(
            PreparedFilmSurface(
                FilmSurfaceTopology(view.mesh),
                view.mesh.vertices,
                geometry_revision=revision,
            )
            for view in views.views
        )
        offsets = [0]
        for surface in surfaces:
            offsets.append(offsets[-1] + surface.topology.num_vertices)
        packed_indices = np.zeros((packed_capacity,), dtype=np.int64)
        packed_valid = np.zeros((packed_capacity,), dtype=np.bool_)
        packed_area = np.zeros((packed_capacity,), dtype=np.float64)
        packed_indices[: slot_indices.size] = slot_indices
        packed_valid[: slot_indices.size] = True
        if surfaces:
            packed_area[: slot_indices.size] = np.concatenate(
                [np.asarray(surface.vertex_area) for surface in surfaces]
            )

        active_edges = np.asarray(topology.edges[: topology.edge_count], dtype=np.int64)
        edge_lookup = {
            (int(active_edges[index, 0]), int(active_edges[index, 1])): index
            for index in range(active_edges.shape[0])
        }
        valence = np.sum(
            np.asarray(topology.edge_faces[: topology.edge_count]) >= 0, axis=1
        )
        route_sheet: list[int] = []
        route_local_edge: list[int] = []
        route_endpoint: list[int] = []
        route_slot: list[int] = []
        route_edge: list[int] = []
        route_supported: list[bool] = []
        for sheet_index, (view, surface) in enumerate(
            zip(views.views, surfaces, strict=True)
        ):
            local_edges = np.asarray(surface.topology.edges, dtype=np.int64)
            boundary = np.flatnonzero(np.asarray(surface.topology.boundary_edges))
            vertices = np.asarray(view.vertex_indices, dtype=np.int64)
            slots = np.asarray(view.slot_indices, dtype=np.int64)
            for local_edge_index in boundary:
                endpoints = local_edges[local_edge_index]
                global_vertices = vertices[endpoints]
                first, second = int(global_vertices[0]), int(global_vertices[1])
                key = (first, second) if first < second else (second, first)
                global_edge = edge_lookup[key]
                for endpoint, local_vertex in enumerate(endpoints):
                    route_sheet.append(sheet_index)
                    route_local_edge.append(int(local_edge_index))
                    route_endpoint.append(endpoint)
                    route_slot.append(int(slots[local_vertex]))
                    route_edge.append(global_edge)
                    route_supported.append(bool(valence[global_edge] == 3))
        boundary_count = len(route_slot)
        if boundary_count > boundary_capacity:
            evidence = _evidence(
                prepared,
                views,
                slot_indices,
                boundary_count,
                boundary_capacity,
                int(np.count_nonzero(route_supported)),
                FilmSheetSlotStatus.BOUNDARY_CAPACITY_EXCEEDED,
            )
            raise FilmSheetSlotPreparationError(evidence)

        def padded(values: list[int], fill: int = 0) -> np.ndarray:
            result = np.full((boundary_capacity,), fill, dtype=np.int64)
            result[:boundary_count] = np.asarray(values, dtype=np.int64)
            return result

        route_valid = np.arange(boundary_capacity) < boundary_count
        supported = np.zeros((boundary_capacity,), dtype=np.bool_)
        supported[:boundary_count] = np.asarray(route_supported, dtype=np.bool_)
        route_slots = padded(route_slot)
        route_edges = padded(route_edge)
        evidence = _evidence(
            prepared,
            views,
            slot_indices,
            boundary_count,
            boundary_capacity,
            int(np.count_nonzero(supported)),
            FilmSheetSlotStatus.ACCEPTED,
        )
        self.multiregion = prepared
        self.views = views
        self.surfaces = surfaces
        self.packed_slot_indices = jnp.asarray(packed_indices, dtype=jnp.int32)
        self.packed_slot_valid = jnp.asarray(packed_valid)
        self.packed_vertex_area_m2 = jnp.asarray(packed_area)
        self.boundary_sheet_indices = jnp.asarray(padded(route_sheet), dtype=jnp.int32)
        self.boundary_local_edge_indices = jnp.asarray(
            padded(route_local_edge), dtype=jnp.int32
        )
        self.boundary_endpoint_indices = jnp.asarray(
            padded(route_endpoint), dtype=jnp.int32
        )
        self.boundary_slot_indices = jnp.asarray(route_slots, dtype=jnp.int32)
        self.boundary_global_edge_indices = jnp.asarray(route_edges, dtype=jnp.int32)
        self.boundary_route_valid = jnp.asarray(route_valid)
        self.boundary_plateau_supported = jnp.asarray(supported)
        route_axis = np.arange(boundary_capacity, dtype=np.int64)
        self.boundary_to_slots = EdgeRelation(
            route_axis,
            route_slots,
            source_size=boundary_capacity,
            target_size=packed_capacity,
            valid=route_valid,
        )
        self.boundary_to_edges = EdgeRelation(
            route_axis,
            route_edges,
            source_size=boundary_capacity,
            target_size=topology.edge_capacity,
            valid=route_valid,
        )
        self.geometry_revision = revision
        self.evidence = evidence
        self.sheet_offsets = tuple(offsets)
        self.topology_id = topology.topology_id
        self.lineage_id = topology.lineage_id
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "prepared-film-sheet-slots",
                "topology": topology.topology_id,
                "lineage": topology.lineage_id,
                "slot_indices": array_tree_fingerprint(slot_indices),
                "boundary_slots": array_tree_fingerprint(
                    np.asarray(route_slot, dtype=np.int64)
                ),
                "boundary_edges": array_tree_fingerprint(
                    np.asarray(route_edge, dtype=np.int64)
                ),
            }
        )

    @property
    def packed_slot_capacity(self) -> int:
        return self.packed_slot_indices.shape[0]

    @property
    def boundary_route_capacity(self) -> int:
        return self.boundary_slot_indices.shape[0]

    def gather_slot_content(self, slot_content: ArrayLike, /) -> Array:
        """Gather capacity-shaped E slot content into sheet-grouped storage."""
        values = jnp.asarray(slot_content)
        topology = self.multiregion.topology
        expected = (topology.vertex_capacity, topology.slot_width)
        if values.shape[:2] != expected:
            raise ValueError(f"slot_content must begin with shape {expected}.")
        trailing = values.shape[2:]
        flat = values.reshape((self.packed_slot_capacity,) + trailing)
        gathered = flat[self.packed_slot_indices]
        valid = self.packed_slot_valid.reshape(
            self.packed_slot_valid.shape + (1,) * len(trailing)
        )
        return jnp.where(valid, gathered, jnp.zeros((), dtype=gathered.dtype))

    def scatter_slot_content(self, packed_content: ArrayLike, /) -> Array:
        """Scatter sheet-grouped extensive content into E's slot layout."""
        values = jnp.asarray(packed_content)
        if values.shape[:1] != (self.packed_slot_capacity,):
            raise ValueError("packed_content must use packed_slot_capacity.")
        trailing = values.shape[1:]
        valid = self.packed_slot_valid.reshape(
            self.packed_slot_valid.shape + (1,) * len(trailing)
        )
        material = jnp.where(valid, values, jnp.zeros((), dtype=values.dtype))
        flat = jnp.zeros((self.packed_slot_capacity,) + trailing, dtype=values.dtype)
        flat = flat.at[self.packed_slot_indices].add(material)
        topology = self.multiregion.topology
        return flat.reshape((topology.vertex_capacity, topology.slot_width) + trailing)

    def sheet_content(self, slot_content: ArrayLike, sheet_index: int, /) -> Array:
        """Return one sheet's vertex content in its B operator ordering."""
        if not isinstance(sheet_index, int):
            raise TypeError("sheet_index must be an int.")
        if sheet_index < 0 or sheet_index >= len(self.surfaces):
            raise ValueError("sheet_index lies outside the prepared sheet tuple.")
        packed = self.gather_slot_content(slot_content)
        return packed[
            self.sheet_offsets[sheet_index] : self.sheet_offsets[sheet_index + 1]
        ]

    def boundary_slot_rate(self, route_rate: ArrayLike, /) -> Array:
        """Sum half-edge route rates onto E sheet slots."""
        values = self._boundary_values(route_rate)
        flat = route_reduce(self.boundary_to_slots, values)
        topology = self.multiregion.topology
        return flat.reshape((topology.vertex_capacity, topology.slot_width))

    def distribute_slot_boundary_rate(self, slot_rate: ArrayLike, /) -> Array:
        """Partition an aggregated slot rate uniformly over its boundary halves.

        This is the conservative lumped adapter for B1's per-vertex
        ``boundary_exchange_m3``. The returned route sum at every slot equals
        the supplied rate exactly; the method does not invent a flux on
        interior slots.
        """
        values = jnp.asarray(slot_rate, dtype=jnp.float64)
        topology = self.multiregion.topology
        expected = (topology.vertex_capacity, topology.slot_width)
        if values.shape != expected:
            raise ValueError(f"slot_rate must have shape {expected}.")
        route_count = route_reduce(
            self.boundary_to_slots,
            self.boundary_route_valid.astype(jnp.float64),
        )
        flat = values.reshape((-1,))
        counts = route_count[self.boundary_slot_indices]
        return jnp.where(
            self.boundary_route_valid,
            flat[self.boundary_slot_indices] / jnp.maximum(counts, 1.0),
            0.0,
        )

    def boundary_global_edge_rate(self, route_rate: ArrayLike, /) -> Array:
        """Sum half-edge route rates onto global multiregion edges."""
        return route_reduce(self.boundary_to_edges, self._boundary_values(route_rate))

    def unsupported_boundary_rate(self, route_rate: ArrayLike, /) -> Array:
        """Absolute flux requested on a boundary without Plateau-border support."""
        values = self._boundary_values(route_rate)
        invalid = self.boundary_route_valid & ~self.boundary_plateau_supported
        return jnp.sum(jnp.where(invalid, jnp.abs(values), 0.0))

    def refresh(
        self,
        prepared: PreparedMultiRegionSurface,
        state: MultiRegionSurfaceState,
        /,
        *,
        geometry_revision: ArrayLike,
    ) -> PreparedFilmSheetSlots:
        """Refresh only numeric B geometry after fixed-topology B4 motion."""
        if prepared.topology.topology_id != self.topology_id:
            raise ValueError("A geometry refresh must preserve the topology epoch.")
        state.require_topology(prepared.topology)
        revision = jnp.asarray(geometry_revision, dtype=jnp.int32)
        if revision.shape != ():
            raise ValueError("geometry_revision must be a scalar.")
        positions = jnp.asarray(state.positions)
        surfaces = tuple(
            PreparedFilmSurface(
                surface.topology,
                positions[view.vertex_indices],
                geometry_revision=revision,
            )
            for surface, view in zip(self.surfaces, self.views.views, strict=True)
        )
        active_area = jnp.concatenate(tuple(surface.vertex_area for surface in surfaces))
        packed_area = jnp.zeros_like(self.packed_vertex_area_m2)
        packed_area = packed_area.at[: self.evidence.active_slot_count].set(active_area)
        return eqx.tree_at(
            lambda value: (
                value.multiregion,
                value.surfaces,
                value.packed_vertex_area_m2,
                value.geometry_revision,
            ),
            self,
            (prepared, surfaces, packed_area, revision),
        )

    def _boundary_values(self, route_rate: ArrayLike, /) -> Array:
        values = jnp.asarray(route_rate, dtype=jnp.float64)
        if values.shape != (self.boundary_route_capacity,):
            raise ValueError("Boundary route rates must use boundary_route_capacity.")
        return jnp.where(self.boundary_route_valid, values, 0.0)


def prepare_film_sheet_slots(
    prepared: PreparedMultiRegionSurface,
    state: MultiRegionSurfaceState,
    /,
    *,
    geometry_revision: ArrayLike = 0,
) -> PreparedFilmSheetSlots:
    """Prepare E sheet slots as independent manifold B film surfaces."""
    return PreparedFilmSheetSlots(prepared, state, geometry_revision=geometry_revision)


__all__ = [
    "FilmSheetSlotEvidence",
    "FilmSheetSlotPreparationError",
    "FilmSheetSlotStatus",
    "PreparedFilmSheetSlots",
    "prepare_film_sheet_slots",
]
