#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class VortexRingSheetTopology(StrictModule, NonTrainableState):
    edge_start: Array
    edge_end: Array
    edge_active: Array
    ring_edges: Array
    ring_signs: Array
    ring_active: Array
    vertex_capacity: int = eqx.field(static=True)
    edge_capacity: int = eqx.field(static=True)
    ring_capacity: int = eqx.field(static=True)
    edges_per_ring: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertex_capacity: int,
        edge_start: ArrayLike,
        edge_end: ArrayLike,
        ring_edges: ArrayLike,
        ring_signs: ArrayLike,
        /,
        *,
        edge_active: ArrayLike | None = None,
        ring_active: ArrayLike | None = None,
    ):
        vertices = int(vertex_capacity)
        start, end = np.asarray(edge_start), np.asarray(edge_end)
        rings, signs = np.asarray(ring_edges), np.asarray(ring_signs)
        if (
            vertices <= 0
            or start.ndim != 1
            or end.shape != start.shape
            or rings.ndim != 2
            or signs.shape != rings.shape
        ):
            raise ValueError("Ring-sheet topology arrays are incompatible.")
        edge_capacity, ring_capacity, per_ring = (
            int(start.size),
            int(rings.shape[0]),
            int(rings.shape[1]),
        )
        if edge_capacity <= 0 or ring_capacity <= 0 or per_ring < 3:
            raise ValueError("Ring-sheet topology requires edges and closed rings.")
        active_edge = (
            np.ones((edge_capacity,), dtype=bool)
            if edge_active is None
            else np.asarray(edge_active, dtype=bool)
        )
        active_ring = (
            np.ones((ring_capacity,), dtype=bool)
            if ring_active is None
            else np.asarray(ring_active, dtype=bool)
        )
        if active_edge.shape != start.shape or active_ring.shape != (ring_capacity,):
            raise ValueError("Ring-sheet active masks are incompatible.")
        if np.any(
            (start[active_edge] < 0)
            | (start[active_edge] >= vertices)
            | (end[active_edge] < 0)
            | (end[active_edge] >= vertices)
            | (start[active_edge] == end[active_edge])
        ):
            raise ValueError("Active ring-sheet edges are invalid.")
        if np.any(
            (rings[active_ring] < 0) | (rings[active_ring] >= edge_capacity)
        ) or np.any(~np.isin(signs[active_ring], (-1, 1))):
            raise ValueError("Active ring incidence is invalid.")
        self.edge_start = jnp.asarray(np.where(active_edge, start, 0), dtype=jnp.int32)
        self.edge_end = jnp.asarray(np.where(active_edge, end, 0), dtype=jnp.int32)
        self.edge_active = jnp.asarray(active_edge)
        self.ring_edges = jnp.asarray(
            np.where(active_ring[:, None], rings, 0), dtype=jnp.int32
        )
        self.ring_signs = jnp.asarray(
            np.where(active_ring[:, None], signs, 0), dtype=jnp.int8
        )
        self.ring_active = jnp.asarray(active_ring)
        (
            self.vertex_capacity,
            self.edge_capacity,
            self.ring_capacity,
            self.edges_per_ring,
        ) = vertices, edge_capacity, ring_capacity, per_ring
        self.topology_id = canonical_fingerprint(
            {
                "kind": "vortex-ring-sheet-topology",
                "vertex_capacity": vertices,
                "edges": array_tree_fingerprint(
                    {"start": start, "end": end, "active": active_edge}
                ),
                "rings": array_tree_fingerprint(
                    {"edges": rings, "signs": signs, "active": active_ring}
                ),
            }
        )

    def edge_circulation(self, ring_circulation: ArrayLike, /) -> Array:
        gamma = jnp.asarray(ring_circulation)
        if gamma.shape != (self.ring_capacity,):
            raise ValueError("ring_circulation must have ring-capacity shape.")
        contribution = jnp.where(
            self.ring_active[:, None], gamma[:, None] * self.ring_signs, 0.0
        )
        return (
            jnp.zeros((self.edge_capacity,), dtype=gamma.dtype)
            .at[self.ring_edges.reshape(-1)]
            .add(contribution.reshape(-1))
        )


class VortexRingSheetState(StrictModule):
    topology: VortexRingSheetTopology
    vertices: Array
    ring_circulation: Array
    edge_core_radius: Array
    edge_age: Array

    def __init__(
        self,
        topology: VortexRingSheetTopology,
        vertices: ArrayLike,
        ring_circulation: ArrayLike,
        edge_core_radius: ArrayLike,
        edge_age: ArrayLike | None = None,
        /,
    ):
        if not isinstance(topology, VortexRingSheetTopology):
            raise TypeError("topology must be VortexRingSheetTopology.")
        vertex = jnp.asarray(vertices)
        gamma = jnp.asarray(ring_circulation, dtype=vertex.dtype)
        core = jnp.asarray(edge_core_radius, dtype=vertex.dtype)
        age = (
            jnp.zeros((topology.edge_capacity,), dtype=vertex.dtype)
            if edge_age is None
            else jnp.asarray(edge_age, dtype=vertex.dtype)
        )
        if (
            vertex.shape != (topology.vertex_capacity, 3)
            or gamma.shape != (topology.ring_capacity,)
            or core.shape != (topology.edge_capacity,)
            or age.shape != core.shape
        ):
            raise ValueError("Ring-sheet state arrays are incompatible.")
        finite = (
            jnp.all(jnp.isfinite(vertex))
            & jnp.all(jnp.isfinite(gamma))
            & jnp.all(
                jnp.where(
                    topology.edge_active,
                    jnp.isfinite(core) & (core > 0.0) & jnp.isfinite(age) & (age >= 0.0),
                    True,
                )
            )
        )
        self.topology = topology
        self.vertices = eqx.error_if(
            vertex, ~finite, "Active ring-sheet state must be finite."
        )
        self.ring_circulation = gamma
        self.edge_core_radius = jnp.where(topology.edge_active, core, 1.0)
        self.edge_age = jnp.where(topology.edge_active, age, 0.0)

    def edge_geometry(self, /) -> tuple[Array, Array, Array]:
        start = self.vertices[self.topology.edge_start]
        end = self.vertices[self.topology.edge_end]
        circulation = self.topology.edge_circulation(self.ring_circulation)
        return start, end, jnp.where(self.topology.edge_active, circulation, 0.0)


class VortexRingSheetEvidence(StrictModule):
    active_ring_count: Array
    active_edge_count: Array
    minimum_edge_length: Array
    circulation_residual: Array
    finite: Array
    state_id: str = eqx.field(static=True)


def ring_sheet_evidence(state: VortexRingSheetState, /) -> VortexRingSheetEvidence:
    start, end, edge_gamma = state.edge_geometry()
    length = jnp.linalg.norm(end - start, axis=-1)
    active = state.topology.edge_active
    # Closed sheets have no boundary circulation defect when oriented ring incidence sums on shared edges.
    residual = jnp.sum(jnp.where(active, edge_gamma, 0.0))
    finite = jnp.all(jnp.isfinite(state.vertices)) & jnp.all(jnp.isfinite(edge_gamma))
    return VortexRingSheetEvidence(
        jnp.sum(state.topology.ring_active, dtype=jnp.int32),
        jnp.sum(active, dtype=jnp.int32),
        jnp.min(jnp.where(active, length, jnp.inf)),
        residual,
        finite,
        canonical_fingerprint(
            {"kind": "vortex-ring-sheet-state", "topology": state.topology.topology_id}
        ),
    )


__all__ = [
    "VortexRingSheetEvidence",
    "VortexRingSheetState",
    "VortexRingSheetTopology",
    "ring_sheet_evidence",
]
