#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared geometry, energy and force evaluation of multiregion surfaces.

For face ``f`` with corners ``x0, x1, x2`` the area vector is
``a_f = (x1 - x0) x (x2 - x0) / 2`` and points out of ``left(f)``. The signed
volume of a finite region ``r`` follows from the divergence theorem,

``V_r = sum_f s_rf (x0 - c) . ((x1 - c) x (x2 - c)) / 6``,

with ``s_rf = +1`` when ``r = left(f)``, ``-1`` when ``r = right(f)`` and a fixed
reference point ``c`` (exact for closed cycles, chosen to limit cancellation).
Boundary labels have no volume. The surface energy is ``E = sum_f gamma_f A_f``
with the effective pair tension ``gamma_f`` of the face's two regions (a soap
film carries ``2 sigma``); forces are the fixed-topology negative gradient
``-dE/dx``, so junction balance (Plateau/Herring angles) emerges without any
junction-curvature construction.
"""

from __future__ import annotations

from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._bvh import PackedBVH, prepare_bvh, refit_packed_bvh_bounds
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...sparse import EdgeRelation, gather_routes, linear_apply, RowRelation
from ...typing import Bool, Dim, Float, Identifier, Integer, Scalar, Size
from ._contracts import (
    MultiRegionSurfaceEvidence,
    MultiRegionSurfacePreparationError,
    MultiRegionSurfaceValidationPolicy,
)
from ._state import MultiRegionSurfaceState
from ._topology import MultiRegionSurfaceTopology
from ._validation import validate_multiregion_surface


class _GeometryFaceDim(Dim, minimum=1):
    """Face slots."""


class _GeometryEdgeDim(Dim, minimum=1):
    """Edge slots."""


class _GeometryValenceDim(Dim, minimum=1):
    """Incident-face slots per edge."""


class _GeometryRegionDim(Dim, minimum=2):
    """Region slots."""


class _GeometryPairDim(Dim, minimum=1):
    """Region-pair slots."""


class _GeometryRouteDim(Dim, minimum=2):
    """Face-to-region routes (two per face slot)."""


class _GeometryVertexDim(Dim, minimum=1):
    """Vertex slots."""


class _GeometrySlotDim(Dim, minimum=1):
    """Sheet slots per vertex."""


@final
class MultiRegionSurfaceGeometry(StrictModule):
    """Geometry, energy and conservative forces of one configuration."""

    __strict_contract__ = True

    face_areas: Float[_GeometryFaceDim]
    face_normals: Float[_GeometryFaceDim, Literal[3]]
    region_volumes: Float[_GeometryRegionDim]
    pair_areas: Float[_GeometryPairDim]
    slot_areas: Float[_GeometryVertexDim, _GeometrySlotDim]
    surface_energy: Float[Scalar]
    forces: Float[_GeometryVertexDim, Literal[3]]
    minimum_face_area: Float[Scalar]
    finite: Bool[Scalar]


@final
class JunctionWedges(StrictModule):
    """Counterclockwise wedges between consecutive faces around every edge.

    Row ``e`` lists the wedge angles (radians, summing to ``2 pi`` on active
    edges) in counterclockwise order about ``v1 - v0`` together with the region
    occupying each wedge. For a Plateau border (three films) these are the
    three junction angles; for a manifold sheet edge they are ``pi +/- bend``.
    """

    __strict_contract__ = True

    angles: Float[_GeometryEdgeDim, _GeometryValenceDim]
    regions: Integer[_GeometryEdgeDim, _GeometryValenceDim]
    valid: Bool[_GeometryEdgeDim, _GeometryValenceDim]
    valence: Integer[_GeometryEdgeDim]


def _host_opposite(topology: MultiRegionSurfaceTopology, /) -> np.ndarray:
    faces = np.asarray(topology.faces, dtype=np.int64)
    edges = np.asarray(topology.edges, dtype=np.int64)
    edge_faces = np.asarray(topology.edge_faces, dtype=np.int64)
    rows = faces[np.maximum(edge_faces, 0)]
    on_edge = (rows == edges[:, None, 0:1]) | (rows == edges[:, None, 1:2])
    opposite = np.take_along_axis(
        rows, np.argmax(~on_edge, axis=2)[..., None], axis=2
    )[..., 0]
    return np.where(edge_faces >= 0, opposite, -1)


def _safe_norm(vectors: Array, valid: Array, /) -> Array:
    squared = jnp.sum(vectors * vectors, axis=-1)
    safe = jnp.where(valid, squared, jnp.ones_like(squared))
    return jnp.where(valid, jnp.sqrt(safe), jnp.zeros_like(squared))


@final
class PreparedMultiRegionSurface(StrictModule):
    """Validated topology epoch with reusable sparse routes and a face BVH.

    Construction runs `validate_multiregion_surface` on the supplied state and
    refuses anything but ``ACCEPTED`` with `MultiRegionSurfacePreparationError`.
    Sparse relations route face corners (gather), faces to finite regions
    (signed volume accumulation), faces to region pairs, and face corners to
    ``(vertex, region-pair)`` sheet slots; all evaluation methods are pure JAX
    over capacity-shaped arrays and differentiable at fixed topology.
    """

    __strict_contract__ = True

    topology: MultiRegionSurfaceTopology
    evidence: MultiRegionSurfaceEvidence
    face_corners: RowRelation
    face_regions: EdgeRelation
    face_region_signs: Float[_GeometryRouteDim]
    face_pair_relation: EdgeRelation
    corner_slots: EdgeRelation
    edge_opposite: Integer[_GeometryEdgeDim, _GeometryValenceDim]
    volume_reference: Float[Literal[3]]
    face_bvh: PackedBVH
    face_count: Size[_GeometryFaceDim] = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        topology: MultiRegionSurfaceTopology,
        state: MultiRegionSurfaceState,
        /,
        *,
        policy: MultiRegionSurfaceValidationPolicy | None = None,
    ) -> None:
        evidence = validate_multiregion_surface(topology, state, policy=policy)
        if not evidence.accepted:
            raise MultiRegionSurfacePreparationError(evidence)
        vcap = topology.vertex_capacity
        fcap = topology.face_capacity
        slots = topology.slot_width
        face_active = np.asarray(topology.face_active)
        faces = np.asarray(topology.faces, dtype=np.int64)
        labels = np.asarray(topology.face_labels, dtype=np.int64)
        finite = np.asarray(topology.region_finite)
        region_valid = np.repeat(face_active, 2) & finite[
            np.maximum(labels, 0).reshape(-1)
        ]
        corner_targets = faces * slots + np.asarray(topology.face_corner_slots)
        points = np.asarray(state.positions[: topology.vertex_count], dtype=np.float64)
        triangles = points[faces[: topology.face_count]]
        coordinate = np.dtype(topology.plan.coordinate_dtype)
        self.topology = topology
        self.evidence = evidence
        self.face_corners = RowRelation(
            np.maximum(faces, 0),
            source_size=vcap,
            valid=np.broadcast_to(face_active[:, None], faces.shape),
        )
        self.face_regions = EdgeRelation(
            np.repeat(np.arange(fcap), 2),
            np.maximum(labels, 0).reshape(-1),
            source_size=fcap,
            target_size=topology.region_capacity,
            valid=region_valid,
        )
        self.face_region_signs = jnp.asarray(
            np.tile(np.asarray((1.0, -1.0)), fcap), dtype=coordinate
        )
        self.face_pair_relation = EdgeRelation(
            np.arange(fcap),
            np.maximum(np.asarray(topology.face_pairs), 0),
            source_size=fcap,
            target_size=topology.region_pair_capacity,
            valid=face_active,
        )
        self.corner_slots = EdgeRelation(
            np.repeat(np.arange(fcap), 3),
            np.maximum(corner_targets, 0).reshape(-1),
            source_size=fcap,
            target_size=vcap * slots,
            valid=np.repeat(face_active, 3),
        )
        self.edge_opposite = jnp.asarray(
            _host_opposite(topology), dtype=topology.edges.dtype
        )
        self.volume_reference = jnp.asarray(np.mean(points, axis=0), dtype=coordinate)
        self.face_bvh = prepare_bvh(
            np.min(triangles, axis=1), np.max(triangles, axis=1), dtype=coordinate
        )
        self.face_count = fcap
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-multiregion-surface",
                "topology": topology.topology_id,
                "lineage": topology.lineage_id,
                "evidence": evidence.evidence_id,
                "reference": array_tree_fingerprint(points),
            }
        )

    # ----------------------------------------------------------------- geometry

    def face_corner_positions(self, positions: ArrayLike, /) -> Array:
        """Corner coordinates ``(faces, 3, 3)``; inactive faces are zero."""
        return gather_routes(self.face_corners, jnp.asarray(positions))

    def face_area_vectors(self, positions: ArrayLike, /) -> Array:
        """``(x1 - x0) x (x2 - x0) / 2`` per face, pointing out of ``left``."""
        corners = self.face_corner_positions(positions)
        return 0.5 * jnp.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])

    def face_areas(self, positions: ArrayLike, /) -> Array:
        return _safe_norm(self.face_area_vectors(positions), self.topology.face_active)

    def region_volumes(self, positions: ArrayLike, /) -> Array:
        """Signed volumes of finite regions; zero for boundary and inactive labels."""
        corners = self.face_corner_positions(positions) - self.volume_reference
        contribution = (
            jnp.sum(corners[:, 0] * jnp.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0
        )
        contribution = jnp.where(self.topology.face_active, contribution, 0.0)
        return linear_apply(self.face_regions, self.face_region_signs, contribution)

    def pair_areas(self, positions: ArrayLike, /) -> Array:
        """Total area of every region-pair sheet."""
        areas = self.face_areas(positions)
        return linear_apply(
            self.face_pair_relation, jnp.ones((self.face_count,), areas.dtype), areas
        )

    def slot_areas(self, positions: ArrayLike, /) -> Array:
        """Barycentric area ``sum A_f / 3`` of every ``(vertex, region-pair)`` slot."""
        topology = self.topology
        areas = self.face_areas(positions)
        weights = jnp.full((3 * self.face_count,), 1.0 / 3.0, dtype=areas.dtype)
        flat = linear_apply(self.corner_slots, weights, areas)
        return flat.reshape((topology.vertex_capacity, topology.slot_width))

    def surface_energy(self, positions: ArrayLike, face_tension: ArrayLike, /) -> Array:
        """``sum_f gamma_f A_f`` with capacity-shaped effective face tensions."""
        tension = jnp.asarray(face_tension)
        if tension.shape != (self.face_count,):
            raise ValueError("face_tension must have one entry per face slot.")
        areas = self.face_areas(positions)
        return jnp.sum(jnp.where(self.topology.face_active, tension * areas, 0.0))

    def forces(self, positions: ArrayLike, face_tension: ArrayLike, /) -> Array:
        """Fixed-topology conservative forces ``-dE/dx``; inactive vertices are zero."""
        gradient = jax.grad(self.surface_energy)(jnp.asarray(positions), face_tension)
        return jnp.where(self.topology.vertex_active[:, None], -gradient, 0.0)

    def evaluate(
        self, state: MultiRegionSurfaceState, face_tension: ArrayLike, /
    ) -> MultiRegionSurfaceGeometry:
        """Complete geometry, energy and force evaluation of one state."""
        if state.topology_id != self.topology.topology_id:
            raise ValueError("State belongs to a different multiregion topology.")
        positions = state.positions
        vectors = self.face_area_vectors(positions)
        active = self.topology.face_active
        areas = _safe_norm(vectors, active)
        safe = jnp.where(active, areas, 1.0)
        normals = jnp.where(active[:, None], vectors / safe[:, None], 0.0)
        energy, gradient = jax.value_and_grad(self.surface_energy)(positions, face_tension)
        forces = jnp.where(self.topology.vertex_active[:, None], -gradient, 0.0)
        volumes = self.region_volumes(positions)
        return MultiRegionSurfaceGeometry(
            face_areas=areas,
            face_normals=normals,
            region_volumes=volumes,
            pair_areas=self.pair_areas(positions),
            slot_areas=self.slot_areas(positions),
            surface_energy=energy,
            forces=forces,
            minimum_face_area=jnp.min(jnp.where(active, areas, jnp.inf)),
            finite=jnp.all(jnp.isfinite(positions)) & jnp.isfinite(energy),
        )

    def junction_wedges(self, positions: ArrayLike, /) -> JunctionWedges:
        """Counterclockwise wedge angles and wedge regions around every edge."""
        topology = self.topology
        points = jnp.asarray(positions)
        edges = jnp.maximum(topology.edges, 0)
        valid = (topology.edge_faces >= 0) & topology.edge_active[:, None]
        start = points[edges[:, 0]]
        axis = points[edges[:, 1]] - start
        axis_norm = _safe_norm(axis, topology.edge_active)
        unit = axis / jnp.where(topology.edge_active, axis_norm, 1.0)[:, None]
        rays = points[jnp.maximum(self.edge_opposite, 0)] - start[:, None, :]
        rays = rays - jnp.sum(rays * unit[:, None, :], axis=-1, keepdims=True) * unit[:, None, :]
        first = rays[:, 0]
        first = first / jnp.where(
            topology.edge_active, _safe_norm(first, topology.edge_active), 1.0
        )[:, None]
        second = jnp.cross(unit, first)
        phase = jnp.arctan2(
            jnp.sum(rays * second[:, None, :], axis=-1),
            jnp.sum(rays * first[:, None, :], axis=-1),
        )
        phase = jnp.where(phase < 0.0, phase + 2.0 * jnp.pi, phase)
        phase = jnp.where(valid, phase, jnp.inf)
        order = jnp.argsort(phase, axis=1)
        sorted_phase = jnp.take_along_axis(phase, order, axis=1)
        sorted_valid = jnp.take_along_axis(valid, order, axis=1)
        valence = jnp.sum(valid, axis=1)
        position = jnp.arange(topology.valence_width)[None, :]
        following = jnp.roll(sorted_phase, -1, axis=1)
        last = position == (valence[:, None] - 1)
        angles = jnp.where(
            last, sorted_phase[:, :1] + 2.0 * jnp.pi - sorted_phase, following - sorted_phase
        )
        faces = jnp.take_along_axis(jnp.maximum(topology.edge_faces, 0), order, axis=1)
        signs = jnp.take_along_axis(topology.edge_face_signs, order, axis=1)
        labels = topology.face_labels[faces]
        regions = jnp.where(signs > 0, labels[..., 1], labels[..., 0])
        return JunctionWedges(
            angles=jnp.where(sorted_valid, angles, 0.0),
            regions=jnp.where(sorted_valid, regions, -1),
            valid=sorted_valid,
            valence=valence.astype(topology.edges.dtype),
        )

    def refit_face_bvh(self, positions: ArrayLike, /) -> PackedBVH:
        """Refit the prepared face BVH to current active face bounds."""
        corners = self.face_corner_positions(positions)[: self.topology.face_count]
        return refit_packed_bvh_bounds(
            self.face_bvh, jnp.min(corners, axis=1), jnp.max(corners, axis=1)
        )


__all__ = [
    "JunctionWedges",
    "MultiRegionSurfaceGeometry",
    "PreparedMultiRegionSurface",
]
