# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded point location in the represented planar-face polyhedral complex.

Polyhedral FV fields use physical Cartesian charts: the reference domain of a
cell is its represented polyhedron, and its chart map is the identity. This is
not an affine finite-element surrogate. A certified star fan supplies exact
closed-tetrahedron containment; the returned cell rows refer to the original
polyhedral cells, never the fan's private tetrahedra. Host preparation/queries
use native exact predicates, traced queries retain filtered uncertainty.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

from .._bvh import PackedBVH, point_select_leaf_items, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import orient3d, PredicateMode
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._contracts import CompiledGeometry
from ._cell_complex import PolyhedralConnectivity
from ._cell_geometry import CellVertexGeometryElement
from ._cell_mesh import CellMesh
from ._simplicial_locator import (
    AbstractCellLocator,
    CellLocationResult,
    CellLocationStatus,
    SimplicialLocationPolicy,
)
from .fem._cell_map import FiniteElementCellMapEvaluation


def _fan_membership(
    points: Array, tetrahedra: Array, valid: Array, /
) -> tuple[Array, Array]:
    """Membership/resolution for (points, candidates, fan, 4, 3) exact domains."""
    query = jnp.broadcast_to(points[:, None, None, :], tetrahedra.shape[:-2] + (3,))
    a, b, c, d = (tetrahedra[..., index, :] for index in range(4))
    mode = (
        PredicateMode.FILTERED_DEVICE
        if isinstance(points, Tracer)
        else PredicateMode.EXACT
    )
    outcomes = (
        orient3d(query, b, c, d, mode=mode),
        orient3d(a, query, c, d, mode=mode),
        orient3d(a, b, query, d, mode=mode),
        orient3d(a, b, c, query, mode=mode),
    )
    signs = jnp.stack(tuple(jnp.asarray(item.signs) for item in outcomes), axis=-1)
    certain = jnp.stack(tuple(jnp.asarray(item.certain) for item in outcomes), axis=-1)
    inside_tet = valid & jnp.all(certain & (signs >= 0), axis=-1)
    outside_tet = (~valid) | jnp.any(certain & (signs < 0), axis=-1)
    inside = jnp.any(inside_tet, axis=-1)
    # An uncertain private fan boundary does not invalidate another fully
    # established fan witness in the same physical cell.
    resolved = inside | jnp.all(outside_tet, axis=-1)
    return inside, resolved


@final
class PreparedPolyhedralCellMap(StrictModule, NonTrainableState):
    """Physical Cartesian chart of every original cell of one polyhedral mesh."""

    mesh: CellMesh
    coordinates: Array
    tetrahedra: Array
    tetrahedron_valid: Array
    cell_lower: Array
    cell_upper: Array
    cell_volumes: Array
    coordinate_element: CellVertexGeometryElement
    coordinate_dofs: Array
    cell_count: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    reference_dimension: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    cell_map_id: str = eqx.field(static=True)
    block_name: str = eqx.field(static=True)
    block_names: tuple[str, ...] = eqx.field(static=True)
    maximum_workset_entries: int = eqx.field(static=True)

    def __init__(
        self, mesh: CellMesh, /, *, maximum_workset_entries: int = 100_000_000
    ) -> None:
        if not isinstance(mesh, CellMesh) or not isinstance(
            mesh.connectivity, PolyhedralConnectivity
        ):
            raise TypeError(
                "Polyhedral Cartesian charts require canonical polyhedral connectivity."
            )
        mesh.require_dense("polyhedral point location")
        if mesh.ambient_dimension != 3 or mesh.topological_dimension != 3:
            raise ValueError(
                "Polyhedral Cartesian charts are full-dimensional 3D regions."
            )
        if (
            isinstance(maximum_workset_entries, bool)
            or not isinstance(maximum_workset_entries, int)
            or maximum_workset_entries < 1
        ):
            raise ValueError("maximum_workset_entries must be a positive integer.")
        connectivity = mesh.connectivity
        points = np.asarray(mesh.coordinates, dtype=np.float64)
        face_offsets = np.asarray(connectivity.face_vertex_offsets)
        face_values = np.asarray(connectivity.face_vertex_values)
        cell_offsets = np.asarray(connectivity.cell_face_offsets)
        cell_faces = np.asarray(connectivity.cell_face_values)
        cell_signs = np.asarray(connectivity.cell_face_sign_values)
        fan_counts = np.asarray(
            [
                sum(
                    int(face_offsets[face + 1] - face_offsets[face] - 2)
                    for face in cell_faces[cell_offsets[cell] : cell_offsets[cell + 1]]
                )
                for cell in range(connectivity.cell_count)
            ],
            dtype=np.int64,
        )
        maximum_fan = int(np.max(fan_counts))
        entries = connectivity.cell_count * maximum_fan * 4
        if entries > maximum_workset_entries:
            raise ValueError(
                "Polyhedral star-fan workset exceeds maximum_workset_entries."
            )
        from .finite_volume._polyhedral import prepare_polyhedral_finite_volume_geometry

        geometry = prepare_polyhedral_finite_volume_geometry(
            mesh, maximum_workset_entries=maximum_workset_entries
        )
        tetrahedra = np.zeros(
            (connectivity.cell_count, maximum_fan, 4, 3), dtype=np.float64
        )
        valid = np.zeros((connectivity.cell_count, maximum_fan), dtype=np.bool_)
        lower, upper, vertices = [], [], []
        for cell in range(connectivity.cell_count):
            faces = cell_faces[cell_offsets[cell] : cell_offsets[cell + 1]]
            signs = cell_signs[cell_offsets[cell] : cell_offsets[cell + 1]]
            rows = np.unique(
                np.concatenate(
                    [
                        face_values[face_offsets[face] : face_offsets[face + 1]]
                        for face in faces
                    ]
                )
            )
            vertices.append(rows)
            lower.append(np.min(points[rows], axis=0))
            upper.append(np.max(points[rows], axis=0))
            star = np.mean(points[rows], axis=0)
            slot = 0
            for face, sign in zip(faces, signs, strict=True):
                polygon = face_values[face_offsets[face] : face_offsets[face + 1]]
                for local in range(1, polygon.size - 1):
                    triangle = polygon[[0, local, local + 1]]
                    if sign < 0:
                        triangle = triangle[[0, 2, 1]]
                    tetrahedra[cell, slot] = np.concatenate(
                        (star[None, :], points[triangle]), axis=0
                    )
                    valid[cell, slot] = True
                    slot += 1
        selected = tetrahedra[valid]
        decision = orient3d(
            *(selected[:, index] for index in range(4)), mode=PredicateMode.EXACT
        )
        if not np.all(np.asarray(decision.certain)) or np.any(
            np.asarray(decision.signs) <= 0
        ):
            raise ValueError(
                "Rounded polyhedral star construction is not an exact positive fan."
            )
        width = max(len(rows) for rows in vertices)
        routes = np.asarray(
            [
                np.pad(rows, (0, width - rows.size), constant_values=int(rows[0]))
                for rows in vertices
            ],
            dtype=np.int32,
        )
        self.mesh, self.coordinates = mesh, mesh.coordinates
        self.tetrahedra, self.tetrahedron_valid = (
            jnp.asarray(tetrahedra),
            jnp.asarray(valid),
        )
        self.cell_lower, self.cell_upper = jnp.asarray(lower), jnp.asarray(upper)
        self.cell_volumes = geometry.cell_volumes
        self.coordinate_element = CellVertexGeometryElement("polyhedron", width)
        self.coordinate_dofs = jnp.asarray(routes)
        self.cell_count = connectivity.cell_count
        self.coordinate_count = points.shape[0]
        self.ambient_dimension = self.reference_dimension = 3
        self.topology_id = mesh.topology_id
        self.block_names = tuple(block.name for block in mesh.blocks)
        self.block_name = (
            self.block_names[0]
            if len(self.block_names) == 1
            else "polyhedral-cartesian-domain"
        )
        self.geometry_layout_id = canonical_fingerprint(
            {"kind": "polyhedral-cartesian-layout", "topology": mesh.topology_id}
        )
        self.cell_map_id = canonical_fingerprint(
            {
                "kind": "polyhedral-cartesian-chart",
                "mesh": mesh.mesh_id,
                "fan": array_tree_fingerprint((tetrahedra, valid)),
            }
        )
        self.maximum_workset_entries = maximum_workset_entries

    def evaluate(
        self,
        coordinates: ArrayLike,
        cell_indices: ArrayLike,
        reference_points: ArrayLike,
        /,
    ) -> FiniteElementCellMapEvaluation:
        values, indices, points = (
            jnp.asarray(coordinates),
            jnp.asarray(cell_indices),
            jnp.asarray(reference_points),
        )
        if (
            values.shape != self.coordinates.shape
            or indices.ndim != 1
            or points.shape != (indices.size, 3)
        ):
            raise ValueError(
                "Cartesian chart coordinates/cell routes/reference points have incompatible shapes."
            )
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("Cartesian chart cell indices must be integers.")
        safe = jnp.clip(indices, 0, self.cell_count - 1)
        inside, resolved = _fan_membership(
            points, self.tetrahedra[safe, None], self.tetrahedron_valid[safe, None]
        )
        active = (
            (indices >= 0)
            & (indices < self.cell_count)
            & inside[:, 0]
            & resolved[:, 0]
            & jnp.all(values == self.coordinates)
            & jnp.all(jnp.isfinite(points), axis=1)
        )
        identity = jnp.broadcast_to(jnp.eye(3, dtype=points.dtype), (indices.size, 3, 3))
        ones = jnp.ones((indices.size,), dtype=points.dtype)
        return FiniteElementCellMapEvaluation(
            points,
            identity,
            identity,
            ones,
            ones,
            ones,
            jnp.where(active, ones, -ones),
            active,
        )

    def support_geometry(self, support_id: str, /) -> CompiledGeometry:
        """Compile all exact represented physical boundary facets of the source."""
        from ..geometry.simplicial import MeshRegion

        connectivity = self.mesh.connectivity
        if not isinstance(connectivity, PolyhedralConnectivity):
            raise TypeError(
                "Polyhedral support requires canonical polyhedral connectivity."
            )
        face_offsets = np.asarray(connectivity.face_vertex_offsets)
        face_values = np.asarray(connectivity.face_vertex_values)
        cell_offsets = np.asarray(connectivity.cell_face_offsets)
        cell_faces = np.asarray(connectivity.cell_face_values)
        cell_signs = np.asarray(connectivity.cell_face_sign_values)
        neighbor, owner = (
            np.asarray(connectivity.face_neighbor),
            np.asarray(connectivity.face_owner),
        )
        triangles = []
        for face in np.flatnonzero(neighbor < 0):
            cell = owner[face]
            rows = cell_faces[cell_offsets[cell] : cell_offsets[cell + 1]]
            signs = cell_signs[cell_offsets[cell] : cell_offsets[cell + 1]]
            sign = signs[np.flatnonzero(rows == face)[0]]
            polygon = face_values[face_offsets[face] : face_offsets[face + 1]]
            for local in range(1, polygon.size - 1):
                triangle = polygon[[0, local, local + 1]]
                triangles.append(triangle if sign > 0 else triangle[[0, 2, 1]])
        if not triangles:
            raise ValueError(
                "The represented polyhedral domain has no physical closed boundary."
            )
        faces = np.asarray(triangles, dtype=np.int32)
        used, compact = np.unique(faces, return_inverse=True)
        return MeshRegion(
            self.coordinates[jnp.asarray(used)],
            compact.reshape(faces.shape).astype(np.int32),
            feature_id=f"polyhedral-field-support:{support_id}",
        ).compile()


@final
class PreparedPolyhedralCellLocator(AbstractCellLocator, NonTrainableState):
    """Exhaustive bounded BVH location on original planar polyhedral cell rows."""

    cell_map: PreparedPolyhedralCellMap
    coordinates: Array
    cell_lower: Array
    cell_upper: Array
    bvh: PackedBVH
    policy: SimplicialLocationPolicy
    locator_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        policy: SimplicialLocationPolicy | None = None,
        /,
        *,
        maximum_workset_entries: int = 100_000_000,
    ) -> None:
        cell_map = PreparedPolyhedralCellMap(
            mesh, maximum_workset_entries=maximum_workset_entries
        )
        policy_ = (
            SimplicialLocationPolicy(min(cell_map.cell_count, 64), 1, 1)
            if policy is None
            else policy
        )
        if not isinstance(policy_, SimplicialLocationPolicy):
            raise TypeError("policy must be SimplicialLocationPolicy or None.")
        self.cell_map, self.coordinates = cell_map, cell_map.coordinates
        self.cell_lower, self.cell_upper = cell_map.cell_lower, cell_map.cell_upper
        self.bvh = prepare_bvh(self.cell_lower, self.cell_upper, dtype=jnp.float64)
        self.policy = policy_
        self.locator_id = canonical_fingerprint(
            {
                "kind": "polyhedral-cell-locator",
                "map": cell_map.cell_map_id,
                "policy": policy_.policy_id,
            }
        )

    def support_geometry(self, support_id: str, /) -> CompiledGeometry:
        return self.cell_map.support_geometry(support_id)

    def locate(
        self, points: ArrayLike, /, *, cell_mask: ArrayLike | None = None
    ) -> CellLocationResult:
        values = jnp.asarray(points, dtype=self.coordinates.dtype)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("Polyhedral locator points must have shape (points, 3).")
        capacity = min(self.policy.maximum_candidates, self.cell_map.cell_count)
        if (
            values.shape[0] * capacity * self.cell_map.tetrahedra.shape[1]
            > self.cell_map.maximum_workset_entries
        ):
            raise ValueError("Polyhedral query fan exceeds maximum_workset_entries.")
        candidates, valid, search_complete = point_select_leaf_items(
            values,
            bvh=self.bvh,
            maximum_candidates=capacity,
            tolerance=0.0,
        )
        if cell_mask is not None:
            mask = jnp.asarray(cell_mask)
            if mask.shape != (self.cell_map.cell_count,) or mask.dtype != jnp.bool_:
                raise ValueError(
                    "cell_mask must be one boolean per original polyhedral cell."
                )
            valid &= mask[jnp.where(valid, candidates, 0)]
        safe = jnp.where(valid, candidates, 0)
        inside, resolved = _fan_membership(
            values, self.cell_map.tetrahedra[safe], self.cell_map.tetrahedron_valid[safe]
        )
        finite = jnp.all(jnp.isfinite(values), axis=1)
        accepted = valid & inside & finite[:, None]
        complete = search_complete & jnp.all(~valid | resolved, axis=1)
        chosen = jnp.min(
            jnp.where(accepted, candidates, self.cell_map.cell_count), axis=1
        )
        found = chosen < self.cell_map.cell_count
        status = jnp.select(
            (~finite, ~search_complete, ~complete, found),
            (
                int(CellLocationStatus.NONFINITE),
                int(CellLocationStatus.RESOURCE_EXCEEDED),
                int(CellLocationStatus.INVERSE_MAP_EXHAUSTED),
                int(CellLocationStatus.LOCATED),
            ),
            default=int(CellLocationStatus.OUTSIDE),
        ).astype(jnp.int32)
        zeros = jnp.zeros((values.shape[0],), dtype=values.dtype)
        reference = jnp.where(found[:, None], values, 0.0)
        return CellLocationResult(
            jnp.where(found, chosen, -1),
            reference,
            jnp.empty((values.shape[0], 0), dtype=values.dtype),
            jnp.where(found, zeros, jnp.inf),
            jnp.zeros(values.shape[0], dtype=jnp.int32),
            jnp.where(found, 1.0, jnp.inf),
            found,
            jnp.zeros(values.shape[0], dtype=jnp.bool_),
            jnp.sum(accepted, axis=1, dtype=jnp.int32),
            status,
            found & complete & finite,
            jnp.where(accepted, candidates, -1),
            jnp.where(accepted[:, :, None], values[:, None, :], 0.0),
            complete,
            self.locator_id,
        )


__all__ = ["PreparedPolyhedralCellMap", "PreparedPolyhedralCellLocator"]
