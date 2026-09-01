#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation
from .._topology import CellComplexTopology, EntitySet, OrientedIncidence


class SurfaceTopologyReport3D(StrictModule, NonTrainableState):
    """Exact finite-complex topology evidence, not continuum geometry certification."""

    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    resource_evidence: str = eqx.field(static=True)
    error_evidence: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    euler_characteristic: int = eqx.field(static=True)
    genus: int = eqx.field(static=True)
    harmonic_dimension: int = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    consistently_oriented: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class OrientedTriangleSurfaceComplex3D(StrictModule, NonTrainableState):
    """Finite oriented closed triangular 2-complex embedded in three dimensions."""

    vertices: Array
    triangles: Array
    edge_vertices: Array
    face_edges: Array
    face_edge_signs: Array
    opposite_vertices: Array
    face_centroids: Array
    face_normals: Array
    face_areas: Array
    edge_lengths: Array
    face_component_ids: Array
    topology: CellComplexTopology
    topology_report: SurfaceTopologyReport3D
    complex_id: str = eqx.field(static=True)

    def __init__(self, vertices: ArrayLike, triangles: ArrayLike, /):
        points = np.asarray(vertices)
        faces = np.asarray(triangles)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
            raise ValueError(
                "vertices must have shape (vertex_count, 3), with at least four vertices."
            )
        if not np.issubdtype(points.dtype, np.floating):
            points = points.astype(float)
        points = points.astype(
            np.dtype(jax.dtypes.canonicalize_dtype(points.dtype)), copy=False
        )
        if not np.all(np.isfinite(points)):
            raise ValueError("Surface vertices must be finite.")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
        ):
            raise TypeError(
                "triangles must be one integer array of shape (face_count, 3)."
            )
        faces = faces.astype(np.int64, copy=False)
        if faces.shape[0] < 4 or np.any(faces < 0) or np.any(faces >= points.shape[0]):
            raise ValueError(
                "Triangle indices are out of bounds or the surface has fewer than four faces."
            )
        if np.any(np.sort(faces, axis=1)[:, 1:] == np.sort(faces, axis=1)[:, :-1]):
            raise ValueError("Every triangle must contain three distinct vertices.")
        if np.unique(faces).size != points.shape[0]:
            raise ValueError("Every declared vertex must belong to the surface.")

        edge_lookup: dict[tuple[int, int], int] = {}
        edge_rows: list[tuple[int, int]] = []
        face_edges = np.empty((faces.shape[0], 3), dtype=np.int32)
        face_signs = np.empty((faces.shape[0], 3), dtype=np.int8)
        opposite = np.empty((faces.shape[0], 3), dtype=np.int32)
        uses: list[list[tuple[int, int]]] = []
        for face_id, (a, b, c) in enumerate(faces.tolist()):
            directed = ((a, b, c), (b, c, a), (c, a, b))
            for local_id, (start, stop, other) in enumerate(directed):
                key = (min(start, stop), max(start, stop))
                edge_id = edge_lookup.get(key)
                if edge_id is None:
                    edge_id = len(edge_rows)
                    edge_lookup[key] = edge_id
                    edge_rows.append(key)
                    uses.append([])
                sign = 1 if (start, stop) == key else -1
                face_edges[face_id, local_id] = edge_id
                face_signs[face_id, local_id] = sign
                opposite[face_id, local_id] = other
                uses[edge_id].append((face_id, sign))
        if any(len(edge_uses) != 2 for edge_uses in uses):
            raise ValueError(
                "RWG surfaces must be closed: every edge must have exactly two incident triangles."
            )
        if any(edge_uses[0][1] + edge_uses[1][1] != 0 for edge_uses in uses):
            raise ValueError("Triangle orientations disagree across a shared edge.")

        edges = np.asarray(edge_rows, dtype=np.int64)
        first = points[faces[:, 0]]
        second = points[faces[:, 1]]
        third = points[faces[:, 2]]
        cross = np.cross(second - first, third - first)
        doubled_area = np.linalg.norm(cross, axis=1)
        if np.any(~np.isfinite(doubled_area)) or np.any(doubled_area <= 0.0):
            raise ValueError("Surface triangles must be finite and nondegenerate.")
        areas = 0.5 * doubled_area
        normals = cross / doubled_area[:, None]
        centroids = (first + second + third) / 3.0
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)

        parent = np.arange(faces.shape[0])

        def root(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = int(parent[value])
            return value

        for edge_uses in uses:
            left, right = edge_uses[0][0], edge_uses[1][0]
            left_root, right_root = root(left), root(right)
            if left_root != right_root:
                parent[right_root] = left_root
        roots = [root(index) for index in range(faces.shape[0])]
        root_ids = {value: index for index, value in enumerate(sorted(set(roots)))}
        components = np.asarray([root_ids[value] for value in roots], dtype=np.int32)
        component_count = len(root_ids)
        chi = int(points.shape[0] - edges.shape[0] + faces.shape[0])
        genus_numerator = 2 * component_count - chi
        if genus_numerator < 0 or genus_numerator % 2:
            raise ValueError(
                "Closed oriented surface has an invalid Euler characteristic."
            )
        genus = genus_numerator // 2

        vertex_entities = EntitySet("surface_vertices", 0, np.arange(points.shape[0]))
        edge_entities = EntitySet("surface_edges", 1, np.arange(edges.shape[0]))
        face_entities = EntitySet("surface_triangles", 2, np.arange(faces.shape[0]))
        vertex_edge_relation = EdgeRelation(
            edges.reshape(-1),
            np.repeat(np.arange(edges.shape[0]), 2),
            source_size=points.shape[0],
            target_size=edges.shape[0],
        )
        vertex_edge = OrientedIncidence(
            1,
            vertex_entities,
            edge_entities,
            vertex_edge_relation,
            np.tile(np.asarray((-1, 1)), edges.shape[0]),
        )
        edge_face_relation = EdgeRelation(
            face_edges.reshape(-1),
            np.repeat(np.arange(faces.shape[0]), 3),
            source_size=edges.shape[0],
            target_size=faces.shape[0],
        )
        edge_face = OrientedIncidence(
            2,
            edge_entities,
            face_entities,
            edge_face_relation,
            face_signs.reshape(-1),
        )
        topology = CellComplexTopology(
            (vertex_entities, edge_entities, face_entities),
            (vertex_edge, edge_face),
        )
        complex_id = canonical_fingerprint(
            {
                "kind": "oriented-triangle-surface-complex-3d-v1",
                "vertices": array_tree_fingerprint(points),
                "triangles": array_tree_fingerprint(faces),
            }
        )
        report_id = canonical_fingerprint(
            {
                "kind": "surface-topology-report-3d-v1",
                "complex": complex_id,
                "components": component_count,
                "euler_characteristic": chi,
                "genus": genus,
            }
        )
        report = SurfaceTopologyReport3D(
            ambient_dimension=3,
            pde="topological substrate for time-harmonic Maxwell boundary currents",
            geometry="finite oriented closed triangular 2-complex embedded in R^3",
            formulation="signed cellular boundary incidence and Euler characteristic",
            provider="phydrax.discretization.bem",
            precision=str(points.dtype),
            resource_evidence=f"V={points.shape[0]}, E={edges.shape[0]}, F={faces.shape[0]}",
            error_evidence="exact integer incidence and Euler arithmetic; floating geometry only checks nondegeneracy",
            non_goals=(
                "continuum topology certification",
                "nonmanifold or open surfaces",
                "curved geometry",
            ),
            component_count=component_count,
            euler_characteristic=chi,
            genus=genus,
            harmonic_dimension=2 * genus,
            closed=True,
            consistently_oriented=True,
            report_id=report_id,
        )
        self.vertices = jnp.asarray(points)
        self.triangles = jnp.asarray(faces, dtype=jnp.int32)
        self.edge_vertices = jnp.asarray(edges, dtype=jnp.int32)
        self.face_edges = jnp.asarray(face_edges, dtype=jnp.int32)
        self.face_edge_signs = jnp.asarray(face_signs, dtype=self.vertices.dtype)
        self.opposite_vertices = jnp.asarray(opposite, dtype=jnp.int32)
        self.face_centroids = jnp.asarray(centroids)
        self.face_normals = jnp.asarray(normals)
        self.face_areas = jnp.asarray(areas)
        self.edge_lengths = jnp.asarray(lengths)
        self.face_component_ids = jnp.asarray(components)
        self.topology = topology
        self.topology_report = report
        self.complex_id = complex_id

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.edge_vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.triangles.shape[0])
