#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, DenseLinearOperator
from ._rwg import RWGSurfaceCurrentSpace3D
from ._surface_complex import OrientedTriangleSurfaceComplex3D


class BuffaChristiansenDualEvidence3D(StrictModule, NonTrainableState):
    cross_mass_condition_number: float = eqx.field(static=True)
    minimum_cross_mass_singular_value: float = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    barycentric_area_defect: float = eqx.field(static=True)
    minimum_orientation_alignment: float = eqx.field(static=True)
    harmonic_dimension: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class BuffaChristiansenDualSpace3D(StrictModule, NonTrainableState):
    """BC edge space represented in RWGs on the barycentric refinement."""

    primal: RWGSurfaceCurrentSpace3D
    barycentric_surface: OrientedTriangleSurfaceComplex3D
    barycentric_rwg: RWGSurfaceCurrentSpace3D
    barycentric_transform: Array
    cross_mass: DenseLinearOperator
    vector_space: ArraySpace
    evidence: BuffaChristiansenDualEvidence3D
    space_id: str = eqx.field(static=True)

    @property
    def size(self) -> int:
        return self.primal.size

    def validate(self, coefficients: ArrayLike, /) -> Array:
        return self.vector_space.validate(coefficients)

    def barycentric_rwg_coefficients(self, coefficients: ArrayLike, /) -> Array:
        """Map BC coefficients to the refined RWG representation."""
        values = self.validate(coefficients)
        return contract("re,e->r", self.barycentric_transform, values)


def _barycentric_refinement(
    surface: OrientedTriangleSurfaceComplex3D, /
) -> tuple[OrientedTriangleSurfaceComplex3D, np.ndarray]:
    points = np.asarray(surface.vertices)
    faces = np.asarray(surface.triangles, dtype=np.int64)
    face_edges = np.asarray(surface.face_edges, dtype=np.int64)
    vertex_count = surface.vertex_count
    edge_count = surface.edge_count

    edge_vertices = np.asarray(surface.edge_vertices, dtype=np.int64)
    edge_midpoints = 0.5 * (points[edge_vertices[:, 0]] + points[edge_vertices[:, 1]])
    face_centroids = np.asarray(surface.face_centroids)
    refined_points = np.concatenate((points, edge_midpoints, face_centroids), axis=0)
    refined_faces = np.empty((6 * surface.face_count, 3), dtype=np.int64)
    parent_faces = np.repeat(np.arange(surface.face_count, dtype=np.int32), 6)
    for face_id, (a, b, c) in enumerate(faces):
        ab, bc, ca = vertex_count + face_edges[face_id]
        centroid = vertex_count + edge_count + face_id
        refined_faces[6 * face_id : 6 * face_id + 6] = (
            (a, ab, centroid),
            (a, centroid, ca),
            (b, bc, centroid),
            (b, centroid, ab),
            (c, ca, centroid),
            (c, centroid, bc),
        )
    return (
        OrientedTriangleSurfaceComplex3D(refined_points, refined_faces),
        parent_faces,
    )


def _oriented_vertex_links(
    surface: OrientedTriangleSurfaceComplex3D, /
) -> tuple[dict[int, tuple[int, int]], ...]:
    """Map each outgoing link vertex to its CCW face and successor."""
    faces = np.asarray(surface.triangles, dtype=np.int64)
    links: list[dict[int, tuple[int, int]]] = [{} for _ in range(surface.vertex_count)]
    for face_id, face in enumerate(faces):
        for local_id, vertex in enumerate(face):
            outgoing = int(face[(local_id + 1) % 3])
            successor = int(face[(local_id - 1) % 3])
            link = links[int(vertex)]
            if outgoing in link:
                raise ValueError("BC construction requires manifold vertex links.")
            link[outgoing] = (face_id, successor)

    for link in links:
        if not link or set(link) != {successor for _, successor in link.values()}:
            raise ValueError("BC construction requires closed oriented vertex links.")
        start = next(iter(link))
        current = start
        visited: set[int] = set()
        for _ in range(len(link)):
            if current in visited:
                raise ValueError("BC construction requires one fan around every vertex.")
            visited.add(current)
            current = link[current][1]
        if current != start or len(visited) != len(link):
            raise ValueError("BC construction requires one fan around every vertex.")
    return tuple(links)


def _bc_barycentric_transform(
    surface: OrientedTriangleSurfaceComplex3D,
    barycentric_surface: OrientedTriangleSurfaceComplex3D,
    /,
) -> np.ndarray:
    """Assemble the orientation-correct BC combination of refined RWGs."""
    links = _oriented_vertex_links(surface)
    coarse_edges = np.asarray(surface.edge_vertices, dtype=np.int64)
    coarse_face_edges = np.asarray(surface.face_edges, dtype=np.int64)
    coarse_face_signs = np.asarray(surface.face_edge_signs)
    coarse_edge_ids = {
        (int(start), int(stop)): edge_id
        for edge_id, (start, stop) in enumerate(coarse_edges)
    }
    refined_edges = np.asarray(barycentric_surface.edge_vertices, dtype=np.int64)
    refined_edge_ids = {
        (int(start), int(stop)): edge_id
        for edge_id, (start, stop) in enumerate(refined_edges)
    }
    refined_edge_lengths = np.asarray(barycentric_surface.edge_lengths, dtype=float)
    transform = np.zeros(
        (barycentric_surface.edge_count, surface.edge_count), dtype=float
    )
    vertex_count = surface.vertex_count
    edge_count = surface.edge_count

    def add_oriented(
        coarse_edge_id: int, start: int, stop: int, coefficient: float
    ) -> None:
        key = (min(start, stop), max(start, stop))
        refined_edge_id = refined_edge_ids[key]
        orientation = 1.0 if (start, stop) == key else -1.0
        # A topological BC weight is an integrated refined-edge DOF, whereas
        # the adopted RWG has unit pointwise co-normal trace on its edge.
        transform[refined_edge_id, coarse_edge_id] += (
            orientation * coefficient / refined_edge_lengths[refined_edge_id]
        )

    for coarse_edge_id, (first, second) in enumerate(coarse_edges):
        # RWGs on edges incident to a pole are oriented so that their vector
        # traces circulate counter-clockwise. The two endpoint patches carry
        # opposite signs, as in the BC edge construction.
        for pole, neighbor, pole_sign in (
            (int(first), int(second), -1.0),
            (int(second), int(first), 1.0),
        ):
            link = links[pole]
            valence = len(link)
            current = neighbor
            sequence: list[tuple[int, int]] = []
            for _ in range(valence):
                face_id, successor = link[current]
                sequence.append((pole, vertex_count + edge_count + face_id))
                if successor != neighbor:
                    successor_edge = coarse_edge_ids[
                        (min(pole, successor), max(pole, successor))
                    ]
                    sequence.append((pole, vertex_count + successor_edge))
                current = successor
            if current != neighbor or len(sequence) != 2 * valence - 1:
                raise ValueError("BC endpoint support has an invalid oriented fan.")
            for index, (pole_vertex, barycentric_vertex) in enumerate(sequence, start=2):
                coefficient = pole_sign * (valence + 1 - index) / (2.0 * valence)
                # For the adopted RWG convention, an edge directed into the
                # pole has counter-clockwise vector trace around that pole.
                add_oriented(
                    coarse_edge_id,
                    barycentric_vertex,
                    pole_vertex,
                    coefficient,
                )

        # The two refined dual-edge pieces crossing the coarse edge carry
        # equal 1/2 weights and the physical direction first -> second.
        midpoint = vertex_count + coarse_edge_id
        for face_id, local_id in np.argwhere(coarse_face_edges == coarse_edge_id):
            centroid = vertex_count + edge_count + int(face_id)
            if coarse_face_signs[face_id, local_id] > 0.0:
                add_oriented(coarse_edge_id, midpoint, centroid, 0.5)
            else:
                add_oriented(coarse_edge_id, centroid, midpoint, 0.5)
    if np.any(~np.isfinite(transform)) or np.linalg.matrix_rank(transform) != edge_count:
        raise ValueError("BC barycentric coefficient map is nonfinite or rank deficient.")
    return transform


def _rwg_bc_cross_mass(
    primal: RWGSurfaceCurrentSpace3D,
    barycentric_rwg: RWGSurfaceCurrentSpace3D,
    parent_faces: np.ndarray,
    transform: np.ndarray,
    /,
) -> np.ndarray:
    """Integrate the Maxwell antisymmetric RWG/BC duality exactly."""
    coarse = primal.surface
    refined = barycentric_rwg.surface
    coarse_edges = np.asarray(coarse.face_edges, dtype=np.int64)
    refined_edges = np.asarray(refined.face_edges, dtype=np.int64)
    coarse_opposite = np.asarray(coarse.opposite_vertices, dtype=np.int64)
    refined_opposite = np.asarray(refined.opposite_vertices, dtype=np.int64)
    coarse_points = np.asarray(coarse.vertices, dtype=float)
    refined_points = np.asarray(refined.vertices, dtype=float)
    coarse_scale = (
        np.asarray(coarse.face_edge_signs, dtype=float)
        * np.asarray(coarse.edge_lengths, dtype=float)[coarse_edges]
        / (2.0 * np.asarray(coarse.face_areas, dtype=float)[:, None])
    )
    refined_scale = (
        np.asarray(refined.face_edge_signs, dtype=float)
        * np.asarray(refined.edge_lengths, dtype=float)[refined_edges]
        / (2.0 * np.asarray(refined.face_areas, dtype=float)[:, None])
    )
    quadrature = np.asarray(
        (
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        )
    )
    refined_faces = np.asarray(refined.triangles, dtype=np.int64)
    refined_areas = np.asarray(refined.face_areas, dtype=float)
    refined_normals = np.asarray(refined.face_normals, dtype=float)
    cross_mass = np.zeros((primal.size, primal.size), dtype=float)
    for refined_face_id, refined_face in enumerate(refined_faces):
        coarse_face_id = int(parent_faces[refined_face_id])
        points = quadrature @ refined_points[refined_face]
        coarse_local_edges = coarse_edges[coarse_face_id]
        refined_local_edges = refined_edges[refined_face_id]
        transformed_local = transform[refined_local_edges]
        for point in points:
            coarse_values = coarse_scale[coarse_face_id, :, None] * (
                point - coarse_points[coarse_opposite[coarse_face_id]]
            )
            refined_values = refined_scale[refined_face_id, :, None] * (
                point - refined_points[refined_opposite[refined_face_id]]
            )
            rotated_refined = np.cross(refined_normals[refined_face_id], refined_values)
            local_pairing = coarse_values @ rotated_refined.T
            cross_mass[coarse_local_edges] += (
                refined_areas[refined_face_id] / 3.0 * (local_pairing @ transformed_local)
            )
    return cross_mass


def prepare_buffa_christiansen_dual_3d(
    primal: RWGSurfaceCurrentSpace3D, /, *, maximum_condition_number: float = 1e10
) -> BuffaChristiansenDualSpace3D:
    """Prepare the genuine barycentrically refined BC dual of an RWG space."""
    if not isinstance(primal, RWGSurfaceCurrentSpace3D):
        raise TypeError("primal must be RWGSurfaceCurrentSpace3D.")
    condition_limit = float(maximum_condition_number)
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("maximum_condition_number must be finite and exceed one.")
    surface = primal.surface
    if (
        not surface.topology_report.closed
        or not surface.topology_report.consistently_oriented
    ):
        raise ValueError(
            "BC construction requires a closed consistently oriented surface."
        )

    barycentric_surface, parent_faces = _barycentric_refinement(surface)
    barycentric_rwg = RWGSurfaceCurrentSpace3D(
        barycentric_surface, coefficient_dtype=primal.vector_space.dtype
    )
    parent_normals = np.asarray(surface.face_normals, dtype=float)[parent_faces]
    refined_normals = np.asarray(barycentric_surface.face_normals, dtype=float)
    orientation_alignment = np.sum(parent_normals * refined_normals, axis=1)
    minimum_alignment = float(np.min(orientation_alignment))
    parent_areas = np.asarray(surface.face_areas, dtype=float)
    refined_areas = np.asarray(barycentric_surface.face_areas, dtype=float).reshape(
        (surface.face_count, 6)
    )
    area_defect = float(
        np.max(np.abs(np.sum(refined_areas, axis=1) - parent_areas) / parent_areas)
    )
    geometry_tolerance = 100.0 * np.finfo(np.asarray(surface.vertices).dtype).eps
    if (
        np.any(~np.isfinite(orientation_alignment))
        or minimum_alignment < 1.0 - geometry_tolerance
        or not np.isfinite(area_defect)
        or area_defect > geometry_tolerance
    ):
        raise ValueError("Barycentric refinement failed orientation or area validation.")

    transform = _bc_barycentric_transform(surface, barycentric_surface)
    storage_dtype = np.asarray(surface.vertices).dtype
    stored_transform = np.asarray(transform, dtype=storage_dtype)
    if (
        np.any(~np.isfinite(stored_transform))
        or np.linalg.matrix_rank(stored_transform) != primal.size
    ):
        raise ValueError("Stored BC coefficient map is nonfinite or rank deficient.")
    cross = _rwg_bc_cross_mass(
        primal,
        barycentric_rwg,
        parent_faces,
        np.asarray(stored_transform, dtype=float),
    )
    stored_cross = np.asarray(cross, dtype=storage_dtype)
    if np.any(~np.isfinite(stored_cross)):
        raise ValueError("RWG/BC cross mass is nonfinite.")
    singular = np.linalg.svd(np.asarray(stored_cross, dtype=float), compute_uv=False)
    minimum_singular = float(singular[-1])
    condition = (
        float(singular[0] / minimum_singular) if minimum_singular > 0.0 else float("inf")
    )
    if (
        minimum_singular <= 0.0
        or not np.isfinite(condition)
        or condition > condition_limit
    ):
        raise ValueError("RWG/BC cross mass exceeds the declared condition envelope.")
    evidence_id = canonical_fingerprint(
        {
            "kind": "buffa-christiansen-dual-evidence-3d-v2",
            "primal": primal.space_id,
            "barycentric_surface": barycentric_surface.complex_id,
            "transform": array_tree_fingerprint(stored_transform),
            "cross_mass": array_tree_fingerprint(stored_cross),
            "condition": condition,
            "minimum_singular_value": minimum_singular,
            "maximum_condition_number": condition_limit,
            "barycentric_area_defect": area_defect,
            "minimum_orientation_alignment": minimum_alignment,
            "harmonic_dimension": surface.topology_report.harmonic_dimension,
        }
    )
    evidence = BuffaChristiansenDualEvidence3D(
        condition,
        minimum_singular,
        condition_limit,
        area_defect,
        minimum_alignment,
        surface.topology_report.harmonic_dimension,
        evidence_id,
    )
    space_id = canonical_fingerprint(
        {
            "kind": "buffa-christiansen-dual-space-3d-v2",
            "primal": primal.space_id,
            "barycentric_rwg": barycentric_rwg.space_id,
            "evidence": evidence_id,
        }
    )
    vector_space = ArraySpace(
        (primal.size,), dtype=primal.vector_space.dtype, space_id=space_id
    )
    operator = DenseLinearOperator(
        jnp.asarray(stored_cross),
        source=vector_space,
        target=primal.vector_space,
        operator_id=f"{evidence_id}:cross-mass",
    )
    return BuffaChristiansenDualSpace3D(
        primal,
        barycentric_surface,
        barycentric_rwg,
        jnp.asarray(stored_transform),
        operator,
        vector_space,
        evidence,
        space_id,
    )


__all__ = [
    "BuffaChristiansenDualEvidence3D",
    "BuffaChristiansenDualSpace3D",
    "prepare_buffa_christiansen_dual_3d",
]
