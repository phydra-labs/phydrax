"""Typed CAD fixtures and independent curving-error references."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import factorial, prod

import numpy as np
from scipy.spatial import Delaunay

import phydrax as phx
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.geometry.brep._model import BRepOccurrence
from phydrax.geometry.brep._query_projection import NativeBRepProjection
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
)
from phydrax.meshing._curving import HighOrderCurvingResult
from phydrax.meshing._lineage_records import EntityLineage, EntityLineageKind, MeshLineage
from tests._support.cad_models import native_repeated_assembly


CURVING_COORDINATE_CONTRACT = phx.SpatialCoordinateContract.si()
# The pre-refactor CAD fixtures requested this classification error budget.
CURVING_ASSOCIATION_POLICY = phx.meshing.AssociationPropagationPolicy(
    classification_tolerance=1e-8
)
CURVING_STATUS = phx.meshing.HighOrderCurvingStatus


def plate_projection() -> NativeBRepProjection:
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(((-2, -2), (2, -2), (2, 2), (-2, 2))),
        (phx.geometry.ProfileLoop.circle((0, 0), 1.0),),
    )
    model = phx.geometry.brep_planar_face(
        profile, coordinate_contract=CURVING_COORDINATE_CONTRACT
    )
    embedding = phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    return phx.geometry.prepare_brep_projection(model, embedding=embedding)


def sphere_projection() -> NativeBRepProjection:
    return phx.geometry.prepare_brep_projection(
        phx.geometry.brep_sphere(1.0, coordinate_contract=CURVING_COORDINATE_CONTRACT)
    )


def plate_mesh(apex: tuple[float, float]) -> CellMesh:
    """Square plate with a unit hole: the hole is meshed by four chords."""
    ring = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
    outer = np.asarray(
        [[x, y] for x in (-2, 0, 2) for y in (-2, 0, 2) if (x, y) != (0, 0)],
        dtype=np.float64,
    )
    points = np.concatenate((ring, outer, np.asarray([apex], dtype=np.float64)))
    triangles = Delaunay(points).simplices
    triangles = triangles[np.abs(points[triangles].mean(axis=1)).sum(axis=1) > 1.0]
    first = points[triangles[:, 1]] - points[triangles[:, 0]]
    second = points[triangles[:, 2]] - points[triangles[:, 0]]
    area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    triangles[area < 0] = triangles[area < 0][:, [0, 2, 1]]
    return phx.discretization.CellMesh.from_triangles(points, triangles)


def icosphere(level: int) -> CellMesh:
    golden = (1.0 + 5.0**0.5) / 2.0
    points = np.asarray(
        [
            [-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0],
            [0, -1, golden], [0, 1, golden], [0, -1, -golden], [0, 1, -golden],
            [golden, 0, -1], [golden, 0, 1], [-golden, 0, -1], [-golden, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    faces = np.asarray(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
            [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
            [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
            [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )  # fmt: skip
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    for _ in range(level):
        edges = np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
        )
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        unique = np.asarray(unique, dtype=np.int64)
        inverse = np.asarray(inverse, dtype=np.int64)
        middle = points[unique].mean(axis=1)
        middle /= np.linalg.norm(middle, axis=1, keepdims=True)
        ab, bc, ca = points.shape[0] + inverse.reshape(3, -1)
        points = np.concatenate((points, middle))
        a, b, c = faces.T
        faces = np.concatenate(
            (
                np.stack((a, ab, ca), axis=1),
                np.stack((b, bc, ab), axis=1),
                np.stack((c, ca, bc), axis=1),
                np.stack((ab, bc, ca), axis=1),
            )
        )
    # This rotation keeps fixture vertices off the sphere seam and poles.
    turn = np.asarray(
        [[np.cos(0.3), -np.sin(0.3), 0], [np.sin(0.3), np.cos(0.3), 0], [0, 0, 1]]
    )
    tilt = np.asarray(
        [[1, 0, 0], [0, np.cos(0.2), -np.sin(0.2)], [0, np.sin(0.2), np.cos(0.2)]]
    )
    return phx.discretization.CellMesh.from_triangles(points @ (tilt @ turn).T, faces)


def association_classes(
    association: GeometryAssociation, mesh: CellMesh
) -> tuple[np.ndarray, np.ndarray]:
    """``(dimension, index)`` B-Rep class and residual of every mesh vertex row."""
    rows = {
        int(value): row for row, value in enumerate(np.asarray(mesh.vertex_global_ids))
    }
    order = np.asarray(
        [rows[int(value)] for value in np.asarray(association.target_global_ids)],
        dtype=np.int64,
    )
    classes = np.empty((order.size, 2), dtype=np.int64)
    residuals = np.empty((order.size,), dtype=np.float64)
    classes[order, 0] = np.asarray(association.source_dimensions, dtype=np.int64)
    classes[order, 1] = np.asarray(association.source_indices, dtype=np.int64)
    residuals[order] = np.asarray(association.residuals, dtype=np.float64)
    return classes, residuals


def coordinate_rows(
    mesh: CellMesh, points: Sequence[Sequence[float]] | np.ndarray
) -> np.ndarray:
    coordinates = np.asarray(mesh.coordinates)
    return np.asarray(
        [
            int(np.flatnonzero(np.all(np.isclose(coordinates, point), axis=1))[0])
            for point in points
        ]
    )


def collapsed_mesh(
    mesh: CellMesh, removed: int, kept: int
) -> tuple[CellMesh, MeshLineage]:
    """Target mesh and lineage of collapsing vertex row ``removed`` into ``kept``."""
    identifiers = np.asarray(mesh.vertex_global_ids)
    cells = np.asarray(mesh.blocks[0].vertices)
    survivors = ~np.any(cells == removed, axis=1)
    remaining = np.flatnonzero(np.arange(identifiers.size) != removed)
    renumber = np.full(identifiers.size, -1, dtype=np.int64)
    renumber[remaining] = np.arange(remaining.size)
    target = phx.discretization.CellMesh.from_triangles(
        np.asarray(mesh.coordinates)[remaining],
        renumber[cells[survivors]],
        vertex_global_ids=identifiers[remaining],
        cell_global_ids=np.asarray(mesh.blocks[0].global_ids)[survivors],
    )
    record = EntityLineage(
        0,
        mesh.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        [identifiers[removed]],
        [identifiers[kept]],
        [EntityLineageKind.COLLAPSED_INTO],
    )
    return target, MeshLineage(mesh.topology_id, target.topology_id, (record,))


def repeated_definition_curving(
    corners: tuple[tuple[int, int], ...], triangles: np.ndarray, /
) -> tuple[tuple[BRepOccurrence, ...], np.ndarray, HighOrderCurvingResult]:
    """Degree-3 curving of triangles authored at ``(occurrence, face-0 corner)`` rows."""

    model = native_repeated_assembly()
    geometry = model.geometry
    if geometry is None:
        raise AssertionError(
            "curving_fixture.model.geometry must retain B-Rep geometry; got None"
        )
    projection = phx.geometry.prepare_brep_projection(model)
    coedges = geometry.face_loops[0][0]
    vertex_indices = np.asarray(
        [
            geometry.edge_vertices[geometry.coedge_edges[coedge]][
                0 if geometry.coedge_senses[coedge] > 0 else 1
            ]
            for coedge in coedges[:3]
        ],
        dtype=np.int32,
    )
    definition = np.asarray(geometry.vertex_points)[vertex_indices]
    instances = geometry.occurrences
    points = np.stack(
        [instances[instance].place(definition)[corner] for instance, corner in corners]
    )
    mesh = phx.discretization.CellMesh.from_triangles(points, triangles)
    paths = tuple(instances[instance].path for instance, _ in corners)
    indices = np.asarray(
        [vertex_indices[corner] for _, corner in corners], dtype=np.int32
    )
    projected = projection.project(
        points, np.zeros(indices.size, dtype=np.int8), indices, occurrence_paths=paths
    )
    np.testing.assert_array_equal(
        projected.status,
        np.full(indices.size, phx.geometry.BRepProjectionStatus.UNIQUE, dtype=np.int8),
    )
    np.testing.assert_array_equal(projected.source_occurrence_paths, paths)
    association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        projection.source_id,
        projection.source_revision,
        mesh.entity_set(0).entity_set_id,
        mesh.vertex_global_ids,
        projected.entity_ids(),
        projected.residuals,
        source_dimensions=projected.dimensions,
        source_indices=projected.indices,
        source_occurrence_paths=projected.source_occurrence_paths,
        resolved=np.ones(indices.size, dtype=np.bool_),
        exact=True,
    )
    result = phx.meshing.curve_cell_mesh(
        mesh,
        association,
        projection,
        policy=phx.meshing.HighOrderCurvingPolicy(
            fidelity_tolerance=1e-7, degree=3, relaxation_rounds=0
        ),
    )
    return instances, definition, result


def cylinder_mesh(rings: int, layers: int, height: float = 2.0) -> CellMesh:
    disk = [np.zeros(2, dtype=np.float64)]
    for ring in range(1, rings + 1):
        angles = 2.0 * np.pi * np.arange(6 * ring) / (6 * ring)
        disk.extend((ring / rings) * np.stack((np.cos(angles), np.sin(angles)), axis=1))
    disk = np.asarray(disk, dtype=np.float64)
    triangles = np.sort(Delaunay(disk).simplices, axis=1)
    count = disk.shape[0]
    heights = np.linspace(0.0, height, layers + 1)
    points = np.concatenate([np.column_stack((disk, np.full(count, z))) for z in heights])
    cells = []
    for layer in range(layers):
        low, high = triangles + layer * count, triangles + (layer + 1) * count
        # Lowest-index prism diagonals preserve conforming shared faces.
        cells += [
            np.stack((low[:, 0], low[:, 1], low[:, 2], high[:, 2]), axis=1),
            np.stack((low[:, 0], low[:, 1], high[:, 1], high[:, 2]), axis=1),
            np.stack((low[:, 0], high[:, 0], high[:, 1], high[:, 2]), axis=1),
        ]
    cells = np.concatenate(cells)
    corners = points[cells]
    volume = np.sum(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        * (corners[:, 3] - corners[:, 0]),
        axis=1,
    )
    cells[volume < 0] = cells[volume < 0][:, [0, 2, 1, 3]]
    return phx.discretization.CellMesh.from_tetrahedra(points, cells)


def mapped_coordinates(
    geometry: CellGeometrySpec,
    mesh: CellMesh,
    reference: np.ndarray,
    cells: np.ndarray | None = None,
) -> np.ndarray:
    elements, routes, coordinates = geometry.resolve(mesh)
    element = elements[0]
    if not isinstance(element, FiniteElementSpec):
        raise AssertionError("P2 convergence evidence requires a scalar finite element.")
    values, _ = element.tabulate(reference)
    route = np.asarray(routes[0])
    nodes = np.asarray(coordinates)[route if cells is None else route[cells]]
    return np.asarray(values)[None] @ nodes


def triangle_lattice(samples: int = 6) -> np.ndarray:
    return np.asarray(
        [
            (i / samples, j / samples)
            for i in range(samples + 1)
            for j in range(samples + 1 - i)
        ],
        dtype=np.float64,
    )


def sphere_error(geometry: CellGeometrySpec, mesh: CellMesh) -> float:
    return float(
        np.max(
            np.abs(
                np.linalg.norm(
                    mapped_coordinates(geometry, mesh, triangle_lattice()), axis=-1
                )
                - 1
            )
        )
    )


def lateral_error(geometry: CellGeometrySpec, mesh: CellMesh) -> float:
    vertices = np.asarray(mesh.blocks[0].vertices)
    radius = np.linalg.norm(np.asarray(mesh.coordinates)[:, :2], axis=1)
    reference = np.eye(4, 3, k=-1, dtype=np.float64)
    lattice = triangle_lattice(4)
    worst = 0.0
    for local in range(4):
        face = [corner for corner in range(4) if corner != local]
        cells = np.flatnonzero(np.all(np.isclose(radius[vertices[:, face]], 1.0), axis=1))
        if cells.size == 0:
            continue
        barycentric = np.column_stack((1.0 - lattice.sum(axis=1), lattice))
        mapped = mapped_coordinates(geometry, mesh, barycentric @ reference[face], cells)
        worst = max(
            worst, float(np.max(np.abs(np.linalg.norm(mapped[..., :2], axis=-1) - 1)))
        )
    return worst


def circular_p2_fidelity_bound(half_angle: float) -> float:
    """A two-sided radial budget for midpoint-interpolated unit-circle arcs."""
    # P(t)=(1-s*t², sin(a)*t), s=1-cos(a), so |P|²=1-s²*t²*(1-t²).
    # Its radial deficit is at most 1-sqrt(1-s²/4). Reserve the same amount
    # for tangential relaxation; certificate acceptance still checks the map.
    sagitta = 1.0 - np.cos(half_angle)
    return float(2.0 * (1.0 - np.sqrt(1.0 - sagitta**2 / 4.0)))


def sphere_fidelity_bound(mesh: CellMesh) -> float:
    """Bound the P2 unit-sphere interpolation error by its quartic hull."""
    # The squared radius is a degree-four Bernstein polynomial. Its coefficient
    # hull bounds every point, not just sampled nodes. Reserve one interpolation
    # error budget for relaxation, independently checked by the certificate.
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    triangles = points[np.asarray(mesh.blocks[0].vertices)]
    powers = tuple(power for power in product(range(3), repeat=3) if sum(power) == 2)
    controls = []
    for power in powers:
        rows = [row for row, weight in enumerate(power) if weight]
        if len(rows) == 1:
            controls.append(triangles[:, rows[0]])
        else:
            chord = triangles[:, rows].mean(axis=1)
            midpoint = chord / np.linalg.norm(chord, axis=1, keepdims=True)
            controls.append(2.0 * midpoint - chord)
    quartic: dict[tuple[int, ...], np.ndarray] = {}
    for first, second in product(range(len(powers)), repeat=2):
        exponent = tuple(
            a + b for a, b in zip(powers[first], powers[second], strict=True)
        )
        coefficient = (
            factorial(2) ** 2
            * prod(factorial(a) for a in exponent)
            / (
                factorial(4)
                * prod(factorial(a) for a in powers[first])
                * prod(factorial(a) for a in powers[second])
            )
        )
        term = coefficient * np.sum(controls[first] * controls[second], axis=1)
        previous = quartic.get(exponent)
        quartic[exponent] = term if previous is None else previous + term
    squared = np.stack(tuple(quartic.values()))
    radial_bound = max(
        1.0 - np.sqrt(float(np.min(squared))),
        np.sqrt(float(np.max(squared))) - 1.0,
    )
    return float(2.0 * radial_bound + 128 * np.finfo(np.float64).eps)
