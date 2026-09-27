from typing import Any

import numpy as np
import pytest
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt
from scipy.spatial import Delaunay

import phydrax as phx
from phydrax.meshing._lineage import EntityLineage, EntityLineageKind, MeshLineage


_CONTRACT = phx.SpatialCoordinateContract.si()
_POLICY = phx.meshing.AssociationPropagationPolicy(classification_tolerance=1e-8)
_STATUS = phx.meshing.HighOrderCurvingStatus


def _bound(shape: Any, embedding: Any = None) -> Any:
    model = phx.geometry.model_from_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    return phx.geometry.prepare_brep_projection(model, shape, embedding=embedding)


def _plate_projection() -> Any:
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in ((-2, -2), (2, -2), (2, 2), (-2, 2)):
        polygon.Add(gp_Pnt(x, y, 0))
    polygon.Close()
    square = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    circle = BRepBuilderAPI_MakeEdge(
        gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1.0)
    ).Edge()
    disk = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(circle).Wire()).Face()
    embedding = phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    return _bound(BRepAlgoAPI_Cut(square, disk).Shape(), embedding)


def _plate_mesh(apex: Any) -> Any:
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


def _icosphere(level: Any) -> Any:
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
    # A generic rotation keeps vertices off the sphere's seam and poles.
    turn = np.asarray(
        [[np.cos(0.3), -np.sin(0.3), 0], [np.sin(0.3), np.cos(0.3), 0], [0, 0, 1]]
    )
    tilt = np.asarray(
        [[1, 0, 0], [0, np.cos(0.2), -np.sin(0.2)], [0, np.sin(0.2), np.cos(0.2)]]
    )
    return phx.discretization.CellMesh.from_triangles(points @ (tilt @ turn).T, faces)


def _cylinder_mesh(rings: Any, layers: Any, height: Any = 2.0) -> Any:
    disk = [np.zeros(2)]
    for ring in range(1, rings + 1):
        angles = 2.0 * np.pi * np.arange(6 * ring) / (6 * ring)
        disk.extend((ring / rings) * np.stack((np.cos(angles), np.sin(angles)), axis=1))
    disk = np.asarray(disk)
    triangles = np.sort(Delaunay(disk).simplices, axis=1)
    count = disk.shape[0]
    heights = np.linspace(0.0, height, layers + 1)
    points = np.concatenate([np.column_stack((disk, np.full(count, z))) for z in heights])
    cells = []
    for layer in range(layers):
        low, high = triangles + layer * count, triangles + (layer + 1) * count
        # Prism diagonals start at the lowest vertex index, so shared faces conform.
        cells += [
            np.stack((low[:, 0], low[:, 1], low[:, 2], high[:, 2]), axis=1),
            np.stack((low[:, 0], low[:, 1], high[:, 1], high[:, 2]), axis=1),
            np.stack((low[:, 0], high[:, 0], high[:, 1], high[:, 2]), axis=1),
        ]
    cells = np.concatenate(cells)
    corners = points[cells]
    volume = np.einsum(
        "ij,ij->i",
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        corners[:, 3] - corners[:, 0],
    )
    cells[volume < 0] = cells[volume < 0][:, [0, 2, 1, 3]]
    return phx.discretization.CellMesh.from_tetrahedra(points, cells)


def _mapped(geometry: Any, reference: Any, cells: Any = None) -> Any:
    """Geometry map evaluated at reference points of every (selected) cell."""
    element = geometry.elements[0]
    routes = np.asarray(geometry.geometry_dofs[0])
    values, _ = element.tabulate(reference)
    nodes = np.asarray(geometry.coordinates)[routes if cells is None else routes[cells]]
    return np.einsum("mn,cna->cma", np.asarray(values), nodes)


def _triangle_lattice(samples: Any = 6) -> Any:
    return np.asarray(
        [
            (i / samples, j / samples)
            for i in range(samples + 1)
            for j in range(samples + 1 - i)
        ]
    )


def _sphere_error(geometry: Any) -> Any:
    return float(
        np.max(
            np.abs(np.linalg.norm(_mapped(geometry, _triangle_lattice()), axis=-1) - 1)
        )
    )


def _lateral_error(geometry: Any, mesh: Any) -> Any:
    """Largest radial deviation of the lateral boundary faces of a cylinder mesh."""
    vertices = np.asarray(mesh.blocks[0].vertices)
    radius = np.linalg.norm(np.asarray(mesh.coordinates)[:, :2], axis=1)
    reference = np.eye(4, 3, k=-1)
    lattice = _triangle_lattice(4)
    worst = 0.0
    for local in range(4):
        face = [corner for corner in range(4) if corner != local]
        cells = np.flatnonzero(np.all(np.isclose(radius[vertices[:, face]], 1.0), axis=1))
        if cells.size == 0:
            continue
        barycentric = np.column_stack((1.0 - lattice.sum(axis=1), lattice))
        mapped = _mapped(geometry, barycentric @ reference[face], cells)
        worst = max(
            worst, float(np.max(np.abs(np.linalg.norm(mapped[..., :2], axis=-1) - 1)))
        )
    return worst


def _classes(association: Any, mesh: Any) -> Any:
    """``(dimension, index)`` B-Rep class and residual of every mesh vertex row."""
    rows = {
        int(value): row for row, value in enumerate(np.asarray(mesh.vertex_global_ids))
    }
    order = np.asarray(
        [rows[int(value)] for value in np.asarray(association.target_global_ids)],
        dtype=np.int64,
    )
    classes = np.empty((order.size, 2), dtype=np.int64)
    residuals = np.empty((order.size,))
    classes[order, 0] = np.asarray(association.source_dimensions, dtype=np.int64)
    classes[order, 1] = np.asarray(association.source_indices, dtype=np.int64)
    residuals[order] = np.asarray(association.residuals, dtype=np.float64)
    return classes, residuals


def _rows(mesh: Any, points: Any) -> Any:
    coordinates = np.asarray(mesh.coordinates)
    return np.asarray(
        [
            int(np.flatnonzero(np.all(np.isclose(coordinates, point), axis=1))[0])
            for point in points
        ]
    )


def test_cad_curving_scenario_1() -> None:
    projection = _plate_projection()
    mesh = _plate_mesh((1.2, 1.2))
    association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    source = phx.meshing.certify_cell_mesh(mesh, _CONTRACT, associations=(association,))
    adapted = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(np.asarray(mesh.blocks[0].global_ids)),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
                association_transfer=phx.meshing.BRepAssociationTransfer(
                    projection, policy=_POLICY
                ),
            ),
        )
    )
    target = adapted.target.associations[0]
    target_mesh = adapted.target.mesh
    source_classes, _ = _classes(association, mesh)
    classes, residuals = _classes(target, target_mesh)
    hole = source_classes[_rows(mesh, [[0.0, 1.0]])[0]]
    bottom = source_classes[_rows(mesh, [[0.0, -2.0]])[0]]
    corner = source_classes[_rows(mesh, [[2.0, -2.0]])[0]]

    assert target.complete
    assert target.provenance is phx.meshing.GeometryAssociationProvenance.LINEAGE
    assert target.parent_association_id == association.association_id
    # Corner vertices keep their B-Rep vertex.
    np.testing.assert_array_equal(classes[_rows(target_mesh, [[2.0, -2.0]])[0]], corner)
    # Children of a hole chord inherit the hole edge; the chord sagitta is the residual.
    chord = _rows(target_mesh, [[0.5, 0.5], [-0.5, -0.5]])
    np.testing.assert_array_equal(classes[chord], np.stack((hole, hole)))
    np.testing.assert_allclose(residuals[chord], 1.0 - 2**-0.5)
    # Children of straight boundary edges lie exactly on their edge.
    side = _rows(target_mesh, [[1.0, -2.0]])
    np.testing.assert_array_equal(classes[side], bottom[None])
    np.testing.assert_allclose(residuals[side], 0.0, atol=1e-12)
    # Interior children inherit the face.
    points = np.asarray(target_mesh.coordinates)
    interior = (np.abs(points).sum(axis=1) > 1.0 + 1e-9) & (
        np.max(np.abs(points), axis=1) < 2.0 - 1e-9
    )
    assert np.all(classes[interior, 0] == 2)
    assert np.all(classes[~interior, 0] < 2)
    projection = _plate_projection()
    mesh = _plate_mesh((1.2, 1.2))
    association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    removed, corner, hole = _rows(mesh, [[0.0, -2.0], [2.0, -2.0], [1.0, 0.0]])

    target, lineage = _collapse(mesh, removed, corner)
    legal = phx.meshing.propagate_association(
        association, lineage, mesh, target, projection, policy=_POLICY
    )
    assert legal.complete
    target, lineage = _collapse(mesh, removed, hole)
    with pytest.raises(phx.meshing.AssociationPropagationError) as error:
        phx.meshing.propagate_association(
            association, lineage, mesh, target, projection, policy=_POLICY
        )
    assert error.value.target_ids.tolist() == [int(mesh.vertex_global_ids[hole])]
    projection = _plate_projection()
    source_mesh = _plate_mesh((1.2, 1.2))
    association = phx.meshing.associate_mesh_vertices(
        source_mesh, projection, policy=_POLICY
    )
    source = phx.meshing.certify_cell_mesh(
        source_mesh, _CONTRACT, associations=(association,)
    )
    target = phx.meshing.certify_cell_mesh(_plate_mesh((1.3, -1.1)), _CONTRACT)

    derived = phx.meshing.rederive_association(
        association, source, target, projection, policy=_POLICY
    )
    shared = np.asarray(source_mesh.coordinates)[:12]
    classes, _ = _classes(derived, target.mesh)
    source_classes, _ = _classes(association, source_mesh)

    assert derived.complete
    assert derived.provenance is phx.meshing.GeometryAssociationProvenance.CLASSIFICATION
    np.testing.assert_array_equal(
        classes[_rows(target.mesh, shared)], source_classes[_rows(source_mesh, shared)]
    )


def _collapse(mesh: Any, removed: Any, kept: Any) -> Any:
    """Target mesh and lineage of collapsing vertex row ``removed`` into ``kept``."""
    identifiers = np.asarray(mesh.vertex_global_ids)
    cells = np.asarray(mesh.blocks[0].vertices)
    survivors = ~np.any(cells == removed, axis=1)
    remaining = np.flatnonzero(np.arange(identifiers.size) != removed)
    renumber = np.full(identifiers.size, -1)
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


def test_cad_curving_scenario_2() -> None:
    projection = _bound(BRepPrimAPI_MakeSphere(1.0).Shape())
    policy = phx.meshing.HighOrderCurvingPolicy(degree=2, relaxation_rounds=1)
    errors = []
    for level in (1, 2):
        mesh = _icosphere(level)
        association = phx.meshing.associate_mesh_vertices(
            mesh, projection, policy=_POLICY
        )
        result = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)
        assert result.status is _STATUS.CURVED
        assert result.evidence.certificate.all_certified
        assert result.evidence.maximum_residual <= policy.residual_tolerance
        errors.append((_sphere_error(result.straight), _sphere_error(result.geometry)))

    (straight_coarse, curved_coarse), (straight_fine, curved_fine) = errors
    assert curved_coarse < 0.05 * straight_coarse
    # Straight facets converge at second order; P2 curving at least at third.
    assert 3.0 < straight_coarse / straight_fine < 5.0
    assert curved_coarse / curved_fine > 7.0
    projection = _bound(BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape())
    mesh = _cylinder_mesh(2, 2)
    association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    result = phx.meshing.certify_cell_mesh(mesh, _CONTRACT, associations=(association,))
    part = phx.meshing.MeshPart("cylinder", result)
    coordinates = np.asarray(mesh.coordinates)
    identifiers = np.asarray(mesh.vertex_global_ids)
    bottom = np.isclose(coordinates[:, 2], 0.0)
    top = np.isclose(coordinates[:, 2], 2.0)
    top_ids = np.sort(identifiers[top])
    below = {
        tuple(np.round(coordinates[row, :2], 12)): identifiers[row]
        for row in np.flatnonzero(bottom)
    }
    source_ids = np.asarray(
        [
            below[tuple(np.round(coordinates[identifiers == value][0, :2], 12))]
            for value in top_ids
        ]
    )
    translation = np.asarray([0.0, 0.0, 2.0])
    coupling = phx.meshing.PeriodicCoupling(
        part,
        part,
        part.scope(0, np.sort(identifiers[bottom])),
        part.scope(0, top_ids),
        np.eye(3),
        translation,
        source_ids=source_ids,
    )
    curved = phx.meshing.curve_cell_mesh(
        mesh,
        association,
        projection,
        policy=phx.meshing.HighOrderCurvingPolicy(degree=2, relaxation_rounds=1),
        periodic=(coupling,),
    )
    nodes = np.asarray(curved.geometry.coordinates)
    straight = np.asarray(curved.straight.coordinates)
    top_nodes = np.flatnonzero(np.isclose(straight[:, 2], 2.0))
    bottom_nodes = np.flatnonzero(np.isclose(straight[:, 2], 0.0))
    source_rows, target_rows = coupling.match_points(
        straight[bottom_nodes], straight[top_nodes]
    )

    assert curved.status is _STATUS.CURVED
    assert curved.evidence.certificate.all_certified
    assert _lateral_error(curved.geometry, mesh) < 0.05 * _lateral_error(
        curved.straight, mesh
    )
    np.testing.assert_array_equal(
        nodes[top_nodes[target_rows]], nodes[bottom_nodes[source_rows]] + translation
    )
    assert np.max(np.abs(nodes[top_nodes] - straight[top_nodes])) > 0.01
    projection = _bound(BRepPrimAPI_MakeSphere(1.0).Shape())
    mesh = _icosphere(1)
    association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    policy = phx.meshing.HighOrderCurvingPolicy(degree=2, relaxation_rounds=1)
    curved = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)

    accepted = phx.meshing.verify_curved_geometry(
        curved.geometry, mesh, association, projection, policy=policy
    )
    straight = phx.meshing.verify_curved_geometry(
        curved.straight, mesh, association, projection, policy=policy
    )
    assert accepted.accepted
    assert accepted.certificate.all_certified
    assert not straight.accepted
    assert straight.maximum_residual > 0.01


def test_inverting_curving_rolls_back_and_converged_relaxation_repairs_it() -> None:
    projection = _plate_projection()
    mesh = _plate_mesh((0.8, 0.8))
    association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)

    def curve(rounds: Any, steps: Any) -> Any:
        policy = phx.meshing.HighOrderCurvingPolicy(
            degree=2,
            relaxation_rounds=rounds,
            termination=phx.optim.OptimizationTermination(maximum_steps=steps),
        )
        return phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)

    projected = curve(0, 64)
    capped = curve(2, 64)
    relaxed = curve(2, 512)

    # Snapping the hole chord onto the arc folds the element at a hole vertex.
    assert projected.status is _STATUS.ROLLED_BACK_INVALID
    assert projected.candidate.certificate.invalid_count > 0
    np.testing.assert_array_equal(
        projected.geometry.coordinates, projected.straight.coordinates
    )
    # Relaxations cut off by the step budget never replace the straight geometry,
    # even when their candidate would pass the certificate.
    assert capped.status is _STATUS.ROLLED_BACK_NONCONVERGED
    assert capped.accepted_round is None and capped.candidate.accepted
    assert phx.optim.OptimizationStatus.SUCCESS not in capped.relaxation_statuses
    np.testing.assert_array_equal(
        capped.geometry.coordinates, capped.straight.coordinates
    )
    assert relaxed.status is _STATUS.CURVED
    assert relaxed.relaxation_statuses[relaxed.accepted_round - 1] is (
        phx.optim.OptimizationStatus.SUCCESS
    )
    assert relaxed.evidence.certificate.all_certified
    assert relaxed.evidence.minimum_scaled_jacobian > 0.0


def test_nonconverged_relaxation_replaces_a_valid_projection_only_when_permitted() -> (
    None
):
    projection = _bound(BRepPrimAPI_MakeSphere(1.0).Shape())
    mesh = _icosphere(1)
    association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)

    def curve(rounds: Any, accept: Any) -> Any:
        policy = phx.meshing.HighOrderCurvingPolicy(
            degree=2,
            relaxation_rounds=rounds,
            termination=phx.optim.OptimizationTermination(maximum_steps=1),
            accept_valid_nonconverged_relaxation=accept,
        )
        return phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)

    projected = curve(0, False)
    retained = curve(1, False)
    replaced = curve(1, True)

    budget = (phx.optim.OptimizationStatus.MAXIMUM_STEPS_REACHED,)
    assert projected.status is _STATUS.CURVED and projected.accepted_round == 0
    # The relaxed candidate is valid but not converged: the projection stays.
    assert retained.status is _STATUS.CURVED and retained.accepted_round == 0
    assert retained.candidate.accepted and retained.relaxation_statuses == budget
    np.testing.assert_array_equal(
        retained.geometry.coordinates, projected.geometry.coordinates
    )
    assert replaced.status is _STATUS.CURVED and replaced.accepted_round == 1
    assert replaced.relaxation_statuses == budget and replaced.evidence.accepted
    assert not np.array_equal(
        replaced.geometry.coordinates, projected.geometry.coordinates
    )
