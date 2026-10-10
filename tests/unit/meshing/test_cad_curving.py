from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import Delaunay

import phydrax as phx
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization.fem._reference import FiniteElementSpec
from phydrax.meshing._lineage import EntityLineage, EntityLineageKind, MeshLineage


_CONTRACT = phx.SpatialCoordinateContract.si()
_POLICY = phx.meshing.AssociationPropagationPolicy(classification_tolerance=1e-8)
_STATUS = phx.meshing.HighOrderCurvingStatus


def _plate_projection() -> Any:
    profile = phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(((-2, -2), (2, -2), (2, 2), (-2, 2))),
        (phx.geometry.ProfileLoop.circle((0, 0), 1.0),),
    )
    model = phx.geometry.brep_planar_face(profile, coordinate_contract=_CONTRACT)
    embedding = phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    return phx.geometry.prepare_brep_projection(model, embedding=embedding)


def _sphere_projection() -> Any:
    return phx.geometry.prepare_brep_projection(
        phx.geometry.brep_sphere(1.0, coordinate_contract=_CONTRACT)
    )


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


def test_native_source_order_realization_certifies_continuous_geometry_improvement() -> (
    None
):
    model = phx.geometry.brep_sphere(
        1.0,
        coordinate_contract=_CONTRACT,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    projection = phx.geometry.prepare_brep_projection(model)
    mesh = _icosphere(1)
    association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    policy = phx.meshing.HighOrderCurvingPolicy(
        degree=10,
        relaxation_rounds=0,
        residual_tolerance=1e-7,
        fidelity_tolerance=1e-7,
    )
    result = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)

    assert result.status is _STATUS.CURVED
    assert result.evidence.accepted
    assert result.evidence.fidelity is not None
    assert result.evidence.fidelity.status == "certified"
    assert result.evidence.fidelity.mesh_to_source_upper <= policy.fidelity_tolerance
    assert result.evidence.fidelity.source_to_mesh_upper <= policy.fidelity_tolerance
    assert result.evidence.source_revision == model.source_revision
    parameters = jnp.asarray(((0.2, 0.2), (0.1, 0.7), (1 / 3, 1 / 3)), dtype=jnp.float64)

    def radial_error(geometry: CellGeometrySpec) -> float:
        elements, routes, coordinates = geometry.resolve(mesh)
        errors = []
        for element, route in zip(elements, routes, strict=True):
            if not isinstance(element, FiniteElementSpec):
                raise TypeError(
                    "The realized Lagrange coordinate map requires its concrete scalar element."
                )
            basis, _ = element.tabulate(parameters)
            points = np.asarray(basis)[None] @ np.asarray(coordinates)[np.asarray(route)]
            errors.append(float(np.max(np.abs(np.linalg.norm(points, axis=-1) - 1.0))))
        return max(errors)

    assert radial_error(result.geometry) <= policy.fidelity_tolerance
    assert radial_error(result.geometry) < 1e-3 * radial_error(result.straight)


def test_native_projection_cannot_accept_nodal_evidence_as_continuous_fidelity() -> None:
    projection = _sphere_projection()
    mesh = _icosphere(1)
    policy = phx.meshing.HighOrderCurvingPolicy(degree=2, relaxation_rounds=0)
    association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    result = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)
    assert isinstance(projection, phx.geometry.NativeBRepProjection)
    assert result.status is _STATUS.ROLLED_BACK_CERTIFICATION
    assert result.candidate is not None
    assert result.candidate.certificate.all_certified
    assert result.candidate.maximum_residual <= policy.residual_tolerance
    assert "source_fidelity" in result.candidate.certification_failures
    assert not result.candidate.accepted
    assert result.accepted_round is None
    np.testing.assert_array_equal(
        result.geometry.coordinates, result.straight.coordinates
    )


def test_converged_local_repair_retains_rollback_without_source_certificate() -> None:
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
    # Local success does not replace missing continuous evidence.
    assert capped.status is _STATUS.ROLLED_BACK_CERTIFICATION
    assert capped.accepted_round is None and not capped.candidate.accepted
    assert phx.optim.OptimizationStatus.SUCCESS not in capped.relaxation_statuses
    np.testing.assert_array_equal(
        capped.geometry.coordinates, capped.straight.coordinates
    )
    assert relaxed.status is _STATUS.ROLLED_BACK_CERTIFICATION
    assert phx.optim.OptimizationStatus.SUCCESS in relaxed.relaxation_statuses
    assert relaxed.candidate.certificate.all_certified
    assert relaxed.candidate.minimum_scaled_jacobian > 0.0
    assert "source_fidelity" in relaxed.candidate.certification_failures
    np.testing.assert_array_equal(
        relaxed.geometry.coordinates, relaxed.straight.coordinates
    )


@pytest.mark.parametrize(
    ("rounds", "accept"),
    ((0, False), (1, False), (1, True)),
    ids=("projection", "nonconverged-refusal", "nonconverged-opt-in"),
)
def test_nonconvergence_opt_in_does_not_bypass_continuous_certificate(
    rounds: int,
    accept: bool,
) -> None:
    projection = _sphere_projection()
    mesh = _icosphere(1)
    association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    policy = phx.meshing.HighOrderCurvingPolicy(
        degree=2,
        relaxation_rounds=rounds,
        termination=phx.optim.OptimizationTermination(maximum_steps=1),
        accept_valid_nonconverged_relaxation=accept,
    )
    result = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)
    assert result.status is _STATUS.ROLLED_BACK_CERTIFICATION
    assert result.accepted_round is None
    assert result.candidate is not None
    assert result.candidate.certificate.all_certified
    assert not result.candidate.accepted
    np.testing.assert_array_equal(
        result.geometry.coordinates, result.straight.coordinates
    )
    expected = (
        () if rounds == 0 else (phx.optim.OptimizationStatus.MAXIMUM_STEPS_REACHED,)
    )
    assert result.relaxation_statuses == expected


def test_native_embedded_sphere_laplace_beltrami_converges_with_zero_mean() -> None:
    from examples.native_surface_meshing import generate, solve
    from phydrax._meshcore import meshcore_available

    if not meshcore_available():
        pytest.skip("Native sphere meshing requires compiled meshcore.")
    coarse = generate(0.8)
    fine = generate(0.4)
    coarse_error, coarse_mean = solve(coarse)
    fine_error, fine_mean = solve(fine)
    # Source is independently the unit sphere; this is a solved harmonic,
    # measured after radial lift, not a surface-area consistency check.
    assert fine_error < 0.5 * coarse_error
    assert fine_error < 0.06
    assert abs(coarse_mean) < 1e-12
    assert abs(fine_mean) < 1e-12


def _repeated_definition_curving(
    corners: tuple[tuple[int, int], ...], triangles: np.ndarray, /
) -> tuple[Any, np.ndarray, Any]:
    """Degree-3 curving of triangles authored at ``(occurrence, face-0 corner)`` rows."""
    from phydrax.meshing._association import GeometryAssociation, GeometryAssociationKind
    from tests._support.cad_models import native_repeated_assembly

    model = native_repeated_assembly()
    geometry = model.geometry
    if geometry is None:
        raise AssertionError("The native fixture must carry exact geometry.")
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
        policy=phx.meshing.HighOrderCurvingPolicy(degree=3, relaxation_rounds=0),
    )
    return instances, definition, result


def test_curving_repeated_definition_preserves_authored_occurrence_projection() -> None:
    instances, definition, result = _repeated_definition_curving(
        tuple((instance, corner) for instance in (0, 1) for corner in range(3)),
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
    )
    assert result.status is _STATUS.ROLLED_BACK_CERTIFICATION
    assert result.candidate is not None
    assert result.candidate.certificate.all_certified
    assert result.candidate.maximum_residual < 1e-10
    paths = result.candidate.node_occurrence_paths
    assert set(paths) == {instance.path for instance in instances}
    # Every degree-three node, including the face-interior diagonal's, carries
    # its own triangle's instance and lies on that instance's placed face.
    straight = np.asarray(result.straight.coordinates)
    for instance in instances:
        rows = [row for row, path in enumerate(paths) if path == instance.path]
        placed = instance.place(definition)
        normal = np.cross(placed[1] - placed[0], placed[2] - placed[0])
        assert len(rows) == 10
        np.testing.assert_allclose(
            (straight[rows] - placed[0]) @ normal, 0.0, atol=1e-12, rtol=0
        )
    # Omitting occurrence identity would snap the translated/rotated instance
    # onto the definition, although both rows name identical definition indices.
    np.testing.assert_allclose(
        result.geometry.coordinates, result.straight.coordinates, atol=0, rtol=0
    )


def test_curving_mixed_occurrence_corners_remain_unresolved() -> None:
    instances, _, result = _repeated_definition_curving(
        ((0, 0), (0, 1), (1, 2)), np.asarray([[0, 1, 2]], dtype=np.int32)
    )
    # No occurrence path is common to the corners, so neither the cell nor its
    # mixed edges derive a class; the refusal keeps the straight geometry.
    assert result.status is _STATUS.UNRESOLVED_ASSOCIATION
    assert result.candidate is None
    dimensions = np.asarray(result.evidence.node_dimensions)
    paths = result.evidence.node_occurrence_paths
    assert np.any(dimensions < 0)
    assert all(paths[row] == () for row in np.flatnonzero(dimensions < 0))
    assert {instance.path for instance in instances} <= set(paths)
    np.testing.assert_allclose(
        result.geometry.coordinates, result.straight.coordinates, atol=0, rtol=0
    )


@pytest.mark.parametrize("degree", (3, 4))
@pytest.mark.parametrize("rotation", (False, True))
def test_quotient_curving_nodes_preserve_oriented_high_order_seam_traces(
    degree: int,
    rotation: bool,
) -> None:
    from phydrax.discretization import (
        FiniteElementSpec,
        PeriodicIsometryGroup,
        PeriodicMeshTopology,
    )
    from phydrax.meshing._curving import (
        _apply_periodic,
        _periodic_nodes,
        _straight_geometry,
    )

    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    block = phx.discretization.CellBlock(
        "wall", "triangle", np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    )
    generators = np.repeat(np.eye(4)[None], 2, axis=0)
    generators[0, 0, 3] = 1.0
    generators[1, 1, 3] = 1.0
    representatives = np.zeros(4, dtype=np.int64)
    shifts = points[:, :2].astype(np.int64)
    if rotation:
        points = np.asarray(
            ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 0.0))
        )
        generators = np.asarray(
            (
                (
                    (0.0, -1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            )
        )
        representatives = np.asarray((0, 1, 1, 0), dtype=np.int64)
        shifts = np.asarray(((0,), (0,), (1,), (1,)), dtype=np.int64)
    lifted = phx.discretization.CellMesh(points, (block,))
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicIsometryGroup(generators),
        representatives,
        shifts,
    )
    wall = phx.discretization.CellMesh(points, (block,), periodic_topology=topology)
    straight = _straight_geometry(wall, degree)
    pairs = _periodic_nodes(wall, straight, ())
    moved = np.asarray(straight.coordinates).copy()
    moved[4:, 2] = 0.02 * np.sin(np.arange(4, moved.shape[0]))
    coordinates = _apply_periodic(moved, pairs)
    elements, routes, _ = straight.resolve(wall)
    t = np.linspace(0.05, 0.95, 9)
    element = elements[0]
    if not isinstance(element, FiniteElementSpec):
        raise AssertionError(
            "The straight coordinate lattice must retain its finite-element basis."
        )
    route = np.asarray(routes[0])
    if rotation:
        source = (
            np.asarray(element.tabulate(np.stack((t, np.zeros_like(t)), axis=1))[0])
            @ coordinates[route[0]]
        )
        target = (
            np.asarray(element.tabulate(np.stack((t, 1.0 - t), axis=1))[0])
            @ coordinates[route[1]]
        )
        np.testing.assert_allclose(target, source @ generators[0, :3, :3].T, atol=1e-12)
        assert np.max(np.abs(source[:, 2])) > 1e-4
    else:
        left = (
            np.asarray(element.tabulate(np.stack((np.zeros_like(t), t), axis=1))[0])
            @ coordinates[route[1]]
        )
        right = (
            np.asarray(element.tabulate(np.stack((1.0 - t, t), axis=1))[0])
            @ coordinates[route[0]]
        )
        bottom = (
            np.asarray(element.tabulate(np.stack((t, np.zeros_like(t)), axis=1))[0])
            @ coordinates[route[0]]
        )
        top = (
            np.asarray(element.tabulate(np.stack((t, 1.0 - t), axis=1))[0])
            @ coordinates[route[1]]
        )
        np.testing.assert_allclose(
            right - left, np.tile((1.0, 0.0, 0.0), (t.size, 1)), atol=1e-12
        )
        np.testing.assert_allclose(
            top - bottom, np.tile((0.0, 1.0, 0.0), (t.size, 1)), atol=1e-12
        )
        assert np.max(np.abs(left[:, 2])) > 1e-4
        assert len(set(topology.entity_keys(1))) == 3


def test_existing_coordinate_node_budget_uses_actual_supplied_layout() -> None:
    from phydrax.meshing._curving import _straight_geometry

    projection = _plate_projection()
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((1.25, 1.25), (1.75, 1.25), (1.25, 1.75)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    geometry = _straight_geometry(mesh, 2)
    evidence = phx.meshing.verify_curved_geometry(
        geometry,
        mesh,
        association,
        projection,
        policy=phx.meshing.HighOrderCurvingPolicy(degree=2, maximum_geometry_nodes=6),
    )
    assert evidence.certificate.certified_valid_count == 1
    assert evidence.node_dimensions.shape == (6,)
    with pytest.raises(ValueError, match="node resource limit"):
        phx.meshing.verify_curved_geometry(
            geometry,
            mesh,
            association,
            projection,
            policy=phx.meshing.HighOrderCurvingPolicy(degree=2, maximum_geometry_nodes=5),
        )
