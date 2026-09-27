from itertools import product
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


meshing = phx.meshing
_TARGET = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)))
_CELLS = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32)
_BOUNDARY = np.asarray((True, True, True, True, False))


def _mesh(center: Any) -> Any:
    coordinates = _TARGET.copy()
    coordinates[4] = center
    return phx.discretization.CellMesh.from_triangles(coordinates, _CELLS)


@pytest.mark.parametrize(
    ("objective", "method"),
    (
        (meshing.MeshQualityObjective.SHAPE_SIZE, None),
        (meshing.MeshQualityObjective.SHAPE, None),
        (meshing.MeshQualityObjective.GRAM_DETERMINANT, None),
        (meshing.MeshQualityObjective.SHAPE_SIZE, phx.optim.NewtonTrustRegion()),
    ),
)
def test_target_matrix_optimization_improves_quality_with_bit_identical_fixed_nodes(
    objective: Any, method: Any
) -> None:
    mesh = _mesh((0.75, 0.25))
    plan = meshing.TargetMatrixOptimizationPlan(
        mesh,
        objective=objective,
        target_coordinates=_TARGET,
        fixed_vertices=_BOUNDARY,
        method=method,
        termination=phx.optim.OptimizationTermination(maximum_steps=30),
    )

    result = meshing.optimize_cell_mesh(plan, phx.SpatialCoordinateContract.si())

    assert result.status is meshing.MeshOptimizationStatus.OPTIMIZED
    assert result.optimizer_status is phx.optim.OptimizationStatus.SUCCESS
    assert result.accepted_steps > 0
    assert result.final_objective < result.initial_objective
    np.testing.assert_array_equal(
        # ty: ignore[unresolved-attribute]
        result.result.mesh.coordinates[:4],
        np.asarray(mesh.coordinates)[:4],
    )
    # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(result.result.mesh.coordinates[4], (0.5, 0.5), atol=1e-6)
    initial = meshing.evaluate_cell_quality(mesh).mean_ratios
    # ty: ignore[unresolved-attribute]
    assert np.min(result.result.quality.evaluation.mean_ratios) > np.min(initial)


def test_coordinate_bounds_are_enforced_by_projection() -> None:
    plan = meshing.TargetMatrixOptimizationPlan(
        _mesh((0.75, 0.25)),
        target_coordinates=_TARGET,
        fixed_vertices=_BOUNDARY,
        # ty: ignore[invalid-argument-type]
        coordinate_bounds=((0.6, 0.0), (1.0, 1.0)),
    )
    result = meshing.optimize_cell_mesh(plan, phx.SpatialCoordinateContract.si())
    assert result.accepted
    assert np.asarray(result.coordinates)[4, 0] >= 0.6
    np.testing.assert_allclose(np.asarray(result.coordinates)[4], (0.6, 0.5), atol=1e-6)


def test_metric_alignment_moves_nodes_toward_unit_metric_edges() -> None:
    mesh = _mesh((0.5, 0.5))
    vertices = mesh.entity_set(0)
    scope = meshing.MeshingScope(
        mesh.mesh_id,
        "r",
        meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    graded = (
        np.tile(np.eye(2), (5, 1, 1))
        * np.where(np.asarray(mesh.coordinates)[:, :1] < 0.5, 16.0, 1.0)[:, :, None]
    )
    metric = meshing.MeshMetricField(scope, graded, minimum_size=0.1, maximum_size=2.0)
    plan = meshing.TargetMatrixOptimizationPlan(
        mesh,
        objective=meshing.MeshQualityObjective.METRIC_ALIGNMENT,
        metric=metric,
        fixed_vertices=_BOUNDARY,
    )
    result = meshing.optimize_cell_mesh(plan, phx.SpatialCoordinateContract.si())
    assert result.accepted and result.final_objective < result.initial_objective
    # The refined left half requests smaller cells there.
    assert np.asarray(result.coordinates)[4, 0] < 0.5


def test_untangling_certifies_a_folded_mesh_before_optimization() -> None:
    mesh = _mesh((1.3, 0.5))
    plan = meshing.TargetMatrixOptimizationPlan(
        mesh, target_coordinates=_TARGET, fixed_vertices=_BOUNDARY
    )
    result = meshing.optimize_cell_mesh(plan, phx.SpatialCoordinateContract.si())

    assert result.status is meshing.MeshOptimizationStatus.OPTIMIZED
    # ty: ignore[unresolved-attribute]
    assert result.untangling.succeeded
    # ty: ignore[unresolved-attribute]
    assert result.untangling.initial_inverted_count > 0
    # ty: ignore[unresolved-attribute]
    assert result.untangling.final_inverted_count == 0
    assert np.isinf(result.initial_objective)
    # ty: ignore[unresolved-attribute]
    assert np.all(np.asarray(result.result.quality.evaluation.sampled_valid))


def test_inverted_meshes_fail_without_mutation() -> None:
    folded = _mesh((1.3, 0.5))
    unguarded = meshing.TargetMatrixOptimizationPlan(
        folded, target_coordinates=_TARGET, fixed_vertices=_BOUNDARY, untangling=None
    )
    refused = meshing.optimize_cell_mesh(unguarded, phx.SpatialCoordinateContract.si())
    assert refused.status is meshing.MeshOptimizationStatus.INVERTED_INPUT
    assert refused.result is None
    np.testing.assert_array_equal(refused.coordinates, folded.coordinates)

    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-0.4, -0.4)))
    trapped = phx.discretization.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32)
    )
    plan = meshing.TargetMatrixOptimizationPlan(
        trapped,
        target_coordinates=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.2, 1.2))),
        fixed_vertices=np.asarray((True, True, True, False)),
        # ty: ignore[invalid-argument-type]
        coordinate_bounds=((-1.0, -1.0), (0.0, 0.0)),
    )
    failed = meshing.optimize_cell_mesh(plan, phx.SpatialCoordinateContract.si())
    assert failed.status is meshing.MeshOptimizationStatus.UNTANGLING_FAILED
    assert failed.result is None and failed.inverted_count > 0
    # ty: ignore[unresolved-attribute]
    assert not failed.untangling.succeeded
    np.testing.assert_array_equal(failed.coordinates, trapped.coordinates)


def test_nonconverged_optimization_commits_only_under_explicit_acceptance() -> None:
    mesh = _mesh((0.75, 0.25))

    def optimize(accept: Any) -> Any:
        plan = meshing.TargetMatrixOptimizationPlan(
            mesh,
            target_coordinates=_TARGET,
            fixed_vertices=_BOUNDARY,
            termination=phx.optim.OptimizationTermination(maximum_steps=1),
            accept_valid_nonconverged=accept,
        )
        return meshing.optimize_cell_mesh(plan, phx.SpatialCoordinateContract.si())

    refused = optimize(False)
    admitted = optimize(True)

    budget = phx.optim.OptimizationStatus.MAXIMUM_STEPS_REACHED
    assert refused.status is meshing.MeshOptimizationStatus.NONCONVERGED
    assert not refused.accepted and refused.result is None
    assert refused.optimizer_status is budget and refused.inverted_count == 0
    np.testing.assert_array_equal(refused.coordinates, mesh.coordinates)
    # The rejected iterate stays in the native evidence for diagnostics only.
    assert not np.allclose(refused.minimization.parameters, mesh.coordinates[4:])
    assert admitted.status is meshing.MeshOptimizationStatus.VALID_NONCONVERGED
    assert admitted.accepted and admitted.optimizer_status is budget
    assert admitted.result.audit.passed
    np.testing.assert_array_equal(admitted.result.mesh.coordinates, admitted.coordinates)
    np.testing.assert_array_equal(
        np.asarray(admitted.coordinates)[4:], refused.minimization.parameters
    )
    assert admitted.final_objective < admitted.initial_objective


def test_nonconverged_untangling_stage_is_accepted_only_when_permitted() -> None:
    folded = _mesh((1.3, 0.5))

    def optimize(accept: Any) -> Any:
        plan = meshing.TargetMatrixOptimizationPlan(
            folded,
            target_coordinates=_TARGET,
            fixed_vertices=_BOUNDARY,
            termination=phx.optim.OptimizationTermination(maximum_steps=1),
            untangling=meshing.MeshUntanglingPolicy(maximum_stages=1),
            accept_valid_nonconverged=accept,
        )
        return meshing.optimize_cell_mesh(plan, phx.SpatialCoordinateContract.si())

    refused = optimize(False)
    admitted = optimize(True)

    stage = refused.untangling
    assert refused.status is meshing.MeshOptimizationStatus.UNTANGLING_FAILED
    assert refused.result is None
    # The stage removed every inversion and passed the audit but did not converge.
    assert stage.final_inverted_count == 0 and stage.audit_passed
    assert not stage.converged and not stage.succeeded
    assert stage.optimizer_statuses == (
        phx.optim.OptimizationStatus.MAXIMUM_STEPS_REACHED,
    )
    np.testing.assert_array_equal(refused.coordinates, folded.coordinates)
    assert admitted.untangling.succeeded and not admitted.untangling.converged
    assert admitted.status is meshing.MeshOptimizationStatus.VALID_NONCONVERGED
    assert admitted.accepted
    assert np.all(np.asarray(admitted.result.quality.evaluation.sampled_valid))


def test_untangling_uses_its_stage_budget_when_a_valid_iterate_fails_audit() -> None:
    folded = _mesh((1.3, 0.5))
    plan = meshing.TargetMatrixOptimizationPlan(
        folded,
        target_coordinates=_TARGET,
        fixed_vertices=_BOUNDARY,
        termination=phx.optim.OptimizationTermination(maximum_steps=1),
        untangling=meshing.MeshUntanglingPolicy(maximum_stages=2),
        audit_policy=meshing.CellMeshAuditPolicy(minimum_mean_ratio=1.0),
    )

    result = meshing.optimize_cell_mesh(plan, phx.SpatialCoordinateContract.si())

    assert result.status is meshing.MeshOptimizationStatus.UNTANGLING_FAILED
    # ty: ignore[unresolved-attribute]
    assert result.untangling.final_inverted_count == 0
    # ty: ignore[unresolved-attribute]
    assert not result.untangling.audit_passed
    # ty: ignore[unresolved-attribute]
    assert len(result.untangling.minimizations) == 2


def test_generic_high_order_coordinate_optimizer_preserves_fixed_nodes() -> None:
    element = phx.discretization.lagrange_element("triangle", 2)
    coordinates = np.array(element.reference_nodes, copy=True)
    coordinates[3:] += 0.1
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": element},
        {"triangles": np.arange(6, dtype=np.int32)[None, :]},
        coordinates,
    )
    target = jnp.asarray(element.reference_nodes)

    optimized = meshing.optimize_cell_geometry_coordinates(
        geometry,
        lambda values: jnp.sum((values - target) ** 2),
        fixed_coordinates=np.asarray((True, True, True, False, False, False)),
        termination=phx.optim.OptimizationTermination(maximum_steps=25),
    )

    np.testing.assert_array_equal(optimized.coordinates[:3], coordinates[:3])
    np.testing.assert_allclose(optimized.coordinates[3:], target[3:], atol=1e-6)
    assert optimized.converged


# Unit-cube faces (vertex x + 2y + 4z), counterclockwise seen from outside.
_CUBE_FACES = (
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
)


def _prism_mesh(center: Any) -> Any:
    base = np.column_stack((_TARGET, np.zeros((5,))))
    points = np.concatenate([base + (0.0, 0.0, height) for height in (0.0, 0.5, 1.0)])
    prisms = np.concatenate(
        [
            np.concatenate((_CELLS + 5 * layer, _CELLS + 5 * (layer + 1)), axis=1)
            for layer in (0, 1)
        ]
    )
    target = points.copy()
    points[9] = center
    block = phx.discretization.CellBlock("prisms", "prism", prisms)
    return phx.discretization.CellMesh.from_mixed_3d(
        points, (block,), polyhedra={}
    ), target


def _pyramid_mesh(apex: Any) -> Any:
    cube = np.asarray(
        [(x, y, z) for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    )
    pyramids = np.asarray([(*face[::-1], 8) for face in _CUBE_FACES])
    block = phx.discretization.CellBlock("pyramids", "pyramid", pyramids)
    points = np.concatenate((cube, np.asarray(apex)[None]))
    target = np.concatenate((cube, np.full((1, 3), 0.5)))
    return phx.discretization.CellMesh.from_mixed_3d(
        points, (block,), polyhedra={}
    ), target


def _polyhedral_mesh(center: Any) -> Any:
    grid = np.asarray(
        [
            (x, y, z)
            for z in (0.0, 0.5, 1.0)
            for y in (0.0, 0.5, 1.0)
            for x in (0.0, 0.5, 1.0)
        ]
    )
    cells = []
    for corner in product(range(2), repeat=3):
        local = [
            (corner[0] + dx) + 3 * (corner[1] + dy) + 9 * (corner[2] + dz)
            for dz in (0, 1)
            for dy in (0, 1)
            for dx in (0, 1)
        ]
        cells.append([[local[vertex] for vertex in face] for face in _CUBE_FACES])
    target = grid.copy()
    grid[13] = center
    # ty: ignore[invalid-argument-type]
    mesh = phx.discretization.CellMesh.from_mixed_3d(grid, (), polyhedra={"cubes": cells})
    return mesh, target


@pytest.mark.parametrize(
    ("build", "perturbed", "free"),
    [
        (_prism_mesh, (0.7, 0.35, 0.62), 9),
        (_pyramid_mesh, (0.62, 0.41, 0.58), 8),
        (_polyhedral_mesh, (0.64, 0.4, 0.57), 13),
    ],
)
def test_prism_pyramid_and_polyhedral_optimization_improves_quality(
    build: Any, perturbed: Any, free: Any
) -> None:
    mesh, target = build(perturbed)
    fixed = np.ones((mesh.coordinates.shape[0],), dtype=np.bool_)
    fixed[free] = False
    initial = np.min(np.asarray(meshing.evaluate_cell_quality(mesh).mean_ratios))

    result = meshing.optimize_cell_mesh(
        meshing.TargetMatrixOptimizationPlan(
            mesh, target_coordinates=target, fixed_vertices=fixed
        ),
        phx.SpatialCoordinateContract.si(),
    )

    assert result.status is meshing.MeshOptimizationStatus.OPTIMIZED
    assert result.final_objective < result.initial_objective
    optimized = np.asarray(result.coordinates)
    np.testing.assert_array_equal(optimized[fixed], np.asarray(mesh.coordinates)[fixed])
    np.testing.assert_allclose(optimized[free], target[free], atol=1e-5)
    # ty: ignore[unresolved-attribute]
    final = np.min(np.asarray(result.result.quality.evaluation.mean_ratios))
    assert final > initial
