import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_target_matrix_optimization_improves_distorted_mesh_without_moving_boundary():
    target = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)))
    distorted = target.copy()
    distorted[4] = (0.75, 0.25)
    cells = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32)
    mesh = phx.discretization.CellMesh.from_triangles(distorted, cells)
    plan = phx.meshing.TargetMatrixOptimizationPlan(
        mesh,
        target_coordinates=target,
        fixed_vertices=np.asarray((True, True, True, True, False)),
        maximum_iterations=30,
        initial_step_size=0.02,
    )

    result = phx.meshing.optimize_cell_mesh(
        plan,
        phx.SpatialCoordinateContract.si(),
    )

    assert result.accepted_steps > 0
    assert result.final_objective < result.initial_objective
    np.testing.assert_allclose(result.result.mesh.coordinates[:4], target[:4])
    assert result.result.quality.minimum_mean_ratio > 0.0


def test_generic_high_order_coordinate_optimizer_preserves_fixed_nodes():
    element = phx.discretization.lagrange_element("triangle", 2)
    coordinates = np.array(element.reference_nodes, copy=True)
    coordinates[3:] += 0.1
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": element},
        {"triangles": np.arange(6, dtype=np.int32)[None, :]},
        coordinates,
    )
    target = jnp.asarray(element.reference_nodes)

    optimized = phx.meshing.optimize_cell_geometry_coordinates(
        geometry,
        lambda values: jnp.sum((values - target) ** 2),
        fixed_coordinates=np.asarray((True, True, True, False, False, False)),
        maximum_iterations=25,
        initial_step_size=0.1,
    )

    np.testing.assert_allclose(optimized[:3], coordinates[:3])
    assert float(jnp.linalg.norm(optimized[3:] - target[3:])) < float(
        jnp.linalg.norm(jnp.asarray(coordinates[3:]) - target[3:])
    )
