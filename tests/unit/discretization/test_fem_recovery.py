from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


fem = phx.discretization.fem


def _square(count: Any) -> Any:
    axis = np.linspace(0.0, 1.0, count + 1)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((first.ravel(), second.ravel()), axis=1)
    rows = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    lower = rows[:-1, :-1].ravel()
    right = rows[1:, :-1].ravel()
    upper = rows[1:, 1:].ravel()
    left = rows[:-1, 1:].ravel()
    cells = np.concatenate(
        (np.stack((lower, right, upper), 1), np.stack((lower, upper, left), 1))
    )
    return phx.discretization.CellMesh.from_triangles(points, cells.astype(np.int32))


def _discretization(mesh: Any, degree: Any) -> Any:
    element = phx.discretization.lagrange_element("triangle", degree)
    field = phx.discretization.FiniteElementFieldSpec("u", element)
    return phx.discretization.FiniteElementPlan(mesh, field).prepare()


def _local_poisson(discretization: Any, source: Any) -> Any:
    geometry = discretization.block_geometries[0][0]
    weights = np.asarray(geometry.physical_weights)
    gradients = np.asarray(geometry.physical_gradients)
    stiffness = np.einsum("cq,cqld,cqmd->clm", weights, gradients, gradients)
    load = np.einsum(
        "cq,cq,ql->cl",
        weights,
        source(np.asarray(geometry.physical_points)),
        np.asarray(geometry.basis_values),
    )
    return stiffness, load


def _assemble(discretization: Any, local_matrix: Any, local_vector: Any) -> Any:
    dofs = np.asarray(discretization.dof_maps[0].cell_dofs[0])
    count = discretization.dof_maps[0].global_dof_count
    matrix = np.zeros((count, count))
    vector = np.zeros((count,))
    np.add.at(matrix, (dofs[:, :, None], dofs[:, None, :]), local_matrix)
    np.add.at(vector, dofs, local_vector)
    return matrix, vector


def _sine_source(points: Any) -> Any:
    return (
        2.0 * np.pi**2 * np.sin(np.pi * points[..., 0]) * np.sin(np.pi * points[..., 1])
    )


def test_fem_recovery_scenario_1() -> None:
    discretization = _discretization(_square(4), 2)
    prepared = fem.prepare_gradient_recovery(discretization, "u")
    nodes = np.asarray(discretization.dof_maps[0].dof_coordinates)
    x, y = nodes[:, 0], nodes[:, 1]
    field = x**2 + 3.0 * x * y - 2.0 * y**2 + x - 5.0 * y

    gradient, gradient_evidence = fem.recover_gradient(prepared, field)
    hessian, hessian_evidence = fem.recover_hessian(prepared, field)

    np.testing.assert_allclose(
        gradient,
        np.stack((2.0 * x + 3.0 * y + 1.0, 3.0 * x - 4.0 * y - 5.0), 1),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        hessian, np.broadcast_to(((2.0, 3.0), (3.0, -4.0)), hessian.shape), atol=1e-9
    )
    assert gradient_evidence.passed and hessian_evidence.passed
    assert hessian_evidence.extended_patch_count > 0
    assert hessian_evidence.maximum_asymmetry < 1e-8
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="ill conditioned"):
        fem.prepare_gradient_recovery(_discretization(mesh, 2), "u")
    coarse = _effectivity(8)
    fine = _effectivity(16)
    assert 0.9 < fine < 1.3
    assert abs(fine - 1.0) < abs(coarse - 1.0)
    mesh = _square(6)
    base = _discretization(mesh, 1)
    enriched = _discretization(mesh, 2)
    local_matrix, local_load = _local_poisson(enriched, _sine_source)
    matrix, load = _assemble(enriched, local_matrix, local_load)
    base_dofs = np.asarray(base.dof_maps[0].cell_dofs[0])
    enriched_dofs = np.asarray(enriched.dof_maps[0].cell_dofs[0])
    base_element = phx.discretization.lagrange_element("triangle", 1)
    values, _ = base_element.tabulate(enriched.elements[0][0].reference_nodes)
    prolongation = np.zeros((matrix.shape[0], base.dof_maps[0].global_dof_count))
    prolongation[enriched_dofs[:, :, None], base_dofs[:, None, :]] = np.asarray(values)[
        None
    ]
    free = ~np.asarray(enriched.dof_maps[0].boundary_dof_mask)
    base_free = ~np.asarray(base.dof_maps[0].boundary_dof_mask)
    # Galerkin-consistent base problem: the enriched system restricted to P1.
    coarse = prolongation[:, base_free]
    base_solution = np.zeros(base.dof_maps[0].global_dof_count)
    base_solution[base_free] = np.linalg.solve(
        coarse.T @ matrix @ coarse, coarse.T @ load
    )
    lifted = prolongation @ base_solution
    enriched_solution = np.zeros_like(load)
    enriched_solution[free] = np.linalg.solve(matrix[np.ix_(free, free)], load[free])
    _, goal = _assemble(
        enriched,
        local_matrix,
        _local_poisson(enriched, lambda points: np.ones(points.shape[:-1]))[1],
    )
    residual = local_load - np.einsum("clm,cm->cl", local_matrix, lifted[enriched_dofs])
    system = phx.linalg.LinearSystem(
        phx.linalg.DenseLinearOperator(jnp.asarray(matrix[np.ix_(free, free)]))
    )

    indicators, adjoint = fem.dual_weighted_residual_indicators(
        enriched,
        "u",
        system,
        jnp.asarray(goal[free]),
        residual,
        base_degree=1,
        free_dofs=np.flatnonzero(free),
    )

    assert bool(jnp.all(adjoint.successful))
    np.testing.assert_allclose(
        float(indicators.global_estimate),
        goal @ (enriched_solution - lifted),
        rtol=1e-9,
    )


def _p1_poisson(count: Any) -> Any:
    discretization = _discretization(_square(count), 1)
    matrix, load = _assemble(
        discretization, *_local_poisson(discretization, _sine_source)
    )
    free = ~np.asarray(discretization.dof_maps[0].boundary_dof_mask)
    solution = np.zeros_like(load)
    solution[free] = np.linalg.solve(matrix[np.ix_(free, free)], load[free])
    return discretization, solution


def _effectivity(count: Any) -> Any:
    discretization, solution = _p1_poisson(count)
    prepared = fem.prepare_gradient_recovery(discretization, "u")
    estimate, evidence = fem.recovery_error_estimate(prepared, solution)
    points, weights = fem._generic._degree_aware_reference_rule("triangle", 6)
    geometry = discretization.evaluate_block_geometry(
        "u", 0, discretization.default_runtime.coordinates, points, weights
    )
    x = np.asarray(geometry.physical_points)
    exact = np.pi * np.stack(
        (
            np.cos(np.pi * x[..., 0]) * np.sin(np.pi * x[..., 1]),
            np.sin(np.pi * x[..., 0]) * np.cos(np.pi * x[..., 1]),
        ),
        -1,
    )
    dofs = np.asarray(discretization.dof_maps[0].cell_dofs[0])
    discrete = np.einsum(
        "cqld,cl->cqd", np.asarray(geometry.physical_gradients), solution[dofs]
    )
    error = np.sqrt(
        np.sum(
            np.asarray(geometry.physical_weights) * np.sum((exact - discrete) ** 2, -1)
        )
    )
    assert evidence.passed
    return float(estimate.global_estimate) / error
