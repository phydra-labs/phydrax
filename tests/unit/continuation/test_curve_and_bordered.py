#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_parameter_path_exposes_declared_pytree_curve_and_jvp():
    problem = phx.continuation.ParameterPathContinuationProblem(
        lambda state, parameters, args: {
            "x": state["x"] - parameters["offset"] - parameters["quadratic"]
        },
        lambda coordinate, args: {
            "offset": coordinate + args["shift"],
            "quadratic": coordinate**2,
        },
        {
            "offset": jnp.asarray(0.0),
            "quadratic": jnp.asarray(0.0),
        },
        coordinate_lower=-1.0,
        coordinate_upper=1.0,
        problem_id="pytree-parameter-path",
    )
    args = {"shift": jnp.asarray(0.25)}

    parameters, tangent = problem.parameters_jvp(
        jnp.asarray(0.5),
        jnp.asarray(2.0),
        args,
    )

    np.testing.assert_allclose(parameters["offset"], 0.75)
    np.testing.assert_allclose(parameters["quadratic"], 0.25)
    np.testing.assert_allclose(tangent["offset"], 2.0)
    np.testing.assert_allclose(tangent["quadratic"], 2.0)
    residual = problem.residual(
        {"x": jnp.asarray(1.0)},
        jnp.asarray(0.5),
        args,
    )
    np.testing.assert_allclose(residual["x"], 0.0)
    assert bool(problem.contains_coordinate(jnp.asarray(1.0)))
    assert not bool(problem.contains_coordinate(jnp.asarray(1.1)))

    result = phx.continuation.continue_branch(
        problem,
        {"x": jnp.asarray(0.25)},
        jnp.asarray(0.0),
        num_steps=1,
        method=phx.continuation.NaturalParameterContinuation(
            initial_step=0.1,
            minimum_step=0.1,
            maximum_step=0.1,
        ),
        args=args,
    )
    point = result.points[-1]
    expected_parameters, expected_tangent = problem.parameters_jvp(
        point.coordinate,
        point.tangent_coordinate,
        args,
    )
    assert int(result.status) == int(phx.continuation.ContinuationStatus.SUCCESS)
    assert jax.tree.structure(point.parameters) == jax.tree.structure(expected_parameters)
    np.testing.assert_allclose(
        point.parameters["offset"],
        expected_parameters["offset"],
    )
    np.testing.assert_allclose(
        point.parameters["quadratic"],
        expected_parameters["quadratic"],
    )
    np.testing.assert_allclose(
        point.tangent_parameters["offset"],
        expected_tangent["offset"],
    )
    np.testing.assert_allclose(
        point.tangent_parameters["quadratic"],
        expected_tangent["quadratic"],
    )


def test_parameter_path_rejects_structure_dtype_and_nonfinite_drift():
    template = {"physical": jnp.asarray([0.0, 0.0])}
    wrong_structure = phx.continuation.ParameterPathContinuationProblem(
        lambda state, parameters, args: state,
        lambda coordinate, args: {"other": jnp.asarray([coordinate, coordinate])},
        template,
    )
    wrong_dtype = phx.continuation.ParameterPathContinuationProblem(
        lambda state, parameters, args: state,
        lambda coordinate, args: {
            "physical": jnp.asarray([coordinate, coordinate], dtype=jnp.float32)
        },
        {"physical": jnp.asarray([0.0, 0.0], dtype=jnp.float64)},
    )
    nonfinite = phx.continuation.ParameterPathContinuationProblem(
        lambda state, parameters, args: state,
        lambda coordinate, args: {
            "physical": jnp.asarray([coordinate, jnp.inf], dtype=coordinate.dtype)
        },
        template,
    )

    with pytest.raises(ValueError, match="same PyTree structure"):
        wrong_structure.parameters(jnp.asarray(0.0))
    with pytest.raises(TypeError, match="leaf dtypes"):
        wrong_dtype.parameters(jnp.asarray(0.0))
    with pytest.raises(RuntimeError, match="must be finite"):
        nonfinite.parameters(jnp.asarray(0.0))


def _bordered_system(matrix, column, row, corner):
    source_space = phx.linalg.PyTreeSpace(
        {"x": jnp.zeros((2,), dtype=jnp.float64)},
        space_id="bordered-source-space",
    )
    target_space = phx.linalg.PyTreeSpace(
        {"f": jnp.zeros((2,), dtype=jnp.float64)},
        space_id="bordered-target-space",
    )
    operator = phx.linalg.DenseLinearOperator(
        jnp.asarray(matrix, dtype=jnp.float64),
        source=source_space,
        target=target_space,
        operator_id="bordered-principal-operator",
    )
    return phx.continuation.BorderedLinearSystem(
        operator,
        {"f": jnp.asarray(column, dtype=jnp.float64)},
        {"x": jnp.asarray(row, dtype=jnp.float64)},
        jnp.asarray(corner, dtype=jnp.float64),
        system_id="bordered-system",
    )


def test_prepared_bordered_solve_reuses_cached_column_across_rhs_and_refresh():
    system = _bordered_system(
        [[2.0, 0.0], [0.0, 3.0]],
        [1.0, -1.0],
        [4.0, 0.5],
        2.0,
    )
    plan = phx.continuation.plan_bordered_solve(
        system,
        phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU()),
        schur_tolerance=1e-12,
        plan_id="bordered-plan",
    )
    prepared = phx.continuation.prepare_bordered_solve(system, plan)

    rhs_values = (
        (jnp.asarray([5.0, 7.0]), jnp.asarray(11.0)),
        (jnp.asarray([-1.0, 2.0]), jnp.asarray(3.0)),
    )
    dense_system = np.block(
        [
            [np.asarray([[2.0, 0.0], [0.0, 3.0]]), np.asarray([[1.0], [-1.0]])],
            [np.asarray([[4.0, 0.5]]), np.asarray([[2.0]])],
        ]
    )
    for principal_rhs, border_rhs in rhs_values:
        result = phx.continuation.solve_bordered(
            prepared,
            {"f": principal_rhs},
            border_rhs,
        )
        expected = np.linalg.solve(
            dense_system,
            np.concatenate([np.asarray(principal_rhs), np.asarray([border_rhs])]),
        )
        assert bool(result.successful)
        np.testing.assert_allclose(result.value.primal["x"], expected[:2])
        np.testing.assert_allclose(result.value.scalar, expected[2])
        assert bool(result.diagnostics.cached_column_solve_reused)
        assert int(result.diagnostics.principal_solve_count) == 1
        assert float(result.diagnostics.residual_norm) < 1e-12
        assert result.provenance.prepared_id == prepared.prepared_id

    refreshed_system = _bordered_system(
        [[4.0, 0.0], [0.0, 5.0]],
        [2.0, 1.0],
        [4.0, 0.5],
        2.0,
    )
    refreshed = phx.continuation.refresh_bordered_solve(prepared, refreshed_system)
    assert refreshed.prepared_id == prepared.prepared_id
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert int(refreshed.numeric_version) == int(prepared.numeric_version) + 1
    assert not np.allclose(
        np.asarray(refreshed.inverse_column["x"]),
        np.asarray(prepared.inverse_column["x"]),
    )


def test_bordered_refresh_preserves_preconditioner_plan_across_branch_steps():
    space = phx.linalg.PyTreeSpace(
        {"x": jnp.zeros((2,), dtype=jnp.float64)},
        space_id="bordered-reuse-space",
    )

    def system_for(matrix, column):
        operator = phx.linalg.DenseLinearOperator(
            jnp.asarray(matrix, dtype=jnp.float64),
            source=space,
            target=space,
            operator_id="bordered-reuse-principal",
        )
        return phx.continuation.BorderedLinearSystem(
            operator,
            {"x": jnp.asarray(column, dtype=jnp.float64)},
            {"x": jnp.asarray([4.0, 0.5], dtype=jnp.float64)},
            jnp.asarray(2.0, dtype=jnp.float64),
            system_id="bordered-reuse-system",
        )

    system = system_for([[2.0, 0.0], [0.0, 3.0]], [1.0, -1.0])
    plan = phx.continuation.plan_bordered_solve(
        system,
        phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(),
            preconditioning=phx.linalg.PreconditioningPolicy(
                phx.linalg.JacobiPreconditionerBuilder()
            ),
        ),
        plan_id="bordered-reuse-plan",
    )
    prepared = phx.continuation.prepare_bordered_solve(system, plan)
    refreshed = phx.continuation.refresh_bordered_solve(
        prepared,
        system_for([[4.0, 0.0], [0.0, 5.0]], [2.0, 1.0]),
    )
    result = phx.continuation.solve_bordered(
        refreshed,
        {"x": jnp.asarray([2.0, 3.0])},
        jnp.asarray(4.0),
    )

    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id == prepared.prepared_id
    assert int(refreshed.numeric_version) == int(prepared.numeric_version) + 1
    assert (
        refreshed.principal.preconditioning_state.plan.plan_id
        == prepared.principal.preconditioning_state.plan.plan_id
    )
    assert result.provenance.principal_plan_id == plan.principal_plan.plan_id
    assert bool(result.diagnostics.cached_column_solve_reused)
    assert int(result.diagnostics.principal_solve_count) == 1


def test_bordered_singular_schur_is_explicit_and_skips_rhs_solve():
    system = _bordered_system(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 0.0],
        [1.0, 0.0],
        1.0,
    )
    plan = phx.continuation.plan_bordered_solve(
        system,
        phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU()),
    )
    prepared = phx.continuation.prepare_bordered_solve(system, plan)
    result = phx.continuation.solve_bordered(
        prepared,
        {"f": jnp.asarray([2.0, 3.0])},
        jnp.asarray(4.0),
    )

    assert int(prepared.status) == int(
        phx.continuation.BorderedSolveStatus.SCHUR_SINGULAR
    )
    assert int(result.status) == int(phx.continuation.BorderedSolveStatus.SCHUR_SINGULAR)
    assert int(result.diagnostics.principal_solve_count) == 0
    assert not bool(result.diagnostics.cached_column_solve_reused)


def test_prepared_bordered_solve_is_filter_jittable_for_success_and_failure():
    system = _bordered_system(
        [[2.0, 0.0], [0.0, 3.0]],
        [1.0, -1.0],
        [4.0, 0.5],
        2.0,
    )
    plan = phx.continuation.plan_bordered_solve(
        system,
        phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU()),
    )
    prepared = phx.continuation.prepare_bordered_solve(system, plan)
    compiled_solve = eqx.filter_jit(phx.continuation.solve_bordered)

    result = compiled_solve(
        prepared,
        {"f": jnp.asarray([5.0, 7.0])},
        jnp.asarray(11.0),
    )

    assert int(result.status) == int(phx.continuation.BorderedSolveStatus.SUCCESS)
    assert int(result.diagnostics.principal_solve_count) == 1
    assert bool(result.diagnostics.cached_column_solve_reused)

    singular_system = _bordered_system(
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 0.0],
        [1.0, 0.0],
        1.0,
    )
    singular_plan = phx.continuation.plan_bordered_solve(
        singular_system,
        phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU()),
    )
    singular_prepared = phx.continuation.prepare_bordered_solve(
        singular_system,
        singular_plan,
    )
    failed = compiled_solve(
        singular_prepared,
        {"f": jnp.asarray([2.0, 3.0])},
        jnp.asarray(4.0),
    )

    assert int(failed.status) == int(phx.continuation.BorderedSolveStatus.SCHUR_SINGULAR)
    assert int(failed.diagnostics.principal_solve_count) == 0
    assert not bool(failed.diagnostics.cached_column_solve_reused)
