#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _positive_definite_properties():
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )


def _block_problem():
    matrix = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
        ]
    )
    operator = la.DenseLinearOperator(
        matrix,
        properties=_positive_definite_properties(),
        operator_id="block-krylov-operator",
    )
    return matrix, la.LinearSystem(operator, problem_id="block-krylov-problem")


@pytest.mark.parametrize("method", [la.BlockGMRES(restart=4), la.BlockCG()])
def test_true_block_krylov_handles_dependent_and_zero_rhs_columns(method):
    matrix, problem = _block_problem()
    first = jnp.asarray([1.0, 2.0, -1.0, 0.5])
    second = jnp.asarray([-2.0, 0.0, 1.0, 3.0])
    rhs = jnp.stack((first, second, first + second, jnp.zeros_like(first)), axis=1)
    layout = la.RHSLayout((4,))
    policy = la.LinearSolvePolicy(
        method,
        differentiation=la.DifferentiationPolicy("none"),
        tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=20),
    )
    prepared = la.prepare(problem, policy, rhs_layout=layout)
    result = la.solve(prepared, rhs)
    compiled = jax.jit(lambda values: la.solve(prepared, values).value)(rhs)
    expected = jnp.linalg.solve(matrix, rhs)

    assert jnp.allclose(result.value, expected, rtol=1e-8, atol=1e-9)
    assert jnp.allclose(compiled, expected, rtol=1e-8, atol=1e-9)
    assert jnp.all(result.successful)
    assert result.provenance.rhs_mode == "true-block"
    assert jnp.all(result.diagnostics.effective_block_rank <= rhs.shape[1])
    assert jnp.any(result.diagnostics.deflated_rhs_count >= 1)
    with pytest.raises(ValueError, match="RHS layout|right-hand side|trailing"):
        la.solve(prepared, first)


def test_scalar_multi_rhs_remains_a_distinct_pseudo_block_path():
    matrix, problem = _block_problem()
    rhs = jnp.eye(4)[:, :2]
    result = la.solve_many(
        la.prepare(
            problem,
            la.LinearSolvePolicy(
                la.GMRES(restart=4),
                differentiation=la.DifferentiationPolicy("none"),
            ),
        ),
        rhs,
    )

    assert jnp.allclose(result.value, jnp.linalg.solve(matrix, rhs), rtol=1e-8, atol=1e-9)
    assert result.provenance.rhs_mode == "pseudo-block"


def test_block_methods_require_a_nonempty_planned_rhs_layout():
    _, problem = _block_problem()
    policy = la.LinearSolvePolicy(
        la.BlockGMRES(restart=3),
        differentiation=la.DifferentiationPolicy("none"),
    )
    with pytest.raises(ValueError, match="RHS layout|block"):
        la.plan(problem, policy)
    with pytest.raises(ValueError, match="at least one"):
        la.RHSLayout(())


def test_recycled_solve_returns_immutable_state_and_refreshes_operator_images():
    matrix, problem = _block_problem()
    policy = la.LinearSolvePolicy(
        la.GMRES(restart=4),
        differentiation=la.DifferentiationPolicy("none"),
        tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=20),
        recycling=la.RecyclingPolicy(capacity=2),
    )
    prepared = la.prepare(problem, policy)
    capacity_one_cost = (
        la.plan(
            problem,
            la.LinearSolvePolicy(
                la.GMRES(restart=4),
                differentiation=la.DifferentiationPolicy("none"),
                tolerance=la.TolerancePolicy(
                    relative=1e-10,
                    absolute=1e-12,
                    max_steps=20,
                ),
                recycling=la.RecyclingPolicy(capacity=1),
            ),
        )
        .candidates[-1]
        .krylov_basis_bytes_per_rhs
    )
    capacity_two_cost = prepared.plan.candidates[-1].krylov_basis_bytes_per_rhs
    assert capacity_two_cost > capacity_one_cost
    with pytest.raises(ValueError, match="Krylov basis"):
        la.plan(
            problem,
            la.LinearSolvePolicy(
                la.GMRES(restart=4),
                differentiation=la.DifferentiationPolicy("none"),
                tolerance=la.TolerancePolicy(
                    relative=1e-10,
                    absolute=1e-12,
                    max_steps=20,
                ),
                recycling=la.RecyclingPolicy(capacity=2),
                resources=la.SolveResourcePolicy(
                    krylov_basis_bytes=capacity_two_cost - 1
                ),
            ),
        )
    first_rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0])
    second_rhs = first_rhs + jnp.asarray([0.1, 0.0, -0.2, 0.3])
    first = la.solve_recycled(prepared, first_rhs)
    second = la.solve_recycled(prepared, second_rhs, recycling=first.recycling)

    assert jnp.allclose(first.value, jnp.linalg.solve(matrix, first_rhs), rtol=1e-8)
    assert jnp.allclose(second.value, jnp.linalg.solve(matrix, second_rhs), rtol=1e-8)
    assert bool(first.successful)
    assert bool(second.successful)
    assert first.recycling is not second.recycling
    assert int(second.recycling.effective_dimension) <= 2
    assert int(second.recycling.update_count) > int(first.recycling.update_count)

    changed_matrix = matrix + 0.25 * jnp.eye(matrix.shape[0])
    changed_operator = la.DenseLinearOperator(
        changed_matrix,
        properties=_positive_definite_properties(),
        operator_id=problem.operator.operator_id,
    )
    changed_problem = la.LinearSystem(
        changed_operator,
        problem_id=problem.problem_id,
    )
    refreshed = la.refresh(prepared, changed_problem)
    refreshed_recycling = la.refresh_recycling(first.recycling, refreshed)
    active = int(refreshed_recycling.effective_dimension)
    expected_images = changed_matrix @ refreshed_recycling.source_basis[:, :active]

    assert int(refreshed_recycling.operator_numeric_version) == 1
    assert jnp.allclose(
        refreshed_recycling.image_basis[:, :active],
        expected_images,
        rtol=1e-8,
        atol=1e-9,
    )


def test_recycled_matvec_diagnostics_match_executed_operator_actions():
    matrix, _ = _block_problem()
    executed_actions = []

    def counted_action(value):
        jax.debug.callback(lambda _: executed_actions.append(None), value[0])
        return matrix @ value

    space = la.ArraySpace((4,), dtype=jnp.float64)
    problem = la.LinearSystem(
        la.FunctionLinearOperator(
            counted_action,
            source=space,
            target=space,
            operator_id="counted-recycling-operator",
        )
    )
    prepared = la.prepare(
        problem,
        la.LinearSolvePolicy(
            la.GMRES(restart=4),
            differentiation=la.DifferentiationPolicy("none"),
            tolerance=la.TolerancePolicy(
                relative=1e-10,
                absolute=1e-12,
                max_steps=8,
            ),
            recycling=la.RecyclingPolicy(capacity=2),
        ),
    )
    executed_actions.clear()
    result = la.solve_recycled(
        prepared,
        jnp.asarray([1.0, -2.0, 0.5, 3.0]),
    )
    jax.block_until_ready(result.value)

    # Result verification performs one provider-neutral action not charged to
    # iterative diagnostics.
    assert int(result.diagnostics.matvec_count) == len(executed_actions) - 1


def test_planned_layout_is_authoritative_for_one_shot_and_transformed_solves():
    matrix, problem = _block_problem()
    layout = la.RHSLayout((1, 2), names=("scenario", "member"))
    policy = la.LinearSolvePolicy(
        la.BlockGMRES(restart=4),
        differentiation=la.DifferentiationPolicy("none"),
        tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=12),
    )
    plan = la.plan(problem, policy, rhs_layout=layout)
    prepared = la.prepare(problem, plan)
    rhs = jnp.eye(4)[:, :2].reshape((4, 1, 2))
    result = la.solve(problem, rhs, policy=plan)
    transposed = la.solve_transpose(prepared, rhs)
    adjointed = la.solve_adjoint(prepared, rhs, rhs_layout=layout)
    expected = jnp.linalg.solve(matrix, rhs.reshape((4, 2))).reshape((4, 1, 2))

    assert result.value.shape == rhs.shape
    assert result.provenance.plan_id == plan.plan_id
    assert jnp.allclose(result.value, expected, rtol=1e-8, atol=1e-9)
    assert jnp.allclose(transposed.value, expected, rtol=1e-8, atol=1e-9)
    assert jnp.allclose(adjointed.value, expected, rtol=1e-8, atol=1e-9)
    with pytest.raises(ValueError, match="RHS|layout|trailing"):
        la.solve(problem, rhs.reshape((4, 2)), policy=plan)
    with pytest.raises(ValueError, match="match.*plan|layout"):
        la.solve_transpose(
            prepared,
            rhs,
            rhs_layout=la.RHSLayout((1, 2), names=("other", "member")),
        )


@pytest.mark.parametrize("method", [la.BlockGMRES(restart=3), la.BlockCG()])
@pytest.mark.parametrize("scale", [1e-20, 1e20])
def test_block_rank_deflation_is_invariant_under_nonzero_scaling(method, scale):
    matrix, problem = _block_problem()
    rhs = scale * jnp.eye(4)[:, :2]
    policy = la.LinearSolvePolicy(
        method,
        differentiation=la.DifferentiationPolicy("none"),
        tolerance=la.TolerancePolicy(relative=1e-10, absolute=0.0, max_steps=20),
    )
    result = la.solve(
        problem,
        rhs,
        policy=policy,
        rhs_layout=la.RHSLayout((2,)),
    )

    assert jnp.all(result.successful)
    assert jnp.all(result.diagnostics.effective_block_rank == 2)
    assert jnp.allclose(
        result.value,
        jnp.linalg.solve(matrix, rhs),
        rtol=1e-8,
        atol=1e-30 if scale < 1.0 else 1e5,
    )


def test_block_breakdown_reports_executed_iterations_and_empty_system_succeeds():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    zero = la.DenseLinearOperator(
        jnp.zeros((2, 2)),
        source=space,
        target=space,
    )
    result = la.solve(
        la.LinearSystem(zero),
        jnp.eye(2),
        policy=la.LinearSolvePolicy(
            la.BlockGMRES(restart=2),
            differentiation=la.DifferentiationPolicy("none"),
            tolerance=la.TolerancePolicy(
                relative=1e-10,
                absolute=0.0,
                max_steps=5,
            ),
        ),
        rhs_layout=la.RHSLayout((2,)),
    )

    assert jnp.all(result.status == int(la.LinearSolveStatus.BREAKDOWN))
    assert jnp.all(result.diagnostics.iterations == 1)

    empty_space = la.ArraySpace((0,), dtype=jnp.float64)
    empty = la.DenseLinearOperator(
        jnp.zeros((0, 0)),
        source=empty_space,
        target=empty_space,
    )
    empty_result = la.solve(
        la.LinearSystem(empty),
        jnp.zeros((0, 2)),
        policy=la.LinearSolvePolicy(
            la.BlockGMRES(restart=2),
            differentiation=la.DifferentiationPolicy("none"),
        ),
        rhs_layout=la.RHSLayout((2,)),
    )

    assert empty_result.value.shape == (0, 2)
    assert jnp.all(empty_result.successful)


def test_block_rhs_only_jvp_solves_rank_deficient_tangent_to_full_capacity():
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    operator = la.DenseLinearOperator(
        matrix,
        properties=_positive_definite_properties(),
    )
    problem = la.LinearSystem(operator)
    rhs = jnp.eye(2)
    tangent = jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
    policy = la.LinearSolvePolicy(
        la.BlockCG(),
        differentiation=la.DifferentiationPolicy("rhs-only"),
        tolerance=la.TolerancePolicy(relative=1e-12, absolute=0.0, max_steps=1),
    )
    plan = la.plan(problem, policy, rhs_layout=la.RHSLayout((2,)))

    _, derivative = jax.jvp(
        lambda values: la.solve(problem, values, policy=plan).value,
        (rhs,),
        (tangent,),
    )

    assert jnp.allclose(
        derivative,
        jnp.linalg.solve(matrix, tangent),
        rtol=1e-8,
        atol=1e-9,
    )


def test_recycled_state_is_fixed_capacity_jittable_and_explicitly_rebuildable():
    matrix, problem = _block_problem()
    policy = la.LinearSolvePolicy(
        la.GMRES(restart=4),
        differentiation=la.DifferentiationPolicy("none"),
        recycling=la.RecyclingPolicy(capacity=2, refresh="reuse-source"),
        tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=16),
    )
    prepared = la.prepare(problem, policy)
    first_rhs = jnp.asarray([1.0, -0.5, 0.25, 2.0])
    first = la.solve_recycled(prepared, first_rhs)
    second_rhs = first_rhs + jnp.asarray([0.1, 0.0, -0.05, 0.2])
    compiled = jax.jit(
        lambda values, state: la.solve_recycled(
            prepared,
            values,
            recycling=state,
        )
    )
    second = compiled(second_rhs, first.recycling)

    assert first.recycling.source_basis.shape == (4, 2)
    assert first.recycling.image_basis.shape == (4, 2)
    assert first.recycling.effective_dimension <= 2
    assert second.recycling.source_basis.shape == first.recycling.source_basis.shape
    assert second.recycling.update_count == first.recycling.update_count + 1
    assert jnp.allclose(
        second.value,
        jnp.linalg.solve(matrix, second_rhs),
        rtol=1e-8,
        atol=1e-9,
    )

    rebuilt = la.refresh_recycling(
        second.recycling,
        prepared,
        refresh="rebuild",
    )
    assert rebuilt.effective_dimension == 0
    assert rebuilt.update_count == second.recycling.update_count + 1
    assert jnp.all(rebuilt.source_basis == 0)
    assert jnp.all(rebuilt.image_basis == 0)


@pytest.mark.parametrize("method", [la.BlockGMRES(restart=2), la.BlockCG()])
def test_true_block_orthogonalization_respects_declared_pairing(method):
    space = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray([1e-4, 1e4])),
    )
    matrix = jnp.diag(jnp.asarray([2.0, 5.0]))
    operator = la.DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        properties=_positive_definite_properties(),
    )
    rhs = jnp.asarray([[1.0, 1.0], [1.0, -1.0]])
    result = la.solve(
        la.LinearSystem(operator),
        rhs,
        policy=la.LinearSolvePolicy(
            method,
            differentiation=la.DifferentiationPolicy("none"),
            tolerance=la.TolerancePolicy(
                relative=1e-12,
                absolute=0.0,
                max_steps=4,
            ),
        ),
        rhs_layout=la.RHSLayout((2,)),
    )

    assert jnp.all(result.successful)
    assert jnp.all(result.diagnostics.effective_block_rank == 2)
    assert jnp.allclose(
        result.value,
        jnp.linalg.solve(matrix, rhs),
        rtol=1e-9,
        atol=1e-10,
    )


@pytest.mark.parametrize("method", [la.BlockGMRES(restart=3), la.BlockCG()])
def test_true_block_rank_is_invariant_to_independent_rhs_column_scales(method):
    matrix, problem = _block_problem()
    scales = jnp.asarray([1e-20, 1e20])
    rhs = jnp.eye(4)[:, :2] * scales[None, :]
    result = la.solve(
        problem,
        rhs,
        policy=la.LinearSolvePolicy(
            method,
            differentiation=la.DifferentiationPolicy("none"),
            tolerance=la.TolerancePolicy(
                relative=1e-10,
                absolute=0.0,
                max_steps=20,
            ),
        ),
        rhs_layout=la.RHSLayout((2,)),
    )
    expected = jnp.linalg.solve(matrix, rhs)
    relative_errors = jnp.linalg.norm(result.value - expected, axis=0) / jnp.linalg.norm(
        expected,
        axis=0,
    )

    assert jnp.all(result.successful)
    assert jnp.all(result.diagnostics.effective_block_rank == 2)
    assert jnp.all(relative_errors < 1e-8)
