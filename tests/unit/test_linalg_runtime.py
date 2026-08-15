#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
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


class _DensePairing(la.AbstractPairing):
    matrix: jax.Array

    def __init__(self, matrix):
        self.matrix = jnp.asarray(matrix)
        self.pairing_id = "test-dense-pairing"

    def inner(self, left, right, /):
        return jnp.vdot(left, self.matrix @ right)

    def riesz(self, vector, /):
        return self.matrix @ vector

    def inverse_riesz(self, covector, /):
        return jnp.linalg.solve(self.matrix, covector)


def test_structured_spaces_pairings_and_duals_preserve_coordinate_semantics():
    weights = {
        "field": jnp.asarray([2.0, 3.0]),
        "parameter": jnp.asarray(5.0),
    }
    space = la.PyTreeSpace(
        {
            "field": jnp.zeros((2,), dtype=jnp.float64),
            "parameter": jnp.zeros((), dtype=jnp.float64),
        },
        pairing=la.DiagonalPairing(weights),
    )
    left = {
        "field": jnp.asarray([1.0, -2.0]),
        "parameter": jnp.asarray(0.5),
    }
    right = {
        "field": jnp.asarray([3.0, 4.0]),
        "parameter": jnp.asarray(-1.0),
    }

    coordinates = space.flatten(left)
    rebuilt = space.unflatten(coordinates)
    expected_inner = sum(
        jnp.vdot(x, weight * y)
        for x, weight, y in zip(
            jax.tree.leaves(left),
            jax.tree.leaves(weights),
            jax.tree.leaves(right),
            strict=True,
        )
    )
    dual = la.DualSpace(space)
    expected_dual_inner = sum(
        jnp.vdot(x, y / weight)
        for x, weight, y in zip(
            jax.tree.leaves(left),
            jax.tree.leaves(weights),
            jax.tree.leaves(right),
            strict=True,
        )
    )

    assert jax.tree.all(jax.tree.map(jnp.array_equal, rebuilt, left))
    assert jnp.allclose(space.inner(left, right), expected_inner)
    assert jnp.allclose(dual.inner(left, right), expected_dual_inner)
    assert jax.tree.all(
        jax.tree.map(
            jnp.allclose,
            dual.inverse_riesz(dual.riesz(left)),
            left,
        )
    )


def test_operator_algebra_materialization_and_pairing_aware_adjoint():
    source = la.ArraySpace(
        (3,),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(jnp.asarray([2.0, 3.0, 5.0])),
    )
    target = la.ArraySpace(
        (2,),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(jnp.asarray([7.0, 11.0])),
    )
    matrix = jnp.asarray([[1.0 + 2.0j, -3.0j, 0.5], [2.0, -1.0 + 1.0j, 4.0j]])
    operator = la.DenseLinearOperator(matrix, source=source, target=target)
    left = jnp.asarray([1.0 - 1.0j, 2.0, -0.5 + 3.0j])
    right = jnp.asarray([0.25 + 2.0j, -1.0j])

    assert jnp.allclose(
        target.inner(operator.mv(left), right),
        source.inner(left, operator.adjoint_mv(right)),
    )
    assert jnp.allclose(la.transpose(operator).mv(right), matrix.T @ right)
    assert jnp.allclose(
        la.materialize(operator, la.MaterializationPolicy(max_entries=6)),
        matrix,
    )

    endomorphism = la.DenseLinearOperator(
        jnp.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=jnp.float64)
    )
    identity = la.IdentityLinearOperator(endomorphism.source)
    combined = (endomorphism + 2.0 * identity) @ endomorphism
    expected = (endomorphism.matrix + 2.0 * jnp.eye(2)) @ endomorphism.matrix
    assert jnp.allclose(
        combined.mv(jnp.asarray([1.0, -2.0])), expected @ jnp.asarray([1.0, -2.0])
    )
    assert jnp.allclose(
        la.materialize(combined, la.MaterializationPolicy()),
        expected,
    )


def test_block_and_jacobian_operators_share_the_solve_runtime():
    vector_space = la.ArraySpace((2,), dtype=jnp.float64)
    scalar_space = la.ArraySpace((1,), dtype=jnp.float64)
    block_space = la.BlockSpace((vector_space, scalar_space))
    block = la.BlockLinearOperator(
        (
            (
                la.DenseLinearOperator(
                    jnp.asarray([[4.0, 1.0], [1.0, 3.0]]),
                    source=vector_space,
                    target=vector_space,
                ),
                la.DenseLinearOperator(
                    jnp.asarray([[1.0], [0.0]]),
                    source=scalar_space,
                    target=vector_space,
                ),
            ),
            (
                la.DenseLinearOperator(
                    jnp.asarray([[1.0, 0.0]]),
                    source=vector_space,
                    target=scalar_space,
                ),
                la.DenseLinearOperator(
                    jnp.asarray([[2.0]]),
                    source=scalar_space,
                    target=scalar_space,
                ),
            ),
        ),
        source=block_space,
        target=block_space,
    )
    dense_block = jnp.asarray([[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]])
    rhs = (jnp.asarray([1.0, -2.0]), jnp.asarray([0.5]))
    block_result = la.solve(la.LinearSystem(block), rhs)
    expected_block = jnp.linalg.solve(dense_block, block_space.flatten(rhs))
    assert bool(block_result.successful)
    assert block_result.provenance.backend == "jax-dense"
    assert jnp.allclose(block_space.flatten(block_result.value), expected_block)
    assert jnp.allclose(
        la.materialize(block, la.MaterializationPolicy(max_entries=9)),
        dense_block,
    )

    tree_space = la.PyTreeSpace({"field": jnp.zeros((2,)), "parameter": jnp.zeros(())})
    tree_matrix = jnp.asarray([[3.0, 0.5, 1.0], [0.5, 2.0, -0.25], [1.0, -0.25, 4.0]])
    tree_rhs = {"field": jnp.asarray([1.0, -2.0]), "parameter": jnp.asarray(0.5)}
    tree_result = la.solve(
        la.LinearSystem(
            la.DenseLinearOperator(
                tree_matrix,
                source=tree_space,
                target=tree_space,
            )
        ),
        tree_rhs,
    )
    assert jax.tree.all(
        jax.tree.map(
            jnp.allclose,
            tree_result.value,
            tree_space.unflatten(
                jnp.linalg.solve(tree_matrix, tree_space.flatten(tree_rhs))
            ),
        )
    )

    tensor_space = la.ArraySpace((2, 2), dtype=jnp.float64)
    tensor_matrix = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    tensor_rhs = jnp.asarray([[1.0, 4.0], [9.0, 16.0]])
    tensor_result = la.solve(
        la.LinearSystem(
            la.DenseLinearOperator(
                tensor_matrix,
                source=tensor_space,
                target=tensor_space,
            )
        ),
        tensor_rhs,
    )
    assert tensor_result.value.shape == tensor_space.shape
    assert jnp.allclose(tensor_result.value, jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))

    point = jnp.asarray([2.0, -1.0])
    linearization = la.prepare_linearization(
        lambda value: jnp.asarray([value[0] ** 2 + 3.0 * value[1], value[0] * value[1]]),
        point,
        source=vector_space,
        target=vector_space,
    )
    jacobian = la.JacobianLinearOperator(linearization)
    expected_jacobian = jnp.asarray([[4.0, 3.0], [-1.0, 2.0]])
    tangent = jnp.asarray([0.25, -0.5])
    cotangent = jnp.asarray([1.5, -2.0])
    assert jnp.allclose(jacobian.mv(tangent), expected_jacobian @ tangent)
    assert jnp.allclose(
        jacobian.transpose_mv(cotangent),
        expected_jacobian.T @ cotangent,
    )
    assert jnp.allclose(
        la.solve(la.LinearSystem(jacobian), cotangent).value,
        jnp.linalg.solve(expected_jacobian, cotangent),
    )


def test_dense_prepare_solve_many_update_batches_and_jit():
    properties = _positive_definite_properties()
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    problem = la.LinearSystem(
        la.DenseLinearOperator(
            matrix,
            properties=properties,
            operator_id="refreshable-spd",
        )
    )
    selected = la.plan(problem)
    prepared = la.prepare(problem, selected)
    right_hand_sides = jnp.asarray([[1.0, 2.0, -1.0], [3.0, 0.5, 4.0]])
    result = la.solve_many(prepared, right_hand_sides)

    assert selected.backend == "jax-dense"
    assert selected.method == "dense-cholesky"
    assert result.value.shape == right_hand_sides.shape
    assert jnp.all(result.successful)
    assert jnp.allclose(result.value, jnp.linalg.solve(matrix, right_hand_sides))
    assert result.provenance.plan_id == selected.plan_id
    assert result.provenance.problem_id == problem.problem_id
    assert jnp.allclose(
        jax.jit(lambda rhs: la.solve(prepared, rhs).value)(right_hand_sides),
        result.value,
    )

    changed_matrix = jnp.asarray([[5.0, 0.5], [0.5, 2.0]])
    changed_problem = la.LinearSystem(
        la.DenseLinearOperator(
            changed_matrix,
            properties=properties,
            operator_id="refreshable-spd",
        )
    )
    refreshed = la.refresh(prepared, changed_problem)
    refreshed_result = la.solve(refreshed, right_hand_sides)
    assert refreshed.numeric_version == prepared.numeric_version + 1
    with pytest.raises(ValueError, match="symbolic solve plan"):
        la.refresh(
            prepared,
            la.LinearSystem(
                la.DenseLinearOperator(
                    changed_matrix,
                    operator_id="refreshable-spd",
                )
            ),
        )
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert jnp.allclose(
        refreshed_result.value,
        jnp.linalg.solve(changed_matrix, right_hand_sides),
    )

    batched_matrix = jnp.stack((matrix, changed_matrix))
    batched_problem = la.LinearSystem(
        la.DenseLinearOperator(batched_matrix, properties=properties)
    )
    shared_rhs = jnp.asarray([2.0, -1.0])
    batched = la.solve(batched_problem, shared_rhs)
    expected_batched = jax.vmap(jnp.linalg.solve)(
        batched_matrix,
        jnp.broadcast_to(shared_rhs, (2, 2)),
    )
    assert batched.value.shape == (2, 2)
    assert batched.status.shape == (2,)
    assert jnp.allclose(batched.value, expected_batched)

    compiled = jax.jit(
        lambda coefficients, rhs: (
            la.solve(
                la.LinearSystem(
                    la.DenseLinearOperator(coefficients, properties=properties)
                ),
                rhs,
            ).value
        )
    )
    assert jnp.allclose(
        compiled(matrix, shared_rhs), jnp.linalg.solve(matrix, shared_rhs)
    )


def test_cholesky_and_cg_respect_non_euclidean_positive_definiteness():
    metric = jnp.asarray([1.0, 4.0])
    space = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(metric),
    )
    riesz_symmetric = jnp.asarray([[2.0, 1.0], [1.0, 2.0]])
    matrix = riesz_symmetric / metric[:, None]
    properties = _positive_definite_properties()
    rhs = jnp.asarray([1.0, -2.0])
    expected = jnp.linalg.solve(matrix, rhs)

    dense = la.solve(
        la.LinearSystem(
            la.DenseLinearOperator(
                matrix,
                source=space,
                target=space,
                properties=properties,
            )
        ),
        rhs,
        policy=la.LinearSolvePolicy(la.DenseCholesky()),
    )
    function = la.FunctionLinearOperator(
        lambda vector: matrix @ vector,
        source=space,
        target=space,
        transpose_action=lambda vector: matrix.T @ vector,
        properties=properties,
    )
    iterative = la.solve(
        la.LinearSystem(function),
        rhs,
        policy=la.LinearSolvePolicy(
            la.ConjugateGradient(),
            tolerance=la.TolerancePolicy(
                relative=1e-11,
                absolute=1e-12,
                max_steps=20,
            ),
        ),
    )

    assert bool(dense.successful)
    assert bool(iterative.successful)
    assert jnp.allclose(dense.value, expected)
    assert jnp.allclose(iterative.value, expected, rtol=1e-9, atol=1e-10)


def test_weighted_regularized_least_squares_and_minimum_norm():
    matrix = jnp.asarray(
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
        dtype=jnp.float64,
    )
    targets = jnp.asarray(
        [[1.0, 0.0], [2.0, 1.0], [2.5, 1.5], [4.0, 2.0]],
        dtype=jnp.float64,
    )
    weights = jnp.asarray([1.0, 0.5, 2.0, 1.5])
    regularizer = la.DiagonalLinearOperator(jnp.asarray([0.2, 0.4]))
    problem = la.LeastSquaresProblem(
        la.DenseLinearOperator(matrix),
        weights=weights,
        regularizer=regularizer,
    )
    result = la.solve(
        problem,
        targets,
        policy=la.LinearSolvePolicy(la.DenseSVD()),
    )
    design = jnp.concatenate(
        (
            jnp.sqrt(weights)[:, None] * matrix,
            la.materialize(regularizer, la.MaterializationPolicy()),
        ),
        axis=0,
    )
    transformed_targets = jnp.concatenate(
        (jnp.sqrt(weights)[:, None] * targets, jnp.zeros((2, 2))),
        axis=0,
    )
    expected = jnp.linalg.lstsq(design, transformed_targets, rcond=None)[0]

    assert jnp.all(result.successful)
    assert jnp.allclose(result.value, expected)
    assert jnp.all(jnp.isfinite(result.diagnostics.normal_residual_norm))
    assert jnp.all(result.diagnostics.rank == 2)

    source = la.ArraySpace(
        (3,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray([4.0, 9.0, 16.0])),
    )
    underdetermined = jnp.asarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    rhs = jnp.asarray([2.0, -1.0])
    minimum_norm = la.solve(
        la.MinimumNormProblem(la.DenseLinearOperator(underdetermined, source=source)),
        rhs,
        policy=la.LinearSolvePolicy(la.DenseSVD()),
    )
    inverse_metric = jnp.diag(jnp.asarray([0.25, 1.0 / 9.0, 1.0 / 16.0]))
    expected_minimum_norm = (
        inverse_metric
        @ underdetermined.T
        @ jnp.linalg.solve(
            underdetermined @ inverse_metric @ underdetermined.T,
            rhs,
        )
    )
    assert bool(minimum_norm.successful)
    assert jnp.allclose(underdetermined @ minimum_norm.value, rhs)
    assert jnp.allclose(minimum_norm.value, expected_minimum_norm)


def test_rank_policy_retains_solution_but_reports_rank_deficiency():
    matrix = jnp.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    rhs = jnp.asarray([2.0, 4.0, 6.0])
    problem = la.LeastSquaresProblem(la.DenseLinearOperator(matrix))
    permissive = la.solve(
        problem,
        rhs,
        policy=la.LinearSolvePolicy(la.DenseSVD()),
    )
    strict = la.solve(
        problem,
        rhs,
        policy=la.LinearSolvePolicy(
            la.DenseSVD(),
            rank=la.RankPolicy(require_full_rank=True),
        ),
    )

    assert bool(permissive.successful)
    assert strict.status == int(la.LinearSolveStatus.RANK_DEFICIENT)
    assert jnp.all(jnp.isfinite(strict.value))
    assert jnp.allclose(matrix @ strict.value, rhs)
    assert strict.diagnostics.rank == 1


def test_matrix_free_iterative_system_and_least_squares_backends():
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    space = la.ArraySpace((2,), dtype=jnp.float64)
    operator = la.FunctionLinearOperator(
        lambda vector: matrix @ vector,
        source=space,
        target=space,
        transpose_action=lambda vector: matrix.T @ vector,
        properties=_positive_definite_properties(),
    )
    rhs = jnp.asarray([1.0, 2.0])
    cg = la.solve(
        la.LinearSystem(operator),
        rhs,
        policy=la.LinearSolvePolicy(
            la.ConjugateGradient(),
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=20),
            preconditioning=la.PreconditioningPolicy(
                la.DiagonalPreconditioner(
                    jnp.diag(matrix),
                    space=space,
                )
            ),
        ),
    )
    assert cg.provenance.backend == "lineax"
    assert bool(cg.successful)
    assert jnp.allclose(cg.value, jnp.linalg.solve(matrix, rhs), rtol=1e-9, atol=1e-10)
    assert cg.diagnostics.matvec_count == (
        1 + cg.diagnostics.iterations + cg.diagnostics.iterations // 10
    )
    assert cg.diagnostics.adjoint_matvec_count == 0

    rectangular = jnp.asarray([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]])
    target = la.ArraySpace((3,), dtype=jnp.float64)
    rectangular_operator = la.FunctionLinearOperator(
        lambda vector: rectangular @ vector,
        source=space,
        target=target,
        transpose_action=lambda vector: rectangular.T @ vector,
    )
    target_value = jnp.asarray([1.0, 0.5, -2.0])
    lsmr = la.solve(
        la.LeastSquaresProblem(rectangular_operator),
        target_value,
        policy=la.LinearSolvePolicy(
            la.LSMR(),
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=50),
        ),
    )
    expected = jnp.linalg.lstsq(rectangular, target_value, rcond=None)[0]
    assert lsmr.provenance.backend == "matfree"
    assert bool(lsmr.successful)
    assert jnp.allclose(lsmr.value, expected, rtol=1e-8, atol=1e-9)
    assert lsmr.diagnostics.matvec_count == lsmr.diagnostics.iterations + 1
    assert lsmr.diagnostics.adjoint_matvec_count == lsmr.diagnostics.iterations + 1

    exact = jnp.linalg.solve(matrix, rhs)
    pcg = la.solve(
        la.LinearSystem(operator),
        rhs,
        policy=la.LinearSolvePolicy(
            la.PCG(),
            tolerance=la.TolerancePolicy(
                relative=1e-10,
                absolute=1e-12,
                max_steps=20,
            ),
        ),
        initial_guess=exact,
    )
    assert bool(pcg.successful)
    assert pcg.diagnostics.iterations == 0
    assert pcg.diagnostics.matvec_count == 2
    assert pcg.diagnostics.adjoint_matvec_count == 0


def test_auto_planner_routes_general_pairings_to_native_krylov():
    metric = jnp.asarray([[2.0, 0.5], [0.5, 1.0]])
    space = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=_DensePairing(metric),
    )
    matrix = jnp.asarray([[3.0, 1.0], [0.5, 2.0]])
    operator = la.FunctionLinearOperator(
        lambda vector: matrix @ vector,
        source=space,
        target=space,
    )
    problem = la.LinearSystem(operator)
    iterative_policy = la.LinearSolvePolicy(
        materialization=la.MaterializationPolicy(max_entries=1, max_bytes=1),
    )
    selected = la.plan(problem, iterative_policy)
    assert selected.backend == "native-krylov"
    assert selected.method == "fgmres"

    rhs = jnp.asarray([1.0, -2.0])
    result = la.solve(problem, rhs, policy=iterative_policy)
    assert bool(result.successful)
    assert jnp.allclose(result.value, jnp.linalg.solve(matrix, rhs))
    assert result.diagnostics.matvec_count == 2 * result.diagnostics.iterations + 3
    assert result.diagnostics.adjoint_matvec_count == 0

    with pytest.raises(ValueError, match="Euclidean or diagonal source pairing"):
        la.plan(problem, la.LinearSolvePolicy(la.GMRES()))

    certified = la.DenseLinearOperator(
        jnp.eye(2),
        source=space,
        target=space,
        properties=_positive_definite_properties(),
    )
    assert la.plan(la.LinearSystem(certified)).method == "dense-lu"

    target_metric = jnp.asarray(
        [
            [2.0, 0.2, 0.0],
            [0.2, 1.5, 0.1],
            [0.0, 0.1, 1.0],
        ]
    )
    target = la.ArraySpace(
        (3,),
        dtype=jnp.float64,
        pairing=_DensePairing(target_metric),
    )
    design = jnp.asarray([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]])
    least_squares = la.LeastSquaresProblem(
        la.FunctionLinearOperator(
            lambda vector: design @ vector,
            source=space,
            target=target,
        )
    )
    least_squares_policy = la.LinearSolvePolicy(
        tolerance=la.TolerancePolicy(
            relative=1e-9,
            absolute=1e-11,
            max_steps=100,
        ),
        materialization=la.MaterializationPolicy(max_entries=1, max_bytes=1),
    )
    target_value = jnp.asarray([1.0, 0.5, -2.0])
    least_squares_plan = la.plan(least_squares, least_squares_policy)
    least_squares_result = la.solve(
        least_squares,
        target_value,
        policy=least_squares_policy,
    )
    normal_matrix = design.T @ target_metric @ design
    normal_rhs = design.T @ target_metric @ target_value
    assert least_squares_plan.method == "generalized-lsmr"
    assert bool(least_squares_result.successful)
    assert jnp.allclose(
        least_squares_result.value,
        jnp.linalg.solve(normal_matrix, normal_rhs),
    )
    assert least_squares_result.diagnostics.matvec_count == (
        least_squares_result.diagnostics.iterations + 3
    )
    assert least_squares_result.diagnostics.adjoint_matvec_count == (
        least_squares_result.diagnostics.iterations + 3
    )


def test_dense_solve_is_differentiable_with_respect_to_operator_values():
    rhs = jnp.asarray([2.0, -3.0])
    properties = _positive_definite_properties()

    def objective(log_diagonal):
        diagonal = jnp.exp(log_diagonal)
        operator = la.DenseLinearOperator(jnp.diag(diagonal), properties=properties)
        return jnp.sum(la.solve(la.LinearSystem(operator), rhs).value)

    point = jnp.asarray([0.2, -0.4])
    gradient = jax.grad(objective)(point)
    expected = -rhs / jnp.exp(point)
    assert jnp.allclose(gradient, expected)


def test_space_operator_and_preconditioner_constructors_enforce_invariants():
    with jax.enable_x64(False):
        canonical = la.ArraySpace((2,), dtype=jnp.float64)
        assert canonical.dtype == jnp.dtype(jnp.float32)
        assert canonical.validate(canonical.zeros()).dtype == jnp.float32

    with pytest.raises(TypeError, match="Dense solve backends require"):
        la.solve(
            la.LinearSystem(la.DenseLinearOperator(jnp.eye(2, dtype=jnp.float16))),
            jnp.ones((2,), dtype=jnp.float16),
        )

    with pytest.raises(ValueError, match="weight shapes"):
        la.ArraySpace(
            (2,),
            pairing=la.DiagonalPairing(jnp.ones((3,), dtype=jnp.float64)),
        )
    with pytest.raises(TypeError, match="share one canonical"):
        la.PyTreeSpace(
            {
                "single": jnp.zeros((1,), dtype=jnp.float32),
                "double": jnp.zeros((1,), dtype=jnp.float64),
            }
        )
    with pytest.raises(TypeError, match="share one canonical"):
        la.BlockSpace(
            (
                la.ArraySpace((1,), dtype=jnp.float32),
                la.ArraySpace((1,), dtype=jnp.float64),
            )
        )

    custom_space = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=_DensePairing(jnp.asarray([[2.0, 0.5], [0.5, 1.0]])),
    )
    diagonal = la.DiagonalLinearOperator(jnp.asarray([1.0, 2.0]), space=custom_space)
    assert not diagonal.properties.self_adjoint
    assert not la.adjoint(diagonal).properties.diagonal
    batched = la.DenseLinearOperator(
        jnp.stack((jnp.eye(2), 2.0 * jnp.eye(2))),
        source=custom_space,
        target=custom_space,
    )
    assert not batched.capabilities.adjoint
    with pytest.raises(la.LinearCapabilityError, match="adjoint"):
        la.adjoint(batched)

    structured = la.PyTreeSpace({"vector": jnp.zeros((2,)), "scalar": jnp.zeros(())})
    with pytest.raises(ValueError, match="Batched dense"):
        la.DenseLinearOperator(
            jnp.ones((2, 3, 3)),
            source=structured,
            target=structured,
        )
    with pytest.raises(ValueError, match="identical source and target"):
        la.DenseLinearOperator(
            jnp.ones((2, 3)),
            properties=la.OperatorProperties(
                self_adjoint=True,
                evidence={"self_adjoint": "asserted"},
            ),
        )
    with pytest.raises(TypeError, match="transposed matrix"):
        la.DenseLinearOperator(
            jnp.eye(2, dtype=jnp.float64),
            source=la.ArraySpace((2,), dtype=jnp.float32),
            target=la.ArraySpace((2,), dtype=jnp.float64),
        )

    with pytest.raises(la.LinearCapabilityError, match="bytes"):
        la.materialize(
            la.DenseLinearOperator(jnp.eye(2, dtype=jnp.complex128)),
            la.MaterializationPolicy(max_entries=4, max_bytes=32),
        )
    scalar_space = la.ArraySpace((1,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="nonsingular"):
        la.BlockDiagonalPreconditioner((jnp.zeros((1, 1)),), space=scalar_space)
    with pytest.raises(ValueError, match="nonsingular"):
        la.IncompleteFactorizationPreconditioner(
            jnp.ones((1, 1)),
            jnp.zeros((1, 1)),
            space=scalar_space,
        )
    with pytest.raises(ValueError, match="core"):
        la.LowRankWoodburyPreconditioner(
            jnp.ones((1,)),
            jnp.ones((1, 1)),
            jnp.zeros((1, 1)),
            space=scalar_space,
        )


def test_batched_pairing_aware_adjoint_preserves_hilbert_identity():
    source = la.ArraySpace(
        (2,),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(jnp.asarray([2.0, 5.0])),
    )
    target = la.ArraySpace(
        (2,),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(jnp.asarray([3.0, 7.0])),
    )
    matrix = jnp.asarray(
        [
            [[1.0 + 1.0j, 2.0], [-1.0j, 0.5]],
            [[2.0, -0.5j], [1.0 + 2.0j, -3.0]],
        ]
    )
    operator = la.DenseLinearOperator(matrix, source=source, target=target)
    left = jnp.asarray([[1.0, 2.0j], [-0.5 + 1.0j, 3.0]])
    right = jnp.asarray([[2.0 - 1.0j, 0.25], [1.5j, -2.0]])
    image = operator.mv(left)
    adjoint_image = operator.adjoint_mv(right)
    lhs = jax.vmap(target.inner)(image, right)
    rhs = jax.vmap(source.inner)(left, adjoint_image)
    adjoint_matrix = la.materialize(
        la.adjoint(operator),
        la.MaterializationPolicy(max_entries=8),
    )

    assert jnp.allclose(lhs, rhs)
    assert jnp.allclose(
        adjoint_matrix,
        jnp.asarray([0.5, 0.2])[:, None]
        * jnp.conj(jnp.swapaxes(matrix, -1, -2))
        * jnp.asarray([3.0, 7.0])[None, :],
    )


def test_matrix_free_jit_and_implicit_gradients_track_dynamic_coefficients():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    policy = la.LinearSolvePolicy(
        la.GMRES(),
        tolerance=la.TolerancePolicy(
            relative=1e-10,
            absolute=1e-12,
            max_steps=20,
        ),
    )

    def objective(coefficients, rhs):
        operator = la.FunctionLinearOperator(
            lambda vector: coefficients * vector,
            source=space,
            target=space,
            operator_id="dynamic-diagonal",
        )
        return jnp.sum(la.solve(la.LinearSystem(operator), rhs, policy=policy).value)

    coefficients = jnp.asarray([2.0, 4.0])
    rhs = jnp.asarray([1.0, 2.0])
    assert jnp.allclose(jax.jit(objective)(coefficients, rhs), 1.0)
    assert jnp.allclose(
        jax.grad(objective, argnums=0)(coefficients, rhs),
        -rhs / coefficients**2,
    )
    assert jnp.allclose(
        jax.grad(objective, argnums=1)(coefficients, rhs),
        1.0 / coefficients,
    )


def test_complex_iterative_transposes_and_normal_residuals_use_conjugation():
    matrix = jnp.asarray([[2.0, 1.0j], [-1.0j, 2.0]])
    space = la.ArraySpace((2,), dtype=matrix.dtype)
    operator = la.FunctionLinearOperator(
        lambda vector: matrix @ vector,
        source=space,
        target=space,
        properties=la.OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "asserted",
                "positive_semidefinite": "asserted",
            },
        ),
        operator_id="complex-hermitian",
    )
    policy = la.LinearSolvePolicy(
        la.GMRES(),
        tolerance=la.TolerancePolicy(relative=1e-8, absolute=1e-10, max_steps=20),
    )
    solve_rhs = lambda rhs: la.solve(la.LinearSystem(operator), rhs, policy=policy).value
    rhs = jnp.asarray([1.0 + 2.0j, -0.5 + 0.2j])
    cotangent = jnp.asarray([0.3 - 0.1j, 2.0 + 0.5j])
    _, pullback = jax.vjp(solve_rhs, rhs)
    assert jnp.allclose(pullback(cotangent)[0], jnp.linalg.solve(matrix.T, cotangent))

    rectangular = jnp.asarray([[1.0 + 1.0j, 0.0], [0.0, 2.0 - 1.0j], [1.0j, 1.0]])
    target = la.ArraySpace((3,), dtype=rectangular.dtype)
    rectangular_operator = la.FunctionLinearOperator(
        lambda vector: rectangular @ vector,
        source=space,
        target=target,
        operator_id="complex-rectangular",
    )
    target_value = jnp.asarray([1.0 + 0.2j, -0.4 + 0.7j, 2.0 - 1.0j])
    result = la.solve(
        la.LeastSquaresProblem(rectangular_operator),
        target_value,
        policy=la.LinearSolvePolicy(
            la.GeneralizedLSMR(),
            tolerance=la.TolerancePolicy(
                relative=1e-8,
                absolute=1e-10,
                max_steps=100,
            ),
        ),
    )
    assert bool(result.successful)
    assert jnp.allclose(
        result.value,
        jnp.linalg.lstsq(rectangular, target_value, rcond=None)[0],
    )


def test_plans_validate_reuse_rank_configuration_and_transformed_batches():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    operator = la.FunctionLinearOperator(
        lambda vector: jnp.asarray([[2.0, 0.0], [0.0, 3.0]]) @ vector,
        source=space,
        target=space,
        operator_id="plan-operator",
    )
    problem = la.LinearSystem(operator)
    first = la.plan(problem, la.LinearSolvePolicy(la.GMRES(restart=5)))
    second = la.plan(problem, la.LinearSolvePolicy(la.GMRES(restart=10)))
    assert first.plan_id != second.plan_id

    rectangular_target = la.ArraySpace((3,), dtype=jnp.float64)
    rectangular = la.FunctionLinearOperator(
        lambda vector: jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) @ vector,
        source=space,
        target=rectangular_target,
        operator_id="reused-weight-operator",
    )
    unweighted = la.LeastSquaresProblem(rectangular, problem_id="shared-problem")
    unweighted_plan = la.plan(
        unweighted,
        la.LinearSolvePolicy(la.LSMR()),
    )
    weighted = la.LeastSquaresProblem(
        rectangular,
        weights=jnp.ones((3,)),
        problem_id="shared-problem",
    )
    with pytest.raises(ValueError, match="weights or regularizers"):
        la.prepare(weighted, unweighted_plan)
    with pytest.raises(ValueError, match="full-rank"):
        la.plan(
            unweighted,
            la.LinearSolvePolicy(
                la.LSMR(),
                rank=la.RankPolicy(require_full_rank=True),
            ),
        )

    batched_matrix = jnp.asarray([[[2.0, 1.0], [0.0, 3.0]], [[1.0, -0.5], [2.0, 4.0]]])
    batched_problem = la.LinearSystem(la.DenseLinearOperator(batched_matrix))
    batched_rhs = jnp.asarray([[1.0, 2.0], [-1.0, 0.5]])
    transposed = la.solve_transpose(batched_problem, batched_rhs)
    expected = jax.vmap(jnp.linalg.solve)(
        jnp.swapaxes(batched_matrix, -1, -2),
        batched_rhs,
    )
    assert transposed.provenance.backend == "jax-dense"
    assert jnp.allclose(transposed.value, expected)

    underdetermined = la.LeastSquaresProblem(la.DenseLinearOperator(jnp.ones((2, 3))))
    with pytest.raises(ValueError, match="at least as many rows"):
        la.plan(underdetermined, la.LinearSolvePolicy(la.DenseQR()))

    singular = la.solve(
        la.LinearSystem(la.DenseLinearOperator(jnp.asarray([[1.0, 1.0], [2.0, 2.0]]))),
        jnp.asarray([1.0, 2.0]),
        policy=la.LinearSolvePolicy(la.DenseLU()),
    )
    assert singular.status == int(la.LinearSolveStatus.SINGULAR)
    assert singular.diagnostics.rank == -1

    with pytest.raises(eqx.EquinoxRuntimeError, match="Linear solve failed"):
        throwing = la.solve(
            la.LinearSystem(
                la.DenseLinearOperator(jnp.asarray([[1.0, 1.0], [2.0, 2.0]]))
            ),
            jnp.asarray([1.0, 2.0]),
            policy=la.LinearSolvePolicy(
                la.DenseLU(),
                failure=la.FailurePolicy("error"),
            ),
        )
        jax.block_until_ready(throwing.value)

    condition_limited = la.solve(
        la.LeastSquaresProblem(
            la.FunctionLinearOperator(
                lambda vector: jnp.asarray([[1.0, 0.0], [0.0, 100.0]]) @ vector,
                source=space,
                target=space,
                operator_id="condition-limited",
            )
        ),
        jnp.ones((2,)),
        policy=la.LinearSolvePolicy(
            la.LSMR(condition_limit=1.01),
            tolerance=la.TolerancePolicy(relative=0.0, absolute=0.0, max_steps=10),
        ),
    )
    assert condition_limited.status == int(la.LinearSolveStatus.CONDITION_LIMIT_REACHED)


def test_rhs_layout_and_pre_regularization_rank_cutoff_are_explicit():
    matrices = jnp.asarray([[[2.0, 0.0], [0.0, 3.0]], [[4.0, 0.0], [0.0, 5.0]]])
    prepared = la.prepare(la.LinearSystem(la.DenseLinearOperator(matrices)))
    right_hand_sides = jnp.asarray([[2.0, 4.0], [3.0, 6.0]])
    single = la.solve(prepared, right_hand_sides)
    multiple = la.solve_many(prepared, right_hand_sides)
    explicit = la.solve(
        prepared,
        right_hand_sides,
        rhs_layout=la.RHSLayout((2,)),
    )
    assert single.value.shape == (2, 2)
    assert multiple.value.shape == (2, 2, 2)
    assert jnp.allclose(multiple.value, explicit.value)

    matrix = jnp.diag(jnp.asarray([1.0, 1e-4]))
    operator = la.DenseLinearOperator(matrix)
    regularizer = la.DiagonalLinearOperator(jnp.ones((2,)), space=operator.source)
    truncated = la.solve(
        la.LeastSquaresProblem(operator, regularizer=regularizer),
        jnp.ones((2,)),
        policy=la.LinearSolvePolicy(
            la.DenseSVD(),
            rank=la.RankPolicy(relative_cutoff=1e-3),
        ),
    )
    assert bool(truncated.successful)
    assert truncated.diagnostics.rank == 1
    assert jnp.allclose(truncated.value, jnp.asarray([0.5, 0.0]))


def test_weighted_least_squares_construction_is_jittable_and_differentiable():
    matrix = jnp.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    operator = la.DenseLinearOperator(matrix)
    rhs = jnp.asarray([1.0, 2.0, 2.5])

    def objective(weights):
        result = la.solve(
            la.LeastSquaresProblem(operator, weights=weights),
            rhs,
            policy=la.LinearSolvePolicy(la.DenseSVD()),
        )
        return jnp.sum(result.value)

    weights = jnp.ones((3,))
    assert jnp.isfinite(jax.jit(objective)(weights))
    assert jnp.all(jnp.isfinite(jax.grad(objective)(weights)))


def test_structured_adjoint_actions_respect_declared_pairings():
    source = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray([2.0, 3.0])),
    )
    target = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray([4.0, 0.5])),
    )
    matrix = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    operators = (
        la.DenseLinearOperator(matrix, source=source, target=target),
        la.PermutationLinearOperator(jnp.asarray([1, 0]), space=source),
        la.TriangularLinearOperator(
            jnp.asarray([[1.0, 2.0], [0.0, 4.0]]),
            lower=False,
            space=source,
        ),
        la.TridiagonalLinearOperator(
            jnp.asarray([3.0]),
            jnp.asarray([1.0, 4.0]),
            jnp.asarray([2.0]),
            space=source,
        ),
        la.LowRankLinearOperator(
            jnp.asarray([[1.0], [3.0]]),
            jnp.asarray([[1.0], [2.0]]),
            source=source,
            target=target,
        ),
        la.DiagonalPlusLowRankLinearOperator(
            jnp.asarray([1.0, 4.0]),
            jnp.asarray([[1.0], [3.0]]),
            jnp.asarray([[1.0], [2.0]]),
            space=source,
        ),
    )
    primal = jnp.asarray([0.2, -0.4])
    dual = jnp.asarray([0.7, 0.3])
    for operator in operators:
        image = operator.mv(primal)
        assert jnp.allclose(
            operator.target.inner(image, dual),
            operator.source.inner(primal, operator.adjoint_mv(dual)),
        )

    first = la.DenseLinearOperator(matrix, source=source, target=target)
    second_space = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray([5.0, 7.0])),
    )
    second = la.DenseLinearOperator(
        jnp.asarray([[2.0, -1.0], [0.5, 3.0]]),
        source=second_space,
        target=second_space,
    )
    kronecker = la.KroneckerLinearOperator((first, second))
    kronecker_primal = jnp.arange(4.0).reshape((2, 2)) / 5.0
    kronecker_dual = jnp.asarray([0.3, -0.2, 0.5, 0.7]).reshape((2, 2))
    assert jnp.allclose(
        kronecker.target.inner(kronecker.mv(kronecker_primal), kronecker_dual),
        kronecker.source.inner(
            kronecker_primal,
            kronecker.adjoint_mv(kronecker_dual),
        ),
    )

    symmetric = la.SymmetricLowRankLinearOperator(
        jnp.asarray([[1.0], [2.0]]),
        space=source,
    )
    assert not symmetric.properties.self_adjoint
    with pytest.raises(ValueError, match="Euclidean pairing"):
        la.SymmetricLowRankLinearOperator(
            jnp.asarray([[1.0], [2.0]]),
            space=source,
            positive_semidefinite=True,
        )
    dense_pairing_space = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=_DensePairing(jnp.asarray([[2.0, 0.5], [0.5, 1.0]])),
    )
    with pytest.raises(TypeError, match="coordinate-diagonal"):
        la.TensorProductSpace((dense_pairing_space,))


def test_generalized_lsmr_respects_weights_regularizers_and_minimum_norm_pairings():
    matrix = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rhs = jnp.asarray([1.0, 2.0, 4.0])
    weights = jnp.asarray([1.0, 3.0, 2.0])
    regularizer_matrix = jnp.asarray([[2.0, 0.0], [0.0, 0.5]])
    operator = la.DenseLinearOperator(matrix)
    regularizer = la.DenseLinearOperator(regularizer_matrix)
    policy = la.LinearSolvePolicy(
        la.GeneralizedLSMR(),
        tolerance=la.TolerancePolicy(
            relative=1e-10,
            absolute=1e-12,
            max_steps=100,
        ),
    )
    result = la.solve(
        la.LeastSquaresProblem(
            operator,
            weights=weights,
            regularizer=regularizer,
        ),
        rhs,
        policy=policy,
    )
    expected = jnp.linalg.solve(
        matrix.T @ (weights[:, None] * matrix)
        + regularizer_matrix.T @ regularizer_matrix,
        matrix.T @ (weights * rhs),
    )
    assert bool(result.successful)
    assert jnp.allclose(result.value, expected)
    assert float(result.diagnostics.normal_residual_norm) < 1e-10

    source = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray([2.0, 5.0])),
    )
    target = la.ArraySpace(
        (1,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray([3.0])),
    )
    underdetermined = jnp.asarray([[1.0, 2.0]])
    minimum_norm_problem = la.MinimumNormProblem(
        la.DenseLinearOperator(
            underdetermined,
            source=source,
            target=target,
        )
    )
    minimum_norm = la.solve(
        minimum_norm_problem,
        jnp.asarray([3.0]),
        policy=policy,
    )
    metric = jnp.diag(jnp.asarray([2.0, 5.0]))
    expected_minimum_norm = jnp.linalg.solve(
        metric, underdetermined.T
    ) @ jnp.linalg.solve(
        underdetermined @ jnp.linalg.solve(metric, underdetermined.T),
        jnp.asarray([3.0]),
    )
    assert bool(minimum_norm.successful)
    assert jnp.allclose(minimum_norm.value, expected_minimum_norm)
    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match="MinimumNormProblem initial_guess must be zero",
    ):
        nonzero = la.solve(
            minimum_norm_problem,
            jnp.asarray([3.0]),
            policy=policy,
            initial_guess=jnp.asarray([1.0, 0.0]),
        )
        jax.block_until_ready(nonzero.value)


def test_multi_rhs_batched_and_rhs_only_derivatives_preserve_contracts():
    right_hand_sides = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    coefficients = jnp.asarray([2.0, 5.0])

    def objective(dynamic):
        prepared = la.prepare(
            la.LinearSystem(
                la.DenseLinearOperator(
                    jnp.diag(dynamic),
                    operator_id="multi-rhs-dynamic",
                )
            )
        )
        return jnp.sum(la.solve_many(prepared, right_hand_sides).value)

    expected_gradient = -jnp.sum(right_hand_sides, axis=1) / coefficients**2
    assert jnp.allclose(jax.grad(objective)(coefficients), expected_gradient)
    assert jnp.allclose(
        jax.jit(jax.grad(objective))(coefficients),
        expected_gradient,
    )

    rhs_only_policy = la.LinearSolvePolicy(
        differentiation=la.DifferentiationPolicy("rhs-only")
    )

    def rhs_only_objective(dynamic):
        prepared = la.prepare(
            la.LinearSystem(
                la.DenseLinearOperator(
                    jnp.diag(dynamic),
                    operator_id="multi-rhs-dynamic",
                )
            ),
            rhs_only_policy,
        )
        return jnp.sum(la.solve_many(prepared, right_hand_sides).value)

    assert jnp.array_equal(
        jax.grad(rhs_only_objective)(coefficients),
        jnp.zeros_like(coefficients),
    )

    batched_rhs = jnp.asarray([[2.0, 3.0], [4.0, 5.0]])
    batched_coefficients = jnp.asarray([[2.0, 3.0], [4.0, 5.0]])

    def batched_objective(dynamic):
        matrices = jax.vmap(jnp.diag)(dynamic)
        result = la.solve(
            la.LinearSystem(
                la.DenseLinearOperator(
                    matrices,
                    operator_id="batched-dynamic",
                )
            ),
            batched_rhs,
        )
        return jnp.sum(result.value)

    assert jnp.allclose(
        jax.grad(batched_objective)(batched_coefficients),
        -batched_rhs / batched_coefficients**2,
    )


def test_nullspace_preconditioner_and_structured_plans_use_primal_coordinates():
    metric = jnp.asarray([2.0, 5.0])
    space = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(metric),
    )
    preconditioner = la.DiagonalPreconditioner(
        jnp.asarray([2.0, 4.0]),
        space=space,
    )
    assert jnp.allclose(
        preconditioner.apply(jnp.asarray([6.0, 8.0])),
        jnp.asarray([3.0, 2.0]),
    )

    laplacian = jnp.asarray([[1.0, -1.0], [-1.0, 1.0]]) / metric[:, None]
    nullspace = la.LinearSubspace(space, jnp.asarray([[1.0], [1.0]]))
    problem = la.LinearSystem(
        la.FunctionLinearOperator(
            lambda vector: laplacian @ vector,
            source=space,
            target=space,
            operator_id="weighted-singular-laplacian",
        ),
        nullspace_policy=la.NullspacePolicy(
            right=nullspace,
            left=nullspace,
            compatibility="project",
            gauge="project",
        ),
    )
    result = la.solve(
        problem,
        jnp.asarray([1.0, 0.0]),
        policy=la.LinearSolvePolicy(
            la.FGMRES(restart=2),
            tolerance=la.TolerancePolicy(
                relative=1e-10,
                absolute=1e-12,
                max_steps=5,
            ),
        ),
    )
    assert bool(result.successful)
    assert jnp.allclose(jnp.dot(metric, result.value), 0.0)
    assert float(result.diagnostics.gauge_residual) < 1e-12

    diagonal = la.DiagonalLinearOperator(jnp.asarray([2.0, 3.0]))
    diagonal_preconditioner = la.DiagonalPreconditioner(jnp.asarray([2.0, 3.0]))
    auto_plan = la.plan(
        la.LinearSystem(diagonal),
        la.LinearSolvePolicy(
            preconditioning=la.PreconditioningPolicy(diagonal_preconditioner)
        ),
    )
    assert auto_plan.backend == "native-krylov"
    with pytest.raises(ValueError, match="does not accept preconditioners"):
        la.plan(
            la.LinearSystem(diagonal),
            la.LinearSolvePolicy(
                la.StructuredDirect(),
                preconditioning=la.PreconditioningPolicy(diagonal_preconditioner),
            ),
        )
    with pytest.raises(ValueError, match="rank cutoff"):
        la.plan(
            la.LinearSystem(diagonal),
            la.LinearSolvePolicy(
                la.StructuredDirect(),
                rank=la.RankPolicy(relative_cutoff=1e-6),
            ),
        )


def test_structured_construction_and_preconditioners_are_jittable():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    residual = jnp.asarray([6.0, 8.0])

    @jax.jit
    def apply_all(diagonal, block, lower, upper, core, permutation):
        return (
            la.DiagonalPreconditioner(
                diagonal,
                space=space,
                positive_definite=True,
            ).apply(residual),
            la.BlockDiagonalPreconditioner((block,), space=space).apply(residual),
            la.IncompleteFactorizationPreconditioner(
                lower,
                upper,
                space=space,
            ).apply(residual),
            la.LowRankWoodburyPreconditioner(
                diagonal,
                jnp.asarray([[1.0], [0.0]]),
                core,
                space=space,
            ).apply(residual),
            la.PermutationLinearOperator(permutation, space=space).mv(residual),
        )

    values = apply_all(
        jnp.asarray([2.0, 4.0]),
        jnp.diag(jnp.asarray([2.0, 4.0])),
        jnp.eye(2),
        jnp.diag(jnp.asarray([2.0, 4.0])),
        jnp.asarray([[3.0]]),
        jnp.asarray([1, 0]),
    )
    assert jnp.allclose(values[0], jnp.asarray([3.0, 2.0]))
    assert jnp.allclose(values[1], jnp.asarray([3.0, 2.0]))
    assert jnp.allclose(values[2], jnp.asarray([3.0, 2.0]))
    assert jnp.allclose(values[3], jnp.asarray([1.2, 2.0]))
    assert jnp.allclose(values[4], jnp.asarray([8.0, 6.0]))

    invalid_diagonal = jax.jit(
        lambda diagonal: la.DiagonalPreconditioner(
            diagonal,
            space=space,
            positive_definite=True,
        ).apply(residual)
    )
    with pytest.raises(jax.errors.JaxRuntimeError, match="finite and nonzero"):
        jax.block_until_ready(invalid_diagonal(jnp.asarray([0.0, 4.0])))

    factor = la.TensorProductSpace(
        (
            la.ArraySpace(
                (2,),
                dtype=jnp.float64,
                pairing=la.DiagonalPairing(jnp.asarray([2.0, 3.0])),
            ),
            space,
        )
    )
    nested_inner = jax.jit(
        lambda: la.TensorProductSpace((factor, space)).inner(
            jnp.ones((4, 2)),
            jnp.ones((4, 2)),
        )
    )
    assert jnp.allclose(nested_inner(), 20.0)


def test_batched_rank_deficient_svd_has_correct_implicit_derivative():
    right_hand_side = jnp.asarray([[2.0, 7.0], [6.0, -1.0]])
    policy = la.LinearSolvePolicy(
        la.DenseSVD(),
        rank=la.RankPolicy(relative_cutoff=1e-8),
    )

    def objective(scales):
        matrices = scales[:, None, None] * jnp.asarray([[[1.0, 0.0], [0.0, 0.0]]])
        result = la.solve(
            la.LeastSquaresProblem(
                la.DenseLinearOperator(
                    matrices,
                    operator_id="batched-rank-deficient",
                )
            ),
            right_hand_side,
            policy=policy,
        )
        return jnp.sum(result.value)

    scales = jnp.asarray([2.0, 3.0])
    expected_gradient = jnp.asarray([-0.5, -2.0 / 3.0])
    assert jnp.allclose(jax.grad(objective)(scales), expected_gradient)
    assert jnp.allclose(jax.jit(jax.grad(objective))(scales), expected_gradient)


def test_planner_accounts_for_densification_and_batched_resources(monkeypatch):
    diagonal_values = jnp.asarray([2.0, 3.0])
    diagonal = la.DiagonalLinearOperator(diagonal_values)
    structured_plan = la.plan(la.LinearSystem(diagonal))
    assert structured_plan.candidates[-1].existing_storage_bytes == diagonal_values.nbytes

    with pytest.raises(ValueError, match="materialization requires 4 entries"):
        la.plan(
            la.LinearSystem(diagonal),
            la.LinearSolvePolicy(
                la.DenseLU(),
                materialization=la.MaterializationPolicy(
                    max_entries=3,
                    max_bytes=1024,
                ),
            ),
        )

    batched = la.DenseLinearOperator(
        jnp.broadcast_to(jnp.eye(2), (3, 2, 2)),
    )
    with pytest.raises(ValueError, match="factorization estimate 96"):
        la.plan(
            la.LinearSystem(batched),
            la.LinearSolvePolicy(
                la.DenseLU(),
                resources=la.SolveResourcePolicy(factorization_bytes=64),
            ),
        )

    method = la.FGMRES(restart=2)
    no_derivatives = la.DifferentiationPolicy("none")
    probe_policy = la.LinearSolvePolicy(
        method,
        differentiation=no_derivatives,
    )
    problem = la.LinearSystem(la.DenseLinearOperator(jnp.eye(2)))
    basis_bytes_per_rhs = (
        la.plan(
            problem,
            probe_policy,
        )
        .candidates[-1]
        .krylov_basis_bytes_per_rhs
    )
    bounded = la.prepare(
        problem,
        la.LinearSolvePolicy(
            method,
            differentiation=no_derivatives,
            resources=la.SolveResourcePolicy(
                krylov_basis_bytes=basis_bytes_per_rhs,
            ),
        ),
    )
    assert bool(la.solve(bounded, jnp.ones((2,))).successful)
    with pytest.raises(ValueError, match="for 2 right-hand sides"):
        la.solve_many(bounded, jnp.eye(2))

    direct = la.prepare(
        problem,
        la.LinearSolvePolicy(
            la.DenseLU(),
            resources=la.SolveResourcePolicy(workspace_bytes=32),
        ),
    )
    assert bool(la.solve(direct, jnp.ones((2,))).successful)
    with pytest.raises(ValueError, match="workspace bytes for 2 right-hand sides"):
        la.solve_many(direct, jnp.eye(2))
    with pytest.raises(ValueError, match="workspace bytes for 2 right-hand sides"):
        la.solve_transpose(direct, jnp.eye(2))
    with pytest.raises(ValueError, match="workspace bytes for 2 right-hand sides"):
        la.solve_adjoint(direct, jnp.eye(2))

    def unexpected_preparation(*args, **kwargs):
        raise AssertionError("Numerical preparation ran before RHS resource rejection.")

    monkeypatch.setattr(
        "phydrax.linalg._runtime._prepare_for_plan",
        unexpected_preparation,
    )
    with pytest.raises(ValueError, match="solve workspace bytes"):
        la.solve(
            problem,
            jnp.eye(2),
            policy=la.LinearSolvePolicy(
                la.DenseLU(),
                resources=la.SolveResourcePolicy(workspace_bytes=32),
            ),
        )


def test_operator_action_costs_account_for_shared_state_and_iterative_scratch():
    values = jnp.asarray([2.0, 3.0, 4.0])
    diagonal = la.DiagonalLinearOperator(values)
    shared_sum = la.SumLinearOperator(diagonal, diagonal)

    leaf_cost = la.estimate_operator_action_cost(diagonal)
    sum_cost = la.estimate_operator_action_cost(shared_sum)
    opaque_cost = la.estimate_operator_action_cost(
        la.FunctionLinearOperator(
            lambda value: diagonal.mv(value),
            source=diagonal.source,
            target=diagonal.target,
        )
    )
    solve_plan = la.plan(
        la.LinearSystem(shared_sum),
        la.LinearSolvePolicy(
            la.GMRES(),
            differentiation=la.DifferentiationPolicy("none"),
        ),
    )
    candidate = solve_plan.candidates[-1]

    assert isinstance(leaf_cost, la.OperatorActionCostEstimate)
    assert leaf_cost.storage_bytes == values.nbytes
    assert leaf_cost.apply_workspace_bytes_per_rhs == 0
    assert sum_cost.storage_bytes == values.nbytes
    assert sum_cost.apply_workspace_bytes_per_rhs == values.nbytes
    assert sum_cost.exact
    assert not opaque_cost.exact
    assert opaque_cost.apply_workspace_bytes_per_rhs == values.nbytes
    assert candidate.existing_storage_bytes == sum_cost.storage_bytes
    assert (
        candidate.operator_apply_workspace_bytes_per_rhs
        == sum_cost.apply_workspace_bytes_per_rhs
    )


def test_planner_accounts_for_preconditioner_state_and_workspace():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    operator = la.DenseLinearOperator(
        jnp.asarray([[4.0, 1.0], [1.0, 3.0]]),
        source=space,
        target=space,
        properties=_positive_definite_properties(),
    )
    problem = la.LinearSystem(operator)
    policy = la.LinearSolvePolicy(
        la.PCG(),
        differentiation=la.DifferentiationPolicy("none"),
        preconditioning=la.PreconditioningPolicy(la.JacobiPreconditionerBuilder()),
    )
    solve_plan = la.plan(problem, policy)
    assert solve_plan.preconditioner_plan is not None
    cost = solve_plan.preconditioner_plan.cost
    estimate = solve_plan.candidates[-1]

    assert isinstance(cost, la.PreconditionerCostEstimate)
    assert cost.storage_bytes == 2 * jnp.dtype(jnp.float64).itemsize
    assert cost.setup_matvec_count == 0
    matrix_free = la.FunctionLinearOperator(
        lambda value: operator.mv(value),
        source=space,
        target=space,
        properties=_positive_definite_properties(),
    )
    assert (
        la.DenseInversePreconditionerBuilder()
        .cost_for(matrix_free, materialization=la.MaterializationPolicy())
        .setup_matvec_count
        == space.size
    )
    assert (
        la.DenseInversePreconditionerBuilder()
        .cost_for(operator + operator, materialization=la.MaterializationPolicy())
        .setup_matvec_count
        == 0
    )
    assert estimate.preconditioner_setup_matvec_count == cost.setup_matvec_count
    assert estimate.preconditioner_storage_bytes == cost.storage_bytes
    assert (
        estimate.preconditioner_preparation_workspace_bytes
        == cost.preparation_workspace_bytes
    )
    assert (
        estimate.preconditioner_apply_workspace_bytes_per_rhs
        == cost.apply_workspace_bytes_per_rhs
    )
    with pytest.raises(ValueError, match="preconditioner state"):
        la.plan(
            problem,
            la.LinearSolvePolicy(
                la.PCG(),
                differentiation=la.DifferentiationPolicy("none"),
                preconditioning=la.PreconditioningPolicy(
                    la.JacobiPreconditionerBuilder()
                ),
                resources=la.SolveResourcePolicy(
                    preconditioner_bytes=cost.storage_bytes - 1
                ),
            ),
        )
    tight_materialization_plan = la.plan(
        problem,
        la.LinearSolvePolicy(
            la.PCG(),
            differentiation=la.DifferentiationPolicy("none"),
            materialization=la.MaterializationPolicy(
                max_entries=1,
                max_bytes=1024,
            ),
            preconditioning=la.PreconditioningPolicy(la.JacobiPreconditionerBuilder()),
        ),
    )
    assert tight_materialization_plan.preconditioner_plan is not None
    assert tight_materialization_plan.preconditioner_plan.cost.setup_matvec_count == 0


def test_composites_do_not_upgrade_unknown_property_evidence():
    matrix = jnp.asarray([[2.0, 1.0], [1.0, 3.0]])
    unverified = la.DenseLinearOperator(
        matrix,
        properties=la.OperatorProperties(
            self_adjoint=True,
            rank=2,
        ),
    )
    summed = unverified + unverified
    adjointed = la.adjoint(unverified)
    blocked = la.BlockDiagonalLinearOperator((unverified, unverified))
    kronecker = la.KroneckerLinearOperator((unverified, unverified))

    assert summed.properties.evidence_for("self_adjoint") == "unknown"
    assert adjointed.properties.evidence_for("self_adjoint") == "unknown"
    assert blocked.properties.evidence_for("rank") == "unknown"
    assert kronecker.properties.evidence_for("rank") == "unknown"
    with pytest.raises(ValueError, match="certified self-adjoint"):
        la.plan(
            la.LinearSystem(summed),
            la.LinearSolvePolicy(la.MINRES()),
        )

    verified = la.DenseLinearOperator(
        matrix,
        properties=la.OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "asserted"},
        ),
    )
    assert (verified + verified).properties.evidence_for("self_adjoint") == "transformed"


def test_dense_qr_reports_rank_from_singular_values_not_diagonal_pivots():
    design = jnp.asarray([[1.0, 1.0e8], [0.0, 1.0]])
    result = la.solve(
        la.LeastSquaresProblem(la.DenseLinearOperator(design)),
        jnp.asarray([1.0, 1.0]),
        policy=la.LinearSolvePolicy(
            la.DenseQR(),
            rank=la.RankPolicy(relative_cutoff=1.0e-12),
        ),
    )

    assert result.diagnostics.rank == 1
    assert result.status == int(la.LinearSolveStatus.RANK_DEFICIENT)
    assert not bool(result.successful)


def test_block_diagonal_dense_factors_use_structured_direct_execution():
    first = la.DenseLinearOperator(jnp.asarray([[3.0, 1.0], [1.0, 2.0]]))
    second = la.DenseLinearOperator(jnp.asarray([[4.0]]))
    operator = la.BlockDiagonalLinearOperator((first, second))
    problem = la.LinearSystem(operator)
    rhs = (jnp.asarray([1.0, -1.0]), jnp.asarray([2.0]))

    selected = la.plan(problem)
    result = la.solve(problem, rhs)

    assert selected.backend == "jax-structured"
    assert bool(result.successful)
    assert jnp.allclose(result.value[0], jnp.linalg.solve(first.matrix, rhs[0]))
    assert jnp.allclose(result.value[1], jnp.asarray([0.5]))


def test_kronecker_sum_structured_direct_matches_dense_and_propagates_properties():
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "asserted",
            "positive_definite": "asserted",
        },
    )
    first = la.DenseLinearOperator(
        jnp.asarray([[3.0, 1.0], [1.0, 2.0]]),
        properties=properties,
    )
    second = la.DenseLinearOperator(
        jnp.asarray(
            [
                [4.0, 0.5, 0.0],
                [0.5, 3.0, 0.25],
                [0.0, 0.25, 2.0],
            ]
        ),
        properties=properties,
    )
    operator = la.KroneckerSumLinearOperator((first, second))
    problem = la.LinearSystem(operator)
    policy = la.LinearSolvePolicy(la.StructuredDirect())
    rhs = jnp.arange(1.0, 7.0).reshape((2, 3))
    dense = la.materialize(
        operator,
        la.MaterializationPolicy(max_entries=36),
    )

    selected = la.plan(problem)
    structured_cost = next(
        candidate
        for candidate in selected.candidates
        if candidate.provider == "jax-structured" and candidate.accepted
    )
    strict_selected = la.plan(
        problem,
        la.LinearSolvePolicy(
            la.StructuredDirect(),
            materialization=la.MaterializationPolicy(
                max_entries=1,
                max_bytes=1,
            ),
        ),
    )
    result = la.solve(problem, rhs, policy=policy)
    jitted = jax.jit(lambda value: la.solve(problem, value, policy=policy).value)(rhs)
    expected = jnp.linalg.solve(dense, rhs.reshape((-1,))).reshape(rhs.shape)

    assert operator.properties.certifies("self_adjoint")
    assert operator.properties.certifies("positive_semidefinite")
    assert operator.properties.certifies("positive_definite")
    assert operator.properties.certifies("rank")
    assert operator.properties.rank == operator.source.size
    assert operator.capabilities.materialize
    assert operator.capabilities.diagonal_assembly
    assert selected.backend == "jax-structured"
    assert strict_selected.backend == "jax-structured"
    assert structured_cost.additional_matrix_bytes == 0
    assert structured_cost.factorization_bytes > 0
    assert structured_cost.solve_workspace_bytes_per_rhs == (
        4 * operator.source.size * rhs.dtype.itemsize
    )
    assert bool(result.successful)
    assert jnp.allclose(result.value, expected)
    assert jnp.allclose(jitted, expected)


def test_kronecker_sum_structured_direct_respects_weighted_complex_pairings():
    weights = jnp.asarray([2.0, 5.0])
    metric_sqrt = jnp.sqrt(weights)
    weighted_space = la.ArraySpace(
        (2,),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(weights),
    )
    hermitian = jnp.asarray(
        [[3.0, 1.0 + 0.2j], [1.0 - 0.2j, 2.0]],
        dtype=jnp.complex128,
    )
    weighted_matrix = hermitian * metric_sqrt[None, :] / metric_sqrt[:, None]
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "asserted",
            "positive_definite": "asserted",
        },
    )
    first = la.DenseLinearOperator(
        weighted_matrix,
        source=weighted_space,
        target=weighted_space,
        properties=properties,
    )
    second = la.DenseLinearOperator(
        jnp.asarray(
            [[4.0, 0.3j], [-0.3j, 2.0]],
            dtype=jnp.complex128,
        ),
        properties=properties,
    )
    operator = la.KroneckerSumLinearOperator((first, second))
    rhs = jnp.asarray([[1.0 + 0.5j, 2.0 - 0.2j], [-1.0 + 0.3j, 0.7 - 0.4j]])
    dense = la.materialize(
        operator,
        la.MaterializationPolicy(max_entries=16),
    )

    result = la.solve(
        la.LinearSystem(operator),
        rhs,
        policy=la.LinearSolvePolicy(la.StructuredDirect()),
    )
    expected = jnp.linalg.solve(dense, rhs.reshape((-1,))).reshape(rhs.shape)

    assert bool(result.successful)
    assert jnp.allclose(result.value, expected)


def test_kronecker_sum_structured_direct_is_differentiable_and_reports_singularity():
    positive_properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "asserted",
            "positive_definite": "asserted",
        },
    )
    second = la.DenseLinearOperator(
        jnp.diag(jnp.asarray([2.0, 4.0, 7.0])),
        properties=positive_properties,
    )
    rhs = jnp.arange(1.0, 7.0).reshape((2, 3))
    policy = la.LinearSolvePolicy(la.StructuredDirect())

    def objective(parameter):
        first = la.DenseLinearOperator(
            jnp.diag(jnp.asarray([parameter, parameter + 1.0])),
            properties=positive_properties,
        )
        operator = la.KroneckerSumLinearOperator((first, second))
        return jnp.sum(la.solve(la.LinearSystem(operator), rhs, policy=policy).value)

    parameter = jnp.asarray(3.0)
    eigenvalue_sums = jnp.asarray([[5.0, 7.0, 10.0], [6.0, 8.0, 11.0]])
    expected_gradient = -jnp.sum(rhs / eigenvalue_sums**2)

    assert jnp.allclose(jax.grad(objective)(parameter), expected_gradient)
    assert jnp.allclose(
        jax.jit(jax.grad(objective))(parameter),
        expected_gradient,
    )

    self_adjoint_properties = la.OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "asserted"},
    )
    singular = la.KroneckerSumLinearOperator(
        (
            la.DenseLinearOperator(
                jnp.diag(jnp.asarray([1.0, -1.0])),
                properties=self_adjoint_properties,
            ),
            la.DenseLinearOperator(
                jnp.diag(jnp.asarray([-1.0, 2.0])),
                properties=self_adjoint_properties,
            ),
        )
    )
    singular_result = la.solve(
        la.LinearSystem(singular),
        jnp.ones((2, 2)),
        policy=policy,
    )

    assert singular_result.status == int(la.LinearSolveStatus.SINGULAR)
    assert not bool(singular_result.successful)
