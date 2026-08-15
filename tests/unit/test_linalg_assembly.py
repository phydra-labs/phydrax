#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import opt_einsum as oe
import pytest

import phydrax as phx


la = phx.linalg


def _materialized_diagonal(operator):
    matrix = la.materialize(
        operator,
        la.MaterializationPolicy(max_entries=100_000, max_bytes=1_000_000),
    )
    return jnp.diagonal(matrix, axis1=-2, axis2=-1)


def test_exact_diagonal_assembly_covers_structured_operator_families():
    space = la.ArraySpace((3,), dtype=jnp.float64)
    operators = (
        la.DenseLinearOperator(
            jnp.asarray([[2.0, 1.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]]),
            source=space,
            target=space,
        ),
        la.DiagonalLinearOperator(jnp.asarray([2.0, 4.0, 7.0]), space=space),
        la.IdentityLinearOperator(space),
        la.PermutationLinearOperator(jnp.asarray([0, 2, 1]), space=space),
        la.TriangularLinearOperator(
            jnp.asarray([[2.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, 6.0, 7.0]]),
            lower=True,
            space=space,
        ),
        la.TridiagonalLinearOperator(
            jnp.asarray([3.0, 6.0]),
            jnp.asarray([2.0, 4.0, 7.0]),
            jnp.asarray([1.0, 5.0]),
            space=space,
        ),
        la.BandedLinearOperator(
            jnp.asarray([[0.0, 1.0, 5.0], [2.0, 4.0, 7.0], [3.0, 6.0, 0.0]]),
            lower_bandwidth=1,
            upper_bandwidth=1,
            space=space,
        ),
        la.LowRankLinearOperator(
            jnp.asarray([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]]),
            jnp.asarray([[2.0, 0.0], [1.0, 3.0], [-1.0, 2.0]]),
            source=space,
            target=space,
        ),
        la.SymmetricLowRankLinearOperator(
            jnp.asarray([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]]),
            weights=jnp.asarray([2.0, -0.5]),
            space=space,
        ),
        la.DiagonalPlusLowRankLinearOperator(
            jnp.asarray([2.0, 4.0, 7.0]),
            jnp.asarray([[1.0], [2.0], [-1.0]]),
            jnp.asarray([[3.0], [-1.0], [2.0]]),
            space=space,
        ),
    )

    for operator in operators:
        assert operator.capabilities.diagonal_assembly
        assert jnp.allclose(
            la.assemble_diagonal(operator), _materialized_diagonal(operator)
        )


def test_composite_block_and_kronecker_diagonals_are_structural_and_jittable():
    factor_space = la.ArraySpace((2,), dtype=jnp.float64)
    left = la.DiagonalLinearOperator(jnp.asarray([2.0, 3.0]), space=factor_space)
    right = la.DenseLinearOperator(
        jnp.asarray([[4.0, 1.0], [-2.0, 5.0]]),
        source=factor_space,
        target=factor_space,
    )
    summed = 2.0 * left + right
    block = la.BlockDiagonalLinearOperator(
        (summed, la.IdentityLinearOperator(factor_space))
    )
    kronecker = la.KroneckerLinearOperator((summed, right))
    kronecker_sum = la.KroneckerSumLinearOperator((summed, right))

    for operator in (
        summed,
        la.transpose(summed),
        la.adjoint(summed),
        block,
        kronecker,
        kronecker_sum,
    ):
        actual = jax.jit(la.assemble_diagonal)(operator)
        assert jnp.allclose(actual, _materialized_diagonal(operator))


def test_sparse_and_batched_dense_diagonal_assembly():
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 2, 1, 2], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 2], dtype=jnp.int32),
        source_size=3,
        target_size=3,
    )
    sparse = phx.sparse.SparseLinearMap(
        relation,
        jnp.asarray([2.0, 8.0, 4.0, 7.0]),
    )
    assert jnp.allclose(la.assemble_diagonal(sparse), jnp.asarray([2.0, 4.0, 7.0]))

    matrices = jnp.asarray(
        [
            [[2.0, 1.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    batched = la.DenseLinearOperator(matrices)
    assert jnp.array_equal(
        jax.jit(la.assemble_diagonal)(batched),
        jnp.asarray([[2.0, 4.0], [5.0, 8.0]]),
    )


def test_dense_fallback_is_explicit_and_budget_bounded():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    matrix = jnp.asarray([[2.0, 1.0], [3.0, 4.0]])
    operator = la.FunctionLinearOperator(
        lambda value: matrix @ value,
        source=space,
        target=space,
    )

    with pytest.raises(la.LinearCapabilityError, match="explicit materialization"):
        la.assemble_diagonal(operator)
    with pytest.raises(la.LinearCapabilityError, match="exceeding"):
        la.assemble_diagonal(
            operator,
            materialization=la.MaterializationPolicy(max_entries=1, max_bytes=8),
        )

    diagonal = la.assemble_diagonal(
        operator,
        materialization=la.MaterializationPolicy(max_entries=4, max_bytes=32),
    )
    assert jnp.array_equal(diagonal, jnp.asarray([2.0, 4.0]))


def test_jacobi_uses_shared_structural_diagonal_without_dense_budget():
    space = la.ArraySpace((3,), dtype=jnp.float64)
    left = la.DiagonalLinearOperator(jnp.asarray([2.0, 4.0, 5.0]), space=space)
    right = la.DiagonalLinearOperator(jnp.asarray([1.0, 2.0, 5.0]), space=space)
    operator = left + right
    builder = la.JacobiPreconditionerBuilder(relaxation=0.5)
    cost = builder.cost_for(
        operator,
        materialization=la.MaterializationPolicy(max_entries=1, max_bytes=8),
    )
    preconditioner = builder.prepare(
        operator,
        materialization=la.MaterializationPolicy(max_entries=1, max_bytes=8),
    )

    assert cost.accepted
    assert cost.setup_matvec_count == 0
    residual = jnp.asarray([3.0, 6.0, 10.0])
    assert jnp.allclose(preconditioner.apply(residual), jnp.asarray([0.5, 0.5, 0.5]))


def test_local_blocks_preserve_grouped_event_and_trailing_rhs_axes():
    blocks = jnp.asarray(
        [
            [[2.0, 1.0, -1.0], [0.0, 3.0, 2.0]],
            [[4.0, 0.0, 1.0], [1.0, 2.0, -2.0]],
        ]
    )
    operator = la.LocalBlockDiagonalLinearOperator(blocks)
    values = jnp.arange(2 * 3 * 2 * 3, dtype=jnp.float64).reshape((2, 3, 2, 3))
    expected = oe.contract("boi,bijk->bojk", blocks, values)

    assert operator.batch_shape == ()
    assert operator.source.shape == (2, 3)
    assert operator.target.shape == (2, 2)
    assert jnp.allclose(jax.jit(lambda value: operator.mv(value))(values), expected)
    assert jnp.allclose(
        operator.transpose_mv(expected),
        oe.contract("boi,bojk->bijk", blocks, expected),
    )
    assert jnp.allclose(
        operator._materialize(), jnp.kron(jnp.eye(2), blocks[0]).at[2:, 3:].set(blocks[1])
    )


def test_local_block_adjoint_respects_nonuniform_coordinate_pairings():
    source_weights = jnp.asarray([[2.0, 3.0], [5.0, 7.0]])
    target_weights = jnp.asarray([[11.0, 13.0], [17.0, 19.0]])
    source = la.ArraySpace(
        (2, 2),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(source_weights),
    )
    target = la.ArraySpace(
        (2, 2),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(target_weights),
    )
    blocks = jnp.asarray(
        [
            [[2.0 + 1.0j, -1.0j], [3.0, 4.0 - 2.0j]],
            [[1.0, 2.0j], [-3.0j, 5.0]],
        ]
    )
    operator = la.LocalBlockDiagonalLinearOperator(
        blocks,
        source=source,
        target=target,
    )
    value = jnp.asarray([[1.0 + 2.0j, -3.0], [2.0j, 4.0 - 1.0j]])
    dense = operator._materialize()
    expected = (
        jnp.conj(dense.T)
        @ (target_weights.reshape((-1,)) * value.reshape((-1,)))
        / source_weights.reshape((-1,))
    ).reshape(source.shape)

    assert jnp.allclose(
        jax.jit(lambda vector: operator.adjoint_mv(vector))(value),
        expected,
    )


def test_local_block_diagonal_materialization_and_assembly():
    blocks = jnp.asarray(
        [
            [[2.0, 1.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 0.0], [-1.0, 10.0]],
        ]
    )
    operator = la.LocalBlockDiagonalLinearOperator(blocks)
    dense = la.materialize(
        operator,
        la.MaterializationPolicy(max_entries=36, max_bytes=288),
    )

    assert jnp.allclose(
        la.assemble_diagonal(operator),
        jnp.asarray([2.0, 4.0, 5.0, 8.0, 9.0, 10.0]),
    )
    assert jnp.allclose(dense, jax.scipy.linalg.block_diag(*blocks))


def test_local_block_preconditioner_uses_batched_lu_and_relaxation():
    blocks = jnp.asarray(
        [
            [[2.0, 1.0], [0.0, 3.0]],
            [[4.0, 0.0], [1.0, 2.0]],
        ]
    )
    solution = jnp.asarray([[1.0, -2.0], [3.0, 4.0]])
    residual = oe.contract("bij,bj->bi", blocks, solution)
    preconditioner = la.LocalBlockPreconditioner(blocks, relaxation=0.25)

    assert jnp.allclose(
        jax.jit(lambda value: preconditioner.apply(value))(residual),
        0.25 * solution,
    )
    with pytest.raises(ValueError, match="nonsingular"):
        la.LocalBlockPreconditioner(blocks.at[1].set(0.0))


def test_local_block_positive_definite_factorization_uses_pairing_transform():
    metric = jnp.asarray([[2.0, 5.0], [3.0, 7.0]])
    metric_sqrt = jnp.sqrt(metric)
    hermitian = jnp.asarray(
        [
            [[4.0, 1.0], [1.0, 3.0]],
            [[5.0, -1.0], [-1.0, 2.0]],
        ]
    )
    blocks = hermitian * metric_sqrt[:, None, :] / metric_sqrt[:, :, None]
    space = la.ArraySpace(
        (2, 2),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(metric),
    )
    preconditioner = la.LocalBlockPreconditioner(
        blocks,
        space=space,
        positive_definite=True,
    )
    solution = jnp.asarray([[1.0, 2.0], [-3.0, 4.0]])
    residual = oe.contract("bij,bj->bi", blocks, solution)

    assert preconditioner.properties.certifies("positive_definite")
    assert jnp.allclose(preconditioner.apply(residual), solution)


def test_structured_local_block_solve_is_exact_resource_bounded_and_differentiable():
    blocks = jnp.asarray(
        [
            [[3.0, 1.0], [0.0, 2.0]],
            [[4.0, -1.0], [2.0, 3.0]],
        ]
    )
    rhs = jnp.asarray([[2.0, 3.0], [5.0, -1.0]])
    operator = la.LocalBlockDiagonalLinearOperator(blocks, operator_id="local-block")
    policy = la.LinearSolvePolicy(la.StructuredDirect())
    solve_plan = la.plan(la.LinearSystem(operator), policy)
    result = la.solve(la.LinearSystem(operator), rhs, policy=policy)
    expected = jax.vmap(jnp.linalg.solve)(blocks, rhs)
    structured_cost = next(
        candidate
        for candidate in solve_plan.candidates
        if candidate.provider == "jax-structured" and candidate.accepted
    )

    assert solve_plan.backend == "jax-structured"
    assert structured_cost.additional_matrix_bytes == 0
    assert structured_cost.factorization_bytes < operator.source.size**2 * 8
    assert bool(result.successful)
    assert jnp.allclose(result.value, expected)

    def objective(values):
        dynamic = la.LocalBlockDiagonalLinearOperator(
            values,
            operator_id="local-block-gradient",
        )
        solved = la.solve(la.LinearSystem(dynamic), rhs, policy=policy)
        return jnp.sum(solved.value)

    gradient = jax.jit(jax.grad(objective))(blocks)
    assert gradient.shape == blocks.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_sparse_assembly_handles_canonical_algebraic_graphs_and_weighted_adjoint():
    weights = jnp.asarray([2.0, 3.0, 5.0])
    space = la.ArraySpace(
        (3,),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(weights),
    )
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([2, 0, 1, 1, 0]),
        jnp.asarray([0, 0, 1, 1, 2]),
        source_size=3,
        target_size=3,
    )
    sparse = phx.sparse.SparseCoordinateOperator(
        relation,
        jnp.asarray([1.0j, 2.0, 3.0 - 1.0j, 4.0, 5.0j]),
        source=space,
        target=space,
    )
    diagonal = la.DiagonalLinearOperator(
        jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.complex128),
        space=space,
    )
    permutation = la.PermutationLinearOperator(
        jnp.asarray([2, 0, 1]),
        space=space,
        dtype=jnp.complex128,
    )
    operator = la.adjoint(2.0 * sparse + diagonal) @ (sparse + permutation)
    plan = la.plan_sparse_assembly(operator)
    prepared = la.prepare_sparse_assembly(plan, operator)
    storage = prepared.operator.sparse_storage()
    dense_policy = la.MaterializationPolicy(max_entries=9, max_bytes=144)
    expected = la.materialize(operator, dense_policy)
    actual = la.materialize(prepared.operator, dense_policy)
    vector = jnp.asarray([1.0 + 0.5j, -2.0j, 3.0])

    assert storage.canonical
    assert storage.sorted_indices
    assert plan.nnz == storage.values.size
    assert plan.cost.result_nnz == plan.nnz
    assert plan.cost.output_bytes > 0
    assert jnp.allclose(actual, expected)
    assert jnp.allclose(
        jax.jit(lambda value: prepared.operator.mv(value))(vector),
        operator.mv(vector),
    )


def test_sparse_assembly_refresh_reuses_structure_and_rejects_pattern_changes():
    space = la.ArraySpace((3,), dtype=jnp.float64)

    def graph(relation, coefficients):
        sparse = phx.sparse.SparseCoordinateOperator(
            relation,
            coefficients,
            source=space,
            target=space,
        )
        diagonal = la.DiagonalLinearOperator(
            jnp.asarray([2.0, 3.0, 4.0]),
            space=space,
        )
        return (sparse + diagonal) @ la.transpose(sparse)

    initial_relation = phx.sparse.EdgeRelation(
        jnp.asarray([2, 0, 1, 1, 0]),
        jnp.asarray([0, 0, 1, 1, 2]),
        source_size=3,
        target_size=3,
    )
    initial = graph(
        initial_relation,
        jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    prepared = la.prepare_sparse_assembly(la.plan_sparse_assembly(initial), initial)

    reordered_relation = phx.sparse.EdgeRelation(
        jnp.asarray([1, 0, 2, 0, 1]),
        jnp.asarray([1, 2, 0, 0, 1]),
        source_size=3,
        target_size=3,
    )
    refreshed_operator = graph(
        reordered_relation,
        jnp.asarray([7.0, 11.0, 13.0, 17.0, 19.0]),
    )
    refreshed = la.refresh_sparse_assembly(prepared, refreshed_operator)
    initial_storage = prepared.operator.sparse_storage()
    refreshed_storage = refreshed.operator.sparse_storage()
    dense_policy = la.MaterializationPolicy(max_entries=9, max_bytes=72)

    assert refreshed.numeric_version == prepared.numeric_version + 1
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert jnp.array_equal(initial_storage.indices, refreshed_storage.indices)
    assert jnp.array_equal(initial_storage.indptr, refreshed_storage.indptr)
    assert not jnp.array_equal(initial_storage.values, refreshed_storage.values)
    assert jnp.allclose(
        la.materialize(refreshed.operator, dense_policy),
        la.materialize(refreshed_operator, dense_policy),
    )

    changed_relation = phx.sparse.EdgeRelation(
        jnp.asarray([1, 0, 2, 0, 2]),
        jnp.asarray([1, 2, 0, 0, 2]),
        source_size=3,
        target_size=3,
    )
    with pytest.raises(ValueError, match="symbolic pattern"):
        la.refresh_sparse_assembly(
            prepared,
            graph(
                changed_relation,
                jnp.asarray([7.0, 11.0, 13.0, 17.0, 19.0]),
            ),
        )


def test_sparse_assembly_dense_fallback_is_explicit_and_resource_bounded():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    matrix = jnp.asarray([[2.0, 0.0], [3.0, 4.0]])
    operator = la.FunctionLinearOperator(
        lambda value: matrix @ value,
        source=space,
        target=space,
    )

    with pytest.raises(la.LinearCapabilityError, match="materialization policy"):
        la.plan_sparse_assembly(operator)
    with pytest.raises(la.LinearCapabilityError, match="4 nonzeros"):
        la.plan_sparse_assembly(
            operator,
            la.SparseAssemblyPolicy(
                max_nnz=3,
                materialization=la.MaterializationPolicy(
                    max_entries=4,
                    max_bytes=32,
                ),
            ),
        )

    policy = la.SparseAssemblyPolicy(
        max_nnz=4,
        max_bytes=64,
        materialization=la.MaterializationPolicy(
            max_entries=4,
            max_bytes=32,
        ),
    )
    plan = la.plan_sparse_assembly(operator, policy)
    assembled = la.assemble_sparse(operator, policy)

    assert plan.nnz == 4
    assert jnp.array_equal(
        la.materialize(
            assembled,
            la.MaterializationPolicy(max_entries=4, max_bytes=32),
        ),
        matrix,
    )


def test_sparse_assembly_covers_block_and_kronecker_structures():
    first = la.TridiagonalLinearOperator(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([3.0, 4.0, 5.0]),
        jnp.asarray([6.0, 7.0]),
    )
    second = la.BandedLinearOperator(
        jnp.asarray([[0.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        lower_bandwidth=0,
        upper_bandwidth=1,
    )
    operators = (
        la.LocalBlockDiagonalLinearOperator(
            jnp.asarray(
                [
                    [[2.0, 1.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            )
        ),
        la.BlockDiagonalLinearOperator((first, second)),
        la.KroneckerLinearOperator((first, second)),
        la.KroneckerSumLinearOperator((first, second)),
    )

    for operator in operators:
        assembled = la.assemble_sparse(operator)
        entries = operator.target.size * operator.source.size
        materialization = la.MaterializationPolicy(
            max_entries=entries,
            max_bytes=entries * 8,
        )
        assert jnp.allclose(
            la.materialize(assembled, materialization),
            la.materialize(operator, materialization),
        )


def test_uniform_block_assembly_extracts_sparse_graph_blocks_without_densifying():
    space = la.ArraySpace((4,), dtype=jnp.float64)
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 2, 3, 2, 0]),
        jnp.asarray([0, 0, 2, 2, 3, 3]),
        source_size=4,
        target_size=4,
    )
    sparse = phx.sparse.SparseCoordinateOperator(
        relation,
        jnp.asarray([4.0, 1.0, 5.0, 2.0, 6.0, 0.5]),
        source=space,
        target=space,
    )
    diagonal = la.DiagonalLinearOperator(
        jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        space=space,
    )
    operator = sparse + la.transpose(sparse) + diagonal
    blocks = la.assemble_uniform_blocks(
        operator,
        2,
        policy=la.SparseAssemblyPolicy(max_nnz=16),
    )
    dense = la.materialize(
        operator,
        la.MaterializationPolicy(max_entries=16, max_bytes=128),
    )

    assert jnp.allclose(
        blocks,
        jnp.stack((dense[:2, :2], dense[2:, 2:])),
    )
    with pytest.raises(ValueError, match="divide"):
        la.assemble_uniform_blocks(operator, 3)


def test_block_jacobi_builder_is_resource_aware_jittable_and_refreshable():
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "asserted",
            "positive_definite": "asserted",
        },
    )
    matrix = jnp.asarray(
        [
            [4.0, 1.0, 0.2, 0.0],
            [1.0, 3.0, 0.0, 0.1],
            [0.2, 0.0, 5.0, 2.0],
            [0.0, 0.1, 2.0, 6.0],
        ]
    )
    operator = la.DenseLinearOperator(
        matrix,
        properties=properties,
        operator_id="block-jacobi-refresh",
    )
    builder = la.BlockJacobiPreconditionerBuilder(2, relaxation=0.5)
    materialization = la.MaterializationPolicy(max_entries=16, max_bytes=128)
    cost = builder.cost_for(operator, materialization=materialization)
    preconditioner = builder.prepare(
        operator,
        materialization=materialization,
    )
    residual = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    expected = 0.5 * jnp.concatenate(
        (
            jnp.linalg.solve(matrix[:2, :2], residual[:2]),
            jnp.linalg.solve(matrix[2:, 2:], residual[2:]),
        )
    )

    assert cost.accepted
    assert cost.setup_matvec_count == 0
    assert cost.storage_bytes < matrix.size * matrix.dtype.itemsize
    assert preconditioner.properties.certifies("positive_definite")
    assert jnp.allclose(
        jax.jit(lambda value: preconditioner.apply(value))(residual),
        expected,
    )

    solve_policy = la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(
            relative=1e-10,
            absolute=1e-12,
            max_steps=20,
        ),
        preconditioning=la.PreconditioningPolicy(builder),
    )
    problem = la.LinearSystem(operator)
    prepared = la.prepare(problem, solve_policy)
    changed_matrix = matrix + jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    changed_problem = la.LinearSystem(
        la.DenseLinearOperator(
            changed_matrix,
            properties=properties,
            operator_id="block-jacobi-refresh",
        ),
        problem_id=problem.problem_id,
    )
    refreshed = la.refresh(prepared, changed_problem)
    result = la.solve(refreshed, residual)

    assert refreshed.preconditioning_state is not None
    assert refreshed.preconditioning_state.refresh_kind == "refreshed"
    assert isinstance(
        refreshed.preconditioning_state.action,
        la.LocalBlockPreconditioner,
    )
    assert bool(result.successful)
    assert jnp.allclose(result.value, jnp.linalg.solve(changed_matrix, residual))

    matrix_free = la.FunctionLinearOperator(
        lambda value: matrix @ value,
        source=operator.source,
        target=operator.target,
        properties=properties,
    )
    rejected = builder.cost_for(
        matrix_free,
        materialization=la.MaterializationPolicy(max_entries=1, max_bytes=8),
    )
    assert not rejected.accepted
    assert "materialization" in rejected.reason
