#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.linalg import (
    ArraySpace,
    assemble_diagonal,
    DiagonalPairing,
    MaterializationPolicy,
    materialize,
    plan_sparse_assembly,
    prepare_sparse_assembly,
    refresh_sparse_assembly,
)
from phydrax.sparse import (
    EdgeRelation,
    RowRelation,
    SparseCoordinateOperator,
    SparseLinearMap,
)
from phydrax.sparse._linear import _SparseStoragePlan
from phydrax.sparse._ops import (
    block_linear_adjoint_apply,
    block_linear_apply,
    block_linear_transpose_apply,
)


def _matrix_from_blocks(
    sources: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    blocks: np.ndarray,
    source_count: int,
    target_count: int,
) -> np.ndarray:
    """Independent cell-major block oracle using only host slice accumulation."""
    target_fiber, source_fiber = blocks.shape[-2:]
    matrix = np.zeros(
        (target_count * target_fiber, source_count * source_fiber),
        dtype=blocks.dtype,
    )
    for route in range(sources.size):
        if valid[route]:
            source = int(sources[route]) * source_fiber
            target = int(targets[route]) * target_fiber
            matrix[target : target + target_fiber, source : source + source_fiber] += (
                blocks[route]
            )
    return matrix


def _rectangular_operator() -> tuple[SparseCoordinateOperator, np.ndarray]:
    sources = np.asarray([0, 1, 0, 1, -99], dtype=np.int32)
    targets = np.asarray([0, 0, 0, 1, 500], dtype=np.int32)
    valid = np.asarray([True, True, True, True, False], dtype=np.bool_)
    blocks = np.asarray(
        [
            [[1 + 2j, 2, -1j], [3, -2 + 1j, 4]],
            [[-1, 0.5j, 2], [1j, 3, -2]],
            [[2, -1j, 1], [0.5, 2j, -3]],
            [[0, 1 + 1j, 2], [-2j, -1, 0.5]],
            [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        ],
        dtype=np.complex128,
    )
    relation = EdgeRelation(sources, targets, source_size=3, target_size=2, valid=valid)
    source = ArraySpace(
        (3, 3),
        dtype=jnp.complex128,
        pairing=DiagonalPairing(jnp.arange(1, 10, dtype=jnp.float64).reshape((3, 3))),
    )
    target = ArraySpace(
        (2, 2),
        dtype=jnp.complex128,
        pairing=DiagonalPairing(jnp.asarray([[2, 3], [5, 7]], dtype=jnp.float64)),
    )
    operator = SparseCoordinateOperator(
        relation,
        blocks,
        source=source,
        target=target,
        block_shape=(2, 3),
        operator_id="rectangular-fiber-routes",
    )
    return operator, _matrix_from_blocks(sources, targets, valid, blocks, 3, 2)


def test_rectangular_complex_routes_and_hilbert_adjoint() -> None:
    operator, matrix = _rectangular_operator()
    source = jnp.asarray(
        [[1 - 1j, 2, -3j], [0.5, 2 + 1j, -1], [jnp.nan, jnp.nan, jnp.nan]],
        dtype=jnp.complex128,
    )
    target = jnp.asarray([[2 + 1j, -1], [3j, 0.5]], dtype=jnp.complex128)
    safe_source = np.asarray(source.at[2].set(0)).reshape(-1)
    expected_forward = (matrix @ safe_source).reshape((2, 2))
    expected_transpose = (matrix.T @ np.asarray(target).reshape(-1)).reshape((3, 3))
    target_weights = np.asarray([2, 3, 5, 7], dtype=np.float64)
    source_weights = np.arange(1, 10, dtype=np.float64)
    expected_adjoint = (
        matrix.conj().T
        @ (target_weights * np.asarray(target).reshape(-1))
        / source_weights
    ).reshape((3, 3))

    np.testing.assert_allclose(
        eqx.filter_jit(lambda prepared, value: prepared.mv(value))(operator, source),
        expected_forward,
    )
    np.testing.assert_allclose(
        eqx.filter_jit(lambda prepared, value: prepared.transpose_mv(value))(
            operator, target
        ),
        expected_transpose,
    )
    np.testing.assert_allclose(
        eqx.filter_jit(lambda prepared, value: prepared.adjoint_mv(value))(
            operator, target
        ),
        expected_adjoint,
    )
    finite_source = source.at[2].set(jnp.asarray([1j, -2, 4]))
    np.testing.assert_allclose(
        operator.target.inner(operator.mv(finite_source), target),
        operator.source.inner(finite_source, operator.adjoint_mv(target)),
    )
    np.testing.assert_allclose(
        materialize(operator, MaterializationPolicy(max_entries=36, max_bytes=576)),
        matrix,
    )


def test_row_blocks_preserve_case_local_routes_and_independent_payload_axes() -> None:
    relation = RowRelation(
        np.asarray([[[0, 1], [1, 1]], [[1, 0], [0, -99]]], dtype=np.int32),
        source_size=2,
        case_shape=(2,),
        valid=np.asarray([[[True, True], [True, True]], [[True, True], [True, False]]]),
    )
    blocks = np.arange(48, dtype=np.float64).reshape((2, 2, 2, 3, 2)) / 7
    blocks = blocks.astype(np.complex128) * (1 + 0.25j)
    blocks[1, 1, 1] = np.nan
    source = np.arange(32, dtype=np.float64).reshape((2, 2, 2, 2, 2)) / 3
    target = np.arange(48, dtype=np.float64).reshape((2, 2, 3, 2, 2)) / 5
    # Cell identities include their case, independently of the runtime conversion.
    source_cells = np.asarray([0, 1, 1, 1, 3, 2, 2, -99], dtype=np.int32)
    target_cells = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    valid = np.asarray([True, True, True, True, True, True, True, False])
    matrix = _matrix_from_blocks(
        source_cells, target_cells, valid, blocks.reshape((8, 3, 2)), 4, 4
    )

    forward = jax.jit(lambda values: block_linear_apply(relation, blocks, values))(source)
    transpose = block_linear_transpose_apply(relation, blocks, target)
    adjoint = block_linear_adjoint_apply(relation, blocks, target)
    np.testing.assert_allclose(forward.reshape((12, 4)), matrix @ source.reshape((8, 4)))
    np.testing.assert_allclose(
        transpose.reshape((8, 4)), matrix.T @ target.reshape((12, 4))
    )
    np.testing.assert_allclose(
        adjoint.reshape((8, 4)), matrix.conj().T @ target.reshape((12, 4))
    )


def test_native_block_storage_and_traced_assembly_refresh_coalesce_routes() -> None:
    operator, matrix = _rectangular_operator()
    storage = operator.sparse_storage()
    rows = np.repeat(np.arange(4), np.diff(np.asarray(storage.indptr)))
    stored_matrix = np.zeros((4, 9), dtype=np.complex128)
    stored_matrix[rows, np.asarray(storage.indices)] = np.asarray(storage.values)
    np.testing.assert_allclose(stored_matrix, matrix)
    # Three distinct cell pairs, each owning all six rectangular block entries.
    assert storage.nnz == 18
    assert storage.batch_shape == ()

    plan = plan_sparse_assembly(operator)
    assert not plan.uses_materialization
    prepared = prepare_sparse_assembly(plan, operator)
    changed = eqx.tree_at(lambda op: op.coefficients, operator, 2 * operator.coefficients)
    source = jnp.arange(9, dtype=jnp.float64).reshape((3, 3)).astype(jnp.complex128)
    apply_refreshed = jax.jit(
        lambda current, values: refresh_sparse_assembly(prepared, current).operator.mv(
            values
        )
    )
    np.testing.assert_allclose(
        apply_refreshed(changed, source),
        (2 * matrix @ np.asarray(source).reshape(-1)).reshape((2, 2)),
    )


def test_block_coefficient_derivatives_follow_valid_route_algebra() -> None:
    operator, _ = _rectangular_operator()
    source = jnp.arange(9, dtype=jnp.float64).reshape((3, 3)).astype(jnp.complex128)
    direction = jnp.ones_like(operator.coefficients).at[-1].set(1000 + 300j)

    def action(coefficients: Array) -> Array:
        return block_linear_apply(operator.relation, coefficients, source)

    _, derivative = jax.jit(
        lambda coefficients: jax.jvp(action, (coefficients,), (direction,))
    )(operator.coefficients)
    # Sum of source-fiber entries at each valid route, including the duplicate.
    np.testing.assert_allclose(
        derivative, np.asarray([[18, 18], [12, 12]], dtype=np.complex128)
    )


def test_block_diagonal_assembly_is_coordinate_diagonal_not_block_trace() -> None:
    relation = EdgeRelation(
        np.asarray([0, 0, 1, -99], dtype=np.int32),
        np.asarray([0, 0, 1, 500], dtype=np.int32),
        source_size=2,
        target_size=2,
        valid=np.asarray([True, True, True, False]),
    )
    space = ArraySpace((2, 2), dtype=jnp.complex128)
    operator = SparseCoordinateOperator(
        relation,
        np.asarray(
            [
                [[1, 8], [9, 2]],
                [[3, 6], [7, 4]],
                [[5, 2], [3, 6]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ]
        ),
        source=space,
        target=space,
        block_shape=(2, 2),
    )
    diagonal = assemble_diagonal(operator)
    np.testing.assert_array_equal(diagonal, np.asarray([4, 6, 5, 6]))
    assert diagonal.dtype == jnp.dtype(jnp.complex128)


def test_scalar_batches_are_not_reinterpreted_as_matrix_fibers() -> None:
    relation = EdgeRelation(
        np.asarray([0, 1, 0], dtype=np.int32),
        np.asarray([0, 0, 1], dtype=np.int32),
        source_size=2,
        target_size=2,
    )
    coefficients = jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.float64)
    operator = SparseLinearMap(relation, coefficients)
    source = jnp.asarray([[7, 8], [9, 10]], dtype=jnp.float64)
    np.testing.assert_array_equal(operator.mv(source), np.asarray([[23, 21], [86, 54]]))
    assert operator.batch_shape == (2,)
    space = ArraySpace((2,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="coefficients must have shape"):
        SparseCoordinateOperator(relation, coefficients, source=space, target=space)


@pytest.mark.parametrize("block_shape", [(0, 3), (2, 0), (-1, 2)])
def test_block_fibers_must_be_positive(block_shape: tuple[int, int]) -> None:
    relation = EdgeRelation(
        np.asarray([0], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        source_size=1,
        target_size=1,
    )
    space = ArraySpace((1,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="positive target and source fiber sizes"):
        SparseCoordinateOperator(
            relation,
            jnp.ones((1,), dtype=jnp.float64),
            source=space,
            target=space,
            block_shape=block_shape,
        )


def test_empty_block_route_set_preserves_rectangular_shapes() -> None:
    relation = EdgeRelation(
        np.asarray([], dtype=np.int32),
        np.asarray([], dtype=np.int32),
        source_size=0,
        target_size=2,
    )
    coefficients = jnp.empty((0, 3, 2), dtype=jnp.float64)
    source = jnp.empty((0, 2), dtype=jnp.float64)
    np.testing.assert_array_equal(
        block_linear_apply(relation, coefficients, source), np.zeros((2, 3))
    )
    np.testing.assert_array_equal(
        block_linear_adjoint_apply(
            relation, coefficients, jnp.ones((2, 3), dtype=jnp.float64)
        ),
        np.zeros((0, 2)),
    )


def test_block_assembly_refuses_changed_valid_connectivity() -> None:
    operator, _ = _rectangular_operator()
    prepared = prepare_sparse_assembly(plan_sparse_assembly(operator), operator)
    changed = eqx.tree_at(
        lambda op: op.relation.source_indices,
        operator,
        operator.relation.source_indices.at[0].set(1),
    )
    with pytest.raises(ValueError, match="unchanged relation routes"):
        refresh_sparse_assembly(prepared, changed)
    traced_refresh = eqx.filter_jit(
        lambda current: materialize(
            refresh_sparse_assembly(prepared, current).operator,
            MaterializationPolicy(max_entries=36, max_bytes=576),
        )
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="unchanged relation routes"):
        traced_refresh(changed).block_until_ready()


def test_explicit_block_mode_refuses_scalar_or_mismatched_fiber_coefficients() -> None:
    operator, _ = _rectangular_operator()
    for coefficients in (
        jnp.ones((5,), dtype=jnp.float64),
        jnp.ones((5, 3, 2), dtype=jnp.float64),
    ):
        with pytest.raises(ValueError, match="coefficients must have shape"):
            SparseCoordinateOperator(
                operator.relation,
                coefficients,
                source=operator.source,
                target=operator.target,
                block_shape=(2, 3),
            )
    with pytest.raises(ValueError, match="including fibers"):
        SparseCoordinateOperator(
            operator.relation,
            operator.coefficients,
            source=ArraySpace((3,), dtype=jnp.complex128),
            target=operator.target,
            block_shape=(2, 3),
        )


@pytest.mark.parametrize("matrix_fibers", [False, True])
def test_real_coefficients_preserve_complex_coordinate_vectors(
    matrix_fibers: bool,
) -> None:
    matrix = np.asarray([[1, -2], [3, 4], [-5, 6]], dtype=np.float64)
    if matrix_fibers:
        relation = EdgeRelation(
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            source_size=1,
            target_size=1,
        )
        coefficients = matrix[None, :, :]
        block_shape = (3, 2)
    else:
        relation = EdgeRelation(
            np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int32),
            np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
            source_size=2,
            target_size=3,
        )
        coefficients = matrix.reshape(-1)
        block_shape = None
    operator = SparseCoordinateOperator(
        relation,
        coefficients,
        source=ArraySpace((2,), dtype=jnp.complex128),
        target=ArraySpace((3,), dtype=jnp.complex128),
        block_shape=block_shape,
    )
    source = jnp.asarray([2 + 3j, -1 + 4j], dtype=jnp.complex128)
    target = jnp.asarray([1 - 2j, 3 + 1j, -4j], dtype=jnp.complex128)
    np.testing.assert_allclose(operator.mv(source), matrix @ np.asarray(source))
    np.testing.assert_allclose(
        operator.transpose_mv(target), matrix.T @ np.asarray(target)
    )
    np.testing.assert_allclose(operator.adjoint_mv(target), matrix.T @ np.asarray(target))
    stored_matrix = materialize(
        operator, MaterializationPolicy(max_entries=6, max_bytes=96)
    )
    np.testing.assert_array_equal(stored_matrix, matrix)
    assert stored_matrix.dtype == jnp.dtype(jnp.complex128)


def test_prepared_block_storage_refresh_is_traceable_without_host_pattern_reads() -> None:
    operator, matrix = _rectangular_operator()
    storage_plan = _SparseStoragePlan(operator.relation, block_shape=operator.block_shape)
    bound = SparseCoordinateOperator(
        operator.relation,
        operator.coefficients,
        source=operator.source,
        target=operator.target,
        block_shape=operator.block_shape,
        storage_plan=storage_plan,
        operator_id=operator.operator_id,
    )
    with jax.ensure_compile_time_eval():
        admitted = bound.sparse_storage()
    refreshed = eqx.tree_at(
        lambda current: current.coefficients, bound, 3 * bound.coefficients
    )
    storage = jax.jit(lambda current: current.sparse_storage())(refreshed)
    rows = np.repeat(np.arange(4), np.diff(np.asarray(storage.indptr)))
    actual = np.zeros((4, 9), dtype=np.complex128)
    actual[rows, np.asarray(storage.indices)] = np.asarray(storage.values)
    np.testing.assert_allclose(actual, 3 * matrix)
    np.testing.assert_allclose(
        admitted.values, matrix[rows, np.asarray(admitted.indices)]
    )
