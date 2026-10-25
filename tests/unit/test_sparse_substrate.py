#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import pytest
from jax import Array
from jax.typing import DTypeLike

import phydrax as phx
from phydrax.linalg import ArraySpace, DiagonalPairing


def test_sparse_substrate_scenario_1() -> None:
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 2, 0], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        source_size=3,
        target_size=2,
        valid=jnp.asarray([True, False, True, True]),
    )
    coefficients = jnp.asarray([2.0 + 1.0j, jnp.nan + 0.0j, -1.0j, 4.0])
    action = phx.sparse.SparseLinearMap(relation, coefficients)
    source = jnp.asarray([1.0 - 1.0j, jnp.nan + 0.0j, 2.0 + 0.5j])
    target = jnp.asarray([3.0 + 2.0j, -1.0 + 0.5j])
    dense = action.as_dense()

    forward = jax.jit(lambda values: action.mv(values))(source)
    transpose = jax.jit(lambda values: action.transpose_mv(values))(target)
    adjoint = jax.jit(lambda values: action.adjoint_mv(values))(target)

    assert jnp.all(jnp.isfinite(forward))
    assert jnp.allclose(forward, dense @ source.at[1].set(0.0))
    assert jnp.allclose(transpose, dense.T @ target)
    assert jnp.allclose(adjoint, jnp.conj(dense).T @ target)
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 2, 0], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        source_size=3,
        target_size=2,
        valid=jnp.asarray([True, False, True, True]),
    )
    coefficients = jnp.asarray([[2.0, jnp.nan, -1.0, 4.0], [-3.0, jnp.nan, 0.5, 2.0]])
    action = phx.sparse.SparseLinearMap(relation, coefficients)
    source = jnp.asarray([[1.0, 0.0, 2.0], [-1.0, 0.0, 3.0]])
    target = jnp.asarray([[2.0, -1.0], [0.5, 4.0]])
    dense = action.as_dense()

    forward = jax.jit(lambda values: action.mv(values))(source)
    transpose = jax.jit(lambda values: action.transpose_mv(values))(target)
    adjoint = jax.jit(lambda values: action.adjoint_mv(values))(target)
    storage = action.sparse_storage()

    assert action.batch_shape == (2,)
    assert dense.shape == (2, 2, 3)
    assert storage.batch_shape == (2,)
    assert storage.values.shape[0] == 2
    assert jnp.allclose(forward, oe.contract("bij,bj->bi", dense, source))
    assert jnp.allclose(transpose, oe.contract("bji,bj->bi", dense, target))
    assert jnp.allclose(adjoint, transpose)
    relation = phx.sparse.RowRelation(
        jnp.asarray(
            [
                [[0, 2], [1, 0]],
                [[2, 1], [0, 2]],
            ],
            dtype=jnp.int32,
        ),
        source_size=3,
        valid=jnp.asarray(
            [
                [[True, True], [True, False]],
                [[True, False], [True, True]],
            ]
        ),
        case_shape=(2,),
    )
    coefficients = jnp.asarray(
        [
            [[0.25, 0.75], [2.0, jnp.nan]],
            [[-1.0, jnp.nan], [0.5, 0.5]],
        ]
    )
    action = phx.sparse.SparseLinearMap(relation, coefficients)
    source = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [jnp.nan, jnp.nan], [9.0, 10.0]],
        ]
    )
    target = jnp.asarray(
        [
            [[1.0, -1.0], [2.0, 3.0]],
            [[-2.0, 0.5], [4.0, -3.0]],
        ]
    )
    dense = action.as_dense()

    forward = action.mv(source)
    adjoint = action.adjoint_mv(target)
    safe_source = source.at[1, 1].set(0.0)

    assert forward.shape == (2, 2, 2)
    assert adjoint.shape == (2, 3, 2)
    assert jnp.all(jnp.isfinite(forward))
    assert jnp.allclose(
        forward.reshape((4, 2)),
        dense @ safe_source.reshape((6, 2)),
    )
    assert jnp.allclose(
        adjoint.reshape((6, 2)),
        jnp.conj(dense).T @ target.reshape((4, 2)),
    )


def test_sparse_substrate_scenario_2() -> None:
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
        source_size=3,
        target_size=3,
        valid=jnp.asarray([True, False, True]),
    )
    route_values = jnp.asarray([[2.0, 4.0], [jnp.nan, jnp.nan], [-1.0, 5.0]])

    summed = phx.sparse.route_reduce(relation, route_values)
    maximum = phx.sparse.route_reduce(relation, route_values, reduction="max")

    expected = jnp.asarray([[2.0, 4.0], [-1.0, 5.0], [0.0, 0.0]])
    assert jnp.array_equal(summed, expected)
    assert jnp.array_equal(maximum, expected)
    boundary = jnp.asarray(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
            [0.0, 1.0],
        ]
    )
    incidence = phx.discretization.OrientedIncidence(
        1,
        phx.discretization.EntitySet("vertices", 0, jnp.arange(3, dtype=jnp.int32)),
        phx.discretization.EntitySet("edges", 1, jnp.arange(2, dtype=jnp.int32)),
        phx.sparse.EdgeRelation(
            jnp.asarray([0, 1, 1, 2], dtype=jnp.int32),
            jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
            source_size=3,
            target_size=2,
        ),
        jnp.asarray([-1.0, 1.0, -1.0, 1.0]),
    )
    lower = jnp.asarray([2.0, 3.0, 5.0])
    upper = jnp.asarray([7.0, 11.0])

    derivative_action = incidence.exterior_derivative()
    boundary_action = incidence.boundary()

    assert jnp.allclose(derivative_action(lower), boundary.T @ lower)
    assert jnp.allclose(boundary_action(upper), boundary @ upper)
    assert jnp.array_equal(derivative_action.as_dense(), boundary.T)
    assert jnp.array_equal(boundary_action.as_dense(), boundary)
    source = ArraySpace(
        (3,),
        dtype=jnp.complex128,
        pairing=DiagonalPairing(jnp.asarray([2.0, 3.0, 5.0])),
    )
    target = ArraySpace(
        (2,),
        dtype=jnp.complex128,
        pairing=DiagonalPairing(jnp.asarray([7.0, 11.0])),
    )
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 2, 0], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        source_size=3,
        target_size=2,
    )
    operator = phx.sparse.SparseCoordinateOperator(
        relation,
        jnp.asarray([1.0 + 2.0j, -3.0j, 4.0 - 1.0j, 2.0]),
        source=source,
        target=target,
    )
    left = jnp.asarray([1.0 - 1.0j, 2.0, -0.5 + 3.0j])
    right = jnp.asarray([0.25 + 2.0j, -1.0j])

    assert jnp.allclose(
        target.inner(operator.mv(left), right),
        source.inner(left, operator.adjoint_mv(right)),
    )

    row_operator = phx.sparse.SparseCoordinateOperator(
        phx.sparse.RowRelation(
            jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32),
            source_size=3,
        ),
        jnp.asarray([[1.0 + 1.0j, 2.0], [3.0, 4.0 - 2.0j]]),
        source=source,
        target=target,
    )
    assert jnp.allclose(
        row_operator.mv(left),
        row_operator.as_dense() @ left,
    )
    assert jnp.allclose(
        target.inner(row_operator.mv(left), right),
        source.inner(left, row_operator.adjoint_mv(right)),
    )


def test_sparse_plans_reuse_global_structural_jacobian_and_hessian_patterns() -> None:
    space = ArraySpace((4,), dtype=jnp.float64)
    target = ArraySpace((3,), dtype=jnp.float64)

    def residual(values: Any, _: Any) -> Any:
        return (values[1:] - values[:-1]) ** 2

    first = jnp.asarray([0.0, 1.0, 3.0, 6.0])
    second = jnp.asarray([1.0, 1.5, 2.5, 4.0])
    jacobian_plan = phx.sparse.compile_sparse_jacobian(
        residual,
        first,
        source=space,
        target=target,
        compiler="auto",
    )
    first_operator = jacobian_plan.operator(first)
    second_operator = jacobian_plan.operator(second)

    assert jacobian_plan.nnz == 6
    assert jacobian_plan.num_colors == 2
    assert jnp.allclose(first_operator.as_dense(), jax.jacfwd(residual)(first, None))
    assert jnp.allclose(second_operator.as_dense(), jax.jacfwd(residual)(second, None))
    assert jnp.allclose(
        jax.jit(lambda vector: second_operator.mv(vector))(jnp.ones_like(second)),
        second_operator.as_dense() @ jnp.ones_like(second),
    )
    dynamic_action = jax.jit(
        lambda point, vector: jacobian_plan.operator(point).mv(vector)
    )
    assert jnp.allclose(
        dynamic_action(second, jnp.ones_like(second)),
        second_operator.as_dense() @ jnp.ones_like(second),
    )

    def energy(values: Any, _: Any) -> Any:
        return jnp.sum((values[1:] - values[:-1]) ** 2) + jnp.sum(values**2)

    hessian_plan = phx.sparse.compile_sparse_hessian(
        energy,
        first,
        space=space,
        compiler="auto",
        contract=phx.sparse.SparseHessianContract("riesz"),
        structure=phx.sparse.SparsePattern.from_coo(
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
            [0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            (4, 4),
            symmetric=True,
        ),
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    hessian = hessian_plan.operator(second)
    hessian_matrix = jax.hessian(energy)(second, None)
    assert hessian_plan.num_colors < space.size
    assert jnp.allclose(hessian.as_dense(), hessian_matrix)
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0])
    system = phx.linalg.LinearSystem(hessian)
    selected_plan = phx.linalg.plan(system)
    assert selected_plan.policy.failure.mode == "status"
    assert selected_plan.method == "pcg"
    result = phx.linalg.solve(system, rhs, policy=selected_plan)
    assert bool(result.successful)
    assert jnp.allclose(result.value, jnp.linalg.solve(hessian_matrix, rhs))


def test_sparse_substrate_scenario_3() -> None:
    space = ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=DiagonalPairing(jnp.asarray([2.0, 4.0])),
    )
    point = jnp.asarray([3.0, -2.0])
    pattern = phx.sparse.SparsePattern.from_coo(
        [0, 1],
        [0, 1],
        (2, 2),
        symmetric=True,
    )
    plan = phx.sparse.compile_sparse_hessian(
        lambda value, _: 0.5 * value[0] ** 2 + 1.5 * value[1] ** 2,
        point,
        space=space,
        structure=pattern,
        compiler="native",
        contract=phx.sparse.SparseHessianContract("riesz"),
    )

    prepared = phx.sparse.prepare_sparse_linearization(plan, point)

    assert plan.hessian_contract is not None
    assert plan.hessian_contract.kind == "riesz"
    assert jnp.allclose(
        prepared.linearization.primal,
        jnp.asarray([point[0] / 2.0, 3.0 * point[1] / 4.0]),
    )
    relation = phx.sparse.EdgeRelation(
        jnp.arange(5, dtype=jnp.int32),
        jnp.asarray([2, 0, 2, 0, 1], dtype=jnp.int32),
        source_size=5,
        target_size=4,
        valid=jnp.asarray([True, True, True, False, True]),
    )
    execution = phx.sparse.RelationExecutionPlan().prepare(relation)
    values = jnp.asarray([1.0 + 2.0j, 3.0, -2.0j, jnp.nan, 4.0 - 1.0j])

    deterministic, evidence = execution.reduce(
        values,
        accumulation="deterministic",
    )
    compensated, _ = execution.reduce(values, accumulation="compensated")

    expected = jnp.asarray([3.0, 4.0 - 1.0j, 1.0, 0.0])
    assert jnp.array_equal(deterministic, expected)
    assert jnp.array_equal(compensated, expected)
    assert bool(evidence.successful)
    matrices = jnp.asarray([[[2.0, -1.0], [-1.0, 2.0]]])
    inputs = jnp.asarray([[0, 1]], dtype=jnp.int32)
    outputs = jnp.asarray([[1, 2]], dtype=jnp.int32)
    baseline = phx.sparse.ElementTensorOperator(matrices, inputs, outputs, 3, 3)
    rerouted = phx.sparse.ElementTensorOperator(
        matrices,
        jnp.asarray([[1, 0]], dtype=jnp.int32),
        outputs,
        3,
        3,
    )
    inactive = phx.sparse.ElementTensorOperator(
        matrices,
        inputs,
        outputs,
        3,
        3,
        valid=jnp.asarray([False]),
    )
    ranked = phx.sparse.ElementTensorOperator(
        matrices,
        inputs,
        outputs,
        3,
        3,
        properties=phx.linalg.OperatorProperties(
            rank=2,
            evidence={"rank": "asserted"},
        ),
    )

    assert (
        len(
            {
                baseline.operator_id,
                rerouted.operator_id,
                inactive.operator_id,
                ranked.operator_id,
            }
        )
        == 4
    )
    plan = phx.sparse.KeyGroupPlan(
        5,
        3,
        9,
        maximum_group_size=2,
        case_shape=(2,),
    )
    keys = jnp.asarray([[4, 1, 4, 2, 9], [3, 3, 8, 2, 9]])
    valid = jnp.asarray([[True, True, True, True, False], [True] * 5])
    stable_ids = jnp.asarray([[5, 4, 3, 2, 1], [0, 1, 2, 3, 4]])
    state = jax.jit(plan.build)(keys, valid, stable_ids=stable_ids)
    lookup = state.lookup(jnp.asarray([[1, 4, 7], [2, 3, 8]]))

    assert jnp.array_equal(state.group_keys[0], jnp.asarray([1, 2, 4]))
    assert jnp.array_equal(state.group_counts[0], jnp.asarray([1, 1, 2]))
    assert bool(state.evidence.successful[0])
    assert not bool(state.evidence.successful[1])
    assert bool(state.evidence.group_overflow[1])
    assert jnp.array_equal(
        lookup.supported,
        jnp.asarray([[True, True, False], [False, False, False]]),
    )


def test_sparse_diagonal_assembly_and_numeric_refresh_are_jit_safe() -> None:
    indices = jnp.arange(3, dtype=jnp.int32)
    relation = phx.sparse.EdgeRelation(
        indices,
        indices,
        source_size=3,
        target_size=3,
    )
    space = ArraySpace((3,), dtype=jnp.float64)
    properties = phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )

    def operator(coefficients: Any) -> Any:
        return phx.sparse.SparseCoordinateOperator(
            relation,
            coefficients,
            source=space,
            target=space,
            properties=properties,
            operator_id="jit-sparse-refresh-operator",
        )

    initial_coefficients = jnp.asarray([2.0, 3.0, 4.0])
    problem = phx.linalg.LinearSystem(
        operator(initial_coefficients),
        problem_id="jit-sparse-refresh-system",
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.ConjugateGradient(),
        preconditioning=phx.linalg.PreconditioningPolicy(
            phx.linalg.JacobiPreconditionerBuilder()
        ),
    )
    prepared = phx.linalg.prepare(problem, policy)
    right_hand_side = jnp.asarray([1.0, -2.0, 3.0])

    def refresh_and_solve(coefficients: Any) -> Any:
        refreshed_problem = phx.linalg.LinearSystem(
            operator(coefficients),
            problem_id=problem.problem_id,
        )
        refreshed = phx.linalg.refresh(prepared, refreshed_problem)
        diagonal = phx.linalg.assemble_diagonal(refreshed_problem.operator)
        result = phx.linalg.solve(refreshed, right_hand_side)
        return result.value, diagonal

    coefficients = jnp.asarray([2.5, 3.5, 4.5])
    value, diagonal = jax.jit(refresh_and_solve)(coefficients)
    assert jnp.allclose(diagonal, coefficients)
    assert jnp.allclose(value, right_hand_side / coefficients)


def test_relation_execution_masks_padding_and_reports_target_capacity_status() -> None:
    padded = phx.sparse.EdgeRelation(
        jnp.asarray([0, -1], dtype=jnp.int32),
        jnp.asarray([0, -1], dtype=jnp.int32),
        source_size=1,
        target_size=1,
        valid=jnp.asarray([True, False]),
    )
    padded_execution = phx.sparse.RelationExecutionPlan().prepare(padded)
    reduced, accepted = padded_execution.reduce(jnp.asarray([2.0, jnp.nan]))

    assert jnp.array_equal(reduced, jnp.asarray([2.0]))
    assert bool(accepted.successful)
    assert int(accepted.valid_routes) == 1

    overflowing = phx.sparse.EdgeRelation(
        jnp.asarray([0, 0], dtype=jnp.int32),
        jnp.asarray([0, 1], dtype=jnp.int32),
        source_size=1,
        target_size=2,
    )
    limited_execution = phx.sparse.RelationExecutionPlan(
        maximum_active_targets=1
    ).prepare(overflowing)
    refused, evidence = limited_execution.reduce(jnp.asarray([3.0, 4.0]))

    assert not bool(evidence.successful)
    assert int(evidence.active_targets) == 2
    assert jnp.array_equal(refused, jnp.zeros((2,)))


def test_eager_edge_metadata_preserves_active_bounds_and_masked_routes() -> None:
    sources = jnp.asarray([0, 2, 3], dtype=jnp.int32)
    targets = jnp.asarray([0, 1, 1], dtype=jnp.int32)
    relation = phx.sparse.EdgeRelation(
        sources,
        targets,
        source_size=3,
        target_size=2,
        valid=jnp.asarray([True, True, False]),
    )
    action = phx.sparse.SparseLinearMap(relation, jnp.ones(3, dtype=jnp.float64))
    np.testing.assert_allclose(action.mv(jnp.asarray([1.0, 2.0, 3.0])), [1.0, 3.0])
    with pytest.raises(ValueError):
        phx.sparse.EdgeRelation(sources, targets, source_size=3, target_size=2)


def test_traced_edge_indices_preserve_runtime_bounds_refusal() -> None:
    def action(indices: jax.Array) -> jax.Array:
        relation = phx.sparse.EdgeRelation(
            indices,
            jnp.asarray([0, 1], dtype=jnp.int32),
            source_size=3,
            target_size=2,
        )
        operator = phx.sparse.SparseLinearMap(
            relation,
            jnp.asarray([2.0, 3.0], dtype=jnp.float64),
            operator_id="traced-route-bounds",
        )
        return operator.mv(jnp.asarray([1.0, 2.0, 3.0]))

    compiled = eqx.filter_jit(action)
    np.testing.assert_allclose(compiled(jnp.asarray([0, 2], dtype=jnp.int32)), [2.0, 9.0])
    with pytest.raises(eqx.EquinoxRuntimeError):
        compiled(jnp.asarray([0, 3], dtype=jnp.int32)).block_until_ready()


def _duplicate_route_relation() -> phx.sparse.EdgeRelation:
    # Routes 1 and 2 share (target 1, source 1) and coalesce into one entry.
    return phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 1, 2], dtype=jnp.int32),
        jnp.asarray([0, 1, 1, 0], dtype=jnp.int32),
        source_size=3,
        target_size=2,
        valid=jnp.asarray([True, True, True, False]),
    )


@pytest.mark.parametrize("block_shape", (None, (2, 3)), ids=("scalar", "block"))
@pytest.mark.parametrize(
    ("capacity", "source_size", "target_size"),
    ((0, 3, 2), (2, 3, 2), (0, 0, 8), (0, 8, 0), (0, 0, 0)),
    ids=("empty", "invalid-padding", "empty-source", "empty-target", "empty-both"),
)
def test_sparse_coordinate_operator_without_valid_routes_applies_zero(
    block_shape: tuple[int, int] | None,
    capacity: int,
    source_size: int,
    target_size: int,
) -> None:
    # Empty support preserves both coordinate extents; non-finite coefficients
    # on invalid padding routes must stay inert.
    target_fiber, source_fiber = (1, 1) if block_shape is None else block_shape
    relation = phx.sparse.EdgeRelation(
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=jnp.int32),
        source_size=source_size,
        target_size=target_size,
        valid=jnp.zeros((capacity,), dtype=jnp.bool_),
    )
    operator = phx.sparse.SparseCoordinateOperator(
        relation,
        jnp.full(
            (capacity,) + (() if block_shape is None else block_shape),
            jnp.nan,
            dtype=jnp.float64,
        ),
        source=ArraySpace((source_size * source_fiber,), dtype=jnp.float64),
        target=ArraySpace((target_size * target_fiber,), dtype=jnp.float64),
        block_shape=block_shape,
    )
    source = jnp.ones((source_size * source_fiber,), dtype=jnp.float64)
    target = jnp.ones((target_size * target_fiber,), dtype=jnp.float64)

    expected_source = np.zeros((source_size * source_fiber,), dtype=np.float64)
    expected_target = np.zeros((target_size * target_fiber,), dtype=np.float64)
    np.testing.assert_array_equal(operator.mv(source), expected_target)
    np.testing.assert_array_equal(
        eqx.filter_jit(lambda op, value: op.mv(value))(operator, source),
        expected_target,
    )
    np.testing.assert_array_equal(operator.transpose_mv(target), expected_source)
    np.testing.assert_array_equal(operator.adjoint_mv(target), expected_source)
    np.testing.assert_array_equal(
        eqx.filter_jit(lambda op, value: op.transpose_mv(value))(operator, target),
        expected_source,
    )
    np.testing.assert_array_equal(
        eqx.filter_jit(lambda op, value: op.adjoint_mv(value))(operator, target),
        expected_source,
    )


def test_host_topology_plans_sparse_solves_with_operator_as_jit_argument() -> None:
    operator = phx.sparse.SparseCoordinateOperator(
        _duplicate_route_relation(),
        jnp.asarray([2.0, 1.0, 3.0, 5.0]),
        source=ArraySpace((3,), dtype=jnp.float64),
        target=ArraySpace((2,), dtype=jnp.float64),
    )
    # Operator: 4 float64 coefficients, 2x4 int32 indices, 4 bool masks (68 B);
    # row-gather layouts: int32 route and input slots of width 2 over 2 targets
    # and over 3 sources (80 B);
    # coalesced CSR: 2 float64 values, 2 int32 columns, 3 int32 row pointers (36 B).
    expected_bytes = 68 + 80 + 36
    traced_bytes: list[int] = []

    def record(value: Any) -> None:
        traced_bytes.append(phx.linalg.estimate_operator_action_cost(value).storage_bytes)

    eqx.filter_jit(record)(operator)
    assert phx.linalg.estimate_operator_action_cost(operator).storage_bytes == (
        expected_bytes
    )
    assert traced_bytes == [expected_bytes]

    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.LSMR(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1e-12, absolute=1e-14, max_steps=20
        ),
        failure=phx.linalg.FailurePolicy("status"),
    )
    compiled = eqx.filter_jit(
        lambda value, rhs: (
            phx.linalg.solve(
                phx.linalg.MinimumNormProblem(value), rhs, policy=policy
            ).value
        )
    )
    # A = [[2, 0, 0], [0, 4, 0]]: the minimum-norm solution leaves source 2 at 0.
    np.testing.assert_allclose(
        compiled(operator, jnp.asarray([1.0, 2.0])), [0.5, 0.5, 0.0], atol=1e-12
    )


def test_traced_topology_refuses_identity_and_cost_estimation() -> None:
    relation = _duplicate_route_relation()
    coefficients = jnp.ones((4,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="host-prepared topology"):
        eqx.filter_jit(lambda value: phx.sparse.SparseLinearMap(value, coefficients))(
            relation
        )
    with pytest.raises(ValueError, match="host-prepared topology"):
        eqx.filter_jit(
            lambda value: phx.linalg.estimate_operator_action_cost(
                phx.sparse.SparseLinearMap(value, coefficients, operator_id="traced")
            )
        )(relation)


def test_compile_time_sparse_pattern_admission_preserves_metric_action() -> None:
    metric = np.asarray(
        [[2.0, 0.5, 0.7], [0.5, 3.0, 1.0], [0.7, 1.0, 4.0]],
        dtype=np.float64,
    )
    rows, columns = np.triu_indices(3)
    mirror = rows != columns
    sources = np.concatenate((columns, rows[mirror])).astype(np.int32)
    targets = np.concatenate((rows, columns[mirror])).astype(np.int32)
    coefficients = np.concatenate(
        (metric[rows, columns], metric[rows[mirror], columns[mirror]])
    )

    def action(vector: jax.Array) -> jax.Array:
        with jax.ensure_compile_time_eval():
            relation = phx.sparse.EdgeRelation(
                sources,
                targets,
                source_size=3,
                target_size=3,
            )
            operator = phx.sparse.SparseLinearMap(relation, coefficients)
        return operator.mv(vector)

    vector = jnp.asarray([0.3, -0.7, 1.4])
    np.testing.assert_allclose(
        jax.jit(action)(vector), metric @ np.asarray(vector), atol=1e-13
    )


def test_packed_key_groups_compare_every_word_and_int64_event_order() -> None:
    maximum = 2**32 - 1
    keys = jnp.asarray(
        [
            [maximum, 0, 3, 1],
            [0, maximum, 3, 1],
            [0, maximum, 3, 2],
            [maximum, 0, 3, 1],
            [maximum, maximum, maximum, maximum],
        ],
        dtype=jnp.uint32,
    )
    groups = jax.jit(phx.sparse.KeyGroupPlan(5, 5, (maximum,) * 4).build)(
        keys,
        jnp.ones((5,), dtype=jnp.bool_),
        stable_ids=jnp.asarray(
            [2**40 + 5, 2**40 + 4, 2**40 + 3, 2**40 + 1, 2**40 + 2], dtype=jnp.int64
        ),
    )
    assert bool(groups.evidence.successful)
    np.testing.assert_array_equal(groups.storage_to_logical, [1, 2, 3, 0, 4])
    np.testing.assert_array_equal(groups.group_keys[:4], np.asarray(keys)[[1, 2, 0, 4]])
    np.testing.assert_array_equal(groups.group_counts, [1, 1, 2, 1, 0])
    query = jnp.asarray(
        [
            [maximum, maximum, maximum, maximum],
            [maximum, 0, 3, 2],
            [0, maximum, 3, 1],
            [0, maximum, 4, 1],
        ],
        dtype=jnp.uint32,
    )
    lookup = eqx.filter_jit(groups.lookup)(query)
    np.testing.assert_array_equal(lookup.supported, [True, False, True, False])
    np.testing.assert_array_equal(lookup.group_slots, [3, 0, 0, 0])


def test_packed_key_alignment_uses_identity_across_changed_capacities() -> None:
    bound = (2**32 - 1,) * 2
    previous = phx.sparse.KeyGroupPlan(3, 3, bound).build(
        jnp.asarray([[1, 7], [1, 8], [9, 0]], dtype=jnp.uint32),
        jnp.ones((3,), dtype=jnp.bool_),
    )
    candidate = phx.sparse.KeyGroupPlan(3, 4, bound).build(
        jnp.asarray([[9, 0], [1, 7], [2, 8]], dtype=jnp.uint32),
        jnp.ones((3,), dtype=jnp.bool_),
    )
    transition = phx.sparse.align_key_groups(previous, candidate)
    np.testing.assert_array_equal(transition.previous_retained, [True, False, True])
    np.testing.assert_array_equal(transition.previous_to_candidate, [0, 0, 2])
    np.testing.assert_array_equal(
        transition.candidate_retained, [True, False, True, False]
    )
    np.testing.assert_array_equal(transition.candidate_to_previous, [0, 0, 2, 0])
    assert bool(transition.topology_changed)
    assert bool(transition.successful)


def test_packed_key_case_lookup_preserves_query_mask_without_word_axis() -> None:
    plan = phx.sparse.KeyGroupPlan(2, 2, (2**32 - 1,) * 2, case_shape=(2,))
    groups = plan.build(
        jnp.asarray([[[1, 2], [3, 4]], [[3, 4], [5, 6]]], dtype=jnp.uint32),
        jnp.ones((2, 2), dtype=jnp.bool_),
    )
    lookup = groups.lookup(
        jnp.asarray([[[3, 4], [1, 2]], [[3, 4], [1, 2]]], dtype=jnp.uint32),
        valid=jnp.asarray([[True, False], [True, True]], dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(lookup.supported, [[True, False], [True, False]])
    np.testing.assert_array_equal(lookup.group_slots, [[1, 0], [0, 0]])


@pytest.mark.parametrize("empty", [True, False], ids=["zero-items", "all-padding"])
def test_packed_key_empty_support_does_not_match_zero_padding(empty: bool) -> None:
    capacity = 0 if empty else 2
    groups = phx.sparse.KeyGroupPlan(capacity, 2, (2**32 - 1,) * 2).build(
        jnp.zeros((capacity, 2), dtype=jnp.uint32),
        jnp.zeros((capacity,), dtype=jnp.bool_),
    )
    lookup = groups.lookup(jnp.zeros((3, 2), dtype=jnp.uint32))
    assert bool(groups.evidence.successful)
    np.testing.assert_array_equal(lookup.supported, [False, False, False])
    result, evidence = phx.sparse.reduce_key_groups(
        groups, jnp.full((capacity,), jnp.nan, dtype=jnp.float64)
    )
    np.testing.assert_array_equal(result.value, [0.0, 0.0])
    assert bool(evidence.successful)


def test_seeded_group_reduction_preserves_complex_cancellation() -> None:
    groups = phx.sparse.KeyGroupPlan(2, 1, 0).build(
        jnp.zeros((2,), dtype=jnp.int32), jnp.ones((2,), dtype=jnp.bool_)
    )
    initial = phx.sparse.KeyGroupAccumulation(
        jnp.asarray([1e16 + 1e16j], dtype=jnp.complex128),
        jnp.asarray([0.25 + 0.5j], dtype=jnp.complex128),
    )
    result, evidence = jax.jit(phx.sparse.reduce_key_groups)(
        groups,
        jnp.asarray([1 + 2j, -1e16 - 1e16j], dtype=jnp.complex128),
        initial=initial,
    )
    np.testing.assert_array_equal(result.high, [2j])
    np.testing.assert_array_equal(result.correction, [1.25 + 0.5j])
    np.testing.assert_array_equal(result.value, [1.25 + 2.5j])
    assert bool(evidence.successful)


@pytest.mark.parametrize("width", [1, 2, 4, 6], ids=["single", "pairs", "four", "whole"])
def test_seeded_group_reduction_replays_identically_across_chunk_width(
    width: int,
) -> None:
    bound = (2**32 - 1,) * 2
    event_keys = jnp.asarray(
        [[1, 9], [2, 8], [1, 9], [1, 9], [2, 8], [1, 9]], dtype=jnp.uint32
    )
    values = jnp.asarray(
        [1e16 + 1e16j, 5j, 1 + 2j, -1e16 - 1e16j, -2j, 4 + 8j], dtype=jnp.complex128
    )
    whole_groups = phx.sparse.KeyGroupPlan(6, 3, bound).build(
        event_keys,
        jnp.ones((6,), dtype=jnp.bool_),
        stable_ids=jnp.arange(6, dtype=jnp.int64),
    )
    whole, whole_evidence = phx.sparse.reduce_key_groups(whole_groups, values)
    previous = phx.sparse.KeyGroupPlan(0, 3, bound).build(
        jnp.zeros((0, 2), dtype=jnp.uint32), jnp.zeros((0,), dtype=jnp.bool_)
    )
    carry = phx.sparse.KeyGroupAccumulation(
        jnp.zeros((3,), dtype=jnp.complex128), jnp.zeros((3,), dtype=jnp.complex128)
    )
    for start in range(0, 6, width):
        count = min(width, 6 - start)
        candidate = phx.sparse.KeyGroupPlan(3 + count, 3, bound).build(
            jnp.concatenate((previous.group_keys, event_keys[start : start + count])),
            jnp.concatenate((previous.group_active, jnp.ones((count,), dtype=jnp.bool_))),
            stable_ids=jnp.concatenate(
                (
                    jnp.arange(-3, 0, dtype=jnp.int64),
                    jnp.arange(start, start + count, dtype=jnp.int64),
                )
            ),
        )
        lookup = previous.lookup(candidate.group_keys, valid=candidate.group_active)
        seed = phx.sparse.KeyGroupAccumulation(
            jnp.where(lookup.supported, carry.high[lookup.group_slots], 0j),
            jnp.where(lookup.supported, carry.correction[lookup.group_slots], 0j),
        )
        carry, evidence = phx.sparse.reduce_key_groups(
            candidate,
            jnp.concatenate(
                (
                    jnp.full((3,), jnp.nan + 0j, dtype=jnp.complex128),
                    values[start : start + count],
                )
            ),
            initial=seed,
            value_valid=jnp.concatenate(
                (jnp.zeros((3,), dtype=jnp.bool_), jnp.ones((count,), dtype=jnp.bool_))
            ),
        )
        assert bool(evidence.successful), start
        previous = candidate
    assert bool(whole_evidence.successful)
    np.testing.assert_array_equal(carry.high, whole.high)
    np.testing.assert_array_equal(carry.correction, whole.correction)
    np.testing.assert_array_equal(carry.value, [5 + 10j, 3j, 0j])


@pytest.mark.parametrize("accumulation", ["fast", "deterministic", "compensated"])
def test_group_reduction_masks_nan_and_keeps_zero_seed_components(
    accumulation: phx.sparse.RelationAccumulation,
) -> None:
    groups = phx.sparse.KeyGroupPlan(3, 2, 1, case_shape=(2,)).build(
        jnp.asarray([[0, 1, 0], [0, 1, 0]], dtype=jnp.int32),
        jnp.asarray([[True, True, False], [True, True, False]], dtype=jnp.bool_),
    )
    high = jnp.asarray(
        [[[-0.0, 7.0], [2.0, 3.0]], [[4.0, -0.0], [5.0, 6.0]]], dtype=jnp.float64
    )
    correction = jnp.asarray(
        [[[-0.0, 0.25], [0.5, 0.75]], [[1.0, -0.0], [1.25, 1.5]]], dtype=jnp.float64
    )
    result, evidence = phx.sparse.reduce_key_groups(
        groups,
        jnp.asarray(
            [
                [[0.0, 0.0], [jnp.nan, jnp.nan], [jnp.nan, jnp.nan]],
                [[0.0, 0.0], [jnp.nan, jnp.nan], [jnp.nan, jnp.nan]],
            ],
            dtype=jnp.float64,
        ),
        accumulation=accumulation,
        initial=phx.sparse.KeyGroupAccumulation(high, correction),
        value_valid=jnp.asarray(
            [[True, False, True], [True, False, True]], dtype=jnp.bool_
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(result.high).view(np.uint64), np.asarray(high).view(np.uint64)
    )
    np.testing.assert_array_equal(
        np.asarray(result.correction).view(np.uint64),
        np.asarray(correction).view(np.uint64),
    )
    np.testing.assert_array_equal(evidence.finite, [True, True])
    np.testing.assert_array_equal(evidence.successful, [True, True])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_group_reduction_reports_active_nonfinite_values(value: float) -> None:
    groups = phx.sparse.KeyGroupPlan(1, 1, 0).build(
        jnp.asarray([0], dtype=jnp.int32), jnp.asarray([True], dtype=jnp.bool_)
    )
    _, evidence = phx.sparse.reduce_key_groups(
        groups, jnp.asarray([value], dtype=jnp.float64)
    )
    assert not bool(evidence.finite)
    assert not bool(evidence.successful)


@pytest.mark.parametrize("member_limit", [None, 1], ids=["groups", "members"])
def test_packed_group_capacity_failure_reaches_reduction_boundary(
    member_limit: int | None,
) -> None:
    keys = [[1, 0], [2, 0]] if member_limit is None else [[1, 0], [1, 0]]
    groups = phx.sparse.KeyGroupPlan(
        2, 1, (2**32 - 1,) * 2, maximum_group_size=member_limit
    ).build(jnp.asarray(keys, dtype=jnp.uint32), jnp.ones((2,), dtype=jnp.bool_))
    _, evidence = phx.sparse.reduce_key_groups(
        groups, jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    )
    assert not bool(evidence.successful)
    assert bool(groups.evidence.group_overflow) == (member_limit is None)
    assert bool(groups.evidence.member_overflow) == (member_limit is not None)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.int64, jnp.uint32])
def test_scalar_key_dtype_maximum_does_not_overflow_padding(dtype: DTypeLike) -> None:
    maximum = np.iinfo(dtype).max
    groups = phx.sparse.KeyGroupPlan(3, 3, int(maximum)).build(
        jnp.asarray([maximum, 0, maximum], dtype=dtype),
        jnp.asarray([True, True, False], dtype=jnp.bool_),
    )
    assert bool(groups.evidence.successful)
    lookup = groups.lookup(jnp.asarray([maximum, 1, 0], dtype=dtype))
    np.testing.assert_array_equal(lookup.supported, [True, False, True])
    np.testing.assert_array_equal(lookup.group_slots, [1, 0, 0])
    result, evidence = phx.sparse.reduce_key_groups(
        groups, jnp.asarray([7.0, 2.0, jnp.nan], dtype=jnp.float64)
    )
    np.testing.assert_array_equal(result.value, [2.0, 7.0, 0.0])
    assert bool(evidence.successful)


def test_packed_key_domain_and_stable_id_refusals_are_explicit() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        phx.sparse.KeyGroupPlan(1, 1, ())
    with pytest.raises(ValueError, match="uint32"):
        phx.sparse.KeyGroupPlan(1, 1, (2**32,))
    plan = phx.sparse.KeyGroupPlan(2, 2, (3, 7))
    with pytest.raises(TypeError, match="uint32"):
        plan.build(jnp.zeros((2, 2), dtype=jnp.int64), jnp.ones((2,), dtype=jnp.bool_))
    with pytest.raises(ValueError, match="shape"):
        plan.build(jnp.zeros((2,), dtype=jnp.uint32), jnp.ones((2,), dtype=jnp.bool_))
    invalid = plan.build(
        jnp.asarray([[3, 7], [3, 8]], dtype=jnp.uint32), jnp.ones((2,), dtype=jnp.bool_)
    )
    assert int(invalid.evidence.invalid_keys) == 1
    assert not bool(invalid.evidence.successful)
    duplicate_ids = plan.build(
        jnp.asarray([[0, 0], [1, 0]], dtype=jnp.uint32),
        jnp.ones((2,), dtype=jnp.bool_),
        stable_ids=jnp.asarray([2**40, 2**40], dtype=jnp.int64),
    )
    assert int(duplicate_ids.evidence.duplicate_stable_ids) == 1
    assert not bool(duplicate_ids.evidence.successful)


def test_group_reduction_adds_case_batched_trailing_values_to_each_seed() -> None:
    groups = phx.sparse.KeyGroupPlan(4, 2, 1, case_shape=(2,)).build(
        jnp.asarray([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=jnp.int32),
        jnp.ones((2, 4), dtype=jnp.bool_),
    )
    high = jnp.asarray(
        [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]], dtype=jnp.float64
    )
    values = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
        ],
        dtype=jnp.float64,
    )
    result, evidence = jax.jit(phx.sparse.reduce_key_groups)(
        groups,
        values,
        initial=phx.sparse.KeyGroupAccumulation(high, jnp.zeros_like(high)),
    )
    np.testing.assert_array_equal(
        result.value, [[[20.0, 32.0], [36.0, 48.0]], [[72.0, 84.0], [96.0, 108.0]]]
    )
    np.testing.assert_array_equal(evidence.successful, [True, True])


def test_group_reduction_refuses_seed_layout_and_value_mask_mismatch() -> None:
    groups = phx.sparse.KeyGroupPlan(2, 1, 0).build(
        jnp.zeros((2,), dtype=jnp.int32), jnp.ones((2,), dtype=jnp.bool_)
    )
    values = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    with pytest.raises(ValueError, match="initial"):
        phx.sparse.reduce_key_groups(
            groups,
            values,
            initial=phx.sparse.KeyGroupAccumulation(
                jnp.zeros((2,), dtype=jnp.float64), jnp.zeros((2,), dtype=jnp.float64)
            ),
        )
    with pytest.raises(TypeError, match="same dtype"):
        phx.sparse.reduce_key_groups(
            groups,
            values,
            initial=phx.sparse.KeyGroupAccumulation(
                jnp.zeros((1,), dtype=jnp.float32), jnp.zeros((1,), dtype=jnp.float32)
            ),
        )
    with pytest.raises(ValueError, match="value_valid"):
        phx.sparse.reduce_key_groups(
            groups, values, value_valid=jnp.ones((2, 1), dtype=jnp.bool_)
        )


def test_group_reduction_reports_finite_input_arithmetic_overflow() -> None:
    groups = phx.sparse.KeyGroupPlan(2, 1, 0).build(
        jnp.zeros((2,), dtype=jnp.int32), jnp.ones((2,), dtype=jnp.bool_)
    )
    maximum = np.finfo(np.float64).max
    _, evidence = phx.sparse.reduce_key_groups(
        groups, jnp.asarray([maximum, maximum], dtype=jnp.float64)
    )
    assert not bool(evidence.finite)
    assert not bool(evidence.successful)


def test_concrete_row_operator_constructed_under_jit_preserves_actions_and_jvp() -> None:
    relation = phx.sparse.RowRelation(
        np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        source_size=3,
    )
    space = ArraySpace((3,), dtype=jnp.float64)
    coefficients = jnp.asarray([[2.0, -1.0], [3.0, 4.0], [-2.0, 1.0]], dtype=jnp.float64)
    source = jnp.asarray([1.0, 2.0, -1.0], dtype=jnp.float64)
    target = jnp.asarray([4.0, -3.0, 2.0], dtype=jnp.float64)
    matrix = np.asarray([[2.0, -1.0, 0.0], [0.0, 3.0, 4.0], [1.0, 0.0, -2.0]])

    @eqx.filter_jit
    def forward(values: Array, argument: Array) -> Array:
        operator = phx.sparse.SparseCoordinateOperator(
            relation, values, source=space, target=space, operator_id="traced-row-forward"
        )
        return operator.mv(argument)

    @eqx.filter_jit
    def reverse(values: Array, argument: Array) -> Array:
        operator = phx.sparse.SparseCoordinateOperator(
            relation, values, source=space, target=space, operator_id="traced-row-reverse"
        )
        return operator.transpose_mv(argument)

    np.testing.assert_allclose(forward(coefficients, source), matrix @ np.asarray(source))
    np.testing.assert_allclose(
        reverse(coefficients, target), matrix.T @ np.asarray(target)
    )
    _, tangent = jax.jvp(
        lambda values: forward(values, source),
        (coefficients,),
        (jnp.ones_like(coefficients),),
    )
    np.testing.assert_allclose(tangent, np.asarray([3.0, 1.0, 0.0]))


@pytest.mark.parametrize("accumulation", ["fast", "deterministic", "compensated"])
@pytest.mark.parametrize("seeded", [False, True], ids=["unseeded", "seeded"])
def test_valid_zero_group_event_preserves_coordinate_and_mixed_derivatives(
    accumulation: phx.sparse.RelationAccumulation, seeded: bool
) -> None:
    groups = phx.sparse.KeyGroupPlan(2, 1, 0).build(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.asarray([True, False], dtype=jnp.bool_),
    )
    initial = (
        phx.sparse.KeyGroupAccumulation(
            jnp.asarray([4.0], dtype=jnp.float64),
            jnp.asarray([0.25], dtype=jnp.float64),
        )
        if seeded
        else None
    )

    def reduced(theta: Array, coordinate: Array) -> Array:
        active = theta * coordinate + coordinate * coordinate
        padded = 7.0 * theta * coordinate
        result, _ = phx.sparse.reduce_key_groups(
            groups,
            jnp.stack((active, padded)),
            accumulation=accumulation,
            initial=initial,
        )
        return result.value[0]

    theta = jnp.asarray(2.0, dtype=jnp.float64)
    coordinate = jnp.asarray(0.0, dtype=jnp.float64)
    coordinate_gradient = jax.grad(reduced, argnums=1)
    value, tangent = jax.jvp(
        lambda point: reduced(theta, point), (coordinate,), (jnp.ones_like(coordinate),)
    )
    _, pullback = jax.vjp(lambda point: reduced(theta, point), coordinate)
    np.testing.assert_array_equal(value, 4.25 if seeded else 0.0)
    np.testing.assert_array_equal(tangent, 2.0)
    np.testing.assert_array_equal(pullback(jnp.ones_like(value))[0], 2.0)
    np.testing.assert_array_equal(jax.jit(coordinate_gradient)(theta, coordinate), 2.0)
    np.testing.assert_array_equal(
        jax.jit(jax.grad(coordinate_gradient, argnums=0))(theta, coordinate), 1.0
    )


@pytest.mark.parametrize("accumulation", ["fast", "deterministic", "compensated"])
def test_cancelling_group_subtotal_preserves_parameter_tangent(
    accumulation: phx.sparse.RelationAccumulation,
) -> None:
    groups = phx.sparse.KeyGroupPlan(2, 1, 0).build(
        jnp.zeros((2,), dtype=jnp.int32), jnp.ones((2,), dtype=jnp.bool_)
    )
    initial = phx.sparse.KeyGroupAccumulation(
        jnp.asarray([3.0], dtype=jnp.float64),
        jnp.asarray([0.5], dtype=jnp.float64),
    )

    def reduced(parameter: Array) -> Array:
        result, _ = phx.sparse.reduce_key_groups(
            groups,
            jnp.stack((parameter, jnp.asarray(-2.0, dtype=parameter.dtype))),
            accumulation=accumulation,
            initial=initial,
        )
        return result.value[0]

    parameter = jnp.asarray(2.0, dtype=jnp.float64)
    value, tangent = jax.jvp(reduced, (parameter,), (jnp.ones_like(parameter),))
    np.testing.assert_array_equal(value, 3.5)
    np.testing.assert_array_equal(tangent, 1.0)
    np.testing.assert_array_equal(jax.jit(jax.grad(reduced))(parameter), 1.0)


@pytest.mark.parametrize("accumulation", ["deterministic", "compensated"])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_route_accumulation_preserves_storage_order_masks_and_derivatives(
    accumulation: phx.sparse.RelationAccumulation,
    reduction: phx.sparse.RouteReduction,
) -> None:
    relation = phx.sparse.EdgeRelation(
        np.zeros((7,), dtype=np.int32),
        np.asarray([1, 0, 1, 0, 1, 0, 99], dtype=np.int32),
        source_size=1,
        target_size=3,
        valid=np.asarray([True] * 6 + [False]),
    )
    events = np.asarray([1e16, 4.0, 1.0, 5.0, -1e16, -9.0, np.nan])
    values = jnp.asarray(np.stack((events, -2.0 * events), axis=-1))

    def reduce(material: Array) -> Array:
        return phx.sparse.route_reduce(
            relation, material, reduction=reduction, accumulation=accumulation
        )

    residual = 1.0 if accumulation == "compensated" else 0.0
    divisor = 3.0 if reduction == "mean" else 1.0
    expected = np.asarray([[0.0, 0.0], [residual, -2.0 * residual], [0.0, 0.0]])
    np.testing.assert_array_equal(jax.jit(reduce)(values), expected / divisor)
    gradient = jax.jit(jax.grad(lambda material: jnp.sum(reduce(material))))(values)
    np.testing.assert_array_equal(
        gradient, np.asarray([[1.0 / divisor] * 2] * 6 + [[0.0, 0.0]])
    )


@pytest.mark.parametrize("accumulation", ["deterministic", "compensated"])
def test_ordered_row_reductions_and_batched_linear_actions(
    accumulation: phx.sparse.RelationAccumulation,
) -> None:
    indices = np.broadcast_to(np.asarray([0, 1, 2, 99], dtype=np.int32), (2, 2, 4))
    relation = phx.sparse.RowRelation(
        indices,
        source_size=3,
        case_shape=(2,),
        valid=np.broadcast_to(np.asarray([True, True, True, False]), indices.shape),
    )
    weights = np.broadcast_to(np.asarray([1e16, 1.0, -1e16, np.nan]), indices.shape)
    scales = np.asarray([1.0, 2.0, 4.0])
    sources = jnp.asarray(np.broadcast_to(scales[:, None, None, None], (3, 2, 3, 2)))
    batched_weights = jnp.asarray(np.broadcast_to(weights, (3,) + weights.shape))
    residual = 1.0 if accumulation == "compensated" else 0.0
    mean = phx.sparse.route_reduce(
        relation, weights, reduction="mean", accumulation=accumulation
    )
    np.testing.assert_array_equal(mean, np.full((2, 2), residual / 3.0))
    output = jax.jit(
        lambda coefficient, source: phx.sparse.linear_apply(
            relation, coefficient, source, accumulation=accumulation
        )
    )(batched_weights, sources)
    np.testing.assert_array_equal(
        output,
        np.broadcast_to((residual * scales)[:, None, None, None], (3, 2, 2, 2)),
    )


@pytest.mark.parametrize("accumulation", ["deterministic", "compensated"])
@pytest.mark.parametrize("rows", [False, True], ids=["edges", "rows"])
def test_ordered_route_reductions_accept_empty_routes(
    accumulation: phx.sparse.RelationAccumulation, rows: bool
) -> None:
    relation = (
        phx.sparse.RowRelation(
            np.full((3, 1), -1, dtype=np.int32),
            source_size=2,
            valid=np.zeros((3, 1), dtype=np.bool_),
        )
        if rows
        else phx.sparse.EdgeRelation(
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            source_size=2,
            target_size=3,
        )
    )
    values = jnp.zeros(relation.route_shape + (2,), dtype=jnp.float64)
    result = phx.sparse.route_reduce(
        relation, values, reduction="mean", accumulation=accumulation
    )
    np.testing.assert_array_equal(result, np.zeros((3, 2)))
    linear = phx.sparse.linear_apply(
        relation,
        jnp.zeros(relation.route_shape, dtype=jnp.float64),
        jnp.ones(relation.input_shape + (2,), dtype=jnp.float64),
        accumulation=accumulation,
    )
    np.testing.assert_array_equal(linear, np.zeros((3, 2)))


def test_sparse_route_accumulation_refuses_invalid_selectors_and_compensated_extrema() -> (
    None
):
    relation = phx.sparse.EdgeRelation(
        np.asarray([0], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        source_size=1,
        target_size=1,
    )
    values = jnp.ones((1,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="accumulation"):
        phx.sparse.route_reduce(relation, values, accumulation="unordered")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="accumulation"):
        phx.sparse.linear_apply(relation, values, values, accumulation="unordered")  # ty: ignore[invalid-argument-type]
    extrema: tuple[phx.sparse.RouteReduction, ...] = ("min", "max")
    for reduction in extrema:
        with pytest.raises(ValueError, match="sum and mean only"):
            phx.sparse.route_reduce(
                relation, values, reduction=reduction, accumulation="compensated"
            )


@pytest.mark.parametrize("accumulation", ["fast", "deterministic", "compensated"])
@pytest.mark.parametrize("rows", [False, True], ids=["edges", "rows"])
@pytest.mark.parametrize("batched", [False, True], ids=["unbatched", "batched"])
def test_linear_apply_source_adjoints_respect_accumulation(
    accumulation: phx.sparse.RelationAccumulation, rows: bool, batched: bool
) -> None:
    rng = np.random.default_rng(97)
    source_count, target_count, width = 8, 8, 64
    case_shape = (2,) if rows else ()
    batch_shape = (2,) if batched else ()
    target_indices = np.empty((0,), dtype=np.int32)
    if rows:
        indices = rng.integers(
            0, source_count, case_shape + (target_count, width), dtype=np.int32
        )
        valid = np.ones(indices.shape, dtype=np.bool_)
        valid[..., -1] = False
        indices[..., -1] = 99
        relation = phx.sparse.RowRelation(
            indices, source_size=source_count, case_shape=case_shape, valid=valid
        )
    else:
        indices = rng.integers(0, source_count, target_count * width, dtype=np.int32)
        target_indices = rng.integers(0, target_count, indices.size, dtype=np.int32)
        valid = np.ones(indices.shape, dtype=np.bool_)
        valid[-1] = False
        indices[-1], target_indices[-1] = 99, 99
        relation = phx.sparse.EdgeRelation(
            indices,
            target_indices,
            source_size=source_count,
            target_size=target_count,
            valid=valid,
        )
    coefficients = np.where(
        valid, rng.normal(scale=0.1, size=batch_shape + relation.route_shape), np.nan
    )
    source = rng.normal(size=batch_shape + case_shape + (source_count, 3))
    cotangent = rng.normal(size=batch_shape + case_shape + (target_count, 3))
    direction = rng.normal(size=source.shape)
    matrix = np.zeros(
        batch_shape + case_shape + (target_count, source_count), dtype=np.float64
    )
    # Independent case-local dense map, assembled from the original host routes.
    for batch in np.ndindex(batch_shape):
        if rows:
            for case in np.ndindex(case_shape):
                for target in range(target_count):
                    for slot in range(width):
                        route = case + (target, slot)
                        if valid[route]:
                            matrix[batch + case + (target, indices[route])] += (
                                coefficients[batch + route]
                            )
        else:
            for route in range(indices.size):
                if valid[route]:
                    matrix[batch + (target_indices[route], indices[route])] += (
                        coefficients[batch + (route,)]
                    )

    @jax.jit
    def evaluate(
        values: Array, weights: Array, seed: Array, tangent: Array
    ) -> tuple[Array, Array, Array, Array]:
        def action(value: Array) -> Array:
            return phx.sparse.linear_apply(
                relation, weights, value, accumulation=accumulation
            )

        output, pullback = jax.vjp(action, values)
        reverse = pullback(seed)[0]
        forward = jax.jvp(action, (values,), (tangent,))[1]
        hessian = jax.jvp(
            jax.grad(lambda value: jnp.sum(jnp.sin(action(value)) * seed)),
            (values,),
            (tangent,),
        )[1]
        return output, reverse, forward, hessian

    operands = tuple(map(jnp.asarray, (source, coefficients, cotangent, direction)))
    expected = evaluate(*operands)
    output = matrix @ source
    tangent = matrix @ direction
    transpose = np.swapaxes(matrix, -1, -2)
    oracle = (
        output,
        transpose @ cotangent,
        tangent,
        transpose @ (-cotangent * np.sin(output) * tangent),
    )
    for actual, reference in zip(expected, oracle, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    if accumulation != "fast":
        for _ in range(5):
            for actual, reference in zip(evaluate(*operands), expected, strict=True):
                np.testing.assert_array_equal(actual, reference)


def _segmented_relation(
    rows: bool,
) -> tuple[phx.sparse.SparseRelation, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return a relation with padding and an empty target, flat route targets,
    route validity and the target shape."""
    if rows:
        indices = np.asarray(
            [[[0, 1, 2], [2, 0, 1]], [[1, 1, 0], [0, 2, 1]]], dtype=np.int32
        )
        valid = np.ones(indices.shape, dtype=np.bool_)
        valid[0, 0, 2] = False
        valid[1, 1, :] = False
        indices[~valid] = 99
        relation = phx.sparse.RowRelation(
            indices, source_size=3, case_shape=(2,), valid=valid
        )
        targets = np.broadcast_to(np.arange(4).reshape((2, 2, 1)), indices.shape)
        return relation, targets, valid, (2, 2)
    targets = np.asarray([2, 0, 2, 0, 3, 2, 99], dtype=np.int32)
    valid = targets != 99
    relation = phx.sparse.EdgeRelation(
        np.where(valid, np.arange(7), 99).astype(np.int32),
        targets,
        source_size=7,
        target_size=4,
        valid=valid,
    )
    return relation, targets, valid, (4,)


@pytest.mark.parametrize("accumulation", ["fast", "deterministic", "compensated"])
@pytest.mark.parametrize("rows", [False, True], ids=["edges", "rows"])
def test_segmented_route_operations_match_independent_route_loops(
    accumulation: phx.sparse.RelationAccumulation, rows: bool
) -> None:
    relation, targets, valid, target_shape = _segmented_relation(rows)
    route_shape = relation.route_shape
    rng = np.random.default_rng(31)
    left = rng.normal(size=route_shape + (3,))
    right = rng.normal(size=route_shape + (3,))
    columns = rng.normal(size=route_shape + (3, 4))
    left[~valid], right[~valid], columns[~valid] = np.nan, np.inf, np.nan
    scalars = rng.normal(size=target_shape)
    vectors = rng.normal(size=target_shape + (3,))
    matrices = rng.normal(size=target_shape + (2, 3))

    target_count = int(np.prod(target_shape))
    dot = np.zeros(target_count)
    sums = np.zeros((target_count, 3))
    squares = np.zeros(target_count)
    counts = np.zeros(target_count, dtype=np.int32)
    scaled = np.zeros(route_shape + (3,))
    weighted = np.zeros(route_shape + (3,))
    mapped = np.zeros(route_shape + (2, 4))
    for route in np.ndindex(route_shape):
        if not valid[route]:
            continue
        target = int(targets[route])
        target_index = np.unravel_index(target, target_shape)
        dot[target] += np.sum(left[route] * right[route])
        sums[target] += left[route]
        squares[target] += np.sum(left[route] ** 2)
        counts[target] += 1
        scaled[route] = left[route] * scalars[target_index]
        weighted[route] = left[route] * vectors[target_index]
        mapped[route] = matrices[target_index] @ columns[route]
    available = counts > 0
    divisor = np.maximum(counts, 1)

    value, support = phx.sparse.route_dot(
        relation, jnp.asarray(left), jnp.asarray(right), accumulation=accumulation
    )
    np.testing.assert_allclose(value, dot.reshape(target_shape), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(support.counts, counts.reshape(target_shape))
    np.testing.assert_array_equal(support.available, available.reshape(target_shape))
    assert bool(support.successful)

    # The existing route mean couples to the same support: its unavailable
    # targets hold the inert zero fill rather than a statistic.
    mean = phx.sparse.route_reduce(
        relation, jnp.asarray(left), reduction="mean", accumulation=accumulation
    )
    expected_mean = np.where(available[:, None], sums / divisor[:, None], 0.0)
    np.testing.assert_allclose(
        mean, expected_mean.reshape(target_shape + (3,)), rtol=1e-12, atol=1e-12
    )

    root = phx.sparse.route_rms(relation, jnp.asarray(left), accumulation=accumulation)
    mean_square = np.where(available, squares / divisor, 0.0)
    np.testing.assert_allclose(
        root.value, np.sqrt(mean_square).reshape(target_shape), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        root.mean_square, mean_square.reshape(target_shape), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(root.regular, available.reshape(target_shape))
    np.testing.assert_array_equal(root.support.counts, counts.reshape(target_shape))

    for factors, expected in ((scalars, scaled), (vectors, weighted)):
        product = phx.sparse.route_multiply(
            relation, jnp.asarray(left), jnp.asarray(factors), accumulation=accumulation
        )
        np.testing.assert_allclose(product, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(np.asarray(product)[~valid], 0.0)
    # Non-square matrices pin the column-vector orientation ``M[target] @ v``.
    action = phx.sparse.route_matvec(
        relation, jnp.asarray(matrices), jnp.asarray(columns), accumulation=accumulation
    )
    assert action.shape == route_shape + (2, 4)
    np.testing.assert_allclose(action, mapped, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(action)[~valid], 0.0)


@pytest.mark.parametrize("accumulation", ["deterministic", "compensated"])
def test_ordered_segmented_dot_folds_components_and_replays_bitwise(
    accumulation: phx.sparse.RelationAccumulation,
) -> None:
    relation = phx.sparse.EdgeRelation(
        np.zeros((5,), dtype=np.int32),
        np.asarray([0, 1, 1, 1, 0], dtype=np.int32),
        source_size=1,
        target_size=2,
        valid=np.asarray([True, True, True, True, False]),
    )
    # Target 0 cancels across one route's payload components; target 1 cancels
    # across its routes within component 0.
    left = jnp.asarray(
        [
            [1e16, 1.0, -1e16],
            [1e16, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1e16, 0.0, 0.0],
            [np.nan, np.nan, np.nan],
        ]
    )
    right = jnp.where(jnp.isnan(left), jnp.inf, 1.0)
    evaluate = jax.jit(
        lambda first, second: phx.sparse.route_dot(
            relation, first, second, accumulation=accumulation
        )[0]
    )
    residual = 1.0 if accumulation == "compensated" else 0.0
    expected = evaluate(left, right)
    np.testing.assert_array_equal(expected, [residual, residual])
    for _ in range(3):
        np.testing.assert_array_equal(evaluate(left, right), expected)


def test_route_rms_declares_zero_norm_empty_and_extreme_scale_targets() -> None:
    targets = np.asarray([0, 0, 1, 1, 3, 4, 4, 99], dtype=np.int32)
    valid = targets != 99
    relation = phx.sparse.EdgeRelation(
        np.where(valid, 0, 99).astype(np.int32),
        targets,
        source_size=1,
        target_size=5,
        valid=valid,
    )
    values = jnp.asarray(
        [
            [1.0, 2.0],
            [-2.0, 0.5],
            [0.0, 0.0],
            [0.0, -0.0],
            [3e-200, 4e-200],
            [3e200, 4e200],
            [0.0, 0.0],
            [np.nan, np.inf],
        ]
    )
    result = jax.jit(lambda routes: phx.sparse.route_rms(relation, routes))(values)
    np.testing.assert_allclose(
        result.value,
        [np.sqrt(4.625), 0.0, 0.0, 5e-200, 5e200 / np.sqrt(2.0)],
        rtol=1e-15,
    )
    np.testing.assert_array_equal(result.support.counts, [2, 2, 0, 1, 2])
    np.testing.assert_array_equal(
        result.support.available, [True, True, False, True, True]
    )
    # Exactly-zero routes are available but irregular; the empty target is not
    # available. Extreme scales stay regular and representable.
    np.testing.assert_array_equal(result.regular, [True, False, False, True, True])
    np.testing.assert_allclose(result.mean_square[:3], [4.625, 0.0, 0.0], rtol=1e-15)
    # The mean square itself leaves the IEEE range; the root does not.
    np.testing.assert_array_equal(result.mean_square[3:], [0.0, np.inf])

    single = phx.sparse.EdgeRelation(
        np.zeros((1,), dtype=np.int32),
        np.zeros((1,), dtype=np.int32),
        source_size=1,
        target_size=1,
    )
    tiny = phx.sparse.route_rms(single, jnp.asarray([[3e-30, 4e-30]], dtype=jnp.float32))
    assert tiny.value.dtype == jnp.float32
    np.testing.assert_allclose(tiny.value, [5e-30], rtol=1e-6)
    complex_root = phx.sparse.route_rms(single, jnp.asarray([[3.0 + 4.0j, 0.0j]]))
    assert complex_root.value.dtype == jnp.float64
    np.testing.assert_allclose(complex_root.value, [5.0], rtol=1e-15)
    pair = jnp.asarray([[1.0 + 1.0j]])
    bilinear, _ = phx.sparse.route_dot(single, pair, pair)
    hermitian, _ = phx.sparse.route_dot(single, jnp.conj(pair), pair)
    np.testing.assert_allclose(bilinear, [2.0j], rtol=1e-15)
    np.testing.assert_allclose(hermitian, [2.0], rtol=1e-15)


def test_prepared_execution_reuses_grouping_for_segmented_operations() -> None:
    relation, _, valid, target_shape = _segmented_relation(False)
    plan = phx.sparse.RelationExecutionPlan()
    execution = plan.prepare(relation)
    assert plan.prepare(relation).execution_id == execution.execution_id
    rng = np.random.default_rng(5)
    traces: list[None] = []

    @jax.jit
    def evaluate(
        owner: phx.sparse.RelationExecutionState, left: Array, right: Array
    ) -> tuple[Array, ...]:
        traces.append(None)
        dot, support = phx.sparse.route_dot(
            owner, left, right, accumulation="deterministic"
        )
        root = phx.sparse.route_rms(owner, left, accumulation="deterministic")
        return dot, support.counts, support.available, root.value, root.mean_square

    for _ in range(2):
        left = rng.normal(size=valid.shape + (3,))
        right = rng.normal(size=valid.shape + (3,))
        left[~valid] = right[~valid] = np.nan
        prepared = evaluate(execution, jnp.asarray(left), jnp.asarray(right))
        direct_dot, direct_support = phx.sparse.route_dot(
            relation, left, right, accumulation="deterministic"
        )
        direct_root = phx.sparse.route_rms(relation, left, accumulation="deterministic")
        # Stable grouped order equals storage order within each target.
        for actual, reference in zip(
            prepared,
            (
                direct_dot,
                direct_support.counts,
                direct_support.available,
                direct_root.value,
                direct_root.mean_square,
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, reference)
    assert len(traces) == 1
    factors = jnp.asarray(rng.normal(size=target_shape))
    np.testing.assert_array_equal(
        phx.sparse.route_multiply(execution, left, factors),
        phx.sparse.route_multiply(relation, left, factors),
    )

    limited = phx.sparse.RelationExecutionPlan(maximum_active_targets=1).prepare(relation)
    refused_dot, refused_support = phx.sparse.route_dot(limited, left, right)
    refused_root = phx.sparse.route_rms(limited, left)
    assert not bool(refused_support.successful)
    np.testing.assert_array_equal(refused_support.counts, [2, 0, 3, 1])
    np.testing.assert_array_equal(refused_support.available, [False] * 4)
    np.testing.assert_array_equal(refused_dot, np.zeros(4))
    np.testing.assert_array_equal(refused_root.value, np.zeros(4))
    np.testing.assert_array_equal(refused_root.regular, [False] * 4)


def test_segmented_operations_refuse_invalid_layouts_and_indices() -> None:
    relation, _, _, _ = _segmented_relation(False)
    routes = jnp.ones((7, 3), dtype=jnp.float64)
    with pytest.raises(ValueError, match="Right route values"):
        phx.sparse.route_dot(relation, routes, jnp.ones((7, 2)))
    with pytest.raises(ValueError, match="must begin with shape"):
        phx.sparse.route_dot(relation, jnp.ones((6, 3)), jnp.ones((6, 3)))
    with pytest.raises(ValueError, match="leading route payload"):
        phx.sparse.route_multiply(relation, routes, jnp.ones((4, 2)))
    with pytest.raises(ValueError, match="Target matrices"):
        phx.sparse.route_matvec(relation, jnp.ones((4, 3)), routes)
    with pytest.raises(ValueError, match="must begin with shape"):
        phx.sparse.route_matvec(relation, jnp.ones((4, 2, 2)), routes)
    with pytest.raises(TypeError, match="floating or complex"):
        phx.sparse.route_rms(relation, jnp.ones((7, 3), dtype=jnp.int32))
    with pytest.raises(TypeError, match="RelationExecutionState"):
        phx.sparse.route_support(object())  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="accumulation"):
        phx.sparse.route_rms(relation, routes, accumulation="unordered")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="outside"):
        phx.sparse.EdgeRelation(
            np.asarray([0], dtype=np.int32),
            np.asarray([4], dtype=np.int32),
            source_size=1,
            target_size=4,
        )

    def traced(target_indices: Array) -> Array:
        traced_relation = phx.sparse.EdgeRelation(
            jnp.zeros((2,), dtype=jnp.int32),
            target_indices,
            source_size=1,
            target_size=2,
        )
        return phx.sparse.route_dot(traced_relation, jnp.ones((2, 3)), jnp.ones((2, 3)))[
            0
        ]

    compiled = eqx.filter_jit(traced)
    np.testing.assert_array_equal(
        compiled(jnp.asarray([0, 1], dtype=jnp.int32)), [3.0, 3.0]
    )
    with pytest.raises(eqx.EquinoxRuntimeError):
        compiled(jnp.asarray([0, 2], dtype=jnp.int32)).block_until_ready()
