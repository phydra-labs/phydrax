#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import tracemalloc

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.sparse._linear import _SparseStoragePlan


la = phx.linalg


def _laplacian(count: int) -> la.AbstractSparseLinearOperator:
    """Nonsymmetric-stored 5-point operator on a ``count x count`` grid."""
    index = np.arange(count * count).reshape((count, count))
    rows = [index.ravel()]
    columns = [index.ravel()]
    values = [np.full(index.size, 4.2)]
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        source = index[
            max(shift[0], 0) : count + min(shift[0], 0),
            max(shift[1], 0) : count + min(shift[1], 0),
        ]
        target = index[
            max(-shift[0], 0) : count + min(-shift[0], 0),
            max(-shift[1], 0) : count + min(-shift[1], 0),
        ]
        rows.append(target.ravel())
        columns.append(source.ravel())
        values.append(np.full(source.size, -1.0 if shift[0] else -0.9))
    relation = phx.sparse.EdgeRelation(
        np.concatenate(columns).astype(np.int32),
        np.concatenate(rows).astype(np.int32),
        source_size=index.size,
        target_size=index.size,
    )
    space = la.ArraySpace((index.size,), dtype=np.float64)
    return phx.sparse.SparseCoordinateOperator(
        relation, jnp.asarray(np.concatenate(values)), source=space, target=space
    )


def _sparse_map(
    matrix: jax.Array,
    /,
    *,
    properties: la.OperatorProperties | None = None,
) -> phx.sparse.SparseLinearMap:
    rows, columns = jnp.nonzero(matrix)
    relation = phx.sparse.EdgeRelation(
        columns,
        rows,
        source_size=matrix.shape[1],
        target_size=matrix.shape[0],
    )
    return phx.sparse.SparseLinearMap(
        relation,
        matrix[rows, columns],
        properties=properties,
    )


def _policy(
    preconditioner: la.AbstractPreconditioner, budget: int
) -> la.LinearSolvePolicy:
    return la.LinearSolvePolicy(
        la.FGMRES(restart=20),
        tolerance=la.TolerancePolicy(relative=1e-12, absolute=0.0, max_steps=20),
        preconditioning=la.PreconditioningPolicy(preconditioner, side="right"),
        differentiation=la.DifferentiationPolicy("none"),
        resources=la.SolveResourcePolicy(preconditioner_bytes=budget),
    )


def test_prepared_sparse_lu_preconditioner_is_charged_its_stored_bytes() -> None:
    operator = _laplacian(40)
    size = operator.source.size
    symbolic = la.prepare_sparse_factorization(
        operator, la.SparseFactorizationPolicy("lu", ordering="reverse-cuthill-mckee")
    )
    factor = la.refresh_sparse_factorization(symbolic, operator)
    preconditioner = la.SparseFactorizationPreconditioner(
        operator,
        factor,
        properties=la.PreconditionerProperties(linear=True, stationary=True),
        preconditioner_id="laplacian-lu",
    )
    stored = _stored_bytes(factor)
    plan = la.plan(la.LinearSystem(operator), _policy(preconditioner, stored))
    assert plan.preconditioner_plan is not None
    assert plan.preconditioner_plan.cost.storage_bytes == stored
    rhs = jnp.linspace(-1.0, 1.0, size)
    result = la.solve(
        la.LinearSystem(operator), rhs, policy=_policy(preconditioner, stored)
    )
    assert bool(result.successful)
    assert int(result.diagnostics.iterations) <= 2
    np.testing.assert_allclose(operator.mv(result.value), rhs, atol=1e-10)
    with pytest.raises(ValueError, match=f"requires {stored} preconditioner state bytes"):
        la.plan(la.LinearSystem(operator), _policy(preconditioner, stored - 1))


@pytest.mark.parametrize("kind", ("lu", "ilu", "ic"))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_sparse_factor_template_binds_traced_topology_without_replanning(
    kind: str,
    dtype: jax.typing.DTypeLike,
) -> None:
    size = 6
    diagonal = np.arange(size, dtype=np.int32)
    sources = np.concatenate((diagonal, diagonal[:-1], diagonal[1:]))
    targets = np.concatenate((diagonal, diagonal[1:], diagonal[:-1]))
    relation = phx.sparse.EdgeRelation(
        sources, targets, source_size=size, target_size=size
    )
    storage_plan = _SparseStoragePlan(relation)
    space = la.ArraySpace((size,), dtype=dtype)
    values = jnp.asarray(
        np.concatenate(
            (np.full(size, 3.0), np.full(size - 1, -0.8), np.full(size - 1, -0.8))
        ),
        dtype=dtype,
    )
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
        },
    )
    operator = phx.sparse.SparseCoordinateOperator(
        relation,
        values,
        source=space,
        target=space,
        properties=properties,
        operator_id="traced-sparse-factor-template",
        storage_plan=storage_plan,
    )
    builder = {
        "lu": la.SparseFactorizationPreconditionerBuilder(
            la.SparseFactorizationPolicy("lu")
        ),
        "ilu": la.ILUPreconditionerBuilder(),
        "ic": la.IncompleteCholeskyPreconditionerBuilder(),
    }[kind]
    policy = la.LinearSolvePolicy(
        la.GMRES(restart=4),
        tolerance=la.TolerancePolicy(
            relative=1e-6 if dtype == jnp.float32 else 1e-12,
            absolute=1e-6 if dtype == jnp.float32 else 1e-12,
            max_steps=8,
        ),
        preconditioning=la.PreconditioningPolicy(builder),
    )
    template = la.prepare_template(la.LinearSystem(operator), policy)
    numeric = eqx.tree_at(lambda value: value.coefficients, operator, 1.1 * values)
    rhs = jnp.linspace(-1.0, 1.0, size, dtype=dtype)

    @eqx.filter_jit
    def bind_and_solve(
        symbolic: la.LinearSolveTemplate,
        current: phx.sparse.SparseCoordinateOperator,
        right: jax.Array,
    ) -> la.LinearSolveResult:
        prepared = la.bind_numeric(symbolic, la.LinearSystem(current))
        return la.solve(prepared, right)

    result = bind_and_solve(template, numeric, rhs)
    assert bool(result.successful)
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[targets, sources] = np.asarray(numeric.coefficients)
    np.testing.assert_allclose(
        result.value,
        np.linalg.solve(matrix, np.asarray(rhs)),
        rtol=1e-5 if dtype == jnp.float32 else 1e-11,
        atol=1e-6 if dtype == jnp.float32 else 1e-12,
    )
    changed = eqx.tree_at(
        lambda value: value.relation.source_indices,
        numeric,
        numeric.relation.source_indices.at[0].set(1),
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="unchanged relation routes"):
        jax.block_until_ready(bind_and_solve(template, changed, rhs).value)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_sparse_factor_template_refuses_wider_numeric_dtype(
    dtype: jax.typing.DTypeLike,
) -> None:
    operator = _sparse_map(jnp.asarray([[3.0, -0.5], [-0.5, 2.0]], dtype=jnp.float32))
    initial = la.prepare_sparse_factorization(
        operator, la.SparseFactorizationPolicy("lu")
    )
    builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("lu", max_factor_bytes=initial.factor_bytes)
    )
    policy = la.LinearSolvePolicy(
        la.GMRES(restart=2),
        preconditioning=la.PreconditioningPolicy(builder),
        resources=la.SolveResourcePolicy(preconditioner_bytes=initial.factor_bytes),
    )
    template = la.prepare_template(la.LinearSystem(operator), policy)
    numeric = eqx.tree_at(
        lambda value: value.coefficients, operator, operator.coefficients.astype(dtype)
    )
    with pytest.raises(ValueError, match="unchanged value dtype"):
        la.bind_numeric(template, la.LinearSystem(numeric))
    with pytest.raises(ValueError, match="unchanged value dtype"):
        jax.jit(lambda values: la.refresh_sparse_factorization_values(initial, values))(
            operator.sparse_storage().values.astype(dtype)
        )


def test_sparse_factor_planned_setup_reprepares_only_for_new_identity() -> None:
    operator = _sparse_map(jnp.asarray([[3.0, -0.5], [-0.5, 2.0]], dtype=jnp.float64))
    other = _sparse_map(jnp.asarray([[4.0, 0.0], [-0.2, 3.0]], dtype=jnp.float64))
    assert operator.operator_id != other.operator_id
    builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("lu")
    )
    planned = builder.plan_setup(operator)
    materialization = la.MaterializationPolicy()
    matching = builder.prepare_planned(planned, operator, materialization=materialization)
    assert matching.factorization.plan is planned.construction_for(operator.operator_id)
    fresh = builder.prepare_planned(planned, other, materialization=materialization)
    assert (
        fresh.factorization.plan.input_pattern_id
        != matching.factorization.plan.input_pattern_id
    )
    rhs = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    np.testing.assert_allclose(other.mv(fresh.apply(rhs)), rhs, atol=1e-12)
    result = la.solve(
        la.LinearSystem(other),
        rhs,
        policy=_policy(fresh, _stored_bytes(fresh.factorization)),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(other.mv(result.value), rhs, atol=1e-12)
    with pytest.raises(ValueError, match="unchanged CSR pattern"):
        builder.refresh(matching, other, materialization=materialization)
    corrupt = la.PlannedPreconditionerSetup(
        planned.cost, construction=object(), content_id=operator.operator_id
    )
    with pytest.raises(ValueError, match="requires its planned construction"):
        builder.prepare_planned(corrupt, operator, materialization=materialization)


def test_sparse_lu_zero_upper_width_preserves_pivot_evidence() -> None:
    operator = _sparse_map(jnp.asarray([[4.0, 0.0], [-0.2, 3.0]], dtype=jnp.float64))
    plan = la.prepare_sparse_factorization(operator, la.SparseFactorizationPolicy("lu"))
    assert plan.upper_width == 0
    factor = la.refresh_sparse_factorization(plan, operator)
    assert int(factor.status) == int(la.SparseFactorizationStatus.SUCCESS)
    assert float(factor.diagnostics.minimum_pivot) == 3.0
    assert int(factor.diagnostics.replaced_pivots) == 0
    assert bool(factor.diagnostics.finite)
    rhs = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    solved = factor.solve(rhs)
    assert bool(solved.success)
    np.testing.assert_allclose(operator.mv(solved.value), rhs, atol=1e-12)
    singular = eqx.tree_at(
        lambda value: value.coefficients,
        operator,
        operator.coefficients.at[-1].set(0.0),
    )
    failed = la.refresh_sparse_factorization(plan, singular)
    assert int(failed.status) == int(la.SparseFactorizationStatus.ZERO_PIVOT)
    assert int(failed.diagnostics.replaced_pivots) == 0
    assert not bool(failed.solve(rhs).success)


@pytest.mark.parametrize("kind", ("lu", "cholesky"))
def test_sparse_factor_adjoint_solve_matches_dense(
    kind: la.SparseFactorizationKind,
) -> None:
    matrix = jnp.asarray(
        [[5.0 + 1.0j, 0.0, 0.3], [1.0j, 4.0 - 0.5j, 0.0], [0.5, 0.2, 6.0 + 0.2j]],
        dtype=jnp.complex128,
    )
    properties = None
    if kind == "cholesky":
        matrix = jnp.conj(matrix.T) @ matrix + jnp.eye(3, dtype=jnp.complex128)
        properties = la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        )
    operator = _sparse_map(matrix, properties=properties)
    policy = la.SparseFactorizationPolicy(kind, ordering="approximate-minimum-degree")
    factor = la.factorize_sparse(operator, policy)
    rhs = jnp.asarray(
        [[1.0 + 0.2j, 0.4], [-0.5, 1.5j], [2.0j, -0.2]], dtype=jnp.complex128
    )
    solved = jax.jit(lambda value: factor.solve(value, adjoint=True))(rhs)
    expected = np.linalg.solve(np.asarray(matrix).conj().T, np.asarray(rhs))

    assert bool(solved.success)
    np.testing.assert_allclose(solved.value, expected, rtol=1e-12, atol=1e-12)
    action = la.SparseFactorizationPreconditionerBuilder(policy).prepare(
        operator, materialization=la.MaterializationPolicy()
    )
    np.testing.assert_allclose(
        action.apply_adjoint(rhs[:, 0]), expected[:, 0], rtol=1e-12, atol=1e-12
    )


def test_builder_estimate_matches_prepared_plan_storage() -> None:
    operator = _laplacian(24)
    builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("lu", ordering="reverse-cuthill-mckee")
    )
    cost = builder.cost_for(operator)
    symbolic = la.prepare_sparse_factorization(operator, builder.policy())
    values_bytes = symbolic.factor_indices.size * 8
    assert cost.storage_bytes == _stored_bytes(symbolic) + values_bytes
    assert cost.preparation_workspace_bytes > values_bytes


def test_symbolic_plan_bytes_scale_with_factor_nonzeros_not_updates() -> None:
    # With a banded RCM factor of bandwidth b ~ count, nnz(L+U) ~ count^3
    # while the elimination performs ~ count^4 updates.
    per_entry = []
    for count in (12, 24):
        symbolic = la.prepare_sparse_factorization(
            _laplacian(count),
            la.SparseFactorizationPolicy("lu", ordering="reverse-cuthill-mckee"),
        )
        per_entry.append(_stored_bytes(symbolic) / symbolic.factor_indices.size)
    assert per_entry[1] <= per_entry[0]
    assert per_entry[1] < 64.0


@pytest.mark.parametrize("kind", ("lu", "cholesky"))
def test_complete_sparse_factorization_refuses_star_fill_before_quadratic_allocation(
    kind: la.SparseFactorizationKind,
) -> None:
    size = 2_048
    vertices = jnp.arange(size, dtype=jnp.int32)
    leaves = jnp.arange(1, size, dtype=jnp.int32)
    rows = jnp.concatenate((vertices, jnp.zeros_like(leaves), leaves))
    columns = jnp.concatenate((vertices, leaves, jnp.zeros_like(leaves)))
    relation = phx.sparse.EdgeRelation(
        columns,
        rows,
        source_size=size,
        target_size=size,
    )
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
        },
    )
    operator = phx.sparse.SparseLinearMap(
        relation,
        jnp.ones(rows.shape, dtype=jnp.float64),
        properties=properties,
    )
    builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy(
            kind,
            max_factor_nnz=20_000,
            max_factor_bytes=8 * 1024 * 1024,
            max_symbolic_work=1_000_000,
        )
    )
    ceiling = la.MaterializationPolicy(
        max_entries=1_000_000,
        max_bytes=8 * 1024 * 1024,
    )

    tracemalloc.start()
    try:
        estimate = builder.cost_for(operator, materialization=ceiling)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert not estimate.accepted
    assert "factor_nnz requires 20001" in estimate.reason
    assert "factor_nnz=20001/20000" in estimate.reason
    assert "factor_bytes=" in estimate.reason
    assert "symbolic_work=" in estimate.reason
    assert peak_bytes < 32 * 1024 * 1024


def test_admitted_complete_and_incomplete_star_factorizations_retain_reference() -> None:
    matrix = jnp.asarray(
        [
            [8.0, 1.0, 1.0, 1.0],
            [1.0, 4.0, 0.0, 0.0],
            [1.0, 0.0, 5.0, 0.0],
            [1.0, 0.0, 0.0, 6.0],
        ],
        dtype=jnp.float64,
    )
    operator = _sparse_map(matrix)
    right_hand_side = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)

    complete_policy = la.SparseFactorizationPolicy(
        "lu",
        max_factor_nnz=100,
        max_factor_bytes=1_000_000,
        max_symbolic_work=10_000,
    )
    complete = la.factorize_sparse(operator, complete_policy)
    complete_cost = la.SparseFactorizationPreconditionerBuilder(complete_policy).cost_for(
        operator
    )
    incomplete = la.factorize_sparse(
        operator,
        la.SparseFactorizationPolicy(
            "lu",
            fill_level=0,
            max_factor_nnz=100,
            max_factor_bytes=1_000_000,
            max_symbolic_work=10_000,
        ),
    )

    expected = jnp.linalg.solve(matrix, right_hand_side)
    assert complete.plan.factor_nnz == 16
    assert complete.plan.factor_bytes <= 1_000_000
    assert complete.plan.symbolic_work <= 10_000
    assert complete_cost.accepted
    assert complete_cost.storage_bytes == complete.plan.factor_bytes
    assert jnp.allclose(complete.solve(right_hand_side).value, expected)
    assert incomplete.plan.factor_nnz == 10
    assert incomplete.status == int(la.SparseFactorizationStatus.SUCCESS)


def test_matrix_free_solve_admits_sparse_ilu_under_its_preconditioner_budget() -> None:
    # A matrix-free Krylov solve forbids dense materialization of the system
    # operator. Factoring the stored sparse setup operator is not a dense
    # materialization: the factor is charged to preconditioner_bytes instead.
    operator = _laplacian(12)
    builder = la.ILUPreconditionerBuilder()
    stored = builder.cost_for(operator).storage_bytes
    assert stored > 16

    def policy(budget: int) -> la.LinearSolvePolicy:
        return la.LinearSolvePolicy(
            la.GMRES(restart=40),
            tolerance=la.TolerancePolicy(relative=1e-12, absolute=0.0, max_steps=200),
            preconditioning=la.PreconditioningPolicy(
                builder, setup_operator=operator, side="right"
            ),
            materialization=la.MaterializationPolicy(max_entries=1, max_bytes=16),
            resources=la.SolveResourcePolicy(preconditioner_bytes=budget),
            failure=la.FailurePolicy("status"),
        )

    problem = la.LinearSystem(operator)
    right_hand_side = jnp.cos(jnp.arange(operator.source.size, dtype=jnp.float64))
    result = la.solve(problem, right_hand_side, policy=policy(stored))
    assert bool(result.successful)
    np.testing.assert_allclose(
        operator.mv(result.value), right_hand_side, rtol=1e-10, atol=1e-10
    )
    with pytest.raises(ValueError, match="preconditioner state bytes, exceeding"):
        la.plan(problem, policy(stored - 1))


def test_sparse_symbolic_retained_byte_cap_reports_observed_and_limit() -> None:
    operator = _laplacian(4)
    baseline = la.prepare_sparse_factorization(
        operator,
        la.SparseFactorizationPolicy(
            "lu",
            ordering="reverse-cuthill-mckee",
            max_factor_nnz=10_000,
            max_factor_bytes=1_000_000,
            max_symbolic_work=1_000_000,
        ),
    )
    limit = baseline.factor_bytes - 1

    with pytest.raises(
        la.LinearCapabilityError,
        match=rf"factor_bytes requires {baseline.factor_bytes}, exceeding limit {limit}",
    ):
        la.prepare_sparse_factorization(
            operator,
            la.SparseFactorizationPolicy(
                "lu",
                ordering="reverse-cuthill-mckee",
                max_factor_nnz=10_000,
                max_factor_bytes=limit,
                max_symbolic_work=1_000_000,
            ),
        )


def test_sparse_symbolic_work_cap_reports_observed_and_limit() -> None:
    operator = _laplacian(4)
    baseline = la.prepare_sparse_factorization(
        operator,
        la.SparseFactorizationPolicy(
            "lu",
            ordering="reverse-cuthill-mckee",
            max_factor_nnz=10_000,
            max_factor_bytes=1_000_000,
            max_symbolic_work=1_000_000,
        ),
    )
    limit = baseline.symbolic_work - 1

    with pytest.raises(
        la.LinearCapabilityError,
        match=rf"symbolic_work requires {baseline.symbolic_work}, exceeding limit {limit}",
    ):
        la.prepare_sparse_factorization(
            operator,
            la.SparseFactorizationPolicy(
                "lu",
                ordering="reverse-cuthill-mckee",
                max_factor_nnz=10_000,
                max_factor_bytes=1_000_000,
                max_symbolic_work=limit,
            ),
        )


def _stored_bytes(tree: object) -> int:
    arrays = {id(x): x for x in jax.tree.leaves(tree) if isinstance(x, jax.Array)}
    return sum(leaf.size * leaf.dtype.itemsize for leaf in arrays.values())
