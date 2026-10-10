#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import tracemalloc
from typing import NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


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


def _congruence_action(
    action: (
        la.SparseFactorizationPreconditioner | la.SparseFactorCongruencePreconditioner
    ),
) -> la.SparseFactorCongruencePreconditioner:
    if not isinstance(action, la.SparseFactorCongruencePreconditioner):
        raise TypeError("LU-congruence form must publish its congruence preconditioner.")
    return action


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


def test_builder_estimate_matches_prepared_plan_storage() -> None:
    operator = _laplacian(24)
    builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("lu", ordering="reverse-cuthill-mckee")
    )
    cost = builder.cost_for(operator)
    symbolic = la.prepare_sparse_factorization(operator, builder.policy())
    numeric = la.refresh_sparse_factorization(symbolic, operator)
    # The plan now reserves refresh-owned numeric substitution caches as well
    # as factor values. Compare its upper to the real prepared logical payload,
    # not the obsolete symbolic-only plus factor-values formula.
    assert cost.storage_bytes >= _stored_bytes(numeric)


def test_symbolic_plan_bytes_scale_with_factor_nonzeros_not_updates() -> None:
    # With a banded RCM factor of bandwidth b ~ count, nnz(L+U) ~ count^3
    # while the elimination performs ~ count^4 updates. Storage proportional to
    # updates would grow per factor entry with count; nonzero-proportional
    # storage (including prepared substitution schedules) does not.
    per_entry = []
    for count in (12, 24):
        symbolic = la.prepare_sparse_factorization(
            _laplacian(count),
            la.SparseFactorizationPolicy("lu", ordering="reverse-cuthill-mckee"),
        )
        per_entry.append(_stored_bytes(symbolic) / symbolic.factor_indices.size)
    assert per_entry[1] <= per_entry[0]


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


@pytest.mark.parametrize("singular", (False, True))
def test_native_cholesky_psd_correction_retains_real_factor_status(
    singular: bool,
) -> None:
    matrix = jnp.asarray(((1.0, 1.0), (1.0, 1.0 if singular else 2.0)), dtype=np.float64)
    operator = _sparse_map(
        matrix,
        properties=la.OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    policy = la.SparseFactorizationPolicy("cholesky")
    plan = la.prepare_sparse_factorization(operator, policy)
    builder = la.SparseFactorizationPreconditionerBuilder(
        policy,
        prepared_plan=plan,
        setup_operator=operator,
    )
    prepared = builder.prepare_planned(
        builder.plan_setup(operator),
        operator,
        materialization=la.MaterializationPolicy(),
    )
    factor = la.sparse_preconditioner_factorization(prepared)
    assert factor is not None
    if singular:
        assert int(factor.status) == int(la.SparseFactorizationStatus.NONPOSITIVE_PIVOT)
        with pytest.raises(eqx.EquinoxRuntimeError, match="refused its failed factor"):
            prepared.apply(jnp.asarray((1.0, -1.0), dtype=np.float64))
    else:
        assert int(factor.status) == int(la.SparseFactorizationStatus.SUCCESS)
        actual = prepared.apply(jnp.asarray((1.0, -1.0), dtype=np.float64))
        np.testing.assert_allclose(
            actual, np.linalg.solve(np.asarray(matrix), (1.0, -1.0)), atol=1.0e-14
        )
    assert int(factor.diagnostics.replaced_pivots) == 0
    # Default block composition must transform the conditional component SPD
    # evidence, without converting a failed native factor into a usable action.
    primal_space = la.ArraySpace((1,), dtype=np.float64)
    block_space = la.BlockSpace((primal_space, operator.source))
    primal = la.DiagonalLinearOperator(
        jnp.asarray((2.0,)),
        space=primal_space,
        properties=la.OperatorProperties(
            diagonal=True,
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "diagonal": "construction",
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    metric = la.BlockLinearOperator(
        ((primal, None), (None, operator)),
        source=block_space,
        target=block_space,
        properties=la.OperatorProperties(
            block_diagonal=True,
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "block_diagonal": "construction",
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    block_builder = la.BlockFactorizationPreconditionerBuilder(
        la.JacobiPreconditionerBuilder(),
        builder,
        "diagonal",
    )
    assert block_builder.properties_for(metric).certifies("positive_definite")
    action = block_builder.prepare(metric, materialization=la.MaterializationPolicy())
    assert action.properties.certifies("positive_definite")
    assert action.properties.evidence_for("positive_definite") == "transformed"
    rhs = (jnp.asarray((4.0,)), jnp.asarray((1.0, -1.0)))
    if singular:
        with pytest.raises(eqx.EquinoxRuntimeError, match="refused its failed factor"):
            action.apply(rhs)
    else:
        actual_first, actual_second = action.apply(rhs)
        np.testing.assert_array_equal(actual_first, (2.0,))
        np.testing.assert_allclose(
            actual_second, np.linalg.solve(np.asarray(matrix), (1.0, -1.0)), atol=1.0e-14
        )
        if not isinstance(operator, phx.sparse.SparseLinearMap):
            raise TypeError(
                "The explicit sparse fixture must preserve its native map type."
            )
        changed_operator = eqx.tree_at(
            lambda item: item.coefficients,
            operator,
            2 * operator.coefficients,
        )
        changed_primal = eqx.tree_at(
            lambda item: item.diagonal,
            primal,
            2 * primal.diagonal,
        )
        changed_metric = eqx.tree_at(
            lambda item: item.blocks,
            metric,
            ((changed_primal, None), (None, changed_operator)),
        )
        from phydrax._linear_refresh import prepare_refresh_state

        _, refresh_state = prepare_refresh_state(
            la.LinearSystem(metric),
            la.LinearSolvePolicy(
                la.MINRES(),
                preconditioning=la.PreconditioningPolicy(block_builder),
            ),
            setup_operator=metric,
        )

        @jax.jit
        def switched(update: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            current = jax.lax.cond(
                update,
                lambda _: refresh_state.refresh(
                    la.LinearSystem(changed_metric),
                    setup_operator=changed_metric,
                )[1],
                lambda _: refresh_state,
                None,
            )
            correction = current.preconditioner
            assert correction is not None
            assert isinstance(correction, la.BlockFactorizationPreconditioner)
            current_factor = la.sparse_preconditioner_factorization(
                correction.schur_action
            )
            assert current_factor is not None
            first, second = correction.apply(rhs)
            return first, second, current_factor.status

        # Both branch structures retain the same actual symbolic support.
        # The update branch must still carry freshly evaluated factor/cache
        # coefficients, rather than borrowing its predecessor's numerical data.
        first_old, second_old, status_old = switched(jnp.asarray(False))
        first_new, second_new, status_new = switched(jnp.asarray(True))
        assert int(status_old) == int(la.SparseFactorizationStatus.SUCCESS)
        assert int(status_new) == int(la.SparseFactorizationStatus.SUCCESS)
        np.testing.assert_array_equal(first_old, actual_first)
        np.testing.assert_array_equal(second_old, actual_second)
        np.testing.assert_allclose(first_new, actual_first / 2, atol=1.0e-14)
        np.testing.assert_allclose(second_new, actual_second / 2, atol=1.0e-14)


def test_prepared_numeric_substitution_preserves_promoted_pivots_and_refresh_status() -> (
    None
):
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )
    operator = _sparse_map(
        jnp.asarray(((9.0,),), dtype=np.float32), properties=properties
    )
    # The true factor pivot is exactly 3. In float32 this tolerance rounds to
    # 3; in float64 it remains strictly below 3. RHS promotion must renew the
    # actual triangular threshold, not reuse cached lower-precision failure.
    factor = la.factorize_sparse(
        operator,
        la.SparseFactorizationPolicy("cholesky", pivot_tolerance=2.99999999),
    )
    assert int(factor.status) == int(la.SparseFactorizationStatus.SUCCESS)
    original = factor.solve(jnp.ones(1, dtype=np.float32))
    assert int(original.status) == int(la.SparseFactorizationStatus.ZERO_PIVOT)
    promoted = factor.solve(jnp.ones(1, dtype=np.float64))
    assert promoted.value.dtype == np.dtype(np.float64)
    assert int(promoted.status) == int(la.SparseFactorizationStatus.SUCCESS)
    np.testing.assert_allclose(promoted.value, (1.0 / 9.0,), atol=1.0e-14)
    nonfinite = factor.solve(jnp.asarray((np.nan,), dtype=np.float64))
    assert int(nonfinite.status) == int(la.SparseFactorizationStatus.NONFINITE)
    assert int(nonfinite.factorization_status) == int(
        la.SparseFactorizationStatus.SUCCESS
    )

    renewed = la.refresh_sparse_factorization_values(
        factor.plan,
        jnp.asarray((16.0,), dtype=np.float32),
    )
    current = renewed.solve(jnp.ones(1, dtype=np.float32))
    assert int(current.status) == int(la.SparseFactorizationStatus.SUCCESS)
    np.testing.assert_array_equal(current.value, (1.0 / 16.0,))
    # Renewing one immutable numeric owner must not mutate its predecessor.
    assert int(factor.solve(jnp.ones(1, dtype=np.float32)).status) == int(
        la.SparseFactorizationStatus.ZERO_PIVOT,
    )


def test_prepared_schedule_cache_keeps_original_slots_and_current_coefficients() -> None:
    matrix = jnp.asarray(
        (
            (5.0, 1.0, 0.0, 0.0),
            (1.0, 6.0, 2.0, 0.0),
            (0.0, 2.0, 7.0, 1.0),
            (0.0, 0.0, 1.0, 8.0),
        ),
        dtype=np.float64,
    )
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={"self_adjoint": "construction", "positive_definite": "construction"},
    )
    operator = _sparse_map(matrix, properties=properties)
    if not isinstance(operator, phx.sparse.SparseLinearMap):
        raise TypeError("The explicit sparse fixture must preserve its native map type.")
    factor = la.factorize_sparse(operator, la.SparseFactorizationPolicy("cholesky"))
    analysis = factor.plan.lower_analysis
    values = factor.factor_values[factor.plan.lower_positions]
    for transpose, cache in (
        (False, factor.lower_substitution),
        (True, factor.upper_substitution),
    ):
        if transpose:
            positions, valid = (
                analysis.transpose_schedule_positions,
                analysis.transpose_schedule_valid,
            )
            rows, diagonal = (
                analysis.transpose_row_indices,
                analysis.transpose_diagonal_positions,
            )
            oriented = jnp.conj(values[analysis.transpose_value_positions])
            schedule, width = (
                analysis.transpose_level_schedule,
                analysis.transpose_row_width,
            )
        else:
            positions, valid = analysis.schedule_positions, analysis.schedule_valid
            rows, diagonal = analysis.row_indices, analysis.diagonal_positions
            oriented = values
            schedule, width = analysis.level_schedule, analysis.row_width
        off = jnp.where(
            jnp.arange(oriented.size) != diagonal[rows],
            oriented,
            0.0,
        )
        expected = jnp.where(valid, off[positions], 0.0)
        assert cache.scheduled_values.shape == (
            1,
            schedule.shape[0],
            width,
            schedule.shape[1],
        )
        np.testing.assert_array_equal(cache.scheduled_values[0], expected)
        assert np.all(np.asarray(cache.scheduled_values[0])[~np.asarray(valid)] == 0.0)
    rhs = jnp.asarray((1.0, -2.0, 3.0, -4.0), dtype=np.float64)
    np.testing.assert_allclose(
        factor.solve(rhs).value, np.linalg.solve(np.asarray(matrix), rhs), atol=1.0e-14
    )
    renewed = la.refresh_sparse_factorization_values(
        factor.plan, 2 * operator.coefficients
    )
    np.testing.assert_allclose(
        renewed.solve(rhs).value, factor.solve(rhs).value / 2, atol=1.0e-14
    )
    assert int(factor.solve(jnp.asarray((np.inf, 0.0, 0.0, 0.0))).status) == int(
        la.SparseFactorizationStatus.NONFINITE,
    )


def test_preprepared_sparse_builder_refresh_preserves_exact_pattern_and_dynamic_values() -> (
    None
):
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )
    operator = _sparse_map(
        jnp.asarray(((2.0, 1.0), (1.0, 3.0)), dtype=np.float64), properties=properties
    )
    if not isinstance(operator, phx.sparse.SparseLinearMap):
        raise TypeError("The explicit sparse fixture must preserve its native map type.")
    plan = la.prepare_sparse_factorization(
        operator, la.SparseFactorizationPolicy("cholesky")
    )
    builder = la.SparseFactorizationPreconditionerBuilder(
        prepared_plan=plan,
        setup_operator=operator,
    )
    action = builder.prepare(operator, materialization=la.MaterializationPolicy())
    changed = eqx.tree_at(
        lambda item: item.coefficients,
        operator,
        jnp.asarray((4.0, 1.0, 1.0, 5.0), dtype=np.float64),
    )
    refreshed = builder.refresh(
        action, changed, materialization=la.MaterializationPolicy()
    )
    np.testing.assert_allclose(
        refreshed.apply(jnp.asarray((1.0, 2.0), dtype=np.float64)),
        np.linalg.solve(np.asarray(((4.0, 1.0), (1.0, 5.0))), (1.0, 2.0)),
        atol=1.0e-14,
    )
    different = _sparse_map(jnp.eye(2, dtype=np.float64), properties=properties)
    with pytest.raises(ValueError, match="different exact CSR pattern"):
        la.SparseFactorizationPreconditionerBuilder(
            prepared_plan=plan, setup_operator=different
        )


@pytest.mark.parametrize("form", ("lower", "upper", "ldu"))
def test_missing_offdiagonals_are_not_a_general_block_factorization(
    form: la.BlockFactorizationForm,
) -> None:
    space = la.BlockSpace(
        (la.ArraySpace((2,), dtype=np.float64), la.ArraySpace((1,), dtype=np.float64))
    )
    block = la.BlockLinearOperator(
        (
            (la.DiagonalLinearOperator(jnp.ones(2), space=space.spaces[0]), None),
            (None, la.DiagonalLinearOperator(jnp.ones(1), space=space.spaces[1])),
        ),
        source=space,
        target=space,
    )
    builder = la.BlockFactorizationPreconditionerBuilder(
        la.JacobiPreconditionerBuilder(),
        la.JacobiPreconditionerBuilder(),
        form,
    )
    with pytest.raises(ValueError, match="A00, A01, and A10"):
        builder.prepare(block, materialization=la.MaterializationPolicy())


def test_sparse_resource_refusal_preserves_typed_original_allowance() -> None:
    operator = _sparse_map(jnp.asarray(((2.0, 1.0), (1.0, 2.0)), dtype=np.float64))
    with pytest.raises(la.LinearResourceLimitError) as refused:
        la.prepare_sparse_factorization(
            operator,
            la.SparseFactorizationPolicy("lu", max_factor_nnz=1),
        )
    assert refused.value.resource == "sparse_factorization:factor_nnz"
    assert refused.value.limit == 1
    assert refused.value.requested == 2
    assert refused.value.completed == 1
    assert refused.value.symbolic_work is not None


@pytest.mark.parametrize(
    ("form", "paired", "positive"),
    (
        ("diagonal", False, True),
        ("lower", True, False),
        ("upper", True, False),
        ("ldu", False, False),
        ("ldu", True, True),
    ),
)
def test_block_component_spd_requires_fixed_pairing_and_both_component_evidence(
    form: la.BlockFactorizationForm,
    paired: bool,
    positive: bool,
) -> None:
    from phydrax.linalg._block_preconditioning import _component_properties

    certified = la.PreconditionerProperties(
        linear=True,
        stationary=True,
        self_adjoint=True,
        positive_definite=True,
        evidence={"positive_definite": "construction"},
    )
    actual = _component_properties(certified, certified, form, paired, None)
    assert actual.certifies("positive_definite") == positive
    unknown_positive = la.PreconditionerProperties(
        linear=True,
        stationary=True,
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "linear": "construction",
            "stationary": "construction",
            "self_adjoint": "construction",
        },
    )
    assert not _component_properties(
        certified,
        unknown_positive,
        form,
        paired,
        None,
    ).certifies("positive_definite")


def _congruence_reference(
    factor: la.PreparedSparseFactorization, rhs: np.ndarray
) -> np.ndarray:
    """Independent dense triangular reference from actual signed native LU."""
    plan = factor.plan
    n = plan.shape[0]
    packed = np.zeros((n, n), dtype=np.asarray(factor.factor_values).dtype)
    packed[np.asarray(plan.factor_rows), np.asarray(plan.factor_indices)] = np.asarray(
        factor.factor_values
    )
    lower = np.tril(packed, -1) + np.eye(n)
    permutation = np.asarray(plan.permutation)
    forward = np.linalg.solve(lower, rhs[permutation])
    scaled = forward / np.abs(np.diag(packed)).reshape((n,) + (1,) * (rhs.ndim - 1))
    permuted = np.linalg.solve(lower.conj().T, scaled)
    result = np.empty_like(permuted)
    result[permutation] = permuted
    return result


@pytest.mark.parametrize("complex_values", (False, True))
def test_native_indefinite_lu_congruence_matches_independent_metric(
    complex_values: bool,
) -> None:
    matrix = np.asarray(((2.0, 1.0, 0.2), (1.0, -3.0, 0.4), (0.2, 0.4, 1.0)))
    if complex_values:
        matrix = matrix.astype(np.complex128)
        matrix[0, 1] += 0.7j
        matrix[1, 0] -= 0.7j
    assert np.linalg.eigvalsh(matrix)[0] < 0 < np.linalg.eigvalsh(matrix)[-1]
    operator = _sparse_map(jnp.asarray(matrix))
    builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("lu", ordering="reverse-cuthill-mckee"),
        form="lu-congruence",
    )
    planned = builder.plan_setup(operator)
    action = builder.prepare_planned(
        planned, operator, materialization=la.MaterializationPolicy()
    )
    assert isinstance(action, la.SparseFactorCongruencePreconditioner)
    factor = la.sparse_preconditioner_factorization(action)
    assert factor is action.artifact.factorization
    assert int(action.artifact.status) == int(la.SparseFactorizationStatus.SUCCESS)
    assert int(action.artifact.rank) == 3
    assert action.properties.certifies("positive_definite")
    assert not operator.properties.certifies("positive_definite")
    rhs = np.asarray((1.0, -2.0, 0.3), dtype=matrix.dtype)
    if complex_values:
        rhs += np.asarray((0.2j, -0.3j, 0.1j))
    np.testing.assert_allclose(
        action.apply(jnp.asarray(rhs)), _congruence_reference(factor, rhs), atol=2.0e-14
    )
    actual_metric = np.column_stack(
        [
            np.asarray(action.apply(jnp.asarray(column)))
            for column in np.eye(3, dtype=matrix.dtype).T
        ]
    )
    np.testing.assert_allclose(actual_metric, actual_metric.conj().T, atol=2.0e-14)
    assert np.linalg.eigvalsh(actual_metric)[0] > 0
    assert planned.cost.storage_bytes >= _stored_bytes(action.artifact)
    assert "extra_preparation_work_units=" in planned.cost.reason
    assert "apply_work_units=" in planned.cost.reason
    inverse = la.SparseFactorizationPreconditionerBuilder(builder.policy())
    assert inverse.builder_id != builder.builder_id
    inverse_action = inverse.prepare(operator, materialization=la.MaterializationPolicy())
    np.testing.assert_allclose(
        inverse_action.apply(jnp.asarray(rhs)), np.linalg.solve(matrix, rhs), atol=2.0e-14
    )


def test_lu_congruence_obeys_declared_source_pairing() -> None:
    weights = jnp.asarray((2.0, 5.0, 3.0), dtype=np.float64)
    space = la.ArraySpace((3,), dtype=np.float64, pairing=la.DiagonalPairing(weights))
    matrix = jnp.asarray(
        ((2.0, 1.0, 0.2), (1.0, -3.0, 0.4), (0.2, 0.4, 1.0)), dtype=np.float64
    )
    rows, columns = jnp.nonzero(matrix)
    relation = phx.sparse.EdgeRelation(columns, rows, source_size=3, target_size=3)
    operator = phx.sparse.SparseCoordinateOperator(
        relation,
        matrix[rows, columns],
        source=space,
        target=space,
    )
    action = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("lu"),
        form="lu-congruence",
    ).prepare(operator, materialization=la.MaterializationPolicy())
    x, y = jnp.asarray((1.0, -0.2, 3.0)), jnp.asarray((-0.4, 2.0, 0.1))
    np.testing.assert_allclose(
        action.apply(x),
        _congruence_reference(action.factorization, np.asarray(weights * x)),
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        space.inner(x, action.apply(y)), space.inner(action.apply(x), y), atol=2.0e-14
    )
    assert float(space.inner(x, action.apply(x))) > 0


def test_jitted_lu_congruence_refresh_changes_real_numeric_cache() -> None:
    operator = _sparse_map(jnp.asarray(((2.0, 1.0), (1.0, -3.0)), dtype=np.float64))
    plan = la.prepare_sparse_factorization(operator, la.SparseFactorizationPolicy("lu"))
    builder = la.SparseFactorizationPreconditionerBuilder(
        prepared_plan=plan,
        setup_operator=operator,
        form="lu-congruence",
    )
    action = _congruence_action(
        builder.prepare(operator, materialization=la.MaterializationPolicy())
    )
    rhs = jnp.asarray((1.0, 2.0), dtype=np.float64)

    @eqx.filter_jit
    def refresh(coefficients: jax.Array) -> la.SparseFactorCongruencePreconditioner:
        changed = eqx.tree_at(lambda value: value.coefficients, operator, coefficients)
        return _congruence_action(
            builder.refresh(action, changed, materialization=la.MaterializationPolicy())
        )

    doubled = refresh(2 * operator.coefficients)
    np.testing.assert_allclose(doubled.apply(rhs), 0.5 * action.apply(rhs), atol=2.0e-14)
    changed = refresh(jnp.asarray((4.0, 1.5, 1.5, -2.0), dtype=np.float64))
    np.testing.assert_allclose(
        changed.apply(rhs),
        _congruence_reference(changed.factorization, np.asarray(rhs)),
        atol=2.0e-14,
    )
    assert int(changed.artifact.status) == int(la.SparseFactorizationStatus.SUCCESS)
    assert not np.array_equal(
        np.asarray(changed.artifact.inverse_absolute_diagonal),
        np.asarray(action.artifact.inverse_absolute_diagonal),
    )
    assert (
        changed.artifact.lower_adjoint_substitution
        is not action.artifact.lower_adjoint_substitution
    )


@pytest.mark.parametrize("bad_value", (0.0, np.nan, np.inf))
def test_lu_congruence_preserves_native_refusal_and_nonfinite_precedence(
    bad_value: float,
) -> None:
    operator = _sparse_map(jnp.asarray(((2.0, 1.0), (1.0, -3.0)), dtype=np.float64))
    builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("lu"), form="lu-congruence"
    )
    action = _congruence_action(
        builder.prepare(operator, materialization=la.MaterializationPolicy())
    )
    bad = eqx.tree_at(
        lambda value: value.coefficients,
        operator,
        jnp.asarray((bad_value, 1.0, 1.0, -3.0), dtype=np.float64),
    )
    failed = _congruence_action(
        builder.refresh(action, bad, materialization=la.MaterializationPolicy())
    )
    expected = (
        la.SparseFactorizationStatus.ZERO_PIVOT
        if bad_value == 0
        else la.SparseFactorizationStatus.NONFINITE
    )
    assert int(failed.factorization.status) == int(expected)
    assert int(failed.artifact.status) == int(expected)
    assert int(failed.artifact.rank) == -1
    solved = failed.artifact.solve(jnp.asarray((np.nan, 1.0)))
    assert int(solved.factorization_status) == int(expected)
    assert int(solved.status) == int(la.SparseFactorizationStatus.NONFINITE)
    with pytest.raises(Exception, match="refused"):
        failed.apply(jnp.ones(2))
    nonfinite = action.artifact.solve(jnp.asarray((np.inf, 0.0)))
    assert int(nonfinite.status) == int(la.SparseFactorizationStatus.NONFINITE)
    assert int(nonfinite.factorization_status) == int(
        la.SparseFactorizationStatus.SUCCESS
    )
    refused_finite = failed.artifact.solve(jnp.ones(2))
    assert int(refused_finite.factorization_status) == int(expected)
    assert int(refused_finite.status) == int(expected)


def test_lu_congruence_original_byte_allowance_charges_added_cache_before_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.linalg import _incomplete_factorizations as owner

    operator = _sparse_map(jnp.asarray(((2.0, 1.0), (1.0, -3.0)), dtype=np.float64))
    base_plan = la.prepare_sparse_factorization(
        operator, la.SparseFactorizationPolicy("lu")
    )
    allowance = base_plan.factor_bytes
    policy = la.SparseFactorizationPolicy("lu", max_factor_bytes=allowance)
    plan = la.prepare_sparse_factorization(operator, policy)
    builder = la.SparseFactorizationPreconditionerBuilder(
        prepared_plan=plan,
        setup_operator=operator,
        form="lu-congruence",
    )
    required = plan.factor_bytes + plan.lu_congruence_storage_bytes_upper(8)
    assert required > allowance
    assert not builder.cost_for(operator).accepted

    def forbidden_refresh(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("resource refusal must precede numeric allocation")

    monkeypatch.setattr(owner, "refresh_sparse_factorization", forbidden_refresh)
    with pytest.raises(la.LinearResourceLimitError) as refused:
        builder.prepare(operator, materialization=la.MaterializationPolicy())
    assert refused.value.resource == "sparse_factorization:factor_bytes"
    assert refused.value.limit == allowance
    assert refused.value.requested == required
    assert refused.value.completed == plan.factor_nnz
    assert refused.value.storage_bytes_upper == required


@pytest.mark.parametrize(
    "policy",
    (
        la.SparseFactorizationPolicy("cholesky"),
        la.SparseFactorizationPolicy("lu", diagonal_shift=0.1),
        la.SparseFactorizationPolicy("lu", allow_pivot_replacement=True),
    ),
)
def test_lu_congruence_rejects_modified_or_non_lu_policies(
    policy: la.SparseFactorizationPolicy,
) -> None:
    with pytest.raises(ValueError, match="unshifted native LU"):
        la.SparseFactorizationPreconditionerBuilder(policy, form="lu-congruence")


def test_lu_congruence_prepares_once_and_supports_matrix_rhs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.linalg import _incomplete_factorizations as owner

    operator = _sparse_map(jnp.asarray(((2.0, 1.0), (1.0, -3.0)), dtype=np.float64))
    builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("lu"),
        form="lu-congruence",
    )
    original_prepare = owner.prepare_sparse_factor_congruence
    calls: list[la.PreparedSparseFactorization] = []

    def counted_prepare(
        factor: la.PreparedSparseFactorization,
    ) -> la.PreparedSparseFactorCongruence:
        calls.append(factor)
        return original_prepare(factor)

    monkeypatch.setattr(owner, "prepare_sparse_factor_congruence", counted_prepare)
    action = _congruence_action(
        builder.prepare(operator, materialization=la.MaterializationPolicy())
    )
    assert len(calls) == 1
    rhs = jnp.asarray(((1.0, 2.0), (3.0, -1.0)), dtype=np.float64)
    matrix_solve = action.artifact.solve(rhs)
    np.testing.assert_allclose(
        matrix_solve.value,
        _congruence_reference(action.factorization, np.asarray(rhs)),
        atol=2.0e-14,
    )
    assert int(matrix_solve.status) == int(la.SparseFactorizationStatus.SUCCESS)
    action.apply(rhs[:, 0])
    action.apply(rhs[:, 1])
    assert len(calls) == 1
    changed = eqx.tree_at(
        lambda value: value.coefficients, operator, 2 * operator.coefficients
    )
    refreshed = builder.refresh(
        action, changed, materialization=la.MaterializationPolicy()
    )
    assert len(calls) == 2
    np.testing.assert_allclose(
        refreshed.apply(rhs[:, 0]), 0.5 * action.apply(rhs[:, 0]), atol=2.0e-14
    )
    assert len(calls) == 2
