#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike, DTypeLike

import phydrax as phx
from phydrax.combinatorial import (
    CapacitatedAuctionPlan,
    CapacitatedAuctionResult,
    CombinatorialStatus,
    PreparedCapacitatedAuction,
)


_SITES = 12
_LABELS = 4
_ACCEPTED = (int(CombinatorialStatus.OPTIMAL), int(CombinatorialStatus.FEASIBLE))

_Solve: Callable[
    [PreparedCapacitatedAuction, ArrayLike, ArrayLike, ArrayLike],
    CapacitatedAuctionResult,
] = jax.jit(lambda prepared, values, lower, upper: prepared.solve(values, lower, upper))


def _topology(rng: np.random.Generator, width: int) -> tuple[np.ndarray, np.ndarray]:
    if width == _LABELS:
        labels = np.tile(np.arange(_LABELS, dtype=np.int32), (_SITES, 1))
        valid = np.ones((_SITES, width), dtype=np.bool_)
    else:
        labels = np.stack(
            [rng.permutation(_LABELS)[:width] for _ in range(_SITES)]
        ).astype(np.int32)
        valid = rng.random((_SITES, width)) < 0.75
        valid[:, 0] = True
    return labels, valid


def _feasible_counts(
    rng: np.random.Generator, labels: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    counts = np.zeros((_LABELS,), dtype=np.int32)
    for site in range(_SITES):
        slot = rng.choice(np.flatnonzero(valid[site]))
        counts[labels[site, slot]] += 1
    return counts


def _flow_optimum(
    labels: np.ndarray,
    valid: np.ndarray,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    sites, width = labels.shape
    sink = sites + _LABELS
    route_sources = np.repeat(np.arange(sites), width)
    route_targets = sites + labels.reshape(-1)
    sources = np.concatenate((route_sources, sites + np.arange(_LABELS)))
    targets = np.concatenate((route_targets, np.full((_LABELS,), sink)))
    edge_valid = np.concatenate((valid.reshape(-1), np.ones((_LABELS,), np.bool_)))
    capacities = np.concatenate(
        (np.ones((sites * width,), np.int32), (upper - lower).astype(np.int32))
    )
    balances = np.concatenate(
        (np.ones((sites,), np.int32), -lower, [-(sites - int(lower.sum()))])
    ).astype(np.int32)
    relation = phx.sparse.EdgeRelation(
        sources,
        targets,
        source_size=sink + 1,
        target_size=sink + 1,
        valid=edge_valid,
    )
    space = phx.combinatorial.CapacitatedFlowSpace(relation, balances, capacities)
    costs = np.concatenate(
        (-np.where(valid, values, 0.0).reshape(-1), np.zeros((_LABELS,)))
    )
    result = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(space, jnp.asarray(costs)),
        phx.combinatorial.CycleCancelingMinCostFlow(),
    )
    assert bool(result.success)
    return -float(result.objective_value)


def _assert_consistent_assignment(
    result: CapacitatedAuctionResult,
    labels: np.ndarray,
    valid: np.ndarray,
    values: np.ndarray,
) -> None:
    slots = np.asarray(result.slots)
    sites = np.arange(_SITES)
    assert np.all(valid[sites, slots])
    np.testing.assert_array_equal(result.labels, labels[sites, slots])
    np.testing.assert_array_equal(
        result.counts, np.bincount(labels[sites, slots], minlength=_LABELS)
    )
    np.testing.assert_allclose(
        result.evidence.primal_value, values[sites, slots].sum(), rtol=1e-12
    )


def _assert_refused(
    result: CapacitatedAuctionResult, status: CombinatorialStatus
) -> None:
    assert int(result.status) == int(status)
    assert not bool(result.success)
    np.testing.assert_array_equal(result.labels, -1)
    np.testing.assert_array_equal(result.slots, -1)
    np.testing.assert_array_equal(result.counts, 0)
    assert np.all(np.isnan(np.asarray(result.prices)))


@pytest.mark.parametrize("width", [_LABELS, 3])
def test_exact_counts_match_min_cost_flow_optimum(width: int) -> None:
    prepared_plan = CapacitatedAuctionPlan(_SITES, _LABELS, width)
    for seed in range(4):
        rng = np.random.default_rng(100 * width + seed)
        labels, valid = _topology(rng, width)
        values = rng.normal(size=(_SITES, width))
        targets = _feasible_counts(rng, labels, valid)
        prepared = prepared_plan.prepare(labels, valid)
        result = _Solve(prepared, jnp.asarray(values), targets, targets)

        assert int(result.status) in _ACCEPTED
        assert bool(result.success)
        assert not bool(result.evidence.price_divergence)
        np.testing.assert_array_equal(result.counts, targets)
        _assert_consistent_assignment(result, labels, valid, values)
        optimum = _flow_optimum(labels, valid, values, targets, targets)
        primal = float(result.evidence.primal_value)
        dual = float(result.evidence.dual_value)
        gap_bound = float(result.evidence.gap_bound)
        assert primal <= optimum + 1e-9
        assert optimum <= dual + 1e-9
        assert optimum - primal <= gap_bound + 1e-9
        assert float(result.evidence.duality_gap) <= gap_bound + 1e-9


def test_count_bounds_match_min_cost_flow_with_sign_complementarity() -> None:
    plan = CapacitatedAuctionPlan(_SITES, _LABELS, 3)
    for seed in range(4):
        rng = np.random.default_rng(700 + seed)
        labels, valid = _topology(rng, 3)
        values = rng.normal(size=(_SITES, 3))
        counts = _feasible_counts(rng, labels, valid)
        lower = np.maximum(counts - rng.integers(0, 3, _LABELS), 0).astype(np.int32)
        upper = (counts + rng.integers(0, 3, _LABELS)).astype(np.int32)
        result = _Solve(plan.prepare(labels, valid), jnp.asarray(values), lower, upper)

        assert bool(result.success)
        assert float(result.evidence.sign_cs_violation) == 0.0
        realized = np.asarray(result.counts)
        assert np.all((lower <= realized) & (realized <= upper))
        _assert_consistent_assignment(result, labels, valid, values)
        optimum = _flow_optimum(labels, valid, values, lower, upper)
        primal = float(result.evidence.primal_value)
        assert primal <= optimum + 1e-9
        assert optimum <= float(result.evidence.dual_value) + 1e-9
        assert optimum - primal <= float(result.evidence.gap_bound) + 1e-9


def test_integer_values_with_fine_epsilon_are_exactly_optimal() -> None:
    schedule = tuple(10.0 ** (-k) for k in range(0, 10))
    assert schedule[-1] < 1.0 / _SITES
    plan = CapacitatedAuctionPlan(
        _SITES, _LABELS, _LABELS, epsilon_schedule=schedule, epsilon_scale="absolute"
    )
    rng = np.random.default_rng(5)
    labels, valid = _topology(rng, _LABELS)
    values = rng.integers(-5, 6, size=(_SITES, _LABELS)).astype(np.float64)
    counts = _feasible_counts(rng, labels, valid)
    lower = np.maximum(counts - 1, 0).astype(np.int32)
    upper = (counts + 1).astype(np.int32)
    result = _Solve(plan.prepare(labels, valid), jnp.asarray(values), lower, upper)

    assert int(result.status) == int(CombinatorialStatus.OPTIMAL)
    assert float(result.evidence.primal_value) == _flow_optimum(
        labels, valid, values, lower, upper
    )


@pytest.mark.parametrize(
    ("valid", "lower", "upper"),
    [
        ([[True, True]] * 3, [2, 2], [3, 3]),  # sum lower > N
        ([[True, True]] * 3, [0, 0], [1, 1]),  # sum upper < N
        ([[True, True], [True, True], [False, False]], [0, 0], [3, 3]),
        ([[True, False], [True, False], [True, True]], [0, 0], [1, 2]),  # Hall
    ],
    ids=["lower-sum", "upper-sum", "no-candidate", "hall-violation"],
)
def test_infeasible_problems_are_refused_without_partial_assignment(
    valid: list[list[bool]], lower: list[int], upper: list[int]
) -> None:
    plan = CapacitatedAuctionPlan(3, 2, 2)
    prepared = plan.prepare([[0, 1], [0, 1], [0, 1]], valid)
    values = jnp.asarray([[1.0, 0.0], [0.5, 0.2], [0.3, 0.9]])
    result = prepared.solve(values, jnp.asarray(lower), jnp.asarray(upper))
    _assert_refused(result, CombinatorialStatus.INFEASIBLE)


def test_hall_violation_is_detected_by_price_divergence() -> None:
    plan = CapacitatedAuctionPlan(3, 2, 2)
    prepared = plan.prepare(
        [[0, 1], [0, 1], [0, 1]], [[True, False], [True, False], [True, True]]
    )
    result = prepared.solve(
        jnp.asarray([[1.0, 0.0], [0.5, 0.0], [0.3, 0.9]]),
        jnp.asarray([0, 0]),
        jnp.asarray([1, 2]),
    )
    assert bool(result.evidence.bounds_consistent)
    assert bool(result.evidence.price_divergence)
    assert int(result.status) == int(CombinatorialStatus.INFEASIBLE)


def test_nonfinite_valid_value_is_refused_and_invalid_slots_are_ignored() -> None:
    plan = CapacitatedAuctionPlan(2, 2, 2)
    prepared = plan.prepare([[0, 1], [0, 1]], [[True, True], [True, False]])
    lower = jnp.asarray([1, 0])
    upper = jnp.asarray([1, 1])
    ignored = prepared.solve(jnp.asarray([[0.0, 1.0], [2.0, jnp.nan]]), lower, upper)
    assert bool(ignored.success)
    np.testing.assert_array_equal(ignored.labels, [1, 0])
    refused = prepared.solve(jnp.asarray([[jnp.inf, 1.0], [2.0, 0.0]]), lower, upper)
    _assert_refused(refused, CombinatorialStatus.NONFINITE_INPUT)
    assert not bool(refused.evidence.finite_values)


def test_round_cap_reports_maximum_steps_without_partial_assignment() -> None:
    plan = CapacitatedAuctionPlan(4, 2, 2, maximum_rounds=1)
    prepared = plan.prepare([[0, 1]] * 4)
    values = jnp.asarray([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
    result = prepared.solve(values, jnp.asarray([2, 2]), jnp.asarray([2, 2]))
    _assert_refused(result, CombinatorialStatus.MAXIMUM_STEPS_REACHED)
    assert bool(result.evidence.round_limit_reached)
    uncapped = CapacitatedAuctionPlan(4, 2, 2).prepare([[0, 1]] * 4)
    solved = uncapped.solve(values, jnp.asarray([2, 2]), jnp.asarray([2, 2]))
    np.testing.assert_array_equal(solved.labels, [0, 0, 1, 1])


def test_jit_warm_start_preserves_counts_and_float32_dtype() -> None:
    plan = CapacitatedAuctionPlan(_SITES, _LABELS, _LABELS)
    rng = np.random.default_rng(11)
    labels, valid = _topology(rng, _LABELS)
    prepared = plan.prepare(labels, valid)
    values = jnp.asarray(rng.normal(size=(_SITES, _LABELS)), dtype=jnp.float32)
    targets = jnp.asarray([3, 3, 3, 3])

    @jax.jit
    def solve(prices: jax.Array) -> CapacitatedAuctionResult:
        return prepared.solve(values, targets, targets, initial_prices=prices)

    cold = solve(jnp.zeros((_LABELS,), dtype=jnp.float32))
    warm = solve(cold.prices)
    assert bool(cold.success) and bool(warm.success)
    assert cold.prices.dtype == jnp.float32
    assert cold.evidence.primal_value.dtype == jnp.float32
    np.testing.assert_array_equal(warm.counts, cold.counts)
    np.testing.assert_array_equal(warm.counts, targets)
    assert int(warm.evidence.rounds) <= int(cold.evidence.rounds)


def test_traced_plan_solve_refuses_inconsistent_device_candidates() -> None:
    plan = CapacitatedAuctionPlan(2, 2, 2)
    values = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    counts = jnp.asarray([1, 1])

    @jax.jit
    def solve(labels: jax.Array, valid: jax.Array) -> CapacitatedAuctionResult:
        return plan.solve(labels, valid, values, counts, counts)

    valid = jnp.ones((2, 2), dtype=jnp.bool_)
    good = solve(jnp.asarray([[0, 1], [0, 1]]), valid)
    assert bool(good.success)
    np.testing.assert_array_equal(good.labels, [0, 1])
    for labels in ([[0, 0], [0, 1]], [[0, 2], [0, 1]]):
        refused = solve(jnp.asarray(labels), valid)
        _assert_refused(refused, CombinatorialStatus.INFEASIBLE)
        assert not bool(refused.evidence.candidates_consistent)
    masked = solve(
        jnp.asarray([[0, 7], [0, 1]]), jnp.asarray([[True, False], [True, True]])
    )
    assert bool(masked.success)
    np.testing.assert_array_equal(masked.labels, [0, 1])


@pytest.mark.strict_jax
def test_traced_candidate_labels_refuse_uint64_wrap_before_narrowing() -> None:
    with jax.enable_x64():
        plan = CapacitatedAuctionPlan(1, 1, 1)
        values = jnp.asarray([[1.0]])
        counts = jnp.asarray([1], dtype=jnp.int64)
        valid = jnp.asarray([[True]])

        @jax.jit
        def solve(labels: jax.Array) -> CapacitatedAuctionResult:
            return plan.solve(labels, valid, values, counts, counts)

        ordinary = solve(jnp.asarray([[0]], dtype=jnp.uint64))
        assert bool(ordinary.success)
        wrapped = solve(jnp.asarray([[2**32]], dtype=jnp.uint64))

    _assert_refused(wrapped, CombinatorialStatus.INFEASIBLE)
    assert not bool(wrapped.evidence.candidates_consistent)
    assert not bool(wrapped.evidence.feasible)


@pytest.mark.strict_jax
@pytest.mark.parametrize("count", [2**31, 2**32], ids=["int32-max-plus-one", "uint32-wrap"])
def test_traced_count_bounds_refuse_values_above_int32(count: int) -> None:
    with jax.enable_x64():
        plan = CapacitatedAuctionPlan(1, 1, 1)
        prepared = plan.prepare([[0]])

        @jax.jit
        def solve(bounds: jax.Array) -> CapacitatedAuctionResult:
            return prepared.solve(jnp.asarray([[1.0]]), bounds, bounds)

        refused = solve(jnp.asarray([count], dtype=jnp.uint64))

    _assert_refused(refused, CombinatorialStatus.INFEASIBLE)
    assert not bool(refused.evidence.bounds_consistent)
    assert not bool(refused.evidence.feasible)


@pytest.mark.strict_jax
def test_traced_count_bound_accepts_int32_max_without_narrowing_change() -> None:
    with jax.enable_x64():
        plan = CapacitatedAuctionPlan(1, 2, 2)
        prepared = plan.prepare([[0, 1]])

        @jax.jit
        def solve(upper: jax.Array) -> CapacitatedAuctionResult:
            return prepared.solve(
                jnp.asarray([[1.0, 0.0]]),
                jnp.asarray([0, 0], dtype=jnp.int64),
                upper,
            )

        maximum = np.iinfo(np.int32).max
        result = solve(jnp.asarray([maximum, maximum], dtype=jnp.int64))

    assert bool(result.success)
    assert bool(result.evidence.bounds_consistent)
    np.testing.assert_array_equal(result.labels, [0])


@pytest.mark.strict_jax
def test_traced_initial_slot_sentinel_and_zero_are_accepted() -> None:
    with jax.enable_x64():
        prepared = CapacitatedAuctionPlan(1, 1, 1).prepare([[0]])

        @jax.jit
        def solve(initial_slots: jax.Array) -> CapacitatedAuctionResult:
            return prepared.solve(
                jnp.asarray([[1.0]]),
                jnp.asarray([1], dtype=jnp.int64),
                jnp.asarray([1], dtype=jnp.int64),
                initial_slots=initial_slots,
            )

        unassigned = solve(jnp.asarray([-1], dtype=jnp.int64))
        assigned = solve(jnp.asarray([0], dtype=jnp.int64))

    assert bool(unassigned.success)
    assert bool(assigned.success)
    assert bool(unassigned.evidence.warm_start_consistent)
    assert bool(assigned.evidence.warm_start_consistent)
    np.testing.assert_array_equal(unassigned.labels, assigned.labels)


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    "slot",
    [np.iinfo(np.int32).max, 2**31, 2**32],
    ids=["int32-max-semantic", "int32-max-plus-one", "uint32-wrap"],
)
def test_traced_initial_slots_refuse_invalid_values_without_wrap(slot: int) -> None:
    with jax.enable_x64():
        prepared = CapacitatedAuctionPlan(1, 1, 1).prepare([[0]])

        @jax.jit
        def solve(initial_slots: jax.Array) -> CapacitatedAuctionResult:
            return prepared.solve(
                jnp.asarray([[1.0]]),
                jnp.asarray([1], dtype=jnp.int64),
                jnp.asarray([1], dtype=jnp.int64),
                initial_slots=initial_slots,
            )

        refused = solve(jnp.asarray([slot], dtype=jnp.uint64))

    _assert_refused(refused, CombinatorialStatus.INFEASIBLE)
    assert not bool(refused.evidence.warm_start_consistent)
    assert not bool(refused.evidence.feasible)


def test_host_boundaries_refuse_integer_values_outside_int32() -> None:
    plan = CapacitatedAuctionPlan(1, 1, 1)
    with pytest.raises(ValueError, match="signed-int32"):
        plan.prepare(np.asarray([[2**32]], dtype=np.uint64))

    relation = phx.sparse.RowRelation(np.asarray([[0]], dtype=np.int32), source_size=1)
    with pytest.raises(ValueError, match="int32"):
        phx.combinatorial.CapacitatedAssignmentSpace(
            relation,
            np.asarray([2**32], dtype=np.uint64),
            np.asarray([2**32], dtype=np.uint64),
        )


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    ("slot", "dtype"),
    [
        (2**32, jnp.uint64),
        (2**31, jnp.int64),
        (-(2**31) - 1, jnp.int64),
        (np.iinfo(np.int32).max, jnp.int32),
        (np.iinfo(np.int32).min, jnp.int32),
    ],
    ids=[
        "uint64-wrap",
        "int64-high-overflow",
        "int64-low-overflow",
        "int32-maximum",
        "int32-minimum",
    ],
)
def test_assignment_canonicalization_refuses_invalid_slots_before_cast(
    slot: int, dtype: DTypeLike
) -> None:
    with jax.enable_x64():
        relation = phx.sparse.RowRelation(
            np.asarray([[0]], dtype=np.int32), source_size=1
        )
        space = phx.combinatorial.CapacitatedAssignmentSpace(relation, [1], [1])

        @jax.jit
        def canonicalize_and_audit(
            slots: jax.Array, /
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            decision = phx.combinatorial.CapacitatedAssignmentDecision(slots)
            canonical = space.canonicalize(decision)
            feasibility = space.audit(decision)
            return canonical.slots, feasibility.feasible, feasibility.residual

        slots, feasible, residual = canonicalize_and_audit(
            jnp.asarray([slot], dtype=dtype)
        )

    np.testing.assert_array_equal(slots, [-1])
    assert not bool(feasible)
    assert float(residual) == 2.0


@pytest.mark.strict_jax
def test_assignment_canonicalization_preserves_valid_slots_and_audit() -> None:
    relation = phx.sparse.RowRelation(
        np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        source_size=2,
        valid=np.asarray([[True, False], [True, True]]),
    )
    space = phx.combinatorial.CapacitatedAssignmentSpace(relation, [1, 1], [1, 1])
    @jax.jit
    def canonicalize_and_audit(
        slots: jax.Array, /
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        decision = phx.combinatorial.CapacitatedAssignmentDecision(slots)
        canonical = space.canonicalize(decision)
        feasibility = space.audit(decision)
        return canonical.slots, feasibility.feasible, feasibility.residual

    slots, feasible, residual = canonicalize_and_audit(
        jnp.asarray([[0, 0], [1, 1]], dtype=jnp.int32)
    )

    np.testing.assert_array_equal(slots, [[0, 0], [-1, 1]])
    np.testing.assert_array_equal(feasible, [True, False])
    np.testing.assert_array_equal(residual, [0.0, 2.0])


@pytest.mark.strict_jax
def test_assignment_audit_saturates_overflowing_residual() -> None:
    relation = phx.sparse.RowRelation(
        np.asarray([[0, 1, 2]], dtype=np.int32), source_size=3
    )
    maximum = np.iinfo(np.int32).max
    counts = np.asarray([maximum, maximum, 3], dtype=np.int32)
    space = phx.combinatorial.CapacitatedAssignmentSpace(relation, counts, counts)

    @jax.jit
    def audit(slots: jax.Array, /) -> tuple[jax.Array, jax.Array]:
        decision = phx.combinatorial.CapacitatedAssignmentDecision(slots)
        feasibility = space.audit(decision)
        return feasibility.feasible, feasibility.residual

    feasible, residual = audit(jnp.asarray([2], dtype=jnp.int32))

    assert not bool(feasible)
    assert float(residual) == float(maximum)


def test_assignment_canonicalization_requires_integer_slots() -> None:
    relation = phx.sparse.RowRelation(np.asarray([[0]], dtype=np.int32), source_size=1)
    space = phx.combinatorial.CapacitatedAssignmentSpace(relation, [1], [1])
    decision = phx.combinatorial.CapacitatedAssignmentDecision(jnp.asarray([0.0]))

    with pytest.raises(TypeError, match="integer dtype"):
        space.canonicalize(decision)


def test_prepare_refuses_duplicate_and_out_of_range_labels() -> None:
    plan = CapacitatedAuctionPlan(2, 2, 2)
    with pytest.raises(ValueError, match="same label"):
        plan.prepare([[0, 0], [0, 1]])
    with pytest.raises(ValueError, match="lie in"):
        plan.prepare([[0, 2], [0, 1]])
    plan.prepare([[0, 0], [0, 1]], [[True, False], [True, True]])


def test_protocol_solve_agrees_with_prepared_kernel_and_refuses_resources() -> None:
    rng = np.random.default_rng(23)
    labels, valid = _topology(rng, 3)
    values = rng.normal(size=(2, _SITES, 3))
    counts = _feasible_counts(rng, labels, valid)
    relation = phx.sparse.RowRelation(
        np.where(valid, labels, 0), source_size=_LABELS, valid=valid
    )
    space = phx.combinatorial.CapacitatedAssignmentSpace(relation, counts, counts)
    method = phx.combinatorial.CapacitatedAuction()
    problem = phx.combinatorial.LinearCombinatorialProblem(space, jnp.asarray(-values))
    result = phx.combinatorial.solve_combinatorial(problem, method)

    prepared = CapacitatedAuctionPlan(_SITES, _LABELS, 3).prepare(labels, valid)
    for batch in range(2):
        direct = _Solve(prepared, jnp.asarray(values[batch]), counts, counts)
        np.testing.assert_array_equal(result.decision.slots[batch], direct.slots)
        np.testing.assert_allclose(
            result.objective_value[batch], -direct.evidence.primal_value, rtol=1e-12
        )
    assert bool(jnp.all(result.valid))
    assert bool(jnp.all(space.audit(result.decision).feasible))
    np.testing.assert_array_equal(
        result.certificate.optimality_proven,
        result.certificate.absolute_gap
        <= phx.combinatorial.CombinatorialCertification().threshold(
            result.objective_value
        ),
    )

    with pytest.raises(ValueError, match="maximum_routes"):
        phx.combinatorial.plan_combinatorial(
            problem, phx.combinatorial.CapacitatedAuction(maximum_routes=10)
        )
    with pytest.raises(ValueError, match="maximum_routes"):
        CapacitatedAuctionPlan(_SITES, _LABELS, 3, maximum_routes=10)
