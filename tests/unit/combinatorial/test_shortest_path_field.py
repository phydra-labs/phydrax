from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from phydrax.combinatorial import (
    ShortestPathFieldMethod,
    ShortestPathFieldPlan,
    ShortestPathFieldStatus,
)
from phydrax.sparse import EdgeRelation


def relation(edges: list[tuple[int, int]], n: int) -> EdgeRelation:
    routes = np.asarray(edges, dtype=np.int32).reshape((-1, 2))
    return EdgeRelation(routes[:, 0], routes[:, 1], source_size=n, target_size=n)


@pytest.mark.parametrize("method", ["host-dijkstra", "bounded-relaxation-reference"])
def test_unique_path_vjp_offsets_and_unreachable(method: str) -> None:
    plan = ShortestPathFieldPlan(relation([(0, 1), (1, 2), (0, 2)], 4), method=method)  # ty: ignore[invalid-argument-type]
    costs = jnp.asarray([1.0, 2.0, 8.0], dtype=jnp.float64)
    offsets = jnp.asarray([0.25], dtype=jnp.float64)
    result = jax.jit(lambda c: plan.solve(c, jnp.asarray([0]), source_offsets=offsets))(
        costs
    )
    np.testing.assert_allclose(result.distance[:3], [0.25, 1.25, 3.25])
    assert np.isinf(result.distance[3])
    assert result.status == ShortestPathFieldStatus.SUCCESS
    assert result.derivative_regular
    assert result.predecessor_edge[3] == -1
    weight = jnp.asarray([1.0, 2.0, 3.0, 100.0])
    cost_grad, offset_grad = jax.grad(
        lambda c, o: jnp.vdot(
            jnp.where(
                jnp.isfinite(plan.solve(c, jnp.asarray([0]), source_offsets=o).distance),
                plan.solve(c, jnp.asarray([0]), source_offsets=o).distance,
                0.0,
            ),
            weight,
        ),
        argnums=(0, 1),
    )(costs, offsets)
    np.testing.assert_allclose(cost_grad, [5.0, 3.0, 0.0])
    np.testing.assert_allclose(offset_grad, [6.0])
    _, pullback = jax.vjp(lambda c: plan.solve(c, jnp.asarray([0])).distance, costs)
    np.testing.assert_allclose(pullback(jnp.asarray([0.0, 0.0, 0.0, 1.0]))[0], 0.0)
    eps = 1e-5
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps
        plus = plan.solve(costs + step, jnp.asarray([0])).distance[:3]
        minus = plan.solve(costs - step, jnp.asarray([0])).distance[:3]
        numerical = np.dot(np.asarray((plus - minus) / (2 * eps)), np.asarray(weight[:3]))
        np.testing.assert_allclose(cost_grad[i], numerical, rtol=1e-8, atol=1e-8)


def test_exact_and_inherited_tie() -> None:
    plan = ShortestPathFieldPlan(relation([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], 5))
    result = plan.solve(jnp.ones(5), jnp.asarray([0]))
    assert result.status == ShortestPathFieldStatus.SUCCESS
    np.testing.assert_allclose(result.margin[3:], 0.0)
    assert not result.derivative_regular
    assert result.predecessor_edge[3] == 2


def test_zero_cost_cycle_forest() -> None:
    rel = relation([(0, 1), (1, 2), (2, 1), (0, 2), (2, 3)], 4)
    plan = ShortestPathFieldPlan(rel)
    result = plan.solve(jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0]), jnp.asarray([0]))
    np.testing.assert_allclose(result.distance, [0.0, 0.0, 0.0, 1.0])
    for vertex, edge in enumerate(np.asarray(result.predecessor_edge)):
        if edge >= 0:
            assert result.hops[rel.source_indices[edge]] + 1 == result.hops[vertex]
    assert not result.derivative_regular


def test_multisource_offsets_and_ambiguity() -> None:
    plan = ShortestPathFieldPlan(relation([(0, 2), (1, 2), (2, 3)], 4), source_count=2)
    result = plan.solve(
        jnp.asarray([1.0, 2.0, 1.0]),
        jnp.asarray([0, 1]),
        source_offsets=jnp.asarray([1.0, 0.0]),
    )
    np.testing.assert_allclose(result.distance, [1.0, 0.0, 2.0, 3.0])
    assert not result.derivative_regular
    np.testing.assert_allclose(result.margin[2:], 0.0)


@pytest.mark.parametrize(
    "cost,status",
    [
        (-1.0, ShortestPathFieldStatus.NEGATIVE_COST),
        (np.nan, ShortestPathFieldStatus.NONFINITE_INPUT),
        (np.inf, ShortestPathFieldStatus.NONFINITE_INPUT),
    ],
)
def test_invalid_cost_status(cost: float, status: ShortestPathFieldStatus) -> None:
    plan = ShortestPathFieldPlan(relation([(0, 1)], 2))
    result = jax.jit(lambda c: plan.solve(c, jnp.asarray([0])))(jnp.asarray([cost]))
    assert result.status == status
    assert not result.derivative_regular
    assert jnp.all(jnp.isnan(result.distance))


def test_reference_budget_and_sweep_refusal() -> None:
    rel = relation([(0, 1), (1, 2), (2, 3)], 4)
    refused = ShortestPathFieldPlan(
        rel, method="bounded-relaxation-reference", maximum_work=1
    ).solve(jnp.ones(3), jnp.asarray([0]))
    assert refused.status == ShortestPathFieldStatus.WORK_BUDGET_EXCEEDED
    assert refused.projected_work == 12
    limited = ShortestPathFieldPlan(
        rel, method="bounded-relaxation-reference", maximum_sweeps=1
    ).solve(jnp.ones(3), jnp.asarray([0]))
    assert limited.status == ShortestPathFieldStatus.MAXIMUM_SWEEPS_REACHED


@pytest.mark.parametrize(
    "budget,status",
    [
        ("work", ShortestPathFieldStatus.WORK_BUDGET_EXCEEDED),
        ("bytes", ShortestPathFieldStatus.HOST_BYTE_BUDGET_EXCEEDED),
    ],
)
def test_host_budget_refusal_never_calls_provider(
    budget: str, status: ShortestPathFieldStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("Refused host work must not execute Dijkstra.")

    monkeypatch.setattr(
        "phydrax.combinatorial._shortest_path_field.dijkstra", forbidden_provider
    )
    plan = ShortestPathFieldPlan(
        relation([(0, 1), (1, 2)], 3),
        maximum_work=1 if budget == "work" else 10_000_000,
        maximum_host_bytes=1 if budget == "bytes" else 268_435_456,
    )
    for solve in (plan.solve, jax.jit(lambda c, s: plan.solve(c, s))):
        result = solve(jnp.ones(2), jnp.asarray([0], dtype=jnp.int32))
        assert result.status == status
        assert result.work == 0
        assert result.sweeps == 0
        assert not result.derivative_regular
        assert jnp.all(jnp.isnan(result.distance))
    assert plan.host_projected_work == 8 * 7 * ((4).bit_length() + (3).bit_length())
    assert plan.projected_work == plan.host_projected_work
    assert plan.projected_host_bytes == (
        17 * 2 + 12 + 88 * 2 + 32 + 256 * 3 + 160 * 4 + 16 + 65_536
    )


@pytest.mark.parametrize("method", ["host-dijkstra", "bounded-relaxation-reference"])
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.strict_jax
def test_source_validation_precedes_integer_narrowing(
    method: ShortestPathFieldMethod, unsigned: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid original source must not execute Dijkstra.")

    monkeypatch.setattr(
        "phydrax.combinatorial._shortest_path_field.dijkstra", forbidden_provider
    )
    plan = ShortestPathFieldPlan(relation([(0, 1)], 2), method=method)
    compiled = jax.jit(lambda c, s: plan.solve(c, s))
    values = [2**31 - 1, 2**31, 2**32, 2**32 + 1, 2**63 - 1]
    if unsigned:
        values.append(2**64 - 1)
    if not unsigned:
        values.extend([-(2**31), -(2**32), -(2**32) + 1, -(2**63)])
    dtype = np.uint64 if unsigned else np.int64
    for value in values:
        host = np.asarray([value], dtype=dtype)
        for source, solve in (
            (host, plan.solve),
            (jnp.asarray(host), plan.solve),
            (jnp.asarray(host), compiled),
        ):
            result = solve(jnp.ones(1), source)
            assert result.status == ShortestPathFieldStatus.INVALID_SOURCE
            assert jnp.all(jnp.isnan(result.distance))
            assert not result.derivative_regular


@pytest.mark.parametrize("method", ["host-dijkstra", "bounded-relaxation-reference"])
def test_fixed_source_cycle_gap_not_inherited(
    method: ShortestPathFieldMethod,
) -> None:
    plan = ShortestPathFieldPlan(relation([(0, 1), (1, 0), (1, 2)], 3), method=method)
    result = plan.solve(jnp.asarray([1e-10, 1e-10, 1.0]), np.asarray([0]))
    assert result.status == ShortestPathFieldStatus.SUCCESS
    np.testing.assert_array_equal(result.margin, np.inf)
    assert result.derivative_regular


def test_valid_narrow_source_dtype_with_larger_vertex_space() -> None:
    plan = ShortestPathFieldPlan(relation([], 300))
    source = jnp.asarray([200], dtype=jnp.uint8)
    result = jax.jit(lambda s: plan.solve(jnp.empty(0, jnp.float64), s))(source)
    assert result.status == ShortestPathFieldStatus.SUCCESS
    assert result.distance[200] == 0
    assert jnp.sum(result.reachable) == 1


@settings(max_examples=8, deadline=None)
@given(st.integers(min_value=0, max_value=10000))
def test_random_graph_independent_scipy(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 12
    mask = rng.random((n, n)) < 0.18
    np.fill_diagonal(mask, False)
    u, v = np.indices((n, n))
    u, v = u.ravel(), v.ravel()
    costs = rng.uniform(0.1, 3.0, n * n)
    rel = EdgeRelation(
        u.astype(np.int32),
        v.astype(np.int32),
        source_size=n,
        target_size=n,
        valid=mask.ravel(),
    )
    expected = dijkstra(
        csr_matrix(
            (costs[mask.ravel()], (u[mask.ravel()], v[mask.ravel()])), shape=(n, n)
        ),
        indices=0,
    )
    result = ShortestPathFieldPlan(rel).solve(costs, jnp.asarray([0]))
    assert result.status == ShortestPathFieldStatus.SUCCESS
    np.testing.assert_allclose(result.distance, expected, atol=1e-12)
    assert result.bellman_residual < 1e-12
    assert result.dual_residual < 1e-12


def test_invalid_routes_impassable_and_parallel_routes() -> None:
    rel = EdgeRelation(
        np.asarray([0, 0, 0], np.int32),
        np.asarray([1, 1, 2], np.int32),
        source_size=3,
        target_size=3,
        valid=np.asarray([True, True, False]),
    )
    result = ShortestPathFieldPlan(rel).solve(
        jnp.asarray([2.0, 1.0, np.nan]), jnp.asarray([0])
    )
    np.testing.assert_allclose(result.distance[:2], [0.0, 1.0])
    assert np.isinf(result.distance[2])
    assert result.predecessor_edge[1] == 1


def test_host_callback_jit_vmap_and_provenance() -> None:
    plan = ShortestPathFieldPlan(relation([(0, 1), (1, 2)], 3))
    costs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    result = jax.jit(jax.vmap(lambda c: plan.solve(c, jnp.asarray([0]))))(costs)
    np.testing.assert_allclose(result.distance, [[0.0, 1.0, 3.0], [0.0, 3.0, 7.0]])
    np.testing.assert_array_equal(result.status, 0)
    assert result.method == "host-dijkstra"
    assert result.host_synchronization


def test_isolated_source_empty_relation_and_negative_offset() -> None:
    plan = ShortestPathFieldPlan(relation([], 3))
    result = plan.solve(
        jnp.asarray([], dtype=jnp.float64),
        jnp.asarray([1]),
        source_offsets=jnp.asarray([-2.0]),
    )
    np.testing.assert_array_equal(result.reachable, [False, True, False])
    assert result.distance[1] == -2.0
    assert result.status == ShortestPathFieldStatus.SUCCESS
    assert result.derivative_regular
    gradient = jax.grad(
        lambda o: plan.solve(
            jnp.asarray([], dtype=jnp.float64), jnp.asarray([1]), source_offsets=o
        ).distance[1]
    )(jnp.asarray([-2.0]))
    np.testing.assert_allclose(gradient, [1.0])


def test_duplicate_sources_nonregular_and_dominated_source() -> None:
    plan = ShortestPathFieldPlan(relation([(0, 1), (1, 2)], 3), source_count=2)
    duplicate = plan.solve(jnp.ones(2), jnp.asarray([0, 0]))
    assert not duplicate.derivative_regular
    np.testing.assert_allclose(duplicate.margin, 0.0)
    dominated = plan.solve(
        jnp.ones(2), jnp.asarray([0, 1]), source_offsets=jnp.asarray([0.0, 5.0])
    )
    np.testing.assert_allclose(dominated.distance, [0.0, 1.0, 2.0])
    assert dominated.predecessor_edge[1] == 0
    assert dominated.derivative_regular
