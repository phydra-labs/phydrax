"""Independent electrical-network oracles for grounded resistance."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import phydrax as phx


def relation(n: int, edges: list[tuple[int, int]]) -> phx.sparse.EdgeRelation:
    pairs = np.asarray(edges, dtype=np.int32).reshape((-1, 2))
    return phx.sparse.EdgeRelation(pairs[:, 0], pairs[:, 1], source_size=n, target_size=n)


def reference(n: int, edges: list[tuple[int, int]], w: np.ndarray) -> np.ndarray:
    incidence = np.zeros((len(edges), n), dtype=np.float64)
    for i, (u, v) in enumerate(edges):
        incidence[i, u], incidence[i, v] = 1, -1
    inverse = np.linalg.pinv(incidence.T @ (w[:, None] * incidence))
    diagonal = np.diag(inverse)
    return diagonal[:, None] + diagonal[None, :] - 2 * inverse


@pytest.mark.strict_jax
def test_series_parallel_complete_cycle() -> None:
    cases = [
        (4, [(0, 1), (1, 2), (2, 3)], np.array([2.0, 3.0, 4.0])),
        (2, [(0, 1), (0, 1)], np.array([2.0, 3.0])),
        (5, [(u, v) for u in range(5) for v in range(u + 1, 5)], np.full(10, 2.0)),
        (6, [(i, i + 1) for i in range(5)] + [(0, 5)], np.full(6, 3.0)),
    ]
    for n, edges, w in cases:
        result = phx.graph.EffectiveResistancePlan(
            relation(n, edges), np.arange(n)
        ).solve(jnp.asarray(w))
        assert result.status == phx.graph.EffectiveResistanceStatus.SUCCESS
        np.testing.assert_allclose(result.resistance, reference(n, edges, w), atol=1e-8)
        if n == 4:
            np.testing.assert_allclose(result.resistance[0, 3], np.sum(1 / w))
        elif n == 2:
            np.testing.assert_allclose(result.resistance[0, 1], 1 / np.sum(w))
        elif n == 5:
            np.testing.assert_allclose(result.resistance[0, 1], 2 / (n * w[0]))
        else:
            np.testing.assert_allclose(result.resistance[0, 2], 2 * (n - 2) / (n * w[0]))


@pytest.mark.strict_jax
def test_random_foster_metric_rayleigh_gradient_blocks_jit() -> None:
    n = 6
    edges = [(i, i + 1) for i in range(n - 1)] + [(0, 3), (1, 5), (2, 4)]
    w = np.random.default_rng(12).uniform(0.5, 2.0, len(edges))
    rel = relation(n, edges)
    expected = reference(n, edges, w)
    results = []
    for block in (1, 3, n):
        padded = ((n + block - 1) // block) * block
        primal = (n * n + n * padded + (2 * (n - 1) + n) * block) * 8 + padded * 17
        budget = (
            primal
            + (
                n * n
                + (n - 1) * padded
                + (2 * (n - 1) + n + len(edges)) * block
                + len(edges)
            )
            * 8
        )
        plan = phx.graph.EffectiveResistancePlan(
            rel, np.arange(n), workspace_bytes=budget
        )
        assert plan.block_size == block
        estimate = plan.resource_estimate
        assert all(isinstance(value, int) for value in estimate.values())
        assert estimate["output_bytes"] == n * n * 8
        assert estimate["rhs_bytes"] == (n - 1) * block * 8
        assert estimate["potential_bytes"] == estimate["rhs_bytes"]
        assert estimate["query_block_bytes"] == n * block * 8
        assert estimate["conductance_gradient_bytes"] == len(edges) * 8
        assert estimate["retained_potential_bytes"] == (n - 1) * padded * 8
        assert estimate["primal_workspace_bytes"] == primal
        assert estimate["derivative_workspace_bytes"] == budget
        assert estimate["budgeted_workspace_bytes"] == max(
            estimate["primal_workspace_bytes"], estimate["derivative_workspace_bytes"]
        )
        assert estimate["budgeted_workspace_bytes"] <= budget
        if block == 1:
            minimum = estimate["budgeted_workspace_bytes"]
            for insufficient in (minimum - 1, estimate["primal_workspace_bytes"]):
                with pytest.raises(ValueError, match="one RHS"):
                    phx.graph.EffectiveResistancePlan(
                        rel, np.arange(n), workspace_bytes=insufficient
                    )
        elif block == 3:
            # Forward-only planning could fit all columns; derivative storage cannot.
            full = phx.graph.EffectiveResistancePlan(rel, np.arange(n)).resource_estimate
            assert full["primal_workspace_bytes"] <= budget
            assert full["derivative_workspace_bytes"] > budget
        results.append(plan.solve(jnp.asarray(w)).resistance)
    limited = phx.graph.EffectiveResistancePlan(
        rel,
        np.arange(n),
        resources=phx.linalg.SolveResourcePolicy(krylov_basis_bytes=2 * 6 * (n - 1) * 8),
    )
    assert limited.block_size == 2
    results.append(limited.solve(jnp.asarray(w)).resistance)
    for result in results:
        np.testing.assert_allclose(result, expected, atol=1e-8)
    np.testing.assert_allclose(
        sum(w[i] * expected[u, v] for i, (u, v) in enumerate(edges)), n - 1, atol=1e-10
    )
    assert np.all(
        expected[:, :, None] <= expected[:, None, :] + expected.T[None, :, :] + 1e-10
    )
    plan = phx.graph.EffectiveResistancePlan(rel, np.arange(n))
    compiled = jax.jit(lambda values: plan.solve(values))
    np.testing.assert_allclose(
        compiled(jnp.asarray(w * 1.1)).resistance, expected / 1.1, atol=1e-8
    )
    assert np.all(
        np.asarray(compiled(jnp.asarray(w * 1.1)).resistance) <= expected + 1e-9
    )

    def objective(values: jax.Array) -> jax.Array:
        return plan.solve(values).resistance[1, 5]

    gradient = jax.grad(objective)(jnp.asarray(w))
    incidence = np.zeros((len(edges), n), dtype=np.float64)
    for i, (u, v) in enumerate(edges):
        incidence[i, u], incidence[i, v] = 1, -1
    rhs = np.zeros(n, dtype=np.float64)
    rhs[1], rhs[5] = 1, -1
    phi = np.linalg.pinv(incidence.T @ (w[:, None] * incidence)) @ rhs
    np.testing.assert_allclose(gradient, -((incidence @ phi) ** 2), atol=1e-7)
    finite = np.zeros_like(w)
    for i in range(len(w)):
        delta = np.zeros_like(w)
        delta[i] = 1e-4
        finite[i] = (
            float(objective(jnp.asarray(w + delta)))
            - float(objective(jnp.asarray(w - delta)))
        ) / 2e-4
    np.testing.assert_allclose(gradient, finite, atol=1e-6)


@pytest.mark.strict_jax
def test_disconnected_ground_relabel_invalid_budget() -> None:
    rel = relation(4, [(0, 1)])
    plan = phx.graph.EffectiveResistancePlan(rel, np.arange(4))
    result = plan.solve(jnp.array([2.0], dtype=jnp.float64))
    assert result.status == 0
    assert not result.connected[0, 2]
    assert np.isinf(result.resistance[0, 2])
    np.testing.assert_array_equal(np.diag(result.resistance), 0)
    other = phx.graph.EffectiveResistancePlan(
        relation(4, [(2, 3)]), np.array([3, 2, 0, 1])
    ).solve(jnp.array([2.0], dtype=jnp.float64))
    np.testing.assert_allclose(result.resistance, other.resistance)
    for value in (0.0, -1.0, np.nan):
        invalid = plan.solve(jnp.array([value], dtype=jnp.float64))
        assert invalid.status == phx.graph.EffectiveResistanceStatus.INVALID_CONDUCTANCE
        assert np.all(np.isnan(invalid.resistance))
    with pytest.raises(ValueError, match="one RHS"):
        phx.graph.EffectiveResistancePlan(rel, np.arange(4), workspace_bytes=1)
    for edges, message in [([(1, 1)], "self loops"), ([(1, 0)], "canonical")]:
        with pytest.raises(ValueError, match=message):
            phx.graph.EffectiveResistancePlan(
                relation(2, edges), np.array([0, 1], dtype=np.int32)
            )


@pytest.mark.strict_jax
def test_singletons_and_parallel_gradient() -> None:
    result = phx.graph.EffectiveResistancePlan(
        relation(2, []), np.array([0, 1], dtype=np.int32)
    ).solve(jnp.empty(0, dtype=jnp.float64))
    assert result.status == 0
    assert np.isinf(result.resistance[0, 1])
    plan = phx.graph.EffectiveResistancePlan(
        relation(2, [(0, 1), (0, 1)]), np.array([0, 1], dtype=np.int32)
    )

    def objective(w: jax.Array) -> jax.Array:
        return plan.solve(w).resistance[0, 1]

    np.testing.assert_allclose(
        jax.grad(objective)(jnp.array([2.0, 3.0])), [-0.04, -0.04], atol=1e-9
    )


@pytest.mark.strict_jax
def test_default_grid_gradient_is_finite_and_finite_difference_correct() -> None:
    """Moderately conditioned grid catches inadequate implicit Krylov solves."""
    rel = phx.topology.grid_adjacency_relation((24, 24), (False, False))
    plan = phx.graph.EffectiveResistancePlan(
        rel, np.array([0, 575, 288, 574], dtype=np.int32)
    )
    w = jnp.ones(rel.capacity, dtype=jnp.float64)

    def objective(values: jax.Array) -> jax.Array:
        return jnp.mean(plan.solve(values).resistance)

    compiled = jax.jit(objective)
    gradient = jax.jit(jax.grad(objective))(w)
    assert np.all(np.isfinite(gradient))
    # Scaling every edge scales every resistance inversely (Euler identity).
    np.testing.assert_allclose(jnp.sum(gradient), -compiled(w), rtol=2e-6, atol=1e-7)
    for edge in (0, rel.capacity // 2, rel.capacity - 1):
        delta = jnp.zeros_like(w).at[edge].set(1e-3)
        finite = (compiled(w + delta) - compiled(w - delta)) / 2e-3
        np.testing.assert_allclose(gradient[edge], finite, atol=2e-6, rtol=2e-4)


@pytest.mark.strict_jax
@settings(max_examples=5, deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.5, max_value=4.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=5,
    )
)
def test_bounded_metric_and_rayleigh_properties(values: list[float]) -> None:
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]
    weights = np.asarray(values, dtype=np.float64)
    plan = phx.graph.EffectiveResistancePlan(
        relation(4, edges), np.arange(4, dtype=np.int32)
    )
    baseline = np.asarray(plan.solve(jnp.asarray(weights)).resistance)
    raised = weights.copy()
    raised[2] += 0.25
    improved = np.asarray(plan.solve(jnp.asarray(raised)).resistance)
    assert np.all(improved <= baseline + 1e-7)
    for middle in range(4):
        assert np.all(
            baseline <= baseline[:, middle, None] + baseline[None, middle, :] + 1e-7
        )
    np.testing.assert_allclose(baseline, reference(4, edges, weights), atol=1e-7)


@pytest.mark.strict_jax
def test_failure_evidence_dtype_padding_and_forest_foster() -> None:
    edges = [(i, i + 1) for i in range(5)]
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        tolerance=phx.linalg.TolerancePolicy(relative=1e-14, absolute=0.0, max_steps=1),
    )
    plan = phx.graph.EffectiveResistancePlan(
        relation(6, edges), np.arange(6, dtype=np.int32), policy=policy
    )
    failed = plan.solve(jnp.ones(5, dtype=jnp.float64))
    assert failed.status == phx.graph.EffectiveResistanceStatus.SOLVE_FAILED
    assert np.any(np.asarray(failed.solve_status) != 0)
    assert np.all(np.isnan(failed.resistance))
    padded = phx.sparse.EdgeRelation(
        np.array([0, 99], dtype=np.int32),
        np.array([1, -5], dtype=np.int32),
        valid=np.array([True, False]),
        source_size=2,
        target_size=2,
    )
    single = phx.graph.EffectiveResistancePlan(
        padded, np.array([0, 1], dtype=np.int32), dtype=jnp.float32
    )
    result = single.solve(jnp.array([2.0, np.nan], dtype=jnp.float32))
    assert result.resistance.dtype == jnp.float32
    np.testing.assert_allclose(result.resistance[0, 1], 0.5, atol=1e-6)
    with pytest.raises(TypeError, match="dtype"):
        single.solve(jnp.ones(2, dtype=jnp.float64))
    with pytest.raises(ValueError, match="shape"):
        single.solve(jnp.ones(3, dtype=jnp.float32))
    forest = phx.graph.EffectiveResistancePlan(
        relation(5, [(0, 1), (2, 3)]), np.arange(5, dtype=np.int32)
    ).solve(jnp.array([2.0, 3.0], dtype=jnp.float64))
    np.testing.assert_allclose(
        2 * forest.resistance[0, 1] + 3 * forest.resistance[2, 3], 5 - 3, atol=1e-9
    )
