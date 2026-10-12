"""Consumer-visible RSP contracts against independent hitting-path oracles."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

import phydrax as phx


def relation(n: int, edges: list[tuple[int, int]]) -> phx.sparse.EdgeRelation:
    pairs = np.asarray(edges, dtype=np.int32).reshape((-1, 2))
    return phx.sparse.EdgeRelation(pairs[:, 0], pairs[:, 1], source_size=n, target_size=n)


def graph(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "directed-leakage":
        edges = [(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (3, 4), (4, 3)]
        n = 5
    else:
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        edges += [(v, u) for u, v in edges]
        n = 4
    affinities = np.zeros((n, n), dtype=np.float64)
    costs = np.zeros_like(affinities)
    for k, (u, v) in enumerate(edges):
        affinities[u, v] = 0.7 + 0.13 * k
        costs[u, v] = 0.4 + 0.17 * (k % 4)
    return affinities, costs


def prepare(
    affinities: np.ndarray,
    costs: np.ndarray,
    nodes: np.ndarray,
    policy: phx.linalg.LinearSolvePolicy | None = None,
) -> tuple[phx.graph.RandomizedShortestPathPlan, jax.Array, jax.Array]:
    source, target = np.nonzero(affinities)
    rel = relation(
        len(affinities), list(zip(source.tolist(), target.tolist(), strict=True))
    )
    return (
        phx.graph.RandomizedShortestPathPlan(rel, nodes, linear_policy=policy),
        jnp.asarray(affinities[source, target]),
        jnp.asarray(costs[source, target]),
    )


def exact(
    affinities: np.ndarray,
    costs: np.ndarray,
    theta: float,
    nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """90-digit partition and first moment; normalize original affinities exactly."""
    n = len(affinities)
    phi = np.full((len(nodes), len(nodes)), np.inf, dtype=np.float64)
    expected = np.full_like(phi, np.inf)
    with mp.workdps(90):
        rows = [mp.fsum(mp.mpf(v) for v in row) for row in affinities]
        for b, target in enumerate(nodes):
            reach = {int(target)}
            while True:
                enlarged = reach | {
                    i for i in range(n) if any(affinities[i, j] > 0 for j in reach)
                }
                if enlarged == reach:
                    break
                reach = enlarged
            keep = sorted(reach - {int(target)})
            w = mp.matrix(n)
            for i, j in zip(*np.nonzero(affinities), strict=True):
                if i != target:
                    w[i, j] = (
                        mp.mpf(affinities[i, j])
                        / rows[i]
                        * mp.exp(-mp.mpf(theta) * mp.mpf(costs[i, j]))
                    )
            if keep:
                operator = mp.matrix(
                    [[mp.mpf(i == j) - w[i, j] for j in keep] for i in keep]
                )
                z = mp.lu_solve(operator, mp.matrix([w[i, target] for i in keep]))
                full = {int(target): mp.mpf(1), **dict(zip(keep, z, strict=True))}
                rhs = mp.matrix(
                    [
                        mp.fsum(w[i, j] * mp.mpf(costs[i, j]) * full[j] for j in reach)
                        for i in keep
                    ]
                )
                moment = mp.lu_solve(operator, rhs)
                for a, source in enumerate(nodes):
                    if source in keep:
                        k = keep.index(source)
                        phi[a, b] = float(-mp.log(z[k]) / mp.mpf(theta))
                        expected[a, b] = float(moment[k] / z[k])
            phi[b, b] = expected[b, b] = 0.0
    return phi, expected


@pytest.mark.parametrize("name", ["directed-leakage", "undirected"], ids=str)
def test_exact_reference_across_temperatures(name: str) -> None:
    affinities, costs = graph(name)
    nodes = np.arange(len(affinities), dtype=np.int32)
    plan, a, c = prepare(affinities, costs, nodes)
    solve = jax.jit(lambda weights, cost, theta: plan.solve(weights, cost, theta))
    for theta in (1e-9, 1e-3, 1.0, 1e3):
        result = solve(a, c, jnp.asarray(theta, dtype=jnp.float64))
        phi, expected = exact(affinities, costs, theta, nodes)
        assert result.status == phx.graph.RandomizedShortestPathStatus.SUCCESS
        np.testing.assert_allclose(result.free_energy, phi, rtol=1e-10, atol=0)
        np.testing.assert_allclose(result.expected_cost, expected, rtol=1e-10, atol=0)


def test_long_path_dynamic_range() -> None:
    affinities = np.diag(np.ones(49, dtype=np.float64), 1)
    costs = 20.0 * affinities
    nodes = np.asarray([0, 49], dtype=np.int32)
    plan, a, c = prepare(affinities, costs, nodes)
    solve = jax.jit(lambda weights, cost, theta: plan.solve(weights, cost, theta))
    for theta in (1.0, 10.0):
        result = solve(a, c, jnp.asarray(theta))
        phi, expected = exact(affinities, costs, theta, nodes)
        assert result.status == phx.graph.RandomizedShortestPathStatus.SUCCESS
        np.testing.assert_allclose(result.free_energy, phi, rtol=1e-12, atol=0)
        np.testing.assert_allclose(result.expected_cost, expected, rtol=1e-12, atol=0)


def test_low_temperature_shortest_path_limit() -> None:
    affinities, costs = graph("undirected")
    # The O(1/theta) path-entropy correction is small relative to these costs.
    costs = 100.0 * costs
    plan, a, c = prepare(affinities, costs, np.arange(4, dtype=np.int32))
    result = plan.solve(a, c, jnp.asarray(1e5))
    shortest = dijkstra(np.where(affinities > 0, costs, 0), directed=True)
    np.testing.assert_allclose(result.free_energy, shortest, rtol=1e-6, atol=0)


def test_high_temperature_hitting_cost_limit() -> None:
    affinities, costs = graph("undirected")
    transition = affinities / affinities.sum(axis=1, keepdims=True)
    expected = np.zeros_like(affinities)
    for target in range(4):
        keep = np.arange(4) != target
        expected[keep, target] = np.linalg.solve(
            np.eye(3) - transition[np.ix_(keep, keep)],
            np.sum(transition * costs, axis=1)[keep],
        )
    plan, a, c = prepare(affinities, costs, np.arange(4, dtype=np.int32))
    result = plan.solve(a, c, jnp.asarray(1e-9))
    np.testing.assert_allclose(result.expected_cost, expected, rtol=1e-8, atol=0)


def test_unit_cost_commute_limit() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]
    affinities = np.zeros((4, 4), dtype=np.float64)
    for u, v in edges:
        affinities[u, v] = affinities[v, u] = 1.0
    nodes = np.arange(4, dtype=np.int32)
    plan, a, c = prepare(affinities, affinities, nodes)
    result = plan.solve(a, c, jnp.asarray(1e-9))
    resistance = phx.graph.EffectiveResistancePlan(relation(4, edges), nodes).solve(
        jnp.ones(len(edges), dtype=jnp.float64)
    )
    assert resistance.status == phx.graph.EffectiveResistanceStatus.SUCCESS
    np.testing.assert_allclose(
        result.expected_cost + result.expected_cost.T,
        2 * len(edges) * resistance.resistance,
        rtol=1e-6,
        atol=0,
    )


def test_support_and_leakage_asymptote() -> None:
    # Node 0 hits target 1 with probability 1/4; the other branch is a closed class.
    edges = [(0, 1), (0, 2), (2, 3), (3, 2)]
    plan = phx.graph.RandomizedShortestPathPlan(
        relation(4, edges), np.arange(4, dtype=np.int32)
    )
    solve = jax.jit(lambda weights, cost, theta: plan.solve(weights, cost, theta))
    values = []
    for theta in (1e-3, 1e-9):
        result = solve(
            jnp.asarray([1.0, 3.0, 1.0, 1.0]),
            jnp.ones(4, dtype=jnp.float64),
            jnp.asarray(theta),
        )
        assert result.status == phx.graph.RandomizedShortestPathStatus.SUCCESS
        support = np.asarray(result.reachable)
        np.testing.assert_array_equal(np.diag(support), True)
        for field in (result.free_energy, result.expected_cost):
            np.testing.assert_array_equal(np.diag(field), 0)
            assert np.all(np.isposinf(np.asarray(field)[~support]))
        assert not support[2, 1]
        assert np.isfinite(result.free_energy[0, 1])
        np.testing.assert_allclose(
            theta * result.free_energy[0, 1], -np.log(0.25) + theta, rtol=1e-12
        )
        values.append(float(result.free_energy[0, 1]))
    assert values[1] > values[0]


def test_symmetrized_properties() -> None:
    affinities, costs = graph("directed-leakage")
    plan, a, c = prepare(affinities, costs, np.arange(5, dtype=np.int32))
    result = plan.solve(a, c, jnp.asarray(0.7))
    np.testing.assert_array_equal(
        result.free_energy_distance, (result.free_energy + result.free_energy.T) / 2
    )
    np.testing.assert_array_equal(
        result.expected_cost_dissimilarity,
        (result.expected_cost + result.expected_cost.T) / 2,
    )


@pytest.mark.parametrize("policy_name", ["default", "superlu"], ids=str)
def test_derivatives_against_finite_differences(policy_name: str) -> None:
    policy = (
        None
        if policy_name == "default"
        else phx.linalg.LinearSolvePolicy(phx.linalg.SparseLU(provider="jax-cpu"))
    )
    affinities, costs = graph("directed-leakage")
    plan, a, c = prepare(affinities, costs, np.arange(5, dtype=np.int32), policy)
    solve = jax.jit(lambda weights, cost, theta: plan.solve(weights, cost, theta))
    theta = jnp.asarray(0.7)
    for field_name in ("free_energy", "expected_cost"):

        def objective(
            weights: jax.Array, cost: jax.Array, temperature: jax.Array
        ) -> jax.Array:
            result = solve(weights, cost, temperature)
            field = (
                result.free_energy
                if field_name == "free_energy"
                else result.expected_cost
            )
            return jnp.sum(jnp.where(result.reachable, field, 0.0))

        gradients = jax.grad(objective, argnums=(0, 1, 2))(a, c, theta)
        for argument, gradient in enumerate(gradients):
            base = (a, c, theta)[argument]
            finite = np.zeros(base.shape, dtype=np.float64)
            for index in np.ndindex(base.shape):
                # O(h^2) truncation balanced against float64 solve error.
                step = 1e-5 * max(1.0, abs(float(base[index])))
                bump = jnp.zeros_like(base).at[index].set(step)
                plus = [a, c, theta]
                minus = [a, c, theta]
                plus[argument], minus[argument] = base + bump, base - bump
                finite[index] = (float(objective(*plus)) - float(objective(*minus))) / (
                    2 * step
                )
            np.testing.assert_allclose(gradient, finite, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("policy_name", ["default", "superlu"], ids=str)
def test_derivative_temperature_identity(policy_name: str) -> None:
    policy = (
        None
        if policy_name == "default"
        else phx.linalg.LinearSolvePolicy(phx.linalg.SparseLU(provider="jax-cpu"))
    )
    affinities, costs = graph("directed-leakage")
    plan, a, c = prepare(affinities, costs, np.arange(5, dtype=np.int32), policy)
    solve = jax.jit(lambda weights, cost, theta: plan.solve(weights, cost, theta))
    theta = jnp.asarray(0.7)
    _, tangent = jax.jvp(
        lambda th: th * solve(a, c, th).free_energy, (theta,), (jnp.asarray(1.0),)
    )
    expected = np.asarray(solve(a, c, theta).expected_cost)
    finite = np.isfinite(expected)
    np.testing.assert_allclose(
        np.asarray(tangent)[finite], expected[finite], rtol=1e-10, atol=0
    )


def test_jit_reuses_trace_with_dynamic_costs_and_temperature() -> None:
    affinities, costs = graph("undirected")
    plan, a, c = prepare(affinities, costs, np.asarray([0, 2], dtype=np.int32))
    traces = []

    def traced(
        weights: jax.Array, cost: jax.Array, theta: jax.Array
    ) -> phx.graph.RandomizedShortestPathResult:
        traces.append(None)
        return plan.solve(weights, cost, theta)

    solve = jax.jit(traced)
    for scale, theta in ((1.0, 0.3), (1.2, 0.7), (0.8, 2.0)):
        result = solve(a, scale * c, jnp.asarray(theta))
        eager = plan.solve(a, scale * c, jnp.asarray(theta))
        np.testing.assert_allclose(
            result.free_energy, eager.free_energy, rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            result.expected_cost, eager.expected_cost, rtol=1e-12, atol=0
        )
        assert (
            result.status
            == eager.status
            == phx.graph.RandomizedShortestPathStatus.SUCCESS
        )
    assert len(traces) == 1


@pytest.fixture(scope="module")
def small_plan() -> phx.graph.RandomizedShortestPathPlan:
    return phx.graph.RandomizedShortestPathPlan(
        relation(2, [(0, 1), (1, 0)]), np.arange(2, dtype=np.int32)
    )


@pytest.fixture(scope="module")
def small_solve(
    small_plan: phx.graph.RandomizedShortestPathPlan,
) -> Callable[[jax.Array, jax.Array, jax.Array], phx.graph.RandomizedShortestPathResult]:
    return jax.jit(lambda a, c, theta: small_plan.solve(a, c, theta))


@pytest.mark.parametrize(
    "kind,value",
    [
        ("affinity", 0.0),
        ("affinity", -1.0),
        ("affinity", np.nan),
        ("cost", -1.0),
        ("cost", np.nan),
        ("theta", 0.0),
        ("theta", -1.0),
        ("theta", np.inf),
    ],
    ids=[
        "zero-affinity",
        "negative-affinity",
        "nan-affinity",
        "negative-cost",
        "nan-cost",
        "zero-theta",
        "negative-theta",
        "infinite-theta",
    ],
)
def test_invalid_values(
    small_solve: Callable[
        [jax.Array, jax.Array, jax.Array], phx.graph.RandomizedShortestPathResult
    ],
    kind: str,
    value: float,
) -> None:
    a, c, theta = (
        jnp.ones(2, dtype=jnp.float64),
        jnp.ones(2, dtype=jnp.float64),
        jnp.asarray(1.0),
    )
    if kind == "affinity":
        a = a.at[0].set(value)
    elif kind == "cost":
        c = c.at[0].set(value)
    else:
        theta = jnp.asarray(value)
    result = small_solve(a, c, theta)
    assert result.status == phx.graph.RandomizedShortestPathStatus.INVALID_INPUT
    assert np.all(np.isnan(result.free_energy))
    assert np.all(np.isnan(result.expected_cost))
    np.testing.assert_array_equal(result.nonlinear_status, -1)


@pytest.mark.strict_jax
@pytest.mark.parametrize("argument", [0, 1, 2], ids=["affinity", "cost", "theta"])
@pytest.mark.parametrize("failure", ["shape", "dtype"], ids=str)
def test_solve_array_contract(
    small_plan: phx.graph.RandomizedShortestPathPlan, argument: int, failure: str
) -> None:
    values = [
        jnp.ones(2, dtype=jnp.float64),
        jnp.ones(2, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    ]
    values[argument] = (
        jnp.ones((1,), dtype=jnp.float64)
        if failure == "shape"
        else values[argument].astype(jnp.float32)
    )
    with pytest.raises(ValueError if failure == "shape" else TypeError):
        small_plan.solve(*values)


@pytest.mark.parametrize(
    "nodes",
    [
        np.asarray([-1]),
        np.asarray([2]),
        np.asarray([], dtype=np.int32),
        np.asarray([[0]]),
    ],
    ids=["negative", "out-of-range", "empty", "rank-two"],
)
def test_refuses_invalid_query_nodes(nodes: np.ndarray) -> None:
    with pytest.raises(ValueError):
        phx.graph.RandomizedShortestPathPlan(relation(2, [(0, 1)]), nodes)


@pytest.mark.parametrize("kind", ["self-loop", "non-square"], ids=str)
def test_refuses_invalid_relations(kind: str) -> None:
    rel = (
        relation(2, [(0, 0)])
        if kind == "self-loop"
        else phx.sparse.EdgeRelation(
            np.asarray([0]), np.asarray([1]), source_size=2, target_size=3
        )
    )
    with pytest.raises(ValueError):
        phx.graph.RandomizedShortestPathPlan(rel, np.asarray([0], dtype=np.int32))


def test_workspace_estimate_and_refusal(
    small_plan: phx.graph.RandomizedShortestPathPlan,
) -> None:
    estimate = small_plan.resource_estimate
    assert set(estimate) == {
        "output_bytes",
        "reach_mask_bytes",
        "primal_workspace_bytes",
        "derivative_workspace_bytes",
        "budgeted_workspace_bytes",
        "seed_work",
        "seed_host_peak_bytes",
        "seed_host_aggregate_bytes",
    }
    assert all(isinstance(value, int) and value >= 0 for value in estimate.values())
    budget = estimate["budgeted_workspace_bytes"]
    assert budget == max(
        estimate["primal_workspace_bytes"], estimate["derivative_workspace_bytes"]
    )
    plan = phx.graph.RandomizedShortestPathPlan(
        relation(2, [(0, 1), (1, 0)]),
        np.arange(2, dtype=np.int32),
        workspace_bytes=budget,
    )
    assert plan.resource_estimate["budgeted_workspace_bytes"] <= budget
    with pytest.raises(ValueError):
        phx.graph.RandomizedShortestPathPlan(
            relation(2, [(0, 1), (1, 0)]),
            np.arange(2, dtype=np.int32),
            workspace_bytes=budget - 1,
        )


def test_isolated_query_target_succeeds() -> None:
    plan = phx.graph.RandomizedShortestPathPlan(
        relation(3, [(0, 1), (1, 0)]), np.arange(3, dtype=np.int32)
    )
    result = plan.solve(
        jnp.ones(2, dtype=jnp.float64), jnp.ones(2, dtype=jnp.float64), jnp.asarray(1.0)
    )
    assert result.status == phx.graph.RandomizedShortestPathStatus.SUCCESS
    np.testing.assert_array_equal(result.reachable[:, 2], [False, False, True])
    for field in (result.free_energy, result.expected_cost):
        assert np.all(np.isposinf(np.asarray(field)[:2, 2]))
        assert field[2, 2] == 0


@pytest.mark.parametrize("mode", ["none", "rhs-only", "algorithmic"], ids=str)
def test_refuses_nonmathematical_linear_derivatives(
    mode: phx.linalg.DifferentiationMode,
) -> None:
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(),
        differentiation=phx.linalg.DifferentiationPolicy(mode),
    )
    with pytest.raises(ValueError, match="mode 'mathematical'"):
        phx.graph.RandomizedShortestPathPlan(
            relation(2, [(0, 1), (1, 0)]),
            np.arange(2, dtype=np.int32),
            linear_policy=policy,
        )


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "outgoing"])
def test_log_affinity_normalization_avoids_row_overflow(parallel: bool) -> None:
    edges = [(0, 1), (0, 1), (1, 0)] if parallel else [(0, 1), (0, 2), (1, 0), (2, 0)]
    plan = phx.graph.RandomizedShortestPathPlan(
        relation(2 if parallel else 3, edges), np.asarray([0, 1], dtype=np.int32)
    )
    costs = jnp.asarray([0.5, 1.5, 0.8] if parallel else [0.5, 1.5, 0.8, 0.7])
    base = jnp.ones(len(edges), dtype=jnp.float64)
    factors = jnp.asarray([1e308, 1e308, 1.0] if parallel else [1e308, 1e308, 1.0, 1.0])
    theta = jnp.asarray(0.7)
    solve = jax.jit(lambda affinity, cost, th: plan.solve(affinity, cost, th))
    expected, actual = solve(base, costs, theta), solve(base * factors, costs, theta)
    assert actual.status == phx.graph.RandomizedShortestPathStatus.SUCCESS
    for field in ("free_energy", "expected_cost"):

        def objective(affinity: jax.Array, cost: jax.Array, th: jax.Array) -> jax.Array:
            result = solve(affinity, cost, th)
            return jnp.sum(
                result.free_energy if field == "free_energy" else result.expected_cost
            )

        gradients = jax.grad(objective, (0, 1, 2))
        reference_gradient, reference_cost, reference_theta = gradients(
            base, costs, theta
        )
        gradient, cost_gradient, theta_gradient = gradients(base * factors, costs, theta)
        assert np.all(np.isfinite(gradient))
        assert np.all(np.isfinite(cost_gradient)) and np.isfinite(theta_gradient)
        np.testing.assert_allclose(cost_gradient, reference_cost, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            theta_gradient, reference_theta, rtol=1e-10, atol=1e-12
        )
        # Raw affinity cotangents at 1e308 can be subnormal and device-flushed.
        # Check their scaling where the mathematical derivative is representable.
        representable_factors = factors.at[:2].set(1e200)
        normal_gradient = gradients(base * representable_factors, costs, theta)[0]
        np.testing.assert_allclose(
            normal_gradient * representable_factors,
            reference_gradient,
            rtol=1e-10,
            atol=1e-12,
        )
    np.testing.assert_allclose(actual.free_energy, expected.free_energy, rtol=1e-12)
    np.testing.assert_allclose(actual.expected_cost, expected.expected_cost, rtol=1e-12)


@pytest.fixture(scope="module", params=["default", "superlu"])
def first_order_solve(
    request: pytest.FixtureRequest,
) -> Callable[[jax.Array], phx.graph.RandomizedShortestPathResult]:
    policy = (
        None
        if request.param == "default"
        else phx.linalg.LinearSolvePolicy(phx.linalg.SparseLU(provider="jax-cpu"))
    )
    plan = phx.graph.RandomizedShortestPathPlan(
        relation(3, [(0, 1), (0, 2), (1, 0), (2, 0)]),
        np.asarray([0, 1], dtype=np.int32),
        linear_policy=policy,
    )
    assert plan.maximum_derivative_order == 1
    return jax.jit(
        lambda theta: plan.solve(
            jnp.ones(4, dtype=jnp.float64),
            jnp.asarray([0.5, 1.5, 0.8, 0.7]),
            theta,
        )
    )


@pytest.mark.parametrize("field", ["free_energy", "expected_cost"], ids=str)
def test_first_order_jvp_vjp_grad_success(
    first_order_solve: Callable[[jax.Array], phx.graph.RandomizedShortestPathResult],
    field: str,
) -> None:
    theta = jnp.asarray(0.7)
    result = first_order_solve(theta)
    assert result.maximum_derivative_order == 1
    assert result.status == phx.graph.RandomizedShortestPathStatus.SUCCESS

    def objective(th: jax.Array) -> jax.Array:
        result = first_order_solve(th)
        return jnp.sum(
            result.free_energy if field == "free_energy" else result.expected_cost
        )

    _, tangent = jax.jvp(objective, (theta,), (jnp.asarray(1.0),))
    _, pullback = jax.vjp(objective, theta)
    gradient = jax.grad(objective)(theta)
    assert np.isfinite(gradient)
    np.testing.assert_allclose(tangent, gradient, rtol=1e-10)
    np.testing.assert_allclose(pullback(jnp.asarray(1.0))[0], gradient, rtol=1e-10)


@pytest.mark.parametrize("field", ["free_energy", "expected_cost"], ids=str)
@pytest.mark.parametrize(
    "route",
    ["grad-grad", "forward-over-reverse", "reverse-over-forward", "hessian"],
    ids=str,
)
def test_nested_derivatives_refused(
    first_order_solve: Callable[[jax.Array], phx.graph.RandomizedShortestPathResult],
    field: str,
    route: str,
) -> None:
    def objective(theta: jax.Array) -> jax.Array:
        result = first_order_solve(theta)
        return jnp.sum(
            result.free_energy if field == "free_energy" else result.expected_cost
        )

    theta = jnp.asarray(0.7)
    with pytest.raises(
        ValueError, match="derivative-unsupported:.*maximum_derivative_order=1"
    ):
        if route == "grad-grad":
            jax.grad(jax.grad(objective))(theta)
        elif route == "forward-over-reverse":
            jax.jvp(jax.grad(objective), (theta,), (jnp.asarray(1.0),))
        elif route == "reverse-over-forward":
            jax.grad(lambda th: jax.jvp(objective, (th,), (jnp.asarray(1.0),))[1])(theta)
        else:
            jax.hessian(objective)(theta)


def test_near_closed_transient_walk_retains_solve_evidence() -> None:
    affinities = np.asarray([[0.0, 1.0, 1e-8], [1.0, 0.0, 1e-8], [0.0, 0.0, 0.0]])
    costs = np.where(affinities > 0, 1.0, 0.0)
    nodes = np.asarray([0, 2], dtype=np.int32)
    plan, a, c = prepare(affinities, costs, nodes)
    result = plan.solve(a, c, jnp.asarray(1e-10))
    if result.status == phx.graph.RandomizedShortestPathStatus.SUCCESS:
        phi, expected = exact(affinities, costs, 1e-10, nodes)
        np.testing.assert_allclose(result.free_energy, phi, rtol=1e-6, atol=0)
        np.testing.assert_allclose(result.expected_cost, expected, rtol=1e-6, atol=0)
        np.testing.assert_array_equal(result.nonlinear_status, 0)
        np.testing.assert_array_equal(result.linear_status, 0)
        assert np.all(np.isfinite(result.residual_norm))
    else:
        assert result.status in (
            phx.graph.RandomizedShortestPathStatus.NONLINEAR_FAILED,
            phx.graph.RandomizedShortestPathStatus.LINEAR_SOLVE_FAILED,
        )
        failed = (np.asarray(result.nonlinear_status) != 0) | (
            np.asarray(result.linear_status) != 0
        )
        assert np.any(failed)
        assert np.all(np.isnan(np.asarray(result.free_energy)[:, failed]))
        assert np.all(np.isnan(np.asarray(result.expected_cost)[:, failed]))


@pytest.mark.parametrize(
    "method", ["host-dijkstra", "bounded-relaxation-reference"], ids=str
)
def test_total_seed_resource_admission(
    method: phx.combinatorial.ShortestPathFieldMethod,
) -> None:
    rel = relation(2, [(0, 1), (1, 0)])
    nodes = np.arange(2, dtype=np.int32)
    plan = phx.graph.RandomizedShortestPathPlan(rel, nodes, seed_method=method)
    estimate = plan.resource_estimate
    assert estimate["seed_work"] == 2 * plan.seed_plan.projected_work
    assert estimate["seed_host_aggregate_bytes"] == 2 * estimate["seed_host_peak_bytes"]
    assert estimate["seed_host_peak_bytes"] == (
        plan.seed_plan.projected_host_bytes if method == "host-dijkstra" else 0
    )
    host_budget = max(1, estimate["seed_host_aggregate_bytes"])
    admitted = phx.graph.RandomizedShortestPathPlan(
        rel,
        nodes,
        seed_method=method,
        seed_maximum_work=estimate["seed_work"],
        seed_maximum_host_bytes=host_budget,
    )
    assert admitted.resource_estimate == estimate
    with pytest.raises(ValueError, match="seed work.*work budget"):
        phx.graph.RandomizedShortestPathPlan(
            rel,
            nodes,
            seed_method=method,
            seed_maximum_work=estimate["seed_work"] - 1,
        )
    if method == "host-dijkstra":
        with pytest.raises(ValueError, match="seed host bytes.*host byte budget"):
            phx.graph.RandomizedShortestPathPlan(
                rel,
                nodes,
                seed_method=method,
                seed_maximum_host_bytes=host_budget - 1,
            )
