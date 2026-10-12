"""Qualify the shipped randomized-shortest-path API and fault-adequacy controls.

Run with PYTHONPATH=. and --output PATH; --smoke reduces graphs and grids but
retains the path50 conditioning counterexample. Exit zero requires all package
API gates to pass and every deliberately inadequate linear route to fail.
NumPy/SciPy and 90-digit mpmath own independent references; JAX executes the API.
Retained records must be regenerated after source/driver changes. The shared
stored-record validator owns last-step evidence, not this gate layout.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import scipy
from jax import Array
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import gmres, splu
from scipy.special import logsumexp

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
)


EVIDENCE_FIELDS = (
    "environment",
    "command",
    "configuration",
    "provenance",
    "gates",
    "negative_controls",
    "fault_adequacy",
    "overall",
    "rsp_api_shipped",
)


jax.config.update("jax_enable_x64", True)

type Matrix = NDArray[np.float64]
type Mask = NDArray[np.bool_]
type Record = dict[str, str | int | float]
type Solve = Callable[[Array, Array, Array], phx.graph.RandomizedShortestPathResult]
THETAS = (1e-9, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
GRAPH_NAMES = ("path50", "grid49", "random40", "directed40")
SMOKE_GRAPH_NAMES = ("path8", "grid9", "random8", "directed8")
SCALING_THETAS = (1e-6, 1e-3, 1.0, 1000.0)
# Explicit opt-in planning budget for the informational SuperLU rows. The planner
# charges a dense n^2 bound because SuperLU chooses its ordering internally; it is
# not an allocation (side 256 needs 32 GiB of budget, far above actual use).
CPU_FACTORIZATION_BYTES = 64 * 1024**3


def graph(name: str) -> tuple[Matrix, Matrix]:
    rng = np.random.default_rng(731)
    if name in ("path50", "path8"):
        nodes = 50 if name == "path50" else 8
        adjacency = np.diag(np.ones(nodes - 1, dtype=np.float64), 1)
        adjacency += adjacency.T
        costs = adjacency.copy()
    elif name in ("grid49", "grid9"):
        adjacency = grid(7 if name == "grid49" else 3).toarray()
        costs = (adjacency > 0).astype(np.float64)
    elif name in ("random40", "directed40", "random8", "directed8"):
        n = 40 if name in ("random40", "directed40") else 8
        adjacency = rng.uniform(0.2, 2.0, (n, n))
        adjacency *= rng.random((n, n)) < 0.12
        np.fill_diagonal(adjacency, 0.0)
        if name in ("random40", "random8"):
            adjacency = adjacency + adjacency.T
        for i in range(n):
            adjacency[i, (i + 1) % n] = 1.0
            adjacency[(i + 1) % n, i] = 1.0
        costs = rng.uniform(0.2, 3.0, (n, n))
        if name in ("random40", "random8"):
            costs = 0.5 * (costs + costs.T)
        costs *= adjacency > 0
    else:
        raise ValueError(f"Unknown graph identity: {name}")
    return adjacency / adjacency.sum(axis=1, keepdims=True), costs


def grid(side: int) -> sparse.csr_matrix[np.float64]:
    line = sparse.diags(
        np.ones((2, side - 1), dtype=np.float64), [-1, 1], shape=(side, side)
    )
    identity = sparse.eye(side, format="csr", dtype=np.float64)
    return sparse.kron(line, identity, format="csr") + sparse.kron(
        identity, line, format="csr"
    )


def route_callable(plan: phx.graph.RandomizedShortestPathPlan) -> Solve:
    """Keep host-prepared topology fixed; numerical solve inputs remain dynamic."""

    def solve(
        affinity: Array, cost: Array, theta: Array
    ) -> phx.graph.RandomizedShortestPathResult:
        return plan.solve(affinity, cost, theta)

    return solve


def package_route(
    p: Matrix, c: Matrix, *, linear_policy: phx.linalg.LinearSolvePolicy | None = None
) -> tuple[Solve, Array, Array]:
    rows, columns = (np.asarray(v, dtype=np.int32) for v in np.nonzero(p))
    n = p.shape[0]
    relation = phx.sparse.EdgeRelation(rows, columns, source_size=n, target_size=n)
    plan = phx.graph.RandomizedShortestPathPlan(
        relation, np.arange(n, dtype=np.int32), linear_policy=linear_policy
    )
    return (
        jax.jit(route_callable(plan)),
        jnp.asarray(p[rows, columns]),
        jnp.asarray(c[rows, columns]),
    )


def potentials(p: Matrix, c: Matrix, theta: float, target: int) -> Matrix:
    rows, columns = np.nonzero(p)
    augmented = sparse.csr_matrix(
        (theta * c[rows, columns] - np.log(p[rows, columns]), (rows, columns)),
        shape=p.shape,
    )
    return np.asarray(dijkstra(augmented.T, indices=target), dtype=np.float64)


def system(
    p: Matrix, c: Matrix, theta: float, target: int, scaled: bool
) -> tuple[Matrix, Matrix, Matrix, Matrix, Mask]:
    h = (
        potentials(p, c, theta, target)
        if scaled
        else np.zeros(p.shape[0], dtype=np.float64)
    )
    rows, columns = np.nonzero(p)
    w = np.zeros_like(p)
    w[rows, columns] = np.exp(
        np.log(p[rows, columns]) - theta * c[rows, columns] + h[rows] - h[columns]
    )
    w[target, :] = 0.0
    keep = np.arange(p.shape[0]) != target
    a = np.eye(p.shape[0] - 1, dtype=np.float64) - w[np.ix_(keep, keep)]
    return a, w[keep, target], w, h, keep


def linear(
    p: Matrix, c: Matrix, theta: float, target: int, scaled: bool
) -> tuple[Matrix, Matrix]:
    a, b, w, h, keep = system(p, c, theta, target, scaled)
    y = np.ones(p.shape[0], dtype=np.float64)
    y[keep] = np.linalg.solve(a, b)
    moment = np.zeros_like(y)
    moment[keep] = np.linalg.solve(a, ((w * c) @ y)[keep])
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (h - np.log(y)) / theta
        cbar = moment / y
    return phi[keep], cbar[keep]


def physical_gmres(
    p: Matrix, c: Matrix, theta: float, target: int
) -> tuple[Matrix, Matrix]:
    h = theta * np.asarray(dijkstra(sparse.csr_matrix(c).T, indices=target))
    rows, columns = np.nonzero(p)
    w = np.zeros_like(p)
    w[rows, columns] = np.exp(
        np.log(p[rows, columns]) - theta * c[rows, columns] + h[rows] - h[columns]
    )
    w[target, :] = 0.0
    keep = np.arange(p.shape[0]) != target
    a = sparse.csr_matrix(np.eye(p.shape[0] - 1) - w[np.ix_(keep, keep)])
    y = np.ones(p.shape[0], dtype=np.float64)
    y[keep], _ = gmres(a, w[keep, target], rtol=1e-11, atol=0.0, restart=50, maxiter=100)
    moment, _ = gmres(
        a, ((w * c) @ y)[keep], rtol=1e-11, atol=0.0, restart=50, maxiter=100
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return (h[keep] - np.log(y[keep])) / theta, moment / y[keep]


def reference(p: Matrix, c: Matrix, theta: float, target: int) -> tuple[Matrix, Matrix]:
    """Normalize affinities exactly, avoiding a 1/theta-amplified float64 defect."""
    mp.mp.dps = 90
    n = p.shape[0]
    keep = [i for i in range(n) if i != target]
    temperature = mp.mpf(theta)
    row_sums = [mp.fsum(mp.mpf(value) for value in p[i]) for i in range(n)]
    w = mp.matrix(n)
    for i, j in zip(*np.nonzero(p), strict=True):
        if i != target:
            w[i, j] = (
                mp.mpf(p[i, j]) / row_sums[i] * mp.exp(-temperature * mp.mpf(c[i, j]))
            )
    a = mp.matrix([[mp.mpf(i == j) - w[i, j] for j in keep] for i in keep])
    z = mp.lu_solve(a, mp.matrix([w[i, target] for i in keep]))
    full = [mp.mpf(1) if i == target else z[keep.index(i)] for i in range(n)]
    rhs = mp.matrix(
        [mp.fsum(w[i, j] * mp.mpf(c[i, j]) * full[j] for j in range(n)) for i in keep]
    )
    moment = mp.lu_solve(a, rhs)
    return (
        np.asarray(
            [float(-mp.log(value) / temperature) for value in z], dtype=np.float64
        ),
        np.asarray([float(moment[i] / z[i]) for i in range(n - 1)], dtype=np.float64),
    )


def relative(actual: Matrix, expected: Matrix) -> float:
    if not np.all(np.isfinite(actual)):
        return np.inf
    return float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-14)))


def accuracy(
    names: tuple[str, ...],
    thetas: tuple[float, ...],
    routes: dict[str, tuple[Solve, Array, Array]],
) -> list[Record]:
    records: list[Record] = []
    for name in names:
        p, c = graph(name)
        target = p.shape[0] - 1
        solve, affinity, cost = routes[name]
        for theta in thetas:
            exact_phi, exact_cost = reference(p, c, theta, target)
            started = time.perf_counter()
            result = jax.block_until_ready(solve(affinity, cost, jnp.asarray(theta)))
            records.append(
                dict(
                    graph=name,
                    theta=theta,
                    route="package_default",
                    status=int(result.status),
                    target=target,
                    source_count=target,
                    phi_error=relative(
                        np.asarray(result.free_energy[:-1, -1]), exact_phi
                    ),
                    cost_error=relative(
                        np.asarray(result.expected_cost[:-1, -1]), exact_cost
                    ),
                    seconds=time.perf_counter() - started,
                )
            )
            for route in ("naive_dense", "physical_cost_gmres", "augmented_scaled"):
                started = time.perf_counter()
                phi, expected_cost = (
                    physical_gmres(p, c, theta, target)
                    if route == "physical_cost_gmres"
                    else linear(p, c, theta, target, route == "augmented_scaled")
                )
                records.append(
                    dict(
                        graph=name,
                        theta=theta,
                        route=route,
                        target=target,
                        source_count=target,
                        phi_error=relative(phi, exact_phi),
                        cost_error=relative(expected_cost, exact_cost),
                        seconds=time.perf_counter() - started,
                    )
                )
            print("accuracy", name, theta, flush=True)
    if "path50" not in names:
        # A short path does not expose normwise GMRES's lost tiny partition values.
        p, c = graph("path50")
        target = p.shape[0] - 1
        for theta in (1.0, 1000.0):
            exact_phi, exact_cost = reference(p, c, theta, target)
            started = time.perf_counter()
            phi, expected_cost = physical_gmres(p, c, theta, target)
            records.append(
                dict(
                    graph="path50",
                    theta=theta,
                    route="physical_cost_gmres",
                    target=target,
                    source_count=target,
                    phi_error=relative(phi, exact_phi),
                    cost_error=relative(expected_cost, exact_cost),
                    seconds=time.perf_counter() - started,
                )
            )
    return records


def conditioning(
    names: tuple[str, ...],
    thetas: tuple[float, ...],
    routes: dict[str, tuple[Solve, Array, Array]],
) -> list[Record]:
    records: list[Record] = []
    for name in names:
        p, c = graph(name)
        solve, affinity, cost = routes[name]
        target = p.shape[0] - 1
        keep = np.arange(p.shape[0]) != target
        rows, columns = np.nonzero(p)
        logp = np.log(p[rows, columns] / p.sum(axis=1)[rows])
        for theta in thetas:
            result = jax.block_until_ready(solve(affinity, cost, jnp.asarray(theta)))
            phi = np.asarray(result.free_energy[:, target])
            pi = np.zeros_like(p)
            pi[rows, columns] = np.exp(
                logp - theta * (c[rows, columns] + phi[columns] - phi[rows])
            )
            jacobian = np.eye(target, dtype=np.float64) - pi[np.ix_(keep, keep)]
            records.append(
                dict(
                    graph=name,
                    theta=theta,
                    route="package_default",
                    status=int(result.status),
                    condition=float(np.linalg.cond(jacobian)),
                )
            )
            old = system(p, c, theta, target, True)[0]
            records.append(
                dict(
                    graph=name,
                    theta=theta,
                    route="augmented_scaled",
                    condition=float(np.linalg.cond(old)),
                )
            )
    return records


def limits(
    names: tuple[str, ...], routes: dict[str, tuple[Solve, Array, Array]]
) -> list[Record]:
    records: list[Record] = []
    for name in names:
        p, c = graph(name)
        target = p.shape[0] - 1
        keep = np.arange(p.shape[0]) != target
        shortest = np.asarray(dijkstra(sparse.csr_matrix(c).T, indices=target))[keep]
        hitting = np.linalg.solve(
            np.eye(target, dtype=np.float64) - p[np.ix_(keep, keep)],
            np.sum(p * c, axis=1)[keep],
        )
        solve, affinity, cost = routes[name]
        for theta in (1e-9, 1e5):
            result = jax.block_until_ready(solve(affinity, cost, jnp.asarray(theta)))
            records.append(
                dict(
                    graph=name,
                    theta=theta,
                    status=int(result.status),
                    phi_shortest_error=relative(
                        np.asarray(result.free_energy[:-1, -1]), shortest
                    ),
                    cost_hitting_error=relative(
                        np.asarray(result.expected_cost[:-1, -1]), hitting
                    ),
                )
            )
    for name in names[:3]:
        p, original_cost = graph(name)
        n = p.shape[0]
        stationary_system = np.eye(n, dtype=np.float64) - p.T
        stationary_system[-1, :] = 1.0
        rhs = np.zeros(n, dtype=np.float64)
        rhs[-1] = 1.0
        stationary = np.linalg.solve(stationary_system, rhs)
        adjacency = stationary[:, None] * p
        degree = adjacency.sum(axis=1)
        inverse = np.linalg.pinv(np.diag(degree) - adjacency, hermitian=True)
        resistance = inverse[0, 0] + inverse[-1, -1] - 2 * inverse[0, -1]
        solve, affinity, _ = routes[name]
        rows, columns = np.nonzero(p)
        for label, costs in (
            ("unit", (p > 0).astype(np.float64)),
            ("general", original_cost),
        ):
            result = jax.block_until_ready(
                solve(affinity, jnp.asarray(costs[rows, columns]), jnp.asarray(1e-9))
            )
            commute = float(result.expected_cost[0, -1] + result.expected_cost[-1, 0])
            expected = float(np.sum(adjacency * costs) * resistance)
            records.append(
                dict(
                    graph=name,
                    costs=label,
                    theta=1e-9,
                    status=int(result.status),
                    commute_cost=commute,
                    cost_volume_resistance=expected,
                    commute_error=abs(commute / expected - 1),
                )
            )
    return records


def grid_log_reference(side: int, theta: float) -> tuple[Matrix, Matrix, int]:
    degree = np.full((side, side), 4.0, dtype=np.float64)
    degree[[0, -1], :] -= 1.0
    degree[:, [0, -1]] -= 1.0
    base = -theta - np.log(degree)
    ell = np.zeros((side, side), dtype=np.float64)
    cost = np.zeros_like(ell)
    neighbors = np.full((4, side, side), -np.inf, dtype=np.float64)
    neighbor_cost = np.zeros_like(neighbors)
    for iteration in range(1, 5001):
        neighbors[0, 1:, :] = ell[:-1, :]
        neighbors[1, :-1, :] = ell[1:, :]
        neighbors[2, :, 1:] = ell[:, :-1]
        neighbors[3, :, :-1] = ell[:, 1:]
        neighbor_cost[0, 1:, :] = cost[:-1, :]
        neighbor_cost[1, :-1, :] = cost[1:, :]
        neighbor_cost[2, :, 1:] = cost[:, :-1]
        neighbor_cost[3, :, :-1] = cost[:, 1:]
        update = base + logsumexp(neighbors, axis=0)
        weights = np.exp(neighbors + base[None, :, :] - update[None, :, :])
        cost_update = np.sum(weights * (1.0 + neighbor_cost), axis=0)
        update[-1, -1] = cost_update[-1, -1] = 0.0
        log_delta = np.max(np.abs(update - ell))
        cost_delta = np.max(
            np.abs(cost_update - cost) / np.maximum(1.0, np.abs(cost_update))
        )
        ell, cost = update, cost_update
        if log_delta <= 1e-11 and cost_delta <= 1e-11:
            break
    if log_delta > 1e-11 or cost_delta > 1e-11:
        raise RuntimeError("Independent grid log reference did not converge.")
    return -ell.ravel()[:-1] / theta, cost.ravel()[:-1], iteration


def grid_reference(side: int, theta: float) -> tuple[Matrix, Matrix, int]:
    if theta > 1e-3:
        return grid_log_reference(side, theta)
    adjacency = grid(side)
    n = side * side
    p = (
        sparse.diags(1.0 / np.asarray(adjacency.sum(axis=1)).ravel()) @ adjacency
    ).tocsr()
    w = p * np.exp(-theta)
    a = sparse.eye(n - 1, format="csc", dtype=np.float64) - w[:-1, :-1].tocsc()
    factor = splu(a)
    z = np.ones(n, dtype=np.float64)
    z[:-1] = factor.solve(w[:-1, -1].toarray().ravel())
    moment = factor.solve((w @ z)[:-1])
    return -np.log(z[:-1]) / theta, moment / z[:-1], 0


def scaling(sizes: tuple[int, ...], *, cpu_provider: bool = False) -> list[Record]:
    records: list[Record] = []
    policy = (
        phx.linalg.LinearSolvePolicy(
            phx.linalg.SparseLU(provider="jax-cpu"),
            # Opt-in route: SuperLU's ordering is provider-chosen, so the planner
            # charges dense bounds to both factor and preparation budgets.
            resources=phx.linalg.SolveResourcePolicy(
                factorization_bytes=CPU_FACTORIZATION_BYTES,
                workspace_bytes=CPU_FACTORIZATION_BYTES,
            ),
        )
        if cpu_provider
        else None
    )
    for side in sizes:
        adjacency = grid(side).tocoo()
        n = side * side
        relation = phx.sparse.EdgeRelation(
            np.asarray(adjacency.row, dtype=np.int32),
            np.asarray(adjacency.col, dtype=np.int32),
            source_size=n,
            target_size=n,
        )
        started = time.perf_counter()
        plan = phx.graph.RandomizedShortestPathPlan(
            relation, np.asarray([0, n - 1], dtype=np.int32), linear_policy=policy
        )
        preparation_seconds = time.perf_counter() - started
        affinity = jnp.ones(adjacency.nnz, dtype=jnp.float64)
        cost = jnp.ones_like(affinity)
        started = time.perf_counter()
        lowered = jax.jit(route_callable(plan)).lower(affinity, cost, jnp.asarray(1.0))
        lowering_seconds = time.perf_counter() - started
        started = time.perf_counter()
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - started
        for theta in SCALING_THETAS:
            temperature = jnp.asarray(theta)
            jax.block_until_ready(compiled(affinity, cost, temperature))
            started = time.perf_counter()
            result = jax.block_until_ready(compiled(affinity, cost, temperature))
            warm_seconds = time.perf_counter() - started
            reference_started = time.perf_counter()
            phi, expected_cost, iterations = grid_reference(side, theta)
            records.append(
                dict(
                    route="package_jax_cpu" if cpu_provider else "package_default",
                    memory_scope="provider-managed; not bounded by Phydrax planning"
                    if cpu_provider
                    else "Phydrax planned",
                    factorization_bytes=CPU_FACTORIZATION_BYTES if cpu_provider else 0,
                    side=side,
                    nodes=n,
                    query_count=plan.query_count,
                    theta=theta,
                    status=int(result.status),
                    preparation_seconds=preparation_seconds,
                    lowering_seconds=lowering_seconds,
                    compile_seconds=compile_seconds,
                    warm_seconds=warm_seconds,
                    warm_per_target_seconds=warm_seconds / plan.query_count,
                    phi_reference_error=relative(
                        np.asarray([result.free_energy[0, 1], result.free_energy[1, 0]]),
                        np.full(2, phi[0], dtype=np.float64),
                    ),
                    cost_reference_error=relative(
                        np.asarray(
                            [result.expected_cost[0, 1], result.expected_cost[1, 0]]
                        ),
                        np.full(2, expected_cost[0], dtype=np.float64),
                    ),
                    reference_iterations=iterations,
                    reference_seconds=time.perf_counter() - reference_started,
                    budgeted_workspace_bytes=plan.resource_estimate[
                        "budgeted_workspace_bytes"
                    ],
                )
            )
            print("scaling", records[-1], flush=True)
    return records


def dense_flows(p: Matrix, c: Matrix, theta: float, target: int) -> Matrix:
    a, _, w, _, keep = system(p, c, theta, target, True)
    y = np.ones(p.shape[0], dtype=np.float64)
    y[keep] = np.linalg.solve(a, w[keep, target])
    source = np.zeros(p.shape[0] - 1, dtype=np.float64)
    source[0] = 1.0
    visits = np.zeros(p.shape[0], dtype=np.float64)
    visits[keep] = np.linalg.solve(a.T, source)
    return visits[:, None] * w * y[None, :] / y[0]


def functional_gradient(
    solve: Solve,
    affinity: Array,
    cost: Array,
    theta: Array,
    *,
    expected_cost: bool,
    affinities: bool,
) -> tuple[float, float]:
    """Check every supported edge using central differences of the same functional."""
    weights = jnp.arange(1, 9, dtype=jnp.float64)

    def objective(value: Array) -> Array:
        result = solve(
            value if affinities else affinity, cost if affinities else value, theta
        )
        observable = result.expected_cost if expected_cost else result.free_energy
        return jnp.sum(weights[:-1] * observable[:-1, -1])

    value = affinity if affinities else cost
    gradient = np.asarray(jax.grad(objective)(value))
    finite_difference = np.empty_like(gradient)
    for edge in range(value.size):
        step = 1e-4 * float(value[edge])
        bump = jnp.zeros_like(value).at[edge].set(step)
        finite_difference[edge] = float(
            (objective(value + bump) - objective(value - bump)) / (2 * step)
        )
    absolute_error = float(np.max(np.abs(gradient - finite_difference)))
    # Include all routes, including the target's exactly zero outgoing derivatives.
    scale = max(float(np.max(np.abs(finite_difference))), 1e-14)
    return absolute_error / scale, absolute_error


def differentiation(
    thetas: tuple[float, ...],
    *,
    linear_policy: phx.linalg.LinearSolvePolicy | None = None,
) -> list[Record]:
    p, c = graph("directed8")
    solve, affinity, cost = package_route(p, c, linear_policy=linear_policy)
    rows, columns = np.nonzero(p)
    records: list[Record] = []
    for theta in thetas:
        temperature = jnp.asarray(theta)

        def energy_at_temperature(value: Array) -> Array:
            return value * solve(affinity, cost, value).free_energy[:-1, -1]

        _, tangent = jax.jvp(energy_at_temperature, (temperature,), (jnp.asarray(1.0),))
        result = jax.block_until_ready(solve(affinity, cost, temperature))
        record: Record = dict(
            graph="directed8",
            theta=theta,
            status=int(result.status),
            route="package_default"
            if linear_policy is None
            else "package_jax_cpu_oracle_check",
            jvp_analytic_error=relative(
                np.asarray(tangent), np.asarray(result.expected_cost[:-1, -1])
            ),
        )
        for observable, expected_cost in (("phi", False), ("cost", True)):
            for parameter, affinities in (("affinities", True), ("costs", False)):
                error, absolute = functional_gradient(
                    solve,
                    affinity,
                    cost,
                    temperature,
                    expected_cost=expected_cost,
                    affinities=affinities,
                )
                record[f"{observable}_{parameter}_fd_error"] = error
                record[f"{observable}_{parameter}_fd_absolute_error"] = absolute

        def source_energy(value: Array) -> Array:
            return solve(affinity, value, temperature).free_energy[0, -1]

        _, pullback = jax.vjp(source_energy, cost)
        flows = np.asarray(pullback(jnp.asarray(1.0))[0])
        expected = dense_flows(p, c, theta, 7)[rows, columns]
        flow_absolute_error = float(np.max(np.abs(flows - expected)))
        record["vjp_flow_absolute_error"] = flow_absolute_error
        record["vjp_flow_norm_error"] = flow_absolute_error / max(
            float(np.max(np.abs(expected))), 1e-14
        )
        records.append(record)
        print("differentiation", record, flush=True)
    return records


def finite_json(value: Any) -> Any:
    """Own JSON interchange: nonfinite scientific evidence is null, not success."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in sorted(value.items())}
    return value


def qualify(
    accuracy_records: list[Record],
    conditioning_records: list[Record],
    limit_records: list[Record],
    scaling_records: list[Record],
    derivative_records: list[Record],
) -> dict[str, Any]:
    for record in accuracy_records:
        passed = (
            float(record["phi_error"]) <= 1e-8 and float(record["cost_error"]) <= 1e-8
        )
        if record["route"] == "package_default":
            passed = passed and record["status"] == 0
        record["verdict"] = "PASS" if passed else "FAIL"
    for record in conditioning_records:
        passed = float(record["condition"]) < 1e12
        if record["route"] == "package_default":
            passed = passed and record["status"] == 0
        record["verdict"] = "PASS" if passed else "FAIL"
    for record in limit_records:
        if "commute_error" in record:
            passed = float(record["commute_error"]) <= 1e-5
        elif record["theta"] == 1e5:
            passed = float(record["phi_shortest_error"]) <= 1e-4
        else:
            passed = float(record["cost_hitting_error"]) <= 1e-5
        record["verdict"] = "PASS" if passed and record["status"] == 0 else "FAIL"
    for record in scaling_records:
        if record["route"] != "package_default":
            record["role"] = "informational_provider_managed_memory"
            continue
        passed = (
            record["status"] == 0
            and float(record["phi_reference_error"]) <= 1e-7
            and float(record["cost_reference_error"]) <= 1e-7
            and float(record["warm_per_target_seconds"]) < 30.0
        )
        record["verdict"] = "PASS" if passed else "FAIL"
    for record in derivative_records:
        passed = (
            record["status"] == 0
            and float(record["jvp_analytic_error"]) <= 1e-10
            and float(record["vjp_flow_absolute_error"]) <= 1e-10
            and float(record["vjp_flow_norm_error"]) <= 1e-10
            and all(
                float(value) <= 1e-6
                for key, value in record.items()
                if key.endswith("_fd_error")
            )
        )
        record["verdict"] = "PASS" if passed else "FAIL"
    gates: dict[str, Any] = {
        "G1_accuracy": dict(
            criterion="Package phi and expected-cost componentwise relative errors <= 1e-8 against exactly normalized 90-digit mpmath; status SUCCESS.",
            records=accuracy_records,
        ),
        "G2_conditioning": dict(
            criterion="Package Newton Jacobian cond2 < 1e12 and status SUCCESS at every strictly positive theta.",
            records=conditioning_records,
        ),
        "G3_limits": dict(
            criterion="theta=1e5 phi/shortest error <= 1e-4; theta=1e-9 cost/hitting and commute-identity errors <= 1e-5; status SUCCESS.",
            hypotheses="Almost-sure absorption; reversible commute cost equals cost-volume times resistance.",
            records=limit_records,
        ),
        "G4_scaling": dict(
            criterion="Default package status SUCCESS, independent reference errors <= 1e-7, warm per-target solve time < 30 seconds.",
            records=scaling_records,
        ),
        "G5_differentiation": dict(
            criterion="Default package JVP theta(theta*phi) vs expected cost <= 1e-10 componentwise; phi and expected-cost gradients wrt affinities/costs vs central FD <= 1e-6 relative infinity norm over every route; VJP edge-flow absolute/norm errors <= 1e-10.",
            records=derivative_records,
        ),
    }
    for name, gate in gates.items():
        candidates = [
            record
            for record in gate["records"]
            if "verdict" in record
            and (
                name not in ("G1_accuracy", "G2_conditioning")
                or record["route"] == "package_default"
            )
        ]
        if not candidates:
            raise RuntimeError(f"Gate has no evaluated candidates: {name}")
        gate["status"] = (
            "PASS"
            if all(record["verdict"] == "PASS" for record in candidates)
            else "FAIL"
        )
        gate["evaluated_cases"] = len(candidates)
        gate["failed_cases"] = sum(record["verdict"] == "FAIL" for record in candidates)
    controls: dict[str, Any] = {}
    for route in ("naive_dense", "physical_cost_gmres", "augmented_scaled"):
        candidates = [
            record
            for record in accuracy_records + conditioning_records
            if record["route"] == route
        ]
        failures = [record for record in candidates if record["verdict"] == "FAIL"]
        controls[route] = dict(
            expected="FAIL somewhere",
            observed="FAIL" if failures else "PASS",
            fault_adequacy="PASS" if failures else "FAIL",
            evaluated_cases=len(candidates),
            failed_cases=len(failures),
        )
    return dict(
        gates=gates,
        negative_controls=controls,
        fault_adequacy="PASS"
        if all(control["fault_adequacy"] == "PASS" for control in controls.values())
        else "FAIL",
    )


def configuration(smoke: bool) -> dict[str, Any]:
    return dict(
        mode="smoke" if smoke else "full",
        seed=731,
        driver=str(Path(__file__).resolve()),
        cwd=str(Path.cwd()),
        pythonpath=os.environ.get("PYTHONPATH"),
        import_path=list(sys.path),
        backend=jax.default_backend(),
        dtype="float64",
        repeats=1,
        warmups=1,
        scaling_queries="first and last grid nodes",
        graph_queries="all nodes; last node is absorbing reference target",
        policies=["package_default", "package_jax_cpu"],
        providers=["jax", "scipy-superlu", "mpmath"],
        package_resource_policy="default RandomizedShortestPathPlan",
        informational_cpu_workspace_bytes=CPU_FACTORIZATION_BYTES,
        graph_names=list(SMOKE_GRAPH_NAMES if smoke else GRAPH_NAMES),
        accuracy_thetas=list((1e-9, 1e-3, 1.0, 1000.0) if smoke else THETAS),
        retained_physical_gmres_counterexample_graph="path50",
        retained_physical_gmres_counterexample_thetas=[1.0, 1000.0],
        conditioning_graph_names=["path50"] if smoke else list(GRAPH_NAMES),
        conditioning_thetas=list(
            (1e-9, 1e-6, 1e-3, 1.0, 1000.0) if smoke else (1e-9, 1e-6, *THETAS[1:])
        ),
        limit_thetas=[1e-9, 1e5],
        grid_sides=[8, 16] if smoke else [32, 48, 64],
        informational_cpu_grid_sides=[] if smoke else [128, 256],
        informational_cpu_memory_scope="Provider-managed memory, not bounded by Phydrax planning",
        informational_cpu_factorization_bytes=CPU_FACTORIZATION_BYTES,
        scaling_thetas=list(SCALING_THETAS),
        mpmath_decimal_digits=90,
        reference_row_normalization="Exact mpmath normalization of the same affinities supplied to the package",
        gmres=dict(restart=50, maxiter=100, rtol=1e-11, atol=0.0),
        differentiation_graph="directed8",
        differentiation_thetas=[0.1, 1.0, 10.0],
        central_difference_relative_step=1e-4,
        gradient_fd_error_norm="infinity norm of AD-FD divided by infinity norm of FD; every supported route",
        grid_log_reference_max_sweeps=5000,
        timing="Separate preparation/lowering/compilation and warmed solve; two absorbing query targets, warm time divided by query_count.",
        nonfinite_json_representation="null means an invalid numerical result, never a successful fallback",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/rsp_free_energy_gate.json")
    )
    args = parser.parse_args()
    driver = Path(__file__).resolve()
    identity = capture_benchmark_identity(driver.parents[1], driver, EVIDENCE_FIELDS)
    config = configuration(args.smoke)
    config["output"] = str(args.output.resolve())
    names = SMOKE_GRAPH_NAMES if args.smoke else GRAPH_NAMES
    thetas = (1e-9, 1e-3, 1.0, 1000.0) if args.smoke else THETAS
    condition_names = ("path50",) if args.smoke else GRAPH_NAMES
    condition_thetas = (
        (1e-9, 1e-6, 1e-3, 1.0, 1000.0) if args.smoke else (1e-9, 1e-6, *THETAS[1:])
    )
    routes = {
        name: package_route(*graph(name))
        for name in dict.fromkeys((*names, *condition_names))
    }
    results = qualify(
        accuracy(names, thetas, routes),
        conditioning(condition_names, condition_thetas, routes),
        limits(names, routes),
        scaling((8, 16) if args.smoke else (32, 48, 64))
        + ([] if args.smoke else scaling((128, 256), cpu_provider=True)),
        differentiation((0.1, 1.0, 10.0)),
    )
    environment = capture_environment().to_dict()
    environment["scipy_version"] = scipy.__version__
    environment["mpmath_version"] = mp.__version__
    passed = all(gate["status"] == "PASS" for gate in results["gates"].values())
    report = finite_json(
        dict(
            environment=environment,
            command=[sys.executable, *sys.argv],
            configuration=config,
            provenance=dict(
                kind="recomputed", driver="benchmarks/rsp_free_energy_gate.py"
            ),
            **results,
            overall="PASSED" if passed else "FAILED",
            rsp_api_shipped=True,
        )
    )
    report["identity"] = identity.to_dict()
    write_json_atomic(args.output, report)
    print(
        f"RSP qualification {report['overall']}; fault adequacy {results['fault_adequacy']}; written to {args.output}."
    )
    if not passed or results["fault_adequacy"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
