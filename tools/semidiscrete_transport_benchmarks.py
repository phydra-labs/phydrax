#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed-realization semidiscrete transport and gradients."
    )
    parser.add_argument("--orders", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--atoms", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _problem(order: int, atoms: int):
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    source = phx.integration.normalized_density(
        phx.integration.over(domain.component()),
        domain.Function("x")(lambda x: jnp.zeros_like(x)),
    )
    realization = phx.integration.materialize(
        source,
        phx.integration.FixedQuadraturePlan(
            phx.integration.GaussLegendreRule(order)
        ),
    )
    support = (jnp.arange(atoms, dtype=float) + 0.5) / atoms
    target = phx.integration.discrete(
        support,
        cx.Field(jnp.full((atoms,), 1.0 / atoms), dims=("atom",)),
        axes="atom",
        normalized=True,
        provenance="semidiscrete-benchmark-support",
    )
    return phx.transport.semidiscrete_problem(
        source,
        realization,
        target,
        cost=phx.transport.SquaredEuclideanCost(),
    )


def _bytes(tree) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _record(order: int, atoms: int, iterations: int, repeats: int):
    problem = _problem(order, atoms)
    solver = phx.transport.SemidiscreteSinkhorn(
        0.05,
        max_iterations=iterations,
        tolerance=1e-6,
        check_every=1,
        early_stop=False,
    )
    compiled = eqx.filter_jit(lambda candidate: solver(candidate))
    started = time.perf_counter()
    result = compiled(problem)
    jax.block_until_ready(result.regularized_cost)
    compile_first_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        replay = compiled(problem)
        jax.block_until_ready(replay.regularized_cost)
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats

    def objective(support):
        return solver(problem.with_target_support(support)).regularized_cost

    differentiated = eqx.filter_jit(jax.grad(objective))
    gradient = differentiated(problem.target_support)
    jax.block_until_ready(gradient)
    started = time.perf_counter()
    gradient = differentiated(problem.target_support)
    jax.block_until_ready(gradient)
    backward_ms = 1e3 * (time.perf_counter() - started)

    replay = compiled(problem)
    jax.block_until_ready(replay.regularized_cost)
    error_available = bool(
        result.integration_diagnostics.objective_error_available
    )
    return {
        "integration_order": order,
        "integration_evaluations": int(
            result.integration_diagnostics.objective_num_evaluations
        ),
        "integration_method": result.provenance.integration_method,
        "integration_realization": result.provenance.realization,
        "integration_status": int(result.integration_status),
        "integration_error_available": error_available,
        "integration_error_estimate": (
            float(result.integration_error_estimate)
            if error_available
            else None
        ),
        "atoms": atoms,
        "iterations": int(result.diagnostics.num_iterations),
        "transport_status": int(result.diagnostics.status),
        "normalized_target_marginal_residual": float(
            result.diagnostics.normalized_target_marginal_residual
        ),
        "dual_residual": float(result.diagnostics.dual_residual),
        "primal_dual_gap": float(result.diagnostics.primal_dual_gap),
        "regularized_cost": float(result.regularized_cost),
        "approximate": result.approximate,
        "deterministic_replay": bool(
            jnp.array_equal(result.regularized_cost, replay.regularized_cost)
            & jnp.array_equal(result.target_potential, replay.target_potential)
        ),
        "compile_first_ms": compile_first_ms,
        "steady_ms": steady_ms,
        "backward_ms": backward_ms,
        "gradient_norm": float(jnp.linalg.norm(gradient)),
        "result_bytes": _bytes(result),
    }


def main() -> None:
    arguments = _parser().parse_args()
    orders = (8, 16) if arguments.smoke else tuple(arguments.orders)
    atoms = min(arguments.atoms, 4) if arguments.smoke else int(arguments.atoms)
    iterations = min(arguments.iterations, 30) if arguments.smoke else int(arguments.iterations)
    repeats = 1 if arguments.smoke else int(arguments.repeats)
    records = [
        _record(order, atoms, iterations, repeats)
        for order in orders
    ]
    print(json.dumps({"records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
