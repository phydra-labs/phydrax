#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _measure(function, *arguments, repeats):
    start = time.perf_counter()
    first = function(*arguments)
    jax.block_until_ready(first)
    first_seconds = time.perf_counter() - start

    start = time.perf_counter()
    result = first
    for _ in range(repeats):
        result = function(*arguments)
        jax.block_until_ready(result)
    steady_seconds = (time.perf_counter() - start) / repeats
    return first_seconds, steady_seconds


def _grid_edges(side):
    senders = []
    receivers = []
    for row in range(side):
        for column in range(side):
            node = row * side + column
            if row + 1 < side:
                senders.append(node)
                receivers.append((row + 1) * side + column)
            if column + 1 < side:
                senders.append(node)
                receivers.append(row * side + column + 1)
    return jnp.stack([jnp.asarray(senders), jnp.asarray(receivers)], axis=-1)


def run(*, side, bp_steps, chains, repeats):
    variables = side * side
    edges = _grid_edges(side)
    graph = phx.pgm.ising_factor_graph(
        jnp.linspace(-0.1, 0.1, variables),
        edges,
        jnp.full((int(edges.shape[0]),), 0.15),
        shape=(side, side),
    )

    prepare_start = time.perf_counter()
    bp_plan = phx.pgm.prepare_belief_propagation(
        graph,
        phx.pgm.SumProductBeliefPropagation(
            maximum_steps=bp_steps,
            relaxation=0.7,
        ),
    )
    bp_prepare_seconds = time.perf_counter() - prepare_start
    bp_state = phx.pgm.initialize_belief_propagation(bp_plan)
    bp_run = jax.jit(lambda state: phx.pgm.run_belief_propagation(bp_plan, state))
    bp_first, bp_steady = _measure(bp_run, bp_state, repeats=repeats)
    batch_cases = min(chains, 16)
    batch_state = phx.pgm.BatchedBeliefPropagationState(
        jnp.broadcast_to(bp_state.messages, (batch_cases, bp_plan.message_count)),
        jnp.broadcast_to(
            bp_state.evidence.values,
            (batch_cases, graph.num_variable_states),
        ),
        structure_id=graph.structure_id,
    )
    batch_run = jax.jit(lambda state: phx.pgm.batch_belief_propagation(bp_plan, state))
    batch_first, batch_steady = _measure(
        batch_run,
        batch_state,
        repeats=repeats,
    )

    prepare_start = time.perf_counter()
    gibbs_plan = phx.pgm.prepare_chromatic_gibbs(graph)
    gibbs_prepare_seconds = time.perf_counter() - prepare_start
    initial = jr.bernoulli(jr.key(5), 0.5, (chains, variables)).astype(jnp.int32)
    gibbs_state = phx.pgm.initialize_gibbs(gibbs_plan, initial)
    gibbs_run = jax.jit(lambda state, key: phx.pgm.gibbs_sweep(gibbs_plan, state, key)[0])
    gibbs_first, gibbs_steady = _measure(
        gibbs_run,
        gibbs_state,
        jr.key(6),
        repeats=repeats,
    )
    random_policy = phx.pgm.GibbsScanPolicy(
        "random-scan",
        updates_per_sweep=variables,
    )
    random_gibbs_run = jax.jit(
        lambda state, key: phx.pgm.gibbs_sweep_with_policy(
            gibbs_plan,
            state,
            key,
            random_policy,
        )[0]
    )
    random_first, random_steady = _measure(
        random_gibbs_run,
        gibbs_state,
        jr.key(7),
        repeats=repeats,
    )

    elimination_start = time.perf_counter()
    elimination_plan = phx.pgm.plan_variable_elimination(
        graph,
        resources=phx.pgm.FactorGraphResourcePolicy(
            maximum_treewidth=max(side, 1),
            maximum_elimination_elements=max(2 ** (side + 1), 2),
        ),
    )
    elimination_prepare_seconds = time.perf_counter() - elimination_start
    elimination_first, elimination_steady = _measure(
        phx.pgm.variable_elimination,
        elimination_plan,
        repeats=repeats,
    )

    packed = phx.pgm.pack_factor_graphs((graph, graph))

    return {
        "platform": jax.default_backend(),
        "side": side,
        "variables": variables,
        "factors": graph.num_factors,
        "incidences": int(graph.topology.incidence_edges.shape[0]),
        "bp": {
            "messages": bp_plan.message_count,
            "prepare_seconds": bp_prepare_seconds,
            "compile_and_first_seconds": bp_first,
            "steady_seconds": bp_steady,
            "maximum_steps": bp_steps,
        },
        "batch_bp": {
            "cases": batch_cases,
            "compile_and_first_seconds": batch_first,
            "steady_seconds": batch_steady,
        },
        "gibbs": {
            "chains": chains,
            "colors": len(gibbs_plan.stages),
            "prepare_seconds": gibbs_prepare_seconds,
            "compile_and_first_seconds": gibbs_first,
            "steady_sweep_seconds": gibbs_steady,
            "random_scan_compile_and_first_seconds": random_first,
            "random_scan_steady_sweep_seconds": random_steady,
        },
        "elimination": {
            "treewidth": elimination_plan.treewidth,
            "maximum_workspace_elements": elimination_plan.maximum_workspace_elements,
            "prepare_seconds": elimination_prepare_seconds,
            "first_seconds": elimination_first,
            "steady_seconds": elimination_steady,
        },
        "packed_graphs": {
            "graphs": packed.num_graphs,
            "nodes": packed.topology.num_nodes,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark native factor-graph inference."
    )
    parser.add_argument("--side", type=int, default=4)
    parser.add_argument("--bp-steps", type=int, default=20)
    parser.add_argument("--chains", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if min(arguments.side, arguments.bp_steps, arguments.chains, arguments.repeats) < 1:
        raise ValueError("Benchmark sizes and repeats must be positive.")
    print(
        json.dumps(
            run(
                side=arguments.side,
                bp_steps=arguments.bp_steps,
                chains=arguments.chains,
                repeats=arguments.repeats,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
