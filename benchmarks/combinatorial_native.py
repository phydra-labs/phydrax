#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _measure(function, argument, repeats):
    start = time.perf_counter()
    first = function(argument)
    jax.block_until_ready(first)
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = first
    for _ in range(repeats):
        result = function(argument)
        jax.block_until_ready(result)
    steady_seconds = (time.perf_counter() - start) / repeats
    return first_seconds, steady_seconds


def _layered_dag(layers, width):
    source = 0
    target = 1 + layers * width
    senders: list[int] = []
    receivers: list[int] = []
    for node in range(width):
        senders.append(source)
        receivers.append(1 + node)
    for layer in range(layers - 1):
        left = 1 + layer * width
        right = left + width
        for source_slot in range(width):
            for target_slot in range(width):
                senders.append(left + source_slot)
                receivers.append(right + target_slot)
    final = 1 + (layers - 1) * width
    for node in range(width):
        senders.append(final + node)
        receivers.append(target)
    relation = phx.sparse.EdgeRelation(
        jnp.asarray(senders, dtype=jnp.int32),
        jnp.asarray(receivers, dtype=jnp.int32),
        source_size=target + 1,
        target_size=target + 1,
    )
    return phx.combinatorial.ShortestPathSpace(relation, source, target)


def run(*, cardinality_size, assignment_size, dag_layers, dag_width, repeats):
    key = jax.random.key(17)
    cardinality_key, assignment_key, dag_key = jax.random.split(key, 3)

    cardinality_space = phx.combinatorial.CardinalitySpace(
        cardinality_size,
        max(1, cardinality_size // 10),
    )
    cardinality_problem = phx.combinatorial.LinearCombinatorialProblem(
        cardinality_space,
        jax.random.normal(cardinality_key, (cardinality_size,)),
    )
    cardinality_method = phx.combinatorial.StableCardinalityOracle()
    cardinality_solve = jax.jit(
        lambda problem: (
            phx.combinatorial.solve_combinatorial(
                problem,
                cardinality_method,
            ).objective_value
        )
    )
    cardinality_first, cardinality_steady = _measure(
        cardinality_solve,
        cardinality_problem,
        repeats,
    )

    assignment_space = phx.combinatorial.BipartiteAssignmentSpace(
        assignment_size,
        assignment_size,
    )
    assignment_problem = phx.combinatorial.LinearCombinatorialProblem(
        assignment_space,
        jax.random.normal(
            assignment_key,
            (assignment_size, assignment_size),
        ),
    )
    assignment_method = phx.combinatorial.HungarianAssignment()
    assignment_solve = jax.jit(
        lambda problem: (
            phx.combinatorial.solve_combinatorial(
                problem,
                assignment_method,
            ).objective_value
        )
    )
    assignment_first, assignment_steady = _measure(
        assignment_solve,
        assignment_problem,
        repeats,
    )

    dag_space = _layered_dag(dag_layers, dag_width)
    dag_costs = jax.random.normal(dag_key, (dag_space.edge_count,))
    dag_problem = phx.combinatorial.LinearCombinatorialProblem(dag_space, dag_costs)
    dag_method = phx.combinatorial.DAGShortestPath()
    dag_solve = jax.jit(
        lambda problem: (
            phx.combinatorial.solve_combinatorial(
                problem,
                dag_method,
            ).objective_value
        )
    )
    dag_first, dag_steady = _measure(dag_solve, dag_problem, repeats)

    interpolation = phx.combinatorial.BlackboxInterpolation(1.0)
    target = jnp.zeros((dag_space.edge_count,), dtype=dag_costs.dtype)

    def loss(costs):
        problem = phx.combinatorial.LinearCombinatorialProblem(dag_space, costs)
        features = phx.combinatorial.blackbox_solution(
            problem,
            dag_method,
            policy=interpolation,
        )
        return 0.5 * jnp.sum((features - target) ** 2)

    dag_gradient = jax.jit(jax.grad(loss))
    blackbox_first, blackbox_steady = _measure(dag_gradient, dag_costs, repeats)

    return {
        "platform": jax.default_backend(),
        "repeats": repeats,
        "cardinality": {
            "items": cardinality_size,
            "selected": cardinality_space.count,
            "compile_and_first_seconds": cardinality_first,
            "steady_seconds": cardinality_steady,
        },
        "assignment": {
            "rows": assignment_size,
            "columns": assignment_size,
            "compile_and_first_seconds": assignment_first,
            "steady_seconds": assignment_steady,
        },
        "dag_shortest_path": {
            "vertices": dag_space.vertex_count,
            "edges": dag_space.edge_count,
            "incoming_width": dag_space.incoming_width,
            "compile_and_first_seconds": dag_first,
            "steady_seconds": dag_steady,
        },
        "dag_blackbox_pullback": {
            "compile_and_first_seconds": blackbox_first,
            "steady_seconds": blackbox_steady,
            "backward_to_forward_ratio": blackbox_steady / dag_steady,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark native combinatorial methods."
    )
    parser.add_argument("--cardinality-size", type=int, default=10_000)
    parser.add_argument("--assignment-size", type=int, default=32)
    parser.add_argument("--dag-layers", type=int, default=16)
    parser.add_argument("--dag-width", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    if (
        min(
            arguments.cardinality_size,
            arguments.assignment_size,
            arguments.dag_layers,
            arguments.dag_width,
            arguments.repeats,
        )
        <= 0
    ):
        raise ValueError("benchmark dimensions and repeats must be positive.")
    print(
        json.dumps(
            run(
                cardinality_size=arguments.cardinality_size,
                assignment_size=arguments.assignment_size,
                dag_layers=arguments.dag_layers,
                dag_width=arguments.dag_width,
                repeats=arguments.repeats,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
