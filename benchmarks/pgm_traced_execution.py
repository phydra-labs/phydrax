#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compiled cost of closed-over versus dynamic-argument PGM execution."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)


OUTPUT = Path(__file__).with_suffix(".json")


def _ring_graph(size: int, /) -> phx.pgm.DiscreteFactorGraph:
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(size,), num_states=2)
    edges = jnp.stack(
        (jnp.arange(size, dtype=jnp.int32), jnp.roll(jnp.arange(size), -1)), axis=-1
    )
    factor = phx.pgm.DenseTableFactorGroup(
        (
            phx.pgm.VariableSelection(variables, edges[:, 0]),
            phx.pgm.VariableSelection(variables, edges[:, 1]),
        ),
        jnp.broadcast_to(jnp.asarray([[0.2, -0.1], [-0.1, 0.2]]), (size, 2, 2)),
    )
    return phx.pgm.DiscreteFactorGraph((variables,), (factor,))


def _compiler_record(compiled: object, /) -> dict[str, object]:
    executable = compiled.compiled
    evidence = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Backend did not provide compiler estimates.",
    )
    return asdict(evidence)


def _measure(function, arguments: tuple[object, ...], repeats: int, /):
    compiled, compilation = measure_lower_and_compile(
        lambda: eqx.filter_jit(function).lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(*arguments), warmup=1, repeats=repeats
    )
    return result, {
        "compilation": asdict(compilation),
        "steady_ms": execution.to_milliseconds_dict(),
        "compiler": _compiler_record(compiled),
        "logical_input_bytes": logical_array_bytes(arguments),
        "logical_output_bytes": logical_array_bytes(result),
    }


def _case(size: int, repeats: int, /) -> dict[str, object]:
    graph = _ring_graph(size)
    bp = phx.pgm.prepare_belief_propagation(
        graph,
        phx.pgm.SumProductBeliefPropagation(maximum_steps=3, relaxation=0.7),
    )
    bp_state = phx.pgm.initialize_belief_propagation(bp)
    closed_bp, closed_bp_record = _measure(
        lambda state: phx.pgm.run_belief_propagation(bp, state),
        (bp_state,),
        repeats,
    )
    dynamic_bp, dynamic_bp_record = _measure(
        phx.pgm.run_belief_propagation,
        (bp, bp_state),
        repeats,
    )

    gibbs = phx.pgm.prepare_chromatic_gibbs(graph)
    positions = jnp.zeros((4, size), dtype=jnp.int32)
    gibbs_state = phx.pgm.initialize_gibbs(gibbs, positions)
    key = jax.random.key(17)
    closed_gibbs, closed_gibbs_record = _measure(
        lambda state, sample_key: phx.pgm.gibbs_sweep(gibbs, state, sample_key),
        (gibbs_state, key),
        repeats,
    )
    dynamic_gibbs, dynamic_gibbs_record = _measure(
        phx.pgm.gibbs_sweep,
        (gibbs, gibbs_state, key),
        repeats,
    )

    _, hash_seconds = measure_host(
        lambda: sum(hash(graph._host_topology) for _ in range(10_000))
    )
    topology = graph._host_topology
    host_topology_bytes = (
        topology.cardinalities.nbytes
        + topology.state_offsets.nbytes
        + sum(scope.nbytes for scope in topology.factor_scopes)
    )
    passed = bool(
        jnp.allclose(dynamic_bp.log_normalizer, closed_bp.log_normalizer)
        and jnp.array_equal(dynamic_gibbs[0].positions, closed_gibbs[0].positions)
    )
    return {
        "variables": size,
        "factors": graph.num_factors,
        "host_topology_bytes": host_topology_bytes,
        "host_topology_hash_ns": hash_seconds * 1e9 / 10_000,
        "bp": {"closed": closed_bp_record, "dynamic": dynamic_bp_record},
        "gibbs": {
            "chains": gibbs_state.num_chains,
            "closed": closed_gibbs_record,
            "dynamic": dynamic_gibbs_record,
        },
        "passed": passed,
    }


def run(sizes: tuple[int, ...], repeats: int, /) -> dict[str, object]:
    if not sizes or any(size < 3 for size in sizes):
        raise ValueError("sizes must contain ring sizes of at least three.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    cases = [_case(size, repeats) for size in sizes]
    return {
        "environment": asdict(capture_environment()),
        "sizes": list(sizes),
        "repeats": repeats,
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=(16, 64, 256))
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    result = run(tuple(arguments.sizes), arguments.repeats)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
