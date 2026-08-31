#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


mn = phx.applications.solid_mechanics.member_network


def catenary_campaign(count):
    reference = mn.ElasticCatenaryReference(10.5, 1.0e5, jnp.asarray((0.0, -2.0, 0.0)))
    starts = jnp.zeros((count, 3))
    ends = jnp.zeros((count, 3)).at[:, 0].set(10.0)
    started = time.perf_counter()
    states = [
        mn.solve_elastic_catenary(starts[index], ends[index], reference)
        for index in range(count)
    ]
    jax.block_until_ready(states[-1].minimum_tension)
    return {
        "count": count,
        "wall_seconds": time.perf_counter() - started,
        "minimum_tension": float(min(state.minimum_tension for state in states)),
        "maximum_sag": float(max(state.sag for state in states)),
        "successful": int(sum(bool(state.valid) for state in states)),
    }


def strip_campaign(wavelength_count):
    section = mn.ThinWalledSection(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 0.2))),
        jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        jnp.asarray((0.01, 0.01)),
    )
    problem = mn.FiniteStripBucklingProblem(
        section,
        200_000.0,
        0.3,
        jnp.asarray((-100.0, -100.0)),
        jnp.geomspace(0.1, 10.0, wavelength_count),
    )
    started = time.perf_counter()
    result = mn.solve_finite_strip_buckling(problem)
    jax.block_until_ready(result.wavelength_curve)
    return {
        "wavelength_count": wavelength_count,
        "wall_seconds": time.perf_counter() - started,
        "critical_stress": float(result.critical_stress),
        "critical_half_wavelength": float(result.critical_half_wavelength),
        "successful": bool(result.successful),
    }


def reliability_campaign(sample_count):
    model = mn.StructuralRandomModel(
        jnp.asarray((0.0,)), jnp.asarray(((1.0,),)), ("load",)
    )
    limit = mn.StructuralLimitState(
        lambda parameter: 1.0 - parameter[0], limit_state_id="normal-threshold"
    )
    started = time.perf_counter()
    result = mn.monte_carlo_reliability(model, limit, jax.random.PRNGKey(0), sample_count)
    jax.block_until_ready(result.margins)
    return {
        "sample_count": sample_count,
        "wall_seconds": time.perf_counter() - started,
        "failure_probability": float(result.failure_probability),
        "standard_error": float(result.standard_error),
    }


def sequence_campaign(operation_count):
    operations = [phx.optim.PrecedenceOperation("op-0")]
    for index in range(1, operation_count):
        operations.append(
            phx.optim.PrecedenceOperation(
                f"op-{index}", predecessors=(f"op-{index - 1}",)
            )
        )
    space = phx.optim.PrecedenceSpace(tuple(operations))
    problem = mn.ConstructionSequenceSearchProblem(
        space,
        lambda node: (True, None),
        lambda node: float(len(node.completed)),
        lambda node: float(len(node.completed)),
    )
    started = time.perf_counter()
    result = mn.search_construction_sequences(problem)
    return {
        "operation_count": operation_count,
        "wall_seconds": time.perf_counter() - started,
        "explored_nodes": int(result.explored_nodes),
        "pruned_nodes": int(result.pruned_nodes),
        "successful": bool(result.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/advanced_structural.json"),
    )
    args = parser.parse_args()
    payload = {
        "catenary": catenary_campaign(2 if args.smoke else 20),
        "finite_strip": strip_campaign(20 if args.smoke else 500),
        "reliability": reliability_campaign(1_000 if args.smoke else 100_000),
        "sequence": sequence_campaign(5 if args.smoke else 20),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
