from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _case(depth: int):
    grid = phx.discretization.AdaptiveDyadicGridPlan(
        phx.discretization.MortonAddressPlan((0.0, 0.0), (1.0, 1.0), depth),
        cell_capacity=4096,
    )
    topology = grid.prepare()
    adapt_seconds = 0.0
    balance_refinements = 0
    for _ in range(depth):
        center = topology.cell_centers
        radius = jnp.linalg.norm(center - jnp.asarray((0.5, 0.5)), axis=-1)
        refine = (
            topology.leaf_active
            & (topology.levels < depth)
            & (jnp.abs(radius - 0.25) < 0.4 / (2.0**topology.levels))
        )
        started = time.perf_counter()
        transition = grid.adapt(topology, refine_mask=refine)
        adapt_seconds += time.perf_counter() - started
        balance_refinements += int(transition.evidence.balance_refinements)
        topology = transition.accepted
    system = phx.equations.EulerSystem(2)
    started = time.perf_counter()
    discretization = phx.discretization.DyadicFiniteVolumePlan(
        topology,
        component_names=system.component_names,
    ).prepare()
    lowering_seconds = time.perf_counter() - started
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "dyadic-benchmark", "state", system, boundaries
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.2, -0.1, 1.0)),
        (discretization.cell_count, system.component_count),
    )
    state = system.primitive_to_conserved(primitive)
    evaluate = eqx.filter_jit(dynamics)
    started = time.perf_counter()
    residual = evaluate(jnp.asarray(0.0), state, None)
    residual.block_until_ready()
    first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    residual = evaluate(jnp.asarray(0.0), state, None)
    residual.block_until_ready()
    steady_seconds = time.perf_counter() - started
    return {
        "depth": depth,
        "allocated_cells": int(topology.evidence.allocated_cells),
        "active_leaves": int(topology.evidence.active_leaves),
        "faces": discretization.face_count,
        "balance_refinements": balance_refinements,
        "adapt_seconds": adapt_seconds,
        "lowering_seconds": lowering_seconds,
        "residual_first_seconds": first_seconds,
        "residual_steady_seconds": steady_seconds,
        "constant_state_defect": float(jnp.max(jnp.abs(residual))),
        "successful": bool(topology.evidence.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dyadic AMR topology and FV.")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    depths = (2,) if arguments.smoke else (2, 3, 4)
    cases = [_case(depth) for depth in depths]
    report = {
        "kind": "dyadic-amr-benchmark",
        "device": str(jax.devices()[0]),
        "cases": cases,
        "passed": all(
            case["successful"] and case["constant_state_defect"] < 1.0e-11
            for case in cases
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
