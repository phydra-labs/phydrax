from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _block(value) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(function, *arguments):
    started = time.perf_counter()
    value = function(*arguments)
    _block(value)
    return value, time.perf_counter() - started


def _treepm_case(count: int):
    key = jax.random.key(8820 + count)
    positions = 0.05 + 0.9 * jax.random.uniform(key, (count, 3))
    masses = 0.5 + jax.random.uniform(jax.random.fold_in(key, 1), (count,))
    tree = phx.solver.ParticleOctreePlan3D((1.0, 1.0, 1.0), 8).prepare(positions, masses)
    split = phx.solver.TreePMSplitPolicy(0.08, 0.4, "treepm-benchmark")
    barnes = phx.solver.BarnesHutGravityPlan(
        1.0,
        softening=0.02,
        opening_angle=0.3,
    )
    fmm = phx.solver.UniformFMMPlan(
        1.0,
        phx.solver.CartesianExpansionSpace(3),
        softening=0.02,
        opening_angle=0.45,
        maximum_leaf_occupancy=2,
        coarsening_factor=2,
        target_top_nodes=1,
        short_range_scale=split.split_scale,
        short_range_cutoff=split.cutoff,
    )
    long_range = jnp.zeros_like(positions)
    barnes_evaluate = eqx.filter_jit(phx.solver.TreePMPlan(barnes, split).evaluate)
    fmm_evaluate = eqx.filter_jit(phx.solver.TreePMPlan(fmm, split).evaluate)
    reference, barnes_first = _measure(barnes_evaluate, tree, long_range)
    _, barnes_steady = _measure(barnes_evaluate, tree, long_range)
    candidate, fmm_first = _measure(fmm_evaluate, tree, long_range)
    _, fmm_steady = _measure(fmm_evaluate, tree, long_range)
    error = jnp.max(
        jnp.linalg.norm(
            candidate.short_range_acceleration - reference.short_range_acceleration,
            axis=-1,
        ),
        initial=0.0,
    )
    return {
        "kind": "treepm-short-range",
        "points": count,
        "barnes_first_seconds": barnes_first,
        "barnes_steady_seconds": barnes_steady,
        "fmm_first_seconds": fmm_first,
        "fmm_steady_seconds": fmm_steady,
        "maximum_absolute_difference": float(error),
        "fmm_successful": bool(candidate.successful),
        "fmm_far_interactions": int(candidate.short_evidence.accepted_leaf_interactions),
        "fmm_direct_interactions": int(
            candidate.short_evidence.direct_particle_interactions
        ),
    }


def _ewald_case(count: int):
    key = jax.random.key(9920 + count)
    positions = 0.05 + 0.9 * jax.random.uniform(key, (count, 3))
    masses = 0.5 + jax.random.uniform(jax.random.fold_in(key, 1), (count,))
    common = {
        "softening": 0.02,
        "alpha": 4.0,
        "real_shells": 0,
        "reciprocal_modes": 3,
    }
    direct = phx.solver.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        **common,
    )
    radius = phx.solver.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        real_space_execution="screened_radius",
        real_cutoff=2.0,
        maximum_real_pairs=count * count,
        **common,
    )
    direct_evaluate = eqx.filter_jit(direct.evaluate)
    radius_evaluate = eqx.filter_jit(radius.evaluate)
    reference, direct_first = _measure(direct_evaluate, positions, masses)
    _, direct_steady = _measure(direct_evaluate, positions, masses)
    candidate, radius_first = _measure(radius_evaluate, positions, masses)
    _, radius_steady = _measure(radius_evaluate, positions, masses)
    return {
        "kind": "ewald-real-space",
        "points": count,
        "direct_first_seconds": direct_first,
        "direct_steady_seconds": direct_steady,
        "radius_first_seconds": radius_first,
        "radius_steady_seconds": radius_steady,
        "maximum_absolute_difference": float(
            jnp.max(jnp.abs(candidate.acceleration - reference.acceleration))
        ),
        "required_real_pairs": int(candidate.evidence.required_real_pairs),
        "real_pair_capacity": int(candidate.evidence.real_pair_capacity),
        "successful": bool(candidate.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TreePM and Ewald execution choices."
    )
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    count = 8 if arguments.smoke else 32
    cases = (_treepm_case(count), _ewald_case(count))
    report = {
        "kind": "treepm-ewald-execution-benchmark",
        "device": str(jax.devices()[0]),
        "cases": cases,
        "passed": all(
            case.get("successful", case.get("fmm_successful", False)) for case in cases
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
