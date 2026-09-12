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


def _positions(count: int, distribution: str) -> jax.Array:
    key = jax.random.key(7300 + count)
    if distribution == "uniform":
        return 0.02 + 0.96 * jax.random.uniform(key, (count, 3), dtype=jnp.float64)
    centers = jnp.asarray([[0.18, 0.18, 0.18], [0.82, 0.82, 0.82]], dtype=jnp.float64)
    assignment = jnp.arange(count) % 2
    noise = 0.035 * jax.random.normal(key, (count, 3), dtype=jnp.float64)
    return jnp.clip(centers[assignment] + noise, 0.02, 0.98)


def _case(
    count: int,
    distribution: str,
    order: int,
    *,
    pallas_interpret: bool,
) -> dict[str, object]:
    positions = _positions(count, distribution)
    masses = 0.5 + jax.random.uniform(
        jax.random.key(9100 + count), (count,), dtype=positions.dtype
    )
    softening = 0.015
    tree = phx.solver.ParticleOctreePlan3D((1.0, 1.0, 1.0), 8).prepare(positions, masses)
    direct_plan = phx.solver.DirectParticleGravityPlan(
        phx.solver.NewtonianPairKernel(1.0, softening=softening)
    )
    fmm_plan = phx.solver.UniformFMMPlan(
        1.0,
        phx.solver.CartesianExpansionSpace(order),
        softening=softening,
        opening_angle=0.5,
        maximum_leaf_occupancy=4,
        coarsening_factor=4,
        target_top_nodes=1,
        execution_backend="pallas" if pallas_interpret else "jax",
        pallas_interpret=pallas_interpret,
    )
    direct = eqx.filter_jit(direct_plan.evaluate)
    fmm = eqx.filter_jit(fmm_plan.evaluate)
    direct_result, direct_first = _measure(direct, positions, masses)
    _, direct_steady = _measure(direct, positions, masses)
    fmm_result, fmm_first = _measure(fmm, tree)
    _, fmm_steady = _measure(fmm, tree)
    reference = direct_result[0]
    absolute = jnp.sqrt(jnp.sum((fmm_result.acceleration - reference) ** 2, axis=-1))
    reference_norm = jnp.sqrt(jnp.sum(reference * reference, axis=-1))
    relative = absolute / jnp.maximum(reference_norm, 1.0e-14)
    resources = fmm_result.fmm_evidence
    assert resources is not None
    return {
        "points": count,
        "distribution": distribution,
        "order": order,
        "backend": fmm_plan.execution_backend,
        "direct_first_seconds": direct_first,
        "direct_steady_seconds": direct_steady,
        "fmm_first_seconds": fmm_first,
        "fmm_steady_seconds": fmm_steady,
        "steady_speedup_over_direct": direct_steady / fmm_steady,
        "maximum_relative_error": float(jnp.max(relative, initial=0.0)),
        "rms_relative_error": float(jnp.sqrt(jnp.mean(relative * relative))),
        "active_nodes": int(fmm_result.evidence.active_nodes),
        "far_interactions": int(resources.required_far),
        "near_interactions": int(resources.required_near),
        "direct_particle_interactions": int(resources.p2p_count),
        "minimum_scale_exponent": float(resources.minimum_scale_exponent),
        "maximum_scale_exponent": float(resources.maximum_scale_exponent),
        "net_force_norm": float(jnp.sqrt(jnp.sum(fmm_result.evidence.net_force**2))),
        "successful": bool(fmm_result.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark scale-normalized Cartesian particle FMM."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pallas-interpret", action="store_true")
    arguments = parser.parse_args()
    counts = (8, 16) if arguments.smoke else (32, 64, 128)
    orders = (1, 3) if arguments.smoke else (1, 3, 5)
    cases = [
        _case(
            count,
            distribution,
            order,
            pallas_interpret=arguments.pallas_interpret,
        )
        for distribution in ("uniform", "clustered")
        for count in counts
        for order in orders
    ]
    report = {
        "kind": "particle-cartesian-fmm-benchmark",
        "device": str(jax.devices()[0]),
        "cases": cases,
        "passed": all(case["successful"] for case in cases),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
