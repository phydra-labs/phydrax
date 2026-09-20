#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from benchmarks._runtime import capture_environment
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm_frequent import FrequentSmallAngleSIDMPlan
from phydrax.applications.cosmology._sidm_kernels import (
    SmallAngleSplitPlan,
    TwoBodyDifferentialKernelPlan,
)
from phydrax.applications.cosmology._sidm_weighted import WeightedSIDMPacketState


def _case(packet_count: int, repetitions: int):
    indices = jnp.arange(packet_count, dtype=jnp.float64)
    positions = jnp.stack(
        (
            jnp.mod(0.5 + indices * 0.754877666, 1.0),
            jnp.mod(0.5 + indices * 0.569840291, 1.0),
            jnp.mod(0.5 + indices * 0.438579021, 1.0),
        ),
        axis=-1,
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(packet_count, dtype=jnp.int64),
        jnp.ones((packet_count,)),
        ambient_dimension=3,
    ).prepare()
    box = phx.discretization.ParticleBox(
        jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        packet_count * (packet_count - 1) // 2, box=box
    ).prepare(particles)
    species = DarkSectorSpeciesPlan("chi", 1.0)
    differential = TwoBodyDifferentialKernelPlan.constant_isotropic(species, 1.0e-4)
    plan = FrequentSmallAngleSIDMPlan(
        neighborhood,
        phx.discretization.WendlandC2SPHKernel(3),
        SmallAngleSplitPlan(differential, 0.9),
        smoothing_length_comoving=0.2,
        maximum_drag_fraction_per_step=0.1,
        maximum_transverse_variance_per_step=0.2,
        moment_tolerance=0.2,
    )
    weights = jnp.ones((packet_count,))
    velocity = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * positions[:, 0]),
            jnp.cos(2.0 * jnp.pi * positions[:, 1]),
            jnp.sin(2.0 * jnp.pi * positions[:, 2]),
        ),
        axis=-1,
    )
    state = WeightedSIDMPacketState(
        positions,
        jnp.ones((packet_count,)),
        weights,
        weights,
        weights[:, None] * 0.5 * velocity,
        jnp.ones((packet_count,), dtype="bool"),
        particles.particle_ids,
        jnp.full((packet_count,), -1, dtype=jnp.int64),
        jnp.zeros((packet_count,), dtype=jnp.int32),
        0.5,
    )
    apply = eqx.filter_jit(plan.apply)
    started = time.perf_counter()
    first = apply(state, jr.key(0), 0, 1.0e-3)
    jax.block_until_ready(first.accepted_state.canonical_momenta)
    compile_and_first_ms = 1000.0 * (time.perf_counter() - started)

    started = time.perf_counter()
    result = first
    for epoch in range(repetitions):
        result = apply(state, jr.key(epoch + 1), epoch + 1, 1.0e-3)
    jax.block_until_ready(result.accepted_state.canonical_momenta)
    execution_ms = 1000.0 * (time.perf_counter() - started) / repetitions
    selected = int(jnp.sum(result.diagnostics.selected_pairs))
    return {
        "packet_count": packet_count,
        "pair_capacity": neighborhood.pair_capacity,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "pairs_per_second": 1000.0 * neighborhood.pair_capacity / execution_ms,
        "covered_pairs": selected,
        "maximum_drag_fraction": float(result.diagnostics.maximum_drag_fraction),
        "maximum_transverse_variance": float(
            result.diagnostics.maximum_transverse_variance
        ),
        "momentum_defect_norm": float(
            jnp.sqrt(jnp.sum(result.diagnostics.total_momentum_defect**2))
        ),
        "kinetic_energy_defect": float(result.diagnostics.total_kinetic_energy_defect),
        "successful": bool(result.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-counts", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/dark_matter_sidm_frequent.json"),
    )
    arguments = parser.parse_args()
    cases = [_case(count, arguments.repeats) for count in arguments.packet_counts]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_successful": all(case["successful"] for case in cases),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
