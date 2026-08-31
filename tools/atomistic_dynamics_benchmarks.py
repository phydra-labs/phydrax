#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualified short-range atomistic dynamics benchmark"
    )
    parser.add_argument("--atoms", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--step-size", type=float, default=2.0e-4)
    return parser


def _positions(count: int) -> tuple[np.ndarray, float]:
    side = int(np.ceil(count ** (1.0 / 3.0)))
    spacing = 1.25
    grid = np.stack(
        np.meshgrid(np.arange(side), np.arange(side), np.arange(side), indexing="ij"),
        axis=-1,
    ).reshape((-1, 3))[:count]
    return 1.0 + spacing * grid, side * spacing + 2.5


def _runtime(arguments):
    if arguments.atoms < 2 or arguments.steps <= 0 or arguments.repeats <= 0:
        raise ValueError(
            "atoms, steps, and repeats must be positive; atoms must exceed one."
        )
    positions, length = _positions(arguments.atoms)
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    cell = phx.discretization.ParticleCell(length * np.eye(3))
    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(arguments.atoms),
        np.ones((arguments.atoms,), dtype=np.int32),
        np.ones((arguments.atoms,)),
        units,
        atom_type_ids=np.zeros((arguments.atoms,), dtype=np.int32),
        cell=cell,
    ).prepare()
    pair_capacity = arguments.atoms * (arguments.atoms - 1) // 2
    base = phx.discretization.DenseParticleNeighborhoodPlan(pair_capacity, box=cell)
    neighborhood = phx.discretization.VerletParticleNeighborhoodPlan(
        base, 2.5, 0.3
    ).prepare(system.particles)
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5, switch_distance=2.2)]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(arguments.step_size),
    ).prepare()
    initial = dynamics.initialize_state(
        positions,
        velocity=jnp.zeros_like(jnp.asarray(positions)),
        key=jax.random.key(0),
    )
    rollout = phx.atomistic.AtomisticRolloutPlan(
        dynamics,
        phx.atomistic.AtomisticTrajectoryPlan(arguments.steps, retention="final"),
        replay=phx.atomistic.AtomisticReplayPolicy("full"),
    )
    return positions, dynamics, initial, rollout


def main() -> None:
    arguments = _parser().parse_args()
    positions, dynamics, initial, rollout = _runtime(arguments)
    compiled = eqx.filter_jit(rollout.rollout)
    started = time.perf_counter()
    first = compiled(initial)
    jax.block_until_ready(first.final_state.energy.total_energy)
    compile_and_first = time.perf_counter() - started
    timings = []
    result = first
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        result = compiled(initial)
        jax.block_until_ready(result.final_state.energy.total_energy)
        timings.append(time.perf_counter() - started)
    initial_energy = float(initial.energy.total_energy)
    final_energy = float(result.final_state.energy.total_energy)
    drift = abs(final_energy - initial_energy) / max(abs(initial_energy), 1.0e-30)
    diagnostics = dynamics.diagnostics(result.final_state)
    payload = {
        "configuration": {
            "atoms": arguments.atoms,
            "steps": arguments.steps,
            "step_size": arguments.step_size,
            "repeats": arguments.repeats,
        },
        "identities": {
            "system": dynamics.system.prepared_id,
            "potential": dynamics.potential.prepared_id,
            "dynamics": dynamics.prepared_id,
            "rollout": rollout.rollout_id,
        },
        "work": {
            "pair_capacity": arguments.atoms * (arguments.atoms - 1) // 2,
            "candidate_pairs": int(result.final_state.neighborhood.candidate_pair_count),
            "rebuild_count": int(diagnostics.neighborhood_rebuild_count),
        },
        "performance": {
            "compile_and_first_seconds": compile_and_first,
            "steady_seconds_mean": float(np.mean(timings)),
            "steady_seconds_min": float(np.min(timings)),
            "atom_steps_per_second": (
                arguments.atoms * arguments.steps / float(np.mean(timings))
            ),
        },
        "physics": {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "relative_energy_drift": drift,
            "net_force_norm": float(jnp.sqrt(jnp.sum(diagnostics.net_internal_force**2))),
            "successful": bool(result.successful),
        },
        "gates": {
            "execution": bool(result.successful),
            "finite_energy": bool(np.isfinite(final_energy)),
            "relative_energy_drift_below_1e-3": drift < 1.0e-3,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "position_bytes": int(positions.nbytes),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
