from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from _runtime import capture_environment, measure_repeated

import phydrax as phx
from phydrax._model import AbstractArrayModel


class _QuadraticModel(AbstractArrayModel):
    stiffness: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, stiffness):
        self.stiffness = jnp.asarray(stiffness)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value, /, *, key=None):
        del key
        return jnp.asarray([0.5 * self.stiffness * value[0] ** 2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--committee", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.committee <= 0:
        raise ValueError("committee must be positive")
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        [10, 20], [1, 1], [1.0, 1.0], units, atom_type_ids=[0, 0]
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.1], [1.0], 2.5)]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()
    distance = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
    ).prepare(system)
    variables = phx.atomistic.sampling.CollectiveVariableProgram((distance,))
    models = tuple(_QuadraticModel(1.0 + 0.01 * index) for index in range(args.committee))
    plan = phx.atomistic.sampling.LearnedFreeEnergyBiasPlan(
        variables,
        models,
        model_ids=tuple(f"member-{index}" for index in range(args.committee)),
        reference=[1.0],
        trusted_uncertainty=0.01,
        rejected_uncertainty=0.2,
    )
    bias = plan.prepare(dynamics)
    state = plan.initialize()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    compiled = jax.jit(lambda value: bias.evaluate(value, state, jnp.asarray(0.0)))
    result, elapsed = measure_repeated(
        lambda: compiled(positions), warmup=args.warmup, repeats=args.repeats
    )
    payload = {
        "environment": capture_environment().to_dict(),
        "committee": args.committee,
        "execution_seconds": elapsed.to_seconds_dict(),
        "successful": bool(result.successful),
        "uncertainty": float(result.uncertainty),
        "trust": float(result.trust),
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
