from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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

    def __init__(self, stiffness: Any) -> None:
        self.stiffness = jnp.asarray(stiffness)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value: Any, /, *, key: Any = None) -> Any:
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
    distance = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
    ).prepare(system)
    variables = phx.atomistic.sampling.CollectiveVariableProgram((distance,))
    models = tuple(_QuadraticModel(1.0 + 0.01 * index) for index in range(args.committee))
    form = phx.atomistic.sampling.LearnedFreeEnergyBiasForm(
        models,
        model_ids=tuple(f"member-{index}" for index in range(args.committee)),
        reference=[1.0],
        trusted_uncertainty=0.01,
        rejected_uncertainty=0.2,
    )
    bias = phx.atomistic.sampling.AtomisticBiasTerm(
        variables, form, name="learned-bias", force_group=1
    ).prepare(system)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])

    def evaluate(value: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        energy, cvs, successful = bias.bias(
            phx.atomistic.sampling.CollectiveVariableGeometry(value), None
        )
        committee = form.committee(cvs.values)
        return energy, successful, committee.uncertainty, committee.trust

    compiled = jax.jit(jax.value_and_grad(lambda value: evaluate(value)[0]))
    diagnostics = jax.jit(evaluate)
    _, elapsed = measure_repeated(
        lambda: compiled(positions), warmup=args.warmup, repeats=args.repeats
    )
    _, successful, uncertainty, trust = diagnostics(positions)
    payload = {
        "environment": capture_environment().to_dict(),
        "committee": args.committee,
        "execution_seconds": elapsed.to_seconds_dict(),
        "successful": bool(successful),
        "uncertainty": float(uncertainty),
        "trust": float(trust),
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
