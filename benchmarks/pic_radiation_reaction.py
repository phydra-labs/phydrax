from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, get_args

import jax
import jax.numpy as jnp
import jax.random as jr
from _runtime import (
    capture_environment,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx
from phydrax.discretization import pic
from phydrax.units import CHARGE, UnitDefinition


_CHARGE = -0.1
_MASS = 0.1


def _scale(chi: float, gamma: float) -> phx.ElectromagneticScaleContract:
    """Code scale whose ħ gives quantum parameter ``chi`` at ``gamma`` in ``B = 1``."""
    hbar = chi * _MASS**2 / (gamma * math.sqrt(1.0 - 1.0 / gamma**2) * abs(_CHARGE))
    return phx.ElectromagneticScaleContract.code_units(
        pic.PIC_CODE_RELATIVITY.dimensional_scale,
        UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
        gravitational_constant=1,
        speed_of_light=1,
        reduced_planck_constant=Fraction(hbar).limit_denominator(10**18),
        boltzmann_constant=1,
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=1,
        constant_set_id="radiation-reaction-benchmark",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=1_000_000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.particles <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("positive particles/repeats and nonnegative warmup are required")

    tables, table_seconds = measure_host(
        lambda: pic.RadiationReactionTables(maximum_chi=5.0)
    )
    scale = _scale(0.5, 1000.0)
    count = args.particles
    keys = jr.split(jr.key(0), 6)
    gamma = jr.uniform(keys[0], (count,), dtype=jnp.float64, minval=500.0, maxval=1500.0)
    direction = jr.normal(keys[1], (count, 3), dtype=jnp.float64)
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    proper = direction * jnp.sqrt(gamma**2 - 1.0)[:, None]
    electric = 0.1 * jr.normal(keys[2], (count, 3), dtype=jnp.float64)
    magnetic = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (count, 3))
    gradient = 0.01 * jr.normal(keys[3], (count, 3, 3), dtype=jnp.float64)
    rate = 0.01 * jr.normal(keys[4], (count, 3), dtype=jnp.float64)
    active = jnp.ones((count,), dtype=bool)
    identity_low = jnp.arange(count, dtype=jnp.uint32)
    identity_high = jnp.zeros((count,), dtype=jnp.uint32)
    dt = 1.0e-5

    models: dict[str, Any] = {}
    for model in get_args(pic.RadiationReactionModel):
        match model:
            case "landau-lifshitz-reduced" | "landau-lifshitz":
                model_tables = None
            case "quantum-corrected-landau-lifshitz" | "stochastic-fokker-planck":
                model_tables = tables
            case _:
                raise ValueError(f"unknown radiation-reaction model {model!r}")
        plan = pic.RadiationReactionPlan(
            model,
            scale,
            _CHARGE,
            _MASS,
            tables=model_tables,
            maximum_chi=5.0,
            minimum_gamma=1.0,
        )

        def evaluate(key: Any, plan: pic.RadiationReactionPlan = plan) -> Any:
            derivatives = (
                {
                    "electric_gradient": gradient,
                    "magnetic_gradient": gradient,
                    "electric_rate": rate,
                    "magnetic_rate": rate,
                }
                if plan.requires_field_derivatives
                else {}
            )
            wiener = (
                plan.wiener_increments(key, identity_high, identity_low)
                if plan.stochastic
                else None
            )
            result = plan.apply(
                proper, electric, magnetic, dt, active, wiener=wiener, **derivatives
            )
            return result.proper_velocity, jnp.sum(result.radiated_energy), result.flags

        function = jax.jit(evaluate)
        compiled, compilation = measure_lower_and_compile(
            lambda function=function: function.lower(keys[5]),
            lambda lowered: lowered.compile(),
        )
        result, execution = measure_repeated(
            lambda compiled=compiled: compiled(keys[5]),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _, radiated, flags = result
        models[model] = {
            "identity": plan.plan_id,
            "compilation": {
                "lowering_seconds": compilation.lowering_seconds,
                "compilation_seconds": compilation.compilation_seconds,
            },
            "execution": execution.to_seconds_dict(),
            "physics": {
                "radiated_energy": float(radiated),
                "flagged_particles": int(jnp.sum(flags != 0)),
            },
        }
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "particles": count,
            "step_size": dt,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "tables": {
            "identity": tables.tables_id,
            "preparation_seconds": table_seconds,
            "node_count": tables.node_count,
            "quadrature_error": tables.quadrature_error,
            "interpolation_error": tables.interpolation_error,
        },
        "models": models,
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
