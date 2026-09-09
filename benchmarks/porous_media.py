from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

from phydrax.applications import porous_media as porous
from phydrax.discretization import UnstructuredFiniteVolumePlan
from phydrax.discretization.finite_volume import HybridDiffusionBoundary
from phydrax.linalg import DenseLU, LinearSolvePolicy
from phydrax.nonlinear import NewtonKrylov, NonlinearTermination


def _problem():
    discretization = UnstructuredFiniteVolumePlan(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            )
        ),
        tetrahedra=np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    ).prepare()
    exterior = np.flatnonzero(np.asarray(discretization.neighbour_cells) < 0)
    pressure_boundary = porous.PorousBoundaryConditions(
        discretization,
        pressure_Pa={int(face): -2.0e4 for face in exterior},
    )
    method = NewtonKrylov(linear_policy=LinearSolvePolicy(DenseLU()))
    termination = NonlinearTermination(
        absolute_residual=1.0e-10,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=40,
    )
    water = porous.RichardsPlan(
        discretization,
        porous.PorousMaterial(
            0.3,
            1.0e-12,
            viscosity_temperature_K_inverse=0.01,
        ),
        porous.VanGenuchtenMualem(1.0e-5, 2.0),
        pressure_boundary,
        gravity_m_s2=(0.0, 0.0, 0.0),
        method=method,
        termination=termination,
    )
    thermal_boundary = HybridDiffusionBoundary(
        discretization,
        dirichlet={int(face): 300.0 for face in exterior},
    )
    coupled = porous.CoupledWaterHeatPlan(
        water,
        porous.PorousThermalMaterial(2.0, dry_conductivity_W_m_K=1.0),
        thermal_boundary,
    )
    state = coupled.initialize(-2.0e4, 300.0)
    return coupled, state, jnp.asarray(exterior, dtype=jnp.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats positive")

    coupled, previous, exterior = _problem()
    source = jnp.asarray(1.0e-5)
    dt = 5.0
    heat_source = 2.0

    def advance(source_rate):
        mass_source = jnp.asarray((source_rate, 0.0))
        heat = jnp.asarray((heat_source, 0.0))
        result = coupled.step(
            previous,
            dt,
            source_kg_s=mass_source,
            source_W=heat,
        )
        heat_flux = coupled.heat_fluxes(result.state, result.fluxes)
        mass_balance = jnp.sum(
            result.state.water_mass_kg - previous.water_mass_kg
        ) + dt * (jnp.sum(result.fluxes.mass_face_rates[exterior]) - jnp.sum(mass_source))
        energy_balance = jnp.sum(result.state.energy_J - previous.energy_J) + dt * (
            jnp.sum(heat_flux.total_face_rates_W[exterior]) - jnp.sum(heat)
        )
        return (
            result.state.water_mass_kg,
            result.state.temperature_K,
            result.residual,
            mass_balance,
            energy_balance,
            result.successful,
        )

    def evaluate(source_rate):
        result = advance(source_rate)
        _, mass_tangent = jax.jvp(
            lambda value: advance(value)[0],
            (source_rate,),
            (jnp.asarray(1.0),),
        )
        mass_adjoint = jax.grad(lambda value: jnp.sum(advance(value)[0]))(source_rate)
        return result + (mass_tangent, mass_adjoint)

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(source), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(source), warmup=args.warmup, repeats=args.repeats
    )
    (
        mass,
        temperature,
        residual,
        mass_balance,
        energy_balance,
        successful,
        tangent,
        adjoint,
    ) = result
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "cells": coupled.water.diffusion.cell_count,
            "faces": coupled.water.diffusion.face_count,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "execution": execution.to_seconds_dict(),
        "physics": {
            "successful": bool(successful),
            "finite": bool(
                jnp.all(jnp.isfinite(mass)) & jnp.all(jnp.isfinite(temperature))
            ),
            "maximum_residual": float(jnp.max(jnp.abs(residual))),
            "mass_balance_kg": float(mass_balance),
            "energy_balance_J": float(energy_balance),
            "tangent_adjoint_residual": float(jnp.abs(jnp.sum(tangent) - adjoint)),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
