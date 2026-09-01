"""Conservation, equilibrium, energy, and execution evidence for electrokinetics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import capture_environment


def qualification():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(32, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    schema = phx.equations.ChemicalSpeciesSchema(
        ("cation", "anion"),
        (
            phx.equations.ChemicalPhaseKind.LIQUID,
            phx.equations.ChemicalPhaseKind.LIQUID,
        ),
        jnp.asarray((0.023, 0.035)),
        ("M", "X"),
        jnp.asarray(((1, 0), (0, 1)), dtype=jnp.int32),
        jnp.asarray((1, -1), dtype=jnp.int32),
    )
    parameters = phx.equations.ElectrolyteTransportParameters(
        schema, jnp.asarray((1e-3, 1e-3)), 300.0, 1e8
    )
    electrostatic = phx.solver.CochainElectrostaticPlan(
        bridge,
        phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge),
        permittivity=parameters.permittivity,
    )
    plan = phx.solver.PoissonNernstPlanckPlan(
        electrostatic,
        phx.equations.IdealDiluteElectrochemicalClosure(schema),
        parameters,
        energy_tolerance=1e-8,
    )
    equilibrium = plan.evaluate(jnp.ones((32, 2)))
    coordinate = (jnp.arange(32) + 0.5) / 32.0
    perturbation = 0.02 * jnp.sin(2.0 * jnp.pi * coordinate)
    concentrations = jnp.stack((1.0 + perturbation, 1.0 - perturbation), axis=-1)
    before = plan.evaluate(concentrations)
    step = jnp.minimum(1e-4, 0.25 * before.explicit_step_restriction)
    advanced = plan.step(concentrations, step)
    mass_error = float(
        jnp.max(
            jnp.abs(
                jnp.sum(advanced.concentrations, axis=0) - jnp.sum(concentrations, axis=0)
            )
        )
    )
    energy_gradient = jax.grad(
        lambda amplitude: (
            plan.evaluate(
                jnp.stack(
                    (
                        1.0 + amplitude * jnp.sin(2.0 * jnp.pi * coordinate),
                        1.0 - amplitude * jnp.sin(2.0 * jnp.pi * coordinate),
                    ),
                    axis=-1,
                )
            ).total_free_energy
        )
    )(jnp.asarray(0.01))
    cases = {
        "boltzmann_equilibrium": {
            "maximum_rate": float(jnp.max(jnp.abs(equilibrium.concentration_rate))),
            "passed": bool(
                equilibrium.successful
                & (jnp.max(jnp.abs(equilibrium.concentration_rate)) < 1e-11)
            ),
        },
        "conservative_relaxation": {
            "mass_error": mass_error,
            "energy_change": float(
                advanced.evaluation.total_free_energy - before.total_free_energy
            ),
            "passed": bool(advanced.successful & (mass_error < 1e-9)),
        },
        "differentiation": {
            "energy_gradient": float(energy_gradient),
            "passed": bool(jnp.isfinite(energy_gradient)),
        },
    }
    return {
        "environment": capture_environment().to_dict(),
        "evidence_levels": {
            "invariant_complete": bool(cases["conservative_relaxation"]["passed"]),
            "physics_qualified": bool(cases["boltzmann_equilibrium"]["passed"]),
            "differentiation_qualified": bool(cases["differentiation"]["passed"]),
            "execution_qualified": all(bool(case["passed"]) for case in cases.values()),
            "deployment_qualified": False,
        },
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/electrokinetics_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualification()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
