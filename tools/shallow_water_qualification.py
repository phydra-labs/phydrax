#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp

import phydrax as phx


def _compiled(count, bed, reconstruction, *, source=None):
    dimension = bed.ndim
    shape = bed.shape
    names = tuple("xy"[:dimension])
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(size, periodic=True) for size in shape
        ),
        axis_names=names,
    ).prepare(jnp.stack((jnp.zeros(dimension), jnp.ones(dimension))))
    system = phx.equations.ShallowWaterSystem(dimension)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    method = phx.discretization.FiniteVolumeMethodPlan(
        reconstruction,
        phx.discretization.ShallowWaterHydrostaticHLLPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        f"shallow-water-{count}",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(names),
        source=source,
        source_id=None if source is None else source.source_id,
    )
    return phx.equations.compile_conservation_problem(
        problem, discretization, method, bathymetry=bed
    )


def _lake(count):
    x = (jnp.arange(count) + 0.5) / count
    bed = 0.1 + 1.2 * jnp.exp(-100.0 * (x - 0.5) ** 2)
    compiled = _compiled(count, bed, phx.discretization.MUSCLReconstruction())
    state = jnp.stack((jnp.maximum(1.0 - bed, 0.0), jnp.zeros_like(bed)), axis=-1)
    residual, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    maximum = float(jnp.max(jnp.abs(residual)))
    defect = float(jnp.max(jnp.abs(diagnostics.conservation_defect)))
    return {
        "case": "partially-dry-lake",
        "resolution": count,
        "maximum_residual": maximum,
        "conservation_defect": defect,
        "passed": maximum <= 1e-11 and defect <= 1e-11,
    }


def _dam_break(count, steps, cfl):
    bed = jnp.zeros((count,))
    compiled = _compiled(count, bed, phx.discretization.MUSCLReconstruction())
    depth = jnp.where(jnp.arange(count) < count // 2, 1.0, 0.0)
    initial = jnp.stack((depth, jnp.zeros_like(depth)), axis=-1)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=cfl),
    )
    step = compiled.stable_step(initial, cfl=cfl)
    state = runtime.initialize_state(initial, 0.0, step)
    activations = 0
    retries = 0
    for _ in range(steps):
        result = runtime.advance(state)
        state = result.runtime_state
        activations += int(result.positivity.activated)
        retries += int(result.retries)
        if not bool(result.accepted):
            break
    final = state.cell_average()
    minimum = float(jnp.min(final[..., 0]))
    mass_defect = float(jnp.sum(final[..., 0]) - jnp.sum(initial[..., 0]))
    finite = bool(jnp.all(jnp.isfinite(final)))
    return {
        "case": "dry-dam-break",
        "resolution": count,
        "accepted_steps": int(state.accepted_step),
        "requested_steps": steps,
        "minimum_depth": minimum,
        "mass_defect": mass_defect,
        "positivity_activations": activations,
        "retries": retries,
        "finite": finite,
        "passed": (
            int(state.accepted_step) == steps
            and minimum >= 0.0
            and abs(mass_defect) <= 1e-10
            and finite
        ),
    }


def _convergence(resolutions):
    errors = []
    for count in resolutions:
        bed = jnp.zeros((count,))
        compiled = _compiled(
            count,
            bed,
            phx.discretization.MUSCLReconstruction(phx.discretization.UnlimitedLimiter()),
        )
        x = (jnp.arange(count) + 0.5) / count
        depth = 1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * x)
        velocity = 0.2
        state = jnp.stack((depth, velocity * depth), axis=-1)
        derivative = 0.2 * jnp.pi * jnp.cos(2.0 * jnp.pi * x)
        exact = jnp.stack(
            (
                -velocity * derivative,
                -(velocity**2 + 9.81 * depth) * derivative,
            ),
            axis=-1,
        )
        errors.append(float(jnp.sqrt(jnp.mean((compiled(0.0, state) - exact) ** 2))))
    orders = [
        float(jnp.log(errors[index - 1] / errors[index]) / jnp.log(2.0))
        for index in range(1, len(errors))
    ]
    return {
        "case": "smooth-residual-convergence",
        "resolutions": list(resolutions),
        "l2_errors": errors,
        "observed_orders": orders,
        "passed": bool(orders) and min(orders) >= 1.8,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("lake", "dam-break", "convergence"), default="lake"
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--resolutions", default="32,64,128")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfl", type=float, default=0.25)
    arguments = parser.parse_args()
    if arguments.resolution < 8 or arguments.steps <= 0:
        raise ValueError(
            "Qualification resolution and steps must be positive and nontrivial."
        )
    resolutions = tuple(int(value) for value in arguments.resolutions.split(","))
    if any(value < 8 for value in resolutions):
        raise ValueError("Convergence resolutions must be at least eight.")
    if arguments.case == "lake":
        report = _lake(arguments.resolution)
    elif arguments.case == "dam-break":
        report = _dam_break(arguments.resolution, arguments.steps, arguments.cfl)
    else:
        report = _convergence(resolutions)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
