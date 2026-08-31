#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

import jax.numpy as jnp

import phydrax as phx


def _periodic_advection(resolutions, cfl):
    case = phx.equations.periodic_advection_verification_case()
    errors = []
    conservation = []
    for resolution in resolutions:
        grid = phx.discretization.TensorGridPlan(
            (phx.discretization.UniformCellAxisSpec(resolution, periodic=True),),
            axis_names=("x",),
        ).prepare(jnp.asarray([[0.0], [1.0]]))
        discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
        problem = phx.equations.ConservationProblemIR(
            case.name,
            "state",
            case.system,
            phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
        )
        method = phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.MUSCLReconstruction(phx.discretization.UnlimitedLimiter()),
            phx.discretization.HLLFluxPlan(),
        )
        compiled = phx.equations.compile_conservation_problem(
            problem, discretization, method
        )
        state = case.initial_state(discretization.cell_centers, 0.0, None)
        initial_integral = jnp.sum(discretization.cell_volumes[..., None] * state)
        time = jnp.asarray(0.0)
        stepper = phx.solver.UnsplitFiniteVolumeSSPRK3Plan(compiled.dynamics)
        while float(time) < case.final_time:
            stable = compiled.stable_step(state, cfl=cfl)
            dt = jnp.minimum(stable, case.final_time - time)
            result = stepper.advance(time, state, dt)
            state, time = result.state, result.time
        exact = case.exact_state(discretization.cell_centers, time, None)
        norms = phx.equations.finite_volume_error_norms(
            state, exact, discretization.cell_volumes
        )
        errors.append(norms.l2)
        conservation.append(
            jnp.sum(discretization.cell_volumes[..., None] * state) - initial_integral
        )
    convergence = phx.equations.finite_volume_convergence_result(
        resolutions, jnp.asarray(errors), 2.0
    )
    return {
        "case": case.name,
        "resolutions": list(resolutions),
        "l2_errors": [float(value) for value in convergence.errors],
        "observed_orders": [float(value) for value in convergence.observed_orders],
        "passed": bool(convergence.passed),
        "conservation_defects": [float(value) for value in conservation],
    }


def _euler_case(case, resolution, steps):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(resolution),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=case.system.component_names
    ).prepare()
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    problem = phx.equations.ConservationProblemIR(
        case.name,
        "state",
        case.system,
        phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(),
        phx.discretization.HLLCFluxPlan(),
        positivity=phx.discretization.ConvexStateLimiterPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.35),
    )
    state = case.initial_state(discretization.cell_centers, 0.0, None)
    step_size = case.final_time / steps
    runtime_state = runtime.initialize_state(state, 0.0, step_size)
    activations = 0
    retries = 0
    attempts = 0
    for _ in range(steps):
        result = runtime.advance_prescribed(runtime_state, step_size)
        activations += int(result.attempted.positivity.activated)
        retries += int(result.attempted.retries)
        attempts += 1
        runtime_state = result.runtime_state
        if not bool(result.accepted):
            break
    primitive = case.system.conserved_to_primitive(runtime_state.cell_average())
    return {
        "case": case.name,
        "resolution": resolution,
        "steps": int(runtime_state.accepted_step),
        "attempts": attempts,
        "target_final_time": case.final_time,
        "actual_final_time": float(runtime_state.time),
        "final_time_error": abs(float(runtime_state.time) - case.final_time),
        "minimum_density": float(jnp.min(primitive[..., 0])),
        "minimum_pressure": float(jnp.min(primitive[..., -1])),
        "positivity_activations": activations,
        "retries": retries,
        "status": int(runtime_state.last_status),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "advection",
            "sod",
            "lax",
            "double-rarefaction",
            "woodward-colella",
        ),
        default="advection",
    )
    parser.add_argument("--resolutions", default="32,64,128")
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--sod-steps", type=int, default=40)
    arguments = parser.parse_args()
    resolutions = tuple(int(value) for value in arguments.resolutions.split(","))
    if any(value < 8 for value in resolutions) or arguments.sod_steps <= 0:
        raise ValueError("Qualification resolutions and step count are too small.")
    cases = {
        "sod": phx.equations.sod_verification_case,
        "lax": phx.equations.lax_verification_case,
        "double-rarefaction": phx.equations.double_rarefaction_verification_case,
        "woodward-colella": phx.equations.woodward_colella_verification_case,
    }
    report = (
        _periodic_advection(resolutions, arguments.cfl)
        if arguments.case == "advection"
        else _euler_case(
            cases[arguments.case](),
            resolutions[-1],
            arguments.sod_steps,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
