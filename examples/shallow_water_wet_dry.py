#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    count = 64
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    system = phx.equations.ShallowWaterSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(),
        phx.discretization.ShallowWaterHydrostaticHLLPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "wet-dry-dam-break",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    bathymetry = jnp.zeros((count,))
    compiled = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        bathymetry=bathymetry,
    )
    depth = jnp.where(jnp.arange(count) < count // 2, 1.0, 0.0)
    state = jnp.stack((depth, jnp.zeros_like(depth)), axis=-1)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.25),
    )
    step = compiled.stable_step(state, cfl=0.25)
    result = runtime.advance(runtime.initialize_state(state, 0.0, step))
    updated = result.runtime_state.cell_average()
    observables = compiled.dynamics.shallow_water_observables(updated)
    return {
        "accepted": bool(result.accepted),
        "minimum_depth": float(jnp.min(observables.depth)),
        "mass_change": float(jnp.sum(observables.depth) - jnp.sum(depth)),
        "wet_cells": int(jnp.count_nonzero(observables.wet_mask)),
        "positivity_activated": bool(result.positivity.activated),
    }


if __name__ == "__main__":
    print(run())
