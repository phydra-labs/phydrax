#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    shape = (8, 8)
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for count in shape
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    system = phx.equations.ShallowWaterSystem(2)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.ShallowWaterHydrostaticHLLPlan(),
    )
    coriolis = phx.equations.ShallowWaterCoriolisSource(0.5, beta=0.1)
    problem = phx.equations.ConservationProblemIR(
        "rotating-shallow-water",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x", "y")),
        source=coriolis,
        source_id=coriolis.source_id,
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        bathymetry=jnp.zeros(shape),
    )
    depth = jnp.ones(shape)
    state = jnp.stack((depth, jnp.ones(shape), jnp.zeros(shape)), axis=-1)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.25),
    )
    runtime_state = runtime.initialize_state(state, 0.0, 0.005)
    for _ in range(5):
        result = runtime.advance(runtime_state)
        runtime_state = result.runtime_state
    updated = runtime_state.cell_average()
    momentum_norm = jnp.sqrt(jnp.sum(updated[..., 1:] ** 2, axis=-1))
    return {
        "time": float(runtime_state.time),
        "mass_change": float(jnp.sum(updated[..., 0]) - jnp.sum(depth)),
        "mean_momentum_norm": float(jnp.mean(momentum_norm)),
        "source_id": coriolis.source_id,
    }


if __name__ == "__main__":
    print(run())
