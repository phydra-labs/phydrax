#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(6, periodic=True),
            phx.discretization.UniformCellAxisSpec(6, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (1.0, 1.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    ocean = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        phx.applications.ocean.LinearSeawaterReference(),
        temperature_diffusivity=jnp.asarray((1.0e-5, 1.0e-5, 1.0e-6)),
        salinity_diffusivity=jnp.asarray((1.0e-5, 1.0e-5, 1.0e-6)),
        coriolis_parameter=1.0e-4,
    ).prepare(discretization)
    velocity = tuple(jnp.zeros(layout.shape) for layout in discretization.face_layouts)
    z = grid.structured_axes[2].interval_centers
    temperature = 10.0 + 0.5 * jnp.broadcast_to(
        z.reshape((1, 1, z.size)), discretization.cell_shape
    )
    salinity = jnp.full(discretization.cell_shape, 35.0)
    coordinates = ocean.initial_state(velocity, temperature, salinity)
    restriction = ocean.dynamics.step_restriction(coordinates)
    stage = ocean.dynamics.stage(0.0, coordinates)
    diagnostics = ocean.dynamics.diagnostics(0.0, coordinates)
    return {
        "successful": bool(stage.success),
        "stable_step": float(restriction.selected),
        "stratification_step": float(restriction.stratification),
        "divergence_norm": float(diagnostics.divergence_norm),
        "buoyancy_exchange_defect": float(stage.buoyancy.exchange_defect),
    }


if __name__ == "__main__":
    print(run())
