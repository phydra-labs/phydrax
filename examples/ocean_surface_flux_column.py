#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (1.0, 1.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("ocean",)
    ).prepare()
    reference = phx.applications.ocean.LinearSeawaterReference()
    temperature_flux = reference.temperature_flux_from_heat_flux(100.0)
    ocean = phx.applications.ocean.CartesianBoussinesqOceanPlan(
        phx.applications.ocean.OceanAxisConvention(),
        reference,
        temperature_surface_flux=phx.discretization.MACScalarBoundaryCondition(
            "flux", temperature_flux
        ),
        temperature_diffusivity=jnp.asarray((0.0, 0.0, 1.0e-5)),
    ).prepare(discretization)
    velocity = tuple(jnp.zeros(layout.shape) for layout in discretization.face_layouts)
    coordinates = ocean.initial_state(
        velocity,
        jnp.full(discretization.cell_shape, reference.reference_temperature),
        jnp.full(discretization.cell_shape, reference.reference_salinity),
    )
    continuation = phx.applications.ocean.OceanBoussinesqContinuationState.initialize(
        coordinates
    )
    result = phx.applications.ocean.OceanBoussinesqSSPRK33Method(ocean).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.01),
        None,
    )
    before = ocean.state_view(continuation.coordinates)
    after = ocean.state_view(result.accepted_state.coordinates)
    volume = discretization.cell_volumes
    return {
        "successful": bool(result.successful),
        "temperature_content_change": float(
            jnp.sum(volume * (after.temperature - before.temperature))
        ),
        "accepted_boundary_content": float(
            result.accepted_state.temperature_boundary_content
        ),
    }


if __name__ == "__main__":
    print(run())
