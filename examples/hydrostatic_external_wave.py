#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -10.0), (8.0, 4.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrostatic",)
    ).prepare()
    geometry = phx.discretization.TensorZHydrostaticGridPlan(
        discretization, jnp.full((8, 4), 10.0)
    ).prepare()
    ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
        geometry,
        coriolis_f0=1.0e-4,
    ).prepare()
    x = jnp.arange(8)[:, None]
    eta = jnp.broadcast_to(1.0e-3 * jnp.sin(2.0 * jnp.pi * x / 8), (8, 4))
    state = ocean.initialize_state(eta)
    continuation = phx.applications.ocean.HydrostaticContinuationState.initialize(
        ocean, state
    )
    result = phx.applications.ocean.HydrostaticIMEXMidpointMethod(ocean).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.02),
        None,
    )
    view = phx.applications.ocean.hydrostatic_diagnostic_view(
        ocean, result.accepted_state
    )
    return {
        "successful": bool(result.successful),
        "volume_residual": float(result.accepted_state.ledger.volume_change),
        "maximum_eta": float(jnp.max(jnp.abs(view.eta))),
        "free_surface_energy": float(view.free_surface_energy),
    }


if __name__ == "__main__":
    print(run())
