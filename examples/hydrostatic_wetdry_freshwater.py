#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (8.0, 3.0, 0.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("coastal",)
    ).prepare()
    depth = jnp.broadcast_to(jnp.linspace(0.01, 1.0, 8)[:, None], (8, 3))
    geometry = phx.discretization.TensorZHydrostaticGridPlan(
        discretization, depth
    ).prepare()
    ocean = phx.applications.ocean.HydrostaticPrimitiveEquationPlan(
        geometry,
        freshwater=phx.applications.ocean.FreshwaterVolumeFluxPlan(
            1.0e-5, absolute_salinity=0.0
        ),
        external_mode="split-explicit",
        wetting_and_drying=True,
        subcycle_policy=phx.applications.ocean.ExternalModeSubcyclePolicy.fixed(10),
    ).prepare()
    state = ocean.initialize_state(jnp.zeros((8, 3)))
    continuation = phx.applications.ocean.HydrostaticContinuationState.initialize(
        ocean, state
    )
    result = phx.applications.ocean.HydrostaticIMEXMidpointMethod(ocean).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.01),
        None,
    )
    epoch = geometry.metric_epoch(result.accepted_state.state.eta)
    return {
        "successful": bool(result.successful),
        "minimum_depth": float(jnp.min(epoch.total_depth)),
        "freshwater_volume": float(result.accepted_state.ledger.freshwater_volume),
        "limiter_correction": float(result.accepted_state.ledger.limiter_correction),
    }


if __name__ == "__main__":
    print(run())
