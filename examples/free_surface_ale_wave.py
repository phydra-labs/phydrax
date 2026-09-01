#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def run():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(3, periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (4.0, 4.0, 0.0))))
    reference = phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrodynamics",)
    ).prepare()
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        reference,
        jnp.full((4, 4), -1.0),
        maximum_slope=0.5,
        maximum_iterations=150,
    )
    hydrodynamics = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        coupling_iterations=5,
        coupling_tolerance=1.0e-7,
    ).prepare()
    x = jnp.arange(4)[:, None]
    eta = jnp.broadcast_to(1.0e-4 * jnp.sin(2.0 * jnp.pi * x / 4), (4, 4))
    state = hydrodynamics.initial_state(eta)
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )
    result = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(
        hydrodynamics
    ).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.002),
        None,
    )
    view = phx.applications.hydrodynamics.free_surface_diagnostic_view(
        hydrodynamics, result.accepted_state
    )
    return {
        "successful": bool(result.successful),
        "maximum_eta": float(jnp.max(jnp.abs(view.eta))),
        "volume_change": float(result.accepted_state.ledger.volume_change),
        "divergence_residual": float(result.accepted_state.ledger.divergence_residual),
        "kinematic_residual": float(result.accepted_state.ledger.kinematic_residual),
    }


if __name__ == "__main__":
    print(run())
