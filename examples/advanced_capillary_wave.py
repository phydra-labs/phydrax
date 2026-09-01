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
    provider = phx.equations.IncidentWavePlan(
        (phx.equations.WaveComponent(1.0e-4, 1.0),), 1.0
    )
    wave = phx.applications.hydrodynamics.WaveForcingPlan(
        provider,
        jnp.zeros((4, 4, 3)).at[:2].set(0.5),
        jnp.zeros((4, 4, 3)).at[-2:].set(0.25),
        active_gain=0.1,
    )
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        reference, jnp.full((4, 4), -1.0), maximum_slope=0.8
    )
    hydro = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        surface_tension=0.072,
        wave=wave,
        coupling_iterations=5,
        coupling_tolerance=1.0e-7,
    ).prepare()
    state = hydro.initial_state(jnp.zeros((4, 4)))
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )
    result = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(hydro).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.0001),
        None,
    )
    view = phx.applications.hydrodynamics.free_surface_diagnostic_view(
        hydro, result.accepted_state
    )
    return {
        "successful": bool(result.successful),
        "surface_energy": float(view.surface_energy),
        "wave_reflection": float(view.wave_reflection_coefficient),
        "capillary_dual_residual": float(
            result.accepted_state.ledger.capillary_dual_residual
        ),
    }


if __name__ == "__main__":
    print(run())
