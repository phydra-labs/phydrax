# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Run a small real global primitive-equation model with paired moist processes.

JAX_ENABLE_X64=1 python examples/global_atmosphere.py --moist --steps 2
"""

import argparse

import jax.numpy as jnp

from phydrax.applications.atmosphere import (
    BulkSurfaceExchangePlan,
    ColumnOpticalProperties,
    ColumnRadiationPlan,
    GlobalAtmosphereProcesses,
    GlobalPrimitiveEquationPlan,
    GlobalSurfacePhysics,
    MoistThermodynamicPlan,
    precondition_global_fluxes,
    WetSlabPlan,
)
from phydrax.applications.geophysics import HybridPressureCoordinate
from phydrax.discretization import SphericalSpectralPlan


def run(*, moist=False, interactive=False, steps=2):
    space = SphericalSpectralPlan(4, sampling="gl").prepare(radius=6.371e6)
    vertical = HybridPressureCoordinate([0.1, 0.05, 0.0], [0.0, 0.5, 1.0])
    processes = (
        GlobalAtmosphereProcesses(
            thermodynamics=MoistThermodynamicPlan(),
            radiative_timescale=30 * 86400.0,
            evaporation_flux=1e-5,
            sensible_heat_flux=10.0,
            cadence=3,
        )
        if moist
        else GlobalAtmosphereProcesses(held_suarez=True, cadence=3)
    )
    if interactive:
        thermo = MoistThermodynamicPlan()
        optics = ColumnOpticalProperties(
            shortwave_absorption=(1e-5, 0.002, 0.01, 0.01),
            shortwave_scattering=(0.0, 0.0, 0.1, 0.1),
            longwave_absorption=(1e-4, 0.03, 0.1, 0.1),
            reference_id="declared-synthetic-example-not-Earth-calibration",
        )
        processes = GlobalAtmosphereProcesses(
            thermodynamics=thermo,
            surface_physics=GlobalSurfacePhysics(
                WetSlabPlan(thermo),
                BulkSurfaceExchangePlan(stability="neutral"),
                ColumnRadiationPlan(optics),
            ),
            cadence=3,
        )
    model = GlobalPrimitiveEquationPlan(
        space,
        vertical,
        processes=processes,
        dt=20.0,
        water_limiter="conservative" if interactive else "reject",
        angular_momentum_projection="energy-neutral" if interactive else "none",
    ).prepare()
    theta = model.work_space.transform.theta[:, None, None]
    phi = model.work_space.transform.phi[None, :, None]
    continuation = model.initialize(
        temperature=280 + 0.1 * jnp.sin(theta) * jnp.cos(phi),
        east=2 * jnp.sin(theta),
        vapor=0.007 if moist else 0.002 if interactive else 0.0,
        liquid=0.001 if moist else 1e-5 if interactive else 0.0,
        ice=1e-5 if interactive else 0.0,
        surface_temperature=290.0,
    )
    preparation = precondition_global_fluxes(model, continuation) if interactive else None
    if preparation is not None:
        if not bool(preparation.successful):
            raise RuntimeError(
                "Interactive global flux preconditioning did not converge."
            )
        continuation = preparation.continuation
    initial = model.inventories(continuation.state)
    for _ in range(steps):
        result = model.advance(continuation)
        if not bool(result.evidence.accepted):
            raise RuntimeError(f"Global step rejected: {result.evidence}")
        continuation = result.continuation
    final = model.inventories(continuation.state)
    return {
        "moist": moist,
        "interactive_surface": interactive,
        "flux_preconditioning": None
        if preparation is None
        else {
            "air_temperature_offset_k": float(preparation.air_temperature_offset),
            "surface_temperature_offset_k": float(preparation.surface_temperature_offset),
            "initial_residual_w_m2": (
                preparation.initial_flux_residual_w_per_m2.tolist()
            ),
            "final_residual_w_m2": (preparation.final_flux_residual_w_per_m2.tolist()),
        },
        "surface_temperature_mean_k": None
        if not interactive
        else float(
            jnp.mean(
                processes.surface_physics.temperature(
                    continuation.state.surface_water,
                    continuation.state.surface_energy,
                    processes.thermodynamics,
                )
            )
        ),
        "closed_drift_w_m2": float(
            continuation.ledger.energy_residual
            / (4 * jnp.pi * space.radius**2 * continuation.time)
        ),
        "accepted_steps": int(continuation.accepted_steps),
        "time_seconds": float(continuation.time),
        "relative_mass_change": float((final[0] - initial[0]) / initial[0]),
        "relative_water_change": float((final[1] - initial[1]) / initial[1]),
        "relative_closed_energy_change": float((final[2] - initial[2]) / initial[2]),
        "maximum_wind": float(
            jnp.max(
                jnp.hypot(
                    model.view(continuation.state).east,
                    model.view(continuation.state).north,
                )
            )
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--moist", action="store_true")
    mode.add_argument("--interactive", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.steps < 1:
        parser.error("steps must be positive")
    print(
        run(
            moist=arguments.moist,
            interactive=arguments.interactive,
            steps=arguments.steps,
        )
    )
