"""Run a dry atmospheric column or Cartesian vertical-slice reference case.

Run from the repository: JAX_ENABLE_X64=1 python -m examples.dry_atmosphere
These inviscid, compressible cases qualify this Cartesian method, not global
circulation or a moist atmospheric model.
"""

from __future__ import annotations

import argparse
import json

import equinox as eqx
import jax.numpy as jnp

from phydrax.applications.atmosphere._dry import (
    DryAtmospherePlan,
    DryHydrostaticReference,
)


def build_case(case="rising_thermal", *, nx=32, nz=16):
    if case == "column":
        plan = DryAtmospherePlan((nz,), ((0.0,), (10000.0,)))
    elif case == "gravity_wave":
        plan = DryAtmospherePlan(
            (nx, nz),
            ((0.0, 0.0), (300000.0, 10000.0)),
            reference=DryHydrostaticReference("isothermal", temperature=300.0),
        )
    elif case == "rising_thermal":
        plan = DryAtmospherePlan(
            (nx, nz),
            ((0.0, 0.0), (20000.0, 10000.0)),
            reference=DryHydrostaticReference("isentropic", temperature=300.0),
        )
    elif case == "density_current":
        plan = DryAtmospherePlan(
            (nx, nz),
            ((0.0, 0.0), (25600.0, 6400.0)),
            reference=DryHydrostaticReference("isentropic", temperature=300.0),
            boundaries=(("closed", "closed"), ("closed", "closed")),
        )
    else:
        raise ValueError("Unknown dry atmospheric case.")
    prepared = plan.prepare()
    if case == "column":
        return prepared, prepared.initial_state()
    points = prepared.balance.discretization.cell_centers
    x, z = points[..., 0], points[..., 1]
    if case == "gravity_wave":
        # Small pressure-balanced buoyancy perturbation of a stably stratified
        # isothermal atmosphere; no analytic compressible-wave claim is made.
        delta = (
            0.01 * jnp.sin(jnp.pi * z / 10000.0) / (1.0 + ((x - 150000.0) / 5000.0) ** 2)
        )
        velocity = jnp.broadcast_to(jnp.asarray((20.0, 0.0)), x.shape + (2,))
    else:
        center_x, center_z, radius_x, radius_z, amplitude = (
            (10000.0, 2000.0, 2000.0, 2000.0, 2.0)
            if case == "rising_thermal"
            else (12800.0, 3000.0, 4000.0, 2000.0, -15.0)
        )
        radius = jnp.sqrt(
            ((x - center_x) / radius_x) ** 2 + ((z - center_z) / radius_z) ** 2
        )
        delta = jnp.where(
            radius < 1.0, 0.5 * amplitude * (1.0 + jnp.cos(jnp.pi * radius)), 0.0
        )
        velocity = None
    return prepared, prepared.thermal_state(delta, velocity=velocity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("column", "gravity_wave", "rising_thermal", "density_current"),
        default="rising_thermal",
    )
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    prepared, state = build_case(args.case, nx=args.nx, nz=args.nz)
    dt = 0.5 * eqx.filter_jit(prepared.stable_step)(state)
    result = eqx.filter_jit(prepared.rollout)(state, jnp.full((args.steps,), dt))
    velocity = (
        result.state.conserved[..., prepared.system.momentum_slice]
        / prepared.system.density(result.state.conserved)[..., None]
    )
    print(
        json.dumps(
            {
                "case": args.case,
                "successful": bool(result.successful),
                "accepted_steps": int(result.state.accepted_steps),
                "time_s": float(result.state.time),
                "step_s": float(dt),
                "maximum_vertical_velocity_m_s": float(jnp.max(velocity[..., -1])),
                "minimum_vertical_velocity_m_s": float(jnp.min(velocity[..., -1])),
                "relative_total_energy_closure": float(
                    result.budget.closure[-1] / result.budget.total_energy
                ),
                "species_mass_closure_kg": [
                    float(x)
                    for x in result.budget.closure[: prepared.system.species_count]
                ],
            },
            indent=2,
        )
    )
    if not bool(result.successful):
        raise RuntimeError(
            "Prescribed atmospheric schedule was rejected; no partial success is claimed."
        )


if __name__ == "__main__":
    main()
