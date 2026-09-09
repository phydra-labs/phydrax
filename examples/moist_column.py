#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run a heated, evaporating, radiating moist column with a real rain reservoir.

From the worktree: PYTHONPATH=. python examples/moist_column.py --steps 120
"""

import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.applications.atmosphere import MoistColumnPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument(
        "--restart", help="Optional output path for a plan-bound native checkpoint."
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    plan = MoistColumnPlan(mixing_rate=2e-5, forcing_cadence=6)
    initial = plan.initialize(
        jnp.asarray((0.7, 0.85, 1.0, 1.15)),
        jnp.asarray((245.0, 260.0, 275.0, 290.0)),
        jnp.asarray((0.008, 0.012, 0.016, 0.018)),
        jnp.full((4,), 150.0),
        reservoir_water=50.0,
        reservoir_energy=1e8,
    )
    advance = eqx.filter_jit(
        lambda state: plan.advance(
            state,
            args.dt,
            args.steps,
            heating_rate=jnp.asarray((0.0, 0.0, 0.0, 20.0)),
            surface_vapor_flux=2e-5,
            surface_energy_flux=10.0,
        )
    )
    final, accepted = advance(initial)
    if not bool(jnp.all(accepted)):
        raise RuntimeError(
            "Column rejected a step; reduce forcing or timestep and inspect budgets."
        )
    diagnosed = plan.diagnose(final)
    if not bool(jnp.all(diagnosed.successful)):
        raise RuntimeError("Final column thermodynamic certification failed.")
    print(
        json.dumps(
            {
                "elapsed_seconds": float(final.time),
                "accepted_steps": int(final.step_count),
                "temperature_K": diagnosed.temperature.tolist(),
                "vapor_fraction": diagnosed.vapor.tolist(),
                "liquid_fraction": diagnosed.liquid.tolist(),
                "ice_fraction": diagnosed.ice.tolist(),
                "reservoir_water_kg_m2": float(final.reservoir_water),
                "reservoir_energy_J_m2": float(final.reservoir_energy),
                "environment_energy_J_m2": float(final.environment_energy),
                "closed_water_residual_kg_m2": float(
                    final.total_water - initial.total_water
                ),
                "forced_energy_residual_J_m2": float(
                    final.total_energy - initial.total_energy - final.external_energy
                ),
                "cadence_phase": int(final.cadence_phase),
            },
            indent=2,
        )
    )
    if args.restart:
        plan.save_checkpoint(args.restart, final)


if __name__ == "__main__":
    main()
