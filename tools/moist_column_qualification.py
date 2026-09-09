#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measured moist-column rollout, conservation, branch derivatives, and restart.

PYTHONPATH=. python tools/moist_column_qualification.py --steps 120 --layers 8
Reports observations, not a release or universal scientific qualification claim.
"""

import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.atmosphere import MoistColumnPlan, MoistThermodynamicPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.layers, args.steps, args.repeats) < 1:
        parser.error("layers, steps, and repeats must be positive")
    jax.config.update("jax_enable_x64", True)
    thermo = MoistThermodynamicPlan()
    plan = MoistColumnPlan(thermo, mixing_rate=2e-5, forcing_cadence=5)
    initial = plan.initialize(
        jnp.linspace(0.7, 1.2, args.layers),
        jnp.linspace(245.0, 292.0, args.layers),
        jnp.linspace(0.008, 0.020, args.layers),
        150.0,
        reservoir_water=50.0,
        reservoir_energy=1e8,
    )
    heating = jnp.zeros(args.layers).at[-1].set(20.0)
    run = eqx.filter_jit(
        lambda state: plan.advance(
            state,
            10.0,
            args.steps,
            heating_rate=heating,
            surface_vapor_flux=2e-5,
            surface_energy_flux=10.0,
        )
    )
    start = time.perf_counter()
    final, successful = run(initial)
    jax.block_until_ready(final.internal_energy)
    compile_and_first = time.perf_counter() - start
    elapsed = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        final, successful = run(initial)
        jax.block_until_ready(final.internal_energy)
        elapsed.append(time.perf_counter() - start)
    if not bool(jnp.all(successful)):
        raise RuntimeError("Scientific rollout rejected at least one step.")
    water_residual = float(final.total_water - initial.total_water)
    energy_residual = float(
        final.total_energy - initial.total_energy - final.external_energy
    )
    np.testing.assert_allclose(water_residual, 0.0, atol=2e-9)
    np.testing.assert_allclose(energy_residual, 0.0, atol=2e-5)
    derivative_errors = {}
    for name, temperature, qt in (
        ("ice", 250.0, 0.018),
        ("liquid", 290.0, 0.025),
        ("vapor", 305.0, 0.002),
    ):
        equilibrium = thermo.equilibrium(1.0, temperature, qt)
        energy = thermo.energy(
            1.0, temperature, equilibrium.vapor, equilibrium.liquid, equilibrium.ice
        )
        observable = lambda e: thermo.adjust(1.0, qt, e).temperature
        result = thermo.adjust(1.0, qt, energy)
        if not bool(result.successful & result.derivative_valid):
            raise RuntimeError(
                f"Derivative state is not certified branch-regular: {name}."
            )
        automatic = jax.grad(observable)(energy)
        numerical = (observable(energy + 0.1) - observable(energy - 0.1)) / 0.2
        relative_error = float(
            jnp.abs(automatic - numerical) / jnp.maximum(jnp.abs(numerical), 1e-12)
        )
        np.testing.assert_allclose(automatic, numerical, rtol=2e-5, atol=2e-8)
        derivative_errors[name] = relative_error
    # A restart at nonzero cadence must use saved forcing, not new call arguments.
    first = plan.step(
        initial,
        10.0,
        heating_rate=heating,
        surface_vapor_flux=2e-5,
        surface_energy_flux=10.0,
    )
    if not bool(first.successful):
        raise RuntimeError("Restart precursor was rejected.")
    with TemporaryDirectory(dir=".") as directory:
        checkpoint = plan.save_checkpoint(Path(directory) / "column.phx", first.state)
        restart = plan.load_checkpoint(checkpoint)
    resumed = plan.step(restart, 10.0, heating_rate=1000.0)
    continued = plan.step(
        first.state,
        10.0,
        heating_rate=heating,
        surface_vapor_flux=2e-5,
        surface_energy_flux=10.0,
    )
    if not bool(resumed.successful & continued.successful):
        raise RuntimeError("Restart continuation was rejected.")
    for actual, expected in zip(
        jax.tree.leaves(resumed.state), jax.tree.leaves(continued.state)
    ):
        np.testing.assert_array_equal(actual, expected)
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "layers": args.layers,
                "steps": args.steps,
                "compile_and_first_seconds": compile_and_first,
                "warm_rollout_seconds": elapsed,
                "median_layer_steps_per_second": args.layers
                * args.steps
                / float(np.median(elapsed)),
                "water_residual_kg_m2": water_residual,
                "energy_residual_J_m2": energy_residual,
                "temperature_energy_derivative_relative_error": derivative_errors,
                "restart_identical": True,
                "forcing_cadence_phase": int(resumed.state.cadence_phase),
                "scope": "fixed-volume constant-caloric column; regular-branch derivatives only",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
