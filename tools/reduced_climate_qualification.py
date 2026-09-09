"""Analytical limits, budgets, nonlinear refinement, and native restart timing.

Run: python tools/reduced_climate_qualification.py
The report is generated from this execution, not a stored release claim.
"""

import json
import tempfile
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.climate import (
    ClimateDrivers,
    GasBoxModel,
    MODEL_YEAR_SECONDS,
    MultilayerEnergyBalance,
    read_reduced_climate_checkpoint,
    ReducedClimatePlan,
    write_reduced_climate_checkpoint,
)
from phydrax.dynamics import TimeGrid
from phydrax.solver import FixedStepRolloutPlan


def _problem(plan, count, end_time=20.0):
    runtime = plan.prepare(
        TimeGrid(
            np.linspace(0.0, end_time, count + 1), time_id=f"qualification-{count}-steps"
        ),
        seconds_per_time_unit=MODEL_YEAR_SECONDS,
    )
    drivers = ClimateDrivers(
        jnp.broadcast_to(jnp.asarray((10.0, 20.0, 3.0)), (count, 3)),
        jnp.broadcast_to(plan.gases.background, (count, 3)),
        jnp.zeros((count, 3)),
        jnp.zeros((count, 0)),
    )
    initial = runtime.initial_state()
    return runtime, initial, drivers


def qualify():
    with jax.enable_x64(True):
        gas = GasBoxModel(((1.0,), (1.0,), (1.0,)), ((0.0,), (1.0e-14,), (0.2,)))
        initial_boxes = jnp.asarray(((2.0,), (3.0,), (4.0,)))
        emission = jnp.asarray((1.0, 2.0, 3.0))
        gas_result = gas.advance(
            initial_boxes,
            jnp.zeros(3),
            jnp.asarray(0.0),
            jnp.asarray(5.0),
            emission,
            gas.background,
            ("emissions",) * 3,
        )
        expected_gas = np.asarray(
            (7.0, 13.0, 4.0 * np.exp(-1.0) + 15.0 * (1.0 - np.exp(-1.0)))
        )
        gas_error = float(jnp.max(jnp.abs(gas_result.boxes[:, 0] - expected_gas)))
        inverse = gas.advance(
            initial_boxes,
            jnp.zeros(3),
            jnp.asarray(0.0),
            jnp.asarray(5.0),
            jnp.zeros(3),
            gas.concentration(gas_result.boxes),
            ("concentration",) * 3,
        )
        inverse_error = float(jnp.max(jnp.abs(inverse.emissions - emission)))
        thermal = MultilayerEnergyBalance((8.0,), (), feedback=1.2)
        thermal_action = eqx.filter_jit(thermal.advance)
        started = time.perf_counter()
        thermal_result = thermal_action(
            jnp.asarray((0.7,)), jnp.asarray(20.0), jnp.asarray(3.0)
        )
        jax.block_until_ready(thermal_result.temperature)
        thermal_cold_seconds = time.perf_counter() - started
        expected_temperature = 3.0 / 1.2 + (0.7 - 3.0 / 1.2) * np.exp(-1.2 * 20.0 / 8.0)
        thermal_error = abs(float(thermal_result.temperature[0]) - expected_temperature)
        started = time.perf_counter()
        for _ in range(20):
            hot = thermal_action(jnp.asarray((0.7,)), jnp.asarray(20.0), jnp.asarray(3.0))
            jax.block_until_ready(hot.temperature)
        thermal_hot_seconds = (time.perf_counter() - started) / 20

        plan = ReducedClimatePlan(
            gases=GasBoxModel(
                response_coefficients=(
                    (0.05, 0.3, 0.06),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                )
            )
        )
        runtime, initial, drivers = _problem(plan, 40)
        rollout = eqx.filter_jit(FixedStepRolloutPlan(retention="final").rollout)
        started = time.perf_counter()
        full = rollout(runtime.problem(initial, drivers))
        jax.block_until_ready(full.final_state.temperature)
        rollout_cold_seconds = time.perf_counter() - started
        started = time.perf_counter()
        full = rollout(runtime.problem(initial, drivers))
        jax.block_until_ready(full.final_state.temperature)
        rollout_hot_seconds = time.perf_counter() - started
        first = rollout(runtime.problem(initial, drivers, stop_step=20))
        with tempfile.TemporaryDirectory(prefix="phydrax-reduced-climate-") as directory:
            path = Path(directory) / "climate.chk"
            write_reduced_climate_checkpoint(path, runtime, first.final_state, drivers)
            restored = read_reduced_climate_checkpoint(path, runtime, initial, drivers)
            restarted = rollout(runtime.problem(restored, drivers, start_step=20))
        restart_exact = all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(
                jax.tree.leaves(full.final_state),
                jax.tree.leaves(restarted.final_state),
                strict=True,
            )
        )
        final = full.final_state
        gas_budget = float(
            jnp.max(
                jnp.abs(
                    jnp.sum(final.boxes, axis=-1)
                    + final.cumulative_sink
                    - final.cumulative_emissions
                )
            )
        )
        energy_budget = float(
            jnp.abs(
                plan.energy.heat_content(final.temperature)
                - final.cumulative_forcing_energy
                + final.cumulative_outgoing_energy
            )
        )
        solutions = {}
        successful = bool(
            gas_result.successful
            & inverse.successful
            & thermal_result.successful
            & full.successful
            & first.successful
            & restarted.successful
        )
        for count in (20, 80, 320):
            refined_runtime, refined_initial, refined_drivers = _problem(plan, count)
            refined = rollout(refined_runtime.problem(refined_initial, refined_drivers))
            successful = successful and bool(refined.successful)
            solutions[count] = float(refined.final_state.temperature[0])
        solutions[40] = float(final.temperature[0])
        refinement_errors = [
            abs(solutions[count] - solutions[320]) for count in (20, 40, 80)
        ]
        refinement_ok = (
            refinement_errors[1] < 0.8 * refinement_errors[0]
            and refinement_errors[2] < 0.8 * refinement_errors[1]
        )
        qualified = (
            successful
            and restart_exact
            and refinement_ok
            and gas_error < 1.0e-10
            and inverse_error < 1.0e-10
            and thermal_error < 1.0e-11
            and gas_budget < 1.0e-9
            and energy_budget < 1.0e-9
        )
        report = {
            "scope": "analytical limits and internal invariants; no external model parity claim",
            "successful": qualified,
            "analytical_gas_max_error_inventory_units": gas_error,
            "inverse_roundtrip_max_error_inventory_per_year": inverse_error,
            "analytical_one_layer_error_K": thermal_error,
            "gas_budget_max_error_inventory_units": gas_budget,
            "energy_budget_error_W_year_m2": energy_budget,
            "native_restart_bitwise_equal": restart_exact,
            "state_dependent_temperature_refinement_steps": [20, 40, 80],
            "state_dependent_temperature_errors_against_320_steps_K": refinement_errors,
            "one_layer_cold_seconds": thermal_cold_seconds,
            "one_layer_hot_seconds": thermal_hot_seconds,
            "forty_step_rollout_cold_seconds": rollout_cold_seconds,
            "forty_step_rollout_hot_seconds": rollout_hot_seconds,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        if not qualified:
            raise RuntimeError(
                "Reduced climate qualification failed; inspect measured evidence above."
            )
        return report


if __name__ == "__main__":
    qualify()
