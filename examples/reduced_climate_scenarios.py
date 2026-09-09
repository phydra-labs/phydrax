"""Run with: python examples/reduced_climate_scenarios.py

Illustrative coefficients and synthetic calibration, not an assessed climate
projection. Scenario/configuration/member axes are ordinary nested JAX vmaps.
"""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.climate import (
    ClimateDrivers,
    GasBoxModel,
    MODEL_YEAR_SECONDS,
    Myhre1998Forcing,
    ReducedClimatePlan,
)
from phydrax.dynamics import TimeGrid
from phydrax.solver import FixedStepRolloutPlan


def run():
    with jax.enable_x64(True):
        gases = GasBoxModel(
            response_coefficients=((0.01, 0.2, 0.01), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        )
        plan = ReducedClimatePlan(
            gases=gases,
            forcing=Myhre1998Forcing(("aerosol_prescribed", "solar_prescribed")),
        )
        runtime = plan.prepare(
            TimeGrid(np.arange(41.0), time_id="illustrative-40-model-years"),
            seconds_per_time_unit=MODEL_YEAR_SECONDS,
        )
        initial = runtime.initial_state()
        emission = jnp.stack(
            (
                jnp.linspace(10.0, 3.0, 40),
                jnp.linspace(40.0, 20.0, 40),
                jnp.full((40,), 3.0),
            ),
            axis=-1,
        )
        external = jnp.stack((jnp.linspace(-0.5, -0.1, 40), jnp.zeros(40)), axis=-1)
        drivers = ClimateDrivers(
            emission,
            jnp.broadcast_to(gases.background, (40, 3)),
            jnp.zeros((40, 3)),
            external,
        )
        problem = runtime.problem(initial, drivers)
        rollout = FixedStepRolloutPlan(retention="final")

        def evaluate(scenario_scale, feedback, methane_lifetime):
            rates = problem.method.climate.plan.gases.decay_rates.at[1, 0].set(
                1.0 / methane_lifetime
            )
            varied = eqx.tree_at(
                lambda p: (
                    p.args.emissions,
                    p.method.climate.plan.energy.feedback,
                    p.method.climate.plan.gases.decay_rates,
                ),
                problem,
                (emission * scenario_scale, feedback, rates),
            )
            result = rollout.rollout(varied)
            return result.final_state.temperature[0], result.successful

        # These are explicit axes, not a second scenario or uncertainty engine.
        ensemble = eqx.filter_jit(
            jax.vmap(
                jax.vmap(
                    jax.vmap(evaluate, in_axes=(None, None, 0)), in_axes=(None, 0, None)
                ),
                in_axes=(0, None, None),
            )
        )
        temperatures, successful = ensemble(
            jnp.asarray((0.7, 1.0, 1.3)),
            jnp.asarray((1.0, 1.2, 1.4)),
            jnp.asarray((8.0, 9.3, 11.0)),
        )
        if not bool(jnp.all(successful)):
            raise RuntimeError(
                "An illustrative scenario/configuration/member was rejected."
            )

        # Calibrate one physical feedback against an explicitly synthetic
        # terminal observation, using the native rollout's JAX sensitivity.
        observed, valid_observation = evaluate(
            jnp.asarray(1.0), jnp.asarray(1.35), jnp.asarray(9.3)
        )

        def prediction(feedback):
            return evaluate(jnp.asarray(1.0), feedback, jnp.asarray(9.3))[0]

        def update(_, feedback):
            predicted, derivative = jax.value_and_grad(prediction)(feedback)
            return jnp.clip(feedback - (predicted - observed) / derivative, 0.3, 3.0)

        calibrated = eqx.filter_jit(
            lambda: jax.lax.fori_loop(0, 8, update, jnp.asarray(1.0))
        )()
        fitted, valid_fit = evaluate(jnp.asarray(1.0), calibrated, jnp.asarray(9.3))
        if (
            not bool(valid_observation & valid_fit)
            or abs(float(fitted - observed)) > 1.0e-7
        ):
            raise RuntimeError("Synthetic feedback calibration did not converge.")
        result = eqx.filter_jit(rollout.rollout)(problem)
        if not bool(result.successful):
            raise RuntimeError("The baseline climate scenario was rejected.")
        final = result.final_state
        gas_residual = (
            jnp.sum(final.boxes, axis=-1)
            + final.cumulative_sink
            - final.cumulative_emissions
        )
        energy_residual = (
            runtime.plan.energy.heat_content(final.temperature)
            - final.cumulative_forcing_energy
            + final.cumulative_outgoing_energy
        )
        report = {
            "scope": "illustrative reduced CO2/CH4/N2O model; synthetic observation",
            "axes": ["scenario", "configuration", "member"],
            "terminal_temperature_K": np.asarray(temperatures).tolist(),
            "member_mean_K": np.asarray(jnp.mean(temperatures, axis=-1)).tolist(),
            "member_standard_deviation_K": np.asarray(
                jnp.std(temperatures, axis=-1)
            ).tolist(),
            "synthetic_feedback_W_m2_K": 1.35,
            "calibrated_feedback_W_m2_K": float(calibrated),
            "gas_budget_max_inventory_units": float(jnp.max(jnp.abs(gas_residual))),
            "energy_budget_J_m2": float(energy_residual * MODEL_YEAR_SECONDS),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return report


if __name__ == "__main__":
    run()
