#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical moist-column spinup, forcing interventions, refinement and restart.

PYTHONPATH=. python tools/interactive_moist_column_qualification.py --steps 600
For a diurnal-length preparation use --spinup-steps 43200 --dt 2. The report
records measured outcomes and validity; it does not certify climate realism.
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

from examples.interactive_moist_column import build_column, physical_report
from phydrax.applications.atmosphere._interactive_column import InteractiveMoistColumnPlan
from phydrax.metrix import EuclideanStateGeometry
from phydrax.solver._fixed_step import FixedStepProblem, FixedStepRolloutPlan


def _accepted(successful, role):
    if not bool(jnp.all(successful)):
        raise RuntimeError(
            f"{role} rejected a physical step; lower dt or inspect inventory/domain limits."
        )


def _same(a, b):
    return all(
        np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


def _roundoff_equivalent(a, b):
    maximum = 0.0
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        left, right = np.asarray(left), np.asarray(right)
        if left.dtype != right.dtype:
            return False, np.inf
        if left.dtype.kind in "fc":
            scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1.0)
            relative = float(np.max(np.abs(left - right))) / scale
            maximum = max(maximum, relative)
            if relative > 64 * np.finfo(left.dtype).eps:
                return False, maximum
        elif not np.array_equal(left, right):
            return False, np.inf
    return True, maximum


def precipitation_limits(thermo):
    plan = InteractiveMoistColumnPlan(
        thermo,
        mixing_length=0.0,
        condensation_timescale=1e12,
        rain_evaporation_timescale=1e12,
        phase_conversion_timescale=1e12,
        autoconversion_timescale=1e12,
        rain_fall_speed=5.0,
        snow_fall_speed=0.0,
    )
    vapor = thermo.saturation_pressure(290.0) * 100 / (thermo.vapor_gas_constant * 290.0)
    initial = plan.initialize(
        jnp.full(3, 100.0), vapor, 290.0, 100.0, rain_mass=jnp.asarray([0.03, 0.0, 0.0])
    )
    state = initial
    cumulative = []
    for _ in range(3):
        result = plan.step(state, 20.0)
        _accepted(result.successful, "rain time-of-flight")
        state = result.state
        cumulative.append(float(state.precipitated_water))
    np.testing.assert_allclose(cumulative, [0.0, 0.0, 0.03], atol=1e-12)
    evaporating = eqx.tree_at(
        lambda p: (p.rain_fall_speed, p.rain_evaporation_timescale),
        plan,
        (jnp.asarray(0.0), jnp.asarray(20.0)),
    )
    dry = evaporating.initialize(jnp.asarray([100.0]), 0.3, 290.0, 100.0, rain_mass=0.2)
    evaporation = evaporating.step(dry, 0.1)
    _accepted(evaporation.successful, "below-cloud evaporation")
    if not 0 < float(evaporation.rain_evaporated_water) < 0.2:
        raise RuntimeError(
            "Below-cloud evaporation did not remain a finite partial transfer."
        )
    np.testing.assert_array_equal(evaporation.state.internal_energy, dry.internal_energy)
    return {
        "time_s": [20.0, 40.0, 60.0],
        "cumulative_surface_rain_kg_m2": cumulative,
        "expected_first_arrival_s": 60.0,
        "below_cloud_evaporated_kg_m2": float(evaporation.rain_evaporated_water),
        "below_cloud_temperature_change_K": float(
            evaporating.diagnose(evaporation.state).temperature[0] - 290.0
        ),
        "phase_change_internal_energy_residual_J_m2": float(
            jnp.sum(evaporation.state.internal_energy - dry.internal_energy)
        ),
    }


def short_derivative(plan):
    plan = eqx.tree_at(lambda p: p.mixing_length, plan, jnp.asarray(0.0))
    initial = plan.initialize(
        jnp.asarray([100.0, 110.0]),
        jnp.asarray([0.2, 0.4]),
        jnp.asarray([285.0, 290.0]),
        100.0,
        surface_temperature=295.0,
    )

    def observable(scale):
        varied = eqx.tree_at(
            lambda p: p.radiation.longwave_absorption_scale,
            plan,
            plan.radiation.longwave_absorption_scale.at[1].set(scale),
        )
        state = initial
        regular = jnp.asarray(True)
        for _ in range(3):
            result = varied.step(state, 1.0, solar_down=340.0, wind_speed=5.0)
            state, regular = result.state, regular & result.derivative_valid
        view = varied.diagnose(state)
        return view.surface_temperature + jnp.mean(view.temperature), regular

    center, regular = observable(jnp.asarray(1.0))
    if not bool(regular):
        raise RuntimeError(
            "Short derivative fixture crossed a declared nonsmooth/invalid branch."
        )
    automatic = jax.grad(lambda s: observable(s)[0])(jnp.asarray(1.0))
    rows = []
    for epsilon in (0.1, 0.01, 0.001, 0.0001):
        plus, plus_valid = observable(1.0 + epsilon)
        minus, minus_valid = observable(1.0 - epsilon)
        numerical = (plus - minus) / (2 * epsilon)
        rows.append(
            {
                "epsilon": epsilon,
                "finite_difference": float(numerical),
                "absolute_error": float(jnp.abs(numerical - automatic)),
                "both_endpoints_derivative_valid": bool(plus_valid & minus_valid),
            }
        )
    np.testing.assert_allclose(
        rows[-2]["finite_difference"], automatic, rtol=2e-4, atol=2e-9
    )
    return {
        "observable": "SST plus mean air temperature after 3 seconds [K]",
        "parameter": "longwave vapor absorption multiplier",
        "value_K": float(center),
        "automatic_derivative_K": float(automatic),
        "derivative_valid": bool(regular),
        "epsilon_sweep": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--spinup-steps", type=int, default=0)
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--solar-change", type=float, default=40.0)
    parser.add_argument("--wind-factor", type=float, default=1.5)
    args = parser.parse_args()
    if (
        args.layers < 2
        or args.steps < 2
        or args.spinup_steps < 0
        or args.dt <= 0
        or args.wind_factor < 0
    ):
        parser.error("layers/steps>=2, spinup>=0, dt>0, wind-factor>=0 are required")
    jax.config.update("jax_enable_x64", True)
    plan, initial = build_column(args.layers)
    forcing = dict(solar_down=340.0, wind_speed=5.0, ventilation=0.05, shear=0.01)
    run = eqx.filter_jit(
        lambda state, dt, steps, forcing: plan.advance(state, dt, steps, **forcing)
    )
    if args.spinup_steps:
        initial, success = run(initial, args.dt, args.spinup_steps, forcing)
        _accepted(success, "spinup")
    start = time.perf_counter()
    baseline, success = run(initial, args.dt, args.steps, forcing)
    jax.block_until_ready(baseline.internal_energy)
    elapsed = time.perf_counter() - start
    _accepted(success, "baseline")
    half, success = run(initial, args.dt / 2, args.steps * 2, forcing)
    _accepted(success, "half-step refinement")
    quarter, success = run(initial, args.dt / 4, args.steps * 4, forcing)
    _accepted(success, "quarter-step refinement")
    temperatures = [plan.diagnose(s).temperature for s in (baseline, half, quarter)]
    coarse_error = float(jnp.sqrt(jnp.mean((temperatures[0] - temperatures[1]) ** 2)))
    fine_error = float(jnp.sqrt(jnp.mean((temperatures[1] - temperatures[2]) ** 2)))
    interventions = {}
    for name, changed in (
        ("solar", dict(forcing, solar_down=340.0 + args.solar_change)),
        ("wind", dict(forcing, wind_speed=5.0 * args.wind_factor)),
    ):
        result, success = run(initial, args.dt, args.steps, changed)
        _accepted(success, name + " intervention")
        interventions[name] = {
            "forcing": changed,
            "SST_change_from_baseline_K": float(
                plan.diagnose(result).surface_temperature
                - plan.diagnose(baseline).surface_temperature
            ),
            "precipitation_change_from_baseline_kg_m2": float(
                result.precipitated_water - baseline.precipitated_water
            ),
            "evaporation_change_from_baseline_kg_m2": float(
                result.evaporated_water - baseline.evaporated_water
            ),
            "physical_outputs": physical_report(plan, initial, result, args.dt, changed),
        }
    split = args.steps // 2
    prefix, success = run(initial, args.dt, split, forcing)
    _accepted(success, "restart prefix")
    with TemporaryDirectory() as directory:
        checkpoint = plan.save_checkpoint(Path(directory) / "column.phx", prefix)
        restart = plan.load_checkpoint(checkpoint)
    resumed, success = run(restart, args.dt, args.steps - split, forcing)
    _accepted(success, "restart continuation")
    if not _same(resumed, baseline):
        raise RuntimeError("Restart changed the exact continuation state.")
    # Exercise the native fixed-step retention/replay substrate, not a second
    # atmosphere-specific execution framework.
    native_problem = FixedStepProblem(
        plan.fixed_step_method(),
        initial,
        t0=float(initial.time),
        t1=float(initial.time) + 2 * args.dt,
        step_size=args.dt,
        args=forcing,
        state_geometry=EuclideanStateGeometry(),
    )
    native = FixedStepRolloutPlan(retention="final").rollout(native_problem)
    _accepted(native.successful, "native fixed-step rollout")
    direct, success = run(initial, args.dt, 2, forcing)
    _accepted(success, "native comparison")
    native_equivalent, native_difference = _roundoff_equivalent(
        native.final_state, direct
    )
    if not native_equivalent:
        raise RuntimeError("Native fixed-step adapter changed accepted physics.")
    baseline_report = physical_report(plan, initial, baseline, args.dt, forcing)
    np.testing.assert_allclose(
        baseline_report["closed_water_residual_kg_m2"], 0.0, atol=2e-8
    )
    np.testing.assert_allclose(baseline_report["energy_residual_J_m2"], 0.0, atol=2e-4)
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "layers": args.layers,
                "spinup_steps": args.spinup_steps,
                "steps": args.steps,
                "dt_s": args.dt,
                "compile_and_baseline_seconds": elapsed,
                "baseline": baseline_report,
                "interventions": interventions,
                "refinement": {
                    "same_physical_duration_s": args.steps * args.dt,
                    "coarse_half_temperature_RMS_K": coarse_error,
                    "half_quarter_temperature_RMS_K": fine_error,
                    "observed_difference_ratio": coarse_error / fine_error
                    if fine_error > 0
                    else None,
                    "nominal_order": 1,
                    "claim": "Measured differences, not unconditional asymptotic-order certification.",
                },
                "precipitation_limits": precipitation_limits(plan.thermodynamics),
                "short_derivative": short_derivative(plan),
                "exact_restart": True,
                "native_fixed_step_roundoff_equivalent": native_equivalent,
                "native_fixed_step_max_scaled_error": native_difference,
                "scope": (
                    "Forced fixed-volume caloric column; illustrative optics; no "
                    "climate calibration or momentum/gravity energy closure."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
