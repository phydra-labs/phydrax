# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""EPW -> receding-horizon native RC control -> storage dispatch -> AC audit."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.energy._artifacts import (
    archive_metrics,
    archive_workflow,
    execution_identity,
)
from phydrax.applications import building_energy as be, energy_planning as ep, power
from phydrax.applications.thermofluids import ConstantCOPHeatPumpLaw
from phydrax.units import (
    conversion_factor,
    ENERGY,
    JOULE,
    SECOND,
    SI_REFERENCE_SYSTEM_ID,
    TIME,
    UnitDefinition,
)


KWH = UnitDefinition("kWh", ENERGY, SI_REFERENCE_SYSTEM_ID, 3600000)
HOUR = UnitDefinition("h", TIME, SI_REFERENCE_SYSTEM_ID, 3600)
CAPACITY = 2e6  # Effective building capacity, J/K; explicitly authored reduction.
CONDUCTANCE = 60.0  # Envelope sensible conductance, W/K.
COP = 3.0


def synthetic_epw_24h():
    """Independently authored winter day, not measured weather or an engine output."""
    output = io.StringIO()
    rows = csv.writer(output, lineterminator="\n")
    rows.writerows(
        [
            (
                "LOCATION",
                "AuthoredWinterDay",
                "NA",
                "Synthetic",
                "PHYDRA-authored",
                "000000",
                45,
                0,
                0,
                100,
            ),
            ("DESIGN CONDITIONS", 0),
            ("TYPICAL/EXTREME PERIODS", 0),
            ("GROUND TEMPERATURES", 0),
            ("HOLIDAYS/DAYLIGHT SAVINGS", "No", 0, 0, 0),
            ("COMMENTS 1", "Synthetic analytic 24h forcing; not measured EPW weather"),
            (
                "COMMENTS 2",
                "Dry bulb = 2 + 4*cos(2*pi*(hour-15)/24) degC; no solar gain used",
            ),
            ("DATA PERIODS", 1, 1, "Authored day", "Monday", "1/1", "1/1"),
        ]
    )
    for hour in range(1, 25):
        dry = 2 + 4 * np.cos(2 * np.pi * (hour - 15) / 24)
        row = [
            2001,
            1,
            1,
            hour,
            60,
            "authored",
            f"{dry:.8f}",
            f"{dry - 3:.8f}",
            75,
            101325,
            0,
            0,
            280,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            180,
            2,
            0,
            0,
            20,
            77777,
            9,
            999999999,
            10,
            0.1,
            0,
            0,
            0.2,
            0,
            0,
        ]
        rows.writerow(row)
    return output.getvalue()


def watts_to_rate(watts, carrier, time_unit):
    """W -> energy-carrier amount / chronology time, with dimension checking."""
    return (
        np.asarray(watts)
        * float(conversion_factor(time_unit, SECOND))
        / float(conversion_factor(carrier.unit, JOULE))
    )


def rate_to_watts(rate, carrier, time_unit):
    return (
        np.asarray(rate)
        * float(conversion_factor(carrier.unit, JOULE))
        / float(conversion_factor(time_unit, SECOND))
    )


def make_dispatch_system(heat_W, electric_W, duration_s):
    heat = np.asarray(heat_W, dtype=float)
    electric = np.asarray(electric_W, dtype=float)
    duration = np.asarray(duration_s, dtype=float)
    if heat.ndim != 1 or electric.shape != heat.shape or duration.shape != heat.shape:
        raise ValueError(
            "Heat, electricity and physical interval durations must share one interval axis."
        )
    carriers = (
        ep.Carrier("electricity", KWH),
        ep.Carrier("heat", KWH),
        ep.Carrier("ambient", KWH, environmental=True),
    )
    prices = np.where(np.arange(heat.size) % 2 == 0, 0.10, 0.40)
    return ep.EnergySystem(
        ep.Chronology(
            (ep.Horizon("day", duration / float(conversion_factor(HOUR, SECOND))),),
            time_unit=HOUR,
        ),
        carriers,
        (
            ep.BalancePoint("meter", "electricity"),
            ep.BalancePoint("room", "heat"),
            ep.BalancePoint("outdoor-source", "ambient"),
        ),
        sources=(
            ep.Source("grid", "meter", 6.0, marginal_cost=prices, emissions=0.2),
            ep.Source("ambient", "outdoor-source", 8.0),
        ),
        demands=(
            ep.Demand(
                "electric-demand", "meter", watts_to_rate(electric, carriers[0], HOUR)
            ),
            ep.Demand("space-heat", "room", watts_to_rate(heat, carriers[1], HOUR)),
        ),
        converters=(
            ep.Converter(
                "heat-pump",
                "meter",
                "input",
                (
                    ep.ConverterPort("meter", -1),
                    ep.ConverterPort("outdoor-source", -(COP - 1)),
                    ep.ConverterPort("room", COP),
                ),
                2.0,
            ),
        ),
        inventories=(
            ep.Inventory(
                "thermal-store",
                "room",
                2.0,
                1.5,
                1.5,
                (ep.InventoryBoundary("day", initial=0.0, target=0.0),),
                throughput_cost=0.005,
            ),
        ),
    )


def replay_dispatch_heat(spec, plan, expected_heat_W, *, tolerance_W=0.01):
    """Independent domain replay AND the cross-domain delivery boundary audit."""
    replay = ep.replay_energy_system(spec, plan, atol=1e-7, rtol=1e-7)
    if not replay.successful:
        raise RuntimeError(f"Independent dispatch replay failed: {replay.failures}")
    delivered_rate = (
        COP * np.asarray(plan.values("converter/heat-pump"))
        - np.asarray(plan.values("inventory/thermal-store/charge"))
        + np.asarray(plan.values("inventory/thermal-store/discharge"))
    )
    heat_carrier = next(carrier for carrier in spec.carriers if carrier.name == "heat")
    delivered = rate_to_watts(delivered_rate, heat_carrier, spec.chronology.time_unit)
    expected = np.asarray(expected_heat_W)
    if delivered.shape != expected.shape or not np.all(np.isfinite(expected)):
        raise ValueError(
            "Replayed heat and requested physical W require identical finite interval axes."
        )
    if np.max(np.abs(delivered - expected)) > tolerance_W:
        raise RuntimeError(
            "Planning-to-building heat boundary mismatch: carrier amount/time is not W."
        )
    return replay, delivered


def run_building_dispatch(output_dir, *, epw_path=None, intervals=4, execution=None):
    """Execute 2--4 hourly intervals; return real native results and archived arrays.

    EPW ending observations are held over their preceding intervals by an explicit
    offline reconstruction policy. This is a forecast specimen, not a causal
    estimator from future measurements. Grid dispatch is at the building meter;
    AC upstream generation separately accounts for feeder losses, without repair.
    """
    if intervals not in (2, 3, 4):
        raise ValueError("Choose 2, 3 or 4 intervals for this bounded workflow.")
    if not jax.config.x64_enabled:
        raise ValueError("This qualification requires JAX_ENABLE_X64=1.")
    execution = execution_identity() if execution is None else execution
    text = (
        synthetic_epw_24h()
        if epw_path is None
        else Path(epw_path).read_text(encoding="utf-8-sig")
    )
    weather = be.parse_epw(text, asset_id="workflow-weather")
    dry = weather.quantity("dry_bulb_temperature")
    if len(dry.samples.values) < intervals or not np.all(
        np.asarray(dry.samples.value_valid)[:intervals]
    ):
        raise ValueError(
            "The selected EPW horizon requires complete dry-bulb observations."
        )
    boundary = np.asarray(dry.samples.values[:intervals])[:, None]
    duration = np.full(intervals, 3600.0 / weather.records_per_hour)
    times = np.r_[0.0, np.cumsum(duration)]
    model = be.compile_building(
        be.BuildingSource(
            (be.Zone("room", capacity=CAPACITY, volume=60),),
            adjacencies=(
                be.Adjacency("envelope", "room", None, CONDUCTANCE, boundary_id="air"),
            ),
            boundaries=(be.BuildingBoundary("air", kind="ambient"),),
            source_id="authored-one-zone-2MJ-per-K",
            provenance=("explicit-effective-RC", "no-solar-no-moisture"),
        )
    )
    state = jnp.array([292.15])
    temperatures = [np.asarray(state)]
    base_heat = np.full((intervals, 1), 150.0)
    base_electric = np.full(
        intervals, 300.0
    )  # Separate load; NOT double-counted as internal heat.
    target = np.full((intervals, 1), 293.15)
    controls, applied_heat, control_results = [], [], []
    for index in range(intervals):
        stop = min(index + 2, intervals)
        count = stop - index
        result = be.optimize_hvac(
            model,
            state,
            times[index : stop + 1],
            boundary[index:stop],
            base_heat[index:stop],
            target[index:stop],
            heat_distribution=jnp.ones((1, 1)),
            conversion_law=ConstantCOPHeatPumpLaw(COP),
            supply_temperature=313.15,
            source_boundary_id="air",
            power_upper=1500.0,
            initial_power=jnp.full((count, 1), 500.0),
            power_scale=1000.0,
        )
        if not bool(result.successful):
            raise RuntimeError("Native receding-horizon HVAC solve failed.")
        step = model.step(
            state,
            boundary[index],
            base_heat[index] + result.delivered_heat[0],
            duration[index],
        )
        if not bool(step.successful):
            raise RuntimeError("Native applied RC step failed.")
        state = step.temperature
        temperatures.append(np.asarray(state))
        controls.append(float(result.electrical_power[0, 0]))
        applied_heat.append(float(result.delivered_heat[0, 0]))
        control_results.append(result)
    temperatures = np.asarray(temperatures)
    applied_heat = np.asarray(applied_heat)
    spec = make_dispatch_system(applied_heat, base_electric, duration)
    compiled = ep.compile_energy_system(spec)
    solution = ep.solve_energy_system(compiled, replay_atol=1e-7, replay_rtol=1e-7)
    if not solution.successful:
        raise RuntimeError(f"Native dispatch failed: {solution.replay.failures}")
    replay, delivered = replay_dispatch_heat(spec, solution.plan, applied_heat)
    physical = be.replay_building(
        model, temperatures[0], times, boundary, base_heat + delivered[:, None]
    )
    replay_error = float(np.max(np.abs(np.asarray(physical.temperature) - temperatures)))
    # Independent one-zone closed-form energy integral; not C*dT/dt evaluated by the model.
    equilibrium = boundary[:, 0] + (base_heat[:, 0] + delivered) / CONDUCTANCE
    decay = np.exp(-CONDUCTANCE * duration / CAPACITY)
    expected_next = equilibrium + (temperatures[:-1, 0] - equilibrium) * decay
    integrated_temperature = equilibrium * duration + (
        temperatures[:-1, 0] - equilibrium
    ) * CAPACITY / CONDUCTANCE * (1 - decay)
    envelope_out_J = CONDUCTANCE * (integrated_temperature - boundary[:, 0] * duration)
    stored_change_J = CAPACITY * np.diff(np.asarray(physical.temperature)[:, 0])
    thermal_balance_J = (
        (base_heat[:, 0] + delivered) * duration - envelope_out_J - stored_change_J
    )
    grid_W = rate_to_watts(
        solution.plan.values("source/grid"), spec.carriers[0], spec.chronology.time_unit
    )
    flows = []
    base_mva = 0.01  # Total three-phase 10 kVA base, not phase power.
    for demand in grid_W:
        pu = float(demand) / (base_mva * 1e6)
        network = power.PowerNetwork(
            (power.Bus("upstream", 0.4), power.Bus("meter", 0.4)),
            (power.Branch("feeder", "upstream", "meter", 0.005, 0.03, rate=1.4),),
            (power.Generator("grid", "upstream", p_min=0, p_max=2, q_min=-2, q_max=2),),
            (power.Load("meter-demand", "meter", pu, 0.2 * pu),),
            base_mva=base_mva,
        )
        study = power.PowerStudy(
            (power.BusControl("upstream", "reference"), power.BusControl("meter"))
        )
        flow = power.solve_power_flow(network, study=study)
        if not bool(flow.operationally_feasible):
            raise RuntimeError(
                f"Unmodified dispatch is AC-infeasible: {flow.status}; no repair is attempted."
            )
        flows.append(flow)
    ac_balance_pu = max(float(np.max(np.abs(flow.bus_balance))) for flow in flows)
    upstream_W = np.asarray(
        [float(flow.generator_power[0].real) * base_mva * 1e6 for flow in flows]
    )
    loss_W = np.asarray(
        [float(np.sum(flow.branch_loss.real)) * base_mva * 1e6 for flow in flows]
    )
    electrical_balance_J = (upstream_W - grid_W - loss_W) * duration
    store = np.asarray(solution.plan.values("inventory/thermal-store/state/day"))
    ambient_W = rate_to_watts(
        solution.plan.values("source/ambient"), spec.carriers[2], HOUR
    )
    store_change_J = np.diff(store) * float(conversion_factor(KWH, JOULE))
    global_balance_J = (
        (upstream_W + ambient_W + base_heat[:, 0] - base_electric - loss_W) * duration
        - envelope_out_J
        - stored_change_J
        - store_change_J
    )
    metrics = {
        "scope": (
            "one-zone sensible exact-frozen RC; 2-step MPC; continuous heat storage; "
            "meter-side dispatch; balanced AC feasibility"
        ),
        "intervals": intervals,
        "duration_seconds": duration.tolist(),
        "weather_sha256": weather.content_sha256,
        "weather_origin": dry.origin,
        "weather_timezone": dry.timezone,
        "weather_source": "independently-authored-synthetic-24h"
        if epw_path is None
        else str(Path(epw_path).resolve()),
        "weather_reconstruction": "hold each interval-ending dry-bulb observation over its preceding interval",
        "units": {
            "building": "K,W,J,s",
            "planning_amount": "kWh",
            "planning_time": "h",
            "rate_W_factor": 1000.0,
            "inventory_J_factor": 3600000.0,
            "power_base_total_three_phase_MVA": base_mva,
            "power_W_per_pu": base_mva * 1e6,
            "grid_boundary": "delivered meter; feeder losses upstream",
        },
        "control_mode": [result.mode for result in control_results],
        "native_control_result": [
            type(result.optimization).__name__ for result in control_results
        ],
        "planning_native_status": str(solution.native_result.status),
        "planning_replay_maximum_violation": float(replay.maximum_physical_violation),
        "temperature_replay_error_K": replay_error,
        "analytic_temperature_error_K": float(
            np.max(np.abs(expected_next - np.asarray(physical.temperature)[1:, 0]))
        ),
        "maximum_tracking_error_K": float(
            np.max(np.abs(np.asarray(physical.temperature)[1:] - target))
        ),
        "temperature_excursion_K": float(np.ptp(temperatures[:, 0])),
        "maximum_thermal_balance_error_J": float(np.max(np.abs(thermal_balance_J))),
        "maximum_electric_balance_error_J": float(np.max(np.abs(electrical_balance_J))),
        "maximum_global_balance_error_J": float(np.max(np.abs(global_balance_J))),
        "ledger_totals_J": {
            "upstream_electricity": float(upstream_W @ duration),
            "ambient_heat_input": float(ambient_W @ duration),
            "external_internal_gains": float(base_heat[:, 0] @ duration),
            "baseline_electricity_consumption": float(base_electric @ duration),
            "feeder_losses": float(loss_W @ duration),
            "envelope_heat_out": float(np.sum(envelope_out_J)),
            "building_storage_change": float(np.sum(stored_change_J)),
            "thermal_store_change": float(np.sum(store_change_J)),
        },
        "maximum_AC_bus_balance_pu": ac_balance_pu,
        "maximum_voltage_violation_pu": max(
            float(flow.voltage_violation) for flow in flows
        ),
        "maximum_branch_limit_violation_pu": max(
            float(flow.branch_limit_violation) for flow in flows
        ),
        "storage_peak_kWh": float(np.max(store)),
        "cost_currency": float(replay.cost),
        "emissions_kg": float(replay.emissions),
        "criteria": {
            "temperature_error_K": 1e-5,
            "balance_error_J": 1.0,
            "AC_balance_pu": 1e-6,
            "minimum_temperature_excursion_K": 0.1,
            "tracking_error_K": 0.05,
            "minimum_storage_kWh": 1e-4,
        },
    }
    metrics["passed"] = (
        bool(physical.successful)
        and replay_error <= 1e-5
        and metrics["analytic_temperature_error_K"] <= 1e-5
        and metrics["maximum_tracking_error_K"] <= 0.05
        and metrics["temperature_excursion_K"] >= 0.1
        and metrics["storage_peak_kWh"] >= 1e-4
        and metrics["maximum_thermal_balance_error_J"] <= 1.0
        and metrics["maximum_electric_balance_error_J"] <= 1.0
        and metrics["maximum_global_balance_error_J"] <= 1.0
        and ac_balance_pu <= 1e-6
    )
    arrays = {
        "time": times,
        "temperature": physical.temperature,
        "boundary_temperature": boundary,
        "heat_request": applied_heat,
        "delivered_heat": delivered,
        "baseline_electricity": base_electric,
        "mpc_electrical_power": controls,
        "meter_power": grid_W,
        "upstream_power": upstream_W,
        "feeder_loss": loss_W,
        "thermal_store": store,
        "envelope_out_energy": envelope_out_J,
        "stored_energy_change": stored_change_J,
        "thermal_balance": thermal_balance_J,
        "electrical_balance": electrical_balance_J,
        "global_balance": global_balance_J,
        "thermal_store_change": store_change_J,
        "ac_voltage": np.stack([np.asarray(flow.voltage) for flow in flows]),
        "ambient_heat": ambient_W,
    }
    units = {name: "W" for name in arrays}
    units.update(
        time="s",
        temperature="K",
        boundary_temperature="K",
        thermal_store="kWh",
        ac_voltage="pu",
        envelope_out_energy="J",
        stored_energy_change="J",
        thermal_balance="J",
        electrical_balance="J",
        global_balance="J",
        thermal_store_change="J",
    )
    for item in solution.plan.dispatch:
        name = "dispatch/" + item.name
        arrays[name] = item.values
        units[name] = "kWh" if "/state/" in item.name else "kWh/h"
    archives = archive_workflow(
        output_dir,
        "building-dispatch",
        metrics,
        arrays,
        units,
        {
            "temperature": physical.temperature[-1],
            "time": times[-1:],
            "thermal_store": store[-1:],
            "planning_primal": solution.native_result.primal,
        },
        execution=execution,
    )
    if not metrics["passed"]:
        raise RuntimeError(
            f"Building workflow acceptance failed: {json.dumps(metrics, allow_nan=False)}"
        )
    return {
        "metrics": metrics,
        "archives": archives,
        "execution": execution,
        "weather": weather,
        "model": model,
        "controls": tuple(control_results),
        "spec": spec,
        "solution": solution,
        "replay": replay,
        "physical_replay": physical,
        "power_flows": tuple(flows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("energy-results/building"))
    parser.add_argument("--epw", type=Path)
    parser.add_argument("--intervals", type=int, choices=(2, 3, 4), default=4)
    args = parser.parse_args()
    result = run_building_dispatch(
        args.output, epw_path=args.epw, intervals=args.intervals
    )
    print(
        json.dumps(
            {**result["metrics"], **archive_metrics(result["archives"])},
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
