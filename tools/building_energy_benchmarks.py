#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Analytic reduced-building qualification and optional pinned external references."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.building_energy import (
    Adjacency,
    BuildingBoundary,
    BuildingExperiment,
    BuildingSource,
    calibrate_building,
    compile_building,
    energyplus_adiabatic_reference,
    energyplus_reference_weather,
    EnergyPlusVariable,
    optimize_hvac,
    parse_energyplus_csv,
    produce_uniform_sky_reference,
    replay_building,
    VentilationExchange,
    Zone,
)
from phydrax.applications.thermofluids import ResistiveHeatingLaw
from phydrax.interchange.energy_runtime import pin_energy_executable
from phydrax.optim import OptimizationTermination
from phydrax.units import derived_unit, JOULE, KELVIN, SECOND


def run_native():
    sources = (
        BuildingSource(
            (Zone("a", 10000),),
            adjacencies=(Adjacency("out", "a", None, 10),),
            source_id="one-zone",
        ),
        BuildingSource(
            (Zone("a", 10000), Zone("b", 10000)),
            adjacencies=(Adjacency("wall", "a", "b", 10),),
            source_id="two-zone",
        ),
        BuildingSource(
            (Zone("a", 10000), Zone("junction", 0, massless=True)),
            adjacencies=(
                Adjacency("inside", "a", "junction", 20),
                Adjacency("outside", "junction", None, 20),
            ),
            source_id="massless",
        ),
    )
    initial = (jnp.array([300.0]), jnp.array([300.0, 280.0]), jnp.array([300.0, 290.0]))
    expected = (
        jnp.array([280 + 20 * np.exp(-0.5)]),
        jnp.array([290 + 10 * np.exp(-1), 290 - 10 * np.exp(-1)]),
        jnp.array([280 + 20 * np.exp(-0.5), 280 + 10 * np.exp(-0.5)]),
    )
    output = {}
    for source, state, target in zip(sources, initial, expected, strict=True):
        model = compile_building(source)
        start = time.perf_counter()
        result = model.step(state, 280.0, jnp.zeros_like(state), 500.0)
        result.temperature.block_until_ready()
        elapsed = time.perf_counter() - start
        error = float(jnp.max(jnp.abs(result.temperature - target)))
        output[source.source_id] = {
            "seconds_including_first_dispatch": elapsed,
            "maximum_temperature_error_K": error,
            "successful": bool(result.successful),
            "native_system": type(model.system).__name__,
            "passed": bool(result.successful) and error <= 1e-6,
        }
    boundary_model = compile_building(
        BuildingSource(
            (Zone("room", 1000),),
            boundaries=(BuildingBoundary("soil", kind="ground"), BuildingBoundary("air")),
            adjacencies=(Adjacency("floor", "room", None, 2, boundary_id="soil"),),
            ventilation=(
                VentilationExchange(
                    "leak", "room", 1, boundary_id="air", kind="infiltration"
                ),
            ),
            source_id="ground-and-air",
        )
    )
    net = boundary_model.observe(
        jnp.array([300.0]), jnp.array([270.0, 290.0]), jnp.zeros(1)
    ).net_heat[0]
    output["ground-and-air"] = {
        "net_heat_W": float(net),
        "expected_W": -70.0,
        "passed": abs(float(net) + 70) <= 1e-8,
    }
    model = compile_building(sources[0])
    times = jnp.array([0.0, 300.0, 900.0, 1200.0, 2400.0])
    control = optimize_hvac(
        model,
        jnp.array([293.15]),
        times,
        jnp.full(4, 283.15),
        jnp.zeros((4, 1)),
        293.15,
        heat_distribution=jnp.ones((1, 1)),
        conversion_law=ResistiveHeatingLaw(),
        supply_temperature=313.15,
        power_upper=200,
        initial_power=jnp.full((4, 1), 50.0),
        power_scale=100,
        termination=OptimizationTermination(maximum_steps=60),
    )
    control_error = float(jnp.max(jnp.abs(control.electrical_power - 100)))
    output["hvac-control"] = {
        "successful": bool(control.successful),
        "maximum_equilibrium_power_error_W": control_error,
        "passed": bool(control.successful) and control_error <= 0.02,
    }

    def make_source(p):
        return BuildingSource(
            (Zone("a", 10000 * jnp.exp(p[0])),),
            adjacencies=(Adjacency("out", "a", None, 10),),
            source_id="calibration",
        )

    heat = jnp.array([[0.0], [100.0], [20.0], [80.0]])
    train_target = replay_building(
        model, jnp.array([300.0]), times, jnp.full(4, 280.0), heat
    ).temperature[1:]
    hold_target = replay_building(
        model, jnp.array([295.0]), times, jnp.full(4, 275.0), heat
    ).temperature[1:]
    train = BuildingExperiment(
        [300.0], times, jnp.full(4, 280.0), heat, train_target, experiment_id="training"
    )
    heldout = BuildingExperiment(
        [295.0], times, jnp.full(4, 275.0), heat, hold_target, experiment_id="heldout"
    )
    calibration = calibrate_building(
        make_source,
        jnp.array([0.2]),
        train,
        heldout,
        observation_nodes=(0,),
        termination=OptimizationTermination(maximum_steps=30),
    )
    output["calibration"] = {
        "successful": bool(calibration.successful),
        "identifiable": bool(calibration.identifiable),
        "heldout_rmse_K": float(calibration.heldout_rmse),
        "passed": bool(calibration.successful) and float(calibration.heldout_rmse) < 1e-4,
    }
    return output


def run_energyplus_reference(path, version, license_id):
    executable = pin_energy_executable(path, version=version, license_id=license_id)
    idf_version = ".".join(version.split(".")[:2])
    reference = energyplus_adiabatic_reference(version=idf_version)
    run = reference.run(executable, energyplus_reference_weather())
    watt = derived_unit("W", ((JOULE, 1), (SECOND, -1)))
    variables = (
        EnergyPlusVariable(
            "ZONE:Zone Mean Air Temperature [C](Hourly)",
            "zone_temperature",
            KELVIN,
            offset=273.15,
        ),
        EnergyPlusVariable(
            "IDEAL:Zone Ideal Loads Zone Sensible Cooling Rate [W](Hourly)",
            "sensible_cooling",
            watt,
        ),
    )
    temperature, cooling = parse_energyplus_csv(
        run.output("eplusout.csv").decode(),
        variables,
        year=2001,
        standard_utc_offset=0,
        interval_seconds=3600,
    )
    # Predeclared model-matched tolerances; no post hoc widening.
    temperature_error = float(jnp.max(jnp.abs(temperature.samples.values - 293.15)))
    cooling_error = float(jnp.max(jnp.abs(cooling.samples.values - 100.0)))
    complete = bool(
        jnp.all(temperature.samples.sample_valid) & jnp.all(cooling.samples.sample_valid)
    )
    return {
        "maximum_temperature_error_K": temperature_error,
        "maximum_cooling_error_W": cooling_error,
        "temperature_tolerance_K": 0.05,
        "cooling_tolerance_W": 1.0,
        "passed": complete and temperature_error <= 0.05 and cooling_error <= 1.0,
        "executable_sha256": executable.sha256,
        "model_sha256": reference.content_sha256,
    }


def run_radiance_reference(oconv_path, rtrace_path, raypath, version, license_id):
    oconv = pin_energy_executable(oconv_path, version=version, license_id=license_id)
    rtrace = pin_energy_executable(rtrace_path, version=version, license_id=license_id)
    operator, runs = produce_uniform_sky_reference(
        oconv, rtrace, environment={"RAYPATH": str(Path(raypath).resolve())}
    )
    irradiance = operator.apply(jnp.ones((1, 3)))
    error = float(jnp.max(jnp.abs(irradiance - np.pi)))
    return {
        "irradiance_RGB": np.asarray(irradiance).tolist(),
        "maximum_error": error,
        "absolute_tolerance": 0.05,
        "passed": error <= 0.05,
        "rtrace_sha256": rtrace.sha256,
        "external_seconds": sum(run.elapsed_seconds for run in runs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energyplus")
    parser.add_argument("--energyplus-version", default="26.1")
    parser.add_argument("--energyplus-license", default="BSD-3-Clause")
    parser.add_argument("--oconv")
    parser.add_argument("--rtrace")
    parser.add_argument("--raypath")
    parser.add_argument("--radiance-version", default="6.0.2")
    parser.add_argument("--radiance-license", default="Radiance")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    report = run_native()
    if args.energyplus:
        report["energyplus-reference"] = run_energyplus_reference(
            args.energyplus, args.energyplus_version, args.energyplus_license
        )
    if any((args.oconv, args.rtrace, args.raypath)):
        if not all((args.oconv, args.rtrace, args.raypath)):
            parser.error("Radiance requires --oconv, --rtrace, and --raypath together")
        report["radiance-reference"] = run_radiance_reference(
            args.oconv,
            args.rtrace,
            args.raypath,
            args.radiance_version,
            args.radiance_license,
        )
    print(json.dumps(report, indent=2))
    if not all(case["passed"] for case in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
