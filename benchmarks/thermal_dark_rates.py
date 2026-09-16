#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from benchmarks._runtime import capture_environment
from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._thermal_dark_rates import (
    HTLPolarizationPlan,
    LPMIntegralPlan,
    ThermalDarkRatePlan,
    ThermalKernelArtifact,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)


def _context():
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 3, 2, 1)
    units = RelativisticUnitContract(
        scale, RelativityConvention(metric_signature="mostly_minus")
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(1),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="single-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        source_id="thermal-benchmark-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="thermal-benchmark-observer",
        orientation_id="right-handed-future",
    )
    return units, frame


def _case(repetitions: int, basis_size: int, htl_samples: int):
    units, frame = _context()
    temperature = jnp.linspace(1.0, 8.0, 16)
    momentum = jnp.linspace(0.0, 4.0, 24)
    frequency = jnp.linspace(-4.0, 4.0, 32)
    spectral_shape = (4, temperature.size, momentum.size, frequency.size)
    pressure = temperature**2
    entropy = 2.0 * temperature
    species_plan_ids = tuple(
        DarkSectorSpeciesPlan(
            f"thermal-benchmark-species-{index}",
            1.0,
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        ).species_plan_id
        for index in range(4)
    )
    artifact = ThermalKernelArtifact(
        temperature,
        momentum,
        frequency,
        jnp.ones((4, temperature.size)) * temperature[None, :],
        0.1 * jnp.ones((4, temperature.size)),
        0.1j * jnp.ones(spectral_shape),
        jnp.ones(spectral_shape),
        jnp.ones(spectral_shape[1:], dtype=complex),
        jnp.ones(spectral_shape[1:], dtype=complex),
        jnp.stack((temperature, 2.0 * temperature, 3.0 * temperature)),
        pressure,
        temperature * entropy - pressure,
        entropy,
        jnp.eye(3 * temperature.size),
        units,
        frame,
        species_plan_ids=species_plan_ids,
        rate_channel_ids=(
            "thermal-benchmark-rate-0",
            "thermal-benchmark-rate-1",
            "thermal-benchmark-rate-2",
        ),
        source_kind="native-analytic",
        thermodynamic_tolerance=1.0e-5,
    )
    rate_plan = ThermalDarkRatePlan(artifact)
    htl_plan = HTLPolarizationPlan(2.0, units, frame)
    nodes = jnp.linspace(0.0, 1.0, basis_size)
    collision = 2.0 * jnp.eye(basis_size)
    lpm_plan = LPMIntegralPlan(
        nodes,
        jnp.ones((basis_size,)) / basis_size,
        collision,
        jnp.linspace(0.5, 1.5, basis_size),
        jnp.ones((basis_size,)),
        units,
        frame,
        rate_prefactor=1.0,
    )
    frequencies = jnp.linspace(0.0, 2.0, htl_samples)
    wave_numbers = jnp.ones((htl_samples,))
    compiled_rate = eqx.filter_jit(rate_plan.evaluate)
    compiled_htl = eqx.filter_jit(htl_plan.evaluate)
    compiled_lpm = eqx.filter_jit(lpm_plan.solve)

    started = time.perf_counter()
    rate = compiled_rate(4.0)
    htl = compiled_htl(frequencies, wave_numbers)
    lpm = compiled_lpm(1.0, 0.0)
    jax.block_until_ready((rate.rates, htl.longitudinal, lpm.amplitude))
    compile_and_first_ms = 1_000.0 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repetitions):
        rate = compiled_rate(4.0)
        htl = compiled_htl(frequencies, wave_numbers)
        lpm = compiled_lpm(1.0, 0.0)
    jax.block_until_ready((rate.rates, htl.longitudinal, lpm.amplitude))
    execution_ms = 1_000.0 * (time.perf_counter() - started) / repetitions
    table_arrays = (
        artifact.temperature,
        artifact.momentum,
        artifact.frequency,
        artifact.thermal_masses,
        artifact.widths,
        artifact.self_energies,
        artifact.spectral_functions,
        artifact.screening_longitudinal,
        artifact.screening_transverse,
        artifact.rates,
        artifact.pressure,
        artifact.energy_density,
        artifact.entropy_density,
        artifact.eos_covariance,
    )
    table_bytes = sum(value.size * value.dtype.itemsize for value in table_arrays)
    return {
        "profile": "thermal-dark-htl-lpm-eos",
        "table_resources": {
            "temperature_nodes": artifact.temperature.size,
            "momentum_nodes": artifact.momentum.size,
            "frequency_nodes": artifact.frequency.size,
            "species": artifact.species_count,
            "rate_channels": artifact.rate_count,
            "spectral_entries": artifact.spectral_functions.size,
            "resident_array_bytes": table_bytes,
        },
        "rate_resources": {
            "interpolated_rate_channels": artifact.rate_count,
            "htl_samples": htl_samples,
            "landau_samples": int(jnp.count_nonzero(htl.evidence.landau_damping_support)),
            "maximum_ward_residual": float(jnp.max(htl.evidence.ward_residual)),
        },
        "solve_resources": {
            **lpm_plan.solve_resources,
            "matrix_bytes": lpm_plan.collision_matrix.size
            * lpm_plan.collision_matrix.dtype.itemsize,
            "relative_residual": float(lpm.evidence.relative_residual),
        },
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "rate_htl_lpm_batches_per_second": 1_000.0 / execution_ms,
        "successful": bool(rate.valid & jnp.all(htl.evidence.valid) & lpm.successful),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--basis-size", type=int, default=32)
    parser.add_argument("--htl-samples", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/thermal_dark_rates.json"),
    )
    arguments = parser.parse_args()
    if arguments.repeats <= 0 or arguments.basis_size <= 0 or arguments.htl_samples <= 0:
        raise ValueError("Benchmark repetitions and fixed capacities must be positive.")
    case = _case(arguments.repeats, arguments.basis_size, arguments.htl_samples)
    payload = {
        "environment": capture_environment().to_dict(),
        "case": case,
        "all_successful": case["successful"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
