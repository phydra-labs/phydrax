"""Coupled-mode, optical-response, and traveling-wave laser benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from phydrax.applications.semiconductor import (
    evaluate_semiconductor_optical_response,
    LinearizedCarrierOpticalResponsePlan,
    simulate_traveling_wave_laser,
    TravelingWaveLaserInput,
    TravelingWaveSemiconductorLaserPlan,
    TravelingWaveSemiconductorLaserState,
)
from phydrax.optics.wave import BidirectionalCoupledModePlan


jax.config.update("jax_enable_x64", True)


def _timed(function, ready):
    start = time.perf_counter()
    value = function()
    jax.block_until_ready(ready(value))
    return value, time.perf_counter() - start


def _response_plan():
    return LinearizedCarrierOpticalResponsePlan(
        1.0e24,
        5.0e-21,
        1.0e15,
        reference_temperature=300.0,
        background_internal_loss=1.0e3,
        carrier_internal_loss_cross_section=1.0e-22,
        density_range=(0.5e24, 4.0e24),
        temperature_range=(290.0, 310.0),
        angular_frequency_range=(0.9e15, 1.1e15),
        active_volume=6.4e-15,
        confinement_factor=0.4,
        provenance="analytic semiconductor-laser benchmark workload",
        model_id="semiconductor-laser-benchmark-response",
    )


def benchmark() -> dict[str, object]:
    section_count = 1024
    coordinate = jnp.linspace(0.0, 0.02, section_count + 1)
    index = jnp.arange(section_count)
    coupled_plan = BidirectionalCoupledModePlan(
        coordinate,
        1.0e15,
        80.0 * jnp.sin(0.013 * index),
        120.0 * jnp.exp(0.002j * index),
        2.0 + jnp.sin(0.017 * index) ** 2,
        maximum_workspace_bytes=1 << 28,
    )
    coupled_prepared, coupled_preparation_seconds = _timed(
        coupled_plan.prepare, lambda value: value.scattering_matrix
    )
    coupled_result, coupled_solve_seconds = _timed(
        lambda: coupled_prepared.execute(
            left_incoming=jnp.ones((32,), dtype=complex),
            right_incoming=jnp.zeros((32,), dtype=complex),
        ),
        lambda value: value.forward_amplitude,
    )

    response_plan = _response_plan()
    response_count = 65_536
    density = jnp.linspace(0.6e24, 3.9e24, response_count)
    temperature = jnp.linspace(290.5, 309.5, response_count)
    frequency = jnp.linspace(0.91e15, 1.09e15, response_count)
    response_result, response_seconds = _timed(
        lambda: evaluate_semiconductor_optical_response(
            response_plan, density, temperature, frequency
        ),
        lambda value: value.modal_power_gain,
    )

    laser_sections = 64
    laser_steps = 256
    reflection = jnp.sqrt(0.7)
    laser_plan = TravelingWaveSemiconductorLaserPlan(
        jnp.linspace(0.0, 6.4e-3, laser_sections + 1),
        response_plan,
        1.0e15,
        group_velocity=1.0e8,
        active_area=1.0e-12,
        recombination_a=1.0e8,
        recombination_b=0.0,
        recombination_c=0.0,
        left_facet_amplitude_reflection=reflection + 0.0j,
        right_facet_amplitude_reflection=-reflection + 0.0j,
        carrier_density_bounds=(0.5e24, 3.5e24),
        step_count=laser_steps,
        gain_compression=0.01,
        linewidth_enhancement_factor=3.0,
        detuning=jnp.linspace(-50.0, 50.0, laser_sections),
        coupling=300.0 * jnp.exp(1j * jnp.linspace(0.0, 0.5, laser_sections)),
        grating_provenance="analytic distributed-grating benchmark workload",
        maximum_workspace_bytes=1 << 29,
        ledger_tolerance=1.0e-9,
    )
    laser_prepared, laser_preparation_seconds = _timed(
        laser_plan.prepare, lambda value: value.times
    )
    initial = TravelingWaveSemiconductorLaserState(
        jnp.full((laser_sections,), 1.2e24),
        jnp.full((laser_sections,), 1.0e-9 + 0.0j),
        jnp.full((laser_sections,), 0.0 + 1.0e-9j),
    )
    inputs = TravelingWaveLaserInput(0.01, 300.0)
    laser_result, laser_first_seconds = _timed(
        lambda: simulate_traveling_wave_laser(laser_prepared, initial, inputs),
        lambda value: value.carrier_pair_density,
    )
    replay_result, laser_replay_seconds = _timed(
        lambda: simulate_traveling_wave_laser(laser_prepared, initial, inputs),
        lambda value: value.carrier_pair_density,
    )

    return {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
        },
        "problem": {
            "coupled_mode_sections": section_count,
            "coupled_mode_rhs_count": 32,
            "optical_response_count": response_count,
            "laser_sections": laser_sections,
            "laser_steps": laser_steps,
            "coupled_mode_workspace_complex_elements": (
                coupled_prepared.workspace_complex_elements
            ),
            "laser_workspace_bytes": laser_prepared.workspace_bytes,
            "laser_retained_result_bytes": laser_prepared.retained_result_bytes,
        },
        "timings_seconds": {
            "coupled_mode_preparation": coupled_preparation_seconds,
            "coupled_mode_solve_32_rhs": coupled_solve_seconds,
            "optical_response_65536": response_seconds,
            "laser_preparation": laser_preparation_seconds,
            "laser_first_transient": laser_first_seconds,
            "laser_replay_transient": laser_replay_seconds,
        },
        "evidence": {
            "coupled_mode_status": int(coupled_result.status),
            "coupled_mode_power_balance_residual": float(
                jnp.max(coupled_result.evidence.power_balance_residual)
            ),
            "coupled_mode_passivity_excess": float(
                coupled_result.evidence.passivity_excess
            ),
            "optical_response_success_fraction": float(
                jnp.mean(response_result.successful)
            ),
            "laser_status": int(laser_result.status),
            "laser_carrier_ledger_error": float(
                laser_result.evidence.carrier_ledger_error
            ),
            "laser_photon_ledger_error": float(laser_result.evidence.photon_ledger_error),
            "laser_grating_unitarity_error": float(
                laser_result.evidence.maximum_grating_unitarity_error
            ),
            "laser_grating_id": laser_result.evidence.grating_id,
            "laser_replay_maximum_carrier_error": float(
                jnp.max(
                    jnp.abs(
                        laser_result.carrier_pair_density
                        - replay_result.carrier_pair_density
                    )
                )
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/semiconductor_laser.json"),
    )
    arguments = parser.parse_args()
    payload = benchmark()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
