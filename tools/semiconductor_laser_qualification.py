"""Deterministic semiconductor laser-domain qualification artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from phydrax.applications.semiconductor import (
    ActiveRegionOpticalProjection,
    evaluate_semiconductor_optical_response,
    LinearizedCarrierOpticalResponsePlan,
    simulate_stochastic_traveling_wave_laser,
    simulate_traveling_wave_laser,
    solve_traveling_wave_laser_threshold,
    TravelingWaveLaserInput,
    TravelingWaveLaserNoisePlan,
    TravelingWaveSemiconductorLaserPlan,
    TravelingWaveSemiconductorLaserState,
)
from phydrax.optics.wave import BidirectionalCoupledModePlan


jax.config.update("jax_enable_x64", True)


def _coupled_mode_cases() -> dict[str, object]:
    no_grating = (
        BidirectionalCoupledModePlan(
            jnp.linspace(0.0, 1.0, 17),
            1.0e15,
            jnp.full((16,), 0.3),
            jnp.zeros((16,), dtype="complex128"),
            jnp.full((16,), 0.2),
            reference_propagation_constant=0.7,
        )
        .prepare()
        .execute(left_incoming=1.0)
    )
    stop_band = (
        BidirectionalCoupledModePlan(
            jnp.asarray((0.0, 1.0)),
            1.0e15,
            jnp.asarray((0.0,)),
            jnp.asarray((600.0 + 0.0j,)),
            jnp.asarray((0.0,)),
        )
        .prepare()
        .execute(left_incoming=1.0)
    )
    expected_transmission_power = jnp.exp(-0.2)
    return {
        "no_grating": {
            "status": int(no_grating.status),
            "transmission_power_error": float(
                jnp.abs(no_grating.evidence.output_power - expected_transmission_power)
            ),
            "power_balance_residual": float(no_grating.evidence.power_balance_residual),
        },
        "deep_stop_band": {
            "status": int(stop_band.status),
            "reflection_power_error": float(
                jnp.abs(jnp.abs(stop_band.boundary.left_outgoing) ** 2 - 1.0)
            ),
            "transmission_power": float(jnp.abs(stop_band.boundary.right_outgoing) ** 2),
            "reciprocity_error": float(stop_band.evidence.reciprocity_error),
            "passivity_excess": float(stop_band.evidence.passivity_excess),
        },
    }


def _response_plan() -> LinearizedCarrierOpticalResponsePlan:
    return LinearizedCarrierOpticalResponsePlan(
        1.0e24,
        5.0e-21,
        1.0e15,
        reference_temperature=300.0,
        differential_refractive_index=-2.0e-27,
        background_internal_loss=1.0e3,
        density_range=(0.5e24, 4.0e24),
        temperature_range=(290.0, 310.0),
        angular_frequency_range=(0.9e15, 1.1e15),
        active_volume=3.0e-16,
        confinement_factor=1.0,
        provenance="analytic semiconductor-laser qualification fixture",
        model_id="semiconductor-laser-qualification-response",
    )


def _optical_response_case() -> dict[str, object]:
    projection = ActiveRegionOpticalProjection(
        jnp.asarray((1.0, 2.0, 1.0)) * 1.0e-18,
        jnp.asarray((0.0, 1.0, 0.5)),
        jnp.asarray((1.0, 3.0, 2.0)),
        support_id="qualification-active-support",
        mode_id="qualification-fundamental-mode",
        provenance="analytic qualification projection",
    )
    result = evaluate_semiconductor_optical_response(
        _response_plan(),
        jnp.asarray((1.0, 2.0, 4.0)) * 1.0e24,
        jnp.asarray((300.0, 300.0, 300.0)),
        1.0e15,
        projection=projection,
    )
    return {
        "status": int(result.status),
        "projected_carrier_pair_density": float(result.carrier_pair_density),
        "active_volume": float(result.active_volume),
        "confinement_factor": float(result.confinement_factor),
        "field_power_factor_error": float(
            jnp.abs(2.0 * result.field_gain - result.modal_power_gain)
        ),
        "model_id": result.evidence.model_id,
        "projection_id": result.evidence.projection_id,
    }


def _prepared_laser(step_count: int = 12, *, coupling=None):
    reflection = jnp.sqrt(0.3)
    return TravelingWaveSemiconductorLaserPlan(
        jnp.linspace(0.0, 3.0e-4, 4),
        _response_plan(),
        1.0e15,
        group_velocity=1.0e8,
        active_area=1.0e-12,
        recombination_a=1.0e9,
        recombination_b=0.0,
        recombination_c=0.0,
        left_facet_amplitude_reflection=reflection + 0.0j,
        right_facet_amplitude_reflection=-reflection + 0.0j,
        carrier_density_bounds=(0.5e24, 3.0e24),
        step_count=step_count,
        coupling=coupling,
        grating_provenance="analytic qualification distributed grating",
        ledger_tolerance=2.0e-12,
    ).prepare()


def _laser_cases() -> dict[str, object]:
    prepared = _prepared_laser()
    threshold = solve_traveling_wave_laser_threshold(prepared, 300.0)
    density = threshold.threshold_carrier_pair_density
    initial = TravelingWaveSemiconductorLaserState(
        jnp.full((prepared.section_count,), density),
        jnp.zeros((prepared.section_count,), dtype="complex128"),
        jnp.zeros((prepared.section_count,), dtype="complex128"),
    )
    inputs = TravelingWaveLaserInput(threshold.threshold_injection_current, 300.0)
    deterministic = simulate_traveling_wave_laser(prepared, initial, inputs)
    forcing = TravelingWaveLaserNoisePlan(
        1.0e-10,
        0.0,
        provenance="explicit qualification stochastic forcing",
        noise_id="qualification-noise",
    ).realize(prepared, jax.random.key(1234))
    stochastic = simulate_stochastic_traveling_wave_laser(
        prepared, initial, inputs, forcing
    )
    replay = simulate_stochastic_traveling_wave_laser(prepared, initial, inputs, forcing)
    grating_prepared = TravelingWaveSemiconductorLaserPlan(
        jnp.asarray((0.0, 1.0e-4)),
        _response_plan(),
        1.0e15,
        group_velocity=1.0e8,
        active_area=1.0e-12,
        recombination_a=1.0e9,
        recombination_b=0.0,
        recombination_c=0.0,
        left_facet_amplitude_reflection=jnp.sqrt(0.2),
        right_facet_amplitude_reflection=jnp.sqrt(0.7),
        carrier_density_bounds=(0.5e24, 3.0e24),
        step_count=1,
        coupling=jnp.asarray((4.0e3j,)),
        grating_provenance="analytic qualification distributed grating",
        ledger_tolerance=2.0e-12,
    ).prepare()
    grating_initial = TravelingWaveSemiconductorLaserState(
        jnp.asarray((1.0e24,)),
        jnp.asarray((0.1 + 0.0j,)),
        jnp.asarray((0.0 + 0.0j,)),
    )
    grating = simulate_traveling_wave_laser(
        grating_prepared,
        grating_initial,
        TravelingWaveLaserInput(0.0, 300.0),
    )
    grating_threshold = solve_traveling_wave_laser_threshold(
        grating_prepared, 300.0, mode_iteration_count=8
    )
    return {
        "threshold": {
            "status": int(threshold.status),
            "carrier_pair_density": float(threshold.threshold_carrier_pair_density),
            "injection_current": float(threshold.threshold_injection_current),
            "mirror_loss": float(threshold.mirror_loss),
            "round_trip_log_power_residual": float(
                threshold.round_trip_log_power_residual
            ),
        },
        "deterministic_zero_field": {
            "status": int(deterministic.status),
            "maximum_field_amplitude": float(
                jnp.max(
                    jnp.maximum(
                        jnp.abs(deterministic.forward_field),
                        jnp.abs(deterministic.backward_field),
                    )
                )
            ),
            "carrier_ledger_error": float(deterministic.evidence.carrier_ledger_error),
            "photon_ledger_error": float(deterministic.evidence.photon_ledger_error),
        },
        "stochastic_replay": {
            "status": int(stochastic.status),
            "maximum_field_amplitude": float(
                jnp.max(
                    jnp.maximum(
                        jnp.abs(stochastic.forward_field),
                        jnp.abs(stochastic.backward_field),
                    )
                )
            ),
            "maximum_replay_error": float(
                jnp.max(
                    jnp.maximum(
                        jnp.abs(stochastic.forward_field - replay.forward_field),
                        jnp.abs(stochastic.backward_field - replay.backward_field),
                    )
                )
            ),
            "carrier_ledger_error": float(stochastic.evidence.carrier_ledger_error),
            "photon_ledger_error": float(stochastic.evidence.photon_ledger_error),
            "realization_id": stochastic.noise_realization_id,
        },
        "distributed_grating": {
            "transient_status": int(grating.status),
            "maximum_unitarity_error": float(
                grating.evidence.maximum_grating_unitarity_error
            ),
            "photon_ledger_error": float(grating.evidence.photon_ledger_error),
            "threshold_status": int(grating_threshold.status),
            "threshold_modal_residual": float(grating_threshold.modal_residual),
            "threshold_modal_gap": float(grating_threshold.modal_gap),
            "threshold_mode_iterations": (
                grating_threshold.evidence.threshold_mode_iteration_count
            ),
            "threshold_map_applications": (
                grating_threshold.evidence.threshold_map_applications
            ),
            "grating_id": grating.evidence.grating_id,
        },
        "resources": {
            "section_count": prepared.section_count,
            "step_count": prepared.step_count,
            "workspace_bytes": prepared.workspace_bytes,
            "retained_result_bytes": prepared.retained_result_bytes,
        },
    }


def qualify() -> dict[str, object]:
    coupled_mode = _coupled_mode_cases()
    optical_response = _optical_response_case()
    laser = _laser_cases()
    accepted = (
        coupled_mode["no_grating"]["status"] == 0
        and coupled_mode["no_grating"]["transmission_power_error"] < 1.0e-12
        and coupled_mode["deep_stop_band"]["status"] == 0
        and coupled_mode["deep_stop_band"]["reflection_power_error"] < 1.0e-12
        and optical_response["status"] == 0
        and optical_response["field_power_factor_error"] == 0.0
        and laser["threshold"]["status"] == 0
        and laser["threshold"]["round_trip_log_power_residual"] < 1.0e-10
        and laser["deterministic_zero_field"]["status"] == 0
        and laser["deterministic_zero_field"]["maximum_field_amplitude"] == 0.0
        and laser["stochastic_replay"]["status"] == 0
        and laser["stochastic_replay"]["maximum_replay_error"] == 0.0
        and laser["distributed_grating"]["transient_status"] == 0
        and laser["distributed_grating"]["maximum_unitarity_error"] < 1.0e-12
        and laser["distributed_grating"]["threshold_status"] == 0
        and laser["distributed_grating"]["threshold_modal_residual"] < 1.0e-8
        and laser["distributed_grating"]["threshold_modal_gap"] > 1.0e-5
    )
    return {
        "accepted": bool(accepted),
        "cases": {
            "coupled_mode": coupled_mode,
            "optical_response": optical_response,
            "traveling_wave_laser": laser,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/semiconductor_laser_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualify()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(arguments.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
