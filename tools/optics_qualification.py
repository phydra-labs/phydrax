"""Deterministic geometric and plane-wave optics qualification artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from phydrax.artifacts import ArtifactManifest
from phydrax.discretization import (
    CylindricalHankelPlan,
    FourierAxisSpec,
    TensorGridPlan,
    UniformAxisSpec,
)
from phydrax.geometry import RigidFrame
from phydrax.optics.geometric import (
    DifferentialRayMap,
    evaluate_refractive_interface,
    NonSequentialOpticsPlan,
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
    OpticalRayState,
    ParaxialResonatorPlan,
    prepare_nonsequential_optics,
    trace_nonsequential_optics,
)
from phydrax.optics.materials import (
    AngularFrequencyValidity,
    ConstantRefractiveIndex,
    RefractiveIndexProvenance,
)
from phydrax.optics.wave import (
    analytic_field_to_envelope,
    AngularSpectrumPlan,
    BidirectionalCoupledModePlan,
    CylindricalAnalyticPulseField,
    CylindricalUnidirectionalPropagationPlan,
    DirectFresnelPlan,
    DrudePlasmaResponsePlan,
    envelope_to_analytic_field,
    fraunhofer_psf,
    FraunhoferImagingPlan,
    InstantaneousScalarSusceptibility,
    IonizingDrudeResponsePlan,
    MultiphotonIonizationRatePlan,
    PlaneFieldSpace,
    prepare_cylindrical_unidirectional_propagation,
    prepare_direct_fresnel,
    prepare_pulse_envelope_bridge,
    propagate_angular_spectrum,
    propagate_cylindrical_unidirectional,
    propagate_direct_fresnel,
    PulseEnvelopeBridgePlan,
    PulseEnvelopeField,
    PulseTimeSpace,
    ScalarPlaneField,
)


jax.config.update("jax_enable_x64", True)


def _interface_case() -> dict[str, float | int | bool]:
    result = evaluate_refractive_interface(
        jnp.asarray((0.0, 0.0, 1.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
        1.0,
        1.5,
    )
    expected_reflectance = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2
    return {
        "status": int(result.status),
        "transmission_valid": bool(result.transmission_valid),
        "reflectance_error": float(
            jnp.max(jnp.abs(result.reflectance - expected_reflectance))
        ),
        "energy_balance_error": float(result.energy_balance_error),
    }


def _nonsequential_attenuation_case() -> dict[str, float | int | bool]:
    vertices = jnp.asarray(
        (
            (-10.0, -10.0, 0.0),
            (10.0, -10.0, 0.0),
            (10.0, 10.0, 0.0),
            (-10.0, 10.0, 0.0),
        )
    )
    surfaces = NonSequentialSurfaceTable(
        vertices,
        jnp.asarray(((0, 1, 2), (0, 2, 3)), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.asarray((1.0,)),
        surface_ids=jnp.asarray((0, 0), dtype=jnp.int32),
        surface_kinds=jnp.full(
            (2,), int(NonSequentialSurfaceKind.ABSORBER), dtype=jnp.int32
        ),
        medium_power_attenuation_coefficients=jnp.asarray((jnp.log(2.0),)),
        medium_attenuation_model_ids=("qualification-half-power",),
    )
    prepared = prepare_nonsequential_optics(
        NonSequentialOpticsPlan(surfaces, maximum_interactions=1, branch_capacity=1)
    )
    result = trace_nonsequential_optics(
        prepared,
        OpticalRayState(
            jnp.asarray(((0.0, 0.0, -1.0),)),
            jnp.asarray(((0.0, 0.0, 1.0),)),
            jnp.asarray((1.0,)),
        ),
        jnp.asarray((1.0,)),
        jnp.asarray((0,), dtype=jnp.int32),
    )
    return {
        "status": int(result.status[0]),
        "successful": bool(result.successful[0]),
        "volume_absorption_error": float(jnp.abs(result.volume_absorbed_power[0] - 0.5)),
        "surface_absorption_error": float(
            jnp.abs(result.surface_absorbed_power[0, 0] - 0.5)
        ),
        "deposition_residual": float(jnp.abs(result.deposition_power_residual[0])),
        "power_ledger_residual": float(jnp.abs(result.power_ledger_residual[0])),
    }


def _paraxial_resonator_case() -> dict[str, float | int | bool]:
    phases = jnp.asarray((0.37, 0.71))
    curvatures = jnp.asarray((1.4, 0.65))
    cosine = jnp.diag(jnp.cos(phases))
    sine = jnp.diag(jnp.sin(phases))
    jacobian = jnp.block(
        [
            [cosine, sine @ jnp.diag(1.0 / curvatures)],
            [-jnp.diag(curvatures) @ sine, cosine],
        ]
    )
    zero = jnp.zeros((4,))
    ray_map = DifferentialRayMap(
        zero,
        zero,
        jacobian,
        jnp.asarray(1.0),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        input_frame_id="resonator-loop",
        output_frame_id="resonator-loop",
        source_prepared_id="optics-qualification-resonator",
        coordinate_convention="(u,v,nθu,nθv)",
    )
    result = ParaxialResonatorPlan((ray_map,)).prepare().execute()
    return {
        "status": int(result.status),
        "successful": bool(result.successful),
        "stable": bool(result.stable),
        "closed_orbit_residual": float(result.evidence.closed_orbit_residual),
        "symplectic_error": float(result.evidence.symplectic_error),
        "eigen_residual": float(result.evidence.eigen_residual),
        "mode_invariance_residual": float(result.mode.invariance_residual),
        "mode_positivity_minimum": float(result.evidence.mode_positivity_minimum),
    }


def _periodic_space(size: int) -> PlaneFieldSpace:
    grid = TensorGridPlan(
        (FourierAxisSpec(size), FourierAxisSpec(size)),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((-jnp.pi, -jnp.pi), (jnp.pi, jnp.pi))))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "periodic-cell")


def _angular_spectrum_case() -> dict[str, float | int | bool]:
    space = _periodic_space(32)
    transverse_wavevector = jnp.asarray((2.0, -3.0))
    values = jnp.exp(
        1j * jnp.sum(space.transverse_coordinates * transverse_wavevector, axis=-1)
    )
    field = ScalarPlaneField(space, values, 13.0, 0.0)
    distance = 0.37
    medium_wavenumber = 8.0
    result = propagate_angular_spectrum(
        AngularSpectrumPlan().prepare(space),
        field,
        distance,
        medium_wavenumber,
    )
    longitudinal = jnp.sqrt(medium_wavenumber**2 - jnp.sum(transverse_wavevector**2))
    expected = values * jnp.exp(1j * longitudinal * distance)
    return {
        "status": int(result.status),
        "successful": bool(result.successful),
        "maximum_field_error": float(jnp.max(jnp.abs(result.field.values - expected))),
        "leakage_fraction": float(result.leakage_fraction),
        "cropped_energy": float(result.cropped_energy),
    }


def _finite_space(size: int, lower: float, upper: float, z: float) -> PlaneFieldSpace:
    grid = TensorGridPlan(
        (UniformAxisSpec(size), UniformAxisSpec(size)),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((lower, lower), (upper, upper))))
    return PlaneFieldSpace(
        grid,
        RigidFrame(jnp.eye(3), jnp.asarray((0.0, 0.0, z))),
        "finite-window",
    )


def _fraunhofer_case() -> dict[str, float | int | bool]:
    pupil_space = _finite_space(33, -0.5, 0.5, 0.0)
    image_space = _finite_space(51, -4.0, 4.0, 1.0)
    radius = jnp.sqrt(jnp.sum(pupil_space.transverse_coordinates**2, axis=-1))
    field = ScalarPlaneField(pupil_space, (radius <= 0.5).astype("float64"), 1.0, 0.0)
    prepared = FraunhoferImagingPlan(
        pupil_space,
        image_space,
        1.0,
        2.0 * jnp.pi,
        1.0,
    ).prepare()
    result = fraunhofer_psf(prepared, field)
    integrated = jnp.sum(result.plane.values * image_space.area_weights)
    return {
        "status": int(result.status),
        "valid": bool(result.valid),
        "integrated_power_error": float(jnp.abs(integrated - 1.0)),
        "samples_per_airy_radius": float(result.sampling.samples_per_airy_radius),
    }


def _pulse_envelope_bridge_case() -> dict[str, float | int | bool]:
    plane = _periodic_space(8)
    temporal_grid = TensorGridPlan((FourierAxisSpec(64),), axis_names=("time",)).prepare(
        jnp.asarray(((-jnp.pi,), (jnp.pi,)))
    )
    time_space = PulseTimeSpace(temporal_grid, topology="periodic-cell")
    time = time_space.coordinates
    carrier = 8.0
    envelope_values = 0.8 + 0.15 * jnp.exp(-2j * time) + 0.05j * jnp.exp(1j * time)
    envelope = PulseEnvelopeField(
        plane,
        time_space,
        jnp.broadcast_to(envelope_values, plane.shape + time_space.shape),
        carrier,
        0.0,
    )
    prepared = prepare_pulse_envelope_bridge(PulseEnvelopeBridgePlan(time_space, carrier))
    analytic = envelope_to_analytic_field(prepared, envelope)
    recovered = analytic_field_to_envelope(prepared, analytic.field)
    return {
        "analytic_status": int(analytic.status),
        "analytic_successful": bool(analytic.successful),
        "recovered_status": int(recovered.status),
        "recovered_successful": bool(recovered.successful),
        "roundtrip_error": float(
            jnp.max(jnp.abs(recovered.field.values - envelope.values))
        ),
        "rejected_spectral_fraction": float(
            jnp.maximum(
                analytic.evidence.rejected_spectral_fraction,
                recovered.evidence.rejected_spectral_fraction,
            )
        ),
    }


def _direct_fresnel_case() -> dict[str, float | int | bool]:
    source = _finite_space(81, -4.0, 4.0, 0.0)
    target = _finite_space(101, -6.0, 6.0, 0.0)
    coordinates = source.transverse_coordinates
    waist = 0.8
    field = ScalarPlaneField(
        source,
        jnp.exp(-jnp.sum(coordinates**2, axis=-1) / waist**2),
        15.0,
        0.0,
    )
    prepared = prepare_direct_fresnel(
        DirectFresnelPlan(
            source,
            target,
            maximum_sampling_phase_step=100.0,
            maximum_paraxial_angle=1.5,
            maximum_power_error=3.0e-2,
        )
    )
    result = propagate_direct_fresnel(prepared, field, 4.0, 20.0)
    output_intensity = jnp.abs(result.field.values) ** 2
    normalization = jnp.sum(target.area_weights * output_intensity)
    measured_radius = (
        jnp.sum(
            target.area_weights
            * output_intensity
            * jnp.sum(target.transverse_coordinates**2, axis=-1)
        )
        / normalization
    )
    rayleigh_range = 0.5 * 20.0 * waist**2
    expected_radius = 0.5 * waist**2 * (1.0 + (4.0 / rayleigh_range) ** 2)
    return {
        "status": int(result.status),
        "successful": bool(result.successful),
        "relative_power_error": float(result.evidence.relative_power_error),
        "rms_radius_error": float(jnp.abs(measured_radius - expected_radius)),
        "maximum_sampling_phase_step": float(result.evidence.maximum_sampling_phase_step),
    }


def _cylindrical_hankel_case() -> dict[str, float | int | bool]:
    prepared = CylindricalHankelPlan(4.0, 48).prepare()
    radius = prepared.radial_coordinates
    values = jnp.exp(-0.7 * radius * radius)
    reconstructed = prepared.inverse(prepared.forward(values))
    return {
        "status": int(prepared.evidence.status),
        "successful": bool(prepared.evidence.successful),
        "root_residual": float(prepared.evidence.root_residual),
        "orthogonality_defect": float(prepared.evidence.orthogonality_defect),
        "inverse_defect": float(prepared.evidence.inverse_defect),
        "parseval_defect": float(prepared.evidence.parseval_defect),
        "roundtrip_error": float(
            jnp.linalg.norm(reconstructed - values)
            / jnp.maximum(jnp.linalg.norm(values), 1.0e-30)
        ),
    }


def _cylindrical_propagation_case() -> dict[str, float | int | bool]:
    hankel = CylindricalHankelPlan(20.0, 16).prepare()
    temporal_grid = TensorGridPlan((FourierAxisSpec(64),), axis_names=("time",)).prepare(
        jnp.asarray(((0.0,), (2.0 * jnp.pi,)))
    )
    time_space = PulseTimeSpace(temporal_grid, topology="periodic-cell")
    temporal_mode = 8
    radial_mode = 1
    radial_spectrum = jnp.zeros((hankel.plan.radial_count,), dtype=jnp.complex128)
    radial_spectrum = radial_spectrum.at[radial_mode].set(1.0)
    radial_values = hankel.inverse(radial_spectrum)
    carrier = jnp.exp(-1j * temporal_mode * time_space.coordinates)
    field = CylindricalAnalyticPulseField(
        hankel,
        time_space,
        radial_values[:, None] * carrier[None, :],
        float(temporal_mode),
        0.0,
    )
    manifest = ArtifactManifest(
        artifact_id="cylindrical-optics-qualification-index",
        producer="phydrax",
        version="current",
        sha256="0" * 64,
        byte_size=0,
        source_uri="generated://cylindrical-optics-qualification",
        license_id="LicenseRef-PHYDRA",
        model="constant cylindrical qualification index",
        coverage="positive qualification frequencies",
    )
    law = ConstantRefractiveIndex(
        1.5,
        validity=AngularFrequencyValidity(0.5, 40.0),
        reference_wave_speed=1.0,
        provenance=RefractiveIndexProvenance(
            manifest, record_id="cylindrical-qualification-index"
        ),
        law_id="cylindrical-qualification-index",
    )
    prepared = prepare_cylindrical_unidirectional_propagation(
        CylindricalUnidirectionalPropagationPlan(
            hankel,
            time_space,
            float(temporal_mode),
            step_count=2,
            maximum_spectral_edge_fraction=1.0,
            maximum_analytic_signal_defect=1.0,
            maximum_hermitian_reconstruction_defect=1.0,
            maximum_nonlinear_rejected_fraction=1.0,
            maximum_refinement_error=1.0,
            maximum_backward_wave_estimate=1.0,
            maximum_radial_boundary_fraction=1.0,
            maximum_radial_high_mode_fraction=1.0,
            maximum_longitudinal_cutoff_fraction=1.0,
        ),
        law,
    )
    distance = 0.2
    result = propagate_cylindrical_unidirectional(
        prepared,
        field,
        InstantaneousScalarSusceptibility(),
        distance,
    )
    expected_radial = hankel.inverse(
        hankel.forward(radial_values)
        * jnp.exp(1j * prepared.longitudinal_wavenumbers[:, temporal_mode] * distance)
    )
    expected = expected_radial[:, None] * carrier[None, :]
    return {
        "status": int(result.status),
        "successful": bool(result.successful),
        "maximum_field_error": float(jnp.max(jnp.abs(result.field.values - expected))),
        "radial_measure_change": float(result.evidence.radial_measure_relative_change),
        "response_successful": bool(result.response_evaluation.successful),
    }


def _ionizing_drude_case() -> dict[str, float | int | bool]:
    temporal_grid = TensorGridPlan((FourierAxisSpec(160),), axis_names=("time",)).prepare(
        jnp.asarray(((0.0,), (2.0,)))
    )
    time_space = PulseTimeSpace(temporal_grid, topology="periodic-cell")
    amplitude = 2.0
    coefficient = 0.3
    neutral_density = 5.0
    field = jnp.full(time_space.shape, amplitude + 0.0j)
    response = (
        IonizingDrudeResponsePlan(
            MultiphotonIonizationRatePlan(
                coefficient,
                2,
                provenance_id="qualification-two-photon-rate",
            ),
            DrudePlasmaResponsePlan(
                0.0,
                provenance_id="qualification-collisionless-drude",
                electron_charge_magnitude=1.0,
                electron_mass=1.0,
            ),
            neutral_density,
            0.0,
        )
        .prepare(
            time_space,
            jnp.fft.fftfreq(time_space.size) > 0.0,
            field.shape,
            temporal_axis=0,
        )
        .evaluate(field)
    )
    elapsed = time_space.coordinates[-1] - time_space.coordinates[0]
    expected = neutral_density * (1.0 - jnp.exp(-coefficient * amplitude**4 * elapsed))
    electron = response.physical_state[..., 0]
    return {
        "status": int(response.status),
        "successful": bool(response.successful),
        "terminal_electron_error": float(jnp.abs(electron[-1] - expected)),
        "minimum_electron_density": float(response.evidence.minimum_electron_density),
        "maximum_electron_density": float(response.evidence.maximum_electron_density),
        "electron_bound_violation": float(response.evidence.electron_bound_violation),
    }


def _bidirectional_coupled_mode_case() -> dict[str, float | int | bool]:
    coupling = 3.0
    result = (
        BidirectionalCoupledModePlan(
            jnp.asarray((0.0, 1.0)),
            2.0e15,
            jnp.asarray((0.0,)),
            jnp.asarray((coupling + 0.0j,)),
            jnp.asarray((0.0,)),
        )
        .prepare()
        .execute(left_incoming=1.0 + 0.0j)
    )
    expected_reflection = jnp.tanh(coupling)
    expected_transmission = 1.0 / jnp.cosh(coupling)
    return {
        "status": int(result.status),
        "successful": bool(result.successful),
        "reflection_magnitude_error": float(
            jnp.abs(jnp.abs(result.boundary.left_outgoing) - expected_reflection)
        ),
        "transmission_magnitude_error": float(
            jnp.abs(jnp.abs(result.boundary.right_outgoing) - expected_transmission)
        ),
        "power_balance_residual": float(result.evidence.power_balance_residual),
        "reciprocity_error": float(result.evidence.reciprocity_error),
        "passivity_excess": float(result.evidence.passivity_excess),
    }


def qualify() -> dict[str, object]:
    cases = {
        "refractive_interface": _interface_case(),
        "angular_spectrum": _angular_spectrum_case(),
        "fraunhofer": _fraunhofer_case(),
        "cylindrical_hankel": _cylindrical_hankel_case(),
        "nonsequential_attenuation": _nonsequential_attenuation_case(),
        "paraxial_resonator": _paraxial_resonator_case(),
        "pulse_envelope_bridge": _pulse_envelope_bridge_case(),
        "direct_fresnel": _direct_fresnel_case(),
        "bidirectional_coupled_mode": _bidirectional_coupled_mode_case(),
        "cylindrical_propagation": _cylindrical_propagation_case(),
        "ionizing_drude": _ionizing_drude_case(),
    }
    accepted = (
        cases["refractive_interface"]["reflectance_error"] < 1.0e-12
        and cases["refractive_interface"]["energy_balance_error"] < 1.0e-12
        and cases["angular_spectrum"]["maximum_field_error"] < 5.0e-5
        and cases["fraunhofer"]["integrated_power_error"] < 5.0e-6
        and cases["cylindrical_hankel"]["successful"]
        and cases["cylindrical_hankel"]["roundtrip_error"] < 5.0e-6
        and cases["nonsequential_attenuation"]["successful"]
        and cases["nonsequential_attenuation"]["volume_absorption_error"] < 5.0e-12
        and cases["nonsequential_attenuation"]["surface_absorption_error"] < 5.0e-12
        and cases["nonsequential_attenuation"]["power_ledger_residual"] < 5.0e-12
        and cases["paraxial_resonator"]["successful"]
        and cases["paraxial_resonator"]["stable"]
        and cases["paraxial_resonator"]["closed_orbit_residual"] < 5.0e-10
        and cases["paraxial_resonator"]["mode_invariance_residual"] < 5.0e-8
        and cases["pulse_envelope_bridge"]["analytic_successful"]
        and cases["pulse_envelope_bridge"]["recovered_successful"]
        and cases["pulse_envelope_bridge"]["roundtrip_error"] < 5.0e-12
        and cases["direct_fresnel"]["successful"]
        and cases["direct_fresnel"]["relative_power_error"] < 3.0e-2
        and cases["direct_fresnel"]["rms_radius_error"] < 2.0e-2
        and cases["bidirectional_coupled_mode"]["successful"]
        and cases["bidirectional_coupled_mode"]["reflection_magnitude_error"] < 5.0e-12
        and cases["bidirectional_coupled_mode"]["transmission_magnitude_error"] < 5.0e-12
        and cases["bidirectional_coupled_mode"]["power_balance_residual"] < 5.0e-12
        and cases["cylindrical_propagation"]["successful"]
        and cases["cylindrical_propagation"]["response_successful"]
        and cases["cylindrical_propagation"]["maximum_field_error"] < 5.0e-8
        and cases["cylindrical_propagation"]["radial_measure_change"] < 5.0e-5
        and cases["ionizing_drude"]["successful"]
        and cases["ionizing_drude"]["terminal_electron_error"] < 5.0e-10
        and cases["ionizing_drude"]["electron_bound_violation"] == 0.0
    )
    return {"accepted": bool(accepted), "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/optics_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualify()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
