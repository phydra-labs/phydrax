import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cosmology._sidm_gravothermal import (
    gravothermal_calibration_payload,
    GravothermalSIDMPlan,
    GravothermalSIDMState,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest


def _calibration_records(
    faces,
    calibration,
    *,
    calibration_id="fixed-reference-calibration",
    license_id="test-calibration-license",
    commercial_use_permitted=True,
):
    payload = gravothermal_calibration_payload(
        faces,
        1.0,
        0.2,
        calibration,
        calibration_id=calibration_id,
    )
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = ReferenceArtifactManifest(
        "gravothermal-test-calibration",
        checksum_algorithm="sha256",
        checksum=checksum,
        size_bytes=len(payload),
        license_id=license_id,
        commercial_use_permitted=commercial_use_permitted,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"radius": 1.0, "mass": 1.0, "time": 1.0},
        uncertainty={"conductivity_calibration": 0.05},
        lineage_ids=("gravothermal-test-source",),
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="gravothermal-sidm-calibration",
        content_digest=checksum,
        producer="phydrax-test",
        producer_version="1",
        build_id="gravothermal-test-build",
        license_id=license_id,
        resource_id=calibration_id,
        status="complete",
        parent_artifact_ids=(manifest.manifest_id,),
    )
    return manifest, artifact


def _plan(*, shells=32, calibration=0.75, maximum_change=0.1, **kwargs):
    faces = jnp.linspace(0.0, 8.0, shells + 1)
    calibration_id = kwargs.pop("calibration_id", "fixed-reference-calibration")
    manifest = kwargs.pop("calibration_manifest", None)
    artifact = kwargs.pop("calibration_artifact", None)
    if manifest is None or artifact is None:
        manifest, artifact = _calibration_records(
            faces, calibration, calibration_id=calibration_id
        )
    return GravothermalSIDMPlan(
        faces,
        1.0,
        0.2,
        calibration,
        maximum_fractional_energy_change=maximum_change,
        calibration_id=calibration_id,
        calibration_manifest=manifest,
        calibration_artifact=artifact,
        commercial_use=kwargs.pop("commercial_use", False),
        redistribution=kwargs.pop("redistribution", False),
        training_use=kwargs.pop("training_use", False),
        export=kwargs.pop("export", False),
        **kwargs,
    )


def test_plummer_sphere_has_discretely_small_spherical_equilibrium_residual():
    plan = _plan(shells=256)
    radius = plan.radial_centers
    total_mass = 1.0
    scale_radius = 1.0
    density = (
        3.0
        * total_mass
        / (4.0 * jnp.pi * scale_radius**3)
        * (1.0 + (radius / scale_radius) ** 2) ** (-2.5)
    )
    dispersion_squared = total_mass / (6.0 * jnp.sqrt(radius**2 + scale_radius**2))
    state = plan.initialize(density, dispersion_squared)
    result = plan.advance(state, 1.0e-8)
    pressure_scale = jnp.abs(result.diagnostics.pressure / radius)
    gravity_scale = density * result.diagnostics.enclosed_mass / radius**2
    relative = jnp.abs(result.diagnostics.hydrostatic_residual) / jnp.maximum(
        pressure_scale + gravity_scale, jnp.finfo(radius.dtype).tiny
    )

    assert bool(result.successful)
    assert float(jnp.max(relative[4:-4])) < 0.03


def test_underflow_zero_conductive_increment_preserves_dispersion_bit_exactly():
    plan = _plan()
    radius = plan.radial_centers
    density = 3.0 / (4.0 * jnp.pi) * (1.0 + radius**2) ** (-2.5)
    dispersion = 1.0 / (6.0 * jnp.sqrt(1.0 + radius**2))
    state = plan.initialize(density, dispersion)
    zero_increment_step = jnp.asarray(
        jnp.finfo(state.time.dtype).tiny, dtype=state.time.dtype
    )
    result = plan.advance(state, zero_increment_step)

    assert bool(result.successful)
    np.testing.assert_array_equal(
        result.diagnostics.thermal_energy_after,
        result.diagnostics.thermal_energy_before,
    )
    np.testing.assert_array_equal(
        result.accepted_state.velocity_dispersion_squared,
        state.velocity_dispersion_squared,
    )


def test_conduction_moves_energy_outward_and_closes_global_energy_ledger():
    plan = _plan(shells=24)
    density = jnp.exp(-plan.radial_centers / 3.0) + 0.1
    dispersion = 1.0 + 2.0 * jnp.exp(-((plan.radial_centers / 1.5) ** 2))
    state = plan.initialize(density, dispersion)
    unit = plan.advance(state, 1.0)
    dt = 0.02 / unit.diagnostics.maximum_fractional_energy_change
    result = plan.advance(state, dt)

    assert bool(result.successful)
    assert float(result.diagnostics.luminosity_faces[1]) > 0.0
    assert float(result.diagnostics.post_conduction_thermal_energy[0]) < float(
        result.diagnostics.thermal_energy_before[0]
    )
    np.testing.assert_allclose(
        jnp.sum(result.diagnostics.post_conduction_thermal_energy),
        jnp.sum(result.diagnostics.thermal_energy_before),
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        result.diagnostics.energy_balance_defect, 0.0, atol=3.0e-13
    )
    assert bool(result.diagnostics.total_energy_valid)
    assert abs(float(result.diagnostics.total_energy_defect)) <= (
        plan.total_energy_relative_tolerance
        * max(abs(float(result.diagnostics.total_energy_before)), 1.0)
    )


def test_fixed_calibration_scales_conductivity_and_preserves_boundary_contract():
    first = _plan(calibration=0.5)
    second = _plan(calibration=1.0)
    density = jnp.exp(-first.radial_centers / 2.0) + 0.2
    dispersion = 1.0 + 0.5 * jnp.exp(-first.radial_centers)
    first_result = first.advance(first.initialize(density, dispersion), 1.0e-4)
    second_result = second.advance(second.initialize(density, dispersion), 1.0e-4)

    np.testing.assert_allclose(
        second_result.diagnostics.conductivity,
        2.0 * first_result.diagnostics.conductivity,
        rtol=2.0e-13,
    )
    assert float(first_result.diagnostics.heat_flux_faces[0]) == 0.0
    assert float(first_result.diagnostics.heat_flux_faces[-1]) == 0.0
    assert float(first_result.diagnostics.boundary_energy_transfer) == 0.0


def test_large_conduction_step_rolls_back_atomically():
    plan = _plan(maximum_change=0.01)
    density = jnp.ones((plan.shell_count,))
    dispersion = jnp.linspace(3.0, 1.0, plan.shell_count)
    state = plan.initialize(density, dispersion)
    result = plan.advance(state, 1.0e6)

    assert not bool(result.diagnostics.timestep_valid)
    assert not bool(result.successful)
    for actual, expected in zip(
        jax.tree.leaves(result.accepted_state),
        jax.tree.leaves(state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_nonuniform_radial_geometry_uses_resistance_weighting_and_exact_quadratic_gradient():
    faces = jnp.asarray((0.0, 0.1, 0.4, 1.0, 2.0))
    manifest, artifact = _calibration_records(
        faces, 0.75, calibration_id="nonuniform-calibration"
    )
    plan = GravothermalSIDMPlan(
        faces,
        1.0,
        0.2,
        0.75,
        calibration_id="nonuniform-calibration",
        calibration_manifest=manifest,
        calibration_artifact=artifact,
        commercial_use=False,
        redistribution=False,
        training_use=False,
        export=False,
    )
    density = jnp.ones((plan.shell_count,))
    dispersion = 1.0 + plan.radial_centers**2
    state = GravothermalSIDMState(density, dispersion, faces, 0.0)
    closure = plan._closure(state)
    centers = closure[0]
    enclosed = closure[3]
    pressure = closure[4]
    hydrostatic_residual = closure[5]
    conductivity = closure[10]
    heat_flux = closure[11]

    expected_enclosed = (4.0 * jnp.pi / 3.0) * centers**3
    np.testing.assert_allclose(enclosed, expected_enclosed, rtol=2.0e-13)
    gravity = enclosed / centers**2
    recovered_pressure_gradient = hydrostatic_residual - gravity
    np.testing.assert_allclose(
        recovered_pressure_gradient,
        2.0 * centers,
        rtol=2.0e-12,
        atol=2.0e-13,
    )
    left_distance = faces[1:-1] - centers[:-1]
    right_distance = centers[1:] - faces[1:-1]
    expected_face_conductivity = (left_distance + right_distance) / (
        left_distance / conductivity[:-1] + right_distance / conductivity[1:]
    )
    expected_flux = -expected_face_conductivity * (
        jnp.diff(dispersion) / jnp.diff(centers)
    )
    np.testing.assert_allclose(
        heat_flux[1:-1],
        expected_flux,
        rtol=2.0e-13,
    )


def test_nonhydrostatic_input_is_rejected_and_rolled_back():
    plan = _plan()
    radius = plan.radial_centers
    density = 3.0 / (4.0 * jnp.pi) * (1.0 + radius**2) ** (-2.5)
    dispersion = 1.0 / (6.0 * jnp.sqrt(1.0 + radius**2))
    state = plan.initialize(density, dispersion)
    invalid = GravothermalSIDMState(
        state.mass_density.at[0].multiply(2.0),
        state.velocity_dispersion_squared,
        state.radial_faces,
        state.time,
    )
    result = plan.advance(invalid, 1.0e-6)

    assert not bool(result.diagnostics.initial_hydrostatic_valid)
    assert not bool(result.successful)
    for actual, expected in zip(
        jax.tree.leaves(result.accepted_state),
        jax.tree.leaves(invalid),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_structural_nonconvergence_rolls_back_conductive_update():
    plan = _plan()
    radius = plan.radial_centers
    density = 3.0 / (4.0 * jnp.pi) * (1.0 + radius**2) ** (-2.5)
    dispersion = 1.0 / (6.0 * jnp.sqrt(1.0 + radius**2))
    state = plan.initialize(density, dispersion)
    limited = _plan(
        structural_residual_tolerance=1.0e-12,
        maximum_structural_iterations=1,
    )
    result = limited.advance(state, 1.0e-3)

    assert not bool(result.diagnostics.structural_converged)
    assert not bool(result.successful)
    for actual, expected in zip(
        jax.tree.leaves(result.accepted_state),
        jax.tree.leaves(state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_reflecting_center_contract_rejects_nonzero_inner_radius():
    with pytest.raises(ValueError, match="payload inputs are invalid"):
        gravothermal_calibration_payload(
            jnp.asarray((0.1, 0.2, 0.4, 0.8)),
            1.0,
            0.2,
            0.75,
            calibration_id="invalid-inner-radius",
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("geometry", "periodic-cartesian", "isolated-spherical"),
        ("velocity_model", "anisotropic", "isotropic"),
        ("collision_model", "inelastic-multispecies", "elastic-single-species"),
        ("boundary_condition", "open", "reflecting-center-zero-flux-outer"),
    ),
)
def test_gravothermal_profile_refuses_out_of_regime_claims(keyword, value, message):
    with pytest.raises(ValueError, match=message):
        _plan(**{keyword: value})


def test_calibration_requested_use_denial_is_fail_closed():
    faces = jnp.linspace(0.0, 8.0, 33)
    manifest, artifact = _calibration_records(faces, 0.75, commercial_use_permitted=False)
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        _plan(
            calibration_manifest=manifest,
            calibration_artifact=artifact,
            commercial_use=True,
        )


def test_calibration_value_substitution_fails_exact_digest_and_size_contract():
    faces = jnp.linspace(0.0, 8.0, 33)
    manifest, artifact = _calibration_records(faces, 0.75)
    with pytest.raises(ValueError, match="mismatch"):
        _plan(
            calibration=0.8,
            calibration_manifest=manifest,
            calibration_artifact=artifact,
        )


def test_calibration_envelope_license_and_lineage_must_match_manifest():
    faces = jnp.linspace(0.0, 8.0, 33)
    manifest, artifact = _calibration_records(faces, 0.75)
    substituted = ScientificArtifactEnvelope(
        artifact_kind=artifact.artifact_kind,
        content_digest=artifact.content_digest,
        producer=artifact.producer,
        producer_version=artifact.producer_version,
        build_id=artifact.build_id,
        license_id="substituted-license",
        resource_id=artifact.resource_id,
        status="complete",
        parent_artifact_ids=(),
    )
    with pytest.raises(ValueError, match="license, or lineage disagree"):
        _plan(
            calibration_manifest=manifest,
            calibration_artifact=substituted,
        )
