#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.lattice_boltzmann._collision import (
    macroscopic_raw_moments,
    quadratic_equilibrium,
)
from phydrax.discretization.lattice_boltzmann._commercial_qualification import (
    c0_guo_baseline_profiles,
    c1_collision_native_forcing_profile,
    c2_binary_interface_profiles,
    c2_dynamic_wetting_profile,
    c3_passive_transport_profiles,
    conjugate_thermal_qualification_profile,
    LatticeBoltzmannCommercialTier,
    LatticeBoltzmannDeploymentRecord,
    LatticeBoltzmannQualificationClaim,
    LatticeBoltzmannQualificationError,
    LatticeBoltzmannQualificationProfile,
    reference_lattice_boltzmann_hardware,
)
from phydrax.discretization.lattice_boltzmann._conjugate_thermal import (
    ConjugateThermalPlan,
)
from phydrax.discretization.lattice_boltzmann._interfacial import (
    constitutive_dynamic_contact_angle_normal,
    ConstitutiveDynamicWettingPlan,
)
from phydrax.discretization.lattice_boltzmann._method import (
    LatticeBoltzmannMethodPlan,
)
from phydrax.discretization.lattice_boltzmann._operating_envelope import (
    LatticeBoltzmannEnvelopeError,
    LatticeBoltzmannHardwareTarget,
    LatticeBoltzmannOperatingPoint,
)
from phydrax.discretization.lattice_boltzmann._thermal import (
    ThermalLatticeBoltzmannPlan,
)
from phydrax.qualification._evidence import QualificationEvidence, SupportDependency


def _point(**updates):
    values = {
        "mach_number": 0.05,
        "knudsen_number": 0.005,
        "relaxation_rate": 1.0,
        "minimum_density": 1.0,
        "maximum_density": 1.05,
        "force_number": 0.005,
        "interface_width_cells": 8.0,
        "wall_resolution_cells": 16.0,
        "viscosity_ratio": 1.05,
        "cahn_number": 0.0,
        "capillary_number": 0.0,
        "relative_mass_drift": 1.0e-10,
        "spurious_current_ratio": 0.0,
    }
    values.update(updates)
    return LatticeBoltzmannOperatingPoint(**values)


def _interface_point(**updates):
    values = {
        "mach_number": 0.05,
        "knudsen_number": 0.005,
        "relaxation_rate": 1.0,
        "minimum_density": 1.0,
        "maximum_density": 5.0,
        "force_number": 0.005,
        "interface_width_cells": 6.0,
        "wall_resolution_cells": 16.0,
        "viscosity_ratio": 5.0,
        "cahn_number": 0.05,
        "capillary_number": 0.05,
        "relative_mass_drift": 1e-10,
        "spurious_current_ratio": 1e-4,
    }
    values.update(updates)
    return LatticeBoltzmannOperatingPoint(**values)


def _deployment(
    profile,
    *,
    host_count=None,
    devices_per_host=None,
    restart_topology="topology-a",
    relation=None,
    parity_evidence_ids=("parity-evidence",),
):
    return LatticeBoltzmannDeploymentRecord(
        "sharded",
        "array-archive",
        "kinetic-array-archive",
        host_count=profile.envelope.hardware.host_count
        if host_count is None
        else host_count,
        devices_per_host=(
            profile.envelope.hardware.devices_per_host
            if devices_per_host is None
            else devices_per_host
        ),
        execution_plan_id="execution-plan",
        output_plan_id="output-plan",
        checkpoint_plan_id="checkpoint-plan",
        execution_topology_id="topology-a",
        restart_topology_id=restart_topology,
        topology_restart_relation_id=relation,
        parity_evidence_ids=parity_evidence_ids,
    )


def _evidence(profile, *, at_time=10):
    scientific = tuple(
        claim.value
        for claim in profile.required_claims
        if claim
        not in (
            LatticeBoltzmannQualificationClaim.FUSED_PARITY,
            LatticeBoltzmannQualificationClaim.AA_PARITY,
            LatticeBoltzmannQualificationClaim.SHARDED_PARITY,
            LatticeBoltzmannQualificationClaim.CHECKPOINT_PARITY,
            LatticeBoltzmannQualificationClaim.OUTPUT_PARITY,
        )
    )
    operational = tuple(
        claim.value for claim in profile.required_claims if claim.value not in scientific
    )
    common = {
        "build_id": "build-a",
        "environment_id": "environment-a",
        "backend": "jax",
        "topology": "topology-a",
        "precision": profile.envelope.precision.policy_id,
        "reduction": "deterministic-tree",
        "replay_id": "replay-a",
        "raw_artifact_ids": ("artifact-a",),
        "reviewer_id": "reviewer-a",
        "issued_at": at_time - 1,
        "expires_at": at_time + 1,
        "reason": "bounded observations satisfy the named criteria",
    }
    return (
        QualificationEvidence(
            "scientific",
            "passed",
            (profile.profile_id,),
            criteria_ids=scientific,
            **common,
        ),
        QualificationEvidence(
            "operational",
            "passed",
            (profile.profile_id,),
            criteria_ids=operational,
            **common,
        ),
    )


def test_c0_exact_support_and_envelope_admission_refusal():
    profiles = c0_guo_baseline_profiles()
    assert len(profiles) == 4
    assert {
        (profile.envelope.lattice.name, profile.envelope.collision.family)
        for profile in profiles
    } == {("D2Q9", "bgk"), ("D2Q9", "trt"), ("D3Q19", "bgk"), ("D3Q19", "trt")}
    profile = profiles[0]
    admission = profile.envelope.evaluate(_point())
    assert bool(admission.admitted)
    assert profile.envelope.forcing_model == "guo"
    assert not profile.signed
    assert not profile.released

    refused = profile.envelope.evaluate(_point(mach_number=0.1001))
    assert not bool(refused.admitted)
    assert "mach-number" in refused.failed_checks()
    with pytest.raises(LatticeBoltzmannEnvelopeError, match="mach-number"):
        profile.envelope.require(_point(mach_number=0.1001))


@pytest.mark.parametrize(
    ("coordinate", "outside", "failed_check"),
    (
        ("mach_number", 0.11, "mach-number"),
        ("knudsen_number", 0.02, "knudsen-number"),
        ("relaxation_rate", 1.95, "relaxation-rate"),
        ("minimum_density", 0.0, "density"),
        ("maximum_density", 11.0, "density-ratio"),
        ("force_number", 0.02, "force-number"),
        ("interface_width_cells", 3.9, "interface-width"),
        ("wall_resolution_cells", 7.9, "wall-resolution"),
        ("viscosity_ratio", 10.1, "viscosity-ratio"),
        ("cahn_number", 0.11, "cahn-number"),
        ("capillary_number", 0.11, "capillary-number"),
        ("relative_mass_drift", 1.1e-8, "mass-drift"),
        ("spurious_current_ratio", 1.1e-3, "spurious-current"),
    ),
)
def test_binary_interface_envelope_refuses_each_bounded_axis(
    coordinate, outside, failed_check
):
    profile = c2_binary_interface_profiles()[0]
    admission = profile.envelope.evaluate(_interface_point(**{coordinate: outside}))
    assert not bool(admission.admitted)
    assert failed_check in admission.failed_checks()


def test_resource_preflight_precedes_preparation_and_fails_closed():
    profile = c0_guo_baseline_profiles()[0]
    estimate = profile.envelope.preflight(
        local_cell_count=64,
        population_field_count=1,
        scalar_field_count=2,
        temporary_population_field_count=1,
        checkpoint_copies=1,
        output_copies=1,
    )
    assert estimate.fits_budget
    prepared = profile.envelope.prepare(
        local_cell_count=64,
        population_field_count=1,
        scalar_field_count=2,
    )
    assert bool(prepared.execute(_point()).admitted)

    hardware = LatticeBoltzmannHardwareTarget(
        "cpu",
        "test-provider",
        "test-cpu",
        maximum_device_bytes=64,
    )
    constrained = c0_guo_baseline_profiles(hardware=hardware)[0]
    with pytest.raises(LatticeBoltzmannEnvelopeError, match="bytes per device"):
        constrained.envelope.prepare(local_cell_count=64)


def test_c1_collision_native_guo_conserves_mass_and_applies_exact_force_momentum():
    profile = c1_collision_native_forcing_profile()
    assert profile.tier is LatticeBoltzmannCommercialTier.C1
    support = dict(profile.support_tuple.attributes)
    assert support["forcing_route"] == "collision-native-central-moment"
    lattice = profile.envelope.lattice
    precision = profile.envelope.precision
    density = jnp.ones((2, 2))
    velocity = jnp.zeros((2, 2, 3))
    force = jnp.broadcast_to(jnp.asarray((1e-7, -2e-7, 3e-7)), velocity.shape)
    populations = quadratic_equilibrium(density, velocity, lattice, precision)
    old_mass, old_momentum = macroscopic_raw_moments(populations, lattice, precision)
    result = LatticeBoltzmannMethodPlan(
        profile.envelope.collision,
        forcing=profile.envelope.forcing,
    ).collide(
        populations,
        density,
        velocity,
        force,
        jnp.asarray(1.0),
        lattice,
        precision,
    )
    new_mass, new_momentum = macroscopic_raw_moments(
        result.populations, lattice, precision
    )
    assert bool(result.successful)
    np.testing.assert_allclose(new_mass, old_mass, atol=2e-12)
    np.testing.assert_allclose(new_momentum - old_momentum, force, atol=2e-12)
    np.testing.assert_allclose(result.diagnostics.mass_error, 0.0, atol=2e-12)
    np.testing.assert_allclose(result.diagnostics.momentum_error, 0.0, atol=2e-12)


def test_c2_profiles_bind_interface_laplace_capillary_droplet_and_wetting_gates():
    wetting = ConstitutiveDynamicWettingPlan(
        np.deg2rad(90.0),
        np.deg2rad(30.0),
        np.deg2rad(150.0),
        microscopic_length=1e-6,
        macroscopic_length=1e-3,
        maximum_absolute_capillary_number=0.1,
    )
    profiles = c2_binary_interface_profiles(dynamic_wetting=wetting)
    assert {
        dict(profile.support_tuple.attributes)["interface_family"] for profile in profiles
    } == {
        "binary-free-energy",
        "binary-colour-gradient",
    }
    dynamic = c2_dynamic_wetting_profile(wetting)
    required = set(dynamic.required_claims)
    assert {
        LatticeBoltzmannQualificationClaim.INTERFACE_EQUILIBRIUM,
        LatticeBoltzmannQualificationClaim.LAPLACE_PRESSURE,
        LatticeBoltzmannQualificationClaim.CAPILLARY_WAVE,
        LatticeBoltzmannQualificationClaim.DROPLET_DEFORMATION,
        LatticeBoltzmannQualificationClaim.DYNAMIC_WETTING,
    } <= required
    at_rest = wetting.evaluate(0.0, 1.0, 1.0)
    assert bool(at_rest.successful)
    np.testing.assert_allclose(at_rest.contact_angle, np.pi / 2)
    outside = wetting.evaluate(0.3, 1.0, 1.0)
    assert not bool(outside.successful)
    assert float(outside.contact_angle) > wetting.advancing_contact_angle

    interface = jnp.asarray(((1.0, 0.0), (1.0, 0.0)))
    wall = jnp.asarray(((0.0, 1.0), (0.0, 1.0)))
    normal, evidence = constitutive_dynamic_contact_angle_normal(
        interface,
        wall,
        jnp.asarray((True, False)),
        0.0,
        1.0,
        1.0,
        wetting,
    )
    assert bool(evidence.successful)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=-1), 1.0)


def test_c3_profiles_distinguish_passive_thermal_species_and_reactive_splitting():
    profiles = c3_passive_transport_profiles()
    by_physics = {profile.envelope.physics_model: profile for profile in profiles}
    assert set(by_physics) == {
        "passive-sensible-energy",
        "passive-fickian-species",
        "reactive-thermal-species-strang",
    }
    assert (
        LatticeBoltzmannQualificationClaim.THERMAL_CONSERVATION
        in by_physics["passive-sensible-energy"].required_claims
    )
    assert (
        LatticeBoltzmannQualificationClaim.SPECIES_CONSERVATION
        in by_physics["passive-fickian-species"].required_claims
    )
    reactive = by_physics["reactive-thermal-species-strang"]
    assert (
        LatticeBoltzmannQualificationClaim.REACTIVE_SPLITTING_CONSERVATION
        in reactive.required_claims
    )
    assert dict(reactive.support_tuple.attributes)["splitting"] == "symmetric-strang"


def test_conjugate_thermal_has_solid_state_and_conservative_interface_flux():
    fluid = ThermalLatticeBoltzmannPlan(2.0, 4.0, reference_temperature=300.0)
    plan = ConjugateThermalPlan(
        fluid,
        3.0,
        6.0,
        contact_resistance=0.25,
    )
    prepared = plan.prepare(
        jnp.asarray((0.5, 0.5)),
        jnp.asarray((0.25, 0.25)),
        jnp.asarray((1.0, 2.0)),
    )
    solid = plan.initialize_solid(jnp.asarray((300.0, 300.0)))
    fluid_energy = jnp.asarray((20.0, 40.0))
    result = prepared.execute(
        fluid_energy,
        solid,
        0.01,
        jnp.asarray((2.0, 2.0)),
        jnp.asarray((3.0, 3.0)),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.interface_flux.fluid_energy_rate + result.interface_flux.solid_energy_rate,
        0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(result.interface_flux.conservation_residual, 0.0, atol=0.0)
    np.testing.assert_allclose(result.conservation_residual, 0.0, atol=1e-12)
    assert int(result.solid_state.step_index) == 1

    refused = prepared.execute(fluid_energy, solid, -0.01, 2.0, 3.0)
    assert not bool(refused.successful)
    np.testing.assert_array_equal(refused.fluid_sensible_energy, fluid_energy)
    np.testing.assert_array_equal(
        refused.solid_state.sensible_energy, solid.sensible_energy
    )

    profile = conjugate_thermal_qualification_profile(plan)
    assert profile.conjugate_thermal.plan_id == plan.plan_id
    assert (
        LatticeBoltzmannQualificationClaim.CONJUGATE_THERMAL_CONSERVATION
        in profile.required_claims
    )
    assert dict(profile.support_tuple.attributes)["solid_energy_state"] is True


def test_commercial_gate_requires_all_scientific_and_execution_evidence():
    profile = c0_guo_baseline_profiles()[0]
    evidence = _evidence(profile)
    result = profile.evaluate(
        evidence,
        profile.envelope.evaluate(_point()),
        _deployment(
            profile,
            parity_evidence_ids=tuple(item.evidence_id for item in evidence),
        ),
        at_time=10,
    )
    assert result.passed
    assert result.coverage.passed
    assert result.deployment.compatible

    missing_operational = profile.evaluate(
        evidence[:1],
        profile.envelope.evaluate(_point()),
        _deployment(
            profile,
            parity_evidence_ids=tuple(item.evidence_id for item in evidence[:1]),
        ),
        at_time=10,
    )
    assert not missing_operational.passed
    assert set(missing_operational.coverage.inconclusive_predicate_ids) == {
        LatticeBoltzmannQualificationClaim.FUSED_PARITY.value,
        LatticeBoltzmannQualificationClaim.AA_PARITY.value,
        LatticeBoltzmannQualificationClaim.SHARDED_PARITY.value,
        LatticeBoltzmannQualificationClaim.CHECKPOINT_PARITY.value,
        LatticeBoltzmannQualificationClaim.OUTPUT_PARITY.value,
    }


def test_commercial_gate_requires_exact_support_dependencies():
    baseline = c0_guo_baseline_profiles()[0]
    dependency = SupportDependency("chemistry-profile", "chemistry-support-tuple")
    profile = LatticeBoltzmannQualificationProfile(
        "c0-dependent-profile",
        baseline.tier,
        baseline.envelope,
        baseline.required_claims,
        execution_modes=baseline.execution_modes,
        output_modes=baseline.output_modes,
        checkpoint_modes=baseline.checkpoint_modes,
        dependencies=(dependency,),
    )
    evidence = _evidence(profile)
    deployment = _deployment(
        profile,
        parity_evidence_ids=tuple(item.evidence_id for item in evidence),
    )
    missing = profile.evaluate(
        evidence,
        profile.envelope.evaluate(_point()),
        deployment,
        at_time=10,
    )
    assert not missing.passed
    assert missing.missing_dependency_ids == (dependency.dependency_id,)
    with pytest.raises(LatticeBoltzmannQualificationError, match="dependency"):
        missing.require()

    qualified = profile.evaluate(
        evidence,
        profile.envelope.evaluate(_point()),
        deployment,
        at_time=10,
        satisfied_dependencies=(dependency,),
    )
    assert qualified.passed
    qualified.require()


def test_multi_host_output_checkpoint_and_restart_topology_are_exact():
    hardware = reference_lattice_boltzmann_hardware(host_count=2, devices_per_host=4)
    profile = c0_guo_baseline_profiles(hardware=hardware)[0]
    compatible = profile.deployment_compatibility(
        _deployment(profile, restart_topology="topology-b", relation="restart-relation")
    )
    assert compatible.compatible
    wrong_hosts = profile.deployment_compatibility(_deployment(profile, host_count=1))
    assert not wrong_hosts.compatible
    assert "host-count" in wrong_hosts.failed_checks
    wrong_devices = profile.deployment_compatibility(
        _deployment(profile, devices_per_host=1)
    )
    assert not wrong_devices.compatible
    assert "devices-per-host" in wrong_devices.failed_checks
    missing_relation = profile.deployment_compatibility(
        _deployment(profile, restart_topology="topology-b")
    )
    assert not missing_relation.compatible
    assert "restart-topology-relation" in missing_relation.failed_checks


def test_profiles_cover_named_benchmarks_and_do_not_claim_a_continuum_method():
    profile = c0_guo_baseline_profiles()[0]
    assert {
        LatticeBoltzmannQualificationClaim.SHEAR_WAVE_DECAY,
        LatticeBoltzmannQualificationClaim.ACOUSTIC_ATTENUATION,
        LatticeBoltzmannQualificationClaim.COUETTE_FLOW,
        LatticeBoltzmannQualificationClaim.POISEUILLE_FLOW,
        LatticeBoltzmannQualificationClaim.CYLINDER_WAKE,
        LatticeBoltzmannQualificationClaim.RELAXATION_SWEEP,
        LatticeBoltzmannQualificationClaim.MACH_SWEEP,
        LatticeBoltzmannQualificationClaim.FUSED_PARITY,
        LatticeBoltzmannQualificationClaim.AA_PARITY,
        LatticeBoltzmannQualificationClaim.SHARDED_PARITY,
        LatticeBoltzmannQualificationClaim.CHECKPOINT_PARITY,
    } <= set(profile.required_claims)
    labels = (profile.name, profile.envelope.physics_model) + tuple(
        str(value) for _, value in profile.support_tuple.attributes
    )
    assert all("direct-numerical-simulation" not in label.lower() for label in labels)

    with pytest.raises(ValueError, match="continuum-flow"):
        LatticeBoltzmannQualificationProfile(
            "forbidden-continuum-dns-profile",
            LatticeBoltzmannCommercialTier.C0,
            profile.envelope,
            profile.required_claims,
            execution_modes=profile.execution_modes,
            output_modes=profile.output_modes,
            checkpoint_modes=profile.checkpoint_modes,
        )
