#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._thermal_dark_rates import (
    HTLPolarizationPlan,
    LPMIntegralPlan,
    thermal_kernel_payload_bytes,
    ThermalDarkRatePlan,
    ThermalKernelArtifact,
    ThermalKernelStatus,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)
from phydrax.qualification import ReferenceArtifactManifest


def _units_and_frame():
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
        tolerance=1.0e-7,
        source_id="thermal-test-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="thermal-test-observer",
        orientation_id="right-handed-future",
    )
    return units, frame


def _tables():
    temperature = np.asarray((1.0, 2.0, 3.0, 4.0))
    momentum = np.asarray((0.0, 1.0))
    frequency = np.asarray((-1.0, 0.0, 1.0))
    shape = (1, temperature.size, momentum.size, frequency.size)
    pressure = temperature**2
    entropy = 2.0 * temperature
    energy = temperature * entropy - pressure
    return (
        temperature,
        momentum,
        frequency,
        np.sqrt(0.5 + temperature[None, :] ** 2),
        0.1 * np.ones((1, temperature.size)),
        0.2j * np.ones(shape, dtype=complex),
        np.ones(shape),
        np.ones(shape[1:], dtype=complex),
        2.0 * np.ones(shape[1:], dtype=complex),
        temperature[None, :] ** 2,
        pressure,
        energy,
        entropy,
        0.01 * np.eye(3 * temperature.size),
    )


def _artifact(*, external=False, commercial_use=False):
    units, frame = _units_and_frame()
    tables = _tables()
    species_ids = (
        DarkSectorSpeciesPlan(
            "thermal-test-species",
            1.0,
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        ).species_plan_id,
    )
    channel_ids = ("thermal-test-rate-channel",)
    if not external:
        return ThermalKernelArtifact(
            *tables,
            units,
            frame,
            species_plan_ids=species_ids,
            rate_channel_ids=channel_ids,
            source_kind="native-analytic",
            thermodynamic_tolerance=1.0e-10,
        )
    payload = thermal_kernel_payload_bytes(
        *tables,
        species_plan_ids=species_ids,
        rate_channel_ids=channel_ids,
    )
    digest = hashlib.sha256(payload).hexdigest()
    manifest = ReferenceArtifactManifest(
        "thermal-test-tables",
        checksum_algorithm="sha256",
        checksum=digest,
        size_bytes=len(payload),
        license_id="thermal-test-license",
        commercial_use_permitted=False,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy": 1.0},
        uncertainty={"relative_rate": 0.01},
        lineage_ids=("thermal-test-source",),
    )
    envelope = ScientificArtifactEnvelope(
        artifact_kind="thermal-kernel-tables",
        content_digest=digest,
        producer="phydrax-test",
        producer_version="1",
        build_id="thermal-test-build",
        license_id=manifest.license_id,
        resource_id="thermal-test-resource",
        status="complete",
        parent_artifact_ids=(manifest.manifest_id,),
    )
    return ThermalKernelArtifact(
        *tables,
        units,
        frame,
        species_plan_ids=species_ids,
        rate_channel_ids=channel_ids,
        source_kind="external-table",
        source_manifest=manifest,
        source_artifact=envelope,
        commercial_use=commercial_use,
        redistribution=False,
        training_use=False,
        export=False,
        thermodynamic_tolerance=1.0e-10,
    )


def test_external_artifact_binds_rights_provenance_and_stops_table_gradients():
    artifact = _artifact(external=True)

    assert artifact.evidence.qualified
    assert artifact.differentiation == "external-table-stop-gradient"
    assert (
        artifact.source_manifest.manifest_id
        in artifact.source_artifact.parent_artifact_ids
    )
    assert artifact.frame_realization_id == artifact.frame.realization_id()
    np.testing.assert_array_equal(artifact.frame_token, artifact.frame.frame_token)
    derivative = jax.grad(lambda value: value * 0.0 + jnp.sum(artifact.rates))(1.0)
    np.testing.assert_allclose(derivative, 0.0)
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        _artifact(external=True, commercial_use=True)


def test_htl_has_vacuum_limit_ward_identity_and_only_spacelike_landau_support():
    units, frame = _units_and_frame()
    vacuum = HTLPolarizationPlan(0.0, units, frame).evaluate(
        jnp.asarray((0.5, 4.0)), jnp.asarray((1.0, 1.0))
    )
    np.testing.assert_allclose(vacuum.longitudinal, 0.0, atol=0.0)
    np.testing.assert_allclose(vacuum.transverse, 0.0, atol=0.0)

    response = HTLPolarizationPlan(2.0, units, frame).evaluate(
        jnp.asarray((0.5, 4.0)), jnp.asarray((1.0, 1.0))
    )
    np.testing.assert_array_equal(
        response.evidence.landau_damping_support, jnp.asarray((True, False))
    )
    assert abs(float(response.longitudinal[0].imag)) > 0.0
    np.testing.assert_allclose(response.longitudinal[1].imag, 0.0, atol=0.0)
    np.testing.assert_allclose(response.evidence.ward_residual, 0.0, atol=2.0e-7)
    assert bool(jnp.all(response.evidence.valid))


def test_lpm_fixed_basis_matches_diagonal_analytic_solution_and_subtracts_overlap():
    units, frame = _units_and_frame()
    plan = LPMIntegralPlan(
        jnp.asarray((0.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(((2.0,),)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        units,
        frame,
        rate_prefactor=1.0,
    )
    result = plan.solve(1.0, 0.1)

    assert bool(result.successful)
    np.testing.assert_allclose(result.amplitude, 1.0 / (2.0 + 1.0j), rtol=2.0e-6)
    np.testing.assert_allclose(result.raw_rate, 0.8, rtol=2.0e-6)
    np.testing.assert_allclose(result.rate, 0.7, rtol=2.0e-6)
    assert bool(result.evidence.overlap_subtracted)
    assert plan.solve_resources["matrix_elements"] == 1
    assert plan.rate_unit_id

    free = LPMIntegralPlan(
        jnp.asarray((0.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(((0.0,),)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        units,
        frame,
        rate_prefactor=1.0,
    ).solve()
    assert bool(free.successful)
    np.testing.assert_allclose(free.amplitude, -1.0j, rtol=2.0e-6)
    np.testing.assert_allclose(free.rate, 0.0, atol=2.0e-7)


def test_eos_identities_stability_covariance_and_domain_refusal_are_explicit():
    artifact = _artifact()
    plan = ThermalDarkRatePlan(artifact)
    state = plan.evaluate(2.0)

    assert artifact.evidence.thermodynamically_consistent
    assert artifact.evidence.thermodynamically_stable
    assert artifact.evidence.covariance_positive_semidefinite
    assert bool(state.valid)
    np.testing.assert_allclose(
        state.energy_density, 2.0 * state.entropy_density - state.pressure
    )
    assert float(state.heat_capacity) > 0.0

    refused = plan.evaluate(10.0)
    assert not bool(refused.valid)
    assert int(refused.status) == int(ThermalKernelStatus.OUTSIDE_TEMPERATURE_SUPPORT)
    assert bool(jnp.isnan(refused.rates[0]))
