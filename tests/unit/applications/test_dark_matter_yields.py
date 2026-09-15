import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.astrophysics._operators import BinnedResponsePlan, SpectralField
from phydrax.applications.astrophysics._photometry import ObservationDataProvenance
from phydrax.applications.dark_matter._indirect_detection import (
    annihilation_flux,
    BinnedIndirectDetectionPlan,
    decay_flux,
    DFactor,
    JFactor,
)
from phydrax.applications.dark_matter._yields import (
    AnnihilationProcessDescriptor,
    DecayProcessDescriptor,
    ExactLineTable,
    ExternalYieldProviderResult,
    mix_particle_yields,
    ParticleYieldSpectrum,
    YieldUncertainty,
)
from phydrax.qualification import ReferenceArtifactManifest


def _manifest(*, checksum="0" * 64, commercial=True):
    return ReferenceArtifactManifest(
        "provider-yield.npz",
        checksum_algorithm="sha256",
        checksum=checksum,
        size_bytes=1,
        license_id="BSD-3-Clause",
        commercial_use_permitted=commercial,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="EAR99",
        nondimensionalization={"energy_gev": 1.0},
        uncertainty={"relative": 0.01},
        lineage_ids=("provider-release",),
    )


def _spectrum(
    value=1.0,
    *,
    continuum_sigma=0.1,
    line_multiplicity=0.5,
    line_sigma=0.2,
    provenance=None,
):
    provenance = provenance or ObservationDataProvenance.native("test-yield")
    continuum = SpectralField(
        [1.0, 2.0, 3.0],
        jnp.full((3,), value),
        provenance,
        coordinate_unit="GeV",
        value_unit="event^-1 GeV^-1",
        field_id="photon-yield",
    )
    return ParticleYieldSpectrum(
        continuum,
        ExactLineTable([2.0], [line_multiplicity]),
        YieldUncertainty(
            jnp.full((3,), continuum_sigma),
            jnp.asarray([line_sigma]),
        ),
        product_species="photon",
    )


def test_continuum_and_exact_lines_preserve_multiplicity_energy_and_uncertainty():
    evidence = _spectrum().integrate()

    np.testing.assert_allclose(evidence.continuum_multiplicity, 2.0)
    np.testing.assert_allclose(evidence.line_multiplicity, 0.5)
    np.testing.assert_allclose(evidence.total_multiplicity, 2.5)
    np.testing.assert_allclose(evidence.continuum_energy_gev, 4.0)
    np.testing.assert_allclose(evidence.line_energy_gev, 1.0)
    np.testing.assert_allclose(evidence.total_energy_gev, 5.0)
    np.testing.assert_allclose(
        evidence.multiplicity_standard_deviation,
        np.sqrt(0.05**2 + 0.1**2 + 0.05**2 + 0.2**2),
    )
    np.testing.assert_allclose(
        evidence.energy_standard_deviation_gev,
        np.sqrt((0.05 * 1.0) ** 2 + (0.1 * 2.0) ** 2 + (0.05 * 3.0) ** 2 + 0.4**2),
    )
    assert bool(evidence.valid)


def test_branching_mixture_is_weighted_and_rejects_incomplete_branching():
    first = _spectrum(
        value=1.0, continuum_sigma=0.1, line_multiplicity=1.0, line_sigma=0.2
    )
    second = _spectrum(
        value=3.0, continuum_sigma=0.2, line_multiplicity=2.0, line_sigma=0.4
    )

    mixed = mix_particle_yields(
        ((0.25, first), (0.75, second)), mixture_id="gamma-mixture"
    )

    np.testing.assert_allclose(mixed.continuum.values, 2.5)
    np.testing.assert_allclose(
        mixed.uncertainty.continuum_standard_deviation,
        np.sqrt((0.25 * 0.1) ** 2 + (0.75 * 0.2) ** 2),
    )
    np.testing.assert_allclose(mixed.lines.multiplicity, [0.25, 1.5])
    np.testing.assert_allclose(mixed.lines.energy_gev, [2.0, 2.0])
    with pytest.raises(ValueError, match="sum to one"):
        mix_particle_yields(((0.2, first), (0.7, second)), mixture_id="invalid")


def test_annihilation_and_decay_use_distinct_normalizations_and_propagate_uncertainty():
    spectrum = _spectrum(value=2.0, continuum_sigma=0.25)
    annihilation = annihilation_flux(
        AnnihilationProcessDescriptor(2.0, 1.0),
        spectrum,
        JFactor(1.0, 0.1, target_id="annihilation-target"),
    )
    decay = decay_flux(
        DecayProcessDescriptor(2.0, 1.0),
        spectrum,
        DFactor(1.0, 0.1, target_id="decay-target"),
    )

    np.testing.assert_allclose(
        decay.continuum.values / annihilation.continuum.values,
        4.0,
    )
    coefficient = 1.0 / (32.0 * np.pi)
    np.testing.assert_allclose(
        annihilation.continuum_standard_deviation,
        coefficient * np.sqrt(0.25**2 + (0.1 * 2.0) ** 2),
    )


def test_exact_lines_and_continuum_compose_with_identity_binned_response():
    spectrum = _spectrum(
        value=1.0, continuum_sigma=0.0, line_multiplicity=0.5, line_sigma=0.0
    )
    process = AnnihilationProcessDescriptor(2.0, 32.0 * np.pi)
    flux = annihilation_flux(process, spectrum, JFactor(1.0, 0.0, target_id="target"))
    response = BinnedResponsePlan(np.eye(2), response_id="identity-two-bin")
    plan = BinnedIndirectDetectionPlan([1.0, 2.0, 3.0], response, exposure_cm2_s=2.0)

    result = plan.evaluate(flux)

    np.testing.assert_allclose(result.evidence.continuum_bin_flux_cm2_s, [1.0, 1.0])
    np.testing.assert_allclose(result.evidence.line_bin_flux_cm2_s, [0.0, 0.5])
    np.testing.assert_allclose(result.response.predicted, [2.0, 3.0])
    np.testing.assert_allclose(result.predicted_standard_deviation, 0.0)
    assert bool(result.valid)


def test_external_yield_provider_requires_manifest_identity_rights_and_constant_data():
    manifest = _manifest()
    provenance = ObservationDataProvenance(
        producer="external-yield-code",
        producer_version="1.0",
        source_id="yield-table",
        checksum=manifest.checksum,
        license_id=manifest.license_id,
        differentiation="constant",
    )
    spectrum = _spectrum(provenance=provenance)

    admitted = ExternalYieldProviderResult(
        spectrum,
        manifest,
        provider="host-yield-provider",
        provider_version="1.0",
        execution="host",
        commercial_use=True,
    )

    assert bool(admitted.valid)
    assert admitted.manifest.manifest_id == manifest.manifest_id
    denied = _manifest(commercial=False)
    denied_provenance = ObservationDataProvenance(
        producer="external-yield-code",
        producer_version="1.0",
        source_id="yield-table",
        checksum=denied.checksum,
        license_id=denied.license_id,
        differentiation="constant",
    )
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        ExternalYieldProviderResult(
            _spectrum(provenance=denied_provenance),
            denied,
            provider="subprocess-yield-provider",
            provider_version="1.0",
            execution="subprocess",
            commercial_use=True,
        )
    with pytest.raises(PermissionError, match="training-use-not-permitted"):
        ExternalYieldProviderResult(
            spectrum,
            manifest,
            provider="host-yield-provider",
            provider_version="1.0",
            execution="host",
            training_use=True,
        )
    mismatched = _manifest(checksum="1" * 64)
    with pytest.raises(ValueError, match="checksum/license"):
        ExternalYieldProviderResult(
            spectrum,
            mismatched,
            provider="host-yield-provider",
            provider_version="1.0",
            execution="host",
        )
