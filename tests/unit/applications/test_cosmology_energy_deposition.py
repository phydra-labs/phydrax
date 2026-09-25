import equinox as eqx
import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.cosmology._background import FLRWBackground
from phydrax.applications.cosmology._energy_deposition import (
    CascadeKernelProduct,
    EnergyDepositionStatus,
    ExternalEnergyDepositionProviderResult,
    InjectionSpectrum,
    project_to_thermodynamics_history,
    SpeciesResolvedThermodynamicsHistory,
)
from phydrax.applications.cosmology._products import CosmologyProductProvenance
from phydrax.applications.cosmology._scales import CODE_COSMOLOGY_SCALE
from phydrax.interchange import AdapterStatus
from phydrax.qualification import ReferenceArtifactManifest


def _manifest(*, commercial=True):
    return ReferenceArtifactManifest(
        "deposition-history.npz",
        checksum_algorithm="sha256",
        checksum="2" * 64,
        size_bytes=1,
        license_id="BSD-3-Clause",
        commercial_use_permitted=commercial,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="EAR99",
        nondimensionalization={"energy_gev": 1.0},
        uncertainty={"relative": 0.02},
        lineage_ids=("deposition-provider-release",),
    )


def _kernel(*, escaped_fraction=0.4):
    scale = np.asarray([0.25, 0.5, 1.0])
    energy = np.asarray([1.0, 3.0])
    deposited_fraction = np.asarray([0.2, 0.1, 0.4])
    deposited = (
        deposited_fraction[None, :, None, None]
        * energy[None, None, None, :]
        * np.ones((3, 1, 2, 1))
    )
    escaped = escaped_fraction * energy[None, None, :] * np.ones((3, 2, 1))
    borrowed = 0.1 * energy[None, None, :] * np.ones((3, 2, 1))
    return CascadeKernelProduct(
        scale,
        energy,
        np.asarray([[0.1, 1000.0], [0.2, 500.0], [0.3, 100.0]]),
        deposited,
        escaped,
        borrowed,
        ("photon", "electron"),
        ("hydrogen_ionization", "helium_ionization", "heating"),
        ("electron_fraction", "matter_temperature_k"),
    )


def _injection(values=None, *, redshift=(4.0, 2.0, 1.0), energy=(1.0, 3.0)):
    if values is None:
        values = np.ones((3, 2, 2))
    return InjectionSpectrum(
        redshift,
        energy,
        values,
        ("photon", "electron"),
        source_id="test-injection",
    )


def _history_context(manifest):
    scale = CODE_COSMOLOGY_SCALE
    background = FLRWBackground(1.0, 0.3, scale=scale)
    provenance = CosmologyProductProvenance(
        producer="external-history-provider",
        producer_version="1.0",
        model_form_id=background.model_form_id,
        request_id="history-request",
        numerical_policy_id="table-action-r1",
        physics_policy_id="hydrogen-helium-cascade",
        scale_id=scale.scale_id,
        source_kind="external",
        differentiation=phx.DerivativeContract(route=phx.DerivativeRoute.DIRECT),
        parent_product_ids=(manifest.manifest_id,),
    )
    return scale, background, provenance


def _history(manifest, *, redshift=(1.0, 2.0, 4.0)):
    scale, background, provenance = _history_context(manifest)
    h_ii = np.asarray([0.8, 0.4, 0.1])
    he_ii = np.asarray([0.1, 0.2, 0.3])
    he_iii = np.asarray([0.05, 0.03, 0.01])
    helium_ratio = 0.1
    electron = h_ii + helium_ratio * (he_ii + 2.0 * he_iii)
    history = SpeciesResolvedThermodynamicsHistory(
        redshift,
        h_ii,
        he_ii,
        he_iii,
        electron,
        [100.0, 500.0, 1000.0],
        helium_ratio,
        scale,
        provenance,
        background.realization,
        manifest=manifest,
    )
    return history


def test_injection_preserves_source_redshift_metadata_and_reverses_to_increasing_scale_factor():
    values = np.arange(12.0).reshape(3, 2, 2)
    injection = _injection(values, redshift=(1.0, 2.0, 4.0))

    np.testing.assert_allclose(injection.source_one_plus_redshift, [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(injection.canonical_from_source_index, [2, 1, 0])
    np.testing.assert_allclose(injection.scale_factors, [0.25, 0.5, 1.0])
    np.testing.assert_allclose(injection.differential_number_per_gev, values[::-1])
    assert injection.source_axis_direction == "increasing"

    decreasing = _injection(values, redshift=(4.0, 2.0, 1.0))
    np.testing.assert_allclose(decreasing.source_one_plus_redshift, [4.0, 2.0, 1.0])
    np.testing.assert_array_equal(decreasing.canonical_from_source_index, [0, 1, 2])
    assert decreasing.source_axis_direction == "decreasing"


def test_native_cascade_action_has_complete_ledger_and_separate_cmb_energy():
    result = _kernel().apply(_injection())

    np.testing.assert_allclose(
        result.injected_energy_gev + result.borrowed_cmb_energy_gev,
        np.sum(result.deposited_energy_gev, axis=-1) + result.escaped_energy_gev,
    )
    assert result.deposition_channels == (
        "hydrogen_ionization",
        "helium_ionization",
        "heating",
    )
    np.testing.assert_allclose(
        result.borrowed_cmb_energy_gev, 0.1 * result.injected_energy_gev
    )
    assert np.all(np.asarray(result.borrowed_cmb_energy_gev) > 0.0)
    np.testing.assert_allclose(result.evidence.closure_residual_gev, 0.0, atol=1e-6)
    np.testing.assert_array_equal(
        result.status, np.full(3, int(EnergyDepositionStatus.SUCCESS))
    )
    assert np.all(np.asarray(result.valid))


def test_zero_injection_is_valid_physical_baseline_not_a_numerical_failure():
    result = _kernel().apply(_injection(np.zeros((3, 2, 2))))

    np.testing.assert_allclose(result.injected_energy_gev, 0.0)
    np.testing.assert_allclose(result.deposited_energy_gev, 0.0)
    np.testing.assert_allclose(result.escaped_energy_gev, 0.0)
    np.testing.assert_allclose(result.borrowed_cmb_energy_gev, 0.0)
    np.testing.assert_array_equal(
        result.status, np.full(3, int(EnergyDepositionStatus.NO_INJECTION))
    )
    assert np.all(np.asarray(result.valid))


def test_numerical_nonclosure_is_distinct_from_no_injection():
    result = _kernel(escaped_fraction=0.2).apply(_injection())

    assert not np.any(np.asarray(result.valid))
    np.testing.assert_array_equal(
        result.status,
        np.full(3, int(EnergyDepositionStatus.KERNEL_ENERGY_NONCLOSURE)),
    )


def test_cascade_domain_mismatch_is_rejected_without_table_clamping():
    outside = _injection(energy=(1.0, 4.0))
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="no clamp|exact cascade table domain"
    ):
        jax.block_until_ready(_kernel().apply(outside).injected_energy_gev)


def test_species_history_enforces_hydrogen_helium_electron_relation_and_projects_with_declared_loss():
    manifest = _manifest()
    history = _history(manifest)

    np.testing.assert_allclose(history.scale_factors, [0.25, 0.5, 1.0])
    np.testing.assert_allclose(history.h_ii_fraction, [0.1, 0.4, 0.8])
    relation = history.h_ii_fraction + history.helium_to_hydrogen_number_ratio * (
        history.he_ii_fraction + 2.0 * history.he_iii_fraction
    )
    np.testing.assert_allclose(history.electron_fraction, relation)
    assert history.provenance.differentiation.supported_surfaces == ()

    projected, report = project_to_thermodynamics_history(
        history,
        np.zeros(3),
        np.asarray([0.1, 0.2, 0.1]),
    )

    np.testing.assert_allclose(projected.ionization_fraction, history.electron_fraction)
    np.testing.assert_allclose(projected.baryon_temperature, history.matter_temperature_k)
    assert report.status == AdapterStatus.DECLARED_LOSS
    assert report.valid
    assert len(report.losses) == 1

    scale, background, provenance = _history_context(manifest)
    with pytest.raises(ValueError, match="Electron fraction"):
        SpeciesResolvedThermodynamicsHistory(
            [4.0, 2.0, 1.0],
            [0.1, 0.4, 0.8],
            [0.3, 0.2, 0.1],
            [0.01, 0.03, 0.05],
            [0.2, 0.5, 0.9],
            [1000.0, 500.0, 100.0],
            0.1,
            scale,
            provenance,
            background.realization,
            manifest=manifest,
        )


def test_external_history_and_injection_require_admitted_manifest_rights():
    denied = _manifest(commercial=False)
    scale, background, provenance = _history_context(denied)
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        SpeciesResolvedThermodynamicsHistory(
            [4.0, 2.0, 1.0],
            [0.1, 0.4, 0.8],
            [0.3, 0.2, 0.1],
            [0.01, 0.03, 0.05],
            np.asarray([0.1, 0.4, 0.8])
            + 0.1 * (np.asarray([0.3, 0.2, 0.1]) + 2.0 * np.asarray([0.01, 0.03, 0.05])),
            [1000.0, 500.0, 100.0],
            0.1,
            scale,
            provenance,
            background.realization,
            manifest=denied,
            commercial_use=True,
        )
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        InjectionSpectrum(
            [4.0, 2.0, 1.0],
            [1.0, 3.0],
            np.ones((3, 2, 2)),
            ("photon", "electron"),
            source_id="external-injection",
            source_kind="external",
            differentiation=phx.DerivativeContract(route=phx.DerivativeRoute.DIRECT),
            manifest=denied,
            commercial_use=True,
        )


def test_generic_subprocess_provider_result_binds_history_ledger_and_manifest():
    manifest = _manifest()
    history = _history(manifest, redshift=(4.0, 2.0, 1.0))
    ledger = _kernel().apply(_injection())

    result = ExternalEnergyDepositionProviderResult(
        history,
        ledger,
        manifest,
        provider="history-subprocess",
        provider_version="1.0",
        execution="subprocess",
    )

    assert bool(result.valid)
    assert result.execution == "subprocess"
    assert result.manifest.manifest_id in result.history.provenance.parent_product_ids
    with pytest.raises(PermissionError, match="training-use-not-permitted"):
        ExternalEnergyDepositionProviderResult(
            history,
            ledger,
            manifest,
            provider="history-subprocess",
            provider_version="1.0",
            execution="subprocess",
            training_use=True,
        )
