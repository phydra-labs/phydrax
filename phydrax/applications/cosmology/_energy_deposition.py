#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._differentiation import DerivativeContract, DerivativeSurface
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...interchange import AdapterLoss, AdapterReport, AdapterStatus
from ...qualification import ReferenceArtifactManifest
from ._closure import CosmologyRealizationSignature
from ._products import (
    _validate_common,
    cosmology_product_content_id,
    CosmologyProductProvenance,
    ThermodynamicsHistory,
)
from ._scales import CosmologyScaleContract


DepositionSourceKind = Literal["native", "external"]
ProviderExecution = Literal["host", "subprocess"]


class EnergyDepositionStatus(IntEnum):
    SUCCESS = 0
    NO_INJECTION = 1
    NONPHYSICAL_INPUT = 2
    KERNEL_ENERGY_NONCLOSURE = 3
    NUMERICAL_FAILURE = 4


def _identifiers(values: tuple[str, ...], name: str, /) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if (
        not result
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must be non-empty unique identifiers.")
    return result


def _source_axis(
    one_plus_redshift: ArrayLike, name: str, /
) -> tuple[np.ndarray, np.ndarray, str]:
    source = np.asarray(one_plus_redshift, dtype=np.float64)
    if (
        source.ndim != 1
        or source.size < 2
        or np.any(~np.isfinite(source))
        or np.any(source <= 0.0)
    ):
        raise ValueError(
            f"{name} must be a positive finite vector with at least two nodes."
        )
    difference = np.diff(source)
    if np.all(difference < 0.0):
        permutation = np.arange(source.size)
        direction = "decreasing"
    elif np.all(difference > 0.0):
        permutation = np.arange(source.size - 1, -1, -1)
        direction = "increasing"
    else:
        raise ValueError(
            f"{name} must be strictly monotone; values are never sorted or clamped."
        )
    scale_factor = 1.0 / source[permutation]
    if np.any(np.diff(scale_factor) <= 0.0):
        raise ValueError("Canonical scale factor must be strictly increasing.")
    return scale_factor, permutation, direction


_NATIVE_DIFFERENTIATION = DerivativeContract.smooth(
    (
        DerivativeSurface.INPUT,
        DerivativeSurface.MODEL_PARAMETER,
        DerivativeSurface.PHYSICAL_PARAMETER,
        DerivativeSurface.STORED_VALUES,
    )
)


def _admit_external(
    source_kind: DepositionSourceKind,
    differentiation: DerivativeContract,
    manifest: ReferenceArtifactManifest | None,
    /,
    *,
    commercial_use: bool,
    redistribution: bool,
    training_use: bool,
    export: bool,
) -> bool:
    if source_kind not in ("native", "external"):
        raise ValueError("source_kind must be 'native' or 'external'.")
    if not isinstance(differentiation, DerivativeContract):
        raise TypeError("differentiation must be a DerivativeContract.")
    external = source_kind == "external"
    if external:
        if not isinstance(manifest, ReferenceArtifactManifest):
            raise TypeError(
                "External deposition products require a rights/checksum manifest."
            )
        manifest.require_rights(
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        if differentiation.supported_surfaces:
            raise ValueError(
                "External deposition products must declare constant differentiation."
            )
    elif manifest is not None and not isinstance(manifest, ReferenceArtifactManifest):
        raise TypeError("manifest must be a ReferenceArtifactManifest or None.")
    return external


def _trapezoid_weights(coordinate: Array, /) -> Array:
    spacing = jnp.diff(coordinate)
    return jnp.concatenate(
        (spacing[:1] / 2.0, (spacing[:-1] + spacing[1:]) / 2.0, spacing[-1:] / 2.0)
    )


class InjectionSpectrum(StrictModule, NonTrainableState):
    """Lossless species-resolved dN/dE on the provider's exact redshift and energy grid."""

    source_one_plus_redshift: Array
    canonical_from_source_index: Array
    scale_factors: Array
    energy_gev: Array
    differential_number_per_gev: Array
    manifest: ReferenceArtifactManifest | None
    differentiation: DerivativeContract
    species: tuple[str, ...] = eqx.field(static=True)
    source_axis_direction: str = eqx.field(static=True)
    source_kind: DepositionSourceKind = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)

    def __init__(
        self,
        one_plus_redshift: ArrayLike,
        energy_gev: ArrayLike,
        differential_number_per_gev: ArrayLike,
        species: tuple[str, ...],
        /,
        *,
        source_id: str,
        source_kind: DepositionSourceKind = "native",
        differentiation: DerivativeContract = _NATIVE_DIFFERENTIATION,
        manifest: ReferenceArtifactManifest | None = None,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        scale_factor, permutation, direction = _source_axis(
            one_plus_redshift, "Injection source 1+z"
        )
        energies = np.asarray(energy_gev, dtype=np.float64)
        species_ = _identifiers(species, "injected species")
        values = np.asarray(differential_number_per_gev, dtype=np.float64)
        expected = (scale_factor.size, len(species_), energies.size)
        if (
            energies.ndim != 1
            or energies.size < 2
            or np.any(~np.isfinite(energies))
            or np.any(energies <= 0.0)
            or np.any(np.diff(energies) <= 0.0)
            or values.shape != expected
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError(
                "Injection spectra require matching (redshift, species, energy) values "
                "on a positive increasing energy grid."
            )
        identifier = str(source_id).strip()
        if not identifier:
            raise ValueError("source_id must be non-empty.")
        external = _admit_external(
            source_kind,
            differentiation,
            manifest,
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        canonical_values = values[permutation]
        scale_array = jnp.asarray(scale_factor)
        energy_array = jnp.asarray(energies)
        values_array = jnp.asarray(canonical_values)
        if external:
            scale_array, energy_array, values_array = jax.tree.map(
                jax.lax.stop_gradient, (scale_array, energy_array, values_array)
            )
        source_redshift_array = jnp.asarray(one_plus_redshift)
        self.source_one_plus_redshift = (
            jax.lax.stop_gradient(source_redshift_array)
            if external
            else source_redshift_array
        )
        self.canonical_from_source_index = jnp.asarray(permutation, dtype=jnp.int32)
        self.scale_factors = scale_array
        self.energy_gev = energy_array
        self.differential_number_per_gev = values_array
        self.manifest = manifest
        self.differentiation = differentiation
        self.species = species_
        self.source_axis_direction = direction
        self.source_kind = source_kind
        self.source_id = identifier
        self.spectrum_id = canonical_fingerprint(
            {
                "kind": "species-resolved-injection-spectrum",
                "source_id": identifier,
                "source_kind": source_kind,
                "source_axis_direction": direction,
                "source_one_plus_redshift": array_tree_fingerprint(
                    np.asarray(one_plus_redshift)
                ),
                "canonical_arrays": array_tree_fingerprint(
                    (scale_factor, energies, canonical_values)
                ),
                "species": list(species_),
                "units": {
                    "energy": "GeV",
                    "spectrum": "injection^-1 GeV^-1",
                },
                "manifest": None if manifest is None else manifest.manifest_id,
                "differentiation": differentiation.contract_id,
            }
        )


class CascadeKernelEvidence(StrictModule, NonTrainableState):
    closure_residual_gev: Array
    relative_closure_residual: Array
    maximum_relative_closure_residual: Array
    valid: Array


class CascadeKernelProduct(StrictModule, NonTrainableState):
    """State-conditioned energy action; CMB borrowing is an independent kernel term."""

    scale_factors: Array
    energy_gev: Array
    state_values: Array
    deposited_energy_per_particle_gev: Array
    escaped_energy_per_particle_gev: Array
    borrowed_cmb_energy_per_particle_gev: Array
    evidence: CascadeKernelEvidence
    valid: Array
    status: Array
    manifest: ReferenceArtifactManifest | None
    differentiation: DerivativeContract
    species: tuple[str, ...] = eqx.field(static=True)
    deposition_channels: tuple[str, ...] = eqx.field(static=True)
    state_names: tuple[str, ...] = eqx.field(static=True)
    source_kind: DepositionSourceKind = eqx.field(static=True)
    relative_closure_tolerance: float = eqx.field(static=True)
    absolute_closure_tolerance_gev: float = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale_factors: ArrayLike,
        energy_gev: ArrayLike,
        state_values: ArrayLike,
        deposited_energy_per_particle_gev: ArrayLike,
        escaped_energy_per_particle_gev: ArrayLike,
        borrowed_cmb_energy_per_particle_gev: ArrayLike,
        species: tuple[str, ...],
        deposition_channels: tuple[str, ...],
        state_names: tuple[str, ...],
        /,
        *,
        source_kind: DepositionSourceKind = "native",
        differentiation: DerivativeContract = _NATIVE_DIFFERENTIATION,
        manifest: ReferenceArtifactManifest | None = None,
        relative_closure_tolerance: float = 1e-6,
        absolute_closure_tolerance_gev: float = 1e-12,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        scales = np.asarray(scale_factors, dtype=np.float64)
        energies = np.asarray(energy_gev, dtype=np.float64)
        states = np.asarray(state_values, dtype=np.float64)
        deposited = np.asarray(deposited_energy_per_particle_gev, dtype=np.float64)
        escaped = np.asarray(escaped_energy_per_particle_gev, dtype=np.float64)
        borrowed = np.asarray(borrowed_cmb_energy_per_particle_gev, dtype=np.float64)
        species_ = _identifiers(species, "cascade injected species")
        channels = _identifiers(deposition_channels, "deposition channels")
        state_names_ = _identifiers(state_names, "cascade state names")
        expected_kernel = (scales.size, len(channels), len(species_), energies.size)
        expected_tail = (scales.size, len(species_), energies.size)
        tolerance = float(relative_closure_tolerance)
        absolute_tolerance = float(absolute_closure_tolerance_gev)
        if (
            scales.ndim != 1
            or scales.size < 2
            or np.any(~np.isfinite(scales))
            or np.any(scales <= 0.0)
            or np.any(np.diff(scales) <= 0.0)
            or energies.ndim != 1
            or energies.size < 2
            or np.any(~np.isfinite(energies))
            or np.any(energies <= 0.0)
            or np.any(np.diff(energies) <= 0.0)
            or states.shape != (scales.size, len(state_names_))
            or np.any(~np.isfinite(states))
            or deposited.shape != expected_kernel
            or escaped.shape != expected_tail
            or borrowed.shape != expected_tail
            or np.any(~np.isfinite(deposited))
            or np.any(~np.isfinite(escaped))
            or np.any(~np.isfinite(borrowed))
            or np.any(deposited < 0.0)
            or np.any(escaped < 0.0)
            or np.any(borrowed < 0.0)
            or not np.isfinite(tolerance)
            or tolerance < 0.0
            or not np.isfinite(absolute_tolerance)
            or absolute_tolerance < 0.0
        ):
            raise ValueError(
                "Cascade kernel axes, state, values, or closure tolerances are invalid."
            )
        external = _admit_external(
            source_kind,
            differentiation,
            manifest,
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        closure = np.sum(deposited, axis=1) + escaped - energies[None, None, :] - borrowed
        scale = energies[None, None, :] + borrowed
        relative = np.abs(closure) / np.maximum(
            scale, absolute_tolerance if absolute_tolerance > 0.0 else 1.0
        )
        closure_valid = np.all(np.abs(closure) <= absolute_tolerance + tolerance * scale)
        arrays = tuple(
            jnp.asarray(value)
            for value in (scales, energies, states, deposited, escaped, borrowed)
        )
        if external:
            arrays = jax.tree.map(jax.lax.stop_gradient, arrays)
        (
            self.scale_factors,
            self.energy_gev,
            self.state_values,
            self.deposited_energy_per_particle_gev,
            self.escaped_energy_per_particle_gev,
            self.borrowed_cmb_energy_per_particle_gev,
        ) = arrays
        valid = jnp.asarray(closure_valid)
        self.evidence = CascadeKernelEvidence(
            jnp.asarray(closure),
            jnp.asarray(relative),
            jnp.asarray(np.max(relative, initial=0.0)),
            valid,
        )
        self.valid = valid
        self.status = jnp.asarray(
            int(
                EnergyDepositionStatus.SUCCESS
                if closure_valid
                else EnergyDepositionStatus.KERNEL_ENERGY_NONCLOSURE
            ),
            dtype=jnp.int32,
        )
        self.manifest = manifest
        self.differentiation = differentiation
        self.species = species_
        self.deposition_channels = channels
        self.state_names = state_names_
        self.source_kind = source_kind
        self.relative_closure_tolerance = tolerance
        self.absolute_closure_tolerance_gev = absolute_tolerance
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "state-conditioned-cascade-kernel",
                "arrays": array_tree_fingerprint(
                    (scales, energies, states, deposited, escaped, borrowed)
                ),
                "species": list(species_),
                "deposition_channels": list(channels),
                "state_names": list(state_names_),
                "source_kind": source_kind,
                "manifest": None if manifest is None else manifest.manifest_id,
                "differentiation": differentiation.contract_id,
                "relative_closure_tolerance": tolerance,
                "absolute_closure_tolerance_gev": absolute_tolerance,
            }
        )

    def apply(self, injection: InjectionSpectrum, /) -> EnergyDepositionLedger:
        """Apply the native table action with no interpolation, extrapolation, or clamp."""

        if not isinstance(injection, InjectionSpectrum):
            raise TypeError("injection must be an InjectionSpectrum.")
        if injection.species != self.species:
            raise ValueError("Injection and cascade species axes differ.")
        if (
            injection.scale_factors.shape != self.scale_factors.shape
            or injection.energy_gev.shape != self.energy_gev.shape
        ):
            raise ValueError("Injection and cascade table-axis shapes differ.")
        token = eqx.error_if(
            injection.differential_number_per_gev,
            jnp.any(injection.scale_factors != self.scale_factors)
            | jnp.any(injection.energy_gev != self.energy_gev),
            "Injection lies outside the exact cascade table domain; no clamp is available.",
        )
        quadrature = _trapezoid_weights(self.energy_gev)
        integrated_number = token * quadrature[None, None, :]
        injected = contract("ase,e->a", integrated_number, self.energy_gev)
        deposited = contract(
            "ase,acse->ac", integrated_number, self.deposited_energy_per_particle_gev
        )
        escaped = contract(
            "ase,ase->a", integrated_number, self.escaped_energy_per_particle_gev
        )
        borrowed = contract(
            "ase,ase->a", integrated_number, self.borrowed_cmb_energy_per_particle_gev
        )
        accounted = jnp.sum(deposited, axis=-1) + escaped
        residual = injected + borrowed - accounted
        finite = (
            jnp.isfinite(injected)
            & jnp.isfinite(escaped)
            & jnp.isfinite(borrowed)
            & jnp.all(jnp.isfinite(deposited), axis=-1)
        )
        available = injected + borrowed
        residual_scale = jnp.maximum(
            available,
            self.absolute_closure_tolerance_gev
            if self.absolute_closure_tolerance_gev > 0.0
            else 1.0,
        )
        closure_valid = jnp.abs(residual) <= (
            self.absolute_closure_tolerance_gev
            + self.relative_closure_tolerance * available
        )
        valid = finite & closure_valid & self.valid
        no_injection = injected == 0.0
        status = jnp.where(
            ~self.valid,
            int(EnergyDepositionStatus.KERNEL_ENERGY_NONCLOSURE),
            jnp.where(
                ~finite | ~closure_valid,
                int(EnergyDepositionStatus.NUMERICAL_FAILURE),
                jnp.where(
                    no_injection,
                    int(EnergyDepositionStatus.NO_INJECTION),
                    int(EnergyDepositionStatus.SUCCESS),
                ),
            ),
        ).astype(jnp.int32)
        evidence = EnergyDepositionEvidence(
            accounted,
            available,
            residual,
            jnp.abs(residual) / residual_scale,
            finite,
            closure_valid,
        )
        return EnergyDepositionLedger(
            self.scale_factors,
            injected,
            deposited,
            escaped,
            borrowed,
            evidence,
            valid,
            status,
            self.deposition_channels,
            canonical_fingerprint(
                {
                    "kind": "energy-deposition-ledger",
                    "kernel": self.kernel_id,
                    "injection": injection.spectrum_id,
                }
            ),
        )


class EnergyDepositionEvidence(StrictModule, NonTrainableState):
    accounted_energy_gev: Array
    available_energy_gev: Array
    closure_residual_gev: Array
    relative_closure_residual: Array
    finite: Array
    closure_valid: Array


class EnergyDepositionLedger(StrictModule, NonTrainableState):
    """Complete per-epoch energy accounting with deposited channels kept separate."""

    scale_factors: Array
    injected_energy_gev: Array
    deposited_energy_gev: Array
    escaped_energy_gev: Array
    borrowed_cmb_energy_gev: Array
    evidence: EnergyDepositionEvidence
    valid: Array
    status: Array
    deposition_channels: tuple[str, ...] = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)


class ThermodynamicsHistoryEvidence(StrictModule, NonTrainableState):
    electron_relation_residual: Array
    maximum_electron_relation_residual: Array
    physical_fraction_mask: Array
    finite_temperature_mask: Array
    valid: Array


class SpeciesResolvedThermodynamicsHistory(StrictModule, NonTrainableState):
    """H/He ion stages, total electrons, and matter temperature on increasing a."""

    source_one_plus_redshift: Array
    canonical_from_source_index: Array
    scale_factors: Array
    h_ii_fraction: Array
    he_ii_fraction: Array
    he_iii_fraction: Array
    electron_fraction: Array
    matter_temperature_k: Array
    helium_to_hydrogen_number_ratio: Array
    evidence: ThermodynamicsHistoryEvidence
    valid: Array
    status: Array
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance
    realization: CosmologyRealizationSignature
    manifest: ReferenceArtifactManifest | None
    source_axis_direction: str = eqx.field(static=True)
    history_id: str = eqx.field(static=True)

    def __init__(
        self,
        one_plus_redshift: ArrayLike,
        h_ii_fraction: ArrayLike,
        he_ii_fraction: ArrayLike,
        he_iii_fraction: ArrayLike,
        electron_fraction: ArrayLike,
        matter_temperature_k: ArrayLike,
        helium_to_hydrogen_number_ratio: ArrayLike,
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        realization: CosmologyRealizationSignature,
        /,
        *,
        manifest: ReferenceArtifactManifest | None = None,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        _validate_common(scale, provenance, realization)
        scale_factor, permutation, direction = _source_axis(
            one_plus_redshift, "Thermodynamics source 1+z"
        )
        source_values = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                h_ii_fraction,
                he_ii_fraction,
                he_iii_fraction,
                electron_fraction,
                matter_temperature_k,
            )
        )
        if any(value.shape != (scale_factor.size,) for value in source_values):
            raise ValueError("Thermodynamics fields must match the source redshift axis.")
        helium_ratio = np.asarray(helium_to_hydrogen_number_ratio, dtype=np.float64)
        if (
            helium_ratio.shape != ()
            or not np.isfinite(helium_ratio)
            or helium_ratio < 0.0
        ):
            raise ValueError(
                "Helium-to-hydrogen number ratio must be finite and non-negative."
            )
        values = tuple(value[permutation] for value in source_values)
        h_ii, he_ii, he_iii, electron, temperature = values
        fraction_mask = (
            np.isfinite(h_ii)
            & np.isfinite(he_ii)
            & np.isfinite(he_iii)
            & np.isfinite(electron)
            & (h_ii >= 0.0)
            & (h_ii <= 1.0)
            & (he_ii >= 0.0)
            & (he_iii >= 0.0)
            & (he_ii + he_iii <= 1.0)
            & (electron >= 0.0)
        )
        temperature_mask = np.isfinite(temperature) & (temperature >= 0.0)
        relation = h_ii + helium_ratio * (he_ii + 2.0 * he_iii)
        relation_residual = electron - relation
        if not np.all(fraction_mask):
            raise ValueError("H/He ion fractions are outside their physical domain.")
        if not np.all(temperature_mask):
            raise ValueError("Matter temperature must be finite and non-negative.")
        if not np.allclose(relation_residual, 0.0, rtol=1e-8, atol=1e-10):
            raise ValueError(
                "Electron fraction must equal x_HII + n_He/n_H * (x_HeII + 2*x_HeIII)."
            )
        external = provenance.source_kind == "external"
        if external:
            if not isinstance(manifest, ReferenceArtifactManifest):
                raise TypeError("External thermodynamics history requires a manifest.")
            manifest.require_rights(
                commercial_use=commercial_use,
                redistribution=redistribution,
                training_use=training_use,
                export=export,
            )
            if manifest.manifest_id not in provenance.parent_product_ids:
                raise ValueError(
                    "External thermodynamics provenance must name its rights manifest."
                )
            if provenance.differentiation.supported_surfaces:
                raise ValueError(
                    "External thermodynamics history must be constant under differentiation."
                )
        elif manifest is not None and not isinstance(manifest, ReferenceArtifactManifest):
            raise TypeError("manifest must be a ReferenceArtifactManifest or None.")
        arrays = tuple(
            jnp.asarray(value)
            for value in (scale_factor, h_ii, he_ii, he_iii, electron, temperature)
        )
        helium_array = jnp.asarray(helium_ratio)
        if external:
            arrays = jax.tree.map(jax.lax.stop_gradient, arrays)
            helium_array = jax.lax.stop_gradient(helium_array)
        (
            self.scale_factors,
            self.h_ii_fraction,
            self.he_ii_fraction,
            self.he_iii_fraction,
            self.electron_fraction,
            self.matter_temperature_k,
        ) = arrays
        source_redshift_array = jnp.asarray(one_plus_redshift)
        self.source_one_plus_redshift = (
            jax.lax.stop_gradient(source_redshift_array)
            if external
            else source_redshift_array
        )
        self.canonical_from_source_index = jnp.asarray(permutation, dtype=jnp.int32)
        self.helium_to_hydrogen_number_ratio = helium_array
        valid = jnp.asarray(True)
        self.evidence = ThermodynamicsHistoryEvidence(
            jnp.asarray(relation_residual),
            jnp.asarray(np.max(np.abs(relation_residual), initial=0.0)),
            jnp.asarray(fraction_mask),
            jnp.asarray(temperature_mask),
            valid,
        )
        self.valid = valid
        self.status = jnp.asarray(int(EnergyDepositionStatus.SUCCESS), dtype=jnp.int32)
        self.scale = scale
        self.provenance = provenance
        self.realization = realization
        self.manifest = manifest
        self.source_axis_direction = direction
        self.history_id = canonical_fingerprint(
            {
                "kind": "species-resolved-thermodynamics-history",
                "source_one_plus_redshift": array_tree_fingerprint(
                    np.asarray(one_plus_redshift)
                ),
                "canonical_arrays": array_tree_fingerprint(
                    (scale_factor, *values, helium_ratio)
                ),
                "source_axis_direction": direction,
                "scale": scale.scale_id,
                "provenance": provenance.provenance_id,
                "realization": realization.content_id(),
                "manifest": None if manifest is None else manifest.manifest_id,
            }
        )


class ExternalEnergyDepositionProviderResult(StrictModule, NonTrainableState):
    """Rights-qualified host/subprocess history and its complete energy ledger."""

    history: SpeciesResolvedThermodynamicsHistory
    ledger: EnergyDepositionLedger
    manifest: ReferenceArtifactManifest
    valid: Array
    status: Array
    provider: str = eqx.field(static=True)
    provider_version: str = eqx.field(static=True)
    execution: ProviderExecution = eqx.field(static=True)
    return_code: int = eqx.field(static=True)
    standard_output: str = eqx.field(static=True)
    standard_error: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        history: SpeciesResolvedThermodynamicsHistory,
        ledger: EnergyDepositionLedger,
        manifest: ReferenceArtifactManifest,
        /,
        *,
        provider: str,
        provider_version: str,
        execution: ProviderExecution,
        return_code: int = 0,
        standard_output: str = "",
        standard_error: str = "",
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        if not isinstance(
            history, SpeciesResolvedThermodynamicsHistory
        ) or not isinstance(ledger, EnergyDepositionLedger):
            raise TypeError(
                "Provider result requires the detailed history and energy ledger."
            )
        if not isinstance(manifest, ReferenceArtifactManifest):
            raise TypeError("Provider result requires a ReferenceArtifactManifest.")
        provider_ = str(provider).strip()
        version_ = str(provider_version).strip()
        if not provider_ or not version_:
            raise ValueError("Provider name and version must be non-empty.")
        if execution not in ("host", "subprocess"):
            raise ValueError("Provider execution must be 'host' or 'subprocess'.")
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            raise TypeError("return_code must be an integer.")
        manifest.require_rights(
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
        if (
            history.manifest is None
            or history.manifest.manifest_id != manifest.manifest_id
        ):
            raise ValueError(
                "History and provider result must use the same admitted manifest."
            )
        if history.scale_factors.shape != ledger.scale_factors.shape:
            raise ValueError("Provider history and ledger axis shapes differ.")
        axis_equal = jnp.all(history.scale_factors == ledger.scale_factors)
        valid = history.valid & jnp.all(ledger.valid) & axis_equal & (return_code == 0)
        status = jnp.where(
            valid,
            int(EnergyDepositionStatus.SUCCESS),
            int(EnergyDepositionStatus.NUMERICAL_FAILURE),
        ).astype(jnp.int32)
        self.history = history
        self.ledger = ledger
        self.manifest = manifest
        self.valid = valid
        self.status = status
        self.provider = provider_
        self.provider_version = version_
        self.execution = execution
        self.return_code = return_code
        self.standard_output = str(standard_output)
        self.standard_error = str(standard_error)
        self.result_id = canonical_fingerprint(
            {
                "kind": "external-energy-deposition-provider-result",
                "history": history.history_id,
                "ledger": ledger.ledger_id,
                "manifest": manifest.manifest_id,
                "provider": provider_,
                "provider_version": version_,
                "execution": execution,
                "return_code": return_code,
            }
        )


def project_to_thermodynamics_history(
    history: SpeciesResolvedThermodynamicsHistory,
    opacity_derivative: ArrayLike,
    visibility: ArrayLike,
    /,
) -> tuple[ThermodynamicsHistory, AdapterReport]:
    """Collapse resolved ion stages to x_e and explicitly report that semantic loss."""

    if not isinstance(history, SpeciesResolvedThermodynamicsHistory):
        raise TypeError("history must be SpeciesResolvedThermodynamicsHistory.")
    opacity = np.asarray(opacity_derivative, dtype=np.float64)
    visibility_ = np.asarray(visibility, dtype=np.float64)
    expected = (history.scale_factors.size,)
    if (
        opacity.shape != expected
        or visibility_.shape != expected
        or np.any(~np.isfinite(opacity))
        or np.any(~np.isfinite(visibility_))
        or np.any(visibility_ < 0.0)
    ):
        raise ValueError(
            "Projection opacity and visibility must be finite vectors on the history axis."
        )
    projected = ThermodynamicsHistory(
        history.scale_factors,
        history.electron_fraction,
        history.matter_temperature_k,
        opacity_derivative,
        visibility,
        history.scale,
        history.provenance,
        history.realization,
    )
    target_id = cosmology_product_content_id(projected)
    loss = AdapterLoss(
        "ionization.{h_ii_fraction,he_ii_fraction,he_iii_fraction}",
        "export",
        "transformed",
        "Resolved hydrogen and helium ion stages are collapsed to the existing total free-electron fraction field.",
        changes_interpretation=False,
        affected_capability_ids=("species-resolved-recombination-history",),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "phydrax-species-resolved-thermodynamics-history",
        "phydrax-thermodynamics-history",
        source_id=history.history_id,
        target_id=target_id,
        coordinate_mapping=("scale_factors -> scale_factors",),
        preserved_fields=(
            "electron_fraction -> ionization_fraction",
            "matter_temperature_k -> baryon_temperature",
            "scale",
            "realization",
            "provenance",
        ),
        assumptions=(
            "caller supplied opacity_derivative and visibility on the identical scale-factor grid",
        ),
        losses=(loss,),
        stage="thermodynamics-projection",
    )
    return projected, report


__all__ = [
    "CascadeKernelEvidence",
    "CascadeKernelProduct",
    "DepositionSourceKind",
    "EnergyDepositionEvidence",
    "EnergyDepositionLedger",
    "EnergyDepositionStatus",
    "ExternalEnergyDepositionProviderResult",
    "InjectionSpectrum",
    "ProviderExecution",
    "SpeciesResolvedThermodynamicsHistory",
    "ThermodynamicsHistoryEvidence",
    "project_to_thermodynamics_history",
]
