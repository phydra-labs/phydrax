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

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import DifferentiationContract
from ...qualification import ReferenceArtifactManifest
from ..astrophysics._operators import SpectralField
from ..astrophysics._photometry import ObservationDataProvenance


DarkMatterProcessKind = Literal["annihilation", "decay"]
ProviderExecution = Literal["host", "subprocess"]


class YieldProviderStatus(IntEnum):
    SUCCESS = 0
    NONPHYSICAL_PRODUCT = 1
    NUMERICAL_FAILURE = 2


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    host = np.asarray(value, dtype=np.float64)
    if host.shape != () or not np.isfinite(host) or host <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return jnp.asarray(value)


def _trapezoid_weights(coordinate: Array, /) -> Array:
    spacing = jnp.diff(coordinate)
    return jnp.concatenate(
        (spacing[:1] / 2.0, (spacing[:-1] + spacing[1:]) / 2.0, spacing[-1:] / 2.0)
    )


class AnnihilationProcessDescriptor(StrictModule):
    """Self-conjugate or symmetric particle/antiparticle annihilation in GeV-cgs."""

    mass_gev: Array
    velocity_averaged_cross_section_cm3_s: Array
    self_conjugate: bool = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    kind: DarkMatterProcessKind = eqx.field(static=True, default="annihilation")

    def __init__(
        self,
        mass_gev: ArrayLike,
        velocity_averaged_cross_section_cm3_s: ArrayLike,
        /,
        *,
        self_conjugate: bool = True,
    ):
        if not isinstance(self_conjugate, bool):
            raise TypeError("self_conjugate must be a boolean.")
        mass = _positive_scalar(mass_gev, "dark-matter mass in GeV")
        cross_section = _positive_scalar(
            velocity_averaged_cross_section_cm3_s,
            "velocity-averaged annihilation cross section in cm^3 s^-1",
        )
        self.mass_gev = mass
        self.velocity_averaged_cross_section_cm3_s = cross_section
        self.self_conjugate = self_conjugate
        self.process_id = canonical_fingerprint(
            {
                "kind": "dark-matter-annihilation",
                "mass_gev": array_tree_fingerprint(mass),
                "velocity_averaged_cross_section_cm3_s": array_tree_fingerprint(
                    cross_section
                ),
                "self_conjugate": self_conjugate,
            }
        )


class DecayProcessDescriptor(StrictModule):
    """One-body dark-matter decay in GeV-cgs."""

    mass_gev: Array
    lifetime_s: Array
    process_id: str = eqx.field(static=True)
    kind: DarkMatterProcessKind = eqx.field(static=True, default="decay")

    def __init__(self, mass_gev: ArrayLike, lifetime_s: ArrayLike, /):
        mass = _positive_scalar(mass_gev, "dark-matter mass in GeV")
        lifetime = _positive_scalar(lifetime_s, "dark-matter lifetime in s")
        self.mass_gev = mass
        self.lifetime_s = lifetime
        self.process_id = canonical_fingerprint(
            {
                "kind": "dark-matter-decay",
                "mass_gev": array_tree_fingerprint(mass),
                "lifetime_s": array_tree_fingerprint(lifetime),
            }
        )


class ExactLineTable(StrictModule, NonTrainableState):
    """Unbroadened particles per process at exact line energies."""

    energy_gev: Array
    multiplicity: Array
    table_id: str = eqx.field(static=True)

    def __init__(self, energy_gev: ArrayLike, multiplicity: ArrayLike, /):
        energy_host = np.asarray(energy_gev, dtype=np.float64)
        multiplicity_host = np.asarray(multiplicity, dtype=np.float64)
        if energy_host.ndim != 1 or multiplicity_host.shape != energy_host.shape:
            raise ValueError("Line energies and multiplicities must be matching vectors.")
        if (
            np.any(~np.isfinite(energy_host))
            or np.any(~np.isfinite(multiplicity_host))
            or np.any(energy_host <= 0.0)
            or np.any(multiplicity_host < 0.0)
            or (energy_host.size > 1 and np.any(np.diff(energy_host) < 0.0))
        ):
            raise ValueError(
                "Line energies must be positive, finite, and ordered; multiplicities must be finite and non-negative."
            )
        self.energy_gev = jnp.asarray(energy_gev)
        self.multiplicity = jnp.asarray(multiplicity)
        self.table_id = canonical_fingerprint(
            {
                "kind": "exact-particle-line-table",
                "units": {"energy": "GeV", "multiplicity": "event^-1"},
                "arrays": array_tree_fingerprint((energy_host, multiplicity_host)),
            }
        )


class YieldUncertainty(StrictModule, NonTrainableState):
    """Independent absolute one-sigma numerical uncertainty on a yield product."""

    continuum_standard_deviation: Array
    line_standard_deviation: Array
    uncertainty_id: str = eqx.field(static=True)

    def __init__(
        self,
        continuum_standard_deviation: ArrayLike,
        line_standard_deviation: ArrayLike,
        /,
    ):
        continuum_host = np.asarray(continuum_standard_deviation, dtype=np.float64)
        line_host = np.asarray(line_standard_deviation, dtype=np.float64)
        if continuum_host.ndim != 1 or line_host.ndim != 1:
            raise ValueError("Yield uncertainties must be vectors.")
        if (
            np.any(~np.isfinite(continuum_host))
            or np.any(~np.isfinite(line_host))
            or np.any(continuum_host < 0.0)
            or np.any(line_host < 0.0)
        ):
            raise ValueError("Yield uncertainties must be finite and non-negative.")
        self.continuum_standard_deviation = jnp.asarray(continuum_standard_deviation)
        self.line_standard_deviation = jnp.asarray(line_standard_deviation)
        self.uncertainty_id = canonical_fingerprint(
            {
                "kind": "independent-absolute-yield-uncertainty",
                "arrays": array_tree_fingerprint((continuum_host, line_host)),
            }
        )


class YieldIntegralEvidence(StrictModule, NonTrainableState):
    continuum_multiplicity: Array
    line_multiplicity: Array
    total_multiplicity: Array
    continuum_energy_gev: Array
    line_energy_gev: Array
    total_energy_gev: Array
    multiplicity_standard_deviation: Array
    energy_standard_deviation_gev: Array
    valid: Array


class ParticleYieldSpectrum(StrictModule, NonTrainableState):
    """Continuum SpectralField and exact lines for one stable product species."""

    continuum: SpectralField
    lines: ExactLineTable
    uncertainty: YieldUncertainty
    product_species: str = eqx.field(static=True)
    yield_id: str = eqx.field(static=True)

    def __init__(
        self,
        continuum: SpectralField,
        lines: ExactLineTable,
        uncertainty: YieldUncertainty,
        /,
        *,
        product_species: str,
    ):
        if not isinstance(continuum, SpectralField):
            raise TypeError("continuum must be a SpectralField.")
        if not isinstance(lines, ExactLineTable) or not isinstance(
            uncertainty, YieldUncertainty
        ):
            raise TypeError("lines and uncertainty must use the yield product types.")
        species = _identifier(product_species, "product_species")
        energy_host = np.asarray(continuum.coordinate, dtype=np.float64)
        values_host = np.asarray(continuum.values, dtype=np.float64)
        if (
            energy_host.ndim != 1
            or energy_host.size < 2
            or values_host.shape != energy_host.shape
            or np.any(~np.isfinite(energy_host))
            or np.any(~np.isfinite(values_host))
            or np.any(energy_host <= 0.0)
            or np.any(np.diff(energy_host) <= 0.0)
            or np.any(values_host < 0.0)
        ):
            raise ValueError(
                "Continuum yield requires positive increasing energies and finite "
                "non-negative differential multiplicity."
            )
        if (
            continuum.coordinate_unit != "GeV"
            or continuum.value_unit != "event^-1 GeV^-1"
        ):
            raise ValueError("Continuum yield units must be GeV and event^-1 GeV^-1.")
        if uncertainty.continuum_standard_deviation.shape != energy_host.shape:
            raise ValueError(
                "Continuum uncertainty must match the continuum energy grid."
            )
        if uncertainty.line_standard_deviation.shape != lines.energy_gev.shape:
            raise ValueError("Line uncertainty must match the exact line table.")
        constant = continuum.provenance.differentiation.contract_id == (
            DifferentiationContract.constant().contract_id
        )
        coordinate = (
            jax.lax.stop_gradient(continuum.coordinate)
            if constant
            else continuum.coordinate
        )
        values = jax.lax.stop_gradient(continuum.values) if constant else continuum.values
        self.continuum = SpectralField(
            coordinate,
            values,
            continuum.provenance,
            coordinate_unit="GeV",
            value_unit="event^-1 GeV^-1",
            field_id=continuum.field_id,
        )
        self.lines = (
            ExactLineTable(
                jax.lax.stop_gradient(lines.energy_gev),
                jax.lax.stop_gradient(lines.multiplicity),
            )
            if constant
            else lines
        )
        self.uncertainty = (
            YieldUncertainty(
                jax.lax.stop_gradient(uncertainty.continuum_standard_deviation),
                jax.lax.stop_gradient(uncertainty.line_standard_deviation),
            )
            if constant
            else uncertainty
        )
        self.product_species = species
        self.yield_id = canonical_fingerprint(
            {
                "kind": "particle-yield-spectrum",
                "species": species,
                "continuum": array_tree_fingerprint((energy_host, values_host)),
                "lines": lines.table_id,
                "uncertainty": uncertainty.uncertainty_id,
                "provenance": continuum.provenance.provenance_id,
            }
        )

    def integrate(self, /) -> YieldIntegralEvidence:
        weights = _trapezoid_weights(self.continuum.coordinate)
        continuum_multiplicity = contract("e,e->", weights, self.continuum.values)
        line_multiplicity = jnp.sum(self.lines.multiplicity)
        continuum_energy = contract(
            "e,e,e->", weights, self.continuum.coordinate, self.continuum.values
        )
        line_energy = contract("l,l->", self.lines.energy_gev, self.lines.multiplicity)
        multiplicity_variance = contract(
            "e,e->",
            weights**2,
            self.uncertainty.continuum_standard_deviation**2,
        ) + jnp.sum(self.uncertainty.line_standard_deviation**2)
        energy_variance = contract(
            "e,e,e->",
            weights**2,
            self.continuum.coordinate**2,
            self.uncertainty.continuum_standard_deviation**2,
        ) + contract(
            "l,l->",
            self.lines.energy_gev**2,
            self.uncertainty.line_standard_deviation**2,
        )
        total_multiplicity = continuum_multiplicity + line_multiplicity
        total_energy = continuum_energy + line_energy
        valid = (
            jnp.isfinite(total_multiplicity)
            & jnp.isfinite(total_energy)
            & (total_multiplicity >= 0.0)
            & (total_energy >= 0.0)
        )
        return YieldIntegralEvidence(
            continuum_multiplicity,
            line_multiplicity,
            total_multiplicity,
            continuum_energy,
            line_energy,
            total_energy,
            jnp.sqrt(multiplicity_variance),
            jnp.sqrt(energy_variance),
            valid,
        )


def mix_particle_yields(
    components: tuple[tuple[float, ParticleYieldSpectrum], ...],
    /,
    *,
    mixture_id: str,
) -> ParticleYieldSpectrum:
    """Form a branching-fraction mixture without broadening or merging exact lines."""

    identifier = _identifier(mixture_id, "mixture_id")
    if not isinstance(components, tuple) or not components:
        raise ValueError("Yield mixtures require a non-empty component tuple.")
    fractions = np.asarray([item[0] for item in components], dtype=np.float64)
    spectra = tuple(item[1] for item in components)
    if any(not isinstance(item, ParticleYieldSpectrum) for item in spectra):
        raise TypeError("Every mixture component must be a ParticleYieldSpectrum.")
    if (
        np.any(~np.isfinite(fractions))
        or np.any(fractions < 0.0)
        or not np.isclose(np.sum(fractions), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "Branching fractions must be finite, non-negative, and sum to one."
        )
    reference = spectra[0]
    if any(item.product_species != reference.product_species for item in spectra[1:]):
        raise ValueError("Yield mixture product species must match.")
    reference_energy = np.asarray(reference.continuum.coordinate)
    if any(
        item.continuum.coordinate_unit != reference.continuum.coordinate_unit
        or item.continuum.value_unit != reference.continuum.value_unit
        or not np.array_equal(np.asarray(item.continuum.coordinate), reference_energy)
        for item in spectra[1:]
    ):
        raise ValueError("Yield mixture continuum grids and units must match exactly.")
    weights = jnp.asarray(fractions, dtype=reference.continuum.values.dtype)
    continuum_values = contract(
        "c,ce->e", weights, jnp.stack(tuple(item.continuum.values for item in spectra))
    )
    continuum_variance = contract(
        "c,ce->e",
        weights**2,
        jnp.stack(
            tuple(item.uncertainty.continuum_standard_deviation**2 for item in spectra)
        ),
    )
    line_energies = jnp.concatenate(tuple(item.lines.energy_gev for item in spectra))
    line_multiplicities = jnp.concatenate(
        tuple(weight * item.lines.multiplicity for weight, item in zip(weights, spectra))
    )
    line_variances = jnp.concatenate(
        tuple(
            (weight * item.uncertainty.line_standard_deviation) ** 2
            for weight, item in zip(weights, spectra)
        )
    )
    order = np.argsort(np.asarray(line_energies), kind="stable")
    line_energies = line_energies[jnp.asarray(order)]
    line_multiplicities = line_multiplicities[jnp.asarray(order)]
    line_standard_deviation = jnp.sqrt(line_variances[jnp.asarray(order)])
    differentiation = reference.continuum.provenance.differentiation.meet(
        *(item.continuum.provenance.differentiation for item in spectra[1:])
    )
    parent_ids = tuple(item.yield_id for item in spectra)
    provenance = ObservationDataProvenance(
        producer="phydrax",
        producer_version="branching-mixture",
        source_id=identifier,
        checksum=canonical_fingerprint(
            {
                "components": list(parent_ids),
                "fractions": fractions.tolist(),
            }
        ),
        license_id="derived-from-declared-components",
        differentiation=differentiation,
    )
    continuum = SpectralField(
        reference.continuum.coordinate,
        continuum_values,
        provenance,
        coordinate_unit="GeV",
        value_unit="event^-1 GeV^-1",
        field_id=identifier,
    )
    return ParticleYieldSpectrum(
        continuum,
        ExactLineTable(line_energies, line_multiplicities),
        YieldUncertainty(jnp.sqrt(continuum_variance), line_standard_deviation),
        product_species=reference.product_species,
    )


class ExternalYieldProviderResult(StrictModule, NonTrainableState):
    """Rights-admitted constant yield returned by a host or subprocess provider."""

    spectrum: ParticleYieldSpectrum
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
        spectrum: ParticleYieldSpectrum,
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
        if not isinstance(spectrum, ParticleYieldSpectrum) or not isinstance(
            manifest, ReferenceArtifactManifest
        ):
            raise TypeError("External yield results require a spectrum and manifest.")
        provider_ = _identifier(provider, "provider")
        version_ = _identifier(provider_version, "provider_version")
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
        provenance = spectrum.continuum.provenance
        if (
            provenance.checksum != manifest.checksum
            or provenance.license_id != manifest.license_id
        ):
            raise ValueError(
                "Yield provenance checksum/license must match its rights manifest."
            )
        if (
            provenance.differentiation.contract_id
            != DifferentiationContract.constant().contract_id
        ):
            raise ValueError(
                "External yield products must declare constant differentiation."
            )
        evidence = spectrum.integrate()
        valid = evidence.valid & (return_code == 0)
        status = jnp.where(
            return_code != 0,
            int(YieldProviderStatus.NUMERICAL_FAILURE),
            jnp.where(
                evidence.valid,
                int(YieldProviderStatus.SUCCESS),
                int(YieldProviderStatus.NONPHYSICAL_PRODUCT),
            ),
        ).astype(jnp.int32)
        self.spectrum = spectrum
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
                "kind": "external-yield-provider-result",
                "yield": spectrum.yield_id,
                "manifest": manifest.manifest_id,
                "provider": provider_,
                "provider_version": version_,
                "execution": execution,
                "return_code": return_code,
            }
        )


__all__ = [
    "AnnihilationProcessDescriptor",
    "DarkMatterProcessKind",
    "DecayProcessDescriptor",
    "ExactLineTable",
    "ExternalYieldProviderResult",
    "ParticleYieldSpectrum",
    "ProviderExecution",
    "YieldIntegralEvidence",
    "YieldProviderStatus",
    "YieldUncertainty",
    "mix_particle_yields",
]
