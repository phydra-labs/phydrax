#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rights-qualified thermal dark-sector kernels, HTL response, LPM, and EOS."""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein
import phydrax.linalg as la

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ScientificArtifactEnvelope
from ...qualification import ReferenceArtifactManifest
from ...units import derived_unit
from ..relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)


class ThermalKernelStatus(IntEnum):
    """Status shared by bounded thermal-kernel consumers."""

    SUCCESS = 0
    OUTSIDE_TEMPERATURE_SUPPORT = 1
    NONFINITE = 2
    LINEAR_SOLVE_FAILURE = 3
    NEGATIVE_SUBTRACTED_RATE = 4
    EOS_INCONSISTENT = 5


def _array_bytes(value: np.ndarray, /) -> bytes:
    contiguous = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(header).to_bytes(8, "big") + header + contiguous.tobytes(order="C")


def thermal_kernel_payload_bytes(
    temperature: ArrayLike,
    momentum: ArrayLike,
    frequency: ArrayLike,
    thermal_masses: ArrayLike,
    widths: ArrayLike,
    self_energies: ArrayLike,
    spectral_functions: ArrayLike,
    screening_longitudinal: ArrayLike,
    screening_transverse: ArrayLike,
    rates: ArrayLike,
    pressure: ArrayLike,
    energy_density: ArrayLike,
    entropy_density: ArrayLike,
    eos_covariance: ArrayLike,
    /,
    *,
    species_plan_ids: Sequence[str],
    rate_channel_ids: Sequence[str],
) -> bytes:
    """Return deterministic bytes covering every imported thermal table value."""

    arrays = (
        temperature,
        momentum,
        frequency,
        thermal_masses,
        widths,
        self_energies,
        spectral_functions,
        screening_longitudinal,
        screening_transverse,
        rates,
        pressure,
        energy_density,
        entropy_density,
        eos_covariance,
    )
    names = (
        "temperature",
        "momentum",
        "frequency",
        "thermal_masses",
        "widths",
        "self_energies",
        "spectral_functions",
        "screening_longitudinal",
        "screening_transverse",
        "rates",
        "pressure",
        "energy_density",
        "entropy_density",
        "eos_covariance",
    )
    payload = bytearray(b"phydrax-thermal-kernel\0")
    for name, values in (
        ("species_plan_ids", species_plan_ids),
        ("rate_channel_ids", rate_channel_ids),
    ):
        encoded_name = name.encode("utf-8")
        encoded_values = json.dumps(
            tuple(str(value) for value in values),
            separators=(",", ":"),
        ).encode("utf-8")
        payload.extend(len(encoded_name).to_bytes(2, "big"))
        payload.extend(encoded_name)
        payload.extend(len(encoded_values).to_bytes(8, "big"))
        payload.extend(encoded_values)
    for name, value in zip(names, arrays, strict=True):
        encoded = name.encode("utf-8")
        payload.extend(len(encoded).to_bytes(2, "big"))
        payload.extend(encoded)
        payload.extend(_array_bytes(np.asarray(value)))
    return bytes(payload)


class ThermalKernelEvidence(StrictModule, NonTrainableState):
    """Host-established validity of one complete bounded thermal artifact."""

    finite: bool = eqx.field(static=True)
    axes_monotone: bool = eqx.field(static=True)
    nonnegative_widths: bool = eqx.field(static=True)
    nonnegative_spectral_functions: bool = eqx.field(static=True)
    nonnegative_rates: bool = eqx.field(static=True)
    covariance_symmetric: bool = eqx.field(static=True)
    covariance_positive_semidefinite: bool = eqx.field(static=True)
    thermodynamically_consistent: bool = eqx.field(static=True)
    thermodynamically_stable: bool = eqx.field(static=True)
    maximum_entropy_identity_residual: float = eqx.field(static=True)
    maximum_energy_identity_residual: float = eqx.field(static=True)
    minimum_heat_capacity: float = eqx.field(static=True)
    qualified: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class ThermalKernelArtifact(StrictModule, NonTrainableState):
    """Complete thermal tables with units, frame, rights, and provenance.

    Tables use the supplied contract's temperature, physical-momentum, angular-
    frequency, energy, inverse-time, and energy-density units. External and
    native-generated table arrays are immutable stop-gradient data.
    Differentiable analytic HTL and LPM evaluations remain separate plan families
    rather than silently differentiating through tabulated values.
    """

    temperature: Array
    momentum: Array
    frequency: Array
    thermal_masses: Array
    widths: Array
    self_energies: Array
    spectral_functions: Array
    screening_longitudinal: Array
    screening_transverse: Array
    rates: Array
    pressure: Array
    energy_density: Array
    entropy_density: Array
    eos_covariance: Array
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    frame_token: Array
    source_manifest: ReferenceArtifactManifest | None
    source_artifact: ScientificArtifactEnvelope | None
    evidence: ThermalKernelEvidence
    species_count: int = eqx.field(static=True)
    rate_count: int = eqx.field(static=True)
    species_plan_ids: tuple[str, ...] = eqx.field(static=True)
    rate_channel_ids: tuple[str, ...] = eqx.field(static=True)
    source_kind: str = eqx.field(static=True)
    requested_use: tuple[tuple[str, bool], ...] = eqx.field(static=True)
    requested_use_id: str = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperature: ArrayLike,
        momentum: ArrayLike,
        frequency: ArrayLike,
        thermal_masses: ArrayLike,
        widths: ArrayLike,
        self_energies: ArrayLike,
        spectral_functions: ArrayLike,
        screening_longitudinal: ArrayLike,
        screening_transverse: ArrayLike,
        rates: ArrayLike,
        pressure: ArrayLike,
        energy_density: ArrayLike,
        entropy_density: ArrayLike,
        eos_covariance: ArrayLike,
        units: RelativisticUnitContract,
        frame: LocalRelativisticFramePlan,
        /,
        *,
        species_plan_ids: Sequence[str],
        rate_channel_ids: Sequence[str],
        source_kind: str,
        source_manifest: ReferenceArtifactManifest | None = None,
        source_artifact: ScientificArtifactEnvelope | None = None,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
        thermodynamic_tolerance: float = 5.0e-3,
        covariance_tolerance: float = 1.0e-12,
    ):
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be a RelativisticUnitContract.")
        if (
            not isinstance(frame, LocalRelativisticFramePlan)
            or frame.units.contract_id != units.contract_id
        ):
            raise ValueError("Thermal kernel frame and unit contract disagree.")
        if frame.geometry.leading_shape != ():
            raise ValueError(
                "Thermal kernel tables require one scalar local-frame realization."
            )
        source_kind_ = str(source_kind).strip()
        if source_kind_ not in ("native-analytic", "external-table"):
            raise ValueError("source_kind must be 'native-analytic' or 'external-table'.")
        rights_values = (commercial_use, redistribution, training_use, export)
        if any(not isinstance(value, bool) for value in rights_values):
            raise TypeError("Thermal-kernel requested-use flags must be Boolean.")
        requested = tuple(
            zip(
                ("commercial_use", "redistribution", "training_use", "export"),
                rights_values,
                strict=True,
            )
        )
        arrays = tuple(
            np.asarray(value)
            for value in (
                temperature,
                momentum,
                frequency,
                thermal_masses,
                widths,
                self_energies,
                spectral_functions,
                screening_longitudinal,
                screening_transverse,
                rates,
                pressure,
                energy_density,
                entropy_density,
                eos_covariance,
            )
        )
        (
            temperature_,
            momentum_,
            frequency_,
            masses,
            widths_,
            self_energy,
            spectral,
            longitudinal,
            transverse,
            rates_,
            pressure_,
            energy_,
            entropy_,
            covariance,
        ) = arrays
        if (
            temperature_.ndim != 1
            or temperature_.size < 3
            or momentum_.ndim != 1
            or momentum_.size == 0
            or frequency_.ndim != 1
            or frequency_.size == 0
        ):
            raise ValueError(
                "Thermal temperature, momentum, and frequency axes have invalid rank or size."
            )
        nt, nk, nw = temperature_.size, momentum_.size, frequency_.size
        if masses.ndim != 2 or masses.shape[1] != nt:
            raise ValueError("thermal_masses must have shape (species, temperature).")
        species_count = int(masses.shape[0])
        if species_count == 0 or widths_.shape != masses.shape:
            raise ValueError("widths must match a nonempty thermal-mass table.")
        expected_spectral = (species_count, nt, nk, nw)
        if self_energy.shape != expected_spectral or spectral.shape != expected_spectral:
            raise ValueError(
                "Self energies and spectral functions must have shape (species,T,k,omega)."
            )
        if longitudinal.shape != (nt, nk, nw) or transverse.shape != (nt, nk, nw):
            raise ValueError(
                "Screening tables must have shape (temperature,momentum,frequency)."
            )
        if rates_.ndim != 2 or rates_.shape[1] != nt or rates_.shape[0] == 0:
            raise ValueError("rates must have shape (channel, temperature).")
        if pressure_.shape != (nt,) or energy_.shape != (nt,) or entropy_.shape != (nt,):
            raise ValueError(
                "EOS pressure, energy density, and entropy density must match temperature."
            )
        species_ids = tuple(str(value).strip() for value in species_plan_ids)
        channel_ids = tuple(str(value).strip() for value in rate_channel_ids)
        if (
            len(species_ids) != species_count
            or any(not value for value in species_ids)
            or len(set(species_ids)) != len(species_ids)
        ):
            raise ValueError(
                "species_plan_ids must uniquely identify every thermal species row."
            )
        if (
            len(channel_ids) != rates_.shape[0]
            or any(not value for value in channel_ids)
            or len(set(channel_ids)) != len(channel_ids)
        ):
            raise ValueError(
                "rate_channel_ids must uniquely identify every thermal rate row."
            )
        if covariance.shape != (3 * nt, 3 * nt) or np.iscomplexobj(covariance):
            raise ValueError(
                "eos_covariance must be a real covariance over concatenated "
                "pressure/energy/entropy tables."
            )
        thermodynamic_tol = float(thermodynamic_tolerance)
        covariance_tol = float(covariance_tolerance)
        if not np.isfinite(thermodynamic_tol) or thermodynamic_tol <= 0.0:
            raise ValueError("thermodynamic_tolerance must be finite and positive.")
        if not np.isfinite(covariance_tol) or covariance_tol < 0.0:
            raise ValueError("covariance_tolerance must be finite and nonnegative.")
        axes_monotone = bool(
            np.all(np.diff(temperature_) > 0.0)
            and np.all(temperature_ > 0.0)
            and np.all(np.diff(momentum_) > 0.0)
            and np.all(momentum_ >= 0.0)
            and np.all(np.diff(frequency_) > 0.0)
        )
        finite = bool(
            all(
                np.all(np.isfinite(value.real)) & np.all(np.isfinite(value.imag))
                if np.iscomplexobj(value)
                else np.all(np.isfinite(value))
                for value in arrays
            )
        )
        nonnegative_widths = bool(np.all(widths_ >= 0.0) and np.all(masses >= 0.0))
        nonnegative_spectral = bool(np.all(spectral >= 0.0))
        nonnegative_rates = bool(np.all(rates_ >= 0.0))
        numerical_floor = np.finfo(np.float64).tiny
        covariance_scale = max(float(np.max(np.abs(covariance))), numerical_floor)
        covariance_symmetry_residual = float(
            np.max(np.abs(covariance - covariance.T)) / covariance_scale
        )
        covariance_symmetric = covariance_symmetry_residual <= covariance_tol
        covariance_eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
        minimum_covariance_eigenvalue = float(np.min(covariance_eigenvalues))
        covariance_eigenvalue_scale = max(
            float(np.max(np.abs(covariance_eigenvalues))), numerical_floor
        )
        covariance_psd = (
            minimum_covariance_eigenvalue >= -covariance_tol * covariance_eigenvalue_scale
        )
        entropy_from_pressure = np.gradient(pressure_, temperature_, edge_order=2)
        energy_from_identity = temperature_ * entropy_ - pressure_
        entropy_scale = np.maximum(
            np.maximum(np.abs(entropy_), np.abs(entropy_from_pressure)),
            numerical_floor,
        )
        energy_scale = np.maximum(
            np.maximum(np.abs(energy_), np.abs(energy_from_identity)),
            numerical_floor,
        )
        entropy_residual = float(
            np.max(np.abs(entropy_ - entropy_from_pressure) / entropy_scale)
        )
        energy_residual = float(
            np.max(np.abs(energy_ - energy_from_identity) / energy_scale)
        )
        heat_capacity = np.gradient(energy_, temperature_, edge_order=2)
        minimum_heat_capacity = float(np.min(heat_capacity))
        heat_capacity_scale = max(float(np.max(np.abs(heat_capacity))), numerical_floor)
        entropy_magnitude = max(float(np.max(np.abs(entropy_))), numerical_floor)
        consistent = (
            entropy_residual <= thermodynamic_tol and energy_residual <= thermodynamic_tol
        )
        stable = (
            minimum_heat_capacity >= -thermodynamic_tol * heat_capacity_scale
            and float(np.min(entropy_)) >= -thermodynamic_tol * entropy_magnitude
        )
        qualified = bool(
            finite
            and axes_monotone
            and nonnegative_widths
            and nonnegative_spectral
            and nonnegative_rates
            and covariance_symmetric
            and covariance_psd
            and consistent
            and stable
        )
        payload = thermal_kernel_payload_bytes(
            *arrays,
            species_plan_ids=species_ids,
            rate_channel_ids=channel_ids,
        )
        if source_kind_ == "external-table":
            if not isinstance(source_manifest, ReferenceArtifactManifest):
                raise TypeError(
                    "External thermal tables require ReferenceArtifactManifest."
                )
            if not isinstance(source_artifact, ScientificArtifactEnvelope):
                raise TypeError(
                    "External thermal tables require ScientificArtifactEnvelope."
                )
            source_manifest.require_rights(**dict(requested))
            source_manifest.verify_bytes(payload)
            if (
                source_artifact.status != "complete"
                or source_artifact.artifact_kind != "thermal-kernel-tables"
                or source_artifact.content_digest != source_manifest.checksum
                or source_artifact.license_id != source_manifest.license_id
                or source_manifest.manifest_id not in source_artifact.parent_artifact_ids
            ):
                raise ValueError(
                    "Thermal table envelope disagrees with manifest digest, license, or lineage."
                )
            differentiation = "external-table-stop-gradient"
            source_identity = source_artifact.artifact_id
        else:
            if source_manifest is not None or source_artifact is not None:
                raise ValueError(
                    "Native analytic thermal artifacts cannot claim an external table source."
                )
            if any(rights_values):
                raise ValueError(
                    "Native analytic thermal artifacts do not accept external requested-use flags."
                )
            differentiation = "native-generated-table-stop-gradient"
            source_identity = "native-analytic"
        requested_use_id = canonical_fingerprint(
            {
                "kind": "thermal-kernel-requested-use",
                "source": None
                if source_manifest is None
                else source_manifest.manifest_id,
                "rights": dict(requested),
            }
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "thermal-kernel-evidence",
                "finite": finite,
                "axes_monotone": axes_monotone,
                "nonnegative_widths": nonnegative_widths,
                "nonnegative_spectral": nonnegative_spectral,
                "nonnegative_rates": nonnegative_rates,
                "covariance_symmetric": covariance_symmetric,
                "covariance_psd": covariance_psd,
                "entropy_residual": entropy_residual,
                "energy_residual": energy_residual,
                "minimum_heat_capacity": minimum_heat_capacity,
            }
        )
        evidence = ThermalKernelEvidence(
            finite,
            axes_monotone,
            nonnegative_widths,
            nonnegative_spectral,
            nonnegative_rates,
            covariance_symmetric,
            covariance_psd,
            consistent,
            stable,
            entropy_residual,
            energy_residual,
            minimum_heat_capacity,
            qualified,
            evidence_id,
        )
        if not qualified:
            raise ValueError(
                "Thermal kernel artifact fails finite, spectral, covariance, or EOS qualification."
            )
        frame_realization_id = frame.realization_id()
        converted = tuple(jax.lax.stop_gradient(jnp.asarray(value)) for value in arrays)
        (
            self.temperature,
            self.momentum,
            self.frequency,
            self.thermal_masses,
            self.widths,
            self.self_energies,
            self.spectral_functions,
            self.screening_longitudinal,
            self.screening_transverse,
            self.rates,
            self.pressure,
            self.energy_density,
            self.entropy_density,
            self.eos_covariance,
        ) = converted
        self.units = units
        self.frame = frame
        self.frame_token = jax.lax.stop_gradient(jnp.asarray(frame.frame_token))
        self.source_manifest = source_manifest
        self.source_artifact = source_artifact
        self.evidence = evidence
        self.species_count = species_count
        self.rate_count = int(rates_.shape[0])
        self.species_plan_ids = species_ids
        self.rate_channel_ids = channel_ids
        self.source_kind = source_kind_
        self.requested_use = requested
        self.requested_use_id = requested_use_id
        self.differentiation = differentiation
        self.unit_contract_id = units.contract_id
        self.frame_id = frame.frame_id
        self.frame_realization_id = frame_realization_id
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "thermal-dark-kernel-artifact",
                "source": source_identity,
                "requested_use": requested_use_id,
                "unit_contract": units.contract_id,
                "frame": frame.frame_id,
                "frame_realization": frame_realization_id,
                "arrays": array_tree_fingerprint(arrays),
                "evidence": evidence_id,
                "differentiation": differentiation,
                "species_plan_ids": list(species_ids),
                "rate_channel_ids": list(channel_ids),
            }
        )


class HTLPolarizationEvidence(StrictModule):
    """Support and Ward-identity evidence for one HTL evaluation."""

    landau_damping_support: Array
    ward_residual: Array
    transverse: Array
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class HTLPolarizationResult(StrictModule):
    """Retarded longitudinal/transverse HTL response and four-tensor."""

    longitudinal: Array
    transverse: Array
    polarization_tensor: Array
    evidence: HTLPolarizationEvidence
    plan_id: str = eqx.field(static=True)


class HTLPolarizationPlan(StrictModule, NonTrainableState):
    """Isotropic retarded hard-thermal-loop polarization in a local frame."""

    debye_mass_squared: float = eqx.field(static=True)
    ward_tolerance: float = eqx.field(static=True)
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        debye_mass_squared: float,
        units: RelativisticUnitContract,
        frame: LocalRelativisticFramePlan,
        /,
        *,
        ward_tolerance: float = 1.0e-10,
    ):
        mass_squared = float(debye_mass_squared)
        tolerance = float(ward_tolerance)
        if not np.isfinite(mass_squared) or mass_squared < 0.0:
            raise ValueError("debye_mass_squared must be finite and nonnegative.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("ward_tolerance must be finite and positive.")
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be RelativisticUnitContract.")
        if (
            not isinstance(frame, LocalRelativisticFramePlan)
            or frame.units.contract_id != units.contract_id
        ):
            raise ValueError("HTL frame and unit contract disagree.")
        if frame.geometry.leading_shape != ():
            raise ValueError("HTL evaluation requires one scalar local frame.")
        self.debye_mass_squared = mass_squared
        self.ward_tolerance = tolerance
        self.units = units
        self.frame = frame
        self.plan_id = canonical_fingerprint(
            {
                "kind": "isotropic-retarded-htl-polarization",
                "debye_mass_squared": mass_squared,
                "unit_contract": units.contract_id,
                "frame": frame.frame_id,
                "frame_realization": frame.realization_id(),
                "ward_tolerance": tolerance,
            }
        )

    def evaluate(
        self, angular_frequency: ArrayLike, wave_number: ArrayLike, /
    ) -> HTLPolarizationResult:
        """Evaluate retarded HTL response with explicit ``hbar`` and ``c`` bridges."""

        omega, k = jnp.broadcast_arrays(
            jnp.asarray(angular_frequency), jnp.asarray(wave_number)
        )
        if not jnp.issubdtype(omega.dtype, jnp.floating):
            raise TypeError("HTL frequency and wave number must be real floating arrays.")
        frequency_energy = self.units.angular_frequency_to_energy(omega)
        wave_energy = self.units.wave_number_to_energy(k)
        tiny = jnp.finfo(omega.dtype).tiny
        nonzero_k = wave_energy > 0.0
        safe_wave_energy = jnp.where(nonzero_k, wave_energy, 1.0)
        x = frequency_energy / safe_wave_energy
        numerator = jnp.abs(frequency_energy + safe_wave_energy)
        denominator = jnp.maximum(jnp.abs(frequency_energy - safe_wave_energy), tiny)
        log_real = jnp.log(jnp.maximum(numerator, tiny) / denominator)
        landau = nonzero_k & (jnp.abs(frequency_energy) < wave_energy)
        complex_dtype = jnp.complex64 if omega.dtype == jnp.float32 else jnp.complex128
        retarded_log = log_real.astype(complex_dtype) - 1j * jnp.pi * landau
        logarithmic = 0.5 * x * retarded_log
        debye_mass = jnp.sqrt(jnp.asarray(self.debye_mass_squared, dtype=omega.dtype))
        debye_energy = self.units.mass_to_rest_energy(debye_mass)
        debye_energy_squared = debye_energy**2
        longitudinal = debye_energy_squared * (1.0 - logarithmic)
        transverse = 0.5 * debye_energy_squared * (x**2 + (1.0 - x**2) * logarithmic)
        longitudinal = jnp.where(nonzero_k, longitudinal, 0.0 + 0.0j)
        transverse = jnp.where(nonzero_k, transverse, debye_energy_squared / 3.0 + 0.0j)
        shape = omega.shape
        tensor = jnp.zeros((*shape, 4, 4), dtype=longitudinal.dtype)
        ratio = jnp.where(nonzero_k, frequency_energy / safe_wave_energy, 0.0)
        tensor = tensor.at[..., 0, 0].set(longitudinal)
        tensor = tensor.at[..., 0, 3].set(ratio * longitudinal)
        tensor = tensor.at[..., 3, 0].set(ratio * longitudinal)
        tensor = tensor.at[..., 3, 3].set(ratio**2 * longitudinal)
        tensor = tensor.at[..., 1, 1].set(transverse)
        tensor = tensor.at[..., 2, 2].set(transverse)
        four_covector = jnp.stack(
            (
                frequency_energy,
                jnp.zeros_like(frequency_energy),
                jnp.zeros_like(frequency_energy),
                -wave_energy,
            ),
            axis=-1,
        )
        ward = ein.contract("...m,...mn->...n", four_covector, tensor)
        ward_residual = jnp.max(jnp.abs(ward), axis=-1)
        tensor_scale = jnp.maximum(jnp.max(jnp.abs(tensor), axis=(-2, -1)), tiny)
        momentum_scale = jnp.maximum(
            jnp.maximum(jnp.abs(frequency_energy), jnp.abs(wave_energy)), tiny
        )
        transverse_valid = (
            ward_residual <= self.ward_tolerance * momentum_scale * tensor_scale
        )
        finite = (
            jnp.isfinite(longitudinal.real)
            & jnp.isfinite(longitudinal.imag)
            & jnp.isfinite(transverse.real)
            & jnp.isfinite(transverse.imag)
        )
        valid = finite & transverse_valid & (k >= 0.0)
        evidence = HTLPolarizationEvidence(
            landau,
            ward_residual,
            transverse_valid,
            finite,
            valid,
            self.plan_id,
        )
        return HTLPolarizationResult(
            longitudinal, transverse, tensor, evidence, self.plan_id
        )


class LPMSolveEvidence(StrictModule):
    """Linear residual and overlap-subtraction evidence for an LPM solve."""

    residual_norm: Array
    relative_residual: Array
    linear_successful: Array
    overlap_subtracted: Array
    finite: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class LPMSolveResult(StrictModule):
    """Fixed-basis LPM amplitude and non-double-counted rate."""

    amplitude: Array
    raw_rate: Array
    overlap_rate: Array
    rate: Array
    evidence: LPMSolveEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


class LPMIntegralPlan(StrictModule, NonTrainableState):
    """Fixed-basis LPM equation with explicit physical normalization.

    Basis nodes use the contract momentum unit. Collision and formation-energy
    entries use its energy unit. ``rate_prefactor`` maps the solved quadrature
    bilinear to the derived inverse-scale-time rate unit before overlap removal.
    """

    basis_nodes: Array
    quadrature_weights: Array
    collision_matrix: Array
    formation_energies: Array
    source: Array
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    basis_size: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    rate_prefactor: float = eqx.field(static=True)
    rate_unit_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis_nodes: ArrayLike,
        quadrature_weights: ArrayLike,
        collision_matrix: ArrayLike,
        formation_energies: ArrayLike,
        source: ArrayLike,
        units: RelativisticUnitContract,
        frame: LocalRelativisticFramePlan,
        /,
        *,
        rate_prefactor: float,
        residual_tolerance: float = 1.0e-9,
    ):
        nodes = np.asarray(basis_nodes, dtype=float)
        weights = np.asarray(quadrature_weights, dtype=float)
        collision = np.asarray(collision_matrix)
        formation = np.asarray(formation_energies, dtype=float)
        source_ = np.asarray(source)
        if nodes.ndim != 1 or nodes.size == 0 or np.any(~np.isfinite(nodes)):
            raise ValueError("basis_nodes must be a nonempty finite vector.")
        size = int(nodes.size)
        if (
            weights.shape != (size,)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("quadrature_weights must be finite positive basis weights.")
        if collision.shape != (size, size) or np.any(~np.isfinite(collision)):
            raise ValueError("collision_matrix must be a finite square basis matrix.")
        if formation.shape != (size,) or np.any(~np.isfinite(formation)):
            raise ValueError("formation_energies must be a finite basis vector.")
        if source_.shape != (size,) or np.any(~np.isfinite(source_)):
            raise ValueError("source must be a finite basis vector.")
        tolerance = float(residual_tolerance)
        prefactor = float(rate_prefactor)
        if not np.isfinite(prefactor) or prefactor <= 0.0:
            raise ValueError("rate_prefactor must be finite and positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive.")
        symmetric = 0.5 * (collision + collision.conj().T)
        collision_eigenvalues = np.linalg.eigvalsh(symmetric).real
        collision_scale = max(
            float(np.max(np.abs(collision_eigenvalues))), np.finfo(float).tiny
        )
        numerical_tolerance = 64.0 * np.finfo(float).eps
        if float(np.min(collision_eigenvalues)) < -numerical_tolerance * collision_scale:
            raise ValueError(
                "LPM collision matrix Hermitian part must be positive semidefinite."
            )
        combined = collision.astype(complex) + 1j * np.diag(formation)
        singular_values = np.linalg.svd(combined, compute_uv=False)
        if float(np.min(singular_values)) <= numerical_tolerance * max(
            float(np.max(singular_values)), np.finfo(float).tiny
        ):
            raise ValueError(
                "The collision-plus-formation LPM operator must be nonsingular."
            )
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be RelativisticUnitContract.")
        if (
            not isinstance(frame, LocalRelativisticFramePlan)
            or frame.units.contract_id != units.contract_id
        ):
            raise ValueError("LPM frame and unit contract disagree.")
        if frame.geometry.leading_shape != ():
            raise ValueError("LPM evaluation requires one scalar local frame.")
        time_unit = units.scale.dimensional_scale.time_unit
        rate_unit = derived_unit(f"1/{time_unit.symbol}", ((time_unit, -1),))
        self.basis_nodes = jax.lax.stop_gradient(jnp.asarray(nodes))
        self.quadrature_weights = jax.lax.stop_gradient(jnp.asarray(weights))
        self.collision_matrix = jnp.asarray(collision)
        self.formation_energies = jnp.asarray(formation)
        self.source = jnp.asarray(source_)
        self.units = units
        self.frame = frame
        self.basis_size = size
        self.residual_tolerance = tolerance
        self.rate_prefactor = prefactor
        self.rate_unit_id = rate_unit.unit_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-basis-lpm-integral",
                "arrays": array_tree_fingerprint(
                    (nodes, weights, collision, formation, source_)
                ),
                "unit_contract": units.contract_id,
                "frame": frame.frame_id,
                "frame_realization": frame.realization_id(),
                "residual_tolerance": tolerance,
                "rate_prefactor": prefactor,
                "rate_unit": rate_unit.unit_id,
            }
        )

    @property
    def solve_resources(self) -> dict[str, int]:
        return {
            "basis_size": self.basis_size,
            "matrix_elements": self.basis_size * self.basis_size,
            "right_hand_sides": 1,
        }

    def solve(
        self,
        driving_scale: ArrayLike = 1.0,
        overlap_rate: ArrayLike = 0.0,
        /,
    ) -> LPMSolveResult:
        """Solve and subtract ``overlap_rate`` in the declared rate unit."""

        scale = jnp.asarray(driving_scale)
        overlap = jnp.asarray(overlap_rate, dtype=scale.dtype)
        if scale.shape != () or overlap.shape != ():
            raise ValueError("driving_scale and overlap_rate must be scalars.")
        dtype = jnp.result_type(
            self.collision_matrix, 1j * self.formation_energies, self.source, scale
        )
        matrix = self.collision_matrix.astype(dtype) + 1j * jnp.diag(
            self.formation_energies.astype(dtype)
        )
        rhs = scale.astype(dtype) * self.source.astype(dtype)
        operator = la.DenseLinearOperator(matrix, operator_id=f"{self.plan_id}:operator")
        solved = la.solve(la.LinearSystem(operator), rhs)
        amplitude = solved.value
        residual = operator.mv(amplitude) - rhs
        residual_norm = jnp.sqrt(jnp.real(jnp.vdot(residual, residual)))
        rhs_norm = jnp.sqrt(jnp.real(jnp.vdot(rhs, rhs)))
        relative = jnp.where(rhs_norm > 0.0, residual_norm / rhs_norm, residual_norm)
        raw_rate = (
            2.0
            * self.rate_prefactor
            * jnp.real(
                jnp.vdot(
                    self.source.astype(dtype),
                    self.quadrature_weights.astype(dtype) * amplitude,
                )
            )
            * scale
        )
        rate = raw_rate - overlap
        linear_successful = jnp.all(solved.successful) & (
            relative <= self.residual_tolerance
        )
        finite = (
            jnp.isfinite(raw_rate)
            & jnp.isfinite(overlap)
            & jnp.isfinite(rate)
            & jnp.all(jnp.isfinite(amplitude.real) & jnp.isfinite(amplitude.imag))
        )
        valid = linear_successful & finite & (overlap >= 0.0) & (rate >= 0.0)
        status = jnp.where(
            valid,
            int(ThermalKernelStatus.SUCCESS),
            jnp.where(
                ~linear_successful,
                int(ThermalKernelStatus.LINEAR_SOLVE_FAILURE),
                jnp.where(
                    rate < 0.0,
                    int(ThermalKernelStatus.NEGATIVE_SUBTRACTED_RATE),
                    int(ThermalKernelStatus.NONFINITE),
                ),
            ),
        ).astype(jnp.int32)
        evidence = LPMSolveEvidence(
            residual_norm,
            relative,
            linear_successful,
            overlap > 0.0,
            finite,
            valid,
            status,
            self.plan_id,
        )
        return LPMSolveResult(
            amplitude, raw_rate, overlap, rate, evidence, valid, self.plan_id
        )


class ThermalRateEvaluation(StrictModule):
    """Interpolated rates and thermodynamic state on admitted support."""

    rates: Array
    pressure: Array
    energy_density: Array
    entropy_density: Array
    heat_capacity: Array
    valid: Array
    status: Array
    artifact_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ThermalDarkRatePlan(StrictModule, NonTrainableState):
    """Bounded linear evaluation of one qualified thermal artifact."""

    artifact: ThermalKernelArtifact
    heat_capacity: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, artifact: ThermalKernelArtifact, /):
        if not isinstance(artifact, ThermalKernelArtifact):
            raise TypeError("artifact must be ThermalKernelArtifact.")
        if not artifact.evidence.qualified:
            raise ValueError("Thermal rate evaluation requires a qualified artifact.")
        heat_capacity = np.gradient(
            np.asarray(artifact.energy_density),
            np.asarray(artifact.temperature),
            edge_order=2,
        )
        self.artifact = artifact
        self.heat_capacity = jax.lax.stop_gradient(jnp.asarray(heat_capacity))
        self.plan_id = canonical_fingerprint(
            {"kind": "bounded-thermal-dark-rates", "artifact": artifact.artifact_id}
        )

    def evaluate(self, temperature: ArrayLike, /) -> ThermalRateEvaluation:
        """Interpolate without clipping; out-of-domain queries are explicitly invalid."""

        query = jnp.asarray(temperature, dtype=self.artifact.temperature.dtype)
        if query.ndim != 0:
            raise ValueError("Thermal rate temperature query must be scalar.")
        axis = self.artifact.temperature
        in_domain = jnp.isfinite(query) & (query >= axis[0]) & (query <= axis[-1])
        insertion = jnp.searchsorted(axis, query, side="right")
        left = jnp.clip(insertion - 1, 0, axis.size - 2)
        right = left + 1
        fraction = (query - axis[left]) / (axis[right] - axis[left])

        def interpolate(values):
            return values[..., left] + fraction * (values[..., right] - values[..., left])

        rates = interpolate(self.artifact.rates)
        pressure = interpolate(self.artifact.pressure)
        energy = interpolate(self.artifact.energy_density)
        entropy = interpolate(self.artifact.entropy_density)
        heat_capacity = interpolate(self.heat_capacity)
        finite = (
            jnp.all(jnp.isfinite(rates))
            & jnp.isfinite(pressure)
            & jnp.isfinite(energy)
            & jnp.isfinite(entropy)
            & jnp.isfinite(heat_capacity)
        )
        stable = (heat_capacity >= 0.0) & (entropy >= 0.0)
        valid = in_domain & finite & stable
        status = jnp.where(
            valid,
            int(ThermalKernelStatus.SUCCESS),
            jnp.where(
                ~in_domain,
                int(ThermalKernelStatus.OUTSIDE_TEMPERATURE_SUPPORT),
                jnp.where(
                    ~finite,
                    int(ThermalKernelStatus.NONFINITE),
                    int(ThermalKernelStatus.EOS_INCONSISTENT),
                ),
            ),
        ).astype(jnp.int32)
        nan = jnp.asarray(jnp.nan, dtype=query.dtype)
        return ThermalRateEvaluation(
            jnp.where(valid, rates, nan),
            jnp.where(valid, pressure, nan),
            jnp.where(valid, energy, nan),
            jnp.where(valid, entropy, nan),
            jnp.where(valid, heat_capacity, nan),
            valid,
            status,
            self.artifact.artifact_id,
            self.plan_id,
        )


__all__ = [
    "HTLPolarizationEvidence",
    "HTLPolarizationPlan",
    "HTLPolarizationResult",
    "LPMIntegralPlan",
    "LPMSolveEvidence",
    "LPMSolveResult",
    "ThermalDarkRatePlan",
    "ThermalKernelArtifact",
    "ThermalKernelEvidence",
    "ThermalKernelStatus",
    "ThermalRateEvaluation",
    "thermal_kernel_payload_bytes",
]
