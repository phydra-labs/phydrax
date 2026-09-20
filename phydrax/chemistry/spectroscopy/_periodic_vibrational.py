#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-backed Γ-point harmonic IR and nonresonant Raman line profiles."""

from __future__ import annotations

from math import isfinite
from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...units import UnitDefinition
from ._response import (
    SpectralResponseConvention,
    SpectralResponseEvidence,
    SpectralResponseProduct,
    SpectralResponseRepresentation,
)


class _PhononDispersion(Protocol):
    fractional_qpoints: Array
    angular_frequencies: Array
    eigenvectors: Array
    acoustic_mask: Array
    imaginary_mask: Array
    result_id: str
    successful: Array


class PeriodicSpectroscopyRequest(StrictModule, NonTrainableState):
    structure_id: str = eqx.field(static=True)
    phonon_result_id: str = eqx.field(static=True)
    atom_order: tuple[str, ...] = eqx.field(static=True)
    request_born_charges: bool = eqx.field(static=True)
    request_raman_tensors: bool = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        structure_id: str,
        phonon_result_id: str,
        atom_order: tuple[str, ...],
        /,
        *,
        request_born_charges: bool = True,
        request_raman_tensors: bool = True,
    ):
        structure = str(structure_id).strip()
        phonon = str(phonon_result_id).strip()
        atoms = tuple(str(atom).strip() for atom in atom_order)
        if (
            not structure
            or not phonon
            or not atoms
            or len(set(atoms)) != len(atoms)
            or any(not atom for atom in atoms)
            or not (request_born_charges or request_raman_tensors)
        ):
            raise ValueError("Periodic spectroscopy request metadata are invalid.")
        self.structure_id = structure
        self.phonon_result_id = phonon
        self.atom_order = atoms
        self.request_born_charges = bool(request_born_charges)
        self.request_raman_tensors = bool(request_raman_tensors)
        self.request_id = canonical_fingerprint(
            {
                "kind": "periodic-spectroscopy-request",
                "structure": structure,
                "phonon": phonon,
                "atom_order": list(atoms),
                "born": request_born_charges,
                "raman": request_raman_tensors,
            }
        )


class PeriodicSpectroscopyEvidence(StrictModule, NonTrainableState):
    born_neutrality_residual: Array
    raman_symmetry_residual: Array
    converged: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        born_neutrality_residual: ArrayLike,
        raman_symmetry_residual: ArrayLike,
        converged: ArrayLike,
        /,
    ):
        neutrality = jnp.asarray(born_neutrality_residual, dtype=jnp.float64).reshape(())
        symmetry = jnp.asarray(raman_symmetry_residual, dtype=jnp.float64).reshape(())
        if (
            bool(~jnp.isfinite(neutrality))
            or bool(~jnp.isfinite(symmetry))
            or bool(neutrality < 0.0)
            or bool(symmetry < 0.0)
        ):
            raise ValueError("Provider tensor residuals must be finite and non-negative.")
        self.born_neutrality_residual = neutrality
        self.raman_symmetry_residual = symmetry
        self.converged = jnp.asarray(converged, dtype=jnp.bool_).reshape(())
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-spectroscopy-evidence",
                "neutrality": float(neutrality),
                "raman_symmetry": float(symmetry),
                "converged": bool(self.converged),
            }
        )


class PeriodicSpectroscopyTensorResult(StrictModule, NonTrainableState):
    """Normalized provider tensors; missing requested fields are rejected."""

    born_effective_charges: Array
    raman_tensors: Array
    request: PeriodicSpectroscopyRequest
    evidence: PeriodicSpectroscopyEvidence
    provider_id: str = eqx.field(static=True)
    source_hashes: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        born_effective_charges: ArrayLike,
        raman_tensors: ArrayLike,
        request: PeriodicSpectroscopyRequest,
        evidence: PeriodicSpectroscopyEvidence,
        provider_id: str,
        source_hashes: tuple[str, ...],
        /,
    ):
        born = jnp.asarray(born_effective_charges)
        raman = jnp.asarray(raman_tensors)
        provider = str(provider_id).strip()
        hashes = tuple(str(value).strip() for value in source_hashes)
        atom_count = len(request.atom_order)
        if request.request_born_charges and born.shape != (atom_count, 3, 3):
            raise ValueError(
                "Provider Born charges are missing or have the wrong atom order."
            )
        if request.request_raman_tensors and (
            raman.ndim != 3 or raman.shape[1:] != (3, 3)
        ):
            raise ValueError("Provider Raman tensors are missing or malformed.")
        if (
            not provider
            or not hashes
            or any(not value for value in hashes)
            or bool(jnp.any(~jnp.isfinite(born)))
            or bool(jnp.any(~jnp.isfinite(raman)))
        ):
            raise ValueError("Provider tensor values or provenance are invalid.")
        self.born_effective_charges = born
        self.raman_tensors = raman
        self.request = request
        self.evidence = evidence
        self.provider_id = provider
        self.source_hashes = hashes
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-spectroscopy-tensors",
                "request": request.request_id,
                "provider": provider,
                "source_hashes": list(hashes),
                "evidence": evidence.evidence_id,
                "arrays": array_tree_fingerprint(
                    {"born": np.asarray(born), "raman": np.asarray(raman)}
                ),
            }
        )


class PeriodicVibrationalSpectroscopyPlan(StrictModule, NonTrainableState):
    masses: Array
    incident_polarizations: Array
    scattered_polarizations: Array
    polarization_labels: tuple[str, ...] = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    laser_angular_frequency: float = eqx.field(static=True)
    gamma_tolerance: float = eqx.field(static=True)
    tensor_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        masses: ArrayLike,
        incident_polarizations: ArrayLike,
        scattered_polarizations: ArrayLike,
        polarization_labels: tuple[str, ...],
        /,
        *,
        temperature: float,
        laser_angular_frequency: float,
        gamma_tolerance: float = 1.0e-10,
        tensor_tolerance: float = 1.0e-8,
    ):
        mass = jnp.asarray(masses, dtype=jnp.float64)
        incident = jnp.asarray(incident_polarizations, dtype=jnp.float64)
        scattered = jnp.asarray(scattered_polarizations, dtype=jnp.float64)
        labels = tuple(str(label).strip() for label in polarization_labels)
        thermal = float(temperature)
        laser = float(laser_angular_frequency)
        gamma = float(gamma_tolerance)
        tolerance = float(tensor_tolerance)
        if (
            mass.ndim != 1
            or mass.size == 0
            or bool(jnp.any(~jnp.isfinite(mass)))
            or bool(jnp.any(mass <= 0.0))
            or incident.shape != scattered.shape
            or incident.shape != (len(labels), 3)
            or not labels
            or len(set(labels)) != len(labels)
            or any(not label for label in labels)
            or any(not isfinite(value) for value in (thermal, laser, gamma, tolerance))
            or thermal <= 0.0
            or laser <= 0.0
            or gamma <= 0.0
            or tolerance <= 0.0
        ):
            raise ValueError("Periodic vibrational spectroscopy plan is invalid.")
        incident_norm = jnp.linalg.norm(incident, axis=1)
        scattered_norm = jnp.linalg.norm(scattered, axis=1)
        if bool(jnp.any(jnp.abs(incident_norm - 1.0) > tolerance)) or bool(
            jnp.any(jnp.abs(scattered_norm - 1.0) > tolerance)
        ):
            raise ValueError("Raman polarization vectors must be normalized.")
        self.masses = mass
        self.incident_polarizations = incident
        self.scattered_polarizations = scattered
        self.polarization_labels = labels
        self.temperature = thermal
        self.laser_angular_frequency = laser
        self.gamma_tolerance = gamma
        self.tensor_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-vibrational-spectroscopy-plan",
                "masses": array_tree_fingerprint(np.asarray(mass)),
                "incident": array_tree_fingerprint(np.asarray(incident)),
                "scattered": array_tree_fingerprint(np.asarray(scattered)),
                "labels": list(labels),
                "temperature": thermal,
                "laser": laser,
                "gamma_tolerance": gamma,
                "tensor_tolerance": tolerance,
            }
        )


class PeriodicVibrationalSpectrumResult(StrictModule, NonTrainableState):
    ir_lines: SpectralResponseProduct
    raman_lines: SpectralResponseProduct
    mass_orthonormality_residual: Array
    detailed_balance_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    provider_result_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        ir_lines: SpectralResponseProduct,
        raman_lines: SpectralResponseProduct,
        mass_orthonormality_residual: ArrayLike,
        detailed_balance_residual: ArrayLike,
        successful: ArrayLike,
        plan_id: str,
        provider_result_id: str,
        /,
    ):
        self.ir_lines = ir_lines
        self.raman_lines = raman_lines
        self.mass_orthonormality_residual = jnp.asarray(
            mass_orthonormality_residual
        ).reshape(())
        self.detailed_balance_residual = jnp.asarray(detailed_balance_residual).reshape(
            ()
        )
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.plan_id = str(plan_id)
        self.provider_result_id = str(provider_result_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "periodic-vibrational-spectrum",
                "plan": self.plan_id,
                "provider": self.provider_result_id,
                "ir": ir_lines.product_id,
                "raman": raman_lines.product_id,
                "successful": bool(self.successful),
            }
        )


def periodic_vibrational_lines(
    plan: PeriodicVibrationalSpectroscopyPlan,
    phonons: _PhononDispersion,
    tensors: PeriodicSpectroscopyTensorResult,
    coordinate_unit: UnitDefinition,
    response_unit: UnitDefinition,
    /,
) -> PeriodicVibrationalSpectrumResult:
    if not bool(phonons.successful) or not bool(tensors.evidence.converged):
        raise ValueError("Only successful phonon and provider results may be profiled.")
    if tensors.request.phonon_result_id != phonons.result_id:
        raise ValueError("Provider tensors do not belong to this phonon result.")
    qpoints = jnp.asarray(phonons.fractional_qpoints)
    gamma_rows = jnp.all(jnp.abs(qpoints) <= plan.gamma_tolerance, axis=1)
    if int(jnp.sum(gamma_rows)) != 1:
        raise ValueError(
            "The first periodic vibrational profile requires exactly one Γ row."
        )
    gamma_index = int(jnp.argmax(gamma_rows))
    frequencies = jnp.asarray(phonons.angular_frequencies[gamma_index])
    eigenvectors = jnp.asarray(phonons.eigenvectors[gamma_index])
    mode_count = 3 * plan.masses.size
    if (
        frequencies.shape != (mode_count,)
        or eigenvectors.shape != (mode_count, mode_count)
        or bool(jnp.any(~jnp.isfinite(frequencies)))
        or bool(jnp.any(~jnp.isfinite(eigenvectors)))
    ):
        raise ValueError("Γ phonon modes must match atom masses and remain finite.")
    if tensors.raman_tensors.shape != (mode_count, 3, 3):
        raise ValueError("Provider Raman tensors must follow the Γ mode order exactly.")
    active = (
        ~jnp.asarray(phonons.acoustic_mask[gamma_index], dtype=jnp.bool_)
        & ~jnp.asarray(phonons.imaginary_mask[gamma_index], dtype=jnp.bool_)
        & (frequencies > 0.0)
    )
    maximum_active_frequency = jnp.max(jnp.where(active, frequencies, 0.0))
    if plan.laser_angular_frequency <= float(maximum_active_frequency):
        raise ValueError(
            "The Raman laser frequency must exceed every active Stokes shift."
        )

    gram = jnp.conj(eigenvectors.T) @ eigenvectors
    orthogonality = jnp.max(jnp.abs(gram - jnp.eye(mode_count, dtype=gram.dtype)))
    displacements = (
        eigenvectors.reshape((plan.masses.size, 3, mode_count))
        / jnp.sqrt(plan.masses)[:, None, None]
    )
    mode_charge = contract(
        "aij,ajm->mi",
        tensors.born_effective_charges,
        displacements,
    )
    ir_values = jnp.where(
        active[None, :],
        jnp.real(mode_charge * jnp.conj(mode_charge)).T,
        0.0,
    )

    amplitudes = contract(
        "ci,mij,cj->cm",
        plan.scattered_polarizations,
        tensors.raman_tensors,
        plan.incident_polarizations,
    )
    amplitude_squared = jnp.real(amplitudes * jnp.conj(amplitudes))
    thermal_frequency = jnp.where(active, frequencies, plan.temperature)
    spectral_shift = jnp.where(active, frequencies, 0.0)
    bose = 1.0 / jnp.expm1(thermal_frequency / plan.temperature)
    stokes_factor = (plan.laser_angular_frequency - spectral_shift) ** 4 * (bose + 1.0)
    anti_factor = (plan.laser_angular_frequency + spectral_shift) ** 4 * bose
    stokes = jnp.where(active[None, :], amplitude_squared * stokes_factor[None, :], 0.0)
    anti = jnp.where(active[None, :], amplitude_squared * anti_factor[None, :], 0.0)
    raman_values = jnp.concatenate((stokes, anti), axis=0)
    raman_channels = tuple(
        f"{label}:stokes" for label in plan.polarization_labels
    ) + tuple(f"{label}:anti-stokes" for label in plan.polarization_labels)

    expected_ratio = (
        (plan.laser_angular_frequency + spectral_shift)
        / (plan.laser_angular_frequency - spectral_shift)
    ) ** 4 * jnp.exp(-spectral_shift / plan.temperature)
    observed_ratio = anti / jnp.maximum(stokes, jnp.finfo(stokes.dtype).tiny)
    active_ratio = active[None, :] & (stokes > jnp.finfo(stokes.dtype).tiny)
    balance_residual = jnp.max(
        jnp.where(active_ratio, jnp.abs(observed_ratio - expected_ratio[None, :]), 0.0)
    )
    provider_residual = jnp.maximum(
        tensors.evidence.born_neutrality_residual,
        tensors.evidence.raman_symmetry_residual,
    )
    successful = (
        orthogonality <= plan.tensor_tolerance
        and provider_residual <= plan.tensor_tolerance
        and balance_residual <= plan.tensor_tolerance
        and bool(jnp.all(jnp.isfinite(ir_values)))
        and bool(jnp.all(jnp.isfinite(raman_values)))
    )
    evidence = SpectralResponseEvidence(
        0.0,
        jnp.maximum(orthogonality, tensors.evidence.born_neutrality_residual),
        balance_residual,
        tensors.evidence.raman_symmetry_residual,
        successful,
    )
    ir = SpectralResponseProduct(
        frequencies,
        ir_values,
        active,
        coordinate_unit,
        response_unit,
        ("x", "y", "z"),
        SpectralResponseRepresentation.LINES,
        "periodic-harmonic-provider-born-ir",
        phonons.result_id,
        evidence,
        convention=SpectralResponseConvention.RETARDED_EXP_MINUS_IWT_POSITIVE_LOSS,
    )
    raman = SpectralResponseProduct(
        frequencies,
        raman_values,
        active,
        coordinate_unit,
        response_unit,
        raman_channels,
        SpectralResponseRepresentation.LINES,
        "periodic-nonresonant-provider-raman",
        phonons.result_id,
        evidence,
        convention=SpectralResponseConvention.RETARDED_EXP_MINUS_IWT_POSITIVE_LOSS,
    )
    return PeriodicVibrationalSpectrumResult(
        ir,
        raman,
        orthogonality,
        balance_residual,
        successful,
        plan.plan_id,
        tensors.result_id,
    )


def periodic_ir_lines(
    result: PeriodicVibrationalSpectrumResult, /
) -> SpectralResponseProduct:
    return result.ir_lines


def periodic_raman_lines(
    result: PeriodicVibrationalSpectrumResult, /
) -> SpectralResponseProduct:
    return result.raman_lines


__all__ = [
    "PeriodicSpectroscopyEvidence",
    "PeriodicSpectroscopyRequest",
    "PeriodicSpectroscopyTensorResult",
    "PeriodicVibrationalSpectroscopyPlan",
    "PeriodicVibrationalSpectrumResult",
    "periodic_ir_lines",
    "periodic_raman_lines",
    "periodic_vibrational_lines",
]
