#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Macroscopic longitudinal single-scattering valence EELS."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import UnitDefinition
from ._response import (
    SpectralResponseEvidence,
    SpectralResponseProduct,
    SpectralResponseRepresentation,
)


class MacroscopicDielectricResult(StrictModule, NonTrainableState):
    """Provider/native-certified scalar longitudinal dielectric response."""

    q_magnitudes: Array
    positive_energy: Array
    dielectric: Array
    fsum_targets: Array
    causal_residual: Array
    source_profile_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    source_hashes: tuple[str, ...] = eqx.field(static=True)
    successful: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        q_magnitudes: ArrayLike,
        positive_energy: ArrayLike,
        dielectric: ArrayLike,
        fsum_targets: ArrayLike,
        causal_residual: ArrayLike,
        source_profile_id: str,
        provider_id: str,
        source_hashes: tuple[str, ...],
        successful: ArrayLike,
        /,
    ):
        q = jnp.asarray(q_magnitudes, dtype=jnp.float64)
        energy = jnp.asarray(positive_energy, dtype=jnp.float64)
        epsilon = jnp.asarray(dielectric)
        targets = jnp.asarray(fsum_targets, dtype=jnp.float64)
        causal = jnp.asarray(causal_residual, dtype=jnp.float64).reshape(())
        profile = str(source_profile_id).strip()
        provider = str(provider_id).strip()
        hashes = tuple(str(value).strip() for value in source_hashes)
        if (
            q.ndim != 1
            or q.size == 0
            or bool(jnp.any(~jnp.isfinite(q)))
            or bool(jnp.any(q <= 0.0))
            or energy.ndim != 1
            or energy.size < 2
            or bool(jnp.any(~jnp.isfinite(energy)))
            or bool(jnp.any(energy <= 0.0))
            or bool(jnp.any(jnp.diff(energy) <= 0.0))
            or epsilon.shape != (q.size, energy.size)
            or targets.shape != q.shape
            or bool(jnp.any(~jnp.isfinite(epsilon)))
            or bool(jnp.any(~jnp.isfinite(targets)))
            or bool(jnp.any(targets <= 0.0))
            or bool(~jnp.isfinite(causal))
            or bool(causal < 0.0)
            or not profile
            or not provider
            or not hashes
            or any(not value for value in hashes)
        ):
            raise ValueError("Macroscopic dielectric response or provenance is invalid.")
        self.q_magnitudes = q
        self.positive_energy = energy
        self.dielectric = epsilon
        self.fsum_targets = targets
        self.causal_residual = causal
        self.source_profile_id = profile
        self.provider_id = provider
        self.source_hashes = hashes
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.result_id = canonical_fingerprint(
            {
                "kind": "macroscopic-longitudinal-dielectric-result",
                "source_profile": profile,
                "provider": provider,
                "hashes": list(hashes),
                "causal_residual": float(causal),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "q": np.asarray(q),
                        "energy": np.asarray(energy),
                        "epsilon": np.asarray(epsilon),
                        "fsum": np.asarray(targets),
                    }
                ),
            }
        )


class ElectronEnergyLossEvidence(StrictModule, NonTrainableState):
    passivity_residual: Array
    fsum_residual: Array
    causal_residual: Array
    fluctuation_dissipation_residual: Array
    successful: Array


class ElectronEnergyLossResult(StrictModule, NonTrainableState):
    loss_response: SpectralResponseProduct
    dynamic_structure_response: SpectralResponseProduct
    evidence: ElectronEnergyLossEvidence
    dielectric_result_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        loss_response: SpectralResponseProduct,
        dynamic_structure_response: SpectralResponseProduct,
        evidence: ElectronEnergyLossEvidence,
        dielectric_result_id: str,
        /,
    ):
        self.loss_response = loss_response
        self.dynamic_structure_response = dynamic_structure_response
        self.evidence = evidence
        self.dielectric_result_id = str(dielectric_result_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "macroscopic-valence-eels-result",
                "loss": loss_response.product_id,
                "structure": dynamic_structure_response.product_id,
                "dielectric": self.dielectric_result_id,
                "successful": bool(evidence.successful),
            }
        )


class ElectronEnergyLossPlan(StrictModule, NonTrainableState):
    temperature: float = eqx.field(static=True)
    q_capacity: int = eqx.field(static=True)
    energy_capacity: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        temperature: float,
        q_capacity: int,
        energy_capacity: int,
        residual_tolerance: float = 1.0e-2,
    ):
        thermal = float(temperature)
        tolerance = float(residual_tolerance)
        if (
            not isfinite(thermal)
            or thermal <= 0.0
            or int(q_capacity) <= 0
            or int(energy_capacity) < 2
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("EELS temperature, capacities, or tolerance are invalid.")
        self.temperature = thermal
        self.q_capacity = int(q_capacity)
        self.energy_capacity = int(energy_capacity)
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "macroscopic-valence-eels-plan",
                "approximation": "longitudinal-macroscopic-single-scattering",
                "temperature": thermal,
                "q_capacity": self.q_capacity,
                "energy_capacity": self.energy_capacity,
                "residual_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        dielectric: MacroscopicDielectricResult,
        energy_unit: UnitDefinition,
        response_unit: UnitDefinition,
        /,
    ) -> ElectronEnergyLossResult:
        if not bool(dielectric.successful):
            raise ValueError("An unsuccessful dielectric result cannot enter EELS.")
        if (
            dielectric.q_magnitudes.size > self.q_capacity
            or dielectric.positive_energy.size > self.energy_capacity
        ):
            raise ValueError("Dielectric response exceeds the planned EELS capacity.")
        inverse = 1.0 / dielectric.dielectric
        loss = -jnp.imag(inverse)
        passivity = jnp.maximum(-jnp.min(loss), 0.0)
        weighted_integrals = jnp.trapezoid(
            loss * dielectric.positive_energy[None, :],
            dielectric.positive_energy,
            axis=-1,
        )
        fsum_per_q = (
            jnp.abs(weighted_integrals - dielectric.fsum_targets)
            / dielectric.fsum_targets
        )
        fsum = jnp.max(fsum_per_q)
        bose_denominator = -jnp.expm1(-dielectric.positive_energy / self.temperature)
        structure = loss / bose_denominator[None, :]
        reconstructed_loss = structure * bose_denominator[None, :]
        scale = jnp.maximum(jnp.max(jnp.abs(loss)), jnp.finfo(loss.dtype).tiny)
        fdt = jnp.max(jnp.abs(reconstructed_loss - loss)) / scale
        successful = (
            bool(passivity <= self.residual_tolerance)
            and bool(fsum <= self.residual_tolerance)
            and bool(dielectric.causal_residual <= self.residual_tolerance)
            and bool(fdt <= self.residual_tolerance)
        )
        evidence = ElectronEnergyLossEvidence(
            passivity, fsum, dielectric.causal_residual, fdt, jnp.asarray(successful)
        )
        response_evidence = SpectralResponseEvidence(
            passivity, fsum, 0.0, 0.0, successful
        )
        labels = tuple(f"q[{index}]" for index in range(dielectric.q_magnitudes.size))
        active = jnp.ones(dielectric.positive_energy.shape, dtype=jnp.bool_)
        loss_response = SpectralResponseProduct(
            dielectric.positive_energy,
            loss,
            active,
            energy_unit,
            response_unit,
            labels,
            SpectralResponseRepresentation.DENSITY,
            "macroscopic-longitudinal-valence-eels",
            dielectric.result_id,
            response_evidence,
        )
        structure_response = SpectralResponseProduct(
            dielectric.positive_energy,
            structure,
            active,
            energy_unit,
            response_unit,
            labels,
            SpectralResponseRepresentation.DENSITY,
            "fluctuation-dissipation-dynamic-structure",
            dielectric.result_id,
            response_evidence,
        )
        return ElectronEnergyLossResult(
            loss_response, structure_response, evidence, dielectric.result_id
        )


__all__ = [
    "ElectronEnergyLossEvidence",
    "ElectronEnergyLossPlan",
    "ElectronEnergyLossResult",
    "MacroscopicDielectricResult",
]
