#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""q=0 optical conductivity and dielectric projection from periodic Kubo."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...units import (
    ANGLE,
    CONDUCTANCE,
    LENGTH,
    SI_REFERENCE_SYSTEM_ID,
    TIME,
    UnitDefinition,
)
from ..periodic._kubo import FiniteFrequencyKuboResponse
from ._response import (
    SpectralResponseEvidence,
    SpectralResponseProduct,
    SpectralResponseRepresentation,
)


_VACUUM_PERMITTIVITY_SI = 8.8541878128e-12


class OpticalDielectricEvidence(StrictModule, NonTrainableState):
    passivity_residual: Array
    f_sum_relative_residual: Array
    successful: Array


class OpticalDielectricResult(StrictModule, NonTrainableState):
    angular_frequencies_rad_per_s: Array
    conductivity_tensor_siemens_per_m: Array
    dielectric_tensor: Array
    dissipative_response: SpectralResponseProduct
    drude_weight: Array
    evidence: OpticalDielectricEvidence
    kubo_plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        angular_frequencies_rad_per_s: ArrayLike,
        conductivity_tensor_siemens_per_m: ArrayLike,
        dielectric_tensor: ArrayLike,
        dissipative_response: SpectralResponseProduct,
        drude_weight: ArrayLike,
        evidence: OpticalDielectricEvidence,
        kubo_plan_id: str,
        /,
    ):
        self.angular_frequencies_rad_per_s = jnp.asarray(angular_frequencies_rad_per_s)
        self.conductivity_tensor_siemens_per_m = jnp.asarray(
            conductivity_tensor_siemens_per_m
        )
        self.dielectric_tensor = jnp.asarray(dielectric_tensor)
        self.dissipative_response = dissipative_response
        self.drude_weight = jnp.asarray(drude_weight)
        self.evidence = evidence
        self.kubo_plan_id = str(kubo_plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "q-zero-optical-dielectric-result",
                "kubo": self.kubo_plan_id,
                "raw": dissipative_response.product_id,
                "successful": bool(evidence.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "epsilon": np.asarray(self.dielectric_tensor),
                        "drude": np.asarray(self.drude_weight),
                    }
                ),
            }
        )


class OpticalDielectricPlan(StrictModule, NonTrainableState):
    polarizations: Array
    channel_labels: tuple[str, ...] = eqx.field(static=True)
    passivity_tolerance: float = eqx.field(static=True)
    f_sum_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        polarizations: ArrayLike,
        channel_labels: tuple[str, ...],
        /,
        *,
        passivity_tolerance: float = 1.0e-10,
        f_sum_tolerance: float = 1.0e-8,
    ):
        vectors = jnp.asarray(polarizations)
        labels = tuple(str(label).strip() for label in channel_labels)
        passivity = float(passivity_tolerance)
        f_sum = float(f_sum_tolerance)
        if (
            vectors.shape != (len(labels), 3)
            or not labels
            or len(set(labels)) != len(labels)
            or any(not label for label in labels)
            or bool(jnp.any(~jnp.isfinite(vectors)))
            or bool(jnp.any(jnp.abs(jnp.linalg.norm(vectors, axis=1) - 1.0) > 1.0e-10))
            or any(not isfinite(value) or value <= 0.0 for value in (passivity, f_sum))
        ):
            raise ValueError("Optical polarization channels or tolerances are invalid.")
        self.polarizations = vectors
        self.channel_labels = labels
        self.passivity_tolerance = passivity
        self.f_sum_tolerance = f_sum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "q-zero-optical-dielectric-plan",
                "approximation": "independent-particle-interband-kubo",
                "polarizations": array_tree_fingerprint(np.asarray(vectors)),
                "labels": list(labels),
                "passivity_tolerance": passivity,
                "f_sum_tolerance": f_sum,
            }
        )

    def evaluate(
        self,
        kubo: FiniteFrequencyKuboResponse,
        angular_frequency_unit: UnitDefinition,
        conductivity_unit: UnitDefinition,
        /,
    ) -> OpticalDielectricResult:
        if not isinstance(kubo, FiniteFrequencyKuboResponse):
            raise TypeError("Optical response requires a FiniteFrequencyKuboResponse.")
        if not bool(kubo.successful):
            raise ValueError(
                "An unsuccessful periodic Kubo response cannot enter optics."
            )
        if kubo.evidence.includes_drude or kubo.evidence.infers_relaxation_time:
            raise ValueError(
                "The interband optical profile excludes broadened Drude weight and inferred relaxation time."
            )
        if (
            angular_frequency_unit.dimension != ANGLE / TIME
            or conductivity_unit.dimension != CONDUCTANCE / LENGTH
            or angular_frequency_unit.reference_system_id != SI_REFERENCE_SYSTEM_ID
            or conductivity_unit.reference_system_id != SI_REFERENCE_SYSTEM_ID
        ):
            raise ValueError(
                "Optical Kubo arrays require SI angular-frequency and conductivity units."
            )
        omega = kubo.angular_frequencies_rad_per_s
        conductivity = kubo.regular_conductivity_siemens_per_m
        identity = jnp.eye(3, dtype=conductivity.dtype)
        dielectric = identity[None, :, :] + 1j * conductivity / (
            _VACUUM_PERMITTIVITY_SI * omega[:, None, None]
        )
        projected = jnp.real(
            contract(
                "ca,wab,cb->cw",
                jnp.conj(self.polarizations),
                conductivity,
                self.polarizations,
            )
        )
        passivity_residual = jnp.maximum(-jnp.min(projected), 0.0)
        f_sum_residual = kubo.evidence.f_sum_relative_residual
        successful = bool(passivity_residual <= self.passivity_tolerance) and bool(
            f_sum_residual <= self.f_sum_tolerance
        )
        evidence = OpticalDielectricEvidence(
            passivity_residual,
            f_sum_residual,
            jnp.asarray(successful),
        )
        raw_evidence = SpectralResponseEvidence(
            passivity_residual, f_sum_residual, 0.0, 0.0, successful
        )
        raw = SpectralResponseProduct(
            omega,
            projected,
            jnp.ones(omega.shape, dtype=jnp.bool_),
            angular_frequency_unit,
            conductivity_unit,
            self.channel_labels,
            SpectralResponseRepresentation.DENSITY,
            "q-zero-independent-particle-interband-optical",
            kubo.plan_id,
            raw_evidence,
        )
        return OpticalDielectricResult(
            omega,
            conductivity,
            dielectric,
            raw,
            kubo.raw.drude_weight,
            evidence,
            kubo.plan_id,
        )


def optical_dielectric_from_conductivity(
    plan: OpticalDielectricPlan,
    kubo: FiniteFrequencyKuboResponse,
    angular_frequency_unit: UnitDefinition,
    conductivity_unit: UnitDefinition,
    /,
) -> OpticalDielectricResult:
    return plan.evaluate(kubo, angular_frequency_unit, conductivity_unit)


__all__ = [
    "OpticalDielectricEvidence",
    "OpticalDielectricPlan",
    "OpticalDielectricResult",
    "optical_dielectric_from_conductivity",
]
