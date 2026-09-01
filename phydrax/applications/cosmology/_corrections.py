#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._closure import DifferentiationContract
from ._products import (
    combine_differentiation,
    CosmologyProductProvenance,
    MatterPowerDescriptor,
    MatterPowerTable,
)


class CorrectionModelCard(StrictModule, NonTrainableState):
    """Immutable scientific identity, domain, denominator, and evidence contract."""

    name: str = eqx.field(static=True)
    model_version: str = eqx.field(static=True)
    source_reference: str = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)
    denominator_stage: str = eqx.field(static=True)
    output_stage: str = eqx.field(static=True)
    minimum_scale_factor: float = eqx.field(static=True)
    maximum_scale_factor: float = eqx.field(static=True)
    minimum_wavenumber: float = eqx.field(static=True)
    maximum_wavenumber: float = eqx.field(static=True)
    expected_error: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    card_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        name: str,
        model_version: str,
        source_reference: str,
        calibration_id: str,
        denominator_stage: str,
        output_stage: str,
        scale_factor_domain: tuple[float, float],
        wavenumber_domain: tuple[float, float],
        expected_error: str,
        license_id: str,
    ):
        strings = tuple(
            str(value).strip()
            for value in (
                name,
                model_version,
                source_reference,
                calibration_id,
                denominator_stage,
                output_stage,
                expected_error,
                license_id,
            )
        )
        if any(not value for value in strings):
            raise ValueError("Correction model-card strings must be non-empty.")
        if denominator_stage not in ("linear", "nonlinear") or output_stage not in (
            "linear",
            "nonlinear",
        ):
            raise ValueError("Correction stages must be linear or nonlinear.")
        a_min, a_max = (float(value) for value in scale_factor_domain)
        k_min, k_max = (float(value) for value in wavenumber_domain)
        if (
            not np.isfinite(a_min)
            or not np.isfinite(a_max)
            or not np.isfinite(k_min)
            or not np.isfinite(k_max)
            or a_min <= 0.0
            or a_max <= a_min
            or k_min <= 0.0
            or k_max <= k_min
        ):
            raise ValueError("Correction model-card domains are invalid.")
        (
            self.name,
            self.model_version,
            self.source_reference,
            self.calibration_id,
            self.denominator_stage,
            self.output_stage,
            self.expected_error,
            self.license_id,
        ) = strings
        self.minimum_scale_factor = a_min
        self.maximum_scale_factor = a_max
        self.minimum_wavenumber = k_min
        self.maximum_wavenumber = k_max
        self.card_id = canonical_fingerprint(
            {
                "kind": "matter-power-correction-card",
                "name": strings[0],
                "model_version": strings[1],
                "source_reference": strings[2],
                "calibration_id": strings[3],
                "denominator_stage": strings[4],
                "output_stage": strings[5],
                "scale_factor_domain": [a_min, a_max],
                "wavenumber_domain": [k_min, k_max],
                "expected_error": strings[6],
                "license_id": strings[7],
            }
        )


class MatterPowerCorrectionEvidence(StrictModule):
    minimum_factor: Array
    maximum_factor: Array
    minimum_output: Array
    finite: Array
    within_domain: Array
    successful: Array
    card_id: str = eqx.field(static=True)


class MatterPowerCorrectionResult(StrictModule):
    power: MatterPowerTable
    evidence: MatterPowerCorrectionEvidence
    successful: Array


class MultiplicativeMatterPowerCorrectionPlan(StrictModule, NonTrainableState):
    """Apply a named multiplicative boost on exactly one native (a, k) grid."""

    scale_factors: Array
    wavenumbers: Array
    factor_values: Array
    card: CorrectionModelCard
    differentiation: DifferentiationContract
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale_factors: ArrayLike,
        wavenumbers: ArrayLike,
        factor_values: ArrayLike,
        card: CorrectionModelCard,
        /,
        *,
        differentiation: DifferentiationContract | str = "constant",
    ):
        if not isinstance(card, CorrectionModelCard):
            raise TypeError("card must be CorrectionModelCard.")
        if isinstance(differentiation, str):
            if differentiation == "native-parameter":
                differentiation_ = DifferentiationContract(
                    upstream_physical_parameters=False,
                    stored_values=True,
                    query_coordinates=True,
                    local_parameters=True,
                )
            else:
                differentiation_ = DifferentiationContract.from_label(differentiation)
        else:
            differentiation_ = differentiation
        if not isinstance(differentiation_, DifferentiationContract):
            raise TypeError("differentiation must be DifferentiationContract.")
        scales = np.asarray(scale_factors, dtype=float).reshape((-1,))
        wavenumbers_ = np.asarray(wavenumbers, dtype=float).reshape((-1,))
        factors = jnp.asarray(factor_values)
        if (
            scales.size < 2
            or wavenumbers_.size < 2
            or np.any(~np.isfinite(scales))
            or np.any(~np.isfinite(wavenumbers_))
            or np.any(np.diff(scales) <= 0.0)
            or np.any(np.diff(wavenumbers_) <= 0.0)
            or factors.shape != (scales.size, wavenumbers_.size)
        ):
            raise ValueError("Correction grid or factor shape is invalid.")
        factors = eqx.error_if(
            factors,
            jnp.any(~jnp.isfinite(factors)) | jnp.any(factors < 0.0),
            "Correction factors must be finite and non-negative.",
        )
        if not differentiation_.stored_values:
            factors = jax.lax.stop_gradient(factors)
        self.scale_factors = jnp.asarray(scales)
        self.wavenumbers = jnp.asarray(wavenumbers_)
        self.factor_values = factors
        self.card = card
        self.differentiation = differentiation_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multiplicative-matter-power-correction",
                "scale_factors": scales.tolist(),
                "wavenumbers": wavenumbers_.tolist(),
                "card": card.card_id,
                "differentiation": differentiation_.contract_id,
            }
        )

    def apply(
        self, power: MatterPowerTable, /, *, strength: ArrayLike = 1.0
    ) -> MatterPowerCorrectionResult:
        if not isinstance(power, MatterPowerTable):
            raise TypeError("power must be MatterPowerTable.")
        if power.descriptor.stage != self.card.denominator_stage:
            raise ValueError("Correction denominator stage does not match input power.")
        if (
            power.scale_factors.shape != self.scale_factors.shape
            or power.wavenumbers.shape != self.wavenumbers.shape
        ):
            raise ValueError("Correction and matter-power grids disagree.")
        scales_equal = jnp.all(power.scale_factors == self.scale_factors)
        wavenumbers_equal = jnp.all(power.wavenumbers == self.wavenumbers)
        epsilon = jnp.finfo(power.power_values.dtype).eps
        scale_tolerance = (
            16.0 * epsilon * jnp.maximum(jnp.abs(power.scale_factors[-1]), 1.0)
        )
        wavenumber_tolerance = (
            16.0 * epsilon * jnp.maximum(jnp.abs(power.wavenumbers[-1]), 1.0)
        )
        within_domain = (
            (power.scale_factors[0] >= self.card.minimum_scale_factor - scale_tolerance)
            & (
                power.scale_factors[-1]
                <= self.card.maximum_scale_factor + scale_tolerance
            )
            & (
                power.wavenumbers[0]
                >= self.card.minimum_wavenumber - wavenumber_tolerance
            )
            & (
                power.wavenumbers[-1]
                <= self.card.maximum_wavenumber + wavenumber_tolerance
            )
        )
        strength_ = jnp.asarray(strength, dtype=power.power_values.dtype)
        if strength_.shape != ():
            raise ValueError("Correction strength must be scalar.")
        effective = 1.0 + strength_ * (self.factor_values - 1.0)
        output_values = power.power_values * effective
        successful = (
            scales_equal
            & wavenumbers_equal
            & within_domain
            & jnp.isfinite(strength_)
            & jnp.all(jnp.isfinite(output_values))
            & jnp.all(output_values >= 0.0)
        )
        output_values = eqx.error_if(
            output_values,
            ~successful,
            "Matter-power correction failed grid, domain, or finite-output checks.",
        )
        differentiation = combine_differentiation(
            power.provenance.differentiation,
            self.differentiation,
        )
        provenance = CosmologyProductProvenance(
            producer="phydrax.applications.cosmology.MultiplicativeMatterPowerCorrectionPlan",
            producer_version="native",
            model_form_id=power.provenance.model_form_id,
            request_id=power.provenance.request_id,
            numerical_policy_id=self.plan_id,
            physics_policy_id=self.card.card_id,
            scale_id=power.scale.scale_id,
            source_kind="native",
            differentiation=differentiation,
            parent_product_ids=(power.provenance.provenance_id,),
        )
        descriptor = MatterPowerDescriptor(
            power.descriptor.left_field,
            power.descriptor.right_field,
            gauge=power.descriptor.gauge,
            normalization=power.descriptor.normalization,
            stage=self.card.output_stage,
            shot_noise=power.descriptor.shot_noise,
            spatial_dimension=power.descriptor.spatial_dimension,
        )
        corrected = MatterPowerTable(
            power.scale_factors,
            power.wavenumbers,
            output_values,
            descriptor,
            power.scale,
            provenance,
            power.realization,
        )
        evidence = MatterPowerCorrectionEvidence(
            minimum_factor=jnp.min(effective),
            maximum_factor=jnp.max(effective),
            minimum_output=jnp.min(output_values),
            finite=jnp.all(jnp.isfinite(output_values)),
            within_domain=within_domain,
            successful=successful,
            card_id=self.card.card_id,
        )
        return MatterPowerCorrectionResult(corrected, evidence, successful)


__all__ = [
    "CorrectionModelCard",
    "MatterPowerCorrectionEvidence",
    "MatterPowerCorrectionResult",
    "MultiplicativeMatterPowerCorrectionPlan",
]
