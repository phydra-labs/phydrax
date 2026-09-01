#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.spectral import (
    ModeTransferCorrection,
    PeriodicFourierField,
    PeriodicFourierShellPlan,
)
from ...observation import CoordinateLayout, TheoryVector
from ._closure import CosmologyRealizationSignature
from ._products import (
    CosmologyProductProvenance,
    MatterPowerDescriptor,
    MatterPowerTable,
)
from ._scales import CosmologyScaleContract


FieldDensityConvention = Literal["density-contrast", "density", "extensive-content"]


class MatterPowerEstimate(StrictModule):
    scale_factor: Array
    wavenumbers: Array
    power_values: Array
    mode_counts: Array
    valid_shells: Array
    descriptor: MatterPowerDescriptor
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance
    realization: CosmologyRealizationSignature
    source_product_ids: tuple[str, ...] = eqx.field(static=True)
    estimator_id: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    finite: Array
    successful: Array

    def __init__(
        self,
        scale_factor: ArrayLike,
        wavenumbers: ArrayLike,
        power_values: ArrayLike,
        mode_counts: ArrayLike,
        valid_shells: ArrayLike,
        descriptor: MatterPowerDescriptor,
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        realization: CosmologyRealizationSignature,
        source_product_ids: tuple[str, ...],
        estimator_id: str,
        /,
    ):
        epoch = jnp.asarray(scale_factor).reshape(())
        k = jnp.asarray(wavenumbers, dtype=epoch.dtype).reshape((-1,))
        power = jnp.asarray(power_values, dtype=epoch.dtype).reshape((-1,))
        counts = jnp.asarray(mode_counts, dtype=epoch.dtype).reshape((-1,))
        valid = jnp.asarray(valid_shells, dtype=bool).reshape((-1,))
        sources = tuple(str(value).strip() for value in source_product_ids)
        estimator = str(estimator_id).strip()
        if (
            power.shape != k.shape
            or counts.shape != k.shape
            or valid.shape != k.shape
            or not sources
            or any(not value for value in sources)
            or not estimator
            or scale.scale_id != provenance.scale_id
            or scale.scale_id != realization.scale_id
        ):
            raise ValueError("Matter-power estimate contracts or array shapes disagree.")
        finite = (
            jnp.isfinite(epoch)
            & (epoch > 0.0)
            & jnp.all(jnp.isfinite(k))
            & jnp.all(jnp.isfinite(power))
            & jnp.all(jnp.isfinite(counts))
            & jnp.all(counts >= 0.0)
        )
        physical = jnp.all(
            jnp.where(
                valid & descriptor.is_auto,
                power >= 0.0,
                True,
            )
        )
        successful = finite & physical & jnp.any(valid)
        self.scale_factor = epoch
        self.wavenumbers = k
        self.power_values = power
        self.mode_counts = counts
        self.valid_shells = valid
        self.descriptor = descriptor
        self.scale = scale
        self.provenance = provenance
        self.realization = realization
        self.source_product_ids = sources
        self.estimator_id = estimator
        self.content_id = canonical_fingerprint(
            {
                "kind": "matter-power-estimate",
                "scale_factor": array_tree_fingerprint(epoch),
                "wavenumbers": array_tree_fingerprint(k),
                "power_values": array_tree_fingerprint(power),
                "mode_counts": array_tree_fingerprint(counts),
                "valid_shells": array_tree_fingerprint(valid),
                "descriptor": descriptor.descriptor_id,
                "provenance": provenance.provenance_id,
                "realization": realization.content_id(),
                "sources": list(sources),
                "estimator": estimator,
            }
        )
        self.finite = finite
        self.successful = successful


def stack_matter_power_estimates(
    estimates: tuple[MatterPowerEstimate, ...], /
) -> MatterPowerTable:
    if len(estimates) < 2:
        raise ValueError("At least two matter-power estimates are required for a table.")
    first = estimates[0]
    if any(
        estimate.descriptor.descriptor_id != first.descriptor.descriptor_id
        or estimate.scale.scale_id != first.scale.scale_id
        or estimate.estimator_id != first.estimator_id
        or estimate.wavenumbers.shape != first.wavenumbers.shape
        for estimate in estimates[1:]
    ):
        raise ValueError("Matter-power estimates have incompatible descriptors or grids.")
    scale_factors = jnp.stack(tuple(estimate.scale_factor for estimate in estimates))
    if not bool(jnp.all(jnp.diff(scale_factors) > 0.0)):
        raise ValueError(
            "Matter-power estimate scale factors must be strictly increasing."
        )
    for estimate in estimates:
        first.realization.require_compatible(estimate.realization, jnp.asarray(1.0))
        if not bool(estimate.successful):
            raise ValueError("Only successful estimates can form a table.")
        if not bool(jnp.all(estimate.wavenumbers == first.wavenumbers)):
            raise ValueError("Matter-power estimate wavenumbers differ.")
        if not bool(jnp.all(estimate.valid_shells == first.valid_shells)):
            raise ValueError("Matter-power estimate valid-shell masks differ.")
    provenance = CosmologyProductProvenance(
        producer="phydrax.applications.cosmology.stack_matter_power_estimates",
        producer_version="native",
        model_form_id=first.provenance.model_form_id,
        request_id=first.provenance.request_id,
        numerical_policy_id=first.estimator_id,
        physics_policy_id=first.provenance.physics_policy_id,
        scale_id=first.scale.scale_id,
        source_kind="native",
        differentiation=first.provenance.differentiation,
        parent_product_ids=tuple(estimate.content_id for estimate in estimates),
    )
    return MatterPowerTable(
        scale_factors,
        first.wavenumbers[first.valid_shells],
        jnp.stack(
            tuple(estimate.power_values[first.valid_shells] for estimate in estimates)
        ),
        first.descriptor,
        first.scale,
        provenance,
        first.realization,
    )


class CosmologicalFieldSpectrumPlan(StrictModule, NonTrainableState):
    shells: PeriodicFourierShellPlan
    descriptor: MatterPowerDescriptor
    density_convention: FieldDensityConvention = eqx.field(static=True)
    correction: ModeTransferCorrection | None
    shot_noise: float = eqx.field(static=True)
    imaginary_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        shells: PeriodicFourierShellPlan,
        descriptor: MatterPowerDescriptor,
        /,
        *,
        density_convention: FieldDensityConvention = "density-contrast",
        correction: ModeTransferCorrection | None = None,
        shot_noise: float = 0.0,
        imaginary_tolerance: float = 1.0e-10,
    ):
        convention = str(density_convention)
        noise = float(shot_noise)
        tolerance = float(imaginary_tolerance)
        if (
            convention not in ("density-contrast", "density", "extensive-content")
            or not jnp.isfinite(noise)
            or not jnp.isfinite(tolerance)
            or tolerance <= 0.0
            or descriptor.spatial_dimension != len(shells.source_shape)
        ):
            raise ValueError("Cosmological field-spectrum policy is invalid.")
        self.shells = shells
        self.descriptor = descriptor
        self.density_convention = convention
        self.correction = correction
        self.shot_noise = noise
        self.imaginary_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cosmological-field-spectrum",
                "shells": shells.plan_id,
                "descriptor": descriptor.descriptor_id,
                "density_convention": convention,
                "correction": "none" if correction is None else correction.correction_id,
                "shot_noise": noise,
                "imaginary_tolerance": tolerance,
            }
        )

    def _density_contrast(self, field: ArrayLike, /) -> tuple[Array, Array]:
        values = jnp.asarray(field)
        if values.shape != self.shells.source_shape:
            raise ValueError(
                "Cosmological field does not match the Fourier-shell geometry."
            )
        if self.density_convention == "density-contrast":
            return values, jnp.asarray(0.0, dtype=values.dtype)
        density = (
            values
            if self.density_convention == "density"
            else values / self.shells.cell_volume
        )
        mean = jnp.mean(density)
        mean = eqx.error_if(
            mean,
            ~jnp.isfinite(mean) | (mean <= 0.0),
            "Density field mean must be finite and positive.",
        )
        contrast = density / mean - 1.0
        return contrast, jnp.abs(jnp.mean(contrast))

    def transform(self, field: ArrayLike, /) -> tuple[PeriodicFourierField, Array]:
        contrast, mean_defect = self._density_contrast(field)
        return self.shells.transform(contrast), mean_defect

    def _provenance(
        self,
        parent: CosmologyProductProvenance,
        source_ids: tuple[str, ...],
    ) -> CosmologyProductProvenance:
        return CosmologyProductProvenance(
            producer="phydrax.applications.cosmology.CosmologicalFieldSpectrumPlan",
            producer_version="native",
            model_form_id=parent.model_form_id,
            request_id=parent.request_id,
            numerical_policy_id=self.plan_id,
            physics_policy_id=parent.physics_policy_id,
            scale_id=parent.scale_id,
            source_kind="native",
            differentiation=parent.differentiation,
            parent_product_ids=source_ids,
        )

    def estimate_auto(
        self,
        field: ArrayLike,
        scale_factor: ArrayLike,
        scale: CosmologyScaleContract,
        realization: CosmologyRealizationSignature,
        parent_provenance: CosmologyProductProvenance,
        source_product_id: str,
        /,
    ) -> MatterPowerEstimate:
        transformed, mean_defect = self.transform(field)
        statistic = self.shells.auto_power(
            transformed,
            correction=self.correction,
            shot_noise=self.shot_noise,
        )
        provenance = self._provenance(parent_provenance, (source_product_id,))
        estimate = MatterPowerEstimate(
            scale_factor,
            statistic.representative_wavenumbers,
            statistic.shell_values,
            statistic.weighted_mode_count,
            statistic.valid_shells,
            self.descriptor,
            scale,
            provenance,
            realization,
            (source_product_id,),
            self.plan_id,
        )
        return eqx.tree_at(
            lambda value: value.successful,
            estimate,
            estimate.successful & (mean_defect <= 1.0e-10),
        )

    def estimate_cross(
        self,
        left_field: ArrayLike,
        right_field: ArrayLike,
        scale_factor: ArrayLike,
        scale: CosmologyScaleContract,
        realization: CosmologyRealizationSignature,
        parent_provenance: CosmologyProductProvenance,
        source_product_ids: tuple[str, str],
        /,
    ) -> MatterPowerEstimate:
        left, left_mean = self.transform(left_field)
        right, right_mean = self.transform(right_field)
        statistic = self.shells.cross_power(left, right, correction=self.correction)
        provenance = self._provenance(parent_provenance, source_product_ids)
        estimate = MatterPowerEstimate(
            scale_factor,
            statistic.representative_wavenumbers,
            statistic.shell_values,
            statistic.weighted_mode_count,
            statistic.valid_shells,
            self.descriptor,
            scale,
            provenance,
            realization,
            source_product_ids,
            self.plan_id,
        )
        return eqx.tree_at(
            lambda value: value.successful,
            estimate,
            estimate.successful
            & (left_mean <= 1.0e-10)
            & (right_mean <= 1.0e-10)
            & (statistic.imaginary_residual <= self.imaginary_tolerance),
        )


class SpectralFieldDiscrepancyResult(StrictModule):
    shell_discrepancy: Array
    wavenumbers: Array
    mode_counts: Array
    valid_shells: Array
    total_discrepancy: Array
    parseval_residual: Array
    finite: Array
    successful: Array
    source_product_ids: tuple[str, str] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def as_theory_vector(self) -> TheoryVector:
        labels = tuple(
            f"spectral-discrepancy:{self.plan_id}:{index}"
            for index in range(self.shell_discrepancy.size)
        )
        return TheoryVector(
            self.shell_discrepancy,
            CoordinateLayout(labels),
            canonical_fingerprint(
                {
                    "kind": "spectral-discrepancy-vector",
                    "plan": self.plan_id,
                    "sources": list(self.source_product_ids),
                }
            ),
        )


class SpectralFieldDiscrepancyPlan(StrictModule, NonTrainableState):
    shells: PeriodicFourierShellPlan
    correction: ModeTransferCorrection | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        shells: PeriodicFourierShellPlan,
        /,
        *,
        correction: ModeTransferCorrection | None = None,
    ):
        self.shells = shells
        self.correction = correction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-field-discrepancy",
                "shells": shells.plan_id,
                "correction": "none" if correction is None else correction.correction_id,
            }
        )

    def evaluate(
        self,
        predicted_field: ArrayLike,
        target_field: ArrayLike,
        predicted_product_id: str,
        target_product_id: str,
        /,
    ) -> SpectralFieldDiscrepancyResult:
        predicted = jnp.asarray(predicted_field)
        target = jnp.asarray(target_field, dtype=predicted.dtype)
        if predicted.shape != self.shells.source_shape or target.shape != predicted.shape:
            raise ValueError("Spectral discrepancy fields must match shell geometry.")
        predicted_fourier = self.shells.transform(predicted)
        target_fourier = self.shells.transform(target)
        statistic = self.shells.discrepancy(
            predicted_fourier,
            target_fourier,
            correction=self.correction,
        )
        real_discrepancy = self.shells.cell_volume * jnp.sum((predicted - target) ** 2)
        parseval = jnp.abs(statistic.total_weighted_value - real_discrepancy)
        finite = statistic.finite & jnp.isfinite(parseval)
        return SpectralFieldDiscrepancyResult(
            statistic.shell_values,
            statistic.representative_wavenumbers,
            statistic.weighted_mode_count,
            statistic.valid_shells,
            statistic.total_weighted_value,
            parseval,
            finite,
            finite & statistic.successful,
            (str(predicted_product_id), str(target_product_id)),
            self.plan_id,
        )


__all__ = [
    "CosmologicalFieldSpectrumPlan",
    "FieldDensityConvention",
    "MatterPowerEstimate",
    "SpectralFieldDiscrepancyPlan",
    "SpectralFieldDiscrepancyResult",
    "stack_matter_power_estimates",
]
