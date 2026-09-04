#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomic homogenized Shorten-A2 to GASAM prescribed-activation coupling."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ..fibers import SkeletalFiberBundleCandidate, SkeletalFiberBundleState
from ._gasam import (
    GasamMaterialCandidate,
    GasamMaterialCommit,
    GasamMaterialState,
    PreparedEngelhardtGasam2025Material,
    PrescribedActivationEvidence,
)


class ShortenGasamActivationCalibration(StrictModule):
    """Source-backed A2 concentration anchors for one normalized activation map."""

    resting_crossbridge_uM: Array
    saturated_crossbridge_uM: Array

    def __init__(
        self,
        resting_crossbridge_uM: ArrayLike,
        saturated_crossbridge_uM: ArrayLike,
        /,
    ):
        resting = jnp.asarray(resting_crossbridge_uM)
        saturated = jnp.asarray(saturated_crossbridge_uM)
        if resting.shape != () or saturated.shape != ():
            raise ValueError("Crossbridge calibration anchors must be scalar.")
        if not jnp.issubdtype(resting.dtype, jnp.inexact):
            resting = resting.astype(float)
        saturated = saturated.astype(resting.dtype)
        self.resting_crossbridge_uM = resting
        self.saturated_crossbridge_uM = saturated


class ShortenGasamCouplingEvidence(StrictModule, NonTrainableState):
    source_successful: Array
    target_successful: Array
    weights_valid: Array
    calibration_valid: Array
    finite: Array
    activation_in_support: Array
    branch_smooth: Array
    weighted_crossbridge_uM: Array
    prescribed_activation: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    force_owner: str = eqx.field(static=True, default="engelhardt-gasam-2025")


class ShortenGasamCouplingCommit(StrictModule, NonTrainableState):
    fiber_state: SkeletalFiberBundleState
    material: PreparedEngelhardtGasam2025Material
    committed: Array
    rollback_applied: Array
    plan_id: str = eqx.field(static=True)


class ShortenGasamCouplingCandidate(StrictModule, NonTrainableState):
    source: SkeletalFiberBundleCandidate
    material_source: PreparedEngelhardtGasam2025Material
    material_candidate: GasamMaterialCandidate
    evidence: ShortenGasamCouplingEvidence

    def commit(self, /) -> ShortenGasamCouplingCommit:
        accepted = self.evidence.successful
        source_state = self.source.source_state
        proposed_source = self.source.commit()
        fiber_state = SkeletalFiberBundleState(
            jnp.where(accepted, proposed_source.time_ms, source_state.time_ms),
            jnp.where(accepted, proposed_source.values, source_state.values),
        )
        target_commit = self.material_candidate.commit()
        source_evidence = self.material_source.state.evidence
        target_evidence = target_commit.state.evidence
        selected_evidence = PrescribedActivationEvidence(
            jnp.where(
                accepted,
                target_evidence.activation,
                source_evidence.activation,
            ),
            jnp.where(accepted, target_evidence.finite, source_evidence.finite),
            jnp.where(
                accepted,
                target_evidence.in_support,
                source_evidence.in_support,
            ),
            jnp.where(accepted, target_evidence.valid, source_evidence.valid),
            target_evidence.source_id,
        )
        selected_state = GasamMaterialState(
            jnp.where(
                accepted,
                target_commit.state.activation,
                self.material_source.state.activation,
            ),
            selected_evidence,
            jnp.where(
                accepted,
                target_commit.state.state_id,
                self.material_source.state.state_id,
            ),
        )
        selected_target = GasamMaterialCommit(
            selected_state,
            accepted,
            ~accepted,
            target_commit.source_state_id,
            target_commit.source_activation,
            target_commit.prepared_id,
            target_commit.candidate_id,
        )
        material = self.material_source.with_commit(selected_target)
        return ShortenGasamCouplingCommit(
            fiber_state,
            material,
            accepted,
            ~accepted,
            self.evidence.plan_id,
        )


class HomogenizedShortenGasamCouplingPlan(StrictModule, NonTrainableState):
    source_weights: Array
    calibration_asset_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, source_weights: ArrayLike, /, *, calibration_asset_id: str
    ):
        weights = jnp.asarray(source_weights)
        if weights.ndim != 2:
            raise ValueError("source_weights must have shape (fiber, node).")
        if not jnp.issubdtype(weights.dtype, jnp.inexact):
            weights = weights.astype(float)
        asset = str(calibration_asset_id).strip()
        if not asset:
            raise ValueError("calibration_asset_id must be nonempty.")
        host = np.asarray(weights)
        if (
            not np.all(np.isfinite(host))
            or np.any(host < 0.0)
            or not np.isclose(np.sum(host), 1.0, rtol=0.0, atol=1.0e-10)
        ):
            raise ValueError("source_weights must be finite, nonnegative, and sum to one.")
        self.source_weights = weights
        self.calibration_asset_id = asset
        self.plan_id = canonical_fingerprint(
            {
                "kind": "homogenized-shorten-a2-to-gasam-activation",
                "weights": array_tree_fingerprint(weights),
                "calibration_asset_id": asset,
                "source_quantity": "shorten-2007-A_2-uM",
                "force_owner": "engelhardt-gasam-2025",
            }
        )

    def prepare(
        self,
        calibration: ShortenGasamActivationCalibration,
        /,
    ) -> PreparedHomogenizedShortenGasamCoupling:
        if not isinstance(calibration, ShortenGasamActivationCalibration):
            raise TypeError("calibration must be ShortenGasamActivationCalibration.")
        return PreparedHomogenizedShortenGasamCoupling(self, calibration)


class PreparedHomogenizedShortenGasamCoupling(StrictModule):
    plan: HomogenizedShortenGasamCouplingPlan
    calibration: ShortenGasamActivationCalibration

    def activation_from_crossbridge(self, crossbridge_uM: ArrayLike, /) -> tuple[Array, Array]:
        values = jnp.asarray(crossbridge_uM)
        if values.shape != self.plan.source_weights.shape:
            raise ValueError(
                "crossbridge_uM must match the prepared (fiber, node) support."
            )
        weighted = jnp.sum(values * self.plan.source_weights)
        span = (
            self.calibration.saturated_crossbridge_uM
            - self.calibration.resting_crossbridge_uM
        )
        safe_span = jnp.where(span > 0.0, span, 1.0)
        activation = jnp.clip(
            (weighted - self.calibration.resting_crossbridge_uM) / safe_span,
            0.0,
            1.0,
        )
        return activation, weighted

    def candidate(
        self,
        source: SkeletalFiberBundleCandidate,
        material: PreparedEngelhardtGasam2025Material,
        /,
    ) -> ShortenGasamCouplingCandidate:
        if not isinstance(source, SkeletalFiberBundleCandidate):
            raise TypeError("source must be SkeletalFiberBundleCandidate.")
        if not isinstance(material, PreparedEngelhardtGasam2025Material):
            raise TypeError("material must be PreparedEngelhardtGasam2025Material.")
        activation, weighted = self.activation_from_crossbridge(
            source.output.force_bearing_crossbridge_uM
        )
        target = material.propose_activation(activation)
        weights_valid = (
            jnp.all(jnp.isfinite(self.plan.source_weights))
            & jnp.all(self.plan.source_weights >= 0.0)
            & jnp.isclose(jnp.sum(self.plan.source_weights), 1.0)
        )
        span = (
            self.calibration.saturated_crossbridge_uM
            - self.calibration.resting_crossbridge_uM
        )
        calibration_valid = (
            jnp.isfinite(self.calibration.resting_crossbridge_uM)
            & jnp.isfinite(self.calibration.saturated_crossbridge_uM)
            & (self.calibration.resting_crossbridge_uM >= 0.0)
            & (span > 0.0)
        )
        finite = jnp.isfinite(weighted) & jnp.isfinite(activation)
        in_support = (activation >= 0.0) & (activation <= 1.0)
        unclipped = (
            weighted > self.calibration.resting_crossbridge_uM
        ) & (weighted < self.calibration.saturated_crossbridge_uM)
        source_ok = source.evidence.successful
        target_ok = target.evidence.valid
        successful = (
            source_ok
            & target_ok
            & weights_valid
            & calibration_valid
            & finite
            & in_support
        )
        evidence = ShortenGasamCouplingEvidence(
            source_ok,
            target_ok,
            weights_valid,
            calibration_valid,
            finite,
            in_support,
            unclipped,
            weighted,
            activation,
            successful,
            self.plan.plan_id,
        )
        return ShortenGasamCouplingCandidate(source, material, target, evidence)


__all__ = [
    "HomogenizedShortenGasamCouplingPlan",
    "PreparedHomogenizedShortenGasamCoupling",
    "ShortenGasamActivationCalibration",
    "ShortenGasamCouplingCandidate",
    "ShortenGasamCouplingCommit",
    "ShortenGasamCouplingEvidence",
]
