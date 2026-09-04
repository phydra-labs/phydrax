#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent identifiability qualification for physical force calibration."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._force_calibration import (
    PhysicalRelativeForceCalibrationCandidate,
    PhysicalRelativeForceCalibrationStatus,
)


class PhysicalRelativeForceCalibrationQualificationEvidence(
    StrictModule, NonTrainableState
):
    """Positive-control recovery and confounded negative-control evidence."""

    recovered_scale_newton_per_relative_force: Array
    expected_scale_newton_per_relative_force: Array
    relative_scale_error: Array
    identifiable_control_accepted: Array
    confounded_control_rejected: Array
    confounded_scale_flagged: Array
    valid: Array
    claim_scope: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)


class PhysicalRelativeForceCalibrationQualificationPlan(
    StrictModule, NonTrainableState
):
    """Qualification policy requiring both identifiable and confounded controls."""

    relative_scale_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, relative_scale_tolerance: float = 1.0e-5):
        tolerance = float(relative_scale_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("relative_scale_tolerance must be positive and finite.")
        self.relative_scale_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "physical-relative-force-calibration-qualification",
                "relative_scale_tolerance": tolerance.hex(),
                "controls": ["identifiable", "scale-confounded-with-nuisance"],
            }
        )

    def evaluate(
        self,
        identifiable_candidate: PhysicalRelativeForceCalibrationCandidate,
        confounded_candidate: PhysicalRelativeForceCalibrationCandidate,
        expected_scale_newton_per_relative_force: float,
        /,
    ) -> PhysicalRelativeForceCalibrationQualificationEvidence:
        """Require scale recovery and fail-closed nuisance confounding detection."""

        if not isinstance(
            identifiable_candidate, PhysicalRelativeForceCalibrationCandidate
        ):
            raise TypeError(
                "identifiable_candidate must be PhysicalRelativeForceCalibrationCandidate."
            )
        if not isinstance(confounded_candidate, PhysicalRelativeForceCalibrationCandidate):
            raise TypeError(
                "confounded_candidate must be PhysicalRelativeForceCalibrationCandidate."
            )
        expected = jnp.asarray(
            expected_scale_newton_per_relative_force,
            dtype=identifiable_candidate.proposed.scale_newton_per_relative_force.dtype,
        )
        if expected.shape != ():
            raise ValueError("expected_scale_newton_per_relative_force must be scalar.")
        recovered = identifiable_candidate.proposed.scale_newton_per_relative_force
        error = jnp.abs(recovered - expected) / jnp.maximum(jnp.abs(expected), 1.0)
        accepted = identifiable_candidate.evidence.successful
        rejected = ~confounded_candidate.evidence.successful
        scale_flag = (
            jnp.bitwise_and(
                confounded_candidate.evidence.status,
                int(PhysicalRelativeForceCalibrationStatus.SCALE_NOT_IDENTIFIABLE),
            )
            != 0
        )
        valid = (
            jnp.isfinite(error)
            & (expected > 0.0)
            & (error <= self.relative_scale_tolerance)
            & accepted
            & rejected
            & scale_flag
        )
        return PhysicalRelativeForceCalibrationQualificationEvidence(
            recovered,
            expected,
            error,
            accepted,
            rejected,
            scale_flag,
            valid,
            (
                "linear observation equation under the exact declared protocol/asset "
                "designs; no quantity-conversion or cross-fidelity identity claim"
            ),
            canonical_fingerprint(
                {
                    "kind": "physical-relative-force-calibration-qualified-controls",
                    "plan": self.plan_id,
                    "identifiable_calibration": identifiable_candidate.proposed.plan_id,
                    "confounded_calibration": confounded_candidate.proposed.plan_id,
                }
            ),
        )


__all__ = [
    "PhysicalRelativeForceCalibrationQualificationEvidence",
    "PhysicalRelativeForceCalibrationQualificationPlan",
]
