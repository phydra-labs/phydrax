#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Observational evaluation of prepared deformation registrations.

Registration here evaluates externally estimated deformation observations.  It
is intentionally not a mechanics constitutive model or equilibrium solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace.")
    return normalized


def _determinant_3x3(matrix: Array, /) -> Array:
    return (
        matrix[..., 0, 0]
        * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1]
        * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2]
        * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )


class RegistrationDirection(Enum):
    """Declared domain/codomain direction of a displacement observation."""

    REFERENCE_TO_TARGET = "reference-to-target"
    TARGET_TO_REFERENCE = "target-to-reference"


class RegistrationEvidence(StrictModule):
    """Frame, folding, inverse-consistency, and uncertainty evidence."""

    reference_frame_matched: Array
    target_frame_matched: Array
    jacobian_determinant: Array
    folding_mask: Array
    folding_count: Array
    folding_fraction: Array
    inverse_consistency_available: Array
    inverse_consistency_rms_mm: Array
    inverse_consistency_max_mm: Array
    inverse_consistent: Array
    uncertainty_available: Array
    uncertainty_rms_mm: Array
    uncertainty_valid: Array
    finite: Array
    successful: Array


class RegistrationCandidate(StrictModule):
    """Evaluated deformation observation before an explicit host commit."""

    reference_points_mm: Array
    deformed_points_mm: Array
    displacement_mm: Array
    displacement_gradient: Array
    deformation_gradient: Array
    inverse_consistency_residual_mm: Array
    displacement_standard_deviation_mm: Array
    evidence: RegistrationEvidence
    prepared_id: str = eqx.field(static=True)


class RegistrationCheckpoint(StrictModule, NonTrainableState):
    """Successful committed deformation evaluation with stable lineage."""

    displacement_mm: Array
    displacement_gradient: Array
    deformation_gradient: Array
    checkpoint_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class RegistrationEvaluationPlan:
    """Fixed-topology plan for evaluating a deformation observation."""

    reference_points_mm: np.ndarray
    reference_frame_id: str
    target_frame_id: str
    direction: RegistrationDirection = RegistrationDirection.REFERENCE_TO_TARGET
    minimum_jacobian: float = 0.0
    inverse_consistency_tolerance_mm: float = 0.5
    require_inverse_consistency: bool = False
    require_uncertainty: bool = False
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        original = np.asarray(self.reference_points_mm)
        points = np.asarray(
            self.reference_points_mm, dtype=np.result_type(original.dtype, np.float64)
        )
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError(
                "reference_points_mm must end with a coordinate axis of length three."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("reference_points_mm must be finite.")
        if not isinstance(self.direction, RegistrationDirection):
            raise TypeError("direction must be a RegistrationDirection.")
        if self.direction is not RegistrationDirection.REFERENCE_TO_TARGET:
            raise ValueError(
                "RegistrationEvaluationPlan currently supports only "
                "REFERENCE_TO_TARGET displacement observations."
            )
        minimum = float(self.minimum_jacobian)
        tolerance = float(self.inverse_consistency_tolerance_mm)
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_jacobian must be finite and non-negative.")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "inverse_consistency_tolerance_mm must be finite and non-negative."
            )
        if not isinstance(self.require_inverse_consistency, bool):
            raise TypeError("require_inverse_consistency must be boolean.")
        if not isinstance(self.require_uncertainty, bool):
            raise TypeError("require_uncertainty must be boolean.")
        points = np.array(points, copy=True)
        points.setflags(write=False)
        reference = _identifier(self.reference_frame_id, "reference_frame_id")
        target = _identifier(self.target_frame_id, "target_frame_id")
        if reference == target:
            raise ValueError("reference_frame_id and target_frame_id must be distinct.")
        object.__setattr__(self, "reference_points_mm", points)
        object.__setattr__(self, "reference_frame_id", reference)
        object.__setattr__(self, "target_frame_id", target)
        object.__setattr__(self, "minimum_jacobian", minimum)
        object.__setattr__(self, "inverse_consistency_tolerance_mm", tolerance)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-registration-evaluation-plan",
                    "reference_points_mm": array_tree_fingerprint(points),
                    "reference_frame_id": reference,
                    "target_frame_id": target,
                    "direction": self.direction.value,
                    "minimum_jacobian": minimum,
                    "inverse_consistency_tolerance_mm": tolerance,
                    "require_inverse_consistency": self.require_inverse_consistency,
                    "require_uncertainty": self.require_uncertainty,
                }
            ),
        )

    def prepare(self) -> "PreparedRegistrationEvaluation":
        return PreparedRegistrationEvaluation(
            self.reference_points_mm,
            self.reference_frame_id,
            self.target_frame_id,
            self.direction,
            self.minimum_jacobian,
            self.inverse_consistency_tolerance_mm,
            self.require_inverse_consistency,
            self.require_uncertainty,
            self.plan_id,
        )


class PreparedRegistrationEvaluation(StrictModule, NonTrainableState):
    """Prepared registration diagnostics over a fixed reference support."""

    reference_points_mm: Array
    reference_frame_id: str = eqx.field(static=True)
    target_frame_id: str = eqx.field(static=True)
    direction: RegistrationDirection = eqx.field(static=True)
    minimum_jacobian: float = eqx.field(static=True)
    inverse_consistency_tolerance_mm: float = eqx.field(static=True)
    require_inverse_consistency: bool = eqx.field(static=True)
    require_uncertainty: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_points_mm: ArrayLike,
        reference_frame_id: str,
        target_frame_id: str,
        direction: RegistrationDirection,
        minimum_jacobian: float,
        inverse_consistency_tolerance_mm: float,
        require_inverse_consistency: bool,
        require_uncertainty: bool,
        plan_id: str,
        /,
    ):
        points = jax.lax.stop_gradient(jnp.asarray(reference_points_mm))
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("reference_points_mm must end in three coordinates.")
        self.reference_points_mm = points
        self.reference_frame_id = _identifier(reference_frame_id, "reference_frame_id")
        self.target_frame_id = _identifier(target_frame_id, "target_frame_id")
        self.direction = direction
        self.minimum_jacobian = float(minimum_jacobian)
        self.inverse_consistency_tolerance_mm = float(inverse_consistency_tolerance_mm)
        self.require_inverse_consistency = bool(require_inverse_consistency)
        self.require_uncertainty = bool(require_uncertainty)
        self.plan_id = _identifier(plan_id, "plan_id")
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-registration-evaluation",
                "plan_id": self.plan_id,
            }
        )

    def evaluate(
        self,
        displacement_mm: ArrayLike,
        displacement_gradient: ArrayLike,
        /,
        *,
        inverse_displacement_at_deformed_mm: ArrayLike | None = None,
        displacement_standard_deviation_mm: ArrayLike | None = None,
        reference_frame_id: str,
        target_frame_id: str,
    ) -> RegistrationCandidate:
        displacement = jnp.asarray(displacement_mm, dtype=self.reference_points_mm.dtype)
        gradient = jnp.asarray(
            displacement_gradient, dtype=self.reference_points_mm.dtype
        )
        if displacement.shape != self.reference_points_mm.shape:
            raise ValueError("displacement_mm must match reference_points_mm shape.")
        if gradient.shape != self.reference_points_mm.shape[:-1] + (3, 3):
            raise ValueError(
                "displacement_gradient must match reference point shape and end with (3, 3)."
            )
        deformation_gradient = gradient + jnp.eye(3, dtype=gradient.dtype)
        determinant = _determinant_3x3(deformation_gradient)
        folding = determinant <= self.minimum_jacobian
        folding_count = jnp.sum(folding, dtype=jnp.int32)
        sample_count = jnp.asarray(folding.size, dtype=jnp.int32)
        folding_fraction = folding_count.astype(determinant.dtype) / jnp.maximum(
            sample_count, 1
        ).astype(determinant.dtype)
        deformed = self.reference_points_mm + displacement

        inverse_available = inverse_displacement_at_deformed_mm is not None
        inverse = (
            jnp.zeros_like(displacement)
            if inverse_displacement_at_deformed_mm is None
            else jnp.asarray(
                inverse_displacement_at_deformed_mm,
                dtype=self.reference_points_mm.dtype,
            )
        )
        if inverse.shape != displacement.shape:
            raise ValueError(
                "inverse_displacement_at_deformed_mm must match displacement_mm shape."
            )
        inverse_available_array = jnp.asarray(inverse_available)
        inverse_residual = jnp.where(
            inverse_available_array,
            displacement + inverse,
            jnp.zeros_like(displacement),
        )
        inverse_norm = jnp.sqrt(jnp.sum(inverse_residual * inverse_residual, axis=-1))
        inverse_rms = jnp.sqrt(jnp.mean(inverse_norm * inverse_norm))
        inverse_max = jnp.max(inverse_norm)
        inverse_consistent = inverse_available_array & (
            inverse_max <= self.inverse_consistency_tolerance_mm
        )

        uncertainty_available = displacement_standard_deviation_mm is not None
        uncertainty = (
            jnp.zeros_like(displacement)
            if displacement_standard_deviation_mm is None
            else jnp.asarray(
                displacement_standard_deviation_mm,
                dtype=self.reference_points_mm.dtype,
            )
        )
        if uncertainty.shape != displacement.shape:
            raise ValueError(
                "displacement_standard_deviation_mm must match displacement_mm shape."
            )
        uncertainty_available_array = jnp.asarray(uncertainty_available)
        uncertainty_valid = uncertainty_available_array & jnp.all(
            jnp.isfinite(uncertainty) & (uncertainty >= 0.0)
        )
        uncertainty_rms = jnp.sqrt(jnp.mean(uncertainty * uncertainty))

        runtime_reference_frame_id = _identifier(reference_frame_id, "reference_frame_id")
        runtime_target_frame_id = _identifier(target_frame_id, "target_frame_id")
        reference_matched = jnp.asarray(
            runtime_reference_frame_id == self.reference_frame_id
        )
        target_matched = jnp.asarray(runtime_target_frame_id == self.target_frame_id)
        finite = (
            jnp.all(jnp.isfinite(self.reference_points_mm))
            & jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(deformation_gradient))
            & jnp.all(jnp.isfinite(determinant))
            & jnp.all(jnp.isfinite(inverse))
            & jnp.all(jnp.isfinite(uncertainty))
        )
        inverse_requirement = jnp.where(
            inverse_available_array,
            inverse_consistent,
            jnp.asarray(not self.require_inverse_consistency),
        )
        uncertainty_requirement = jnp.where(
            uncertainty_available_array,
            uncertainty_valid,
            jnp.asarray(not self.require_uncertainty),
        )
        successful = (
            finite
            & reference_matched
            & target_matched
            & (folding_count == 0)
            & inverse_requirement
            & uncertainty_requirement
        )
        evidence = RegistrationEvidence(
            reference_matched,
            target_matched,
            determinant,
            folding,
            folding_count,
            folding_fraction,
            inverse_available_array,
            inverse_rms,
            inverse_max,
            inverse_consistent,
            uncertainty_available_array,
            uncertainty_rms,
            uncertainty_valid,
            finite,
            successful,
        )
        return RegistrationCandidate(
            self.reference_points_mm,
            deformed,
            displacement,
            gradient,
            deformation_gradient,
            inverse_residual,
            uncertainty,
            evidence,
            self.prepared_id,
        )

    def commit(self, candidate: RegistrationCandidate, /) -> RegistrationCheckpoint:
        """Commit only a successful host-evaluated candidate."""
        if not isinstance(candidate, RegistrationCandidate):
            raise TypeError("candidate must be a RegistrationCandidate.")
        if candidate.prepared_id != self.prepared_id:
            raise ValueError("Registration candidate belongs to another prepared plan.")
        if not bool(np.asarray(candidate.evidence.successful)):
            raise ValueError("Cannot commit an unsuccessful registration candidate.")
        checkpoint_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-registration-checkpoint",
                "plan_id": self.plan_id,
                "prepared_id": self.prepared_id,
                "displacement_mm": array_tree_fingerprint(candidate.displacement_mm),
                "displacement_gradient": array_tree_fingerprint(
                    candidate.displacement_gradient
                ),
            }
        )
        return RegistrationCheckpoint(
            jax.lax.stop_gradient(candidate.displacement_mm),
            jax.lax.stop_gradient(candidate.displacement_gradient),
            jax.lax.stop_gradient(candidate.deformation_gradient),
            checkpoint_id,
            self.plan_id,
            self.prepared_id,
        )


__all__ = [
    "PreparedRegistrationEvaluation",
    "RegistrationCandidate",
    "RegistrationCheckpoint",
    "RegistrationDirection",
    "RegistrationEvaluationPlan",
    "RegistrationEvidence",
]
