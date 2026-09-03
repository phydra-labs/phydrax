#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Reference-explicit finite strain derived from deformation observations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import canonical_fingerprint
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


def _inverse_3x3(matrix: Array, determinant: Array, /) -> Array:
    a, b, c = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    d, e, f = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    g, h, i = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    rows = (
        jnp.stack((e * i - f * h, c * h - b * i, b * f - c * e), axis=-1),
        jnp.stack((f * g - d * i, a * i - c * g, c * d - a * f), axis=-1),
        jnp.stack((d * h - e * g, b * g - a * h, a * e - b * d), axis=-1),
    )
    adjugate = jnp.stack(rows, axis=-2)
    epsilon = jnp.finfo(matrix.dtype).eps
    safe = jnp.where(
        jnp.abs(determinant) > epsilon,
        determinant,
        jnp.full_like(determinant, jnp.nan),
    )
    return adjugate / safe[..., None, None]


class StrainMeasure(Enum):
    """Reference configuration of a finite-strain observation."""

    GREEN_LAGRANGE = "green-lagrange-reference"
    EULERIAN = "euler-almansi-current"


def green_lagrange_strain(deformation_gradient: ArrayLike, /) -> Array:
    """Return E = 1/2 (FᵀF - I) in the reference configuration."""
    gradient = jnp.asarray(deformation_gradient)
    if gradient.ndim < 2 or gradient.shape[-2:] != (3, 3):
        raise ValueError("deformation_gradient must end with shape (3, 3).")
    if jnp.issubdtype(gradient.dtype, jnp.complexfloating):
        raise TypeError("deformation_gradient must be real.")
    if not jnp.issubdtype(gradient.dtype, jnp.floating):
        gradient = gradient.astype(float)
    right_cauchy_green = contract("...ki,...kj->...ij", gradient, gradient)
    return 0.5 * (right_cauchy_green - jnp.eye(3, dtype=gradient.dtype))


def eulerian_strain(deformation_gradient: ArrayLike, /) -> Array:
    """Return Euler–Almansi e = 1/2 (I - F⁻ᵀF⁻¹)."""
    gradient = jnp.asarray(deformation_gradient)
    if gradient.ndim < 2 or gradient.shape[-2:] != (3, 3):
        raise ValueError("deformation_gradient must end with shape (3, 3).")
    if jnp.issubdtype(gradient.dtype, jnp.complexfloating):
        raise TypeError("deformation_gradient must be real.")
    if not jnp.issubdtype(gradient.dtype, jnp.floating):
        gradient = gradient.astype(float)
    determinant = _determinant_3x3(gradient)
    inverse = _inverse_3x3(gradient, determinant)
    inverse_left_cauchy_green = contract("...ki,...kj->...ij", inverse, inverse)
    return 0.5 * (jnp.eye(3, dtype=gradient.dtype) - inverse_left_cauchy_green)


def _evaluate_measure(deformation_gradient: Array, measure: StrainMeasure, /) -> Array:
    if measure is StrainMeasure.GREEN_LAGRANGE:
        return green_lagrange_strain(deformation_gradient)
    if measure is StrainMeasure.EULERIAN:
        return eulerian_strain(deformation_gradient)
    raise TypeError("measure must be a StrainMeasure.")


class StrainEvidence(StrictModule):
    """Reference, orientation, symmetry, and uncertainty evidence."""

    reference_frame_matched: Array
    jacobian_determinant: Array
    folding_mask: Array
    folding_count: Array
    folding_fraction: Array
    invertible: Array
    symmetry_residual: Array
    uncertainty_available: Array
    uncertainty_valid: Array
    finite: Array
    successful: Array


class StrainResult(StrictModule):
    """Finite-strain tensor and first-order independent-input uncertainty."""

    strain: Array
    strain_standard_deviation: Array
    deformation_gradient: Array
    deformation_gradient_standard_deviation: Array
    evidence: StrainEvidence
    measure: StrainMeasure = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class StrainEvaluationPlan:
    """Fixed-shape strain evaluation in a declared spatial reference frame."""

    sample_shape: tuple[int, ...]
    reference_frame_id: str
    measure: StrainMeasure
    minimum_jacobian: float = 0.0
    require_uncertainty: bool = False
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        sample_shape = tuple(int(size) for size in self.sample_shape)
        if any(size <= 0 for size in sample_shape):
            raise ValueError("sample_shape dimensions must be positive.")
        reference = _identifier(self.reference_frame_id, "reference_frame_id")
        if not isinstance(self.measure, StrainMeasure):
            raise TypeError("measure must be a StrainMeasure.")
        minimum = float(self.minimum_jacobian)
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_jacobian must be finite and non-negative.")
        if not isinstance(self.require_uncertainty, bool):
            raise TypeError("require_uncertainty must be boolean.")
        object.__setattr__(self, "sample_shape", sample_shape)
        object.__setattr__(self, "reference_frame_id", reference)
        object.__setattr__(self, "minimum_jacobian", minimum)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-strain-evaluation-plan",
                    "sample_shape": list(sample_shape),
                    "reference_frame_id": reference,
                    "measure": self.measure.value,
                    "minimum_jacobian": minimum,
                    "require_uncertainty": self.require_uncertainty,
                }
            ),
        )

    def prepare(self) -> "PreparedStrainEvaluation":
        return PreparedStrainEvaluation(
            self.sample_shape,
            self.reference_frame_id,
            self.measure,
            self.minimum_jacobian,
            self.require_uncertainty,
            self.plan_id,
        )


class PreparedStrainEvaluation(StrictModule, NonTrainableState):
    """Prepared differentiable strain and uncertainty evaluation."""

    sample_shape: tuple[int, ...] = eqx.field(static=True)
    reference_frame_id: str = eqx.field(static=True)
    measure: StrainMeasure = eqx.field(static=True)
    minimum_jacobian: float = eqx.field(static=True)
    require_uncertainty: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_shape: Sequence[int],
        reference_frame_id: str,
        measure: StrainMeasure,
        minimum_jacobian: float,
        require_uncertainty: bool,
        plan_id: str,
        /,
    ):
        self.sample_shape = tuple(int(size) for size in sample_shape)
        self.reference_frame_id = _identifier(reference_frame_id, "reference_frame_id")
        self.measure = measure
        self.minimum_jacobian = float(minimum_jacobian)
        self.require_uncertainty = bool(require_uncertainty)
        self.plan_id = _identifier(plan_id, "plan_id")
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-cardiovascular-strain-evaluation", "plan_id": self.plan_id}
        )

    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        /,
        *,
        deformation_gradient_standard_deviation: ArrayLike | None = None,
        reference_frame_id: str,
    ) -> StrainResult:
        gradient = jnp.asarray(deformation_gradient)
        expected = self.sample_shape + (3, 3)
        if gradient.shape != expected:
            raise ValueError(
                f"deformation_gradient must have shape {expected}; got {gradient.shape}."
            )
        if jnp.issubdtype(gradient.dtype, jnp.complexfloating):
            raise TypeError("deformation_gradient must be real.")
        if not jnp.issubdtype(gradient.dtype, jnp.floating):
            gradient = gradient.astype(float)
        uncertainty_available = deformation_gradient_standard_deviation is not None
        gradient_std = (
            jnp.zeros_like(gradient)
            if deformation_gradient_standard_deviation is None
            else jnp.asarray(
                deformation_gradient_standard_deviation,
                dtype=gradient.dtype,
            )
        )
        if gradient_std.shape != gradient.shape:
            raise ValueError(
                "deformation_gradient_standard_deviation must match deformation_gradient shape."
            )

        strain = _evaluate_measure(gradient, self.measure)
        if uncertainty_available:
            variance = jnp.zeros_like(strain)
            action = lambda value: _evaluate_measure(value, self.measure)
            for row in range(3):
                for column in range(3):
                    tangent = (
                        jnp.zeros_like(gradient)
                        .at[..., row, column]
                        .set(gradient_std[..., row, column])
                    )
                    _, strain_tangent = jax.jvp(action, (gradient,), (tangent,))
                    variance = variance + strain_tangent * strain_tangent
            strain_std = jnp.sqrt(jnp.maximum(variance, 0.0))
        else:
            strain_std = jnp.zeros_like(strain)

        determinant = _determinant_3x3(gradient)
        folding = determinant <= self.minimum_jacobian
        folding_count = jnp.sum(folding, dtype=jnp.int32)
        sample_count = jnp.asarray(determinant.size, dtype=jnp.int32)
        folding_fraction = folding_count.astype(gradient.real.dtype) / jnp.maximum(
            sample_count, 1
        ).astype(gradient.real.dtype)
        invertible = jnp.all(jnp.abs(determinant) > jnp.finfo(gradient.real.dtype).eps)
        symmetry_residual = jnp.max(jnp.abs(strain - jnp.swapaxes(strain, -1, -2)))
        uncertainty_available_array = jnp.asarray(uncertainty_available)
        uncertainty_valid = uncertainty_available_array & jnp.all(
            jnp.isfinite(gradient_std) & (gradient_std >= 0.0)
        )
        finite = (
            jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(strain))
            & jnp.all(jnp.isfinite(strain_std))
            & jnp.all(jnp.isfinite(determinant))
            & jnp.isfinite(symmetry_residual)
        )
        runtime_reference_frame_id = _identifier(reference_frame_id, "reference_frame_id")
        reference_matched = jnp.asarray(
            runtime_reference_frame_id == self.reference_frame_id
        )
        uncertainty_requirement = jnp.where(
            uncertainty_available_array,
            uncertainty_valid,
            jnp.asarray(not self.require_uncertainty),
        )
        successful = (
            finite
            & reference_matched
            & invertible
            & (folding_count == 0)
            & uncertainty_requirement
        )
        evidence = StrainEvidence(
            reference_matched,
            determinant,
            folding,
            folding_count,
            folding_fraction,
            invertible,
            symmetry_residual,
            uncertainty_available_array,
            uncertainty_valid,
            finite,
            successful,
        )
        return StrainResult(
            strain,
            strain_std,
            gradient,
            gradient_std,
            evidence,
            self.measure,
            self.prepared_id,
        )


__all__ = [
    "PreparedStrainEvaluation",
    "StrainEvaluationPlan",
    "StrainEvidence",
    "StrainMeasure",
    "StrainResult",
    "eulerian_strain",
    "green_lagrange_strain",
]
