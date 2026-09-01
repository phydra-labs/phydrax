#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._topology import MacromolecularStructure
from ._types import AlignmentStatus


def _alignment_contract(dtype: str) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "weighted-rigid-structure-alignment",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Weighted Kabsch SVD; derivatives are undefined at repeated singular "
            "values and reflection branch changes."
        ),
        truncation_statement="No points are truncated; zero-weight points are excluded exactly.",
        capacity_semantics="Coordinate capacity is the input leading dimension.",
        assumptions=("Corresponding rows denote corresponding sites.",),
        nondifferentiable_outputs=("status", "valid", "rank"),
        input_dtype=dtype,
        compute_dtype=dtype,
        output_dtype=dtype,
    )


class RigidAlignmentResult(StrictModule):
    """Optimal proper rigid transform mapping mobile coordinates to reference."""

    rotation: Array
    translation: Array
    aligned: Array
    residuals: Array
    rmsd: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    evidence_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        rotation: Array,
        translation: Array,
        aligned: Array,
        residuals: Array,
        rmsd: Array,
        valid: Array,
        status: Array,
        evidence: Array,
        method_contract: BioinformaticsMethodContract,
    ):
        self.rotation = rotation
        self.translation = translation
        self.aligned = aligned
        self.residuals = residuals
        self.rmsd = rmsd
        self.valid = valid
        self.status = status
        self.evidence = evidence
        self.method_contract = method_contract
        self.evidence_labels = (
            "positive_weight_count",
            "singular_value_0",
            "singular_value_1",
            "singular_value_2",
        )


def align_coordinates(
    mobile: ArrayLike,
    reference: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    mask: ArrayLike | None = None,
) -> RigidAlignmentResult:
    """Compute the weighted least-squares proper rigid alignment."""

    mobile_ = jnp.asarray(mobile)
    reference_ = jnp.asarray(reference)
    if mobile_.ndim != 2 or mobile_.shape[-1] != 3 or reference_.shape != mobile_.shape:
        raise ValueError(
            "mobile and reference must have identical shape (point_count, 3)."
        )
    if not jnp.issubdtype(mobile_.dtype, jnp.inexact):
        mobile_ = mobile_.astype(jnp.float64)
    reference_ = reference_.astype(mobile_.dtype)
    count = mobile_.shape[0]
    weight = (
        jnp.ones((count,), dtype=mobile_.dtype)
        if weights is None
        else jnp.asarray(weights, dtype=mobile_.dtype)
    )
    active = (
        jnp.ones((count,), dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    )
    if weight.shape != (count,) or active.shape != (count,):
        raise ValueError("weights and mask must align with the point dimension.")
    finite = (
        jnp.all(jnp.isfinite(mobile_), axis=-1)
        & jnp.all(jnp.isfinite(reference_), axis=-1)
        & jnp.isfinite(weight)
    )
    active = active & finite & (weight > 0.0)
    effective = jnp.where(active, weight, 0.0)
    total = jnp.sum(effective)
    safe_total = jnp.maximum(total, jnp.asarray(1.0, dtype=mobile_.dtype))
    mobile_center = jnp.sum(effective[:, None] * mobile_, axis=0) / safe_total
    reference_center = jnp.sum(effective[:, None] * reference_, axis=0) / safe_total
    centered_mobile = mobile_ - mobile_center
    centered_reference = reference_ - reference_center
    covariance = contract("n,ni,nj->ij", effective, centered_mobile, centered_reference)
    left, singular, right_t = jnp.linalg.svd(covariance, full_matrices=False)
    orientation = jnp.linalg.det(right_t.T @ left.T)
    correction = (
        jnp.eye(3, dtype=mobile_.dtype)
        .at[2, 2]
        .set(jnp.where(orientation < 0.0, -1.0, 1.0))
    )
    rotation = right_t.T @ correction @ left.T
    translation = reference_center - mobile_center @ rotation.T
    aligned = mobile_ @ rotation.T + translation
    residuals = jnp.sqrt(jnp.sum((aligned - reference_) ** 2, axis=-1))
    rmsd = jnp.sqrt(jnp.sum(effective * residuals**2) / safe_total)
    active_count = jnp.sum(active, dtype=jnp.int32)
    scale = jnp.maximum(singular[0], jnp.asarray(1.0, dtype=singular.dtype))
    rank = jnp.sum(
        singular > scale * jnp.finfo(singular.dtype).eps * 16.0, dtype=jnp.int32
    )
    nonfinite = jnp.any(~finite & (active | (weight > 0.0)))
    status = jnp.where(
        nonfinite,
        int(AlignmentStatus.NONFINITE),
        jnp.where(
            active_count < 3,
            int(AlignmentStatus.INSUFFICIENT_POINTS),
            jnp.where(
                rank < 2, int(AlignmentStatus.DEGENERATE), int(AlignmentStatus.SUCCESS)
            ),
        ),
    ).astype(jnp.int32)
    valid = status == int(AlignmentStatus.SUCCESS)
    safe_rotation = jnp.where(valid, rotation, jnp.eye(3, dtype=rotation.dtype))
    safe_translation = jnp.where(
        valid, translation, jnp.zeros((3,), dtype=translation.dtype)
    )
    safe_aligned = jnp.where(valid, aligned, mobile_)
    safe_residuals = jnp.where(valid, residuals, jnp.inf)
    safe_rmsd = jnp.where(valid, rmsd, jnp.inf)
    evidence = jnp.concatenate((active_count.astype(singular.dtype)[None], singular))
    return RigidAlignmentResult(
        safe_rotation,
        safe_translation,
        safe_aligned,
        safe_residuals,
        safe_rmsd,
        valid,
        status,
        evidence,
        _alignment_contract(np.dtype(mobile_.dtype).name),
    )


def align_structure_models(
    structure: MacromolecularStructure,
    mobile_model_index: int,
    reference_model_index: int,
    /,
) -> RigidAlignmentResult:
    """Align two coordinate models using atoms present in both coupled conformers."""

    if not isinstance(structure, MacromolecularStructure):
        raise TypeError("structure must be a MacromolecularStructure.")
    if (
        not 0 <= mobile_model_index < structure.model_capacity
        or not 0 <= reference_model_index < structure.model_capacity
    ):
        raise IndexError("Model index is outside the compiled capacity.")
    mask = structure.altloc_mask(mobile_model_index) & structure.altloc_mask(
        reference_model_index
    )
    weights = jnp.minimum(
        structure.occupancies[mobile_model_index],
        structure.occupancies[reference_model_index],
    )
    return align_coordinates(
        structure.positions[mobile_model_index],
        structure.positions[reference_model_index],
        weights=weights,
        mask=mask,
    )


__all__ = ["RigidAlignmentResult", "align_coordinates", "align_structure_models"]
