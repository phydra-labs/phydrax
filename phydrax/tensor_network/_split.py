#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._precision import TensorNetworkPrecisionPolicy


SingularValueAbsorption: TypeAlias = Literal["left", "right", "split"]


class TensorTruncationEvidence(StrictModule):
    """Fixed-capacity SVD rank and discarded squared-weight evidence."""

    retained_rank: int = eqx.field(static=True)
    available_rank: int = eqx.field(static=True)
    discarded_weight: Array
    valid: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        retained_rank: int,
        available_rank: int,
        discarded_weight: ArrayLike,
        /,
        precision_evidence: PrecisionEvidenceEnvelope,
        precision_policy_id: str,
    ):
        retained = int(retained_rank)
        available = int(available_rank)
        if available < 1 or not 1 <= retained <= available:
            raise ValueError("Truncation ranks must satisfy 1 <= retained <= available.")
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        discarded = jnp.asarray(discarded_weight)
        if discarded.shape != ():
            raise ValueError("discarded_weight must be scalar.")
        self.retained_rank = retained
        self.available_rank = available
        self.discarded_weight = discarded
        self.valid = jnp.isfinite(discarded) & (discarded >= 0.0)
        self.precision_evidence = precision_evidence
        self.precision_policy_id = str(precision_policy_id)


def truncated_svd(
    matrix: ArrayLike,
    /,
    *,
    maximum_rank: int,
    absorb: SingularValueAbsorption,
    precision: TensorNetworkPrecisionPolicy,
    evidence_source: Any,
    evidence_children: dict[str, PrecisionEvidenceEnvelope] | None = None,
) -> tuple[Array, Array, TensorTruncationEvidence]:
    """Split one matrix at a static maximum rank and cast factors to storage."""

    if not isinstance(precision, TensorNetworkPrecisionPolicy):
        raise TypeError("precision must be TensorNetworkPrecisionPolicy.")
    capacity = int(maximum_rank)
    if capacity < 1:
        raise ValueError("maximum_rank must be positive.")
    if absorb not in ("left", "right", "split"):
        raise ValueError("absorb must be 'left', 'right', or 'split'.")
    value = precision.factorization(jnp.asarray(matrix))
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("truncated_svd requires a nonempty rank-two matrix.")

    u, singular_values, vh = jnp.linalg.svd(value, full_matrices=False)
    available = int(singular_values.shape[0])
    retained = min(capacity, available)
    discarded = precision.decision(
        precision.sum(jnp.abs(singular_values[retained:]) ** 2)
    )
    u = u[:, :retained]
    singular_values = singular_values[:retained]
    vh = vh[:retained, :]

    if absorb == "left":
        left = u * singular_values[None, :]
        right = vh
    elif absorb == "right":
        left = u
        right = singular_values[:, None] * vh
    else:
        root = jnp.sqrt(singular_values)
        left = u * root[None, :]
        right = root[:, None] * vh

    left = precision.storage(left)
    right = precision.storage(right)
    evidence = precision.evidence_for(
        evidence_source,
        children={} if evidence_children is None else evidence_children,
        output_value=(left, right),
    )
    return (
        left,
        right,
        TensorTruncationEvidence(
            retained,
            available,
            discarded,
            evidence,
            precision.policy_id,
        ),
    )


__all__ = ["SingularValueAbsorption", "TensorTruncationEvidence", "truncated_svd"]
