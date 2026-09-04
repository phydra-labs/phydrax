#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class FiberArchitectureEvidence(StrictModule, NonTrainableState):
    """Reference-fiber normalization and support evidence."""

    supplied_norm: Array
    norm_error: Array
    finite: Array
    supported: Array
    valid: Array
    tolerance: float = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)


class PreparedUniformFiberArchitecture(StrictModule, NonTrainableState):
    """One uniform, sign-indifferent reference fiber family.

    The direction is a material/reference vector.  ``m`` and ``-m`` define the
    same structural tensor, which is the only orientation object consumed by
    the constitutive law.
    """

    reference_direction: Array
    structural_tensor: Array
    support: Array
    evidence: FiberArchitectureEvidence
    architecture_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def evaluate(self, /) -> FiberArchitectureEvidence:
        """Return the immutable preparation evidence."""
        return self.evidence


class UniformFiberArchitecturePlan(StrictModule, NonTrainableState):
    """Static plan for a uniform three-dimensional reference fiber family."""

    normalization_tolerance: float = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        architecture_id: str,
        /,
        *,
        normalization_tolerance: float = 1.0e-6,
    ):
        identifier = str(architecture_id).strip()
        tolerance = float(normalization_tolerance)
        if not identifier:
            raise ValueError("architecture_id must be non-empty.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("normalization_tolerance must be positive and finite.")
        self.normalization_tolerance = tolerance
        self.architecture_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "uniform-skeletal-reference-fiber-plan",
                "architecture_id": identifier,
                "dimension": 3,
                "normalization_tolerance": tolerance.hex(),
            }
        )

    def prepare(
        self,
        reference_direction: ArrayLike,
        /,
        *,
        support: ArrayLike = True,
    ) -> PreparedUniformFiberArchitecture:
        """Normalize one direction and fail closed when its support is absent."""
        direction = jnp.asarray(reference_direction)
        if direction.shape != (3,):
            raise ValueError("reference_direction must have shape (3,).")
        if jnp.issubdtype(direction.dtype, jnp.complexfloating):
            raise ValueError("reference_direction must be real-valued.")
        support_ = jnp.asarray(support, dtype=bool)
        if support_.shape != ():
            raise ValueError("Uniform fiber support must be one scalar mask.")
        norm = jnp.linalg.norm(direction)
        finite = jnp.all(jnp.isfinite(direction)) & jnp.isfinite(norm)
        nonzero = finite & (norm > 0.0)
        safe_norm = jnp.where(nonzero, norm, 1.0)
        normalized = direction / safe_norm
        error = jnp.abs(jnp.linalg.norm(normalized) - 1.0)
        valid = nonzero & support_ & (error <= self.normalization_tolerance)
        normalized = jnp.where(valid, normalized, jnp.full_like(normalized, jnp.nan))
        structural = normalized[:, None] * normalized[None, :]
        evidence = FiberArchitectureEvidence(
            norm,
            error,
            finite,
            support_,
            valid,
            self.normalization_tolerance,
            self.architecture_id,
        )
        return PreparedUniformFiberArchitecture(
            normalized,
            structural,
            support_,
            evidence,
            self.architecture_id,
            canonical_fingerprint(
                {
                    "kind": "prepared-uniform-skeletal-reference-fiber",
                    "plan": self.plan_id,
                    "direction": array_tree_fingerprint(direction),
                    "support": bool(support_),
                }
            ),
        )


__all__ = [
    "FiberArchitectureEvidence",
    "PreparedUniformFiberArchitecture",
    "UniformFiberArchitecturePlan",
]
