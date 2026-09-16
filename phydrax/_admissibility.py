#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, IntFlag

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


class AdmissibilityReason(IntFlag):
    """Reason bits shared by every fail-closed model-admission contract."""

    NONE = 0
    NONFINITE = 1 << 0
    OUTSIDE_SUPPORT = 1 << 1
    UNCERTAINTY_UNRESOLVED = 1 << 2
    CAPACITY_INSUFFICIENT = 1 << 3
    TRANSITION_NOT_ACCEPTED_BOUNDARY = 1 << 4


DOMAIN_REASON_SHIFT = 8


class DerivativeAvailability(str, Enum):
    """Derivative semantics promised by one fixed physical model."""

    NONE = "none"
    WITHIN_FIXED_MODEL = "within-fixed-model"
    ALGORITHMIC_FIXED_MODEL = "algorithmic-fixed-model"
    IMPLICIT_FIXED_MODEL = "implicit-fixed-model"


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier or identifier != identifier.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return identifier


class AdmissibilityHeader(StrictModule, NonTrainableState):
    """Minimal JAX-native evidence shared by independently owned models.

    ``margin`` is nonnegative inside the declared support. Nonfinite margins are
    normalized to ``-inf`` and receive the common nonfinite reason bit. Runtime
    values remain dynamic leaves; the static identifiers name the physical model
    and the evidence policy rather than a particular numerical evaluation.
    """

    eligible: Array
    margin: Array
    reason_bits: Array
    model_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        margin: ArrayLike,
        reason_bits: ArrayLike,
        model_id: str,
        evidence_id: str,
        /,
    ) -> None:
        margin_ = jnp.asarray(margin)
        if not jnp.issubdtype(margin_.dtype, jnp.inexact):
            margin_ = margin_.astype(jnp.float64)
        reasons = jnp.asarray(reason_bits, dtype=jnp.uint32)
        if reasons.shape != margin_.shape:
            raise ValueError("Admissibility margin and reason bits must share a shape.")
        finite = jnp.isfinite(margin_)
        normalized_margin = jnp.where(finite, margin_, -jnp.inf)
        normalized_reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), dtype=jnp.uint32),
        )
        self.eligible = (normalized_margin >= 0.0) & (normalized_reasons == 0)
        self.margin = normalized_margin
        self.reason_bits = normalized_reasons
        self.model_id = _identifier(model_id, "model_id")
        self.evidence_id = _identifier(evidence_id, "evidence_id")

    @property
    def globally_eligible(self) -> Array:
        """Return the scalar conjunction over every represented region."""

        return jnp.all(self.eligible)


class AdmissibilityTransitionRequest(StrictModule, NonTrainableState):
    """Pure request for a host-authorized model transition between accepted steps."""

    region_mask: Array
    requested_epoch: Array
    evidence: AdmissibilityHeader
    current_model_id: str = eqx.field(static=True)
    target_model_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        region_mask: ArrayLike,
        requested_epoch: ArrayLike,
        evidence: AdmissibilityHeader,
        current_model_id: str,
        target_model_id: str,
        /,
    ) -> None:
        if not isinstance(evidence, AdmissibilityHeader):
            raise TypeError("evidence must be AdmissibilityHeader.")
        region = jnp.asarray(region_mask, dtype=bool)
        epoch = jnp.asarray(requested_epoch, dtype=jnp.int32)
        if epoch.shape != ():
            raise ValueError("requested_epoch must be scalar.")
        if region.shape != evidence.eligible.shape:
            raise ValueError("Transition region and evidence must share a shape.")
        current = _identifier(current_model_id, "current_model_id")
        target = _identifier(target_model_id, "target_model_id")
        if current == target:
            raise ValueError("A transition request must change the physical model.")
        self.region_mask = region
        self.requested_epoch = epoch
        self.evidence = evidence
        self.current_model_id = current
        self.target_model_id = target
        self.request_id = canonical_fingerprint(
            {
                "kind": "admissibility-transition-request",
                "current_model": current,
                "target_model": target,
                "evidence": evidence.evidence_id,
            }
        )


def reason_bits_where(
    predicate: ArrayLike,
    reason: AdmissibilityReason | int,
    /,
) -> Array:
    """Encode ``reason`` where a required predicate is false."""

    value = jnp.asarray(predicate, dtype=bool)
    return jnp.where(
        value,
        jnp.asarray(0, dtype=jnp.uint32),
        jnp.asarray(int(reason), dtype=jnp.uint32),
    )


def combine_admissibility(
    evidence: Sequence[AdmissibilityHeader],
    model_id: str,
    /,
    *,
    evidence_id: str | None = None,
) -> AdmissibilityHeader:
    """Conjoin compatible evidence without erasing margins or failure reasons."""

    values = tuple(evidence)
    if not values:
        raise ValueError("At least one admissibility header is required.")
    if any(not isinstance(value, AdmissibilityHeader) for value in values):
        raise TypeError("Every evidence item must be AdmissibilityHeader.")
    shape = values[0].margin.shape
    if any(value.margin.shape != shape for value in values[1:]):
        raise ValueError("Combined admissibility headers must share a shape.")
    margin = jnp.minimum.reduce(jnp.stack(tuple(value.margin for value in values)))
    reasons = jnp.bitwise_or.reduce(
        jnp.stack(tuple(value.reason_bits for value in values)), axis=0
    )
    identity = (
        canonical_fingerprint(
            {
                "kind": "combined-admissibility",
                "model": _identifier(model_id, "model_id"),
                "evidence": tuple(value.evidence_id for value in values),
            }
        )
        if evidence_id is None
        else _identifier(evidence_id, "evidence_id")
    )
    return AdmissibilityHeader(margin, reasons, model_id, identity)


__all__ = [
    "DOMAIN_REASON_SHIFT",
    "AdmissibilityHeader",
    "AdmissibilityReason",
    "AdmissibilityTransitionRequest",
    "DerivativeAvailability",
    "combine_admissibility",
    "reason_bits_where",
]
