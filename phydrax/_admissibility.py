#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntFlag
from functools import partial
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState
from ._validation import canonical_identifier


class AdmissibilityReason(IntFlag):
    """Reason bits shared by every fail-closed model-admission contract."""

    NONE = 0
    NONFINITE = 1 << 0
    OUTSIDE_SUPPORT = 1 << 1
    UNCERTAINTY_UNRESOLVED = 1 << 2
    CAPACITY_INSUFFICIENT = 1 << 3
    TRANSITION_NOT_ACCEPTED_BOUNDARY = 1 << 4


DOMAIN_REASON_SHIFT = 8


DerivativeFailureMode: TypeAlias = Literal["status", "error"]


@partial(jax.custom_jvp, nondiff_argnums=(3, 4))
def _guard_derivative_leaf(
    value: Array,
    dependency: Array,
    valid: Array,
    failure: DerivativeFailureMode,
    message: str,
    /,
) -> Array:
    del dependency, valid, failure, message
    return value


@_guard_derivative_leaf.defjvp
def _guard_derivative_leaf_jvp(failure, message, primals, tangents):
    value, _, valid = primals
    value_tangent, dependency_tangent, _ = tangents
    valid_ = jnp.asarray(valid, dtype=jnp.bool_)
    if failure == "error":
        value_tangent = eqx.error_if(
            value_tangent,
            ~jnp.all(valid_),
            message,
        )
    else:
        value_scale = jnp.where(
            valid_,
            jnp.asarray(1.0, dtype=value_tangent.dtype),
            jnp.asarray(jnp.nan, dtype=value_tangent.dtype),
        )
        dependency_scale = jnp.where(
            valid_,
            jnp.asarray(0.0, dtype=value_tangent.dtype),
            jnp.asarray(jnp.nan, dtype=value_tangent.dtype),
        )
        value_tangent = value_scale * value_tangent + dependency_scale * jnp.asarray(
            dependency_tangent, dtype=value_tangent.dtype
        )
    return value, value_tangent


def guard_derivative_validity(
    tree: Any,
    valid: ArrayLike,
    /,
    *,
    dependencies: Any = (),
    failure: DerivativeFailureMode = "status",
    message: str = "Derivative is invalid for the accepted primal result.",
) -> Any:
    """Keep a primal inspectable while poisoning or rejecting invalid derivatives."""
    if failure not in ("status", "error"):
        raise ValueError("Derivative failure mode must be 'status' or 'error'.")
    message_ = str(message).strip()
    if not message_:
        raise ValueError("Derivative failure message must be non-empty.")
    valid_ = jnp.asarray(valid, dtype=jnp.bool_)
    dependency_leaves = tuple(
        leaf for leaf in jax.tree.leaves(dependencies) if eqx.is_inexact_array(leaf)
    )
    dependency = sum(
        (jnp.sum(jnp.real(leaf)) for leaf in dependency_leaves),
        start=jnp.asarray(0.0),
    )
    return jax.tree.map(
        lambda leaf: (
            _guard_derivative_leaf(
                leaf,
                dependency,
                valid_,
                failure,
                message_,
            )
            if eqx.is_inexact_array(leaf)
            else leaf
        ),
        tree,
    )


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
        self.model_id = canonical_identifier(model_id, "model_id")
        self.evidence_id = canonical_identifier(evidence_id, "evidence_id")

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
        region = jnp.asarray(region_mask, dtype=jnp.bool_)
        epoch = jnp.asarray(requested_epoch, dtype=jnp.int32)
        if epoch.shape != ():
            raise ValueError("requested_epoch must be scalar.")
        if region.shape != evidence.eligible.shape:
            raise ValueError("Transition region and evidence must share a shape.")
        current = canonical_identifier(current_model_id, "current_model_id")
        target = canonical_identifier(target_model_id, "target_model_id")
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

    value = jnp.asarray(predicate, dtype=jnp.bool_)
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
                "model": canonical_identifier(model_id, "model_id"),
                "evidence": tuple(value.evidence_id for value in values),
            }
        )
        if evidence_id is None
        else canonical_identifier(evidence_id, "evidence_id")
    )
    return AdmissibilityHeader(margin, reasons, model_id, identity)


__all__ = [
    "DOMAIN_REASON_SHIFT",
    "AdmissibilityHeader",
    "AdmissibilityReason",
    "AdmissibilityTransitionRequest",
    "DerivativeFailureMode",
    "combine_admissibility",
    "guard_derivative_validity",
    "reason_bits_where",
]
