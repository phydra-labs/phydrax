#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..typing import checked
from ._ir import ConditionQuantifier


def _identifier(value: str, name: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _scalar(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.shape:
        raise ValueError(f"{name} must be scalar, got shape {result.shape}.")
    return result


class ConditionRealizationStamp(StrictModule):
    """Identity and exactness metadata shared by all condition realizations."""

    condition_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    quantifier: ConditionQuantifier = eqx.field(static=True)
    exact: bool = eqx.field(static=True)

    def __init__(
        self,
        condition_id: str,
        source_id: str,
        realization_id: str,
        provider_id: str,
        /,
        *,
        quantifier: ConditionQuantifier,
        exact: bool,
    ) -> None:
        self.condition_id = _identifier(condition_id, "condition_id")
        self.source_id = _identifier(source_id, "source_id")
        self.realization_id = _identifier(realization_id, "realization_id")
        self.provider_id = _identifier(provider_id, "provider_id")
        self.quantifier = ConditionQuantifier(quantifier)
        self.exact = bool(exact)


class ConditionEvidence(StrictModule):
    """Numerical evidence for one stamped condition realization."""

    stamp: ConditionRealizationStamp
    evidence_id: str = eqx.field(static=True)
    residual_norm: Array
    satisfied: Array

    @checked
    def __init__(
        self,
        stamp: ConditionRealizationStamp,
        residual_norm: ArrayLike,
        satisfied: ArrayLike,
        /,
        *,
        evidence_id: str,
    ) -> None:
        satisfied_ = _scalar(satisfied, "satisfied")
        if satisfied_.dtype != jnp.dtype(jnp.bool_):
            raise TypeError("ConditionEvidence.satisfied must have boolean dtype.")
        self.stamp = stamp
        self.evidence_id = _identifier(evidence_id, "evidence_id")
        self.residual_norm = _scalar(residual_norm, "residual_norm")
        self.satisfied = satisfied_


class ConditionCertificate(StrictModule):
    """Base contract for independently checkable condition certificates."""

    __strict_abstract__ = True
    stamp: eqx.AbstractVar[ConditionRealizationStamp]
    certificate_id: eqx.AbstractVar[str]
    residual_norm: eqx.AbstractVar[Array]
    tolerance: eqx.AbstractVar[Array]
    verified: eqx.AbstractVar[Array]


class AffineProjectionCertificate(ConditionCertificate):
    stamp: ConditionRealizationStamp
    certificate_id: str = eqx.field(static=True)
    residual_norm: Array
    tolerance: Array
    verified: Array
    rank: int = eqx.field(static=True)
    nullity: int = eqx.field(static=True)

    def __init__(
        self,
        stamp: ConditionRealizationStamp,
        residual_norm: ArrayLike,
        tolerance: ArrayLike,
        verified: ArrayLike,
        /,
        *,
        certificate_id: str,
        rank: int,
        nullity: int,
    ) -> None:
        rank_ = int(rank)
        nullity_ = int(nullity)
        if rank_ < 0 or nullity_ < 0:
            raise ValueError("Affine ranks and nullities must be nonnegative.")
        self.stamp = stamp
        self.certificate_id = _identifier(certificate_id, "certificate_id")
        self.residual_norm = _scalar(residual_norm, "residual_norm")
        self.tolerance = _scalar(tolerance, "tolerance")
        self.verified = _scalar(verified, "verified")
        self.rank = rank_
        self.nullity = nullity_


class NonlinearRetractionCertificate(ConditionCertificate):
    stamp: ConditionRealizationStamp
    certificate_id: str = eqx.field(static=True)
    residual_norm: Array
    tolerance: Array
    verified: Array
    iterations: int = eqx.field(static=True)

    def __init__(
        self,
        stamp: ConditionRealizationStamp,
        residual_norm: ArrayLike,
        tolerance: ArrayLike,
        verified: ArrayLike,
        /,
        *,
        certificate_id: str,
        iterations: int,
    ) -> None:
        iterations_ = int(iterations)
        if iterations_ < 0:
            raise ValueError("Nonlinear retraction iterations must be nonnegative.")
        self.stamp = stamp
        self.certificate_id = _identifier(certificate_id, "certificate_id")
        self.residual_norm = _scalar(residual_norm, "residual_norm")
        self.tolerance = _scalar(tolerance, "tolerance")
        self.verified = _scalar(verified, "verified")
        self.iterations = iterations_


class FeasibilityCertificate(ConditionCertificate):
    stamp: ConditionRealizationStamp
    certificate_id: str = eqx.field(static=True)
    residual_norm: Array
    tolerance: Array
    verified: Array
    maximum_violation: Array

    def __init__(
        self,
        stamp: ConditionRealizationStamp,
        residual_norm: ArrayLike,
        tolerance: ArrayLike,
        verified: ArrayLike,
        maximum_violation: ArrayLike,
        /,
        *,
        certificate_id: str,
    ) -> None:
        self.stamp = stamp
        self.certificate_id = _identifier(certificate_id, "certificate_id")
        self.residual_norm = _scalar(residual_norm, "residual_norm")
        self.tolerance = _scalar(tolerance, "tolerance")
        self.verified = _scalar(verified, "verified")
        self.maximum_violation = _scalar(maximum_violation, "maximum_violation")


class ProbabilisticConditioningEvidence(StrictModule):
    """Evidence from conditioning a probabilistic source on one condition."""

    stamp: ConditionRealizationStamp
    evidence_id: str = eqx.field(static=True)
    log_normalizer: Array
    effective_sample_size: Array
    accepted_probability: Array

    @checked
    def __init__(
        self,
        stamp: ConditionRealizationStamp,
        log_normalizer: ArrayLike,
        effective_sample_size: ArrayLike,
        accepted_probability: ArrayLike,
        /,
        *,
        evidence_id: str,
    ) -> None:
        self.stamp = stamp
        self.evidence_id = _identifier(evidence_id, "evidence_id")
        self.log_normalizer = _scalar(log_normalizer, "log_normalizer")
        self.effective_sample_size = _scalar(
            effective_sample_size, "effective_sample_size"
        )
        self.accepted_probability = _scalar(accepted_probability, "accepted_probability")


__all__ = [
    "AffineProjectionCertificate",
    "ConditionCertificate",
    "ConditionEvidence",
    "ConditionRealizationStamp",
    "FeasibilityCertificate",
    "NonlinearRetractionCertificate",
    "ProbabilisticConditioningEvidence",
]
