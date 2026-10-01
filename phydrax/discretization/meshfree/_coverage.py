# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Train-fitted marginal and joint feature support, without covariance repair."""

from __future__ import annotations

from enum import IntEnum
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DensePropertyEvidence,
    DensePropertyVerificationPolicy,
    FactorizationPolicy,
    factorize,
    PreparedFactorization,
    RHSLayout,
    verify_dense_properties,
)
from ...typing import AnyShape, Bool, Dim, Float64, Int32, parse, Scalar


EdgeCoveragePolicy: TypeAlias = Literal["refuse", "report"]


class _CoverageEdgeDim(Dim):
    """Queried edges of one coverage assessment."""


class _CoverageFeatureDim(Dim):
    """Jointly fitted feature components."""


class EdgeCoverageStatus(IntEnum):
    COVERED = 0
    MARGINAL_OUTSIDE = 1
    JOINT_OUTSIDE = 2
    NONFINITE = 3
    COVARIANCE_SOLVE_FAILED = 4


@final
class EdgeCoverageAssessment(StrictModule, NonTrainableState):
    __strict_contract__ = True
    covered: Bool[_CoverageEdgeDim]
    status: Int32[_CoverageEdgeDim]
    marginal_covered: Bool[_CoverageEdgeDim]
    joint_covered: Bool[_CoverageEdgeDim]
    mahalanobis_squared: Float64[_CoverageEdgeDim]
    covariance_solve_status: Int32[AnyShape]
    admitted: Bool[Scalar]
    policy: EdgeCoveragePolicy = eqx.field(static=True)


@final
class EdgeFeatureCoverage(StrictModule, NonTrainableState):
    """Quantile box intersected with a train-fitted Mahalanobis support domain.

    Existing ML covariance recipes own regularized/floored statistical estimates;
    this admission owner instead requires the actual empirical covariance to be
    positive definite. Its property evidence and reusable Cholesky solve are native
    linalg objects. Collinear training data are refused rather than repaired.
    ``report`` permits an uncovered prediction but still reports the refusal
    evidence; ``refuse`` makes ``assessment.admitted`` false. Neither clips input.
    """

    __strict_contract__ = True
    lower: Float64[_CoverageFeatureDim]
    upper: Float64[_CoverageFeatureDim]
    mean: Float64[_CoverageFeatureDim]
    covariance_evidence: DensePropertyEvidence
    covariance_factor: PreparedFactorization
    maximum_mahalanobis_squared: Float64[Scalar]
    feature_names: tuple[str, ...] = eqx.field(static=True)
    training_domain: str = eqx.field(static=True)
    lower_quantile: float = eqx.field(static=True)
    upper_quantile: float = eqx.field(static=True)
    joint_quantile: float = eqx.field(static=True)
    policy: EdgeCoveragePolicy = eqx.field(static=True)

    @classmethod
    def fit(
        cls,
        samples: ArrayLike,
        /,
        *,
        training_domain: str,
        feature_names: tuple[str, ...] | None = None,
        lower_quantile: float = 0.0,
        upper_quantile: float = 1.0,
        joint_quantile: float = 1.0,
        policy: EdgeCoveragePolicy = "refuse",
    ) -> EdgeFeatureCoverage:
        values = np.asarray(samples)
        if values.ndim != 2 or values.shape[1] == 0 or values.shape[0] <= values.shape[1]:
            raise ValueError(
                "Joint coverage fitting requires more finite training rows than nonzero feature columns."
            )
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError("Coverage fitting requires finite real training features.")
        if not 0 <= lower_quantile < upper_quantile <= 1 or not 0 < joint_quantile <= 1:
            raise ValueError(
                "Coverage quantiles must specify a nonempty marginal interval and positive joint quantile."
            )
        if (
            not isinstance(training_domain, str)
            or not training_domain
            or training_domain.strip() != training_domain
        ):
            raise ValueError("Coverage requires a named, train-only fitted domain.")
        names = (
            tuple(f"feature-{index}" for index in range(values.shape[1]))
            if feature_names is None
            else feature_names
        )
        if (
            len(names) != values.shape[1]
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError("Coverage feature names must uniquely name each column.")
        array = jnp.asarray(values, dtype=jnp.float64)
        mean = jnp.mean(array, axis=0)
        centered = array - mean
        covariance = centered.T @ centered / (array.shape[0] - 1)
        evidence = verify_dense_properties(
            covariance,
            policy=DensePropertyVerificationPolicy(require_positive_definite=True),
        )
        if not bool(evidence.successful):
            raise ValueError(
                "Training feature covariance is not positive definite; no ridge, inverse, or eigenvalue repair is applied."
            )
        factor = factorize(
            DenseLinearOperator(evidence.matrix, properties=evidence.properties),
            FactorizationPolicy("cholesky"),
        )
        distances = factor.solve(centered.T, rhs_layout=RHSLayout((array.shape[0],)))
        if not bool(jnp.all(distances.successful)):
            raise ValueError(
                "Native covariance factorization failed during coverage fitting."
            )
        squared = jnp.sum(centered * distances.value.T, axis=-1)
        if not bool(jnp.all(jnp.isfinite(squared) & (squared >= 0))):
            raise ValueError(
                "Native covariance solve did not yield valid squared distances."
            )
        return cls(
            lower=jnp.quantile(array, lower_quantile, axis=0),
            upper=jnp.quantile(array, upper_quantile, axis=0),
            mean=mean,
            covariance_evidence=evidence,
            covariance_factor=factor,
            maximum_mahalanobis_squared=jnp.quantile(squared, joint_quantile),
            feature_names=names,
            training_domain=training_domain,
            lower_quantile=float(lower_quantile),
            upper_quantile=float(upper_quantile),
            joint_quantile=float(joint_quantile),
            policy=parse(policy, EdgeCoveragePolicy, "policy"),
        )

    def assess(self, features: ArrayLike, /) -> EdgeCoverageAssessment:
        values = jnp.asarray(features, dtype=self.mean.dtype)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError(
                "Coverage queries must use the fitted feature-column layout."
            )
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        centered = jnp.where(jnp.isfinite(values), values, self.mean) - self.mean
        solved = self.covariance_factor.solve(
            centered.T, rhs_layout=RHSLayout((values.shape[0],))
        )
        squared = jnp.sum(centered * solved.value.T, axis=-1)
        solve_valid = jnp.all(solved.successful) & jnp.isfinite(squared) & (squared >= 0)
        # Relative roundoff allowance is a comparison tolerance, not an enlargement
        # of the fitted covariance or a replacement for an invalid solve.
        tolerance = 64 * jnp.finfo(values.dtype).eps
        marginal = jnp.all(
            (values >= self.lower - tolerance * jnp.maximum(1, jnp.abs(self.lower)))
            & (values <= self.upper + tolerance * jnp.maximum(1, jnp.abs(self.upper))),
            axis=-1,
        )
        joint = squared <= self.maximum_mahalanobis_squared * (1 + tolerance) + tolerance
        covered = finite & solve_valid & marginal & joint
        status = jnp.where(
            ~finite,
            int(EdgeCoverageStatus.NONFINITE),
            jnp.where(
                ~solve_valid,
                int(EdgeCoverageStatus.COVARIANCE_SOLVE_FAILED),
                jnp.where(
                    ~marginal,
                    int(EdgeCoverageStatus.MARGINAL_OUTSIDE),
                    jnp.where(
                        ~joint,
                        int(EdgeCoverageStatus.JOINT_OUTSIDE),
                        int(EdgeCoverageStatus.COVERED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        match self.policy:
            case "refuse":
                admitted = jnp.all(covered)
            case "report":
                admitted = jnp.all(finite & solve_valid)
        return EdgeCoverageAssessment(
            covered,
            status,
            marginal,
            joint,
            squared,
            solved.status,
            admitted,
            self.policy,
        )


__all__ = [
    "EdgeCoverageAssessment",
    "EdgeCoveragePolicy",
    "EdgeCoverageStatus",
    "EdgeFeatureCoverage",
]
