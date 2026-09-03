#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from numbers import Integral
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..conditions._evidence import (
    ConditionRealizationStamp,
    ProbabilisticConditioningEvidence,
)
from ..conditions._ir import ArrayCodomain, ConditionQuantifier
from ..conditions._lowering import BoundCondition
from ..conditions._relations import Equality, NoisyObservation
from ..linalg._constraint_operators import PreparedConstraintOperator
from ._conditional_moments import condition_gaussian, ConditionalGaussianMoments
from ._factor_law import GaussianFactorLaw
from ._gaussian_factor import (
    add_independent_gaussian_factors,
    gaussian_factor_from_covariance,
    GaussianFactor,
)
from ._nonlinear_gaussian import (
    first_order_gaussian_transform,
    gauss_hermite_transform,
    scaled_unscented_transform,
    spherical_radial_cubature,
)


ConstraintApproximation = Literal[
    "exact-linear", "first-order", "cubature", "unscented", "gauss-hermite"
]
ConstraintConditioningStatus = Literal[0, 1, 2, 3]
CONSTRAINT_CONDITIONING_SUCCESS: ConstraintConditioningStatus = 0
CONSTRAINT_CONDITIONING_INVALID_FACTOR: ConstraintConditioningStatus = 1
CONSTRAINT_CONDITIONING_INCONSISTENT_SUPPORT: ConstraintConditioningStatus = 2
CONSTRAINT_CONDITIONING_NONFINITE: ConstraintConditioningStatus = 3


class ConstraintLikelihoodTerm(StrictModule):
    """One normalized finite Gaussian condition likelihood.

    Physical observation noise is represented by a covariance root, so correlated
    and singular laws retain their support.  Numerical jitter is deliberately not
    part of this term and can only be supplied to a conditioner.
    """

    observed: Array
    physical_noise: GaussianFactor
    stamp: ConditionRealizationStamp
    support_tolerance: Array
    likelihood_id: str = eqx.field(static=True)
    bound: BoundCondition | None

    def __init__(
        self,
        observed: ArrayLike,
        /,
        *,
        noise_scale: ArrayLike | None = None,
        noise_covariance: ArrayLike | None = None,
        noise_factor: GaussianFactor | None = None,
        support_tolerance: ArrayLike = 1e-8,
        rank_tolerance: ArrayLike = 0.0,
        likelihood_id: str = "constraint-likelihood",
        stamp: ConditionRealizationStamp | None = None,
        bound: BoundCondition | None = None,
    ):
        raw_value = jnp.asarray(observed)
        if jnp.iscomplexobj(raw_value):
            raise TypeError("observed must be real-valued.")
        value = raw_value.astype(float)
        if value.ndim != 1 or value.size <= 0:
            raise ValueError("observed must be a nonempty flat vector.")
        value = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)),
            "observed must be finite.",
        )
        supplied = sum(
            item is not None for item in (noise_scale, noise_covariance, noise_factor)
        )
        if supplied > 1:
            raise ValueError(
                "Supply only one of noise_scale, noise_covariance, or noise_factor."
            )
        tolerance = jnp.asarray(support_tolerance, dtype=value.dtype)
        if tolerance.ndim != 0:
            raise ValueError("support_tolerance must be scalar.")
        rank = jnp.asarray(rank_tolerance, dtype=value.dtype)
        if rank.ndim != 0:
            raise ValueError("rank_tolerance must be scalar.")
        tolerance = eqx.error_if(
            tolerance,
            ~jnp.isfinite(tolerance) | (tolerance < 0.0),
            "support_tolerance must be finite and nonnegative.",
        )
        rank = eqx.error_if(
            rank,
            ~jnp.isfinite(rank) | (rank < 0.0),
            "rank_tolerance must be finite and nonnegative.",
        )
        if noise_factor is not None:
            if not isinstance(noise_factor, GaussianFactor):
                raise TypeError("noise_factor must be a GaussianFactor.")
            if jnp.iscomplexobj(noise_factor.factor):
                raise TypeError("noise_factor must be real-valued.")
            if noise_factor.factor.ndim != 2:
                raise ValueError("noise_factor must describe one unbatched event.")
            if noise_factor.event_size != value.size:
                raise ValueError("noise_factor must match the observed event size.")
        elif noise_covariance is not None:
            raw_covariance = jnp.asarray(noise_covariance)
            if jnp.iscomplexobj(raw_covariance):
                raise TypeError("noise_covariance must be real-valued.")
            covariance = raw_covariance.astype(value.dtype)
            if covariance.shape != (value.size, value.size):
                raise ValueError("noise_covariance must be square over observations.")
            physical = gaussian_factor_from_covariance(
                covariance,
                rank_tolerance=rank,
                hermitian_tolerance=rank,
                factor_id=f"{likelihood_id}-physical-noise",
            )
        elif noise_scale is not None:
            raw_scale = jnp.asarray(noise_scale)
            if jnp.iscomplexobj(raw_scale):
                raise TypeError("noise_scale must be real-valued.")
            scale = jnp.broadcast_to(raw_scale.astype(value.dtype), value.shape)
            scale = eqx.error_if(
                scale,
                jnp.any(~jnp.isfinite(scale)) | jnp.any(scale < 0.0),
                "noise_scale must be finite and nonnegative.",
            )
            physical = GaussianFactor(
                jnp.diag(scale),
                rank_tolerance=rank,
                factor_id=f"{likelihood_id}-physical-noise",
                resolved_method="diagonal-physical-noise",
            )
        else:
            physical = GaussianFactor(
                jnp.zeros((value.size, 0), dtype=value.dtype),
                rank_tolerance=rank,
                factor_id=f"{likelihood_id}-zero-noise",
                resolved_method="exact-zero-noise",
            )
        identifier = str(likelihood_id)
        if not identifier:
            raise ValueError("likelihood_id must be nonempty.")
        if stamp is None:
            stamp = ConditionRealizationStamp(
                identifier,
                identifier,
                identifier,
                "phydrax.uq.ConstraintLikelihoodTerm",
                quantifier=ConditionQuantifier.deterministic,
                exact=False,
            )
        if not isinstance(stamp, ConditionRealizationStamp):
            raise TypeError("stamp must be a ConditionRealizationStamp.")
        if bound is not None and not isinstance(bound, BoundCondition):
            raise TypeError("bound must be a BoundCondition or None.")
        self.observed = value
        self.physical_noise = physical
        self.stamp = stamp
        self.support_tolerance = tolerance
        self.likelihood_id = identifier
        self.bound = bound

    @classmethod
    def from_bound_condition(
        cls,
        bound: BoundCondition,
        /,
        *,
        noise_covariance: ArrayLike | None = None,
        noise_factor: GaussianFactor | None = None,
        support_tolerance: ArrayLike = 1e-8,
        rank_tolerance: ArrayLike = 0.0,
    ) -> ConstraintLikelihoodTerm:
        """Build from the shared typed action and its authoritative relation."""
        if not isinstance(bound, BoundCondition):
            raise TypeError("bound must be a BoundCondition.")
        if not isinstance(bound.codomain, ArrayCodomain):
            raise TypeError(
                "Probabilistic conditioning currently needs an ArrayCodomain."
            )
        relation = bound.relation
        if isinstance(relation, Equality):
            observed = (
                jnp.zeros(bound.codomain.shape, dtype=float)
                if not relation.has_target
                else relation.target
            )
            relation_scale = None
        elif isinstance(relation, NoisyObservation):
            if noise_covariance is not None or noise_factor is not None:
                raise ValueError(
                    "A NoisyObservation owns its physical noise; do not override it."
                )
            observed = relation.observed
            relation_scale = jnp.asarray(relation.noise_scale).reshape((-1,))
        else:
            raise TypeError(
                "Gaussian likelihoods require Equality or NoisyObservation relations."
            )
        stamp = ConditionRealizationStamp(
            bound.condition_id,
            bound.bound_id,
            f"{bound.bound_id}:probabilistic",
            "phydrax.uq.ConstraintLikelihoodTerm",
            quantifier=bound.condition.quantifier,
            exact=False,
        )
        return cls(
            jnp.asarray(observed).reshape((-1,)),
            noise_scale=relation_scale,
            noise_covariance=noise_covariance,
            noise_factor=noise_factor,
            support_tolerance=support_tolerance,
            rank_tolerance=rank_tolerance,
            likelihood_id=f"{bound.condition_id}:likelihood",
            stamp=stamp,
            bound=bound,
        )

    @property
    def event_size(self) -> int:
        return int(self.observed.size)

    @property
    def physical_noise_rank(self) -> Array:
        return self.physical_noise.numerical_rank

    def log_likelihood(self, prediction: ArrayLike, /) -> Array:
        """Return the normalized density on the physical noise support."""
        predicted = jnp.asarray(prediction, dtype=self.observed.dtype)
        if predicted.shape != self.observed.shape:
            raise ValueError("prediction must align with observed.")
        return _factor_log_probability(
            self.physical_noise,
            self.observed - predicted,
            support_tolerance=self.support_tolerance,
        )[0]

    def log_bound_likelihood(
        self,
        /,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Array:
        """Evaluate the shared bound action without introducing a UQ operator IR."""
        if self.bound is None:
            raise ValueError("This likelihood was not built from a BoundCondition.")
        prediction = jnp.asarray(self.bound.apply(key=key, **kwargs)).reshape((-1,))
        return self.log_likelihood(prediction)


class ConstraintConditioningResult(StrictModule):
    """Posterior Gaussian and auditable hard/noisy conditioning separation."""

    posterior_mean: Array
    posterior_factor: GaussianFactor
    coordinate_mean: Array
    coordinate_factor: GaussianFactor
    feasible_origin: Array | None
    feasible_basis: Array | None
    innovation: Array
    log_evidence: Array
    evidence: ProbabilisticConditioningEvidence
    physical_noise_rank: Array
    numerical_jitter: Array
    valid: Array
    status: Array
    approximation: ConstraintApproximation = eqx.field(static=True)
    zero_noise_bridge: bool = eqx.field(static=True)

    @property
    def posterior_covariance(self) -> Array:
        return self.posterior_factor.covariance

    @property
    def coordinate_covariance(self) -> Array:
        return self.coordinate_factor.covariance


class LinearGaussianConstraintConditioner(StrictModule):
    """Exact Gaussian update for linear condition coordinates."""

    numerical_jitter: Array
    rank_tolerance: Array
    support_tolerance: Array

    def __init__(
        self,
        *,
        numerical_jitter: ArrayLike = 0.0,
        rank_tolerance: ArrayLike = 1e-8,
        support_tolerance: ArrayLike = 1e-8,
    ):
        jitter = jnp.asarray(numerical_jitter, dtype=float)
        rank = jnp.asarray(rank_tolerance, dtype=float)
        support = jnp.asarray(support_tolerance, dtype=float)
        if jitter.ndim != 0 or rank.ndim != 0 or support.ndim != 0:
            raise ValueError("Conditioner tolerances and jitter must be scalar.")
        jitter = eqx.error_if(
            jitter,
            ~jnp.isfinite(jitter) | (jitter < 0.0),
            "numerical_jitter must be finite and nonnegative.",
        )
        rank = eqx.error_if(
            rank,
            ~jnp.isfinite(rank) | (rank < 0.0),
            "rank_tolerance must be finite and nonnegative.",
        )
        support = eqx.error_if(
            support,
            ~jnp.isfinite(support) | (support < 0.0),
            "support_tolerance must be finite and nonnegative.",
        )
        self.numerical_jitter = jitter
        self.rank_tolerance = rank
        self.support_tolerance = support

    def condition(
        self,
        prior_mean: ArrayLike,
        prior_factor: GaussianFactor,
        matrix: ArrayLike,
        likelihood: ConstraintLikelihoodTerm,
        /,
        *,
        offset: ArrayLike | None = None,
    ) -> ConstraintConditioningResult:
        if not isinstance(prior_factor, GaussianFactor):
            raise TypeError("prior_factor must be a GaussianFactor.")
        if not isinstance(likelihood, ConstraintLikelihoodTerm):
            raise TypeError("likelihood must be a ConstraintLikelihoodTerm.")
        mean = jnp.asarray(prior_mean)
        action = jnp.asarray(matrix)
        if mean.ndim != 1 or prior_factor.event_size != mean.size:
            raise ValueError("prior_mean and prior_factor must describe one flat event.")
        if action.shape != (likelihood.event_size, mean.size):
            raise ValueError(
                "matrix must map prior coordinates to likelihood coordinates."
            )
        shift = (
            jnp.zeros((likelihood.event_size,), dtype=mean.dtype)
            if offset is None
            else jnp.asarray(offset, dtype=mean.dtype)
        )
        if shift.shape != (likelihood.event_size,):
            raise ValueError("offset must align with likelihood coordinates.")
        transformed = GaussianFactor(
            action @ prior_factor.factor,
            rank_tolerance=self.rank_tolerance,
            factor_id=f"{likelihood.likelihood_id}-linear-prediction",
            resolved_method="certified-linear-action",
        )
        cross = prior_factor.factor @ jnp.conj(transformed.factor.T)
        return self._condition_blocks(
            mean,
            prior_factor,
            action @ mean + shift,
            transformed,
            cross,
            likelihood,
        )

    def condition_from_covariances(
        self,
        query_mean: ArrayLike,
        query_covariance: ArrayLike,
        condition_mean: ArrayLike,
        condition_covariance: ArrayLike,
        cross_covariance: ArrayLike,
        likelihood: ConstraintLikelihoodTerm,
        /,
    ) -> ConstraintConditioningResult:
        """Condition query coordinates from kernel/linear-action Gram blocks."""
        if not isinstance(likelihood, ConstraintLikelihoodTerm):
            raise TypeError("likelihood must be a ConstraintLikelihoodTerm.")
        query = jnp.asarray(query_mean)
        condition = jnp.asarray(condition_mean)
        query_cov = _covariance_matrix(query_covariance, query.size, "query_covariance")
        condition_cov = _covariance_matrix(
            condition_covariance, likelihood.event_size, "condition_covariance"
        )
        cross = jnp.asarray(cross_covariance)
        if query.ndim != 1 or condition.shape != (likelihood.event_size,):
            raise ValueError("Condition and query means must be flat aligned vectors.")
        if cross.shape != (query.size, likelihood.event_size):
            raise ValueError("cross_covariance has incompatible shape.")
        query_factor = gaussian_factor_from_covariance(
            query_cov,
            rank_tolerance=self.rank_tolerance,
            hermitian_tolerance=self.rank_tolerance,
            factor_id=f"{likelihood.likelihood_id}-query-prior",
        )
        condition_factor = gaussian_factor_from_covariance(
            condition_cov,
            rank_tolerance=self.rank_tolerance,
            hermitian_tolerance=self.rank_tolerance,
            factor_id=f"{likelihood.likelihood_id}-condition-prior",
        )
        return self._condition_blocks(
            query,
            query_factor,
            condition,
            condition_factor,
            cross,
            likelihood,
        )

    def log_evidence_from_covariance(
        self,
        condition_mean: ArrayLike,
        condition_covariance: ArrayLike,
        likelihood: ConstraintLikelihoodTerm,
        /,
    ) -> Array:
        if not isinstance(likelihood, ConstraintLikelihoodTerm):
            raise TypeError("likelihood must be a ConstraintLikelihoodTerm.")
        condition = jnp.asarray(condition_mean)
        if condition.shape != likelihood.observed.shape:
            raise ValueError("condition_mean must align with likelihood coordinates.")
        covariance = _covariance_matrix(
            condition_covariance, likelihood.event_size, "condition_covariance"
        )
        factor = gaussian_factor_from_covariance(
            covariance,
            rank_tolerance=self.rank_tolerance,
            hermitian_tolerance=self.rank_tolerance,
            factor_id=f"{likelihood.likelihood_id}-evidence-prior",
        )
        physical, solved = _predictive_factors(
            factor,
            likelihood,
            self.numerical_jitter,
            self.rank_tolerance,
        )
        residual = likelihood.observed - condition
        value, supported = _factor_log_probability(
            solved, residual, support_tolerance=self.support_tolerance
        )
        physical_supported = _factor_log_probability(
            physical, residual, support_tolerance=self.support_tolerance
        )[1]
        return jnp.where(supported & physical_supported, value, -jnp.inf)

    def _condition_blocks(
        self,
        query_mean: Array,
        query_factor: GaussianFactor,
        condition_mean: Array,
        condition_factor: GaussianFactor,
        cross_covariance: Array,
        likelihood: ConstraintLikelihoodTerm,
    ) -> ConstraintConditioningResult:
        physical, solved = _predictive_factors(
            condition_factor,
            likelihood,
            self.numerical_jitter,
            self.rank_tolerance,
        )
        moments = ConditionalGaussianMoments(
            condition_mean,
            solved,
            cross_covariance,
            moments_id=f"{likelihood.likelihood_id}-linear-moments",
            resolved_method="exact-linear-gaussian-moments",
        )
        posterior = condition_gaussian(
            query_mean,
            query_factor,
            moments,
            likelihood.observed,
            rank_tolerance=self.rank_tolerance,
            support_tolerance=self.support_tolerance,
            moments_id=f"{likelihood.likelihood_id}-posterior",
        )
        return _conditioning_result(
            query_mean,
            query_factor,
            posterior,
            condition_mean,
            physical,
            solved,
            likelihood,
            numerical_jitter=self.numerical_jitter,
            approximation="exact-linear",
            support_tolerance=self.support_tolerance,
        )


class ApproximateGaussianConstraintConditioner(StrictModule):
    """Moment-matched nonlinear Gaussian conditioning with an explicit rule."""

    method: Literal["first-order", "cubature", "unscented", "gauss-hermite"] = eqx.field(
        static=True
    )
    numerical_jitter: Array
    rank_tolerance: Array
    support_tolerance: Array
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    kappa: float = eqx.field(static=True)
    hermite_order: int = eqx.field(static=True)

    def __init__(
        self,
        method: Literal["first-order", "cubature", "unscented", "gauss-hermite"],
        /,
        *,
        numerical_jitter: ArrayLike = 0.0,
        rank_tolerance: ArrayLike = 1e-8,
        support_tolerance: ArrayLike = 1e-8,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
        hermite_order: int = 3,
    ):
        if method not in ("first-order", "cubature", "unscented", "gauss-hermite"):
            raise ValueError("Unknown nonlinear Gaussian approximation method.")
        jitter = jnp.asarray(numerical_jitter, dtype=float)
        rank = jnp.asarray(rank_tolerance, dtype=float)
        support = jnp.asarray(support_tolerance, dtype=float)
        if jitter.ndim != 0 or rank.ndim != 0 or support.ndim != 0:
            raise ValueError("Conditioner tolerances and jitter must be scalar.")
        alpha_value = float(alpha)
        beta_value = float(beta)
        kappa_value = float(kappa)
        if not isfinite(alpha_value) or alpha_value <= 0.0:
            raise ValueError("alpha must be finite and strictly positive.")
        if not isfinite(beta_value) or not isfinite(kappa_value):
            raise ValueError("beta and kappa must be finite.")
        if not isinstance(hermite_order, Integral) or isinstance(hermite_order, bool):
            raise TypeError("hermite_order must be an integer.")
        order = int(hermite_order)
        if order <= 0:
            raise ValueError("hermite_order must be positive.")
        self.method = method
        self.numerical_jitter = eqx.error_if(
            jitter,
            ~jnp.isfinite(jitter) | (jitter < 0.0),
            "numerical_jitter must be finite and nonnegative.",
        )
        self.rank_tolerance = eqx.error_if(
            rank,
            ~jnp.isfinite(rank) | (rank < 0.0),
            "rank_tolerance must be finite and nonnegative.",
        )
        self.support_tolerance = eqx.error_if(
            support,
            ~jnp.isfinite(support) | (support < 0.0),
            "support_tolerance must be finite and nonnegative.",
        )
        self.alpha = alpha_value
        self.beta = beta_value
        self.kappa = kappa_value
        self.hermite_order = order

    def condition(
        self,
        prior_mean: ArrayLike,
        prior_factor: GaussianFactor,
        function: Callable[[Array], Array],
        likelihood: ConstraintLikelihoodTerm,
        /,
    ) -> ConstraintConditioningResult:
        if not isinstance(prior_factor, GaussianFactor):
            raise TypeError("prior_factor must be a GaussianFactor.")
        if not isinstance(likelihood, ConstraintLikelihoodTerm):
            raise TypeError("likelihood must be a ConstraintLikelihoodTerm.")
        mean = jnp.asarray(prior_mean)
        if mean.ndim != 1 or prior_factor.event_size != mean.size:
            raise ValueError("prior_mean and prior_factor must describe one flat event.")
        if self.method == "first-order":
            transformed = first_order_gaussian_transform(function, mean, prior_factor)
        elif self.method == "cubature":
            transformed = spherical_radial_cubature(function, mean, prior_factor)
        elif self.method == "unscented":
            transformed = scaled_unscented_transform(
                function,
                mean,
                prior_factor,
                alpha=self.alpha,
                beta=self.beta,
                kappa=self.kappa,
            )
        else:
            transformed = gauss_hermite_transform(
                function,
                mean,
                prior_factor,
                order=self.hermite_order,
            )
        condition_mean = jnp.asarray(transformed.mean)
        if condition_mean.shape != likelihood.observed.shape:
            raise ValueError("Nonlinear condition output must align with observed.")
        physical, solved = _predictive_factors(
            transformed.factor,
            likelihood,
            self.numerical_jitter,
            self.rank_tolerance,
        )
        moments = ConditionalGaussianMoments(
            condition_mean,
            solved,
            transformed.cross_covariance,
            moments_id=f"{likelihood.likelihood_id}-{self.method}-moments",
            resolved_method=transformed.method_id,
        )
        posterior = condition_gaussian(
            mean,
            prior_factor,
            moments,
            likelihood.observed,
            rank_tolerance=self.rank_tolerance,
            support_tolerance=self.support_tolerance,
            moments_id=f"{likelihood.likelihood_id}-posterior",
        )
        return _conditioning_result(
            mean,
            prior_factor,
            posterior,
            condition_mean,
            physical,
            solved,
            likelihood,
            numerical_jitter=self.numerical_jitter,
            approximation=self.method,
            support_tolerance=self.support_tolerance,
        )


def build_constraint_posterior(
    prior: GaussianFactorLaw | tuple[ArrayLike, GaussianFactor],
    condition: ConstraintLikelihoodTerm,
    /,
    *,
    hard_operator: PreparedConstraintOperator | None = None,
    approximation: LinearGaussianConstraintConditioner
    | ApproximateGaussianConstraintConditioner
    | None = None,
    matrix: ArrayLike | None = None,
    function: Callable[[Array], Array] | None = None,
) -> ConstraintConditioningResult:
    """Condition in feasible coordinates, then lift through a hard constraint map."""
    if not isinstance(condition, ConstraintLikelihoodTerm):
        raise TypeError("condition must be a ConstraintLikelihoodTerm.")
    if hard_operator is not None and not isinstance(
        hard_operator, PreparedConstraintOperator
    ):
        raise TypeError("hard_operator must be a PreparedConstraintOperator or None.")
    coordinate_mean, coordinate_factor = _prior_coordinates(prior)
    if approximation is None:
        if matrix is not None or function is not None:
            raise ValueError("A zero-noise hard bridge does not take matrix or function.")
        if hard_operator is None or condition.physical_noise.rank != 0:
            raise ValueError(
                "A noisy or unbridged condition requires an explicit Gaussian conditioner."
            )
        coordinate_result = _identity_conditioning_result(
            coordinate_mean, coordinate_factor, condition
        )
    elif isinstance(approximation, LinearGaussianConstraintConditioner):
        if matrix is None or function is not None:
            raise ValueError("Linear conditioning requires matrix and no function.")
        coordinate_result = approximation.condition(
            coordinate_mean, coordinate_factor, matrix, condition
        )
    elif isinstance(approximation, ApproximateGaussianConstraintConditioner):
        if function is None or matrix is not None:
            raise ValueError("Nonlinear conditioning requires function and no matrix.")
        coordinate_result = approximation.condition(
            coordinate_mean, coordinate_factor, function, condition
        )
    else:
        raise TypeError(
            "approximation must be a Gaussian constraint conditioner or None."
        )
    if hard_operator is None:
        return coordinate_result
    target = hard_operator.target_space.unflatten(condition.observed)
    origin = hard_operator.source_space.flatten(hard_operator.minimum_norm_lift(target))
    basis = jnp.asarray(hard_operator.nullspace_basis)
    if basis.shape[-1] != coordinate_result.posterior_mean.size:
        raise ValueError(
            "Prior coordinates must align with the hard-constraint nullspace."
        )
    lifted_mean = origin + oe.contract("ij,j->i", basis, coordinate_result.posterior_mean)
    lifted_factor = GaussianFactor(
        basis @ coordinate_result.posterior_factor.factor,
        rank_tolerance=coordinate_result.posterior_factor.rank_tolerance,
        factor_id=f"{condition.likelihood_id}-feasible-posterior",
        resolved_method="hard-nullspace-coordinate-lift",
    )
    return ConstraintConditioningResult(
        posterior_mean=lifted_mean,
        posterior_factor=lifted_factor,
        coordinate_mean=coordinate_result.posterior_mean,
        coordinate_factor=coordinate_result.posterior_factor,
        feasible_origin=origin,
        feasible_basis=basis,
        innovation=coordinate_result.innovation,
        log_evidence=coordinate_result.log_evidence,
        evidence=coordinate_result.evidence,
        physical_noise_rank=coordinate_result.physical_noise_rank,
        numerical_jitter=coordinate_result.numerical_jitter,
        valid=coordinate_result.valid,
        status=coordinate_result.status,
        approximation=coordinate_result.approximation,
        zero_noise_bridge=approximation is None,
    )


def _conditioning_result(
    prior_mean: Array,
    prior_factor: GaussianFactor,
    posterior: ConditionalGaussianMoments,
    condition_mean: Array,
    physical: GaussianFactor,
    solved: GaussianFactor,
    likelihood: ConstraintLikelihoodTerm,
    /,
    *,
    numerical_jitter: Array,
    approximation: ConstraintApproximation,
    support_tolerance: Array,
) -> ConstraintConditioningResult:
    innovation = likelihood.observed - condition_mean
    log_evidence, supported = _factor_log_probability(
        solved, innovation, support_tolerance=support_tolerance
    )
    physical_supported = _factor_log_probability(
        physical, innovation, support_tolerance=support_tolerance
    )[1]
    factors_valid = (
        posterior.factor.valid & prior_factor.valid & physical.valid & solved.valid
    )
    finite = (
        jnp.all(jnp.isfinite(prior_mean))
        & jnp.all(jnp.isfinite(condition_mean))
        & jnp.all(jnp.isfinite(innovation))
        & jnp.all(jnp.isfinite(posterior.mean))
        & jnp.all(jnp.isfinite(posterior.factor.factor))
    )
    valid = (
        posterior.valid
        & factors_valid
        & finite
        & supported
        & physical_supported
        & jnp.isfinite(log_evidence)
    )
    status = jnp.where(
        ~factors_valid,
        CONSTRAINT_CONDITIONING_INVALID_FACTOR,
        jnp.where(
            ~finite,
            CONSTRAINT_CONDITIONING_NONFINITE,
            jnp.where(
                ~(supported & physical_supported),
                CONSTRAINT_CONDITIONING_INCONSISTENT_SUPPORT,
                jnp.where(
                    ~jnp.isfinite(log_evidence),
                    CONSTRAINT_CONDITIONING_NONFINITE,
                    CONSTRAINT_CONDITIONING_SUCCESS,
                ),
            ),
        ),
    ).astype(jnp.int32)
    log_value = jnp.where(valid, log_evidence, -jnp.inf)
    evidence = ProbabilisticConditioningEvidence(
        likelihood.stamp,
        log_value,
        jnp.where(valid, 1.0, 0.0),
        jnp.where(valid, 1.0, 0.0),
        evidence_id=(
            f"{likelihood.likelihood_id}:{approximation}:"
            f"physical-noise+separate-numerical-jitter"
        ),
    )
    return ConstraintConditioningResult(
        posterior_mean=jnp.where(valid, posterior.mean, prior_mean),
        posterior_factor=posterior.factor,
        coordinate_mean=jnp.where(valid, posterior.mean, prior_mean),
        coordinate_factor=posterior.factor,
        feasible_origin=None,
        feasible_basis=None,
        innovation=innovation,
        log_evidence=log_value,
        evidence=evidence,
        physical_noise_rank=likelihood.physical_noise_rank,
        numerical_jitter=numerical_jitter,
        valid=valid,
        status=status,
        approximation=approximation,
        zero_noise_bridge=False,
    )


def _identity_conditioning_result(
    mean: Array, factor: GaussianFactor, likelihood: ConstraintLikelihoodTerm, /
) -> ConstraintConditioningResult:
    zero = jnp.asarray(0.0, dtype=mean.dtype)
    source_stamp = likelihood.stamp
    stamp = ConditionRealizationStamp(
        source_stamp.condition_id,
        source_stamp.source_id,
        f"{source_stamp.realization_id}:hard-bridge",
        "phydrax.uq.build_constraint_posterior",
        quantifier=source_stamp.quantifier,
        exact=True,
    )
    evidence = ProbabilisticConditioningEvidence(
        stamp,
        zero,
        jnp.where(factor.valid, 1.0, 0.0),
        jnp.where(factor.valid, 1.0, 0.0),
        evidence_id=f"{likelihood.likelihood_id}:exact-zero-noise-hard-bridge",
    )
    return ConstraintConditioningResult(
        posterior_mean=mean,
        posterior_factor=factor,
        coordinate_mean=mean,
        coordinate_factor=factor,
        feasible_origin=None,
        feasible_basis=None,
        innovation=jnp.zeros_like(likelihood.observed),
        log_evidence=zero,
        evidence=evidence,
        physical_noise_rank=likelihood.physical_noise_rank,
        numerical_jitter=zero,
        valid=factor.valid,
        status=jnp.where(
            factor.valid,
            CONSTRAINT_CONDITIONING_SUCCESS,
            CONSTRAINT_CONDITIONING_INVALID_FACTOR,
        ).astype(jnp.int32),
        approximation="exact-linear",
        zero_noise_bridge=True,
    )


def _predictive_factors(
    prior: GaussianFactor,
    likelihood: ConstraintLikelihoodTerm,
    jitter: Array,
    rank_tolerance: Array,
    /,
) -> tuple[GaussianFactor, GaussianFactor]:
    physical = add_independent_gaussian_factors(
        prior,
        likelihood.physical_noise,
        compress=False,
        factor_id=f"{likelihood.likelihood_id}-physical-predictive",
    )
    identity = jnp.eye(likelihood.event_size, dtype=prior.factor.dtype)
    numerical = GaussianFactor(
        jnp.sqrt(jitter) * identity,
        regularization=jitter,
        rank_tolerance=rank_tolerance,
        factor_id=f"{likelihood.likelihood_id}-numerical-jitter",
        resolved_method="separate-diagonal-numerical-jitter",
    )
    solved = add_independent_gaussian_factors(
        physical,
        numerical,
        compress=False,
        factor_id=f"{likelihood.likelihood_id}-solved-predictive",
    )
    return physical, solved


def _factor_log_probability(
    factor: GaussianFactor,
    residual: Array,
    /,
    *,
    support_tolerance: Array,
) -> tuple[Array, Array]:
    value = jnp.asarray(residual)
    if value.shape != (factor.event_size,):
        raise ValueError("residual must align with factor.event_size.")
    if factor.rank == 0:
        norm = jnp.linalg.norm(value)
        supported = norm <= support_tolerance * (1.0 + norm)
        valid = supported & factor.valid & jnp.isfinite(norm)
        return jnp.where(valid, 0.0, -jnp.inf), supported
    left, singular, _ = jnp.linalg.svd(factor.factor, full_matrices=False)
    active = singular > factor.rank_tolerance
    coefficients = oe.contract("ir,i->r", jnp.conj(left), value)
    scaled = jnp.where(active, coefficients / jnp.where(active, singular, 1.0), 0.0)
    projected = oe.contract("ir,r->i", left, jnp.where(active, coefficients, 0.0))
    support_error = jnp.linalg.norm(value - projected)
    supported = support_error <= support_tolerance * (1.0 + jnp.linalg.norm(value))
    rank = jnp.sum(active, dtype=jnp.real(value).dtype)
    log_pseudodeterminant = 2.0 * jnp.sum(
        jnp.where(active, jnp.log(jnp.where(active, singular, 1.0)), 0.0)
    )
    quadratic = jnp.real(jnp.sum(jnp.conj(scaled) * scaled))
    log_probability = -0.5 * (
        quadratic + log_pseudodeterminant + rank * jnp.log(2.0 * jnp.pi)
    )
    valid = supported & factor.valid & jnp.isfinite(log_probability)
    return jnp.where(valid, log_probability, -jnp.inf), supported


def _covariance_matrix(value: ArrayLike, size: int, name: str, /) -> Array:
    matrix = jnp.asarray(value)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}).")
    return matrix


def _prior_coordinates(
    prior: GaussianFactorLaw | tuple[ArrayLike, GaussianFactor], /
) -> tuple[Array, GaussianFactor]:
    if isinstance(prior, GaussianFactorLaw):
        return prior.location.reshape((-1,)), prior.factor
    if not isinstance(prior, tuple) or len(prior) != 2:
        raise TypeError("prior must be a GaussianFactorLaw or (mean, GaussianFactor).")
    mean = jnp.asarray(prior[0])
    factor = prior[1]
    if not isinstance(factor, GaussianFactor):
        raise TypeError("prior factor must be a GaussianFactor.")
    if mean.ndim != 1 or factor.event_size != mean.size:
        raise ValueError("prior mean and factor must describe one flat event.")
    return mean, factor


__all__ = [
    "ApproximateGaussianConstraintConditioner",
    "CONSTRAINT_CONDITIONING_INCONSISTENT_SUPPORT",
    "CONSTRAINT_CONDITIONING_INVALID_FACTOR",
    "CONSTRAINT_CONDITIONING_NONFINITE",
    "CONSTRAINT_CONDITIONING_SUCCESS",
    "ConstraintApproximation",
    "ConstraintConditioningResult",
    "ConstraintConditioningStatus",
    "ConstraintLikelihoodTerm",
    "LinearGaussianConstraintConditioner",
    "build_constraint_posterior",
]
