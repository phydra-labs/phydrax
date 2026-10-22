#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import phydrax.axes as cx
from phydrax.kernels import kernel_feature_rank, kernel_features
from phydrax.linalg import ConditionTargetResult, LinearSolveResult, PreparedFactorization

from .._strict import StrictModule
from ._gp_backend import (
    exact_gp_conditioner,
    fitc_factors,
    low_rank_gp_conditioner,
    low_rank_gp_correction_cholesky,
    sparse_gp_conditioner,
    sparse_gp_log_probability_from_factors,
)
from ._gp_condition import GaussianProcessCondition, GaussianProcessConditioner
from ._gp_exact import (
    exact_factor_storage,
    exact_gp_factorization,
    exact_gp_log_marginal_likelihood,
    exact_gp_log_probability_from_factor,
    require_exact_factor_storage,
)
from ._gp_likelihood import _require_unconditioned_state, GaussianProcessLikelihoodState


class ExactGaussianProcessFactor(StrictModule):
    """Reusable exact covariance factor for one GP likelihood state."""

    observation_points: Array
    factorization: PreparedFactorization
    evidence: LinearSolveResult
    regularization: ConditionTargetResult | None
    state: GaussianProcessLikelihoodState

    def __init__(
        self,
        observation_points: ArrayLike | cx.AxisArray,
        /,
        *,
        state: GaussianProcessLikelihoodState,
    ) -> None:
        points = _validated_factor_points(observation_points)
        _require_state(state)
        self.observation_points = points
        factorization, regularization = exact_gp_factorization(
            points,
            kernel=state.kernel,
            noise_scale=state.noise_scale,
            jitter=state.jitter,
            execution=state.execution,
            conditioning=state.conditioning,
        )
        evidence = factorization.solve(jnp.ones((points.shape[0],), dtype=points.dtype))
        require_exact_factor_storage(
            factorization, evidence, regularization, state.execution
        )
        self.factorization = factorization
        self.evidence = evidence
        self.regularization = regularization
        self.state = state

    @property
    def factor_storage_elements(self) -> int:
        """Distinct native factor/evidence payloads, including matrix and factor."""
        return exact_factor_storage(
            self.factorization, self.evidence, self.regularization
        )[0]

    @property
    def factor_storage_bytes(self) -> int:
        """Byte count of the same native numerical payloads as element storage."""
        return exact_factor_storage(
            self.factorization, self.evidence, self.regularization
        )[1]

    def log_probability(self, residual: ArrayLike, /) -> Array:
        """Evaluate a residual log density without refactorizing covariance."""
        values = _as_vector(residual, name="GP residual")
        if values.shape[0] != self.observation_points.shape[0]:
            raise ValueError("GP residual must align with factor observations.")
        return exact_gp_log_probability_from_factor(values, self.factorization)

    def conditioner(
        self,
        query_points: ArrayLike | cx.AxisArray,
        /,
        *,
        output_dim: str | None = "point",
    ) -> GaussianProcessConditioner:
        """Precompute conditioning geometry for one fixed query design."""
        query_data = _field_data(query_points)
        projection, covariance, variance = exact_gp_conditioner(
            self.observation_points,
            query_data,
            factorization=self.factorization,
            kernel=self.state.kernel,
        )
        return GaussianProcessConditioner(
            query_points=query_data,
            residual_projection=projection,
            covariance=covariance,
            variance=variance,
            output_dims=_query_output_dims(query_points, output_dim=output_dim),
        )

    def condition(
        self,
        residual: ArrayLike,
        query_points: ArrayLike | cx.AxisArray,
        /,
        *,
        output_dim: str | None = "point",
    ) -> GaussianProcessCondition:
        """Condition a residual vector using this reusable covariance factor."""
        return self.conditioner(query_points, output_dim=output_dim).condition(residual)


class FiniteFeatureGaussianProcessFactor(StrictModule):
    """Exact weight-space factor for a finite-feature GP covariance."""

    observation_points: Array
    features: Array
    diagonal: Array
    correction_cholesky: Array
    state: GaussianProcessLikelihoodState

    def __init__(
        self,
        observation_points: ArrayLike | cx.AxisArray,
        /,
        *,
        state: GaussianProcessLikelihoodState,
    ) -> None:
        points = _validated_factor_points(observation_points)
        _require_unconditioned_state(state)
        if kernel_feature_rank(state.kernel) is None:
            raise TypeError(
                "FiniteFeatureGaussianProcessFactor requires an exact finite-feature kernel representation."
            )
        noise = _observation_noise(
            state.noise_scale,
            count=points.shape[0],
        )
        features = kernel_features(state.kernel, points)
        diagonal = noise * noise + state.jitter
        self.observation_points = points
        self.features = features
        self.diagonal = diagonal
        self.correction_cholesky = low_rank_gp_correction_cholesky(
            features,
            diagonal,
        )
        self.state = state

    @property
    def factor_storage_elements(self) -> int:
        """Number of elements retained by the exact weight-space factor."""
        return self.features.size + self.diagonal.size + self.correction_cholesky.size

    def log_probability(self, residual: ArrayLike, /) -> Array:
        """Evaluate an exact finite-feature GP log density."""
        values = _as_vector(residual, name="GP residual")
        if values.shape[0] != self.observation_points.shape[0]:
            raise ValueError("GP residual must align with factor observations.")
        return sparse_gp_log_probability_from_factors(
            values,
            self.features,
            self.diagonal,
            self.correction_cholesky,
        )

    def conditioner(
        self,
        query_points: ArrayLike | cx.AxisArray,
        /,
        *,
        output_dim: str | None = "point",
    ) -> GaussianProcessConditioner:
        """Precompute exact weight-space conditioning geometry."""
        query_data = _field_data(query_points)
        query_features = kernel_features(self.state.kernel, query_data)
        projection, covariance, variance = low_rank_gp_conditioner(
            query_features,
            jnp.zeros((query_features.shape[0],), dtype=query_features.dtype),
            features=self.features,
            diagonal=self.diagonal,
            correction_cholesky=self.correction_cholesky,
        )
        return GaussianProcessConditioner(
            query_points=query_data,
            residual_projection=projection,
            covariance=covariance,
            variance=variance,
            output_dims=_query_output_dims(query_points, output_dim=output_dim),
        )

    def condition(
        self,
        residual: ArrayLike,
        query_points: ArrayLike | cx.AxisArray,
        /,
        *,
        output_dim: str | None = "point",
    ) -> GaussianProcessCondition:
        """Condition a residual vector using exact weight-space factors."""
        return self.conditioner(query_points, output_dim=output_dim).condition(residual)


class SparseGaussianProcessFactor(StrictModule):
    """Reusable FITC factors for one GP likelihood state and inducing design."""

    observation_points: Array
    inducing_points: Array
    features: Array
    diagonal: Array
    correction_cholesky: Array
    inducing_cholesky: Array
    state: GaussianProcessLikelihoodState

    def __init__(
        self,
        observation_points: ArrayLike | cx.AxisArray,
        inducing_points: ArrayLike | cx.AxisArray,
        /,
        *,
        state: GaussianProcessLikelihoodState,
    ) -> None:
        points = _validated_factor_points(observation_points)
        inducing = _validated_factor_points(inducing_points)
        _validate_inducing_design(points, inducing)
        _require_unconditioned_state(state)
        factors = fitc_factors(
            points,
            inducing,
            kernel=state.kernel,
            noise_scale=state.noise_scale,
            jitter=state.jitter,
        )
        self.observation_points = points
        self.inducing_points = inducing
        self.features = factors[0]
        self.diagonal = factors[1]
        self.correction_cholesky = factors[2]
        self.inducing_cholesky = factors[3]
        self.state = state

    @property
    def factor_storage_elements(self) -> int:
        """Number of elements retained by the reusable FITC factors."""
        return (
            self.features.size
            + self.diagonal.size
            + self.correction_cholesky.size
            + self.inducing_cholesky.size
        )

    def log_probability(self, residual: ArrayLike, /) -> Array:
        """Evaluate a residual FITC log density without rebuilding factors."""
        values = _as_vector(residual, name="GP residual")
        if values.shape[0] != self.observation_points.shape[0]:
            raise ValueError("GP residual must align with factor observations.")
        return sparse_gp_log_probability_from_factors(
            values,
            self.features,
            self.diagonal,
            self.correction_cholesky,
        )

    def conditioner(
        self,
        query_points: ArrayLike | cx.AxisArray,
        /,
        *,
        output_dim: str | None = "point",
    ) -> GaussianProcessConditioner:
        """Precompute FITC conditioning geometry for one fixed query design."""
        query_data = _field_data(query_points)
        projection, covariance, variance = sparse_gp_conditioner(
            self.observation_points,
            self.inducing_points,
            query_data,
            features=self.features,
            diagonal=self.diagonal,
            correction_cholesky=self.correction_cholesky,
            inducing_cholesky=self.inducing_cholesky,
            kernel=self.state.kernel,
        )
        return GaussianProcessConditioner(
            query_points=query_data,
            residual_projection=projection,
            covariance=covariance,
            variance=variance,
            output_dims=_query_output_dims(query_points, output_dim=output_dim),
        )

    def condition(
        self,
        residual: ArrayLike,
        query_points: ArrayLike | cx.AxisArray,
        /,
        *,
        output_dim: str | None = "point",
    ) -> GaussianProcessCondition:
        """Condition a residual vector using reusable FITC factors."""
        return self.conditioner(query_points, output_dim=output_dim).condition(residual)


class ExactGaussianProcessDiscrepancy(StrictModule):
    """Exact scalar-output GP model for additive model-form discrepancy."""

    observation_points: Array
    observations: Array

    def __init__(
        self,
        observation_points: ArrayLike | cx.AxisArray,
        observations: ArrayLike | cx.AxisArray,
        /,
    ) -> None:
        points, values = _validated_observations(
            observation_points,
            observations,
            name="GP observations",
        )
        self.observation_points = points
        self.observations = values

    def residual(self, physical_mean: ArrayLike, /) -> Array:
        """Return observations minus the physical-model mean."""
        mean = _as_vector(physical_mean, name="physical observation mean")
        if mean.shape != self.observations.shape:
            raise ValueError("physical_mean must align with GP observations.")
        return self.observations - mean

    def factor(
        self,
        *,
        state: GaussianProcessLikelihoodState,
    ) -> ExactGaussianProcessFactor | FiniteFeatureGaussianProcessFactor:
        """Use exact weight space unless full-covariance conditioning is requested."""
        _require_state(state)
        rank = kernel_feature_rank(state.kernel)
        if (
            state.conditioning is None
            and rank is not None
            and rank < self.observation_points.shape[0]
        ):
            return FiniteFeatureGaussianProcessFactor(
                self.observation_points,
                state=state,
            )
        return ExactGaussianProcessFactor(self.observation_points, state=state)

    def log_marginal_likelihood(
        self,
        physical_mean: ArrayLike,
        /,
        *,
        state: GaussianProcessLikelihoodState,
    ) -> Array:
        """Marginalize latent discrepancy values under a Gaussian likelihood."""
        _require_state(state)
        residual = self.residual(physical_mean)
        rank = kernel_feature_rank(state.kernel)
        if (
            state.conditioning is None
            and rank is not None
            and rank < self.observation_points.shape[0]
        ):
            return self.factor(state=state).log_probability(residual)
        return exact_gp_log_marginal_likelihood(
            self.observation_points,
            residual,
            state.kernel,
            state.noise_scale,
            state.jitter,
            state.execution,
            state.conditioning,
        )

    def condition(
        self,
        physical_mean: ArrayLike,
        query_points: ArrayLike | cx.AxisArray,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        output_dim: str | None = "point",
    ) -> GaussianProcessCondition:
        """Condition the latent discrepancy on residual observations."""
        factor = self.factor(state=state)
        return factor.condition(
            self.residual(physical_mean),
            query_points,
            output_dim=output_dim,
        )


class SparseGaussianProcessDiscrepancy(StrictModule):
    """Scalar-output FITC discrepancy with explicit inducing points."""

    observation_points: Array
    observations: Array
    inducing_points: Array

    def __init__(
        self,
        observation_points: ArrayLike | cx.AxisArray,
        observations: ArrayLike | cx.AxisArray,
        inducing_points: ArrayLike | cx.AxisArray,
        /,
    ) -> None:
        points, values = _validated_observations(
            observation_points,
            observations,
            name="sparse GP observations",
        )
        inducing = _validated_factor_points(inducing_points)
        _validate_inducing_design(points, inducing)
        self.observation_points = points
        self.observations = values
        self.inducing_points = inducing

    @classmethod
    def from_evenly_spaced_subset(
        cls,
        observation_points: ArrayLike | cx.AxisArray,
        observations: ArrayLike | cx.AxisArray,
        /,
        *,
        num_inducing: int,
    ) -> SparseGaussianProcessDiscrepancy:
        """Choose a deterministic index-spaced inducing subset."""
        points = _as_design(_field_data(observation_points))
        count = int(num_inducing)
        if not 0 < count < points.shape[0]:
            raise ValueError(
                "num_inducing must be positive and smaller than observation count."
            )
        indices = jnp.round(jnp.linspace(0, points.shape[0] - 1, count)).astype(jnp.int32)
        return cls(observation_points, observations, points[indices])

    @property
    def num_inducing(self) -> int:
        return self.inducing_points.shape[0]

    @property
    def factor_storage_elements(self) -> int:
        """Dominant FITC factor storage, versus n squared for the exact GP."""
        observations = self.observation_points.shape[0]
        inducing = self.num_inducing
        return observations * inducing + 2 * inducing**2 + observations

    def residual(self, physical_mean: ArrayLike, /) -> Array:
        mean = _as_vector(physical_mean, name="physical observation mean")
        if mean.shape != self.observations.shape:
            raise ValueError("physical_mean must align with GP observations.")
        return self.observations - mean

    def factor(
        self,
        *,
        state: GaussianProcessLikelihoodState,
    ) -> SparseGaussianProcessFactor:
        """Build reusable FITC factors for repeated residual evaluations."""
        return SparseGaussianProcessFactor(
            self.observation_points,
            self.inducing_points,
            state=state,
        )

    def log_marginal_likelihood(
        self,
        physical_mean: ArrayLike,
        /,
        *,
        state: GaussianProcessLikelihoodState,
    ) -> Array:
        """Evaluate the matrix-free FITC marginal likelihood."""
        return self.factor(state=state).log_probability(self.residual(physical_mean))

    def condition(
        self,
        physical_mean: ArrayLike,
        query_points: ArrayLike | cx.AxisArray,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        output_dim: str | None = "point",
    ) -> GaussianProcessCondition:
        """Condition the FITC latent discrepancy at query points."""
        return self.factor(state=state).condition(
            self.residual(physical_mean),
            query_points,
            output_dim=output_dim,
        )


def _field_data(value: Any) -> Array:
    return jnp.asarray(
        value.data if isinstance(value, cx.AxisArray) else value, dtype=jnp.float64
    )


def _as_design(value: ArrayLike) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim < 2:
        raise ValueError(
            "GP inputs must have one design axis and at least one input axis."
        )
    if any(size <= 0 for size in array.shape[1:]):
        raise ValueError("GP kernel input axes must be nonempty.")
    return array


def _as_vector(value: ArrayLike, /, *, name: str) -> Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional scalar-output array.")
    return array


def _query_output_dims(
    query_points: Any,
    /,
    *,
    output_dim: str | None,
) -> tuple[str | None, ...]:
    if isinstance(query_points, cx.AxisArray):
        if query_points.data.ndim == 1:
            return tuple(query_points.dims)
        if query_points.data.ndim >= 2:
            return (query_points.dims[0],)
    return (output_dim,)


def _validated_factor_points(value: ArrayLike | cx.AxisArray, /) -> Array:
    points = _as_design(_field_data(value))
    return eqx.error_if(
        points,
        jnp.any(~jnp.isfinite(points)),
        "GP factor points must be finite.",
    )


def _validated_observations(
    observation_points: ArrayLike | cx.AxisArray,
    observations: ArrayLike | cx.AxisArray,
    /,
    *,
    name: str,
) -> tuple[Array, Array]:
    points = _as_design(_field_data(observation_points))
    values = _as_vector(_field_data(observations), name=name)
    if points.shape[0] != values.shape[0]:
        raise ValueError("GP observations must align with observation points.")
    return eqx.error_if(
        (points, values),
        jnp.any(~jnp.isfinite(points)) | jnp.any(~jnp.isfinite(values)),
        "GP observations and points must be finite.",
    )


def _observation_noise(noise_scale: ArrayLike, /, *, count: int) -> Array:
    noise = jnp.asarray(noise_scale)
    if noise.ndim == 0:
        return jnp.broadcast_to(noise, (count,))
    if noise.shape != (count,):
        raise ValueError("Vector GP noise must align with observation points.")
    return noise


def _validate_inducing_design(points: Array, inducing: Array, /) -> None:
    if not 0 < inducing.shape[0] < points.shape[0]:
        raise ValueError("Sparse GP requires fewer inducing points than observations.")


def _require_state(state: GaussianProcessLikelihoodState, /) -> None:
    if not isinstance(state, GaussianProcessLikelihoodState):
        raise TypeError("state must be a GaussianProcessLikelihoodState.")


__all__ = [
    "ExactGaussianProcessDiscrepancy",
    "ExactGaussianProcessFactor",
    "FiniteFeatureGaussianProcessFactor",
    "SparseGaussianProcessDiscrepancy",
    "SparseGaussianProcessFactor",
]
