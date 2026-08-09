#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.kernels import kernel_feature_rank, kernel_features

from .._strict import StrictModule
from ._gp_backend import (
    exact_gp_cholesky,
    exact_gp_conditioner,
    exact_gp_log_probability,
    fitc_factors,
    low_rank_gp_conditioner,
    low_rank_gp_correction_cholesky,
    sparse_gp_conditioner,
    sparse_gp_log_probability_from_factors,
)
from ._gp_condition import GaussianProcessCondition, GaussianProcessConditioner
from ._gp_likelihood import GaussianProcessLikelihoodState


class ExactGaussianProcessFactor(StrictModule):
    """Reusable exact covariance factor for one GP likelihood state."""

    observation_points: Array
    cholesky: Array
    state: GaussianProcessLikelihoodState

    def __init__(
        self,
        observation_points: ArrayLike | cx.Field,
        /,
        *,
        state: GaussianProcessLikelihoodState,
    ):
        points = _validated_factor_points(observation_points)
        _require_state(state)
        self.observation_points = points
        self.cholesky = exact_gp_cholesky(
            points,
            kernel=state.kernel,
            noise_scale=state.noise_scale,
            jitter=state.jitter,
        )
        self.state = state

    @property
    def factor_storage_elements(self) -> int:
        """Number of elements in the dominant dense covariance factor."""
        return int(self.cholesky.size)

    def log_probability(self, residual: ArrayLike, /) -> Array:
        """Evaluate a residual log density without refactorizing covariance."""
        values = _as_vector(residual, name="GP residual")
        if int(values.shape[0]) != int(self.observation_points.shape[0]):
            raise ValueError("GP residual must align with factor observations.")
        return exact_gp_log_probability(values, self.cholesky)

    def conditioner(
        self,
        query_points: ArrayLike | cx.Field,
        /,
        *,
        output_dim: str | None = "point",
    ) -> GaussianProcessConditioner:
        """Precompute conditioning geometry for one fixed query design."""
        query_data = _field_data(query_points)
        projection, covariance, variance = exact_gp_conditioner(
            self.observation_points,
            query_data,
            cholesky=self.cholesky,
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
        query_points: ArrayLike | cx.Field,
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
        observation_points: ArrayLike | cx.Field,
        /,
        *,
        state: GaussianProcessLikelihoodState,
    ):
        points = _validated_factor_points(observation_points)
        _require_state(state)
        if kernel_feature_rank(state.kernel) is None:
            raise TypeError(
                "FiniteFeatureGaussianProcessFactor requires an exact "
                "finite-feature kernel representation."
            )
        noise = _observation_noise(
            state.noise_scale,
            count=int(points.shape[0]),
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
        return (
            int(self.features.size)
            + int(self.diagonal.size)
            + int(self.correction_cholesky.size)
        )

    def log_probability(self, residual: ArrayLike, /) -> Array:
        """Evaluate an exact finite-feature GP log density."""
        values = _as_vector(residual, name="GP residual")
        if int(values.shape[0]) != int(self.observation_points.shape[0]):
            raise ValueError("GP residual must align with factor observations.")
        return sparse_gp_log_probability_from_factors(
            values,
            self.features,
            self.diagonal,
            self.correction_cholesky,
        )

    def conditioner(
        self,
        query_points: ArrayLike | cx.Field,
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
        query_points: ArrayLike | cx.Field,
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
        observation_points: ArrayLike | cx.Field,
        inducing_points: ArrayLike | cx.Field,
        /,
        *,
        state: GaussianProcessLikelihoodState,
    ):
        points = _validated_factor_points(observation_points)
        inducing = _validated_factor_points(inducing_points)
        _validate_inducing_design(points, inducing)
        _require_state(state)
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
            int(self.features.size)
            + int(self.diagonal.size)
            + int(self.correction_cholesky.size)
            + int(self.inducing_cholesky.size)
        )

    def log_probability(self, residual: ArrayLike, /) -> Array:
        """Evaluate a residual FITC log density without rebuilding factors."""
        values = _as_vector(residual, name="GP residual")
        if int(values.shape[0]) != int(self.observation_points.shape[0]):
            raise ValueError("GP residual must align with factor observations.")
        return sparse_gp_log_probability_from_factors(
            values,
            self.features,
            self.diagonal,
            self.correction_cholesky,
        )

    def conditioner(
        self,
        query_points: ArrayLike | cx.Field,
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
        query_points: ArrayLike | cx.Field,
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
        observation_points: ArrayLike | cx.Field,
        observations: ArrayLike | cx.Field,
        /,
    ):
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
        """Use exact weight space when feature rank is below observation count."""
        _require_state(state)
        rank = kernel_feature_rank(state.kernel)
        if rank is not None and rank < int(self.observation_points.shape[0]):
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
        factor = self.factor(state=state)
        return factor.log_probability(self.residual(physical_mean))

    def condition(
        self,
        physical_mean: ArrayLike,
        query_points: ArrayLike | cx.Field,
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
        observation_points: ArrayLike | cx.Field,
        observations: ArrayLike | cx.Field,
        inducing_points: ArrayLike | cx.Field,
        /,
    ):
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
        observation_points: ArrayLike | cx.Field,
        observations: ArrayLike | cx.Field,
        /,
        *,
        num_inducing: int,
    ) -> SparseGaussianProcessDiscrepancy:
        """Choose a deterministic index-spaced inducing subset."""
        points = _as_points(_field_data(observation_points))
        count = int(num_inducing)
        if not 0 < count < int(points.shape[0]):
            raise ValueError(
                "num_inducing must be positive and smaller than observation count."
            )
        indices = jnp.round(jnp.linspace(0, int(points.shape[0]) - 1, count)).astype(
            jnp.int32
        )
        return cls(observation_points, observations, points[indices])

    @property
    def num_inducing(self) -> int:
        return int(self.inducing_points.shape[0])

    @property
    def factor_storage_elements(self) -> int:
        """Dominant FITC factor storage, versus n squared for the exact GP."""
        observations = int(self.observation_points.shape[0])
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
        _require_state(state)
        factors = fitc_factors(
            self.observation_points,
            self.inducing_points,
            kernel=state.kernel,
            noise_scale=state.noise_scale,
            jitter=state.jitter,
        )
        return sparse_gp_log_probability_from_factors(
            self.residual(physical_mean),
            factors[0],
            factors[1],
            factors[2],
        )

    def condition(
        self,
        physical_mean: ArrayLike,
        query_points: ArrayLike | cx.Field,
        /,
        *,
        state: GaussianProcessLikelihoodState,
        output_dim: str | None = "point",
    ) -> GaussianProcessCondition:
        """Condition the FITC latent discrepancy at query points."""
        _require_state(state)
        query_data = _field_data(query_points)
        factors = fitc_factors(
            self.observation_points,
            self.inducing_points,
            kernel=state.kernel,
            noise_scale=state.noise_scale,
            jitter=state.jitter,
        )
        projection, covariance, variance = sparse_gp_conditioner(
            self.observation_points,
            self.inducing_points,
            query_data,
            features=factors[0],
            diagonal=factors[1],
            correction_cholesky=factors[2],
            inducing_cholesky=factors[3],
            kernel=state.kernel,
        )
        return GaussianProcessCondition(
            query_points=query_data,
            mean=projection @ self.residual(physical_mean),
            covariance=covariance,
            variance=variance,
            output_dims=_query_output_dims(query_points, output_dim=output_dim),
        )


def _field_data(value: Any) -> Array:
    return jnp.asarray(value.data if isinstance(value, cx.Field) else value, dtype=float)


def _as_points(value: ArrayLike) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError("GP points must have shape (point, coordinate).")
    return array


def _as_vector(value: ArrayLike, /, *, name: str) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 2 and int(array.shape[1]) == 1:
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
    if isinstance(query_points, cx.Field):
        if query_points.data.ndim == 1:
            return tuple(query_points.dims)
        if query_points.data.ndim == 2:
            return (query_points.dims[0],)
    return (output_dim,)


def _validated_factor_points(value: ArrayLike | cx.Field, /) -> Array:
    points = _as_points(_field_data(value))
    return eqx.error_if(
        points,
        jnp.any(~jnp.isfinite(points)),
        "GP factor points must be finite.",
    )


def _validated_observations(
    observation_points: ArrayLike | cx.Field,
    observations: ArrayLike | cx.Field,
    /,
    *,
    name: str,
) -> tuple[Array, Array]:
    points = _as_points(_field_data(observation_points))
    values = _as_vector(_field_data(observations), name=name)
    if int(points.shape[0]) != int(values.shape[0]):
        raise ValueError("GP observations must align with observation points.")
    if not bool(jnp.all(jnp.isfinite(points))) or not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("GP observations and points must be finite.")
    return points, values


def _observation_noise(noise_scale: ArrayLike, /, *, count: int) -> Array:
    noise = jnp.asarray(noise_scale)
    if noise.ndim == 0:
        return jnp.broadcast_to(noise, (count,))
    if noise.shape != (count,):
        raise ValueError("Vector GP noise must align with observation points.")
    return noise


def _validate_inducing_design(points: Array, inducing: Array, /) -> None:
    if inducing.shape[1] != points.shape[1]:
        raise ValueError(
            "inducing_points and observation_points need equal coordinate size."
        )
    if not 0 < int(inducing.shape[0]) < int(points.shape[0]):
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
