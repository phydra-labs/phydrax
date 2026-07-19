#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._predictive import PredictiveField, SampleAxis
from ._tinygp_backend import (
    exact_gp_cholesky,
    exact_gp_conditioner,
    exact_gp_log_probability,
    fitc_factors,
    gp_condition,
    gp_log_probability,
    KernelName,
    multi_output_gp_condition,
    multi_output_gp_log_probability,
    sparse_gp_condition,
    sparse_gp_conditioner,
    sparse_gp_log_probability,
    sparse_gp_log_probability_from_factors,
)


class GaussianProcessCondition(StrictModule):
    """Conditioned latent discrepancy at one fixed query design."""

    query_points: Array
    mean: Array
    covariance: Array
    variance: Array
    output_dims: tuple[str | None, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        query_points: ArrayLike,
        mean: ArrayLike,
        covariance: ArrayLike,
        variance: ArrayLike,
        output_dims: tuple[str | None, ...],
    ):
        points = _as_points(query_points)
        mean_array = _as_vector(mean, name="conditioned GP mean")
        covariance_array = jnp.asarray(covariance, dtype=float)
        variance_array = _as_vector(variance, name="conditioned GP variance")
        count = int(points.shape[0])
        if mean_array.shape != (count,) or variance_array.shape != (count,):
            raise ValueError("Conditioned GP moments must align with query points.")
        if covariance_array.shape != (count, count):
            raise ValueError(
                "Conditioned GP covariance must be square over query points."
            )
        if len(output_dims) != 1:
            raise ValueError("Scalar GP output currently requires one output dimension.")
        self.query_points = points
        self.mean = mean_array
        self.covariance = covariance_array
        self.variance = variance_array
        self.output_dims = tuple(output_dims)

    def sample(self, key: Array, /, *, num_samples: int) -> Array:
        """Draw coherent latent-discrepancy functions at all query points."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        cholesky = jnp.linalg.cholesky(self.covariance)
        noise = jr.normal(key, (count, self.mean.size), dtype=self.mean.dtype)
        return self.mean + noise @ cholesky.T

    def predictive_field(
        self,
        base_mean: ArrayLike,
        key: Array,
        /,
        *,
        num_samples: int,
        observation_variance: ArrayLike | None = None,
        sample_dim: str = "__phydra_uq_discrepancy",
    ) -> PredictiveField:
        """Add discrepancy draws to a physical mean and preserve observation variance."""
        base = _as_vector(base_mean, name="physical predictive mean")
        if base.shape != self.mean.shape:
            raise ValueError("base_mean must align with conditioned GP query points.")
        data = base + self.sample(key, num_samples=num_samples)
        valid_data = jnp.all(jnp.isfinite(data), axis=1)
        conditional = None
        if observation_variance is not None:
            variance = jnp.asarray(observation_variance, dtype=float)
            if bool(jnp.any(variance < 0.0)):
                raise ValueError("observation_variance must be non-negative.")
            variance = jnp.broadcast_to(variance, self.mean.shape)
            conditional = cx.Field(variance, dims=self.output_dims)
        return PredictiveField(
            cx.Field(data, dims=(sample_dim, *self.output_dims)),
            (SampleAxis(sample_dim, "epistemic"),),
            conditional_variance=conditional,
            valid=cx.Field(valid_data, dims=(sample_dim,)),
        )


class GaussianProcessConditioner(StrictModule):
    """Reusable residual projection and covariance for one fixed query design."""

    query_points: Array
    residual_projection: Array
    covariance: Array
    variance: Array
    output_dims: tuple[str | None, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        query_points: ArrayLike,
        residual_projection: ArrayLike,
        covariance: ArrayLike,
        variance: ArrayLike,
        output_dims: tuple[str | None, ...],
    ):
        points = _as_points(query_points)
        projection = jnp.asarray(residual_projection, dtype=float)
        covariance_array = jnp.asarray(covariance, dtype=float)
        variance_array = _as_vector(variance, name="conditioned GP variance")
        count = int(points.shape[0])
        if projection.ndim != 2 or int(projection.shape[0]) != count:
            raise ValueError("GP residual projection must have one row per query point.")
        if covariance_array.shape != (count, count) or variance_array.shape != (count,):
            raise ValueError("Conditioner moments must align with query points.")
        if len(output_dims) != 1:
            raise ValueError("Scalar GP output currently requires one output dimension.")
        self.query_points = points
        self.residual_projection = projection
        self.covariance = covariance_array
        self.variance = variance_array
        self.output_dims = tuple(output_dims)

    def condition(self, residual: ArrayLike, /) -> GaussianProcessCondition:
        """Project a new residual vector without rebuilding any GP factors."""
        values = _as_vector(residual, name="GP residual")
        if int(values.shape[0]) != int(self.residual_projection.shape[1]):
            raise ValueError("GP residual must align with conditioner observations.")
        return GaussianProcessCondition(
            query_points=self.query_points,
            mean=self.residual_projection @ values,
            covariance=self.covariance,
            variance=self.variance,
            output_dims=self.output_dims,
        )


class ExactGaussianProcessFactor(StrictModule):
    """Reusable exact covariance factor for fixed GP hyperparameters."""

    observation_points: Array
    cholesky: Array
    amplitude: Array
    length_scale: Array
    jitter: Array
    kernel: KernelName = eqx.field(static=True)

    def __init__(
        self,
        observation_points: ArrayLike | cx.Field,
        /,
        *,
        amplitude: ArrayLike,
        length_scale: ArrayLike,
        noise_scale: ArrayLike,
        kernel: KernelName = "matern32",
        jitter: ArrayLike = 1e-8,
    ):
        points = _validated_factor_points(observation_points)
        amplitude_array = _positive_scalar(amplitude, name="amplitude")
        length_scale_array = _positive_length_scale(
            length_scale,
            coordinate_size=int(points.shape[1]),
        )
        noise_array = _nonnegative_scalar(noise_scale, name="noise_scale")
        jitter_array = _positive_scalar(jitter, name="jitter")
        kernel_name = _validated_kernel(kernel)
        self.observation_points = points
        self.cholesky = exact_gp_cholesky(
            points,
            amplitude=amplitude_array,
            length_scale=length_scale_array,
            noise_scale=noise_array,
            kernel=kernel_name,
            jitter=jitter_array,
        )
        self.amplitude = amplitude_array
        self.length_scale = length_scale_array
        self.kernel = kernel_name
        self.jitter = jitter_array

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
            amplitude=self.amplitude,
            length_scale=self.length_scale,
            kernel=self.kernel,
            jitter=self.jitter,
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


class SparseGaussianProcessFactor(StrictModule):
    """Reusable FITC factors for fixed GP hyperparameters and inducing points."""

    observation_points: Array
    inducing_points: Array
    features: Array
    diagonal: Array
    correction_cholesky: Array
    inducing_cholesky: Array
    amplitude: Array
    length_scale: Array
    jitter: Array
    kernel: KernelName = eqx.field(static=True)

    def __init__(
        self,
        observation_points: ArrayLike | cx.Field,
        inducing_points: ArrayLike | cx.Field,
        /,
        *,
        amplitude: ArrayLike,
        length_scale: ArrayLike,
        noise_scale: ArrayLike,
        kernel: KernelName = "matern32",
        jitter: ArrayLike = 1e-8,
    ):
        points = _validated_factor_points(observation_points)
        inducing = _validated_factor_points(inducing_points)
        if inducing.shape[1] != points.shape[1]:
            raise ValueError(
                "inducing_points and observation_points need equal coordinate size."
            )
        if not 0 < int(inducing.shape[0]) < int(points.shape[0]):
            raise ValueError(
                "Sparse GP requires fewer inducing points than observations."
            )
        amplitude_array = _positive_scalar(amplitude, name="amplitude")
        length_scale_array = _positive_length_scale(
            length_scale,
            coordinate_size=int(points.shape[1]),
        )
        noise_array = _nonnegative_scalar(noise_scale, name="noise_scale")
        jitter_array = _positive_scalar(jitter, name="jitter")
        kernel_name = _validated_kernel(kernel)
        factors = fitc_factors(
            points,
            inducing,
            amplitude=amplitude_array,
            length_scale=length_scale_array,
            noise_scale=noise_array,
            kernel=kernel_name,
            jitter=jitter_array,
        )
        self.observation_points = points
        self.inducing_points = inducing
        self.features = factors[0]
        self.diagonal = factors[1]
        self.correction_cholesky = factors[2]
        self.inducing_cholesky = factors[3]
        self.amplitude = amplitude_array
        self.length_scale = length_scale_array
        self.kernel = kernel_name
        self.jitter = jitter_array

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
            amplitude=self.amplitude,
            length_scale=self.length_scale,
            kernel=self.kernel,
            jitter=self.jitter,
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


class MultiOutputGaussianProcessCondition(StrictModule):
    """Conditioned correlated discrepancy over query points and output channels."""

    query_points: Array
    mean: Array
    covariance: Array
    variance: Array
    output_dims: tuple[str | None, str | None] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        query_points: ArrayLike,
        mean: ArrayLike,
        covariance: ArrayLike,
        variance: ArrayLike,
        output_dims: tuple[str | None, str | None],
        output_names: tuple[str, ...],
    ):
        points = _as_points(query_points)
        mean_array = _as_matrix(mean, name="conditioned multi-output GP mean")
        variance_array = _as_matrix(variance, name="conditioned multi-output GP variance")
        covariance_array = jnp.asarray(covariance, dtype=float)
        count, outputs = mean_array.shape
        if int(points.shape[0]) != count or variance_array.shape != mean_array.shape:
            raise ValueError("Conditioned GP moments must align with query points.")
        if covariance_array.shape != (count * outputs, count * outputs):
            raise ValueError(
                "Conditioned multi-output covariance has incompatible shape."
            )
        if len(output_dims) != 2 or len(output_names) != outputs:
            raise ValueError("Output dimensions and names must align with GP outputs.")
        self.query_points = points
        self.mean = mean_array
        self.covariance = covariance_array
        self.variance = variance_array
        self.output_dims = tuple(output_dims)
        self.output_names = tuple(output_names)

    def sample(self, key: Array, /, *, num_samples: int) -> Array:
        """Draw coherent correlated functions over points and outputs."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        cholesky = jnp.linalg.cholesky(self.covariance)
        noise = jr.normal(
            key,
            (count, self.mean.size),
            dtype=self.mean.dtype,
        )
        return (self.mean.reshape((-1,)) + noise @ cholesky.T).reshape(
            (count, *self.mean.shape)
        )

    def predictive_field(
        self,
        base_mean: ArrayLike,
        key: Array,
        /,
        *,
        num_samples: int,
        observation_variance: ArrayLike | None = None,
        sample_dim: str = "__phydra_uq_discrepancy",
    ) -> PredictiveField:
        """Add correlated discrepancy draws to a multi-output physical mean."""
        base = _as_matrix(base_mean, name="multi-output physical predictive mean")
        if base.shape != self.mean.shape:
            raise ValueError("base_mean must align with conditioned GP outputs.")
        data = base + self.sample(key, num_samples=num_samples)
        valid_data = jnp.all(jnp.isfinite(data), axis=(1, 2))
        conditional = None
        if observation_variance is not None:
            variance = jnp.asarray(observation_variance, dtype=float)
            if bool(jnp.any(variance < 0.0)):
                raise ValueError("observation_variance must be non-negative.")
            variance = jnp.broadcast_to(variance, self.mean.shape)
            conditional = cx.Field(variance, dims=self.output_dims)
        return PredictiveField(
            cx.Field(data, dims=(sample_dim, *self.output_dims)),
            (SampleAxis(sample_dim, "epistemic"),),
            conditional_variance=conditional,
            valid=cx.Field(valid_data, dims=(sample_dim,)),
        )


class ExactGaussianProcessDiscrepancy(StrictModule):
    """Exact scalar-output GP model for additive model-form discrepancy."""

    observation_points: Array
    observations: Array
    jitter: Array
    kernel: KernelName = eqx.field(static=True)

    def __init__(
        self,
        observation_points: ArrayLike | cx.Field,
        observations: ArrayLike | cx.Field,
        /,
        *,
        kernel: KernelName = "matern32",
        jitter: float = 1e-8,
    ):
        points = _field_data(observation_points)
        values = _as_vector(_field_data(observations), name="GP observations")
        points = _as_points(points)
        if int(points.shape[0]) != int(values.shape[0]):
            raise ValueError("GP observations must align with observation points.")
        if not bool(jnp.all(jnp.isfinite(points))) or not bool(
            jnp.all(jnp.isfinite(values))
        ):
            raise ValueError("GP observations and points must be finite.")
        if kernel not in ("exp_squared", "matern32", "matern52"):
            raise ValueError(f"Unknown GP kernel {kernel!r}.")
        jitter_value = float(jitter)
        if not jnp.isfinite(jitter_value) or jitter_value <= 0.0:
            raise ValueError("jitter must be finite and positive.")
        self.observation_points = points
        self.observations = values
        self.kernel = kernel
        self.jitter = jnp.asarray(jitter_value, dtype=float)

    def residual(self, physical_mean: ArrayLike, /) -> Array:
        """Return observations minus the physical-model mean."""
        mean = _as_vector(physical_mean, name="physical observation mean")
        if mean.shape != self.observations.shape:
            raise ValueError("physical_mean must align with GP observations.")
        return self.observations - mean

    def factor(
        self,
        *,
        amplitude: ArrayLike,
        length_scale: ArrayLike,
        noise_scale: ArrayLike,
    ) -> ExactGaussianProcessFactor:
        """Factor fixed GP hyperparameters once for repeated residual evaluations."""
        return ExactGaussianProcessFactor(
            self.observation_points,
            amplitude=amplitude,
            length_scale=length_scale,
            noise_scale=noise_scale,
            kernel=self.kernel,
            jitter=self.jitter,
        )

    def log_marginal_likelihood(
        self,
        physical_mean: ArrayLike,
        /,
        *,
        amplitude: ArrayLike,
        length_scale: ArrayLike,
        noise_scale: ArrayLike,
    ) -> Array:
        """Marginalize latent discrepancy values under a Gaussian likelihood."""
        return gp_log_probability(
            self.observation_points,
            self.residual(physical_mean),
            amplitude=amplitude,
            length_scale=length_scale,
            noise_scale=noise_scale,
            kernel=self.kernel,
            jitter=self.jitter,
        )

    def condition(
        self,
        physical_mean: ArrayLike,
        query_points: ArrayLike | cx.Field,
        /,
        *,
        amplitude: ArrayLike,
        length_scale: ArrayLike,
        noise_scale: ArrayLike,
        output_dim: str | None = "point",
    ) -> GaussianProcessCondition:
        """Condition the latent discrepancy on residual observations."""
        query_data = _field_data(query_points)
        output_dims = _query_output_dims(query_points, output_dim=output_dim)
        mean, covariance, variance = gp_condition(
            self.observation_points,
            self.residual(physical_mean),
            query_data,
            amplitude=amplitude,
            length_scale=length_scale,
            noise_scale=noise_scale,
            kernel=self.kernel,
            jitter=self.jitter,
        )
        return GaussianProcessCondition(
            query_points=query_data,
            mean=mean,
            covariance=covariance,
            variance=variance,
            output_dims=output_dims,
        )


class MultiOutputGaussianProcessDiscrepancy(StrictModule):
    """Exact separable GP with an explicit positive-definite output covariance."""

    observation_points: Array
    observations: Array
    output_covariance: Array
    jitter: Array
    kernel: KernelName = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        observation_points: ArrayLike | cx.Field,
        observations: ArrayLike | cx.Field,
        /,
        *,
        output_covariance: ArrayLike,
        output_names: tuple[str, ...] | None = None,
        kernel: KernelName = "matern32",
        jitter: float = 1e-8,
    ):
        points = _as_points(_field_data(observation_points))
        values = _as_matrix(
            _field_data(observations), name="multi-output GP observations"
        )
        if int(points.shape[0]) != int(values.shape[0]):
            raise ValueError("GP observations must align with observation points.")
        outputs = int(values.shape[1])
        covariance = jnp.asarray(output_covariance, dtype=float)
        if covariance.shape != (outputs, outputs):
            raise ValueError("output_covariance must have one row and column per output.")
        if not bool(jnp.all(jnp.isfinite(covariance))) or not bool(
            jnp.allclose(covariance, covariance.T)
        ):
            raise ValueError("output_covariance must be finite and symmetric.")
        if not bool(jnp.all(jnp.linalg.eigvalsh(covariance) > 0.0)):
            raise ValueError("output_covariance must be positive definite.")
        names = (
            tuple(f"output_{index}" for index in range(outputs))
            if output_names is None
            else tuple(output_names)
        )
        if (
            len(names) != outputs
            or len(set(names)) != outputs
            or any(not name for name in names)
        ):
            raise ValueError(
                "output_names must contain one distinct non-empty name per output."
            )
        if kernel not in ("exp_squared", "matern32", "matern52"):
            raise ValueError(f"Unknown GP kernel {kernel!r}.")
        jitter_value = float(jitter)
        if not jnp.isfinite(jitter_value) or jitter_value <= 0.0:
            raise ValueError("jitter must be finite and positive.")
        if not bool(jnp.all(jnp.isfinite(points))) or not bool(
            jnp.all(jnp.isfinite(values))
        ):
            raise ValueError("GP observations and points must be finite.")
        self.observation_points = points
        self.observations = values
        self.output_covariance = covariance
        self.output_names = names
        self.kernel = kernel
        self.jitter = jnp.asarray(jitter_value, dtype=float)

    @property
    def num_outputs(self) -> int:
        return int(self.observations.shape[1])

    def residual(self, physical_mean: ArrayLike, /) -> Array:
        mean = _as_matrix(physical_mean, name="multi-output physical observation mean")
        if mean.shape != self.observations.shape:
            raise ValueError("physical_mean must align with GP observations.")
        return self.observations - mean

    def log_marginal_likelihood(
        self,
        physical_mean: ArrayLike,
        /,
        *,
        amplitude: ArrayLike,
        length_scale: ArrayLike,
        noise_scale: ArrayLike,
    ) -> Array:
        """Marginalize a correlated multi-output latent discrepancy."""
        return multi_output_gp_log_probability(
            self.observation_points,
            self.residual(physical_mean),
            amplitude=amplitude,
            length_scale=length_scale,
            output_covariance=self.output_covariance,
            noise_scale=noise_scale,
            kernel=self.kernel,
            jitter=self.jitter,
        )

    def condition(
        self,
        physical_mean: ArrayLike,
        query_points: ArrayLike | cx.Field,
        /,
        *,
        amplitude: ArrayLike,
        length_scale: ArrayLike,
        noise_scale: ArrayLike,
        point_dim: str | None = "point",
        output_dim: str | None = "output",
    ) -> MultiOutputGaussianProcessCondition:
        """Condition all correlated output channels at one query design."""
        query_data = _field_data(query_points)
        point_dims = _query_output_dims(query_points, output_dim=point_dim)
        mean, covariance, variance = multi_output_gp_condition(
            self.observation_points,
            self.residual(physical_mean),
            query_data,
            amplitude=amplitude,
            length_scale=length_scale,
            output_covariance=self.output_covariance,
            noise_scale=noise_scale,
            kernel=self.kernel,
            jitter=self.jitter,
        )
        return MultiOutputGaussianProcessCondition(
            query_points=query_data,
            mean=mean,
            covariance=covariance,
            variance=variance,
            output_dims=(point_dims[0], output_dim),
            output_names=self.output_names,
        )


class SparseGaussianProcessDiscrepancy(StrictModule):
    """Scalar-output FITC discrepancy with explicit inducing points."""

    observation_points: Array
    observations: Array
    inducing_points: Array
    jitter: Array
    kernel: KernelName = eqx.field(static=True)

    def __init__(
        self,
        observation_points: ArrayLike | cx.Field,
        observations: ArrayLike | cx.Field,
        inducing_points: ArrayLike | cx.Field,
        /,
        *,
        kernel: KernelName = "matern32",
        jitter: float = 1e-8,
    ):
        points = _as_points(_field_data(observation_points))
        values = _as_vector(_field_data(observations), name="sparse GP observations")
        inducing = _as_points(_field_data(inducing_points))
        if int(points.shape[0]) != int(values.shape[0]):
            raise ValueError("GP observations must align with observation points.")
        if inducing.shape[1] != points.shape[1]:
            raise ValueError(
                "inducing_points and observation_points need equal coordinate size."
            )
        if not 0 < int(inducing.shape[0]) < int(points.shape[0]):
            raise ValueError(
                "Sparse GP requires fewer inducing points than observations."
            )
        if not bool(
            jnp.all(
                jnp.isfinite(
                    jnp.concatenate(
                        [points.reshape((-1,)), inducing.reshape((-1,)), values]
                    )
                )
            )
        ):
            raise ValueError(
                "GP observations, points, and inducing points must be finite."
            )
        if kernel not in ("exp_squared", "matern32", "matern52"):
            raise ValueError(f"Unknown GP kernel {kernel!r}.")
        jitter_value = float(jitter)
        if not jnp.isfinite(jitter_value) or jitter_value <= 0.0:
            raise ValueError("jitter must be finite and positive.")
        self.observation_points = points
        self.observations = values
        self.inducing_points = inducing
        self.kernel = kernel
        self.jitter = jnp.asarray(jitter_value, dtype=float)

    @classmethod
    def from_evenly_spaced_subset(
        cls,
        observation_points: ArrayLike | cx.Field,
        observations: ArrayLike | cx.Field,
        /,
        *,
        num_inducing: int,
        kernel: KernelName = "matern32",
        jitter: float = 1e-8,
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
        return cls(
            observation_points,
            observations,
            points[indices],
            kernel=kernel,
            jitter=jitter,
        )

    @property
    def num_inducing(self) -> int:
        return int(self.inducing_points.shape[0])

    @property
    def factor_storage_elements(self) -> int:
        """Dominant FITC factor storage, versus n² for the exact GP."""
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
        amplitude: ArrayLike,
        length_scale: ArrayLike,
        noise_scale: ArrayLike,
    ) -> SparseGaussianProcessFactor:
        """Build reusable FITC factors for repeated residual evaluations."""
        return SparseGaussianProcessFactor(
            self.observation_points,
            self.inducing_points,
            amplitude=amplitude,
            length_scale=length_scale,
            noise_scale=noise_scale,
            kernel=self.kernel,
            jitter=self.jitter,
        )

    def log_marginal_likelihood(
        self,
        physical_mean: ArrayLike,
        /,
        *,
        amplitude: ArrayLike,
        length_scale: ArrayLike,
        noise_scale: ArrayLike,
    ) -> Array:
        """Evaluate the matrix-free FITC marginal likelihood."""
        return sparse_gp_log_probability(
            self.observation_points,
            self.inducing_points,
            self.residual(physical_mean),
            amplitude=amplitude,
            length_scale=length_scale,
            noise_scale=noise_scale,
            kernel=self.kernel,
            jitter=self.jitter,
        )

    def condition(
        self,
        physical_mean: ArrayLike,
        query_points: ArrayLike | cx.Field,
        /,
        *,
        amplitude: ArrayLike,
        length_scale: ArrayLike,
        noise_scale: ArrayLike,
        output_dim: str | None = "point",
    ) -> GaussianProcessCondition:
        """Condition the FITC latent discrepancy at query points."""
        query_data = _field_data(query_points)
        output_dims = _query_output_dims(query_points, output_dim=output_dim)
        mean, covariance, variance = sparse_gp_condition(
            self.observation_points,
            self.inducing_points,
            self.residual(physical_mean),
            query_data,
            amplitude=amplitude,
            length_scale=length_scale,
            noise_scale=noise_scale,
            kernel=self.kernel,
            jitter=self.jitter,
        )
        return GaussianProcessCondition(
            query_points=query_data,
            mean=mean,
            covariance=covariance,
            variance=variance,
            output_dims=output_dims,
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


def _as_matrix(value: ArrayLike, /, *, name: str) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim != 2 or int(array.shape[1]) < 1:
        raise ValueError(f"{name} must have shape (point, output).")
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
    if not bool(jnp.all(jnp.isfinite(points))):
        raise ValueError("GP factor points must be finite.")
    return points


def _positive_scalar(value: ArrayLike, /, *, name: str) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape != () or not bool(jnp.isfinite(array)) or not bool(array > 0.0):
        raise ValueError(f"{name} must be a finite positive scalar.")
    return array


def _nonnegative_scalar(value: ArrayLike, /, *, name: str) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape != () or not bool(jnp.isfinite(array)) or not bool(array >= 0.0):
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    return array


def _positive_length_scale(value: ArrayLike, /, *, coordinate_size: int) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape not in ((), (coordinate_size,)):
        raise ValueError(
            "length_scale must be scalar or have one value per point coordinate."
        )
    if not bool(jnp.all(jnp.isfinite(array))) or not bool(jnp.all(array > 0.0)):
        raise ValueError("length_scale must contain finite positive values.")
    return array


def _validated_kernel(kernel: KernelName, /) -> KernelName:
    if kernel not in ("exp_squared", "matern32", "matern52"):
        raise ValueError(f"Unknown GP kernel {kernel!r}.")
    return kernel


__all__ = [
    "ExactGaussianProcessDiscrepancy",
    "ExactGaussianProcessFactor",
    "GaussianProcessCondition",
    "GaussianProcessConditioner",
    "MultiOutputGaussianProcessCondition",
    "MultiOutputGaussianProcessDiscrepancy",
    "SparseGaussianProcessDiscrepancy",
    "SparseGaussianProcessFactor",
]
