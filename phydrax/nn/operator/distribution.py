#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import prod

import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import AbstractAttribute, StrictModule
from ..._uncertainty import UncertaintySource, validate_uncertainty_source
from .._keys import EvalKey
from .data import (
    FunctionSamples,
    OperatorBatch,
    OperatorOutputSpec,
    OperatorPrediction,
)
from .engine import AbstractOperatorModel


class AbstractOperatorDistribution(StrictModule):
    """Distribution over one complete operator-output field per physical case."""

    query: AbstractAttribute[FunctionSamples]
    output_spec: AbstractAttribute[OperatorOutputSpec]
    case_axes: AbstractAttribute[tuple[str, ...]]
    case_shape: AbstractAttribute[tuple[int, ...]]
    uncertainty_source: AbstractAttribute[UncertaintySource]

    @property
    @abstractmethod
    def location(self) -> Array:
        """Deterministic representative; not necessarily the distribution mean."""
        raise NotImplementedError

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.query.sample_shape + self.output_spec.channel_shape

    @property
    def event_size(self) -> int:
        return prod(self.event_shape)

    @abstractmethod
    def sample(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, target: Array, /) -> Array:
        raise NotImplementedError

    def negative_log_likelihood(
        self,
        target: Array,
        /,
        *,
        reduction: str = "mean",
    ) -> Array:
        values = -self.log_prob(target)
        if reduction == "none":
            return values
        if reduction == "sum":
            return jnp.sum(values)
        if reduction == "mean":
            return jnp.mean(values)
        raise ValueError(
            "Operator distribution reduction must be 'none', 'sum', or 'mean'."
        )

    def location_prediction(self) -> OperatorPrediction:
        return OperatorPrediction.from_field(
            "output",
            self.location,
            "query",
            self.query,
            spec=self.output_spec,
            case_axes=self.case_axes,
            case_shape=self.case_shape,
        )


class AbstractProbabilisticOperatorModel(AbstractOperatorModel):
    """Neural operator whose primary output is a complete-field distribution."""

    @abstractmethod
    def distribution(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> AbstractOperatorDistribution:
        raise NotImplementedError

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        return self.distribution(batch, key=key).location

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError(f"{type(self).__name__} requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)

    def sample(
        self,
        batch: OperatorBatch,
        /,
        *,
        num_samples: int,
        key: Array,
    ) -> Array:
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        parameter_key, sample_key = jr.split(key)
        return self.distribution(batch, key=parameter_key).sample(
            sample_key,
            (count,),
        )

    def sample_predictive(
        self,
        batch: OperatorBatch,
        /,
        *,
        num_samples: int,
        key: Array,
        sample_dim: str | None = None,
    ):
        """Return coordinate-aware samples labeled by their uncertainty source."""
        from ...uq._operator import operator_predictive_from_samples
        from ...uq._predictive import SampleAxis

        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        parameter_key, sample_key = jr.split(key)
        distribution = self.distribution(batch, key=parameter_key)
        source = distribution.uncertainty_source
        default_dim = (
            "__phydra_uq_aleatoric"
            if source == "observation"
            else f"__phydra_uq_{source}"
        )
        dim = default_dim if sample_dim is None else sample_dim
        samples = distribution.sample(sample_key, (count,))
        return operator_predictive_from_samples(
            samples,
            batch,
            distribution.output_spec,
            sample_axes=(SampleAxis(dim, source),),
            field_name="output",
            query_name=batch.single_query_name(),
        )


class GaussianOperatorDistribution(AbstractOperatorDistribution):
    """Diagonal-plus-low-rank Gaussian distribution over complete output fields."""

    mean: Array
    scale: Array
    factors: Array
    query: FunctionSamples
    output_spec: OperatorOutputSpec
    case_axes: tuple[str, ...]
    case_shape: tuple[int, ...]
    uncertainty_source: UncertaintySource

    def __init__(
        self,
        *,
        mean: Array,
        scale: Array,
        factors: Array | None,
        query: FunctionSamples,
        output_spec: OperatorOutputSpec,
        case_axes: tuple[str, ...] = (),
        case_shape: tuple[int, ...] = (),
        uncertainty_source: UncertaintySource = "observation",
    ):
        mean_array = jnp.asarray(mean)
        scale_array = jnp.asarray(scale)
        axes = tuple(str(axis) for axis in case_axes)
        cases = tuple(int(size) for size in case_shape)
        expected = cases + query.sample_shape + output_spec.channel_shape
        if mean_array.shape != expected or scale_array.shape != expected:
            raise ValueError(
                "Gaussian operator mean and scale must match case/query/output shape "
                f"{expected}; got {mean_array.shape} and {scale_array.shape}."
            )
        if len(axes) != len(cases):
            raise ValueError("Gaussian operator case axes and shape ranks differ.")
        if factors is None:
            factor_array = jnp.zeros(expected + (0,), dtype=mean_array.dtype)
        else:
            factor_array = jnp.asarray(factors)
            if factor_array.ndim != mean_array.ndim + 1:
                raise ValueError(
                    "Gaussian operator factors require one trailing rank axis."
                )
            if factor_array.shape[:-1] != expected:
                raise ValueError("Gaussian operator factors must align with the mean.")
        self.mean = mean_array
        self.scale = scale_array
        self.factors = factor_array
        self.query = query
        self.output_spec = output_spec
        self.case_axes = axes
        self.case_shape = cases
        self.uncertainty_source = validate_uncertainty_source(
            uncertainty_source,
            owner="GaussianOperatorDistribution uncertainty_source",
        )

    @property
    def rank(self) -> int:
        return int(self.factors.shape[-1])

    @property
    def location(self) -> Array:
        return self.mean

    def _flat_mask(self) -> Array:
        mask = self.query.mask_array(case_shape=self.case_shape)
        if self.output_spec.channels != "scalar":
            mask = jnp.broadcast_to(mask[..., None], self.mean.shape)
        return jnp.asarray(mask, dtype=bool).reshape((-1, self.event_size))

    def marginal_variance(self) -> Array:
        """Return pointwise variance including the shared latent factors."""
        variance = self.scale**2 + jnp.sum(self.factors**2, axis=-1)
        mask = self._flat_mask().reshape(self.mean.shape)
        return jnp.where(mask, variance, 0.0)

    def dense_covariance(self) -> Array:
        """Materialize per-case covariance over flattened query/output events."""
        cases = prod(self.case_shape) if self.case_shape else 1
        scale = self.scale.reshape((cases, self.event_size))
        factors = self.factors.reshape((cases, self.event_size, self.rank))
        mask = self._flat_mask()
        diagonal = jax.vmap(jnp.diag)(jnp.where(mask, scale**2, 0.0))
        covariance = diagonal + oe.contract(
            "cer,cfr->cef", factors * mask[..., None], factors * mask[..., None]
        )
        return covariance.reshape(self.case_shape + (self.event_size, self.event_size))

    def sample(
        self,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw coherent full-function samples with leading sample dimensions."""
        shape = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("Gaussian operator sample dimensions must be positive.")
        samples = prod(shape) if shape else 1
        cases = prod(self.case_shape) if self.case_shape else 1
        mean = self.mean.reshape((cases, self.event_size))
        scale = self.scale.reshape((cases, self.event_size))
        factors = self.factors.reshape((cases, self.event_size, self.rank))
        diagonal_key, latent_key = jr.split(key)
        diagonal_noise = jr.normal(
            diagonal_key,
            (samples, cases, self.event_size),
            dtype=self.mean.dtype,
        )
        values = mean[None, ...] + scale[None, ...] * diagonal_noise
        if self.rank:
            latent_noise = jr.normal(
                latent_key,
                (samples, cases, self.rank),
                dtype=self.mean.dtype,
            )
            values = values + oe.contract("scr,cer->sce", latent_noise, factors)
        mask = self._flat_mask()
        values = jnp.where(mask[None, ...], values, 0.0)
        return values.reshape(shape + self.case_shape + self.event_shape)

    def log_prob(self, target: Array, /) -> Array:
        """Return one exact log density per case using Woodbury identities."""
        target_array = jnp.asarray(target)
        if target_array.shape != self.mean.shape:
            raise ValueError(
                f"Gaussian operator target must have shape {self.mean.shape}; "
                f"got {target_array.shape}."
            )
        cases = prod(self.case_shape) if self.case_shape else 1
        mean = self.mean.reshape((cases, self.event_size))
        target_flat = target_array.reshape((cases, self.event_size))
        scale = self.scale.reshape((cases, self.event_size))
        factors = self.factors.reshape((cases, self.event_size, self.rank))
        mask = self._flat_mask()

        def case_log_prob(
            mean_case: Array,
            target_case: Array,
            scale_case: Array,
            factor_case: Array,
            mask_case: Array,
        ) -> Array:
            residual = jnp.where(mask_case, target_case - mean_case, 0.0)
            inverse_diagonal = jnp.where(mask_case, jnp.reciprocal(scale_case**2), 0.0)
            quadratic = jnp.sum(residual**2 * inverse_diagonal)
            log_determinant = jnp.sum(
                jnp.where(mask_case, 2.0 * jnp.log(scale_case), 0.0)
            )
            if self.rank:
                weighted_factors = inverse_diagonal[:, None] * factor_case
                correction = (
                    jnp.eye(self.rank, dtype=mean_case.dtype)
                    + factor_case.T @ weighted_factors
                )
                cholesky = jnp.linalg.cholesky(correction)
                projected = factor_case.T @ (inverse_diagonal * residual)
                solved = jax.scipy.linalg.solve_triangular(
                    cholesky, projected, lower=True
                )
                quadratic = quadratic - jnp.sum(solved**2)
                log_determinant = log_determinant + 2.0 * jnp.sum(
                    jnp.log(jnp.diag(cholesky))
                )
            count = jnp.sum(mask_case)
            return -0.5 * (quadratic + log_determinant + count * jnp.log(2.0 * jnp.pi))

        result = jax.vmap(case_log_prob)(mean, target_flat, scale, factors, mask)
        return result.reshape(self.case_shape)

    def mean_prediction(self) -> OperatorPrediction:
        return self.location_prediction()


__all__ = [
    "AbstractOperatorDistribution",
    "AbstractProbabilisticOperatorModel",
    "GaussianOperatorDistribution",
]
