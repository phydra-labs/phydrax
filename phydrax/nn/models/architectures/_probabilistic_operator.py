#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import sqrt
from typing import Any, Literal

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from ...._doc import DOC_KEY0
from ..core._base import _AbstractOperatorModel
from ..core._keys import EvalKey
from ..core._operator import OperatorBatch, OperatorOutputSpec
from ..core._operator_distribution import GaussianOperatorDistribution


class GaussianFunctionOperator(_AbstractOperatorModel):
    """Neural operator with a coherent diagonal-plus-low-rank Gaussian field head.

    The wrapped operator emits, per output component, one mean, one unconstrained
    marginal scale, and ``factor_rank`` shared-latent loadings. A single latent draw
    is reused over every query point, which produces coherent function samples rather
    than independent calls at individual coordinates.
    """

    base: _AbstractOperatorModel
    output_spec: OperatorOutputSpec
    factor_rank: int
    min_scale: float
    factor_scale: float
    in_size: Any
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        base: _AbstractOperatorModel,
        /,
        *,
        out_channels: int | Literal["scalar"] = "scalar",
        factor_rank: int = 0,
        min_scale: float = 1e-4,
        factor_scale: float = 1.0,
    ):
        if not isinstance(base, _AbstractOperatorModel):
            raise TypeError("GaussianFunctionOperator base must be a neural operator.")
        rank = int(factor_rank)
        if rank < 0:
            raise ValueError("factor_rank must be non-negative.")
        if float(min_scale) <= 0.0 or float(factor_scale) < 0.0:
            raise ValueError("Gaussian operator scales must be positive/non-negative.")
        output_spec = OperatorOutputSpec(out_channels)
        channels = 1 if out_channels == "scalar" else int(out_channels)
        required = channels * (2 + rank)
        base_channels = base.operator_output_specs["output"].channels
        if base_channels == "scalar" or int(base_channels) != required:
            raise ValueError(
                "Gaussian operator base must emit "
                f"{required} channel-last parameters; got {base_channels!r}."
            )
        self.base = base
        self.output_spec = output_spec
        self.factor_rank = rank
        self.min_scale = float(min_scale)
        self.factor_scale = float(factor_scale)
        self.in_size = base.in_size
        self.out_size = out_channels

    @property
    def operator_output_specs(self) -> dict[str, OperatorOutputSpec]:
        return {"output": self.output_spec}

    def distribution(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> GaussianOperatorDistribution:
        raw = jnp.asarray(self.base.__call_operator_batch__(batch, key=key))
        channels = 1 if self.out_size == "scalar" else int(self.out_size)
        expected = batch.case_shape + batch.require_single_query().sample_shape + (
            channels * (2 + self.factor_rank),
        )
        if raw.shape != expected:
            raise ValueError(
                f"Gaussian operator parameter output must have shape {expected}; "
                f"got {raw.shape}."
            )
        parameters = raw.reshape(
            batch.case_shape
            + batch.require_single_query().sample_shape
            + (channels, 2 + self.factor_rank)
        )
        mean = parameters[..., 0]
        scale = self.min_scale + jnn.softplus(parameters[..., 1])
        factors = parameters[..., 2:]
        if self.factor_rank:
            factors = (
                self.factor_scale / sqrt(float(self.factor_rank))
            ) * factors
        if self.out_size == "scalar":
            mean = mean[..., 0]
            scale = scale[..., 0]
            factors = factors[..., 0, :]
        return GaussianOperatorDistribution(
            mean=mean,
            scale=scale,
            factors=factors,
            query=batch.require_single_query(),
            output_spec=self.output_spec,
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        return self.distribution(batch, key=key).mean

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("GaussianFunctionOperator requires an OperatorBatch.")
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
            sample_key, (count,)
        )

    def sample_predictive(
        self,
        batch: OperatorBatch,
        /,
        *,
        num_samples: int,
        key: Array,
        sample_dim: str = "__phydra_uq_aleatoric",
    ) -> Any:
        """Return the native coordinate-aware UQ container for sampled functions."""
        from ....uq._operator import operator_predictive_from_samples
        from ....uq._predictive import SampleAxis

        parameter_key, sample_key = jr.split(key)
        distribution = self.distribution(batch, key=parameter_key)
        samples = distribution.sample(sample_key, (int(num_samples),))
        return operator_predictive_from_samples(
            samples,
            batch,
            self.output_spec,
            sample_axes=(SampleAxis(sample_dim, "observation"),),
            field_name="output",
            query_name=batch.single_query_name(),
        )


def gaussian_operator_nll(
    model: GaussianFunctionOperator,
    batch: OperatorBatch,
    target: Array,
    /,
    *,
    key: EvalKey = DOC_KEY0,
    reduction: str = "mean",
) -> Array:
    """Exact masked Gaussian field negative log-likelihood training objective."""
    return model.distribution(batch, key=key).negative_log_likelihood(
        target, reduction=reduction
    )


__all__ = ["GaussianFunctionOperator", "gaussian_operator_nll"]
