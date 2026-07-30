#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, sqrt
from typing import Any, Literal

import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array

from ...._doc import DOC_KEY0
from ...._uncertainty import UncertaintySource, validate_uncertainty_source
from ..core._base import _AbstractOperatorModel
from ..core._keys import EvalKey
from ..core._operator import OperatorBatch, OperatorOutputSpec
from ..core._operator_distribution import (
    AbstractProbabilisticOperatorModel,
    GaussianOperatorDistribution,
)


class GaussianFunctionOperator(AbstractProbabilisticOperatorModel):
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
    scale_mode: Literal["learned", "fixed"]
    fixed_scale: float
    uncertainty_source: UncertaintySource
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
        scale_mode: Literal["learned", "fixed"] = "learned",
        fixed_scale: float = 1e-4,
        uncertainty_source: UncertaintySource = "observation",
    ):
        if not isinstance(base, _AbstractOperatorModel):
            raise TypeError("GaussianFunctionOperator base must be a neural operator.")
        rank = int(factor_rank)
        if rank < 0:
            raise ValueError("factor_rank must be non-negative.")
        if (
            not isfinite(float(min_scale))
            or not isfinite(float(factor_scale))
            or float(min_scale) <= 0.0
            or float(factor_scale) < 0.0
        ):
            raise ValueError(
                "Gaussian operator scales must be finite and positive/non-negative."
            )
        if scale_mode not in ("learned", "fixed"):
            raise ValueError("scale_mode must be 'learned' or 'fixed'.")
        if not isfinite(float(fixed_scale)) or float(fixed_scale) <= 0.0:
            raise ValueError("fixed_scale must be finite and positive.")
        output_spec = OperatorOutputSpec(out_channels)
        channels = 1 if out_channels == "scalar" else int(out_channels)
        parameters_per_channel = (2 if scale_mode == "learned" else 1) + rank
        required = channels * parameters_per_channel
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
        self.scale_mode = scale_mode
        self.fixed_scale = float(fixed_scale)
        self.uncertainty_source = validate_uncertainty_source(
            uncertainty_source,
            owner="GaussianFunctionOperator uncertainty_source",
        )
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
        parameters_per_channel = (
            2 if self.scale_mode == "learned" else 1
        ) + self.factor_rank
        expected = (
            batch.case_shape
            + batch.require_single_query().sample_shape
            + (channels * parameters_per_channel,)
        )
        if raw.shape != expected:
            raise ValueError(
                f"Gaussian operator parameter output must have shape {expected}; "
                f"got {raw.shape}."
            )
        parameters = raw.reshape(
            batch.case_shape
            + batch.require_single_query().sample_shape
            + (channels, parameters_per_channel)
        )
        mean = parameters[..., 0]
        if self.scale_mode == "learned":
            scale = self.min_scale + jnn.softplus(parameters[..., 1])
            factor_offset = 2
        else:
            scale = jnp.full_like(mean, self.fixed_scale)
            factor_offset = 1
        factors = parameters[..., factor_offset:]
        if self.factor_rank:
            factors = (self.factor_scale / sqrt(float(self.factor_rank))) * factors
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
            uncertainty_source=self.uncertainty_source,
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
