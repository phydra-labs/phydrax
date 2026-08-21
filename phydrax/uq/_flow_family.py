#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.flows import coupling_flow, triangular_spline_flow
from jaxtyping import Array


def build_default_flow(
    key: Array,
    data: Array,
    /,
    *,
    flow_layers: int,
    num_knots: int,
    nn_width: int,
    nn_depth: int,
) -> AbstractDistribution:
    """Initialize one unconditional spline flow from a sample matrix."""

    samples = jnp.asarray(data)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError("Flow initialization requires a non-empty matrix of samples.")
    if not jnp.issubdtype(samples.dtype, jnp.floating):
        raise TypeError("Flow samples must use a real floating dtype.")
    location = jnp.mean(samples, axis=0)
    scale = jnp.std(samples, axis=0)
    tolerance = jnp.sqrt(jnp.finfo(samples.dtype).eps) * jnp.maximum(
        jnp.ones_like(location), jnp.abs(location)
    )
    base = Normal(location, jnp.maximum(scale, tolerance))
    dimension = int(samples.shape[1])
    if dimension == 1:
        return triangular_spline_flow(
            key,
            base_dist=base,
            flow_layers=int(flow_layers),
            knots=int(num_knots),
            invert=True,
        )
    transformer = RationalQuadraticSpline(knots=int(num_knots), interval=3.0)
    return coupling_flow(
        key,
        base_dist=base,
        transformer=transformer,
        flow_layers=int(flow_layers),
        nn_width=int(nn_width),
        nn_depth=int(nn_depth),
        invert=True,
    )


def validate_flow(
    flow: AbstractDistribution,
    data: Array,
    key: Array,
    /,
) -> None:
    """Validate event shape and finite sample/density behavior."""

    samples = jnp.asarray(data)
    expected_shape = (int(samples.shape[-1]),)
    if flow.shape != expected_shape:
        raise ValueError(
            f"Flow event shape {flow.shape} does not match {expected_shape}."
        )
    if flow.cond_shape is not None:
        raise ValueError("Flow proposal must be unconditional.")
    data_log_density = flow.log_prob(samples)
    proposed, proposed_log_density = flow.sample_and_log_prob(
        key,
        sample_shape=(min(8, int(samples.shape[0])),),
    )
    if proposed.shape[1:] != expected_shape:
        raise ValueError("Flow proposal samples have an incompatible event shape.")
    if not bool(jnp.all(jnp.isfinite(data_log_density))):
        raise FloatingPointError("Flow density is nonfinite on its training data.")
    if not bool(jnp.all(jnp.isfinite(proposed))) or not bool(
        jnp.all(jnp.isfinite(proposed_log_density))
    ):
        raise FloatingPointError("Flow proposal produced a nonfinite sample or density.")


__all__ = ["build_default_flow", "validate_flow"]
