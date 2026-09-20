#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ..nn.flows import (
    AbstractFlowDistribution,
    coupling_flow,
    NormalFlowDistribution,
    triangular_flow,
)


def build_default_flow(
    key: Array,
    data: Array,
    /,
    *,
    flow_layers: int,
    nn_width: int,
    nn_depth: int,
) -> AbstractFlowDistribution:
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
    base = NormalFlowDistribution(location, jnp.maximum(scale, tolerance))
    dimension = samples.shape[1]
    if dimension == 1:
        return triangular_flow(
            key,
            base_dist=base,
            flow_layers=int(flow_layers),
        )
    return coupling_flow(
        key,
        base_dist=base,
        flow_layers=int(flow_layers),
        nn_width=int(nn_width),
        nn_depth=int(nn_depth),
        invert=True,
    )


def validate_flow(
    flow: AbstractFlowDistribution,
    data: Array,
    key: Array,
    /,
) -> None:
    """Validate event shape and finite sample/density behavior."""

    samples = jnp.asarray(data)
    expected_shape = (samples.shape[-1],)
    if flow.shape != expected_shape:
        raise ValueError(
            f"Flow event shape {flow.shape} does not match {expected_shape}."
        )
    if flow.cond_shape is not None:
        raise ValueError("Flow proposal must be unconditional.")
    data_log_density = flow.log_prob(samples)
    proposed, proposed_log_density = flow.sample_and_log_prob(
        key,
        sample_shape=(min(8, samples.shape[0]),),
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
