#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import coordax as cx
import jax.numpy as jnp

from .._sampling import MarkovSampleResult
from ._targets import WeightedSampleTarget


def _dimension(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def markov_chain_measure(
    result: MarkovSampleResult,
    /,
    *,
    chain_dim: str = "__phydrax_markov_chain",
    draw_dim: str = "__phydrax_markov_draw",
) -> WeightedSampleTarget:
    """Lower correlated chain-by-draw samples to an equal-weight empirical measure."""
    if not isinstance(result, MarkovSampleResult):
        raise TypeError("result must be a MarkovSampleResult.")
    chain = _dimension(chain_dim, "chain_dim")
    draw = _dimension(draw_dim, "draw_dim")
    if chain == draw:
        raise ValueError("chain_dim and draw_dim must be distinct.")
    log_weights = cx.Field(
        jnp.zeros((result.num_chains, result.num_draws), dtype=float),
        dims=(chain, draw),
    )
    replicate_ids = jnp.repeat(
        jnp.arange(result.num_chains, dtype=jnp.int32),
        result.num_draws,
    )
    return WeightedSampleTarget(
        result.samples,
        log_weights,
        normalized=True,
        replicate_ids=replicate_ids,
        sample_axes=(chain, draw),
        provenance=f"markov:{result.kernel_id}:{result.proposal_id}",
        independent=False,
    )


__all__ = ["markov_chain_measure"]
