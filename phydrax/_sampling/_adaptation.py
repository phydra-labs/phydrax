#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Between-chunk proposal adaptation with frozen production epochs."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class RobbinsMonroScalePolicy(StrictModule):
    target_acceptance: float = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    decay_power: float = eqx.field(static=True)
    minimum_scale: float = eqx.field(static=True)
    maximum_scale: float = eqx.field(static=True)
    warmup_chunks: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        target_acceptance: float = 0.5,
        learning_rate: float = 0.1,
        decay_power: float = 0.5,
        minimum_scale: float = 1e-5,
        maximum_scale: float = 10.0,
        warmup_chunks: int,
    ):
        target, rate, power = map(float, (target_acceptance, learning_rate, decay_power))
        lower, upper = map(float, (minimum_scale, maximum_scale))
        warmup = int(warmup_chunks)
        if not 0.0 < target < 1.0 or rate <= 0.0 or not 0.0 <= power <= 1.0:
            raise ValueError("Invalid Robbins-Monro target/rate/decay_power.")
        if (
            not 0.0 < lower < upper
            or warmup < 0
            or not all(np.isfinite(v) for v in (rate, lower, upper))
        ):
            raise ValueError("Invalid scale bounds or warmup_chunks.")
        self.target_acceptance = target
        self.learning_rate = rate
        self.decay_power = power
        self.minimum_scale = lower
        self.maximum_scale = upper
        self.warmup_chunks = warmup

    def scale_from_raw(self, raw_scale: ArrayLike, /) -> Array:
        raw = jnp.asarray(raw_scale)
        fraction = jax_sigmoid(raw)
        return self.minimum_scale + (self.maximum_scale - self.minimum_scale) * fraction

    def raw_from_scale(self, scale: ArrayLike, /) -> Array:
        value = jnp.asarray(scale)
        fraction = (value - self.minimum_scale) / (
            self.maximum_scale - self.minimum_scale
        )
        if value.shape != ():
            raise ValueError("scale must be scalar.")
        return jnp.log(fraction) - jnp.log1p(-fraction)


class AdaptiveProposalState(StrictModule):
    raw_scale: Array
    scale: Array
    chunk_index: Array
    last_acceptance: Array
    frozen: Array
    valid: Array


def jax_sigmoid(value: Array, /) -> Array:
    return jnp.where(
        value >= 0.0,
        1.0 / (1.0 + jnp.exp(-value)),
        jnp.exp(value) / (1.0 + jnp.exp(value)),
    )


def initialize_proposal_adaptation(
    policy: RobbinsMonroScalePolicy, initial_scale: ArrayLike, /
) -> AdaptiveProposalState:
    if not isinstance(policy, RobbinsMonroScalePolicy):
        raise TypeError("policy must be RobbinsMonroScalePolicy.")
    scale = jnp.asarray(initial_scale)
    if scale.shape != ():
        raise ValueError("initial_scale must be scalar.")
    raw = policy.raw_from_scale(scale)
    valid = (
        jnp.isfinite(raw)
        & (scale > policy.minimum_scale)
        & (scale < policy.maximum_scale)
    )
    return AdaptiveProposalState(
        raw_scale=raw,
        scale=policy.scale_from_raw(raw),
        chunk_index=jnp.asarray(0, dtype=jnp.int32),
        last_acceptance=jnp.asarray(jnp.nan),
        frozen=jnp.asarray(policy.warmup_chunks == 0),
        valid=valid,
    )


def adapt_proposal_scale(
    policy: RobbinsMonroScalePolicy,
    state: AdaptiveProposalState,
    acceptance: ArrayLike,
    /,
) -> AdaptiveProposalState:
    """Commit one adaptation only at a completed chunk boundary."""
    rate = jnp.asarray(acceptance)
    if rate.shape != ():
        raise ValueError("acceptance must be scalar.")
    adapting = state.chunk_index < policy.warmup_chunks
    eta = (
        policy.learning_rate
        / (state.chunk_index.astype(state.raw_scale.dtype) + 1.0) ** policy.decay_power
    )
    proposed_raw = state.raw_scale + eta * (rate - policy.target_acceptance)
    raw = jnp.where(adapting, proposed_raw, state.raw_scale)
    next_index = state.chunk_index + 1
    valid = (
        state.valid
        & jnp.isfinite(rate)
        & (rate >= 0.0)
        & (rate <= 1.0)
        & jnp.isfinite(raw)
    )
    return AdaptiveProposalState(
        raw_scale=raw,
        scale=policy.scale_from_raw(raw),
        chunk_index=next_index,
        last_acceptance=rate,
        frozen=next_index >= policy.warmup_chunks,
        valid=valid,
    )


__all__ = [
    "AdaptiveProposalState",
    "RobbinsMonroScalePolicy",
    "adapt_proposal_scale",
    "initialize_proposal_adaptation",
]
