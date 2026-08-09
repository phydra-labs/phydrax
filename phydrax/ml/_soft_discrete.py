#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def _positive_temperature(temperature: ArrayLike, /) -> Array:
    value = jnp.asarray(temperature, dtype=float)
    return eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(value) | (value <= 0.0)),
        "temperature must be finite and positive.",
    )


def _real_values(values: ArrayLike, /, *, name: str) -> Array:
    array = jnp.asarray(values)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a real floating dtype.")
    return array


def masked_softmax(
    logits: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
    axis: int = -1,
) -> Array:
    """Stable softmax that returns zeros for an entirely masked slice."""
    values = _real_values(logits, name="logits")
    if mask is not None:
        values = jnp.where(jnp.asarray(mask, dtype=bool), values, -jnp.inf)
    probabilities = jax.nn.softmax(values, axis=axis)
    return jnp.where(jnp.isfinite(probabilities), probabilities, 0.0)


def temperature_softmax(
    logits: ArrayLike,
    /,
    *,
    temperature: ArrayLike,
    mask: ArrayLike | None = None,
    axis: int = -1,
) -> Array:
    """Masked softmax with an explicit positive continuous temperature."""
    return masked_softmax(
        _real_values(logits, name="logits") / _positive_temperature(temperature),
        mask=mask,
        axis=axis,
    )


def temperature_sigmoid(
    logits: ArrayLike,
    /,
    *,
    temperature: ArrayLike,
) -> Array:
    """Logistic gate with an explicit positive continuous temperature."""
    return jax.nn.sigmoid(
        _real_values(logits, name="logits") / _positive_temperature(temperature)
    )


def gumbel_softmax(
    key: Array,
    logits: ArrayLike,
    /,
    *,
    temperature: ArrayLike,
    axis: int = -1,
) -> Array:
    """Sample a relaxed categorical probability vector without hardening."""
    values = _real_values(logits, name="logits")
    uniform = jax.random.uniform(
        key,
        values.shape,
        dtype=values.dtype,
        minval=jnp.finfo(values.dtype).tiny,
        maxval=1.0,
    )
    noise = -jnp.log(-jnp.log(uniform))
    return temperature_softmax(
        values + noise,
        temperature=temperature,
        axis=axis,
    )


def soft_ranks(
    values: ArrayLike,
    /,
    *,
    temperature: ArrayLike,
    axis: int = -1,
    descending: bool = False,
) -> Array:
    """Return one-based pairwise-logistic ranks.

    Ranks are ascending by default and descending when requested. Equal values
    receive equal ranks; an all-equal axis receives its one-based midpoint rank.
    """
    array = _real_values(values, name="values")
    moved = jnp.moveaxis(array, axis, -1)
    difference = moved[..., :, None] - moved[..., None, :]
    if descending:
        difference = -difference
    comparisons = jax.nn.sigmoid(difference / _positive_temperature(temperature))
    ranks = 0.5 + jnp.sum(comparisons, axis=-1)
    return jnp.moveaxis(ranks, -1, axis)


def soft_topk_weights(
    scores: ArrayLike,
    /,
    *,
    k: int,
    rank_temperature: ArrayLike,
    gate_temperature: ArrayLike | None = None,
    axis: int = -1,
) -> Array:
    """Return logistic memberships derived from one-based descending soft ranks.

    Values lie in ``[0, 1]`` but do not generally sum to ``k``. This is a
    membership surrogate, not a cardinality-preserving transport mask.
    """
    values = _real_values(scores, name="scores")
    count = int(k)
    if count <= 0 or count > values.shape[axis]:
        raise ValueError("k must lie within the selected score axis.")
    ranks = soft_ranks(
        values,
        temperature=rank_temperature,
        axis=axis,
        descending=True,
    )
    gate = rank_temperature if gate_temperature is None else gate_temperature
    return temperature_sigmoid(float(count) + 0.5 - ranks, temperature=gate)


__all__ = [
    "gumbel_softmax",
    "masked_softmax",
    "soft_ranks",
    "soft_topk_weights",
    "temperature_sigmoid",
    "temperature_softmax",
]
