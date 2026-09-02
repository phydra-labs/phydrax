#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..transport._ordering import (
    ordered_ranks,
    OrderingSurrogate,
    PAVOrdering,
    SinkhornOrdering,
    WeightedPAVOrdering,
)


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


class RelaxedDiscreteSample(StrictModule):
    """Relaxed sample plus optional hard forward estimator evidence."""

    value: Array
    relaxed: Array
    hard: Array | None
    estimator: str = eqx.field(static=True)


def relaxed_bernoulli(
    logits: ArrayLike,
    /,
    *,
    key,
    temperature: ArrayLike = 1.0,
    hard: bool = False,
) -> RelaxedDiscreteSample:
    """Stateless Binary Concrete sample with optional straight-through hardening."""
    if not isinstance(hard, bool):
        raise TypeError("hard must be a bool.")
    logits_ = _real_values(logits, name="logits")
    temperature_ = _positive_temperature(temperature)
    uniform = jax.random.uniform(
        key,
        logits_.shape,
        dtype=logits_.dtype,
        minval=jnp.finfo(logits_.dtype).tiny,
        maxval=1.0 - jnp.finfo(logits_.dtype).eps,
    )
    logistic = jnp.log(uniform) - jnp.log1p(-uniform)
    relaxed = jax.nn.sigmoid((logits_ + logistic) / temperature_)
    if not hard:
        return RelaxedDiscreteSample(relaxed, relaxed, None, "binary-concrete")
    hard_sample = (logits_ + logistic >= 0.0).astype(relaxed.dtype)
    value = relaxed + jax.lax.stop_gradient(hard_sample - relaxed)
    return RelaxedDiscreteSample(
        value,
        relaxed,
        hard_sample,
        "binary-concrete-straight-through",
    )


def relaxed_top_k(
    logits: ArrayLike,
    k: int,
    /,
    *,
    key,
    ordering: OrderingSurrogate | None = None,
    temperature: ArrayLike = 1.0,
    axis: int = -1,
    hard: bool = False,
) -> RelaxedDiscreteSample:
    """Gumbel top-k relaxation under one declared ordering estimator."""
    if isinstance(k, bool) or int(k) <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(axis, int) or not isinstance(hard, bool):
        raise TypeError("axis must be an int and hard must be a bool.")
    logits_ = _real_values(logits, name="logits")
    position = axis if axis >= 0 else logits_.ndim + axis
    if position < 0 or position >= logits_.ndim:
        raise ValueError("axis is out of range.")
    count = int(logits_.shape[position])
    cardinality = int(k)
    if cardinality > count:
        raise ValueError("k cannot exceed the selected axis size.")
    temperature_ = _positive_temperature(temperature)
    method = PAVOrdering(float(temperature_)) if ordering is None else ordering
    if not isinstance(method, (PAVOrdering, SinkhornOrdering, WeightedPAVOrdering)):
        raise TypeError("ordering must be a declared soft ordering surrogate.")
    if isinstance(method, WeightedPAVOrdering):
        raise ValueError(
            "relaxed_top_k does not define atom masses; use PAV or Sinkhorn ordering."
        )
    uniform = jax.random.uniform(
        key,
        logits_.shape,
        dtype=logits_.dtype,
        minval=jnp.finfo(logits_.dtype).tiny,
        maxval=1.0 - jnp.finfo(logits_.dtype).eps,
    )
    perturbed = logits_ - jnp.log(-jnp.log(uniform))
    ranks = ordered_ranks(
        perturbed,
        method,
        axis=position,
        descending=True,
    )
    relaxed = jax.nn.sigmoid(
        (jnp.asarray(cardinality, dtype=logits_.dtype) - 0.5 - ranks) / temperature_
    )
    if not hard:
        return RelaxedDiscreteSample(relaxed, relaxed, None, "gumbel-soft-top-k")
    indices = jax.lax.top_k(jnp.moveaxis(perturbed, position, -1), cardinality)[1]
    hard_moved = jnp.sum(
        jax.nn.one_hot(indices, count, dtype=logits_.dtype),
        axis=-2,
    )
    hard_sample = jnp.moveaxis(hard_moved, -1, position)
    value = relaxed + jax.lax.stop_gradient(hard_sample - relaxed)
    return RelaxedDiscreteSample(
        value,
        relaxed,
        hard_sample,
        "gumbel-top-k-straight-through",
    )


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
    "RelaxedDiscreteSample",
    "gumbel_softmax",
    "masked_softmax",
    "relaxed_bernoulli",
    "relaxed_top_k",
    "soft_ranks",
    "soft_topk_weights",
    "temperature_sigmoid",
    "temperature_softmax",
]
