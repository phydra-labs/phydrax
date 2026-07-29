#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array

from ..models.core._keys import (
    EvalKey,
    fold_in_eval_key,
    split_eval_key,
)
from ..models.core._operator import OperatorBatch


SemigroupReduction = Literal["mean", "sum"]
SemigroupKeyMode = Literal["fold_in", "split"]


def _require_batch(
    value: Any,
    reference: OperatorBatch,
    /,
    *,
    owner: str,
) -> OperatorBatch:
    if not isinstance(value, OperatorBatch):
        raise TypeError(f"{owner} must return an OperatorBatch.")
    if value.case_axes != reference.case_axes or value.case_shape != reference.case_shape:
        raise ValueError(f"{owner} must preserve the OperatorBatch case layout.")
    if (
        value.require_single_query().sample_shape
        != reference.require_single_query().sample_shape
    ):
        raise ValueError(f"{owner} must preserve the query sample shape.")
    return value


def _duration(
    value: Any,
    batch: OperatorBatch,
    /,
    *,
    name: str,
) -> Array:
    duration = jnp.asarray(value)
    if duration.shape not in ((), batch.case_shape):
        raise ValueError(
            f"{name} must be scalar or have the OperatorBatch case shape "
            f"{batch.case_shape}; got {duration.shape}."
        )
    return duration


def _operator_output(model: Callable, batch: OperatorBatch, key: EvalKey, /) -> Array:
    values = jnp.asarray(model(batch, key=key))
    prefix = batch.case_shape + batch.require_single_query().sample_shape
    if (
        values.ndim not in (len(prefix), len(prefix) + 1)
        or values.shape[: len(prefix)] != prefix
    ):
        raise ValueError(
            "Conditioned transition output must have shape case_shape + "
            "query.sample_shape, optionally followed by one channel axis."
        )
    return values


def _evaluation_keys(key: EvalKey, mode: SemigroupKeyMode, /) -> tuple[EvalKey, ...]:
    if mode == "split":
        return tuple(split_eval_key(key, 3))
    return tuple(fold_in_eval_key(key, site) for site in range(3))


def _reduce_discrepancy(
    discrepancy: Array,
    batch: OperatorBatch,
    reduction: SemigroupReduction,
    /,
) -> Array:
    weights = batch.require_single_query().weights(case_shape=batch.case_shape)
    while weights.ndim < discrepancy.ndim:
        weights = weights[..., None]
    weighted = discrepancy * weights
    total = jnp.sum(weighted)
    if reduction == "sum":
        return total
    denominator = jnp.sum(jnp.broadcast_to(weights, discrepancy.shape))
    safe_denominator = jnp.where(denominator > 0, denominator, 1)
    return jnp.where(denominator > 0, total / safe_denominator, jnp.zeros_like(total))


@dataclass(frozen=True)
class ConditionedSemigroupObjective:
    """Weighted consistency between a direct and a composed conditioned transition.

    ``condition(batch, dt)`` must return a new ``OperatorBatch`` carrying ``dt``
    while preserving the case layout and query metadata. ``advance(batch, values)``
    must return a new batch whose evolving state is ``values``; the objective then
    applies the second condition to that batch. These callbacks keep the objective
    independent of input names and conditioning encodings.

    ``key_mode="fold_in"`` assigns stable sites 0, 1, and 2 to the direct, first,
    and second evaluations. ``key_mode="split"`` explicitly splits the root key
    into the same three evaluation keys. A ``None`` root key remains ``None`` at
    every site.
    """

    reduction: SemigroupReduction = "mean"
    weight: float = 1.0
    key_mode: SemigroupKeyMode = "fold_in"

    def __post_init__(self):
        if self.reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        if not isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError("weight must be finite and nonnegative.")
        if self.key_mode not in ("fold_in", "split"):
            raise ValueError("key_mode must be 'fold_in' or 'split'.")

    def __call__(
        self,
        model: Callable,
        batch: OperatorBatch,
        dt1: Any,
        dt2: Any,
        condition: Callable[[OperatorBatch, Array], OperatorBatch],
        advance: Callable[[OperatorBatch, Array], OperatorBatch],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Evaluate the scalar semigroup-consistency objective."""
        first_duration = _duration(dt1, batch, name="dt1")
        second_duration = _duration(dt2, batch, name="dt2")
        total_duration = first_duration + second_duration
        direct_key, first_key, second_key = _evaluation_keys(key, self.key_mode)

        direct_batch = _require_batch(
            condition(batch, total_duration),
            batch,
            owner="condition",
        )
        first_batch = _require_batch(
            condition(batch, first_duration),
            batch,
            owner="condition",
        )
        direct = _operator_output(model, direct_batch, direct_key)
        first = _operator_output(model, first_batch, first_key)

        advanced_batch = _require_batch(
            advance(first_batch, first),
            batch,
            owner="advance",
        )
        second_batch = _require_batch(
            condition(advanced_batch, second_duration),
            batch,
            owner="condition",
        )
        composed = _operator_output(model, second_batch, second_key)
        if direct.shape != composed.shape:
            raise ValueError(
                "Direct and composed transition outputs must have equal shape."
            )

        discrepancy = jnp.abs(direct - composed) ** 2
        reduced = _reduce_discrepancy(discrepancy, direct_batch, self.reduction)
        return jnp.asarray(self.weight, dtype=reduced.dtype) * reduced


def conditioned_semigroup_consistency_loss(
    model: Callable,
    batch: OperatorBatch,
    dt1: Any,
    dt2: Any,
    condition: Callable[[OperatorBatch, Array], OperatorBatch],
    advance: Callable[[OperatorBatch, Array], OperatorBatch],
    /,
    *,
    reduction: SemigroupReduction = "mean",
    weight: float = 1.0,
    key_mode: SemigroupKeyMode = "fold_in",
    key: EvalKey = None,
) -> Array:
    """Evaluate a configured :class:`ConditionedSemigroupObjective`."""
    return ConditionedSemigroupObjective(
        reduction=reduction,
        weight=weight,
        key_mode=key_mode,
    )(model, batch, dt1, dt2, condition, advance, key=key)


__all__ = [
    "ConditionedSemigroupObjective",
    "SemigroupKeyMode",
    "SemigroupReduction",
    "conditioned_semigroup_consistency_loss",
]
