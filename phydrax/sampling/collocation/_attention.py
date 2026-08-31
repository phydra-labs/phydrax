#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ._adaptive import (
    _normalized_importance,
    _single_axis_and_size,
    _validate_axis_field,
    AbstractCollocationPolicy,
    PointwiseSamplingTerm,
)


if TYPE_CHECKING:
    from phydrax.domain import DomainFunction, PointBatch


class ResidualAttentionPopulation(StrictModule):
    """Fixed collocation support with mass-preserving residual attention."""

    batch: PointBatch
    probability: cx.Field
    weight: cx.Field
    refresh_count: Array
    last_refresh: Array
    score_mean: Array
    score_max: Array
    score_nonfinite_count: Array
    effective_sample_size: Array
    entropy: Array
    effective_uniform_fraction: Array
    ess_guard_triggered: Array

    def __init__(
        self,
        batch: PointBatch,
        probability: cx.Field,
        weight: cx.Field,
        /,
        *,
        refresh_count: int | Array = 0,
        last_refresh: int | Array = 0,
        score_mean: float | Array = 0.0,
        score_max: float | Array = 0.0,
        score_nonfinite_count: int | Array = 0,
        effective_sample_size: float | Array | None = None,
        entropy: float | Array = 1.0,
        effective_uniform_fraction: float | Array = 1.0,
        ess_guard_triggered: bool | Array = False,
    ):
        axis, size = _single_axis_and_size(batch)
        _validate_axis_field(probability, axis=axis, size=size, name="probability")
        _validate_axis_field(weight, axis=axis, size=size, name="weight")
        self.batch = batch
        self.probability = probability
        self.weight = weight
        self.refresh_count = jnp.asarray(refresh_count, dtype=jnp.int32)
        self.last_refresh = jnp.asarray(last_refresh, dtype=jnp.int32)
        self.score_mean = jnp.asarray(score_mean, dtype=float)
        self.score_max = jnp.asarray(score_max, dtype=float)
        self.score_nonfinite_count = jnp.asarray(score_nonfinite_count, dtype=jnp.int32)
        self.effective_sample_size = jnp.asarray(
            size if effective_sample_size is None else effective_sample_size,
            dtype=float,
        )
        self.entropy = jnp.asarray(entropy, dtype=float)
        self.effective_uniform_fraction = jnp.asarray(
            effective_uniform_fraction, dtype=float
        )
        self.ess_guard_triggered = jnp.asarray(ess_guard_triggered, dtype=bool)


class ResidualAttentionCollocation(AbstractCollocationPolicy):
    """EMA residual attention over one immutable paired-point population.

    Pointwise scores define a probability distribution mixed with a uniform floor and
    guarded by a minimum effective sample size. The returned local multipliers have
    unit arithmetic mean, so attention does not silently change the global term scale.
    """

    refresh_every: int
    decay: Array
    score_exponent: Array
    uniform_fraction: Array
    minimum_ess_fraction: Array
    epsilon: Array

    def __init__(
        self,
        *,
        refresh_every: int = 1,
        decay: float = 0.999,
        score_exponent: float = 0.5,
        uniform_fraction: float = 0.0,
        minimum_ess_fraction: float = 0.25,
        epsilon: float = 1e-12,
    ):
        refresh = int(refresh_every)
        decay_ = float(decay)
        exponent = float(score_exponent)
        uniform = float(uniform_fraction)
        minimum_ess = float(minimum_ess_fraction)
        epsilon_ = float(epsilon)
        if refresh <= 0:
            raise ValueError("refresh_every must be positive.")
        if not isfinite(decay_) or not 0.0 <= decay_ < 1.0:
            raise ValueError("decay must lie in [0, 1).")
        if not isfinite(exponent) or exponent < 0.0:
            raise ValueError("score_exponent must be finite and non-negative.")
        if not isfinite(uniform) or not 0.0 <= uniform <= 1.0:
            raise ValueError("uniform_fraction must lie in [0, 1].")
        if not isfinite(minimum_ess) or not 0.0 < minimum_ess <= 1.0:
            raise ValueError("minimum_ess_fraction must lie in (0, 1].")
        if not isfinite(epsilon_) or epsilon_ <= 0.0:
            raise ValueError("epsilon must be finite and strictly positive.")
        self.refresh_every = refresh
        self.decay = jnp.asarray(decay_, dtype=float)
        self.score_exponent = jnp.asarray(exponent, dtype=float)
        self.uniform_fraction = jnp.asarray(uniform, dtype=float)
        self.minimum_ess_fraction = jnp.asarray(minimum_ess, dtype=float)
        self.epsilon = jnp.asarray(epsilon_, dtype=float)

    def initialize(
        self,
        term: PointwiseSamplingTerm,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> ResidualAttentionPopulation:
        batch = term.sample(key=key)
        from phydrax.domain import PointBatch

        if not isinstance(batch, PointBatch):
            raise TypeError("Residual attention requires a PointBatch.")
        axis, size = _single_axis_and_size(batch)
        probability = cx.Field(
            jnp.full((size,), 1.0 / size, dtype=float),
            dims=(axis,),
        )
        weight = cx.Field(jnp.ones((size,), dtype=float), dims=(axis,))
        return ResidualAttentionPopulation(batch, probability, weight)

    def should_refresh(
        self,
        population: ResidualAttentionPopulation,
        iter_: int | Array,
    ) -> Array:
        step = jnp.asarray(iter_, dtype=jnp.int32)
        return (step - population.last_refresh) >= self.refresh_every

    def loss_batch_and_weight(
        self,
        population: ResidualAttentionPopulation,
        /,
    ) -> tuple[PointBatch, cx.Field]:
        return population.batch, population.weight

    def data_metrics(
        self,
        population: ResidualAttentionPopulation,
        /,
    ) -> dict[str, Array]:
        _, size = _single_axis_and_size(population.batch)
        values = jnp.asarray(population.weight.data, dtype=float)
        return {
            "refresh_count": jnp.asarray(population.refresh_count, dtype=float),
            "last_refresh": jnp.asarray(population.last_refresh, dtype=float),
            "point_count": jnp.asarray(size, dtype=float),
            "active_point_count": jnp.asarray(size, dtype=float),
            "effective_sample_size": population.effective_sample_size,
            "mean_age": jnp.asarray(0.0, dtype=float),
            "attention_score_mean": population.score_mean,
            "attention_score_max": population.score_max,
            "attention_score_nonfinite_count": jnp.asarray(
                population.score_nonfinite_count, dtype=float
            ),
            "attention_effective_sample_size": population.effective_sample_size,
            "attention_effective_sample_fraction": (
                population.effective_sample_size / float(size)
            ),
            "attention_entropy": population.entropy,
            "attention_weight_min": jnp.min(values),
            "attention_weight_max": jnp.max(values),
            "attention_weight_mean": jnp.mean(values),
            "attention_effective_uniform_fraction": (
                population.effective_uniform_fraction
            ),
            "attention_ess_guard_triggered": jnp.asarray(
                population.ess_guard_triggered, dtype=float
            ),
        }

    def refresh_residual_evaluations(
        self,
        population: ResidualAttentionPopulation,
        /,
    ) -> int:
        _, size = _single_axis_and_size(population.batch)
        return size

    def refresh(
        self,
        term: PointwiseSamplingTerm,
        functions: Mapping[str, DomainFunction],
        population: ResidualAttentionPopulation,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array,
    ) -> ResidualAttentionPopulation:
        score = term.pointwise_score(functions, population.batch, key=key)
        axis, size = _single_axis_and_size(population.batch)
        _validate_axis_field(score, axis=axis, size=size, name="pointwise score")
        raw_score = jax.lax.stop_gradient(jnp.asarray(score.data, dtype=float))
        finite = jnp.isfinite(raw_score)
        safe_score = jnp.nan_to_num(
            jnp.maximum(raw_score, 0.0),
            nan=0.0,
            posinf=jnp.finfo(raw_score.dtype).max,
            neginf=0.0,
        )
        proposal, _proposal_ess, effective_uniform, guard = _normalized_importance(
            safe_score,
            exponent=self.score_exponent,
            uniform_fraction=self.uniform_fraction,
            minimum_ess_fraction=self.minimum_ess_fraction,
            epsilon=self.epsilon,
        )
        previous = jnp.asarray(population.probability.data, dtype=float)
        probability = self.decay * previous + (1.0 - self.decay) * proposal
        probability = probability / jnp.sum(probability)
        probability = jax.lax.stop_gradient(probability)
        weight = jax.lax.stop_gradient(float(size) * probability)
        ess = jnp.reciprocal(jnp.sum(probability * probability))
        entropy = -jnp.sum(
            jnp.where(
                probability > 0.0,
                probability * jnp.log(probability),
                0.0,
            )
        ) / jnp.maximum(jnp.log(jnp.asarray(float(size))), 1.0)
        return ResidualAttentionPopulation(
            population.batch,
            cx.Field(probability, dims=(axis,)),
            cx.Field(weight, dims=(axis,)),
            refresh_count=population.refresh_count + 1,
            last_refresh=jnp.asarray(iter_, dtype=jnp.int32),
            score_mean=jnp.mean(safe_score),
            score_max=jnp.max(safe_score),
            score_nonfinite_count=jnp.sum(~finite, dtype=jnp.int32),
            effective_sample_size=ess,
            entropy=entropy,
            effective_uniform_fraction=effective_uniform,
            ess_guard_triggered=guard,
        )


__all__ = ["ResidualAttentionCollocation", "ResidualAttentionPopulation"]
