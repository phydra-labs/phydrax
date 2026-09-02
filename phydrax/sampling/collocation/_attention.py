#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from typing import TYPE_CHECKING

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ._adaptive import (
    _normalized_importance,
    _set_batch_rows,
    _single_axis_and_size,
    _validate_axis_field,
    AbstractCollocationPolicy,
    PointwiseSamplingTerm,
)


if TYPE_CHECKING:
    from phydrax.domain import DomainFunction, PointBatch


def _take_batch_rows(batch: PointBatch, indices: Array, /) -> PointBatch:
    from phydrax.domain import PointBatch

    axis, _ = _single_axis_and_size(batch)

    def take(value):
        if not isinstance(value, cx.Field) or axis not in value.named_dims:
            return value
        position = value.dims.index(axis)
        data = jnp.take(value.data, indices, axis=position)
        return cx.Field(data, dims=value.dims)

    points = jax.tree.map(
        take,
        batch.points,
        is_leaf=lambda value: isinstance(value, cx.Field),
    )
    metadata = jax.tree.map(
        take,
        batch.metadata,
        is_leaf=lambda value: isinstance(value, cx.Field),
    )
    return PointBatch(points, batch.structure, metadata=metadata)


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
    raw_score: Array
    point_id: Array
    age: Array
    anchor_mask: Array
    replacement_count: Array
    candidate_evaluations: Array

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
        raw_score: Array | None = None,
        point_id: Array | None = None,
        age: Array | None = None,
        anchor_mask: Array | None = None,
        replacement_count: int | Array = 0,
        candidate_evaluations: int | Array = 0,
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
        self.raw_score = (
            jnp.zeros((size,), dtype=float)
            if raw_score is None
            else jnp.asarray(raw_score, dtype=float)
        )
        self.point_id = (
            jnp.arange(size, dtype=jnp.int32)
            if point_id is None
            else jnp.asarray(point_id, dtype=jnp.int32)
        )
        self.age = (
            jnp.zeros((size,), dtype=jnp.int32)
            if age is None
            else jnp.asarray(age, dtype=jnp.int32)
        )
        self.anchor_mask = (
            jnp.zeros((size,), dtype=bool)
            if anchor_mask is None
            else jnp.asarray(anchor_mask, dtype=bool)
        )
        for name, value in (
            ("raw_score", self.raw_score),
            ("point_id", self.point_id),
            ("age", self.age),
            ("anchor_mask", self.anchor_mask),
        ):
            if value.shape != (size,):
                raise ValueError(f"{name} must have the population point shape.")
        self.replacement_count = jnp.asarray(replacement_count, dtype=jnp.int32)
        self.candidate_evaluations = jnp.asarray(
            candidate_evaluations,
            dtype=jnp.int32,
        )


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

    candidate_count: int = eqx.field(static=True)
    replacement_count: int = eqx.field(static=True)
    candidate_sampler: Callable[..., PointBatch] | None = eqx.field(static=True)
    anchor_fraction: float = eqx.field(static=True)
    anchor_probability_floor: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        refresh_every: int = 1,
        decay: float = 0.999,
        score_exponent: float = 0.5,
        uniform_fraction: float = 0.0,
        minimum_ess_fraction: float = 0.25,
        epsilon: float = 1e-12,
        candidate_count: int = 0,
        replacement_count: int = 0,
        candidate_sampler: Callable[..., PointBatch] | None = None,
        anchor_fraction: float = 0.0,
        anchor_probability_floor: float = 1.0e-6,
    ):
        refresh = int(refresh_every)
        decay_ = float(decay)
        exponent = float(score_exponent)
        uniform = float(uniform_fraction)
        minimum_ess = float(minimum_ess_fraction)
        epsilon_ = float(epsilon)
        candidate_count_ = int(candidate_count)
        replacement_count_ = int(replacement_count)
        anchor_fraction_ = float(anchor_fraction)
        anchor_floor = float(anchor_probability_floor)
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
        if candidate_count_ < 0 or replacement_count_ < 0:
            raise ValueError("Candidate and replacement counts must be nonnegative.")
        if replacement_count_ > candidate_count_:
            raise ValueError("replacement_count must not exceed candidate_count.")
        if replacement_count_ > 0 and candidate_sampler is None:
            raise ValueError("Support replacement requires a candidate_sampler.")
        if not isfinite(anchor_fraction_) or not 0.0 <= anchor_fraction_ < 1.0:
            raise ValueError("anchor_fraction must lie in [0, 1).")
        if not isfinite(anchor_floor) or not 0.0 < anchor_floor < 1.0:
            raise ValueError("anchor_probability_floor must lie in (0, 1).")
        self.refresh_every = refresh
        self.decay = jnp.asarray(decay_, dtype=float)
        self.score_exponent = jnp.asarray(exponent, dtype=float)
        self.uniform_fraction = jnp.asarray(uniform, dtype=float)
        self.minimum_ess_fraction = jnp.asarray(minimum_ess, dtype=float)
        self.epsilon = jnp.asarray(epsilon_, dtype=float)
        self.candidate_count = candidate_count_
        self.replacement_count = replacement_count_
        self.candidate_sampler = candidate_sampler
        self.anchor_fraction = anchor_fraction_
        self.anchor_probability_floor = anchor_floor

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
        anchor_count = (
            0
            if self.anchor_fraction == 0.0
            else max(1, int(round(size * self.anchor_fraction)))
        )
        if anchor_count + self.replacement_count > size:
            raise ValueError(
                "The declared anchors leave too few replaceable population points."
            )
        if anchor_count * self.anchor_probability_floor >= 1.0:
            raise ValueError(
                "Anchor probability floor exhausts the population probability mass."
            )
        anchor_mask = jnp.arange(size) < anchor_count
        probability = cx.Field(
            jnp.full((size,), 1.0 / size, dtype=float),
            dims=(axis,),
        )
        weight = cx.Field(jnp.ones((size,), dtype=float), dims=(axis,))
        return ResidualAttentionPopulation(
            batch,
            probability,
            weight,
            anchor_mask=anchor_mask,
        )

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
            "attention_score_mean": population.score_mean,
            "mean_age": jnp.mean(population.age.astype(float)),
            "anchor_count": jnp.sum(population.anchor_mask, dtype=float),
            "replacement_count": population.replacement_count.astype(float),
            "candidate_evaluations": population.candidate_evaluations.astype(float),
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
        return size + self.candidate_count

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
        score_key, candidate_key, candidate_score_key = jax.random.split(key, 3)
        score = term.pointwise_score(
            functions,
            population.batch,
            key=score_key,
        )
        axis, size = _single_axis_and_size(population.batch)
        _validate_axis_field(score, axis=axis, size=size, name="pointwise score")
        raw_current = jax.lax.stop_gradient(jnp.asarray(score.data, dtype=float))
        current_finite = jnp.isfinite(raw_current)
        safe_current = jnp.nan_to_num(
            jnp.maximum(raw_current, 0.0),
            nan=0.0,
            posinf=jnp.finfo(raw_current.dtype).max,
            neginf=0.0,
        )
        transferred_score = (
            self.decay * population.raw_score + (1.0 - self.decay) * safe_current
        )
        batch = population.batch
        point_id = population.point_id
        age = population.age + 1
        candidate_nonfinite = jnp.asarray(0, dtype=jnp.int32)
        if self.replacement_count:
            assert self.candidate_sampler is not None
            candidate_batch = self.candidate_sampler(
                key=candidate_key,
                count=self.candidate_count,
            )
            candidate_axis, candidate_size = _single_axis_and_size(candidate_batch)
            if candidate_axis != axis or candidate_size != self.candidate_count:
                raise ValueError(
                    "Candidate sampler must return the declared count on the "
                    "population point axis."
                )
            candidate_score = term.pointwise_score(
                functions,
                candidate_batch,
                key=candidate_score_key,
            )
            _validate_axis_field(
                candidate_score,
                axis=axis,
                size=self.candidate_count,
                name="candidate pointwise score",
            )
            raw_candidate = jax.lax.stop_gradient(
                jnp.asarray(candidate_score.data, dtype=float)
            )
            candidate_finite = jnp.isfinite(raw_candidate)
            safe_candidate = jnp.nan_to_num(
                jnp.maximum(raw_candidate, 0.0),
                nan=0.0,
                posinf=jnp.finfo(raw_candidate.dtype).max,
                neginf=0.0,
            )
            replace_priority = jnp.where(
                population.anchor_mask,
                jnp.inf,
                transferred_score,
            )
            replace_indices = jnp.argsort(
                replace_priority,
                stable=True,
            )[: self.replacement_count]
            candidate_indices = jnp.argsort(
                -safe_candidate,
                stable=True,
            )[: self.replacement_count]
            inserted = _take_batch_rows(candidate_batch, candidate_indices)
            batch = _set_batch_rows(batch, replace_indices, inserted)
            transferred_score = transferred_score.at[replace_indices].set(
                safe_candidate[candidate_indices]
            )
            next_id = (
                size + population.refresh_count * self.candidate_count + candidate_indices
            )
            point_id = point_id.at[replace_indices].set(next_id)
            age = age.at[replace_indices].set(0)
            candidate_nonfinite = jnp.sum(~candidate_finite, dtype=jnp.int32)
        proposal, _proposal_ess, effective_uniform, guard = _normalized_importance(
            transferred_score,
            exponent=self.score_exponent,
            uniform_fraction=self.uniform_fraction,
            minimum_ess_fraction=self.minimum_ess_fraction,
            epsilon=self.epsilon,
        )
        anchor_count = jnp.sum(population.anchor_mask, dtype=proposal.dtype)
        reserved = anchor_count * self.anchor_probability_floor
        probability = (
            self.anchor_probability_floor * population.anchor_mask.astype(proposal.dtype)
            + (1.0 - reserved) * proposal
        )
        probability = jax.lax.stop_gradient(probability / jnp.sum(probability))
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
            batch,
            cx.Field(probability, dims=(axis,)),
            cx.Field(weight, dims=(axis,)),
            refresh_count=population.refresh_count + 1,
            last_refresh=jnp.asarray(iter_, dtype=jnp.int32),
            score_mean=jnp.mean(transferred_score),
            score_max=jnp.max(transferred_score),
            score_nonfinite_count=(
                jnp.sum(~current_finite, dtype=jnp.int32) + candidate_nonfinite
            ),
            effective_sample_size=ess,
            entropy=entropy,
            effective_uniform_fraction=effective_uniform,
            ess_guard_triggered=guard,
            raw_score=jax.lax.stop_gradient(transferred_score),
            point_id=jax.lax.stop_gradient(point_id),
            age=jax.lax.stop_gradient(age),
            anchor_mask=population.anchor_mask,
            replacement_count=self.replacement_count,
            candidate_evaluations=(
                population.candidate_evaluations
                + (self.candidate_count if self.replacement_count else 0)
            ),
        )


__all__ = ["ResidualAttentionCollocation", "ResidualAttentionPopulation"]
