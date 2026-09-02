#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import isfinite
from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._strict import StrictModule
from ._api import _requires_random_key, IntegrationRealization, materialize
from ._batches import PointIntegrationBatch, WeightedSampleBatch
from ._status import IntegrationStatus


class AdaptiveSignedDiagnostics(StrictModule):
    allocation: Array | None
    effective_sample_size: Array | None
    minimum_weight: Array | None
    maximum_weight: Array | None
    stratum_variance: Array | None
    evaluation_count: Array
    status: Array


class AdaptiveSignedPopulation(StrictModule):
    realization: IntegrationRealization
    active: Array
    age: Array
    epoch: Array
    diagnostics: AdaptiveSignedDiagnostics


def _population_active(realization: IntegrationRealization, /) -> Array:
    batch = realization.batch
    if isinstance(batch, PointIntegrationBatch):
        active = jnp.ones(batch.weights.data.shape, dtype=bool)
        if batch.mask is not None:
            active &= jnp.asarray(batch.mask.data, dtype=bool)
        return active.reshape((-1,))
    if isinstance(batch, WeightedSampleBatch):
        if batch.mask is None:
            return jnp.ones((batch.num_samples,), dtype=bool)
        mask = batch.mask.data if isinstance(batch.mask, cx.Field) else batch.mask
        return jnp.asarray(mask, dtype=bool).reshape((-1,))
    raise TypeError(
        "Adaptive signed estimators require point or weighted-sample batches."
    )


def _materialize_source(source, key):
    if _requires_random_key(source.initial_plan):
        return materialize(source.target, source.initial_plan, key=key)
    return materialize(source.target, source.initial_plan)


class AdaptiveSignedEstimator(StrictModule):
    """Typed estimator policy preserving a signed or complex estimand."""

    __strict_abstract__ = True

    refresh_interval: int = eqx.field(static=True)

    def __init__(self, *, refresh_interval: int = 1):
        interval = int(refresh_interval)
        if interval < 1:
            raise ValueError("refresh_interval must be positive.")
        self.refresh_interval = interval

    @abstractmethod
    def validate_source(self, source) -> None:
        raise NotImplementedError

    def initialize(self, term, /, *, key: Key[Array, ""]) -> AdaptiveSignedPopulation:
        source = term.source
        self.validate_source(source)
        realization = _materialize_source(source, key)
        active = _population_active(realization)
        if active.size == 0 or not bool(jnp.any(active)):
            raise ValueError("Adaptive signed populations cannot be empty.")
        diagnostics = AdaptiveSignedDiagnostics(
            allocation=None,
            effective_sample_size=None,
            minimum_weight=None,
            maximum_weight=None,
            stratum_variance=None,
            evaluation_count=jnp.asarray(0, dtype=jnp.int32),
            status=jnp.asarray(int(IntegrationStatus.CONVERGED), dtype=jnp.int32),
        )
        return AdaptiveSignedPopulation(
            realization=realization,
            active=active,
            age=jnp.asarray(0, dtype=jnp.int32),
            epoch=jnp.asarray(0, dtype=jnp.int32),
            diagnostics=diagnostics,
        )

    def should_refresh(
        self, population: AdaptiveSignedPopulation, iteration: Any
    ) -> bool:
        return bool(jnp.asarray(iteration) % self.refresh_interval == 0)

    def refresh(
        self,
        term,
        functions,
        population: AdaptiveSignedPopulation,
        /,
        *,
        key: Key[Array, ""],
        iter_: Any,
    ) -> AdaptiveSignedPopulation:
        del functions, iter_
        epoch = population.epoch + 1
        realization = _materialize_source(term.source, jr.fold_in(key, epoch))
        active = _population_active(realization)
        diagnostics = eqx.tree_at(
            lambda value: value.evaluation_count,
            population.diagnostics,
            jnp.asarray(0, dtype=jnp.int32),
        )
        return AdaptiveSignedPopulation(
            realization=realization,
            active=active,
            age=population.age + 1,
            epoch=epoch,
            diagnostics=diagnostics,
        )

    def loss_realization(
        self, population: AdaptiveSignedPopulation
    ) -> IntegrationRealization:
        return population.realization

    def record_training_evaluation(
        self,
        population: AdaptiveSignedPopulation,
        /,
        *,
        multiplier: int = 1,
    ) -> AdaptiveSignedPopulation:
        diagnostics = eqx.tree_at(
            lambda value: value.evaluation_count,
            population.diagnostics,
            population.diagnostics.evaluation_count + int(multiplier),
        )
        return eqx.tree_at(lambda value: value.diagnostics, population, diagnostics)

    def data_metrics(self, population: AdaptiveSignedPopulation) -> dict[str, Array]:
        return {
            "epoch": population.epoch,
            "active_samples": jnp.sum(population.active, dtype=jnp.int32),
            "evaluation_count": population.diagnostics.evaluation_count,
            "status": population.diagnostics.status,
        }

    @staticmethod
    def signed_reduce(
        values: Array, weights: Array, active: Array | None = None
    ) -> Array:
        values_ = jnp.asarray(values)
        weights_ = jnp.asarray(weights, dtype=jnp.real(values_).dtype)
        if values_.shape[0] != weights_.shape[0]:
            raise ValueError(
                "Signed estimator values and weights must share sample axis."
            )
        mask = jnp.ones(weights_.shape, dtype=bool) if active is None else active
        safe = jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (values_.ndim - 1)),
            values_,
            jnp.zeros((), dtype=values_.dtype),
        )
        return jnp.tensordot(jnp.where(mask, weights_, 0.0), safe, axes=(0, 0))


class AdaptiveStratifiedEstimator(AdaptiveSignedEstimator):
    """Independent next-epoch bounded Neyman allocation for exact strata."""

    stratum_masses: Array
    variance_floor: float = eqx.field(static=True)
    minimum_per_stratum: int = eqx.field(static=True)

    def __init__(
        self,
        stratum_masses: Array,
        /,
        *,
        variance_floor: float = 1.0e-12,
        minimum_per_stratum: int = 1,
        refresh_interval: int = 1,
    ):
        super().__init__(refresh_interval=refresh_interval)
        masses = jnp.asarray(stratum_masses, dtype=float).reshape((-1,))
        floor = float(variance_floor)
        minimum = int(minimum_per_stratum)
        if (
            masses.size == 0
            or bool(jnp.any(~jnp.isfinite(masses)))
            or bool(jnp.any(masses <= 0))
        ):
            raise ValueError("Stratum masses must be finite and positive.")
        if not bool(jnp.isclose(jnp.sum(masses), 1.0)):
            raise ValueError("Stratum masses must sum to one.")
        if not isfinite(floor) or floor <= 0.0 or minimum < 1:
            raise ValueError("Variance floor and minimum allocation must be positive.")
        self.stratum_masses = masses
        self.variance_floor = floor
        self.minimum_per_stratum = minimum

    def validate_source(self, source) -> None:
        from ._plans import StratifiedMonteCarloPlan

        if not isinstance(source.initial_plan, StratifiedMonteCarloPlan):
            raise TypeError(
                "AdaptiveStratifiedEstimator requires StratifiedMonteCarloPlan."
            )

    def next_allocation(self, variances: Array, budget: int, /) -> Array:
        variance = jnp.asarray(variances, dtype=self.stratum_masses.dtype).reshape((-1,))
        if variance.shape != self.stratum_masses.shape:
            raise ValueError("variances must contain one value per stratum.")
        if bool(jnp.any(~jnp.isfinite(variance))) or bool(jnp.any(variance < 0.0)):
            raise ValueError("Stratum variances must be finite and nonnegative.")
        budget_ = int(budget)
        base = self.minimum_per_stratum * int(variance.size)
        if budget_ < base:
            raise ValueError("Budget cannot leave a positive-mass stratum unsampled.")
        score = self.stratum_masses * jnp.sqrt(jnp.maximum(variance, self.variance_floor))
        remaining = budget_ - base
        ideal = remaining * score / jnp.sum(score)
        extra = jnp.floor(ideal).astype(jnp.int32)
        remainder = remaining - jnp.sum(extra)
        order = jnp.argsort(-(ideal - extra), stable=True)
        extra = extra.at[order].add(jnp.arange(extra.size) < remainder)
        return extra + self.minimum_per_stratum


class AdaptiveImportanceEstimator(AdaptiveSignedEstimator):
    """Ordinary signed importance estimator with a declared defensive floor."""

    defensive_mixture_floor: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        defensive_mixture_floor: float = 1.0e-3,
        refresh_interval: int = 1,
    ):
        super().__init__(refresh_interval=refresh_interval)
        floor = float(defensive_mixture_floor)
        if not isfinite(floor) or not 0.0 < floor <= 1.0:
            raise ValueError("defensive_mixture_floor must lie in (0, 1].")
        self.defensive_mixture_floor = floor

    def validate_source(self, source) -> None:
        from ._plans import ImportanceSamplingPlan

        if not isinstance(source.initial_plan, ImportanceSamplingPlan):
            raise TypeError(
                "AdaptiveImportanceEstimator requires ImportanceSamplingPlan."
            )

    def validate_log_ratios(self, log_ratios: Array, active: Array | None = None) -> None:
        ratios = jnp.asarray(log_ratios)
        mask = jnp.ones(ratios.shape, dtype=bool) if active is None else active
        if not bool(jnp.all(jnp.isfinite(jnp.where(mask, ratios, 0.0)))):
            raise ValueError("Active target/proposal log ratios must be finite.")


__all__ = [
    "AdaptiveImportanceEstimator",
    "AdaptiveSignedDiagnostics",
    "AdaptiveSignedEstimator",
    "AdaptiveSignedPopulation",
    "AdaptiveStratifiedEstimator",
]
