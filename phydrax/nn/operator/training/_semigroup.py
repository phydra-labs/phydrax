#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ....transport import (
    AbstractBalancedTransportSolver,
    AbstractGroundCost,
)
from ..._keys import (
    EvalKey,
    fold_in_eval_key,
    split_eval_key,
)
from ..data import OperatorBatch
from ..distribution import AbstractProbabilisticOperatorModel


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


@dataclass(frozen=True)
class DistributionalSemigroupObjective:
    """Energy distance between direct and independently composed transition laws."""

    num_samples: int = 16
    measure: Literal["quadrature", "uniform"] = "quadrature"
    beta: float = 1.0
    chunk_size: int | None = None
    reduction: SemigroupReduction = "mean"
    weight: float = 1.0
    key_mode: SemigroupKeyMode = "fold_in"

    def __post_init__(self):
        if int(self.num_samples) < 2:
            raise ValueError("num_samples must be at least two.")
        if self.measure not in ("quadrature", "uniform"):
            raise ValueError("measure must be 'quadrature' or 'uniform'.")
        if not 0.0 < float(self.beta) <= 2.0:
            raise ValueError("beta must satisfy 0 < beta <= 2.")
        if self.chunk_size is not None and int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive when provided.")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        if not isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError("weight must be finite and nonnegative.")
        if self.key_mode not in ("fold_in", "split"):
            raise ValueError("key_mode must be 'fold_in' or 'split'.")

    def __call__(
        self,
        model: AbstractProbabilisticOperatorModel,
        batch: OperatorBatch,
        dt1: Any,
        dt2: Any,
        condition: Callable[[OperatorBatch, Array], OperatorBatch],
        advance: Callable[[OperatorBatch, Array], OperatorBatch],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        direct_predictive, composed_predictive = _distributional_predictives(
            model,
            batch,
            dt1,
            dt2,
            condition,
            advance,
            num_samples=self.num_samples,
            key_mode=self.key_mode,
            key=key,
            owner="DistributionalSemigroupObjective",
        )
        from ....uq._operator_metrics import operator_ensemble_energy_distance

        distance = operator_ensemble_energy_distance(
            direct_predictive,
            composed_predictive,
            measure=self.measure,
            beta=self.beta,
            chunk_size=self.chunk_size,
            reduction=self.reduction,
        )
        return jnp.asarray(self.weight, dtype=distance.dtype) * distance


def _distributional_predictives(
    model: AbstractProbabilisticOperatorModel,
    batch: OperatorBatch,
    dt1: Any,
    dt2: Any,
    condition: Callable[[OperatorBatch, Array], OperatorBatch],
    advance: Callable[[OperatorBatch, Array], OperatorBatch],
    /,
    *,
    num_samples: int,
    key_mode: SemigroupKeyMode,
    key: EvalKey,
    owner: str,
):
    if not isinstance(model, AbstractProbabilisticOperatorModel):
        raise TypeError(f"{owner} requires a probabilistic operator.")
    if key is None:
        raise ValueError(f"{owner} requires a PRNG key.")
    first_duration = _duration(dt1, batch, name="dt1")
    second_duration = _duration(dt2, batch, name="dt2")
    direct_batch = _require_batch(
        condition(batch, first_duration + second_duration),
        batch,
        owner="condition",
    )
    first_batch = _require_batch(
        condition(batch, first_duration),
        batch,
        owner="condition",
    )
    key_count = 4 + 2 * int(num_samples)
    if key_mode == "split":
        keys = tuple(split_eval_key(key, key_count))
    else:
        keys = tuple(fold_in_eval_key(key, site) for site in range(key_count))
    if any(value is None for value in keys):
        raise AssertionError("Distributional semigroup keys unexpectedly resolved to None.")
    resolved_keys = tuple(cast(Array, value) for value in keys)

    direct_distribution = model.distribution(direct_batch, key=resolved_keys[0])
    first_distribution = model.distribution(first_batch, key=resolved_keys[2])
    if (
        direct_distribution.uncertainty_source != "process"
        or first_distribution.uncertainty_source != "process"
    ):
        raise ValueError(
            "Distributional semigroup objectives require process distributions."
        )
    direct_samples = direct_distribution.sample(
        resolved_keys[1],
        (int(num_samples),),
    )
    first_samples = first_distribution.sample(
        resolved_keys[3],
        (int(num_samples),),
    )
    composed: list[Array] = []
    for index in range(int(num_samples)):
        advanced_batch = _require_batch(
            advance(first_batch, first_samples[index]),
            batch,
            owner="advance",
        )
        second_batch = _require_batch(
            condition(advanced_batch, second_duration),
            batch,
            owner="condition",
        )
        second_distribution = model.distribution(
            second_batch,
            key=resolved_keys[4 + 2 * index],
        )
        if second_distribution.uncertainty_source != "process":
            raise ValueError(
                "Composed transition distributions must use process uncertainty."
            )
        composed.append(
            second_distribution.sample(
                resolved_keys[5 + 2 * index],
                (1,),
            )[0]
        )
    composed_samples = jnp.stack(tuple(composed), axis=0)

    from ....uq._operator import operator_predictive_from_samples
    from ....uq._predictive import SampleAxis

    direct_predictive = operator_predictive_from_samples(
        direct_samples,
        direct_batch,
        direct_distribution.output_spec,
        sample_axes=(SampleAxis("__phydra_semigroup_direct", "process"),),
        field_name="output",
        query_name=direct_batch.single_query_name(),
    )
    composed_predictive = operator_predictive_from_samples(
        composed_samples,
        direct_batch,
        direct_distribution.output_spec,
        sample_axes=(SampleAxis("__phydra_semigroup_composed", "process"),),
        field_name="output",
        query_name=direct_batch.single_query_name(),
    )
    return direct_predictive, composed_predictive


@dataclass(frozen=True)
class SinkhornDistributionalSemigroupObjective:
    """Sinkhorn divergence between direct and independently composed process laws."""

    num_samples: int = 16
    measure: Literal["quadrature", "uniform"] = "quadrature"
    reduction: SemigroupReduction = "mean"
    weight: float = 1.0
    key_mode: SemigroupKeyMode = "fold_in"
    epsilon: float = 0.5
    cost: AbstractGroundCost | None = None
    solver: AbstractBalancedTransportSolver | None = None

    def __post_init__(self):
        if int(self.num_samples) < 2:
            raise ValueError("num_samples must be at least two.")
        if self.measure not in ("quadrature", "uniform"):
            raise ValueError("measure must be 'quadrature' or 'uniform'.")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        if not isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError("weight must be finite and nonnegative.")
        if self.key_mode not in ("fold_in", "split"):
            raise ValueError("key_mode must be 'fold_in' or 'split'.")
        if not isfinite(float(self.epsilon)) or float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        if self.cost is not None and not isinstance(self.cost, AbstractGroundCost):
            raise TypeError("cost must be an AbstractGroundCost or None.")
        if self.solver is not None and not isinstance(
            self.solver, AbstractBalancedTransportSolver
        ):
            raise TypeError(
                "solver must implement the balanced transport solver contract or be None."
            )

    def __call__(
        self,
        model: AbstractProbabilisticOperatorModel,
        batch: OperatorBatch,
        dt1: Any,
        dt2: Any,
        condition: Callable[[OperatorBatch, Array], OperatorBatch],
        advance: Callable[[OperatorBatch, Array], OperatorBatch],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        direct_predictive, composed_predictive = _distributional_predictives(
            model,
            batch,
            dt1,
            dt2,
            condition,
            advance,
            num_samples=self.num_samples,
            key_mode=self.key_mode,
            key=key,
            owner="SinkhornDistributionalSemigroupObjective",
        )
        from ....uq._transport_metrics import (
            operator_ensemble_sinkhorn_divergence,
        )

        result = operator_ensemble_sinkhorn_divergence(
            direct_predictive,
            composed_predictive,
            measure=self.measure,
            reduction=self.reduction,
            cost=self.cost,
            solver=self.solver,
            epsilon=self.epsilon,
        )
        if result.transport is None:
            raise AssertionError("Sinkhorn semigroup metric omitted transport diagnostics.")
        value = eqx.error_if(
            result.value,
            jnp.any(~result.transport.converged),
            "Sinkhorn distributional semigroup transport did not converge.",
        )
        return jnp.asarray(self.weight, dtype=value.dtype) * value


def conditioned_distributional_semigroup_loss(
    model: AbstractProbabilisticOperatorModel,
    batch: OperatorBatch,
    dt1: Any,
    dt2: Any,
    condition: Callable[[OperatorBatch, Array], OperatorBatch],
    advance: Callable[[OperatorBatch, Array], OperatorBatch],
    /,
    *,
    num_samples: int = 16,
    measure: Literal["quadrature", "uniform"] = "quadrature",
    beta: float = 1.0,
    chunk_size: int | None = None,
    reduction: SemigroupReduction = "mean",
    weight: float = 1.0,
    key_mode: SemigroupKeyMode = "fold_in",
    key: EvalKey = None,
) -> Array:
    """Evaluate distributional conditioned semigroup consistency."""
    return DistributionalSemigroupObjective(
        num_samples=num_samples,
        measure=measure,
        beta=beta,
        chunk_size=chunk_size,
        reduction=reduction,
        weight=weight,
        key_mode=key_mode,
    )(model, batch, dt1, dt2, condition, advance, key=key)


def conditioned_sinkhorn_semigroup_loss(
    model: AbstractProbabilisticOperatorModel,
    batch: OperatorBatch,
    dt1: Any,
    dt2: Any,
    condition: Callable[[OperatorBatch, Array], OperatorBatch],
    advance: Callable[[OperatorBatch, Array], OperatorBatch],
    /,
    *,
    num_samples: int = 16,
    measure: Literal["quadrature", "uniform"] = "quadrature",
    reduction: SemigroupReduction = "mean",
    weight: float = 1.0,
    key_mode: SemigroupKeyMode = "fold_in",
    epsilon: float = 0.5,
    cost: AbstractGroundCost | None = None,
    solver: AbstractBalancedTransportSolver | None = None,
    key: EvalKey = None,
) -> Array:
    """Evaluate Sinkhorn distributional conditioned semigroup consistency."""
    return SinkhornDistributionalSemigroupObjective(
        num_samples=num_samples,
        measure=measure,
        reduction=reduction,
        weight=weight,
        key_mode=key_mode,
        epsilon=epsilon,
        cost=cost,
        solver=solver,
    )(model, batch, dt1, dt2, condition, advance, key=key)


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
    "DistributionalSemigroupObjective",
    "SinkhornDistributionalSemigroupObjective",
    "SemigroupKeyMode",
    "SemigroupReduction",
    "conditioned_semigroup_consistency_loss",
    "conditioned_distributional_semigroup_loss",
    "conditioned_sinkhorn_semigroup_loss",
]
