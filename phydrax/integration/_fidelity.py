#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..fidelity import FidelityLevelSpec, FidelityPath
from ._multilevel import MultilevelSampleBatch
from ._targets import MultilevelTarget


FidelityInputSampler: TypeAlias = Callable[[Array, Any], Any]
FidelityLevelEvaluator: TypeAlias = Callable[[FidelityLevelSpec, Any], Any]


class FidelityBatchEvaluation(StrictModule):
    """Canonical batched observable, validity, and cost for one fidelity level."""

    values: Any
    valid: Array
    costs: Array
    level_id: str = eqx.field(static=True)
    evaluator_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: Any,
        /,
        *,
        level_id: str,
        evaluator_id: str,
        valid: ArrayLike | None = None,
        costs: ArrayLike = 1.0,
        evidence_id: str | None = None,
    ):
        leaves = tuple(jax.tree_util.tree_leaves(values))
        if not leaves:
            raise ValueError("values must contain at least one array leaf.")
        arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
        if any(array.ndim == 0 for array in arrays):
            raise ValueError("Fidelity batch value leaves require a leading sample axis.")
        counts = {int(array.shape[0]) for array in arrays}
        if len(counts) != 1:
            raise ValueError("Fidelity batch value leaves must share a sample axis.")
        count = counts.pop()
        finite = jnp.ones((count,), dtype=bool)
        for array in arrays:
            axes = tuple(range(1, array.ndim))
            leaf_finite = jnp.isfinite(array)
            if axes:
                leaf_finite = jnp.all(leaf_finite, axis=axes)
            finite = finite & leaf_finite
        valid_ = finite if valid is None else jnp.asarray(valid, dtype=bool) & finite
        if valid_.shape != (count,):
            raise ValueError("valid must contain one entry per fidelity sample.")
        costs_ = jnp.broadcast_to(jnp.asarray(costs, dtype=float), (count,))
        if bool(jnp.any(~jnp.isfinite(costs_) | (costs_ <= 0.0))):
            raise ValueError("Fidelity evaluation costs must be finite and positive.")
        level = str(level_id)
        evaluator = str(evaluator_id)
        if not level or not evaluator:
            raise ValueError("level_id and evaluator_id must be non-empty.")
        evidence = f"{evaluator}:{level}" if evidence_id is None else str(evidence_id)
        if not evidence:
            raise ValueError("evidence_id must be non-empty.")
        self.values = values
        self.valid = valid_
        self.costs = costs_
        self.level_id = level
        self.evaluator_id = evaluator
        self.evidence_id = evidence

    @property
    def num_samples(self) -> int:
        return int(self.valid.size)


class FidelityMultilevelSampler(StrictModule):
    """Adapt canonical input and level evaluators to a prefix-addressed MLMC sampler."""

    path: FidelityPath
    input_sampler: FidelityInputSampler
    level_evaluator: FidelityLevelEvaluator
    sampler_id: str = eqx.field(static=True)
    input_sampler_id: str = eqx.field(static=True)
    evaluator_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: FidelityPath,
        input_sampler: FidelityInputSampler,
        level_evaluator: FidelityLevelEvaluator,
        /,
        *,
        sampler_id: str,
        input_sampler_id: str,
        evaluator_id: str,
    ):
        if not isinstance(path, FidelityPath):
            raise TypeError("path must be a FidelityPath.")
        if not callable(input_sampler) or not callable(level_evaluator):
            raise TypeError("input_sampler and level_evaluator must be callable.")
        sampler = str(sampler_id)
        input_identifier = str(input_sampler_id)
        evaluator = str(evaluator_id)
        if not sampler or not input_identifier or not evaluator:
            raise ValueError("Fidelity sampler identities must be non-empty.")
        self.path = path
        self.input_sampler = input_sampler
        self.level_evaluator = level_evaluator
        self.sampler_id = sampler
        self.input_sampler_id = input_identifier
        self.evaluator_id = evaluator

    def __call__(
        self,
        level_index: int,
        sample_indices: Array,
        root_key: Any,
        /,
    ) -> MultilevelSampleBatch:
        level = int(level_index)
        if level < 0 or level >= self.path.num_levels:
            raise IndexError("Fidelity multilevel level_index is out of bounds.")
        indices = jnp.asarray(sample_indices, dtype=jnp.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("sample_indices must be a non-empty vector.")
        if bool(jnp.any(indices < 0)) or bool(jnp.any(jnp.diff(indices) != 1)):
            raise ValueError("sample_indices must be consecutive and non-negative.")
        inputs = self.input_sampler(indices, jax.random.fold_in(root_key, level))
        fine = self.level_evaluator(self.path.levels[level], inputs)
        self._validate_evaluation(fine, level, int(indices.size))
        if level == 0:
            return MultilevelSampleBatch(
                fine.values,
                None,
                indices,
                fine.costs,
                level_index=0,
                fine_valid=fine.valid,
                pair_ids=indices,
                provenance=self.sampler_id,
            )
        coarse = self.level_evaluator(self.path.levels[level - 1], inputs)
        self._validate_evaluation(coarse, level - 1, int(indices.size))
        return MultilevelSampleBatch(
            fine.values,
            coarse.values,
            indices,
            fine.costs + coarse.costs,
            level_index=level,
            fine_valid=fine.valid,
            coarse_valid=coarse.valid,
            pair_ids=indices,
            provenance=self.sampler_id,
        )

    def _validate_evaluation(
        self,
        evaluation: Any,
        level_index: int,
        count: int,
        /,
    ) -> None:
        if not isinstance(evaluation, FidelityBatchEvaluation):
            raise TypeError("level_evaluator must return FidelityBatchEvaluation values.")
        expected = self.path.levels[level_index].level_id
        if evaluation.level_id != expected:
            raise ValueError(
                f"Fidelity evaluator returned level {evaluation.level_id!r}; expected {expected!r}."
            )
        if evaluation.evaluator_id != self.evaluator_id:
            raise ValueError("Fidelity evaluator identity changed during MLMC execution.")
        if evaluation.num_samples != count:
            raise ValueError(
                "Fidelity evaluator output must preserve the requested sample count."
            )


def fidelity_multilevel_target(
    path: FidelityPath,
    input_sampler: FidelityInputSampler,
    level_evaluator: FidelityLevelEvaluator,
    /,
    *,
    sampler_id: str,
    input_sampler_id: str,
    evaluator_id: str,
) -> MultilevelTarget:
    """Construct a native multilevel target over one explicit fidelity path."""

    sampler = FidelityMultilevelSampler(
        path,
        input_sampler,
        level_evaluator,
        sampler_id=sampler_id,
        input_sampler_id=input_sampler_id,
        evaluator_id=evaluator_id,
    )
    return MultilevelTarget(path, sampler, sampler_id=sampler.sampler_id)


__all__ = [
    "FidelityBatchEvaluation",
    "FidelityInputSampler",
    "FidelityLevelEvaluator",
    "FidelityMultilevelSampler",
    "fidelity_multilevel_target",
]
