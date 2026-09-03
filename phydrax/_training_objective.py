#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._precision import complex_precision_dtype, real_precision_dtype_name


@dataclass(frozen=True, slots=True)
class _ObjectiveContribution:
    """Scaled additive objective numerator and stop-gradient support."""

    numerator: Array
    support: Array
    log_scale: Array = 0.0

    def __post_init__(self) -> None:
        numerator = jnp.asarray(self.numerator)
        if not jnp.issubdtype(numerator.dtype, jnp.inexact):
            numerator = numerator.astype(float)
        support_dtype = jnp.real(numerator).dtype
        support = jnp.asarray(self.support, dtype=support_dtype)
        log_scale = jnp.asarray(self.log_scale, dtype=support_dtype)
        if numerator.shape != () or support.shape != () or log_scale.shape != ():
            raise ValueError(
                "Objective numerator, support, and log_scale must be scalar arrays."
            )
        support = eqx.error_if(
            support,
            ~jnp.isfinite(support) | (support < 0.0),
            "Objective support must be finite and nonnegative.",
        )
        log_scale = eqx.error_if(
            log_scale,
            ~jnp.isfinite(log_scale),
            "Objective log_scale must be finite.",
        )
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "support", jax.lax.stop_gradient(support))
        object.__setattr__(self, "log_scale", jax.lax.stop_gradient(log_scale))

    @property
    def value(self) -> Array:
        return _normalize_objective_contribution(self)


def _normalize_objective_contribution(
    contribution: _ObjectiveContribution,
    /,
) -> Array:
    """Normalize one contribution, returning exact zero for zero support."""

    if not isinstance(contribution, _ObjectiveContribution):
        raise TypeError("contribution must be an _ObjectiveContribution.")
    positive = contribution.support > 0.0
    denominator = jnp.where(
        positive,
        contribution.support,
        jnp.ones_like(contribution.support),
    )
    return jnp.where(
        positive,
        contribution.numerator / denominator,
        jnp.zeros_like(contribution.numerator),
    )


def _merge_objective_contributions(
    left: _ObjectiveContribution,
    right: _ObjectiveContribution,
    /,
) -> _ObjectiveContribution:
    """Merge two independently scaled additive contributions."""

    support_dtype = jnp.result_type(left.support.dtype, right.support.dtype)
    left_support = left.support.astype(support_dtype)
    right_support = right.support.astype(support_dtype)
    left_scale = left.log_scale.astype(support_dtype)
    right_scale = right.log_scale.astype(support_dtype)
    left_active = left_support > 0.0
    right_active = right_support > 0.0
    scale = jnp.maximum(
        jnp.where(left_active, left_scale, -jnp.inf),
        jnp.where(right_active, right_scale, -jnp.inf),
    )
    scale = jnp.where(left_active | right_active, scale, 0.0)
    left_factor = jnp.where(left_active, jnp.exp(left_scale - scale), 0.0)
    right_factor = jnp.where(right_active, jnp.exp(right_scale - scale), 0.0)
    numerator_dtype = jnp.result_type(left.numerator.dtype, right.numerator.dtype)
    numerator = jnp.where(
        left_active,
        left_factor * left.numerator.astype(numerator_dtype),
        jnp.zeros((), dtype=numerator_dtype),
    )
    numerator = numerator + jnp.where(
        right_active,
        right_factor * right.numerator.astype(numerator_dtype),
        jnp.zeros((), dtype=numerator_dtype),
    )
    support = left_factor * left_support + right_factor * right_support
    return _ObjectiveContribution(numerator, support, scale)


def _combine_objective_contributions(
    contributions: Sequence[_ObjectiveContribution],
    /,
) -> _ObjectiveContribution:
    """Combine scaled additive contributions without intermediate normalization."""

    values = tuple(contributions)
    if not values:
        raise ValueError("At least one objective contribution is required.")
    if any(not isinstance(value, _ObjectiveContribution) for value in values):
        raise TypeError("Every contribution must be an _ObjectiveContribution.")
    result = values[0]
    for value in values[1:]:
        result = _merge_objective_contributions(result, value)
    return result


def _tree_accumulation_zeros(tree: Any, dtype: Any, /) -> Any:
    def zero(value: Any) -> Any:
        if value is None:
            return None
        resolved_dtype = (
            complex_precision_dtype(dtype)
            if jnp.issubdtype(value.dtype, jnp.complexfloating)
            else real_precision_dtype_name(dtype)
        )
        return jnp.zeros(value.shape, dtype=resolved_dtype)

    return jax.tree_util.tree_map(
        zero,
        tree,
        is_leaf=lambda value: value is None,
    )


def _tree_cast_for_accumulation(tree: Any, dtype: Any, /) -> Any:
    def cast(value: Any) -> Any:
        if value is None:
            return None
        resolved_dtype = (
            complex_precision_dtype(dtype)
            if jnp.issubdtype(value.dtype, jnp.complexfloating)
            else real_precision_dtype_name(dtype)
        )
        return value.astype(resolved_dtype)

    return jax.tree_util.tree_map(
        cast,
        tree,
        is_leaf=lambda value: value is None,
    )


@dataclass(frozen=True, slots=True)
class _GradientAccumulationState:
    """Host-controlled, log-stable accumulator of objective numerator gradients."""

    gradient_numerator: Any
    support: Array
    log_scale: Array
    microsteps: int
    accumulation_dtype: str

    @classmethod
    def empty(
        cls,
        gradient_template: Any,
        /,
        *,
        accumulation_dtype: Any,
    ) -> _GradientAccumulationState:
        dtype = real_precision_dtype_name(accumulation_dtype)
        return cls(
            gradient_numerator=_tree_accumulation_zeros(
                gradient_template,
                dtype,
            ),
            support=jnp.asarray(0.0, dtype=dtype),
            log_scale=jnp.asarray(0.0, dtype=dtype),
            microsteps=0,
            accumulation_dtype=dtype,
        )

    def add(
        self,
        numerator_gradient: Any,
        contribution: _ObjectiveContribution,
        /,
    ) -> _GradientAccumulationState:
        if not isinstance(contribution, _ObjectiveContribution):
            raise TypeError("contribution must be an _ObjectiveContribution.")
        support = contribution.support.astype(self.support.dtype)
        incoming_scale = contribution.log_scale.astype(self.support.dtype)
        current_active = self.support > 0.0
        incoming_active = support > 0.0
        scale = jnp.maximum(
            jnp.where(current_active, self.log_scale, -jnp.inf),
            jnp.where(incoming_active, incoming_scale, -jnp.inf),
        )
        scale = jnp.where(current_active | incoming_active, scale, 0.0)
        current_factor = jnp.where(
            current_active,
            jnp.exp(self.log_scale - scale),
            0.0,
        )
        incoming_factor = jnp.where(
            incoming_active,
            jnp.exp(incoming_scale - scale),
            0.0,
        )
        incoming = _tree_cast_for_accumulation(
            numerator_gradient,
            self.accumulation_dtype,
        )

        def merge(current: Any, value: Any) -> Any:
            if current is None:
                return None
            scaled_incoming = jnp.where(
                incoming_active,
                incoming_factor * value,
                jnp.zeros_like(value),
            )
            return current_factor * current + scaled_incoming

        gradient = jax.tree_util.tree_map(
            merge,
            self.gradient_numerator,
            incoming,
            is_leaf=lambda value: value is None,
        )
        accumulated_support = current_factor * self.support + incoming_factor * support
        return _GradientAccumulationState(
            gradient_numerator=gradient,
            support=accumulated_support,
            log_scale=scale,
            microsteps=self.microsteps + 1,
            accumulation_dtype=self.accumulation_dtype,
        )

    @property
    def has_positive_support(self) -> Array:
        return self.support > 0.0

    @property
    def is_empty(self) -> bool:
        return self.microsteps == 0

    def normalized_gradient(self, like: Any, /) -> Any:
        support = eqx.error_if(
            self.support,
            self.support <= 0.0,
            "Cannot normalize a zero-support gradient accumulation window.",
        )

        def normalize(value: Any, reference: Any) -> Any:
            if value is None:
                return None
            return (value / support).astype(reference.dtype)

        return jax.tree_util.tree_map(
            normalize,
            self.gradient_numerator,
            like,
            is_leaf=lambda value: value is None,
        )


@dataclass(frozen=True, slots=True)
class _ObjectiveAccumulator:
    """Online scalar accumulator for independently scaled contributions."""

    contribution: _ObjectiveContribution | None = None

    @property
    def is_empty(self) -> bool:
        return self.contribution is None

    def add(
        self,
        contribution: _ObjectiveContribution,
        /,
    ) -> _ObjectiveAccumulator:
        if self.contribution is None:
            return _ObjectiveAccumulator(contribution)
        return _ObjectiveAccumulator(
            _merge_objective_contributions(self.contribution, contribution)
        )

    @property
    def value(self) -> Array:
        if self.contribution is None:
            raise ValueError("Cannot normalize an empty objective accumulator.")
        return self.contribution.value


__all__ = [
    "_GradientAccumulationState",
    "_ObjectiveAccumulator",
    "_ObjectiveContribution",
    "_combine_objective_contributions",
    "_normalize_objective_contribution",
]
