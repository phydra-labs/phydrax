#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


@dataclass(frozen=True, slots=True)
class _ObjectiveContribution:
    """Additive weighted objective numerator and stop-gradient support."""

    numerator: Array
    support: Array

    def __post_init__(self) -> None:
        numerator = jnp.asarray(self.numerator)
        if not jnp.issubdtype(numerator.dtype, jnp.inexact):
            numerator = numerator.astype(float)
        support = jnp.asarray(self.support, dtype=numerator.dtype)
        if numerator.shape != () or support.shape != ():
            raise ValueError("Objective numerator and support must be scalar arrays.")
        support = eqx.error_if(
            support,
            ~jnp.isfinite(support) | (support < 0.0),
            "Objective support must be finite and nonnegative.",
        )
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "support", jax.lax.stop_gradient(support))

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


def _combine_objective_contributions(
    contributions: Sequence[_ObjectiveContribution],
    /,
) -> _ObjectiveContribution:
    """Combine additive contributions without normalizing intermediate values."""

    values = tuple(contributions)
    if not values:
        raise ValueError("At least one objective contribution is required.")
    if any(not isinstance(value, _ObjectiveContribution) for value in values):
        raise TypeError("Every contribution must be an _ObjectiveContribution.")
    numerator = sum(
        (value.numerator for value in values),
        start=jnp.zeros_like(values[0].numerator),
    )
    support = sum(
        (value.support for value in values),
        start=jnp.zeros_like(values[0].support),
    )
    return _ObjectiveContribution(numerator, support)


__all__ = [
    "_ObjectiveContribution",
    "_combine_objective_contributions",
    "_normalize_objective_contribution",
]
