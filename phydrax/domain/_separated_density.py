#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._function import DomainFunction


class _SeparatedEvaluator(StrictModule):
    factors: tuple[Any, ...]
    rank_weights: Array

    def __call__(self, *coordinates, key=None):
        if len(coordinates) != len(self.factors):
            raise ValueError("Separated density requires one coordinate per factor.")
        products = self.rank_weights
        for factor, coordinate in zip(self.factors, coordinates, strict=True):
            values = jnp.asarray(factor(coordinate, key=key))
            if values.shape[-1] != self.rank_weights.size:
                raise ValueError("Every coordinate factor must end in the fixed rank.")
            products = products * values
        return jnp.sum(products, axis=-1)


class SeparatedLogDensityField(StrictModule):
    """Fixed-rank CP-style log density without a hidden tensor grid."""

    factors: tuple[Any, ...]
    rank_weights: Array
    rank: int = eqx.field(static=True)
    state_labels: tuple[str, ...] = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        factors: tuple[Any, ...],
        rank: int,
        state_labels: tuple[str, ...],
        /,
        *,
        rank_weights: Array | None = None,
        field_id: str | None = None,
    ):
        selected = tuple(factors)
        labels = tuple(state_labels)
        fixed_rank = int(rank)
        if fixed_rank <= 0:
            raise ValueError("rank must be positive.")
        if not selected or any(not callable(item) for item in selected):
            raise TypeError("factors must contain coordinate callables.")
        if len(labels) != len(selected) or len(set(labels)) != len(labels):
            raise ValueError("state_labels must uniquely align with factors.")
        weights = (
            jnp.ones((fixed_rank,), dtype=float)
            if rank_weights is None
            else jnp.asarray(rank_weights)
        )
        if weights.shape != (fixed_rank,) or not bool(jnp.all(jnp.isfinite(weights))):
            raise ValueError("rank_weights must be finite and rank-aligned.")
        resolved_id = field_id or canonical_fingerprint(
            {
                "kind": "separated-log-density-v1",
                "rank": fixed_rank,
                "state_labels": labels,
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("field_id must be non-empty.")
        self.factors = selected
        self.rank_weights = weights
        self.rank = fixed_rank
        self.state_labels = labels
        self.field_id = resolved_id

    def __call__(self, *coordinates, key=None):
        return _SeparatedEvaluator(self.factors, self.rank_weights)(*coordinates, key=key)

    def as_domain_function(self, domain: Any, /) -> DomainFunction:
        """Bind the represented factorization to one explicit labeled domain."""
        if tuple(self.state_labels) != tuple(domain.labels):
            raise ValueError("domain labels must exactly match state_labels.")
        return DomainFunction(
            domain=domain,
            deps=self.state_labels,
            func=_SeparatedEvaluator(self.factors, self.rank_weights),
            metadata={
                "approximation_kind": "fixed-separated-rank",
                "separated_rank": self.rank,
                "field_id": self.field_id,
            },
        )


__all__ = ["SeparatedLogDensityField"]
