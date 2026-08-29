#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ...domain import DomainFunction, GridBatch, PointBatch
from ...operators.differential import regularized_delta
from ._adaptive import (
    AbstractCollocationPolicy,
    CollocationPolicy,
    PointwiseSamplingTerm,
)


class _NarrowBandTermProxy:
    def __init__(
        self, term: PointwiseSamplingTerm, policy: "NarrowBandCollocationPolicy"
    ):
        self.term = term
        self.narrow_band_policy = policy

    @property
    def sampling(self):
        return self.term.sampling

    @property
    def component(self):
        return self.term.component

    @property
    def policy(self):
        return self.narrow_band_policy

    def sample(self, *, key):
        return self.term.sample(key=key)

    def pointwise_score(self, functions, batch, /, *, key, **kwargs):
        residual = self.term.pointwise_score(functions, batch, key=key, **kwargs)
        name = self.narrow_band_policy.level_set_field
        if name not in functions:
            raise KeyError(f"Missing level-set field {name!r}.")
        band = regularized_delta(
            functions[name],
            width=self.narrow_band_policy.band_width,
        )(batch, key=key)
        if band.dims != residual.dims or band.data.shape != residual.data.shape:
            raise ValueError("Level-set band and residual collocation axes must match.")
        residual_data = jax.lax.stop_gradient(jnp.asarray(residual.data, dtype=float))
        band_data = jax.lax.stop_gradient(
            self.narrow_band_policy.band_width * jnp.asarray(band.data, dtype=float)
        )
        epsilon = self.narrow_band_policy.normalization_epsilon
        score = self.narrow_band_policy.residual_strength * residual_data / (
            jnp.mean(residual_data) + epsilon
        ) + self.narrow_band_policy.band_strength * band_data / (
            jnp.mean(band_data) + epsilon
        )
        return cx.Field(jnp.maximum(score, 0.0), dims=residual.dims)


class NarrowBandCollocationPolicy(AbstractCollocationPolicy):
    """Residual-adaptive collocation augmented by an implicit-interface band."""

    base_policy: CollocationPolicy
    level_set_field: str
    band_width: float
    band_strength: Array
    residual_strength: Array
    normalization_epsilon: Array
    refresh_every: int

    def __init__(
        self,
        level_set_field: str,
        band_width: float,
        /,
        *,
        base_policy: CollocationPolicy | None = None,
        band_strength: float = 1.0,
        residual_strength: float = 1.0,
        normalization_epsilon: float = 1.0e-12,
    ):
        name = str(level_set_field)
        width = float(band_width)
        band = float(band_strength)
        residual = float(residual_strength)
        epsilon = float(normalization_epsilon)
        if not name:
            raise ValueError("level_set_field must be non-empty.")
        if not jnp.isfinite(width) or width <= 0.0:
            raise ValueError("band_width must be finite and positive.")
        if not jnp.isfinite(band) or band < 0.0:
            raise ValueError("band_strength must be finite and nonnegative.")
        if not jnp.isfinite(residual) or residual < 0.0:
            raise ValueError("residual_strength must be finite and nonnegative.")
        if band == 0.0 and residual == 0.0:
            raise ValueError("At least one collocation score strength must be positive.")
        if not jnp.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be finite and positive.")
        base = CollocationPolicy("r3") if base_policy is None else base_policy
        if not isinstance(base, CollocationPolicy):
            raise TypeError("base_policy must be a CollocationPolicy or None.")
        self.base_policy = base
        self.level_set_field = name
        self.band_width = width
        self.band_strength = jnp.asarray(band)
        self.residual_strength = jnp.asarray(residual)
        self.normalization_epsilon = jnp.asarray(epsilon)
        self.refresh_every = base.refresh_every

    def initialize(self, constraint: PointwiseSamplingTerm, /, *, key):
        return self.base_policy.initialize(constraint, key=key)

    def should_refresh(self, population, iter_):
        return self.base_policy.should_refresh(population, iter_)

    def data_metrics(self, population, /):
        metrics = self.base_policy.data_metrics(population)
        metrics["narrow_band_width"] = jnp.asarray(self.band_width)
        metrics["narrow_band_strength"] = self.band_strength
        return metrics

    def refresh(
        self,
        constraint: PointwiseSamplingTerm,
        functions: Mapping[str, DomainFunction],
        population,
        /,
        *,
        key: Key[Array, ""],
        iter_: int | Array,
    ):
        proxy = _NarrowBandTermProxy(constraint, self)
        return self.base_policy.refresh(
            proxy,
            functions,
            population,
            key=key,
            iter_=iter_,
        )

    def loss_batch_and_weight(
        self,
        population,
        /,
    ) -> tuple[PointBatch | GridBatch, cx.Field | None]:
        return self.base_policy.loss_batch_and_weight(population)

    def refresh_residual_evaluations(self, population, /) -> int:
        return self.base_policy.refresh_residual_evaluations(population)


__all__ = ["NarrowBandCollocationPolicy"]
