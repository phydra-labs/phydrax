#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule


class CoresetSelection(StrictModule):
    """Fixed-capacity source indices and normalized weights for a coreset."""

    indices: Array
    log_weights: Array
    mask: Array
    diagnostics: Any
    method: str = eqx.field(static=True)

    def __init__(
        self,
        indices: Array,
        log_weights: Array,
        mask: Array,
        diagnostics: Any,
        /,
        *,
        method: str,
    ):
        indices_ = jnp.asarray(indices, dtype=jnp.int32)
        log_weights_ = jnp.asarray(log_weights, dtype=float)
        mask_ = jnp.asarray(mask, dtype=bool)
        if indices_.ndim != 1:
            raise ValueError("Coreset indices must be one-dimensional.")
        if log_weights_.shape != indices_.shape or mask_.shape != indices_.shape:
            raise ValueError(
                "Coreset indices, log_weights, and mask must have equal shape."
            )
        method_ = str(method)
        if not method_:
            raise ValueError("Coreset selection method must be non-empty.")
        self.indices = indices_
        self.log_weights = log_weights_
        self.mask = mask_
        self.diagnostics = diagnostics
        self.method = method_

    @property
    def capacity(self) -> int:
        return int(self.indices.shape[0])

    @property
    def active_points(self) -> Array:
        return jnp.sum(self.mask, dtype=jnp.int32)

    @property
    def weights(self) -> Array:
        return jnp.where(self.mask, jnp.exp(self.log_weights), 0.0)


class MomentRecombinationDiagnostics(StrictModule):
    """Mass, rank, and moment evidence for hierarchical recombination."""

    valid: Array
    active_points: Array
    numerical_rank: Array
    mass_error: Array
    max_moment_error: Array
    minimum_weight: Array
    log_source_mass: Array
    source_points: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    tree_depth: int = eqx.field(static=True)


class KernelHerdingDiagnostics(StrictModule):
    """Weighted empirical-measure discrepancy after kernel herding."""

    valid: Array
    active_points: Array
    mmd: Array
    minimum_weight: Array
    log_source_mass: Array
    source_points: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    coordinate_size: int = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)


class PivotedCholeskyDiagnostics(StrictModule):
    """Residual-kernel evidence for an inducing-point selection."""

    valid: Array
    active_points: Array
    initial_trace: Array
    residual_trace: Array
    explained_trace_fraction: Array
    source_points: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)


__all__ = [
    "CoresetSelection",
    "KernelHerdingDiagnostics",
    "MomentRecombinationDiagnostics",
    "PivotedCholeskyDiagnostics",
]
