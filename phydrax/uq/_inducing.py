#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.coresets import (
    randomized_pivoted_cholesky,
    RandomizedPivotedCholesky,
)
from phydrax.kernels import AbstractPositiveDefiniteKernel

from .._doc import DOC_KEY0
from .._strict import StrictModule


class InducingPointSelection(StrictModule):
    """Selected source points and residual-kernel approximation evidence."""

    points: Array
    indices: Array
    diagnostics: Any

    def __init__(self, points: Array, indices: Array, diagnostics: Any, /):
        points_ = jnp.asarray(points, dtype=float)
        indices_ = jnp.asarray(indices, dtype=jnp.int32)
        if points_.ndim != 2:
            raise ValueError("Selected inducing points must be two-dimensional.")
        if indices_.shape != (points_.shape[0],):
            raise ValueError("Inducing indices must match the selected point count.")
        self.points = points_
        self.indices = indices_
        self.diagnostics = diagnostics


def select_inducing_points(
    observation_points: ArrayLike | cx.Field,
    num_points: int,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kernel: AbstractPositiveDefiniteKernel | None = None,
) -> InducingPointSelection:
    """Select sparse-GP inducing inputs with randomized pivoted Cholesky."""
    raw = (
        observation_points.data
        if isinstance(observation_points, cx.Field)
        else observation_points
    )
    points = jnp.asarray(raw, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2:
        raise ValueError("observation_points must have shape (num_points, coordinates).")
    if bool(jnp.any(~jnp.isfinite(points))):
        raise ValueError("observation_points must be finite.")
    selection = randomized_pivoted_cholesky(
        points,
        RandomizedPivotedCholesky(num_points, kernel=kernel),
        key=key,
    )
    if not bool(selection.diagnostics.valid):
        raise ValueError(
            "The requested inducing-point count exceeds the numerical kernel rank."
        )
    return InducingPointSelection(
        points[selection.indices],
        selection.indices,
        selection.diagnostics,
    )


__all__ = ["InducingPointSelection", "select_inducing_points"]
