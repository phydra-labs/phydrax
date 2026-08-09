#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._base import _as_point, _as_points, AbstractPositiveDefiniteKernel


class LinearKernel(AbstractPositiveDefiniteKernel):
    """Euclidean linear kernel with no implicit scale or offset."""

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_point = _as_point(left, name="left")
        right_point = _as_point(right, name="right")
        if left_point.shape != right_point.shape:
            raise ValueError("LinearKernel points must have equal coordinate size.")
        return jnp.vdot(left_point, right_point).real

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_points = _as_points(left, name="left")
        right_points = _as_points(right, name="right")
        if left_points.shape[1] != right_points.shape[1]:
            raise ValueError("LinearKernel designs must have equal coordinate size.")
        return left_points @ right_points.T

    def diagonal(self, points: ArrayLike, /) -> Array:
        point_design = _as_points(points, name="points")
        return jnp.sum(point_design * point_design, axis=1)

    @property
    def max_derivative_order(self) -> int | None:
        return None

    @property
    def is_unit_diagonal(self) -> bool:
        return False

    @property
    def kernel_id(self) -> str:
        return "LinearKernel"


__all__ = ["LinearKernel"]
