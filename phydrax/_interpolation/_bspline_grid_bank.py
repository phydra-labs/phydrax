#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._bspline_grid import BSplineGrid


class BSplineGridBank(StrictModule, NonTrainableState):
    """Homogeneous fixed B-spline grids aligned with one KAN input axis."""

    knots: Array
    degree: int = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)
    positive_span_count: int = eqx.field(static=True)

    def __init__(self, knots: ArrayLike, degree: int, /):
        knots_ = jnp.asarray(knots)
        if knots_.ndim != 2 or knots_.shape[0] == 0:
            raise ValueError(
                "B-spline grid-bank knots must have shape (num_grids, knot_count)."
            )
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise TypeError("B-spline grid-bank degree must be an integer.")
        grids = tuple(BSplineGrid(row, int(degree)) for row in knots_)
        coefficient_count = grids[0].coefficient_count
        span_count = grids[0].num_intervals
        active_interval = np.asarray(grids[0].active_interval)
        if any(grid.coefficient_count != coefficient_count for grid in grids):
            raise ValueError(
                "Every B-spline grid-bank row must have one coefficient count."
            )
        if any(grid.num_intervals != span_count for grid in grids):
            raise ValueError(
                "Every B-spline grid-bank row must have one positive-span count."
            )
        if any(
            not np.array_equal(np.asarray(grid.active_interval), active_interval)
            for grid in grids[1:]
        ):
            raise ValueError(
                "Every B-spline grid-bank row must have the same active interval."
            )
        self.knots = jnp.stack(tuple(grid.knots for grid in grids))
        self.degree = int(degree)
        self.coefficient_count = coefficient_count
        self.positive_span_count = span_count

    @classmethod
    def from_grids(cls, grids: tuple[BSplineGrid, ...], /) -> BSplineGridBank:
        """Stack validated homogeneous grids without repeating evaluation storage."""
        if not grids:
            raise ValueError("A B-spline grid bank requires at least one grid.")
        if not all(isinstance(grid, BSplineGrid) for grid in grids):
            raise TypeError("Every grid-bank entry must be a BSplineGrid.")
        degree = grids[0].degree
        if any(grid.degree != degree for grid in grids[1:]):
            raise ValueError("Every B-spline grid-bank row must have one degree.")
        return cls(jnp.stack(tuple(grid.knots for grid in grids)), degree)

    @classmethod
    def repeat(cls, grid: BSplineGrid, count: int, /) -> BSplineGridBank:
        """Realize one fixed grid independently for each input channel."""
        if not isinstance(grid, BSplineGrid):
            raise TypeError("grid must be a BSplineGrid.")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("grid-bank count must be a positive integer.")
        return cls(
            jnp.broadcast_to(grid.knots, (int(count), grid.knots.size)), grid.degree
        )

    @property
    def num_grids(self) -> int:
        return int(self.knots.shape[0])

    @property
    def active_interval(self) -> tuple[Array, Array]:
        return self.knots[0, self.degree], self.knots[0, self.coefficient_count]

    @property
    def grids(self) -> tuple[BSplineGrid, ...]:
        return tuple(BSplineGrid(row, self.degree) for row in self.knots)

    @property
    def greville_abscissae(self) -> Array:
        return jnp.stack(tuple(grid.greville_abscissae for grid in self.grids))

    def derivative_quadrature(self, derivative_order: int, /) -> tuple[Array, Array]:
        rules = tuple(grid.derivative_quadrature(derivative_order) for grid in self.grids)
        return (
            jnp.stack(tuple(rule[0] for rule in rules)),
            jnp.stack(tuple(rule[1] for rule in rules)),
        )


__all__ = ["BSplineGridBank"]
