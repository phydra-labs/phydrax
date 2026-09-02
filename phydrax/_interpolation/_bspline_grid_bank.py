#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import ceil, isfinite
from numbers import Integral

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._numerics._quadrature_rules import gauss_legendre_data
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._bspline_grid import BSplineGrid, TrainableBSplineGrid


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


class TrainableBSplineGridBank(StrictModule):
    """Homogeneous fixed-capacity bank of independently trainable knot grids."""

    raw_span_logits: Array
    degree: int = eqx.field(static=True)
    _intervals: tuple[tuple[float, float], ...] = eqx.field(static=True)
    _minimum_spans: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        raw_span_logits: ArrayLike,
        degree: int,
        /,
        *,
        intervals: ArrayLike,
        minimum_spans: ArrayLike | None = None,
    ):
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise TypeError("Trainable B-spline grid-bank degree must be an integer.")
        degree_ = int(degree)
        if degree_ < 1:
            raise ValueError("Trainable B-spline grid-bank degree must be positive.")
        logits = jnp.asarray(raw_span_logits)
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError(
                "raw_span_logits must have shape (num_inputs, num_intervals)."
            )
        if jnp.issubdtype(logits.dtype, jnp.complexfloating):
            raise TypeError("raw_span_logits must be real-valued.")
        logits_host = np.asarray(logits, dtype=float)
        if not np.all(np.isfinite(logits_host)):
            raise ValueError("raw_span_logits must be finite.")

        intervals_ = jnp.asarray(intervals, dtype=jnp.result_type(logits, float))
        if intervals_.shape == (2,):
            intervals_ = jnp.broadcast_to(intervals_, (int(logits.shape[0]), 2))
        if intervals_.shape != (int(logits.shape[0]), 2):
            raise ValueError("intervals must have shape (num_inputs, 2).")
        intervals_host = np.asarray(intervals_, dtype=float)
        extents = intervals_host[:, 1] - intervals_host[:, 0]
        if not np.all(np.isfinite(intervals_host)) or np.any(extents <= 0.0):
            raise ValueError(
                "Every trainable grid interval must be finite and increasing."
            )

        if minimum_spans is None:
            minimum_host = extents / (1000.0 * int(logits.shape[1]))
        else:
            minimum_host = np.asarray(minimum_spans, dtype=float)
            if minimum_host.ndim == 0:
                minimum_host = np.broadcast_to(minimum_host, (int(logits.shape[0]),))
        if minimum_host.shape != (int(logits.shape[0]),):
            raise ValueError("minimum_spans must contain one value per input.")
        if (
            not np.all(np.isfinite(minimum_host))
            or np.any(minimum_host <= 0.0)
            or np.any(int(logits.shape[1]) * minimum_host >= extents)
        ):
            raise ValueError(
                "minimum_spans must be positive and leave movable length in every row."
            )

        dtype = jnp.result_type(logits, intervals_, float)
        self.raw_span_logits = logits.astype(dtype)
        self.degree = degree_
        self._intervals = tuple(
            (float(lower), float(upper)) for lower, upper in intervals_host
        )
        self._minimum_spans = tuple(float(value) for value in minimum_host)

    @classmethod
    def open_uniform(
        cls,
        num_grids: int,
        degree: int = 3,
        num_intervals: int = 8,
        /,
        *,
        intervals: ArrayLike = (-1.0, 1.0),
        minimum_spans: ArrayLike | None = None,
    ) -> TrainableBSplineGridBank:
        """Construct independently trainable rows with uniform initial spans."""
        if isinstance(num_grids, bool) or not isinstance(num_grids, Integral):
            raise TypeError("num_grids must be an integer.")
        if isinstance(num_intervals, bool) or not isinstance(num_intervals, Integral):
            raise TypeError("num_intervals must be an integer.")
        if int(num_grids) < 1 or int(num_intervals) < 1:
            raise ValueError("num_grids and num_intervals must be positive.")
        return cls(
            jnp.zeros((int(num_grids), int(num_intervals))),
            degree,
            intervals=intervals,
            minimum_spans=minimum_spans,
        )

    @classmethod
    def from_grids(
        cls,
        grids: tuple[BSplineGrid | TrainableBSplineGrid, ...],
        /,
        *,
        minimum_spans: ArrayLike | None = None,
    ) -> TrainableBSplineGridBank:
        """Stack homogeneous open grids while preserving their current knots."""
        if not grids:
            raise ValueError("from_grids requires at least one grid.")
        trainable = tuple(
            grid
            if isinstance(grid, TrainableBSplineGrid)
            else TrainableBSplineGrid.from_grid(grid)
            for grid in grids
        )
        degree = trainable[0].degree
        count = trainable[0].num_intervals
        if any(
            grid.degree != degree or grid.num_intervals != count for grid in trainable
        ):
            raise ValueError("Grid-bank rows must share degree and interval count.")
        intervals = jnp.asarray(
            tuple((grid.lower, grid.upper) for grid in trainable),
            dtype=trainable[0].raw_span_logits.dtype,
        )
        minimum = (
            jnp.asarray(
                tuple(grid.minimum_span for grid in trainable),
                dtype=trainable[0].raw_span_logits.dtype,
            )
            if minimum_spans is None
            else minimum_spans
        )
        if minimum_spans is None:
            logits = jnp.stack(tuple(grid.raw_span_logits for grid in trainable))
            return cls(
                logits,
                degree,
                intervals=intervals,
                minimum_spans=minimum,
            )
        reconstructed = tuple(
            TrainableBSplineGrid.from_grid(
                BSplineGrid(grid.knots, degree),
                minimum_span=float(np.asarray(minimum)[index]),
            )
            for index, grid in enumerate(trainable)
        )
        return cls(
            jnp.stack(tuple(grid.raw_span_logits for grid in reconstructed)),
            degree,
            intervals=intervals,
            minimum_spans=minimum,
        )

    @property
    def num_grids(self) -> int:
        return int(self.raw_span_logits.shape[0])

    @property
    def num_intervals(self) -> int:
        return int(self.raw_span_logits.shape[1])

    @property
    def intervals(self) -> Array:
        return jnp.asarray(self._intervals, dtype=self.raw_span_logits.dtype)

    @property
    def minimum_spans(self) -> Array:
        return jnp.asarray(self._minimum_spans, dtype=self.raw_span_logits.dtype)

    @property
    def coefficient_count(self) -> int:
        return self.num_intervals + self.degree

    @property
    def span_widths(self) -> Array:
        extents = self.intervals[:, 1] - self.intervals[:, 0]
        movable = extents - self.num_intervals * self.minimum_spans
        return self.minimum_spans[:, None] + movable[:, None] * jnn.softmax(
            self.raw_span_logits,
            axis=-1,
        )

    @property
    def breakpoints(self) -> Array:
        interior = self.intervals[:, :1] + jnp.cumsum(self.span_widths, axis=-1)[:, :-1]
        return jnp.concatenate(
            (self.intervals[:, :1], interior, self.intervals[:, 1:]),
            axis=-1,
        )

    @property
    def knots(self) -> Array:
        return jnp.concatenate(
            (
                jnp.broadcast_to(
                    self.intervals[:, :1],
                    (self.num_grids, self.degree + 1),
                ),
                self.breakpoints[:, 1:-1],
                jnp.broadcast_to(
                    self.intervals[:, 1:],
                    (self.num_grids, self.degree + 1),
                ),
            ),
            axis=-1,
        )

    @property
    def active_interval(self) -> tuple[Array, Array]:
        return self.intervals[:, 0], self.intervals[:, 1]

    @property
    def greville_abscissae(self) -> Array:
        indices = (
            jnp.arange(self.coefficient_count)[:, None]
            + jnp.arange(1, self.degree + 1)[None, :]
        )
        return jnp.mean(self.knots[:, indices], axis=-1)

    def quadrature(self, polynomial_degree: int, /) -> tuple[Array, Array]:
        if (
            isinstance(polynomial_degree, bool)
            or not isinstance(polynomial_degree, Integral)
            or polynomial_degree < 0
        ):
            raise ValueError("quadrature polynomial_degree must be nonnegative.")
        order = max(1, ceil((int(polynomial_degree) + 1) / 2))
        reference = gauss_legendre_data(order)
        nodes = jnp.asarray(reference.nodes, dtype=self.raw_span_logits.dtype)
        weights = jnp.asarray(reference.weights, dtype=self.raw_span_logits.dtype)
        lower = self.breakpoints[:, :-1, None]
        upper = self.breakpoints[:, 1:, None]
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        return (
            (midpoint + half_width * nodes).reshape((self.num_grids, -1)),
            (half_width * weights).reshape((self.num_grids, -1)),
        )

    def derivative_quadrature(self, derivative_order: int, /) -> tuple[Array, Array]:
        if (
            isinstance(derivative_order, bool)
            or not isinstance(derivative_order, Integral)
            or not 0 <= derivative_order <= self.degree
        ):
            raise ValueError("derivative_order must lie between zero and degree.")
        return self.quadrature(2 * (self.degree - int(derivative_order)))

    def regularization(
        self,
        *,
        entropy_weight: float = 1.0,
        neighbor_weight: float = 1.0,
    ) -> Array:
        entropy = float(entropy_weight)
        neighbor = float(neighbor_weight)
        if (
            not isfinite(entropy)
            or entropy < 0.0
            or not isfinite(neighbor)
            or neighbor < 0.0
        ):
            raise ValueError(
                "Knot regularization weights must be finite and nonnegative."
            )
        extents = self.intervals[:, 1] - self.intervals[:, 0]
        probabilities = self.span_widths / extents[:, None]
        relative_entropy = jnp.sum(
            probabilities * jnp.log(probabilities * self.num_intervals),
            axis=-1,
        )
        log_widths = jnp.log(self.span_widths)
        neighbor_variation = jnp.sum(jnp.diff(log_widths, axis=-1) ** 2, axis=-1)
        return entropy * relative_entropy + neighbor * neighbor_variation


__all__ = ["BSplineGridBank", "TrainableBSplineGridBank"]
