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


class BSplineGrid(StrictModule, NonTrainableState):
    """Validated fixed B-spline knot grid with reusable span geometry."""

    knots: Array
    breakpoints: Array
    degree: int = eqx.field(static=True)
    continuity_orders: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, knots: ArrayLike, degree: int, /):
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise TypeError("B-spline grid degree must be an integer.")
        degree_ = int(degree)
        if degree_ < 0:
            raise ValueError("B-spline grid degree must be nonnegative.")

        knots_raw = jnp.asarray(knots)
        if knots_raw.ndim != 1:
            raise ValueError("B-spline grid knots must be a rank-one array.")
        if jnp.issubdtype(knots_raw.dtype, jnp.complexfloating):
            raise TypeError("B-spline grid knots must be real-valued.")
        knots_host = np.asarray(knots_raw, dtype=float)
        if not np.all(np.isfinite(knots_host)):
            raise ValueError("B-spline grid knots must be finite.")
        if np.any(np.diff(knots_host) < 0.0):
            raise ValueError("B-spline grid knots must be nondecreasing.")
        _, multiplicities = np.unique(knots_host, return_counts=True)
        if np.any(multiplicities > degree_ + 1):
            raise ValueError("B-spline knot multiplicity cannot exceed degree + 1.")

        coefficient_count = int(knots_host.size) - degree_ - 1
        if coefficient_count <= degree_:
            raise ValueError(
                "B-spline grid knots must define at least degree + 1 coefficients."
            )
        lower = float(knots_host[degree_])
        upper = float(knots_host[coefficient_count])
        if not upper > lower:
            raise ValueError("B-spline grid must define a nonempty active interval.")

        active_knots = knots_host[degree_ : coefficient_count + 1]
        breakpoints = np.unique(active_knots)
        if breakpoints.size < 2:
            raise ValueError("B-spline grid must contain at least one positive span.")
        interior = breakpoints[1:-1]
        continuity_orders = tuple(
            degree_ - int(np.count_nonzero(knots_host == knot)) for knot in interior
        )

        dtype = jnp.result_type(knots_raw, float)
        self.knots = knots_raw.astype(dtype)
        self.breakpoints = jnp.asarray(breakpoints, dtype=dtype)
        self.degree = degree_
        self.continuity_orders = continuity_orders

    @classmethod
    def open_uniform(
        cls,
        degree: int = 3,
        num_intervals: int = 8,
        /,
        *,
        interval: tuple[float, float] = (-1.0, 1.0),
    ) -> BSplineGrid:
        """Construct an open-uniform grid on a finite interval."""
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise TypeError("B-spline grid degree must be an integer.")
        if isinstance(num_intervals, bool) or not isinstance(num_intervals, Integral):
            raise TypeError("num_intervals must be an integer.")
        degree_ = int(degree)
        intervals = int(num_intervals)
        if degree_ < 0:
            raise ValueError("B-spline grid degree must be nonnegative.")
        if intervals < 1:
            raise ValueError("num_intervals must be positive.")
        lower, upper = (float(value) for value in interval)
        if not isfinite(lower) or not isfinite(upper) or not upper > lower:
            raise ValueError("B-spline grid interval must be finite and increasing.")
        breakpoints = jnp.linspace(lower, upper, intervals + 1)
        knots = jnp.concatenate(
            (
                jnp.full((degree_ + 1,), lower),
                breakpoints[1:-1],
                jnp.full((degree_ + 1,), upper),
            )
        )
        return cls(knots, degree_)

    @property
    def coefficient_count(self) -> int:
        return int(self.knots.shape[0]) - self.degree - 1

    @property
    def active_interval(self) -> tuple[Array, Array]:
        return self.breakpoints[0], self.breakpoints[-1]

    @property
    def num_intervals(self) -> int:
        return int(self.breakpoints.shape[0]) - 1

    @property
    def is_uniform(self) -> bool:
        widths = np.diff(np.asarray(self.breakpoints))
        return bool(np.allclose(widths, widths[0], rtol=1e-12, atol=1e-14))

    @property
    def greville_abscissae(self) -> Array:
        if self.degree == 0:
            raise ValueError("Degree-zero B-splines do not have Greville abscissae.")
        return jnp.stack(
            tuple(
                jnp.mean(self.knots[index + 1 : index + self.degree + 1])
                for index in range(self.coefficient_count)
            )
        )

    def quadrature(self, polynomial_degree: int, /) -> tuple[Array, Array]:
        """Return Gauss-Legendre nodes and weights over every positive span."""
        if (
            isinstance(polynomial_degree, bool)
            or not isinstance(polynomial_degree, Integral)
            or polynomial_degree < 0
        ):
            raise ValueError("quadrature polynomial_degree must be nonnegative.")
        order = max(1, ceil((int(polynomial_degree) + 1) / 2))
        reference = gauss_legendre_data(order)
        reference_nodes = np.asarray(reference.nodes)
        reference_weights = np.asarray(reference.weights)
        breakpoints = np.asarray(self.breakpoints)
        nodes: list[float] = []
        weights: list[float] = []
        for lower, upper in zip(breakpoints[:-1], breakpoints[1:], strict=True):
            midpoint = 0.5 * (lower + upper)
            half_width = 0.5 * (upper - lower)
            nodes.extend((midpoint + half_width * reference_nodes).tolist())
            weights.extend((half_width * reference_weights).tolist())
        return (
            jnp.asarray(nodes, dtype=self.knots.dtype),
            jnp.asarray(weights, dtype=self.knots.dtype),
        )

    def derivative_quadrature(self, derivative_order: int, /) -> tuple[Array, Array]:
        """Return exact quadrature for the squared requested derivative."""
        if (
            isinstance(derivative_order, bool)
            or not isinstance(derivative_order, Integral)
            or not 0 <= derivative_order <= self.degree
        ):
            raise ValueError("derivative_order must lie between zero and degree.")
        return self.quadrature(2 * (self.degree - int(derivative_order)))


class TrainableBSplineGrid(StrictModule):
    """Open B-spline grid with ordered fixed-count spans parameterized by logits."""

    raw_span_logits: Array
    degree: int = eqx.field(static=True)
    lower: float = eqx.field(static=True)
    upper: float = eqx.field(static=True)
    minimum_span: float = eqx.field(static=True)

    def __init__(
        self,
        raw_span_logits: ArrayLike,
        degree: int,
        /,
        *,
        interval: tuple[float, float] = (-1.0, 1.0),
        minimum_span: float | None = None,
    ):
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise TypeError("Trainable B-spline grid degree must be an integer.")
        degree_ = int(degree)
        if degree_ < 1:
            raise ValueError("Trainable B-spline grid degree must be positive.")
        logits = jnp.asarray(raw_span_logits)
        if logits.ndim != 1 or logits.size == 0:
            raise ValueError("raw_span_logits must be a nonempty rank-one array.")
        if jnp.issubdtype(logits.dtype, jnp.complexfloating):
            raise TypeError("raw_span_logits must be real-valued.")
        logits_host = np.asarray(logits, dtype=float)
        if not np.all(np.isfinite(logits_host)):
            raise ValueError("raw_span_logits must be finite.")
        lower, upper = (float(value) for value in interval)
        if not isfinite(lower) or not isfinite(upper) or not upper > lower:
            raise ValueError("Trainable B-spline interval must be finite and increasing.")
        intervals = int(logits.size)
        minimum = (
            (upper - lower) / (1000.0 * intervals)
            if minimum_span is None
            else float(minimum_span)
        )
        if (
            not isfinite(minimum)
            or minimum <= 0.0
            or intervals * minimum >= upper - lower
        ):
            raise ValueError(
                "minimum_span must be positive and leave movable interval length."
            )
        self.raw_span_logits = logits.astype(jnp.result_type(logits, float))
        self.degree = degree_
        self.lower = lower
        self.upper = upper
        self.minimum_span = minimum

    @classmethod
    def open_uniform(
        cls,
        degree: int = 3,
        num_intervals: int = 8,
        /,
        *,
        interval: tuple[float, float] = (-1.0, 1.0),
        minimum_span: float | None = None,
    ) -> TrainableBSplineGrid:
        """Construct uniform initial spans with fixed endpoint multiplicities."""
        if isinstance(num_intervals, bool) or not isinstance(num_intervals, Integral):
            raise TypeError("num_intervals must be an integer.")
        if num_intervals < 1:
            raise ValueError("num_intervals must be positive.")
        return cls(
            jnp.zeros((int(num_intervals),)),
            degree,
            interval=interval,
            minimum_span=minimum_span,
        )

    @classmethod
    def from_grid(
        cls,
        grid: BSplineGrid,
        /,
        *,
        minimum_span: float | None = None,
    ) -> TrainableBSplineGrid:
        """Initialize logits from a simple open fixed grid."""
        if not isinstance(grid, BSplineGrid):
            raise TypeError("grid must be a BSplineGrid.")
        knots_host = np.asarray(grid.knots)
        lower, upper = (float(value) for value in grid.active_interval)
        if (
            np.count_nonzero(knots_host == lower) != grid.degree + 1
            or np.count_nonzero(knots_host == upper) != grid.degree + 1
            or any(order != grid.degree - 1 for order in grid.continuity_orders)
        ):
            raise ValueError(
                "Trainable grids require simple interior knots and open endpoints."
            )
        spans = np.diff(np.asarray(grid.breakpoints))
        minimum = (
            (upper - lower) / (1000.0 * grid.num_intervals)
            if minimum_span is None
            else float(minimum_span)
        )
        movable = upper - lower - grid.num_intervals * minimum
        adjusted = spans - minimum
        if movable <= 0.0 or np.any(adjusted <= 0.0):
            raise ValueError(
                "minimum_span must be strictly smaller than every grid span."
            )
        logits = np.log(adjusted / movable)
        logits = logits - np.mean(logits)
        return cls(
            jnp.asarray(logits, dtype=grid.knots.dtype),
            grid.degree,
            interval=(lower, upper),
            minimum_span=minimum,
        )

    @property
    def num_intervals(self) -> int:
        return int(self.raw_span_logits.size)

    @property
    def coefficient_count(self) -> int:
        return self.num_intervals + self.degree

    @property
    def span_widths(self) -> Array:
        movable = self.upper - self.lower - self.num_intervals * self.minimum_span
        return self.minimum_span + movable * jnn.softmax(self.raw_span_logits)

    @property
    def breakpoints(self) -> Array:
        interior = self.lower + jnp.cumsum(self.span_widths)[:-1]
        return jnp.concatenate(
            (
                jnp.asarray([self.lower], dtype=self.raw_span_logits.dtype),
                interior,
                jnp.asarray([self.upper], dtype=self.raw_span_logits.dtype),
            )
        )

    @property
    def knots(self) -> Array:
        return jnp.concatenate(
            (
                jnp.full(
                    (self.degree + 1,),
                    self.lower,
                    dtype=self.raw_span_logits.dtype,
                ),
                self.breakpoints[1:-1],
                jnp.full(
                    (self.degree + 1,),
                    self.upper,
                    dtype=self.raw_span_logits.dtype,
                ),
            )
        )

    @property
    def active_interval(self) -> tuple[Array, Array]:
        return self.breakpoints[0], self.breakpoints[-1]

    @property
    def continuity_orders(self) -> tuple[int, ...]:
        return (self.degree - 1,) * (self.num_intervals - 1)

    @property
    def greville_abscissae(self) -> Array:
        return jnp.stack(
            tuple(
                jnp.mean(self.knots[index + 1 : index + self.degree + 1])
                for index in range(self.coefficient_count)
            )
        )

    def quadrature(self, polynomial_degree: int, /) -> tuple[Array, Array]:
        """Return differentiable fixed-shape quadrature over all live spans."""
        if (
            isinstance(polynomial_degree, bool)
            or not isinstance(polynomial_degree, Integral)
            or polynomial_degree < 0
        ):
            raise ValueError("quadrature polynomial_degree must be nonnegative.")
        order = max(1, ceil((int(polynomial_degree) + 1) / 2))
        reference = gauss_legendre_data(order)
        reference_nodes_ = jnp.asarray(reference.nodes, dtype=self.raw_span_logits.dtype)
        reference_weights_ = jnp.asarray(
            reference.weights, dtype=self.raw_span_logits.dtype
        )
        lower = self.breakpoints[:-1, None]
        upper = self.breakpoints[1:, None]
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        nodes = midpoint + half_width * reference_nodes_[None, :]
        weights = half_width * reference_weights_[None, :]
        return nodes.reshape((-1,)), weights.reshape((-1,))

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
        """Penalize collapsed allocation and abrupt neighboring span ratios."""
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
        probabilities = self.span_widths / (self.upper - self.lower)
        relative_entropy = jnp.sum(
            probabilities * jnp.log(probabilities * self.num_intervals)
        )
        log_widths = jnp.log(self.span_widths)
        neighbor_variation = jnp.sum(jnp.diff(log_widths) ** 2)
        return entropy * relative_entropy + neighbor * neighbor_variation


__all__ = ["BSplineGrid", "TrainableBSplineGrid"]
