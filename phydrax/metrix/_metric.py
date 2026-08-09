#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule
from ._chart import ChartTransition, CoordinateChart
from ._utils import _pointwise_array


LorentzianConvention: TypeAlias = Literal["mostly_plus", "mostly_minus"]


class MetricSignature(StrictModule):
    """Static positive and negative inertia counts of a nondegenerate metric."""

    positive: int = eqx.field(static=True)
    negative: int = eqx.field(static=True)

    def __init__(self, positive: int, negative: int, /):
        positive_count = int(positive)
        negative_count = int(negative)
        if positive_count < 0 or negative_count < 0:
            raise ValueError("Metric signature counts must be non-negative.")
        if positive_count + negative_count == 0:
            raise ValueError("A metric signature must have positive dimension.")
        self.positive = positive_count
        self.negative = negative_count

    @property
    def dimension(self) -> int:
        return self.positive + self.negative

    @property
    def index(self) -> int:
        return self.negative


class AbstractSemiRiemannianMetric(StrictModule):
    """Common nondegenerate signed-metric calculus in one coordinate chart."""

    matrix_function: AbstractAttribute[Callable[[Array], Array]]
    chart: AbstractAttribute[CoordinateChart]
    signature: AbstractAttribute[MetricSignature]

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        matrix = _pointwise_array(
            self.matrix_function,
            coordinates,
            self.chart.dimension,
        )
        expected = (self.chart.dimension, self.chart.dimension)
        if matrix.shape[-2:] != expected:
            raise ValueError(
                f"Metric matrix must have trailing shape {expected}; got {matrix.shape}."
            )
        return matrix

    def inverse(self, coordinates: ArrayLike, /) -> Array:
        matrix = self(coordinates)
        identity = jnp.broadcast_to(
            jnp.eye(self.chart.dimension, dtype=matrix.dtype),
            matrix.shape,
        )
        return jnp.linalg.solve(matrix, identity)

    @abstractmethod
    def determinant_sign(self, coordinates: ArrayLike, /) -> Array:
        """Return the sign of the metric determinant."""
        raise NotImplementedError

    @abstractmethod
    def log_abs_determinant(self, coordinates: ArrayLike, /) -> Array:
        """Return the logarithm of the absolute metric determinant."""
        raise NotImplementedError

    def volume_density(self, coordinates: ArrayLike, /) -> Array:
        return jnp.exp(0.5 * self.log_abs_determinant(coordinates))

    def log_volume_density(self, coordinates: ArrayLike, /) -> Array:
        return 0.5 * self.log_abs_determinant(coordinates)

    def bilinear(
        self,
        left: ArrayLike,
        right: ArrayLike,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        left_array = jnp.asarray(left)
        right_array = jnp.asarray(right)
        dimension = self.chart.dimension
        if left_array.shape[-1:] != (dimension,) or right_array.shape[-1:] != (
            dimension,
        ):
            raise ValueError(
                f"Metric pairing requires vector trailing dimension {dimension}."
            )
        return oe.contract(
            "...i,...ij,...j->...",
            left_array,
            self(coordinates),
            right_array,
        )

    def quadratic_form(
        self,
        vector: ArrayLike,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        return self.bilinear(vector, vector, coordinates)

    def flat(self, vector: ArrayLike, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(vector)
        if values.shape[-1:] != (self.chart.dimension,):
            raise ValueError(
                f"Metric flat map requires trailing dimension {self.chart.dimension}."
            )
        return oe.contract("...ij,...j->...i", self(coordinates), values)

    def sharp(self, covector: ArrayLike, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(covector)
        if values.shape[-1:] != (self.chart.dimension,):
            raise ValueError(
                f"Metric sharp map requires trailing dimension {self.chart.dimension}."
            )
        return oe.contract("...ij,...j->...i", self.inverse(coordinates), values)


class SemiRiemannianMetric(AbstractSemiRiemannianMetric):
    """A symmetric nondegenerate metric with a declared constant signature."""

    matrix_function: Callable[[Array], Array]
    chart: CoordinateChart
    signature: MetricSignature

    def __init__(
        self,
        matrix: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
        signature: MetricSignature,
    ):
        if not callable(matrix):
            raise TypeError("Metric matrix must be callable.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Metric chart must be a CoordinateChart.")
        if not isinstance(signature, MetricSignature):
            raise TypeError("signature must be a MetricSignature.")
        if signature.dimension != chart.dimension:
            raise ValueError(
                f"Metric signature dimension {signature.dimension} does not match "
                f"chart dimension {chart.dimension}."
            )
        self.matrix_function = matrix
        self.chart = chart
        self.signature = signature

    def determinant_sign(self, coordinates: ArrayLike, /) -> Array:
        sign, _ = jnp.linalg.slogdet(self(coordinates))
        return sign

    def log_abs_determinant(self, coordinates: ArrayLike, /) -> Array:
        _, value = jnp.linalg.slogdet(self(coordinates))
        return value


class RiemannianMetric(AbstractSemiRiemannianMetric):
    """A symmetric positive-definite metric tensor in one coordinate chart."""

    matrix_function: Callable[[Array], Array]
    chart: CoordinateChart
    signature: MetricSignature

    def __init__(
        self,
        matrix: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
    ):
        if not callable(matrix):
            raise TypeError("Metric matrix must be callable.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Metric chart must be a CoordinateChart.")
        self.matrix_function = matrix
        self.chart = chart
        self.signature = MetricSignature(chart.dimension, 0)

    def determinant_sign(self, coordinates: ArrayLike, /) -> Array:
        matrix = self(coordinates)
        return jnp.ones(matrix.shape[:-2], dtype=matrix.dtype)

    def log_abs_determinant(self, coordinates: ArrayLike, /) -> Array:
        factor = jnp.linalg.cholesky(self(coordinates))
        diagonal = jnp.diagonal(factor, axis1=-2, axis2=-1)
        return 2.0 * jnp.sum(jnp.log(diagonal), axis=-1)

    def inner(
        self,
        left: ArrayLike,
        right: ArrayLike,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        return self.bilinear(left, right, coordinates)

    def norm_squared(self, vector: ArrayLike, coordinates: ArrayLike, /) -> Array:
        return self.inner(vector, vector, coordinates)


class LorentzianMetric(AbstractSemiRiemannianMetric):
    """A signed metric with exactly one declared time direction."""

    matrix_function: Callable[[Array], Array]
    chart: CoordinateChart
    signature: MetricSignature
    convention: LorentzianConvention = eqx.field(static=True)

    def __init__(
        self,
        matrix: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
        convention: LorentzianConvention = "mostly_plus",
    ):
        if not callable(matrix):
            raise TypeError("Metric matrix must be callable.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Metric chart must be a CoordinateChart.")
        if chart.dimension < 2:
            raise ValueError("A Lorentzian metric requires dimension at least two.")
        if convention not in ("mostly_plus", "mostly_minus"):
            raise ValueError("convention must be 'mostly_plus' or 'mostly_minus'.")
        self.matrix_function = matrix
        self.chart = chart
        self.convention = convention
        self.signature = (
            MetricSignature(chart.dimension - 1, 1)
            if convention == "mostly_plus"
            else MetricSignature(1, chart.dimension - 1)
        )

    @property
    def timelike_sign(self) -> int:
        return -1 if self.convention == "mostly_plus" else 1

    def determinant_sign(self, coordinates: ArrayLike, /) -> Array:
        sign, _ = jnp.linalg.slogdet(self(coordinates))
        return sign

    def log_abs_determinant(self, coordinates: ArrayLike, /) -> Array:
        _, value = jnp.linalg.slogdet(self(coordinates))
        return value


class _EuclideanMetricMap(StrictModule):
    dimension: int

    def __init__(self, dimension: int, /):
        self.dimension = int(dimension)

    def __call__(self, coordinates: Array, /) -> Array:
        return jnp.eye(self.dimension, dtype=coordinates.dtype)


class _DiagonalMetricMap(StrictModule):
    diagonal: Callable[[Array], Array]
    dimension: int

    def __init__(self, diagonal: Callable[[Array], Array], dimension: int, /):
        self.diagonal = diagonal
        self.dimension = int(dimension)

    def __call__(self, coordinates: Array, /) -> Array:
        values = jnp.asarray(self.diagonal(coordinates))
        if values.shape != (self.dimension,):
            raise ValueError(
                "Pointwise diagonal metric output must have shape "
                f"{(self.dimension,)}; got {values.shape}."
            )
        return jnp.diag(values)


class _CholeskyMetricMap(StrictModule):
    model: Callable[[Array], Array]
    dimension: int
    minimum_diagonal: float

    def __init__(
        self,
        model: Callable[[Array], Array],
        dimension: int,
        minimum_diagonal: float,
        /,
    ):
        if minimum_diagonal < 0.0:
            raise ValueError("minimum_diagonal must be non-negative.")
        self.model = model
        self.dimension = int(dimension)
        self.minimum_diagonal = float(minimum_diagonal)

    def __call__(self, coordinates: Array, /) -> Array:
        raw = jnp.asarray(self.model(coordinates))
        expected = (self.dimension, self.dimension)
        if raw.shape != expected:
            raise ValueError(
                f"Pointwise Cholesky model must return shape {expected}; got {raw.shape}."
            )
        diagonal = jax.nn.softplus(jnp.diagonal(raw)) + self.minimum_diagonal
        factor = jnp.tril(raw, k=-1) + jnp.diag(diagonal)
        return factor @ jnp.swapaxes(factor, -1, -2)


class _PullbackMetricMap(StrictModule):
    target_metric: AbstractSemiRiemannianMetric
    transition: ChartTransition

    def __init__(
        self,
        target_metric: AbstractSemiRiemannianMetric,
        transition: ChartTransition,
        /,
    ):
        self.target_metric = target_metric
        self.transition = transition

    def __call__(self, coordinates: Array, /) -> Array:
        target_coordinates = self.transition.map_function(coordinates)
        jacobian = jax.jacfwd(self.transition.map_function)(coordinates)
        target_matrix = self.target_metric.matrix_function(target_coordinates)
        return oe.contract("ai,ab,bj->ij", jacobian, target_matrix, jacobian)


def euclidean_metric(chart: CoordinateChart, /) -> RiemannianMetric:
    return RiemannianMetric(_EuclideanMetricMap(chart.dimension), chart=chart)


def diagonal_metric(
    diagonal: Callable[[Array], Array],
    /,
    *,
    chart: CoordinateChart,
) -> RiemannianMetric:
    return RiemannianMetric(
        _DiagonalMetricMap(diagonal, chart.dimension),
        chart=chart,
    )


def diagonal_semi_riemannian_metric(
    diagonal: Callable[[Array], Array],
    /,
    *,
    chart: CoordinateChart,
    signature: MetricSignature,
) -> SemiRiemannianMetric:
    return SemiRiemannianMetric(
        _DiagonalMetricMap(diagonal, chart.dimension),
        chart=chart,
        signature=signature,
    )


def cholesky_metric(
    model: Callable[[Array], Array],
    /,
    *,
    chart: CoordinateChart,
    minimum_diagonal: float = 0.0,
) -> RiemannianMetric:
    return RiemannianMetric(
        _CholeskyMetricMap(model, chart.dimension, minimum_diagonal),
        chart=chart,
    )


def _validate_pullback(
    metric: AbstractSemiRiemannianMetric,
    transition: ChartTransition,
    /,
) -> None:
    if not transition.target.compatible_with(metric.chart):
        raise ValueError(
            "Pullback transition target chart must match the metric chart; got "
            f"{transition.target.name!r} and {metric.chart.name!r}."
        )
    if transition.source.dimension != transition.target.dimension:
        raise ValueError("Nondegenerate metric pullback requires equal dimensions.")


def pullback_metric(
    metric: RiemannianMetric,
    transition: ChartTransition,
    /,
) -> RiemannianMetric:
    if not isinstance(metric, RiemannianMetric):
        raise TypeError("pullback_metric requires a RiemannianMetric.")
    _validate_pullback(metric, transition)
    return RiemannianMetric(
        _PullbackMetricMap(metric, transition),
        chart=transition.source,
    )


def pullback_semi_riemannian_metric(
    metric: SemiRiemannianMetric,
    transition: ChartTransition,
    /,
) -> SemiRiemannianMetric:
    if not isinstance(metric, SemiRiemannianMetric):
        raise TypeError(
            "pullback_semi_riemannian_metric requires a SemiRiemannianMetric."
        )
    _validate_pullback(metric, transition)
    return SemiRiemannianMetric(
        _PullbackMetricMap(metric, transition),
        chart=transition.source,
        signature=metric.signature,
    )


def pullback_lorentzian_metric(
    metric: LorentzianMetric,
    transition: ChartTransition,
    /,
) -> LorentzianMetric:
    if not isinstance(metric, LorentzianMetric):
        raise TypeError("pullback_lorentzian_metric requires a LorentzianMetric.")
    _validate_pullback(metric, transition)
    return LorentzianMetric(
        _PullbackMetricMap(metric, transition),
        chart=transition.source,
        convention=metric.convention,
    )


__all__ = [
    "AbstractSemiRiemannianMetric",
    "LorentzianConvention",
    "LorentzianMetric",
    "MetricSignature",
    "RiemannianMetric",
    "SemiRiemannianMetric",
    "cholesky_metric",
    "diagonal_metric",
    "diagonal_semi_riemannian_metric",
    "euclidean_metric",
    "pullback_lorentzian_metric",
    "pullback_metric",
    "pullback_semi_riemannian_metric",
]
