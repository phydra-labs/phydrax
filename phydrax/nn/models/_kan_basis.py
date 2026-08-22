#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import comb, isfinite
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import ArrayLike, Key

from ..._interpolation import (
    bspline_batched_evaluate,
    bspline_evaluate,
    BSplineGrid,
    BSplineGridBank,
    TrainableBSplineGrid,
)
from ..._polynomial._orthogonal import (
    OrthogonalFamily,
    standard_affine_coefficients,
    standard_series_value,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


EdgeInitialization = Literal["default", "identity"]


_FAMILY_ALIASES: dict[str, OrthogonalFamily] = {
    "chebyshev": "chebyshev",
    "cheb": "chebyshev",
    "t": "chebyshev",
    "legendre": "legendre",
    "leg": "legendre",
    "p": "legendre",
    "hermite": "hermite",
    "herm": "hermite",
    "hermite_e": "hermite_e",
    "herme": "hermite_e",
    "laguerre": "laguerre",
    "lag": "laguerre",
}


def _validate_edge_arrays(
    coefficients: ArrayLike,
    inputs: ArrayLike,
    coefficient_count: int,
) -> tuple[Array, Array]:
    coefficients_ = jnp.asarray(coefficients)
    inputs_ = jnp.asarray(inputs)
    if coefficients_.ndim != 3:
        raise ValueError("Edge coefficients must have shape (out_size, in_size, count).")
    if int(coefficients_.shape[-1]) != coefficient_count:
        raise ValueError(
            f"Expected {coefficient_count} coefficients per edge, got "
            f"{coefficients_.shape[-1]}."
        )
    if inputs_.shape != coefficients_.shape[:2]:
        raise ValueError(
            "Edge inputs must have shape (out_size, in_size) matching the "
            "coefficient array."
        )
    return coefficients_, inputs_


class AbstractEdgeBasis(StrictModule):
    """Typed numerical contract for one family of scalar KAN edge functions."""

    @property
    @abstractmethod
    def degree(self) -> int:
        """Return the polynomial or piecewise-polynomial degree."""

    @property
    @abstractmethod
    def coefficient_count(self) -> int:
        """Return the number of trainable coefficients on each edge."""

    @abstractmethod
    def for_layer(self, in_size: int, out_size: int, /) -> AbstractEdgeBasis:
        """Realize any dimension-dependent basis state for one KAN layer."""

    @abstractmethod
    def initialize_coefficients(
        self,
        out_size: int,
        in_size: int,
        initialization: EdgeInitialization,
        key: Key[Array, ""],
    ) -> Any:
        """Initialize one output-by-input edge-parameter PyTree."""

    @abstractmethod
    def evaluate(self, coefficients: Any, inputs: ArrayLike) -> Array:
        """Evaluate one scalar input for every output-by-input edge."""

    @abstractmethod
    def regularization(self, coefficients: Any) -> Array:
        """Return this basis family's unscaled parameter penalty."""


class OrthogonalPolynomialEdgeBasis(AbstractEdgeBasis):
    """Global orthogonal-polynomial KAN edge basis in standard normalization."""

    _degree: int = eqx.field(static=True)
    family: OrthogonalFamily = eqx.field(static=True)
    regularization_start: int = eqx.field(static=True)
    regularization_power: float = eqx.field(static=True)

    def __init__(
        self,
        degree: int = 5,
        *,
        family: str = "chebyshev",
        regularization_start: int = 2,
        regularization_power: float = 2.0,
    ):
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValueError(
                "Orthogonal polynomial degree must be a nonnegative integer."
            )
        family_ = _FAMILY_ALIASES.get(str(family).lower())
        if family_ is None:
            raise ValueError(f"Unsupported orthogonal polynomial family: {family!r}.")
        if (
            isinstance(regularization_start, bool)
            or not isinstance(regularization_start, int)
            or regularization_start < 0
        ):
            raise ValueError("regularization_start must be a nonnegative integer.")
        if not isfinite(float(regularization_power)) or regularization_power < 0.0:
            raise ValueError("regularization_power must be finite and nonnegative.")
        self._degree = degree
        self.family = family_
        self.regularization_start = regularization_start
        self.regularization_power = float(regularization_power)

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def coefficient_count(self) -> int:
        return self.degree + 1

    def for_layer(self, in_size: int, out_size: int, /) -> AbstractEdgeBasis:
        del in_size, out_size
        return self

    def initialize_coefficients(
        self,
        out_size: int,
        in_size: int,
        initialization: EdgeInitialization,
        key: Key[Array, ""],
    ) -> Array:
        coefficients = jnp.zeros((out_size, in_size, self.coefficient_count))
        if initialization == "identity":
            if self.degree < 1:
                raise ValueError(
                    "Identity initialization requires polynomial degree >= 1."
                )
            slope = jnp.eye(out_size, in_size, dtype=coefficients.dtype)
            affine = standard_affine_coefficients(
                self.family, jnp.zeros_like(slope), slope
            )
            return coefficients.at[..., :2].set(affine)
        if initialization == "default":
            if self.degree == 0:
                return coefficients
            slopes = 0.05 * jr.normal(key, (out_size, in_size), dtype=coefficients.dtype)
            affine = standard_affine_coefficients(
                self.family, jnp.zeros_like(slopes), slopes
            )
            return coefficients.at[..., :2].set(affine)
        raise ValueError(f"Unknown edge initialization: {initialization!r}.")

    def evaluate(self, coefficients: ArrayLike, inputs: ArrayLike) -> Array:
        coefficients_, inputs_ = _validate_edge_arrays(
            coefficients, inputs, self.coefficient_count
        )
        return jax.vmap(
            lambda coefficient_row, input_row: jax.vmap(
                lambda edge_coefficients, edge_input: standard_series_value(
                    self.family, edge_coefficients, edge_input
                )
            )(coefficient_row, input_row)
        )(coefficients_, inputs_)

    def regularization(self, coefficients: ArrayLike) -> Array:
        coefficients_ = jnp.asarray(coefficients)
        orders = jnp.arange(self.coefficient_count, dtype=coefficients_.real.dtype)
        weights = jnp.where(
            orders >= self.regularization_start,
            orders**self.regularization_power,
            0.0,
        )
        magnitudes = jnp.real(coefficients_ * jnp.conj(coefficients_))
        return jnp.sum(magnitudes * weights**2)


class _BSplineQuadrature(StrictModule, NonTrainableState):
    points: Array
    weights: Array

    def __init__(self, points: Array, weights: Array, /):
        self.points = points
        self.weights = weights


class BSplineEdgeBasis(AbstractEdgeBasis):
    """Compactly supported B-spline KAN edge basis on fixed or trainable grids."""

    grid: BSplineGrid | BSplineGridBank | TrainableBSplineGrid
    regularization_order: int = eqx.field(static=True)
    per_input: bool = eqx.field(static=True)
    knot_entropy_weight: float = eqx.field(static=True)
    knot_neighbor_weight: float = eqx.field(static=True)
    _quadrature: _BSplineQuadrature | None

    def __init__(
        self,
        degree: int | None = None,
        *,
        num_intervals: int | None = None,
        grid: BSplineGrid | BSplineGridBank | TrainableBSplineGrid | None = None,
        regularization_order: int = 2,
        per_input: bool = False,
        knot_entropy_weight: float = 0.0,
        knot_neighbor_weight: float = 0.0,
    ):
        if grid is None:
            degree_ = 3 if degree is None else degree
            interval_count = 8 if num_intervals is None else num_intervals
            grid_ = BSplineGrid.open_uniform(degree_, interval_count)
        else:
            if not isinstance(grid, (BSplineGrid, BSplineGridBank, TrainableBSplineGrid)):
                raise TypeError(
                    "grid must be a BSplineGrid, BSplineGridBank, or "
                    "TrainableBSplineGrid."
                )
            if degree is not None and degree != grid.degree:
                raise ValueError("degree must match the explicit B-spline grid.")
            if num_intervals is not None:
                raise ValueError(
                    "num_intervals cannot be combined with an explicit grid."
                )
            grid_ = grid
        if grid_.degree < 1:
            raise ValueError("B-spline KAN edge degree must be positive.")
        if per_input and isinstance(grid_, TrainableBSplineGrid):
            raise ValueError("Trainable per-input B-spline grid banks are not supported.")
        if (
            isinstance(regularization_order, bool)
            or not isinstance(regularization_order, int)
            or not 1 <= regularization_order <= grid_.degree
        ):
            raise ValueError("regularization_order must lie between one and degree.")
        entropy_weight = float(knot_entropy_weight)
        neighbor_weight = float(knot_neighbor_weight)
        if (
            not isfinite(entropy_weight)
            or entropy_weight < 0.0
            or not isfinite(neighbor_weight)
            or neighbor_weight < 0.0
        ):
            raise ValueError(
                "Knot regularization weights must be finite and nonnegative."
            )
        if isinstance(grid_, TrainableBSplineGrid):
            quadrature = None
        else:
            quadrature_points, quadrature_weights = grid_.derivative_quadrature(
                regularization_order
            )
            quadrature = _BSplineQuadrature(
                quadrature_points,
                quadrature_weights,
            )
        self.grid = grid_
        self.regularization_order = regularization_order
        self.per_input = bool(per_input or isinstance(grid_, BSplineGridBank))
        self.knot_entropy_weight = entropy_weight
        self.knot_neighbor_weight = neighbor_weight
        self._quadrature = quadrature

    @property
    def degree(self) -> int:
        return self.grid.degree

    @property
    def coefficient_count(self) -> int:
        return self.grid.coefficient_count

    def for_layer(self, in_size: int, out_size: int, /) -> AbstractEdgeBasis:
        del out_size
        if isinstance(self.grid, BSplineGridBank):
            if self.grid.num_grids != in_size:
                raise ValueError(
                    "B-spline grid-bank size must match the KAN layer input size."
                )
            return self
        if isinstance(self.grid, TrainableBSplineGrid):
            return self
        if not self.per_input:
            return self
        return BSplineEdgeBasis(
            grid=BSplineGridBank.repeat(self.grid, in_size),
            regularization_order=self.regularization_order,
            per_input=True,
            knot_entropy_weight=self.knot_entropy_weight,
            knot_neighbor_weight=self.knot_neighbor_weight,
        )

    def initialize_coefficients(
        self,
        out_size: int,
        in_size: int,
        initialization: EdgeInitialization,
        key: Key[Array, ""],
    ) -> Array:
        if isinstance(self.grid, BSplineGridBank):
            if self.grid.num_grids != in_size:
                raise ValueError(
                    "B-spline grid-bank size must match the KAN layer input size."
                )
            greville = self.grid.greville_abscissae[None, :, :]
        else:
            greville = self.grid.greville_abscissae
        if initialization == "identity":
            slopes = jnp.eye(out_size, in_size)
        elif initialization == "default":
            slopes = 0.05 * jr.normal(key, (out_size, in_size))
        else:
            raise ValueError(f"Unknown edge initialization: {initialization!r}.")
        return slopes[..., None] * greville

    def evaluate(self, coefficients: ArrayLike, inputs: ArrayLike) -> Array:
        coefficients_, inputs_ = _validate_edge_arrays(
            coefficients, inputs, self.coefficient_count
        )
        if isinstance(self.grid, BSplineGridBank):
            if self.grid.num_grids != int(coefficients_.shape[1]):
                raise ValueError(
                    "B-spline grid-bank size must match the edge input axis."
                )
            return bspline_batched_evaluate(
                self.grid.knots,
                coefficients_,
                inputs_,
                degree=self.degree,
                bounds="clip",
            ).values
        case_shape = tuple(int(size) for size in coefficients_.shape[:2])
        return bspline_evaluate(
            self.grid.knots,
            coefficients_,
            inputs_,
            degree=self.degree,
            bounds="clip",
            case_shape=case_shape,
        ).values

    def regularization(self, coefficients: ArrayLike) -> Array:
        coefficients_ = jnp.asarray(coefficients)
        if (
            coefficients_.ndim != 3
            or int(coefficients_.shape[-1]) != self.coefficient_count
        ):
            raise ValueError(
                "B-spline edge coefficients must have shape "
                "(out_size, in_size, coefficient_count)."
            )
        if self._quadrature is None:
            quadrature_points, quadrature_weights = self.grid.derivative_quadrature(
                self.regularization_order
            )
        else:
            quadrature_points = self._quadrature.points
            quadrature_weights = self._quadrature.weights
        quadrature_points = quadrature_points.astype(coefficients_.real.dtype)
        quadrature_weights = quadrature_weights.astype(coefficients_.real.dtype)
        if isinstance(self.grid, BSplineGridBank):
            if self.grid.num_grids != int(coefficients_.shape[1]):
                raise ValueError(
                    "B-spline grid-bank size must match the edge input axis."
                )
            query = jnp.broadcast_to(
                quadrature_points[None, :, :],
                (
                    int(coefficients_.shape[0]),
                    int(coefficients_.shape[1]),
                    int(quadrature_points.shape[1]),
                ),
            )
            derivative = bspline_batched_evaluate(
                self.grid.knots,
                coefficients_,
                query,
                degree=self.degree,
                derivative_order=self.regularization_order,
                bounds="error",
            ).values
            magnitude = jnp.real(derivative * jnp.conj(derivative))
            return jnp.sum(magnitude * quadrature_weights[None, :, :])
        case_shape = tuple(int(size) for size in coefficients_.shape[:2])
        query = jnp.broadcast_to(
            quadrature_points,
            (*case_shape, int(quadrature_points.shape[0])),
        )
        derivative = bspline_evaluate(
            self.grid.knots,
            coefficients_,
            query,
            degree=self.degree,
            derivative_order=self.regularization_order,
            bounds="error",
            case_shape=case_shape,
        ).values
        magnitude = jnp.real(derivative * jnp.conj(derivative))
        penalty = jnp.sum(magnitude * quadrature_weights)
        if isinstance(self.grid, TrainableBSplineGrid):
            penalty = penalty + self.grid.regularization(
                entropy_weight=self.knot_entropy_weight,
                neighbor_weight=self.knot_neighbor_weight,
            )
        return penalty


class RationalBSplineEdgeParameters(StrictModule):
    """Trainable control values and bounded log-weights for rational spline edges."""

    control_values: Array
    raw_log_weights: Array

    def __init__(
        self,
        control_values: ArrayLike,
        raw_log_weights: ArrayLike,
        /,
    ):
        control_values_ = jnp.asarray(control_values)
        raw_log_weights_ = jnp.asarray(raw_log_weights)
        if control_values_.ndim != 3:
            raise ValueError(
                "Rational B-spline control values must have shape "
                "(out_size, in_size, coefficient_count)."
            )
        if raw_log_weights_.shape != control_values_.shape:
            raise ValueError(
                "Rational B-spline raw log-weights must match the control values."
            )
        if jnp.issubdtype(raw_log_weights_.dtype, jnp.complexfloating):
            raise TypeError("Rational B-spline raw log-weights must be real-valued.")
        self.control_values = control_values_
        self.raw_log_weights = raw_log_weights_


class RationalBSplineEdgeBasis(AbstractEdgeBasis):
    """Positive-weight rational B-spline KAN edges on fixed grids."""

    grid: BSplineGrid | BSplineGridBank
    regularization_order: int = eqx.field(static=True)
    per_input: bool = eqx.field(static=True)
    maximum_log_weight: float = eqx.field(static=True)
    weight_magnitude_weight: float = eqx.field(static=True)
    weight_variation_weight: float = eqx.field(static=True)
    minimum_denominator: float = eqx.field(static=True)
    denominator_weight: float = eqx.field(static=True)
    _quadrature: _BSplineQuadrature

    def __init__(
        self,
        degree: int | None = None,
        *,
        num_intervals: int | None = None,
        grid: BSplineGrid | BSplineGridBank | None = None,
        regularization_order: int = 2,
        per_input: bool = False,
        maximum_log_weight: float = 4.0,
        weight_magnitude_weight: float = 1.0e-4,
        weight_variation_weight: float = 1.0e-4,
        minimum_denominator: float = 1.0e-4,
        denominator_weight: float = 1.0,
    ):
        if grid is None:
            degree_ = 3 if degree is None else degree
            interval_count = 8 if num_intervals is None else num_intervals
            grid_ = BSplineGrid.open_uniform(degree_, interval_count)
        else:
            if not isinstance(grid, (BSplineGrid, BSplineGridBank)):
                raise TypeError("grid must be a fixed BSplineGrid or BSplineGridBank.")
            if degree is not None and degree != grid.degree:
                raise ValueError("degree must match the explicit rational spline grid.")
            if num_intervals is not None:
                raise ValueError(
                    "num_intervals cannot be combined with an explicit grid."
                )
            grid_ = grid
        if grid_.degree < 1:
            raise ValueError("Rational B-spline KAN edge degree must be positive.")
        if (
            isinstance(regularization_order, bool)
            or not isinstance(regularization_order, int)
            or not 1 <= regularization_order <= grid_.degree
        ):
            raise ValueError("regularization_order must lie between one and degree.")
        static_values = (
            float(maximum_log_weight),
            float(weight_magnitude_weight),
            float(weight_variation_weight),
            float(minimum_denominator),
            float(denominator_weight),
        )
        if (
            not all(isfinite(value) for value in static_values)
            or static_values[0] <= 0.0
            or any(value < 0.0 for value in static_values[1:])
            or static_values[3] <= 0.0
        ):
            raise ValueError(
                "Rational weight bounds and regularizers must be finite and nonnegative, "
                "with positive maximum_log_weight and minimum_denominator."
            )
        quadrature_points, quadrature_weights = grid_.derivative_quadrature(
            regularization_order
        )
        self.grid = grid_
        self.regularization_order = regularization_order
        self.per_input = bool(per_input or isinstance(grid_, BSplineGridBank))
        self.maximum_log_weight = static_values[0]
        self.weight_magnitude_weight = static_values[1]
        self.weight_variation_weight = static_values[2]
        self.minimum_denominator = static_values[3]
        self.denominator_weight = static_values[4]
        self._quadrature = _BSplineQuadrature(
            quadrature_points,
            quadrature_weights,
        )

    @property
    def degree(self) -> int:
        return self.grid.degree

    @property
    def coefficient_count(self) -> int:
        return self.grid.coefficient_count

    def for_layer(self, in_size: int, out_size: int, /) -> AbstractEdgeBasis:
        del out_size
        if isinstance(self.grid, BSplineGridBank):
            if self.grid.num_grids != in_size:
                raise ValueError(
                    "Rational spline grid-bank size must match the KAN input size."
                )
            return self
        if not self.per_input:
            return self
        return RationalBSplineEdgeBasis(
            grid=BSplineGridBank.repeat(self.grid, in_size),
            regularization_order=self.regularization_order,
            per_input=True,
            maximum_log_weight=self.maximum_log_weight,
            weight_magnitude_weight=self.weight_magnitude_weight,
            weight_variation_weight=self.weight_variation_weight,
            minimum_denominator=self.minimum_denominator,
            denominator_weight=self.denominator_weight,
        )

    def initialize_coefficients(
        self,
        out_size: int,
        in_size: int,
        initialization: EdgeInitialization,
        key: Key[Array, ""],
    ) -> RationalBSplineEdgeParameters:
        if isinstance(self.grid, BSplineGridBank):
            if self.grid.num_grids != in_size:
                raise ValueError(
                    "Rational spline grid-bank size must match the KAN input size."
                )
            greville = self.grid.greville_abscissae[None, :, :]
        else:
            greville = self.grid.greville_abscissae
        if initialization == "identity":
            slopes = jnp.eye(out_size, in_size)
        elif initialization == "default":
            slopes = 0.05 * jr.normal(key, (out_size, in_size))
        else:
            raise ValueError(f"Unknown edge initialization: {initialization!r}.")
        control_values = slopes[..., None] * greville
        return RationalBSplineEdgeParameters(
            control_values,
            jnp.zeros(control_values.shape, dtype=control_values.real.dtype),
        )

    def _validated_parameters(
        self,
        parameters: Any,
        inputs: ArrayLike | None = None,
    ) -> tuple[Array, Array, Array | None]:
        if not isinstance(parameters, RationalBSplineEdgeParameters):
            raise TypeError(
                "Rational B-spline edges require RationalBSplineEdgeParameters."
            )
        control_values = jnp.asarray(parameters.control_values)
        raw_log_weights = jnp.asarray(parameters.raw_log_weights)
        if (
            control_values.ndim != 3
            or int(control_values.shape[-1]) != self.coefficient_count
            or raw_log_weights.shape != control_values.shape
        ):
            raise ValueError(
                "Rational B-spline parameter arrays must have shape "
                "(out_size, in_size, coefficient_count)."
            )
        if jnp.issubdtype(raw_log_weights.dtype, jnp.complexfloating):
            raise TypeError("Rational B-spline raw log-weights must be real-valued.")
        inputs_ = None if inputs is None else jnp.asarray(inputs)
        if inputs_ is not None and inputs_.shape != control_values.shape[:2]:
            raise ValueError(
                "Rational edge inputs must match the output-by-input parameter axes."
            )
        if isinstance(self.grid, BSplineGridBank) and (
            self.grid.num_grids != int(control_values.shape[1])
        ):
            raise ValueError(
                "Rational spline grid-bank size must match the edge input axis."
            )
        return control_values, raw_log_weights, inputs_

    def _positive_weights(self, raw_log_weights: Array) -> tuple[Array, Array]:
        log_weights = self.maximum_log_weight * jnp.tanh(raw_log_weights)
        log_weights = log_weights - jnp.mean(log_weights, axis=-1, keepdims=True)
        return jnp.exp(log_weights), log_weights

    def _evaluate_coefficients(
        self,
        coefficients: Array,
        query: Array,
        derivative_order: int = 0,
    ) -> Array:
        if isinstance(self.grid, BSplineGridBank):
            return bspline_batched_evaluate(
                self.grid.knots,
                coefficients,
                query,
                degree=self.degree,
                derivative_order=derivative_order,
                bounds="clip",
            ).values
        case_shape = tuple(int(size) for size in coefficients.shape[:2])
        return bspline_evaluate(
            self.grid.knots,
            coefficients,
            query,
            degree=self.degree,
            derivative_order=derivative_order,
            bounds="clip",
            case_shape=case_shape,
        ).values

    def evaluate(self, coefficients: Any, inputs: ArrayLike) -> Array:
        control_values, raw_log_weights, inputs_ = self._validated_parameters(
            coefficients, inputs
        )
        if inputs_ is None:
            raise RuntimeError("Rational edge inputs are missing.")
        weights, _ = self._positive_weights(raw_log_weights)
        numerator = self._evaluate_coefficients(
            control_values * weights,
            inputs_,
        )
        denominator = self._evaluate_coefficients(weights, inputs_)
        return numerator / denominator

    def regularization(self, coefficients: Any) -> Array:
        control_values, raw_log_weights, _ = self._validated_parameters(coefficients)
        weights, log_weights = self._positive_weights(raw_log_weights)
        quadrature_points = self._quadrature.points.astype(control_values.real.dtype)
        quadrature_weights = self._quadrature.weights.astype(control_values.real.dtype)
        if isinstance(self.grid, BSplineGridBank):
            query = jnp.broadcast_to(
                quadrature_points[None, :, :],
                (
                    int(control_values.shape[0]),
                    int(control_values.shape[1]),
                    int(quadrature_points.shape[1]),
                ),
            )
            integration_weights = quadrature_weights[None, :, :]
        else:
            query = jnp.broadcast_to(
                quadrature_points,
                (
                    int(control_values.shape[0]),
                    int(control_values.shape[1]),
                    int(quadrature_points.shape[0]),
                ),
            )
            integration_weights = quadrature_weights
        numerator_coefficients = control_values * weights
        numerator_jets = tuple(
            self._evaluate_coefficients(numerator_coefficients, query, order)
            for order in range(self.regularization_order + 1)
        )
        denominator_jets = tuple(
            self._evaluate_coefficients(weights, query, order)
            for order in range(self.regularization_order + 1)
        )
        denominator = denominator_jets[0]
        output_jets = [numerator_jets[0] / denominator]
        for order in range(1, self.regularization_order + 1):
            correction = sum(
                comb(order, index) * denominator_jets[index] * output_jets[order - index]
                for index in range(1, order + 1)
            )
            output_jets.append((numerator_jets[order] - correction) / denominator)
        derivative = output_jets[-1]
        energy = jnp.sum(
            jnp.real(derivative * jnp.conj(derivative)) * integration_weights
        )
        magnitude_penalty = self.weight_magnitude_weight * jnp.sum(log_weights**2)
        variation_penalty = self.weight_variation_weight * jnp.sum(
            jnp.diff(log_weights, axis=-1) ** 2
        )
        denominator_penalty = self.denominator_weight * jnp.sum(
            jax.nn.relu(self.minimum_denominator - denominator) ** 2 * integration_weights
        )
        return energy + magnitude_penalty + variation_penalty + denominator_penalty


__all__ = [
    "AbstractEdgeBasis",
    "BSplineEdgeBasis",
    "TrainableBSplineGrid",
    "BSplineGrid",
    "BSplineGridBank",
    "OrthogonalPolynomialEdgeBasis",
    "RationalBSplineEdgeBasis",
    "RationalBSplineEdgeParameters",
]
