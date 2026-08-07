#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import isfinite
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import ArrayLike, Key
from orthax import (
    chebyshev as _o_cheb,
    hermite as _o_herm,
    hermite_e as _o_herme,
    laguerre as _o_lag,
    legendre as _o_leg,
)

from ...._interpolation import apply_gather_stencil, bspline_stencil
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


EdgeInitialization = Literal["default", "identity"]


_FAMILY_ALIASES = {
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


def _poly_eval(coefficients: Array, inputs: Array, family: str) -> Array:
    if family == "chebyshev":
        return _o_cheb.chebval(inputs, coefficients)
    if family == "legendre":
        return _o_leg.legval(inputs, coefficients)
    if family == "hermite":
        return _o_herm.hermval(inputs, coefficients)
    if family == "hermite_e":
        return _o_herme.hermeval(inputs, coefficients)
    if family == "laguerre":
        return _o_lag.lagval(inputs, coefficients)
    raise ValueError(f"Unsupported orthogonal polynomial family: {family!r}.")


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
    def initialize_coefficients(
        self,
        out_size: int,
        in_size: int,
        initialization: EdgeInitialization,
        key: Key,
    ) -> Array:
        """Initialize a dense output-by-input collection of edge functions."""

    @abstractmethod
    def evaluate(self, coefficients: ArrayLike, inputs: ArrayLike) -> Array:
        """Evaluate one scalar input for every output-by-input edge."""

    @abstractmethod
    def regularization(self, coefficients: ArrayLike) -> Array:
        """Return this basis family's unscaled coefficient penalty."""


class OrthogonalPolynomialEdgeBasis(AbstractEdgeBasis):
    """Global orthogonal-polynomial KAN edge basis evaluated by Orthax."""

    _degree: int = eqx.field(static=True)
    family: str = eqx.field(static=True)
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

    def initialize_coefficients(
        self,
        out_size: int,
        in_size: int,
        initialization: EdgeInitialization,
        key: Key,
    ) -> Array:
        coefficients = jnp.zeros((out_size, in_size, self.coefficient_count))
        if initialization == "identity":
            if self.degree < 1:
                raise ValueError(
                    "Identity initialization requires polynomial degree >= 1."
                )
            diagonal = jnp.eye(out_size, in_size)
            return coefficients.at[..., 1].set(diagonal)
        if initialization == "default":
            if self.degree == 0:
                return coefficients
            slopes = 0.05 * jr.normal(key, (out_size, in_size))
            return coefficients.at[..., 1].set(slopes)
        raise ValueError(f"Unknown edge initialization: {initialization!r}.")

    def evaluate(self, coefficients: ArrayLike, inputs: ArrayLike) -> Array:
        coefficients_, inputs_ = _validate_edge_arrays(
            coefficients, inputs, self.coefficient_count
        )
        return jax.vmap(
            lambda coefficient_row, input_row: jax.vmap(
                lambda edge_coefficients, edge_input: _poly_eval(
                    edge_coefficients, edge_input, self.family
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


class BSplineGrid(StrictModule, NonTrainableState):
    """Fixed open-uniform B-spline grid over the KAN interval ``[-1, 1]``."""

    knots: Array
    degree: int = eqx.field(static=True)
    num_intervals: int = eqx.field(static=True)

    def __init__(self, degree: int = 3, num_intervals: int = 8):
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise ValueError("B-spline degree must be a positive integer.")
        if (
            isinstance(num_intervals, bool)
            or not isinstance(num_intervals, int)
            or num_intervals < 1
        ):
            raise ValueError("num_intervals must be a positive integer.")
        interior = jnp.linspace(-1.0, 1.0, num_intervals + 1)[1:-1]
        self.knots = jnp.concatenate(
            (
                jnp.full((degree + 1,), -1.0),
                interior,
                jnp.full((degree + 1,), 1.0),
            )
        )
        self.degree = degree
        self.num_intervals = num_intervals

    @property
    def coefficient_count(self) -> int:
        return int(self.knots.shape[0]) - self.degree - 1

    @property
    def greville_abscissae(self) -> Array:
        return jnp.stack(
            [
                jnp.mean(self.knots[index + 1 : index + self.degree + 1])
                for index in range(self.coefficient_count)
            ]
        )


class BSplineEdgeBasis(AbstractEdgeBasis):
    """Compactly supported B-spline KAN edge basis on a fixed shared grid."""

    grid: BSplineGrid
    regularization_order: int = eqx.field(static=True)
    _quadrature_points: tuple[float, ...] = eqx.field(static=True)
    _quadrature_weights: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        degree: int = 3,
        *,
        num_intervals: int = 8,
        regularization_order: int = 2,
    ):
        grid = BSplineGrid(degree=degree, num_intervals=num_intervals)
        if (
            isinstance(regularization_order, bool)
            or not isinstance(regularization_order, int)
            or not 1 <= regularization_order <= degree
        ):
            raise ValueError("regularization_order must lie between one and degree.")

        quadrature_order = degree - regularization_order + 1
        reference_points, reference_weights = np.polynomial.legendre.leggauss(
            quadrature_order
        )
        boundaries = np.linspace(-1.0, 1.0, num_intervals + 1)
        quadrature_points: list[float] = []
        quadrature_weights: list[float] = []
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
            midpoint = 0.5 * (lower + upper)
            half_width = 0.5 * (upper - lower)
            quadrature_points.extend((midpoint + half_width * reference_points).tolist())
            quadrature_weights.extend((half_width * reference_weights).tolist())

        self.grid = grid
        self.regularization_order = regularization_order
        self._quadrature_points = tuple(quadrature_points)
        self._quadrature_weights = tuple(quadrature_weights)

    @property
    def degree(self) -> int:
        return self.grid.degree

    @property
    def coefficient_count(self) -> int:
        return self.grid.coefficient_count

    def initialize_coefficients(
        self,
        out_size: int,
        in_size: int,
        initialization: EdgeInitialization,
        key: Key,
    ) -> Array:
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
        case_shape = tuple(int(size) for size in coefficients_.shape[:2])
        stencil = bspline_stencil(
            self.grid.knots,
            inputs_,
            degree=self.degree,
            bounds="clip",
            case_shape=case_shape,
        )
        return apply_gather_stencil(coefficients_, stencil).values

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
        quadrature_points = jnp.asarray(
            self._quadrature_points, dtype=coefficients_.real.dtype
        )
        quadrature_weights = jnp.asarray(
            self._quadrature_weights, dtype=coefficients_.real.dtype
        )
        case_shape = tuple(int(size) for size in coefficients_.shape[:2])
        query = jnp.broadcast_to(
            quadrature_points,
            (*case_shape, int(quadrature_points.shape[0])),
        )
        stencil = bspline_stencil(
            self.grid.knots,
            query,
            degree=self.degree,
            derivative_order=self.regularization_order,
            bounds="error",
            case_shape=case_shape,
        )
        derivative = apply_gather_stencil(coefficients_, stencil).values
        magnitude = jnp.real(derivative * jnp.conj(derivative))
        return jnp.sum(magnitude * quadrature_weights)


__all__ = [
    "AbstractEdgeBasis",
    "BSplineEdgeBasis",
    "BSplineGrid",
    "OrthogonalPolynomialEdgeBasis",
]
