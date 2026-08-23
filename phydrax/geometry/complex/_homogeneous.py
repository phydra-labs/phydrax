#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class HomogeneousPolynomialReport(StrictModule):
    homogeneous_residual: Array
    euler_residual: Array
    finite: Array
    valid: Array

    def __init__(
        self,
        *,
        homogeneous_residual: ArrayLike,
        euler_residual: ArrayLike,
        finite: ArrayLike,
        tolerance: float,
    ):
        self.homogeneous_residual = jnp.asarray(homogeneous_residual)
        self.euler_residual = jnp.asarray(euler_residual)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.valid = (
            self.finite
            & (self.homogeneous_residual <= tolerance)
            & (self.euler_residual <= tolerance)
        )


class HomogeneousPolynomial(StrictModule):
    function: Callable[[Array], Array]
    projective_dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    polynomial_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Array], Array],
        projective_dimension: int,
        degree: int,
        /,
        *,
        polynomial_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        dimension = int(projective_dimension)
        degree_ = int(degree)
        if dimension < 1 or degree_ < 1:
            raise ValueError("Projective dimension and degree must be positive.")
        identifier = str(polynomial_id)
        if not identifier:
            raise ValueError("polynomial_id must be non-empty.")
        self.function = function
        self.projective_dimension = dimension
        self.degree = degree_
        self.polynomial_id = identifier

    @property
    def homogeneous_dimension(self) -> int:
        return self.projective_dimension + 1

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinates)
        if value.shape[-1:] != (self.homogeneous_dimension,):
            raise ValueError("Homogeneous coordinates have the wrong trailing dimension.")
        if value.ndim == 1:
            result = jnp.asarray(self.function(value))
        else:
            flat = value.reshape((-1, self.homogeneous_dimension))
            result = jax.vmap(self.function)(flat).reshape(value.shape[:-1])
        if result.shape != value.shape[:-1]:
            raise ValueError("Homogeneous polynomial must be scalar-valued.")
        return result

    def gradient(self, coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinates)
        if value.ndim == 1:
            return jax.jacfwd(self.function, holomorphic=True)(value)
        flat = value.reshape((-1, self.homogeneous_dimension))
        result = jax.vmap(jax.jacfwd(self.function, holomorphic=True))(flat)
        return result.reshape(value.shape)

    def validate(
        self,
        coordinates: ArrayLike,
        /,
        *,
        scale: complex = 0.7 + 0.4j,
        tolerance: float = 1e-8,
    ) -> HomogeneousPolynomialReport:
        value = jnp.asarray(coordinates)
        polynomial = self(value)
        scaled = self(scale * value)
        homogeneous = jnp.max(jnp.abs(scaled - scale**self.degree * polynomial))
        gradient = self.gradient(value)
        euler = jnp.max(
            jnp.abs(jnp.sum(value * gradient, axis=-1) - self.degree * polynomial)
        )
        finite = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(jnp.isfinite(polynomial))
            & jnp.all(jnp.isfinite(gradient))
        )
        return HomogeneousPolynomialReport(
            homogeneous_residual=homogeneous,
            euler_residual=euler,
            finite=finite,
            tolerance=tolerance,
        )


def fermat_polynomial(projective_dimension: int, /) -> HomogeneousPolynomial:
    dimension = int(projective_dimension)
    degree = dimension + 1
    return HomogeneousPolynomial(
        lambda point: jnp.sum(point**degree),
        dimension,
        degree,
        polynomial_id=f"fermat:{degree}:CP{dimension}",
    )


__all__ = [
    "HomogeneousPolynomial",
    "HomogeneousPolynomialReport",
    "fermat_polynomial",
]
