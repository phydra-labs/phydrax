#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._connection import AbstractAffineConnection
from ._density import VolumeDensity
from ._utils import _pointwise_array


def apply_cotangent_map(
    covector: ArrayLike,
    contravariant_map: ArrayLike,
    /,
) -> Array:
    """Apply a trailing square covector-to-vector matrix without copying inputs."""
    covector_array = jnp.asarray(covector)
    matrix = jnp.asarray(contravariant_map)
    if covector_array.ndim < 1:
        raise ValueError("A covector must have a trailing component axis.")
    dimension = covector_array.shape[-1]
    if matrix.shape[-2:] != (dimension, dimension):
        raise ValueError(
            "A cotangent map must have trailing shape "
            f"{(dimension, dimension)}; got {matrix.shape}."
        )
    return oe.contract("...ij,...j->...i", matrix, covector_array)


class _DensityDivergenceEvaluator(StrictModule):
    field: Callable[[Array], Array]
    density: VolumeDensity

    def __init__(
        self,
        field: Callable[[Array], Array],
        density: VolumeDensity,
    ):
        self.field = field
        self.density = density

    def __call__(self, coordinates: Array, /) -> Array:
        vector = jnp.asarray(self.field(coordinates))
        dimension = self.density.chart.dimension
        if vector.shape != (dimension,):
            raise ValueError(
                f"Pointwise vector field must have shape {(dimension,)}; "
                f"got {vector.shape}."
            )
        derivative = jax.jacfwd(self.field)(coordinates)
        log_density_derivative = jax.grad(self.density.log_value)(coordinates)
        return jnp.trace(derivative) + jnp.dot(vector, log_density_derivative)


def density_divergence(
    field: Callable[[Array], Array],
    density: VolumeDensity,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return divergence relative to a declared positive volume density."""
    if not callable(field):
        raise TypeError("field must be callable.")
    if not isinstance(density, VolumeDensity):
        raise TypeError("density must be a VolumeDensity.")
    return _pointwise_array(
        _DensityDivergenceEvaluator(field, density),
        coordinates,
        density.chart.dimension,
    )


class _CovariantSymbolEvaluator(StrictModule):
    field: Callable[[Array], Array]
    symbol: Callable[[Array], Array]
    connection: AbstractAffineConnection
    drift: Callable[[Array], Array] | None

    def __init__(
        self,
        field: Callable[[Array], Array],
        symbol: Callable[[Array], Array],
        connection: AbstractAffineConnection,
        drift: Callable[[Array], Array] | None,
    ):
        self.field = field
        self.symbol = symbol
        self.connection = connection
        self.drift = drift

    def __call__(self, coordinates: Array, /) -> Array:
        value = jnp.asarray(self.field(coordinates))
        if value.shape != ():
            raise ValueError("Covariant symbol contraction requires a scalar field.")
        dimension = self.connection.chart.dimension
        differential = jax.grad(self.field)(coordinates)
        second_derivative = jax.hessian(self.field)(coordinates)
        coefficients = self.connection.coefficients(coordinates)
        covariant_hessian = second_derivative - oe.contract(
            "kij,k->ij", coefficients, differential
        )
        symbol = jnp.asarray(self.symbol(coordinates))
        if symbol.shape != (dimension, dimension):
            raise ValueError(
                f"Pointwise principal symbol must have shape {(dimension, dimension)}; "
                f"got {symbol.shape}."
            )
        result = oe.contract("ij,ij->", symbol, covariant_hessian)
        if self.drift is not None:
            drift = jnp.asarray(self.drift(coordinates))
            if drift.shape != (dimension,):
                raise ValueError(
                    f"Pointwise drift must have shape {(dimension,)}; got {drift.shape}."
                )
            result = result + jnp.dot(drift, differential)
        return result


def covariant_symbol_contraction(
    field: Callable[[Array], Array],
    symbol: Callable[[Array], Array],
    connection: AbstractAffineConnection,
    coordinates: ArrayLike,
    /,
    *,
    drift: Callable[[Array], Array] | None = None,
) -> Array:
    """Contract a principal symbol with a covariant Hessian and optional drift."""
    if not callable(field) or not callable(symbol):
        raise TypeError("field and symbol must be callable.")
    if not isinstance(connection, AbstractAffineConnection):
        raise TypeError("connection must be an AbstractAffineConnection.")
    if drift is not None and not callable(drift):
        raise TypeError("drift must be callable when supplied.")
    return _pointwise_array(
        _CovariantSymbolEvaluator(field, symbol, connection, drift),
        coordinates,
        connection.chart.dimension,
    )


__all__ = [
    "apply_cotangent_map",
    "covariant_symbol_contraction",
    "density_divergence",
]
