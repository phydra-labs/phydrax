#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._holomorphic import ComplexAffineNormalization
from ..._holomorphic_linear import (
    HolomorphicLinearFrameCertificate,
    HolomorphicMultiIndexSet,
    MultiIndex,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _multi_factorial(value: MultiIndex, /) -> int:
    result = 1
    for item in value:
        result *= math.factorial(item)
    return result


def _monomial_derivative(
    normalized: Array,
    matrix: Array,
    exponent: MultiIndex,
    derivative: MultiIndex,
    /,
) -> Array:
    dimension = len(exponent)
    zero = (0,) * dimension
    coefficients: dict[MultiIndex, Array] = {
        zero: jnp.asarray(1.0, dtype=normalized.dtype)
    }
    for normalized_axis, power in enumerate(exponent):
        for _ in range(power):
            updated: dict[MultiIndex, Array] = {}
            for current, coefficient in coefficients.items():
                updated[current] = (
                    updated.get(
                        current,
                        jnp.asarray(0.0, dtype=normalized.dtype),
                    )
                    + coefficient * normalized[normalized_axis]
                )
                for physical_axis in range(dimension):
                    candidate = tuple(
                        item + 1 if axis == physical_axis else item
                        for axis, item in enumerate(current)
                    )
                    if any(
                        candidate[axis] > derivative[axis] for axis in range(dimension)
                    ):
                        continue
                    updated[candidate] = (
                        updated.get(
                            candidate,
                            jnp.asarray(0.0, dtype=normalized.dtype),
                        )
                        + coefficient * matrix[normalized_axis, physical_axis]
                    )
            coefficients = updated
    return coefficients.get(
        derivative,
        jnp.asarray(0.0, dtype=normalized.dtype),
    ) * _multi_factorial(derivative)


class HolomorphicPolynomialFrame(StrictModule, NonTrainableState):
    """Finite multivariate polynomial frame with real Cartesian coefficients."""

    index_set: HolomorphicMultiIndexSet
    normalization: ComplexAffineNormalization
    complex_output_size: int = eqx.field(static=True)
    _certificate: HolomorphicLinearFrameCertificate

    def __init__(
        self,
        index_set: HolomorphicMultiIndexSet,
        complex_output_size: int = 1,
        /,
        *,
        normalization: ComplexAffineNormalization | None = None,
    ):
        if not isinstance(index_set, HolomorphicMultiIndexSet):
            raise TypeError("index_set must be HolomorphicMultiIndexSet.")
        output_size = int(complex_output_size)
        if output_size <= 0:
            raise ValueError("Holomorphic polynomial frame output size must be positive.")
        normalization_ = (
            ComplexAffineNormalization.identity(index_set.complex_dimension)
            if normalization is None
            else normalization
        )
        if not isinstance(normalization_, ComplexAffineNormalization):
            raise TypeError("normalization must be ComplexAffineNormalization or None.")
        if normalization_.dimension != index_set.complex_dimension:
            raise ValueError("Polynomial frame normalization dimension is incompatible.")
        self.index_set = index_set
        self.normalization = normalization_
        self.complex_output_size = output_size
        self._certificate = HolomorphicLinearFrameCertificate(
            complex_input_size=index_set.complex_dimension,
            complex_output_size=output_size,
            real_coefficient_count=2 * output_size * index_set.count,
            maximum_derivative_order=index_set.maximum_total_order,
            normalization_id=normalization_.normalization_id,
            basis_construction="multivariate-complex-monomial-frame",
            construction_dependencies=(index_set.index_set_id,),
        )

    @classmethod
    def one_variable(
        cls,
        maximum_degree: int,
        complex_output_size: int = 1,
        /,
        *,
        normalization: ComplexAffineNormalization | None = None,
    ) -> HolomorphicPolynomialFrame:
        return cls(
            HolomorphicMultiIndexSet.total_degree(1, maximum_degree),
            complex_output_size,
            normalization=normalization,
        )

    @property
    def monomial_count(self) -> int:
        return self.index_set.count

    @property
    def real_coefficient_count(self) -> int:
        return self._certificate.real_coefficient_count

    def linear_frame_certificate(self) -> HolomorphicLinearFrameCertificate:
        return self._certificate

    def _coordinate_vector(self, coordinates: ArrayLike, /) -> Array:
        values = jnp.asarray(coordinates)
        dimension = self.index_set.complex_dimension
        if dimension == 1 and values.shape == ():
            values = values.reshape((1,))
        if values.shape != (dimension,):
            raise ValueError(
                f"Holomorphic polynomial frame expected shape ({dimension},); "
                f"got {values.shape}."
            )
        return values

    def basis_derivative(
        self,
        coordinates: ArrayLike,
        multi_index: Sequence[int],
        /,
    ) -> Array:
        derivative = tuple(int(item) for item in multi_index)
        dimension = self.index_set.complex_dimension
        if len(derivative) != dimension or any(item < 0 for item in derivative):
            raise ValueError("Polynomial frame derivative multi-index is invalid.")
        if sum(derivative) > self._certificate.maximum_derivative_order:
            raise ValueError("Polynomial frame derivative order is unavailable.")
        values = self._coordinate_vector(coordinates)
        normalized = self.normalization(values)
        matrix = self.normalization.matrix.astype(normalized.dtype)
        monomial_derivatives = jnp.stack(
            tuple(
                _monomial_derivative(
                    normalized,
                    matrix,
                    exponent,
                    derivative,
                )
                for exponent in self.index_set.indices
            )
        )
        output_size = self.complex_output_size
        monomials = self.monomial_count
        result = jnp.zeros(
            (output_size, self.real_coefficient_count),
            dtype=monomial_derivatives.dtype,
        )
        for output in range(output_size):
            start = 2 * output * monomials
            result = result.at[output, start : start + monomials].set(
                monomial_derivatives
            )
            result = result.at[
                output,
                start + monomials : start + 2 * monomials,
            ].set(1j * monomial_derivatives)
        return result

    def evaluate(
        self,
        coordinates: ArrayLike,
        coefficients: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape != (self.real_coefficient_count,):
            raise ValueError("Polynomial frame coefficients have invalid shape.")
        if jnp.iscomplexobj(values):
            raise TypeError("Polynomial frame coefficients must be real Cartesian.")
        basis = self.basis_derivative(
            coordinates,
            (0,) * self.index_set.complex_dimension,
        )
        return basis @ values


__all__ = ["HolomorphicPolynomialFrame"]
