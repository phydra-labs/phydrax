#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._holomorphic import (
    ComplexAffineNormalization,
    HolomorphicJet,
    HolomorphicMapCertificate,
)
from ..._strict import StrictModule


def _horner(coefficients: Array, coordinate: Array, /) -> Array:
    value = coefficients[..., -1]
    for index in range(int(coefficients.shape[-1]) - 2, -1, -1):
        value = value * coordinate + coefficients[..., index]
    return value


def _derivative_coefficients(coefficients: Array, order: int, /) -> Array:
    order_ = int(order)
    degree = int(coefficients.shape[-1]) - 1
    if order_ > degree:
        return jnp.zeros(coefficients.shape[:-1] + (1,), dtype=coefficients.dtype)
    factors = jnp.asarray(
        [
            math.factorial(power) / math.factorial(power - order_)
            for power in range(order_, degree + 1)
        ],
        dtype=coefficients.real.dtype,
    )
    return coefficients[..., order_:] * factors


class HolomorphicPolynomialPotential(StrictModule):
    """Independent complex polynomial potential branches with analytic jets."""

    coefficient_real: Array
    coefficient_imag: Array
    normalization: ComplexAffineNormalization
    branches: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    _certificate: HolomorphicMapCertificate

    def __init__(
        self,
        branches: int,
        maximum_degree: int,
        /,
        *,
        normalization: ComplexAffineNormalization | None = None,
        initial_scale: float = 0.0,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        branches_ = int(branches)
        degree = int(maximum_degree)
        scale = float(initial_scale)
        if branches_ <= 0 or degree < 0:
            raise ValueError(
                "Polynomial branches must be positive and degree nonnegative."
            )
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("initial_scale must be finite and nonnegative.")
        normalization_ = (
            ComplexAffineNormalization.identity(1)
            if normalization is None
            else normalization
        )
        if not isinstance(normalization_, ComplexAffineNormalization):
            raise TypeError("normalization must be ComplexAffineNormalization or None.")
        if normalization_.dimension != 1:
            raise ValueError("HolomorphicPolynomialPotential requires one complex input.")
        shape = (branches_, degree + 1)
        if scale == 0.0:
            real = jnp.zeros(shape, dtype=float)
            imaginary = jnp.zeros(shape, dtype=float)
        else:
            real_key, imaginary_key = jr.split(key)
            component_scale = scale / math.sqrt(float(degree + 1))
            real = component_scale * jr.normal(real_key, shape)
            imaginary = component_scale * jr.normal(imaginary_key, shape)
        self.coefficient_real = real
        self.coefficient_imag = imaginary
        self.normalization = normalization_
        self.branches = branches_
        self.maximum_degree = degree
        self._certificate = HolomorphicMapCertificate(
            complex_input_size=1,
            complex_output_size=branches_,
            construction="complex-polynomial-horner",
            normalization_id=normalization_.normalization_id,
            maximum_derivative_order=max(degree, 4),
            operations=("complex-affine", "complex-polynomial"),
            parameter_coverage="finite-subspace",
            linear_in_parameters=True,
        )

    @property
    def coefficients(self) -> Array:
        return self.coefficient_real + 1j * self.coefficient_imag

    def _normalized_scalar(self, coordinate: Array, /) -> Array:
        value = jnp.asarray(coordinate)
        if value.shape == ():
            vector = value.reshape((1,))
        elif value.shape == (1,):
            vector = value
        else:
            raise ValueError("HolomorphicPolynomialPotential expects one complex scalar.")
        return self.normalization(vector)[0]

    def __call__(self, coordinate: Array, /) -> Array:
        return _horner(self.coefficients, self._normalized_scalar(coordinate))

    def jet(self, coordinate: Array, order: int, /) -> HolomorphicJet:
        order_ = int(order)
        if order_ < 0:
            raise ValueError("Holomorphic jet order must be nonnegative.")
        normalized = self._normalized_scalar(coordinate)
        scale = self.normalization.matrix[0, 0]
        value = _horner(self.coefficients, normalized)
        derivatives = tuple(
            _horner(_derivative_coefficients(self.coefficients, current), normalized)
            * scale**current
            for current in range(1, order_ + 1)
        )
        return HolomorphicJet(value, derivatives)

    def holomorphic_certificate(self) -> HolomorphicMapCertificate:
        return self._certificate


__all__ = ["HolomorphicPolynomialPotential"]
