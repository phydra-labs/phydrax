#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._holomorphic import (
    ComplexAffineNormalization,
    HolomorphicJet,
    HolomorphicMapCertificate,
)
from .._base import _AbstractBaseModel
from .._keys import EvalKey
from ..layers._complex_linear import ComplexLinear


class HolomorphicMLP(_AbstractBaseModel):
    """Complex MLP using only complex-affine maps and the entire exponential."""

    layers: tuple[ComplexLinear, ...]
    normalization: ComplexAffineNormalization
    in_size: int
    out_size: int
    _certificate: HolomorphicMapCertificate

    def __init__(
        self,
        *,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int],
        normalization: ComplexAffineNormalization | None = None,
        use_bias: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        input_size = int(in_size)
        output_size = int(out_size)
        hidden = tuple(int(value) for value in hidden_sizes)
        if input_size <= 0 or output_size <= 0:
            raise ValueError("HolomorphicMLP input and output sizes must be positive.")
        if not hidden or any(value <= 0 for value in hidden):
            raise ValueError("HolomorphicMLP requires positive hidden sizes.")
        normalization_ = (
            ComplexAffineNormalization.identity(input_size)
            if normalization is None
            else normalization
        )
        if not isinstance(normalization_, ComplexAffineNormalization):
            raise TypeError("normalization must be ComplexAffineNormalization or None.")
        if normalization_.dimension != input_size:
            raise ValueError("HolomorphicMLP normalization dimension must match in_size.")
        sizes = (input_size, *hidden, output_size)
        keys = jr.split(key, len(sizes) - 1)
        self.layers = tuple(
            ComplexLinear(
                in_size=source,
                out_size=target,
                use_bias=use_bias,
                key=layer_key,
            )
            for source, target, layer_key in zip(
                sizes[:-1],
                sizes[1:],
                keys,
                strict=True,
            )
        )
        self.normalization = normalization_
        self.in_size = input_size
        self.out_size = output_size
        self._certificate = HolomorphicMapCertificate(
            complex_input_size=input_size,
            complex_output_size=output_size,
            construction="complex-affine-exponential-mlp",
            normalization_id=normalization_.normalization_id,
            maximum_derivative_order=4,
            operations=("complex-affine", "complex-exponential"),
            parameter_coverage="finite-parametric-family",
            linear_in_parameters=False,
        )

    def _input_vector(self, coordinates: Array, /) -> Array:
        values = jnp.asarray(coordinates)
        if self.in_size == 1 and values.shape == ():
            values = values.reshape((1,))
        if values.shape != (self.in_size,):
            raise ValueError(
                f"HolomorphicMLP expected shape ({self.in_size},); got {values.shape}."
            )
        return self.normalization(values)

    def __call__(
        self,
        coordinates: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        value = self._input_vector(coordinates)
        for layer in self.layers[:-1]:
            value = jnp.exp(layer(value))
        return self.layers[-1](value)

    def jet(self, coordinate: Array, order: int, /) -> HolomorphicJet:
        order_ = int(order)
        if self.in_size != 1:
            raise ValueError("Scalar holomorphic jets require in_size=1.")
        if order_ < 0 or order_ > self._certificate.maximum_derivative_order:
            raise ValueError("Requested holomorphic jet order is unavailable.")
        scalar = jnp.asarray(coordinate).reshape(())

        def evaluate(value):
            return self(value)

        derivatives = []
        derivative = evaluate
        for _ in range(order_):
            derivative = jax.jacfwd(derivative, holomorphic=True)
            derivatives.append(derivative(scalar))
        return HolomorphicJet(evaluate(scalar), derivatives)

    def holomorphic_certificate(self) -> HolomorphicMapCertificate:
        return self._certificate


__all__ = ["HolomorphicMLP"]
