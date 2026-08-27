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
from ..._fingerprint import canonical_fingerprint
from ..._holomorphic import (
    ComplexAffineNormalization,
    HolomorphicJet,
    HolomorphicMapCertificate,
)
from ..._holomorphic_linear import (
    HolomorphicMultiIndexSet,
    HolomorphicMultiJet,
)
from ..._holomorphic_taylor import multijet_from_normalized, taylor_exp
from .._base import _AbstractBaseModel
from .._keys import EvalKey
from ..layers._complex_linear import ComplexLinear
from ..layers._low_rank_complex_linear import LowRankComplexLinear


class HolomorphicMLP(_AbstractBaseModel):
    """Complex MLP using only complex-affine maps and the entire exponential."""

    layers: tuple[ComplexLinear | LowRankComplexLinear, ...]
    normalization: ComplexAffineNormalization
    in_size: int
    out_size: int
    linear_ranks: tuple[int | None, ...]
    architecture_id: str
    _certificate: HolomorphicMapCertificate

    def __init__(
        self,
        *,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int],
        normalization: ComplexAffineNormalization | None = None,
        use_bias: bool = True,
        linear_ranks: Sequence[int | None] | None = None,
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
        layer_count = len(sizes) - 1
        if linear_ranks is None:
            ranks = (None,) * layer_count
        else:
            raw_ranks = tuple(linear_ranks)
            if any(
                value is not None and (isinstance(value, bool) or int(value) != value)
                for value in raw_ranks
            ):
                raise TypeError("HolomorphicMLP linear ranks must be integers or None.")
            ranks = tuple(None if value is None else int(value) for value in raw_ranks)
        if len(ranks) != layer_count:
            raise ValueError(
                "HolomorphicMLP linear_ranks must provide one entry per affine layer."
            )
        keys = jr.split(key, layer_count)
        layers: list[ComplexLinear | LowRankComplexLinear] = []
        for source, target, rank, layer_key in zip(
            sizes[:-1],
            sizes[1:],
            ranks,
            keys,
            strict=True,
        ):
            if rank is None:
                layers.append(
                    ComplexLinear(
                        in_size=source,
                        out_size=target,
                        use_bias=use_bias,
                        key=layer_key,
                    )
                )
            else:
                layers.append(
                    LowRankComplexLinear(
                        in_size=source,
                        out_size=target,
                        rank=rank,
                        use_bias=use_bias,
                        key=layer_key,
                    )
                )
        architecture_id = canonical_fingerprint(
            {
                "kind": "holomorphic-mlp-architecture",
                "sizes": list(sizes),
                "linear_ranks": list(ranks),
                "use_bias": bool(use_bias),
                "normalization": normalization_.normalization_id,
                "activation": "complex-exponential",
            }
        )
        self.layers = tuple(layers)
        self.normalization = normalization_
        self.in_size = input_size
        self.out_size = output_size
        self.linear_ranks = ranks
        self.architecture_id = architecture_id
        self._certificate = HolomorphicMapCertificate(
            complex_input_size=input_size,
            complex_output_size=output_size,
            construction="complex-affine-exponential-mlp",
            normalization_id=normalization_.normalization_id,
            maximum_derivative_order=4,
            operations=(
                (
                    "complex-affine",
                    "low-rank-complex-affine",
                    "complex-exponential",
                )
                if any(rank is not None for rank in ranks)
                else ("complex-affine", "complex-exponential")
            ),
            parameter_coverage="finite-parametric-family",
            linear_in_parameters=False,
            construction_dependencies=(architecture_id,),
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

    def _linear_taylor(
        self,
        layer: ComplexLinear | LowRankComplexLinear,
        coefficients: Array,
        index_set: HolomorphicMultiIndexSet,
        /,
    ) -> Array:
        mapped = jax.vmap(layer)(coefficients)
        bias = layer.bias
        if bias is None:
            return mapped
        active = jnp.asarray(
            [sum(multi_index) > 0 for multi_index in index_set.indices],
            dtype=mapped.real.dtype,
        )
        return mapped - active[:, None] * bias

    def multi_jet(
        self,
        coordinates: Array,
        index_set: HolomorphicMultiIndexSet,
        /,
    ) -> HolomorphicMultiJet:
        if not isinstance(index_set, HolomorphicMultiIndexSet):
            raise TypeError("index_set must be HolomorphicMultiIndexSet.")
        if index_set.complex_dimension != self.in_size:
            raise ValueError("HolomorphicMLP and multijet dimensions differ.")
        if not index_set.downward_closed:
            raise ValueError("HolomorphicMLP multijets require downward-closed indices.")
        if index_set.maximum_total_order > self._certificate.maximum_derivative_order:
            raise ValueError("Requested holomorphic multijet order is unavailable.")
        values = self._input_vector(coordinates)
        zero = (0,) * self.in_size
        coefficients = []
        for multi_index in index_set.indices:
            if multi_index == zero:
                coefficients.append(values)
                continue
            if sum(multi_index) == 1:
                axis = multi_index.index(1)
                coefficients.append(self.normalization.matrix[:, axis])
                continue
            coefficients.append(jnp.zeros_like(values))
        taylor = jnp.stack(tuple(coefficients))
        for layer in self.layers[:-1]:
            taylor = taylor_exp(
                self._linear_taylor(layer, taylor, index_set),
                index_set,
            )
        taylor = self._linear_taylor(self.layers[-1], taylor, index_set)
        return multijet_from_normalized(taylor, index_set)

    def jet(self, coordinate: Array, order: int, /) -> HolomorphicJet:
        order_ = int(order)
        if self.in_size != 1:
            raise ValueError("Scalar holomorphic jets require in_size=1.")
        if order_ < 0 or order_ > self._certificate.maximum_derivative_order:
            raise ValueError("Requested holomorphic jet order is unavailable.")
        multijet = self.multi_jet(
            jnp.asarray(coordinate).reshape(()),
            HolomorphicMultiIndexSet.total_degree(1, order_),
        )
        return HolomorphicJet(
            multijet.value,
            tuple(multijet.derivative((current,)) for current in range(1, order_ + 1)),
        )

    def holomorphic_certificate(self) -> HolomorphicMapCertificate:
        return self._certificate


__all__ = ["HolomorphicMLP"]
