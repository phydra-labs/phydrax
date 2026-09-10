#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

import phydrax.axes as cx

from .._fingerprint import canonical_fingerprint
from ._pairings import DiagonalPairing
from ._spaces import (
    _coordinate_dtype,
    _has_diagonal_pairing,
    AbstractVectorSpace,
    ArraySpace,
    PyTreeSpace,
)


class TensorProductSpace(AbstractVectorSpace):
    """Tensor product of finite-dimensional spaces in factor-major axis order."""

    factors: tuple[AbstractVectorSpace, ...]
    delegate: ArraySpace

    def __init__(
        self,
        factors: Sequence[AbstractVectorSpace],
        /,
        *,
        space_id: str | None = None,
    ):
        factors_ = tuple(factors)
        if not factors_ or not all(
            isinstance(factor, AbstractVectorSpace) for factor in factors_
        ):
            raise TypeError("factors must contain AbstractVectorSpace values.")
        dtypes = {_coordinate_dtype(factor) for factor in factors_}
        if len(dtypes) != 1:
            raise TypeError("Tensor-product factors must share one coordinate dtype.")
        metric = jnp.asarray([1.0], dtype=next(iter(dtypes)).type)
        for factor in factors_:
            if not (
                _has_diagonal_pairing(factor) or isinstance(factor, TensorProductSpace)
            ):
                raise TypeError(
                    "TensorProductSpace requires coordinate-diagonal factor pairings."
                )
            ones = jnp.ones((factor.size,), dtype=_coordinate_dtype(factor))
            factor_metric = jnp.real(factor.flatten(factor.riesz(factor.unflatten(ones))))
            metric = jnp.kron(metric, factor_metric)
        shape = tuple(factor.size for factor in factors_)
        delegate = ArraySpace(
            shape,
            dtype=next(iter(dtypes)),
            pairing=DiagonalPairing(metric.reshape(shape)),
        )
        self.factors = factors_
        self.delegate = delegate
        self.space_id = (
            canonical_fingerprint(
                {
                    "kind": "tensor-product-space",
                    "factors": [f.space_id for f in factors_],
                }
            )
            if space_id is None
            else _nonempty(space_id, "space_id")
        )

    def structure(self, /) -> jax.ShapeDtypeStruct:
        return self.delegate.structure()

    def validate(self, vector: Any, /) -> Array:
        return self.delegate.validate(vector)

    def inner(self, left: Any, right: Any, /) -> Array:
        return self.delegate.inner(left, right)

    def riesz(self, vector: Any, /) -> Array:
        return self.delegate.riesz(vector)

    def inverse_riesz(self, covector: Any, /) -> Array:
        return self.delegate.inverse_riesz(covector)

    def flatten(self, vector: Any, /) -> Array:
        return self.delegate.flatten(vector)

    def unflatten(self, coordinates: Array, /) -> Array:
        return self.delegate.unflatten(coordinates)


class AxisArraySpace(AbstractVectorSpace):
    """Vector space preserving one native axis array layout."""

    delegate: PyTreeSpace
    layout: cx.AxisLayout = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, template: cx.AxisArray, /, *, space_id: str | None = None):
        if not isinstance(template, cx.AxisArray):
            raise TypeError("template must be an AxisArray.")
        delegate = PyTreeSpace(template)
        self.delegate = delegate
        self.layout = template.layout
        self.shape = tuple(template.shape)
        self.space_id = (
            canonical_fingerprint(
                {
                    "kind": "axis-array-space",
                    "delegate": delegate.space_id,
                    "layout": repr(self.layout),
                    "shape": list(self.shape),
                }
            )
            if space_id is None
            else _nonempty(space_id, "space_id")
        )

    def structure(self, /) -> PyTree[jax.ShapeDtypeStruct]:
        return self.delegate.structure()

    def validate(self, vector: PyTree[Any], /) -> PyTree[Array]:
        if not isinstance(vector, cx.AxisArray):
            raise TypeError("AxisArraySpace vectors must be AxisArray values.")
        if vector.layout != self.layout or tuple(vector.shape) != self.shape:
            raise ValueError("Axis array layout or shape does not match the space.")
        return self.delegate.validate(vector)

    def inner(self, left: PyTree[Any], right: PyTree[Any], /) -> Array:
        return self.delegate.inner(self.validate(left), self.validate(right))

    def riesz(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.delegate.riesz(self.validate(vector))

    def inverse_riesz(self, covector: PyTree[Any], /) -> PyTree[Array]:
        return self.delegate.inverse_riesz(self.validate(covector))

    def flatten(self, vector: PyTree[Any], /) -> Array:
        return self.delegate.flatten(self.validate(vector))

    def unflatten(self, coordinates: Array, /) -> PyTree[Array]:
        return self.delegate.unflatten(coordinates)


def _nonempty(value: str, name: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


__all__ = ["AxisArraySpace", "TensorProductSpace"]
