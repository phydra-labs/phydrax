#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ._pairings import AbstractPairing
from ._spaces import AbstractVectorSpace, ArraySpace


class AlgebraCoefficientPairing(AbstractPairing):
    """Real Euclidean coefficient pairing for one algebra coordinate layout."""

    def __init__(self, algebra_id: str, /):
        identifier = str(algebra_id)
        if not identifier:
            raise ValueError("algebra_id must be non-empty.")
        self.pairing_id = canonical_fingerprint(
            {"kind": "algebra-coefficient-pairing-v1", "algebra": identifier}
        )

    def inner(self, left, right, /) -> Array:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if left_.shape != right_.shape:
            raise ValueError("Algebra pairing arguments must have identical shapes.")
        if jnp.iscomplexobj(left_) or jnp.iscomplexobj(right_):
            raise TypeError("Algebra coefficient pairing requires real coordinates.")
        return jnp.vdot(left_, right_)

    def riesz(self, vector, /) -> Array:
        value = jnp.asarray(vector)
        if jnp.iscomplexobj(value):
            raise TypeError("Algebra coefficient coordinates must be real.")
        return value

    def inverse_riesz(self, covector, /) -> Array:
        return self.riesz(covector)


class AlgebraArraySpace(AbstractVectorSpace):
    """Array-valued real vector space with one explicit finite-algebra axis."""

    algebra: Any
    base_shape: tuple[int, ...] = eqx.field(static=True)
    algebra_axis: int = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    pairing: AbstractPairing

    def __init__(
        self,
        base_shape: Sequence[int],
        algebra: Any,
        /,
        *,
        algebra_axis: int = -1,
        dtype: Any = np.float64,
        pairing: AbstractPairing | None = None,
        space_id: str | None = None,
    ):
        from ..metrix.algebra import AbstractFiniteRealAlgebraSpec

        if not isinstance(algebra, AbstractFiniteRealAlgebraSpec):
            raise TypeError("algebra must implement AbstractFiniteRealAlgebraSpec.")
        base = tuple(int(size) for size in base_shape)
        if any(size <= 0 for size in base):
            raise ValueError("Algebra base shape must contain positive dimensions.")
        rank = len(base) + 1
        axis = int(algebra_axis)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError("algebra_axis lies outside the algebra value rank.")
        shape = list(base)
        shape.insert(axis, algebra.coordinate_dimension)
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        if not np.issubdtype(dtype_, np.floating):
            raise TypeError("Algebra coordinate spaces require real floating dtype.")
        pairing_ = (
            AlgebraCoefficientPairing(algebra.algebra_id) if pairing is None else pairing
        )
        delegate = ArraySpace(tuple(shape), dtype=dtype_, pairing=pairing_)
        self.algebra = algebra
        self.base_shape = base
        self.algebra_axis = axis
        self.shape = tuple(shape)
        self.dtype = dtype_
        self.pairing = pairing_
        self.space_id = (
            canonical_fingerprint(
                {
                    "kind": "algebra-array-space-v1",
                    "base_shape": list(base),
                    "shape": shape,
                    "dtype": dtype_.str,
                    "algebra": algebra.algebra_id,
                    "axis": axis,
                    "pairing": pairing_.pairing_id,
                }
            )
            if space_id is None
            else str(space_id)
        )
        if not self.space_id:
            raise ValueError("space_id must be non-empty.")
        del delegate

    def structure(self, /) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.shape, self.dtype)

    def validate(self, vector, /) -> Array:
        value = jnp.asarray(vector)
        if value.shape != self.shape:
            raise ValueError(
                f"Algebra vector must have shape {self.shape}; got {value.shape}."
            )
        if np.dtype(value.dtype) != self.dtype:
            raise TypeError(
                f"Algebra vector must have dtype {self.dtype}; got {value.dtype}."
            )
        return value

    def inner(self, left, right, /) -> Array:
        return self.pairing.inner(self.validate(left), self.validate(right))

    def riesz(self, vector, /) -> Array:
        return self.pairing.riesz(self.validate(vector))

    def inverse_riesz(self, covector, /) -> Array:
        return self.pairing.inverse_riesz(self.validate(covector))

    def flatten(self, vector, /) -> Array:
        return self.validate(vector).reshape((-1,))

    def unflatten(self, coordinates, /) -> Array:
        value = jnp.asarray(coordinates)
        if value.shape != (prod(self.shape),) or np.dtype(value.dtype) != self.dtype:
            raise ValueError("Algebra coordinates do not match the flattened space.")
        return value.reshape(self.shape)


__all__ = ["AlgebraArraySpace", "AlgebraCoefficientPairing"]
