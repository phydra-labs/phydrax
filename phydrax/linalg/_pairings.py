#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


def _arrays(value: PyTree[Any], name: str, /) -> PyTree[Array]:
    arrays = jax.tree.map(jnp.asarray, value)
    leaves = jax.tree.leaves(arrays)
    if not leaves:
        raise ValueError(f"{name} must contain at least one array leaf.")
    for leaf in leaves:
        if not jnp.issubdtype(leaf.dtype, jnp.inexact):
            raise TypeError(f"{name} leaves must have real or complex inexact dtypes.")
    return arrays


def _matching_arrays(
    left: PyTree[Any],
    right: PyTree[Any],
    /,
) -> tuple[PyTree[Array], PyTree[Array]]:
    left_arrays = _arrays(left, "left")
    right_arrays = _arrays(right, "right")
    if jax.tree.structure(left_arrays) != jax.tree.structure(right_arrays):
        raise ValueError("Pairing arguments must have identical PyTree structures.")
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left_arrays),
        jax.tree.leaves(right_arrays),
        strict=True,
    ):
        if left_leaf.shape != right_leaf.shape:
            raise ValueError("Pairing argument leaves must have identical shapes.")
    return left_arrays, right_arrays


def _sum_scalars(values: list[Array], /) -> Array:
    if not values:
        raise ValueError("A pairing requires at least one array leaf.")
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


class AbstractPairing(StrictModule):
    """Riesz pairing used to define Hilbert adjoints on a vector space."""

    pairing_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def inner(self, left: PyTree[Any], right: PyTree[Any], /) -> Array:
        """Return the inner product, conjugate-linear in ``left``."""
        raise NotImplementedError

    @abc.abstractmethod
    def riesz(self, vector: PyTree[Any], /) -> PyTree[Array]:
        """Map primal coordinates to dual coordinates."""
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_riesz(self, covector: PyTree[Any], /) -> PyTree[Array]:
        """Map dual coordinates to primal coordinates."""
        raise NotImplementedError


class EuclideanPairing(AbstractPairing):
    """Euclidean/Hermitian pairing on an array or PyTree of arrays."""

    def __init__(self):
        self.pairing_id = "euclidean-hermitian"

    def inner(self, left: PyTree[Any], right: PyTree[Any], /) -> Array:
        left_arrays, right_arrays = _matching_arrays(left, right)
        products = [
            jnp.vdot(left_leaf, right_leaf)
            for left_leaf, right_leaf in zip(
                jax.tree.leaves(left_arrays),
                jax.tree.leaves(right_arrays),
                strict=True,
            )
        ]
        return _sum_scalars(products)

    def riesz(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _arrays(vector, "vector")

    def inverse_riesz(self, covector: PyTree[Any], /) -> PyTree[Array]:
        return _arrays(covector, "covector")


class DiagonalPairing(AbstractPairing):
    """Positive diagonal pairing with the same PyTree structure as its vectors."""

    weights: PyTree[Array]

    def __init__(self, weights: PyTree[Any], /, *, pairing_id: str | None = None):
        values = jax.tree.map(
            lambda value: eqx.error_if(
                value,
                jnp.any(
                    ~jnp.isfinite(value) | (jnp.real(value) <= 0) | (jnp.imag(value) != 0)
                ),
                "Diagonal pairing weights must be finite, positive, and real-valued.",
            ),
            _arrays(weights, "weights"),
        )
        if pairing_id is None:
            identifier = canonical_fingerprint(
                {
                    "kind": "diagonal-pairing",
                    "weights": array_tree_fingerprint(values),
                }
            )
        else:
            identifier = str(pairing_id)
            if not identifier:
                raise ValueError("pairing_id must be non-empty.")
        self.weights = values
        self.pairing_id = identifier

    def _validated(
        self,
        left: PyTree[Any],
        right: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        left_arrays, right_arrays = _matching_arrays(left, right)
        if jax.tree.structure(left_arrays) != jax.tree.structure(self.weights):
            raise ValueError("Pairing arguments must match the weight PyTree structure.")
        for left_leaf, weight in zip(
            jax.tree.leaves(left_arrays),
            jax.tree.leaves(self.weights),
            strict=True,
        ):
            if left_leaf.shape != weight.shape:
                raise ValueError("Pairing argument leaves must match weight shapes.")
        return left_arrays, right_arrays

    def inner(self, left: PyTree[Any], right: PyTree[Any], /) -> Array:
        left_arrays, right_arrays = self._validated(left, right)
        products = [
            jnp.vdot(left_leaf, weight * right_leaf)
            for left_leaf, right_leaf, weight in zip(
                jax.tree.leaves(left_arrays),
                jax.tree.leaves(right_arrays),
                jax.tree.leaves(self.weights),
                strict=True,
            )
        ]
        return _sum_scalars(products)

    def riesz(self, vector: PyTree[Any], /) -> PyTree[Array]:
        values, _ = self._validated(vector, vector)
        return jax.tree.map(lambda weight, value: weight * value, self.weights, values)

    def inverse_riesz(self, covector: PyTree[Any], /) -> PyTree[Array]:
        values, _ = self._validated(covector, covector)
        return jax.tree.map(lambda weight, value: value / weight, self.weights, values)


__all__ = ["AbstractPairing", "DiagonalPairing", "EuclideanPairing"]
