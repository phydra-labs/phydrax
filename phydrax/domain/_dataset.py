#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._doc import DOC_KEY0
from ._domain import _AbstractUnaryDomain


class DatasetDomain(_AbstractUnaryDomain):
    """A unary domain over a finite in-memory dataset.

    A `DatasetDomain` stores a PyTree of arrays where every leaf has a leading dataset
    axis of the same length `N`. Sampling draws a batch of indices uniformly and
    returns the corresponding slice from each leaf.

    This is intended for product domains like `Omega_data @ Omega_x`, where `data`
    samples are paired/broadcast with spatial points.
    """

    data: PyTree[Array]
    _label: str
    _size: int
    _measure_mode: Literal["probability", "count"]

    def __init__(
        self,
        data: PyTree[ArrayLike],
        /,
        *,
        label: str = "data",
        measure: Literal["probability", "count"] = "probability",
    ):
        leaves = jax.tree_util.tree_leaves(data)
        if not leaves:
            raise ValueError("DatasetDomain requires at least one array leaf.")

        arrays = jax.tree_util.tree_map(lambda x: jnp.asarray(x), data)
        leaves_arr = jax.tree_util.tree_leaves(arrays)
        first = jnp.asarray(leaves_arr[0])
        if first.ndim == 0:
            raise ValueError("DatasetDomain leaves must have a leading dataset axis.")
        n = int(first.shape[0])
        if n <= 0:
            raise ValueError("DatasetDomain dataset axis must be non-empty.")

        for leaf in leaves_arr:
            arr = jnp.asarray(leaf)
            if arr.ndim == 0:
                raise ValueError("DatasetDomain leaves must have a leading dataset axis.")
            if int(arr.shape[0]) != n:
                raise ValueError(
                    "DatasetDomain requires all leaves to share the same leading axis; "
                    f"got {int(arr.shape[0])} and {n}."
                )

        self.data = arrays
        self._label = str(label)
        self._size = n
        self._measure_mode = measure

    @property
    def label(self) -> str:
        return self._label

    @property
    def var_dim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return int(self._size)

    @property
    def measure(self) -> Array:
        if self._measure_mode == "count":
            return jnp.asarray(float(self._size), dtype=float)
        return jnp.asarray(1.0, dtype=float)

    def sample(
        self,
        num_points: int,
        *,
        sampler: str = "uniform",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PyTree[Array]:
        del sampler
        n = int(num_points)
        if n < 0:
            raise ValueError("num_points must be non-negative.")
        if n == 0:
            return jax.tree_util.tree_map(lambda a: jnp.asarray(a)[:0], self.data)

        idx = jr.randint(key, shape=(n,), minval=0, maxval=int(self._size))
        return jax.tree_util.tree_map(lambda a: jnp.asarray(a)[idx], self.data)

    def equivalent(self, other: object, /) -> bool:
        if not isinstance(other, DatasetDomain):
            return False
        if self.label != other.label:
            return False
        if int(self._size) != int(other._size):
            return False
        if self._measure_mode != other._measure_mode:
            return False

        leaves_a, treedef_a = jax.tree_util.tree_flatten(self.data)
        leaves_b, treedef_b = jax.tree_util.tree_flatten(other.data)
        if treedef_a != treedef_b:
            return False

        for a, b in zip(leaves_a, leaves_b, strict=True):
            arr_a = jnp.asarray(a)
            arr_b = jnp.asarray(b)
            if arr_a.shape != arr_b.shape:
                return False
            if arr_a.dtype != arr_b.dtype:
                return False

        return True


__all__ = ["DatasetDomain"]
