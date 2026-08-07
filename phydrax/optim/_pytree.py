#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._strict import StrictModule


class _PyTreeVectorizer(StrictModule):
    tree_definition: Any = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    sizes: tuple[int, ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)
    dtype_name: str = eqx.field(static=True)

    def __init__(self, example: PyTree[Any], /):
        path_leaves, tree_definition = jax.tree_util.tree_flatten_with_path(example)
        if not path_leaves:
            raise ValueError("A bounded PyTree search requires at least one array leaf.")
        paths: list[str] = []
        shapes: list[tuple[int, ...]] = []
        sizes: list[int] = []
        dtypes: list[str] = []
        numpy_dtypes: list[np.dtype] = []
        for path, leaf in path_leaves:
            if not eqx.is_inexact_array(leaf):
                raise TypeError(
                    "Every bounded PyTree search leaf must be an inexact JAX array."
                )
            array = jnp.asarray(leaf)
            if not jnp.issubdtype(array.dtype, jnp.floating):
                raise TypeError(
                    "Differential evolution supports only real floating-point leaves."
                )
            paths.append(jax.tree_util.keystr(path) or "<root>")
            shape = tuple(int(size) for size in array.shape)
            shapes.append(shape)
            sizes.append(int(array.size))
            dtype_name = str(array.dtype)
            dtypes.append(dtype_name)
            numpy_dtypes.append(np.dtype(dtype_name))
        search_dtype = np.result_type(*numpy_dtypes)
        self.tree_definition = tree_definition
        self.paths = tuple(paths)
        self.shapes = tuple(shapes)
        self.sizes = tuple(sizes)
        self.dtypes = tuple(dtypes)
        self.dtype_name = str(search_dtype)

    @property
    def dimension(self) -> int:
        return sum(self.sizes)

    def ravel(self, tree: PyTree[Any], /, *, name: str) -> Array:
        leaves, tree_definition = jax.tree_util.tree_flatten(tree)
        if tree_definition != self.tree_definition:
            raise ValueError(f"{name} has an incompatible PyTree structure.")
        vectors = []
        for path, leaf, shape in zip(self.paths, leaves, self.shapes, strict=True):
            array = jnp.asarray(leaf)
            if tuple(int(size) for size in array.shape) != shape:
                raise ValueError(
                    f"{name} leaf {path} must have shape {shape}, got {array.shape}."
                )
            if not jnp.issubdtype(array.dtype, jnp.floating):
                raise TypeError(f"{name} leaf {path} must be real floating-point.")
            vectors.append(
                jnp.asarray(array, dtype=np.dtype(self.dtype_name)).reshape((-1,))
            )
        return jnp.concatenate(tuple(vectors))

    def ravel_bound(self, tree: PyTree[Any], /, *, side: str) -> Array:
        leaves, tree_definition = jax.tree_util.tree_flatten(tree)
        if tree_definition != self.tree_definition:
            raise ValueError(f"{side} bounds have an incompatible PyTree structure.")
        vectors = []
        for path, leaf, shape in zip(self.paths, leaves, self.shapes, strict=True):
            array = np.asarray(leaf)
            if array.shape == ():
                array = np.full(shape, float(array), dtype=np.dtype(self.dtype_name))
            elif array.shape != shape:
                raise ValueError(
                    f"{side} bound for leaf {path} must be scalar or have shape "
                    f"{shape}, got {array.shape}."
                )
            if not np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.dtype(self.dtype_name))
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{side} bound for leaf {path} must be finite.")
            vectors.append(
                jnp.asarray(array, dtype=np.dtype(self.dtype_name)).reshape((-1,))
            )
        return jnp.concatenate(tuple(vectors))

    def unravel(self, vector: Array, /) -> PyTree[Array]:
        array = jnp.asarray(vector, dtype=np.dtype(self.dtype_name))
        if array.ndim != 1 or int(array.shape[0]) != self.dimension:
            raise ValueError(
                f"Search vector must have shape ({self.dimension},), got {array.shape}."
            )
        leaves = []
        start = 0
        for shape, size, dtype_name in zip(
            self.shapes,
            self.sizes,
            self.dtypes,
            strict=True,
        ):
            leaves.append(
                jnp.asarray(
                    array[start : start + size], dtype=np.dtype(dtype_name)
                ).reshape(shape)
            )
            start += size
        return self.tree_definition.unflatten(leaves)

    def unravel_population(self, population: Array, /) -> PyTree[Array]:
        vectors = jnp.asarray(population, dtype=np.dtype(self.dtype_name))
        if vectors.ndim != 2 or int(vectors.shape[1]) != self.dimension:
            raise ValueError(
                "Population vectors must have shape "
                f"(population_size, {self.dimension}), got {vectors.shape}."
            )
        return jax.vmap(self.unravel)(vectors)
