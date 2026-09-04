#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._pairings import AbstractPairing, DiagonalPairing, EuclideanPairing


def _identifier(value: str | None, payload: dict[str, Any], /) -> str:
    if value is None:
        return canonical_fingerprint(payload)
    identifier = str(value)
    if not identifier:
        raise ValueError("space_id must be non-empty.")
    return identifier


def _shape(value: Sequence[int], name: str, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size < 0 for size in shape):
        raise ValueError(f"{name} dimensions must be non-negative.")
    return shape


def _dtype(value: Any, /) -> np.dtype:
    dtype = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(value)))
    if not jnp.issubdtype(dtype, jnp.inexact):
        raise TypeError("Vector spaces require real or complex inexact dtypes.")
    return dtype


def _space_metadata(structure: PyTree[jax.ShapeDtypeStruct], /) -> list[dict[str, Any]]:
    return [
        {
            "path": jax.tree_util.keystr(path) or "<root>",
            "shape": list(spec.shape),
            "dtype": np.dtype(spec.dtype).str,
        }
        for path, spec in jax.tree_util.tree_flatten_with_path(structure)[0]
    ]


def _validate_pairing(
    pairing: AbstractPairing,
    structure: PyTree[jax.ShapeDtypeStruct],
    /,
) -> None:
    riesz = jax.eval_shape(pairing.riesz, structure)
    if eqx.tree_equal(riesz, structure) is not True:
        raise TypeError("Pairing Riesz output must match the vector-space structure.")
    inverse = jax.eval_shape(pairing.inverse_riesz, riesz)
    if eqx.tree_equal(inverse, structure) is not True:
        raise TypeError(
            "Pairing inverse-Riesz output must match the vector-space structure."
        )
    inner = jax.eval_shape(pairing.inner, structure, structure)
    if not isinstance(inner, jax.ShapeDtypeStruct) or inner.shape != ():
        raise TypeError("Pairing inner products must return one scalar array.")


class AbstractVectorSpace(StrictModule):
    """Finite-dimensional vector space with explicit structure and Riesz pairing."""

    space_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def structure(self, /) -> PyTree[jax.ShapeDtypeStruct]:
        """Return the exact shape/dtype PyTree for one mathematical vector."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self, vector: PyTree[Any], /) -> PyTree[Array]:
        """Validate and return one vector in this space."""
        raise NotImplementedError

    @abc.abstractmethod
    def inner(self, left: PyTree[Any], right: PyTree[Any], /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def riesz(self, vector: PyTree[Any], /) -> PyTree[Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_riesz(self, covector: PyTree[Any], /) -> PyTree[Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def flatten(self, vector: PyTree[Any], /) -> Array:
        """Flatten exactly one validated vector into canonical coordinates."""
        raise NotImplementedError

    @abc.abstractmethod
    def unflatten(self, coordinates: Array, /) -> PyTree[Array]:
        """Reconstruct exactly one vector from canonical coordinates."""
        raise NotImplementedError

    @property
    def size(self) -> int:
        return sum(prod(spec.shape) for spec in jax.tree.leaves(self.structure()))

    def zeros(self, /) -> PyTree[Array]:
        return jax.tree.map(
            lambda spec: jnp.zeros(spec.shape, dtype=spec.dtype),
            self.structure(),
        )

    def compatible(self, other: object, /) -> bool:
        return isinstance(other, AbstractVectorSpace) and self.space_id == other.space_id


class ArraySpace(AbstractVectorSpace):
    """One array-valued vector space."""

    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    pairing: AbstractPairing

    def __init__(
        self,
        shape: Sequence[int],
        /,
        *,
        dtype: Any = np.float64,
        pairing: AbstractPairing | None = None,
        space_id: str | None = None,
    ):
        shape_ = _shape(shape, "shape")
        dtype_ = _dtype(dtype)
        pairing_ = EuclideanPairing() if pairing is None else pairing
        if not isinstance(pairing_, AbstractPairing):
            raise TypeError("pairing must be an AbstractPairing.")
        _validate_pairing(pairing_, jax.ShapeDtypeStruct(shape_, dtype_))
        self.shape = shape_
        self.dtype = dtype_
        self.pairing = pairing_
        self.space_id = _identifier(
            space_id,
            {
                "kind": "array-space",
                "shape": list(shape_),
                "dtype": dtype_.str,
                "pairing": pairing_.pairing_id,
            },
        )

    def structure(self, /) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.shape, self.dtype)

    def validate(self, vector: Any, /) -> Array:
        value = jnp.asarray(vector)
        if value.shape != self.shape:
            raise ValueError(f"Vector must have shape {self.shape}; got {value.shape}.")
        if np.dtype(value.dtype) != self.dtype:
            raise TypeError(f"Vector must have dtype {self.dtype}; got {value.dtype}.")
        return value

    def inner(self, left: Any, right: Any, /) -> Array:
        return self.pairing.inner(self.validate(left), self.validate(right))

    def riesz(self, vector: Any, /) -> Array:
        return self.pairing.riesz(self.validate(vector))

    def inverse_riesz(self, covector: Any, /) -> Array:
        return self.pairing.inverse_riesz(self.validate(covector))

    def flatten(self, vector: Any, /) -> Array:
        return self.validate(vector).reshape((-1,))

    def unflatten(self, coordinates: Array, /) -> Array:
        value = jnp.asarray(coordinates)
        if value.shape != (self.size,):
            raise ValueError(
                f"Coordinates must have shape {(self.size,)}; got {value.shape}."
            )
        if np.dtype(value.dtype) != self.dtype:
            raise TypeError(
                f"Coordinates must have dtype {self.dtype}; got {value.dtype}."
            )
        return value.reshape(self.shape)


class PyTreeSpace(AbstractVectorSpace):
    """Vector space whose vectors are a fixed PyTree of array leaves."""

    structure_leaves: tuple[jax.ShapeDtypeStruct, ...] = eqx.field(static=True)
    treedef: jax.tree_util.PyTreeDef = eqx.field(static=True)
    pairing: AbstractPairing

    def __init__(
        self,
        structure: PyTree[Any],
        /,
        *,
        pairing: AbstractPairing | None = None,
        space_id: str | None = None,
        _allow_mixed_dtypes: bool = False,
    ):
        leaves, treedef = jax.tree.flatten(structure)
        if not leaves:
            raise ValueError("A PyTreeSpace requires at least one array leaf.")
        specs: list[jax.ShapeDtypeStruct] = []
        for leaf in leaves:
            if isinstance(leaf, jax.ShapeDtypeStruct):
                shape = _shape(leaf.shape, "leaf shape")
                dtype = _dtype(leaf.dtype)
            else:
                array = jnp.asarray(leaf)
                shape = tuple(int(size) for size in array.shape)
                dtype = _dtype(array.dtype)
            specs.append(jax.ShapeDtypeStruct(shape, dtype))
        if len({np.dtype(spec.dtype) for spec in specs}) != 1 and not _allow_mixed_dtypes:
            raise TypeError(
                "PyTreeSpace leaves must share one canonical coordinate dtype."
            )
        pairing_ = EuclideanPairing() if pairing is None else pairing
        if not isinstance(pairing_, AbstractPairing):
            raise TypeError("pairing must be an AbstractPairing.")
        normalized = jax.tree.unflatten(treedef, specs)
        _validate_pairing(pairing_, normalized)
        self.structure_leaves = tuple(specs)
        self.treedef = treedef
        self.pairing = pairing_
        self.space_id = _identifier(
            space_id,
            {
                "kind": "pytree-space",
                "structure": _space_metadata(normalized),
                "treedef": str(treedef),
                "pairing": pairing_.pairing_id,
            },
        )

    def structure(self, /) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.tree.unflatten(self.treedef, self.structure_leaves)

    def validate(self, vector: PyTree[Any], /) -> PyTree[Array]:
        leaves, treedef = jax.tree.flatten(vector)
        if treedef != self.treedef:
            raise ValueError("Vector PyTree structure does not match the vector space.")
        arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
        for array, spec in zip(arrays, self.structure_leaves, strict=True):
            if array.shape != spec.shape:
                raise ValueError(
                    f"Vector leaf must have shape {spec.shape}; got {array.shape}."
                )
            if np.dtype(array.dtype) != np.dtype(spec.dtype):
                raise TypeError(
                    f"Vector leaf must have dtype {spec.dtype}; got {array.dtype}."
                )
        return jax.tree.unflatten(self.treedef, arrays)

    def inner(self, left: PyTree[Any], right: PyTree[Any], /) -> Array:
        return self.pairing.inner(self.validate(left), self.validate(right))

    def riesz(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.pairing.riesz(self.validate(vector))

    def inverse_riesz(self, covector: PyTree[Any], /) -> PyTree[Array]:
        return self.pairing.inverse_riesz(self.validate(covector))

    def flatten(self, vector: PyTree[Any], /) -> Array:
        leaves = jax.tree.leaves(self.validate(vector))
        flattened = tuple(leaf.reshape((-1,)) for leaf in leaves)
        return flattened[0] if len(flattened) == 1 else jnp.concatenate(flattened)

    def unflatten(self, coordinates: Array, /) -> PyTree[Array]:
        value = jnp.asarray(coordinates)
        if value.shape != (self.size,):
            raise ValueError(
                f"Coordinates must have shape {(self.size,)}; got {value.shape}."
            )
        leaves: list[Array] = []
        offset = 0
        for spec in self.structure_leaves:
            count = prod(spec.shape)
            leaf = value[offset : offset + count].astype(spec.dtype).reshape(spec.shape)
            leaves.append(leaf)
            offset += count
        return jax.tree.unflatten(self.treedef, leaves)


class BlockSpace(AbstractVectorSpace):
    """Ordered product of independently paired vector spaces."""

    spaces: tuple[AbstractVectorSpace, ...]
    names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        spaces: Sequence[AbstractVectorSpace],
        /,
        *,
        names: Sequence[str] | None = None,
        space_id: str | None = None,
    ):
        spaces_ = tuple(spaces)
        if not spaces_ or not all(
            isinstance(space, AbstractVectorSpace) for space in spaces_
        ):
            raise TypeError("spaces must contain one or more AbstractVectorSpace values.")
        names_ = (
            tuple(str(index) for index in range(len(spaces_)))
            if names is None
            else tuple(str(name) for name in names)
        )
        if len(names_) != len(spaces_) or any(not name for name in names_):
            raise ValueError("names must contain one non-empty name per block.")
        if len(set(names_)) != len(names_):
            raise ValueError("Block names must be unique.")
        coordinate_dtypes = {
            np.dtype(spec.dtype)
            for space in spaces_
            for spec in jax.tree.leaves(space.structure())
        }
        if len(coordinate_dtypes) != 1:
            raise TypeError(
                "BlockSpace members must share one canonical coordinate dtype."
            )
        self.spaces = spaces_
        self.names = names_
        self.space_id = _identifier(
            space_id,
            {
                "kind": "block-space",
                "blocks": [
                    {"name": name, "space": space.space_id}
                    for name, space in zip(names_, spaces_, strict=True)
                ],
            },
        )

    def structure(self, /) -> tuple[PyTree[jax.ShapeDtypeStruct], ...]:
        return tuple(space.structure() for space in self.spaces)

    def validate(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        if not isinstance(vector, tuple) or len(vector) != len(self.spaces):
            raise ValueError("Block vectors must be tuples with one value per block.")
        return tuple(
            space.validate(value)
            for space, value in zip(self.spaces, vector, strict=True)
        )

    def inner(self, left: PyTree[Any], right: PyTree[Any], /) -> Array:
        left_ = self.validate(left)
        right_ = self.validate(right)
        values = [
            space.inner(left_value, right_value)
            for space, left_value, right_value in zip(
                self.spaces, left_, right_, strict=True
            )
        ]
        total = values[0]
        for value in values[1:]:
            total = total + value
        return total

    def riesz(self, vector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        values = self.validate(vector)
        return tuple(
            space.riesz(value) for space, value in zip(self.spaces, values, strict=True)
        )

    def inverse_riesz(self, covector: PyTree[Any], /) -> tuple[PyTree[Array], ...]:
        values = self.validate(covector)
        return tuple(
            space.inverse_riesz(value)
            for space, value in zip(self.spaces, values, strict=True)
        )

    def flatten(self, vector: PyTree[Any], /) -> Array:
        values = self.validate(vector)
        flattened = tuple(
            space.flatten(value) for space, value in zip(self.spaces, values, strict=True)
        )
        return flattened[0] if len(flattened) == 1 else jnp.concatenate(flattened)

    def unflatten(self, coordinates: Array, /) -> tuple[PyTree[Array], ...]:
        value = jnp.asarray(coordinates)
        if value.shape != (self.size,):
            raise ValueError(
                f"Coordinates must have shape {(self.size,)}; got {value.shape}."
            )
        blocks: list[PyTree[Array]] = []
        offset = 0
        for space in self.spaces:
            blocks.append(space.unflatten(value[offset : offset + space.size]))
            offset += space.size
        return tuple(blocks)


class DualSpace(AbstractVectorSpace):
    """Coordinate dual of one declared primal vector space."""

    primal: AbstractVectorSpace

    def __init__(self, primal: AbstractVectorSpace, /, *, space_id: str | None = None):
        if not isinstance(primal, AbstractVectorSpace):
            raise TypeError("primal must be an AbstractVectorSpace.")
        self.primal = primal
        self.space_id = _identifier(
            space_id,
            {"kind": "dual-space", "primal": primal.space_id},
        )

    def structure(self, /) -> PyTree[jax.ShapeDtypeStruct]:
        return self.primal.structure()

    def validate(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.primal.validate(vector)

    def pair(self, covector: PyTree[Any], vector: PyTree[Any], /) -> Array:
        """Evaluate one covector on one vector using the algebraic duality pairing."""
        covector_ = self.validate(covector)
        vector_ = self.primal.validate(vector)
        products = [
            jnp.sum(covector_leaf * vector_leaf)
            for covector_leaf, vector_leaf in zip(
                jax.tree.leaves(covector_),
                jax.tree.leaves(vector_),
                strict=True,
            )
        ]
        total = products[0]
        for product in products[1:]:
            total = total + product
        return total

    def inner(self, left: PyTree[Any], right: PyTree[Any], /) -> Array:
        return self.primal.inner(
            self.primal.inverse_riesz(left),
            self.primal.inverse_riesz(right),
        )

    def riesz(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.primal.inverse_riesz(vector)

    def inverse_riesz(self, covector: PyTree[Any], /) -> PyTree[Array]:
        return self.primal.riesz(covector)

    def flatten(self, vector: PyTree[Any], /) -> Array:
        return self.primal.flatten(vector)

    def unflatten(self, coordinates: Array, /) -> PyTree[Array]:
        return self.primal.unflatten(coordinates)


def _coordinate_dtype(space: AbstractVectorSpace, /) -> np.dtype:
    dtypes = {np.dtype(spec.dtype) for spec in jax.tree.leaves(space.structure())}
    if len(dtypes) != 1:
        raise TypeError("Vector space does not have one canonical coordinate dtype.")
    return next(iter(dtypes))


def _coordinate_pairing_weights(space: AbstractVectorSpace, /) -> Array:
    """Return coordinate-diagonal Riesz weights for a supported vector space."""
    if not _has_diagonal_pairing(space):
        raise TypeError("Vector space does not have a coordinate-diagonal pairing.")
    ones = jax.tree.map(
        lambda spec: jnp.ones(spec.shape, dtype=spec.dtype),
        space.structure(),
    )
    return space.flatten(space.riesz(ones))


def _coordinate_pairing_matrix(space: AbstractVectorSpace, /) -> Array:
    """Materialize the coordinate Gram matrix of one finite-dimensional pairing."""
    basis = jnp.eye(space.size, dtype=_coordinate_dtype(space))

    def row(left_coordinates):
        left = space.unflatten(left_coordinates)
        return jax.vmap(
            lambda right_coordinates: space.inner(
                left,
                space.unflatten(right_coordinates),
            )
        )(basis)

    return jax.vmap(row)(basis)


def _has_diagonal_pairing(space: AbstractVectorSpace, /) -> bool:
    from ._space_extensions import CoordaxSpace, TensorProductSpace

    if isinstance(space, (CoordaxSpace, TensorProductSpace)):
        return _has_diagonal_pairing(space.delegate)
    if isinstance(space, (ArraySpace, PyTreeSpace)):
        return isinstance(space.pairing, (EuclideanPairing, DiagonalPairing))
    if isinstance(space, BlockSpace):
        return all(_has_diagonal_pairing(block) for block in space.spaces)
    if isinstance(space, DualSpace):
        return _has_diagonal_pairing(space.primal)
    return False


def _has_euclidean_pairing(space: AbstractVectorSpace, /) -> bool:
    from ._space_extensions import CoordaxSpace, TensorProductSpace

    if isinstance(space, (CoordaxSpace, TensorProductSpace)):
        return _has_euclidean_pairing(space.delegate)
    if isinstance(space, (ArraySpace, PyTreeSpace)):
        return isinstance(space.pairing, EuclideanPairing)
    if isinstance(space, BlockSpace):
        return all(_has_euclidean_pairing(block) for block in space.spaces)
    if isinstance(space, DualSpace):
        return _has_euclidean_pairing(space.primal)
    return False


class RHSLayout(StrictModule):
    """Explicit trailing axes indexing independent right-hand sides."""

    shape: tuple[int, ...] = eqx.field(static=True)
    names: tuple[str | None, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: Sequence[int],
        /,
        *,
        names: Sequence[str | None] | None = None,
    ):
        shape_ = _shape(shape, "RHS shape")
        if not shape_:
            raise ValueError("RHSLayout requires at least one trailing RHS axis.")
        if any(size == 0 for size in shape_):
            raise ValueError("RHS axes must be non-empty.")
        names_ = (
            (None,) * len(shape_)
            if names is None
            else tuple(None if name is None else str(name) for name in names)
        )
        if len(names_) != len(shape_):
            raise ValueError("RHS axis names must match the RHS shape.")
        named = tuple(name for name in names_ if name is not None)
        if any(not name for name in named) or len(set(named)) != len(named):
            raise ValueError("Named RHS axes must be non-empty and unique.")
        self.shape = shape_
        self.names = names_
        self.layout_id = canonical_fingerprint(
            {
                "kind": "rhs-layout",
                "shape": list(shape_),
                "names": list(names_),
            }
        )

    @property
    def size(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def rhs_count(self) -> int:
        """Number of canonical independent right-hand-side columns."""
        return self.size


__all__ = [
    "AbstractVectorSpace",
    "ArraySpace",
    "BlockSpace",
    "DualSpace",
    "PyTreeSpace",
    "RHSLayout",
]
