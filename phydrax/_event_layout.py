#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._fingerprint import canonical_fingerprint
from ._strict import AbstractAttribute, StrictModule
from .domain._measure import MeasureKind


def _shape(value, /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{owner} must contain positive dimensions.")
    return shape


def _real_inexact(value: ArrayLike, /, *, owner: str) -> Array:
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        raise TypeError(f"{owner} must be real-valued.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


class AbstractEventLayout(StrictModule):
    """Invertible public-event to real-coordinate layout."""

    coordinate_size: AbstractAttribute[int]
    measure_kind: AbstractAttribute[MeasureKind]
    layout_id: AbstractAttribute[str]

    @abstractmethod
    def to_real_coordinates(self, value: Any, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def from_real_coordinates(self, coordinates: ArrayLike, /) -> Any:
        raise NotImplementedError


class ArrayEventLayout(AbstractEventLayout):
    """Real array event with arbitrary positive trailing shape."""

    event_shape: tuple[int, ...] = eqx.field(static=True)
    coordinate_size: int = eqx.field(static=True)
    measure_kind: MeasureKind = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, event_shape, /, *, layout_id: str | None = None):
        shape = _shape(event_shape, owner="event_shape")
        size = prod(shape)
        resolved = layout_id or canonical_fingerprint(
            {"kind": "array-event-layout", "event_shape": list(shape)}
        )
        if not isinstance(resolved, str) or not resolved:
            raise ValueError("layout_id must be a non-empty string or None.")
        self.event_shape = shape
        self.coordinate_size = size
        self.measure_kind = "lebesgue"
        self.layout_id = resolved

    def to_real_coordinates(self, value: Any, /) -> Array:
        array = _real_inexact(value, owner="Array event")
        rank = len(self.event_shape)
        if array.ndim < rank or tuple(array.shape[-rank:]) != self.event_shape:
            raise ValueError(
                f"Array event must end in shape {self.event_shape}; got {array.shape}."
            )
        return array.reshape(array.shape[:-rank] + (self.coordinate_size,))

    def from_real_coordinates(self, coordinates: ArrayLike, /) -> Array:
        array = _real_inexact(coordinates, owner="Event coordinates")
        if array.ndim < 1 or int(array.shape[-1]) != self.coordinate_size:
            raise ValueError(
                f"Coordinates must end in size {self.coordinate_size}; got {array.shape}."
            )
        return array.reshape(array.shape[:-1] + self.event_shape)


class ComplexEventLayout(AbstractEventLayout):
    """Complex array event represented by ordered real and imaginary coordinates."""

    event_shape: tuple[int, ...] = eqx.field(static=True)
    complex_size: int = eqx.field(static=True)
    coordinate_size: int = eqx.field(static=True)
    measure_kind: MeasureKind = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, event_shape, /, *, layout_id: str | None = None):
        shape = _shape(event_shape, owner="event_shape")
        size = prod(shape)
        resolved = layout_id or canonical_fingerprint(
            {"kind": "complex-event-layout", "event_shape": list(shape)}
        )
        if not isinstance(resolved, str) or not resolved:
            raise ValueError("layout_id must be a non-empty string or None.")
        self.event_shape = shape
        self.complex_size = size
        self.coordinate_size = 2 * size
        self.measure_kind = "lebesgue"
        self.layout_id = resolved

    def to_real_coordinates(self, value: Any, /) -> Array:
        array = jnp.asarray(value)
        if not jnp.iscomplexobj(array):
            raise TypeError("Complex event layout requires complex-valued events.")
        rank = len(self.event_shape)
        if array.ndim < rank or tuple(array.shape[-rank:]) != self.event_shape:
            raise ValueError(
                f"Complex event must end in shape {self.event_shape}; got {array.shape}."
            )
        leading = array.shape[:-rank]
        flat = array.reshape(leading + (self.complex_size,))
        return jnp.concatenate((jnp.real(flat), jnp.imag(flat)), axis=-1)

    def from_real_coordinates(self, coordinates: ArrayLike, /) -> Array:
        array = _real_inexact(coordinates, owner="Complex event coordinates")
        if array.ndim < 1 or int(array.shape[-1]) != self.coordinate_size:
            raise ValueError(
                f"Coordinates must end in size {self.coordinate_size}; got {array.shape}."
            )
        real = array[..., : self.complex_size]
        imag = array[..., self.complex_size :]
        return jax.lax.complex(real, imag).reshape(array.shape[:-1] + self.event_shape)


class PyTreeEventLayout(AbstractEventLayout):
    """Real PyTree event with stable leaf paths, shapes, and coordinate slices."""

    treedef: Any = eqx.field(static=True)
    leaf_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    leaf_dtypes: tuple[Any, ...] = eqx.field(static=True)
    leaf_sizes: tuple[int, ...] = eqx.field(static=True)
    leaf_paths: tuple[str, ...] = eqx.field(static=True)
    coordinate_size: int = eqx.field(static=True)
    measure_kind: MeasureKind = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, template: Any, /, *, layout_id: str | None = None):
        path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
        if not path_leaves:
            raise ValueError("PyTree event template must contain array leaves.")
        shapes = []
        dtypes = []
        sizes = []
        paths = []
        for path, leaf in path_leaves:
            array = _real_inexact(leaf, owner="PyTree event leaf")
            shape = tuple(int(size) for size in array.shape)
            if any(size <= 0 for size in shape):
                raise ValueError("PyTree event leaves cannot contain empty axes.")
            dtypes.append(array.dtype)
            shapes.append(shape)
            sizes.append(int(array.size))
            paths.append(jax.tree_util.keystr(path))
        coordinate_size = sum(sizes)
        resolved = layout_id or canonical_fingerprint(
            {
                "kind": "pytree-event-layout",
                "leaf_paths": paths,
                "leaf_dtypes": [str(dtype) for dtype in dtypes],
                "leaf_shapes": [list(shape) for shape in shapes],
            }
        )
        if not isinstance(resolved, str) or not resolved:
            raise ValueError("layout_id must be a non-empty string or None.")
        self.treedef = treedef
        self.leaf_shapes = tuple(shapes)
        self.leaf_dtypes = tuple(dtypes)
        self.leaf_sizes = tuple(sizes)
        self.leaf_paths = tuple(paths)
        self.coordinate_size = coordinate_size
        self.measure_kind = "lebesgue"
        self.layout_id = resolved

    def to_real_coordinates(self, value: Any, /) -> Array:
        leaves, treedef = jax.tree_util.tree_flatten(value)
        if treedef != self.treedef or len(leaves) != len(self.leaf_shapes):
            raise ValueError("PyTree event structure does not match its layout.")
        flattened = []
        leading_shape = None
        for leaf, event_shape, size, path in zip(
            leaves,
            self.leaf_shapes,
            self.leaf_sizes,
            self.leaf_paths,
            strict=True,
        ):
            array = _real_inexact(leaf, owner=f"PyTree event leaf {path}")
            rank = len(event_shape)
            leading = array.shape[:-rank] if rank else array.shape
            trailing = array.shape[-rank:] if rank else ()
            if tuple(trailing) != event_shape:
                raise ValueError(
                    f"PyTree event leaf {path} must end in {event_shape}; got {array.shape}."
                )
            if leading_shape is None:
                leading_shape = leading
            elif tuple(leading) != tuple(leading_shape):
                raise ValueError("Every PyTree event leaf must share leading sample axes.")
            flattened.append(array.reshape(tuple(leading) + (size,)))
        return jnp.concatenate(flattened, axis=-1)

    def from_real_coordinates(self, coordinates: ArrayLike, /) -> Any:
        array = _real_inexact(coordinates, owner="PyTree event coordinates")
        if array.ndim < 1 or int(array.shape[-1]) != self.coordinate_size:
            raise ValueError(
                f"Coordinates must end in size {self.coordinate_size}; got {array.shape}."
            )
        leading = tuple(array.shape[:-1])
        leaves = []
        offset = 0
        for shape, dtype, size in zip(
            self.leaf_shapes, self.leaf_dtypes, self.leaf_sizes, strict=True
        ):
            leaf = array[..., offset : offset + size].reshape(leading + shape)
            leaves.append(leaf.astype(dtype))
            offset += size
        return jax.tree_util.tree_unflatten(self.treedef, leaves)


__all__ = [
    "AbstractEventLayout",
    "ArrayEventLayout",
    "ComplexEventLayout",
    "PyTreeEventLayout",
]
