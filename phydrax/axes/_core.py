#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class AxisKey:
    """Namespaced, stable semantic axis identity."""

    scope: str
    name: str

    def __post_init__(self):
        if not self.scope or not self.name:
            raise ValueError("AxisKey scope and name must be non-empty.")

    @property
    def identifier(self) -> str:
        return f"{self.scope}:{self.name}"


@dataclass(frozen=True, slots=True)
class Axis:
    """Finite semantic axis schema."""

    key: AxisKey
    size: int
    labels: tuple[str, ...] = ()
    support_id: str | None = None

    def __post_init__(self):
        if self.size < 0:
            raise ValueError("Axis size must be non-negative.")
        if self.labels and len(self.labels) != self.size:
            raise ValueError("Axis labels must match axis size.")

    def ref(self, *, slot: str = "value", variance: str = "neutral") -> AxisRef:
        return AxisRef(self, slot=slot, variance=variance)


@dataclass(frozen=True, slots=True)
class AxisRef:
    """One role-specific use of a semantic axis in an array layout."""

    axis: Axis
    slot: str = "value"
    variance: str = "neutral"

    def __post_init__(self):
        if not self.slot or not self.variance:
            raise ValueError("AxisRef slot and variance must be non-empty.")


@dataclass(frozen=True, slots=True)
class UnboundAxis:
    """One positional axis whose scientific identity has not been bound."""

    size: int

    def __post_init__(self):
        if self.size < 0:
            raise ValueError("Unbound axis size must be non-negative.")


AxisEntry = AxisRef | UnboundAxis


@dataclass(frozen=True, slots=True)
class AxisLayout:
    """Ordered finite axis layout for one dense JAX array."""

    axes: tuple[AxisEntry, ...]

    def __post_init__(self):
        references = [axis for axis in self.axes if isinstance(axis, AxisRef)]
        identities = [(axis.axis.key, axis.slot, axis.variance) for axis in references]
        if len(set(identities)) != len(identities):
            raise ValueError("AxisLayout cannot repeat an identical AxisRef.")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            axis.axis.size if isinstance(axis, AxisRef) else axis.size
            for axis in self.axes
        )

    @property
    def named_axes(self) -> tuple[AxisRef, ...]:
        return tuple(axis for axis in self.axes if isinstance(axis, AxisRef))

    @property
    def positional_axes(self) -> tuple[UnboundAxis, ...]:
        return tuple(axis for axis in self.axes if isinstance(axis, UnboundAxis))

    @classmethod
    def from_dims(
        cls,
        dims: tuple[str | None | AxisRef, ...],
        shape: tuple[int, ...],
        /,
    ) -> AxisLayout:
        if len(dims) != len(shape):
            raise ValueError("Axis dimensions must match array rank.")
        axes: list[AxisEntry] = []
        for dim, size in zip(dims, shape, strict=True):
            if isinstance(dim, AxisRef):
                if dim.axis.size != size:
                    raise ValueError("AxisRef size does not match array shape.")
                axes.append(dim)
            elif dim is None:
                axes.append(UnboundAxis(int(size)))
            else:
                axes.append(Axis(AxisKey("field", str(dim)), int(size)).ref())
        return cls(tuple(axes))


@jax.tree_util.register_pytree_node_class
class AxisArray:
    """JAX array carrying a static semantic axis layout."""

    __array_priority__ = 1000

    def __init__(
        self,
        data: ArrayLike,
        /,
        *,
        axes: AxisLayout | tuple[str | None | AxisRef, ...] | None = None,
        dims: tuple[str | None | AxisRef, ...] | None = None,
    ):
        value = jnp.asarray(data)
        if axes is not None and dims is not None:
            raise ValueError("Provide axes or dims, not both.")
        specification = dims if axes is None else axes
        if specification is None:
            layout = AxisLayout(tuple(UnboundAxis(int(size)) for size in value.shape))
        elif isinstance(specification, AxisLayout):
            layout = specification
        else:
            layout = AxisLayout.from_dims(tuple(specification), value.shape)
        if layout.shape != value.shape:
            raise ValueError(
                f"Axis layout shape {layout.shape} does not match data shape {value.shape}."
            )
        self.data = value
        self.layout = layout

    def tree_flatten(self):
        return (self.data,), self.layout

    @classmethod
    def tree_unflatten(cls, layout: AxisLayout, children):
        (data,) = children
        if not eqx.is_array(data):
            obj = object.__new__(cls)
            obj.data = data
            obj.layout = layout
            return obj
        value = data
        axes = list(layout.axes)
        if value.ndim > len(axes):
            axes = [
                UnboundAxis(int(size)) for size in value.shape[: value.ndim - len(axes)]
            ] + axes
        elif value.ndim < len(axes):
            removed = axes[: len(axes) - value.ndim]
            if any(not isinstance(axis, UnboundAxis) for axis in removed):
                raise ValueError("A JAX transform removed a bound semantic axis.")
            axes = axes[len(removed) :]
        adjusted = AxisLayout(
            tuple(
                UnboundAxis(int(size)) if isinstance(axis, UnboundAxis) else axis
                for axis, size in zip(axes, value.shape, strict=True)
            )
        )
        obj = object.__new__(cls)
        obj.data = value
        obj.layout = adjusted
        return obj

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def dims(self) -> tuple[str | None, ...]:
        return tuple(
            axis.axis.key.name if isinstance(axis, AxisRef) else None
            for axis in self.layout.axes
        )

    @property
    def named_dims(self) -> tuple[str, ...]:
        return tuple(dim for dim in self.dims if dim is not None)

    @property
    def named_shape(self) -> dict[str, int]:
        return {
            axis.axis.key.name: axis.axis.size
            for axis in self.layout.axes
            if isinstance(axis, AxisRef)
        }

    @property
    def positional_shape(self) -> tuple[int, ...]:
        return tuple(
            axis.size for axis in self.layout.axes if isinstance(axis, UnboundAxis)
        )

    def unwrap(self, expected: AxisLayout | None = None, /) -> Array:
        if expected is not None and self.layout != expected:
            raise ValueError("AxisArray layout does not match the expected layout.")
        return self.data

    def with_data(self, data: ArrayLike, /) -> AxisArray:
        return AxisArray(data, axes=self.layout)

    def _aligned(self, names: tuple[str, ...], sizes: dict[str, int]) -> Array:
        dims = self.dims
        named_positions = {
            dim: index for index, dim in enumerate(dims) if dim is not None
        }
        positional_positions = [index for index, dim in enumerate(dims) if dim is None]
        present = [name for name in names if name in named_positions]
        permutation = [named_positions[name] for name in present] + positional_positions
        value = (
            self.data
            if permutation == list(range(self.ndim))
            else jnp.transpose(self.data, permutation)
        )
        present_sizes = [self.named_shape[name] for name in present]
        positional = [self.shape[index] for index in positional_positions]
        reshape: list[int] = []
        cursor = 0
        for name in names:
            if name in named_positions:
                reshape.append(present_sizes[cursor])
                cursor += 1
            else:
                reshape.append(1)
        reshape.extend(positional)
        value = jnp.reshape(value, tuple(reshape))
        target = tuple(sizes[name] for name in names) + tuple(positional)
        return jnp.broadcast_to(value, target)

    @staticmethod
    def _coerce(value: Any) -> AxisArray:
        return value if isinstance(value, AxisArray) else AxisArray(value)

    def _binary(self, other: Any, operation, /) -> AxisArray:
        right = self._coerce(other)
        names = tuple(dict.fromkeys((*self.named_dims, *right.named_dims)))
        references = {
            axis.axis.key.name: axis
            for field in (self, right)
            for axis in field.layout.axes
            if isinstance(axis, AxisRef)
        }
        sizes: dict[str, int] = {}
        for field in (self, right):
            for name, size in field.named_shape.items():
                previous = sizes.get(name)
                if previous is not None and previous != size:
                    raise ValueError(f"Axis {name!r} has incompatible sizes.")
                sizes[name] = size
        left_data = self._aligned(names, sizes)
        right_data = right._aligned(names, sizes)
        positional_rank = max(
            len(self.positional_shape),
            len(right.positional_shape),
        )
        left_data = jnp.reshape(
            left_data,
            left_data.shape + (1,) * (positional_rank - len(self.positional_shape)),
        )
        right_data = jnp.reshape(
            right_data,
            right_data.shape + (1,) * (positional_rank - len(right.positional_shape)),
        )
        result = operation(left_data, right_data)
        positional_rank = result.ndim - len(names)
        layout = AxisLayout(
            tuple(references[name] for name in names)
            + tuple(UnboundAxis(int(size)) for size in result.shape[len(names) :])
        )
        return AxisArray(result, axes=layout)

    def broadcast_like(self, other: AxisArray, /) -> AxisArray:
        if not isinstance(other, AxisArray):
            raise TypeError("broadcast_like expects an AxisArray.")
        names = tuple(dict.fromkeys((*self.named_dims, *other.named_dims)))
        sizes = dict(other.named_shape)
        for name, size in self.named_shape.items():
            if name in sizes and sizes[name] != size:
                raise ValueError(f"Axis {name!r} has incompatible sizes.")
            sizes.setdefault(name, size)
        value = self._aligned(names, sizes)
        positional_shape = jnp.broadcast_shapes(
            value.shape[len(names) :],
            other.positional_shape,
        )
        value = jnp.broadcast_to(
            value, tuple(sizes[name] for name in names) + positional_shape
        )
        return AxisArray(value, dims=names + (None,) * len(positional_shape))

    def order_as(self, *dims: str) -> AxisArray:
        if set(dims) != set(self.named_dims):
            raise ValueError("order_as must name every bound axis exactly once.")
        sizes = dict(self.named_shape)
        value = self._aligned(tuple(dims), sizes)
        return AxisArray(value, dims=tuple(dims) + (None,) * len(self.positional_shape))

    def __add__(self, other):
        return self._binary(other, jnp.add)

    def __radd__(self, other):
        return self._coerce(other)._binary(self, jnp.add)

    def __sub__(self, other):
        return self._binary(other, jnp.subtract)

    def __rsub__(self, other):
        return self._coerce(other)._binary(self, jnp.subtract)

    def __mul__(self, other):
        return self._binary(other, jnp.multiply)

    def __rmul__(self, other):
        return self._coerce(other)._binary(self, jnp.multiply)

    def __truediv__(self, other):
        return self._binary(other, jnp.divide)

    def __rtruediv__(self, other):
        return self._coerce(other)._binary(self, jnp.divide)

    def __pow__(self, other):
        return self._binary(other, jnp.power)

    def __neg__(self):
        return AxisArray(-self.data, axes=self.layout)

    def __abs__(self):
        return AxisArray(jnp.abs(self.data), axes=self.layout)

    def __matmul__(self, other):
        return self._binary(other, jnp.matmul)

    def __eq__(self, other):
        return self._binary(other, jnp.equal)

    def __ne__(self, other):
        return self._binary(other, jnp.not_equal)

    def __lt__(self, other):
        return self._binary(other, jnp.less)

    def __le__(self, other):
        return self._binary(other, jnp.less_equal)

    def __gt__(self, other):
        return self._binary(other, jnp.greater)

    def __ge__(self, other):
        return self._binary(other, jnp.greater_equal)


def axis_array(
    data: ArrayLike,
    /,
    *,
    dims: tuple[str | None | AxisRef, ...] | None = None,
) -> AxisArray:
    return AxisArray(data, dims=dims)


@dataclass(frozen=True, slots=True)
class AxisAlignmentPlan:
    """Prepared named-axis alignment into one target layout."""

    source: AxisLayout
    target: AxisLayout

    def apply(self, value: AxisArray, /) -> AxisArray:
        if value.layout != self.source:
            raise ValueError("Axis alignment input layout does not match its plan.")
        target_names = tuple(
            axis.axis.key.name for axis in self.target.axes if isinstance(axis, AxisRef)
        )
        target_sizes = {
            axis.axis.key.name: axis.axis.size
            for axis in self.target.axes
            if isinstance(axis, AxisRef)
        }
        aligned = value._aligned(target_names, target_sizes)
        if aligned.shape != self.target.shape:
            aligned = jnp.broadcast_to(aligned, self.target.shape)
        return AxisArray(aligned, axes=self.target)


@dataclass(frozen=True, slots=True)
class AxisReductionPlan:
    """Prepared reduction of selected semantic axes."""

    source: AxisLayout
    reduced: tuple[AxisKey, ...]

    @property
    def target(self) -> AxisLayout:
        keys = set(self.reduced)
        return AxisLayout(
            tuple(
                axis
                for axis in self.source.axes
                if not isinstance(axis, AxisRef) or axis.axis.key not in keys
            )
        )

    def apply(self, value: AxisArray, /) -> AxisArray:
        if value.layout != self.source:
            raise ValueError("Axis reduction input layout does not match its plan.")
        positions = tuple(
            index
            for index, axis in enumerate(self.source.axes)
            if isinstance(axis, AxisRef) and axis.axis.key in set(self.reduced)
        )
        result = jnp.sum(value.data, axis=positions)
        return AxisArray(result, axes=self.target)


@dataclass(frozen=True, slots=True)
class AxisContractionPlan:
    """Prepared pairwise contraction between two axis layouts."""

    left: AxisLayout
    right: AxisLayout
    pairs: tuple[tuple[AxisRef, AxisRef], ...]

    @property
    def target(self) -> AxisLayout:
        left_reduced = {left for left, _ in self.pairs}
        right_reduced = {right for _, right in self.pairs}
        return AxisLayout(
            tuple(axis for axis in self.left.axes if axis not in left_reduced)
            + tuple(axis for axis in self.right.axes if axis not in right_reduced)
        )

    def apply(self, left: AxisArray, right: AxisArray, /) -> AxisArray:
        if left.layout != self.left or right.layout != self.right:
            raise ValueError("Axis contraction inputs do not match their plan.")
        left_positions = tuple(self.left.axes.index(axis) for axis, _ in self.pairs)
        right_positions = tuple(self.right.axes.index(axis) for _, axis in self.pairs)
        for left_axis, right_axis in self.pairs:
            if left_axis.axis.size != right_axis.axis.size:
                raise ValueError("Contracted axes must have equal sizes.")
        result = jnp.tensordot(
            left.data,
            right.data,
            axes=(left_positions, right_positions),
        )
        return AxisArray(result, axes=self.target)


def align_to(value: AxisArray, layout: AxisLayout, /) -> AxisArray:
    return AxisAlignmentPlan(value.layout, layout).apply(value)


def reduce_axes(
    value: AxisArray,
    axes: Sequence[AxisKey],
    /,
) -> AxisArray:
    return AxisReductionPlan(value.layout, tuple(axes)).apply(value)


def cmap(function=None, /, *, out_axes: str = "leading"):
    """Map one positional kernel over the union of bound array axes."""
    if out_axes != "leading":
        raise ValueError("Native cmap supports out_axes='leading'.")

    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            leaves = jax.tree.leaves(
                (args, kwargs),
                is_leaf=lambda value: isinstance(value, AxisArray),
            )
            fields = [leaf for leaf in leaves if isinstance(leaf, AxisArray)]
            if not fields:
                return fn(*args, **kwargs)
            names = tuple(
                dict.fromkeys(name for field in fields for name in field.named_dims)
            )
            sizes: dict[str, int] = {}
            for field in fields:
                for name, size in field.named_shape.items():
                    if name in sizes and sizes[name] != size:
                        raise ValueError(f"Axis {name!r} has incompatible sizes.")
                    sizes[name] = size

            def unwrap(value):
                if not isinstance(value, AxisArray):
                    return value
                return value._aligned(names, sizes)

            packed = jax.tree.map(
                unwrap,
                (args, kwargs),
                is_leaf=lambda value: isinstance(value, AxisArray),
            )
            packed_axes = jax.tree.map(
                lambda value: 0 if isinstance(value, AxisArray) else None,
                (args, kwargs),
                is_leaf=lambda value: isinstance(value, AxisArray),
            )

            def call_packed(arguments):
                positional, keyword = arguments
                return fn(*positional, **keyword)

            mapped = call_packed
            for _ in names:
                mapped = jax.vmap(mapped, in_axes=(packed_axes,))
            result = mapped(packed)

            def rewrap(value):
                if not isinstance(value, (jax.Array, jnp.ndarray)):
                    return value
                positional_rank = value.ndim - len(names)
                return AxisArray(value, dims=names + (None,) * positional_rank)

            return jax.tree.map(rewrap, result)

        return wrapped

    return decorate(function) if function is not None else decorate


__all__ = [
    "Axis",
    "AxisAlignmentPlan",
    "AxisArray",
    "AxisContractionPlan",
    "AxisKey",
    "AxisLayout",
    "AxisReductionPlan",
    "AxisRef",
    "UnboundAxis",
    "align_to",
    "axis_array",
    "cmap",
    "reduce_axes",
]
