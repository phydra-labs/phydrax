#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._core import TensorTrain, TensorTrainCompressionResult, tt_svd


DigitOrdering = Literal["blocked", "interleaved"]
GridRule = Literal["trapezoid", "midpoint"]


def _bounded_count(count: int, maximum: int, label: str, /) -> int:
    value = int(count)
    limit = int(maximum)
    if value < 0 or limit <= 0 or value > limit:
        raise ValueError(f"{label} count {value} exceeds explicit budget {limit}.")
    return value


class TensorizedGrid(StrictModule):
    """Finite Cartesian grid with explicit one-dimensional nodes and weights."""

    axis_nodes: tuple[Array, ...]
    axis_weights: tuple[Array, ...]
    mode_sizes: tuple[int, ...] = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_nodes: Sequence[ArrayLike],
        axis_weights: Sequence[ArrayLike],
        /,
    ):
        nodes = tuple(jnp.asarray(axis) for axis in axis_nodes)
        weights = tuple(jnp.asarray(axis) for axis in axis_weights)
        if not nodes or len(nodes) != len(weights):
            raise ValueError(
                "A TensorizedGrid needs matching nonempty node and weight axes."
            )
        for node, weight in zip(nodes, weights, strict=True):
            if node.ndim != 1 or node.size == 0 or weight.shape != node.shape:
                raise ValueError(
                    "Every grid axis needs nonempty vector nodes and weights."
                )
        dtype = nodes[0].dtype
        if any(node.dtype != dtype for node in nodes):
            raise TypeError("TensorizedGrid nodes must use one dtype.")
        self.axis_nodes = nodes
        self.axis_weights = weights
        self.mode_sizes = tuple(int(axis.size) for axis in nodes)
        self.grid_id = canonical_fingerprint(
            {
                "kind": "tensorized-grid",
                "mode_sizes": self.mode_sizes,
                "node_dtype": str(dtype),
                "weight_dtypes": tuple(str(weight.dtype) for weight in weights),
            }
        )

    @staticmethod
    def uniform(
        bounds: Sequence[tuple[float, float]],
        mode_sizes: Sequence[int],
        /,
        *,
        rule: GridRule = "trapezoid",
        dtype=jnp.float32,
    ) -> TensorizedGrid:
        sizes = tuple(int(size) for size in mode_sizes)
        intervals = tuple((float(lower), float(upper)) for lower, upper in bounds)
        if not sizes or len(sizes) != len(intervals) or any(size <= 0 for size in sizes):
            raise ValueError("Uniform grid bounds and positive mode sizes must agree.")
        if any(
            not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower
            for lower, upper in intervals
        ):
            raise ValueError("Uniform grid bounds must be finite increasing intervals.")
        if rule not in ("trapezoid", "midpoint"):
            raise ValueError("Grid rule must be 'trapezoid' or 'midpoint'.")
        nodes: list[Array] = []
        weights: list[Array] = []
        for (lower, upper), size in zip(intervals, sizes, strict=True):
            if rule == "midpoint":
                spacing = (upper - lower) / size
                node = lower + (jnp.arange(size, dtype=dtype) + 0.5) * spacing
                weight = jnp.full((size,), spacing, dtype=dtype)
            else:
                if size < 2:
                    raise ValueError("Trapezoid axes require at least two nodes.")
                node = jnp.linspace(lower, upper, size, dtype=dtype)
                spacing = (upper - lower) / (size - 1)
                weight = jnp.full((size,), spacing, dtype=dtype)
                weight = weight.at[0].set(0.5 * spacing)
                weight = weight.at[-1].set(0.5 * spacing)
            nodes.append(node)
            weights.append(weight)
        return TensorizedGrid(tuple(nodes), tuple(weights))

    @property
    def dimension(self) -> int:
        return len(self.mode_sizes)

    @property
    def point_count(self) -> int:
        return prod(self.mode_sizes)

    def indices(self, /, *, max_points: int) -> Array:
        _bounded_count(self.point_count, max_points, "grid point")
        flat = jnp.arange(self.point_count, dtype=jnp.int32)
        columns = []
        stride = self.point_count
        for size in self.mode_sizes:
            stride //= size
            columns.append((flat // stride) % size)
        return jnp.stack(columns, axis=-1)

    def coordinates(self, indices: ArrayLike, /) -> Array:
        points = jnp.asarray(indices, dtype=jnp.int32)
        if points.ndim < 1 or points.shape[-1] != self.dimension:
            raise ValueError("Grid indices need one trailing coordinate per axis.")
        return jnp.stack(
            tuple(
                axis[points[..., position]]
                for position, axis in enumerate(self.axis_nodes)
            ),
            axis=-1,
        )

    def weights(self, indices: ArrayLike, /) -> Array:
        points = jnp.asarray(indices, dtype=jnp.int32)
        if points.ndim < 1 or points.shape[-1] != self.dimension:
            raise ValueError("Grid indices need one trailing coordinate per axis.")
        result = jnp.ones(points.shape[:-1], dtype=self.axis_weights[0].dtype)
        for position, weights in enumerate(self.axis_weights):
            result = result * weights[points[..., position]]
        return result


class QuanticsLayout(StrictModule):
    """Invertible mixed-radix digit ordering for Cartesian tensor indices."""

    axis_sizes: tuple[int, ...] = eqx.field(static=True)
    axis_digit_sizes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    digit_axes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    digit_mode_sizes: tuple[int, ...] = eqx.field(static=True)
    ordering: DigitOrdering = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_sizes: Sequence[int],
        axis_digit_sizes: Sequence[Sequence[int]],
        /,
        *,
        ordering: DigitOrdering = "interleaved",
    ):
        sizes = tuple(int(size) for size in axis_sizes)
        digits = tuple(tuple(int(base) for base in axis) for axis in axis_digit_sizes)
        if not sizes or len(sizes) != len(digits) or any(size <= 0 for size in sizes):
            raise ValueError(
                "QuanticsLayout needs positive axes and one digitization each."
            )
        if any(not axis or any(base <= 1 for base in axis) for axis in digits):
            raise ValueError("Every quantics digit base must exceed one.")
        if any(prod(axis) != size for size, axis in zip(sizes, digits, strict=True)):
            raise ValueError("Each axis size must equal the product of its digit bases.")
        if ordering not in ("blocked", "interleaved"):
            raise ValueError("Quantics ordering must be 'blocked' or 'interleaved'.")
        if ordering == "blocked":
            digit_axes = tuple(
                (axis, digit)
                for axis, axis_digits in enumerate(digits)
                for digit in range(len(axis_digits))
            )
        else:
            depth = max(len(axis) for axis in digits)
            digit_axes = tuple(
                (axis, digit)
                for digit in range(depth)
                for axis, axis_digits in enumerate(digits)
                if digit < len(axis_digits)
            )
        self.axis_sizes = sizes
        self.axis_digit_sizes = digits
        self.digit_axes = digit_axes
        self.digit_mode_sizes = tuple(digits[axis][digit] for axis, digit in digit_axes)
        self.ordering = ordering
        self.layout_id = canonical_fingerprint(
            {
                "kind": "quantics-layout",
                "axis_sizes": sizes,
                "axis_digit_sizes": digits,
                "ordering": ordering,
            }
        )

    @staticmethod
    def binary(
        axis_sizes: Sequence[int],
        /,
        *,
        ordering: DigitOrdering = "interleaved",
    ) -> QuanticsLayout:
        sizes = tuple(int(size) for size in axis_sizes)
        digit_sizes: list[tuple[int, ...]] = []
        for size in sizes:
            if size <= 0 or size & (size - 1):
                raise ValueError("Binary quantics axes must be positive powers of two.")
            digit_sizes.append((2,) * (size.bit_length() - 1))
        return QuanticsLayout(sizes, tuple(digit_sizes), ordering=ordering)

    @property
    def axis_count(self) -> int:
        return len(self.axis_sizes)

    @property
    def digit_count(self) -> int:
        return len(self.digit_axes)

    def digitize(self, indices: ArrayLike, /) -> Array:
        points = jnp.asarray(indices, dtype=jnp.int32)
        if points.ndim < 1 or points.shape[-1] != self.axis_count:
            raise ValueError("Quantics digitization needs one index per physical axis.")
        blocked: list[list[Array]] = []
        for axis, bases in enumerate(self.axis_digit_sizes):
            value = points[..., axis]
            axis_digits: list[Array] = []
            for position, base in enumerate(bases):
                stride = prod(bases[position + 1 :])
                axis_digits.append((value // stride) % base)
            blocked.append(axis_digits)
        return jnp.stack(
            tuple(blocked[axis][digit] for axis, digit in self.digit_axes), axis=-1
        )

    def undigitize(self, digits: ArrayLike, /) -> Array:
        values = jnp.asarray(digits, dtype=jnp.int32)
        if values.ndim < 1 or values.shape[-1] != self.digit_count:
            raise ValueError("Quantics digits do not match this layout.")
        physical = []
        for axis, bases in enumerate(self.axis_digit_sizes):
            value = jnp.zeros(values.shape[:-1], dtype=jnp.int32)
            for digit, base in enumerate(bases):
                position = self.digit_axes.index((axis, digit))
                value = value * base + values[..., position]
            physical.append(value)
        return jnp.stack(tuple(physical), axis=-1)

    def tensorize(self, dense: ArrayLike, /) -> Array:
        values = jnp.asarray(dense)
        if values.shape != self.axis_sizes:
            raise ValueError("Dense tensor shape does not match QuanticsLayout axes.")
        blocked_modes = tuple(base for axis in self.axis_digit_sizes for base in axis)
        blocked_axes = tuple(
            (axis, digit)
            for axis, bases in enumerate(self.axis_digit_sizes)
            for digit in range(len(bases))
        )
        reshaped = values.reshape(blocked_modes)
        permutation = tuple(blocked_axes.index(pair) for pair in self.digit_axes)
        return jnp.transpose(reshaped, permutation)

    def untensorize(self, digit_tensor: ArrayLike, /) -> Array:
        values = jnp.asarray(digit_tensor)
        if values.shape != self.digit_mode_sizes:
            raise ValueError("Digit tensor shape does not match QuanticsLayout.")
        blocked_axes = tuple(
            (axis, digit)
            for axis, bases in enumerate(self.axis_digit_sizes)
            for digit in range(len(bases))
        )
        permutation = tuple(self.digit_axes.index(pair) for pair in blocked_axes)
        return jnp.transpose(values, permutation).reshape(self.axis_sizes)


class TensorFunction(StrictModule):
    """A callable bound to a finite tensorized grid and explicit vectorization contract."""

    function: Callable = eqx.field(static=True)
    grid: TensorizedGrid
    vectorized: bool = eqx.field(static=True)
    function_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable,
        grid: TensorizedGrid,
        /,
        *,
        vectorized: bool,
        name: str,
    ):
        if not callable(function):
            raise TypeError("TensorFunction function must be callable.")
        name_ = str(name)
        if not name_:
            raise ValueError("TensorFunction name must be nonempty.")
        self.function = function
        self.grid = grid
        self.vectorized = bool(vectorized)
        self.function_id = canonical_fingerprint(
            {"kind": "tensor-function", "name": name_, "grid": grid.grid_id}
        )

    def evaluate(self, indices: ArrayLike, /, *, max_evaluations: int) -> Array:
        points = jnp.asarray(indices, dtype=jnp.int32)
        if points.ndim < 1 or points.shape[-1] != self.grid.dimension:
            raise ValueError("TensorFunction indices do not match its grid.")
        count = prod(points.shape[:-1])
        _bounded_count(count, max_evaluations, "function evaluation")
        coordinates = self.grid.coordinates(points)
        flat = coordinates.reshape((-1, self.grid.dimension))
        values = self.function(flat) if self.vectorized else jax.vmap(self.function)(flat)
        result = jnp.asarray(values)
        if result.shape != (flat.shape[0],):
            raise ValueError("TensorFunction must return one scalar per grid point.")
        return result.reshape(points.shape[:-1])

    def tabulate(self, /, *, max_evaluations: int) -> Array:
        indices = self.grid.indices(max_points=max_evaluations)
        return self.evaluate(indices, max_evaluations=max_evaluations).reshape(
            self.grid.mode_sizes
        )

    def quadrature(self, /, *, max_evaluations: int) -> Array:
        indices = self.grid.indices(max_points=max_evaluations)
        values = self.evaluate(indices, max_evaluations=max_evaluations)
        return oe.contract("i,i->", values, self.grid.weights(indices))


def qtt_digitize(
    tensor: ArrayLike,
    layout: QuanticsLayout,
    /,
    *,
    max_ranks: int | Sequence[int],
    relative_tolerance: float = 0.0,
) -> TensorTrainCompressionResult:
    """Digitize a Cartesian tensor and apply explicitly bounded TT-SVD."""
    return tt_svd(
        layout.tensorize(tensor),
        max_ranks=max_ranks,
        relative_tolerance=relative_tolerance,
    )


def qtt_evaluate(
    tensor: TensorTrain,
    layout: QuanticsLayout,
    indices: ArrayLike,
    /,
) -> Array:
    if tensor.mode_sizes != layout.digit_mode_sizes:
        raise ValueError("QTT tensor modes do not match the QuanticsLayout.")
    return tensor.evaluate(layout.digitize(indices))


def qtt_quadrature(
    tensor: TensorTrain,
    layout: QuanticsLayout,
    grid: TensorizedGrid,
    /,
    *,
    max_entries: int,
) -> Array:
    """Contract QTT values with Cartesian quadrature weights under a hard budget."""
    if (
        grid.mode_sizes != layout.axis_sizes
        or tensor.mode_sizes != layout.digit_mode_sizes
    ):
        raise ValueError("QTT quadrature grid, layout, and tensor do not agree.")
    _bounded_count(grid.point_count, max_entries, "quadrature entry")
    digit_values = tensor.to_dense(max_entries=max_entries)
    weights = grid.axis_weights[0]
    for axis_weight in grid.axis_weights[1:]:
        weights = weights[..., None] * axis_weight
    return jnp.sum(layout.untensorize(digit_values) * weights.reshape(grid.mode_sizes))


def qtt_sample(
    tensor: TensorTrain,
    layout: QuanticsLayout,
    key: PRNGKeyArray,
    /,
    *,
    sample_count: int,
    max_entries: int,
) -> Array:
    """Sample physical grid indices from explicitly materialized nonnegative mass."""
    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive and explicit.")
    values = layout.untensorize(tensor.to_dense(max_entries=max_entries)).reshape((-1,))
    if bool(np.any(np.asarray(values) < 0)) or not bool(
        np.all(np.isfinite(np.asarray(values)))
    ):
        raise ValueError("QTT sampling requires finite nonnegative mass.")
    total = jnp.sum(values)
    if float(np.asarray(total)) <= 0.0:
        raise ValueError("QTT sampling requires positive total mass.")
    flat = jax.random.categorical(key, jnp.log(values / total), shape=(count,))
    columns = []
    stride = prod(layout.axis_sizes)
    for size in layout.axis_sizes:
        stride //= size
        columns.append((flat // stride) % size)
    return jnp.stack(columns, axis=-1).astype(jnp.int32)


__all__ = [
    "DigitOrdering",
    "GridRule",
    "QuanticsLayout",
    "TensorFunction",
    "TensorizedGrid",
    "qtt_digitize",
    "qtt_evaluate",
    "qtt_quadrature",
    "qtt_sample",
]
