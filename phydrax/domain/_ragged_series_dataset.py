#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from ._domain import _AbstractUnaryDomain
from ._structure import _validate_label, PointsBatch, ProductStructure


RAGGED_SERIES_INDEX_KEY = "__phydrax_ragged_series_index__"
RaggedSeriesMeasureMode = Literal["probability", "count"]


def _tree_leading_axis_size(tree: PyTree[ArrayLike], /, *, name: str) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError(f"{name} requires at least one array leaf.")

    first = jnp.asarray(leaves[0])
    if first.ndim == 0:
        raise ValueError(f"{name} leaves must have a leading case axis.")
    n = int(first.shape[0])
    if n <= 0:
        raise ValueError(f"{name} leading case axis must be non-empty.")

    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if arr.ndim == 0:
            raise ValueError(f"{name} leaves must have a leading case axis.")
        if int(arr.shape[0]) != n:
            raise ValueError(
                f"{name} requires all leaves to share the same leading case axis; "
                f"got {int(arr.shape[0])} and {n}."
            )
    return n


def _validate_series(series: PyTree[ArrayLike], /) -> tuple[PyTree[Array], int, int]:
    arrays = jax.tree_util.tree_map(lambda x: jnp.asarray(x), series)
    leaves = jax.tree_util.tree_leaves(arrays)
    if not leaves:
        raise ValueError("RaggedSeriesDatasetDomain requires at least one series leaf.")

    first = jnp.asarray(leaves[0])
    if first.ndim < 2:
        raise ValueError(
            "Ragged series leaves must have shape (N, Lmax, ...); "
            f"got {first.shape}."
        )
    n = int(first.shape[0])
    max_length = int(first.shape[1])
    if n <= 0:
        raise ValueError("Ragged series leading case axis must be non-empty.")
    if max_length <= 0:
        raise ValueError("Ragged series time axis must be non-empty.")

    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if arr.ndim < 2:
            raise ValueError(
                "Ragged series leaves must have shape (N, Lmax, ...); "
                f"got {arr.shape}."
            )
        if int(arr.shape[0]) != n:
            raise ValueError(
                "Ragged series leaves must share leading case axis; "
                f"got {int(arr.shape[0])} and {n}."
            )
        if int(arr.shape[1]) != max_length:
            raise ValueError(
                "Ragged series leaves must share padded time axis; "
                f"got {int(arr.shape[1])} and {max_length}."
            )
    return arrays, n, max_length


def _validate_static(
    static: PyTree[ArrayLike] | None,
    /,
    *,
    n: int,
) -> PyTree[Array] | None:
    if static is None:
        return None
    arrays = jax.tree_util.tree_map(lambda x: jnp.asarray(x), static)
    static_n = _tree_leading_axis_size(arrays, name="Ragged static data")
    if int(static_n) != int(n):
        raise ValueError(
            "Ragged static data leading case axis must match series leading axis; "
            f"got {static_n} and {n}."
        )
    return arrays


def _as_lengths(lengths: ArrayLike, n: int, max_length: int, /) -> Array:
    arr = jnp.asarray(lengths)
    if arr.ndim != 1:
        raise ValueError(f"lengths must have shape (N,), got {arr.shape}.")
    if int(arr.shape[0]) != int(n):
        raise ValueError(f"lengths must have length {n}, got {int(arr.shape[0])}.")
    arr_i = arr.astype(jnp.int32)
    if bool(jnp.any(arr_i <= 0)):
        raise ValueError("All ragged series lengths must be positive.")
    if bool(jnp.any(arr_i > int(max_length))):
        raise ValueError("Ragged series lengths cannot exceed the padded time axis.")
    if bool(jnp.any(arr_i != arr)):
        raise ValueError("Ragged series lengths must be integer-valued.")
    return arr_i


def _as_scalar(name: str, value: ArrayLike, /) -> Array:
    arr = jnp.asarray(value, dtype=float)
    if arr.shape != ():
        raise ValueError(f"{name} must be scalar, got shape {arr.shape}.")
    return arr.reshape(())


class RaggedSeriesDatasetDomain(_AbstractUnaryDomain):
    """A finite dataset domain for conditional variable-length input series.

    Each case owns optional static data, one or more aligned padded series leaves,
    and an integer valid length. The padded time axis is part of the empirical row,
    so sampling is by case. This is intended for conditional operators such as
    `(static, ragged series) -> scalar/vector target`.
    """

    static: PyTree[Array] | None
    series: PyTree[Array]
    lengths: Array
    start: Array
    dt: Array
    _label: str
    _measure_mode: RaggedSeriesMeasureMode
    _size: int
    _max_length: int
    _time_axis: Array

    def __init__(
        self,
        series: PyTree[ArrayLike],
        lengths: ArrayLike,
        /,
        *,
        static: PyTree[ArrayLike] | None = None,
        start: ArrayLike = 0.0,
        dt: ArrayLike = 1.0,
        label: str = "data",
        measure: RaggedSeriesMeasureMode = "probability",
    ):
        _validate_label(str(label))
        series_arrays, n, max_length = _validate_series(series)
        static_arrays = _validate_static(static, n=n)
        lengths_arr = _as_lengths(lengths, n, max_length)
        start_arr = _as_scalar("start", start)
        dt_arr = _as_scalar("dt", dt)
        if bool(dt_arr <= 0):
            raise ValueError("dt must be positive.")
        if measure not in ("probability", "count"):
            raise ValueError("measure must be 'probability' or 'count'.")

        self.static = static_arrays
        self.series = series_arrays
        self.lengths = lengths_arr
        self.start = start_arr
        self.dt = dt_arr
        self._label = str(label)
        self._measure_mode = measure
        self._size = int(n)
        self._max_length = int(max_length)
        self._time_axis = start_arr + dt_arr * jnp.arange(max_length, dtype=float)

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
    def max_length(self) -> int:
        return int(self._max_length)

    @property
    def measure_mode(self) -> RaggedSeriesMeasureMode:
        return self._measure_mode

    @property
    def measure(self) -> Array:
        if self._measure_mode == "count":
            return jnp.asarray(float(self._size), dtype=float)
        return jnp.asarray(1.0, dtype=float)

    @property
    def time_axis(self) -> Array:
        return self._time_axis

    def sample(
        self,
        num_points: int,
        *,
        sampler: str = "uniform",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> dict[str, Any]:
        indices = self.sample_indices(num_points, sampler=sampler, key=key)
        return self.input_rows(indices)

    def sample_indices(
        self,
        num_points: int,
        *,
        sampler: str = "uniform",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        del sampler
        n = int(num_points)
        if n < 0:
            raise ValueError("num_points must be non-negative.")
        if n == 0:
            return jnp.zeros((0,), dtype=jnp.int32)
        return jr.randint(
            key,
            shape=(n,),
            minval=0,
            maxval=int(self._size),
            dtype=jnp.int32,
        )

    def input_rows(self, indices: ArrayLike, /) -> dict[str, Any]:
        idx = jnp.asarray(indices, dtype=jnp.int32)
        lengths = self.lengths[idx]
        time = jnp.broadcast_to(self._time_axis, (int(idx.shape[0]), self.max_length))
        mask = jnp.arange(self.max_length, dtype=jnp.int32)[None, :] < lengths[:, None]
        rows: dict[str, Any] = {
            "series": jax.tree_util.tree_map(lambda a: jnp.asarray(a)[idx], self.series),
            "time": time,
            "mask": mask,
            "length": lengths,
        }
        if self.static is not None:
            rows["static"] = jax.tree_util.tree_map(
                lambda a: jnp.asarray(a)[idx],
                self.static,
            )
        return rows

    def points_from_indices(
        self,
        indices: ArrayLike,
        /,
        *,
        structure: ProductStructure | None = None,
    ) -> PointsBatch:
        structure_in = structure or ProductStructure(((self.label,),))
        structure_out = structure_in.canonicalize(self.labels)
        axis = structure_out.axis_for(self.label)
        if axis is None:
            raise ValueError(
                f"RaggedSeriesDatasetDomain points require a sampling axis for "
                f"label {self.label!r}."
            )

        idx = jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
        rows = self.input_rows(idx)

        def _to_field(value: ArrayLike):
            arr = jnp.asarray(value)
            if arr.ndim == 0:
                raise ValueError(
                    "Ragged series indexed rows must retain a leading sample axis."
                )
            return cx.Field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

        points = {
            self.label: jax.tree_util.tree_map(_to_field, rows),
            RAGGED_SERIES_INDEX_KEY: cx.Field(idx, dims=(axis,)),
        }
        return PointsBatch(points=frozendict(points), structure=structure_out)

    def equivalent(self, other: object, /) -> bool:
        if not isinstance(other, RaggedSeriesDatasetDomain):
            return False
        if self.label != other.label:
            return False
        if self.measure_mode != other.measure_mode:
            return False
        if self.size != other.size:
            return False
        if self.max_length != other.max_length:
            return False
        if bool(jnp.any(self.lengths != other.lengths)):
            return False
        if bool(jnp.any(self.start != other.start)):
            return False
        if bool(jnp.any(self.dt != other.dt)):
            return False
        if jax.tree_util.tree_structure(self.static) != jax.tree_util.tree_structure(
            other.static
        ):
            return False
        if jax.tree_util.tree_structure(self.series) != jax.tree_util.tree_structure(
            other.series
        ):
            return False

        for a, b in zip(
            jax.tree_util.tree_leaves(self.static),
            jax.tree_util.tree_leaves(other.static),
            strict=True,
        ):
            arr_a = jnp.asarray(a)
            arr_b = jnp.asarray(b)
            if arr_a.shape != arr_b.shape:
                return False
            if arr_a.dtype != arr_b.dtype:
                return False

        for a, b in zip(
            jax.tree_util.tree_leaves(self.series),
            jax.tree_util.tree_leaves(other.series),
            strict=True,
        ):
            arr_a = jnp.asarray(a)
            arr_b = jnp.asarray(b)
            if arr_a.shape != arr_b.shape:
                return False
            if arr_a.dtype != arr_b.dtype:
                return False

        return True


__all__ = [
    "RAGGED_SERIES_INDEX_KEY",
    "RaggedSeriesDatasetDomain",
    "RaggedSeriesMeasureMode",
]
