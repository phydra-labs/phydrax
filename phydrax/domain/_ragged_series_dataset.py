#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from ._domain import _AbstractUnaryDomain
from ._structure import _validate_label, PointsBatch, ProductStructure


RAGGED_SERIES_INDEX_KEY = "__phydrax_ragged_series_index__"
RaggedSeriesMeasureMode = Literal["probability", "count"]
RaggedSeriesSampling = Literal[
    "full",
    "points_uniform",
    "window_uniform",
    "prefix",
    "suffix",
]


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


def _pack_padded_series(series: PyTree[Array], lengths: Array, /) -> PyTree[Array]:
    length_np = np.asarray(lengths, dtype=np.int32)

    def _pack_leaf(leaf: Array) -> Array:
        arr = np.asarray(leaf)
        mask = np.arange(arr.shape[1], dtype=np.int32)[None, :] < length_np[:, None]
        return jnp.asarray(arr[mask])

    return jax.tree_util.tree_map(_pack_leaf, series)


def _mask_series_tree(series: PyTree[Array], mask: Array, /) -> PyTree[Array]:
    valid = jnp.asarray(mask, dtype=bool)

    def _mask_leaf(leaf: Array) -> Array:
        arr = jnp.asarray(leaf)
        mask_arr = valid
        while mask_arr.ndim < arr.ndim:
            mask_arr = jnp.expand_dims(mask_arr, axis=-1)
        return jnp.where(mask_arr, arr, jnp.zeros((), dtype=arr.dtype))

    return jax.tree_util.tree_map(_mask_leaf, series)


class RaggedSeriesDatasetDomain(_AbstractUnaryDomain):
    """A finite dataset domain for conditional variable-length input series.

    Each case owns optional static data, one or more aligned padded series leaves,
    and an integer valid length. The padded time axis is part of the empirical row,
    so sampling is by case. This is intended for conditional operators such as
    `(static, ragged series) -> scalar/vector target`.
    """

    static: PyTree[Array] | None
    series: PyTree[Array] | None
    series_packed: PyTree[Array]
    offsets: Array
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
        self.series_packed = _pack_padded_series(series_arrays, lengths_arr)
        self.offsets = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.cumsum(lengths_arr, dtype=jnp.int32),
            ],
            axis=0,
        )
        self.lengths = lengths_arr
        self.start = start_arr
        self.dt = dt_arr
        self._label = str(label)
        self._measure_mode = measure
        self._size = int(n)
        self._max_length = int(max_length)
        self._time_axis = start_arr + dt_arr * jnp.arange(max_length, dtype=float)

    @classmethod
    def from_padded(
        cls,
        series: PyTree[ArrayLike],
        lengths: ArrayLike,
        /,
        *,
        static: PyTree[ArrayLike] | None = None,
        start: ArrayLike = 0.0,
        dt: ArrayLike = 1.0,
        label: str = "data",
        measure: RaggedSeriesMeasureMode = "probability",
    ) -> "RaggedSeriesDatasetDomain":
        """Construct a ragged-series dataset from padded series arrays."""
        return cls(
            series,
            lengths,
            static=static,
            start=start,
            dt=dt,
            label=label,
            measure=measure,
        )

    @classmethod
    def from_sequences(
        cls,
        series: Sequence[PyTree[ArrayLike]],
        /,
        *,
        static: PyTree[ArrayLike] | None = None,
        start: ArrayLike = 0.0,
        dt: ArrayLike = 1.0,
        label: str = "data",
        measure: RaggedSeriesMeasureMode = "probability",
    ) -> "RaggedSeriesDatasetDomain":
        """Construct from one PyTree of valid series arrays per case.

        This convenience constructor accepts records whose leaves have leading
        shape `(Li, ...)` and pads once to the maximum valid length. The domain
        also stores a packed valid representation used by sampled mini-batches.
        """
        if len(series) == 0:
            raise ValueError("from_sequences requires at least one series record.")
        treedef = jax.tree_util.tree_structure(series[0])
        leaf_rows = [jax.tree_util.tree_leaves(record) for record in series]
        lengths: list[int] = []
        for record, leaves in zip(series, leaf_rows, strict=True):
            if jax.tree_util.tree_structure(record) != treedef:
                raise ValueError("All series records must share the same PyTree structure.")
            if not leaves:
                raise ValueError("Series records must contain at least one leaf.")
            first = jnp.asarray(leaves[0])
            if first.ndim == 0:
                raise ValueError("Series record leaves must have a leading time axis.")
            length = int(first.shape[0])
            if length <= 0:
                raise ValueError("Series record lengths must be positive.")
            for leaf in leaves:
                arr = jnp.asarray(leaf)
                if arr.ndim == 0:
                    raise ValueError(
                        "Series record leaves must have a leading time axis."
                    )
                if int(arr.shape[0]) != length:
                    raise ValueError(
                        "All leaves in a series record must share the same length."
                    )
            lengths.append(length)

        max_length = max(lengths)
        per_leaf: list[list[Array]] = [[] for _ in leaf_rows[0]]
        for leaves, length in zip(leaf_rows, lengths, strict=True):
            pad = int(max_length) - int(length)
            for i, leaf in enumerate(leaves):
                arr = jnp.asarray(leaf)
                if pad > 0:
                    arr = jnp.concatenate(
                        [
                            arr,
                            jnp.zeros((pad,) + arr.shape[1:], dtype=arr.dtype),
                        ],
                        axis=0,
                    )
                per_leaf[i].append(arr)
        padded_leaves = [jnp.stack(parts, axis=0) for parts in per_leaf]
        padded = jax.tree_util.tree_unflatten(treedef, padded_leaves)
        return cls(
            padded,
            jnp.asarray(lengths, dtype=jnp.int32),
            static=static,
            start=start,
            dt=dt,
            label=label,
            measure=measure,
        )

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

    @property
    def total_observations(self) -> int:
        return int(self.offsets[-1])

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
        positions = jnp.broadcast_to(
            jnp.arange(self.max_length, dtype=jnp.int32),
            (int(idx.shape[0]), self.max_length),
        )
        mask = positions < lengths[:, None]
        return self._rows_from_positions(idx, positions, mask)

    def sampled_input_rows(
        self,
        indices: ArrayLike,
        /,
        *,
        num_series_points: int,
        sampling: RaggedSeriesSampling,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> dict[str, Any]:
        """Return fixed-width sampled series views for selected cases."""
        idx = jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
        k = int(num_series_points)
        if k <= 0:
            raise ValueError("num_series_points must be positive.")
        sampling_str = str(sampling)
        if sampling_str not in (
            "points_uniform",
            "window_uniform",
            "prefix",
            "suffix",
        ):
            raise ValueError(
                "sampled_input_rows sampling must be 'points_uniform', "
                "'window_uniform', 'prefix', or 'suffix'."
            )

        lengths = self.lengths[idx]
        arange_k = jnp.arange(k, dtype=jnp.int32)
        arange_grid = jnp.broadcast_to(arange_k[None, :], (int(idx.shape[0]), k))

        if sampling_str == "points_uniform":
            u = jr.uniform(key, shape=(int(idx.shape[0]), k))
            random_pos = jnp.floor(u * lengths[:, None].astype(float)).astype(jnp.int32)
            positions = jnp.where(lengths[:, None] >= k, random_pos, arange_grid)
            mask = jnp.where(
                lengths[:, None] >= k,
                jnp.ones((int(idx.shape[0]), k), dtype=bool),
                arange_grid < lengths[:, None],
            )
        elif sampling_str == "window_uniform":
            max_start = jnp.maximum(lengths - k, 0)
            u = jr.uniform(key, shape=(int(idx.shape[0]),))
            start = jnp.floor(u * (max_start.astype(float) + 1.0)).astype(jnp.int32)
            positions = start[:, None] + arange_grid
            mask = positions < lengths[:, None]
        elif sampling_str == "prefix":
            positions = arange_grid
            mask = positions < lengths[:, None]
        else:
            start = jnp.maximum(lengths - k, 0)
            positions = start[:, None] + arange_grid
            mask = positions < lengths[:, None]

        positions = jnp.minimum(positions, lengths[:, None] - 1)
        return self._rows_from_positions(idx, positions, mask)

    def _rows_from_positions(
        self,
        indices: Array,
        positions: Array,
        mask: Array,
        /,
    ) -> dict[str, Any]:
        idx = jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
        pos = jnp.asarray(positions, dtype=jnp.int32)
        valid = jnp.asarray(mask, dtype=bool)
        if pos.ndim != 2:
            raise ValueError("series positions must have shape (B, K).")
        if valid.shape != pos.shape:
            raise ValueError("series mask must match sampled positions shape.")
        if int(pos.shape[0]) != int(idx.shape[0]):
            raise ValueError("series positions leading axis must match indices.")

        lengths = self.lengths[idx]
        source_pos = jnp.minimum(pos, lengths[:, None] - 1)
        flat = self.offsets[idx][:, None] + source_pos
        series_rows = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a)[flat],
            self.series_packed,
        )
        valid_counts = jnp.maximum(
            jnp.sum(valid.astype(jnp.int32), axis=1),
            jnp.ones((int(idx.shape[0]),), dtype=jnp.int32),
        )
        rows: dict[str, Any] = {
            "series": _mask_series_tree(series_rows, valid),
            "time": self.start + self.dt * pos.astype(float),
            "mask": valid,
            "length": lengths,
            "sample_index": source_pos,
            "sample_scale": lengths.astype(float) / valid_counts.astype(float),
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

    def sampled_points_from_indices(
        self,
        indices: ArrayLike,
        /,
        *,
        num_series_points: int,
        sampling: RaggedSeriesSampling = "points_uniform",
        structure: ProductStructure | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PointsBatch:
        """Materialize fixed-width sampled series rows for selected cases."""
        structure_in = structure or ProductStructure(((self.label,),))
        structure_out = structure_in.canonicalize(self.labels)
        axis = structure_out.axis_for(self.label)
        if axis is None:
            raise ValueError(
                f"RaggedSeriesDatasetDomain points require a sampling axis for "
                f"label {self.label!r}."
            )
        idx = jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
        rows = self.sampled_input_rows(
            idx,
            num_series_points=num_series_points,
            sampling=sampling,
            key=key,
        )

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
        if jax.tree_util.tree_structure(
            self.series_packed
        ) != jax.tree_util.tree_structure(other.series_packed):
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
            jax.tree_util.tree_leaves(self.series_packed),
            jax.tree_util.tree_leaves(other.series_packed),
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
    "RaggedSeriesSampling",
]
