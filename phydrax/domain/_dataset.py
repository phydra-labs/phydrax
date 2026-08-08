#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TYPE_CHECKING

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from ._coordinate import CoordinateSpec
from ._domain import JointFactor
from ._factor_component import FactorComponent
from ._measure import BaseMeasure, ExactMass
from ._selection import Interior, Selection
from ._structure import PointBatch, SampleLayout


if TYPE_CHECKING:
    from ._function import DomainFunction


DATASET_INDEX_KEY = "__phydrax_dataset_index__"


class DatasetDomain(JointFactor):
    """A single-label domain over a finite in-memory dataset.

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
    def labels(self) -> tuple[str, ...]:
        return (self.label,)

    @property
    def coordinate_specs(self) -> tuple[CoordinateSpec, ...]:
        return (CoordinateSpec(None, kind="pytree", differentiable=False, dtype=None),)

    def bind_component(
        self,
        selections: Mapping[str, Selection],
        /,
    ) -> FactorComponent:
        if tuple(selections) != self.labels:
            raise ValueError(
                f"Dataset factor {self.labels} requires exactly one ordered selection."
            )
        if not isinstance(selections[self.label], Interior):
            raise TypeError("Dataset factors support only Interior selection.")
        normalized = self._measure_mode == "probability"
        kind = "probability" if normalized else "counting"
        return FactorComponent(
            factor=self,
            selections=selections,
            measure=BaseMeasure(kind, ExactMass(self.measure), normalized=normalized),
        )

    def _replace_labels(
        self,
        labels: tuple[str, ...],
        /,
    ) -> "DatasetDomain":
        return eqx.tree_at(lambda factor: factor._label, self, labels[0])

    @property
    def size(self) -> int:
        return int(self._size)

    @property
    def measure(self) -> Array:
        if self._measure_mode == "count":
            return jnp.asarray(float(self._size), dtype=float)
        return jnp.asarray(1.0, dtype=float)

    def field(self, values: ArrayLike, /) -> "DomainFunction":
        """Expose row-aligned target values as a non-trainable domain function."""
        from ._observation import indexed_field

        return indexed_field(
            self,
            values,
            size=self.size,
            index_key=DATASET_INDEX_KEY,
            owner="DatasetDomain.field",
        )

    def sample(
        self,
        num_points: int,
        *,
        sampler: str = "uniform",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> PyTree[Array]:
        idx = self.sample_indices(num_points, sampler=sampler, key=key)
        return self.input_rows(idx)

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

    def input_rows(self, indices: ArrayLike, /) -> PyTree[Array]:
        idx = jnp.asarray(indices, dtype=jnp.int32)
        return jax.tree_util.tree_map(lambda a: jnp.asarray(a)[idx], self.data)

    def points_from_indices(
        self,
        indices: ArrayLike,
        /,
        *,
        structure: SampleLayout | None = None,
    ) -> PointBatch:
        structure_in = structure or SampleLayout(((self.label,),))
        structure_out = structure_in.canonicalize(self.labels)
        axis = structure_out.axis_for(self.label)
        if axis is None:
            raise ValueError(
                f"DatasetDomain points require a sampling axis for label {self.label!r}."
            )

        idx = jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
        rows = self.input_rows(idx)

        def _to_field(v: ArrayLike):
            arr = jnp.asarray(v)
            if arr.ndim == 0:
                raise ValueError(
                    "DatasetDomain indexed rows must retain a leading sample axis."
                )
            return cx.Field(arr, dims=(axis,) + (None,) * (arr.ndim - 1))

        points = {
            self.label: jax.tree_util.tree_map(_to_field, rows),
            DATASET_INDEX_KEY: cx.Field(idx, dims=(axis,)),
        }
        return PointBatch(points=frozendict(points), structure=structure_out)

    def _same_factor_support(self, other: object, /) -> bool:
        if not isinstance(other, DatasetDomain):
            return False
        if self.coordinate_specs != other.coordinate_specs:
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


__all__ = ["DATASET_INDEX_KEY", "DatasetDomain"]
