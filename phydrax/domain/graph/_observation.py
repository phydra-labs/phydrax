#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._interpolation import (
    apply_gather_stencil,
    linear_stencil_from_indices,
    nearest_stencil_from_indices,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._evaluation import BatchEvaluator
from .._function import DomainFunction
from ._batch import GRAPH_ENTITY_INDEX_KEY, GraphBatch
from ._components import GraphComponentKind
from ._dataset import (
    GRAPH_DATASET_INDEX_KEY,
    GRAPH_ENTITY_OFFSET_KEY,
    GraphDatasetDomain,
)
from ._trajectory import (
    GRAPH_TRAJECTORY_TIME_INDEX_KEY,
    GraphTrajectoryDatasetDomain,
)


GraphTargetInterpolation = Literal["nearest", "linear"]


def _size_for_kind(graph, kind: GraphComponentKind, /) -> int:
    if kind == "nodes":
        return int(graph.num_nodes)
    if kind == "edges":
        return int(graph.num_edges)
    return int(graph.num_graphs)


def _case_arrays(values: ArrayLike | Sequence[ArrayLike], n: int, /) -> tuple[Array, ...]:
    if (
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and not hasattr(values, "shape")
    ):
        if len(values) != n:
            raise ValueError(f"Graph target values must contain {n} case arrays.")
        return tuple(jnp.asarray(value, dtype=float) for value in values)

    arr = jnp.asarray(values, dtype=float)
    if arr.ndim == 0:
        raise ValueError("Graph target values must have a case leading axis.")
    if int(arr.shape[0]) != n:
        raise ValueError(
            f"Graph target values case axis must have length {n}, got {arr.shape[0]}."
        )
    return tuple(arr[i] for i in range(n))


def _validate_graph_case_arrays(
    domain: GraphDatasetDomain,
    values: ArrayLike | Sequence[ArrayLike],
    kind: GraphComponentKind,
    /,
) -> tuple[Array, Array]:
    cases = _case_arrays(values, domain.size)
    parts: list[Array] = []
    offsets: list[int] = []
    running = 0
    trailing_shape = None
    for graph, value in zip(domain.graphs, cases, strict=True):
        expected = _size_for_kind(graph, kind)
        arr = jnp.asarray(value, dtype=float)
        if arr.ndim == 0:
            raise ValueError("Graph target case arrays must have an entity axis.")
        if int(arr.shape[0]) != expected:
            raise ValueError(
                f"Graph target case leading axis must match {kind} count {expected}; "
                f"got {arr.shape[0]}."
            )
        if trailing_shape is None:
            trailing_shape = arr.shape[1:]
        elif arr.shape[1:] != trailing_shape:
            raise ValueError("Graph target case arrays must share trailing shape.")
        offsets.append(running)
        running += expected
        parts.append(arr)
    return jnp.concatenate(parts, axis=0), jnp.asarray(offsets, dtype=jnp.int32)


def _validate_graph_trajectory_case_arrays(
    domain: GraphTrajectoryDatasetDomain,
    values: ArrayLike | Sequence[ArrayLike],
    kind: GraphComponentKind,
    /,
) -> tuple[Array, Array, Array, Array]:
    cases = _case_arrays(values, domain.size)
    parts: list[Array] = []
    offsets: list[int] = []
    lengths: list[int] = []
    entity_sizes: list[int] = []
    running = 0
    trailing_shape = None
    for graph, length, value in zip(
        domain.graphs, domain.lengths.tolist(), cases, strict=True
    ):
        expected_entities = _size_for_kind(graph, kind)
        expected_length = int(length)
        arr = jnp.asarray(value, dtype=float)
        if arr.ndim < 2:
            raise ValueError(
                "Graph trajectory target case arrays must have shape (time, entity, ...)."
            )
        if int(arr.shape[0]) != expected_length:
            raise ValueError(
                "Graph trajectory target time axis must match the case length; "
                f"expected {expected_length}, got {arr.shape[0]}."
            )
        if int(arr.shape[1]) != expected_entities:
            raise ValueError(
                f"Graph trajectory target entity axis must match {kind} count "
                f"{expected_entities}; got {arr.shape[1]}."
            )
        if trailing_shape is None:
            trailing_shape = arr.shape[2:]
        elif arr.shape[2:] != trailing_shape:
            raise ValueError("Graph trajectory target arrays must share trailing shape.")
        offsets.append(running)
        lengths.append(expected_length)
        entity_sizes.append(expected_entities)
        flattened = arr.reshape((expected_length * expected_entities,) + arr.shape[2:])
        running += int(flattened.shape[0])
        parts.append(flattened)
    return (
        jnp.concatenate(parts, axis=0),
        jnp.asarray(offsets, dtype=jnp.int32),
        jnp.asarray(lengths, dtype=jnp.int32),
        jnp.asarray(entity_sizes, dtype=jnp.int32),
    )


def _required_field(batch: GraphBatch, key: str, /) -> cx.Field:
    field = batch.points.get(key)
    if not isinstance(field, cx.Field):
        raise ValueError(f"Graph target evaluation requires GraphBatch field {key!r}.")
    return field


def _graph_axis(batch: GraphBatch, /) -> str:
    axis = batch.structure.axis_for(batch.graph_label)
    if axis is None:
        raise ValueError("GraphBatch is missing its graph sampling axis.")
    return axis


def _local_entity_indices(batch: GraphBatch, /) -> Array:
    entity = jnp.asarray(
        _required_field(batch, GRAPH_ENTITY_INDEX_KEY).data, dtype=jnp.int32
    )
    offset_field = batch.points.get(GRAPH_ENTITY_OFFSET_KEY)
    if isinstance(offset_field, cx.Field):
        offset = jnp.asarray(offset_field.data, dtype=jnp.int32)
        return entity - offset
    return entity


def _dataset_indices(batch: GraphBatch, /) -> Array:
    return jnp.asarray(
        _required_field(batch, GRAPH_DATASET_INDEX_KEY).data, dtype=jnp.int32
    )


def _field_from_target(batch: GraphBatch, value: Array, /) -> cx.Field:
    axis = _graph_axis(batch)
    return cx.Field(value, dims=(axis,) + (None,) * max(value.ndim - 1, 0))


class _GraphTargetCallable(StrictModule, BatchEvaluator, NonTrainableState):
    values: Array
    offsets: Array
    kind: GraphComponentKind

    def __init__(self, *, values: Array, offsets: Array, kind: GraphComponentKind):
        self.values = jax.lax.stop_gradient(jnp.asarray(values, dtype=float))
        self.offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.kind = kind

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("GraphTarget requires GraphBatch evaluation.")

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, GraphBatch):
            raise TypeError("GraphTarget requires GraphBatch evaluation.")
        if batch.component_kind != self.kind:
            raise ValueError(
                f"GraphTarget was built for {self.kind}, got {batch.component_kind}."
            )
        dataset_idx = _dataset_indices(batch)
        local_idx = _local_entity_indices(batch)
        flat_idx = self.offsets[dataset_idx] + local_idx
        return _field_from_target(batch, self.values[flat_idx])


class _GraphTrajectorySignalCallable(StrictModule, BatchEvaluator, NonTrainableState):
    domain: GraphTrajectoryDatasetDomain
    values: Array
    offsets: Array
    lengths: Array
    entity_sizes: Array
    kind: GraphComponentKind
    interpolation: GraphTargetInterpolation

    def __init__(
        self,
        *,
        domain: GraphTrajectoryDatasetDomain,
        values: Array,
        offsets: Array,
        lengths: Array,
        entity_sizes: Array,
        kind: GraphComponentKind,
        interpolation: GraphTargetInterpolation,
    ):
        self.domain = domain
        self.values = jax.lax.stop_gradient(jnp.asarray(values, dtype=float))
        self.offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.lengths = jnp.asarray(lengths, dtype=jnp.int32)
        self.entity_sizes = jnp.asarray(entity_sizes, dtype=jnp.int32)
        self.kind = kind
        self.interpolation = interpolation

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("GraphTrajectorySignal requires GraphBatch evaluation.")

    def _flat_index(self, case_idx: Array, time_idx: Array, local_idx: Array, /) -> Array:
        return self.offsets[case_idx] + time_idx * self.entity_sizes[case_idx] + local_idx

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, GraphBatch):
            raise TypeError("GraphTrajectorySignal requires GraphBatch evaluation.")
        if batch.component_kind != self.kind:
            raise ValueError(
                "GraphTrajectorySignal was built for "
                f"{self.kind}, got {batch.component_kind}."
            )
        case_idx = _dataset_indices(batch)
        local_idx = _local_entity_indices(batch)
        lengths = self.lengths[case_idx]

        if self.interpolation == "nearest":
            time_field = batch.points.get(GRAPH_TRAJECTORY_TIME_INDEX_KEY)
            if isinstance(time_field, cx.Field):
                time_idx = jnp.asarray(time_field.data, dtype=jnp.int32)
            else:
                t = jnp.asarray(
                    _required_field(batch, self.domain.time_label).data,
                    dtype=float,
                )
                time_idx = jnp.rint((t - self.domain.start) / self.domain.dt).astype(
                    jnp.int32
                )
            time_idx = jnp.clip(time_idx, 0, lengths - 1)
            stencil = nearest_stencil_from_indices(
                self._flat_index(case_idx, time_idx, local_idx),
                source_size=int(self.values.shape[0]),
            )
            return _field_from_target(
                batch,
                apply_gather_stencil(self.values, stencil).values,
            )

        if self.interpolation != "linear":
            raise ValueError(
                "GraphTrajectorySignal interpolation must be 'nearest' or 'linear'."
            )

        t = jnp.asarray(
            _required_field(batch, self.domain.time_label).data,
            dtype=float,
        )
        tau = (t - self.domain.start) / self.domain.dt
        lo = jnp.floor(tau).astype(jnp.int32)
        lo = jnp.clip(lo, 0, lengths - 1)
        hi = jnp.clip(lo + 1, 0, lengths - 1)
        fraction = jnp.clip(tau - lo.astype(float), 0.0, 1.0)
        stencil = linear_stencil_from_indices(
            self._flat_index(case_idx, lo, local_idx),
            self._flat_index(case_idx, hi, local_idx),
            fraction,
            source_size=int(self.values.shape[0]),
        )
        return _field_from_target(
            batch,
            apply_gather_stencil(self.values, stencil).values,
        )


def GraphTarget(
    domain: GraphDatasetDomain,
    values: ArrayLike | Sequence[ArrayLike],
    /,
    *,
    component_kind: GraphComponentKind = "nodes",
) -> DomainFunction:
    """Expose fixed graph-family targets as a `DomainFunction`.

    `values` is aligned by graph case and entity kind. It can be one padded array
    with leading graph-case axis or a sequence of per-case arrays whose leading
    length matches each case's selected node, edge, or global count. The returned
    function reads graph metadata from a `GraphBatch` and returns targets aligned
    to the sampled entities.
    """
    if not isinstance(domain, GraphDatasetDomain):
        raise TypeError("GraphTarget requires a GraphDatasetDomain.")
    values_flat, offsets = _validate_graph_case_arrays(domain, values, component_kind)
    return DomainFunction(
        domain=domain,
        deps=(domain.label,),
        func=_GraphTargetCallable(
            values=values_flat,
            offsets=offsets,
            kind=component_kind,
        ),
        metadata={},
    )


def GraphTrajectorySignal(
    domain: GraphTrajectoryDatasetDomain,
    values: ArrayLike | Sequence[ArrayLike],
    /,
    *,
    component_kind: GraphComponentKind = "nodes",
    interpolation: GraphTargetInterpolation = "nearest",
) -> DomainFunction:
    """Expose fixed graph-trajectory data as a `DomainFunction` over `(graph, t)`.

    `values` is aligned by graph case, local time index, and entity kind. Use
    `interpolation="nearest"` for observation lookup and `"linear"` for continuous
    time interpolation between neighboring stored frames.
    """
    if not isinstance(domain, GraphTrajectoryDatasetDomain):
        raise TypeError("GraphTrajectorySignal requires a GraphTrajectoryDatasetDomain.")
    interpolation_str = str(interpolation)
    if interpolation_str not in ("nearest", "linear"):
        raise ValueError("interpolation must be 'nearest' or 'linear'.")
    interpolation_value: GraphTargetInterpolation = (
        "linear" if interpolation_str == "linear" else "nearest"
    )
    values_flat, offsets, lengths, entity_sizes = _validate_graph_trajectory_case_arrays(
        domain,
        values,
        component_kind,
    )
    return DomainFunction(
        domain=domain,
        deps=domain.labels,
        func=_GraphTrajectorySignalCallable(
            domain=domain,
            values=values_flat,
            offsets=offsets,
            lengths=lengths,
            entity_sizes=entity_sizes,
            kind=component_kind,
            interpolation=interpolation_value,
        ),
        metadata={},
    )
