#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import coordax as cx
import equinox as eqx
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
GraphClassificationTargetEncoding = Literal["hard", "soft"]


def _classification_array(
    name: str,
    value: ArrayLike,
    encoding: GraphClassificationTargetEncoding,
    /,
    *,
    require_boolean: bool = False,
) -> Array:
    if encoding not in ("hard", "soft"):
        raise ValueError("target_encoding must be 'hard' or 'soft'.")
    try:
        arr = jnp.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must contain JSON-compatible numeric values.") from error
    if jnp.iscomplexobj(arr):
        raise TypeError(f"{name} must be real.")
    if require_boolean:
        if arr.size == 0:
            return arr.astype(bool)
        if arr.dtype != jnp.bool_:
            raise TypeError(f"{name} must contain Boolean values.")
        return arr
    if encoding == "hard":
        if arr.size == 0:
            return arr
        if not (
            jnp.issubdtype(arr.dtype, jnp.integer) or jnp.issubdtype(arr.dtype, jnp.bool_)
        ):
            raise TypeError(f"{name} hard values must have integer or Boolean dtype.")
        return arr
    return arr.astype(jnp.result_type(arr, 0.0))


def _classification_case_arrays(
    values: ArrayLike | Sequence[ArrayLike],
    n: int,
    encoding: GraphClassificationTargetEncoding,
    /,
    *,
    require_boolean: bool = False,
) -> tuple[Array, ...]:
    if (
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and not hasattr(values, "shape")
    ):
        if len(values) != n:
            raise ValueError(f"Graph classification values must contain {n} case arrays.")
        cases = tuple(
            _classification_array(
                "Graph classification values",
                value,
                encoding,
                require_boolean=require_boolean,
            )
            for value in values
        )
        if encoding == "hard" and not require_boolean:
            nonempty = tuple(arr for arr in cases if arr.size != 0)
            dtype = (
                jnp.result_type(*(arr.dtype for arr in nonempty))
                if nonempty
                else jnp.dtype(jnp.int32)
            )
            cases = tuple(arr if arr.size != 0 else arr.astype(dtype) for arr in cases)
        return cases

    arr = _classification_array(
        "Graph classification values",
        values,
        encoding,
        require_boolean=require_boolean,
    )
    if arr.ndim == 0:
        raise ValueError("Graph classification values must have a case leading axis.")
    if int(arr.shape[0]) != n:
        raise ValueError(
            "Graph classification values case axis must have length "
            f"{n}, got {arr.shape[0]}."
        )
    return tuple(arr[i] for i in range(n))


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


def _validate_graph_classification_case_arrays(
    domain: GraphDatasetDomain,
    values: ArrayLike | Sequence[ArrayLike],
    kind: GraphComponentKind,
    encoding: GraphClassificationTargetEncoding,
    /,
    *,
    require_boolean: bool = False,
) -> tuple[Array, Array]:
    if kind not in ("nodes", "edges", "globals"):
        raise ValueError("component_kind must be 'nodes', 'edges', or 'globals'.")
    cases = _classification_case_arrays(
        values,
        domain.size,
        encoding,
        require_boolean=require_boolean,
    )
    parts: list[Array] = []
    offsets: list[int] = []
    running = 0
    trailing_shape = None
    for graph, value in zip(domain.graphs, cases, strict=True):
        expected = _size_for_kind(graph, kind)
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            if expected != 1:
                raise ValueError(
                    "Scalar graph classification cases require exactly one entity."
                )
            arr = arr.reshape((1,))
        if int(arr.shape[0]) != expected:
            raise ValueError(
                "Graph classification case leading axis must match "
                f"{kind} count {expected}; got {arr.shape[0]}."
            )
        if trailing_shape is None:
            trailing_shape = arr.shape[1:]
        elif arr.shape[1:] != trailing_shape:
            raise ValueError(
                "Graph classification case arrays must share trailing shape."
            )
        offsets.append(running)
        running += expected
        parts.append(arr)
    return jnp.concatenate(parts, axis=0), jnp.asarray(offsets, dtype=jnp.int32)


def _validate_graph_trajectory_classification_case_arrays(
    domain: GraphTrajectoryDatasetDomain,
    values: ArrayLike | Sequence[ArrayLike],
    kind: GraphComponentKind,
    encoding: GraphClassificationTargetEncoding,
    /,
    *,
    require_boolean: bool = False,
) -> tuple[Array, Array, Array, Array]:
    if kind not in ("nodes", "edges", "globals"):
        raise ValueError("component_kind must be 'nodes', 'edges', or 'globals'.")
    cases = _classification_case_arrays(
        values,
        domain.size,
        encoding,
        require_boolean=require_boolean,
    )
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
        arr = jnp.asarray(value)
        if arr.ndim == 1 and expected_entities == 1:
            arr = arr[:, None]
        if arr.ndim < 2:
            raise ValueError(
                "Graph trajectory classification case arrays must have shape "
                "(time, entity, ...)."
            )
        if int(arr.shape[0]) != expected_length:
            raise ValueError(
                "Graph trajectory classification target time axis must match the "
                f"case length; expected {expected_length}, got {arr.shape[0]}."
            )
        if int(arr.shape[1]) != expected_entities:
            raise ValueError(
                "Graph trajectory classification target entity axis must match "
                f"{kind} count {expected_entities}; got {arr.shape[1]}."
            )
        if trailing_shape is None:
            trailing_shape = arr.shape[2:]
        elif arr.shape[2:] != trailing_shape:
            raise ValueError(
                "Graph trajectory classification target arrays must share trailing shape."
            )
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


class _GraphClassificationTargetCallable(StrictModule, BatchEvaluator, NonTrainableState):
    values: Array
    offsets: Array
    kind: GraphComponentKind

    def __init__(self, *, values: Array, offsets: Array, kind: GraphComponentKind):
        self.values = jax.lax.stop_gradient(jnp.asarray(values))
        self.offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.kind = kind

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("GraphClassificationTarget requires GraphBatch evaluation.")

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
            raise TypeError("GraphClassificationTarget requires GraphBatch evaluation.")
        if batch.component_kind != self.kind:
            raise ValueError(
                "GraphClassificationTarget was built for "
                f"{self.kind}, got {batch.component_kind}."
            )
        dataset_idx = _dataset_indices(batch)
        local_idx = _local_entity_indices(batch)
        flat_idx = self.offsets[dataset_idx] + local_idx
        return _field_from_target(batch, self.values[flat_idx])


class _GraphTrajectoryClassificationSignalCallable(
    StrictModule, BatchEvaluator, NonTrainableState
):
    domain: GraphTrajectoryDatasetDomain
    values: Array
    offsets: Array
    lengths: Array
    entity_sizes: Array
    kind: GraphComponentKind
    interpolation: GraphTargetInterpolation
    logical_interpolation: bool = eqx.field(static=True)

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
        logical_interpolation: bool = False,
    ):
        self.domain = domain
        self.values = jax.lax.stop_gradient(jnp.asarray(values))
        self.offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.lengths = jnp.asarray(lengths, dtype=jnp.int32)
        self.entity_sizes = jnp.asarray(entity_sizes, dtype=jnp.int32)
        self.kind = kind
        self.interpolation = interpolation
        self.logical_interpolation = bool(logical_interpolation)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError(
            "GraphTrajectoryClassificationSignal requires GraphBatch evaluation."
        )

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
            raise TypeError(
                "GraphTrajectoryClassificationSignal requires GraphBatch evaluation."
            )
        if batch.component_kind != self.kind:
            raise ValueError(
                "GraphTrajectoryClassificationSignal was built for "
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
            return _field_from_target(
                batch,
                self.values[self._flat_index(case_idx, time_idx, local_idx)],
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
        lower_index = self._flat_index(case_idx, lo, local_idx)
        upper_index = self._flat_index(case_idx, hi, local_idx)
        if self.logical_interpolation:
            lower_value = self.values[lower_index]
            upper_value = self.values[upper_index]
            event_axes = (1,) * (lower_value.ndim - fraction.ndim)
            fraction_ = fraction.reshape(fraction.shape + event_axes)
            interpolated = ((fraction_ >= 1.0) | lower_value) & (
                (fraction_ <= 0.0) | upper_value
            )
            return _field_from_target(batch, interpolated)
        stencil = linear_stencil_from_indices(
            lower_index,
            upper_index,
            fraction,
            source_size=int(self.values.shape[0]),
        )
        return _field_from_target(
            batch,
            apply_gather_stencil(self.values, stencil).values,
        )


def _graph_classification_target(
    domain: GraphDatasetDomain,
    values: ArrayLike | Sequence[ArrayLike],
    /,
    *,
    component_kind: GraphComponentKind,
    target_encoding: GraphClassificationTargetEncoding,
    require_boolean: bool,
) -> DomainFunction:
    if not isinstance(domain, GraphDatasetDomain):
        raise TypeError("GraphClassificationTarget requires a GraphDatasetDomain.")
    values_flat, offsets = _validate_graph_classification_case_arrays(
        domain,
        values,
        component_kind,
        target_encoding,
        require_boolean=require_boolean,
    )
    return DomainFunction(
        domain=domain,
        deps=(domain.label,),
        func=_GraphClassificationTargetCallable(
            values=values_flat,
            offsets=offsets,
            kind=component_kind,
        ),
        metadata={},
    )


def GraphClassificationTarget(
    domain: GraphDatasetDomain,
    values: ArrayLike | Sequence[ArrayLike],
    /,
    *,
    component_kind: GraphComponentKind = "nodes",
    target_encoding: GraphClassificationTargetEncoding = "hard",
) -> DomainFunction:
    """Expose dtype-preserving classification targets on graph entities.

    Hard targets retain their integer or Boolean dtype and are gathered directly
    by graph case and local entity index. Soft targets are explicitly selected
    with ``target_encoding="soft"`` and converted to an inexact dtype.
    """
    return _graph_classification_target(
        domain,
        values,
        component_kind=component_kind,
        target_encoding=target_encoding,
        require_boolean=False,
    )


def _graph_trajectory_classification_signal(
    domain: GraphTrajectoryDatasetDomain,
    values: ArrayLike | Sequence[ArrayLike],
    /,
    *,
    component_kind: GraphComponentKind,
    interpolation: GraphTargetInterpolation,
    target_encoding: GraphClassificationTargetEncoding,
    require_boolean: bool,
) -> DomainFunction:
    if not isinstance(domain, GraphTrajectoryDatasetDomain):
        raise TypeError(
            "GraphTrajectoryClassificationSignal requires a GraphTrajectoryDatasetDomain."
        )
    if interpolation not in ("nearest", "linear"):
        raise ValueError("interpolation must be 'nearest' or 'linear'.")
    if target_encoding == "hard" and interpolation != "nearest" and not require_boolean:
        raise ValueError(
            "Hard graph trajectory classification targets require nearest interpolation."
        )
    values_flat, offsets, lengths, entity_sizes = (
        _validate_graph_trajectory_classification_case_arrays(
            domain,
            values,
            component_kind,
            target_encoding,
            require_boolean=require_boolean,
        )
    )
    return DomainFunction(
        domain=domain,
        deps=domain.labels,
        func=_GraphTrajectoryClassificationSignalCallable(
            domain=domain,
            values=values_flat,
            offsets=offsets,
            lengths=lengths,
            entity_sizes=entity_sizes,
            kind=component_kind,
            interpolation=interpolation,
            logical_interpolation=require_boolean,
        ),
        metadata={},
    )


def GraphTrajectoryClassificationSignal(
    domain: GraphTrajectoryDatasetDomain,
    values: ArrayLike | Sequence[ArrayLike],
    /,
    *,
    component_kind: GraphComponentKind = "nodes",
    interpolation: GraphTargetInterpolation = "nearest",
    target_encoding: GraphClassificationTargetEncoding = "hard",
) -> DomainFunction:
    """Expose dtype-safe graph-trajectory classification observations.

    Hard labels support nearest observation lookup only. Linear interpolation is
    available only after explicitly declaring ``target_encoding="soft"``.
    """
    return _graph_trajectory_classification_signal(
        domain,
        values,
        component_kind=component_kind,
        interpolation=interpolation,
        target_encoding=target_encoding,
        require_boolean=False,
    )
