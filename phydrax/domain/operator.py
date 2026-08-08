#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reversible adapters between PhydraX domain batches and operator batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._frozendict import frozendict
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..graph._operator_topology import (
    broadcast_operator_topology,
    OperatorTopology,
    OperatorTopologyEntity,
    OperatorTopologyKind,
    OperatorTopologySite,
)
from ..nn.operator.data import (
    FunctionSamples,
    OperatorAxis,
    OperatorBatch,
    OperatorPrediction,
    tensor_product,
)
from ..nn.operator.protocols import OperatorModel


OperatorDomainKind = Literal[
    "points",
    "coord_separable",
    "graph",
    "simplicial",
    "ragged_series",
    "trajectory",
]


class OperatorDomainLayout(StrictModule, NonTrainableState):
    """How one operator query is restored to a domain ``coordax.Field``."""

    gather_indices: Array | None
    query_name: str = eqx.field(static=True)
    dims: tuple[str | None, ...] = eqx.field(static=True)
    leading_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        query_name: str,
        dims: Sequence[str | None],
        leading_shape: Sequence[int],
        /,
        *,
        gather_indices: Any | None = None,
    ):
        name = str(query_name)
        if not name:
            raise ValueError("Operator domain query names must be non-empty.")
        shape = tuple(int(size) for size in leading_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("Operator domain layout dimensions must be positive.")
        dimensions = tuple(dims)
        indices = (
            None
            if gather_indices is None
            else jnp.asarray(gather_indices, dtype=jnp.int32)
        )
        if indices is not None and indices.ndim == 0:
            raise ValueError("Operator domain gather indices must have positive rank.")
        expected_dims = indices.ndim if indices is not None else len(shape)
        if len(dimensions) != expected_dims:
            raise ValueError(
                f"Operator domain layout requires {expected_dims} output dim(s); "
                f"got {dimensions}."
            )
        if indices is not None:
            total = prod(shape)
            if bool(jnp.any(indices < 0)) or bool(jnp.any(indices >= total)):
                raise ValueError("Operator domain gather indices are out of range.")
        self.gather_indices = indices
        self.query_name = name
        self.dims = dimensions
        self.leading_shape = shape

    def restore(self, values: Any, /) -> cx.Field:
        """Restore one channel-last operator array to its domain field layout."""
        array = jnp.asarray(values)
        leading_rank = len(self.leading_shape)
        if (
            array.ndim < leading_rank
            or tuple(array.shape[:leading_rank]) != self.leading_shape
        ):
            raise ValueError(
                f"Operator output must start with shape {self.leading_shape}; "
                f"got {array.shape}."
            )
        trailing_shape = tuple(int(size) for size in array.shape[leading_rank:])
        if self.gather_indices is not None:
            array = array.reshape((prod(self.leading_shape),) + trailing_shape)
            array = array[self.gather_indices]
        trailing_dims = (None,) * len(trailing_shape)
        return cx.Field(array, dims=self.dims + trailing_dims)


class OperatorDomainView(StrictModule, NonTrainableState):
    """Canonical operator batch plus reversible domain output layouts."""

    batch: OperatorBatch
    layouts: frozendict[str, OperatorDomainLayout]
    kind: OperatorDomainKind = eqx.field(static=True)

    def __init__(
        self,
        batch: OperatorBatch,
        layouts: Mapping[str, OperatorDomainLayout],
        /,
        *,
        kind: OperatorDomainKind,
    ):
        if not isinstance(batch, OperatorBatch):
            raise TypeError("OperatorDomainView requires an OperatorBatch.")
        if kind not in (
            "points",
            "coord_separable",
            "graph",
            "simplicial",
            "ragged_series",
            "trajectory",
        ):
            raise ValueError("Unknown operator domain view kind.")
        frozen = frozendict(layouts)
        if tuple(frozen) != tuple(batch.queries):
            raise ValueError(
                "Operator domain layouts must cover every named query in batch order."
            )
        for name, layout in frozen.items():
            if not isinstance(layout, OperatorDomainLayout):
                raise TypeError(
                    "Operator domain layouts must be OperatorDomainLayout values."
                )
            if layout.query_name != name:
                raise ValueError(
                    "Operator domain layout names must match their mapping keys."
                )
            expected = batch.case_shape + batch.query(name).sample_shape
            if layout.leading_shape != expected:
                raise ValueError(
                    f"Layout {name!r} leading shape {layout.leading_shape} does not match "
                    f"operator geometry {expected}."
                )
        self.batch = batch
        self.layouts = frozen
        self.kind = kind

    def restore_field(
        self,
        prediction: OperatorPrediction,
        field_name: str,
        /,
    ) -> cx.Field:
        """Restore one named prediction field to its original domain axes."""
        if not isinstance(prediction, OperatorPrediction):
            raise TypeError("prediction must be an OperatorPrediction.")
        field = prediction.field(field_name)
        if field.query_name not in self.layouts:
            raise KeyError(
                f"Prediction field {field_name!r} targets unknown query "
                f"{field.query_name!r}."
            )
        return self.layouts[field.query_name].restore(field.values)

    def restore(
        self,
        prediction: OperatorPrediction,
        /,
    ) -> frozendict[str, cx.Field]:
        """Restore every named prediction field without dropping query identity."""
        return frozendict(
            {name: self.restore_field(prediction, name) for name in prediction.fields}
        )

    def compatibility(self, model: Any, /, **kwargs: Any):
        """Return the configured model's capability report for this view."""
        if not isinstance(model, OperatorModel):
            raise TypeError("Operator domain preflight requires a neural operator model.")
        return model.operator_contract.validate(self.batch, **kwargs)

    def require_compatible(self, model: Any, /, **kwargs: Any) -> None:
        """Fail before execution when a model cannot consume this domain view."""
        self.compatibility(model, **kwargs).require_runtime()


def _field(batch: Mapping[str, Any], label: str, /) -> cx.Field:
    value = batch[str(label)]
    if not isinstance(value, cx.Field):
        raise TypeError(
            f"Domain label {label!r} must resolve to one coordax.Field; "
            "use a specialized PyTree adapter for structured payloads."
        )
    return value


def _selection(
    selections: Mapping[str, Any] | None,
    name: str,
    /,
) -> Any | None:
    if selections is None or name not in selections:
        return None
    value = selections[name]
    return value.data if isinstance(value, cx.Field) else value


def _point_geometry(
    field: cx.Field,
    case_axes: tuple[str, ...],
    /,
    *,
    case_shape: tuple[int, ...] | None = None,
) -> tuple[Array, tuple[int, ...], str]:
    dims = tuple(field.dims)
    named = tuple(dim for dim in dims if dim is not None)
    sample_axes = tuple(dim for dim in named if dim not in case_axes)
    if len(sample_axes) != 1:
        raise ValueError(
            "PointBatch operator coordinates require exactly one named sample axis."
        )
    sample_axis = sample_axes[0]
    unnamed_positions = tuple(index for index, dim in enumerate(dims) if dim is None)
    if len(unnamed_positions) > 1:
        raise ValueError("Point coordinates may have only one coordinate-channel axis.")
    named_positions = {dim: index for index, dim in enumerate(dims) if dim is not None}
    permutation = (
        tuple(named_positions[axis] for axis in case_axes if axis in named_positions)
        + (named_positions[sample_axis],)
        + unnamed_positions
    )
    array = jnp.asarray(field.data, dtype=float)
    if permutation != tuple(range(array.ndim)):
        array = jnp.transpose(array, permutation)
    present_cases = tuple(
        int(field.data.shape[named_positions[axis]])
        for axis in case_axes
        if axis in named_positions
    )
    target_cases = (
        present_cases if case_shape is None else tuple(int(size) for size in case_shape)
    )
    expanded_cases = tuple(
        int(field.data.shape[named_positions[axis]]) if axis in named_positions else 1
        for axis in case_axes
    )
    sample_size = int(field.data.shape[named_positions[sample_axis]])
    coordinate_dimension = 1 if not unnamed_positions else int(array.shape[-1])
    if coordinate_dimension <= 0:
        raise ValueError("Point coordinates require a positive coordinate dimension.")
    coordinates = array.reshape(expanded_cases + (sample_size, coordinate_dimension))
    coordinates = jnp.broadcast_to(
        coordinates,
        target_cases + (sample_size, coordinate_dimension),
    )
    return coordinates, target_cases, sample_axis


def operator_domain_view_from_points(
    batch: Any,
    /,
    *,
    inputs: Mapping[str, str],
    queries: Mapping[str, str],
    input_coordinates: Mapping[str, str] | None = None,
    quadrature: Mapping[str, Any] | None = None,
    masks: Mapping[str, Any] | None = None,
    case_axes: Sequence[str] = (),
) -> OperatorDomainView:
    """Adapt a ``PointBatch`` while retaining reversible named domain axes."""

    from phydrax.domain import PointBatch

    if not isinstance(batch, PointBatch):
        raise TypeError("operator_domain_view_from_points requires a PointBatch.")
    cases = tuple(str(axis) for axis in case_axes)
    if len(set(cases)) != len(cases):
        raise ValueError("case_axes must be unique.")
    if not inputs or not queries:
        raise ValueError("Point-domain operator views require inputs and queries.")
    case_shape = tuple(_axis_size_from_points(batch.points, axis) for axis in cases)

    query_samples: dict[str, FunctionSamples] = {}
    layouts: dict[str, OperatorDomainLayout] = {}
    query_geometry_by_label: dict[str, FunctionSamples] = {}
    for query_name, label in queries.items():
        coordinate_field = _field(batch, label)
        coordinates, query_cases, sample_axis = _point_geometry(
            coordinate_field,
            cases,
            case_shape=case_shape,
        )
        query = FunctionSamples(
            values=None,
            coordinates=coordinates,
            quadrature_weights=_selection(quadrature, query_name),
            mask=_selection(masks, query_name),
        )
        query_samples[str(query_name)] = query
        query_geometry_by_label[str(label)] = query
        layouts[str(query_name)] = OperatorDomainLayout(
            str(query_name),
            cases + (sample_axis,),
            query_cases + query.sample_shape,
        )

    sources: dict[str, FunctionSamples] = {}
    for source_name, label in inputs.items():
        value_field = _field(batch, label)
        coordinate_label = (
            None if input_coordinates is None else input_coordinates.get(source_name)
        )
        if coordinate_label is None:
            sources[str(source_name)] = FunctionSamples(values=value_field.data)
            continue
        coordinate_name = str(coordinate_label)
        if coordinate_name in query_geometry_by_label:
            geometry = query_geometry_by_label[coordinate_name]
        else:
            coordinates, source_cases, _ = _point_geometry(
                _field(batch, coordinate_name),
                cases,
                case_shape=case_shape,
            )
            if source_cases != case_shape:
                raise ValueError(
                    "Source and query point geometries must share case axes."
                )
            geometry = FunctionSamples(
                values=None,
                coordinates=coordinates,
                quadrature_weights=_selection(quadrature, source_name),
                mask=_selection(masks, source_name),
            )
        sources[str(source_name)] = FunctionSamples(
            values=value_field.data,
            coordinates=geometry.coordinates,
            quadrature_weights=geometry.quadrature_weights,
            mask=geometry.mask,
            topology=geometry.topology,
        )

    operator_batch = OperatorBatch(
        inputs=sources,
        queries=query_samples,
        case_axes=cases,
        case_shape=case_shape,
    )
    return OperatorDomainView(operator_batch, layouts, kind="points")


def _axis_from_coord_batch(batch: Any, name: str, /) -> OperatorAxis:
    discretization = batch.axis_discretization_by_axis.get(name)
    if discretization is not None:
        return OperatorAxis.from_discretization(name, discretization)
    for value in batch.points.values():
        if not isinstance(value, tuple):
            continue
        for field in value:
            if isinstance(field, cx.Field) and field.dims == (name,):
                return OperatorAxis(name, jnp.asarray(field.data, dtype=float))
    raise KeyError(f"Cannot locate coordinate values for axis {name!r}.")


def _coord_query(batch: Any, labels: Sequence[str], /) -> FunctionSamples:
    axes: list[OperatorAxis] = []
    label_axes: list[tuple[str, ...]] = []
    for label in labels:
        if label not in batch.coord_axes_by_label:
            raise KeyError(f"Unknown coord-separable label {label!r}.")
        names = batch.coord_axes_by_label[label]
        label_axes.append(names)
        axes.extend(_axis_from_coord_batch(batch, name) for name in names)
    shape = tuple(axis.size for axis in axes)
    mask = jnp.ones(shape, dtype=bool)
    weights = tensor_product(
        tuple(
            jnp.ones_like(axis.nodes, dtype=float)
            if axis.quadrature_weights is None
            else axis.quadrature_weights
            for axis in axes
        )
    )
    offset = 0
    for label, names in zip(labels, label_axes, strict=True):
        count = len(names)
        local_shape = shape[offset : offset + count]
        prefix = (1,) * offset
        suffix = (1,) * (len(shape) - offset - count)
        local_mask = jnp.asarray(batch.coord_mask_by_label[label].data, dtype=bool)
        mask = mask & local_mask.reshape(prefix + local_shape + suffix)
        correction = batch.coord_geometry_weight_by_label.get(label)
        if correction is not None:
            weights = weights * jnp.asarray(correction.data, dtype=float).reshape(
                prefix + local_shape + suffix
            )
        offset += count
    return FunctionSamples(
        values=None,
        axes=tuple(axes),
        quadrature_weights=weights,
        mask=mask,
    )


def operator_domain_view_from_grid(
    batch: Any,
    /,
    *,
    inputs: Mapping[str, str],
    queries: Mapping[str, Sequence[str]],
    input_queries: Mapping[str, str] | None = None,
) -> OperatorDomainView:
    """Adapt a ``GridBatch`` without collapsing tensor-product axes."""
    from phydrax.domain import GridBatch

    if not isinstance(batch, GridBatch):
        raise TypeError("operator_domain_view_from_grid requires a GridBatch.")
    if not inputs or not queries:
        raise ValueError("Grid operator views require inputs and queries.")
    case_axes = tuple(batch.dense_structure.axis_names or ())
    query_samples = {
        str(name): _coord_query(batch, tuple(str(label) for label in labels))
        for name, labels in queries.items()
    }
    layouts = {
        name: OperatorDomainLayout(
            name,
            case_axes + sample.axis_names,
            tuple(
                int(_field(batch, next(iter(inputs.values()))).data.shape[index])
                for index in range(len(case_axes))
            )
            + sample.sample_shape,
        )
        for name, sample in query_samples.items()
    }
    case_shape = next(iter(layouts.values())).leading_shape[: len(case_axes)]

    sources: dict[str, FunctionSamples] = {}
    for source_name, label in inputs.items():
        values = _field(batch, label).data
        query_name = None if input_queries is None else input_queries.get(source_name)
        if query_name is None and len(query_samples) == 1:
            candidate_name, candidate = next(iter(query_samples.items()))
            sample_rank = len(candidate.sample_shape)
            value_sample_shape = tuple(
                int(size)
                for size in values.shape[len(case_shape) : len(case_shape) + sample_rank]
            )
            if value_sample_shape == candidate.sample_shape:
                query_name = candidate_name
        if query_name is None:
            sources[str(source_name)] = FunctionSamples(values=values)
            continue
        if query_name not in query_samples:
            raise KeyError(f"Unknown source geometry query {query_name!r}.")
        geometry = query_samples[query_name]
        sources[str(source_name)] = FunctionSamples(
            values=values,
            axes=geometry.axes,
            quadrature_weights=geometry.quadrature_weights,
            mask=geometry.mask,
        )
    operator_batch = OperatorBatch(
        inputs=sources,
        queries=query_samples,
        case_axes=case_axes,
        case_shape=case_shape,
    )
    return OperatorDomainView(
        operator_batch,
        layouts,
        kind="coord_separable",
    )


def _field_leaves(value: Any, /) -> tuple[cx.Field, ...]:
    leaves = tuple(
        jax.tree_util.tree_leaves(
            value,
            is_leaf=lambda item: isinstance(item, cx.Field),
        )
    )
    if not leaves or any(not isinstance(leaf, cx.Field) for leaf in leaves):
        raise TypeError(
            "Operator domain payloads must be PyTrees of coordax.Field leaves."
        )
    return leaves


def _payload_feature_array(
    value: Any,
    leading_rank: int,
    /,
    *,
    name: str,
) -> Array:
    rank = int(leading_rank)
    if rank <= 0:
        raise ValueError("Payload feature arrays require a positive leading rank.")
    parts: list[Array] = []
    leading_shape: tuple[int, ...] | None = None
    for field in _field_leaves(value):
        array = jnp.asarray(field.data)
        if array.ndim < rank:
            raise ValueError(f"{name} leaves require at least {rank} leading dimensions.")
        current = tuple(int(size) for size in array.shape[:rank])
        if leading_shape is None:
            leading_shape = current
        elif current != leading_shape:
            raise ValueError(f"{name} leaves must share their leading shape.")
        parts.append(array.reshape(current + (-1,)))
    return jnp.concatenate(tuple(parts), axis=-1)


def _graph_payload_array(value: Any, entity_axis: str, /) -> Array:
    parts: list[Array] = []
    entity_size: int | None = None
    for field in _field_leaves(value):
        dims = tuple(field.dims)
        if not dims or dims[0] != entity_axis or any(dim is not None for dim in dims[1:]):
            raise ValueError(
                "Graph payload fields must use the graph entity axis followed only "
                f"by unnamed feature axes; got {dims}."
            )
        array = jnp.asarray(field.data)
        if entity_size is None:
            entity_size = int(array.shape[0])
        elif int(array.shape[0]) != entity_size:
            raise ValueError("Graph payload leaves must share one entity-axis size.")
        parts.append(array.reshape((int(array.shape[0]), -1)))
    return jnp.concatenate(tuple(parts), axis=-1)


def _axis_size_from_points(points: Mapping[str, Any], axis: str, /) -> int:
    sizes: set[int] = set()
    for value in points.values():
        for field in _field_leaves(value):
            if axis in field.dims:
                sizes.add(int(field.data.shape[field.dims.index(axis)]))
    if len(sizes) != 1:
        raise ValueError(f"Cannot infer one size for domain axis {axis!r}.")
    return next(iter(sizes))


def _case_value_array(
    value: Any,
    case_axes: tuple[str, ...],
    case_shape: tuple[int, ...],
    /,
) -> Array:
    parts: list[Array] = []
    for field in _field_leaves(value):
        dims = tuple(field.dims)
        named = tuple(dim for dim in dims if dim is not None)
        if len(set(named)) != len(named) or any(dim not in case_axes for dim in named):
            raise ValueError(
                f"Case input field dims must be a subset of {case_axes}; got {dims}."
            )
        named_positions = {
            dim: index for index, dim in enumerate(dims) if dim is not None
        }
        unnamed_positions = tuple(index for index, dim in enumerate(dims) if dim is None)
        permutation = (
            tuple(named_positions[axis] for axis in case_axes if axis in named_positions)
            + unnamed_positions
        )
        array = jnp.asarray(field.data)
        if permutation != tuple(range(array.ndim)):
            array = jnp.transpose(array, permutation)
        present_shape = {
            axis: int(field.data.shape[named_positions[axis]]) for axis in named
        }
        trailing_shape = tuple(
            int(array.shape[index]) for index in range(len(named), array.ndim)
        )
        expanded_shape = (
            tuple(present_shape.get(axis, 1) for axis in case_axes) + trailing_shape
        )
        array = array.reshape(expanded_shape)
        array = jnp.broadcast_to(array, case_shape + trailing_shape)
        parts.append(array.reshape(case_shape + (-1,)))
    return jnp.concatenate(tuple(parts), axis=-1)


def _case_function_samples(
    value: Any,
    case_axes: tuple[str, ...],
    case_shape: tuple[int, ...],
    /,
) -> FunctionSamples:
    channels = _case_value_array(value, case_axes, case_shape)
    values: Array = channels[..., 0] if int(channels.shape[-1]) == 1 else channels
    values = jnp.expand_dims(values, axis=len(case_shape))
    return FunctionSamples(
        values=values,
        coordinates=jnp.zeros(case_shape + (1, 1), dtype=float),
        quadrature_weights=jnp.ones(case_shape + (1,), dtype=float),
        mask=jnp.ones(case_shape + (1,), dtype=bool),
    )


def operator_domain_view_from_ragged_series(
    batch: Any,
    label: str,
    /,
    *,
    input_name: str | None = None,
    query_name: str = "query",
) -> OperatorDomainView:
    """Adapt padded or subsampled ragged series with exact masks and weights."""

    from phydrax.domain import PointBatch

    if not isinstance(batch, PointBatch):
        raise TypeError("Ragged-series operator views require a PointBatch.")
    label_name = str(label)
    if label_name not in batch.points:
        raise KeyError(f"Unknown ragged-series label {label_name!r}.")
    payload = batch.points[label_name]
    if not isinstance(payload, Mapping):
        raise TypeError("Ragged-series operator payloads must be mappings.")
    required = ("series", "time", "mask", "length")
    missing = tuple(key for key in required if key not in payload)
    if missing:
        raise KeyError(f"Ragged-series payload is missing fields {missing}.")
    axis = batch.structure.axis_for(label_name)
    if axis is None:
        raise ValueError("Ragged-series labels require one sampled case axis.")

    series = _payload_feature_array(
        payload["series"],
        2,
        name="Ragged-series values",
    )
    case_count, width = (int(series.shape[0]), int(series.shape[1]))
    if "static" in payload:
        static = _payload_feature_array(
            payload["static"],
            1,
            name="Ragged-series static values",
        )
        static = jnp.broadcast_to(
            static[:, None, :],
            (case_count, width, int(static.shape[-1])),
        )
        series = jnp.concatenate((series, static), axis=-1)
    time_field = payload["time"]
    mask_field = payload["mask"]
    if not isinstance(time_field, cx.Field) or not isinstance(mask_field, cx.Field):
        raise TypeError("Ragged-series time and mask values must be coordax.Field.")
    times = jnp.asarray(time_field.data, dtype=float)
    mask = jnp.asarray(mask_field.data, dtype=bool)
    if times.shape != (case_count, width) or mask.shape != times.shape:
        raise ValueError("Ragged-series time and mask shapes must match series cases.")
    weights = jnp.ones((case_count, width), dtype=float)
    if "sample_scale" in payload:
        scale_field = payload["sample_scale"]
        if not isinstance(scale_field, cx.Field):
            raise TypeError("Ragged-series sample_scale must be a coordax.Field.")
        scale = jnp.asarray(scale_field.data, dtype=float)
        if scale.shape != (case_count,):
            raise ValueError("Ragged-series sample_scale must have one value per case.")
        weights = jnp.broadcast_to(scale[:, None], (case_count, width))

    coordinates = times[..., None]
    values: Array = series[..., 0] if int(series.shape[-1]) == 1 else series
    source = FunctionSamples(
        values=values,
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=mask,
    )
    query = FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=mask,
    )
    source_name = label_name if input_name is None else str(input_name)
    name = str(query_name)
    operator_batch = OperatorBatch(
        inputs={source_name: source},
        queries={name: query},
        case_axes=(axis,),
        case_shape=(case_count,),
    )
    layout = OperatorDomainLayout(
        name,
        (axis, None),
        (case_count, width),
    )
    return OperatorDomainView(
        operator_batch,
        {name: layout},
        kind="ragged_series",
    )


def operator_domain_view_from_trajectory(
    batch: Any,
    /,
    *,
    inputs: Mapping[str, str],
    query_label: str,
    query_name: str = "query",
) -> OperatorDomainView:
    """Group paired trajectory observations into reversible ragged operator cases."""

    from phydrax.domain import PointBatch, TRAJECTORY_CASE_INDEX_KEY

    if not isinstance(batch, PointBatch):
        raise TypeError("Trajectory operator views require a PointBatch.")
    if not inputs:
        raise ValueError("Trajectory operator views require at least one input.")
    time_label = str(query_label)
    if time_label not in batch.points:
        raise KeyError(f"Unknown trajectory query label {time_label!r}.")
    axis = batch.structure.axis_for(time_label)
    if axis is None:
        raise ValueError("Trajectory queries require one paired observation axis.")
    if TRAJECTORY_CASE_INDEX_KEY not in batch.points:
        raise TypeError("PointBatch does not carry trajectory case indices.")

    case_ids = np.asarray(
        jax.device_get(_field(batch.points, TRAJECTORY_CASE_INDEX_KEY).data),
        dtype=np.int64,
    )
    times = jnp.asarray(_field(batch.points, time_label).data, dtype=float)
    if case_ids.ndim != 1 or times.shape != case_ids.shape:
        raise ValueError("Trajectory case indices and times must be aligned vectors.")
    if case_ids.size == 0:
        raise ValueError("Trajectory operator views require observations.")
    unique_cases, inverse = np.unique(case_ids, return_inverse=True)
    case_count = int(unique_cases.size)
    selected_counts = np.bincount(inverse, minlength=case_count)
    width = int(selected_counts.max(initial=0))
    positions = np.full((case_count, width), -1, dtype=np.int64)
    restore_slots = np.empty(case_ids.shape, dtype=np.int32)
    first_positions = np.empty((case_count,), dtype=np.int64)
    for case_index in range(case_count):
        selected_positions = np.flatnonzero(inverse == case_index)
        count = int(selected_positions.size)
        first_positions[case_index] = selected_positions[0]
        positions[case_index, :count] = selected_positions
        restore_slots[selected_positions] = case_index * width + np.arange(
            count, dtype=np.int32
        )

    safe_positions = jnp.maximum(jnp.asarray(positions, dtype=jnp.int32), 0)
    occupied = jnp.asarray(positions >= 0)
    grouped_times = jnp.where(occupied, times[safe_positions], 0.0)
    geometry_case_shape = (case_count,) if case_count > 1 else ()
    base_times = grouped_times if case_count > 1 else grouped_times[0]
    base_mask = occupied if case_count > 1 else occupied[0]
    query = FunctionSamples(
        values=None,
        coordinates=base_times[..., None],
        mask=base_mask,
    )

    sources: dict[str, FunctionSamples] = {}
    for source_name, label in inputs.items():
        label_name = str(label)
        if label_name not in batch.points:
            raise KeyError(f"Unknown trajectory input label {label_name!r}.")
        if label_name == time_label:
            sources[str(source_name)] = FunctionSamples(
                values=base_times,
                coordinates=base_times[..., None],
                mask=base_mask,
            )
            continue
        observations = _payload_feature_array(
            batch.points[label_name],
            1,
            name=f"Trajectory input {label_name!r}",
        )
        case_values = observations[jnp.asarray(first_positions, dtype=jnp.int32)]
        if int(case_values.shape[-1]) == 1:
            case_values = case_values[..., 0]
        if case_count == 1:
            case_values = case_values[0]
        sources[str(source_name)] = FunctionSamples(values=case_values)

    name = str(query_name)
    case_axis = f"{axis}__trajectory_case"
    operator_batch = OperatorBatch(
        inputs=sources,
        queries={name: query},
        case_axes=((case_axis,) if case_count > 1 else ()),
        case_shape=geometry_case_shape,
    )
    layout = OperatorDomainLayout(
        name,
        (axis,),
        geometry_case_shape + (width,),
        gather_indices=jnp.asarray(restore_slots, dtype=jnp.int32),
    )
    return OperatorDomainView(
        operator_batch,
        {name: layout},
        kind="trajectory",
    )


def _graph_entity_metadata(
    batch: Any,
    /,
) -> tuple[OperatorTopologyEntity, OperatorTopologySite, Array, Array | None]:
    if batch.component_kind == "nodes":
        return "node", "node", batch.graph.n_node, batch.graph.node_mask
    if batch.component_kind == "edges":
        return "edge", "edge", batch.graph.n_edge, batch.graph.edge_mask
    if batch.component_kind == "globals":
        return (
            "global",
            "global",
            jnp.ones((batch.graph.num_graphs,), dtype=jnp.int32),
            batch.graph.graph_mask,
        )
    raise ValueError(f"Unknown GraphBatch component kind {batch.component_kind!r}.")


def operator_domain_view_from_graph(
    batch: Any,
    /,
    *,
    inputs: Mapping[str, str],
    query_name: str = "query",
    query_labels: Sequence[str] = (),
    topology_kind: OperatorTopologyKind = "graph",
    topology_site: OperatorTopologySite | None = None,
    topology_entity: OperatorTopologyEntity | None = None,
    view_kind: Literal["graph", "simplicial"] = "graph",
) -> OperatorDomainView:
    """Adapt a graph entity batch to padded per-graph operator cases."""

    from phydrax.domain.graph import (
        GRAPH_ENTITY_INDEX_KEY,
        GRAPH_GRAPH_INDEX_KEY,
        GraphBatch,
    )

    if not isinstance(batch, GraphBatch):
        raise TypeError("operator_domain_view_from_graph requires a GraphBatch.")
    if not inputs:
        raise ValueError("Graph operator views require at least one named input.")
    name = str(query_name)
    if not name:
        raise ValueError("Graph operator query names must be non-empty.")
    if batch.structure.axis_names is None:
        raise ValueError("Graph operator views require a canonical domain structure.")
    graph_axis = batch.structure.axis_for(batch.graph_label)
    if graph_axis is None:
        raise ValueError("GraphBatch graph labels require one sampled entity axis.")

    original_axes = tuple(batch.structure.axis_names)
    other_axes = tuple(axis for axis in original_axes if axis != graph_axis)
    other_shape = tuple(_axis_size_from_points(batch.points, axis) for axis in other_axes)
    graph_count = int(batch.graph.num_graphs)
    graph_case_axis = f"{graph_axis}__graph"
    while graph_case_axis in original_axes:
        graph_case_axis += "_"
    case_axes = other_axes + ((graph_case_axis,) if graph_count > 1 else ())
    case_shape = other_shape + ((graph_count,) if graph_count > 1 else ())

    default_entity, default_site, default_counts, default_mask = _graph_entity_metadata(
        batch
    )
    entity = default_entity if topology_entity is None else topology_entity
    site = default_site if topology_site is None else topology_site
    if entity == default_entity:
        counts_array = default_counts
        entity_mask_array = default_mask
    elif entity == "node":
        counts_array = batch.graph.n_node
        entity_mask_array = batch.graph.node_mask
    elif entity == "edge":
        counts_array = batch.graph.n_edge
        entity_mask_array = batch.graph.edge_mask
    else:
        counts_array = jnp.ones((graph_count,), dtype=jnp.int32)
        entity_mask_array = batch.graph.graph_mask

    entity_indices = np.asarray(
        jax.device_get(_field(batch.points, GRAPH_ENTITY_INDEX_KEY).data),
        dtype=np.int64,
    )
    graph_ids = np.asarray(
        jax.device_get(_field(batch.points, GRAPH_GRAPH_INDEX_KEY).data),
        dtype=np.int64,
    )
    if entity_indices.ndim != 1 or graph_ids.shape != entity_indices.shape:
        raise ValueError(
            "Graph entity and graph indices must be aligned rank-one arrays."
        )
    if entity_indices.size == 0:
        raise ValueError("Graph operator views require at least one selected entity.")
    if np.any(graph_ids < 0) or np.any(graph_ids >= graph_count):
        raise ValueError("GraphBatch graph indices are out of range.")

    counts = np.asarray(jax.device_get(counts_array), dtype=np.int64)
    offsets = np.concatenate(
        (np.zeros((1,), dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64))
    )
    local_entities = entity_indices - offsets[graph_ids]
    if np.any(local_entities < 0) or np.any(local_entities >= counts[graph_ids]):
        raise ValueError(
            "GraphBatch entity indices do not belong to their declared graphs."
        )
    selected_counts = np.bincount(graph_ids, minlength=graph_count)
    width = int(selected_counts.max(initial=0))
    positions = np.full((graph_count, width), -1, dtype=np.int64)
    mapping = np.full((graph_count, width), -1, dtype=np.int32)
    restore_slots = np.empty(entity_indices.shape, dtype=np.int32)

    active = np.ones(entity_indices.shape, dtype=bool)
    if entity_mask_array is not None:
        entity_mask = np.asarray(jax.device_get(entity_mask_array), dtype=bool)
        active &= entity_mask[entity_indices]
    if batch.graph.graph_mask is not None:
        graph_mask = np.asarray(jax.device_get(batch.graph.graph_mask), dtype=bool)
        active &= graph_mask[graph_ids]
    for graph_index in range(graph_count):
        selected_positions = np.flatnonzero(graph_ids == graph_index)
        count = int(selected_positions.size)
        positions[graph_index, :count] = selected_positions
        mapping[graph_index, :count] = np.where(
            active[selected_positions],
            local_entities[selected_positions],
            -1,
        )
        restore_slots[selected_positions] = graph_index * width + np.arange(
            count, dtype=np.int32
        )

    payload = _graph_payload_array(batch.points[batch.graph_label], graph_axis)
    safe_positions = jnp.maximum(jnp.asarray(positions, dtype=jnp.int32), 0)
    occupied = jnp.asarray(positions >= 0)
    grouped_payload = payload[safe_positions]
    grouped_payload = jnp.where(occupied[..., None], grouped_payload, 0)
    coordinate_parts = [grouped_payload]
    for coordinate_label in query_labels:
        label_name = str(coordinate_label)
        if label_name == batch.graph_label:
            continue
        if label_name not in batch.points:
            raise KeyError(f"Unknown GraphBatch query label {label_name!r}.")
        query_values = _graph_payload_array(batch.points[label_name], graph_axis)
        grouped_query = query_values[safe_positions]
        coordinate_parts.append(jnp.where(occupied[..., None], grouped_query, 0))
    grouped_coordinates = jnp.concatenate(tuple(coordinate_parts), axis=-1)
    grouped_mask = occupied & jnp.asarray(
        np.where(positions >= 0, active[np.maximum(positions, 0)], False)
    )

    base_case_shape = (graph_count,) if graph_count > 1 else ()
    base_mapping = mapping if graph_count > 1 else mapping[0]
    base_topology = OperatorTopology(
        batch.graph,
        base_mapping,
        case_shape=base_case_shape,
        kind=topology_kind,
        site=site,
        entity=entity,
    )
    topology = broadcast_operator_topology(base_topology, case_shape)
    base_payload = grouped_payload if graph_count > 1 else grouped_payload[0]
    base_coordinates = grouped_coordinates if graph_count > 1 else grouped_coordinates[0]
    base_mask = grouped_mask if graph_count > 1 else grouped_mask[0]
    coordinate_dimension = int(base_coordinates.shape[-1])
    coordinates = jnp.broadcast_to(
        base_coordinates,
        case_shape + (width, coordinate_dimension),
    )
    mask = jnp.broadcast_to(base_mask, case_shape + (width,))
    valid_count = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)
    quadrature_weights = jnp.where(mask, 1.0 / valid_count, 0.0)
    source_dimension = int(base_payload.shape[-1])
    source_values = jnp.broadcast_to(
        base_payload,
        case_shape + (width, source_dimension),
    )
    if source_dimension == 1:
        source_values = source_values[..., 0]
    graph_source = FunctionSamples(
        values=source_values,
        coordinates=coordinates,
        mask=mask,
        quadrature_weights=quadrature_weights,
        topology=topology,
    )
    query = FunctionSamples(
        values=None,
        coordinates=coordinates,
        mask=mask,
        quadrature_weights=quadrature_weights,
        topology=topology,
    )

    sources: dict[str, FunctionSamples] = {}
    for source_name, label in inputs.items():
        label_name = str(label)
        if label_name == batch.graph_label:
            sources[str(source_name)] = graph_source
        else:
            if label_name not in batch.points:
                raise KeyError(f"Unknown GraphBatch input label {label_name!r}.")
            fields = _field_leaves(batch.points[label_name])
            if all(graph_axis in field.dims for field in fields):
                entity_values = _graph_payload_array(
                    batch.points[label_name],
                    graph_axis,
                )
                grouped_values = entity_values[safe_positions]
                grouped_values = jnp.where(
                    occupied[..., None],
                    grouped_values,
                    0,
                )
                base_values = grouped_values if graph_count > 1 else grouped_values[0]
                value_dimension = int(base_values.shape[-1])
                values = jnp.broadcast_to(
                    base_values,
                    case_shape + (width, value_dimension),
                )
                if value_dimension == 1:
                    values = values[..., 0]
                sources[str(source_name)] = FunctionSamples(
                    values=values,
                    coordinates=coordinates,
                    mask=mask,
                    quadrature_weights=quadrature_weights,
                    topology=topology,
                )
            else:
                sources[str(source_name)] = _case_function_samples(
                    batch.points[label_name],
                    case_axes,
                    case_shape,
                )

    operator_batch = OperatorBatch(
        inputs=sources,
        queries={name: query},
        case_axes=case_axes,
        case_shape=case_shape,
    )
    case_count = prod(other_shape) if other_shape else 1
    restore_indices = jnp.arange(case_count, dtype=jnp.int32).reshape(
        other_shape + (1,)
    ) * (graph_count * width) + jnp.asarray(restore_slots, dtype=jnp.int32)
    current_axes = other_axes + (graph_axis,)
    permutation = tuple(current_axes.index(axis) for axis in original_axes)
    if permutation != tuple(range(len(permutation))):
        restore_indices = jnp.transpose(restore_indices, permutation)
    layout = OperatorDomainLayout(
        name,
        original_axes,
        case_shape + (width,),
        gather_indices=restore_indices,
    )
    return OperatorDomainView(operator_batch, {name: layout}, kind=view_kind)


def operator_domain_view_from_simplicial(
    batch: Any,
    /,
    *,
    inputs: Mapping[str, str],
    site: Literal["vertex", "edge", "face", "cell"],
    query_name: str = "query",
) -> OperatorDomainView:
    """Adapt simplicial cells represented as nodes of a canonical graph."""

    if batch.component_kind != "nodes":
        raise ValueError(
            "Simplicial graph batches must expose their selected cells as graph nodes."
        )
    return operator_domain_view_from_graph(
        batch,
        inputs=inputs,
        query_name=query_name,
        topology_kind="simplicial",
        topology_site=site,
        topology_entity="node",
        view_kind="simplicial",
    )


__all__ = [
    "OperatorDomainKind",
    "OperatorDomainLayout",
    "OperatorDomainView",
    "operator_domain_view_from_grid",
    "operator_domain_view_from_graph",
    "operator_domain_view_from_points",
    "operator_domain_view_from_ragged_series",
    "operator_domain_view_from_simplicial",
    "operator_domain_view_from_trajectory",
]
