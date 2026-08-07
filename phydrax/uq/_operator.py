#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from .._frozendict import frozendict
from .._strict import StrictModule
from ..nn.models.core._base import _AbstractOperatorModel
from ..nn.models.core._operator import (
    FunctionSamples,
    OperatorBatch,
    OperatorOutputSpec,
    OperatorPrediction,
)
from ..nn.operator_training._linearization import OperatorLinearization
from ._covariance import AbstractCovariance
from ._linearized import LinearizedPropagationResult, propagate_linearized_map
from ._predictive import (
    PredictionInterval,
    PredictiveField,
    SampleAxis,
    UncertaintySource,
)


ValidPolicy = Literal["record", "raise"]
_OPERATOR_POINT_DIM = "__phydra_operator_point"
_OPERATOR_CHANNEL_DIM = "__phydra_operator_channel"


def _validate_valid_policy(value: ValidPolicy) -> ValidPolicy:
    if value not in ("record", "raise"):
        raise ValueError("valid_policy must be 'record' or 'raise'.")
    return value


def _physical_dims(
    query: FunctionSamples,
    output_spec: OperatorOutputSpec,
    case_axes: Sequence[str],
    /,
) -> tuple[str, ...]:
    cases = tuple(str(axis) for axis in case_axes)
    if any(not axis for axis in cases):
        raise ValueError("Operator case dimensions must be non-empty strings.")
    if query.axes:
        query_dims = query.axis_names
    elif query.coordinates is not None:
        query_dims = (_OPERATOR_POINT_DIM,)
    elif query.sample_shape:
        raise ValueError("Operator query samples have no named or point-cloud geometry.")
    else:
        query_dims = ()
    channel_dims = () if output_spec.channels == "scalar" else (_OPERATOR_CHANNEL_DIM,)
    dims = cases + query_dims + channel_dims
    if len(set(dims)) != len(dims):
        raise ValueError(
            "Operator case, query, and reserved output dimensions must be unique; "
            f"got {dims!r}."
        )
    return dims


def _expected_output_shape(
    query: FunctionSamples,
    output_spec: OperatorOutputSpec,
    case_shape: Sequence[int],
    /,
) -> tuple[int, ...]:
    return (
        tuple(int(size) for size in case_shape)
        + query.sample_shape
        + output_spec.channel_shape
    )


def _output_mask(
    query: FunctionSamples,
    output_spec: OperatorOutputSpec,
    case_shape: Sequence[int],
    /,
) -> Array:
    mask = query.mask_array(case_shape=case_shape)
    if output_spec.channels != "scalar":
        mask = mask[..., None]
        mask = jnp.broadcast_to(mask, mask.shape[:-1] + output_spec.channel_shape)
    return jnp.asarray(mask, dtype=bool)


def _output_weights(
    query: FunctionSamples,
    output_spec: OperatorOutputSpec,
    case_shape: Sequence[int],
    /,
    *,
    normalized: bool,
) -> Array:
    weights = query.weights(case_shape=case_shape, normalized=normalized)
    if output_spec.channels != "scalar":
        weights = weights[..., None]
        weights = jnp.broadcast_to(
            weights, weights.shape[:-1] + output_spec.channel_shape
        )
    return jnp.asarray(weights)


def _broadcast_named(source: cx.Field, target: cx.Field, /) -> Array:
    if any(dim is None for dim in source.dims) or any(dim is None for dim in target.dims):
        if source.dims != target.dims or source.data.shape != target.data.shape:
            raise ValueError(
                "Fields with positional dimensions must have identical structure."
            )
        return jnp.asarray(source.data)
    source_dims = tuple(str(dim) for dim in source.dims)
    target_dims = tuple(str(dim) for dim in target.dims)
    unknown = tuple(dim for dim in source_dims if dim not in target_dims)
    if unknown:
        raise ValueError(
            f"Cannot broadcast field dimensions {source.dims!r} to {target.dims!r}; "
            f"unknown dimensions {unknown!r}."
        )
    ordered = tuple(dim for dim in target_dims if dim in source_dims)
    permutation = tuple(source_dims.index(dim) for dim in ordered)
    data = jnp.asarray(source.data)
    if permutation != tuple(range(data.ndim)):
        data = jnp.transpose(data, permutation)
    shape = tuple(
        int(data.shape[ordered.index(dim)]) if dim in ordered else 1
        for dim in target_dims
    )
    try:
        return jnp.broadcast_to(data.reshape(shape), target.data.shape)
    except ValueError as exc:
        raise ValueError(
            f"Field shape {source.data.shape} with dimensions {source.dims!r} cannot "
            f"broadcast to shape {target.data.shape} with dimensions {target.dims!r}."
        ) from exc


def _optional_array_equal(left: Array | None, right: Array | None, /) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(jnp.array_equal(jnp.asarray(left), jnp.asarray(right)))


def _queries_equal(left: FunctionSamples, right: FunctionSamples, /) -> bool:
    if left is right:
        return True
    if len(left.axes) != len(right.axes):
        return False
    for left_axis, right_axis in zip(left.axes, right.axes, strict=True):
        if (
            left_axis.name != right_axis.name
            or left_axis.basis != right_axis.basis
            or left_axis.periodic != right_axis.periodic
            or not bool(jnp.array_equal(left_axis.nodes, right_axis.nodes))
            or not _optional_array_equal(
                left_axis.quadrature_weights, right_axis.quadrature_weights
            )
        ):
            return False
    return (
        _optional_array_equal(left.coordinates, right.coordinates)
        and _optional_array_equal(left.quadrature_weights, right.quadrature_weights)
        and _optional_array_equal(left.mask, right.mask)
    )


def _select_prediction_field(
    prediction: OperatorPrediction,
    field_name: str,
    /,
):
    name = str(field_name)
    if not name:
        raise ValueError("field_name must be non-empty.")
    field = prediction.field(name)
    return name, field, prediction.query_geometry(field.query_name)


def operator_prediction_field(
    prediction: OperatorPrediction,
    /,
    *,
    field_name: str,
) -> cx.Field:
    """Convert one selected deterministic operator output to a named field.

    ``field_name`` explicitly selects one named physical output. Masked query
    padding is replaced with zero; the original prediction remains unchanged.
    """
    if not isinstance(prediction, OperatorPrediction):
        raise TypeError("prediction must be an OperatorPrediction.")
    _, field, query = _select_prediction_field(prediction, field_name)
    dims = _physical_dims(query, field.spec, prediction.case_axes)
    expected = _expected_output_shape(query, field.spec, prediction.case_shape)
    values = jnp.asarray(field.values)
    if values.shape != expected:
        raise ValueError(
            f"Operator prediction values must have shape {expected}; got {values.shape}."
        )
    mask = _output_mask(query, field.spec, prediction.case_shape)
    values = jnp.where(mask, values, jnp.zeros((), dtype=values.dtype))
    return cx.Field(values, dims=dims)


def _collapse_case_array(
    array: Array | None,
    /,
    *,
    case_shape: tuple[int, ...],
    sample_ndim: int,
    positions: tuple[int, ...],
    owner: str,
) -> Array | None:
    if array is None:
        return None
    value = jnp.asarray(array)
    if value.ndim == sample_ndim:
        return value
    if tuple(int(size) for size in value.shape[: len(case_shape)]) != case_shape:
        raise ValueError(
            f"{owner} must be shared or carry full case shape {case_shape}; "
            f"got {value.shape}."
        )
    current = value
    for position in sorted(positions, reverse=True):
        first = jnp.take(current, 0, axis=position)
        expanded = jnp.expand_dims(first, axis=position)
        if not bool(jnp.array_equal(current, jnp.broadcast_to(expanded, current.shape))):
            raise ValueError(
                f"Output query geometry varies along input sample axis at position "
                f"{position}; pointwise predictive statistics require a common query."
            )
        current = first
    return current


def _collapse_query(
    query: FunctionSamples,
    /,
    *,
    case_shape: tuple[int, ...],
    positions: tuple[int, ...],
) -> FunctionSamples:
    if not positions or not query.geometry_case_shape:
        return query
    coordinates = _collapse_case_array(
        query.coordinates,
        case_shape=case_shape,
        sample_ndim=2,
        positions=positions,
        owner="Query coordinates",
    )
    sample_ndim = len(query.sample_shape)
    quadrature = _collapse_case_array(
        query.quadrature_weights,
        case_shape=case_shape,
        sample_ndim=sample_ndim,
        positions=positions,
        owner="Query quadrature",
    )
    mask = _collapse_case_array(
        query.mask,
        case_shape=case_shape,
        sample_ndim=sample_ndim,
        positions=positions,
        owner="Query mask",
    )
    if query.values is None:
        values = None
    else:
        values = jax.tree_util.tree_map(
            lambda leaf: _collapse_case_array(
                jnp.asarray(leaf),
                case_shape=case_shape,
                sample_ndim=sample_ndim,
                positions=positions,
                owner="Query values",
            ),
            query.values,
        )
    return FunctionSamples(
        values=values,
        axes=query.axes,
        coordinates=coordinates,
        quadrature_weights=quadrature,
        mask=mask,
    )


def _case_contract_without_axes(
    case_axes: tuple[str, ...],
    case_shape: tuple[int, ...],
    selected: Sequence[str],
    /,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    names = tuple(str(axis) for axis in selected)
    if not names or len(set(names)) != len(names):
        raise ValueError("input_sample_axes must contain distinct case-axis names.")
    missing = tuple(name for name in names if name not in case_axes)
    if missing:
        raise ValueError(
            f"Unknown input sample axes {missing!r}; expected axes from {case_axes!r}."
        )
    positions = tuple(case_axes.index(name) for name in names)
    remaining_axes = tuple(
        axis for index, axis in enumerate(case_axes) if index not in positions
    )
    remaining_shape = tuple(
        size for index, size in enumerate(case_shape) if index not in positions
    )
    return remaining_axes, remaining_shape, positions


def _sample_validity(
    samples: cx.Field,
    sample_axes: tuple[SampleAxis, ...],
    output_mask: cx.Field,
    existing: cx.Field | None,
    /,
    *,
    valid_policy: ValidPolicy,
) -> cx.Field:
    policy = _validate_valid_policy(valid_policy)
    sample_dims = tuple(axis.dim for axis in sample_axes)
    sample_positions = tuple(samples.dims.index(dim) for dim in sample_dims)
    physical_positions = tuple(
        index for index, dim in enumerate(samples.dims) if dim not in sample_dims
    )
    permutation = sample_positions + physical_positions
    data = jnp.asarray(samples.data)
    mask = _broadcast_named(output_mask, samples).astype(bool)
    finite = jnp.isfinite(data) | ~mask
    if permutation != tuple(range(data.ndim)):
        finite = jnp.transpose(finite, permutation)
    sample_shape = tuple(
        int(samples.data.shape[position]) for position in sample_positions
    )
    physical_count = 1
    for position in physical_positions:
        physical_count *= int(samples.data.shape[position])
    valid_data = jnp.all(finite.reshape(sample_shape + (physical_count,)), axis=-1)
    validity_template = cx.Field(jnp.ones(sample_shape, dtype=bool), dims=sample_dims)
    if existing is not None:
        valid_data = valid_data & _broadcast_named(existing, validity_template).astype(
            bool
        )
    if policy == "raise" and not bool(jnp.all(valid_data)):
        failed = tuple(
            tuple(int(index) for index in row) for row in jnp.argwhere(~valid_data)
        )
        raise FloatingPointError(
            f"Operator prediction produced invalid realizations at {failed!r}."
        )
    return cx.Field(valid_data, dims=sample_dims)


class OperatorPredictionInterval(StrictModule):
    """Prediction bounds retaining operator query and output metadata."""

    lower: OperatorPrediction
    upper: OperatorPrediction
    nominal_coverage: float
    simultaneous: bool
    calibrated: bool

    def __init__(
        self,
        lower: OperatorPrediction,
        upper: OperatorPrediction,
        /,
        *,
        nominal_coverage: float,
        simultaneous: bool = False,
        calibrated: bool = False,
    ):
        if not isinstance(lower, OperatorPrediction) or not isinstance(
            upper, OperatorPrediction
        ):
            raise TypeError("Operator interval bounds must be OperatorPrediction values.")
        if (
            lower.case_axes != upper.case_axes
            or lower.case_shape != upper.case_shape
            or tuple(lower.fields) != tuple(upper.fields)
            or tuple(lower.queries) != tuple(upper.queries)
        ):
            raise ValueError("Operator interval bounds must have identical contracts.")
        for name, lower_field in lower.fields.items():
            upper_field = upper.field(name)
            lower_query = lower.query_geometry(lower_field.query_name)
            upper_query = upper.query_geometry(upper_field.query_name)
            if (
                lower_field.query_name != upper_field.query_name
                or lower_field.spec != upper_field.spec
                or lower_field.values.shape != upper_field.values.shape
                or not _queries_equal(lower_query, upper_query)
            ):
                raise ValueError(
                    "Operator interval bounds must have identical field contracts."
                )
            if bool(
                jnp.any(jnp.asarray(lower_field.values) > jnp.asarray(upper_field.values))
            ):
                raise ValueError(
                    f"Operator interval lower bounds exceed upper bounds for {name!r}."
                )
        coverage = float(nominal_coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("nominal_coverage must lie strictly between zero and one.")
        self.lower = lower
        self.upper = upper
        self.nominal_coverage = coverage
        self.simultaneous = bool(simultaneous)
        self.calibrated = bool(calibrated)


class OperatorPredictiveField(StrictModule):
    """Predictive samples paired with one operator output geometry."""

    predictive: PredictiveField
    query: FunctionSamples
    output_spec: OperatorOutputSpec
    case_axes: tuple[str, ...]
    case_shape: tuple[int, ...]
    field_name: str = eqx.field(static=True)
    query_name: str = eqx.field(static=True)

    def __init__(
        self,
        predictive: PredictiveField,
        query: FunctionSamples,
        output_spec: OperatorOutputSpec,
        /,
        *,
        case_axes: Sequence[str] = (),
        case_shape: Sequence[int] = (),
        field_name: str,
        query_name: str,
        valid_policy: ValidPolicy = "record",
    ):
        if not isinstance(predictive, PredictiveField):
            raise TypeError("predictive must be a PredictiveField.")
        if not isinstance(query, FunctionSamples):
            raise TypeError("query must be FunctionSamples.")
        if not isinstance(output_spec, OperatorOutputSpec):
            raise TypeError("output_spec must be an OperatorOutputSpec.")
        if jnp.iscomplexobj(predictive.samples.data):
            raise TypeError(
                "Operator UQ currently requires real physical outputs; expose complex "
                "values as real channels or observables."
            )
        axes = tuple(str(axis) for axis in case_axes)
        shape = tuple(int(size) for size in case_shape)
        if len(axes) != len(shape):
            raise ValueError("case_axes and case_shape ranks differ.")
        selected_field = str(field_name)
        selected_query = str(query_name)
        if not selected_field or not selected_query:
            raise ValueError("field_name and query_name must be non-empty.")
        physical_dims = _physical_dims(query, output_spec, axes)
        sample_dims = tuple(axis.dim for axis in predictive.sample_axes)
        remaining_dims = tuple(
            dim for dim in predictive.samples.dims if dim not in sample_dims
        )
        if remaining_dims != physical_dims:
            raise ValueError(
                "Predictive physical dimensions do not match the operator contract: "
                f"expected {physical_dims!r}, got {remaining_dims!r}."
            )
        physical_shape = _expected_output_shape(query, output_spec, shape)
        remaining_shape = tuple(
            int(predictive.samples.data.shape[index])
            for index, dim in enumerate(predictive.samples.dims)
            if dim not in sample_dims
        )
        if remaining_shape != physical_shape:
            raise ValueError(
                f"Predictive physical shape must be {physical_shape}; got "
                f"{remaining_shape}."
            )
        mask_data = _output_mask(query, output_spec, shape)
        mask_field = cx.Field(mask_data, dims=physical_dims)
        sample_mask = _broadcast_named(mask_field, predictive.samples).astype(bool)
        sample_values = jnp.asarray(predictive.samples.data)
        sample_data = jnp.where(
            sample_mask,
            sample_values,
            jnp.zeros((), dtype=sample_values.dtype),
        )
        sample_field = cx.Field(sample_data, dims=predictive.samples.dims)
        valid = _sample_validity(
            sample_field,
            predictive.sample_axes,
            mask_field,
            predictive.valid,
            valid_policy=valid_policy,
        )
        self.predictive = PredictiveField(
            sample_field,
            predictive.sample_axes,
            conditional_variance=predictive.conditional_variance,
            valid=valid,
        )
        self.query = query
        self.output_spec = output_spec
        self.case_axes = axes
        self.case_shape = shape
        self.field_name = selected_field
        self.query_name = selected_query

    @classmethod
    def from_predictive(
        cls,
        predictive: PredictiveField,
        batch: OperatorBatch,
        output_spec: OperatorOutputSpec,
        /,
        *,
        field_name: str,
        query_name: str,
        input_sample_axes: Sequence[str] = (),
        valid_policy: ValidPolicy = "record",
    ) -> OperatorPredictiveField:
        if not isinstance(batch, OperatorBatch):
            raise TypeError("batch must be an OperatorBatch.")
        selected_query = str(query_name)
        query = batch.query(selected_query)
        if not input_sample_axes:
            return cls(
                predictive,
                query,
                output_spec,
                case_axes=batch.case_axes,
                case_shape=batch.case_shape,
                valid_policy=valid_policy,
                field_name=field_name,
                query_name=selected_query,
            )
        case_axes, case_shape, positions = _case_contract_without_axes(
            batch.case_axes, batch.case_shape, input_sample_axes
        )
        query = _collapse_query(
            query,
            case_shape=batch.case_shape,
            positions=positions,
        )
        extra_axes = tuple(SampleAxis(str(axis), "input") for axis in input_sample_axes)
        augmented = PredictiveField(
            predictive.samples,
            predictive.sample_axes + extra_axes,
            conditional_variance=predictive.conditional_variance,
            valid=predictive.valid,
        )
        return cls(
            augmented,
            query,
            output_spec,
            case_axes=case_axes,
            case_shape=case_shape,
            valid_policy=valid_policy,
            field_name=field_name,
            query_name=selected_query,
        )

    def output_mask(self) -> Array:
        """Return the query validity mask broadcast over output channels."""
        return _output_mask(self.query, self.output_spec, self.case_shape)

    def output_weights(self, *, normalized: bool = False) -> Array:
        """Return masked quadrature broadcast over output channels."""
        return _output_weights(
            self.query,
            self.output_spec,
            self.case_shape,
            normalized=bool(normalized),
        )

    def _prediction(self, field: cx.Field, /) -> OperatorPrediction:
        dims = _physical_dims(self.query, self.output_spec, self.case_axes)
        if field.dims != dims:
            raise ValueError(
                f"Operator statistic must have physical dimensions {dims!r}; "
                f"got {field.dims!r}."
            )
        values = jnp.asarray(field.data)
        mask = self.output_mask()
        values = jnp.where(mask, values, jnp.zeros((), dtype=values.dtype))
        return OperatorPrediction.from_field(
            self.field_name,
            values,
            self.query_name,
            self.query,
            spec=self.output_spec,
            case_axes=self.case_axes,
            case_shape=self.case_shape,
        )

    def mean(self) -> OperatorPrediction:
        return self._prediction(self.predictive.mean())

    def _statistic(
        self,
        field: cx.Field,
        /,
    ) -> OperatorPrediction | OperatorPredictiveField:
        physical_dims = _physical_dims(self.query, self.output_spec, self.case_axes)
        if field.dims == physical_dims:
            return self._prediction(field)
        remaining_axes = tuple(
            axis for axis in self.predictive.sample_axes if axis.dim in field.dims
        )
        expected_dims = tuple(axis.dim for axis in remaining_axes) + physical_dims
        if field.dims != expected_dims:
            raise ValueError(
                "Operator statistic retained unexpected dimensions: "
                f"expected {expected_dims!r}, got {field.dims!r}."
            )
        return OperatorPredictiveField(
            PredictiveField(field, remaining_axes),
            self.query,
            self.output_spec,
            case_axes=self.case_axes,
            case_shape=self.case_shape,
            field_name=self.field_name,
            query_name=self.query_name,
        )

    def variance(
        self,
        *,
        sources: UncertaintySource | Iterable[UncertaintySource] | None = None,
    ) -> OperatorPrediction | OperatorPredictiveField:
        return self._statistic(self.predictive.variance(sources=sources))

    def std(
        self,
        *,
        sources: UncertaintySource | Iterable[UncertaintySource] | None = None,
    ) -> OperatorPrediction | OperatorPredictiveField:
        return self._statistic(self.predictive.std(sources=sources))

    def quantile(
        self,
        q: float | Array,
        /,
        *,
        sources: UncertaintySource | Iterable[UncertaintySource] | None = None,
    ) -> OperatorPrediction | OperatorPredictiveField:
        return self._statistic(self.predictive.quantile(q, sources=sources))

    def interval(
        self,
        lower_q: float,
        upper_q: float,
        /,
        *,
        sources: UncertaintySource | Iterable[UncertaintySource] | None = None,
    ) -> OperatorPredictionInterval:
        interval = self.predictive.interval(
            lower_q,
            upper_q,
            sources=sources,
        )
        return self._operator_interval(interval)

    def epistemic_variance(
        self,
    ) -> OperatorPrediction | OperatorPredictiveField:
        return self.variance(sources="epistemic")

    def input_variance(self) -> OperatorPrediction | OperatorPredictiveField:
        return self.variance(sources="input")

    def observation_variance(
        self,
    ) -> OperatorPrediction | OperatorPredictiveField:
        return self._statistic(self.predictive.observation_variance())

    def total_variance(self) -> OperatorPrediction:
        return self._prediction(self.predictive.total_variance())

    def decompose_variance(
        self,
    ) -> frozendict[str, OperatorPrediction | OperatorPredictiveField]:
        return frozendict(
            {
                name: self._statistic(field)
                for name, field in self.predictive.decompose_variance().items()
            }
        )

    def _operator_interval(
        self, interval: PredictionInterval, /
    ) -> OperatorPredictionInterval:
        return OperatorPredictionInterval(
            self._prediction(interval.lower),
            self._prediction(interval.upper),
            nominal_coverage=interval.nominal_coverage,
            simultaneous=interval.simultaneous,
            calibrated=interval.calibrated,
        )


def operator_predictive_from_samples(
    samples: Array,
    batch: OperatorBatch,
    output_spec: OperatorOutputSpec,
    /,
    *,
    sample_axes: Sequence[SampleAxis],
    field_name: str,
    query_name: str,
    conditional_variance: Array | cx.Field | None = None,
    input_sample_axes: Sequence[str] = (),
    valid_policy: ValidPolicy = "record",
) -> OperatorPredictiveField:
    """Construct an operator-aware predictive result from leading sample axes."""
    axes = tuple(sample_axes)
    if not axes:
        raise ValueError("sample_axes must be non-empty.")
    data = jnp.asarray(samples)
    sample_rank = len(axes)
    selected_query = str(query_name)
    query = batch.query(selected_query)
    expected = output_spec.expected_shape(batch, query_name=selected_query)
    if (
        data.ndim < sample_rank
        or tuple(int(size) for size in data.shape[sample_rank:]) != expected
    ):
        raise ValueError(
            "Operator predictive samples must have shape sample_shape + "
            f"{expected}; got {data.shape}."
        )
    dims = tuple(axis.dim for axis in axes) + _physical_dims(
        query, output_spec, batch.case_axes
    )
    sample_field = cx.Field(data, dims=dims)
    if conditional_variance is None or isinstance(conditional_variance, cx.Field):
        variance_field = conditional_variance
    else:
        variance = jnp.asarray(conditional_variance)
        physical_dims = _physical_dims(query, output_spec, batch.case_axes)
        if variance.shape == expected:
            variance_field = cx.Field(variance, dims=physical_dims)
        elif variance.shape == data.shape:
            variance_field = cx.Field(variance, dims=dims)
        else:
            raise ValueError(
                "conditional_variance must match the physical output or predictive "
                f"sample shape; got {variance.shape}."
            )
    predictive = PredictiveField(
        sample_field,
        axes,
        conditional_variance=variance_field,
    )
    return OperatorPredictiveField.from_predictive(
        predictive,
        batch,
        output_spec,
        input_sample_axes=input_sample_axes,
        valid_policy=valid_policy,
        field_name=field_name,
        query_name=selected_query,
    )


def operator_input_predictive(
    prediction: OperatorPrediction,
    /,
    *,
    input_sample_axes: Sequence[str],
    field_name: str,
    valid_policy: ValidPolicy = "record",
) -> OperatorPredictiveField:
    """Treat selected deterministic operator case axes as uncertain-input draws."""
    selected_name, selected_field, selected_query = _select_prediction_field(
        prediction,
        field_name,
    )
    case_axes, case_shape, positions = _case_contract_without_axes(
        prediction.case_axes,
        prediction.case_shape,
        input_sample_axes,
    )
    query = _collapse_query(
        selected_query,
        case_shape=prediction.case_shape,
        positions=positions,
    )
    field = operator_prediction_field(prediction, field_name=selected_name)
    predictive = PredictiveField(
        field,
        tuple(SampleAxis(str(axis), "input") for axis in input_sample_axes),
    )
    return OperatorPredictiveField(
        predictive,
        query,
        selected_field.spec,
        case_axes=case_axes,
        case_shape=case_shape,
        valid_policy=valid_policy,
        field_name=selected_name,
        query_name=selected_field.query_name,
    )


def propagate_operator_linearized(
    linearization: OperatorLinearization,
    covariance: AbstractCovariance,
    /,
    *,
    geometry: Literal["discrete", "hilbert"] = "discrete",
    source_channel_metric: Array | None = None,
    output_channel_metric: Array | None = None,
) -> LinearizedPropagationResult:
    """Propagate source covariance through one physical operator linearization."""
    if not isinstance(linearization, OperatorLinearization):
        raise TypeError("linearization must be an OperatorLinearization.")
    if not isinstance(covariance, AbstractCovariance):
        raise TypeError("covariance must implement AbstractCovariance.")
    if geometry not in ("discrete", "hilbert"):
        raise ValueError("geometry must be 'discrete' or 'hilbert'.")
    if geometry == "discrete" and (
        source_channel_metric is not None or output_channel_metric is not None
    ):
        raise ValueError("Channel metrics are available only for Hilbert geometry.")
    if geometry == "hilbert" and (
        not linearization.source_samples.has_physical_quadrature
        or not linearization.output_query.has_physical_quadrature
    ):
        raise ValueError(
            "Hilbert covariance propagation requires explicit physical quadrature "
            "for both source and output geometries."
        )

    dims = _physical_dims(
        linearization.output_query,
        linearization.output_spec,
        linearization.batch.case_axes,
    )
    expected = _expected_output_shape(
        linearization.output_query,
        linearization.output_spec,
        linearization.batch.case_shape,
    )
    if linearization.base_output.shape != expected:
        raise ValueError(
            "Operator linearization output shape does not match its query contract; "
            f"expected {expected}, got {linearization.base_output.shape}."
        )
    mask = _output_mask(
        linearization.output_query,
        linearization.output_spec,
        linearization.batch.case_shape,
    )
    mean = cx.Field(
        jnp.where(mask, linearization.base_output, 0.0),
        dims=dims,
    )

    def pushforward(tangent):
        values = linearization.pushforward(tangent)
        return cx.Field(jnp.where(mask, values, 0.0), dims=dims)

    def pullback(cotangent):
        if not isinstance(cotangent, cx.Field):
            raise TypeError("Operator covariance cotangents must be coordax.Field.")
        values = jnp.where(mask, jnp.asarray(cotangent.data), 0.0)
        if geometry == "discrete":
            return linearization.pullback(values)
        return linearization.adjoint(
            values,
            source_channel_metric=source_channel_metric,
            output_channel_metric=output_channel_metric,
        )

    return propagate_linearized_map(
        mean,
        linearization.base_input,
        covariance,
        pushforward=pushforward,
        pullback=pullback,
        source="input",
        coordinate_covariance=geometry == "discrete",
    )


def sample_operator_predictive(
    model: _AbstractOperatorModel,
    batch: OperatorBatch,
    /,
    *,
    num_samples: int,
    key: Array,
    field_name: str,
    query_name: str,
    sample_dim: str = "__phydra_uq_epistemic",
    sample_batch_size: int | None = None,
    input_sample_axes: Sequence[str] = (),
    valid_policy: ValidPolicy = "record",
) -> OperatorPredictiveField:
    """Evaluate one keyed operator repeatedly as coherent full-function draws."""
    if not isinstance(model, _AbstractOperatorModel):
        raise TypeError("model must implement the native neural-operator protocol.")
    if not isinstance(batch, OperatorBatch):
        raise TypeError("batch must be an OperatorBatch.")
    count = int(num_samples)
    if count <= 0:
        raise ValueError("num_samples must be positive.")
    chunk = count if sample_batch_size is None else int(sample_batch_size)
    if chunk <= 0:
        raise ValueError("sample_batch_size must be positive.")
    keys = jr.split(key, count)
    parts = []
    for start in range(0, count, chunk):
        selected = keys[start : min(start + chunk, count)]
        values = eqx.filter_vmap(
            lambda sample_key: (
                model.predict(batch, key=sample_key).field(field_name).values
            )
        )(selected)
        parts.append(jnp.asarray(values))
    samples = jnp.concatenate(tuple(parts), axis=0)
    return operator_predictive_from_samples(
        samples,
        batch,
        model.operator_output_specs[field_name],
        sample_axes=(SampleAxis(str(sample_dim), "epistemic"),),
        field_name=field_name,
        query_name=query_name,
        input_sample_axes=input_sample_axes,
        valid_policy=valid_policy,
    )


__all__ = [
    "OperatorPredictionInterval",
    "OperatorPredictiveField",
    "operator_input_predictive",
    "operator_prediction_field",
    "propagate_operator_linearized",
    "operator_predictive_from_samples",
    "sample_operator_predictive",
]
