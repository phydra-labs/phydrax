#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._frozendict import frozendict
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._uncertainty import (
    UNCERTAINTY_SOURCES,
    UncertaintySource,
    validate_uncertainty_source,
)
from ._precision import PredictivePrecisionPolicy


class SampleAxis(StrictModule):
    """A named predictive-sample dimension and its uncertainty source."""

    dim: str
    source: UncertaintySource

    def __init__(self, dim: str, source: UncertaintySource):
        if not isinstance(dim, str) or not dim:
            raise ValueError("SampleAxis.dim must be a non-empty string.")
        self.dim = dim
        self.source = validate_uncertainty_source(
            source,
            owner="SampleAxis.source",
        )


class PredictionInterval(StrictModule):
    """Lower and upper predictive bounds with coverage semantics."""

    lower: cx.Field
    upper: cx.Field
    nominal_coverage: float
    simultaneous: bool
    calibrated: bool

    def __init__(
        self,
        lower: cx.Field,
        upper: cx.Field,
        *,
        nominal_coverage: float,
        simultaneous: bool = False,
        calibrated: bool = False,
    ):
        if not isinstance(lower, cx.Field) or not isinstance(upper, cx.Field):
            raise TypeError("PredictionInterval bounds must be coordax.Field objects.")
        if lower.dims != upper.dims or lower.data.shape != upper.data.shape:
            raise ValueError(
                "PredictionInterval bounds must have matching shapes and dims."
            )
        coverage = float(nominal_coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("nominal_coverage must lie strictly between zero and one.")
        if bool(jnp.any(jnp.asarray(lower.data) > jnp.asarray(upper.data))):
            raise ValueError(
                "PredictionInterval lower bounds must not exceed upper bounds."
            )
        self.lower = lower
        self.upper = upper
        self.nominal_coverage = coverage
        self.simultaneous = bool(simultaneous)
        self.calibrated = bool(calibrated)


class PredictiveField(StrictModule):
    """Coordinate-aware predictive samples with explicit uncertainty axes."""

    samples: cx.Field
    sample_axes: tuple[SampleAxis, ...]
    conditional_variance: cx.Field | None
    valid: cx.Field | None
    precision: PredictivePrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)

    def __init__(
        self,
        samples: cx.Field,
        sample_axes: Iterable[SampleAxis],
        *,
        conditional_variance: cx.Field | None = None,
        valid: cx.Field | None = None,
        precision: PredictivePrecisionPolicy | None = None,
    ):
        if not isinstance(samples, cx.Field):
            raise TypeError("PredictiveField.samples must be a coordax.Field.")
        axes = tuple(sample_axes)
        if not axes:
            raise ValueError("PredictiveField requires at least one sample axis.")
        dims = tuple(axis.dim for axis in axes)
        if len(set(dims)) != len(dims):
            raise ValueError("PredictiveField sample dimensions must be unique.")
        missing = tuple(dim for dim in dims if dim not in samples.named_shape)
        if missing:
            raise ValueError(
                f"Predictive sample dimensions {missing!r} are absent from field dims "
                f"{samples.dims!r}."
            )
        if any(int(samples.named_shape[dim]) <= 0 for dim in dims):
            raise ValueError("Predictive sample dimensions must be non-empty.")
        if conditional_variance is not None and any(
            axis.source == "observation" for axis in axes
        ):
            raise ValueError(
                "conditional_variance and an explicit observation sample axis are "
                "mutually exclusive."
            )
        if conditional_variance is not None:
            if not isinstance(conditional_variance, cx.Field):
                raise TypeError("conditional_variance must be a coordax.Field or None.")
            _broadcast_field_data(conditional_variance, samples)
            if bool(jnp.any(jnp.asarray(conditional_variance.data) < 0)):
                raise ValueError("conditional_variance must be non-negative.")
        if valid is not None:
            if not isinstance(valid, cx.Field):
                raise TypeError("valid must be a coordax.Field or None.")
            unknown = tuple(
                dim for dim in valid.dims if dim is not None and dim not in dims
            )
            if unknown:
                raise ValueError(
                    "Predictive validity masks may use only sample dimensions; "
                    f"got {unknown!r}."
                )
            if any(dim is None for dim in valid.dims):
                raise ValueError(
                    "Predictive validity masks may use only named dimensions."
                )
            _broadcast_field_data(valid, samples)
        precision_ = PredictivePrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, PredictivePrecisionPolicy):
            raise TypeError("precision must be a PredictivePrecisionPolicy.")
        stored_samples = cx.Field(
            precision_.storage(samples.data),
            dims=samples.dims,
        )
        self.samples = stored_samples
        self.sample_axes = axes
        self.conditional_variance = conditional_variance
        self.valid = valid
        self.precision = precision_
        self.precision_evidence = precision_.evidence(
            jnp.asarray(stored_samples.data).dtype
        )

    def _selected_dims(
        self, sources: UncertaintySource | Iterable[UncertaintySource] | None
    ) -> tuple[str, ...]:
        if sources is None:
            return tuple(axis.dim for axis in self.sample_axes)
        if isinstance(sources, str):
            selected = (sources,)
        else:
            selected = tuple(sources)
        invalid = tuple(
            source for source in selected if source not in UNCERTAINTY_SOURCES
        )
        if invalid:
            raise ValueError(f"Unknown uncertainty sources: {invalid!r}.")
        dims = tuple(
            axis.dim for axis in self.sample_axes if axis.source in frozenset(selected)
        )
        missing = tuple(
            source
            for source in selected
            if not any(axis.source == source for axis in self.sample_axes)
        )
        if missing:
            raise ValueError(f"Predictive field has no sample axes for {missing!r}.")
        return dims

    def mean(
        self,
        *,
        sources: UncertaintySource | Iterable[UncertaintySource] | None = None,
    ) -> cx.Field:
        return _masked_moment(
            self.samples,
            self._selected_dims(sources),
            self.valid,
            1,
            dtype=self.precision.summary_dtype,
        )

    def variance(
        self,
        *,
        sources: UncertaintySource | Iterable[UncertaintySource] | None = None,
    ) -> cx.Field:
        dims = self._selected_dims(sources)
        mean = _masked_moment(
            self.samples, dims, self.valid, 1, dtype=self.precision.summary_dtype
        )
        second = _masked_moment(
            self.samples, dims, self.valid, 2, dtype=self.precision.summary_dtype
        )
        return cx.Field(
            jnp.maximum(jnp.asarray(second.data) - jnp.asarray(mean.data) ** 2, 0.0),
            dims=mean.dims,
        )

    def std(
        self,
        *,
        sources: UncertaintySource | Iterable[UncertaintySource] | None = None,
    ) -> cx.Field:
        variance = self.variance(sources=sources)
        return cx.Field(jnp.sqrt(jnp.asarray(variance.data)), dims=variance.dims)

    def quantile(
        self,
        q: float | Array,
        *,
        sources: UncertaintySource | Iterable[UncertaintySource] | None = None,
    ) -> cx.Field:
        q_arr = jnp.asarray(
            q,
            dtype=(
                float
                if self.precision.summary_dtype is None
                else self.precision.summary_dtype
            ),
        )
        if q_arr.ndim != 0:
            raise ValueError("PredictiveField.quantile currently requires a scalar q.")
        q_value = float(q_arr)
        if not 0.0 <= q_value <= 1.0:
            raise ValueError("q must lie between zero and one.")
        dims = self._selected_dims(sources)
        positions = tuple(self.samples.dims.index(dim) for dim in dims)
        data = self.precision.summary(self.samples.data)
        if self.valid is not None:
            valid = _broadcast_field_data(self.valid, self.samples).astype(bool)
            data = jnp.where(valid, data, jnp.nan)
            reduced = jnp.nanquantile(data, q_value, axis=positions)
        else:
            reduced = jnp.quantile(data, q_value, axis=positions)
        out_dims = tuple(dim for dim in self.samples.dims if dim not in dims)
        return cx.Field(reduced, dims=out_dims)

    def interval(
        self,
        lower_q: float,
        upper_q: float,
        *,
        sources: UncertaintySource | Iterable[UncertaintySource] | None = None,
    ) -> PredictionInterval:
        lower_value, upper_value = float(lower_q), float(upper_q)
        if not 0.0 <= lower_value < upper_value <= 1.0:
            raise ValueError("Require 0 <= lower_q < upper_q <= 1.")
        return PredictionInterval(
            self.quantile(lower_value, sources=sources),
            self.quantile(upper_value, sources=sources),
            nominal_coverage=upper_value - lower_value,
        )

    def epistemic_variance(self) -> cx.Field:
        return self.variance(sources="epistemic")

    def input_variance(self) -> cx.Field:
        return self.variance(sources="input")

    def observation_variance(self) -> cx.Field:
        observation_axes = tuple(
            axis.dim for axis in self.sample_axes if axis.source == "observation"
        )
        if observation_axes:
            return self.variance(sources="observation")
        if self.conditional_variance is None:
            raise ValueError("Predictive field has no observation uncertainty.")
        data = _broadcast_field_data(self.conditional_variance, self.samples)
        field = cx.Field(data, dims=self.samples.dims)
        sample_dims = tuple(axis.dim for axis in self.sample_axes)
        return _masked_moment(
            field,
            sample_dims,
            self.valid,
            1,
            dtype=self.precision.summary_dtype,
        )

    def process_variance(self) -> cx.Field:
        return self.variance(sources="process")

    def numerical_variance(self) -> cx.Field:
        return self.variance(sources="numerical")

    def total_variance(self) -> cx.Field:
        sample_variance = self.variance()
        if self.conditional_variance is None:
            return sample_variance
        conditional = self.observation_variance()
        conditional_data = _broadcast_field_data(conditional, sample_variance)
        return cx.Field(
            jnp.asarray(sample_variance.data) + conditional_data,
            dims=sample_variance.dims,
        )

    def decompose_variance(self) -> frozendict[str, cx.Field]:
        parts: dict[str, cx.Field] = {}
        sources = {axis.source for axis in self.sample_axes}
        if "epistemic" in sources:
            parts["epistemic"] = self.epistemic_variance()
        if "input" in sources:
            parts["input"] = self.input_variance()
        if "process" in sources:
            parts["process"] = self.process_variance()
        if "numerical" in sources:
            parts["numerical"] = self.numerical_variance()
        if "observation" in sources or self.conditional_variance is not None:
            parts["observation"] = self.observation_variance()
        parts["total"] = self.total_variance()
        return frozendict(parts)


def _sample_validity(
    data: Array,
    /,
    *,
    sample_dim: str,
    valid_policy: Literal["record", "raise"],
    owner: str,
) -> cx.Field:
    if valid_policy not in ("record", "raise"):
        raise ValueError("valid_policy must be 'record' or 'raise'.")
    sample_data = jnp.asarray(data)
    count = int(sample_data.shape[0])
    valid_data = jnp.all(jnp.isfinite(sample_data).reshape((count, -1)), axis=1)
    if valid_policy == "raise" and not bool(jnp.all(valid_data)):
        failed = tuple(int(index) for index in jnp.where(~valid_data)[0])
        raise FloatingPointError(f"{owner} produced invalid realizations at {failed!r}.")
    return cx.Field(valid_data, dims=(sample_dim,))


def _masked_moment(
    field: cx.Field,
    dims: tuple[str, ...],
    valid: cx.Field | None,
    power: int,
    *,
    dtype: Any | None = None,
) -> cx.Field:
    if not dims:
        data = jnp.asarray(field.data) ** power
        if dtype is not None:
            data = data.astype(dtype)
        return cx.Field(data, dims=field.dims)
    positions = tuple(field.dims.index(dim) for dim in dims)
    values = jnp.asarray(field.data)
    if dtype is not None:
        values = values.astype(dtype)
    values = values**power
    if valid is None:
        reduced = jnp.mean(values, axis=positions)
    else:
        mask = _broadcast_field_data(valid, field).astype(values.dtype)
        count = jnp.sum(mask, axis=positions)
        total = jnp.sum(jnp.where(mask.astype(bool), values, 0.0), axis=positions)
        reduced = total / jnp.maximum(count, 1.0)
        reduced = jnp.where(count > 0, reduced, jnp.nan)
    out_dims = tuple(dim for dim in field.dims if dim not in dims)
    return cx.Field(reduced, dims=out_dims)


def _broadcast_field_data(source: cx.Field, target: cx.Field) -> Array:
    source_named = tuple(dim for dim in source.dims if dim is not None)
    target_named = tuple(dim for dim in target.dims if dim is not None)
    unknown = tuple(dim for dim in source_named if dim not in target_named)
    if unknown:
        raise ValueError(
            f"Cannot broadcast field dims {source.dims!r} to {target.dims!r}; "
            f"unknown dimensions {unknown!r}."
        )
    if any(dim is None for dim in source.dims):
        if source.dims != target.dims or source.data.shape != target.data.shape:
            raise ValueError(
                "Fields with positional dimensions must match the target shape and dims."
            )
        return jnp.asarray(source.data)
    ordered = tuple(dim for dim in target.dims if dim in source_named)
    permutation = tuple(source.dims.index(dim) for dim in ordered)
    data = jnp.asarray(source.data)
    if permutation != tuple(range(data.ndim)):
        data = jnp.transpose(data, permutation)
    shape: list[int] = []
    ordered_i = 0
    for dim in target.dims:
        if dim in source_named:
            shape.append(int(data.shape[ordered_i]))
            ordered_i += 1
        else:
            shape.append(1)
    try:
        return jnp.broadcast_to(data.reshape(tuple(shape)), target.data.shape)
    except ValueError as exc:
        raise ValueError(
            f"Field shape {source.data.shape} with dims {source.dims!r} cannot "
            f"broadcast to shape {target.data.shape} with dims {target.dims!r}."
        ) from exc


__all__ = [
    "PredictionInterval",
    "PredictiveField",
    "SampleAxis",
    "UncertaintySource",
]
