#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, cast

import coordax as cx
import jax.numpy as jnp
from jaxtyping import Array, Key

from phydrax.domain import DomainFunction, PointBatch

from .._doc import DOC_KEY0
from .._numerics import LogWeightedAccumulator, weighted_diagnostics
from .._precision import complex_precision_dtype, real_precision_dtype_name
from ._batches import (
    PointIntegrationBatch,
    SeparableIntegrationBatch,
    WeightedSampleBatch,
)
from ._estimates import (
    FixedQuadratureDiagnostics,
    IntegrationEstimate,
    IntegrationProvenance,
    WeightedSampleDiagnostics,
)
from ._lowering import sum_over
from ._status import IntegrationStatus
from ._targets import DiscreteMeasureTarget, WeightedSampleTarget


def materialize_discrete_target(
    target: DiscreteMeasureTarget, /
) -> PointIntegrationBatch | SeparableIntegrationBatch:
    """Lower an external deterministic measure to an existing fixed batch."""
    if isinstance(target.weights, cx.Field):
        return PointIntegrationBatch(
            target.points,
            target.weights,
            axes=target.axes,
            mask=target.mask,
            target_mass=target.target_mass,
            provenance=target.provenance,
        )
    return SeparableIntegrationBatch(
        target.points,
        target.weights,
        axes=target.axes,
        mask=target.mask,
        target_mass=target.target_mass,
        provenance=target.provenance,
    )


def materialize_weighted_target(target: WeightedSampleTarget, /) -> WeightedSampleBatch:
    """Lower an external log-weighted measure without changing its semantics."""
    return WeightedSampleBatch(
        target.samples,
        target.log_weights,
        mask=target.mask,
        target_mass=target.target_mass,
        support_valid=target.support_valid,
        stratum_ids=target.stratum_ids,
        pair_ids=target.pair_ids,
        replicate_ids=target.replicate_ids,
        ancestry_ids=target.ancestry,
        sample_axes=target.sample_axes,
        independent=target.independent,
        provenance=target.provenance,
    )


def _evaluate_external(
    integrand: Any,
    samples: Any,
    /,
    *,
    key: Key[Array, ""],
    kwargs: dict[str, Any],
) -> Any:
    if isinstance(samples, PointBatch):
        if isinstance(integrand, DomainFunction):
            return integrand(samples, key=key, **kwargs)
        if isinstance(integrand, cx.Field) or not callable(integrand):
            return integrand
        raise TypeError("External PointBatch callables must be DomainFunction instances.")
    return integrand(samples, **kwargs) if callable(integrand) else integrand


def _mask_data(mask: Array | cx.Field | None, weights: cx.Field, /) -> Array:
    if mask is None:
        return jnp.ones(weights.shape, dtype=bool)
    if isinstance(mask, cx.Field):
        return jnp.asarray(mask.broadcast_like(weights).data, dtype=bool)
    return jnp.broadcast_to(jnp.asarray(mask, dtype=bool), weights.shape)


def _as_weight_field(value: Any, weights: cx.Field, /) -> cx.Field:
    if isinstance(value, cx.Field):
        return value
    data = jnp.asarray(value)
    if data.ndim == 0:
        return cx.Field(data, dims=())
    if data.ndim < weights.ndim or data.shape[: weights.ndim] != weights.shape:
        raise ValueError(
            "Raw external-measure values must begin with the complete weight shape."
        )
    return cx.Field(data, dims=weights.dims + (None,) * (data.ndim - weights.ndim))


def _canonical_named(
    value: Any,
    weights: cx.Field,
    sample_axes: tuple[str, ...],
    mask: Array | cx.Field | None,
    /,
) -> tuple[Array, Array, Array, tuple[Any, ...]]:
    field = _as_weight_field(value, weights)
    template = cx.Field(jnp.ones(weights.shape), dims=weights.dims)
    expanded = field * template
    sample_positions = tuple(expanded.dims.index(axis) for axis in sample_axes)
    retained_axes = tuple(dim for dim in weights.dims if dim not in sample_axes)
    retained_positions = tuple(expanded.dims.index(axis) for axis in retained_axes)
    used = frozenset(sample_positions + retained_positions)
    output_positions = tuple(
        position for position in range(expanded.ndim) if position not in used
    )
    permutation = sample_positions + retained_positions + output_positions
    data = jnp.transpose(jnp.asarray(expanded.data), permutation)

    weight_sample_positions = tuple(weights.dims.index(axis) for axis in sample_axes)
    weight_retained_positions = tuple(weights.dims.index(axis) for axis in retained_axes)
    weight_permutation = weight_sample_positions + weight_retained_positions
    weight_data = jnp.transpose(jnp.asarray(weights.data), weight_permutation)
    mask_data = jnp.transpose(_mask_data(mask, weights), weight_permutation)
    sample_shape = tuple(weights.shape[position] for position in weight_sample_positions)
    retained_shape = tuple(
        weights.shape[position] for position in weight_retained_positions
    )
    sample_count = prod(sample_shape)
    output_shape = tuple(
        data.shape[position]
        for position in range(len(sample_shape) + len(retained_shape), data.ndim)
    )
    output_dims = retained_axes + tuple(
        expanded.dims[position] for position in output_positions
    )
    return (
        data.reshape((sample_count,) + retained_shape + output_shape),
        weight_data.reshape((sample_count,) + retained_shape),
        mask_data.reshape((sample_count,) + retained_shape),
        output_dims,
    )


def _canonical_raw(
    value: Any,
    weights: Array,
    sample_axes: tuple[int, ...],
    mask: Array | cx.Field | None,
    /,
) -> tuple[Array, Array, Array, tuple[Any, ...]]:
    weights_ = jnp.asarray(weights, dtype=float)
    if isinstance(value, cx.Field):
        data = jnp.asarray(value.data)
        dims = value.dims
    else:
        data = jnp.asarray(value)
        dims = (None,) * data.ndim
    if data.ndim == 0:
        data = jnp.broadcast_to(data, weights_.shape)
        dims = (None,) * weights_.ndim
    if data.ndim < weights_.ndim or data.shape[: weights_.ndim] != weights_.shape:
        raise ValueError(
            "External-measure values must begin with the complete weight shape."
        )
    retained_positions = tuple(
        position for position in range(weights_.ndim) if position not in sample_axes
    )
    output_positions = tuple(range(weights_.ndim, data.ndim))
    permutation = sample_axes + retained_positions + output_positions
    canonical = jnp.transpose(data, permutation)
    weight_permutation = sample_axes + retained_positions
    weight_data = jnp.transpose(weights_, weight_permutation)
    if isinstance(mask, cx.Field):
        mask_array = jnp.asarray(mask.data, dtype=bool)
    elif mask is None:
        mask_array = jnp.ones(weights_.shape, dtype=bool)
    else:
        mask_array = jnp.asarray(mask, dtype=bool)
    mask_array = jnp.broadcast_to(mask_array, weights_.shape)
    mask_data = jnp.transpose(mask_array, weight_permutation)
    sample_shape = tuple(weights_.shape[position] for position in sample_axes)
    retained_shape = tuple(weights_.shape[position] for position in retained_positions)
    sample_count = prod(sample_shape)
    output_shape = data.shape[weights_.ndim :]
    output_dims = tuple(dims[position] for position in retained_positions) + tuple(
        dims[position] for position in output_positions
    )
    return (
        canonical.reshape((sample_count,) + retained_shape + output_shape),
        weight_data.reshape((sample_count,) + retained_shape),
        mask_data.reshape((sample_count,) + retained_shape),
        output_dims,
    )


def _canonical_weighted(
    value: Any,
    batch: WeightedSampleBatch,
    /,
) -> tuple[Array, Array, Array, tuple[Any, ...]]:
    if isinstance(batch.log_weights, cx.Field):
        if not all(isinstance(axis, str) for axis in batch.sample_axes):
            raise TypeError("Named log weights require named sample axes.")
        sample_axes = cast(tuple[str, ...], batch.sample_axes)
        return _canonical_named(
            value,
            batch.log_weights,
            sample_axes,
            batch.mask,
        )
    if not all(isinstance(axis, int) for axis in batch.sample_axes):
        raise TypeError("Raw log weights require integer sample axes.")
    sample_axes = cast(tuple[int, ...], batch.sample_axes)
    return _canonical_raw(
        value,
        batch.log_weights,
        sample_axes,
        batch.mask,
    )


def _expand_retained(value: Array, output_ndim: int, /) -> Array:
    return jnp.reshape(value, value.shape + (1,) * output_ndim)


def _mask_empty_values(values: Array, status: Array, /) -> Array:
    output_ndim = values.ndim - status.ndim
    available = _expand_retained(
        status != int(IntegrationStatus.NO_VALID_SAMPLES), output_ndim
    )
    return jnp.where(available, values, jnp.asarray(jnp.nan, dtype=values.dtype))


def _finite_values(values: Array, active: Array, /) -> Array:
    output_ndim = values.ndim - active.ndim
    finite = jnp.isfinite(values)
    if output_ndim:
        finite = jnp.all(
            finite,
            axis=tuple(range(active.ndim, values.ndim)),
        )
    return jnp.all(~active | finite, axis=0)


def _cast_precision(value: Any, dtype: Any | None, /) -> Array:
    array = jnp.asarray(value)
    if dtype is None or not jnp.issubdtype(array.dtype, jnp.inexact):
        return array
    real_dtype = real_precision_dtype_name(dtype)
    target = (
        complex_precision_dtype(real_dtype)
        if jnp.issubdtype(array.dtype, jnp.complexfloating)
        else real_dtype
    )
    return array.astype(target)


def _error_norm(error: Array, /) -> Array:
    return jnp.max(jnp.asarray(error))


def integrate_weighted_samples(
    integrand: Any,
    target: WeightedSampleTarget,
    batch: WeightedSampleBatch,
    /,
    *,
    normalized: bool | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
    evaluation_dtype: Any | None = None,
    accumulation_dtype: Any | None = None,
) -> IntegrationEstimate:
    """Reduce masked log-weighted samples over explicit sample axes."""
    callback_kwargs = {} if kwargs is None else kwargs
    evaluated = _evaluate_external(
        integrand,
        batch.samples,
        key=key,
        kwargs=callback_kwargs,
    )
    values, log_weights, included, output_dims = _canonical_weighted(evaluated, batch)
    values = _cast_precision(values, evaluation_dtype)
    output_ndim = values.ndim - log_weights.ndim
    sample_count = int(log_weights.shape[0])
    accumulator = LogWeightedAccumulator.from_values(
        values,
        log_weights,
        sample_axes=0,
        mask=included,
        accumulation_dtype=accumulation_dtype,
    )
    normalized_ = target.normalized if normalized is None else bool(normalized)
    normalizer = accumulator.raw_normalizer
    if normalized_:
        estimate = accumulator.normalized_mean
    elif batch.target_mass is not None:
        mass = jnp.broadcast_to(jnp.asarray(batch.target_mass), normalizer.shape)
        estimate = _expand_retained(mass, output_ndim) * accumulator.normalized_mean
    else:
        estimate = accumulator.raw_mean

    standard_error = None
    normalizer_standard_error = None
    if batch.independent and sample_count > 1:
        normalizer_standard_error = accumulator.raw_normalizer_standard_error
        if normalized_:
            standard_error = accumulator.normalized_standard_error
        elif batch.target_mass is not None:
            mass = jnp.broadcast_to(jnp.asarray(batch.target_mass), normalizer.shape)
            standard_error = (
                _expand_retained(mass, output_ndim)
                * accumulator.normalized_standard_error
            )
        else:
            standard_error = accumulator.raw_standard_error

    admissible = jnp.isfinite(log_weights) | jnp.isneginf(log_weights)
    active = included & jnp.isfinite(log_weights)
    included_count = jnp.sum(included, axis=0, dtype=jnp.int32)
    weight_inputs_valid = jnp.all(~included | admissible, axis=0)
    positive_weight = jnp.any(active, axis=0)
    value_inputs_valid = _finite_values(values, active)
    status = jnp.where(
        included_count == 0,
        int(IntegrationStatus.NO_VALID_SAMPLES),
        jnp.where(
            ~(weight_inputs_valid & positive_weight),
            int(IntegrationStatus.INVALID_WEIGHTS),
            jnp.where(
                value_inputs_valid,
                int(IntegrationStatus.CONVERGED),
                int(IntegrationStatus.NONFINITE_INTEGRAND),
            ),
        ),
    )
    estimate_finite = jnp.isfinite(estimate)
    if output_ndim:
        estimate_finite = jnp.all(
            estimate_finite,
            axis=tuple(range(status.ndim, estimate.ndim)),
        )
    requires_estimated_mass = not normalized_ and batch.target_mass is None
    reduction_finite = estimate_finite & (
        jnp.isfinite(normalizer) if requires_estimated_mass else True
    )
    reduction_failure = jnp.where(
        requires_estimated_mass & ~jnp.isfinite(normalizer),
        int(IntegrationStatus.INVALID_WEIGHTS),
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    status = jnp.where(
        (status != int(IntegrationStatus.CONVERGED)) | reduction_finite,
        status,
        reduction_failure,
    )
    if batch.support_valid is not None:
        support_valid = jnp.broadcast_to(
            jnp.asarray(batch.support_valid, dtype=bool), status.shape
        )
        status = jnp.where(
            support_valid,
            status,
            int(IntegrationStatus.PROPOSAL_SUPPORT_FAILURE),
        )
    estimate = _mask_empty_values(estimate, status)
    moments = weighted_diagnostics(
        accumulator,
        log_weights,
        sample_axes=0,
        mask=included,
    )
    evaluations = jnp.full(status.shape, sample_count, dtype=jnp.int32)
    diagnostics = WeightedSampleDiagnostics(
        status=status,
        num_evaluations=evaluations,
        active_samples=included_count,
        standard_error=standard_error,
        normalizer_estimate=normalizer,
        normalizer_standard_error=normalizer_standard_error,
        weights=moments,
        stratum_ids=(
            None if batch.stratum_ids is None else jnp.asarray(batch.stratum_ids)
        ),
        pair_ids=(
            None
            if batch.pair_ids is None
            else jnp.asarray(
                batch.pair_ids.data
                if isinstance(batch.pair_ids, cx.Field)
                else batch.pair_ids
            )
        ),
        replicate_ids=(
            None
            if batch.replicate_ids is None
            else jnp.asarray(
                batch.replicate_ids.data
                if isinstance(batch.replicate_ids, cx.Field)
                else batch.replicate_ids
            )
        ),
        ancestry_ids=(
            None
            if batch.ancestry_ids is None
            else jnp.asarray(
                batch.ancestry_ids.data
                if isinstance(batch.ancestry_ids, cx.Field)
                else batch.ancestry_ids
            )
        ),
        normalized=normalized_,
        independent=batch.independent,
    )
    reported_error = None if standard_error is None else _error_norm(standard_error)
    method = "importance" if batch.provenance.startswith("importance:") else "weighted"
    return IntegrationEstimate(
        cx.Field(estimate, dims=output_dims),
        status=status,
        num_evaluations=evaluations,
        error_estimate=reported_error,
        error_kind=(
            "weighted-iid-standard-error" if reported_error is not None else None
        ),
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            method,
            "weighted-samples",
            batch.provenance,
        ),
    )


def _separable_value_field(
    value: Any,
    batch: SeparableIntegrationBatch,
    /,
) -> cx.Field:
    if isinstance(value, cx.Field):
        return value
    data = jnp.asarray(value)
    if data.ndim == 0:
        return cx.Field(data, dims=())
    sample_shape = tuple(batch.weights_by_axis[axis].shape[0] for axis in batch.axes)
    if data.ndim < len(sample_shape) or data.shape[: len(sample_shape)] != sample_shape:
        raise ValueError(
            "Raw separable-measure values must begin with the complete sample shape."
        )
    return cx.Field(
        data,
        dims=batch.axes + (None,) * (data.ndim - len(sample_shape)),
    )


def _integrate_separable_discrete(
    evaluated: Any,
    target: DiscreteMeasureTarget,
    batch: SeparableIntegrationBatch,
    /,
    *,
    evaluation_dtype: Any | None,
    accumulation_dtype: Any | None,
) -> IntegrationEstimate:
    base_included = (
        cx.Field(jnp.asarray(True), dims=()) if batch.mask is None else batch.mask
    )
    included = base_included
    for axis in batch.axes:
        if axis not in included.named_dims:
            weight = batch.weights_by_axis[axis]
            included = included * cx.Field(
                jnp.ones(weight.shape, dtype=bool),
                dims=(axis,),
            )
    admissible = cx.Field(jnp.asarray(True), dims=())
    positive = cx.Field(jnp.asarray(True), dims=())
    for axis in batch.axes:
        weight = batch.weights_by_axis[axis]
        weight_data = jnp.asarray(weight.data)
        finite_nonnegative = jnp.isfinite(weight_data) & (weight_data >= 0.0)
        admissible = admissible * cx.Field(finite_nonnegative, dims=(axis,))
        positive = positive * cx.Field(
            finite_nonnegative & (weight_data > 0.0),
            dims=(axis,),
        )
    if batch.coupled_weight is not None:
        coupled = batch.coupled_weight
        coupled_data = jnp.asarray(coupled.data)
        finite_nonnegative = jnp.isfinite(coupled_data) & (coupled_data >= 0.0)
        admissible = admissible * cx.Field(
            finite_nonnegative,
            dims=coupled.dims,
        )
        positive = positive * cx.Field(
            finite_nonnegative & (coupled_data > 0.0),
            dims=coupled.dims,
        )
    active = included * admissible * positive
    field = _separable_value_field(evaluated, batch)
    field = cx.Field(
        _cast_precision(field.data, evaluation_dtype),
        dims=field.dims,
    )
    expanded = field
    for axis in batch.axes:
        if axis not in expanded.named_dims:
            weight = batch.weights_by_axis[axis]
            expanded = expanded * cx.Field(
                jnp.ones(weight.shape),
                dims=(axis,),
            )
    active_values = jnp.asarray(active.broadcast_like(expanded).data, dtype=bool)
    expanded_data = _cast_precision(expanded.data, accumulation_dtype)
    safe_values = cx.Field(
        jnp.where(active_values, expanded_data, 0),
        dims=expanded.dims,
    )
    numerator = safe_values
    mass = cx.Field(
        _cast_precision(base_included.data, accumulation_dtype),
        dims=base_included.dims,
    )
    if batch.coupled_weight is not None:
        coupled = batch.coupled_weight
        coupled_data = jnp.asarray(coupled.data)
        safe_coupled = cx.Field(
            jnp.where(
                jnp.isfinite(coupled_data) & (coupled_data >= 0.0),
                coupled_data,
                0.0,
            ),
            dims=coupled.dims,
        )
        numerator = numerator * safe_coupled
        mass = mass * safe_coupled
    for axis in batch.axes:
        weight = batch.weights_by_axis[axis]
        weight_data = jnp.asarray(weight.data)
        safe_weight = cx.Field(
            jnp.where(
                jnp.isfinite(weight_data) & (weight_data >= 0.0),
                weight_data,
                0.0,
            ),
            dims=(axis,),
        )
        numerator = sum_over(
            numerator * safe_weight,
            axis,
            accumulation_dtype=accumulation_dtype,
        )
        mass = sum_over(
            mass * safe_weight,
            axis,
            accumulation_dtype=accumulation_dtype,
        )
    mass_data = jnp.asarray(mass.data)
    estimate_field = numerator / mass if target.normalized else numerator
    estimate = jnp.asarray(estimate_field.data)
    included_data = jnp.asarray(included.data, dtype=bool)
    included_count = jnp.sum(included_data, dtype=jnp.int32)
    admissible_data = jnp.asarray(
        admissible.broadcast_like(included).data,
        dtype=bool,
    )
    weight_inputs_valid = jnp.all(~included_data | admissible_data)
    positive_weight = jnp.any(jnp.asarray(active.data, dtype=bool))
    value_inputs_valid = jnp.all(~active_values | jnp.isfinite(expanded_data))
    status = jnp.where(
        included_count == 0,
        int(IntegrationStatus.NO_VALID_SAMPLES),
        jnp.where(
            ~(weight_inputs_valid & positive_weight),
            int(IntegrationStatus.INVALID_WEIGHTS),
            jnp.where(
                value_inputs_valid,
                int(IntegrationStatus.CONVERGED),
                int(IntegrationStatus.NONFINITE_INTEGRAND),
            ),
        ),
    )
    status = jnp.where(
        (status != int(IntegrationStatus.CONVERGED)) | jnp.all(jnp.isfinite(estimate)),
        status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    estimate = _mask_empty_values(estimate, status)
    sample_count = prod(int(batch.weights_by_axis[axis].shape[0]) for axis in batch.axes)
    evaluations = jnp.asarray(sample_count, dtype=jnp.int32)
    target_mass = batch.target_mass
    if target_mass is None:
        target_mass = mass_data
    diagnostics = FixedQuadratureDiagnostics(
        status=status,
        num_evaluations=evaluations,
        target_mass=target_mass,
        rule=batch.provenance,
    )
    return IntegrationEstimate(
        cx.Field(estimate, dims=estimate_field.dims),
        status=status,
        num_evaluations=evaluations,
        error_estimate=None,
        error_kind=None,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance("fixed", "discrete", batch.provenance),
    )


def integrate_discrete_measure(
    integrand: Any,
    target: DiscreteMeasureTarget,
    batch: PointIntegrationBatch | SeparableIntegrationBatch,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
    evaluation_dtype: Any | None = None,
    accumulation_dtype: Any | None = None,
) -> IntegrationEstimate:
    """Reduce an externally supplied deterministic nonnegative measure."""
    callback_kwargs = {} if kwargs is None else kwargs
    evaluated = _evaluate_external(
        integrand,
        batch.points,
        key=key,
        kwargs=callback_kwargs,
    )
    if isinstance(batch, SeparableIntegrationBatch):
        return _integrate_separable_discrete(
            evaluated,
            target,
            batch,
            evaluation_dtype=evaluation_dtype,
            accumulation_dtype=accumulation_dtype,
        )
    weights = batch.weights
    values, canonical_weights, included, output_dims = _canonical_named(
        evaluated,
        weights,
        batch.axes,
        batch.mask,
    )
    values = _cast_precision(values, evaluation_dtype)
    canonical_weights = _cast_precision(canonical_weights, accumulation_dtype)
    values = _cast_precision(values, accumulation_dtype)
    output_ndim = values.ndim - canonical_weights.ndim
    sample_count = int(canonical_weights.shape[0])
    admissible = jnp.isfinite(canonical_weights) & (canonical_weights >= 0.0)
    active = included & admissible & (canonical_weights > 0.0)
    safe_weights = jnp.where(active, canonical_weights, 0.0)
    safe_values = jnp.where(_expand_retained(active, output_ndim), values, 0)
    weighted_sum = jnp.sum(
        _expand_retained(safe_weights, output_ndim) * safe_values,
        axis=0,
    )
    mass = jnp.sum(safe_weights, axis=0)
    estimate = (
        weighted_sum / _expand_retained(mass, output_ndim)
        if target.normalized
        else weighted_sum
    )
    included_count = jnp.sum(included, axis=0, dtype=jnp.int32)
    weight_inputs_valid = jnp.all(~included | admissible, axis=0)
    positive_weight = jnp.any(active, axis=0)
    value_inputs_valid = _finite_values(values, active)
    status = jnp.where(
        included_count == 0,
        int(IntegrationStatus.NO_VALID_SAMPLES),
        jnp.where(
            ~(weight_inputs_valid & positive_weight),
            int(IntegrationStatus.INVALID_WEIGHTS),
            jnp.where(
                value_inputs_valid,
                int(IntegrationStatus.CONVERGED),
                int(IntegrationStatus.NONFINITE_INTEGRAND),
            ),
        ),
    )
    estimate_finite = jnp.isfinite(estimate)
    if output_ndim:
        estimate_finite = jnp.all(
            estimate_finite,
            axis=tuple(range(status.ndim, estimate.ndim)),
        )
    status = jnp.where(
        (status != int(IntegrationStatus.CONVERGED)) | estimate_finite,
        status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    estimate = _mask_empty_values(estimate, status)
    evaluations = jnp.full(status.shape, sample_count, dtype=jnp.int32)
    target_mass = batch.target_mass
    if target_mass is None:
        target_mass = mass
    diagnostics = FixedQuadratureDiagnostics(
        status=status,
        num_evaluations=evaluations,
        target_mass=target_mass,
        rule=batch.provenance,
    )
    return IntegrationEstimate(
        cx.Field(estimate, dims=output_dims),
        status=status,
        num_evaluations=evaluations,
        error_estimate=None,
        error_kind=None,
        diagnostics=diagnostics,
        provenance=IntegrationProvenance("fixed", "discrete", batch.provenance),
    )


__all__ = [
    "integrate_discrete_measure",
    "integrate_weighted_samples",
    "materialize_discrete_target",
    "materialize_weighted_target",
]
