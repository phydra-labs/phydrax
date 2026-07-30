#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from ..models.core._operator import (
    FunctionSamples,
    OperatorBatch,
    OperatorFieldBatch,
    OperatorPrediction,
    OperatorTargetBatch,
)
from ..models.core._operator_task import OperatorTask
from ._normalization import OperatorNormalizationPolicy


def samples_with_values(
    samples: FunctionSamples,
    values: Any,
    /,
) -> FunctionSamples:
    """Replace sample values while preserving physical geometry metadata."""
    return FunctionSamples(
        values=values,
        axes=samples.axes,
        coordinates=samples.coordinates,
        quadrature_weights=samples.quadrature_weights,
        mask=samples.mask,
        topology=samples.topology,
    )


def nondimensionalize_batch(
    batch: OperatorBatch,
    task: OperatorTask,
    /,
) -> OperatorBatch:
    """Map physical source values into task execution units."""
    inputs = dict(batch.inputs)
    for field in task.source_fields:
        assert field.source_name is not None
        if field.source_name not in inputs:
            continue
        samples = inputs[field.source_name]
        if samples.values is None:
            raise ValueError(f"Source {field.source_name!r} has no values.")
        inputs[field.source_name] = samples_with_values(
            samples,
            field.nondimensionalize(jnp.asarray(samples.values)),
        )
    return OperatorBatch(
        inputs=inputs,
        queries=batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def nondimensionalize_targets(
    targets: OperatorTargetBatch,
    task: OperatorTask,
    /,
) -> OperatorTargetBatch:
    """Map physical target values into task execution units."""
    if not targets.fields:
        return OperatorTargetBatch(
            {},
            case_axes=targets.case_axes,
            case_shape=targets.case_shape,
        )
    expected = tuple(field.name for field in task.target_fields)
    if set(targets.fields) != set(expected):
        raise ValueError(
            "Operator target names must match the task; "
            f"expected {expected!r}, got {tuple(targets.fields)!r}."
        )
    by_name = task.field_by_name
    return OperatorTargetBatch(
        {
            name: OperatorFieldBatch(
                by_name[name].nondimensionalize(field.values),
                query_name=field.query_name,
                spec=field.spec,
            )
            for name, field in targets.fields.items()
        },
        case_axes=targets.case_axes,
        case_shape=targets.case_shape,
    )


def physicalize_prediction(
    prediction: OperatorPrediction,
    physical_batch: OperatorBatch,
    task: OperatorTask,
    output_field_map: Mapping[str, str],
    normalization: OperatorNormalizationPolicy | None,
    /,
) -> OperatorPrediction:
    """Map model-named execution output into task-named physical output."""
    if set(prediction.fields) != set(output_field_map):
        raise ValueError(
            "Model prediction fields do not match the output field map; "
            f"expected {tuple(output_field_map)!r}, got {tuple(prediction.fields)!r}."
        )
    model_name_by_target = {
        target_name: model_name for model_name, target_name in output_field_map.items()
    }
    fields: dict[str, OperatorFieldBatch] = {}
    for target in task.target_fields:
        assert target.output_spec is not None
        assert target.query_name is not None
        raw_field = prediction.field(model_name_by_target[target.name])
        if raw_field.query_name != target.query_name:
            raise ValueError(
                f"Model output {target.name!r} is bound to query "
                f"{raw_field.query_name!r}, expected {target.query_name!r}."
            )
        values = raw_field.values
        if normalization is not None:
            if target.name not in normalization.targets:
                raise KeyError(f"Missing normalizer for target field {target.name!r}.")
            values = normalization.targets[target.name].denormalize(values)
        values = target.dimensionalize(values)
        values = target.output_spec.validate(
            values,
            physical_batch,
            query_name=target.query_name,
        )
        fields[target.name] = OperatorFieldBatch(
            values,
            query_name=target.query_name,
            spec=target.output_spec,
        )
    physical = OperatorPrediction(
        fields,
        physical_batch.queries,
        case_axes=physical_batch.case_axes,
        case_shape=physical_batch.case_shape,
    )
    task.validate_prediction(physical)
    return physical


def executionize_prediction(
    prediction: OperatorPrediction,
    template: OperatorPrediction,
    execution_batch: OperatorBatch,
    task: OperatorTask,
    output_field_map: Mapping[str, str],
    normalization: OperatorNormalizationPolicy | None,
    /,
) -> OperatorPrediction:
    """Map task-named physical output back into model execution coordinates."""
    task.validate_prediction(prediction)
    fields: dict[str, OperatorFieldBatch] = {}
    by_name = task.field_by_name
    for model_name, target_name in output_field_map.items():
        target = by_name[target_name]
        physical_field = prediction.field(target_name)
        template_field = template.field(model_name)
        values = target.nondimensionalize(physical_field.values)
        if normalization is not None:
            if target_name not in normalization.targets:
                raise KeyError(f"Missing normalizer for target field {target_name!r}.")
            values = normalization.targets[target_name].normalize(values)
        query = execution_batch.query(physical_field.query_name)
        mask = query.mask_array(case_shape=execution_batch.case_shape)
        trailing = (1,) * (values.ndim - mask.ndim)
        values = jnp.where(
            mask.reshape(mask.shape + trailing),
            values,
            jnp.zeros((), dtype=values.dtype),
        ).astype(template_field.values.dtype)
        fields[model_name] = OperatorFieldBatch(
            values,
            query_name=template_field.query_name,
            spec=template_field.spec,
        )
    return OperatorPrediction(
        fields,
        execution_batch.queries,
        case_axes=execution_batch.case_axes,
        case_shape=execution_batch.case_shape,
    )


__all__ = [
    "executionize_prediction",
    "nondimensionalize_batch",
    "nondimensionalize_targets",
    "physicalize_prediction",
    "samples_with_values",
]
