#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._model import AbstractArrayModel, ModelBinding
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._batch import MLBatch
from .._contracts import FitResult, GradientContract
from .._schema import FeatureSchema
from .._sparse_features import FeatureArray, SparseFeatures


@runtime_checkable
class SchemaTransformModel(Protocol):
    """Structural schema contract implemented by fitted feature transforms."""

    input_schema: FeatureSchema
    output_schema: FeatureSchema


@runtime_checkable
class BatchTransformModel(Protocol):
    """Structural contract for transforms that require a complete selected batch."""

    def transform_batch(self, batch: MLBatch, /, *, key: Any = None) -> MLBatch: ...


@runtime_checkable
class ReversibleTransformModel(Protocol):
    """Structural contract for a fitted transform with a true mathematical inverse."""

    def inverse_transform(self, values: Any, /, *, key: Any = None) -> Array: ...


class CompositionProvenance(StrictModule, NonTrainableState):
    """Frozen child fit results retained outside trainable model partitions."""

    names: tuple[str, ...] = eqx.field(static=True)
    results: tuple[FitResult, ...]

    def __init__(self, names: Sequence[str], results: Sequence[FitResult], /):
        names_ = tuple(str(name) for name in names)
        results_ = tuple(results)
        if len(names_) != len(results_):
            raise ValueError("Composition provenance names and results must align.")
        self.names = names_
        self.results = results_


class CompositionDiagnostics(StrictModule):
    """Ordered child diagnostics retained by an immutable composition fit."""

    names: tuple[str, ...] = eqx.field(static=True)
    children: tuple[Any, ...]
    methods: tuple[str, ...] = eqx.field(static=True)
    child_valid: tuple[Array, ...]
    child_status: tuple[Array, ...]
    gradient_contracts: tuple[GradientContract, ...]
    valid: Array
    status: Array

    def __init__(
        self,
        names: Sequence[str],
        results: Sequence[FitResult],
        /,
        *,
        valid: Any,
        status: Any,
    ):
        names_ = tuple(str(name) for name in names)
        results_ = tuple(results)
        if len(names_) != len(results_):
            raise ValueError("Composition diagnostic names and results must align.")
        self.names = names_
        self.children = tuple(result.diagnostics for result in results_)
        self.methods = tuple(result.method for result in results_)
        self.child_valid = tuple(result.valid for result in results_)
        self.child_status = tuple(result.status for result in results_)
        self.gradient_contracts = tuple(result.gradient_contract for result in results_)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)


def _normalize_recipe_specs(
    specs: Sequence[tuple[str, Any]],
    /,
    *,
    kind: str,
    recipe_type: type,
) -> tuple[tuple[str, Any], ...]:
    specs_ = tuple(specs)
    if not specs_:
        raise ValueError(f"{kind} requires at least one named recipe.")
    names: list[str] = []
    normalized: list[tuple[str, Any]] = []
    for spec in specs_:
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise TypeError(f"Each {kind} entry must be a (name, recipe) tuple.")
        name, recipe = spec
        name_ = str(name)
        if not name_:
            raise ValueError(f"{kind} names must be non-empty.")
        if not isinstance(recipe, recipe_type):
            raise TypeError(f"{kind} entry {name_!r} is not an AbstractRecipe.")
        names.append(name_)
        normalized.append((name_, recipe))
    if len(set(names)) != len(names):
        raise ValueError(f"{kind} names must be unique.")
    return tuple(normalized)


def _split_key(key: Any, count: int, /) -> tuple[Any, ...]:
    count_ = int(count)
    if count_ < 0:
        raise ValueError("Key split count cannot be negative.")
    if key is None:
        return (None,) * count_
    if count_ == 0:
        return ()
    return tuple(jax.random.split(key, count_))


def _feature_width(size: Any, /, *, role: str) -> int:
    if size == "scalar":
        return 1
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"{role} size must be positive.")
        return int(size)
    if isinstance(size, tuple) and len(size) == 1:
        width = int(size[0])
        if width <= 0:
            raise ValueError(f"{role} size must be positive.")
        return width
    raise ValueError(
        f"{role} must be scalar or one-dimensional for an ML feature axis; got {size!r}."
    )


def _validate_model_input(
    model: AbstractArrayModel,
    feature_count: int,
    /,
    *,
    schema: FeatureSchema | None = None,
) -> None:
    expected = _feature_width(model.in_size, role="Model input")
    if expected != int(feature_count):
        raise ValueError(
            f"Fitted model expects {expected} features but the composed batch has "
            f"{feature_count}."
        )
    if isinstance(model, SchemaTransformModel) and schema is not None:
        declared = model.input_schema
        if (
            declared.names != schema.names
            or declared.kinds != schema.kinds
            or declared.layout_id != schema.layout_id
        ):
            raise ValueError(
                "Fitted transform input_schema does not match the selected batch schema."
            )


def _model_output_schema(
    model: AbstractArrayModel,
    input_schema: FeatureSchema,
    /,
) -> FeatureSchema:
    width = _feature_width(model.out_size, role="Transform output")
    if isinstance(model, SchemaTransformModel):
        schema = model.output_schema
        if len(schema.names) != width:
            raise ValueError(
                "Fitted transform output_schema does not match its declared out_size."
            )
        return schema
    if width == len(input_schema.names):
        return input_schema
    return FeatureSchema.anonymous(width)


def _prepare_input(values: Any, in_size: Any, /) -> Any:
    if isinstance(values, SparseFeatures):
        return values
    array = jnp.asarray(values)
    if in_size == "scalar":
        if array.ndim == 0:
            return array
        if int(array.shape[-1]) != 1:
            raise ValueError("Scalar model input requires a singleton feature axis.")
        return jnp.squeeze(array, axis=-1)
    return array


def _call_pointwise(
    model: AbstractArrayModel,
    values: Array,
    /,
    *,
    key: Any,
) -> Array:
    array = jnp.asarray(values)
    if array.ndim < 1:
        raise ValueError("Pointwise feature transforms require a feature axis.")
    width = int(array.shape[-1])
    _validate_model_input(model, width)
    leading = tuple(int(size) for size in array.shape[:-1])
    flat = array.reshape((-1, width))
    count = int(flat.shape[0])
    if key is None:
        mapped = jax.vmap(
            lambda row: model(_prepare_input(row, model.in_size), key=None)
        )(flat)
    else:
        keys = jax.random.split(key, count)
        mapped = jax.vmap(
            lambda row, point_key: model(
                _prepare_input(row, model.in_size), key=point_key
            )
        )(flat, keys)
    return jnp.asarray(mapped).reshape(leading + tuple(mapped.shape[1:]))


def _call_blockwise_cases(
    model: AbstractArrayModel,
    values: FeatureArray,
    case_shape: tuple[int, ...],
    /,
    *,
    key: Any,
) -> Any:
    if isinstance(values, SparseFeatures):
        if case_shape:
            raise TypeError(
                "Case-vmapped blockwise sparse transforms are unsupported; no dense "
                "fallback is performed."
            )
        return model(_prepare_input(values, model.in_size), key=key)
    array = jnp.asarray(values)
    if not case_shape:
        return model(_prepare_input(array, model.in_size), key=key)
    sample_count, width = int(array.shape[-2]), int(array.shape[-1])
    flat = array.reshape((-1, sample_count, width))
    count = int(flat.shape[0])
    if key is None:
        mapped = jax.vmap(
            lambda block: model(_prepare_input(block, model.in_size), key=None)
        )(flat)
    else:
        keys = jax.random.split(key, count)
        mapped = jax.vmap(
            lambda block, case_key: model(
                _prepare_input(block, model.in_size), key=case_key
            )
        )(flat, keys)
    return jnp.asarray(mapped).reshape(case_shape + tuple(mapped.shape[1:]))


def _canonical_feature_output(
    values: Any,
    /,
    *,
    leading_shape: tuple[int, ...],
    out_size: Any,
) -> FeatureArray:
    width = _feature_width(out_size, role="Transform output")
    if isinstance(values, SparseFeatures):
        if values.case_shape + (values.sample_count,) != leading_shape:
            raise ValueError(
                "Sparse transform output does not preserve case/sample axes."
            )
        if values.feature_count != width:
            raise ValueError(
                "Sparse transform output width does not match the model out_size."
            )
        return values
    array = jnp.asarray(values)
    if out_size == "scalar" and tuple(array.shape) == leading_shape:
        array = jnp.expand_dims(array, axis=-1)
    expected = leading_shape + (width,)
    if tuple(int(size) for size in array.shape) != expected:
        raise ValueError(
            "Feature transform must preserve case/sample axes and return one feature "
            f"axis of width {width}; got {array.shape}, expected {expected}."
        )
    return array


def _generic_output_mask(batch: MLBatch, output: FeatureArray, /) -> Array | None:
    if isinstance(output, SparseFeatures):
        return None
    row_valid = jnp.all(batch.feature_mask, axis=-1, keepdims=True)
    return jnp.broadcast_to(row_valid, output.shape)


def _feature_only_batch(batch: MLBatch, /) -> MLBatch:
    """Remove target values before invoking an already-fitted feature transform."""
    return MLBatch(
        batch.features,
        feature_mask=(
            None if isinstance(batch.features, SparseFeatures) else batch.feature_mask
        ),
        sample_mask=batch.sample_mask,
        sample_weight=batch.sample_weight,
        measure_weight=batch.measure_weight,
        groups=batch.groups,
        feature_schema=batch.feature_schema,
        target_schema=batch.target_schema,
    )


def _preserve_batch_metadata(source: MLBatch, transformed: MLBatch, /) -> MLBatch:
    if transformed.case_shape != source.case_shape:
        raise ValueError("A batch transform cannot change case axes.")
    if transformed.sample_count != source.sample_count:
        raise ValueError("A batch transform cannot change the selected sample axis.")
    return source.with_features(
        transformed.features,
        feature_schema=transformed.feature_schema,
        feature_mask=(
            None
            if isinstance(transformed.features, SparseFeatures)
            else transformed.feature_mask
        ),
    )


def _transform_batch(
    model: AbstractArrayModel,
    batch: MLBatch,
    /,
    *,
    key: Any,
) -> MLBatch:
    _validate_model_input(model, batch.feature_count, schema=batch.feature_schema)
    if isinstance(model, BatchTransformModel):
        transformed = model.transform_batch(_feature_only_batch(batch), key=key)
        if not isinstance(transformed, MLBatch):
            raise TypeError("transform_batch must return an MLBatch.")
        return _preserve_batch_metadata(batch, transformed)

    binding = model.input_binding()
    if binding.input_mode != "flat":
        raise TypeError("ML feature composition requires flat model inputs.")
    if binding.batch_mode == "axis":
        raise TypeError("Axis-bound models cannot be used as ML feature transforms.")
    if binding.batch_mode == "pointwise":
        if isinstance(batch.features, SparseFeatures):
            raise TypeError(
                "Pointwise transforms of SparseFeatures are unsupported; no dense "
                "fallback is performed."
            )
        output = _call_pointwise(model, batch.features, key=key)
    else:
        output = _call_blockwise_cases(model, batch.features, batch.case_shape, key=key)
    features = _canonical_feature_output(
        output,
        leading_shape=batch.case_shape + (batch.sample_count,),
        out_size=model.out_size,
    )
    return batch.with_features(
        features,
        feature_schema=_model_output_schema(model, batch.feature_schema),
        feature_mask=_generic_output_mask(batch, features),
    )


def _transform_values(
    model: AbstractArrayModel,
    values: FeatureArray,
    input_schema: FeatureSchema,
    /,
    *,
    key: Any,
    composed_blockwise: bool,
) -> FeatureArray:
    width = (
        values.feature_count
        if isinstance(values, SparseFeatures)
        else int(values.shape[-1])
    )
    _validate_model_input(model, width, schema=input_schema)
    binding = model.input_binding()
    if isinstance(model, BatchTransformModel) and binding.batch_mode == "blockwise":
        if not composed_blockwise:
            raise TypeError("Batch-dependent transforms require a blockwise binding.")
        batch = MLBatch(values, feature_schema=input_schema)
        transformed = model.transform_batch(batch, key=key)
        if not isinstance(transformed, MLBatch):
            raise TypeError("transform_batch must return an MLBatch.")
        return _preserve_batch_metadata(batch, transformed).features

    if binding.input_mode != "flat":
        raise TypeError("ML feature composition requires flat model inputs.")
    if binding.batch_mode == "axis":
        raise TypeError("Axis-bound models cannot be used as ML feature transforms.")
    if isinstance(values, SparseFeatures):
        if binding.batch_mode != "blockwise":
            raise TypeError(
                "Pointwise transforms of SparseFeatures are unsupported; no dense "
                "fallback is performed."
            )
        output = model(values, key=key)
        leading = values.case_shape + (values.sample_count,)
    else:
        array = jnp.asarray(values)
        leading = tuple(int(size) for size in array.shape[:-1])
        if composed_blockwise and binding.batch_mode == "pointwise":
            output = _call_pointwise(model, array, key=key)
        else:
            output = model(_prepare_input(array, model.in_size), key=key)
    return _canonical_feature_output(
        output, leading_shape=leading, out_size=model.out_size
    )


def _predict_values(
    model: AbstractArrayModel,
    values: FeatureArray,
    /,
    *,
    key: Any,
    composed_blockwise: bool,
) -> Any:
    width = (
        values.feature_count
        if isinstance(values, SparseFeatures)
        else int(values.shape[-1])
    )
    _validate_model_input(model, width)
    binding = model.input_binding()
    if binding.input_mode != "flat":
        raise TypeError("ML composition requires flat model inputs.")
    if binding.batch_mode == "axis":
        raise TypeError("Axis-bound models cannot be used in ML composition.")
    if isinstance(values, SparseFeatures):
        if binding.batch_mode != "blockwise":
            raise TypeError(
                "Pointwise prediction from SparseFeatures is unsupported; no dense "
                "fallback is performed."
            )
        return model(values, key=key)
    array = jnp.asarray(values)
    if composed_blockwise and binding.batch_mode == "pointwise":
        return _call_pointwise(model, array, key=key)
    return model(_prepare_input(array, model.in_size), key=key)


def _composition_binding(
    models: Sequence[AbstractArrayModel],
    /,
) -> ModelBinding:
    models_ = tuple(models)
    blockwise = False
    for model in models_:
        binding = model.input_binding()
        if binding.input_mode != "flat":
            raise TypeError("ML composition requires flat model inputs.")
        if binding.batch_mode == "axis":
            raise TypeError("Axis-bound models cannot be used in ML composition.")
        blockwise = blockwise or binding.batch_mode == "blockwise"
    return ModelBinding.blockwise("flat") if blockwise else ModelBinding.pointwise("flat")


def _combine_results(
    results: Sequence[FitResult],
    /,
) -> tuple[Array, Array, GradientContract]:
    results_ = tuple(results)
    if not results_:
        raise ValueError("Cannot combine an empty set of fit results.")
    valid = jnp.asarray(True)
    status = jnp.asarray(0, dtype=jnp.int32)
    for result in results_:
        valid = valid & result.valid
        status = jnp.where(status == 0, result.status, status)

    level_order = {"none": 0, "conditional": 1, "almost-everywhere": 2, "smooth": 3}

    def minimum_level(select: Callable[[GradientContract], str], /) -> Any:
        return min(
            (select(result.gradient_contract) for result in results_),
            key=level_order.__getitem__,
        )

    modes = tuple(result.gradient_contract.fit_mode for result in results_)
    fit_mode = modes[0] if all(mode == modes[0] for mode in modes) else "stopped"
    nondifferentiable: list[str] = []
    conditions: list[str] = []
    for result in results_:
        for name in result.gradient_contract.nondifferentiable_outputs:
            if name not in nondifferentiable:
                nondifferentiable.append(name)
        for condition in result.gradient_contract.conditions:
            if condition not in conditions:
                conditions.append(condition)
    if fit_mode == "stopped" and any(mode != "stopped" for mode in modes):
        conditions.append(
            "Child fits use different differentiation modes; the composite fit mode "
            "is conservatively declared stopped."
        )
    contract = GradientContract(
        prediction_inputs=minimum_level(lambda contract: contract.prediction_inputs),
        prediction_parameters=minimum_level(
            lambda contract: contract.prediction_parameters
        ),
        fit_features=minimum_level(lambda contract: contract.fit_features),
        fit_targets=minimum_level(lambda contract: contract.fit_targets),
        fit_weights=minimum_level(lambda contract: contract.fit_weights),
        fit_hyperparameters=minimum_level(lambda contract: contract.fit_hyperparameters),
        fit_mode=fit_mode,
        nondifferentiable_outputs=tuple(nondifferentiable),
        conditions=tuple(conditions),
    )
    return jnp.asarray(valid, dtype=bool), jnp.asarray(status, dtype=jnp.int32), contract


def _prefixed_schema(name: str, schema: FeatureSchema, /) -> FeatureSchema:
    return FeatureSchema(
        tuple(f"{name}__{feature_name}" for feature_name in schema.names),
        kinds=schema.kinds,
        layout_id=(f"{name}:{schema.layout_id}" if schema.layout_id else name),
    )


def _join_schemas(named: Sequence[tuple[str, FeatureSchema]], /) -> FeatureSchema:
    schemas = tuple((str(name), schema) for name, schema in named)
    names = tuple(
        output_name
        for name, schema in schemas
        for output_name in _prefixed_schema(name, schema).names
    )
    kinds = tuple(kind for _, schema in schemas for kind in schema.kinds)
    return FeatureSchema(
        names, kinds=kinds, layout_id="|".join(name for name, _ in schemas)
    )


def _join_sparse(blocks: Sequence[SparseFeatures], /) -> SparseFeatures:
    blocks_ = tuple(blocks)
    if not blocks_:
        raise ValueError("Cannot join an empty sparse feature collection.")
    first = blocks_[0]
    if any(
        block.case_shape != first.case_shape or block.sample_count != first.sample_count
        for block in blocks_[1:]
    ):
        raise ValueError("Sparse feature blocks must share case/sample axes.")
    offsets: list[int] = []
    offset = 0
    for block in blocks_:
        offsets.append(offset)
        offset += block.feature_count
    return SparseFeatures(
        jnp.concatenate(tuple(block.values for block in blocks_), axis=-1),
        jnp.concatenate(
            tuple(
                block.columns.source_indices + block_offset
                for block, block_offset in zip(blocks_, offsets, strict=True)
            ),
            axis=-1,
        ),
        feature_count=offset,
        valid=jnp.concatenate(tuple(block.columns.valid for block in blocks_), axis=-1),
        case_shape=first.case_shape,
    )


def _require_sparse_blocks(
    values: Sequence[FeatureArray],
    /,
) -> tuple[SparseFeatures, ...]:
    blocks: list[SparseFeatures] = []
    for value in values:
        if not isinstance(value, SparseFeatures):
            raise TypeError("Expected an exclusively sparse feature collection.")
        blocks.append(value)
    return tuple(blocks)


def _join_feature_batches(
    source: MLBatch,
    named_batches: Sequence[tuple[str, MLBatch]],
    /,
) -> MLBatch:
    entries = tuple(named_batches)
    if not entries:
        raise ValueError("Feature composition produced no output blocks.")
    for _, batch in entries:
        if (
            batch.case_shape != source.case_shape
            or batch.sample_count != source.sample_count
        ):
            raise ValueError("Feature blocks must preserve selected case/sample axes.")
    sparse = tuple(isinstance(batch.features, SparseFeatures) for _, batch in entries)
    if any(sparse) and not all(sparse):
        raise TypeError(
            "Joining sparse and dense feature blocks is unsupported; no implicit "
            "densification is performed."
        )
    schema = _join_schemas(tuple((name, batch.feature_schema) for name, batch in entries))
    if all(sparse):
        features = _join_sparse(
            _require_sparse_blocks(tuple(batch.features for _, batch in entries))
        )
        return source.with_features(features, feature_schema=schema)
    features = jnp.concatenate(
        tuple(jnp.asarray(batch.features) for _, batch in entries), axis=-1
    )
    feature_mask = jnp.concatenate(
        tuple(batch.feature_mask for _, batch in entries), axis=-1
    )
    return source.with_features(
        features, feature_schema=schema, feature_mask=feature_mask
    )


def _join_feature_values(named: Sequence[tuple[str, FeatureArray]], /) -> FeatureArray:
    entries = tuple(named)
    if not entries:
        raise ValueError("Feature composition produced no output blocks.")
    sparse = tuple(isinstance(values, SparseFeatures) for _, values in entries)
    if any(sparse) and not all(sparse):
        raise TypeError(
            "Joining sparse and dense feature blocks is unsupported; no implicit "
            "densification is performed."
        )
    if all(sparse):
        return _join_sparse(
            _require_sparse_blocks(tuple(values for _, values in entries))
        )
    arrays = tuple(jnp.asarray(values) for _, values in entries)
    leading = tuple(int(size) for size in arrays[0].shape[:-1])
    if any(
        tuple(int(size) for size in array.shape[:-1]) != leading for array in arrays[1:]
    ):
        raise ValueError("Feature blocks must preserve the same leading axes.")
    return jnp.concatenate(arrays, axis=-1)


__all__ = [
    "BatchTransformModel",
    "CompositionProvenance",
    "CompositionDiagnostics",
    "ReversibleTransformModel",
    "SchemaTransformModel",
]
