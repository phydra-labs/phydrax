#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from ...._doc import DOC_KEY0
from ...._frozendict import frozendict
from ...._strict import StrictModule
from ..._keys import EvalKey, split_eval_key
from ..capabilities import ConfiguredOperatorContract, OperatorTrainingEvidence
from ..data import (
    FunctionSamples,
    OperatorBatch,
    OperatorFieldBatch,
    OperatorPrediction,
    OperatorTargetBatch,
)
from ..engine import AbstractOperatorModel
from ..sharding import (
    OperatorShardingPolicy,
    shard_operator_batch,
)
from ..task import OperatorTask
from ._dtype import OperatorDTypePolicy, OperatorPrecisionEvidence
from ._normalization import OperatorNormalizationPolicy
from ._physics import OperatorOutputPipeline


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
        support_id=samples.support_id,
        measure_id=samples.measure_id,
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
        values = field.nondimensionalize(jnp.asarray(samples.values))
        mask = samples.mask_array(case_shape=batch.case_shape)
        trailing = (1,) * (values.ndim - mask.ndim)
        values = jnp.where(
            mask.reshape(mask.shape + trailing),
            values,
            jnp.zeros((), dtype=values.dtype),
        )
        inputs[field.source_name] = samples_with_values(samples, values)
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
    *,
    target_aliases: Mapping[str, str] | None = None,
) -> OperatorTargetBatch:
    """Map physical target values, including rollout aliases, into execution units."""
    if not targets.fields:
        return OperatorTargetBatch(
            {},
            case_axes=targets.case_axes,
            case_shape=targets.case_shape,
        )
    aliases = {} if target_aliases is None else dict(target_aliases)
    target_names = tuple(field.name for field in task.target_fields)
    replaced = set(aliases.values())
    expected = tuple(name for name in target_names if name not in replaced) + tuple(
        aliases
    )
    if set(targets.fields) != set(expected):
        raise ValueError(
            "Operator target names must match the task and rollout aliases; "
            f"expected {expected!r}, got {tuple(targets.fields)!r}."
        )
    by_name = task.field_by_name
    fields: dict[str, OperatorFieldBatch] = {}
    for name, field in targets.fields.items():
        canonical_name = aliases.get(name, name)
        if canonical_name not in by_name or not by_name[canonical_name].is_target:
            raise KeyError(
                f"Target alias {name!r} resolves to unknown task target "
                f"{canonical_name!r}."
            )
        specification = by_name[canonical_name]
        assert specification.query_name is not None
        assert specification.output_spec is not None
        if (
            field.query_name != specification.query_name
            or field.spec.to_dict() != specification.output_spec.to_dict()
        ):
            raise ValueError(
                f"Target field {name!r} does not match routed task field "
                f"{canonical_name!r}."
            )
        values = specification.nondimensionalize(field.values)
        fields[name] = OperatorFieldBatch(
            values,
            query_name=field.query_name,
            spec=field.spec,
        )
    return OperatorTargetBatch(
        fields,
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
        if (
            target.output_spec.classification is not None
            and raw_field.spec.to_dict() != target.output_spec.to_dict()
        ):
            raise ValueError(
                f"Model output {target.name!r} does not preserve classification semantics."
            )
        values = raw_field.values
        if normalization is not None and target.output_spec.classification is None:
            if target.name not in normalization.targets:
                raise KeyError(f"Missing normalizer for target field {target.name!r}.")
            values = normalization.targets[target.name].denormalize(values)
        values = target.dimensionalize(values)
        values = target.output_spec.validate_prediction(
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
        output_spec = target.output_spec
        if output_spec is None:
            raise ValueError(f"Task output field {target_name!r} has no output spec.")
        physical_field = prediction.field(target_name)
        template_field = template.field(model_name)
        if (
            output_spec.classification is not None
            and template_field.spec.to_dict() != output_spec.to_dict()
        ):
            raise ValueError(
                f"Execution template {model_name!r} does not preserve classification semantics."
            )
        values = target.nondimensionalize(physical_field.values)
        if normalization is not None and output_spec.classification is None:
            if target_name not in normalization.targets:
                raise KeyError(f"Missing normalizer for target field {target_name!r}.")
            values = normalization.targets[target_name].normalize(values)
        query = execution_batch.query(template_field.query_name)
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


OperatorCompilationStrategy = Literal["eager", "compiled"]
OperatorPaddingPolicy = Literal["explicit_mask"]


def _canonical_hash(value: Any, /) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def operator_contract_fingerprint(contract: ConfiguredOperatorContract, /) -> str:
    """Return a stable digest of one configured runtime/training contract."""
    return _canonical_hash(
        {
            "architecture": contract.architecture,
            "configuration": [list(item) for item in contract.configuration],
            "capabilities": asdict(contract.capabilities),
            "training": asdict(contract.training),
            "field_specs": [field.to_dict() for field in contract.field_specs],
        }
    )


def operator_normalization_fingerprint(
    normalization: OperatorNormalizationPolicy | None, /
) -> str:
    """Return a stable digest of normalization semantics, including the null policy."""
    return _canonical_hash(None if normalization is None else normalization.to_dict())


def _operator_prediction(
    model: AbstractOperatorModel,
    batch: OperatorBatch,
    key: EvalKey,
    dtype_policy: OperatorDTypePolicy,
    /,
) -> OperatorPrediction:
    compute_model = dtype_policy.compute_model(model)
    precision = dtype_policy.matmul_precision
    precision_context = (
        nullcontext() if precision is None else jax.default_matmul_precision(precision)
    )
    with precision_context:
        return compute_model.predict_prevalidated(batch, key=key)


_compiled_operator_prediction = eqx.filter_jit(_operator_prediction)


def _evaluate_operator_step(
    model: AbstractOperatorModel,
    execution_batch: OperatorBatch,
    physical_batch: OperatorBatch,
    task: OperatorTask,
    output_field_map: Mapping[str, str],
    output_pipeline: OperatorOutputPipeline | None,
    normalization: OperatorNormalizationPolicy | None,
    dtype_policy: OperatorDTypePolicy,
    key: EvalKey,
    /,
    *,
    predictor: Callable[
        [AbstractOperatorModel, OperatorBatch, EvalKey, OperatorDTypePolicy],
        OperatorPrediction,
    ] = _operator_prediction,
) -> tuple[OperatorPrediction, OperatorPrediction]:
    """Run the one canonical task-bound prediction and constraint pipeline."""
    model_key = key
    pipeline_key = key
    if output_pipeline is not None:
        model_key, pipeline_key = split_eval_key(key, 2)
    raw_prediction = predictor(
        model,
        execution_batch,
        model_key,
        dtype_policy,
    )
    physical_prediction = physicalize_prediction(
        raw_prediction,
        physical_batch,
        task,
        output_field_map,
        normalization,
    )
    if output_pipeline is not None:
        physical_prediction = output_pipeline(
            physical_prediction,
            physical_batch,
            key=pipeline_key,
        )
        task.validate_prediction(physical_prediction)
    execution_prediction = executionize_prediction(
        physical_prediction,
        raw_prediction,
        execution_batch,
        task,
        output_field_map,
        normalization,
    )
    return execution_prediction, physical_prediction


class PreparedOperatorInput(StrictModule):
    """Physical and execution batches prepared for exactly one execution plan."""

    physical_batch: OperatorBatch
    execution_batch: OperatorBatch
    plan_fingerprint: str

    def __init__(
        self,
        physical_batch: OperatorBatch,
        execution_batch: OperatorBatch,
        /,
        *,
        plan_fingerprint: str,
    ):
        self.physical_batch = physical_batch
        self.execution_batch = execution_batch
        self.plan_fingerprint = str(plan_fingerprint)


class OperatorExecutionPlan(StrictModule):
    """Prepared runtime decisions and lowered callable for one trained operator."""

    execution_model: AbstractOperatorModel
    task: OperatorTask
    contract: ConfiguredOperatorContract
    output_field_map: frozendict[str, str]
    fixed_query_fingerprints: frozendict[str, str]
    output_pipeline: OperatorOutputPipeline | None
    normalization: OperatorNormalizationPolicy | None
    dtype_policy: OperatorDTypePolicy
    precision_evidence: OperatorPrecisionEvidence
    training_evidence: OperatorTrainingEvidence
    sharding_policy: OperatorShardingPolicy | None
    compilation_strategy: OperatorCompilationStrategy
    padding_policy: OperatorPaddingPolicy
    lowered_callable: Callable[
        [
            AbstractOperatorModel,
            OperatorBatch,
            EvalKey,
            OperatorDTypePolicy,
        ],
        OperatorPrediction,
    ] = eqx.field(static=True)

    def __init__(
        self,
        execution_model: AbstractOperatorModel,
        task: OperatorTask,
        /,
        *,
        training_evidence: OperatorTrainingEvidence,
        output_field_map: Mapping[str, str] | None = None,
        fixed_query_fingerprints: Mapping[str, str] | None = None,
        output_pipeline: OperatorOutputPipeline | None = None,
        normalization: OperatorNormalizationPolicy | None = None,
        dtype_policy: OperatorDTypePolicy | None = None,
        sharding_policy: OperatorShardingPolicy | None = None,
        compilation_strategy: OperatorCompilationStrategy = "eager",
        padding_policy: OperatorPaddingPolicy = "explicit_mask",
    ):
        if not isinstance(execution_model, AbstractOperatorModel):
            raise TypeError("OperatorExecutionPlan requires a PhydraX execution model.")
        if not isinstance(task, OperatorTask):
            raise TypeError("OperatorExecutionPlan requires an OperatorTask.")
        if not isinstance(training_evidence, OperatorTrainingEvidence):
            raise TypeError("training_evidence must be an OperatorTrainingEvidence.")
        if normalization is not None and not isinstance(
            normalization, OperatorNormalizationPolicy
        ):
            raise TypeError("normalization must be an OperatorNormalizationPolicy.")
        if output_pipeline is not None and not isinstance(
            output_pipeline, OperatorOutputPipeline
        ):
            raise TypeError("output_pipeline must be an OperatorOutputPipeline.")
        policy = OperatorDTypePolicy() if dtype_policy is None else dtype_policy
        if not isinstance(policy, OperatorDTypePolicy):
            raise TypeError("dtype_policy must be an OperatorDTypePolicy.")
        if sharding_policy is not None and not isinstance(
            sharding_policy, OperatorShardingPolicy
        ):
            raise TypeError("sharding_policy must be an OperatorShardingPolicy.")
        if compilation_strategy not in ("eager", "compiled"):
            raise ValueError("compilation_strategy must be 'eager' or 'compiled'.")
        if padding_policy != "explicit_mask":
            raise ValueError("padding_policy must be 'explicit_mask'.")

        cast_model = policy.cast_model(execution_model)
        contract = cast_model.operator_contract
        targets = task.target_fields
        declared = cast_model.operator_output_specs
        target_names = tuple(field.name for field in targets)
        if output_field_map is None:
            resolved_output_field_map = {
                name: name for name in declared if name in target_names
            }
        else:
            resolved_output_field_map = {
                str(model_name): str(target_name)
                for model_name, target_name in output_field_map.items()
            }
        if set(resolved_output_field_map) != set(declared):
            raise ValueError(
                "output_field_map must name every model output exactly; "
                f"expected {tuple(declared)!r}, got "
                f"{tuple(resolved_output_field_map)!r}."
            )
        if len(set(resolved_output_field_map.values())) != len(
            resolved_output_field_map
        ) or set(resolved_output_field_map.values()) != set(target_names):
            raise ValueError(
                "output_field_map must map bijectively onto the task target fields; "
                f"expected {target_names!r}, got "
                f"{tuple(resolved_output_field_map.values())!r}."
            )
        model_name_by_target = {
            target_name: model_name
            for model_name, target_name in resolved_output_field_map.items()
        }
        declared_specs = tuple(
            declared[model_name_by_target[field.name]] for field in targets
        )
        for target, model_spec in zip(targets, declared_specs, strict=True):
            target_spec = target.output_spec
            assert target_spec is not None
            if model_spec.to_dict() != target_spec.to_dict():
                raise ValueError(
                    f"Model output contract for {target.name!r} disagrees with the task."
                )
        if output_pipeline is not None:
            unknown_pipeline_fields = {
                transform.field_name
                for transform in output_pipeline.transforms
                if transform.field_name not in target_names
            }
            if unknown_pipeline_fields:
                raise ValueError(
                    "Output pipeline transforms reference unknown task targets: "
                    f"{tuple(sorted(unknown_pipeline_fields))!r}."
                )

        resolved_fixed_queries = {
            str(name): str(fingerprint)
            for name, fingerprint in (
                {} if fixed_query_fingerprints is None else fixed_query_fingerprints
            ).items()
        }
        unknown_fixed_queries = set(resolved_fixed_queries) - set(task.query_by_name)
        if unknown_fixed_queries:
            raise ValueError(
                "Fixed query fingerprints reference unknown task queries: "
                f"{tuple(sorted(unknown_fixed_queries))!r}."
            )
        if task.problem.query_is_fixed is True and set(resolved_fixed_queries) != set(
            task.query_by_name
        ):
            raise ValueError(
                "Fixed-query tasks require a geometry fingerprint for every query."
            )

        self.execution_model = cast_model
        self.task = task
        self.contract = contract
        self.output_field_map = frozendict(resolved_output_field_map)
        self.fixed_query_fingerprints = frozendict(resolved_fixed_queries)
        self.output_pipeline = output_pipeline
        self.normalization = normalization
        self.dtype_policy = policy
        self.precision_evidence = policy.precision_evidence
        self.training_evidence = training_evidence
        self.sharding_policy = sharding_policy
        self.compilation_strategy = compilation_strategy
        self.padding_policy = padding_policy
        self.lowered_callable = (
            _compiled_operator_prediction
            if compilation_strategy == "compiled"
            else _operator_prediction
        )

    @property
    def task_fingerprint(self) -> str:
        return self.task.fingerprint

    @property
    def contract_fingerprint(self) -> str:
        return _canonical_hash(
            {
                "operator_contract": operator_contract_fingerprint(self.contract),
                "output_field_map": dict(self.output_field_map),
                "fixed_query_fingerprints": dict(self.fixed_query_fingerprints),
                "output_pipeline": (
                    None
                    if self.output_pipeline is None
                    else self.output_pipeline.fingerprint
                ),
            }
        )

    @property
    def normalization_fingerprint(self) -> str:
        return operator_normalization_fingerprint(self.normalization)

    @property
    def fingerprint(self) -> str:
        sharding = self.sharding_policy
        return _canonical_hash(
            {
                "task": self.task_fingerprint,
                "contract": self.contract_fingerprint,
                "normalization": self.normalization_fingerprint,
                "dtype": self.dtype_policy.to_dict(),
                "precision_evidence": self.precision_evidence.to_dict(),
                "training_evidence": asdict(self.training_evidence),
                "compilation": self.compilation_strategy,
                "padding": self.padding_policy,
                "sharding": (
                    None
                    if sharding is None
                    else {
                        "mesh_axis": sharding.mesh_axis,
                        "case_axis": sharding.case_axis,
                        "mesh_shape": tuple(
                            int(size) for size in sharding.mesh.devices.shape
                        ),
                    }
                ),
            }
        )

    def prepare_prevalidated(self, batch: OperatorBatch, /) -> PreparedOperatorInput:
        """Transform a batch whose semantic contracts were checked on the host."""
        physical_batch = batch
        execution_batch = nondimensionalize_batch(batch, self.task)
        if self.normalization is not None:
            execution_batch = self.normalization.normalize_batch(execution_batch)
        execution_batch = self.dtype_policy.cast_batch(execution_batch)
        if self.sharding_policy is not None:
            physical_batch = shard_operator_batch(
                physical_batch,
                self.sharding_policy,
            )
            execution_batch = shard_operator_batch(
                execution_batch,
                self.sharding_policy,
            )
        return PreparedOperatorInput(
            physical_batch,
            execution_batch,
            plan_fingerprint=self.fingerprint,
        )

    def prepare(self, batch: OperatorBatch, /) -> PreparedOperatorInput:
        """Validate and lower one physical batch outside the compiled hot path."""
        for name, expected in self.fixed_query_fingerprints.items():
            actual = batch.query(name).geometry_fingerprint()
            if actual != expected:
                raise ValueError(
                    f"Fixed query {name!r} has a different physical geometry."
                )
        self.task.validate_batch(batch)
        report = self.contract.validate(
            batch,
            problem=self.task.problem,
            training_evidence=self.training_evidence,
            fields=self.task.fields,
        )
        report.require()
        return self.prepare_prevalidated(batch)

    def predict_prepared(
        self,
        prepared: PreparedOperatorInput,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        """Execute one prepared batch and restore task names and physical units."""
        if not isinstance(prepared, PreparedOperatorInput):
            raise TypeError("predict_prepared requires a PreparedOperatorInput.")
        if prepared.plan_fingerprint != self.fingerprint:
            raise ValueError(
                "Prepared operator input belongs to a different runtime contract."
            )
        _, prediction = _evaluate_operator_step(
            self.execution_model,
            prepared.execution_batch,
            prepared.physical_batch,
            self.task,
            self.output_field_map,
            self.output_pipeline,
            self.normalization,
            self.dtype_policy,
            key,
            predictor=self.lowered_callable,
        )
        return prediction

    def predict(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> OperatorPrediction:
        """Validate, prepare, execute, and restore one physical prediction."""
        return self.predict_prepared(self.prepare(batch), key=key)


__all__ = [
    "OperatorCompilationStrategy",
    "OperatorExecutionPlan",
    "OperatorPaddingPolicy",
    "PreparedOperatorInput",
    "executionize_prediction",
    "nondimensionalize_batch",
    "nondimensionalize_targets",
    "operator_contract_fingerprint",
    "operator_normalization_fingerprint",
    "physicalize_prediction",
    "samples_with_values",
]
