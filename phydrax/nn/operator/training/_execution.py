#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp

from ...._doc import DOC_KEY0
from ...._external_runtime import _require_execution
from ...._frozendict import frozendict
from ...._model._component import ExecutionCapabilities
from ...._model._ports import (
    ModelPorts,
    PortBindingEvidence,
    PortMapping,
    resolve_port_mapping,
    ValuePort,
)
from ...._strict import StrictModule
from ...._trainable import fixed_field, NonTrainableState
from ..._keys import EvalKey, split_eval_key
from ..capabilities import ConfiguredOperatorContract, OperatorTrainingEvidence
from ..data import (
    function_samples_with_values,
    OperatorBatch,
    OperatorFieldBatch,
    OperatorPrediction,
    OperatorTargetBatch,
)
from ..engine import AbstractOperatorModel
from ..field import OperatorFieldSpec
from ..sharding import (
    OperatorShardingPolicy,
    shard_operator_batch,
)
from ..task import OperatorTask
from ._dtype import OperatorDTypePolicy, OperatorPrecisionEvidence
from ._normalization import OperatorNormalizationPolicy
from ._physics import OperatorOutputPipeline


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
        inputs[field.source_name] = function_samples_with_values(samples, values)
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
                f"Target alias {name!r} resolves to unknown task target {canonical_name!r}."
            )
        specification = by_name[canonical_name]
        assert specification.query_name is not None
        assert specification.output_spec is not None
        if (
            field.query_name != specification.query_name
            or field.spec.to_dict() != specification.output_spec.to_dict()
        ):
            raise ValueError(
                f"Target field {name!r} does not match routed task field {canonical_name!r}."
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


OperatorOutputRoutes: TypeAlias = tuple[tuple[str, OperatorFieldSpec], ...]
"""`(model output name, task target field)` pairs bound by port ID, in task target order."""


def bind_operator_outputs(
    model: AbstractOperatorModel,
    task: OperatorTask,
    output_ports: Mapping[str, ValuePort] | None,
    port_mapping: PortMapping | None,
    /,
) -> tuple[frozendict[str, ValuePort], PortBindingEvidence]:
    """Bind every named model output to one task target field through explicit ports.

    Operator outputs carry no intrinsic field identity, so `output_ports` declares
    the `ValuePort` of each named model output. The outputs of `port_mapping` bind
    each declared port ID to the `OperatorFieldSpec.value_port()` ID of one task
    target field; `resolve_port_mapping` checks every pair and records aspects a
    side leaves undeclared. Every model output and task target is bound exactly
    once, and each bound output contract must equal its target's output spec.
    """
    if output_ports is None or port_mapping is None:
        raise ValueError(
            "Task-bound operators require explicit output_ports and a port_mapping "
            "binding every model output port to a task target field port."
        )
    if not isinstance(output_ports, Mapping) or any(
        not isinstance(name, str) or not isinstance(port, ValuePort)
        for name, port in output_ports.items()
    ):
        raise TypeError("output_ports must map model output names to ValuePort values.")
    if not isinstance(port_mapping, PortMapping):
        raise TypeError("port_mapping must be a PortMapping.")
    declared = model.operator_output_specs
    if set(output_ports) != set(declared):
        raise ValueError(
            "output_ports must declare every model output exactly; "
            f"expected {tuple(declared)!r}, got {tuple(output_ports)!r}."
        )
    targets = task.target_fields
    target_ports = tuple(field.value_port() for field in targets)
    evidence = resolve_port_mapping(
        ModelPorts(inputs=(), outputs=tuple(output_ports[name] for name in declared)),
        ModelPorts(inputs=(), outputs=target_ports),
        port_mapping,
    )
    bound_targets = {owner for _, owner in evidence.outputs}
    unbound = [
        field.name
        for field, port in zip(targets, target_ports, strict=True)
        if port.port_id not in bound_targets
    ]
    if unbound:
        raise ValueError(f"port_mapping leaves task target fields {unbound} unbound.")
    ports = frozendict(output_ports)
    for name, target in _operator_output_routes(task, ports, evidence):
        assert target.output_spec is not None
        if declared[name].to_dict() != target.output_spec.to_dict():
            raise ValueError(
                f"Model output {name!r} contract disagrees with task target "
                f"{target.name!r}."
            )
    return ports, evidence


def _operator_output_routes(
    task: OperatorTask,
    output_ports: Mapping[str, ValuePort],
    port_binding: PortBindingEvidence,
    /,
) -> OperatorOutputRoutes:
    """Route model outputs to task targets by the bound port IDs, never by name."""
    name_by_port = {port.port_id: name for name, port in output_ports.items()}
    model_by_owner = {owner: model for model, owner in port_binding.outputs}
    return tuple(
        (name_by_port[model_by_owner[field.value_port().port_id]], field)
        for field in task.target_fields
    )


def _port_binding_record(
    output_ports: Mapping[str, ValuePort],
    port_binding: PortBindingEvidence,
    /,
) -> dict[str, Any]:
    """Canonical port-ID record of one operator output binding."""
    return {
        "output_ports": {name: port.port_id for name, port in output_ports.items()},
        "binding": port_binding.binding_fingerprint,
    }


def physicalize_prediction(
    prediction: OperatorPrediction,
    physical_batch: OperatorBatch,
    task: OperatorTask,
    routes: OperatorOutputRoutes,
    normalization: OperatorNormalizationPolicy | None,
    /,
) -> OperatorPrediction:
    """Map model-named execution output into task-named physical output."""
    expected = tuple(name for name, _ in routes)
    if set(prediction.fields) != set(expected):
        raise ValueError(
            "Model prediction fields do not match the bound model outputs; "
            f"expected {expected!r}, got {tuple(prediction.fields)!r}."
        )
    fields: dict[str, OperatorFieldBatch] = {}
    for model_name, target in routes:
        assert target.output_spec is not None
        assert target.query_name is not None
        raw_field = prediction.field(model_name)
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
    routes: OperatorOutputRoutes,
    normalization: OperatorNormalizationPolicy | None,
    /,
) -> OperatorPrediction:
    """Map task-named physical output back into model execution coordinates."""
    task.validate_prediction(prediction)
    fields: dict[str, OperatorFieldBatch] = {}
    for model_name, target in routes:
        output_spec = target.output_spec
        if output_spec is None:
            raise ValueError(f"Task output field {target.name!r} has no output spec.")
        physical_field = prediction.field(target.name)
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
            if target.name not in normalization.targets:
                raise KeyError(f"Missing normalizer for target field {target.name!r}.")
            values = normalization.targets[target.name].normalize(values)
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
    routes: OperatorOutputRoutes,
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
        routes,
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
        routes,
        normalization,
    )
    return execution_prediction, physical_prediction


class PreparedOperatorInput(StrictModule, NonTrainableState):
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
    """Prepared runtime decisions and lowered callable for one trained operator.

    The plan is a neutral container: ``execution_model`` keeps its own roles while
    the task and normalization statistics are FIXED. ``output_ports`` declares the
    port of each named model output and ``port_binding`` records their audited
    binding to task target field ports; model outputs are routed to task targets
    by those port IDs.

    ``execution`` holds the model's declared `ExecutionCapabilities`, consulted
    before casting, preparation, and dispatch: the ``"compiled"`` strategy
    requires ``jit``, and every prediction is admitted against them before the
    model is invoked.
    """

    execution_model: AbstractOperatorModel
    task: OperatorTask = fixed_field()
    contract: ConfiguredOperatorContract
    execution: ExecutionCapabilities
    output_ports: frozendict[str, ValuePort]
    port_binding: PortBindingEvidence
    fixed_query_fingerprints: frozendict[str, str]
    output_pipeline: OperatorOutputPipeline | None
    normalization: OperatorNormalizationPolicy | None = fixed_field()
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
        output_ports: Mapping[str, ValuePort] | None = None,
        port_mapping: PortMapping | None = None,
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
        execution = execution_model.model_execution_contract().execution
        if compilation_strategy == "compiled" and not execution.jit:
            raise ValueError(
                f"The compiled strategy requires jit; the {execution.tier!r} "
                "execution model does not support it."
            )

        cast_model = policy.cast_model(execution_model)
        contract = cast_model.operator_contract
        target_names = tuple(field.name for field in task.target_fields)
        output_ports_, port_binding = bind_operator_outputs(
            cast_model, task, output_ports, port_mapping
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
                f"Fixed query fingerprints reference unknown task queries: {tuple(sorted(unknown_fixed_queries))!r}."
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
        self.execution = execution
        self.output_ports = output_ports_
        self.port_binding = port_binding
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
    def output_routes(self) -> OperatorOutputRoutes:
        """`(model output name, task target field)` pairs bound by port ID."""
        return _operator_output_routes(self.task, self.output_ports, self.port_binding)

    @property
    def contract_fingerprint(self) -> str:
        return _canonical_hash(
            {
                "operator_contract": operator_contract_fingerprint(self.contract),
                "port_binding": _port_binding_record(
                    self.output_ports, self.port_binding
                ),
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
                        "mesh_shape": tuple(sharding.mesh.devices.shape),
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

    def replace_prepared_source(
        self,
        prepared: PreparedOperatorInput,
        source_name: str,
        physical_values: Any,
        /,
    ) -> PreparedOperatorInput:
        """Replace one prepared physical source without relowering static inputs."""
        if not isinstance(prepared, PreparedOperatorInput):
            raise TypeError("prepared must be a PreparedOperatorInput.")
        if prepared.plan_fingerprint != self.fingerprint:
            raise ValueError(
                "Prepared operator input belongs to a different runtime contract."
            )
        if self.sharding_policy is not None:
            raise ValueError(
                "Prepared source replacement does not yet support operator sharding."
            )
        source = str(source_name)
        matches = tuple(
            field for field in self.task.source_fields if field.source_name == source
        )
        if len(matches) != 1:
            raise KeyError(
                f"Prepared source {source!r} must name exactly one task source."
            )
        field = matches[0]
        physical_sample = prepared.physical_batch.input(source)
        execution_sample = prepared.execution_batch.input(source)
        if physical_sample.values is None or execution_sample.values is None:
            raise ValueError("Prepared source replacement requires source values.")
        values = jnp.asarray(physical_values)
        if values.shape != physical_sample.values.shape:
            raise ValueError(
                f"Prepared source {source!r} expects shape {physical_sample.values.shape}, got {values.shape}."
            )
        physical_inputs = dict(prepared.physical_batch.inputs)
        physical_inputs[source] = function_samples_with_values(
            physical_sample,
            values,
        )
        execution_values = field.nondimensionalize(values)
        mask = physical_sample.mask_array(case_shape=prepared.physical_batch.case_shape)
        trailing = (1,) * (execution_values.ndim - mask.ndim)
        execution_values = jnp.where(
            mask.reshape(mask.shape + trailing),
            execution_values,
            jnp.zeros((), dtype=execution_values.dtype),
        )
        if self.normalization is not None:
            normalizer = self.normalization.input_values.get(source)
            if normalizer is not None:
                execution_values = normalizer.normalize(execution_values)
        execution_values = execution_values.astype(execution_sample.values.dtype)
        execution_inputs = dict(prepared.execution_batch.inputs)
        execution_inputs[source] = function_samples_with_values(
            execution_sample,
            execution_values,
        )
        return PreparedOperatorInput(
            OperatorBatch(
                inputs=physical_inputs,
                queries=prepared.physical_batch.queries,
                case_axes=prepared.physical_batch.case_axes,
                case_shape=prepared.physical_batch.case_shape,
            ),
            OperatorBatch(
                inputs=execution_inputs,
                queries=prepared.execution_batch.queries,
                case_axes=prepared.execution_batch.case_axes,
                case_shape=prepared.execution_batch.case_shape,
            ),
            plan_fingerprint=prepared.plan_fingerprint,
        )

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
        _require_execution(self.execution, prepared, key)
        _, prediction = _evaluate_operator_step(
            self.execution_model,
            prepared.execution_batch,
            prepared.physical_batch,
            self.task,
            self.output_routes,
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
    "OperatorOutputRoutes",
    "OperatorPaddingPolicy",
    "PreparedOperatorInput",
    "bind_operator_outputs",
    "executionize_prediction",
    "nondimensionalize_batch",
    "nondimensionalize_targets",
    "operator_contract_fingerprint",
    "operator_normalization_fingerprint",
    "physicalize_prediction",
]
