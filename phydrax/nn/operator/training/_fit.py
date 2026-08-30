#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from math import ceil, prod
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ...._frozendict import frozendict
from ...._trainable import combine_trainable, partition_trainable
from ...._training import (
    EvaluationParametersFn,
    resolve_evaluation_parameters,
    TensorBoardLogger,
    TrainingCallback,
    TrainingController,
    TrainingProgress,
    TrainingSignalGuard,
)
from ..._keys import split_eval_key
from ..._loss import model_loss_labels, model_loss_values
from ...layers._dropout import inference_mode
from ...parameters import ParameterSubspace
from ...parameters._low_rank import (
    contains_low_rank_updates,
    validate_low_rank_subspace,
)
from ..capabilities import OperatorTrainingEvidence
from ..data import OperatorBatch, OperatorTargetBatch
from ..engine import AbstractOperatorModel
from ..sampling import OperatorCaseSource
from ..sharding import (
    OperatorShardingPolicy,
    replicate_operator_model,
    shard_operator_batch,
    shard_operator_targets,
)
from ..task import OperatorTask
from ._checkpoint import (
    _read_operator_training_manifest,
    load_operator_training_checkpoint,
    save_operator_training_checkpoint,
)
from ._dataset import OperatorDataset
from ._dtype import OperatorDTypePolicy, OperatorPrecisionEvidence
from ._execution import (
    _operator_prediction,
    executionize_prediction,
    nondimensionalize_batch,
    nondimensionalize_targets,
    physicalize_prediction,
)
from ._fingerprint import operator_fit_schema
from ._loader import OperatorBatchLoader, OperatorTrainingBatch
from ._loss_scale import (
    OperatorLossScalePolicy,
    OperatorLossScaleState,
    tree_all_finite,
)
from ._losses import (
    AbstractOperatorLossTerm,
    OperatorLossContext,
    SupervisedOperatorLoss,
)
from ._normalization import (
    fit_operator_normalization,
    OperatorNormalizationPolicy,
)
from ._physics import OperatorOutputPipeline
from ._trained_operator import (
    operator_contract_fingerprint,
    TrainedOperator,
)


@dataclass(frozen=True)
class OperatorValidationPolicy:
    """Validation cadence, early stopping, and selected-model semantics."""

    every: int = 1
    monitor: str = "loss"
    mode: Literal["min", "max"] = "min"
    patience: int | None = None
    minimum_delta: float = 0.0
    relative_minimum_delta: float = 0.0
    select_best: bool = True

    def __post_init__(self):
        if int(self.every) <= 0:
            raise ValueError("Validation cadence must be positive.")
        if not self.monitor:
            raise ValueError("Validation monitor must be non-empty.")
        if self.mode not in ("min", "max"):
            raise ValueError("Validation mode must be 'min' or 'max'.")
        if self.patience is not None and int(self.patience) <= 0:
            raise ValueError("Validation patience must be positive when provided.")
        if self.minimum_delta < 0.0 or self.relative_minimum_delta < 0.0:
            raise ValueError("Validation improvement deltas must be non-negative.")


@dataclass(frozen=True)
class OperatorFitHistory:
    """Immutable learning curves and validation records from one fit run."""

    initial_metrics: frozendict[str, float]
    train_steps: tuple[int, ...]
    train_metrics: tuple[frozendict[str, float], ...]
    validation_steps: tuple[int, ...]
    validation_metrics: tuple[frozendict[str, float], ...]
    final_metrics: frozendict[str, float]

    @property
    def losses(self) -> tuple[float, ...]:
        return tuple(metrics["loss"] for metrics in self.train_metrics)

    @property
    def validation_losses(self) -> tuple[float, ...]:
        return tuple(metrics["loss"] for metrics in self.validation_metrics)


@dataclass(frozen=True)
class OperatorFitResult:
    """Execution models, training state, and optional task-bound runtime."""

    execution_model: AbstractOperatorModel
    last_execution_model: AbstractOperatorModel
    trained_operator: TrainedOperator | None
    output_field_map: frozendict[str, str]
    output_pipeline: OperatorOutputPipeline | None
    history: OperatorFitHistory
    normalization: OperatorNormalizationPolicy | None
    dtype_policy: OperatorDTypePolicy
    precision_evidence: OperatorPrecisionEvidence
    loss_scale_state: OperatorLossScaleState | None
    progress: TrainingProgress
    resumed_from_step: int
    training_seconds: float
    checkpoint_path: Path | None
    stopped_by_signal: bool = False
    stopped_by_callback: bool = False

    @property
    def initial_loss(self) -> float:
        return self.history.initial_metrics["loss"]

    @property
    def final_loss(self) -> float:
        return self.history.final_metrics["loss"]

    @property
    def completed_steps(self) -> int:
        return self.progress.update_step


FitInput = OperatorDataset | OperatorCaseSource | OperatorBatchLoader


def _canonical_json(value: Any, /) -> Any:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _canonical_hash(value: Any, /) -> str:
    payload = json.dumps(
        _canonical_json(value),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _raw_loader(
    data: FitInput,
    /,
    *,
    batch_size: int | None,
    shuffle: bool,
    seed: int,
    prefetch: int,
    split: str,
) -> OperatorBatchLoader:
    if isinstance(data, OperatorBatchLoader):
        return OperatorBatchLoader(
            data.source,
            batch_size=data.batch_size,
            shuffle=data.shuffle,
            seed=data.seed,
            drop_last=data.drop_last,
            prefetch=data.prefetch,
            sampling=data.sampling,
            split=data.split,
        )
    size = data.size
    resolved_batch_size = size if batch_size is None else int(batch_size)
    return OperatorBatchLoader(
        data,
        batch_size=resolved_batch_size,
        shuffle=shuffle,
        seed=seed,
        prefetch=prefetch,
        split=split,
    )


def _nondimensionalize(
    batch: OperatorBatch,
    targets: OperatorTargetBatch,
    task: OperatorTask | None,
    /,
) -> tuple[OperatorBatch, OperatorTargetBatch]:
    if task is None:
        return batch, targets
    task.validate_batch(batch)
    return (
        nondimensionalize_batch(batch, task),
        nondimensionalize_targets(targets, task),
    )


def _place_batch(
    raw: OperatorTrainingBatch,
    /,
    *,
    task: OperatorTask | None,
    normalization: OperatorNormalizationPolicy | None,
    dtype_policy: OperatorDTypePolicy,
    sharding_policy: OperatorShardingPolicy | None,
) -> OperatorTrainingBatch:
    physical_batch = raw.batch if raw.physical_batch is None else raw.physical_batch
    physical_targets = (
        raw.targets if raw.physical_targets is None else raw.physical_targets
    )
    batch, targets = _nondimensionalize(physical_batch, physical_targets, task)
    if normalization is not None:
        batch = normalization.normalize_batch(batch)
        targets = normalization.normalize_targets(targets)
    batch = dtype_policy.cast_batch(batch)
    targets = dtype_policy.cast_targets(targets)
    if sharding_policy is not None:
        batch = shard_operator_batch(batch, sharding_policy)
        targets = shard_operator_targets(targets, sharding_policy)
        physical_batch = shard_operator_batch(physical_batch, sharding_policy)
        physical_targets = shard_operator_targets(
            physical_targets,
            sharding_policy,
        )
    return replace(
        raw,
        batch=batch,
        targets=targets,
        physical_batch=physical_batch,
        physical_targets=physical_targets,
    )


def _resolve_output_map(
    model: AbstractOperatorModel,
    targets: OperatorTargetBatch,
    output_field_map: Mapping[str, str] | None,
    task: OperatorTask | None,
    /,
) -> dict[str, str]:
    declared = tuple(model.operator_output_specs)
    target_names = (
        tuple(targets.fields)
        if targets.fields
        else (
            tuple(field.name for field in task.target_fields)
            if task is not None
            else declared
        )
    )
    if output_field_map is None:
        if set(declared) == set(target_names):
            resolved = {name: name for name in declared}
        elif len(declared) == len(target_names) == 1:
            resolved = {declared[0]: target_names[0]}
        else:
            raise ValueError(
                "output_field_map is required when model outputs and physical output "
                "fields do not have identical names."
            )
    else:
        resolved = {
            str(model_name): str(target_name)
            for model_name, target_name in output_field_map.items()
        }
    if set(resolved) != set(declared) or set(resolved.values()) != set(target_names):
        raise ValueError(
            "output_field_map must bijectively map every model output to a physical "
            "output field."
        )
    return resolved


def _default_losses(
    output_map: Mapping[str, str],
    /,
    *,
    physical_names: bool,
) -> tuple[SupervisedOperatorLoss, ...]:
    multiple = len(output_map) > 1
    return tuple(
        SupervisedOperatorLoss(
            name=f"supervised_l2/{target_name}" if multiple else "supervised_l2",
            prediction_field=target_name if physical_names else model_name,
            target_field=target_name,
        )
        for model_name, target_name in output_map.items()
    )


def _case_count(batch: OperatorBatch, /) -> int:
    return prod(batch.case_shape) if batch.case_shape else 1


def _tree_zeros(tree: Any, /) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_add_scaled(left: Any, right: Any, scale: float, /) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y * scale, left, right)


def _tree_scale(tree: Any, scale: Any, /) -> Any:
    return jax.tree_util.tree_map(lambda value: value * scale, tree)


def _validate_training_precision(
    dtype_policy: OperatorDTypePolicy,
    loss_scale_policy: OperatorLossScalePolicy | None,
    /,
) -> None:
    if dtype_policy.parameter_dtype not in ("float32", "float64"):
        raise ValueError(
            "Operator fitting requires float32 or float64 persistent parameters."
        )
    if dtype_policy.compute_dtype in ("float16", "bfloat16") and (
        dtype_policy.reduction_dtype not in ("float32", "float64")
    ):
        raise ValueError(
            "Low-precision operator compute requires float32 or float64 reductions."
        )
    if dtype_policy.compute_dtype == "float16":
        if loss_scale_policy is None:
            raise ValueError(
                "float16 operator compute requires an explicit loss_scale_policy."
            )
    elif loss_scale_policy is not None:
        raise ValueError("Loss scaling is supported only for float16 operator compute.")


def _has_trainable_arrays(parameters: Any, /) -> bool:
    return any(eqx.is_array(leaf) for leaf in jax.tree_util.tree_leaves(parameters))


def _metric_dict(names: tuple[str, ...], values: Sequence[Any], /) -> dict[str, float]:
    return {
        name: float(jax.device_get(jnp.asarray(value, dtype=float).reshape(())))
        for name, value in zip(names, values, strict=True)
    }


def fit_operator(
    model: AbstractOperatorModel,
    train: FitInput,
    /,
    *,
    validation: FitInput | None = None,
    task: OperatorTask | None = None,
    training_evidence: OperatorTrainingEvidence | None = None,
    output_field_map: Mapping[str, str] | None = None,
    loss_terms: Sequence[AbstractOperatorLossTerm] | None = None,
    output_pipeline: OperatorOutputPipeline | None = None,
    include_model_losses: bool = True,
    optimizer: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs
    | None = None,
    optimizer_id: str | None = None,
    evaluation_parameters: EvaluationParametersFn | None = None,
    evaluation_parameters_id: str | None = None,
    parameter_subspace: ParameterSubspace | None = None,
    learning_rate: float = 1e-3,
    epochs: int = 1,
    steps: int | None = None,
    batch_size: int | None = None,
    validation_batch_size: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    key: Any | None = None,
    prefetch: int = 2,
    gradient_accumulation: int = 1,
    normalization: OperatorNormalizationPolicy | Literal["fit"] | None = None,
    normalize_coordinates: bool = False,
    normalization_weighting: Literal["uniform", "quadrature"] = "uniform",
    dtype_policy: OperatorDTypePolicy | None = None,
    loss_scale_policy: OperatorLossScalePolicy | None = None,
    validation_policy: OperatorValidationPolicy | None = None,
    sharding_policy: OperatorShardingPolicy | None = None,
    jit: bool = True,
    callbacks: Sequence[TrainingCallback] = (),
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_every: int = 1,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 1,
    resume: bool = False,
    configuration: Mapping[str, Any] | None = None,
    artifact_id: str = "",
    provenance: dict[str, Any] | None = None,
) -> OperatorFitResult:
    """Fit a neural operator through one deterministic production control plane.

    ``evaluation_parameters`` maps ``(optimizer_state, training_parameters)`` to
    the parameter view used for validation, best-model selection, and returned
    execution models. ``parameter_subspace`` restricts differentiation and
    optimizer state to exact model leaves. Checkpointed fits require
    ``evaluation_parameters_id`` so resume cannot silently change that lifecycle
    contract.
    """
    if not isinstance(model, AbstractOperatorModel):
        raise TypeError("fit_operator requires a PhydraX operator model.")
    if parameter_subspace is None:
        parameter_paths: tuple[str, ...] | None = None
        if contains_low_rank_updates(model):
            raise ValueError(
                "Low-rank operator fitting requires an explicit parameter_subspace."
            )
    else:
        if not isinstance(parameter_subspace, ParameterSubspace):
            raise TypeError(
                "parameter_subspace must be a ParameterSubspace or None."
            )
        parameter_subspace.validate_root(model)
        validate_low_rank_subspace(model, parameter_subspace)
        parameter_paths = parameter_subspace.leaf_paths
    if int(epochs) < 0:
        raise ValueError("epochs must be non-negative.")
    if steps is not None and int(steps) < 0:
        raise ValueError("steps must be non-negative when provided.")
    if int(gradient_accumulation) <= 0:
        raise ValueError("gradient_accumulation must be positive.")
    if int(checkpoint_every) <= 0:
        raise ValueError("checkpoint_every must be positive.")
    if int(tensorboard_every) <= 0:
        raise ValueError("tensorboard_every must be positive.")
    if evaluation_parameters is None:
        if evaluation_parameters_id is not None:
            raise ValueError("evaluation_parameters_id requires evaluation_parameters.")
        resolved_evaluation_parameters_id = None
    else:
        if not callable(evaluation_parameters):
            raise TypeError("evaluation_parameters must be callable.")
        resolved_evaluation_parameters_id = (
            None
            if evaluation_parameters_id is None
            else str(evaluation_parameters_id).strip()
        )
        if checkpoint_path is not None and not resolved_evaluation_parameters_id:
            raise ValueError(
                "Checkpointed fits with evaluation_parameters require a stable "
                "evaluation_parameters_id."
            )
    if task is not None and not isinstance(task, OperatorTask):
        raise TypeError("task must be an OperatorTask.")
    if task is None and training_evidence is not None:
        raise ValueError("training_evidence requires a task-bound fit.")
    if output_pipeline is not None:
        if task is None:
            raise ValueError("output_pipeline requires a task-bound fit.")
        if not isinstance(output_pipeline, OperatorOutputPipeline):
            raise TypeError("output_pipeline must be an OperatorOutputPipeline.")
    if optimizer is None:
        if learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative.")
        optimizer = optax.adam(float(learning_rate))
        resolved_optimizer_id = f"optax.adam:{float(learning_rate):.17g}"
    else:
        if not optimizer_id:
            raise ValueError("Custom optimizers require a stable optimizer_id.")
        resolved_optimizer_id = str(optimizer_id)

    raw_train_loader = _raw_loader(
        train,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        prefetch=prefetch,
        split="train",
    )
    raw_validation_loader = (
        None
        if validation is None
        else _raw_loader(
            validation,
            batch_size=validation_batch_size,
            shuffle=False,
            seed=seed,
            prefetch=prefetch,
            split="validation",
        )
    )
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    resume_manifest: dict[str, Any] | None = None
    resume_probe: tuple[int, int] | None = None
    if checkpoint is not None and resume and (checkpoint / "manifest.json").is_file():
        resume_manifest, _ = _read_operator_training_manifest(checkpoint)
    current_data_contract = {
        "train_loader_fingerprint": raw_train_loader.fingerprint,
        "validation_loader_fingerprint": (
            None if raw_validation_loader is None else raw_validation_loader.fingerprint
        ),
    }
    if resume_manifest is not None:
        metadata = resume_manifest["metadata"]
        data_contract = metadata.get("data_contract")
        if data_contract != current_data_contract:
            raise ValueError("Operator fit checkpoint data contract mismatch.")
        saved_progress = metadata.get("progress")
        if not isinstance(saved_progress, dict):
            raise ValueError("Operator fit checkpoint progress is missing or invalid.")
        probe_epoch = int(saved_progress["epoch"])
        probe_batch = int(saved_progress["next_batch_index"])
        if probe_epoch < 0:
            raise ValueError("Operator fit checkpoint epoch is invalid.")
        plan = raw_train_loader.epoch_plan(probe_epoch)
        if probe_batch < 0 or probe_batch > plan.batch_count:
            raise ValueError("Operator fit checkpoint batch cursor is invalid.")
        if probe_batch == plan.batch_count:
            probe_epoch += 1
            probe_batch = 0
            plan = raw_train_loader.epoch_plan(probe_epoch)
        if probe_epoch >= int(epochs):
            probe_epoch = max(0, int(epochs) - 1)
            probe_batch = 0
            plan = raw_train_loader.epoch_plan(probe_epoch)
        if plan.batch_count == 0:
            raise ValueError("Training data must contain at least one batch.")
        resume_probe = (probe_epoch, probe_batch)
    else:
        plan = raw_train_loader.epoch_plan(0)
        if plan.batch_count == 0:
            raise ValueError("Training data must contain at least one batch.")
        probe_epoch = 0
        probe_batch = 0
    first_raw = raw_train_loader.prepare_indices(
        plan.batch(probe_batch),
        epoch=probe_epoch,
        batch_index=probe_batch,
    )

    resolved_dtype = OperatorDTypePolicy() if dtype_policy is None else dtype_policy
    if not isinstance(resolved_dtype, OperatorDTypePolicy):
        raise TypeError("dtype_policy must be an OperatorDTypePolicy.")
    if loss_scale_policy is not None and not isinstance(
        loss_scale_policy,
        OperatorLossScalePolicy,
    ):
        raise TypeError("loss_scale_policy must be an OperatorLossScalePolicy.")
    _validate_training_precision(resolved_dtype, loss_scale_policy)
    model = resolved_dtype.cast_model(model)
    if sharding_policy is not None:
        model = replicate_operator_model(model, sharding_policy)
    if parameter_paths is None:
        effective_parameter_shapes: tuple[tuple[int, ...], ...] = ()
        effective_parameter_dtypes: tuple[str, ...] = ()
        effective_parameter_dimension = None
    else:
        assert parameter_subspace is not None
        effective_subspace = parameter_subspace.rebase(model, exact_dtype=False)
        validate_low_rank_subspace(model, effective_subspace)
        effective_parameter_shapes = effective_subspace.leaf_shapes
        effective_parameter_dtypes = effective_subspace.leaf_dtypes
        effective_parameter_dimension = effective_subspace.total_dimension

    def partition_fit_model(current_model):
        if parameter_paths is None:
            return partition_trainable(current_model)
        current_subspace = ParameterSubspace.from_leaf_paths(
            current_model,
            parameter_paths,
        )
        if current_subspace.leaf_shapes != effective_parameter_shapes:
            raise ValueError("Operator fit parameter-subspace shapes changed.")
        if current_subspace.leaf_dtypes != effective_parameter_dtypes:
            raise ValueError("Operator fit parameter-subspace dtypes changed.")
        validate_low_rank_subspace(current_model, current_subspace)
        return current_subspace.initial, current_subspace.frozen

    def reconstruct_fit_model(current_parameters, current_fixed):
        if parameter_paths is None:
            return combine_trainable(current_parameters, current_fixed)
        return eqx.combine(current_parameters, current_fixed)

    if normalization == "fit":
        if not isinstance(train, OperatorDataset):
            raise ValueError("normalization='fit' requires an in-memory OperatorDataset.")
        if not train.targets.fields:
            raise ValueError(
                "normalization='fit' requires supervised targets; targetless physics "
                "training must use explicit physical scaling or a fitted policy."
            )
        normalization_batch, normalization_targets = _nondimensionalize(
            train.batch,
            train.targets,
            task,
        )
        resolved_normalization = fit_operator_normalization(
            normalization_batch,
            normalization_targets,
            normalize_coordinates=normalize_coordinates,
            weighting=normalization_weighting,
            fields=() if task is None else task.fields,
        )
    else:
        resolved_normalization = normalization
    if resolved_normalization is not None and not isinstance(
        resolved_normalization, OperatorNormalizationPolicy
    ):
        raise TypeError(
            "normalization must be an OperatorNormalizationPolicy, 'fit', or None."
        )

    first = _place_batch(
        first_raw,
        task=task,
        normalization=resolved_normalization,
        dtype_policy=resolved_dtype,
        sharding_policy=sharding_policy,
    )
    physical_first = first.batch if first.physical_batch is None else first.physical_batch
    physical_targets = (
        first.targets if first.physical_targets is None else first.physical_targets
    )
    physical_targets.validate(physical_first)
    evidence = training_evidence
    if task is not None:
        if task.problem.source_query_relation is None:
            raise ValueError(
                "Task problem.source_query_relation must be explicit for fitting."
            )
        if task.problem.query_is_fixed is None:
            raise ValueError("Task problem.query_is_fixed must be explicit for fitting.")
        if evidence is None:
            evidence = OperatorTrainingEvidence(model.operator_contract.training.regime)
        task.validate_batch(physical_first)
        model.operator_contract.validate(
            physical_first,
            problem=task.problem,
            training_evidence=evidence,
            fields=task.fields,
        ).require()
    else:
        model.operator_contract.validate(physical_first).require_runtime()

    fixed_query_fingerprints: dict[str, str] = {}
    if task is not None and (
        task.problem.query_is_fixed is True
        or model.operator_contract.capabilities.requires_fixed_query
    ):
        fixed_query_fingerprints = raw_train_loader.fixed_query_fingerprints(
            tuple(task.query_by_name)
        )
        if raw_validation_loader is not None:
            validation_fixed_queries = raw_validation_loader.fixed_query_fingerprints(
                tuple(task.query_by_name)
            )
            if validation_fixed_queries != fixed_query_fingerprints:
                raise ValueError(
                    "Validation fixed queries differ from the training discretization."
                )

    resolved_output_map = _resolve_output_map(
        model,
        first.targets,
        output_field_map,
        task,
    )
    if loss_terms is None and not first.targets.fields:
        raise ValueError(
            "Targetless operator fitting requires explicit physics loss_terms."
        )
    terms = (
        _default_losses(resolved_output_map, physical_names=task is not None)
        if loss_terms is None
        else tuple(loss_terms)
    )
    if not terms:
        raise ValueError("fit_operator requires at least one operator loss term.")
    if any(not isinstance(term, AbstractOperatorLossTerm) for term in terms):
        raise TypeError("loss_terms must contain AbstractOperatorLossTerm instances.")
    term_names = tuple(term.name for term in terms)
    if len(set(term_names)) != len(term_names):
        raise ValueError("Operator loss term names must be unique.")

    validation_config = (
        OperatorValidationPolicy()
        if validation is not None and validation_policy is None
        else validation_policy
    )
    if validation_config is not None and raw_validation_loader is None:
        raise ValueError("validation_policy requires validation data.")
    model_labels = model_loss_labels(model) if include_model_losses else ()
    metric_names = (
        ("loss",) + term_names + tuple(f"model_loss/{label}" for label in model_labels)
    )
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("Training metric names must be unique.")

    def predict_for_loss(
        evaluated_model,
        batch,
        physical_batch,
        key,
    ):
        if output_pipeline is None:
            raw_prediction = _operator_prediction(
                evaluated_model,
                batch,
                key,
                resolved_dtype,
            )
            if task is None:
                return raw_prediction, raw_prediction
            physical_prediction = physicalize_prediction(
                raw_prediction,
                physical_batch,
                task,
                resolved_output_map,
                resolved_normalization,
            )
            return raw_prediction, physical_prediction
        assert task is not None
        model_key, pipeline_key = split_eval_key(key, 2)
        raw_prediction = _operator_prediction(
            evaluated_model,
            batch,
            model_key,
            resolved_dtype,
        )
        physical_prediction = physicalize_prediction(
            raw_prediction,
            physical_batch,
            task,
            resolved_output_map,
            resolved_normalization,
        )
        constrained_prediction = output_pipeline(
            physical_prediction,
            physical_batch,
            key=pipeline_key,
        )
        execution_prediction = executionize_prediction(
            constrained_prediction,
            raw_prediction,
            batch,
            task,
            resolved_output_map,
            resolved_normalization,
        )
        return execution_prediction, constrained_prediction

    if output_pipeline is not None:
        predict_for_loss(
            inference_mode(model),
            first.batch,
            physical_first,
            jr.key(seed),
        )

    parameters, fixed = partition_fit_model(model)
    optimizer_state = optimizer.init(parameters)
    evaluated_parameters = resolve_evaluation_parameters(
        evaluation_parameters,
        optimizer_state,
        parameters,
    )
    evaluation_model = reconstruct_fit_model(evaluated_parameters, fixed)
    gradient_accumulator = _tree_zeros(parameters)
    accumulated_cases = 0.0
    accumulated_microsteps = 0
    accumulated_metrics = [0.0] * len(metric_names)
    reduction_dtype = jnp.dtype(resolved_dtype.reduction_dtype)
    loss_scale_state = (
        OperatorLossScaleState(jnp.asarray(1.0, dtype=reduction_dtype))
        if loss_scale_policy is None
        else loss_scale_policy.initial_state(reduction_dtype)
    )

    def loss_components(
        current_model,
        batch,
        targets,
        physical_batch,
        physical_targets,
        key,
        step,
        *,
        training: bool,
    ):
        storage_model = current_model if training else inference_mode(current_model)
        evaluated_model = resolved_dtype.compute_model(storage_model)
        execution_prediction, physical_prediction = predict_for_loss(
            evaluated_model,
            batch,
            physical_batch,
            key,
        )
        context = OperatorLossContext(
            execution_prediction=execution_prediction,
            execution_batch=batch,
            execution_targets=targets,
            physical_prediction=physical_prediction,
            physical_batch=physical_batch,
            physical_targets=physical_targets,
            normalization=resolved_normalization,
            task=task,
        )
        values = tuple(
            resolved_dtype.reduction(
                term(
                    evaluated_model,
                    physical_prediction,
                    physical_batch,
                    physical_targets,
                    key=jr.fold_in(key, index),
                    step=step,
                    training=training,
                    context=context,
                )
            )
            for index, term in enumerate(terms)
        )
        attached = (
            tuple(
                resolved_dtype.reduction(value)
                for value in model_loss_values(
                    evaluated_model,
                    key=jr.fold_in(key, len(terms)),
                    iter_=step,
                )
            )
            if include_model_losses
            else ()
        )
        components = values + attached
        total = sum(components, start=jnp.asarray(0.0, dtype=jnp.float32))
        return resolved_dtype.reduction(total), components

    def gradient_fn(
        current_parameters,
        batch,
        targets,
        physical_batch,
        physical_targets,
        key,
        step,
        loss_scale_state_,
    ):
        def objective(candidate):
            current_model = reconstruct_fit_model(candidate, fixed)
            total, components = loss_components(
                current_model,
                batch,
                targets,
                physical_batch,
                physical_targets,
                key,
                step,
                training=True,
            )
            scaled = (
                total
                if loss_scale_policy is None
                else loss_scale_policy.scale_loss(total, loss_scale_state_)
            )
            return scaled, (total, components)

        (_, (total, components)), gradient = eqx.filter_value_and_grad(
            objective,
            has_aux=True,
        )(current_parameters)
        if loss_scale_policy is not None:
            gradient = loss_scale_policy.unscale_gradients(
                gradient,
                loss_scale_state_,
            )
        finite = tree_all_finite((gradient, total, components))
        if sharding_policy is not None:
            finite = eqx.filter_shard(finite, sharding_policy.replicated)
        return total, components, gradient, finite

    def update_fn(current_parameters, current_state, gradient):
        updates, next_state = optimizer.update(
            gradient,
            current_state,
            current_parameters,
        )
        next_parameters = eqx.apply_updates(current_parameters, updates)
        finite = tree_all_finite((next_parameters, next_state))
        if sharding_policy is not None:
            finite = eqx.filter_shard(finite, sharding_policy.replicated)
        return next_parameters, next_state, finite

    run_gradient_fn = eqx.filter_jit(gradient_fn) if jit else gradient_fn
    run_update_fn = eqx.filter_jit(update_fn) if jit else update_fn

    def prepared_epoch(
        loader: OperatorBatchLoader,
        epoch: int,
        *,
        start_batch: int = 0,
        retained_first: OperatorTrainingBatch | None = None,
    ):
        next_batch = int(start_batch)
        retained_batch = None
        if retained_first is not None:
            if (
                retained_first.epoch != int(epoch)
                or retained_first.batch_index != next_batch
            ):
                raise ValueError(
                    "Retained training probe does not match the resume cursor."
                )
            retained_batch = _place_batch(
                retained_first,
                task=task,
                normalization=resolved_normalization,
                dtype_policy=resolved_dtype,
                sharding_policy=sharding_policy,
            )
            next_batch += 1
        with loader.epoch(epoch, start_batch=next_batch) as batches:
            if retained_batch is not None:
                yield retained_batch
            for raw in batches:
                yield _place_batch(
                    raw,
                    task=task,
                    normalization=resolved_normalization,
                    dtype_policy=resolved_dtype,
                    sharding_policy=sharding_policy,
                )

    def evaluate(current_model, loader: OperatorBatchLoader, step: int):
        totals = [0.0] * len(metric_names)
        total_cases = 0
        for batch_index, training_batch in enumerate(prepared_epoch(loader, 0)):
            count = _case_count(training_batch.batch)
            total, components = loss_components(
                current_model,
                training_batch.batch,
                training_batch.targets,
                (
                    training_batch.batch
                    if training_batch.physical_batch is None
                    else training_batch.physical_batch
                ),
                (
                    training_batch.targets
                    if training_batch.physical_targets is None
                    else training_batch.physical_targets
                ),
                jr.fold_in(jr.fold_in(master_key, int(step)), 1000 + batch_index),
                jnp.asarray(step, dtype=float),
                training=False,
            )
            values = (total,) + components
            for index, value in enumerate(values):
                totals[index] += float(jax.device_get(value)) * count
            total_cases += count
        if total_cases == 0:
            raise ValueError("Evaluation data must contain at least one batch.")
        return {
            name: value / total_cases
            for name, value in zip(metric_names, totals, strict=True)
        }

    maximum_steps = (
        int(steps)
        if steps is not None
        else int(epochs)
        * ceil(raw_train_loader.batches_per_epoch / int(gradient_accumulation))
    )
    master_key = jr.key(seed) if key is None else key
    progress = TrainingProgress()
    control = TrainingController(
        total_steps=maximum_steps,
        key=master_key,
        progress=progress,
        callbacks=(
            callbacks
            if sharding_policy is None or sharding_policy.is_primary_process
            else ()
        ),
    )
    best_model = evaluation_model
    train_steps: list[int] = []
    train_history: list[dict[str, float]] = []
    validation_steps: list[int] = []
    validation_history: list[dict[str, float]] = []
    resumed_from_step = 0
    prior_training_seconds = 0.0

    fit_contract_data = {
        "model_contract": operator_contract_fingerprint(model.operator_contract),
        "task_fingerprint": None if task is None else task.fingerprint,
        "output_field_map": resolved_output_map,
        "loss_terms": [term.fingerprint for term in terms],
        "include_model_losses": bool(include_model_losses),
        "optimizer_id": resolved_optimizer_id,
        "gradient_accumulation": int(gradient_accumulation),
        "parameter_subspace": (
            None
            if parameter_paths is None
            else {
                "paths": list(parameter_paths),
                "shapes": [list(shape) for shape in effective_parameter_shapes],
                "dtypes": list(effective_parameter_dtypes),
                "total_dimension": effective_parameter_dimension,
            }
        ),
        "normalization": (
            None if resolved_normalization is None else resolved_normalization.to_dict()
        ),
        "fixed_query_fingerprints": fixed_query_fingerprints,
        "dtype_policy": resolved_dtype.to_dict(),
        "loss_scale_policy": (
            None if loss_scale_policy is None else asdict(loss_scale_policy)
        ),
        "output_pipeline": (
            None if output_pipeline is None else output_pipeline.fingerprint
        ),
        "validation_policy": (
            None if validation_config is None else asdict(validation_config)
        ),
        "train_loader": raw_train_loader.configuration(),
        "validation_loader": (
            None
            if raw_validation_loader is None
            else raw_validation_loader.configuration()
        ),
        "train_loader_fingerprint": raw_train_loader.fingerprint,
        "validation_loader_fingerprint": (
            None if raw_validation_loader is None else raw_validation_loader.fingerprint
        ),
        "sharding": (
            None
            if sharding_policy is None
            else {
                "mesh_axis": sharding_policy.mesh_axis,
                "case_axis": sharding_policy.case_axis,
                "mesh_shape": list(sharding_policy.mesh.devices.shape),
                "device_count": int(sharding_policy.mesh.devices.size),
            }
        ),
        "configuration": {} if configuration is None else dict(configuration),
    }
    if resolved_evaluation_parameters_id is not None:
        fit_contract_data["evaluation_parameters_id"] = resolved_evaluation_parameters_id
    fit_contract = _canonical_json(fit_contract_data)
    schema = {
        "fit": operator_fit_schema(first.batch, target=first.targets),
    }

    initial_metrics: dict[str, float]
    if resume_manifest is not None:
        assert checkpoint is not None
        state_template = (
            optimizer_state,
            gradient_accumulator,
            jnp.asarray(accumulated_cases),
            jnp.asarray(accumulated_microsteps, dtype=jnp.int32),
            loss_scale_state,
        )
        restored = load_operator_training_checkpoint(
            checkpoint,
            (model, best_model),
            state_template,
            expected_schema=schema,
        )
        if restored.metadata["fit_contract"] != fit_contract:
            raise ValueError("Operator fit checkpoint contract mismatch.")
        model, best_model = restored.model
        (
            optimizer_state,
            gradient_accumulator,
            accumulated_cases_array,
            accumulated_microsteps_array,
            loss_scale_state,
        ) = restored.optimizer_state
        accumulated_cases = float(jax.device_get(accumulated_cases_array))
        accumulated_microsteps = int(jax.device_get(accumulated_microsteps_array))
        metadata = restored.metadata
        progress = TrainingProgress(**metadata["progress"])
        if progress.update_step != restored.step:
            raise ValueError("Checkpoint progress disagrees with its update step.")
        if progress.update_step > maximum_steps:
            raise ValueError("Checkpoint step exceeds the requested training ceiling.")
        control = TrainingController(
            total_steps=maximum_steps,
            key=restored.key,
            progress=progress,
            callbacks=callbacks,
        )
        master_key = restored.key
        control.best_payload = best_model
        train_steps = [int(value) for value in metadata["train_steps"]]
        train_history = [dict(values) for values in metadata["train_metrics"]]
        validation_steps = [int(value) for value in metadata["validation_steps"]]
        validation_history = [dict(values) for values in metadata["validation_metrics"]]
        initial_metrics = dict(metadata["initial_metrics"])
        accumulated_metrics = [float(value) for value in metadata["accumulated_metrics"]]
        prior_training_seconds = float(metadata["training_seconds"])
        resumed_from_step = progress.update_step
        parameters, fixed = partition_fit_model(model)
        evaluated_parameters = resolve_evaluation_parameters(
            evaluation_parameters,
            optimizer_state,
            parameters,
        )
        evaluation_model = reconstruct_fit_model(evaluated_parameters, fixed)
    else:
        initial_metrics = evaluate(evaluation_model, raw_train_loader, 0)
        if raw_validation_loader is not None:
            validation_metrics = evaluate(
                evaluation_model,
                raw_validation_loader,
                0,
            )
            validation_steps.append(0)
            validation_history.append(validation_metrics)
            assert validation_config is not None
            if validation_config.monitor not in validation_metrics:
                raise KeyError(
                    f"Unknown validation monitor {validation_config.monitor!r}."
                )
            control.progress = replace(
                control.progress,
                best_value=validation_metrics[validation_config.monitor],
                best_step=0,
            )
            control.best_payload = evaluation_model

    def save_progress(training_seconds: float) -> None:
        if checkpoint is None:
            return
        primary = sharding_policy is None or sharding_policy.is_primary_process
        if not primary:
            sharding_policy.synchronize(
                f"fit_operator_checkpoint_{control.progress.update_step}"
            )
            return
        save_operator_training_checkpoint(
            checkpoint,
            (model, best_model),
            (
                optimizer_state,
                gradient_accumulator,
                jnp.asarray(accumulated_cases),
                jnp.asarray(accumulated_microsteps, dtype=jnp.int32),
                loss_scale_state,
            ),
            step=control.progress.update_step,
            key=master_key,
            normalization=resolved_normalization,
            dtype_policy=resolved_dtype,
            schema=schema,
            metadata={
                "fit_contract": fit_contract,
                "data_contract": current_data_contract,
                "progress": asdict(control.progress),
                "initial_metrics": initial_metrics,
                "train_steps": train_steps,
                "train_metrics": train_history,
                "validation_steps": validation_steps,
                "validation_metrics": validation_history,
                "accumulated_metrics": accumulated_metrics,
                "training_seconds": float(training_seconds),
            },
        )
        if sharding_policy is not None:
            sharding_policy.synchronize(
                f"fit_operator_checkpoint_{control.progress.update_step}"
            )
        control.emit("checkpoint", metrics={"step": control.progress.update_step})

    def consider_validation(
        metrics: Mapping[str, float],
        current_evaluation_model: AbstractOperatorModel,
        /,
    ) -> None:
        nonlocal best_model
        assert validation_config is not None
        score = float(metrics[validation_config.monitor])
        previous = control.progress.best_value
        strict_better = previous is None or (
            score < previous if validation_config.mode == "min" else score > previous
        )
        required = (
            float(validation_config.minimum_delta)
            if previous is None
            else max(
                float(validation_config.minimum_delta),
                float(validation_config.relative_minimum_delta)
                * max(abs(previous), 1e-12),
            )
        )
        meaningful = previous is None or (
            score < previous - required
            if validation_config.mode == "min"
            else score > previous + required
        )
        if strict_better:
            best_model = current_evaluation_model
            control.best_payload = current_evaluation_model
        stale = 0 if meaningful else control.progress.stale_validations + 1
        stopped = validation_config.patience is not None and stale >= int(
            validation_config.patience
        )
        control.progress = replace(
            control.progress,
            best_value=score if strict_better else previous,
            best_step=(
                control.progress.update_step
                if strict_better
                else control.progress.best_step
            ),
            stale_validations=stale,
            stopped_early=stopped,
        )
        if stopped:
            control.stop_requested = True

    logger_context = (
        nullcontext(None)
        if tensorboard_log_dir is None
        or (sharding_policy is not None and not sharding_policy.is_primary_process)
        else TensorBoardLogger(tensorboard_log_dir)
    )
    started = time.perf_counter()
    stopped_by_signal = False
    control.emit("train_begin", metrics=initial_metrics)
    with logger_context as tensorboard, TrainingSignalGuard() as signal_guard:
        if not control.progress.stopped_early and _has_trainable_arrays(parameters):
            for epoch in range(control.progress.epoch, int(epochs)):
                if control.stop_requested or signal_guard.stop_requested:
                    break
                control.emit("epoch_begin", metrics={"epoch": epoch})
                epoch_start_batch = control.progress.next_batch_index
                retained_first = (
                    first_raw if resume_probe == (epoch, epoch_start_batch) else None
                )
                for training_batch in prepared_epoch(
                    raw_train_loader,
                    epoch,
                    start_batch=epoch_start_batch,
                    retained_first=retained_first,
                ):
                    if control.progress.update_step >= maximum_steps:
                        break
                    if control.stop_requested or signal_guard.stop_requested:
                        break
                    key = control.key_for(control.progress.microstep, site=0)
                    total, components, gradient, finite_array = run_gradient_fn(
                        parameters,
                        training_batch.batch,
                        training_batch.targets,
                        (
                            training_batch.batch
                            if training_batch.physical_batch is None
                            else training_batch.physical_batch
                        ),
                        (
                            training_batch.targets
                            if training_batch.physical_targets is None
                            else training_batch.physical_targets
                        ),
                        key,
                        jnp.asarray(control.progress.update_step + 1, dtype=float),
                        loss_scale_state,
                    )
                    count = _case_count(training_batch.batch)
                    values = (total,) + components
                    finite = bool(jax.device_get(finite_array))
                    control.progress = replace(
                        control.progress,
                        microstep=control.progress.microstep + 1,
                        next_batch_index=training_batch.batch_index + 1,
                    )
                    if not finite:
                        gradient_accumulator = _tree_zeros(parameters)
                        accumulated_cases = 0.0
                        accumulated_microsteps = 0
                        accumulated_metrics = [0.0] * len(metric_names)
                        if loss_scale_policy is None or not loss_scale_policy.dynamic:
                            raise FloatingPointError(
                                "Non-finite operator loss or gradient encountered."
                            )
                        loss_scale_state = loss_scale_policy.on_nonfinite_microstep(
                            loss_scale_state
                        )
                        control.emit(
                            "nonfinite",
                            metrics={
                                "loss_scale": float(
                                    jax.device_get(loss_scale_state.scale)
                                ),
                                "nonfinite_microsteps": int(
                                    jax.device_get(loss_scale_state.nonfinite_microsteps)
                                ),
                            },
                        )
                        continue
                    gradient_accumulator = _tree_add_scaled(
                        gradient_accumulator,
                        gradient,
                        float(count),
                    )
                    accumulated_cases += float(count)
                    accumulated_microsteps += 1
                    for index, value in enumerate(values):
                        accumulated_metrics[index] += float(jax.device_get(value)) * count
                    if (
                        accumulated_microsteps < int(gradient_accumulation)
                        and training_batch.batch_index + 1
                        < raw_train_loader.batches_per_epoch
                    ):
                        continue

                    averaged_gradient = _tree_scale(
                        gradient_accumulator,
                        1.0 / accumulated_cases,
                    )
                    (
                        candidate_parameters,
                        candidate_optimizer_state,
                        candidate_finite_array,
                    ) = run_update_fn(
                        parameters,
                        optimizer_state,
                        averaged_gradient,
                    )
                    if not bool(jax.device_get(candidate_finite_array)):
                        raise FloatingPointError(
                            "Operator optimizer produced non-finite state from "
                            "finite gradients."
                        )
                    parameters = candidate_parameters
                    optimizer_state = candidate_optimizer_state
                    model = reconstruct_fit_model(parameters, fixed)
                    update_step = control.progress.update_step + 1
                    control.complete_update(update_step)
                    metrics = {
                        name: value / accumulated_cases
                        for name, value in zip(
                            metric_names,
                            accumulated_metrics,
                            strict=True,
                        )
                    }
                    train_steps.append(update_step)
                    train_history.append(metrics)
                    gradient_accumulator = _tree_zeros(parameters)
                    accumulated_cases = 0.0
                    accumulated_microsteps = 0
                    accumulated_metrics = [0.0] * len(metric_names)
                    if loss_scale_policy is not None:
                        loss_scale_state = loss_scale_policy.on_finite_update(
                            loss_scale_state
                        )
                    control.emit("batch_end", metrics=metrics)
                    if (
                        tensorboard is not None
                        and update_step % int(tensorboard_every) == 0
                    ):
                        for name, value in metrics.items():
                            tensorboard.scalar(f"train/{name}", value, update_step)
                        tensorboard.scalar(
                            "train/loss_scale",
                            float(jax.device_get(loss_scale_state.scale)),
                            update_step,
                        )
                    if (
                        raw_validation_loader is not None
                        and validation_config is not None
                        and update_step % int(validation_config.every) == 0
                    ):
                        evaluated_parameters = resolve_evaluation_parameters(
                            evaluation_parameters,
                            optimizer_state,
                            parameters,
                        )
                        evaluation_model = reconstruct_fit_model(
                            evaluated_parameters,
                            fixed,
                        )
                        validation_metrics = evaluate(
                            evaluation_model,
                            raw_validation_loader,
                            update_step,
                        )
                        validation_steps.append(update_step)
                        validation_history.append(validation_metrics)
                        if validation_config.monitor not in validation_metrics:
                            raise KeyError(
                                f"Unknown validation monitor {validation_config.monitor!r}."
                            )
                        consider_validation(validation_metrics, evaluation_model)
                        control.emit("validation_end", metrics=validation_metrics)
                        if tensorboard is not None:
                            for name, value in validation_metrics.items():
                                tensorboard.scalar(
                                    f"validation/{name}",
                                    value,
                                    update_step,
                                )
                    elapsed = prior_training_seconds + time.perf_counter() - started
                    if (
                        checkpoint is not None
                        and update_step % int(checkpoint_every) == 0
                    ):
                        save_progress(elapsed)
                    if control.stop_requested:
                        break
                if (
                    control.progress.next_batch_index
                    >= raw_train_loader.batches_per_epoch
                ):
                    control.progress = replace(
                        control.progress,
                        epoch=epoch + 1,
                        next_batch_index=0,
                    )
                if control.progress.update_step >= maximum_steps:
                    break
        stopped_by_signal = signal_guard.stop_requested

    training_seconds = prior_training_seconds + time.perf_counter() - started
    evaluated_parameters = resolve_evaluation_parameters(
        evaluation_parameters,
        optimizer_state,
        parameters,
    )
    evaluation_model = reconstruct_fit_model(evaluated_parameters, fixed)
    if (
        raw_validation_loader is not None
        and validation_config is not None
        and (not validation_steps or validation_steps[-1] != control.progress.update_step)
    ):
        validation_metrics = evaluate(
            evaluation_model,
            raw_validation_loader,
            control.progress.update_step,
        )
        validation_steps.append(control.progress.update_step)
        validation_history.append(validation_metrics)
        consider_validation(validation_metrics, evaluation_model)
    save_progress(training_seconds)
    selected_model = (
        best_model
        if validation_config is not None and validation_config.select_best
        else evaluation_model
    )
    final_metrics = evaluate(
        selected_model,
        raw_train_loader,
        control.progress.update_step,
    )
    control.emit("train_end", metrics=final_metrics)

    trained = None
    if task is not None:
        assert evidence is not None
        trained = TrainedOperator(
            selected_model,
            task,
            training_evidence=evidence,
            output_field_map=resolved_output_map,
            fixed_query_fingerprints=fixed_query_fingerprints,
            output_pipeline=output_pipeline,
            normalization=resolved_normalization,
            dtype_policy=resolved_dtype,
            sharding_policy=sharding_policy,
            compilation_strategy="compiled" if jit else "eager",
            artifact_id=artifact_id,
            provenance=provenance,
        )
    history = OperatorFitHistory(
        initial_metrics=frozendict(initial_metrics),
        train_steps=tuple(train_steps),
        train_metrics=tuple(frozendict(values) for values in train_history),
        validation_steps=tuple(validation_steps),
        validation_metrics=tuple(frozendict(values) for values in validation_history),
        final_metrics=frozendict(final_metrics),
    )
    return OperatorFitResult(
        execution_model=selected_model,
        last_execution_model=evaluation_model,
        trained_operator=trained,
        output_field_map=frozendict(resolved_output_map),
        output_pipeline=output_pipeline,
        history=history,
        normalization=resolved_normalization,
        dtype_policy=resolved_dtype,
        precision_evidence=resolved_dtype.precision_evidence,
        loss_scale_state=(None if loss_scale_policy is None else loss_scale_state),
        progress=control.progress,
        resumed_from_step=resumed_from_step,
        training_seconds=training_seconds,
        checkpoint_path=checkpoint,
        stopped_by_signal=stopped_by_signal,
        stopped_by_callback=control.stop_requested and not control.progress.stopped_early,
    )


__all__ = [
    "OperatorFitHistory",
    "OperatorFitResult",
    "OperatorValidationPolicy",
    "fit_operator",
]
