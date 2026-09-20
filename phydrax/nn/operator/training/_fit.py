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
from math import ceil
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ...._execution_runtime import ExecutionGroup
from ...._frozendict import frozendict
from ...._iteration import IterationSession
from ...._trainable import combine_trainable, partition_trainable
from ...._training import (
    _update_validation_selection,
    DelayedTargetPolicy,
    EvaluationParametersFn,
    ExponentialMovingAverageTargetPolicy,
    resolve_evaluation_parameters,
    TargetParameterState,
    TensorBoardLogger,
    TrainingController,
    TrainingIterationKind,
    TrainingProgress,
    TrainingSignalGuard,
)
from ...._training_objective import (
    _combine_objective_contributions,
    _GradientAccumulationState,
    _ObjectiveAccumulator,
    _ObjectiveContribution,
)
from ...._tree_math import tree_negative, tree_where
from ....optim import (
    OptimizerStateCompressionPolicy,
    prepare_compressed_optimizer,
)
from ....optim._gradient_composition import (
    conflict_free_gradient,
    ConflictFreeGradientPolicy,
)
from ....optim._update_alignment import (
    _alignment_conflicts,
    ConflictFreeUpdatePolicy,
    ConflictFreeUpdateStatistics,
    project_conflict_free_direction,
)
from ....privacy import PrivacyCertificate, PrivateTrainingPlan
from ....privacy._provider import (
    _prepare_private_gradient,
    _PreparedPrivateGradient,
)
from ..._loss import model_loss_labels, model_loss_values
from ...layers._dropout import inference_mode
from ...parameters import ParameterSubspace
from ...parameters._low_rank import (
    contains_low_rank_updates,
    validate_low_rank_subspace,
)
from ..capabilities import OperatorTrainingEvidence
from ..data import OperatorBatch, OperatorTargetBatch, slice_operator_batch
from ..engine import AbstractOperatorModel
from ..metrics import operator_l2_loss
from ..sampling import InMemoryOperatorCaseSource, OperatorCaseSource
from ..sharding import (
    OperatorShardingPolicy,
    replicate_operator_model,
    shard_operator_batch,
    shard_operator_case_array,
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
    _evaluate_operator_step,
    _operator_prediction,
    nondimensionalize_batch,
    nondimensionalize_targets,
)
from ._fingerprint import operator_fit_schema
from ._loader import (
    _pad_case_payload,
    OperatorBatchLoader,
    OperatorTrainingBatch,
)
from ._loss_scale import (
    OperatorLossScalePolicy,
    OperatorLossScaleState,
    tree_all_finite,
)
from ._losses import (
    _case_mean_contribution,
    _weighted_case_reduction,
    AbstractOperatorLossTerm,
    OperatorLossContext,
    ResidualOperatorRolloutLoss,
    SupervisedOperatorLoss,
    SupervisedOperatorRolloutLoss,
)
from ._normalization import (
    fit_operator_normalization,
    OperatorNormalizationPolicy,
)
from ._physics import OperatorOutputPipeline
from ._privacy import prepare_private_operator_batch
from ._rollout import (
    _operator_rollout_scan,
    _ROLLOUT_MODEL_KEY_DOMAIN,
    _validate_rollout_route,
    OperatorRolloutPolicy,
    OperatorRolloutRoute,
)
from ._target_consistency import TargetOperatorConsistencyLoss
from ._trained_operator import (
    operator_contract_fingerprint,
    TrainedOperator,
)


_LOSS_TERM_KEY_DOMAIN = 200
_RESIDUAL_ROLLOUT_KEY_DOMAIN = 300
_MODEL_OBJECTIVE_KEY_DOMAIN = 400
_PRIVACY_NOISE_KEY_DOMAIN = 700
_PRIVACY_SAMPLER_KEY_DOMAIN = 701


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
    update_alignment_statistics: ConflictFreeUpdateStatistics | None
    privacy_certificate: PrivacyCertificate | None
    stopped_by_signal: bool = False
    stopped_by_host_control: bool = False

    @property
    def initial_loss(self) -> float:
        if "loss" not in self.history.initial_metrics:
            raise ValueError("Raw initial loss was not released by this training run.")
        return self.history.initial_metrics["loss"]

    @property
    def final_loss(self) -> float:
        if "loss" not in self.history.final_metrics:
            raise ValueError("Raw final loss was not released by this training run.")
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
    sharding_policy: OperatorShardingPolicy | None,
) -> OperatorBatchLoader:
    if isinstance(data, OperatorBatchLoader):
        return OperatorBatchLoader(
            data.source,
            batch_size=data.batch_size,
            shuffle=data.shuffle,
            seed=data.seed,
            drop_last=data.drop_last,
            prefetch=data.prefetch,
            sharding_policy=sharding_policy,
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
        sharding_policy=sharding_policy,
        split=split,
    )


def _nondimensionalize(
    batch: OperatorBatch,
    targets: OperatorTargetBatch,
    task: OperatorTask | None,
    /,
    *,
    target_aliases: Mapping[str, str] | None = None,
) -> tuple[OperatorBatch, OperatorTargetBatch]:
    if task is None:
        return batch, targets
    task.validate_batch(batch)
    return (
        nondimensionalize_batch(batch, task),
        nondimensionalize_targets(
            targets,
            task,
            target_aliases=target_aliases,
        ),
    )


def _place_batch(
    raw: OperatorTrainingBatch,
    /,
    *,
    task: OperatorTask | None,
    normalization: OperatorNormalizationPolicy | None,
    dtype_policy: OperatorDTypePolicy,
    sharding_policy: OperatorShardingPolicy | None,
    target_aliases: Mapping[str, str] | None = None,
) -> OperatorTrainingBatch:
    physical_batch = raw.batch if raw.physical_batch is None else raw.physical_batch
    physical_targets = (
        raw.targets if raw.physical_targets is None else raw.physical_targets
    )
    case_log_weights = raw.case_log_weights
    case_mask = raw.case_mask
    if sharding_policy is not None:
        divisor = sharding_policy.data_axis_size
        size = int(physical_batch.case_shape[0])
        capacity = ((size + divisor - 1) // divisor) * divisor
        (
            physical_batch,
            physical_targets,
            case_log_weights,
            case_mask,
        ) = _pad_case_payload(
            physical_batch,
            physical_targets,
            case_log_weights,
            case_mask,
            capacity=capacity,
        )
    sampling_probabilities = jnp.ones(
        jnp.asarray(case_log_weights).shape,
        dtype=dtype_policy.reduction_dtype,
    )
    batch, targets = _nondimensionalize(
        physical_batch,
        physical_targets,
        task,
        target_aliases=target_aliases,
    )
    if normalization is not None:
        batch = normalization.normalize_batch(batch)
        targets = normalization.normalize_targets(
            targets,
            target_aliases=target_aliases,
        )
    batch = dtype_policy.cast_batch(batch)
    targets = dtype_policy.cast_targets(targets)
    case_log_weights = jnp.asarray(
        case_log_weights,
        dtype=dtype_policy.reduction_dtype,
    )
    case_mask = jnp.asarray(case_mask, dtype=jnp.bool_)
    if sharding_policy is not None:
        batch = shard_operator_batch(batch, sharding_policy)
        targets = shard_operator_targets(targets, sharding_policy)
        physical_batch = shard_operator_batch(physical_batch, sharding_policy)
        physical_targets = shard_operator_targets(
            physical_targets,
            sharding_policy,
        )
        case_log_weights = shard_operator_case_array(
            case_log_weights,
            sharding_policy,
        )
        case_mask = shard_operator_case_array(case_mask, sharding_policy)
        sampling_probabilities = shard_operator_case_array(
            sampling_probabilities,
            sharding_policy,
        )
    else:
        case_log_weights = jax.device_put(case_log_weights)
        case_mask = jax.device_put(case_mask)
        sampling_probabilities = jax.device_put(sampling_probabilities)
    return replace(
        raw,
        batch=batch,
        targets=targets,
        case_log_weights=case_log_weights,
        case_mask=case_mask,
        sampling_probabilities=sampling_probabilities,
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
        tuple(field.name for field in task.target_fields)
        if task is not None
        else (tuple(targets.fields) if targets.fields else declared)
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
            "output_field_map must bijectively map every model output to a physical output field."
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
        name: float(jax.device_get(jnp.asarray(value, dtype=jnp.float64).reshape(())))
        for name, value in zip(names, values, strict=True)
    }


def _rollout_target_aliases(
    model: AbstractOperatorModel,
    terms: Sequence[AbstractOperatorLossTerm],
    route: OperatorRolloutRoute | None,
    policy: OperatorRolloutPolicy | None,
    task: OperatorTask | None,
    /,
) -> dict[str, str]:
    rollout_terms = tuple(
        term
        for term in terms
        if isinstance(
            term,
            (SupervisedOperatorRolloutLoss, ResidualOperatorRolloutLoss),
        )
    )
    if not rollout_terms:
        if route is not None or policy is not None:
            raise ValueError(
                "rollout_route and rollout_policy require rollout loss terms."
            )
        return {}
    if task is None:
        raise ValueError("Operator rollout losses require a task-bound fit.")
    if not isinstance(route, OperatorRolloutRoute):
        raise TypeError("Operator rollout losses require an OperatorRolloutRoute.")
    if not isinstance(policy, OperatorRolloutPolicy):
        raise TypeError("Operator rollout losses require an OperatorRolloutPolicy.")
    if not model.operator_contract.capabilities.autoregressive_rollout:
        raise ValueError("The configured operator architecture does not support rollout.")
    if int(policy.maximum_horizon) > int(task.problem.rollout_steps):
        raise ValueError(
            "rollout_policy.maximum_horizon exceeds the task rollout contract."
        )
    supervised_terms = tuple(
        term for term in rollout_terms if isinstance(term, SupervisedOperatorRolloutLoss)
    )
    if len(supervised_terms) > 1:
        raise ValueError(
            "Operator rollout training accepts one ordered future-target route."
        )
    if route.task_field not in task.field_by_name:
        raise KeyError(f"Unknown rollout task field {route.task_field!r}.")
    task_targets = {field.name for field in task.target_fields}
    aliases: dict[str, str] = {}
    for term in rollout_terms:
        if len(term.time_weights) != int(policy.maximum_horizon):
            raise ValueError(
                f"Rollout loss {term.name!r} must provide one time weight per maximum-horizon step."
            )
        if any(
            sum(term.time_weights[:horizon]) <= 0.0
            for horizon in range(
                int(policy.initial_horizon),
                int(policy.maximum_horizon) + 1,
            )
        ):
            raise ValueError(
                f"Rollout loss {term.name!r} must have positive time-weight mass at every reachable horizon."
            )
        if isinstance(term, SupervisedOperatorRolloutLoss):
            if len(term.target_fields) != int(policy.maximum_horizon):
                raise ValueError(
                    f"Rollout loss {term.name!r} must provide one ordered target alias per maximum-horizon step."
                )
            for alias in term.target_fields:
                if alias in task_targets:
                    raise ValueError(
                        "Future-step aliases must not reuse canonical task output names."
                    )
                aliases[alias] = route.task_field
    return aliases


def _scan_step_value(tree: Any, index: int, /) -> Any:
    if isinstance(tree, tuple):
        return tree[index]
    return jax.tree_util.tree_map(
        lambda value: value[index] if eqx.is_array(value) else value,
        tree,
    )


def _resolve_operator_fit_execution(
    model: AbstractOperatorModel,
    /,
    *,
    key: Any,
    seed: int,
    privacy: PrivateTrainingPlan | None,
    execution_group: ExecutionGroup | None,
    sharding_policy: OperatorShardingPolicy | None,
    gradient_composition: ConflictFreeGradientPolicy | None,
    update_alignment: ConflictFreeUpdatePolicy | None,
    loss_scale_policy: OperatorLossScalePolicy | None,
    gradient_accumulation: int,
    include_model_losses: bool,
    normalization: Any,
    validation: Any,
    tensorboard_log_dir: Any,
    batch_size: int | None,
    epochs: int,
    steps: int | None,
) -> tuple[Any, OperatorShardingPolicy | None]:
    if not isinstance(model, AbstractOperatorModel):
        raise TypeError("fit_operator requires a PhydraX operator model.")
    master_key = jr.key(seed) if key is None else key
    if privacy is not None:
        if not isinstance(privacy, PrivateTrainingPlan):
            raise TypeError("privacy must be a PrivateTrainingPlan or None.")
        if execution_group is not None or sharding_policy is not None:
            raise ValueError(
                "Private operator training initially supports one process/device."
            )
        if gradient_composition is not None or update_alignment is not None:
            raise ValueError(
                "Private operator training does not expose unnoised objective gradients "
                "to gradient composition or update alignment."
            )
        if loss_scale_policy is not None:
            raise ValueError("Private operator training does not support loss scaling.")
        if int(gradient_accumulation) != 1:
            raise ValueError(
                "Private operator training uses provider microbatching and requires gradient_accumulation=1."
            )
        if include_model_losses:
            raise ValueError(
                "Private operator training requires include_model_losses=False until "
                "parameter-only public objectives have an explicit partition."
            )
        if normalization == "fit":
            raise ValueError(
                "Private operator training requires public or separately certified normalization."
            )
        if validation is not None and not privacy.validation_is_public:
            raise ValueError(
                "Validation data must be explicitly public under private training."
            )
        if tensorboard_log_dir is not None:
            raise ValueError(
                "Private operator training does not release raw TensorBoard metrics."
            )
        if batch_size is not None:
            raise ValueError(
                "Private operator batches are owned by the privacy sampler; batch_size must be None."
            )
        if int(epochs) != 1:
            raise ValueError(
                "Private operator training uses its fixed mechanism iteration schedule and requires epochs=1."
            )
        if steps is not None and not (0 <= int(steps) <= privacy.mechanism.iterations):
            raise ValueError(
                "steps must lie within the private mechanism iteration schedule."
            )
    if execution_group is not None:
        if not isinstance(execution_group, ExecutionGroup):
            raise TypeError("execution_group must be an ExecutionGroup or None.")
        if sharding_policy is not None:
            raise ValueError(
                "execution_group and sharding_policy are mutually exclusive."
            )
        sharding_policy = OperatorShardingPolicy.from_execution_group(execution_group)
        if normalization == "fit" and jax.process_count() > 1:
            raise ValueError(
                "Multi-process fitting requires an explicit fitted normalization "
                "policy; normalization='fit' would materialize global source data."
            )
    return master_key, sharding_policy


def _resolve_operator_parameter_paths(
    model: AbstractOperatorModel,
    parameter_subspace: ParameterSubspace | None,
    /,
) -> tuple[str, ...] | None:
    if parameter_subspace is None:
        parameter_paths: tuple[str, ...] | None = None
        if contains_low_rank_updates(model):
            raise ValueError(
                "Low-rank operator fitting requires an explicit parameter_subspace."
            )
    else:
        if not isinstance(parameter_subspace, ParameterSubspace):
            raise TypeError("parameter_subspace must be a ParameterSubspace or None.")
        parameter_subspace.validate_root(model)
        validate_low_rank_subspace(model, parameter_subspace)
        parameter_paths = parameter_subspace.leaf_paths
    return parameter_paths


def _validate_operator_fit_configuration(
    *,
    epochs: int,
    steps: int | None,
    gradient_accumulation: int,
    checkpoint_every: int,
    tensorboard_every: int,
    evaluation_parameters: EvaluationParametersFn | None,
    evaluation_parameters_id: str | None,
    checkpoint_path: str | Path | None,
    task: OperatorTask | None,
    training_evidence: OperatorTrainingEvidence | None,
    output_pipeline: OperatorOutputPipeline | None,
    loss_terms: Sequence[AbstractOperatorLossTerm] | None,
    gradient_composition: ConflictFreeGradientPolicy | None,
    include_model_losses: bool,
    loss_scale_policy: OperatorLossScalePolicy | None,
    update_alignment: ConflictFreeUpdatePolicy | None,
    target_policy: DelayedTargetPolicy | ExponentialMovingAverageTargetPolicy | None,
) -> tuple[str | None, tuple[AbstractOperatorLossTerm, ...]]:
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
                "Checkpointed fits with evaluation_parameters require a stable evaluation_parameters_id."
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
    specified_terms = () if loss_terms is None else tuple(loss_terms)
    if any(not isinstance(term, AbstractOperatorLossTerm) for term in specified_terms):
        raise TypeError("loss_terms must contain AbstractOperatorLossTerm instances.")
    if int(gradient_accumulation) > 1:
        unsupported = tuple(
            term.name for term in specified_terms if term.accumulation_kind != "case_mean"
        )
        if unsupported:
            raise ValueError(
                f"gradient_accumulation > 1 requires case-additive mean loss terms; unsupported terms: {unsupported}."
            )
    if gradient_composition is not None:
        if not isinstance(gradient_composition, ConflictFreeGradientPolicy):
            raise TypeError(
                "gradient_composition must be a ConflictFreeGradientPolicy or None."
            )
        if int(gradient_accumulation) != 1:
            raise ValueError(
                "Operator gradient composition does not support gradient accumulation."
            )
        if include_model_losses:
            raise ValueError(
                "Operator gradient composition does not yet support attached model losses."
            )
        if loss_scale_policy is not None:
            raise ValueError(
                "Operator gradient composition does not yet support dynamic loss scaling."
            )
    if update_alignment is not None:
        if not isinstance(update_alignment, ConflictFreeUpdatePolicy):
            raise TypeError(
                "update_alignment must be a ConflictFreeUpdatePolicy or None."
            )
        if int(gradient_accumulation) != 1:
            raise ValueError(
                "Operator update alignment does not support gradient accumulation."
            )
        if include_model_losses:
            raise ValueError(
                "Operator update alignment does not yet support attached model losses."
            )
        if loss_scale_policy is not None:
            raise ValueError(
                "Operator update alignment does not yet support dynamic loss scaling."
            )
    if (
        any(isinstance(term, TargetOperatorConsistencyLoss) for term in specified_terms)
        and target_policy is None
    ):
        raise ValueError(
            "TargetOperatorConsistencyLoss requires a delayed or EMA target policy."
        )
    return resolved_evaluation_parameters_id, specified_terms


def _resolve_operator_fit_optimizer(
    model: AbstractOperatorModel,
    specified_terms: tuple[AbstractOperatorLossTerm, ...],
    rollout_route: OperatorRolloutRoute | None,
    rollout_policy: OperatorRolloutPolicy | None,
    task: OperatorTask | None,
    optimizer: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs
    | None,
    learning_rate: float,
    optimizer_id: str | None,
    /,
):
    target_aliases = _rollout_target_aliases(
        model,
        specified_terms,
        rollout_route,
        rollout_policy,
        task,
    )
    if optimizer is None:
        if learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative.")
        optimizer = optax.adam(float(learning_rate))
        resolved_optimizer_id = f"optax.adam:{float(learning_rate):.17g}"
    else:
        if not optimizer_id:
            raise ValueError("Custom optimizers require a stable optimizer_id.")
        resolved_optimizer_id = str(optimizer_id)
    return target_aliases, optimizer, resolved_optimizer_id


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
    rollout_route: OperatorRolloutRoute | None = None,
    rollout_policy: OperatorRolloutPolicy | None = None,
    include_model_losses: bool = True,
    gradient_composition: ConflictFreeGradientPolicy | None = None,
    update_alignment: ConflictFreeUpdatePolicy | None = None,
    optimizer: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs
    | None = None,
    optimizer_id: str | None = None,
    optimizer_state_compression: OptimizerStateCompressionPolicy | None = None,
    evaluation_parameters: EvaluationParametersFn | None = None,
    evaluation_parameters_id: str | None = None,
    target_policy: DelayedTargetPolicy
    | ExponentialMovingAverageTargetPolicy
    | None = None,
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
    privacy: PrivateTrainingPlan | None = None,
    normalize_coordinates: bool = False,
    normalization_weighting: Literal["uniform", "quadrature"] = "uniform",
    dtype_policy: OperatorDTypePolicy | None = None,
    loss_scale_policy: OperatorLossScalePolicy | None = None,
    validation_policy: OperatorValidationPolicy | None = None,
    sharding_policy: OperatorShardingPolicy | None = None,
    execution_group: ExecutionGroup | None = None,
    jit: bool = True,
    session: IterationSession | None = None,
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

    Rollout loss terms share one task-bound ``rollout_route`` and one static-
    maximum ``rollout_policy``; future targets remain aliases rather than model
    outputs.

    Experimental ``update_alignment`` projects the exact emitted optimizer
    proposal against every supported explicit loss term before parameter
    application. It requires one microstep, excludes attached model losses and
    loss scaling, and checkpoints cumulative mismatch evidence.
    """
    master_key, sharding_policy = _resolve_operator_fit_execution(
        model,
        key=key,
        seed=seed,
        privacy=privacy,
        execution_group=execution_group,
        sharding_policy=sharding_policy,
        gradient_composition=gradient_composition,
        update_alignment=update_alignment,
        loss_scale_policy=loss_scale_policy,
        gradient_accumulation=gradient_accumulation,
        include_model_losses=include_model_losses,
        normalization=normalization,
        validation=validation,
        tensorboard_log_dir=tensorboard_log_dir,
        batch_size=batch_size,
        epochs=epochs,
        steps=steps,
    )
    parameter_paths = _resolve_operator_parameter_paths(model, parameter_subspace)
    resolved_evaluation_parameters_id, specified_terms = (
        _validate_operator_fit_configuration(
            epochs=epochs,
            steps=steps,
            gradient_accumulation=gradient_accumulation,
            checkpoint_every=checkpoint_every,
            tensorboard_every=tensorboard_every,
            evaluation_parameters=evaluation_parameters,
            evaluation_parameters_id=evaluation_parameters_id,
            checkpoint_path=checkpoint_path,
            task=task,
            training_evidence=training_evidence,
            output_pipeline=output_pipeline,
            loss_terms=loss_terms,
            gradient_composition=gradient_composition,
            include_model_losses=include_model_losses,
            loss_scale_policy=loss_scale_policy,
            update_alignment=update_alignment,
            target_policy=target_policy,
        )
    )
    target_aliases, optimizer, resolved_optimizer_id = _resolve_operator_fit_optimizer(
        model,
        specified_terms,
        rollout_route,
        rollout_policy,
        task,
        optimizer,
        learning_rate,
        optimizer_id,
    )

    raw_train_loader = _raw_loader(
        train,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        prefetch=prefetch,
        split="train",
        sharding_policy=sharding_policy,
    )
    if privacy is not None and not isinstance(
        raw_train_loader.source, InMemoryOperatorCaseSource
    ):
        raise ValueError(
            "The initial private operator profile requires an in-memory case source."
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
            sharding_policy=sharding_policy,
        )
    )
    private_prepared: _PreparedPrivateGradient | None = None
    if privacy is not None:
        privacy_noise_key = jr.fold_in(master_key, _PRIVACY_NOISE_KEY_DOMAIN)
        sampler_key = jr.fold_in(master_key, _PRIVACY_SAMPLER_KEY_DOMAIN)
        sampler_seed = int(
            jax.device_get(
                jr.randint(
                    sampler_key,
                    (),
                    minval=0,
                    maxval=jnp.iinfo(jnp.int32).max,
                    dtype=jnp.int32,
                )
            )
        )
        private_prepared = _prepare_private_gradient(
            privacy,
            noise_key=privacy_noise_key,
            sampler_seed=sampler_seed,
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
        if private_prepared is not None:
            assert privacy is not None
            probe_epoch = 0
            probe_batch = 0
            saved_step = int(saved_progress["update_step"])
            if not 0 <= saved_step <= privacy.mechanism.iterations:
                raise ValueError("Private checkpoint step is outside its mechanism.")
        else:
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
    elif private_prepared is not None:
        probe_epoch = 0
        probe_batch = 0
    else:
        plan = raw_train_loader.epoch_plan(0)
        if plan.batch_count == 0:
            raise ValueError("Training data must contain at least one batch.")
        probe_epoch = 0
        probe_batch = 0
    first_raw = raw_train_loader.prepare_indices(
        (0,) if private_prepared is not None else plan.batch(probe_batch),
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
    if (
        privacy is not None
        and jnp.dtype(resolved_dtype.reduction_dtype).name != privacy.mechanism.dtype
    ):
        raise ValueError(
            "Private mechanism dtype must equal the operator reduction dtype."
        )
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
            alias_groups=effective_subspace.alias_groups,
        )
        if current_subspace.leaf_shapes != effective_parameter_shapes:
            raise ValueError("Operator fit parameter-subspace shapes changed.")
        if current_subspace.leaf_dtypes != effective_parameter_dtypes:
            raise ValueError("Operator fit parameter-subspace dtypes changed.")
        validate_low_rank_subspace(current_model, current_subspace)
        return current_subspace.initial, current_subspace

    def reconstruct_fit_model(current_parameters, current_fixed):
        if parameter_paths is None:
            return combine_trainable(current_parameters, current_fixed)
        if not isinstance(current_fixed, ParameterSubspace):
            raise TypeError("Low-rank fit fixed state must be ParameterSubspace.")
        return current_fixed.reconstruct(current_parameters)

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
            target_aliases=target_aliases,
        )
        resolved_normalization = fit_operator_normalization(
            normalization_batch,
            normalization_targets,
            normalize_coordinates=normalize_coordinates,
            weighting=normalization_weighting,
            fields=() if task is None else task.fields,
            target_aliases=target_aliases,
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
        target_aliases=target_aliases,
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
        if private_prepared is not None and (
            evidence.checkpoint_id or evidence.corpus_id
        ):
            raise ValueError(
                "Private operator artifacts require training evidence without raw checkpoint or corpus identities."
            )
    else:
        model.operator_contract.validate(physical_first).require_runtime()
    if private_prepared is not None and provenance:
        raise ValueError(
            "Private operator artifacts require empty provenance until a public-safe provenance contract is available."
        )

    fixed_query_fingerprints: dict[str, str] = {}
    if (
        private_prepared is not None
        and task is not None
        and (
            task.problem.query_is_fixed is True
            or model.operator_contract.capabilities.requires_fixed_query
        )
    ):
        raise ValueError(
            "Private operator training does not publish data-derived fixed-query fingerprints."
        )
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
        else specified_terms
    )
    if not terms:
        raise ValueError("fit_operator requires at least one operator loss term.")
    term_names = tuple(term.name for term in terms)
    if len(set(term_names)) != len(term_names):
        raise ValueError("Operator loss term names must be unique.")
    rollout_terms = tuple(
        term
        for term in terms
        if isinstance(
            term,
            (SupervisedOperatorRolloutLoss, ResidualOperatorRolloutLoss),
        )
    )
    if rollout_terms:
        assert task is not None
        assert rollout_route is not None
        assert rollout_policy is not None
        _validate_rollout_route(
            rollout_route,
            task,
            resolved_output_map,
            physical_first,
        )

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
    if gradient_composition is not None and len(terms) < 1:
        raise ValueError("Gradient composition requires at least one operator objective.")

    def predict_for_loss(
        evaluated_model,
        batch,
        physical_batch,
        key,
    ):
        if task is None:
            raw_prediction = _operator_prediction(
                evaluated_model,
                batch,
                key,
                resolved_dtype,
            )
            return raw_prediction, raw_prediction
        return _evaluate_operator_step(
            evaluated_model,
            batch,
            physical_batch,
            task,
            resolved_output_map,
            output_pipeline,
            resolved_normalization,
            resolved_dtype,
            key,
        )

    if output_pipeline is not None:
        predict_for_loss(
            inference_mode(model),
            first.batch,
            physical_first,
            jr.key(seed),
        )

    parameters, fixed = partition_fit_model(model)
    if private_prepared is not None:
        assert privacy is not None
        parameter_arrays = tuple(
            leaf for leaf in jax.tree.leaves(parameters) if eqx.is_array(leaf)
        )
        if any(
            jnp.issubdtype(leaf.dtype, jnp.complexfloating) for leaf in parameter_arrays
        ):
            raise ValueError(
                "The initial private operator profile supports real parameters."
            )
        if any(leaf.dtype.name != privacy.mechanism.dtype for leaf in parameter_arrays):
            raise ValueError(
                "Every private trainable parameter must use the mechanism dtype."
            )
    if optimizer_state_compression is not None:
        if not isinstance(
            optimizer_state_compression,
            OptimizerStateCompressionPolicy,
        ):
            raise TypeError(
                "optimizer_state_compression must be OptimizerStateCompressionPolicy."
            )
        optimizer = prepare_compressed_optimizer(
            optimizer,
            parameters,
            optimizer_state_compression,
            transformation_id=resolved_optimizer_id,
        )
    optimizer_state = optimizer.init(parameters)
    evaluated_parameters = resolve_evaluation_parameters(
        evaluation_parameters,
        optimizer_state,
        parameters,
    )
    initial_target_parameters = (
        evaluated_parameters
        if isinstance(target_policy, ExponentialMovingAverageTargetPolicy)
        and target_policy.source == "evaluation"
        else parameters
    )
    target_state = (
        None
        if target_policy is None
        else TargetParameterState.initialize(initial_target_parameters, target_policy)
    )
    evaluation_model = reconstruct_fit_model(evaluated_parameters, fixed)
    reduction_dtype = jnp.dtype(resolved_dtype.reduction_dtype)
    update_alignment_statistics = (
        None
        if update_alignment is None
        else ConflictFreeUpdateStatistics.zeros(reduction_dtype)
    )
    gradient_accumulator = _GradientAccumulationState.empty(
        parameters,
        accumulation_dtype=reduction_dtype,
    )
    accumulated_metrics = [_ObjectiveAccumulator() for _ in metric_names]
    loss_scale_state = (
        OperatorLossScaleState(jnp.asarray(1.0, dtype=reduction_dtype))
        if loss_scale_policy is None
        else loss_scale_policy.initial_state(reduction_dtype)
    )
    privacy_noise_state = (
        None if private_prepared is None else private_prepared.init_noise(parameters)
    )

    def loss_components(
        current_model,
        target_model,
        batch,
        targets,
        physical_batch,
        physical_targets,
        case_log_weights,
        case_mask,
        sampling_probabilities,
        key,
        step,
        active_rollout_horizon,
        *,
        training: bool,
    ):
        storage_model = current_model if training else inference_mode(current_model)
        evaluated_model = resolved_dtype.compute_model(storage_model)
        scanned = None
        if rollout_terms:
            assert task is not None
            assert rollout_route is not None
            assert rollout_policy is not None
            _, scanned = _operator_rollout_scan(
                _operator_prediction,
                evaluated_model,
                physical_batch,
                batch,
                rollout_route,
                rollout_policy,
                task,
                resolved_output_map,
                output_pipeline,
                resolved_normalization,
                resolved_dtype,
                sharding_policy,
                key,
                active_horizon=active_rollout_horizon,
            )
            (
                execution_predictions,
                physical_predictions,
                execution_batches,
                physical_batches,
            ) = scanned
            execution_prediction = _scan_step_value(execution_predictions, 0)
            physical_prediction = _scan_step_value(physical_predictions, 0)
        else:
            execution_prediction, physical_prediction = predict_for_loss(
                evaluated_model,
                batch,
                physical_batch,
                key,
            )
        if target_model is None:
            target_execution_prediction = None
            target_physical_prediction = None
        else:
            (
                target_execution_prediction,
                target_physical_prediction,
            ) = predict_for_loss(
                resolved_dtype.compute_model(inference_mode(target_model)),
                batch,
                physical_batch,
                jr.fold_in(key, 991),
            )
        context = OperatorLossContext(
            execution_prediction=execution_prediction,
            execution_batch=batch,
            execution_targets=targets,
            physical_prediction=physical_prediction,
            physical_batch=physical_batch,
            physical_targets=physical_targets,
            normalization=resolved_normalization,
            case_log_weights=case_log_weights,
            case_mask=case_mask,
            sampling_probabilities=sampling_probabilities,
            task=task,
            target_execution_prediction=target_execution_prediction,
            target_physical_prediction=target_physical_prediction,
        )

        def recurrent_contribution(term, term_index):
            assert scanned is not None
            assert rollout_policy is not None
            assert rollout_route is not None
            assert active_rollout_horizon is not None
            depth_contributions = []
            empty_targets = OperatorTargetBatch(
                {},
                case_axes=batch.case_axes,
                case_shape=batch.case_shape,
            )
            for depth in range(int(active_rollout_horizon)):
                time_weight = jnp.asarray(
                    term.time_weights[depth],
                    dtype=reduction_dtype,
                )
                physical_step_prediction = _scan_step_value(
                    physical_predictions,
                    depth,
                )
                physical_step_batch = _scan_step_value(physical_batches, depth)
                if isinstance(term, SupervisedOperatorRolloutLoss):
                    predicted = physical_step_prediction.field(rollout_route.task_field)
                    truth = physical_targets.field(term.target_fields[depth])
                    query = physical_step_batch.query(predicted.query_name)
                    mask = query.mask_array(case_shape=physical_step_batch.case_shape)
                    trailing = (1,) * (predicted.values.ndim - mask.ndim)
                    expanded_mask = mask.reshape(mask.shape + trailing)
                    prediction_values = jnp.where(
                        expanded_mask,
                        predicted.values,
                        0.0,
                    )
                    target_values = jnp.where(
                        expanded_mask,
                        truth.values,
                        0.0,
                    )
                    per_case = operator_l2_loss(
                        prediction_values,
                        target_values,
                        query,
                        squared=True,
                        reduction="none",
                    )
                    case_value = _weighted_case_reduction(
                        per_case,
                        context,
                        "mean",
                    )
                    contribution = _case_mean_contribution(
                        jnp.asarray(term.weight, dtype=reduction_dtype) * case_value,
                        context,
                    )
                else:
                    execution_step_prediction = _scan_step_value(
                        execution_predictions,
                        depth,
                    )
                    execution_step_batch = _scan_step_value(execution_batches, depth)
                    residual_context = OperatorLossContext(
                        execution_prediction=execution_step_prediction,
                        execution_batch=execution_step_batch,
                        execution_targets=empty_targets,
                        physical_prediction=physical_step_prediction,
                        physical_batch=physical_step_batch,
                        physical_targets=empty_targets,
                        normalization=resolved_normalization,
                        case_log_weights=case_log_weights,
                        case_mask=case_mask,
                        sampling_probabilities=sampling_probabilities,
                        task=task,
                    )
                    residual = term.residual_term.contribution(
                        evaluated_model,
                        physical_step_prediction,
                        physical_step_batch,
                        empty_targets,
                        key=jr.fold_in(
                            jr.fold_in(key, _RESIDUAL_ROLLOUT_KEY_DOMAIN),
                            term_index * int(rollout_policy.maximum_horizon) + depth,
                        ),
                        step=step,
                        training=training,
                        context=residual_context,
                    )
                    contribution = _ObjectiveContribution(
                        jnp.asarray(term.weight, dtype=reduction_dtype)
                        * residual.numerator,
                        residual.support,
                        residual.log_scale,
                    )
                depth_contributions.append(
                    _ObjectiveContribution(
                        contribution.numerator * time_weight,
                        contribution.support * time_weight,
                        contribution.log_scale,
                    )
                )
            return _combine_objective_contributions(tuple(depth_contributions))

        term_contributions = tuple(
            (
                recurrent_contribution(term, index)
                if isinstance(
                    term,
                    (SupervisedOperatorRolloutLoss, ResidualOperatorRolloutLoss),
                )
                else term.contribution(
                    evaluated_model,
                    physical_prediction,
                    physical_batch,
                    physical_targets,
                    key=jr.fold_in(
                        jr.fold_in(key, _LOSS_TERM_KEY_DOMAIN),
                        index,
                    ),
                    step=step,
                    training=training,
                    context=context,
                )
            )
            for index, term in enumerate(terms)
        )
        attached = (
            tuple(
                _case_mean_contribution(
                    resolved_dtype.reduction(value),
                    context,
                )
                for value in model_loss_values(
                    evaluated_model,
                    key=jr.fold_in(key, _MODEL_OBJECTIVE_KEY_DOMAIN),
                    iter_=step,
                )
            )
            if include_model_losses
            else ()
        )
        components = term_contributions + attached
        total_value = sum(
            (component.value for component in components),
            start=jnp.asarray(0.0, dtype=reduction_dtype),
        )
        total = _case_mean_contribution(
            resolved_dtype.reduction(total_value),
            context,
        )
        return total, components

    _, private_batch_static = eqx.partition(first.batch, eqx.is_array)
    _, private_targets_static = eqx.partition(first.targets, eqx.is_array)
    _, private_physical_batch_static = eqx.partition(physical_first, eqx.is_array)
    _, private_physical_targets_static = eqx.partition(physical_targets, eqx.is_array)
    private_execution_schema = operator_fit_schema(first.batch, target=first.targets)
    private_physical_schema = operator_fit_schema(physical_first, target=physical_targets)

    def private_objective(
        current_parameters,
        target_parameters,
        case_indices,
        batch_arrays,
        target_arrays,
        physical_batch_arrays,
        physical_target_arrays,
        case_log_weights,
        case_mask,
        sampling_probabilities,
        key,
        step,
        active_rollout_horizon,
    ):
        batch = eqx.combine(batch_arrays, private_batch_static)
        targets = eqx.combine(target_arrays, private_targets_static)
        physical_batch = eqx.combine(physical_batch_arrays, private_physical_batch_static)
        physical_targets_ = eqx.combine(
            physical_target_arrays, private_physical_targets_static
        )
        one_batch = slice_operator_batch(batch, case_indices, axis=0)
        one_targets = targets.take(case_indices, axis=0)
        one_physical_batch = slice_operator_batch(physical_batch, case_indices, axis=0)
        one_physical_targets = physical_targets_.take(case_indices, axis=0)
        current_model = reconstruct_fit_model(current_parameters, fixed)
        target_model = (
            reconstruct_fit_model(target_parameters, fixed)
            if target_state is not None
            else None
        )
        total, _ = loss_components(
            current_model,
            target_model,
            one_batch,
            one_targets,
            one_physical_batch,
            one_physical_targets,
            jnp.take(case_log_weights, case_indices, axis=0),
            jnp.take(case_mask, case_indices, axis=0),
            jnp.take(sampling_probabilities, case_indices, axis=0),
            key,
            step,
            active_rollout_horizon,
            training=True,
        )
        return total.numerator

    private_clipped_gradient = (
        None
        if private_prepared is None
        else private_prepared.clipped_grad(
            private_objective,
            argnums=0,
            batch_argnums=2,
            keep_batch_dim=True,
            prng_argnum=10,
        )
    )

    def gradient_fn(
        current_parameters,
        target_parameters,
        batch,
        targets,
        physical_batch,
        physical_targets,
        case_log_weights,
        case_mask,
        sampling_probabilities,
        is_padding_example,
        key,
        step,
        active_rollout_horizon,
        loss_scale_state_,
        privacy_noise_state_,
    ):
        if private_clipped_gradient is not None:
            assert private_prepared is not None
            padding = jnp.asarray(is_padding_example, dtype=jnp.bool_)
            real = ~padding
            case_mask = eqx.error_if(
                jnp.asarray(case_mask, dtype=jnp.bool_),
                jnp.any(real & ~jnp.asarray(case_mask, dtype=jnp.bool_)),
                "The initial private profile requires every sampled case active.",
            )
            case_log_weights = eqx.error_if(
                jnp.asarray(case_log_weights),
                jnp.any(
                    real
                    & (
                        ~jnp.isfinite(case_log_weights)
                        | (jnp.asarray(case_log_weights) != 0.0)
                    )
                ),
                "The initial private profile requires uniform case weights.",
            )
            sampling_probabilities = eqx.error_if(
                jnp.asarray(sampling_probabilities),
                jnp.any(
                    real
                    & (
                        ~jnp.isfinite(sampling_probabilities)
                        | (jnp.asarray(sampling_probabilities) != 1.0)
                    )
                ),
                "The private sampler owns inclusion probabilities.",
            )
            batch_arrays, _ = eqx.partition(batch, eqx.is_array)
            target_arrays, _ = eqx.partition(targets, eqx.is_array)
            physical_batch_arrays, _ = eqx.partition(physical_batch, eqx.is_array)
            physical_target_arrays, _ = eqx.partition(physical_targets, eqx.is_array)
            case_indices = jnp.arange(batch.case_shape[0], dtype=jnp.int32)
            clipped_gradient = private_clipped_gradient(
                current_parameters,
                target_parameters,
                case_indices,
                batch_arrays,
                target_arrays,
                physical_batch_arrays,
                physical_target_arrays,
                case_log_weights,
                case_mask,
                sampling_probabilities,
                key,
                step,
                active_rollout_horizon,
                is_padding_example=is_padding_example,
            )
            gradient, next_privacy_noise_state = private_prepared.privatize(
                clipped_gradient,
                privacy_noise_state_,
            )
            zero = jnp.asarray(0.0, dtype=reduction_dtype)
            one = jnp.asarray(1.0, dtype=reduction_dtype)
            total_arrays = (zero, one, zero)
            component_arrays = tuple((zero, one, zero) for _ in terms)
            finite = tree_all_finite(gradient)
            return (
                total_arrays,
                component_arrays,
                gradient,
                (),
                jnp.zeros((0,), dtype=jnp.bool_),
                finite,
                next_privacy_noise_state,
            )

        def objective(candidate):
            current_model = reconstruct_fit_model(candidate, fixed)
            target_model = (
                reconstruct_fit_model(target_parameters, fixed)
                if target_state is not None
                else None
            )
            total, components = loss_components(
                current_model,
                target_model,
                batch,
                targets,
                physical_batch,
                physical_targets,
                case_log_weights,
                case_mask,
                sampling_probabilities,
                key,
                step,
                active_rollout_horizon,
                training=True,
            )
            scaled = (
                total.numerator
                if loss_scale_policy is None
                else loss_scale_policy.scale_loss(
                    total.numerator,
                    loss_scale_state_,
                )
            )
            total_arrays = (total.numerator, total.support, total.log_scale)
            component_arrays = tuple(
                (component.numerator, component.support, component.log_scale)
                for component in components
            )
            return scaled, (total_arrays, component_arrays)

        if gradient_composition is None and update_alignment is None:
            (_, (total_arrays, component_arrays)), gradient = eqx.filter_value_and_grad(
                objective,
                has_aux=True,
            )(current_parameters)
            if loss_scale_policy is not None:
                gradient = loss_scale_policy.unscale_gradients(
                    gradient,
                    loss_scale_state_,
                )
            component_gradients = ()
            active = jnp.zeros((0,), dtype=jnp.bool_)
            composition_finite = jnp.asarray(True)
        else:

            def component_objective(candidate):
                current_model = reconstruct_fit_model(candidate, fixed)
                target_model = (
                    reconstruct_fit_model(target_parameters, fixed)
                    if target_state is not None
                    else None
                )
                total, components = loss_components(
                    current_model,
                    target_model,
                    batch,
                    targets,
                    physical_batch,
                    physical_targets,
                    case_log_weights,
                    case_mask,
                    sampling_probabilities,
                    key,
                    step,
                    active_rollout_horizon,
                    training=True,
                )
                values = jnp.stack(tuple(component.value for component in components))
                total_arrays_ = (total.numerator, total.support, total.log_scale)
                component_arrays_ = tuple(
                    (component.numerator, component.support, component.log_scale)
                    for component in components
                )
                active_ = jnp.stack(
                    tuple(component.support > 0.0 for component in components)
                )
                return (total.numerator, values), (
                    total_arrays_,
                    component_arrays_,
                    active_,
                )

            (
                (total_numerator, component_values),
                pullback,
                auxiliary,
            ) = eqx.filter_vjp(
                component_objective,
                current_parameters,
                has_aux=True,
            )
            total_arrays, component_arrays, active = auxiliary
            component_gradients = tuple(
                pullback(
                    (
                        jnp.zeros_like(total_numerator),
                        jnp.zeros_like(component_values).at[index].set(1.0),
                    )
                )[0]
                for index in range(component_values.shape[0])
            )
            if gradient_composition is None:
                gradient = pullback(
                    (
                        jnp.ones_like(total_numerator),
                        jnp.zeros_like(component_values),
                    )
                )[0]
                composition_finite = jnp.asarray(True)
            else:
                composition = conflict_free_gradient(
                    component_gradients,
                    active=active,
                    policy=gradient_composition,
                )
                gradient = jax.tree.map(
                    lambda leaf: eqx.error_if(
                        leaf,
                        ~composition.successful,
                        "Operator objectives do not admit a conflict-free direction.",
                    ),
                    composition.direction,
                )
                composition_finite = composition.successful
        finite = tree_all_finite(
            (gradient, total_arrays, component_arrays, component_gradients)
        )
        finite = finite & composition_finite
        if sharding_policy is not None:
            finite = eqx.filter_shard(finite, sharding_policy.replicated)
        return (
            total_arrays,
            component_arrays,
            gradient,
            component_gradients,
            active,
            finite,
            privacy_noise_state_,
        )

    def update_fn(
        current_parameters,
        current_state,
        gradient,
        component_gradients,
        active,
    ):
        updates, next_state = optimizer.update(
            gradient,
            current_state,
            current_parameters,
        )
        alignment_result = None
        gradient_conflict = None
        constructed_conflict = None
        if update_alignment is not None:
            proposal = tree_negative(updates)
            alignment_result = project_conflict_free_direction(
                proposal,
                component_gradients,
                active=active,
                policy=update_alignment,
            )
            aligned_updates = tree_negative(alignment_result.direction)
            updates = tree_where(
                alignment_result.projected,
                aligned_updates,
                updates,
            )
            updates = jax.tree.map(
                lambda leaf: eqx.error_if(
                    leaf,
                    ~alignment_result.successful,
                    "Operator optimizer proposal could not be aligned.",
                ),
                updates,
            )
            gradient_conflict, constructed_conflict = _alignment_conflicts(
                component_gradients,
                gradient,
                alignment_result,
                update_alignment,
            )
        next_parameters = eqx.apply_updates(current_parameters, updates)
        finite = tree_all_finite((next_parameters, next_state))
        if alignment_result is not None:
            finite = finite & alignment_result.successful
        if sharding_policy is not None:
            finite = eqx.filter_shard(finite, sharding_policy.replicated)
        return (
            next_parameters,
            next_state,
            finite,
            alignment_result,
            gradient_conflict,
            constructed_conflict,
        )

    run_gradient_fn = eqx.filter_jit(gradient_fn) if jit else gradient_fn
    run_update_fn = eqx.filter_jit(update_fn) if jit else update_fn

    def prepared_epoch(
        loader: OperatorBatchLoader,
        epoch: int,
        *,
        start_batch: int = 0,
        retained_first: OperatorTrainingBatch | None = None,
    ):
        if private_prepared is not None:
            if int(epoch) != 0 or retained_first is not None:
                raise ValueError(
                    "Private sampler execution uses one fixed iteration schedule."
                )
            selected_batches = private_prepared.batch_iterator(
                loader.source.size,
                start_step=int(start_batch),
            )
            for batch_index, indices in enumerate(
                selected_batches,
                start=int(start_batch),
            ):
                raw = prepare_private_operator_batch(
                    loader,
                    private_prepared,
                    indices,
                    step=batch_index,
                )
                placed = _place_batch(
                    raw,
                    task=task,
                    normalization=resolved_normalization,
                    dtype_policy=resolved_dtype,
                    sharding_policy=sharding_policy,
                    target_aliases=target_aliases,
                )
                placed_physical_batch = (
                    placed.batch
                    if placed.physical_batch is None
                    else placed.physical_batch
                )
                placed_physical_targets = (
                    placed.targets
                    if placed.physical_targets is None
                    else placed.physical_targets
                )
                if (
                    operator_fit_schema(
                        placed.batch,
                        target=placed.targets,
                    )
                    != private_execution_schema
                    or operator_fit_schema(
                        placed_physical_batch,
                        target=placed_physical_targets,
                    )
                    != private_physical_schema
                ):
                    raise ValueError(
                        "Private operator batches changed static structure or semantics."
                    )
                yield placed
            return
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
                target_aliases=target_aliases,
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
                    target_aliases=target_aliases,
                )

    def resolved_active_horizon(step: int) -> int | None:
        if not rollout_terms:
            return None
        assert rollout_policy is not None
        return int(
            jax.device_get(
                rollout_policy.active_horizon(jnp.asarray(step, dtype=jnp.float64))
            )
        )

    def evaluate(current_model, loader: OperatorBatchLoader, step: int):
        metric_accumulators = [_ObjectiveAccumulator() for _ in metric_names]
        batch_count = 0
        active_rollout_horizon = resolved_active_horizon(step)
        target_model = (
            reconstruct_fit_model(target_state.target, fixed)
            if target_state is not None
            else None
        )
        for batch_index, training_batch in enumerate(prepared_epoch(loader, 0)):
            total, components = loss_components(
                current_model,
                target_model,
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
                training_batch.case_log_weights,
                training_batch.case_mask,
                training_batch.sampling_probabilities,
                jr.fold_in(jr.fold_in(master_key, int(step)), 1000 + batch_index),
                jnp.asarray(step, dtype=jnp.float64),
                active_rollout_horizon,
                training=False,
            )
            contributions = (total,) + components
            metric_accumulators = [
                accumulator.add(contribution)
                for accumulator, contribution in zip(
                    metric_accumulators,
                    contributions,
                    strict=True,
                )
            ]
            batch_count += 1
        if batch_count == 0:
            raise ValueError("Evaluation data must contain at least one batch.")
        return {
            name: float(jax.device_get(accumulator.value))
            for name, accumulator in zip(
                metric_names,
                metric_accumulators,
                strict=True,
            )
        }

    maximum_steps = (
        (
            private_prepared.training_plan.mechanism.iterations
            if steps is None
            else int(steps)
        )
        if private_prepared is not None
        else (
            int(steps)
            if steps is not None
            else int(epochs)
            * ceil(raw_train_loader.batches_per_epoch / int(gradient_accumulation))
        )
    )
    batches_per_training_epoch = (
        private_prepared.training_plan.mechanism.iterations
        if private_prepared is not None
        else raw_train_loader.batches_per_epoch
    )
    progress = TrainingProgress()
    iteration_session = (
        session if sharding_policy is None or sharding_policy.is_primary_process else None
    )
    control = TrainingController(
        total_steps=maximum_steps,
        key=master_key,
        algorithm_id="operator-training",
        progress=progress,
        session=iteration_session,
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
        "objective_aggregation": "numerator_support",
        "rollout_route": (None if rollout_route is None else asdict(rollout_route)),
        "rollout_policy": (None if rollout_policy is None else asdict(rollout_policy)),
        "include_model_losses": bool(include_model_losses),
        "privacy": (
            None
            if private_prepared is None
            else {
                "training_plan": private_prepared.training_plan.to_record(),
                "prepared_id": private_prepared.prepared_id,
                "noise_multiplier": private_prepared.noise_multiplier,
                "per_step_trace_id": private_prepared.per_step_trace.trace_id,
                "randomness": private_prepared.randomness.value,
                "qualification_profile_id": (private_prepared.qualification_profile_id()),
            }
        ),
        "key_domains": {
            "rollout_model": _ROLLOUT_MODEL_KEY_DOMAIN,
            "loss_term": _LOSS_TERM_KEY_DOMAIN,
            "residual_rollout": _RESIDUAL_ROLLOUT_KEY_DOMAIN,
            "model_objective": _MODEL_OBJECTIVE_KEY_DOMAIN,
            "privacy_noise": _PRIVACY_NOISE_KEY_DOMAIN,
            "privacy_sampler": _PRIVACY_SAMPLER_KEY_DOMAIN,
        },
        "optimizer_id": resolved_optimizer_id,
        "target_policy": (None if target_policy is None else asdict(target_policy)),
        "optimizer_state_compression": (
            None
            if optimizer_state_compression is None
            else {
                "format": repr(optimizer_state_compression.format),
                "block_axes": optimizer_state_compression.block_axes,
                "exact_roles": optimizer_state_compression.exact_roles,
                "overflow": optimizer_state_compression.overflow,
            }
        ),
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
        "gradient_composition": (
            None if gradient_composition is None else gradient_composition.policy_id
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
                "device_count": sharding_policy.mesh.devices.size,
            }
        ),
        "iteration_session": (
            None if iteration_session is None else iteration_session.session_id
        ),
        "iteration_control": (
            None if iteration_session is None else iteration_session.control_id
        ),
        "configuration": {} if configuration is None else dict(configuration),
    }
    if update_alignment is not None:
        fit_contract_data["update_alignment"] = update_alignment.policy_id
    if resolved_evaluation_parameters_id is not None:
        fit_contract_data["evaluation_parameters_id"] = resolved_evaluation_parameters_id
    fit_contract = _canonical_json(fit_contract_data)
    schema = {
        "fit": operator_fit_schema(first.batch, target=first.targets),
    }

    initial_metrics: dict[str, float]
    if resume_manifest is not None:
        assert checkpoint is not None
        if resume_manifest["metadata"].get("fit_contract") != fit_contract:
            raise ValueError("Operator fit checkpoint contract mismatch.")
        if private_prepared is not None:
            state_template = (
                optimizer_state,
                loss_scale_state,
                target_state,
                private_prepared.checkpoint_noise_state(privacy_noise_state),
                jnp.asarray(private_prepared.sampler_seed, dtype=jnp.uint32),
            )
        elif update_alignment is None:
            state_template = (optimizer_state, loss_scale_state, target_state)
        else:
            state_template = (
                optimizer_state,
                loss_scale_state,
                target_state,
                update_alignment_statistics,
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
        if private_prepared is not None:
            (
                optimizer_state,
                loss_scale_state,
                target_state,
                privacy_noise_checkpoint,
                restored_sampler_seed,
            ) = restored.optimizer_state
            privacy_noise_state = private_prepared.restore_noise_state(
                privacy_noise_checkpoint
            )
            private_prepared = replace(
                private_prepared,
                sampler_seed=int(jax.device_get(restored_sampler_seed)),
            )
        elif update_alignment is None:
            optimizer_state, loss_scale_state, target_state = restored.optimizer_state
        else:
            (
                optimizer_state,
                loss_scale_state,
                target_state,
                update_alignment_statistics,
            ) = restored.optimizer_state
            if not isinstance(
                update_alignment_statistics,
                ConflictFreeUpdateStatistics,
            ):
                raise ValueError(
                    "Operator checkpoint update-alignment statistics are invalid."
                )
        metadata = restored.metadata
        if metadata.get("update_boundary") is not True:
            raise ValueError(
                "Operator fit checkpoints must publish at optimizer-update boundaries."
            )
        expected_privacy_classification = (
            None if private_prepared is None else "restricted"
        )
        if metadata.get("privacy_classification") != expected_privacy_classification:
            raise ValueError("Operator checkpoint privacy classification changed.")
        progress = TrainingProgress(**metadata["progress"])
        if progress.update_step != restored.step:
            raise ValueError("Checkpoint progress disagrees with its update step.")
        if progress.update_step > maximum_steps:
            raise ValueError("Checkpoint step exceeds the requested training ceiling.")
        control = TrainingController(
            total_steps=maximum_steps,
            key=restored.key,
            algorithm_id="operator-training",
            progress=progress,
            session=iteration_session,
        )
        master_key = restored.key
        control.best_payload = best_model
        train_steps = [int(value) for value in metadata["train_steps"]]
        train_history = [dict(values) for values in metadata["train_metrics"]]
        validation_steps = [int(value) for value in metadata["validation_steps"]]
        validation_history = [dict(values) for values in metadata["validation_metrics"]]
        initial_metrics = dict(metadata["initial_metrics"])
        prior_training_seconds = float(metadata["training_seconds"])
        resumed_from_step = progress.update_step
        parameters, fixed = partition_fit_model(model)
        gradient_accumulator = _GradientAccumulationState.empty(
            parameters,
            accumulation_dtype=reduction_dtype,
        )
        accumulated_metrics = [_ObjectiveAccumulator() for _ in metric_names]
        evaluated_parameters = resolve_evaluation_parameters(
            evaluation_parameters,
            optimizer_state,
            parameters,
        )
        evaluation_model = reconstruct_fit_model(evaluated_parameters, fixed)
    else:
        initial_metrics = (
            {}
            if private_prepared is not None
            else evaluate(evaluation_model, raw_train_loader, 0)
        )
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

    def save_progress(training_seconds: float, *, emit_event: bool = True) -> None:
        if checkpoint is None or not gradient_accumulator.is_empty:
            return
        primary = sharding_policy is None or sharding_policy.is_primary_process
        if not primary:
            sharding_policy.synchronize(
                f"fit_operator_checkpoint_{control.progress.update_step}"
            )
            return
        if emit_event:
            control.emit(
                TrainingIterationKind.CHECKPOINT,
                metrics={"step": control.progress.update_step},
            )
        if private_prepared is not None:
            checkpoint_state = (
                optimizer_state,
                loss_scale_state,
                target_state,
                private_prepared.checkpoint_noise_state(privacy_noise_state),
                jnp.asarray(private_prepared.sampler_seed, dtype=jnp.uint32),
            )
        elif update_alignment is None:
            checkpoint_state = (optimizer_state, loss_scale_state, target_state)
        else:
            checkpoint_state = (
                optimizer_state,
                loss_scale_state,
                target_state,
                update_alignment_statistics,
            )
        save_operator_training_checkpoint(
            checkpoint,
            (model, best_model),
            checkpoint_state,
            step=control.progress.update_step,
            key=master_key,
            normalization=resolved_normalization,
            dtype_policy=resolved_dtype,
            schema=schema,
            metadata={
                "fit_contract": fit_contract,
                "data_contract": current_data_contract,
                "progress": asdict(control.progress),
                "update_boundary": True,
                "privacy_classification": (
                    None if private_prepared is None else "restricted"
                ),
                "initial_metrics": initial_metrics,
                "train_steps": train_steps,
                "train_metrics": train_history,
                "validation_steps": validation_steps,
                "validation_metrics": validation_history,
                "training_seconds": float(training_seconds),
            },
        )
        if sharding_policy is not None:
            sharding_policy.synchronize(
                f"fit_operator_checkpoint_{control.progress.update_step}"
            )

    def consider_validation(
        metrics: Mapping[str, float],
        current_evaluation_model: AbstractOperatorModel,
        /,
    ) -> None:
        nonlocal best_model
        assert validation_config is not None
        score = float(metrics[validation_config.monitor])
        control.progress, strict_better = _update_validation_selection(
            control.progress,
            score,
            step=control.progress.update_step,
            mode=validation_config.mode,
            minimum_delta=validation_config.minimum_delta,
            relative_minimum_delta=validation_config.relative_minimum_delta,
            patience=validation_config.patience,
        )
        if strict_better:
            best_model = current_evaluation_model
            control.best_payload = current_evaluation_model
        if control.progress.stopped_early:
            control.stop_requested = True

    logger_context = (
        nullcontext(None)
        if tensorboard_log_dir is None
        or (sharding_policy is not None and not sharding_policy.is_primary_process)
        else TensorBoardLogger(tensorboard_log_dir)
    )
    started = time.perf_counter()
    stopped_by_signal = False
    control.emit(TrainingIterationKind.RUN_START, metrics=initial_metrics)
    with logger_context as tensorboard, TrainingSignalGuard() as signal_guard:
        if not control.progress.stopped_early and _has_trainable_arrays(parameters):
            for epoch in range(control.progress.epoch, int(epochs)):
                if control.stop_requested or signal_guard.stop_requested:
                    break
                control.emit(TrainingIterationKind.EPOCH_START, metrics={"epoch": epoch})
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
                    (
                        total,
                        components,
                        gradient,
                        component_gradients,
                        component_active,
                        finite_array,
                        next_privacy_noise_state,
                    ) = run_gradient_fn(
                        parameters,
                        (target_state.target if target_state is not None else parameters),
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
                        training_batch.case_log_weights,
                        training_batch.case_mask,
                        training_batch.sampling_probabilities,
                        training_batch.is_padding_example,
                        key,
                        jnp.asarray(control.progress.update_step + 1, dtype=jnp.float64),
                        resolved_active_horizon(control.progress.update_step + 1),
                        loss_scale_state,
                        privacy_noise_state,
                    )
                    privacy_noise_state = next_privacy_noise_state
                    total_contribution = _ObjectiveContribution(*total)
                    component_contributions = tuple(
                        _ObjectiveContribution(*component) for component in components
                    )
                    contributions = (total_contribution,) + component_contributions
                    finite = bool(jax.device_get(finite_array))
                    control.progress = replace(
                        control.progress,
                        microstep=control.progress.microstep + 1,
                        next_batch_index=training_batch.batch_index + 1,
                    )
                    if not finite:
                        gradient_accumulator = _GradientAccumulationState.empty(
                            parameters,
                            accumulation_dtype=reduction_dtype,
                        )
                        accumulated_metrics = [
                            _ObjectiveAccumulator() for _ in metric_names
                        ]
                        if loss_scale_policy is None or not loss_scale_policy.dynamic:
                            raise FloatingPointError(
                                "Non-finite operator loss or gradient encountered."
                            )
                        loss_scale_state = loss_scale_policy.on_nonfinite_microstep(
                            loss_scale_state
                        )
                        control.emit(
                            TrainingIterationKind.FAILURE,
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
                    gradient_accumulator = gradient_accumulator.add(
                        gradient,
                        total_contribution,
                    )
                    accumulated_metrics = [
                        accumulator.add(contribution)
                        for accumulator, contribution in zip(
                            accumulated_metrics,
                            contributions,
                            strict=True,
                        )
                    ]
                    if (
                        gradient_accumulator.microsteps < int(gradient_accumulation)
                        and training_batch.batch_index + 1 < batches_per_training_epoch
                    ):
                        continue
                    if not bool(
                        jax.device_get(gradient_accumulator.has_positive_support)
                    ):
                        gradient_accumulator = _GradientAccumulationState.empty(
                            parameters,
                            accumulation_dtype=reduction_dtype,
                        )
                        accumulated_metrics = [
                            _ObjectiveAccumulator() for _ in metric_names
                        ]
                        continue

                    averaged_gradient = gradient_accumulator.normalized_gradient(
                        parameters
                    )
                    (
                        candidate_parameters,
                        candidate_optimizer_state,
                        candidate_finite_array,
                        update_alignment_result,
                        gradient_conflict,
                        constructed_conflict,
                    ) = run_update_fn(
                        parameters,
                        optimizer_state,
                        averaged_gradient,
                        component_gradients,
                        component_active,
                    )
                    if not bool(jax.device_get(candidate_finite_array)):
                        raise FloatingPointError(
                            "Operator optimizer produced non-finite state from finite gradients."
                        )
                    parameters = candidate_parameters
                    optimizer_state = candidate_optimizer_state
                    model = reconstruct_fit_model(parameters, fixed)
                    update_step = control.progress.update_step + 1
                    control.complete_update(update_step)
                    if target_state is not None:
                        target_state = target_state.update(
                            parameters,
                            accepted=True,
                            evaluation_parameters=resolve_evaluation_parameters(
                                evaluation_parameters,
                                optimizer_state,
                                parameters,
                            ),
                        )
                    metrics = (
                        {}
                        if private_prepared is not None
                        else {
                            name: float(jax.device_get(accumulator.value))
                            for name, accumulator in zip(
                                metric_names,
                                accumulated_metrics,
                                strict=True,
                            )
                        }
                    )
                    if update_alignment_result is not None:
                        if update_alignment_statistics is None:
                            raise RuntimeError(
                                "Operator update-alignment statistics are missing."
                            )
                        update_alignment_statistics = update_alignment_statistics.update(
                            update_alignment_result,
                            gradient_conflict=gradient_conflict,
                            constructed_conflict=constructed_conflict,
                        )
                        metrics.update(
                            {
                                "update_alignment/raw_conflict": float(
                                    jax.device_get(
                                        jnp.any(update_alignment_result.raw_conflicts)
                                    )
                                ),
                                "update_alignment/applied_conflict": float(
                                    jax.device_get(
                                        jnp.any(update_alignment_result.aligned_conflicts)
                                    )
                                ),
                                "update_alignment/projected": float(
                                    jax.device_get(update_alignment_result.projected)
                                ),
                                "update_alignment/relative_correction": float(
                                    jax.device_get(
                                        update_alignment_result.relative_correction
                                    )
                                ),
                                "update_alignment/metric_correction_norm": float(
                                    jax.device_get(
                                        update_alignment_result.metric_correction_norm
                                    )
                                ),
                                "update_alignment/active_constraints": float(
                                    jax.device_get(
                                        update_alignment_result.active_constraint_count
                                    )
                                ),
                                "update_alignment/pareto_stationary": float(
                                    jax.device_get(
                                        update_alignment_result.pareto_stationary
                                    )
                                ),
                                "update_alignment/kkt_residual": float(
                                    jax.device_get(
                                        update_alignment_result.kkt_residual_norm
                                    )
                                ),
                                "update_alignment/status": float(
                                    jax.device_get(update_alignment_result.status)
                                ),
                                "update_alignment/gradient_conflict_rate": float(
                                    jax.device_get(
                                        update_alignment_statistics.gradient_conflict_rate
                                    )
                                ),
                                "update_alignment/constructed_conflict_rate": float(
                                    jax.device_get(
                                        update_alignment_statistics.constructed_conflict_rate
                                    )
                                ),
                                "update_alignment/proposal_conflict_rate": float(
                                    jax.device_get(
                                        update_alignment_statistics.proposal_conflict_rate
                                    )
                                ),
                                "update_alignment/applied_conflict_rate": float(
                                    jax.device_get(
                                        update_alignment_statistics.applied_conflict_rate
                                    )
                                ),
                            }
                        )
                    train_steps.append(update_step)
                    train_history.append(metrics)
                    gradient_accumulator = _GradientAccumulationState.empty(
                        parameters,
                        accumulation_dtype=reduction_dtype,
                    )
                    accumulated_metrics = [_ObjectiveAccumulator() for _ in metric_names]
                    if loss_scale_policy is not None:
                        loss_scale_state = loss_scale_policy.on_finite_update(
                            loss_scale_state
                        )
                    control.emit(TrainingIterationKind.UPDATE, metrics=metrics)
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
                        control.emit(
                            TrainingIterationKind.VALIDATION,
                            metrics=validation_metrics,
                        )
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
                if control.progress.next_batch_index >= batches_per_training_epoch:
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
    selected_model = (
        best_model
        if validation_config is not None and validation_config.select_best
        else evaluation_model
    )
    final_metrics = (
        {}
        if private_prepared is not None
        else evaluate(
            selected_model,
            raw_train_loader,
            control.progress.update_step,
        )
    )
    control.emit(TrainingIterationKind.RUN_TERMINAL, metrics=final_metrics)
    save_progress(training_seconds, emit_event=False)
    privacy_certificate = (
        None
        if private_prepared is None or control.progress.update_step == 0
        else private_prepared.certificate(control.progress.update_step)
    )

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
            privacy_certificate=privacy_certificate,
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
        update_alignment_statistics=update_alignment_statistics,
        privacy_certificate=privacy_certificate,
        stopped_by_signal=stopped_by_signal,
        stopped_by_host_control=control.stop_requested
        and not control.progress.stopped_early,
    )


__all__ = [
    "OperatorFitHistory",
    "OperatorFitResult",
    "OperatorValidationPolicy",
    "fit_operator",
]
