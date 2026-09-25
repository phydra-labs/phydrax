#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import functools
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from math import ceil, inf, isfinite, isnan, log, nan
from pathlib import Path
from typing import Any, ClassVar, final, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ...._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from ...._execution_runtime import ExecutionGroup
from ...._fingerprint import canonical_fingerprint
from ...._frozendict import frozendict
from ...._iteration import IterationSession
from ...._model._ports import PortBindingEvidence, PortMapping, ValuePort
from ...._sampling._addressing import derive_key, SampleAddress
from ...._strict import StrictModule
from ...._trainable import (
    combine_parameters,
    ExplicitFreeze,
    partition_parameters,
    require_parameter_roles,
)
from ...._training import (
    _update_validation_selection,
    DelayedTargetPolicy,
    EvaluationParametersFn,
    ExponentialMovingAverageTargetPolicy,
    resolve_evaluation_parameters,
    TensorBoardLogger,
    TrainingController,
    TrainingIterationKind,
    TrainingProgress,
    TrainingSignalGuard,
)
from ...._training_checkpoint import (
    load_training_checkpoint,
    read_training_checkpoint_metadata,
    save_training_checkpoint,
)
from ...._training_kernel import (
    AbstractKernelUpdateRule,
    build_training_checkpoint,
    enforce_rejection_budget,
    KernelObjective,
    KernelUpdateContext,
    OptaxUpdateRule,
    prepare_training_kernel,
    SubspaceTrainingTree,
    training_accepted_site_key,
    TrainingAttemptOutcome,
    TrainingKernelSpec,
    TrainingKernelState,
    TrainingKeys,
    TrainingRejectionBudgetError,
)
from ...._training_objective import (
    _combine_objective_contributions,
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
from ._dataset import OperatorDataset
from ._dtype import OperatorDTypePolicy, OperatorPrecisionEvidence
from ._execution import (
    _evaluate_operator_step,
    _operator_output_routes,
    _operator_prediction,
    _port_binding_record,
    bind_operator_outputs,
    nondimensionalize_batch,
    nondimensionalize_targets,
    OperatorOutputRoutes,
)
from ._fingerprint import operator_fit_schema
from ._loader import (
    _pad_case_payload,
    OperatorBatchLoader,
    OperatorTrainingBatch,
)
from ._loss_scale import OperatorLossScalePolicy, OperatorLossScaleState
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
    _validate_rollout_route,
    OperatorRolloutPolicy,
    OperatorRolloutRoute,
)
from ._target_consistency import TargetOperatorConsistencyLoss
from ._trained_operator import (
    operator_contract_fingerprint,
    TrainedOperator,
)


_FIT_OBJECTIVE_ID = "operator-fit"
_FIT_CHECKPOINT_FORMAT = "phydrax-operator-fit-checkpoint"
_LOSS_KEY_SITE = "loss"
_PRIVACY_NOISE_KEY_SITE = "privacy-noise"
_MODEL_ADDRESS = SampleAddress(_FIT_OBJECTIVE_ID, "model")
_TARGET_MODEL_ADDRESS = SampleAddress(_FIT_OBJECTIVE_ID, "target-model")
_MODEL_LOSS_ADDRESS = SampleAddress(_FIT_OBJECTIVE_ID, "model-loss")
_PREFLIGHT_ADDRESS = SampleAddress(
    "training", _FIT_OBJECTIVE_ID, target=("output-pipeline",), role="preflight"
)
_PRIVACY_NOISE_ADDRESS = SampleAddress(
    "training", _FIT_OBJECTIVE_ID, target=("privacy-noise",), role="mechanism"
)
_PRIVACY_SAMPLER_ADDRESS = SampleAddress(
    "training", _FIT_OBJECTIVE_ID, target=("privacy-sampler",), role="seed"
)
# Semantic randomness of one fit, recorded in the fit contract so a resume under
# a different addressing scheme fails closed.
_FIT_KEY_SITES = {
    "objective": _FIT_OBJECTIVE_ID,
    "attempt_sites": [_LOSS_KEY_SITE, _PRIVACY_NOISE_KEY_SITE],
    "evaluation_site": "metrics/<split>",
    "loss_addresses": [
        "model",
        "target-model",
        "model-loss",
        "loss-term/<name>",
        "residual-rollout/<name>/<depth>",
    ],
    "rollout_step": "operator-rollout/model/<step>",
    "privacy": ["privacy-noise/mechanism", "privacy-sampler/seed"],
}


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
    port_binding: PortBindingEvidence | None
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


_NONFINITE_METRICS = {"nan": nan, "inf": inf, "-inf": -inf}


def _encode_metrics(metrics: Mapping[str, float], /) -> dict[str, Any]:
    """Encode one metric record for JSON checkpoints, tagging nonfinite values.

    Overflowed (discarded) updates record nonfinite losses; resume must restore
    the identical history, so nonfinite values are kept rather than dropped.
    """
    return {
        name: value
        if isfinite(value)
        else {"nonfinite": "nan" if isnan(value) else ("inf" if value > 0 else "-inf")}
        for name, value in metrics.items()
    }


def _decode_metrics(metrics: Mapping[str, Any], /) -> dict[str, float]:
    return {
        name: _NONFINITE_METRICS[value["nonfinite"]]
        if isinstance(value, dict)
        else float(value)
        for name, value in metrics.items()
    }


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


def _resolve_output_binding(
    model: AbstractOperatorModel,
    task: OperatorTask | None,
    output_ports: Mapping[str, ValuePort] | None,
    port_mapping: PortMapping | None,
    /,
) -> tuple[frozendict[str, ValuePort], PortBindingEvidence] | None:
    if task is None:
        if output_ports is not None or port_mapping is not None:
            raise ValueError(
                "output_ports and port_mapping bind model outputs to task target "
                "fields; they require a task-bound fit."
            )
        return None
    return bind_operator_outputs(model, task, output_ports, port_mapping)


def _default_losses(
    model: AbstractOperatorModel,
    routes: OperatorOutputRoutes | None,
    /,
) -> tuple[SupervisedOperatorLoss, ...]:
    """Supervise every task target, or every raw model output of a taskless fit."""
    if routes is None:
        names = tuple(model.operator_output_specs)
        return tuple(
            SupervisedOperatorLoss(
                name=f"supervised_l2/{name}" if len(names) > 1 else "supervised_l2",
                prediction_field=name,
            )
            for name in names
        )
    return tuple(
        SupervisedOperatorLoss(
            name=f"supervised_l2/{target.name}" if len(routes) > 1 else "supervised_l2",
            prediction_field=target.name,
            target_field=target.name,
        )
        for _, target in routes
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


def _typed_root_key(key: Any, /) -> Any:
    """Return `key` as one typed JAX PRNG key (raw threefry words are wrapped)."""
    array = jnp.asarray(key)
    if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
        return array
    return jr.wrap_key_data(array.astype(jnp.uint32))


def _private_sampler_seed(root_key: Any, /) -> int:
    sampler_key = derive_key(root_key, _PRIVACY_SAMPLER_ADDRESS)
    return int(
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


def _loss_scale_rejection_budget(policy: OperatorLossScalePolicy | None, /) -> int:
    """Consecutive nonfinite windows a dynamic loss scale may absorb.

    Each overflow backs the scale off once. After
    `ceil(log(maximum / minimum) / log(1 / backoff))` consecutive backoffs the
    scale sits at its minimum from any start; one more window is tried there,
    and a further nonfinite window is not an overflow the scale can cure.
    Without dynamic scaling every nonfinite window is a domain error.
    """
    if policy is None or not policy.dynamic:
        return 0
    backoffs = ceil(
        log(policy.maximum_scale / policy.minimum_scale) / -log(policy.backoff_factor)
    )
    return backoffs + 1


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _loss_scaled(
    policy: OperatorLossScalePolicy, numerator: jax.Array, scale: jax.Array
) -> jax.Array:
    """Identity whose backward pass seeds the loss-scaled cotangent."""
    del policy, scale
    return numerator


def _loss_scaled_forward(policy, numerator, scale):
    del policy
    return numerator, scale


def _loss_scaled_backward(policy, scale, cotangent):
    return (
        policy.scale_loss(cotangent, OperatorLossScaleState(scale)),
        jnp.zeros_like(scale),
    )


_loss_scaled.defvjp(_loss_scaled_forward, _loss_scaled_backward)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _loss_unscaled(
    policy: OperatorLossScalePolicy, parameters: Any, scale: jax.Array
) -> Any:
    """Identity whose backward pass unscales the parameter cotangents."""
    del policy, scale
    return parameters


def _loss_unscaled_forward(policy, parameters, scale):
    del policy
    return parameters, scale


def _loss_unscaled_backward(policy, scale, cotangents):
    return (
        policy.unscale_gradients(cotangents, OperatorLossScaleState(scale)),
        jnp.zeros_like(scale),
    )


_loss_unscaled.defvjp(_loss_unscaled_forward, _loss_unscaled_backward)


@jax.custom_vjp
def _prescribed_gradient(value: jax.Array, parameters: Any, gradient: Any) -> jax.Array:
    """`value`, whose derivative with respect to `parameters` is `gradient`.

    Used by objectives whose training direction is not the plain derivative of
    their numerator (conflict-free composition, clipped and noised private
    gradients); `value` and `gradient` are evaluated at stopped parameters.
    """
    del parameters, gradient
    return value


def _prescribed_gradient_forward(value, parameters, gradient):
    del parameters
    return value, gradient


def _prescribed_gradient_backward(gradient, cotangent):
    return (
        jnp.zeros_like(cotangent),
        jax.tree.map(lambda leaf: cotangent.astype(leaf.dtype) * leaf, gradient),
        jax.tree.map(jnp.zeros_like, gradient),
    )


_prescribed_gradient.defvjp(_prescribed_gradient_forward, _prescribed_gradient_backward)


class _OperatorFitPayload(NamedTuple):
    """Arrays of one fit microbatch plus the attempt's loss scale and targets."""

    batch: OperatorBatch
    targets: OperatorTargetBatch
    physical_batch: OperatorBatch
    physical_targets: OperatorTargetBatch
    case_log_weights: Any
    case_mask: Any
    sampling_probabilities: Any
    is_padding_example: Any
    step: Any
    target_parameters: Any
    loss_scale: Any
    active_horizon: int | None


@final
class _OperatorFitObjective(StrictModule, ExplicitFreeze):
    """Kernel objective of one operator fit.

    The evaluation closes over the fit's configuration (task, output routes and
    pipeline, normalization, loss terms, privacy mechanism), which is frozen on
    purpose; trained arrays reach it only through the kernel's parameter,
    model-state, and fixed lanes.
    """

    evaluate: Callable[..., Any] = eqx.field(static=True)

    def __call__(
        self,
        parameters: Any,
        model_state: Any,
        fixed: Any,
        payload: _OperatorFitPayload,
        keys: TrainingKeys,
    ) -> tuple[_ObjectiveContribution, Any, Any]:
        return self.evaluate(parameters, model_state, fixed, payload, keys)


_ALIGNMENT_UPDATE_METRICS = (
    "update_alignment/raw_conflict",
    "update_alignment/applied_conflict",
    "update_alignment/projected",
    "update_alignment/relative_correction",
    "update_alignment/metric_correction_norm",
    "update_alignment/active_constraints",
    "update_alignment/pareto_stationary",
    "update_alignment/kkt_residual",
    "update_alignment/status",
)


@final
class _AlignedUpdateState(StrictModule):
    """Optimizer state, cumulative alignment evidence, and the last update's."""

    optimizer_state: Any
    statistics: ConflictFreeUpdateStatistics
    last_update: jax.Array


@final
class _AlignedOptaxUpdateRule(AbstractKernelUpdateRule):
    """Optax proposal projected against every supported explicit loss-term gradient.

    The objective reports its component gradients and active mask as
    diagnostics. A proposal that cannot be aligned is a runtime error; nothing
    commits on a finite rejection.
    """

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ()
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    policy: ConflictFreeUpdatePolicy
    evaluation_view: EvaluationParametersFn | None = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)
    statistics_dtype: str = eqx.field(static=True)

    def init(self, parameters: Any, /) -> _AlignedUpdateState:
        dtype = jnp.dtype(self.statistics_dtype)
        return _AlignedUpdateState(
            self.optimizer.init(parameters),
            ConflictFreeUpdateStatistics.zeros(dtype),
            jnp.zeros((len(_ALIGNMENT_UPDATE_METRICS),), dtype=dtype),
        )

    def evaluation_parameters(
        self, rule_state: _AlignedUpdateState, parameters: Any, /
    ) -> Any:
        return resolve_evaluation_parameters(
            self.evaluation_view, rule_state.optimizer_state, parameters
        )

    def propose(
        self,
        parameters: Any,
        gradients: Any,
        value: jax.Array,
        rule_state: _AlignedUpdateState,
        context: KernelUpdateContext,
        /,
    ) -> tuple[Any, _AlignedUpdateState, _AlignedUpdateState, jax.Array]:
        del value
        diagnostics = context.diagnostics[0]
        component_gradients = diagnostics["component_gradients"]
        updates, optimizer_state = self.optimizer.update(
            gradients, rule_state.optimizer_state, parameters
        )
        result = project_conflict_free_direction(
            tree_negative(updates),
            component_gradients,
            active=diagnostics["active"],
            policy=self.policy,
        )
        updates = tree_where(result.projected, tree_negative(result.direction), updates)
        updates = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                ~result.successful,
                "Operator optimizer proposal could not be aligned.",
            ),
            updates,
        )
        gradient_conflict, constructed_conflict = _alignment_conflicts(
            component_gradients, gradients, result, self.policy
        )
        statistics = jax.tree.map(
            lambda updated, previous: updated.astype(previous.dtype),
            rule_state.statistics.update(
                result,
                gradient_conflict=gradient_conflict,
                constructed_conflict=constructed_conflict,
            ),
            rule_state.statistics,
        )
        dtype = rule_state.last_update.dtype
        last_update = jnp.stack(
            tuple(
                jnp.asarray(metric).astype(dtype)
                for metric in (
                    jnp.any(result.raw_conflicts),
                    jnp.any(result.aligned_conflicts),
                    result.projected,
                    result.relative_correction,
                    result.metric_correction_norm,
                    result.active_constraint_count,
                    result.pareto_stationary,
                    result.kkt_residual_norm,
                    result.status,
                )
            )
        )
        candidate = eqx.apply_updates(parameters, updates)
        return (
            candidate,
            _AlignedUpdateState(optimizer_state, statistics, last_update),
            rule_state,
            result.successful,
        )


def _alignment_metrics(state: _AlignedUpdateState, /) -> dict[str, float]:
    statistics = state.statistics
    last_update, *rates = jax.device_get(
        (
            state.last_update,
            statistics.gradient_conflict_rate,
            statistics.constructed_conflict_rate,
            statistics.proposal_conflict_rate,
            statistics.applied_conflict_rate,
        )
    )
    metrics = {
        name: float(value)
        for name, value in zip(_ALIGNMENT_UPDATE_METRICS, last_update, strict=True)
    }
    metrics.update(
        {
            f"update_alignment/{name}_conflict_rate": float(rate)
            for name, rate in zip(
                ("gradient", "constructed", "proposal", "applied"), rates, strict=True
            )
        }
    )
    return metrics


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
    root_key = jr.key(seed) if key is None else _typed_root_key(key)
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
    return root_key, sharding_policy


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
        require_parameter_roles(model, context="fit_operator")
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
    output_ports: Mapping[str, ValuePort] | None = None,
    port_mapping: PortMapping | None = None,
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

    Task-bound fits require ``output_ports`` (the ``ValuePort`` of each named
    model output) and a ``port_mapping`` binding those ports to task target field
    ports; taskless fits train in raw model coordinates and bind no ports.

    Experimental ``update_alignment`` projects the exact emitted optimizer
    proposal against every supported explicit loss term before parameter
    application. It requires one microstep, excludes attached model losses and
    loss scaling, and checkpoints cumulative mismatch evidence.
    """
    root_key, sharding_policy = _resolve_operator_fit_execution(
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
    output_binding = _resolve_output_binding(model, task, output_ports, port_mapping)
    output_routes = (
        None if output_binding is None else _operator_output_routes(task, *output_binding)
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
        private_prepared = _prepare_private_gradient(
            privacy,
            noise_key=derive_key(root_key, _PRIVACY_NOISE_ADDRESS),
            sampler_seed=_private_sampler_seed(root_key),
        )
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    resume_metadata: dict[str, Any] | None = None
    resume_probe: tuple[int, int] | None = None
    if checkpoint is not None and resume and (checkpoint / "manifest.json").is_file():
        resume_metadata = read_training_checkpoint_metadata(
            checkpoint, format=_FIT_CHECKPOINT_FORMAT
        )
    current_data_contract = {
        "train_loader_fingerprint": raw_train_loader.fingerprint,
        "validation_loader_fingerprint": (
            None if raw_validation_loader is None else raw_validation_loader.fingerprint
        ),
    }
    if resume_metadata is not None:
        metadata = resume_metadata
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

    training_tree = (
        model
        if parameter_paths is None
        else SubspaceTrainingTree.from_subspace(effective_subspace)
    )

    def fit_model(tree):
        return tree.model() if isinstance(tree, SubspaceTrainingTree) else tree

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

    if loss_terms is None and not first.targets.fields:
        raise ValueError(
            "Targetless operator fitting requires explicit physics loss_terms."
        )
    terms = (
        _default_losses(model, output_routes) if loss_terms is None else specified_terms
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
        assert output_binding is not None
        _validate_rollout_route(
            rollout_route,
            task,
            *output_binding,
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
        assert output_routes is not None
        return _evaluate_operator_step(
            evaluated_model,
            batch,
            physical_batch,
            task,
            output_routes,
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
            derive_key(root_key, _PREFLIGHT_ADDRESS),
        )

    parameters = partition_parameters(training_tree)[0]
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
    compression_record = None
    if optimizer_state_compression is not None:
        if not isinstance(
            optimizer_state_compression,
            OptimizerStateCompressionPolicy,
        ):
            raise TypeError(
                "optimizer_state_compression must be OptimizerStateCompressionPolicy."
            )
        compressed = prepare_compressed_optimizer(
            optimizer,
            parameters,
            optimizer_state_compression,
            transformation_id=resolved_optimizer_id,
        )
        transformation_type = (
            optax.GradientTransformationExtraArgs
            if isinstance(optimizer, optax.GradientTransformationExtraArgs)
            else optax.GradientTransformation
        )
        optimizer = transformation_type(compressed.init, compressed.update)
        compression_record = {
            "format": repr(optimizer_state_compression.format),
            "block_axes": optimizer_state_compression.block_axes,
            "exact_roles": optimizer_state_compression.exact_roles,
            "overflow": optimizer_state_compression.overflow,
        }
    reduction_dtype = jnp.dtype(resolved_dtype.reduction_dtype)
    loss_scale_state = (
        OperatorLossScaleState(jnp.asarray(1.0, dtype=reduction_dtype))
        if loss_scale_policy is None
        else loss_scale_policy.initial_state(reduction_dtype)
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
        model_key = derive_key(key, _MODEL_ADDRESS)
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
                output_routes,
                output_pipeline,
                resolved_normalization,
                resolved_dtype,
                sharding_policy,
                model_key,
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
                model_key,
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
                derive_key(key, _TARGET_MODEL_ADDRESS),
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

        def recurrent_contribution(term):
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
                        key=derive_key(
                            key,
                            SampleAddress(
                                _FIT_OBJECTIVE_ID,
                                "residual-rollout",
                                target=(term.name,),
                            ),
                            depth,
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
                recurrent_contribution(term)
                if isinstance(
                    term,
                    (SupervisedOperatorRolloutLoss, ResidualOperatorRolloutLoss),
                )
                else term.contribution(
                    evaluated_model,
                    physical_prediction,
                    physical_batch,
                    physical_targets,
                    key=derive_key(
                        key,
                        SampleAddress(
                            _FIT_OBJECTIVE_ID, "loss-term", target=(term.name,)
                        ),
                    ),
                    step=step,
                    training=training,
                    context=context,
                )
            )
            for term in terms
        )
        attached = (
            tuple(
                _case_mean_contribution(
                    resolved_dtype.reduction(value),
                    context,
                )
                for value in model_loss_values(
                    evaluated_model,
                    key=derive_key(key, _MODEL_LOSS_ADDRESS),
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
    has_targets = target_policy is not None

    def payload_models(parameters_, model_state, fixed, payload):
        current_model = fit_model(combine_parameters(parameters_, model_state, fixed))
        target_model = (
            None
            if payload.target_parameters is None
            else fit_model(
                combine_parameters(payload.target_parameters, model_state, fixed)
            )
        )
        return current_model, target_model

    def payload_components(parameters_, model_state, fixed, payload, key):
        current_model, target_model = payload_models(
            parameters_, model_state, fixed, payload
        )
        return loss_components(
            current_model,
            target_model,
            payload.batch,
            payload.targets,
            payload.physical_batch,
            payload.physical_targets,
            payload.case_log_weights,
            payload.case_mask,
            payload.sampling_probabilities,
            key,
            payload.step,
            payload.active_horizon,
            training=True,
        )

    def contribution_diagnostics(total, components):
        return {
            "total": (total.numerator, total.support, total.log_scale),
            "components": tuple(
                (component.numerator, component.support, component.log_scale)
                for component in components
            ),
        }

    def direct_contribution(parameters_, model_state, fixed, payload, keys):
        candidate = (
            parameters_
            if loss_scale_policy is None
            else _loss_unscaled(loss_scale_policy, parameters_, payload.loss_scale)
        )
        total, components = payload_components(
            candidate, model_state, fixed, payload, keys.attempt_key(_LOSS_KEY_SITE)
        )
        numerator = (
            total.numerator
            if loss_scale_policy is None
            else _loss_scaled(loss_scale_policy, total.numerator, payload.loss_scale)
        )
        return (
            _ObjectiveContribution(numerator, total.support, total.log_scale),
            model_state,
            contribution_diagnostics(total, components),
        )

    def component_contribution(parameters_, model_state, fixed, payload, keys):
        key = keys.attempt_key(_LOSS_KEY_SITE)

        def component_objective(candidate):
            total, components = payload_components(
                candidate, model_state, fixed, payload, key
            )
            values = jnp.stack(tuple(component.value for component in components))
            active_ = jnp.stack(
                tuple(component.support > 0.0 for component in components)
            )
            return (total.numerator, values), (
                contribution_diagnostics(total, components),
                active_,
            )

        (
            (total_numerator, component_values),
            pullback,
            (diagnostics, active),
        ) = eqx.filter_vjp(
            component_objective,
            jax.lax.stop_gradient(parameters_),
            has_aux=True,
        )
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
            direction = pullback(
                (jnp.ones_like(total_numerator), jnp.zeros_like(component_values))
            )[0]
        else:
            composition = conflict_free_gradient(
                component_gradients,
                active=active,
                policy=gradient_composition,
            )
            direction = jax.tree.map(
                lambda leaf: eqx.error_if(
                    leaf,
                    ~composition.successful,
                    "Operator objectives do not admit a conflict-free direction.",
                ),
                composition.direction,
            )
        numerator, support, log_scale = diagnostics["total"]
        return (
            _ObjectiveContribution(
                _prescribed_gradient(numerator, parameters_, direction),
                support,
                log_scale,
            ),
            model_state,
            {
                **diagnostics,
                "component_gradients": component_gradients,
                "active": active,
            },
        )

    def private_contribution(parameters_, model_state, fixed, payload, keys):
        assert private_prepared is not None

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
            physical_batch = eqx.combine(
                physical_batch_arrays, private_physical_batch_static
            )
            physical_targets_ = eqx.combine(
                physical_target_arrays, private_physical_targets_static
            )
            current_model = fit_model(
                combine_parameters(current_parameters, model_state, fixed)
            )
            target_model = (
                fit_model(combine_parameters(target_parameters, model_state, fixed))
                if has_targets
                else None
            )
            total, _ = loss_components(
                current_model,
                target_model,
                slice_operator_batch(batch, case_indices, axis=0),
                targets.take(case_indices, axis=0),
                slice_operator_batch(physical_batch, case_indices, axis=0),
                physical_targets_.take(case_indices, axis=0),
                jnp.take(case_log_weights, case_indices, axis=0),
                jnp.take(case_mask, case_indices, axis=0),
                jnp.take(sampling_probabilities, case_indices, axis=0),
                key,
                step,
                active_rollout_horizon,
                training=True,
            )
            return total.numerator

        clipped_gradient_fn = private_prepared.clipped_grad(
            private_objective,
            argnums=0,
            batch_argnums=2,
            keep_batch_dim=True,
            prng_argnum=10,
        )
        padding = jnp.asarray(payload.is_padding_example, dtype=jnp.bool_)
        real = ~padding
        case_mask = eqx.error_if(
            jnp.asarray(payload.case_mask, dtype=jnp.bool_),
            jnp.any(real & ~jnp.asarray(payload.case_mask, dtype=jnp.bool_)),
            "The initial private profile requires every sampled case active.",
        )
        case_log_weights = eqx.error_if(
            jnp.asarray(payload.case_log_weights),
            jnp.any(
                real
                & (
                    ~jnp.isfinite(payload.case_log_weights)
                    | (jnp.asarray(payload.case_log_weights) != 0.0)
                )
            ),
            "The initial private profile requires uniform case weights.",
        )
        sampling_probabilities = eqx.error_if(
            jnp.asarray(payload.sampling_probabilities),
            jnp.any(
                real
                & (
                    ~jnp.isfinite(payload.sampling_probabilities)
                    | (jnp.asarray(payload.sampling_probabilities) != 1.0)
                )
            ),
            "The private sampler owns inclusion probabilities.",
        )
        stopped = jax.lax.stop_gradient(parameters_)
        clipped_gradient = clipped_gradient_fn(
            stopped,
            stopped if payload.target_parameters is None else payload.target_parameters,
            jnp.arange(payload.batch.case_shape[0], dtype=jnp.int32),
            eqx.filter(payload.batch, eqx.is_array),
            eqx.filter(payload.targets, eqx.is_array),
            eqx.filter(payload.physical_batch, eqx.is_array),
            eqx.filter(payload.physical_targets, eqx.is_array),
            case_log_weights,
            case_mask,
            sampling_probabilities,
            keys.attempt_key(_LOSS_KEY_SITE),
            payload.step,
            payload.active_horizon,
            is_padding_example=payload.is_padding_example,
        )
        gradient = private_prepared.privatize(
            clipped_gradient, keys.attempt_key(_PRIVACY_NOISE_KEY_SITE)
        )
        zero = jnp.asarray(0.0, dtype=reduction_dtype)
        one = jnp.asarray(1.0, dtype=reduction_dtype)
        return (
            _ObjectiveContribution(
                _prescribed_gradient(zero, parameters_, gradient), one, zero
            ),
            model_state,
            {
                "total": (zero, one, zero),
                "components": tuple((zero, one, zero) for _ in terms),
            },
        )

    def operator_objective(parameters_, model_state, fixed, payload, keys):
        if private_prepared is not None:
            return private_contribution(parameters_, model_state, fixed, payload, keys)
        if gradient_composition is not None or update_alignment is not None:
            return component_contribution(parameters_, model_state, fixed, payload, keys)
        return direct_contribution(parameters_, model_state, fixed, payload, keys)

    rule_id = canonical_fingerprint(
        {
            "kind": "operator-fit-update-rule",
            "optimizer_id": resolved_optimizer_id,
            "optimizer_state_compression": compression_record,
            "evaluation_parameters_id": resolved_evaluation_parameters_id,
            "update_alignment": (
                None if update_alignment is None else update_alignment.policy_id
            ),
        }
    )
    rule: AbstractKernelUpdateRule = (
        OptaxUpdateRule(
            optimizer, rule_id=rule_id, evaluation_parameters=evaluation_parameters
        )
        if update_alignment is None
        else _AlignedOptaxUpdateRule(
            optimizer=optimizer,
            policy=update_alignment,
            evaluation_view=evaluation_parameters,
            rule_id=rule_id,
            statistics_dtype=reduction_dtype.name,
        )
    )
    kernel = None
    kernel_state: TrainingKernelState | None = None
    if _has_trainable_arrays(parameters):
        kind, route = (
            (ObjectiveKind.ROLLOUT, DerivativeRoute.UNROLLED)
            if rollout_terms
            else (ObjectiveKind.DATA_FIT, DerivativeRoute.DIRECT)
        )
        kernel = prepare_training_kernel(
            training_tree,
            (
                KernelObjective(
                    objective_id=_FIT_OBJECTIVE_ID,
                    kind=kind,
                    route=route,
                    fn=_OperatorFitObjective(operator_objective),
                ),
            ),
            TrainingKernelSpec(
                rule,
                context="fit_operator",
                rejection_budget=_loss_scale_rejection_budget(loss_scale_policy),
                target_policy=target_policy,
                accumulation_dtype=reduction_dtype,
            ),
            root_authority=ComponentAuthority.SURROGATE,
        )
        kernel_state = kernel.init(training_tree, root_key)
    elif checkpoint is not None:
        raise ValueError("Checkpointed operator fits require trainable parameters.")

    def evaluation_model_of(state):
        if kernel is None:
            return model
        evaluated = kernel.rule.evaluation_parameters(state.rule_state, state.parameters)
        return fit_model(combine_parameters(evaluated, state.model_state, kernel.fixed))

    def target_model_of(state):
        if not has_targets:
            return None
        return model if kernel is None else fit_model(kernel.target_tree(state))

    def attempt(kernel_, state, payload):
        return kernel_.attempt(state, payload)

    def accumulate(kernel_, state, payload):
        return kernel_.accumulate_with_diagnostics(state, payload)

    run_attempt = eqx.filter_jit(attempt) if jit else attempt
    run_accumulation = eqx.filter_jit(accumulate) if jit else accumulate
    evaluation_model = evaluation_model_of(kernel_state)

    def add_window_metrics(accumulators, diagnostics):
        if private_prepared is not None:
            return accumulators
        contributions = (_ObjectiveContribution(*diagnostics["total"]),) + tuple(
            _ObjectiveContribution(*component) for component in diagnostics["components"]
        )
        return [
            accumulator.add(contribution)
            for accumulator, contribution in zip(accumulators, contributions, strict=True)
        ]

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

    def evaluate(current_model, target_model, loader: OperatorBatchLoader, step: int):
        metric_accumulators = [_ObjectiveAccumulator() for _ in metric_names]
        batch_count = 0
        active_rollout_horizon = resolved_active_horizon(step)
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
                training_accepted_site_key(
                    root_key,
                    objective_id=_FIT_OBJECTIVE_ID,
                    site=f"metrics/{loader.split}",
                    accepted=int(step),
                    microstep=batch_index,
                ),
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
        "port_binding": (
            None if output_binding is None else _port_binding_record(*output_binding)
        ),
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
        "key_sites": _FIT_KEY_SITES,
        "optimizer_id": resolved_optimizer_id,
        "target_policy": (None if target_policy is None else asdict(target_policy)),
        "optimizer_state_compression": compression_record,
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
    sharding_identity = (
        None
        if sharding_policy is None
        else canonical_fingerprint(fit_contract["sharding"])
    )
    schema = {
        "fit": operator_fit_schema(first.batch, target=first.targets),
    }
    expected_privacy_classification = None if private_prepared is None else "restricted"

    initial_metrics: dict[str, float]
    if resume_metadata is not None:
        assert checkpoint is not None
        assert kernel is not None and kernel_state is not None
        if resume_metadata.get("fit_contract") != fit_contract:
            raise ValueError("Operator fit checkpoint contract mismatch.")
        loaded = load_training_checkpoint(
            checkpoint,
            kernel,
            kernel_state,
            (best_model, loss_scale_state),
            format=_FIT_CHECKPOINT_FORMAT,
            sharding_identity=sharding_identity,
        )
        metadata = loaded.metadata
        if metadata.get("fit_contract") != fit_contract:
            raise ValueError("Operator fit checkpoint contract mismatch.")
        if metadata.get("schema") != schema:
            raise ValueError("Operator fit checkpoint schema mismatch.")
        if metadata.get("privacy_classification") != expected_privacy_classification:
            raise ValueError("Operator checkpoint privacy classification changed.")
        kernel_state = loaded.restored.state
        best_model, loss_scale_state = loaded.extra
        progress = TrainingProgress(**metadata["progress"])
        if progress.update_step != int(jax.device_get(kernel_state.accepted_cursor)):
            raise ValueError("Checkpoint progress disagrees with its update step.")
        if progress.update_step > maximum_steps:
            raise ValueError("Checkpoint step exceeds the requested training ceiling.")
        root_key = kernel_state.root_key
        if private_prepared is not None:
            private_prepared = replace(
                private_prepared, sampler_seed=_private_sampler_seed(root_key)
            )
        control = TrainingController(
            total_steps=maximum_steps,
            algorithm_id="operator-training",
            progress=progress,
            session=iteration_session,
        )
        control.best_payload = best_model
        train_steps = [int(value) for value in metadata["train_steps"]]
        train_history = [_decode_metrics(values) for values in metadata["train_metrics"]]
        validation_steps = [int(value) for value in metadata["validation_steps"]]
        validation_history = [
            _decode_metrics(values) for values in metadata["validation_metrics"]
        ]
        initial_metrics = _decode_metrics(metadata["initial_metrics"])
        prior_training_seconds = float(metadata["training_seconds"])
        resumed_from_step = progress.update_step
        evaluation_model = evaluation_model_of(kernel_state)
    else:
        initial_metrics = (
            {}
            if private_prepared is not None
            else evaluate(
                evaluation_model,
                target_model_of(kernel_state),
                raw_train_loader,
                0,
            )
        )
        if raw_validation_loader is not None:
            validation_metrics = evaluate(
                evaluation_model,
                target_model_of(kernel_state),
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

    # Checkpoints publish accepted-update boundaries (or the untouched start) only.
    at_accepted_boundary = True

    def save_progress(training_seconds: float, *, emit_event: bool = True) -> None:
        if checkpoint is None or not at_accepted_boundary:
            return
        assert kernel is not None and kernel_state is not None
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
        save_training_checkpoint(
            checkpoint,
            build_training_checkpoint(
                kernel, kernel_state, sharding_identity=sharding_identity
            ),
            (best_model, loss_scale_state),
            format=_FIT_CHECKPOINT_FORMAT,
            metadata={
                "fit_contract": fit_contract,
                "data_contract": current_data_contract,
                "schema": schema,
                "progress": asdict(control.progress),
                "privacy_classification": expected_privacy_classification,
                "initial_metrics": _encode_metrics(initial_metrics),
                "train_steps": train_steps,
                "train_metrics": [_encode_metrics(values) for values in train_history],
                "validation_steps": validation_steps,
                "validation_metrics": [
                    _encode_metrics(values) for values in validation_history
                ],
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
        if not control.progress.stopped_early and kernel is not None:
            assert kernel_state is not None
            window_microsteps = 0
            window_metrics = [_ObjectiveAccumulator() for _ in metric_names]
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
                    update_step = control.progress.update_step + 1
                    payload = _OperatorFitPayload(
                        batch=training_batch.batch,
                        targets=training_batch.targets,
                        physical_batch=(
                            training_batch.batch
                            if training_batch.physical_batch is None
                            else training_batch.physical_batch
                        ),
                        physical_targets=(
                            training_batch.targets
                            if training_batch.physical_targets is None
                            else training_batch.physical_targets
                        ),
                        case_log_weights=training_batch.case_log_weights,
                        case_mask=training_batch.case_mask,
                        sampling_probabilities=training_batch.sampling_probabilities,
                        is_padding_example=training_batch.is_padding_example,
                        step=jnp.asarray(update_step, dtype=jnp.float64),
                        target_parameters=(
                            None
                            if kernel_state.targets is None
                            else kernel_state.targets.target
                        ),
                        loss_scale=(
                            None if loss_scale_policy is None else loss_scale_state.scale
                        ),
                        active_horizon=resolved_active_horizon(update_step),
                    )
                    control.progress = replace(
                        control.progress,
                        microstep=control.progress.microstep + 1,
                        next_batch_index=training_batch.batch_index + 1,
                    )
                    window_microsteps += 1
                    at_accepted_boundary = False
                    if (
                        window_microsteps < int(gradient_accumulation)
                        and training_batch.batch_index + 1 < batches_per_training_epoch
                    ):
                        kernel_state, diagnostics = run_accumulation(
                            kernel, kernel_state, payload
                        )
                        window_metrics = add_window_metrics(
                            window_metrics, diagnostics[0]
                        )
                        continue
                    kernel_state, attempt_evidence = run_attempt(
                        kernel, kernel_state, payload
                    )
                    closed_metrics = add_window_metrics(
                        window_metrics, attempt_evidence.diagnostics[0]
                    )
                    window_microsteps = 0
                    window_metrics = [_ObjectiveAccumulator() for _ in metric_names]
                    outcome_array, supported, consecutive, attempt_cursor = (
                        jax.device_get(
                            (
                                attempt_evidence.outcome,
                                attempt_evidence.supported,
                                kernel_state.consecutive_rejections,
                                kernel_state.attempt_cursor,
                            )
                        )
                    )
                    outcome = TrainingAttemptOutcome(int(outcome_array))
                    if outcome is TrainingAttemptOutcome.REJECTED_FINITE:
                        # Optax rules accept every finite proposal, so a supported
                        # finite rejection is a nonfinite update from a finite
                        # evaluation; an unsupported window is skipped.
                        if bool(supported):
                            raise FloatingPointError(
                                "Operator optimizer produced non-finite state from "
                                "finite gradients; the update was rolled back."
                            )
                        continue
                    if outcome is TrainingAttemptOutcome.NONFINITE:
                        if loss_scale_policy is None or not loss_scale_policy.dynamic:
                            raise FloatingPointError(
                                "Non-finite operator loss or gradient encountered; "
                                "the update was rolled back."
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
                        try:
                            enforce_rejection_budget(
                                kernel, outcome_array, consecutive, attempt_cursor
                            )
                        except TrainingRejectionBudgetError as error:
                            raise FloatingPointError(
                                "Non-finite operator loss or gradient persists at the "
                                "minimum loss scale; the fit was rolled back to its "
                                "last accepted update."
                            ) from error
                        continue
                    at_accepted_boundary = True
                    control.complete_update(update_step)
                    metrics = (
                        {}
                        if private_prepared is not None
                        else {
                            name: float(jax.device_get(accumulator.value))
                            for name, accumulator in zip(
                                metric_names,
                                closed_metrics,
                                strict=True,
                            )
                        }
                    )
                    if update_alignment is not None:
                        metrics.update(_alignment_metrics(kernel_state.rule_state))
                    train_steps.append(update_step)
                    train_history.append(metrics)
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
                        evaluation_model = evaluation_model_of(kernel_state)
                        validation_metrics = evaluate(
                            evaluation_model,
                            target_model_of(kernel_state),
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
    evaluation_model = evaluation_model_of(kernel_state)
    if (
        raw_validation_loader is not None
        and validation_config is not None
        and (not validation_steps or validation_steps[-1] != control.progress.update_step)
    ):
        validation_metrics = evaluate(
            evaluation_model,
            target_model_of(kernel_state),
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
            target_model_of(kernel_state),
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
            output_ports=output_ports,
            port_mapping=port_mapping,
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
        port_binding=None if output_binding is None else output_binding[1],
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
        update_alignment_statistics=(
            None
            if update_alignment is None or kernel_state is None
            else kernel_state.rule_state.statistics
        ),
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
