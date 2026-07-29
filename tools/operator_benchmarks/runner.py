from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax._trainable import combine_trainable, partition_trainable
from phydrax.nn import (
    fit_operator,
    FunctionSamples,
    operator_architecture_contract,
    operator_conservation_error,
    operator_h1_loss,
    operator_l2_loss,
    operator_spectral_loss,
    OperatorBatch,
    OperatorCaseProvenance,
    OperatorDataset,
    OperatorDTypePolicy,
    OperatorPrediction,
    OperatorTargetBatch,
    OperatorValidationPolicy,
    SupervisedOperatorLoss,
)
from phydrax.nn.models.core._base import _AbstractOperatorModel

from .scenarios import (
    _apply_square_group_action,
    OperatorBenchmarkEvaluation,
    OperatorBenchmarkScenario,
    OperatorSymmetrySpec,
    SquareFieldRepresentation,
    SquareSymmetryGroup,
)


class _BenchmarkFitOperator(_AbstractOperatorModel):
    """Production-fitter adapter for callable benchmark-only baselines."""

    model: Callable[[OperatorBatch], jax.Array]
    in_size: Literal["scalar"] = eqx.field(static=True)
    out_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        model: Callable[[OperatorBatch], jax.Array],
        scenario: OperatorBenchmarkScenario,
    ):
        if isinstance(scenario.train_target, OperatorTargetBatch):
            raise TypeError(
                "Named benchmark targets require a native operator model."
            )
        query = scenario.train_batch.require_single_query()
        prefix_rank = len(scenario.train_batch.case_shape) + len(query.sample_shape)
        trailing = tuple(
            int(size) for size in jnp.asarray(scenario.train_target).shape[prefix_rank:]
        )
        self.model = model
        self.in_size = "scalar"
        if not trailing:
            self.out_size = "scalar"
        elif len(trailing) == 1:
            self.out_size = trailing[0]
        else:
            self.out_size = trailing

    @property
    def operator_contract(self):
        return operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        return self.model(batch)

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


@dataclass(frozen=True)
class OperatorFieldEvaluationResult:
    """Metrics for one named physical output field."""

    name: str
    query_name: str
    relative_l2: float
    relative_l2_per_case: tuple[float, ...]
    absolute_l2: float
    h1: float | None
    spectral: float | None
    conservation_error: float
    maximum_absolute_error: float


@dataclass(frozen=True)
class OperatorEvaluationResult:
    name: str
    split: str
    shift: str
    rollout_steps: int
    relative_l2: float
    relative_l2_per_case: tuple[float, ...]
    absolute_l2: float
    h1: float | None
    spectral: float | None
    conservation_error: float
    maximum_absolute_error: float
    compile_seconds: float
    inference_seconds: float
    peak_memory_bytes: int | None
    field_metrics: tuple[OperatorFieldEvaluationResult, ...] = ()


@dataclass(frozen=True)
class OperatorSymmetryEvaluationResult:
    """Paired group-transform evaluation for one trained model and split."""

    name: str
    declared_group: str | None
    audit_group: str
    element_relative_l2: tuple[float, ...]
    element_maximum_absolute_error: tuple[float, ...]
    mean_equivariance_defect: float | None
    worst_equivariance_defect: float | None
    maximum_absolute_equivariance_error: float | None
    mean_rotated_pair_difference: float
    mean_reflected_pair_difference: float | None


@dataclass(frozen=True)
class OperatorBenchmarkResult:
    scenario: str
    architecture: str
    family: str
    seed: int
    parameter_count: int
    training_steps: int
    training_seconds: float
    initial_loss: float
    final_loss: float
    validation_loss: float | None
    losses: tuple[float, ...]
    evaluations: tuple[OperatorEvaluationResult, ...]
    validation_steps: tuple[int, ...] = ()
    validation_losses: tuple[float, ...] = ()
    stopped_early: bool = False
    converged: bool = False
    resumed_from_step: int = 0
    size_scale: float = 1.0
    architecture_configuration: tuple[tuple[str, str], ...] = ()

    def to_dict(self):
        return asdict(self)


def parameter_count(model) -> int:
    trainable, _ = partition_trainable(model)
    return sum(
        int(leaf.size) * (2 if jnp.issubdtype(leaf.dtype, jnp.complexfloating) else 1)
        for leaf in jax.tree_util.tree_leaves(trainable)
        if isinstance(leaf, jax.Array)
    )


def _memory_bytes() -> int | None:
    statistics = jax.devices()[0].memory_stats()
    if statistics is None:
        return None
    for key in ("peak_bytes_in_use", "bytes_in_use"):
        if key in statistics:
            return int(statistics[key])
    return None


def _target_batch(
    target: jax.Array | OperatorTargetBatch,
    batch: OperatorBatch,
    /,
) -> OperatorTargetBatch:
    if isinstance(target, OperatorTargetBatch):
        target.validate(batch)
        return target
    return OperatorTargetBatch.from_arrays({"output": target}, batch)


def _prediction_for_target(
    model,
    batch: OperatorBatch,
    target: jax.Array | OperatorTargetBatch,
    /,
) -> jax.Array | OperatorPrediction:
    if isinstance(target, OperatorTargetBatch):
        if not isinstance(model, _AbstractOperatorModel):
            raise TypeError(
                "Named benchmark targets require an operator model with predict()."
            )
        return model.predict(batch)
    return model(batch)


def _field_arrays_from_prediction(
    prediction: jax.Array | OperatorPrediction,
    batch: OperatorBatch,
    target: jax.Array | OperatorTargetBatch,
    /,
) -> tuple[tuple[str, str, jax.Array, jax.Array], ...]:
    if not isinstance(target, OperatorTargetBatch):
        return (
            (
                "output",
                next(iter(batch.queries)),
                jnp.asarray(prediction),
                jnp.asarray(target),
            ),
        )
    if not isinstance(prediction, OperatorPrediction):
        raise TypeError("Named benchmark targets require an OperatorPrediction.")
    missing = tuple(name for name in target.fields if name not in prediction.fields)
    if missing:
        raise KeyError(f"Operator prediction is missing benchmark fields {missing}.")
    return tuple(
        (
            name,
            field.query_name,
            jnp.asarray(prediction.field(name).values),
            jnp.asarray(field.values),
        )
        for name, field in target.fields.items()
    )


def _field_arrays(
    model,
    batch: OperatorBatch,
    target: jax.Array | OperatorTargetBatch,
    /,
) -> tuple[tuple[str, str, jax.Array, jax.Array], ...]:
    return _field_arrays_from_prediction(
        _prediction_for_target(model, batch, target),
        batch,
        target,
    )


def _loss(
    model,
    batch: OperatorBatch,
    target: jax.Array | OperatorTargetBatch,
) -> jax.Array:
    losses = tuple(
        operator_l2_loss(
            prediction,
            truth,
            batch.query(query_name),
            relative=True,
            squared=True,
        )
        for _, query_name, prediction, truth in _field_arrays(
            model,
            batch,
            target,
        )
    )
    return jnp.mean(jnp.stack(losses))


def training_step_cost(
    model,
    scenario: OperatorBenchmarkScenario,
    /,
) -> tuple[int, int]:
    """Return XLA-estimated FLOPs and accessed bytes for one loss/gradient step."""
    if parameter_count(model) == 0:
        return 0, 0
    parameters, fixed = partition_trainable(model)

    @eqx.filter_jit
    def value_and_gradient(current_parameters):
        def objective(candidate):
            current_model = combine_trainable(candidate, fixed)
            return _loss(
                current_model,
                scenario.train_batch,
                scenario.train_target,
            )

        return eqx.filter_value_and_grad(objective)(current_parameters)

    compiled = value_and_gradient.lower(parameters).compile().compiled
    analysis = compiled.cost_analysis()
    flops = int(round(float(analysis.get("flops", 0.0))))
    accessed_bytes = int(round(float(analysis.get("bytes accessed", 0.0))))
    return flops, accessed_bytes


def _train_operator_with_trace(
    model,
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    steps: int = 100,
    learning_rate: float = 1e-3,
    trainable: bool = True,
    validation_interval: int = 10,
    patience: int | None = None,
    minimum_delta: float = 0.0,
    relative_minimum_delta: float = 0.0,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    checkpoint_metadata: Mapping[str, object] | None = None,
    checkpoint_key: jax.Array | None = None,
):
    """Adapt an immutable benchmark scenario to the production operator fitter."""
    if int(steps) < 0:
        raise ValueError("steps must be non-negative.")
    if int(validation_interval) <= 0:
        raise ValueError("validation_interval must be positive.")
    if patience is not None and int(patience) <= 0:
        raise ValueError("patience must be positive when supplied.")
    if float(minimum_delta) < 0.0 or float(relative_minimum_delta) < 0.0:
        raise ValueError("Convergence deltas must be non-negative.")

    if not isinstance(model, _AbstractOperatorModel):
        if not isinstance(model, eqx.Module):
            raise TypeError("Benchmark models must be Equinox modules.")
        model = _BenchmarkFitOperator(model, scenario)

    def dataset(
        batch: OperatorBatch,
        target: jax.Array | OperatorTargetBatch,
        case_ids: tuple[str, ...],
        /,
        *,
        prefix: str,
    ) -> OperatorDataset:
        if len(batch.case_shape) != 1:
            raise ValueError("Operator benchmarks require exactly one case axis.")
        resolved_ids = (
            tuple(case_ids)
            if case_ids
            else tuple(f"{prefix}:{index}" for index in range(batch.case_shape[0]))
        )
        if len(resolved_ids) != batch.case_shape[0]:
            raise ValueError("Benchmark case IDs must cover every operator case.")
        return OperatorDataset(
            batch,
            _target_batch(target, batch),
            tuple(OperatorCaseProvenance(case_id) for case_id in resolved_ids),
        )

    training_data = dataset(
        scenario.train_batch,
        scenario.train_target,
        scenario.case_ids,
        prefix=f"{scenario.name}:train",
    )
    if scenario.validation is None:
        validation_data = training_data
    else:
        validation_data = dataset(
            scenario.validation.batch,
            scenario.validation.target,
            scenario.validation.case_ids,
            prefix=f"{scenario.name}:validation",
        )
    trainable_leaves, _ = partition_trainable(model)
    use_float64 = any(
        isinstance(leaf, jax.Array)
        and (leaf.dtype == jnp.float64 or leaf.dtype == jnp.complex128)
        for leaf in jax.tree_util.tree_leaves(trainable_leaves)
    )
    dtype_name = "float64" if use_float64 else "float32"
    loss_terms = tuple(
        SupervisedOperatorLoss(
            name=f"supervised_l2:{name}",
            prediction_field=name,
            target_field=name,
            relative=True,
            squared=True,
        )
        for name in training_data.targets.fields
    )

    result = fit_operator(
        model,
        training_data,
        validation=validation_data,
        loss_terms=loss_terms,
        include_model_losses=False,
        task=scenario.task,
        learning_rate=float(learning_rate),
        epochs=max(int(steps), 1),
        steps=int(steps) if trainable else 0,
        batch_size=training_data.size,
        validation_batch_size=validation_data.size,
        shuffle=False,
        seed=int(scenario.seed),
        key=jr.key(0) if checkpoint_key is None else checkpoint_key,
        dtype_policy=OperatorDTypePolicy(
            parameter_dtype=dtype_name,
            compute_dtype=dtype_name,
            reduction_dtype=dtype_name,
        ),
        validation_policy=OperatorValidationPolicy(
            every=int(validation_interval),
            patience=patience,
            minimum_delta=float(minimum_delta),
            relative_minimum_delta=float(relative_minimum_delta),
            select_best=patience is not None,
        ),
        checkpoint_path=checkpoint_path,
        checkpoint_every=int(validation_interval),
        resume=resume,
        configuration={
            "scenario": scenario.name,
            "architecture_metadata": dict(checkpoint_metadata or {}),
        },
    )
    return (
        result.execution_model,
        result.initial_loss,
        result.final_loss,
        result.training_seconds,
        result.history.losses,
        result.history.validation_steps,
        result.history.validation_losses,
        result.progress.stopped_early,
        result.progress.stopped_early or not trainable,
        result.resumed_from_step,
    )


def train_operator(
    model,
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    steps: int = 100,
    learning_rate: float = 1e-3,
    trainable: bool = True,
    validation_interval: int = 10,
    patience: int | None = None,
    minimum_delta: float = 0.0,
    relative_minimum_delta: float = 0.0,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    checkpoint_metadata: Mapping[str, object] | None = None,
    checkpoint_key: jax.Array | None = None,
):
    """Train an operator and return the stable five-element public result tuple."""
    result = _train_operator_with_trace(
        model,
        scenario,
        steps=steps,
        learning_rate=learning_rate,
        trainable=trainable,
        validation_interval=validation_interval,
        patience=patience,
        minimum_delta=minimum_delta,
        relative_minimum_delta=relative_minimum_delta,
        checkpoint_path=checkpoint_path,
        resume=resume,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_key=checkpoint_key,
    )
    return result[:5]


def _with_source_values(
    batch: OperatorBatch,
    source_key: str,
    values: jax.Array,
    /,
) -> OperatorBatch:
    source = batch.input(source_key)
    updated = FunctionSamples(
        values=values,
        axes=source.axes,
        coordinates=source.coordinates,
        quadrature_weights=source.quadrature_weights,
        mask=source.mask,
        topology=source.topology,
    )
    inputs = dict(batch.inputs)
    inputs[source_key] = updated
    return OperatorBatch(
        inputs=inputs,
        queries={"query": batch.require_single_query()},
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def _predict_evaluation(model, evaluation: OperatorBenchmarkEvaluation):
    batch = evaluation.batch
    prediction = _prediction_for_target(model, batch, evaluation.target)
    if evaluation.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if isinstance(evaluation.target, OperatorTargetBatch):
        if evaluation.rollout_steps != 1:
            raise ValueError("Named multi-field evaluations do not support rollout.")
        return prediction
    if isinstance(prediction, OperatorPrediction):
        raise TypeError("Array rollout evaluations require array predictions.")
    if evaluation.rollout_steps > 1:
        if evaluation.rollout_source_key is None:
            raise ValueError("Rollout evaluations require rollout_source_key.")
        for _ in range(1, evaluation.rollout_steps):
            batch = _with_source_values(
                batch,
                evaluation.rollout_source_key,
                prediction,
            )
            prediction = model(batch)
            if isinstance(prediction, OperatorPrediction):
                raise TypeError("Array rollout evaluations require array predictions.")
    return prediction


def _sample_spatial_axes(
    value: jax.Array,
    sample_shape: tuple[int, ...],
    case_ndim: int,
    /,
) -> tuple[int, int]:
    array = jnp.asarray(value)
    if (
        array.ndim >= case_ndim + 2
        and tuple(array.shape[case_ndim : case_ndim + 2]) == sample_shape
    ):
        return case_ndim, case_ndim + 1
    if array.ndim >= 2 and tuple(array.shape[:2]) == sample_shape:
        return 0, 1
    raise ValueError(
        f"Cannot locate square sample shape {sample_shape} in array {array.shape}."
    )


def _transform_square_samples(
    samples: FunctionSamples,
    element: int,
    /,
    *,
    group: SquareSymmetryGroup,
    representation: SquareFieldRepresentation,
    case_ndim: int,
) -> FunctionSamples:
    if (
        len(samples.sample_shape) != 2
        or samples.sample_shape[0] != samples.sample_shape[1]
    ):
        raise ValueError(
            "Square-group evaluation requires a square rank-two sample grid."
        )
    if (
        not samples.axes
        or samples.coordinates is not None
        or samples.topology is not None
    ):
        raise ValueError(
            "Square-group evaluation requires axis-separable regular-grid samples."
        )
    sample_shape = samples.sample_shape

    def transform(value, current_representation):
        if value is None:
            return None
        return jax.tree_util.tree_map(
            lambda leaf: _apply_square_group_action(
                leaf,
                element,
                group=group,
                representation=current_representation,
                spatial_axes=_sample_spatial_axes(leaf, sample_shape, case_ndim),
            ),
            value,
        )

    return FunctionSamples(
        values=transform(samples.values, representation),
        axes=samples.axes,
        quadrature_weights=transform(samples.quadrature_weights, "scalar"),
        mask=transform(samples.mask, "scalar"),
    )


def transform_square_operator_batch(
    batch: OperatorBatch,
    symmetry: OperatorSymmetrySpec,
    element: int,
    /,
) -> OperatorBatch:
    """Apply one declared square-group action to every represented sample field."""
    representations = dict(symmetry.source_representations)
    inputs = {}
    for name, samples in batch.inputs.items():
        if name in representations:
            inputs[name] = _transform_square_samples(
                samples,
                element,
                group=symmetry.audit_group,
                representation=representations[name],
                case_ndim=len(batch.case_shape),
            )
        else:
            inputs[name] = samples
    query = _transform_square_samples(
        batch.require_single_query(),
        element,
        group=symmetry.audit_group,
        representation="scalar",
        case_ndim=len(batch.case_shape),
    )
    return OperatorBatch(
        inputs=inputs,
        queries={"query": query},
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def evaluate_operator_symmetry(
    model,
    evaluation: OperatorBenchmarkEvaluation,
    symmetry: OperatorSymmetrySpec,
    /,
) -> OperatorSymmetryEvaluationResult:
    """Measure paired rotational/reflection defects after the physical data split."""

    @eqx.filter_jit
    def predict(current_model, current_batch):
        return _predict_evaluation(
            current_model,
            OperatorBenchmarkEvaluation(
                evaluation.name,
                current_batch,
                evaluation.target,
                split=evaluation.split,
                shift=evaluation.shift,
                rollout_steps=evaluation.rollout_steps,
                rollout_source_key=evaluation.rollout_source_key,
                case_ids=evaluation.case_ids,
            ),
        )

    base_prediction = jax.block_until_ready(predict(model, evaluation.batch))
    count = 4 if symmetry.audit_group == "p4" else 8
    relative_defects = []
    maximum_errors = []
    case_ndim = len(evaluation.batch.case_shape)
    spatial_axes = (case_ndim, case_ndim + 1)
    reduction_axes = tuple(range(case_ndim, jnp.asarray(base_prediction).ndim))
    for element in range(count):
        transformed_batch = transform_square_operator_batch(
            evaluation.batch,
            symmetry,
            element,
        )
        transformed_prediction = jax.block_until_ready(predict(model, transformed_batch))
        expected = _apply_square_group_action(
            base_prediction,
            element,
            group=symmetry.audit_group,
            representation=symmetry.target_representation,
            spatial_axes=spatial_axes,
        )
        difference = jnp.asarray(transformed_prediction) - expected
        numerator = jnp.sqrt(jnp.sum(jnp.abs(difference) ** 2, axis=reduction_axes))
        denominator = jnp.maximum(
            jnp.sqrt(jnp.sum(jnp.abs(expected) ** 2, axis=reduction_axes)),
            1e-12,
        )
        relative_defects.append(float(jnp.mean(numerator / denominator)))
        maximum_errors.append(float(jnp.max(jnp.abs(difference))))

    exact_count = symmetry.exact_element_count
    exact_indices = tuple(range(1, exact_count))
    mean_equivariance = (
        None
        if not exact_indices
        else float(np.mean([relative_defects[index] for index in exact_indices]))
    )
    worst_equivariance = (
        None
        if not exact_indices
        else float(max(relative_defects[index] for index in exact_indices))
    )
    maximum_equivariance = (
        None
        if not exact_indices
        else float(max(maximum_errors[index] for index in exact_indices))
    )
    rotations = tuple(relative_defects[index] for index in range(1, min(4, count)))
    reflections = (
        tuple(relative_defects[index] for index in range(4, count)) if count == 8 else ()
    )
    return OperatorSymmetryEvaluationResult(
        name=evaluation.name,
        declared_group=symmetry.group,
        audit_group=symmetry.audit_group,
        element_relative_l2=tuple(relative_defects),
        element_maximum_absolute_error=tuple(maximum_errors),
        mean_equivariance_defect=mean_equivariance,
        worst_equivariance_defect=worst_equivariance,
        maximum_absolute_equivariance_error=maximum_equivariance,
        mean_rotated_pair_difference=float(np.mean(rotations)),
        mean_reflected_pair_difference=(
            None if not reflections else float(np.mean(reflections))
        ),
    )


def evaluate_operator(
    model,
    evaluation: OperatorBenchmarkEvaluation,
    /,
    *,
    repeats: int = 10,
) -> OperatorEvaluationResult:
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive.")

    @eqx.filter_jit
    def predict(current_model):
        return _predict_evaluation(current_model, evaluation)

    compile_started = time.perf_counter()
    prediction = jax.block_until_ready(predict(model))
    compile_seconds = time.perf_counter() - compile_started
    started = time.perf_counter()
    for _ in range(int(repeats)):
        prediction = jax.block_until_ready(predict(model))
    inference_seconds = (time.perf_counter() - started) / float(repeats)
    field_results = []
    per_case_metrics = []
    for name, query_name, field_prediction, field_target in (
        _field_arrays_from_prediction(
            prediction,
            evaluation.batch,
            evaluation.target,
        )
    ):
        query = evaluation.batch.query(query_name)
        relative_per_case = operator_l2_loss(
            field_prediction,
            field_target,
            query,
            relative=True,
            reduction="none",
        )
        relative = jnp.mean(relative_per_case)
        absolute = operator_l2_loss(
            field_prediction,
            field_target,
            query,
            relative=False,
        )
        conservation = operator_conservation_error(
            field_prediction,
            field_target,
            query,
            relative=False,
        )
        if query.axes:
            h1 = float(
                jax.block_until_ready(
                    operator_h1_loss(
                        field_prediction,
                        field_target,
                        query,
                        relative=True,
                    )
                )
            )
            spectral = float(
                jax.block_until_ready(
                    operator_spectral_loss(
                        field_prediction,
                        field_target,
                        query,
                        frequency_power=1.0,
                        relative=True,
                    )
                )
            )
        else:
            h1 = None
            spectral = None
        per_case_values = tuple(
            float(value)
            for value in jnp.asarray(relative_per_case).reshape((-1,)).tolist()
        )
        per_case_metrics.append(jnp.asarray(relative_per_case).reshape((-1,)))
        field_results.append(
            OperatorFieldEvaluationResult(
                name=name,
                query_name=query_name,
                relative_l2=float(jax.block_until_ready(relative)),
                relative_l2_per_case=per_case_values,
                absolute_l2=float(jax.block_until_ready(absolute)),
                h1=h1,
                spectral=spectral,
                conservation_error=float(jax.block_until_ready(conservation)),
                maximum_absolute_error=float(
                    jax.block_until_ready(
                        jnp.max(jnp.abs(field_prediction - field_target))
                    )
                ),
            )
        )
    relative_per_case = jnp.max(jnp.stack(per_case_metrics), axis=0)
    relative = max(field.relative_l2 for field in field_results)
    absolute = max(field.absolute_l2 for field in field_results)
    conservation = max(field.conservation_error for field in field_results)
    maximum_absolute_error = max(
        field.maximum_absolute_error for field in field_results
    )
    h1_values = tuple(field.h1 for field in field_results if field.h1 is not None)
    spectral_values = tuple(
        field.spectral for field in field_results if field.spectral is not None
    )
    return OperatorEvaluationResult(
        name=evaluation.name,
        split=evaluation.split,
        shift=evaluation.shift,
        rollout_steps=evaluation.rollout_steps,
        relative_l2=relative,
        relative_l2_per_case=tuple(
            float(value) for value in relative_per_case.tolist()
        ),
        absolute_l2=absolute,
        h1=None if not h1_values else max(h1_values),
        spectral=None if not spectral_values else max(spectral_values),
        conservation_error=conservation,
        maximum_absolute_error=maximum_absolute_error,
        compile_seconds=compile_seconds,
        inference_seconds=inference_seconds,
        peak_memory_bytes=_memory_bytes(),
        field_metrics=tuple(field_results),
    )


def run_operator_benchmark(
    model,
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    steps: int = 100,
    learning_rate: float = 1e-3,
    repeats: int = 10,
    size_scale: float = 1.0,
    architecture: str = "unspecified",
    family: str = "unspecified",
    architecture_configuration: tuple[tuple[str, str], ...] = (),
    seed: int = 0,
    trainable: bool = True,
    validation_interval: int = 10,
    patience: int | None = None,
    minimum_delta: float = 0.0,
    relative_minimum_delta: float = 0.0,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    checkpoint_metadata: Mapping[str, object] | None = None,
    checkpoint_key: jax.Array | None = None,
    run_evaluations: bool = True,
) -> tuple[object, OperatorBenchmarkResult]:
    (
        trained,
        initial,
        final,
        training_seconds,
        losses,
        validation_steps,
        validation_losses,
        stopped_early,
        converged,
        resumed_from_step,
    ) = _train_operator_with_trace(
        model,
        scenario,
        steps=steps,
        learning_rate=learning_rate,
        trainable=trainable,
        validation_interval=validation_interval,
        patience=patience,
        minimum_delta=minimum_delta,
        relative_minimum_delta=relative_minimum_delta,
        checkpoint_path=checkpoint_path,
        resume=resume,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_key=checkpoint_key,
    )
    evaluations = (
        tuple(
            evaluate_operator(trained, evaluation, repeats=repeats)
            for evaluation in scenario.evaluations
        )
        if run_evaluations
        else ()
    )
    if scenario.validation is None:
        validation_loss = None
    else:
        validation_loss = float(
            jax.block_until_ready(
                _loss(
                    trained,
                    scenario.validation.batch,
                    scenario.validation.target,
                )
            )
        )
    result = OperatorBenchmarkResult(
        scenario=scenario.name,
        architecture=str(architecture),
        family=str(family),
        seed=int(seed),
        parameter_count=parameter_count(trained),
        training_steps=len(losses),
        training_seconds=training_seconds,
        initial_loss=initial,
        size_scale=float(size_scale),
        final_loss=final,
        validation_loss=validation_loss,
        losses=losses,
        evaluations=evaluations,
        validation_steps=validation_steps,
        validation_losses=validation_losses,
        stopped_early=stopped_early,
        converged=converged,
        resumed_from_step=resumed_from_step,
        architecture_configuration=tuple(architecture_configuration),
    )
    return trained, result


__all__ = [
    "OperatorBenchmarkResult",
    "OperatorEvaluationResult",
    "OperatorFieldEvaluationResult",
    "OperatorSymmetryEvaluationResult",
    "evaluate_operator",
    "evaluate_operator_symmetry",
    "transform_square_operator_batch",
    "parameter_count",
    "run_operator_benchmark",
    "train_operator",
    "training_step_cost",
]
