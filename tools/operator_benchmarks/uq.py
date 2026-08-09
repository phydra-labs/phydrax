from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import polars as pl

import phydrax as phx
from phydrax.nn.operator import AbstractOperatorModel

from .matrix import benchmark_metadata, BenchmarkRunMetadata
from .models import compatible_architectures, OperatorArchitecture
from .runner import _loss, _with_source_values, parameter_count, train_operator
from .scenarios import OperatorBenchmarkEvaluation, OperatorBenchmarkScenario


@dataclass(frozen=True)
class OperatorUQBenchmarkProfile:
    scenario: OperatorBenchmarkScenario
    architecture: str


@dataclass(frozen=True)
class OperatorUQEvaluationResult:
    name: str
    split: str
    shift: str
    rollout_steps: int
    num_cases: int
    query_shape: tuple[int, ...]
    relative_l2: float
    h1: float | None
    epistemic_standard_deviation: float
    crps: float
    energy_score: float
    pointwise_coverage: float
    simultaneous_coverage: float
    interval_width: float
    valid_draw_count: int
    total_draw_count: int
    compile_seconds: float
    inference_seconds: float
    peak_memory_bytes: int | None
    coverage_confidence_lower: float | None
    coverage_confidence_upper: float | None
    nominal_coverage_compatible: bool | None


@dataclass(frozen=True)
class OperatorUQLaplaceResult:
    parameter_dimension: int
    parameter_variance_mean: float
    output_variance_mean: float
    posterior_sample_count: int
    geometry_preserved: bool


@dataclass(frozen=True)
class OperatorUQBenchmarkResult:
    scenario: str
    architecture: str
    family: str
    seeds: tuple[int, ...]
    ensemble_size: int
    parameter_count_mean: float
    training_steps: tuple[int, ...]
    training_seconds: float
    initial_losses: tuple[float, ...]
    final_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    calibration_cases: int
    calibration_radius: float
    nominal_coverage: float
    observation_scale: float
    evaluations: tuple[OperatorUQEvaluationResult, ...]
    laplace: OperatorUQLaplaceResult | None


@dataclass(frozen=True)
class OperatorUQBenchmarkSuite:
    metadata: BenchmarkRunMetadata
    calibration_case_checksums: tuple[tuple[str, str], ...]
    results: tuple[OperatorUQBenchmarkResult, ...]

    def to_dict(self):
        return asdict(self)


def run_operator_uq_benchmark(
    scenario: OperatorBenchmarkScenario,
    /,
    *,
    architecture: str,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    steps: int = 100,
    learning_rate: float = 1e-3,
    repeats: int = 5,
    alpha: float = 0.1,
    quick: bool = False,
    validation_interval: int = 10,
    patience: int | None = None,
    minimum_delta: float = 0.0,
    fit_projection_laplace: bool = True,
    posterior_samples: int = 32,
) -> tuple[tuple[AbstractOperatorModel, ...], OperatorUQBenchmarkResult]:
    """Train an independent deep ensemble and evaluate operator-aware UQ."""
    if scenario.validation is None:
        raise ValueError("Operator UQ benchmarks require a disjoint calibration split.")
    if not seeds:
        raise ValueError("Operator UQ benchmarks require at least one ensemble seed.")
    if int(repeats) <= 0:
        raise ValueError("repeats must be positive.")
    if int(posterior_samples) <= 0:
        raise ValueError("posterior_samples must be positive.")
    selected = _select_architecture(scenario, architecture, quick=quick)
    if not selected.trainable:
        raise ValueError("Operator UQ benchmark architectures must be trainable.")
    if any(
        isinstance(target, phx.nn.operator.OperatorTargetBatch)
        for target in (
            scenario.train_target,
            scenario.validation.target,
            *(evaluation.target for evaluation in scenario.evaluations),
        )
    ):
        raise TypeError(
            "Operator UQ benchmarks currently require one anonymous array target."
        )

    trained_members: list[AbstractOperatorModel] = []
    training_steps: list[int] = []
    initial_losses: list[float] = []
    final_losses: list[float] = []
    validation_losses: list[float] = []
    training_seconds = 0.0
    for seed in seeds:
        model = selected.build(scenario, seed)
        if not isinstance(model, AbstractOperatorModel):
            raise TypeError(
                "Operator UQ benchmarks require native geometry-aware operator models."
            )
        trained, initial, final, duration, losses = train_operator(
            model,
            scenario,
            steps=steps,
            learning_rate=learning_rate,
            trainable=True,
            validation_interval=validation_interval,
            patience=patience,
            minimum_delta=minimum_delta,
        )
        trained_members.append(trained)
        training_steps.append(len(losses))
        training_seconds += float(duration)
        initial_losses.append(float(initial))
        final_losses.append(float(final))
        validation_losses.append(
            float(
                jax.block_until_ready(
                    _loss(
                        trained,
                        scenario.validation.batch,
                        scenario.validation.target,
                    )
                )
            )
        )
    members = tuple(trained_members)
    ensemble = phx.uq.HomogeneousFunctionEnsemble.from_members(
        members,
        source_dim="ensemble_member",
    )

    calibration = scenario.validation
    calibration_target = calibration.target
    if isinstance(calibration_target, phx.nn.operator.OperatorTargetBatch):
        raise TypeError(
            "Operator UQ benchmarks currently require one anonymous array target."
        )
    calibration_prediction = ensemble.predict_operator(
        calibration.batch,
        key=jr.key(91_001),
        field_name="output",
        query_name=calibration.batch.single_query_name(),
    )
    calibration_center = calibration_prediction.mean()
    calibrator = phx.uq.OperatorFunctionalConformal.calibrate(
        calibration_center,
        calibration_target,
        alpha=alpha,
        field_name="output",
    )
    residual_scale = _observation_scale(calibration_center, calibration_target)

    evaluations = (calibration,) + scenario.evaluations
    evaluation_results = tuple(
        _evaluate_operator_uq(
            ensemble,
            members,
            calibrator,
            evaluation,
            key=jr.key(92_000 + index),
            repeats=repeats,
        )
        for index, evaluation in enumerate(evaluations)
    )

    laplace = None
    if fit_projection_laplace and architecture in ("fno", "tfno"):
        laplace = _fit_final_projection_laplace(
            members[0],
            calibration,
            observation_scale=residual_scale,
            num_samples=posterior_samples,
            key=jr.key(93_001),
        )

    return members, OperatorUQBenchmarkResult(
        scenario=scenario.name,
        architecture=selected.name,
        family=selected.family,
        seeds=tuple(int(seed) for seed in seeds),
        ensemble_size=len(members),
        parameter_count_mean=float(
            np.mean([parameter_count(model) for model in members])
        ),
        training_steps=tuple(training_steps),
        training_seconds=training_seconds,
        initial_losses=tuple(initial_losses),
        final_losses=tuple(final_losses),
        validation_losses=tuple(validation_losses),
        calibration_cases=_case_count(calibration.batch.case_shape),
        calibration_radius=float(calibrator.calibrator.radius),
        nominal_coverage=1.0 - float(alpha),
        observation_scale=residual_scale,
        evaluations=evaluation_results,
        laplace=laplace,
    )


def run_operator_uq_suite(
    profiles: tuple[OperatorUQBenchmarkProfile, ...],
    /,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    steps: int = 100,
    learning_rate: float = 1e-3,
    repeats: int = 5,
    alpha: float = 0.1,
    quick: bool = False,
    validation_interval: int = 10,
    patience: int | None = None,
    minimum_delta: float = 0.0,
    fit_projection_laplace: bool = True,
    posterior_samples: int = 32,
    commit_identity: str = "working-tree",
) -> OperatorUQBenchmarkSuite:
    if not profiles:
        raise ValueError("Operator UQ benchmark profiles must be non-empty.")
    scenarios = tuple(profile.scenario for profile in profiles)
    results = []
    for profile in profiles:
        _, result = run_operator_uq_benchmark(
            profile.scenario,
            architecture=profile.architecture,
            seeds=seeds,
            steps=steps,
            learning_rate=learning_rate,
            repeats=repeats,
            alpha=alpha,
            quick=quick,
            validation_interval=validation_interval,
            patience=patience,
            minimum_delta=minimum_delta,
            fit_projection_laplace=fit_projection_laplace,
            posterior_samples=posterior_samples,
        )
        results.append(result)
    return OperatorUQBenchmarkSuite(
        metadata=benchmark_metadata(scenarios, commit_identity=commit_identity),
        calibration_case_checksums=tuple(
            (scenario.name, calibration_case_checksum(scenario)) for scenario in scenarios
        ),
        results=tuple(results),
    )


def save_operator_uq_artifacts(
    directory: str | Path,
    suite: OperatorUQBenchmarkSuite,
    /,
) -> tuple[Path, Path]:
    """Write the separate operator-UQ JSON artifact and evaluation table."""
    if not isinstance(suite, OperatorUQBenchmarkSuite):
        raise TypeError("suite must be an OperatorUQBenchmarkSuite.")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "operator_uq_benchmarks.json"
    parquet_path = root / "operator_uq_benchmarks.parquet"
    json_path.write_text(
        json.dumps(suite.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = []
    for result in suite.results:
        common = {
            "scenario": result.scenario,
            "architecture": result.architecture,
            "family": result.family,
            "seeds": result.seeds,
            "ensemble_size": result.ensemble_size,
            "parameter_count_mean": result.parameter_count_mean,
            "training_seconds": result.training_seconds,
            "initial_losses": result.initial_losses,
            "final_losses": result.final_losses,
            "validation_losses": result.validation_losses,
            "calibration_cases": result.calibration_cases,
            "calibration_radius": result.calibration_radius,
            "nominal_coverage": result.nominal_coverage,
            "observation_scale": result.observation_scale,
        }
        for evaluation in result.evaluations:
            rows.append(common | asdict(evaluation))
    pl.DataFrame(rows).write_parquet(parquet_path)
    return json_path, parquet_path


def calibration_case_checksum(scenario: OperatorBenchmarkScenario, /) -> str:
    if scenario.validation is None:
        raise ValueError("Calibration checksum requires a validation split.")
    digest = hashlib.sha256(f"{scenario.name}:calibration".encode("utf-8"))
    for leaf in jax.tree_util.tree_leaves(scenario.validation.batch):
        if isinstance(leaf, jax.Array):
            _checksum_array(digest, leaf)
    _checksum_array(digest, scenario.validation.target)
    return digest.hexdigest()


def _select_architecture(
    scenario: OperatorBenchmarkScenario,
    architecture: str,
    /,
    *,
    quick: bool,
) -> OperatorArchitecture:
    matches = tuple(
        candidate
        for candidate in compatible_architectures(scenario, quick=quick)
        if candidate.name == architecture
    )
    if len(matches) != 1:
        available = tuple(
            candidate.name
            for candidate in compatible_architectures(scenario, quick=quick)
        )
        raise ValueError(
            f"Architecture {architecture!r} is not uniquely compatible with "
            f"scenario {scenario.name!r}; available={available!r}."
        )
    return matches[0]


def _evaluate_operator_uq(
    ensemble: phx.uq.HomogeneousFunctionEnsemble,
    members: tuple[AbstractOperatorModel, ...],
    calibrator: phx.uq.OperatorFunctionalConformal,
    evaluation: OperatorBenchmarkEvaluation,
    /,
    *,
    key,
    repeats: int,
) -> OperatorUQEvaluationResult:
    target = evaluation.target
    if isinstance(target, phx.nn.operator.OperatorTargetBatch):
        raise TypeError(
            "Operator UQ benchmarks currently require one anonymous array target."
        )
    compile_started = time.perf_counter()
    prediction = _predict_ensemble_evaluation(ensemble, members, evaluation, key=key)
    jax.block_until_ready(prediction.predictive.samples.data)
    compile_seconds = time.perf_counter() - compile_started
    started = time.perf_counter()
    for repeat in range(int(repeats)):
        prediction = _predict_ensemble_evaluation(
            ensemble,
            members,
            evaluation,
            key=jr.fold_in(key, repeat),
        )
        jax.block_until_ready(prediction.predictive.samples.data)
    inference_seconds = (time.perf_counter() - started) / float(repeats)

    center = prediction.mean()
    center_field = center.field("output")
    standard_deviation = prediction.std()
    if not isinstance(standard_deviation, phx.nn.operator.OperatorPrediction):
        raise TypeError(
            "Benchmark epistemic reduction retained an unexpected sample axis."
        )
    interval = calibrator.interval(center)
    relative_l2 = phx.nn.operator.operator_l2_loss(
        center_field.values,
        target,
        evaluation.batch.require_single_query(),
        relative=True,
    )
    if evaluation.batch.require_single_query().axes:
        h1 = float(
            phx.nn.operator.operator_h1_loss(
                center_field.values,
                target,
                evaluation.batch.require_single_query(),
                relative=True,
            )
        )
    else:
        h1 = None
    crps = phx.uq.operator_ensemble_crps(prediction, target)
    field_energy_score = phx.uq.operator_energy_score(prediction, target)
    pointwise = phx.uq.operator_interval_coverage(
        interval,
        target,
        field_name="output",
        mode="pointwise",
    )
    simultaneous_per_case = phx.uq.operator_interval_coverage(
        interval,
        target,
        field_name="output",
        mode="simultaneous",
        reduction="none",
    )
    simultaneous = jnp.mean(simultaneous_per_case)
    width = phx.uq.operator_interval_width(interval, field_name="output")
    standard_deviation_field = standard_deviation.field("output")
    spread = _physical_mean(
        standard_deviation_field.values,
        evaluation.batch,
        center_field.spec,
    )
    valid = prediction.predictive.valid
    total_draw_count = len(members)
    valid_draw_count = (
        total_draw_count if valid is None else int(np.sum(np.asarray(valid.data)))
    )
    confidence_lower = None
    confidence_upper = None
    compatible = None
    if evaluation.split == "test" and evaluation.shift == "in_distribution":
        covered = int(jnp.sum(simultaneous_per_case))
        total = int(jnp.size(simultaneous_per_case))
        confidence_lower, confidence_upper = _wilson_interval(covered, total)
        nominal = interval.nominal_coverage
        compatible = confidence_lower <= nominal <= confidence_upper
    return OperatorUQEvaluationResult(
        name=evaluation.name,
        split=evaluation.split,
        shift=evaluation.shift,
        rollout_steps=evaluation.rollout_steps,
        num_cases=_case_count(evaluation.batch.case_shape),
        query_shape=evaluation.batch.require_single_query().sample_shape,
        relative_l2=float(jax.block_until_ready(relative_l2)),
        h1=h1,
        epistemic_standard_deviation=float(jax.block_until_ready(spread)),
        crps=float(jax.block_until_ready(crps)),
        energy_score=float(jax.block_until_ready(field_energy_score)),
        pointwise_coverage=float(jax.block_until_ready(pointwise)),
        simultaneous_coverage=float(jax.block_until_ready(simultaneous)),
        interval_width=float(jax.block_until_ready(width)),
        valid_draw_count=valid_draw_count,
        total_draw_count=total_draw_count,
        compile_seconds=compile_seconds,
        inference_seconds=inference_seconds,
        peak_memory_bytes=_memory_bytes(),
        coverage_confidence_lower=confidence_lower,
        coverage_confidence_upper=confidence_upper,
        nominal_coverage_compatible=compatible,
    )


def _predict_ensemble_evaluation(
    ensemble: phx.uq.HomogeneousFunctionEnsemble,
    members: tuple[AbstractOperatorModel, ...],
    evaluation: OperatorBenchmarkEvaluation,
    /,
    *,
    key,
) -> phx.uq.OperatorPredictiveField:
    if evaluation.rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if evaluation.rollout_steps == 1:
        return ensemble.predict_operator(
            evaluation.batch,
            key=key,
            field_name="output",
            query_name=evaluation.batch.single_query_name(),
        )
    if evaluation.rollout_source_key is None:
        raise ValueError("Rollout evaluations require rollout_source_key.")
    predictions = []
    final_batch = evaluation.batch
    for member, member_key in zip(
        members,
        jr.split(key, len(members)),
        strict=True,
    ):
        current_batch = evaluation.batch
        member_prediction = member.predict(current_batch, key=member_key)
        for step in range(1, evaluation.rollout_steps):
            current_batch = _with_source_values(
                current_batch,
                evaluation.rollout_source_key,
                member_prediction.field("output").values,
            )
            member_prediction = member.predict(
                current_batch,
                key=jr.fold_in(member_key, step),
            )
        predictions.append(member_prediction.field("output").values)
        final_batch = current_batch
    return phx.uq.operator_predictive_from_samples(
        jnp.stack(tuple(predictions), axis=0),
        final_batch,
        members[0].operator_output_specs["output"],
        field_name="output",
        query_name=final_batch.single_query_name(),
        sample_axes=(phx.uq.SampleAxis("ensemble_member", "epistemic"),),
    )


def _fit_final_projection_laplace(
    model: AbstractOperatorModel,
    calibration: OperatorBenchmarkEvaluation,
    /,
    *,
    observation_scale: float,
    num_samples: int,
    key,
) -> OperatorUQLaplaceResult:
    subspace = phx.nn.parameters.ParameterSubspace.from_subtree_paths(
        model, (".projection",)
    )
    initial = subspace.initial
    prior_scale = 1.0
    initial_leaves = tuple(jax.tree_util.tree_leaves(initial))

    def log_prior(selected):
        selected_leaves = tuple(jax.tree_util.tree_leaves(selected))
        terms = tuple(
            -0.5 * jnp.sum(((value - center) / prior_scale) ** 2)
            for value, center in zip(selected_leaves, initial_leaves, strict=True)
        )
        return sum(terms, jnp.asarray(0.0))

    def predict(selected):
        reconstructed = subspace.reconstruct(selected)
        return reconstructed.predict(calibration.batch)

    term = phx.uq.FixedOperatorObservationLikelihood(
        predict,
        calibration.batch,
        calibration.target,
        phx.uq.GaussianLikelihood(observation_scale),
        output_spec=model.operator_output_specs["output"],
        field_name="output",
        query_name=calibration.batch.single_query_name(),
    )
    parameter_space = phx.uq.ParameterSpace(initial, log_prior=log_prior)
    problem = phx.uq.PosteriorProblem.from_terms(
        parameter_space,
        (term,),
        predict=lambda selected: phx.uq.operator_prediction_field(
            predict(selected),
            field_name="output",
        ),
        gauss_newton_residual=lambda selected: term.standardized_residual(selected),
    )
    map_result = phx.uq.find_map(
        problem,
        max_steps=1_000,
        gradient_tolerance=1e-4,
    )
    laplace = phx.uq.fit_laplace(
        problem,
        map_result.position,
        max_dimension=256,
        stationarity_tolerance=1e-3,
    )
    if not isinstance(laplace, phx.uq.LaplaceResult):
        raise TypeError("Operator UQ benchmark requires a dense Laplace result.")
    predictive = laplace.predict(
        key,
        num_samples=num_samples,
        sample_dim="posterior_draw",
    )
    if isinstance(predictive, Mapping):
        raise TypeError("Operator UQ benchmark requires a single predictive field.")
    operator_predictive = phx.uq.OperatorPredictiveField.from_predictive(
        predictive,
        calibration.batch,
        model.operator_output_specs["output"],
        field_name="output",
        query_name=calibration.batch.single_query_name(),
    )
    output_variance = operator_predictive.epistemic_variance()
    if not isinstance(output_variance, phx.nn.operator.OperatorPrediction):
        raise TypeError("Laplace output variance retained an unexpected sample axis.")
    geometry_preserved = (
        operator_predictive.query.sample_shape
        == calibration.batch.require_single_query().sample_shape
        and operator_predictive.case_axes == calibration.batch.case_axes
        and operator_predictive.predictive.samples.data.shape[0] == num_samples
    )
    return OperatorUQLaplaceResult(
        parameter_dimension=laplace.dimension,
        parameter_variance_mean=float(jnp.mean(jnp.diag(laplace.covariance))),
        output_variance_mean=float(
            _physical_mean(
                output_variance.field("output").values,
                calibration.batch,
                model.operator_output_specs["output"],
            )
        ),
        posterior_sample_count=num_samples,
        geometry_preserved=geometry_preserved,
    )


def _physical_mean(values, batch, output_spec) -> jax.Array:
    weights = batch.require_single_query().weights(case_shape=batch.case_shape)
    mask = batch.require_single_query().mask_array(case_shape=batch.case_shape)
    if output_spec.channels != "scalar":
        weights = jnp.broadcast_to(
            weights[..., None],
            weights.shape + output_spec.channel_shape,
        )
        mask = jnp.broadcast_to(
            mask[..., None],
            mask.shape + output_spec.channel_shape,
        )
    effective = jnp.where(mask, weights, 0.0)
    return jnp.sum(jnp.where(mask, jnp.asarray(values) * effective, 0.0)) / jnp.sum(
        effective
    )


def _observation_scale(
    center: phx.nn.operator.OperatorPrediction,
    target,
    /,
) -> float:
    field = center.field("output")
    query = center.query_geometry(field.query_name)
    mask = query.mask_array(case_shape=center.case_shape)
    spec = field.spec
    if spec.channels != "scalar":
        mask = jnp.broadcast_to(
            mask[..., None],
            mask.shape + spec.channel_shape,
        )
    residual = jnp.asarray(target) - jnp.asarray(field.values)
    observed = residual[mask]
    return max(float(jnp.std(observed)), 1e-3)


def _wilson_interval(successes: int, total: int, /) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError(
            "Wilson interval requires 0 <= successes <= total and total > 0."
        )
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z**2 / total
    center = (probability + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1.0 - probability) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _memory_bytes() -> int | None:
    statistics = jax.devices()[0].memory_stats()
    if statistics is None:
        return None
    for name in ("peak_bytes_in_use", "bytes_in_use"):
        if name in statistics:
            return int(statistics[name])
    return None


def _case_count(case_shape: tuple[int, ...], /) -> int:
    count = 1
    for size in case_shape:
        count *= int(size)
    return count


def _checksum_array(digest, value) -> None:
    array = np.asarray(jax.device_get(value))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))


__all__ = [
    "OperatorUQBenchmarkProfile",
    "OperatorUQBenchmarkResult",
    "OperatorUQBenchmarkSuite",
    "OperatorUQEvaluationResult",
    "OperatorUQLaplaceResult",
    "calibration_case_checksum",
    "run_operator_uq_benchmark",
    "run_operator_uq_suite",
    "save_operator_uq_artifacts",
]
