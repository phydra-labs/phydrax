from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
import polars as pl

from .models import compatible_architectures
from .runner import OperatorBenchmarkResult, run_operator_benchmark
from .scenarios import (
    OperatorBenchmarkScenario,
    OperatorSymmetrySpec,
    ReferenceSolverEvidence,
)


@dataclass(frozen=True)
class BenchmarkRunMetadata:
    commit_identity: str
    phydrax_version: str
    jax_version: str
    python_version: str
    platform: str
    device: str
    default_float_dtype: str
    scenario_checksums: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class OperatorBenchmarkAggregate:
    scenario: str
    architecture: str
    family: str
    evaluation: str
    split: str
    shift: str
    seeds: tuple[int, ...]
    parameter_count_mean: float
    relative_l2_mean: float
    relative_l2_std: float
    absolute_l2_mean: float
    h1_mean: float | None
    spectral_mean: float | None
    conservation_error_mean: float
    maximum_absolute_error_mean: float
    compile_seconds_mean: float
    inference_seconds_mean: float
    training_seconds_mean: float
    peak_memory_bytes_mean: float | None = None
    size_scale: float = 1.0
    convergence_rate: float = 0.0


@dataclass(frozen=True)
class OperatorBenchmarkMatrixResult:
    metadata: BenchmarkRunMetadata
    results: tuple[OperatorBenchmarkResult, ...]
    aggregates: tuple[OperatorBenchmarkAggregate, ...]

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class OperatorBenchmarkThreshold:
    scenario: str
    architecture: str
    evaluation: str
    maximum_relative_l2: float
    maximum_inference_seconds: float | None = None


class OperatorBenchmarkRegressionError(RuntimeError):
    pass


def _update_checksum(digest, value) -> None:
    array = np.asarray(jax.device_get(value))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))


def _update_tree_checksum(digest, value) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if isinstance(leaf, (jax.Array, np.ndarray)):
            _update_checksum(digest, leaf)


def _reference_evidence_contract(
    evidence: ReferenceSolverEvidence | None,
) -> dict[str, object] | None:
    if evidence is None:
        return None
    return {
        "method": evidence.method,
        "verification": evidence.verification,
        "resolutions": evidence.resolutions,
        "tolerance": evidence.tolerance,
        "passed": evidence.passed,
    }


def _symmetry_contract(
    symmetry: OperatorSymmetrySpec | None,
) -> dict[str, object] | None:
    if symmetry is None:
        return None
    contract = asdict(symmetry)
    contract.pop("reference_defects")
    contract["reference_verdicts"] = tuple(
        (index, float(defect) <= float(symmetry.reference_tolerance))
        for index, defect in symmetry.reference_defects
    )
    return contract


def scenario_checksum(scenario: OperatorBenchmarkScenario, /) -> str:
    digest = hashlib.sha256(scenario.name.encode("utf-8"))
    contract = {
        "case_ids": scenario.case_ids,
        "metadata": scenario.metadata,
        "regimes": scenario.regimes,
        "domain_support": {
            "key": scenario.domain_support_key,
            "kind": scenario.domain_support_kind,
            "threshold": scenario.domain_support_threshold,
        },
        "conservation_source_key": scenario.conservation_source_key,
        "symmetry": _symmetry_contract(scenario.symmetry),
        "task_fingerprint": None if scenario.task is None else scenario.task.fingerprint,
        "ladder": scenario.ladder,
        "difficulty": scenario.difficulty,
        "provenance": (
            None if scenario.provenance is None else asdict(scenario.provenance)
        ),
        "dimensional_parameters": tuple(
            asdict(parameter) for parameter in scenario.dimensional_parameters
        ),
        "nondimensional_parameters": tuple(
            asdict(parameter) for parameter in scenario.nondimensional_parameters
        ),
        "reference_evidence": _reference_evidence_contract(scenario.reference_evidence),
        "evaluations": tuple(
            {
                "name": evaluation.name,
                "split": evaluation.split,
                "shift": evaluation.shift,
                "rollout_steps": evaluation.rollout_steps,
                "rollout_source_key": evaluation.rollout_source_key,
                "case_ids": evaluation.case_ids,
            }
            for evaluation in scenario.evaluations
        ),
    }
    digest.update(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for leaf in jax.tree_util.tree_leaves(scenario.train_batch):
        if isinstance(leaf, jax.Array):
            _update_checksum(digest, leaf)
    _update_tree_checksum(digest, scenario.train_target)
    if scenario.validation is not None:
        digest.update(
            json.dumps(
                {
                    "name": scenario.validation.name,
                    "split": scenario.validation.split,
                    "shift": scenario.validation.shift,
                    "case_ids": scenario.validation.case_ids,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for leaf in jax.tree_util.tree_leaves(scenario.validation.batch):
            if isinstance(leaf, jax.Array):
                _update_checksum(digest, leaf)
        _update_tree_checksum(digest, scenario.validation.target)
    for evaluation in scenario.evaluations:
        digest.update(evaluation.name.encode("utf-8"))
        for leaf in jax.tree_util.tree_leaves(evaluation.batch):
            if isinstance(leaf, jax.Array):
                _update_checksum(digest, leaf)
        _update_tree_checksum(digest, evaluation.target)
    return digest.hexdigest()


def benchmark_metadata(
    scenarios: tuple[OperatorBenchmarkScenario, ...],
    /,
    *,
    commit_identity: str,
) -> BenchmarkRunMetadata:
    device = jax.devices()[0]
    return BenchmarkRunMetadata(
        commit_identity=str(commit_identity),
        phydrax_version=version("phydrax"),
        jax_version=version("jax"),
        python_version=platform.python_version(),
        platform=sys.platform,
        device=f"{device.platform}:{device.device_kind}",
        default_float_dtype=str(jax.numpy.asarray(0.0).dtype),
        scenario_checksums=tuple(
            (scenario.name, scenario_checksum(scenario)) for scenario in scenarios
        ),
    )


def aggregate_benchmark_results(
    results: tuple[OperatorBenchmarkResult, ...],
    /,
) -> tuple[OperatorBenchmarkAggregate, ...]:
    groups = {}
    for result in results:
        for evaluation in result.evaluations:
            key = (
                result.scenario,
                result.architecture,
                result.family,
                result.size_scale,
                evaluation.name,
                evaluation.split,
                evaluation.shift,
            )
            groups.setdefault(key, []).append((result, evaluation))

    aggregates = []
    for key in sorted(groups):
        rows = groups[key]
        scenario, architecture, family, size_scale, evaluation, split, shift = key
        relative = np.asarray([row.relative_l2 for _, row in rows])
        h1_values = [row.h1 for _, row in rows if row.h1 is not None]
        spectral_values = [row.spectral for _, row in rows if row.spectral is not None]
        memory_values = [
            row.peak_memory_bytes for _, row in rows if row.peak_memory_bytes is not None
        ]
        aggregates.append(
            OperatorBenchmarkAggregate(
                scenario=scenario,
                architecture=architecture,
                family=family,
                evaluation=evaluation,
                split=split,
                shift=shift,
                size_scale=float(size_scale),
                seeds=tuple(result.seed for result, _ in rows),
                parameter_count_mean=float(
                    np.mean([result.parameter_count for result, _ in rows])
                ),
                relative_l2_mean=float(np.mean(relative)),
                relative_l2_std=float(np.std(relative)),
                absolute_l2_mean=float(np.mean([row.absolute_l2 for _, row in rows])),
                h1_mean=None if not h1_values else float(np.mean(h1_values)),
                spectral_mean=(
                    None if not spectral_values else float(np.mean(spectral_values))
                ),
                conservation_error_mean=float(
                    np.mean([row.conservation_error for _, row in rows])
                ),
                maximum_absolute_error_mean=float(
                    np.mean([row.maximum_absolute_error for _, row in rows])
                ),
                compile_seconds_mean=float(
                    np.mean([row.compile_seconds for _, row in rows])
                ),
                inference_seconds_mean=float(
                    np.mean([row.inference_seconds for _, row in rows])
                ),
                training_seconds_mean=float(
                    np.mean([result.training_seconds for result, _ in rows])
                ),
                peak_memory_bytes_mean=(
                    None if not memory_values else float(np.mean(memory_values))
                ),
                convergence_rate=float(np.mean([result.converged for result, _ in rows])),
            )
        )
    return tuple(aggregates)


def run_benchmark_matrix(
    scenarios: tuple[OperatorBenchmarkScenario, ...],
    /,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    architecture_names: tuple[str, ...] | None = None,
    steps: int = 100,
    learning_rate: float = 1e-3,
    repeats: int = 10,
    validation_interval: int = 10,
    patience: int | None = None,
    minimum_delta: float = 0.0,
    quick: bool = False,
    commit_identity: str = "working-tree",
) -> OperatorBenchmarkMatrixResult:
    if not seeds:
        raise ValueError("At least one benchmark seed is required.")
    selected = None if architecture_names is None else set(architecture_names)
    results = []
    for scenario in scenarios:
        architectures = compatible_architectures(scenario, quick=quick)
        if selected is not None:
            architectures = tuple(
                architecture
                for architecture in architectures
                if architecture.name in selected
            )
        if not architectures:
            raise ValueError(f"No compatible architectures for {scenario.name!r}.")
        for architecture in architectures:
            for seed in seeds:
                model = architecture.build(scenario, seed)
                _, result = run_operator_benchmark(
                    model,
                    scenario,
                    steps=steps,
                    learning_rate=learning_rate,
                    repeats=repeats,
                    architecture=architecture.name,
                    family=architecture.family,
                    architecture_configuration=architecture.configuration(scenario),
                    seed=seed,
                    trainable=architecture.trainable,
                    validation_interval=validation_interval,
                    patience=patience,
                    minimum_delta=minimum_delta,
                )
                results.append(result)
    result_tuple = tuple(results)
    return OperatorBenchmarkMatrixResult(
        metadata=benchmark_metadata(
            scenarios,
            commit_identity=commit_identity,
        ),
        results=result_tuple,
        aggregates=aggregate_benchmark_results(result_tuple),
    )


def assert_benchmark_thresholds(
    matrix: OperatorBenchmarkMatrixResult,
    thresholds: tuple[OperatorBenchmarkThreshold, ...],
    /,
) -> None:
    lookup = {
        (row.scenario, row.architecture, row.evaluation): row for row in matrix.aggregates
    }
    violations = []
    for threshold in thresholds:
        key = (
            threshold.scenario,
            threshold.architecture,
            threshold.evaluation,
        )
        if key not in lookup:
            violations.append(f"missing aggregate {key}")
            continue
        aggregate = lookup[key]
        if aggregate.relative_l2_mean > threshold.maximum_relative_l2:
            violations.append(
                f"{key} relative_l2={aggregate.relative_l2_mean:.6g} exceeds "
                f"{threshold.maximum_relative_l2:.6g}"
            )
        if (
            threshold.maximum_inference_seconds is not None
            and aggregate.inference_seconds_mean > threshold.maximum_inference_seconds
        ):
            violations.append(
                f"{key} inference={aggregate.inference_seconds_mean:.6g}s exceeds "
                f"{threshold.maximum_inference_seconds:.6g}s"
            )
    if violations:
        raise OperatorBenchmarkRegressionError("; ".join(violations))


def _aggregate_rows(matrix: OperatorBenchmarkMatrixResult):
    return [asdict(row) for row in matrix.aggregates]


def save_benchmark_artifacts(
    directory: str | Path,
    matrix: OperatorBenchmarkMatrixResult,
    /,
) -> tuple[Path, Path]:
    """Write canonical JSON metadata/results and a columnar aggregate table."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "operator_benchmarks.json"
    parquet_path = root / "operator_benchmarks.parquet"
    json_path.write_text(
        json.dumps(matrix.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pl.DataFrame(_aggregate_rows(matrix)).write_parquet(parquet_path)
    return json_path, parquet_path


__all__ = [
    "BenchmarkRunMetadata",
    "OperatorBenchmarkAggregate",
    "OperatorBenchmarkMatrixResult",
    "OperatorBenchmarkRegressionError",
    "OperatorBenchmarkThreshold",
    "aggregate_benchmark_results",
    "assert_benchmark_thresholds",
    "benchmark_metadata",
    "run_benchmark_matrix",
    "save_benchmark_artifacts",
    "scenario_checksum",
]
