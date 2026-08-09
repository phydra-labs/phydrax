#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic end-to-end benchmarks for the native :mod:`phydrax.ml` API."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from phydrax.kernels import SquaredExponentialKernel
from phydrax.ml import fit, ML_SUCCESS
from phydrax.ml.clustering import KMeans
from phydrax.ml.decomposition import PCA
from phydrax.ml.kernel_methods import KernelRidgeRecipe
from phydrax.ml.linear import RidgeRecipe
from phydrax.ml.tree import DecisionTreeRegressor


Operation = Callable[..., tuple[Any, Any, Any]]


@dataclass(frozen=True)
class BenchmarkCase:
    """One fully prepared fit-and-inference workload."""

    name: str
    family: str
    scale: str
    configuration: dict[str, Any]
    operation: Operation
    arguments: tuple[Any, ...]
    execution_mode: str


_SCALE_CONFIGURATION: dict[str, dict[str, dict[str, int]]] = {
    "small": {
        "direct_linear": {"training_samples": 256, "features": 16, "query_samples": 128},
        "spectral": {
            "training_samples": 256,
            "features": 24,
            "components": 8,
            "query_samples": 128,
        },
        "iterative_clustering": {
            "training_samples": 512,
            "features": 8,
            "clusters": 6,
            "query_samples": 256,
        },
        "kernel": {"training_samples": 96, "features": 6, "query_samples": 96},
        "fixed_capacity_tree": {
            "training_samples": 256,
            "features": 6,
            "query_samples": 128,
            "max_depth": 3,
            "max_nodes": 15,
        },
    },
    "medium": {
        "direct_linear": {"training_samples": 1024, "features": 32, "query_samples": 256},
        "spectral": {
            "training_samples": 768,
            "features": 48,
            "components": 12,
            "query_samples": 256,
        },
        "iterative_clustering": {
            "training_samples": 2048,
            "features": 12,
            "clusters": 10,
            "query_samples": 512,
        },
        "kernel": {"training_samples": 192, "features": 8, "query_samples": 128},
        "fixed_capacity_tree": {
            "training_samples": 768,
            "features": 8,
            "query_samples": 256,
            "max_depth": 4,
            "max_nodes": 31,
        },
    },
}


def _regular_matrix(
    sample_count: int, feature_count: int, *, offset: float = 0.0
) -> jax.Array:
    """Return a regular, full-rank analytic design without host-side randomness."""
    row = (jnp.arange(sample_count, dtype=jnp.float32) + 1.0 + offset)[:, None]
    column = (jnp.arange(feature_count, dtype=jnp.float32) + 1.0)[None, :]
    phase = row / jnp.asarray(sample_count + 1.0 + offset, dtype=jnp.float32)
    return jnp.sin(2.0 * jnp.pi * phase * column) + 0.35 * jnp.cos(
        jnp.pi * phase * (column + 1.0)
    )


def _scaled_count(value: int, factor: float, minimum: int) -> int:
    return max(minimum, int(round(value * factor)))


def _case_seed(base_seed: int, name: str) -> int:
    name_bits = int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
    )
    return (base_seed + name_bits) % (2**31 - 1)


def _direct_linear_case(scale: str, factor: float, seed: int) -> BenchmarkCase:
    base = _SCALE_CONFIGURATION[scale]["direct_linear"]
    sample_count = _scaled_count(base["training_samples"], factor, 64)
    query_count = _scaled_count(base["query_samples"], factor, 32)
    feature_count = base["features"]
    features = _regular_matrix(sample_count, feature_count)
    coefficients = jnp.where(
        jnp.arange(feature_count) % 2 == 0,
        1.0,
        -1.0,
    ) / jnp.sqrt(jnp.asarray(feature_count, dtype=jnp.float32))
    targets = features @ coefficients + 0.1 * jnp.sin(features[:, 0] * features[:, 1])
    query = _regular_matrix(query_count, feature_count, offset=0.375)
    recipe = RidgeRecipe(0.05, fit_intercept=True)
    key = jax.random.PRNGKey(_case_seed(seed, f"direct-linear/{scale}"))

    def operation(x, y, points, fit_key):
        result = fit(recipe, x, y, key=fit_key)
        prediction = result.as_trainable()(points)
        return prediction, result.valid, result.status

    configuration = {
        "recipe": "RidgeRecipe",
        "alpha": 0.05,
        "fit_intercept": True,
        "training_samples": sample_count,
        "features": feature_count,
        "query_samples": query_count,
        "target_shape": [sample_count],
        "key_seed": _case_seed(seed, f"direct-linear/{scale}"),
    }
    return BenchmarkCase(
        name=f"direct-linear-ridge/{scale}",
        family="direct_linear",
        scale=scale,
        configuration=configuration,
        operation=jax.jit(operation),
        arguments=(features, targets, query, key),
        execution_mode="whole-workflow-jit",
    )


def _spectral_case(scale: str, factor: float, seed: int) -> BenchmarkCase:
    base = _SCALE_CONFIGURATION[scale]["spectral"]
    sample_count = _scaled_count(base["training_samples"], factor, 64)
    query_count = _scaled_count(base["query_samples"], factor, 32)
    feature_count = base["features"]
    component_count = min(base["components"], feature_count, sample_count - 1)
    features = _regular_matrix(sample_count, feature_count)
    features = features + 0.05 * jnp.sin(features * jnp.arange(1, feature_count + 1))
    query = _regular_matrix(query_count, feature_count, offset=0.625)
    recipe = PCA(component_count, differentiate="projector")
    key_seed = _case_seed(seed, f"spectral-pca/{scale}")
    key = jax.random.PRNGKey(key_seed)

    def operation(x, points, fit_key):
        result = fit(recipe, x, key=fit_key)
        scores = result.as_trainable()(points)
        return scores, result.valid, result.status

    configuration = {
        "recipe": "PCA",
        "training_samples": sample_count,
        "features": feature_count,
        "components": component_count,
        "query_samples": query_count,
        "differentiate": "projector",
        "key_seed": key_seed,
    }
    return BenchmarkCase(
        name=f"spectral-pca/{scale}",
        family="spectral",
        scale=scale,
        configuration=configuration,
        operation=jax.jit(operation),
        arguments=(features, query, key),
        execution_mode="whole-workflow-jit",
    )


def _cluster_data(
    sample_count: int, feature_count: int, cluster_count: int, *, offset: float
) -> jax.Array:
    labels = jnp.arange(sample_count) % cluster_count
    angle = 2.0 * jnp.pi * jnp.arange(cluster_count, dtype=jnp.float32) / cluster_count
    coordinate = jnp.arange(feature_count, dtype=jnp.float32) + 1.0
    centers = 0.4 * jnp.cos(angle[:, None] * coordinate[None, :])
    centers = centers.at[:, 0].set(4.0 * jnp.cos(angle))
    centers = centers.at[:, 1].set(4.0 * jnp.sin(angle))
    row = (jnp.arange(sample_count, dtype=jnp.float32) + 1.0 + offset)[:, None]
    perturbation = 0.03 * jnp.sin(row * coordinate[None, :] * 0.173)
    return centers[labels] + perturbation


def _iterative_clustering_case(scale: str, factor: float, seed: int) -> BenchmarkCase:
    base = _SCALE_CONFIGURATION[scale]["iterative_clustering"]
    cluster_count = base["clusters"]
    sample_count = _scaled_count(base["training_samples"], factor, 8 * cluster_count)
    query_count = _scaled_count(base["query_samples"], factor, 4 * cluster_count)
    feature_count = base["features"]
    features = _cluster_data(sample_count, feature_count, cluster_count, offset=0.0)
    query = _cluster_data(query_count, feature_count, cluster_count, offset=0.5)
    max_iterations = 16
    recipe = KMeans(
        cluster_count,
        max_iterations=max_iterations,
        tolerance=1e-5,
        initialization="first",
        empty_policy="reseed",
    )
    key_seed = _case_seed(seed, f"iterative-kmeans/{scale}")
    key = jax.random.PRNGKey(key_seed)

    def operation(x, points, fit_key):
        result = fit(recipe, x, key=fit_key)
        labels = result.as_trainable()(points)
        return labels, result.valid, result.status

    configuration = {
        "recipe": "KMeans",
        "training_samples": sample_count,
        "features": feature_count,
        "clusters": cluster_count,
        "query_samples": query_count,
        "max_iterations": max_iterations,
        "tolerance": 1e-5,
        "initialization": "first",
        "key_seed": key_seed,
    }
    return BenchmarkCase(
        name=f"iterative-kmeans/{scale}",
        family="iterative_clustering",
        scale=scale,
        configuration=configuration,
        operation=jax.jit(operation),
        arguments=(features, query, key),
        execution_mode="whole-workflow-jit",
    )


def _kernel_case(scale: str, factor: float, seed: int) -> BenchmarkCase:
    base = _SCALE_CONFIGURATION[scale]["kernel"]
    sample_count = _scaled_count(base["training_samples"], factor, 48)
    query_count = _scaled_count(base["query_samples"], factor, 32)
    feature_count = base["features"]
    features = 0.7 * _regular_matrix(sample_count, feature_count)
    targets = (
        jnp.sin(1.3 * features[:, 0])
        + 0.4 * jnp.cos(features[:, 1])
        - 0.2 * features[:, 2] * features[:, 3]
    )
    query = 0.7 * _regular_matrix(query_count, feature_count, offset=0.25)
    recipe = KernelRidgeRecipe(
        SquaredExponentialKernel(length_scale=2.0),
        alpha=0.05,
        fit_intercept=True,
    )
    key_seed = _case_seed(seed, f"kernel-ridge/{scale}")
    key = jax.random.PRNGKey(key_seed)

    def operation(x, y, points, fit_key):
        result = fit(recipe, x, y, key=fit_key)
        prediction = result.as_trainable()(points)
        return prediction, result.valid, result.status

    configuration = {
        "recipe": "KernelRidgeRecipe",
        "kernel": "SquaredExponentialKernel",
        "length_scale": 2.0,
        "alpha": 0.05,
        "fit_intercept": True,
        "training_samples": sample_count,
        "features": feature_count,
        "query_samples": query_count,
        "target_shape": [sample_count],
        "key_seed": key_seed,
    }
    return BenchmarkCase(
        name=f"kernel-ridge/{scale}",
        family="kernel",
        scale=scale,
        configuration=configuration,
        operation=jax.jit(operation),
        arguments=(features, targets, query, key),
        execution_mode="whole-workflow-jit",
    )


def _fixed_capacity_tree_case(scale: str, factor: float, seed: int) -> BenchmarkCase:
    base = _SCALE_CONFIGURATION[scale]["fixed_capacity_tree"]
    sample_count = _scaled_count(base["training_samples"], factor, 64)
    query_count = _scaled_count(base["query_samples"], factor, 32)
    feature_count = base["features"]
    features = _regular_matrix(sample_count, feature_count)
    targets = (
        jnp.sin(2.5 * features[:, 0]) + 0.35 * features[:, 1] - 0.15 * features[:, 2] ** 2
    )
    query = _regular_matrix(query_count, feature_count, offset=0.875)
    recipe = DecisionTreeRegressor(
        max_depth=base["max_depth"],
        max_nodes=base["max_nodes"],
        min_samples_leaf=max(2, sample_count // 128),
        split_search="histogram",
        max_bins=8,
    )
    key_seed = _case_seed(seed, f"fixed-capacity-tree/{scale}")
    key = jax.random.PRNGKey(key_seed)
    predict = jax.jit(lambda model, points: model(points))

    def operation(x, y, points, fit_key):
        result = fit(recipe, x, y, key=fit_key)
        prediction = predict(result.as_trainable(), points)
        return prediction, result.valid, result.status

    configuration = {
        "recipe": "DecisionTreeRegressor",
        "training_samples": sample_count,
        "features": feature_count,
        "query_samples": query_count,
        "target_shape": [sample_count],
        "max_depth": base["max_depth"],
        "max_nodes": base["max_nodes"],
        "min_samples_leaf": max(2, sample_count // 128),
        "split_search": "histogram",
        "max_bins": 8,
        "key_seed": key_seed,
    }
    return BenchmarkCase(
        name=f"fixed-capacity-decision-tree/{scale}",
        family="fixed_capacity_tree",
        scale=scale,
        configuration=configuration,
        operation=operation,
        arguments=(features, targets, query, key),
        execution_mode="eager-fit-jitted-inference",
    )


_CASE_BUILDERS = (
    _direct_linear_case,
    _spectral_case,
    _iterative_clustering_case,
    _kernel_case,
    _fixed_capacity_tree_case,
)


def _block_tree(value: Any) -> Any:
    """Synchronize every device-backed leaf and return the original pytree."""
    jax.block_until_ready(value)
    return value


def _timed_call(
    operation: Operation, arguments: tuple[Any, ...]
) -> tuple[tuple[Any, Any, Any], float]:
    _block_tree(arguments)
    start_ns = time.perf_counter_ns()
    result = operation(*arguments)
    _block_tree(result)
    elapsed_seconds = (time.perf_counter_ns() - start_ns) / 1_000_000_000.0
    return result, elapsed_seconds


def _output_evidence(prediction: Any) -> dict[str, Any]:
    host = np.asarray(jax.device_get(prediction))
    if host.size == 0:
        raise RuntimeError("benchmark inference produced an empty output")
    if not bool(np.all(np.isfinite(host))):
        raise RuntimeError("benchmark inference produced nonfinite output")
    contiguous = np.ascontiguousarray(host)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(contiguous.tobytes(order="C"))
    values = contiguous.astype(np.float64, copy=False)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "elements": int(contiguous.size),
        "checksum_sha256": digest.hexdigest(),
        "sum": float(np.sum(values, dtype=np.float64)),
        "l2_norm": float(np.linalg.norm(values.ravel())),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "standard_deviation": float(np.std(values, dtype=np.float64)),
        "nonzero_elements": int(np.count_nonzero(contiguous)),
        "finite": True,
    }


def _validate_result(
    case_name: str, result: tuple[Any, Any, Any]
) -> tuple[dict[str, Any], Any, Any]:
    prediction, valid, status = result
    valid_host = np.asarray(jax.device_get(valid), dtype=bool)
    status_host = np.asarray(jax.device_get(status), dtype=np.int64)
    if not bool(np.all(valid_host)):
        raise RuntimeError(
            f"{case_name}: fit was invalid (valid={valid_host.tolist()}, status={status_host.tolist()})"
        )
    if not bool(np.all(status_host == ML_SUCCESS)):
        raise RuntimeError(
            f"{case_name}: fit returned non-success status {status_host.tolist()}"
        )
    evidence = _output_evidence(prediction)
    valid_payload: Any = (
        bool(valid_host.item()) if valid_host.ndim == 0 else valid_host.tolist()
    )
    status_payload: Any = (
        int(status_host.item()) if status_host.ndim == 0 else status_host.tolist()
    )
    return evidence, valid_payload, status_payload


def _run_case(case: BenchmarkCase, *, warmup: int, repeat: int) -> dict[str, Any]:
    warmup_seconds: list[float] = []
    for _ in range(warmup):
        result, elapsed = _timed_call(case.operation, case.arguments)
        _validate_result(case.name, result)
        warmup_seconds.append(elapsed)

    steady_seconds: list[float] = []
    steady_evidence: list[dict[str, Any]] = []
    valid_payload: Any = False
    status_payload: Any = None
    for _ in range(repeat):
        result, elapsed = _timed_call(case.operation, case.arguments)
        evidence, valid_payload, status_payload = _validate_result(case.name, result)
        steady_seconds.append(elapsed)
        steady_evidence.append(evidence)

    median_seconds = statistics.median(steady_seconds)
    training_samples = int(case.configuration["training_samples"])
    query_samples = int(case.configuration["query_samples"])
    checksums = [entry["checksum_sha256"] for entry in steady_evidence]
    return {
        "name": case.name,
        "family": case.family,
        "scale": case.scale,
        "configuration": case.configuration,
        "execution_mode": case.execution_mode,
        "timing_scope": "fit_and_inference",
        "compile_and_first_warmup_seconds": warmup_seconds[0],
        "additional_warmup_seconds": warmup_seconds[1:],
        "total_warmup_seconds": float(sum(warmup_seconds)),
        "steady_fit_inference_seconds": steady_seconds,
        "median_steady_fit_inference_seconds": median_seconds,
        "throughput": {
            "training_samples_per_second": training_samples / median_seconds,
            "query_samples_per_second": query_samples / median_seconds,
            "total_rows_per_second": (training_samples + query_samples) / median_seconds,
        },
        "valid": valid_payload,
        "status": status_payload,
        "output": steady_evidence[-1],
        "steady_output_checksums": checksums,
        "bitwise_deterministic_steady_output": len(set(checksums)) == 1,
    }


def _distribution_version() -> str:
    try:
        return importlib.metadata.version("phydrax")
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _environment() -> dict[str, Any]:
    devices = []
    for device in jax.devices():
        devices.append(
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "process_index": int(device.process_index),
            }
        )
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "phydrax": _distribution_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
        },
        "jax": {
            "default_backend": jax.default_backend(),
            "enable_x64": bool(jax.config.jax_enable_x64),
            "process_index": int(jax.process_index()),
            "process_count": int(jax.process_count()),
            "local_device_count": int(jax.local_device_count()),
            "device_count": int(jax.device_count()),
            "devices": devices,
        },
    }


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark representative native phydrax.ml fit-and-inference workflows."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSON output path, or '-' for standard output.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of synchronized steady-state repetitions per case (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of synchronized warmups per case; the first includes compilation (default: 1).",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        choices=tuple(_SCALE_CONFIGURATION),
        default=list(_SCALE_CONFIGURATION),
        help="Benchmark scales to run (default: small medium).",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Multiplier for training and query sample counts (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260809,
        help="Base seed used to derive explicit per-case JAX keys.",
    )
    arguments = parser.parse_args(argv)
    if arguments.repeat <= 0:
        parser.error("--repeat must be positive")
    if arguments.warmup <= 0:
        parser.error("--warmup must be positive")
    if not np.isfinite(arguments.scale_factor) or arguments.scale_factor <= 0.0:
        parser.error("--scale-factor must be finite and positive")
    if arguments.seed < 0:
        parser.error("--seed must be nonnegative")
    arguments.scales = list(dict.fromkeys(arguments.scales))
    return arguments


def _write_document(output: str, document: dict[str, Any]) -> None:
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output == "-":
        sys.stdout.write(serialized)
        return
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    cases = [
        builder(scale, arguments.scale_factor, arguments.seed)
        for scale in arguments.scales
        for builder in _CASE_BUILDERS
    ]
    _block_tree(tuple(case.arguments for case in cases))
    started = datetime.now(timezone.utc)
    case_results = [
        _run_case(case, warmup=arguments.warmup, repeat=arguments.repeat)
        for case in cases
    ]
    finished = datetime.now(timezone.utc)
    document = {
        "schema_version": 1,
        "benchmark": "phydrax.ml native scientific workflows",
        "generated_at_utc": finished.isoformat(),
        "elapsed_wall_seconds": (finished - started).total_seconds(),
        "environment": _environment(),
        "run_configuration": {
            "repeat": arguments.repeat,
            "warmup": arguments.warmup,
            "scales": arguments.scales,
            "scale_factor": arguments.scale_factor,
            "base_seed": arguments.seed,
            "case_order": [case.name for case in cases],
            "setup_and_data_generation_timed": False,
            "device_synchronization": "all input leaves before dispatch; all result leaves before stopping timer",
        },
        "cases": case_results,
        "summary": {
            "case_count": len(case_results),
            "all_valid": all(bool(np.all(case["valid"])) for case in case_results),
            "all_outputs_finite": all(case["output"]["finite"] for case in case_results),
            "families": list(dict.fromkeys(case["family"] for case in case_results)),
        },
    }
    _write_document(arguments.output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
