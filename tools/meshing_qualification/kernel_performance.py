# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fresh import and deterministic native batch measurements for the benchmark CLI."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from phydrax import _meshcore as mc, config
from phydrax.meshing import NativeExecutionRecord
from tools.meshing_qualification._runtime import (
    _runtime_identity,
    peak_resident_bytes,
    run_campaign_process,
)


_IMPORT_CHILD = """import importlib, json, os, resource, sys, time
name = sys.argv[1]
if name in sys.modules:
    raise RuntimeError("ImportBenchmark.module must not be cached; got " + name)
started = time.perf_counter()
try:
    importlib.import_module(name)
except Exception as error:
    print(json.dumps({"module": name, "status": "failed", "import_seconds": time.perf_counter()-started, "error_type": type(error).__name__, "error": str(error), "process_id": os.getpid()}))
    raise
seconds = time.perf_counter()-started
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
print(json.dumps({"module": name, "status": "completed", "import_seconds": seconds, "process_id": os.getpid(), "peak_resident_bytes": rss, "phase": "fresh-process-first-import"}))
"""


def benchmark_fresh_imports(resolution: int, repeats: int, /) -> dict[str, object]:
    """Measure each import in a fresh interpreter; never warm a module cache."""
    environment = config.environment(config.snapshot().requested)
    modules = ("tools.meshing_qualification", "phydrax.meshing")
    samples: list[dict[str, Any]] = []
    for module in modules:
        for index in range(repeats):
            command = [
                "env",
                "-i",
                *(f"{k}={v}" for k, v in sorted(environment.items())),
                sys.executable,
                "-c",
                _IMPORT_CHILD,
                module,
            ]
            started = perf_counter()
            try:
                completed = run_campaign_process(
                    command, cwd=Path(__file__).resolve().parents[2], timeout=120.0
                )
            except subprocess.TimeoutExpired as error:
                samples.append(
                    {
                        "module": module,
                        "sample": index,
                        "status": "timeout",
                        "startup_status": "terminated",
                        "child_total_launch_seconds": perf_counter() - started,
                        "import_seconds": None,
                        "error": str(error),
                    }
                )
                continue
            total = perf_counter() - started
            record = (
                json.loads(completed.stdout)
                if completed.stdout.strip()
                else {"import_seconds": None}
            )
            samples.append(
                {
                    **record,
                    "module": module,
                    "sample": index,
                    "startup_status": "completed"
                    if completed.returncode == 0
                    else "failed",
                    "status": "completed" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                    "stdout": completed.stdout,
                    "child_total_launch_seconds": total,
                }
            )
    return {
        "kind": "fresh-process-import-benchmark",
        "resolution": resolution,
        "resolution_semantics": "unused-import-has-no-size-axis",
        "repeats": repeats,
        "samples": samples,
        "warm_samples": None,
        "warm_measurement": "not-supported-fresh-import-only",
        "cache_policy": "fresh-interpreter-each-sample-filesystem-cache-not-flushed",
        "runtime": _runtime_identity(),
        "thread_policy": "inherited-config-environment",
        "configuration": config.snapshot().to_record(),
        "status": "completed"
        if all(s["status"] == "completed" for s in samples)
        else "failed",
    }


def array_record(value: np.ndarray, /) -> dict[str, object]:
    """Preserve every result bit, shape, and dtype at the JSON boundary."""
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "bytes": value.tobytes().hex(),
    }


def require_identical(reference: object, candidate: object, /) -> None:
    """Reject any result or non-timing evidence mismatch."""
    if reference != candidate:
        raise ValueError(
            "NativeBatchBenchmark.deterministic_evidence must match serial bits and counters; got a mismatch"
        )


def _cases(count: int) -> list[tuple[str, Callable[[], object]]]:
    rng = np.random.default_rng(971)
    points = rng.integers(-16, 17, size=(5, count, 3)).astype(np.float64)
    points[:, ::3] = 0
    points[:, 1::3, 2] = points[:, 1::3, 0]
    cases: list[tuple[str, Callable[[], object]]] = []
    for name, width, arity in (
        ("orient2d", 2, 3),
        ("orient3d", 3, 4),
        ("incircle", 2, 4),
        ("insphere", 3, 5),
    ):
        args = tuple(np.ascontiguousarray(points[k, :, :width]) for k in range(arity))
        for symbolic in (False, True):
            label = f"{'symbolic' if symbolic else 'exact'}_{name}"
            arguments = (
                args + (np.tile(np.arange(arity, dtype=np.int64), (count, 1)),)
                if symbolic
                else args
            )
            kernel = getattr(mc, label)
            cases.append(
                (label, lambda kernel=kernel, arguments=arguments: kernel(*arguments))
            )
    triangles = np.tile(
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64), (count, 1, 1)
    )
    queries = np.tile(np.asarray((0.25, 0.25, 0), dtype=np.float64), (count, 1))
    queries[1::5, 2] = 1
    queries[2::5] = 0
    triangles[3::5] = 0
    cases.append(
        (
            "point_triangle_locations",
            lambda: mc.point_triangle_locations(queries, triangles),
        )
    )
    return cases


def _result_record(value: Any) -> object:
    match value:
        case np.ndarray():
            return array_record(value)
        case tuple() | list():
            return [_result_record(item) for item in value]
        case _:
            raise TypeError(
                f"NativeBatchBenchmark.result must have a supported native result representation; got {type(value).__name__}"
            )


def _sample(call: Callable[[], object], threads: int) -> tuple[dict[str, Any], object]:
    limits = mc.NativeExecutionLimits(0, 0, 0, 1 << 30, 120.0)
    scope = mc.NativeExecutionScope(limits)
    with mc.NativeParallelExecution(threads, 120.0), scope:
        started = perf_counter()
        result = call()
        seconds = perf_counter() - started
    evidence = scope.completed_evidence()
    full = NativeExecutionRecord(evidence).to_record()
    # Normalize clocks through the evidence owner, not by dropping record fields.
    deterministic = NativeExecutionRecord(
        dataclasses.replace(evidence, elapsed_seconds=0.0, prior_elapsed_seconds=0.0)
    ).to_record()
    payload = {"result": _result_record(result), "native_evidence": deterministic}
    return {
        "seconds": seconds,
        "native_evidence": full,
        "peak_resident_bytes": peak_resident_bytes(),
        "status": evidence.status.name,
    }, payload


def benchmark_parallel_batches(resolution: int, repeats: int, /) -> dict[str, object]:
    """Compare affine thread policies on identical prepared host batches."""
    count = 10000 * resolution
    cases = _cases(count)
    available = len(os.sched_getaffinity(0))
    threads = (1, 8, min(32, available)) if available > 8 else (1, 8)
    records: list[dict[str, Any]] = []
    for name, call in cases:
        reference = None
        candidates = []
        serial_median = None
        for requested in threads:
            first, payload = _sample(call, requested)
            if reference is None:
                reference = payload
            require_identical(reference, payload)
            warm = []
            for _ in range(repeats):
                sample, actual = _sample(call, requested)
                require_identical(reference, actual)
                warm.append(sample)
            median = float(np.median([sample["seconds"] for sample in warm]))
            if requested == 1:
                serial_median = median
            if serial_median is None:
                raise RuntimeError(
                    "NativeBatchBenchmark.serial_median must precede parallel samples; got absent serial sample"
                )
            speedup = serial_median / median
            candidates.append(
                {
                    "requested_threads": requested,
                    "first": first,
                    "warm_samples": warm,
                    "warm_median_seconds": median,
                    "speedup": speedup,
                    "required_speedup": 0.6 * requested,
                    "threshold_met": speedup >= 0.6 * requested
                    and first["status"] == "OK"
                    and all(sample["status"] == "OK" for sample in warm),
                    "bitwise_equivalent": True,
                }
            )
        records.append(
            {
                "kernel": name,
                "deterministic_evidence": reference,
                "candidates": candidates,
            }
        )
    return {
        "kind": "native-parallel-batch-benchmark",
        "resolution": resolution,
        "batch_count": count,
        "batch_count_mapping": "10000 * resolution; 1=10000, 2=20000, 4=40000",
        "repeats": repeats,
        "samples": records,
        "runtime": _runtime_identity(),
        "thread_policy": {
            "owner": "mc.NativeParallelExecution",
            "requested": list(threads),
            "affinity_cpu_count": available,
            "larger_count_status": "supported"
            if available > 8
            else "not-supported-affinity-at-most-eight",
        },
        "native_scope_limits": {
            "work": 0,
            "queries": 0,
            "cavity": 0,
            "bytes": 1 << 30,
            "seconds": 120.0,
        },
        "input_provenance": "seed971-integer-host-inputs; exact-zero/coplanar and nondegenerate rows; symbolic vertex IDs; point-triangle interior/exterior/vertex/degenerate rows",
        "cache_policy": "inputs-prepared-once-no-result-cache",
        "peak_resident_bytes": peak_resident_bytes(),
        "python_resource_accounting_seconds": None,
        "python_resource_accounting_measurement": "unmeasured",
        "status": "completed"
        if all(
            candidate["first"]["status"] == "OK"
            and all(sample["status"] == "OK" for sample in candidate["warm_samples"])
            for record in records
            for candidate in record["candidates"]
        )
        else "failed",
    }
