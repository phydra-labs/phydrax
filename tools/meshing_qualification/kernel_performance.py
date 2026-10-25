# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fresh import and deterministic native batch measurements for the benchmark CLI."""

from __future__ import annotations

import dataclasses
import json
import os
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
    CampaignProcessTimeout,
    peak_resident_bytes,
    run_campaign_process,
)


_IMPORT_CHILD = """import json, os, sys, time
name = sys.argv[1]
if name in sys.modules:
    raise RuntimeError("ImportBenchmark.module must not be cached; got " + name)
started = time.perf_counter()
try:
    match name:
        case "tools.meshing_qualification":
            from tools.meshing_qualification.__main__ import main
            from tools.meshing_qualification.scenarios.native_lifecycle import run
            entry_points = (main, run)
        case "phydrax.meshing":
            from phydrax.meshing import (
                NativeMeshingProvider, MeshAdaptationPolicy, prepare_mesh_adaptation,
            )
            entry_points = (
                NativeMeshingProvider, MeshAdaptationPolicy, prepare_mesh_adaptation,
            )
        case _:
            raise ValueError("ImportBenchmark.module must name a measured boundary; got " + name)
    resolved = [entry.__module__ + "." + entry.__qualname__ for entry in entry_points]
except Exception as error:
    print(json.dumps({"module": name, "status": "failed", "import_seconds": time.perf_counter()-started, "error_type": type(error).__name__, "error": str(error), "process_id": os.getpid()}))
    raise
seconds = time.perf_counter()-started
# VmHWM is this process's post-exec high water; ru_maxrss would carry the launcher's.
rss = None
if sys.platform == "linux":
    with open("/proc/self/status", encoding="ascii") as status:
        rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmHWM:"))
print(json.dumps({"module": name, "status": "completed", "import_seconds": seconds, "entry_points": resolved, "process_id": os.getpid(), "peak_resident_bytes": rss, "phase": "fresh-process-functional-import"}))
"""


def benchmark_fresh_imports(resolution: int, repeats: int, /) -> dict[str, object]:
    """Resolve the CLI/native entry points in fresh interpreters, not lazy shells."""
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
            except CampaignProcessTimeout as error:
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
            # The env-scrubbed child cannot see the launch deadline, so the
            # launcher's own clock decides; the reporting grace adds no import time.
            status = (
                "timeout"
                if completed.deadline_expired
                else "completed"
                if completed.returncode == 0
                else "failed"
            )
            samples.append(
                {
                    **record,
                    "module": module,
                    "sample": index,
                    "startup_status": "completed"
                    if completed.returncode == 0
                    else "failed",
                    "status": status,
                    "exited_after_scientific_deadline": completed.deadline_expired,
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
        "measurement_boundary": "qualification-cli-and-native-lifecycle-or-native-provider-and-adaptation",
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


_BATCH_LIMITS = mc.NativeExecutionLimits(0, 0, 0, 1 << 30, 120.0)
"""Declared per-sample allowance: unlimited work/queries/cavity, 1 GiB managed, 120 s."""


def _limits_record(limits: mc.NativeExecutionLimits, /) -> dict[str, int | float]:
    return {
        "work": limits.maximum_work_units,
        "queries": limits.maximum_geometry_queries,
        "cavity": limits.maximum_cavity_cells,
        "bytes": limits.maximum_managed_bytes,
        "seconds": limits.maximum_seconds,
    }


def _exceeded_limits(
    error: mc.MeshcoreError, limits: mc.NativeExecutionLimits, /
) -> list[str]:
    """Name each declared allowance the refused batch demand exceeds.

    The native refusal reports the requested increment and, when available, the
    allowance remaining at refusal; a zero declared limit is unlimited.
    """
    requested = error.batch_requested_upper
    if requested is None:
        return []
    declared = (
        limits.maximum_work_units,
        limits.maximum_geometry_queries,
        limits.maximum_cavity_cells,
        limits.maximum_managed_bytes,
    )
    evidence = error.batch_refusal_evidence
    remaining = declared if evidence is None else tuple(int(v) for v in evidence[:4])
    return [
        name
        for name, limit, available, demand in zip(
            ("work", "queries", "cavity", "bytes"),
            declared,
            remaining,
            requested,
            strict=True,
        )
        if limit != 0 and demand > available
    ]


def _sample(
    call: Callable[[], object],
    threads: int,
    /,
    *,
    kernel: str,
    batch_count: int,
    limits: mc.NativeExecutionLimits,
) -> tuple[dict[str, Any], object]:
    scope = mc.NativeExecutionScope(limits)
    try:
        with mc.NativeParallelExecution(threads, limits.maximum_seconds), scope:
            started = perf_counter()
            result = call()
            seconds = perf_counter() - started
    except mc.MeshcoreError as error:
        exceeded = _exceeded_limits(error, limits)
        # Only an admission refusal explained by declared demand is a nonpositive
        # outcome; any other native failure stays a harness/native defect.
        if error.status is not mc.MeshcoreStatus.CAPACITY_EXCEEDED or not exceeded:
            raise
        evidence = scope.completed_evidence()
        return {
            "kernel": kernel,
            "batch_count": batch_count,
            "requested_threads": threads,
            "seconds": None,
            "timing_is_headline": False,
            "status": "resource-refused",
            "native_status": error.status.name,
            "native_scope_limits": _limits_record(limits),
            "native_evidence": NativeExecutionRecord(evidence).to_record(),
            "peak_resident_bytes": peak_resident_bytes(),
            "refusal": {
                "error": str(error),
                "exceeded_limits": exceeded,
                "batch_admitted_upper": error.batch_admitted_upper,
                "batch_requested_upper": error.batch_requested_upper,
                "batch_refusal_evidence": None
                if error.batch_refusal_evidence is None
                else error.batch_refusal_evidence.tolist(),
                "work_counters": None
                if error.work_counters is None
                else dict(error.work_counters.items()),
                "memory_counters": None
                if error.memory_counters is None
                else dict(error.memory_counters.items()),
            },
        }, None
    evidence = scope.completed_evidence()
    full = NativeExecutionRecord(evidence).to_record()
    # Normalize clocks through the evidence owner, not by dropping record fields.
    deterministic = NativeExecutionRecord(
        dataclasses.replace(evidence, elapsed_seconds=0.0, prior_elapsed_seconds=0.0)
    ).to_record()
    payload = {"result": _result_record(result), "native_evidence": deterministic}
    return {
        "seconds": seconds,
        "timing_is_headline": True,
        "native_evidence": full,
        "peak_resident_bytes": peak_resident_bytes(),
        "status": evidence.status.name,
    }, payload


def _available_cpu_count() -> int:
    """CPUs this process may run on: its affinity mask where the platform has one.

    Darwin and Windows expose no process affinity mask to Python; every logical
    CPU the system reports is schedulable there.
    """
    if sys.platform != "win32" and sys.platform != "darwin":
        return len(os.sched_getaffinity(0))
    count = os.cpu_count()
    if count is None:
        raise RuntimeError(
            "kernel_performance.available_cpu_count must be reported by the platform; got None"
        )
    return count


def benchmark_parallel_batches(resolution: int, repeats: int, /) -> dict[str, object]:
    """Compare affine thread policies on identical prepared host batches."""
    count = 10000 * resolution
    cases = _cases(count)
    available = _available_cpu_count()
    threads = (1, 8, min(32, available)) if available > 8 else (1, 8)
    records: list[dict[str, Any]] = []
    for name, call in cases:
        reference = None
        candidates = []
        serial_median = None
        for requested in threads:
            first, payload = _sample(
                call, requested, kernel=name, batch_count=count, limits=_BATCH_LIMITS
            )
            if requested == 1 and payload is not None:
                reference = payload
            if reference is not None and payload is not None:
                require_identical(reference, payload)
            warm = []
            for _ in range(repeats):
                sample, actual = _sample(
                    call, requested, kernel=name, batch_count=count, limits=_BATCH_LIMITS
                )
                if reference is not None and actual is not None:
                    require_identical(reference, actual)
                warm.append(sample)
            admitted = first["status"] == "OK" and all(
                sample["status"] == "OK" for sample in warm
            )
            median = (
                float(np.median([sample["seconds"] for sample in warm]))
                if admitted
                else None
            )
            if requested == 1:
                serial_median = median
            speedup = (
                serial_median / median
                if serial_median is not None and median is not None
                else None
            )
            candidates.append(
                {
                    "requested_threads": requested,
                    "first": first,
                    "warm_samples": warm,
                    "warm_median_seconds": median,
                    "speedup": speedup,
                    "required_speedup": 0.6 * requested,
                    "threshold_met": speedup is not None and speedup >= 0.6 * requested,
                    "bitwise_equivalent": True
                    if admitted and serial_median is not None
                    else None,
                    "status": "completed" if admitted else "resource-refused",
                    "timing_is_headline": admitted and serial_median is not None,
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
        "native_scope_limits": _limits_record(_BATCH_LIMITS),
        "input_provenance": "seed971-integer-host-inputs; exact-zero/coplanar and nondegenerate rows; symbolic vertex IDs; point-triangle interior/exterior/vertex/degenerate rows",
        "cache_policy": "inputs-prepared-once-no-result-cache",
        "peak_resident_bytes": peak_resident_bytes(),
        "python_resource_accounting_seconds": None,
        "python_resource_accounting_measurement": "unmeasured",
        "timing_is_headline": all(
            candidate["timing_is_headline"]
            for record in records
            for candidate in record["candidates"]
        ),
        "status": "completed"
        if all(
            candidate["status"] == "completed"
            for record in records
            for candidate in record["candidates"]
        )
        else "resource-refused",
    }
