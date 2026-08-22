#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

import numpy as np

from .adapters.base import (
    Availability,
    BenchmarkAdapter,
    CaseSpec,
    NOT_APPLICABLE_REFRESH,
    RefreshEvidence,
    SKIPPED_TRANSFERS,
    SolveResult,
)
from .certificates import independent_certificate
from .schema import (
    empty_distribution,
    SCHEMA_VERSION,
    skip_certificate,
    stable_fingerprint,
    TIMING_PHASES,
    validate_report,
)


_T = TypeVar("_T")
_THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "XLA_FLAGS",
    "JAX_PLATFORMS",
)


def capture_environment() -> dict[str, Any]:
    """Capture deterministic execution-environment evidence for every row."""
    import jax

    devices = jax.devices()
    jax_evidence = {
        "version": jax.__version__,
        "backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "devices": [
            {
                "platform": device.platform,
                "kind": device.device_kind,
            }
            for device in devices
        ],
    }
    evidence: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
        "numpy_version": np.__version__,
        "jax": jax_evidence,
        "thread_environment": {
            key: os.environ.get(key) for key in _THREAD_ENVIRONMENT_KEYS
        },
    }
    return {"fingerprint": stable_fingerprint(evidence), **evidence}


def synchronize(value: Any, /) -> None:
    """Recursively block JAX leaves so asynchronous work is included in timing."""
    import jax

    def visit(item: Any) -> None:
        if isinstance(item, jax.Array):
            item.block_until_ready()
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(object.__getattribute__(item, field.name))
            return
        leaves = jax.tree.leaves(item)
        if len(leaves) == 1 and leaves[0] is item:
            return
        for leaf in leaves:
            visit(leaf)

    visit(value)


def duration_distribution(samples_ms: Sequence[float], /) -> dict[str, Any]:
    samples = [float(value) for value in samples_ms]
    if not samples:
        return empty_distribution()
    return {
        "count": len(samples),
        "samples_ms": samples,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples),
        "max_ms": max(samples),
    }


def execute_case(
    adapter: BenchmarkAdapter,
    spec: CaseSpec,
    /,
    *,
    environment: Mapping[str, Any],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Execute one adapter/case pair with explicit synchronized phase boundaries."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    availability = adapter.availability(spec.capability)
    implementation = adapter.implementation(spec)
    if not availability.available:
        return _skip_row(
            spec,
            implementation=implementation.as_dict(),
            availability=availability,
            environment=environment,
        )

    timing = {phase: empty_distribution() for phase in TIMING_PHASES}
    setup_state, sample = _measure_once(lambda: adapter.setup(spec))
    timing["setup"] = duration_distribution([sample])

    compilation_applicable = adapter.compilation_applicable(setup_state)
    compile_after_preparation = (
        compilation_applicable and adapter.compilation_after_preparation(setup_state)
    )
    compiled_state = setup_state
    if compilation_applicable and not compile_after_preparation:
        compiled_state, sample = _measure_once(lambda: adapter.compile(setup_state))
        timing["compilation"] = duration_distribution([sample])

    prepared_state = compiled_state
    if adapter.preparation_applicable(compiled_state):
        prepared_state, sample = _measure_once(lambda: adapter.prepare(compiled_state))
        timing["preparation"] = duration_distribution([sample])

    if compile_after_preparation:
        prepared_state, sample = _measure_once(lambda: adapter.compile(prepared_state))
        timing["compilation"] = duration_distribution([sample])

    for _ in range(warmup):
        warmup_result = adapter.solve(prepared_state)
        synchronize(warmup_result)

    solve_samples: list[float] = []
    result: SolveResult | None = None
    for _ in range(repeats):
        result, sample = _measure_once(lambda: adapter.solve(prepared_state))
        solve_samples.append(sample)
    if result is None:
        raise RuntimeError("positive repeat count did not produce a solve result")
    timing["solve"] = duration_distribution(solve_samples)

    if adapter.differentiation_applicable(prepared_state):
        prepared_state, sample = _measure_once(
            lambda: adapter.compile_differentiation(prepared_state)
        )
        timing["differentiation_compilation"] = duration_distribution([sample])
        differentiation_samples = []
        for _ in range(repeats):
            _, sample = _measure_once(lambda: adapter.differentiate(prepared_state))
            differentiation_samples.append(sample)
        timing["differentiation"] = duration_distribution(differentiation_samples)

    certificate_problem = adapter.certificate_problem(prepared_state)
    verification, sample = _measure_once(
        lambda: _materialize_and_certify(
            adapter,
            prepared_state,
            certificate_problem,
            result,
            spec.tolerances,
        )
    )
    certificate, converged, operations = verification
    timing["verification"] = duration_distribution([sample])

    refresh_evidence: RefreshEvidence = NOT_APPLICABLE_REFRESH
    refreshed_result: SolveResult | None = None
    refreshed_certificate_problem = None
    refreshed_certificate = None
    refreshed_converged = None
    if adapter.refresh_applicable(prepared_state):
        refresh_result, refresh_sample = _measure_once(
            lambda: adapter.refresh(prepared_state)
        )
        prepared_state, refresh_evidence = refresh_result
        timing["refresh"] = duration_distribution([refresh_sample])
        refreshed_result, refreshed_solve_sample = _measure_once(
            lambda: adapter.solve(prepared_state)
        )
        timing["refreshed_solve"] = duration_distribution([refreshed_solve_sample])
        refreshed_certificate_problem = adapter.certificate_problem(prepared_state)
        refreshed_verification, refreshed_verification_sample = _measure_once(
            lambda: _materialize_and_certify(
                adapter,
                prepared_state,
                refreshed_certificate_problem,
                refreshed_result,
                spec.tolerances,
            )
        )
        refreshed_certificate, refreshed_converged, _ = refreshed_verification
        timing["refreshed_verification"] = duration_distribution(
            [refreshed_verification_sample]
        )

    transfer_results = [result]
    if refreshed_result is not None:
        transfer_results.append(refreshed_result)
    device_to_host_bytes = _device_array_bytes(
        tuple(
            (
                measured_result.solution,
                measured_result.auxiliary,
                measured_result.converged,
                measured_result.operations,
            )
            for measured_result in transfer_results
        )
    )
    transfers = adapter.transfers(
        prepared_state,
        result,
        device_to_host_bytes=device_to_host_bytes,
    )
    refresh_record = {
        **refresh_evidence.as_dict(),
        "certificate_problem_fingerprint": (
            None
            if refreshed_certificate_problem is None
            else refreshed_certificate_problem.identity()["fingerprint"]
        ),
        "certificate_kind": (
            None if refreshed_certificate is None else refreshed_certificate["kind"]
        ),
        "certificate_relative_residual": (
            None
            if refreshed_certificate is None
            else refreshed_certificate["relative_residual"]
        ),
        "certificate_backward_error": (
            None
            if refreshed_certificate is None
            else refreshed_certificate["backward_error"]
        ),
        "certificate_converged": refreshed_converged,
        "independently_certified": (
            None
            if refreshed_certificate is None
            else refreshed_certificate["independently_computed"]
        ),
    }
    status = "success" if converged else "nonconverged"
    row = {
        "case_id": spec.name,
        "schema_version": SCHEMA_VERSION,
        "environment": dict(environment),
        "problem": spec.problem.identity(),
        "implementation": implementation.as_dict(),
        "sizes": spec.problem.sizes(),
        "tolerances": spec.tolerances.as_dict(),
        "outcome": {
            "status": status,
            "converged": converged,
            "message": result.message,
            "skip_reason": None,
        },
        "certificate": certificate,
        "operations": _operation_evidence(operations),
        "refresh": refresh_record,
        "memory": adapter.memory(prepared_state, result),
        "transfers": transfers.as_dict(),
        "timing": timing,
        "availability": availability.as_dict(),
    }
    adapter.release(prepared_state)
    return row


def run_campaign(
    adapters: Mapping[str, BenchmarkAdapter],
    cases: Mapping[str, CaseSpec],
    /,
    *,
    selected_adapters: Sequence[str],
    selected_cases: Sequence[str],
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Run a deterministic selected cross-product and return validated JSON data."""
    import jax

    jax.config.update("jax_enable_x64", True)
    unknown_adapters = sorted(set(selected_adapters) - adapters.keys())
    unknown_cases = sorted(set(selected_cases) - cases.keys())
    if unknown_adapters:
        raise ValueError(f"unknown adapters: {', '.join(unknown_adapters)}")
    if unknown_cases:
        raise ValueError(f"unknown cases: {', '.join(unknown_cases)}")
    if len(set(selected_adapters)) != len(selected_adapters):
        raise ValueError("selected_adapters must not contain duplicates")
    if len(set(selected_cases)) != len(selected_cases):
        raise ValueError("selected_cases must not contain duplicates")
    environment = capture_environment()
    rows = [
        execute_case(
            adapters[adapter_name],
            cases[case_name],
            environment=environment,
            warmup=warmup,
            repeats=repeats,
        )
        for case_name in selected_cases
        for adapter_name in selected_adapters
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment,
        "campaign": {
            "seed": seed,
            "warmup": warmup,
            "repeats": repeats,
            "selected_adapters": list(selected_adapters),
            "selected_cases": list(selected_cases),
            "python_executable": sys.executable,
        },
        "rows": rows,
    }
    validate_report(report)
    return report


def _measure_once(operation: Callable[[], _T], /) -> tuple[_T, float]:
    started = time.perf_counter_ns()
    value = operation()
    synchronize(value)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return value, elapsed_ms


def _skip_row(
    spec: CaseSpec,
    *,
    implementation: dict[str, Any],
    availability: Availability,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    if availability.reason is None or not availability.reason.strip():
        raise ValueError("unavailable adapter must provide a precise skip reason")
    return {
        "case_id": spec.name,
        "schema_version": SCHEMA_VERSION,
        "environment": dict(environment),
        "problem": spec.problem.identity(),
        "implementation": implementation,
        "sizes": spec.problem.sizes(),
        "tolerances": spec.tolerances.as_dict(),
        "outcome": {
            "status": "skipped",
            "converged": None,
            "message": "not executed",
            "skip_reason": availability.reason,
        },
        "certificate": skip_certificate(_certificate_kind(spec.capability)),
        "operations": _operation_evidence({}),
        "refresh": {
            **NOT_APPLICABLE_REFRESH.as_dict(),
            "certificate_problem_fingerprint": None,
            "certificate_kind": None,
            "certificate_relative_residual": None,
            "certificate_backward_error": None,
            "certificate_converged": None,
            "independently_certified": None,
        },
        "memory": {
            "matrix_bytes": None,
            "setup_bytes": None,
            "peak_estimate_bytes": None,
            "evidence": "not measured because the row was skipped",
        },
        "transfers": SKIPPED_TRANSFERS.as_dict(),
        "timing": {phase: empty_distribution() for phase in TIMING_PHASES},
        "availability": availability.as_dict(),
    }


def _operation_evidence(values: Mapping[str, Any | None]) -> dict[str, int | None]:
    operation_fields = (
        "iterations",
        "matvecs",
        "preconditioner_applications",
        "linear_solves",
        "nonlinear_evaluations",
        "jacobian_evaluations",
    )
    evidence: dict[str, int | None] = {}
    for field in operation_fields:
        value = values.get(field)
        evidence[field] = None if value is None else int(np.asarray(value))
    return evidence


def _materialize_and_certify(
    adapter: BenchmarkAdapter,
    prepared_state: Any,
    problem: Any,
    result: SolveResult,
    tolerances: Any,
    /,
) -> tuple[dict[str, Any], bool, dict[str, Any | None]]:
    solution, auxiliary, converged, operations = adapter.materialize_result(
        prepared_state,
        result,
    )
    converged_array = np.asarray(converged)
    if converged_array.shape != () or converged_array.dtype.kind != "b":
        raise TypeError("adapter convergence evidence must be one boolean scalar")
    certificate = independent_certificate(problem, solution, auxiliary)
    effective_converged = bool(converged_array)
    tolerance_satisfied = (
        certificate["residual_norm"] <= tolerances.absolute
        or certificate["relative_residual"] <= tolerances.relative
    )
    effective_converged = effective_converged and tolerance_satisfied
    from .problems import ContinuationProblem, GeneralEigenProblem

    if isinstance(problem, ContinuationProblem):
        effective_converged = (
            effective_converged and certificate["details"]["successful_fold_traversal"]
        )
    if isinstance(problem, GeneralEigenProblem):
        effective_converged = (
            effective_converged
            and certificate["details"]["count_satisfied"]
            and certificate["details"]["largest_magnitude_membership_satisfied"]
        )
    return certificate, effective_converged, operations


def _device_array_bytes(value: Any, /) -> int:
    import jax

    seen: set[int] = set()
    total = 0

    def visit(item: Any) -> None:
        nonlocal total
        if isinstance(item, jax.Array):
            identifier = id(item)
            if identifier not in seen:
                seen.add(identifier)
                total += int(item.size * item.dtype.itemsize)
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(object.__getattribute__(item, field.name))
            return
        leaves = jax.tree.leaves(item)
        if len(leaves) == 1 and leaves[0] is item:
            return
        for leaf in leaves:
            visit(leaf)

    visit(value)
    return total


def _certificate_kind(capability: str) -> str:
    return {
        "linear.scalar": "linear-system",
        "linear.block": "linear-system",
        "nonlinear.root": "nonlinear-root",
        "nonlinear.vi": "variational-inequality-natural-map",
        "eigen.general": "eigenpair-or-schur-relation",
        "continuation.fold": "continuation-branch-residual",
        "optimization.unconstrained": "optimization-stationarity",
        "optimization.constrained": "optimization-kkt",
        "optimization.proximal": "optimization-proximal-stationarity",
        "optimization.linear-program": "optimization-program-kkt",
        "optimization.quadratic-program": "optimization-program-kkt",
        "optimization.conic-program": "optimization-program-kkt",
    }[capability]


__all__ = [
    "capture_environment",
    "duration_distribution",
    "execute_case",
    "run_campaign",
    "synchronize",
]
