#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Benchmark the qualified float32, matrix-free native PCG route.

Run with JAX_PLATFORMS=cpu or JAX_PLATFORMS=mps, for example:
    python -m tools.native_krylov_benchmarks --sizes 16,1024 --repeats 3

The NumPy/SciPy float64 reference is host-only, uses the same rounded inputs,
and requires linear storage. It is never a fallback for the native solve.
Compiler memory estimates and logical retained bytes are not allocator peaks.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.linalg import solveh_banded

from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import config as pxconfig, linalg as la


_ROOT = Path(__file__).resolve().parents[1]
_RELATIVE_TOLERANCE = 1e-4
_MAXIMUM_STEPS = 64
_MAXIMUM_SIZE = 1_048_576
_MAXIMUM_REPEATS = 100


def _action(vector: Array) -> Array:
    """Apply the SPD Toeplitz operator (3, -1, -1), with zero end values."""
    zero = jnp.zeros((1,), dtype=jnp.float32)
    below = jnp.concatenate((zero, vector[:-1]))
    above = jnp.concatenate((vector[1:], zero))
    return jnp.asarray(3.0, dtype=jnp.float32) * vector - below - above


def _prepare(size: int) -> la.PreparedLinearSolve:
    space = la.ArraySpace((size,), dtype=jnp.float32)
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )
    problem = la.LinearSystem(
        la.FunctionLinearOperator(
            _action,
            source=space,
            target=space,
            transpose_action=_action,
            properties=properties,
        )
    )
    policy = la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(
            relative=_RELATIVE_TOLERANCE,
            absolute=0.0,
            max_steps=_MAXIMUM_STEPS,
        ),
        differentiation=la.DifferentiationPolicy("none"),
        failure=la.FailurePolicy("status"),
    )
    prepared = la.prepare(problem, policy)
    if prepared.plan.backend != "native-krylov":
        raise RuntimeError("This benchmark requires the native-krylov route.")
    return prepared


def _solve(prepared: la.PreparedLinearSolve, rhs: Array) -> dict[str, Array]:
    result, checks = la.solve_checked(
        prepared,
        rhs,
        check_policy=la.LinearSolveCheckPolicy(
            relative_tolerance=2e-4, absolute_tolerance=0.0
        ),
    )
    if not isinstance(result.value, Array):
        raise TypeError("The declared ArraySpace must return one JAX array.")
    return {
        "value": result.value,
        "status": result.status,
        "successful": result.successful,
        "residual_norm": result.diagnostics.residual_norm,
        "relative_residual": result.diagnostics.relative_residual,
        "iterations": result.diagnostics.iterations,
        "matvec_count": result.diagnostics.matvec_count,
        "adjoint_matvec_count": result.diagnostics.adjoint_matvec_count,
        "finite": result.diagnostics.finite,
        "converged": result.diagnostics.converged,
        "checked_true_residual_norm": checks.true_residual_norm,
        "check_valid": checks.valid,
        "check_residual_ok": checks.residual_ok,
    }


def _reference(rhs: NDArray[np.float32]) -> NDArray[np.float64]:
    """Independent LAPACK banded Cholesky solve, never a dense n-by-n matrix."""
    band = np.zeros((2, rhs.size), dtype=np.float64)
    band[0, :] = np.float64(np.float32(3.0))
    band[1, :-1] = -np.float64(np.float32(1.0))
    return np.asarray(
        solveh_banded(band, rhs.astype(np.float64), lower=True), dtype=np.float64
    )


def _json_metric(value: float) -> float | None:
    """Keep failed nonfinite evidence JSON-portable without changing acceptance."""
    return value if np.isfinite(value) else None


def _accuracy(
    output: dict[str, NDArray[np.generic]],
    rhs: NDArray[np.float32],
    reference: NDArray[np.float64],
) -> dict[str, Any]:
    """Recompute the original residual and forward error in host float64."""
    value = np.asarray(output["value"], dtype=np.float64)
    target = rhs.astype(np.float64)
    residual = np.float64(np.float32(3.0)) * value - target
    residual[1:] -= value[:-1]
    residual[:-1] -= value[1:]
    rhs_norm = float(np.linalg.norm(target))
    residual_norm = float(np.linalg.norm(residual))
    reference_norm = float(np.linalg.norm(reference))
    error_norm = float(np.linalg.norm(value - reference))
    smallest_eigenvalue = float(3.0 - 2.0 * np.cos(np.pi / (rhs.size + 1)))
    reported = float(output["residual_norm"])
    checked = float(output["checked_true_residual_norm"])
    iterations = int(output["iterations"])
    matvecs = int(output["matvec_count"])
    adjoint_matvecs = int(output["adjoint_matvec_count"])
    # kappa_2 < 5 uniformly in n. This retains the integration test's 1e-4
    # solve tolerance, 1.1e-4 host residual gate and 1e-5 residual agreement.
    # The independent forward gate follows kappa_2 * relative residual; the
    # sharper a-posteriori bound uses the exact smallest eigenvalue.
    residual_ok = residual_norm <= 1.1e-4 * rhs_norm
    agreement_ok = (
        abs(reported - residual_norm) <= 1e-5 * rhs_norm
        and abs(checked - residual_norm) <= 1e-5 * rhs_norm
    )
    solution_ok = (
        error_norm <= 5.5e-4 * reference_norm
        and error_norm <= residual_norm / smallest_eigenvalue + 1e-12 * rhs_norm
    )
    work_ok = (
        0 < iterations <= _MAXIMUM_STEPS
        and iterations + 2 <= matvecs <= 2 * iterations + 2
        and adjoint_matvecs == 0
    )
    status = int(output["status"])
    passed = (
        status == int(la.LinearSolveStatus.SUCCESS)
        and bool(output["successful"])
        and bool(output["finite"])
        and bool(output["converged"])
        and bool(output["check_valid"])
        and bool(output["check_residual_ok"])
        and output["value"].dtype == jnp.float32
        and bool(np.all(np.isfinite(value)))
        and residual_ok
        and agreement_ok
        and solution_ok
        and work_ok
    )
    return {
        "passed": passed,
        "status": status,
        "status_message": la.linear_status_message(status),
        "successful": bool(output["successful"]),
        "finite": bool(output["finite"]),
        "converged": bool(output["converged"]),
        "check_valid": bool(output["check_valid"]),
        "check_residual_ok": bool(output["check_residual_ok"]),
        "true_residual_norm": _json_metric(residual_norm),
        "true_relative_residual": _json_metric(residual_norm / rhs_norm),
        "reported_residual_norm": _json_metric(reported),
        "checked_true_residual_norm": _json_metric(checked),
        "reference_solution_error_norm": _json_metric(error_norm),
        "reference_solution_relative_error": _json_metric(error_norm / reference_norm),
        "forward_error_upper_bound": _json_metric(residual_norm / smallest_eigenvalue),
        "smallest_eigenvalue": smallest_eigenvalue,
        "residual_ok": residual_ok,
        "reported_residual_agreement_ok": agreement_ok,
        "independent_solution_ok": solution_ok,
        "work_ok": work_ok,
        "iterations": iterations,
        "matvec_count": matvecs,
        "adjoint_matvec_count": adjoint_matvecs,
    }


def _case(size: int, repeats: int) -> dict[str, Any]:
    host_rhs, host_preparation = measure_host(
        lambda: np.random.default_rng(23).normal(size=size).astype(np.float32)
    )
    reference, reference_seconds = measure_host(lambda: _reference(host_rhs))
    # Preparation creates small JAX metadata as well as the operator/template;
    # its synchronized duration is not mislabeled as guaranteed host-only work.
    prepared, preparation = measure_synchronized(lambda: _prepare(size))
    rhs, input_transfer = measure_synchronized(lambda: jax.device_put(host_rhs))
    dynamic, static = eqx.partition(prepared, eqx.is_array)

    def solve(numeric: la.PreparedLinearSolve, target: Array) -> dict[str, Array]:
        return _solve(eqx.combine(numeric, static), target)

    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(solve).lower(dynamic, rhs), lambda lowered: lowered.compile()
    )
    first, first_dispatch = measure_host(lambda: compiled(dynamic, rhs))
    _, first_wait = measure_synchronized(lambda value=first: value)
    first_host, first_output_transfer = measure_host(
        lambda value=first: jax.device_get(value)
    )
    first_accuracy = _accuracy(first_host, host_rhs, reference)
    del first, first_host
    output, warm = measure_repeated(
        lambda: compiled(dynamic, rhs), warmup=0, repeats=repeats
    )
    _, ready_synchronization = measure_synchronized(lambda: output)
    host_output, output_transfer = measure_host(lambda: jax.device_get(output))
    accuracy = _accuracy(host_output, host_rhs, reference)
    compiler = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Backend did not expose compiler cost or memory analysis.",
    )
    devices = [
        {"id": device.id, "platform": device.platform, "kind": device.device_kind}
        for device in sorted(output["value"].devices(), key=lambda device: device.id)
    ]
    return {
        "size": size,
        "passed": first_accuracy["passed"] and accuracy["passed"],
        "route": prepared.plan.backend,
        "method": "pcg",
        "dtype": str(output["value"].dtype),
        "rhs_dtype": str(rhs.dtype),
        "devices": devices,
        "operator": "matrix-free SPD tridiagonal: diagonal 3, off-diagonals -1",
        "reference": "host scipy.linalg.solveh_banded float64 on float32-rounded inputs",
        "accuracy": accuracy,
        "first_accuracy": first_accuracy,
        "timing": {
            "host_input_preparation_seconds": host_preparation,
            "host_reference_seconds": reference_seconds,
            "operator_preparation_synchronized_seconds": preparation,
            "input_transfer_synchronized_seconds": input_transfer,
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
            "first_dispatch_seconds": first_dispatch,
            "first_completion_wait_seconds": first_wait,
            "first_execution_synchronized_seconds": first_dispatch + first_wait,
            "first_output_transfer_seconds": first_output_transfer,
            "warm_synchronized": warm.to_seconds_dict(),
            "ready_result_synchronization_seconds": ready_synchronization,
            "output_transfer_seconds": output_transfer,
            "amortized_preparation_through_execution_seconds_per_solve": (
                host_preparation
                + preparation
                + input_transfer
                + compilation.lowering_seconds
                + compilation.compilation_seconds
                + first_dispatch
                + first_wait
                + sum(warm.samples_seconds)
            )
            / (repeats + 1),
        },
        "compiler": asdict(compiler),
        "compiler_estimated_device_memory_bytes": compiler.estimated_device_memory_bytes,
        "logical_bytes": {
            "prepared": logical_array_bytes(prepared),
            "rhs": logical_array_bytes(rhs),
            "output": logical_array_bytes(output),
            "host_input_and_reference": logical_array_bytes((host_rhs, reference)),
            "retained_total": logical_array_bytes(
                (prepared, rhs, output, host_rhs, reference, host_output)
            ),
        },
    }


def run(*, sizes: Sequence[int], repeats: int) -> dict[str, Any]:
    if not sizes or len(sizes) > 16 or len(set(sizes)) != len(sizes):
        raise ValueError("Supply between one and 16 distinct problem sizes.")
    if any(size < 2 or size > _MAXIMUM_SIZE for size in sizes):
        raise ValueError(f"Each size must be between 2 and {_MAXIMUM_SIZE}.")
    if repeats < 1 or repeats > _MAXIMUM_REPEATS:
        raise ValueError(f"repeats must be between 1 and {_MAXIMUM_REPEATS}.")
    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
    records = [_case(size, repeats) for size in sizes]
    identity = capture_benchmark_identity(_ROOT, Path(__file__), records[0].keys())
    return {
        "identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "settings": {
            "sizes": list(sizes),
            "repeats": repeats,
            "relative_tolerance": _RELATIVE_TOLERANCE,
            "maximum_steps": _MAXIMUM_STEPS,
            "dtype": "float32",
            "x64_enabled": True,
        },
        "passed": all(record["passed"] for record in records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="16,1024,65536")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--require-backend", choices=("cpu", "mps"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.require_backend is not None
        and jax.default_backend() != arguments.require_backend
    ):
        raise RuntimeError(
            f"Required backend {arguments.require_backend!r} is not active."
        )
    payload = run(
        sizes=tuple(int(size) for size in arguments.sizes.split(",")),
        repeats=arguments.repeats,
    )
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(text, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
