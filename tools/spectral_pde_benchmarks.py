#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from math import ceil, log2, prod
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    synchronize,
)
from phydrax.signal import fourier_resample


@dataclass(frozen=True)
class SpectralPDEBenchmarkRecord:
    mode_count: int
    evaluation_count: int
    compilation_id: str
    operator_id: str
    resolved_method: str
    dealiasing_report_id: str
    coefficient_bytes: int
    padded_coefficient_bytes: int
    compiler_wall_ms: float
    first_jit_ms: float
    steady_wall_ms: float
    maximum_drift_error: float
    parameter_gradient_error: float
    alias_defect: float
    state_packing_evidence_id: str
    packed_backend_shape: tuple[int, ...]
    packed_public_bytes: int
    packed_backend_bytes: int
    packed_diffrax_wall_ms: float
    explicit_real_wall_ms: float
    packed_pathwise_defect: float
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.maximum_drift_error <= 1e-9
            and self.parameter_gradient_error <= 1e-8
            and self.alias_defect <= 1e-9
            and self.packed_pathwise_defect <= 1e-8
            and self.packed_public_bytes == self.packed_backend_bytes
        )


def _problem() -> Any:
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    t = phx.equations.PDECoordinate("t", "time")
    field = phx.equations.PDEField("u", coordinates=("x", "t"))
    parameter = phx.equations.PDEParameter("kappa", value=0.05)
    u = phx.equations.PDEExpression.field("u")
    return phx.equations.PDEProblemIR(
        (x, t),
        (field,),
        parameters=(parameter,),
        equations=(
            phx.equations.PDEEquation(
                "reaction-diffusion",
                u.derivative("t"),
                phx.equations.PDEExpression.parameter("kappa") * u.laplacian("x")
                + u * (1.0 - u),
            ),
        ),
    )


def _measure(function: Any, argument: Any, repeats: Any) -> Any:
    compiled = jax.jit(function)
    value, first_seconds = measure_synchronized(lambda: compiled(argument))
    value, distribution = measure_repeated(
        lambda: compiled(argument),
        warmup=0,
        repeats=repeats,
    )
    return (
        value,
        1_000.0 * first_seconds,
        # ty: ignore[invalid-argument-type]
        1_000.0 * float(distribution.mean_seconds),
    )


def run_spectral_pde_benchmark(
    mode_count: int = 128,
    /,
    *,
    repeats: int = 5,
) -> SpectralPDEBenchmarkRecord:
    count = int(mode_count)
    repeat_count = int(repeats)
    if count < 8 or repeat_count <= 0:
        raise ValueError("mode_count must be at least eight and repeats positive.")
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="u",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2),
    )
    started = time.perf_counter()
    compiled = phx.equations.compile_semidiscrete_pde(
        _problem(),
        space,
        method,
    )
    compiler_wall = 1e3 * (time.perf_counter() - started)
    x = space.axes[0].nodes
    physical = 0.2 + 0.1 * jnp.sin(2.0 * jnp.pi * x)
    state = compiled.project_state(physical)
    coefficient = jnp.asarray(0.07)

    def drift(arguments: Any) -> Any:
        value, kappa = arguments
        return compiled(0.0, value, {"kappa": kappa})

    actual, first, steady = _measure(
        drift,
        (state, coefficient),
        repeat_count,
    )
    expected_physical = coefficient * space.laplacian(physical) + physical * (
        1.0 - physical
    )
    expected = space.project(expected_physical)
    actual_gradient = jax.grad(
        lambda kappa: jnp.real(jnp.vdot(drift((state, kappa)), state))
    )(coefficient)
    expected_gradient = jax.grad(
        lambda kappa: jnp.real(
            jnp.vdot(
                space.project(
                    kappa * space.laplacian(physical) + physical * (1.0 - physical)
                ),
                state,
            )
        )
    )(coefficient)

    high = space.project(jnp.sin(2.0 * jnp.pi * (count // 2 - 1) * x))
    dealiasing = compiled.spatial_method.dealiasing
    product = dealiasing.project(dealiasing.reconstruct(high) ** 2)
    retained_product = space.reconstruct(product)
    alias_defect = jnp.max(jnp.abs(retained_product - 0.5))
    maximum_error = jnp.max(jnp.abs(actual - expected))
    gradient_error = jnp.abs(actual_gradient - expected_gradient)
    report = dealiasing.report
    itemsize = jnp.dtype(space.plan.precision.coefficient_dtype).itemsize
    real_rate = -0.2
    imaginary_rate = 0.5
    conjugate_rate = 0.1
    complex_problem = phx.solver.DifferentialProblem(
        lambda time, value, args: (
            (real_rate + 1j * imaginary_rate) * value + conjugate_rate * jnp.conj(value)
        ),
        state,
        t0=0.0,
        t1=0.05,
    )

    def real_drift(time: Any, value: Any, args: Any) -> Any:
        del time, args
        real, imag = value
        return jnp.stack(
            (
                (real_rate + conjugate_rate) * real - imaginary_rate * imag,
                imaginary_rate * real + (real_rate - conjugate_rate) * imag,
            )
        )

    real_problem = phx.solver.DifferentialProblem(
        real_drift,
        jnp.stack((jnp.real(state), jnp.imag(state))),
        t0=0.0,
        t1=0.05,
    )
    packed_started = time.perf_counter()
    packed_solution = phx.solver.solve_diffrax(
        complex_problem,
        save_times=jnp.asarray([0.05]),
        rtol=1e-8,
        atol=1e-10,
    )
    jax.block_until_ready(packed_solution.states)
    packed_wall = 1e3 * (time.perf_counter() - packed_started)
    real_started = time.perf_counter()
    real_solution = phx.solver.solve_diffrax(
        real_problem,
        save_times=jnp.asarray([0.05]),
        rtol=1e-8,
        atol=1e-10,
    )
    jax.block_until_ready(real_solution.states)
    real_wall = 1e3 * (time.perf_counter() - real_started)
    expected_packed = jax.lax.complex(
        real_solution.states[:, 0],
        real_solution.states[:, 1],
    )
    packed_defect = jnp.max(jnp.abs(packed_solution.states - expected_packed))
    # ty: ignore[unresolved-attribute]
    coordinates = packed_solution.temporal_evidence.state_coordinates
    if coordinates is None:
        raise RuntimeError(
            "Complex Diffrax benchmark did not prepare real state coordinates."
        )
    backend_itemsize = jnp.dtype(coordinates.coordinate_dtype).itemsize
    public_bytes = state.size * state.dtype.itemsize
    backend_bytes = prod(coordinates.coordinate_shape) * backend_itemsize
    finite = bool(
        jnp.all(jnp.isfinite(actual))
        & jnp.isfinite(maximum_error)
        & jnp.isfinite(gradient_error)
        & jnp.isfinite(alias_defect)
        & jnp.isfinite(packed_defect)
    )
    return SpectralPDEBenchmarkRecord(
        mode_count=count,
        evaluation_count=report.evaluation_shape[0],
        compilation_id=compiled.compilation_id,
        operator_id=compiled.semilinear_drift.operator_id,
        resolved_method=compiled.resolved_method,
        dealiasing_report_id=report.report_id,
        coefficient_bytes=count * itemsize,
        padded_coefficient_bytes=report.evaluation_shape[0] * itemsize,
        compiler_wall_ms=float(compiler_wall),
        first_jit_ms=float(first),
        steady_wall_ms=float(steady),
        maximum_drift_error=float(maximum_error),
        parameter_gradient_error=float(gradient_error),
        alias_defect=float(alias_defect),
        state_packing_evidence_id=coordinates.evidence_id,
        packed_backend_shape=coordinates.coordinate_shape,
        packed_public_bytes=int(public_bytes),
        packed_backend_bytes=int(backend_bytes),
        packed_diffrax_wall_ms=float(packed_wall),
        explicit_real_wall_ms=float(real_wall),
        packed_pathwise_defect=float(packed_defect),
        finite=finite,
    )


def split_form_capacity_metrics(
    mode_count: int,
    /,
    *,
    pair_chunk_size: int,
) -> dict[str, int | float]:
    """Analytic pair/workspace evidence for the certified Fourier split route."""
    count = int(mode_count)
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),)
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    prepared = (
        phx.discretization.SpectralConservationMethodPlan(
            split_form=phx.discretization.SpectralSplitFormPlan(
                phx.discretization.EntropyConservativeEulerFluxPlan(),
                pair_chunk_size=int(pair_chunk_size),
            )
        )
        .prepare(space)
        .split_form
    )
    return {
        "mode_count": count,
        # ty: ignore[unresolved-attribute]
        "pair_count": prepared.report.pair_count,
        # ty: ignore[unresolved-attribute]
        "pair_chunk_size": prepared.report.pair_chunk_size,
        # ty: ignore[unresolved-attribute]
        "pair_workspace_bytes": prepared.report.pair_workspace_bytes,
        # ty: ignore[unresolved-attribute]
        "skew_sbp_defect": prepared.report.skew_sbp_defect,
    }


def _prepare_fourier_inputs(
    count: int, dtype: str
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, ...], tuple[float, ...]
]:
    """Round once on the host; directly evaluate the resulting Fourier series."""
    x = np.arange(count, dtype=np.float64) / count
    values = 0.25 + np.sin(2.0 * np.pi * x) + np.cos(np.pi * count * x) / count
    tangent = 0.5 * np.cos(4.0 * np.pi * x)
    if dtype == "complex64":
        values = values + 0.3j * np.cos(6.0 * np.pi * x)
        tangent = tangent + 0.2j * np.sin(2.0 * np.pi * x)
    host_dtype = np.dtype(dtype)
    host = np.asarray(values, dtype=host_dtype)
    direction = np.asarray(tangent, dtype=host_dtype)
    offset = np.asarray(0.173, dtype=np.float32)
    target = 3 * count // 2 + 1
    coefficients = np.fft.fft(host.astype(np.complex128)) / count
    tangent_coefficients = np.fft.fft(direction.astype(np.complex128)) / count
    frequencies = np.fft.fftfreq(count) * count
    output = np.empty(target, dtype=np.complex128)
    input_jvp = np.empty_like(output)
    phase_jvp = np.empty_like(output)
    # Only the host oracle uses dense Fourier evaluation, in bounded row chunks.
    # The even Nyquist coefficient splits equally between the two signed modes.
    for start in range(0, target, 32):
        stop = min(start + 32, target)
        queries = float(offset) + np.arange(start, stop, dtype=np.float64) / target
        angles = 2.0 * np.pi * queries[:, None] * frequencies[None, :]
        basis = np.exp(1j * angles)
        slope = basis * (2j * np.pi * frequencies[None, :])
        if count % 2 == 0:
            basis[:, count // 2] = np.cos(np.pi * count * queries)
            slope[:, count // 2] = -np.pi * count * np.sin(np.pi * count * queries)
        output[start:stop] = basis @ coefficients
        input_jvp[start:stop] = basis @ tangent_coefficients
        phase_jvp[start:stop] = slope @ coefficients
    if dtype == "float32":
        output, input_jvp, phase_jvp = output.real, input_jvp.real, phase_jvp.real
    # The target grid resolves every source mode: sum of the interpolant is
    # target*c_0, so reverse differentiation yields target/count at every input.
    total_gradient = np.full(count, target / count, dtype=np.complex128)
    if dtype == "float32":
        total_gradient = total_gradient.real
    # A priori normwise FFT envelope, not a fitted tolerance. Account for the
    # even-source split (count+1), both FFTs and differentiated phase arithmetic.
    # Cauchy-Schwarz propagates forward coefficient error through evaluation;
    # the inverse FFT term uses its unitary 2-norm scaling. A conservative
    # 32*eps per transform stage also covers non-power-of-two FFT arithmetic.
    eps = float(np.finfo(np.float32).eps)
    stages = ceil(log2(count + 1)) + ceil(log2(target)) + 4
    gamma = 32.0 * eps * stages
    norm = float(np.linalg.norm(host.astype(np.complex128)))
    tangent_norm = float(np.linalg.norm(direction.astype(np.complex128)))
    angular = 2.0 * np.pi * np.abs(frequencies)
    phase_roundoff = 16.0 * eps * (1.0 + abs(float(offset)))
    value_bound = gamma * (1.0 + np.sqrt(target / count)) * norm + phase_roundoff * float(
        angular @ np.abs(coefficients)
    )
    tangent_bound = gamma * (
        1.0 + np.sqrt(target / count)
    ) * tangent_norm + phase_roundoff * float(angular @ np.abs(tangent_coefficients))
    slope_bound = (
        gamma * norm * float(np.linalg.norm(angular)) / np.sqrt(count)
        + gamma * np.sqrt(target) * float(np.linalg.norm(angular * coefficients))
        + phase_roundoff * float((angular * angular) @ np.abs(coefficients))
    )
    gradient_bound = gamma * target / np.sqrt(count)
    return (
        (host, direction, offset),
        (output, input_jvp, phase_jvp, total_gradient),
        (
            float(value_bound),
            float(tangent_bound),
            float(slope_bound),
            float(gradient_bound),
        ),
    )


def run_fourier_resampling_benchmark(
    mode_count: int = 64,
    /,
    *,
    dtype: str = "float32",
    repeats: int = 5,
) -> dict[str, Any]:
    """Benchmark only the declared Fourier API, not the high-precision PDE compiler."""
    if mode_count < 8 or mode_count > 512 or repeats < 1:
        raise ValueError("Fourier mode_count must be in [8, 512] and repeats positive.")
    if dtype not in ("float32", "complex64"):
        raise ValueError("Fourier dtype must be float32 or complex64.")
    if not phx.config.x64_enabled():
        raise ValueError("Fourier qualification requires JAX_ENABLE_X64=1.")
    devices = jax.devices()
    if len(devices) != 1:
        raise ValueError("Select exactly one CPU or Metal device for this benchmark.")
    device = devices[0]
    prepared, preparation_seconds = measure_host(
        lambda: _prepare_fourier_inputs(mode_count, dtype)
    )
    host_inputs, expected, bounds = prepared
    arguments, transfer_seconds = measure_synchronized(
        lambda: tuple(jax.device_put(value, device) for value in host_inputs)
    )
    target = expected[0].size

    def kernel(values: Array, tangent: Array, offset: Array) -> tuple[Array, ...]:
        def resample(array: Array, shift: Array) -> Array:
            return fourier_resample(array, (target,), phase_offsets=(shift,))

        output, input_jvp = jax.jvp(
            lambda array: resample(array, offset), (values,), (tangent,)
        )
        _, phase_jvp = jax.jvp(
            lambda shift: resample(values, shift),
            (offset,),
            (jnp.ones_like(offset),),
        )
        total_gradient = jax.grad(
            lambda array: jnp.sum(jnp.real(resample(array, offset)))
        )(values)
        return output, input_jvp, phase_jvp, total_gradient

    compiled, timing = measure_lower_and_compile(
        lambda: jax.jit(kernel).lower(*arguments), lambda lowered: lowered.compile()
    )
    first, first_seconds = measure_synchronized(lambda: compiled(*arguments))
    result, warm = measure_repeated(
        lambda: compiled(*arguments), warmup=0, repeats=repeats
    )
    _, ready_sync_seconds = measure_host(lambda: synchronize(result))
    observed, readback_seconds = measure_host(lambda: jax.device_get(result))
    names = ("value", "input_jvp", "phase_jvp", "total_gradient")
    checks = {
        name: {
            "maximum_absolute_error": float(np.max(np.abs(actual - reference))),
            "absolute_bound": bound,
            "finite": bool(np.all(np.isfinite(actual))),
        }
        for name, actual, reference, bound in zip(
            names, observed, expected, bounds, strict=True
        )
    }
    dtype_valid = all(str(value.dtype) == dtype for value in (*arguments[:2], *result))
    placement_valid = all(
        value.devices() == {device} for value in (*arguments, *first, *result)
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-fourier-resample-and-derivatives",
        unavailable_reason="The selected compiler did not expose all cost/memory estimates.",
    )
    return {
        "scenario": "fourier-resample",
        "scope": "1D Fourier resampling, input/phase JVP and reverse total gradient; no PDE compiler claim",
        "mode_count": mode_count,
        "evaluation_count": target,
        "repeats": repeats,
        "input_dtype": str(arguments[0].dtype),
        "output_dtypes": [str(value.dtype) for value in result],
        "offset_dtype": str(arguments[2].dtype),
        "device": {
            "platform": device.platform,
            "platform_version": device.client.platform_version,
            "kind": device.device_kind,
            "id": device.id,
        },
        "dtype_valid": dtype_valid,
        "placement_valid": placement_valid,
        "host_preparation_and_reference_seconds": preparation_seconds,
        "host_to_device_and_sync_seconds": transfer_seconds,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "first_execution_and_sync_seconds": first_seconds,
        "warm_execution_and_sync": warm.to_seconds_dict(),
        "already_ready_sync_seconds": ready_sync_seconds,
        "device_to_host_seconds": readback_seconds,
        "compiler": {
            **asdict(evidence),
            "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
        },
        "logical_host_input_bytes": logical_array_bytes(host_inputs),
        "logical_device_input_bytes": logical_array_bytes(arguments),
        "logical_result_bytes": logical_array_bytes(result),
        "sampled_allocator_bytes": None,
        "sampled_process_rss_bytes": None,
        "sampled_memory_unavailable_reason": "Allocator and RSS were not sampled; logical and compiler bytes are not physical memory measurements.",
        "checks": checks,
        "passed": dtype_valid
        and placement_valid
        and all(
            check["finite"] and check["maximum_absolute_error"] <= check["absolute_bound"]
            for check in checks.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark strict spectral PDE execution or explicit FP32 Fourier resampling."
    )
    parser.add_argument("--mode-count", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--scenario",
        choices=("pde", "fourier-resample"),
        default="pde",
        help="The default PDE scenario retains its strict high-precision contract.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "complex64"),
        help="Fourier-only precision (default: float32); never downcasts the PDE plan.",
    )
    parser.add_argument(
        "--mode-counts",
        nargs="+",
        type=int,
        help="Fourier-only bounded capacity sweep (8 to 512 modes each).",
    )
    parser.add_argument(
        "--capacity-sweep",
        action="store_true",
        help="Fourier-only small/medium/large sweep: 16, 64, 256 modes.",
    )
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.scenario == "fourier-resample":
        if arguments.mode_counts is not None and arguments.capacity_sweep:
            parser.error("--mode-counts and --capacity-sweep are mutually exclusive.")
        counts = (
            arguments.mode_counts
            if arguments.mode_counts is not None
            else [16, 64, 256]
            if arguments.capacity_sweep
            else [arguments.mode_count]
        )
        counts = [16] if arguments.smoke else counts
        repeats = 1 if arguments.smoke else arguments.repeats
        if any(count < 8 or count > 512 for count in counts) or repeats < 1:
            parser.error("Fourier mode counts must be in [8, 512] and repeats positive.")
        records = [
            run_fourier_resampling_benchmark(
                count, dtype=arguments.dtype or "float32", repeats=repeats
            )
            for count in counts
        ]
        root = Path(__file__).resolve().parents[1]
        identity = capture_benchmark_identity(root, Path(__file__), records[0].keys())
        payload = {
            "benchmark_identity": identity.to_dict(),
            "environment": capture_environment().to_dict(),
            "records": records,
            "passed": all(record["passed"] for record in records),
        }
    else:
        if (
            arguments.mode_counts is not None
            or arguments.capacity_sweep
            or arguments.dtype is not None
        ):
            parser.error("Dtype/capacity selection requires --scenario fourier-resample.")
        count = 16 if arguments.smoke else arguments.mode_count
        repeats = 1 if arguments.smoke else arguments.repeats
        record = run_spectral_pde_benchmark(count, repeats=repeats)
        payload = {**asdict(record), "passed": record.passed}
    print(json.dumps(payload, indent=2, allow_nan=False))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
