#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated native exterior PDE execution on periodic Fourier complexes.

The capacity campaign changes both modal axes. Numerical plan leaves and field
coefficients are executable arguments, not closed-over numerical constants.
Cold admission/lowering/compilation is separated from prepared executable reuse.
Independent Fourier symbols gate the screened Laplacian and d-squared outputs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.stages import Compiled

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    synchronize,
)
from phydrax.discretization import AxisDomain
from phydrax.discretization.spectral import (
    FourierBasisPlan,
    FourierDeRhamComplex,
    TensorSpectralPlan,
)
from phydrax.equations import (
    compile_exterior_pde,
    CompiledExteriorPDE,
    PDECoordinate,
    PDEEquation,
    PDEExpression,
    PDEField,
    PDEProblemIR,
)
from phydrax.exterior import FormType, FormValueSpec


OUTPUT = Path(__file__).with_suffix(".json")


def _problem() -> PDEProblemIR:
    scalar = PDEExpression.field("u")
    derivative = scalar.exterior_derivative()
    return PDEProblemIR(
        coordinates=(PDECoordinate("x", "space"), PDECoordinate("y", "space")),
        fields=(
            PDEField(
                "u",
                coordinates=("x", "y"),
                form=FormValueSpec(FormType(2, 0), proxy="scalar"),
            ),
        ),
        equations=(
            PDEEquation("screened", derivative.codifferential() + scalar),
            PDEEquation("nilpotency", derivative.exterior_derivative()),
        ),
    )


def _realization(size: int, /) -> FourierDeRhamComplex:
    space = TensorSpectralPlan(
        (FourierBasisPlan(size), FourierBasisPlan(size)),
        axis_names=("x", "y"),
    ).prepare((AxisDomain.periodic(0.0, 2.0 * np.pi),) * 2)
    return FourierDeRhamComplex(space, nyquist_policy="zero-self-conjugate")


def _oracle_inputs(size: int, /) -> tuple[Array, Array, Array]:
    """Prepare independent integer Fourier symbols and active-coordinate ordering."""
    modes = np.rint(np.fft.fftfreq(size) * size).astype(np.int64)
    kx, ky = np.meshgrid(modes, modes, indexing="ij")
    active = np.ones((size, size), dtype=np.bool_)
    if size % 2 == 0:
        active[size // 2, :] = False
        active[:, size // 2] = False
    indices = np.flatnonzero(active.reshape(-1)).astype(np.int32)
    symbol = (1.0 + kx * kx + ky * ky).reshape(-1)[indices]
    coefficients = ((1.0 + 0.13 * kx) + 1j * (0.2 - 0.17 * ky)) / (
        1.0 + kx * kx + ky * ky
    )
    return (
        jnp.asarray(indices, dtype=jnp.int32),
        jnp.asarray(symbol, dtype=jnp.float64),
        jnp.asarray(coefficients.reshape(-1)[indices], dtype=jnp.complex128),
    )


def _cold(
    size: int,
    problem: PDEProblemIR,
    values: Mapping[str, Array],
    /,
) -> tuple[
    FourierDeRhamComplex,
    CompiledExteriorPDE,
    CompiledExteriorPDE,
    Compiled,
    dict[str, Array],
    dict[str, object],
]:
    realization, preparation_seconds = measure_synchronized(lambda: _realization(size))
    plan, ir_lowering_seconds = measure_synchronized(
        lambda: compile_exterior_pde(
            problem, realization, fields=values, boundary="absolute"
        )
    )
    numerical, static = eqx.partition(plan, eqx.is_array)

    def evaluate(
        dynamic_plan: CompiledExteriorPDE, fields: Mapping[str, Array]
    ) -> dict[str, Array]:
        return eqx.combine(dynamic_plan, static).residuals(fields)

    executable, compilation = measure_lower_and_compile(
        lambda: jax.jit(evaluate).lower(numerical, values),
        lambda lowered: lowered.compile(),
    )
    first, first_seconds = measure_synchronized(lambda: executable(numerical, values))
    evidence = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The selected backend did not provide compiler estimates.",
    )
    compiler = asdict(evidence)
    compiler["estimated_device_memory_bytes"] = evidence.estimated_device_memory_bytes
    record: dict[str, object] = {
        "realization_preparation_seconds": preparation_seconds,
        "ir_lowering_seconds": ir_lowering_seconds,
        "compilation": asdict(compilation),
        "first_execution_seconds": first_seconds,
        "compiler": compiler,
    }
    return realization, plan, numerical, executable, first, record


def _case(size: int, repeats: int, /) -> dict[str, object]:
    active_indices, symbol, initial = _oracle_inputs(size)
    problem = _problem()
    initial_fields = {"u": initial}
    synchronize((active_indices, symbol, initial_fields))
    cold, cold_seconds = measure_synchronized(
        lambda: _cold(size, problem, initial_fields)
    )
    realization, plan, numerical, executable, first, record = cold
    samples = tuple(
        {"u": (1.0 + 0.1 * (index + 1)) * initial + (index + 1) * (0.03 - 0.02j)}
        for index in range(repeats)
    )
    synchronize(samples)
    stream = iter(samples)
    final, steady = measure_repeated(
        lambda: executable(numerical, next(stream)), warmup=0, repeats=repeats
    )
    first_error = jnp.max(jnp.abs(first["screened"] - symbol * initial))
    final_error = jnp.max(jnp.abs(final["screened"] - symbol * samples[-1]["u"]))
    nilpotency_error = jnp.maximum(
        jnp.max(jnp.abs(first["nilpotency"])),
        jnp.max(jnp.abs(final["nilpotency"])),
    )
    output_change = jnp.max(jnp.abs(final["screened"] - first["screened"]))
    ordering_matches = jnp.array_equal(realization.active_modes, active_indices)
    tolerance = 1.0e-10
    error_values = {
        "initial_screened_absolute_error": float(first_error),
        "updated_screened_absolute_error": float(final_error),
        "nilpotency_absolute_error": float(nilpotency_error),
    }
    finite = all(np.isfinite(value) for value in error_values.values())
    return {
        "modal_shape": [size, size],
        "modal_capacity": size * size,
        "active_mode_count": active_indices.size,
        "degree_coordinate_counts": list(realization.cell_counts),
        "realization_id": realization.realization_id,
        "compilation_id": plan.compilation_id,
        "boundary": "absolute",
        "nyquist_policy": "zero-self-conjugate",
        "cold_end_to_end_seconds": cold_seconds,
        **record,
        "prepared_reuse": steady.to_seconds_dict(),
        "prepared_reuse_changes_inputs": True,
        "retained_plan_array_bytes": logical_array_bytes(plan),
        "retained_realization_array_bytes": logical_array_bytes(realization),
        "retained_union_array_bytes": logical_array_bytes((realization, plan)),
        "logical_input_bytes": logical_array_bytes((numerical, initial_fields)),
        "logical_output_bytes": logical_array_bytes(final),
        "absolute_tolerance": tolerance,
        "errors": error_values,
        "active_ordering_matches_oracle": bool(ordering_matches),
        "updated_output_maximum_change": float(output_change),
        "passed": finite
        and bool(ordering_matches)
        and all(value <= tolerance for value in error_values.values())
        and bool(jnp.isfinite(output_change) & (output_change > tolerance)),
    }


def run(sizes: tuple[int, ...] = (8, 16, 32), repeats: int = 5, /) -> dict[str, object]:
    if not sizes or len(set(sizes)) != len(sizes) or any(size < 3 for size in sizes):
        raise ValueError("sizes must contain unique modal capacities of at least three.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if not jax.config.x64_enabled:
        raise ValueError("The exterior-calculus benchmark requires JAX_ENABLE_X64=1.")
    cases = [_case(size, repeats) for size in sizes]
    return {
        "capability": "platform.exterior-calculus",
        "environment": asdict(capture_environment()),
        "controlling_capacity": "two-dimensional Fourier modal shape",
        "sizes": list(sizes),
        "repeats": repeats,
        "cold_scope": "realization preparation, IR lowering, executable lowering/compilation, first synchronized execution",
        "prepared_scope": "same executable and numerical plan, changing coefficient arguments",
        "retained_memory_scope": "unique logical array payloads; excludes Python metadata and allocator overhead",
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    report = run(tuple(arguments.sizes), arguments.repeats)
    write_json_atomic(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
