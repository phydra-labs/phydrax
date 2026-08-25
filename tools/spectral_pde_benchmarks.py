#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp

import phydrax as phx


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
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.maximum_drift_error <= 1e-9
            and self.parameter_gradient_error <= 1e-8
            and self.alias_defect <= 1e-9
        )


def _problem():
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


def _measure(function, argument, repeats):
    compiled = jax.jit(function)
    started = time.perf_counter()
    value = compiled(argument)
    jax.block_until_ready(value)
    first = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    for _ in range(repeats):
        value = compiled(argument)
        jax.block_until_ready(value)
    steady = 1e3 * (time.perf_counter() - started) / repeats
    return value, first, steady


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
    ).prepare(jnp.asarray([[0.0], [1.0]]))
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

    def drift(arguments):
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
    finite = bool(
        jnp.all(jnp.isfinite(actual))
        & jnp.isfinite(maximum_error)
        & jnp.isfinite(gradient_error)
        & jnp.isfinite(alias_defect)
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
        finite=finite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark coefficient-resident dealiased spectral PDE execution."
    )
    parser.add_argument("--mode-count", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    count = 16 if arguments.smoke else arguments.mode_count
    repeats = 1 if arguments.smoke else arguments.repeats
    record = run_spectral_pde_benchmark(count, repeats=repeats)
    print(json.dumps({**asdict(record), "passed": record.passed}, indent=2))
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
