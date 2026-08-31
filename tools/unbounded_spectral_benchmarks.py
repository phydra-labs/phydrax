"""Deterministic qualification for unbounded spectral and eigen evidence paths."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class UnboundedSpectralBenchmarkRecord:
    mode_count: int
    refined_mode_count: int
    line_prepared_id: str
    transform_error: float
    derivative_error: float
    half_line_quadrature_error: float
    derivative_closure_residual: float
    modal_tail_ratio: float
    first_jit_ms: float
    steady_wall_ms: float
    oscillator_first_eigenvalue_error: float
    oscillator_trusted_modes: int
    resolvent_amplification: float
    polynomial_residual: float
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.transform_error < 1e-9
            and self.derivative_error < 1e-8
            and self.half_line_quadrature_error < 1e-5
            and self.modal_tail_ratio < 1e-8
            and self.oscillator_first_eigenvalue_error < 5e-3
            and self.oscillator_trusted_modes >= 1
            and self.resolvent_amplification > 10.0
            and self.polynomial_residual < 1e-8
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


def _oscillator(mode_count):
    domain = phx.discretization.AxisDomain.real_line()
    basis = phx.discretization.ConstrainedBasisPlan(
        phx.discretization.RationalChebyshevLineBasisPlan(mode_count, 4.0),
        phx.discretization.SpectralBoundaryConditionPlan.decay(),
    )
    space = phx.discretization.TensorSpectralPlan((basis,)).prepare((domain,))
    second = phx.discretization.spectral_derivative_operator(space, 0, 2).operator
    nodes = space.axes[0].nodes
    potential = phx.linalg.FunctionLinearOperator(
        lambda coefficients: space.project(nodes**2 * space.reconstruct(coefficients)),
        source=space.modal_space.vector_space,
        target=space.modal_space.vector_space,
        operator_id=f"benchmark-oscillator-potential:{space.prepared_id}",
    )
    result = phx.linalg.eigen.general_eigensolve(
        phx.linalg.eigen.GeneralEigenproblem(-second + potential)
    )
    return space, result


def run_unbounded_spectral_benchmark(
    mode_count: int = 16,
    /,
    *,
    repeats: int = 5,
) -> UnboundedSpectralBenchmarkRecord:
    count = int(mode_count)
    repeat_count = int(repeats)
    if count < 8 or repeat_count <= 0:
        raise ValueError("mode_count must be at least eight and repeats positive.")
    refined_count = count + max(4, count // 2)
    line = phx.discretization.TensorSpectralPlan(
        (phx.discretization.RationalChebyshevLineBasisPlan(count, 1.0),)
    ).prepare((phx.discretization.AxisDomain.real_line(),))
    nodes = line.axes[0].nodes
    values = 1.0 / (1.0 + nodes**2)
    coefficients = line.project(values)
    derivative, first, steady = _measure(
        lambda modal: line.derivative_values(modal, axis=0),
        coefficients,
        repeat_count,
    )
    transform_error = jnp.max(jnp.abs(line.reconstruct(coefficients) - values))
    derivative_error = jnp.max(jnp.abs(derivative + 2.0 * nodes / (1.0 + nodes**2) ** 2))
    decay = (
        phx.discretization.SpectralModalDiagnosticsPlan(line)
        .prepare()
        .evaluate(coefficients)
    )

    half_line = phx.discretization.TensorSpectralPlan(
        (phx.discretization.RationalChebyshevHalfLineBasisPlan(count, 2.0),)
    ).prepare((phx.discretization.AxisDomain.half_line(0.0),))
    quadrature = jnp.sum(half_line.quadrature_weights * jnp.exp(-half_line.axes[0].nodes))

    coarse_space, coarse = _oscillator(count)
    fine_space, fine = _oscillator(refined_count)
    transfer = phx.discretization.prepare_spectral_modal_transfer(
        coarse_space,
        fine_space,
    )
    evidence = phx.discretization.compare_spectral_eigen_resolutions(
        coarse,
        fine,
        coarse_space,
        fine_space,
        transfer,
        policy=phx.discretization.SpectralEigenResolutionPolicy(
            phx.linalg.eigen.GeneralEigenResolutionPolicy(
                chordal_tolerance=1e-5,
                normalized_drift_tolerance=0.1,
            ),
            subspace_tolerance=1e-2,
        ),
    )
    finite_values = jnp.sort(jnp.real(coarse.eigenvalues[coarse.finite_mask]))
    oscillator_error = jnp.abs(finite_values[0] - 1.0)

    jordan = phx.linalg.DenseLinearOperator(jnp.asarray([[0.0, 20.0], [0.0, 0.0]]))
    resolvent = phx.linalg.eigen.resolvent_scan(
        phx.linalg.eigen.ResolventScanProblem(jordan, jnp.asarray([1.0 + 0.0j]))
    )
    polynomial = phx.linalg.eigen.polynomial_eigensolve(
        phx.linalg.eigen.PolynomialEigenproblem(
            (
                phx.linalg.DenseLinearOperator(jnp.asarray([[-1.0]])),
                phx.linalg.DenseLinearOperator(jnp.asarray([[0.0]])),
                phx.linalg.DenseLinearOperator(jnp.asarray([[1.0]])),
            )
        )
    )
    scalars = jnp.asarray(
        [
            transform_error,
            derivative_error,
            jnp.abs(quadrature - 1.0),
            decay.relative_tail_norms[0],
            line.axes[0].derivative_residual,
            oscillator_error,
            resolvent.resolvent_norms[0],
            jnp.max(polynomial.diagnostics.original_relative_residuals),
        ]
    )
    return UnboundedSpectralBenchmarkRecord(
        mode_count=count,
        refined_mode_count=refined_count,
        line_prepared_id=line.prepared_id,
        transform_error=float(transform_error),
        derivative_error=float(derivative_error),
        half_line_quadrature_error=float(jnp.abs(quadrature - 1.0)),
        derivative_closure_residual=line.axes[0].derivative_residual,
        modal_tail_ratio=float(decay.relative_tail_norms[0]),
        first_jit_ms=first,
        steady_wall_ms=steady,
        oscillator_first_eigenvalue_error=float(oscillator_error),
        oscillator_trusted_modes=int(evidence.trusted_count),
        resolvent_amplification=float(resolvent.resolvent_norms[0]),
        polynomial_residual=float(
            jnp.max(polynomial.diagnostics.original_relative_residuals)
        ),
        finite=bool(jnp.all(jnp.isfinite(scalars))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-count", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/unbounded_spectral.json"),
    )
    arguments = parser.parse_args()
    record = run_unbounded_spectral_benchmark(
        arguments.mode_count,
        repeats=arguments.repeats,
    )
    payload = {**asdict(record), "passed": record.passed}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
