#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class OperatorSpectralResidualBenchmarkRecord:
    mode_count: int
    closure_count: int
    poisson_residual_energy: float
    poisson_gradient_norm: float
    mixed_basis_residual_energy: float
    retained_high_mode_energy: float
    full_high_mode_energy: float
    expected_full_high_mode_energy: float
    first_jit_ms: float
    steady_ms: float
    finite: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.finite
            and self.poisson_residual_energy < 1e-18
            and self.poisson_gradient_norm < 1e-8
            and self.mixed_basis_residual_energy < 1e-18
            and self.retained_high_mode_energy < 1e-18
            and abs(self.full_high_mode_energy - self.expected_full_high_mode_energy)
            < 1e-10
        )


def _space(count: int):
    return phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="u",
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _poisson_problem():
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    field = phx.equations.PDEField("u", coordinates=("x",))
    source = phx.equations.PDEParameter("source", functional=True)
    u = phx.equations.PDEExpression.field("u")
    forcing = phx.equations.PDEExpression.parameter("source")
    return phx.equations.PDEProblemIR(
        (x,),
        (field,),
        parameters=(source,),
        equations=(
            phx.equations.PDEEquation(
                "poisson",
                -u.laplacian("x"),
                forcing,
            ),
        ),
    )


def _quadratic_problem():
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    field = phx.equations.PDEField("u", coordinates=("x",))
    u = phx.equations.PDEExpression.field("u")
    return phx.equations.PDEProblemIR(
        (x,),
        (field,),
        equations=(
            phx.equations.PDEEquation(
                "quadratic",
                u * u,
                phx.equations.PDEExpression.constant(0.5),
            ),
        ),
    )


def _mixed_space_time_problem():
    t = phx.equations.PDECoordinate("t", "time", bounds=(0.0, 1.0))
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    field = phx.equations.PDEField("u", coordinates=("t", "x"))
    u = phx.equations.PDEExpression.field("u")
    return phx.equations.PDEProblemIR(
        (t, x),
        (field,),
        equations=(
            phx.equations.PDEEquation(
                "unit-rate",
                u.derivative("t"),
                phx.equations.PDEExpression.constant(1.0),
            ),
        ),
    )


def _measure(function, argument, repeats: int):
    compiled = jax.jit(function)
    start = time.perf_counter()
    value = jax.block_until_ready(compiled(argument))
    first = 1e3 * (time.perf_counter() - start)
    start = time.perf_counter()
    for _ in range(repeats):
        value = jax.block_until_ready(compiled(argument))
    steady = 1e3 * (time.perf_counter() - start) / repeats
    return value, first, steady


def run_operator_spectral_residual_benchmark(
    mode_count: int = 12,
    /,
    *,
    repeats: int = 5,
) -> OperatorSpectralResidualBenchmarkRecord:
    count = int(mode_count)
    if count < 8:
        raise ValueError("mode_count must be at least eight.")
    space = _space(count)
    x = space.axes[0].nodes

    poisson = phx.equations.compile_spectral_residual(
        _poisson_problem(),
        space,
        phx.discretization.PseudospectralMethodPlan(),
        parameter_values={"source": jnp.zeros(space.physical_shape)},
    )
    exact = jnp.sin(2.0 * jnp.pi * x)
    source = (2.0 * jnp.pi) ** 2 * exact
    state = poisson.project_state(exact)
    args = {"source": source}
    poisson_energy = poisson.residual_energy(state, args)
    gradient = jax.grad(lambda value: poisson.residual_energy(value, args))(state)
    gradient_norm = jnp.sqrt(jnp.real(jnp.vdot(gradient, gradient)))
    _, first, steady = _measure(
        lambda value: poisson.residual_energy(value, args),
        state,
        repeats,
    )
    mixed_space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.ChebyshevBasisPlan(6),
            phx.discretization.FourierBasisPlan(count),
        ),
        axis_names=("t", "x"),
        field_name="u",
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    mixed = phx.equations.compile_spectral_residual(
        _mixed_space_time_problem(),
        mixed_space,
        phx.discretization.PseudospectralMethodPlan(),
    )
    mixed_values = (
        mixed_space.axes[0].nodes[:, None]
        + jnp.sin(2.0 * jnp.pi * mixed_space.axes[1].nodes)[None, :]
    )
    mixed_energy = mixed.residual_energy(mixed.project_state(mixed_values))

    wave_number = count // 2 - 1
    high_mode = jnp.sin(2.0 * jnp.pi * wave_number * x)
    high_state = space.project(high_mode)
    retained = phx.equations.compile_spectral_residual(
        _quadratic_problem(),
        space,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2),
        ),
        scope="retained",
    )
    full = phx.equations.compile_spectral_residual(
        _quadratic_problem(),
        space,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PolynomialClosureDealiasingPlan(2),
        ),
    )
    retained_energy = retained.residual_energy(high_state)
    full_energy = full.residual_energy(high_state)
    expected = jnp.asarray(0.125, dtype=full_energy.dtype)
    values = jnp.asarray(
        (
            poisson_energy,
            gradient_norm,
            mixed_energy,
            retained_energy,
            full_energy,
            expected,
        )
    )
    finite = bool(jnp.all(jnp.isfinite(values)))
    return OperatorSpectralResidualBenchmarkRecord(
        mode_count=count,
        closure_count=full.evaluation.num_modes,
        poisson_residual_energy=float(poisson_energy),
        poisson_gradient_norm=float(gradient_norm),
        mixed_basis_residual_energy=float(mixed_energy),
        retained_high_mode_energy=float(retained_energy),
        full_high_mode_energy=float(full_energy),
        expected_full_high_mode_energy=float(expected),
        first_jit_ms=float(first),
        steady_ms=float(steady),
        finite=finite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify all-coordinate spectral operator residuals."
    )
    parser.add_argument("--mode-count", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    count = 8 if arguments.smoke else arguments.mode_count
    repeats = 1 if arguments.smoke else arguments.repeats
    record = run_operator_spectral_residual_benchmark(count, repeats=repeats)
    payload = json.dumps({**asdict(record), "passed": record.passed}, indent=2)
    if arguments.output is not None:
        target = arguments.output
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, target)
    print(payload)
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
