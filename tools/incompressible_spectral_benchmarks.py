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
class IncompressibleSpectralBenchmarkRecord:
    periodic_mode_count: int
    channel_shape: tuple[int, int, int]
    periodic_first_jit_ms: float
    periodic_steady_ms: float
    periodic_divergence_norm: float
    periodic_finite: bool
    channel_prepare_ms: float
    channel_solve_ms: float
    channel_factor_bytes: int
    couette_maximum_error: float
    fixed_flux_error: float
    channel_divergence_norm: float
    channel_sbdf2_error: float
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.periodic_finite
            and self.periodic_divergence_norm <= 1e-9
            and self.couette_maximum_error <= 1e-9
            and self.fixed_flux_error <= 1e-9
            and self.channel_divergence_norm <= 1e-9
            and self.channel_sbdf2_error <= 1e-8
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


def run_incompressible_spectral_benchmark(
    periodic_mode_count: int = 32,
    channel_shape: tuple[int, int, int] = (8, 16, 8),
    /,
    *,
    repeats: int = 5,
) -> IncompressibleSpectralBenchmarkRecord:
    count = int(periodic_mode_count)
    nx, ny, nz = (int(value) for value in channel_shape)
    repeat_count = int(repeats)
    if count < 8 or min(nx, ny, nz) < 4 or repeat_count < 1:
        raise ValueError(
            "Benchmark resolutions and repeats must be positive and nontrivial."
        )
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    periodic_space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(count),
            phx.discretization.FourierBasisPlan(count),
        ),
        axis_names=("x", "y"),
        field_name="velocity",
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    periodic = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 1e-2),
        periodic_space,
        method,
    )
    x, y = jnp.meshgrid(
        periodic_space.axes[0].nodes,
        periodic_space.axes[1].nodes,
        indexing="ij",
    )
    periodic_state = periodic.project_state(
        jnp.stack(
            (jnp.sin(2.0 * jnp.pi * y), jnp.sin(2.0 * jnp.pi * x)),
            axis=-1,
        )
    )
    periodic_rate, first_jit, steady = _measure(
        lambda state: periodic(0.0, state, None),
        periodic_state,
        repeat_count,
    )
    periodic_divergence = periodic.projector.divergence_norm(periodic_rate)

    channel_space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(nx),
            phx.discretization.ChebyshevBasisPlan(ny),
            phx.discretization.FourierBasisPlan(nz),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(jnp.asarray([[0.0, -1.0, 0.0], [2.0 * jnp.pi, 1.0, 2.0 * jnp.pi]]))
    y_channel = channel_space.axes[1].nodes
    couette = (
        jnp.zeros(channel_space.physical_shape + (3,))
        .at[..., 0]
        .set(y_channel[None, :, None])
    )
    couette_modal = channel_space.project(couette)
    started = time.perf_counter()
    prescribed_plan = phx.discretization.ChannelStokesPlan(
        channel_space,
        0.1,
        lower_wall_velocity=(-1.0, 0.0, 0.0),
        upper_wall_velocity=(1.0, 0.0, 0.0),
    )
    prescribed = prescribed_plan.prepare(1.0)
    jax.block_until_ready(prescribed.factorization.factors)
    channel_prepare = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    prescribed_result = prescribed.solve(couette_modal)
    jax.block_until_ready(prescribed_result.velocity)
    channel_solve = 1e3 * (time.perf_counter() - started)
    couette_error = jnp.max(
        jnp.abs(channel_space.reconstruct(prescribed_result.velocity) - couette)
    )
    fixed_flux = phx.discretization.ChannelStokesPlan(
        channel_space,
        0.1,
        mean_constraint=phx.discretization.ChannelMeanConstraint("bulk_flux", (0.4, 0.0)),
    ).prepare(1.0)
    flux_result = fixed_flux.solve(jnp.zeros_like(couette_modal))
    flux_error = jnp.max(
        jnp.abs(flux_result.diagnostics.bulk_velocity - jnp.asarray([0.4, 0.0]))
    )
    channel = phx.equations.compile_channel_flow(prescribed_plan, method)
    sbdf = phx.solver.solve_channel_sbdf2(
        channel,
        channel.project_state(couette),
        jnp.asarray([0.0, 0.01, 0.02]),
    )
    sbdf_error = jnp.max(jnp.abs(channel.reconstruct_state(sbdf.velocity[-1]) - couette))
    factor_bytes = int(prescribed.blocks.nbytes + prescribed.factorization.factors.nbytes)
    finite = bool(
        jnp.all(jnp.isfinite(periodic_rate))
        & jnp.isfinite(periodic_divergence)
        & jnp.isfinite(couette_error)
        & jnp.isfinite(flux_error)
        & jnp.isfinite(sbdf_error)
        & prescribed_result.successful
        & flux_result.successful
        & sbdf.successful
    )
    return IncompressibleSpectralBenchmarkRecord(
        periodic_mode_count=count,
        channel_shape=(nx, ny, nz),
        periodic_first_jit_ms=float(first_jit),
        periodic_steady_ms=float(steady),
        periodic_divergence_norm=float(periodic_divergence),
        periodic_finite=bool(jnp.all(jnp.isfinite(periodic_rate))),
        channel_prepare_ms=float(channel_prepare),
        channel_solve_ms=float(channel_solve),
        channel_factor_bytes=factor_bytes,
        couette_maximum_error=float(couette_error),
        fixed_flux_error=float(flux_error),
        channel_divergence_norm=float(prescribed_result.diagnostics.divergence_norm),
        channel_sbdf2_error=float(sbdf_error),
        finite=finite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark incompressible periodic and channel spectral workflows."
    )
    parser.add_argument("--periodic-mode-count", type=int, default=32)
    parser.add_argument("--channel-shape", type=int, nargs=3, default=(8, 16, 8))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    periodic_count = 8 if arguments.smoke else arguments.periodic_mode_count
    channel = (4, 8, 4) if arguments.smoke else tuple(arguments.channel_shape)
    repeats = 1 if arguments.smoke else arguments.repeats
    record = run_incompressible_spectral_benchmark(
        periodic_count,
        channel,
        repeats=repeats,
    )
    print(json.dumps({**asdict(record), "passed": record.passed}, indent=2))
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
