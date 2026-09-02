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
from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized


@dataclass(frozen=True)
class IncompressibleSpectralBenchmarkRecord:
    """Raw smoke, resource, invariant, and informational timing evidence."""

    kind: str
    periodic: dict[str, Any]
    channel: dict[str, Any]
    finite: bool
    smoke_successful: bool


def _measure(function, argument, repeats):
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
        1_000.0 * float(distribution.mean_seconds),
    )


def _execution_record(value: jax.Array) -> dict[str, object]:
    device = value.device
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "backend": jax.default_backend(),
        "device_platform": device.platform,
        "device_kind": device.device_kind,
    }


def _channel_resource_record(report) -> dict[str, object]:
    return {
        "lower_bandwidth": int(report.lower_bandwidth),
        "upper_bandwidth": int(report.upper_bandwidth),
        "horizontal_batch_size": int(report.horizontal_batch_size),
        "correction_rank": int(report.correction_rank),
        "constraint_rank": int(report.constraint_rank),
        "shared_basis_bytes": int(report.shared_basis_bytes),
        "operator_bytes": int(report.operator_bytes),
        "factor_bytes": int(report.factor_bytes),
        "workspace_bytes": int(report.workspace_bytes),
        "persistent_bytes": int(report.persistent_bytes),
        "preparation_bytes": int(report.preparation_bytes),
        "pivot_margin": float(report.pivot_margin),
        "requires_unsharded_axis": bool(report.requires_unsharded_axis),
        "required_unsharded_axes": list(report.required_unsharded_axes),
    }


def run_incompressible_spectral_benchmark(
    periodic_mode_count: int = 32,
    channel_shape: tuple[int, int, int] = (8, 16, 8),
    /,
    *,
    repeats: int = 5,
) -> IncompressibleSpectralBenchmarkRecord:
    """Exercise both spectral routes without interpreting scientific accuracy.

    Timings are deliberately informational. ``smoke_successful`` only reports
    finite execution and the solvers' own success states; scientific thresholds
    belong to ``incompressible_flow_qualification.py``.
    """

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
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
        )
    )
    periodic_problem = phx.equations.IncompressibleFlowProblem(2, 1e-2)
    periodic = phx.equations.compile_periodic_incompressible_flow(
        periodic_problem,
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
    periodic_diagnostics = periodic.diagnostics(0.0, periodic_state)
    storage = packed_dns_storage_metrics(periodic_space, component_count=2)
    periodic_finite = bool(
        jnp.all(jnp.isfinite(periodic_rate))
        & jnp.isfinite(periodic_divergence)
        & jnp.isfinite(periodic_diagnostics.nonlinear_energy_rate)
        & jnp.isfinite(periodic_diagnostics.energy_balance_defect)
    )

    channel_space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(nx),
            phx.discretization.ChebyshevBasisPlan(ny),
            phx.discretization.FourierBasisPlan(nz),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            phx.discretization.AxisDomain.interval(-1.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
        )
    )
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
    jax.block_until_ready(
        prescribed.factorization.factors
        if prescribed.factorization is not None
        else prescribed.ultraspherical.bulk_influence
    )
    channel_prepare = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    prescribed_result = prescribed.solve(couette_modal)
    jax.block_until_ready(prescribed_result.velocity)
    channel_solve = 1e3 * (time.perf_counter() - started)
    couette_error = jnp.max(
        jnp.abs(channel_space.reconstruct(prescribed_result.velocity) - couette)
    )
    fixed_flux_plan = phx.discretization.ChannelStokesPlan(
        channel_space,
        0.1,
        mean_constraint=phx.discretization.ChannelMeanConstraint("bulk_flux", (0.4, 0.0)),
    )
    fixed_flux = fixed_flux_plan.prepare(1.0)
    flux_result = fixed_flux.solve(jnp.zeros_like(couette_modal))
    flux_error = jnp.max(
        jnp.abs(flux_result.diagnostics.bulk_velocity - jnp.asarray([0.4, 0.0]))
    )
    channel_problem = phx.equations.IncompressibleFlowProblem(3, 0.1)
    channel = phx.equations.compile_channel_flow(
        channel_problem,
        prescribed_plan,
        method,
    )
    sbdf = phx.solver.solve_channel_sbdf2(
        channel,
        channel.project_state(couette),
        jnp.asarray([0.0, 0.01, 0.02]),
    )
    sbdf_error = jnp.max(jnp.abs(channel.reconstruct_state(sbdf.velocity[-1]) - couette))
    channel_finite = bool(
        jnp.isfinite(couette_error)
        & jnp.isfinite(flux_error)
        & jnp.isfinite(sbdf_error)
        & jnp.isfinite(prescribed_result.diagnostics.divergence_norm)
        & jnp.isfinite(prescribed_result.diagnostics.momentum_constraint_residual)
        & jnp.isfinite(prescribed_result.diagnostics.wall_residual)
        & jnp.isfinite(prescribed_result.diagnostics.pressure_gauge_residual)
    )
    channel_successful = bool(
        prescribed_result.successful & flux_result.successful & sbdf.successful
    )

    periodic_record = {
        "route": periodic.resolved_method,
        "problem_id": periodic_problem.problem_id,
        "discretization_plan_id": periodic_space.plan_id,
        "discretization_prepared_id": periodic_space.prepared_id,
        "method_id": method.method_id,
        "method_prepared_id": periodic.spatial_method.prepared_id,
        "projector_id": periodic.projector.projector_id,
        "compilation_id": periodic.compilation_id,
        "execution": _execution_record(periodic_state),
        "storage": storage,
        "raw_invariants": {
            "divergence_norm": float(periodic_divergence),
            "nonlinear_energy_rate": float(periodic_diagnostics.nonlinear_energy_rate),
            "energy_balance_defect": float(periodic_diagnostics.energy_balance_defect),
            "finite": periodic_finite,
        },
        "informational_timings_ms": {
            "first_jit": float(first_jit),
            "steady_mean": float(steady),
            "repeats": repeat_count,
        },
    }
    channel_report = prescribed.report
    channel_record = {
        "route": prescribed_plan.route,
        "problem_id": channel_problem.problem_id,
        "discretization_plan_id": channel_space.plan_id,
        "discretization_prepared_id": channel_space.prepared_id,
        "method_id": method.method_id,
        "method_prepared_id": channel.spatial_method.prepared_id,
        "stokes_plan_id": prescribed_plan.plan_id,
        "stokes_prepared_id": prescribed.prepared_id,
        "preparation_report_id": channel_report.report_id,
        "compilation_id": channel.compilation_id,
        "execution": _execution_record(couette_modal),
        "resources": _channel_resource_record(channel_report),
        "raw_invariants": {
            "couette_maximum_error": float(couette_error),
            "fixed_flux_error": float(flux_error),
            "divergence_norm": float(prescribed_result.diagnostics.divergence_norm),
            "momentum_constraint_residual": float(
                prescribed_result.diagnostics.momentum_constraint_residual
            ),
            "wall_residual": float(prescribed_result.diagnostics.wall_residual),
            "pressure_gauge_residual": float(
                prescribed_result.diagnostics.pressure_gauge_residual
            ),
            "sbdf2_state_error": float(sbdf_error),
            "finite": channel_finite,
            "solver_successful": channel_successful,
        },
        "informational_timings_ms": {
            "prepare": float(channel_prepare),
            "solve": float(channel_solve),
        },
    }
    finite = periodic_finite and channel_finite
    return IncompressibleSpectralBenchmarkRecord(
        kind="incompressible-spectral-smoke-performance",
        periodic=periodic_record,
        channel=channel_record,
        finite=finite,
        smoke_successful=finite and channel_successful,
    )


def packed_dns_storage_metrics(
    discretization: phx.discretization.TensorSpectralDiscretization,
    /,
    *,
    component_count: int = 3,
) -> dict[str, int | float]:
    """Analytic persistent/checkpoint storage counters for the Hermitian chart."""

    coordinates = phx.discretization.HermitianSpectralCoordinates(
        discretization,
        component_shape=(int(component_count),),
    )
    return {
        "full_complex_bytes": coordinates.full_state_bytes,
        "packed_independent_real_bytes": coordinates.coordinate_state_bytes,
        "fixed_mode_count": coordinates.fixed_mode_count,
        "conjugate_pair_count": coordinates.conjugate_pair_count,
        "packed_to_full_ratio": (
            coordinates.coordinate_state_bytes / coordinates.full_state_bytes
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record raw smoke, resource, invariant, and informational timing evidence "
            "for incompressible periodic and channel spectral workflows."
        )
    )
    parser.add_argument("--periodic-mode-count", type=int, default=32)
    parser.add_argument("--channel-shape", type=int, nargs=3, default=(8, 16, 8))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    periodic_count = 8 if arguments.smoke else arguments.periodic_mode_count
    channel = (4, 8, 4) if arguments.smoke else tuple(arguments.channel_shape)
    repeats = 1 if arguments.smoke else arguments.repeats
    record = run_incompressible_spectral_benchmark(
        periodic_count,
        channel,
        repeats=repeats,
    )
    payload = json.dumps(asdict(record), indent=2, sort_keys=True)
    if arguments.output is not None:
        target = arguments.output
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, target)
    print(payload)
    if not record.smoke_successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
