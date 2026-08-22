#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _timed(function, repeats):
    started = time.perf_counter()
    value = None
    for _ in range(repeats):
        value = function()
        jax.block_until_ready(value)
    return value, (time.perf_counter() - started) / repeats


def _periodic_grid(points):
    return phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                points,
                periodic=True,
                endpoint=False,
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _finite_difference(points, repeats):
    grid = _periodic_grid(points)
    request = phx.discretization.DerivativeRequest(
        "dxx",
        grid,
        "x",
        derivative_order=2,
        accuracy_order=4,
    )
    started = time.perf_counter()
    prepared = phx.discretization.FiniteDifferencePlan(grid, (request,)).prepare()
    preparation = time.perf_counter() - started
    values = jnp.sin(2.0 * jnp.pi * grid.axes[0].nodes)
    action = eqx.filter_jit(prepared.operator("dxx").mv)
    action(values).block_until_ready()
    result, steady = _timed(lambda: action(values), repeats)
    return {
        "preparation_seconds": preparation,
        "steady_action_seconds": steady,
        "maximum_absolute_value": float(jnp.max(jnp.abs(result))),
        "unique_coefficient_plans": len(
            prepared.stencil("dxx").stencil.coefficient_plans
        ),
        "stencil_width": int(prepared.stencil("dxx").stencil.indices.shape[1]),
    }


def _execution_pipeline(points, repeats):
    grid = _periodic_grid(points)
    request = phx.discretization.DerivativeRequest(
        "dxx",
        grid,
        "x",
        derivative_order=2,
        accuracy_order=4,
    )
    discretization = phx.discretization.FiniteDifferencePlan(
        grid,
        (request,),
    ).prepare()
    canonical = discretization.operator("dxx")
    lowered = phx.discretization.lower_stencil_operator(canonical)
    values = jnp.sin(2.0 * jnp.pi * grid.axes[0].nodes)
    canonical_action = eqx.filter_jit(canonical.mv)
    lowered_action = eqx.filter_jit(lowered.mv)
    canonical_action(values).block_until_ready()
    lowered_action(values).block_until_ready()
    _, canonical_seconds = _timed(lambda: canonical_action(values), repeats)
    _, lowered_seconds = _timed(lambda: lowered_action(values), repeats)
    pipeline = phx.discretization.StencilProgramPlan(
        discretization,
        ("u", "v"),
        (
            phx.discretization.StencilAssignment("u", "v", "dxx"),
            phx.discretization.StencilAssignment("u", "v", "dxx", scale=2.0),
        ),
    ).prepare()
    pipeline_action = eqx.filter_jit(pipeline)
    state = {"u": jnp.zeros_like(values), "v": values}
    pipeline_action(state)["u"].block_until_ready()
    _, pipeline_seconds = _timed(lambda: pipeline_action(state)["u"], repeats)
    schedule = phx.discretization.DistributedHaloSchedule(
        (points,),
        (1,),
        discretization.halo_plan,
        periodic_axes=(True,),
    )
    blocks = values.reshape((1, points))
    halo_action = eqx.filter_jit(schedule.exchange_reference)
    halo_action(blocks).block_until_ready()
    _, halo_seconds = _timed(lambda: halo_action(blocks), repeats)
    return {
        "canonical_action_seconds": canonical_seconds,
        "lowered_action_seconds": lowered_seconds,
        "pipeline_seconds": pipeline_seconds,
        "reference_halo_exchange_seconds": halo_seconds,
        "canonical_metadata_bytes": lowered.execution.report.canonical_metadata_bytes,
        "lowered_metadata_bytes": lowered.execution.report.lowered_metadata_bytes,
        "metadata_compression_ratio": (
            lowered.execution.report.canonical_metadata_bytes
            / lowered.execution.report.lowered_metadata_bytes
        ),
        "cse_reused_applications": pipeline.report.reused_application_count,
        "halo_exchange_descriptors": len(schedule.exchanges),
    }


def _transform_solve(points, repeats):
    grid = _periodic_grid(points)
    request = phx.discretization.DerivativeRequest(
        "dxx",
        grid,
        "x",
        derivative_order=2,
        accuracy_order=2,
    )
    prepared = phx.discretization.FiniteDifferencePlan(grid, (request,)).prepare()
    representation = prepared.transform_diagonalization("dxx")
    direct = phx.linalg.TransformDiagonalSolvePlan(
        representation,
        compatibility="project_rhs",
    ).prepare()
    rhs = jnp.sin(2.0 * jnp.pi * grid.axes[0].nodes)
    solve = eqx.filter_jit(direct.solve)
    solve(rhs).value.block_until_ready()
    result, steady = _timed(lambda: solve(rhs).value, repeats)
    return {
        "steady_solve_seconds": steady,
        "maximum_absolute_value": float(jnp.max(jnp.abs(result))),
    }


def _mixed_boundary_solve(points, repeats):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(points),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    diagonalization = phx.discretization.diagonalize_fd_laplacian(
        grid,
        {"x": ("dirichlet", "neumann")},
    )
    exact = jnp.sin(0.5 * jnp.pi * diagonalization.unknown_coordinates[0])
    rhs = diagonalization.apply(exact)
    prepared = phx.discretization.FDLaplacianSolvePlan(diagonalization)
    solve = eqx.filter_jit(prepared.solve)
    solve(rhs).value.block_until_ready()
    result, steady = _timed(lambda: solve(rhs).value, repeats)
    return {
        "steady_solve_seconds": steady,
        "maximum_absolute_error": float(jnp.max(jnp.abs(result - exact))),
        "transform_family": diagonalization.axis_reports[0].transform_family,
        "transform_type": diagonalization.axis_reports[0].transform_type,
    }


def _pml_reflection(points):
    width = max(2, min(points // 6, (points - 1) // 2))
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(points),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    acoustic = phx.solver.StaggeredAcousticPlan(
        grid,
        bulk_modulus=1.0,
        density=1.0,
        pml=phx.solver.SplitFieldPMLPlan(
            width,
            maximum_attenuation=60.0,
            polynomial_order=3,
        ),
    ).prepare()
    cell_x = grid.cells().coordinates_by_axis[0]
    face_x = grid.faces("x").coordinates_by_axis[0]
    initial = acoustic.pack(
        jnp.exp(-(((cell_x - 0.3) / 0.04) ** 2)),
        (jnp.exp(-(((face_x - 0.3) / 0.04) ** 2)),),
    )
    step = eqx.filter_jit(acoustic.leapfrog_step)
    dt = 0.7 * float(acoustic.stable_dt)
    step(jnp.asarray(0.0), initial, dt).pressure.block_until_ready()
    state = initial
    time_value = jnp.asarray(0.0)
    steps = int(1.2 / dt)
    started = time.perf_counter()
    for _ in range(steps):
        state = step(time_value, state, dt)
        time_value = time_value + dt
    state.pressure.block_until_ready()
    elapsed = time.perf_counter() - started
    reflected = jnp.max(jnp.abs(state.pressure[width:-width]))
    return {
        "simulation_seconds": elapsed,
        "steps": steps,
        "pml_width": width,
        "maximum_reflected_amplitude": float(reflected),
    }


def _patch_kernel(points, repeats):
    prepared = phx.discretization.PatchKernelPlan(
        (3,),
        lambda patch, args: jnp.sum(patch),
    ).prepare((points,))
    values = jnp.arange(float(points))
    action = eqx.filter_jit(prepared)
    action(values).block_until_ready()
    result, steady = _timed(lambda: action(values), repeats)
    return {
        "steady_action_seconds": steady,
        "output_points": int(result.size),
    }


def _weno(points, repeats):
    reconstruction = phx.discretization.WENOReconstructionPlan(5)
    values = jnp.sin(2.0 * jnp.pi * jnp.arange(points) / points)
    action = eqx.filter_jit(reconstruction.reconstruct)
    action(values)[0].block_until_ready()
    result, steady = _timed(lambda: action(values)[0], repeats)
    return {
        "steady_reconstruction_seconds": steady,
        "maximum_absolute_value": float(jnp.max(jnp.abs(result))),
    }


def _industrial_extensions(points, repeats):
    bounded = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(points),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    diffusion = phx.discretization.ConservativeDiffusionPlan(
        bounded,
        boundaries={"x": ("dirichlet", "dirichlet")},
    ).prepare(1.0)
    started = time.perf_counter()
    multigrid = phx.discretization.StructuredMultigridPlan(
        diffusion,
        minimum_coarse_points=4,
    ).prepare()
    multigrid_preparation = time.perf_counter() - started
    exact = jnp.sin(jnp.pi * bounded.axes[0].nodes)
    rhs = multigrid.level_operators[0].mv(exact)
    cycle = eqx.filter_jit(multigrid.apply)
    cycle(rhs).block_until_ready()
    correction, multigrid_seconds = _timed(lambda: cycle(rhs), repeats)
    residual_reduction = jnp.linalg.norm(
        rhs - multigrid.level_operators[0].mv(correction)
    ) / jnp.linalg.norm(rhs)

    values = jnp.sin(2.0 * jnp.pi * (jnp.arange(points) + 0.5) / points)
    reconstruction_times = {}
    for method in ("weno_z", "teno", "mp5"):
        reconstruction = phx.discretization.HighResolutionReconstructionPlan(method)
        action = eqx.filter_jit(reconstruction.reconstruct)
        action(values)[0].block_until_ready()
        _, seconds = _timed(lambda: action(values)[0], repeats)
        reconstruction_times[method] = seconds

    euler = phx.discretization.Euler1DSystem()
    primitive = jnp.stack(
        (
            jnp.where(bounded.axes[0].nodes < 0.5, 1.0, 0.125),
            jnp.zeros((points,)),
            jnp.where(bounded.axes[0].nodes < 0.5, 1.0, 0.1),
        ),
        axis=-1,
    )
    euler_state = euler.conservative(primitive)
    euler_dynamics = phx.discretization.Euler1DDynamics(
        euler,
        phx.discretization.HighResolutionReconstructionPlan(
            "weno_z",
            boundary="outflow",
        ),
        1.0 / points,
    )
    euler_step = eqx.filter_jit(euler_dynamics.ssprk3_step)
    dt = 0.3 * euler_dynamics.stable_step(euler_state)
    euler_step(jnp.asarray(0.0), euler_state, dt).block_until_ready()
    euler_result, euler_seconds = _timed(
        lambda: euler_step(jnp.asarray(0.0), euler_state, dt),
        repeats,
    )

    transfer = phx.discretization.AMREntityTransferPlan.cells(1)
    transfer_action = eqx.filter_jit(transfer.prolong)
    transfer_action(values).block_until_ready()
    _, transfer_seconds = _timed(lambda: transfer_action(values), repeats)
    return {
        "multigrid_preparation_seconds": multigrid_preparation,
        "multigrid_cycle_seconds": multigrid_seconds,
        "multigrid_one_cycle_residual_ratio": float(residual_reduction),
        "multigrid_levels": len(multigrid.grids),
        "reconstruction_seconds": reconstruction_times,
        "euler_ssprk3_seconds": euler_seconds,
        "euler_minimum_density": float(jnp.min(euler_result[:, 0])),
        "euler_minimum_pressure": float(jnp.min(euler.pressure(euler_result))),
        "amr_cell_prolongation_seconds": transfer_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.points < 8 or arguments.repeats < 1:
        raise ValueError("points must be at least eight and repeats positive.")
    report = {
        "finite_difference": _finite_difference(arguments.points, arguments.repeats),
        "execution_pipeline": _execution_pipeline(
            arguments.points,
            arguments.repeats,
        ),
        "transform_solve": _transform_solve(arguments.points, arguments.repeats),
        "mixed_boundary_solve": _mixed_boundary_solve(
            arguments.points,
            arguments.repeats,
        ),
        "pml_reflection": _pml_reflection(arguments.points),
        "patch_kernel": _patch_kernel(arguments.points, arguments.repeats),
        "weno5": _weno(arguments.points, arguments.repeats),
        "industrial_extensions": _industrial_extensions(
            arguments.points,
            arguments.repeats,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
