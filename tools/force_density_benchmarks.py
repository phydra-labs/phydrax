#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from math import sqrt
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


fd = phx.applications.solid_mechanics


def _chain(node_count: int):
    nodes = int(node_count)
    coordinates = jnp.stack(
        (jnp.linspace(-5.0, 5.0, nodes), jnp.zeros((nodes,))), axis=-1
    )
    edges = jnp.stack((jnp.arange(nodes - 1), jnp.arange(1, nodes)), axis=-1).astype(
        jnp.int32
    )
    structure = fd.ForceDensityStructure.from_edges(
        edges,
        nodes,
        2,
        fixed_nodes=(0, nodes - 1),
    )
    loads = jnp.zeros((nodes, 2)).at[:, 1].set(-10.0 / nodes)
    inputs = fd.ForceDensityInputs(
        jnp.full((nodes - 1,), 2.0),
        structure.prescribed_values(coordinates),
        loads,
    )
    problem = fd.ForceDensityProblem(
        structure,
        sign_mode="tension",
        problem_id=f"benchmark-chain-{nodes}",
    )
    return problem, inputs


def run_forward(node_count: int, repeats: int, /) -> dict[str, object]:
    problem, inputs = _chain(node_count)
    started = time.perf_counter()
    plan = fd.plan_force_density(problem, inputs)
    plan_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prepared = fd.prepare_force_density(plan, inputs)
    prepare_seconds = time.perf_counter() - started
    started = time.perf_counter()
    first = fd.solve_force_density(prepared)
    jax.block_until_ready(first.state.positions)
    first_seconds = time.perf_counter() - started

    current = prepared
    started = time.perf_counter()
    for repeat in range(repeats):
        scale = 1.0 + 0.01 * (repeat + 1)
        refreshed_inputs = fd.ForceDensityInputs(
            inputs.force_densities * scale,
            inputs.prescribed_values,
            inputs.load_parameters,
        )
        current = fd.refresh_force_density(current, refreshed_inputs)
        refreshed = fd.solve_force_density(current)
        jax.block_until_ready(refreshed.state.positions)
    steady_seconds = (time.perf_counter() - started) / repeats

    def objective(force_densities):
        dynamic = fd.ForceDensityInputs(
            force_densities,
            inputs.prescribed_values,
            inputs.load_parameters,
        )
        result = fd.solve_force_density(fd.prepare_force_density(plan, dynamic))
        return jnp.sum(result.state.positions**2)

    gradient = jax.jit(jax.grad(objective))
    started = time.perf_counter()
    gradient_value = gradient(inputs.force_densities)
    jax.block_until_ready(gradient_value)
    gradient_seconds = time.perf_counter() - started
    linear = first.linear_result
    return {
        "node_count": node_count,
        "member_count": problem.structure.member_count,
        "free_dof_count": problem.structure.free_dof_count,
        "contribution_count": problem.structure.equilibrium_relation.capacity,
        "linear_backend": None if linear is None else linear.provenance.backend,
        "linear_method": None if linear is None else linear.provenance.method,
        "plan_seconds": plan_seconds,
        "prepare_seconds": prepare_seconds,
        "first_solve_seconds": first_seconds,
        "steady_refresh_solve_seconds": steady_seconds,
        "gradient_seconds": gradient_seconds,
        "gradient_norm": float(jnp.sqrt(jnp.sum(gradient_value**2))),
        "residual_norm": float(first.diagnostics.free_residual_norm),
        "global_balance_norm": float(first.diagnostics.global_balance_norm),
        "status": int(first.status),
        "plan_id": plan.plan_id,
    }


def run_arch(node_count: int, /) -> dict[str, object]:
    span = 10.0
    load_density = 1.0
    nodes = int(node_count)
    coordinates = jnp.stack(
        (jnp.linspace(-span / 2.0, span / 2.0, nodes), jnp.zeros((nodes,))),
        axis=-1,
    )
    edges = jnp.stack((jnp.arange(nodes - 1), jnp.arange(1, nodes)), axis=-1).astype(
        jnp.int32
    )
    structure = fd.ForceDensityStructure.from_edges(
        edges,
        nodes,
        2,
        fixed_nodes=(0, nodes - 1),
    )
    loads = jnp.zeros((nodes, 2)).at[:, 1].set(-(load_density * span) / nodes)
    prescribed = structure.prescribed_values(coordinates)
    sample = fd.ForceDensityInputs(jnp.full((nodes - 1,), -10.0), prescribed, loads)
    equilibrium = fd.ForceDensityProblem(
        structure,
        sign_mode="compression",
        problem_id=f"benchmark-arch-{nodes}",
    )
    strict_linear = phx.linalg.LinearSolvePolicy(
        tolerance=phx.linalg.TolerancePolicy(
            relative=1.0e-12,
            absolute=1.0e-14,
            max_steps=2000,
        ),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
        require_device_binding=True,
    )
    plan = fd.plan_force_density(
        equilibrium,
        sample,
        linear_policy=strict_linear,
    )

    def decode(magnitude, _):
        return fd.ForceDensityInputs(
            jnp.full((nodes - 1,), -magnitude.reshape(())),
            prescribed,
            loads,
        )

    def objective(magnitude, _):
        inputs = decode(magnitude, None)
        state = fd.solve_force_density(
            fd.prepare_force_density(plan, inputs)
        ).state
        return fd.force_density_load_path(state)

    initial = fd.force_density_equilibrium(equilibrium, sample)
    initial_load_path = fd.force_density_load_path(initial.state)
    started = time.perf_counter()
    optimized = phx.optim.minimize(
        objective,
        jnp.asarray(10.0),
        bounds=phx.optim.Bounds(1.0e-3, 1.0e3),
        method=phx.optim.ProjectedLBFGS(),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-8,
            relative_optimality=0.0,
            maximum_steps=200,
        ),
    )
    final_inputs = decode(optimized.parameters, None)
    result = fd.solve_force_density(fd.prepare_force_density(plan, final_inputs))
    jax.block_until_ready(result.state.positions)
    wall_seconds = time.perf_counter() - started
    rise = jnp.max(jnp.abs(result.state.positions[:, 1]))
    load_path = fd.force_density_load_path(result.state)
    target_rise = sqrt(3.0) * span / 4.0
    target_load_path = load_density * span**2 / sqrt(3.0)
    return {
        "node_count": nodes,
        "status": int(optimized.status),
        "initial_load_path": float(initial_load_path),
        "final_load_path": float(load_path),
        "target_load_path": target_load_path,
        "relative_load_path_error": abs(float(load_path) - target_load_path)
        / target_load_path,
        "rise": float(rise),
        "target_rise": target_rise,
        "relative_rise_error": abs(float(rise) - target_rise) / target_rise,
        "force_density_magnitude": float(optimized.parameters),
        "iterations": int(optimized.diagnostics.iterations),
        "linear_solves": int(optimized.diagnostics.linear_solves),
        "residual_norm": float(result.diagnostics.free_residual_norm),
        "wall_seconds": wall_seconds,
    }


def run_nonlinear(node_count: int, repeats: int, /) -> dict[str, object]:
    nodes = int(node_count)
    coordinates = jnp.stack(
        (jnp.linspace(-5.0, 5.0, nodes), jnp.zeros((nodes,))), axis=-1
    )
    edges = jnp.stack((jnp.arange(nodes - 1), jnp.arange(1, nodes)), axis=-1).astype(
        jnp.int32
    )
    structure = fd.ForceDensityStructure.from_edges(
        edges, nodes, 2, fixed_nodes=(0, nodes - 1)
    )
    parameters = jnp.zeros((nodes - 1, 2)).at[:, 1].set(-0.02)
    inputs = fd.ForceDensityInputs(
        jnp.full((nodes - 1,), 20.0),
        structure.prescribed_values(coordinates),
        parameters,
    )
    problem = fd.ForceDensityProblem(
        structure,
        load_model=fd.EdgeLineLoadModel(measure="current"),
        sign_mode="tension",
        tolerances=fd.ForceDensityTolerances(
            absolute_equilibrium=1.0e-7,
            relative_equilibrium=1.0e-8,
        ),
        problem_id=f"benchmark-current-line-{nodes}",
    )
    started = time.perf_counter()
    plan = fd.plan_force_density(problem, inputs, initial_positions=coordinates)
    plan_seconds = time.perf_counter() - started
    prepared = fd.prepare_force_density(plan, inputs, initial_positions=coordinates)
    started = time.perf_counter()
    first = fd.solve_force_density(prepared)
    jax.block_until_ready(first.state.positions)
    first_seconds = time.perf_counter() - started
    current = prepared
    started = time.perf_counter()
    for repeat in range(repeats):
        changed = fd.ForceDensityInputs(
            inputs.force_densities,
            inputs.prescribed_values,
            parameters * (1.0 + 0.01 * (repeat + 1)),
        )
        current = fd.refresh_force_density(
            current, changed, initial_positions=first.state.positions
        )
        refreshed = fd.solve_force_density(current)
        jax.block_until_ready(refreshed.state.positions)
    refresh_seconds = (time.perf_counter() - started) / repeats
    nonlinear = first.nonlinear_result
    return {
        "node_count": nodes,
        "member_count": structure.member_count,
        "plan_seconds": plan_seconds,
        "first_solve_seconds": first_seconds,
        "steady_refresh_solve_seconds": refresh_seconds,
        "status": int(first.status),
        "iterations": int(nonlinear.diagnostics.iterations),
        "linear_iterations": int(nonlinear.diagnostics.linear_iterations),
        "linear_plan_id": nonlinear.provenance.linear_plan_id,
        "weighted_setup": plan.nonlinear_uses_setup,
        "residual_norm": float(first.diagnostics.free_residual_norm),
    }


def run_batch(case_count: int = 16) -> dict[str, object]:
    problem, sample = _chain(50)
    plan = fd.plan_force_density(problem, sample)
    scale = jnp.linspace(0.8, 1.2, case_count)
    densities = scale[:, None] * sample.force_densities
    prescribed = jnp.repeat(sample.prescribed_values[None, :], case_count, axis=0)
    loads = jnp.repeat(sample.load_parameters[None, :, :], case_count, axis=0)
    started = time.perf_counter()
    result = fd.solve_force_density_batch(plan, densities, prescribed, loads)
    jax.block_until_ready(result.results.state.positions)
    wall_seconds = time.perf_counter() - started
    return {
        "case_count": case_count,
        "successful_count": int(jnp.sum(result.successful)),
        "wall_seconds": wall_seconds,
        "maximum_residual": float(jnp.max(result.results.diagnostics.free_residual_norm)),
    }


def run_stability() -> dict[str, object]:
    problem, inputs = _chain(12)
    result = fd.force_density_equilibrium(problem, inputs)
    started = time.perf_counter()
    mechanisms = fd.analyze_force_density_mechanisms(
        problem.structure, result.state.positions
    )
    tangent = fd.analyze_force_density_tangent_stability(
        problem.structure,
        result.state,
        jnp.full((problem.structure.member_count,), 100.0),
    )
    wall_seconds = time.perf_counter() - started
    return {
        "mechanism_count": int(mechanisms.mechanism_count),
        "self_stress_count": int(mechanisms.self_stress_count),
        "minimum_tangent_eigenvalue": float(tangent.minimum_eigenvalue),
        "stable": bool(tangent.stable),
        "wall_seconds": wall_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/force_density.json"),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=(100, 1000, 10000))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sizes = (25,) if args.smoke else tuple(int(value) for value in args.sizes)
    repeats = 1 if args.smoke else int(args.repeats)
    payload = {
        "forward": [run_forward(size, repeats) for size in sizes],
        "nonlinear": run_nonlinear(20 if args.smoke else 50, repeats),
        "batch": run_batch(4 if args.smoke else 32),
        "stability": run_stability(),
        "arch": run_arch(20 if args.smoke else 100),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
