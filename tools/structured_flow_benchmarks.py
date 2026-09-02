#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class StructuredFlowBenchmarkRecord:
    """Raw structured-discretization and MAC route smoke evidence."""

    kind: str
    compact: dict[str, Any]
    sbp: dict[str, Any]
    mac: dict[str, Any]
    finite: bool
    smoke_successful: bool


def _norm(value) -> float:
    return float(jnp.linalg.norm(jnp.asarray(value)))


def _projection_record(
    requested_route: str,
    plan: phx.solver.MACPressureProjectionPlan,
    result: phx.solver.MACPressureProjectionResult,
    /,
) -> dict[str, object]:
    linear = result.linear
    transform = result.transform
    hybrid = result.hybrid
    linear_actions = (
        None
        if linear is None
        else {
            "iterations": int(linear.diagnostics.iterations),
            "matvec_count": int(linear.diagnostics.matvec_count),
            "adjoint_matvec_count": int(linear.diagnostics.adjoint_matvec_count),
        }
    )
    hybrid_resources = (
        None
        if hybrid is None
        else {
            "line_count": int(hybrid.resources.line_count),
            "line_size": int(hybrid.resources.line_size),
            "factor_count": int(hybrid.resources.factor_count),
            "factor_bytes": int(hybrid.resources.factor_bytes),
            "workspace_bytes": int(hybrid.resources.workspace_bytes),
            "total_bytes": int(hybrid.resources.total_bytes),
            "periodic_rank": int(hybrid.resources.periodic_rank),
        }
    )
    route_residuals: dict[str, float] = {}
    if transform is not None:
        route_residuals = {
            "transform_residual_norm": float(transform.residual_norm),
            "transform_compatibility_residual": float(transform.compatibility_residual),
            "transform_removed_component_norm": float(transform.removed_component_norm),
        }
    elif hybrid is not None:
        route_residuals = {
            "hybrid_residual_norm": float(hybrid.residual_norm),
            "hybrid_relative_residual": float(hybrid.relative_residual),
            "hybrid_compatibility_defect": float(hybrid.compatibility_defect),
            "hybrid_gauge_defect": float(hybrid.gauge_defect),
            "hybrid_trace_defect": float(hybrid.trace_defect),
            "hybrid_factor_residual": float(hybrid.factor_residual),
            "hybrid_minimum_pivot": float(hybrid.minimum_pivot),
        }
    elif linear is not None:
        route_residuals = {
            "linear_residual_norm": float(linear.diagnostics.residual_norm),
            "linear_relative_residual": float(linear.diagnostics.relative_residual),
            "linear_compatibility_residual": float(
                linear.diagnostics.compatibility_residual
            ),
            "linear_gauge_residual": float(linear.diagnostics.gauge_residual),
        }
    return {
        "available": True,
        "requested_route": requested_route,
        "solve_method": result.solve_method,
        "constant_route": plan.constant_route,
        "plan_id": plan.plan_id,
        "projection_id": result.projection_id,
        "operator_id": plan.operator_id,
        "pressure_problem_id": plan.pressure_problem_id,
        "closure_id": plan.closure_id,
        "hybrid_line_axis": result.hybrid_line_axis,
        "resources": {
            "maximum_resource_bytes": int(result.maximum_resource_bytes),
            "pressure_state_bytes": int(result.pressure.nbytes),
            "velocity_state_bytes": int(sum(value.nbytes for value in result.velocity)),
            "hybrid": hybrid_resources,
        },
        "actions": {
            "hybrid_action_defect": (
                None if result.hybrid is None else float(result.hybrid_action_defect)
            ),
            "iterative": linear_actions,
        },
        "residuals": {
            "divergence_before_norm": _norm(result.divergence_before),
            "divergence_after_norm": _norm(result.divergence_after),
            "pressure_residual_norm": _norm(result.pressure_residual),
            "gauge_defect": float(result.gauge_defect),
            "closure_mass_defect": float(result.closure.mass_defect),
            **route_residuals,
        },
        "finite": bool(result.finite),
        "converged": bool(result.converged),
    }


def _unavailable_route(requested_route: str, reason: str) -> dict[str, object]:
    return {
        "available": False,
        "requested_route": requested_route,
        "reason": reason,
    }


def _uniform_mac_routes(count: int) -> dict[str, object]:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0 * jnp.pi, 2.0 * jnp.pi]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    x_faces = finite_volume.face_centers[0]
    y_faces = finite_volume.face_centers[1]
    velocity = (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]),
    )
    full_plan = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="transform",
        tolerance=1e-10,
    )
    iterative_plan = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="iterative",
        tolerance=1e-10,
    )
    full = full_plan.project(velocity, 0.1)
    iterative = iterative_plan.project(velocity, 0.1)
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    compiled_mac = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        momentum,
        full_plan,
    )
    momentum_state = compiled_mac.project_state(velocity)
    momentum_diagnostics = compiled_mac.diagnostics(0.0, momentum_state)
    difference = max(
        float(jnp.max(jnp.abs(left - right)))
        for left, right in zip(full.velocity, iterative.velocity, strict=True)
    )
    return {
        "geometry": "uniform-periodic",
        "shape": list(finite_volume.cell_shape),
        "dtype": str(finite_volume.cell_volumes.dtype),
        "grid_prepared_id": grid.prepared_id,
        "finite_volume_prepared_id": finite_volume.prepared_id,
        "operator_prepared_id": operators.prepared_id,
        "operator_report_id": operators.report.report_id,
        "operator_invariants": {
            "weighted_adjoint_residual": float(
                operators.report.weighted_adjoint_residual
            ),
            "constant_laplacian_residual": float(
                operators.report.constant_laplacian_residual
            ),
            "transform_eligible": bool(operators.report.transform_eligible),
        },
        "momentum": {
            "count": count,
            "momentum_prepared_id": momentum.prepared_id,
            "momentum_report_id": momentum.report.report_id,
            "compilation_id": compiled_mac.compilation_id,
            "raw_invariants": {
                "weighted_skew_residual": float(momentum.report.weighted_skew_residual),
                "diffusion_symmetry_residual": float(
                    momentum.report.diffusion_symmetry_residual
                ),
                "projected_rate_divergence": float(momentum_diagnostics.divergence_norm),
                "nonlinear_energy_rate": float(
                    momentum_diagnostics.nonlinear_energy_rate
                ),
                "finite": bool(momentum_diagnostics.finite),
            },
        },
        "routes": {
            "full": _projection_record("full", full_plan, full),
            "hybrid": _unavailable_route(
                "hybrid",
                "Hybrid requires one explicitly nonperiodic line axis.",
            ),
            "iterative": _projection_record("iterative", iterative_plan, iterative),
        },
        "comparisons": {"full_iterative_velocity_maximum_difference": difference},
        "finite": bool(full.finite & iterative.finite & momentum_diagnostics.finite),
        "successful": bool(
            full.converged
            & iterative.converged
            & momentum_diagnostics.projection_converged
        ),
    }


def _stretched_mac_routes(count: int) -> dict[str, object]:
    transverse_count = max(4, count)
    line_count = max(4, count)
    normalized_edges = jnp.linspace(0.0, 1.0, line_count + 1) ** 1.5
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(transverse_count, periodic=True),
            phx.discretization.NonuniformCellAxisSpec(normalized_edges),
            phx.discretization.UniformCellAxisSpec(transverse_count, periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [2.0 * jnp.pi, 1.0, 2.0 * jnp.pi]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    x_faces, y_faces, z_faces = finite_volume.face_centers
    velocity = (
        jnp.sin(x_faces[..., 0])
        * jnp.cos(jnp.pi * x_faces[..., 1])
        * jnp.cos(x_faces[..., 2]),
        -2.0
        * jnp.cos(y_faces[..., 0])
        * jnp.sin(jnp.pi * y_faces[..., 1])
        * jnp.cos(y_faces[..., 2]),
        jnp.zeros(z_faces.shape[:-1], dtype=z_faces.dtype),
    )
    hybrid_plan = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="hybrid",
        hybrid_line_axis=1,
        tolerance=1e-10,
    )
    iterative_plan = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="iterative",
        tolerance=1e-10,
    )
    hybrid = hybrid_plan.project(velocity, 0.1)
    iterative = iterative_plan.project(velocity, 0.1)
    difference = max(
        float(jnp.max(jnp.abs(left - right)))
        for left, right in zip(hybrid.velocity, iterative.velocity, strict=True)
    )
    return {
        "geometry": "stretched-wall-normal",
        "shape": list(finite_volume.cell_shape),
        "dtype": str(finite_volume.cell_volumes.dtype),
        "grid_prepared_id": grid.prepared_id,
        "finite_volume_prepared_id": finite_volume.prepared_id,
        "operator_prepared_id": operators.prepared_id,
        "operator_report_id": operators.report.report_id,
        "operator_invariants": {
            "weighted_adjoint_residual": float(
                operators.report.weighted_adjoint_residual
            ),
            "constant_laplacian_residual": float(
                operators.report.constant_laplacian_residual
            ),
            "transform_eligible": bool(operators.report.transform_eligible),
        },
        "routes": {
            "full": _unavailable_route(
                "full",
                "A full tensor transform is ineligible on the stretched line.",
            ),
            "hybrid": _projection_record("hybrid", hybrid_plan, hybrid),
            "iterative": _projection_record("iterative", iterative_plan, iterative),
        },
        "comparisons": {"hybrid_iterative_velocity_maximum_difference": difference},
        "finite": bool(hybrid.finite & iterative.finite),
        "successful": bool(hybrid.converged & iterative.converged),
    }


def run_structured_flow_benchmark(
    compact_count: int = 64,
    sbp_count: int = 32,
    mac_count: int = 32,
    /,
) -> StructuredFlowBenchmarkRecord:
    """Record raw structured-flow smoke evidence, not qualification gates."""

    compact_grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                compact_count, periodic=True, endpoint=False
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    first_request = phx.discretization.DerivativeRequest(
        "dx",
        compact_grid,
        "x",
        derivative_order=1,
        accuracy_order=6,
    )
    second_request = phx.discretization.DerivativeRequest(
        "dxx",
        compact_grid,
        "x",
        derivative_order=2,
        accuracy_order=6,
    )
    first = phx.discretization.CompactDerivativePlan(
        compact_grid, first_request
    ).prepare()
    second = phx.discretization.CompactDerivativePlan(
        compact_grid, second_request
    ).prepare()
    x = compact_grid.axes[0].nodes
    compact_value = jnp.sin(2.0 * jnp.pi * x)
    first_error = jnp.max(
        jnp.abs(first.mv(compact_value) - 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x))
    )
    second_error = jnp.max(
        jnp.abs(second.mv(compact_value) + (2.0 * jnp.pi) ** 2 * compact_value)
    )

    sbp_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(sbp_count, periodic=True, endpoint=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem(1)
    sbp = phx.discretization.TensorSBPPlan(
        sbp_grid,
        field_name="state",
        component_names=system.component_names,
        interior_order=6,
    ).prepare()
    entropy_pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    method = phx.discretization.SBPFluxDifferencingMethodPlan(
        phx.discretization.EntropyConservativeEulerFluxPlan(),
        entropy_diagnostics=True,
    )
    problem = phx.equations.ConservationProblemIR("euler", "state", system, None)
    compiled = phx.equations.compile_conservation_problem(
        problem, sbp, method, entropy_pair=entropy_pair
    )
    sx = sbp_grid.axes[0].nodes
    sbp_state = system.primitive_to_conserved(
        jnp.stack(
            (
                1.0 + 0.05 * jnp.sin(2.0 * jnp.pi * sx),
                0.2 + 0.02 * jnp.cos(2.0 * jnp.pi * sx),
                jnp.ones_like(sx),
            ),
            axis=-1,
        )
    )
    sbp_rate, sbp_diagnostics = compiled.residual_with_diagnostics(0.0, sbp_state)
    conservation = jnp.max(
        jnp.abs(jnp.sum(sbp.quadrature_weights[..., None] * sbp_rate, axis=0))
    )

    momentum_count = max(4, mac_count // 2)
    uniform = _uniform_mac_routes(momentum_count)
    momentum = uniform.pop("momentum")
    stretched = _stretched_mac_routes(max(4, momentum_count // 2))

    compact_finite = bool(jnp.isfinite(first_error) & jnp.isfinite(second_error))
    sbp_finite = bool(
        jnp.all(jnp.isfinite(sbp_rate))
        & jnp.isfinite(conservation)
        & jnp.isfinite(sbp_diagnostics.convective_entropy_rate)
    )
    finite = (
        compact_finite and sbp_finite and bool(uniform["finite"] and stretched["finite"])
    )
    smoke_successful = finite and bool(uniform["successful"] and stretched["successful"])
    return StructuredFlowBenchmarkRecord(
        kind="structured-flow-smoke-resource-evidence",
        compact={
            "count": int(compact_count),
            "grid_prepared_id": compact_grid.prepared_id,
            "first_prepared_id": first.prepared_id,
            "second_prepared_id": second.prepared_id,
            "raw_invariants": {
                "first_derivative_maximum_error": float(first_error),
                "second_derivative_maximum_error": float(second_error),
                "storage_bytes": int(
                    first.report.storage_bytes + second.report.storage_bytes
                ),
                "dense_materialization_entries": int(
                    first.report.dense_materialization_entries
                    + second.report.dense_materialization_entries
                ),
                "finite": compact_finite,
            },
        },
        sbp={
            "count": int(sbp_count),
            "grid_prepared_id": sbp_grid.prepared_id,
            "method_id": method.method_id,
            "compilation_id": compiled.compilation_id,
            "raw_invariants": {
                "conservation_defect": float(conservation),
                "entropy_defect": float(jnp.abs(sbp_diagnostics.convective_entropy_rate)),
                "pair_count": int(compiled.dynamics.report.pair_counts[0]),
                "finite": sbp_finite,
            },
        },
        mac={
            "momentum": momentum,
            "uniform": uniform,
            "stretched": stretched,
            "routes_are_raw_smoke_evidence": True,
        },
        finite=finite,
        smoke_successful=smoke_successful,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record raw compact, SBP, and MAC route smoke/resource evidence; "
            "this is not scientific qualification."
        )
    )
    parser.add_argument("--compact-count", type=int, default=64)
    parser.add_argument("--sbp-count", type=int, default=32)
    parser.add_argument("--mac-count", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    count = 16 if arguments.smoke else None
    record = run_structured_flow_benchmark(
        count or arguments.compact_count,
        count or arguments.sbp_count,
        count or arguments.mac_count,
    )
    payload = json.dumps(asdict(record), indent=2, sort_keys=True)
    if arguments.output is not None:
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, arguments.output)
    print(payload)
    if not record.smoke_successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
