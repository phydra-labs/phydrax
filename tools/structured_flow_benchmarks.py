#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class StructuredFlowBenchmarkRecord:
    schema_version: int
    compact_count: int
    compact_first_error: float
    compact_second_error: float
    compact_storage_bytes: int
    compact_dense_entries: int
    sbp_count: int
    sbp_conservation_defect: float
    sbp_entropy_defect: float
    sbp_pair_count: int
    mac_count: int
    mac_transform_divergence: float
    mac_iterative_divergence: float
    mac_route_difference: float
    mac_momentum_count: int
    mac_momentum_skew_defect: float
    mac_diffusion_symmetry_defect: float
    mac_projected_rate_divergence: float
    mac_nonlinear_energy_defect: float
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.compact_first_error <= 1e-7
            and self.compact_second_error <= 1e-6
            and self.compact_dense_entries == 0
            and self.sbp_conservation_defect <= 1e-10
            and self.sbp_entropy_defect <= 1e-10
            and self.mac_transform_divergence <= 1e-9
            and self.mac_iterative_divergence <= 1e-7
            and self.mac_route_difference <= 1e-7
            and self.mac_momentum_skew_defect <= 1e-9
            and self.mac_diffusion_symmetry_defect <= 1e-9
            and self.mac_projected_rate_divergence <= 1e-8
            and self.mac_nonlinear_energy_defect <= 1e-8
        )


def run_structured_flow_benchmark(
    compact_count: int = 64,
    sbp_count: int = 32,
    mac_count: int = 32,
    /,
) -> StructuredFlowBenchmarkRecord:
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

    mac_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(mac_count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(mac_grid).prepare()
    mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    velocity = (jnp.sin(2.0 * jnp.pi * jnp.arange(mac_count) / mac_count),)
    transform = phx.solver.MACPressureProjectionPlan(
        mac, solve_method="transform", tolerance=1e-10
    ).project(velocity, 0.1)
    iterative = phx.solver.MACPressureProjectionPlan(
        mac, solve_method="iterative", tolerance=1e-10
    ).project(velocity, 0.1)
    transform_divergence = jnp.linalg.norm(transform.divergence_after)
    iterative_divergence = jnp.linalg.norm(iterative.divergence_after)
    route_difference = jnp.max(jnp.abs(transform.velocity[0] - iterative.velocity[0]))

    momentum_count = max(4, mac_count // 2)
    momentum_grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(momentum_count, periodic=True),
            phx.discretization.UniformCellAxisSpec(momentum_count, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0 * jnp.pi, 2.0 * jnp.pi]]))
    momentum_finite_volume = phx.discretization.FiniteVolumePlan(momentum_grid).prepare()
    momentum_mac = phx.discretization.MACOperatorPlan(momentum_finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(momentum_mac).prepare()
    momentum_projection = phx.solver.MACPressureProjectionPlan(
        momentum_mac,
        solve_method="transform",
        tolerance=1e-10,
    )
    compiled_mac = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        momentum,
        momentum_projection,
    )
    x_faces = momentum_finite_volume.face_centers[0]
    y_faces = momentum_finite_volume.face_centers[1]
    momentum_velocity = (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]),
    )
    momentum_state = compiled_mac.project_state(momentum_velocity)
    momentum_diagnostics = compiled_mac.diagnostics(0.0, momentum_state)
    values = jnp.asarray(
        (
            first_error,
            second_error,
            conservation,
            sbp_diagnostics.convective_entropy_rate,
            transform_divergence,
            iterative_divergence,
            route_difference,
            momentum.report.weighted_skew_residual,
            momentum.report.diffusion_symmetry_residual,
            momentum_diagnostics.divergence_norm,
            momentum_diagnostics.nonlinear_energy_rate,
        )
    )
    return StructuredFlowBenchmarkRecord(
        schema_version=1,
        compact_count=compact_count,
        compact_first_error=float(first_error),
        compact_second_error=float(second_error),
        compact_storage_bytes=first.report.storage_bytes + second.report.storage_bytes,
        compact_dense_entries=(
            first.report.dense_materialization_entries
            + second.report.dense_materialization_entries
        ),
        sbp_count=sbp_count,
        sbp_conservation_defect=float(conservation),
        sbp_entropy_defect=float(jnp.abs(sbp_diagnostics.convective_entropy_rate)),
        sbp_pair_count=compiled.dynamics.report.pair_counts[0],
        mac_count=mac_count,
        mac_transform_divergence=float(transform_divergence),
        mac_iterative_divergence=float(iterative_divergence),
        mac_route_difference=float(route_difference),
        mac_momentum_count=momentum_count,
        mac_momentum_skew_defect=float(momentum.report.weighted_skew_residual),
        mac_diffusion_symmetry_defect=float(momentum.report.diffusion_symmetry_residual),
        mac_projected_rate_divergence=float(momentum_diagnostics.divergence_norm),
        mac_nonlinear_energy_defect=float(
            jnp.abs(momentum_diagnostics.nonlinear_energy_rate)
        ),
        finite=bool(
            jnp.all(jnp.isfinite(values))
            & transform.converged
            & iterative.converged
            & momentum_diagnostics.projection_converged
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify compact, SBP flux-differencing, and MAC flow substrates."
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
    payload = json.dumps({**asdict(record), "passed": record.passed}, indent=2)
    if arguments.output is not None:
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, arguments.output)
    print(payload)
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
