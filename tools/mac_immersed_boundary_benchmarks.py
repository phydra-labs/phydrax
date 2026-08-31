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
class MACImmersedBoundaryBenchmarkRecord:
    grid_count: int
    marker_count: int
    route_width: int
    relation_bytes: int
    scalar_workspace_bytes: int
    partition_defect: float
    first_moment_defect: float
    gradient_sum_defect: float
    force_defect: float
    torque_defect: float
    work_defect: float
    divergence_norm: float
    marker_slip_norm: float
    kkt_residual_norm: float
    marker_numerical_rank: int
    marker_condition: float
    marker_rank_certified: bool
    finite: bool
    passed: bool


def run_mac_immersed_boundary_benchmark(
    grid_count: int = 20, marker_count: int = 32, /
) -> MACImmersedBoundaryBenchmarkRecord:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(grid_count, periodic=True)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    angle = 2.0 * jnp.pi * jnp.arange(marker_count) / marker_count
    position = jnp.stack(
        (0.5 + 0.15 * jnp.cos(angle), 0.5 + 0.15 * jnp.sin(angle)), axis=-1
    )
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(marker_count),
        position,
        jnp.full((marker_count,), 2.0 * jnp.pi * 0.15 / marker_count),
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers)
    prepared = transfer.prepare()
    relation = prepared.relation(position)
    velocity = tuple(
        jnp.zeros(layout.shape) for layout in finite_volume.face_layouts
    )
    marker_force = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
    diagnostics = prepared.diagnostics(relation, velocity, marker_force)
    projection = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators, prepared, boundaries=boundaries, tolerance=1.0e-8
    ).project(
        velocity,
        jnp.asarray(1.0e-3),
        markers.kinematics(position, jnp.zeros_like(position)),
    )
    values = jnp.asarray(
        (
            diagnostics.maximum_partition_residual,
            diagnostics.maximum_first_moment_residual,
            diagnostics.maximum_gradient_sum_residual,
            jnp.max(jnp.abs(diagnostics.force_residual)),
            jnp.max(jnp.abs(diagnostics.torque_residual)),
            jnp.abs(diagnostics.work_adjoint_residual),
            jnp.linalg.norm(projection.divergence_after),
            jnp.linalg.norm(projection.marker_slip),
            projection.kkt_residual_norm,
        )
    )
    finite = bool(jnp.all(jnp.isfinite(values)))
    passed = bool(
        finite
        and relation.successful
        and diagnostics.successful
        and projection.successful
        and projection.marker_rank_certified
        and projection.marker_numerical_rank == 2 * marker_count
        and jnp.isfinite(projection.marker_condition)
        and jnp.max(values[:6]) <= 1.0e-8
        and jnp.max(values[6:]) <= 1.0e-7
    )
    return MACImmersedBoundaryBenchmarkRecord(
        grid_count=grid_count,
        marker_count=marker_count,
        route_width=transfer.route_width,
        relation_bytes=transfer.relation_bytes,
        scalar_workspace_bytes=transfer.scalar_workspace_bytes,
        partition_defect=float(values[0]),
        first_moment_defect=float(values[1]),
        gradient_sum_defect=float(values[2]),
        force_defect=float(values[3]),
        torque_defect=float(values[4]),
        work_defect=float(values[5]),
        divergence_norm=float(values[6]),
        marker_slip_norm=float(values[7]),
        kkt_residual_norm=float(values[8]),
        marker_numerical_rank=int(projection.marker_numerical_rank),
        marker_condition=float(projection.marker_condition),
        marker_rank_certified=bool(projection.marker_rank_certified),
        finite=finite,
        passed=passed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify material-measure MAC immersed-boundary coupling."
    )
    parser.add_argument("--grid-count", type=int, default=20)
    parser.add_argument("--marker-count", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    record = run_mac_immersed_boundary_benchmark(
        8 if arguments.smoke else arguments.grid_count,
        12 if arguments.smoke else arguments.marker_count,
    )
    payload = json.dumps(asdict(record), indent=2)
    if arguments.output is not None:
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, arguments.output)
    print(payload)
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
