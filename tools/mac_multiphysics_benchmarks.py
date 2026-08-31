#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class MACMultiphysicsBenchmarkRecord:
    count: int
    boundary_mass_defect: float
    scalar_content_balance_defect: float
    buoyancy_exchange_defect: float
    helmholtz_relative_residual: float
    variable_density_divergence: float
    marker_work_adjoint_defect: float
    distributed_divergence: float
    mapped_adjoint_defect: float
    ale_gcl_defect: float
    adaptive_completed: bool
    finite: bool
    passed: bool


def _grid(count, *, periodic):
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))


def run_mac_multiphysics_benchmark(*, count=6):
    grid = _grid(count, periodic=True)
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="transform", tolerance=1e-10
    )
    flow = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.02), momentum, projection
    )
    x_faces = finite_volume.face_centers[0]
    y_faces = finite_volume.face_centers[1]
    velocity = (
        jnp.sin(2.0 * jnp.pi * x_faces[..., 0]) * jnp.cos(2.0 * jnp.pi * x_faces[..., 1]),
        -jnp.cos(2.0 * jnp.pi * y_faces[..., 0])
        * jnp.sin(2.0 * jnp.pi * y_faces[..., 1]),
    )
    state = flow.project_state(velocity)
    boundary_stage = momentum.boundaries.evaluate(0.0)

    scalar_problem = phx.discretization.MACScalarProblem(
        (
            phx.discretization.MACScalarTransport(
                "temperature", 0.01, advection="centered"
            ),
        )
    )
    scalar_transport = scalar_problem.prepare(operators)
    buoyancy = phx.equations.MACBuoyancyLaw(
        jnp.asarray([0.0, -1.0]), {"temperature": 0.1}
    )
    scalar_flow = phx.equations.compile_mac_scalar_buoyancy(
        flow.problem,
        momentum,
        projection,
        scalar_problem,
        scalar_transport,
        buoyancy,
    )
    temperature = jnp.sin(2.0 * jnp.pi * finite_volume.cell_centers[..., 0])
    scalar_state = scalar_flow.project_state(velocity, {"temperature": temperature})
    scalar_diagnostics = scalar_flow.diagnostics(0.0, scalar_state)
    scalar_balance = jnp.abs(
        scalar_diagnostics.scalars.fields["temperature"].content_balance_defect
    )

    helmholtz = phx.solver.MACHelmholtzSolvePlan(
        momentum,
        solve_method="transform",
        fixed_mass_coefficient=1.0,
        fixed_diffusion_coefficient=0.001,
    ).solve(velocity, boundary_stage)

    variable = phx.discretization.MACVariableDensityPlan(momentum).prepare()
    variable_projection = phx.solver.MACVariableDensityProjectionPlan(operators)
    variable_flow = phx.equations.compile_mac_variable_density_flow(
        phx.equations.MACVariableDensityFlowProblem(2, 0.02),
        variable,
        variable_projection,
    )
    variable_state = variable_flow.project_coordinates(
        variable_flow.pack_state(jnp.ones(finite_volume.cell_shape), velocity)
    )
    variable_diagnostics = variable_flow.diagnostics(0.0, variable_state)

    marker_transfer = phx.discretization.MACMarkerTransferPlan(
        operators, 0.3, min(12, count * count)
    ).prepare()
    marker_relation = marker_transfer.relation(jnp.asarray([[0.25, 0.25]]))
    marker_diagnostics = marker_transfer.diagnostics(
        marker_relation, velocity, jnp.asarray([[0.2, -0.1]])
    )

    distributed = phx.discretization.MACDistributedTopologyPlan.single_device(
        operators
    ).prepare(momentum)
    distributed_state = distributed.distribute(
        jnp.zeros(finite_volume.cell_shape), velocity
    )
    distributed_result = phx.solver.MACDistributedProjectionPlan(
        distributed, relative_tolerance=1e-8, absolute_tolerance=1e-8
    ).project(distributed_state, 1.0)

    bounded = phx.discretization.FiniteVolumePlan(
        _grid(max(4, count // 2), periodic=False)
    ).prepare()
    mapped = phx.discretization.MappedMACGeometryPlan(
        bounded, lambda points: points, mapping_id="benchmark-identity"
    ).prepare()
    ale = phx.solver.MACALEGeometryPlan(
        bounded,
        lambda _time, points, _args: points,
        lambda _time, points, _args: jnp.zeros_like(points),
        mapping_id="benchmark-stationary-ale",
    )
    ale_stage = ale.evaluate(0.0)

    method = phx.solver.SSPRK33FixedStepMethod(flow)
    adaptive = phx.solver.MACAdaptiveRolloutPlan(
        flow,
        method,
        phx.solver.MACCompositeStepController(flow),
        phx.solver.MACAdaptivePolicy(4, maximum_step_size=0.0025),
        final_time=0.005,
        initial_step_size=0.0025,
    ).rollout(jnp.asarray(0.0), state)

    values = jnp.asarray(
        [
            boundary_stage.compatibility_defect,
            scalar_balance,
            jnp.abs(scalar_diagnostics.buoyancy.exchange_defect),
            helmholtz.relative_residual,
            variable_diagnostics.divergence_norm,
            jnp.abs(marker_diagnostics.work_adjoint_residual),
            distributed_result.divergence_norm,
            mapped.report.weighted_adjoint_residual,
            ale_stage.maximum_gcl_residual,
        ]
    )
    finite = bool(jnp.all(jnp.isfinite(values)))
    passed = bool(
        finite
        and boundary_stage.successful
        and scalar_diagnostics.projection_converged
        and scalar_diagnostics.scalars.success
        and scalar_diagnostics.buoyancy.success
        and helmholtz.successful
        and variable_diagnostics.projection_converged
        and marker_diagnostics.successful
        and distributed_result.successful
        and mapped.report.passed
        and ale_stage.passed
        and adaptive.successful
        and jnp.max(values) <= 1e-6
    )
    return MACMultiphysicsBenchmarkRecord(
        count=count,
        boundary_mass_defect=float(values[0]),
        scalar_content_balance_defect=float(values[1]),
        buoyancy_exchange_defect=float(values[2]),
        helmholtz_relative_residual=float(values[3]),
        variable_density_divergence=float(values[4]),
        marker_work_adjoint_defect=float(values[5]),
        distributed_divergence=float(values[6]),
        mapped_adjoint_defect=float(values[7]),
        ale_gcl_defect=float(values[8]),
        adaptive_completed=bool(adaptive.successful),
        finite=finite,
        passed=passed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    record = run_mac_multiphysics_benchmark(count=4 if args.smoke else 6)
    payload = json.dumps(asdict(record), indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
