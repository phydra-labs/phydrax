#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    count = 16
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=boundaries
    ).prepare()
    pressure = phx.solver.MACPressureProjectionPlan(
        operators, boundaries=boundaries, solve_method="transform"
    )
    flow = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        momentum,
        pressure,
    )

    marker_count = 24
    angle = 2.0 * jnp.pi * jnp.arange(marker_count) / marker_count
    marker_position = jnp.stack(
        (0.5 + 0.15 * jnp.cos(angle), 0.5 + 0.15 * jnp.sin(angle)), axis=-1
    )
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(marker_count),
        marker_position,
        jnp.full((marker_count,), 2.0 * jnp.pi * 0.15 / marker_count),
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    immersed_projection = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators,
        transfer,
        boundaries=boundaries,
        tolerance=1.0e-8,
    )

    def stationary_motion(_time, _args):
        return markers.kinematics(marker_position, jnp.zeros_like(marker_position))

    method = phx.solver.MACImmersedBoundaryIMEXEulerMethod(
        flow,
        immersed_projection,
        stationary_motion,
        motion_id="stationary-cylinder",
        fixed_step_size=1.0e-3,
    )
    zero_velocity = tuple(
        jnp.zeros(layout.shape) for layout in finite_volume.face_layouts
    )
    state = flow.project_state(zero_velocity)
    result = method.step(0.0, state)
    print(
        {
            "successful": bool(result.successful),
            "divergence_norm": float(jnp.linalg.norm(result.projection.divergence_after)),
            "maximum_slip": float(
                jnp.max(jnp.linalg.norm(result.projection.marker_slip, axis=-1))
            ),
            "marker_force_norm": float(jnp.linalg.norm(result.marker_force_density)),
        }
    )


if __name__ == "__main__":
    main()
