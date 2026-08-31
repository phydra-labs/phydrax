#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_prescribed_immersed_imex_pipeline_accepts_fixed_zero_state():
    count = 8
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for _ in range(2)
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
        phx.equations.IncompressibleFlowProblem(2, 0.01), momentum, pressure
    )
    position = jnp.asarray(
        [[0.35, 0.35], [0.65, 0.35], [0.65, 0.65], [0.35, 0.65]]
    )
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(4), position, jnp.full((4,), 0.25)
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(
        operators, markers
    ).prepare()
    immersed = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators, transfer, boundaries=boundaries, tolerance=1.0e-8
    )

    def motion(_time, _args):
        return markers.kinematics(position, jnp.zeros_like(position))

    method = phx.solver.MACImmersedBoundaryIMEXEulerMethod(
        flow,
        immersed,
        motion,
        motion_id="fixed-zero",
        fixed_step_size=1.0e-3,
    )
    zero_velocity = tuple(
        jnp.zeros(layout.shape) for layout in finite_volume.face_layouts
    )
    state = flow.project_state(zero_velocity)
    first = method.step(0.0, state)
    second = method.step(
        first.time,
        first.state,
        pressure=first.pressure,
        marker_force_density=first.marker_force_density,
    )
    sbdf = phx.solver.MACImmersedBoundarySBDF2Method(
        flow,
        immersed,
        motion,
        1.0e-3,
        motion_id="fixed-zero-sbdf2",
    )
    startup = sbdf.initialize(0.0, state)
    advanced = sbdf.step(startup.history)

    assert first.successful
    assert second.successful
    assert jnp.linalg.norm(second.projection.divergence_after) < 1.0e-8
    assert jnp.linalg.norm(second.projection.marker_slip) < 1.0e-8
    assert startup.successful
    assert advanced.successful
    assert jnp.linalg.norm(advanced.projection.divergence_after) < 1.0e-8
    assert jnp.linalg.norm(advanced.projection.marker_slip) < 1.0e-8
