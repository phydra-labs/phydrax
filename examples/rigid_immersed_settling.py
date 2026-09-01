#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One accepted-time step of a freely settling two-dimensional immersed body."""

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(6, periodic=True) for _ in range(2)),
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
    dynamics = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 1.0e-2),
        momentum,
        pressure,
    )

    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.asarray([1.0]), ambient_dimension=2
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.asarray([0]), jnp.asarray([0.1])
    ).prepare(particles)
    offsets = jnp.asarray([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(4), offsets, jnp.full((4,), 0.25)
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    marker_map = phx.discretization.RigidMarkerMapPlan(
        markers,
        bodies,
        jnp.zeros((4,), dtype=jnp.int32),
    ).prepare()
    projection = phx.solver.MACRigidImmersedProjectionPlan(
        dynamics,
        marker_map,
        transfer,
        constraint_length=1.0 / finite_volume.cell_shape[0],
        tolerance=2.5e-7,
    )
    base = phx.solver.MACRigidImmersedEulerMethod(dynamics, projection, 1.0e-3)
    method = phx.solver.MACRigidImmersedBackwardEulerMethod(
        base,
        maximum_iterations=8,
        tolerance=2.5e-7,
    )
    zero_velocity = tuple(
        jnp.zeros(layout.shape) for layout in finite_volume.face_layouts
    )
    body = bodies.kinematics(
        jnp.asarray([[0.5, 0.65]]),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
    )
    result = method.step(
        0.0,
        dynamics.project_state(zero_velocity),
        body,
        body_load=phx.discretization.RigidBodyLoad(
            jnp.asarray([[0.0, -1.0e-3]]), jnp.zeros((1, 1))
        ),
    )
    if not bool(result.accepted):
        raise RuntimeError(
            "Settling step failed: "
            f"status={int(result.status)}, "
            f"divergence={float(result.projection.divergence_norm)}, "
            f"slip={float(result.projection.slip_norm)}, "
            f"KKT={float(result.projection.kkt_residual_norm)}."
        )
    print(
        {
            "time": float(result.time),
            "vertical_velocity": float(result.body_kinematics.velocity[0, 1]),
            "slip_norm": float(result.projection.slip_norm),
            "coupling_power_residual": float(result.energy.coupling_power_residual),
        }
    )


if __name__ == "__main__":
    main()
