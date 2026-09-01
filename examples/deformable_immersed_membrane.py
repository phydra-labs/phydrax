#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit elastic marker membrane accelerated by a periodic background flow."""

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(10, periodic=True) for _ in range(2)
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
    dynamics = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 2.0e-2),
        momentum,
        pressure,
    )

    reference = jnp.asarray([[0.42, 0.42], [0.58, 0.42], [0.58, 0.58], [0.42, 0.58]])
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(4), reference, jnp.full((4,), 0.04)
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    projection = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators,
        transfer,
        boundaries=boundaries,
        tolerance=1.0e-8,
    )
    configuration_space = phx.linalg.ArraySpace((8,))
    marker_map = phx.discretization.FiniteElementImmersedMarkerMapPlan(
        markers, configuration_space, jnp.eye(8)
    ).prepare()
    reference_coordinates = reference.reshape((-1,))
    structure = phx.dynamics.SecondOrderDifferentialSystem(
        lambda _time, _q, _velocity, acceleration, _args: acceleration,
        state_shape=(8,),
        system_id="free-immersed-membrane",
    )
    backward_euler = phx.solver.MACDeformableImmersedBackwardEulerMethod(
        dynamics,
        projection,
        marker_map,
        structure,
        lambda q, _args: jnp.sum(0.0 * q),
        2.0e-3,
        energy_id="free-membrane-energy",
    )
    method = phx.solver.MACDeformableImmersedNewmarkMethod(backward_euler)
    background = (
        jnp.full(finite_volume.face_layouts[0].shape, 0.05),
        jnp.zeros(finite_volume.face_layouts[1].shape),
    )
    state = method.initialize(
        dynamics.project_state(background),
        reference_coordinates,
        jnp.broadcast_to(jnp.asarray([0.05, 0.0]), reference.shape).reshape((-1,)),
    )
    result = method.step(0.0, state)
    if not bool(result.accepted):
        raise RuntimeError(f"Membrane step failed with status {int(result.status)}.")
    print(
        {
            "time": float(result.accepted_state.time),
            "maximum_displacement": float(
                jnp.max(
                    jnp.abs(result.accepted_state.configuration - reference_coordinates)
                )
            ),
            "slip_norm": float(result.slip_norm),
            "energy_change": float(result.energy.total_energy_change),
        }
    )


if __name__ == "__main__":
    main()
