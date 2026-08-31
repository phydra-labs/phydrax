#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_periodic_mac_taylor_green_pipeline_preserves_constraint_and_decay():
    count = 12
    viscosity = 0.02
    final_time = 0.01
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0 * jnp.pi, 2.0 * jnp.pi]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="transform",
        tolerance=1e-10,
    )
    compiled = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, viscosity),
        momentum,
        projection,
    )
    x_faces = finite_volume.face_centers[0]
    y_faces = finite_volume.face_centers[1]
    initial_velocity = (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]),
    )
    initial_state = compiled.project_state(initial_velocity)
    solution = phx.solver.solve_fixed_step(
        phx.solver.FixedStepProblem(
            phx.solver.SSPRK33FixedStepMethod(compiled),
            initial_state,
            t0=0.0,
            t1=final_time,
            step_size=0.002,
            discretization_bundle=compiled.discretization_bundle,
        )
    )
    final_velocity = compiled.unpack_velocity(solution.states[-1])
    amplitude = jnp.exp(-2.0 * viscosity * final_time)

    assert solution.successful
    assert jnp.linalg.norm(operators.divergence(final_velocity)) < 2e-9
    np.testing.assert_allclose(
        final_velocity[0], amplitude * initial_velocity[0], rtol=3e-4, atol=3e-4
    )
    np.testing.assert_allclose(
        final_velocity[1], amplitude * initial_velocity[1], rtol=3e-4, atol=3e-4
    )
