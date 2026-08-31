#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _compiled(count=6):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0 * jnp.pi, 2.0 * jnp.pi]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="transform", tolerance=1e-10
    )
    compiled = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.1), momentum, projection
    )
    x_faces = finite_volume.face_centers[0]
    y_faces = finite_volume.face_centers[1]
    velocity = (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]),
    )
    return operators, momentum, compiled, compiled.project_state(velocity)


def test_component_helmholtz_and_imex_euler_are_fail_closed_and_divergence_free():
    operators, momentum, compiled, state = _compiled()
    stage = momentum.boundaries.evaluate(0.01)
    velocity = compiled.unpack_velocity(state)
    helmholtz = phx.solver.MACHelmholtzSolvePlan(
        momentum,
        solve_method="transform",
        fixed_mass_coefficient=1.0,
        fixed_diffusion_coefficient=0.001,
    )
    solved = helmholtz.solve(velocity, stage)
    method = phx.solver.MACIMEXEulerMethod(
        compiled, fixed_step_size=0.01, solve_method="transform"
    )
    step = method.step(0.0, state)

    assert solved.successful
    assert solved.residual_norm < 1e-8
    assert all(route == "transform" for route in solved.routes)
    assert step.successful
    assert jnp.linalg.norm(operators.divergence(step.velocity)) < 1e-8
    assert jnp.all(jnp.isfinite(step.state))


def test_mac_sbdf2_startup_and_history_step_complete():
    operators, _, compiled, state = _compiled()
    method = phx.solver.MACSBDF2Method(
        compiled, 0.005, solve_method="transform", tolerance=1e-9
    )
    startup = method.initialize(0.0, state)
    following = method.step(startup.history)

    assert startup.successful
    assert following.successful
    assert following.history.accepted_steps == 2
    assert jnp.linalg.norm(operators.divergence(following.velocity)) < 1e-8
    assert jnp.isclose(following.pressure_correction_coefficient, 2.0 * 0.005 / 3.0)


def test_transform_line_solver_matches_its_physical_action():
    representation = phx.linalg.TransformLineRepresentation(
        (phx.linalg.FFTLinearTransform(4),),
        1,
        -jnp.ones(2),
        2.0 * jnp.ones(3),
        -jnp.ones(2),
        jnp.asarray([0.0, 2.0, 4.0, 2.0]),
    )
    prepared = phx.linalg.TransformLineSolvePlan(
        representation,
        operator_scale=1.0,
        diagonal_shift=1.0,
        maximum_resource_bytes=1_000_000,
    ).prepare()
    expected = jnp.arange(12.0).reshape((4, 3)) / 13.0
    right_hand_side = representation.apply(expected) + expected
    result = prepared.solve(right_hand_side)

    assert representation.report.exact
    assert result.converged
    assert result.relative_residual < 1e-9
    assert jnp.allclose(result.value, expected, rtol=1e-8, atol=1e-8)
    assert result.resources.total_bytes <= 1_000_000
