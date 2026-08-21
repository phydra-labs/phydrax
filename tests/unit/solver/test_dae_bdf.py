import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _decay_problem(*, initial=1.0, parameter=1.0):
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, rate: state_rate + rate * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="scalar-decay",
    )
    return phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((initial,)),
        args=jnp.asarray(parameter),
        problem_id="scalar-decay",
    )


def _strict_termination(*, maximum_steps=20):
    return phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-11,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=maximum_steps,
    )


def test_bdf1_and_bdf2_follow_their_fixed_grid_discrete_maps():
    problem = _decay_problem()
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 0.5, 6), time_id="bdf-maps")
    bdf1 = phx.solver.solve_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            integration_method="bdf1",
            nonlinear_termination=_strict_termination(),
        ),
    )
    bdf2 = phx.solver.solve_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            integration_method="bdf2",
            nonlinear_termination=_strict_termination(),
        ),
    )
    step = grid.durations[0]
    expected_bdf1 = (1.0 + step) ** -jnp.arange(grid.num_times)
    expected_bdf2 = [jnp.asarray(1.0), 1.0 / (1.0 + step)]
    for _ in range(2, grid.num_times):
        expected_bdf2.append(
            (2.0 * expected_bdf2[-1] - 0.5 * expected_bdf2[-2]) / (1.5 + step)
        )

    assert jnp.all(bdf1.valid)
    assert jnp.all(bdf2.valid)
    assert jnp.allclose(bdf1.states[:, 0], expected_bdf1, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(
        bdf2.states[:, 0],
        jnp.asarray(expected_bdf2),
        rtol=1e-9,
        atol=1e-11,
    )
    assert jnp.array_equal(bdf1.orders, jnp.ones(grid.num_steps, dtype=jnp.int32))
    assert jnp.array_equal(
        bdf2.orders,
        jnp.asarray((1, 2, 2, 2, 2), dtype=jnp.int32),
    )


def test_prepared_bdf_is_jittable_vmappable_and_implicitly_differentiable():
    problem = _decay_problem()
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 0.4, 5), time_id="bdf-gradient")
    policy = phx.solver.DAESolvePolicy(
        integration_method="bdf1",
        nonlinear_termination=_strict_termination(),
    )
    prepared = phx.solver.prepare_dae(problem, grid, policy=policy)

    def terminal(parameter):
        return phx.solver.solve_dae(prepared, args=parameter).states[-1, 0]

    parameters = jnp.asarray((0.5, 1.0, 2.0))
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(terminal)))(parameters)
    _, tangent = jax.jvp(
        terminal,
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )
    step = grid.durations[0]
    expected_values = (1.0 + step * parameters) ** -grid.num_steps
    expected_gradients = (
        -grid.num_steps * step * (1.0 + step * parameters) ** (-grid.num_steps - 1)
    )

    assert jnp.allclose(values, expected_values, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(gradients, expected_gradients, rtol=1e-7, atol=1e-9)
    assert jnp.allclose(tangent, expected_gradients[1], rtol=1e-7, atol=1e-9)

    def terminal_from_initial(initial):
        state = jnp.asarray((initial,))
        return phx.solver.solve_dae(
            prepared,
            initial_state=state,
        ).states[-1, 0]

    initial_gradient = jax.jit(jax.grad(terminal_from_initial))(jnp.asarray(1.0))
    assert jnp.allclose(
        initial_gradient,
        (1.0 + step) ** -grid.num_steps,
        rtol=1e-8,
        atol=1e-10,
    )


def test_prepared_solve_reports_native_nonlinear_lifecycle_and_provenance():
    problem = _decay_problem()
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 0.2, 4), time_id="bdf-evidence")
    prepared = phx.solver.prepare_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            integration_method="bdf2",
            nonlinear_termination=_strict_termination(),
        ),
    )
    solution = phx.solver.solve_dae(prepared)

    assert solution.successful
    assert solution.plan_id == prepared.plan.plan_id
    assert solution.prepared_id == prepared.prepared_id
    assert solution.stage_linear_plan_id == prepared.stage_linear_plan_id
    assert (
        solution.initialization_linear_plan_id == prepared.initialization_linear_plan_id
    )
    assert solution.nonlinear_method_id == prepared.plan.policy.nonlinear_method.method_id
    assert jnp.all(solution.nonlinear_status_valid)
    assert jnp.all(
        solution.nonlinear_status == int(phx.nonlinear.NonlinearStatus.SUCCESS)
    )
    assert jnp.all(solution.nonlinear_iterations >= 0)
    assert jnp.all(solution.residual_evaluations > 0)
    assert jnp.all(solution.jacobian_preparations > 0)
    assert jnp.all(solution.linear_solves >= 0)
    assert jnp.all(solution.numeric_refreshes >= 1)
    assert jnp.all(solution.residual_norm <= solution.residual_threshold)


def test_failed_bdf_stage_is_reported_once_and_later_nodes_are_not_run():
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, args: state_rate + state**3,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="nonlinear-stage-failure",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((10.0,)),
        problem_id="nonlinear-stage-failure",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1, 0.2, 0.3)),
        time_id="nonlinear-stage-failure",
    )
    solution = phx.solver.solve_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            integration_method="bdf1",
            nonlinear_termination=_strict_termination(maximum_steps=1),
        ),
    )

    assert jnp.array_equal(
        solution.valid,
        jnp.asarray((True, False, False, False)),
    )
    assert solution.status[1] == int(phx.solver.DAEStatus.NONLINEAR_FAILED)
    assert jnp.array_equal(
        solution.status[2:],
        jnp.full((2,), int(phx.solver.DAEStatus.NOT_RUN), dtype=jnp.int32),
    )
    assert jnp.array_equal(
        solution.nonlinear_status_valid,
        jnp.asarray((True, False, False)),
    )
    assert solution.nonlinear_status[0] != int(phx.nonlinear.NonlinearStatus.SUCCESS)
    assert jnp.all(jnp.isnan(solution.states[2:]))


def test_bdf2_rejects_grid_ratios_outside_declared_stability_contract():
    problem = _decay_problem()
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.01, 0.11)),
        time_id="bad-bdf2-ratio",
    )

    with pytest.raises(ValueError, match="step ratios"):
        phx.solver.plan_dae(
            problem,
            grid,
            policy=phx.solver.DAESolvePolicy(
                integration_method="bdf2",
                max_step_ratio=2.0,
            ),
        )
