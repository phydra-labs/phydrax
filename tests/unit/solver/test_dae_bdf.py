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
            method=phx.solver.BDFMethod(1),
            nonlinear_termination=_strict_termination(),
        ),
    )
    bdf2 = phx.solver.solve_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            method=phx.solver.BDFMethod(2),
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
    assert jnp.array_equal(
        bdf1.step_history.orders,
        jnp.ones(grid.num_steps, dtype=jnp.int32),
    )
    assert jnp.array_equal(
        bdf2.step_history.orders,
        jnp.asarray((1, 2, 2, 2, 2), dtype=jnp.int32),
    )


def test_prepared_bdf_is_jittable_vmappable_and_implicitly_differentiable():
    problem = _decay_problem()
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 0.4, 5), time_id="bdf-gradient")
    policy = phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(1),
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
            method=phx.solver.BDFMethod(2),
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
    attempts = solution.attempt_history
    assert jnp.all(attempts.valid)
    assert jnp.all(
        attempts.nonlinear_status == int(phx.nonlinear.NonlinearStatus.SUCCESS)
    )
    assert jnp.all(attempts.nonlinear_iterations >= 0)
    assert jnp.all(attempts.residual_evaluations > 0)
    assert jnp.all(attempts.jacobian_preparations > 0)
    assert jnp.all(attempts.linear_solves >= 0)
    assert jnp.all(attempts.numeric_refreshes >= 1)
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
            method=phx.solver.BDFMethod(1),
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
        solution.attempt_history.valid,
        jnp.asarray((True, False, False)),
    )
    assert solution.attempt_history.nonlinear_status[0] != int(
        phx.nonlinear.NonlinearStatus.SUCCESS
    )
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
                method=phx.solver.BDFMethod(2),
                max_step_ratio=2.0,
            ),
        )


def test_bdf_rate_preserves_small_changes_under_exact_state_translation():
    from phydrax.solver._bdf_method import bdf_rate

    times = jnp.asarray((0.0, -0.13, -0.37, -0.51, -0.93))
    history = jnp.arange(5, dtype=jnp.float64) * 2.0**-20
    state = jnp.asarray(2.0**-21)
    target, order = jnp.asarray(0.0031), jnp.asarray(2)
    rate = bdf_rate(state, history, times, target, order)
    translated = bdf_rate(state + 256.0, history + 256.0, times, target, order)
    assert jnp.abs(translated - rate) <= 1e-12 * jnp.abs(rate)


@pytest.mark.parametrize("mode", ("fixed-bdf", "adaptive-bdf", "theta"))
def test_small_implicit_increments_preserve_rates_on_large_state_offsets(mode):
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, power: 100.0 * state_rate - power,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="offset-thermal-increment",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((300.0,)),
        args=jnp.asarray(0.2),
        problem_id="offset-thermal-increment",
    )
    times = jnp.asarray((0.0, 2e-8, 4e-8))
    policy = phx.solver.DAESolvePolicy(
        method=phx.solver.ThetaMethod(0.5, endpoint=True)
        if mode == "theta"
        else phx.solver.BDFMethod(2),
        nonlinear_termination=_strict_termination(),
        adaptive=phx.solver.DAEAdaptivePolicy(
            initial_step=1e-8,
            relative_tolerance=1e-7,
            absolute_tolerance=1e-9,
            maximum_accepted_steps=16,
            maximum_attempts=32,
        )
        if mode == "adaptive-bdf"
        else None,
    )
    prepared = phx.solver.prepare_dae(
        problem,
        phx.dynamics.TimeGrid(times, time_id="offset-thermal-increment"),
        policy=policy,
    )
    solution = phx.solver.solve_dae(prepared)
    assert bool(solution.successful)
    assert bool(jnp.all(solution.valid))
    assert jnp.max(jnp.abs(solution.state_rates[:, 0] - 0.002)) < 1e-12
    assert jnp.max(jnp.abs(solution.states[:, 0] - (300.0 + 0.002 * times))) < 1e-12
    assert jnp.max(solution.residual_norm) < 1e-11

    def terminal(power):
        result = phx.solver.solve_dae(prepared, args=power)
        return jnp.stack((result.states[-1, 0], result.state_rates[-1, 0]))

    _, tangent = jax.jvp(terminal, (jnp.asarray(0.2),), (jnp.asarray(1.0),))
    _, pullback = jax.vjp(terminal, jnp.asarray(0.2))
    expected = jnp.asarray((times[-1] / 100.0, 0.01))
    assert jnp.allclose(tangent, expected, rtol=1e-8, atol=1e-16)
    assert jnp.allclose(
        pullback(jnp.asarray((0.3, -0.7)))[0],
        jnp.dot(expected, jnp.asarray((0.3, -0.7))),
        rtol=1e-8,
        atol=1e-16,
    )
