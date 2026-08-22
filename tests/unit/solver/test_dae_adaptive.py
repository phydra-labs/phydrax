import jax.numpy as jnp

import phydrax as phx


def _decay_problem(*, system_id="adaptive-decay"):
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, parameter: state_rate + parameter * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id=system_id,
    )
    return phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0,)),
        args=jnp.asarray(1.0),
        problem_id=system_id,
    )


def _adaptive_policy(**overrides):
    values = {
        "relative_tolerance": 1e-5,
        "absolute_tolerance": 1e-8,
        "initial_step": 0.1,
        "maximum_accepted_steps": 256,
        "maximum_attempts": 512,
    }
    values.update(overrides)
    return phx.solver.DAESolvePolicy(
        integration_method="bdf2",
        adaptive=phx.solver.DAEAdaptivePolicy(**values),
        failure="status",
    )


def test_adaptive_bdf_accepts_certified_steps_and_lands_on_every_save_time():
    problem = _decay_problem()
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 0.5, 6),
        time_id="adaptive-decay",
    )
    policy = _adaptive_policy()

    solution = phx.solver.solve_dae(problem, grid, policy=policy)
    count = int(solution.step_history.count)
    attempt_count = int(solution.attempt_history.count)
    steps = solution.step_history.step_sizes[:count]
    orders = solution.step_history.orders[:count]
    errors = solution.step_history.error_ratios[:count]
    ratios = steps[1:] / steps[:-1]
    second_order = orders[1:] == 2

    assert solution.successful
    assert solution.termination_status == int(phx.solver.DAETerminationStatus.SUCCESS)
    assert jnp.allclose(
        solution.states[:, 0],
        jnp.exp(-grid.times),
        rtol=1e-4,
        atol=2e-6,
    )
    assert jnp.all(errors <= 1.0)
    assert jnp.any(orders == 2)
    assert jnp.all(ratios[second_order] <= policy.max_step_ratio)
    assert jnp.all(ratios[second_order] >= 1.0 / policy.max_step_ratio)
    assert attempt_count > count
    assert jnp.any(
        solution.attempt_history.status[:attempt_count]
        != int(phx.solver.DAEAttemptStatus.ACCEPTED)
    )
    for save_index in range(1, grid.num_points):
        step_index = int(solution.step_history.save_step_indices[save_index])
        assert step_index >= 0
        assert solution.step_history.accepted_times[step_index] == grid.times[save_index]


def test_adaptive_error_control_excludes_algebraic_variables_but_certifies_constraints():
    frequency = 25.0
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, parameter: jnp.asarray(
            (
                state_rate[0] + state[0],
                state[1] - jnp.sin(parameter * state[0]),
            )
        ),
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
        system_id="adaptive-algebraic-error-mask",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0, jnp.sin(frequency))),
        initial_state_rate=jnp.zeros(2),
        args=jnp.asarray(frequency),
        problem_id="adaptive-algebraic-error-mask",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1, 0.2)),
        time_id="adaptive-algebraic-error-mask",
    )
    policy = _adaptive_policy(
        relative_tolerance=jnp.asarray((1e-4, 1e-14)),
        absolute_tolerance=jnp.asarray((1e-7, 1e-14)),
        initial_step=None,
        constraint_tolerance=1e-8,
        residual_tolerance=1e-8,
    )

    solution = phx.solver.solve_dae(problem, grid, policy=policy)

    assert solution.successful
    assert jnp.allclose(solution.states[:, 0], jnp.exp(-grid.times), rtol=5e-4)
    assert jnp.allclose(
        solution.states[:, 1],
        jnp.sin(frequency * solution.states[:, 0]),
        atol=1e-10,
    )
    assert jnp.all(solution.constraint_norm <= policy.adaptive.constraint_tolerance)
    assert solution.step_history.count < 64


def test_adaptive_capacity_failure_preserves_unsaved_nodes_as_not_run():
    problem = _decay_problem(system_id="adaptive-capacity")
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1, 0.2)),
        time_id="adaptive-capacity",
    )
    policy = _adaptive_policy(
        initial_step=1e-3,
        maximum_accepted_steps=2,
        maximum_attempts=4,
    )

    solution = phx.solver.solve_dae(problem, grid, policy=policy)

    assert solution.termination_status == int(
        phx.solver.DAETerminationStatus.MAXIMUM_ACCEPTED_STEPS_REACHED
    )
    assert jnp.array_equal(solution.valid, jnp.asarray((True, False, False)))
    assert jnp.all(solution.status[1:] == int(phx.solver.DAEStatus.NOT_RUN))
    assert jnp.all(jnp.isnan(solution.states[1:]))


def test_adaptive_step_within_roundoff_of_save_boundary_is_snapped():
    problem = _decay_problem(system_id="adaptive-save-roundoff")
    interval = jnp.asarray(0.0025)
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, interval, 2.0 * interval)),
        time_id="adaptive-save-roundoff",
    )
    policy = _adaptive_policy(
        relative_tolerance=1e-4,
        absolute_tolerance=1e-7,
        initial_step=float(jnp.nextafter(interval, 0.0)),
        maximum_accepted_steps=16,
        maximum_attempts=32,
    )

    solution = phx.solver.solve_dae(problem, grid, policy=policy)
    count = int(solution.step_history.count)
    accepted_times = solution.step_history.accepted_times[:count]

    assert solution.successful
    assert solution.termination_status == int(phx.solver.DAETerminationStatus.SUCCESS)
    assert jnp.all(jnp.isin(grid.times[1:], accepted_times))
    assert jnp.min(solution.step_history.step_sizes[:count]) > 1e-6
