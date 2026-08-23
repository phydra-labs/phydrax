import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _prepared(*, replay, problem_id):
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, parameter: state_rate + parameter * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id=problem_id,
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0,)),
        args=jnp.asarray(1.0),
        problem_id=problem_id,
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 0.4, 5),
        time_id=problem_id,
    )
    policy = phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        adaptive=phx.solver.DAEAdaptivePolicy(
            relative_tolerance=1e-6,
            absolute_tolerance=1e-9,
            maximum_accepted_steps=256,
            maximum_attempts=512,
        ),
        replay=replay,
        failure="status",
    )
    return phx.solver.prepare_dae(problem, grid, policy=policy)


def test_frozen_grid_replay_jvp_vjp_and_vmap_match_continuous_sensitivity():
    prepared = _prepared(
        replay=phx.solver.DAEReplayPolicy("full"),
        problem_id="adaptive-replay-derivatives",
    )

    def terminal(parameter):
        return phx.solver.solve_dae(prepared, args=parameter).states[-1, 0]

    parameters = jnp.asarray((0.5, 1.0, 1.5))
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(terminal)))(parameters)
    _, tangent = jax.jvp(
        terminal,
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )
    value, pullback = jax.vjp(terminal, jnp.asarray(1.0))
    cotangent = pullback(jnp.asarray(1.0))[0]
    expected_values = jnp.exp(-0.4 * parameters)
    expected_gradients = -0.4 * expected_values

    assert jnp.allclose(values, expected_values, rtol=2e-4, atol=2e-6)
    assert jnp.allclose(gradients, expected_gradients, rtol=5e-4, atol=3e-6)
    assert jnp.allclose(tangent, expected_gradients[1], rtol=5e-4, atol=3e-6)
    assert jnp.allclose(cotangent, tangent, rtol=1e-10, atol=1e-11)
    assert jnp.allclose(value, values[1], rtol=1e-10, atol=1e-11)


def test_frozen_grid_replay_differentiates_semiexplicit_constraints():
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, parameter: jnp.asarray(
            (
                state_rate[0] + parameter * state[0],
                state[1] - parameter * state[0],
            )
        ),
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
        system_id="adaptive-replay-semiexplicit",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0, 1.0)),
        initial_state_rate=jnp.zeros(2),
        args=jnp.asarray(1.0),
        problem_id="adaptive-replay-semiexplicit",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 0.3, 4),
        time_id="adaptive-replay-semiexplicit",
    )
    prepared = phx.solver.prepare_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            adaptive=phx.solver.DAEAdaptivePolicy(
                relative_tolerance=1e-6,
                absolute_tolerance=1e-9,
                maximum_accepted_steps=256,
                maximum_attempts=512,
            ),
            failure="status",
        ),
    )

    def terminal_constraint(parameter):
        solution = phx.solver.solve_dae(prepared, args=parameter)
        return solution.states[-1, 1]

    parameter = jnp.asarray(1.0)
    value, gradient = jax.jit(jax.value_and_grad(terminal_constraint))(parameter)
    expected_value = parameter * jnp.exp(-0.3 * parameter)
    expected_gradient = jnp.exp(-0.3 * parameter) * (1.0 - 0.3 * parameter)

    assert jnp.allclose(value, expected_value, rtol=3e-4, atol=3e-6)
    assert jnp.allclose(gradient, expected_gradient, rtol=8e-4, atol=5e-6)


def test_segmented_replay_gradient_matches_monolithic_frozen_grid_gradient():
    replay = phx.solver.DAEReplayPolicy("chunked", chunk_size=5)
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, parameter: state_rate + parameter * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="adaptive-replay-segmented",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0,)),
        args=jnp.asarray(1.0),
        problem_id="adaptive-replay-segmented",
    )
    policy = phx.solver.DAESolvePolicy(
        adaptive=phx.solver.DAEAdaptivePolicy(
            relative_tolerance=1e-6,
            absolute_tolerance=1e-9,
            maximum_accepted_steps=256,
            maximum_attempts=512,
        ),
        replay=replay,
        failure="status",
    )
    first = phx.solver.prepare_dae(
        problem,
        phx.dynamics.TimeGrid(
            jnp.asarray((0.0, 0.1, 0.2)),
            time_id="adaptive-replay-segmented-first",
        ),
        policy=policy,
    )
    second = phx.solver.prepare_dae(
        problem,
        phx.dynamics.TimeGrid(
            jnp.asarray((0.2, 0.3, 0.4)),
            time_id="adaptive-replay-segmented-second",
        ),
        policy=policy,
    )
    monolithic = phx.solver.prepare_dae(
        problem,
        phx.dynamics.TimeGrid(
            jnp.linspace(0.0, 0.4, 5),
            time_id="adaptive-replay-segmented-full",
        ),
        policy=policy,
    )

    def segmented(parameter):
        leading = phx.solver.solve_dae(first, args=parameter)
        trailing = phx.solver.solve_dae(
            second,
            args=parameter,
            continuation=leading.continuation,
        )
        return trailing.states[-1, 0]

    def full(parameter):
        return phx.solver.solve_dae(monolithic, args=parameter).states[-1, 0]

    parameter = jnp.asarray(1.0)
    segmented_result = jax.jit(jax.value_and_grad(segmented))(parameter)
    full_result = jax.jit(jax.value_and_grad(full))(parameter)

    assert jnp.allclose(segmented_result[0], full_result[0], rtol=1e-11, atol=1e-12)
    assert jnp.allclose(segmented_result[1], full_result[1], rtol=2e-10, atol=2e-11)


def test_chunked_replay_matches_full_replay_values_and_gradients():
    full = _prepared(
        replay=phx.solver.DAEReplayPolicy("full"),
        problem_id="adaptive-replay-full",
    )
    chunked = _prepared(
        replay=phx.solver.DAEReplayPolicy("chunked", chunk_size=7),
        problem_id="adaptive-replay-chunked",
    )

    def evaluate(prepared, parameter):
        solution = phx.solver.solve_dae(prepared, args=parameter)
        return solution.states[-1, 0]

    full_value, full_gradient = jax.jit(
        jax.value_and_grad(lambda parameter: evaluate(full, parameter))
    )(jnp.asarray(1.0))
    chunked_value, chunked_gradient = jax.jit(
        jax.value_and_grad(lambda parameter: evaluate(chunked, parameter))
    )(jnp.asarray(1.0))
    chunked_solution = phx.solver.solve_dae(chunked, args=jnp.asarray(1.0))

    assert jnp.allclose(chunked_value, full_value, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(chunked_gradient, full_gradient, rtol=1e-11, atol=1e-12)
    assert chunked_solution.replay.checkpointing == "chunked"
    assert chunked_solution.replay.selected_chunk_size == 7
    assert chunked_solution.replay.estimated_memory_bytes < full.plan.replay_memory_bytes


def test_replay_memory_budget_selects_a_feasible_chunk_or_fails_at_planning():
    budget = 4096
    prepared = _prepared(
        replay=phx.solver.DAEReplayPolicy(
            "chunked",
            memory_budget_bytes=budget,
        ),
        problem_id="adaptive-replay-budget",
    )

    assert prepared.plan.replay_chunk_size >= 1
    assert prepared.plan.replay_memory_bytes <= budget

    with pytest.raises(ValueError, match="minimum feasible"):
        _prepared(
            replay=phx.solver.DAEReplayPolicy(
                "chunked",
                memory_budget_bytes=1,
            ),
            problem_id="adaptive-replay-budget-failure",
        )


def test_failed_adaptive_primal_has_no_valid_derivative():
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, parameter: state_rate + parameter * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="adaptive-replay-failure",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0,)),
        args=jnp.asarray(1.0),
        problem_id="adaptive-replay-failure",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1, 0.2)),
        time_id="adaptive-replay-failure",
    )
    prepared = phx.solver.prepare_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            adaptive=phx.solver.DAEAdaptivePolicy(
                relative_tolerance=1e-8,
                absolute_tolerance=1e-11,
                initial_step=1e-3,
                maximum_accepted_steps=2,
                maximum_attempts=4,
            ),
            failure="status",
        ),
    )

    def terminal(parameter):
        return phx.solver.solve_dae(prepared, args=parameter).states[-1, 0]

    value, tangent = jax.jvp(
        terminal,
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )

    assert jnp.isnan(value)
    assert jnp.isnan(tangent)
