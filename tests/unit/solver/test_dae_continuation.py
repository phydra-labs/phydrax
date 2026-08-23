import jax.numpy as jnp
import pytest

import phydrax as phx


def _problem(*, problem_id="adaptive-continuation"):
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, parameter: state_rate + parameter * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="adaptive-continuation-system",
    )
    return phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0,)),
        args=jnp.asarray(1.0),
        problem_id=problem_id,
    )


def _policy():
    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        adaptive=phx.solver.DAEAdaptivePolicy(
            relative_tolerance=1e-5,
            absolute_tolerance=1e-8,
            maximum_accepted_steps=256,
            maximum_attempts=512,
        ),
        failure="status",
    )


def test_segmented_continuation_matches_monolithic_accepted_history():
    problem = _problem()
    policy = _policy()
    full_grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1, 0.2, 0.3, 0.4)),
        time_id="adaptive-continuation-full",
    )
    first_grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1, 0.2)),
        time_id="adaptive-continuation-first",
    )
    second_grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.2, 0.3, 0.4)),
        time_id="adaptive-continuation-second",
    )

    full = phx.solver.solve_dae(problem, full_grid, policy=policy)
    first = phx.solver.solve_dae(problem, first_grid, policy=policy)
    second_prepared = phx.solver.prepare_dae(problem, second_grid, policy=policy)
    second = phx.solver.solve_dae(
        second_prepared,
        continuation=first.continuation,
    )
    first_count = int(first.step_history.count)
    second_count = int(second.step_history.count)
    full_count = int(full.step_history.count)
    segmented_times = jnp.concatenate(
        (
            first.step_history.accepted_times[:first_count],
            second.step_history.accepted_times[:second_count],
        )
    )
    segmented_steps = jnp.concatenate(
        (
            first.step_history.step_sizes[:first_count],
            second.step_history.step_sizes[:second_count],
        )
    )
    segmented_orders = jnp.concatenate(
        (
            first.step_history.orders[:first_count],
            second.step_history.orders[:second_count],
        )
    )

    assert full.successful & first.successful & second.successful
    assert second.initialization.nonlinear_result is None
    assert jnp.allclose(
        jnp.concatenate((first.states, second.states[1:])),
        full.states,
        rtol=1e-11,
        atol=1e-12,
    )
    assert jnp.allclose(segmented_times, full.step_history.accepted_times[:full_count])
    assert jnp.allclose(segmented_steps, full.step_history.step_sizes[:full_count])
    assert jnp.array_equal(segmented_orders, full.step_history.orders[:full_count])


def test_continuation_recertifies_changed_arguments_and_reports_inconsistency():
    problem = _problem()
    policy = _policy()
    first_grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1, 0.2)),
        time_id="adaptive-recertification-first",
    )
    second_grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.2, 0.3)),
        time_id="adaptive-recertification-second",
    )
    first = phx.solver.solve_dae(problem, first_grid, policy=policy)
    second = phx.solver.solve_dae(
        phx.solver.prepare_dae(problem, second_grid, policy=policy),
        args=jnp.asarray(2.0),
        continuation=first.continuation,
    )

    assert second.termination_status == int(
        phx.solver.DAETerminationStatus.CONTINUATION_INCONSISTENT
    )
    assert not jnp.any(second.valid)
    assert second.initialization.status == int(
        phx.solver.DAEInitializationStatus.RESIDUAL_TOO_LARGE
    )


def test_explicit_restart_resets_bdf_history_and_continuation_is_exclusive():
    problem = _problem()
    policy = _policy()
    first_grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1, 0.2)),
        time_id="adaptive-restart-first",
    )
    second_grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.2, 0.3, 0.4)),
        time_id="adaptive-restart-second",
    )
    first = phx.solver.solve_dae(problem, first_grid, policy=policy)
    prepared = phx.solver.prepare_dae(problem, second_grid, policy=policy)
    restarted = phx.solver.solve_dae(
        prepared,
        initial_state=first.states[-1],
        initial_state_rate=first.state_rates[-1],
    )

    assert restarted.successful
    assert restarted.step_history.orders[0] == 1
    with pytest.raises(ValueError, match="continuation cannot be combined"):
        phx.solver.solve_dae(
            prepared,
            initial_state=first.states[-1],
            continuation=first.continuation,
        )


def test_continuation_rejects_incompatible_problem_identity():
    policy = _policy()
    first_problem = _problem(problem_id="continuation-source")
    second_problem = _problem(problem_id="continuation-target")
    first = phx.solver.solve_dae(
        first_problem,
        phx.dynamics.TimeGrid(
            jnp.asarray((0.0, 0.1)),
            time_id="continuation-source",
        ),
        policy=policy,
    )
    second_prepared = phx.solver.prepare_dae(
        second_problem,
        phx.dynamics.TimeGrid(
            jnp.asarray((0.1, 0.2)),
            time_id="continuation-target",
        ),
        policy=policy,
    )

    with pytest.raises(ValueError, match="problem identity"):
        phx.solver.solve_dae(
            second_prepared,
            continuation=first.continuation,
        )
