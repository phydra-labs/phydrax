import jax.numpy as jnp

import phydrax as phx


def _policy(*, regularity, reuse=None, initial_step=None):
    return phx.solver.DAESolvePolicy(
        adaptive=phx.solver.DAEAdaptivePolicy(
            relative_tolerance=1e-4,
            absolute_tolerance=1e-7,
            initial_step=initial_step,
            maximum_accepted_steps=128,
            maximum_attempts=256,
        ),
        temporal_reuse=reuse,
        regularity=regularity,
        failure="status",
    )


def _scalar_problem(residual, *, identity):
    system = phx.dynamics.DifferentialAlgebraicSystem(
        residual,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id=identity,
    )
    return phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((1.0,)),
        initial_state_rate=jnp.zeros(1),
        args=jnp.asarray(1.0),
        problem_id=identity,
    )


def test_periodic_regularity_verifies_consistency_and_bdf_stage_operators():
    problem = _scalar_problem(
        lambda time, state, state_rate, parameter: state_rate + parameter * state,
        identity="adaptive-regularity",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1, 0.2)),
        time_id="adaptive-regularity",
    )
    solution = phx.solver.solve_dae(
        problem,
        grid,
        policy=_policy(regularity=phx.solver.DAERegularityPolicy("periodic", interval=1)),
    )
    count = int(solution.step_history.count)
    statuses = solution.regularity.stage_status[:count]

    assert solution.successful
    assert solution.regularity.consistency_status == int(
        phx.solver.DAERegularityStatus.VERIFIED
    )
    assert solution.regularity.consistency_rank == 1
    assert solution.regularity.consistency_condition_estimate == 1.0
    assert jnp.all(solution.regularity.stage_valid[:count])
    assert jnp.all(statuses == int(phx.solver.DAERegularityStatus.VERIFIED))
    assert jnp.all(solution.regularity.stage_rank[:count] == 1)
    assert jnp.all(solution.regularity.stage_condition_estimate[:count] == 1.0)


def test_singular_local_operator_is_recorded_or_promoted_to_terminal_failure():
    template = _scalar_problem(
        lambda time, state, state_rate, parameter: state_rate - 10.0 * state,
        identity="adaptive-singular-regularity",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        template.system,
        jnp.zeros(1),
        initial_state_rate=jnp.zeros(1),
        args=template.args,
        problem_id=template.problem_id,
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1)),
        time_id="adaptive-singular-regularity",
    )
    recorded = phx.solver.solve_dae(
        problem,
        grid,
        policy=_policy(
            regularity=phx.solver.DAERegularityPolicy(
                "periodic",
                interval=1,
                failure="record",
            ),
            initial_step=0.1,
        ),
    )
    failed = phx.solver.solve_dae(
        problem,
        grid,
        policy=_policy(
            regularity=phx.solver.DAERegularityPolicy(
                "periodic",
                interval=1,
                failure="status",
            ),
            initial_step=0.1,
        ),
    )

    assert recorded.successful
    assert recorded.regularity.consistency_status == int(
        phx.solver.DAERegularityStatus.VERIFIED
    )
    assert recorded.regularity.consistency_rank == 1
    assert recorded.regularity.stage_status[0] == int(
        phx.solver.DAERegularityStatus.NUMERICALLY_SINGULAR
    )
    assert failed.termination_status == int(
        phx.solver.DAETerminationStatus.REGULARITY_FAILED
    )
    assert failed.attempt_history.status[0] == int(
        phx.solver.DAEAttemptStatus.REGULARITY_REJECTED
    )
    assert not failed.valid[1]


def test_singular_consistency_operator_can_stop_before_stage_execution():
    problem = _scalar_problem(
        lambda time, state, state_rate, parameter: state,
        identity="adaptive-singular-consistency",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        problem.system,
        jnp.zeros(1),
        initial_state_rate=jnp.zeros(1),
        args=problem.args,
        problem_id=problem.problem_id,
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1)),
        time_id="adaptive-singular-consistency",
    )

    solution = phx.solver.solve_dae(
        problem,
        grid,
        policy=_policy(
            regularity=phx.solver.DAERegularityPolicy(
                "periodic",
                failure="status",
            )
        ),
    )

    assert solution.termination_status == int(
        phx.solver.DAETerminationStatus.REGULARITY_FAILED
    )
    assert solution.regularity.consistency_status == int(
        phx.solver.DAERegularityStatus.NUMERICALLY_SINGULAR
    )
    assert solution.regularity.consistency_rank == 0
    assert solution.attempt_history.count == 0
    assert jnp.array_equal(solution.valid, jnp.asarray((True, False)))


def test_fixed_grid_periodic_regularity_uses_explicit_operator_probes():
    problem = _scalar_problem(
        lambda time, state, state_rate, parameter: state_rate + parameter * state,
        identity="fixed-periodic-regularity",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1, 0.2)),
        time_id="fixed-periodic-regularity",
    )

    solution = phx.solver.solve_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            regularity=phx.solver.DAERegularityPolicy("periodic"),
        ),
    )

    assert solution.successful
    assert solution.regularity.consistency_status == int(
        phx.solver.DAERegularityStatus.VERIFIED
    )
    assert jnp.all(solution.regularity.stage_valid)
    assert jnp.all(
        solution.regularity.stage_status == int(phx.solver.DAERegularityStatus.VERIFIED)
    )
    assert jnp.all(solution.regularity.stage_rank == 1)


def test_temporal_reuse_preserves_values_and_reduces_jacobian_preparations():
    problem = _scalar_problem(
        lambda time, state, state_rate, parameter: state_rate + parameter * state,
        identity="adaptive-temporal-reuse",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 0.5, 6),
        time_id="adaptive-temporal-reuse",
    )
    regularity = phx.solver.DAERegularityPolicy("solver-evidence")
    reused = phx.solver.solve_dae(
        problem,
        grid,
        policy=_policy(
            regularity=regularity,
            reuse=phx.solver.DAETemporalReusePolicy(
                enabled=True,
                maximum_jacobian_age=2,
                maximum_alpha_ratio=1.25,
                refresh_after_iterations=3,
            ),
        ),
    )
    refreshed = phx.solver.solve_dae(
        problem,
        grid,
        policy=_policy(
            regularity=regularity,
            reuse=phx.solver.DAETemporalReusePolicy(enabled=False),
        ),
    )

    assert reused.successful & refreshed.successful
    assert jnp.allclose(reused.states, refreshed.states, rtol=2e-7, atol=2e-9)
    assert jnp.sum(reused.attempt_history.jacobian_preparations) < jnp.sum(
        refreshed.attempt_history.jacobian_preparations
    )
    assert jnp.sum(reused.attempt_history.stale_jacobian_retries) == 0
