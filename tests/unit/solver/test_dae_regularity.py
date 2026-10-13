from typing import Any

import jax.numpy as jnp
import pytest

import phydrax as phx


def _policy(*, regularity: Any, reuse: Any = None, initial_step: Any = None) -> Any:
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


def _scalar_problem(residual: Any, *, identity: Any) -> Any:
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


def test_dae_regularity_scenario_1() -> None:
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


def test_dae_regularity_scenario_2() -> None:
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
    first = phx.solver.DAERegularityDomain(
        jnp.asarray(((0.0, 1.0),)),
        jnp.asarray(((1.0, 2.0),)),
        domain_id="same-user-domain",
    )
    shifted = phx.solver.DAERegularityDomain(
        jnp.asarray(((0.5, 1.0),)),
        jnp.asarray(((1.5, 2.0),)),
        domain_id="same-user-domain",
    )

    assert first.domain_id != shifted.domain_id


@pytest.mark.parametrize("batch", [None, 1, 2])
def test_fixed_probe_batches_preserve_order_and_requested_singular_probe(
    batch: int | None,
) -> None:
    def residual(time: Any, state: Any, rate: Any, parameter: Any) -> Any:
        del parameter
        coefficient = jnp.where(time == 0.375, 8.0, 0.0)
        return rate - coefficient * state

    template = _scalar_problem(residual, identity="requested-singular-probe")
    problem = phx.solver.DifferentialAlgebraicProblem(
        template.system,
        jnp.zeros(1),
        initial_state_rate=jnp.zeros(1),
        args=template.args,
        problem_id=template.problem_id,
    )
    grid = phx.dynamics.TimeGrid(
        jnp.arange(5, dtype=jnp.float64) * 0.125,
        time_id="requested-singular-probe",
    )
    solution = phx.solver.solve_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            method=phx.solver.BDFMethod(1),
            regularity=phx.solver.DAERegularityPolicy(
                "periodic",
                interval=2,
                probe_batch_size=batch,
                failure="status",
            ),
        ),
    )
    assert solution.termination_status == int(
        phx.solver.DAETerminationStatus.REGULARITY_FAILED
    )
    assert jnp.array_equal(
        solution.regularity.stage_valid, jnp.asarray([True, False, True, False])
    )
    assert jnp.array_equal(solution.regularity.stage_rank, jnp.asarray([1, -1, 0, -1]))
    assert jnp.array_equal(
        solution.regularity.stage_status,
        jnp.asarray(
            [
                int(phx.solver.DAERegularityStatus.VERIFIED),
                int(phx.solver.DAERegularityStatus.NOT_RUN),
                int(phx.solver.DAERegularityStatus.NUMERICALLY_SINGULAR),
                int(phx.solver.DAERegularityStatus.NOT_RUN),
            ]
        ),
    )


def test_probe_batch_policy_validation_and_plan_identity() -> None:
    with pytest.raises(ValueError, match="probe_batch_size"):
        phx.solver.DAERegularityPolicy("periodic", probe_batch_size=0)
    problem = _scalar_problem(
        lambda time, state, rate, parameter: rate + parameter * state,
        identity="probe-policy-identity",
    )
    grid = phx.dynamics.TimeGrid(jnp.asarray([0.0, 0.1]), time_id="probe-policy-identity")
    dense = phx.solver.plan_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            regularity=phx.solver.DAERegularityPolicy("periodic"),
        ),
    )
    bounded = phx.solver.plan_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            regularity=phx.solver.DAERegularityPolicy("periodic", probe_batch_size=1),
        ),
    )
    assert dense.plan_id != bounded.plan_id


@pytest.mark.parametrize("requested_count", [0, 1])
def test_bounded_probe_empty_and_single_archives(requested_count: int) -> None:
    residual = (
        (lambda time, state, rate, parameter: state)
        if requested_count == 0
        else (lambda time, state, rate, parameter: rate + parameter * state)
    )
    problem = _scalar_problem(residual, identity="probe-empty-single")
    grid = phx.dynamics.TimeGrid(jnp.asarray([0.0, 0.125]), time_id="probe-empty-single")
    solutions = tuple(
        phx.solver.solve_dae(
            problem,
            grid,
            policy=phx.solver.DAESolvePolicy(
                method=phx.solver.BDFMethod(1),
                regularity=phx.solver.DAERegularityPolicy(
                    "periodic", probe_batch_size=batch
                ),
            ),
        )
        for batch in (None, 1)
    )
    parallel, bounded = solutions
    assert jnp.sum(bounded.regularity.stage_valid) == requested_count
    assert jnp.array_equal(
        parallel.regularity.stage_valid, bounded.regularity.stage_valid
    )
    assert jnp.array_equal(
        parallel.regularity.stage_status, bounded.regularity.stage_status
    )
    assert jnp.array_equal(parallel.regularity.stage_rank, bounded.regularity.stage_rank)
    assert jnp.array_equal(
        parallel.regularity.stage_condition_estimate,
        bounded.regularity.stage_condition_estimate,
        equal_nan=True,
    )
