import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.solver._dae_events import _DAEEventRootArguments
from phydrax.solver._dae_initialization import _DAEInitializationArguments


def _positive_residual(time, state, rate, args):
    checked = eqx.error_if(state, jnp.any(state <= 0.0), "invalid DAE residual evaluated")
    return rate + jnp.log(checked) - args[0]


def _system(**hooks):
    return phx.dynamics.DifferentialAlgebraicSystem(
        _positive_residual,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        state_scale=jnp.asarray([2.0]),
        residual_scale=jnp.asarray([0.5]),
        trial_validity=lambda time, state, rate, args, inputs: jnp.all(state > 0.0),
        trial_validity_id="positive-dae-domain-v1",
        system_id="positive-dae",
        **hooks,
    )


@pytest.mark.parametrize(
    "initialization",
    (
        phx.solver.DAEInitializationSpec.index_one(),
        phx.solver.DAEInitializationSpec.check_only(),
    ),
)
def test_initialization_rejects_domain_before_physical_evaluation(initialization):
    problem = phx.solver.DifferentialAlgebraicProblem(
        _system(),
        jnp.asarray([-1.0]),
        initial_state_rate=jnp.zeros(1),
        initialization=initialization,
        args=jnp.asarray([0.0, 0.35]),
    )
    result = phx.solver.initialize_dae(problem, 0.0)
    assert not result.valid
    assert result.status == int(phx.solver.DAEInitializationStatus.DOMAIN_FAILURE)
    assert result.domain_failures == 1
    assert jnp.array_equal(result.state, jnp.asarray([-1.0]))


def test_invalid_stage_predictor_is_rejected_without_clipping_or_residual_call():
    problem = phx.solver.DifferentialAlgebraicProblem(
        _system(),
        jnp.ones(1),
        initial_state_rate=jnp.asarray([-4.0]),
        initialization=phx.solver.DAEInitializationSpec.check_only(),
        args=jnp.asarray([-4.0, 0.35]),
    )
    grid = phx.dynamics.TimeGrid(jnp.asarray([0.0, 1.0]), time_id="invalid-predictor")
    result = phx.solver.solve_dae(problem, grid)
    assert not result.successful
    assert result.initialization.valid
    assert result.attempt_history.domain_failures[0] == 1
    assert result.attempt_history.residual_evaluations[0] == 0


def _event_plan(*, invalid_reset=False):
    guard = phx.solver.HybridGuardPlan(
        lambda time, state, args: time - args[1],
        direction=1,
        terminal=True,
        guard_id="time-inside-domain",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (phx.solver.ScheduledHybridGuard(guard),),
        maximum_events=1,
    )
    reset = phx.solver.DAEResetMap(
        lambda time, state, rate, args: (-state if invalid_reset else state, rate),
        phx.solver.DAEInitializationSpec.index_one(),
        reset_id=f"domain-reset-{invalid_reset}",
    )
    return phx.solver.DAEEventPlan(
        schedule,
        (reset,),
        phx.solver.DAEConsistencyPolicy(1e-8, 1e-8, 1e-8),
    )


def test_event_reset_consistency_rejects_invalid_state_without_evaluating_it():
    problem = phx.solver.DifferentialAlgebraicProblem(
        _system(),
        jnp.ones(1),
        initial_state_rate=jnp.ones(1),
        args=jnp.asarray([1.0, 0.35]),
    )
    grid = phx.dynamics.TimeGrid(jnp.asarray([0.0, 0.5]), time_id="invalid-event-reset")
    result = phx.solver.solve_dae(
        problem, grid, event_plan=_event_plan(invalid_reset=True)
    )
    assert not result.successful
    assert result.events.event_count == 1
    assert not result.events.valid[0]
    assert result.events.domain_failures > 0


def _root_diagonal_setup(unknown, arguments, source, target):
    if isinstance(arguments, _DAEInitializationArguments):
        diagonal = jnp.full_like(unknown, 2.0)
    elif isinstance(arguments, _DAEEventRootArguments):
        time = unknown[-1]
        diagonal = jnp.asarray(
            [
                2.0
                * (
                    1.0 / (time - arguments.history_times[0])
                    + 1.0 / arguments.physical_state(unknown)[0]
                ),
                1.0 / arguments.guard_scale,
            ]
        )
    else:
        diagonal = 2.0 * (arguments.shift + 1.0 / arguments.physical_state(unknown))
    return phx.linalg.DenseLinearOperator(
        jnp.diag(diagonal),
        source=source,
        target=target,
    )


def test_native_setup_actions_cover_initialization_stage_event_and_replay():
    system = _system(
        initialization_linear_setup=_root_diagonal_setup,
        stage_linear_setup=_root_diagonal_setup,
        event_linear_setup=_root_diagonal_setup,
        tangent_linear_setup=_root_diagonal_setup,
        adjoint_linear_setup=_root_diagonal_setup,
    )
    linear = phx.linalg.LinearSolvePolicy(
        phx.linalg.FGMRES(restart=2),
        tolerance=phx.linalg.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=8),
        preconditioning=phx.linalg.PreconditioningPolicy(
            phx.linalg.JacobiPreconditionerBuilder()
        ),
        materialization=phx.linalg.MaterializationPolicy(max_entries=1, max_bytes=8),
    )
    newton = phx.nonlinear.NewtonKrylov(linear_policy=linear)
    termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=0.0,
        maximum_steps=24,
    )
    policy = phx.solver.DAESolvePolicy(
        nonlinear_method=newton,
        initialization_method=newton,
        nonlinear_termination=termination,
        initialization_termination=termination,
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.ones(1),
        initial_state_rate=jnp.ones(1),
        args=jnp.asarray([1.0, 0.35]),
    )
    grid = phx.dynamics.TimeGrid(jnp.asarray([0.0, 0.5]), time_id="native-domain-hooks")
    prepared = phx.solver.prepare_dae(
        problem, grid, policy=policy, event_plan=_event_plan()
    )
    result = phx.solver.solve_dae(prepared)
    assert result.successful
    assert result.events.derivative_valid[0]
    assert jnp.allclose(result.events.event_times[0], 0.35, atol=1e-8)

    def event_time(threshold):
        return phx.solver.solve_dae(
            prepared, args=jnp.asarray([1.0, threshold])
        ).events.event_times[0]

    _, tangent = jax.jvp(event_time, (jnp.asarray(0.35),), (jnp.asarray(1.0),))
    gradient = jax.grad(event_time)(jnp.asarray(0.35))
    assert jnp.allclose(tangent, 1.0, atol=1e-7)
    assert jnp.allclose(gradient, tangent, atol=1e-7)


def test_bordered_event_trials_outside_bracket_are_rejected_not_clipped():
    from phydrax.solver._dae_events import _DAEEventRootArguments, _DAEEventRootResidual

    system = _system()
    guard = lambda time, state, args: time - args[1]
    arguments = _DAEEventRootArguments(
        jnp.full((5, 1), 0.5),
        jnp.ones((5, 1)),
        jnp.full((5,), 0.5),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray([1.0, 0.25]),
        jnp.asarray(True),
        jnp.zeros(2),
        jnp.asarray(0.5),
        jnp.asarray(1.0),
        guard,
        jnp.asarray(1.0),
    )
    residual = _DAEEventRootResidual(system, None, guard, 1.0, (1,), 1)
    space = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    root = phx.nonlinear.NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        **system.root_options("event", residual, space, space),
    )
    result = phx.nonlinear.NewtonKrylov().solve(
        root,
        jnp.asarray([0.5, 1.0]),
        args=arguments,
        termination=phx.nonlinear.NonlinearTermination(maximum_steps=12),
    )
    assert not result.successful
    assert result.diagnostics.domain_failures > 0
    assert 0.5 < result.state[-1] <= 1.0


def test_mapped_native_initialization_keeps_invalid_lane_outside_physics():
    problem = phx.solver.DifferentialAlgebraicProblem(
        _system(),
        jnp.ones(1),
        initial_state_rate=jnp.zeros(1),
        args=jnp.asarray([0.0, 0.35]),
    )
    result = jax.jit(
        jax.vmap(
            lambda state: phx.solver.initialize_dae(problem, 0.0, initial_state=state)
        )
    )(jnp.asarray([[-1.0], [4.0]]))
    assert jnp.array_equal(result.valid, jnp.asarray([False, True]))
    assert result.domain_failures[0] == 1
    assert jnp.array_equal(result.state[:, 0], jnp.asarray([-1.0, 4.0]))
    assert jnp.allclose(result.state_rate[1, 0], -jnp.log(4.0), atol=1e-8)
