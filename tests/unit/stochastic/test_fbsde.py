from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_coupled_fbsde_explicit_replays_one_wiener_realization():
    problem = phx.solver.CoupledFBSDEProblem(
        jnp.linspace(0.0, 1.0, 9),
        jnp.asarray([0.0]),
        lambda time, state, value, control, args: 0.1 * value,
        lambda time, state, value, control, args: jnp.ones((1, 1)),
        lambda time, state, value, control, args: jnp.zeros((1,)),
        lambda state, args: state,
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        num_paths=32,
        problem_id="coupled",
        process_id="wiener",
        wiener_tolerance=1e-3,
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(50),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(32,),
        tolerance=1e-3,
        noise_id="wiener",
        label="coupled",
    )
    value = lambda time, state: state
    control = lambda time, state: jnp.ones((1, 1))

    first = phx.solver.solve_coupled_fbsde_explicit(
        jr.key(51), problem, value, control, realization=realization
    )
    replay = phx.solver.solve_coupled_fbsde_explicit(
        jr.key(52), problem, value, control, realization=realization
    )

    assert first.paths.realization is not None
    assert first.paths.states.shape == (32, 9, 1)
    assert jnp.array_equal(first.paths.states, replay.paths.states)
    assert first.paths.realization.realization_id == realization.realization_id
    assert jnp.all(first.successful)


def test_coupled_fbsde_rejects_scalar_state_shape():
    with pytest.raises(ValueError, match="state_shape"):
        phx.solver.CoupledFBSDEProblem(
            jnp.asarray([0.0, 1.0]),
            jnp.asarray(0.0),
            lambda time, state, value, control, args: state,
            lambda time, state, value, control, args: jnp.asarray([1.0]),
            lambda time, state, value, control, args: value,
            lambda state, args: state,
            state_shape=(),
            noise_shape=(1,),
            output_shape=(1,),
            num_paths=1,
            problem_id="scalar-state",
            process_id="wiener",
        )


def _jump_problem(*, status=None, realization=None):
    sample_shape = (2,)
    times = jnp.asarray([0.0, 0.5, 1.0])
    states = jnp.asarray(
        [
            [[0.0], [1.0], [1.0]],
            [[0.0], [0.0], [1.0]],
        ]
    )
    event_status = (
        jnp.zeros(sample_shape, dtype=jnp.int32)
        if status is None
        else jnp.asarray(status, dtype=jnp.int32)
    )
    events = phx.stochastic.JumpEventBatch(
        jnp.asarray([[0.25], [0.75]]),
        jnp.zeros((2, 1), dtype=jnp.int32),
        jnp.zeros((2, 1)),
        jnp.ones((2, 1), dtype="bool"),
        event_status,
        state_shape=(1,),
        pre_states=jnp.zeros((2, 1, 1)),
        post_states=jnp.ones((2, 1, 1)),
    )
    paths = phx.stochastic.BSDEPathBatch(
        times,
        states,
        jnp.zeros((2, 2, 1)),
        sample_shape=sample_shape,
        state_shape=(1,),
        noise_shape=(1,),
        path_id="poisson-paths",
        process_id="wiener",
        jump_events={"jump": events},
        realization=realization,
    )
    base = phx.stochastic.BSDEProblem(
        lambda key: paths,
        lambda time, state, args: jnp.zeros((1,)),
        lambda time, state, args: jnp.zeros((1, 1)),
        lambda time, state, value, control, args: jnp.zeros((1,)),
        lambda state, args: state,
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=(1,),
        problem_id="poisson-bsde",
        process_id="wiener",
    )
    problem = phx.stochastic.JumpBSDEProblem(
        base,
        lambda label, time, state, jump_control, args: jnp.ones((1,)),
        {"jump": "poisson"},
    )
    return problem, paths


def _jump_control(label, time, state, channel, mark, args, *, key):
    del label, time, state, channel, mark, args, key
    return jnp.ones((1,))


def test_compensated_poisson_bsde_is_exact_on_grid():
    problem, paths = _jump_problem()
    evaluation = phx.stochastic.evaluate_jump_bsde(
        problem,
        paths,
        lambda time, state: jnp.asarray([state[0] + 1.0 - time]),
        _jump_control,
        control_predictor=lambda time, state: jnp.zeros((1, 1)),
    )

    assert jnp.allclose(evaluation.compensated_jump_increments.sum(axis=-2), 0.0)
    assert jnp.allclose(evaluation.local_residuals, 0.0)
    assert jnp.allclose(evaluation.global_residual, 0.0)
    assert jnp.allclose(phx.stochastic.jump_bsde_objective_loss(evaluation), 0.0)
    diagnostics = phx.stochastic.jump_bsde_diagnostics(evaluation)
    assert diagnostics.passed
    assert diagnostics.num_valid == 2


@pytest.mark.parametrize("output_shape", [(), (2, 2)])
def test_jump_bsde_reduces_the_declared_step_axis(output_shape):
    _, paths = _jump_problem()

    def output(value):
        return jnp.asarray(value) if not output_shape else jnp.full(output_shape, value)

    base = phx.stochastic.BSDEProblem(
        lambda _key: paths,
        lambda _time, _state, _args: jnp.zeros((1,)),
        lambda _time, _state, _args: jnp.zeros((1, 1)),
        lambda _time, _state, value, _control, _args: jnp.zeros_like(value),
        lambda state, _args: output(state[0]),
        state_shape=(1,),
        noise_shape=(1,),
        output_shape=output_shape,
        problem_id=f"jump-axis-{len(output_shape)}",
        process_id="wiener",
    )
    problem = phx.stochastic.JumpBSDEProblem(
        base,
        lambda _label, _time, _state, _control, _args: output(1.0),
        {"jump": "poisson"},
    )
    evaluation = phx.stochastic.evaluate_jump_bsde(
        problem,
        paths,
        lambda time, state: output(state[0] + 1.0 - time),
        lambda _label, _time, _state, _channel, _mark, _args, *, key: output(
            jr.uniform(key)
        ),
        control_predictor=lambda _time, _state: jnp.zeros(output_shape + (1,)),
        key=jr.key(44),
    )
    diagnostics = phx.stochastic.jump_bsde_diagnostics(evaluation)

    assert evaluation.global_residual.shape == (2,) + output_shape
    assert diagnostics.compensated_increment_mean.shape == output_shape


def test_jump_bsde_random_streams_are_namespaced_by_label():
    template_problem, template_paths = _jump_problem()
    events = template_paths.jump_events["jump"]

    def evaluate(label):
        paths = phx.stochastic.BSDEPathBatch(
            template_paths.times,
            template_paths.states,
            template_paths.wiener_increments,
            sample_shape=template_paths.sample_shape,
            state_shape=template_paths.state_shape,
            noise_shape=template_paths.noise_shape,
            path_id=f"label-{label}",
            process_id=template_paths.process_id,
            jump_events={label: events},
        )
        problem = phx.stochastic.JumpBSDEProblem(
            template_problem.base,
            template_problem.compensator_rate,
            {label: "poisson"},
        )
        return phx.stochastic.evaluate_jump_bsde(
            problem,
            paths,
            lambda time, state: jnp.asarray([state[0] + 1.0 - time]),
            lambda _label, _time, _state, _channel, _mark, _args, *, key: jr.uniform(
                key,
                (1,),
            ),
            control_predictor=lambda _time, _state: jnp.zeros((1, 1)),
            key=jr.key(45),
        )

    first = evaluate("first")
    second = evaluate("second")
    assert not jnp.array_equal(first.jump_sums, second.jump_sums)


def test_jump_bsde_propagates_event_failure_status():
    problem, paths = _jump_problem(
        status=jnp.asarray([phx.stochastic.JUMP_SUCCESS, phx.stochastic.JUMP_MAX_EVENTS])
    )
    kwargs: dict[str, Any] = dict(
        control_predictor=lambda time, state: jnp.zeros((1, 1)),
    )
    evaluation = phx.stochastic.evaluate_jump_bsde(
        problem,
        paths,
        lambda time, state: jnp.asarray([state[0] + 1.0 - time]),
        _jump_control,
        **kwargs,
    )

    assert jnp.array_equal(evaluation.valid_paths, jnp.asarray([True, False]))
    assert not phx.stochastic.jump_bsde_diagnostics(evaluation).passed
    with pytest.raises(RuntimeError, match="failed event paths"):
        phx.stochastic.evaluate_jump_bsde(
            problem,
            paths,
            lambda time, state: jnp.asarray([state[0] + 1.0 - time]),
            _jump_control,
            raise_on_failure=True,
            **kwargs,
        )


def test_jump_bsde_requires_composite_provenance_when_present():
    wiener = phx.stochastic.WienerRealization(
        jr.key(53),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(2,),
        tolerance=1e-3,
        noise_id="wiener",
        label="wiener",
    )
    problem, paths = _jump_problem(realization=wiener)

    with pytest.raises(ValueError, match="CompositeStochasticRealization"):
        phx.stochastic.evaluate_jump_bsde(
            problem,
            paths,
            lambda time, state: jnp.asarray([state[0] + 1.0 - time]),
            _jump_control,
            control_predictor=lambda time, state: jnp.zeros((1, 1)),
        )


def test_jump_bsde_accepts_matching_composite_realization():
    wiener = phx.stochastic.WienerRealization(
        jr.key(54),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(2,),
        tolerance=1e-3,
        noise_id="wiener",
        label="wiener",
    )
    poisson = phx.stochastic.PoissonClockRealization(
        jr.key(55),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=2,
        sample_shape=(2,),
        process_id="poisson",
        label="jump",
    )
    realization = phx.stochastic.CompositeStochasticRealization(
        {"wiener": wiener, "jump": poisson}
    )
    problem, paths = _jump_problem(realization=realization)
    evaluation = phx.stochastic.evaluate_jump_bsde(
        problem,
        paths,
        lambda time, state: jnp.asarray([state[0] + 1.0 - time]),
        _jump_control,
        control_predictor=lambda time, state: jnp.zeros((1, 1)),
    )

    assert jnp.all(evaluation.valid_paths)
