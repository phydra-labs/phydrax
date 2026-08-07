#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _constant_history_terminal(parameters, adjoint):
    scale, rate = parameters
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: args * memory["lag"],
        lambda time, args: scale * jnp.ones((1,)),
        (phx.solver.ConstantDelay("lag", 0.8),),
        t0=0.0,
        t1=0.5,
        args=rate,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.1, 0.3, 0.5]),
        adjoint=adjoint,
        max_steps=128,
    )
    return jnp.dot(solution.states[:, 0], jnp.asarray([0.2, -0.5, 1.3]))


def test_checkpointed_delay_adjoint_matches_direct_discrete_gradient():
    parameters = jnp.asarray([1.2, 0.4])
    checkpointed = jax.grad(
        lambda value: _constant_history_terminal(
            value,
            phx.solver.CheckpointedDelayAdjoint(checkpoints=2),
        )
    )(parameters)
    direct = jax.grad(
        lambda value: _constant_history_terminal(value, dfx.DirectAdjoint())
    )(parameters)

    assert jnp.allclose(checkpointed, direct, rtol=1e-8, atol=1e-9)
    expected_scale = 0.2 * 1.04 - 0.5 * 1.12 + 1.3 * 1.2
    expected_rate = 1.2 * (0.2 * 0.1 - 0.5 * 0.3 + 1.3 * 0.5)
    assert jnp.allclose(
        checkpointed,
        jnp.asarray([expected_scale, expected_rate]),
        rtol=2e-7,
        atol=2e-8,
    )


def test_checkpoint_count_does_not_change_delay_gradient():
    parameters = jnp.asarray([0.9, -0.2])
    gradients = tuple(
        jax.grad(
            lambda value: _constant_history_terminal(
                value,
                phx.solver.CheckpointedDelayAdjoint(checkpoints=count),
            )
        )(parameters)
        for count in (1, 2, 4)
    )
    assert all(jnp.allclose(gradients[0], gradient) for gradient in gradients[1:])


def test_checkpointed_delay_adjoint_differentiates_trainable_delay():
    terminal_time = 0.25

    def terminal(lag):
        problem = phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: memory["lag"],
            lambda time, args: jnp.asarray([time]),
            (phx.solver.ConstantDelay("lag", lag),),
            t0=0.0,
            t1=terminal_time,
        )
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([terminal_time]),
            adjoint=phx.solver.CheckpointedDelayAdjoint(checkpoints=2),
            dt0=0.02,
            max_steps=128,
        ).states[0, 0]

    lag = jnp.asarray(0.5)
    assert jnp.isclose(terminal(lag), 0.5 * terminal_time**2 - lag * terminal_time)
    assert jnp.isclose(jax.grad(terminal)(lag), -terminal_time, atol=2e-9)


def test_checkpointed_delay_adjoint_is_jittable():
    operation = jax.jit(
        jax.value_and_grad(
            lambda parameters: _constant_history_terminal(
                parameters,
                phx.solver.CheckpointedDelayAdjoint(checkpoints=2),
            )
        )
    )
    value, gradient = operation(jnp.asarray([1.0, 0.3]))
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))


def test_checkpointed_delay_adjoint_supports_bounded_rolling_history():
    def terminal(parameters, history_mode):
        scale, rate = parameters
        problem = phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: args * memory["lag"],
            lambda time, args: scale * jnp.ones((1,)),
            (phx.solver.ConstantDelay("lag", 0.2),),
            t0=0.0,
            t1=0.5,
            args=rate,
        )
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.1, 0.3, 0.5]),
            solver=dfx.Euler(),
            adjoint=phx.solver.CheckpointedDelayAdjoint(checkpoints=2),
            dt0=0.05,
            history_mode=history_mode,
            history_capacity=8 if history_mode == "rolling" else None,
            max_steps=32,
        ).states[:, 0] @ jnp.asarray([0.2, -0.5, 1.3])

    parameters = jnp.asarray([1.2, 0.4])
    rolling = jax.grad(lambda value: terminal(value, "rolling"))(parameters)
    full = jax.grad(lambda value: terminal(value, "full"))(parameters)

    assert jnp.array_equal(rolling, full)


def test_segmented_delay_adjoint_matches_whole_discrete_gradient_and_jits():
    def terminal(parameters):
        scale, rate = parameters
        problem = phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: args * memory["lag"],
            lambda time, args: scale * jnp.ones((1,)),
            (phx.solver.ConstantDelay("lag", 0.2),),
            t0=0.0,
            t1=0.5,
            args=rate,
        )
        solution = phx.solver.solve_diffrax_delay_segmented(
            problem,
            save_times=jnp.asarray([0.1, 0.3, 0.5]),
            solver=dfx.Euler(),
            adjoint=phx.solver.SegmentedDelayAdjoint(4, checkpoints=2),
            dt0=0.05,
            max_steps_per_segment=3,
        )
        return solution.states[:, 0] @ jnp.asarray([0.2, -0.5, 1.3])

    parameters = jnp.asarray([1.2, 0.4])
    value, gradient = jax.jit(jax.value_and_grad(terminal))(parameters)

    assert jnp.isclose(value, 1.45872)
    assert jnp.allclose(gradient, jnp.asarray([1.2156, 0.6696]))


@pytest.mark.parametrize("max_segments", [0, -1, True, 1.5])
def test_segmented_delay_adjoint_rejects_invalid_segment_bound(max_segments):
    with pytest.raises(ValueError, match="positive integer"):
        phx.solver.SegmentedDelayAdjoint(max_segments)


@pytest.mark.parametrize("checkpoints", [0, -1, True, 1.5])
def test_checkpointed_delay_adjoint_rejects_invalid_checkpoint_count(checkpoints):
    with pytest.raises(ValueError, match="positive integer"):
        phx.solver.CheckpointedDelayAdjoint(checkpoints)
