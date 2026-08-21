#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


nl = phx.nonlinear


def _transition(parameter, previous, driver):
    return jnp.tanh(parameter * previous + driver)


def _serial(parameter, initial, drivers):
    def step(state, driver):
        next_state = _transition(parameter, state, driver)
        return next_state, next_state

    return jax.lax.scan(step, initial, drivers)[1]


def _termination(steps=24):
    return nl.NonlinearTermination(
        absolute_residual=1e-11,
        relative_residual=1e-11,
        maximum_steps=steps,
    )


@pytest.mark.parametrize(
    "method",
    (
        nl.CausalNewton(),
        nl.CausalNewton(linearization=nl.CausalLinearizationPolicy("diagonal-exact")),
        nl.CausalLevenbergMarquardt(),
    ),
)
def test_causal_solver_matches_serial_recurrence_and_jit(method):
    drivers = jnp.linspace(-0.2, 0.3, 16, dtype=jnp.float64)
    parameter = jnp.asarray(0.7, dtype=jnp.float64)
    initial = jnp.asarray(0.1, dtype=jnp.float64)
    problem = nl.CausalRecurrenceProblem(
        _transition,
        initial,
        drivers,
        parameters=parameter,
        problem_id="tanh-recurrence",
    )

    result = jax.jit(
        lambda current: nl.solve_causal_recurrence(
            current,
            method=method,
            termination=_termination(),
        )
    )(problem)

    assert bool(result.successful)
    assert jnp.allclose(result.states, _serial(parameter, initial, drivers), atol=1e-10)
    assert float(jnp.max(jnp.abs(result.flat_residuals))) < 1e-10
    assert int(result.diagnostics.iteration_count) <= problem.num_steps


def test_hutchinson_quasi_solver_replays_fixed_probes():
    drivers = jnp.stack(
        (
            jnp.linspace(-0.1, 0.2, 12),
            jnp.linspace(0.2, -0.1, 12),
        ),
        axis=-1,
    )

    def transition(matrix, previous, driver):
        return jnp.tanh(matrix @ previous + driver)

    problem = nl.CausalRecurrenceProblem(
        transition,
        jnp.zeros((2,)),
        drivers,
        parameters=jnp.asarray([[0.4, 0.1], [-0.15, 0.3]]),
    )
    method = nl.CausalLevenbergMarquardt(
        linearization=nl.CausalLinearizationPolicy(
            "diagonal-hutchinson",
            probe_count=4,
        )
    )

    first = nl.solve_causal_recurrence(
        problem,
        method=method,
        termination=_termination(steps=30),
        probe_key=jax.random.key(7),
    )
    second = nl.solve_causal_recurrence(
        problem,
        method=method,
        termination=_termination(steps=30),
        probe_key=jax.random.key(7),
    )

    assert bool(first.successful)
    assert jnp.array_equal(first.flat_states, second.flat_states)
    assert jnp.array_equal(
        first.diagnostics.residual_norm,
        second.diagnostics.residual_norm,
        equal_nan=True,
    )


def test_causal_implicit_derivative_matches_serial_reverse_mode():
    drivers = jnp.linspace(-0.2, 0.3, 16, dtype=jnp.float64)

    def causal_objective(parameter, initial, forcing):
        problem = nl.CausalRecurrenceProblem(
            _transition,
            initial,
            forcing,
            parameters=parameter,
        )
        result = nl.solve_causal_recurrence(
            problem,
            method=nl.CausalNewton(
                linearization=nl.CausalLinearizationPolicy("diagonal-exact")
            ),
            termination=_termination(),
        )
        return jnp.sum(jnp.square(result.states))

    def serial_objective(parameter, initial, forcing):
        return jnp.sum(jnp.square(_serial(parameter, initial, forcing)))

    arguments = (
        jnp.asarray(0.7, dtype=jnp.float64),
        jnp.asarray(0.1, dtype=jnp.float64),
        drivers,
    )
    causal_gradient = jax.jit(jax.grad(causal_objective, argnums=(0, 1, 2)))(*arguments)
    serial_gradient = jax.grad(serial_objective, argnums=(0, 1, 2))(*arguments)

    assert jnp.allclose(causal_gradient[0], serial_gradient[0], atol=1e-10)
    assert jnp.allclose(causal_gradient[1], serial_gradient[1], atol=1e-10)
    assert jnp.allclose(causal_gradient[2], serial_gradient[2], atol=1e-10)


def test_causal_solver_supports_pytree_states_and_fixed_block_linearization():
    initial = {
        "position": jnp.asarray([0.0, 0.1]),
        "memory": jnp.asarray(0.2),
    }
    drivers = {
        "forcing": jnp.linspace(-0.1, 0.2, 9),
    }

    def transition(parameters, previous, driver):
        forcing = driver["forcing"]
        position = jnp.tanh(
            parameters["matrix"] @ previous["position"] + forcing + previous["memory"]
        )
        memory = 0.25 * previous["memory"] + 0.1 * jnp.sum(position)
        return {"position": position, "memory": memory}

    problem = nl.CausalRecurrenceProblem(
        transition,
        initial,
        drivers,
        parameters={"matrix": jnp.asarray([[0.3, 0.1], [-0.1, 0.25]])},
    )

    def block_builder(parameters, previous, driver):
        del parameters, previous, driver
        return jnp.zeros((3, 3))

    result = nl.solve_causal_recurrence(
        problem,
        method=nl.CausalNewton(
            linearization=nl.CausalLinearizationPolicy(
                "fixed-block",
                block_builder=block_builder,
                linearization_id="zero-block",
            )
        ),
        termination=_termination(steps=problem.num_steps + 2),
    )

    assert bool(result.successful)
    assert result.states["position"].shape == (problem.num_steps, 2)
    assert result.states["memory"].shape == (problem.num_steps,)


def test_nonconverged_causal_result_is_observable_and_not_differentiable():
    drivers = jnp.linspace(-0.2, 0.3, 8)

    def objective(parameter):
        result = nl.solve_causal_recurrence(
            nl.CausalRecurrenceProblem(
                _transition,
                jnp.asarray(0.1),
                drivers,
                parameters=parameter,
            ),
            method=nl.CausalNewton(),
            termination=nl.NonlinearTermination(
                absolute_residual=0.0,
                relative_residual=0.0,
                maximum_steps=1,
            ),
        )
        return jnp.sum(result.states)

    result = nl.solve_causal_recurrence(
        nl.CausalRecurrenceProblem(
            _transition,
            jnp.asarray(0.1),
            drivers,
            parameters=jnp.asarray(0.7),
        ),
        method=nl.CausalNewton(),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            maximum_steps=1,
        ),
    )
    assert not bool(result.successful)
    assert int(result.status) == int(nl.NonlinearStatus.MAXIMUM_STEPS_REACHED)
    with pytest.raises(Exception, match="successfully converged"):
        jax.grad(objective)(jnp.asarray(0.7))
