#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _history(time, scale):
    return scale * jnp.stack((jnp.cos(time), jnp.sin(time)))


def _drift(time, state, memory, args):
    del time
    return args * state + memory["short"] - memory["long"]


def test_delay_problem_declares_named_shape_stable_memory():
    problem = phx.solver.DelayDifferentialProblem(
        _drift,
        _history,
        (
            phx.solver.ConstantDelay("short", 0.2),
            phx.solver.ConstantDelay("long", 0.7),
        ),
        t0=0.0,
        t1=2.0,
        args=1.5,
        problem_id="named-vector-delay",
    )

    assert problem.delay_names == ("short", "long")
    assert problem.state_shape == (2,)
    assert problem.num_delays == 2
    assert not problem.stochastic
    assert not problem.neutral
    assert jnp.isclose(problem.minimum_delay, 0.2)
    assert jnp.isclose(problem.maximum_delay, 0.7)
    assert problem.problem_id == "named-vector-delay"


def test_delay_values_support_static_name_and_index_access_under_jit():
    values = phx.solver.DelayValues(
        ("matrix", "other"),
        (jnp.arange(4.0).reshape(2, 2), jnp.ones((2, 2))),
    )

    operation = eqx.filter_jit(
        lambda memory: memory["matrix"] + 2.0 * memory[1] + memory[-2]
    )
    assert jnp.array_equal(
        operation(values),
        2.0 * jnp.arange(4.0).reshape(2, 2) + 2.0,
    )
    assert values.stacked.shape == (2, 2, 2)
    with pytest.raises(KeyError, match="missing"):
        _ = values["missing"]


def test_delay_problem_is_a_transformable_pytree():
    def terminal(scale):
        problem = phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: args * memory[0],
            lambda time, args: args * jnp.ones((2, 3)),
            (phx.solver.ConstantDelay("lag", 0.4),),
            t0=0.0,
            t1=1.0,
            args=scale,
        )
        return jnp.sum(problem.initial_state) + problem.minimum_delay

    observed = jax.jit(jax.vmap(terminal))(jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(observed, jnp.asarray([6.4, 12.4]))
    assert jnp.isclose(jax.grad(terminal)(2.0), 6.0)


def test_delay_problem_validates_term_names_and_history_shapes():
    history = lambda time, args: jnp.ones((2,))
    drift = lambda time, state, memory, args: state

    with pytest.raises(ValueError, match="unique"):
        phx.solver.DelayDifferentialProblem(
            drift,
            history,
            (
                phx.solver.ConstantDelay("same", 0.2),
                phx.solver.StateDependentDelay(
                    "same",
                    lambda time, state, args: 0.3,
                    minimum_delay=0.1,
                ),
            ),
            t0=0.0,
            t1=1.0,
        )

    with pytest.raises(ValueError, match="state shape"):
        phx.solver.DelayDifferentialProblem(
            drift,
            history,
            (
                phx.solver.DistributedDelay(
                    "distributed",
                    lambda time, lag, state, args: jnp.ones((3,)),
                    (0.1, 0.5),
                ),
            ),
            t0=0.0,
            t1=1.0,
        )

    with pytest.raises(ValueError, match="drift"):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: jnp.ones((3,)),
            history,
            (phx.solver.ConstantDelay("lag", 0.2),),
            t0=0.0,
            t1=1.0,
        )


def test_delay_problem_requires_neutral_prehistory_derivative():
    delayed_derivative = phx.solver.DerivativeDelay(
        "velocity",
        phx.solver.ConstantDelay("source", 0.25),
    )
    with pytest.raises(ValueError, match="history_derivative"):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: memory["velocity"],
            lambda time, args: jnp.ones((1,)),
            (delayed_derivative,),
            t0=0.0,
            t1=1.0,
        )

    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["velocity"],
        lambda time, args: jnp.exp(time).reshape((1,)),
        (delayed_derivative,),
        history_derivative=lambda time, args: jnp.exp(time).reshape((1,)),
        t0=0.0,
        t1=1.0,
    )
    assert problem.neutral


def test_delay_problem_validates_stochastic_coefficient_shape_and_identity():
    wiener_term = phx.solver.DelayWienerTerm(
        "forcing",
        lambda time, state, memory, args: jnp.ones(state.shape + (2,)),
        (2,),
        structure="commutative",
        basis_id="shared-basis",
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: jnp.zeros_like(state),
        lambda time, args: jnp.ones((3,)),
        (phx.solver.ConstantDelay("lag", 0.5),),
        wiener_terms=(wiener_term,),
        t0=0.0,
        t1=1.0,
    )
    assert problem.stochastic
    assert problem.noise_shape == (2,)
    assert problem.noise_id == "shared-basis"
    assert problem.wiener_term_slices["forcing"] == (0, 2)

    with pytest.raises(ValueError, match="coefficient"):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: jnp.zeros_like(state),
            lambda time, args: jnp.ones((3,)),
            (phx.solver.ConstantDelay("lag", 0.5),),
            wiener_terms=(
                phx.solver.DelayWienerTerm(
                    "bad",
                    lambda time, state, memory, args: jnp.ones((3,)),
                    (2,),
                ),
            ),
            t0=0.0,
            t1=1.0,
        )


@pytest.mark.parametrize("delay", [0.0, -0.1, jnp.inf, jnp.nan])
def test_constant_delay_rejects_nonpositive_or_nonfinite_values(delay):
    with pytest.raises(Exception, match="finite and positive"):
        phx.solver.ConstantDelay("invalid", delay)


@pytest.mark.parametrize(
    ("t0", "t1", "message"),
    [
        (0.0, 0.0, "t1 > t0"),
        (1.0, 0.0, "t1 > t0"),
        (jnp.nan, 1.0, "finite"),
        (0.0, jnp.inf, "finite"),
    ],
)
def test_delay_problem_rejects_invalid_time_intervals(t0, t1, message):
    with pytest.raises(Exception, match=message):
        phx.solver.DelayDifferentialProblem(
            lambda time, state, memory, args: state,
            lambda time, args: jnp.ones((1,)),
            (phx.solver.ConstantDelay("lag", 0.2),),
            t0=t0,
            t1=t1,
        )
