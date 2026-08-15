import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _deterministic_trajectory():
    times = jnp.linspace(0.0, 1.0, 5)
    states = jnp.broadcast_to(times[None, :, None], (3, 5, 1))
    realization = phx.stochastic.WienerRealization(
        jr.key(0), (1,), support=(0.0, 1.0), sample_shape=(3,)
    )
    return phx.stochastic.StochasticTrajectory(
        times,
        states,
        realization_axes=("process",),
        realization_shape=(3,),
        state_axes=("state",),
        realizations=(realization,),
    )


def test_martingale_increments_match_exact_compensator_and_stop():
    trajectory = _deterministic_trajectory()
    problem = phx.stochastic.MartingaleProblem(
        lambda state: state,
        lambda state, time: jnp.ones_like(state),
        observable_shape=(1,),
        bracket_density=lambda state, time: jnp.asarray([[0.0]]),
    )

    residuals = phx.stochastic.martingale_increments(trajectory, problem)
    stopping = phx.stochastic.first_stopping_indices(
        trajectory, lambda state, time: time >= 0.5
    )
    stopped = phx.stochastic.stopped_martingale_increments(residuals, stopping)

    assert jnp.allclose(residuals.increments, 0.0)
    assert jnp.array_equal(stopping.indices, jnp.full((3,), 2))
    assert jnp.all(stopping.hit)
    assert jnp.array_equal(stopped.interval_valid[:, :2], jnp.ones((3, 2), bool))
    assert not jnp.any(stopped.interval_valid[:, 2:])
    assert jnp.allclose(phx.stochastic.predictable_bracket_increments(residuals), 0.0)


def test_quadratic_covariation_and_moment_loss_preserve_event_shape():
    trajectory = _deterministic_trajectory()
    problem = phx.stochastic.MartingaleProblem(
        lambda state: jnp.concatenate((state, state**2)),
        lambda state, time: jnp.concatenate((jnp.ones_like(state), 2.0 * state)),
        observable_shape=(2,),
    )
    residuals = phx.stochastic.martingale_increments(
        trajectory, problem, quadrature="trapezoid"
    )

    covariation = phx.stochastic.quadratic_covariation(residuals)
    loss = phx.stochastic.martingale_moment_loss(
        residuals, (lambda state, time: 1.0,), reduction="none"
    )

    assert covariation.shape == (3, 5, 2, 2)
    assert loss.shape == (1, 2)
    assert jnp.allclose(loss, 0.0, atol=1e-12)


def test_combined_generator_and_carre_du_champ():
    generator = phx.stochastic.combined_generator_observable(
        lambda state, time: state,
        lambda state, time: 2.0 * state,
    )
    assert jnp.allclose(
        generator(jnp.asarray([2.0]), jnp.asarray(0.0)), jnp.asarray([6.0])
    )

    def brownian_generator(observable, state, time):
        return 0.5 * jax_hessian_scalar(observable, state)

    gamma = phx.stochastic.carre_du_champ(
        brownian_generator, lambda state: state[0], jnp.asarray([2.0]), 0.0
    )
    assert jnp.allclose(gamma, 1.0)


def jax_hessian_scalar(function, state):
    import jax

    return jax.hessian(lambda value: function(value))(state)[0, 0]


def test_martingale_formulation_checks_spde_solution_concept():
    trajectory = _deterministic_trajectory()
    trajectory = phx.stochastic.StochasticTrajectory(
        trajectory.times,
        trajectory.states,
        realization_axes=trajectory.realization_axes,
        realization_shape=trajectory.realization_shape,
        state_axes=trajectory.state_axes,
        realizations=trajectory.realizations,
        metadata={"spde_solution_spec": phx.stochastic.SPDESolutionSpec("mild")},
    )
    problem = phx.stochastic.MartingaleProblem(
        lambda state: state,
        lambda state, time: state,
        observable_shape=(1,),
    )

    with pytest.raises(ValueError, match="do not support"):
        phx.stochastic.martingale_increments(trajectory, problem)
