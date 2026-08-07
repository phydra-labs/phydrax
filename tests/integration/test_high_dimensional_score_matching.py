import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
from phydrax.objectives._score_matching import (
    ScoreMatchingObjective,
    ScoreMatchingPolicy,
)
from phydrax.stochastic._state_time import trajectory_state_time_samples


class _LinearTimeScore(eqx.Module):
    coefficient: jnp.ndarray

    def __call__(self, state, time):
        del time
        return self.coefficient * state


def test_dimension_100_ornstein_uhlenbeck_score_improves_over_zero_field():
    dimension = 100
    time = 1.0
    variance = 1.0 - jnp.exp(-2.0 * time)
    train_states = jnp.sqrt(variance) * jr.normal(
        jr.key(41),
        (512, 1, dimension),
    )
    trajectory = phx.stochastic.StochasticTrajectory(
        jnp.asarray([time]),
        train_states,
        realization_axes=("path",),
        realization_shape=(512,),
        time_axis="saved_time",
        state_axes=("state",),
    )
    samples = trajectory_state_time_samples(trajectory, time_label="t")
    space = phx.domain.HyperRectangle(
        jnp.full((dimension,), -10.0),
        jnp.full((dimension,), 10.0),
        label="x",
    )
    domain = space @ phx.domain.TimeInterval(0.0, 1.0)
    score = domain.Function("x", "t")(_LinearTimeScore(jnp.asarray(0.0)))
    objective = ScoreMatchingObjective(
        "score",
        samples,
        policy=ScoreMatchingPolicy("implicit", num_probes=2),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"score": score},
        constraints=(),
        objectives=(objective,),
    )
    trained = solver.solve(
        num_iter=100,
        optim=optax.adam(0.05),
        jit=True,
        keep_best=False,
        log_every=0,
    )

    heldout = jnp.sqrt(variance) * jr.normal(jr.key(42), (256, dimension))
    exact = -heldout / variance
    zero_error = jnp.sqrt(jnp.mean(exact**2))
    predicted = jnp.asarray(
        [
            trained.functions["score"].func(state, jnp.asarray(time))
            for state in heldout
        ]
    )
    trained_error = jnp.sqrt(jnp.mean((predicted - exact) ** 2))

    assert trained_error < 0.15 * zero_error
