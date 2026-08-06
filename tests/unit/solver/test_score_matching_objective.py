import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx
from phydrax.objectives._score_matching import (
    ScoreMatchingObjective,
    ScoreMatchingPolicy,
)
from phydrax.stochastic._state_time import trajectory_state_time_samples


class _LinearScore(eqx.Module):
    coefficient: jnp.ndarray

    def __call__(self, state, time):
        del time
        return self.coefficient * state


class _MatrixScore(eqx.Module):
    matrix: jnp.ndarray

    def __call__(self, state, time):
        del time
        return self.matrix @ state


def _trajectory(*, dimension=3, paths=512, times=2, seed=0, valid=None):
    states = jr.normal(jr.key(seed), (paths, times, dimension))
    if valid is None:
        valid = jnp.ones((paths, times), dtype=bool)
    return phx.stochastic.StochasticTrajectory(
        jnp.linspace(0.0, 1.0, times),
        states,
        valid=valid,
        realization_axes=("path",),
        realization_shape=(paths,),
        time_axis="t_node",
        state_axes=("state",),
    )


def _score_function(model, dimension):
    space = phx.domain.HyperRectangle(
        jnp.full((dimension,), -10.0),
        jnp.full((dimension,), 10.0),
        label="x",
    )
    domain = space @ phx.domain.TimeInterval(0.0, 1.0)
    return phx.domain.DomainFunction(domain=domain, deps=("x", "t"), func=model)


def test_exact_and_implicit_score_matching_agree_for_diagonal_linear_score():
    dimension = 4
    trajectory = _trajectory(dimension=dimension, paths=128, times=3)
    samples = trajectory_state_time_samples(trajectory, time_label="t")
    score = _score_function(_LinearScore(jnp.asarray(-0.7)), dimension)
    exact = ScoreMatchingObjective(
        "score",
        samples,
        policy=ScoreMatchingPolicy("exact"),
    )
    implicit = ScoreMatchingObjective(
        "score",
        samples,
        policy=ScoreMatchingPolicy("implicit", num_probes=4),
    )
    shared_key = jr.key(3)

    exact_value = exact.loss({"score": score}, key=shared_key)
    implicit_value = implicit.loss({"score": score}, key=shared_key)
    empirical_norm = jnp.mean(jnp.sum(trajectory.states**2, axis=-1))
    expected = 0.5 * 0.7**2 * empirical_norm - 0.7 * dimension

    assert jnp.allclose(exact_value, expected)
    assert jnp.allclose(implicit_value, expected)


def test_sliced_score_matching_matches_implicit_objective_in_expectation():
    dimension = 5
    trajectory = _trajectory(dimension=dimension, paths=512, times=2, seed=2)
    samples = trajectory_state_time_samples(trajectory, time_label="t")
    score = _score_function(_LinearScore(jnp.asarray(-0.5)), dimension)
    implicit = ScoreMatchingObjective(
        "score",
        samples,
        policy=ScoreMatchingPolicy("implicit", num_probes=8),
    )
    sliced = ScoreMatchingObjective(
        "score",
        samples,
        policy=ScoreMatchingPolicy("sliced", num_probes=256),
    )

    implicit_value = implicit.loss({"score": score}, key=jr.key(8))
    sliced_value = sliced.loss({"score": score}, key=jr.key(8))

    assert jnp.allclose(sliced_value, implicit_value, atol=8e-2)


def test_masks_exclude_invalid_particle_states_and_time_coverage_is_reported():
    states = jnp.asarray(
        [
            [[1.0], [2.0], [1000.0]],
            [[3.0], [4.0], [5.0]],
        ]
    )
    valid = jnp.asarray([[True, True, False], [True, False, False]])
    trajectory = phx.stochastic.StochasticTrajectory(
        jnp.asarray([0.0, 0.5, 1.0]),
        states,
        valid=valid,
        realization_axes=("path",),
        realization_shape=(2,),
        time_axis="t_node",
        state_axes=("state",),
    )
    samples = trajectory_state_time_samples(trajectory, time_label="t")
    score = _score_function(_LinearScore(jnp.asarray(-1.0)), 1)
    objective = ScoreMatchingObjective(
        "score",
        samples,
        policy=ScoreMatchingPolicy("exact"),
    )

    diagnostics = objective.diagnostics({"score": score}, key=jr.key(0))
    expected = 0.5 * jnp.mean(jnp.asarray([1.0, 4.0, 9.0])) - 1.0

    assert jnp.allclose(diagnostics.objective, expected)
    assert jnp.allclose(diagnostics.valid_fraction, 0.5)
    assert jnp.array_equal(
        diagnostics.time_coverage,
        jnp.asarray([1.0, 0.5, 0.0]),
    )
    assert diagnostics.num_paths == 2
    assert diagnostics.num_times == 3


def test_probe_standard_error_decreases_for_off_diagonal_divergence():
    dimension = 12
    trajectory = _trajectory(dimension=dimension, paths=32, times=1, seed=3)
    samples = trajectory_state_time_samples(trajectory, time_label="t")
    matrix = jnp.eye(dimension) + 0.3 * (
        jnp.ones((dimension, dimension)) - jnp.eye(dimension)
    )
    score = _score_function(_MatrixScore(matrix), dimension)
    small = ScoreMatchingObjective(
        "score",
        samples,
        policy=ScoreMatchingPolicy("implicit", num_probes=8),
    )
    large = ScoreMatchingObjective(
        "score",
        samples,
        policy=ScoreMatchingPolicy("implicit", num_probes=512),
    )

    small_error = small.diagnostics(
        {"score": score}, key=jr.key(7)
    ).divergence_standard_error
    large_error = large.diagnostics(
        {"score": score}, key=jr.key(7)
    ).divergence_standard_error

    assert large_error < small_error


def test_implicit_score_matching_trains_gaussian_score_field():
    dimension = 3
    trajectory = _trajectory(dimension=dimension, paths=1024, times=2, seed=4)
    samples = trajectory_state_time_samples(trajectory, time_label="t")
    score = _score_function(_LinearScore(jnp.asarray(0.2)), dimension)
    objective = ScoreMatchingObjective(
        "score",
        samples,
        policy=ScoreMatchingPolicy("implicit", num_probes=4),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"score": score},
        constraints=(),
        objectives=(objective,),
    )

    trained = solver.solve(
        num_iter=120,
        optim=optax.adam(0.05),
        jit=True,
        keep_best=False,
        log_every=0,
    )
    coefficient = jnp.mean(
        trained.functions["score"].func(jnp.ones((dimension,)), jnp.asarray(0.0))
    )

    assert jnp.allclose(coefficient, -1.0, atol=0.12)


def test_resampled_particle_provider_runs_once_per_optimizer_update():
    calls = []

    def provider(key):
        calls.append(key)
        return _trajectory(dimension=2, paths=32, times=1, seed=len(calls))

    score = _score_function(_LinearScore(jnp.asarray(0.1)), 2)
    objective = ScoreMatchingObjective(
        "score",
        provider,
        sampling_mode="resample",
        policy=ScoreMatchingPolicy("implicit", num_probes=2),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"score": score},
        constraints=(),
        objectives=(objective,),
    )

    solver.solve(
        num_iter=4,
        optim=optax.sgd(0.01),
        jit=True,
        keep_best=False,
        log_every=0,
    )

    assert len(calls) == 4


def test_dimension_100_implicit_smoke_uses_jvps_and_rejects_scalar_score():
    dimension = 100
    trajectory = _trajectory(dimension=dimension, paths=4, times=1, seed=6)
    samples = trajectory_state_time_samples(trajectory, time_label="t")
    objective = ScoreMatchingObjective(
        "score",
        samples,
        policy=ScoreMatchingPolicy("implicit", num_probes=4),
    )
    score = _score_function(_LinearScore(jnp.asarray(-1.0)), dimension)
    loss = eqx.filter_jit(
        lambda current: objective.loss({"score": current}, key=jr.key(9))
    )(score)

    assert jnp.isfinite(loss)

    scalar = _score_function(lambda state, time: jnp.sum(state), dimension)
    with pytest.raises(ValueError, match="preserve|same shape"):
        objective.loss({"score": scalar}, key=jr.key(9))
