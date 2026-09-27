from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _trajectory(*, valid: Any = None, independent: Any = True) -> Any:
    times = jnp.asarray([0.0, 0.2, 0.5, 1.0])
    states = jnp.arange(2 * 3 * 4, dtype="float64").reshape((2, 3, 4, 1))
    realizations = (
        tuple(
            phx.stochastic.WienerRealization(
                jr.key(index),
                (1,),
                support=(0.0, 1.0),
                sample_shape=(3,),
            )
            for index in range(2)
        )
        if independent
        else (None, None)
    )
    return phx.stochastic.StochasticTrajectory(
        times,
        states,
        valid=valid,
        case_axes=("case",),
        case_shape=(2,),
        realization_axes=("path",),
        realization_shape=(3,),
        state_axes=("state",),
        realizations=realizations,
        case_ids=("left", "right"),
    )


def test_integration_measures_scenario_1() -> None:
    valid = jnp.ones((2, 3, 4), dtype="bool").at[0, 0, 2].set(False)
    trajectory = _trajectory(valid=valid)
    target = phx.stochastic.trajectory_measure(trajectory, mode="marginal")

    estimate = phx.integration.integrate(lambda states: states, target)

    safe = jnp.where(valid[..., None], trajectory.states, 0.0)
    expected = jnp.sum(safe, axis=1) / jnp.sum(valid, axis=1)[..., None]
    assert estimate.value.dims == ("case", "time", "state")
    assert jnp.allclose(jnp.asarray(estimate.value.data), expected)
    assert jnp.all(estimate.successful)
    assert estimate.diagnostics.independent
    assert estimate.error_kind == "weighted-iid-standard-error"
    valid = jnp.ones((2, 3, 4), dtype="bool").at[0, 0, 2].set(False)
    trajectory = _trajectory(valid=valid)
    target = phx.stochastic.trajectory_measure(trajectory, mode="path")

    estimate = phx.integration.integrate(lambda states: states, target)

    expected_left = jnp.mean(trajectory.states[0, 1:], axis=0)
    expected_right = jnp.mean(trajectory.states[1], axis=0)
    assert estimate.value.dims == ("case", "time", "state")
    assert jnp.allclose(
        jnp.asarray(estimate.value.data),
        jnp.stack((expected_left, expected_right)),
    )
    assert jnp.array_equal(estimate.diagnostics.active_samples, jnp.asarray([2, 3]))
    invalid_mode: Any = "unknown"
    with pytest.raises(ValueError, match="mode"):
        phx.stochastic.trajectory_measure(_trajectory(), mode=invalid_mode)
    trajectory = _trajectory(independent=False)
    target = phx.stochastic.trajectory_measure(trajectory)

    estimate = phx.integration.integrate(lambda states: states, target)

    assert not estimate.diagnostics.independent
    assert estimate.diagnostics.standard_error is None
    assert estimate.error_estimate is None
    for rule in ("left", "trapezoid"):
        times = jnp.asarray(
            [
                [0.0, 0.2, 0.5, 1.0],
                [0.0, 0.3, 0.7, 1.2],
            ]
        )
        states = times[..., None]
        valid = jnp.asarray(
            [
                [True, True, True, True],
                [True, True, True, False],
            ]
        )
        trajectory = phx.stochastic.StochasticTrajectory(
            times,
            states,
            valid=valid,
            realization_axes=("path",),
            realization_shape=(2,),
            state_axes=("state",),
            realizations=(None,),
        )
        target = phx.stochastic.time_measure(trajectory, rule=rule)

        estimate = phx.integration.integrate(1.0, target)

        assert estimate.value.dims == ("path",)
        assert jnp.allclose(jnp.asarray(estimate.value.data), jnp.asarray([1.0, 0.7]))
        assert jnp.all(estimate.successful)


def test_integration_measures_scenario_2() -> None:
    times = jnp.asarray([[0.0, 0.2, 0.5, 1.0], [0.0, 0.3, 0.7, 1.2]])
    trajectory = phx.stochastic.StochasticTrajectory(
        times,
        times[..., None],
        realization_axes=("path",),
        realization_shape=(2,),
        state_axes=("state",),
        realizations=(None,),
    )
    target = phx.stochastic.time_measure(
        trajectory,
        rule="trapezoid",
        normalized=True,
    )

    estimate = phx.integration.integrate(target.points, target)

    assert jnp.allclose(jnp.asarray(estimate.value.data), jnp.asarray([0.5, 0.6]))
    valid = jnp.asarray([[True, False, True, False]])
    trajectory = phx.stochastic.StochasticTrajectory(
        jnp.asarray([0.0, 0.2, 0.5, 1.0]),
        jnp.ones((1, 4, 1)),
        valid=valid,
        realization_axes=("path",),
        realization_shape=(1,),
        state_axes=("state",),
        realizations=(None,),
    )

    with pytest.raises(ValueError, match="contiguous prefixes"):
        phx.stochastic.time_measure(trajectory)
