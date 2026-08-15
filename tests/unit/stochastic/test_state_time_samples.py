import coordax as cx
import jax.numpy as jnp
import pytest

from phydrax.stochastic._state_time import (
    trajectory_state_time_measure,
    trajectory_state_time_samples,
)
from phydrax.stochastic._trajectory import StochasticTrajectory


def _trajectory():
    states = jnp.arange(3 * 4 * 2, dtype=float).reshape((3, 4, 2))
    valid = jnp.asarray(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    return StochasticTrajectory(
        jnp.linspace(0.0, 1.0, 4),
        states,
        valid=valid,
        realization_axes=("path",),
        realization_shape=(3,),
        time_axis="saved_time",
        state_axes=("state",),
    )


def test_state_time_samples_preserve_axes_masks_and_path_clusters():
    trajectory = _trajectory()
    batch = trajectory_state_time_samples(trajectory)

    assert batch.states.dims == ("path", "saved_time", "state")
    assert batch.times.dims == ("path", "saved_time")
    assert batch.valid.dims == ("path", "saved_time")
    assert batch.sample_axes == ("path", "saved_time")
    assert batch.num_nodes == 12
    assert batch.num_paths == 3
    assert batch.num_times == 4
    assert jnp.array_equal(jnp.asarray(batch.valid.data), trajectory.valid)
    assert jnp.array_equal(jnp.asarray(batch.path_indices.data[:, 0]), jnp.arange(3))
    assert jnp.all(jnp.asarray(batch.path_indices.data) == jnp.arange(3)[:, None])
    assert set(batch.samples) == {
        "x",
        "t",
        "path_index",
        "independence_index",
        "time_index",
    }


def test_state_time_measure_retains_time_for_per_time_reductions():
    trajectory = _trajectory()
    batch = trajectory_state_time_samples(
        trajectory,
        mode="per_time",
        state_label="state_position",
        time_label="time_value",
    )
    target = batch.target()
    assert isinstance(target.mask, cx.Field)
    assert isinstance(target.ancestry, cx.Field)

    assert target.sample_axes == ("path",)
    assert target.mask.dims == ("path", "saved_time")
    assert target.ancestry.dims == ("path", "saved_time")
    assert set(target.samples) == {
        "state_position",
        "time_value",
        "path_index",
        "independence_index",
        "time_index",
    }
    assert not target.independent


def test_adapter_adds_a_synthetic_realization_axis_for_one_path_per_case():
    trajectory = StochasticTrajectory(
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.zeros((3, 2)),
        state_axes=("state",),
    )
    batch = trajectory_state_time_samples(trajectory)

    assert batch.leading_axes == ("trajectory_sample",)
    assert batch.states.shape == (1, 3, 2)
    assert batch.times.shape == (1, 3)
    assert batch.sample_axes == ("trajectory_sample", "time")


def test_state_time_measure_carries_user_log_weights_without_flattening():
    trajectory = _trajectory()
    log_weights = jnp.linspace(-1.0, 1.0, 12).reshape((3, 4))
    target = trajectory_state_time_measure(
        trajectory,
        log_weights=log_weights,
    )
    assert isinstance(target.log_weights, cx.Field)

    assert target.log_weights.dims == ("path", "saved_time")
    assert jnp.array_equal(jnp.asarray(target.log_weights.data), log_weights)
    assert target.provenance == "stochastic-trajectory:state-time:global"


def test_state_time_adapter_rejects_misaligned_weights_and_labels():
    trajectory = _trajectory()
    with pytest.raises(ValueError, match="match"):
        trajectory_state_time_samples(trajectory, log_weights=jnp.ones((3, 3)))
    with pytest.raises(ValueError, match="distinct"):
        trajectory_state_time_samples(trajectory, state_label="x", time_label="x")
