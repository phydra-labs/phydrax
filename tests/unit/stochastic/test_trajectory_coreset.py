# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

import phydrax as phx


def test_trajectory_blocks_preserve_boundaries_weights_and_references():
    trajectory = phx.stochastic.StochasticTrajectory(
        jnp.asarray([0.0, 0.2, 0.5, 1.0]),
        jnp.arange(8.0).reshape((4, 2)),
        valid=jnp.asarray([True, True, True, False]),
        state_axes=("state",),
    )
    view = phx.stochastic.trajectory_blocks(trajectory, block_length=2, stride=1)
    assert jnp.array_equal(view.valid, jnp.asarray([True, True, False]))
    result = phx.stochastic.compress_trajectory_blocks(
        view,
        phx.coresets.MomentRecombination(),
        features=view.states.reshape((view.count, -1)),
        weighting="duration",
    )
    assert result.states.shape[1:] == (2, 2)
    assert jnp.all(result.weights >= 0.0)
    assert len(result.references) == result.selection.capacity

    dataset = result.to_operator_dataset()
    assert jnp.array_equal(dataset.case_mask, result.mask)
    assert jnp.array_equal(dataset.case_log_weights, result.selection.log_weights)
