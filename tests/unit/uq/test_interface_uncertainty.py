#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_interface_predictive_summary_retains_phase_measure_draws():
    samples = jnp.asarray(
        (
            (-1.0, 1.0),
            (-0.1, 0.1),
            (1.0, -1.0),
        )
    )
    summary = phx.uq.interface_predictive_summary(
        samples,
        jnp.asarray((2.0, 3.0)),
        width=0.2,
    )

    assert summary.occupancy_probability.shape == (2,)
    assert summary.phase_measure_samples.shape == (3,)
    np.testing.assert_allclose(summary.phase_measure_samples[0], 2.0, atol=1.0e-14)
    np.testing.assert_allclose(summary.phase_measure_samples[2], 3.0, atol=1.0e-14)
    assert jnp.all(
        (summary.interface_probability >= 0.0) & (summary.interface_probability <= 1.0)
    )


def test_active_acquisition_combines_signals_without_duplicate_points():
    points = jnp.arange(6.0)[:, None]
    result = phx.uq.select_interface_acquisition(
        points,
        jnp.asarray((0.0, 0.1, 1.0, 0.2, 0.1, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.5)),
        3,
        existing_points=jnp.asarray(((0.0,),)),
    )

    assert result.indices.shape == (3,)
    assert jnp.unique(result.indices).size == 3
    assert 2 in set(map(int, result.indices))


def test_bounded_context_adaptation_improves_residual_without_leaving_ball():
    policy = phx.nn.operator.training.BoundedResidualAdaptationPolicy(
        iterations=20,
        learning_rate=0.2,
        maximum_update_norm=0.5,
    )
    result = phx.nn.operator.training.adapt_operator_context(
        jnp.asarray((0.0, 0.0)),
        lambda context: jnp.sum((context - jnp.asarray((1.0, 0.0))) ** 2),
        policy=policy,
        jit=True,
    )

    assert bool(result.accepted)
    assert result.final_objective < result.initial_objective
    assert result.update_norm <= 0.5 + 1.0e-12
    np.testing.assert_allclose(result.context, jnp.asarray((0.5, 0.0)), atol=1.0e-6)
