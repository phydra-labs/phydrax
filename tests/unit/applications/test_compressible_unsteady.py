import jax.numpy as jnp
import numpy as np

from phydrax.applications.compressible_flow import (
    CompressibleLoadHistory,
    CompressibleShockTrackPlan,
    CompressibleSnapshotMetricPlan,
)


def test_shock_tracking_recovers_moving_compression_location():
    coordinate = jnp.linspace(0.0, 1.0, 201)
    locations = jnp.asarray((0.35, 0.55))
    pressure = jnp.stack(
        tuple(1.0 + 0.2 * jnp.tanh(100.0 * (coordinate - x)) for x in locations)
    )
    result = CompressibleShockTrackPlan((0.2, 0.8), minimum_gradient=1.0).evaluate(
        coordinate, pressure
    )

    assert bool(jnp.all(result.successful))
    np.testing.assert_allclose(result.location, locations, atol=0.006)
    assert jnp.all(result.peak_margin >= 0.0)


def test_snapshot_metric_is_volume_weighted_and_invertible():
    volumes = jnp.asarray(((1.0, 4.0), (9.0, 16.0)))
    plan = CompressibleSnapshotMetricPlan(volumes, (0, 2), (2.0, 5.0))
    state = jnp.arange(2 * 4 * 3, dtype=float).reshape((2, 2, 2, 3))
    encoded = plan.encode(state)
    decoded = plan.decode(encoded)

    expected = jnp.take(state, jnp.asarray((0, 2)), axis=-1)
    np.testing.assert_allclose(decoded, expected, rtol=1.0e-12, atol=1.0e-12)
    assert encoded.shape == (2, 8)
    first_cell = expected[:, 0, 0, :] / jnp.asarray((2.0, 5.0))
    np.testing.assert_allclose(encoded[:, :2], first_cell)


def test_load_history_binds_validity_and_source_identity():
    times = jnp.asarray((0.0, 0.1, 0.2))
    history = CompressibleLoadHistory(
        times,
        jnp.asarray(((0.1, 0.4), (0.2, 0.5), (0.15, 0.45))),
        jnp.asarray((0.01, 0.02, 0.015)),
        jnp.asarray((0.4, 0.42, 0.41)),
        valid=jnp.asarray((True, False, True)),
        source_id="buffet-test",
    )

    assert history.force_coefficients.shape == (3, 2)
    assert history.valid.tolist() == [True, False, True]
    assert history.history_id
