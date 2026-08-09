#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _polynomial_data(coordinates, *, reset_mask=None):
    time = jnp.asarray(coordinates)
    states = jnp.stack((time**2 + 2.0 * time, -0.5 * time**2), axis=-1)
    return phx.dynamics.TrajectoryData(
        time,
        states,
        state_layout=phx.dynamics.StateLayout(
            (2,), component_names=("position", "energy")
        ),
        reset_mask=reset_mask,
        coordinate_id="physical-time",
        source_id="quadratic",
    )


def test_finite_difference_is_exact_for_quadratic_on_irregular_interior():
    time = jnp.asarray([0.0, 0.15, 0.6, 1.4, 2.0])
    data = _polynomial_data(time)

    estimate = phx.dynamics.identification.finite_difference_derivative(
        data, endpoint="invalid"
    )

    expected = jnp.stack((2.0 * time + 2.0, -time), axis=-1)
    np.testing.assert_allclose(
        np.asarray(estimate.values[1:-1]),
        np.asarray(expected[1:-1]),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(estimate.valid),
        np.asarray([False, True, True, True, False]),
    )
    attached = estimate.attach(data)
    assert attached.derivatives is not None
    assert attached.source_id.endswith("finite-difference:invalid")


def test_local_polynomial_is_exact_and_never_crosses_a_reset():
    time = jnp.arange(7.0)
    data = _polynomial_data(
        time,
        reset_mask=jnp.asarray([False, False, False, True, False, False]),
    )

    estimate = phx.dynamics.identification.local_polynomial_derivative(
        data, degree=2, window_radius=3
    )

    expected = jnp.stack((2.0 * time + 2.0, -time), axis=-1)
    np.testing.assert_array_equal(
        np.asarray(estimate.valid),
        np.asarray([True, True, True, True, True, True, True]),
    )
    np.testing.assert_allclose(
        np.asarray(estimate.values), np.asarray(expected), rtol=2e-5, atol=2e-5
    )


def test_local_polynomial_reports_underdetermined_segment_as_invalid():
    time = jnp.arange(6.0)
    data = _polynomial_data(
        time,
        reset_mask=jnp.asarray([False, True, False, False, False]),
    )

    estimate = phx.dynamics.identification.local_polynomial_derivative(
        data, degree=2, window_radius=2
    )

    assert not bool(estimate.valid[0])
    assert not bool(estimate.valid[1])
    assert bool(jnp.all(estimate.valid[2:]))
    assert bool(jnp.all(jnp.isnan(estimate.values[:2])))


def test_bspline_derivative_fits_each_segment_independently():
    time = jnp.linspace(0.0, 2.0, 8)
    values = jnp.stack((time**3, 2.0 * time**3 - time), axis=-1)
    data = phx.dynamics.TrajectoryData(
        time,
        values,
        state_layout=phx.dynamics.StateLayout((2,)),
        reset_mask=jnp.asarray([False, False, False, True, False, False, False]),
        source_id="piecewise-cubic",
    )

    estimate = phx.dynamics.identification.bspline_derivative(data)

    expected = jnp.stack((3.0 * time**2, 6.0 * time**2 - 1.0), axis=-1)
    assert bool(jnp.all(estimate.valid))
    np.testing.assert_allclose(
        np.asarray(estimate.values), np.asarray(expected), rtol=3e-4, atol=3e-4
    )
