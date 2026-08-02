import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _polar_metric():
    chart = phx.metrix.CoordinateChart("polar", ("r", "theta"))
    return phx.metrix.diagonal_metric(
        lambda q: jnp.array([1.0, q[0] ** 2]),
        chart=chart,
    )


def test_coordinate_stratonovich_correction_supports_rectangular_noise_and_batches():
    drift = lambda q: jnp.array([q[1], -q[0]])

    def diffusion(q):
        return jnp.array(
            [
                [q[0], 0.0, q[1]],
                [0.0, 2.0, q[0]],
            ]
        )

    point = jnp.array([0.6, -0.4])
    corrected = phx.metrix.coordinate_stratonovich_to_ito_drift(
        drift,
        diffusion,
        point,
    )
    expected = drift(point) + jnp.array([point[0], 0.5 * point[1]])

    assert jnp.allclose(corrected, expected)
    points = jnp.array([[0.6, -0.4], [0.2, 0.7]])
    batched = phx.metrix.coordinate_stratonovich_to_ito_drift(
        drift,
        diffusion,
        points,
    )
    assert jnp.allclose(
        batched,
        jax.vmap(
            lambda q: phx.metrix.coordinate_stratonovich_to_ito_drift(
                drift,
                diffusion,
                q,
            )
        )(points),
    )
    assert jnp.allclose(
        jax.jit(
            lambda q: phx.metrix.coordinate_stratonovich_to_ito_drift(
                drift,
                diffusion,
                q,
            )
        )(point),
        corrected,
    )


def test_coordinate_and_covariant_drifts_agree_for_polar_brownian_motion():
    metric = _polar_metric()
    covariance = lambda q: metric.inverse(q)
    coordinate_drift = lambda q: jnp.array([0.5 / q[0], 0.0])
    points = jnp.array([[1.2, 0.1], [2.0, -0.4]])

    covariant = phx.metrix.coordinate_to_covariant_drift(
        coordinate_drift,
        covariance,
        metric,
        points,
    )
    assert jnp.allclose(covariant, 0.0, atol=1e-10)


def test_covariant_generators_and_fokker_planck_are_chart_consistent():
    cartesian_chart = phx.metrix.CoordinateChart("cartesian", ("x", "y"))
    cartesian_metric = phx.metrix.euclidean_metric(cartesian_chart)
    polar_metric = _polar_metric()
    cartesian_point = jnp.array([1.3, -0.7])
    polar_point = jnp.array(
        [
            jnp.linalg.norm(cartesian_point),
            jnp.arctan2(cartesian_point[1], cartesian_point[0]),
        ]
    )
    zero = lambda q: jnp.zeros(2)
    cartesian_covariance = lambda q: cartesian_metric.inverse(q)
    polar_covariance = lambda q: polar_metric.inverse(q)
    cartesian_observable = lambda q: jnp.dot(q, q)
    polar_observable = lambda q: q[0] ** 2

    cartesian_generator = phx.metrix.covariant_kolmogorov_generator(
        cartesian_observable,
        zero,
        cartesian_metric,
        cartesian_point,
        covariance=cartesian_covariance,
    )
    polar_generator = phx.metrix.covariant_kolmogorov_generator(
        polar_observable,
        zero,
        polar_metric,
        polar_point,
        covariance=polar_covariance,
    )
    assert jnp.allclose(cartesian_generator, 2.0)
    assert jnp.allclose(polar_generator, cartesian_generator, atol=1e-10)
    assert jnp.allclose(
        phx.metrix.brownian_generator(
            polar_observable,
            polar_metric,
            polar_point,
        ),
        polar_generator,
    )

    cartesian_forward = phx.metrix.covariant_fokker_planck_operator(
        cartesian_observable,
        zero,
        cartesian_metric,
        cartesian_point,
        covariance=cartesian_covariance,
    )
    polar_forward = phx.metrix.covariant_fokker_planck_operator(
        polar_observable,
        zero,
        polar_metric,
        polar_point,
        covariance=polar_covariance,
    )
    assert jnp.allclose(cartesian_forward, 2.0)
    assert jnp.allclose(polar_forward, cartesian_forward, atol=1e-9)


def test_covariant_stochastic_operators_support_vector_outputs_jit_and_gradients():
    chart = phx.metrix.CoordinateChart("line", ("x",))
    metric = phx.metrix.euclidean_metric(chart)
    observable = lambda q: jnp.array([q[0] ** 2, q[0] ** 3])
    drift = lambda q: jnp.array([0.4])
    covariance = lambda q: jnp.array([[0.7 + q[0] ** 2]])
    point = jnp.array([0.3])

    generated = phx.metrix.covariant_kolmogorov_generator(
        observable,
        drift,
        metric,
        point,
        covariance=covariance,
    )
    expected = jnp.array(
        [
            0.8 * point[0] + 0.7 + point[0] ** 2,
            1.2 * point[0] ** 2 + 3.0 * point[0] * (0.7 + point[0] ** 2),
        ]
    )
    assert jnp.allclose(generated, expected)
    assert jnp.allclose(
        jax.jit(
            lambda q: phx.metrix.covariant_kolmogorov_generator(
                observable,
                drift,
                metric,
                q,
                covariance=covariance,
            )
        )(point),
        generated,
    )

    def value(scale):
        return phx.metrix.covariant_kolmogorov_generator(
            lambda q: q[0] ** 2,
            drift,
            metric,
            point,
            diffusion=lambda q: jnp.array([[scale * q[0]]]),
        )

    derivative = jax.grad(value)(jnp.array(0.8))
    assert jnp.isfinite(derivative)
    assert not jnp.allclose(derivative, 0.0)


def test_stochastic_geometry_rejects_ambiguous_or_malformed_coefficients():
    chart = phx.metrix.CoordinateChart("plane", ("x", "y"))
    metric = phx.metrix.euclidean_metric(chart)
    point = jnp.zeros(2)
    scalar = lambda q: jnp.dot(q, q)
    drift = lambda q: jnp.zeros(2)

    with pytest.raises(ValueError, match="either diffusion or covariance"):
        phx.metrix.covariant_kolmogorov_generator(
            scalar,
            drift,
            metric,
            point,
            diffusion=lambda q: jnp.eye(2),
            covariance=lambda q: jnp.eye(2),
        )
    with pytest.raises(ValueError, match="diffusion"):
        phx.metrix.coordinate_stratonovich_to_ito_drift(
            drift,
            lambda q: jnp.ones(2),
            point,
        )
