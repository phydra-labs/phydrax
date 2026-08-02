import jax
import jax.numpy as jnp

import phydrax as phx


def _polar_metric():
    chart = phx.metrix.CoordinateChart("polar", ("r", "theta"))
    metric = phx.metrix.diagonal_metric(
        lambda q: jnp.array([1.0, q[0] ** 2]),
        chart=chart,
    )
    return chart, metric


def test_polar_connection_intrinsic_operators_and_geodesics():
    _, metric = _polar_metric()
    point = jnp.array([2.0, 0.3])
    coefficients = phx.metrix.LeviCivitaConnection(metric).coefficients(point)

    expected = jnp.zeros((2, 2, 2))
    expected = expected.at[0, 1, 1].set(-2.0)
    expected = expected.at[1, 0, 1].set(0.5)
    expected = expected.at[1, 1, 0].set(0.5)
    assert jnp.allclose(coefficients, expected, atol=1e-10)

    scalar = lambda q: q[0] ** 2
    assert jnp.allclose(phx.metrix.gradient(scalar, metric, point), jnp.array([4.0, 0.0]))
    assert jnp.allclose(
        phx.metrix.covariant_hessian(scalar, metric, point),
        jnp.diag(jnp.array([2.0, 8.0])),
    )
    assert jnp.allclose(phx.metrix.laplace_beltrami(scalar, metric, point), 4.0)
    assert jnp.allclose(
        phx.metrix.divergence(lambda q: jnp.array([q[0], 0.0]), metric, point),
        2.0,
    )

    metric_derivative = phx.metrix.covariant_derivative(
        lambda q: metric(q),
        metric,
        phx.metrix.TensorType(("covariant", "covariant")),
        point,
    )
    assert jnp.allclose(metric_derivative, 0.0, atol=1e-10)

    velocity = jnp.array([0.4, -0.2])
    acceleration = phx.metrix.geodesic_acceleration(metric, point, velocity)
    assert jnp.allclose(
        acceleration,
        jnp.array(
            [point[0] * velocity[1] ** 2, -2.0 * velocity[0] * velocity[1] / point[0]]
        ),
    )
    state = jnp.concatenate((point, velocity))
    assert jnp.allclose(
        phx.metrix.geodesic_rhs(metric, state),
        jnp.concatenate((velocity, acceleration)),
    )


def test_flat_polar_curvature_vanishes_in_batches_and_under_jit():
    _, metric = _polar_metric()
    points = jnp.array([[1.2, 0.1], [2.0, -0.3], [3.5, 0.8]])

    riemann = jax.jit(lambda q: phx.metrix.riemann_tensor(metric, q))(points)
    ricci = phx.metrix.ricci_tensor(metric, points)
    scalar = phx.metrix.scalar_curvature(metric, points)

    assert riemann.shape == (3, 2, 2, 2, 2)
    assert jnp.allclose(riemann, 0.0, atol=1e-9)
    assert jnp.allclose(ricci, 0.0, atol=1e-9)
    assert jnp.allclose(scalar, 0.0, atol=1e-9)


def _sphere_metric(radius):
    chart = phx.metrix.CoordinateChart("sphere", ("theta", "phi"))
    metric = phx.metrix.diagonal_metric(
        lambda q: radius**2 * jnp.array([1.0, jnp.sin(q[0]) ** 2]),
        chart=chart,
    )
    return chart, metric


def test_sphere_curvature_contractions_symmetries_and_parameter_gradient():
    radius = 2.3
    _, metric = _sphere_metric(radius)
    point = jnp.array([1.1, 0.4])
    matrix = metric(point)
    riemann = phx.metrix.riemann_tensor(metric, point)
    ricci = phx.metrix.ricci_tensor(metric, point)
    scalar = phx.metrix.scalar_curvature(metric, point)
    einstein = phx.metrix.einstein_tensor(metric, point)
    sectional = phx.metrix.sectional_curvature(
        metric,
        point,
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
    )

    assert jnp.allclose(riemann + jnp.swapaxes(riemann, -1, -2), 0.0, atol=1e-9)
    assert jnp.allclose(ricci, matrix / radius**2, atol=1e-9)
    assert jnp.allclose(scalar, 2.0 / radius**2, atol=1e-9)
    assert jnp.allclose(einstein, 0.0, atol=1e-9)
    assert jnp.allclose(sectional, 1.0 / radius**2, atol=1e-9)

    def scalar_from_radius(value):
        _, learned_metric = _sphere_metric(value)
        return phx.metrix.scalar_curvature(learned_metric, point)

    derivative = jax.jit(jax.grad(scalar_from_radius))(jnp.array(radius))
    assert jnp.allclose(derivative, -4.0 / radius**3, atol=1e-8)


def test_hyperbolic_plane_has_negative_constant_curvature():
    chart = phx.metrix.CoordinateChart("half_plane", ("x", "y"))
    metric = phx.metrix.diagonal_metric(
        lambda q: jnp.ones(2) / q[1] ** 2,
        chart=chart,
    )
    point = jnp.array([0.2, 1.7])

    assert jnp.allclose(phx.metrix.ricci_tensor(metric, point), -metric(point), atol=1e-9)
    assert jnp.allclose(phx.metrix.scalar_curvature(metric, point), -2.0, atol=1e-9)
    assert jnp.allclose(
        phx.metrix.sectional_curvature(
            metric,
            point,
            jnp.array([1.0, 0.0]),
            jnp.array([0.0, 1.0]),
        ),
        -1.0,
        atol=1e-9,
    )
