import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _polar_transition():
    polar = phx.metrix.CoordinateChart("polar", ("r", "theta"))
    cartesian = phx.metrix.CoordinateChart("cartesian", ("x", "y"))

    def to_cartesian(q):
        return jnp.array([q[0] * jnp.cos(q[1]), q[0] * jnp.sin(q[1])])

    def to_polar(x):
        return jnp.array([jnp.linalg.norm(x), jnp.arctan2(x[1], x[0])])

    return (
        polar,
        cartesian,
        phx.metrix.ChartTransition(
            polar,
            cartesian,
            to_cartesian,
            inverse=to_polar,
        ),
    )


def test_chart_transition_batches_derivatives_inverse_and_composition():
    polar, cartesian, transition = _polar_transition()
    points = jnp.array([[2.0, 0.3], [1.5, -0.4]])

    mapped = transition(points)
    restored = transition.inverse(mapped)
    jacobian = transition.jacobian(points)
    inverse_jacobian = transition.inverse_jacobian(mapped)

    assert mapped.shape == points.shape
    assert jacobian.shape == (2, 2, 2)
    assert transition.hessian(points).shape == (2, 2, 2, 2)
    assert jnp.allclose(restored, points)
    assert jnp.allclose(
        inverse_jacobian @ jacobian,
        jnp.eye(2),
        atol=1e-10,
    )

    identity = phx.metrix.ChartTransition.identity(cartesian)
    composed = transition.compose(identity)
    assert jnp.allclose(composed(points), mapped)
    assert jnp.allclose(composed.inverse(mapped), points)

    incompatible = phx.metrix.ChartTransition.identity(
        phx.metrix.CoordinateChart("other", ("x", "y"))
    )
    with pytest.raises(ValueError, match="mismatched intermediate charts"):
        transition.compose(incompatible)


def test_metric_constructors_pullback_jets_validation_and_parameter_gradients():
    polar, cartesian, transition = _polar_transition()
    cartesian_metric = phx.metrix.euclidean_metric(cartesian)
    metric = phx.metrix.pullback_metric(cartesian_metric, transition)
    point = jnp.array([2.0, 0.4])
    batch = jnp.array([[2.0, 0.4], [3.0, -0.2]])

    assert jnp.allclose(metric(point), jnp.diag(jnp.array([1.0, 4.0])), atol=1e-10)
    assert jnp.allclose(metric.inverse(point), jnp.diag(jnp.array([1.0, 0.25])))
    assert jnp.allclose(metric.volume_density(batch), jnp.array([2.0, 3.0]))
    assert jnp.allclose(jax.jit(metric)(batch), metric(batch))

    jet = phx.metrix.metric_jet(metric, point, order=2)
    assert jet.first_derivative is not None
    assert jet.second_derivative is not None
    assert jet.matrix.shape == (2, 2)
    assert jet.first_derivative.shape == (2, 2, 2)
    assert jet.second_derivative.shape == (2, 2, 2, 2)
    assert jnp.allclose(jet.first_derivative[1, 1], jnp.array([4.0, 0.0]))
    assert jnp.allclose(jet.second_derivative[1, 1, 0, 0], 2.0)

    report = phx.metrix.validate_metric(metric, batch)
    assert bool(report.valid)
    assert report.minimum_eigenvalue > 0.0

    asymmetric = phx.metrix.RiemannianMetric(
        lambda q: jnp.array([[1.0, q[0]], [0.0, 1.0]]),
        chart=polar,
    )
    invalid = phx.metrix.validate_metric(asymmetric, point, raise_on_error=False)
    assert not bool(invalid.valid)
    with pytest.raises(ValueError, match="Metric validation failed"):
        phx.metrix.validate_metric(asymmetric, point)

    def learned_log_volume(parameter):
        learned = phx.metrix.cholesky_metric(
            lambda q: jnp.array([[parameter + q[0], 0.0], [0.2, parameter]]),
            chart=polar,
            minimum_diagonal=1e-3,
        )
        return learned.log_volume_density(point)

    value = learned_log_volume(jnp.array(0.7))
    derivative = jax.jit(jax.grad(learned_log_volume))(jnp.array(0.7))
    assert jnp.isfinite(value)
    assert jnp.isfinite(derivative)


def test_tensor_index_operations_and_coordinate_transformation_laws():
    source = phx.metrix.CoordinateChart("source", ("x", "y"))
    target = phx.metrix.CoordinateChart("target", ("u", "v"))
    transition = phx.metrix.ChartTransition(
        source,
        target,
        lambda q: jnp.array([2.0 * q[0], 3.0 * q[1]]),
        inverse=lambda q: jnp.array([q[0] / 2.0, q[1] / 3.0]),
    )
    point = jnp.array([0.2, -0.3])
    vector = jnp.array([1.0, 2.0])
    covector = jnp.array([1.0, 2.0])

    assert jnp.allclose(
        phx.metrix.pushforward_vector(transition, vector, point),
        jnp.array([2.0, 6.0]),
    )
    assert jnp.allclose(
        phx.metrix.reexpress_tensor(
            transition,
            covector,
            phx.metrix.COVECTOR_TENSOR,
            point,
        ),
        jnp.array([0.5, 2.0 / 3.0]),
    )
    assert jnp.allclose(
        phx.metrix.pullback_covector(
            transition,
            jnp.array([0.5, 2.0 / 3.0]),
            point,
        ),
        covector,
    )

    mixed_type = phx.metrix.TensorType(("contravariant", "covariant"))
    mixed = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    jacobian = transition.jacobian(point)
    expected_mixed = jacobian @ mixed @ jnp.linalg.inv(jacobian)
    transformed = phx.metrix.reexpress_tensor(
        transition,
        mixed,
        mixed_type,
        point,
    )
    assert jnp.allclose(transformed, expected_mixed)
    assert jnp.allclose(
        phx.metrix.contract_indices(transformed, mixed_type, 0, 1, point, 2),
        jnp.trace(mixed),
    )

    density = phx.metrix.reexpress_tensor(
        transition,
        jnp.array(12.0),
        phx.metrix.DENSITY_TENSOR,
        point,
    )
    assert jnp.allclose(density, 2.0)

    metric = phx.metrix.diagonal_metric(
        lambda q: jnp.array([4.0, 9.0]),
        chart=source,
    )
    lowered = phx.metrix.lower_index(
        vector,
        metric,
        point,
        tensor_type=phx.metrix.VECTOR_TENSOR,
    )
    raised = phx.metrix.raise_index(
        lowered,
        metric,
        point,
        tensor_type=phx.metrix.COVECTOR_TENSOR,
    )
    assert jnp.allclose(lowered, jnp.array([4.0, 18.0]))
    assert jnp.allclose(raised, vector)
    assert jnp.allclose(phx.metrix.inner_product(vector, vector, metric, point), 40.0)
    assert jnp.allclose(
        phx.metrix.tensor_norm_squared(
            vector,
            metric,
            phx.metrix.VECTOR_TENSOR,
            point,
        ),
        40.0,
    )


def test_tensor_contracts_reject_invalid_variance_and_shapes():
    chart = phx.metrix.CoordinateChart("plane", ("x", "y"))
    metric = phx.metrix.euclidean_metric(chart)
    point = jnp.zeros(2)

    with pytest.raises(ValueError, match="Tensor variance"):
        phx.metrix.TensorType(("invalid",))
    with pytest.raises(ValueError, match="covariant source axis"):
        phx.metrix.raise_index(
            jnp.ones(2),
            metric,
            point,
            tensor_type=phx.metrix.VECTOR_TENSOR,
        )
    with pytest.raises(ValueError, match="requires shape"):
        phx.metrix.tensor_norm_squared(
            jnp.ones(3),
            metric,
            phx.metrix.VECTOR_TENSOR,
            point,
        )
