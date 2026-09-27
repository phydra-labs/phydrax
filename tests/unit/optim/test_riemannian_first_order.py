#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

import phydrax as phx


def _single_leaf_geometry(parameters: Any, manifold: Any) -> Any:
    return phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {"['point']": manifold},
    )


def test_riemannian_first_order_scenario_1() -> None:
    parameters = {"point": jnp.array([1.5, -2.0])}
    gradients = {"point": jnp.array([0.25, -0.75])}
    geometry = _single_leaf_geometry(
        parameters,
        phx.metrix.EuclideanManifold((2,)),
    )
    optimizer = phx.optim.riemannian_sgd(geometry, learning_rate=0.2)
    state = optimizer.init(parameters)
    actual, state = optimizer.update(gradients, state, parameters)

    transformation = optax.sgd(0.2)
    optax_state = transformation.init(parameters)
    updates, optax_state = transformation.update(gradients, optax_state, parameters)
    expected = eqx.apply_updates(parameters, updates)

    assert jax.tree.all(jax.tree.map(jnp.array_equal, actual, expected))
    assert int(state.step) == 1
    assert jnp.allclose(state.metrics.gradient_norm, jnp.sqrt(0.625))
    assert jnp.allclose(state.metrics.tangent_step_norm, 0.2 * jnp.sqrt(0.625))
    parameters = {
        "offset": jnp.array(1.0),
        "point": jnp.array([1.0, 0.0, 0.0]),
    }
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {"['point']": phx.metrix.SphereManifold(3)},
    )
    gradients = {
        "offset": jnp.array(0.0),
        "point": jnp.array([0.0, 3.0, 4.0]),
    }
    optimizer = phx.optim.riemannian_sgd(
        geometry,
        learning_rate=0.2,
        max_gradient_norm=2.5,
    )
    destination, state = optimizer.update(
        gradients,
        optimizer.init(parameters),
        parameters,
    )

    assert jnp.allclose(state.metrics.gradient_norm, 5.0)
    assert jnp.allclose(state.metrics.clipping_scale, 0.5)
    assert jnp.allclose(state.metrics.tangent_step_norm, 0.5)
    assert jnp.allclose(jnp.linalg.norm(destination["point"]), 1.0)
    assert destination["offset"] == parameters["offset"]
    parameters = {"point": jnp.array([1.0, 0.0, 0.0])}
    geometry = _single_leaf_geometry(parameters, phx.metrix.SphereManifold(3))
    optimizer = phx.optim.riemannian_sgd(
        geometry,
        learning_rate=lambda step: 0.2 / (step + 1.0),
    )
    update = eqx.filter_jit(optimizer.update)
    gradients = {"point": jnp.array([0.0, 1.0, 0.0])}
    state = optimizer.init(parameters)

    parameters, state = update(gradients, state, parameters)
    assert jnp.allclose(state.metrics.learning_rate, 0.2)
    parameters, state = update(gradients, state, parameters)
    assert jnp.allclose(state.metrics.learning_rate, 0.1)
    assert bool(geometry.contains(parameters))
    parameters = {"point": jnp.array([1.0, 0.0, 0.0])}
    geometry = _single_leaf_geometry(parameters, phx.metrix.SphereManifold(3))
    invalid_learning_rate: Any = []

    with pytest.raises(ValueError, match="learning_rate"):
        phx.optim.riemannian_sgd(geometry, learning_rate=0.0)
    with pytest.raises(ValueError, match="max_gradient_norm"):
        phx.optim.riemannian_sgd(geometry, max_gradient_norm=-1.0)
    with pytest.raises(ValueError, match="momentum"):
        phx.optim.riemannian_momentum(geometry, momentum=1.0)
    with pytest.raises(TypeError, match="learning_rate"):
        phx.optim.riemannian_sgd(geometry, learning_rate=invalid_learning_rate)

    optimizer = phx.optim.riemannian_sgd(geometry)
    with pytest.raises(Exception, match="gradient norm is not finite"):
        optimizer.update(
            {"point": jnp.array([jnp.nan, 0.0, 0.0])},
            optimizer.init(parameters),
            parameters,
        )
    parameters = {"point": jnp.array([1.0, -2.0])}
    geometry = _single_leaf_geometry(
        parameters,
        phx.metrix.EuclideanManifold((2,)),
    )
    optimizer = phx.optim.riemannian_momentum(
        geometry,
        learning_rate=0.1,
        momentum=0.75,
    )
    state = optimizer.init(parameters)
    expected_momentum = jnp.zeros((2,))

    for gradient in (jnp.array([1.0, -0.5]), jnp.array([0.2, 0.7])):
        expected_momentum = 0.75 * expected_momentum + gradient
        expected_parameters = parameters["point"] - 0.1 * expected_momentum
        parameters, state = optimizer.update(
            {"point": gradient},
            state,
            parameters,
        )
        assert jnp.allclose(parameters["point"], expected_parameters)
        assert jnp.allclose(state.momentum["point"], expected_momentum)
    probability = jnp.asarray([0.2, 0.3, 0.5])
    gradient = jnp.asarray([1.2, -0.4, 0.7])
    learning_rate = 0.15
    parameters = {"point": probability}
    geometry = _single_leaf_geometry(
        parameters,
        phx.metrix.ProbabilitySimplexManifold(3),
    )
    optimizer = phx.optim.riemannian_sgd(
        geometry,
        learning_rate=learning_rate,
    )
    state = optimizer.init(parameters)
    actual, _ = optimizer.update(
        {"point": gradient},
        state,
        parameters,
    )
    shifted, _ = optimizer.update(
        {"point": gradient + 5.0},
        state,
        parameters,
    )
    logits = jnp.log(probability) - learning_rate * gradient
    expected = jax.nn.softmax(logits)

    assert jnp.allclose(actual["point"], expected)
    assert jnp.allclose(shifted["point"], expected)
    parameters = {"point": jnp.array([1.0, 0.0, 0.0])}
    geometry = _single_leaf_geometry(parameters, phx.metrix.SphereManifold(3))
    optimizer = phx.optim.riemannian_momentum(
        geometry,
        learning_rate=0.15,
        momentum=0.8,
    )
    state = optimizer.init(parameters)
    update = eqx.filter_jit(optimizer.update)

    for gradient in (
        jnp.array([0.0, 1.0, 0.2]),
        jnp.array([0.1, 0.5, -0.3]),
        jnp.array([-0.2, 0.4, 0.7]),
    ):
        parameters, state = update({"point": gradient}, state, parameters)
        assert jnp.allclose(
            jnp.vdot(parameters["point"], state.momentum["point"]),
            0.0,
            atol=2e-12,
        )
        assert bool(geometry.contains(parameters))

    assert state.metrics.momentum_norm > 0.0


def test_sphere_rayleigh_quotient_converges_to_extremal_eigenvector() -> None:
    matrix = jnp.diag(jnp.array([5.0, 2.0, 0.5]))
    initial = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0)
    parameters = {"point": initial}
    geometry = _single_leaf_geometry(parameters, phx.metrix.SphereManifold(3))
    optimizer = phx.optim.riemannian_sgd(geometry, learning_rate=0.08)
    state = optimizer.init(parameters)

    def objective(tree: Any) -> Any:
        point = tree["point"]
        return -(point @ matrix @ point)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    update = eqx.filter_jit(optimizer.update)
    initial_value = objective(parameters)
    for _ in range(60):
        _, gradient = value_and_grad(parameters)
        parameters, state = update(gradient, state, parameters)

    assert objective(parameters) < initial_value
    assert jnp.abs(parameters["point"][0]) > 0.9999
    assert bool(geometry.contains(parameters))


def test_grassmann_pca_converges_to_leading_projector() -> None:
    covariance = jnp.diag(jnp.array([6.0, 4.0, 1.0, 0.2]))
    initial, _ = jnp.linalg.qr(
        jnp.array([[1.0, 0.2], [0.3, 1.0], [0.8, -0.4], [0.5, 0.7]])
    )
    parameters = {"point": initial}
    geometry = _single_leaf_geometry(
        parameters,
        phx.metrix.GrassmannManifold(4, 2),
    )
    optimizer = phx.optim.riemannian_sgd(geometry, learning_rate=0.06)
    state = optimizer.init(parameters)

    def objective(tree: Any) -> Any:
        point = tree["point"]
        return -jnp.trace(point.T @ covariance @ point)

    gradient_fn = jax.jit(jax.grad(objective))
    update = eqx.filter_jit(optimizer.update)
    for _ in range(100):
        gradient = gradient_fn(parameters)
        parameters, state = update(gradient, state, parameters)

    projector = parameters["point"] @ parameters["point"].T
    expected = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0]))
    assert jnp.linalg.norm(projector - expected) < 2e-4
    assert bool(geometry.contains(parameters))
