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


def _sphere_geometry(parameters):
    return phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {"['point']": phx.metrix.SphereManifold(3)},
    )


def test_riemannian_adam_matches_optax_adam_with_pointwise_euclidean_factors():
    parameters = {
        "free": jnp.array([1.5, -2.0]),
        "point": jnp.array([1.0, 0.0, 0.0]),
    }
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {
            "['free']": phx.metrix.EuclideanManifold(()),
            "['point']": phx.metrix.SphereManifold(3),
        },
    )
    optimizer = phx.optim.riemannian_adam(
        geometry,
        learning_rate=0.05,
        first_moment_decay=0.8,
        second_moment_decay=0.9,
        epsilon=1e-7,
    )
    state = optimizer.init(parameters)
    transformation = optax.adam(
        0.05,
        b1=0.8,
        b2=0.9,
        eps=1e-7,
    )
    expected = parameters["free"]
    optax_state = transformation.init(expected)

    for free_gradient in (
        jnp.array([0.5, -1.0]),
        jnp.array([-0.2, 0.4]),
        jnp.array([0.7, 0.1]),
    ):
        parameters, state = optimizer.update(
            {
                "free": free_gradient,
                "point": jnp.zeros((3,)),
            },
            state,
            parameters,
        )
        updates, optax_state = transformation.update(
            free_gradient,
            optax_state,
            expected,
        )
        expected = optax.apply_updates(expected, updates)
        assert jnp.allclose(jnp.asarray(parameters["free"]), jnp.asarray(expected))

    assert state.second_moment["free"].shape == (2,)
    assert state.second_moment["point"].shape == ()
    assert jnp.array_equal(parameters["point"], jnp.array([1.0, 0.0, 0.0]))
    assert "riemannian_adam" in phx.optim.__all__


def test_riemannian_adam_is_equivariant_under_ambient_orthogonal_changes():
    angle = jnp.asarray(0.43)
    rotation = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    parameters = {
        "point": jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
    }
    target = jnp.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    transformed_parameters = {"point": parameters["point"] @ rotation.T}
    transformed_target = target @ rotation.T
    geometry = _sphere_geometry(parameters)
    transformed_geometry = _sphere_geometry(transformed_parameters)
    optimizer = phx.optim.riemannian_adam(
        geometry,
        learning_rate=0.04,
        first_moment_decay=0.7,
        second_moment_decay=0.8,
    )
    transformed_optimizer = phx.optim.riemannian_adam(
        transformed_geometry,
        learning_rate=0.04,
        first_moment_decay=0.7,
        second_moment_decay=0.8,
    )
    state = optimizer.init(parameters)
    transformed_state = transformed_optimizer.init(transformed_parameters)

    def objective(tree):
        return jnp.sum((tree["point"] - target) ** 2)

    def transformed_objective(tree):
        return jnp.sum((tree["point"] - transformed_target) ** 2)

    for _ in range(6):
        parameters, state = optimizer.update(
            jax.grad(objective)(parameters),
            state,
            parameters,
        )
        transformed_parameters, transformed_state = transformed_optimizer.update(
            jax.grad(transformed_objective)(transformed_parameters),
            transformed_state,
            transformed_parameters,
        )

    assert jnp.allclose(
        transformed_parameters["point"],
        parameters["point"] @ rotation.T,
        atol=2e-6,
    )
    assert jnp.allclose(
        transformed_state.first_moment["point"],
        state.first_moment["point"] @ rotation.T,
        atol=2e-6,
    )
    assert jnp.allclose(
        transformed_state.second_moment["point"],
        state.second_moment["point"],
        atol=2e-6,
    )


def test_riemannian_amsgrad_tracks_monotone_factor_moments_and_tangent_momentum():
    parameters = {
        "point": jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
    }
    geometry = _sphere_geometry(parameters)
    optimizer = phx.optim.riemannian_adam(
        geometry,
        learning_rate=0.02,
        first_moment_decay=0.6,
        second_moment_decay=0.5,
        amsgrad=True,
    )
    state = optimizer.init(parameters)
    previous_maximum = state.maximum_second_moment["point"]

    for scale in (2.0, 0.1, 1.0):
        gradient = {
            "point": scale
            * jnp.array(
                [[0.0, 1.0, -0.5], [0.5, 0.0, 1.0]],
            )
        }
        parameters, state = optimizer.update(gradient, state, parameters)
        assert jnp.all(state.maximum_second_moment["point"] >= previous_maximum)
        assert jnp.allclose(
            jnp.sum(parameters["point"] * state.first_moment["point"], axis=-1),
            0.0,
            atol=2e-6,
        )
        assert bool(geometry.contains(parameters))
        previous_maximum = state.maximum_second_moment["point"]

    metrics = optimizer.step_metrics(state)
    assert metrics.adaptive_denominator_minimum > 0.0
    assert metrics.adaptive_denominator_maximum >= metrics.adaptive_denominator_minimum
    assert metrics.transported_tangent_residual < 2e-6


def test_riemannian_adam_eager_and_jit_updates_agree():
    parameters = {
        "offset": jnp.array([0.2, -0.3]),
        "point": jnp.array([1.0, 0.0, 0.0]),
    }
    geometry = _sphere_geometry(parameters)
    optimizer = phx.optim.riemannian_adam(
        geometry,
        learning_rate=lambda step: 0.03 / (step + 1.0),
        max_gradient_norm=0.8,
    )
    gradient = {
        "offset": jnp.array([0.1, -0.4]),
        "point": jnp.array([0.0, 2.0, -1.0]),
    }
    initial_state = optimizer.init(parameters)

    eager_parameters, eager_state = optimizer.update(
        gradient,
        initial_state,
        parameters,
    )
    compiled_parameters, compiled_state = eqx.filter_jit(optimizer.update)(
        gradient,
        initial_state,
        parameters,
    )

    assert jax.tree.all(jax.tree.map(jnp.allclose, eager_parameters, compiled_parameters))
    assert jax.tree.all(
        jax.tree.map(
            jnp.allclose,
            eager_state.first_moment,
            compiled_state.first_moment,
        )
    )
    assert jax.tree.all(
        jax.tree.map(
            jnp.allclose,
            eager_state.second_moment,
            compiled_state.second_moment,
        )
    )
    assert jnp.allclose(
        eager_state.metrics.adaptive_denominator_maximum,
        compiled_state.metrics.adaptive_denominator_maximum,
    )


def test_riemannian_adam_rejects_invalid_configuration_and_state():
    parameters = {"point": jnp.array([1.0, 0.0, 0.0])}
    geometry = _sphere_geometry(parameters)
    invalid_amsgrad: Any = 1

    with pytest.raises(ValueError, match="first_moment_decay"):
        phx.optim.riemannian_adam(geometry, first_moment_decay=1.0)
    with pytest.raises(ValueError, match="second_moment_decay"):
        phx.optim.riemannian_adam(geometry, second_moment_decay=jnp.nan)
    with pytest.raises(ValueError, match="epsilon"):
        phx.optim.riemannian_adam(geometry, epsilon=0.0)
    with pytest.raises(TypeError, match="amsgrad"):
        phx.optim.riemannian_adam(geometry, amsgrad=invalid_amsgrad)

    optimizer = phx.optim.riemannian_adam(geometry)
    wrong_state = phx.optim.riemannian_sgd(geometry).init(parameters)
    invalid_state: Any = wrong_state
    with pytest.raises(TypeError, match="RiemannianAdamState"):
        optimizer.update(
            {"point": jnp.zeros((3,))},
            invalid_state,
            parameters,
        )
