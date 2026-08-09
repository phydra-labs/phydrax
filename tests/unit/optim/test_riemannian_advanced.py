#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_weighted_product_geometry_controls_global_metric_without_changing_residuals():
    parameters = {
        "euclidean": jnp.array([1.0, -1.0]),
        "sphere": jnp.array([1.0, 0.0, 0.0]),
    }
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {"['sphere']": phx.metrix.SphereManifold(3)},
        weights={"['sphere']": 3.0},
    )
    tangent = {
        "euclidean": jnp.array([1.0, 2.0]),
        "sphere": jnp.array([0.0, 1.0, 0.0]),
    }

    assert jnp.allclose(geometry.inner(parameters, tangent, tangent), 8.0)
    assert jnp.allclose(geometry.norm(parameters, tangent), jnp.sqrt(8.0))
    assert geometry.maximum_constraint_residual(parameters) == 0.0


def _minimize_sphere(optimizer_factory):
    target = jnp.array([0.0, 1.0, 0.0])
    point = jnp.array([1.0, 0.0, 0.0])
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        point,
        {"<root>": phx.metrix.SphereManifold(3)},
    )
    optimizer = optimizer_factory(geometry)
    state = optimizer.init(point)
    value_and_grad = jax.value_and_grad(lambda value: 1.0 - jnp.dot(value, target))
    accepted = False

    for _ in range(12):
        value, gradient = value_and_grad(point)
        point, state = optimizer.update(
            gradient,
            state,
            point,
            value=value,
            value_fn=lambda candidate: 1.0 - jnp.dot(candidate, target),
        )
        accepted = accepted or bool(state.line_search_accepted)
    return point, state, geometry, accepted


def test_riemannian_conjugate_gradient_uses_armijo_and_converges_on_sphere():
    point, state, geometry, accepted = _minimize_sphere(
        lambda geometry: phx.optim.riemannian_conjugate_gradient(geometry)
    )

    assert jnp.dot(point, jnp.array([0.0, 1.0, 0.0])) > 0.999
    assert bool(geometry.contains(point))
    assert accepted
    assert state.line_search_evaluations >= 1
    metrics = state.metrics
    assert metrics.line_search_evaluations == state.line_search_evaluations
    assert bool(metrics.line_search_accepted) == bool(state.line_search_accepted)
    assert metrics.line_search_reduction >= 0.0
    assert metrics.conjugacy_beta == state.beta


def test_riemannian_lbfgs_transports_bounded_history_and_converges_on_sphere():
    point, state, geometry, accepted = _minimize_sphere(
        lambda geometry: phx.optim.riemannian_lbfgs(geometry, history_size=4)
    )

    assert jnp.dot(point, jnp.array([0.0, 1.0, 0.0])) > 0.999
    assert bool(geometry.contains(point))
    assert state.count <= 4
    assert accepted
    metrics = state.metrics
    assert metrics.line_search_evaluations == state.line_search_evaluations
    assert bool(metrics.line_search_accepted) == bool(state.line_search_accepted)
    assert metrics.line_search_reduction >= 0.0
    assert metrics.history_pair_count == state.count
    for index in range(4):
        if bool(state.active[index]):
            s_value = jax.tree.map(lambda leaf: leaf[index], state.s_history)
            y_value = jax.tree.map(lambda leaf: leaf[index], state.y_history)
            expected_rho = 1.0 / geometry.inner(
                point,
                s_value,
                y_value,
            )
            assert jnp.allclose(state.rho[index], expected_rho)


@pytest.mark.parametrize(
    "optimizer_factory",
    (
        phx.optim.riemannian_conjugate_gradient,
        phx.optim.riemannian_lbfgs,
    ),
)
def test_advanced_optimizers_reject_nonfinite_gradient_norms(optimizer_factory):
    point = jnp.array([1.0, 0.0, 0.0])
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        point,
        {"<root>": phx.metrix.SphereManifold(3)},
    )
    optimizer = optimizer_factory(geometry)

    with pytest.raises(Exception, match="norm is not finite"):
        optimizer.update(
            jnp.array([jnp.nan, 0.0, 0.0]),
            optimizer.init(point),
            point,
            value=jnp.asarray(0.0),
            value_fn=lambda candidate: jnp.sum(candidate**2),
        )


def test_armijo_rejects_a_nonfinite_frozen_objective_value():
    point = jnp.array([1.0, 0.0, 0.0])
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        point,
        {"<root>": phx.metrix.SphereManifold(3)},
    )
    optimizer = phx.optim.riemannian_conjugate_gradient(geometry)

    with pytest.raises(Exception, match="initial value must be finite"):
        optimizer.update(
            jnp.array([0.0, 1.0, 0.0]),
            optimizer.init(point),
            point,
            value=jnp.asarray(jnp.nan),
            value_fn=lambda candidate: jnp.sum(candidate**2),
        )
