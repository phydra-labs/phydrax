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
    assert bool(metrics.restarted) == bool(state.restarted)


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
    assert bool(metrics.restarted) == bool(state.restarted)
    assert bool(metrics.pair_accepted) == bool(state.pair_accepted)
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


def test_rejected_conjugate_gradient_step_preserves_point_and_forces_restart():
    point = jnp.array([1.0, 0.0, 0.0])
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        point,
        {"<root>": phx.metrix.SphereManifold(3)},
    )
    optimizer = phx.optim.riemannian_conjugate_gradient(
        geometry,
        line_search=phx.optim.ArmijoLineSearch(maximum_steps=1),
    )
    state = optimizer.init(point)

    updated, state = optimizer.update(
        jnp.array([0.0, 1.0, 0.0]),
        state,
        point,
        value=jnp.asarray(0.0),
        value_fn=lambda candidate: jnp.asarray(1.0),
    )

    assert jnp.array_equal(updated, point)
    assert not bool(state.line_search_accepted)
    assert bool(state.restarted)
    assert jnp.array_equal(state.previous_direction, jnp.zeros_like(point))

    _, next_state = optimizer.update(
        jnp.array([0.0, 1.0, 0.0]),
        state,
        point,
        value=jnp.asarray(0.0),
        value_fn=lambda candidate: jnp.asarray(1.0),
    )
    assert bool(next_state.restarted)


def test_rejected_lbfgs_step_preserves_point_and_active_history():
    target = jnp.array([0.0, 1.0, 0.0])
    point = jnp.array([1.0, 0.0, 0.0])
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        point,
        {"<root>": phx.metrix.SphereManifold(3)},
    )
    optimizer = phx.optim.riemannian_lbfgs(
        geometry,
        history_size=3,
        line_search=phx.optim.ArmijoLineSearch(maximum_steps=1),
    )
    objective = lambda candidate: 1.0 - jnp.dot(candidate, target)
    value, gradient = jax.value_and_grad(objective)(point)
    point, state = optimizer.update(
        gradient,
        optimizer.init(point),
        point,
        value=value,
        value_fn=objective,
    )
    history_before = state.s_history
    gradient_history_before = state.y_history
    rho_before = state.rho
    active_before = state.active

    updated, rejected = optimizer.update(
        jnp.array([0.0, 0.0, 1.0]),
        state,
        point,
        value=jnp.asarray(0.0),
        value_fn=lambda candidate: jnp.asarray(1.0),
    )

    assert jnp.array_equal(updated, point)
    assert not bool(rejected.line_search_accepted)
    assert not bool(rejected.pair_accepted)
    assert jnp.array_equal(rejected.s_history, history_before)
    assert jnp.array_equal(rejected.y_history, gradient_history_before)
    assert jnp.array_equal(rejected.rho, rho_before)
    assert jnp.array_equal(rejected.active, active_before)
