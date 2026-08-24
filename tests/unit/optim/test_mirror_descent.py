#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _negative_entropy_geometry(dimension):
    primal_chart = phx.metrix.CoordinateChart(
        f"mirror-positive-{dimension}",
        tuple(f"x{index}" for index in range(dimension)),
    )
    dual_chart = phx.metrix.CoordinateChart(
        f"mirror-log-{dimension}",
        tuple(f"y{index}" for index in range(dimension)),
    )
    return phx.metrix.LegendreGeometry(
        phx.metrix.HessianGeometry(
            lambda point: jnp.sum(point * (jnp.log(point) - 1.0)),
            chart=primal_chart,
        ),
        jnp.exp,
        primal_support=phx.metrix.ChartSupport(
            primal_chart,
            lambda point: jnp.all(point > 0.0, axis=-1),
            support_id=f"mirror-positive-support-{dimension}",
        ),
        dual_support=phx.metrix.ChartSupport(
            dual_chart,
            lambda point: jnp.all(jnp.isfinite(point), axis=-1),
            support_id=f"mirror-log-support-{dimension}",
        ),
        geometry_id=f"negative-entropy-{dimension}",
    )


def _path_containing(parameters, name):
    return next(
        path
        for path in phx.optim.ParameterMirrorGeometry.array_leaf_paths(parameters)
        if name in path
    )


def test_unit_mirror_step_solves_negative_entropy_divergence_exactly():
    initial = jnp.asarray([0.3, 1.4, 2.2])
    target = jnp.asarray([0.8, 0.5, 1.7])
    parameters = {"positive": initial}
    path = _path_containing(parameters, "positive")
    parameter_geometry = phx.optim.ParameterMirrorGeometry(
        parameters,
        {path: _negative_entropy_geometry(3)},
    )
    optimizer = phx.optim.mirror_descent(parameter_geometry, learning_rate=1.0)
    state = optimizer.init(parameters)

    def objective(tree):
        value = tree["positive"]
        return jnp.sum(value * jnp.log(value / target) - value + target)

    gradients = jax.grad(objective)(parameters)
    destination, state = optimizer.update(gradients, state, parameters)

    assert jnp.allclose(destination["positive"], target, rtol=1e-6, atol=1e-6)
    assert int(state.step) == 1
    assert state.metrics.bregman_step > 0.0
    assert state.metrics.constraint_residual == 0.0


def test_mirror_descent_updates_mixed_weighted_and_product_leaves():
    parameters = {
        "positive": jnp.asarray([[0.4, 0.9], [1.3, 0.6]]),
        "euclidean": jnp.asarray(1.5),
    }
    gradients = {
        "positive": jnp.asarray([[0.5, -0.2], [0.1, 0.7]]),
        "euclidean": jnp.asarray(2.0),
    }
    path = _path_containing(parameters, "positive")
    geometry = phx.optim.ParameterMirrorGeometry(
        parameters,
        {path: _negative_entropy_geometry(2)},
        weights={path: 2.0},
    )
    optimizer = phx.optim.mirror_descent(geometry, learning_rate=0.2)

    destination, state = optimizer.update(
        gradients,
        optimizer.init(parameters),
        parameters,
    )
    expected_positive = parameters["positive"] * jnp.exp(
        -0.2 * gradients["positive"] / 2.0
    )
    assert jnp.allclose(destination["positive"], expected_positive)
    assert jnp.allclose(destination["euclidean"], 1.1)
    assert bool(geometry.contains(destination))
    assert geometry.num_legendre_leaves == 1
    assert geometry.geometry_ids == ("negative-entropy-2",)
    assert state.metrics.coordinate_gradient_norm > 0.0
    assert state.metrics.dual_displacement_norm > 0.0


def test_mirror_descent_schedule_is_jittable_and_zero_rate_is_noop():
    parameters = {"positive": jnp.asarray([0.6, 1.2])}
    gradients = {"positive": jnp.asarray([0.4, -0.3])}
    path = _path_containing(parameters, "positive")
    geometry = phx.optim.ParameterMirrorGeometry(
        parameters,
        {path: _negative_entropy_geometry(2)},
    )
    optimizer = phx.optim.mirror_descent(
        geometry,
        learning_rate=lambda step: jnp.where(step == 0, 0.0, 0.1),
    )
    update = eqx.filter_jit(optimizer.update)
    first, state = update(gradients, optimizer.init(parameters), parameters)
    second, state = update(gradients, state, first)

    assert jnp.allclose(first["positive"], parameters["positive"])
    assert jnp.allclose(
        second["positive"],
        parameters["positive"] * jnp.exp(-0.1 * gradients["positive"]),
    )
    assert int(state.step) == 2


def test_parameter_mirror_geometry_rejects_invalid_bindings_and_updates():
    parameters = {"positive": jnp.asarray([0.5, 1.0])}
    geometry = _negative_entropy_geometry(2)
    with pytest.raises(ValueError, match="Unknown ParameterMirrorGeometry leaf paths"):
        phx.optim.ParameterMirrorGeometry(parameters, {"missing": geometry})

    path = _path_containing(parameters, "positive")
    with pytest.raises(TypeError, match="must be bound to a LegendreGeometry"):
        phx.optim.ParameterMirrorGeometry(parameters, {path: None})
    with pytest.raises(ValueError, match="finite and positive"):
        phx.optim.ParameterMirrorGeometry(
            parameters,
            {path: geometry},
            weights={path: 0.0},
        )
    with pytest.raises(ValueError, match="outside"):
        phx.optim.ParameterMirrorGeometry(
            {"positive": jnp.asarray([0.5, 0.0])},
            {path: geometry},
        )

    binding = phx.optim.ParameterMirrorGeometry(parameters, {path: geometry})
    optimizer = phx.optim.mirror_descent(binding)
    state = optimizer.init(parameters)
    with pytest.raises(TypeError, match="MirrorDescentState"):
        optimizer.update(
            {"positive": jnp.asarray([0.1, 0.2])},
            object(),
            parameters,
        )
    with pytest.raises(ValueError, match="incompatible PyTree structure"):
        optimizer.update(
            {"other": jnp.asarray([0.1, 0.2])},
            state,
            parameters,
        )


def test_parameter_mirror_geometry_reports_infinite_invalid_residual():
    parameters = {"positive": jnp.asarray([0.5, 1.0])}
    path = _path_containing(parameters, "positive")
    binding = phx.optim.ParameterMirrorGeometry(
        parameters,
        {path: _negative_entropy_geometry(2)},
    )
    invalid = {"positive": jnp.asarray([0.5, 0.0])}
    assert not bool(binding.contains(invalid))
    assert jnp.isinf(binding.maximum_constraint_residual(invalid))
