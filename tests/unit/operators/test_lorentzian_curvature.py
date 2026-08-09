#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


class _TrainableFlatFLRW(eqx.Module):
    expansion_rate: jax.Array

    def __call__(self, coordinates):
        scale_squared = jnp.exp(2.0 * self.expansion_rate * coordinates[0])
        return jnp.diag(
            jnp.stack(
                (
                    jnp.asarray(-1.0, dtype=coordinates.dtype),
                    scale_squared,
                    scale_squared,
                    scale_squared,
                )
            )
        )


def _spacetime():
    domain = phx.domain.HyperRectangle([-1.0] * 4, [1.0] * 4, label="x")
    chart = phx.metrix.CoordinateChart("spacetime", ("t", "x", "y", "z"))
    return domain, chart


def test_domain_curvature_adapters_preserve_labeled_jittable_semantics():
    domain, chart = _spacetime()
    metric = phx.metrix.minkowski_metric(chart)
    point = jnp.array([0.2, -0.1, 0.3, 0.4])
    points = jnp.stack((point, -point))

    riemann = phx.operators.domain_riemann_tensor(domain, metric, var="x")
    ricci = phx.operators.domain_ricci_tensor(domain, metric, var="x")
    scalar = phx.operators.domain_scalar_curvature(domain, metric, var="x")
    einstein = phx.operators.domain_einstein_tensor(domain, metric, var="x")

    assert riemann.domain is domain
    assert riemann.deps == ("x",)
    assert ricci.deps == scalar.deps == einstein.deps == ("x",)
    assert riemann.func(point).shape == (4, 4, 4, 4)
    assert ricci.func(point).shape == (4, 4)
    assert scalar.func(point).shape == ()
    assert einstein.func(point).shape == (4, 4)
    assert jnp.allclose(jax.jit(riemann.func)(point), 0.0)
    assert jnp.allclose(jax.jit(ricci.func)(points), 0.0)
    assert jnp.allclose(jax.jit(scalar.func)(points), 0.0)
    assert jnp.allclose(jax.jit(einstein.func)(points), 0.0)


def test_domain_curvature_is_differentiable_through_trainable_metric_fields():
    domain, chart = _spacetime()
    point = jnp.array([0.3, 0.0, 0.0, 0.0])

    def scalar_from_rate(expansion_rate):
        metric = phx.metrix.LorentzianMetric(
            _TrainableFlatFLRW(expansion_rate),
            chart=chart,
        )
        return phx.operators.domain_scalar_curvature(
            domain,
            metric,
            var="x",
        ).func(point)

    expansion_rate = jnp.asarray(0.2)
    scalar, derivative = jax.jit(jax.value_and_grad(scalar_from_rate))(
        expansion_rate
    )
    metric = phx.metrix.LorentzianMetric(
        _TrainableFlatFLRW(expansion_rate),
        chart=chart,
    )
    einstein = phx.operators.domain_einstein_tensor(
        domain,
        metric,
        var="x",
    ).func(point)
    scale_squared = jnp.exp(2.0 * expansion_rate * point[0])

    assert jnp.allclose(scalar, 12.0 * expansion_rate**2)
    assert jnp.allclose(derivative, 24.0 * expansion_rate)
    assert jnp.allclose(einstein[0, 0], 3.0 * expansion_rate**2)
    assert jnp.allclose(
        jnp.diag(einstein)[1:],
        -3.0 * expansion_rate**2 * scale_squared,
    )


def test_domain_curvature_rejects_incompatible_geometry_contracts():
    domain, chart = _spacetime()
    metric = phx.metrix.minkowski_metric(chart)
    line = phx.domain.Interval1d(-1.0, 1.0)
    riemannian = phx.metrix.diagonal_metric(
        lambda q: jnp.ones((4,)),
        chart=chart,
    )

    with pytest.raises(TypeError, match="require a Domain"):
        phx.operators.domain_scalar_curvature(object(), metric, var="x")
    with pytest.raises(TypeError, match="requires a LorentzianMetric"):
        phx.operators.domain_scalar_curvature(domain, riemannian, var="x")
    with pytest.raises(ValueError, match="does not match"):
        phx.operators.domain_scalar_curvature(line, metric)
