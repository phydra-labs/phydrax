#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.conditions import Residual
from phydrax.domain import GridBatch, Interval1d, PointBatch
from phydrax.terms import ResidualPenalty


def _batch(realization):
    return realization.batch.points


def test_residual_penalty_mean_and_integral_reductions():
    geom = Interval1d(0.0, 2.0)
    component = geom.component()
    u = geom.Function()(0.0)
    condition = Residual("u", component, lambda u_fn: u_fn - 1.0)
    plan = phx.integration.MonteCarloPlan(16)

    mean_term = ResidualPenalty(
        condition,
        phx.integration.per_step(phx.integration.mean_over(condition.on), plan),
        scale=3.0,
    )
    loss_mean = mean_term.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_mean, 3.0)

    integral_term = ResidualPenalty(
        condition,
        phx.integration.per_step(phx.integration.over(condition.on), plan),
        scale=3.0,
    )
    loss_int = integral_term.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_int, 6.0)


def test_residual_penalty_domainfunction_weight():
    geom = Interval1d(0.0, 2.0)
    component = geom.component()
    u = geom.Function()(0.0)

    @geom.Function("x")
    def density(x):
        return x[0] + 1.0

    condition = Residual("u", component, lambda u_fn: u_fn - 1.0)
    plan = phx.integration.MonteCarloPlan(4096)
    mean_term = ResidualPenalty(
        condition,
        phx.integration.per_step(phx.integration.mean_over(condition.on), plan),
        density=density,
    )
    loss_mean = mean_term.loss({"u": u}, key=jr.key(0))
    # E[x + 1] on [0, 2] equals 2.
    assert jnp.allclose(loss_mean, 2.0, rtol=5e-2, atol=5e-2)

    integral_term = ResidualPenalty(
        condition,
        phx.integration.per_step(phx.integration.over(condition.on), plan),
        density=density,
    )
    loss_int = integral_term.loss({"u": u}, key=jr.key(1))
    # Integral of (x + 1) on [0, 2] equals 4.
    assert jnp.allclose(loss_int, 4.0, rtol=5e-2, atol=5e-2)


def test_residual_penalty_resample_sampling_changes_points():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    @geom.Function("x")
    def x_fn(x):
        return x[0]

    condition = Residual("u", component, lambda _u: x_fn)
    source = phx.integration.per_step(
        phx.integration.mean_over(condition.on),
        phx.integration.MonteCarloPlan(12),
    )
    ResidualPenalty(condition, source)

    batch0 = _batch(
        phx.integration.materialize(source.target, source.plan, key=jr.key(0))
    )
    batch1 = _batch(
        phx.integration.materialize(source.target, source.plan, key=jr.key(1))
    )
    assert isinstance(batch0, PointBatch)
    assert isinstance(batch1, PointBatch)
    x0 = jnp.asarray(batch0.points["x"].data)
    x1 = jnp.asarray(batch1.points["x"].data)
    assert not jnp.allclose(x0, x1)


def test_residual_penalty_fixed_sampling_reuses_batch_and_honors_override():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    u = geom.Function()(0.0)

    @geom.Function("x")
    def x_fn(x):
        return x[0]

    condition = Residual("u", component, lambda _u: x_fn)
    target = phx.integration.mean_over(condition.on)
    realization = phx.integration.materialize(
        target,
        phx.integration.MonteCarloPlan(12),
        key=jr.key(123),
    )
    fixed_term = ResidualPenalty(condition, phx.integration.fixed(realization))

    batch0 = _batch(realization)
    batch1 = _batch(fixed_term.source.realization)
    assert isinstance(batch0, PointBatch)
    assert isinstance(batch1, PointBatch)
    x0 = jnp.asarray(batch0.points["x"].data)
    x1 = jnp.asarray(batch1.points["x"].data)
    assert jnp.allclose(x0, x1)

    loss0 = fixed_term.loss({"u": u}, key=jr.key(2))
    loss1 = fixed_term.loss({"u": u}, key=jr.key(3))
    assert jnp.allclose(loss0, loss1)
    assert float(loss0) > 0.0

    x_field = batch0.points["x"]
    zeros_batch = component.points({"x": jnp.zeros_like(x_field.data).reshape((-1, 1))})
    override = phx.integration.from_samples(target, zeros_batch, key=jr.key(4))
    caller_term = ResidualPenalty(condition, phx.integration.caller(target))
    override_loss = caller_term.loss({"u": u}, realization=override)
    assert jnp.allclose(override_loss, 0.0, atol=1e-12)
    assert not jnp.allclose(override_loss, loss0)


def test_residual_penalty_fixed_sampling_coord_separable():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    component = geom.component()
    u = geom.Function()(0.0)

    @geom.Function("x")
    def x0_fn(x):
        x0, _ = x
        return x0

    condition = Residual("u", component, lambda _u: x0_fn)
    target = phx.integration.mean_over(condition.on)
    sampled = component.sample(
        phx.domain.GridSampling({"x": (7, 6)}),
        key=jr.key(7),
    )
    realization = phx.integration.from_samples(target, sampled)
    term = ResidualPenalty(condition, phx.integration.fixed(realization))

    batch0 = _batch(realization)
    batch1 = _batch(term.source.realization)
    assert isinstance(batch0, GridBatch)
    assert isinstance(batch1, GridBatch)
    assert isinstance(batch0.points["x"], tuple)
    assert isinstance(batch1.points["x"], tuple)
    assert len(batch0.points["x"]) == 2
    assert jnp.allclose(
        jnp.asarray(batch0.points["x"][0].data),
        jnp.asarray(batch1.points["x"][0].data),
    )
    assert jnp.allclose(
        jnp.asarray(batch0.points["x"][1].data),
        jnp.asarray(batch1.points["x"][1].data),
    )

    loss0 = term.loss({"u": u}, key=jr.key(10))
    loss1 = term.loss({"u": u}, key=jr.key(11))
    assert jnp.allclose(loss0, loss1)


def test_residual_penalty_accepts_domainfunction_density():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    component = geom.component()
    one = geom.Function()(1.0)
    condition = Residual("u", component, lambda _u: one)
    sampled = component.sample(
        phx.domain.GridSampling({"x": (7, 6)}),
        key=jr.key(12),
    )
    realization = phx.integration.from_samples(
        phx.integration.mean_over(condition.on),
        sampled,
    )
    term = ResidualPenalty(
        condition,
        phx.integration.fixed(realization),
        density=geom.Function()(2.0),
    )
    loss = term.loss({"u": geom.Function()(0.0)}, key=jr.key(13))
    assert jnp.allclose(loss, 2.0)


def test_residual_penalty_rejects_untyped_density():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    condition = Residual("u", component, lambda _u: geom.Function()(1.0))
    source = phx.integration.per_step(
        phx.integration.mean_over(condition.on),
        phx.integration.MonteCarloPlan(8),
    )
    with pytest.raises(TypeError, match="DomainFunction"):
        ResidualPenalty(condition, source, density=jnp.ones((4,)))


def test_residual_penalty_source_validation():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()

    @geom.Function("x")
    def x_fn(x):
        return x[0]

    condition = Residual("u", component, lambda _u: x_fn)
    target = phx.integration.mean_over(condition.on)
    plan = phx.integration.MonteCarloPlan(8)
    with pytest.raises(TypeError, match="typed IntegrationSource"):
        ResidualPenalty(condition, plan)

    realization = phx.integration.materialize(target, plan, key=jr.key(9))
    fixed_term = ResidualPenalty(condition, phx.integration.fixed(realization))
    with pytest.raises(ValueError, match="does not accept"):
        fixed_term.loss(
            {"u": geom.Function()(0.0)},
            realization=realization,
        )


@pytest.mark.parametrize("reduction", ["mean", "integral"])
def test_quadratic_residual_data_reconstructs_weighted_loss(reduction):
    geom = Interval1d(0.0, 2.0)
    component = geom.component()
    structure = phx.domain.SampleLayout((("x",),))

    @geom.Function("x")
    def u(x):
        return x[0] - 0.25

    @geom.Function("x")
    def density(x):
        return 1.0 + x[0]

    condition = Residual("u", component, lambda field: field + 0.5)
    target = (
        phx.integration.mean_over(component)
        if reduction == "mean"
        else phx.integration.over(component)
    )
    sampled = component.sample(
        phx.domain.PointSampling(31, layout=structure),
        key=jr.key(70),
    )
    realization = phx.integration.from_samples(target, sampled, key=jr.key(71))
    term = ResidualPenalty(
        condition,
        phx.integration.caller(target),
        scale=2.5,
        density=density,
    )
    data = term._quadratic_residual_data(
        {"u": u},
        realization=realization,
    )
    expected = term.loss({"u": u}, realization=realization)

    assert len(data.residuals) == 1
    assert len(data.coefficients) == 1
    assert all(dim is not None for dim in data.coefficients[0].dims)
    assert jnp.all(data.coefficients[0].data >= 0.0)
    assert jnp.allclose(data.loss, expected, rtol=1e-12, atol=1e-12)


def test_quadratic_residual_data_reconstructs_component_sum_loss():
    geom = Interval1d(0.0, 1.0)
    left = geom.component(where={"x": lambda point: point[0] < 0.5})
    right = geom.component(where={"x": lambda point: point[0] >= 0.5})
    component = phx.domain.ComponentSum((left, right), assume_disjoint=True)
    structure = phx.domain.SampleLayout((("x",),))

    @geom.Function("x")
    def u(x):
        return 1.0 + x[0]

    condition = Residual("u", component, lambda field: field)
    target = phx.integration.mean_over(component)
    sampled = component.sample(
        phx.domain.PointSampling(17, layout=structure),
        key=jr.key(72),
    )
    realization = phx.integration.from_samples(target, sampled, key=jr.key(73))
    term = ResidualPenalty(condition, phx.integration.caller(target))
    data = term._quadratic_residual_data(
        {"u": u},
        realization=realization,
    )
    expected = term.loss({"u": u}, realization=realization)

    assert len(data.residuals) == 2
    assert len(data.coefficients) == 2
    assert jnp.allclose(data.loss, expected, rtol=1e-12, atol=1e-12)


def test_quadratic_residual_data_includes_adaptive_batch_weights():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    component = geom.component()
    condition = Residual("u", component, lambda field: field)
    policy = phx.sampling.collocation.PeriodicCollocation(
        refresh_every=1,
        sampler="uniform",
    )
    target = phx.integration.mean_over(component)
    term = ResidualPenalty(
        condition,
        phx.integration.adaptive(
            target,
            phx.domain.GridSampling({"x": (5, 4)}),
            policy,
        ),
    )
    batch = term.sample(key=jr.key(74))
    assert isinstance(batch, GridBatch)
    axis = batch.coord_axes_by_label["x"][0]
    size = batch.points["x"][0].data.shape[0]
    local_weight = cx.Field(jnp.linspace(0.5, 1.5, size), dims=(axis,))
    realization = term._adaptive_realization(
        batch,
        local_weight,
        key=jr.key(75),
    )
    functions = {"u": geom.Function()(2.0)}
    data = term._quadratic_residual_data(
        functions,
        realization=realization,
    )
    expected = term.loss(functions, realization=realization)

    assert jnp.allclose(data.loss, expected, rtol=1e-12, atol=1e-12)
