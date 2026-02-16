#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import cast, Literal

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax._frozendict import frozendict
from phydrax.constraints import FunctionalConstraint
from phydrax.domain import (
    CoordSeparableBatch,
    Interval1d,
    PointsBatch,
    ProductStructure,
    Square,
)


def test_functional_constraint_mean_and_integral_reductions():
    geom = Interval1d(0.0, 2.0)
    component = geom.component()
    structure = ProductStructure((("x",),))

    u = geom.Function()(0.0)

    def operator(u_fn):
        return u_fn - 1.0

    c_mean = FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars="u",
        num_points=16,
        structure=structure,
        reduction="mean",
        weight=3.0,
    )
    loss_mean = c_mean.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_mean, 3.0)

    c_int = FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars="u",
        num_points=16,
        structure=structure,
        reduction="integral",
        weight=3.0,
    )
    loss_int = c_int.loss({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss_int, 6.0)


def test_functional_constraint_domainfunction_weight():
    geom = Interval1d(0.0, 2.0)
    component = geom.component()
    structure = ProductStructure((("x",),))

    u = geom.Function()(0.0)

    @geom.Function("x")
    def w(x):
        return x[0] + 1.0

    def operator(u_fn):
        return u_fn - 1.0

    c_mean = FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars="u",
        num_points=4096,
        structure=structure,
        reduction="mean",
        weight=w,
    )
    loss_mean = c_mean.loss({"u": u}, key=jr.key(0))
    # E[x + 1] on [0, 2] equals 2.
    assert jnp.allclose(loss_mean, 2.0, rtol=5e-2, atol=5e-2)

    c_int = FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars="u",
        num_points=4096,
        structure=structure,
        reduction="integral",
        weight=w,
    )
    loss_int = c_int.loss({"u": u}, key=jr.key(1))
    # Integral of (x + 1) on [0, 2] equals 4.
    assert jnp.allclose(loss_int, 4.0, rtol=5e-2, atol=5e-2)


def test_functional_constraint_resample_sampling_changes_points():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def x_fn(x):
        return x[0]

    constraint = FunctionalConstraint.from_operator(
        component=component,
        operator=lambda _u: x_fn,
        constraint_vars="u",
        num_points=12,
        structure=structure,
        reduction="mean",
        sampling_mode="resample",
    )

    batch0 = constraint.sample(key=jr.key(0))
    batch1 = constraint.sample(key=jr.key(1))
    assert isinstance(batch0, PointsBatch)
    assert isinstance(batch1, PointsBatch)
    x0 = jnp.asarray(batch0.points["x"].data)
    x1 = jnp.asarray(batch1.points["x"].data)
    assert not jnp.allclose(x0, x1)


def test_functional_constraint_fixed_sampling_reuses_batch_and_honors_override():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    structure = ProductStructure((("x",),))
    u = geom.Function()(0.0)

    @geom.Function("x")
    def x_fn(x):
        return x[0]

    constraint = FunctionalConstraint.from_operator(
        component=component,
        operator=lambda _u: x_fn,
        constraint_vars="u",
        num_points=12,
        structure=structure,
        reduction="mean",
        sampling_mode="fixed",
        fixed_batch_key=jr.key(123),
    )

    batch0 = constraint.sample(key=jr.key(0))
    batch1 = constraint.sample(key=jr.key(1))
    assert isinstance(batch0, PointsBatch)
    assert isinstance(batch1, PointsBatch)
    x0 = jnp.asarray(batch0.points["x"].data)
    x1 = jnp.asarray(batch1.points["x"].data)
    assert jnp.allclose(x0, x1)

    loss0 = constraint.loss({"u": u}, key=jr.key(2))
    loss1 = constraint.loss({"u": u}, key=jr.key(3))
    assert jnp.allclose(loss0, loss1)
    assert float(loss0) > 0.0

    x_field = batch0.points["x"]
    zeros_batch = PointsBatch(
        points=frozendict(
            {"x": cx.Field(jnp.zeros_like(x_field.data), dims=x_field.dims)}
        ),
        structure=batch0.structure,
    )
    override_loss = constraint.loss({"u": u}, key=jr.key(4), batch=zeros_batch)
    assert jnp.allclose(override_loss, 0.0, atol=1e-12)
    assert not jnp.allclose(override_loss, loss0)


def test_functional_constraint_fixed_sampling_coord_separable():
    geom = Square(center=(0.0, 0.0), side=2.0)
    component = geom.component()
    structure = ProductStructure((("x",),))
    u = geom.Function()(0.0)

    @geom.Function("x")
    def x0_fn(x):
        x0, _ = x
        return x0

    constraint = FunctionalConstraint.from_operator(
        component=component,
        operator=lambda _u: x0_fn,
        constraint_vars="u",
        num_points={"x": (7, 6)},
        structure=structure,
        reduction="mean",
        sampling_mode="fixed",
        fixed_batch_key=jr.key(7),
    )

    batch0 = constraint.sample(key=jr.key(8))
    batch1 = constraint.sample(key=jr.key(9))
    assert isinstance(batch0, CoordSeparableBatch)
    assert isinstance(batch1, CoordSeparableBatch)
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

    loss0 = constraint.loss({"u": u}, key=jr.key(10))
    loss1 = constraint.loss({"u": u}, key=jr.key(11))
    assert jnp.allclose(loss0, loss1)


def test_functional_constraint_sampling_mode_validation():
    geom = Interval1d(0.0, 1.0)
    component = geom.component()
    structure = ProductStructure((("x",),))

    @geom.Function("x")
    def x_fn(x):
        return x[0]

    with pytest.raises(ValueError, match="sampling_mode"):
        bad_mode = cast(Literal["resample", "fixed"], "bad-mode")
        FunctionalConstraint.from_operator(
            component=component,
            operator=lambda _u: x_fn,
            constraint_vars="u",
            num_points=8,
            structure=structure,
            sampling_mode=bad_mode,
        )

    batch = component.sample(8, structure=structure, key=jr.key(9))
    with pytest.raises(ValueError, match="fixed_batch"):
        FunctionalConstraint.from_operator(
            component=component,
            operator=lambda _u: x_fn,
            constraint_vars="u",
            num_points=8,
            structure=structure,
            sampling_mode="resample",
            fixed_batch=batch,
        )
