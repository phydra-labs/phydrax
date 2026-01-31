#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import cast

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from phydrax.domain import (
    DomainFunction,
    Interval1d,
    ProductStructure,
    Square,
    TimeInterval,
)
from phydrax.nn.models import LatentContractionModel
from phydrax.nn.models.core._base import _AbstractBaseModel
from phydrax.operators.differential import partial_t, partial_x, partial_y
from phydrax.operators.integral import integral


def _as_scalar(x):
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return arr
    if arr.size != 1:
        raise ValueError("Expected scalar input for scalar factor model.")
    return arr.reshape(())


class XYLatentModel(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str

    def __init__(self) -> None:
        self.in_size = 2
        self.out_size = 2

    def __call__(self, x, /, *, key=jr.key(0)):
        x = jnp.asarray(x)
        return jnp.stack([x[0] + x[1], jnp.array(1.0)], axis=-1)


class ScalarLatentModel(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str

    def __init__(self) -> None:
        self.in_size = "scalar"
        self.out_size = 2

    def __call__(self, x, /, *, key=jr.key(0)):
        x = _as_scalar(x)
        return jnp.stack([x, jnp.array(1.0)], axis=-1)


class ConstantXYLatentModel(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str

    def __init__(self) -> None:
        self.in_size = 2
        self.out_size = 1

    def __call__(self, x, /, *, key=jr.key(0)):
        return jnp.array([1.0])


class ScalarOffsetLatentModel(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str
    offset: float

    def __init__(self, offset: float) -> None:
        self.in_size = "scalar"
        self.out_size = 1
        self.offset = float(offset)

    def __call__(self, x, /, *, key=jr.key(0)):
        x = _as_scalar(x)
        return jnp.array([x + self.offset])


def _latent_function(domain, model):
    def f(x, p, t, *, key=jr.key(0)):
        return model({"x": x, "p": p, "t": t}, key=key)

    return DomainFunction(domain=domain, deps=("x", "p", "t"), func=f)


def _squeeze_field_for_compare(
    field: cx.Field,
) -> tuple[jnp.ndarray, tuple[str | None, ...]]:
    data = jnp.asarray(field.data)
    dims = list(field.dims)
    squeeze_axes = [
        i
        for i, (d, n) in enumerate(zip(dims, data.shape, strict=True))
        if d is None and n == 1
    ]
    if squeeze_axes:
        data = jnp.squeeze(data, axis=tuple(squeeze_axes))
        dims = [d for i, d in enumerate(dims) if i not in squeeze_axes]
    return data, tuple(dims)


def _assert_field_allclose(field, expected, *, atol=1e-6):
    lhs, lhs_dims = _squeeze_field_for_compare(field)
    rhs, rhs_dims = _squeeze_field_for_compare(expected)
    assert lhs_dims == rhs_dims
    assert jnp.allclose(lhs, rhs, atol=atol)


def _assert_array_allclose(actual, expected, *, atol=1e-6):
    a = jnp.asarray(actual)
    b = jnp.asarray(expected)
    if a.ndim > 0 and a.shape[-1] == 1:
        a = a[..., 0]
    if b.ndim > 0 and b.shape[-1] == 1:
        b = b[..., 0]
    assert jnp.allclose(a, b, atol=atol)


def _scalar_field_from_dense(field: cx.Field, axis: str) -> cx.Field:
    data = jnp.asarray(field.data)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 1:
        raise ValueError(
            f"Expected dense scalar field with shape (N,) or (N,1); got {data.shape}."
        )
    return cx.Field(data, dims=(axis,))


def test_latent_contraction_product_domain_partials():
    geom = Square(center=(0.0, 0.0), side=2.0)
    param = Interval1d(-1.0, 1.0).relabel("p")
    time = TimeInterval(0.0, 1.0)
    domain = geom @ param @ time

    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=XYLatentModel(),
        p=ScalarLatentModel(),
        t=ScalarLatentModel(),
    )
    u = _latent_function(domain, model)

    component = domain.component()
    sep = component.sample_coord_separable(
        {"x": (4, 3)},
        num_points=(5, 4),
        dense_structure=ProductStructure((("p",), ("t",))),
        key=jr.key(0),
    )

    p_axis = sep.dense_structure.axis_for("p")
    t_axis = sep.dense_structure.axis_for("t")
    x_axes = sep.coord_axes_by_label["x"]

    p_field = _scalar_field_from_dense(sep.points["p"], p_axis)
    t_field = _scalar_field_from_dense(sep.points["t"], t_axis)
    x0_data = jnp.asarray(sep.points["x"][0].data)
    x1_data = jnp.asarray(sep.points["x"][1].data)
    x0 = cx.Field(x0_data, dims=(x_axes[0],))
    x1 = cx.Field(x1_data, dims=(x_axes[1],))
    x_sum = x0 + x1
    one_x0 = cx.Field(jnp.ones_like(x0_data), dims=(x_axes[0],))
    one_x1 = cx.Field(jnp.ones_like(x1_data), dims=(x_axes[1],))
    p_data = jnp.asarray(p_field.data)
    t_data = jnp.asarray(t_field.data)
    one_p = cx.Field(jnp.ones_like(p_data), dims=(p_axis,))
    one_t = cx.Field(jnp.ones_like(t_data), dims=(t_axis,))

    expected_u = (p_field * t_field) * x_sum + 1.0
    expected_dx = (p_field * t_field) * one_x0 * one_x1
    expected_dy = expected_dx
    expected_dp = (one_p * t_field) * x_sum
    expected_dt = (p_field * one_t) * x_sum

    _assert_field_allclose(u(sep), expected_u)

    du_dx = partial_x(u, var="x")
    du_dy = partial_y(u, var="x")
    du_dp = partial_x(u, var="p")
    du_dt = partial_t(u, var="t")

    eval_jit = eqx.filter_jit(lambda f, b: f(b).data)
    _assert_array_allclose(eval_jit(du_dx, sep), expected_dx.data)
    _assert_array_allclose(eval_jit(du_dy, sep), expected_dy.data)
    _assert_array_allclose(eval_jit(du_dp, sep), expected_dp.data)
    _assert_array_allclose(eval_jit(du_dt, sep), expected_dt.data)


def test_latent_contraction_product_domain_integral_over_x():
    geom = Square(center=(0.0, 0.0), side=2.0)
    param = Interval1d(-1.0, 1.0).relabel("p")
    time = TimeInterval(0.0, 1.0)
    domain = geom @ param @ time

    model = LatentContractionModel(
        latent_size=1,
        out_size="scalar",
        x=ConstantXYLatentModel(),
        p=ScalarOffsetLatentModel(1.0),
        t=ScalarOffsetLatentModel(2.0),
    )
    u = _latent_function(domain, model)

    component = domain.component()
    sep = component.sample_coord_separable(
        {"x": (6, 5)},
        num_points=(4, 3),
        dense_structure=ProductStructure((("p",), ("t",))),
        key=jr.key(1),
    )

    out = integral(u, sep, component=component, over="x")
    p_axis = sep.dense_structure.axis_for("p")
    t_axis = sep.dense_structure.axis_for("t")
    p_field = _scalar_field_from_dense(sep.points["p"], p_axis)
    t_field = _scalar_field_from_dense(sep.points["t"], t_axis)
    area = jnp.asarray(geom.volume, dtype=float)
    param_base = cast(Interval1d, param.base)
    p_weight = jnp.asarray(param_base.volume, dtype=float) / float(
        sep.points["p"].data.shape[0]
    )
    t_weight = jnp.asarray(time.measure, dtype=float) / float(
        sep.points["t"].data.shape[0]
    )
    expected = area * p_weight * t_weight * (p_field + 1.0) * (t_field + 2.0)
    _assert_field_allclose(out, expected)

    eval_jit = eqx.filter_jit(
        lambda b: integral(u, b, component=component, over="x").data
    )
    _assert_array_allclose(eval_jit(sep), expected.data)


def test_latent_contraction_product_domain_paired_dense_block():
    geom = Square(center=(0.0, 0.0), side=2.0)
    param = Interval1d(-1.0, 1.0).relabel("p")
    time = TimeInterval(0.0, 1.0)
    domain = geom @ param @ time

    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=XYLatentModel(),
        p=ScalarLatentModel(),
        t=ScalarLatentModel(),
    )
    u = _latent_function(domain, model)

    component = domain.component()
    sep = component.sample_coord_separable(
        {"x": (3, 2)},
        num_points=6,
        dense_structure=ProductStructure((("p", "t"),)),
        key=jr.key(2),
    )

    axis = sep.dense_structure.axis_for("p")
    x_axes = sep.coord_axes_by_label["x"]
    p_field = _scalar_field_from_dense(sep.points["p"], axis)
    t_field = _scalar_field_from_dense(sep.points["t"], axis)
    x0 = cx.Field(sep.points["x"][0].data, dims=(x_axes[0],))
    x1 = cx.Field(sep.points["x"][1].data, dims=(x_axes[1],))
    expected = (p_field * t_field) * (x0 + x1) + 1.0

    _assert_field_allclose(u(sep), expected)
    eval_jit = eqx.filter_jit(lambda b: u(b).data)
    _assert_array_allclose(eval_jit(sep), expected.data)


def test_latent_contraction_multi_coord_separable_labels():
    geom = Square(center=(0.0, 0.0), side=2.0)
    param = Interval1d(-1.0, 1.0).relabel("p")
    time = TimeInterval(0.0, 1.0)
    domain = geom @ param @ time

    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=XYLatentModel(),
        p=ScalarLatentModel(),
        t=ScalarLatentModel(),
    )
    u = _latent_function(domain, model)

    component = domain.component()
    sep = component.sample_coord_separable(
        {"x": (3, 2), "p": 4},
        num_points=5,
        dense_structure=ProductStructure((("t",),)),
        key=jr.key(3),
    )

    t_axis = sep.dense_structure.axis_for("t")
    x_axes = sep.coord_axes_by_label["x"]
    p_axis = sep.coord_axes_by_label["p"][0]
    t_field = cx.Field(sep.points["t"].data, dims=(t_axis,))
    x0 = cx.Field(sep.points["x"][0].data, dims=(x_axes[0],))
    x1 = cx.Field(sep.points["x"][1].data, dims=(x_axes[1],))
    p_field = sep.points["p"][0]
    expected = (t_field * (x0 + x1)) * p_field + 1.0

    _assert_field_allclose(u(sep), expected)
    eval_jit = eqx.filter_jit(lambda b: u(b).data)
    assert jnp.allclose(eval_jit(sep), expected.data, atol=1e-6)
