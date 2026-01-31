#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.domain import (
    DatasetDomain,
    FourierAxisSpec,
    Interval1d,
    ProductStructure,
    Square,
)
from phydrax.nn.models import DeepONet, FNO1d, FNO2d, MLP
from phydrax.operators.differential import laplacian


def test_deeponet_domain_model_coord_separable_output_shape():
    data = jnp.ones((3, 4), dtype=float)
    data_dom = DatasetDomain(data)
    geom = Interval1d(0.0, 1.0)
    domain = data_dom @ geom

    latent = 5
    branch = MLP(in_size=4, out_size=latent, width_size=8, depth=2, key=jr.key(0))
    trunk = MLP(in_size="scalar", out_size=latent, width_size=8, depth=2, key=jr.key(1))
    model = DeepONet(
        branch=branch,
        trunk=trunk,
        coord_dim=1,
        latent_size=latent,
        out_size="scalar",
        in_size=4,
    )
    u = domain.Model("data", "x")(model)

    component = domain.component()
    batch = component.sample_coord_separable(
        {"x": FourierAxisSpec(8)},
        num_points=2,
        dense_structure=ProductStructure((("data",),)),
        key=jr.key(0),
    )
    out = u(batch)

    data_axis = batch.dense_structure.axis_for("data")
    (x_axis,) = batch.coord_axes_by_label["x"]
    assert out.dims == (data_axis, x_axis)
    assert out.data.shape == (2, 8)
    assert jnp.all(jnp.isfinite(jnp.asarray(out.data)))


def test_fno1d_domain_model_coord_separable_output_shape_and_basis_laplacian():
    n = 16
    data = jnp.ones((3, n), dtype=float)
    data_dom = DatasetDomain(data)
    geom = Interval1d(0.0, 1.0)
    domain = data_dom @ geom

    model = FNO1d(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=2,
        modes=6,
        key=jr.key(0),
    )
    u = domain.Model("data", "x")(model)
    du = laplacian(u, var="x", backend="basis", basis="fourier", periodic=True)

    component = domain.component()
    batch = component.sample_coord_separable(
        {"x": FourierAxisSpec(n)},
        num_points=2,
        dense_structure=ProductStructure((("data",),)),
        key=jr.key(0),
    )
    out = u(batch)
    out_lap = du(batch)

    data_axis = batch.dense_structure.axis_for("data")
    (x_axis,) = batch.coord_axes_by_label["x"]
    assert out.dims == (data_axis, x_axis)
    assert out.data.shape == (2, n)
    assert out_lap.dims == (data_axis, x_axis)
    assert out_lap.data.shape == (2, n)
    assert jnp.all(jnp.isfinite(jnp.asarray(out_lap.data)))


def test_fno1d_rejects_point_like_x_input():
    model = FNO1d(width=8, depth=2, modes=6, key=jr.key(0))
    data = jnp.ones((8,), dtype=float)
    with pytest.raises(ValueError, match="coord-separable grid evaluation"):
        _ = model((data, jnp.asarray([0.5], dtype=float)))


def test_fno2d_domain_model_coord_separable_output_shape_and_basis_laplacian():
    nx = 12
    ny = 10
    data = jnp.ones((3, nx, ny), dtype=float)
    data_dom = DatasetDomain(data)
    geom = Square(center=(0.0, 0.0), side=1.0)
    domain = data_dom @ geom

    model = FNO2d(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=2,
        modes=6,
        key=jr.key(0),
    )
    u = domain.Model("data", "x")(model)
    du = laplacian(u, var="x", backend="basis", basis="fourier", periodic=True)

    component = domain.component()
    batch = component.sample_coord_separable(
        {"x": (FourierAxisSpec(nx), FourierAxisSpec(ny))},
        num_points=2,
        dense_structure=ProductStructure((("data",),)),
        key=jr.key(0),
    )
    out = u(batch)
    out_lap = du(batch)

    data_axis = batch.dense_structure.axis_for("data")
    x_axis0, x_axis1 = batch.coord_axes_by_label["x"]
    assert out.dims == (data_axis, x_axis0, x_axis1)
    assert out.data.shape == (2, nx, ny)
    assert out_lap.dims == (data_axis, x_axis0, x_axis1)
    assert out_lap.data.shape == (2, nx, ny)
    assert jnp.all(jnp.isfinite(jnp.asarray(out_lap.data)))


def test_fno2d_rejects_point_like_xy_input():
    model = FNO2d(width=8, depth=2, modes=6, key=jr.key(0))
    data = jnp.ones((8, 8), dtype=float)
    with pytest.raises(ValueError, match="coord-separable grid evaluation"):
        _ = model(
            (data, jnp.asarray([0.5], dtype=float), jnp.asarray([0.25], dtype=float))
        )


def test_domain_model_structured_kwarg_allows_plain_callable_tuple_input():
    data = jnp.ones((3, 2), dtype=float)
    data_dom = DatasetDomain(data)
    geom = Square(center=(0.0, 0.0), side=1.0)
    domain = data_dom @ geom

    def plain_callable(inp, *, key=None, iter_=None):
        del key, iter_
        data_vec, x0, x1 = inp
        base = jnp.sum(jnp.asarray(data_vec, dtype=float))
        x0 = jnp.asarray(x0, dtype=float).reshape((-1, 1))
        x1 = jnp.asarray(x1, dtype=float).reshape((1, -1))
        return base + 0.0 * x0 + 0.0 * x1

    component = domain.component()
    nx, ny = 6, 5
    batch = component.sample_coord_separable(
        {"x": (FourierAxisSpec(nx), FourierAxisSpec(ny))},
        num_points=2,
        dense_structure=ProductStructure((("data",),)),
        key=jr.key(0),
    )

    u_unmarked = domain.Model("data", "x")(plain_callable)
    with pytest.raises(ValueError, match="does not support structured inputs"):
        _ = u_unmarked(batch)

    u_marked = domain.Model("data", "x", structured=True)(plain_callable)
    out = u_marked(batch)

    data_axis = batch.dense_structure.axis_for("data")
    x_axis0, x_axis1 = batch.coord_axes_by_label["x"]
    assert out.dims == (data_axis, x_axis0, x_axis1)
    assert out.data.shape == (2, nx, ny)
