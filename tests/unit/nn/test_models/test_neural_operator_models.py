#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.domain import (
    DatasetDomain,
    FourierAxisSpec,
    Interval1d,
    SampleLayout,
)
from phydrax.nn.models import DeepONet, FNO, MLP, SeparableMLP
from phydrax.operators.differential import laplacian


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_deeponet_domain_model_coord_separable_output_shape(scan):
    data = jnp.ones((3, 4), dtype=float)
    data_dom = DatasetDomain(data)
    geom = Interval1d(0.0, 1.0)
    domain = data_dom @ geom

    latent = 5
    branch = MLP(
        in_size=4, out_size=latent, width_size=8, depth=2, scan=scan, key=jr.key(0)
    )
    trunk = MLP(
        in_size="scalar",
        out_size=latent,
        width_size=8,
        depth=2,
        scan=scan,
        key=jr.key(1),
    )
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
    batch = component.sample(
        phx.domain.GridSampling(
            {"x": FourierAxisSpec(8)},
            dense=phx.domain.PointSampling(2, layout=SampleLayout((("data",),))),
        ),
        key=jr.key(0),
    )
    out = u(batch)

    data_axis = batch.dense_structure.axis_for("data")
    (x_axis,) = batch.coord_axes_by_label["x"]
    assert out.dims == (data_axis, x_axis)
    assert out.data.shape == (2, 8)
    assert jnp.all(jnp.isfinite(jnp.asarray(out.data)))


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_fno_one_dimensional_domain_model_output_shape_and_basis_laplacian(scan):
    n = 16
    data = jnp.ones((3, n), dtype=float)
    data_dom = DatasetDomain(data)
    geom = Interval1d(0.0, 1.0)
    domain = data_dom @ geom

    model = FNO(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=2,
        n_modes=(6,),
        scan=scan,
        key=jr.key(0),
    )
    u = domain.Model("data", "x")(model)
    du = laplacian(u, var="x", backend="basis", basis="fourier", periodic=True)

    component = domain.component()
    batch = component.sample(
        phx.domain.GridSampling(
            {"x": FourierAxisSpec(n)},
            dense=phx.domain.PointSampling(2, layout=SampleLayout((("data",),))),
        ),
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


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_fno_one_dimensional_rejects_point_like_input(scan):
    model = FNO(width=8, depth=2, n_modes=(6,), scan=scan, key=jr.key(0))
    data = jnp.ones((8,), dtype=float)
    with pytest.raises(ValueError, match="coord-separable grid evaluation"):
        _ = model((data, jnp.asarray([0.5], dtype=float)))


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_fno_two_dimensional_domain_model_output_shape_and_basis_laplacian(scan):
    nx = 12
    ny = 10
    data = jnp.ones((3, nx, ny), dtype=float)
    data_dom = DatasetDomain(data)
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=1.0).compile()
    )
    domain = data_dom @ geom

    model = FNO(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=2,
        n_modes=(6, 6),
        scan=scan,
        key=jr.key(0),
    )
    u = domain.Model("data", "x")(model)
    du = laplacian(u, var="x", backend="basis", basis="fourier", periodic=True)

    component = domain.component()
    batch = component.sample(
        phx.domain.GridSampling(
            {"x": (FourierAxisSpec(nx), FourierAxisSpec(ny))},
            dense=phx.domain.PointSampling(2, layout=SampleLayout((("data",),))),
        ),
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


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_fno_two_dimensional_rejects_point_like_input(scan):
    model = FNO(width=8, depth=2, n_modes=(6, 6), scan=scan, key=jr.key(0))
    data = jnp.ones((8, 8), dtype=float)
    with pytest.raises(ValueError, match="coord-separable grid evaluation"):
        _ = model(
            (data, jnp.asarray([0.5], dtype=float), jnp.asarray([0.25], dtype=float))
        )


def test_domain_model_explicit_binding_supports_plain_callable_blockwise_input():
    data = jnp.ones((3, 2), dtype=float)
    data_dom = DatasetDomain(data)
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=1.0).compile()
    )
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
    batch = component.sample(
        phx.domain.GridSampling(
            {"x": (FourierAxisSpec(nx), FourierAxisSpec(ny))},
            dense=phx.domain.PointSampling(2, layout=SampleLayout((("data",),))),
        ),
        key=jr.key(0),
    )

    with pytest.raises(TypeError, match="Plain callable models require"):
        domain.Model("data", "x")(plain_callable)

    binding = phx.nn.ModelBinding.blockwise(
        "structured",
        pass_key=True,
        pass_iter=True,
    )
    u = domain.Model("data", "x", binding=binding)(plain_callable)
    out = u(batch)

    data_axis = batch.dense_structure.axis_for("data")
    x_axis0, x_axis1 = batch.coord_axes_by_label["x"]
    assert out.dims == (data_axis, x_axis0, x_axis1)
    assert out.data.shape == (2, nx, ny)


def test_separable_mlp_domain_model_defaults_to_flat_point_packing():
    data = jnp.ones((3, 2), dtype=float)
    data_dom = DatasetDomain(data)
    geom = Interval1d(0.0, 1.0)
    domain = data_dom @ geom

    model = SeparableMLP(
        in_size=3,
        out_size=2,
        width_size=8,
        depth=2,
        latent_size=4,
        key=jr.key(0),
    )
    u = domain.Model("data", "x")(model)

    batch = domain.component().sample(
        phx.domain.PointSampling(5, layout=SampleLayout((("data", "x"),))), key=jr.key(1)
    )
    out = u(batch)

    sample_axis = batch.structure.axis_for("data")
    assert out.dims == (sample_axis, None)
    assert out.data.shape == (5, 2)
    assert jnp.all(jnp.isfinite(jnp.asarray(out.data)))


def test_separable_mlp_domain_model_still_uses_structured_blockwise_grids():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=1.0).compile()
    )
    model = SeparableMLP(
        in_size=2,
        out_size=2,
        width_size=8,
        depth=2,
        latent_size=4,
        key=jr.key(2),
    )
    u = geom.Model("x")(model)

    batch = geom.component().sample(
        phx.domain.GridSampling({"x": (FourierAxisSpec(5), FourierAxisSpec(4))}),
        key=jr.key(3),
    )
    out = u(batch)

    x_axis0, x_axis1 = batch.coord_axes_by_label["x"]
    assert out.dims == (x_axis0, x_axis1, None)
    assert out.data.shape == (5, 4, 2)
    assert jnp.all(jnp.isfinite(jnp.asarray(out.data)))


@pytest.mark.parametrize("scan", (False, True), ids=("no_scan", "scan"))
def test_separable_mlp_key_none_avoids_eval_time_random_split(scan):
    model = SeparableMLP(
        in_size=3,
        out_size="scalar",
        width_size=8,
        depth=2,
        latent_size=4,
        scan=scan,
        key=jr.key(4),
    )
    x = jnp.asarray([0.1, 0.2, 0.3])

    y_none = model(x, key=None)
    y_keyed = model(x, key=jr.key(5))
    assert jnp.allclose(y_none, y_keyed)

    keyless_jaxpr = str(jax.make_jaxpr(lambda z: model(z, key=None))(x))
    keyed_jaxpr = str(jax.make_jaxpr(lambda z, k: model(z, key=k))(x, jr.key(6)))
    assert "random_split" not in keyless_jaxpr
    assert "random_split" in keyed_jaxpr


def test_domain_model_explicit_key_none_reaches_model_export_path():
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=1.0).compile()
    )
    model = SeparableMLP(
        in_size=2,
        out_size="scalar",
        width_size=8,
        depth=2,
        latent_size=4,
        key=jr.key(7),
    )
    u = geom.Model("x")(model)
    x = jnp.asarray([0.1, 0.2])

    y_none = u.func(x, key=None)
    y_keyed = u.func(x, key=jr.key(8))
    assert jnp.allclose(y_none, y_keyed)

    keyless_jaxpr = str(jax.make_jaxpr(lambda z: u.func(z, key=None))(x))
    assert "random_split" not in keyless_jaxpr


def test_domain_model_rejects_binding_override_for_phydrax_model():
    data = jnp.ones((3, 2), dtype=float)
    domain = DatasetDomain(data) @ Interval1d(0.0, 1.0)
    model = MLP(in_size=3, out_size=1, width_size=8, depth=2, key=jr.key(0))

    override = phx.nn.ModelBinding.pointwise("structured")
    with pytest.raises(ValueError, match="caller overrides"):
        domain.Model("data", "x", binding=override)(model)


def test_domain_model_axis_binding_requires_phydrax_model():
    data = jnp.ones((3, 2), dtype=float)
    domain = DatasetDomain(data) @ Interval1d(0.0, 1.0)
    binding = phx.nn.ModelBinding.axis()

    with pytest.raises(TypeError, match="Axis-batch model bindings require"):
        domain.Model("data", "x", binding=binding)(lambda x: x)
