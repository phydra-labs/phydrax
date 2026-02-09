#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax.operators.differential._domain_ops as differential_domain_ops
from phydrax.constraints import enforce_dirichlet, enforce_initial
from phydrax.domain import (
    Boundary,
    FixedStart,
    Interval1d,
    ProductStructure,
    TimeInterval,
)
from phydrax.domain._function import DomainFunction
from phydrax.nn.models import LatentContractionModel, LatentExecutionPolicy
from phydrax.nn.models.core._base import _AbstractBaseModel
from phydrax.operators.differential import dt_n, laplacian, partial_n, partial_t
from phydrax.operators.differential._hooks import (
    get_derivative_hook,
    with_derivative_hook,
)


def _as_scalar(x):
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return arr
    if arr.size != 1:
        raise ValueError("Expected scalar input for scalar factor model.")
    return arr.reshape(())


def _align_expected_tx_to_out_dims(
    expected_xt: jax.Array,
    out,
    /,
    *,
    x_axis: str,
    t_axis: str,
) -> jax.Array:
    out_dims = tuple(out.dims)
    out_shape = tuple(jnp.asarray(out.data).shape)

    aligned = expected_xt
    if x_axis in out_dims and t_axis in out_dims:
        i_x = int(out_dims.index(x_axis))
        i_t = int(out_dims.index(t_axis))
        if i_t == i_x:
            raise ValueError("t-axis and x-axis resolved to the same output dimension.")
        aligned = jnp.moveaxis(expected_xt, (0, 1), (i_x, i_t))

    aligned_shape = tuple(aligned.shape)
    if aligned_shape == out_shape:
        return aligned
    if tuple(expected_xt.shape) == out_shape:
        return expected_xt
    if tuple(expected_xt.T.shape) == out_shape:
        return expected_xt.T
    raise ValueError(
        "Could not align expected shape "
        f"{tuple(expected_xt.shape)} to output shape {out_shape} "
        f"with output dims {out_dims!r}."
    )


def _axis_for_label_in_coord_batch(batch, label: str) -> str:
    if label in batch.coord_axes_by_label:
        axes = batch.coord_axes_by_label[label]
        if len(axes) != 1:
            raise ValueError(
                f"Test helper expects 1D coord-separable axes for {label!r}, got {axes!r}."
            )
        return axes[0]
    axis = batch.dense_structure.axis_for(label)
    if axis is None:
        raise ValueError(f"Could not resolve axis for label {label!r}.")
    return axis


class ScalarLatentModel(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str

    def __init__(self) -> None:
        self.in_size = "scalar"
        self.out_size = 2

    def __call__(self, x, /, *, key=jr.key(0)):
        x = _as_scalar(x)
        return jnp.stack([x, jnp.array(1.0)], axis=-1)


class MonomialScalarLatentModel(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str
    power: int

    def __init__(self, power: int) -> None:
        self.in_size = "scalar"
        self.out_size = 1
        self.power = int(power)

    def __call__(self, x, /, *, key=jr.key(0)):
        x = _as_scalar(x)
        return jnp.asarray([x**self.power], dtype=float)


def test_domain_model_blockwise_pointsbatch_singleton_blocks():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=ScalarLatentModel(),
        t=ScalarLatentModel(),
        execution_policy=LatentExecutionPolicy(fallback="warn"),
    )
    u = domain.Model("x", "t")(model)

    component = domain.component()
    structure = ProductStructure((("x",), ("t",)))
    batch = component.sample((5, 7), structure=structure, key=jr.key(0))
    out = u(batch)

    axis_x = batch.structure.axis_for("x")
    axis_t = batch.structure.axis_for("t")
    assert axis_x is not None and axis_t is not None
    assert out.dims[:2] == (axis_x, axis_t)

    x_vals = jnp.asarray(batch.points["x"].data)[:, 0]
    t_vals = jnp.asarray(batch.points["t"].data)
    expected = x_vals[:, None] * t_vals[None, :] + 1.0
    assert jnp.allclose(out.data, expected)


def test_domain_model_warns_on_blockwise_fallback_for_paired_blocks():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=ScalarLatentModel(),
        t=ScalarLatentModel(),
        execution_policy=LatentExecutionPolicy(fallback="warn"),
    )
    u = domain.Model("x", "t")(model)

    component = domain.component()
    structure = ProductStructure((("x", "t"),))
    batch = component.sample(9, structure=structure, key=jr.key(1))

    with pytest.warns(UserWarning, match="singleton block"):
        out = u(batch)

    x_vals = jnp.asarray(batch.points["x"].data)[:, 0]
    t_vals = jnp.asarray(batch.points["t"].data)
    expected = x_vals * t_vals + 1.0
    assert jnp.allclose(jnp.asarray(out.data), expected)


def test_latent_derivative_path_matches_exact_values():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=1,
        out_size="scalar",
        x=MonomialScalarLatentModel(2),
        t=MonomialScalarLatentModel(3),
    )
    u = domain.Model("x", "t")(model)

    component = domain.component()
    batch = component.sample_coord_separable(
        {"x": 9},
        num_points=7,
        dense_structure=ProductStructure((("t",),)),
        key=jr.key(2),
    )
    x_axis = _axis_for_label_in_coord_batch(batch, "x")
    t_axis = _axis_for_label_in_coord_batch(batch, "t")
    x = jnp.asarray(batch.points["x"][0].data)
    t = jnp.asarray(batch.points["t"].data).reshape((-1,))
    expected_dt_tx = (x**2)[:, None] * (3.0 * t**2)[None, :]
    expected_dxx_tx = jnp.ones((x.shape[0], 1), dtype=float) * (2.0 * t**3)[None, :]

    du_dt = dt_n(u, var="t", order=1, backend="ad")
    du_dt_partial = partial_t(u, var="t")
    dxx = partial_n(u, var="x", axis=0, order=2, backend="ad")
    lap = laplacian(u, var="x", backend="ad")

    out_dt = du_dt(batch)
    out_dt_partial = du_dt_partial(batch)
    out_dxx = dxx(batch)
    out_lap = lap(batch)

    expected_dt = _align_expected_tx_to_out_dims(
        expected_dt_tx, out_dt, t_axis=t_axis, x_axis=x_axis
    )
    expected_dt_partial = _align_expected_tx_to_out_dims(
        expected_dt_tx, out_dt_partial, t_axis=t_axis, x_axis=x_axis
    )
    expected_dxx = _align_expected_tx_to_out_dims(
        expected_dxx_tx, out_dxx, t_axis=t_axis, x_axis=x_axis
    )
    expected_lap = _align_expected_tx_to_out_dims(
        expected_dxx_tx, out_lap, t_axis=t_axis, x_axis=x_axis
    )

    assert jnp.allclose(jnp.asarray(out_dt.data), expected_dt, atol=1e-6)
    assert jnp.allclose(jnp.asarray(out_dt_partial.data), expected_dt_partial, atol=1e-6)
    assert jnp.allclose(jnp.asarray(out_dxx.data), expected_dxx, atol=1e-6)
    assert jnp.allclose(jnp.asarray(out_lap.data), expected_lap, atol=1e-6)


def test_latent_derivative_path_flat_topology_matches_exact_values():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=1,
        out_size="scalar",
        x=MonomialScalarLatentModel(2),
        t=MonomialScalarLatentModel(3),
        execution_policy=LatentExecutionPolicy(topology="flat", fallback="warn"),
    )
    u = domain.Model("x", "t")(model)
    component = domain.component()
    batch = component.sample_coord_separable(
        {"x": 9},
        num_points=7,
        dense_structure=ProductStructure((("t",),)),
        key=jr.key(12),
    )
    x_axis = _axis_for_label_in_coord_batch(batch, "x")
    t_axis = _axis_for_label_in_coord_batch(batch, "t")
    x = jnp.asarray(batch.points["x"][0].data)
    t = jnp.asarray(batch.points["t"].data).reshape((-1,))
    expected_dt_tx = (x**2)[:, None] * (3.0 * t**2)[None, :]

    du_dt = dt_n(u, var="t", order=1, backend="ad")
    out_dt = du_dt(batch)
    expected_dt = _align_expected_tx_to_out_dims(
        expected_dt_tx, out_dt, t_axis=t_axis, x_axis=x_axis
    )
    assert jnp.allclose(jnp.asarray(out_dt.data), expected_dt, atol=1e-6)


def test_latent_backend_ad_does_not_use_jet(monkeypatch):
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=1,
        out_size="scalar",
        x=MonomialScalarLatentModel(2),
        t=MonomialScalarLatentModel(3),
    )
    u = domain.Model("x", "t")(model)
    du_dt = dt_n(u, var="t", order=2, backend="ad")
    dxx = laplacian(u, var="x", backend="ad")

    def _jet_fail(*args, **kwargs):
        raise AssertionError("jet_dn should not be used for backend='ad'.")

    monkeypatch.setattr(differential_domain_ops, "jet_dn", _jet_fail)

    out_dt = du_dt.func(jnp.asarray([0.5]), 0.25)
    out_dxx = dxx.func(jnp.asarray([0.5]), 0.25)
    assert jnp.isfinite(jnp.asarray(out_dt))
    assert jnp.isfinite(jnp.asarray(out_dxx))


def test_latent_derivative_path_ignores_iter_kwarg():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=1,
        out_size="scalar",
        x=MonomialScalarLatentModel(2),
        t=MonomialScalarLatentModel(3),
        execution_policy=LatentExecutionPolicy(fallback="warn"),
    )
    u = domain.Model("x", "t")(model)
    du_dt = dt_n(u, var="t", order=2, backend="jet")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = du_dt.func(jnp.asarray([0.5]), 0.25, iter_=3)

    fallback_msgs = [
        str(w.message)
        for w in caught
        if "Falling back to generic derivative path for LatentContractionModel"
        in str(w.message)
    ]
    assert not fallback_msgs
    expected = (0.5**2) * (6.0 * 0.25)
    assert jnp.allclose(jnp.asarray(out), expected, atol=1e-6)


def test_enforced_dirichlet_preserves_latent_derivative_fast_path():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=1,
        out_size="scalar",
        x=MonomialScalarLatentModel(2),
        t=MonomialScalarLatentModel(3),
        execution_policy=LatentExecutionPolicy(fallback="warn"),
    )
    u = domain.Model("x", "t")(model)
    boundary = domain.component({"x": Boundary()})
    u_enforced = enforce_dirichlet(u, boundary, var="x", target=0.0)
    assert get_derivative_hook(u_enforced) is not None

    du_dt = dt_n(u_enforced, var="t", order=1, backend="jet")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = du_dt.func(jnp.asarray([0.25]), 0.4, iter_=2)
    fallback_msgs = [
        str(w.message)
        for w in caught
        if "Falling back to generic derivative path for LatentContractionModel"
        in str(w.message)
    ]
    assert not fallback_msgs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = du_dt.func(jnp.asarray([0.25]), 0.4, force_generic=True)
    assert jnp.allclose(jnp.asarray(out), jnp.asarray(expected), atol=1e-6)


def test_latent_derivative_path_warns_on_auto_fallback():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=ScalarLatentModel(),
        t=ScalarLatentModel(),
        execution_policy=LatentExecutionPolicy(fallback="warn"),
    )
    u = domain.Model("x", "t")(model)
    du_dt = dt_n(u, var="t", order=2, backend="jet")

    with pytest.warns(
        UserWarning,
        match="Falling back to generic derivative path for LatentContractionModel",
    ):
        _ = du_dt.func(jnp.asarray([0.25]), 0.4, force_generic=True)


def test_latent_derivative_path_respects_error_fallback():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=ScalarLatentModel(),
        t=ScalarLatentModel(),
        execution_policy=LatentExecutionPolicy(fallback="error"),
    )
    u = domain.Model("x", "t")(model)
    du_dt = dt_n(u, var="t", order=2, backend="jet")

    with pytest.raises(
        ValueError,
        match="Falling back to generic derivative path for LatentContractionModel",
    ):
        _ = du_dt.func(jnp.asarray([0.25]), 0.4, force_generic=True)


def test_binary_expression_derivative_hook_composes():
    domain = Interval1d(0.0, 1.0)

    @domain.Function("x")
    def base(x):
        return x[0]

    def _base_hook(
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: str,
        backend: str,
        basis: str,
        periodic: bool,
    ):
        del mode, basis, periodic
        if backend not in ("ad", "jet"):
            return None
        if var != "x":
            return None
        if axis not in (None, 0):
            return None
        n = int(order)
        if n == 0:
            return base
        if n == 1:
            return DomainFunction(domain=domain, deps=("x",), func=lambda x: 1.0)
        return DomainFunction(domain=domain, deps=("x",), func=lambda x: 0.0)

    hooked = with_derivative_hook(base, _base_hook)
    expr = (2.0 * hooked + 3.0) / (1.0 + hooked)
    assert get_derivative_hook(expr) is not None

    dexpr = partial_n(expr, var="x", axis=0, order=1, backend="ad")
    x = jnp.asarray([0.25], dtype=float)
    out = jnp.asarray(dexpr.func(x))
    expected = -1.0 / ((1.0 + x[0]) ** 2)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_boundary_gate_style_blend_preserves_hook():
    domain = Interval1d(0.0, 1.0) @ TimeInterval(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=1,
        out_size="scalar",
        x=MonomialScalarLatentModel(2),
        t=MonomialScalarLatentModel(3),
        execution_policy=LatentExecutionPolicy(fallback="warn"),
    )
    u_raw = domain.Model("x", "t")(model)
    initial = domain.component({"t": FixedStart()})
    u_init = enforce_initial(u_raw, initial, var="t", targets={0: 0.0})

    blended = u_raw + 0.5 * (u_init - u_raw)
    assert get_derivative_hook(blended) is not None

    du_dt = dt_n(blended, var="t", order=1, backend="ad")
    out = jnp.asarray(du_dt.func(jnp.asarray([0.2], dtype=float), 0.4))
    assert out.shape == ()


def test_coord_separable_laplacian_uses_fwdfwd_path(monkeypatch):
    domain = Interval1d(0.0, 1.0)
    model = LatentContractionModel(
        latent_size=1,
        out_size="scalar",
        x=MonomialScalarLatentModel(2),
        execution_policy=LatentExecutionPolicy(fallback="error"),
    )
    u = domain.Model("x")(model)
    lap = laplacian(u, var="x", backend="ad")

    def _latent_fast_should_not_run(*args, **kwargs):
        raise AssertionError("coord-separable laplacian should bypass latent fast path.")

    monkeypatch.setattr(
        differential_domain_ops,
        "_latent_try_partial_n_eval",
        _latent_fast_should_not_run,
    )

    batch = domain.component().sample_coord_separable(
        {"x": 17},
        dense_structure=ProductStructure(()),
        key=jr.key(111),
    )
    out = jnp.asarray(lap(batch).data)
    assert jnp.allclose(out, 2.0, atol=1e-6)
