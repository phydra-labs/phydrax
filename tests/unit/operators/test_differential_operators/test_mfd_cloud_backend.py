#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.domain import Interval1d, Square
from phydrax.operators.differential import (
    build_mfd_cloud_plan,
    build_mfd_cloud_plans,
    laplacian,
    MFDCloudPlan,
    partial_n,
)


def _x_values(x):
    x_arr = jnp.asarray(x, dtype=float)
    if x_arr.ndim == 0:
        return x_arr.reshape(())
    if x_arr.ndim == 1:
        if int(x_arr.shape[0]) == 1:
            return x_arr[0]
        return x_arr
    if x_arr.ndim == 2 and int(x_arr.shape[1]) == 1:
        return x_arr[:, 0]
    raise ValueError(f"Unsupported x shape {x_arr.shape}.")


def _xy_values(x):
    x_arr = jnp.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        return x_arr[0], x_arr[1]
    if x_arr.ndim == 2 and int(x_arr.shape[1]) == 2:
        return x_arr[:, 0], x_arr[:, 1]
    raise ValueError(f"Unsupported x shape {x_arr.shape}.")


def _square_points(n: int = 7):
    a = jnp.linspace(-1.0, 1.0, int(n))
    xx, yy = jnp.meshgrid(a, a, indexing="ij")
    pts = jnp.stack((xx.reshape((-1,)), yy.reshape((-1,))), axis=-1)
    return pts


def test_partial_n_mfd_cloud_polynomial_exact_on_irregular_points():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        xx = _x_values(x)
        return xx**3

    coords = jnp.array(
        [
            0.00,
            0.02,
            0.05,
            0.09,
            0.14,
            0.20,
            0.27,
            0.35,
            0.44,
            0.54,
            0.65,
            0.77,
            0.90,
            1.00,
        ],
        dtype=float,
    )
    plan = build_mfd_cloud_plan(coords, order=1, k=5)

    d1 = partial_n(u, var="x", order=1, backend="mfd")
    out = jnp.asarray(d1.func(coords, mfd_mode="cloud", mfd_cloud_plan=plan))
    expected = 3.0 * coords**2
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_partial_n_mfd_cloud_second_derivative_polynomial_exact():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        xx = _x_values(x)
        return xx**4

    coords = jnp.array(
        [0.0, 0.01, 0.03, 0.06, 0.10, 0.16, 0.23, 0.32, 0.43, 0.56, 0.71, 0.88, 1.0],
        dtype=float,
    )
    plan = build_mfd_cloud_plan(coords, order=2, k=7)
    d2 = partial_n(u, var="x", order=2, backend="mfd")
    out = jnp.asarray(d2.func(coords, mfd_mode="cloud", mfd_cloud_plan=plan))
    expected = 12.0 * coords**2
    assert jnp.allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_partial_n_mfd_cloud_zero_mask_zeroes_output():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        xx = _x_values(x)
        return jnp.sin(2.0 * jnp.pi * xx)

    coords = jnp.linspace(0.0, 1.0, 32)
    plan = build_mfd_cloud_plan(coords, order=1, k=7)
    zero_mask_plan = MFDCloudPlan(
        neighbors_idx=plan.neighbors_idx,
        weights=plan.weights,
        mask=jnp.zeros_like(plan.mask),
        axis_i=plan.axis_i,
        order_i=plan.order_i,
    )

    d1 = partial_n(u, var="x", order=1, backend="mfd")
    out = jnp.asarray(d1.func(coords, mfd_mode="cloud", mfd_cloud_plan=zero_mask_plan))
    assert jnp.allclose(out, jnp.zeros_like(coords), rtol=0.0, atol=0.0)


def test_partial_n_mfd_cloud_requires_plan_and_matching_size():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        xx = _x_values(x)
        return xx**2

    d1 = partial_n(u, var="x", order=1, backend="mfd")
    coords = jnp.linspace(0.0, 1.0, 16)
    plan = build_mfd_cloud_plan(coords, order=1, k=5)

    with pytest.raises(ValueError, match="requires mfd_cloud_plan or mfd_cloud_plans"):
        d1.func(coords, mfd_mode="cloud")
    with pytest.raises(ValueError, match="num_points mismatch"):
        d1.func(coords[:-1], mfd_mode="cloud", mfd_cloud_plan=plan)


def test_partial_n_mfd_cloud_2d_axis_first_derivative_polynomial_exact():
    geom = Square(center=(0.0, 0.0), side=2.0)
    points = _square_points(7)

    @geom.Function("x")
    def u(x):
        xx, yy = _xy_values(x)
        return xx**3 + xx * yy + 0.5 * yy**2

    d1x = partial_n(u, var="x", axis=0, order=1, backend="mfd")
    plan = build_mfd_cloud_plan(points, order=1, axis=0, k=20, degree=3)
    out = jnp.asarray(d1x.func(points, mfd_mode="cloud", mfd_cloud_plan=plan))
    xx, yy = _xy_values(points)
    expected = 3.0 * xx**2 + yy
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected, rtol=2e-4, atol=2e-4)


def test_partial_n_mfd_cloud_2d_axis_second_derivative_polynomial_exact():
    geom = Square(center=(0.0, 0.0), side=2.0)
    points = _square_points(7)

    @geom.Function("x")
    def u(x):
        xx, yy = _xy_values(x)
        return xx**2 + xx * yy + yy**3

    d2x = partial_n(u, var="x", axis=0, order=2, backend="mfd")
    plan = build_mfd_cloud_plan(points, order=2, axis=0, k=20, degree=3)
    out = jnp.asarray(d2x.func(points, mfd_mode="cloud", mfd_cloud_plan=plan))
    expected = jnp.full((points.shape[0],), 2.0)
    assert out.shape == expected.shape
    xx, yy = _xy_values(points)
    interior = (jnp.abs(xx) < 0.6) & (jnp.abs(yy) < 0.6)
    assert jnp.allclose(out[interior], expected[interior], rtol=2e-3, atol=2e-3)
    assert jnp.max(jnp.abs(out - expected)) < 0.2


def test_partial_n_mfd_cloud_plan_order_axis_mismatch_errors():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        xx = _x_values(x)
        return xx**2

    coords = jnp.linspace(0.0, 1.0, 32)
    base_plan = build_mfd_cloud_plan(coords, order=1, k=5)
    bad_order_plan = MFDCloudPlan(
        neighbors_idx=base_plan.neighbors_idx,
        weights=base_plan.weights,
        mask=base_plan.mask,
        axis_i=base_plan.axis_i,
        order_i=2,
    )
    bad_axis_plan = MFDCloudPlan(
        neighbors_idx=base_plan.neighbors_idx,
        weights=base_plan.weights,
        mask=base_plan.mask,
        axis_i=1,
        order_i=base_plan.order_i,
        var_dim=2,
    )
    d1 = partial_n(u, var="x", order=1, backend="mfd")
    with pytest.raises(ValueError, match="order mismatch"):
        d1.func(coords, mfd_mode="cloud", mfd_cloud_plan=bad_order_plan)
    with pytest.raises(ValueError, match="var_dim mismatch"):
        d1.func(coords, mfd_mode="cloud", mfd_cloud_plan=bad_axis_plan)


def test_partial_n_mfd_cloud_plan_axis_mismatch_errors_nd():
    geom = Square(center=(0.0, 0.0), side=2.0)
    points = _square_points(6)

    @geom.Function("x")
    def u(x):
        xx, yy = _xy_values(x)
        return xx + yy

    d1x = partial_n(u, var="x", axis=0, order=1, backend="mfd")
    plan_axis1 = build_mfd_cloud_plan(points, order=1, axis=1, k=16, degree=2)
    with pytest.raises(ValueError, match="axis mismatch"):
        d1x.func(points, mfd_mode="cloud", mfd_cloud_plan=plan_axis1)


def test_partial_n_mfd_cloud_plan_var_dim_mismatch_errors():
    geom = Square(center=(0.0, 0.0), side=2.0)
    points = _square_points(6)

    @geom.Function("x")
    def u(x):
        xx, yy = _xy_values(x)
        return xx + yy

    plan = build_mfd_cloud_plan(points, order=1, axis=0, k=16, degree=2)
    bad_var_dim_plan = MFDCloudPlan(
        neighbors_idx=plan.neighbors_idx,
        weights=plan.weights,
        mask=plan.mask,
        axis_i=plan.axis_i,
        order_i=plan.order_i,
        var_dim=1,
    )
    d1 = partial_n(u, var="x", axis=0, order=1, backend="mfd")
    with pytest.raises(ValueError, match="var_dim mismatch"):
        d1.func(points, mfd_mode="cloud", mfd_cloud_plan=bad_var_dim_plan)


def test_partial_n_mfd_cloud_preserves_channels_and_partial_mask():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        xx = _x_values(x)
        return jnp.stack((xx**3, xx**2), axis=-1)

    coords = jnp.linspace(0.0, 1.0, 48)
    plan = build_mfd_cloud_plan(coords, order=1, k=5)
    partial_mask = jnp.ones_like(plan.mask)
    partial_mask = partial_mask.at[:, -1].set(False)
    masked_plan = MFDCloudPlan(
        neighbors_idx=plan.neighbors_idx,
        weights=plan.weights,
        mask=partial_mask,
        axis_i=plan.axis_i,
        order_i=plan.order_i,
    )
    d1 = partial_n(u, var="x", order=1, backend="mfd")
    out_full = jnp.asarray(d1.func(coords, mfd_mode="cloud", mfd_cloud_plan=plan))
    out_masked = jnp.asarray(
        d1.func(coords, mfd_mode="cloud", mfd_cloud_plan=masked_plan)
    )
    assert out_full.shape == (coords.shape[0], 2)
    assert out_masked.shape == out_full.shape
    assert not jnp.allclose(out_full, out_masked, rtol=1e-12, atol=1e-12)
    assert jnp.any(jnp.abs(out_masked) > 0.0)


def test_partial_n_mfd_cloud_plan_table_lookup_for_axis_order():
    geom = Square(center=(0.0, 0.0), side=2.0)
    points = _square_points(7)

    @geom.Function("x")
    def u(x):
        xx, yy = _xy_values(x)
        return xx**3 + yy**3

    plans = build_mfd_cloud_plans(points, specs=((0, 1), (1, 1)), k=20, degree=3)
    d1x = partial_n(u, var="x", axis=0, order=1, backend="mfd")
    out = jnp.asarray(d1x.func(points, mfd_mode="cloud", mfd_cloud_plans=plans))
    xx, _ = _xy_values(points)
    expected = 3.0 * xx**2
    assert jnp.allclose(out, expected, rtol=3e-4, atol=3e-4)

    d2x = partial_n(u, var="x", axis=0, order=2, backend="mfd")
    with pytest.raises(ValueError, match="missing required key"):
        d2x.func(points, mfd_mode="cloud", mfd_cloud_plans=plans)


def test_laplacian_mfd_cloud_nd_uses_axis_specific_plans():
    geom = Square(center=(0.0, 0.0), side=2.0)
    points = _square_points(7)

    @geom.Function("x")
    def u(x):
        xx, yy = _xy_values(x)
        return xx**2 + yy**2

    plans = build_mfd_cloud_plans(points, specs=((0, 2), (1, 2)), k=20, degree=3)
    lap = laplacian(u, var="x", backend="mfd")
    out = jnp.asarray(lap.func(points, mfd_mode="cloud", mfd_cloud_plans=plans))
    expected = jnp.full((points.shape[0],), 4.0)
    assert jnp.allclose(out, expected, rtol=3e-4, atol=3e-4)


def test_build_mfd_cloud_plan_rejects_axis_out_of_range():
    pts = _square_points(5)
    with pytest.raises(ValueError, match=r"axis must be in \[0,2\)"):
        build_mfd_cloud_plan(pts, order=1, axis=2, k=8)


def test_build_mfd_cloud_plan_rejects_duplicate_points():
    pts = jnp.array([0.0, 0.2, 0.2, 0.7], dtype=float)
    with pytest.raises(ValueError, match="requires unique cloud points"):
        build_mfd_cloud_plan(pts, order=1, k=3)


def test_build_mfd_cloud_plan_rejects_insufficient_points_for_order():
    pts = jnp.array([0.0, 1.0], dtype=float)
    with pytest.raises(ValueError, match="Need at least order\\+1 points"):
        build_mfd_cloud_plan(pts, order=2, k=3)


def test_build_mfd_cloud_plan_rejects_insufficient_neighbors_for_basis():
    pts = _square_points(4)
    with pytest.raises(ValueError, match="k is too small"):
        build_mfd_cloud_plan(pts, order=1, axis=0, k=6, degree=4)


def test_build_mfd_cloud_plan_rejects_degenerate_local_geometry():
    x = jnp.linspace(-1.0, 1.0, 24)
    pts = jnp.stack((x, jnp.zeros_like(x)), axis=-1)
    with pytest.raises(ValueError, match="rank-deficient|ill-conditioned"):
        build_mfd_cloud_plan(pts, order=1, axis=1, k=10, degree=2)


def test_mfd_cloud_plan_shape_validation():
    with pytest.raises(ValueError, match="neighbors_idx must have shape"):
        MFDCloudPlan(
            neighbors_idx=jnp.zeros((4,), dtype=jnp.int32),
            weights=jnp.zeros((4,), dtype=float),
            mask=jnp.zeros((4,), dtype=bool),
            axis_i=0,
            order_i=1,
        )
    with pytest.raises(ValueError, match="weights must match"):
        MFDCloudPlan(
            neighbors_idx=jnp.zeros((4, 3), dtype=jnp.int32),
            weights=jnp.zeros((4, 2), dtype=float),
            mask=jnp.ones((4, 3), dtype=bool),
            axis_i=0,
            order_i=1,
        )
    with pytest.raises(ValueError, match="mask must match"):
        MFDCloudPlan(
            neighbors_idx=jnp.zeros((4, 3), dtype=jnp.int32),
            weights=jnp.zeros((4, 3), dtype=float),
            mask=jnp.ones((4, 2), dtype=bool),
            axis_i=0,
            order_i=1,
        )
    with pytest.raises(ValueError, match="axis_i must be in"):
        MFDCloudPlan(
            neighbors_idx=jnp.zeros((4, 3), dtype=jnp.int32),
            weights=jnp.zeros((4, 3), dtype=float),
            mask=jnp.ones((4, 3), dtype=bool),
            axis_i=2,
            order_i=1,
            var_dim=2,
        )
    with pytest.raises(ValueError, match="degree must be >="):
        MFDCloudPlan(
            neighbors_idx=jnp.zeros((4, 3), dtype=jnp.int32),
            weights=jnp.zeros((4, 3), dtype=float),
            mask=jnp.ones((4, 3), dtype=bool),
            axis_i=0,
            order_i=2,
            degree=1,
        )


def test_partial_n_mfd_tuple_ignores_cloud_mode_kwargs():
    geom = Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.sin(2.0 * jnp.pi * x[0])

    d1 = partial_n(u, var="x", order=1, backend="mfd")
    coords = jnp.linspace(0.0, 1.0, 96)
    plan = build_mfd_cloud_plan(coords, order=1, k=5)
    out_base = d1.func((coords,), mfd_boundary_mode="biased", mfd_stencil_size=5)
    out_cloud_kwargs = d1.func(
        (coords,),
        mfd_boundary_mode="biased",
        mfd_stencil_size=5,
        mfd_mode="cloud",
        mfd_cloud_plan=plan,
    )
    assert jnp.allclose(out_base, out_cloud_kwargs, rtol=0.0, atol=0.0)
