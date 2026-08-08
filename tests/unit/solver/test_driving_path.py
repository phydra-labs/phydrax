#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._interpolation import BSplineGrid
from phydrax._trainable import partition_trainable


def test_declared_callable_path_has_closed_support_and_explicit_schedule():
    path = phx.solver.CallableDrivingPath(
        lambda time, side: jnp.stack((time**2, 3.0 * time)),
        lambda time, side: jnp.stack((2.0 * time, 3.0)),
        support=jnp.asarray([-1.0, 2.0]),
        value_shape=(2,),
        path_id="declared:quadratic",
        breakpoints=jnp.asarray([0.0, 1.0, jnp.nan]),
        breakpoint_mask=jnp.asarray([True, True, False]),
    )

    assert path.path_id == "declared:quadratic"
    assert path.value_shape == (2,)
    assert jnp.allclose(jnp.stack(path.support), jnp.asarray([-1.0, 2.0]))
    assert jnp.array_equal(path.breakpoint_mask, jnp.asarray([True, True, False]))
    assert jnp.allclose(path.evaluate(-1.0), jnp.asarray([1.0, -3.0]))
    assert jnp.allclose(path.evaluate(2.0), jnp.asarray([4.0, 6.0]))
    assert jnp.allclose(path.derivative(0.5), jnp.asarray([1.0, 3.0]))
    assert jnp.allclose(path.increment(-0.5, 1.5), jnp.asarray([2.0, 6.0]))
    assert jnp.allclose(
        jax.jit(lambda time: path.evaluate(time))(0.25),
        jnp.asarray([0.0625, 0.75]),
    )

    with pytest.raises(Exception, match="outside its closed support"):
        path.evaluate(2.5)


def test_piecewise_linear_fit_respects_irregular_masked_samples_and_knot_sides():
    times = jnp.asarray([0.0, 0.25, 1.5, jnp.nan, jnp.nan])
    values = jnp.asarray([[0.0], [1.0], [2.0], [jnp.nan], [jnp.nan]])
    time_mask = jnp.asarray([True, True, True, False, False])
    value_mask = jnp.asarray([[True], [True], [True], [False], [False]])
    path, diagnostics = phx.solver.PiecewiseLinearDrivingPath.fit(
        times,
        values,
        time_mask=time_mask,
        value_mask=value_mask,
        path_id="sampled:irregular-linear",
    )

    assert path.num_samples == 3
    assert path.value_shape == (1,)
    assert jnp.allclose(jnp.stack(path.support), jnp.asarray([0.0, 1.5]))
    assert jnp.allclose(path.breakpoints[:2], jnp.asarray([0.25, 1.5]))
    assert jnp.array_equal(path.breakpoint_mask, jnp.asarray([True, False, False]))
    assert jnp.allclose(path.evaluate(0.875), jnp.asarray([1.5]))
    assert jnp.allclose(path.derivative(0.25, side="left"), jnp.asarray([4.0]))
    assert jnp.allclose(path.derivative(0.25, "right"), jnp.asarray([0.8]))
    assert jnp.allclose(path.increment(0.125, 0.875), jnp.asarray([1.0]))

    queries = jnp.asarray([0.0, 0.125, 0.25, 0.875, 1.5])
    expected = jnp.asarray([[0.0], [0.5], [1.0], [1.5], [2.0]])
    assert jnp.allclose(jax.jit(jax.vmap(path.evaluate))(queries), expected)
    assert diagnostics.status == "success"
    assert diagnostics.method_id == "sampled-piecewise-linear"
    assert diagnostics.sample_count == 3
    assert diagnostics.sample_capacity == 5
    assert bool(diagnostics.valid)
    assert jnp.allclose(diagnostics.maximum_residual, 0.0)


def test_causal_backward_hermite_uses_backward_slopes_and_is_c1_at_knots():
    times = jnp.asarray([0.0, 1.0, 3.0])
    values = jnp.asarray([0.0, 1.0, 5.0])
    mask = jnp.ones((3,), dtype=bool)
    path, diagnostics = phx.solver.CausalBackwardHermiteDrivingPath.fit(
        times,
        values,
        time_mask=mask,
        value_mask=mask,
        path_id="sampled:backward-hermite",
    )

    assert jnp.allclose(path.slopes, jnp.asarray([1.0, 1.0, 2.0]))
    assert jnp.allclose(path.evaluate(0.5), 0.5)
    assert jnp.allclose(path.evaluate(2.0), 2.75)
    assert jnp.allclose(path.derivative(2.0), 2.25)
    assert jnp.allclose(path.derivative(1.0, "left"), 1.0)
    assert jnp.allclose(path.derivative(1.0, "right"), 1.0)
    assert path.breakpoints.shape == (0,)
    assert path.breakpoint_mask.shape == (0,)
    assert diagnostics.approximation_id == (
        "backward-difference-cubic-hermite-interpolant"
    )
    assert jnp.allclose(diagnostics.maximum_residual, 0.0)


def test_offline_natural_cubic_has_hand_computed_value_derivative_and_increment():
    times = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = jnp.asarray([0.0, 1.0, 0.0, 1.0])
    mask = jnp.ones((4,), dtype=bool)
    path, diagnostics = phx.solver.OfflineCubicDrivingPath.fit(
        times,
        values,
        time_mask=mask,
        value_mask=mask,
        path_id="sampled:natural-cubic",
    )

    assert jnp.allclose(path.second_derivatives, jnp.asarray([0.0, -4.0, 4.0, 0.0]))
    assert jnp.allclose(path.evaluate(0.5), 0.75)
    assert jnp.allclose(path.derivative(0.5), 7.0 / 6.0)
    assert jnp.allclose(path.derivative(1.0, "left"), -1.0 / 3.0)
    assert jnp.allclose(path.derivative(1.0, "right"), -1.0 / 3.0)
    assert jnp.allclose(path.increment(0.5, 2.5), -0.5)
    assert diagnostics.backend == "jax-tridiagonal-solve"
    assert jnp.allclose(diagnostics.minimum_spacing, 1.0)
    assert jnp.allclose(diagnostics.maximum_spacing, 1.0)
    assert jnp.allclose(diagnostics.maximum_residual, 0.0)


def test_fixed_bspline_has_exact_one_sided_derivatives_and_coefficient_gradients():
    grid = BSplineGrid.open_uniform(1, 2, interval=(0.0, 2.0))
    coefficients = jnp.asarray([0.0, 1.0, 4.0])
    path = phx.solver.FixedBSplineDrivingPath(
        grid,
        coefficients,
        path_id="bspline:piecewise-linear",
    )

    assert path.value_shape == ()
    assert jnp.allclose(jnp.stack(path.support), jnp.asarray([0.0, 2.0]))
    assert jnp.allclose(path.breakpoints, jnp.asarray([1.0]))
    assert jnp.array_equal(path.breakpoint_mask, jnp.asarray([True]))
    assert jnp.allclose(path.evaluate(1.0, "left"), 1.0)
    assert jnp.allclose(path.evaluate(1.0, "right"), 1.0)
    assert jnp.allclose(path.derivative(1.0, "left"), 1.0)
    assert jnp.allclose(path.derivative(1.0, "right"), 3.0)
    assert jnp.allclose(path.increment(0.5, 1.5), 2.0)

    gradient = jax.grad(
        lambda control: phx.solver.FixedBSplineDrivingPath(
            grid, control, path_id="bspline:gradient"
        ).evaluate(0.5)
    )(coefficients)
    assert jnp.allclose(gradient, jnp.asarray([0.5, 0.5, 0.0]))

    trainable, fixed = partition_trainable(path)
    assert trainable.coefficients is not None
    assert trainable.grid is None
    assert fixed.coefficients is None
    assert fixed.grid is grid


def test_sample_fits_reject_nonprefix_partial_insufficient_and_duplicate_inputs():
    times = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = jnp.arange(8.0).reshape((4, 2))
    valid = jnp.ones((4,), dtype=bool)
    full_value_valid = jnp.ones(values.shape, dtype=bool)

    with pytest.raises(ValueError, match="time_mask must be a prefix"):
        phx.solver.PiecewiseLinearDrivingPath.fit(
            times,
            values,
            time_mask=jnp.asarray([True, False, True, False]),
            value_mask=full_value_valid,
            path_id="invalid:nonprefix-time",
        )
    with pytest.raises(ValueError, match="wholly valid or wholly invalid"):
        phx.solver.PiecewiseLinearDrivingPath.fit(
            times,
            values,
            time_mask=valid,
            value_mask=full_value_valid.at[2, 1].set(False),
            path_id="invalid:partial-value",
        )
    with pytest.raises(ValueError, match="at least 2 valid samples"):
        phx.solver.PiecewiseLinearDrivingPath.fit(
            times,
            values,
            time_mask=jnp.asarray([True, False, False, False]),
            value_mask=jnp.asarray([True, False, False, False]),
            path_id="invalid:insufficient-linear",
        )
    with pytest.raises(ValueError, match="at least 4 valid samples"):
        phx.solver.OfflineCubicDrivingPath.fit(
            times,
            values,
            time_mask=jnp.asarray([True, True, True, False]),
            value_mask=jnp.asarray([True, True, True, False]),
            path_id="invalid:insufficient-cubic",
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        phx.solver.CausalBackwardHermiteDrivingPath.fit(
            jnp.asarray([0.0, 1.0, 1.0, 2.0]),
            values,
            time_mask=valid,
            value_mask=valid,
            path_id="invalid:duplicate-time",
        )


def test_fixed_bspline_rejects_value_discontinuities_and_nonfinite_coefficients():
    discontinuous_grid = BSplineGrid(
        jnp.asarray([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
        1,
    )
    with pytest.raises(ValueError, match="continuous at knots"):
        phx.solver.FixedBSplineDrivingPath(
            discontinuous_grid,
            jnp.asarray([0.0, 1.0, 2.0, 3.0]),
            path_id="invalid:discontinuous-bspline",
        )

    grid = BSplineGrid.open_uniform(2, 2, interval=(0.0, 1.0))
    with pytest.raises(Exception, match="coefficients must be finite"):
        phx.solver.FixedBSplineDrivingPath(
            grid,
            jnp.asarray([0.0, 1.0, jnp.nan, 2.0]),
            path_id="invalid:nonfinite-coefficients",
        )


def test_sampled_and_fixed_bspline_paths_reject_zero_sized_payload_dimensions():
    with pytest.raises(ValueError, match="value_shape dimensions must be positive"):
        phx.solver.PiecewiseLinearDrivingPath(
            jnp.asarray([0.0, 1.0]),
            jnp.empty((2, 0)),
            time_mask=jnp.ones((2,), dtype=bool),
            value_mask=jnp.ones((2,), dtype=bool),
            path_id="invalid:empty-sampled-payload",
        )

    grid = BSplineGrid.open_uniform(1, 1, interval=(0.0, 1.0))
    with pytest.raises(ValueError, match="value_shape dimensions must be positive"):
        phx.solver.FixedBSplineDrivingPath(
            grid,
            jnp.empty((grid.coefficient_count, 0)),
            path_id="invalid:empty-bspline-payload",
        )
