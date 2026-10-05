#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from phydrax._interpolation import (
    cubic_hermite_interpolate,
    cubic_hermite_knot_jets,
    cubic_hermite_uniform_interpolate,
    CubicSplineEndCondition,
    CubicSplineSlopePlan,
    local_cubic_slopes,
    nearest_interpolate,
    UniformNodeGrid,
)
from phydrax.operators.interpolation import InterpolationResult, linear_interpolate


# SciPy's boundary vocabulary, kept at the oracle boundary only.
type _ScipyEnd = (
    Literal["not-a-knot", "clamped", "natural"] | tuple[Literal[1, 2], np.ndarray]
)
type _ScipyBoundary = (
    Literal["not-a-knot", "clamped", "natural"] | tuple[_ScipyEnd, _ScipyEnd]
)


def test_piecewise_scenario_1() -> None:
    public = linear_interpolate(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([2.0, 4.0]),
        jnp.asarray(0.5),
    )
    assert isinstance(public, InterpolationResult)
    assert jnp.allclose(public.values, 3.0)
    assert bool(public.support)

    nodes = jnp.asarray([-1.0, 0.5, 2.0])
    values = 3.0 * nodes - 2.0
    query = jnp.asarray([-0.25, 1.25])
    assert jnp.allclose(
        linear_interpolate(nodes, values, query).values,
        3.0 * query - 2.0,
    )
    assert jnp.allclose(
        linear_interpolate(nodes, values, query, derivative_order=1).values,
        3.0,
    )
    assert jnp.allclose(
        jax.jacfwd(lambda q: linear_interpolate(nodes, values, q).values)(query),
        jnp.eye(2) * 3.0,
    )

    payload_values = jnp.asarray([[0.0, 2.0, 8.0], [10.0, 12.0, 18.0]])
    payload = linear_interpolate(
        jnp.asarray([0.0, 1.0, 3.0]),
        payload_values,
        jnp.asarray([0.5, 2.0]),
        axis=1,
    )
    assert payload.values.shape == (2, 2)
    assert jnp.allclose(payload.values, jnp.asarray([[1.0, 11.0], [5.0, 15.0]]))
    assert jnp.array_equal(payload.support, jnp.ones(2, dtype=jnp.bool_))

    knot_nodes = jnp.asarray([0.0, 1.0, 2.0])
    knot_values = jnp.asarray([0.0, 1.0, 3.0])
    knot_query = jnp.asarray([0.5, 1.0])
    _, tangent = jax.jvp(
        lambda q: (
            linear_interpolate(
                knot_nodes,
                knot_values,
                q,
                bounds="extrapolate",
            ).values
        ),
        (knot_query,),
        (jnp.ones_like(knot_query),),
    )
    assert jnp.allclose(tangent, jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(
        linear_interpolate(
            knot_nodes,
            knot_values,
            jnp.asarray(1.0),
            derivative_order=1,
            bounds="extrapolate",
        ).values,
        2.0,
    )
    nodes = jnp.asarray([0.0, 1.0, 3.0])
    values = jnp.asarray([0.0, 2.0, 8.0])
    query = jnp.asarray([-1.0, 0.0, 0.5, 3.0, 4.0])
    clipped = linear_interpolate(nodes, values, query, bounds="clip")
    filled = linear_interpolate(nodes, values, query, bounds="fill", fill_value=-7.0)
    extrapolated = linear_interpolate(nodes, values, query, bounds="extrapolate")
    assert jnp.allclose(clipped.values, jnp.asarray([0.0, 0.0, 1.0, 8.0, 8.0]))
    assert jnp.array_equal(clipped.support, jnp.ones(5, dtype=jnp.bool_))
    assert jnp.allclose(filled.values, jnp.asarray([-7.0, 0.0, 1.0, 8.0, -7.0]))
    assert jnp.array_equal(
        filled.support,
        jnp.asarray([False, True, True, True, False]),
    )
    assert jnp.allclose(extrapolated.values, jnp.asarray([-2.0, 0.0, 1.0, 8.0, 11.0]))
    assert jnp.array_equal(extrapolated.support, jnp.ones(5, dtype=jnp.bool_))
    with pytest.raises(eqx.EquinoxRuntimeError, match="outside"):
        unsupported = linear_interpolate(nodes, values, query, bounds="error")
        jax.block_until_ready(unsupported.values)

    asymmetric = linear_interpolate(
        jnp.asarray([1.0, 2.0, 4.0]),
        jnp.asarray([3.0, 5.0, 9.0]),
        jnp.asarray([0.0, 1.5, 5.0]),
        bounds="fill",
        left_fill_value=0.0,
        right_fill_value=11.0,
    )
    assert jnp.allclose(asymmetric.values, jnp.asarray([0.0, 4.0, 11.0]))
    assert jnp.array_equal(asymmetric.support, jnp.asarray([False, True, False]))
    with pytest.raises(ValueError, match="bounds='fill'"):
        linear_interpolate(
            jnp.asarray([0.0, 1.0]),
            jnp.asarray([0.0, 1.0]),
            jnp.asarray(0.5),
            left_fill_value=0.0,
        )

    source_mask = jnp.asarray([True, False, True])
    masked_query = jnp.asarray([0.5, 2.0])
    strict = linear_interpolate(
        nodes,
        values,
        masked_query,
        source_mask=source_mask,
        mask_mode="strict",
        fill_value=-7.0,
    )
    renormalized = linear_interpolate(
        nodes,
        values,
        masked_query,
        source_mask=source_mask,
        mask_mode="renormalize",
    )
    assert jnp.array_equal(strict.support, jnp.asarray([False, False]))
    assert jnp.allclose(strict.values, -7.0)
    assert jnp.array_equal(renormalized.support, jnp.asarray([True, True]))
    assert jnp.allclose(renormalized.values, jnp.asarray([0.0, 8.0]))
    nodes = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = jnp.arange(4.0)
    query = jnp.asarray([0.5, 1.5, -1.0, 4.0])

    lower = nearest_interpolate(
        nodes, values, query, tie_policy="lower", bounds="fill", fill_value=-1.0
    )
    round_even = nearest_interpolate(
        nodes, values, query, tie_policy="round_even", bounds="clip"
    )

    assert jnp.allclose(lower.values, jnp.asarray([0.0, 1.0, -1.0, -1.0]))
    assert jnp.array_equal(lower.support, jnp.asarray([True, True, False, False]))
    assert jnp.allclose(round_even.values, jnp.asarray([0.0, 2.0, 0.0, 3.0]))
    nodes = jnp.asarray([0.0, 1.0, 2.0])
    values = nodes**2
    invalid_orders: tuple[object, ...] = (True, 1.5, "1")
    for derivative_order in invalid_orders:
        with pytest.raises(TypeError, match="integer"):
            # ty: ignore[invalid-argument-type]
            linear_interpolate(
                nodes,
                values,
                jnp.asarray(0.5),
                derivative_order=derivative_order,
            )
        with pytest.raises(TypeError, match="integer"):
            # ty: ignore[invalid-argument-type]
            cubic_hermite_interpolate(
                nodes,
                values,
                jnp.asarray(0.5),
                derivative_order=derivative_order,
            )

    matrix = jnp.ones((2, 3))
    with pytest.raises(TypeError, match="axis"):
        # ty: ignore[invalid-argument-type]
        linear_interpolate(nodes, matrix, 0.5, axis=1.5)
    with pytest.raises(ValueError, match="out of bounds"):
        linear_interpolate(nodes, matrix, 0.5, axis=2)


def test_piecewise_scenario_2() -> None:
    nodes = jnp.asarray([0.0, 1.0, 3.0, 6.0])
    values = jnp.asarray([0.0, 2.0, 8.0, 20.0])

    slopes = local_cubic_slopes(nodes, values)

    assert jnp.allclose(slopes, jnp.asarray([2.0, 2.5, 3.5, 4.0]))
    nodes = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    values = nodes**2
    query = jnp.asarray([1.25, 1.5, 1.75])

    interpolated = cubic_hermite_interpolate(nodes, values, query).values
    first = cubic_hermite_interpolate(nodes, values, query, derivative_order=1).values
    second = cubic_hermite_interpolate(nodes, values, query, derivative_order=2).values

    assert jnp.allclose(interpolated, query**2)
    assert jnp.allclose(first, 2.0 * query)
    assert jnp.allclose(second, 2.0)
    assert jnp.allclose(
        jax.jacfwd(lambda q: cubic_hermite_interpolate(nodes, values, q).values)(query),
        jnp.diag(2.0 * query),
    )
    nodes = jnp.asarray([2.0])
    values = jnp.asarray([[1.0 + 2.0j, 3.0 - 4.0j]])
    query = jnp.asarray([-1.0, 2.0, 5.0])

    linear = linear_interpolate(nodes, values, query).values
    cubic = cubic_hermite_interpolate(nodes, values, query).values
    derivative = cubic_hermite_interpolate(
        nodes, values, query, derivative_order=2
    ).values

    assert linear.shape == (3, 2)
    assert jnp.allclose(linear, values[0])
    assert jnp.allclose(cubic, values[0])
    assert jnp.allclose(derivative, 0.0)


def _smooth_samples(nodes: np.ndarray, /) -> np.ndarray:
    return np.stack((np.sin(2.0 * nodes), np.exp(-nodes)), axis=-1)


@pytest.mark.parametrize("derivative_order", [0, 1, 2], ids=["value", "d1", "d2"])
def test_uniform_hermite_matches_general_hermite(derivative_order: int) -> None:
    grid = UniformNodeGrid(-0.5, 2.5, 7)
    nodes = np.asarray(grid.nodes)
    values = _smooth_samples(nodes)
    slopes = np.stack((2.0 * np.cos(2.0 * nodes), -np.exp(-nodes)), axis=-1)
    query = jnp.concatenate((grid.nodes, jnp.asarray([-0.31, 0.77, 1.9, 2.5])))

    uniform = cubic_hermite_uniform_interpolate(
        grid, values, slopes, query, derivative_order=derivative_order
    )
    general = cubic_hermite_interpolate(
        nodes,
        values,
        query,
        slopes=slopes,
        derivative_order=derivative_order,
        bounds="error",
    )

    assert uniform.values.shape == (query.shape[0], 2)
    np.testing.assert_allclose(uniform.values, general.values, rtol=1e-12, atol=1e-12)
    assert bool(jnp.all(uniform.support))


def test_uniform_location_at_stop_is_last_span_end() -> None:
    grid = UniformNodeGrid(0.0, 1.0, 11)
    location = grid.locate(jnp.asarray([0.0, 0.35, 1.0]))
    assert jnp.array_equal(location.lower, jnp.asarray([0, 3, 9], dtype=jnp.int32))
    np.testing.assert_allclose(location.fraction, [0.0, 0.5, 1.0], atol=1e-12)


def test_uniform_hermite_bounds_modes() -> None:
    grid = UniformNodeGrid(0.0, 3.0, 4)
    values = jnp.asarray([0.0, 2.0, 8.0, 18.0])
    slopes = 4.0 * grid.nodes
    query = jnp.asarray([-1.0, 1.5, 4.0])
    exact = 2.0 * query**2

    clipped = cubic_hermite_uniform_interpolate(
        grid, values, slopes, query, bounds="clip"
    )
    np.testing.assert_allclose(clipped.values, [0.0, 4.5, 18.0], atol=1e-12)
    assert bool(jnp.all(clipped.support))

    filled = cubic_hermite_uniform_interpolate(
        grid, values, slopes, query, bounds="fill", fill_value=-5.0
    )
    np.testing.assert_allclose(filled.values, [-5.0, 4.5, -5.0], atol=1e-12)
    assert jnp.array_equal(filled.support, jnp.asarray([False, True, False]))

    extrapolated = cubic_hermite_uniform_interpolate(
        grid, values, slopes, query, bounds="extrapolate"
    )
    np.testing.assert_allclose(extrapolated.values, exact, atol=1e-12)

    with pytest.raises(eqx.EquinoxRuntimeError, match="outside"):
        result = cubic_hermite_uniform_interpolate(grid, values, slopes, query)
        jax.block_until_ready(result.values)


@pytest.mark.parametrize(
    ("left", "right", "bc_type", "left_slope", "right_slope"),
    [
        ("not-a-knot", "clamped", ("not-a-knot", (1, np.zeros(2))), None, 0.0),
        ("natural", "natural", "natural", None, None),
        ("clamped", "clamped", ((1, np.full(2, 0.4)), (1, np.full(2, -1.3))), 0.4, -1.3),
        ("not-a-knot", "not-a-knot", "not-a-knot", None, None),
    ],
    ids=["compact-radial", "natural", "clamped", "not-a-knot"],
)
def test_spline_slopes_match_scipy(
    left: CubicSplineEndCondition,
    right: CubicSplineEndCondition,
    bc_type: _ScipyBoundary,
    left_slope: float | None,
    right_slope: float | None,
) -> None:
    grid = UniformNodeGrid(0.25, 4.0, 13)
    nodes = np.asarray(grid.nodes)
    values = _smooth_samples(nodes)
    plan = CubicSplineSlopePlan(grid, left=left, right=right)

    result = plan.slopes(values, left_slope=left_slope, right_slope=right_slope)

    oracle = CubicSpline(nodes, values, bc_type=bc_type)(nodes, 1)
    np.testing.assert_allclose(result.slopes, oracle, atol=1e-11)
    assert bool(result.successful)
    assert jnp.array_equal(result.status, jnp.zeros((2,), dtype=jnp.int32))


def test_spline_slopes_multi_rhs_equals_columns() -> None:
    grid = UniformNodeGrid(0.0, 2.0, 9)
    values = np.random.default_rng(3).normal(size=(9, 3, 2))
    plan = CubicSplineSlopePlan(grid, left="clamped", right="natural")
    left_slope = np.asarray([[0.1, -0.2], [0.3, 0.0], [1.0, 2.0]])

    batched = plan.slopes(values, left_slope=left_slope)

    assert batched.slopes.shape == (9, 3, 2)
    for i in range(3):
        for j in range(2):
            column = plan.slopes(values[:, i, j], left_slope=left_slope[i, j])
            np.testing.assert_allclose(
                batched.slopes[:, i, j], column.slopes, rtol=1e-12, atol=1e-12
            )


def test_not_a_knot_spline_reproduces_cubic() -> None:
    grid = UniformNodeGrid(-1.0, 2.0, 7)

    def cubic(x: jax.Array) -> jax.Array:
        return 0.5 * x**3 - x**2 + 2.0 * x - 3.0

    plan = CubicSplineSlopePlan(grid, left="not-a-knot", right="not-a-knot")
    slopes = plan.slopes(cubic(grid.nodes)).slopes
    query = jnp.linspace(-1.0, 2.0, 23)

    for order, exact in (
        (0, cubic(query)),
        (1, 1.5 * query**2 - 2.0 * query + 2.0),
        (2, 3.0 * query - 2.0),
    ):
        result = cubic_hermite_uniform_interpolate(
            grid, cubic(grid.nodes), slopes, query, derivative_order=order
        )
        np.testing.assert_allclose(result.values, exact, atol=1e-10)


def test_knot_jets_distinguish_spline_from_local_slopes() -> None:
    grid = UniformNodeGrid(0.0, 3.0, 10)
    values = np.sin(2.0 * np.asarray(grid.nodes))
    spline = CubicSplineSlopePlan(grid, left="natural", right="natural").slopes(values)
    local = local_cubic_slopes(grid.nodes, values)

    spline_jets = cubic_hermite_knot_jets(grid, values, spline.slopes)
    local_jets = cubic_hermite_knot_jets(grid, values, local)
    spline_jump = spline_jets.segment_end[:, :-1] - spline_jets.segment_start[:, 1:]
    local_jump = local_jets.segment_end[:, :-1] - local_jets.segment_start[:, 1:]

    assert spline_jets.segment_start.shape == (2, 9)
    np.testing.assert_allclose(spline_jump, 0.0, atol=1e-10)
    np.testing.assert_allclose(local_jump[0], 0.0, atol=1e-12)
    assert float(jnp.max(jnp.abs(local_jump[1]))) > 1e-2


def test_clamped_zero_slope_does_not_zero_curvature() -> None:
    grid = UniformNodeGrid(0.0, 2.0, 8)
    values = np.cos(np.asarray(grid.nodes))
    plan = CubicSplineSlopePlan(grid, left="not-a-knot", right="clamped")
    slopes = plan.slopes(values, right_slope=0.0).slopes

    jets = cubic_hermite_knot_jets(grid, values, slopes)

    np.testing.assert_allclose(jets.segment_end[0, -1], 0.0, atol=1e-12)
    assert abs(float(jets.segment_end[1, -1])) > 1e-2


def test_spline_slope_plan_refusals() -> None:
    grid = UniformNodeGrid(0.0, 1.0, 5)
    values = jnp.zeros(5)
    clamped = CubicSplineSlopePlan(grid, left="clamped", right="natural")
    with pytest.raises(ValueError, match="left_slope"):
        clamped.slopes(values)
    natural = CubicSplineSlopePlan(grid, left="natural", right="natural")
    with pytest.raises(ValueError, match="right_slope"):
        natural.slopes(values, right_slope=0.0)
    with pytest.raises(ValueError, match="at least 4"):
        CubicSplineSlopePlan(
            UniformNodeGrid(0.0, 1.0, 3), left="not-a-knot", right="natural"
        )
    with pytest.raises(ValueError):
        CubicSplineSlopePlan(grid, left="periodic", right="natural")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="exceed"):
        UniformNodeGrid(1.0, 1.0, 4)
    with pytest.raises(ValueError, match="exceed"):
        UniformNodeGrid(2.0, 1.0, 4)
