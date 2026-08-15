#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline as SciPyBSpline

from phydrax._interpolation import (
    apply_gather_stencil,
    BoundsMode,
    bspline_evaluate,
    bspline_stencil,
)


def _open_knots(degree: int, control_count: int) -> np.ndarray:
    interior_count = control_count - degree - 1
    interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    return np.concatenate(
        (
            np.zeros(degree + 1),
            interior,
            np.ones(degree + 1),
        )
    )


def _dense_basis(
    knots,
    query,
    *,
    degree: int,
    derivative_order: int = 0,
    bounds: BoundsMode = "error",
):
    control_count = len(knots) - degree - 1
    stencil = bspline_stencil(
        knots,
        query,
        degree=degree,
        derivative_order=derivative_order,
        bounds=bounds,
    )
    return apply_gather_stencil(jnp.eye(control_count), stencil).values, stencil


@pytest.mark.parametrize("degree", range(6))
def test_bspline_basis_and_derivatives_match_scipy(degree):
    control_count = degree + 5
    knots = _open_knots(degree, control_count)
    query = np.asarray([0.0, 0.07, 0.31, 0.5, 0.83, 1.0 - 1e-8, 1.0])
    oracle = SciPyBSpline(knots, np.eye(control_count), degree)

    for derivative_order in range(degree + 1):
        actual, stencil = _dense_basis(
            knots,
            query,
            degree=degree,
            derivative_order=derivative_order,
        )
        expected = oracle(query, nu=derivative_order)
        assert stencil.indices.shape == query.shape + (degree + 1,)
        assert stencil.relation.width == degree + 1
        assert np.allclose(np.asarray(actual), expected, rtol=2e-10, atol=2e-10)
        expected_sum = 1.0 if derivative_order == 0 else 0.0
        assert np.allclose(
            np.asarray(jnp.sum(stencil.weights, axis=-1)),
            expected_sum,
            atol=2e-10,
        )


def test_bspline_repeated_and_unclamped_knots_match_scipy():
    cases = (
        (
            np.asarray(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.3,
                    0.5,
                    0.5,
                    0.75,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
            np.asarray(
                [
                    0.0,
                    0.3,
                    np.nextafter(0.5, 0.0),
                    0.5,
                    np.nextafter(0.5, 1.0),
                    1.0,
                ]
            ),
        ),
        (
            np.asarray([-2.0, -1.0, 0.0, 0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.5]),
            np.asarray([0.2, 0.4, 1.0, 1.4]),
        ),
    )

    for knots, query in cases:
        control_count = len(knots) - 4
        oracle = SciPyBSpline(knots, np.eye(control_count), 3)
        for derivative_order in range(4):
            actual, _ = _dense_basis(
                knots,
                query,
                degree=3,
                derivative_order=derivative_order,
            )
            assert np.allclose(
                np.asarray(actual),
                oracle(query, nu=derivative_order),
                rtol=2e-10,
                atol=2e-10,
            )


def test_bspline_explicit_derivatives_match_autodiff_at_endpoints():
    degree = 3
    knots = jnp.asarray(_open_knots(degree, 7))
    controls = jnp.asarray([0.1, -0.7, 0.4, 1.2, -0.2, 0.9, 1.4])

    def evaluate(query):
        return bspline_evaluate(knots, controls, query, degree=degree).values

    first = jax.jacfwd(evaluate)
    second = jax.jacfwd(first)
    third = jax.jacfwd(second)
    for query in jnp.asarray([0.0, 0.37, 1.0]):
        explicit_first = apply_gather_stencil(
            controls,
            bspline_stencil(
                knots,
                query,
                degree=degree,
                derivative_order=1,
            ),
        ).values
        explicit_second = apply_gather_stencil(
            controls,
            bspline_stencil(
                knots,
                query,
                degree=degree,
                derivative_order=2,
            ),
        ).values
        explicit_third = apply_gather_stencil(
            controls,
            bspline_stencil(
                knots,
                query,
                degree=degree,
                derivative_order=3,
            ),
        ).values
        assert float(first(query)) == pytest.approx(float(explicit_first), abs=1e-10)
        assert float(second(query)) == pytest.approx(float(explicit_second), abs=1e-10)
        assert float(third(query)) == pytest.approx(float(explicit_third), abs=1e-10)
        assert np.isfinite(np.asarray(jax.jit(evaluate)(query)))


def test_bspline_custom_jvp_combines_query_and_coefficient_tangents():
    degree = 3
    knots = jnp.asarray(_open_knots(degree, 7))
    coefficients = jnp.arange(14, dtype=float).reshape((7, 2)) / 9.0
    coefficient_tangent = jnp.cos(coefficients)
    query = jnp.asarray(0.43)
    query_tangent = jnp.asarray(-0.7)

    def evaluate(coefficients_, query_):
        return bspline_evaluate(
            knots,
            coefficients_,
            query_,
            degree=degree,
        ).values

    values, tangent = jax.jvp(
        evaluate,
        (coefficients, query),
        (coefficient_tangent, query_tangent),
    )
    expected_values = evaluate(coefficients, query)
    expected_tangent = (
        evaluate(coefficient_tangent, query)
        + query_tangent
        * bspline_evaluate(
            knots,
            coefficients,
            query,
            degree=degree,
            derivative_order=1,
        ).values
    )
    assert np.allclose(np.asarray(values), np.asarray(expected_values), atol=1e-12)
    assert np.allclose(np.asarray(tangent), np.asarray(expected_tangent), atol=1e-12)

    coefficient_gradient = jax.grad(
        lambda candidate: jnp.sum(evaluate(candidate, query))
    )(coefficients)
    active_rows = np.flatnonzero(
        np.any(np.abs(np.asarray(coefficient_gradient)) > 1e-12, axis=1)
    )
    assert active_rows.size == degree + 1
    assert np.all(np.diff(active_rows) == 1)


def test_bspline_case_shape_and_complex_payloads_are_preserved():
    degree = 2
    knots = np.asarray(_open_knots(degree, 6))
    query = jnp.asarray([[0.0, 0.4, 1.0], [0.1, 0.7, 0.9]])
    real = jnp.arange(2 * 6 * 4, dtype=float).reshape((2, 6, 2, 2))
    controls = real + 1j * (real + 0.5)
    stencil = bspline_stencil(
        knots,
        query,
        degree=degree,
        case_shape=(2,),
    )
    actual = apply_gather_stencil(controls, stencil).values

    expected = np.stack(
        tuple(
            SciPyBSpline(knots, np.asarray(controls[index]), degree)(
                np.asarray(query[index])
            )
            for index in range(2)
        )
    )
    assert stencil.relation.input_shape == (2, 6)
    assert stencil.relation.output_shape == (2, 3)
    assert actual.shape == (2, 3, 2, 2)
    assert np.allclose(np.asarray(actual), expected)


def test_bspline_bounds_modes_are_explicit():
    degree = 2
    knots = np.asarray(_open_knots(degree, 6))
    controls = jnp.arange(6, dtype=float)
    query = jnp.asarray([-0.2, 0.25, 1.2])
    oracle = SciPyBSpline(knots, np.asarray(controls), degree)
    clipped_expected = oracle(np.clip(np.asarray(query), 0.0, 1.0))

    clipped_stencil = bspline_stencil(knots, query, degree=degree, bounds="clip")
    clipped = apply_gather_stencil(controls, clipped_stencil)
    assert np.all(clipped.support)
    assert np.allclose(np.asarray(clipped.values), clipped_expected)

    def clipped_value(value):
        return apply_gather_stencil(
            controls,
            bspline_stencil(knots, value, degree=degree, bounds="clip"),
        ).values

    assert float(jax.grad(clipped_value)(jnp.asarray(1.0))) == pytest.approx(
        float(oracle(1.0, nu=1)),
        abs=1e-10,
    )
    assert float(jax.grad(clipped_value)(jnp.asarray(1.2))) == pytest.approx(0.0)

    filled_stencil = bspline_stencil(knots, query, degree=degree, bounds="fill")
    filled = apply_gather_stencil(controls, filled_stencil)
    assert np.array_equal(np.asarray(filled.support), [False, True, False])
    assert np.allclose(
        np.asarray(filled.values),
        np.where(np.asarray(filled.support), clipped_expected, 0.0),
    )

    extrapolated = apply_gather_stencil(
        controls,
        bspline_stencil(knots, query, degree=degree, bounds="extrapolate"),
    ).values
    expected = oracle(np.asarray(query))
    assert np.allclose(np.asarray(extrapolated), expected)

    with pytest.raises(eqx.EquinoxRuntimeError, match="outside"):
        stencil = bspline_stencil(knots, query, degree=degree, bounds="error")
        jax.block_until_ready(stencil.weights)


def test_bspline_validation_is_transformation_safe():
    knots = jnp.asarray(_open_knots(2, 5))
    invalid_degree: Any = 2.0

    with pytest.raises(TypeError, match="real-valued"):
        bspline_stencil(knots, jnp.asarray(0.2 + 0.1j), degree=2)
    with pytest.raises(TypeError, match="degree must be an integer"):
        bspline_stencil(knots, 0.2, degree=invalid_degree)
    with pytest.raises(ValueError, match="between zero and the degree"):
        bspline_stencil(knots, 0.2, degree=2, derivative_order=3)
    with pytest.raises(ValueError, match="non-empty"):
        bspline_stencil(knots, jnp.empty((0,)), degree=2)
    with pytest.raises(ValueError, match="begin with case_shape"):
        bspline_stencil(knots, jnp.ones((2, 3)), degree=2, case_shape=(3,))
    with pytest.raises(ValueError, match=r"degree \+ 1"):
        bspline_stencil(jnp.asarray([0.0, 0.0, 1.0, 1.0]), 0.5, degree=2)

    invalid_knots = (
        jnp.asarray([0.0, 0.0, 0.7, 0.4, 1.0, 1.0]),
        jnp.asarray([0.0, 0.0, jnp.nan, 0.7, 1.0, 1.0]),
        jnp.zeros((6,)),
    )
    for invalid in invalid_knots:
        with pytest.raises(eqx.EquinoxRuntimeError, match="B-spline knots"):
            stencil = eqx.filter_jit(lambda value: bspline_stencil(value, 0.5, degree=2))(
                invalid
            )
            jax.block_until_ready(stencil.weights)
