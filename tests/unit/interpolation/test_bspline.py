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
    bspline_jet_stencil,
    bspline_stencil,
    RationalSplineJet,
    TensorBSplineJetPlan,
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
def test_local_spline_basis_and_derivatives_match_scipy(degree):
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


def test_bspline_jet_uses_explicit_spans_and_zeroes_orders_above_degree():
    degree = 2
    control_count = 6
    knots = _open_knots(degree, control_count)
    query = np.asarray([[0.08, 0.21, 0.39], [0.57, 0.76, 0.94]])
    spans = np.clip(
        np.searchsorted(knots, query, side="right") - 1,
        degree,
        control_count - 1,
    ).astype(np.int32)
    jet = bspline_jet_stencil(
        knots,
        jnp.asarray(query),
        degree=degree,
        maximum_order=degree + 3,
        spans=jnp.asarray(spans),
    )
    oracle = SciPyBSpline(knots, np.eye(control_count), degree)

    assert jet.indices.shape == query.shape + (degree + 1,)
    assert jet.jets.shape == query.shape + (degree + 4, degree + 1)
    for order in range(degree + 1):
        dense = apply_gather_stencil(
            jnp.eye(control_count),
            jet.derivative(order),
        ).values
        assert np.allclose(
            np.asarray(dense),
            oracle(query, nu=order),
            rtol=2e-10,
            atol=2e-10,
        )
    assert np.array_equal(
        np.asarray(jet.jets[..., degree + 1 :, :]),
        np.zeros(query.shape + (3, degree + 1)),
    )


def test_tensor_bspline_complete_hessian_multi_indices_have_exact_transposes():
    degree = 2
    knots = jnp.asarray(_open_knots(degree, 5))
    u_stencil = bspline_jet_stencil(
        knots,
        jnp.asarray([0.17, 0.63]),
        degree=degree,
        maximum_order=2,
    )
    v_stencil = bspline_jet_stencil(
        knots,
        jnp.asarray([0.31, 0.82]),
        degree=degree,
        maximum_order=2,
    )
    plan = TensorBSplineJetPlan((u_stencil, v_stencil), maximum_order=2)
    controls = jnp.arange(50, dtype=float).reshape((5, 5, 2)) / 13.0
    messages = jnp.cos(jnp.arange(8, dtype=float)).reshape((2, 2, 2))

    assert plan.multi_indices == (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
    )
    assert plan.gradient(controls).shape == (2, 2, 2, 2)
    assert plan.hessian(controls).shape == (2, 2, 2, 2, 2)
    for multi_index in plan.multi_indices:
        forward_pairing = jnp.vdot(plan.apply(controls, multi_index), messages)
        transpose_pairing = jnp.vdot(
            controls,
            plan.transpose(messages, multi_index),
        )
        assert float(forward_pairing) == pytest.approx(
            float(transpose_pairing),
            rel=2e-12,
            abs=2e-12,
        )


def test_degree_one_rational_tensor_has_complete_nonuniform_weight_hessian():
    knots = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    point = jnp.asarray([0.37, 0.58])
    stencils = tuple(
        bspline_jet_stencil(
            knots,
            point[axis],
            degree=1,
            maximum_order=2,
        )
        for axis in range(2)
    )
    plan = TensorBSplineJetPlan(stencils, maximum_order=2)
    weights = jnp.asarray([[1.0, 1.7], [2.4, 4.1]])
    controls = jnp.asarray([[0.2, 1.3], [2.1, 4.7]])
    rational = RationalSplineJet(plan, weights)

    def dense(parameters):
        u_basis = jnp.asarray([1.0 - parameters[0], parameters[0]])
        v_basis = jnp.asarray([1.0 - parameters[1], parameters[1]])
        weighted_basis = u_basis[:, None] * v_basis[None, :] * weights
        return jnp.sum(weighted_basis * controls) / jnp.sum(weighted_basis)

    expected_gradient = jax.jacfwd(dense)(point)
    expected_hessian = jax.jacfwd(jax.jacfwd(dense))(point)
    actual_gradient = rational.gradient_apply(controls)
    actual_hessian = rational.hessian_apply(controls)

    assert rational.values.shape == (4,)
    assert rational.gradients.shape == (4, 2)
    assert rational.hessians.shape == (4, 2, 2)
    assert np.allclose(np.asarray(actual_gradient), np.asarray(expected_gradient))
    assert np.allclose(np.asarray(actual_hessian), np.asarray(expected_hessian))
    assert np.allclose(np.asarray(actual_hessian), np.asarray(actual_hessian.T))
    assert np.all(np.abs(np.asarray(actual_hessian)[(0, 0, 1), (0, 1, 1)]) > 1e-8)

    message = jnp.asarray(1.7)
    assert float(jnp.vdot(rational.apply(controls), message)) == pytest.approx(
        float(jnp.vdot(controls, rational.transpose(message))),
        rel=2e-12,
        abs=2e-12,
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
    above_degree = bspline_stencil(knots, 0.2, degree=2, derivative_order=3)
    assert np.array_equal(np.asarray(above_degree.weights), np.zeros((3,)))
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
