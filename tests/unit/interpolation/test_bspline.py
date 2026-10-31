#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction
from math import comb
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from scipy.interpolate import BSpline as SciPyBSpline

from phydrax._interpolation import (
    apply_gather_stencil,
    bezier_refinement,
    BoundsMode,
    bspline_evaluate,
    bspline_jet_stencil,
    bspline_stencil,
    exact_bezier_refinement,
    rational_bspline_curve_continuity,
    rational_bspline_curve_spans,
    rational_bspline_surface_spans,
    RationalSplineJet,
    tensor_bspline_span_extraction,
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
    knots: Any,
    query: Any,
    *,
    degree: int,
    derivative_order: int = 0,
    bounds: BoundsMode = "error",
) -> Any:
    control_count = len(knots) - degree - 1
    stencil = bspline_stencil(
        knots,
        query,
        degree=degree,
        derivative_order=derivative_order,
        bounds=bounds,
    )
    return apply_gather_stencil(jnp.eye(control_count), stencil).values, stencil


def test_bspline_scenario_1() -> None:
    for degree in range(6):
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
    controls = jnp.arange(50, dtype="float64").reshape((5, 5, 2)) / 13.0
    messages = jnp.cos(jnp.arange(8, dtype="float64")).reshape((2, 2, 2))

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
    knots = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    stencil = bspline_jet_stencil(
        knots,
        0.4,
        degree=1,
        maximum_order=1,
    )

    with pytest.raises(TypeError, match="components must be integers"):
        TensorBSplineJetPlan(
            (stencil,),
            # ty: ignore[invalid-argument-type]
            multi_indices=((0,), (0.5,)),
        )
    knots = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    stencils = tuple(
        bspline_jet_stencil(
            knots,
            0.4,
            degree=1,
            maximum_order=1,
        )
        for _ in range(2)
    )
    requested = ((0, 0), (1, 1), (1, 0), (0, 1))
    plan = TensorBSplineJetPlan(stencils, multi_indices=requested)
    rational = RationalSplineJet(plan, jnp.asarray([[1.0, 1.5], [2.0, 2.5]]))

    assert plan.multi_indices == ((0, 0), (1, 0), (0, 1), (1, 1))
    assert rational.derivative((1, 1)).shape == (4,)


def test_bspline_contracts() -> None:
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
    degree = 2
    knots = np.asarray(_open_knots(degree, 6))
    query = jnp.asarray([[0.0, 0.4, 1.0], [0.1, 0.7, 0.9]])
    real = jnp.arange(2 * 6 * 4, dtype="float64").reshape((2, 6, 2, 2))
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


def test_degree_one_rational_tensor_has_complete_nonuniform_weight_hessian() -> None:
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

    def dense(parameters: Any) -> Any:
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


def test_bspline_explicit_derivatives_match_autodiff_at_endpoints() -> None:
    degree = 3
    knots = jnp.asarray(_open_knots(degree, 7))
    controls = jnp.asarray([0.1, -0.7, 0.4, 1.2, -0.2, 0.9, 1.4])

    def evaluate(query: Any) -> Any:
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


def test_bspline_custom_jvp_combines_query_and_coefficient_tangents() -> None:
    degree = 3
    knots = jnp.asarray(_open_knots(degree, 7))
    coefficients = jnp.arange(14, dtype="float64").reshape((7, 2)) / 9.0
    coefficient_tangent = jnp.cos(coefficients)
    query = jnp.asarray(0.43)
    query_tangent = jnp.asarray(-0.7)

    def evaluate(coefficients_: Any, query_: Any) -> Any:
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


def test_bspline_bounds_modes_are_explicit() -> None:
    degree = 2
    knots = np.asarray(_open_knots(degree, 6))
    controls = jnp.arange(6, dtype="float64")
    query = jnp.asarray([-0.2, 0.25, 1.2])
    oracle = SciPyBSpline(knots, np.asarray(controls), degree)
    clipped_expected = oracle(np.clip(np.asarray(query), 0.0, 1.0))

    clipped_stencil = bspline_stencil(knots, query, degree=degree, bounds="clip")
    clipped = apply_gather_stencil(controls, clipped_stencil)
    assert np.all(clipped.support)
    assert np.allclose(np.asarray(clipped.values), clipped_expected)

    def clipped_value(value: Any) -> Any:
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


def test_tensor_bezier_extraction_is_exact_local_and_id_preserving() -> None:
    u_knots = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1, 3),
        Fraction(1),
        Fraction(1),
        Fraction(1),
        Fraction(1),
    )
    v_knots = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(2, 5),
        Fraction(1),
        Fraction(1),
        Fraction(1),
    )
    u_degree, v_degree = 3, 2
    u_span, v_span = 4, 3
    ids = tuple(1000 + 17 * row for row in range(20))
    anchor_ids, extraction = tensor_bspline_span_extraction(
        u_knots, v_knots, u_degree, v_degree, u_span, v_span, ids
    )
    expected_ids = tuple(
        ids[u_row * 4 + v_row]
        for u_row in range(u_span - u_degree, u_span + 1)
        for v_row in range(v_span - v_degree, v_span + 1)
    )
    assert anchor_ids == expected_ids
    assert len(extraction) == (u_degree + 1) * (v_degree + 1)
    assert all(len(row) == (u_degree + 1) * (v_degree + 1) for row in extraction)
    assert any(value == Fraction(1, 3) for row in extraction for value in row)
    assert all(isinstance(value, Fraction) for row in extraction for value in row)

    def cox(
        knots: tuple[Fraction, ...], degree: int, index: int, point: Fraction
    ) -> Fraction:
        if degree == 0:
            return Fraction(int(knots[index] <= point < knots[index + 1]))
        left_width = knots[index + degree] - knots[index]
        right_width = knots[index + degree + 1] - knots[index + 1]
        left = (
            (point - knots[index]) * cox(knots, degree - 1, index, point) / left_width
            if left_width
            else Fraction(0)
        )
        right = (
            (knots[index + degree + 1] - point)
            * cox(knots, degree - 1, index + 1, point)
            / right_width
            if right_width
            else Fraction(0)
        )
        return left + right

    def bernstein(degree: int, index: int, parameter: Fraction) -> Fraction:
        return (
            Fraction(comb(degree, index))
            * parameter**index
            * (1 - parameter) ** (degree - index)
        )

    u_lower, u_upper = u_knots[u_span], u_knots[u_span + 1]
    v_lower, v_upper = v_knots[v_span], v_knots[v_span + 1]
    for u_index in range(u_degree + 1):
        u_reference = Fraction(u_index + 1, u_degree + 2)
        u = u_lower + (u_upper - u_lower) * u_reference
        for v_index in range(v_degree + 1):
            v_reference = Fraction(v_index + 1, v_degree + 2)
            v = v_lower + (v_upper - v_lower) * v_reference
            for local_u in range(u_degree + 1):
                for local_v in range(v_degree + 1):
                    row = local_u * (v_degree + 1) + local_v
                    actual = sum(
                        (
                            extraction[row][bern_u * (v_degree + 1) + bern_v]
                            * bernstein(u_degree, bern_u, u_reference)
                            * bernstein(v_degree, bern_v, v_reference)
                            for bern_u in range(u_degree + 1)
                            for bern_v in range(v_degree + 1)
                        ),
                        Fraction(0),
                    )
                    expected = cox(
                        u_knots, u_degree, u_span - u_degree + local_u, u
                    ) * cox(v_knots, v_degree, v_span - v_degree + local_v, v)
                    assert actual == expected


def test_tensor_bezier_extraction_rejects_zero_spans_and_forged_ids() -> None:
    knots = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1),
        Fraction(1),
        Fraction(1),
    )
    ids = tuple(range(25))
    with pytest.raises(ValueError, match="strictly positive width"):
        tensor_bspline_span_extraction(knots, knots, 2, 2, 3, 2, ids)
    with pytest.raises(ValueError, match="unique and nonnegative"):
        tensor_bspline_span_extraction(knots, knots, 2, 2, 2, 2, (0, *ids[1:24], 0))


def test_exact_bezier_refinement_is_enclosed_by_boehm_bounds() -> None:
    knots = np.asarray((0.0, 0.0, 0.0, 0.0, 0.25, 1.0, 1.0, 1.0, 1.0))
    controls = np.asarray((0.2, -0.5, 1.1, 0.3, 2.0))
    exact_knots = np.asarray(
        tuple(Fraction.from_float(float(value)) for value in knots), dtype=object
    )
    exact_controls = np.asarray(
        tuple(Fraction.from_float(float(value)) for value in controls), dtype=object
    )
    exact_pieces, exact_bounds = exact_bezier_refinement(
        exact_controls, exact_knots, 3, 0
    )
    lower = np.nextafter(controls, -np.inf)
    upper = np.nextafter(controls, np.inf)
    bounded_refinement = bezier_refinement(lower, knots, 3, 0, control_upper=upper)
    assert len(bounded_refinement) == 3
    lower_pieces, upper_pieces, bounds = bounded_refinement
    assert exact_pieces.shape == lower_pieces.shape == upper_pieces.shape == (2, 4)
    assert tuple(tuple(row) for row in exact_bounds) == (
        (Fraction(0), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(1)),
    )
    assert np.array_equal(bounds, np.asarray(((0.0, 0.25), (0.25, 1.0))))
    for span in range(exact_pieces.shape[0]):
        for coefficient in range(exact_pieces.shape[1]):
            exact_value = exact_pieces[span, coefficient]
            assert (
                Fraction.from_float(float(lower_pieces[span, coefficient])) <= exact_value
            )
            assert exact_value <= Fraction.from_float(
                float(upper_pieces[span, coefficient])
            )


def test_rational_weight_gauge_is_a_jvp_vjp_hessian_null_direction() -> None:
    knots = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    point = jnp.asarray([0.29, 0.63])
    stencils = tuple(
        bspline_jet_stencil(knots, point[axis], degree=1, maximum_order=2)
        for axis in range(2)
    )
    plan = TensorBSplineJetPlan(stencils, maximum_order=2)
    raw_weights = jnp.asarray([[1.0, 1.7], [2.4, 4.1]])
    controls = jnp.asarray([[0.2, 1.3], [2.1, 4.7]])

    def value(scale: ArrayLike) -> Any:
        return RationalSplineJet(plan, raw_weights * jnp.asarray(scale)).apply(controls)

    scale = jnp.asarray(1.6)
    base_value = value(scale)
    scaled_value = value(7.25 * scale)
    np.testing.assert_allclose(scaled_value, base_value, rtol=2.0e-14, atol=2.0e-14)
    tangent = jax.jvp(value, (scale,), (jnp.ones_like(scale),))[1]
    pullback = jax.vjp(value, scale)[1](jnp.ones_like(base_value))[0]
    curvature = jax.hessian(lambda factor: jnp.sum(value(factor)))(scale)
    np.testing.assert_allclose(tangent, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(pullback, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(curvature, 0.0, atol=2.0e-14)


def test_rational_spline_jet_retains_the_raw_weight_bank() -> None:
    knots = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    point = jnp.asarray([0.29, 0.63])
    stencils = tuple(
        bspline_jet_stencil(knots, point[axis], degree=1, maximum_order=1)
        for axis in range(2)
    )
    plan = TensorBSplineJetPlan(stencils, maximum_order=1)
    weights = jnp.asarray([[0.7, 1.3], [2.1, 4.6]])

    rational = RationalSplineJet(plan, weights)

    np.testing.assert_array_equal(np.asarray(rational.weights), np.asarray(weights))


def test_exact_rational_curve_span_enclosures_include_two_derivatives() -> None:
    spans = rational_bspline_curve_spans(
        ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))),
        (Fraction(1), Fraction(2)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(1)),
        1,
    )
    assert len(spans) == 1
    span = spans[0]
    assert span.parameter_bounds == (Fraction(0), Fraction(1))
    assert span.positive_denominator_lower_bound == 1
    full = span.enclose((Fraction(0), Fraction(1)))
    assert full.positive_denominator
    assert full.denominator_bounds == (Fraction(1), Fraction(2))
    restricted = span.enclose((Fraction(1, 4), Fraction(1, 2)))
    assert restricted.denominator_bounds == (Fraction(5, 4), Fraction(3, 2))
    nonbinary_spans = rational_bspline_curve_spans(
        ((Fraction(0),), (Fraction(2),), (Fraction(4),)),
        (Fraction(1), Fraction(2), Fraction(3)),
        (Fraction(0), Fraction(0), Fraction(1, 3), Fraction(1), Fraction(1)),
        1,
    )
    assert tuple(span.span_index for span in nonbinary_spans) == (1, 2)
    assert tuple(span.parameter_bounds for span in nonbinary_spans) == (
        (Fraction(0), Fraction(1, 3)),
        (Fraction(1, 3), Fraction(1)),
    )

    start = span.enclose((Fraction(0), Fraction(0)))
    assert start.value_bounds == ((Fraction(0), Fraction(0)),) * 2
    assert start.first_derivative_bounds == (
        (Fraction(2), Fraction(2)),
        (Fraction(0), Fraction(0)),
    )
    assert start.second_derivative_bounds == (
        (Fraction(-4), Fraction(-4)),
        (Fraction(0), Fraction(0)),
    )
    end = span.enclose((Fraction(1), Fraction(1)))
    assert end.value_bounds == ((Fraction(1), Fraction(1)),) + (
        (Fraction(0), Fraction(0)),
    )
    assert end.first_derivative_bounds[0] == (Fraction(1, 2), Fraction(1, 2))
    assert end.second_derivative_bounds[0] == (Fraction(-1, 2), Fraction(-1, 2))
    outward = start.outward_float_bounds()
    assert outward.position_lower == (0.0, 0.0)
    assert outward.first_derivative_lower == (2.0, 0.0)
    assert outward.second_derivative_upper == (-4.0, 0.0)


def test_rational_curve_span_continuity_retains_knot_sides_and_closed_seam() -> None:
    closed_spans = rational_bspline_curve_spans(
        ((0, 0), (1, 0), (-1, 0), (0, 0)),
        (1, 1, 1, 1),
        (0, 0, 0, 0, 1, 1, 1, 1),
        3,
    )
    closed_report = rational_bspline_curve_continuity(closed_spans, closed=True)
    assert closed_report.junctions == ()
    assert closed_report.closed_seam is not None
    assert closed_report.closed_seam.observed_continuity_order == 1
    assert (
        closed_report.closed_seam.left_endpoint.value_bounds
        == closed_report.closed_seam.right_endpoint.value_bounds
    )
    assert (
        closed_report.closed_seam.left_endpoint.first_derivative_bounds
        == closed_report.closed_seam.right_endpoint.first_derivative_bounds
    )
    assert (
        closed_report.closed_seam.left_endpoint.second_derivative_bounds
        != closed_report.closed_seam.right_endpoint.second_derivative_bounds
    )

    discontinuous_spans = rational_bspline_curve_spans(
        ((0,), (1,), (4,), (5,)),
        (1, 1, 1, 1),
        (0, 0, Fraction(1, 2), Fraction(1, 2), 1, 1),
        1,
    )
    report = rational_bspline_curve_continuity(discontinuous_spans)
    assert len(discontinuous_spans) == 2
    assert len(report.junctions) == 1
    knot = report.junctions[0]
    assert knot.endpoint_parameters == (Fraction(1, 2), Fraction(1, 2))
    assert knot.guaranteed_continuity_order == -1
    assert knot.observed_continuity_order == -1
    assert knot.left_endpoint.value_bounds == ((Fraction(1), Fraction(1)),)
    assert knot.right_endpoint.value_bounds == ((Fraction(4), Fraction(4)),)


def test_algebraic_rational_curve_spans_preserve_exact_source_banks() -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    root_two = RadicalElement.sqrt(Fraction(2))
    interior_knot = root_two / 2
    exact_control_points = (
        (root_two,),
        (root_two + 1,),
        (root_two + 2,),
    )
    exact_weights = (Fraction(1), root_two + 1, Fraction(2))
    exact_knots = (
        Fraction(0),
        Fraction(0),
        interior_knot,
        Fraction(1),
        Fraction(1),
    )
    spans = rational_bspline_curve_spans(
        tuple(tuple(float(value) for value in row) for row in exact_control_points),
        tuple(float(value) for value in exact_weights),
        tuple(float(value) for value in exact_knots),
        1,
        exact_control_points=exact_control_points,
        exact_weights=exact_weights,
        exact_knots=exact_knots,
    )

    assert tuple(span.parameter_bounds for span in spans) == (
        (Fraction(0), interior_knot),
        (interior_knot, Fraction(1)),
    )
    assert spans[0].homogeneous_bernstein_coefficients[0][0] == root_two
    assert all(span.positive_denominator_lower_bound > 0 for span in spans)
    start = spans[0].enclose((Fraction(0), Fraction(0)))
    assert start.value_bounds[0][0] <= root_two <= start.value_bounds[0][1]
    left_knot = spans[0].enclose((interior_knot, interior_knot))
    right_knot = spans[1].enclose((interior_knot, interior_knot))
    knot_value = exact_control_points[1][0]
    assert left_knot.value_bounds[0][0] <= knot_value <= left_knot.value_bounds[0][1]
    assert right_knot.value_bounds[0][0] <= knot_value <= right_knot.value_bounds[0][1]
    continuity = rational_bspline_curve_continuity(spans)
    assert continuity.junctions[0].observed_continuity_order >= 0
    tiny = Fraction(1, 10**80)
    nearby_control_points = (
        (root_two + tiny,),
        exact_control_points[1],
        exact_control_points[2],
    )
    assert float(nearby_control_points[0][0]) == float(root_two)
    nearby_spans = rational_bspline_curve_spans(
        (
            (float(root_two),),
            (float(root_two + 1),),
            (float(root_two + 2),),
        ),
        tuple(float(value) for value in exact_weights),
        tuple(float(value) for value in exact_knots),
        1,
        exact_control_points=nearby_control_points,
        exact_weights=exact_weights,
        exact_knots=exact_knots,
    )
    assert nearby_spans[0].source_id != spans[0].source_id
    assert (
        nearby_spans[0].homogeneous_bernstein_coefficients[0][0]
        != spans[0].homogeneous_bernstein_coefficients[0][0]
    )


def test_exact_rational_surface_spans_enclose_position_and_jacobian() -> None:
    knots = (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    control_points = (
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
        ),
        (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(1), Fraction(1)),
        ),
    )
    weights = ((Fraction(1), Fraction(2)), (Fraction(3), Fraction(4)))
    spans = rational_bspline_surface_spans(control_points, weights, knots, knots, 1, 1)

    assert len(spans) == 1
    span = spans[0]
    assert span.span_indices == (1, 1)
    assert span.coefficient_rows == (0, 1, 2, 3)
    enclosure = span.enclose(
        (
            (Fraction(1, 4), Fraction(3, 4)),
            (Fraction(1, 5), Fraction(4, 5)),
        )
    )
    assert enclosure.positive_denominator

    def evaluate(
        u: Fraction, v: Fraction
    ) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]]:
        basis_u = (1 - u, u)
        basis_v = (1 - v, v)
        derivative_u = (Fraction(-1), Fraction(1))
        derivative_v = (Fraction(-1), Fraction(1))
        denominator = sum(
            (weights[i][j] * basis_u[i] * basis_v[j] for i in range(2) for j in range(2)),
            Fraction(0),
        )
        denominator_u = sum(
            (
                weights[i][j] * derivative_u[i] * basis_v[j]
                for i in range(2)
                for j in range(2)
            ),
            Fraction(0),
        )
        denominator_v = sum(
            (
                weights[i][j] * basis_u[i] * derivative_v[j]
                for i in range(2)
                for j in range(2)
            ),
            Fraction(0),
        )
        value: list[Fraction] = []
        du: list[Fraction] = []
        dv: list[Fraction] = []
        for component in range(3):
            numerator = sum(
                (
                    weights[i][j]
                    * control_points[i][j][component]
                    * basis_u[i]
                    * basis_v[j]
                    for i in range(2)
                    for j in range(2)
                ),
                Fraction(0),
            )
            numerator_u = sum(
                (
                    weights[i][j]
                    * control_points[i][j][component]
                    * derivative_u[i]
                    * basis_v[j]
                    for i in range(2)
                    for j in range(2)
                ),
                Fraction(0),
            )
            numerator_v = sum(
                (
                    weights[i][j]
                    * control_points[i][j][component]
                    * basis_u[i]
                    * derivative_v[j]
                    for i in range(2)
                    for j in range(2)
                ),
                Fraction(0),
            )
            value.append(numerator / denominator)
            du.append(
                (numerator_u * denominator - numerator * denominator_u) / denominator**2
            )
            dv.append(
                (numerator_v * denominator - numerator * denominator_v) / denominator**2
            )
        return tuple(value), tuple(du), tuple(dv)

    for u in (Fraction(1, 4), Fraction(2, 5), Fraction(3, 4)):
        for v in (Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)):
            values = evaluate(u, v)
            for observed, bounds in zip(
                values,
                (enclosure.value_bounds, enclosure.du_bounds, enclosure.dv_bounds),
                strict=True,
            ):
                assert all(
                    lower <= value <= upper
                    for value, (lower, upper) in zip(observed, bounds, strict=True)
                )


def test_surface_span_enclosures_keep_internal_knot_pieces_separate() -> None:
    u_knots = (
        Fraction(0),
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
        Fraction(1),
    )
    v_knots = (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    control_points = tuple(
        (
            (Fraction(index, 2), Fraction(0), Fraction(0)),
            (Fraction(index, 2), Fraction(1), Fraction(0)),
        )
        for index in range(3)
    )
    weights = tuple((Fraction(1), Fraction(1)) for _ in range(3))
    spans = rational_bspline_surface_spans(
        control_points, weights, u_knots, v_knots, 1, 1
    )

    assert tuple(span.span_indices for span in spans) == ((1, 1), (2, 1))
    assert tuple(span.parameter_bounds[0] for span in spans) == (
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1)),
    )
    resolution = Fraction(1, 2**40)
    for span in spans:
        enclosure = span.enclose(span.parameter_bounds)
        # One span is affine in u, so each enclosure tightly contains its slope.
        for bounds, expected in zip(
            (*enclosure.du_bounds, *enclosure.dv_bounds),
            (
                Fraction(1),
                Fraction(0),
                Fraction(0),
                Fraction(0),
                Fraction(1),
                Fraction(0),
            ),
            strict=True,
        ):
            assert bounds[0] <= expected <= bounds[1]
            assert bounds[1] - bounds[0] < resolution


def test_exact_surface_spans_preserve_subfloat_source_coefficients() -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    knots = (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    tiny = RadicalElement.sqrt(Fraction(2, 10**800))
    exact_control_points = (
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
        ),
        (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(1), tiny),
        ),
    )
    zero_control_points = (
        exact_control_points[0],
        (
            exact_control_points[1][0],
            (Fraction(1), Fraction(1), Fraction(0)),
        ),
    )
    numeric_control_points = tuple(
        tuple(tuple(float(value) for value in point) for point in row)
        for row in exact_control_points
    )
    numeric_knots = tuple(float(value) for value in knots)
    weights = ((1.0, 1.0), (1.0, 1.0))
    spans = rational_bspline_surface_spans(
        numeric_control_points,
        weights,
        numeric_knots,
        numeric_knots,
        1,
        1,
        exact_control_points=exact_control_points,
        exact_weights=((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1))),
        exact_u_knots=knots,
        exact_v_knots=knots,
    )
    zero_spans = rational_bspline_surface_spans(
        numeric_control_points,
        weights,
        numeric_knots,
        numeric_knots,
        1,
        1,
        exact_control_points=zero_control_points,
        exact_weights=((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1))),
        exact_u_knots=knots,
        exact_v_knots=knots,
    )

    assert float(tiny) == 0.0
    assert spans[0].source_id != zero_spans[0].source_id
    assert isinstance(
        spans[0].homogeneous_bernstein_coefficients[-1][-1][-2], RadicalElement
    )
    full_enclosure = spans[0].enclose(
        ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)))
    )
    assert full_enclosure.value_bounds[2][1] > 0


def test_interval_span_restriction_contains_exact_restricted_nets() -> None:
    from phydrax._interpolation._bspline_extraction import (
        _bernstein_restrict,
        _interval_net,
        _interval_tensor_restrict,
    )

    generator = np.random.default_rng(20261031)
    degree_u, degree_v = 5, 3
    knots_u = (Fraction(0),) * (degree_u + 1) + (Fraction(1),) * (degree_u + 1)
    knots_v = (Fraction(0),) * (degree_v + 1) + (Fraction(1),) * (degree_v + 1)
    control_points = tuple(
        tuple(
            tuple(Fraction(float(value)) for value in generator.normal(size=3))
            for _ in range(degree_v + 1)
        )
        for _ in range(degree_u + 1)
    )
    weights = tuple(
        tuple(
            Fraction(float(value)) for value in generator.uniform(0.5, 2.0, degree_v + 1)
        )
        for _ in range(degree_u + 1)
    )
    (span,) = rational_bspline_surface_spans(
        control_points, weights, knots_u, knots_v, degree_u, degree_v
    )
    nets = (*span._du_numerators, *span._dv_numerators)

    def exact_restrict(
        net: Any,
        lower_u: Fraction,
        upper_u: Fraction,
        lower_v: Fraction,
        upper_v: Fraction,
    ) -> tuple[tuple[Any, ...], ...]:
        columns = tuple(
            _bernstein_restrict(tuple(row[column] for row in net), lower_u, upper_u)
            for column in range(len(net[0]))
        )
        rows = tuple(
            tuple(column[row] for column in columns) for row in range(len(columns[0]))
        )
        return tuple(_bernstein_restrict(row, lower_v, upper_v) for row in rows)

    for _ in range(12):
        lower_u, upper_u = sorted(
            Fraction(float(value)) for value in generator.uniform(size=2)
        )
        lower_v, upper_v = sorted(
            Fraction(float(value)) for value in generator.uniform(size=2)
        )
        for net in nets:
            exact = exact_restrict(net, lower_u, upper_u, lower_v, upper_v)
            lower, upper = _interval_tensor_restrict(
                _interval_net(net), lower_u, upper_u, lower_v, upper_v
            )
            assert lower.shape == upper.shape == (len(exact), len(exact[0]))
            for row, exact_row in enumerate(exact):
                for column, value in enumerate(exact_row):
                    assert Fraction(float(lower[row, column])) <= value
                    assert value <= Fraction(float(upper[row, column]))
                    assert float(upper[row, column] - lower[row, column]) < 1e-9 * (
                        1.0 + abs(float(value))
                    )
        enclosure = span.enclose_derivatives(((lower_u, upper_u), (lower_v, upper_v)))
        hull = tuple(
            (
                min(
                    value
                    for row in exact_restrict(net, lower_u, upper_u, lower_v, upper_v)
                    for value in row
                ),
                max(
                    value
                    for row in exact_restrict(net, lower_u, upper_u, lower_v, upper_v)
                    for value in row
                ),
            )
            for net in nets
        )
        assert enclosure.positive_denominator
        for (low, high), exact_hull in zip(
            (*enclosure.du_bounds, *enclosure.dv_bounds), hull, strict=True
        ):
            denominator_squared = (
                enclosure.denominator_bounds[0] ** 2,
                enclosure.denominator_bounds[1] ** 2,
            )
            for numerator in exact_hull:
                for denominator in denominator_squared:
                    assert low <= numerator / denominator <= high
