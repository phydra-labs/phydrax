#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._interpolation import (
    BoundsMode,
    bspline_jet_stencil,
    RationalSplineJet,
    TensorBSplineJetPlan,
)
from phydrax.geometry._interval_enclosure import (
    device_interval_evaluation_error,
    interval_dot,
    interval_midpoint_radius,
    IntervalExtensionError,
    prepare_interval_function,
)
from phydrax.geometry.brep._differentiable import _spline_seam_source_box
from phydrax.geometry.brep._patches import (
    BSplineSurfacePatch,
    device_bspline_surface_box,
)
from phydrax.geometry.brep._placed import PlacedSurface
from phydrax.geometry.brep._spline_box import device_bspline_surface_bounds


pytestmark = pytest.mark.strict_jax


def _scan_polynomial(point: Array) -> Array:
    coefficients = jnp.asarray((2.0, 3.0, -1.0), dtype=jnp.float64)

    def step(carry: Array, coefficient: Array) -> tuple[Array, Array]:
        result = carry * point[0] + coefficient
        return result, result

    result, _ = jax.lax.scan(step, jnp.asarray(0.0, dtype=jnp.float64), coefficients)
    return result


def test_host_nominal_scan_preserves_batch_and_recurrence_order() -> None:
    prepared = prepare_interval_function(_scan_polynomial, 1, batch_capacity=2)
    points = np.asarray(((0.25,), (0.5,), (0.75,)), dtype=np.float64)
    values = prepared.evaluate_host_points(points)
    # Dyadic coordinates make these degree-two values representable without rounding.
    expected = (2.0 * points[:, 0] + 3.0) * points[:, 0] - 1.0
    np.testing.assert_array_equal(values, expected)
    lower, upper = prepared.evaluate(points, points)
    assert np.all(lower <= expected)
    assert np.all(expected <= upper)


def test_host_nominal_constant_gather_retains_index_order() -> None:
    def function(point: Array) -> Array:
        constants = jnp.asarray((3.0, 5.0, 7.0), dtype=jnp.float64)
        indices = jnp.asarray((2, 0, 1), dtype=jnp.int32)
        return constants[indices] * point[0]

    prepared = prepare_interval_function(function, 1, batch_capacity=2)
    values = prepared.evaluate_host_points(
        np.asarray(((0.5,), (1.0,), (2.0,)), dtype=np.float64)
    )
    np.testing.assert_array_equal(
        values,
        np.asarray(
            ((3.5, 1.5, 2.5), (7.0, 3.0, 5.0), (14.0, 6.0, 10.0)), dtype=np.float64
        ),
    )


def test_ordered_interval_dot_contains_rational_reference() -> None:
    first = np.asarray((2.0**53, 1.0, -(2.0**53)), dtype=np.float64)
    second = np.ones((3,), dtype=np.float64)
    lower, upper = interval_dot((first, first), (second, second))
    reference = sum(
        (
            Fraction(float(left)) * Fraction(float(right))
            for left, right in zip(first, second, strict=True)
        ),
        Fraction(0),
    )
    assert Fraction(float(lower)) <= reference <= Fraction(float(upper))


def test_midpoint_radius_contains_both_endpoints() -> None:
    lower = np.asarray((-3.0, 0.25), dtype=np.float64)
    upper = np.asarray((5.0, 0.75), dtype=np.float64)
    middle, radius = interval_midpoint_radius((lower, upper))
    assert np.all(np.subtract(middle, radius) <= lower)
    assert np.all(upper <= np.add(middle, radius))


def test_device_error_encloses_affine_rational_rounding() -> None:
    matrix = jnp.asarray(((3.0, 1.0), (-2.0, 5.0)), dtype=jnp.float64)
    offset = jnp.asarray((0.1, -0.2), dtype=jnp.float64)
    point = jnp.asarray((0.3, 0.4), dtype=jnp.float64)

    def function(coordinate: Array) -> Array:
        return matrix @ coordinate + offset

    realized = np.asarray(jax.device_get(function(point)))
    error = np.asarray(jax.device_get(device_interval_evaluation_error(function, point)))
    reference = (
        3 * Fraction(0.3) + Fraction(0.4) + Fraction(0.1),
        -2 * Fraction(0.3) + 5 * Fraction(0.4) + Fraction(-0.2),
    )
    assert np.all(np.isfinite(error))
    for value, exact, bound in zip(realized, reference, error, strict=True):
        assert abs(Fraction(float(value)) - exact) <= Fraction(float(bound))


def test_device_error_retains_dynamic_callback_leaves() -> None:
    point = jnp.asarray((0.3, 0.4), dtype=jnp.float64)

    @jax.jit
    def evaluate(scale: Array) -> Array:
        def function(coordinate: Array) -> Array:
            return scale * coordinate

        return device_interval_evaluation_error(function, point)

    first = np.asarray(jax.device_get(evaluate(jnp.asarray(1.0, dtype=jnp.float64))))
    second = np.asarray(jax.device_get(evaluate(jnp.asarray(1024.0, dtype=jnp.float64))))
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))
    assert np.all(second > first)


def test_device_error_includes_declared_input_rounding() -> None:
    point = jnp.asarray((0.3,), dtype=jnp.float64)
    lower = jnp.nextafter(point, -jnp.inf)
    upper = jnp.nextafter(point, jnp.inf)

    def function(coordinate: Array) -> Array:
        return 8.0 * coordinate

    singleton = np.asarray(
        jax.device_get(device_interval_evaluation_error(function, point))
    )
    widened = np.asarray(
        jax.device_get(
            device_interval_evaluation_error(function, point, input_bounds=(lower, upper))
        )
    )
    assert np.all(np.isfinite(widened))
    assert np.all(widened >= singleton)
    # A binary exponent shift introduces no endpoint displacement rounding.
    displacement = np.asarray(jax.device_get(8.0 * (upper - lower)))
    assert np.all(widened >= displacement)


def test_device_equal_singleton_subtraction_keeps_branch_decidable() -> None:
    repeated = jnp.asarray((1.0, 1.0), dtype=jnp.float64)
    point = jnp.asarray((0.25,), dtype=jnp.float64)

    def function(coordinate: Array) -> Array:
        denominator = repeated[1] - repeated[0]
        return jnp.where(denominator == 0.0, coordinate, coordinate / denominator)

    error = np.asarray(jax.device_get(device_interval_evaluation_error(function, point)))
    assert np.all(np.isfinite(error))


def test_device_error_refuses_unsupported_primitive() -> None:
    def function(point: Array) -> Array:
        return jnp.arctan2(point[0], point[1])

    with pytest.raises(IntervalExtensionError, match="interval.primitive"):
        device_interval_evaluation_error(
            function, jnp.asarray((0.25, 0.5), dtype=jnp.float64)
        )


def test_device_integer_power_error_encloses_every_lowered_multiply() -> None:
    point = jnp.asarray((1.001,), dtype=jnp.float64)

    def function(coordinate: Array) -> Array:
        return coordinate**127

    realized = float(np.asarray(jax.device_get(function(point)))[0])
    error = float(
        np.asarray(jax.device_get(device_interval_evaluation_error(function, point)))[0]
    )
    reference = Fraction(1.001) ** 127
    assert np.isfinite(error)
    assert abs(Fraction(realized) - reference) <= Fraction(error)


def _nonunit_domain_map(point: Array) -> Array:
    parameter = eqx.error_if(
        point[0],
        (point[0] < 2.0) | (point[0] > 4.0),
        "nonunit.parameter must lie in [2, 4]; got an outside query",
    )
    return jnp.stack((parameter, parameter * parameter))


def test_preparation_does_not_invent_a_unit_parameter_domain() -> None:
    prepared = prepare_interval_function(_nonunit_domain_map, 1, batch_capacity=2)
    points = np.asarray(((2.5,), (3.0,)), dtype=np.float64)
    values = prepared.evaluate_host_points(points)
    np.testing.assert_array_equal(values, ((2.5, 6.25), (3.0, 9.0)))
    lower, upper = prepared.evaluate(points, points)
    assert np.all(lower <= values)
    assert np.all(values <= upper)


@pytest.mark.parametrize(
    ("lower", "upper"),
    ((1.0, 1.0), (1.5, 2.5)),
    ids=("outside-domain", "unresolved-domain-guard"),
)
def test_interval_query_refuses_a_possible_opaque_guard_branch(
    lower: float, upper: float
) -> None:
    prepared = prepare_interval_function(_nonunit_domain_map, 1, batch_capacity=2)
    with pytest.raises(IntervalExtensionError, match="interval.primitive"):
        prepared.evaluate(
            np.asarray(((lower,),), dtype=np.float64),
            np.asarray(((upper,),), dtype=np.float64),
        )


def test_host_scatter_overwrite_retains_fixed_update_order() -> None:
    def function(point: Array) -> Array:
        table = jnp.zeros((2, 2), dtype=jnp.float64)
        table = table.at[0, 0].set(1.0)
        table = table.at[1, 0].set(point[0])
        table = table.at[1, 1].set(2.0 * point[0])
        return table

    prepared = prepare_interval_function(function, 1, batch_capacity=2)
    points = np.asarray(((0.25,), (0.75,)), dtype=np.float64)
    values = prepared.evaluate_host_points(points)
    np.testing.assert_array_equal(
        values,
        (((1.0, 0.0), (0.25, 0.5)), ((1.0, 0.0), (0.75, 1.5))),
    )
    lower, upper = prepared.evaluate(points, points)
    assert np.all(lower <= values)
    assert np.all(values <= upper)


def test_host_fixed_scatter_arithmetic_bounds_contain_exact_dyadic_values() -> None:
    def function(point: Array) -> Array:
        table = jnp.stack((point, 2.0 * point))
        table = table.at[0].multiply(4.0)
        table = table.at[1].add(1.0)
        return table

    prepared = prepare_interval_function(function, 1, batch_capacity=2)
    points = np.asarray(((0.25,), (0.75,)), dtype=np.float64)
    expected = np.asarray((((1.0,), (1.5,)), ((3.0,), (2.5,))), dtype=np.float64)
    np.testing.assert_array_equal(prepared.evaluate_host_points(points), expected)
    lower, upper = prepared.evaluate(points, points)
    assert np.all(lower <= expected)
    assert np.all(expected <= upper)


def test_index_extremum_bounds_preserve_first_ties_and_uncertain_winners() -> None:
    def function(point: Array) -> Array:
        return jnp.stack((jnp.argmax(point), jnp.argmin(point))).astype(jnp.float64)

    point = jnp.asarray((0.0, 2.0, 2.0, 1.0), dtype=jnp.float64)
    singleton_error = np.asarray(
        jax.device_get(device_interval_evaluation_error(function, point))
    )
    assert np.all(np.isfinite(singleton_error))
    assert np.all(singleton_error < 1.0)
    lower = jnp.asarray((0.0, 1.5, 1.5, 1.0), dtype=jnp.float64)
    upper = jnp.asarray((0.0, 2.5, 2.5, 1.0), dtype=jnp.float64)
    uncertain_error = np.asarray(
        jax.device_get(
            device_interval_evaluation_error(function, point, input_bounds=(lower, upper))
        )
    )
    assert uncertain_error[0] >= 1.0
    assert uncertain_error[1] < 1.0
    prepared = prepare_interval_function(function, 4, batch_capacity=1)
    host_lower, host_upper = prepared.evaluate(
        np.asarray(jax.device_get(lower))[None, :],
        np.asarray(jax.device_get(upper))[None, :],
    )
    for first in (1.5, 2.0, 2.5):
        for second in (1.5, 2.0, 2.5):
            sample = np.asarray((0.0, first, second, 1.0))
            winner = np.asarray((np.argmax(sample), np.argmin(sample)))
            assert np.all(host_lower[0] <= winner)
            assert np.all(winner <= host_upper[0])


def _surface_box_test_patch(degree: int, *, interior_knots: bool) -> BSplineSurfacePatch:
    knots = (
        (0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0)
        if interior_knots
        else (0.0,) * (degree + 1) + (1.0,) * (degree + 1)
    )
    count = len(knots) - degree - 1
    axis = jnp.linspace(0.0, 1.0, count, dtype=jnp.float64)
    first, second = jnp.meshgrid(axis, axis, indexing="ij")
    controls = jnp.stack((first, second, first * second + first), axis=-1)
    weights = (
        jnp.ones((count, count), dtype=jnp.float64).at[count // 2, count // 2].set(1.01)
    )
    knot_array = jnp.asarray(knots, dtype=jnp.float64)
    return BSplineSurfacePatch(controls, weights, knot_array, knot_array, degree, degree)


def _surface_denominator(patch: BSplineSurfacePatch, point: Array) -> Array:
    plan = TensorBSplineJetPlan(
        tuple(
            bspline_jet_stencil(
                knots, point[axis], degree=degree, maximum_order=0, bounds="clip"
            )
            for axis, (knots, degree) in enumerate(
                ((patch.u_knots, patch.u_degree), (patch.v_knots, patch.v_degree))
            )
        ),
        maximum_order=0,
    )
    return RationalSplineJet(plan, patch.weights).denominator_jets[0]


@pytest.mark.parametrize(
    ("degree", "interior_knots", "weight_case", "box_lower", "box_upper"),
    (
        pytest.param(
            2,
            False,
            "positive",
            (0.25, 0.25),
            (0.75, 0.75),
            id="quadratic-single-span",
        ),
        pytest.param(
            1,
            True,
            "positive",
            (0.3, 0.3),
            (0.7, 0.7),
            id="interior-knot-crossing",
        ),
        pytest.param(
            1,
            True,
            "inactive-rows",
            (0.3, 0.1),
            (0.7, 0.2),
            id="inactive-nonpositive-weights",
        ),
        pytest.param(
            1,
            True,
            "inactive-cartesian",
            (0.3, 0.3),
            (0.7, 0.7),
            id="cartesian-active-window",
        ),
        pytest.param(
            2,
            False,
            "positive",
            (-0.25, 0.25),
            (0.25, 1.25),
            id="clipped-domain",
        ),
    ),
)
def test_positive_rational_surface_box_encloses_samples(
    degree: int,
    interior_knots: bool,
    weight_case: str,
    box_lower: tuple[float, float],
    box_upper: tuple[float, float],
) -> None:
    patch = _surface_box_test_patch(degree, interior_knots=interior_knots)
    if weight_case == "inactive-rows":
        weights = patch.weights.at[0, :].set(0.0).at[-1, :].set(-1.0)
        patch = eqx.tree_at(lambda value: value.weights, patch, weights)
    elif weight_case == "inactive-cartesian":
        weights = jnp.full_like(patch.weights, -1.0).at[0, 0].set(0.0)
        weights = weights.at[1:4, 1:4].set(patch.weights[1:4, 1:4])
        patch = eqx.tree_at(lambda value: value.weights, patch, weights)
    lower = jnp.asarray(box_lower, dtype=jnp.float64)
    upper = jnp.asarray(box_upper, dtype=jnp.float64)
    coordinate_lower, coordinate_upper, denominator_lower, denominator_upper = (
        device_bspline_surface_bounds(
            patch.control_points,
            patch.weights,
            patch.u_knots,
            patch.v_knots,
            lower,
            upper,
            u_degree=degree,
            v_degree=degree,
        )
    )
    assert np.all(np.isfinite(np.asarray(coordinate_lower)))
    assert np.all(np.isfinite(np.asarray(coordinate_upper)))
    assert float(denominator_lower) > 0.0
    axes = tuple(
        np.linspace(low, high, 9) for low, high in zip(box_lower, box_upper, strict=True)
    )
    samples = jnp.asarray(
        np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 2)),
        dtype=jnp.float64,
    )
    positions = np.asarray(jax.device_get(patch.evaluate(samples)))
    denominators = np.asarray(
        jax.device_get(
            jax.vmap(lambda point: _surface_denominator(patch, point))(samples)
        )
    )
    assert np.all(np.asarray(coordinate_lower) <= positions)
    assert np.all(positions <= np.asarray(coordinate_upper))
    assert np.all(float(denominator_lower) <= denominators)
    assert np.all(denominators <= float(denominator_upper))
    wrapped_lower, wrapped_upper = device_bspline_surface_box(patch, lower, upper)
    np.testing.assert_array_equal(wrapped_lower, coordinate_lower)
    np.testing.assert_array_equal(wrapped_upper, coordinate_upper)


def test_rational_surface_singleton_box_contracts_with_parameter_box() -> None:
    patch = _surface_box_test_patch(2, interior_knots=False)
    point = jnp.asarray((0.375, 0.625), dtype=jnp.float64)
    singleton_lower, singleton_upper = device_bspline_surface_box(patch, point, point)
    narrow_lower, narrow_upper = device_bspline_surface_box(
        patch, point - 2.0**-20, point + 2.0**-20
    )
    wide_lower, wide_upper = device_bspline_surface_box(patch, point - 0.25, point + 0.25)
    position = patch.evaluate(point)
    assert np.all(np.asarray(singleton_lower) <= np.asarray(position))
    assert np.all(np.asarray(position) <= np.asarray(singleton_upper))
    singleton_width = np.asarray(singleton_upper - singleton_lower)
    narrow_width = np.asarray(narrow_upper - narrow_lower)
    wide_width = np.asarray(wide_upper - wide_lower)
    # Nested dyadic source boxes must narrow the certificate, not reuse the
    # entire active-control hull as a pointwise roundoff bound.
    assert np.all(singleton_width < narrow_width)
    assert np.all(narrow_width < wide_width)


@pytest.mark.parametrize("weight_case", ("endpoint-zero", "active-sign-change"))
def test_rational_surface_box_retains_zero_capable_refusal(weight_case: str) -> None:
    patch = _surface_box_test_patch(1, interior_knots=False)
    first_weight = 0.0 if weight_case == "endpoint-zero" else -1.0
    weights = jnp.asarray(((first_weight, first_weight), (1.0, 1.0)), dtype=jnp.float64)
    patch = eqx.tree_at(lambda value: value.weights, patch, weights)
    lower = jnp.asarray((0.0, 0.25), dtype=jnp.float64)
    upper = jnp.asarray((1.0, 0.75), dtype=jnp.float64)
    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match="Rational spline weights produced a zero or non-finite denominator.",
    ):
        jax.block_until_ready(device_bspline_surface_box(patch, lower, upper))


def _assert_float64_bitwise_equal(actual: Array, expected: Array) -> None:
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(actual)).view(np.uint64),
        np.asarray(jax.device_get(expected)).view(np.uint64),
    )


@pytest.mark.parametrize("transformation", ("primal", "jvp", "grad"))
def test_rational_surface_certification_preserves_bitwise_numerics(
    transformation: str,
) -> None:
    patch = _surface_box_test_patch(2, interior_knots=False)
    point = jnp.asarray((0.375, 0.625), dtype=jnp.float64)

    def evaluate(point: Array, controls: Array, weights: Array) -> Array:
        dynamic = eqx.tree_at(
            lambda value: (value.control_points, value.weights),
            patch,
            (controls, weights),
        )
        return dynamic.evaluate(point)

    def certified(
        point: Array, controls: Array, weights: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        dynamic = eqx.tree_at(
            lambda value: (value.control_points, value.weights),
            patch,
            (controls, weights),
        )
        return dynamic.evaluate(point), device_bspline_surface_box(dynamic, point, point)

    arguments = (point, patch.control_points, patch.weights)
    if transformation == "primal":
        expected = jax.jit(evaluate)(*arguments)
        actual, _ = jax.jit(certified)(*arguments)
        _assert_float64_bitwise_equal(actual, expected)
    elif transformation == "jvp":
        tangents = tuple(jnp.full_like(value, 0.125) for value in arguments)
        expected_value, expected_tangent = jax.jvp(evaluate, arguments, tangents)
        (actual_value, _), (actual_tangent, certificate_tangent) = jax.jvp(
            certified, arguments, tangents
        )
        _assert_float64_bitwise_equal(actual_value, expected_value)
        _assert_float64_bitwise_equal(actual_tangent, expected_tangent)
        for tangent in certificate_tangent:
            np.testing.assert_array_equal(tangent, jnp.zeros_like(tangent))
    else:

        def loss(point: Array, controls: Array, weights: Array) -> Array:
            return jnp.sum(evaluate(point, controls, weights))

        def certified_loss(
            point: Array, controls: Array, weights: Array
        ) -> tuple[Array, tuple[Array, Array]]:
            position, certificate = certified(point, controls, weights)
            return jnp.sum(position), certificate

        expected_value, expected_gradient = jax.value_and_grad(loss, argnums=(0, 1, 2))(
            *arguments
        )
        (actual_value, _), actual_gradient = jax.value_and_grad(
            certified_loss, argnums=(0, 1, 2), has_aux=True
        )(*arguments)
        _assert_float64_bitwise_equal(actual_value, expected_value)
        for actual, expected in zip(actual_gradient, expected_gradient, strict=True):
            _assert_float64_bitwise_equal(actual, expected)


@pytest.mark.parametrize(
    ("weights", "point", "bounds"),
    (
        pytest.param((0.0, 1.0), 0.0, "error", id="zero-active-weight"),
        pytest.param((2.0, 1.0), 2.0, "extrapolate", id="extrapolated-zero"),
    ),
)
def test_rational_denominator_zero_retains_numerical_refusal(
    weights: tuple[float, float], point: float, bounds: BoundsMode
) -> None:
    def function(coordinate: Array) -> Array:
        stencil = bspline_jet_stencil(
            jnp.asarray((0.0, 0.0, 1.0, 1.0), dtype=jnp.float64),
            coordinate[0],
            degree=1,
            maximum_order=0,
            bounds=bounds,
        )
        plan = TensorBSplineJetPlan((stencil,), maximum_order=0)
        return RationalSplineJet(plan, jnp.asarray(weights, dtype=jnp.float64)).values

    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match="Rational spline weights produced a zero or non-finite denominator.",
    ):
        jax.block_until_ready(
            eqx.filter_jit(function)(jnp.asarray((point,), dtype=jnp.float64))
        )


@pytest.mark.parametrize(
    ("weights", "lower", "upper", "bounds"),
    (
        pytest.param((0.0, 1.0), 0.0, 0.25, "error", id="zero-active-weight"),
        pytest.param((2.0, 1.0), 1.75, 2.25, "extrapolate", id="extrapolated-zero"),
    ),
)
@pytest.mark.parametrize("consumer", ("host", "device"))
def test_rational_denominator_zero_capable_box_retains_interval_refusal(
    weights: tuple[float, float],
    lower: float,
    upper: float,
    bounds: BoundsMode,
    consumer: str,
) -> None:
    def function(coordinate: Array) -> Array:
        stencil = bspline_jet_stencil(
            jnp.asarray((0.0, 0.0, 1.0, 1.0), dtype=jnp.float64),
            coordinate[0],
            degree=1,
            maximum_order=0,
            bounds=bounds,
        )
        plan = TensorBSplineJetPlan((stencil,), maximum_order=0)
        return RationalSplineJet(plan, jnp.asarray(weights, dtype=jnp.float64)).values

    if consumer == "host":
        prepared = prepare_interval_function(function, 1, batch_capacity=2)
        with pytest.raises(
            IntervalExtensionError,
            match=r"interval\.primitive must have a supported extension; got 'pure_callback'",
        ):
            prepared.evaluate(
                np.asarray(((lower,),), dtype=np.float64),
                np.asarray(((upper,),), dtype=np.float64),
            )
    else:
        with pytest.raises(
            eqx.EquinoxRuntimeError,
            match=(
                r"interval\.cond must exclude opaque callback branches throughout "
                "the input interval; got a possible callback branch"
            ),
        ):
            jax.block_until_ready(
                device_interval_evaluation_error(
                    function,
                    jnp.asarray(((lower + upper) * 0.5,), dtype=jnp.float64),
                    input_bounds=(
                        jnp.asarray((lower,), dtype=jnp.float64),
                        jnp.asarray((upper,), dtype=jnp.float64),
                    ),
                )
            )


def _quadratic_denominator_plan(point: Array) -> TensorBSplineJetPlan:
    knots = jnp.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=jnp.float64)
    return TensorBSplineJetPlan(
        tuple(
            bspline_jet_stencil(knots, point[axis], degree=2, maximum_order=2)
            for axis in range(2)
        ),
        maximum_order=2,
    )


def _produced_rational_denominator(point: Array, weights: Array) -> Array:
    return RationalSplineJet(_quadratic_denominator_plan(point), weights).denominator_jets


def _sum_rational_denominator(point: Array, weights: Array) -> Array:
    plan = _quadratic_denominator_plan(point)
    normalized_weights = weights / jnp.max(jnp.abs(weights))
    polynomial_jets = jnp.stack(
        tuple(plan.basis(index) for index in plan.multi_indices), axis=plan.jet_axis
    )
    weighted_jets = polynomial_jets * jnp.expand_dims(
        plan.gather(normalized_weights), axis=plan.jet_axis
    )
    return jnp.sum(weighted_jets, axis=-1)


def test_rational_denominator_matches_sum_eager_and_jit() -> None:
    point = jnp.asarray(((0.25, 0.375), (0.5, 0.625)), dtype=jnp.float64)
    weights = jnp.arange(1.0, 10.0, dtype=jnp.float64).reshape((3, 3))
    np.testing.assert_array_equal(
        _produced_rational_denominator(point, weights),
        _sum_rational_denominator(point, weights),
    )
    np.testing.assert_array_equal(
        jax.jit(_produced_rational_denominator)(point, weights),
        jax.jit(_sum_rational_denominator)(point, weights),
    )


@pytest.mark.parametrize(
    "in_axes",
    ((0, None), (None, 0), (0, 0)),
    ids=("mapped-query", "mapped-weights", "mapped-query-and-weights"),
)
def test_rational_denominator_vmap_matches_sum(
    in_axes: tuple[int | None, int | None],
) -> None:
    point = jnp.asarray(((0.25, 0.375), (0.5, 0.625)), dtype=jnp.float64)
    weights = jnp.arange(1.0, 10.0, dtype=jnp.float64).reshape((3, 3))
    points = jnp.stack((point, point + 0.125)) if in_axes[0] == 0 else point
    weight_batch = jnp.stack((weights, weights + 0.5)) if in_axes[1] == 0 else weights
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(_produced_rational_denominator, in_axes=in_axes))(
            points, weight_batch
        ),
        jax.jit(jax.vmap(_sum_rational_denominator, in_axes=in_axes))(
            points, weight_batch
        ),
    )


def test_rational_denominator_nested_vmap_matches_sum() -> None:
    point = jnp.asarray(((0.25, 0.375), (0.5, 0.625)), dtype=jnp.float64)
    points = jnp.stack(
        (jnp.stack((point, point + 0.125)), jnp.stack((point + 0.25, point)))
    )
    weights = jnp.arange(1.0, 10.0, dtype=jnp.float64).reshape((3, 3))
    weight_batch = jnp.stack((jnp.stack((weights, weights + 0.5)),) * 2)
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(jax.vmap(_produced_rational_denominator)))(points, weight_batch),
        jax.jit(jax.vmap(jax.vmap(_sum_rational_denominator)))(points, weight_batch),
    )


def test_rational_denominator_jvp_matches_sum() -> None:
    point = jnp.asarray(((0.25, 0.375), (0.5, 0.625)), dtype=jnp.float64)
    weights = jnp.arange(1.0, 10.0, dtype=jnp.float64).reshape((3, 3))
    tangents = (jnp.full_like(point, 0.125), jnp.full_like(weights, 0.25))
    actual = jax.jvp(_produced_rational_denominator, (point, weights), tangents)
    expected = jax.jvp(_sum_rational_denominator, (point, weights), tangents)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_rational_denominator_vjp_matches_sum() -> None:
    point = jnp.asarray(((0.25, 0.375), (0.5, 0.625)), dtype=jnp.float64)
    weights = jnp.arange(1.0, 10.0, dtype=jnp.float64).reshape((3, 3))
    value, pullback = jax.vjp(_produced_rational_denominator, point, weights)
    expected_value, expected_pullback = jax.vjp(_sum_rational_denominator, point, weights)
    cotangent = jnp.arange(value.size, dtype=jnp.float64).reshape(value.shape)
    actual = pullback(cotangent)
    expected = expected_pullback(cotangent)
    np.testing.assert_array_equal(value, expected_value)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_rational_placed_source_boxes_contain_fraction_reference() -> None:
    patch = _surface_box_test_patch(2, interior_knots=False)
    rotation = jnp.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=jnp.float64,
    )
    translation = jnp.asarray((2.0, -3.0, 0.5), dtype=jnp.float64)
    placed = PlacedSurface(patch, rotation, translation)
    parameters = jnp.asarray(((0.0, 0.0), (0.25, 0.5), (0.75, 1.0)), dtype=jnp.float64)
    box = _spline_seam_source_box(placed, parameters, False)
    assert box is not None
    lower, upper = (np.asarray(value) for value in box)
    controls, weights = np.asarray(patch.control_points), np.asarray(patch.weights)
    for row, (first, second) in enumerate(np.asarray(parameters)):
        u, v = Fraction(float(first)), Fraction(float(second))
        first_basis = ((1 - u) ** 2, 2 * u * (1 - u), u**2)
        second_basis = ((1 - v) ** 2, 2 * v * (1 - v), v**2)
        coefficients = [
            first_basis[i] * second_basis[j] * Fraction(float(weights[i, j]))
            for i in range(3)
            for j in range(3)
        ]
        denominator = sum(coefficients, Fraction())
        point = [
            sum(
                (
                    coefficient * Fraction(float(control[axis]))
                    for coefficient, control in zip(
                        coefficients, controls.reshape((-1, 3)), strict=True
                    )
                ),
                Fraction(),
            )
            / denominator
            for axis in range(3)
        ]
        reference = (-point[1] + 2, point[0] - 3, point[2] + Fraction(1, 2))
        for axis, value in enumerate(reference):
            assert Fraction(float(lower[row, axis])) <= value
            assert value <= Fraction(float(upper[row, axis]))
