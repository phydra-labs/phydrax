from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from itertools import product
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._interpolation import bspline_jet_stencil
from phydrax._meshcore import load_meshcore
from phydrax.geometry import _interval_enclosure as interval_enclosure, Superquadric
from phydrax.geometry._interval_enclosure import (
    interval_multiply,
    IntervalExtensionError,
    prepare_interval_function,
)
from phydrax.geometry.analytic._primitives import _finite_norm
from phydrax.geometry.brep._patches import CylinderPatch


@pytest.mark.parametrize("host", (True, False))
def test_interval_products_enclose_exact_rational_endpoints(host: bool) -> None:
    operation = (
        interval_enclosure._host_interval_multiply
        if host
        else interval_enclosure.interval_multiply
    )
    tiny = float.fromhex("0x0.0000000000001p-1022")
    cases = (
        ((0.0, tiny), (0.0, 0.5)),
        ((-tiny, -0.0), (0.0, 0.5)),
        ((-tiny, tiny), (-0.5, 0.5)),
        ((-0.0, 0.0), (-1.0, 1.0)),
        ((tiny, 3 * tiny), (0.25, 0.5)),
        ((-3 * tiny, -tiny), (0.25, 0.5)),
        ((1.0, np.nextafter(1.0, np.inf)), (0.1, 0.3)),
        ((-1.0e200, 2.0e200), (-1.0e-100, 1.0e-100)),
    )
    for first, second in cases:
        lower, upper = operation(
            (
                np.asarray(first[0], dtype=np.float64),
                np.asarray(first[1], dtype=np.float64),
            ),
            (
                np.asarray(second[0], dtype=np.float64),
                np.asarray(second[1], dtype=np.float64),
            ),
        )
        exact = [Fraction(a) * Fraction(b) for a, b in product(first, second)]
        assert Fraction(float(lower)) <= min(exact)
        assert Fraction(float(upper)) >= max(exact)
    positive = operation(
        (np.asarray(0.0), np.asarray(tiny)), (np.asarray(0.0), np.asarray(0.5))
    )
    negative = operation(
        (np.asarray(-tiny), np.asarray(-0.0)), (np.asarray(0.0), np.asarray(0.5))
    )
    assert positive[0] == 0.0 and positive[1] == tiny
    assert negative[0] == -tiny and negative[1] == 0.0
    assert not np.signbit(positive[0])


@pytest.mark.parametrize("host", (True, False))
@pytest.mark.parametrize("operation", ("add", "subtract", "multiply"))
@pytest.mark.parametrize("nan_endpoint", range(4))
def test_interval_nan_endpoint_means_whole_line(
    host: bool, operation: str, nan_endpoint: int
) -> None:
    operations = {
        "add": (interval_enclosure._host_interval_add, interval_enclosure.interval_add),
        "subtract": (
            interval_enclosure._host_interval_subtract,
            interval_enclosure.interval_subtract,
        ),
        "multiply": (
            interval_enclosure._host_interval_multiply,
            interval_enclosure.interval_multiply,
        ),
    }
    apply = operations[operation][0 if host else 1]
    values = [np.asarray(value) for value in (0.0, 1.0, 0.0, 1.0)]
    values[nan_endpoint] = np.asarray(np.nan)
    lower, upper = apply((values[0], values[1]), (values[2], values[3]))
    assert float(lower) == -np.inf and float(upper) == np.inf


@pytest.mark.parametrize("host", (True, False))
def test_interval_sum_zero_identity_does_not_hide_nan(host: bool) -> None:
    apply = (
        interval_enclosure._host_interval_add if host else interval_enclosure.interval_add
    )
    zero = (np.asarray(0.0), np.asarray(0.0))
    for invalid in (
        (np.asarray(np.nan), np.asarray(-1.0)),
        (np.asarray(1.0), np.asarray(np.nan)),
    ):
        for first, second in ((zero, invalid), (invalid, zero)):
            lower, upper = apply(first, second)
            assert float(lower) == -np.inf and float(upper) == np.inf


@pytest.mark.parametrize("host", (True, False))
def test_interval_sum_encloses_exact_endpoints_and_preserves_signed_zero(
    host: bool,
) -> None:
    add = (
        interval_enclosure._host_interval_add if host else interval_enclosure.interval_add
    )
    subtract = (
        interval_enclosure._host_interval_subtract
        if host
        else interval_enclosure.interval_subtract
    )
    tiny = float.fromhex("0x0.0000000000001p-1022")
    for apply, subtracting in ((add, False), (subtract, True)):
        for a, b in ((tiny, tiny), (1.0, tiny), (1.0, 0.1), (-1.0, -0.1)):
            lower, upper = apply(
                (np.asarray(a), np.asarray(a)), (np.asarray(b), np.asarray(b))
            )
            exact = Fraction(a) + (-Fraction(b) if subtracting else Fraction(b))
            assert Fraction(float(lower)) <= exact <= Fraction(float(upper))
    lower, upper = subtract(
        (np.asarray(-0.0), np.asarray(-0.0)),
        (np.asarray(0.0), np.asarray(0.0)),
    )
    assert lower == upper == 0.0
    assert np.signbit(lower) and np.signbit(upper)


def _assert_nested_enclosures(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> None:
    first_contains = (first[0] <= second[0]) & (first[1] >= second[1])
    second_contains = (second[0] <= first[0]) & (second[1] >= first[1])
    assert np.all(first_contains | second_contains)


def test_interval_jaxpr_encloses_value_and_directional_gradient_batches() -> None:
    def function(point: jax.Array) -> jax.Array:
        return jnp.asarray((point[0] * point[0] + jnp.sin(point[1]),))

    value = prepare_interval_function(function, 2)
    gradient = prepare_interval_function(jax.grad(lambda point: function(point)[0]), 2)
    lower = np.asarray(((0.5, -0.2), (0.75, 0.0)), dtype=np.float64)
    upper = np.asarray(((1.0, 0.2), (1.0, 0.1)), dtype=np.float64)

    values = value.evaluate(lower, upper)
    gradients = gradient.evaluate(lower, upper)

    assert np.all(values[0] <= values[1])
    assert np.all(gradients[0] <= gradients[1])
    points = np.asarray(((0.6, -0.1), (0.9, 0.05)), dtype=np.float64)
    nominal = value.evaluate_host_points(points)
    assert np.all(nominal >= values[0])
    assert np.all(nominal <= values[1])


def test_native_interval_polynomial_trigonometric_and_derivative_enclosures() -> None:

    def function(point: jax.Array) -> jax.Array:
        first, second = point
        return jnp.stack(
            (
                first * first * first + jnp.sin(second),
                jnp.sqrt(first + 3.0) * jnp.cos(second),
            )
        )

    lower = np.asarray(((0.25, -0.2), (0.4, 0.0)), dtype=np.float64)
    upper = np.asarray(((0.5, 0.1), (0.75, 0.2)), dtype=np.float64)
    for mapped_function in (function, jax.jacfwd(function)):
        native = prepare_interval_function(mapped_function, 2, execution="native")
        host = prepare_interval_function(mapped_function, 2, execution="host")
        native_bounds = native.evaluate(lower, upper)
        host_bounds = host.evaluate(lower, upper)
        assert native.execution_reason
        assert native.execution == "native"
        _assert_nested_enclosures(native_bounds, host_bounds)


def test_native_program_refuses_unadvertised_opcode() -> None:
    def function(point: jax.Array) -> jax.Array:
        return jnp.asarray((jnp.sqrt(point[0] + 1.0),))

    prepared = prepare_interval_function(function, 1, execution="native")
    program = prepared.native_program
    if program is None:
        raise AssertionError("Native preparation must retain its executable program.")
    supported_opcodes = load_meshcore().interval_program_capabilities()
    missing_opcode = next(
        opcode for opcode in range(1, 64) if opcode not in supported_opcodes
    )
    unsupported_program = replace(
        program,
        instructions=np.concatenate(
            (
                program.instructions,
                np.asarray(
                    ((missing_opcode, program.workspace_size, 0, 0, -1, -1, -1, 0),),
                    dtype=np.int64,
                ),
            ),
            axis=0,
        ),
        opcodes=program.opcodes | {missing_opcode},
    )
    with pytest.raises(IntervalExtensionError, match="opcodes"):
        unsupported_program.require_supported_opcodes(supported_opcodes)


def test_native_cylinder_chart_and_derivative_enclosures() -> None:
    cylinder = CylinderPatch(
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        np.asarray(1.25, dtype=np.float64),
    )
    lower = np.asarray(((0.1, -0.5), (0.4, 0.0)), dtype=np.float64)
    upper = np.asarray(((0.2, -0.25), (0.6, 0.1)), dtype=np.float64)
    for function in (cylinder.evaluate, jax.jacfwd(cylinder.evaluate)):
        native = prepare_interval_function(function, 2, execution="native")
        host = prepare_interval_function(function, 2, execution="host")
        native_bounds = native.evaluate(lower, upper)
        host_bounds = host.evaluate(lower, upper)
        assert native.execution_reason
        assert native.execution == "native"
        _assert_nested_enclosures(native_bounds, host_bounds)


def test_interval_repeated_multiply_preserves_square_dependency() -> None:
    def function(point: jax.Array) -> jax.Array:
        value = point[0]
        return jnp.asarray((value * value,))

    prepared = prepare_interval_function(function, 1)
    lower, upper = prepared.evaluate(
        np.asarray(((-1.0,), (0.3671875,), (-0.5,)), dtype=np.float64),
        np.asarray(((1.0,), (0.375,), (-0.3125,)), dtype=np.float64),
    )

    np.testing.assert_array_equal(lower[0], (0.0,))
    assert np.all(lower >= 0.0)
    assert upper[0, 0] >= 1.0


def test_interval_real_power_of_sum_of_squares_encloses_origin_box() -> None:
    def function(point: jax.Array) -> jax.Array:
        planar = jnp.square(point[0]) + point[1] * point[1] + point[2] ** 4
        return jnp.asarray((planar**0.75 - 0.25, planar ** (-0.5)))

    prepared = prepare_interval_function(function, 3)
    box_lower = np.asarray(((-0.5, -0.25, -0.125),), dtype=np.float64)
    box_upper = np.asarray(((0.25, 0.5, 0.5),), dtype=np.float64)
    lower, upper = prepared.evaluate(box_lower, box_upper)

    planar_upper = 0.5**2 + 0.5**2 + 0.5**4
    assert -0.25 - 1.0e-15 <= lower[0, 0] <= -0.25
    assert upper[0, 0] >= planar_upper**0.75 - 0.25
    assert lower[0, 1] <= planar_upper ** (-0.5)
    assert upper[0, 1] == np.inf

    def negative_base(point: jax.Array) -> jax.Array:
        return jnp.asarray(((point[0] - 1.0) ** 0.5,))

    with pytest.raises(IntervalExtensionError, match="nonnegative base"):
        prepare_interval_function(negative_base, 1).evaluate(
            np.asarray(((0.0,),), dtype=np.float64),
            np.asarray(((0.5,),), dtype=np.float64),
        )


def test_interval_exact_constant_arithmetic_keeps_an_integer_power_exponent() -> None:
    # ``blockiness / 2 - 1`` with blockiness 2 is exactly 0; exact operations
    # are not rounded out, so the power stays the integer power 0 ** 0 = 1
    # on a box straddling the origin.
    def function(point: jax.Array) -> jax.Array:
        exponent = jnp.asarray(2.0) * 0.5 - 1.0
        return jnp.asarray((jnp.power(jnp.square(point[0]), exponent),))

    lower, upper = prepare_interval_function(function, 1).evaluate(
        np.asarray(((-0.05,),), dtype=np.float64),
        np.asarray(((0.05,),), dtype=np.float64),
    )

    assert lower[0, 0] == 1.0
    assert upper[0, 0] == 1.0

    # Inexact sums and products are still rounded outward.
    def inexact(point: jax.Array) -> jax.Array:
        return jnp.asarray((point[0] + 0.1, point[0] * 0.1))

    lower, upper = prepare_interval_function(inexact, 1).evaluate(
        np.asarray(((0.2,),), dtype=np.float64),
        np.asarray(((0.2,),), dtype=np.float64),
    )

    assert lower[0, 0] < 0.2 + 0.1 < upper[0, 0]
    assert lower[0, 1] < 0.2 * 0.1 < upper[0, 1]


def test_interval_multiply_does_not_infer_correlation_from_equal_bounds() -> None:
    def function(point: jax.Array) -> jax.Array:
        return jnp.asarray((point[0] * point[1],))

    prepared = prepare_interval_function(function, 2)
    lower, upper = prepared.evaluate(
        np.asarray(((-1.0, -1.0),), dtype=np.float64),
        np.asarray(((1.0, 1.0),), dtype=np.float64),
    )

    assert lower[0, 0] <= -1.0
    assert upper[0, 0] >= 1.0


def test_interval_multiply_takes_exact_zero_times_an_unbounded_side_as_zero() -> None:
    zero = (np.zeros(1), np.zeros(1))
    unbounded = (np.ones(1), np.full(1, np.inf))
    lower, upper = (np.asarray(value) for value in interval_multiply(zero, unbounded))
    assert lower[0] == 0.0 and upper[0] == 0.0
    lower, upper = (
        np.asarray(value)
        for value in interval_multiply((np.zeros(1), np.ones(1)), unbounded)
    )
    assert lower[0] == 0.0 and upper[0] == np.inf

    # The ellipsoidal superquadric Hessian multiplies an exact zero exponent
    # factor by pow(y², -1); boxes straddling y = 0 keep a finite enclosure.
    import phydrax as phx

    geometry = phx.geometry.Superquadric((0.0, 0.0, 0.0), (0.45, 0.50, 0.55)).compile()
    kernel, state = geometry.kernel, geometry.state

    def hessian(point: jax.Array) -> jax.Array:
        return jax.hessian(lambda x: kernel.boundary_field(state, x[None, :])[0])(point)

    prepared = prepare_interval_function(hessian, 3)
    box_lower = np.asarray(((-0.2826, -1.0e-5, -0.4275),), dtype=np.float64)
    box_upper = np.asarray(((-0.2824, 1.0e-5, -0.4273),), dtype=np.float64)
    lower, upper = prepared.evaluate(box_lower, box_upper)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    sample = jax.jit(hessian)(jnp.asarray((-0.2825, 5.0e-6, -0.4274)))
    sample = np.asarray(sample).reshape((3, 3))
    assert np.all(lower.reshape((3, 3)) <= sample)
    assert np.all(sample <= upper.reshape((3, 3)))


def test_interval_sphere_norm_preserves_a_strictly_positive_hidden_box() -> None:
    center = jnp.asarray((0.37, 0.13, -0.36), dtype=jnp.float64)
    radius = 0.04

    def function(point: jax.Array) -> jax.Array:
        return _finite_norm(point - center) - radius

    prepared = prepare_interval_function(function, 3)
    box_lower = np.asarray((0.3671875, 0.1171875, -0.3203125), dtype=np.float64)
    box_upper = np.asarray((0.375, 0.125, -0.3125), dtype=np.float64)
    lower, upper = prepared.evaluate(box_lower[None, :], box_upper[None, :])
    corners = box_lower + np.asarray(tuple(np.ndindex(2, 2, 2)), dtype=np.float64) * (
        box_upper - box_lower
    )
    witnesses = np.concatenate((corners, ((box_lower + box_upper) / 2.0)[None, :]))
    values = np.asarray(jax.vmap(function)(jnp.asarray(witnesses)), dtype=np.float64)

    assert np.min(values) > 0.0
    assert lower[0] > 0.0
    assert upper[0] >= np.max(values)


def test_interval_jacfwd_preserves_discrete_identity_tangent_basis() -> None:
    def function(point: jax.Array) -> jax.Array:
        return jnp.stack((point[0], point[1]))

    prepared = prepare_interval_function(jax.jacfwd(function), 2)
    point = np.asarray(((0.0, 0.0),), dtype=np.float64)
    lower, upper = prepared.evaluate(point, point)

    np.testing.assert_allclose(
        lower, np.asarray((((1.0, 0.0), (0.0, 1.0)),)), rtol=0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        upper, np.asarray((((1.0, 0.0), (0.0, 1.0)),)), rtol=0.0, atol=1.0e-14
    )


def test_interval_min_max_preserve_exact_discrete_index_bounds() -> None:
    def function(point: jax.Array) -> jax.Array:
        integer = jnp.asarray(point[0], dtype=jnp.int32)
        return jnp.stack((jnp.minimum(integer, 9), jnp.maximum(integer, 9)))

    prepared = prepare_interval_function(function, 1)
    point = np.asarray(((9.0,),), dtype=np.float64)
    lower, upper = prepared.evaluate(point, point)

    np.testing.assert_array_equal(lower, ((9.0, 9.0),))
    np.testing.assert_array_equal(upper, ((9.0, 9.0),))


def test_interval_dot_and_reduction_enclose_all_endpoint_combinations() -> None:
    function = lambda point: jnp.dot(point, point)
    prepared = prepare_interval_function(function, 3)
    lower = np.asarray(((-1.0, 2.0, 0.0),), dtype=np.float64)
    upper = np.asarray(((2.0, 3.0, 1.0),), dtype=np.float64)

    result = prepared.evaluate(lower, upper)

    assert result[0][0] <= 4.0
    assert result[1][0] >= 14.0


def test_unsupported_interval_primitives_refuse_native_lowering() -> None:
    def function(point: jax.Array) -> jax.Array:
        return jnp.linalg.svd(point.reshape((1, 1)), compute_uv=False)

    with pytest.raises(IntervalExtensionError, match="native"):
        prepare_interval_function(function, 1, execution="native")

    prepared = prepare_interval_function(function, 1, execution="host")
    assert prepared.execution == "host"
    assert prepared.execution_reason
    lower = np.asarray(((1.0,),), dtype=np.float64)
    upper = np.asarray(((2.0,),), dtype=np.float64)
    with pytest.raises(IntervalExtensionError):
        prepared.evaluate(lower, upper)


def test_interval_vector_map_preserves_component_bounds_through_reshape() -> None:
    def function(point: jax.Array) -> jax.Array:
        values = jnp.stack((point[0] * point[0] + 2.0 * point[1], point[0] - point[1]))
        return values.reshape((1, 2))

    prepared = prepare_interval_function(function, 2)
    lower, upper = prepared.evaluate(
        np.asarray(((1.0, 3.0),), dtype=np.float64),
        np.asarray(((2.0, 4.0),), dtype=np.float64),
    )
    minimum = np.asarray((((7.0, -3.0),),), dtype=np.float64)
    maximum = np.asarray((((12.0, -1.0),),), dtype=np.float64)
    assert np.all(lower <= minimum)
    assert np.all(upper >= maximum)
    np.testing.assert_allclose(lower, minimum, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(upper, maximum, rtol=0.0, atol=1.0e-13)


def test_interval_padding_preserves_cropped_and_inserted_bounds() -> None:
    def function(point: jax.Array) -> jax.Array:
        return jax.lax.pad(point[:3], point[3], ((-1, 2, 1),))

    prepared = prepare_interval_function(function, 4)
    lower, upper = prepared.evaluate(
        np.asarray(((1.0, 2.0, 3.0, 10.0),), dtype=np.float64),
        np.asarray(((2.0, 4.0, 6.0, 20.0),), dtype=np.float64),
    )
    np.testing.assert_array_equal(lower, ((10.0, 2.0, 10.0, 3.0, 10.0, 10.0),))
    np.testing.assert_array_equal(upper, ((20.0, 4.0, 20.0, 6.0, 20.0, 20.0),))


def test_interval_custom_derivative_primal_retains_captured_constants() -> None:
    coefficient = jnp.asarray((3.0,), dtype=jnp.float64)

    @jax.custom_jvp
    def function(point: jax.Array) -> jax.Array:
        return coefficient * point * point

    @function.defjvp
    def derivative(
        primals: tuple[jax.Array], tangents: tuple[jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        (point,) = primals
        (tangent,) = tangents
        return function(point), 2.0 * coefficient * point * tangent

    prepared = prepare_interval_function(function, 1)
    lower, upper = prepared.evaluate(
        np.asarray(((1.0,),), dtype=np.float64),
        np.asarray(((2.0,),), dtype=np.float64),
    )
    assert lower[0, 0] <= 3.0 and upper[0, 0] >= 12.0
    np.testing.assert_allclose(lower, ((3.0,),), rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(upper, ((12.0,),), rtol=0.0, atol=1.0e-13)


def test_interval_selection_keeps_uncertain_and_decided_lanes_distinct() -> None:
    def function(point: jax.Array) -> jax.Array:
        return jnp.where(point > 0.0, 1.0, -1.0)

    prepared = prepare_interval_function(function, 1)
    lower, upper = prepared.evaluate(
        np.asarray(((-1.0,), (0.1,), (-0.2,)), dtype=np.float64),
        np.asarray(((1.0,), (0.2,), (-0.1,)), dtype=np.float64),
    )
    np.testing.assert_array_equal(lower, ((-1.0,), (1.0,), (-1.0,)))
    np.testing.assert_array_equal(upper, ((1.0,), (1.0,), (-1.0,)))


_COMPARISON_LOWER = np.asarray(
    (
        (2.0, 0.0, -1.0, 5.0),
        (0.0, 1.0, -1.0, 5.0),
        (0.0, 0.5, -1.0, 5.0),
        (1.0, 1.0, -1.0, 5.0),
        (-np.inf, 0.0, -np.inf, 1.0),
        (-1.0, 0.0, 0.0, 3.0),
    ),
    dtype=np.float64,
)
_COMPARISON_UPPER = np.asarray(
    (
        (3.0, 1.0, 0.0, 6.0),
        (1.0, 2.0, 0.0, 6.0),
        (1.0, 0.5, 0.0, 6.0),
        (1.0, 1.0, 0.0, 6.0),
        (0.0, np.inf, 0.0, 2.0),
        (1.0, 0.0, 1.0, np.inf),
    ),
    dtype=np.float64,
)


@pytest.mark.parametrize("comparison", ("gt", "ge", "lt", "le"))
def test_native_interval_comparison_selection_matches_host(comparison: str) -> None:
    # Shared vectors with the native test_interval_program compare/select case:
    # decided lanes return their branch exactly, undecided lanes the hull.
    compare = {
        "gt": jnp.greater,
        "ge": jnp.greater_equal,
        "lt": jnp.less,
        "le": jnp.less_equal,
    }[comparison]

    def function(point: jax.Array) -> jax.Array:
        return jnp.where(compare(point[0], point[1]), point[2], point[3])

    native = prepare_interval_function(function, 4, execution="native")
    host = prepare_interval_function(function, 4, execution="host")
    native_bounds = native.evaluate(_COMPARISON_LOWER, _COMPARISON_UPPER)
    host_bounds = host.evaluate(_COMPARISON_LOWER, _COMPARISON_UPPER)

    assert native.execution == "native"
    np.testing.assert_array_equal(native_bounds[0], host_bounds[0])
    np.testing.assert_array_equal(native_bounds[1], host_bounds[1])


@pytest.mark.parametrize("extremum", ("maximum", "minimum"))
def test_native_interval_extrema_match_host(extremum: str) -> None:
    # Shared vectors with the native test_interval_program maximum/minimum case,
    # plus a constant operand (the superquadric norm floor).
    select = jnp.maximum if extremum == "maximum" else jnp.minimum

    def function(point: jax.Array) -> jax.Array:
        return jnp.stack((select(point[0], point[1]), select(point[0], 1.0e-12)))

    lower = np.asarray(
        ((-1.0, 0.0), (-np.inf, 1.0e-12), (3.0, -np.inf), (-2.0, -3.0)), dtype=np.float64
    )
    upper = np.asarray(
        ((2.0, 1.0), (0.5, 1.0e-12), (np.inf, 4.0), (-1.0, 5.0)), dtype=np.float64
    )
    native = prepare_interval_function(function, 2, execution="native")
    host = prepare_interval_function(function, 2, execution="host")
    native_bounds = native.evaluate(lower, upper)
    host_bounds = host.evaluate(lower, upper)

    assert native.execution == "native"
    np.testing.assert_array_equal(native_bounds[0], host_bounds[0])
    np.testing.assert_array_equal(native_bounds[1], host_bounds[1])


def test_native_interval_real_power_matches_host_and_refuses_negative_bases() -> None:
    # exp(y log x) clamped at 0 encloses the host result (native products always
    # round outward, the host only when inexact, so they differ by ulps); a base
    # touching 0 keeps lower bound 0, and a negative base is a typed refusal.
    def function(point: jax.Array) -> jax.Array:
        return jnp.stack((point[0] ** 0.75, point[1] ** -1.5, jnp.exp(point[0])))

    lower = np.asarray(((0.0, 0.5), (0.25, 1.0), (2.0, 3.0)), dtype=np.float64)
    upper = np.asarray(((1.0, 2.0), (0.5, np.inf), (4.0, 3.0)), dtype=np.float64)
    native = prepare_interval_function(function, 2, execution="native")
    host = prepare_interval_function(function, 2, execution="host")
    native_bounds = native.evaluate(lower, upper)
    host_bounds = host.evaluate(lower, upper)

    assert native.execution == "native"
    assert np.all(native_bounds[0] <= host_bounds[0])
    assert np.all(native_bounds[1] >= host_bounds[1])
    np.testing.assert_allclose(native_bounds[0], host_bounds[0], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(native_bounds[1], host_bounds[1], rtol=1e-14, atol=0.0)
    assert native_bounds[0][0, 0] == 0.0
    negative = np.asarray(((-0.5, 1.0),), dtype=np.float64)
    for prepared in (native, host):
        with pytest.raises(IntervalExtensionError, match="nonnegative base"):
            prepared.evaluate(negative, negative + 1.0)


def test_native_interval_dot_general_matches_host() -> None:
    rotation = jnp.asarray(
        ((0.6, -0.8, 0.0), (0.8, 0.6, 0.0), (0.0, 0.0, 1.0)), dtype=jnp.float64
    )

    def function(point: jax.Array) -> jax.Array:
        rotated = rotation @ point
        return jnp.concatenate((rotated, jnp.dot(point, point)[None]))

    lower = np.asarray(((0.1, -0.2, 0.3), (-1.0, -1.0, -1.0)), dtype=np.float64)
    upper = np.asarray(((0.2, 0.1, 0.5), (1.0, 1.0, 1.0)), dtype=np.float64)
    native = prepare_interval_function(function, 3, execution="native")
    host = prepare_interval_function(function, 3, execution="host")
    native_bounds = native.evaluate(lower, upper)
    host_bounds = host.evaluate(lower, upper)

    assert native.execution == "native"
    _assert_nested_enclosures(native_bounds, host_bounds)
    np.testing.assert_allclose(native_bounds[0], host_bounds[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(native_bounds[1], host_bounds[1], rtol=0.0, atol=1e-12)


def test_native_interval_keeps_an_exact_derived_integer_exponent() -> None:
    # 0.5 * blockiness is exactly 1: the power stays an integer power, so the
    # gradient of (v^2)^(0.5 b) is finite on a box through v = 0 (natively and
    # on the host) instead of exp((k - 1) log 0) blowing up.
    blockiness = jnp.asarray(2.0, dtype=jnp.float64)

    def field(point: jax.Array) -> jax.Array:
        return jnp.square(point[0]) ** (0.5 * blockiness) + point[1]

    lower = np.asarray(((-0.1, 0.0),), dtype=np.float64)
    upper = np.asarray(((0.1, 1.0),), dtype=np.float64)
    for function in (field, jax.grad(field)):
        native = prepare_interval_function(function, 2, execution="native")
        host = prepare_interval_function(function, 2, execution="host")
        native_bounds = native.evaluate(lower, upper)
        host_bounds = host.evaluate(lower, upper)
        assert np.all(np.isfinite(native_bounds[0])) and np.all(
            np.isfinite(native_bounds[1])
        )
        _assert_nested_enclosures(native_bounds, host_bounds)


def test_native_interval_lowers_the_superquadric_field_and_gradient() -> None:
    geometry = Superquadric((0.0, 0.0, 0.0), (0.45, 0.50, 0.55)).compile()
    kernel, state = geometry.kernel, geometry.state

    def field(point: jax.Array) -> jax.Array:
        return kernel.boundary_field(state, point[None, :])[0]

    lower = np.asarray(((0.40, 0.0, 0.0), (0.1, 0.2, 0.3)), dtype=np.float64)
    upper = np.asarray(((0.42, 0.02, 0.02), (0.15, 0.25, 0.35)), dtype=np.float64)
    for function in (field, jax.grad(field)):
        native = prepare_interval_function(function, 3, execution="native")
        host = prepare_interval_function(function, 3, execution="host")
        native_bounds = native.evaluate(lower, upper)
        host_bounds = host.evaluate(lower, upper)
        assert native.execution == "native"
        assert np.all(native_bounds[0] <= host_bounds[1])
        assert np.all(host_bounds[0] <= native_bounds[1])
        # Away from the coordinate planes the real powers are smooth and finite.
        assert np.all(np.isfinite(native_bounds[0][1]))
        assert np.all(np.isfinite(native_bounds[1][1]))


def test_native_interval_lowers_the_finite_norm_gradient() -> None:
    center = jnp.asarray((0.37, 0.13, -0.36), dtype=jnp.float64)

    def gradient(point: jax.Array) -> jax.Array:
        return jax.grad(lambda value: _finite_norm(value - center))(point)

    native = prepare_interval_function(gradient, 3, execution="native")
    host = prepare_interval_function(gradient, 3, execution="host")
    # A box away from the centre (decided guard) and one around it (hull).
    lower = np.asarray(((1.0, 1.0, 1.0), (0.3, 0.1, -0.4)), dtype=np.float64)
    upper = np.asarray(((1.25, 1.25, 1.25), (0.4, 0.2, -0.3)), dtype=np.float64)
    native_bounds = native.evaluate(lower, upper)
    host_bounds = host.evaluate(lower, upper)

    _assert_nested_enclosures(native_bounds, host_bounds)
    assert np.all(np.isfinite(native_bounds[0][0])) and np.all(native_bounds[0][0] > 0.0)
    assert not np.all(np.isfinite(native_bounds[0][1]))


def test_interval_split_keeps_unequal_component_bounds() -> None:
    def function(point: jax.Array) -> jax.Array:
        first, middle, last = jnp.split(point, (1, 3))
        return jnp.concatenate((last, middle, first))

    prepared = prepare_interval_function(function, 4)
    lower, upper = prepared.evaluate(
        np.asarray(((1.0, 2.0, 3.0, 4.0),), dtype=np.float64),
        np.asarray(((2.0, 4.0, 6.0, 8.0),), dtype=np.float64),
    )
    np.testing.assert_array_equal(lower, ((4.0, 2.0, 3.0, 1.0),))
    np.testing.assert_array_equal(upper, ((8.0, 4.0, 6.0, 2.0),))


def test_interval_is_finite_and_boolean_guards_enclose_nonfinite_lanes() -> None:
    def function(point: jax.Array) -> jax.Array:
        nonfinite = jnp.any(~jnp.isfinite(point))
        negative = jnp.any(point < 0.0)
        above = jnp.any(point > 2.0)
        return jnp.where(nonfinite | (negative & above), 1.0, 0.0)

    prepared = prepare_interval_function(function, 2)
    lower = np.asarray(
        (
            (0.5, 1.5),
            (-3.0, 4.0),
            (1.0, 0.0),
            (np.inf, 0.0),
            (-np.inf, 0.0),
            (-np.inf, 0.0),
        ),
        dtype=np.float64,
    )
    upper = np.asarray(
        (
            (0.75, 1.75),
            (-2.0, 5.0),
            (np.inf, 1.0),
            (np.inf, 0.0),
            (-np.inf, 0.0),
            (-1.0, 1.0),
        ),
        dtype=np.float64,
    )

    lower_result, upper_result = prepared.evaluate(lower, upper)

    np.testing.assert_array_equal(lower_result, (0.0, 1.0, 0.0, 1.0, 1.0, 0.0))
    np.testing.assert_array_equal(upper_result, (0.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    invalid_lower = lower[:1].copy()
    invalid_lower[0, 0] = np.nan
    with pytest.raises(ValueError, match="ordered and non-NaN"):
        prepared.evaluate(invalid_lower, upper[:1])


def test_interval_error_if_validation_callbacks_are_not_suppressed() -> None:
    def function(point: jax.Array) -> jax.Array:
        return eqx.error_if(
            point,
            jnp.any(~jnp.isfinite(point)),
            "interval inputs must remain finite",
        )

    prepared = prepare_interval_function(function, 1, batch_capacity=4)
    lower, upper = prepared.evaluate(
        np.asarray(((1.0,),), dtype=np.float64),
        np.asarray(((2.0,),), dtype=np.float64),
    )
    assert lower[0, 0] <= 1.0 <= upper[0, 0]
    assert lower[0, 0] <= 2.0 <= upper[0, 0]

    with pytest.raises(IntervalExtensionError):
        prepared.evaluate(
            np.asarray(((-np.inf,),), dtype=np.float64),
            np.asarray(((2.0,),), dtype=np.float64),
        )

    with pytest.raises(IntervalExtensionError):
        prepared.evaluate(
            np.asarray(((np.inf,),), dtype=np.float64),
            np.asarray(((np.inf,),), dtype=np.float64),
        )


def test_interval_gather_hulls_only_integer_starts_per_lane() -> None:
    values = jnp.asarray((2.0, 5.0, 9.0), dtype=jnp.float64)
    dimensions = jax.lax.GatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
    )

    def function(point: jax.Array) -> jax.Array:
        start = jnp.asarray(point[0], dtype=jnp.int32)[None]
        return jax.lax.gather(
            values,
            start,
            dimensions,
            slice_sizes=(1,),
            mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
        )

    prepared = prepare_interval_function(function, 1)
    lower, upper = prepared.evaluate(
        np.asarray(((0.0,), (2.0,), (0.0,), (1.0,), (0.2,)), dtype=np.float64),
        np.asarray(((0.0,), (2.0,), (1.0,), (2.0,), (0.8,)), dtype=np.float64),
    )

    np.testing.assert_array_equal(lower, (2.0, 9.0, 2.0, 5.0, 2.0))
    np.testing.assert_array_equal(upper, (2.0, 9.0, 5.0, 9.0, 2.0))
    with pytest.raises(IntervalExtensionError, match="outside the operand"):
        prepared.evaluate(
            np.asarray(((2.0,),), dtype=np.float64),
            np.asarray(((3.0,),), dtype=np.float64),
        )


def test_interval_scatter_add_tracks_repeats_optional_and_dropped_indices() -> None:
    operand = jnp.zeros((3, 1), dtype=jnp.float64)
    updates = jnp.asarray(((2.0,), (-3.0,), (4.0,)), dtype=jnp.float64)
    dimensions = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    def function(point: jax.Array) -> jax.Array:
        indices = jnp.asarray(point, dtype=jnp.int32)[:, None]
        return jax.lax.scatter_add(
            operand,
            indices,
            updates,
            dimensions,
            mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
        )

    prepared = prepare_interval_function(function, 3)
    lower_boxes = np.asarray(
        ((1.0, 1.0, 2.0), (0.0, 2.0, 2.0), (-1.0, 2.0, 2.0)),
        dtype=np.float64,
    )
    upper_boxes = np.asarray(
        ((1.0, 1.0, 2.0), (1.0, 2.0, 2.0), (0.0, 2.0, 2.0)),
        dtype=np.float64,
    )

    lower, upper = prepared.evaluate(lower_boxes, upper_boxes)

    expected_lower = np.asarray(
        (
            ((0.0,), (-1.0,), (4.0,)),
            ((0.0,), (0.0,), (1.0,)),
            ((0.0,), (0.0,), (1.0,)),
        ),
        dtype=np.float64,
    )
    expected_upper = np.asarray(
        (
            ((0.0,), (-1.0,), (4.0,)),
            ((2.0,), (2.0,), (1.0,)),
            ((2.0,), (0.0,), (1.0,)),
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(lower, expected_lower, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(upper, expected_upper, rtol=0.0, atol=1.0e-13)
    assert lower[1, 0, 0] <= 0.0 and upper[1, 0, 0] >= 2.0
    assert lower[1, 1, 0] <= 0.0 and upper[1, 1, 0] >= 2.0
    assert lower[2, 0, 0] <= 0.0 and upper[2, 0, 0] >= 2.0


def test_interval_argmin_encloses_possible_first_minimum_indices() -> None:
    def function(point: jax.Array) -> jax.Array:
        return jnp.argmin(point)

    prepared = prepare_interval_function(function, 3)
    lower, upper = prepared.evaluate(
        np.asarray(
            ((2.0, 1.0, 3.0), (0.0, 1.0, 4.0), (1.0, 1.0, 2.0), (3.0, 2.0, 1.0)),
            dtype=np.float64,
        ),
        np.asarray(
            ((2.0, 1.0, 3.0), (2.0, 3.0, 4.0), (1.0, 1.0, 2.0), (3.0, 2.0, 1.0)),
            dtype=np.float64,
        ),
    )

    np.testing.assert_array_equal(lower, (1.0, 0.0, 0.0, 2.0))
    np.testing.assert_array_equal(upper, (1.0, 1.0, 0.0, 2.0))


@pytest.mark.parametrize("reverse", (False, True))
def test_interval_scan_unrolls_carry_and_stacked_outputs(reverse: bool) -> None:
    def function(point: jax.Array) -> jax.Array:
        def step(carry: jax.Array, scale: jax.Array) -> tuple[jax.Array, jax.Array]:
            updated = carry * scale + point[1]
            return updated, updated * updated

        final, stacked = jax.lax.scan(
            step, point[0], jnp.asarray((1.0, -2.0, 0.5)), reverse=reverse
        )
        return jnp.concatenate((final[None], stacked))

    prepared = prepare_interval_function(function, 2, execution="host")
    lower = np.asarray(((0.5, -0.25),), dtype=np.float64)
    upper = np.asarray(((1.0, 0.25),), dtype=np.float64)
    bounds = prepared.evaluate(lower, upper)
    grid = np.stack(
        np.meshgrid(np.linspace(0.5, 1.0, 9), np.linspace(-0.25, 0.25, 9)), -1
    ).reshape(-1, 2)
    values = np.asarray(jax.vmap(function)(jnp.asarray(grid)), dtype=np.float64)
    assert np.all(bounds[0][0] <= values.min(axis=0))
    assert np.all(bounds[1][0] >= values.max(axis=0))


_JET_KNOTS = jnp.asarray((0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0))
_JET_COEFFICIENTS = jnp.asarray((0.0, 1.0, -0.5, 2.0, 0.7))


def _quadratic_spline(
    bounds: Literal["clip", "extrapolate"],
) -> Callable[[jax.Array], jax.Array]:
    def value(point: jax.Array) -> jax.Array:
        stencil = bspline_jet_stencil(
            _JET_KNOTS, point[0], degree=2, maximum_order=0, bounds=bounds
        )
        return jnp.sum(stencil.jets[0] * _JET_COEFFICIENTS[stencil.indices])

    return value


def _sampled_range(
    function: Callable[[jax.Array], jax.Array], first: float, last: float
) -> tuple[float, float]:
    samples = jnp.linspace(first, last, 257)[:, None]
    values = np.asarray(jax.vmap(function)(samples), dtype=np.float64)
    return float(values.min()), float(values.max())


def test_interval_bspline_jets_enclose_each_span_in_bernstein_form() -> None:
    value = _quadratic_spline("clip")
    derivatives = {
        "value": value,
        "first": jax.jacfwd(value),
        "second": jax.jacfwd(jax.jacfwd(value)),
    }
    boxes = ((0.1, 0.2), (0.4, 0.55), (0.25, 0.35), (0.0, 1.0), (0.3, 0.3))
    lower = np.asarray([[box[0]] for box in boxes], dtype=np.float64)
    upper = np.asarray([[box[1]] for box in boxes], dtype=np.float64)
    for name, function in derivatives.items():
        prepared = prepare_interval_function(
            lambda point, function=function: jnp.reshape(function(point), (1,)),
            1,
            batch_capacity=len(boxes),
        )
        bounds = prepared.evaluate(lower, upper)
        assert np.all(np.isfinite(bounds[0])) and np.all(np.isfinite(bounds[1]))
        for row, (first, last) in enumerate(boxes):
            low, high = _sampled_range(function, first, last)
            assert bounds[0][row, 0] <= low and high <= bounds[1][row, 0]
            inside_span = last <= 0.3 or (first >= 0.3 and last <= 0.6)
            if name == "second" and inside_span:
                # Inside one span every quadratic basis function has a
                # constant second derivative; its Bernstein range is exact,
                # so the enclosure is the sampled value up to rounding.
                np.testing.assert_allclose(
                    (bounds[0][row, 0], bounds[1][row, 0]),
                    (low, high),
                    rtol=1e-12,
                    atol=1e-12,
                )


def test_interval_bspline_piece_jets_extrapolate_their_span_polynomial() -> None:
    function = jax.jacfwd(jax.jacfwd(_quadratic_spline("extrapolate")))
    prepared = prepare_interval_function(
        lambda point: jnp.reshape(function(point), (1,)), 1, batch_capacity=2
    )
    lower = np.asarray(((1.0,), (-0.4,)), dtype=np.float64)
    upper = np.asarray(((1.4,), (-0.1,)), dtype=np.float64)
    bounds = prepared.evaluate(lower, upper)
    for row in range(2):
        low, high = _sampled_range(function, lower[row, 0], upper[row, 0])
        assert bounds[0][row, 0] <= low and high <= bounds[1][row, 0]
        np.testing.assert_allclose(
            (bounds[0][row, 0], bounds[1][row, 0]), (low, high), rtol=1e-12
        )
