import math
from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry._rational import (
    binary64_difference_magnitude,
    binary64_dot,
    binary64_matmul,
    binary64_norm,
    binary64_norm_lower,
    binary64_norm_upper,
    exact_float,
    fraction_endpoint,
    rational_cross,
    rational_vector,
    translation_ranges,
)


def _ordered_fraction_dot(first: np.ndarray, second: np.ndarray) -> float:
    products = [
        float(Fraction(float(a)) * Fraction(float(b)))
        for a, b in zip(first, second, strict=True)
    ]
    if not products:
        return 0.0
    result = products[0]
    for product in products[1:]:
        result = float(Fraction(result) + Fraction(product))
    return result


def test_binary64_ordered_products_match_fraction_rounding_and_batches() -> None:
    a = np.asarray(
        [
            [[1e16, 1.0, -1e16], [0.1, -0.3, 0.7]],
            [[2.0, -3.0, 4.0], [1.1, 0.2, -0.9]],
        ]
    )
    b = np.asarray([[1.0, 0.8], [1.0, -0.4], [1.0, 0.2]])
    expected = np.asarray(
        [
            [
                [_ordered_fraction_dot(row, b[:, column]) for column in range(2)]
                for row in batch
            ]
            for batch in a
        ]
    )
    np.testing.assert_array_equal(binary64_matmul(a, b), expected)
    np.testing.assert_array_equal(binary64_matmul(a, b[:, 0]), expected[..., 0])
    np.testing.assert_array_equal(binary64_matmul(a[0, 0], b), expected[0, 0])
    assert binary64_matmul(a[0, 0], b[:, 0]) == expected[0, 0, 0]
    np.testing.assert_array_equal(binary64_dot(a, b[:, 0]), expected[..., 0])
    squares = np.asarray(
        [[_ordered_fraction_dot(row, row) for row in batch] for batch in a]
    )
    np.testing.assert_array_equal(binary64_norm(a), np.sqrt(squares))
    np.testing.assert_array_equal(
        binary64_matmul(a[:1], np.stack([b, b])), expected[:1] + np.zeros((2, 2, 2))
    )


def test_binary64_hex_rounding_trees_and_fused_counterexample() -> None:
    first = np.asarray([1e16, -1e16, 1.0])
    second = np.ones(3)
    ordered = _ordered_fraction_dot(first, second)
    right_associated = float(
        Fraction(float(first[0] * second[0]))
        + Fraction(float(Fraction(float(first[1])) + Fraction(float(first[2]))))
    )
    # Two distinct results occur among these two association trees.
    assert len({float(ordered).hex(), float(right_associated).hex()}) == 2
    assert float(binary64_dot(first, second)).hex() == float(ordered).hex()
    assert float(binary64_matmul(first[None, :], second)[0]).hex() == float(ordered).hex()
    squared = _ordered_fraction_dot(first, first)
    assert float(binary64_norm(first)).hex() == float(math.sqrt(squared)).hex()

    # With two terms there is only one addition tree, but a fused final
    # multiply-add differs from the required separately rounded operations.
    delta = 2.0**-27
    a = np.asarray([-1.0, 1.0 + delta])
    b = np.asarray([1.0, 1.0 - delta])
    unfused = _ordered_fraction_dot(a, b)
    fused = float(Fraction(-1) + Fraction(float(a[1])) * Fraction(float(b[1])))
    assert len({float(unfused).hex(), float(fused).hex()}) == 2
    assert float(binary64_dot(a, b)).hex() == float(unfused).hex()
    assert float(binary64_matmul(a, b)).hex() == float(unfused).hex()


def test_ordered_dot_differs_from_alternative_rounding() -> None:
    first = [
        float.fromhex(value)
        for value in (
            "0x1.da5dcac364928p-1",
            "0x1.390c8604c6aa8p-2",
            "0x1.1ff5669fe18d4p+0",
            "0x1.4aeaf6ff72b34p-1",
        )
    ]
    second = [
        float.fromhex(value)
        for value in (
            "0x1.5b70f08258c26p+0",
            "-0x1.f07aec346c9a0p-3",
            "-0x1.613ab981b4f48p+0",
            "-0x1.6691a38290500p+0",
        )
    ]
    ordered = _ordered_fraction_dot(np.asarray(first), np.asarray(second))
    reverse = _ordered_fraction_dot(np.asarray(first[::-1]), np.asarray(second[::-1]))
    products = [Fraction(a) * Fraction(b) for a, b in zip(first, second, strict=True)]
    fused = float(products[0])
    for product in products[1:]:
        fused = float(Fraction(fused) + product)
    exact = float(sum(products, Fraction()))
    assert ordered != reverse
    assert ordered != fused
    assert ordered != exact
    assert (
        float(binary64_dot(np.asarray(first), np.asarray(second))).hex()
        == float(ordered).hex()
    )
    assert (
        float(binary64_matmul(np.asarray(first), np.asarray(second))).hex()
        == float(ordered).hex()
    )


@pytest.mark.parametrize("values", [(3.0, 4.0), (0.1, 0.2, 0.3), (1.0, 1.0)])
def test_binary64_norm_and_directed_bounds(values: tuple[float, ...]) -> None:
    exact_squared = sum((Fraction(value) ** 2 for value in values), Fraction())
    array = np.asarray(values, dtype=np.float64)
    lower = float(binary64_norm_lower(array))
    norm = float(binary64_norm(array))
    upper = float(binary64_norm_upper(array))
    assert lower <= norm <= upper
    assert Fraction(lower) ** 2 <= exact_squared <= Fraction(upper) ** 2
    rounded_squared = _ordered_fraction_dot(np.asarray(values), np.asarray(values))
    assert float(norm).hex() == float(math.sqrt(rounded_squared)).hex()


def test_binary64_ordered_products_zero_exceptional_and_shapes() -> None:
    negative_zero = np.asarray([-0.0])
    assert np.signbit(binary64_dot(negative_zero, np.asarray([1.0])))
    assert np.signbit(binary64_matmul(negative_zero, np.asarray([1.0])))
    assert not np.signbit(binary64_dot(np.empty(0), np.empty(0)))
    assert not np.signbit(binary64_norm(np.empty(0)))
    empty = binary64_matmul(np.empty((2, 3, 0)), np.empty((1, 0, 4)))
    assert empty.shape == (2, 3, 4)
    assert np.all(empty == 0) and not np.any(np.signbit(empty))
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        assert np.isnan(binary64_dot(np.asarray([np.inf]), np.asarray([0.0])))
        assert np.isinf(binary64_norm(np.asarray([1e308])))
        assert binary64_norm(np.asarray([1e-300])) == 0.0
    with pytest.raises(ValueError):
        binary64_dot(np.asarray([1.0]), np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError):
        binary64_matmul(np.asarray([1.0]), np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError):
        binary64_matmul(1.0, np.asarray([1.0]))


def _squared_norm(components: tuple[float, ...], /) -> Fraction:
    return sum((Fraction(value) ** 2 for value in components), Fraction(0))


# WIN-5 C5 and a LIN aarch64 two-component case: ``nextafter`` once after
# ``sqrt(sum of squares)`` or ``numpy.linalg.norm`` lies below the exact norm.
_ROUNDING_COUNTEREXAMPLES = {
    "win5-c5-three-squares": (
        float.fromhex("0x1.43d55519530b8p-6"),
        float.fromhex("0x1.70abf9fc8c4c0p+2"),
        float.fromhex("0x1.3cecce40c6800p-3"),
    ),
    "lin-linalg-norm-two-components": (
        float.fromhex("0x1.03a8e6c503eccp-1"),
        float.fromhex("0x1.77da20fd1661bp+1"),
    ),
    "negative-components": (-3.0, 0.1, -1e-12),
}


@pytest.mark.parametrize(
    "components",
    tuple(_ROUNDING_COUNTEREXAMPLES.values()),
    ids=tuple(_ROUNDING_COUNTEREXAMPLES),
)
def test_binary64_norm_bounds_enclose_the_exact_norm_within_ulps(
    components: tuple[float, ...],
) -> None:
    exact = _squared_norm(components)
    upper = float(binary64_norm_upper(np.asarray(components, dtype=np.float64)))
    lower = float(binary64_norm_lower(np.asarray(components, dtype=np.float64)))
    assert Fraction(lower) ** 2 <= exact <= Fraction(upper) ** 2
    assert Fraction(upper) ** 2 <= exact * (1 + Fraction(1, 2**47))
    assert Fraction(lower) ** 2 >= exact * (1 - Fraction(1, 2**47))


def test_binary64_norm_bounds_keep_zero_and_underflow_outward() -> None:
    rows = np.asarray(((0.0, -0.0, 0.0), (1e-200, -3e-201, 0.0)), dtype=np.float64)
    upper = binary64_norm_upper(rows)
    lower = binary64_norm_lower(rows)
    assert upper.shape == lower.shape == (2,)
    assert upper[0] == lower[0] == 0.0
    exact = _squared_norm((1e-200, -3e-201, 0.0))
    assert Fraction(float(lower[1])) ** 2 <= exact <= Fraction(float(upper[1])) ** 2
    with pytest.raises(ValueError, match="component axis"):
        binary64_norm_upper(np.float64(1.0))


def test_binary64_difference_magnitude_encloses_signed_differences() -> None:
    # WIN-5 C6: ``nextafter(p - q, +inf)`` shrinks negative differences.
    first = np.asarray(
        (
            float.fromhex("-0x1.48456c21b0cf2p-1"),
            float.fromhex("0x1.9b4987011ef3ep-1"),
            float.fromhex("-0x1.2773cb11b4cdep-1"),
        ),
        dtype=np.float64,
    )
    second = np.asarray(
        (
            float.fromhex("-0x1.4770d3afb54ebp-1"),
            float.fromhex("0x1.9b32525efa419p-1"),
            float.fromhex("-0x1.2632afb2e198ap-1"),
        ),
        dtype=np.float64,
    )
    lower, upper = binary64_difference_magnitude(first, second)
    differences = tuple(
        abs(Fraction(float(a)) - Fraction(float(b)))
        for a, b in zip(first, second, strict=True)
    )
    assert all(
        Fraction(float(low)) <= exact <= Fraction(float(high))
        for low, exact, high in zip(lower, differences, upper, strict=True)
    )
    exact = sum((value * value for value in differences), Fraction(0))
    assert Fraction(float(binary64_norm_lower(lower))) ** 2 <= exact
    assert exact <= Fraction(float(binary64_norm_upper(upper))) ** 2
    same_lower, same_upper = binary64_difference_magnitude(first, first)
    assert np.all(same_lower == 0.0) and np.all(same_upper == 0.0)


def test_binary64_lifts_and_directed_rational_endpoints_are_exact() -> None:
    value = 0.1
    assert rational_vector((value,)) == (Fraction.from_float(value),)

    rational = Fraction(1, 3)
    lower = fraction_endpoint(rational, upper=False)
    upper = fraction_endpoint(rational, upper=True)
    assert Fraction(lower) <= rational <= Fraction(upper)
    with pytest.raises(ValueError, match="exact"):
        exact_float(rational)


def test_rational_cross_and_translation_ranges_keep_exact_dimensions() -> None:
    assert rational_cross(
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
    ) == (Fraction(0), Fraction(0), Fraction(1))
    boxes = [
        (
            np.asarray((0.0, 0.0), dtype=np.float64),
            np.asarray((1.0, 1.0), dtype=np.float64),
        )
    ]
    assert translation_ranges(
        boxes,
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
    ) == (range(-1, 2), range(-1, 2))


def test_rational_endpoint_saturation_is_outward() -> None:
    maximum = Fraction(float(np.finfo(np.float64).max))
    assert fraction_endpoint(maximum + 1, upper=False, saturate=True) == float(
        np.finfo(np.float64).max
    )
    assert fraction_endpoint(maximum + 1, upper=True, saturate=True) == np.inf
