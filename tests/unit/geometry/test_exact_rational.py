import math
from collections.abc import Callable
from fractions import Fraction
from typing import Literal

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


@pytest.mark.parametrize("size", [2, 3])
def test_short_reductions_match_ordered_bits(size: int) -> None:
    rng = np.random.default_rng(481)
    special = np.asarray(
        [0.0, -0.0, np.inf, -np.inf, np.nan, np.nextafter(0.0, 1.0), 1e308],
        dtype=np.float64,
    )
    special = np.concatenate(
        (
            special,
            np.asarray([0x7FF8000000000123, 0xFFF0000000000123], dtype=np.uint64).view(
                np.float64
            ),
        )
    )
    for _ in range(500):
        first = (
            rng.choice(special, size=size)
            if rng.integers(2)
            else rng.standard_normal(size)
        )
        second = (
            rng.choice(special, size=size)
            if rng.integers(2)
            else rng.standard_normal(size)
        )
        with np.errstate(all="ignore"):
            expected = np.multiply(first[..., 0], second[..., 0])
            squared = np.multiply(first[..., 0], first[..., 0])
            for index in range(1, size):
                expected = np.add(
                    expected, np.multiply(first[..., index], second[..., index])
                )
                squared = np.add(
                    squared, np.multiply(first[..., index], first[..., index])
                )
            actual = binary64_dot(first, second)
            norm = binary64_norm(first)
            assert actual.shape == norm.shape == ()
            assert actual.dtype == norm.dtype == np.dtype(np.float64)
            assert actual.view(np.uint64) == np.asarray(expected).view(np.uint64)
            assert norm.view(np.uint64) == np.asarray(np.sqrt(squared)).view(np.uint64)
            if (
                np.all(np.isfinite(first))
                and np.all(np.isfinite(second))
                and np.all(np.abs(first) < 1e100)
                and np.all(np.abs(second) < 1e100)
            ):
                assert float(actual) == _ordered_fraction_dot(first, second)
                assert float(norm) == math.sqrt(_ordered_fraction_dot(first, first))


@pytest.mark.parametrize("rows", [0, 2, 3, 300])
def test_matrix_vector_fast_path_matches_ordered_bits(rows: int) -> None:
    rng = np.random.default_rng(57)
    special = np.asarray([0.0, -0.0, np.inf, -np.inf, np.nan, 1e308, 1e-310])
    special = np.concatenate(
        (
            special,
            np.asarray([0x7FF8000000000123, 0xFFF0000000000123], dtype=np.uint64).view(
                np.float64
            ),
        )
    )
    for _ in range(100):
        first = rng.choice(special, size=(rows, 3))
        second = rng.choice(special, size=3)
        right = second[:, None]
        with np.errstate(all="ignore"):
            expected = np.multiply(first[..., :, 0, None], right[..., None, 0, :])
            for index in (1, 2):
                expected = np.add(
                    expected,
                    np.multiply(first[..., :, index, None], right[..., None, index, :]),
                )
            actual = binary64_matmul(first, second)
        np.testing.assert_array_equal(
            actual.view(np.uint64), np.squeeze(expected, axis=-1).view(np.uint64)
        )
    first = rng.standard_normal((rows, 3))
    second = rng.standard_normal(3)
    actual = binary64_matmul(first, second)
    for index in range(rows):
        assert float(actual[index]) == _ordered_fraction_dot(first[index], second)


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [
        ((3,), (3, 2)),
        ((3,), (3, 3)),
        ((2, 3), (3, 2)),
        ((2, 3), (3, 3)),
        ((3, 3), (3, 2)),
        ((3, 3), (3, 3)),
    ],
)
def test_small_matrix_products_match_ordered_bits(
    left_shape: tuple[int, ...], right_shape: tuple[int, ...]
) -> None:
    rng = np.random.default_rng(218)
    special = np.asarray(
        [0.0, -0.0, np.inf, -np.inf, np.nan, 1e308, 1e-310], dtype=np.float64
    )
    for index in range(200):
        if index % 2:
            first = rng.standard_normal(left_shape)
            second = rng.standard_normal(right_shape)
        else:
            first = rng.choice(special, size=left_shape)
            second = rng.choice(special, size=right_shape)
        left = first[None, :] if first.ndim == 1 else first
        with np.errstate(all="ignore"):
            expected = np.multiply(left[..., :, 0, None], second[..., None, 0, :])
            for term in (1, 2):
                expected = np.add(
                    expected,
                    np.multiply(left[..., :, term, None], second[..., None, term, :]),
                )
            if first.ndim == 1:
                expected = np.squeeze(expected, axis=-2)
            actual = binary64_matmul(first, second)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        if index % 2:
            for row in range(left.shape[0]):
                for column in range(second.shape[1]):
                    result = actual[column] if first.ndim == 1 else actual[row, column]
                    assert float(result) == _ordered_fraction_dot(
                        left[row], second[:, column]
                    )


@pytest.mark.parametrize("operation", [binary64_dot, binary64_matmul])
@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
@pytest.mark.parametrize(
    "first,second,category",
    [
        ([1e308, 1.0], [1e308, 1.0], "overflow"),
        ([np.inf, 1.0], [0.0, 1.0], "invalid"),
    ],
)
def test_ordered_reduction_float_errors(
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
    mode: Literal["ignore", "warn", "raise"],
    first: list[float],
    second: list[float],
    category: str,
) -> None:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    with np.errstate(all=mode):
        if mode == "raise":
            with pytest.raises(FloatingPointError, match=category):
                operation(a, b)
        elif mode == "warn":
            with pytest.warns(RuntimeWarning, match=category):
                operation(a, b)
        else:
            operation(a, b)


@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
@pytest.mark.parametrize("matrix", [False, True])
def test_norm_and_matrix_fast_path_overflow(
    mode: Literal["ignore", "warn", "raise"], matrix: bool
) -> None:
    values = np.asarray([1e308, 1.0, 1.0], dtype=np.float64)

    def operation() -> np.ndarray:
        return (
            binary64_matmul(np.tile(values, (3, 1)), values)
            if matrix
            else binary64_norm(values)
        )

    with np.errstate(all=mode):
        if mode == "raise":
            with pytest.raises(FloatingPointError, match="overflow"):
                operation()
        elif mode == "warn":
            with pytest.warns(RuntimeWarning, match="overflow"):
                operation()
        else:
            operation()


@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
def test_small_matrix_invalid_warning(mode: Literal["ignore", "warn", "raise"]) -> None:
    matrix = np.asarray([[np.inf, 1.0, 1.0]] * 3, dtype=np.float64)
    vector = np.asarray([0.0, 1.0, 1.0], dtype=np.float64)
    with np.errstate(invalid=mode):
        if mode == "raise":
            with pytest.raises(FloatingPointError, match="invalid"):
                binary64_matmul(matrix, vector)
        elif mode == "warn":
            with pytest.warns(RuntimeWarning, match="invalid"):
                binary64_matmul(matrix, vector)
        else:
            binary64_matmul(matrix, vector)


def test_short_scalar_paths_do_not_report_underflow() -> None:
    vector = np.asarray([1e-200, 1.0, 1.0], dtype=np.float64)
    matrix = np.tile(vector, (3, 1))
    with np.errstate(under="ignore"):
        expected_dot = binary64_dot(vector[None, :], vector)[0]
        expected_norm = binary64_norm(vector[None, :])[0]
        expected_matrix = binary64_matmul(matrix[None, :, :], vector)[0]
    with np.errstate(under="raise"):
        assert binary64_dot(vector, vector).view(np.uint64) == expected_dot.view(
            np.uint64
        )
        assert binary64_norm(vector).view(np.uint64) == expected_norm.view(np.uint64)
        np.testing.assert_array_equal(
            binary64_matmul(matrix, vector).view(np.uint64),
            expected_matrix.view(np.uint64),
        )
        with pytest.raises(FloatingPointError, match="underflow"):
            binary64_dot(vector[None, :], vector)
        with pytest.raises(FloatingPointError, match="underflow"):
            binary64_norm(vector[None, :])
        with pytest.raises(FloatingPointError, match="underflow"):
            binary64_matmul(matrix[None, :, :], vector)


def test_ordered_reductions_reject_complex() -> None:
    value = np.asarray([1 + 2j, 3])
    with pytest.raises(TypeError):
        binary64_dot(value, value)
    with pytest.raises(TypeError):
        binary64_norm(value)
    with pytest.raises(TypeError):
        binary64_matmul(value, value)
    values = [1 + 2j, 3]
    with pytest.raises(TypeError):
        binary64_dot(values, values)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        binary64_norm(values)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        binary64_matmul(values, values)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "values",
    [
        [1, 0, 1],
        np.asarray([1, 0, 1], dtype=np.int64),
        np.asarray([1, 0, 1], dtype=np.float32),
        np.asarray([True, False, True], dtype=np.bool_),
    ],
)
def test_ordered_reductions_convert_real_inputs(values: np.ndarray | list[int]) -> None:
    converted = np.asarray(values, dtype=np.float64)
    assert binary64_dot(values, converted) == binary64_dot(converted, converted)  # ty: ignore[invalid-argument-type]
    assert binary64_matmul(values, converted) == binary64_matmul(converted, converted)  # ty: ignore[invalid-argument-type]
    assert binary64_norm(values) == binary64_norm(converted)  # ty: ignore[invalid-argument-type]


def test_ordered_norm_rejects_pairwise_summation() -> None:
    encoded = (
        "0x1.8c43f79a2db24p-1 0x1.c16959869e47ep-2 "
        "0x1.b79a2584ddb42p-1 0x1.650d6c1c2c011p-1 "
        "0x1.81c01b5339380p-4 0x1.f384c5db0d8eap-1 "
        "0x1.85b41a5f78b48p-1 0x1.927705476f592p-1 "
        "0x1.066070b41b168p-3 0x1.cd31f8a75be6ep-2 "
        "0x1.7bb27a2e005c4p-2 0x1.da80f0cc5d772p-1 "
        "0x1.49a8b06390299p-1 0x1.a541029ace094p-1 "
        "0x1.c60e5f2a2e84ep-2 0x1.d16288f39a8a8p-3 "
        "0x1.1bf289862f2c7p-1 0x1.05653e3e34cb8p-4 "
        "0x1.a7bf45e1b771fp-1 0x1.436984208db37p-1 "
        "0x1.842413865498ep-1 0x1.6b08dafacd55ep-2 "
        "0x1.f0ff54da22c66p-1 0x1.c9472beee4312p-1 "
        "0x1.8e88481f5b183p-1 0x1.8e9ebd2614460p-3 "
        "0x1.ddec1c5d74a3cp-2 0x1.66d727afede80p-5 "
        "0x1.3bfc21145f5fcp-3 0x1.5db897a77c9bcp-1 "
        "0x1.7d51771dd8559p-1 0x1.ef5d6f86bbb89p-1"
    )
    values = np.asarray(
        [float.fromhex(value) for value in encoded.split()], dtype=np.float64
    )
    assert float(binary64_norm(values)).hex() == "0x1.d4b6e144fc507p+1"


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


@pytest.mark.meshcore
def test_portable_angles_match_native_bits_on_generated_and_special_inputs() -> None:
    from phydrax._meshcore import angle_arctangent2, angle_between
    from phydrax.geometry._rational import binary64_angle_between, binary64_arctangent2

    rng = np.random.default_rng(20261118)
    # Full bit patterns cover subnormals, extreme ratios and nonfinite inputs;
    # bounded pairs also exercise the polynomial's reduction intervals densely.
    y = rng.integers(0, 2**64, size=1_000_000, dtype=np.uint64).view(np.float64)
    x = rng.integers(0, 2**64, size=1_000_000, dtype=np.uint64).view(np.float64)
    special = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            np.inf,
            -np.inf,
            np.nan,
            np.nextafter(0.0, 1.0),
            -np.nextafter(0.0, 1.0),
            np.finfo(np.float64).max,
            -np.finfo(np.float64).max,
        ]
    )
    yy, xx = np.meshgrid(special, special)
    y = np.concatenate((y, rng.uniform(-1.0, 1.0, 100_000), yy.ravel()))
    x = np.concatenate((x, rng.uniform(-1.0, 1.0, 100_000), xx.ravel()))
    for portable, native, first in (
        (binary64_angle_between, angle_between, y),
        (binary64_arctangent2, angle_arctangent2, y),
    ):
        actual = portable(first, x)
        expected = native(first, x)
        assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert binary64_angle_between(
        np.array([[1.0], [0.0]]), np.array([1.0, -1.0])
    ).shape == (2, 2)


def _integer_rounded_cube_root(value: float) -> float:
    """Correctly rounded positive cube root using integer midpoint comparisons."""
    numerator, denominator = value.as_integer_ratio()
    exponent = (math.frexp(value)[1] - 1) // 3
    shift = 52 - exponent
    if shift >= 0:
        numerator <<= 3 * shift
    else:
        denominator <<= -3 * shift
    low, high = 1 << 52, 1 << 53
    while low + 1 < high:
        middle = (low + high) // 2
        if middle**3 * denominator <= numerator:
            low = middle
        else:
            high = middle
    midpoint_cube = (2 * low + 1) ** 3 * denominator
    if 8 * numerator > midpoint_cube or (8 * numerator == midpoint_cube and low & 1):
        low += 1
    return math.ldexp(float(low), -shift)


def test_binary64_cube_root_against_integer_rounding() -> None:
    from phydrax.geometry._rational import binary64_cube_root

    rng = np.random.default_rng(20261118)
    bits = rng.integers(0, 0x7FF0000000000000, size=1_000_000, dtype=np.uint64)
    values = np.concatenate(
        (
            bits[10:20_006].view(np.float64),
            np.array(
                [
                    np.nextafter(0.0, 1.0),
                    np.finfo(np.float64).tiny,
                    np.finfo(np.float64).max,
                    1.0,
                    8.0,
                    27.0,
                ]
            ),
        )
    )
    actual = binary64_cube_root(values)
    expected = np.array([_integer_rounded_cube_root(float(x)) for x in values])
    # The native Newton iteration is not correctly rounded.  One ULP was the
    # maximum observed on this fixed stream, not a claimed universal bound.
    distances = np.abs(actual.view(np.int64) - expected.view(np.int64))
    assert int(np.max(distances)) <= 1

    # The seventh and eighth native Newton steps differ for these values;
    # the integer midpoint oracle independently establishes the exact rounding.
    iteration_inputs = np.asarray(
        [
            float.fromhex("0x1.aaae9cfed9cc2p-41"),
            float.fromhex("0x1.5f1994d024085p-353"),
        ],
        dtype=np.float64,
    )
    rounded = np.asarray(
        [
            float.fromhex("0x1.7e6af5a5e36bap-14"),
            float.fromhex("0x1.665ae5f8547bap-118"),
        ],
        dtype=np.float64,
    )
    eighth = np.asarray(
        [
            float.fromhex("0x1.7e6af5a5e36bbp-14"),
            float.fromhex("0x1.665ae5f8547b9p-118"),
        ],
        dtype=np.float64,
    )
    oracle = np.asarray(
        [_integer_rounded_cube_root(float(value)) for value in iteration_inputs],
        dtype=np.float64,
    )
    assert np.array_equal(oracle.view(np.uint64), rounded.view(np.uint64))
    assert np.array_equal(
        binary64_cube_root(iteration_inputs).view(np.uint64), eighth.view(np.uint64)
    )

    unchanged_bits = np.array(
        [
            0,
            0x8000000000000000,
            0xBFF0000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000001234,
            0xFFF8000000004321,
        ],
        dtype=np.uint64,
    )
    unchanged = unchanged_bits.view(np.float64)
    assert np.array_equal(binary64_cube_root(unchanged).view(np.uint64), unchanged_bits)
