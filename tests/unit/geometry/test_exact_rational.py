from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry._rational import (
    binary64_difference_magnitude,
    binary64_norm_lower,
    binary64_norm_upper,
    exact_float,
    fraction_endpoint,
    rational_cross,
    rational_vector,
    translation_ranges,
)


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
