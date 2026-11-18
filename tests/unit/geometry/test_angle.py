"""The portable angles as host code sees them.

Stated values with their exact bits, the quadrants of ``angle_between``,
agreement with NumPy to a few binary64 values on generated arguments (NumPy's
functions are within a value or two of the true ones on every platform), and
the refusals. That every platform returns the same doubles is held by the
native bit vectors of ``test_angle``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from phydrax._meshcore import (
    angle_arccosine,
    angle_arcsine,
    angle_arctangent2,
    angle_between,
    angle_cosine,
    angle_sine,
)


_PI = float.fromhex("0x1.921fb54442d18p+1")


def _steps(values: np.ndarray, reference: np.ndarray, /) -> np.ndarray:
    """Distance of each value from its reference in binary64 values."""

    def ordinal(array: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(array, dtype=np.float64).view(np.int64)
        return np.where(bits >= 0, bits, -(bits & np.int64(0x7FFFFFFFFFFFFFFF)))

    return np.abs(ordinal(values) - ordinal(reference))


def test_stated_values_have_their_bits() -> None:
    arguments = np.asarray((0.0, -0.0, 0.5, -0.5, 1.0, -1.0))
    arcsine = angle_arcsine(arguments)
    assert [value.hex() for value in arcsine.tolist()] == [
        "0x0.0p+0",
        "-0x0.0p+0",
        "0x1.0c152382d7366p-1",
        "-0x1.0c152382d7366p-1",
        "0x1.921fb54442d18p+0",
        "-0x1.921fb54442d18p+0",
    ]
    arccosine = angle_arccosine(np.asarray((1.0, 0.5, 0.0, -1.0)))
    assert [value.hex() for value in arccosine.tolist()] == [
        "0x0.0p+0",
        "0x1.0c152382d7366p+0",
        "0x1.921fb54442d18p+0",
        "0x1.921fb54442d18p+1",
    ]
    assert angle_sine(np.asarray((0.0,)))[0] == 0.0
    assert angle_cosine(np.asarray((0.0,)))[0] == 1.0
    # Outside the domain and for values that are not finite: NaN.
    assert np.all(np.isnan(angle_arcsine(np.asarray((1.0000000000000002, np.nan)))))
    assert np.all(np.isnan(angle_arccosine(np.asarray((-2.0, np.inf)))))
    assert np.all(np.isnan(angle_sine(np.asarray((np.inf, np.nan)))))
    assert np.all(np.isnan(angle_cosine(np.asarray((-np.inf, np.nan)))))


def test_between_follows_the_sign_of_its_second_argument() -> None:
    across = np.asarray((0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0))
    along = np.asarray((0.0, 3.0, 0.0, -0.0, 1.0, -1.0, -1.0))
    angles = angle_between(across, along)
    assert [value.hex() for value in angles.tolist()] == [
        "0x0.0p+0",
        "0x0.0p+0",
        "0x1.921fb54442d18p+0",
        "0x1.921fb54442d18p+0",
        "0x1.921fb54442d18p-1",
        "0x1.2d97c7f3321d2p+1",
        "0x1.921fb54442d18p+1",
    ]
    invalid = angle_between(
        np.asarray((-1.0, np.nan, np.inf, 1.0)), np.asarray((1.0, 1.0, 1.0, np.inf))
    )
    assert np.all(np.isnan(invalid))
    with pytest.raises(ValueError, match="one shape"):
        angle_between(np.zeros((3,)), np.zeros((2,)))


def test_arctangent2_takes_the_sign_of_its_ordinate_and_ieee_zeros() -> None:
    ordinates = np.asarray(
        (0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 1.0, -1.0, 2.0, -2.0)
    )
    abscissas = np.asarray(
        (0.0, 0.0, -0.0, -0.0, -3.0, -3.0, 1.0, 1.0, -1.0, -1.0, 0.0, -0.0)
    )
    angles = angle_arctangent2(ordinates, abscissas)
    assert [value.hex() for value in angles.tolist()] == [
        "0x0.0p+0",
        "-0x0.0p+0",
        "0x1.921fb54442d18p+1",
        "-0x1.921fb54442d18p+1",
        "0x1.921fb54442d18p+1",
        "-0x1.921fb54442d18p+1",
        "0x1.921fb54442d18p-1",
        "-0x1.921fb54442d18p-1",
        "0x1.2d97c7f3321d2p+1",
        "-0x1.2d97c7f3321d2p+1",
        "0x1.921fb54442d18p+0",
        "-0x1.921fb54442d18p+0",
    ]
    # The same zeros and quadrants as the platform function.
    assert np.array_equal(
        np.signbit(angles), np.signbit(np.arctan2(ordinates, abscissas))
    )
    generator = np.random.default_rng(20261119)
    y = generator.uniform(-10.0, 10.0, size=2000)
    x = generator.uniform(-10.0, 10.0, size=2000)
    signed = angle_arctangent2(y, x)
    assert int(np.max(_steps(signed, np.arctan2(y, x)))) <= 6
    # The magnitude is the unsigned angle, bit for bit.
    assert np.array_equal(np.abs(signed), angle_between(np.abs(y), x))
    assert np.all(
        np.isnan(angle_arctangent2(np.asarray((np.inf, 1.0)), np.asarray((1.0, np.nan))))
    )
    with pytest.raises(ValueError, match="one shape"):
        angle_arctangent2(np.zeros((3,)), np.zeros((2,)))


_VALID = np.asarray((0.5,))
_ARGUMENTS = (
    pytest.param(angle_sine, 1, 0, id="sine"),
    pytest.param(angle_cosine, 1, 0, id="cosine"),
    pytest.param(angle_arcsine, 1, 0, id="arcsine"),
    pytest.param(angle_arccosine, 1, 0, id="arccosine"),
    pytest.param(angle_between, 2, 0, id="between-across"),
    pytest.param(angle_between, 2, 1, id="between-along"),
    pytest.param(angle_arctangent2, 2, 0, id="arctangent2-ordinates"),
    pytest.param(angle_arctangent2, 2, 1, id="arctangent2-abscissas"),
)
_WRONG_KIND = (
    pytest.param(np.asarray((1.0 + 2.0j,)), id="complex"),
    pytest.param(np.asarray((True,)), id="bool"),
    pytest.param(np.asarray(("1.0",)), id="text"),
    pytest.param(np.asarray((None,), dtype=np.object_), id="object"),
)


def _with_argument(
    function: Callable[..., np.ndarray], arity: int, position: int, values: np.ndarray, /
) -> np.ndarray:
    arguments = [_VALID] * arity
    arguments[position] = values
    return function(*arguments)


@pytest.mark.parametrize(("function", "arity", "position"), _ARGUMENTS)
@pytest.mark.parametrize("values", _WRONG_KIND)
def test_an_argument_of_the_wrong_kind_is_refused(
    function: Callable[..., np.ndarray], arity: int, position: int, values: np.ndarray
) -> None:
    with pytest.raises(TypeError, match="real floating or integer"):
        _with_argument(function, arity, position, values)


@pytest.mark.parametrize(("function", "arity", "position"), _ARGUMENTS)
def test_an_integer_argument_is_taken_exactly_or_refused(
    function: Callable[..., np.ndarray], arity: int, position: int
) -> None:
    # An integer that binary64 holds is its value; one beyond 2**53 is not
    # rounded silently.
    exact = _with_argument(function, arity, position, np.asarray((1,), dtype=np.int64))
    assert np.array_equal(
        exact,
        _with_argument(function, arity, position, np.asarray((1.0,))),
        equal_nan=True,
    )
    with pytest.raises(ValueError, match="exactly representable"):
        _with_argument(
            function, arity, position, np.asarray((2**53 + 1,), dtype=np.uint64)
        )


def test_angles_agree_with_numpy_to_a_few_values_and_keep_their_shape() -> None:
    generator = np.random.default_rng(20261118)
    unit = generator.uniform(-1.0, 1.0, size=(40, 50))
    radians = generator.uniform(-8.0, 8.0, size=(40, 50))
    across = np.ldexp(
        generator.uniform(0.0, 1.0, size=2000), generator.integers(-20, 20, 2000)
    )
    along = generator.uniform(-1.0, 1.0, size=2000)
    assert angle_arcsine(unit).shape == unit.shape
    assert int(np.max(_steps(angle_arcsine(unit), np.arcsin(unit)))) <= 6
    assert int(np.max(_steps(angle_arccosine(unit), np.arccos(unit)))) <= 6
    assert (
        int(np.max(_steps(angle_between(across, along), np.arctan2(across, along)))) <= 6
    )
    sine, cosine = angle_sine(radians), angle_cosine(radians)
    assert float(np.max(np.abs(sine - np.sin(radians)))) <= 2.0**-51
    assert float(np.max(np.abs(cosine - np.cos(radians)))) <= 2.0**-51
    # Lower endpoints: never above the platform value by more than its error.
    assert float(np.max(sine * sine + cosine * cosine)) <= 1.0 + 2.0**-51
    assert np.all(
        (angle_between(across, along) >= 0.0) & (angle_between(across, along) <= _PI)
    )
    # An empty array is an empty result.
    assert angle_arcsine(np.zeros((0,))).shape == (0,)
