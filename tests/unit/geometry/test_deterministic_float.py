"""Error bounds of the deterministic binary64 kernels on their recorded vectors.

``native/meshcore/tests/deterministic_float_vectors.inc`` holds the inputs and
the output bits that the native test requires on every platform. These tests
hold the recorded outputs to the bounds stated at the kernels, against
references evaluated with ``decimal`` at 80 digits and with exact rationals.
They read the recorded file only, so they are independent of the kernels.
"""

from __future__ import annotations

import math
import re
import struct
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest


_VECTORS = (
    Path(__file__).resolve().parents[3]
    / "native"
    / "meshcore"
    / "tests"
    / "deterministic_float_vectors.inc"
)
_NAN_BITS = 0x7FF8000000000000
_EXP_OVERFLOW = float.fromhex("0x1.62e42fefa39efp+9")
_EXP_UNDERFLOW = float.fromhex("-0x1.74910d52d3052p+9")
_UNIT_ROUNDOFF = Fraction(1, 2**53)


def _binary64(bits: int, /) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _rows(macro: str, /) -> list[list[int]]:
    """Integer fields of every recorded row of one macro, in file order."""
    rows = []
    for line in _VECTORS.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(rf"{macro}\((.*)\)", line)
        if match is not None:
            rows.append(
                [int(field, 0) for field in re.findall(r"0x[0-9a-f]+|\b\d+\b", match[1])]
            )
    return rows


def _scalar_rows(kernel: str, /) -> list[tuple[float, int]]:
    rows = []
    for line in _VECTORS.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(
            rf"PHX_DF_SCALAR\({kernel}, (0x[0-9a-f]{{16}}), (0x[0-9a-f]{{16}})\)", line
        )
        if match is not None:
            rows.append((_binary64(int(match[1], 16)), int(match[2], 16)))
    return rows


def _within_ulps(value: float, reference: Decimal, ulps: int, /) -> bool:
    return abs(Decimal(value) - reference) <= ulps * Decimal(math.ulp(value))


def test_recorded_exponentials_are_within_four_ulp() -> None:
    rows = _scalar_rows("exp")
    checked = 0
    with localcontext() as context:
        context.prec = 80
        for x, bits in rows:
            y = _binary64(bits)
            if math.isnan(x):
                assert math.isnan(y)
            elif x > _EXP_OVERFLOW:
                assert y == math.inf
            elif x <= _EXP_UNDERFLOW:
                assert bits == 0
            else:
                assert _within_ulps(y, Decimal(x).exp(), 4), repr(x)
                checked += 1
    assert checked > 200


def test_recorded_logarithms_are_within_six_ulp() -> None:
    rows = _scalar_rows("log")
    checked = 0
    with localcontext() as context:
        context.prec = 80
        for x, bits in rows:
            y = _binary64(bits)
            if math.isnan(x) or x < 0.0:
                assert bits == _NAN_BITS or math.isnan(x)
                assert math.isnan(y)
            elif x == 0.0:
                assert y == -math.inf
            elif x == math.inf:
                assert y == math.inf
            else:
                assert _within_ulps(y, Decimal(x).ln(), 6), repr(x)
                checked += 1
    assert checked > 200


def test_recorded_two_thirds_powers_are_within_the_stated_relative_error() -> None:
    rows = _scalar_rows("pow_two_thirds")
    checked = 0
    with localcontext() as context:
        context.prec = 80
        for x, bits in rows:
            y = _binary64(bits)
            if math.isnan(x) or x < 0.0:
                assert math.isnan(y)
            elif x == 0.0:
                assert bits == 0
            elif x == math.inf:
                assert y == math.inf
            elif 2.0**-128 <= x <= 2.0**128:
                reference = (Decimal(x).ln() * 2 / 3).exp()
                assert abs(Decimal(y) - reference) <= reference * Decimal(2) ** -43, repr(
                    x
                )
                checked += 1
    assert checked > 100


def test_recorded_eigen_decompositions_diagonalize_their_matrices() -> None:
    # Exact rationals. The kernels state no proved bound for the Jacobi
    # iteration; these are the bounds its recorded results are held to. The
    # absolute term covers results rounded in the subnormal range.
    largest = Fraction(sys.float_info.max)
    smallest = Fraction(1, 2**1074)
    converged_rows = 0
    nonfinite_rows = 0
    unrepresentable_rows = 0
    for fields in _rows("PHX_DF_EIGEN"):
        entries = [_binary64(bits) for bits in fields[:6]]
        converged = fields[6]
        values = [_binary64(bits) for bits in fields[7:10]]
        vectors = [_binary64(bits) for bits in fields[10:19]]
        if not all(math.isfinite(entry) for entry in entries):
            assert converged == 0
            assert values == [0.0] * 3 and vectors == [0.0] * 9
            nonfinite_rows += 1
            continue
        a00, a01, a02, a11, a12, a22 = (Fraction(entry) for entry in entries)
        matrix = ((a00, a01, a02), (a01, a11, a12), (a02, a12, a22))
        if converged == 0:
            # A finite matrix is refused only for an eigenvalue beyond the
            # binary64 range: some Rayleigh quotient exceeds the largest number.
            assert values == [0.0] * 3 and vectors == [0.0] * 9
            assert any(
                abs(sum(x[i] * matrix[i][j] * x[j] for i in range(3) for j in range(3)))
                > largest * sum(component * component for component in x)
                for x in product(range(-2, 3), repeat=3)
                if any(x)
            )
            unrepresentable_rows += 1
            continue
        assert values[0] <= values[1] <= values[2]
        columns = [
            [Fraction(vectors[3 * row + column]) for row in range(3)]
            for column in range(3)
        ]
        scale = max(abs(entry) for row in matrix for entry in row)
        tolerance = 32 * _UNIT_ROUNDOFF * scale + 4 * smallest
        for column, value in zip(columns, values, strict=True):
            for row in range(3):
                image = sum(matrix[row][k] * column[k] for k in range(3))
                assert abs(image - Fraction(value) * column[row]) <= tolerance
        for first in range(3):
            for second in range(first, 3):
                dot = sum(columns[first][k] * columns[second][k] for k in range(3))
                assert abs(dot - (first == second)) <= 16 * _UNIT_ROUNDOFF
        converged_rows += 1
    assert converged_rows > 40
    assert nonfinite_rows == 2
    assert unrepresentable_rows == 2


@pytest.mark.parametrize("kernel", ["exp", "log", "pow_two_thirds"])
def test_recorded_scalar_vectors_cover_special_values(kernel: str) -> None:
    inputs = [x for x, _ in _scalar_rows(kernel)]
    assert any(math.isnan(x) for x in inputs)
    assert any(x == math.inf for x in inputs)
    assert any(x == 0.0 for x in inputs)
