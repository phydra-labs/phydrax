"""The recorded ``log1p`` and small solves against independent references.

``native/meshcore/tests/deterministic_solve_vectors.inc`` holds the inputs and
the output bits that the native test requires on every platform. These tests
hold the recorded outputs to the contracts stated at the kernels, evaluated
here with ``decimal``, exact rationals and LAPACK. They read the recorded file
only, so they are independent of the kernels.
"""

from __future__ import annotations

import math
import re
import struct
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np


_VECTORS = (
    Path(__file__).resolve().parents[3]
    / "native"
    / "meshcore"
    / "tests"
    / "deterministic_solve_vectors.inc"
)
_NAN_BITS = 0x7FF8000000000000
_UNIT_ROUNDOFF = Fraction(1, 2**53)
_RANK_CUTOFF = 2.0**-40
_NO_SOLUTION = 255


def _binary64(bits: int, /) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _rows(kind: str, /) -> list[list[int]]:
    rows = []
    for line in _VECTORS.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(rf"PHX_DS_{kind}\((.*)\)", line)
        if match is not None:
            rows.append([int(field, 16) for field in match[1].split(", ")])
    return rows


def _values(fields: list[int], /) -> np.ndarray:
    return np.asarray([_binary64(bits) for bits in fields], dtype=np.float64)


def _symmetric(upper: np.ndarray, /) -> np.ndarray:
    m00, m01, m02, m11, m12, m22 = upper
    return np.asarray(
        ((m00, m01, m02), (m01, m11, m12), (m02, m12, m22)), dtype=np.float64
    )


def test_recorded_log1p_values_are_within_seven_ulp() -> None:
    checked = 0
    with localcontext() as context:
        context.prec = 120
        for input_bits, output_bits in _rows("LOG1P"):
            x = _binary64(input_bits)
            y = _binary64(output_bits)
            if math.isnan(x):
                assert math.isnan(y)
            elif x < -1.0:
                assert output_bits == _NAN_BITS
            elif x == -1.0:
                assert y == -math.inf
            elif x == math.inf:
                assert y == math.inf
            elif x == 0.0:
                assert output_bits == input_bits
            else:
                reference = (1 + Decimal(x)).ln()
                assert abs(Decimal(y) - reference) <= 7 * Decimal(math.ulp(y)), repr(x)
                checked += 1
    assert checked > 150


def test_recorded_pseudo_solutions_are_minimum_norm_least_squares_solutions() -> None:
    solved = 0
    deficient = 0
    refused = 0
    for row in _rows("PSEUDO"):
        order = row[0]
        matrix = _symmetric(_values(row[1:7]))[:order, :order]
        rhs = _values(row[7:10])[:order]
        rank = row[10]
        solution = _values(row[11:14])
        assert not solution[order:].any()
        solution = solution[:order]
        if rank == _NO_SOLUTION:
            # Refused: a value that is not finite, or a solution beyond the range.
            assert not solution.any()
            finite = bool(np.all(np.isfinite(matrix)) and np.all(np.isfinite(rhs)))
            if finite:
                with np.errstate(all="ignore"):
                    reference = np.linalg.lstsq(matrix, rhs, rcond=_RANK_CUTOFF)[0]
                assert not np.all(np.isfinite(reference))
            refused += 1
            continue
        # Scale to the unit range: LAPACK then sees no overflow or underflow.
        scale = float(np.max(np.abs(matrix)))
        if scale == 0.0 or float(np.linalg.eigvalsh(matrix / scale)[-1]) <= 0.0:
            assert rank == 0
            assert not solution.any()
            continue
        eigenvalues = np.linalg.eigvalsh(matrix / scale)
        # The recorded rank is the number of eigenvalues above the cutoff; skip
        # the comparison when an eigenvalue sits at the cutoff itself.
        ratios = eigenvalues / eigenvalues[-1]
        if np.any(np.abs(np.log2(np.maximum(ratios, 2.0**-200)) + 40.0) < 1.0):
            continue
        assert rank == int(np.count_nonzero(ratios > _RANK_CUTOFF))
        reference = np.linalg.lstsq(matrix / scale, rhs / scale, rcond=_RANK_CUTOFF)[0]
        assert np.allclose(
            solution, reference, rtol=1.0e-9, atol=1.0e-9 * np.max(np.abs(reference))
        )
        solved += 1
        deficient += rank < order
    assert solved > 40
    assert deficient > 8
    assert refused == 3


def test_recorded_cholesky_factors_reproduce_their_matrices() -> None:
    factored = 0
    refused = 0
    actions = 0
    factors: dict[tuple[int, ...], np.ndarray] = {}
    for row in _rows("CHOLESKY"):
        key = tuple(row[:6])
        upper = _values(row[:6])
        if row[6] == 2:
            # U v and the solution of U x = v for v = (1, -2, 1/2).
            factor = factors[key]
            vector = np.asarray((1.0, -2.0, 0.5), dtype=np.float64)
            assert np.allclose(
                _values(row[7:10]), factor @ vector, rtol=1.0e-13, atol=0.0
            )
            assert np.allclose(
                factor @ _values(row[10:13]), vector, rtol=1.0e-10, atol=1.0e-10
            )
            actions += 1
            continue
        u00, u01, u02, u11, u12, u22 = (Fraction(value) for value in _values(row[7:13]))
        if row[6] == 0:
            assert not _values(row[7:13]).any()
            # No factor: not finite, or not positive definite within rounding.
            if np.all(np.isfinite(upper)):
                scale = float(np.max(np.abs(upper)))
                smallest = (
                    -1.0
                    if scale == 0.0
                    else np.linalg.eigvalsh(_symmetric(upper) / scale)[0]
                )
                assert smallest < 1.0e-12
            refused += 1
            continue
        assert u00 > 0 and u11 > 0 and u22 > 0
        exact = [Fraction(value) for value in upper]
        product = [
            u00 * u00,
            u00 * u01,
            u00 * u02,
            u01 * u01 + u11 * u11,
            u01 * u02 + u11 * u12,
            u02 * u02 + u12 * u12 + u22 * u22,
        ]
        scale = max(abs(value) for value in exact)
        for reproduced, entry in zip(product, exact, strict=True):
            assert abs(reproduced - entry) <= 16 * _UNIT_ROUNDOFF * scale
        factors[key] = np.asarray(
            ((u00, u01, u02), (0, u11, u12), (0, 0, u22)), dtype=np.float64
        )
        factored += 1
    assert factored > 25
    assert actions == factored
    assert refused >= 8


def test_recorded_bases_are_orthonormal_and_span_their_vectors() -> None:
    spanned = 0
    dropped = 0
    for row in _rows("BASIS"):
        count = row[0]
        vectors = _values(row[1:10]).reshape((3, 3))[:count]
        size = row[10]
        basis = _values(row[11:20]).reshape((3, 3))
        assert not basis[size:].any()
        basis = basis[:size]
        assert size <= count
        gram = basis @ basis.T
        assert np.allclose(gram, np.eye(size), rtol=0.0, atol=2.0**-30)
        assert np.allclose(np.diag(gram), 1.0, rtol=0.0, atol=8.0 * float(_UNIT_ROUNDOFF))
        for vector in vectors:
            with np.errstate(all="ignore"):
                squared = float(vector @ vector)
            if not (np.isfinite(squared) and squared > 0.0):
                dropped += 1
                continue
            remainder = vector - (basis @ vector) @ basis
            # A vector is in the span up to the stated dependence cutoff.
            assert float(remainder @ remainder) <= 2.0**-39 * squared
            spanned += 1
        with np.errstate(all="ignore"):
            finite = [
                vector
                for vector in vectors
                if np.all(np.isfinite(vector))
                and np.isfinite(float(vector @ vector))
                and vector.any()
            ]
        if finite:
            singular = np.linalg.svd(
                np.asarray(finite) / np.max(np.abs(finite)), compute_uv=False
            )
            clear = singular[
                np.abs(np.log2(np.maximum(singular / singular[0], 2.0**-200)) + 20.0)
                > 2.0
            ]
            if clear.size == singular.size:
                assert size == int(np.count_nonzero(singular / singular[0] > 2.0**-20))
        else:
            assert size == 0
    assert spanned > 60
    assert dropped >= 4
