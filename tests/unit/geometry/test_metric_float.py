"""The recorded metric measurements against independent references.

``native/meshcore/tests/metric_float_vectors.inc`` holds the inputs and the
output bits that the native test requires on every platform. These tests hold
the recorded outputs to the definitions of the measurements, evaluated here
with LAPACK, ``decimal`` and exact rationals. They read the recorded file
only, so they are independent of the kernels.
"""

from __future__ import annotations

import math
import re
import struct
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest


_VECTORS = (
    Path(__file__).resolve().parents[3]
    / "native"
    / "meshcore"
    / "tests"
    / "metric_float_vectors.inc"
)
_NAN_BITS = 0x7FF8000000000000
_UPPER = float.fromhex("0x1.6a09e667f3bcdp+0")
_LOWER = float.fromhex("0x1.6a09e667f3bccp-1")
_UNIT_ROUNDOFF = Fraction(1, 2**53)


def _binary64(bits: int, /) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _rows(kind: str, /) -> list[list[int]]:
    rows = []
    for line in _VECTORS.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(rf"PHX_MF_{kind}\((.*)\)", line)
        if match is not None:
            rows.append([int(field, 16) for field in match[1].split(", ")])
    return rows


def _values(fields: list[int], /) -> np.ndarray:
    return np.asarray([_binary64(bits) for bits in fields], dtype=np.float64)


def _bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Metric bank, whether each row is a metric, and the recorded logarithms."""
    rows = _rows("LOG")
    metrics = np.stack([_values(row[:9]).reshape((3, 3)) for row in rows])
    valid = np.asarray([row[9] for row in rows], dtype=np.bool_)
    logarithms = np.stack([_values(row[10:]).reshape((3, 3)) for row in rows])
    return metrics, valid, logarithms


def _is_metric(tensor: np.ndarray, /) -> bool:
    return bool(np.all(np.isfinite(tensor))) and bool(np.linalg.eigvalsh(tensor)[0] > 0.0)


def _spectral(tensor: np.ndarray, function: np.ufunc, /) -> np.ndarray:
    values, vectors = np.linalg.eigh(tensor)
    return (vectors * function(values)) @ vectors.T


def _reference_length(
    p: np.ndarray, q: np.ndarray, metric_p: np.ndarray, metric_q: np.ndarray, /
) -> float:
    delta = q - p
    # Forms beyond the binary64 range are infinite, or NaN where they cancel.
    with np.errstate(all="ignore"):
        first_form = float(delta @ metric_p @ delta)
        second_form = float(delta @ metric_q @ delta)
    if not (first_form >= 0.0 and second_form >= 0.0):
        return math.nan
    first = math.sqrt(first_form)
    second = math.sqrt(second_form)
    if first == 0.0 and second == 0.0:
        return 0.0
    if abs(second / first - 1.0) <= 1.0e-6:
        return 0.5 * (first + second)
    return (first - second) / math.log(first / second)


def test_recorded_logarithms_match_the_spectral_definition() -> None:
    metrics, valid, logarithms = _bank()
    for tensor, accepted, logarithm in zip(metrics, valid, logarithms, strict=True):
        assert accepted == _is_metric(tensor)
        if not accepted:
            assert not logarithm.any()
            continue
        assert np.array_equal(logarithm, logarithm.T)
        reference = _spectral(tensor, np.log)
        assert np.allclose(logarithm, reference, rtol=1.0e-12, atol=1.0e-12)
    assert np.count_nonzero(valid) > 30
    assert np.count_nonzero(~valid) == 4


def test_recorded_interpolations_are_log_euclidean_means() -> None:
    metrics, valid, _ = _bank()
    means = 0
    constants = 0
    refused = 0
    for row in _rows("INTERPOLATE"):
        count = row[0]
        support = row[1 : 1 + count]
        weights = _values(row[5 : 5 + count])
        accepted = row[9]
        result = _values(row[10:]).reshape((3, 3))
        samples = metrics[support]
        if np.all(samples == samples[0]) and valid[support[0]]:
            # A constant field of a metric is returned unchanged.
            assert accepted == 1
            assert result.tobytes() == samples[0].tobytes()
            constants += 1
        elif not np.all(valid[support]):
            assert accepted == 0
            assert not result.any()
            refused += 1
        else:
            assert accepted == 1
            assert np.array_equal(result, result.T)
            mean = sum(
                weight * _spectral(sample, np.log)
                for weight, sample in zip(weights, samples, strict=True)
            )
            reference = _spectral(np.asarray(mean, dtype=np.float64), np.exp)
            assert np.allclose(result, reference, rtol=1.0e-11, atol=0.0)
            means += 1
    assert means > 25
    assert constants == 3
    assert refused == 7


def test_recorded_lengths_are_logarithmic_means_of_the_end_lengths() -> None:
    metrics, _, _ = _bank()
    finite = 0
    undefined = 0
    for row in _rows("LENGTH"):
        p = _values(row[:3])
        q = _values(row[3:6])
        reference = _reference_length(p, q, metrics[row[6]], metrics[row[7]])
        if math.isnan(reference):
            assert row[8] == _NAN_BITS
            undefined += 1
        else:
            assert math.isclose(
                _binary64(row[8]), reference, rel_tol=1.0e-10, abs_tol=0.0
            )
            finite += 1
    assert finite > 70
    assert undefined == 5


def test_recorded_band_quanta_count_the_logarithmic_violation() -> None:
    above_rows = 0
    below_rows = 0
    with localcontext() as context:
        context.prec = 60
        for length_bits, above_bits, below_bits in _rows("BAND"):
            length = _binary64(length_bits)
            above = _binary64(above_bits)
            below = _binary64(below_bits)
            if math.isnan(length) or _LOWER <= length <= _UPPER:
                assert above == 0.0 and below == 0.0
                continue
            # The kernel takes the logarithm of the rounded quotient.
            ratio = length / _UPPER if length > _UPPER else math.inf
            if length < _LOWER:
                ratio = _LOWER / length if length > 0.0 else math.inf
            recorded = above if length > _UPPER else below
            assert (below if length > _UPPER else above) == 0.0
            above_rows += length > _UPPER
            below_rows += length < _LOWER
            if math.isinf(ratio):
                assert recorded == math.inf
                continue
            quanta = Decimal(ratio).ln() * 1024
            nearest = quanta.to_integral_value()
            if abs(quanta - nearest) < Decimal("1e-13"):
                # Within the error of the kernel logarithm of an integer.
                assert recorded in (max(float(nearest), 1.0), float(nearest) + 1.0)
            else:
                assert recorded == max(float(math.ceil(quanta)), 1.0)
    assert above_rows > 20
    assert below_rows > 5


def test_recorded_determinants_are_within_their_stated_error() -> None:
    largest_binary64 = Fraction(sys.float_info.max)
    positive = 0
    overflowed = 0
    for row in _rows("RELATIVE"):
        corners = [_binary64(bits) for bits in row[:12]]
        determinant = _binary64(row[12])
        relative = _binary64(row[13])
        if not all(math.isfinite(value) for value in corners):
            assert row[12] == _NAN_BITS
            assert relative == 0.0
            continue
        points = [
            [Fraction(value) for value in corners[3 * k : 3 * k + 3]] for k in range(4)
        ]
        (a, b, c), (d, e, f), (g, h, i) = (
            [points[k][axis] - points[0][axis] for axis in range(3)] for k in (1, 2, 3)
        )
        if not math.isfinite(determinant):
            # Only an operand beyond the binary64 range gives such a result:
            # a product of two or three entries of the expansion reaches it.
            pairs = (e * i, f * h, d * i, f * g, d * h, e * g)
            triples = (a * e * i, a * f * h, b * d * i, b * f * g, c * d * h, c * e * g)
            assert max(abs(value) for value in pairs + triples) > largest_binary64 / 4
            assert math.isinf(determinant) or row[12] == _NAN_BITS
            assert relative == 0.0
            overflowed += 1
            continue
        exact = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        products = [
            math.prod(
                math.dist(corners[3 * k : 3 * k + 3], corners[3 * j : 3 * j + 3])
                for j in range(4)
                if j != k
            )
            for k in range(4)
        ]
        largest = max(products)
        assert abs(Fraction(determinant) - exact) <= 50 * _UNIT_ROUNDOFF * Fraction(
            largest
        )
        if largest == 0.0:
            assert relative == 0.0
        else:
            assert math.isclose(relative, abs(float(exact)) / largest, rel_tol=1.0e-12)
        positive += relative > 0.0
    assert positive > 15
    assert overflowed == 2


def test_recorded_qualities_are_metric_mean_ratios() -> None:
    metrics, valid, _ = _bank()
    positive = 0
    zero = 0
    for row in _rows("QUALITY"):
        corners = _values(row[:12]).reshape((4, 3))
        sources = row[12:16]
        quality = _binary64(row[16])
        tensors = metrics[sources]
        with np.errstate(all="ignore"):
            volume = abs(float(np.linalg.det(corners[1:] - corners[0]))) / 6.0
            log_determinant = float(np.mean(np.log(np.linalg.det(tensors))))
            metric_volume = volume * math.exp(0.5 * log_determinant)
        squares = sum(
            _reference_length(corners[i], corners[j], tensors[i], tensors[j]) ** 2
            for i, j in combinations(range(4), 2)
        )
        if not (np.all(valid[sources]) and metric_volume > 0.0 and squares > 0.0):
            assert quality == 0.0
            zero += 1
            continue
        reference = 12.0 * (3.0 * metric_volume) ** (2.0 / 3.0) / squares
        assert math.isclose(quality, reference, rel_tol=1.0e-10)
        assert quality > 0.0
        positive += 1
    assert positive > 15
    assert zero == 5


@pytest.mark.parametrize(
    ("kind", "fields"),
    [
        ("LOG", 19),
        ("INTERPOLATE", 19),
        ("LENGTH", 9),
        ("BAND", 3),
        ("RELATIVE", 14),
        ("QUALITY", 17),
    ],
)
def test_recorded_rows_have_their_declared_fields(kind: str, fields: int) -> None:
    rows = _rows(kind)
    assert rows
    assert all(len(row) == fields for row in rows)
