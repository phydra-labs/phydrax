#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Batched native Bernstein conversion of exact polynomial control nets."""

from __future__ import annotations

import ctypes
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import cast

import numpy as np

from .._exact_kernels import (
    _control_count,
    _DOMAIN_CODES,
    _limbs_for,
    _RANGE_ERROR,
    _workers,
)
from .._meshcore import _raise_for_call, load_meshcore, meshcore_available
from ..discretization._coordinate_enclosure import BernsteinDomain, Expression


@dataclass(frozen=True, slots=True)
class _BatchIntegerNet:
    """One net's range in an immutable native batch payload."""

    payload: bytes
    start: int
    end: int
    word_bytes: int
    count: int
    scale: int
    least: int
    greatest: int

    def numerator(self, index: int, /) -> int:
        if not 0 <= index < self.count:
            raise IndexError("control index out of range.")
        start = self.start + index * self.word_bytes
        return int.from_bytes(
            memoryview(self.payload)[start : start + self.word_bytes],
            "little",
            signed=True,
        )

    def numerators(self) -> tuple[int, ...]:
        words = memoryview(self.payload)[self.start : self.end]
        width = self.word_bytes
        return tuple(
            int.from_bytes(words[index : index + width], "little", signed=True)
            for index in range(0, len(words), width)
        )

    def control(self, index: int, /) -> Fraction:
        return Fraction(self.numerator(index), self.scale)

    def controls(self) -> tuple[Fraction, ...]:
        return tuple(Fraction(value, self.scale) for value in self.numerators())


def _pack_words(values: Sequence[int], limbs: int, /) -> bytes:
    """Pack signed native words in little-endian order without per-value encoding."""
    if not values:
        return b""
    if all(-(1 << 63) <= value < (1 << 63) for value in values):
        signed = np.asarray(values, dtype=np.int64)
        words = np.empty((len(values), limbs), dtype="<u8")
        words[:, 0] = signed.view(np.uint64)
        words[:, 1:] = np.where(
            signed[:, None] < 0, np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64(0)
        )
        return words.tobytes()

    integers = np.asarray(values, dtype=np.object_)
    words = np.empty((len(values), limbs), dtype="<u8")
    mask = (1 << 64) - 1
    for limb in range(limbs):
        words[:, limb] = (integers >> (64 * limb)) & mask
    return words.tobytes()


def integer_net_batch(
    requests: Sequence[
        tuple[tuple[Expression, ...], BernsteinDomain, int, tuple[int, ...]]
    ],
    /,
    *,
    parallelism: int | None = None,
) -> list[tuple[_BatchIntegerNet | None, ...]]:
    """Convert eligible columns using common input scales, retaining item refusals."""
    results: list[list[_BatchIntegerNet | None]] = [
        [None] * len(polynomials) for polynomials, _, _, _ in requests
    ]
    if not meshcore_available():
        return [tuple(row) for row in results]
    banks: dict[
        tuple[BernsteinDomain, int, int],
        list[
            tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...], int, int]
        ],
    ] = {}
    for row, (polynomials, domain, dimension, degree) in enumerate(requests):
        rational = cast(
            list[tuple[int, dict[tuple[int, ...], Fraction | int]]],
            [
                (column, polynomial)
                for column, polynomial in enumerate(polynomials)
                if isinstance(polynomial, dict)
                and all(
                    isinstance(value, (Fraction, int)) and not isinstance(value, bool)
                    for value in polynomial.values()
                )
            ],
        )
        scale = lcm(
            *(
                value.denominator
                for _, polynomial in rational
                for value in polynomial.values()
            )
        )
        for column, rational_polynomial in rational:
            if any(len(exponent) != dimension for exponent in rational_polynomial):
                raise ValueError(
                    "Bernstein dimension must match polynomial monomial dimensions."
                )
            numerators = tuple(
                value.numerator * (scale // value.denominator)
                for value in rational_polynomial.values()
            )
            limbs = _limbs_for(
                max((value.bit_length() for value in numerators), default=0)
                + 64
                + len(numerators).bit_length()
            )
            if limbs is None:
                continue
            exponents = tuple(power for exponent in rational_polynomial for power in exponent)
            count = _control_count(domain, dimension, degree)
            banks.setdefault((domain, dimension, limbs), []).append(
                (row, column, degree, exponents, numerators, count, scale)
            )
    workers = _workers(parallelism)
    for (domain, dimension, limbs), items in banks.items():
        term_offsets = [0]
        control_offsets = [0]
        for _, _, _, _, numerators, count, _ in items:
            term_offsets.append(term_offsets[-1] + len(numerators))
            control_offsets.append(control_offsets[-1] + count)
        width = 8 * limbs
        controls = ctypes.create_string_buffer(control_offsets[-1] * width)
        denominators = ctypes.create_string_buffer(8 * len(items))
        extremes = ctypes.create_string_buffer(16 * len(items))
        status = ctypes.create_string_buffer(4 * len(items))
        call = load_meshcore()["phx_mc_bernstein_controls"](
            len(items),
            _DOMAIN_CODES[domain],
            dimension,
            array("i", (value for item in items for value in item[2])).tobytes(),
            array("q", term_offsets).tobytes(),
            array("i", (value for item in items for value in item[3])).tobytes(),
            _pack_words([value for item in items for value in item[4]], limbs),
            array("q", control_offsets).tobytes(),
            controls,
            denominators,
            extremes,
            status,
            limbs,
            workers,
        )
        _raise_for_call(call, "bernstein_controls")
        payload = controls.raw
        den = array("q", denominators.raw)
        ext = array("q", extremes.raw)
        statuses = array("i", status.raw)
        for position, (row, column, _, _, _, count, scale) in enumerate(items):
            if statuses[position] == _RANGE_ERROR:
                continue
            _raise_for_call(statuses[position], "bernstein_controls")
            results[row][column] = _BatchIntegerNet(
                payload,
                control_offsets[position] * width,
                control_offsets[position + 1] * width,
                width,
                count,
                scale * den[position],
                ext[2 * position],
                ext[2 * position + 1],
            )
    return [tuple(row) for row in results]
