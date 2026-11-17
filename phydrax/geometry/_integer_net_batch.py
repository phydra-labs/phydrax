#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Batched native Bernstein conversion of exact polynomial control nets."""

from __future__ import annotations

import ctypes
from array import array
from collections.abc import Sequence
from fractions import Fraction
from math import lcm
from typing import cast

from .._exact_kernels import (
    _control_count,
    _DOMAIN_CODES,
    _limbs_for,
    _RANGE_ERROR,
    _words,
    _workers,
    IntegerBernsteinNet,
)
from .._meshcore import _raise_for_call, load_meshcore, meshcore_available
from ..discretization._coordinate_enclosure import BernsteinDomain, Expression


def integer_net_batch(
    requests: Sequence[
        tuple[tuple[Expression, ...], BernsteinDomain, int, tuple[int, ...]]
    ],
    /,
    *,
    parallelism: int | None = None,
) -> list[tuple[IntegerBernsteinNet | None, ...]]:
    """Convert eligible columns using common input scales, retaining item refusals."""
    results: list[list[IntegerBernsteinNet | None]] = [
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
            list[dict[tuple[int, ...], Fraction | int]],
            [
                polynomial
                for polynomial in polynomials
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
                for polynomial in rational
                for value in polynomial.values()
            )
        )
        for column, polynomial in enumerate(polynomials):
            if not isinstance(polynomial, dict) or not all(
                isinstance(value, (Fraction, int)) and not isinstance(value, bool)
                for value in polynomial.values()
            ):
                continue
            rational_polynomial = cast(dict[tuple[int, ...], Fraction | int], polynomial)
            if any(len(exponent) != dimension for exponent in polynomial):
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
            exponents = tuple(power for exponent in polynomial for power in exponent)
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
            _words([value for item in items for value in item[4]], limbs),
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
            results[row][column] = IntegerBernsteinNet(
                payload[
                    control_offsets[position] * width : control_offsets[position + 1]
                    * width
                ],
                width,
                count,
                scale * den[position],
                ext[2 * position],
                ext[2 * position + 1],
            )
    return [tuple(row) for row in results]
