# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fused exact symmetric-polynomial proofs and canonical control identities.

Rational nets hash the primitive common-denominator representation: the prefix
``phydrax-exact-controls\\0R``, an eight-byte big-endian control count, the
framed positive denominator, then each framed signed numerator in control
order. Magnitudes are minimal big-endian bytes prefixed by an eight-byte length;
signed integers add a leading sign byte (zero is nonnegative). Nets containing
irrational radicals use the ``\\0A`` prefix, followed by the control count;
each control then has an eight-byte term count and its sorted canonical terms,
each comprising a framed radicand, signed numerator, and positive denominator.
"""

from __future__ import annotations

import ctypes
import hashlib
from array import array
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import factorial, gcd, lcm
from typing import cast, TYPE_CHECKING

from ._exact_kernels import (
    _DOMAIN_CODES,
    _RANGE_ERROR,
    _word,
    _WORD_LIMBS,
    _words,
    _workers,
    BernsteinDomain,
)
from ._meshcore import _raise_for_call, load_meshcore, meshcore_available


if TYPE_CHECKING:
    from .discretization._exact_algebraic_geometry import ExactScalar


def _magnitude(value: int, /) -> bytes:
    magnitude = abs(value)
    data = magnitude.to_bytes((magnitude.bit_length() + 7) // 8, "big")
    return len(data).to_bytes(8, "big") + data


def _signed(value: int, /) -> bytes:
    return bytes((int(value < 0),)) + _magnitude(value)


def control_digest(controls: Sequence[ExactScalar], /) -> str:
    """Digest exact control values in their canonical Bernstein order."""
    from .discretization._exact_algebraic_geometry import RadicalElement

    rational: list[Fraction] = []
    for value in controls:
        if isinstance(value, RadicalElement):
            if any(radicand != 1 for radicand, _ in value.terms):
                break
            rational.append(value.terms[0][1] if value.terms else Fraction(0))
        else:
            rational.append(Fraction(value))
    if len(rational) == len(controls):
        scale = lcm(*(value.denominator for value in rational))
        numerators = tuple(
            value.numerator * (scale // value.denominator) for value in rational
        )
        divisor = gcd(scale, *numerators)
        payload = bytearray(b"phydrax-exact-controls\0R")
        payload.extend(len(controls).to_bytes(8, "big"))
        payload.extend(_magnitude(scale // divisor))
        for value in numerators:
            payload.extend(_signed(value // divisor))
    else:
        payload = bytearray(b"phydrax-exact-controls\0A")
        payload.extend(len(controls).to_bytes(8, "big"))
        for value in controls:
            terms = (
                value.terms
                if isinstance(value, RadicalElement)
                else (((1, Fraction(value)),) if value else ())
            )
            payload.extend(len(terms).to_bytes(8, "big"))
            for radicand, coefficient in terms:
                payload.extend(_magnitude(radicand))
                payload.extend(_signed(coefficient.numerator))
                payload.extend(_magnitude(coefficient.denominator))
    return "sha256:" + hashlib.sha256(payload).hexdigest()


type _Polynomial = Mapping[tuple[int, ...], ExactScalar]
type _Jacobian = Sequence[Sequence[_Polynomial]]


def symmetric_jacobian_verdicts(
    jacobians: Sequence[_Jacobian],
    domain: BernsteinDomain,
    dimension: int,
    /,
    *,
    parallelism: int | None = None,
) -> list[tuple[tuple[bool, str], ...] | None]:
    """Prove strict positivity of each leading symmetric-Jacobian minor.

    Native range refusals, missing native library and radical coefficients return
    None. Every decided item returns one positivity verdict and canonical
    digest per minor, including nonpositive controls.
    """
    if (
        dimension not in (1, 2, 3)
        or domain not in _DOMAIN_CODES
        or (domain == "prism" and dimension != 3)
    ):
        raise ValueError("Unsupported Bernstein domain or dimension.")
    workers = _workers(parallelism)
    results: list[tuple[tuple[bool, str], ...] | None] = [None] * len(jacobians)
    prepared: list[tuple[int, list[Mapping[tuple[int, ...], Fraction | int]], int]] = []
    for row, matrix in enumerate(jacobians):
        if len(matrix) != dimension or any(len(line) != dimension for line in matrix):
            raise ValueError("Jacobian must be square with the reference dimension.")
        entries = [entry for line in matrix for entry in line]
        if any(
            not isinstance(v, (Fraction, int)) or isinstance(v, bool)
            for entry in entries
            for v in entry.values()
        ):
            continue
        rational_entries = cast(list[Mapping[tuple[int, ...], Fraction | int]], entries)
        for entry in entries:
            if any(
                len(index) != dimension
                or any(not isinstance(k, int) or k < 0 for k in index)
                for index in entry
            ):
                raise ValueError("Invalid polynomial exponents.")
        if any(k > 715827882 for entry in entries for index in entry for k in index):
            continue
        scale = lcm(
            *(v.denominator for entry in rational_entries for v in entry.values())
        )
        prepared.append((row, rational_entries, scale))
    if not prepared or not meshcore_available():
        return results
    pending = prepared
    for limbs in _WORD_LIMBS:
        eligible = []
        deferred = []
        for item in pending:
            _, entries, scale = item
            if scale.bit_length() <= 64 * limbs - 1 and all(
                abs(v.numerator * (scale // v.denominator)).bit_length() <= 64 * limbs - 1
                for entry in entries
                for v in entry.values()
            ):
                eligible.append(item)
            else:
                deferred.append(item)
        if not eligible:
            continue
        offsets = [0]
        exponents: list[int] = []
        coefficients: list[int] = []
        for _, entries, scale in eligible:
            for entry in entries:
                for index, value in entry.items():
                    exponents.extend(index)
                    coefficients.append(value.numerator * (scale // value.denominator))
                offsets.append(len(coefficients))
        count = len(eligible)
        digests = ctypes.create_string_buffer(count * dimension * 32)
        verdicts = ctypes.create_string_buffer(count * dimension * 4)
        statuses = ctypes.create_string_buffer(count * 4)
        call = load_meshcore()["phx_mc_symmetric_polynomial_verdicts"](
            count,
            dimension,
            _DOMAIN_CODES[domain],
            array("q", offsets).tobytes(),
            array("i", exponents).tobytes() or b"\0",
            _words(coefficients, limbs) or b"\0",
            _words([item[2] for item in eligible], limbs),
            digests,
            verdicts,
            statuses,
            limbs,
            workers,
        )
        _raise_for_call(call, "symmetric_polynomial_verdicts")
        status_values, verdict_values = array("i", statuses.raw), array("i", verdicts.raw)
        digest_bytes = digests.raw
        retry = []
        for position, item in enumerate(eligible):
            row = item[0]
            if status_values[position] == _RANGE_ERROR:
                retry.append(item)
                continue
            _raise_for_call(status_values[position], "symmetric_polynomial_verdicts")
            base = position * dimension * 32
            results[row] = tuple(
                (
                    verdict_values[position * dimension + m] == 1,
                    "sha256:" + digest_bytes[base + m * 32 : base + (m + 1) * 32].hex(),
                )
                for m in range(dimension)
            )
        pending = deferred + retry
    return results


def exact_jacobian_integrals(
    jacobians: Sequence[_Jacobian],
    domain: BernsteinDomain,
    dimension: int,
    /,
    *,
    parallelism: int | None = None,
) -> list[Fraction | None]:
    """Integrate ordinary Jacobian determinants exactly, or refuse each item.

    Radical coefficients and native range refusals return None for the owning
    exact fallback. The signed result includes the exact reference volume.
    """
    if (
        dimension not in (1, 2, 3)
        or domain not in _DOMAIN_CODES
        or (domain == "prism" and dimension != 3)
    ):
        raise ValueError("Unsupported Bernstein domain or dimension.")
    workers = _workers(parallelism)
    results: list[Fraction | None] = [None] * len(jacobians)
    prepared: list[tuple[int, list[Mapping[tuple[int, ...], Fraction | int]], int]] = []
    for row, matrix in enumerate(jacobians):
        if len(matrix) != dimension or any(len(line) != dimension for line in matrix):
            raise ValueError("Jacobian must be square with the reference dimension.")
        entries = [entry for line in matrix for entry in line]
        if any(
            not isinstance(value, (Fraction, int)) or isinstance(value, bool)
            for entry in entries
            for value in entry.values()
        ):
            continue
        rational = cast(list[Mapping[tuple[int, ...], Fraction | int]], entries)
        if any(
            len(index) != dimension or any(not isinstance(k, int) or k < 0 for k in index)
            for entry in entries
            for index in entry
        ):
            raise ValueError("Invalid polynomial exponents.")
        if any(k > 715827882 for entry in entries for index in entry for k in index):
            continue
        scale = lcm(
            *(value.denominator for entry in rational for value in entry.values())
        )
        prepared.append((row, rational, scale))
    if not prepared or not meshcore_available():
        return results
    pending = prepared
    volume_denominator = (
        factorial(dimension) if domain == "simplex" else 2 if domain == "prism" else 1
    )
    for limbs in _WORD_LIMBS:
        eligible = []
        deferred = []
        for item in pending:
            _, entries, scale = item
            if all(
                abs(value.numerator * (scale // value.denominator)).bit_length()
                <= 64 * limbs - 1
                for entry in entries
                for value in entry.values()
            ):
                eligible.append(item)
            else:
                deferred.append(item)
        if not eligible:
            continue
        offsets = [0]
        exponents: list[int] = []
        coefficients: list[int] = []
        for _, entries, scale in eligible:
            for entry in entries:
                for index, value in entry.items():
                    exponents.extend(index)
                    coefficients.append(value.numerator * (scale // value.denominator))
                offsets.append(len(coefficients))
        count = len(eligible)
        sums = ctypes.create_string_buffer(count * 8 * limbs)
        denominators = ctypes.create_string_buffer(count * 8)
        counts = ctypes.create_string_buffer(count * 8)
        statuses = ctypes.create_string_buffer(count * 4)
        call = load_meshcore()["phx_mc_polynomial_jacobian_integrals"](
            count,
            dimension,
            _DOMAIN_CODES[domain],
            array("q", offsets).tobytes(),
            array("i", exponents).tobytes() or b"\0",
            _words(coefficients, limbs) or b"\0",
            sums,
            denominators,
            counts,
            statuses,
            limbs,
            workers,
        )
        _raise_for_call(call, "polynomial_jacobian_integrals")
        status_values = array("i", statuses.raw)
        denominator_values = array("q", denominators.raw)
        count_values = array("q", counts.raw)
        sum_bytes = sums.raw
        retry = []
        for position, item in enumerate(eligible):
            if status_values[position] == _RANGE_ERROR:
                retry.append(item)
                continue
            _raise_for_call(status_values[position], "polynomial_jacobian_integrals")
            row, _, scale = item
            results[row] = Fraction(
                _word(sum_bytes, position, 8 * limbs),
                scale**dimension
                * denominator_values[position]
                * count_values[position]
                * volume_denominator,
            )
        pending = deferred + retry
    return results
