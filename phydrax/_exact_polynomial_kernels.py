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

import numpy as np
from numpy.typing import NDArray

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
    from .discretization._exact_coordinate_bank import ExactCoordinateBank


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
    prepared: list[
        tuple[int, list[Mapping[tuple[int, ...], Fraction | int]], int, list[int], int]
    ] = []
    for row, matrix in enumerate(jacobians):
        if len(matrix) != dimension or any(len(line) != dimension for line in matrix):
            raise ValueError("Jacobian must be square with the reference dimension.")
        entries = [entry for line in matrix for entry in line]
        pairs = []
        rational = True
        for entry in entries:
            for value in entry.values():
                if not isinstance(value, (Fraction, int)) or isinstance(value, bool):
                    rational = False
                    break
                pairs.append((value.numerator, value.denominator))
            if not rational:
                break
        if not rational:
            continue
        rational_entries = cast(list[Mapping[tuple[int, ...], Fraction | int]], entries)
        oversized = False
        for entry in entries:
            for index in entry:
                if len(index) != dimension:
                    raise ValueError("Invalid polynomial exponents.")
                for exponent in index:
                    if not isinstance(exponent, int) or exponent < 0:
                        raise ValueError("Invalid polynomial exponents.")
                    oversized |= exponent > 715827882
        if oversized:
            continue
        scale = lcm(*(denominator for _, denominator in pairs))
        if scale.bit_length() > 64 * _WORD_LIMBS[-1] - 1:
            continue
        numerators = [
            numerator * (scale // denominator) for numerator, denominator in pairs
        ]
        bits = max((abs(value).bit_length() for value in numerators), default=0)
        prepared.append((row, rational_entries, scale, numerators, bits))
    if not prepared or not meshcore_available():
        return results
    pending = prepared
    for limbs in _WORD_LIMBS:
        eligible = []
        deferred = []
        for item in pending:
            _, _, scale, _, bits = item
            if scale.bit_length() <= 64 * limbs - 1 and bits <= 64 * limbs - 1:
                eligible.append(item)
            else:
                deferred.append(item)
        if not eligible:
            continue
        offsets = [0]
        exponents: list[int] = []
        coefficients: list[int] = []
        for _, entries, _, numerators, _ in eligible:
            coefficients.extend(numerators)
            end = len(coefficients) - len(numerators)
            for entry in entries:
                for index in entry:
                    exponents.extend(index)
                end += len(entry)
                offsets.append(end)
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


def exact_basis_jacobian_integrals(
    basis: tuple[_Polynomial, ...],
    bank: ExactCoordinateBank,
    routes: NDArray[np.int64],
    domain: BernsteinDomain,
    dimension: int,
    /,
    *,
    parallelism: int | None = None,
) -> list[Fraction | None]:
    """Integrate routed nodal maps exactly; unsupported items are refused."""
    if (
        dimension not in (1, 2, 3)
        or domain not in _DOMAIN_CODES
        or (domain == "prism" and dimension != 3)
    ):
        raise ValueError("Unsupported Bernstein domain or dimension.")
    workers = _workers(parallelism)
    if routes.ndim != 2 or routes.dtype != np.dtype(np.int64):
        raise ValueError("routes must be a rank-2 int64 array.")
    if routes.shape[1] != len(basis) or bank.shape[1] != dimension:
        raise ValueError("Basis, routes, and coordinate bank dimensions must agree.")
    result: list[Fraction | None] = [None] * routes.shape[0]
    if any(
        not isinstance(value, (Fraction, int)) or isinstance(value, bool)
        for polynomial in basis
        for value in polynomial.values()
    ):
        return result
    if any(
        len(index) != dimension
        or any(not isinstance(k, int) or k < 0 or k > 715827882 for k in index)
        for polynomial in basis
        for index in polynomial
    ):
        return result
    rational = cast(tuple[Mapping[tuple[int, ...], Fraction | int], ...], basis)
    basis_scale = lcm(
        *(value.denominator for poly in rational for value in poly.values())
    )
    offsets = [0]
    powers: list[int] = []
    coefficients: list[int] = []
    for poly in rational:
        for index, value in poly.items():
            powers.extend(index)
            coefficients.append(value.numerator * (basis_scale // value.denominator))
        offsets.append(len(coefficients))
    if not result or not meshcore_available():
        return result
    sig, exp, sparse_index, sparse_rows, sparse_values = bank.canonical_parts()
    route_array = np.ascontiguousarray(routes)
    volume = (
        factorial(dimension) if domain == "simplex" else 2 if domain == "prism" else 1
    )
    pending = np.arange(len(result), dtype=np.int64)
    for limbs in _WORD_LIMBS:
        bits = 64 * limbs - 1
        if any(abs(value).bit_length() > bits for value in coefficients):
            continue
        numerators: list[int] = []
        denominators: list[int] = []
        for value in sparse_values:
            if (
                isinstance(value, Fraction)
                and abs(value.numerator).bit_length() <= bits
                and value.denominator.bit_length() <= bits
            ):
                numerators.append(value.numerator)
                denominators.append(value.denominator)
            else:
                numerators.append(0)
                denominators.append(0)
        selected = np.ascontiguousarray(route_array[pending])
        count = pending.size
        sums = ctypes.create_string_buffer(count * 8 * limbs)
        scales = ctypes.create_string_buffer(count * 8 * limbs)
        control_denominators = ctypes.create_string_buffer(count * 8)
        counts = ctypes.create_string_buffer(count * 8)
        statuses = ctypes.create_string_buffer(count * 4)
        call = load_meshcore()["phx_mc_basis_jacobian_integrals"](
            count,
            dimension,
            _DOMAIN_CODES[domain],
            len(basis),
            len(bank),
            selected.ctypes.data,
            sig.ctypes.data,
            exp.ctypes.data,
            sparse_index.ctypes.data,
            sparse_rows.ctypes.data,
            _words(numerators, limbs) or b"\0",
            _words(denominators, limbs) or b"\0",
            array("q", offsets).tobytes(),
            array("i", powers).tobytes() or b"\0",
            _words(coefficients, limbs) or b"\0",
            sums,
            scales,
            control_denominators,
            counts,
            statuses,
            limbs,
            workers,
        )
        _raise_for_call(call, "basis_jacobian_integrals")
        status_values = array("i", statuses.raw)
        control_values = array("q", control_denominators.raw)
        count_values = array("q", counts.raw)
        sum_words, scale_words = sums.raw, scales.raw
        retry: list[int] = []
        for position, row in enumerate(pending):
            if status_values[position] == _RANGE_ERROR:
                retry.append(int(row))
                continue
            _raise_for_call(status_values[position], "basis_jacobian_integrals")
            scale = _word(scale_words, position, 8 * limbs)
            result[int(row)] = Fraction(
                _word(sum_words, position, 8 * limbs),
                (basis_scale * scale) ** dimension
                * control_values[position]
                * count_values[position]
                * volume,
            )
        if not retry:
            break
        pending = np.asarray(retry, dtype=np.int64)
    return result
