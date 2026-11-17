# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fused exact symmetric-polynomial proofs and canonical control identities.

A digest hashes UTF-8 ``phydrax-exact-controls\n`` followed, for each
control in Bernstein order, by its number of nonzero radical terms and a
newline, then ``radicand:numerator/denominator\n`` for each term in ascending
radicand order. Rational values use radicand 1; zero has no terms. Fractions
are reduced with positive denominator. The result is prefixed ``sha256:``.
"""

from __future__ import annotations

import ctypes
import hashlib
from array import array
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import lcm
from typing import cast, TYPE_CHECKING

from ._exact_kernels import (
    _DOMAIN_CODES,
    _RANGE_ERROR,
    _WORD_LIMBS,
    _words,
    _workers,
    BernsteinDomain,
)
from ._meshcore import _raise_for_call, load_meshcore, meshcore_available


if TYPE_CHECKING:
    from .discretization._exact_algebraic_geometry import ExactScalar


def control_digest(controls: Sequence[ExactScalar], /) -> str:
    """Digest an ordered exact control net, including radical-valued controls."""
    from .discretization._exact_algebraic_geometry import RadicalElement

    digest = hashlib.sha256(b"phydrax-exact-controls\n")
    for value in controls:
        if isinstance(value, RadicalElement):
            terms = value.terms
        else:
            rational = Fraction(value)
            terms = ((1, rational),) if rational else ()
        digest.update(f"{len(terms)}\n".encode())
        for radicand, coefficient in terms:
            digest.update(
                f"{radicand}:{coefficient.numerator}/{coefficient.denominator}\n".encode()
            )
    return "sha256:" + digest.hexdigest()


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
            *(
                Fraction(v).denominator
                for entry in rational_entries
                for v in entry.values()
            )
        )
        prepared.append((row, rational_entries, scale))
    if not prepared or not meshcore_available():
        return results
    pending = prepared
    for limbs in _WORD_LIMBS:
        eligible = []
        for item in pending:
            _, entries, scale = item
            if scale.bit_length() <= 64 * limbs - 1 and all(
                abs(
                    Fraction(v).numerator * (scale // Fraction(v).denominator)
                ).bit_length()
                <= 64 * limbs - 1
                for entry in entries
                for v in entry.values()
            ):
                eligible.append(item)
        if not eligible:
            continue
        offsets = [0]
        exponents: list[int] = []
        coefficients: list[int] = []
        for _, entries, scale in eligible:
            for entry in entries:
                for index, value in entry.items():
                    rational = Fraction(value)
                    exponents.extend(index)
                    coefficients.append(
                        rational.numerator * (scale // rational.denominator)
                    )
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
                    "sha256:" + digests.raw[base + m * 32 : base + (m + 1) * 32].hex(),
                )
                for m in range(dimension)
            )
        pending = [item for item in pending if item not in eligible] + retry
    return results
