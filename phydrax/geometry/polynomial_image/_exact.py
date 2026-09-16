#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ...algebraic._exact import ExactSparsePolynomialSystem, QQ
from ._contracts import (
    EvidenceDisposition,
    ExactCompositionRemainder,
    ExactPolynomialContainmentResult,
    PolynomialImageClaimEvidence,
)
from ._map import SparsePolynomialMap


_Polynomial = dict[tuple[int, ...], Fraction]


def _canonical_fraction(value: Fraction, /) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _add_term(
    polynomial: _Polynomial,
    exponent: tuple[int, ...],
    coefficient: Fraction,
    /,
) -> None:
    if coefficient == 0:
        return
    value = polynomial.get(exponent, Fraction(0)) + coefficient
    if value == 0:
        polynomial.pop(exponent, None)
    else:
        polynomial[exponent] = value


def _multiply(left: _Polynomial, right: _Polynomial, /) -> _Polynomial:
    product: _Polynomial = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = tuple(
                first + second
                for first, second in zip(
                    left_exponent,
                    right_exponent,
                    strict=True,
                )
            )
            _add_term(
                product,
                exponent,
                left_coefficient * right_coefficient,
            )
    return product


def _power(polynomial: _Polynomial, exponent: int, dimension: int, /) -> _Polynomial:
    result: _Polynomial = {(0,) * dimension: Fraction(1)}
    factor = polynomial
    remaining = int(exponent)
    while remaining:
        if remaining & 1:
            result = _multiply(result, factor)
        remaining >>= 1
        if remaining:
            factor = _multiply(factor, factor)
    return result


def _equation_polynomials(
    system: ExactSparsePolynomialSystem, /
) -> tuple[_Polynomial, ...]:
    support = system.support
    equation_indices = np.asarray(support.equation_indices, dtype=np.int32)
    exponents = np.asarray(support.exponents, dtype=np.int32)
    equations: list[_Polynomial] = [dict() for _ in range(support.equation_count)]
    for equation, exponent, coefficient in zip(
        equation_indices,
        exponents,
        system.coefficients,
        strict=True,
    ):
        _add_term(
            equations[int(equation)],
            tuple(int(value) for value in exponent),
            Fraction(coefficient),
        )
    return tuple(equations)


def prove_exact_containment(
    polynomial_map: SparsePolynomialMap,
    relations: ExactSparsePolynomialSystem,
    /,
) -> ExactPolynomialContainmentResult:
    """Compose QQ relations with a QQ map and report exact zero remainders.

    A zero composition proves only that the map image is contained in the target
    zero set. It does not prove generation or equality of ideals, equality of the
    real image with that zero set, or any topological statement.
    """

    if not isinstance(polynomial_map, SparsePolynomialMap):
        raise TypeError("polynomial_map must be a SparsePolynomialMap.")
    if not isinstance(relations, ExactSparsePolynomialSystem):
        raise TypeError("relations must be an ExactSparsePolynomialSystem.")
    exact_map = polynomial_map.exact_system
    if exact_map is None:
        raise ValueError("Exact containment requires exact QQ map provenance.")
    if (
        exact_map.domain.domain_id != QQ.domain_id
        or relations.domain.domain_id != QQ.domain_id
    ):
        raise ValueError("Exact polynomial composition currently requires QQ.")
    if relations.support.variable_count != polynomial_map.target_dimension:
        raise ValueError("Relation variables and map target dimension differ.")
    if relations.support.variable_labels != polynomial_map.target_labels:
        raise ValueError("Relation variables must match map target coordinate labels.")

    coordinate_polynomials = _equation_polynomials(exact_map)
    relation_equations = _equation_polynomials(relations)
    source_dimension = polynomial_map.source_dimension
    remainders: list[ExactCompositionRemainder] = []
    for equation_label, relation in zip(
        relations.support.equation_labels,
        relation_equations,
        strict=True,
    ):
        composed: _Polynomial = {}
        for target_exponent, relation_coefficient in relation.items():
            term: _Polynomial = {(0,) * source_dimension: relation_coefficient}
            for coordinate, power in enumerate(target_exponent):
                if power:
                    term = _multiply(
                        term,
                        _power(
                            coordinate_polynomials[coordinate],
                            power,
                            source_dimension,
                        ),
                    )
            for source_exponent, coefficient in term.items():
                _add_term(composed, source_exponent, coefficient)
        terms = tuple(
            (exponent, _canonical_fraction(composed[exponent]))
            for exponent in sorted(composed)
        )
        remainders.append(ExactCompositionRemainder(equation_label, terms))
    contained = all(remainder.is_zero for remainder in remainders)
    claims = PolynomialImageClaimEvidence(
        exact_containment=(
            EvidenceDisposition.SUPPORTED if contained else EvidenceDisposition.REJECTED
        )
    )
    return ExactPolynomialContainmentResult(
        remainders,
        claims,
        map_id=polynomial_map.map_id,
        relation_system_id=relations.system_id,
    )


__all__ = ["prove_exact_containment"]
