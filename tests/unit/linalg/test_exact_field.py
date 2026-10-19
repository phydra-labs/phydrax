# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Field-arithmetic elimination preserves rational results and algebraic pivots."""

from fractions import Fraction

import pytest

from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.linalg._exact import determinant_exact, solve_exact


def test_rational_elimination_preserves_fraction_values() -> None:
    matrix = [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]]
    right = [[Fraction(1)], [Fraction(0)]]
    assert determinant_exact(matrix) == Fraction(3)
    assert solve_exact(matrix, right) == [[Fraction(2, 3)], [Fraction(-1, 3)]]
    assert matrix == [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]]
    assert right == [[Fraction(1)], [Fraction(0)]]


def test_radical_elimination_matches_symbolic_inverse() -> None:
    root = RadicalElement.sqrt(2)
    one, zero = RadicalElement(1), RadicalElement()
    matrix = [[root, one], [one, root]]
    # det(A)=sqrt(2)^2-1=1, so adj(A) is its inverse.
    assert determinant_exact(matrix) == 1
    assert solve_exact(matrix, [[one, zero], [zero, one]]) == [
        [root, -one],
        [-one, root],
    ]


def test_algebraic_pivot_uses_exact_zero_not_a_float_threshold() -> None:
    root = RadicalElement.sqrt(2)
    zero = root * root - 2
    one = RadicalElement(1)
    matrix = [[zero, root], [one, zero]]
    assert determinant_exact(matrix) == -root
    assert solve_exact(matrix, [[root], [one]]) == [[one], [one]]
    epsilon = root - Fraction(1414213562373095, 10**15)
    assert epsilon > 0
    assert solve_exact([[epsilon, one], [one, zero]], [[zero], [one]]) == [
        [one],
        [-epsilon],
    ]


def test_singular_radical_system_refuses_exactly() -> None:
    root = RadicalElement.sqrt(2)
    matrix = [[root, RadicalElement(2)], [RadicalElement(1), root]]
    assert determinant_exact(matrix) == 0
    with pytest.raises(ValueError, match="not unisolvent"):
        solve_exact(matrix, [[RadicalElement(1)], [RadicalElement()]])


def test_radical_term_comparisons_match_recursive_quadratic_norms() -> None:
    def reference_sign(value: RadicalElement) -> int:
        if not value:
            return 0
        if len(value.terms) == 1:
            return 1 if value.terms[0][1] > 0 else -1
        a, b, generator = value._split()
        sa, sb = reference_sign(a), reference_sign(b)
        if not sa:
            return sb
        if not sb or sa == sb:
            return sa
        return sa * reference_sign(a * a - generator * b * b)

    roots = tuple(RadicalElement.sqrt(n) for n in (2, 3, 5, 6))
    values = (
        RadicalElement(),
        RadicalElement(1),
        *roots,
        roots[0] - Fraction(1414213562373095, 10**15),
        roots[0] - Fraction(1414213562373096, 10**15),
        roots[0] - roots[1],
        roots[0] + roots[1] - roots[2],
        roots[0] - roots[1] + roots[3] - 2,
    )
    values += tuple(-value for value in values[1:])
    for first in values:
        assert first.sign() == reference_sign(first)
        for second in values:
            sign = reference_sign(first - second)
            assert (first < second) == (sign < 0)
            assert (first <= second) == (sign <= 0)
            assert (first > second) == (sign > 0)
            assert (first >= second) == (sign >= 0)


def test_short_radical_comparisons_do_not_construct_scalar_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Iterable
    from typing import NoReturn

    first, second = RadicalElement.sqrt(2), RadicalElement.sqrt(3)
    positive = first - Fraction(7, 5)
    rational = RadicalElement(Fraction(7, 5))

    def unexpected(
        cls: type[RadicalElement], terms: Iterable[tuple[int, Fraction]], /
    ) -> NoReturn:
        raise AssertionError(
            "RadicalElement.comparison must avoid scalar record construction; got a constructor call"
        )

    monkeypatch.setattr(RadicalElement, "_from_canonical", classmethod(unexpected))
    assert positive.sign() == 1
    assert first < second
    assert first > Fraction(7, 5)
    assert Fraction(7, 5) < first
    assert rational == Fraction(7, 5)
