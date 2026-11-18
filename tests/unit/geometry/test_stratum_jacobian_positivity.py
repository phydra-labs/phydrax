from __future__ import annotations

from fractions import Fraction
from random import Random

import pytest

from phydrax.geometry._stratum_subdivision import (
    _Controls,
    _integer_controls,
    _jacobian_controls_positive,
    _JacobianControls,
)


def _reference(
    matrix: tuple[tuple[int, ...], ...],
    prefix: tuple[tuple[Fraction, ...], ...],
    suffix: tuple[tuple[Fraction, ...], ...] | None,
) -> bool:
    dimension = len(prefix)
    factors = (prefix, matrix) if suffix is None else (prefix, matrix, suffix)
    result = tuple(tuple(Fraction(value) for value in row) for row in factors[0])
    for factor in factors[1:]:
        result = tuple(
            tuple(
                sum(
                    (result[i][k] * factor[k][j] for k in range(dimension)),
                    Fraction(0),
                )
                for j in range(dimension)
            )
            for i in range(dimension)
        )
    symmetric = tuple(
        tuple(result[i][j] + result[j][i] for j in range(dimension))
        for i in range(dimension)
    )
    a = symmetric[0][0]
    if a <= 0:
        return False
    if dimension == 1:
        return True
    b, d = symmetric[0][1], symmetric[1][1]
    if a * d - b * b <= 0:
        return False
    if dimension == 2:
        return True
    c, e, f = symmetric[0][2], symmetric[1][2], symmetric[2][2]
    return a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d) > 0


def test_integer_sylvester_signs_match_fraction_reference() -> None:
    random = Random(82173)
    for dimension in (1, 2, 3):
        for _ in range(250):
            factors = [
                tuple(
                    tuple(
                        Fraction(random.randrange(-6, 7), random.randrange(1, 8))
                        for _ in range(dimension)
                    )
                    for _ in range(dimension)
                )
                for _ in range(2)
            ]
            matrices = [
                tuple(
                    tuple(random.randrange(-8, 9) for _ in range(dimension))
                    for _ in range(dimension)
                )
                for _ in range(3)
            ]
            values: dict[tuple[int, ...], tuple[int, ...]] = {
                (index,): tuple(value for row in matrix for value in row)
                for index, matrix in enumerate(matrices)
            }
            controls = _JacobianControls(dimension, _integer_controls(values, 7))
            for suffix in (None, factors[1]):
                expected = all(
                    _reference(matrix, factors[0], suffix) for matrix in matrices
                )
                assert (
                    _jacobian_controls_positive(controls, factors[0], suffix) is expected
                )


def test_zero_minor_and_very_large_integers_are_exact() -> None:
    huge = 1 << 4200
    controls = _JacobianControls(
        2, _integer_controls({(0,): (huge, huge, huge, huge)}, 1)
    )
    identity = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    assert not _jacobian_controls_positive(controls, identity)


def test_malformed_control_preserves_short_circuit_order() -> None:
    identity = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    negative = (-1, 0, 0, -1)
    malformed = (1,)
    first_negative = _JacobianControls(2, _Controls({(0,): negative, (1,): malformed}, 1))
    assert not _jacobian_controls_positive(first_negative, identity)
    first_malformed = _JacobianControls(
        2, _Controls({(0,): malformed, (1,): negative}, 1)
    )
    with pytest.raises(
        ValueError, match="Exact symmetric product factors must have equal square shapes"
    ):
        _jacobian_controls_positive(first_malformed, identity)
