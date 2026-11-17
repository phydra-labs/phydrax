# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import hashlib
import random
from fractions import Fraction

import pytest

from phydrax import _exact_polynomial_kernels as kernels
from phydrax._exact_polynomial_kernels import control_digest, symmetric_jacobian_verdicts
from phydrax._meshcore import meshcore_available
from phydrax.discretization import _coordinate_enclosure as e
from phydrax.discretization._exact_algebraic_geometry import RadicalElement


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(not meshcore_available(), reason="meshcore unavailable"),
]


def _reference(
    matrix: tuple[tuple[e.Polynomial, ...], ...], domain: e.BernsteinDomain
) -> tuple[tuple[bool, str], ...]:
    d = len(matrix)
    s = tuple(
        tuple(
            e.scale(e.add(matrix[i][j], matrix[j][i]), Fraction(1, 2)) for j in range(d)
        )
        for i in range(d)
    )
    minors = [s[0][0]]
    if d >= 2:
        minors.append(
            e.subtract(e.multiply(s[0][0], s[1][1]), e.multiply(s[0][1], s[1][0]))
        )
    if d == 3:
        a = e.subtract(e.multiply(s[1][1], s[2][2]), e.multiply(s[1][2], s[2][1]))
        b = e.subtract(e.multiply(s[1][0], s[2][2]), e.multiply(s[1][2], s[2][0]))
        c = e.subtract(e.multiply(s[1][0], s[2][1]), e.multiply(s[1][1], s[2][0]))
        minors.append(
            e.add(
                e.subtract(e.multiply(s[0][0], a), e.multiply(s[0][1], b)),
                e.multiply(s[0][2], c),
            )
        )
    decisions = []
    for minor in minors:
        if not isinstance(minor, dict):
            raise ValueError("Expected polynomial minor")
        degree = e.bernstein_degree(minor, domain, d)
        controls = e._fraction_controls(minor, domain, d, degree)
        decisions.append((all(value > 0 for value in controls), control_digest(controls)))
    return tuple(decisions)


@pytest.mark.parametrize(
    ("domain", "dimension"),
    [
        ("simplex", 1),
        ("simplex", 2),
        ("simplex", 3),
        ("box", 2),
        ("box", 3),
        ("prism", 3),
    ],
)
def test_batched_minors_match_exact_reference(
    domain: e.BernsteinDomain, dimension: int
) -> None:
    rng = random.Random(8291)
    matrices: list[tuple[tuple[e.Polynomial, ...], ...]] = []
    for _ in range(20):
        matrix: list[tuple[e.Polynomial, ...]] = []
        for i in range(dimension):
            line: list[e.Polynomial] = []
            for j in range(dimension):
                zero = (0,) * dimension
                axis = tuple(
                    1 if k == (i + j) % dimension else 0 for k in range(dimension)
                )
                line.append(
                    {
                        zero: Fraction(rng.randrange(-3, 7), 3),
                        axis: Fraction(rng.randrange(-2, 3), 5),
                    }
                )
            matrix.append(tuple(line))
        matrices.append(tuple(matrix))
    expected = [_reference(matrix, domain) for matrix in matrices]
    assert (
        symmetric_jacobian_verdicts(matrices, domain, dimension, parallelism=1)
        == expected
    )
    assert (
        symmetric_jacobian_verdicts(matrices, domain, dimension, parallelism=4)
        == expected
    )
    assert [
        symmetric_jacobian_verdicts([matrix], domain, dimension)[0] for matrix in matrices
    ] == expected


def test_digest_radical_and_rational_identity() -> None:
    assert control_digest([Fraction(2, 3), Fraction(0)]) == control_digest(
        [RadicalElement(Fraction(2, 3)), RadicalElement()]
    )
    assert control_digest([RadicalElement.sqrt(2)]) != control_digest([Fraction(2)])
    assert control_digest([Fraction(1), Fraction(2)]) != control_digest(
        [Fraction(2), Fraction(1)]
    )


def test_control_identity_uses_primitive_integer_net() -> None:
    payload = (
        b"phydrax-exact-controls\0R"
        b"\0\0\0\0\0\0\0\3"
        b"\0\0\0\0\0\0\0\1\6"
        b"\0\0\0\0\0\0\0\0\1\4"
        b"\1\0\0\0\0\0\0\0\1\1"
        b"\0\0\0\0\0\0\0\0\0"
    )
    assert control_digest([Fraction(2, 3), Fraction(-1, 6), Fraction(0)]) == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )


def test_nonpositive_and_typed_refusal() -> None:
    zero = (0,)
    matrices: list[tuple[tuple[e.Polynomial, ...], ...]] = [
        (({zero: Fraction(0)},),),
        (({zero: Fraction(-1)},),),
        (({zero: RadicalElement.sqrt(2)},),),
        (({zero: Fraction(1 << 4100)},),),
    ]
    result = symmetric_jacobian_verdicts(
        matrices,
        "box",
        1,
    )
    assert result == [
        ((False, control_digest([Fraction(0)])),),
        ((False, control_digest([Fraction(-1)])),),
        None,
        None,
    ]


def test_positive_nondiagonal_matrix() -> None:
    z = (0, 0)
    matrix: tuple[tuple[e.Polynomial, ...], ...] = (
        ({z: Fraction(2)}, {z: Fraction(1)}),
        ({z: Fraction(1)}, {z: Fraction(3)}),
    )
    assert symmetric_jacobian_verdicts([matrix], "box", 2) == [
        ((True, control_digest([Fraction(2)])), (True, control_digest([Fraction(5)])))
    ]


def test_minor_verdicts_are_independent() -> None:
    z = (0, 0)
    matrix: tuple[tuple[e.Polynomial, ...], ...] = (
        ({z: Fraction(1)}, {z: Fraction(2)}),
        ({z: Fraction(2)}, {z: Fraction(3)}),
    )
    assert symmetric_jacobian_verdicts([matrix], "box", 2) == [
        ((True, control_digest([Fraction(1)])), (False, control_digest([Fraction(-1)])))
    ]


def test_large_denominator_and_gcd_reduction() -> None:
    q = (1 << 254) + 9
    z, x = (0, 0, 0), (1, 0, 0)
    matrix: tuple[tuple[e.Polynomial, ...], ...] = (
        ({z: Fraction(3, q), x: Fraction(2, q)}, {}, {}),
        ({}, {z: Fraction(5, q)}, {}),
        ({}, {}, {z: Fraction(7, q)}),
    )
    expected = _reference(matrix, "box")
    assert symmetric_jacobian_verdicts([matrix], "box", 3) == [expected]
    assert all(positive for positive, _ in expected)


def test_large_rational_nets_match_exact_reference() -> None:
    rng = random.Random(91831)
    matrices: list[tuple[tuple[e.Polynomial, ...], ...]] = []
    for _ in range(32):
        denominator = rng.getrandbits(180) | 1
        polynomial: e.Polynomial = {
            (0,): Fraction(rng.getrandbits(150) - (1 << 149), denominator),
            (1,): Fraction(rng.getrandbits(150) - (1 << 149), denominator),
            (2,): Fraction(rng.getrandbits(150) - (1 << 149), denominator),
        }
        matrices.append(((polynomial,),))
    assert symmetric_jacobian_verdicts(matrices, "box", 1) == [
        _reference(matrix, "box") for matrix in matrices
    ]


def test_oracle_rejects_omitted_minor_term_and_control_reordering() -> None:
    zero, x = (0, 0), (1, 0)
    matrix: tuple[tuple[e.Polynomial, ...], ...] = (
        ({zero: Fraction(2), x: Fraction(1)}, {zero: Fraction(1)}),
        ({zero: Fraction(1)}, {zero: Fraction(3)}),
    )
    expected = _reference(matrix, "box")
    assert expected == (
        (True, control_digest([Fraction(2), Fraction(3)])),
        (True, control_digest([Fraction(5), Fraction(8)])),
    )
    assert symmetric_jacobian_verdicts([matrix], "box", 2) == [expected]
    # Deliberately faulty alternative: omit the off-diagonal determinant term.
    wrong_minor = e.multiply(matrix[0][0], matrix[1][1])
    if not isinstance(wrong_minor, dict):
        raise ValueError("Expected polynomial minor")
    wrong_controls = e._fraction_controls(
        wrong_minor, "box", 2, e.bernstein_degree(wrong_minor, "box", 2)
    )
    correct_minor = e.subtract(wrong_minor, e.multiply(matrix[0][1], matrix[1][0]))
    if not isinstance(correct_minor, dict):
        raise ValueError("Expected polynomial minor")
    correct_controls = e._fraction_controls(
        correct_minor, "box", 2, e.bernstein_degree(correct_minor, "box", 2)
    )
    first = control_digest([Fraction(2), Fraction(3)])
    assert ((True, first), (True, control_digest(wrong_controls))) != expected
    assert (
        (True, first),
        (True, control_digest(tuple(reversed(correct_controls)))),
    ) != expected


def test_missing_native_library_is_typed_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "meshcore_available", lambda: False)
    matrix: tuple[tuple[e.Polynomial, ...], ...] = (({(0,): Fraction(1)},),)
    assert symmetric_jacobian_verdicts([matrix], "box", 1) == [None]
