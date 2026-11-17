# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import ctypes
import itertools
import math
import random
from collections.abc import Mapping
from fractions import Fraction

import pytest

from phydrax import _exact_polynomial_kernels as kernels
from phydrax._exact_kernels import BernsteinDomain
from phydrax._exact_polynomial_kernels import exact_jacobian_integrals
from phydrax._meshcore import meshcore_available
from phydrax.discretization._exact_algebraic_geometry import ExactScalar, RadicalElement


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(not meshcore_available(), reason="meshcore unavailable"),
]

type Polynomial = dict[tuple[int, ...], Fraction]
type Matrix = tuple[tuple[Polynomial, ...], ...]


def _reference(matrix: Matrix, domain: str) -> Fraction:
    dimension = len(matrix)
    result = Fraction(0)
    for permutation in itertools.permutations(range(dimension)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(dimension)
            for j in range(i + 1, dimension)
        )
        product: Polynomial = {(0,) * dimension: Fraction((-1) ** inversions)}
        for row, column in enumerate(permutation):
            next_product: Polynomial = {}
            for left, a in product.items():
                for right, b in matrix[row][column].items():
                    power = tuple(x + y for x, y in zip(left, right, strict=True))
                    next_product[power] = next_product.get(power, Fraction(0)) + a * b
            product = next_product
        for powers, coefficient in product.items():
            if domain == "simplex":
                moment = Fraction(
                    math.prod(math.factorial(k) for k in powers),
                    math.factorial(sum(powers) + dimension),
                )
            elif domain == "box":
                moment = Fraction(1, math.prod(k + 1 for k in powers))
            else:
                moment = Fraction(
                    math.factorial(powers[0]) * math.factorial(powers[1]),
                    math.factorial(powers[0] + powers[1] + 2) * (powers[2] + 1),
                )
            result += coefficient * moment
    return result


@pytest.mark.parametrize(
    ("domain", "dimension"),
    [
        ("simplex", 1),
        ("simplex", 2),
        ("simplex", 3),
        ("box", 1),
        ("box", 2),
        ("box", 3),
        ("prism", 3),
    ],
)
def test_randomized_exact_integrals(domain: BernsteinDomain, dimension: int) -> None:
    rng = random.Random(88491)
    matrices: list[Matrix] = []
    for _ in range(40):
        matrix = tuple(
            tuple(
                {
                    tuple(rng.randrange(3) for _ in range(dimension)): Fraction(
                        rng.randrange(-5, 6), rng.randrange(1, 6)
                    )
                    for _ in range(3)
                }
                for _ in range(dimension)
            )
            for _ in range(dimension)
        )
        matrices.append(matrix)
    expected = [_reference(matrix, domain) for matrix in matrices]
    assert (
        exact_jacobian_integrals(matrices, domain, dimension, parallelism=1) == expected
    )
    assert (
        exact_jacobian_integrals(matrices, domain, dimension, parallelism=4) == expected
    )
    assert [
        exact_jacobian_integrals(matrices[i : i + 7], domain, dimension)
        for i in range(0, len(matrices), 7)
    ] == [expected[i : i + 7] for i in range(0, len(expected), 7)]


def test_sign_zero_retry_and_radical_refusal() -> None:
    zero = (0, 0)

    def matrix(a: Fraction, b: Fraction) -> Matrix:
        return (({zero: a}, {}), ({}, {zero: b}))

    matrices = [
        matrix(Fraction(-2), Fraction(3)),
        matrix(Fraction(0), Fraction(7)),
        matrix(Fraction(2**200), Fraction(2**200)),
        matrix(Fraction(2**5000), Fraction(1)),
    ]
    assert exact_jacobian_integrals(matrices, "simplex", 2) == [
        Fraction(-3),
        Fraction(0),
        Fraction(2**399),
        None,
    ]
    radical: tuple[tuple[Mapping[tuple[int, ...], ExactScalar], ...], ...] = (
        ({zero: RadicalElement.sqrt(Fraction(2))}, {}),
        ({}, {zero: Fraction(1)}),
    )
    assert exact_jacobian_integrals([radical], "simplex", 2) == [None]


def test_reference_volume_and_nonsymmetric_determinant() -> None:
    one = (0, 0, 0)
    matrix: Matrix = (
        ({one: Fraction(1)}, {one: Fraction(2)}, {}),
        ({one: Fraction(3)}, {one: Fraction(4)}, {}),
        ({}, {}, {one: Fraction(1)}),
    )
    assert exact_jacobian_integrals([matrix], "simplex", 3) == [Fraction(-1, 3)]
    assert exact_jacobian_integrals([matrix], "prism", 3) == [Fraction(-1)]
    assert exact_jacobian_integrals([matrix], "box", 3) == [Fraction(-2)]


def test_parity_reference_detects_wrong_native_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero = (0, 0)
    matrix: Matrix = (({zero: Fraction(2)}, {}), ({}, {zero: Fraction(3)}))
    expected = _reference(matrix, "simplex")
    native = kernels.load_meshcore()["phx_mc_polynomial_jacobian_integrals"]

    def corrupted(*args: object) -> int:
        result = native(*args)
        sums = args[6]
        if not isinstance(sums, ctypes.Array):
            raise TypeError("Expected native output buffer")
        ctypes.memset(ctypes.addressof(sums), 0, ctypes.sizeof(sums))
        return result

    monkeypatch.setattr(
        kernels,
        "load_meshcore",
        lambda: {"phx_mc_polynomial_jacobian_integrals": corrupted},
    )
    assert exact_jacobian_integrals([matrix], "simplex", 2) != [expected]
