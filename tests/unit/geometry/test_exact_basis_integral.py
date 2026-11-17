# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

import numpy as np
import pytest

from phydrax._exact_polynomial_kernels import exact_basis_jacobian_integrals
from phydrax._meshcore import meshcore_available
from phydrax.discretization._coordinate_enclosure import BernsteinDomain, Polynomial
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.discretization._exact_coordinate_bank import ExactCoordinateBank


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(not meshcore_available(), reason="meshcore unavailable"),
]


def _reference(
    basis: tuple[dict[tuple[int, ...], Fraction], ...],
    points: tuple[tuple[Fraction, ...], ...],
    domain: str,
) -> Fraction:
    d = len(points[0])
    jac: list[list[dict[tuple[int, ...], Fraction]]] = [
        [{} for _ in range(d)] for _ in range(d)
    ]
    for poly, point in zip(basis, points, strict=True):
        for power, coefficient in poly.items():
            for axis in range(d):
                if power[axis] == 0:
                    continue
                derivative = tuple(p - (i == axis) for i, p in enumerate(power))
                for component in range(d):
                    entry = jac[component][axis]
                    entry[derivative] = entry.get(derivative, Fraction(0)) + (
                        coefficient * point[component] * power[axis]
                    )
    result = Fraction(0)
    for permutation in itertools.permutations(range(d)):
        sign = (-1) ** sum(
            permutation[i] > permutation[j] for i in range(d) for j in range(i + 1, d)
        )
        product = {(0,) * d: Fraction(sign)}
        for row, column in enumerate(permutation):
            following: dict[tuple[int, ...], Fraction] = {}
            for a, av in product.items():
                for b, bv in jac[row][column].items():
                    key = tuple(x + y for x, y in zip(a, b, strict=True))
                    following[key] = following.get(key, Fraction(0)) + av * bv
            product = following
        for powers, value in product.items():
            if domain == "simplex":
                moment = Fraction(
                    math.prod(math.factorial(k) for k in powers),
                    math.factorial(sum(powers) + d),
                )
            elif domain == "box":
                moment = Fraction(1, math.prod(k + 1 for k in powers))
            else:
                moment = Fraction(
                    math.factorial(powers[0]) * math.factorial(powers[1]),
                    math.factorial(powers[0] + powers[1] + 2) * (powers[2] + 1),
                )
            result += value * moment
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
def test_randomized_basis_parity_and_partition(
    domain: BernsteinDomain, dimension: int
) -> None:
    rng = random.Random(76213 + dimension)
    basis = tuple(
        {
            tuple(rng.randrange(2) for _ in range(dimension)): Fraction(
                rng.randrange(-3, 4), rng.randrange(1, 5)
            )
            for _ in range(4)
        }
        for _ in range(4)
    )
    rows = tuple(
        tuple(
            Fraction(rng.randrange(-7, 8), rng.randrange(1, 7)) for _ in range(dimension)
        )
        for _ in range(12)
    )
    bank = ExactCoordinateBank.from_rows(rows)
    routes = np.asarray(
        [[rng.randrange(len(rows)) for _ in basis] for _ in range(30)], dtype=np.int64
    )
    expected = [
        _reference(basis, tuple(rows[int(i)] for i in route), domain) for route in routes
    ]
    for workers in (1, 4):
        assert (
            exact_basis_jacobian_integrals(
                basis, bank, routes, domain, dimension, parallelism=workers
            )
            == expected
        )
    actual = [
        value
        for start in range(0, len(routes), 7)
        for value in exact_basis_jacobian_integrals(
            basis, bank, routes[start : start + 7], domain, dimension
        )
    ]
    assert actual == expected


def test_item_local_refusal_and_large_words() -> None:
    basis: tuple[Polynomial, ...] = (
        {(0,): Fraction(1), (1,): Fraction(-1)},
        {(1,): Fraction(1)},
    )
    rows = (
        (Fraction(0),),
        (Fraction(1, 3),),
        (RadicalElement.sqrt(Fraction(2)),),
        (Fraction(2**300),),
        (Fraction(2**5000),),
    )
    bank = ExactCoordinateBank.from_rows(rows)
    routes = np.asarray([[0, 1], [0, 2], [0, 3], [0, 4], [9, 1], [1, 0]], dtype=np.int64)
    assert exact_basis_jacobian_integrals(
        basis, bank, routes, "box", 1, parallelism=3
    ) == [Fraction(1, 3), None, Fraction(2**300), None, None, Fraction(-1, 3)]
