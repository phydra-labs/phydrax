from __future__ import annotations

import math
from fractions import Fraction
from itertools import product
from random import Random

import numpy as np
import pytest

from phydrax.discretization._cell_geometry_validity import _bernstein_plan
from phydrax.discretization._coordinate_enclosure import BernsteinDomain
from phydrax.linalg._exact import solve_exact


def _basis(domain: str, degrees: tuple[int, ...], nodes: np.ndarray) -> list[list[Fraction]]:
    def axis(degree: int, value: Fraction) -> list[Fraction]:
        return [Fraction(math.comb(degree, i)) * value**i * (1 - value)**(degree - i) for i in range(degree + 1)]

    def simplex(degree: int, point: tuple[Fraction, ...]) -> list[Fraction]:
        bary = (1 - sum(point, Fraction(0)), *point)
        indices = (index for index in product(range(degree + 1), repeat=len(bary)) if sum(index) == degree)
        return [
            Fraction(math.factorial(degree), math.prod(math.factorial(i) for i in index))
            * math.prod((value**power for value, power in zip(bary, index, strict=True)), start=Fraction(1))
            for index in indices
        ]

    rows = []
    for node in nodes:
        point = tuple(Fraction(float(value)) for value in node)
        if domain == "box":
            factors = tuple(axis(degree, point[i]) for i, degree in enumerate(degrees))
        elif domain == "simplex":
            factors = (simplex(degrees[0], point),)
        else:
            factors = (simplex(degrees[0], point[:2]), axis(degrees[1], point[2]))
        rows.append([math.prod(values, start=Fraction(1)) for values in product(*factors)])
    return rows


@pytest.mark.parametrize(
    ("domain", "degrees"),
    [
        *((domain, (degree, degree)) for domain in ("box", "prism") for degree in range(5)),
        *(("simplex", (degree, dimension)) for dimension in (1, 2, 3) for degree in range(5)),
        ("box", (2, 3, 1)),
        ("prism", (2, 3)),
    ],
)
def test_conversion_table_encloses_exact_rational_interpolants(
    domain: BernsteinDomain, degrees: tuple[int, ...]
) -> None:
    plan = _bernstein_plan(domain, degrees)
    basis = _basis(domain, degrees, plan.nodes)
    size = len(basis)
    identity = [[Fraction(int(row == col)) for col in range(size)] for row in range(size)]
    inverse = solve_exact(basis, identity)
    assert not plan.nodes.flags.writeable
    assert not plan.coefficients_from_values.flags.writeable
    for row in range(size):
        errors = []
        for col in range(size):
            value = float(plan.coefficients_from_values[row, col])
            exact = inverse[row][col]
            error = abs(Fraction(value) - exact)
            gap = max(
                abs(Fraction(float(np.nextafter(value, np.inf))) - Fraction(value)),
                abs(Fraction(value) - Fraction(float(np.nextafter(value, -np.inf)))),
            )
            assert error <= gap / 2
            errors.append(error)
        assert sum(errors, Fraction(0)) <= Fraction(plan.conversion_rounding_norm)
    rng = Random(716 + sum(degrees))
    for _ in range(3):
        values = tuple(Fraction(rng.randint(-30, 30), 8) for _ in range(size))
        floating = plan.coefficients_from_values @ np.asarray([float(value) for value in values])
        magnitude = max(abs(value) for value in values)
        bound = Fraction(plan.conversion_rounding_norm) * magnitude
        bound += 2 * Fraction(plan.conversion_norm) * size * Fraction(2, 2**52) * magnitude
        for row in range(size):
            exact = sum((inverse[row][col] * values[col] for col in range(size)), Fraction(0))
            assert abs(Fraction(float(floating[row])) - exact) <= bound


def test_exact_conversion_avoids_recorded_float_solve_error() -> None:
    plan = _bernstein_plan("box", (3,))
    assert tuple(plan.nodes[:, 0]) == (0.0, 0.25, 0.75, 1.0)
    exact = Fraction(1, 3)
    # A float solve of this dyadic Bernstein matrix produced this entry.
    old = float.fromhex("0x1.5555555555554p-2")
    old_gap = Fraction(float(np.spacing(old)))
    assert abs(Fraction(old) - exact) > old_gap / 2
    current = float(plan.coefficients_from_values[2, 0])
    assert abs(Fraction(current) - exact) <= Fraction(float(np.spacing(current))) / 2
    # The former Chebyshev-node float solve also exceeded half an ulp at
    # degree two; both the exact inverse entry and observed float are fixed.
    original = float.fromhex("0x1.6dd707e2911c4p-4")
    original_exact = Fraction(
        836100811362416860007308138903,
        9361112124762313742037515367735,
    )
    assert abs(Fraction(original) - original_exact) > Fraction(float(np.spacing(original))) / 2
