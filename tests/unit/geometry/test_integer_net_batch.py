from __future__ import annotations

from fractions import Fraction
from random import Random

import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization._coordinate_enclosure import (
    _controls_input,
    _fraction_controls,
    BernsteinDomain,
    Expression,
)
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry import _stratum_subdivision as subdivision
from phydrax.geometry._integer_net_batch import integer_net_batch


def _exact_controls(
    polynomial: Expression,
    domain: BernsteinDomain,
    dimension: int,
    degree: tuple[int, ...],
) -> tuple[object, ...]:
    if not isinstance(polynomial, dict):
        raise ValueError("expected a polynomial")
    value, normalized_degree = _controls_input(polynomial, domain, dimension, degree)
    return _fraction_controls(value, domain, dimension, normalized_degree)


@pytest.mark.parametrize("workers", [1, 2])
def test_mixed_integer_banks_match_independent_exact_controls(workers: int) -> None:
    if not meshcore_available():
        pytest.skip("native meshcore is unavailable")
    random = Random(1024)
    requests: list[
        tuple[tuple[Expression, ...], BernsteinDomain, int, tuple[int, ...]]
    ] = []
    domains: tuple[tuple[BernsteinDomain, int, tuple[tuple[int, ...], ...]], ...] = (
        ("simplex", 2, ((1,), (2,), (4,))),
        ("prism", 3, ((1, 1), (2, 2), (3, 1))),
        ("box", 2, ((1, 2), (3, 1), (2, 2))),
    )
    for domain, dimension, degrees in domains:
        for degree in degrees:
            polynomials: list[Expression] = []
            for _ in range(3):
                powers = [(0,) * dimension]
                powers.extend(
                    tuple(int(axis == selected) for axis in range(dimension))
                    for selected in range(dimension)
                )
                polynomial: Expression = {
                    power: Fraction(random.randrange(-8, 9), random.randrange(1, 10))
                    for power in powers
                }
                polynomials.append(polynomial)
            requests.append((tuple(polynomials), domain, dimension, degree))
    huge: Expression = {(0, 0): Fraction(1 << 4200, 7)}
    radical: Expression = {(0, 0): RadicalElement.sqrt(2)}
    zero: Expression = {(0, 0): Fraction(0)}
    requests.append(((huge, radical, zero), "simplex", 2, (2,)))

    def values(
        batch: list[tuple[tuple[Expression, ...], BernsteinDomain, int, tuple[int, ...]]],
    ) -> list[tuple[tuple[object, ...], ...]]:
        result = []
        for (polynomials, domain, dimension, degree), nets in zip(
            batch, integer_net_batch(batch, parallelism=workers), strict=True
        ):
            result.append(
                tuple(
                    net.controls()
                    if net is not None
                    else _exact_controls(polynomial, domain, dimension, degree)
                    for polynomial, net in zip(polynomials, nets, strict=True)
                )
            )
        return result

    expected = [
        tuple(
            _exact_controls(polynomial, domain, dimension, degree)
            for polynomial in polynomials
        )
        for polynomials, domain, dimension, degree in requests
    ]
    actual = values(requests)
    assert actual == expected
    assert values(requests[:4]) + values(requests[4:]) == expected


def test_perturbed_input_changes_exact_controls() -> None:
    if not meshcore_available():
        pytest.skip("native meshcore is unavailable")
    polynomials: tuple[Expression, ...] = (
        {(0,): Fraction(1, 3), (1,): Fraction(-2, 7)},
        {(0,): Fraction(1, 3), (1,): Fraction(-1, 7)},
    )
    expected = (
        (Fraction(1, 3), Fraction(1, 21)),
        (Fraction(1, 3), Fraction(4, 21)),
    )
    for polynomial, reference in zip(polynomials, expected, strict=True):
        controls = subdivision._controls((polynomial,), (2,), (1,), 1)
        assert controls is not None
        actual = tuple(
            controls[index][0] for index in subdivision._control_indices((2,), (1,))
        )
        assert actual == reference
    assert expected[0] != expected[1]
