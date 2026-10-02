from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest

from phydrax.operators.differential._taylor_certification import (
    taylor_prime_candidates,
    TaylorCurveCertificate,
    TaylorLinearCertificate,
)
from phydrax.operators.differential._taylor_contracts import TaylorContractionResources


def test_curve_certificate_counts_all_derivative_orders_and_kdv_collision() -> None:
    # Degree five also contains D^5[v^5], not just the desired D^3[v,v,w].
    collision = TaylorCurveCertificate((1, 3), (2, 1))
    assert not collision.valid
    assert collision.partition_count == 2
    isolated = TaylorCurveCertificate((2, 3), (2, 1))
    assert isolated.valid
    assert isolated.coefficient_order == 7
    assert Fraction(*isolated.target_coefficient) == Fraction(1, 2)
    assert Fraction(*isolated.extraction_weight) == 2


def test_mixed_seven_certificate_has_exact_small_factorial_extraction() -> None:
    certificate = TaylorCurveCertificate((5, 7), (3, 4))
    assert certificate.valid
    assert certificate.coefficient_order == 43
    assert certificate.target_coefficient == (1, 144)
    assert certificate.extraction_weight == (144, 1)
    solutions = [(n, m) for n in range(44) for m in range(44) if 5 * n + 7 * m == 43]
    assert solutions == [(3, 4)]


def test_signed_binomial_certificate_is_identity_on_independent_monomials() -> None:
    counts = (2, 1, 2)
    certificate = TaylorLinearCertificate(counts)
    order = sum(counts)
    for powers in itertools.product(range(order + 1), repeat=3):
        if sum(powers) != order:
            continue
        coefficient = sum(
            Fraction(
                weight * math.prod(s**n for s, n in zip(scales, powers, strict=True)),
                math.prod(math.factorial(n) for n in powers),
            )
            for scales, weight in certificate.terms
        )
        assert coefficient == (1 if powers == counts else 0), f"monomial powers={powers}"


def test_prime_candidates_are_bounded_deterministic_and_prime() -> None:
    limits = TaylorContractionResources(max_order=50, max_candidates=20)
    first = tuple(taylor_prime_candidates((2, 1), resources=limits))
    assert first == tuple(taylor_prime_candidates((2, 1), resources=limits))
    assert len(first) <= limits.max_candidates
    assert (2, 3) in first
    for degrees in first:
        assert sum(a * m for a, m in zip(degrees, (2, 1), strict=True)) <= 50, (
            f"degrees={degrees}"
        )
        assert all(
            a >= 2 and all(a % d for d in range(2, math.isqrt(a) + 1)) for a in degrees
        ), f"degrees={degrees}"


def test_prime_degrees_alone_do_not_certify_isolation() -> None:
    assert not TaylorCurveCertificate((2, 3), (3, 2)).valid
    assert TaylorCurveCertificate((5, 7), (3, 4)).valid


def test_curve_certificate_refuses_order_resources() -> None:
    with pytest.raises(ValueError, match="resources"):
        TaylorCurveCertificate((1000, 1001), (1000, 1000))


def test_linear_certificate_refuses_combinatorial_resources() -> None:
    with pytest.raises(ValueError, match="resources"):
        TaylorLinearCertificate((20,) * 20)


def test_curve_certificate_refuses_state_resources() -> None:
    with pytest.raises(ValueError, match="resources"):
        TaylorCurveCertificate(
            (5, 7),
            (3, 4),
            resources=TaylorContractionResources(max_certificate_states=20),
        )


def test_curve_certificate_refuses_boolean_degree() -> None:
    with pytest.raises(TypeError, match="integer"):
        TaylorCurveCertificate((True,), (1,))


def test_curve_certificate_refuses_misaligned_multiplicities() -> None:
    with pytest.raises(ValueError, match="align"):
        TaylorCurveCertificate((1, 2), (1,))
