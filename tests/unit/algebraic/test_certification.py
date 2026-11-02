#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _square_root_system() -> Any:
    return phx.algebraic.SparsePolynomialSystem.from_coo(
        ("x",),
        ("equation",),
        (0, 0),
        ((0,), (2,)),
        jnp.asarray((-2.0, 1.0)),
    )


def test_certification_scenario_1() -> None:
    certificate = phx.algebraic.smale_alpha_certificate(
        _square_root_system(), jnp.asarray([jnp.sqrt(2.0)])
    )

    assert certificate.approximate_root
    assert certificate.alpha_upper < 1.0e-10
    assert certificate.beta < 1.0e-10
    certificate = phx.algebraic.krawczyk_certificate(
        _square_root_system(),
        jnp.asarray([1.4142]),
        jnp.asarray([0.01]),
    )

    assert certificate.unique_root
    assert certificate.image_lower[0] > certificate.center[0] - certificate.radius[0]
    assert certificate.image_upper[0] < certificate.center[0] + certificate.radius[0]
    square_roots = phx.algebraic.isolate_univariate_real_roots((-2, 0, 1))
    repeated = phx.algebraic.isolate_univariate_real_roots((1, -2, 1))

    assert len(square_roots) == 2
    assert square_roots[0].upper < 0 < square_roots[1].lower
    assert repeated == (phx.algebraic.ExactRealRootInterval(Fraction(1), Fraction(1), 2),)


def test_univariate_sturm_isolation_avoids_rational_divisor_enumeration() -> None:
    root = Fraction(1, 2**80)
    isolated = phx.algebraic.isolate_univariate_real_roots(
        (-1, 2**80), tolerance=Fraction(1, 2**90)
    )

    assert isolated == (
        phx.algebraic.ExactRealRootInterval(root, root, 1),
    )



def test_sturm_rational_reconstruction_preserves_large_denominator_roots() -> None:
    denominator = 2**80 - 1
    root = Fraction(1, denominator)
    isolated = phx.algebraic.isolate_univariate_real_roots(
        (-1, denominator)
    )

    assert isolated == (
        phx.algebraic.ExactRealRootInterval(root, root, 1),
    )


def test_high_bit_irrational_roots_use_certified_modular_exclusion() -> None:
    # Modulo 3, 2**400*x**12 - 2 is x**12 + 1 and has no roots.
    # Thus neither real root can be rational, despite the 401-bit leading
    # coefficient. Independent monotonicity and exact endpoint signs certify
    # the two roots without relying on the implementation's Sturm sequence.
    leading = 2**400
    tolerance = Fraction(1, 2**45)
    coefficients = (-2, *(0 for _ in range(11)), leading)
    roots = phx.algebraic.isolate_univariate_real_roots(
        coefficients, tolerance=tolerance
    )
    assert len(roots) == 2
    assert roots[0].upper < 0 < roots[1].lower
    for root in roots:
        assert root.multiplicity == 1
        assert 0 < root.upper - root.lower <= tolerance
        assert root.upper - root.lower > Fraction(1, 2 * leading**2)
        left_value = leading * root.lower**12 - 2
        right_value = leading * root.upper**12 - 2
        assert left_value * right_value < 0


def test_univariate_rational_reconstruction_with_irrational_roots() -> None:
    # (q*x - 1)(x**2 - 2): no prime can exclude all rational roots.
    q = 2**40 - 1
    roots = phx.algebraic.isolate_univariate_real_roots(
        (2, -2 * q, -1, q), tolerance=Fraction(1, 100)
    )
    assert len(roots) == 3
    assert roots[0].upper < Fraction(1, q) < roots[2].lower
    assert roots[1] == phx.algebraic.ExactRealRootInterval(
        Fraction(1, q), Fraction(1, q), 1
    )
    assert roots[0].lower**2 > 2 > roots[0].upper**2
    assert roots[2].lower**2 < 2 < roots[2].upper**2


def test_univariate_distinct_multiplicity_factors_are_globally_separated() -> None:
    # (x**2 - 2)**2 * (10000*x**2 - 20001)
    roots = phx.algebraic.isolate_univariate_real_roots(
        (-80004, 0, 120004, 0, -60001, 0, 10000),
        tolerance=Fraction(1, 5),
    )
    assert len(roots) == 4
    assert [root.multiplicity for root in roots] == [1, 2, 2, 1]
    assert all(left.upper < right.lower for left, right in zip(roots, roots[1:]))
    for root, target in zip(roots, (Fraction(20001, 10000), 2, 2, Fraction(20001, 10000))):
        assert min(root.lower**2, root.upper**2) < target
        assert max(root.lower**2, root.upper**2) > target


def test_binary64_stationary_polynomial_exact_root_contract() -> None:
    # Primitive degree-12 polynomial for a rational cylinder stationary query
    # with binary64-derived q=(0.21721224020356145, -1.006813808941921,
    # 2.112519713576212). Independent exact real-root isolation gives these
    # four disjoint intervals and multiplicities.
    coefficients = (
        195664092564277201694982827693081831859,
        4599382995324375921882509212382648841016,
        8518081739532170686601197341826359935254,
        4659017592733271518370065715896800490296,
        -388071143028358408004716032292744014259,
        -9258400588057647440252574928279449331312,
        -16651349378136178960582928274453395505708,
        -9258400588057647440252574928279449331312,
        -388071143028358408004716032292744014259,
        4659017592733271518370065715896800490296,
        8518081739532170686601197341826359935254,
        4599382995324375921882509212382648841016,
        195664092564277201694982827693081831859,
    )
    roots = phx.algebraic.isolate_univariate_real_roots(
        coefficients, tolerance=Fraction(1, 10**9)
    )
    assert len(roots) == 4
    assert [root.multiplicity for root in roots] == [1, 2, 1, 2]
    assert Fraction(-4135, 192) < roots[0].lower < roots[0].upper < Fraction(-3252, 151)
    assert roots[1] == phx.algebraic.ExactRealRootInterval(Fraction(-1), Fraction(-1), 2)
    assert Fraction(-2, 43) < roots[2].lower < roots[2].upper < Fraction(-13, 280)
    assert roots[3] == phx.algebraic.ExactRealRootInterval(Fraction(1), Fraction(1), 2)

def test_krawczyk_dense_admission_and_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    system = _square_root_system()
    center = np.asarray([1.4142])
    radius = np.asarray([0.01])
    required = 8 * (10 + 12)
    certificate = phx.algebraic.krawczyk_certificate(
        system, center, radius, maximum_dense_workspace_bytes=required
    )
    default = phx.algebraic.krawczyk_certificate(system, center, radius)
    assert certificate.dense_workspace_bytes == required
    assert certificate.maximum_dense_workspace_bytes == required
    assert certificate.certificate_id == default.certificate_id
    inverse = 1.0 / (2 * center[0])
    image_center = center[0] - inverse * (center[0] ** 2 - 2)
    image_radius = 2 * inverse * radius[0] ** 2
    np.testing.assert_allclose(certificate.image_lower, [image_center - image_radius])
    np.testing.assert_allclose(certificate.image_upper, [image_center + image_radius])

    def refuse_dense_jacobian(_system: Any, _point: Any) -> None:
        raise AssertionError("Dense Jacobian evaluated before workspace admission.")

    monkeypatch.setattr(type(system), "jacobian", refuse_dense_jacobian)
    with pytest.raises(ValueError, match="dense workspace requires 176 bytes"):
        phx.algebraic.krawczyk_certificate(
            system, center, radius, maximum_dense_workspace_bytes=required - 1
        )


@pytest.mark.parametrize("budget", (0, -1, True, 1.5))
def test_krawczyk_refuses_invalid_dense_budget(budget: int | float) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        phx.algebraic.krawczyk_certificate(
            _square_root_system(),
            np.asarray([1.4142]),
            np.asarray([0.01]),
            maximum_dense_workspace_bytes=budget,  # ty: ignore[invalid-argument-type]
        )
