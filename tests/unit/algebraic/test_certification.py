#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import mpmath
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

    assert isolated == (phx.algebraic.ExactRealRootInterval(root, root, 1),)


def test_sturm_rational_reconstruction_preserves_large_denominator_roots() -> None:
    denominator = 2**80 - 1
    root = Fraction(1, denominator)
    isolated = phx.algebraic.isolate_univariate_real_roots((-1, denominator))

    assert isolated == (phx.algebraic.ExactRealRootInterval(root, root, 1),)


def test_high_bit_irrational_roots_use_certified_modular_exclusion() -> None:
    # Modulo 3, 2**400*x**12 - 2 is x**12 + 1 and has no roots.
    # Thus neither real root can be rational, despite the 401-bit leading
    # coefficient. Independent monotonicity and exact endpoint signs certify
    # the two roots without relying on the implementation's Sturm sequence.
    leading = 2**400
    tolerance = Fraction(1, 2**45)
    coefficients = (-2, *(0 for _ in range(11)), leading)
    roots = phx.algebraic.isolate_univariate_real_roots(coefficients, tolerance=tolerance)
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
    assert roots[0].lower ** 2 > 2 > roots[0].upper ** 2
    assert roots[2].lower ** 2 < 2 < roots[2].upper ** 2


def test_univariate_distinct_multiplicity_factors_are_globally_separated() -> None:
    # (x**2 - 2)**2 * (10000*x**2 - 20001)
    roots = phx.algebraic.isolate_univariate_real_roots(
        (-80004, 0, 120004, 0, -60001, 0, 10000),
        tolerance=Fraction(1, 5),
    )
    assert len(roots) == 4
    assert [root.multiplicity for root in roots] == [1, 2, 2, 1]
    assert all(left.upper < right.lower for left, right in zip(roots, roots[1:]))
    for root, target in zip(
        roots, (Fraction(20001, 10000), 2, 2, Fraction(20001, 10000))
    ):
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


_WIN_A = float.fromhex("0x1.ec1cb8bd3c649p+1")
_WIN_B = float.fromhex("0x1.df82be8e5f4f2p+1")


def _rotated_squares_system(a: float, b: float) -> Any:
    # (x + y)**2 - a and (x - y)**2 - b.
    return phx.algebraic.SparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("f1", "f2"),
        np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        np.asarray([[2, 0], [1, 1], [0, 2], [0, 0]] * 2),
        np.asarray([1.0, 2.0, 1.0, -a, 1.0, -2.0, 1.0, -b]),
    )


def _rotated_squares_roots(a: float, b: float) -> list[tuple[Any, Any]]:
    # Exact roots ((±√a ± √b)/2, (±√a ∓ √b)/2), evaluated with 60 digits.
    roots = []
    with mpmath.workdps(60):
        for u in (mpmath.sqrt(a), -mpmath.sqrt(a)):
            for v in (mpmath.sqrt(b), -mpmath.sqrt(b)):
                roots.append((+((u + v) / 2), +((u - v) / 2)))
    return roots


def _inside(value: Any, lower: Fraction, upper: Fraction) -> bool:
    with mpmath.workdps(60):
        return mpmath.mpf(lower) <= value <= mpmath.mpf(upper)


def _roots_in_box(
    roots: list[tuple[Any, Any]], center: np.ndarray, radius: np.ndarray
) -> list[tuple[Any, Any]]:
    # The certified box is the exact rational box [center - radius, center + radius].
    bounds = [
        (Fraction(float(c)) - Fraction(float(r)), Fraction(float(c)) + Fraction(float(r)))
        for c, r in zip(center, radius, strict=True)
    ]
    return [
        root
        for root in roots
        if all(_inside(root[axis], *bounds[axis]) for axis in range(2))
    ]


def _image_encloses(certificate: Any, root: tuple[Any, Any]) -> bool:
    return all(
        _inside(
            root[axis],
            Fraction(float(certificate.image_lower[axis])),
            Fraction(float(certificate.image_upper[axis])),
        )
        for axis in range(2)
    )


def test_krawczyk_refuses_rootless_box_with_mixed_sign_preconditioner() -> None:
    # WIN-8 I-46a: J(center)^-1 has mixed signs; endpoint products I - Y J_lo and
    # I - Y J_hi do not enclose I - Y [J] and certified this rootless box.
    center = np.asarray(
        [float.fromhex("-0x1.298deeb1ff55bp-4"), float.fromhex("-0x1.0d97a286b786ap+1")]
    )
    radius = np.asarray(
        [float.fromhex("0x1.e5c26f3e6af36p-5"), float.fromhex("0x1.b21b9578a55b8p-3")]
    )
    roots = _rotated_squares_roots(_WIN_A, _WIN_B)
    assert _roots_in_box(roots, center, radius) == []

    certificate = phx.algebraic.krawczyk_certificate(
        _rotated_squares_system(_WIN_A, _WIN_B), center, radius
    )

    assert not certificate.unique_root


@pytest.mark.parametrize("root_index", range(4))
def test_krawczyk_tight_box_image_encloses_exact_root(root_index: int) -> None:
    # Sub-ulp Krawczyk images only enclose the irrational root when every
    # operation is rounded outward.
    root = _rotated_squares_roots(_WIN_A, _WIN_B)[root_index]
    center = np.asarray([float(root[0]), float(root[1])])
    radius = np.asarray([2.0**-30, 2.0**-30])

    certificate = phx.algebraic.krawczyk_certificate(
        _rotated_squares_system(_WIN_A, _WIN_B), center, radius
    )

    assert certificate.unique_root
    assert _image_encloses(certificate, root)


def test_krawczyk_unique_claims_hold_against_exact_roots() -> None:
    rng = np.random.default_rng(20261106)
    claims = 0
    false_claims = []
    for trial in range(400):
        a, b = (float(value) for value in rng.uniform(0.5, 4.0, 2))
        roots = _rotated_squares_roots(a, b)
        anchor = roots[int(rng.integers(0, 4))]
        if trial % 2:
            spread, radius = 0.6, rng.uniform(0.05, 1.5, 2)
        else:
            spread, radius = 0.02, rng.uniform(0.01, 0.2, 2)
        center = np.asarray([float(anchor[0]), float(anchor[1])]) + rng.normal(
            size=2
        ) * rng.uniform(0.0, spread)
        certificate = phx.algebraic.krawczyk_certificate(
            _rotated_squares_system(a, b), center, radius
        )
        if not certificate.unique_root:
            continue
        claims += 1
        inside = _roots_in_box(roots, center, radius)
        if len(inside) != 1 or not _image_encloses(certificate, inside[0]):
            false_claims.append((trial, a.hex(), b.hex(), len(inside)))

    assert false_claims == []
    assert claims >= 100


_SMALE_X0 = float.fromhex("0x1.422472f0af8bap-1")
_SMALE_C = float.fromhex("0x1.2b6c3e6f16d69p-3")


def _shifted_square_system(c: float) -> Any:
    return phx.algebraic.SparsePolynomialSystem.from_coo(
        ("x",), ("f",), np.asarray([0, 0]), np.asarray([[2], [0]]), np.asarray([1.0, -c])
    )


def _exact_square_alpha(x0: float, c: float) -> tuple[Fraction, Fraction, Fraction]:
    # f = x**2 - c: beta = |f/f'|, gamma = |f''/(2 f')|, alpha = beta * gamma.
    x, constant = Fraction(x0), Fraction(c)
    beta = abs(x * x - constant) / (2 * abs(x))
    gamma = 1 / (2 * abs(x))
    return beta, gamma, beta * gamma


def _below_smale_threshold(alpha: Fraction) -> bool:
    # alpha < (13 - 3 sqrt(17)) / 4  <=>  13 - 4 alpha > 0 and (13 - 4 alpha)**2 > 153.
    gap = 13 - 4 * alpha
    return gap > 0 and gap * gap > 153


def test_smale_alpha_refuses_exact_alpha_above_threshold() -> None:
    # WIN-8 I-46b: round-to-nearest alpha fell below alpha_0 at this point.
    _, _, alpha = _exact_square_alpha(_SMALE_X0, _SMALE_C)
    assert not _below_smale_threshold(alpha)

    certificate = phx.algebraic.smale_alpha_certificate(
        _shifted_square_system(_SMALE_C), jnp.asarray([_SMALE_X0])
    )

    assert not certificate.approximate_root
    assert Fraction(float(certificate.alpha_upper)) >= alpha


def test_smale_alpha_decision_and_bounds_match_exact_alpha_near_threshold() -> None:
    rng = np.random.default_rng(20261106)
    threshold = (13.0 - 3.0 * np.sqrt(17.0)) / 4.0
    outcomes = set()
    mismatches = []
    for trial in range(300):
        x0 = float(rng.uniform(0.5, 2.0))
        target = threshold * (1.0 + float(rng.uniform(-4e-15, 4e-15)))
        c = x0 * x0 * (1.0 - 4.0 * target)
        beta, gamma, alpha = _exact_square_alpha(x0, c)
        certificate = phx.algebraic.smale_alpha_certificate(
            _shifted_square_system(c), jnp.asarray([x0])
        )
        expected = _below_smale_threshold(alpha)
        outcomes.add(expected)
        if (
            certificate.approximate_root != expected
            or Fraction(float(certificate.beta)) < beta
            or Fraction(float(certificate.gamma_upper)) < gamma
            or Fraction(float(certificate.alpha_upper)) < alpha
        ):
            mismatches.append((trial, x0.hex(), c.hex(), expected))

    assert mismatches == []
    assert outcomes == {True, False}
