#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import jax.numpy as jnp

import phydrax as phx


def _square_root_system():
    return phx.algebraic.SparsePolynomialSystem.from_coo(
        ("x",),
        ("equation",),
        (0, 0),
        ((0,), (2,)),
        jnp.asarray((-2.0, 1.0)),
    )


def test_smale_alpha_certifies_nearby_square_root() -> None:
    certificate = phx.algebraic.smale_alpha_certificate(
        _square_root_system(), jnp.asarray([jnp.sqrt(2.0)])
    )

    assert certificate.approximate_root
    assert certificate.alpha_upper < 1.0e-10
    assert certificate.beta < 1.0e-10


def test_krawczyk_certifies_unique_root_in_box() -> None:
    certificate = phx.algebraic.krawczyk_certificate(
        _square_root_system(),
        jnp.asarray([1.4142]),
        jnp.asarray([0.01]),
    )

    assert certificate.unique_root
    assert certificate.image_lower[0] > certificate.center[0] - certificate.radius[0]
    assert certificate.image_upper[0] < certificate.center[0] + certificate.radius[0]


def test_exact_real_root_isolation_preserves_multiplicity() -> None:
    square_roots = phx.algebraic.isolate_univariate_real_roots((-2, 0, 1))
    repeated = phx.algebraic.isolate_univariate_real_roots((1, -2, 1))

    assert len(square_roots) == 2
    assert square_roots[0].upper < 0 < square_roots[1].lower
    assert repeated == (
        phx.algebraic.ExactRealRootInterval(Fraction(1), Fraction(1), 2),
    )
