#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import pytest

from phydrax.algebraic import (
    multihomogeneous_bezout_forecast,
    polynomial_degree_profile,
    PolynomialVariableGroup,
    SparsePolynomialSupport,
    total_degree_bezout_forecast,
)


def test_bilinear_forecast_is_two_instead_of_total_degree_four():
    support = SparsePolynomialSupport(
        ("x", "y"),
        ("f", "g"),
        (0, 1),
        ((1, 1), (1, 1)),
        groups=(
            PolynomialVariableGroup("left", (0,)),
            PolynomialVariableGroup("right", (1,)),
        ),
    )

    profile = polynomial_degree_profile(support)
    assert profile.total_degrees == (2, 2)
    assert profile.multidegrees == ((1, 1), (1, 1))
    assert total_degree_bezout_forecast(support).path_count == 4
    grouped = multihomogeneous_bezout_forecast(support)
    assert grouped.path_count == 2
    assert grouped.status == "applicable"


def test_ten_equation_bilinear_forecast_uses_arbitrary_precision_host_integers():
    exponents = tuple((1, 0, 0, 0, 0, 1, 0, 0, 0, 0) for _ in range(10))
    support = SparsePolynomialSupport(
        tuple(f"x{index}" for index in range(10)),
        tuple(f"f{index}" for index in range(10)),
        tuple(range(10)),
        exponents,
        groups=(
            PolynomialVariableGroup("left", tuple(range(5))),
            PolynomialVariableGroup("right", tuple(range(5, 10))),
        ),
    )

    assert total_degree_bezout_forecast(support).path_count == 1024
    grouped = multihomogeneous_bezout_forecast(support)
    assert grouped.path_count == 252
    assert isinstance(grouped.path_count, int)


def test_projective_group_rejects_nonhomogeneous_equation_support():
    projective = PolynomialVariableGroup(
        "projective-line",
        (0, 1),
        geometry="projective",
    )
    assert projective.dimension == 1
    with pytest.raises(ValueError, match="not homogeneous"):
        SparsePolynomialSupport(
            ("x0", "x1"),
            ("f",),
            (0, 0),
            ((1, 0), (0, 0)),
            groups=(projective,),
        )
