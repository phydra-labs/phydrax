#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import numpy as np
import pytest

import phydrax.geometry._common_root_volume_overlap as overlap
from phydrax._polynomial._cubature import cubature_rule_data
from phydrax.discretization._coordinate_enclosure import (
    axes,
    constant,
    Expression,
    sum_polynomials,
)


def _tet(*points: tuple[Fraction, Fraction, Fraction]) -> overlap.Tet:
    return overlap._as_tetrahedron(points)


def _integrate(
    chart: tuple[Expression, ...], jacobian: Expression, pieces: tuple[overlap.Tet, ...]
) -> tuple[float, float, int, int]:
    rule = cubature_rule_data("tetrahedron", 8)
    return overlap._integrate_pulled_reference_overlap(
        chart,
        jacobian,
        pieces,
        overlap.CurvedOverlapIntegrationPolicy(),
        np.asarray(rule.points, dtype=np.float64),
        np.asarray(rule.weights, dtype=np.float64),
    )


_HALF, _ZERO = Fraction(1, 2), Fraction(0)
# Endpoint chart: the root sub-tetrahedron scaled by 1/2 at the root origin.
_CHART = overlap._affine_arguments_from_vertices(
    _tet(
        (_ZERO, _ZERO, _ZERO),
        (_HALF, _ZERO, _ZERO),
        (_ZERO, _HALF, _ZERO),
        (_ZERO, _ZERO, _HALF),
    )
)
_JACOBIAN = sum_polynomials((constant(1, 3), axes(3)[0]), 3)


def test_affine_pullback_matches_the_dyadic_classifier_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quarter = Fraction(1, 4)
    pieces = (
        _tet(
            (_ZERO, _ZERO, _ZERO),
            (quarter, _ZERO, _ZERO),
            (_ZERO, quarter, _ZERO),
            (_ZERO, _ZERO, quarter),
        ),
    )
    fast = _integrate(_CHART, _JACOBIAN, pieces)
    monkeypatch.setattr(overlap, "_affine_reference_preimages", lambda *_: None)
    classified = _integrate(_CHART, _JACOBIAN, pieces)
    # The dyadic piece is representable: the pullback is exact (zero error) and
    # equals the classifier, whose leaves resolve the same exact integral.
    assert fast[1] == 0.0
    assert fast[0] == classified[0]
    assert abs(fast[0] - classified[0]) <= classified[1]
    assert fast[2:] == (1, 0)


def test_affine_pullback_is_exact_on_non_dyadic_pieces() -> None:
    third, quarter = Fraction(1, 3), Fraction(1, 4)
    piece = _tet(
        (_ZERO, _ZERO, _ZERO),
        (third, _ZERO, _ZERO),
        (_ZERO, quarter, _ZERO),
        (third / 4, quarter / 4, quarter / 2),
    )
    # The piece lies in the chart image, so the classifier is never consulted.
    assert overlap._affine_reference_preimages(_CHART, (piece,)) is not None
    value, error, exact, enclosed = _integrate(_CHART, _JACOBIAN, (piece,))
    # Preimage under the chart doubles every coordinate; integrate 1 + x exactly.
    preimage = overlap._as_tetrahedron(
        tuple(tuple(2 * value for value in point) for point in piece)
    )
    volume = abs(overlap._determinant3(preimage)) / 6
    centroid_x = sum((point[0] for point in preimage), Fraction(0)) / 4
    expected = volume * (1 + centroid_x)
    # One exact integral, reported at binary64 resolution of its exact value.
    assert (exact, enclosed) == (1, 0)
    assert abs(Fraction(value) - expected) <= Fraction(np.spacing(float(expected)))
    assert error <= np.spacing(float(expected))
