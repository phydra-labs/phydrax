"""Affine spline trim proofs retain source coefficients and reject wider families."""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.brep import BSplineCurve, CurveTrimSegment, PlanePatch
from phydrax.geometry.brep._constructors import NormalizedTrimCurve


def _normalized(curve: BSplineCurve) -> NormalizedTrimCurve:
    domain = curve.parameter_domain
    if domain is None:
        raise ValueError("_normalized.curve must have a finite active domain; got None.")
    return NormalizedTrimCurve(
        CurveTrimSegment(curve, *domain),
        jnp.asarray((0.125, -0.25), dtype=jnp.float64),
        jnp.asarray((0.5, 0.25), dtype=jnp.float64),
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        None,
        None,
        None,
        None,
    )


def test_two_control_clamped_equal_weight_spline_retains_rational_chord() -> None:
    controls = np.asarray(((0.317, 0.487), (0.211, 0.719)), dtype=np.float64)
    source = BSplineCurve(controls, (2, 2), (-1, -1, 2, 2), 1)
    curve = _normalized(source)
    evidence = curve.affine_chord_evidence(0, 1)
    assert evidence is not None
    expected = tuple(
        (
            (Fraction(float(row[0])) - Fraction(1, 8)) / Fraction(1, 2),
            (Fraction(float(row[1])) + Fraction(1, 4)) / Fraction(1, 4),
        )
        for row in controls
    )
    assert evidence.nominal_start == expected[0]
    assert evidence.nominal_end == expected[1]
    assert np.array_equal(np.asarray(source.control_points), controls)


@pytest.mark.parametrize(
    "source",
    [
        BSplineCurve(((0, 0), (1, 0), (2, 0)), (1, 1, 1), (0, 0, 0.5, 1, 1), 1),
        BSplineCurve(((0, 0), (1, 1)), (1, 2), (0, 0, 1, 1), 1),
        BSplineCurve(((0, 0), (1, 1)), (1, 1), (-1, 0, 1, 2), 1),
    ],
    ids=["three-controls", "unequal-weights", "unclamped-knots"],
)
def test_spline_outside_proved_affine_family_uses_generic_trim(
    source: BSplineCurve,
) -> None:
    assert _normalized(source).affine_chord_evidence(0, 1) is None
