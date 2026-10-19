"""White-box contracts for affine curve/face coefficient agreement admission."""

from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry._rational import rational_vector
from phydrax.geometry.brep import BRepTessellationPolicy
from phydrax.geometry.brep._affine_query import _affine_segment_agrees
from phydrax.geometry.brep._intersection_curve import affine_curve_coefficients
from phydrax.geometry.brep._patches import AbstractCurve, PlanePatch
from tests._support.cad_models import native_solid


@pytest.mark.parametrize(
    ("translation", "agrees"),
    ((Fraction(0), True), (Fraction(1), False)),
    ids=("identical-coefficients", "nonzero-gap"),
)
def test_affine_correspondence_requires_coefficient_equality(
    translation: Fraction, agrees: bool
) -> None:
    model = native_solid(
        "box", catalog="geometry", tessellation=BRepTessellationPolicy(realize=False)
    )
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError("box_model.geometry must contain native geometry; got None")
    patch = model.patches[0]
    assert isinstance(patch, PlanePatch)
    coedge = geometry.face_loops[0][0][0]
    pcurve = geometry.pcurves[coedge]
    assert isinstance(pcurve, AbstractCurve)
    parameters = affine_curve_coefficients(pcurve, np.zeros((2,), dtype=np.float64))
    assert parameters is not None
    origin = rational_vector(np.asarray(patch.origin))
    shifted = (origin[0] + translation, *origin[1:])
    plane = (
        shifted,
        rational_vector(np.asarray(patch.first_axis)),
        rational_vector(np.asarray(patch.second_axis)),
    )

    assert (
        _affine_segment_agrees(geometry, geometry.coedge_edges[coedge], parameters, plane)
        is agrees
    )
