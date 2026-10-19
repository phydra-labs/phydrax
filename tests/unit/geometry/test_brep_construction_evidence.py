"""Profile residual evidence and whole-curve correspondence outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.geometry import PlanarProfile, ProfileLoop, ProfilePlane
from phydrax.geometry.brep._correspondence import CurveSurfaceCorrespondence
from phydrax.geometry.brep._intersection_curve import AffinePCurve
from phydrax.geometry.brep._patches import (
    CircleCurve,
    CylinderPatch,
    LineCurve,
    PlanePatch,
)
from phydrax.geometry.brep._source_correspondence import circle_correspondence


def test_profile_retains_weighted_solver_residual_without_changing_geometry() -> None:
    loop = ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    authored = PlanarProfile(ProfilePlane(), loop)
    solved = PlanarProfile(ProfilePlane(), loop, residual_bound=0.125)
    assert authored.residual_bound == 0.0
    assert solved.residual_bound == 0.125
    assert solved.outer == authored.outer
    assert solved.holes == authored.holes


@pytest.mark.parametrize(
    "bound", [-1.0, np.inf, np.nan], ids=["negative", "infinite", "nan"]
)
def test_profile_refuses_invalid_residual_bound(bound: float) -> None:
    loop = ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="PlanarProfile.residual_bound"):
        PlanarProfile(ProfilePlane(), loop, residual_bound=bound)


def test_correspondence_outcomes_preserve_interval_evidence_when_rescaled() -> None:
    unresolved = ((2.0, 3.0), (0.0, 1.0))
    evidence = CurveSurfaceCorrespondence.unproved(unresolved, 7)
    transported = evidence.rescale(np.diag(np.asarray((2.0, 1.0, 1.0))))
    assert not transported.complete
    assert transported.deviation_bound == np.inf
    assert transported.unresolved == unresolved
    assert transported.intervals_processed == 7


def test_correspondence_identity_remains_zero_under_placement() -> None:
    evidence = CurveSurfaceCorrespondence.identity(3)
    transported = evidence.rescale(np.diag(np.asarray((2.0, 1.0, 1.0))))
    assert transported.complete
    assert transported.deviation_bound == 0.0
    assert transported.intervals_processed == 3


def test_correspondence_enclosure_is_transported_outward() -> None:
    evidence = CurveSurfaceCorrespondence.enclosed(0.25, 5)
    transported = evidence.rescale(np.diag(np.asarray((2.0, 1.0, 1.0))))
    # The diagonal map has norm two; the upper limit covers directed sqrt and
    # product rounding with eight binary64 epsilons on the closed-form result.
    assert (
        0.5 <= transported.deviation_bound <= 0.5 * (1.0 + 8.0 * np.finfo(np.float64).eps)
    )
    assert transported.complete
    assert transported.intervals_processed == 5


def test_circle_plane_correspondence_retains_affine_pcurve_harmonics() -> None:
    source = CircleCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0)
    parameters = AffinePCurve(
        CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 0.5),
        ((2.0, 0.0), (0.0, 2.0)),
        (0.0, 0.0),
    )
    plane = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert circle_correspondence(source, parameters, plane) == 0.0


def test_circle_cylinder_correspondence_requires_identity_angular_parameter() -> None:
    source = CircleCurve((0.0, 0.0, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0)
    cylinder = CylinderPatch(
        origin=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        first_axis=(1.0, 0.0, 0.0),
        second_axis=(0.0, 1.0, 0.0),
        radius=1.0,
    )
    assert (
        circle_correspondence(source, LineCurve((0.0, 2.0), (1.0, 0.0)), cylinder) == 0.0
    )
    assert (
        circle_correspondence(source, LineCurve((0.0, 2.0), (2.0, 0.0)), cylinder) is None
    )
