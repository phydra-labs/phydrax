import math
from pathlib import Path

import numpy as np

from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    PatchCurveUse,
    SourcePreparation,
)
from phydrax.geometry.brep._boolean_core import PeriodicPCurve
from phydrax.geometry.brep._intersection import intersect_surface_regions
from phydrax.geometry.brep._intersection_curve import (
    CurveTrimSegment,
    IntersectionPCurve,
    SurfaceRegion,
)
from phydrax.geometry.brep._patches import PlanePatch, SpherePatch
from tests.unit.geometry.test_meshing_domain import _square_domain
from tools.meshing_qualification.scenarios.native_cad_profiles import _quadric_case


def test_closed_intersection_survives_trim_wrappers(tmp_path: Path) -> None:
    model, _, _, _ = _quadric_case("iges", tmp_path)
    domain = MeshingDomain.from_brep(model)
    seam = next(
        use
        for loop in domain.patches[1].loops
        for use in loop
        if isinstance(use, PatchCurveUse) and use.curve == 1
    )
    assert isinstance(seam.pcurve, IntersectionPCurve)
    assert seam.pcurve.curve.closed
    assert np.all(seam.pcurve.curve.closure_period_shifts()[:2] == 0)
    end = float(seam.pcurve.curve.num_charts)
    full = CurveTrimSegment(seam.pcurve, 0.0, end)
    reversed_full = CurveTrimSegment(full, 0.0, 1.0, reversed=True)
    translated = PeriodicPCurve(reversed_full, domain.patches[1].surface)

    assert PatchCurveUse(seam.curve, full, 0.0, 1.0).closes_chart_loop()
    assert PatchCurveUse(seam.curve, translated, 0.0, 1.0).closes_chart_loop()
    assert not PatchCurveUse(seam.curve, full, 0.0, 0.5).closes_chart_loop()
    partial = CurveTrimSegment(seam.pcurve, 0.0, end / 2.0)
    assert not PatchCurveUse(seam.curve, partial, 0.0, 1.0).closes_chart_loop()


def test_closed_intersection_rejects_nonzero_deck_shift(tmp_path: Path) -> None:
    model, _, _, _ = _quadric_case("iges", tmp_path)
    domain = MeshingDomain.from_brep(model)
    seam = next(
        use
        for loop in domain.patches[1].loops
        for use in loop
        if isinstance(use, PatchCurveUse) and use.curve == 1
    )
    assert isinstance(seam.pcurve, IntersectionPCurve)
    branch = seam.pcurve.curve
    np.testing.assert_array_equal(branch.closure_period_shifts(), (0, 0, 1, 0))
    pcurve = IntersectionPCurve(branch, "second")
    assert not PatchCurveUse(
        seam.curve, pcurve, 0.0, float(branch.num_charts)
    ).closes_chart_loop()
    wrapped = CurveTrimSegment(pcurve, 0.0, float(branch.num_charts))
    assert not PatchCurveUse(seam.curve, wrapped, 0.0, 1.0).closes_chart_loop()


def test_stored_intersection_point_uses_tighter_span_bound() -> None:
    plane = SurfaceRegion(
        PlanePatch((0, 0, 0.5), (1, 0, 0), (0, 1, 0)),
        np.asarray(((-2, -2), (2, 2)), dtype=np.float64),
    )
    sphere = SurfaceRegion(
        SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0),
        np.asarray(((0, -math.pi / 2), (math.tau, math.pi / 2)), dtype=np.float64),
    )
    intersection = intersect_surface_regions(plane, sphere)
    assert intersection.complete and len(intersection.curves) == 1
    pcurve = IntersectionPCurve(intersection.curves[0], "second")
    preparation = SourcePreparation(_square_domain())
    point_box = preparation.curve_box(pcurve, 0.25, 0.25)
    span_box = preparation.curve_box(pcurve, 0.25, 0.5)
    assert span_box[0, 0] > point_box[0, 0]
    assert np.all(
        np.maximum(point_box[0], span_box[0]) <= np.minimum(point_box[1], span_box[1])
    )
    use = PatchCurveUse(0, pcurve, 0.25, 0.5)
    actual = use.charts((0.25,), preparation=preparation)
    np.testing.assert_array_equal(actual[0], np.maximum(point_box[0], span_box[0]))
