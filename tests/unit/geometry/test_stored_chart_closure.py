from pathlib import Path

import numpy as np

from phydrax.geometry._meshing_domain import MeshingDomain, PatchCurveUse
from phydrax.geometry.brep._boolean_core import PeriodicPCurve
from phydrax.geometry.brep._intersection_curve import CurveTrimSegment, IntersectionPCurve
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
