from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from phydrax.geometry._domain_boundary import _curved_trim_lens, CurvedTrimLens
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    PatchCurveUse,
    PatchPoleUse,
    SourcePreparation,
)
from phydrax.geometry.brep import BRepAffinePlacement, edit_brep_occurrence_placement
from phydrax.geometry.brep._patches import BSplineCurve
from tests.unit.geometry.test_brep_blend import _with_rational_spline_cap_pcurve
from tests.unit.geometry.test_brep_placement_edit import _shared_model


def _boundary_use(
    domain: MeshingDomain, pcurve: BSplineCurve
) -> PatchCurveUse | PatchPoleUse:
    return next(
        use
        for patch in domain.patches
        for loop in patch.loops
        for use in loop
        if use.pcurve is pcurve
    )


def _lens(
    domain: MeshingDomain, pcurve: BSplineCurve, preparation: SourcePreparation
) -> CurvedTrimLens:
    use = _boundary_use(domain, pcurve)
    points = np.asarray(pcurve.evaluate(jnp.asarray((use.first, use.last))))
    first = (
        Fraction.from_float(float(points[0, 0])),
        Fraction.from_float(float(points[0, 1])),
    )
    last = (
        Fraction.from_float(float(points[1, 0])),
        Fraction.from_float(float(points[1, 1])),
    )
    return _curved_trim_lens(use, first, last, use.first, use.last, preparation)


def test_boundary_pcurve_spans_rebase_only_for_unchanged_trim() -> None:
    # This is an actual solid edge and its two face coedge trims, not a synthetic
    # carrier detached from the domain's boundary-use graph.
    source = _with_rational_spline_cap_pcurve(_shared_model(), 0)
    pcurve = next(
        curve for curve in source.geometry.pcurves if isinstance(curve, BSplineCurve)
    )
    domain = MeshingDomain.from_brep(source)
    prepared = SourcePreparation(domain)
    source_lens = _lens(domain, pcurve, prepared)
    in_hand = prepared.curve_spans(pcurve)

    translation = BRepAffinePlacement.from_matrix(
        ((1, 0, 0, 3), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )
    edited = edit_brep_occurrence_placement(source, ("first", "leaf"), translation).model
    moved_domain = MeshingDomain.from_brep(edited)
    rebased = prepared.for_placement_domain(moved_domain)
    assert rebased.curve_spans(pcurve) is in_hand
    warm_lens = _lens(moved_domain, pcurve, rebased)
    cold = SourcePreparation(moved_domain)
    cold_lens = _lens(moved_domain, pcurve, cold)
    assert cold.curve_spans(pcurve) is not in_hand
    assert warm_lens == cold_lens == source_lens

    # A second edge's new spline trim changes local source identity while the
    # original pcurve object remains in hand; it must not inherit the old cache.
    changed = _with_rational_spline_cap_pcurve(source, 1)
    assert any(curve is pcurve for curve in changed.geometry.pcurves)
    changed_domain = MeshingDomain.from_brep(changed)
    invalidated = prepared.for_placement_domain(changed_domain)
    assert invalidated.curve_spans(pcurve) is not in_hand
    assert _lens(changed_domain, pcurve, invalidated) == _lens(
        changed_domain, pcurve, SourcePreparation(changed_domain)
    )
