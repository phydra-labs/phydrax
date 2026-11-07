from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry._domain_boundary import _curved_trim_lens, CurvedTrimLens
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    PatchCurveUse,
    PatchPoleUse,
    SourcePreparation,
)
from phydrax.geometry._rational import fraction_down, fraction_up
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


def test_spline_curve_bounds_respect_exact_positive_weight_control_hull() -> None:
    curve = BSplineCurve(
        np.asarray(((0.0, 0.0), (1.0, 0.0))),
        np.asarray((1.0, 3.0)),
        np.asarray((0.0, 0.0, 1.0, 1.0)),
        1,
        exact_control_points=((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))),
        exact_weights=(Fraction(1), Fraction(3)),
        exact_knots=(Fraction(0), Fraction(0), Fraction(1), Fraction(1)),
    )
    full = curve.bounding_box(0.0, 1.0)
    np.testing.assert_array_equal(full, ((0.0, 0.0), (1.0, 0.0)))
    restricted = curve.bounding_box(0.25, 0.75)
    points = np.asarray(curve.evaluate(jnp.linspace(0.25, 0.75, 101)))
    assert np.all(points >= restricted[0])
    assert np.all(points <= restricted[1])
    assert 0.0 < restricted[0, 0] < restricted[1, 0] <= 1.0
    span = curve.prepare_spans().spans[0]
    exterior = span.enclose((Fraction(0), Fraction(1, 2))).value_bounds
    np.testing.assert_array_equal(
        curve.bounding_box(-0.25, 0.5),
        (
            tuple(fraction_down(bounds[0]) for bounds in exterior),
            tuple(fraction_up(bounds[1]) for bounds in exterior),
        ),
    )
    with pytest.raises(ValueError, match="do not intersect"):
        curve.bounding_box(2.0, 3.0)


def test_spline_curve_bounds_accept_outward_rounded_exact_knots() -> None:
    first, last = Fraction(1, 10), Fraction(9, 10)
    curve = BSplineCurve(
        np.asarray(((0.0, 0.0), (1.0, 0.0))),
        np.ones(2),
        np.asarray((first, first, last, last), dtype=float),
        1,
        exact_control_points=((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))),
        exact_weights=(Fraction(1), Fraction(1)),
        exact_knots=(first, first, last, last),
    )
    bounds = curve.bounding_box(fraction_down(first), fraction_up(last))
    assert np.all(bounds[0] <= (0.0, 0.0))
    assert np.all(bounds[1] >= (1.0, 0.0))
