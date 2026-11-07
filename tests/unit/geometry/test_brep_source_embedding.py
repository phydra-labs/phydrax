from fractions import Fraction as F
from math import pi, tau
from pathlib import Path

import numpy as np

from phydrax.geometry._meshing_domain import _PlacedSurfacePatch
from phydrax.geometry.brep._assembly import BRepAffinePlacement
from phydrax.geometry.brep._boolean_core import PeriodicPCurve
from phydrax.geometry.brep._patches import (
    CircleCurve,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    TorusPatch,
)
from phydrax.geometry.brep._placed import PlacedSurface
from phydrax.geometry.brep._source_embedding import (
    _constant_chart_line,
    _sheet_sign,
    certify_brep_trim_sheet,
    certify_source_embedding,
)
from phydrax.geometry.brep._trim_helpers import PeriodOffset


O = (0, 0, 0)
X = (1, 0, 0)
Y = (0, 1, 0)
Z = (0, 0, 1)
T = PeriodOffset(F(0), F(1))
Q = PeriodOffset(F(0), F(1, 4))
NQ = PeriodOffset(F(0), F(-1, 4))
BOX = ((F(0), F(0)), (F(1), F(1)))


def test_seam_coordinates_use_symbolic_native_period() -> None:
    surface = CylinderPatch(O, X, Y, Z, 1)
    base = LineCurve((0, 0), (0, 1))
    low = PeriodicPCurve(base, surface, (0, 0))
    high = PeriodicPCurve(base, surface, (1, 0))
    assert _constant_chart_line(low, 0) == (F(0), F(0), F(1))
    assert _constant_chart_line(high, 0) == (T, F(0), F(1))
    assert _constant_chart_line(LineCurve((0, 0), (1, 1)), 0) is None
    # Rational binary64 tau lies below mathematical tau, but is not equal to it.
    assert _sheet_sign(F.from_float(tau) - T) == -1


def test_plane_and_invertible_placement_bind_replay() -> None:
    plane = PlanePatch(O, X, Y)
    placed = PlacedSurface(plane, np.eye(3), np.asarray((3, 4, 5)))
    proof = certify_source_embedding(placed, BOX)
    assert proof is not None and proof.family == "plane"
    assert proof.replay(placed, BOX)
    assert not proof.replay(plane, BOX)
    assert not proof.replay(placed, ((F(0), F(0)), (F(2), F(1))))
    assert certify_source_embedding(PlanePatch(O, X, X), BOX) is None
    assert certify_source_embedding(PlanePatch((float("nan"), 0, 0), X, Y), BOX) is None


def test_domain_occurrence_placement_and_reflection_bind_source() -> None:
    cylinder = CylinderPatch(O, X, Y, Z, 2)
    placement = BRepAffinePlacement.from_matrix(
        ((-1, 0, 0, 3), (0, 2, 0, 4), (0, 0, 1, 5), (0, 0, 0, 1))
    )
    placed = _PlacedSurfacePatch(cylinder, placement)
    full = ((F(0), F(0)), (T, F(1)))
    proof = certify_source_embedding(placed, full, deck_axes=(0,))
    assert proof is not None and proof.family == "cylinder"
    assert proof.replay(placed, full, deck_axes=(0,))
    assert not proof.replay(cylinder, full, deck_axes=(0,))
    bad = _PlacedSurfacePatch(CylinderPatch(O, X, X, Z, 2), placement)
    assert certify_source_embedding(bad, full, deck_axes=(0,)) is None


def test_cylinder_requires_exact_native_deck_and_independent_frame() -> None:
    cylinder = CylinderPatch(O, X, Y, Z, 2)
    full = ((F(0), F(0)), (T, F(1)))
    proof = certify_source_embedding(cylinder, full, deck_axes=(0,))
    assert proof is not None and proof.family == "cylinder"
    assert proof.replay(cylinder, full, deck_axes=(0,))
    assert not proof.replay(cylinder, full)
    assert certify_source_embedding(cylinder, full) is None
    assert certify_source_embedding(cylinder, BOX, deck_axes=(0,)) is None
    assert (
        certify_source_embedding(
            cylinder, ((F(0), F(0)), (PeriodOffset(F(0), F(2)), F(1))), deck_axes=(0,)
        )
        is None
    )
    # Binary64 tau is not the symbolic deck, regardless of its proximity.
    rounded = ((F(0), F(0)), (F.from_float(tau), F(1)))
    assert certify_source_embedding(cylinder, rounded, deck_axes=(0,)) is None
    assert certify_source_embedding(CylinderPatch(O, X, X, Z, 2), BOX) is None
    assert certify_source_embedding(CylinderPatch(O, X, Y, Z, 0), BOX) is None


def test_revolution_cylinder_and_polar_plane_embedding() -> None:
    cylinder = RevolutionSurface(LineCurve(X, Z), O, Z)
    full = ((F(0), F(-2)), (T, F(3)))
    proof = certify_source_embedding(cylinder, full, deck_axes=(0,))
    assert proof is not None and proof.family == "revolution-cylinder"
    plane = RevolutionSurface(LineCurve(O, X), O, Z)
    disk = ((F(0), F(0)), (T, F(1)))
    assert certify_source_embedding(plane, disk, deck_axes=(0,)) is None
    polar = certify_source_embedding(plane, disk, deck_axes=(0,), pole_roots=(F(0),))
    assert polar is not None and polar.family == "revolution-plane"
    assert polar.replay(plane, disk, deck_axes=(0,), pole_roots=(F(0),))
    assert (
        certify_source_embedding(
            plane, ((F(0), F(-1)), (T, F(1))), deck_axes=(0,), pole_roots=(F(0),)
        )
        is None
    )
    assert (
        certify_source_embedding(plane, disk, deck_axes=(0,), pole_roots=(F(1),)) is None
    )
    assert (
        certify_source_embedding(plane, ((F(0), F(1)), (T, F(2))), deck_axes=(0,))
        is not None
    )
    assert certify_source_embedding(RevolutionSurface(LineCurve(O, Z), O, Z), BOX) is None
    assert (
        certify_source_embedding(RevolutionSurface(LineCurve(X, (1, 0, 1)), O, Z), BOX)
        is None
    )


def test_sphere_requires_explicit_actual_pole_quotients() -> None:
    sphere = SpherePatch(O, X, Y, Z, 2)
    box = ((F(0), NQ), (T, Q))
    assert certify_source_embedding(sphere, box, deck_axes=(0,)) is None
    proof = certify_source_embedding(sphere, box, deck_axes=(0,), pole_roots=(NQ, Q))
    assert proof is not None and proof.family == "sphere"
    assert proof.replay(sphere, box, deck_axes=(0,), pole_roots=(NQ, Q))
    assert (
        certify_source_embedding(sphere, box, deck_axes=(0,), pole_roots=(Q, NQ)) is None
    )
    rounded = ((F(0), F.from_float(-pi / 2)), (F(1), F.from_float(pi / 2)))
    assert certify_source_embedding(sphere, rounded, pole_roots=(NQ, Q)) is None
    assert (
        certify_source_embedding(sphere, ((F(0), NQ), (F(1), T)), pole_roots=(NQ,))
        is None
    )
    assert certify_source_embedding(SpherePatch(O, X, Y, X, 1), BOX) is None


def test_ring_torus_double_deck_and_strict_radii() -> None:
    torus = TorusPatch(O, X, Y, Z, 3, 1)
    full = ((F(0), F(0)), (T, T))
    proof = certify_source_embedding(torus, full, deck_axes=(0, 1))
    assert proof is not None and proof.family == "ring-torus"
    assert certify_source_embedding(torus, full, deck_axes=(0,)) is None
    assert certify_source_embedding(TorusPatch(O, X, Y, Z, 1, 1), BOX) is None
    assert certify_source_embedding(TorusPatch(O, X, Y, Z, 1, 2), BOX) is None
    assert certify_source_embedding(TorusPatch(O, X, Y, Z, 3, -1), BOX) is None
    assert certify_source_embedding(TorusPatch(O, X, Y, X, 3, 1), BOX) is None
    # The same binary64 shadows must not decide the ring inequality.
    epsilon = F(1, 2**60)
    ring = TorusPatch(O, X, Y, Z, 1, 1, major_radius_exact=F(1) + epsilon)
    spindle = TorusPatch(O, X, Y, Z, 1, 1, major_radius_exact=F(1) - epsilon)
    assert certify_source_embedding(ring, BOX) is not None
    assert certify_source_embedding(spindle, BOX) is None


def test_ring_revolution_uses_exact_meridian_frame_and_both_decks() -> None:
    full = ((F(0), F(0)), (T, T))
    ring = RevolutionSurface(CircleCurve((3, 0, 0), X, Z, 1), O, Z)
    proof = certify_source_embedding(ring, full, deck_axes=(0, 1))
    assert proof is not None and proof.family == "ring-torus"
    assert proof.replay(ring, full, deck_axes=(0, 1))
    assert certify_source_embedding(ring, full, deck_axes=(0,)) is None
    spindle = RevolutionSurface(CircleCurve((1, 0, 0), X, Z, 2), O, Z)
    assert certify_source_embedding(spindle, BOX) is None
    off_plane = RevolutionSurface(CircleCurve((3, 1, 0), X, Z, 1), O, Z)
    assert certify_source_embedding(off_plane, BOX) is None


def test_rejects_inexact_and_forged_chart_inputs() -> None:
    plane = PlanePatch(O, X, Y)
    # ty: ignore[invalid-argument-type]
    assert certify_source_embedding(plane, ((0.0, F(0)), (F(1), F(1)))) is None
    assert certify_source_embedding(plane, ((F(0), F(0)), (F(1), F(0)))) is None
    assert certify_source_embedding(plane, BOX, deck_axes=(False,)) is None
    assert certify_source_embedding(plane, BOX, pole_roots=(Q,)) is None
    inexact_period = PeriodOffset(0.0, 1.0)  # ty: ignore[invalid-argument-type]
    assert certify_source_embedding(plane, ((F(0), F(0)), (inexact_period, F(1)))) is None


def test_iges_seam_roots_and_curved_source_monotonicity(tmp_path: Path) -> None:
    from phydrax.geometry._meshing_domain import (
        _same_brep_trim_endpoint,
        MeshingDomain,
        SourcePreparation,
    )
    from tools.meshing_qualification.scenarios.native_cad_profiles import _quadric_case

    model, _, _, _ = _quadric_case("iges", tmp_path)
    domain = MeshingDomain.from_brep(model)
    assert certify_brep_trim_sheet(None, 0) is None
    assert certify_brep_trim_sheet(domain, -1) is None
    for source_patch in range(3):
        proof = certify_brep_trim_sheet(domain, source_patch)
        assert proof is not None
        assert proof.replay(domain, source_patch)
        assert proof.embedding.deck_axes == ((0,) if source_patch == 2 else ())
        assert not proof.replay(domain, (source_patch + 1) % 3)
    patch = 2
    face = domain.source_indices[2][patch]
    uses = domain.patches[patch].loops[0]
    coedges = model.geometry.face_loops[face][0]
    low = _constant_chart_line(uses[1].pcurve, 0)
    high = _constant_chart_line(uses[3].pcurve, 0)
    assert low is not None and high is not None
    assert low[0] == F(0) and high[0] == T
    assert (
        model.geometry.coedge_edges[coedges[1]] == model.geometry.coedge_edges[coedges[3]]
    )
    assert uses[1].sense == -uses[3].sense
    for index in range(len(uses)):
        following = (index + 1) % len(uses)
        assert _same_brep_trim_endpoint(
            model, face, coedges[index], coedges[following], uses[index], uses[following]
        )
    preparation = SourcePreparation(domain)
    for index in (0, 2):
        for first in range(4):
            lower, _ = preparation.curve_derivative_bounds(
                uses[index].pcurve, float(first), float(first + 1)
            )
            assert lower[0] > 0


def test_repaired_step_original_source_sheets(tmp_path: Path) -> None:
    from phydrax.geometry._meshing_domain import MeshingDomain
    from tools.meshing_qualification.scenarios.native_cad_profiles import _external_case

    model, _, _, _ = _external_case("external-routine", "step", tmp_path)
    domain = MeshingDomain.from_brep(model)
    assert len(domain.patches) == 6
    for patch, family in enumerate(
        ("cylinder", "plane", "cylinder", "plane", "ring-torus", "ring-torus")
    ):
        proof = certify_brep_trim_sheet(domain, patch)
        assert proof is not None
        assert proof.embedding.family == family
        assert proof.replay(domain, patch)
