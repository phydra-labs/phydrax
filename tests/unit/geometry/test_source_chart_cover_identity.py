"""Source chart covers are decided by exact data, never by a surface evaluation.

A cover vertex is the nearest binary64 point of its exact source point or of
the center of a certified enclosure of it, so the cover is the same on every
platform: another platform's XLA surface evaluation, a few ulps away, decides
nothing. The cover identity hashes only exact content; its corners and bounds
are stored evidence, so another platform's rounding of them restores too.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import jax
import mpmath as mp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry._domain_boundary import (
    _enclosure_center,
    _source_point_box,
    MeshingDomainBoundarySource,
)
from phydrax.geometry._meshing_domain import MeshingDomain, SourcePreparation
from phydrax.geometry._source_boundary_records import SourceBoundaryChartCover
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_cone,
    brep_cylinder,
    brep_sphere,
    brep_torus,
)
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import CylinderPatch, SpherePatch, TorusPatch
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native_options import NativeMeshingOptions


_CONTRACT = phx.SpatialCoordinateContract.si()

_MODELS: dict[str, Callable[[], BRepModel]] = {
    "box": lambda: brep_box(
        np.zeros(3), np.asarray((1.0, 0.5, 0.25)), coordinate_contract=_CONTRACT
    ),
    "cylinder": lambda: brep_cylinder(0.5, 1.0, coordinate_contract=_CONTRACT),
    "cone": lambda: brep_cone(0.5, 0.25, 1.0, coordinate_contract=_CONTRACT),
    "sphere": lambda: brep_sphere(0.5, coordinate_contract=_CONTRACT),
    "torus": lambda: brep_torus(1.0, 0.25, coordinate_contract=_CONTRACT),
}


def _other_platform_evaluate(monkeypatch: pytest.MonkeyPatch, /) -> list[int]:
    """Make every XLA surface evaluation land one ulp away, as on another platform."""
    evaluate = MeshingDomain.evaluate
    calls: list[int] = []

    def other_platform(self: MeshingDomain, patches: Any, charts: Any, /) -> np.ndarray:
        values = evaluate(self, patches, charts)
        calls.append(values.shape[0])
        return np.nextafter(values, np.inf)

    monkeypatch.setattr(MeshingDomain, "evaluate", other_platform)
    return calls


def _cover(model: BRepModel, /) -> SourceBoundaryChartCover:
    domain = MeshingDomain.from_brep(model)
    return MeshingDomainBoundarySource(
        domain, tuple(range(len(domain.patches)))
    ).boundary_chart_cover()


@pytest.mark.parametrize("name", tuple(_MODELS))
def test_chart_cover_identity_ignores_other_platform_surface_rounding(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _MODELS[name]()
    here = _cover(model)
    calls = _other_platform_evaluate(monkeypatch)
    there = _cover(model)

    assert not calls
    assert there.cover_id == here.cover_id
    assert np.array_equal(there.simplices, here.simplices)
    assert np.array_equal(there.deviation_bounds, here.deviation_bounds)
    assert here.simplices.shape[0] > 0


def _mp(value: Fraction, /) -> mp.mpf:
    return mp.mpf(value.numerator) / value.denominator


def _cylinder_point(u: mp.mpf, v: mp.mpf, /) -> tuple[mp.mpf, ...]:
    # CylinderPatch((1, 2, 3), (1, 0, 0), (0, 1, 0), (0, 0, 1), 3 / 4)
    radius = mp.mpf(3) / 4
    return (1 + radius * mp.cos(u), 2 + radius * mp.sin(u), 3 + v)


def _sphere_point(u: mp.mpf, v: mp.mpf, /) -> tuple[mp.mpf, ...]:
    # SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 5 / 8)
    radius = mp.mpf(5) / 8
    return (
        radius * mp.cos(v) * mp.cos(u),
        radius * mp.cos(v) * mp.sin(u),
        radius * mp.sin(v),
    )


def _torus_point(u: mp.mpf, v: mp.mpf, /) -> tuple[mp.mpf, ...]:
    # TorusPatch((1, 2, 3), (1, 0, 0), (0, 1, 0), (0, 0, 1), 3, 1)
    ring = 3 + mp.cos(v)
    return (1 + ring * mp.cos(u), 2 + ring * mp.sin(u), 3 + mp.sin(v))


_SURFACES = {
    "cylinder": (
        lambda: CylinderPatch((1, 2, 3), (1, 0, 0), (0, 1, 0), (0, 0, 1), 0.75),
        _cylinder_point,
    ),
    "sphere": (
        lambda: SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 0.625),
        _sphere_point,
    ),
    "torus": (
        lambda: TorusPatch((1, 2, 3), (1, 0, 0), (0, 1, 0), (0, 0, 1), 3, 1),
        _torus_point,
    ),
}


def _near_cardinal_angles() -> tuple[Fraction, ...]:
    """Binary64 angles one ulp around the cardinal angles, where cos or sin vanish."""
    angles: list[Fraction] = [Fraction(0), Fraction(1, 2**40), Fraction(-1, 2**40)]
    for multiple in (0.5, 1.0, 1.5):
        nearest = multiple * np.pi
        for value in (
            np.nextafter(nearest, -np.inf),
            nearest,
            np.nextafter(nearest, np.inf),
        ):
            angles.append(Fraction.from_float(float(value)))
    angles.append(Fraction(3, 7))
    return tuple(angles)


@pytest.mark.parametrize("name", tuple(_SURFACES))
def test_decided_corner_bound_contains_exact_source_point(name: str) -> None:
    """mpmath oracle: the decided corner is within its bound of the exact point."""
    make, exact_point = _SURFACES[name]
    surface = make()
    preparation = SourcePreparation(
        MeshingDomain.from_brep(brep_sphere(0.5, coordinate_contract=_CONTRACT))
    )
    angles = _near_cardinal_angles()
    latitudes = (
        angles
        if name != "sphere"
        else tuple(value for value in angles if abs(value) < Fraction(3, 2))
    )
    with mp.workdps(80):
        for u in angles:
            for v in latitudes:
                center, gap = _enclosure_center(
                    _source_point_box(surface, (u, v), preparation)
                )
                exact = exact_point(_mp(u), _mp(v))
                distance = mp.sqrt(
                    sum(
                        (exact[axis] - mp.mpf(float(center[axis]))) ** 2
                        for axis in range(3)
                    )
                )
                assert distance <= _mp(gap), (u, v)
                # The bound is a few ulps of the point, not a sampled radius.
                assert gap < Fraction(1, 2**40)


def _cad_cylinder_volume() -> CellMeshingResult:
    model = brep_cylinder(
        0.5, 1.0, coordinate_contract=_CONTRACT, source_id="chart-cover-identity"
    )
    boundary = MeshingScope.whole(model, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, 0.3),),
        maximum_deviation=0.05,
    )
    return (
        phx.meshing.NativeMeshingProvider(
            NativeMeshingOptions(
                "cad_tetrahedral",
                volume_schedule=NativeVolumeSchedule(
                    radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
                ),
            )
        )
        .plan(model, specification)
        .execute()
    )


@pytest.fixture(scope="module")
def cad_cylinder_volume() -> CellMeshingResult:
    return _cad_cylinder_volume()


@pytest.mark.meshcore
def test_restore_under_other_platform_surface_rounding_keeps_identities(
    cad_cylinder_volume: CellMeshingResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = cad_cylinder_volume
    assert accepted.certification is not None and accepted.certification.passed
    fidelity = accepted.certification.fidelity
    assert fidelity is not None and fidelity.coverage is not None
    assert fidelity.coverage.complete

    _other_platform_evaluate(monkeypatch)
    restored = CellMeshingResult.from_stored_fields(accepted)

    assert restored.certification is not None
    restored.certification.require_passed()
    restored_fidelity = restored.certification.fidelity
    assert restored_fidelity is not None and restored_fidelity.coverage is not None
    assert (
        restored.result_id,
        restored.certification.report_id,
        restored_fidelity.certificate_id,
        restored_fidelity.coverage.cover_id,
    ) == (
        accepted.result_id,
        accepted.certification.report_id,
        fidelity.certificate_id,
        fidelity.coverage.cover_id,
    )


@pytest.mark.meshcore
@pytest.mark.parametrize("ulps", (1, 7))
def test_restore_keeps_other_platform_rounding_of_stored_chart_cover(
    cad_cylinder_volume: CellMeshingResult, ulps: int
) -> None:
    """A stored cover corner and bound moved by a few ulps are stored evidence."""
    accepted = cad_cylinder_volume

    def nudge(node: Any) -> Any:
        if not isinstance(node, SourceBoundaryChartCover):
            return node
        simplices = np.array(node.simplices, copy=True)
        bounds = np.array(node.deviation_bounds, copy=True)
        for _ in range(ulps):
            simplices[0, 0, 0] = np.nextafter(simplices[0, 0, 0], np.inf)
            bounds[0] = np.nextafter(bounds[0], np.inf)
        return SourceBoundaryChartCover(
            simplices,
            bounds,
            node.semantics,
            node.complete,
            node.source_id,
            node.source_revision,
            findings=node.findings,
            restriction_ids=node.restriction_ids,
        )

    forged = jax.tree_util.tree_map(
        nudge, accepted, is_leaf=lambda node: isinstance(node, SourceBoundaryChartCover)
    )
    assert forged.certification is not None and forged.certification.fidelity is not None
    coverage = forged.certification.fidelity.coverage
    assert coverage is not None
    assert accepted.certification is not None
    assert accepted.certification.fidelity is not None
    assert accepted.certification.fidelity.coverage is not None
    assert coverage.cover_id == accepted.certification.fidelity.coverage.cover_id
    restored = CellMeshingResult.from_stored_fields(forged)
    assert restored.certification is not None
    kept = restored.certification.fidelity
    assert kept is not None and kept.coverage is not None
    assert np.array_equal(kept.coverage.simplices, coverage.simplices)
    assert (restored.result_id, restored.certification.report_id) == (
        accepted.result_id,
        accepted.certification.report_id,
    )
