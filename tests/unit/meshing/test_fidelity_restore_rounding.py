"""Restore adopts stored source-fidelity evidence across platform rounding.

A source-fidelity certificate hashes only exact content. Its certified float
bounds are evidence: restore recomputes the certificate, requires equal exact
content and intersecting enclosures, keeps the bound-limit decisions taken
when the result was generated, and ships the outward hull of the stored and
recomputed enclosures, on which every kept decision must still hold. These
tests stand in for another platform by perturbing the recomputed bounds by a
few ulps.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import numpy as np
import pytest

from phydrax._strict import reconstruct_strict
from phydrax.geometry._embedding_state import MeshCertificateFinding
from phydrax.geometry._mesh_certificates import certificate_status
from phydrax.geometry._source_boundary_records import SourceBoundaryChartCover
from phydrax.geometry._source_fidelity_certificate import (
    _fidelity_certificate_identity,
    SourceFidelityCertificate,
    SourceFidelityReproofError,
)
from phydrax.meshing import _certification
from phydrax.meshing._result import CellMeshingResult
from tests.unit.geometry.test_source_chart_cover_identity import _cad_cylinder_volume


@pytest.fixture(scope="module")
def accepted() -> CellMeshingResult:
    result = _cad_cylinder_volume()
    assert result.certification is not None and result.certification.passed
    fidelity = result.certification.fidelity
    assert fidelity is not None and fidelity.status == "certified"
    assert fidelity.coverage is not None
    return result


def _step(values: np.ndarray, ulps: np.ndarray, /) -> np.ndarray:
    """Move each value ``ulps`` binary64 steps (sign gives direction); keep zero and inf."""
    result = np.array(values, dtype=np.float64, copy=True)
    for _ in range(int(np.max(np.abs(ulps), initial=0))):
        active = (np.abs(ulps) > 0) & np.isfinite(result) & (result != 0.0)
        result = np.where(active, np.nextafter(result, np.sign(ulps) * np.inf), result)
        ulps = ulps - np.sign(ulps)
    return result


def _alternating(shape: tuple[int, ...], ulps: int, /) -> np.ndarray:
    signs = np.where(np.arange(int(np.prod(shape))) % 2 == 0, 1, -1).reshape(shape)
    return signs * ulps


def _with_fields(certificate: SourceFidelityCertificate, /, **changes: Any) -> Any:
    attributes = {
        item.name: getattr(certificate, item.name)
        for item in dataclasses.fields(certificate)
    }
    attributes.update(changes)
    attributes["status"] = certificate_status(attributes["findings"])
    attributes["certificate_id"] = _fidelity_certificate_identity(attributes)
    return reconstruct_strict(SourceFidelityCertificate, attributes)


def _other_platform(certificate: SourceFidelityCertificate, ulps: int, /) -> Any:
    """The same proof as another platform's libm/XLA/BLAS rounding would publish it.

    Lower bounds move down and upper bounds move either way by ``ulps``; an
    upper bound never drops below the stored lower bound, as both platforms'
    sound enclosures contain the same exact value.
    """
    bounds = np.asarray(certificate.deviation_bounds, dtype=np.float64)
    lower = _step(bounds[:, 0], -np.full(bounds.shape[0], ulps))
    upper = _step(bounds[:, 1], _alternating((bounds.shape[0],), ulps))
    upper = np.maximum(upper, bounds[:, 0])
    cover = certificate.coverage
    if cover is not None:
        simplices = np.asarray(cover.simplices, dtype=np.float64)
        cover_bounds = np.asarray(cover.deviation_bounds, dtype=np.float64)
        cover = SourceBoundaryChartCover(
            _step(simplices, _alternating(simplices.shape, ulps)),
            _step(cover_bounds, _alternating(cover_bounds.shape, ulps)),
            cover.semantics,
            cover.complete,
            cover.source_id,
            cover.source_revision,
            findings=cover.findings,
            restriction_ids=cover.restriction_ids,
        )
    normal = certificate.normal_deviation_bounds
    mesh_to_source = max(
        float(_step(np.asarray(certificate.mesh_to_source_upper), np.asarray(-ulps))),
        certificate.mesh_to_source_lower,
    )
    return _with_fields(
        certificate,
        deviation_bounds=np.stack((lower, upper), axis=1),
        mesh_to_source_upper=mesh_to_source,
        source_to_mesh_upper=float(
            _step(np.asarray(certificate.source_to_mesh_upper), np.asarray(ulps))
        ),
        normal_deviation_bounds=None
        if normal is None
        else _step(np.asarray(normal), _alternating(np.shape(normal), ulps)),
        coverage=cover,
    )


def _recompute_as(monkeypatch: pytest.MonkeyPatch, transform: Any, /) -> list[int]:
    """Route restore's fidelity recomputation through ``transform``."""
    original = _certification.certify_source_fidelity
    calls: list[int] = []

    def other_platform(*args: Any, **kwargs: Any) -> SourceFidelityCertificate:
        calls.append(1)
        return transform(original(*args, **kwargs))

    monkeypatch.setattr(_certification, "certify_source_fidelity", other_platform)
    return calls


def _replace_stored(result: CellMeshingResult, transform: Any, /) -> CellMeshingResult:
    return jax.tree_util.tree_map(
        lambda node: (
            transform(node) if isinstance(node, SourceFidelityCertificate) else node
        ),
        result,
        is_leaf=lambda node: isinstance(node, SourceFidelityCertificate),
    )


def _identities(result: CellMeshingResult, /) -> tuple[str, ...]:
    assert result.certification is not None
    fidelity = result.certification.fidelity
    assert fidelity is not None and fidelity.coverage is not None
    return (
        result.result_id,
        result.certification.report_id,
        fidelity.certificate_id,
        fidelity.coverage.cover_id,
    )


@pytest.mark.meshcore
@pytest.mark.parametrize("ulps", (1, 7, 64))
def test_restore_accepts_other_platform_bound_rounding_with_identical_ids(
    accepted: CellMeshingResult, ulps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recompute_as(monkeypatch, lambda value: _other_platform(value, ulps))
    restored = CellMeshingResult.from_stored_fields(accepted)

    assert calls
    assert _identities(restored) == _identities(accepted)
    assert restored.certification is not None and accepted.certification is not None
    stored, kept = accepted.certification.fidelity, restored.certification.fidelity
    assert stored is not None and kept is not None
    # The shipped evidence is the hull of the stored and the recomputed bounds.
    recomputed = np.asarray(
        _other_platform(stored, ulps).deviation_bounds, dtype=np.float64
    )
    stored_bounds = np.asarray(stored.deviation_bounds, dtype=np.float64)
    hull = np.stack(
        (
            np.minimum(stored_bounds[:, 0], recomputed[:, 0]),
            np.maximum(stored_bounds[:, 1], recomputed[:, 1]),
        ),
        axis=1,
    )
    assert np.array_equal(kept.deviation_bounds, hull)
    assert kept.coverage is not None and stored.coverage is not None
    assert np.array_equal(kept.coverage.simplices, stored.coverage.simplices)


@pytest.mark.meshcore
def test_near_threshold_decision_the_reproof_does_not_hold_is_refused(
    accepted: CellMeshingResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1 ulp below tolerance when written, 1 ulp above when recomputed: refused.

    The shipped hull reaches the recomputed upper bound, which does not meet the
    tolerance, so the stored pass is not proved on this platform.
    """
    assert accepted.certification is not None
    fidelity = accepted.certification.fidelity
    assert fidelity is not None
    tolerance = fidelity.tolerance
    row = int(np.argmax(np.asarray(fidelity.deviation_bounds)[:, 1]))
    cell = fidelity.source_cell_global_ids[row]

    # Every stored fidelity proof (the result's and the CAD PLC's) is written
    # near the threshold at this row; restore names whichever it reaches first.
    forged_cells: set[int] = set()

    def written(certificate: SourceFidelityCertificate) -> SourceFidelityCertificate:
        bounds = np.array(certificate.deviation_bounds, copy=True)
        bounds[row, 1] = np.nextafter(tolerance, -np.inf)
        forged = _with_fields(certificate, deviation_bounds=bounds)
        # Bounds are evidence, not identity: the stored claim keeps its ID.
        assert forged.certificate_id == certificate.certificate_id
        forged_cells.add(certificate.source_cell_global_ids[row])
        return forged

    def recomputed(certificate: SourceFidelityCertificate) -> SourceFidelityCertificate:
        bounds = np.array(certificate.deviation_bounds, copy=True)
        bounds[row, 1] = np.nextafter(tolerance, np.inf)
        finding = MeshCertificateFinding(
            "source_fidelity_tolerance",
            "unresolved",
            "cell",
            (cell,),
            detail="The continuous directed deviation upper bound exceeds the requested tolerance.",
        )
        fresh = _with_fields(
            certificate,
            deviation_bounds=bounds,
            findings=(*certificate.findings, finding),
        )
        assert fresh.status != "certified"
        return fresh

    stored = _replace_stored(accepted, written)
    _recompute_as(monkeypatch, recomputed)
    with pytest.raises(SourceFidelityReproofError) as refusal:
        CellMeshingResult.from_stored_fields(stored)
    assert refusal.value.bound == "deviation_bounds"
    assert refusal.value.cell in forged_cells


def _stored_and_fresh(
    accepted: CellMeshingResult, fresh_fraction: float, /
) -> tuple[SourceFidelityCertificate, SourceFidelityCertificate, int]:
    """A stored claim of ``[0, 0]`` at one cell and its reproof.

    A pre-fix chord bound across a C0 knot claimed zero deviation there; the
    reproof encloses the true deviation as ``[0, fresh_fraction * tolerance]``.
    """
    assert accepted.certification is not None
    fidelity = accepted.certification.fidelity
    assert fidelity is not None
    row = int(np.argmax(np.asarray(fidelity.deviation_bounds)[:, 1]))
    fresh_upper = fresh_fraction * fidelity.tolerance
    stored_bounds = np.array(fidelity.deviation_bounds, copy=True)
    stored_bounds[row] = (0.0, 0.0)
    fresh_bounds = np.array(fidelity.deviation_bounds, copy=True)
    fresh_bounds[row] = (0.0, fresh_upper)
    findings = fidelity.findings
    if fresh_upper > fidelity.tolerance:
        findings = (
            *findings,
            MeshCertificateFinding(
                "source_fidelity_tolerance",
                "unresolved",
                "cell",
                (fidelity.source_cell_global_ids[row],),
                detail="The continuous directed deviation upper bound exceeds the requested tolerance.",
            ),
        )
    return (
        _with_fields(fidelity, deviation_bounds=stored_bounds),
        _with_fields(fidelity, deviation_bounds=fresh_bounds, findings=findings),
        row,
    )


@pytest.mark.meshcore
def test_adoption_refuses_a_stored_zero_bound_whose_reproof_exceeds_tolerance(
    accepted: CellMeshingResult,
) -> None:
    stored, fresh, row = _stored_and_fresh(accepted, 2.0)
    assert stored.status == "certified" and fresh.status != "certified"
    with pytest.raises(SourceFidelityReproofError) as refusal:
        stored.adopt_reproof(fresh)
    assert refusal.value.bound == "deviation_bounds"
    assert refusal.value.cell == stored.source_cell_global_ids[row]


@pytest.mark.meshcore
def test_adoption_ships_the_reproved_bound_over_a_stored_zero_bound(
    accepted: CellMeshingResult,
) -> None:
    stored, fresh, row = _stored_and_fresh(accepted, 0.5)
    adopted = stored.adopt_reproof(fresh)
    assert adopted.status == "certified"
    assert adopted.certificate_id == stored.certificate_id
    shipped = np.asarray(adopted.deviation_bounds)[row]
    assert tuple(shipped) == (0.0, 0.5 * stored.tolerance)


@pytest.mark.meshcore
def test_adoption_ships_the_reproved_one_sided_normal_bound(
    accepted: CellMeshingResult,
) -> None:
    assert accepted.certification is not None
    fidelity = accepted.certification.fidelity
    assert fidelity is not None
    cells = len(fidelity.source_cell_global_ids)
    small = np.full((cells,), 0.01, dtype=np.float64)
    reproved = np.array(small, copy=True)
    reproved[cells // 2] = 0.25
    stored = _with_fields(
        fidelity,
        normal_deviation_bounds=small,
        maximum_normal_deviation_lower=0.0,
        maximum_normal_deviation_bound=0.01,
    )
    fresh = _with_fields(
        fidelity,
        normal_deviation_bounds=reproved,
        maximum_normal_deviation_lower=0.0,
        maximum_normal_deviation_bound=0.25,
    )
    adopted = stored.adopt_reproof(fresh)
    assert adopted.normal_deviation_bounds is not None
    assert np.array_equal(adopted.normal_deviation_bounds, reproved)
    assert adopted.maximum_normal_deviation_bound == 0.25


@pytest.mark.meshcore
def test_restore_refuses_a_stored_enclosure_disjoint_from_its_reproof(
    accepted: CellMeshingResult,
) -> None:
    # Every stored fidelity proof (the result's and the CAD PLC's) is forged
    # at its own smallest-bound cell; restore names whichever it reaches first.
    forged_cells: set[int] = set()

    def disjoint(certificate: SourceFidelityCertificate) -> SourceFidelityCertificate:
        forged = np.array(certificate.deviation_bounds, copy=True)
        row = int(np.argmin(forged[:, 1]))
        assert forged[row, 1] < certificate.tolerance
        forged[row] = (certificate.tolerance, certificate.tolerance)
        forged_cells.add(certificate.source_cell_global_ids[row])
        return _with_fields(certificate, deviation_bounds=forged)

    with pytest.raises(SourceFidelityReproofError) as refusal:
        CellMeshingResult.from_stored_fields(_replace_stored(accepted, disjoint))
    assert refusal.value.bound == "deviation_bounds"
    assert refusal.value.cell in forged_cells


@pytest.mark.meshcore
def test_restore_refuses_a_certified_claim_its_stored_bound_does_not_support(
    accepted: CellMeshingResult,
) -> None:
    forged_cells: set[int] = set()

    def overclaimed(certificate: SourceFidelityCertificate) -> SourceFidelityCertificate:
        forged = np.array(certificate.deviation_bounds, copy=True)
        row = int(np.argmax(forged[:, 1]))
        forged[row, 1] = np.nextafter(certificate.tolerance, np.inf)
        forged_cells.add(certificate.source_cell_global_ids[row])
        return _with_fields(certificate, deviation_bounds=forged)

    with pytest.raises(SourceFidelityReproofError) as refusal:
        CellMeshingResult.from_stored_fields(_replace_stored(accepted, overclaimed))
    assert refusal.value.bound == "deviation_bounds"
    assert refusal.value.cell in forged_cells


@pytest.mark.meshcore
def test_restore_refuses_changed_exact_cover_content(accepted: CellMeshingResult) -> None:
    def changed(certificate: SourceFidelityCertificate) -> SourceFidelityCertificate:
        cover = certificate.coverage
        assert cover is not None
        forged_cover = SourceBoundaryChartCover(
            cover.simplices,
            cover.deviation_bounds,
            cover.semantics,
            cover.complete,
            cover.source_id,
            cover.source_revision,
            findings=cover.findings,
            restriction_ids=cover.restriction_ids[:-1],
        )
        assert forged_cover.cover_id != cover.cover_id
        return _with_fields(certificate, coverage=forged_cover)

    with pytest.raises(SourceFidelityReproofError) as refusal:
        CellMeshingResult.from_stored_fields(_replace_stored(accepted, changed))
    assert refusal.value.bound == "cover.restriction_ids"


def test_identity_hashes_decisions_not_bounds(accepted: CellMeshingResult) -> None:
    assert accepted.certification is not None
    fidelity = accepted.certification.fidelity
    assert fidelity is not None
    moved = _other_platform(fidelity, 3)
    assert moved.certificate_id == fidelity.certificate_id
    assert moved.coverage is not None and fidelity.coverage is not None
    assert moved.coverage.cover_id == fidelity.coverage.cover_id
    finding = MeshCertificateFinding("source_fidelity_tolerance", "unresolved", "mesh")
    decided = _with_fields(fidelity, findings=(*fidelity.findings, finding))
    assert decided.certificate_id != fidelity.certificate_id
