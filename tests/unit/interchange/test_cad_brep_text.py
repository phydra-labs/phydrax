#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exercise OCCT BRep text round trips, external references and refusals.

OCCT writes independent source files and reads native writer output.
Volume references are closed forms or OCCT BRepGProp quadrature; carrier
comparisons use explicit binary64 evaluation budgets. Written tolerances
are compared as rationals with certified bounds, not sampled residuals.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._external_resource import ResourceLimits
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._atlas import TrimDomain
from phydrax.geometry.brep import (
    assemble_brep_model,
    brep_box,
    BRepEntityId,
    BRepGapEvidence,
    BRepGeometry,
    BRepModel,
    BRepOccurrence,
    BRepPartitionOperand,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionRole,
    BRepTessellationPolicy,
    certify_brep_gaps,
    HyperbolaCurve,
    OffsetCurve,
    ParabolaCurve,
    partition_brep,
    prepare_brep_query,
)
from phydrax.geometry.brep._intersection_curve import IntersectionCurve
from phydrax.geometry.brep._model import BRepAssemblyContainer
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    BSplineSurfacePatch,
    CylinderPatch,
    LineCurve,
    OffsetSurface,
    PlanePatch,
)
from phydrax.interchange import (
    AdapterStatus,
    CadCurveFitPolicy,
    CadExportPolicy,
    CadExportResult,
    CadImportPolicy,
    CadInterchangeError,
    decode_brep_text_bytes,
    load_brep_archive,
    read_brep_text,
    save_brep_archive,
    write_brep_text,
)
from phydrax.interchange._cad import CadStage, edge_range, recover_pcurve, StagedCoedge
from phydrax.interchange._cad_carriers import place_curve, place_surface, RigidPlacement
from phydrax.units import METER, MILLIMETER, UnitDefinition
from tests._support.cad_models import (
    BREP_TEXT_SOLIDS,
    BREP_TEXT_VOLUMES,
    geometry_of,
    native_brep_text_assembly,
    native_brep_text_sheet,
    native_intersection_cap,
    native_offset_plane,
    native_solid,
    solid_volumes,
)


_CONTRACT = SpatialCoordinateContract(METER)
_LIMITS = ResourceLimits(
    max_bytes=1 << 22,
    max_depth=32,
    max_nodes=20_000,
    max_attributes=2_000_000,
    max_losses=8,
)
_POLICY = CadImportPolicy(_CONTRACT, _LIMITS)
_OCCT_SPLINE_POLICY = CadImportPolicy(
    _CONTRACT,
    _LIMITS,
    tessellation=BRepTessellationPolicy(realize=False),
    relative_geometric_tolerance=2.0e-6,
)

# Fixed query quadrature acceptance budgets against the closed forms above and
# OCCT BRepGProp, retained from these analytic-solid and spline-loft fixtures.
_SOLID_VOLUME_RTOL = 1.0e-6
_LOFT_VOLUME_RTOL = 2.0e-5
_DISK_AREA_RTOL = 2.0e-6
_PLANAR_AREA_RTOL = 1.0e-12
_OFFSET_AREA_RTOL = 1.0e-10
# Binary64 evaluation budgets, in fixture length units: one scaled coordinate
# product, conic/rational evaluations, mapped charts, nested surface lifts and
# two-axis periodic spline expansion respectively. Values preserve prior limits.
_UNIT_PRODUCT_ATOL = 1.0e-15
_CARRIER_ATOL = 1.0e-14
_MAPPED_CHART_ATOL = 1.0e-13
_SURFACE_LIFT_ATOL = 1.0e-12
_PERIODIC_SURFACE_ATOL = 1.0e-11
# OCCT Precision::Confusion in the meter fixtures; fitting policies explicitly
# request the same source deviation budget. The radial oracle allows two such
# errors for the source/fit comparison, not a rewritten carrier tolerance.
_CONSTRUCTION_TOLERANCE = 1.0e-7
_FIT_TOLERANCE = 1.0e-7
_FIT_RADIAL_ATOL = 2.0e-7

_ISOLATED_VERTEX_SOURCE = (
    "DBRep_DrawableShape\n\nCASCADE Topology V3, (c) Open Cascade\n"
    "Locations 0\nCurve2ds 0\nCurves 0\nPolygon3D 0\nPolygonOnTriangulations 0\n"
    "Surfaces 0\nTriangulations 0\nTShapes 1\n"
    "Ve\n0.125\n1 2 3\n0 0\n0101101\n*\n\n+1 0\n"
)


_T = TypeVar("_T")


def _require_type(value: object, expected: type[_T], /) -> _T:
    if not isinstance(value, expected):
        raise TypeError(
            f"_require_type.value must be {expected.__name__}; got {type(value).__name__}"
        )
    return value


def _counts(model: BRepModel) -> tuple[int, int, int, int]:
    topology = model.topology
    return (
        topology.num_solids,
        topology.num_faces,
        topology.num_edges - sum(geometry_of(model).degenerate_edges),
        topology.num_vertices,
    )


def _tshape_declarations(text: str) -> dict[int, tuple[str, int]]:
    """Locate external topology declarations by OCCT's reverse record numbering."""
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("TShapes "))
    records = [
        (line, index + 1)
        for index, line in enumerate(lines)
        if index > start and line in ("Ve", "Ed", "Wi", "Fa", "Sh", "So", "Co")
    ]
    return {len(records) - index: record for index, record in enumerate(records)}


def _written_tolerances(
    text: str, provenance: tuple[tuple[str, str], ...]
) -> dict[str, str]:
    """Read raw decimal tolerance declarations through published entity provenance."""
    lines = text.splitlines()
    declarations = _tshape_declarations(text)
    tolerances: dict[str, str] = {}
    for label, external in provenance:
        if not external.startswith("TShapes#"):
            continue
        kind, _ = label.split(":", 1)
        if kind not in ("vertex", "edge", "face"):
            continue
        number = int(external.split(" ", 1)[0].removeprefix("TShapes#"))
        record_kind, line = declarations[number]
        tokens = lines[line].split()
        tolerances[label] = tokens[1] if record_kind == "Fa" else tokens[0]
    return tolerances


def _require_outward_decimal(decimal: str, bound: float, label: str) -> None:
    # One binary64 successor corrects downward 17-digit rounding; the corrected
    # decimal can lie up to half an ulp above that successor. Two successors
    # therefore enclose the decimal without admitting an inflated format floor.
    written = Fraction(decimal)
    upper = float(np.nextafter(np.nextafter(bound, np.inf), np.inf))
    assert Fraction(bound) <= written <= Fraction(upper), label


@pytest.mark.parametrize("kind", ("vertex", "edge", "face"))
@pytest.mark.parametrize("name", ("box", "cylinder"), ids=("affine", "conic"))
def test_writer_entity_tolerances_enclose_certified_gap_bounds(
    name: str, kind: str, tmp_path: Path
) -> None:
    model = native_solid(name, catalog="brep-text")
    gaps: BRepGapEvidence = certify_brep_gaps(model)
    path = tmp_path / "bounded.brep"
    exported = write_brep_text(model, path)
    written = _written_tolerances(path.read_text(), exported.provenance)
    entities: tuple[BRepEntityId, ...] = {
        "vertex": model.vertex_ids,
        "edge": model.edge_ids,
        "face": model.face_ids,
    }[kind]
    assert entities
    for entity in entities:
        label = f"{entity.kind}:{entity.index}"
        _require_outward_decimal(written[label], gaps.bound(entity), label)
    assert dict(exported.report.target_profile.qualifiers)[
        "entity-tolerance-evidence"
    ] == ("certified-brep-gaps; outward decimal rounding")


@pytest.mark.parametrize(
    "source_unit", (METER, MILLIMETER), ids=("meters", "millimeters")
)
def test_reader_carries_source_tolerances_within_declared_model_limit_losslessly(
    source_unit: UnitDefinition, tmp_path: Path
) -> None:
    path = tmp_path / "bounded-source.brep"
    exported = write_brep_text(native_solid("box", catalog="brep-text"), path)
    text = path.read_text()
    lines = text.splitlines()
    # The source explicitly declares a nanounit tolerance; it exceeds the
    # affine fixture's measured gaps but remains below the imported model limit.
    declared = 1.0e-9
    declarations = _tshape_declarations(text)
    for record_kind, line in declarations.values():
        if record_kind not in ("Ve", "Ed", "Fa"):
            continue
        fields = lines[line].split()
        fields[1 if record_kind == "Fa" else 0] = repr(declared)
        lines[line] = " ".join(fields)
    policy = CadImportPolicy(_CONTRACT, replace(_LIMITS, max_losses=0))
    result = decode_brep_text_bytes(
        ("\n".join(lines) + "\n").encode("ascii"),
        policy,
        source_length_unit=source_unit,
    )
    source = declared * float(source_unit.scale_to_reference)
    gaps = certify_brep_gaps(result.model)
    assert source <= result.model.report.curve_surface_tolerance
    assert source > max(float(value) for value in gaps.upper_bounds)
    assert result.source_tolerance == source
    assert result.source_tolerances == tuple(
        sorted(
            (external, declared)
            for label, external in exported.provenance
            if label.split(":", 1)[0] in ("vertex", "edge", "face")
        )
    )
    assert result.resource_manifest.observed_losses == 0
    assert result.report.valid
    assert result.report.status == AdapterStatus.LOSSLESS
    assert result.report.losses == ()
    evidence = result.tolerance_evidence
    assert evidence is not None
    assert evidence.model_id == result.model.model_id
    assert evidence.declared_limit == result.model.report.curve_surface_tolerance
    entities = (*result.model.vertex_ids, *result.model.edge_ids, *result.model.face_ids)
    assert set(evidence.entities) == set(entities)
    for entity, certified, supplied, carried in zip(
        evidence.entities,
        evidence.certified_bounds,
        evidence.source_bounds,
        evidence.upper_bounds,
        strict=True,
    ):
        assert certified == gaps.bound(entity)
        assert supplied == source
        assert carried == max(gaps.bound(entity), source)


@pytest.mark.parametrize("kind", ("vertex", "edge", "face"))
@pytest.mark.parametrize(
    "source_unit", (METER, MILLIMETER), ids=("meters", "millimeters")
)
@pytest.mark.parametrize("loss_limit", (0, 8), ids=("retained-only", "loss-budget"))
def test_reader_declares_larger_source_entity_tolerances(
    kind: str, source_unit: UnitDefinition, loss_limit: int, tmp_path: Path
) -> None:
    model = native_solid("box", catalog="brep-text")
    path = tmp_path / "source.brep"
    exported = write_brep_text(model, path)
    text = path.read_text()
    external = dict(exported.provenance)[f"{kind}:0"]
    number = int(external.split(" ", 1)[0].removeprefix("TShapes#"))
    record_kind, line = _tshape_declarations(text)[number]
    lines = text.splitlines()
    fields = lines[line].split()
    # A source author explicitly declares two source length units of uncertainty;
    # this is metadata, not an oracle tolerance or a geometry perturbation.
    declared = 2.0
    fields[1 if record_kind == "Fa" else 0] = repr(declared)
    lines[line] = " ".join(fields)
    policy = CadImportPolicy(_CONTRACT, replace(_LIMITS, max_losses=loss_limit))
    damaged = "\n".join(lines) + "\n"
    written = _written_tolerances(damaged, exported.provenance)
    result = decode_brep_text_bytes(
        damaged.encode("ascii"),
        policy,
        source_length_unit=source_unit,
    )
    factor = float(source_unit.scale_to_reference)
    assert result.source_tolerance == declared * factor
    assert result.source_tolerances == tuple(
        sorted(
            (external_reference, float(written[label]))
            for label, external_reference in exported.provenance
            if label in written
        )
    )
    assert result.resource_manifest.observed_losses == 0
    assert result.report.valid
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    loss = next(
        item for item in result.report.losses if item.path == f"source-tolerance:{kind}"
    )
    assert not loss.changes_interpretation
    assert declared * factor > result.model.report.curve_surface_tolerance
    assert f"{kind}:0" in loss.rationale
    assert f"declared={declared * factor:.17g}" in loss.rationale
    assert f"limit={result.model.report.curve_surface_tolerance:.17g}" in loss.rationale
    assert dict(result.provenance)[f"{kind}:0"] == external
    evidence = result.tolerance_evidence
    assert evidence is not None
    assert evidence.declared_limit == result.model.report.curve_surface_tolerance
    entity = BRepEntityId(result.model.source_revision, kind, 0)
    row = evidence.entities.index(entity)
    assert evidence.source_bounds[row] == declared * factor
    certified = evidence.certified_bounds[row]
    assert certified is not None
    assert evidence.upper_bounds[row] == max(certified, declared * factor)


@pytest.mark.parametrize("loss_limit", (0, 8), ids=("refused-budget", "declared-loss"))
def test_reader_declares_unrepresentable_converted_entity_tolerance(
    loss_limit: int, tmp_path: Path
) -> None:
    path = tmp_path / "overflow-source.brep"
    exported = write_brep_text(native_solid("box", catalog="brep-text"), path)
    external = dict(exported.provenance)["vertex:0"]
    number = int(external.split(" ", 1)[0].removeprefix("TShapes#"))
    text = path.read_text()
    _, line = _tshape_declarations(text)[number]
    lines = text.splitlines()
    fields = lines[line].split()
    fields[0] = "1e308"
    lines[line] = " ".join(fields)
    data = ("\n".join(lines) + "\n").encode("ascii")
    policy = CadImportPolicy(
        SpatialCoordinateContract(MILLIMETER),
        replace(_LIMITS, max_losses=loss_limit),
    )
    if loss_limit == 0:
        with pytest.raises(CadInterchangeError) as refused:
            decode_brep_text_bytes(data, policy, source_length_unit=METER)
        assert refused.value.refusal.reason == "limit"
        return
    imported = decode_brep_text_bytes(data, policy, source_length_unit=METER)
    assert imported.report.status == AdapterStatus.DECLARED_LOSS
    assert dict(imported.source_tolerances)[external] == 1.0e308
    assert imported.resource_manifest.observed_losses == 1
    evidence = imported.tolerance_evidence
    assert evidence is not None
    row = evidence.entities.index(imported.model.vertex_ids[0])
    assert evidence.source_bounds[row] is None
    assert evidence.upper_bounds[row] == evidence.certified_bounds[row]
    assert len(evidence.losses) == 1
    loss = evidence.losses[0]
    assert f"entity=vertex:0 ({external})" in loss.rationale
    assert f"declared={1.0e308:.17g}" in loss.rationale
    assert f"source-unit={METER.symbol} ({METER.unit_id})" in loss.rationale
    assert f"limit={evidence.declared_limit:.17g}" in loss.rationale


@pytest.mark.parametrize("kind", ("vertex", "edge", "face"))
def test_reader_refuses_negative_source_tolerance(kind: str, tmp_path: Path) -> None:
    path = tmp_path / "source.brep"
    exported = write_brep_text(native_solid("box", catalog="brep-text"), path)
    text = path.read_text()
    external = dict(exported.provenance)[f"{kind}:0"]
    number = int(external.split(" ", 1)[0].removeprefix("TShapes#"))
    record_kind, line = _tshape_declarations(text)[number]
    lines = text.splitlines()
    fields = lines[line].split()
    fields[1 if record_kind == "Fa" else 0] = "-1"
    lines[line] = " ".join(fields)
    with pytest.raises(CadInterchangeError) as refused:
        _decode("\n".join(lines) + "\n")
    assert refused.value.refusal.reason == "malformed"
    assert refused.value.refusal.entity == external
    assert f"{external}.tolerance" in refused.value.refusal.message
    assert "must be nonnegative; got -1.0" in refused.value.refusal.message
    assert " at byte " in refused.value.refusal.message


def test_writer_refuses_missing_entity_gap_evidence_before_publication(
    tmp_path: Path,
) -> None:
    stage = CadStage(1.0, None)
    stage.vertex(np.asarray((0.0, 0.0, 0.0)), "isolated")
    model = stage.publish(
        coordinate_contract=_CONTRACT,
        source_id="isolated",
        source_format="native",
        source_digest="0" * 64,
        import_policy_id=_POLICY.policy_id,
        tessellation=None,
        default_occurrences=True,
    )
    assert certify_brep_gaps(model).entities == ()
    path = tmp_path / "isolated.brep"
    with pytest.raises(CadInterchangeError) as refused:
        write_brep_text(model, path)
    assert refused.value.refusal.reason == "inexact-export"
    assert refused.value.refusal.entity == "vertex:0"
    assert not path.exists()


def test_reader_carries_unmeasured_vertex_declarations_within_model_limit() -> None:
    policy = CadImportPolicy(_CONTRACT, replace(_LIMITS, max_losses=0))
    # Unit extent gives the model a declared physical limit of 1e-8. A singleton
    # has zero geometric extent and therefore a zero enforced model limit.
    text = (
        "DBRep_DrawableShape\n\nCASCADE Topology V3, (c) Open Cascade\n"
        "Locations 0\nCurve2ds 0\nCurves 0\nPolygon3D 0\nPolygonOnTriangulations 0\n"
        "Surfaces 0\nTriangulations 0\nTShapes 3\n"
        "Ve\n1e-9\n1 2 3\n0 0\n0101101\n*\n"
        "Ve\n1e-9\n2 2 3\n0 0\n0101101\n*\n"
        "Co\n\n1100000\n+3 0 +2 0 *\n\n+1 0\n"
    )
    result = decode_brep_text_bytes(
        text.encode("ascii"), policy, source_length_unit=METER
    )
    assert result.model.topology.num_vertices == 2
    np.testing.assert_array_equal(
        geometry_of(result.model).vertex_points, ((1, 2, 3), (2, 2, 3))
    )
    assert result.source_tolerances == (
        ("TShapes#2 Ve", 1.0e-9),
        ("TShapes#3 Ve", 1.0e-9),
    )
    assert result.source_tolerance == 1.0e-9
    assert result.source_tolerance is not None
    assert result.source_tolerance <= result.model.report.curve_surface_tolerance
    assert certify_brep_gaps(result.model).entities == ()
    assert result.resource_manifest.observed_losses == 0
    assert result.report.valid
    assert result.report.status == AdapterStatus.LOSSLESS
    assert result.report.losses == ()
    evidence = result.tolerance_evidence
    assert evidence is not None
    assert evidence.entities == result.model.vertex_ids
    assert evidence.certified_bounds == (None, None)
    assert evidence.source_bounds == (1.0e-9, 1.0e-9)
    assert evidence.upper_bounds == (1.0e-9, 1.0e-9)


def test_reader_preserves_isolated_vertex_with_above_limit_tolerance_loss() -> None:
    text = _ISOLATED_VERTEX_SOURCE
    result = decode_brep_text_bytes(
        text.encode("ascii"), _POLICY, source_length_unit=METER
    )
    assert result.model.topology.num_vertices == 1
    np.testing.assert_array_equal(geometry_of(result.model).vertex_points, ((1, 2, 3),))
    assert result.source_tolerance == 0.125
    assert result.report.valid
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    loss = next(
        item for item in result.report.losses if item.path == "source-tolerance:vertex"
    )
    assert result.source_tolerance is not None
    assert result.source_tolerance > result.model.report.curve_surface_tolerance
    assert "vertex:0" in loss.rationale
    assert "declared=0.125" in loss.rationale
    assert f"limit={result.model.report.curve_surface_tolerance:.17g}" in loss.rationale


@pytest.mark.parametrize(
    "source_unit", (METER, MILLIMETER), ids=("meters", "millimeters")
)
def test_reader_retains_unused_source_tolerances_without_semantic_loss_charge(
    source_unit: UnitDefinition,
) -> None:
    policy = CadImportPolicy(_CONTRACT, replace(_LIMITS, max_losses=0))
    text = _splice_once(
        _ISOLATED_VERTEX_SOURCE,
        "TShapes 1\n",
        "TShapes 2\nVe\n0.25\n7 8 9\n0 0\n0101101\n*\n",
    )
    result = decode_brep_text_bytes(
        text.encode("ascii"), policy, source_length_unit=source_unit
    )
    factor = float(source_unit.scale_to_reference)
    assert result.model.topology.num_vertices == 1
    np.testing.assert_array_equal(
        geometry_of(result.model).vertex_points,
        factor * np.asarray(((1.0, 2.0, 3.0),), dtype=np.float64),
    )
    assert result.source_tolerances == (
        ("TShapes#1 Ve", 0.125),
        ("TShapes#2 Ve", 0.25),
    )
    assert result.source_tolerance == 0.25 * factor
    assert result.resource_manifest.observed_losses == 0
    assert result.report.valid
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    assert tuple(loss.path for loss in result.report.losses) == (
        "source-tolerance:vertex",
    )


def test_reader_refuses_discarded_derived_fields_without_semantic_loss_budget() -> None:
    policy = CadImportPolicy(_CONTRACT, replace(_LIMITS, max_losses=0))
    text = _splice_once(
        _ISOLATED_VERTEX_SOURCE, "Polygon3D 0\n", "Polygon3D 1\n1 0 0.125\n0 0 0\n"
    )
    with pytest.raises(CadInterchangeError) as refused:
        decode_brep_text_bytes(text.encode("ascii"), policy, source_length_unit=METER)
    assert refused.value.refusal.reason == "limit"
    assert refused.value.refusal.entity == "resource"
    assert "semantic-loss count" in refused.value.refusal.message


def test_reader_charges_discarded_fields_but_retains_tolerance_diagnostics() -> None:
    policy = CadImportPolicy(_CONTRACT, replace(_LIMITS, max_losses=1))
    text = _splice_once(
        _ISOLATED_VERTEX_SOURCE, "Polygon3D 0\n", "Polygon3D 1\n1 0 0.125\n0 0 0\n"
    )
    result = decode_brep_text_bytes(
        text.encode("ascii"), policy, source_length_unit=METER
    )
    assert result.source_tolerances == (("TShapes#1 Ve", 0.125),)
    assert result.resource_manifest.observed_losses == 1
    assert {loss.path for loss in result.report.losses} == {
        "Polygon3D",
        "source-tolerance:vertex",
    }
    assert result.report.valid
    assert result.report.status == AdapterStatus.DECLARED_LOSS


@pytest.mark.parametrize("name", sorted(BREP_TEXT_SOLIDS))
def test_native_models_round_trip_with_exact_identity(name: str, tmp_path: Path) -> None:
    model = native_solid(name, catalog="brep-text")
    write_brep_text(model, tmp_path / "model.brep")
    result = read_brep_text(
        tmp_path / "model.brep", _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert geometry_of(result.model).geometry_id == geometry_of(model).geometry_id
    assert np.array_equal(result.model.orientation, model.orientation)
    assert result.coverage.pcurves_honored == len(geometry_of(model).pcurves)
    assert result.coverage.pcurves_fitted == 0
    assert result.report.valid
    assert result.report.status == AdapterStatus.LOSSLESS
    assert float(solid_volumes(result.model)[0]) == pytest.approx(
        BREP_TEXT_VOLUMES[name], rel=_SOLID_VOLUME_RTOL
    )


def test_writer_output_is_deterministic(tmp_path: Path) -> None:
    model = native_solid("torus", catalog="brep-text")
    first = write_brep_text(model, tmp_path / "a.brep")
    second = write_brep_text(model, tmp_path / "b.brep")
    assert (tmp_path / "a.brep").read_bytes() == (tmp_path / "b.brep").read_bytes()
    assert first.receipt.content_sha256 == second.receipt.content_sha256


@pytest.mark.parametrize("version", ("V1", "V2", "V3"), ids=("V1", "V2", "V3"))
def test_versions_share_one_model(
    version: Literal["V1", "V2", "V3"], tmp_path: Path
) -> None:
    model = native_solid("cylinder", catalog="brep-text")
    path = tmp_path / f"{version}.brep"
    write_brep_text(model, path, version=version)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert result.schema == f"cascade-topology-{version}"
    assert geometry_of(result.model).geometry_id == geometry_of(model).geometry_id


def test_source_length_unit_is_converted_exactly(tmp_path: Path) -> None:
    model = native_solid("box", catalog="brep-text")
    write_brep_text(model, tmp_path / "box.brep")
    result = read_brep_text(
        tmp_path / "box.brep",
        _POLICY,
        trusted_root=tmp_path,
        source_length_unit=MILLIMETER,
    )
    assert result.source_length_unit_meters == pytest.approx(1.0e-3)
    upper = np.max(np.asarray(geometry_of(result.model).vertex_points), axis=0)
    assert np.allclose(upper, (1.0e-3, 2.0e-3, 3.0e-3), rtol=0.0, atol=_UNIT_PRODUCT_ATOL)
    assert float(solid_volumes(result.model)[0]) == pytest.approx(
        6.0e-9, rel=_SOLID_VOLUME_RTOL
    )
    # Unit changes preserve supplied p-curves rather than refitting their shape.
    for coedge, pcurve in enumerate(geometry_of(result.model).pcurves):
        edge = geometry_of(result.model).coedge_edges[coedge]
        first, last = np.asarray(geometry_of(result.model).edge_ranges[edge])
        parameters = np.linspace(float(first), float(last), 7)
        face = next(
            face
            for face, loops in enumerate(geometry_of(result.model).face_loops)
            if any(coedge in loop for loop in loops)
        )
        on_surface = np.asarray(
            result.model.patches[face].evaluate(pcurve.evaluate(jnp.asarray(parameters)))
        )
        carrier = geometry_of(result.model).curves[
            geometry_of(result.model).edge_curves[edge]
        ]
        np.testing.assert_allclose(
            np.asarray(on_surface),
            np.asarray(carrier.evaluate(jnp.asarray(parameters))),
            atol=_SURFACE_LIFT_ATOL,
        )


def test_placed_occurrences_round_trip(tmp_path: Path) -> None:
    ring = native_solid("revolved-ring", catalog="brep-text")
    geometry = geometry_of(ring)
    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    placed = BRepOccurrence(
        ("solid0", "instance1"),
        0,
        np.asarray(rotation),
        np.asarray((5.0, 0.0, 0.0)),
    )
    first = BRepOccurrence(("solid0", "instance0"), 0)
    rebuilt = BRepGeometry(
        vertex_points=geometry.vertex_points,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        occurrences=(first, placed),
        assembly_containers=(
            BRepAssemblyContainer(("model",), (first.path, placed.path)),
        ),
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
    )
    model = assemble_brep_model(
        rebuilt,
        ring.patches,
        ring.parameter_bounds,
        ring.orientation,
        ring.physical_tags,
        coordinate_contract=_CONTRACT,
        source_id="placed-ring",
        source_format="native",
        source_digest=ring.source_digest,
        import_policy_id=ring.import_policy_id,
    )
    write_brep_text(model, tmp_path / "assembly.brep")
    result = read_brep_text(
        tmp_path / "assembly.brep",
        _POLICY,
        trusted_root=tmp_path,
        source_length_unit=METER,
    )
    assert geometry_of(result.model).occurrences == (first, placed)
    assert geometry_of(result.model).assembly_containers == (
        BRepAssemblyContainer(("model",), (first.path, placed.path)),
    )
    assert geometry_of(result.model).geometry_id == rebuilt.geometry_id


# ------------------------------------------------------------ malformed input


def _splice_once(text: str, needle: str, replacement: str) -> str:
    """Refuse fixture drift before replacing one format declaration."""
    assert text.count(needle) == 1, needle
    return text.replace(needle, replacement, 1)


@pytest.mark.parametrize(
    "text", ("absent", "needle needle"), ids=("missing", "ambiguous")
)
def test_text_splice_refuses_missing_or_ambiguous_declaration(text: str) -> None:
    with pytest.raises(AssertionError, match="needle"):
        _splice_once(text, "needle", "replacement")


def _valid_text(tmp_path: Path) -> str:
    write_brep_text(native_solid("box", catalog="brep-text"), tmp_path / "box.brep")
    return (tmp_path / "box.brep").read_text()


def _decode(text: str) -> None:
    decode_brep_text_bytes(text.encode("ascii"), _POLICY, source_length_unit=METER)


def test_unknown_header_is_refused() -> None:
    with pytest.raises(CadInterchangeError) as refused:
        _decode("DBRep_DrawableShape\n\nCASCADE Topology V9, (c) Nobody\n")
    assert refused.value.refusal.reason == "malformed"


def test_truncated_file_is_refused(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    with pytest.raises(CadInterchangeError) as refused:
        _decode(text[: len(text) // 2])
    assert refused.value.refusal.reason == "malformed"


def test_dangling_shape_reference_is_refused(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    head, marker, tail = text.rpartition("+1 0")
    assert marker and tail == "\n"
    with pytest.raises(CadInterchangeError) as refused:
        _decode(head + "+999 0\n")
    assert refused.value.refusal.reason == "dangling-reference"


def test_forward_shape_reference_is_refused_as_cycle() -> None:
    text = (
        "DBRep_DrawableShape\n\nCASCADE Topology V3, (c) Open Cascade\n"
        "Locations 0\nCurve2ds 0\nCurves 0\nPolygon3D 0\nPolygonOnTriangulations 0\n"
        "Surfaces 0\nTriangulations 0\nTShapes 1\nCo\n\n1100000\n+1 0 *\n\n+1 0\n"
    )
    with pytest.raises(CadInterchangeError) as refused:
        _decode(text)
    assert refused.value.refusal.reason == "cyclic-reference"


def test_entity_limit_is_enforced(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    tight = CadImportPolicy(
        _CONTRACT,
        ResourceLimits(
            max_bytes=1 << 22,
            max_depth=32,
            max_nodes=20,
            max_attributes=2_000_000,
            max_losses=8,
        ),
    )
    with pytest.raises(CadInterchangeError) as refused:
        decode_brep_text_bytes(text.encode("ascii"), tight, source_length_unit=METER)
    assert refused.value.refusal.reason == "limit"


def test_nested_offset_operation_refuses_unknown_basis_record(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    head, marker, tail = text.partition("Surfaces 6\n")
    assert text.count("Surfaces 6\n") == 1 and marker
    surface, separator, rest = tail.partition("\n")
    assert surface and separator
    with pytest.raises(CadInterchangeError) as refused:
        _decode(head + marker + "11 0.25\n6 0 0 1\n10\n" + rest)
    assert refused.value.refusal.reason == "malformed"
    assert "curve_kind" in refused.value.refusal.message
    assert "got 10" in refused.value.refusal.message


def test_unreferenced_exact_parabola_is_admitted(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    text = text.replace("Curves 12\n", "Curves 13\n", 1).replace(
        "Polygon3D 0", "4 0 0 0 0 0 1 1 0 0 0 1 0 2.5\nPolygon3D 0", 1
    )
    result = decode_brep_text_bytes(
        text.encode("ascii"), _POLICY, source_length_unit=METER
    )
    assert result.model.topology.num_faces == 6


# ------------------------------------------------------- OCCT as a reference


def _occt_shapes() -> dict[str, object]:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeRevol,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeTorus,
    )
    from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt

    plate = BRepPrimAPI_MakeBox(gp_Pnt(-2, -2, 0), 4.0, 4.0, 0.5).Shape()
    hole = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), 0.5, 3.0
    ).Shape()
    rectangle = BRepBuilderAPI_MakePolygon(
        gp_Pnt(1, 0, 0), gp_Pnt(2, 0, 0), gp_Pnt(2, 0, 1), gp_Pnt(1, 0, 1), True
    ).Wire()
    return {
        "box": BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 2.0, 3.0).Shape(),
        "cylinder": BRepPrimAPI_MakeCylinder(0.5, 2.0).Shape(),
        "sphere": BRepPrimAPI_MakeSphere(0.75).Shape(),
        "torus": BRepPrimAPI_MakeTorus(2.0, 0.5).Shape(),
        "plate-with-hole": BRepAlgoAPI_Cut(plate, hole).Shape(),
        "revolved-ring": BRepPrimAPI_MakeRevol(
            BRepBuilderAPI_MakeFace(rectangle).Face(),
            gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
        ).Shape(),
    }


def _occt_counts(shape: object) -> tuple[int, int, int, int]:
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    counts = []
    for kind in (TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX):
        explorer = TopExp_Explorer(shape, kind)
        shapes = []
        while explorer.More():
            current = explorer.Current()
            if not any(current.IsSame(previous) for previous in shapes):
                shapes.append(current)
            explorer.Next()
        counts.append(
            sum(
                kind != TopAbs_EDGE or not BRep_Tool.Degenerated_s(TopoDS.Edge(current))
                for current in shapes
            )
        )
    return counts[0], counts[1], counts[2], counts[3]


def _occt_volume(shape: object) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return properties.Mass()


@pytest.mark.parametrize("name", sorted(BREP_TEXT_SOLIDS))
def test_occt_written_files_match_occt_topology_and_volume(
    name: str, tmp_path: Path
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools

    shape = _occt_shapes()[name]
    # Derived triangulations are written too; the reader must drop them explicitly.
    BRepMesh_IncrementalMesh(shape, 0.2)
    BRepTools.Write_s(shape, str(tmp_path / "occt.brep"))
    result = read_brep_text(
        tmp_path / "occt.brep", _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _counts(result.model) == _occt_counts(shape)
    assert float(solid_volumes(result.model)[0]) == pytest.approx(
        _occt_volume(shape), rel=_SOLID_VOLUME_RTOL
    )
    assert result.coverage.pcurves_fitted == 0
    assert "Triangulations" in {loss.path for loss in result.report.losses}


@pytest.mark.parametrize("name", sorted(BREP_TEXT_SOLIDS))
def test_occt_reads_native_files_as_valid_solids(name: str, tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape

    model = native_solid(name, catalog="brep-text")
    write_brep_text(model, tmp_path / "native.brep")
    shape = TopoDS_Shape()
    assert BRepTools.Read_s(shape, str(tmp_path / "native.brep"), BRep_Builder())
    assert BRepCheck_Analyzer(shape).IsValid()
    assert _occt_counts(shape) == _counts(model)
    assert _occt_volume(shape) == pytest.approx(
        BREP_TEXT_VOLUMES[name], rel=_SOLID_VOLUME_RTOL
    )


def test_occt_located_assembly_becomes_occurrences(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Compound

    unit = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 2.0, 3.0).Shape()
    turn = gp_Trsf()
    turn.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.5)
    shift = gp_Trsf()
    shift.SetTranslation(gp_Vec(0, 0, 7))
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, unit)
    builder.Add(compound, unit.Moved(TopLoc_Location(shift * turn)))
    BRepTools.Write_s(compound, str(tmp_path / "assembly.brep"))
    result = read_brep_text(
        tmp_path / "assembly.brep",
        _POLICY,
        trusted_root=tmp_path,
        source_length_unit=METER,
    )
    occurrences = geometry_of(result.model).occurrences
    assert [occurrence.path for occurrence in occurrences] == [
        ("solid0", "instance0"),
        ("solid0", "instance1"),
    ]
    assert result.model.topology.num_solids == 1
    expected = np.asarray(
        (
            (np.cos(0.5), -np.sin(0.5), 0.0),
            (np.sin(0.5), np.cos(0.5), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    assert np.allclose(occurrences[1].rotation, expected, atol=_CARRIER_ATOL)
    assert np.allclose(occurrences[1].translation, (0.0, 0.0, 7.0), atol=_CARRIER_ATOL)
    assert geometry_of(result.model).assembly_containers == (
        BRepAssemblyContainer(
            ("model",), tuple(occurrence.path for occurrence in occurrences)
        ),
    )


@pytest.mark.parametrize("source_unit", (METER, MILLIMETER))
def test_rational_spline_sheet_preserves_carriers_and_pcurves(
    source_unit: UnitDefinition, tmp_path: Path
) -> None:
    model = native_brep_text_sheet(
        _POLICY,
    )
    path = tmp_path / "rational.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=source_unit
    )
    factor = 1.0 if source_unit is METER else 1.0e-3
    uv = np.asarray(((0.2, 0.7), (0.9, 0.1), (0.5, 0.5)))
    np.testing.assert_allclose(
        np.asarray(result.model.patches[0].evaluate(jnp.asarray(uv))),
        np.asarray(factor * model.patches[0].evaluate(jnp.asarray(uv))),
        atol=_CARRIER_ATOL,
    )
    np.testing.assert_array_equal(
        np.asarray(_require_type(result.model.patches[0], BSplineSurfacePatch).weights),
        np.asarray(_require_type(model.patches[0], BSplineSurfacePatch).weights),
    )
    assert result.model.topology.num_faces == 1 and result.model.topology.num_edges == 4
    for actual, expected in zip(
        geometry_of(result.model).pcurves, geometry_of(model).pcurves, strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(actual.evaluate(jnp.asarray((0.0, 0.3, 1.0)))),
            np.asarray(expected.evaluate(jnp.asarray((0.0, 0.3, 1.0)))),
        )


def test_inconsistent_supplied_pcurve_is_not_silently_recovered(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    head, marker, tail = text.partition("Curve2ds ")
    count, separator, rest = tail.partition("\n")
    first, _, remaining = rest.partition("\n")
    fields = first.split()
    assert marker and separator and first
    assert fields[0] in ("1", "7")
    coordinate = 6 if fields[0] == "7" else 1
    fields[coordinate] = str(float(fields[coordinate]) + 10.0)
    damaged = head + marker + count + separator + " ".join(fields) + "\n" + remaining
    with pytest.raises(CadInterchangeError) as refused:
        _decode(damaged)
    assert refused.value.refusal.reason == "inconsistent-geometry"
    assert refused.value.refusal.chain


def test_sample_invisible_curve_is_not_promoted_to_exact_pcurve() -> None:
    patch = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    parameters = np.asarray((0.0, 0.010, 0.011, 0.012, 1.0))
    controls = np.stack(
        (np.asarray((1.0, 1.0, 2.0, 1.0, 1.0)), np.zeros(5), parameters), axis=1
    )
    curve = BSplineCurve(controls, np.ones(5), np.r_[0.0, parameters, 1.0], 1)
    with pytest.raises(ValueError, match=r"recover_pcurve\.policy"):
        recover_pcurve(patch, curve, 0.0, 1.0, 1.0, None)


def test_distinct_nearby_locations_never_merge() -> None:
    first = RigidPlacement.identity()
    second = RigidPlacement(np.eye(3), np.asarray((1.0e-13, 0.0, 0.0)))
    assert first.key() != second.key()
    assert first.compose(second).translation[0] == 1.0e-13


def test_topological_reference_depth_is_bounded(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    tight = CadImportPolicy(
        _CONTRACT,
        ResourceLimits(
            max_bytes=1 << 22,
            max_depth=3,
            max_nodes=20_000,
            max_attributes=2_000_000,
            max_losses=8,
        ),
    )
    with pytest.raises(CadInterchangeError) as refused:
        decode_brep_text_bytes(text.encode("ascii"), tight, source_length_unit=METER)
    assert refused.value.refusal.reason == "limit"


def test_expanded_knot_multiplicities_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "rational.brep"
    write_brep_text(
        native_brep_text_sheet(
            _POLICY,
        ),
        path,
    )
    head, marker, tail = path.read_text().partition("Surfaces 1\n")
    surface, separator, rest = tail.partition("\nTriangulations ")
    needle = "0.0 2 1.0 2"
    assert marker and separator and surface.count(needle) >= 1
    damaged = surface.replace(needle, "0.0 2000000000 1.0 2", 1)
    text = head + marker + damaged + separator + rest
    with pytest.raises(CadInterchangeError) as refused:
        _decode(text)
    assert refused.value.refusal.reason == "limit"


def test_occt_produced_rational_surface_has_exact_native_pcurves(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepTools import BRepTools
    from OCP.collections import (
        Array1_double,
        Array1_int,
        Array2_double,
        Array2_gp_Pnt,
    )
    from OCP.Geom import Geom_BSplineSurface
    from OCP.gp import gp_Pnt

    native = native_brep_text_sheet(
        _POLICY,
    )
    patch = _require_type(native.patches[0], BSplineSurfacePatch)
    poles = Array2_gp_Pnt(1, 2, 1, 2)
    weights = Array2_double(1, 2, 1, 2)
    knots = Array1_double(1, 2)
    multiplicities = Array1_int(1, 2)
    for i in range(2):
        knots.SetValue(i + 1, float(i))
        multiplicities.SetValue(i + 1, 2)
        for j in range(2):
            poles.SetValue(
                i + 1, j + 1, gp_Pnt(*map(float, np.asarray(patch.control_points)[i, j]))
            )
            weights.SetValue(i + 1, j + 1, float(patch.weights[i, j]))
    surface = Geom_BSplineSurface(
        poles, weights, knots, knots, multiplicities, multiplicities, 1, 1
    )
    face = BRepBuilderAPI_MakeFace(surface, _CONSTRUCTION_TOLERANCE).Face()
    path = tmp_path / "occt-rational.brep"
    BRepTools.Write_s(face, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    uv = np.asarray(((0.1, 0.4), (0.8, 0.7), (0.5, 0.5)))
    np.testing.assert_allclose(
        np.asarray(result.model.patches[0].evaluate(jnp.asarray(uv))),
        np.asarray(patch.evaluate(jnp.asarray(uv))),
        atol=_MAPPED_CHART_ATOL,
    )
    assert result.model.topology.num_edges == 4
    for coedge, pcurve in enumerate(geometry_of(result.model).pcurves):
        edge = geometry_of(result.model).coedge_edges[coedge]
        first, last = np.asarray(geometry_of(result.model).edge_ranges[edge])
        parameters = np.linspace(float(first), float(last), 11)
        carrier = geometry_of(result.model).curves[
            geometry_of(result.model).edge_curves[edge]
        ]
        np.testing.assert_allclose(
            np.asarray(
                result.model.patches[0].evaluate(pcurve.evaluate(jnp.asarray(parameters)))
            ),
            np.asarray(carrier.evaluate(jnp.asarray(parameters))),
            atol=_SURFACE_LIFT_ATOL,
        )


@pytest.mark.parametrize("name", ("sphere", "torus", "revolved-ring"))
def test_unit_conversion_keeps_seams_and_pole_edges(name: str, tmp_path: Path) -> None:
    model = native_solid(name, catalog="brep-text")
    path = tmp_path / f"{name}.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=MILLIMETER
    )
    assert geometry_of(result.model).edge_curves == geometry_of(model).edge_curves
    assert geometry_of(result.model).edge_vertices == geometry_of(model).edge_vertices
    assert geometry_of(result.model).face_loops == geometry_of(model).face_loops
    np.testing.assert_allclose(
        np.asarray(geometry_of(result.model).vertex_points),
        np.asarray(1.0e-3 * np.asarray(geometry_of(model).vertex_points)),
        atol=_CARRIER_ATOL,
    )
    assert float(solid_volumes(result.model)[0]) == pytest.approx(
        BREP_TEXT_VOLUMES[name] * 1.0e-9, rel=_SOLID_VOLUME_RTOL
    )


def test_rectangular_surface_trim_is_not_discarded(tmp_path: Path) -> None:
    path = tmp_path / "sheet.brep"
    write_brep_text(
        native_brep_text_sheet(
            _POLICY,
        ),
        path,
    )
    text = path.read_text()
    admitted = _splice_once(text, "Surfaces 1\n", "Surfaces 1\n10 0 1 0 1\n")
    result = decode_brep_text_bytes(
        admitted.encode("ascii"), _POLICY, source_length_unit=METER
    )
    assert result.model.topology.num_faces == 1
    clipped = _splice_once(text, "Surfaces 1\n", "Surfaces 1\n10 0 0.5 0 1\n")
    with pytest.raises(CadInterchangeError) as refused:
        _decode(clipped)
    assert refused.value.refusal.reason == "inconsistent-geometry"
    assert refused.value.refusal.chain[-1].endswith("Fa")


def test_explicit_pcurve_approximation_reports_observed_error() -> None:
    controls = np.asarray(
        (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((1.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    knots = np.asarray((0.0, 0.0, 1.0, 1.0))
    patch = BSplineSurfacePatch(controls, np.ones((2, 2)), knots, knots, 1, 1)
    curve = BSplineCurve.bezier(
        np.asarray(((0, 0, 0), (0.5, 0.5, 0), (1, 1, 1))),
        np.ones(3),
    )
    with pytest.raises(ValueError, match=r"recover_pcurve\.policy"):
        recover_pcurve(patch, curve, 0.0, 1.0, 1.0, None)
    fitted = recover_pcurve(
        patch, curve, 0.0, 1.0, 1.0, CadCurveFitPolicy(tolerance=_FIT_TOLERANCE)
    )
    assert fitted.kind == "fitted" and fitted.error <= _FIT_TOLERANCE
    parameters = np.asarray((0.013, 0.273, 0.817))
    np.testing.assert_allclose(
        np.asarray(patch.evaluate(fitted.curve.evaluate(jnp.asarray(parameters)))),
        np.asarray(curve.evaluate(jnp.asarray(parameters))),
        atol=_FIT_TOLERANCE,
    )


def test_shared_face_shell_incidences_preserve_opposite_senses(tmp_path: Path) -> None:
    operands = tuple(
        BRepPartitionOperand(
            name,
            brep_box(lower, upper, coordinate_contract=_CONTRACT),
            BRepPartitionRole.REGION,
        )
        for name, lower, upper in (
            ("left", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ("right", (1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
        )
    )
    plan = BRepPartitionPlan(
        _CONTRACT,
        operands,
        BRepPartitionPolicy(("left", "right")),
    )
    model = partition_brep(plan, destination=tmp_path / "shared.phx").model
    geometry = geometry_of(model)
    shared = [
        face for face, solids in enumerate(model.topology.face_solids) if len(solids) == 2
    ]
    assert len(shared) == 1
    face = shared[0]
    signs = tuple(
        geometry.shell_orientations[shell][geometry.shell_faces[shell].index(face)]
        for shell, faces in enumerate(geometry.shell_faces)
        if face in faces
    )
    assert tuple(sorted(signs)) == (-1, 1)
    path = tmp_path / "shared.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert geometry_of(result.model).geometry_id == geometry.geometry_id


@pytest.mark.parametrize(
    "policy", (CadExportPolicy(maximum_bytes=16), CadExportPolicy(maximum_entities=12))
)
def test_export_limits_refuse_before_publication(
    policy: CadExportPolicy, tmp_path: Path
) -> None:
    path = tmp_path / "limited.brep"
    with pytest.raises(CadInterchangeError) as refused:
        write_brep_text(
            native_brep_text_sheet(
                _POLICY,
            ),
            path,
            policy=policy,
        )
    assert refused.value.refusal.reason == "limit"
    assert not path.exists()


def test_exact_empty_compound_round_trip_has_no_fake_occurrences(tmp_path: Path) -> None:
    stage = CadStage(1.0, None)
    model = stage.publish(
        coordinate_contract=_CONTRACT,
        source_id="empty",
        source_format="native",
        source_digest="0" * 64,
        import_policy_id=_POLICY.policy_id,
        tessellation=None,
        default_occurrences=True,
    )
    path = tmp_path / "empty.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert geometry_of(result.model).geometry_id == geometry_of(model).geometry_id
    assert _counts(result.model) == (0, 0, 0, 0)
    assert geometry_of(result.model).occurrences == ()
    assert np.asarray(result.model.parameter_bounds).shape == (0, 2, 2)


def test_export_honors_declared_trim_closure(tmp_path: Path) -> None:
    model = native_brep_text_sheet(
        _POLICY,
    )
    exact = eqx.tree_at(
        lambda value: value.trim_domains,
        model,
        (TrimDomain(np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)))),),
    )
    path = tmp_path / "polygon.brep"
    write_brep_text(exact, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert geometry_of(result.model).geometry_id == geometry_of(model).geometry_id
    different = eqx.tree_at(
        lambda value: value.trim_domains,
        model,
        (TrimDomain(np.asarray(((0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)))),),
    )
    with pytest.raises(CadInterchangeError) as refused:
        write_brep_text(different, tmp_path / "different.brep")
    assert refused.value.refusal.reason == "inexact-export"
    assert refused.value.refusal.chain == ("face:0", "loop:0")


def test_free_edge_and_face_round_trip_without_dropping_geometry(tmp_path: Path) -> None:
    model = native_brep_text_sheet(_POLICY, free_edge=True)
    path = tmp_path / "face-and-edge.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert geometry_of(result.model).geometry_id == geometry_of(model).geometry_id
    assert result.model.topology.num_faces == 1 and result.model.topology.num_edges == 5
    free = [
        index for index, faces in enumerate(result.model.topology.edge_faces) if not faces
    ]
    assert free == [4]
    edge = geometry_of(result.model).curves[
        geometry_of(result.model).edge_curves[free[0]]
    ]
    np.testing.assert_allclose(
        np.asarray(edge.evaluate(jnp.asarray((0.0, 0.3, 2.0)))),
        np.asarray(((5, 0, 0), (5, 0.3, 0), (5, 2, 0))),
    )


def test_occt_planar_face_compound_preserves_unattached_edge(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    face = BRepBuilderAPI_MakeFace(
        gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, 2, 0, 1
    ).Face()
    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(5, 0, 0), gp_Pnt(5, 3, 0)).Edge()
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, face)
    builder.Add(compound, edge)
    path = tmp_path / "occt-face-and-edge.brep"
    BRepTools.Write_s(compound, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert result.model.topology.num_faces == 1 and result.model.topology.num_edges == 5
    free = [
        index for index, faces in enumerate(result.model.topology.edge_faces) if not faces
    ]
    assert len(free) == 1
    endpoints = np.asarray(geometry_of(result.model).vertex_points)[
        list(geometry_of(result.model).edge_vertices[free[0]])
    ]
    np.testing.assert_allclose(endpoints, np.asarray(((5, 0, 0), (5, 3, 0))))
    exported = tmp_path / "round-trip.brep"
    write_brep_text(result.model, exported)
    restored = read_brep_text(
        exported, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert (
        geometry_of(restored.model).geometry_id == geometry_of(result.model).geometry_id
    )


@pytest.mark.parametrize("periodic_axes", ((True, False), (False, True), (True, True)))
def test_occt_periodic_rational_surface_unrolls_exactly(
    periodic_axes: tuple[bool, bool], tmp_path: Path
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepTools import BRepTools
    from OCP.collections import (
        Array1_double,
        Array1_int,
        Array2_double,
        Array2_gp_Pnt,
    )
    from OCP.Geom import Geom_BSplineSurface
    from OCP.gp import gp_Pnt

    counts = tuple(6 if periodic else 2 for periodic in periodic_axes)
    degrees = tuple(3 if periodic else 1 for periodic in periodic_axes)
    knots, mults = [], []
    for count, periodic in zip(counts, periodic_axes, strict=True):
        size = count + 1 if periodic else 2
        values = Array1_double(1, size)
        multiplicities = Array1_int(1, size)
        for index in range(size):
            values.SetValue(index + 1, float(index))
            multiplicities.SetValue(index + 1, 1 if periodic else 2)
        knots.append(values)
        mults.append(multiplicities)
    poles = Array2_gp_Pnt(1, counts[0], 1, counts[1])
    weights = Array2_double(1, counts[0], 1, counts[1])
    for i in range(counts[0]):
        u = 2 * np.pi * i / counts[0] if periodic_axes[0] else 0.5 * i
        for j in range(counts[1]):
            v = 2 * np.pi * j / counts[1] if periodic_axes[1] else 0.5 * j
            point = ((3 + np.cos(v)) * np.cos(u), (3 + np.cos(v)) * np.sin(u), np.sin(v))
            poles.SetValue(i + 1, j + 1, gp_Pnt(*point))
            weights.SetValue(i + 1, j + 1, 1.0 + 0.1 * np.sin(u) + 0.05 * np.cos(v))
    surface = Geom_BSplineSurface(
        poles,
        weights,
        knots[0],
        knots[1],
        mults[0],
        mults[1],
        degrees[0],
        degrees[1],
        periodic_axes[0],
        periodic_axes[1],
    )
    face = BRepBuilderAPI_MakeFace(surface, _CONSTRUCTION_TOLERANCE).Face()
    path = tmp_path / "periodic-surface.brep"
    BRepTools.Write_s(face, str(path))
    result = read_brep_text(
        path, _OCCT_SPLINE_POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    uv = np.asarray(
        [
            (u, v)
            for u in np.linspace(0, counts[0] if periodic_axes[0] else 1, 7)
            for v in np.linspace(0, counts[1] if periodic_axes[1] else 1, 7)
        ]
    )
    expected = []
    for u, v in uv:
        point = surface.Value(float(u), float(v))
        expected.append((point.X(), point.Y(), point.Z()))
    np.testing.assert_allclose(
        np.asarray(result.model.patches[0].evaluate(jnp.asarray(uv))),
        np.asarray(expected),
        atol=_PERIODIC_SURFACE_ATOL,
    )
    assert result.model.topology.num_faces == 1
    assert result.coverage.pcurves_fitted == 0
    exported = tmp_path / "unrolled.brep"
    write_brep_text(result.model, exported)
    restored = read_brep_text(
        exported, _OCCT_SPLINE_POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert (
        geometry_of(restored.model).geometry_id == geometry_of(result.model).geometry_id
    )


def test_occt_periodic_rational_curve_keeps_source_pole_origin(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepTools import BRepTools
    from OCP.collections import Array1_double, Array1_gp_Pnt, Array1_int
    from OCP.Geom import Geom_BSplineCurve
    from OCP.gp import gp_Pnt

    poles = Array1_gp_Pnt(1, 6)
    weights = Array1_double(1, 6)
    knots = Array1_double(1, 7)
    mults = Array1_int(1, 7)
    for index in range(6):
        angle = index * 2 * np.pi / 6
        poles.SetValue(
            index + 1, gp_Pnt(np.cos(angle), np.sin(angle), 0.2 * np.sin(2 * angle))
        )
        weights.SetValue(index + 1, 1.0 + 0.1 * index)
    for index in range(7):
        knots.SetValue(index + 1, float(index))
        mults.SetValue(index + 1, 1)
    curve = Geom_BSplineCurve(poles, weights, knots, mults, 3, True)
    edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    path = tmp_path / "periodic-edge.brep"
    BRepTools.Write_s(edge, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    parameters = np.asarray((0.0, 0.01, 0.8, 2.3, 5.99, 6.0))
    expected = []
    for parameter in parameters:
        point = curve.Value(float(parameter))
        expected.append((point.X(), point.Y(), point.Z()))
    native = geometry_of(result.model).curves[geometry_of(result.model).edge_curves[0]]
    np.testing.assert_allclose(
        np.asarray(native.evaluate(jnp.asarray(parameters))),
        np.asarray(expected),
        atol=_SURFACE_LIFT_ATOL,
    )
    assert result.model.topology.edge_faces == ((),)


def test_occt_surface_array_fixture_imports_native_exact_surface(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepTools import BRepTools
    from OCP.collections import Array2_gp_Pnt
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.gp import gp_Pnt

    points = Array2_gp_Pnt(1, 5, 1, 5)
    for i, v in enumerate(np.linspace(0, 1, 5)):
        for j, u in enumerate(np.linspace(0, 1, 5)):
            points.SetValue(
                i + 1, j + 1, gp_Pnt(u, v, 0.15 * u * v + 0.05 * u**2 - 0.03 * v**2)
            )
    surface = GeomAPI_PointsToBSplineSurface(points, 2, 3, GeomAbs_C2, 1.0e-6).Surface()
    path = tmp_path / "surface-array.brep"
    BRepTools.Write_s(BRepBuilderAPI_MakeFace(surface, 1.0e-6).Face(), str(path))
    result = read_brep_text(
        path, _OCCT_SPLINE_POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    _require_type(result.model.patches[0], BSplineSurfacePatch)
    bounds = np.asarray(result.model.parameter_bounds[0])
    uv = np.asarray(
        [
            (u, v)
            for u in np.linspace(float(bounds[0, 0]), float(bounds[1, 0]), 5)
            for v in np.linspace(float(bounds[0, 1]), float(bounds[1, 1]), 5)
        ]
    )
    expected = []
    for u, v in uv:
        point = surface.Value(float(u), float(v))
        expected.append((point.X(), point.Y(), point.Z()))
    np.testing.assert_allclose(
        np.asarray(result.model.patches[0].evaluate(jnp.asarray(uv))),
        np.asarray(expected),
        atol=_SURFACE_LIFT_ATOL,
    )


def test_occt_three_circle_loft_preserves_spline_solid(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt

    loft = BRepOffsetAPI_ThruSections(True, False, _CONSTRUCTION_TOLERANCE)
    for center, radius in (
        ((0, 0, 0), 1.0),
        ((0.15, -0.05, 0.8), 1.25),
        ((-0.1, 0.1, 1.7), 0.9),
    ):
        circle = gp_Circ(gp_Ax2(gp_Pnt(*center), gp_Dir(0, 0, 1)), radius)
        loft.AddWire(
            BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circle).Edge()).Wire()
        )
    loft.Build()
    shape = loft.Shape()
    path = tmp_path / "circle-loft.brep"
    BRepTools.Write_s(shape, str(path))
    result = read_brep_text(
        path, _OCCT_SPLINE_POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert result.model.mesh_vertices.shape == (0, 3)
    assert result.model.mesh_faces.shape == (0, 3)
    assert any(isinstance(patch, BSplineSurfacePatch) for patch in result.model.patches)
    assert _counts(result.model) == _occt_counts(shape)
    assert float(solid_volumes(result.model)[0]) == pytest.approx(
        _occt_volume(shape), rel=_LOFT_VOLUME_RTOL
    )


def _periodic_circle_text(*, last_multiplicity: int = 2) -> str:
    weight = float(np.sqrt(0.5))
    points = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
    poles = " ".join(
        f"{x} {y} 0 {1.0 if i % 2 == 0 else weight!r}" for i, (x, y) in enumerate(points)
    )
    return (
        "DBRep_DrawableShape\n\nCASCADE Topology V3, (c) Open Cascade\n"
        "Locations 0\nCurve2ds 0\nCurves 1\n"
        f"7 1 1 2 8 5 {poles} 0 2 1 2 2 2 3 2 4 {last_multiplicity}\n"
        "Polygon3D 0\nPolygonOnTriangulations 0\nSurfaces 0\nTriangulations 0\n"
        "TShapes 2\nVe\n1e-07\n1 0 0\n0 0\n0101101\n*\n"
        "Ed\n1e-07 1 1 0\n1 1 0 0 4\n0\n0101000\n+2 0 -2 0 *\n\n+1 0\n"
    )


def test_periodic_rational_circle_imports_without_occt_runtime(tmp_path: Path) -> None:
    result = decode_brep_text_bytes(
        _periodic_circle_text().encode("ascii"),
        _POLICY,
        source_length_unit=METER,
    )
    curve = geometry_of(result.model).curves[0]
    root_half = np.sqrt(0.5)
    parameters = np.asarray((0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4))
    expected = np.asarray(
        (
            (1, 0, 0),
            (root_half, root_half, 0),
            (0, 1, 0),
            (-root_half, root_half, 0),
            (-1, 0, 0),
            (-root_half, -root_half, 0),
            (0, -1, 0),
            (root_half, -root_half, 0),
            (1, 0, 0),
        )
    )
    np.testing.assert_allclose(
        np.asarray(curve.evaluate(jnp.asarray(parameters))),
        np.asarray(expected),
        atol=_CARRIER_ATOL,
    )
    assert result.model.topology.num_edges == 1 and result.model.topology.num_faces == 0
    assert result.model.topology.edge_faces == ((),)
    path = tmp_path / "finite-circle.brep"
    write_brep_text(result.model, path)
    restored = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert (
        geometry_of(restored.model).geometry_id == geometry_of(result.model).geometry_id
    )


def test_periodic_spline_rejects_mismatched_seam_multiplicities() -> None:
    with pytest.raises(CadInterchangeError) as refused:
        _decode(_periodic_circle_text(last_multiplicity=1))
    assert refused.value.refusal.reason == "malformed"


def test_occt_box_with_spherical_cavity_preserves_void_shell(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Pnt

    box = BRepPrimAPI_MakeBox(gp_Pnt(-2, -2, -2), 4, 4, 4).Shape()
    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0), 1).Shape()
    shape = BRepAlgoAPI_Cut(box, sphere).Shape()
    path = tmp_path / "spherical-cavity.brep"
    BRepTools.Write_s(shape, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _counts(result.model) == _occt_counts(shape)
    assert len(geometry_of(result.model).solid_shells[0]) == 2
    assert float(solid_volumes(result.model)[0]) == pytest.approx(
        64 - 4 * np.pi / 3, rel=_SOLID_VOLUME_RTOL
    )


@pytest.mark.parametrize("source_unit", (METER, MILLIMETER))
def test_occt_split_boxes_preserve_shared_face_and_solid_incidence(
    source_unit: UnitDefinition, tmp_path: Path
) -> None:
    pytest.importorskip("OCP")
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Pnt

    splitter = BOPAlgo_Splitter()
    for x in (0.0, 1.0):
        splitter.AddArgument(BRepPrimAPI_MakeBox(gp_Pnt(x, 0, 0), 1, 1, 1).Shape())
    splitter.Perform()
    shape = splitter.Shape()
    path = tmp_path / "split-boxes.brep"
    BRepTools.Write_s(shape, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=source_unit
    )
    assert result.model.topology.num_solids == 2
    assert geometry_of(result.model).assembly_containers == (
        BRepAssemblyContainer(("model",), (("solid0",), ("solid1",))),
    )
    assert result.model.topology.num_faces == 11
    shared = set(geometry_of(result.model).shell_faces[0]) & set(
        geometry_of(result.model).shell_faces[1]
    )
    assert len(shared) == 1
    face = next(iter(shared))
    senses = []
    for faces, signs in zip(
        geometry_of(result.model).shell_faces,
        geometry_of(result.model).shell_orientations,
        strict=True,
    ):
        senses.append(signs[faces.index(face)])
    assert tuple(sorted(senses)) == (-1, 1)
    factor = 1.0 if source_unit is METER else 1.0e-3
    np.testing.assert_allclose(
        np.asarray(prepare_brep_query(result.model).measures.solid_volumes),
        np.asarray(np.asarray((factor**3, factor**3))),
        rtol=_SOLID_VOLUME_RTOL,
    )
    exported = tmp_path / "shared-round-trip.brep"
    write_brep_text(result.model, exported)
    restored = read_brep_text(
        exported, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert (
        geometry_of(restored.model).geometry_id == geometry_of(result.model).geometry_id
    )


def test_independent_instances_do_not_gain_inferred_container_from_paths_or_placement(
    tmp_path: Path,
) -> None:
    occurrences = (
        BRepOccurrence(("same-prefix", "first"), 0),
        BRepOccurrence(("same-prefix", "second"), 0),
    )
    model = native_brep_text_assembly(occurrences, ())
    path = tmp_path / "independent.brep"
    with pytest.raises(CadInterchangeError) as refused:
        write_brep_text(model, path)
    assert refused.value.refusal.reason == "inexact-export"
    assert not path.exists()
    assert geometry_of(model).assembly_containers == ()


def test_arbitrary_occurrence_and_container_names_declare_normalization_loss(
    tmp_path: Path,
) -> None:
    occurrence = BRepOccurrence(
        ("experiment", "specimen-A"),
        0,
        translation=np.asarray((3.0, 0.0, 0.0)),
    )
    group = BRepAssemblyContainer(("scientific-group",), (occurrence.path,))
    model = native_brep_text_assembly((occurrence,), (group,))
    path = tmp_path / "named.brep"
    exported = write_brep_text(model, path)
    assert exported.report.status == AdapterStatus.DECLARED_LOSS
    assert {loss.path.split(":")[0] for loss in exported.report.losses} == {
        "occurrence-path",
        "container-path",
    }
    restored = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    ).model
    assert geometry_of(restored).occurrences[0].translation == occurrence.translation
    assert geometry_of(restored).assembly_containers == (
        BRepAssemblyContainer(("model",), (("solid0",),)),
    )
    assert geometry_of(restored).geometry_id != geometry_of(model).geometry_id


def test_occt_nested_compound_membership_is_explicit_and_round_trips(
    tmp_path: Path,
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Compound

    unit = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    along = gp_Trsf()
    along.SetTranslation(gp_Vec(3, 0, 0))
    above = gp_Trsf()
    above.SetTranslation(gp_Vec(0, 0, 5))
    builder = BRep_Builder()
    child, root = TopoDS_Compound(), TopoDS_Compound()
    builder.MakeCompound(child)
    builder.MakeCompound(root)
    builder.Add(child, unit)
    builder.Add(child, unit.Moved(TopLoc_Location(along)))
    builder.Add(root, unit)
    builder.Add(root, child.Moved(TopLoc_Location(above)))
    path = tmp_path / "nested.brep"
    BRepTools.Write_s(root, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    expected = (
        BRepAssemblyContainer(
            ("model",), (("solid0", "instance0"),), (("model", "container0"),)
        ),
        BRepAssemblyContainer(
            ("model", "container0"), (("solid0", "instance1"), ("solid0", "instance2"))
        ),
    )
    assert geometry_of(result.model).assembly_containers == expected
    assert tuple(
        occurrence.translation for occurrence in geometry_of(result.model).occurrences
    ) == (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 5.0),
        (3.0, 0.0, 5.0),
    )
    round_trip = tmp_path / "nested-round-trip.brep"
    write_brep_text(result.model, round_trip)
    restored = read_brep_text(
        round_trip, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert (
        geometry_of(restored.model).geometry_id == geometry_of(result.model).geometry_id
    )


@pytest.fixture(scope="module")
def approximated_cap(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[BRepModel, Path, CadExportResult]:
    model = native_intersection_cap()
    policy = CadExportPolicy(
        intersection_approximation=CadCurveFitPolicy(
            tolerance=_FIT_TOLERANCE,
            maximum_control_points=512,
            check_samples=1025,
        )
    )
    path = tmp_path_factory.mktemp("brep-cap") / "sampled-cap.brep"
    return model, path, write_brep_text(model, path, policy=policy)


def test_approximated_entity_tolerances_include_continuous_deviation(
    approximated_cap: tuple[BRepModel, Path, CadExportResult],
) -> None:
    model, path, exported = approximated_cap
    gaps: BRepGapEvidence = certify_brep_gaps(model)
    written = _written_tolerances(path.read_text(), exported.provenance)
    proof = exported.approximations[0].continuous_evidence
    deviation = max(
        proof.distance_bound,
        proof.first_correspondence_bound,
        proof.second_correspondence_bound,
    )
    geometry = geometry_of(model)
    affected: set[BRepEntityId] = {
        model.edge_ids[0],
        *(model.vertex_ids[index] for index in geometry.edge_vertices[0]),
        *(
            model.face_ids[geometry.coedge_faces[index]]
            for index, edge in enumerate(geometry.coedge_edges)
            if edge == 0
        ),
    }
    vertex_errors = dict(exported.approximations[0].source_vertex_error_bounds)
    for entity in (*model.vertex_ids, *model.edge_ids, *model.face_ids):
        bound = gaps.bound(entity)
        if entity in affected:
            entity_deviation = deviation
            if entity.kind == "vertex" and entity.index in vertex_errors:
                entity_deviation = float(
                    np.nextafter(deviation + vertex_errors[entity.index], np.inf)
                )
            bound = float(np.nextafter(bound + entity_deviation, np.inf))
        label = f"{entity.kind}:{entity.index}"
        _require_outward_decimal(written[label], bound, label)


def test_brep_substitutes_real_intersection_branch_only_under_explicit_policy(
    tmp_path: Path,
    approximated_cap: tuple[BRepModel, Path, CadExportResult],
) -> None:
    model, path, exported = approximated_cap
    source_id = geometry_of(model).geometry_id
    branch = _require_type(geometry_of(model).curves[0], IntersectionCurve)
    with pytest.raises(CadInterchangeError) as exact:
        write_brep_text(model, tmp_path / "exact-cap.brep")
    assert exact.value.refusal.reason == "inexact-export"
    assert exported.report.status == AdapterStatus.DECLARED_LOSS
    assert tuple(
        approximation.branch_id for approximation in exported.approximations
    ) == (branch.branch_id,)
    restored = read_brep_text(
        path, _POLICY, trusted_root=path.parent, source_length_unit=METER
    ).model
    restored_curve = _require_type(geometry_of(restored).curves[0], BSplineCurve)
    _require_type(geometry_of(restored).pcurves[0], BSplineCurve)
    parameters = np.linspace(
        float(branch.parameter_interval[0]), float(branch.parameter_interval[1]), 129
    )
    points = np.asarray(restored_curve.evaluate(jnp.asarray(parameters)))
    np.testing.assert_allclose(
        np.asarray(points[:, 2]), np.asarray(0.3), atol=_SURFACE_LIFT_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(np.linalg.norm(points[:, :2], axis=1)),
        np.asarray(np.sqrt(1.0 - 0.3**2)),
        atol=_FIT_RADIAL_ATOL,
    )
    np.testing.assert_array_equal(
        np.asarray(geometry_of(restored).edge_ranges),
        np.asarray(geometry_of(model).edge_ranges),
    )
    assert geometry_of(restored).coedge_senses == geometry_of(model).coedge_senses
    assert float(prepare_brep_query(restored).measures.face_areas[0]) == pytest.approx(
        np.pi * (1.0 - 0.3**2), rel=_DISK_AREA_RTOL
    )
    assert geometry_of(model).geometry_id == source_id


def test_scaled_oblique_plane_chart_exports_exact_pcurve_geometry(tmp_path: Path) -> None:
    patch = PlanePatch((0, 0, 0.5), (2, 0, 0), (1, 3, 0))
    uv = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)))
    points = np.asarray(patch.evaluate(jnp.asarray(uv)))
    stage = CadStage(3.0, None)
    vertices = [
        stage.vertex(point, f"corner:{index}") for index, point in enumerate(points)
    ]
    loop = []
    for index in range(4):
        following = (index + 1) % 4
        edge = stage.edge(
            LineCurve(points[index], points[following] - points[index]),
            vertices[index],
            vertices[following],
            f"edge:{index}",
            parameter_range=(0, 1),
        )
        loop.append(
            StagedCoedge(
                edge,
                1,
                LineCurve(uv[index], uv[following] - uv[index]),
                f"edge:{index}",
            )
        )
    stage.face(patch, [loop], 1, "plane", "oblique", outer_known=True)
    model = stage.publish(
        coordinate_contract=_CONTRACT,
        source_id="oblique",
        source_format="native",
        source_digest="0" * 64,
        import_policy_id=_POLICY.policy_id,
        tessellation=None,
        default_occurrences=True,
    )
    path = tmp_path / "oblique.brep"
    write_brep_text(model, path)
    restored = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    ).model
    roundoff_bound = (
        32.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(points))))
    )
    np.testing.assert_allclose(
        np.asarray(geometry_of(restored).vertex_points),
        np.asarray(points),
        rtol=0.0,
        atol=roundoff_bound,
    )
    for coedge, pcurve in enumerate(geometry_of(restored).pcurves):
        edge = geometry_of(restored).coedge_edges[coedge]
        first, last = np.asarray(geometry_of(restored).edge_ranges[edge])
        parameters = first + np.asarray((0.17, 0.63, 0.94)) * (last - first)
        carrier = geometry_of(restored).curves[geometry_of(restored).edge_curves[edge]]
        np.testing.assert_allclose(
            np.asarray(
                restored.patches[0].evaluate(pcurve.evaluate(jnp.asarray(parameters)))
            ),
            np.asarray(carrier.evaluate(jnp.asarray(parameters))),
            atol=_MAPPED_CHART_ATOL,
        )
    assert float(prepare_brep_query(restored).measures.face_areas[0]) == pytest.approx(
        6.0, rel=_PLANAR_AREA_RTOL
    )


def test_single_located_independent_root_keeps_placement_without_container(
    tmp_path: Path,
) -> None:
    occurrence = BRepOccurrence(("solid0",), 0, translation=np.asarray((4.0, 5.0, 6.0)))
    model = native_brep_text_assembly((occurrence,), ())
    path = tmp_path / "located-root.brep"
    write_brep_text(model, path)
    restored = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    ).model
    assert geometry_of(restored).occurrences == (occurrence,)
    assert geometry_of(restored).assembly_containers == ()
    assert geometry_of(restored).geometry_id == geometry_of(model).geometry_id


@pytest.mark.parametrize("source_unit", (METER, MILLIMETER))
def test_exact_offset_operation_tree_round_trips_with_units(
    source_unit: UnitDefinition, tmp_path: Path
) -> None:
    model = native_offset_plane(_POLICY)
    path = tmp_path / "offset.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=source_unit
    )
    patch = _require_type(result.model.patches[0], OffsetSurface)
    _require_type(patch.base, PlanePatch)
    factor = 1.0 if source_unit is METER else 1.0e-3
    assert float(patch.distance) == 0.25 * factor
    uv = factor * np.asarray(((0.2, 0.7), (1.7, 0.2)))
    np.testing.assert_allclose(
        np.asarray(patch.evaluate(jnp.asarray(uv))),
        np.asarray(factor * np.asarray(((0.2, 0.7, 0.75), (1.7, 0.2, 0.75)))),
        atol=_CARRIER_ATOL,
    )
    assert result.model.topology.num_faces == 1 and result.model.topology.num_edges == 4
    assert float(
        prepare_brep_query(result.model).measures.face_areas[0]
    ) == pytest.approx(2 * factor**2, rel=_OFFSET_AREA_RTOL)


def test_offset_placement_preserves_signed_normal_under_reflection() -> None:
    patch = OffsetSurface(PlanePatch((0, 0, 0.5), (1, 0, 0), (0, 1, 0)), 0.25)
    matrix = np.asarray(((-1, 0, 0, 3), (0, 1, 0, 2), (0, 0, 1, 1)), dtype=np.float64)
    placement = RigidPlacement.from_matrix(matrix, "reflection", reflection=True)
    placed = place_surface(patch, placement, "offset-plane")
    uv = np.asarray(((0.2, 0.7), (1.7, 0.2)))
    expected = (
        np.asarray(patch.evaluate(jnp.asarray(uv))) @ placement.rotation.T
        + placement.translation
    )
    np.testing.assert_allclose(
        np.asarray(placed.evaluate(jnp.asarray(uv))),
        np.asarray(expected),
        atol=_CARRIER_ATOL,
    )


@pytest.mark.parametrize("kind", ("parabola", "hyperbola", "offset"))
@pytest.mark.parametrize("source_unit", (METER, MILLIMETER))
def test_independent_conic_and_offset_edge_retains_source_parameter_and_archive(
    kind: Literal["parabola", "hyperbola", "offset"],
    source_unit: UnitDefinition,
    tmp_path: Path,
) -> None:
    """Hand-authored BRep text; the coordinate oracle does not call a writer."""
    frame = "0 0 0 0 0 1 1 0 0 0 1 0"
    if kind == "hyperbola":
        record = f"5 {frame} 2 1"
    elif kind == "parabola":
        record = f"4 {frame} 0.5"
    else:
        record = f"9 0.25 0 0 1\n4 {frame} 0.5"

    def analytic(parameter: np.ndarray) -> np.ndarray:
        if kind == "hyperbola":
            return np.stack(
                (2.0 * np.cosh(parameter), np.sinh(parameter), np.zeros_like(parameter)),
                axis=-1,
            )
        x, y = 0.5 * parameter**2, parameter
        if kind == "offset":
            denominator = np.sqrt(1.0 + parameter**2)
            x, y = x + 0.25 / denominator, y - 0.25 * parameter / denominator
        return np.stack((x, y, np.zeros_like(parameter)), axis=-1)

    endpoints = analytic(np.asarray([-1.0, 1.0], dtype=np.float64))
    start = " ".join(repr(float(value)) for value in endpoints[0])
    end = " ".join(repr(float(value)) for value in endpoints[1])
    text = (
        "DBRep_DrawableShape\n\nCASCADE Topology V3, (c) Open Cascade\n"
        f"Locations 0\nCurve2ds 0\nCurves 1\n{record}\n"
        "Polygon3D 0\nPolygonOnTriangulations 0\nSurfaces 0\nTriangulations 0\n"
        f"TShapes 3\nVe\n1e-07\n{start}\n0 0\n0101101\n*\n"
        f"Ve\n1e-07\n{end}\n0 0\n0101101\n*\n"
        "Ed\n1e-07 1 1 0\n1 1 0 -1 1\n0\n0101000\n+3 0 -2 0 *\n\n+1 0\n"
    )
    imported = decode_brep_text_bytes(
        text.encode("ascii"), _POLICY, source_length_unit=source_unit
    )
    assert imported.report.valid
    assert imported.model.topology.num_edges == 1
    assert imported.model.topology.num_faces == 0
    assert imported.model.topology.edge_faces == ((),)
    geometry = geometry_of(imported.model)
    curve = geometry.curves[geometry.edge_curves[0]]
    expected_type = {
        "parabola": ParabolaCurve,
        "hyperbola": HyperbolaCurve,
        "offset": OffsetCurve,
    }[kind]
    assert isinstance(curve, expected_type)
    factor = 1.0 if source_unit is METER else 1.0e-3
    parameter_factor = 1.0 if kind == "hyperbola" else factor
    parameters = np.asarray([-1.0, -0.3, 0.0, 0.6, 1.0], dtype=np.float64)
    expected = analytic(parameters) * factor
    values = curve.evaluate(jnp.asarray(parameters * parameter_factor))
    np.testing.assert_allclose(values, expected, rtol=_CARRIER_ATOL, atol=_CARRIER_ATOL)
    lower, upper = curve.derivative_bounds(-parameter_factor, parameter_factor)
    jets = jax.vmap(jax.jacfwd(curve.evaluate))(
        jnp.asarray(parameters * parameter_factor)
    )
    assert np.all(np.asarray(jets) >= lower) and np.all(np.asarray(jets) <= upper)
    path = tmp_path / "exact-edge.brep"
    write_brep_text(imported.model, path)
    reread = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert geometry_of(reread.model).geometry_id == geometry.geometry_id
    receipt = save_brep_archive(imported.model, tmp_path / "exact-edge.phx")
    restored = load_brep_archive(receipt.path)
    assert restored.model_id == imported.model.model_id == receipt.model_id
    assert restored.source_id == imported.model.source_id
    np.testing.assert_array_equal(
        geometry_of(restored)
        .curves[0]
        .evaluate(jnp.asarray(parameters * parameter_factor)),
        values,
    )


def test_offset_edge_distance_jvp_vjp_retains_basis_parameters() -> None:
    source = OffsetCurve(ParabolaCurve((0, 0), (1, 0), (0, 1), 0.5), 0.25)
    parameter = jnp.asarray(0.4, dtype=jnp.float64)

    def moved(distance: Array) -> Array:
        return eqx.tree_at(lambda curve: curve.distance, source, distance).evaluate(
            parameter
        )

    _, tangent = jax.jvp(
        moved, (source.distance,), (jnp.asarray(1.0, dtype=jnp.float64),)
    )
    expected = np.asarray([1.0, -0.4], dtype=np.float64) / np.sqrt(1.0 + 0.4**2)
    np.testing.assert_allclose(tangent, expected, rtol=_CARRIER_ATOL, atol=_CARRIER_ATOL)
    cotangent = jnp.asarray([0.3, -0.7], dtype=jnp.float64)
    _, pullback = jax.vjp(moved, source.distance)
    np.testing.assert_allclose(
        pullback(cotangent)[0],
        expected @ np.asarray(cotangent),
        rtol=_CARRIER_ATOL,
        atol=_CARRIER_ATOL,
    )
    assert eqx.tree_equal(source.base, ParabolaCurve((0, 0), (1, 0), (0, 1), 0.5))


@pytest.mark.parametrize("kind", ("parabola", "hyperbola"))
def test_exact_conic_source_coefficients_remain_live_through_jit_jvp_and_vjp(
    kind: Literal["parabola", "hyperbola"],
) -> None:
    parameter = jnp.asarray(0.4, dtype=jnp.float64)
    source: ParabolaCurve | HyperbolaCurve
    if kind == "parabola":
        source = ParabolaCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5)
        coefficients = source.focal_length[None]
        direction = jnp.asarray([0.3], dtype=jnp.float64)
        jacobian = np.asarray([[-0.16], [0.0], [0.0]], dtype=np.float64)
    else:
        source = HyperbolaCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0, 1.0)
        coefficients = jnp.stack((source.first_radius, source.second_radius))
        direction = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
        jacobian = np.asarray(
            [[np.cosh(0.4), 0.0], [0.0, np.sinh(0.4)], [0.0, 0.0]],
            dtype=np.float64,
        )

    def changed(values: Array) -> ParabolaCurve | HyperbolaCurve:
        if isinstance(source, ParabolaCurve):
            return eqx.tree_at(lambda curve: curve.focal_length, source, values[0])
        return eqx.tree_at(
            lambda curve: (curve.first_radius, curve.second_radius),
            source,
            (values[0], values[1]),
        )

    def point(values: Array) -> Array:
        return changed(values).evaluate(parameter)

    _, tangent = jax.jvp(point, (coefficients,), (direction,))
    np.testing.assert_allclose(
        tangent, jacobian @ np.asarray(direction), rtol=_CARRIER_ATOL, atol=_CARRIER_ATOL
    )
    cotangent = jnp.asarray([0.3, -0.7, 0.2], dtype=jnp.float64)
    _, pullback = jax.vjp(point, coefficients)
    np.testing.assert_allclose(
        pullback(cotangent)[0],
        jacobian.T @ np.asarray(cotangent),
        rtol=_CARRIER_ATOL,
        atol=_CARRIER_ATOL,
    )

    def evaluate(curve: ParabolaCurve | HyperbolaCurve, value: Array) -> Array:
        return curve.evaluate(value)

    compiled = eqx.filter_jit(evaluate)
    shifted = coefficients + 1.0e-3 * direction
    moved = changed(shifted)
    if isinstance(moved, ParabolaCurve):
        expected = np.asarray(
            [0.4**2 / (4.0 * float(shifted[0])), 0.4, 0.0], dtype=np.float64
        )
    else:
        expected = np.asarray(
            [float(shifted[0]) * np.cosh(0.4), float(shifted[1]) * np.sinh(0.4), 0.0],
            dtype=np.float64,
        )
    np.testing.assert_allclose(
        compiled(moved, parameter), expected, rtol=_CARRIER_ATOL, atol=_CARRIER_ATOL
    )
    np.testing.assert_allclose(
        compiled(source, parameter),
        source.evaluate(parameter),
        rtol=_CARRIER_ATOL,
        atol=_CARRIER_ATOL,
    )


@pytest.mark.parametrize("kind", ("parabola", "hyperbola", "offset"))
@pytest.mark.parametrize("reflection", (False, True), ids=("quarter-turn", "reflection"))
def test_exact_conic_and_offset_placement_preserves_source_parameter_jvp(
    kind: Literal["parabola", "hyperbola", "offset"],
    reflection: bool,
) -> None:
    source: ParabolaCurve | HyperbolaCurve | OffsetCurve
    if kind == "parabola":
        source = ParabolaCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5)
    elif kind == "hyperbola":
        source = HyperbolaCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0, 1.0)
    else:
        source = OffsetCurve(
            ParabolaCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5),
            0.25,
            (0, 0, 1),
        )
    matrix = np.asarray(
        ((-1, 0, 0, 3), (0, 1, 0, 2), (0, 0, 1, 1))
        if reflection
        else ((0, -1, 0, 3), (1, 0, 0, 2), (0, 0, 1, 1)),
        dtype=np.float64,
    )
    placement = RigidPlacement.from_matrix(
        matrix, "conic-placement", reflection=reflection
    )
    placed = place_curve(source, placement, "exact-source-edge")
    parameters = jnp.asarray([-1.0, -0.3, 0.0, 0.6, 1.0], dtype=jnp.float64)
    expected = (
        np.asarray(source.evaluate(parameters)) @ placement.rotation.T
        + placement.translation
    )
    np.testing.assert_allclose(
        placed.evaluate(parameters), expected, rtol=_CARRIER_ATOL, atol=_CARRIER_ATOL
    )
    parameter = jnp.asarray(0.4, dtype=jnp.float64)
    _, original_jet = jax.jvp(source.evaluate, (parameter,), (jnp.ones_like(parameter),))
    _, placed_jet = jax.jvp(placed.evaluate, (parameter,), (jnp.ones_like(parameter),))
    np.testing.assert_allclose(
        placed_jet,
        np.asarray(original_jet) @ placement.rotation.T,
        rtol=_CARRIER_ATOL,
        atol=_CARRIER_ATOL,
    )


@pytest.mark.parametrize(
    "foreign_vertex", (False, True), ids=("declared-endpoint", "interior-vertex")
)
def test_bounded_closed_source_edge_admits_only_its_declared_domain_endpoints(
    foreign_vertex: bool,
) -> None:
    result = decode_brep_text_bytes(
        _periodic_circle_text().encode("ascii"),
        _POLICY,
        source_length_unit=METER,
    )
    curve = _require_type(geometry_of(result.model).curves[0], BSplineCurve)
    vertex = np.asarray((0, 1, 0) if foreign_vertex else (1, 0, 0), dtype=np.float64)
    if foreign_vertex:
        with pytest.raises(ValueError, match=r"edge_range\.start"):
            edge_range(curve, vertex, vertex, True, 1.0e-10)
    else:
        assert edge_range(curve, vertex, vertex, True, 1.0e-10) == (0.0, 4.0)
