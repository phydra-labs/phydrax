"""Certified CAD tolerance evidence and boundary fault-injection contracts."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from phydrax._external_resource import ResourceLimits
from phydrax.geometry.brep import (
    BRepEntityId,
    BRepGapEvidence,
    BRepModel,
    BRepTessellationPolicy,
    BRepValidationError,
    certify_brep_gaps,
)
from phydrax.interchange import (
    _cad,
    build_cad_import_tolerance_evidence,
    CadImportPolicy,
    CadImportResult,
    CadInterchangeError,
    certified_cad_export_tolerances,
    read_step,
    write_step,
)
from tests._support.cad_models import native_solid


@pytest.fixture(scope="module")
def imported_box_result(tmp_path_factory: pytest.TempPathFactory) -> CadImportResult:
    model = native_solid("box")
    root = tmp_path_factory.mktemp("cad-export-evidence")
    path = root / "box.step"
    write_step(model, path)
    return read_step(
        path,
        CadImportPolicy(
            model.coordinate_contract,
            ResourceLimits(1 << 24, 64, 200_000, 4_000_000, 0),
            tessellation=BRepTessellationPolicy(realize=False),
        ),
        trusted_root=root,
    )


def test_export_tolerances_round_certified_bounds_upward() -> None:
    model = native_solid("box")
    evidence = certify_brep_gaps(model)
    tolerances = certified_cad_export_tolerances(model)
    for kind, row in (
        ("vertex", tolerances.vertex_bounds),
        ("edge", tolerances.edge_bounds),
        ("face", tolerances.face_bounds),
    ):
        for index, tolerance in enumerate(row):
            bound = evidence.bound(BRepEntityId(model.source_revision, kind, index))
            written = Fraction(format(tolerance, ".17g"))
            # At most one binary successor plus less than half an ulp of decimal rounding.
            upper = float(np.nextafter(np.nextafter(bound, np.inf), np.inf))
            assert Fraction(bound) <= written <= Fraction(upper)
    assert tolerances.uncertainty == max(
        *tolerances.vertex_bounds, *tolerances.edge_bounds, *tolerances.face_bounds
    )


@pytest.mark.parametrize(
    "bound",
    [np.inf, float(np.finfo(np.float64).max)],
    ids=["unbounded-proof", "unrepresentable-outward-decimal"],
)
def test_export_refuses_unbounded_entity_evidence(
    bound: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = native_solid("box")
    measured = certify_brep_gaps(model)
    unbounded = BRepGapEvidence(
        entities=measured.entities,
        partners=measured.partners,
        kinds=measured.kinds,
        use_indices=measured.use_indices,
        upper_bounds=np.full_like(measured.upper_bounds, bound),
        model_id=model.model_id,
    )

    def gaps(candidate: BRepModel) -> BRepGapEvidence:
        return unbounded

    monkeypatch.setattr(_cad, "certify_brep_gaps", gaps)
    with pytest.raises(CadInterchangeError) as caught:
        certified_cad_export_tolerances(model)
    assert caught.value.refusal.reason == "inexact-export"
    assert caught.value.refusal.entity == "tolerance"


def test_import_retains_geometry_with_unbounded_gap_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = native_solid("box")
    measured = certify_brep_gaps(model)
    unbounded = BRepGapEvidence(
        entities=measured.entities,
        partners=measured.partners,
        kinds=measured.kinds,
        use_indices=measured.use_indices,
        upper_bounds=np.full_like(measured.upper_bounds, np.inf),
        model_id=model.model_id,
    )

    def gaps(candidate: BRepModel) -> BRepGapEvidence:
        return unbounded

    monkeypatch.setattr(_cad, "certify_brep_gaps", gaps)
    evidence = build_cad_import_tolerance_evidence(model, ((None, 0.0),))
    assert evidence.losses == ()
    assert evidence.declared_limit == model.report.curve_surface_tolerance
    assert all(
        bound is not None and np.isinf(bound) for bound in evidence.certified_bounds
    )
    assert all(bound is not None and np.isinf(bound) for bound in evidence.upper_bounds)
    assert evidence.source_bounds == (0.0,) * len(evidence.entities)
    np.testing.assert_array_equal(
        unbounded.upper_bounds, np.full_like(measured.upper_bounds, np.inf)
    )


def test_import_carries_declared_entity_bounds_below_model_limit() -> None:
    model = native_solid("box")
    evidence = certify_brep_gaps(model)
    first = BRepEntityId(model.source_revision, "vertex", 0)
    second = BRepEntityId(model.source_revision, "vertex", 1)
    declared = (
        (first, float(np.nextafter(evidence.bound(first), np.inf))),
        (second, float(np.nextafter(evidence.bound(second), np.inf))),
    )
    carried = build_cad_import_tolerance_evidence(model, declared)
    assert carried.losses == ()
    assert carried.declared_limit == model.report.curve_surface_tolerance
    for entity, tolerance in declared:
        row = carried.entities.index(entity)
        assert carried.certified_bounds[row] == evidence.bound(entity)
        assert carried.source_bounds[row] == tolerance
        assert carried.upper_bounds[row] == max(evidence.bound(entity), tolerance)


def test_import_source_tolerance_changes_result_not_model_identity(
    imported_box_result: CadImportResult,
) -> None:
    result = imported_box_result
    changed = replace(result, source_tolerance=1.0 + (result.source_tolerance or 0.0))
    assert changed.result_id != result.result_id
    assert changed.model.model_id == result.model.model_id


@pytest.mark.parametrize(
    "tolerance",
    [-1.0, np.inf, -np.inf, np.nan],
    ids=["negative", "infinite", "negative-infinite", "nan"],
)
def test_import_result_refuses_invalid_source_tolerance(
    tolerance: float,
    imported_box_result: CadImportResult,
) -> None:
    result = imported_box_result
    with pytest.raises(ValueError, match="CadImportResult.source_tolerance"):
        replace(result, source_tolerance=tolerance)


def test_geometry_replacement_preserves_source_and_derived_identity() -> None:
    geometry = native_solid("box").geometry
    assert geometry is not None
    rebuilt = replace(geometry)
    assert rebuilt.geometry_id == geometry.geometry_id
    np.testing.assert_array_equal(rebuilt.vertex_points, geometry.vertex_points)
    np.testing.assert_array_equal(
        rebuilt.vertex_evaluation_bounds, geometry.vertex_evaluation_bounds
    )
    assert rebuilt.occurrences == geometry.occurrences
    assert rebuilt.assembly_containers == geometry.assembly_containers


def test_geometry_replacement_revalidates_corrupt_stored_fields() -> None:
    geometry = native_solid("box").geometry
    assert geometry is not None
    with pytest.raises(BRepValidationError) as caught:
        replace(
            geometry,
            vertex_points=np.full_like(np.asarray(geometry.vertex_points), np.nan),
        )
    assert any(
        finding.code == "edge-incidence" for finding in caught.value.report.findings
    )


@pytest.mark.parametrize(
    "tolerance",
    [True, np.bool_(False), "0.0", 1.0j],
    ids=["boolean", "numpy-boolean", "string", "complex"],
)
def test_import_result_refuses_nonreal_source_tolerance(
    tolerance: object,
    imported_box_result: CadImportResult,
) -> None:
    with pytest.raises(TypeError, match="CadImportResult.source_tolerance"):
        replace(imported_box_result, source_tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [True, "0.0"], ids=["boolean", "string"])
def test_import_gap_comparison_refuses_nonreal_source_tolerance(
    tolerance: object,
) -> None:
    with pytest.raises(TypeError, match=r"build_cad_import_tolerance_evidence\.declared"):
        build_cad_import_tolerance_evidence(
            native_solid("box"),
            ((None, tolerance),),  # ty: ignore[invalid-argument-type]
        )


def test_import_retains_each_raw_source_tolerance_in_canonical_order(
    imported_box_result: CadImportResult,
) -> None:
    declarations = (("#98/uncertainty", 0.25), ("#12/uncertainty", 0.125))
    result = replace(imported_box_result, source_tolerances=declarations)
    assert result.source_tolerances == tuple(sorted(declarations))
    assert result.source_tolerance == imported_box_result.source_tolerance
    assert (
        result.source_length_unit_meters == imported_box_result.source_length_unit_meters
    )
    assert result.model.model_id == imported_box_result.model.model_id


def test_import_tolerance_declaration_identity_is_independent_of_input_order(
    imported_box_result: CadImportResult,
) -> None:
    declarations = (("#98/uncertainty", 0.25), ("#12/uncertainty", 0.125))
    first = replace(imported_box_result, source_tolerances=declarations)
    second = replace(imported_box_result, source_tolerances=declarations[::-1])
    assert first.result_id == second.result_id


def test_individual_source_tolerance_changes_result_identity_without_changing_summary(
    imported_box_result: CadImportResult,
) -> None:
    first = replace(
        imported_box_result,
        source_tolerances=(("#98/uncertainty", 0.25), ("#12/uncertainty", 0.125)),
    )
    second = replace(
        first,
        source_tolerances=(("#98/uncertainty", 0.25), ("#12/uncertainty", 0.0625)),
    )
    assert second.result_id != first.result_id
    assert second.source_tolerance == first.source_tolerance
    assert second.model.model_id == first.model.model_id


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        ("0.125", TypeError),
        (-1.0, ValueError),
        (np.inf, ValueError),
        (np.nan, ValueError),
    ],
    ids=["boolean", "string", "negative", "infinite", "nan"],
)
def test_import_result_refuses_invalid_raw_source_tolerance(
    value: object,
    error: type[Exception],
    imported_box_result: CadImportResult,
) -> None:
    with pytest.raises(error, match="CadImportResult.source_tolerances"):
        replace(imported_box_result, source_tolerances=(("#12/uncertainty", value),))


@pytest.mark.parametrize(
    ("declarations", "error"),
    [
        ([("#12/uncertainty", 0.125)], TypeError),
        (("", 0.125), ValueError),
        ((("", 0.125),), ValueError),
        (((12, 0.125),), TypeError),
        ((("#12/uncertainty", 0.125), ("#12/uncertainty", 0.25)), ValueError),
    ],
    ids=[
        "mutable-inventory",
        "malformed-pair",
        "empty-reference",
        "nonstring-reference",
        "duplicate-reference",
    ],
)
def test_import_result_refuses_invalid_source_tolerance_inventory(
    declarations: object,
    error: type[Exception],
    imported_box_result: CadImportResult,
) -> None:
    with pytest.raises(error, match="CadImportResult.source_tolerances"):
        replace(imported_box_result, source_tolerances=declarations)


def test_import_declared_limit_losses_group_entity_evidence() -> None:
    model = native_solid("box")
    first, second = model.vertex_ids[:2]
    limit = model.report.curve_surface_tolerance
    above_limit = float(np.nextafter(limit, np.inf))
    evidence = build_cad_import_tolerance_evidence(
        model, ((first, above_limit), (second, above_limit))
    )
    assert len(evidence.losses) == 1
    loss = evidence.losses[0]
    assert loss.path == "source-tolerance:vertex"
    assert "entity=vertex:0@" in loss.rationale
    assert "entity=vertex:1@" in loss.rationale
    assert f"declared={above_limit:.17g}" in loss.rationale
    assert f"limit={limit:.17g}" in loss.rationale
    assert not loss.changes_interpretation
    assert evidence.declared_limit == limit
    assert evidence.source_bounds[evidence.entities.index(first)] == above_limit


def test_global_source_uncertainty_is_carried_by_every_entity() -> None:
    model = native_solid("box")
    declared = model.report.curve_surface_tolerance
    evidence = build_cad_import_tolerance_evidence(model, ((None, declared),))
    assert evidence.losses == ()
    assert evidence.source_tolerance == declared
    assert evidence.source_bounds == (declared,) * len(evidence.entities)
    for native, published in zip(
        evidence.certified_bounds, evidence.upper_bounds, strict=True
    ):
        assert (
            published == declared
            if native is None
            else published == max(native, declared)
        )


def test_unrepresented_source_entity_is_reported_without_growing_limit() -> None:
    model = native_solid("box")
    unsupported = BRepEntityId(
        model.source_revision, "vertex", model.topology.num_vertices
    )
    evidence = build_cad_import_tolerance_evidence(model, ((unsupported, 0.0),))
    assert len(evidence.losses) == 1
    assert "reason=unrepresented-entity" in evidence.losses[0].rationale
    assert "declared=0" in evidence.losses[0].rationale
    assert (
        f"limit={model.report.curve_surface_tolerance:.17g}"
        in evidence.losses[0].rationale
    )
    assert unsupported not in evidence.entities
    assert evidence.declared_limit == model.report.curve_surface_tolerance


def test_result_identity_binds_carried_tolerance_evidence(
    imported_box_result: CadImportResult,
) -> None:
    first = build_cad_import_tolerance_evidence(imported_box_result.model, ((None, 0.0),))
    declared = imported_box_result.model.report.curve_surface_tolerance
    second = build_cad_import_tolerance_evidence(
        imported_box_result.model, ((None, declared),)
    )
    original = replace(imported_box_result, tolerance_evidence=first)
    changed = replace(original, tolerance_evidence=second)
    assert changed.result_id != original.result_id
    assert changed.model.model_id == original.model.model_id
    assert second.declared_limit == first.declared_limit


def test_result_refuses_foreign_model_tolerance_evidence(
    imported_box_result: CadImportResult,
) -> None:
    evidence = build_cad_import_tolerance_evidence(native_solid("sphere"), ((None, 0.0),))
    with pytest.raises(ValueError, match=r"CadImportResult.tolerance_evidence.model_id"):
        replace(imported_box_result, tolerance_evidence=evidence)
