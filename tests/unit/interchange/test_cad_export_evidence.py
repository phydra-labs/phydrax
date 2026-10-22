"""Certified CAD tolerance evidence and boundary fault-injection contracts."""

import json
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from phydrax._external_resource import ResourceLimits
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._interval_enclosure import prepare_interval_function
from phydrax.geometry.brep import (
    _correspondence as correspondence_owner,
    _curve_approximation_bounds as approximation_owner,
    _model as model_owner,
    BRepEntityId,
    BRepGapEvidence,
    BRepModel,
    BRepTessellationPolicy,
    BRepValidationError,
    certify_brep_gaps,
)
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._correspondence import RationalCurveSurfaceJets
from phydrax.geometry.brep._intersection_curve import curve_pieces_for_interval
from phydrax.geometry.brep._model import _certify_edge_face_gap_bounds
from phydrax.geometry.brep._patches import BSplineCurve, CylinderPatch, LineCurve
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations
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

    def gaps(
        candidate: BRepModel, *, budget: CoordinateEnclosureBudget | None = None
    ) -> BRepGapEvidence:
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

    def gaps(
        candidate: BRepModel, *, budget: CoordinateEnclosureBudget | None = None
    ) -> BRepGapEvidence:
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


def _bounded_descriptor() -> dict[str, Any]:
    gap = _cad._written_upper_bound(float(np.nextafter(
        0.01 + float(np.nextafter(2.0 * 0.01, np.inf)), np.inf
    )))
    return {
        "representation": "bounded-coupled-approximation",
        "source_model_id": "source-model",
        "source_geometry_id": "source-geometry",
        "source_gap_bound": 0.0,
        "gap_bound": gap,
        "rows": [{
            "source_branch_id": "source-branch",
            "approximation_id": "new-carrier",
            "spatial_bound": 0.01,
            "parameter_bound": 0.01,
            "distance_bound": 0.001,
            "parameter_error": 0.001,
            "surface_jacobian_lipschitz": 2.0,
            "gap_bound": gap,
        }],
    }


def test_bounded_descriptor_uses_frozen_not_achieved_error_law() -> None:
    descriptor = _bounded_descriptor()
    gap = _cad.decode_bounded_approximation_descriptor(json.dumps(descriptor))
    descriptor["rows"][0]["distance_bound"] = 0.0
    descriptor["rows"][0]["parameter_error"] = 0.0
    assert _cad.decode_bounded_approximation_descriptor(json.dumps(descriptor)) == gap
    assert _cad.decode_bounded_approximation_descriptor(
        json.dumps(descriptor), length_factor=0.001
    ) >= gap * 0.001


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distance_bound", 0.02),
        ("parameter_error", 0.02),
        ("gap_bound", 1.0),
        ("surface_jacobian_lipschitz", float("inf")),
        ("spatial_bound", True),
    ],
)
def test_bounded_descriptor_refuses_invalid_or_posthoc_law(field: str, value: object) -> None:
    descriptor = _bounded_descriptor()
    descriptor["rows"][0][field] = value
    with pytest.raises(CadInterchangeError, match="complete finite frozen"):
        _cad.decode_bounded_approximation_descriptor(json.dumps(descriptor))


def test_stored_spline_quadric_gap_requires_actual_continuous_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # x=1-t²/2, y=t is a bounded polynomial representative, not a
    # parametrically exact cylinder circle. No descriptor/provenance is used.
    curve = BSplineCurve(
        ((1.0, 0.0, 0.0), (1.0, 0.05, 0.0), (0.995, 0.1, 0.0)),
        np.ones(3), (0.0, 0.0, 0.0, 0.1, 0.1, 0.1), 2,
    )
    pcurve = LineCurve((0.0, 0.0), (1.0, 0.0))
    patch = CylinderPatch(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), 1.0,
    )
    args = ((curve,), (0,), np.asarray(((0.0, 0.1),)), (pcurve,), (0,), (0,), (patch,))
    with coordinate_source_request((curve, pcurve, patch)) as budget:
        proved = _certify_edge_face_gap_bounds(*args, declared_bounds=(0.0002,), budget=budget)
    assert proved is not None and 0.0 < proved[0] <= 0.0002
    enclosures: list[int] = []
    original = correspondence_owner._residual_interval_bounds

    def enclose(*args: Any, **kwargs: Any) -> Any:
        enclosures.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(correspondence_owner, "_residual_interval_bounds", enclose)
    with coordinate_source_request((curve, pcurve, patch)) as budget:
        with pytest.raises(ValueError, match="unproved or exceeded"):
            _certify_edge_face_gap_bounds(*args, declared_bounds=(1.0e-8,), budget=budget)
    # The outward center intervals already disprove the original tight
    # allowance; do not subdivide an impossible bound 65,536 times.
    assert enclosures == [1]


def test_coedge_proof_reuse_requires_same_earned_request_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = native_solid("box")
    with coordinate_source_request((model,)) as budget:
        actual = model.certified_coedge_bounds(budget=budget)
        proof = budget.brep_coedge_proof_cache[id(model)]
        assert proof.owner is model
        assert proof.model_id == model.model_id
        assert proof.tolerance == model.report.curve_surface_tolerance

        def refuse_reproof(*args: object, **kwargs: object) -> None:
            raise RuntimeError("independent coedge validation reached")

        monkeypatch.setattr(model_owner, "_validate_exact_geometry", refuse_reproof)
        reused = model.certified_coedge_bounds(budget=budget)
        np.testing.assert_array_equal(reused, actual)
        assert not reused.flags.writeable
    with coordinate_source_request((model,)) as fresh:
        assert id(model) not in fresh.brep_coedge_proof_cache
        with pytest.raises(RuntimeError, match="independent coedge validation reached"):
            model.certified_coedge_bounds(budget=fresh)


@pytest.mark.parametrize("with_curve", (False, True), ids=("point", "rational-curve"))
def test_joint_residual_jets_equal_individual_orders_and_share_bounds(
    with_curve: bool,
) -> None:
    curve = BSplineCurve(
        ((1.0, 0.0, 0.0), (1.0, 0.05, 0.025), (0.995, 0.1, 0.1)),
        (1.0, 0.8, 1.0), (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), 2,
    )
    pcurve = BSplineCurve(
        ((0.0, 0.0), (0.05, 0.025), (0.1, 0.1)),
        (1.0, 0.9, 1.0), (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), 2,
    )
    patch = CylinderPatch(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), 1.0,
    )
    calls: list[list[tuple[bytes, bytes]]] = []
    curve_calls: list[None] = []

    def prepare(function: Any, dimension: int, coefficients: Any) -> Any:
        program = prepare_interval_function(
            function, dimension, batch_capacity=1, constant_bounds=coefficients
        )
        bounds: list[tuple[bytes, bytes]] = []
        calls.append(bounds)

        class Capture:
            def evaluate(
                self, lower: np.ndarray, upper: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
                bounds.append((lower.tobytes(), upper.tobytes()))
                return program.evaluate(lower, upper)

        return Capture()

    jets = RationalCurveSurfaceJets(
        curve.bezier_pieces()[0] if with_curve else None,
        pcurve.bezier_pieces()[0], patch, point=np.asarray((1.0, 0.0, 0.0)),
        prepare=prepare, observe_curve_jet=lambda: curve_calls.append(None),
    )
    starts = np.asarray(((0.125,), (0.5,)))
    ends = np.asarray(((0.375,), (0.875,)))
    centers = 0.5 * (starts + ends)
    with coordinate_source_request((curve, pcurve, patch)) as budget:
        actual = jets.residual_jets_batch(
            starts, ends, centers, budget=budget, preparations=None
        )
        batch_calls = tuple(tuple(items) for items in calls)
        batch_curve_calls = len(curve_calls)
        for items in calls:
            items.clear()
        curve_calls.clear()
        for index, (lower, upper, order) in enumerate((
            (centers, centers, 0), (starts, ends, 1),
            (centers, centers, 1), (starts, ends, 2),
        )):
            expected = [
                jets.jet(float(a[0]), float(b[0]), order, budget=budget)
                for a, b in zip(lower, upper, strict=True)
            ]
            shape = (len(starts), 3) + (1,) * order
            for side in (0, 1):
                np.testing.assert_array_equal(
                    actual[index][side],
                    np.asarray([row[side] for row in expected]).reshape(shape),
                )
    # Identical UV boxes feed every value/Jacobian/Hessian premise; the joint
    # route only removes the duplicated full-interval Jacobian and quotient jets.
    assert tuple(map(len, batch_calls)) == (2, 4, 2)
    assert tuple(map(len, calls)) == (2, 6, 2)
    for batch, individual in zip(batch_calls, calls, strict=True):
        assert set(batch) == set(individual)
    assert 2 * batch_curve_calls == len(curve_calls)


@pytest.mark.parametrize("owns_center", (False, True), ids=("overlap-only", "contains-center"))
def test_residual_disproof_requires_center_uv_in_supporting_piece(
    owns_center: bool,
) -> None:
    curve = BSplineCurve(
        ((1.0, 0.0, 0.0), (1.0, 0.05, 0.0), (0.995, 0.1, 0.0)),
        np.ones(3), (0.0, 0.0, 0.0, 0.1, 0.1, 0.1), 2,
    )
    pcurve = LineCurve((0.0, 0.0), (1.0, 0.0))
    patch = CylinderPatch(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), 1.0,
    )
    preparations = TrimRequestPreparations()
    with coordinate_source_request((curve, pcurve, patch)) as budget:
        workset = correspondence_owner._prepare_correspondence_workset(
            curve, pcurve, patch, np.asarray(((0.0, -1.0), (0.1, 1.0))),
            0.0, 0.1, preparations, budget,
        )
        patches = tuple(
            replace(
                piece,
                lower=np.asarray((0.0 if owns_center else 0.075, -1.0)),
            )
            for piece in workset.patches
        )
        outcome = correspondence_owner._residual_interval_bounds(
            [(0.0, 0.1)], curve, np.zeros(3),
            workset.curve_segments, workset.parameter_segments, patches, {},
            budget=budget, tolerance=1.0e-8, preparations=preparations,
        )[(0.0, 0.1)]
    if owns_center:
        assert outcome is None
    else:
        assert outcome is not None and outcome > 1.0e-8


@pytest.mark.parametrize("with_curve", (False, True), ids=("point-carrier", "actual-curve"))
def test_correspondence_cache_point_premise_only_applies_without_curve(
    with_curve: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = BSplineCurve(
        ((1.0, 0.0, 0.0), (1.0, 0.05, 0.0), (0.995, 0.1, 0.0)),
        np.ones(3), (0.0, 0.0, 0.0, 0.1, 0.1, 0.1), 2,
    )
    pcurve = LineCurve((0.0, 0.0), (1.0, 0.0))
    patch = CylinderPatch(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), 1.0,
    )
    original = correspondence_owner._execute_correspondence_workset
    invocations: list[None] = []

    def execute(*args: Any, **kwargs: Any) -> Any:
        invocations.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(correspondence_owner, "_execute_correspondence_workset", execute)
    with coordinate_source_request((curve, pcurve, patch)) as budget:
        source_curve = curve if with_curve else None
        first = correspondence_owner.certify_curve_surface(
            source_curve, pcurve, patch, np.asarray(((0.0, -1.0), (0.1, 1.0))),
            0.0, 0.1, point=np.asarray((1.0, 0.0, 0.0)),
            tolerance=0.2, budget=budget,
        )
        changed = correspondence_owner.certify_curve_surface(
            source_curve, pcurve, patch, np.asarray(((0.0, -1.0), (0.1, 1.0))),
            0.0, 0.1,
            point=np.asarray((9.0, 8.0, 7.0) if with_curve else (1.0, 0.01, 0.0)),
            tolerance=0.2, budget=budget,
        )
    assert first.complete and changed.complete
    if with_curve:
        assert changed is first
        assert len(invocations) == 1
    else:
        assert changed is not first
        assert len(invocations) == 2


@pytest.mark.parametrize("premise", ("shared", "unequal-weights", "unequal-knots"))
def test_grouped_fitted_jets_preserve_individual_outward_bits_and_fallback(
    premise: str,
) -> None:
    controls = (
        ((0.0, 0.0, 0.0), (0.5, 0.75, 0.125), (1.0, 0.0, 0.25)),
        ((0.0, 0.0), (0.5, 0.75), (1.0, 0.0)),
        ((0.25, 0.0), (0.5, 0.25), (0.75, 0.5)),
    )

    def spline(side: int, points: Any) -> BSplineCurve:
        knots = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        if side == 2 and premise == "unequal-knots":
            points = (*points[:2], (0.625, 0.375), points[2])
            knots = (0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)
        weights = np.ones(len(points))
        weights[1] = 0.7 if side == 1 and premise == "unequal-weights" else 0.8
        return BSplineCurve(points, weights, knots, 2)

    curves = tuple(spline(side, points) for side, points in enumerate(controls))
    shared_knots = all(
        np.array_equal(np.asarray(curve.knots), np.asarray(curves[0].knots))
        for curve in curves[1:]
    )
    with coordinate_source_request(curves) as budget:
        selected = tuple(
            curve_pieces_for_interval(curve, 0.0, 1.0, coordinate_budget=budget)[0]
            for curve in curves
        )
    actual_budget = approximation_owner._Budget(100, 4, 1 << 20)
    individual_budget = approximation_owner._Budget(100, 4, 1 << 20)
    cache: dict[tuple[int, ...], Any] = {}
    actual = approximation_owner._fitted_cell_jets(
        selected, 0.125, 0.375, 0.25, shared_knots, cache, actual_budget
    )
    expected = approximation_owner._fitted_cell_jets(
        selected, 0.125, 0.375, 0.25, False, {}, individual_budget
    )
    for actual_location, expected_location in zip(actual, expected, strict=True):
        for actual_piece, expected_piece in zip(actual_location, expected_location, strict=True):
            for actual_jet, expected_jet in zip(actual_piece, expected_piece, strict=True):
                for side in (0, 1):
                    np.testing.assert_array_equal(actual_jet[side], expected_jet[side])
    assert individual_budget.jets == 6
    assert actual_budget.jets == (2 if premise == "shared" else 6)
    retained = actual_budget.bytes
    repeated = approximation_owner._fitted_cell_jets(
        selected, 0.125, 0.375, 0.25, shared_knots, cache, actual_budget
    )
    assert actual_budget.bytes == retained
    for first_location, second_location in zip(actual, repeated, strict=True):
        for first_piece, second_piece in zip(first_location, second_location, strict=True):
            for first_jet, second_jet in zip(first_piece, second_piece, strict=True):
                for side in (0, 1):
                    np.testing.assert_array_equal(first_jet[side], second_jet[side])


@pytest.mark.parametrize("covered", (False, True), ids=("overlap-only", "whole-range-covered"))
def test_correspondence_domain_factorization_requires_whole_pcurve_coverage(
    covered: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = BSplineCurve(
        ((1.0, 0.0, 0.0), (1.0, 0.05, 0.0), (0.995, 0.1, 0.0)),
        np.ones(3), (0.0, 0.0, 0.0, 0.1, 0.1, 0.1), 2,
    )
    pcurve = LineCurve((0.0, 0.0), (1.0, 0.0))
    patch = CylinderPatch(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), 1.0,
    )
    original = correspondence_owner._prepare_correspondence_workset
    boxes: list[np.ndarray] = []

    def prepare(*args: Any, **kwargs: Any) -> Any:
        boxes.append(np.array(args[3], copy=True))
        return original(*args, **kwargs)

    monkeypatch.setattr(correspondence_owner, "_prepare_correspondence_workset", prepare)
    broad = np.asarray(((-1.0, -1.0), (1.0, 1.0)))
    with coordinate_source_request((curve, pcurve, patch)) as budget:
        first = correspondence_owner.certify_curve_surface(
            curve, pcurve, patch, broad, 0.0, 0.1, point=np.zeros(3),
            tolerance=0.0002, maximum_intervals=64, budget=budget,
        )
        assert first.complete
        supplied = (
            pcurve.bounding_box(0.0, 0.1, budget=budget)
            if covered else np.asarray(((0.075, -1.0), (0.1, 1.0)))
        )
        second = correspondence_owner.certify_curve_surface(
            curve, pcurve, patch, supplied, 0.0, 0.1, point=np.ones(3),
            tolerance=first.deviation_bound, maximum_intervals=64, budget=budget,
        )
    np.testing.assert_array_equal(boxes[0], broad)
    if covered:
        assert second is first
        assert len(boxes) == 1
    else:
        assert second is not first
        assert len(boxes) == 2
        np.testing.assert_array_equal(boxes[1], supplied)


def test_correspondence_factorization_guards_tolerance_cap_and_partial_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = BSplineCurve(
        ((1.0, 0.0, 0.0), (1.0, 0.05, 0.0), (0.995, 0.1, 0.0)),
        np.ones(3), (0.0, 0.0, 0.0, 0.1, 0.1, 0.1), 2,
    )
    pcurve = LineCurve((0.0, 0.0), (1.0, 0.0))
    patch = CylinderPatch(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), 1.0,
    )
    original = correspondence_owner._execute_correspondence_workset
    invocations: list[None] = []

    def execute(*args: Any, **kwargs: Any) -> Any:
        invocations.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(correspondence_owner, "_execute_correspondence_workset", execute)
    with coordinate_source_request((curve, pcurve, patch)) as budget:
        def prove(
            tolerance: float, maximum_intervals: int = 64,
            source: BSplineCurve = curve,
        ) -> Any:
            return correspondence_owner.certify_curve_surface(
                source, pcurve, patch, np.asarray(((-1.0, -1.0), (1.0, 1.0))),
                0.0, 0.1, point=np.zeros(3), tolerance=tolerance,
                maximum_intervals=maximum_intervals, budget=budget,
            )

        failed = prove(1.0e-8)
        assert not failed.complete
        assert not budget.brep_correspondence_cache
        complete = prove(0.0002)
        assert complete.complete
        assert len(invocations) == 2
        assert prove(complete.deviation_bound) is complete
        assert len(invocations) == 2
        tighter = prove(0.5 * complete.deviation_bound)
        assert not tighter.complete
        assert len(invocations) == 3
        assert prove(complete.deviation_bound) is complete
        assert prove(0.0002, maximum_intervals=1) is not complete
        assert len(invocations) == 4
        points = np.array(curve.control_points, copy=True)
        points[0, 0] += 0.01
        changed = BSplineCurve(points, curve.weights, curve.knots, curve.degree)
        assert not prove(0.0002, source=changed).complete
        assert len(invocations) == 5
