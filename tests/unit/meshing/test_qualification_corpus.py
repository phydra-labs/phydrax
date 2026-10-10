#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Consumer-visible scientific-oracle boundaries for the native corpus tools."""

from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

import phydrax as phx
from tools._meshing_cases import meshing_case
from tools.meshing_qualification import (
    _bisection_p1_integral,
    _corpus_domain,
    _physical_error,
)


@pytest.mark.parametrize(
    ("dimension", "scale", "squared_reference_error"),
    (
        (2, 1.0, 11.0 / 180.0),
        (3, 1.0, 1.0 / 28.0),
        (2, 0.25, 11.0 / 180.0),
        (3, 2.0, 1.0 / 28.0),
    ),
    ids=("triangle", "tetrahedron", "scaled-triangle", "scaled-tetrahedron"),
)
def test_quadratic_manufactured_error_has_exact_simplex_integral(
    dimension: int,
    scale: float,
    squared_reference_error: float,
) -> None:
    points = scale * np.concatenate(
        (np.zeros((1, dimension), dtype=np.float64), np.eye(dimension, dtype=np.float64))
    )
    points += np.arange(dimension, dtype=np.float64) + 3.0
    cells = np.arange(dimension + 1, dtype=np.int32)[None, :]
    mesh = (
        phx.discretization.CellMesh.from_triangles(points, cells)
        if dimension == 2
        else phx.discretization.CellMesh.from_tetrahedra(points, cells)
    )
    nodal_values = np.sum(points * points, axis=1)
    # On the unit simplex, u_h=sum(x_i) and u=sum(x_i^2). The exact
    # monomial integral is prod(a_i!)/(sum(a_i)+d)!. Translation adds only
    # reproduced linear terms; scaling contributes scale^(d+4) to ||e||².
    expected = np.sqrt(squared_reference_error * scale ** (dimension + 4))
    assert _physical_error(mesh, nodal_values) == pytest.approx(
        expected, rel=1.0e-10, abs=1.0e-12
    )


def test_immersed_triangle_p1_inventory_uses_surface_measure() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    values = np.asarray((1.0, 2.0, 4.0), dtype=np.float64)
    assert _bisection_p1_integral(mesh, values) == pytest.approx(
        7.0 * np.sqrt(3.0) / 6.0, rel=1.0e-12
    )


@pytest.mark.parametrize("reverse_blocks", (False, True))
def test_multiblock_scalar_inventory_integrates_every_routed_tetrahedron(
    reverse_blocks: bool,
) -> None:
    simplex = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    points = np.concatenate(
        (simplex, 2.0 * simplex + np.asarray((3.0, 0.0, 0.0), dtype=np.float64))
    )
    blocks = (
        phx.discretization.CellBlock(
            "first",
            "tetrahedron",
            np.asarray(((0, 1, 2, 3),), dtype=np.int64),
            global_ids=np.asarray((101,), dtype=np.int64),
        ),
        phx.discretization.CellBlock(
            "second",
            "tetrahedron",
            np.asarray(((4, 5, 6, 7),), dtype=np.int64),
            global_ids=np.asarray((205,), dtype=np.int64),
        ),
    )
    mesh = phx.discretization.CellMesh(
        points,
        blocks[::-1] if reverse_blocks else blocks,
        vertex_global_ids=np.asarray((90, 12, 41, 8, 73, 62, 24, 35), dtype=np.int64),
    )
    # Integral of x+y+z: (1/6)*(3/4) + (8/6)*(18/4) = 49/8.
    values = np.sum(points, axis=1)
    assert _bisection_p1_integral(mesh, values) == pytest.approx(49.0 / 8.0, rel=1.0e-12)


def test_material_quadratic_error_respects_conductivity_and_continuous_trace() -> None:
    points = np.asarray(
        ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    values = 0.5 * points[:, 0] ** 2 + 0.5
    # On this right-material tetrahedron the interpolation error is
    # (xi-xi^2)/2. Integrating xi²-2xi³+xi⁴ gives 1/210, hence 1/840.
    assert _physical_error(mesh, values, material_x=True) == pytest.approx(
        np.sqrt(1.0 / 840.0), rel=1.0e-10, abs=1.0e-12
    )


def test_independent_material_cavity_domain_has_correct_oriented_volumes() -> None:
    domain = _corpus_domain(meshing_case("plc-material-cavity"))
    corners = np.asarray(domain.vertices)[np.asarray(domain.facets)]
    contributions = (
        np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0
    )
    adjacency = np.asarray(domain.facet_regions)
    volumes = {
        name: float(
            np.sum(contributions[adjacency[:, 0] == index])
            - np.sum(contributions[adjacency[:, 1] == index])
        )
        for index, name in enumerate(domain.region_ids)
    }
    assert volumes == pytest.approx({"left": 1.0, "right": 1.0 - 0.5**3}, rel=1.0e-12)


def test_design_surface_qoi_and_error_have_independent_triangle_moments() -> None:
    from tools.meshing_qualification import _design_qualification_physical

    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    values = space.dof_maps[0].dof_coordinates[:, 0]
    physical = _design_qualification_physical(result, space, values, 4)
    # The surface Jacobian is sqrt(3); ∫_T x^k = 1/((k+1)(k+2)).
    assert physical["field_L2_error"] == pytest.approx(
        np.sqrt(np.sqrt(3.0) / 60.0), rel=1.0e-12
    )
    assert physical["exact_integral"] == pytest.approx(np.sqrt(3.0) / 12.0, rel=1.0e-12)
    assert physical["QoI_absolute_error"] == pytest.approx(
        np.sqrt(3.0) / 12.0, rel=1.0e-12
    )
    assert physical["L2_quadrature_refinement_delta"] < 1.0e-14
    assert physical["QoI_quadrature_refinement_delta"] < 1.0e-14


@pytest.mark.parametrize(
    "name",
    ("plc-nonconvex", "plc-disconnected", "plc-thin-channel", "plc-triple-junction"),
)
def test_declared_solid_oracle_and_oriented_boundary_agree(name: str) -> None:
    case = meshing_case(name)
    domain = _corpus_domain(case)
    corners = np.asarray(domain.vertices)[np.asarray(domain.facets)]
    contributions = (
        np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])) / 6.0
    )
    pairs = np.asarray(domain.facet_regions)
    measured = {
        region: float(
            contributions[pairs[:, 0] == index].sum()
            - contributions[pairs[:, 1] == index].sum()
        )
        for index, (region, _) in enumerate(case.expected_regions)
    }
    assert measured == pytest.approx(dict(case.expected_regions), abs=1.0e-12)


def test_declared_material_oracle_does_not_use_x_threshold() -> None:
    from tools._meshing_cases import analytic_region_indices

    points = np.asarray(
        (
            (0.5, 0.5, 0.5),
            (1.5, 0.5, 0.5),
            (0.5, 1.5, 0.5),
            (1.5, 1.5, 0.5),
            (3.0, 3.0, 3.0),
        )
    )
    assert np.array_equal(
        analytic_region_indices(meshing_case("plc-triple-junction"), points),
        np.asarray((0, 1, 2, 2, -1)),
    )


def test_disconnected_oracle_rejects_gap_and_cavity_oracle_rejects_void() -> None:
    from tools._meshing_cases import analytic_region_indices

    assert (
        analytic_region_indices(
            meshing_case("plc-disconnected"), np.asarray(((1.5, 0.5, 0.5),))
        )[0]
        == -1
    )
    assert (
        analytic_region_indices(
            meshing_case("plc-material-cavity"), np.asarray(((1.5, 0.5, 0.5),))
        )[0]
        == -1
    )


@pytest.mark.parametrize(
    ("kind", "field"),
    (
        ("work", "maximum_work_units"),
        ("queries", "maximum_geometry_queries"),
        ("cavity", "maximum_cavity_cells"),
        ("entities", "maximum_vertices"),
        ("storage", "maximum_data_bytes"),
    ),
)
def test_refusal_corpus_changes_only_its_declared_allowance(
    kind: str, field: str
) -> None:
    from tools._meshing_cases import prepare_case, source_payload

    positive = meshing_case("plc-material-cavity")
    refusal = meshing_case(f"plc-{kind}-refusal")
    assert source_payload(positive) == source_payload(refusal)
    _, original, _ = prepare_case(positive, 4, 20000, 120.0)
    _, limited, _ = prepare_case(refusal, 4, 20000, 120.0)
    fields = (
        "maximum_vertices",
        "maximum_edges",
        "maximum_faces",
        "maximum_cells",
        "maximum_connectivity_entries",
        "maximum_data_bytes",
        "maximum_work_units",
        "maximum_cavity_cells",
        "maximum_geometry_queries",
        "maximum_scratch_bytes",
        "maximum_wall_seconds",
    )
    for name in fields:
        assert getattr(limited.limits, name) == (
            1 if name == field else getattr(original.limits, name)
        )


def test_phase_campaign_keeps_unmeasured_phases_unmeasured() -> None:
    from tools.meshing_benchmarks import corpus_phase_evidence

    refused = corpus_phase_evidence({"status": "expected-refusal"}, 20000)
    assert refused["status"] == "unmeasured"
    assert refused["compiler_epochs"] == []
    measured = corpus_phase_evidence(
        {
            "epochs": [
                {
                    "epoch": 0,
                    "retained_bytes": 321,
                    "compiler": {
                        "temporary_bytes": 17,
                        "output_bytes": 23,
                        "generated_code_bytes": 41,
                        "warm_samples_seconds": [0.2, 0.3],
                    },
                }
            ],
            "stages_seconds": {
                "solver_lowering_epoch_0": 0.4,
                "solver_compile_epoch_0": 0.5,
            },
        },
        20000,
    )["compiler_epochs"][0]
    assert measured["temporary_bytes"] == 17
    assert measured["logical_retained_bytes"] == 321
    assert measured["first_execution_seconds"] is None


@pytest.mark.parametrize(
    ("name", "inside", "outside"),
    (
        ("plc-small-angle", (8.0, 0.25, 0.5), (8.0, 1.0, 0.5)),
        ("plc-fixed-boundary", (0.5, 0.5, 0.5), (1.5, 0.5, 0.5)),
        ("plc-sliver-rich", (0.5, 0.5, 0.5), (0.5, 0.5, 1.5)),
    ),
)
def test_original_positive_declared_geometry_classification(
    name: str,
    inside: tuple[float, ...],
    outside: tuple[float, ...],
) -> None:
    from tools._meshing_cases import analytic_region_indices

    assert np.array_equal(
        analytic_region_indices(meshing_case(name), np.asarray((inside, outside))),
        np.asarray((0, -1)),
    )


def test_periodic_polyhedral_source_orbits_close_exactly_on_declared_cube() -> None:
    from tools._meshing_cases import prepare_polyhedral_case

    source, request, _ = prepare_polyhedral_case(
        "polyhedral-periodic-cube", 4, 20000, 120.0
    )
    points = np.asarray(source.complex.vertices)
    offsets = np.asarray(source.complex.polygon_offsets)
    values = np.asarray(source.complex.polygon_vertices)
    for constraint in request.periodic_constraints:
        source_face = int(np.asarray(constraint.source_scope.entity_ids)[0])
        target_face = int(np.asarray(constraint.target_scope.entity_ids)[0])
        first = points[values[offsets[source_face] : offsets[source_face + 1]]]
        second = points[values[offsets[target_face] : offsets[target_face + 1]]]
        matrix = np.asarray(constraint.transform)
        mapped = first @ matrix[:3, :3].T + matrix[:3, 3]
        assert set(map(tuple, mapped)) == set(map(tuple, second))
        normal_first = np.cross(first[1] - first[0], first[2] - first[0])
        normal_second = np.cross(second[1] - second[0], second[2] - second[0])
        assert np.dot(normal_first, normal_second) < 0.0


def test_periodic_prescribed_source_is_not_replaced_by_resolution_sites() -> None:
    from tools._meshing_cases import prepare_polyhedral_case

    original, original_request, _ = prepare_polyhedral_case(
        "polyhedral-periodic-cube", 4, 20000, 120.0, source=True
    )
    later, later_request, _ = prepare_polyhedral_case(
        "polyhedral-periodic-cube", 8, 20000, 120.0, source=False
    )
    assert original.binding_id == later.binding_id
    assert original_request.specification_id == later_request.specification_id
    assert np.array_equal(
        np.asarray(original.sites), np.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)))
    )
    assert np.array_equal(np.asarray(original.weights), np.asarray((0.0, 0.0)))


def test_thin_solid_retains_original_planar_source_bits_on_both_caps() -> None:
    from tools._meshing_cases import source_payload

    planar = np.asarray(
        source_payload(meshing_case("planar-thin-channel"))["vertices"], dtype=np.float64
    )
    solid = np.asarray(
        source_payload(meshing_case("plc-thin-channel"))["vertices"], dtype=np.float64
    )
    for height in (0.0, 1.0):
        cap = solid[solid[:, 2] == height, :2].copy()
        assert np.array_equal(cap.view(np.uint64), planar.view(np.uint64))


def test_original_thin_channel_oracle_rejects_the_nearby_bulk_gap() -> None:
    from tools._meshing_cases import analytic_region_indices

    points = np.asarray(
        (
            (0.2, 0.2, 0.5),
            (0.5, 0.2, 0.5),
            (0.8, 0.2, 0.5),
            (0.5, 0.18, 0.5),
            (0.5, 0.22, 0.5),
        )
    )
    assert np.array_equal(
        analytic_region_indices(meshing_case("plc-thin-channel"), points),
        np.asarray((0, 0, 0, -1, -1)),
    )


@pytest.mark.parametrize(
    ("name", "analytic_volume"),
    (
        ("plc-nonconvex", 3.0),
        ("plc-small-angle", 16.0),
        ("plc-fixed-boundary", 1.0),
        ("plc-sliver-rich", 1.0),
        ("plc-thin-channel", 0.324),
    ),
)
def test_feature_free_public_source_preparation_preserves_declared_solid(
    name: str,
    analytic_volume: float,
) -> None:
    from tools._meshing_cases import prepare_case

    source, _, _ = prepare_case(meshing_case(name), 4, 20000, 120.0)
    complex_ = source.complex
    points = np.asarray(complex_.vertices)
    offsets = np.asarray(complex_.polygon_offsets)
    values = np.asarray(complex_.polygon_vertices)
    contributions = []
    for first, last in zip(offsets[:-1], offsets[1:], strict=True):
        polygon = points[values[first:last]]
        contributions.extend(
            np.dot(polygon[0], np.cross(polygon[index], polygon[index + 1])) / 6.0
            for index in range(1, len(polygon) - 1)
        )
    assert sum(contributions) == pytest.approx(analytic_volume, abs=1.0e-12)


def test_corpus_registration_is_current_licensed_and_deterministic() -> None:
    from tools._meshing_cases import (
        corpus_registration_record,
        load_corpus_registration,
    )

    stored = load_corpus_registration()
    assert stored == corpus_registration_record()
    assert stored["registration_id"] == corpus_registration_record()["registration_id"]
    cases = stored["cases"]
    assert isinstance(cases, list)
    assert {case["expected_admission"] for case in cases} == {"positive", "refusal"}
    for case in cases:
        assert case["source_provenance"]["license"] == "LicenseRef-PHYDRA-Proprietary"
        assert case["source_artifact"]["sha256"]
        assert case["independent_reference"]["reference_digest"]


def test_cad_fixture_has_reviewed_provenance_and_content_identity() -> None:
    from tools._meshing_cases import load_cad_fixture

    fixture = load_cad_fixture()
    provenance = fixture["provenance"]
    reference = fixture["independent_reference"]
    assert isinstance(provenance, dict)
    assert isinstance(reference, dict)
    assert fixture["license"] == "LicenseRef-PHYDRA-Proprietary"
    assert provenance["method"] == "independently-authored analytic coordinates"
    assert reference["intersection_volume"] == 0.5


def test_frozen_request_record_retains_every_hard_resource_limit() -> None:
    from tools._meshing_cases import frozen_request_record, prepare_case

    case = meshing_case("planar-hole-feature")
    source, request, options = prepare_case(case, 4, 20000, 120.0)
    first = frozen_request_record(source, request, options)
    second = frozen_request_record(source, request, options)
    assert first == second
    assert first["frozen_request_id"] == second["frozen_request_id"]
    assert first["limits"] == {
        "maximum_vertices": 20000,
        "maximum_edges": 160000,
        "maximum_faces": 160000,
        "maximum_cells": 160000,
        "maximum_connectivity_entries": 1280000,
        "maximum_data_bytes": 20480000,
        "maximum_work_units": 640000,
        "maximum_cavity_cells": 20000,
        "maximum_geometry_queries": 1280000,
        "maximum_scratch_bytes": 81920000,
        "maximum_wall_seconds": 120.0,
        "limits_id": request.limits.limits_id,
    }


def test_mandatory_result_never_credits_refusal_as_positive() -> None:
    from tools.meshing_qualification import _mandatory_result

    profile = {
        "expected_outcome": "expected-refusal",
        "workflow": "resource-boundary",
        "scenario": None,
        "source_reference": {"source_digest": "source"},
    }
    assessment = {
        "positive_completion": True,
        "refusal_certified": True,
        "execution": "expected-refusal",
        "scientific_certification": "unassessed",
        "blocking_status": None,
        "release": "unassessed",
        "leadership": "unassessed",
    }
    result = _mandatory_result("bounded-refusal", profile, assessment)
    assert result["status"] == "refusal-certified"
    assert result["positive_completion"] is False
    assert result["refusal_certified"] is True


def test_routine_case_certification_does_not_complete_a_workflow() -> None:
    from tools.meshing_qualification import (
        _corpus_profile_assessment,
        _mandatory_result,
    )

    profile = {
        "expected_outcome": "passed",
        "workflow": "constrained-solid",
        "scenario": None,
        "full_workflow": False,
        "source_reference": {"source_digest": "source"},
    }
    assessment = _corpus_profile_assessment(
        profile,
        {
            "status": "passed",
            "case_contract_completion": "complete",
            "mandatory_workflow_completion": "incomplete",
            "independent_scientific_certification": "passed",
        },
    )
    assert assessment["qualification_eligible"] is True
    assert assessment["case_certified"] is True
    assert assessment["positive_completion"] is False
    result = _mandatory_result("routine-solid", profile, assessment)
    assert result["status"] == "case-certified"
    assert result["case_certified"] is True
    assert result["positive_completion"] is False


def test_mandatory_cli_result_is_strict_and_requires_scientific_success() -> None:
    from tools.meshing_qualification import (
        _finalize_mandatory_cli_record,
        _validate_mandatory_workflow_result,
    )

    arguments = argparse.Namespace(
        scenario="native-surface-pde", corpus_case="planar-hole-feature"
    )
    incomplete = _finalize_mandatory_cli_record(
        arguments,
        {
            "status": "passed",
            "mandatory_workflow_completion": "complete",
            "independent_scientific_certification": "unassessed",
        },
    )
    result = incomplete["workflow_result"]
    assert isinstance(result, dict)
    assert _validate_mandatory_workflow_result(result) == result
    assert result["status"] == "blocked"
    assert result["positive_completion"] is False

    complete = _finalize_mandatory_cli_record(
        arguments,
        {
            "status": "passed",
            "mandatory_workflow_completion": "complete",
            "independent_scientific_certification": "passed",
        },
    )
    qualified = complete["workflow_result"]
    assert isinstance(qualified, dict)
    assert _validate_mandatory_workflow_result(qualified) == qualified
    assert qualified["status"] == "positive-complete"
    assert qualified["positive_completion"] is True

    with pytest.raises(ValueError, match="unknown fields"):
        _validate_mandatory_workflow_result({**qualified, "unexpected": True})
    with pytest.raises(ValueError, match="identity"):
        _validate_mandatory_workflow_result({**qualified, "result_id": "tampered"})


def test_benchmark_sample_does_not_promote_bare_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.meshing_benchmarks as benchmarks
    from tools.meshing_qualification import _finalize_mandatory_cli_record

    blocked = _finalize_mandatory_cli_record(
        argparse.Namespace(
            scenario="native-surface-pde", corpus_case="planar-hole-feature"
        ),
        {
            "status": "passed",
            "mandatory_workflow_completion": "complete",
            "independent_scientific_certification": "unassessed",
        },
    )

    def completed(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["qualification"], returncode=0, stdout=json.dumps(blocked), stderr=""
        )

    monkeypatch.setattr(benchmarks.subprocess, "run", completed)
    sample = benchmarks._native_campaign_sample(
        "sphere-feature-surface", 1, 16, 1.0, 1, 0.05, 0, "native"
    )
    assert sample["status"] == "passed"
    assert sample["positive_completion"] is False
    assert sample["phase_evidence"]["status"] == "recorded"
    assert sample["resource_evidence"] == sample["phase_evidence"]["resources"]


def test_runtime_identity_binds_the_loaded_native_artifact() -> None:
    from phydrax._meshcore import (
        load_meshcore,
        meshcore_available,
        meshcore_runtime_identity,
    )
    from tools._meshing_cases import python_source_revision
    from tools.meshing_qualification import _runtime_identity

    if not meshcore_available():
        pytest.skip("Exact loaded native identity requires the configured meshcore.")
    expected = meshcore_runtime_identity()
    library = load_meshcore()
    runtime = _runtime_identity()
    native = runtime["meshcore_build"]
    build = runtime["build"]
    environment = runtime["environment"]
    assert isinstance(native, dict)
    assert isinstance(build, dict)
    assert isinstance(environment, dict)
    assert {
        name: native[name]
        for name in ("release", "source_digest", "binary_digest", "configuration")
    } == expected
    loaded_path = native["loaded_path"]
    assert isinstance(loaded_path, str)
    assert Path(loaded_path) == library.path.resolve()
    assert sha256(Path(loaded_path).read_bytes()).hexdigest() == native["binary_digest"]
    assert environment["native_configuration"] == native["configuration"]
    assert phx.__file__ is not None
    assert build["loaded_python_source_digest"] == python_source_revision(
        Path(phx.__file__).resolve().parent
    )


def test_isolated_campaign_timeout_retains_hard_launcher_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.meshing_benchmarks as benchmarks

    def expire(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise subprocess.TimeoutExpired(
            cmd=["qualification"], timeout=0.01, output=b"partial", stderr=b"blocked"
        )

    monkeypatch.setattr(benchmarks.subprocess, "run", expire)
    sample = benchmarks._native_campaign_sample(
        "planar-hole-feature", 1, 16, 0.01, 1, 0.05, 0, "native"
    )
    assert sample["status"] == "timeout"
    assert sample["positive_completion"] is False
    assert sample["hard_timeout"] == {
        "limit_seconds": 0.01,
        "mechanism": "subprocess.run-timeout",
        "completed_before_deadline": False,
        "terminated_by_launcher": True,
        "returncode": None,
    }
    assert sample["stdout"] == "partial"
    assert sample["stderr"] == "blocked"
    assert sample["phase_evidence"]["status"] == "recorded"
    assert sample["resource_evidence"] == sample["phase_evidence"]["resources"]


def test_registered_benchmark_corpus_preserves_missing_prerequisites() -> None:
    from tools.meshing_benchmarks import _benchmark_registered_native_corpus

    profiles = {
        "native-hybrid-lifecycle:archived-layer-core": {
            "case_name": "native-hybrid-lifecycle",
            "workflow": "hybrid-layers",
            "scenario": "native-hybrid-lifecycle",
            "controls": {"hybrid_profile": "archived-layer-core"},
            "prerequisites": ("hybrid_archive", "hybrid_content_id"),
            "source_reference": {
                "kind": "owning-scenario-source-descriptor",
                "source_digest": "frozen-source",
            },
        },
    }
    arguments = argparse.Namespace(
        corpus_profile=None,
        hybrid_archive=None,
        hybrid_content_id=None,
        distributed_checkpoint_root=None,
        resolution=[4],
        capacity=[20000],
        timeout=120.0,
        repeats=3,
        target_error=0.05,
        adaptation_rounds=1,
        comparison=None,
    )
    first = _benchmark_registered_native_corpus(arguments, profiles)
    second = _benchmark_registered_native_corpus(arguments, profiles)
    assert first == second
    records = first["profiles"]
    assert isinstance(records, dict)
    record = records["native-hybrid-lifecycle:archived-layer-core"]
    assert isinstance(record, dict)
    assert record["status"] == "blocked-missing-input"
    assert record["missing_inputs"] == ["hybrid_archive", "hybrid_content_id"]
    assert record["campaigns"] == []


def test_distributed_benchmark_attempts_have_fresh_checkpoint_roots() -> None:
    from tools.meshing_benchmarks import _native_campaign_sample

    sample = _native_campaign_sample(
        "native-distributed-lifecycle",
        4,
        20000,
        120.0,
        3,
        0.05,
        1,
        "triangle",
        scenario_override="native-distributed-lifecycle",
        controls={"distributed_checkpoint_root": "/shared/frozen-campaign"},
        attempt_index=2,
    )
    frozen = sample["frozen_controls"]
    assert isinstance(frozen, dict)
    controls = frozen["scenario_controls"]
    assert isinstance(controls, dict)
    assert controls["distributed_checkpoint_root"] == (
        "/shared/frozen-campaign-resolution-4-capacity-20000-attempt-2"
    )
    assert sample["process_started"] is False
    assert sample["phase_evidence"]["status"] == "unmeasured"


def test_mandatory_benchmark_campaign_registration_is_current() -> None:
    from tools._meshing_cases import MANDATORY_WORKFLOWS
    from tools.meshing_benchmarks import load_native_campaign_registration

    record = load_native_campaign_registration()
    workflows = record["workflows"]
    assert isinstance(workflows, list)
    assert [workflow["workflow"] for workflow in workflows] == list(MANDATORY_WORKFLOWS)
    assert all(workflow["expected_admission"] == "positive" for workflow in workflows)
    assert all(
        workflow["refusal_counts_as_positive_completion"] is False
        for workflow in workflows
    )
    assert record["contains_benchmark_results"] is False
