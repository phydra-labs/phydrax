from dataclasses import replace
from typing import get_args

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax.geometry.brep import (
    BRepGeometry,
    BRepModel,
    BRepTopology,
    BRepValidationCheck,
    BRepValidationError,
    BRepValidationFinding,
    BRepValidationReport,
    PlanePatch,
)
from tests._support.cad_models import native_solid


@pytest.fixture(scope="module")
def box() -> BRepModel:
    return native_solid("box")


def test_construction_collects_independent_phase_failures(box: BRepModel) -> None:
    vertices = np.asarray(box.mesh_vertices).copy()
    vertices[0, 0] = np.nan
    parameters = np.asarray(box.triangle_parameters).copy()
    parameters[0, 0, 0] = np.inf
    negative_bounds = -np.ones(box.mesh_faces.shape[0], dtype=np.float64)
    with pytest.raises(BRepValidationError) as refused:
        box.evolve(
            orientation=np.zeros(len(box.patches), dtype=np.float64),
            mesh_vertices=vertices,
            triangle_parameters=parameters,
            tessellation_deviation_bounds=negative_bounds,
            vertex_occurrence_ids=np.full(
                box.mesh_vertices.shape[0], 100, dtype=np.int32
            ),
        )
    report = refused.value.report
    assert not report.valid
    assert report.findings[0].code == "orientation"
    codes = {finding.code for finding in report.findings}
    assert {
        "orientation",
        "mesh-vertices",
        "triangle-parameters",
        "tessellation-bounds",
        "occurrence-lineage",
    } <= codes
    assert all(
        finding.code in get_args(BRepValidationCheck.__value__)
        for finding in report.findings
    )
    assert all(finding.measured > finding.limit for finding in report.findings)


def test_construction_collects_bad_coedges_on_multiple_faces(box: BRepModel) -> None:
    # A binary64-representable power-of-two translation exceeds the declared
    # curve/surface length limit without changing topology or UV closure checks.
    displacement = 2.0**-16
    patches = list(box.patches)
    for face in (0, 1):
        patch = patches[face]
        if not isinstance(patch, PlanePatch):
            raise TypeError(f"box.patch must be PlanePatch; got {type(patch).__name__}")
        patches[face] = PlanePatch(
            np.asarray(patch.origin) + (displacement, displacement, displacement),
            patch.first_axis,
            patch.second_axis,
        )
    with pytest.raises(BRepValidationError) as refused:
        box.evolve(patches=tuple(patches), coedge_deviation_bounds=None)
    findings = [
        finding
        for finding in refused.value.report.findings
        if finding.code == "curve-surface"
    ]
    expected = sum(
        len(loop) for loops in box.require_geometry().face_loops[:2] for loop in loops
    )
    assert len(findings) == expected
    assert {finding.entity for finding in findings} == {box.face_ids[0], box.face_ids[1]}
    assert all(finding.measured >= displacement for finding in findings)
    assert all(
        finding.limit == box.report.curve_surface_tolerance for finding in findings
    )


def test_validation_report_cannot_soft_pass_findings() -> None:
    finding = BRepValidationFinding(
        "orientation", None, 1.0, 0.0, "orientation must be signed; got zero"
    )
    report = BRepValidationReport(findings=(finding,))
    with pytest.raises(BRepValidationError) as refused:
        report.require_valid()
    assert refused.value.report is report
    assert report.findings == (finding,)


def test_validation_check_vocabulary_is_closed() -> None:
    with pytest.raises(ValueError, match="BRepValidationFinding.code"):
        BRepValidationFinding("misspelled", None, 1.0, 0.0, "invalid")  # ty: ignore[invalid-argument-type]


def test_model_evolution_preserves_scientific_identity_for_source_name(
    box: BRepModel,
) -> None:
    changed = box.evolve(report=replace(box.report, source_id="renamed source"))
    assert changed.source_id == "renamed source"
    assert changed.model_id == box.model_id
    assert changed.tessellation_id == box.tessellation_id
    assert changed.source_revision == box.source_revision
    assert changed.require_geometry() is box.require_geometry()


def test_model_geometry_accessor_refuses_external_comparison(box: BRepModel) -> None:
    external = box.evolve(
        geometry=None,
        coedge_deviation_bounds=None,
        triangle_occurrence_ids=None,
        vertex_occurrence_ids=None,
    )
    with pytest.raises(ValueError, match=r"BRepModel\.geometry"):
        external.require_geometry()


def test_geometry_construction_collects_root_loop_and_shell_failures(
    box: BRepModel,
) -> None:
    geometry = box.require_geometry()
    vertices = ((-1, geometry.edge_vertices[0][1]), *geometry.edge_vertices[1:])
    senses = (0, *geometry.coedge_senses[1:])
    shell_signs = (
        (0, *geometry.shell_orientations[0][1:]),
        *geometry.shell_orientations[1:],
    )
    with pytest.raises(BRepValidationError) as refused:
        type(geometry)(
            vertex_points=geometry.vertex_points,
            curves=geometry.curves,
            edge_curves=geometry.edge_curves,
            edge_ranges=geometry.edge_ranges,
            edge_vertices=vertices,
            pcurves=geometry.pcurves,
            coedge_edges=geometry.coedge_edges,
            coedge_senses=senses,
            face_loops=geometry.face_loops,
            shell_faces=geometry.shell_faces,
            shell_orientations=shell_signs,
            solid_shells=geometry.solid_shells,
            occurrences=geometry.occurrences,
            assembly_containers=geometry.assembly_containers,
        )
    assert {"root-binding", "topology", "shell-incidence"} <= {
        finding.code for finding in refused.value.report.findings
    }


def test_topology_refusal_is_not_lowered_to_a_cad_coedge_limit(box: BRepModel) -> None:
    source = box.topology
    topology = BRepTopology(
        face_edges=(source.face_edges[0][::-1], *source.face_edges[1:]),
        edge_faces=source.edge_faces,
        face_wires=source.face_wires,
        solid_faces=source.solid_faces,
        solid_face_orientations=source.solid_face_orientations,
        num_vertices=source.num_vertices,
    )
    with pytest.raises(BRepValidationError) as construction:
        box.evolve(topology=topology)
    report = construction.value.report
    assert report.findings[0].code == "topology"
    with pytest.raises(BRepValidationError) as classification:
        report.require_coedge_failure()
    assert classification.value.report is report
    assert classification.value.report.findings[0].code == "topology"


def test_coedge_refusal_retains_unresolved_intervals_and_work(box: BRepModel) -> None:
    intervals = ((0.0, 1.0),)
    finding = BRepValidationFinding(
        "curve-surface",
        box.face_ids[0],
        np.inf,
        box.report.curve_surface_tolerance,
        "coedge must resolve; got an unproved interval",
        coedge=0,
        unresolved=intervals,
        intervals_processed=65_536,
    )
    report = BRepValidationReport(findings=(finding,))
    diagnostic = report.require_coedge_failure()
    assert diagnostic.face == 0
    assert diagnostic.coedge == 0
    assert diagnostic.unresolved == intervals
    assert diagnostic.intervals_processed == 65_536


def test_stored_model_reconstruction_preserves_scientific_content(box: BRepModel) -> None:
    restored = BRepModel.from_stored_fields(box)
    assert restored.model_id == box.model_id
    assert restored.tessellation_id == box.tessellation_id
    assert restored.source_revision == box.source_revision
    np.testing.assert_array_equal(restored.mesh_vertices, box.mesh_vertices)
    np.testing.assert_array_equal(restored.triangle_parameters, box.triangle_parameters)
    np.testing.assert_array_equal(
        restored.coedge_deviation_bounds, box.coedge_deviation_bounds
    )


def test_stored_geometry_reconstruction_recomputes_bounds_and_identity(
    box: BRepModel,
) -> None:
    geometry = box.require_geometry()
    restored = BRepGeometry.from_stored_fields(geometry)
    assert restored.geometry_id == geometry.geometry_id
    assert restored.face_loops == geometry.face_loops
    assert restored.edge_endpoint_roots == geometry.edge_endpoint_roots
    assert restored.coedge_endpoint_roots == geometry.coedge_endpoint_roots
    np.testing.assert_array_equal(restored.vertex_points, geometry.vertex_points)
    np.testing.assert_array_equal(
        restored.vertex_evaluation_bounds, geometry.vertex_evaluation_bounds
    )
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(geometry)
    assert any(
        leaf is restored.vertex_evaluation_bounds
        for leaf in jax.tree_util.tree_leaves(restored)
    )
    # Constructor-input replacement recomputes the derived bounds and identity.
    replaced = replace(geometry, shell_orientations=geometry.shell_orientations)
    assert replaced.geometry_id == geometry.geometry_id
    np.testing.assert_array_equal(
        replaced.vertex_evaluation_bounds, geometry.vertex_evaluation_bounds
    )


def test_stored_topology_reconstruction_recomputes_reverse_incidence(
    box: BRepModel,
) -> None:
    topology = box.topology
    restored = BRepTopology.from_stored_fields(topology)
    assert restored.face_edges == topology.face_edges
    assert restored.edge_faces == topology.edge_faces
    assert restored.face_wires == topology.face_wires
    assert restored.solid_faces == topology.solid_faces
    assert restored.solid_face_orientations == topology.solid_face_orientations
    assert restored.face_solids == topology.face_solids
    assert restored.num_vertices == topology.num_vertices


def test_stored_model_reconstruction_refuses_invalid_numerical_fields(
    box: BRepModel,
) -> None:
    corrupted = eqx.tree_at(
        lambda value: value.orientation,
        box,
        np.zeros_like(np.asarray(box.orientation)),
    )
    with pytest.raises(BRepValidationError) as refused:
        BRepModel.from_stored_fields(corrupted)
    assert refused.value.report.findings[0].code == "orientation"


def test_stored_model_reconstruction_requires_its_node_type(box: BRepModel) -> None:
    with pytest.raises(TypeError, match=r"BRepModel\.from_stored_fields.*value"):
        BRepModel.from_stored_fields(box.require_geometry())  # ty: ignore[invalid-argument-type]
