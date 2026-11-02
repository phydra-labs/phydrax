import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry import edit_brep_occurrence_placement
from phydrax.geometry.brep import (
    brep_box,
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
    BRepEntityId,
    BRepGeometry,
    BRepModel,
    BRepPlacementEvidence,
    cad_revision_from_brep_model,
)
from phydrax.geometry.brep._constructors import BRepTessellationPolicy
from phydrax.geometry.brep._partition import _assembly_namespace_occurrence_id
from phydrax.geometry.brep._placement import entity_occurrence
from phydrax.geometry.brep._tessellation import tessellate_brep_model
from phydrax.interchange._cad_archive import save_brep_archive
from phydrax.meshing._model_codec import read_model_archive, write_model_archive


def _shared_model() -> BRepModel:
    box = brep_box(
        (0, 0, 0), (1, 1, 1), coordinate_contract=SpatialCoordinateContract.si()
    )
    source = box.geometry
    graph = BRepAssemblyGraph(
        (
            BRepAssemblyUse("first", definition_id="shared"),
            BRepAssemblyUse("second", definition_id="shared"),
        ),
        (BRepAssemblyDefinition("shared", (BRepAssemblyUse("leaf", solid_index=0),)),),
        1,
    )
    geometry = BRepGeometry(
        vertex_points=source.vertex_points,
        vertex_exact_points=source.vertex_exact_points,
        vertex_roots=source.vertex_roots,
        edge_endpoint_roots=source.edge_endpoint_roots,
        coedge_endpoint_roots=source.coedge_endpoint_roots,
        edge_vertex_gap_bounds=source.edge_vertex_gap_bounds,
        edge_face_gap_bounds=source.edge_face_gap_bounds,
        curves=source.curves,
        edge_curves=source.edge_curves,
        edge_ranges=source.edge_ranges,
        edge_vertices=source.edge_vertices,
        pcurves=source.pcurves,
        coedge_edges=source.coedge_edges,
        coedge_senses=source.coedge_senses,
        face_loops=source.face_loops,
        shell_faces=source.shell_faces,
        shell_orientations=source.shell_orientations,
        solid_shells=source.solid_shells,
        assembly_graph=graph,
    )
    return BRepModel(
        geometry=geometry,
        patches=box.patches,
        parameter_bounds=box.parameter_bounds,
        orientation=box.orientation,
        physical_tags=box.physical_tags,
        coordinate_contract=box.coordinate_contract,
        report=box.report,
        trim_domains=box.trim_domains,
    )


def test_exact_edit_maps_every_qualified_entity_and_namespace() -> None:
    model = _shared_model()
    original_graph = model.geometry.assembly_graph
    original_revision = model.source_revision
    evidence = BRepPlacementEvidence(0.0, 1.0, 0.0)
    reflection = BRepAffinePlacement.from_matrix(
        ((Fraction(-1, 3), 0, 0, 2), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )
    result = edit_brep_occurrence_placement(
        model, ("first", "leaf"), reflection, evidence=evidence
    )
    target = result.model
    assert model.source_revision == original_revision
    assert model.geometry.assembly_graph is original_graph
    assert (
        tuple(original_graph.iter_occurrences())[0].placement
        == BRepAffinePlacement.identity()
    )
    occurrences = {item.path: item for item in target.geometry.iter_occurrences()}
    assert set(occurrences) == {("first", "leaf"), ("second", "leaf")}
    assert occurrences[("first", "leaf")].placement == reflection
    assert occurrences[("first", "leaf")].placement_evidence is not None
    assert occurrences[("second", "leaf")].placement == BRepAffinePlacement.identity()
    assert target.source_revision != original_revision
    assert target.tessellation is None
    assert target.binding_proof.source_revision == target.source_revision
    graph = result.association_graph
    assert graph.source_revision == cad_revision_from_brep_model(model)
    assert graph.target_revision == cad_revision_from_brep_model(target)
    transaction = graph.transaction
    assert (
        transaction.coverage.source_exhaustive and transaction.coverage.target_exhaustive
    )
    assert not transaction.unresolved_source_ids and not transaction.unresolved_target_ids
    edges = {edge.source_occurrence_id: edge for edge in transaction.correspondences}
    assert set(edges) == {
        item.occurrence_id for item in graph.source_revision.occurrences
    }
    assert {edge.target_occurrence_id for edge in edges.values()} == {
        item.occurrence_id for item in graph.target_revision.occurrences
    }
    assert all(edge.evidence_id == result.certificate_id for edge in edges.values())
    for record in model.geometry.qualified_entity_incidence(
        model.source_id, model.source_revision
    ):
        entity = record.member
        edge = edges[entity_occurrence(entity)]
        mapped = BRepEntityId(
            target.source_id,
            target.source_revision,
            entity.kind,
            entity.index,
            entity.occurrence_path,
        )
        assert edge.target_occurrence_id == entity_occurrence(mapped)
        assert edge.orientation_transform == 1
    for path, sign in (
        (("first",), 1),
        (("first", "leaf"), -1),
        (("second",), 1),
        (("second", "leaf"), 1),
    ):
        identifier = _assembly_namespace_occurrence_id(
            model.source_id, model.source_revision, path
        )
        assert edges[identifier].orientation_transform == 1
        assert dict(result.placement_orientation_changes)[path] == sign


def test_container_edit_rebases_all_descendants_and_rejects_bad_path() -> None:
    model = _shared_model()
    reflection = BRepAffinePlacement.from_matrix(
        ((-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )
    result = edit_brep_occurrence_placement(model, ("first",), reflection)
    edges = {
        edge.source_occurrence_id: edge
        for edge in result.association_graph.transaction.correspondences
    }
    for path, sign in (
        (("first",), -1),
        (("first", "leaf"), -1),
        (("second",), 1),
        (("second", "leaf"), 1),
    ):
        identifier = _assembly_namespace_occurrence_id(
            model.source_id, model.source_revision, path
        )
        assert edges[identifier].orientation_transform == 1
        assert dict(result.placement_orientation_changes)[path] == sign
    with pytest.raises(ValueError, match="path"):
        edit_brep_occurrence_placement(model, ("first", "missing"), reflection)


def test_edit_refuses_stale_source() -> None:
    model = _shared_model()
    object.__setattr__(model, "source_revision", "stale")
    with pytest.raises(ValueError, match="changed"):
        edit_brep_occurrence_placement(model, ("first",), BRepAffinePlacement.identity())


def test_edit_discards_prior_tessellation_and_source_proof() -> None:
    source = brep_box(
        (0, 0, 0), (1, 1, 1), coordinate_contract=SpatialCoordinateContract.si()
    )
    tessellation, certificate = tessellate_brep_model(
        source,
        BRepTessellationPolicy(
            linear_deflection=0.02, angular_deflection=0.2, trim_samples_per_edge=5
        ),
    )
    model = BRepModel(
        geometry=source.geometry,
        patches=source.patches,
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        physical_tags=source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        report=source.report,
        trim_domains=source.trim_domains,
        tessellation=tessellation,
        tessellation_certificate=certificate,
    )
    translation = BRepAffinePlacement.from_matrix(
        ((1, 0, 0, 2), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )
    result = edit_brep_occurrence_placement(model, ("solid-0",), translation)
    assert model.tessellation is tessellation
    assert result.model.tessellation is None
    assert (
        result.model.binding_proof.source_revision != model.binding_proof.source_revision
    )


def test_edited_shared_ancestry_cold_archive_and_tamper(tmp_path: Path) -> None:
    model = _shared_model()
    placement = BRepAffinePlacement.from_matrix(
        ((Fraction(-1, 3), 0, 0, 2), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )
    evidence = BRepPlacementEvidence(0.0, 1.0, 0.0)
    edited = edit_brep_occurrence_placement(
        model, ("first", "leaf"), placement, evidence=evidence
    ).model
    archive = tmp_path / "edited.phx"
    save_brep_archive(edited, archive)
    reader = """
import sys
from fractions import Fraction
from phydrax.interchange._cad_archive import load_brep_archive
from phydrax.geometry.brep import BRepAffinePlacement, cad_revision_from_brep_model
model = load_brep_archive(sys.argv[1])
model.require_current()
assert model.source_revision == sys.argv[2]
assert model.tessellation is None
occurrences = {item.path: item for item in model.geometry.iter_occurrences()}
assert set(occurrences) == {("first", "leaf"), ("second", "leaf")}
assert occurrences[("first", "leaf")].placement.matrix[0][0] == Fraction(-1, 3)
assert occurrences[("first", "leaf")].placement_evidence is not None
assert occurrences[("second", "leaf")].placement == BRepAffinePlacement.identity()
assert cad_revision_from_brep_model(model).revision_id == model.source_revision
"""
    restored = subprocess.run(
        [sys.executable, "-c", reader, str(archive), edited.source_revision],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert restored.returncode == 0, restored.stderr

    manifest, arrays = read_model_archive(archive, limits=None)

    def tamper_placement(node: object) -> bool:
        if isinstance(node, dict):
            if (
                node.get("kind") == "fraction"
                and node.get("numerator") == -1
                and node.get("denominator") == 3
            ):
                node["numerator"] = -2
                return True
            return any(tamper_placement(child) for child in node.values())
        if isinstance(node, list):
            return any(tamper_placement(child) for child in node)
        return False

    assert tamper_placement(manifest["model_recipe"])
    corrupt = tmp_path / "edited-tampered.phx"
    write_model_archive(corrupt, manifest=manifest, arrays=arrays, limits=None)
    refused = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from phydrax.interchange._cad_archive import load_brep_archive; "
            "load_brep_archive(sys.argv[1])",
            str(corrupt),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0


def test_edited_model_refuses_changed_assembly_source() -> None:
    source = _shared_model()
    placement = BRepAffinePlacement.from_matrix(
        ((1, 0, 0, 2), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )
    edited = edit_brep_occurrence_placement(source, ("first", "leaf"), placement).model
    object.__setattr__(edited.geometry, "assembly_graph", source.geometry.assembly_graph)
    with pytest.raises(ValueError, match="changed"):
        edited.require_current()
    with pytest.raises(ValueError, match="changed"):
        edit_brep_occurrence_placement(edited, ("second", "leaf"), placement)
