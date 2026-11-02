from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellMesh, SimplicialConnectivity
from phydrax.geometry._cad_revision import CADSelectionSet, CADSelector
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep import brep_box
from phydrax.geometry.brep._assembly import BRepAssemblyGraph, BRepAssemblyUse
from phydrax.geometry.brep._model import BRepEntityId, BRepGeometry, BRepModel
from phydrax.geometry.brep._partition import (
    _entity_occurrence_id,
    BRepPartitionOperand,
    BRepPartitionPlan,
    BRepPartitionPolicy,
    BRepPartitionRole,
    cad_revision_from_brep_model,
    partition_brep,
)
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.meshing import (
    CellMeshingResult,
    MeshingScope,
    PatchControl,
    RegionControl,
    RegionRole,
)
from phydrax.meshing._association import GeometryAssociationKind
from phydrax.meshing._model_codec import (
    MODEL_ARCHIVE_LIMITS,
    read_model_artifact,
    restore_model_archive_value,
    write_model_artifact,
)


class _SharedFaceCase(NamedTuple):
    model: BRepModel
    face: int
    selectors: tuple[CADSelector, ...]
    scopes: tuple[MeshingScope, ...]
    members: frozenset[BRepEntityId]


def _shared_face_case(order: tuple[float, float]) -> _SharedFaceCase:
    coordinates = SpatialCoordinateContract.si()
    operands = tuple(
        BRepPartitionOperand(
            f"region-{index}",
            brep_box(
                (x, 0.0, 0.0),
                (x + 1.0, 1.0, 1.0),
                coordinate_contract=coordinates,
            ),
            BRepPartitionRole.REGION,
        )
        for index, x in enumerate(order)
    )
    partition = partition_brep(
        BRepPartitionPlan(
            coordinates, operands, BRepPartitionPolicy(("region-0", "region-1"))
        )
    )
    model = partition.model
    shared = tuple(
        index
        for index, owners in enumerate(model.topology.face_solids)
        if len(owners) == 2
    )
    assert len(shared) == 1
    revision = cad_revision_from_brep_model(model)
    members = frozenset(
        record.member
        for record in model.geometry.qualified_entity_incidence(
            model.source_id, model.source_revision
        )
        if record.member.kind == "face" and record.member.index == shared[0]
    )
    assert len(members) == 2
    selectors = tuple(
        revision.select(_entity_occurrence_id(member)) for member in members
    )
    scopes = tuple(
        MeshingScope.from_geometry(
            model, CADSelectionSet(revision.revision_id, (selector,))
        )
        for selector in selectors
    )
    return _SharedFaceCase(model, shared[0], selectors, scopes, members)


@pytest.fixture(
    scope="module",
    params=((0.0, 1.0), (1.0, 0.0)),
    ids=("forward", "reversed"),
)
def shared_face(request: pytest.FixtureRequest) -> _SharedFaceCase:
    return _shared_face_case(request.param)


def _specifications(
    case: _SharedFaceCase,
) -> tuple[phx.meshing.VolumeMeshingSpec, phx.meshing.VolumeMeshingSpec]:
    model, _, selectors, scopes, _ = case
    revision = cad_revision_from_brep_model(model)
    complete = MeshingScope.from_geometry(
        model, CADSelectionSet(revision.revision_id, selectors)
    )
    regions = tuple(
        RegionControl(
            MeshingScope.from_geometry(model, entity),
            f"region-{index}",
            f"material-{index}",
            RegionRole.SOLID,
        )
        for index, entity in enumerate(model.solid_ids)
    )
    boundary = MeshingScope.whole(model, 2)
    target = phx.meshing.CellMeshingTarget(
        3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
    )

    def specification(scope: MeshingScope) -> phx.meshing.VolumeMeshingSpec:
        return phx.meshing.VolumeMeshingSpec(
            target,
            boundary,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=(phx.meshing.UniformSizeControl(boundary, 0.5),),
            region_controls=regions,
            patch_controls=(
                PatchControl("interface", scope, ("region-0", "region-1")),
            ),
            maximum_deviation=0.05,
        )

    return specification(complete), specification(scopes[0])


@pytest.fixture(scope="module")
def shared_volume(shared_face: _SharedFaceCase) -> CellMeshingResult:
    specification, _ = _specifications(shared_face)
    return (
        phx.meshing.NativeMeshingProvider()
        .plan(shared_face.model, specification)
        .execute()
    )


def _composition(
    model: BRepModel, selected: MeshingScope, *, include_side: bool = True
) -> Composition:
    source = CompositionEntry(
        model,
        entry_id="cad/source",
        role="model-state",
        owner_id=model.source_id,
        structure_id=model.model_id,
        revision_id=model.source_revision,
        semantics_id="brep-source",
    )
    side = CompositionEntry(
        selected,
        entry_id="cad/side",
        role="model-state",
        owner_id=model.source_id,
        structure_id=selected.scope_id,
        revision_id=selected.source_revision,
        semantics_id="cad-face-side",
        dependencies=(source.binding("revision"),),
    )
    patch = PatchControl("interface", selected, ("left", "right"))
    patch_entry = CompositionEntry(
        patch,
        entry_id="cad/patch",
        role="model-state",
        owner_id=model.source_id,
        structure_id=patch.control_id,
        revision_id=selected.source_revision,
        semantics_id="cad-patch-control",
        dependencies=(source.binding("revision"),),
    )
    entries = (source, side, patch_entry) if include_side else (source, patch_entry)
    return Composition(entries, boundary_id="cad-side-archive")


def test_shared_face_side_identity_and_set_algebra(shared_face: _SharedFaceCase) -> None:
    model, face, selectors, scopes, members = shared_face
    assert len({selector.path for selector in selectors}) == 2
    assert scopes[0].scope_id != scopes[1].scope_id
    np.testing.assert_array_equal(scopes[0].entity_ids, scopes[1].entity_ids)
    assert all(len(scope.side_records) == 1 for scope in scopes)
    assert {scope.side_records[0].face.occurrence_id for scope in scopes} == {
        selector.occurrence_id for selector in selectors
    }
    assert len({scope.side_records[0].solid.occurrence_id for scope in scopes}) == 2
    assert {scope.side_records[0].orientation for scope in scopes} == {-1, 1}
    revision = cad_revision_from_brep_model(model)
    complete = MeshingScope.from_geometry(
        model, CADSelectionSet(revision.revision_id, selectors)
    )
    assert scopes[0].union(scopes[1]).scope_id == complete.scope_id
    assert complete.difference(scopes[0]).scope_id == scopes[1].scope_id
    with pytest.raises(ValueError, match="empty"):
        scopes[0].intersection(scopes[1])
    whole = MeshingScope.from_geometry(model, model.face_ids[face])
    assert whole.intersection(scopes[0]).scope_id == scopes[0].scope_id
    assert scopes[0].intersection(whole).scope_id == scopes[0].scope_id
    with pytest.raises(ValueError, match="empty"):
        scopes[0].difference(whole)
    with pytest.raises(ValueError):
        whole.difference(scopes[0])
    mixed = whole.union(scopes[0])
    assert mixed.unqualified_entity_ids == whole.unqualified_entity_ids
    assert mixed.side_records == scopes[0].side_records
    assert mixed.scope_id != whole.scope_id
    domain = MeshingDomain.from_brep(model)
    rows = tuple(
        row for row, index in enumerate(domain.source_indices[2]) if index == face
    )
    assert len(rows) == 1
    assert domain.source_occurrences[2][rows[0]] in {
        member.occurrence_path for member in members
    }
    assert (
        PatchControl("interface", complete, ("left", "right")).scope.scope_id
        == complete.scope_id
    )


def test_sided_scope_and_patch_restore_cold(
    shared_face: _SharedFaceCase, tmp_path: Path
) -> None:
    model, _, _, scopes, _ = shared_face
    path = write_model_artifact(
        tmp_path / "side.phx",
        _composition(model, scopes[0]),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from phydrax.lifecycle import Composition; "
                "from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, "
                "read_model_artifact, restore_model_archive_value; "
                "c=read_model_artifact(sys.argv[1], expected_type=Composition, "
                "restore=restore_model_archive_value, limits=MODEL_ARCHIVE_LIMITS); "
                "assert c.value('cad/source').source_revision == sys.argv[2]; "
                "assert c.value('cad/side').scope_id == sys.argv[3]; "
                "assert c.value('cad/side').side_records[0].face.occurrence_id == sys.argv[4]; "
                "assert c.value('cad/patch').scope.scope_id == sys.argv[3]"
            ),
            str(path),
            model.source_revision,
            scopes[0].scope_id,
            scopes[0].side_records[0].face.occurrence_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize(
    "include_side", (True, False), ids=("scope-and-patch", "patch-only")
)
def test_rehashed_sided_archive_tampering_is_refused(
    shared_face: _SharedFaceCase, tmp_path: Path, include_side: bool
) -> None:
    model, _, _, scopes, _ = shared_face
    side = scopes[0].side_records[0]
    forged = MeshingScope._from_parts(
        scopes[0].source_id,
        scopes[0].source_revision,
        scopes[0].entity_kind,
        scopes[0].entity_dimension,
        scopes[0].entity_set_id,
        scopes[0].unqualified_entity_ids,
        (dataclasses.replace(side, orientation=-side.orientation),),
    )
    path = write_model_artifact(
        tmp_path / "forged.phx",
        _composition(model, forged, include_side=include_side),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    with pytest.raises((ArrayArchiveCorruptionError, ValueError)):
        read_model_artifact(
            path,
            expected_type=Composition,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
        )


def test_sided_patch_restoration_requires_current_source(
    shared_face: _SharedFaceCase,
) -> None:
    model, _, _, scopes, _ = shared_face
    _, spec = _specifications(shared_face)
    patch = PatchControl.from_stored_fields(spec.patch_controls[0], source_model=model)
    restored = phx.meshing.VolumeMeshingSpec.from_stored_fields(spec, source_model=model)
    assert patch.scope.scope_id == scopes[0].scope_id
    assert restored.patch_controls[0].scope.scope_id == scopes[0].scope_id


@pytest.mark.parametrize("foreign", (False, True), ids=("missing", "foreign"))
def test_sided_patch_restore_refuses_missing_or_foreign_source(
    shared_face: _SharedFaceCase, foreign: bool
) -> None:
    _, spec = _specifications(shared_face)
    source = (
        brep_box(
            (0.0, 0.0, 0.0),
            (3.0, 1.0, 1.0),
            coordinate_contract=SpatialCoordinateContract.si(),
        )
        if foreign
        else None
    )
    with pytest.raises((TypeError, ValueError)):
        PatchControl.from_stored_fields(spec.patch_controls[0], source_model=source)
    with pytest.raises((TypeError, ValueError)):
        phx.meshing.VolumeMeshingSpec.from_stored_fields(spec, source_model=source)


def test_gmsh_patch_preflight_preserves_interface_side(
    shared_face: _SharedFaceCase,
) -> None:
    from phydrax.meshing.providers._gmsh_preflight import _semantic_region_issues

    complete, one_sided = _specifications(shared_face)
    issues: list[str] = []
    _semantic_region_issues(one_sided, shared_face.model, issues)
    assert any("individual incident side" in issue for issue in issues)
    complete_issues: list[str] = []
    _semantic_region_issues(complete, shared_face.model, complete_issues)
    assert not complete_issues


def test_native_patch_preflight_refuses_single_interface_side(
    shared_face: _SharedFaceCase,
) -> None:
    _, one_sided = _specifications(shared_face)
    with pytest.raises(
        (ValueError, phx.meshing.MeshingFailure),
        match="single interface side cannot be flattened",
    ):
        phx.meshing.NativeMeshingProvider().plan(shared_face.model, one_sided).execute()


def test_source_association_selects_the_shared_interface(
    shared_face: _SharedFaceCase, shared_volume: CellMeshingResult
) -> None:
    model, _, _, scopes, _ = shared_face
    domain = MeshingDomain.from_brep(model)
    associations = tuple(
        association
        for association in shared_volume.associations
        if association.association_kind is GeometryAssociationKind.BREP
        and association.target_entity_set_id == shared_volume.mesh.entity_set(2).entity_set_id
        and association.source_id == model.source_id
        and association.source_revision == model.source_revision
    )
    assert len(associations) == 1
    association = associations[0]
    face_key, face_path = domain.source_entity_key_for_scope(
        2, int(scopes[0].side_records[0].entity_id)
    )
    rows = np.asarray(
        [
            row
            for row, (key, path) in enumerate(
                zip(
                    association.source_entity_ids,
                    association.source_occurrence_paths,
                    strict=True,
                )
            )
            if key == face_key and path == face_path
        ],
        dtype=np.int64,
    )
    assert rows.size and np.all(np.asarray(association.resolved)[rows])
    assert not np.any(np.asarray(association.ambiguous)[rows])
    selected = np.asarray(association.target_global_ids)[rows]
    face_rows = np.flatnonzero(
        np.isin(np.asarray(shared_volume.mesh.entity_set(2).entity_ids), selected)
    )
    assert face_rows.size == selected.size
    assert isinstance(shared_volume.mesh.connectivity, SimplicialConnectivity)
    vertices = np.asarray(shared_volume.mesh.connectivity.entities[2])[face_rows]
    np.testing.assert_allclose(
        np.asarray(shared_volume.mesh.coordinates)[vertices, 0], 1.0, atol=1e-10
    )
    assert {
        _expected_side_value(shared_volume, model, scope) for scope in scopes
    } == {-1.0, 1.0}


def _permuted_mesh(mesh: CellMesh) -> CellMesh:
    block = mesh.blocks[0]
    cells = np.asarray(block.vertices)
    result = CellMesh(
        mesh.coordinates,
        (
            CellBlock(
                block.name,
                block.cell_kind,
                cells[::-1],
                global_ids=np.asarray(block.global_ids)[::-1],
            ),
        ),
        vertex_global_ids=mesh.vertex_global_ids,
        numeric_version=mesh.numeric_version,
    )
    np.testing.assert_array_equal(result.coordinates, mesh.coordinates)
    np.testing.assert_array_equal(result.vertex_global_ids, mesh.vertex_global_ids)
    np.testing.assert_array_equal(result.blocks[0].vertices, cells[::-1])
    np.testing.assert_array_equal(
        result.blocks[0].global_ids, np.asarray(block.global_ids)[::-1]
    )
    ids = np.asarray(mesh.vertex_global_ids)
    original = {
        int(cell_id): tuple(sorted(ids[cell].tolist()))
        for cell_id, cell in zip(np.asarray(block.global_ids), cells, strict=True)
    }
    reversed_ = {
        int(cell_id): tuple(sorted(ids[cell].tolist()))
        for cell_id, cell in zip(
            np.asarray(result.blocks[0].global_ids),
            np.asarray(result.blocks[0].vertices),
            strict=True,
        )
    }
    assert original == reversed_
    return result


def _expected_side_value(
    result: CellMeshingResult, model: BRepModel, scope: MeshingScope
) -> float:
    associations = tuple(
        association
        for association in result.associations
        if association.association_kind is GeometryAssociationKind.BREP
        and association.target_entity_set_id == result.mesh.entity_set(3).entity_set_id
        and association.source_id == model.source_id
        and association.source_revision == model.source_revision
    )
    assert len(associations) == 1
    association = associations[0]
    side = scope.side_records[0]
    rows = np.asarray(
        [
            row
            for row, (index, path) in enumerate(
                zip(
                    association.source_indices,
                    association.source_occurrence_paths,
                    strict=True,
                )
            )
            if int(index) == side.solid_index and path == side.region_path
        ],
        dtype=np.int64,
    )
    assert rows.size and np.all(np.asarray(association.resolved)[rows])
    assert not np.any(np.asarray(association.ambiguous)[rows])
    selected = np.asarray(association.target_global_ids)[rows]
    block = result.mesh.blocks[0]
    centers = np.asarray(result.mesh.coordinates)[np.asarray(block.vertices)].mean(axis=1)
    selected_centers = centers[np.isin(np.asarray(block.global_ids), selected)]
    assert selected_centers.shape[0] == selected.size
    assert np.all(selected_centers[:, 0] < 1.0) or np.all(
        selected_centers[:, 0] > 1.0
    )
    return -1.0 if selected_centers[0, 0] < 1.0 else 1.0


@pytest.mark.parametrize("side_index", (0, 1), ids=("first-side", "second-side"))
@pytest.mark.parametrize(
    "reverse", (False, True), ids=("native-storage", "reversed-storage")
)
def test_shared_face_drives_sided_fe_trace(
    shared_face: _SharedFaceCase,
    shared_volume: CellMeshingResult,
    side_index: int,
    reverse: bool,
) -> None:
    model, _, _, scopes, _ = shared_face
    mesh = _permuted_mesh(shared_volume.mesh) if reverse else shared_volume.mesh
    centers = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)].mean(axis=1)
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.discontinuous_element("simplex:3", 1)
    )
    prepared = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    cell_dofs = np.asarray(prepared.dof_maps[0].cell_dofs[0])
    values = np.zeros(prepared.dof_maps[0].global_dof_count)
    values[cell_dofs] = np.where(centers[:, 0, None] < 1.0, -1.0, 1.0)
    scope = scopes[side_index]
    expected = _expected_side_value(shared_volume, model, scope)
    domains = phx.meshing.resolve_sided_integration_domains(
        shared_volume, model, scope, prepared, tolerance=1e-8
    )
    assert domains
    for integration_domain, side in domains:
        trace = prepared.prepare_side_trace(
            "u",
            integration_domain,
            rule=phx.discretization.FacetTraceRule(points=2),
            side=side,
        )
        np.testing.assert_allclose(trace.sites[..., 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(trace.apply(jnp.asarray(values)), expected, atol=1e-10)
        np.testing.assert_allclose(trace.normals[..., 0], -expected, atol=1e-10)
        np.testing.assert_allclose(trace.normals[..., 1:], 0.0, atol=1e-10)


@pytest.mark.parametrize("side_index", (0, 1), ids=("first-side", "second-side"))
@pytest.mark.parametrize(
    "reverse", (False, True), ids=("native-storage", "reversed-storage")
)
def test_shared_face_drives_sided_fv_trace(
    shared_face: _SharedFaceCase,
    shared_volume: CellMeshingResult,
    side_index: int,
    reverse: bool,
) -> None:
    model, _, _, scopes, _ = shared_face
    mesh = _permuted_mesh(shared_volume.mesh) if reverse else shared_volume.mesh
    prepared = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(mesh.coordinates),
        tetrahedra=np.asarray(mesh.blocks[0].vertices),
        vertex_global_ids=np.asarray(mesh.vertex_global_ids),
        cell_global_ids=np.asarray(mesh.blocks[0].global_ids),
    ).prepare(numeric_version=mesh.numeric_version)
    values = np.where(
        np.asarray(prepared.cell_centers)[:, 0] < 1.0, -1.0, 1.0
    )[:, None]
    scope = scopes[side_index]
    expected = _expected_side_value(shared_volume, model, scope)
    domains = phx.meshing.resolve_sided_integration_domains(
        shared_volume, model, scope, prepared, tolerance=1e-8
    )
    assert domains
    if prepared.cell_space is None:
        raise ValueError("Prepared FV mesh must retain its cell field space.")
    for integration_domain, side in domains:
        trace = prepared.prepare_side_trace(
            prepared.cell_space.name,
            integration_domain,
            rule=phx.discretization.FacetTraceRule(points=2),
            side=side,
        )
        np.testing.assert_allclose(trace.sites[..., 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(trace.apply(jnp.asarray(values)), expected, atol=1e-10)
        np.testing.assert_allclose(trace.normals[..., 0], -expected, atol=1e-10)
        np.testing.assert_allclose(trace.normals[..., 1:], 0.0, atol=1e-10)


def test_sided_route_refuses_mixed_intent(
    shared_face: _SharedFaceCase, shared_volume: CellMeshingResult
) -> None:
    model, face, _, scopes, _ = shared_face
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.discontinuous_element("simplex:3", 1)
    )
    prepared = phx.discretization.FiniteElementPlan(shared_volume.mesh, field).prepare()
    whole = MeshingScope.from_geometry(model, model.face_ids[face])
    with pytest.raises(ValueError):
        phx.meshing.resolve_sided_integration_domains(
            shared_volume, model, scopes[0].union(whole), prepared, tolerance=1e-8
        )


@pytest.mark.parametrize(
    "tolerance", (-1.0, float("nan"), float("inf")), ids=("negative", "nan", "infinite")
)
def test_sided_route_refuses_invalid_tolerance(
    shared_face: _SharedFaceCase, shared_volume: CellMeshingResult, tolerance: float
) -> None:
    model, _, _, scopes, _ = shared_face
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.discontinuous_element("simplex:3", 1)
    )
    prepared = phx.discretization.FiniteElementPlan(shared_volume.mesh, field).prepare()
    with pytest.raises(ValueError):
        phx.meshing.resolve_sided_integration_domains(
            shared_volume, model, scopes[0], prepared, tolerance=tolerance
        )


def test_coincident_repeated_shared_copies_refuse_ordinal_pairing(
    shared_face: _SharedFaceCase,
) -> None:
    model, face, _, _, _ = shared_face
    original = model.geometry
    assembly = BRepAssemblyGraph(
        roots=(
            BRepAssemblyUse("first-0", solid_index=0),
            BRepAssemblyUse("second-0", solid_index=0),
            BRepAssemblyUse("second-1", solid_index=1),
            BRepAssemblyUse("first-1", solid_index=1),
        ),
        definitions=(),
        solid_count=2,
    )
    geometry = BRepGeometry(
        vertex_points=original.vertex_points,
        vertex_exact_points=original.vertex_exact_points,
        vertex_roots=original.vertex_roots,
        curves=original.curves,
        edge_curves=original.edge_curves,
        edge_ranges=original.edge_ranges,
        edge_vertices=original.edge_vertices,
        edge_endpoint_roots=original.edge_endpoint_roots,
        pcurves=original.pcurves,
        coedge_edges=original.coedge_edges,
        coedge_senses=original.coedge_senses,
        coedge_endpoint_roots=original.coedge_endpoint_roots,
        face_loops=original.face_loops,
        shell_faces=original.shell_faces,
        shell_orientations=original.shell_orientations,
        solid_shells=original.solid_shells,
        assembly_graph=assembly,
    )
    repeated = BRepModel(
        geometry=geometry,
        patches=model.patches,
        parameter_bounds=model.parameter_bounds,
        orientation=model.orientation,
        trim_domains=model.trim_domains,
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
    )
    revision = cad_revision_from_brep_model(repeated)
    members = {
        record.member
        for record in geometry.qualified_entity_incidence(
            repeated.source_id, repeated.source_revision
        )
        if record.member.kind == "face" and record.member.index == face
    }
    assert len(members) == 4
    for member in members:
        with pytest.raises(ValueError):
            MeshingScope.from_geometry(
                repeated,
                CADSelectionSet.from_revision(
                    revision, (_entity_occurrence_id(member),)
                ),
            )


def test_gmsh_periodic_preflight_preserves_cad_sides_without_gmsh(
    shared_face: _SharedFaceCase,
) -> None:
    from types import SimpleNamespace

    from phydrax.meshing.providers._gmsh_preflight import _validate_gmsh_periodicity

    model, face, _, shared_sides, _ = shared_face
    exterior = next(
        index for index, owners in enumerate(model.topology.face_solids)
        if len(owners) == 1
    )
    revision = cad_revision_from_brep_model(model)
    incidence = model.geometry.qualified_entity_incidence(
        model.source_id, model.source_revision
    )
    exterior_sides = tuple(
        MeshingScope.from_geometry(
            model,
            CADSelectionSet(
                revision.revision_id,
                (revision.select(_entity_occurrence_id(use.member)),),
            ),
        )
        for use in incidence
        if use.container.kind == "solid"
        and use.member.kind == "face"
        and use.member.index == exterior
    )
    assert len(shared_sides) == 2 and len(exterior_sides) == 1
    whole = MeshingScope.from_geometry(model, model.face_ids[face])
    both = shared_sides[0].union(shared_sides[1])
    record = shared_sides[0].side_records[0]
    forged = MeshingScope(
        shared_sides[0].source_id,
        shared_sides[0].source_revision,
        shared_sides[0].entity_kind,
        2,
        shared_sides[0].entity_set_id,
        shared_sides[0].entity_ids,
        side_records=(dataclasses.replace(record, orientation=-record.orientation),),
        unqualified_entity_ids=(),
    )

    def periodic_issues(left: MeshingScope, right: MeshingScope) -> list[str]:
        constraint = phx.meshing.PeriodicConstraint(left, right, np.eye(4))
        issues: list[str] = []
        _validate_gmsh_periodicity(
            SimpleNamespace(periodic_constraints=(constraint,)), model, issues
        )
        return issues

    assert any(
        "individual incident side" in issue
        for issue in periodic_issues(shared_sides[0], whole)
    )
    assert any(
        "individual incident side" in issue
        for issue in periodic_issues(whole, shared_sides[0])
    )
    assert not periodic_issues(both, whole)
    assert not periodic_issues(exterior_sides[0], whole)
    assert any(
        "source BRep curves or surfaces" in issue
        for issue in periodic_issues(forged, whole)
    )
