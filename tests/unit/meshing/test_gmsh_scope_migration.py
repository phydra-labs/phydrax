
from pathlib import Path

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep import brep_box
from phydrax.geometry.brep._assembly import (
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._model import BRepGeometry, BRepModel
from phydrax.interchange import CadImportPolicy, read_brep_text, write_brep_text
from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing.providers._gmsh_import import (
    _CadEntityMap,
    _CadImport,
    _CadImportCache,
    _resolve_entities,
    _scope_indices,
    _scope_samples,
    _scope_source_records,
    _source_entity_key_banks,
    _source_entity_reference_banks,
    _source_mapping_issue,
    _source_occurrence_issue,
    _source_tag_banks,
)


def _box() -> BRepModel:
    return brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )


def _reused_definition() -> BRepModel:
    source = _box()
    assembly = BRepAssemblyGraph(
        roots=(BRepAssemblyUse("pair", definition_id="copies"),),
        definitions=(
            BRepAssemblyDefinition(
                "copies",
                (
                    BRepAssemblyUse("left", solid_index=0),
                    BRepAssemblyUse("right", solid_index=0),
                ),
            ),
        ),
        solid_count=1,
    )
    original = source.geometry
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
    return BRepModel(
        geometry=geometry,
        patches=source.patches,
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        trim_domains=source.trim_domains,
        physical_tags=source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        report=source.report,
    )


def _entity_map(source: BRepModel) -> tuple[_CadEntityMap, dict[int, tuple[int, ...]]]:
    domain = MeshingDomain.from_brep(source)
    keys = _source_entity_key_banks(source, domain)
    tag_banks = {
        dimension: tuple(
            20_000 * dimension + 100 + 17 * (count - row)
            for row in range(count)
        )
        for dimension, count in enumerate(
            (
                source.report.num_vertices,
                source.report.num_edges,
                source.report.num_faces,
                source.topology.num_solids,
            )
        )
    }
    entity_map = _CadEntityMap(
        source.report.source_id,
        source.source_revision,
        keys,
        vertex_to_point=tag_banks[0],
        edge_to_curve=tag_banks[1],
        face_to_surface=tag_banks[2],
        solid_to_volume=tag_banks[3],
    )
    return entity_map, tag_banks


def test_source_directed_provider_tags_resolve_through_canonical_source_rows() -> None:
    source = _box()
    domain = MeshingDomain.from_brep(source)
    assert _source_mapping_issue(source) is None
    keys = _source_entity_key_banks(source, domain)
    counts = (
        source.report.num_vertices,
        source.report.num_edges,
        source.report.num_faces,
        source.topology.num_solids,
    )
    tag_banks = tuple(
        tuple(
            1_000_003 + 1_000_000 * dimension + 97 * (count - row)
            for row in range(count)
        )
        for dimension, count in enumerate(counts)
    )
    source_tag_entities = tuple(
        (dimension, row, tag_banks[dimension][row])
        for dimension in (3, 1, 0, 2)
        for row in reversed(range(len(tag_banks[dimension])))
    )
    assert _source_tag_banks(source_tag_entities, source) == tag_banks

    entity_map = _CadEntityMap(
        source.report.source_id,
        source.source_revision,
        keys,
        vertex_to_point=tag_banks[0],
        edge_to_curve=tag_banks[1],
        face_to_surface=tag_banks[2],
        solid_to_volume=tag_banks[3],
    )
    scope = MeshingScope.whole(source, 2)
    records = _scope_source_records(source, scope, 2)
    assert records is not None
    expected = tuple(tag_banks[2][row] for _, _, row in records)
    assert (
        _resolve_entities(source, scope, cad_entities=entity_map) == expected
    )

    with pytest.raises(MeshingFailure) as missing_map:
        _resolve_entities(source, scope, cad_entities=None)
    assert missing_map.value.category is MeshingFailureCategory.SCOPE_RESOLUTION_FAILED


def test_exporter_provenance_roots_exact_native_source_entities(
    tmp_path: Path,
) -> None:
    source = _box()
    snapshot = tmp_path / "box.brep"
    exported = write_brep_text(source, snapshot)
    references = _source_entity_reference_banks(source, exported.provenance)
    entry = _CadImport("digest", snapshot, object(), references, None, {})
    cache = _CadImportCache(1)
    try:
        root_path, root_offset, root_width = cache._entity_root_template(entry)
        policy = CadImportPolicy(
            source.coordinate_contract,
            relative_geometric_tolerance=(
                source.report.curve_surface_tolerance
                / source.report.curve_surface_scale
            ),
        )
        for dimension, bank in enumerate(references):
            for reference in bank:
                cache._select_entity_root(root_path, root_offset, root_width, reference)
                decoded = read_brep_text(
                    root_path,
                    policy,
                    trusted_root=tmp_path,
                    source_length_unit=source.coordinate_contract.length_unit,
                )
                model = decoded.model
                counts = (
                    model.report.num_vertices,
                    model.report.num_edges,
                    model.report.num_faces,
                    model.topology.num_solids,
                )
                assert counts[dimension] == 1
                assert not any(counts[dimension + 1 :])
        assert _source_mapping_issue(model) is None
    finally:
        cache.close()


@pytest.mark.parametrize("dimension", (0, 1, 2, 3))
def test_canonical_source_scopes_resolve_through_import_tag_tables(
    dimension: int,
) -> None:
    source = _box()
    scope = MeshingScope.whole(source, dimension)
    entity_map, tag_banks = _entity_map(source)

    rows = _scope_indices(source, scope, dimension)
    records = _scope_source_records(source, scope, dimension)
    assert rows is not None and records is not None
    assert {path for _, path, _ in records} == {("solid-0",)}
    assert not np.array_equal(np.asarray(scope.entity_ids), rows)
    expected = tuple(tag_banks[dimension][row] for _, _, row in records)
    actual = _resolve_entities(source, scope, cad_entities=entity_map)

    np.testing.assert_array_equal(rows, np.arange(rows.size))
    assert actual == expected


@pytest.mark.parametrize("identity_field", ("source_id", "source_revision"))
def test_gmsh_preflight_rejects_wrong_scope_owner_or_revision(
    identity_field: str,
) -> None:
    source = _box()
    domain = MeshingDomain.from_brep(source)
    scope = MeshingScope.whole(source, 2)
    entity_ids = np.asarray(scope.entity_ids, dtype=np.int64)
    owner = "another-owner" if identity_field == "source_id" else scope.source_id
    revision = (
        "another-revision"
        if identity_field == "source_revision"
        else scope.source_revision
    )
    invalid = MeshingScope(
        owner,
        revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        entity_ids,
    )

    assert _scope_indices(source, invalid, 2) is None
    entity_map, _ = _entity_map(source)
    with pytest.raises(MeshingFailure) as failure:
        _resolve_entities(source, invalid, cad_entities=entity_map)
    assert failure.value.category is MeshingFailureCategory.SCOPE_RESOLUTION_FAILED


def test_gmsh_refuses_occurrence_scopes_without_merging_reused_definitions() -> None:
    source = _reused_definition()
    domain = MeshingDomain.from_brep(source)
    scope = MeshingScope.whole(source, 2)
    records = _scope_source_records(source, scope, 2)

    assert records is not None
    assert len(records) == 2 * source.report.num_faces
    assert len({row for _, _, row in records}) == source.report.num_faces
    assert {path for _, path, _ in records} == {
        ("pair", "left"),
        ("pair", "right"),
    }
    assert _source_mapping_issue(source) is not None
    assert _source_occurrence_issue(source, domain) is not None
    assert _scope_indices(source, scope, 2) is None
    with pytest.raises(MeshingFailure) as failure:
        _scope_samples(source, scope)
    assert failure.value.category is MeshingFailureCategory.SCOPE_RESOLUTION_FAILED
    with pytest.raises(MeshingFailure) as map_failure:
        _source_entity_key_banks(source, domain)
    assert map_failure.value.category is MeshingFailureCategory.SCOPE_RESOLUTION_FAILED
