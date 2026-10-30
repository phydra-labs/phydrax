from fractions import Fraction

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellMesh
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry._source_identity import source_entity_id
from phydrax.geometry.brep import brep_box, prepare_brep_projection
from phydrax.geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._model import BRepEntityId, BRepGeometry, BRepModel
from phydrax.meshing._association import (
    associate_mesh_entities,
    associate_mesh_vertices,
    AssociationPropagationPolicy,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._scope import MeshingScope


def test_surface_association_keeps_large_source_indices_and_occurrence_identity() -> None:
    source_index = 2**40
    paths = (("assembly", "left"), ("assembly", "right"))
    keys = tuple(
        source_entity_id(
            "surface", "revision", "surface", source_index, occurrence_path=path
        )
        for path in paths
    )
    association = GeometryAssociation(
        GeometryAssociationKind.SURFACE,
        "surface",
        "revision",
        "mesh-faces",
        np.asarray((7, 19), dtype=np.int64),
        keys,
        np.zeros(2, dtype=np.float64),
        source_dimensions=np.asarray((2, 2), dtype=np.int8),
        source_indices=np.asarray((source_index, source_index), dtype=np.int64),
        source_roles=(GeometrySourceEntityRole.FACET, GeometrySourceEntityRole.FACET),
        source_occurrence_paths=paths,
    )

    assert keys[0] != keys[1]
    rows = association.target_rows(np.asarray((19, 7), dtype=np.int64))
    assert tuple(association.source_entity_ids[int(row)] for row in rows) == keys[::-1]
    assert (
        tuple(association.source_occurrence_paths[int(row)] for row in rows)
        == paths[::-1]
    )
    np.testing.assert_array_equal(
        association.source_indices, (source_index, source_index)
    )


@pytest.mark.parametrize(
    "dimensions, indices",
    (
        (np.asarray((-1,), dtype=np.int8), np.asarray((0,), dtype=np.int64)),
        (np.asarray((2,), dtype=np.int8), np.asarray((2**63,), dtype=np.uint64)),
    ),
    ids=("unpaired-stratum", "unrepresentable-index"),
)
def test_surface_association_refuses_invalid_source_strata(
    dimensions: np.ndarray, indices: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        GeometryAssociation(
            GeometryAssociationKind.SURFACE,
            "surface",
            "revision",
            "mesh-faces",
            np.asarray((7,), dtype=np.int64),
            ("source-face",),
            np.zeros(1, dtype=np.float64),
            source_dimensions=dimensions,
            source_indices=indices,
        )


def _two_box_occurrences() -> BRepModel:
    source = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    placement = BRepAffinePlacement.from_matrix(
        (
            (1, 0, 0, Fraction(3, 2)),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    assembly = BRepAssemblyGraph(
        roots=(BRepAssemblyUse("pair", definition_id="copies"),),
        definitions=(
            BRepAssemblyDefinition(
                "copies",
                (
                    BRepAssemblyUse("left", solid_index=0),
                    BRepAssemblyUse("right", placement=placement, solid_index=0),
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


def test_native_associations_do_not_merge_reused_definition_occurrences() -> None:
    model = _two_box_occurrences()
    left = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64
    )
    points = np.concatenate((left, left + np.asarray((1.5, 0.0, 0.0), dtype=np.float64)))
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "faces", "triangle", np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
            ),
        ),
    )
    projection = prepare_brep_projection(model)
    policy = AssociationPropagationPolicy()
    vertices = associate_mesh_vertices(mesh, projection, policy=policy)
    faces = associate_mesh_entities(mesh, vertices, projection, 2, policy=policy)

    assert vertices.complete and faces.complete
    assert vertices.source_occurrence_paths == (
        (("pair", "left"),) * 3 + (("pair", "right"),) * 3
    )
    assert faces.source_occurrence_paths == (("pair", "left"), ("pair", "right"))
    np.testing.assert_array_equal(faces.source_indices, (0, 0))
    assert faces.source_entity_ids[0] != faces.source_entity_ids[1]


@pytest.mark.parametrize(
    ("foreign_source", "foreign_path"),
    (("different-source", ("left",)), ("source", ("right",))),
    ids=("source-owner", "occurrence"),
)
def test_unclassified_brep_rows_remain_bound_to_source_occurrence(
    foreign_source: str, foreign_path: tuple[str, ...]
) -> None:
    key = source_entity_id(
        "source",
        "revision",
        "unclassified",
        "unclassified",
        occurrence_path=("left",),
    )
    association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        "source",
        "revision",
        "vertices",
        np.asarray((7,), dtype=np.int64),
        (key,),
        np.asarray((np.inf,), dtype=np.float64),
        resolved=np.asarray((False,), dtype=np.bool_),
        source_dimensions=np.asarray((-1,), dtype=np.int8),
        source_indices=np.asarray((-1,), dtype=np.int64),
        source_occurrence_paths=(("left",),),
    )
    assert not association.complete
    assert association.source_entity_ids == (key,)
    foreign = source_entity_id(
        foreign_source,
        "revision",
        "unclassified",
        "unclassified",
        occurrence_path=foreign_path,
    )
    with pytest.raises(ValueError):
        GeometryAssociation(
            GeometryAssociationKind.BREP,
            "source",
            "revision",
            "vertices",
            np.asarray((7,), dtype=np.int64),
            (foreign,),
            np.asarray((np.inf,), dtype=np.float64),
            resolved=np.asarray((False,), dtype=np.bool_),
            source_dimensions=np.asarray((-1,), dtype=np.int8),
            source_indices=np.asarray((-1,), dtype=np.int64),
            source_occurrence_paths=(("left",),),
        )


def test_cad_scopes_preserve_occurrences_and_refuse_ambiguous_definitions() -> None:
    model = _two_box_occurrences()
    domain = MeshingDomain.from_brep(model)
    left_entity = BRepEntityId(
        model.source_id, model.source_revision, "face", 0, ("pair", "left")
    )
    right_entity = BRepEntityId(
        model.source_id, model.source_revision, "face", 0, ("pair", "right")
    )
    left = MeshingScope.from_geometry(model, left_entity)
    right = MeshingScope.from_geometry(model, right_entity)
    assert not np.intersect1d(
        np.asarray(left.entity_ids), np.asarray(right.entity_ids)
    ).size
    for scope, entity in ((left, left_entity), (right, right_entity)):
        keys = tuple(
            domain.source_entity_key_for_scope(2, int(identifier))
            for identifier in np.asarray(scope.entity_ids)
        )
        assert keys == (
            (
                source_entity_id(
                    entity.source_id,
                    entity.source_revision,
                    entity.kind,
                    entity.index,
                    occurrence_path=entity.occurrence_path,
                ),
                entity.occurrence_path,
            ),
        )
    whole = MeshingScope.whole(model, 2)
    np.testing.assert_array_equal(whole.entity_ids, domain.scope_entity_ids(2))
    assert whole.entity_set_id == domain.scope_entity_set_id(2)
    assert set(domain.source_occurrences[2]) == {("pair", "left"), ("pair", "right")}
    with pytest.raises(ValueError):
        MeshingScope.from_geometry(model, model.face_ids[0])


def test_cad_scope_rejects_a_foreign_owner_with_the_same_revision() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    selected = MeshingScope.from_geometry(model, model.face_ids[0])
    assert selected.source_id == model.source_id
    foreign = BRepEntityId("foreign-owner", model.source_revision, "face", 0)
    with pytest.raises(ValueError):
        MeshingScope.from_geometry(model, foreign)
