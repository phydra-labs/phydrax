import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellMesh
from phydrax.geometry._planar_embedding import PlanarEmbedding
from phydrax.meshing._adaptation import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    prepare_mesh_adaptation,
)
from phydrax.meshing._canonical import certify_cell_mesh
from phydrax.meshing._organization import MeshLabel, MeshPatch, RegionBoundaryEvidence
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope


def test_native_refinement_preserves_world_frame_and_oriented_source_boundary() -> None:
    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "triangle",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int64),
                global_ids=np.asarray((41,), dtype=np.int64),
            ),
        ),
    )
    boundary_scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        1,
        mesh.entity_set(1).entity_set_id,
        mesh.entity_set(1).entity_ids,
    )
    source_scope = MeshingScope(
        "authored-sheet",
        "sheet-revision",
        MeshingEntityKind.GEOMETRY,
        2,
        "authored-regions",
        np.asarray((7,), dtype=np.int64),
    )
    patch = MeshPatch(
        "wall",
        boundary_scope,
        source_adjacent_region_ids=("sheet-region", None),
        source_adjacent_region_occurrence_paths=(("sheet-occurrence",), None),
    )
    label = MeshLabel("region-boundary", boundary_scope)
    boundary = RegionBoundaryEvidence(
        mesh,
        source_scope,
        "sheet-region",
        "region-control",
        None,
        None,
        label,
        ((patch.patch_id, 1),),
        source_occurrence_path=("sheet-occurrence",),
    )
    embedding = PlanarEmbedding(
        (3.0, 4.0, 5.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    )
    source = certify_cell_mesh(
        mesh,
        SpatialCoordinateContract.si(),
        patches=(patch,),
        labels=(label,),
        region_boundary_evidence=(boundary,),
        planar_embedding=embedding,
    )
    prepared = prepare_mesh_adaptation(
        source,
        MarkedMeshAdaptation(np.asarray((41,), dtype=np.int64)),
        policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION),
    )

    target = execute_mesh_adaptation(prepared).target

    assert target.planar_embedding == embedding
    assert target.planar_embedding is not None
    np.testing.assert_allclose(
        target.planar_embedding.to_world(target.mesh.coordinates)[:, 2], 5.0
    )
    assert target.patches[0].source_adjacent_region_ids == ("sheet-region", None)
    assert target.patches[0].source_adjacent_region_occurrence_paths == (
        ("sheet-occurrence",),
        None,
    )
    (evidence,) = target.region_boundary_evidence
    assert evidence.source_scope.scope_id == source_scope.scope_id
    assert evidence.source_region_id == "sheet-region"
    assert evidence.source_occurrence_path == ("sheet-occurrence",)
    assert evidence.patch_sides == ((target.patches[0].patch_id, 1),)
    evidence.require_current(target.mesh, target.labels, target.patches)
    edges = target.mesh.entity_set(1)
    np.testing.assert_array_equal(
        target.labels[0].scope.entity_ids,
        np.sort(np.asarray(edges.entity_ids[edges.subset("boundary").mask])),
    )
    np.testing.assert_array_equal(source.mesh.coordinates, coordinates)
