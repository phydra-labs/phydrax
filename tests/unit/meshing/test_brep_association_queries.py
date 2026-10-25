#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""B-Rep association source queries reach the caller's request ticket.

Every point submitted to a B-Rep projection, classification or solid location
is one geometry query. A request budget keeps its own caps; standalone calls
settle against the propagation policy's declared allowance.
"""

from typing import Any

import numpy as np
import pytest
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.gp import gp_Pnt

import phydrax as phx
from phydrax._resource_ticket import HostBatchEstimateError, HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from tests._support.cell_meshes import kuhn_cube, triangle_grid


_CONTRACT = phx.SpatialCoordinateContract.si()
_EMBEDDING = phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
_POLICY = phx.meshing.AssociationPropagationPolicy(classification_tolerance=1e-8)
_WORK = 1 << 30
_BYTES = 1 << 30


def _projection() -> Any:
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        polygon.Add(gp_Pnt(x, y, 0.0))
    polygon.Close()
    face = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    model = phx.interchange.model_from_occt_shape(
        face,
        coordinate_contract=_CONTRACT,
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    return phx.interchange.prepare_occt_projection(model, face, embedding=_EMBEDDING)


def _mesh() -> phx.discretization.CellMesh:
    mesh = triangle_grid(2, 2)
    return phx.meshing.canonicalize_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            np.asarray(mesh.coordinates), mesh.blocks[0].vertices
        )
    )


def _entity_rows(mesh: phx.discretization.CellMesh, /) -> int:
    return sum(
        mesh.entity_set(dimension).count
        for dimension in range(mesh.topological_dimension + 1)
    )


@pytest.fixture(scope="module")
def bisected() -> tuple[Any, Any, Any]:
    projection = _projection()
    mesh = _mesh()
    vertex = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    associations = (vertex,) + tuple(
        phx.meshing.associate_mesh_entities(
            mesh, vertex, projection, dimension, policy=_POLICY
        )
        for dimension in (1, 2)
    )
    source = phx.meshing.certify_cell_mesh(mesh, _CONTRACT, associations=associations)
    adapted = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(
                np.concatenate(
                    [
                        np.asarray(block.global_ids, dtype=np.int64)
                        for block in source.mesh.blocks
                    ]
                )
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                association_transfer=phx.meshing.BRepAssociationTransfer(
                    projection, policy=_POLICY
                ),
            ),
        )
    )
    return projection, source, adapted


def test_projection_allowance_is_bound_into_policy_identity() -> None:
    narrow = phx.meshing.AssociationPropagationPolicy(
        classification_tolerance=1e-8, maximum_projection_queries=1
    )
    assert narrow.maximum_projection_queries == 1
    assert narrow.policy_id != _POLICY.policy_id
    assert (
        _POLICY.maximum_projection_queries
        == phx.meshing.MeshingLimits().maximum_geometry_queries
    )
    with pytest.raises(ValueError, match="maximum_projection_queries"):
        phx.meshing.AssociationPropagationPolicy(maximum_projection_queries=0)


def test_standalone_vertex_association_refuses_beyond_declared_allowance() -> None:
    mesh = _mesh()
    narrow = phx.meshing.AssociationPropagationPolicy(
        classification_tolerance=1e-8,
        maximum_projection_queries=mesh.coordinates.shape[0] - 1,
    )
    with pytest.raises(HostBatchEstimateError):
        phx.meshing.associate_mesh_vertices(mesh, _projection(), policy=narrow)


def test_request_budget_is_charged_for_every_projected_entity(
    bisected: tuple[Any, Any, Any],
) -> None:
    projection, source, adapted = bisected
    target = adapted.target.mesh
    # A tiny standalone allowance must not replace the request's own caps.
    transfer = phx.meshing.BRepAssociationTransfer(
        projection,
        policy=phx.meshing.AssociationPropagationPolicy(
            classification_tolerance=1e-8, maximum_projection_queries=1
        ),
    )
    with HostBatchTicket(_WORK, queries_upper=1 << 20, bytes_upper=_BYTES) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=_WORK, maximum_memory_bytes=_BYTES
        )
        propagated = transfer.propagate(source, adapted.lineage, target, budget=budget)
        charged = ticket.geometry_queries
    # Each target vertex, edge and face is projected once onto its class.
    assert charged >= _entity_rows(target)
    standalone = phx.meshing.BRepAssociationTransfer(projection, policy=_POLICY)
    expected = standalone.propagate(source, adapted.lineage, target)
    assert tuple(value.association_id for value in propagated) == tuple(
        value.association_id for value in expected
    )


def test_request_budget_refuses_before_exceeding_its_query_cap(
    bisected: tuple[Any, Any, Any],
) -> None:
    projection, source, adapted = bisected
    target = adapted.target.mesh
    transfer = phx.meshing.BRepAssociationTransfer(projection, policy=_POLICY)
    cap = target.coordinates.shape[0] - 1
    with HostBatchTicket(_WORK, queries_upper=cap, bytes_upper=_BYTES) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=_WORK, maximum_memory_bytes=_BYTES
        )
        with pytest.raises(HostBatchEstimateError):
            transfer.propagate(source, adapted.lineage, target, budget=budget)
        assert ticket.geometry_queries <= cap


@pytest.fixture(scope="module")
def mixed_box_refinement() -> tuple[Any, Any, Any]:
    model = phx.geometry.brep_box(
        (0, 0, 0), (1, 1, 1), coordinate_contract=_CONTRACT
    )
    transfer = phx.meshing.BRepAssociationTransfer(
        phx.geometry.prepare_brep_projection(model), policy=_POLICY
    )
    mesh = phx.meshing.canonicalize_cell_mesh(kuhn_cube())
    vertex = phx.meshing.associate_mesh_vertices(
        mesh, transfer.projection, policy=_POLICY
    )
    source = phx.meshing.certify_cell_mesh(mesh, _CONTRACT, associations=(vertex,))
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_MIXED, association_transfer=transfer
    )
    # Freeze the source skeleton to select the face-preserving, interior-center
    # tetra template used by wall/core hybrid refinement.
    skeleton = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        1,
        mesh.entity_set(1).entity_set_id,
        np.asarray(mesh.entity_set(1).entity_ids),
    )
    refined = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(np.asarray(mesh.blocks[0].global_ids)),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
                association_transfer=transfer,
                protected_scopes=(skeleton,),
            ),
        )
    )
    return source, refined, policy


def test_created_tetra_centers_inherit_solid_and_allow_full_family_inverse(
    mixed_box_refinement: tuple[Any, Any, Any],
) -> None:
    source, refined, policy = mixed_box_refinement
    transfer = policy.association_transfer
    fine = refined.target
    source_ids = np.asarray(source.mesh.vertex_global_ids)
    fine_ids = np.asarray(fine.mesh.vertex_global_ids)
    created = ~np.isin(fine_ids, source_ids)
    assert np.count_nonzero(created) == source.mesh.blocks[0].cell_count
    vertex = fine.associations[0]
    assert vertex.complete
    classes = transfer.classes(fine)[0]
    np.testing.assert_array_equal(classes.dimensions[created], 3)
    np.testing.assert_array_equal(classes.indices[created], 0)
    np.testing.assert_array_equal(np.asarray(vertex.parent_dimensions)[created], 3)
    # Original CAD corners still own their zero-dimensional strata.
    np.testing.assert_array_equal(classes.dimensions[~created], 0)
    fine_cell_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in fine.mesh.blocks]
    )
    prepared = phx.meshing.prepare_mesh_adaptation(
        fine,
        phx.meshing.MarkedMeshAdaptation(
            np.zeros(0, dtype=np.int64),
            fine_cell_ids,
            hierarchy=refined.hierarchy,
        ),
        policy=policy,
    )
    np.testing.assert_array_equal(
        prepared.constraints.protected_vertex_ids, source_ids
    )
    restored = phx.meshing.execute_mesh_adaptation(prepared)
    assert restored.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert isinstance(restored.evidence, phx.meshing.MixedAdaptationEvidence)
    assert restored.evidence.rejected_coarsening_ids.size == 0
    np.testing.assert_array_equal(restored.target.mesh.vertex_global_ids, source_ids)
    np.testing.assert_array_equal(
        restored.target.mesh.blocks[0].global_ids, source.mesh.blocks[0].global_ids
    )


def test_unrouted_interior_vertices_locate_solids_without_reclassifying_corners(
    mixed_box_refinement: tuple[Any, Any, Any],
) -> None:
    source, refined, policy = mixed_box_refinement
    # A provider may publish created vertices but no source-cell refinement
    # routes. These vertices still need canonical interior solid location.
    records = []
    for dimension in range(source.mesh.topological_dimension + 1):
        record = refined.lineage.entity_lineage(dimension)
        preserved = (
            np.asarray(record.relation_kinds)
            == phx.meshing.EntityLineageKind.PRESERVED
        )
        records.append(
            phx.meshing.EntityLineage(
                dimension,
                record.source_entity_set_id,
                record.target_entity_set_id,
                np.asarray(record.source_global_ids)[preserved],
                np.asarray(record.target_global_ids)[preserved],
                np.asarray(record.relation_kinds)[preserved],
                created_target_ids=record.created_target_ids,
                deleted_source_ids=record.deleted_source_ids,
            )
        )
    lineage = phx.meshing.MeshLineage(
        source.mesh.topology_id, refined.target.mesh.topology_id, tuple(records)
    )
    vertex = policy.association_transfer.propagate(
        source, lineage, refined.target.mesh
    )[0]
    created = ~np.isin(
        np.asarray(vertex.target_global_ids), np.asarray(source.mesh.vertex_global_ids)
    )
    assert vertex.complete
    np.testing.assert_array_equal(np.asarray(vertex.source_dimensions)[created], 3)
    np.testing.assert_array_equal(np.asarray(vertex.source_dimensions)[~created], 0)
