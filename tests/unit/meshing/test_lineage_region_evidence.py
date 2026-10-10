#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Region evidence and interface attachments across adaptation and motion.

Oracles are coordinate facts of the unit plate independent of the lineage: wall
edges lie on ``x = 0``, region cells cover the plate, and the attached side's
outward normal on the wall points to ``+x``.
"""

from typing import Any

import numpy as np
import pytest
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.gp import gp_Pnt

import phydrax as phx
from phydrax.meshing import (
    MeshInterfaceAttachment,
    MeshPart,
    transition_interface_attachment,
)


_CONTRACT = phx.SpatialCoordinateContract.si()
_POLICY = phx.meshing.AssociationPropagationPolicy(classification_tolerance=1e-8)
_EMBEDDING = phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
_TOLERANCE = 1e-9


def _projection() -> Any:
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in ((-1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (-1.0, 1.0)):
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
    xs = np.linspace(-1.0, 0.0, 3)
    points = np.asarray([[x, y] for y in np.linspace(0.0, 1.0, 3) for x in xs])
    cells = []
    for row in range(2):
        for column in range(2):
            a = row * 3 + column
            cells += [[a, a + 1, a + 4], [a, a + 4, a + 3]]
    return phx.meshing.canonicalize_cell_mesh(
        phx.discretization.CellMesh.from_triangles(points, np.asarray(cells))
    )


def _scope(mesh: Any, dimension: int, ids: Any) -> phx.meshing.MeshingScope:
    return phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        dimension,
        mesh.entity_set(dimension).entity_set_id,
        np.asarray(ids),
    )


def _wall_edge_ids(mesh: Any) -> np.ndarray:
    edges = np.asarray(mesh.connectivity.edges)
    on_wall = np.all(np.abs(np.asarray(mesh.coordinates)[edges][..., 0]) < 1e-12, axis=1)
    return np.sort(np.asarray(mesh.entity_set(1).entity_ids)[on_wall])


def _plate() -> tuple[Any, Any, Any]:
    projection = _projection()
    mesh = _mesh()
    vertex = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    edges = phx.meshing.associate_mesh_entities(
        mesh, vertex, projection, 1, policy=_POLICY
    )
    cells = phx.meshing.associate_mesh_entities(
        mesh, vertex, projection, 2, policy=_POLICY
    )
    wall = phx.meshing.MeshPatch("wall", _scope(mesh, 1, _wall_edge_ids(mesh)))
    region = phx.meshing.MeshZone(
        "plate",
        phx.meshing.MeshZoneRole.REGION,
        _scope(mesh, 2, mesh.entity_set(2).entity_ids),
        material_id="steel",
        region_role=phx.meshing.RegionRole.SOLID,
    )
    result = phx.meshing.certify_cell_mesh(
        mesh,
        _CONTRACT,
        associations=(vertex, edges, cells),
        patches=(wall,),
        zones=(region,),
    )
    rows = [
        row
        for row, value in enumerate(edges.source_entity_ids)
        if int(np.asarray(edges.target_global_ids)[row]) in set(_wall_edge_ids(mesh))
    ]
    wall_entity = edges.source_entity_ids[rows[0]]
    part = MeshPart("plate", result)
    attachment = MeshInterfaceAttachment(
        part, wall, edges, (wall_entity,), tolerance=_TOLERANCE, region=region
    )
    return projection, part, attachment


def _refined(projection: Any, part: Any, hierarchy: Any = None) -> Any:
    source = part.carrier
    return phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(
                np.concatenate(
                    [
                        np.asarray(block.global_ids, dtype=np.int64)
                        for block in source.mesh.blocks
                    ]
                ),
                hierarchy=hierarchy,
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                association_transfer=phx.meshing.BRepAssociationTransfer(
                    projection, policy=_POLICY
                ),
            ),
        )
    )


def test_attachment_follows_bisection_onto_the_successor_part() -> None:
    projection, part, attachment = _plate()
    first = _refined(projection, part)
    middle = MeshPart("plate", first.target)
    carried = transition_interface_attachment(attachment, part, middle, first.lineage)
    second = _refined(projection, middle, first.hierarchy)
    target = MeshPart("plate", second.target)

    moved = transition_interface_attachment(carried, middle, target, second.lineage)

    mesh = second.target.mesh
    assert moved.part_revision == target.part_id
    np.testing.assert_array_equal(
        np.sort(np.asarray(moved.scope.entity_ids)), _wall_edge_ids(mesh)
    )
    assert _wall_edge_ids(mesh).size > _wall_edge_ids(part.carrier.mesh).size
    np.testing.assert_array_equal(
        np.sort(np.asarray(moved.region.entity_ids)),  # ty: ignore[unresolved-attribute]
        np.sort(np.asarray(mesh.entity_set(2).entity_ids)),
    )
    assert moved.orientation == attachment.orientation
    _, corners, normals = moved.oriented_simplices(target)
    assert np.all(np.abs(corners[..., 0]) < 1e-12)
    # The attached plate lies at x < 0, so its outward wall normal points to +x.
    assert np.all(normals[:, 0] * moved.orientation > 0.0)
    zone = second.target.zones[0]
    assert (zone.material_id, zone.region_role) == ("steel", phx.meshing.RegionRole.SOLID)
    with pytest.raises(ValueError, match="stale"):
        attachment.require_current(target)


def test_attachment_transition_refuses_a_foreign_lineage() -> None:
    projection, part, attachment = _plate()
    adaptation = _refined(projection, part)
    again = _refined(projection, MeshPart("plate", adaptation.target))
    with pytest.raises(ValueError, match="lineage must join"):
        transition_interface_attachment(
            attachment, part, MeshPart("plate", again.target), again.lineage
        )


def test_motion_carries_region_evidence_and_refuses_unrevalidated_associations() -> None:
    projection, part, _ = _plate()
    source = part.carrier
    monitor = phx.meshing.MeshMotionMonitor(source.mesh)
    moved = np.asarray(source.mesh.coordinates).copy()
    interior = np.all(
        (moved > np.asarray((-1.0, 0.0)) + 1e-9)
        & (moved < np.asarray((0.0, 1.0)) - 1e-9),
        axis=1,
    )
    moved[interior] += np.asarray((0.01, 0.01))

    with pytest.raises(ValueError, match="association_transfer"):
        phx.meshing.advance_mesh_motion(monitor, source, moved, boundary_residual=0.0)

    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D,
        association_transfer=phx.meshing.BRepAssociationTransfer(
            projection, policy=_POLICY
        ),
    )
    advance = phx.meshing.advance_mesh_motion(
        monitor, source, moved, boundary_residual=0.0, adaptation_policy=policy
    )
    result = advance.result
    assert advance.decision is phx.meshing.MeshMotionDecision.ACCEPT_MOTION
    assert result is not None
    np.testing.assert_array_equal(np.asarray(result.mesh.coordinates), moved)
    assert [zone.name for zone in result.zones] == ["plate"]
    assert result.zones[0].scope.source_id == result.mesh.mesh_id
    np.testing.assert_array_equal(
        np.sort(np.asarray(result.patches[0].scope.entity_ids)),
        _wall_edge_ids(result.mesh),
    )
    assert {value.association_id for value in result.associations}.isdisjoint(
        {value.association_id for value in source.associations}
    )
    assert len(result.associations) == len(source.associations)
