#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bind one physical interface between two independently meshed parts of a B-Rep.

Two rectangles share the B-Rep edge x = 0. Each rectangle is meshed as its own
named part; each part's certified edge association classifies its interface
edges on that B-Rep edge. The binding records the geometry revision, each part
revision, the association witnesses, and which side of the B-Rep normal each
part lies on. Moving one part produces a new part revision and the binding
refuses it until the interface is re-attached.
"""

import json

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.gp import gp_Pnt
from OCP.TopoDS import TopoDS_Face

import phydrax as phx
from phydrax.geometry.surface import InterfaceSide
from phydrax.solver.coupling import InterfaceBinding, InterfaceEndpoint, InterfaceSource


CONTRACT = phx.SpatialCoordinateContract.si()
POLICY = phx.meshing.AssociationPropagationPolicy(classification_tolerance=1e-8)


def rectangle(x0: float, x1: float) -> TopoDS_Face:
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in ((x0, 0.0), (x1, 0.0), (x1, 1.0), (x0, 1.0)):
        polygon.Add(gp_Pnt(x, y, 0.0))
    polygon.Close()
    return BRepBuilderAPI_MakeFace(polygon.Wire()).Face()


def grid(x0: float, x1: float, count: int) -> phx.discretization.CellMesh:
    xs = np.linspace(x0, x1, count)
    points = np.asarray([[x, y] for y in np.linspace(0.0, 1.0, count) for x in xs])
    triangles = [
        cell
        for row in range(count - 1)
        for column in range(count - 1)
        for a in (row * count + column,)
        for cell in ([a, a + 1, a + count + 1], [a, a + count + 1, a + count])
    ]
    return phx.discretization.CellMesh.from_triangles(
        points, np.asarray(triangles, dtype=np.int64)
    )


shape = BRepAlgoAPI_Fuse(rectangle(-1.0, 0.0), rectangle(0.0, 1.0)).Shape()
model = phx.geometry.model_from_occt_shape(
    shape, coordinate_contract=CONTRACT, linear_deflection=0.05, angular_deflection=0.2
)
projection = phx.geometry.prepare_brep_projection(
    model,
    shape,
    embedding=phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
)

# The authoritative wall is the B-Rep edge through (0, 0.5), found by projection.
edge_count = int(np.asarray(projection.entity_counts)[1])
residuals = np.asarray(
    projection.project(
        np.tile([[0.0, 0.5]], (edge_count, 1)),
        np.ones((edge_count,), dtype=np.int64),
        np.arange(edge_count, dtype=np.int64),
    ).residuals
)
wall = phx.geometry.brep_entity_id(
    model.source_revision, 1, int(np.flatnonzero(residuals < 1e-12)[0])
)


def attach(
    name: str, mesh: phx.discretization.CellMesh
) -> tuple[phx.meshing.MeshPart, phx.meshing.MeshInterfaceAttachment]:
    """Certify one part with its B-Rep associations and attach its wall edges."""
    vertex = phx.meshing.associate_mesh_vertices(mesh, projection, policy=POLICY)
    edges = phx.meshing.associate_mesh_entities(
        mesh, vertex, projection, 1, policy=POLICY
    )
    part = phx.meshing.MeshPart(
        name, phx.meshing.certify_cell_mesh(mesh, CONTRACT, associations=(vertex, edges))
    )
    rows = [row for row, value in enumerate(edges.source_entity_ids) if value == wall]
    attachment = phx.meshing.MeshInterfaceAttachment(
        part,
        part.scope(1, np.asarray(edges.target_global_ids)[rows]),
        edges,
        (wall,),
        tolerance=1e-9,
        region=part.scope(2, np.asarray(mesh.entity_set(2).entity_ids)),
    )
    return part, attachment


left, left_wall = attach("spectral-region", grid(-1.0, 0.0, 4))
right, right_wall = attach("virtual-region", grid(0.0, 1.0, 3))
minus, plus = (
    (left_wall, right_wall) if left_wall.orientation > 0 else (right_wall, left_wall)
)
binding = InterfaceBinding(
    "material-wall",
    InterfaceSource(model.source_id, model.source_revision, (wall,)),
    "two-sided",
    (
        InterfaceEndpoint(minus.part_name, minus, fields={"value": "temperature"}),
        InterfaceEndpoint(plus.part_name, plus, fields={"value": "temperature"}),
    ),
)
binding.require_current(phx.meshing.MeshAssembly((left, right)))

moved_mesh = grid(0.0, 1.0, 3).with_coordinates(
    grid(0.0, 1.0, 3).coordinates + np.asarray([0.05, 0.0]), numeric_version="moved"
)
moved = phx.meshing.MeshPart(
    "virtual-region", phx.meshing.certify_cell_mesh(moved_mesh, CONTRACT)
)
refusals = []
try:
    binding.require_current(left, moved)
except ValueError as error:
    refusals.append(str(error))
refined, refined_wall = attach("virtual-region", grid(0.0, 1.0, 5))
rebound = InterfaceBinding(
    binding.interface_id,
    binding.source,
    "two-sided",
    tuple(
        InterfaceEndpoint(
            endpoint.role,
            refined_wall if endpoint.attachment is right_wall else endpoint.attachment,
            fields=dict(endpoint.fields),
        )
        for endpoint in binding.endpoints
    ),
)
rebound.require_current(left, refined)
if not refusals or rebound.interface_id != binding.interface_id:
    raise RuntimeError("Interface binding did not refuse the moved endpoint.")
print(
    json.dumps(
        {
            "interface_id": binding.interface_id,
            "geometry_revision": binding.source.source_revision[:16],
            "wall_entity": wall.rsplit(":", 2)[1:],
            "minus": binding.side(InterfaceSide.MINUS).role,
            "plus": binding.side(InterfaceSide.PLUS).role,
            "endpoints": [
                {
                    "role": endpoint.role,
                    "part_revision": endpoint.attachment.part_revision[:16],
                    "orientation": endpoint.orientation,
                    "witness": endpoint.witness_id[:16],
                    "maximum_residual": endpoint.attachment.maximum_residual,
                }
                for endpoint in binding.endpoints
                if isinstance(endpoint.attachment, phx.meshing.MeshInterfaceAttachment)
            ],
            "binding_id": binding.binding_id[:16],
            "moved_endpoint_refusal": refusals[0],
            "rebound_binding_id": rebound.binding_id[:16],
        },
        indent=2,
    )
)
