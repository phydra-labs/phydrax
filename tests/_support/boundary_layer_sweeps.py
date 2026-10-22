"""Independent source and request builders for periodic, CAD, and affine-sweep layers."""

from dataclasses import dataclass

import numpy as np

import phydrax as phx
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    PeriodicIsometryGroup,
    PeriodicMeshTopology,
)
from phydrax.geometry import PiecewiseLinearDomain
from tests._support.boundary_layer_walls import SCHEDULE, whole


def face_points(model: phx.geometry.BRepModel, face: int) -> np.ndarray:
    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            "boundary_layer_fixture.geometry must contain native incidence; got None"
        )
    vertices = sorted(
        {
            vertex
            for loop in geometry.face_loops[face]
            for coedge in loop
            for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
        }
    )
    return np.asarray(geometry.vertex_points, dtype=np.float64)[vertices]


def brep_scope(
    model: phx.geometry.BRepModel, dimension: int, entities: tuple[int, ...]
) -> phx.meshing.MeshingScope:
    domain = phx.geometry.MeshingDomain.from_brep(model)
    definitions = (
        domain.region_source_indices
        if dimension == 3
        else domain.source_indices[dimension]
    )
    selected = set(entities)
    slots = domain.scope_indices(dimension)
    identifiers = tuple(
        slots[row] for row, definition in enumerate(definitions) if definition in selected
    )
    return phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        dimension,
        domain.entity_set_id(dimension),
        np.asarray(identifiers, dtype=np.int64),
    )


def periodic_wall(
    periodicity: str, quadrilateral: bool
) -> tuple[
    CellMesh,
    phx.meshing.BoundaryLayerControl,
    PeriodicIsometryGroup,
    PeriodicMeshTopology,
    float,
    float,
]:
    if periodicity == "rotation":
        points = np.asarray(
            ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 0.0))
        )
        generators = np.asarray(
            (
                (
                    (0.0, -1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            )
        )
        representatives = np.asarray((0, 1, 1, 0), dtype=np.int64)
        shifts = np.asarray(((0,), (0,), (1,), (1,)), dtype=np.int64)
        area = 0.375
    elif periodicity in ("translation", "partial"):
        points = np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
        generators = np.repeat(np.eye(4)[None], 2, axis=0)
        generators[0, 0, 3] = 1.0
        generators[1, 1, 3] = 1.0
        representatives = np.zeros(4, dtype=np.int64)
        shifts = points[:, :2].astype(np.int64)
        if periodicity == "partial":
            generators = generators[:1]
            representatives = np.asarray((0, 0, 3, 3), dtype=np.int64)
            shifts = points[:, :1].astype(np.int64)
        area = 1.0
    else:
        raise ValueError(
            f"layer_fixture.periodicity must select a supported pairing; got {periodicity!r}"
        )
    rows = np.asarray(
        ((0, 1, 2, 3),) if quadrilateral else ((0, 1, 2), (0, 2, 3)), dtype=np.int64
    )
    block = phx.discretization.CellBlock(
        "wall", "quadrilateral" if quadrilateral else "triangle", rows
    )
    lifted = phx.discretization.CellMesh(points, (block,))
    identification = PeriodicIsometryGroup(generators)
    topology = PeriodicMeshTopology(lifted, identification, representatives, shifts)
    wall = phx.discretization.CellMesh(points, (block,), periodic_topology=topology)
    schedule = SCHEDULE
    control = phx.meshing.BoundaryLayerControl(
        whole(wall),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
    )
    transition_height = (
        0.5 * np.mean(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1))
        if quadrilateral
        else 0.0
    )
    return wall, control, identification, topology, area, transition_height


def ridge_wall() -> tuple[CellMesh, phx.meshing.BoundaryLayerControl]:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, 0.1),
            (1.0, 0.5, 0.1),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        )
    )
    block = phx.discretization.CellBlock(
        "wall",
        "triangle",
        np.asarray(((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)), dtype=np.int64),
    )
    lifted = phx.discretization.CellMesh(points, (block,))
    generator = np.eye(4)[None]
    generator[0, 0, 3] = 1.0
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicIsometryGroup(generator),
        np.asarray((0, 0, 2, 2, 4, 4), dtype=np.int64),
        np.asarray(((0,), (1,), (0,), (1,), (0,), (1,)), dtype=np.int64),
    )
    wall = phx.discretization.CellMesh(points, (block,), periodic_topology=topology)
    schedule = SCHEDULE
    control = phx.meshing.BoundaryLayerControl(
        whole(wall),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        feature_angle=0.1,
        corner=phx.meshing.BoundaryLayerCornerPolicy.FAN,
    )
    return wall, control


def cad_request(
    remaining_core: bool,
) -> tuple[phx.geometry.BRepModel, phx.meshing.BoundaryLayerControl]:
    thicknesses = (0.05, 0.10) if remaining_core else (0.25, 0.75)
    source = phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 1.0),
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
        tessellation=phx.geometry.BRepTessellationPolicy(
            linear_deflection=0.05, angular_deflection=0.2
        ),
    )
    walls = tuple(
        face
        for face in range(source.topology.num_faces)
        if np.all(face_points(source, face)[:, 2] == 0.0)
    )
    schedule = phx.meshing.LayerSchedule(np.asarray(thicknesses, dtype=np.float64))
    request = phx.meshing.BoundaryLayerControl(
        brep_scope(source, 2, walls),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.CAD_EXTRUSION,
        volume_scope=brep_scope(source, 3, (0,)),
    )
    return source, request


def invalid_cad_request(
    opposite_walls: bool,
) -> tuple[phx.geometry.BRepModel, phx.meshing.BoundaryLayerControl, str]:
    thickness, reason = (
        (0.75, "slabs of different walls overlap")
        if opposite_walls
        else (1.25, "leaves its controlled solid")
    )
    source = phx.geometry.brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 1.0),
        coordinate_contract=phx.SpatialCoordinateContract(phx.units.MILLIMETER),
    )
    walls = tuple(
        face
        for face in range(source.topology.num_faces)
        if np.all(face_points(source, face)[:, 2] == 0.0)
        or (opposite_walls and np.all(face_points(source, face)[:, 2] == 1.0))
    )
    request = phx.meshing.BoundaryLayerControl(
        brep_scope(source, 2, walls),
        phx.meshing.LayerSchedule(np.asarray((thickness,), dtype=np.float64)),
        route=phx.meshing.BoundaryLayerRoute.CAD_EXTRUSION,
        volume_scope=brep_scope(source, 3, (0,)),
    )
    return source, request, reason


@dataclass(frozen=True)
class ExactLayerSweepFixture:
    wall: CellMesh
    domain: PiecewiseLinearDomain
    control: phx.meshing.BoundaryLayerControl
    sweep: phx.meshing.SweepControl
    association: phx.meshing.GeometryAssociation
    physical: phx.meshing.BoundaryLayerMesh
    points: np.ndarray
    stations: np.ndarray
    actual_points: np.ndarray
    actual_columns: np.ndarray


def exact_sweep_fixture(
    periodic: bool, perturb_cap: bool = False
) -> ExactLayerSweepFixture:
    M = phx.meshing
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (1.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    rows = np.asarray(((0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)), dtype=np.int64)
    block = CellBlock("wall", "triangle", rows)
    wall = CellMesh(points, (block,))
    if periodic:
        generator = np.eye(4, dtype=np.float64)[None]
        generator[0, 0, 3] = 1.0
        topology = PeriodicMeshTopology(
            wall,
            PeriodicIsometryGroup(generator),
            np.asarray((0, 0, 2, 2, 4, 4), dtype=np.int64),
            np.asarray(((0,), (1,), (0,), (1,), (0,), (1,)), dtype=np.int64),
        )
        wall = CellMesh(points, (block,), periodic_topology=topology)
    schedule = M.LayerSchedule.geometric(2, 0.01, growth_rate=1.0)
    physical_control = M.BoundaryLayerControl(
        whole(wall),
        schedule,
        route=M.BoundaryLayerRoute.ADVANCING,
        corner=M.BoundaryLayerCornerPolicy.SMOOTH,
    )
    physical = M.prepare_boundary_layers(wall, physical_control)
    actual_points = np.asarray(physical.mesh.coordinates).copy()
    actual_columns = np.asarray(physical.column_index).copy()
    stations = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(np.asarray(schedule.thicknesses, dtype=np.float64)),
        )
    )
    upper = points + stations[-1] * np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if perturb_cap:
        upper[2, 2] = np.nextafter(upper[2, 2], np.inf)
    edges = np.asarray(((0, 1), (1, 3), (3, 5), (5, 4), (4, 2), (2, 0)), dtype=np.int64)
    a, b = edges.T
    sides = np.stack(
        (np.stack((a, b, 6 + b), axis=1), np.stack((a, 6 + b, 6 + a), axis=1)), axis=1
    ).reshape(-1, 3)
    facets = np.concatenate((rows[:, ::-1], rows + 6, sides))
    domain = PiecewiseLinearDomain(
        np.concatenate((points, upper)),
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (facets.shape[0], 1)),
        ("wall-material",),
        source_id="independent-reference-slab",
    )
    scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        f"{domain.domain_id}:facets",
        np.arange(4, dtype=np.int64),
    )
    cap_scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        scope.entity_set_id,
        np.arange(4, 8, dtype=np.int64),
    )
    volume = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        3,
        f"{domain.domain_id}:regions",
        np.asarray([0], dtype=np.int64),
    )
    association = M.GeometryAssociation(
        M.GeometryAssociationKind.PIECEWISE_LINEAR,
        domain.source_id,
        domain.source_revision,
        wall.entity_set(2).entity_set_id,
        wall.entity_set(2).entity_ids,
        tuple(f"{domain.source_revision}:facet:{row}" for row in range(4)),
        np.zeros(4, dtype=np.float64),
        exact=True,
        source_dimensions=np.full(4, 2, dtype=np.int64),
        source_indices=np.arange(4, dtype=np.int64),
        source_entity_roles=(M.GeometrySourceEntityRole.FACET,) * 4,
        orientations=-np.ones(4, dtype=np.int32),
    )
    control = M.BoundaryLayerControl(
        scope,
        schedule,
        route=M.BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=volume,
        cap_scope=cap_scope,
    )
    sweep = M.SweepControl(
        M.SweepMapKind.EXTRUSION, schedule, "original-reference-schedule"
    )
    return ExactLayerSweepFixture(
        wall,
        domain,
        control,
        sweep,
        association,
        physical,
        points,
        stations,
        actual_points,
        actual_columns,
    )
