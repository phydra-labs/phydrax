"""Author independent polyhedral cut carriers and sheared mapped hex sources."""

from __future__ import annotations

from itertools import permutations, product

import numpy as np

import phydrax as phx
from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    ExactPowerCellGeometrySource,
)
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry import MappedReferenceDomain, PiecewiseLinearDomain
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingResult,
    MeshingEntityKind,
    MeshingScope,
    PiecewiseLinearComplex,
)
from phydrax.meshing._hex_frame_records import PreparedSourceHexFrame
from phydrax.meshing._host_banks import entities
from phydrax.meshing._volume_generation import declared_plc_domain
from phydrax.meshing.qualification import (
    curved_occupied_source,
)


def pentagonal_cut_prisms(
    layers: int = 1,
) -> tuple[CellMesh, CellGeometrySpec, PiecewiseLinearDomain]:
    # Independently authored non-orthogonal pentagon, not a fitted source chart.
    polygon = ((0.0, 0.0), (2.0, 0.0), (3.0, 1.0), (1.0, 3.0), (0.0, 2.0))
    points = np.asarray([(x, y, float(z)) for z in range(layers + 1) for x, y in polygon])
    cells, tets = [], []
    for layer in range(layers):
        lower, upper = 5 * layer, 5 * (layer + 1)
        cells.append(
            [
                tuple(lower + row for row in reversed(range(5))),
                tuple(upper + row for row in range(5)),
                *[
                    (
                        lower + row,
                        lower + (row + 1) % 5,
                        upper + (row + 1) % 5,
                        upper + row,
                    )
                    for row in range(5)
                ],
            ]
        )
        for row in range(1, 4):
            a, b, c = lower, lower + row, lower + row + 1
            A, B, C = upper, upper + row, upper + row + 1
            tets.extend(((a, b, c, C), (a, C, b, B), (a, B, C, A)))
    mesh = CellMesh.from_polyhedra(points, cells)
    # One site retains the independently authored prism carrier.
    # Vertex references name its carrier points rather than fitted planes.
    source = ExactPowerCellGeometrySource(
        np.asarray([[1.0, 1.0, 0.5]]),
        np.zeros(1),
        points,
        np.asarray(tets),
        np.arange(points.shape[0] + 1),
        np.zeros(points.shape[0], dtype=np.int64),
        np.column_stack((np.arange(points.shape[0]), np.full((points.shape[0], 3), -1))),
    )
    original_faces = {}
    for region, loops in enumerate(cells):
        for loop in loops:
            original_faces.setdefault(tuple(sorted(loop)), []).append(
                (tuple(loop), region)
            )
    polygons, regions = [], []
    for entries in original_faces.values():
        loop, negative = entries[0]
        positive = entries[1][1] if len(entries) == 2 else -1
        polygons.append(loop)
        regions.append((positive, negative))
    plc = PiecewiseLinearComplex(
        points,
        polygons,
        np.arange(len(polygons)),
        np.asarray(regions),
        tuple(f"original-material-{row}" for row in range(layers)),
    )
    domain = declared_plc_domain(plc, "independent-pentagonal-original")
    return mesh, CellGeometrySpec.power(mesh, source), domain


def sheared_mapped_hex_source() -> phx.meshing.NativeMappedHexSource:
    """Author two oblique reference charts with the curved material geometry."""
    original_source = curved_occupied_source({(1, 0, 0), (2, 0, 0)}, materials=True)
    original = original_source.domain
    shear = np.asarray(((1.0, 0.5, 0.0), (0.0, 1.0, 0.25), (0.0, 0.0, 1.0)))
    reference = CellMesh(
        np.asarray(original.reference_mesh.coordinates) @ shear.T,
        original.reference_mesh.blocks,
        vertex_global_ids=original.reference_mesh.vertex_global_ids,
    )
    original_plc = original_source.reference.complex
    loops = [
        original_plc.polygon_vertices[
            original_plc.polygon_offsets[row] : original_plc.polygon_offsets[row + 1]
        ]
        for row in range(len(original_plc.polygon_facets))
    ]
    plc = PiecewiseLinearComplex(
        np.asarray(original_plc.vertices) @ shear.T,
        loops,
        original_plc.polygon_facets,
        original_plc.facet_regions,
        original_plc.region_ids,
        segments=original_plc.segments,
    )
    reference_source = phx.meshing.NativePlcSource(
        plc, "authored-oblique-reference", "original-r1"
    )
    boundary = declared_plc_domain(plc, reference_source.source_id)
    domain = MappedReferenceDomain(
        boundary,
        reference,
        original.source_geometry,
        original.cell_regions,
        source_id="independently-authored-oblique-mapped-source",
        source_revision="original-r1",
    )
    return phx.meshing.NativeMappedHexSource(reference_source, domain)


def sheared_hex_specification(
    source: phx.meshing.NativeMappedHexSource,
    prepared: PreparedSourceHexFrame,
) -> phx.meshing.VolumeMeshingSpec:
    """Declare the two-material hex request and shared interface."""
    domain = source.domain
    reference = domain.reference_mesh
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        reference.topology.entities(2).entity_ids,
    )
    root_ids = np.asarray(reference.topology.entities(3).entity_ids)
    regions = tuple(
        phx.meshing.RegionControl(
            phx.meshing.MeshingScope(
                source.source_id,
                source.source_revision,
                phx.meshing.MeshingEntityKind.GEOMETRY,
                3,
                domain.entity_set_id(3),
                root_ids[domain.cell_regions == row],
            ),
            name,
            f"material-{name}",
            phx.meshing.RegionRole.SOLID,
        )
        for row, name in enumerate(domain.region_ids)
    )
    connectivity = reference.connectivity
    if not isinstance(connectivity, HexahedralConnectivity):
        raise TypeError(
            f"sheared_source.connectivity must be hexahedral; got {type(connectivity).__name__}"
        )
    common = set(prepared.root_cells[0]) & set(prepared.root_cells[1])
    source_faces = np.asarray(connectivity.faces)
    interface_ids = np.asarray(reference.topology.entities(2).entity_ids)[
        np.asarray([set(face) == common for face in source_faces], dtype=np.bool_)
    ]
    interface = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        interface_ids,
    )
    return phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("hexahedron",)),
            geometry_order=2,
        ),
        scope,
        phx.meshing.VolumeFillStrategy.MULTIZONE,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.5, maximum_size=1.0),),
        size_compliance=phx.meshing.SizeCompliancePolicy(relative_tolerance=0.95),
        region_controls=regions,
        patch_controls=(
            phx.meshing.PatchControl(
                "original-material-interface",
                interface,
                ("left", "right"),
            ),
        ),
    )


def hex_volumes(mesh: CellMesh, /) -> np.ndarray:
    """Independent tensor Gauss integration of trilinear cell maps."""
    cells = np.concatenate(
        [
            np.asarray(block.vertices)
            for block in mesh.blocks
            if block.cell_kind == "hexahedron"
        ]
    )
    vertices = np.asarray(mesh.coordinates)[cells]
    reference = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    samples = 0.5 + np.asarray((-1.0, 1.0), dtype=np.float64) / (2.0 * np.sqrt(3.0))
    volumes = np.zeros((vertices.shape[0],), dtype=np.float64)
    for point in product(samples, repeat=3):
        point_ = np.asarray(point, dtype=np.float64)
        factors = np.where(reference == 1.0, point_, 1.0 - point_)
        derivative = np.empty((8, 3), dtype=np.float64)
        for axis in range(3):
            others = [dimension for dimension in range(3) if dimension != axis]
            derivative[:, axis] = np.where(
                reference[:, axis] == 1.0, 1.0, -1.0
            ) * np.prod(factors[:, others], axis=1)
        jacobians = np.swapaxes(vertices, 1, 2) @ derivative
        determinants = np.linalg.det(jacobians)
        assert np.all(determinants > 0.0)
        volumes += determinants / 8.0
    return volumes


def two_tetrahedra() -> CellMesh:
    return CellMesh.from_tetrahedra(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int64),
        numeric_version="material-source",
    )


def occupied_cells(occupied: set[tuple[int, int, int]], /) -> CellMesh:
    keys = sorted(
        {
            tuple(origin[axis] + offset[axis] for axis in range(3))
            for origin in occupied
            for offset in product((0, 1), repeat=3)
        }
    )
    identity = {key: row for row, key in enumerate(keys)}
    tetrahedra = []
    for origin in sorted(occupied):
        for order in permutations(range(3)):
            point = list(origin)
            vertices = [identity[tuple(point)]]
            for axis in order:
                point[axis] += 1
                vertices.append(identity[tuple(point)])
            tetrahedra.append(vertices)
    coordinates = np.asarray(keys, dtype=np.float64)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    corners = coordinates[cells]
    negative = (
        np.linalg.det(
            np.stack(
                (
                    corners[:, 1] - corners[:, 0],
                    corners[:, 2] - corners[:, 0],
                    corners[:, 3] - corners[:, 0],
                ),
                axis=-1,
            )
        )
        < 0.0
    )
    cells[negative] = cells[negative][:, (0, 2, 1, 3)]
    return CellMesh.from_tetrahedra(
        coordinates, cells, numeric_version="occupied-cell-domain"
    )


def mapped_occupied_source(
    occupied: set[tuple[int, int, int]],
    /,
    *,
    materials: bool = False,
    curvature: float = 1.0 / 64.0,
) -> tuple[PiecewiseLinearComplex, MappedReferenceDomain]:
    source = curved_occupied_source(occupied, materials=materials, curvature=curvature)
    return source.reference.complex, source.domain


def nonunit_material_publication(
    axis_scale: float, maximum_size: float, /
) -> tuple[CellMeshingResult, MappedReferenceDomain]:
    source = curved_occupied_source(
        {(0, 0, 0), (1, 0, 0), (2, 0, 0)},
        materials=True,
        chart_scale=(axis_scale, 0.5, 1.0),
    )
    domain, root = source.domain, source.domain.reference_mesh
    faces = root.topology.entities(2)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        faces.entity_ids,
    )
    root_ids = np.asarray(root.topology.entities(3).entity_ids)
    controls = tuple(
        phx.meshing.RegionControl(
            MeshingScope(
                source.source_id,
                source.source_revision,
                MeshingEntityKind.GEOMETRY,
                3,
                domain.entity_set_id(3),
                root_ids[domain.cell_regions == index],
            ),
            name,
            f"material-{name}",
            phx.meshing.RegionRole.SOLID,
        )
        for index, name in enumerate(domain.region_ids)
    )
    face_vertices = entities(root, 2)
    positions = np.asarray(root.coordinates)[face_vertices]
    interface_ids = np.asarray(faces.entity_ids)[
        np.all(positions[:, :, 0] == 2.0 * axis_scale, axis=1)
    ]
    interface = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        interface_ids,
    )
    request = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, CellFamilyPolicy(required=("hexahedron",)), geometry_order=2
        ),
        scope,
        phx.meshing.VolumeFillStrategy.MULTIZONE,
        size_controls=(
            phx.meshing.UniformSizeControl(scope, 1.0, maximum_size=maximum_size),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.95, target_statistics=("p50",)
        ),
        region_controls=controls,
        patch_controls=(
            phx.meshing.PatchControl("material-interface", interface, ("left", "right")),
        ),
    )
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("mapped_frame_grid_hex")
        )
        .plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute()
    )
    return result, domain
