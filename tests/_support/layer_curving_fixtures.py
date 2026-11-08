from __future__ import annotations

import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    CellValidityPolicy,
    certify_cell_geometry_validity,
    reference_cell_topology,
)
from phydrax.discretization._cell_geometry import (
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_enclosure import axes
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryQuery,
)
from phydrax.geometry._surface_source_support import SurfaceSourceCharts
from phydrax.geometry.brep._patches import LineCurve, RuledSurface
from phydrax.meshing._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._boundary_layer import (
    BoundaryLayerMesh,
    BoundaryLayerPolicy,
    prepare_boundary_layers,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import (
    BoundaryLayerControl,
    BoundaryLayerCornerPolicy,
    BoundaryLayerRoute,
    LayerSchedule,
    RegionSeed,
)
from phydrax.meshing._curving import HighOrderCurvingPolicy
from phydrax.meshing._layer_core import generate_layer_core
from phydrax.meshing._layer_core_topology import LayerCoreConstruction
from phydrax.meshing._layer_curving import (
    _construction_with_geometry,
    curve_layer_core_construction,
    LayerCurvingConstructionResult,
    LayerCurvingStatus,
    renew_layer_curving_source_evidence,
)
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers._native_layer import (
    _lower_core_specification,
    _mapped_layer_surface_chart_witness,
)
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import (
    NativeLayerCoreSource,
    NativeMappedHexSource,
    NativePlcSource,
)


def _boundary_layer_control(
    wall: CellMesh,
    thickness: float,
    /,
    *,
    corner: BoundaryLayerCornerPolicy = BoundaryLayerCornerPolicy.FAN,
) -> BoundaryLayerControl:
    cells = wall.entity_set(2)
    scope = MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        np.asarray(cells.entity_ids, dtype=np.int64),
    )
    return BoundaryLayerControl(
        scope,
        LayerSchedule.geometric(1, thickness, growth_rate=1.2),
        route=BoundaryLayerRoute.ADVANCING,
        corner=corner,
    )


def _layer_mesh(
    wall: CellMesh,
    wall_association: GeometryAssociation | None,
    /,
    *,
    thickness: float = 0.02,
    corner: BoundaryLayerCornerPolicy = BoundaryLayerCornerPolicy.FAN,
) -> BoundaryLayerMesh:
    return prepare_boundary_layers(
        wall,
        _boundary_layer_control(wall, thickness, corner=corner),
        wall_association=wall_association,
    )


def _mapped_cube_source_for_layer() -> tuple[
    NativeMappedHexSource, MappedReferenceDomain, CellMesh
]:
    source_id = "mapped-layer-curving-source"
    source_revision = "mapped-layer-curving-r1"
    scale = 1.0
    translation = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    vertices = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    triangles = np.asarray(
        (
            (0, 3, 2),
            (0, 2, 1),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ),
        dtype=np.int64,
    )
    root_vertex_ids = np.asarray((11, 13, 17, 19, 23, 29, 31, 37), dtype=np.int64)
    segment_vertices = np.asarray(
        tuple(
            tuple(sorted(int(value) for value in edge))
            for edge in reference_cell_topology("hexahedron").entities[1]
        ),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        tuple(row for row in triangles),
        np.arange(triangles.shape[0], dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (triangles.shape[0], 1)),
        ("solid",),
        segments=segment_vertices,
        vertex_ids=tuple(f"mapped-cube-vertex-{row}" for row in range(8)),
        vertex_occurrence_paths=((),) * 8,
        facet_ids=tuple(f"mapped-cube-facet-{row}" for row in range(12)),
        facet_occurrence_paths=((),) * 12,
        segment_ids=tuple(f"mapped-cube-edge-{row}" for row in range(12)),
        segment_occurrence_paths=((),) * 12,
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    reference_domain = PiecewiseLinearDomain(
        vertices,
        triangles,
        np.tile(np.asarray((0, -1), dtype=np.int64), (triangles.shape[0], 1)),
        ("solid",),
        source_id=source_id,
    )
    reference_mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "root",
                "hexahedron",
                np.arange(8, dtype=np.int32)[None, :],
                global_ids=np.asarray((701,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=root_vertex_ids,
    )
    affine = CellGeometrySpec.affine(reference_mesh)
    source_geometry = CellGeometrySpec(
        dict(zip(affine.block_names, affine.elements, strict=True)),
        {
            name: np.asarray(route)
            for name, route in zip(affine.block_names, affine.geometry_dofs, strict=True)
        },
        scale * vertices + translation,
    )
    domain = MappedReferenceDomain(
        reference_domain,
        reference_mesh,
        source_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id=source_id,
        source_revision=source_revision,
        reference_domain_vertex_root_ids=root_vertex_ids,
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    source = NativeMappedHexSource(
        NativePlcSource(complex_, source_id, source_revision),
        domain,
        plc_vertex_root_global_ids=root_vertex_ids,
    )
    return source, domain, reference_mesh


def _mapped_cube_wall(
    source: NativeMappedHexSource,
    domain: MappedReferenceDomain,
    /,
    *,
    physical: bool,
) -> CellMesh:
    complex_ = source.reference.complex
    triangles = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape((-1, 3))
    points = (
        np.asarray(domain.source_geometry.coordinates, dtype=np.float64)
        if physical
        else np.asarray(domain.reference_mesh.coordinates, dtype=np.float64)
    )
    cell_ids = np.asarray(complex_.facet_entity_ids, dtype=np.int64)[
        np.asarray(complex_.polygon_facets, dtype=np.int64)
    ]
    return CellMesh(
        points,
        (
            CellBlock(
                "mapped-wall",
                "triangle",
                triangles[:, ::-1],
                global_ids=cell_ids,
            ),
        ),
        vertex_global_ids=np.asarray(
            domain.reference_mesh.vertex_global_ids, dtype=np.int64
        ),
    )


def _mapped_cube_surface_charts(
    wall: CellMesh,
    source_id: str,
    source_revision: str,
    /,
) -> tuple[MeshingDomain, SurfaceSourceCharts]:
    """Build direct CAD chart authority for every planar face of the root cube."""
    face_vertices = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    chart_corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    root_points = np.asarray(wall.coordinates, dtype=np.float64)
    patches = []
    curves = []
    corners = []
    for patch_row, vertex_rows in enumerate(face_vertices):
        base = 4 * patch_row
        p0, p1, p2, p3 = root_points[np.asarray(vertex_rows, dtype=np.int64)]
        surface = RuledSurface(
            LineCurve(tuple(p0), tuple(p1 - p0)),
            LineCurve(tuple(p3), tuple(p2 - p3)),
        )
        uses = (
            PatchCurveUse(base, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
            PatchCurveUse(base + 1, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
            PatchCurveUse(base + 2, LineCurve((1.0, 1.0), (-1.0, 0.0)), 0.0, 1.0),
            PatchCurveUse(
                base + 3,
                LineCurve((0.0, 0.0), (0.0, 1.0)),
                0.0,
                1.0,
                sense=-1,
            ),
        )
        patches.append(MeshingSurfacePatch(surface, (uses,)))
        curves.extend(
            (
                MeshingDomainCurve(base, base + 1),
                MeshingDomainCurve(base + 1, base + 2),
                MeshingDomainCurve(base + 2, base + 3),
                MeshingDomainCurve(base, base + 3),
            )
        )
        corners.extend(
            tuple(float(value) for value in root_points[row]) for row in vertex_rows
        )
    domain = MeshingDomain(
        tuple(patches),
        tuple(curves),
        len(corners),
        source_id=source_id,
        source_revision=source_revision,
        source_indices=(
            tuple(1000 + row for row in range(len(corners))),
            tuple(2000 + row for row in range(len(curves))),
            tuple(3000 + row for row in range(len(face_vertices))),
        ),
        source_occurrences=(
            ((),) * len(corners),
            ((),) * len(curves),
            ((),) * len(face_vertices),
        ),
        corner_points=np.asarray(corners, dtype=np.float64),
    )
    block = wall.blocks[0]
    if not isinstance(block, CellBlock) or block.cell_kind != "triangle":
        raise TypeError("Mapped cube chart fixture requires a triangle wall block.")
    cell_ids = np.asarray(block.global_ids, dtype=np.int64).reshape((-1,))
    triangles = np.asarray(block.vertices, dtype=np.int64).reshape((-1, 3))
    patch_rows = np.repeat(np.arange(len(face_vertices), dtype=np.int64), 2)
    if triangles.shape != (patch_rows.size, 3) or cell_ids.shape != patch_rows.shape:
        raise ValueError(
            "Mapped cube wall facet identities do not match its six source charts."
        )
    chart_uvs = []
    for row in range(triangles.shape[0]):
        triangle = triangles[row]
        chart_vertices = {
            int(vertex): chart_corners[corner]
            for corner, vertex in enumerate(face_vertices[int(patch_rows[row])])
        }
        try:
            chart_uvs.append(
                tuple(chart_vertices[int(triangle[corner])] for corner in range(3))
            )
        except KeyError as error:
            raise ValueError(
                "Mapped cube facet is outside its authored source face."
            ) from error
    target_entities = np.asarray(wall.entity_set(2).entity_ids, dtype=np.int64)
    patch_by_cell_id = {
        int(cell_id): int(patch_rows[row]) for row, cell_id in enumerate(cell_ids)
    }
    association_patch_rows = np.asarray(
        [patch_by_cell_id[int(cell_id)] for cell_id in target_entities],
        dtype=np.int64,
    )
    root_scope_ids_by_key = {
        domain.source_entity_key_for_scope(2, int(scope_id)): int(scope_id)
        for scope_id in domain.scope_entity_ids(2)
    }
    root_patch_ids = np.asarray(
        [
            root_scope_ids_by_key[
                (domain.entity_id(2, patch), tuple(domain.source_occurrences[2][patch]))
            ]
            for patch in range(len(face_vertices))
        ],
        dtype=np.int64,
    )
    association = GeometryAssociation(
        GeometryAssociationKind.SURFACE,
        source_id,
        source_revision,
        wall.entity_set(2).entity_set_id,
        target_entities,
        tuple(domain.entity_id(2, int(patch)) for patch in association_patch_rows),
        np.zeros((target_entities.size,), dtype=np.float64),
        source_dimensions=np.full((target_entities.size,), 2, dtype=np.int8),
        source_indices=np.asarray(
            [
                int(domain.source_indices[2][int(patch)])
                for patch in association_patch_rows
            ],
            dtype=np.int64,
        ),
        source_roles=(GeometrySourceEntityRole.FACET,) * target_entities.size,
        source_occurrence_paths=((),) * target_entities.size,
    )
    root_cell_ids = root_patch_ids[np.asarray(patch_rows, dtype=np.int64)]
    source_pieces = tuple(
        (
            int(patch_rows[row]),
            int(domain.source_indices[2][int(patch_rows[row])]),
        )
        for row in range(cell_ids.size)
    )
    charts = SurfaceSourceCharts(
        domain,
        wall,
        CellGeometrySpec.affine(wall),
        SpatialCoordinateContract.si(),
        np.asarray(chart_uvs, dtype=np.float64),
        patch_rows,
        (association,),
        MeshingDomainBoundarySource(domain, tuple(range(len(face_vertices)))),
        source_root_cell_ids=root_cell_ids,
        source_piece_ids=source_pieces,
    )
    return domain, charts


def _mapped_core_cap_complex(
    layers: BoundaryLayerMesh,
    root_complex: PiecewiseLinearComplex,
    root_vertex_ids: np.ndarray,
    /,
) -> tuple[PiecewiseLinearComplex, np.ndarray]:
    cap = layers.cap
    if cap is None or any(block.cell_kind != "triangle" for block in cap.blocks):
        raise ValueError("Mapped layer fixture requires a closed triangular core cap.")
    cap_vertices = np.asarray(layers.cap_vertices, dtype=np.int64)
    cap_root_vertices = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)[
        cap_vertices
    ]
    cap_triangles = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in cap.blocks)
    )
    root_polygons = np.asarray(root_complex.polygon_vertices, dtype=np.int64).reshape(
        (-1, 3)
    )
    polygon_facets = np.asarray(root_complex.polygon_facets, dtype=np.int64)
    root_vertex_ids = np.asarray(root_vertex_ids, dtype=np.int64)
    if root_vertex_ids.shape != (root_complex.vertices.shape[0],):
        raise ValueError("Mapped cap root vertex IDs do not match the source PLC.")
    root_vertex_rows: dict[int, int] = {}
    for root_row, root_vertex_id in enumerate(root_vertex_ids):
        if int(root_vertex_id) in root_vertex_rows:
            raise ValueError("Mapped cap root vertex IDs must be unique.")
        root_vertex_rows[int(root_vertex_id)] = root_row
    root_face_rows: dict[tuple[int, ...], int] = {}
    for polygon_row, polygon in enumerate(root_polygons):
        key = tuple(sorted(int(root_vertex_ids[row]) for row in polygon))
        if key in root_face_rows:
            raise ValueError("Mapped core root PLC repeats a source face.")
        root_face_rows[key] = int(polygon_facets[polygon_row])
    cap_root_vertex_rows: dict[int, int] = {}
    for cap_row, root_vertex_id in enumerate(cap_root_vertices):
        if root_vertex_id < 0 or int(root_vertex_id) in cap_root_vertex_rows:
            raise ValueError(
                "Mapped cap vertices must preserve unique exact source-root identities."
            )
        cap_root_vertex_rows[int(root_vertex_id)] = cap_row
    cap_facet_rows: list[int] = []
    for triangle in cap_triangles:
        root_rows = cap_root_vertices[triangle]
        if np.any(root_rows < 0) or np.unique(root_rows).size != 3:
            raise ValueError(
                "Mapped cap triangle lost its three exact root source vertices."
            )
        key = tuple(sorted(int(value) for value in root_rows))
        try:
            cap_facet_rows.append(root_face_rows[key])
        except KeyError as error:
            raise ValueError(
                "Mapped cap triangle has no exact authored root-face ancestry."
            ) from error
    if set(cap_facet_rows) != set(range(len(root_complex.facet_ids))):
        raise ValueError("Mapped core cap omitted an authored source face.")
    root_segments = np.asarray(root_complex.segments, dtype=np.int64).reshape((-1, 2))
    cap_segments: list[tuple[int, int]] = []
    segment_rows: list[int] = []
    for segment_row, root_segment in enumerate(root_segments):
        root_segment_ids = tuple(
            sorted(int(root_vertex_ids[row]) for row in root_segment)
        )
        try:
            cap_segment = tuple(
                sorted(
                    cap_root_vertex_rows[root_vertex_id]
                    for root_vertex_id in root_segment_ids
                )
            )
        except KeyError as error:
            raise ValueError(
                "Mapped cap omitted an authored source-root segment endpoint."
            ) from error
        cap_segments.append((cap_segment[0], cap_segment[1]))
        segment_rows.append(segment_row)
    try:
        cap_vertex_root_rows = tuple(
            root_vertex_rows[int(root_vertex_id)] for root_vertex_id in cap_root_vertices
        )
    except KeyError as error:
        raise ValueError(
            "Mapped cap vertex is absent from the exact source-root PLC."
        ) from error
    segment_ids = tuple(root_complex.segment_ids[row] for row in segment_rows)
    segment_paths = tuple(
        root_complex.segment_occurrence_paths[row] for row in segment_rows
    )
    cap_complex = PiecewiseLinearComplex(
        np.asarray(cap.coordinates, dtype=np.float64),
        tuple(row for row in cap_triangles),
        np.asarray(cap_facet_rows, dtype=np.int64),
        np.tile(
            np.asarray((0, -1), dtype=np.int64),
            (len(root_complex.facet_ids), 1),
        ),
        root_complex.region_ids,
        segments=np.asarray(cap_segments, dtype=np.int64).reshape((-1, 2)),
        vertex_ids=tuple(root_complex.vertex_ids[row] for row in cap_vertex_root_rows),
        vertex_occurrence_paths=tuple(
            root_complex.vertex_occurrence_paths[row] for row in cap_vertex_root_rows
        ),
        facet_ids=root_complex.facet_ids,
        facet_occurrence_paths=root_complex.facet_occurrence_paths,
        segment_ids=segment_ids,
        segment_occurrence_paths=segment_paths,
        region_occurrence_paths=root_complex.region_occurrence_paths,
        boundary="fixed",
    )
    return cap_complex, cap_vertices


def _mixed_construction(
    layers: BoundaryLayerMesh,
    source_id: str,
    source_revision: str,
    source_mesh: CellMesh,
    frame_id: str,
    /,
    *,
    fidelity_source: SourceBoundaryQuery | None = None,
) -> tuple[NativeLayerCoreSource, LayerCoreConstruction]:
    """Generate a native core around an adaptive implicit source wall."""
    if any(block.cell_kind != "triangle" for block in source_mesh.blocks):
        raise ValueError("Implicit layer core fixture requires a triangular source wall.")
    triangles = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in source_mesh.blocks)
    )
    source_cell_ids = np.concatenate(
        tuple(
            np.asarray(block.global_ids, dtype=np.int64) for block in source_mesh.blocks
        )
    )
    if triangles.shape != (source_cell_ids.size, 3):
        raise ValueError(
            "Implicit source-wall cell identities do not align with its facets."
        )
    root_complex = PiecewiseLinearComplex(
        np.asarray(source_mesh.coordinates, dtype=np.float64),
        tuple(row for row in triangles),
        np.arange(triangles.shape[0], dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (triangles.shape[0], 1)),
        (f"{source_id}:layer-core-interior",),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(
            f"{source_id}:root-vertex:{int(value)}"
            for value in np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
        ),
        vertex_occurrence_paths=((),) * source_mesh.coordinates.shape[0],
        facet_ids=tuple(
            f"{source_id}:root-face:{int(value)}" for value in source_cell_ids
        ),
        facet_occurrence_paths=((),) * source_cell_ids.size,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    core_complex, _ = _mapped_core_cap_complex(
        layers,
        root_complex,
        np.asarray(source_mesh.vertex_global_ids, dtype=np.int64),
    )
    probe = _mapped_layer_core_source(
        layers,
        core_complex,
        source_id,
        source_revision,
        source_boundary_scope=None,
    )
    boundary_scope = MeshingScope(
        source_id,
        source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        probe.scope_entity_set_id(2),
        probe.scope_entity_ids(2),
    )
    source = _mapped_layer_core_source(
        layers,
        core_complex,
        source_id,
        source_revision,
        source_boundary_scope=None,
    )
    layer_kinds = {block.cell_kind for block in layers.mesh.blocks}
    family_policy = CellFamilyPolicy(
        required=("tetrahedron",),
        preferred=tuple(sorted(layer_kinds - {"tetrahedron"})),
        allow_mixed=True,
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, family_policy, geometry_order=2),
        boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(
                boundary_scope,
                0.75,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        region_seeds=(
            RegionSeed(
                np.zeros((3,), dtype=np.float64),
                core_complex.region_ids[0],
                "implicit-layer-core",
                RegionRole.SOLID,
            ),
        ),
        maximum_deviation=0.0,
    )
    schedule = NativeMeshingOptions("layer_core").volume_schedule
    if schedule is None:
        raise ValueError(
            "Implicit layer fixture requires the native core volume schedule."
        )
    construction = generate_layer_core(
        source,
        specification,
        _lower_core_specification(source, specification),
        schedule,
        frame_id=frame_id,
    )
    if fidelity_source is not None:
        source = _mapped_layer_core_source(
            layers,
            core_complex,
            source_id,
            source_revision,
            source_boundary_scope=None,
            fidelity_source=fidelity_source,
        )
    return source, construction


def _mapped_layer_core_source(
    layers: BoundaryLayerMesh,
    complex_: PiecewiseLinearComplex,
    source_id: str,
    source_revision: str,
    /,
    *,
    source_boundary_scope: MeshingScope | None,
    fidelity_source: SourceBoundaryQuery | None = None,
    reference_source: NativeLayerCoreSource | None = None,
    mapped_domain: MappedReferenceDomain | None = None,
) -> NativeLayerCoreSource:
    layer_cell_count = sum(block.cell_count for block in layers.mesh.blocks)
    cap = layers.cap
    if cap is None:
        raise ValueError("Native layer/core fixture requires its immutable source cap.")
    cap_cell_count = sum(block.cell_count for block in cap.blocks)
    return NativeLayerCoreSource(
        layers,
        complex_,
        source_id,
        source_revision,
        vertex_layer_ids=np.asarray(layers.cap_vertices, dtype=np.int64),
        cap_polygon_ids=np.arange(cap_cell_count, dtype=np.int64),
        layer_regions=np.zeros((layer_cell_count,), dtype=np.int64),
        region_entity_ids=np.asarray(complex_.region_entity_ids, dtype=np.int64),
        region_occurrence_paths=complex_.region_occurrence_paths,
        region_ids=complex_.region_ids,
        core_region_map=np.arange(len(complex_.region_ids), dtype=np.int64),
        source_boundary_scope=source_boundary_scope,
        fidelity_source=fidelity_source,
        maximum_deviation=(
            0.0 if isinstance(fidelity_source, MappedReferenceBoundarySource) else None
        ),
        core_facet_source_ids=complex_.facet_ids,
        core_facet_source_occurrence_paths=complex_.facet_occurrence_paths,
        reference_source=reference_source,
        mapped_domain=mapped_domain,
    )


def _mapped_layer_core_construction_for_fixture() -> tuple[
    NativeMappedHexSource,
    MappedReferenceDomain,
    CellMesh,
    MappedReferenceBoundarySource,
    CellMesh,
    BoundaryLayerMesh,
    NativeLayerCoreSource,
    LayerCoreConstruction,
    BoundaryLayerControl,
    BoundaryLayerPolicy,
]:
    mapped_source, domain, reference_mesh = _mapped_cube_source_for_layer()
    query = mapped_source.boundary_query()
    if not isinstance(query, MappedReferenceBoundarySource):
        raise TypeError(
            "Mapped layer fixture requires the native mapped-root boundary query."
        )
    wall = _mapped_cube_wall(mapped_source, domain, physical=False)
    control = _boundary_layer_control(wall, 0.02, corner=BoundaryLayerCornerPolicy.SMOOTH)
    policy = BoundaryLayerPolicy()
    layers = prepare_boundary_layers(wall, control, policy=policy)
    core_complex, _ = _mapped_core_cap_complex(
        layers,
        mapped_source.reference.complex,
        np.asarray(reference_mesh.vertex_global_ids, dtype=np.int64),
    )
    reference_source = _mapped_layer_core_source(
        layers,
        core_complex,
        mapped_source.source_id,
        mapped_source.source_revision,
        source_boundary_scope=None,
    )
    probe = _mapped_layer_core_source(
        layers,
        core_complex,
        mapped_source.source_id,
        mapped_source.source_revision,
        source_boundary_scope=None,
        fidelity_source=query,
        reference_source=reference_source,
        mapped_domain=domain,
    )
    boundary_scope = MeshingScope(
        mapped_source.source_id,
        mapped_source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        probe.scope_entity_set_id(2),
        probe.scope_entity_ids(2),
    )
    source = _mapped_layer_core_source(
        layers,
        core_complex,
        mapped_source.source_id,
        mapped_source.source_revision,
        source_boundary_scope=None,
        fidelity_source=query,
        reference_source=reference_source,
        mapped_domain=domain,
    )
    layer_kinds = {block.cell_kind for block in layers.mesh.blocks}
    family_policy = CellFamilyPolicy(
        required=("tetrahedron",),
        preferred=tuple(sorted(layer_kinds - {"tetrahedron"})),
        allow_mixed=True,
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, family_policy, geometry_order=2),
        boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(
                boundary_scope,
                0.75,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        region_seeds=(
            RegionSeed(
                np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
                core_complex.region_ids[0],
                "mapped-layer-core",
                RegionRole.SOLID,
            ),
        ),
        maximum_deviation=0.0,
    )
    options = NativeMeshingOptions("layer_core")
    if options.volume_schedule is None:
        raise ValueError("Mapped layer fixture requires the native core volume schedule.")
    construction = generate_layer_core(
        source,
        specification,
        _lower_core_specification(source, specification),
        options.volume_schedule,
        frame_id=domain.coordinate_contract.spatial_id,
    )
    return (
        mapped_source,
        domain,
        reference_mesh,
        query,
        wall,
        layers,
        source,
        construction,
        control,
        policy,
    )


def make_mapped_layer_curving_archive_case() -> tuple[
    LayerCurvingConstructionResult,
    BoundaryLayerMesh,
    BoundaryLayerControl,
    BoundaryLayerPolicy,
]:
    mapped_source, domain, _, query, _, layers, source, construction, control, policy = (
        _mapped_layer_core_construction_for_fixture()
    )
    witness = _mapped_layer_surface_chart_witness(source, construction, query)
    result = curve_layer_core_construction(
        construction,
        layers,
        None,
        policy=HighOrderCurvingPolicy(degree=2, relaxation_rounds=0),
        fidelity_tolerance=0.0,
        maximum_normal_deviation=0.0,
        fidelity_source=query,
        mapped_chart_witness=witness,
        frame_id=domain.coordinate_contract.spatial_id,
    )
    if (
        result.status is not LayerCurvingStatus.CURVED
        or result.candidate_geometry is None
        or result.source_evidence is None
        or result.source_evidence.mapped_chart_trace is None
    ):
        raise ValueError(
            "Mapped layer archive case requires certified original-root chart evidence: "
            + (result.failure or result.status.value)
        )
    return result, layers, control, policy


def make_layer_curving_archive_case(
    degree: int = 2,
) -> tuple[
    LayerCurvingConstructionResult,
    BoundaryLayerMesh,
    BoundaryLayerControl,
    BoundaryLayerPolicy,
]:
    mapped_source, _, _, _, wall, layers, _, construction, control, policy = (
        _mapped_layer_core_construction_for_fixture()
    )
    _, charts = _mapped_cube_surface_charts(
        wall, mapped_source.source_id, mapped_source.source_revision
    )
    result = curve_layer_core_construction(
        construction,
        layers,
        charts,
        policy=HighOrderCurvingPolicy(degree=degree, relaxation_rounds=0),
        fidelity_tolerance=0.0,
    )
    if (
        result.status is not LayerCurvingStatus.CURVED
        or result.candidate_geometry is None
        or result.source_evidence is None
    ):
        raise ValueError(
            "The layer-curving archive fixture requires accepted source evidence: "
            + (result.failure or result.status.value)
        )
    return result, layers, control, policy


def make_restricted_layer_curving_archive_case() -> tuple[
    LayerCurvingConstructionResult,
    BoundaryLayerMesh,
    BoundaryLayerControl,
    BoundaryLayerPolicy,
]:
    """Return accepted layer evidence on an exact per-cell restricted map."""
    original, layers, control, policy = make_layer_curving_archive_case()
    evidence = original.source_evidence
    source_geometry = original.candidate_geometry
    if evidence is None or source_geometry is None:
        raise ValueError(
            "Restricted layer fixture requires accepted source evidence and geometry."
        )
    mesh = original.construction.mesh
    elements, routes, coordinates = source_geometry.resolve(mesh)
    restricted_elements = {}
    for block, element in zip(mesh.blocks, elements, strict=True):
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError(
                "Restricted layer fixture requires a scalar coordinate element."
            )
        dimension = reference_cell_topology(block.cell_kind).dimension
        identity_chart = axes(dimension)
        restricted_elements[block.name] = PerCellRestrictedCellGeometryElement(
            element,
            block.cell_kind,
            tuple(int(value) for value in block.global_ids),
            tuple(int(value) for value in block.global_ids),
            tuple(identity_chart for _ in range(block.cell_count)),
        )
    target_geometry = CellGeometrySpec(
        restricted_elements,
        {
            block.name: np.asarray(route)
            for block, route in zip(mesh.blocks, routes, strict=True)
        },
        np.asarray(coordinates, dtype=np.float64),
    )
    old_witness = evidence.chart_witness
    target_witness = SourceBoundaryChartWitness(
        mesh_id=old_witness.mesh_id,
        geometry_id=cell_geometry_id(target_geometry),
        topology_id=old_witness.topology_id,
        source_id=old_witness.source_id,
        source_revision=old_witness.source_revision,
        source_query_id=old_witness.source_query_id,
        chart_dimension=old_witness.chart_dimension,
        reference_dimension=old_witness.reference_dimension,
        cell_global_ids=np.asarray(old_witness.cell_global_ids),
        entity_global_ids=np.asarray(old_witness.entity_global_ids),
        local_entity_indices=np.asarray(old_witness.local_entity_indices),
        target_reference_vertices=np.asarray(old_witness.target_reference_vertices),
        target_reference_rational_tokens=old_witness.target_reference_rational_tokens,
        source_chart_ids=old_witness.source_chart_ids,
        source_entity_ids=old_witness.source_entity_ids,
        occurrence_paths=old_witness.occurrence_paths,
        source_parameters=np.asarray(old_witness.source_parameters),
        source_parameter_rational_tokens=old_witness.source_parameter_rational_tokens,
        orientations=np.asarray(old_witness.orientations),
        implicit_cell_keys=np.asarray(old_witness.implicit_cell_keys),
        implicit_root_edge_keys=np.asarray(old_witness.implicit_root_edge_keys),
        target_face_vertex_global_ids=np.asarray(
            old_witness.target_face_vertex_global_ids
        ),
    )
    from phydrax.meshing._adaptation import (
        _identity_lineage,
        _identity_reference_maps_for_transition,
    )

    renewed = renew_layer_curving_source_evidence(
        evidence,
        mesh,
        source_geometry,
        mesh,
        target_geometry,
        target_witness,
        _identity_lineage(mesh, mesh),
        _identity_reference_maps_for_transition(mesh, mesh),
    )
    renewed.require_current(mesh, target_geometry)
    validity = certify_cell_geometry_validity(
        target_geometry, mesh=mesh, policy=CellValidityPolicy()
    )
    if not validity.all_certified:
        raise ValueError("Restricted layer fixture did not certify its cell maps.")
    embedding = certify_global_embedding(mesh, target_geometry, validity)
    if embedding.status != "certified":
        raise ValueError("Restricted layer fixture did not certify its embedding.")
    construction = _construction_with_geometry(original.construction, target_geometry)
    result = LayerCurvingConstructionResult(
        LayerCurvingStatus.CURVED,
        construction,
        target_geometry,
        validity,
        embedding,
        renewed,
        None,
        projection_trace=renewed.projection_trace,
        implicit_root_trace=renewed.implicit_root_trace,
        implicit_root_chart_witness=renewed.implicit_root_chart_witness,
        implicit_map_witness_id=renewed.implicit_map_witness_id,
        implicit_chart_evidence=renewed.implicit_chart_evidence,
        chart_witness=target_witness,
        source_fidelity=renewed.fidelity,
    )
    if (
        result.source_evidence is None
        or result.source_evidence.source_surface_charts is None
        or result.source_evidence.projection_trace is not None
    ):
        raise ValueError(
            "Restricted layer fixture lost its direct source-chart authority."
        )
    return result, layers, control, policy


__all__ = [
    "_layer_mesh",
    "_mixed_construction",
    "_mapped_cube_source_for_layer",
    "_mapped_cube_surface_charts",
    "_mapped_cube_wall",
    "_mapped_layer_core_construction_for_fixture",
    "make_layer_curving_archive_case",
    "make_mapped_layer_curving_archive_case",
    "make_restricted_layer_curving_archive_case",
]
