from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    lagrange_element,
    reference_cell_topology,
)
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    swept_coordinate_element,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.geometry import (
    BSplineSurfacePatch,
    MappedReferenceDomain,
    PiecewiseLinearDomain,
)
from phydrax.geometry._curve_boundary import ParametricCurveBoundarySource
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._piecewise_linear_boundary import PiecewiseLinearBoundarySource
from phydrax.geometry.brep import CircleCurve, LineCurve, PlanePatch
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshQualityTarget,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import (
    BlockInterfaceControl,
    FeatureKind,
    HoleSeed,
    ProtectedFeature,
    RegionSeed,
    TransfiniteCurveControl,
    TransfiniteSurfaceControl,
)
from phydrax.meshing._hex_schedule import NativeHexGridSchedule
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import (
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing._structured import TransfiniteBlock
from phydrax.meshing._sweep import SweepControl, SweepMapKind
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_sources import (
    NativeMappedHexSource,
    NativePlanarSource,
    NativePlcSource,
    NativeStructuredSource,
    NativeSweepSource,
)

from .._runtime import accepted_result_record, runtime_identity


_CONTRACT = SpatialCoordinateContract.si()
_PROFILE_PREFIX = "native-family-lifecycle:"
_BASE_RESOLUTION = 4
_PROFILE_TOLERANCE = 1.0e-10
_QUALITY_LIMITS = {
    "minimum_scaled_jacobian": 1.0e-8,
    "minimum_mean_ratio": 0.01,
    "maximum_aspect_ratio": 100.0,
}


@dataclass(frozen=True)
class _FamilyInput:
    source: Any
    specification: SurfaceMeshingSpec | VolumeMeshingSpec
    options: NativeMeshingOptions
    expected_route: str
    physical_input: dict[str, object]
    expected_measure: float
    expected_regions: dict[str, float] = field(default_factory=dict)
    expected_domain: Any | None = None
    fidelity_tolerance: float | None = None
    require_region_coverage: bool = False
    require_transition: bool = False
    quality_limits: dict[str, float] = field(default_factory=dict)


def _profile_ids() -> tuple[str, ...]:
    migration_path = Path(__file__).resolve().parents[1] / "native_case_migration.json"
    with migration_path.open("r", encoding="utf-8") as stream:
        migration = json.load(stream)
    workflows = migration["workflow_contracts"]
    workflow = next(
        item for item in workflows if item["workflow"] == "general-quads-hexes"
    )
    return tuple(workflow["profiles"])


def _size_control(scope: MeshingScope, resolution: int) -> UniformSizeControl:
    return UniformSizeControl(
        scope,
        1.0 / resolution,
        maximum_size=2.0 / resolution,
        strength=SizeControlStrength.HARD,
    )


def _size_compliance() -> SizeCompliancePolicy:
    return SizeCompliancePolicy(relative_tolerance=0.95, target_statistics=("p50",))


def _surface_specification(
    source: Any,
    resolution: int,
    family: str,
    *,
    ambient_dimension: int = 2,
    geometry_order: int = 1,
    planar_embedding: Any | None = None,
    protected_features: tuple[ProtectedFeature, ...] = (),
) -> SurfaceMeshingSpec:
    scope = MeshingScope.whole(source, 2)
    return SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            ambient_dimension,
            CellFamilyPolicy(required=(family,)),
            geometry_order=geometry_order,
        ),
        scope,
        planar_embedding=planar_embedding,
        size_controls=(_size_control(scope, resolution),),
        protected_features=protected_features,
        size_compliance=_size_compliance(),
        quality_target=MeshQualityTarget(minimum_angle=0.01, hard=True),
    )


def _volume_specification(
    source: Any,
    resolution: int,
    family: str,
    *,
    geometry_order: int = 1,
    fill_strategy: VolumeFillStrategy = VolumeFillStrategy.MULTIZONE,
    region_seeds: tuple[RegionSeed, ...] = (),
    hole_seeds: tuple[HoleSeed, ...] = (),
    mixed: bool = False,
) -> VolumeMeshingSpec:
    scope = MeshingScope.whole(source, 2)
    policy = (
        CellFamilyPolicy(
            required=("hexahedron",),
            allowed_transitions=("pyramid", "tetrahedron"),
            allow_mixed=True,
        )
        if mixed
        else CellFamilyPolicy(required=(family,))
    )
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, policy, geometry_order=geometry_order),
        scope,
        fill_strategy,
        size_controls=(_size_control(scope, resolution),),
        size_compliance=_size_compliance(),
        region_seeds=region_seeds,
        hole_seeds=hole_seeds,
    )


def _curve_control(
    curve: Any,
    name: str,
    interval: tuple[float, float],
    intervals: int,
) -> TransfiniteCurveControl:
    return TransfiniteCurveControl(curve, name, interval, intervals)


def _rectangle(name: str, left: float, right: float, resolution: int) -> TransfiniteBlock:
    return TransfiniteBlock(
        name,
        tuple(
            _curve_control(
                LineCurve(origin, direction), f"{name}:{edge}", (0.0, 1.0), intervals
            )
            for edge, origin, direction, intervals in (
                ("left", (left, 0.0), (0.0, 1.0), resolution),
                ("right", (right, 0.0), (0.0, 1.0), resolution),
                ("bottom", (left, 0.0), (right - left, 0.0), resolution),
                ("top", (left, 1.0), (right - left, 0.0), resolution),
            )
        ),
    )




def _pwl_owner(domain: PiecewiseLinearDomain) -> Any:
    class Owner:
        def __init__(self) -> None:
            self.domain = domain
            self.source_id = domain.source_id
            self.source_revision = domain.source_revision
            self.envelope_id = domain.domain_id
            self._domain_id = domain.domain_id
            edge_rows = {
                tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
                for face in np.asarray(domain.facets, dtype=np.int64)
                for index in range(3)
            }
            self.edges = np.asarray(sorted(edge_rows), dtype=np.int64).reshape((-1, 2))
            entity_rows = (
                np.arange(domain.vertices.shape[0], dtype=np.int64),
                np.arange(self.edges.shape[0], dtype=np.int64),
                np.arange(domain.facets.shape[0], dtype=np.int64),
                np.arange(len(domain.region_ids), dtype=np.int64),
            )
            self.scope_entity_ids_by_mesh_row = entity_rows
            self._scope_entity_set_ids = tuple(
                f"{domain.domain_id}:pwl:{dimension}" for dimension in range(4)
            )
            self._entity_keys = (
                tuple(
                    (f"{domain.domain_id}:vertex:{row}", ())
                    for row in range(entity_rows[0].size)
                ),
                tuple(
                    (f"{domain.domain_id}:edge:{int(edge[0])}:{int(edge[1])}", ())
                    for edge in self.edges
                ),
                tuple(
                    (f"{domain.domain_id}:facet:{row}", ())
                    for row in range(entity_rows[2].size)
                ),
                tuple(
                    (name, path)
                    for name, path in zip(
                        domain.region_ids, domain.region_occurrence_paths, strict=True
                    )
                ),
            )

        def require_current(self) -> None:
            if (domain.domain_id, domain.source_id, domain.source_revision) != (
                self._domain_id,
                self.source_id,
                self.source_revision,
            ):
                raise ValueError(
                    "The retained PWL source domain changed after preparation."
                )

        def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
            return self.scope_entity_ids_by_mesh_row[dimension]

        def scope_entity_set_id(self, dimension: int, /) -> str:
            return self._scope_entity_set_ids[dimension]

        def source_entity_key_for_scope(
            self, dimension: int, entity_id: int, /
        ) -> tuple[str, tuple[str, ...]]:
            identifiers = self.scope_entity_ids_by_mesh_row[dimension]
            position = int(np.searchsorted(identifiers, entity_id))
            if position >= identifiers.size or int(identifiers[position]) != entity_id:
                raise ValueError(
                    "The requested PWL entity is outside its retained source bank."
                )
            return self._entity_keys[dimension][position]

    return Owner()


def _pwl_query(domain: PiecewiseLinearDomain) -> PiecewiseLinearBoundarySource:
    return PiecewiseLinearBoundarySource(_pwl_owner(domain), 2)


def _cube_domain(source_id: str = "box") -> PiecewiseLinearDomain:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    facets = np.asarray(
        tuple(triangle for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))),
        dtype=np.int64,
    )
    return PiecewiseLinearDomain(
        vertices,
        facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (facets.shape[0], 1)),
        ("solid",),
        source_id=source_id,
    )


def _cube_profile(resolution: int, family: str) -> CellMesh:
    axis = np.linspace(0.0, 1.0, resolution + 1)
    points = np.asarray(tuple((u, v, 0.0) for u in axis for v in axis), dtype=np.float64)
    cells: list[tuple[int, ...]] = []
    for i in range(resolution):
        for j in range(resolution):
            a = i * (resolution + 1) + j
            b = (i + 1) * (resolution + 1) + j
            if family == "quadrilateral":
                cells.append((a, b, b + 1, a + 1))
            else:
                cells.extend(((a, b, b + 1), (a, b + 1, a + 1)))
    kind = "quadrilateral" if family == "quadrilateral" else "triangle"
    block = CellBlock(
        "profile",
        kind,
        np.asarray(cells, dtype=np.int32),
        global_ids=np.arange(len(cells), dtype=np.int64) + 500,
    )
    return CellMesh(
        points,
        (block,),
        vertex_global_ids=np.arange(points.shape[0], dtype=np.int64) + 40,
    )


def _cube_domain_vertex_map(
    domain: PiecewiseLinearDomain, profile: CellMesh, resolution: int
) -> np.ndarray:
    profile_points = np.asarray(profile.coordinates, dtype=np.float64).reshape((-1, 3))
    profile_ids = np.asarray(profile.vertex_global_ids, dtype=np.int64).reshape((-1,))
    row_by_xy: dict[tuple[float, float], int] = {}
    for row in range(profile_points.shape[0]):
        row_by_xy[(float(profile_points[row, 0]), float(profile_points[row, 1]))] = int(
            profile_ids[row]
        )
    domain_points = np.asarray(domain.vertices, dtype=np.float64).reshape((-1, 3))
    mapping: list[tuple[int, int]] = []
    for row in range(domain_points.shape[0]):
        x = float(domain_points[row, 0])
        y = float(domain_points[row, 1])
        z = float(domain_points[row, 2])
        mapping.append((row_by_xy[(x, y)], 0 if z == 0.0 else resolution))
    return np.asarray(mapping, dtype=np.int64)


def _plc(
    vertices: np.ndarray,
    polygons: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    polygon_facets: np.ndarray,
    facet_regions: np.ndarray,
    region_ids: tuple[str, ...],
    *,
    vertex_prefix: str,
    facet_ids: tuple[str, ...],
) -> PiecewiseLinearComplex:
    return PiecewiseLinearComplex(
        vertices,
        polygons,
        polygon_facets,
        facet_regions,
        region_ids,
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(
            f"{vertex_prefix}-vertex-{row}" for row in range(vertices.shape[0])
        ),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=facet_ids,
        facet_occurrence_paths=((),) * len(facet_ids),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),) * len(region_ids),
    )


def _cavity_plc() -> PiecewiseLinearComplex:
    box_loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )

    def box(
        lower: tuple[float, float, float], upper: tuple[float, float, float]
    ) -> np.ndarray:
        return np.asarray(
            [
                [(upper if index >> axis & 1 else lower)[axis] for axis in range(3)]
                for index in range(8)
            ],
            dtype=np.float64,
        )

    left = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    right = box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    cavity = box((1.25, 0.25, 0.25), (1.75, 0.75, 0.75))
    vertices = np.concatenate((left, right[[1, 3, 5, 7]], cavity))
    right_index = {0: 1, 2: 3, 4: 5, 6: 7, 1: 8, 3: 9, 5: 10, 7: 11}
    polygons: list[tuple[int, ...]] = []
    polygon_facets: list[int] = []
    for loop in box_loops:
        if loop != (1, 3, 7, 5):
            polygons.append(loop)
            polygon_facets.append(0)
        if loop != (0, 4, 6, 2):
            polygons.append(tuple(right_index[index] for index in loop))
            polygon_facets.append(1)
    polygons.append((1, 3, 7, 5))
    polygon_facets.append(2)
    polygons.extend(tuple(index + 12 for index in loop) for loop in box_loops)
    polygon_facets.extend([3] * len(box_loops))
    return _plc(
        vertices,
        polygons,
        np.asarray(polygon_facets, dtype=np.int64),
        np.asarray(((-1, 0), (-1, 1), (1, 0), (1, -1)), dtype=np.int64),
        ("left", "right"),
        vertex_prefix="material-cavity",
        facet_ids=("left-outer", "right-outer", "material-interface", "right-cavity"),
    )


def _nonsweepable_plc() -> tuple[PiecewiseLinearComplex, float]:
    tetrahedron = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64
    )
    vertices = np.concatenate(
        (
            tetrahedron,
            (2 * tetrahedron + (2, 2, 2)) / 16,
            (2 * tetrahedron + (7, 2, 2)) / 16,
        )
    )
    outward = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    polygons: list[tuple[int, ...]] = list(outward)
    for offset in (4, 8):
        polygons.extend(
            (offset + face[2], offset + face[1], offset + face[0]) for face in outward
        )
    complex_ = _plc(
        vertices,
        polygons,
        np.arange(12, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (12, 1)),
        ("solid",),
        vertex_prefix="nonsweepable-dual-hex",
        facet_ids=tuple(f"nonsweepable-shell-facet-{row}" for row in range(12)),
    )
    outer = Fraction(1, 6)
    cavity = Fraction(1, 6) * Fraction(1, 8) ** 3
    return complex_, float(outer - 2 * cavity)


def _mapped_reference_domain(
    complex_: PiecewiseLinearComplex,
    points: np.ndarray,
    cells: np.ndarray,
    cell_regions: np.ndarray,
    root_vertex_ids: np.ndarray,
    root_cell_ids: np.ndarray,
    root_geometry: CellGeometrySpec,
    source_id: str,
    source_revision: str,
) -> MappedReferenceDomain:
    polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    owners = np.asarray(complex_.polygon_facets, dtype=np.int64)
    facet_sides = np.asarray(complex_.facet_regions, dtype=np.int64)
    triangles: list[tuple[int, int, int]] = []
    region_sides: list[tuple[int, int]] = []
    for polygon in range(owners.size):
        loop = polygons[offsets[polygon] : offsets[polygon + 1]]
        facet_row = int(owners[polygon])
        side: tuple[int, int] = (
            int(facet_sides[facet_row, 1]),
            int(facet_sides[facet_row, 0]),
        )
        for index in range(1, loop.size - 1):
            triangles.append((int(loop[0]), int(loop[index]), int(loop[index + 1])))
            region_sides.append(side)
    reference_domain = PiecewiseLinearDomain(
        points,
        np.asarray(triangles, dtype=np.int64),
        np.asarray(region_sides, dtype=np.int64),
        complex_.region_ids,
        source_id=source_id,
    )
    root_mesh = CellMesh(
        points,
        (CellBlock("root", "hexahedron", cells, global_ids=root_cell_ids),),
        vertex_global_ids=root_vertex_ids,
    )
    return MappedReferenceDomain(
        reference_domain,
        root_mesh,
        root_geometry,
        cell_regions,
        source_id=source_id,
        source_revision=source_revision,
        coordinate_contract=_CONTRACT,
        reference_domain_vertex_root_ids=root_vertex_ids,
    )


def _curved_occupied_source(
    kind: str,
) -> tuple[NativeMappedHexSource, dict[str, float], dict[str, object]]:
    if kind == "curved-nonsweepable-hex":
        occupied = {
            (1, 1, 1),
            (0, 1, 1),
            (2, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2),
        }
        materials = False
    else:
        occupied = {
            origin for origin in product(range(3), repeat=3) if origin != (1, 1, 1)
        }
        materials = kind == "curved-material-hex"
    topology = reference_cell_topology("hexahedron")
    offsets = np.asarray(topology.vertices, dtype=np.int64)
    origins = sorted(occupied)
    keys = sorted(
        {
            tuple((np.asarray(origin) + offset).tolist())
            for origin in origins
            for offset in offsets
        }
    )
    index = {key: row for row, key in enumerate(keys)}
    points = np.asarray(keys, dtype=np.float64)
    cells = np.asarray(
        [
            [index[tuple((np.asarray(origin) + offset).tolist())] for offset in offsets]
            for origin in origins
        ],
        dtype=np.int32,
    )
    region_ids = (
        ("left", "right")
        if materials
        else (("material",) if kind != "curved-nonsweepable-hex" else ("solid",))
    )
    cell_regions = np.asarray(
        [int(materials and origin[0] >= 2) for origin in origins], dtype=np.int64
    )
    owners: dict[tuple[int, ...], list[tuple[int, tuple[int, ...]]]] = {}
    for row, cell in enumerate(cells):
        for face in topology.entities[2]:
            face_vertices = tuple(int(value) for value in cell[list(face)])
            owners.setdefault(tuple(sorted(face_vertices)), []).append(
                (row, face_vertices)
            )
    polygons: list[tuple[int, ...]] = []
    polygon_facets: list[int] = []
    facet_sides: list[tuple[int, int]] = []
    for values in owners.values():
        row, oriented = values[0]
        region = int(cell_regions[row])
        if len(values) == 1:
            sides = (-1, region)
        else:
            other_row = int(values[1][0])
            other_region = int(cell_regions[other_row])
            if other_region == region:
                continue
            sides = (other_region, region)
        facet_index = len(facet_sides)
        polygons.extend(
            (
                (oriented[0], oriented[1], oriented[2]),
                (oriented[0], oriented[2], oriented[3]),
            )
        )
        polygon_facets.extend((facet_index, facet_index))
        facet_sides.append(sides)
    complex_ = _plc(
        points,
        polygons,
        np.asarray(polygon_facets, dtype=np.int64),
        np.asarray(facet_sides, dtype=np.int64),
        region_ids,
        vertex_prefix=f"{kind}-root",
        facet_ids=tuple(f"{kind}-facet-{row}" for row in range(len(facet_sides))),
    )
    source_id, source_revision = "curved-polynomial-domain", "declared-curved-state"
    root_vertex_ids = np.arange(points.shape[0], dtype=np.int64) + 1000
    root_cell_ids = np.arange(cells.shape[0], dtype=np.int64) + 900
    element = lagrange_element("hexahedron", 2)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    positions = (
        np.asarray(origins, dtype=np.float64)[:, None, :] + reference_nodes[None, :, :]
    ).reshape((-1, 3))
    physical = positions.copy()
    physical[:, 2] += (1.0 / 64.0) * physical[:, 0] ** 2 * physical[:, 1]
    routes = np.arange(physical.shape[0], dtype=np.int32).reshape(
        (cells.shape[0], element.local_dof_count)
    )
    root_geometry = CellGeometrySpec({"root": element}, {"root": routes}, physical)
    domain = _mapped_reference_domain(
        complex_,
        points,
        cells,
        cell_regions,
        root_vertex_ids,
        root_cell_ids,
        root_geometry,
        source_id,
        source_revision,
    )
    reference = NativePlcSource(complex_, source_id, source_revision)
    source = NativeMappedHexSource(
        reference,
        domain,
        plc_vertex_root_global_ids=root_vertex_ids,
    )
    region_measures = {
        name: float(sum(int(cell_regions[row] == region) for row in range(len(origins))))
        for region, name in enumerate(region_ids)
    }
    details = {
        "root_cell_origins": [list(origin) for origin in origins],
        "root_cell_count": len(origins),
        "root_cell_family": "hexahedron",
        "root_geometry_order": 2,
        "physical_map": "(x, y, z) -> (x, y, z + x^2*y/64)",
        "jacobian_determinant": 1.0,
        "material_partition": "x-index >= 2 is right; all other occupied roots are left"
        if materials
        else None,
        "curvature": 1.0 / 64.0,
    }
    return source, region_measures, details


def _quarter_extrusion() -> MeshingDomain:
    controls = np.asarray(
        [[(x, 1.0, 0.0), (x, 1.0, 1.0), (x, 0.0, 1.0)] for x in (0.0, 1.0)],
        dtype=np.float64,
    )
    weights = np.tile(np.asarray((1.0, np.sqrt(0.5), 1.0), dtype=np.float64), (2, 1))
    surface = BSplineSurfacePatch(
        controls,
        weights,
        np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64),
        1,
        2,
    )
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64)
    loop = tuple(
        PatchCurveUse(
            index, LineCurve(uv[index], uv[(index + 1) % 4] - uv[index]), 0.0, 1.0
        )
        for index in range(4)
    )
    return MeshingDomain(
        (MeshingSurfacePatch(surface, (loop,)),),
        tuple(MeshingDomainCurve(index, (index + 1) % 4) for index in range(4)),
        4,
        source_id="original-quarter-extrusion",
        source_revision="original-r1",
    )


def _planar_dual_quad(resolution: int) -> _FamilyInput:
    vertices = np.asarray(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.8, 0.8),
            (0.8, 1.2),
            (1.2, 1.2),
            (1.2, 0.8),
        ),
        dtype=np.float64,
    )
    region = PlanarMeshRegion(
        vertices, ((0, 1, 2, 3), (4, 5, 6, 7)), feature_id="quad-plate"
    )
    source = NativePlanarSource(region, "plate-revision")
    specification = _surface_specification(source, resolution, "quadrilateral")
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("planar_dual_quad"),
        "planar_dual_quad",
        {
            "fixture": "NativePlanarSource:holed-plate",
            "outer_loop_m": vertices[:4].tolist(),
            "hole_loop_m": vertices[4:].tolist(),
            "domain_area_m2": 3.84,
            "resolution": resolution,
            "size_target_m": 1.0 / resolution,
            "maximum_size_m": 2.0 / resolution,
        },
        3.84,
        fidelity_tolerance=0.0,
        quality_limits={"minimum_angle": 0.01},
    )


def _structured_square(resolution: int, multiblock: bool) -> _FamilyInput:
    source_id = "unit-square"
    if multiblock:
        blocks = (
            _rectangle("left", 0.0, 0.5, resolution),
            _rectangle("right", 0.5, 1.0, resolution),
        )
        interfaces = (BlockInterfaceControl("left", 1, "right", 0, (0,), (False,)),)
        fixture = "NativeStructuredSource:two-glued-transfinite-blocks"
    else:
        blocks = (_rectangle("square", 0.0, 1.0, resolution),)
        interfaces = ()
        fixture = "NativeStructuredSource:single-transfinite-block"
    domain = PiecewiseLinearDomain(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("square",),
        source_id=source_id,
    )
    revision = domain.source_revision
    interface_faces = {
        face
        for interface in interfaces
        for face in (
            (interface.first_block, interface.first_face),
            (interface.second_block, interface.second_face),
        )
    }
    boundaries = tuple(
        cast(TransfiniteCurveControl, control)
        for block in blocks
        for face, control in enumerate(block.boundaries)
        if (block.name, face) not in interface_faces
    )
    boundary = ParametricCurveBoundarySource(
        tuple(control.curve for control in boundaries),
        tuple(control.parameter_range for control in boundaries),
        source_id=source_id,
        source_revision=revision,
        curve_entity_ids=tuple(control.source_id for control in boundaries),
        occurrence_paths=((),) * len(boundaries),
        chart_ids=tuple(control.source_id for control in boundaries),
    )
    source = NativeStructuredSource(
        blocks,
        interfaces,
        source_id,
        revision,
        fidelity_source=boundary,
        maximum_deviation=0.08,
        domain=domain,
    )
    embedding = phx.geometry.PlanarEmbedding(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    )
    specification = _surface_specification(
        source,
        resolution,
        "quadrilateral",
        planar_embedding=embedding,
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("structured_transfinite"),
        "structured_transfinite",
        {
            "fixture": fixture,
            "domain_m": ((0.0, 0.0), (1.0, 1.0)),
            "blocks": [block.name for block in blocks],
            "block_interfaces": [interface.control_id for interface in interfaces],
            "intervals_per_block_axis": resolution,
            "physical_domain_area_m2": 1.0,
            "maximum_deviation_m": 0.08,
            "resolution": resolution,
        },
        1.0,
        fidelity_tolerance=0.08,
        quality_limits={"minimum_angle": 0.01},
    )


def _annular_blocks(resolution: int) -> _FamilyInput:
    inner = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    outer = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 2.0)
    angular = 4 * resolution
    angles = np.concatenate(
        tuple(
            np.linspace(index * np.pi / 2.0, (index + 1) * np.pi / 2.0, angular + 1)[:-1]
            for index in range(4)
        )
    )
    points = np.concatenate(
        (
            np.asarray(inner.evaluate(jnp.asarray(angles)), dtype=np.float64),
            np.asarray(outer.evaluate(jnp.asarray(angles)), dtype=np.float64),
        )
    )
    count = angles.size
    facets = tuple(((index + 1) % count, index) for index in range(count)) + tuple(
        (count + index, count + (index + 1) % count) for index in range(count)
    )
    domain = PiecewiseLinearDomain(
        points,
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(facets), 1)),
        ("annulus",),
        source_id="annular-block-source",
    )
    revision = domain.source_revision
    query = ParametricCurveBoundarySource(
        (inner, outer),
        ((0.0, 2.0 * np.pi),) * 2,
        source_id=domain.source_id,
        source_revision=revision,
        curve_entity_ids=("inner", "outer"),
        occurrence_paths=((), ()),
        chart_ids=("annulus-inner-chart", "annulus-outer-chart"),
    )
    blocks = []
    for index in range(4):
        first, last = index * np.pi / 2.0, (index + 1) * np.pi / 2.0
        first_ray = np.asarray((np.cos(first), np.sin(first)))
        last_ray = np.asarray((np.cos(last), np.sin(last)))
        blocks.append(
            TransfiniteBlock(
                f"sector:{index}",
                (
                    _curve_control(inner, "inner", (first, last), angular),
                    _curve_control(outer, "outer", (first, last), angular),
                    _curve_control(
                        LineCurve(first_ray, first_ray),
                        f"ray:{index}",
                        (0.0, 1.0),
                        resolution,
                    ),
                    _curve_control(
                        LineCurve(last_ray, last_ray),
                        f"ray:{(index + 1) % 4}",
                        (0.0, 1.0),
                        resolution,
                    ),
                ),
            )
        )
    interfaces = tuple(
        BlockInterfaceControl(
            f"sector:{index}", 3, f"sector:{(index + 1) % 4}", 2, (0,), (False,)
        )
        for index in range(4)
    )
    source = NativeStructuredSource(
        tuple(blocks),
        interfaces,
        domain.source_id,
        revision,
        fidelity_source=query,
        maximum_deviation=0.08,
        domain=domain,
    )
    exact_area = (
        sum(
            Fraction(float(points[a, 0])) * Fraction(float(points[b, 1]))
            - Fraction(float(points[a, 1])) * Fraction(float(points[b, 0]))
            for a, b in facets
        )
        / 2
    )
    specification = _surface_specification(
        source,
        resolution,
        "quadrilateral",
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("structured_transfinite"),
        "structured_transfinite",
        {
            "fixture": "NativeStructuredSource:source-circle-annular-block-cycle",
            "inner_radius_m": 1.0,
            "outer_radius_m": 2.0,
            "sector_count": 4,
            "angular_intervals_per_quadrant": angular,
            "radial_intervals": resolution,
            "polygon_domain_area_m2": float(exact_area),
            "continuous_circle_domain_area_m2": 3.0 * np.pi,
            "maximum_deviation_m": 0.08,
            "resolution": resolution,
        },
        float(exact_area),
        expected_regions={"annulus": float(exact_area)},
        expected_domain=domain,
        fidelity_tolerance=0.08,
        quality_limits={"minimum_angle": 0.01},
    )


def _reversed_multiblock(resolution: int, materials: bool) -> _FamilyInput:
    width = 2.0 if materials else 1.0
    split = width / 2.0
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (split, 0.0, 0.0),
            (split, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (split, 0.0, 1.0),
            (split, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (width, 0.0, 0.0),
            (width, 1.0, 0.0),
            (width, 0.0, 1.0),
            (width, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    source_cells = ((0, 1, 2, 3, 4, 5, 6, 7), (1, 8, 9, 2, 5, 10, 11, 6))
    face_template = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    face_rows: dict[tuple[int, ...], tuple[tuple[int, ...], list[int]]] = {}
    for material, cell in enumerate(source_cells):
        for face in face_template:
            row = tuple(cell[index] for index in face)
            key = tuple(sorted(row))
            if key in face_rows:
                face_rows[key][1][1] = material
            else:
                face_rows[key] = (row, [material, -1])
    facets: list[tuple[int, int, int]] = []
    adjacency: list[tuple[int, int]] = []
    for row, labels in face_rows.values():
        if not materials and labels[1] >= 0:
            continue
        a, b, c, d = row
        facets.extend(((a, b, c), (a, c, d)))
        if materials:
            positive, negative = int(labels[0]), int(labels[1])
            adjacency.extend(((positive, negative), (positive, negative)))
        else:
            adjacency.extend(((0, -1), (0, -1)))
    region_ids = ("left", "right") if materials else ("solid",)
    domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(facets, dtype=np.int64),
        np.asarray(adjacency, dtype=np.int64),
        region_ids,
        source_id="reversed-block-source",
    )
    query = _pwl_query(domain)
    blocks = []
    for name, origin, diagonal in (
        ("left", (0.0, 0.0, 0.0), (split, 1.0, 1.0)),
        ("right", (split, 1.0, 1.0), (split, -1.0, -1.0)),
    ):
        matrix = np.diag(np.asarray(diagonal, dtype=np.float64))
        controls = []
        for axis in range(3):
            other = tuple(index for index in range(3) if index != axis)
            for side in range(2):
                patch = PlanePatch(
                    np.asarray(origin) + side * matrix[:, axis],
                    matrix[:, other[0]],
                    matrix[:, other[1]],
                )
                controls.append(
                    TransfiniteSurfaceControl(
                        patch,
                        f"{name}:{2 * axis + side}",
                        np.asarray(((0.0, 0.0), (1.0, 1.0))),
                        (resolution, resolution),
                    )
                )
        blocks.append(TransfiniteBlock(name, tuple(controls)))
    interface = BlockInterfaceControl("left", 1, "right", 0, (0, 1), (True, True))
    block_regions = (("left", "left"), ("right", "right")) if materials else ()
    source = NativeStructuredSource(
        tuple(blocks),
        (interface,),
        domain.source_id,
        domain.source_revision,
        fidelity_source=query,
        maximum_deviation=0.2,
        domain=domain,
        block_regions=block_regions,
    )
    region_seeds = (
        (
            RegionSeed(
                np.asarray((0.5, 0.5, 0.5)), "left", "material-left", RegionRole.SOLID
            ),
            RegionSeed(
                np.asarray((1.5, 0.5, 0.5)), "right", "material-right", RegionRole.SOLID
            ),
        )
        if materials
        else ()
    )
    specification = _volume_specification(
        source,
        resolution,
        "hexahedron",
        region_seeds=region_seeds,
    )
    expected_regions = {"left": 1.0, "right": 1.0} if materials else {"solid": 1.0}
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("structured_transfinite"),
        "structured_transfinite",
        {
            "fixture": "NativeStructuredSource:reversed-two-material-boxes"
            if materials
            else "NativeStructuredSource:reversed-logical-face-boxes",
            "box_union_m": ((0.0, 0.0, 0.0), (width, 1.0, 1.0)),
            "block_widths_m": (split, split),
            "reversed_right_logical_axes": ("y", "z"),
            "interface_permutation": (0, 1),
            "interface_flips": (True, True),
            "material_regions": expected_regions if materials else {"solid": 1.0},
            "resolution": resolution,
            "maximum_deviation_m": 0.2,
        },
        sum(expected_regions.values()),
        expected_regions=expected_regions,
        expected_domain=domain,
        fidelity_tolerance=0.2,
        require_region_coverage=True,
        quality_limits=dict(_QUALITY_LIMITS),
    )


def _sweep_source(resolution: int, family: str) -> _FamilyInput:
    domain = _cube_domain()
    profile = _cube_profile(resolution, family)
    control = SweepControl(
        SweepMapKind.EXTRUSION,
        phx.meshing.LayerSchedule(
            np.full(resolution, 1.0 / resolution, dtype=np.float64)
        ),
        "box-extrusion",
    )
    source = NativeSweepSource(
        profile,
        control,
        domain.source_id,
        domain.source_revision,
        fidelity_source=_pwl_query(domain),
        maximum_deviation=0.2,
        domain=domain,
        domain_vertex_map=_cube_domain_vertex_map(domain, profile, resolution),
    )
    specification = _volume_specification(
        source,
        resolution,
        "prism" if family == "triangle" else "hexahedron",
        fill_strategy=VolumeFillStrategy.SWEEP,
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("sweep"),
        "sweep",
        {
            "fixture": "NativeSweepSource:declared-triangle-extrusion"
            if family == "triangle"
            else "NativeSweepSource:declared-square-extrusion",
            "base_profile": "unit-square triangulated into two triangles per grid square"
            if family == "triangle"
            else "unit-square quadrilateral grid",
            "base_profile_resolution": resolution,
            "extrusion_axis": (0.0, 0.0, 1.0),
            "layer_thicknesses_m": [1.0 / resolution] * resolution,
            "domain_vertices_m": np.asarray(domain.vertices).tolist(),
            "domain_volume_m3": 1.0,
            "domain_vertex_map": np.asarray(
                source.domain_vertex_map, dtype=np.int64
            ).tolist(),
            "maximum_deviation_m": 0.2,
        },
        1.0,
        expected_regions={"solid": 1.0},
        expected_domain=domain,
        fidelity_tolerance=0.2,
        require_region_coverage=True,
        quality_limits=dict(_QUALITY_LIMITS),
    )


def _frame_sweep_source(resolution: int) -> _FamilyInput:
    axis = np.asarray(
        tuple(index / resolution for index in range(resolution + 1)), dtype=np.float64
    )
    uv = np.asarray(tuple((u, v) for u in axis for v in axis), dtype=np.float64)
    profile_cells = np.asarray(
        tuple(
            (
                i * (resolution + 1) + j,
                (i + 1) * (resolution + 1) + j,
                (i + 1) * (resolution + 1) + j + 1,
                i * (resolution + 1) + j + 1,
            )
            for i in range(resolution)
            for j in range(resolution)
        ),
        dtype=np.int32,
    )
    xy = np.column_stack(
        (uv[:, 0] + 0.25 * uv[:, 1], uv[:, 1] + 0.125 * uv[:, 0] * (1.0 - uv[:, 0]))
    )
    profile_points = np.column_stack((xy, np.zeros(xy.shape[0], dtype=np.float64)))
    profile = CellMesh(
        profile_points,
        (
            CellBlock(
                "profile",
                "quadrilateral",
                profile_cells,
                global_ids=np.arange(resolution**2, dtype=np.int64) + 500,
            ),
        ),
        vertex_global_ids=np.arange(uv.shape[0], dtype=np.int64) + 40,
    )
    element = lagrange_element("quadrilateral", 2)
    local = np.asarray(element.reference_nodes, dtype=np.float64)
    controls = []
    for i in range(resolution):
        for j in range(resolution):
            u, v = (i + local[:, 0]) / resolution, (j + local[:, 1]) / resolution
            controls.append(
                np.column_stack(
                    (u + 0.25 * v, v + 0.125 * u * (1.0 - u), np.zeros_like(u))
                )
            )
    bank = np.concatenate(controls)
    routes = np.arange(bank.shape[0], dtype=np.int32).reshape(
        (resolution**2, element.local_dof_count)
    )
    profile_geometry = CellGeometrySpec({"profile": element}, {"profile": routes}, bank)
    angles = 0.15 * axis
    frames = np.asarray(
        tuple(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
            for angle in angles
        ),
        dtype=np.float64,
    )
    translations = axis[:, None] * np.asarray((0.1, -0.05, 1.0), dtype=np.float64)
    control = SweepControl(
        SweepMapKind.FRAMES,
        phx.meshing.LayerSchedule(
            np.full(resolution, 1.0 / resolution, dtype=np.float64)
        ),
        "authored-curved-profile-absolute-frames",
        frames=frames,
        translations=translations,
    )
    reference_points = np.asarray(
        tuple((u, v, w) for w in axis for u, v in uv), dtype=np.float64
    )
    root_cells = np.asarray(
        tuple(
            tuple(profile_cell + layer * uv.shape[0])
            + tuple(profile_cell + (layer + 1) * uv.shape[0])
            for layer in range(resolution)
            for profile_cell in profile_cells
        ),
        dtype=np.int32,
    )
    root_cell_ids = np.arange(root_cells.shape[0], dtype=np.int64) + 900
    root_vertex_ids = np.arange(reference_points.shape[0], dtype=np.int64) + 100
    reference_mesh = CellMesh(
        reference_points,
        (CellBlock("root", "hexahedron", root_cells, global_ids=root_cell_ids),),
        vertex_global_ids=root_vertex_ids,
    )
    cube_vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    reference_facets = np.asarray(
        tuple(triangle for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))),
        dtype=np.int64,
    )
    reference_domain = PiecewiseLinearDomain(
        cube_vertices,
        reference_facets,
        np.tile(np.asarray((0, -1), dtype=np.int64), (reference_facets.shape[0], 1)),
        ("solid",),
        source_id="curved-frame-reference",
    )
    transformed = (
        bank[None, :, :] @ np.swapaxes(frames, -1, -2) + translations[:, None, :]
    )
    root_controls = np.concatenate(
        tuple(
            np.concatenate((transformed[layer, route], transformed[layer + 1, route]))
            for layer in range(resolution)
            for route in routes
        )
    )
    root_element = swept_coordinate_element(element)
    root_routes = np.arange(root_controls.shape[0], dtype=np.int32).reshape(
        root_cells.shape[0], root_element.local_dof_count
    )
    root_geometry = CellGeometrySpec(
        {"root": root_element}, {"root": root_routes}, root_controls
    )
    revision = canonical_fingerprint(
        {
            "kind": "authored-curved-frame-source",
            "profile_basis": element.element_id,
            "profile_controls": bank,
            "frames": frames,
            "translations": translations,
        }
    )
    reference_points_array = np.asarray(reference_points, dtype=np.float64).reshape(
        (-1, 3)
    )
    reference_vertex_row: dict[tuple[float, float, float], int] = {}
    for row in range(reference_points_array.shape[0]):
        key = (
            float(reference_points_array[row, 0]),
            float(reference_points_array[row, 1]),
            float(reference_points_array[row, 2]),
        )
        reference_vertex_row[key] = row
    reference_domain_vertices = np.asarray(cube_vertices, dtype=np.float64).reshape(
        (-1, 3)
    )
    reference_domain_root_ids = np.empty(
        (reference_domain_vertices.shape[0],), dtype=np.int64
    )
    for row in range(reference_domain_vertices.shape[0]):
        key = (
            float(reference_domain_vertices[row, 0]),
            float(reference_domain_vertices[row, 1]),
            float(reference_domain_vertices[row, 2]),
        )
        reference_domain_root_ids[row] = root_vertex_ids[reference_vertex_row[key]]
    source_id = "curved-frame-source"
    domain = MappedReferenceDomain(
        reference_domain,
        reference_mesh,
        root_geometry,
        np.zeros(root_cells.shape[0], dtype=np.int64),
        source_id=source_id,
        source_revision=revision,
        coordinate_contract=_CONTRACT,
        reference_domain_vertex_root_ids=reference_domain_root_ids,
    )
    reference_polygons: tuple[tuple[int, int, int], ...] = tuple(
        triangle for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))
    )
    reference_complex = _plc(
        cube_vertices,
        reference_polygons,
        np.arange(len(reference_polygons), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(reference_polygons), 1)),
        ("solid",),
        vertex_prefix=source_id,
        facet_ids=tuple(
            f"{source_id}:facet:{row}" for row in range(len(reference_polygons))
        ),
    )
    reference_source = NativePlcSource(reference_complex, source_id, revision)
    mapped_authority = NativeMappedHexSource(
        reference_source,
        domain,
        plc_vertex_root_global_ids=reference_domain_root_ids,
    )
    query = mapped_authority.boundary_query()
    correspondence = CellGeometryRestrictionSource(
        cell_geometry_id(root_geometry),
        reference_mesh.topology_id,
        {"sweep:profile": root_cell_ids},
        {"sweep:profile": root_vertex_ids[root_cells]},
    )
    source = NativeSweepSource(
        profile,
        control,
        source_id,
        revision,
        fidelity_source=query,
        maximum_deviation=1.0e-10,
        domain=domain,
        profile_geometry=profile_geometry,
        root_correspondence=correspondence,
    )
    specification = _volume_specification(
        source,
        resolution,
        "hexahedron",
        geometry_order=2,
        fill_strategy=VolumeFillStrategy.SWEEP,
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("sweep"),
        "sweep",
        {
            "fixture": "NativeSweepSource:authored-curved-profile-absolute-frames",
            "profile_map": "(u,v) -> (u+v/4, v+u*(1-u)/8, 0)",
            "station_frames": np.asarray(frames).tolist(),
            "station_translations_m": np.asarray(translations).tolist(),
            "layer_thicknesses_m": [1.0 / resolution] * resolution,
            "reference_domain_m": "unit cube; determinant-one profile Jacobian and unit axial translation",
            "root_hex_count": int(root_cells.shape[0]),
            "geometry_order": 2,
            "maximum_deviation_m": 1.0e-10,
            "resolution": resolution,
        },
        1.0,
        expected_regions={"solid": 1.0},
        expected_domain=domain,
        fidelity_tolerance=1.0e-10,
        require_region_coverage=True,
        quality_limits=dict(_QUALITY_LIMITS),
    )


def _plc_source_case(name: str, resolution: int) -> _FamilyInput:
    complex_ = _cavity_plc()
    source = NativePlcSource(complex_, "material-cavity", "cavity-revision")
    boundary = MeshingScope.whole(source, 2)
    interface_id = int(complex_.facet_entity_ids[2])
    cavity_id = int(complex_.facet_entity_ids[3])
    cavity_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        boundary.entity_kind,
        2,
        boundary.entity_set_id,
        np.asarray((cavity_id,), dtype=np.int64),
    )
    seeds = (
        RegionSeed(
            np.asarray((0.5, 0.5, 0.5)), "left", "material-left", RegionRole.SOLID
        ),
        RegionSeed(
            np.asarray((1.1, 0.5, 0.5)), "right", "material-right", RegionRole.SOLID
        ),
    )
    holes = (HoleSeed(np.asarray((1.5, 0.5, 0.5)), cavity_scope),)
    if name == "hex-dominant":
        route = "plc_hex_dominant"
        fill = VolumeFillStrategy.HEX_DOMINANT_SIMPLEX_TRANSITION
        options = NativeMeshingOptions(
            route,
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
            ),
            hex_core_fraction=0.8,
        )
    else:
        route = "plc_dual_hex"
        fill = VolumeFillStrategy.SIMPLEX
        options = NativeMeshingOptions(
            route,
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
            ),
        )
    specification = _volume_specification(
        source,
        resolution,
        "hexahedron",
        fill_strategy=fill,
        region_seeds=seeds,
        hole_seeds=holes,
        mixed=name == "hex-dominant",
    )
    expected = {"left": 1.0, "right": 0.875}
    return _FamilyInput(
        source,
        specification,
        options,
        route,
        {
            "fixture": "NativePlcSource:two-unit-box-material-interface-with-right-box-cavity",
            "outer_box_m": ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            "left_region_m": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "right_region_m": ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            "cavity_m": ((1.25, 0.25, 0.25), (1.75, 0.75, 0.75)),
            "material_interface_facet_id": interface_id,
            "cavity_facet_id": cavity_id,
            "region_seed_points_m": ((0.5, 0.5, 0.5), (1.1, 0.5, 0.5)),
            "hole_seed_point_m": (1.5, 0.5, 0.5),
            "size_target_m": 1.0 / resolution,
            "maximum_size_m": 2.0 / resolution,
            "resolution": resolution,
            "hex_core_fraction": 0.8 if name == "hex-dominant" else None,
        },
        sum(expected.values()),
        expected_regions=expected,
        require_region_coverage=True,
        fidelity_tolerance=0.0,
        require_transition=name == "hex-dominant",
        quality_limits=dict(_QUALITY_LIMITS),
    )


def _nonsweepable_dual_hex(resolution: int) -> _FamilyInput:
    complex_, exact_volume = _nonsweepable_plc()
    source = NativePlcSource(
        complex_, "qualification-nonsweepable-dual-hex", "dyadic-two-cavities-r1"
    )
    boundary = MeshingScope.whole(source, 2)
    facet_ids = np.asarray(complex_.facet_entity_ids, dtype=np.int64)
    first_cavity_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        boundary.entity_kind,
        2,
        boundary.entity_set_id,
        np.sort(facet_ids[4:8]),
    )
    second_cavity_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        boundary.entity_kind,
        2,
        boundary.entity_set_id,
        np.sort(facet_ids[8:12]),
    )
    regions = (
        RegionSeed(
            np.asarray((0.1, 0.1, 0.1)), "solid", "material-solid", RegionRole.SOLID
        ),
    )
    cavity_vertices = np.asarray(complex_.vertices, dtype=np.float64)
    holes = (
        HoleSeed(cavity_vertices[4:8].mean(axis=0), first_cavity_scope),
        HoleSeed(cavity_vertices[8:12].mean(axis=0), second_cavity_scope),
    )
    route = "plc_dual_hex"
    specification = _volume_specification(
        source,
        resolution,
        "hexahedron",
        fill_strategy=VolumeFillStrategy.SIMPLEX,
        region_seeds=regions,
        hole_seeds=tuple(holes),
    )
    options = NativeMeshingOptions(
        route,
        volume_schedule=NativeVolumeSchedule(
            radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
        ),
    )
    return _FamilyInput(
        source,
        specification,
        options,
        route,
        {
            "fixture": "NativePlcSource:unit-tetrahedron-two-enclosed-tetrahedral-cavities",
            "outer_tetrahedron_vertices_m": np.asarray(complex_.vertices[:4]).tolist(),
            "cavity_tetrahedron_vertices_m": [
                np.asarray(complex_.vertices[4:8]).tolist(),
                np.asarray(complex_.vertices[8:12]).tolist(),
            ],
            "boundary_components": 3,
            "cavities_strictly_enclosed_and_disjoint": True,
            "exact_rational_domain_volume_m3": str(Fraction(85, 512)),
            "region_seed_point_m": (0.1, 0.1, 0.1),
            "resolution": resolution,
        },
        exact_volume,
        expected_regions={"solid": exact_volume},
        fidelity_tolerance=0.0,
        require_region_coverage=True,
        quality_limits=dict(_QUALITY_LIMITS),
    )


def _mapped_hex_case(name: str, resolution: int) -> _FamilyInput:
    source, region_measures, details = _curved_occupied_source(name)
    seed_points = {
        "solid": (1.5, 1.5, 1.5),
        "material": (0.5, 0.5, 0.5),
        "left": (0.5, 0.5, 0.5),
        "right": (2.5, 0.5, 0.5),
    }
    seeds = tuple(
        RegionSeed(
            np.asarray(seed_points[region], dtype=np.float64),
            region,
            f"material-{region}",
            RegionRole.SOLID,
        )
        for region in region_measures
    )
    route = (
        "mapped_balanced_grid_hex"
        if name == "curved-cavity-hex"
        else "mapped_frame_grid_hex"
    )
    grid_route = "balanced_grid" if route == "mapped_balanced_grid_hex" else "frame_grid"
    grid = NativeHexGridSchedule(
        grid_route,
        minimum_scaled_jacobian=_QUALITY_LIMITS["minimum_scaled_jacobian"],
        minimum_mean_ratio=_QUALITY_LIMITS["minimum_mean_ratio"],
        maximum_aspect_ratio=_QUALITY_LIMITS["maximum_aspect_ratio"],
    )
    options = NativeMeshingOptions(route, grid_schedule=grid)
    specification = _volume_specification(
        source,
        resolution,
        "hexahedron",
        geometry_order=2,
        region_seeds=seeds,
    )
    return _FamilyInput(
        source,
        specification,
        options,
        route,
        {
            "fixture": f"NativeMappedHexSource:{name}",
            **details,
            "source_map_revision": source.source_revision,
            "requested_geometry_order": 2,
            "resolution": resolution,
            "grid_schedule": {
                "route": grid.route,
                **_QUALITY_LIMITS,
            },
        },
        sum(region_measures.values()),
        expected_regions=region_measures,
        expected_domain=source.domain,
        fidelity_tolerance=0.0,
        require_region_coverage=True,
        quality_limits=dict(_QUALITY_LIMITS),
    )


def _quarter_surface_case(resolution: int) -> _FamilyInput:
    source = _quarter_extrusion()
    scope = MeshingScope.whole(source, 2)
    protected = (ProtectedFeature(scope, FeatureKind.SURFACE, maximum_deviation=0.0),)
    specification = _surface_specification(
        source,
        resolution,
        "quadrilateral",
        ambient_dimension=3,
        geometry_order=3,
        protected_features=protected,
    )
    return _FamilyInput(
        source,
        specification,
        NativeMeshingOptions("parametric_surface"),
        "parametric_surface",
        {
            "fixture": "NativeSurfaceSource:original-quarter-extrusion",
            "rational_quarter_arc_radius_m": 1.0,
            "extrusion_length_m": 1.0,
            "surface_control_points_m": np.asarray(
                [[(x, 1.0, 0.0), (x, 1.0, 1.0), (x, 0.0, 1.0)] for x in (0.0, 1.0)]
            ).tolist(),
            "rational_weights": (1.0, float(np.sqrt(0.5)), 1.0),
            "exact_surface_area_m2": float(np.pi / 2.0),
            "geometry_order": 3,
            "maximum_deviation_m": 0.0,
            "resolution": resolution,
        },
        float(np.pi / 2.0),
        fidelity_tolerance=0.0,
        quality_limits={"minimum_angle": 0.01},
    )


def _make_profile(profile_id: str) -> _FamilyInput:
    case_name = profile_id.removeprefix(_PROFILE_PREFIX)
    larger = case_name.endswith(":larger")
    name = case_name.removesuffix(":larger")
    resolution = _BASE_RESOLUTION * (2 if larger else 1)
    if name == "dual-quad":
        return _planar_dual_quad(resolution)
    if name == "curved-quarter-extrusion-quad":
        return _quarter_surface_case(resolution)
    if name in ("dual-hex", "hex-dominant"):
        return _plc_source_case(name, resolution)
    if name == "nonsweepable-dual-hex":
        return _nonsweepable_dual_hex(resolution)
    if name == "curved-frame-sweep-hex":
        return _frame_sweep_source(resolution)
    if name == "multiblock-reversed-hex":
        return _reversed_multiblock(resolution, False)
    if name == "multiblock-material-hex":
        return _reversed_multiblock(resolution, True)
    if name == "multiblock-cycle-quad":
        return _annular_blocks(resolution)
    if name in ("curved-nonsweepable-hex", "curved-cavity-hex", "curved-material-hex"):
        return _mapped_hex_case(name, resolution)
    if name == "transfinite-quad":
        return _structured_square(resolution, False)
    if name == "multiblock-quad":
        return _structured_square(resolution, True)
    if name == "sweep-hex":
        return _sweep_source(resolution, "quadrilateral")
    if name == "sweep-prism":
        return _sweep_source(resolution, "triangle")
    raise ValueError(f"Unregistered native family source profile {profile_id!r}.")


def _source_record(source: Any) -> dict[str, object]:
    try:
        domain = getattr(source, "domain", None)
    except (AttributeError, ValueError):
        domain = None
    if domain is None and isinstance(source, MeshingDomain):
        domain = source
    result: dict[str, object] = {
        "source_type": f"{type(source).__module__}.{type(source).__qualname__}",
        "source_id": getattr(source, "source_id", None),
        "source_revision": getattr(source, "source_revision", None),
        "binding_id": getattr(source, "binding_id", None),
        "domain_id": getattr(domain, "domain_id", None),
    }
    complex_ = getattr(source, "complex", None)
    if complex_ is not None:
        result.update(
            {
                "complex_id": complex_.complex_id,
                "source_vertices": int(np.asarray(complex_.vertices).shape[0]),
                "source_vertex_coordinates_m": np.asarray(complex_.vertices).tolist(),
                "source_vertex_ids": tuple(complex_.vertex_ids),
                "source_facets": int(complex_.facet_count),
                "source_facet_ids": tuple(complex_.facet_ids),
                "source_facet_region_sides": np.asarray(complex_.facet_regions).tolist(),
                "source_polygon_offsets": np.asarray(complex_.polygon_offsets).tolist(),
                "source_polygon_vertices": np.asarray(complex_.polygon_vertices).tolist(),
                "source_polygon_facet_indices": np.asarray(
                    complex_.polygon_facets
                ).tolist(),
                "source_regions": tuple(complex_.region_ids),
            }
        )
    if isinstance(domain, PiecewiseLinearDomain):
        result["domain_region_ids"] = tuple(domain.region_ids)
        result["domain_vertices"] = np.asarray(domain.vertices).tolist()
        result["domain_facets"] = np.asarray(domain.facets).tolist()
        result["domain_facet_region_sides"] = np.asarray(domain.facet_regions).tolist()
    elif isinstance(domain, MappedReferenceDomain):
        reference = domain.reference_domain
        root_mesh = domain.reference_mesh
        result["domain_region_ids"] = tuple(domain.region_ids)
        result["reference_domain_vertices"] = np.asarray(reference.vertices).tolist()
        result["reference_domain_facets"] = np.asarray(reference.facets).tolist()
        result["reference_domain_facet_region_sides"] = np.asarray(
            reference.facet_regions
        ).tolist()
        result["root_coordinates"] = np.asarray(root_mesh.coordinates).tolist()
        result["root_vertex_ids"] = np.asarray(root_mesh.vertex_global_ids).tolist()
        result["root_cell_connectivity"] = {
            block.name: np.asarray(block.vertices).tolist() for block in root_mesh.blocks
        }
        result["root_cell_regions"] = np.asarray(domain.cell_regions).tolist()
        result["root_physical_coordinate_dofs"] = np.asarray(
            domain.source_geometry.coordinates
        ).tolist()
        result["mapped_domain_id"] = domain.domain_id
    blocks = getattr(source, "blocks", None)
    if blocks is not None:
        result["block_ids"] = tuple(block.name for block in blocks)
        result["block_intervals"] = {
            block.name: tuple(block.intervals) for block in blocks
        }
        result["interface_ids"] = tuple(
            interface.control_id for interface in source.interfaces
        )
    profile = getattr(source, "profile", None)
    if profile is not None:
        result["profile_mesh_id"] = profile.mesh_id
        result["profile_coordinates"] = np.asarray(profile.coordinates).tolist()
        result["profile_cell_families"] = tuple(
            block.cell_kind for block in profile.blocks
        )
        result["profile_cell_connectivity"] = {
            block.name: np.asarray(block.vertices).tolist() for block in profile.blocks
        }
        result["profile_cell_count"] = sum(block.cell_count for block in profile.blocks)
        result["profile_geometry_dofs"] = np.asarray(
            source.profile_geometry.coordinates
        ).tolist()
    return result


def _request_record(case: _FamilyInput) -> dict[str, object]:
    specification = case.specification
    target = specification.target
    sizes = tuple(
        {
            "type": type(control).__name__,
            "target_size_m": getattr(control, "target_size", None),
            "maximum_size_m": getattr(control, "maximum_size", None),
            "strength": getattr(getattr(control, "strength", None), "value", None),
        }
        for control in specification.size_controls
    )
    families = target.cell_families
    schedule_records: dict[str, object] = {}
    schedule_fields = (
        "route",
        "radius_edge_bound",
        "minimum_dihedral_degrees",
        "minimum_scaled_jacobian",
        "minimum_mean_ratio",
        "maximum_aspect_ratio",
        "quality_angle_degrees",
        "spacing_fraction",
        "minimum_edge_samples",
        "optimization_steps",
    )
    for schedule_name in (
        "curve_schedule",
        "surface_schedule",
        "volume_schedule",
        "structured_schedule",
        "grid_schedule",
    ):
        schedule = getattr(case.options, schedule_name)
        if schedule is None:
            continue
        schedule_record: dict[str, object] = {
            "schedule_id": getattr(schedule, "schedule_id", None)
        }
        for field_name in schedule_fields:
            if hasattr(schedule, field_name):
                value = getattr(schedule, field_name)
                schedule_record[field_name] = getattr(value, "value", value)
        schedule_records[schedule_name] = schedule_record
    quality_target = (
        specification.quality_target
        if isinstance(specification, SurfaceMeshingSpec)
        else None
    )
    return {
        "specification_id": specification.specification_id,
        "requested_route": case.expected_route,
        "options": {
            "route": case.options.route,
            "hex_core_fraction": case.options.hex_core_fraction,
            "schedules": schedule_records,
        },
        "topological_dimension": target.topological_dimension,
        "ambient_dimension": target.ambient_dimension,
        "geometry_order": target.geometry_order,
        "family_policy": {
            "required": tuple(families.required),
            "preferred": tuple(families.preferred),
            "allowed_transitions": tuple(families.allowed_transitions),
            "allow_mixed": families.allow_mixed,
        },
        "fill_strategy": getattr(
            getattr(specification, "fill_strategy", None), "value", None
        ),
        "size_controls": sizes,
        "size_compliance": {
            "relative_tolerance": specification.size_compliance.relative_tolerance,
            "target_statistics": tuple(specification.size_compliance.target_statistics),
        },
        "quality_target": None
        if quality_target is None
        else {
            "minimum_angle_radians": quality_target.minimum_angle,
            "hard": quality_target.hard,
        },
    }


def _measure_coverage(result: Any) -> dict[str, float]:
    certification = result.certification
    coverage = None if certification is None else certification.coverage
    if coverage is None:
        return {}
    return {
        name: float(measure)
        for name, measure in zip(
            coverage.region_ids,
            coverage.achieved_region_measures,
            strict=True,
        )
        if measure is not None
    }


def _family_counts(result: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in result.mesh.blocks:
        counts[block.cell_kind] = counts.get(block.cell_kind, 0) + block.cell_count
    return counts


def _fidelity_record(case: _FamilyInput, result: Any) -> dict[str, object]:
    certification = result.certification
    fidelity = None if certification is None else certification.fidelity
    record: dict[str, object] = {
        "declared_maximum_deviation_m": case.fidelity_tolerance,
        "certificate_present": fidelity is not None,
    }
    if fidelity is not None:
        record.update(
            {
                "mesh_to_source_upper_m": float(fidelity.mesh_to_source_upper),
                "source_to_mesh_upper_m": float(fidelity.source_to_mesh_upper),
                "certificate_id": fidelity.certificate_id,
            }
        )
        if case.fidelity_tolerance is not None and (
            fidelity.mesh_to_source_upper > case.fidelity_tolerance + 1.0e-12
            or fidelity.source_to_mesh_upper > case.fidelity_tolerance + 1.0e-12
        ):
            raise RuntimeError(
                "Source-fidelity certificate exceeds the profile's declared maximum deviation."
            )
    elif case.fidelity_tolerance is not None:
        if (
            isinstance(case.source, NativePlcSource)
            and case.source.fidelity_source is None
        ):
            record["fidelity_basis"] = (
                "exact retained PLC facet identity and complete source associations"
            )
        elif isinstance(case.source, NativeMappedHexSource):
            record["fidelity_basis"] = (
                "exact mapped root-cell charts, PLC-to-root vertex relation, and output source-map identity"
            )
        elif (
            isinstance(case.source, NativePlanarSource) and case.fidelity_tolerance == 0.0
        ):
            record["fidelity_basis"] = (
                "exact authored planar loops, source-edge associations, and zero-deviation source contract"
            )
        else:
            raise RuntimeError(
                "A source-bound fidelity certificate is required by this profile."
            )
    return record


def _accept_profile(
    profile_id: str, case: _FamilyInput, plan: Any, result: Any
) -> dict[str, object]:
    if plan.route != case.expected_route:
        raise RuntimeError(
            f"Native source/request resolved to route {plan.route!r}, expected exact profile route {case.expected_route!r}."
        )
    accepted = accepted_result_record(result)
    counts = _family_counts(result)
    policy = case.specification.target.cell_families
    required = set(policy.required)
    permitted = required | set(policy.preferred) | set(policy.allowed_transitions)
    if not required <= set(counts):
        raise RuntimeError(
            f"Required cell-family policy {sorted(required)} is absent from {counts}."
        )
    if policy.allow_mixed:
        if not set(counts) <= permitted:
            raise RuntimeError(
                f"Output contains a cell family outside the declared transition policy: {counts}."
            )
        if case.require_transition and not (
            set(counts) & set(policy.allowed_transitions)
        ):
            raise RuntimeError(
                "The hex-dominant source produced no declared transition cells."
            )
    elif set(counts) != required:
        raise RuntimeError(
            f"Pure-family request produced {counts}, expected only {sorted(required)}."
        )

    measures = np.asarray(result.quality.evaluation.measures, dtype=np.float64)
    if measures.size == 0 or not np.all(np.isfinite(measures)) or np.any(measures <= 0.0):
        raise RuntimeError(
            "Independent cell measures contain an empty, non-finite, or non-positive value."
        )
    measured_total = float(np.sum(measures, dtype=np.float64))
    relative_error = abs(measured_total - case.expected_measure) / case.expected_measure
    if relative_error > _PROFILE_TOLERANCE:
        raise RuntimeError(
            f"Independent physical measure {measured_total:.17g} differs from authored oracle {case.expected_measure:.17g} by relative error {relative_error:.17g}."
        )

    measured_regions = _measure_coverage(result)
    if case.require_region_coverage:
        if set(measured_regions) != set(case.expected_regions):
            raise RuntimeError(
                f"Independent region coverage names {sorted(measured_regions)}, expected {sorted(case.expected_regions)}."
            )
    region_errors: dict[str, float] = {}
    for region, expected in case.expected_regions.items():
        if region not in measured_regions:
            if case.require_region_coverage:
                raise RuntimeError(
                    f"Independent material measure for source region {region!r} is missing."
                )
            continue
        error = abs(measured_regions[region] - expected) / expected
        region_errors[region] = error
        if error > _PROFILE_TOLERANCE:
            raise RuntimeError(
                f"Source region {region!r} measure {measured_regions[region]:.17g} differs from its independent oracle {expected:.17g}."
            )
    if case.require_region_coverage and len(case.expected_regions) > 1:
        expected_materials = {f"material-{region}" for region in case.expected_regions}
        observed_materials = {
            zone.material_id for zone in result.zones if zone.material_id is not None
        }
        if observed_materials != expected_materials:
            raise RuntimeError(
                f"Material ancestry {sorted(observed_materials)} differs from expected {sorted(expected_materials)}."
            )

    associations = tuple(result.associations)
    if not associations or not any(item.complete for item in associations):
        raise RuntimeError(
            "Accepted native result omitted complete source ancestry associations."
        )
    source_id = getattr(case.source, "source_id", None)
    source_revision = getattr(case.source, "source_revision", None)
    matching_associations = tuple(
        item
        for item in associations
        if item.source_id == source_id and item.source_revision == source_revision
    )
    if not matching_associations or not any(
        item.complete for item in matching_associations
    ):
        raise RuntimeError(
            "Complete source associations do not retain this profile's source/revision identity."
        )
    source_roles = sorted(
        {
            getattr(role, "value", str(role))
            for association in matching_associations
            for role in association.source_roles
        }
    )

    quality = result.quality
    quality_record: dict[str, object] = {
        "minimum_measure": quality.minimum_measure,
        "minimum_scaled_jacobian": quality.minimum_scaled_jacobian,
        "minimum_mean_ratio": quality.minimum_mean_ratio,
        "maximum_aspect_ratio": quality.maximum_aspect_ratio,
        "sampled_invalid_count": quality.sampled_invalid_count,
        "hard_thresholds": dict(case.quality_limits),
    }
    if quality.sampled_invalid_count != 0 or quality.minimum_measure <= 0.0:
        raise RuntimeError(
            "Independent cell-quality report found invalid or non-positive cells."
        )
    if (
        "minimum_scaled_jacobian" in case.quality_limits
        and quality.minimum_scaled_jacobian
        < case.quality_limits["minimum_scaled_jacobian"]
    ):
        raise RuntimeError(
            "Minimum scaled Jacobian is below the authored hex-grid schedule."
        )
    if (
        "minimum_mean_ratio" in case.quality_limits
        and quality.minimum_mean_ratio < case.quality_limits["minimum_mean_ratio"]
    ):
        raise RuntimeError("Minimum mean ratio is below the authored hex-grid schedule.")
    if (
        "maximum_aspect_ratio" in case.quality_limits
        and quality.maximum_aspect_ratio > case.quality_limits["maximum_aspect_ratio"]
    ):
        raise RuntimeError("Maximum aspect ratio exceeds the authored hex-grid schedule.")
    if "minimum_angle" in case.quality_limits:
        minimum_angle = quality.minimum_angle
        if minimum_angle is None or minimum_angle < case.quality_limits["minimum_angle"]:
            raise RuntimeError(
                "Minimum angle is below the authored profile quality target."
            )
        quality_record["minimum_angle_radians"] = minimum_angle

    fidelity = _fidelity_record(case, result)
    if (
        case.expected_domain is not None
        and result.certification.inputs.domain is not case.expected_domain
    ):
        raise RuntimeError(
            "Native certification lost the exact authored source-domain object."
        )
    mapped_source_map = result.geometry.source_map
    if isinstance(case.source, NativeMappedHexSource) or (
        isinstance(case.source, NativeSweepSource)
        and isinstance(case.source.domain, MappedReferenceDomain)
    ):
        if mapped_source_map is None or (
            mapped_source_map.source_id,
            mapped_source_map.source_revision,
        ) != (source_id, source_revision):
            raise RuntimeError(
                "Mapped native output lost its exact source ID/revision fidelity map."
            )

    return {
        "status": "passed",
        "profile_id": profile_id,
        "expected_admission": "positive",
        "route": plan.route,
        "plan_id": plan.plan_id,
        "source_provenance": _source_record(case.source),
        "physical_domain_oracle": {
            "independent_total_measure": case.expected_measure,
            "measured_total_measure": measured_total,
            "relative_error": relative_error,
            "tolerance_relative": _PROFILE_TOLERANCE,
            "independent_region_measures": dict(case.expected_regions),
            "measured_region_measures": measured_regions,
            "region_relative_errors": region_errors,
        },
        "family_inventory": counts,
        "quality_oracle": quality_record,
        "ancestry": {
            "association_count": len(associations),
            "complete_association_count": sum(
                bool(item.complete) for item in associations
            ),
            "matching_source_association_count": len(matching_associations),
            "source_roles": source_roles,
        },
        "source_fidelity": fidelity,
        "accepted_native_evidence": accepted,
    }


def _run_profile(profile_id: str) -> dict[str, object]:
    phase = "source-and-request-construction"
    case: _FamilyInput | None = None
    scenario_name = profile_id.removeprefix(_PROFILE_PREFIX)
    larger = scenario_name.endswith(":larger")
    family_case = scenario_name.removesuffix(":larger")
    resolution_scale = 2 if larger else 1
    resolved_resolution = _BASE_RESOLUTION * resolution_scale
    source_provenance: dict[str, object] = {
        "migration_registry": "tools/meshing_qualification/native_case_migration.json#general-quads-hexes",
        "profile_identity": profile_id,
        "fixture_origin": "donor native-family-lifecycle source registry; ported onto current native public APIs",
        "api_contract_sources": (
            "tests/unit/meshing/test_native_general_families.py",
            "tests/unit/meshing/test_native_structured_routes.py",
        ),
    }
    run_record: dict[str, object] = {
        "profile_id": profile_id,
        "expected_admission": "positive",
        "source_provenance": source_provenance,
        "accepted_native_evidence": None,
        "profile_controls": {
            "scenario_controls": {"family_case": family_case},
            "resolution_scale": resolution_scale,
            "base_resolution": _BASE_RESOLUTION,
            "resolved_resolution": resolved_resolution,
            "size_target_m": 1.0 / resolved_resolution,
            "maximum_size_m": 2.0 / resolved_resolution,
            "size_strength": "hard",
            "size_compliance": {
                "relative_tolerance": 0.95,
                "target_statistics": ("p50",),
            },
        },
    }
    try:
        case = _make_profile(profile_id)
        source_provenance.update(_source_record(case.source))
        run_record["source_input"] = case.physical_input
        run_record["request"] = _request_record(case)
        run_record["oracles_and_tolerances"] = {
            "independent_physical_measure": case.expected_measure,
            "independent_region_measures": dict(case.expected_regions),
            "region_measure_relative_tolerance": _PROFILE_TOLERANCE,
            "source_fidelity_maximum_deviation_m": case.fidelity_tolerance,
            "cell_quality_hard_limits": dict(case.quality_limits),
            "family_policy": "exact required family; only explicitly named hex-dominant transitions are permitted",
        }
        phase = "native-plan"
        plan = NativeMeshingProvider(case.options).plan(
            case.source,
            case.specification,
            coordinate_contract=_CONTRACT,
        )
        run_record["plan_route"] = plan.route
        phase = "native-execute"
        result = plan.execute()
        run_record["native_result"] = {
            "result_id": result.result_id,
            "topology_id": result.mesh.topology_id,
            "provider": result.provider.name,
            "audit_passed": result.audit.passed,
            "certification_passed": result.certification is not None
            and result.certification.passed,
            "compliance_passed": result.compliance.passed,
        }
        phase = "independent-acceptance"
        accepted = _accept_profile(profile_id, case, plan, result)
        run_record.update(accepted)
        run_record["source_input"] = case.physical_input
        run_record["request"] = _request_record(case)
        return run_record
    except Exception as error:
        category = getattr(error, "category", None)
        run_record.update(
            {
                "status": "failed",
                "failure": {
                    "phase": phase,
                    "error_type": type(error).__name__,
                    "error_category": getattr(category, "value", category),
                    "api_or_oracle_error": str(error),
                },
            }
        )
        if case is not None:
            source_provenance.update(_source_record(case.source))
            run_record["source_input"] = case.physical_input
            run_record["request"] = _request_record(case)
            run_record["oracles_and_tolerances"] = {
                "independent_physical_measure": case.expected_measure,
                "independent_region_measures": dict(case.expected_regions),
                "region_measure_relative_tolerance": _PROFILE_TOLERANCE,
                "source_fidelity_maximum_deviation_m": case.fidelity_tolerance,
                "cell_quality_hard_limits": dict(case.quality_limits),
                "family_policy": "exact required family; only explicitly named hex-dominant transitions are permitted",
            }
        return run_record


def native_general_families() -> dict[str, object]:
    """Run every positive native general-quad/hex source profile as an independent gate."""
    profile_ids = _profile_ids()
    records = {profile_id: _run_profile(profile_id) for profile_id in profile_ids}
    runtime = runtime_identity()
    return {
        "status": "passed"
        if all(record["status"] == "passed" for record in records.values())
        else "failed",
        "workflow": "all original native general quadrilateral and hexahedron source profiles",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#general-quads-hexes",
        "expected_admission": "positive",
        "mandatory_profile_count": len(profile_ids),
        "passed_profile_count": sum(
            record["status"] == "passed" for record in records.values()
        ),
        "failed_profile_count": sum(
            record["status"] != "passed" for record in records.values()
        ),
        "profile_run_records": records,
        "runtime": runtime,
        "meshcore_identity": runtime["meshcore"],
    }
