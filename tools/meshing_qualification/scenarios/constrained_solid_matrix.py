from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    TetrahedralConnectivity,
)
from phydrax.meshing._association import GeometryAssociationKind, GeometrySourceEntityRole
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingScope,
    PatchControl,
    RegionControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import FeatureKind, ProtectedFeature
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex, PlcBoundaryPolicy
from phydrax.meshing._scope import MeshingEntityKind
from phydrax.meshing._sizing import (
    SizeCompliancePolicy,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_sources import NativePlcSource

from .._runtime import accepted_result_record, runtime_identity


_CONTRACT = SpatialCoordinateContract.si()
_REGION_RELATIVE_TOLERANCE = 1.0e-10
_FACET_AREA_RELATIVE_TOLERANCE = 1.0e-10
_FE_TOLERANCE = 2.0e-9


@dataclass(frozen=True, slots=True)
class _SolidProfile:
    profile_id: str
    source_case_id: str
    controls: tuple[str, ...]
    region_ids: tuple[str, ...]
    expected_region_volumes: tuple[tuple[str, Fraction], ...]
    reference_boxes: tuple[
        tuple[int, tuple[float, float, float], tuple[float, float, float]], ...
    ]
    resolution_scale: int = 1


@dataclass(frozen=True, slots=True)
class _SolidInput:
    vertices: np.ndarray
    polygons: tuple[tuple[int, ...], ...]
    polygon_facets: np.ndarray
    facet_regions: np.ndarray
    region_ids: tuple[str, ...]
    segments: np.ndarray
    boundary: PlcBoundaryPolicy
    cavity: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    small_angle_wedge: bool = False


_PROFILES = (
    _SolidProfile(
        "plc-disconnected",
        "plc-disconnected",
        ("disconnected", "hard-size", "minimum-quality"),
        ("solid",),
        (("solid", Fraction(2)),),
        (
            (0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            (0, (2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
        ),
    ),
    _SolidProfile(
        "plc-fixed-boundary",
        "plc-fixed-boundary",
        ("fixed-boundary", "soft-size"),
        ("solid",),
        (("solid", Fraction(1)),),
        ((0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),),
    ),
    _SolidProfile(
        "plc-material-cavity",
        "plc-material-cavity",
        (
            "cavity",
            "material-interface",
            "embedded-segment",
            "hard-size",
            "minimum-quality",
        ),
        ("left", "right"),
        (("left", Fraction(1)), ("right", Fraction(7, 8))),
        (
            (0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            (1, (1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
        ),
    ),
    _SolidProfile(
        "plc-material-cavity:larger",
        "plc-material-cavity",
        (
            "cavity",
            "material-interface",
            "embedded-segment",
            "hard-size",
            "minimum-quality",
        ),
        ("left", "right"),
        (("left", Fraction(1)), ("right", Fraction(7, 8))),
        (
            (0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            (1, (1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
        ),
        resolution_scale=2,
    ),
    _SolidProfile(
        "plc-nonconvex",
        "plc-nonconvex",
        ("nonconvex", "soft-size"),
        ("solid",),
        (("solid", Fraction(3)),),
        (
            (0, (0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (0, (0.0, 1.0, 0.0), (1.0, 2.0, 1.0)),
        ),
    ),
    _SolidProfile(
        "plc-sliver-rich",
        "plc-sliver-rich",
        ("sliver-rich", "soft-size", "dihedral-25"),
        ("solid",),
        (("solid", Fraction(1)),),
        ((0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),),
    ),
    _SolidProfile(
        "plc-small-angle",
        "plc-small-angle",
        ("small-angle", "soft-size"),
        ("solid",),
        (("solid", Fraction(16)),),
        (),
    ),
    _SolidProfile(
        "plc-thin-channel",
        "plc-thin-channel",
        ("thin-channel", "hard-size", "minimum-quality"),
        ("solid",),
        (("solid", Fraction(81, 250)),),
        (
            (0, (0.0, 0.0, 0.0), (0.4, 0.4, 1.0)),
            (0, (0.4, 0.19, 0.0), (0.6, 0.21, 1.0)),
            (0, (0.6, 0.0, 0.0), (1.0, 0.4, 1.0)),
        ),
    ),
    _SolidProfile(
        "plc-triple-junction",
        "plc-triple-junction",
        ("triple-junction", "material-interface", "hard-size", "minimum-quality"),
        ("first", "second", "third"),
        (("first", Fraction(1)), ("second", Fraction(1)), ("third", Fraction(2))),
        (
            (0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            (1, (1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (2, (0.0, 1.0, 0.0), (2.0, 2.0, 1.0)),
        ),
    ),
)


def _box_vertices(
    lower: tuple[float, float, float], upper: tuple[float, float, float], /
) -> np.ndarray:
    return np.asarray(
        [
            tuple(
                upper[axis] if (index >> axis) & 1 else lower[axis] for axis in range(3)
            )
            for index in range(8)
        ],
        dtype=np.float64,
    )


def _box_union_input(profile: _SolidProfile) -> _SolidInput:
    boxes = profile.reference_boxes
    axes = tuple(
        sorted({bound[axis] for _, lower, upper in boxes for bound in (lower, upper)})
        for axis in range(3)
    )
    vertices: list[tuple[float, float, float]] = []
    vertex_rows: dict[tuple[float, float, float], int] = {}
    faces: dict[tuple[int, ...], tuple[tuple[int, ...], int, int]] = {}
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    for intervals in np.ndindex(*(len(axis) - 1 for axis in axes)):
        lower = tuple(axes[axis][intervals[axis]] for axis in range(3))
        upper = tuple(axes[axis][intervals[axis] + 1] for axis in range(3))
        center = tuple((lower[axis] + upper[axis]) / 2.0 for axis in range(3))
        region = next(
            (
                index
                for index, box_lower, box_upper in boxes
                if all(
                    box_lower[axis] < center[axis] < box_upper[axis] for axis in range(3)
                )
            ),
            None,
        )
        if region is None:
            continue
        local_vertices = _box_vertices(lower, upper)
        local_rows: list[int] = []
        for point in local_vertices:
            key = (float(point[0]), float(point[1]), float(point[2]))
            if key not in vertex_rows:
                vertex_rows[key] = len(vertices)
                vertices.append(key)
            local_rows.append(vertex_rows[key])
        for loop in loops:
            polygon = tuple(local_rows[row] for row in loop)
            key = tuple(sorted(polygon))
            previous = faces.pop(key, None)
            if previous is None:
                faces[key] = (polygon, region, -1)
            elif previous[1] != region:
                faces[key] = (previous[0], previous[1], region)
    polygons: list[tuple[int, ...]] = []
    incidences: list[tuple[int, int]] = []
    for polygon, positive_region, negative_region in faces.values():
        polygons.append(polygon)
        incidences.append((positive_region, negative_region))
    return _SolidInput(
        np.asarray(vertices, dtype=np.float64),
        tuple(polygons),
        np.arange(len(polygons), dtype=np.int64),
        np.asarray(incidences, dtype=np.int64),
        profile.region_ids,
        np.empty((0, 2), dtype=np.int64),
        "conforming",
    )


def _extruded_concave_input(profile: _SolidProfile, /) -> _SolidInput:
    if profile.source_case_id == "plc-nonconvex":
        base = (
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        )
    else:
        base = (
            (0.0, 0.0),
            (0.4, 0.0),
            (0.4, 0.19),
            (0.6, 0.19),
            (0.6, 0.0),
            (1.0, 0.0),
            (1.0, 0.4),
            (0.6, 0.4),
            (0.6, 0.21),
            (0.4, 0.21),
            (0.4, 0.4),
            (0.0, 0.4),
        )
    count = len(base)
    vertices = np.asarray(
        [(x, y, z) for z in (0.0, 1.0) for x, y in base], dtype=np.float64
    )
    polygons = (
        tuple(reversed(range(count))),
        tuple(range(count, 2 * count)),
        *tuple(
            (index, (index + 1) % count, (index + 1) % count + count, index + count)
            for index in range(count)
        ),
    )
    return _SolidInput(
        vertices,
        polygons,
        np.zeros((len(polygons),), dtype=np.int64),
        np.asarray(((-1, 0),), dtype=np.int64),
        ("solid",),
        np.empty((0, 2), dtype=np.int64),
        "conforming",
    )


def _material_cavity_input() -> _SolidInput:
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    left = _box_vertices((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    right = _box_vertices((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    cavity = _box_vertices((1.25, 0.25, 0.25), (1.75, 0.75, 0.75))
    vertices = np.concatenate((left, right[[1, 3, 5, 7]], cavity))
    right_index = {0: 1, 2: 3, 4: 5, 6: 7, 1: 8, 3: 9, 5: 10, 7: 11}
    polygons: list[tuple[int, ...]] = []
    polygon_facets: list[int] = []
    for loop in loops:
        if loop != (1, 3, 7, 5):
            polygons.append(loop)
            polygon_facets.append(0)
        if loop != (0, 4, 6, 2):
            polygons.append(tuple(right_index[row] for row in loop))
            polygon_facets.append(1)
    polygons.append((1, 3, 7, 5))
    polygon_facets.append(2)
    polygons.extend(tuple(row + 12 for row in loop) for loop in loops)
    polygon_facets.extend((3,) * len(loops))
    vertices = np.concatenate(
        (vertices, np.asarray(((0.25, 0.25, 0.25), (0.75, 0.75, 0.75)), dtype=np.float64))
    )
    return _SolidInput(
        vertices,
        tuple(polygons),
        np.asarray(polygon_facets, dtype=np.int64),
        np.asarray(((-1, 0), (-1, 1), (1, 0), (1, -1)), dtype=np.int64),
        ("left", "right"),
        np.asarray(((20, 21),), dtype=np.int64),
        "conforming",
        cavity=((1.25, 0.25, 0.25), (1.75, 0.75, 0.75)),
    )


def _small_angle_input() -> _SolidInput:
    base = ((0.0, 0.0), (16.0, -1.0), (16.0, 1.0))
    vertices = np.asarray(
        [(x, y, z) for z in (0.0, 1.0) for x, y in base], dtype=np.float64
    )
    polygons = (
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    )
    return _SolidInput(
        vertices,
        polygons,
        np.zeros((len(polygons),), dtype=np.int64),
        np.asarray(((-1, 0),), dtype=np.int64),
        ("solid",),
        np.empty((0, 2), dtype=np.int64),
        "conforming",
        small_angle_wedge=True,
    )


def _cube_input(profile: _SolidProfile) -> _SolidInput:
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    vertices = _box_vertices((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    if profile.profile_id == "plc-sliver-rich":
        vertices = np.concatenate(
            (
                vertices,
                np.asarray(((0.5, 0.5, 0.001), (0.5, 0.5, 0.999)), dtype=np.float64),
            )
        )
        polygons = loops
        polygon_facets = np.arange(6, dtype=np.int64)
        facet_regions = np.tile(np.asarray((-1, 0), dtype=np.int64), (6, 1))
    else:
        triangles = tuple(
            triangle
            for first, second, third, fourth in loops
            for triangle in (
                ((first, second, third), (first, third, fourth))
                if first % 2 == 0
                else ((first, second, fourth), (second, third, fourth))
            )
        )
        polygons = triangles
        polygon_facets = np.zeros((12,), dtype=np.int64)
        facet_regions = np.asarray(((-1, 0),), dtype=np.int64)
    return _SolidInput(
        vertices,
        tuple(tuple(row) for row in polygons),
        polygon_facets,
        facet_regions,
        ("solid",),
        np.empty((0, 2), dtype=np.int64),
        "fixed" if profile.profile_id == "plc-fixed-boundary" else "conforming",
    )


def _input_for(profile: _SolidProfile) -> _SolidInput:
    if profile.source_case_id in {"plc-disconnected", "plc-triple-junction"}:
        return _box_union_input(profile)
    if profile.source_case_id in {"plc-nonconvex", "plc-thin-channel"}:
        return _extruded_concave_input(profile)
    if profile.source_case_id == "plc-material-cavity":
        return _material_cavity_input()
    if profile.source_case_id == "plc-small-angle":
        return _small_angle_input()
    return _cube_input(profile)


def _source_for(profile: _SolidProfile, source_input: _SolidInput, /) -> NativePlcSource:
    source_identity = {
        "kind": "authored-constrained-solid-plc",
        "source_case_id": profile.source_case_id,
        "vertices": source_input.vertices,
        "polygons": source_input.polygons,
        "polygon_facets": source_input.polygon_facets,
        "facet_regions": source_input.facet_regions,
        "regions": source_input.region_ids,
        "segments": source_input.segments,
        "boundary": source_input.boundary,
    }
    source_revision = canonical_fingerprint(source_identity)
    complex_ = PiecewiseLinearComplex(
        source_input.vertices,
        tuple(np.asarray(polygon, dtype=np.int64) for polygon in source_input.polygons),
        source_input.polygon_facets,
        source_input.facet_regions,
        source_input.region_ids,
        segments=source_input.segments,
        vertex_ids=tuple(
            f"{profile.source_case_id}:vertex:{row}"
            for row in range(source_input.vertices.shape[0])
        ),
        vertex_occurrence_paths=((),) * source_input.vertices.shape[0],
        facet_ids=tuple(
            f"{profile.source_case_id}:facet:{row}"
            for row in range(source_input.facet_regions.shape[0])
        ),
        facet_occurrence_paths=((),) * source_input.facet_regions.shape[0],
        segment_ids=tuple(
            f"{profile.source_case_id}:segment:{row}"
            for row in range(source_input.segments.shape[0])
        ),
        segment_occurrence_paths=((),) * source_input.segments.shape[0],
        region_occurrence_paths=((),) * len(source_input.region_ids),
        boundary=source_input.boundary,
    )
    return NativePlcSource(complex_, profile.source_case_id, source_revision)


def _scope(
    source: NativePlcSource, dimension: int, identifiers: np.ndarray, /
) -> MeshingScope:
    return MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        dimension,
        source.scope_entity_set_id(dimension),
        identifiers,
    )


def _request(
    profile: _SolidProfile,
    source: NativePlcSource,
    source_input: _SolidInput,
    /,
) -> tuple[VolumeMeshingSpec, NativeMeshingOptions, dict[str, object]]:
    complex_ = source.complex
    boundary_scope = MeshingScope.whole(source, 2)
    region_controls = tuple(
        RegionControl(
            _scope(
                source,
                3,
                np.asarray((complex_.region_entity_ids[index],), dtype=np.int64),
            ),
            region_id,
            f"{region_id}-material",
            RegionRole.SOLID,
        )
        for index, region_id in enumerate(source_input.region_ids)
    )
    patch_controls: list[PatchControl] = []
    interface_facets: list[int] = []
    facet_regions = np.asarray(source_input.facet_regions, dtype=np.int64).reshape((-1, 2))
    for facet in range(facet_regions.shape[0]):
        positive = int(facet_regions[facet, 0])
        negative = int(facet_regions[facet, 1])
        if positive < 0 or negative < 0 or positive == negative:
            continue
        interface_facets.append(facet)
        name = (
            "left-right-interface"
            if profile.source_case_id == "plc-material-cavity"
            else f"interface-{facet}"
        )
        patch_controls.append(
            PatchControl(
                name,
                _scope(
                    source,
                    2,
                    np.asarray((complex_.facet_entity_ids[facet],), dtype=np.int64),
                ),
                (source_input.region_ids[positive], source_input.region_ids[negative]),
            )
        )
    protected_features: list[ProtectedFeature] = []
    protected_segment_edges: list[int] = []
    for segment in np.asarray(source_input.segments, dtype=np.int64):
        key = tuple(sorted(int(value) for value in segment))
        edge_rows = [
            row
            for row, edge in enumerate(complex_.plc_edge_vertices)
            if tuple(sorted(int(value) for value in edge)) == key
        ]
        if len(edge_rows) != 1:
            raise RuntimeError(
                f"Authored embedded PLC segment {key} did not map to one stable source edge."
            )
        edge_row = edge_rows[0]
        protected_segment_edges.append(edge_row)
        protected_features.append(
            ProtectedFeature(
                _scope(
                    source,
                    1,
                    np.asarray((complex_.plc_edge_entity_ids[edge_row],), dtype=np.int64),
                ),
                FeatureKind.CURVE,
                hard=True,
            )
        )
    size_strength = (
        SizeControlStrength.HARD
        if "hard-size" in profile.controls
        else SizeControlStrength.SOFT
    )
    if profile.profile_id == "plc-sliver-rich":
        target_size = 0.25
    elif profile.profile_id == "plc-fixed-boundary":
        target_size = 0.3
    elif "soft-size" in profile.controls:
        target_size = 4.0
    else:
        target_size = 0.8 / profile.resolution_scale
    size_control = UniformSizeControl(
        boundary_scope,
        target_size,
        maximum_size=1.5 * target_size,
        strength=size_strength,
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(size_control,),
        protected_features=tuple(protected_features),
        region_controls=region_controls,
        patch_controls=tuple(patch_controls),
        size_compliance=SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
    )
    minimum_dihedral = 25.0 if "dihedral-25" in profile.controls else 10.0
    volume_schedule = NativeVolumeSchedule(
        radius_edge_bound=2.0,
        minimum_dihedral_degrees=minimum_dihedral,
    )
    options = NativeMeshingOptions(
        "plc_tetrahedral",
        volume_schedule=volume_schedule,
    )
    controls = {
        "source_boundary_policy": source_input.boundary,
        "region_controls": [
            {
                "region": control.region_name,
                "material": control.material_id,
                "role": control.role.value,
                "source_region_entity_id": int(control.scope.entity_ids[0]),
            }
            for control in region_controls
        ],
        "interface_patch_controls": [
            {
                "name": control.name,
                "adjacent_regions": list(control.adjacent_region_names),
                "source_facet_entity_id": int(control.scope.entity_ids[0]),
            }
            for control in patch_controls
        ],
        "protected_embedded_segment_edges": [
            {
                "segment_id": source.complex.segment_ids[index],
                "source_edge_id": source.complex.plc_edge_ids[edge_row],
                "source_edge_entity_id": int(
                    source.complex.plc_edge_entity_ids[edge_row]
                ),
                "feature_kind": FeatureKind.CURVE.value,
                "hard": True,
            }
            for index, edge_row in enumerate(protected_segment_edges)
        ],
        "size_control": {
            "target_size": target_size,
            "maximum_size": 1.5 * target_size,
            "strength": size_strength.value,
            "campaign_resolution_scale": profile.resolution_scale,
            "compliance_relative_tolerance": 0.5,
            "compliance_statistics": ["p50"],
        },
        "native_volume_schedule": {
            "radius_edge_bound": volume_schedule.radius_edge_bound,
            "minimum_dihedral_degrees": minimum_dihedral,
        },
        "interface_facet_rows": interface_facets,
        "fill_strategy": VolumeFillStrategy.SIMPLEX.value,
        "cell_family": "tetrahedron",
        "provider_route": "plc_tetrahedral",
    }
    return specification, options, controls


def _input_record(
    profile: _SolidProfile,
    source: NativePlcSource,
    source_input: _SolidInput,
    controls: dict[str, object],
    /,
) -> dict[str, object]:
    return {
        "profile_id": profile.profile_id,
        "source_case_id": profile.source_case_id,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "complex_id": source.complex.complex_id,
        "boundary_policy": source_input.boundary,
        "vertices": np.asarray(source_input.vertices, dtype=np.float64).tolist(),
        "polygons": [list(polygon) for polygon in source_input.polygons],
        "polygon_facets": np.asarray(
            source_input.polygon_facets, dtype=np.int64
        ).tolist(),
        "facet_ids": list(source.complex.facet_ids),
        "facet_regions": [
            [
                None if int(value) < 0 else source_input.region_ids[int(value)]
                for value in pair
            ]
            for pair in np.asarray(source_input.facet_regions, dtype=np.int64)
        ],
        "regions": list(source_input.region_ids),
        "segments": np.asarray(source_input.segments, dtype=np.int64).tolist(),
        "explicit_controls": controls,
    }


def _analytic_region_indices(
    profile: _SolidProfile,
    source_input: _SolidInput,
    points: np.ndarray,
    /,
) -> np.ndarray:
    if source_input.small_angle_wedge:
        x, y, z = np.moveaxis(np.asarray(points, dtype=np.float64), -1, 0)
        inside = (
            (x >= 0.0) & (x <= 16.0) & (np.abs(y) <= x / 16.0) & (z >= 0.0) & (z <= 1.0)
        )
        return np.where(inside, 0, -1).astype(np.int64)
    labels = np.full(np.asarray(points).shape[:-1], -1, dtype=np.int64)
    for region, lower, upper in profile.reference_boxes:
        inside = np.all(
            (points >= np.asarray(lower, dtype=np.float64))
            & (points <= np.asarray(upper, dtype=np.float64)),
            axis=-1,
        )
        labels[inside] = region
    if source_input.cavity is not None:
        lower, upper = source_input.cavity
        in_cavity = np.all(
            (points > np.asarray(lower, dtype=np.float64))
            & (points < np.asarray(upper, dtype=np.float64)),
            axis=-1,
        )
        labels[in_cavity] = -1
    return labels


def _independent_region_oracle(
    profile: _SolidProfile,
    source_input: _SolidInput,
    result: phx.meshing.CellMeshingResult,
    /,
) -> dict[str, object]:
    mesh = result.mesh
    if any(block.cell_kind != "tetrahedron" for block in mesh.blocks):
        raise TypeError(
            f"{profile.profile_id} published non-tetrahedral blocks: "
            f"{sorted({block.cell_kind for block in mesh.blocks})}."
        )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    corners = points[cells]
    signed = np.linalg.det(corners[:, 1:, :] - corners[:, :1, :]) / 6.0
    measures = np.abs(signed)
    centroids = np.mean(corners, axis=1)
    independent_regions = _analytic_region_indices(profile, source_input, centroids)
    if np.any(independent_regions < 0):
        bad_rows = np.flatnonzero(independent_regions < 0)
        raise RuntimeError(
            f"Independent analytic domain oracle classified {bad_rows.size} output tetrahedra "
            "outside every declared solid region."
        )
    certification = result.certification
    if certification is None or certification.coverage is None:
        raise RuntimeError("Native certification omitted per-cell source region labels.")
    native_coverage = certification.coverage
    if native_coverage.status != "certified":
        raise RuntimeError("Native domain-coverage certification is absent or did not pass.")
    certified_regions = np.asarray(native_coverage.cell_regions, dtype=np.int64)
    if certified_regions.shape != independent_regions.shape:
        raise RuntimeError(
            "Native per-cell region labels do not align with the tetrahedral carrier: "
            f"{certified_regions.shape=} != {independent_regions.shape=}."
        )
    if not np.array_equal(certified_regions, independent_regions):
        mismatch = np.flatnonzero(certified_regions != independent_regions)
        raise RuntimeError(
            f"Native region labels differ from independent analytic classification "
            f"in {mismatch.size} tetrahedra."
        )
    expected = dict(profile.expected_region_volumes)
    measured = {
        region_id: float(np.sum(measures[independent_regions == region_index]))
        for region_index, region_id in enumerate(profile.region_ids)
    }
    relative_errors = {
        region_id: abs(measured[region_id] - float(expected[region_id]))
        / float(expected[region_id])
        for region_id in profile.region_ids
    }
    if any(error > _REGION_RELATIVE_TOLERANCE for error in relative_errors.values()):
        raise RuntimeError(
            "Independent exact region volume oracle failed: "
            f"expected={ {name: str(expected[name]) for name in profile.region_ids} }, "
            f"measured={measured}, relative_errors={relative_errors}."
        )
    if np.any(measures <= 0.0):
        raise RuntimeError(
            "Independent region oracle found a non-positive tetrahedron measure."
        )
    return {
        "analytic_region_volumes_exact": {
            region_id: str(expected[region_id]) for region_id in profile.region_ids
        },
        "analytic_region_volumes": {
            region_id: float(expected[region_id]) for region_id in profile.region_ids
        },
        "independently_measured_region_volumes": measured,
        "region_relative_errors": relative_errors,
        "tolerance_relative": _REGION_RELATIVE_TOLERANCE,
        "tetrahedra": int(cells.shape[0]),
        "minimum_signed_tetrahedron_measure": float(np.min(signed)),
        "native_coverage_certificate_id": native_coverage.certificate_id,
        "native_coverage_status": native_coverage.status,
        "per_cell_region_labels_match_independent_geometry": True,
    }


def _polygon_area(points: np.ndarray, /) -> float:
    boundary = np.roll(points, -1, axis=0)
    area_vector = np.sum(np.cross(points, boundary), axis=0)
    return 0.5 * float(np.linalg.norm(area_vector))


def _point_in_source_polygon(
    point: np.ndarray, polygon: np.ndarray, tolerance: float, /
) -> bool:
    boundary = np.roll(polygon, -1, axis=0)
    area_vector = np.sum(np.cross(polygon, boundary), axis=0)
    normal = area_vector / np.linalg.norm(area_vector)
    if abs(float(np.dot(point - polygon[0], normal))) > tolerance:
        return False
    projection_axis = int(np.argmax(np.abs(normal)))
    polygon_2d = np.delete(polygon, projection_axis, axis=1)
    point_2d = np.delete(point, projection_axis)
    inside = False
    for index, first in enumerate(polygon_2d):
        second = polygon_2d[(index + 1) % polygon_2d.shape[0]]
        edge = second - first
        relative = point_2d - first
        edge_length = float(np.linalg.norm(edge))
        cross = float(edge[0] * relative[1] - edge[1] * relative[0])
        projection = float(np.dot(relative, edge))
        if (
            abs(cross) <= tolerance * max(1.0, edge_length)
            and -tolerance <= projection <= edge_length * edge_length + tolerance
        ):
            return True
        if (first[1] > point_2d[1]) != (second[1] > point_2d[1]):
            crossing = first[0] + (
                (point_2d[1] - first[1]) * (second[0] - first[0]) / (second[1] - first[1])
            )
            if point_2d[0] < crossing:
                inside = not inside
    return inside


def _boundary_facet_oracle(
    source: NativePlcSource,
    source_input: _SolidInput,
    result: phx.meshing.CellMeshingResult,
    /,
) -> dict[str, object]:
    mesh = result.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise TypeError(
            "Native constrained-solid boundary coverage requires tetrahedral connectivity."
        )
    face_set = mesh.entity_set(2)
    face_associations = tuple(
        association
        for association in result.associations
        if association.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
        and association.target_entity_set_id == face_set.entity_set_id
        and GeometrySourceEntityRole.FACET in association.source_roles
    )
    if len(face_associations) != 1:
        raise RuntimeError(
            "Native result must publish one complete PLC facet association on all source faces; "
            f"observed {len(face_associations)}."
        )
    association = face_associations[0]
    if (
        association.source_id != source.source_id
        or association.source_revision != source.source_revision
        or not association.complete
        or not association.exact
    ):
        raise RuntimeError(
            "Native PLC facet association is incomplete, inexact, or bound to another source revision."
        )
    target_ids = np.asarray(association.target_global_ids, dtype=np.int64)
    source_ids = association.source_entity_ids
    roles = association.source_roles
    region_sides = association.source_region_sides
    if len(set(target_ids.tolist())) != target_ids.size:
        raise RuntimeError(
            "Native PLC facet association duplicates a target face identity."
        )
    target_rows_by_id = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(face_set.entity_ids, dtype=np.int64).tolist()
        )
    }
    face_vertices = np.asarray(connectivity.faces, dtype=np.int64)
    point_coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    certification = result.certification
    if certification is None or certification.coverage is None:
        raise RuntimeError(
            "Native certification omitted per-cell region labels for facet coverage."
        )
    coverage = certification.coverage
    if coverage.status != "certified":
        raise RuntimeError("Native domain-coverage certification did not pass facet coverage.")
    cell_regions = np.asarray(coverage.cell_regions, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int64).reshape((-1, 4))
    adjacent_regions: list[set[int]] = [set() for _ in range(face_vertices.shape[0])]
    for cell_row in range(cell_faces.shape[0]):
        for local_face in range(cell_faces.shape[1]):
            face_row = int(cell_faces[cell_row, local_face])
            adjacent_regions[face_row].add(int(cell_regions[cell_row]))
    expected_rows = {
        row
        for row, regions in enumerate(adjacent_regions)
        if bool(np.asarray(connectivity.boundary_faces, dtype=np.bool_)[row])
        or len(regions) > 1
    }
    expected_ids = {
        int(np.asarray(face_set.entity_ids, dtype=np.int64)[row]) for row in expected_rows
    }
    facet_rows_by_name = {name: row for row, name in enumerate(source.complex.facet_ids)}
    if any(role is not GeometrySourceEntityRole.FACET for role in roles):
        raise RuntimeError("A PLC facet association contains a non-facet source role.")
    if set(target_ids.tolist()) != expected_ids:
        missing = sorted(expected_ids - set(target_ids.tolist()))
        extra = sorted(set(target_ids.tolist()) - expected_ids)
        raise RuntimeError(
            f"Source boundary/interface coverage is not exact: {len(missing)} faces omitted, "
            f"{len(extra)} non-boundary faces added."
        )
    expected_facet_names = set(source.complex.facet_ids)
    if set(source_ids) != expected_facet_names:
        missing = sorted(expected_facet_names - set(source_ids))
        extra = sorted(set(source_ids) - expected_facet_names)
        raise RuntimeError(
            f"Source facet identity coverage is not exact: {missing=} {extra=}."
        )
    source_polygons = np.asarray(source_input.vertices, dtype=np.float64)
    polygon_owner = np.asarray(source_input.polygon_facets, dtype=np.int64)
    polygons_by_facet: dict[int, list[np.ndarray]] = {
        facet: [] for facet in range(len(source.complex.facet_ids))
    }
    expected_polygon_areas = np.zeros((len(source.complex.facet_ids),), dtype=np.float64)
    for polygon_row, polygon in enumerate(source_input.polygons):
        coordinates = source_polygons[np.asarray(polygon, dtype=np.int64)]
        facet = int(polygon_owner[polygon_row])
        polygons_by_facet[facet].append(coordinates)
        expected_polygon_areas[facet] += _polygon_area(coordinates)
    measured_polygon_areas = np.zeros_like(expected_polygon_areas)
    face_counts = np.zeros_like(expected_polygon_areas, dtype=np.int64)
    for target_id, source_id, sides in zip(
        target_ids.tolist(), source_ids, region_sides, strict=True
    ):
        target_row = target_rows_by_id[int(target_id)]
        facet_row = facet_rows_by_name[str(source_id)]
        region_pair = np.asarray(source_input.facet_regions[facet_row], dtype=np.int64)
        expected_sides = (
            None
            if int(region_pair[1]) < 0
            else source_input.region_ids[int(region_pair[1])],
            None
            if int(region_pair[0]) < 0
            else source_input.region_ids[int(region_pair[0])],
        )
        if sides != expected_sides:
            raise RuntimeError(
                f"Source region-side orientation changed on facet {source_id!r}: "
                f"{sides!r} != {expected_sides!r}."
            )
        triangle = point_coordinates[face_vertices[target_row]]
        triangle_area = 0.5 * float(
            np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        )
        scale = max(1.0, float(np.max(np.abs(triangle))))
        polygon_tolerance = 1.0e-10 * scale
        matching = [
            polygon
            for polygon in polygons_by_facet[facet_row]
            if all(
                _point_in_source_polygon(point, polygon, polygon_tolerance)
                for point in triangle
            )
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"Native face {int(target_id)} maps to {len(matching)} declared polygons "
                f"on source facet {source_id!r}, expected one."
            )
        measured_polygon_areas[facet_row] += triangle_area
        face_counts[facet_row] += 1
    relative_area_errors = (
        np.abs(measured_polygon_areas - expected_polygon_areas) / expected_polygon_areas
    )
    if np.any(face_counts == 0) or np.any(
        relative_area_errors > _FACET_AREA_RELATIVE_TOLERANCE
    ):
        raise RuntimeError(
            "Exact source-facet area coverage failed: "
            f"source_areas={expected_polygon_areas.tolist()}, "
            f"measured_areas={measured_polygon_areas.tolist()}, "
            f"relative_errors={relative_area_errors.tolist()}."
        )
    fixed_boundary = None
    if source_input.boundary == "fixed":
        fixed_rows = np.flatnonzero(
            np.asarray(connectivity.boundary_faces, dtype=np.bool_)
        )
        published_triangles = {
            tuple(
                sorted(
                    tuple(float(value) for value in point)
                    for point in point_coordinates[face_vertices[row]]
                )
            )
            for row in fixed_rows.tolist()
        }
        declared_triangles = {
            tuple(
                sorted(
                    tuple(float(value) for value in source_input.vertices[index])
                    for index in polygon
                )
            )
            for polygon in source_input.polygons
        }
        if published_triangles != declared_triangles:
            raise RuntimeError(
                "Fixed-boundary PLC triangles were subdivided, moved, omitted, or duplicated."
            )
        fixed_boundary = {
            "source_triangles": len(declared_triangles),
            "published_triangles": len(published_triangles),
            "unchanged": True,
        }
    return {
        "association_id": association.association_id,
        "association_exact": association.exact,
        "association_complete": association.complete,
        "source_facets_expected": len(expected_facet_names),
        "source_facets_covered": len(set(source_ids)),
        "declared_boundary_and_interface_faces": len(expected_ids),
        "covered_boundary_and_interface_faces": int(target_ids.size),
        "omitted_or_duplicated_boundary_interface_faces": 0,
        "source_facet_area_expected": {
            source.complex.facet_ids[row]: float(expected_polygon_areas[row])
            for row in range(expected_polygon_areas.size)
        },
        "source_facet_area_measured": {
            source.complex.facet_ids[row]: float(measured_polygon_areas[row])
            for row in range(measured_polygon_areas.size)
        },
        "source_facet_area_relative_errors": {
            source.complex.facet_ids[row]: float(relative_area_errors[row])
            for row in range(relative_area_errors.size)
        },
        "facet_area_relative_tolerance": _FACET_AREA_RELATIVE_TOLERANCE,
        "fixed_boundary_triangles": fixed_boundary,
        "covered_target_face_ids": [int(value) for value in target_ids.tolist()],
        "covered_source_facet_ids": list(source_ids),
    }


def _material_control_oracle(
    profile: _SolidProfile,
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    result: phx.meshing.CellMeshingResult,
    /,
) -> dict[str, object]:
    mesh = result.mesh
    certification = result.certification
    if certification is None or certification.coverage is None:
        raise RuntimeError(
            "Native certification omitted region assignments for material checks."
        )
    coverage = certification.coverage
    if coverage.status != "certified":
        raise RuntimeError("Native domain-coverage certification did not pass material checks.")
    cell_regions = np.asarray(coverage.cell_regions, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    if cell_ids.shape != cell_regions.shape:
        raise RuntimeError(
            "Native cell identities and certified region labels are misaligned."
        )
    region_controls = {
        control.region_name: control for control in specification.region_controls
    }
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    if set(region_controls) != set(profile.region_ids) or set(zones) != set(
        profile.region_ids
    ):
        raise RuntimeError(
            f"Material region controls/zones differ from the exact source region list: "
            f"controls={sorted(region_controls)}, zones={sorted(zones)}, expected={sorted(profile.region_ids)}."
        )
    for region_index, region_id in enumerate(profile.region_ids):
        control = region_controls[region_id]
        zone = zones[region_id]
        selected_ids = np.sort(cell_ids[cell_regions == region_index])
        if (
            control.material_id != f"{region_id}-material"
            or control.role is not RegionRole.SOLID
            or zone.material_id != f"{region_id}-material"
            or zone.region_role is not RegionRole.SOLID
            or not np.array_equal(
                np.sort(np.asarray(zone.scope.entity_ids, dtype=np.int64)), selected_ids
            )
        ):
            raise RuntimeError(
                f"Region/material assignment was not preserved for {region_id!r}."
            )
    evidence = {item.source_region_id: item for item in result.region_boundary_evidence}
    if set(evidence) != set(profile.region_ids):
        raise RuntimeError(
            "Native result omitted a material region boundary evidence record."
        )
    evidence_ids: dict[str, str] = {}
    for region_id in profile.region_ids:
        item = evidence[region_id]
        control = region_controls[region_id]
        if (
            item.control_id != control.control_id
            or item.material_id != f"{region_id}-material"
            or item.role is not RegionRole.SOLID
            or item.source_scope.source_id != source.source_id
            or item.source_scope.source_revision != source.source_revision
        ):
            raise RuntimeError(
                f"Region boundary evidence lost the explicit {region_id!r} control."
            )
        item.require_current(mesh, result.labels, result.patches)
        evidence_ids[region_id] = item.evidence_id
    expected_patch_names = {control.name for control in specification.patch_controls}
    patches = {patch.name: patch for patch in result.patches}
    if not expected_patch_names <= set(patches):
        raise RuntimeError(
            f"Native result omitted declared material-interface patches: "
            f"{sorted(expected_patch_names - set(patches))}."
        )
    for control in specification.patch_controls:
        patch = patches[control.name]
        if patch.source_adjacent_region_ids is None or set(
            patch.source_adjacent_region_ids
        ) != set(control.adjacent_region_names):
            raise RuntimeError(
                f"Material-interface patch {control.name!r} changed its declared region sides."
            )
    roles: dict[str, RegionRole] = {}
    for region_id in profile.region_ids:
        role = zones[region_id].region_role
        if role is None:
            raise RuntimeError(
                f"Native zone {region_id!r} omitted its physical region role."
            )
        roles[region_id] = role
    return {
        "regions": {
            region_id: {
                "material_id": zones[region_id].material_id,
                "role": roles[region_id].value,
                "region_control_id": region_controls[region_id].control_id,
                "region_boundary_evidence_id": evidence_ids[region_id],
                "cells": int(np.count_nonzero(cell_regions == region_index)),
            }
            for region_index, region_id in enumerate(profile.region_ids)
        },
        "interface_patches": [
            {
                "name": patch.name,
                "adjacent_regions": list(patch.source_adjacent_region_ids or ()),
                "patch_id": patch.patch_id,
            }
            for patch in result.patches
            if patch.name in expected_patch_names
        ],
        "controls_and_evidence_preserved": True,
    }


def _solve_fe(result: phx.meshing.CellMeshingResult, /) -> dict[str, object]:
    mesh = result.mesh
    field = FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 1))
    space = FiniteElementPlan(mesh, field, coordinate_spec=result.geometry).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    form = phx.equations.FiniteElementForm(
        "native-constrained-solid-harmonic",
        "u",
        (phx.equations.DiffusionAction("u", 1.0),),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=lambda points: (
            1.0 + 2.0 * points[..., 0] - points[..., 1] + 0.5 * points[..., 2]
        ),
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    exact = 1.0 + 2.0 * coordinates[:, 0] - coordinates[:, 1] + 0.5 * coordinates[:, 2]
    error = float(np.max(np.abs(values - exact), initial=0.0))
    residual = float(jnp.linalg.norm(problem.residual(solved.value)))
    rhs_norm = float(jnp.linalg.norm(rhs))
    normalized_residual = residual / max(1.0, rhs_norm)
    if (
        not bool(jnp.all(solved.successful))
        or values.shape != exact.shape
        or error > _FE_TOLERANCE
        or normalized_residual > _FE_TOLERANCE
    ):
        raise RuntimeError(
            "Native constrained-solid FE solve failed its manufactured affine solution/residual "
            f"oracle: maximum_error={error}, normalized_residual={normalized_residual}, "
            f"tolerance={_FE_TOLERANCE}."
        )
    return {
        "pde": "steady homogeneous diffusion with exact affine Dirichlet data",
        "form_id": form.form_id,
        "field_name": field.name,
        "dofs": int(values.size),
        "successful": bool(jnp.all(solved.successful)),
        "maximum_affine_error": error,
        "residual_norm": residual,
        "normalized_residual": normalized_residual,
        "exact_field": "u=1+2x-y+0.5z",
        "maximum_affine_error_tolerance": _FE_TOLERANCE,
        "normalized_residual_tolerance": _FE_TOLERANCE,
    }


def _record_failure(
    failures: list[dict[str, str]], stage: str, error: Exception, /
) -> None:
    failures.append(
        {
            "stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    )


def _attempt(
    profile_record: dict[str, object],
    failures: list[dict[str, str]],
    stage: str,
    action: Callable[[], dict[str, object]],
    /,
) -> None:
    try:
        profile_record[stage] = action()
    except Exception as error:
        _record_failure(failures, stage, error)


def native_constrained_solids() -> dict[str, object]:
    profile_records: list[dict[str, object]] = []
    for profile in _PROFILES:
        failures: list[dict[str, str]] = []
        record: dict[str, object] = {
            "profile_id": profile.profile_id,
            "source_case_id": profile.source_case_id,
            "positive_intent": True,
            "expected_region_volumes_exact": {
                name: str(value) for name, value in profile.expected_region_volumes
            },
            "tolerances": {
                "region_measure_relative": _REGION_RELATIVE_TOLERANCE,
                "boundary_facet_area_relative": _FACET_AREA_RELATIVE_TOLERANCE,
                "boundary_coverage": "all declared source facets, no omitted or duplicated boundary/interface",
                "fe_normalized_residual": _FE_TOLERANCE,
                "fe_affine_solution_maximum_error": _FE_TOLERANCE,
            },
            "provenance": {
                "kind": "PHYDRA-authored deterministic constrained PLC",
                "source_branch": "feature/native-meshing-20260929",
                "source_commit": "c56614a9fe52c4a8f56484874e91fde6498ec9ff",
                "source_construction": "tools/_meshing_cases.py::source_payload and declared_boxes; tools/meshing_qualification/scenarios/native_lifecycle.py::_corpus_domain",
                "campaign_profile": profile.profile_id,
                "campaign_resolution_scale": profile.resolution_scale,
                "campaign_scenario_controls": {},
                "declared_profile_controls": list(profile.controls),
            },
        }
        profile_records.append(record)
        try:
            source_input = _input_for(profile)
            source = _source_for(profile, source_input)
            specification, options, controls = _request(profile, source, source_input)
            record["input"] = _input_record(profile, source, source_input, controls)
            record["provenance"] = {
                **record["provenance"],
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "complex_id": source.complex.complex_id,
                "source_vertex_ids": list(source.complex.vertex_ids),
                "source_facet_ids": list(source.complex.facet_ids),
                "source_segment_ids": list(source.complex.segment_ids),
                "source_region_ids": list(source.complex.region_ids),
            }
        except Exception as error:
            _record_failure(failures, "source_and_explicit_request", error)
            record["failures"] = failures
            record["status"] = "failed"
            continue
        record["route"] = "phydrax-native:plc_tetrahedral"
        result: phx.meshing.CellMeshingResult | None = None
        try:
            provider = NativeMeshingProvider(options)
            plan = provider.plan(source, specification, coordinate_contract=_CONTRACT)
            record["planned_route"] = plan.route
            result = plan.execute()
            record["provider_execution"] = {
                "route": plan.route,
                "result_id": result.result_id,
                "topology_id": result.mesh.topology_id,
            }
        except Exception as error:
            _record_failure(failures, "native_provider_plan_and_execution", error)
        if result is not None:
            _attempt(
                record,
                failures,
                "native_audit_certification_compliance",
                lambda: accepted_result_record(result),
            )
            _attempt(
                record,
                failures,
                "independent_exact_region_volume_and_coverage_oracle",
                lambda: _independent_region_oracle(profile, source_input, result),
            )
            _attempt(
                record,
                failures,
                "independent_source_facet_boundary_and_interface_oracle",
                lambda: _boundary_facet_oracle(source, source_input, result),
            )
            _attempt(
                record,
                failures,
                "explicit_region_material_and_interface_controls",
                lambda: _material_control_oracle(profile, source, specification, result),
            )
            _attempt(
                record,
                failures,
                "finite_element_physical_solve",
                lambda: _solve_fe(result),
            )
        record["failures"] = failures
        record["status"] = "passed" if not failures else "failed"
    all_passed = len(profile_records) == len(_PROFILES) and all(
        record["status"] == "passed" for record in profile_records
    )
    return {
        "status": "passed" if all_passed else "failed",
        "workflow": "nine authored constrained-solid PLC profiles -> native tetrahedral construction -> independent analytic/material/facet acceptance -> manufactured FE solve",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#constrained-solid",
        "profile_order": [profile.profile_id for profile in _PROFILES],
        "profile_count": len(profile_records),
        "positive_profile_count": len(_PROFILES),
        "profiles": profile_records,
        "runtime": runtime_identity(),
    }
