from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from phydrax._meshcore import (
    MeshcoreStatus,
    PlcRecovery3D,
    prepare_plc_source,
)
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.meshing._association import (
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._certification_inputs import _prepare_region_assignments
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import (
    FeatureKind,
    ProtectedFeature,
    RegionControl,
    RegionSeed,
)
from phydrax.meshing._native_construction import NativeMeshConstruction
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex, PlcBoundaryPolicy
from phydrax.meshing._result import _validated_source_evidence
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing._volume_generation import (
    _source_face_ids,
    _source_tables,
    generate_plc_volume,
    NativeVolumeSchedule,
    VolumeConstruction,
)
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import NativePlcSource
from phydrax.meshing.providers._native_volume import (
    _organization,
    prepare_native_plc_volume,
)


pytestmark = pytest.mark.meshcore


_POINTS = np.asarray(
    (
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ),
    dtype=np.float64,
)

_POLYGONS = np.asarray(
    (
        (1, 2, 3),
        (0, 3, 2),
        (0, 1, 3),
        (2, 1, 4),
        (0, 4, 1),
        (0, 2, 4),
        (0, 1, 2),
    ),
    dtype=np.int64,
)

_FACET_REGIONS = np.asarray(
    (
        (-1, 0),
        (-1, 0),
        (-1, 0),
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (0, 1),
    ),
    dtype=np.int64,
)


def _source(boundary: PlcBoundaryPolicy = "fixed") -> NativePlcSource:
    complex_ = PiecewiseLinearComplex(
        _POINTS,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"vertex-{row}" for row in range(_POINTS.shape[0])),
        vertex_occurrence_paths=((),) * _POINTS.shape[0],
        facet_ids=tuple(f"facet-{row}" for row in range(_POLYGONS.shape[0])),
        facet_occurrence_paths=((),) * _POLYGONS.shape[0],
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
        boundary=boundary,
    )
    return NativePlcSource(complex_, "bipyramid", "revision-1")


def _grouped_cube_source(
    boundary: PlcBoundaryPolicy = "fixed", angle: float | None = None
) -> NativePlcSource:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    if angle is not None:
        # A cube of side 1/3 centred at the origin, rotated so that the two
        # authored triangles of a face are not exactly coplanar for angle != 0.
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.asarray(((c, -s, 0.0), (s * c, c * c, -s), (s * s, s * c, c)))
        points = (points - 0.5) @ rotation.T / 3.0
    triangles = np.asarray(
        (
            (0, 2, 3),
            (0, 3, 1),
            (4, 5, 7),
            (4, 7, 6),
            (0, 1, 5),
            (0, 5, 4),
            (2, 6, 7),
            (2, 7, 3),
            (0, 4, 6),
            (0, 6, 2),
            (1, 3, 7),
            (1, 7, 5),
        ),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        points,
        triangles,
        np.repeat(np.arange(6, dtype=np.int64), 2),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (6, 1)),
        ("solid",),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"cube-vertex-{row}" for row in range(points.shape[0])),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=tuple(f"cube-facet-{row}" for row in range(6)),
        facet_occurrence_paths=((),) * 6,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
        boundary=boundary,
    )
    return NativePlcSource(complex_, "grouped-cube", "revision-1")


def _repeated_region_source() -> NativePlcSource:
    base = _source()
    complex_ = PiecewiseLinearComplex(
        _POINTS,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("shared-region", "shared-region"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=base.complex.vertex_ids,
        vertex_occurrence_paths=base.complex.vertex_occurrence_paths,
        facet_ids=base.complex.facet_ids,
        facet_occurrence_paths=base.complex.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=(("assembly-a",), ("assembly-b",)),
        boundary=base.complex.boundary,
    )
    return NativePlcSource(complex_, "repeated-bipyramid", "revision-1")


def _request(
    source: NativePlcSource,
    /,
    *,
    region_controls: tuple[RegionControl, ...] = (),
    region_seeds: tuple[RegionSeed, ...] | None = None,
) -> VolumeMeshingSpec:
    boundary = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    seeds = (
        (
            RegionSeed(
                np.asarray((0.0, 0.25, 0.25), dtype=np.float64),
                "upper",
                "upper-material",
                RegionRole.SOLID,
            ),
            RegionSeed(
                np.asarray((0.0, 0.25, -0.25), dtype=np.float64),
                "lower",
                "lower-material",
                RegionRole.SOLID,
            ),
        )
        if region_seeds is None
        else region_seeds
    )
    return VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        region_controls=region_controls,
        region_seeds=seeds,
    )


def _schedule() -> NativeVolumeSchedule:
    return NativeVolumeSchedule(
        radius_edge_bound=8.0,
        minimum_dihedral_degrees=0.0,
    )


def _construct(
    source: NativePlcSource, specification: VolumeMeshingSpec
) -> VolumeConstruction:
    return generate_plc_volume(
        source.complex,
        specification,
        _schedule(),
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_map_id="",
        frame_id=SpatialCoordinateContract.si().spatial_id,
        source_domain=source.mapped_domain,
    )


def _prepare(
    source: NativePlcSource, specification: VolumeMeshingSpec
) -> NativeMeshConstruction:
    options = NativeMeshingOptions("plc_tetrahedral", volume_schedule=_schedule())
    return prepare_native_plc_volume(
        source,
        specification,
        options,
        SpatialCoordinateContract.si(),
    ).execute()


def test_plc_ids_and_occurrences_are_authored_and_not_coordinate_derived() -> None:
    constructor = cast(Callable[..., PiecewiseLinearComplex], PiecewiseLinearComplex)
    with pytest.raises(TypeError, match="vertex_occurrence_paths"):
        constructor(
            _POINTS,
            _POLYGONS,
            np.arange(_POLYGONS.shape[0], dtype=np.int64),
            _FACET_REGIONS,
            ("upper", "lower"),
            segments=np.empty((0, 2), dtype=np.int64),
            vertex_ids=tuple(f"vertex-{row}" for row in range(_POINTS.shape[0])),
            facet_ids=tuple(f"facet-{row}" for row in range(_POLYGONS.shape[0])),
            facet_occurrence_paths=((),) * _POLYGONS.shape[0],
            segment_ids=(),
            segment_occurrence_paths=(),
            region_occurrence_paths=((), ()),
        )

    source = _source()
    moved = np.asarray(_POINTS, dtype=np.float64).copy()
    moved[3, 2] = 1.25
    changed_geometry = PiecewiseLinearComplex(
        moved,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=source.complex.vertex_ids,
        vertex_occurrence_paths=source.complex.vertex_occurrence_paths,
        facet_ids=source.complex.facet_ids,
        facet_occurrence_paths=source.complex.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=source.complex.region_occurrence_paths,
    )
    assert changed_geometry.vertex_ids == source.complex.vertex_ids
    assert changed_geometry.vertex_entity_ids == source.complex.vertex_entity_ids
    assert changed_geometry.complex_id != source.complex.complex_id

    permutation = np.asarray((2, 0, 4, 1, 3), dtype=np.int64)
    old_to_new = np.empty_like(permutation)
    old_to_new[permutation] = np.arange(permutation.size, dtype=np.int64)
    reordered = PiecewiseLinearComplex(
        _POINTS[permutation],
        old_to_new[_POLYGONS],
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(source.complex.vertex_ids[int(row)] for row in permutation),
        vertex_occurrence_paths=tuple(
            source.complex.vertex_occurrence_paths[int(row)] for row in permutation
        ),
        facet_ids=source.complex.facet_ids,
        facet_occurrence_paths=source.complex.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=source.complex.region_occurrence_paths,
    )
    original_vertex_ids = dict(
        zip(source.complex.vertex_ids, source.complex.vertex_entity_ids, strict=True)
    )
    reordered_vertex_ids = dict(
        zip(reordered.vertex_ids, reordered.vertex_entity_ids, strict=True)
    )
    assert original_vertex_ids == reordered_vertex_ids
    original_edge_ids = dict(
        zip(source.complex.plc_edge_ids, source.complex.plc_edge_entity_ids, strict=True)
    )
    reordered_edge_ids = dict(
        zip(reordered.plc_edge_ids, reordered.plc_edge_entity_ids, strict=True)
    )
    assert original_edge_ids == reordered_edge_ids

    repeated = PiecewiseLinearComplex(
        _POINTS,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=("shared", "shared", "vertex-2", "vertex-3", "vertex-4"),
        vertex_occurrence_paths=(
            ("assembly-a",),
            ("assembly-b",),
            (),
            (),
            (),
        ),
        facet_ids=source.complex.facet_ids,
        facet_occurrence_paths=source.complex.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=source.complex.region_occurrence_paths,
    )
    assert repeated.vertex_entity_ids[0] != repeated.vertex_entity_ids[1]
    repeated_face_ids = _source_face_ids(
        np.asarray(((0, 2, 3), (1, 2, 3)), dtype=np.int64),
        np.zeros((2,), dtype=np.int64),
        repeated.vertex_ids,
        repeated.vertex_occurrence_paths,
        repeated.facet_ids,
        repeated.facet_occurrence_paths,
    )
    assert len(set(repeated_face_ids)) == 2


def test_shared_polygon_loop_seam_is_a_stable_plc_edge() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    loops = (
        np.asarray((0, 1, 2), dtype=np.int64),
        np.asarray((0, 2, 3), dtype=np.int64),
    )

    def make_complex(polygons: tuple[np.ndarray, ...], /) -> PiecewiseLinearComplex:
        return PiecewiseLinearComplex(
            points,
            polygons,
            np.zeros((2,), dtype=np.int64),
            np.asarray(((-1, 0),), dtype=np.int64),
            ("solid",),
            segments=np.empty((0, 2), dtype=np.int64),
            vertex_ids=("v0", "v1", "v2", "v3"),
            vertex_occurrence_paths=((),) * 4,
            facet_ids=("one-quad-facet",),
            facet_occurrence_paths=((),),
            segment_ids=(),
            segment_occurrence_paths=(),
            region_occurrence_paths=((),),
        )

    complex_ = make_complex(loops)
    reordered = make_complex(tuple(reversed(loops)))
    rows = {
        tuple(sorted(edge)): row for row, edge in enumerate(complex_.plc_edge_vertices)
    }
    assert set(rows) == {(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)}
    seam = rows[(0, 2)]
    assert complex_.plc_edge_facets[seam] == (0,)
    reordered_rows = {
        tuple(sorted(edge)): row for row, edge in enumerate(reordered.plc_edge_vertices)
    }
    assert complex_.plc_edge_ids[seam] == reordered.plc_edge_ids[reordered_rows[(0, 2)]]
    assert (
        complex_.plc_edge_entity_ids[seam]
        == reordered.plc_edge_entity_ids[reordered_rows[(0, 2)]]
    )


def test_grouped_coplanar_facet_diagonals_are_facet_owned_unless_requested() -> None:
    source = _grouped_cube_source()
    seed = RegionSeed(
        np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
        "solid",
        "solid-material",
        RegionRole.SOLID,
    )
    construction = _construct(source, _request(source, region_seeds=(seed,)))
    complex_ = source.complex
    boundary_rows = {
        row for row, facets in enumerate(complex_.plc_edge_facets) if len(facets) == 2
    }
    diagonal_rows = set(range(len(complex_.plc_edge_ids))) - boundary_rows
    assert len(boundary_rows) == 12
    assert len(diagonal_rows) == 6
    boundary_ids = {complex_.plc_edge_ids[row] for row in boundary_rows}
    diagonal_ids = {complex_.plc_edge_ids[row] for row in diagonal_rows}
    assert set(construction.segment_source_ids) == boundary_ids
    assert not set(construction.segment_source_ids) & diagonal_ids
    assert set(construction.face_source_ids) == set(complex_.facet_ids)
    assert construction.region_ids == ("solid",)
    np.testing.assert_array_equal(
        construction.cell_regions, np.zeros_like(construction.cell_regions)
    )

    requested_diagonal_row = min(diagonal_rows)
    feature_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        1,
        source.scope_entity_set_id(1),
        np.asarray(
            (complex_.plc_edge_entity_ids[requested_diagonal_row],), dtype=np.int64
        ),
    )
    requested_feature = ProtectedFeature(feature_scope, FeatureKind.CURVE)
    base_specification = _request(source, region_seeds=(seed,))
    feature_specification = VolumeMeshingSpec(
        base_specification.target,
        base_specification.boundary_scope,
        base_specification.fill_strategy,
        size_controls=base_specification.size_controls,
        protected_features=(requested_feature,),
        region_seeds=base_specification.region_seeds,
    )
    # A requested diagonal is an edge of the authored face triangulation, so
    # even a fixed boundary recovers it without boundary Steiner points.
    requested = _construct(source, feature_specification)
    requested_id = complex_.plc_edge_ids[requested_diagonal_row]
    assert set(requested.segment_source_ids) == boundary_ids | {requested_id}
    assert set(requested.face_source_ids) == set(complex_.facet_ids)
    assert np.all(np.asarray(requested.boundary_faces) < complex_.vertices.shape[0])


def _declared_rotated_cube(
    facet_planes: np.ndarray | None, facet_plane_tolerances: np.ndarray | None = None
) -> NativePlcSource:
    base = _grouped_cube_source(angle=0.3).complex
    loops = np.asarray(base.polygon_vertices).reshape(-1, 3)
    complex_ = PiecewiseLinearComplex(
        base.vertices,
        loops,
        base.polygon_facets,
        base.facet_regions,
        base.region_ids,
        segments=base.segments,
        vertex_ids=base.vertex_ids,
        vertex_occurrence_paths=base.vertex_occurrence_paths,
        facet_ids=base.facet_ids,
        facet_occurrence_paths=base.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=base.region_occurrence_paths,
        boundary="fixed",
        facet_planes=facet_planes,
        facet_plane_tolerances=facet_plane_tolerances,
    )
    return NativePlcSource(complex_, "declared-cube", "revision-1")


def _rotated_cube_planes() -> np.ndarray:
    points = np.asarray(_grouped_cube_source(angle=0.3).complex.vertices)
    corners = ((0, 2, 1), (4, 5, 6), (0, 1, 4), (2, 6, 3), (0, 4, 2), (1, 3, 5))
    return np.asarray(
        [(points[a], points[b] - points[a], points[c] - points[a]) for a, b, c in corners]
    )


def test_declared_facet_planes_group_rounded_cube_faces_without_diagonal_segments() -> (
    None
):
    seed = RegionSeed(
        np.zeros((3,), dtype=np.float64), "solid", "solid-material", RegionRole.SOLID
    )
    undeclared = _declared_rotated_cube(None)
    declared = _declared_rotated_cube(_rotated_cube_planes(), np.full((6,), 1e-12))
    # The rotated faces' two triangles are not exactly coplanar: undeclared,
    # each face diagonal is a PLC edge and a recovered segment source.
    assert undeclared.complex.facet_planes is None
    assert (
        undeclared.complex.complex_id
        == _grouped_cube_source(angle=0.3).complex.complex_id
    )
    assert declared.complex.complex_id != undeclared.complex.complex_id
    loose = _construct(undeclared, _request(undeclared, region_seeds=(seed,)))
    grouped = _construct(declared, _request(declared, region_seeds=(seed,)))
    complex_ = declared.complex
    boundary_ids = {
        complex_.plc_edge_ids[row]
        for row, facets in enumerate(complex_.plc_edge_facets)
        if len(facets) == 2
    }
    assert len(boundary_ids) == 12
    assert len(set(loose.segment_source_ids)) == 18
    assert set(grouped.segment_source_ids) == boundary_ids
    assert set(grouped.face_source_ids) == set(complex_.facet_ids)


def test_declared_facet_planes_are_validated_rows() -> None:
    planes = _rotated_cube_planes()
    with pytest.raises(ValueError, match="facet_plane_tolerances require"):
        _declared_rotated_cube(None, np.zeros((6,)))
    partial = planes.copy()
    partial[2, 1, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        _declared_rotated_cube(partial)
    dependent = planes.copy()
    dependent[2, 2] = 2.0 * dependent[2, 1]
    with pytest.raises(ValueError, match="exactly independent"):
        _declared_rotated_cube(dependent)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _declared_rotated_cube(planes, np.full((6,), -1.0))
    undeclared_rows = np.full((6, 3, 3), np.nan)
    accepted = _declared_rotated_cube(undeclared_rows)
    np.testing.assert_array_equal(accepted.complex.facet_plane_tolerances, np.zeros((6,)))


def test_raw_recovery_witnesses_map_to_canonical_source_identity_rows() -> None:
    base = _source("conforming")
    complex_ = PiecewiseLinearComplex(
        base.complex.vertices,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        base.complex.region_ids,
        segments=np.asarray((base.complex.plc_edge_vertices[0],), dtype=np.int64),
        vertex_ids=base.complex.vertex_ids,
        vertex_occurrence_paths=base.complex.vertex_occurrence_paths,
        facet_ids=base.complex.facet_ids,
        facet_occurrence_paths=base.complex.facet_occurrence_paths,
        segment_ids=("explicit-loop-edge",),
        segment_occurrence_paths=(base.complex.plc_edge_occurrence_paths[0],),
        region_occurrence_paths=base.complex.region_occurrence_paths,
        boundary="conforming",
    )
    source = NativePlcSource(complex_, "bipyramid-with-explicit-edge", "revision-1")
    prepared = prepare_plc_source(
        complex_.vertices,
        complex_.polygon_offsets,
        complex_.polygon_vertices,
        complex_.polygon_facets,
        complex_.facet_regions,
        complex_.segments,
    )
    if prepared.status is not MeshcoreStatus.OK:
        raise AssertionError("Native PLC source preparation failed.")
    canonical_edge = tuple(int(value) for value in complex_.plc_edge_vertices[0])
    raw_edges = np.asarray(prepared.plc_edges, dtype=np.int64)
    raw_edge_rows = [
        row
        for row, edge in enumerate(raw_edges.tolist())
        if tuple(sorted(int(value) for value in edge)) == tuple(sorted(canonical_edge))
    ]
    assert len(raw_edge_rows) == 1
    raw_edge_row = raw_edge_rows[0]
    raw_edge = tuple(int(value) for value in raw_edges[raw_edge_row])
    segment_point = 0.75 * np.asarray(
        complex_.vertices[raw_edge[0]], dtype=np.float64
    ) + 0.25 * np.asarray(complex_.vertices[raw_edge[1]], dtype=np.float64)
    source_triangle = np.asarray(prepared.input_triangles[0], dtype=np.int64)
    face_point = (
        np.asarray(complex_.vertices[source_triangle[0]], dtype=np.float64)
        + 0.2
        * (
            np.asarray(complex_.vertices[source_triangle[1]], dtype=np.float64)
            - np.asarray(complex_.vertices[source_triangle[0]], dtype=np.float64)
        )
        + 0.3
        * (
            np.asarray(complex_.vertices[source_triangle[2]], dtype=np.float64)
            - np.asarray(complex_.vertices[source_triangle[0]], dtype=np.float64)
        )
    )
    recovery = PlcRecovery3D(
        prepared.status,
        np.vstack((prepared.points, segment_point, face_point)),
        prepared.tetrahedra,
        prepared.tetrahedron_regions,
        prepared.faces,
        prepared.face_sources,
        prepared.face_orientations,
        prepared.segments,
        prepared.segment_sources,
        np.concatenate((prepared.protection, np.zeros((2,), dtype=np.float64))),
        np.concatenate((prepared.vertex_dimensions, np.asarray((1, 2), dtype=np.int8))),
        prepared.plc_edges,
        prepared.input_triangles,
        prepared.input_polygons,
        np.concatenate((prepared.source_strata, np.asarray((1, 2), dtype=np.int8))),
        np.concatenate(
            (
                prepared.source_entities,
                np.asarray((raw_edge_row, 0), dtype=np.int32),
            )
        ),
        np.concatenate(
            (
                prepared.source_parameters,
                np.asarray(((0.25, 0.0), (0.2, 0.3)), dtype=np.float64),
            )
        ),
        np.concatenate((prepared.source_deviations, np.zeros((2,), dtype=np.float64))),
        prepared.suppression_search,
    )
    canonical_edges = np.asarray(complex_.plc_edge_vertices, dtype=np.int64)
    vertex_identity_rows = tuple(
        zip(complex_.vertex_ids, complex_.vertex_occurrence_paths, strict=True)
    )
    canonical_row_by_endpoints = {
        tuple(sorted(vertex_identity_rows[int(vertex)] for vertex in edge)): row
        for row, edge in enumerate(canonical_edges.tolist())
    }
    raw_edge_complex_rows = np.asarray(
        [
            canonical_row_by_endpoints[
                tuple(sorted(vertex_identity_rows[int(vertex)] for vertex in edge))
            ]
            for edge in recovery.plc_edges.tolist()
        ],
        dtype=np.int64,
    )
    expected_rows = np.unique(raw_edge_complex_rows)
    expected_local_row_by_complex = {
        int(complex_row): local_row
        for local_row, complex_row in enumerate(expected_rows.tolist())
    }
    expected_carrier_source_rows = np.asarray(
        [
            expected_local_row_by_complex[int(complex_row)]
            for complex_row in raw_edge_complex_rows.tolist()
        ],
        dtype=np.int64,
    )
    tables = _source_tables(complex_, _request(source), recovery)
    np.testing.assert_array_equal(tables.source_segment_complex_rows, expected_rows)
    np.testing.assert_array_equal(
        tables.source_segment_vertices, canonical_edges[expected_rows]
    )
    np.testing.assert_array_equal(
        tables.plc_edge_source_rows, expected_carrier_source_rows
    )
    assert tables.segment_entity_ids == tuple(
        complex_.plc_edge_ids[int(row)] for row in expected_rows
    )
    assert tables.segment_occurrence_paths == tuple(
        complex_.plc_edge_occurrence_paths[int(row)] for row in expected_rows
    )

    expected_segment_parameter = 0.25 if raw_edge == canonical_edge else 0.75
    assert int(tables.source.witness_entities[-2]) == 0
    assert tables.source.witness_parameters[-2, 0] == pytest.approx(
        expected_segment_parameter
    )
    assert int(tables.source.witness_entities[-1]) == 0
    np.testing.assert_array_equal(
        tables.source.witness_parameters[-1],
        np.asarray((0.2, 0.3), dtype=np.float64),
    )
    assert tables.segment_entity_ids[0] == complex_.plc_edge_ids[0]
    assert tables.segment_occurrence_paths[0] == complex_.plc_edge_occurrence_paths[0]


def test_native_plc_fill_recovers_material_interface_and_fixed_boundary() -> None:
    source = _source("fixed")
    construction = _construct(source, _request(source))
    assert construction.boundary_policy == "fixed"
    assert isinstance(construction.domain, PiecewiseLinearDomain)
    assert construction.region_ids == ("upper", "lower")
    assert (
        construction.domain.region_occurrence_paths
        == source.complex.region_occurrence_paths
    )
    assert set(np.asarray(construction.cell_regions, dtype=np.int64).tolist()) == {0, 1}
    assert construction.input_vertices.shape == (construction.mesh.coordinates.shape[0],)
    boundary_vertex_rows = np.unique(construction.boundary_faces)
    assert np.all(construction.input_vertices[boundary_vertex_rows] >= 0)
    for face, facet in zip(
        construction.boundary_faces,
        construction.face_source_facets,
        strict=True,
    ):
        source_vertices = construction.input_vertices[np.asarray(face, dtype=np.int64)]
        assert np.all(source_vertices >= 0)
        assert tuple(sorted(source_vertices.tolist())) == tuple(
            sorted(_POLYGONS[int(facet)].tolist())
        )

    interface_facets = np.flatnonzero(np.all(_FACET_REGIONS >= 0, axis=1))
    assert interface_facets.size == 1
    interface_rows = np.flatnonzero(
        np.isin(construction.face_source_facets, interface_facets)
    )
    assert interface_rows.size == 1
    assert (
        construction.face_source_ids[int(interface_rows[0])]
        == source.complex.facet_ids[int(interface_facets[0])]
    )

    result = _prepare(source, _request(source))

    region_evidence = result.organization.region_boundary_evidence
    assert len(region_evidence) == 2
    evidence_by_region = {
        evidence.source_region_id: evidence for evidence in region_evidence
    }
    assert set(evidence_by_region) == {"upper", "lower"}
    for region_id, evidence in evidence_by_region.items():
        assert evidence.control_id is None
        assert evidence.material_id == f"{region_id}-material"
        assert evidence.role is RegionRole.SOLID
        assert evidence.source_scope.source_id == source.source_id
        assert evidence.source_scope.source_revision == source.source_revision
        assert evidence.source_scope.entity_set_id == source.scope_entity_set_id(3)
        assert source.source_entity_key_for_scope(
            3, int(evidence.source_scope.entity_ids[0])
        ) == (evidence.source_region_id, evidence.source_occurrence_path)
        assert evidence.boundary_label.label_id in {
            label.label_id for label in result.organization.labels
        }
        evidence.require_current(
            result.mesh, result.organization.labels, result.organization.patches
        )
    interface_patch = next(
        patch
        for patch in result.organization.patches
        if patch.source_adjacent_region_ids == ("upper", "lower")
    )
    assert interface_patch.source_adjacent_region_ids is not None
    assert interface_patch.source_adjacent_region_occurrence_paths is not None
    assert interface_patch.source_adjacent_region_occurrence_paths == ((), ())
    assert dict(evidence_by_region["upper"].patch_sides)[interface_patch.patch_id] == 1
    assert dict(evidence_by_region["lower"].patch_sides)[interface_patch.patch_id] == -1
    face_association = next(
        association
        for association in result.organization.associations
        if association.association_kind is GeometryAssociationKind.PIECEWISE_LINEAR
        and association.target_entity_set_id == result.mesh.entity_set(2).entity_set_id
        and any(
            role is GeometrySourceEntityRole.FACET for role in association.source_roles
        )
    )
    assert any(
        positive is not None and negative is not None and positive != negative
        for positive, negative in face_association.source_region_sides
    )
    assert result.certification.cell_regions is not None
    assert set(
        np.asarray(result.certification.cell_regions, dtype=np.int64).tolist()
    ) == {
        0,
        1,
    }


def test_native_plc_region_control_identity_is_preserved_in_boundary_evidence() -> None:
    source = _source()
    region_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        np.asarray((source.complex.region_entity_ids[0],), dtype=np.int64),
    )
    control = RegionControl(
        region_scope,
        "upper",
        "controlled-upper",
        RegionRole.FLUID,
    )

    construction = _prepare(source, _request(source, region_controls=(control,)))

    upper = next(
        evidence
        for evidence in construction.organization.region_boundary_evidence
        if evidence.source_region_id == "upper"
    )
    assert upper.control_id == control.control_id
    assert upper.source_scope.scope_id == control.scope.scope_id
    assert upper.material_id == "controlled-upper"
    assert upper.role is RegionRole.FLUID
    lower = next(
        evidence
        for evidence in construction.organization.region_boundary_evidence
        if evidence.source_region_id == "lower"
    )
    assert lower.control_id is None


def test_native_plc_boundary_evidence_separates_repeated_region_occurrences() -> None:
    source = _repeated_region_source()
    specification = _request(source, region_seeds=())
    volume = _construct(source, specification)
    organization = _organization(source, specification, volume)

    assert isinstance(volume.domain, PiecewiseLinearDomain)
    assert volume.domain.region_ids == source.complex.region_ids
    assert volume.domain.region_occurrence_paths == source.complex.region_occurrence_paths
    _, accepted_regions = _prepare_region_assignments(
        volume.mesh,
        MeshCertificationSchedule("volume_plc"),
        volume.domain,
        volume.cell_regions,
        source.source_id,
    )
    np.testing.assert_array_equal(accepted_regions, volume.cell_regions)
    evidence_by_path = {
        evidence.source_occurrence_path: evidence
        for evidence in organization.region_boundary_evidence
    }
    assert set(evidence_by_path) == {("assembly-a",), ("assembly-b",)}
    assert all(
        evidence.source_region_id == "shared-region" and evidence.control_id is None
        for evidence in evidence_by_path.values()
    )
    assert (
        len({evidence.source_scope.scope_id for evidence in evidence_by_path.values()})
        == 2
    )
    assert (
        len(
            {
                int(evidence.source_scope.entity_ids[0])
                for evidence in evidence_by_path.values()
            }
        )
        == 2
    )
    zone_names = tuple(zone.name for zone in organization.zones)
    assert len(set(zone_names)) == 2
    assert all(name.startswith("shared-region@") for name in zone_names)
    for evidence in evidence_by_path.values():
        assert source.source_entity_key_for_scope(
            3, int(evidence.source_scope.entity_ids[0])
        ) == (evidence.source_region_id, evidence.source_occurrence_path)
        evidence.require_current(volume.mesh, organization.labels, organization.patches)
    accepted_evidence = _validated_source_evidence(
        volume.mesh,
        SpatialCoordinateContract.si(),
        None,
        organization.region_boundary_evidence,
        None,
        organization.labels,
        organization.patches,
    )
    assert {
        (evidence.source_region_id, evidence.source_occurrence_path)
        for evidence in accepted_evidence
    } == {
        ("shared-region", ("assembly-a",)),
        ("shared-region", ("assembly-b",)),
    }
    interface_patch = next(
        patch
        for patch in organization.patches
        if patch.source_adjacent_region_ids == ("shared-region", "shared-region")
    )
    assert interface_patch.source_adjacent_region_ids is not None
    assert interface_patch.source_adjacent_region_occurrence_paths is not None
    assert interface_patch.source_adjacent_region_occurrence_paths == (
        ("assembly-a",),
        ("assembly-b",),
    )
    assert (
        dict(evidence_by_path[("assembly-a",)].patch_sides)[interface_patch.patch_id] == 1
    )
    assert (
        dict(evidence_by_path[("assembly-b",)].patch_sides)[interface_patch.patch_id]
        == -1
    )


def test_conforming_plc_fill_retains_source_ancestry_and_region_coverage() -> None:
    source = _source("conforming")
    construction = _construct(source, _request(source))
    assert construction.boundary_policy == "conforming"
    assert construction.source_id == source.source_id
    assert construction.source_revision == source.source_revision
    assert construction.complex_id == source.complex.complex_id
    assert len(construction.vertex_source_ids) == construction.mesh.coordinates.shape[0]
    assert len(construction.face_source_ids) == construction.boundary_faces.shape[0]
    assert len(construction.segment_source_ids) == construction.segments.shape[0]
    assert construction.region_ids == tuple(construction.domain.region_ids)


def _conforming_refinement(
    source: NativePlcSource, seeds: tuple[RegionSeed, ...], size: float
) -> VolumeConstruction:
    base = _request(source, region_seeds=seeds)
    specification = VolumeMeshingSpec(
        base.target,
        base.boundary_scope,
        base.fill_strategy,
        size_controls=(UniformSizeControl(base.boundary_scope, size),),
        region_seeds=base.region_seeds,
    )
    return generate_plc_volume(
        source.complex,
        specification,
        NativeVolumeSchedule(radius_edge_bound=2.0, minimum_dihedral_degrees=0.0),
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_map_id="",
        frame_id=SpatialCoordinateContract.si().spatial_id,
    )


def _assert_refined_with_boundary_splits(construction: VolumeConstruction) -> None:
    assert construction.boundary_policy == "conforming"
    assert construction.refinement_status is MeshcoreStatus.OK
    assert construction.improvement_status is MeshcoreStatus.OK
    assert construction.refinement_unmet == ()
    assert construction.improvement_unmet == ()
    strata = np.asarray(construction.witness_strata)
    # Facet and segment points were inserted by boundary splits, each on an
    # exact source row.
    assert np.count_nonzero(strata == 1) > 0
    assert np.count_nonzero(strata == 2) > 0


_CUBE_SEED = (
    RegionSeed(
        np.zeros(3, dtype=np.float64), "solid", "solid-material", RegionRole.SOLID
    ),
)


def test_conforming_refinement_splits_grouped_cube_faces_across_source_triangles() -> (
    None
):
    # Facet splits land in whichever source triangle of a face group holds
    # them, and a size-split midpoint that encroaches a subsegment splits the
    # subsegment first, so the size field is met with every cell within bound.
    source = _grouped_cube_source("conforming", 0.0)
    construction = _conforming_refinement(source, _CUBE_SEED, 0.05)
    _assert_refined_with_boundary_splits(construction)
    rows = np.asarray(construction.witness_entities)[
        np.asarray(construction.witness_strata) == 2
    ]
    # Both source triangles of the face groups receive facet points.
    assert np.unique(rows).size == construction.source_faces.shape[0]
    complex_ = source.complex
    diagonal_ids = {
        complex_.plc_edge_ids[row]
        for row, facets in enumerate(complex_.plc_edge_facets)
        if len(facets) != 2
    }
    assert not set(construction.segment_source_ids) & diagonal_ids


def test_conforming_refinement_keeps_undeclared_rotated_cube_triangles_as_authored() -> (
    None
):
    # Without a declared exact plane, the rotated cube's two rounded triangles
    # per face are not coplanar: they are the geometry, and their diagonals
    # stay segments.
    source = _grouped_cube_source("conforming", 0.3)
    construction = _conforming_refinement(source, _CUBE_SEED, 0.05)
    _assert_refined_with_boundary_splits(construction)
    complex_ = source.complex
    diagonal_ids = {
        complex_.plc_edge_ids[row]
        for row, facets in enumerate(complex_.plc_edge_facets)
        if len(facets) != 2
    }
    assert len(diagonal_ids) == 6
    assert diagonal_ids <= set(construction.segment_source_ids)


def test_conforming_refinement_meets_bound_in_rotated_bipyramid() -> None:
    # Acute input dihedral angles (54.7 degrees) and segment angles (45
    # degrees) next to boundary splits of every facet.
    angle = 0.3
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    base = _source("conforming")
    complex_ = PiecewiseLinearComplex(
        _POINTS @ rotation.T / 3.0,
        _POLYGONS,
        np.arange(_POLYGONS.shape[0], dtype=np.int64),
        _FACET_REGIONS,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=base.complex.vertex_ids,
        vertex_occurrence_paths=base.complex.vertex_occurrence_paths,
        facet_ids=base.complex.facet_ids,
        facet_occurrence_paths=base.complex.facet_occurrence_paths,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
        boundary="conforming",
    )
    source = NativePlcSource(complex_, "rotated-bipyramid", "revision-1")
    seeds = tuple(
        RegionSeed(
            np.asarray(seed.point) @ rotation.T / 3.0,
            seed.region_name,
            seed.material_id,
            seed.role,
        )
        for seed in _request(source).region_seeds
    )
    _assert_refined_with_boundary_splits(_conforming_refinement(source, seeds, 0.1))


def test_mapped_carrier_classification_uses_plc_incidence_and_witnesses() -> None:
    from phydrax.meshing._plc_mapped_support import classify_plc_carriers

    source = _source("fixed")
    construction = _construct(source, _request(source))
    native_vertex_ids = np.asarray(construction.mesh.vertex_global_ids, dtype=np.int64)
    original_vertex_row = int(np.flatnonzero(construction.input_vertices >= 0)[0])
    source_face = np.asarray(construction.boundary_faces[0], dtype=np.int64)
    source_cell = np.asarray(construction.mesh.blocks[0].vertices[0], dtype=np.int64)
    carriers = np.full((3, 4), -1, dtype=np.int64)
    carriers[0, 0] = native_vertex_ids[original_vertex_row]
    carriers[1, :3] = native_vertex_ids[source_face]
    carriers[2] = native_vertex_ids[source_cell]

    ancestry = classify_plc_carriers(construction, carriers)
    assert ancestry.dimensions.tolist() == [
        int(construction.vertex_dimension[original_vertex_row]),
        2,
        3,
    ]
    assert (
        ancestry.source_entity_ids[0]
        == construction.vertex_source_ids[original_vertex_row]
    )
    assert ancestry.source_entity_ids[1] == construction.face_source_ids[0]
    region = int(construction.cell_regions[0])
    assert ancestry.source_entity_ids[2] == construction.region_ids[region]
    assert (
        ancestry.source_occurrence_paths[2]
        == construction.region_occurrence_paths[region]
    )


def _stacked_box_source() -> NativePlcSource:
    points = np.asarray(
        [
            (x, y, z)
            for z in (0.0, 0.5, 1.0)
            for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        ],
        dtype=np.float64,
    )

    def sides(base: int, /) -> tuple[tuple[int, int, int], ...]:
        rows: list[tuple[int, int, int]] = []
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
            a, b = base + first, base + second
            rows.extend(((a, b, b + 4), (a, b + 4, a + 4)))
        return tuple(rows)

    triangles = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (8, 9, 10),
            (8, 10, 11),
            (4, 5, 6),
            (4, 6, 7),
            *sides(0),
            *sides(4),
        ),
        dtype=np.int64,
    )
    # Facets: bottom, top, the two-triangle material interface, then the four
    # side squares of the lower and of the upper box.
    facets = np.asarray(
        (0, 0, 1, 1, 2, 2, *np.repeat(np.arange(3, 11), 2).tolist()),
        dtype=np.int64,
    )
    facet_regions = np.asarray(
        ((-1, 1), (-1, 0), (0, 1), *((-1, 1),) * 4, *((-1, 0),) * 4),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        points,
        triangles,
        facets,
        facet_regions,
        ("upper", "lower"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=tuple(f"stack-vertex-{row}" for row in range(points.shape[0])),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=tuple(f"stack-facet-{row}" for row in range(11)),
        facet_occurrence_paths=((),) * 11,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
        boundary="fixed",
    )
    return NativePlcSource(complex_, "stacked-boxes", "revision-1")


def test_mapped_carrier_classification_assigns_interface_diagonals_to_their_facet() -> (
    None
):
    from phydrax.meshing._plc_mapped_support import classify_plc_carriers

    source = _stacked_box_source()
    construction = _construct(
        source,
        _request(
            source,
            region_seeds=(
                RegionSeed(
                    np.asarray((0.5, 0.5, 0.75), dtype=np.float64),
                    "upper",
                    "upper-material",
                    RegionRole.SOLID,
                ),
                RegionSeed(
                    np.asarray((0.5, 0.5, 0.25), dtype=np.float64),
                    "lower",
                    "lower-material",
                    RegionRole.SOLID,
                ),
            ),
        ),
    )
    interface_rows = np.flatnonzero(
        np.asarray(construction.face_source_facets, dtype=np.int64) == 2
    )
    interface_faces = np.asarray(construction.boundary_faces, dtype=np.int64)[
        interface_rows
    ]
    assert interface_faces.shape == (2, 3)
    diagonal = sorted(set(interface_faces[0].tolist()) & set(interface_faces[1].tolist()))
    assert len(diagonal) == 2
    native_vertex_ids = np.asarray(construction.mesh.vertex_global_ids, dtype=np.int64)
    carriers = np.full((1, 4), -1, dtype=np.int64)
    carriers[0, :2] = native_vertex_ids[np.asarray(diagonal, dtype=np.int64)]

    ancestry = classify_plc_carriers(construction, carriers)

    assert ancestry.dimensions.tolist() == [2]
    assert ancestry.source_entity_ids == (
        construction.face_source_ids[int(interface_rows[0])],
    )
    assert ancestry.source_entity_ids[0] == source.complex.facet_ids[2]
