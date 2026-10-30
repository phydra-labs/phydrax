from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest
from jax import Array

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._surface_chart_records import (
    exact_chart_bank,
    SurfaceChartWitness,
    SurfaceRootMeshCellBinding,
)
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._surface_source_support import validate_surface_source_pieces
from phydrax.geometry.brep._patches import LineCurve, PlanePatch
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._lineage import (
    EntityLineage,
    EntityLineageKind,
    MeshLineage,
    VertexInterpolationStencil,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing._surface_boundary_operations import (
    created_weights,
    lineage_chart_witness,
)
from phydrax.meshing._surface_construction import (
    curve_edge_parameter_intervals,
    curve_vertex_chains,
    source_cell_ancestry,
    SurfaceConstruction,
    SurfaceSourceCellAtlas,
)
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_surface import (
    _ChartMesh,
    _split_chart_at_source_cells,
    _surface_root_charts,
)
from phydrax.meshing.providers._native_surface_association import (
    _curve_orientations,
    _face_orientations,
    _validate_shared_junctions,
    simplex_rows,
)


ExactUVPoint = tuple[Fraction, Fraction]
ExactUVTriangle = tuple[ExactUVPoint, ExactUVPoint, ExactUVPoint]


def _vertex_stencil(
    source: CellMesh,
    target: CellMesh,
    created_supports: dict[int, tuple[tuple[int, Fraction], ...]],
) -> VertexInterpolationStencil:
    source_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    source_id_set = set(source_ids.tolist())
    target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    rows = [
        ((int(identifier), Fraction(1)),)
        if int(identifier) in source_id_set
        else created_supports[int(identifier)]
        for identifier in target_ids.tolist()
    ]
    width = max(map(len, rows))
    stencil_sources = np.zeros((len(rows), width), dtype=np.int64)
    weights = np.zeros((len(rows), width), dtype=np.float64)
    valid = np.zeros((len(rows), width), dtype=np.bool_)
    for row, support in enumerate(rows):
        for column, (identifier, coefficient) in enumerate(support):
            stencil_sources[row, column] = identifier
            weights[row, column] = float(coefficient)
            valid[row, column] = True
    return VertexInterpolationStencil(
        source.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        target_ids,
        stencil_sources,
        weights,
        valid,
    )


def test_source_curve_chains_are_discretized_once_with_shared_junction_ids() -> None:
    parameters = (
        (2, np.asarray((0.0, 0.4, 1.0), dtype=np.float64)),
        (5, np.asarray((0.0, 0.6, 1.0), dtype=np.float64)),
    )
    edges = np.asarray(((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int64)
    curves = np.asarray((2, 2, 5, 5), dtype=np.int64)

    chains = curve_vertex_chains(parameters, edges, curves)

    assert tuple(identifier for identifier, _ in chains) == (2, 5)
    np.testing.assert_array_equal(chains[0][1], (0, 1, 2))
    np.testing.assert_array_equal(chains[1][1], (2, 3, 4))
    assert chains[0][1][-1] == chains[1][1][0]


def test_closed_source_curve_reuses_only_its_authored_terminal_vertex() -> None:
    parameters = ((6, np.asarray((0.0, 0.25, 0.5, 0.75, 1.0), dtype=np.float64)),)
    edges = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64)

    chains = curve_vertex_chains(parameters, edges, np.full((4,), 6, dtype=np.int64))

    np.testing.assert_array_equal(chains[0][1], (0, 1, 2, 3, 0))


def test_source_curve_chains_reject_patch_local_duplicates() -> None:
    parameters = ((3, np.asarray((0.0, 1.0), dtype=np.float64)),)
    edges = np.asarray(((0, 1), (0, 1)), dtype=np.int64)
    curves = np.asarray((3, 3), dtype=np.int64)

    with pytest.raises(ValueError, match="one strictly ordered source-curve chain"):
        curve_vertex_chains(parameters, edges, curves)


def test_source_curve_chains_reject_disconnected_edges() -> None:
    parameters = ((7, np.asarray((0.0, 0.5, 1.0), dtype=np.float64)),)
    edges = np.asarray(((0, 1), (2, 3)), dtype=np.int64)
    curves = np.asarray((7, 7), dtype=np.int64)

    with pytest.raises(ValueError, match="one strictly ordered source-curve chain"):
        curve_vertex_chains(parameters, edges, curves)


def test_source_curve_chains_reject_duplicate_curve_parameterizations() -> None:
    parameters = (
        (4, np.asarray((0.0, 1.0), dtype=np.float64)),
        (4, np.asarray((0.0, 1.0), dtype=np.float64)),
    )
    edges = np.asarray(((0, 1), (2, 3)), dtype=np.int64)
    curves = np.asarray((4, 4), dtype=np.int64)

    with pytest.raises(ValueError, match="source-curve ID once"):
        curve_vertex_chains(parameters, edges, curves)


def test_source_curve_chains_refuse_untagged_partition_edges() -> None:
    parameters = ((3, np.asarray((0.0, 1.0), dtype=np.float64)),)
    edges = np.asarray(((0, 1), (1, 2)), dtype=np.int64)
    curves = np.asarray((3, 99), dtype=np.int64)

    with pytest.raises(ValueError, match="missing or foreign curve rows"):
        curve_vertex_chains(parameters, edges, curves)


def test_source_curve_chains_reject_repeated_nonjunction_vertices() -> None:
    parameters = ((9, np.asarray((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), dtype=np.float64)),)
    edges = np.asarray(((0, 1), (1, 2), (2, 3), (3, 1), (1, 4)), dtype=np.int64)
    curves = np.asarray((9, 9, 9, 9, 9), dtype=np.int64)

    with pytest.raises(ValueError, match="distinct source-parameter vertices"):
        curve_vertex_chains(parameters, edges, curves)


def test_surface_associations_match_triangle_rows_without_orientation_loss() -> None:
    canonical = np.asarray(((2, 5, 8), (1, 4, 7)), dtype=np.int64)
    mesh_rows = np.asarray(((7, 4, 1), (8, 2, 5)), dtype=np.int64)

    np.testing.assert_array_equal(simplex_rows(canonical, mesh_rows), (1, 0))


def test_surface_associations_refuse_unmatched_source_edge_rows() -> None:
    canonical = np.asarray(((0, 1), (1, 2)), dtype=np.int64)
    construction = np.asarray(((0, 2),), dtype=np.int64)

    with pytest.raises(ValueError, match="absent simplex"):
        simplex_rows(canonical, construction)


@dataclass
class _ValidatedSourceAtlas:
    root_cell_ids: tuple[int, ...]
    root_source_cell_ids: tuple[tuple[int, ...], ...]
    root_parameter_bounds: tuple[tuple[tuple[Fraction, Fraction], ...], ...]
    root_patches: Array | np.ndarray
    root_parameters: Array | np.ndarray

    def require_current(self) -> None:
        if (
            len(self.root_cell_ids) != len(self.root_source_cell_ids)
            or len(self.root_cell_ids) != len(self.root_parameter_bounds)
            or self.root_patches.shape != (len(self.root_cell_ids),)
            or self.root_parameters.shape != (len(self.root_cell_ids), 4, 2)
        ):
            raise ValueError("Test source atlas ancestry tables are inconsistent.")
        for row, bounds in enumerate(self.root_parameter_bounds):
            represented = np.asarray(
                [[float(u), float(v)] for u, v in bounds], dtype=np.float64
            )
            if not np.array_equal(self.root_parameters[row], represented):
                raise ValueError("Test source atlas chart realization is stale.")


def _tspline_cell_atlas() -> _ValidatedSourceAtlas:
    one_third = Fraction(1, 3)
    bounds = (
        (
            (Fraction(0), Fraction(0)),
            (one_third, Fraction(0)),
            (one_third, Fraction(1)),
            (Fraction(0), Fraction(1)),
        ),
        (
            (one_third, Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(1), Fraction(1)),
            (one_third, Fraction(1)),
        ),
    )
    parameters = np.asarray(
        [[[float(u), float(v)] for u, v in corners] for corners in bounds],
        dtype=np.float64,
    )
    return _ValidatedSourceAtlas(
        root_cell_ids=(101, 202),
        root_source_cell_ids=((4, 7001, 11), (4, 9009, 22)),
        root_parameter_bounds=bounds,
        root_patches=np.asarray((4, 4), dtype=np.int64),
        root_parameters=parameters,
    )


def test_source_cell_ancestry_binds_nonbinary_tspline_cells_exactly() -> None:
    atlas: SurfaceSourceCellAtlas = _tspline_cell_atlas()
    one_third = Fraction(1, 3)
    exact_triangles: tuple[ExactUVTriangle, ...] = (
        (
            (Fraction(1, 20), Fraction(1, 10)),
            (one_third, Fraction(1, 10)),
            (Fraction(1, 5), Fraction(9, 10)),
        ),
        (
            (one_third, Fraction(1, 10)),
            (Fraction(9, 10), Fraction(1, 10)),
            (Fraction(4, 5), Fraction(9, 10)),
        ),
    )
    triangles = np.asarray(
        [
            [[float(value) for value in point] for point in triangle]
            for triangle in exact_triangles
        ],
        dtype=np.float64,
    )

    cell_ids, source_pieces = source_cell_ancestry(
        atlas,
        np.asarray((4, 4), dtype=np.int64),
        triangles,
        exact_triangles,
    )

    np.testing.assert_array_equal(cell_ids, (101, 202))
    assert source_pieces == ((4, 7001, 11), (4, 9009, 22))


def test_source_cell_ancestry_selects_root_charts_per_patch() -> None:
    zero, one, ten, eleven = (
        Fraction(0),
        Fraction(1),
        Fraction(10),
        Fraction(11),
    )
    bounds = (
        ((zero, zero), (one, zero), (one, one), (zero, one)),
        ((ten, zero), (eleven, zero), (eleven, one), (ten, one)),
    )
    atlas = _ValidatedSourceAtlas(
        root_cell_ids=(41, 91),
        root_source_cell_ids=((4, 701), (9, 901)),
        root_parameter_bounds=bounds,
        root_patches=np.asarray((4, 9), dtype=np.int64),
        root_parameters=np.asarray(
            [[[float(u), float(v)] for u, v in corners] for corners in bounds],
            dtype=np.float64,
        ),
    )
    exact_triangles: tuple[ExactUVTriangle, ...] = (
        (
            (Fraction(1, 10), Fraction(1, 10)),
            (Fraction(8, 10), Fraction(1, 10)),
            (Fraction(2, 10), Fraction(8, 10)),
        ),
        (
            (Fraction(101, 10), Fraction(1, 10)),
            (Fraction(108, 10), Fraction(1, 10)),
            (Fraction(102, 10), Fraction(8, 10)),
        ),
    )
    numeric_triangles = np.asarray(
        [[[float(u), float(v)] for u, v in triangle] for triangle in exact_triangles],
        dtype=np.float64,
    )

    cell_ids, source_pieces = source_cell_ancestry(
        atlas,
        np.asarray((4, 9), dtype=np.int64),
        numeric_triangles,
        exact_triangles,
    )

    np.testing.assert_array_equal(cell_ids, (41, 91))
    assert source_pieces == ((4, 701), (9, 901))


def test_source_cell_ancestry_refuses_triangles_crossing_piece_boundaries() -> None:
    atlas: SurfaceSourceCellAtlas = _tspline_cell_atlas()
    exact_triangles: tuple[ExactUVTriangle, ...] = (
        (
            (Fraction(1, 10), Fraction(1, 10)),
            (Fraction(9, 10), Fraction(1, 10)),
            (Fraction(1, 2), Fraction(9, 10)),
        ),
    )
    triangles = np.asarray(
        [
            [[float(value) for value in point] for point in triangle]
            for triangle in exact_triangles
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="cross-cell or ambiguous"):
        source_cell_ancestry(
            atlas,
            np.asarray((4,), dtype=np.int64),
            triangles,
            exact_triangles,
        )


def test_native_surface_subdivides_tjunction_root_cells_exactly() -> None:
    zero, one = Fraction(0), Fraction(1)
    third = Fraction(1, 3)
    bounds = (
        ((zero, zero), (third, zero), (third, third), (zero, third)),
        ((zero, third), (third, third), (third, one), (zero, one)),
        ((third, zero), (one, zero), (one, one), (third, one)),
    )
    atlas = _ValidatedSourceAtlas(
        root_cell_ids=(101, 202, 303),
        root_source_cell_ids=((4, 7001, 11), (4, 9009, 22), (4, 3333, 33)),
        root_parameter_bounds=bounds,
        root_patches=np.asarray((4, 4, 4), dtype=np.int64),
        root_parameters=np.asarray(
            [[[float(u), float(v)] for u, v in corners] for corners in bounds],
            dtype=np.float64,
        ),
    )
    occurrence = (4, 0, 0)
    chart = _ChartMesh(
        4,
        [(zero, zero), (one, zero), (zero, one)],
        [(0, 1, 2)],
        {},
        {(0, 7): zero, (2, 7): one},
        {(0, 2): (7, occurrence, zero, one)},
    )

    _split_chart_at_source_cells(chart, _surface_root_charts(atlas)[4])

    cut_row = chart.points.index((zero, third))
    assert chart.curve_parameters[(cut_row, 7)] == third
    assert chart.boundary_edges[(0, cut_row)] == (7, occurrence, zero, third)
    boundary_edge_key = (min(cut_row, 2), max(cut_row, 2))
    assert chart.boundary_edges[boundary_edge_key] == (
        7,
        occurrence,
        third,
        one,
    )
    exact_triangles: tuple[ExactUVTriangle, ...] = tuple(
        (
            chart.points[triangle[0]],
            chart.points[triangle[1]],
            chart.points[triangle[2]],
        )
        for triangle in chart.triangles
    )
    triangle_parameters = np.asarray(
        [[[float(u), float(v)] for u, v in triangle] for triangle in exact_triangles],
        dtype=np.float64,
    )
    cell_ids, source_pieces = source_cell_ancestry(
        atlas,
        np.full((len(exact_triangles),), 4, dtype=np.int64),
        triangle_parameters,
        exact_triangles,
    )
    assert set(cell_ids.tolist()) == {101, 202, 303}
    assert set(source_pieces) == {
        (4, 7001, 11),
        (4, 9009, 22),
        (4, 3333, 33),
    }


def test_surface_subdivision_reference_children_tile_exactly() -> None:
    from phydrax.meshing._surface_boundary_sources import (
        _tiles_reference_triangle,
    )

    zero, one, half = Fraction(0), Fraction(1), Fraction(1, 2)
    lower_left = ((zero, zero), (half, zero), (zero, half))
    lower_right = ((half, zero), (one, zero), (half, half))
    upper_left = ((zero, half), (half, half), (zero, one))
    center = ((half, zero), (half, half), (zero, half))

    assert _tiles_reference_triangle([lower_left, lower_right, upper_left, center])
    assert not _tiles_reference_triangle([lower_left, lower_right, upper_left])
    assert not _tiles_reference_triangle(
        [lower_left, lower_right, upper_left, center[::-1]]
    )


def test_surface_face_orientation_matches_native_source_normal() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    faces = np.asarray(((0, 1, 2), (0, 2, 1)), dtype=np.int64)
    normals = np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)))

    np.testing.assert_array_equal(_face_orientations(points, faces, normals), (1, -1))


def test_surface_curve_orientation_combines_mesh_and_coedge_senses() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    canonical_edges = np.asarray(((0, 1), (1, 0)), dtype=np.int64)
    curve_edges = np.asarray(((0, 1), (0, 1)), dtype=np.int64)

    np.testing.assert_array_equal(
        _curve_orientations(
            points, canonical_edges, curve_edges, np.asarray((1, 1), dtype=np.int8)
        ),
        (1, -1),
    )
    np.testing.assert_array_equal(
        _curve_orientations(
            points,
            np.asarray(((0, 1),), dtype=np.int64),
            np.asarray(((0, 1),), dtype=np.int64),
            np.asarray((-1,), dtype=np.int8),
        ),
        (-1,),
    )


def test_source_curve_edge_intervals_retain_exact_parameter_endpoints() -> None:
    parameters = (
        (2, np.asarray((0.0, 0.25, 1.0), dtype=np.float64)),
        (5, np.asarray((0.0, 0.6, 1.0), dtype=np.float64)),
    )
    curves = np.asarray((2, 5, 2, 5), dtype=np.int64)

    intervals = curve_edge_parameter_intervals(parameters, curves)

    assert intervals == (
        (Fraction(0), Fraction(0.25)),
        (Fraction(0), Fraction(0.6)),
        (Fraction(0.25), Fraction(1)),
        (Fraction(0.6), Fraction(1)),
    )


def _junction_construction(shared: bool) -> SurfaceConstruction:
    if shared:
        dimensions = np.asarray((0, 1, 0, 1, 0), dtype=np.int8)
        indices = np.asarray((0, 0, 1, 1, 2), dtype=np.int64)
        edges = np.asarray(((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int64)
    else:
        dimensions = np.asarray((0, 1, 0, 0, 1, 0), dtype=np.int8)
        indices = np.asarray((0, 0, 1, 1, 1, 2), dtype=np.int64)
        edges = np.asarray(((0, 1), (1, 2), (3, 4), (4, 5)), dtype=np.int64)
    empty_float = np.empty((0,), dtype=np.float64)
    return SurfaceConstruction(
        vertices=np.zeros((dimensions.size, 3), dtype=np.float64),
        vertex_source_dimensions=dimensions,
        vertex_source_indices=indices,
        vertex_parameters=np.zeros((dimensions.size, 2), dtype=np.float64),
        chart_restriction_vertices=np.empty((0,), dtype=np.int64),
        chart_restriction_edges=np.empty((0, 2), dtype=np.int64),
        chart_restriction_endpoint_parameters=np.empty((0, 2, 2), dtype=np.float64),
        chart_restriction_parameters=empty_float,
        triangles=np.empty((0, 3), dtype=np.int64),
        triangle_patches=np.empty((0,), dtype=np.int64),
        triangle_charts=np.empty((0, 2), dtype=np.float64),
        triangle_parameters=np.empty((0, 3, 2), dtype=np.float64),
        triangle_source_cells=np.empty((0,), dtype=np.int64),
        triangle_source_pieces=(),
        triangle_deviations=empty_float,
        triangle_deviation_bounds=empty_float,
        triangle_normal_bounds=empty_float,
        curve_edges=edges,
        curve_edge_curves=np.asarray((0, 0, 1, 1), dtype=np.int64),
        curve_edge_deviations=np.zeros((4,), dtype=np.float64),
        curve_edge_deviation_bounds=np.zeros((4,), dtype=np.float64),
        unresolved=np.empty((0,), dtype=np.bool_),
        boundary_occurrences=np.empty((0, 3, 3), dtype=np.int32),
        chart_triangulations=(),
        curve_parameters=(
            (0, np.asarray((0.0, 1.0, 2.0), dtype=np.float64)),
            (1, np.asarray((0.0, 1.0, 2.0), dtype=np.float64)),
        ),
        minimum_angle=0.0,
    )


def test_shared_source_curve_uses_reuse_authored_junction_identity() -> None:
    curves = (MeshingDomainCurve(0, 1), MeshingDomainCurve(1, 2))
    construction = _junction_construction(shared=True)

    _validate_shared_junctions(curves, construction)


def test_shared_source_curves_reject_duplicate_corner_rows_not_coordinate_welding() -> (
    None
):
    curves = (MeshingDomainCurve(0, 1), MeshingDomainCurve(1, 2))
    construction = _junction_construction(shared=False)

    with pytest.raises(ValueError, match="reuse one source-junction vertex ID"):
        _validate_shared_junctions(curves, construction)


def test_surface_lineage_keeps_root_mesh_atlas_and_piece_ids_distinct() -> None:
    source = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        np.asarray(((0, 1, 2),), dtype=np.int64),
        vertex_global_ids=np.asarray((100, 101, 102), dtype=np.int64),
        cell_global_ids=np.asarray((5001,), dtype=np.int64),
    )
    target = CellMesh.from_triangles(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.5, 0.0, 0.0),
            )
        ),
        np.asarray(((0, 3, 2), (3, 1, 2)), dtype=np.int64),
        vertex_global_ids=np.asarray((100, 101, 102, 199), dtype=np.int64),
        cell_global_ids=np.asarray((8001, 8002), dtype=np.int64),
    )
    reference = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    root_binding = SurfaceRootMeshCellBinding("source-geometry", source.topology_id, 5001)
    source_witness = SurfaceChartWitness(
        geometry_id="source-geometry",
        topology_id=source.topology_id,
        domain_id="domain",
        cell_global_ids=np.asarray((5001,), dtype=np.int64),
        patches=np.asarray((0,), dtype=np.int64),
        charts=np.asarray((reference,), dtype=np.float64),
        source_root_cell_ids=np.asarray((17,), dtype=np.int64),
        root_mesh_cell_bindings=(root_binding,),
        source_piece_ids=((0, 701, 9001),),
        geometry_entity_ids=("source-face-0",),
        occurrence_paths=((),),
    )
    cell_lineage = EntityLineage(
        2,
        source.entity_set(2).entity_set_id,
        target.entity_set(2).entity_set_id,
        np.asarray((5001, 5001), dtype=np.int64),
        np.asarray((8001, 8002), dtype=np.int64),
        np.asarray(
            (EntityLineageKind.SPLIT_FROM, EntityLineageKind.SPLIT_FROM),
            dtype=np.int32,
        ),
    )
    lineage = MeshLineage(
        source.topology_id,
        target.topology_id,
        (cell_lineage,),
    )

    weights = created_weights(
        _vertex_stencil(
            source, target, {199: ((100, Fraction(1, 2)), (101, Fraction(1, 2)))}
        ),
        source,
        target,
    )
    witness = lineage_chart_witness(
        source_witness,
        source,
        lineage,
        target,
        weights,
        source_geometry_id="source-geometry",
        geometry_id="target-geometry",
    )

    assert tuple(np.asarray(witness.cell_global_ids)) == (8001, 8002)
    assert tuple(witness.root_mesh_cell_bindings) == (root_binding, root_binding)
    assert tuple(np.asarray(witness.source_root_cell_ids)) == (17, 17)
    assert witness.source_piece_ids == ((0, 701, 9001), (0, 701, 9001))
    assert witness.exact_chart(0) == (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    assert witness.exact_chart(1) == (
        (Fraction(1, 2), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )


def _two_patch_plane_domain(
    source_patch_ids: tuple[int, int],
    source_patch_paths: tuple[tuple[str, ...], tuple[str, ...]],
) -> MeshingDomain:
    curves = (
        MeshingDomainCurve(0, 1),
        MeshingDomainCurve(1, 4),
        MeshingDomainCurve(4, 5),
        MeshingDomainCurve(5, 0),
        MeshingDomainCurve(1, 2),
        MeshingDomainCurve(2, 3),
        MeshingDomainCurve(3, 4),
    )
    left = MeshingSurfacePatch(
        PlanePatch((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (
            (
                PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
                PatchCurveUse(1, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
                PatchCurveUse(2, LineCurve((1.0, 1.0), (-1.0, 0.0)), 0.0, 1.0),
                PatchCurveUse(3, LineCurve((0.0, 1.0), (0.0, -1.0)), 0.0, 1.0),
            ),
        ),
    )
    right = MeshingSurfacePatch(
        PlanePatch((0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (
            (
                PatchCurveUse(4, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
                PatchCurveUse(5, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
                PatchCurveUse(6, LineCurve((1.0, 1.0), (-1.0, 0.0)), 0.0, 1.0),
                PatchCurveUse(
                    1,
                    LineCurve((0.0, 0.0), (0.0, 1.0)),
                    0.0,
                    1.0,
                    sense=-1,
                ),
            ),
        ),
    )
    return MeshingDomain(
        (left, right),
        curves,
        6,
        source_id="two-patch-source",
        source_revision="source-revision-1",
        source_indices=(tuple(range(6)), tuple(range(7)), source_patch_ids),
        source_occurrences=(
            tuple(() for _ in range(6)),
            tuple(() for _ in range(7)),
            source_patch_paths,
        ),
        corner_points=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.5, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=np.float64,
        ),
    )


def _source_patch_scope_ids(domain: MeshingDomain) -> tuple[int, ...]:
    ids_by_key = {
        domain.source_entity_key_for_scope(2, int(scope_id)): int(scope_id)
        for scope_id in domain.scope_entity_ids(2)
    }
    return tuple(
        ids_by_key[(domain.entity_id(2, patch), domain.source_occurrences[2][patch])]
        for patch in range(len(domain.patches))
    )


def test_source_pieces_resolve_scope_by_entity_and_occurrence() -> None:
    domain = next(
        candidate
        for candidate in (
            _two_patch_plane_domain((10, 10), (("occurrence-a",), ("occurrence-b",))),
            _two_patch_plane_domain((10, 10), (("occurrence-b",), ("occurrence-a",))),
        )
        if _source_patch_scope_ids(candidate)
        != tuple(int(value) for value in candidate.scope_entity_ids(2))
    )
    expected_root_ids = _source_patch_scope_ids(domain)
    patch_ids = np.asarray((0, 1), dtype=np.int64)
    charts = np.asarray(
        (
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        ),
        dtype=np.float64,
    )
    source_pieces = tuple(
        (patch, int(domain.source_indices[2][patch])) for patch in range(2)
    )

    validate_surface_source_pieces(
        domain,
        patch_ids,
        charts,
        np.asarray(expected_root_ids, dtype=np.int64),
        source_pieces,
        exact_charts=exact_chart_bank(charts, patch_ids.size),
    )
    with pytest.raises(ValueError, match="canonical source-facet scope ID"):
        validate_surface_source_pieces(
            domain,
            patch_ids,
            charts,
            np.asarray(domain.scope_entity_ids(2), dtype=np.int64),
            source_pieces,
            exact_charts=exact_chart_bank(charts, patch_ids.size),
        )


def test_native_provider_maps_multi_patch_root_ids_by_entity_occurrence() -> None:
    contract = SpatialCoordinateContract.si()
    domain = next(
        candidate
        for candidate in (
            _two_patch_plane_domain((10, 10), (("occurrence-a",), ("occurrence-b",))),
            _two_patch_plane_domain((10, 10), (("occurrence-b",), ("occurrence-a",))),
        )
        if _source_patch_scope_ids(candidate)
        != tuple(int(value) for value in candidate.scope_entity_ids(2))
    )
    scope = MeshingScope.whole(domain, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(UniformSizeControl(scope, 0.5),),
        maximum_deviation=0.01,
    )
    result = (
        NativeMeshingProvider(NativeMeshingOptions("parametric_surface"))
        .plan(domain, specification, coordinate_contract=contract)
        .execute()
    )
    source = result.surface_source
    assert source is not None
    patches = np.asarray(source.root_patches, dtype=np.int64)
    expected_patch_root_ids = _source_patch_scope_ids(domain)
    expected = np.asarray(
        [expected_patch_root_ids[int(patch)] for patch in patches], dtype=np.int64
    )
    positional = np.asarray(domain.scope_entity_ids(2), dtype=np.int64)[patches]

    assert set(patches.tolist()) == {0, 1}
    assert not np.array_equal(expected, positional)
    np.testing.assert_array_equal(source.source_root_cell_ids, expected)
    assert source.source_piece_ids == tuple(
        (int(patch), int(domain.source_indices[2][int(patch)])) for patch in patches
    )
    witness = source.chart_witness
    witness_cell_ids = tuple(int(value) for value in np.asarray(witness.cell_global_ids))
    bindings = witness.root_mesh_cell_bindings
    assert all(binding is not None for binding in bindings)
    assert (
        tuple(binding.root_cell_global_id for binding in bindings if binding is not None)
        == witness_cell_ids
    )
    assert all(
        binding is not None
        and binding.root_geometry_id == cell_geometry_id(source.geometry)
        and binding.root_topology_id == source.mesh.topology_id
        for binding in bindings
    )


def test_surface_lineage_resolves_root_cells_with_shared_atlas_and_piece_ids() -> None:
    source = CellMesh.from_triangles(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        ),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        vertex_global_ids=np.asarray((100, 101, 102, 103), dtype=np.int64),
        cell_global_ids=np.asarray((5002, 5001), dtype=np.int64),
    )
    target = CellMesh.from_triangles(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.5, 0.5, 0.0),
            )
        ),
        np.asarray(((0, 4, 2), (4, 1, 2), (0, 5, 3), (5, 2, 3)), dtype=np.int64),
        vertex_global_ids=np.asarray((100, 101, 102, 103, 199, 200), dtype=np.int64),
        cell_global_ids=np.asarray((8001, 8002, 8003, 8004), dtype=np.int64),
    )
    first_chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1)),
    )
    second_chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(1)),
    )
    first_binding = SurfaceRootMeshCellBinding(
        "source-geometry", source.topology_id, 5002
    )
    second_binding = SurfaceRootMeshCellBinding(
        "source-geometry", source.topology_id, 5001
    )
    source_witness = SurfaceChartWitness(
        geometry_id="source-geometry",
        topology_id=source.topology_id,
        domain_id="domain",
        cell_global_ids=np.asarray((5002, 5001), dtype=np.int64),
        patches=np.asarray((0, 0), dtype=np.int64),
        charts=np.asarray((first_chart, second_chart), dtype=np.float64),
        source_root_cell_ids=np.asarray((17, 17), dtype=np.int64),
        root_mesh_cell_bindings=(first_binding, second_binding),
        source_piece_ids=((0, 701, 9001), (0, 701, 9001)),
        geometry_entity_ids=("source-face-0", "source-face-0"),
        occurrence_paths=((), ()),
    )
    cell_lineage = EntityLineage(
        2,
        source.entity_set(2).entity_set_id,
        target.entity_set(2).entity_set_id,
        np.asarray((5002, 5002, 5001, 5001), dtype=np.int64),
        np.asarray((8001, 8002, 8003, 8004), dtype=np.int64),
        np.asarray((EntityLineageKind.SPLIT_FROM,) * 4, dtype=np.int32),
    )
    lineage = MeshLineage(
        source.topology_id,
        target.topology_id,
        (cell_lineage,),
    )

    witness = lineage_chart_witness(
        source_witness,
        source,
        lineage,
        target,
        created_weights(
            _vertex_stencil(
                source,
                target,
                {
                    199: ((100, Fraction(1, 2)), (101, Fraction(1, 2))),
                    200: ((100, Fraction(1, 2)), (102, Fraction(1, 2))),
                },
            ),
            source,
            target,
        ),
        source_geometry_id="source-geometry",
        geometry_id="target-geometry",
    )

    assert tuple(np.asarray(witness.cell_global_ids)) == (8001, 8002, 8003, 8004)
    assert witness.root_mesh_cell_bindings == (
        first_binding,
        first_binding,
        second_binding,
        second_binding,
    )
    assert tuple(np.asarray(witness.source_root_cell_ids)) == (17, 17, 17, 17)
    assert witness.source_piece_ids == ((0, 701, 9001),) * 4
    assert witness.exact_chart(0) == (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(0)),
        (Fraction(1), Fraction(1)),
    )
    assert witness.exact_chart(2) == (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(1, 2)),
        (Fraction(0), Fraction(1)),
    )
