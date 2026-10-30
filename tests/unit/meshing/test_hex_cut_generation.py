from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import PolynomialComposedCellGeometryElement
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._coordinate_enclosure import Polynomial
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import lagrange_element
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.meshing._hex_cut_generation import (
    _CutCoordinateMap,
    generate_balanced_cut_hexes,
)
from phydrax.meshing._hex_cut_preparation import (
    BalancedCutGridPlan,
    prepare_balanced_cut_grid,
)
from phydrax.meshing._hex_frame_records import HexFrameIntegerLayout
from phydrax.meshing._hex_schedule import NativeHexGridSchedule
from phydrax.meshing._plc_source import PiecewiseLinearComplex


def _simplex_plc_and_fill() -> tuple[PiecewiseLinearComplex, CellMesh]:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    polygons = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    source = PiecewiseLinearComplex(
        points,
        polygons,
        np.arange(len(polygons), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(polygons), 1)),
        ("bulk",),
        vertex_ids=tuple(f"simplex-root-{row}" for row in range(4)),
        vertex_occurrence_paths=((),) * 4,
        facet_ids=tuple(f"simplex-facet-{row}" for row in range(len(polygons))),
        facet_occurrence_paths=((),) * len(polygons),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    fill = CellMesh.from_tetrahedra(
        points, np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    )
    return source, fill


def _two_region_tetrahedral_fill() -> tuple[PiecewiseLinearComplex, CellMesh]:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (1, 2, 3),
            (0, 3, 2),
            (0, 1, 3),
            (2, 1, 4),
            (0, 4, 1),
            (0, 2, 4),
        ),
        dtype=np.int64,
    )
    segments = np.asarray(
        sorted(
            {
                tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
                for face in faces
                for index in range(3)
            }
        ),
        dtype=np.int64,
    )
    source = PiecewiseLinearComplex(
        points,
        tuple(tuple(int(value) for value in face) for face in faces.tolist()),
        np.arange(faces.shape[0], dtype=np.int64),
        np.asarray(((-1, 0),) * 3 + ((-1, 1),) * 3, dtype=np.int64),
        ("upper-material", "lower-material"),
        segments=segments,
        vertex_ids=tuple(f"bipyramid-root-{row}" for row in range(points.shape[0])),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=tuple(f"bipyramid-facet-{row}" for row in range(faces.shape[0])),
        facet_occurrence_paths=((),) * faces.shape[0],
        segment_ids=tuple(f"bipyramid-edge-{row}" for row in range(segments.shape[0])),
        segment_occurrence_paths=((),) * segments.shape[0],
        region_occurrence_paths=((), ()),
    )
    fill = CellMesh.from_tetrahedra(
        points,
        np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int64),
        vertex_global_ids=np.asarray((100, 210, 330, 470, 590), dtype=np.int64),
        cell_global_ids=np.asarray((901, 1703), dtype=np.int64),
    )
    return source, fill


def _nested_tetrahedron_cavity() -> tuple[PiecewiseLinearComplex, CellMesh]:
    outer = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 2.0),
        ),
        dtype=np.float64,
    )
    center = np.asarray((0.5, 0.5, 0.5), dtype=np.float64)
    inner = center + 0.25 * (outer - center)
    points = np.concatenate((outer, inner), axis=0)
    tetra_faces = reference_cell_topology("tetrahedron").entities[2]
    tetrahedra = []
    for face in tetra_faces:
        a, b, c = sorted(int(value) for value in face)
        ai, bi, ci = a + 4, b + 4, c + 4
        for row in ((a, b, c, ci), (a, b, bi, ci), (a, ai, bi, ci)):
            vertices = tuple(int(value) for value in row)
            origin = points[vertices[0]]
            determinant = float(
                np.linalg.det(
                    np.stack(
                        tuple(points[vertex] - origin for vertex in vertices[1:]),
                        axis=1,
                    )
                )
            )
            assert determinant != 0.0
            tetrahedra.append(
                vertices
                if determinant > 0.0
                else (vertices[1], vertices[0], vertices[2], vertices[3])
            )

    local_faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    occurrences: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for tetrahedron in tetrahedra:
        for face in local_faces:
            cycle = (
                tetrahedron[face[0]],
                tetrahedron[face[1]],
                tetrahedron[face[2]],
            )
            ordered = sorted(cycle)
            key = (ordered[0], ordered[1], ordered[2])
            occurrences.setdefault(key, []).append(cycle)
    boundary_faces = tuple(cycles[0] for cycles in occurrences.values() if len(cycles) == 1)
    source = PiecewiseLinearComplex(
        points,
        tuple(tuple(int(value) for value in face) for face in boundary_faces),
        np.arange(len(boundary_faces), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(boundary_faces), 1)),
        ("solid-with-cavity",),
        vertex_ids=tuple(f"cavity-root-{row}" for row in range(points.shape[0])),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=tuple(f"cavity-facet-{row}" for row in range(len(boundary_faces))),
        facet_occurrence_paths=((),) * len(boundary_faces),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    fill = CellMesh.from_tetrahedra(
        points,
        np.asarray(tetrahedra, dtype=np.int64),
        vertex_global_ids=np.asarray((101, 211, 307, 419, 523, 631, 743, 857), dtype=np.int64),
        cell_global_ids=np.arange(len(tetrahedra), dtype=np.int64) + 900,
    )
    return source, fill


def _hex_volume(mesh: CellMesh) -> float:
    reference = np.asarray(reference_cell_topology("hexahedron").vertices)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    coordinates = np.asarray(mesh.coordinates)
    samples = 0.5 + np.asarray((-1.0, 1.0)) / (2.0 * np.sqrt(3.0))
    total = 0.0
    for cell in cells:
        nodes = coordinates[cell]
        for point in product(samples, repeat=3):
            point_values = np.asarray(point, dtype=np.float64)
            factors = np.where(reference == 1.0, point_values, 1.0 - point_values)
            derivative = np.empty((8, 3), dtype=np.float64)
            for axis in range(3):
                other = [row for row in range(3) if row != axis]
                derivative[:, axis] = np.where(
                    reference[:, axis] == 1.0, 1.0, -1.0
                ) * np.prod(factors[:, other], axis=1)
            determinant = float(np.linalg.det(nodes.T @ derivative))
            assert determinant > 0.0
            total += determinant / 8.0
    return total


def test_exact_partial_cut_cells_close_a_positive_tetrahedral_domain() -> None:
    source, fill = _simplex_plc_and_fill()
    schedule = NativeHexGridSchedule(
        "balanced_grid",
        minimum_scaled_jacobian=1.0e-6,
        minimum_mean_ratio=1.0e-6,
        maximum_aspect_ratio=1.0e6,
    )
    plan = prepare_balanced_cut_grid(
        source,
        fill,
        schedule,
        0.6,
        source_id="cut-simplex",
        source_revision="cut-simplex-r1",
    )

    result = generate_balanced_cut_hexes(
        source, fill, np.asarray((0,), dtype=np.int64), plan, schedule
    )

    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert result.mesh.connectivity.cell_count > 1
    assert result.validity.all_certified
    assert np.all(result.cell_regions == 0)
    assert np.all(result.parent_cells == 0)
    assert np.all(np.diff(result.parent_offsets) == 1)
    assert result.geometry.source_map is not None
    assert result.geometry.source_map.source_id == "cut-simplex"
    assert result.geometry.source_map.source_revision == "cut-simplex-r1"
    assert _hex_volume(result.mesh) == pytest.approx(1.0 / 6.0, rel=2.0e-12)
    cut_map = result.geometry.source_map
    assert isinstance(cut_map, _CutCoordinateMap)
    expected_coefficients = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in np.asarray(fill.coordinates, dtype=np.float64)
    )
    assert result.geometry.source_coordinates() == expected_coefficients
    assert cut_map.source_coordinates() == expected_coefficients
    restriction = result.geometry.restriction_source
    assert restriction is not None
    reference_map = restriction.block_parent_reference_maps["cells"]
    assert reference_map is not None
    assert reference_map.source_cell_kind == "tetrahedron"
    assert reference_map.source_reference_exact_vertices is not None
    assert reference_map.source_reference_chart_arguments is not None
    np.testing.assert_array_equal(
        reference_map.target_cell_ids,
        np.arange(result.mesh.blocks[0].cell_count, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        reference_map.source_cell_ids, result.parent_cell_ids
    )
    assert restriction.source_geometry_id == cell_geometry_id(
        cut_map.source_geometry
    )
    assert restriction.source_topology_id == fill.topology_id
    reference_vertices = tuple(
        tuple(Fraction.from_float(float(value)) for value in point)
        for point in np.asarray(
            reference_cell_topology("hexahedron").vertices, dtype=np.float64
        )
    )
    target_cell_ids = np.asarray(result.mesh.blocks[0].global_ids, dtype=np.int64)
    exact_images = result.geometry.evaluate_exact(
        result.mesh,
        target_cell_ids,
        tuple(reference_vertices for _ in target_cell_ids),
    )
    target_vertices = np.asarray(result.mesh.blocks[0].vertices, dtype=np.int64)
    for cell_row, cell_images in enumerate(exact_images):
        for local_vertex, image in enumerate(cell_images):
            assert image == cut_map.exact_vertices[
                int(target_vertices[cell_row, local_vertex])
            ]
    assert result.construction_id



def test_cut_accepts_dimension_qualified_simplex_blocks_without_relabeling() -> None:
    source, _ = _simplex_plc_and_fill()
    vertex_ids = np.asarray((103, 211, 307, 419), dtype=np.int64)
    cell_id = 1907
    fill = CellMesh(
        np.asarray(source.vertices),
        (
            CellBlock(
                "authored-simplex-fill",
                "simplex:3",
                np.asarray(((0, 1, 2, 3),), dtype=np.int64),
                global_ids=np.asarray((cell_id,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=vertex_ids,
    )
    assert isinstance(fill.connectivity, SimplicialConnectivity)
    schedule = NativeHexGridSchedule("balanced_grid")
    plan = prepare_balanced_cut_grid(
        source,
        fill,
        schedule,
        0.6,
        source_id="named-simplex-cut",
        source_revision="named-simplex-cut-r1",
    )

    result = generate_balanced_cut_hexes(
        source, fill, np.asarray((0,), dtype=np.int64), plan, schedule
    )

    assert result.source_mesh_id == fill.mesh_id
    assert np.all(result.parent_cells == 0)
    assert np.all(result.parent_cell_ids == cell_id)
    restriction = result.geometry.restriction_source
    assert restriction is not None
    assert restriction.source_block_names == ("authored-simplex-fill",)
    assert tuple(key[2] for key in result.cut_vertex_keys[: vertex_ids.size]) == tuple(
        int(value) for value in vertex_ids
    )
    assert result.global_embedding.status == "certified"
def test_nonsimple_material_interface_cut_is_closed_by_a_shared_tetrahedral_template() -> None:
    source, fill = _two_region_tetrahedral_fill()
    schedule = NativeHexGridSchedule(
        "balanced_grid",
        minimum_scaled_jacobian=1.0e-8,
        minimum_mean_ratio=1.0e-8,
        maximum_aspect_ratio=1.0e8,
    )
    xy_axis = tuple(Fraction(value, 4) for value in (-1, 1, 3, 5))
    z_axis = tuple(Fraction(value, 4) for value in (-5, -3, -1, 1, 3, 5))
    plan = BalancedCutGridPlan(
        (xy_axis, xy_axis, z_axis),
        source,
        fill,
        schedule,
        0.5,
        source_id="bipyramid-cut",
        source_revision="bipyramid-cut-r1",
    )

    result = generate_balanced_cut_hexes(
        source, fill, np.asarray((0, 1), dtype=np.int64), plan, schedule
    )

    assert result.validity.all_certified
    assert result.global_embedding.status == "certified"
    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert set(result.cell_regions.tolist()) == {0, 1}
    assert np.array_equal(result.cell_regions, result.parent_cells)
    assert set(fill.blocks[0].global_ids.tolist()) == {
        int(fill.blocks[0].global_ids[parent])
        for parent in result.parent_cells.tolist()
    }
    assert np.array_equal(
        result.parent_cell_ids,
        fill.blocks[0].global_ids[result.parent_cells],
    )
    assert result.source_region_ids == source.region_ids
    assert len(result.cut_vertex_keys) == result.mesh.coordinates.shape[0]
    assert any(key[1] in ("edge", "face") for key in result.cut_vertex_keys)
    assert tuple(key[2] for key in result.cut_vertex_keys[: fill.vertex_global_ids.size]) == tuple(
        int(value) for value in fill.vertex_global_ids
    )
    assert result.source_region_occurrence_paths == source.region_occurrence_paths
    assert result.source_plc_vertex_ids == source.vertex_ids
    assert result.source_plc_vertex_occurrence_paths == source.vertex_occurrence_paths
    assert result.source_plc_edge_ids == source.plc_edge_ids
    assert result.source_plc_edge_occurrence_paths == source.plc_edge_occurrence_paths
    assert result.source_plc_facet_ids == source.facet_ids
    assert result.source_plc_facet_occurrence_paths == source.facet_occurrence_paths
    assert np.array_equal(
        result.mesh.vertex_global_ids[: fill.vertex_global_ids.size],
        fill.vertex_global_ids,
    )
    assert np.all(result.integer_blocks == 0)
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, HexahedralConnectivity)
    face_incidence = np.asarray(connectivity.face_cell_counts)
    assert np.all((face_incidence == 1) | (face_incidence == 2))
    assert _hex_volume(result.mesh) == pytest.approx(1.0 / 3.0, rel=3.0e-12)
    assert result.source_complex_id == source.complex_id
    cut_map = result.geometry.source_map
    assert isinstance(cut_map, _CutCoordinateMap)
    np.testing.assert_array_equal(cut_map.cell_regions, np.asarray((0, 1), dtype=np.int64))
    restored_map = _CutCoordinateMap.from_stored_fields(cut_map, result.mesh)
    np.testing.assert_array_equal(restored_map.cell_regions, cut_map.cell_regions)
    assert np.any(
        np.all(
            np.asarray(result.mesh.coordinates)
            == np.asarray((0.25, 0.75, 0.0), dtype=np.float64),
            axis=1,
        )
    )


def test_frame_grid_cut_binds_the_exact_integer_block_layout() -> None:
    source, fill = _simplex_plc_and_fill()
    schedule = NativeHexGridSchedule(
        "frame_grid",
        minimum_scaled_jacobian=1.0e-8,
        minimum_mean_ratio=1.0e-8,
        maximum_aspect_ratio=1.0e8,
    )
    axis = tuple(Fraction(value, 4) for value in (-1, 1, 3, 5))
    layout = HexFrameIntegerLayout(
        np.asarray(fill.blocks[0].global_ids, dtype=np.int64),
        np.zeros((1,), dtype=np.int64),
        np.zeros((1, 3), dtype=np.int64),
        np.eye(3, dtype=np.float64)[None, :, :],
        np.zeros((1, 3), dtype=np.float64),
        ((axis, axis, axis),),
        np.empty((0, 4), dtype=np.int64),
        np.empty((0, 3, 3), dtype=np.int64),
        np.empty((0, 3), dtype=np.int64),
        np.ones((1,), dtype=np.int8),
        source.complex_id,
        fill.mesh_id,
        fill.topology_id,
        schedule.schedule_id,
        "simplex-frame-field",
    )
    plan = prepare_balanced_cut_grid(
        source,
        fill,
        schedule,
        0.5,
        source_id="frame-cut-simplex",
        source_revision="frame-cut-simplex-r1",
        frame_layout=layout,
    )

    result = generate_balanced_cut_hexes(
        source, fill, np.asarray((0,), dtype=np.int64), plan, schedule
    )

    assert plan.frame_layout_id == layout.layout_id
    assert np.all(result.integer_blocks == 0)
    assert result.validity.all_certified
    assert _hex_volume(result.mesh) == pytest.approx(1.0 / 6.0, rel=3.0e-12)


def test_curved_cut_refusal_names_missing_exact_arrangement_witnesses() -> None:
    source, fill = _simplex_plc_and_fill()
    element = lagrange_element("tetrahedron", 2)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinates = reference_nodes.copy()
    coordinates[:, 2] += 0.05 * reference_nodes[:, 0] * reference_nodes[:, 1]
    geometry = CellGeometrySpec(
        {"tetrahedra": element},
        {"tetrahedra": np.arange(element.local_dof_count, dtype=np.int64)[None, :]},
        coordinates,
    )
    schedule = NativeHexGridSchedule("balanced_grid")
    plan = prepare_balanced_cut_grid(
        source,
        fill,
        schedule,
        0.5,
        source_id="curved-cut-simplex",
        source_revision="curved-cut-simplex-r1",
    )

    with pytest.raises(
        ValueError,
        match="internal grid-plane intersections have no linear root witness",
    ):
        generate_balanced_cut_hexes(
            source,
            fill,
            np.asarray((0,), dtype=np.int64),
            plan,
            schedule,
            source_geometry=geometry,
        )


def _thin_curved_simplex() -> tuple[PiecewiseLinearComplex, CellMesh, CellGeometrySpec]:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 0.1),
        ),
        dtype=np.float64,
    )
    polygons = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    source = PiecewiseLinearComplex(
        points,
        polygons,
        np.arange(len(polygons), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(polygons), 1)),
        ("bulk",),
        vertex_ids=tuple(f"thin-root-{row}" for row in range(4)),
        vertex_occurrence_paths=((),) * 4,
        facet_ids=tuple(f"thin-facet-{row}" for row in range(len(polygons))),
        facet_occurrence_paths=((),) * len(polygons),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    fill = CellMesh.from_tetrahedra(
        points, np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    )
    source_element = lagrange_element("tetrahedron", 1)
    chart_arguments: tuple[Polynomial, ...] = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {
            (0, 0, 1): Fraction(1),
            (1, 1, 0): Fraction(1),
            (2, 1, 0): Fraction(-1),
            (1, 2, 0): Fraction(-1),
            (1, 1, 1): Fraction(-1),
        },
    )
    element = PolynomialComposedCellGeometryElement(
        source_element, "tetrahedron", chart_arguments
    )
    geometry = CellGeometrySpec(
        {"tetrahedra": element},
        {"tetrahedra": np.arange(4, dtype=np.int64)[None, :]},
        points,
    )
    return source, fill, geometry


def test_curved_feature_intersections_are_cut_in_the_exact_carrier_chart() -> None:
    source, fill, geometry = _thin_curved_simplex()
    schedule = NativeHexGridSchedule(
        "balanced_grid",
        minimum_scaled_jacobian=1.0e-10,
        minimum_mean_ratio=1.0e-10,
        maximum_aspect_ratio=1.0e10,
    )
    plan = prepare_balanced_cut_grid(
        source,
        fill,
        schedule,
        0.8,
        source_id="curved-feature-cut",
        source_revision="curved-feature-cut-r1",
    )

    result = generate_balanced_cut_hexes(
        source,
        fill,
        np.asarray((0,), dtype=np.int64),
        plan,
        schedule,
        source_geometry=geometry,
    )

    assert result.validity.all_certified
    assert result.global_embedding.status == "certified"
    assert result.geometry.restriction_source is not None
    assert result.geometry.source_map is not None
    assert result.source_reference_coverage == (Fraction(1, 6),)
    assert _hex_volume(result.mesh) > 0.0
    curved_boundary_nodes = [
        row
        for row, support in enumerate(result.vertex_supports.tolist())
        if set(value for value in support if value >= 0) == {0, 1, 2}
    ]
    assert any(result.mesh.coordinates[row, 2] > 0.0 for row in curved_boundary_nodes)



def test_balanced_cut_all_hex_closure_covers_a_nonsweepable_tetrahedral_cavity() -> None:
    source, fill = _nested_tetrahedron_cavity()
    schedule = NativeHexGridSchedule(
        "balanced_grid",
        minimum_scaled_jacobian=1.0e-8,
        minimum_mean_ratio=1.0e-8,
        maximum_aspect_ratio=1.0e8,
    )
    plan = prepare_balanced_cut_grid(
        source,
        fill,
        schedule,
        0.8,
        source_id="nested-tetra-cavity",
        source_revision="nested-tetra-cavity-r1",
    )

    result = generate_balanced_cut_hexes(
        source,
        fill,
        np.zeros((fill.connectivity.cell_count,), dtype=np.int64),
        plan,
        schedule,
    )

    offsets = np.asarray(source.polygon_offsets, dtype=np.int64)
    packed = np.asarray(source.polygon_vertices, dtype=np.int64)
    facets = np.stack(
        [packed[offsets[row] : offsets[row + 1]] for row in range(len(offsets) - 1)]
    )
    facet_regions = np.asarray(source.facet_regions, dtype=np.int64)[
        np.asarray(source.polygon_facets, dtype=np.int64)
    ]
    domain = PiecewiseLinearDomain(
        np.asarray(source.vertices),
        facets,
        facet_regions,
        source.region_ids,
        source_id="nested-tetra-cavity-domain",
    )
    exact_region_measures = domain.exact_region_measures()
    assert exact_region_measures == (Fraction(21, 16),)
    expected_volume = float(exact_region_measures[0])
    assert result.validity.all_certified
    assert result.global_embedding.status == "certified"
    assert {block.cell_kind for block in result.mesh.blocks} == {"hexahedron"}
    assert result.mesh.connectivity.cell_count > fill.connectivity.cell_count
    assert np.all(result.cell_regions == 0)
    assert np.all(np.diff(result.parent_offsets) == 1)
    assert np.array_equal(
        result.mesh.vertex_global_ids[: fill.vertex_global_ids.size],
        fill.vertex_global_ids,
    )
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, HexahedralConnectivity)
    assert np.all((connectivity.face_cell_counts == 1) | (connectivity.face_cell_counts == 2))
    assert _hex_volume(result.mesh) == pytest.approx(expected_volume, rel=5.0e-12)
    assert result.source_complex_id == source.complex_id
