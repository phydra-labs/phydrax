from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import PerCellRestrictedCellGeometryElement
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._coordinate_enclosure import axes
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import lagrange_element
from phydrax.meshing._hex_dominant import extract_hex_dominant


def _two_tetrahedra() -> CellMesh:
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
    return CellMesh.from_tetrahedra(
        points,
        np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int64),
    )


def test_hex_dominant_accepts_dimension_qualified_simplex_source_blocks() -> None:
    source = _two_tetrahedra()
    cell_ids = np.asarray((901, 1703), dtype=np.int64)
    block = CellBlock(
        "authored-simplex-source",
        "simplex:3",
        np.asarray(source.blocks[0].vertices, dtype=np.int64),
        global_ids=cell_ids,
    )
    named_source = CellMesh(
        np.asarray(source.coordinates, dtype=np.float64),
        (block,),
        vertex_global_ids=np.asarray(source.vertex_global_ids, dtype=np.int64),
    )
    assert isinstance(named_source.connectivity, SimplicialConnectivity)

    result = extract_hex_dominant(named_source, np.asarray((901,), dtype=np.int64))

    assert result.global_embedding.status == "certified"
    assert np.array_equal(result.parent_cell_ids, cell_ids[result.parent_cells])
    restriction = result.geometry.restriction_source
    assert restriction is not None
    assert restriction.source_block_names == ("authored-simplex-source",)


def _simplex_volume(points: np.ndarray, cell: np.ndarray) -> float:
    origin = points[int(cell[0])]
    matrix = np.stack(tuple(points[int(vertex)] - origin for vertex in cell[1:]), axis=1)
    return abs(float(np.linalg.det(matrix))) / 6.0


def _pyramid_volume(points: np.ndarray, cell: np.ndarray) -> float:
    base = tuple(int(value) for value in cell[:4])
    apex = int(cell[4])
    first = _simplex_volume(points, np.asarray((base[0], base[1], base[2], apex)))
    second = _simplex_volume(points, np.asarray((base[0], base[2], base[3], apex)))
    return first + second


def _hexahedron_volume(points: np.ndarray, cell: np.ndarray) -> float:
    reference = np.asarray(reference_cell_topology("hexahedron").vertices)
    nodes = points[np.asarray(cell, dtype=np.int64)]
    abscissas = 0.5 + np.asarray((-1.0, 1.0)) / (2.0 * np.sqrt(3.0))
    total = 0.0
    for point in product(abscissas, repeat=3):
        point_values = np.asarray(point, dtype=np.float64)
        factors = np.where(reference == 1.0, point_values, 1.0 - point_values)
        derivative = np.empty((8, 3), dtype=np.float64)
        for axis in range(3):
            other = [value for value in range(3) if value != axis]
            derivative[:, axis] = np.where(
                reference[:, axis] == 1.0, 1.0, -1.0
            ) * np.prod(factors[:, other], axis=1)
        determinant = float(np.linalg.det(nodes.T @ derivative))
        assert determinant > 0.0
        total += determinant / 8.0
    return total


def test_hex_dominant_core_has_conforming_pyramid_and_tet_closure() -> None:
    source = _two_tetrahedra()
    result = extract_hex_dominant(source, np.asarray((0,), dtype=np.int64))
    points = np.asarray(result.mesh.coordinates)

    assert result.hex_cell_count == 4
    assert result.pyramid_cell_count > 0
    assert result.tetrahedron_cell_count > 0
    assert {block.cell_kind for block in result.mesh.blocks} == {
        "hexahedron",
        "pyramid",
        "tetrahedron",
    }
    assert result.validity.all_certified
    assert result.global_embedding.status == "certified"
    assert result.parent_offsets.shape == (result.mesh.connectivity.cell_count + 1,)
    assert np.all(np.diff(result.parent_offsets) == 1)
    assert np.all(np.isin(result.parent_cells, np.asarray((0, 1))))
    source_cell_ids = np.asarray(source.blocks[0].global_ids, dtype=np.int64)
    assert np.array_equal(result.parent_cell_ids, source_cell_ids[result.parent_cells])

    volume = 0.0
    for block in result.mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int64)
        if block.cell_kind == "hexahedron":
            volume += sum(_hexahedron_volume(points, row) for row in cells)
        elif block.cell_kind == "pyramid":
            volume += sum(_pyramid_volume(points, row) for row in cells)
        else:
            volume += sum(_simplex_volume(points, row) for row in cells)
    source_points = np.asarray(source.coordinates)
    source_cells = np.asarray(source.blocks[0].vertices, dtype=np.int64)
    expected = sum(_simplex_volume(source_points, row) for row in source_cells)
    assert volume == pytest.approx(expected, rel=1.0e-12)

    with pytest.raises(ValueError, match="genuine all-hex generation route"):
        extract_hex_dominant(source, np.asarray((0, 1), dtype=np.int64))


def test_curved_source_map_is_composed_through_mixed_families() -> None:
    source = _two_tetrahedra()
    element = lagrange_element("tetrahedron", 4)
    reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    source_cells = np.asarray(source.blocks[0].vertices, dtype=np.int64)
    geometry_nodes = []
    for cell_row, cell in enumerate(source_cells):
        vertices = np.asarray(source.coordinates)[cell]
        origin = vertices[0]
        physical = origin + reference_nodes @ (vertices[1:] - origin)
        barycentric = np.column_stack(
            (
                1.0 - np.sum(reference_nodes, axis=1),
                reference_nodes,
            )
        )
        if cell_row == 0:
            physical[:, 2] += 0.01 * np.prod(barycentric, axis=1)
        geometry_nodes.append(physical)
    coordinates = np.concatenate(geometry_nodes, axis=0)
    routes = np.arange(coordinates.shape[0], dtype=np.int64).reshape(
        2, element.local_dof_count
    )
    geometry = CellGeometrySpec(
        {source.blocks[0].name: element},
        {source.blocks[0].name: routes},
        coordinates,
    )

    result = extract_hex_dominant(
        source,
        np.asarray((0,), dtype=np.int64),
        source_geometry=geometry,
    )

    assert result.validity.all_certified
    assert result.global_embedding.status == "certified"
    assert np.array_equal(
        result.parent_cell_ids,
        np.asarray(source.blocks[0].global_ids, dtype=np.int64)[result.parent_cells],
    )
    assert result.source_validity_id
    assert (
        len(result.reference_chart_determinant_bounds)
        == result.mesh.connectivity.cell_count
    )
    assert len(result.source_reference_coverage) == source.connectivity.cell_count
    assert result.geometry.restriction_source is not None
    hex_block = next(
        block for block in result.mesh.blocks if block.cell_kind == "hexahedron"
    )
    identifier = np.asarray(hex_block.global_ids, dtype=np.int64)[:1]
    reference = np.asarray(((0.5, 0.5, 0.5),), dtype=np.float64)
    mapped = result.geometry.evaluate(result.mesh, identifier, reference)
    affine = CellGeometrySpec.affine(result.mesh).evaluate(
        result.mesh, identifier, reference
    )
    assert not np.allclose(mapped, affine, rtol=0.0, atol=1.0e-12)


def test_mixed_closure_flattens_nested_per_cell_root_charts() -> None:
    source = _two_tetrahedra()
    root = lagrange_element("tetrahedron", 1)
    block = source.blocks[0]
    identifiers = tuple(int(value) for value in block.global_ids)
    identity = axes(3)
    element = PerCellRestrictedCellGeometryElement(
        root,
        "tetrahedron",
        identifiers,
        identifiers,
        (identity, identity),
    )
    shear = np.asarray(
        (
            (1.0, 0.2, 0.0),
            (0.0, 1.0, 0.1),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    translated = np.asarray(source.coordinates) @ shear.T + np.asarray(
        (2.0, -1.0, 0.5), dtype=np.float64
    )
    geometry = CellGeometrySpec(
        {block.name: element},
        {block.name: np.asarray(block.vertices, dtype=np.int64)},
        translated,
    )

    result = extract_hex_dominant(
        source,
        np.asarray((0,), dtype=np.int64),
        source_geometry=geometry,
    )

    assert result.validity.all_certified
    assert result.source_reference_coverage == (
        Fraction(1, 6),
        Fraction(1, 6),
    )
    assert result.geometry.restriction_source is not None
    assert np.all(np.isin(result.parent_cells, np.asarray((0, 1), dtype=np.int64)))
