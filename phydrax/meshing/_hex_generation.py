# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Hex generation from tetrahedral duals and mapped root restrictions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import final, Literal

import equinox as eqx
import numpy as np

from .. import _meshcore as meshcore_owner
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_complex import PolyhedralConnectivity, TetrahedralConnectivity
from ..discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_effective_basis,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import SimplicialConnectivity
from ..discretization._coordinate_enclosure import (
    compose_expression,
    derivative,
    evaluate_expression,
    Expression,
    linear_combinations,
    multiply,
    Polynomial,
    polynomial_bounds,
    scale,
    sum_polynomials,
)
from ..discretization._exact_algebraic_geometry import ExactScalar, RadicalElement
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._nested_reference_types import (
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
)
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..linalg._exact import solve_exact
from ..typing import checked
from ._dual_extraction import (
    _affine_geometry_matches,
    _RestrictedCoordinateMap,
    DualExtraction,
    fresh_vertex_global_ids,
)
from ._hex_frame_records import PreparedSourceHexFrame, SourceBlockConnection
from ._hex_schedule import (
    NativeHexGridRoute as _NativeHexGridRoute,
    NativeHexGridSchedule as _NativeHexGridSchedule,
)


def _is_tetrahedron_family(cell_kind: str, /) -> bool:
    return reference_cell_topology(cell_kind).family == "tetrahedron"


def _dual_hex_reference_corners(
    hex_vertices: list[int] | np.ndarray,
    parent_vertices: np.ndarray,
    supports: np.ndarray,
    /,
) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    """Exact parent-tetrahedron reference corners of one vertex-star dual hex.

    Each dual vertex is the barycenter of its source support (a vertex, edge,
    face or the cell), so its parent reference point is the mean of the
    supporting reference tetrahedron vertices.
    """
    reference = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in reference_cell_topology("tetrahedron").vertices
    )
    local_by_vertex = {
        int(vertex): reference[local]
        for local, vertex in enumerate(np.asarray(parent_vertices).tolist())
    }
    corners: list[tuple[Fraction, Fraction, Fraction]] = []
    for target_vertex in np.asarray(hex_vertices).tolist():
        support = tuple(
            int(value)
            for value in supports[int(target_vertex)].tolist()
            if int(value) >= 0
        )
        if not support or any(vertex not in local_by_vertex for vertex in support):
            raise ValueError(
                "A dual hexahedron support must lie in its retained source tetrahedron."
            )
        point = tuple(
            sum((local_by_vertex[vertex][axis] for vertex in support), Fraction(0))
            / len(support)
            for axis in range(3)
        )
        corners.append((point[0], point[1], point[2]))
    return tuple(corners)


def _trilinear_chart(
    corners: tuple[tuple[Fraction, Fraction, Fraction], ...],
    /,
) -> tuple[Polynomial, ...]:
    if len(corners) != 8:
        raise ValueError("A hexahedral source chart requires eight reference corners.")
    reference_vertices = tuple(
        tuple(int(value) for value in point)
        for point in reference_cell_topology("hexahedron").vertices
    )
    by_bits = dict(zip(reference_vertices, corners, strict=True))
    result: list[Polynomial] = []
    for axis in range(3):
        polynomial: Polynomial = {}
        for exponent in np.ndindex((2, 2, 2)):
            coefficient = Fraction(0)
            for bits, point in by_bits.items():
                if all(bits[row] <= exponent[row] for row in range(3)):
                    parity = sum(exponent) - sum(bits)
                    coefficient += (-1 if parity % 2 else 1) * point[axis]
            if coefficient:
                powers: tuple[int, ...] = tuple(exponent)
                polynomial[powers] = coefficient
        result.append(polynomial)
    return tuple(result)


def _evaluate_trilinear_chart(
    polynomial: Polynomial,
    point: tuple[Fraction, Fraction, Fraction],
    /,
) -> Fraction:
    value = Fraction(0)
    for powers, coefficient in polynomial.items():
        if not isinstance(coefficient, Fraction):
            raise TypeError("A dual trilinear reference chart must be rational.")
        value += (
            coefficient
            * point[0] ** powers[0]
            * point[1] ** powers[1]
            * point[2] ** powers[2]
        )
    return value


def _compose_restricted_dual_chart(
    element: ScalarCoordinateElement,
    source_cell_row: int,
    source_cell_id: int,
    chart: tuple[Expression, ...],
    /,
) -> tuple[ScalarCoordinateElement, int, tuple[Expression, ...]]:
    """Flatten per-cell source restrictions into one exact root chart."""
    current: ScalarCoordinateElement = element
    row = int(source_cell_row)
    cell_id = int(source_cell_id)
    arguments = chart
    while isinstance(current, PerCellRestrictedCellGeometryElement):
        if not 0 <= row < len(current.source_cell_ids):
            raise ValueError("A source restriction row is outside its exact chart bank.")
        parent_arguments = current.chart_arguments(row)
        if len(parent_arguments) != len(arguments):
            raise ValueError("Nested source chart dimensions are incompatible.")
        arguments = tuple(
            compose_expression(parent, arguments) for parent in parent_arguments
        )
        cell_id = current.source_cell_ids[row]
        current = current.source_element
        if isinstance(current, PerCellRestrictedCellGeometryElement):
            rows = {
                identifier: index
                for index, identifier in enumerate(current.target_cell_ids)
            }
            parent_row = rows.get(cell_id)
            if parent_row is None:
                raise ValueError(
                    "Nested source restriction does not contain its exact parent cell ID."
                )
            row = parent_row
    return current, cell_id, arguments


def _compose_tetrahedral_dual_geometry(
    source: CellMesh,
    target: CellMesh,
    supports: np.ndarray,
    parent_offsets: np.ndarray,
    parent_cells: np.ndarray,
    source_geometry: CellGeometrySpec,
    /,
) -> tuple[CellMesh, CellGeometrySpec, np.ndarray]:
    """Compose each vertex-star hex chart through its exact source tetrahedron."""
    source_block = source.blocks[0]
    source_cells = np.asarray(source_block.vertices, dtype=np.int64)
    source_cell_ids = np.asarray(source_block.global_ids, dtype=np.int64)
    elements, source_routes, _ = source_geometry.resolve(source)
    if len(elements) != 1 or len(target.blocks) != 1:
        raise ValueError(
            "Curved dual hex composition requires one source and target block."
        )
    source_element = elements[0]
    if not isinstance(source_element, ScalarCoordinateElement):
        raise ValueError("Curved dual hex composition requires a scalar chart element.")
    source_routes_host = np.asarray(source_routes[0], dtype=np.int64)
    reference_hexes: tuple[tuple[Fraction, Fraction, Fraction], ...] = tuple(
        (
            Fraction(float(point[0])),
            Fraction(float(point[1])),
            Fraction(float(point[2])),
        )
        for point in reference_cell_topology("hexahedron").vertices
    )
    block = target.blocks[0]
    hexes = np.asarray(block.vertices, dtype=np.int64)
    target_cell_ids = np.asarray(block.global_ids, dtype=np.int64)
    parent_rows = np.empty((hexes.shape[0],), dtype=np.int64)
    parent_vertex_ids = np.empty((hexes.shape[0], 4), dtype=np.int64)
    chart_groups: list[list[int]] = [[], [], [], []]
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    current_charts: list[tuple[Polynomial, ...]] = []
    root_charts: list[tuple[Expression, ...]] = []
    root_cell_ids = np.empty((hexes.shape[0],), dtype=np.int64)
    root_element: ScalarCoordinateElement | None = None
    for row, hex_ in enumerate(hexes.tolist()):
        start, stop = int(parent_offsets[row]), int(parent_offsets[row + 1])
        if stop - start != 1:
            raise ValueError(
                "Each dual hexahedron must retain one source tetrahedron chart."
            )
        parent = int(parent_cells[start])
        if not 0 <= parent < source_cells.shape[0]:
            raise ValueError("Dual hexahedron references an absent source tetrahedron.")
        parent_rows[row] = parent
        vertices = source_cells[parent]
        corners = _dual_hex_reference_corners(hex_, vertices, supports)
        current_chart = _trilinear_chart(tuple(corners))
        root_candidate, root_cell_id, root_chart = _compose_restricted_dual_chart(
            source_element,
            parent,
            int(source_cell_ids[parent]),
            current_chart,
        )
        if root_element is None:
            root_element = root_candidate
        elif root_element is not root_candidate:
            raise ValueError(
                "Dual hexes in one source block require one common root element."
            )
        current_charts.append(current_chart)
        root_charts.append(root_chart)
        root_cell_ids[row] = root_cell_id
        singleton_sources = [
            int(supports[int(target_vertex), 0])
            for target_vertex in hex_
            if int(supports[int(target_vertex), 0]) >= 0
            and int(np.count_nonzero(supports[int(target_vertex)] >= 0)) == 1
        ]
        if len(singleton_sources) != 1:
            raise ValueError(
                "Each dual hex must contain exactly one source tetrahedron vertex."
            )
        chart_groups[vertices.tolist().index(singleton_sources[0])].append(row)
        parent_vertex_ids[row] = source_vertex_ids[vertices]

    target_points = np.asarray(target.coordinates, dtype=np.float64).copy()
    exact_at_vertex: dict[int, tuple[object, ...]] = {}
    for row, hex_ in enumerate(hexes.tolist()):
        chart = current_charts[row]
        source_references = tuple(
            tuple(_evaluate_trilinear_chart(polynomial, point) for polynomial in chart)
            for point in reference_hexes
        )
        exact_coordinates = source_geometry.evaluate_exact(
            source,
            np.asarray((int(source_cell_ids[parent_rows[row]]),), dtype=np.int64),
            (source_references,),
        )[0]
        for vertex, exact_coordinate in zip(hex_, exact_coordinates, strict=True):
            previous = exact_at_vertex.get(int(vertex))
            if previous is not None and previous != exact_coordinate:
                raise ValueError(
                    "Adjacent source tetrahedra disagree on an exact dual hex vertex."
                )
            exact_at_vertex[int(vertex)] = exact_coordinate
            target_points[int(vertex)] = np.asarray(
                [float(value) for value in exact_coordinate], dtype=np.float64
            )

    cell_order = np.asarray(
        [row for group in chart_groups for row in group], dtype=np.int64
    )
    if (
        cell_order.size != hexes.shape[0]
        or np.unique(cell_order).size != cell_order.size
        or any(len(group) != source_cells.shape[0] for group in chart_groups)
    ):
        raise ValueError(
            "Dual hexahedron chart blocks do not cover each source tet once."
        )

    if root_element is None:
        raise ValueError(
            "Curved dual hex composition requires a retained root coordinate element."
        )
    blocks = []
    elements_by_block = {}
    routes_by_block = {}
    parent_cells_by_block = {}
    parent_vertices_by_block = {}
    source_blocks_by_block = {}
    for corner, group in enumerate(chart_groups):
        rows = np.asarray(group, dtype=np.int64)
        name = f"{source_block.name}:dual-chart-{corner}"
        blocks.append(
            CellBlock(
                name,
                "hexahedron",
                hexes[rows],
                global_ids=target_cell_ids[rows],
            )
        )
        row_parents = parent_rows[rows]
        ids = source_cell_ids[row_parents]
        elements_by_block[name] = PerCellRestrictedCellGeometryElement(
            root_element,
            "hexahedron",
            tuple(int(value) for value in target_cell_ids[rows].tolist()),
            tuple(int(value) for value in root_cell_ids[rows].tolist()),
            tuple(root_charts[int(row)] for row in rows.tolist()),
        )
        routes_by_block[name] = source_routes_host[row_parents]
        parent_cells_by_block[name] = ids
        parent_vertices_by_block[name] = parent_vertex_ids[rows]
        source_blocks_by_block[name] = source_block.name
    target = CellMesh(
        target_points,
        tuple(blocks),
        vertex_global_ids=np.asarray(target.vertex_global_ids, dtype=np.int64),
        numeric_version=target.numeric_version,
    )
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        source.topology_id,
        parent_cells_by_block,
        parent_vertices_by_block,
        block_source_blocks=source_blocks_by_block,
    )
    restricted_map = (
        None
        if source_geometry.source_map is None
        else _RestrictedCoordinateMap(source, source_geometry, target)
    )
    geometry = CellGeometrySpec(
        elements_by_block,
        routes_by_block,
        source_geometry.coordinates,
        source_map=restricted_map,
        source_mesh=target if restricted_map is not None else None,
        restriction_source=restriction,
    )
    return target, geometry, cell_order


def _compose_multiblock_tetrahedral_dual_geometry(
    source: CellMesh,
    target: CellMesh,
    supports: np.ndarray,
    parent_offsets: np.ndarray,
    parent_cells: np.ndarray,
    source_geometry: CellGeometrySpec,
    /,
) -> tuple[CellMesh, CellGeometrySpec, np.ndarray]:
    """Compose dual charts without erasing source-block/material boundaries."""
    source_cells_by_block = tuple(
        np.asarray(block.vertices, dtype=np.int64) for block in source.blocks
    )
    source_offsets = np.concatenate(
        (
            np.asarray((0,), dtype=np.int64),
            np.cumsum(
                np.asarray(
                    [cells.shape[0] for cells in source_cells_by_block],
                    dtype=np.int64,
                )
            ),
        )
    )
    elements, source_routes, _ = source_geometry.resolve(source)
    scalar_elements = tuple(
        element for element in elements if isinstance(element, ScalarCoordinateElement)
    )
    if (
        len(scalar_elements) != len(source.blocks)
        or len(target.blocks) != len(source.blocks)
        or tuple(block.name for block in target.blocks)
        != tuple(block.name for block in source.blocks)
        or any(
            not _is_tetrahedron_family(element.cell_kind) for element in scalar_elements
        )
    ):
        raise ValueError(
            "Multi-block dual restriction requires one scalar tetrahedral chart per source block."
        )
    reference_hexes: tuple[tuple[Fraction, Fraction, Fraction], ...] = tuple(
        (
            Fraction(float(point[0])),
            Fraction(float(point[1])),
            Fraction(float(point[2])),
        )
        for point in reference_cell_topology("hexahedron").vertices
    )
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    target_points = np.asarray(target.coordinates, dtype=np.float64).copy()
    exact_at_vertex: dict[int, tuple[object, ...]] = {}
    elements_by_block = {}
    routes_by_block = {}
    parent_cells_by_block = {}
    parent_vertices_by_block = {}
    source_blocks_by_block = {}
    target_order: list[int] = []
    source_cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in source.blocks)
    )
    for block_index, (source_block, target_block, source_cells_block) in enumerate(
        zip(source.blocks, target.blocks, source_cells_by_block, strict=True)
    ):
        target_hexes = np.asarray(target_block.vertices, dtype=np.int64)
        target_cell_ids = np.asarray(target_block.global_ids, dtype=np.int64)
        current_charts: list[tuple[Polynomial, ...]] = []
        root_charts: list[tuple[Expression, ...]] = []
        root_ids = np.empty((target_hexes.shape[0],), dtype=np.int64)
        root_element: ScalarCoordinateElement | None = None
        parent_local_rows = np.empty((target_hexes.shape[0],), dtype=np.int64)
        parent_ids = np.empty((target_hexes.shape[0],), dtype=np.int64)
        parent_vertex_ids = np.empty((target_hexes.shape[0], 4), dtype=np.int64)
        for row, (hex_, target_cell_id) in enumerate(
            zip(target_hexes.tolist(), target_cell_ids.tolist(), strict=True)
        ):
            if not 0 <= target_cell_id < parent_offsets.size - 1:
                raise ValueError("Dual target cell ID is outside its parent table.")
            start, stop = (
                int(parent_offsets[target_cell_id]),
                int(parent_offsets[target_cell_id + 1]),
            )
            if stop - start != 1:
                raise ValueError(
                    "Each dual hexahedron must retain one source tetrahedron chart."
                )
            parent = int(parent_cells[start])
            if (
                not int(source_offsets[block_index])
                <= parent
                < int(source_offsets[block_index + 1])
            ):
                raise ValueError(
                    "Multi-block dual target must retain the corresponding source block."
                )
            local_parent = parent - int(source_offsets[block_index])
            source_cell = source_cells_block[local_parent]
            corners = _dual_hex_reference_corners(hex_, source_cell, supports)
            current_chart = _trilinear_chart(tuple(corners))
            root_candidate, root_cell_id, root_chart = _compose_restricted_dual_chart(
                scalar_elements[block_index],
                local_parent,
                int(source_cell_ids[parent]),
                current_chart,
            )
            if root_element is None:
                root_element = root_candidate
            elif root_element is not root_candidate:
                raise ValueError(
                    "A target source block requires one common root coordinate element."
                )
            current_charts.append(current_chart)
            root_charts.append(root_chart)
            root_ids[row] = root_cell_id
            parent_local_rows[row] = local_parent
            parent_ids[row] = source_cell_ids[parent]
            parent_vertex_ids[row] = source_vertex_ids[source_cell]
            target_order.append(target_cell_id)

            source_references = tuple(
                tuple(
                    _evaluate_trilinear_chart(polynomial, point)
                    for polynomial in current_charts[-1]
                )
                for point in reference_hexes
            )
            exact_coordinates = source_geometry.evaluate_exact(
                source,
                np.asarray((parent_ids[row],), dtype=np.int64),
                (source_references,),
            )[0]
            for vertex, exact_coordinate in zip(hex_, exact_coordinates, strict=True):
                previous = exact_at_vertex.get(int(vertex))
                if previous is not None and previous != exact_coordinate:
                    raise ValueError(
                        "Adjacent source charts disagree on an exact dual vertex."
                    )
                exact_at_vertex[int(vertex)] = exact_coordinate
                target_points[int(vertex)] = np.asarray(
                    [float(value) for value in exact_coordinate], dtype=np.float64
                )

        if root_element is None:
            raise ValueError(
                "Multi-block dual restriction requires a retained root coordinate element."
            )
        target_ids_tuple = tuple(int(value) for value in target_cell_ids.tolist())
        elements_by_block[target_block.name] = PerCellRestrictedCellGeometryElement(
            root_element,
            "hexahedron",
            target_ids_tuple,
            tuple(int(value) for value in root_ids.tolist()),
            tuple(root_charts),
        )
        routes_by_block[target_block.name] = np.asarray(
            source_routes[block_index], dtype=np.int64
        )[parent_local_rows]
        parent_cells_by_block[target_block.name] = parent_ids
        parent_vertices_by_block[target_block.name] = parent_vertex_ids
        source_blocks_by_block[target_block.name] = source_block.name

    target_order_array = np.asarray(target_order, dtype=np.int64)
    source_cell_count = parent_offsets.size - 1
    if target_order_array.shape != (source_cell_count,) or not np.array_equal(
        np.sort(target_order_array),
        np.arange(source_cell_count, dtype=np.int64),
    ):
        raise ValueError(
            "Every multi-block dual cell must retain one unique target identity."
        )

    target = CellMesh(
        target_points,
        target.blocks,
        vertex_global_ids=np.asarray(target.vertex_global_ids, dtype=np.int64),
        numeric_version=target.numeric_version,
    )
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        source.topology_id,
        parent_cells_by_block,
        parent_vertices_by_block,
        block_source_blocks=source_blocks_by_block,
    )
    restricted_map = (
        None
        if source_geometry.source_map is None
        else _RestrictedCoordinateMap(source, source_geometry, target)
    )
    geometry = CellGeometrySpec(
        elements_by_block,
        routes_by_block,
        source_geometry.coordinates,
        source_map=restricted_map,
        source_mesh=target if restricted_map is not None else None,
        restriction_source=restriction,
    )
    return target, geometry, target_order_array


def _mesh_face_vertices(mesh: CellMesh, /) -> np.ndarray:
    """Return dense face-vertex rows for supported volume connectivity."""
    connectivity = mesh.connectivity
    if isinstance(connectivity, (TetrahedralConnectivity, HexahedralConnectivity)):
        return np.asarray(connectivity.faces, dtype=np.int64)
    if isinstance(connectivity, SimplicialConnectivity):
        if connectivity.dimension != 3:
            raise TypeError(
                "Dual face extraction requires three-dimensional simplicial topology."
            )
        return np.asarray(connectivity.entities[2], dtype=np.int64)
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "Volume face extraction requires simplex, hex, or poly connectivity."
        )
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    widths = np.diff(offsets)
    if (
        offsets.ndim != 1
        or offsets.size != int(connectivity.face_count) + 1
        or offsets[0] != 0
        or offsets[-1] != vertices.size
        or widths.size == 0
        or np.any(widths != widths[0])
    ):
        raise ValueError(
            "Dual extraction requires a uniform triangular or quadrilateral face bank."
        )
    return vertices.reshape((widths.size, int(widths[0])))


def _restrict_affine_dual_geometry(
    source: CellMesh,
    target: CellMesh,
    source_geometry: CellGeometrySpec,
    source_cells: np.ndarray,
    block_offsets: np.ndarray,
    supports: np.ndarray,
    /,
) -> CellGeometrySpec:
    """Restrict each dual hex exactly through its affine parent tetrahedron.

    Every hex keeps the exact trilinear chart of its barycentric corners into
    the parent reference tetrahedron, so its coordinate map is the parent map
    composed with that chart (no rounded dual vertex enters the geometry), and
    stable source cell, vertex and material-block identities are retained.
    """
    elements, source_routes, _ = source_geometry.resolve(source)
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    elements_by_block = {}
    routes_by_block = {}
    parent_cells_by_block = {}
    parent_vertices_by_block = {}
    source_blocks_by_block = {}
    for block_index, (source_block, target_block) in enumerate(
        zip(source.blocks, target.blocks, strict=True)
    ):
        source_element = elements[block_index]
        if not isinstance(source_element, ScalarCoordinateElement):
            raise TypeError("Affine dual restriction requires a scalar parent element.")
        start, stop = int(block_offsets[block_index]), int(block_offsets[block_index + 1])
        local_parents = np.repeat(np.arange(stop - start, dtype=np.int64), 4)
        parent_ids = np.asarray(source_block.global_ids, dtype=np.int64)[local_parents]
        hexes = np.asarray(target_block.vertices, dtype=np.int64)
        elements_by_block[target_block.name] = PerCellRestrictedCellGeometryElement(
            source_element,
            "hexahedron",
            tuple(int(value) for value in np.asarray(target_block.global_ids).tolist()),
            tuple(int(value) for value in parent_ids.tolist()),
            tuple(
                _trilinear_chart(
                    _dual_hex_reference_corners(
                        hexes[row], source_cells[start + row // 4], supports
                    )
                )
                for row in range(hexes.shape[0])
            ),
        )
        routes_by_block[target_block.name] = np.asarray(
            source_routes[block_index], dtype=np.int64
        )[local_parents]
        parent_cells_by_block[target_block.name] = parent_ids
        parent_vertices_by_block[target_block.name] = source_vertex_ids[
            source_cells[start + local_parents]
        ]
        source_blocks_by_block[target_block.name] = source_block.name
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        source.topology_id,
        parent_cells_by_block,
        parent_vertices_by_block,
        block_source_blocks=source_blocks_by_block,
    )
    restricted_map = (
        None
        if source_geometry.source_map is None
        else _RestrictedCoordinateMap(source, source_geometry, target)
    )
    return CellGeometrySpec(
        elements_by_block,
        routes_by_block,
        source_geometry.coordinates,
        source_map=restricted_map,
        source_mesh=target if restricted_map is not None else None,
        restriction_source=restriction,
    )


def _dual_hex_source_edge_ancestry(
    source: CellMesh,
    target: CellMesh,
    supports: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_connectivity = source.connectivity
    target_connectivity = target.connectivity
    if isinstance(source_connectivity, (TetrahedralConnectivity, PolyhedralConnectivity)):
        source_edges = np.asarray(source_connectivity.edges, dtype=np.int64)
    elif (
        isinstance(source_connectivity, SimplicialConnectivity)
        and source_connectivity.dimension == 3
    ):
        source_edges = np.asarray(source_connectivity.entities[1], dtype=np.int64)
    else:
        raise TypeError("Dual edge ancestry requires tetrahedral source topology.")
    source_edge_ids = np.asarray(source.entity_set(1).entity_ids, dtype=np.int64)
    if isinstance(target_connectivity, HexahedralConnectivity):
        target_edges = np.asarray(target_connectivity.edges, dtype=np.int64)
    elif isinstance(target_connectivity, PolyhedralConnectivity):
        target_edges = np.asarray(target_connectivity.edges, dtype=np.int64)
    else:
        raise TypeError("Dual edge ancestry requires hexahedral target connectivity.")
    target_edge_ids = np.asarray(target.entity_set(1).entity_ids, dtype=np.int64)
    if source_edges.shape != (source_edge_ids.size, 2) or target_edges.shape != (
        target_edge_ids.size,
        2,
    ):
        raise ValueError(
            "Dual source-edge ancestry must align with canonical edge banks."
        )
    source_by_vertices: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for row, edge in enumerate(source_edges.tolist()):
        oriented_endpoints = (int(edge[0]), int(edge[1]))
        key = (
            min(oriented_endpoints[0], oriented_endpoints[1]),
            max(oriented_endpoints[0], oriented_endpoints[1]),
        )
        source_by_vertices[key] = (int(source_edge_ids[row]), oriented_endpoints)
    edge_source_ids = np.full((target_edge_ids.size,), -1, dtype=np.int64)
    parameter_numerators = np.zeros((target_edge_ids.size, 2), dtype=np.int64)
    parameter_denominators = np.ones((target_edge_ids.size, 2), dtype=np.int64)
    for target_row, edge in enumerate(target_edges.tolist()):
        endpoint_supports: tuple[tuple[int, ...], tuple[int, ...]] = (
            tuple(
                int(value) for value in supports[int(edge[0])].tolist() if int(value) >= 0
            ),
            tuple(
                int(value) for value in supports[int(edge[1])].tolist() if int(value) >= 0
            ),
        )
        source_vertices = sorted(
            {value for support in endpoint_supports for value in support}
        )
        if len(source_vertices) != 2:
            continue
        source_edge = source_by_vertices.get((source_vertices[0], source_vertices[1]))
        if source_edge is None:
            continue
        source_edge_id, oriented_endpoints = source_edge
        parameters: list[tuple[int, int]] = []
        for support in endpoint_supports:
            if len(support) == 1 and support[0] in oriented_endpoints:
                parameters.append((oriented_endpoints.index(support[0]), 1))
            elif len(support) == 2 and set(support) == set(oriented_endpoints):
                parameters.append((1, 2))
            else:
                parameters = []
                break
        if len(parameters) != 2:
            continue
        edge_source_ids[target_row] = source_edge_id
        parameter_numerators[target_row] = np.asarray(
            [value[0] for value in parameters], dtype=np.int64
        )
        parameter_denominators[target_row] = np.asarray(
            [value[1] for value in parameters], dtype=np.int64
        )
    return edge_source_ids, parameter_numerators, parameter_denominators


@checked
def extract_volume_hexes(
    source: CellMesh,
    /,
    *,
    source_geometry: CellGeometrySpec | None = None,
) -> DualExtraction:
    """Partition tetrahedra into vertex-star hexes by explicit dual conversion.

    This is not a general PLC or all-hex solver. Material/source blocks remain
    distinct in the output. Non-affine coordinate charts are restricted exactly
    through each parent tetrahedron. Domain coverage, embedding, and source
    fidelity remain canonical acceptance duties.
    """
    if (
        source.topological_dimension != 3
        or source.ambient_dimension != 3
        or not source.blocks
        or any(
            not _is_tetrahedron_family(block.cell_kind) or block.cell_count == 0
            for block in source.blocks
        )
    ):
        raise ValueError(
            "Dual hex extraction requires a three-dimensional tetrahedral source mesh."
        )
    if source_geometry is None:
        source_geometry = CellGeometrySpec.affine(source)
    elif not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("source_geometry must be a CellGeometrySpec or None.")
    affine_source = _affine_geometry_matches(source, source_geometry)
    source_validity = certify_cell_geometry_validity(source_geometry, mesh=source)
    if source_validity.invalid_count or source_validity.unresolved_count:
        raise ValueError("Dual hex source tetrahedra must be certified positive.")

    cells = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in source.blocks),
        axis=0,
    )
    block_offsets = np.concatenate(
        (
            np.asarray((0,), dtype=np.int64),
            np.cumsum(
                np.asarray([block.cell_count for block in source.blocks], dtype=np.int64)
            ),
        )
    )
    source_connectivity = source.connectivity
    if isinstance(source_connectivity, (TetrahedralConnectivity, PolyhedralConnectivity)):
        edges = np.asarray(source_connectivity.edges, dtype=np.int64)
        face_incidence = np.asarray(source_connectivity.face_cell_counts, dtype=np.int64)
    elif (
        isinstance(source_connectivity, SimplicialConnectivity)
        and source_connectivity.dimension == 3
    ):
        edges = np.asarray(source_connectivity.entities[1], dtype=np.int64)
        face_incidence = np.bincount(
            np.asarray(source_connectivity.cell_entities[2], dtype=np.int64).reshape(-1),
            minlength=len(source_connectivity.entities[2]),
        )
    else:
        raise TypeError("Dual hex extraction requires tetrahedral source connectivity.")
    faces = _mesh_face_vertices(source)
    points = np.asarray(source.coordinates, dtype=np.float64)
    original = points.shape[0]
    edge_count, face_count, cell_count = edges.shape[0], faces.shape[0], cells.shape[0]
    if np.any(face_incidence > 2):
        raise ValueError(
            "Dual hex extraction requires a manifold tetrahedral face complex."
        )

    edge_lookup = {
        tuple(sorted(map(int, edge))): original + row
        for row, edge in enumerate(edges.tolist())
    }
    face_lookup = {
        tuple(sorted(map(int, face))): original + edge_count + row
        for row, face in enumerate(faces.tolist())
    }
    local_edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    local_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    nodes = np.empty((cell_count, 15), dtype=np.int64)
    nodes[:, :4] = cells
    for row, tetrahedron in enumerate(cells.tolist()):
        nodes[row, 4:10] = [
            edge_lookup[tuple(sorted((tetrahedron[a], tetrahedron[b])))]
            for a, b in local_edges
        ]
        nodes[row, 10:14] = [
            face_lookup[tuple(sorted(tetrahedron[index] for index in face))]
            for face in local_faces
        ]
        nodes[row, 14] = original + edge_count + face_count + row

    # Dual vertices are correctly rounded exact barycenters, so they equal the
    # rounded images of the exact restricted hex charts.
    exact_points = tuple(
        tuple(Fraction(float(value)) for value in point) for point in points.tolist()
    )
    coordinates = np.concatenate(
        (
            points,
            np.asarray(
                [
                    [
                        float(
                            sum(
                                (exact_points[vertex][axis] for vertex in row),
                                Fraction(0),
                            )
                            / len(row)
                        )
                        for axis in range(3)
                    ]
                    for row in (*edges.tolist(), *faces.tolist(), *cells.tolist())
                ],
                dtype=np.float64,
            ).reshape((-1, 3)),
        ),
        axis=0,
    )
    hexes = meshcore_owner.tetrahedron_dual_hexes(nodes, coordinates.shape[0])
    if hexes.shape != (4 * cell_count, 8):
        raise ValueError(
            "Native tetrahedral dual extraction returned an invalid cell extent."
        )
    if len(source.blocks) == 1:
        target_blocks = (CellBlock("cells", "hexahedron", hexes),)
    else:
        target_blocks = tuple(
            CellBlock(
                block.name,
                "hexahedron",
                hexes[4 * int(block_offsets[index]) : 4 * int(block_offsets[index + 1])],
                global_ids=np.arange(
                    4 * int(block_offsets[index]),
                    4 * int(block_offsets[index + 1]),
                    dtype=np.int64,
                ),
            )
            for index, block in enumerate(source.blocks)
        )
    mesh = CellMesh(
        coordinates,
        target_blocks,
        vertex_global_ids=fresh_vertex_global_ids(
            source, coordinates.shape[0] - original
        ),
        numeric_version=source.numeric_version,
    )
    supports = np.full((coordinates.shape[0], 4), -1, dtype=np.int64)
    supports[:original, 0] = np.arange(original, dtype=np.int64)
    supports[original : original + edge_count, :2] = edges
    supports[original + edge_count : original + edge_count + face_count, :3] = faces
    supports[original + edge_count + face_count :] = cells
    parents = np.repeat(np.arange(cell_count, dtype=np.int64), 4)
    parent_offsets = np.arange(parents.size + 1, dtype=np.int64)
    if affine_source:
        geometry = _restrict_affine_dual_geometry(
            source, mesh, source_geometry, cells, block_offsets, supports
        )
    elif len(source.blocks) == 1:
        mesh, geometry, cell_order = _compose_tetrahedral_dual_geometry(
            source, mesh, supports, parent_offsets, parents, source_geometry
        )
        parents = parents[cell_order]
        parent_offsets = np.arange(parents.size + 1, dtype=np.int64)
    else:
        mesh, geometry, cell_order = _compose_multiblock_tetrahedral_dual_geometry(
            source, mesh, supports, parent_offsets, parents, source_geometry
        )
        parents = parents[cell_order]
        parent_offsets = np.arange(parents.size + 1, dtype=np.int64)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    if np.any(np.asarray(validity.status) != int(CellValidityStatus.CERTIFIED_VALID)):
        raise ValueError(
            "Dual hex extraction produced a cell without certified positive validity."
        )
    valences = np.bincount(hexes.reshape(-1), minlength=coordinates.shape[0]).astype(
        np.int64
    )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, (HexahedralConnectivity, PolyhedralConnectivity)):
        raise TypeError(
            "Dual boundary extraction requires hexahedral output connectivity."
        )
    boundary = np.zeros((coordinates.shape[0],), dtype=np.bool_)
    boundary_faces = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    boundary[_mesh_face_vertices(mesh)[boundary_faces].reshape(-1)] = True
    singular = np.flatnonzero((~boundary) & (valences != 8)).astype(np.int64)
    edge_source_ids, edge_parameter_numerators, edge_parameter_denominators = (
        _dual_hex_source_edge_ancestry(source, mesh, supports)
    )
    return DualExtraction(
        mesh,
        source,
        supports,
        parent_offsets,
        parents,
        validity,
        valences,
        boundary,
        singular,
        None,
        geometry,
        source_geometry,
        source_edge_ids=edge_source_ids,
        source_edge_parameter_numerators=edge_parameter_numerators,
        source_edge_parameter_denominators=edge_parameter_denominators,
    )


_HEX_TOPOLOGY = reference_cell_topology("hexahedron")
_HEX_REFERENCE_POINTS: tuple[tuple[Fraction, Fraction, Fraction], ...] = tuple(
    (
        Fraction(float(point[0])),
        Fraction(float(point[1])),
        Fraction(float(point[2])),
    )
    for point in _HEX_TOPOLOGY.vertices
)
_HEX_REFERENCE_BITS: tuple[tuple[int, int, int], ...] = tuple(
    (int(point[0]), int(point[1]), int(point[2])) for point in _HEX_TOPOLOGY.vertices
)
_HEX_LOCAL_FACES: tuple[tuple[int, int, int, int], ...] = tuple(
    (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
    for face in _HEX_TOPOLOGY.entities[2]
)
_HEX_LOCAL_EDGES: tuple[tuple[int, int], ...] = tuple(
    (int(edge[0]), int(edge[1])) for edge in _HEX_TOPOLOGY.entities[1]
)
type _RootGridNodeKey = (
    tuple[Literal["vertex"], int]
    | tuple[Literal["edge"], int, Fraction]
    | tuple[Literal["face"], int, Fraction, Fraction]
    | tuple[Literal["cell"], int, Fraction, Fraction, Fraction]
)


@dataclass(frozen=True, slots=True)
class _MappedRootIntegerGrid:
    domain_id: str
    reference_mesh_id: str
    source_geometry_id: str
    frame_id: str
    connection_id: str
    schedule_id: str
    target_size: float
    root_axis_counts: np.ndarray
    vertex_keys: tuple[_RootGridNodeKey, ...]
    vertex_root_rows: np.ndarray
    vertex_reference_points: tuple[tuple[Fraction, Fraction, Fraction], ...]
    block_cells: tuple[np.ndarray, ...]
    block_global_ids: tuple[np.ndarray, ...]
    block_parent_rows: tuple[np.ndarray, ...]
    block_regions: tuple[np.ndarray, ...]
    block_reference_vertices: tuple[
        tuple[tuple[tuple[Fraction, Fraction, Fraction], ...], ...], ...
    ]
    entity_parent_strata: tuple[tuple[tuple[tuple[int, ...], int, int], ...], ...]
    grid_id: str


@final
class MappedGridHexConstruction(StrictModule, NonTrainableState):
    """Exact root restrictions and their physical mapped hexahedral mesh."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    validity: CellValidityCertificate
    cell_regions: np.ndarray
    source_parent_rows: np.ndarray
    root_axis_counts: np.ndarray
    entity_parent_dimensions: tuple[np.ndarray, ...]
    entity_parent_ids: tuple[np.ndarray, ...]
    reference_maps: tuple[ReferenceCellMapWitnesses, ...]
    scaled_jacobian_lower: np.ndarray
    mean_ratio_lower: np.ndarray
    aspect_ratio_upper: np.ndarray
    source_domain_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_frame_id: str = eqx.field(static=True)
    source_connection_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    construction_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        validity: CellValidityCertificate,
        cell_regions: object,
        source_parent_rows: object,
        root_axis_counts: object,
        entity_parent_dimensions: tuple[object, ...],
        entity_parent_ids: tuple[object, ...],
        reference_maps: tuple[ReferenceCellMapWitnesses, ...],
        scaled_jacobian_lower: object,
        mean_ratio_lower: object,
        aspect_ratio_upper: object,
        source_domain_id: str,
        source_frame_id: str,
        source_connection_id: str,
        schedule_id: str,
        /,
    ) -> None:
        regions = np.array(cell_regions, dtype=np.int64, copy=True)
        parents = np.array(source_parent_rows, dtype=np.int64, copy=True)
        counts = np.array(root_axis_counts, dtype=np.int64, copy=True)
        dimensions = tuple(
            np.array(value, dtype=np.int8, copy=True)
            for value in entity_parent_dimensions
        )
        identifiers = tuple(
            np.array(value, dtype=np.int64, copy=True) for value in entity_parent_ids
        )
        scaled = np.array(scaled_jacobian_lower, dtype=np.float64, copy=True)
        mean = np.array(mean_ratio_lower, dtype=np.float64, copy=True)
        aspect = np.array(aspect_ratio_upper, dtype=np.float64, copy=True)
        for value in (
            regions,
            parents,
            counts,
            *dimensions,
            *identifiers,
            scaled,
            mean,
            aspect,
        ):
            value.setflags(write=False)
        cell_count = mesh.connectivity.cell_count
        if (
            mesh.topological_dimension != 3
            or mesh.ambient_dimension != 3
            or any(block.cell_kind != "hexahedron" for block in mesh.blocks)
            or regions.shape != (cell_count,)
            or parents.shape != (cell_count,)
            or counts.ndim != 2
            or counts.shape[1] != 3
            or len(dimensions) != 4
            or len(identifiers) != 4
            or any(
                dimension.shape != (mesh.entity_set(degree).entity_ids.size,)
                or identifier.shape != dimension.shape
                or np.any((dimension < 0) | (dimension > 3))
                for degree, (dimension, identifier) in enumerate(
                    zip(dimensions, identifiers, strict=True)
                )
            )
            or len(reference_maps) != len(mesh.blocks)
            or any(value.shape != (cell_count,) for value in (scaled, mean, aspect))
            or any(not np.all(np.isfinite(value)) for value in (scaled, mean, aspect))
            or not all(
                (source_domain_id, source_frame_id, source_connection_id, schedule_id)
            )
        ):
            raise ValueError(
                "Mapped grid construction banks must align with source-root restrictions."
            )
        self.mesh = mesh
        self.geometry = geometry
        self.validity = validity
        self.cell_regions = regions
        self.source_parent_rows = parents
        self.root_axis_counts = counts
        self.entity_parent_dimensions = dimensions
        self.entity_parent_ids = identifiers
        self.reference_maps = tuple(reference_maps)
        self.scaled_jacobian_lower = scaled
        self.mean_ratio_lower = mean
        self.aspect_ratio_upper = aspect
        self.source_domain_id = source_domain_id
        self.source_geometry_id = cell_geometry_id(geometry)
        self.source_frame_id = source_frame_id
        self.source_connection_id = source_connection_id
        self.schedule_id = schedule_id
        self.construction_id = canonical_fingerprint(
            {
                "kind": "mapped-source-root-grid-hex-construction",
                "domain": source_domain_id,
                "source_geometry": self.source_geometry_id,
                "frame": source_frame_id,
                "connection": source_connection_id,
                "schedule": schedule_id,
                "mesh": mesh.mesh_id,
                "regions": regions,
                "parents": parents,
                "counts": counts,
                "entity_parent_dimensions": dimensions,
                "entity_parent_ids": identifiers,
                "quality_bounds": (scaled, mean, aspect),
                "validity": validity.certificate_id,
            }
        )


def mapped_source_lipschitz_bound(frame: PreparedSourceHexFrame, /) -> float:
    """Return the exact-Bernstein-derived physical root-axis Lipschitz upper bound."""
    if not isinstance(frame, PreparedSourceHexFrame):
        raise TypeError("frame must be PreparedSourceHexFrame.")
    axis_bounds = np.asarray(frame.axis_jacobian_bounds, dtype=np.float64)
    value = math.sqrt(float(np.sum(np.max(axis_bounds, axis=0) ** 2)))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Mapped source Lipschitz bound must be positive and finite.")
    return float(np.nextafter(value, np.inf))


def _equalize_root_axis_counts(
    counts: np.ndarray,
    frame: PreparedSourceHexFrame,
    connection: SourceBlockConnection,
    /,
) -> np.ndarray:
    roots = counts.shape[0]
    parent = np.arange(3 * roots, dtype=np.int64)

    def find(value: int, /) -> int:
        current = int(value)
        while int(parent[current]) != current:
            parent[current] = parent[int(parent[current])]
            current = int(parent[current])
        return current

    def join(first: int, second: int, /) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for row, (left, left_face, right, right_face) in enumerate(
        connection.face_connections.tolist()
    ):
        left_points = np.asarray(_HEX_TOPOLOGY.vertices, dtype=np.int64)[
            np.asarray(_HEX_LOCAL_FACES[left_face], dtype=np.int64)
        ]
        right_points = np.asarray(_HEX_TOPOLOGY.vertices, dtype=np.int64)[
            np.asarray(_HEX_LOCAL_FACES[right_face], dtype=np.int64)
        ]
        left_normal = int(
            np.flatnonzero(np.all(left_points == left_points[:1], axis=0))[0]
        )
        right_normal = int(
            np.flatnonzero(np.all(right_points == right_points[:1], axis=0))[0]
        )
        transport = connection.transports[row]
        for axis in range(3):
            if axis == left_normal:
                continue
            target_axis = int(np.argmax(np.abs(transport[:, axis])))
            if target_axis == right_normal:
                raise ValueError(
                    "Mapped root face connection does not preserve tangent axes."
                )
            join(3 * int(left) + axis, 3 * int(right) + target_axis)
    maxima: dict[int, int] = {}
    for root in range(roots):
        for axis in range(3):
            key = find(3 * root + axis)
            maxima[key] = max(maxima.get(key, 1), int(counts[root, axis]))
    result = counts.copy()
    for root in range(roots):
        for axis in range(3):
            result[root, axis] = maxima[find(3 * root + axis)]
    return result


def _canonical_quad_cycle(
    values: tuple[int, int, int, int], /
) -> tuple[int, int, int, int]:
    rotations: list[tuple[int, int, int, int]] = []
    for index in range(4):
        rotations.append(
            (
                values[index],
                values[(index + 1) % 4],
                values[(index + 2) % 4],
                values[(index + 3) % 4],
            )
        )
    reversed_values = (values[3], values[2], values[1], values[0])
    for index in range(4):
        rotations.append(
            (
                reversed_values[index],
                reversed_values[(index + 1) % 4],
                reversed_values[(index + 2) % 4],
                reversed_values[(index + 3) % 4],
            )
        )
    return min(rotations)


def _root_entity_lookups(
    mesh: CellMesh, /
) -> tuple[np.ndarray, dict[tuple[int, int], int], dict[tuple[int, int, int, int], int]]:
    connectivity = mesh.connectivity
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    if isinstance(connectivity, HexahedralConnectivity):
        edges = np.asarray(connectivity.edges, dtype=np.int64)
        faces = np.asarray(connectivity.faces, dtype=np.int64)
    elif isinstance(connectivity, PolyhedralConnectivity):
        edges = np.asarray(connectivity.edges, dtype=np.int64)
        face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        if np.any(np.diff(face_offsets) != 4):
            raise TypeError("Mapped hexahedral root faces must have four vertices.")
        faces = face_values.reshape((-1, 4))
    else:
        raise TypeError("Mapped root entity lookup requires hexahedral cell incidence.")
    if edges.shape != (edge_ids.size, 2) or faces.shape != (face_ids.size, 4):
        raise ValueError(
            "Mapped root entity IDs do not align with their incidence banks."
        )
    edge_lookup: dict[tuple[int, int], int] = {}
    for index in range(edges.shape[0]):
        first_row = int(edges[index, 0])
        second_row = int(edges[index, 1])
        identifiers = sorted((int(vertex_ids[first_row]), int(vertex_ids[second_row])))
        edge_lookup[(identifiers[0], identifiers[1])] = int(edge_ids[index])
    face_lookup: dict[tuple[int, int, int, int], int] = {}
    for index in range(faces.shape[0]):
        identifiers = sorted(
            int(vertex_ids[int(faces[index, column])]) for column in range(4)
        )
        face_lookup[(identifiers[0], identifiers[1], identifiers[2], identifiers[3])] = (
            int(face_ids[index])
        )
    return vertex_ids, edge_lookup, face_lookup


def _root_entity_for_points(
    points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    root_cell: np.ndarray,
    root_cell_id: int,
    vertex_ids: np.ndarray,
    edge_lookup: dict[tuple[int, int], int],
    face_lookup: dict[tuple[int, int, int, int], int],
    /,
) -> tuple[int, int]:
    fixed: list[tuple[int, Fraction]] = []
    for axis in range(3):
        values = {point[axis] for point in points}
        if len(values) == 1:
            value = next(iter(values))
            if value in (Fraction(0), Fraction(1)):
                fixed.append((axis, value))
    if not fixed:
        return 3, int(root_cell_id)
    root_points = _HEX_REFERENCE_POINTS
    if len(fixed) == 3:
        local = next(
            index for index, point in enumerate(root_points) if point == points[0]
        )
        return 0, int(vertex_ids[int(root_cell[local])])
    if len(fixed) == 2:
        fixed_axes = {axis for axis, _ in fixed}
        varying_axis = next(axis for axis in range(3) if axis not in fixed_axes)
        local_edge = next(
            edge
            for edge in _HEX_LOCAL_EDGES
            if all(
                root_points[vertex][axis] == value
                for axis, value in fixed
                for vertex in edge
            )
            and root_points[edge[0]][varying_axis] != root_points[edge[1]][varying_axis]
        )
        edge_vertices = (
            int(vertex_ids[int(root_cell[local_edge[0]])]),
            int(vertex_ids[int(root_cell[local_edge[1]])]),
        )
        edge_key = (min(edge_vertices), max(edge_vertices))
        return 1, edge_lookup[edge_key]
    axis, value = fixed[0]
    local_face = next(
        face
        for face in _HEX_LOCAL_FACES
        if all(root_points[vertex][axis] == value for vertex in face)
    )
    face_vertices = tuple(
        int(vertex_ids[int(root_cell[local_vertex])]) for local_vertex in local_face
    )
    sorted_face = sorted(face_vertices)
    face_key = (sorted_face[0], sorted_face[1], sorted_face[2], sorted_face[3])
    return 2, face_lookup[face_key]


def _root_grid_node_key(
    point: tuple[Fraction, Fraction, Fraction],
    root_cell: np.ndarray,
    root_cell_id: int,
    vertex_ids: np.ndarray,
    edge_lookup: dict[tuple[int, int], int],
    face_lookup: dict[tuple[int, int, int, int], int],
    /,
) -> _RootGridNodeKey:
    root_points = _HEX_REFERENCE_POINTS
    fixed = [axis for axis in range(3) if point[axis] in (Fraction(0), Fraction(1))]
    if len(fixed) == 3:
        local = next(index for index, value in enumerate(root_points) if value == point)
        return ("vertex", int(vertex_ids[int(root_cell[local])]))
    if len(fixed) == 2:
        free_axis = next(axis for axis in range(3) if axis not in fixed)
        local_edge = next(
            edge
            for edge in _HEX_LOCAL_EDGES
            if all(
                root_points[vertex][axis] == point[axis]
                for axis in fixed
                for vertex in edge
            )
        )
        first = int(vertex_ids[int(root_cell[local_edge[0]])])
        second = int(vertex_ids[int(root_cell[local_edge[1]])])
        edge_key = (min(first, second), max(first, second))
        edge_id = edge_lookup[edge_key]
        local_first = root_points[local_edge[0]][free_axis]
        parameter = point[free_axis] if first < second else Fraction(1) - point[free_axis]
        if local_first == 1:
            parameter = Fraction(1) - parameter
        return ("edge", edge_id, parameter)
    if len(fixed) == 1:
        fixed_axis = fixed[0]
        local_face = next(
            face
            for face in _HEX_LOCAL_FACES
            if all(
                root_points[vertex][fixed_axis] == point[fixed_axis] for vertex in face
            )
        )
        global_cycle = (
            int(vertex_ids[int(root_cell[local_face[0]])]),
            int(vertex_ids[int(root_cell[local_face[1]])]),
            int(vertex_ids[int(root_cell[local_face[2]])]),
            int(vertex_ids[int(root_cell[local_face[3]])]),
        )
        canonical = _canonical_quad_cycle(global_cycle)
        sorted_face = sorted(global_cycle)
        face_key = (sorted_face[0], sorted_face[1], sorted_face[2], sorted_face[3])
        face_id = face_lookup[face_key]
        square = (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(1), Fraction(1)),
            (Fraction(0), Fraction(1)),
        )
        canonical_uv: dict[int, tuple[Fraction, Fraction]] = {
            identifier: square[index] for index, identifier in enumerate(canonical)
        }
        varying_axes = tuple(axis for axis in range(3) if axis != fixed_axis)
        u, v = point[varying_axes[0]], point[varying_axes[1]]
        mapped_u = Fraction(0)
        mapped_v = Fraction(0)
        for local_vertex, identifier in zip(local_face, global_cycle, strict=True):
            local_point = root_points[local_vertex]
            bits = (int(local_point[varying_axes[0]]), int(local_point[varying_axes[1]]))
            weight = (u if bits[0] else Fraction(1) - u) * (
                v if bits[1] else Fraction(1) - v
            )
            mapped_u += weight * canonical_uv[identifier][0]
            mapped_v += weight * canonical_uv[identifier][1]
        return ("face", face_id, mapped_u, mapped_v)
    return ("cell", int(root_cell_id), *point)


def _mapped_reference_stratum(
    point_rows: tuple[tuple[Fraction, Fraction, Fraction], ...],
    root_cell: np.ndarray,
    root_cell_id: int,
    vertex_ids: np.ndarray,
    edge_lookup: dict[tuple[int, int], int],
    face_lookup: dict[tuple[int, int, int, int], int],
    /,
) -> tuple[int, int]:
    return _root_entity_for_points(
        point_rows, root_cell, root_cell_id, vertex_ids, edge_lookup, face_lookup
    )


def _mapped_axis_counts(
    frame: PreparedSourceHexFrame,
    connection: SourceBlockConnection,
    target_size: float,
    route: _NativeHexGridRoute,
    /,
) -> np.ndarray:
    if connection.frame_id != frame.frame_id:
        raise ValueError(
            "Mapped source-block connection belongs to a different prepared frame."
        )
    bounds = np.asarray(frame.axis_jacobian_bounds, dtype=np.float64)
    if route == "balanced_grid":
        count = max(1, int(math.ceil(float(np.max(bounds)) / target_size)))
        counts = np.full(bounds.shape, count, dtype=np.int64)
    else:
        counts = np.maximum(1, np.ceil(bounds / target_size).astype(np.int64))
    return _equalize_root_axis_counts(counts, frame, connection)


def _exact_coordinate_row(values: tuple[object, ...], /) -> tuple[ExactScalar, ...]:
    result: list[ExactScalar] = []
    for value in values:
        if not isinstance(value, (Fraction, RadicalElement)):
            raise TypeError(
                "Mapped root coordinates must retain exact rational or algebraic values."
            )
        result.append(value)
    return tuple(result)


@checked
def generate_integer_grid_hexes(
    domain: MappedReferenceDomain,
    frame: PreparedSourceHexFrame,
    connection: SourceBlockConnection,
    schedule: _NativeHexGridSchedule,
    target_size: float,
    /,
    *,
    maximum_size: float | None = None,
) -> _MappedRootIntegerGrid:
    """Build conforming source-root tensor subdivisions from physical Jacobian bounds."""
    if not isinstance(domain, MappedReferenceDomain):
        raise TypeError("domain must be MappedReferenceDomain.")
    if not isinstance(frame, PreparedSourceHexFrame) or not isinstance(
        connection, SourceBlockConnection
    ):
        raise TypeError(
            "Mapped tensor grids require a prepared source frame and block connection."
        )
    match schedule.route:
        case "balanced_grid" | "frame_grid":
            pass
        case _:
            raise ValueError(
                "Mapped tensor grids require balanced_grid or frame_grid schedules."
            )
    if (
        frame.reference_mesh.mesh_id != domain.reference_mesh.mesh_id
        or frame.source_geometry_id != cell_geometry_id(domain.source_geometry)
        or connection.frame_id != frame.frame_id
    ):
        raise ValueError(
            "Mapped tensor-grid preparation must retain current root map and connectivity."
        )
    size = finite_real_scalar(target_size, "target_size")
    if size <= 0.0:
        raise ValueError("target_size must be positive.")
    if maximum_size is not None:
        upper = finite_real_scalar(maximum_size, "maximum_size")
        if upper <= 0.0:
            raise ValueError("maximum_size must be positive.")
        size = min(size, upper)
    root_axis_counts = _mapped_axis_counts(frame, connection, size, schedule.route)
    roots = sum(block.cell_count for block in domain.reference_mesh.blocks)
    if root_axis_counts.shape != (roots, 3) or np.asarray(domain.cell_regions).shape != (
        roots,
    ):
        raise ValueError(
            "Mapped root counts and region assignments must cover every source root."
        )
    vertex_ids, edge_lookup, face_lookup = _root_entity_lookups(domain.reference_mesh)
    block_cells: list[list[list[int]]] = [[] for _ in domain.reference_mesh.blocks]
    block_global_ids: list[list[int]] = [[] for _ in domain.reference_mesh.blocks]
    block_parent_rows: list[list[int]] = [[] for _ in domain.reference_mesh.blocks]
    block_regions: list[list[int]] = [[] for _ in domain.reference_mesh.blocks]
    block_reference_vertices: list[
        list[tuple[tuple[Fraction, Fraction, Fraction], ...]]
    ] = [[] for _ in domain.reference_mesh.blocks]
    vertex_rows: dict[_RootGridNodeKey, int] = {}
    vertex_keys: list[_RootGridNodeKey] = []
    vertex_root_rows: list[int] = []
    vertex_reference_points: list[tuple[Fraction, Fraction, Fraction]] = []
    exact_vertex_coordinates: list[tuple[ExactScalar, ...]] = []
    entity_sources: list[dict[tuple[int, ...], tuple[int, int]]] = [{} for _ in range(4)]
    parent_rows = np.asarray(frame.root_block_rows, dtype=np.int64)
    block_indices = np.asarray(frame.root_block_indices, dtype=np.int64)
    root_cells = np.asarray(frame.root_cells, dtype=np.int64)
    root_ids = np.asarray(frame.root_cell_ids, dtype=np.int64)
    root_counts = np.asarray(domain.cell_regions, dtype=np.int64)
    global_cell_id = 0
    for root_row in range(root_cells.shape[0]):
        root_cell = np.asarray(root_cells[root_row], dtype=np.int64)
        block_index = int(block_indices[root_row])
        source_row = int(parent_rows[root_row])
        root_id = int(root_ids[root_row])
        counts: tuple[int, int, int] = (
            int(root_axis_counts[root_row, 0]),
            int(root_axis_counts[root_row, 1]),
            int(root_axis_counts[root_row, 2]),
        )
        lattice: tuple[tuple[int, int, int], ...] = tuple(
            (first, second, third)
            for first in range(counts[0] + 1)
            for second in range(counts[1] + 1)
            for third in range(counts[2] + 1)
        )
        local_points: tuple[tuple[Fraction, Fraction, Fraction], ...] = tuple(
            (
                Fraction(index[0], counts[0]),
                Fraction(index[1], counts[1]),
                Fraction(index[2], counts[2]),
            )
            for index in lattice
        )
        raw_exact_points = domain.source_geometry.evaluate_exact(
            domain.reference_mesh,
            np.asarray((root_id,), dtype=np.int64),
            (local_points,),
        )[0]
        exact_points = tuple(
            _exact_coordinate_row(tuple(point)) for point in raw_exact_points
        )
        local_rows: dict[tuple[int, int, int], int] = {}
        for index, point, exact_point in zip(
            lattice, local_points, exact_points, strict=True
        ):
            key = _root_grid_node_key(
                point,
                root_cell,
                root_id,
                vertex_ids,
                edge_lookup,
                face_lookup,
            )
            existing = vertex_rows.get(key)
            if existing is None:
                existing = len(vertex_keys)
                vertex_rows[key] = existing
                vertex_keys.append(key)
                vertex_root_rows.append(root_row)
                vertex_reference_points.append(point)
                exact_vertex_coordinates.append(tuple(exact_point))
            elif exact_vertex_coordinates[existing] != tuple(exact_point):
                raise ValueError(
                    "Adjacent mapped root charts disagree on an exact shared grid vertex."
                )
            local_rows[index] = existing
        for lower_values in product(*(range(value) for value in counts)):
            lower = (
                int(lower_values[0]),
                int(lower_values[1]),
                int(lower_values[2]),
            )
            lower_exact: tuple[Fraction, Fraction, Fraction] = (
                Fraction(lower[0], counts[0]),
                Fraction(lower[1], counts[1]),
                Fraction(lower[2], counts[2]),
            )
            upper_exact: tuple[Fraction, Fraction, Fraction] = (
                Fraction(lower[0] + 1, counts[0]),
                Fraction(lower[1] + 1, counts[1]),
                Fraction(lower[2] + 1, counts[2]),
            )
            reference_cell: tuple[tuple[Fraction, Fraction, Fraction], ...] = tuple(
                (
                    lower_exact[0] + (upper_exact[0] - lower_exact[0]) * bit[0],
                    lower_exact[1] + (upper_exact[1] - lower_exact[1]) * bit[1],
                    lower_exact[2] + (upper_exact[2] - lower_exact[2]) * bit[2],
                )
                for bit in _HEX_REFERENCE_BITS
            )
            cell_nodes: tuple[int, ...] = tuple(
                local_rows[
                    (
                        lower[0] + bit[0],
                        lower[1] + bit[1],
                        lower[2] + bit[2],
                    )
                ]
                for bit in _HEX_REFERENCE_BITS
            )
            target_id = global_cell_id
            global_cell_id += 1
            block_cells[block_index].append(list(cell_nodes))
            block_global_ids[block_index].append(target_id)
            block_parent_rows[block_index].append(source_row)
            block_regions[block_index].append(int(root_counts[root_row]))
            block_reference_vertices[block_index].append(reference_cell)
            for dimension in range(4):
                for local_entity in _HEX_TOPOLOGY.entities[dimension]:
                    target_vertices = tuple(
                        cell_nodes[int(index)] for index in local_entity
                    )
                    key = (
                        (target_vertices[0],)
                        if dimension == 0
                        else tuple(sorted(target_vertices))
                    )
                    if dimension == 3:
                        parent_stratum = (3, root_id)
                    else:
                        parent_stratum = _mapped_reference_stratum(
                            tuple(reference_cell[int(index)] for index in local_entity),
                            root_cell,
                            root_id,
                            vertex_ids,
                            edge_lookup,
                            face_lookup,
                        )
                    previous = entity_sources[dimension].get(key)
                    if previous is not None and previous != parent_stratum:
                        raise ValueError(
                            "Shared mapped root-grid entities disagree on source stratum ancestry."
                        )
                    entity_sources[dimension][key] = parent_stratum
    grid_id = canonical_fingerprint(
        {
            "kind": "mapped-root-integer-grid",
            "domain": domain.domain_id,
            "reference_mesh": domain.reference_mesh.mesh_id,
            "source_geometry": cell_geometry_id(domain.source_geometry),
            "schedule": schedule.schedule_id,
            "target_size": size,
            "root_counts": root_axis_counts,
            "cells": tuple(
                tuple(tuple(cell) for cell in values) for values in block_cells
            ),
            "parents": tuple(tuple(values) for values in block_parent_rows),
            "regions": tuple(tuple(values) for values in block_regions),
            "vertex_keys": tuple(repr(key) for key in vertex_keys),
            "frame": frame.frame_id,
            "connection": connection.connection_id,
        }
    )
    return _MappedRootIntegerGrid(
        domain.domain_id,
        domain.reference_mesh.mesh_id,
        cell_geometry_id(domain.source_geometry),
        frame.frame_id,
        connection.connection_id,
        schedule.schedule_id,
        size,
        root_axis_counts,
        tuple(vertex_keys),
        np.asarray(vertex_root_rows, dtype=np.int64),
        tuple(vertex_reference_points),
        tuple(
            np.asarray(values, dtype=np.int32).reshape((-1, 8)) for values in block_cells
        ),
        tuple(np.asarray(values, dtype=np.int64) for values in block_global_ids),
        tuple(np.asarray(values, dtype=np.int64) for values in block_parent_rows),
        tuple(np.asarray(values, dtype=np.int64) for values in block_regions),
        tuple(tuple(values) for values in block_reference_vertices),
        tuple(
            tuple(
                (
                    key,
                    entity_sources[dimension][key][0],
                    entity_sources[dimension][key][1],
                )
                for key in sorted(entity_sources[dimension])
            )
            for dimension in range(4)
        ),
        grid_id,
    )


def _tensor_box_chart(
    lower: tuple[Fraction, Fraction, Fraction],
    upper: tuple[Fraction, Fraction, Fraction],
    /,
) -> tuple[Mapping[tuple[int, ...], ExactScalar], ...]:
    charts: list[Mapping[tuple[int, ...], ExactScalar]] = []
    for axis in range(3):
        unit: tuple[int, ...] = (
            int(axis == 0),
            int(axis == 1),
            int(axis == 2),
        )
        polynomial: dict[tuple[int, ...], ExactScalar] = {(0, 0, 0): lower[axis]}
        step = upper[axis] - lower[axis]
        if step:
            polynomial[unit] = step
        charts.append(polynomial)
    return tuple(charts)


def _interval_gram_bounds(
    matrix: tuple[tuple[tuple[Fraction, Fraction], ...], ...], /
) -> tuple[Fraction, Fraction]:
    diagonal_lower = [Fraction(0) for _ in range(3)]
    diagonal_upper = [Fraction(0) for _ in range(3)]
    off_diagonal_upper = [[Fraction(0) for _ in range(3)] for _ in range(3)]
    for left in range(3):
        for right in range(3):
            for physical in range(3):
                left_lower, left_upper = matrix[physical][left]
                right_lower, right_upper = matrix[physical][right]
                if left == right:
                    diagonal_lower[left] += (
                        Fraction(0)
                        if left_lower <= 0 <= left_upper
                        else min(left_lower * left_lower, left_upper * left_upper)
                    )
                    diagonal_upper[left] += max(
                        left_lower * left_lower, left_upper * left_upper
                    )
                else:
                    left_abs = max(abs(left_lower), abs(left_upper))
                    right_abs = max(abs(right_lower), abs(right_upper))
                    off_diagonal_upper[left][right] += left_abs * right_abs
    lower = min(
        diagonal_lower[column]
        - sum(
            (off_diagonal_upper[column][other] for other in range(3) if other != column),
            Fraction(0),
        )
        for column in range(3)
    )
    upper = max(
        diagonal_upper[column]
        + sum(
            (off_diagonal_upper[column][other] for other in range(3) if other != column),
            Fraction(0),
        )
        for column in range(3)
    )
    return lower, upper


def _matrix_norm_squared_upper(matrix: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    gram = tuple(
        tuple(
            sum((matrix[row][left] * matrix[row][right] for row in range(3)), Fraction(0))
            for right in range(3)
        )
        for left in range(3)
    )
    return max(
        gram[row][row]
        + sum(
            (abs(gram[row][column]) for column in range(3) if column != row), Fraction(0)
        )
        for row in range(3)
    )


def _exact_binary64_interval(
    interval: tuple[Fraction | float, Fraction | float], /
) -> tuple[Fraction, Fraction]:
    lower, upper = interval
    return (
        lower if isinstance(lower, Fraction) else Fraction.from_float(lower),
        upper if isinstance(upper, Fraction) else Fraction.from_float(upper),
    )


def _mapped_root_quality(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bound mapped-cell quality over exact restricted polynomial coordinate maps."""
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = tuple(
        _exact_coordinate_row(tuple(row)) for row in geometry.source_coordinates()
    )
    scaled_values: list[float] = []
    mean_values: list[float] = []
    aspect_values: list[float] = []
    center = (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(3)) for row in range(3)
    )
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if not isinstance(element, PerCellRestrictedCellGeometryElement):
            raise TypeError(
                "Mapped source quality requires exact per-cell root restrictions."
            )
        routes_host = np.asarray(route, dtype=np.int64)
        for row in range(block.cell_count):
            local = tuple(coefficients[int(index)] for index in routes_host[row])
            basis: tuple[Expression, ...] = coordinate_effective_basis(
                element, cell_row=row
            )
            components: tuple[tuple[ExactScalar, ...], ...] = tuple(
                tuple(local[dof][axis] for dof in range(len(local))) for axis in range(3)
            )
            expressions = linear_combinations(basis, components)
            jacobian = tuple(
                tuple(derivative(expression, axis) for axis in range(3))
                for expression in expressions
            )
            bounds: tuple[tuple[tuple[Fraction, Fraction], ...], ...] = tuple(
                tuple(
                    _exact_binary64_interval(
                        polynomial_bounds(jacobian[physical][axis], "box", 3)
                    )
                    for axis in range(3)
                )
                for physical in range(3)
            )
            determinant = sum_polynomials(
                (
                    multiply(multiply(jacobian[0][0], jacobian[1][1]), jacobian[2][2]),
                    multiply(multiply(jacobian[0][1], jacobian[1][2]), jacobian[2][0]),
                    multiply(multiply(jacobian[0][2], jacobian[1][0]), jacobian[2][1]),
                    scale(
                        multiply(
                            multiply(jacobian[0][2], jacobian[1][1]), jacobian[2][0]
                        ),
                        -1,
                    ),
                    scale(
                        multiply(
                            multiply(jacobian[0][1], jacobian[1][0]), jacobian[2][2]
                        ),
                        -1,
                    ),
                    scale(
                        multiply(
                            multiply(jacobian[0][0], jacobian[1][2]), jacobian[2][1]
                        ),
                        -1,
                    ),
                ),
                3,
            )
            determinant_lower, _ = polynomial_bounds(determinant, "box", 3)
            determinant_lower = float(determinant_lower)
            if not math.isfinite(determinant_lower) or determinant_lower <= 0.0:
                raise ValueError(
                    "Mapped root-cell determinant lacks a positive Bernstein lower bound."
                )
            determinant_lower = float(np.nextafter(determinant_lower, -np.inf))
            component_magnitudes = tuple(
                tuple(max(abs(lower), abs(upper)) for lower, upper in row_bounds)
                for row_bounds in bounds
            )
            column_square_upper = tuple(
                sum(
                    (component_magnitudes[row][axis] ** 2 for row in range(3)),
                    Fraction(0),
                )
                for axis in range(3)
            )
            column_upper = tuple(
                float(
                    np.nextafter(
                        math.sqrt(float(np.nextafter(float(value), np.inf))),
                        np.inf,
                    )
                )
                for value in column_square_upper
            )
            product_upper = float(
                np.nextafter(
                    np.nextafter(column_upper[0] * column_upper[1], np.inf)
                    * column_upper[2],
                    np.inf,
                )
            )
            frobenius_square_exact = sum(
                (value * value for row in component_magnitudes for value in row),
                Fraction(0),
            )
            frobenius_squared_upper = float(
                np.nextafter(float(frobenius_square_exact), np.inf)
            )
            scaled_lower = float(np.nextafter(determinant_lower / product_upper, -np.inf))
            determinant_power_lower = float(
                np.nextafter(determinant_lower ** (2.0 / 3.0), -np.inf)
            )
            mean_numerator_lower = float(
                np.nextafter(3.0 * determinant_power_lower, -np.inf)
            )
            mean_lower = float(
                np.nextafter(mean_numerator_lower / frobenius_squared_upper, -np.inf)
            )
            center_jacobian = np.asarray(
                [
                    [
                        float(evaluate_expression(jacobian[physical][axis], center))
                        for axis in range(3)
                    ]
                    for physical in range(3)
                ],
                dtype=np.float64,
            )
            left, _, right = np.linalg.svd(center_jacobian, full_matrices=False)
            left_rotation = tuple(
                tuple(
                    Fraction.from_float(float(left[column, row])) for column in range(3)
                )
                for row in range(3)
            )
            right_rotation = tuple(
                tuple(
                    Fraction.from_float(float(right[column, row])) for column in range(3)
                )
                for row in range(3)
            )
            inverse_left = solve_exact(left_rotation, [list(row) for row in identity])
            inverse_right = solve_exact(right_rotation, [list(row) for row in identity])
            left_bounds: list[tuple[tuple[Fraction, Fraction], ...]] = []
            for output_axis in range(3):
                row_bounds = []
                for reference_axis in range(3):
                    lower = Fraction(0)
                    upper = Fraction(0)
                    for physical_axis in range(3):
                        coefficient = left_rotation[output_axis][physical_axis]
                        interval_lower, interval_upper = bounds[physical_axis][
                            reference_axis
                        ]
                        if coefficient >= 0:
                            lower += coefficient * interval_lower
                            upper += coefficient * interval_upper
                        else:
                            lower += coefficient * interval_upper
                            upper += coefficient * interval_lower
                    row_bounds.append((lower, upper))
                left_bounds.append(tuple(row_bounds))
            rotated_bounds: list[tuple[tuple[Fraction, Fraction], ...]] = []
            for output_axis in range(3):
                row_bounds = []
                for output_reference_axis in range(3):
                    lower = Fraction(0)
                    upper = Fraction(0)
                    for source_reference_axis in range(3):
                        coefficient = right_rotation[source_reference_axis][
                            output_reference_axis
                        ]
                        interval_lower, interval_upper = left_bounds[output_axis][
                            source_reference_axis
                        ]
                        if coefficient >= 0:
                            lower += coefficient * interval_lower
                            upper += coefficient * interval_upper
                        else:
                            lower += coefficient * interval_upper
                            upper += coefficient * interval_lower
                    row_bounds.append((lower, upper))
                rotated_bounds.append(tuple(row_bounds))
            eigen_lower, eigen_upper = _interval_gram_bounds(tuple(rotated_bounds))
            left_norm_upper = _matrix_norm_squared_upper(left_rotation)
            right_norm_upper = _matrix_norm_squared_upper(right_rotation)
            inverse_left_norm_upper = _matrix_norm_squared_upper(
                tuple(tuple(value for value in row) for row in inverse_left)
            )
            inverse_right_norm_upper = _matrix_norm_squared_upper(
                tuple(tuple(value for value in row) for row in inverse_right)
            )
            if (
                eigen_lower <= 0
                or eigen_upper <= 0
                or left_norm_upper <= 0
                or right_norm_upper <= 0
                or inverse_left_norm_upper <= 0
                or inverse_right_norm_upper <= 0
            ):
                raise ValueError(
                    "Mapped root-cell metric lacks a positive certified singular-value bound."
                )
            minimum_eigenvalue = float(np.nextafter(float(eigen_lower), -np.inf))
            maximum_eigenvalue = float(np.nextafter(float(eigen_upper), np.inf))
            if minimum_eigenvalue <= 0.0:
                raise ValueError(
                    "Mapped root-cell metric lacks a positive certified singular-value bound."
                )
            minimum_rotated = float(
                np.nextafter(
                    math.sqrt(np.nextafter(minimum_eigenvalue, -np.inf)), -np.inf
                )
            )
            maximum_rotated = float(np.nextafter(math.sqrt(maximum_eigenvalue), np.inf))
            left_op_upper = float(
                np.nextafter(
                    math.sqrt(float(np.nextafter(float(left_norm_upper), np.inf))), np.inf
                )
            )
            right_op_upper = float(
                np.nextafter(
                    math.sqrt(float(np.nextafter(float(right_norm_upper), np.inf))),
                    np.inf,
                )
            )
            inverse_left_op_upper = float(
                np.nextafter(
                    math.sqrt(
                        float(np.nextafter(float(inverse_left_norm_upper), np.inf))
                    ),
                    np.inf,
                )
            )
            inverse_right_op_upper = float(
                np.nextafter(
                    math.sqrt(
                        float(np.nextafter(float(inverse_right_norm_upper), np.inf))
                    ),
                    np.inf,
                )
            )
            minimum_denominator = float(
                np.nextafter(left_op_upper * right_op_upper, np.inf)
            )
            maximum_multiplier = float(
                np.nextafter(inverse_left_op_upper * inverse_right_op_upper, np.inf)
            )
            minimum_singular = float(
                np.nextafter(minimum_rotated / minimum_denominator, -np.inf)
            )
            maximum_singular = float(
                np.nextafter(maximum_rotated * maximum_multiplier, np.inf)
            )
            if minimum_singular <= 0.0 or not math.isfinite(maximum_singular):
                raise ValueError(
                    "Mapped root-cell metric lacks a positive certified singular-value bound."
                )
            scaled_values.append(scaled_lower)
            mean_values.append(mean_lower)
            aspect_values.append(
                float(np.nextafter(maximum_singular / minimum_singular, np.inf))
            )
    return (
        np.asarray(scaled_values, dtype=np.float64),
        np.asarray(mean_values, dtype=np.float64),
        np.asarray(aspect_values, dtype=np.float64),
    )


def _target_entity_rows(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    if dimension == 0:
        return np.arange(mesh.coordinates.shape[0], dtype=np.int64).reshape((-1, 1))
    if dimension not in (1, 2, 3):
        raise ValueError(
            "Mapped tensor-grid entities require dimensions in zero through three."
        )
    connectivity = mesh.connectivity
    if isinstance(connectivity, HexahedralConnectivity):
        if dimension == 1:
            return np.asarray(connectivity.edges, dtype=np.int64)
        if dimension == 2:
            return np.asarray(connectivity.faces, dtype=np.int64)
    elif isinstance(connectivity, PolyhedralConnectivity):
        if dimension == 1:
            return np.asarray(connectivity.edges, dtype=np.int64)
        if dimension == 2:
            offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
            values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
            if np.any(np.diff(offsets) != 4):
                raise TypeError("Mapped hexahedral target faces must have four vertices.")
            return values.reshape((-1, 4))
    else:
        raise TypeError(
            "Mapped tensor-grid entity lookup requires hexahedral cell incidence."
        )
    return np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks], axis=0
    )


@checked
def realize_mapped_grid_hexes(
    grid: _MappedRootIntegerGrid,
    domain: MappedReferenceDomain,
    frame: PreparedSourceHexFrame,
    schedule: _NativeHexGridSchedule,
    /,
) -> MappedGridHexConstruction:
    """Compose each integer source-root cell through its exact curved map."""
    if not isinstance(grid, _MappedRootIntegerGrid):
        raise TypeError("grid must be an internal mapped root integer grid.")
    if not isinstance(domain, MappedReferenceDomain):
        raise TypeError("domain must be MappedReferenceDomain.")
    if not isinstance(frame, PreparedSourceHexFrame):
        raise TypeError("frame must be PreparedSourceHexFrame.")
    if (
        grid.domain_id != domain.domain_id
        or grid.reference_mesh_id != domain.reference_mesh.mesh_id
        or grid.source_geometry_id != cell_geometry_id(domain.source_geometry)
        or grid.frame_id != frame.frame_id
        or grid.schedule_id != schedule.schedule_id
        or frame.geometry is not domain.source_geometry
    ):
        raise ValueError(
            "Mapped grid realization requires its exact root geometry and frame inputs."
        )
    point_count = len(grid.vertex_keys)
    exact_points: list[tuple[ExactScalar, ...] | None] = [None] * point_count
    root_ids = np.asarray(frame.root_cell_ids, dtype=np.int64)
    for root_row in range(root_ids.size):
        rows = np.flatnonzero(np.asarray(grid.vertex_root_rows) == root_row)
        if not rows.size:
            continue
        parameters = tuple(
            grid.vertex_reference_points[int(row)] for row in rows.tolist()
        )
        evaluated = domain.source_geometry.evaluate_exact(
            domain.reference_mesh,
            np.asarray((int(root_ids[root_row]),), dtype=np.int64),
            (parameters,),
        )[0]
        for row, point in zip(rows.tolist(), evaluated, strict=True):
            exact_points[int(row)] = _exact_coordinate_row(tuple(point))
    exact_coordinate_rows: list[tuple[ExactScalar, ...]] = []
    for point in exact_points:
        if point is None:
            raise ValueError(
                "Every mapped grid vertex must retain one exact root-chart owner."
            )
        exact_coordinate_rows.append(point)
    coordinates = np.asarray(
        [[float(value) for value in point] for point in exact_coordinate_rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Mapped root-coordinate realization must remain finite.")
    source_vertex_ids = np.asarray(
        domain.reference_mesh.vertex_global_ids, dtype=np.int64
    )
    added = sum(key[0] != "vertex" for key in grid.vertex_keys)
    generated_ids = fresh_vertex_global_ids(domain.reference_mesh, added)[
        source_vertex_ids.size :
    ]
    target_vertex_ids: list[int] = []
    generated_row = 0
    for key in grid.vertex_keys:
        if key[0] == "vertex":
            vertex_id = key[1]
            if not isinstance(vertex_id, int):
                raise TypeError("Mapped root vertex keys must retain integer global IDs.")
            target_vertex_ids.append(vertex_id)
        elif key[0] in {"edge", "face", "cell"}:
            target_vertex_ids.append(int(generated_ids[generated_row]))
            generated_row += 1
        else:
            raise ValueError(
                "Mapped root grid contains an unknown exact vertex key kind."
            )
    blocks = tuple(
        CellBlock(
            f"mapped:{source_block.name}",
            "hexahedron",
            cells,
            global_ids=identifiers,
        )
        for source_block, cells, identifiers in zip(
            domain.reference_mesh.blocks,
            grid.block_cells,
            grid.block_global_ids,
            strict=True,
        )
    )
    target = CellMesh(
        coordinates,
        blocks,
        vertex_global_ids=np.asarray(target_vertex_ids, dtype=np.int64),
        numeric_version=canonical_fingerprint(
            {
                "kind": "mapped-root-grid-numeric-view",
                "domain": domain.domain_id,
                "grid": grid.grid_id,
                "coordinates": coordinates,
            }
        ),
    )
    elements, source_routes, _ = domain.source_geometry.resolve(domain.reference_mesh)
    routes_by_block: dict[str, np.ndarray] = {}
    elements_by_block: dict[str, ScalarCoordinateElement] = {}
    parent_cells_by_block: dict[str, np.ndarray] = {}
    parent_vertices_by_block: dict[str, np.ndarray] = {}
    source_blocks_by_block: dict[str, str] = {}
    parent_maps_by_block: dict[str, ReferenceCellMapWitnesses] = {}
    output_reference_maps: list[ReferenceCellMapWitnesses] = []
    source_parent_rows: list[np.ndarray] = []
    for block_index, (
        source_block,
        target_block,
        parent_rows,
        reference_vertices,
    ) in enumerate(
        zip(
            domain.reference_mesh.blocks,
            target.blocks,
            grid.block_parent_rows,
            grid.block_reference_vertices,
            strict=True,
        )
    ):
        target_ids = np.asarray(target_block.global_ids, dtype=np.int64)
        target_ids_sequence = tuple(int(value) for value in target_ids.tolist())
        root_ids_for_block = np.asarray(source_block.global_ids, dtype=np.int64)[
            parent_rows
        ]
        root_ids_sequence = tuple(int(value) for value in root_ids_for_block.tolist())
        target_count = target_ids.size
        reference_array = np.asarray(
            [
                [[float(value) for value in point] for point in cell]
                for cell in reference_vertices
            ],
            dtype=np.float64,
        )
        cell_chart_arguments = tuple(
            _tensor_box_chart(cell[0], cell[6]) for cell in reference_vertices
        )
        reference_charts = freeze_reference_chart_arguments(cell_chart_arguments)
        local_maps = ReferenceCellMapWitnesses(
            np.arange(target_count, dtype=np.int64),
            root_ids_for_block,
            "hexahedron",
            "hexahedron",
            reference_array,
            reference_vertices,
            source_reference_chart_arguments=reference_charts,
        )
        output_maps = ReferenceCellMapWitnesses(
            target_ids,
            root_ids_for_block,
            "hexahedron",
            "hexahedron",
            reference_array,
            reference_vertices,
            source_reference_chart_arguments=reference_charts,
        )
        root_element_candidate = elements[block_index]
        if not isinstance(root_element_candidate, ScalarCoordinateElement):
            raise TypeError(
                "Mapped root coordinate source must be a scalar cell element."
            )
        elements_by_block[target_block.name] = PerCellRestrictedCellGeometryElement(
            root_element_candidate,
            "hexahedron",
            target_ids_sequence,
            root_ids_sequence,
            cell_chart_arguments,
        )
        routes_host = np.asarray(source_routes[block_index], dtype=np.int64)
        routes_by_block[target_block.name] = routes_host[parent_rows]
        root_cells = np.asarray(source_block.vertices, dtype=np.int64)[parent_rows]
        parent_cells_by_block[target_block.name] = root_ids_for_block
        parent_vertices_by_block[target_block.name] = source_vertex_ids[root_cells]
        source_blocks_by_block[target_block.name] = source_block.name
        parent_maps_by_block[target_block.name] = local_maps
        output_reference_maps.append(output_maps)
        source_parent_rows.append(
            np.asarray(
                [
                    int(
                        np.flatnonzero(
                            (frame.root_block_indices == block_index)
                            & (frame.root_block_rows == int(row))
                        )[0]
                    )
                    for row in parent_rows
                ],
                dtype=np.int64,
            )
        )
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(domain.source_geometry),
        domain.reference_mesh.topology_id,
        parent_cells_by_block,
        parent_vertices_by_block,
        block_source_blocks=source_blocks_by_block,
        block_parent_reference_maps=parent_maps_by_block,
    )
    restricted_map = (
        None
        if domain.source_geometry.source_map is None
        else _RestrictedCoordinateMap(
            domain.reference_mesh, domain.source_geometry, target
        )
    )
    geometry = CellGeometrySpec(
        elements_by_block,
        routes_by_block,
        np.asarray(domain.source_geometry.coordinates),
        source_map=restricted_map,
        source_mesh=target if restricted_map is not None else None,
        restriction_source=restriction,
    )
    validity = certify_cell_geometry_validity(geometry, mesh=target)
    if validity.invalid_count or validity.unresolved_count:
        raise ValueError(
            "Mapped root-grid cells must have certified positive coordinate maps."
        )
    scaled, mean, aspect = _mapped_root_quality(target, geometry)
    if (
        float(np.min(scaled)) < schedule.minimum_scaled_jacobian
        or float(np.min(mean)) < schedule.minimum_mean_ratio
        or float(np.max(aspect)) > schedule.maximum_aspect_ratio
    ):
        raise ValueError(
            "Mapped root-grid cells fail the declared physical quality schedule."
        )
    entity_dimensions: list[np.ndarray] = []
    entity_ids: list[np.ndarray] = []
    for dimension in range(4):
        parent_by_key = {
            key: (int(parent_dimension), int(parent_id))
            for key, parent_dimension, parent_id in grid.entity_parent_strata[dimension]
        }
        entity_rows = _target_entity_rows(target, dimension)
        parents = []
        for row in entity_rows:
            key = (
                (int(row[0]),)
                if dimension == 0
                else tuple(sorted(int(value) for value in row))
            )
            parent = parent_by_key.get(key)
            if parent is None:
                raise ValueError(
                    f"Mapped root-grid entity dimension {dimension} lacks source ancestry."
                )
            parents.append(parent)
        entity_dimensions.append(
            np.asarray([value[0] for value in parents], dtype=np.int8)
        )
        entity_ids.append(np.asarray([value[1] for value in parents], dtype=np.int64))
    regions = np.concatenate(grid.block_regions)
    parents = np.concatenate(source_parent_rows)
    return MappedGridHexConstruction(
        target,
        geometry,
        validity,
        regions,
        parents,
        grid.root_axis_counts,
        tuple(entity_dimensions),
        tuple(entity_ids),
        tuple(output_reference_maps),
        scaled,
        mean,
        aspect,
        domain.domain_id,
        grid.frame_id,
        grid.connection_id,
        schedule.schedule_id,
    )


__all__ = [
    "MappedGridHexConstruction",
    "extract_volume_hexes",
    "generate_integer_grid_hexes",
    "mapped_source_lipschitz_bound",
    "realize_mapped_grid_hexes",
]
