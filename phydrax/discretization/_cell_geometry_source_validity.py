# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Exact source-map validity for variable-topology cell blocks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..linalg._exact import determinant_exact
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..typing import ConvertibleToArray
from ._cell_complex import PolygonalConnectivity, PolyhedralConnectivity
from ._cell_mesh import CellBlock, CellMesh, PolyhedralBlock
from ._coordinate_enclosure import CoordinateSourceBank, ExactScalar
from ._exact_algebraic_geometry import (
    coefficient_endpoint,
    coefficient_interval,
    RadicalElement,
)


if TYPE_CHECKING:
    from ._cell_geometry_validity import _BlockCertificate, CellValidityPolicy


ExactPoint = tuple[ExactScalar, ...]


def _sign(value: ExactScalar, /) -> int:
    if isinstance(value, RadicalElement):
        return value.sign()
    return 1 if value > 0 else -1 if value < 0 else 0


def _sum_exact(values: Sequence[ExactScalar], /) -> ExactScalar:
    result: ExactScalar = Fraction(0)
    for value in values:
        result = result + value
    return result


def _mean_exact(points: Sequence[ExactPoint], /) -> ExactPoint:
    if not points:
        raise ValueError("An exact centroid requires at least one point.")
    inverse_count = Fraction(1, len(points))
    return tuple(
        _sum_exact(tuple(point[axis] for point in points)) * inverse_count
        for axis in range(len(points[0]))
    )


def _subtract(first: ExactPoint, second: ExactPoint, /) -> ExactPoint:
    return tuple(left - right for left, right in zip(first, second, strict=True))


def _dot(first: ExactPoint, second: ExactPoint, /) -> ExactScalar:
    return _sum_exact(
        tuple(left * right for left, right in zip(first, second, strict=True))
    )


def _cross3(first: ExactPoint, second: ExactPoint, /) -> ExactPoint:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _norm_squared(value: ExactPoint, /) -> ExactScalar:
    return _dot(value, value)


def _distance_squared(first: ExactPoint, second: ExactPoint, /) -> ExactScalar:
    return _norm_squared(_subtract(first, second))


def _sqrt_exact(value: ExactScalar, /) -> ExactScalar | None:
    if _sign(value) < 0:
        return None
    if _sign(value) == 0:
        return Fraction(0)
    if isinstance(value, Fraction):
        return RadicalElement.sqrt(value)
    return None


def _compare_area_floor(
    area: ExactScalar,
    squared_norm_products: Sequence[ExactScalar],
    relative_floor: Fraction,
    /,
    *,
    squared: bool,
) -> int | None:
    """Compare the original area floor without factoring separable radicals."""
    if relative_floor == 0:
        return _sign(area)
    lower = upper = Fraction(0)
    for product in squared_norm_products:
        product_lower, product_upper = coefficient_interval(product)
        lower += fraction_sqrt_interval(max(Fraction(0), product_lower))[0]
        upper += fraction_sqrt_interval(max(Fraction(0), product_upper))[1]
    if squared:
        lower, upper = lower * lower, upper * upper
    lower *= relative_floor
    upper *= relative_floor
    if _sign(area - upper) > 0:
        return 1
    if _sign(area - lower) < 0:
        return -1
    if lower == upper:
        return _sign(area - lower)
    # Enclosures are only a filter. An overlapping interval is not equality,
    # failure, or permission to change the policy's original threshold.
    terms = tuple(_sqrt_exact(product) for product in squared_norm_products)
    if any(value is None for value in terms):
        return None
    scale = _sum_exact(tuple(value for value in terms if value is not None))
    return _sign(area - relative_floor * (scale * scale if squared else scale))


def _exact_bounds(value: ExactScalar, /) -> tuple[float, float]:
    return (
        coefficient_endpoint(value, upper=False, saturate=True),
        coefficient_endpoint(value, upper=True, saturate=True),
    )


def _orient2(
    first: ExactPoint,
    second: ExactPoint,
    third: ExactPoint,
    /,
) -> ExactScalar:
    return determinant_exact(
        (
            (second[0] - first[0], second[1] - first[1]),
            (third[0] - first[0], third[1] - first[1]),
        )
    )


def _on_segment(first: ExactPoint, second: ExactPoint, point: ExactPoint, /) -> bool:
    if _sign(_orient2(first, second, point)) != 0:
        return False
    return all(
        (min(first[axis], second[axis]) <= point[axis] <= max(first[axis], second[axis]))
        for axis in range(2)
    )


def _segments_intersect(
    first_start: ExactPoint,
    first_stop: ExactPoint,
    second_start: ExactPoint,
    second_stop: ExactPoint,
    /,
) -> bool:
    first_start_turn = _sign(_orient2(first_start, first_stop, second_start))
    first_stop_turn = _sign(_orient2(first_start, first_stop, second_stop))
    second_start_turn = _sign(_orient2(second_start, second_stop, first_start))
    second_stop_turn = _sign(_orient2(second_start, second_stop, first_stop))
    if (
        first_start_turn * first_stop_turn < 0
        and second_start_turn * second_stop_turn < 0
    ):
        return True
    return (
        (first_start_turn == 0 and _on_segment(first_start, first_stop, second_start))
        or (first_stop_turn == 0 and _on_segment(first_start, first_stop, second_stop))
        or (
            second_start_turn == 0 and _on_segment(second_start, second_stop, first_start)
        )
        or (second_stop_turn == 0 and _on_segment(second_start, second_stop, first_stop))
    )


def _polygon_is_simple(points: Sequence[ExactPoint], /) -> bool:
    count = len(points)
    if count < 3:
        return False
    for first_edge in range(count):
        first_start = points[first_edge]
        first_stop = points[(first_edge + 1) % count]
        for second_edge in range(first_edge + 1, count):
            second_start = points[second_edge]
            second_stop = points[(second_edge + 1) % count]
            if not _segments_intersect(
                first_start, first_stop, second_start, second_stop
            ):
                continue
            adjacent = second_edge == first_edge + 1 or (
                first_edge == 0 and second_edge == count - 1
            )
            if not adjacent:
                return False
            shared = first_stop if second_edge == first_edge + 1 else first_start
            first_other = first_start if second_edge == first_edge + 1 else first_stop
            second_other = second_stop if second_edge == first_edge + 1 else second_start
            if _sign(_orient2(shared, first_other, second_other)) == 0 and (
                _on_segment(shared, first_other, second_other)
                or _on_segment(shared, second_other, first_other)
            ):
                return False
    return True


def _same_exact_point(first: ExactPoint, second: ExactPoint, /) -> bool:
    return all(left == right for left, right in zip(first, second, strict=True))


def _dominant_axis(values: ExactPoint, /) -> int:
    selected = 0
    largest = values[0] * values[0]
    for axis in range(1, len(values)):
        candidate = values[axis] * values[axis]
        if _sign(candidate - largest) > 0:
            selected = axis
            largest = candidate
    return selected


def _polygon_proof(
    points: Sequence[ExactPoint],
    ambient_dimension: int,
    policy: CellValidityPolicy,
    /,
) -> tuple[int, ExactScalar]:
    from ._cell_geometry_validity import CellValidityStatus

    if ambient_dimension not in (2, 3) or len(points) < 3:
        raise ValueError(
            "Exact polygon validity requires at least three vertices in 2D or 3D."
        )
    repeated = any(
        _same_exact_point(points[first], points[second])
        for first in range(len(points))
        for second in range(first + 1, len(points))
    )
    center = _mean_exact(points)
    centered = tuple(_subtract(point, center) for point in points)
    following = tuple(centered[(row + 1) % len(centered)] for row in range(len(centered)))
    spans = tuple(
        max(point[axis] for point in points) - min(point[axis] for point in points)
        for axis in range(ambient_dimension)
    )
    diameter_squared = _norm_squared(spans)
    relative_floor = Fraction.from_float(policy.relative_determinant_floor)
    edge_floor_squared = relative_floor * relative_floor * diameter_squared
    edge_below = False
    edge_at_floor = False
    for first, second in zip(points, (*points[1:], points[0]), strict=True):
        edge_squared = _distance_squared(first, second)
        edge_below |= _sign(edge_squared - edge_floor_squared) < 0
        edge_at_floor |= _sign(edge_squared - edge_floor_squared) == 0

    if ambient_dimension == 2:
        area = _sum_exact(
            tuple(
                first[0] * second[1] - first[1] * second[0]
                for first, second in zip(centered, following, strict=True)
            )
        )
        area_comparison = _compare_area_floor(
            area,
            tuple(
                _norm_squared(first) * _norm_squared(second)
                for first, second in zip(centered, following, strict=True)
            ),
            relative_floor,
            squared=False,
        )
        simple = _polygon_is_simple(points)
        area_sign = _sign(area)
        invalid = repeated or edge_below or not simple or area_sign < 0
        if area_comparison is not None and area_comparison < 0:
            invalid = True
        unresolved = (
            edge_at_floor
            or area_sign == 0
            or area_comparison is None
            or area_comparison == 0
        )
        status = (
            CellValidityStatus.INVALID
            if invalid
            else CellValidityStatus.CERTIFIED_VALID
            if not unresolved and area_sign > 0
            else CellValidityStatus.UNRESOLVED
        )
        return int(status), area

    area_components: list[ExactScalar] = []
    for axis in range(3):
        area_components.append(
            _sum_exact(
                tuple(
                    _cross3(first, second)[axis]
                    for first, second in zip(centered, following, strict=True)
                )
            )
        )
    area_vector: ExactPoint = (
        area_components[0],
        area_components[1],
        area_components[2],
    )
    area_squared = _norm_squared(area_vector)
    planarity_tolerance = Fraction.from_float(policy.relative_planarity_tolerance)
    nonplanar = False
    planarity_unresolved = _sign(area_squared) == 0
    if _sign(area_squared) > 0:
        allowed_squared = (
            planarity_tolerance * planarity_tolerance * diameter_squared * area_squared
        )
        nonplanar = any(
            _sign(_dot(_subtract(point, center), area_vector) ** 2 - allowed_squared) > 0
            for point in points
        )
        dominant_axis = _dominant_axis(area_vector)
    else:
        dominant_axis = _dominant_axis(spans)
    projection_axes = ((1, 2), (0, 2), (0, 1))[dominant_axis]
    projected = tuple(
        (point[projection_axes[0]], point[projection_axes[1]]) for point in points
    )
    simple = _polygon_is_simple(projected)
    area_comparison = _compare_area_floor(
        area_squared,
        tuple(
            _norm_squared(first) * _norm_squared(second)
            for first, second in zip(centered, following, strict=True)
        ),
        relative_floor,
        squared=True,
    )
    invalid = repeated or edge_below or nonplanar or not simple
    if area_comparison is not None and area_comparison < 0:
        invalid = True
    unresolved = (
        edge_at_floor
        or planarity_unresolved
        or area_comparison is None
        or area_comparison == 0
    )
    status = (
        CellValidityStatus.INVALID
        if invalid
        else CellValidityStatus.CERTIFIED_VALID
        if not unresolved and _sign(area_squared) > 0
        else CellValidityStatus.UNRESOLVED
    )
    return int(status), area_squared


def _polyhedron_proof(
    mesh: CellMesh,
    block_index: int,
    block: CellBlock | PolyhedralBlock,
    selected_rows: np.ndarray,
    exact_by_vertex: Mapping[int, ExactPoint],
    policy: CellValidityPolicy,
    /,
    *,
    star_tables: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ._cell_geometry_validity import CellValidityStatus, PolyhedralStarTables

    if not isinstance(star_tables, PolyhedralStarTables):
        raise TypeError("star_tables must be the canonical PolyhedralStarTables record.")
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral exact validity requires PolyhedralConnectivity.")
    global_offset = sum(prior.cell_count for prior in mesh.blocks[:block_index])
    global_rows = global_offset + selected_rows
    output_by_global = {
        int(global_row): output_row
        for output_row, global_row in enumerate(global_rows.tolist())
    }
    selected_star_rows = tuple(
        row
        for row, cell in enumerate(star_tables.star_cell.tolist())
        if int(cell) in output_by_global
    )
    face_rows = {int(star_tables.star_face[row]) for row in selected_star_rows}
    face_vertices: dict[int, list[int]] = {row: [] for row in face_rows}
    for face, vertex in zip(
        star_tables.face_corner_face.tolist(),
        star_tables.face_corner_vertex.tolist(),
        strict=True,
    ):
        if int(face) in face_vertices:
            face_vertices[int(face)].append(int(vertex))
    face_centers = {
        face: _mean_exact(tuple(exact_by_vertex[vertex] for vertex in vertices))
        for face, vertices in face_vertices.items()
    }
    block_vertices = np.asarray(block.vertices, dtype=np.int64)
    valid_vertices = np.asarray(block.vertex_valid, dtype=np.bool_)
    cell_centers = tuple(
        _mean_exact(
            tuple(
                exact_by_vertex[int(vertex)]
                for vertex in block_vertices[int(row), valid_vertices[int(row)]].tolist()
            )
        )
        for row in selected_rows.tolist()
    )
    determinants: list[list[ExactScalar]] = [[] for _ in range(selected_rows.size)]
    scale_squared: list[list[ExactScalar]] = [[] for _ in range(selected_rows.size)]
    for star_row in selected_star_rows:
        global_cell = int(star_tables.star_cell[star_row])
        output_row = output_by_global[global_cell]
        center = cell_centers[output_row]
        face_center = face_centers[int(star_tables.star_face[star_row])]
        first = exact_by_vertex[int(star_tables.star_first[star_row])]
        second = exact_by_vertex[int(star_tables.star_second[star_row])]
        columns = (
            _subtract(face_center, center),
            _subtract(first, center),
            _subtract(second, center),
        )
        matrix = tuple(
            tuple(columns[column][axis] for column in range(3)) for axis in range(3)
        )
        determinant = determinant_exact(matrix)
        scale_product = _norm_squared(columns[0]) * _norm_squared(columns[1])
        scale_product = scale_product * _norm_squared(columns[2])
        determinants[output_row].append(determinant)
        scale_squared[output_row].append(scale_product)

    floor = Fraction.from_float(policy.relative_determinant_floor)
    floor_squared = floor * floor
    statuses = np.full(
        (selected_rows.size,), int(CellValidityStatus.UNRESOLVED), dtype=np.int32
    )
    lower = np.full((selected_rows.size,), np.nan, dtype=np.float64)
    upper = np.full((selected_rows.size,), np.nan, dtype=np.float64)
    for output_row, star_determinants in enumerate(determinants):
        if not star_determinants:
            continue
        star_scales = scale_squared[output_row]
        lower_exact = min(star_determinants)
        upper_exact = max(star_determinants)
        lower[output_row] = _exact_bounds(lower_exact)[0]
        upper[output_row] = _exact_bounds(upper_exact)[1]
        total = _sum_exact(star_determinants)
        star_positive = all(
            _sign(determinant) > 0
            and (
                floor == 0 or _sign(determinant * determinant - floor_squared * scale) > 0
            )
            for determinant, scale in zip(star_determinants, star_scales, strict=True)
        )
        if star_positive:
            statuses[output_row] = int(CellValidityStatus.CERTIFIED_VALID)
            continue
        maximum_scale = max(star_scales)
        total_below_floor = _sign(total) < 0
        if floor > 0 and _sign(maximum_scale) > 0:
            if _sign(total) <= 0:
                total_below_floor = True
            else:
                total_below_floor = (
                    _sign(total * total - floor_squared * maximum_scale) < 0
                )
        if total_below_floor:
            statuses[output_row] = int(CellValidityStatus.INVALID)
    return statuses, lower, upper


def certify_exact_variable_topology_block(
    mesh: CellMesh,
    geometry_dof_routes: Mapping[str, ConvertibleToArray],
    exact_coordinate_bank: CoordinateSourceBank,
    policy: CellValidityPolicy,
    /,
    *,
    block_name: str,
    cell_rows: ConvertibleToArray,
) -> _BlockCertificate:
    """Certify exact polygon or polyhedron coordinates without binary64 shadowing."""
    from ._cell_geometry_validity import (
        _BlockCertificate,
        CellValidityPolicy,
        polyhedral_star_tables,
    )

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(policy, CellValidityPolicy):
        raise TypeError("policy must be CellValidityPolicy.")
    if not isinstance(block_name, str) or not block_name:
        raise ValueError("block_name must be a non-empty mesh block name.")
    if not isinstance(exact_coordinate_bank, tuple) or not exact_coordinate_bank:
        raise TypeError("exact_coordinate_bank must be a non-empty immutable exact bank.")
    ambient = mesh.ambient_dimension
    for coefficient_row in exact_coordinate_bank:
        if len(coefficient_row) != ambient or any(
            not isinstance(value, (Fraction, RadicalElement)) for value in coefficient_row
        ):
            raise TypeError(
                "Every exact geometry coefficient row must contain one ExactScalar "
                "per ambient axis."
            )
    mesh_names = tuple(block.name for block in mesh.blocks)
    if set(geometry_dof_routes) != set(mesh_names):
        raise ValueError("Geometry DOF routes must match the mesh block names exactly.")
    block_indices = tuple(
        index for index, block in enumerate(mesh.blocks) if block.name == block_name
    )
    if len(block_indices) != 1:
        raise ValueError(f"Mesh does not contain exactly one block named {block_name!r}.")
    block_index = block_indices[0]
    block = mesh.blocks[block_index]
    if block.cell_kind not in ("polygon", "polyhedron"):
        raise ValueError(
            "Exact variable-topology validity requires polygon or polyhedron cells."
        )
    rows = np.asarray(cell_rows)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or rows.dtype.kind == "b":
        raise TypeError("cell_rows must be a one-dimensional integer array.")
    if rows.dtype.kind == "u" and rows.size and np.max(rows) > np.iinfo(np.int64).max:
        raise OverflowError("cell_rows exceed the supported int64 range.")
    selected_rows = np.asarray(rows, dtype=np.int64)
    if (
        np.any(selected_rows < 0)
        or np.any(selected_rows >= block.cell_count)
        or np.unique(selected_rows).size != selected_rows.size
    ):
        raise ValueError("cell_rows must be unique rows in the selected mesh block.")
    block_vertices = np.asarray(block.vertices, dtype=np.int64)
    vertex_valid = np.asarray(block.vertex_valid, dtype=np.bool_)
    raw_routes = np.asarray(geometry_dof_routes[block_name])
    if (
        raw_routes.dtype.kind not in "iu"
        or raw_routes.shape != block_vertices.shape
        or vertex_valid.shape != block_vertices.shape
    ):
        raise ValueError(
            "Variable-topology geometry routes must align with the block vertex bank."
        )
    routes = np.asarray(raw_routes, dtype=np.int64)
    active_routes = routes[vertex_valid]
    if active_routes.size and (
        np.any(active_routes < 0) or np.any(active_routes >= len(exact_coordinate_bank))
    ):
        raise ValueError("Geometry DOF routes index outside the exact coordinate bank.")
    exact_by_vertex: dict[int, ExactPoint] = {}
    for cell_row in selected_rows.tolist():
        for vertex, dof in zip(
            block_vertices[cell_row, vertex_valid[cell_row]].tolist(),
            routes[cell_row, vertex_valid[cell_row]].tolist(),
            strict=True,
        ):
            point = exact_coordinate_bank[int(dof)]
            previous = exact_by_vertex.get(int(vertex))
            if previous is not None and previous != point:
                raise ValueError(
                    f"Mesh vertex row {int(vertex)} has inconsistent exact "
                    "coordinate routes."
                )
            exact_by_vertex[int(vertex)] = point
    if block.cell_kind == "polygon":
        if not isinstance(mesh.connectivity, PolygonalConnectivity):
            raise TypeError("Exact polygon validity requires PolygonalConnectivity.")
        records = tuple(
            _polygon_proof(
                tuple(
                    exact_by_vertex[int(vertex)]
                    for vertex in block_vertices[row, vertex_valid[row]].tolist()
                ),
                ambient,
                policy,
            )
            for row in selected_rows.tolist()
        )
        status = np.asarray([record[0] for record in records], dtype=np.int32)
        bounds = tuple(_exact_bounds(record[1]) for record in records)
        lower = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
        upper = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
    else:
        if ambient != 3 or mesh.topological_dimension != 3:
            raise ValueError(
                "Exact polyhedral validity requires three-dimensional cells."
            )
        if not isinstance(mesh.connectivity, PolyhedralConnectivity):
            raise TypeError("Exact polyhedral validity requires PolyhedralConnectivity.")
        star_tables = polyhedral_star_tables(mesh.connectivity)
        status, lower, upper = _polyhedron_proof(
            mesh,
            block_index,
            block,
            selected_rows,
            exact_by_vertex,
            policy,
            star_tables=star_tables,
        )
    return _BlockCertificate(
        status,
        lower,
        upper,
        np.zeros((selected_rows.size,), dtype=np.int64),
    )
