#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent, compact target incidence for implicit source coverage.

Only integer incidence is retained for the volume. Exact coordinate maps and
rational triangles are reconstructed on demand for boundary faces, so their
object graphs do not scale with the volume-cell count.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, TYPE_CHECKING

import numpy as np

from ..discretization._cell_geometry import (
    CellVertexGeometryElement,
    coordinate_effective_basis,
    ScalarCoordinateElement,
)
from ..discretization._coordinate_enclosure import (
    evaluate_expression,
    linear_combinations,
)
from ..discretization._reference_cell import reference_cell_topology
from ..discretization.fem._reference import FiniteElementSpec
from ._approximate_source_coverage import (
    _cross3,
    _dot3,
    _fraction,
    _interval_dot,
    _point3,
    _strict_sign,
    _triangle3,
    SourceCoverageUnavailable,
)
from ._domain_numerics import evaluate_map
from ._domain_planar_boundary import _fraction_point, _reference_corner_image


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh
    from ._implicit_source_coverage import _SortedIntegerKeys
    from ._source_boundary_records import SourceBoundaryChartWitness
    from .implicit._adaptive_discovery import _FieldEnclosures, AdaptiveImplicitScaffold


def prove_implicit_target_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_regions: np.ndarray,
    witness: SourceBoundaryChartWitness,
    bounds: _FieldEnclosures,
    source_vertices: np.ndarray,
    source_face_rows: np.ndarray,
    triangle_cells: np.ndarray,
    triangle_roots: np.ndarray,
    root_keys: _SortedIntegerKeys,
    scaffold: AdaptiveImplicitScaffold,
    /,
) -> None:
    """Prove complete oriented target/root correspondence without volume objects."""
    from ._implicit_source_coverage import _SourceTriangles

    cell_count = sum(block.cell_count for block in mesh.blocks)
    if cell_regions.shape != (cell_count,):
        raise SourceCoverageUnavailable(
            "Target boundary extraction failed: cell_regions must align with every target cell.",
            status="unresolved",
        )
    elements, routes, _ = geometry.resolve(mesh)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_ids = np.empty(cell_count, dtype=np.int64)
    offsets = np.empty(cell_count + 1, dtype=np.int64)
    offsets[0] = 0
    blocks: list[tuple[int, int, Any, Any, np.ndarray, tuple, tuple, tuple]] = []
    cursor = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind in ("polygon", "polyhedron"):
            raise SourceCoverageUnavailable(
                "Adaptive implicit root ancestry requires standard reference volume cells.",
                status="unresolved",
            )
        ref = reference_cell_topology(block.cell_kind)
        if ref.dimension != 3:
            raise SourceCoverageUnavailable(
                "Implicit boundary ancestry requires volume cells.", status="unresolved"
            )
        faces = tuple(tuple(int(v) for v in face) for face in ref.entities[2])
        if any(len(face) != 3 for face in faces):
            raise SourceCoverageUnavailable(
                "Implicit root charts currently require triangular target faces.",
                status="unresolved",
            )
        if isinstance(element, CellVertexGeometryElement):
            raise SourceCoverageUnavailable(
                "Implicit target face has no exact represented coordinate map.",
                status="unresolved",
            )
        if not isinstance(element, ScalarCoordinateElement):
            raise SourceCoverageUnavailable(
                "Target boundary extraction failed: Exact domain maps require a supported scalar coordinate element.",
                status="unresolved",
            )
        vertices = tuple(
            tuple(_fraction(float(v)) for v in point) for point in ref.vertices
        )
        center = tuple(
            sum((point[k] for point in vertices), Fraction(0)) / len(vertices)
            for k in range(3)
        )
        signs = []
        for face in faces:
            points = tuple(vertices[i] for i in face)
            normal = _cross3(_triangle3(points))
            face_center = tuple(
                sum((p[k] for p in points), Fraction(0)) / len(points) for k in range(3)
            )
            outward = _dot3(
                normal,
                (
                    face_center[0] - center[0],
                    face_center[1] - center[1],
                    face_center[2] - center[2],
                ),
            )
            if not outward:
                raise SourceCoverageUnavailable(
                    "Target reference face has no oriented outward normal.",
                    status="unresolved",
                )
            signs.append(1 if outward > 0 else -1)
        stop = cursor + block.cell_count
        cell_ids[cursor:stop] = np.asarray(block.global_ids, dtype=np.int64)
        offsets[cursor + 1 : stop + 1] = offsets[cursor] + np.arange(
            1, block.cell_count + 1, dtype=np.int64
        ) * len(faces)
        blocks.append(
            (
                cursor,
                stop,
                block,
                element,
                np.asarray(route, dtype=np.int64),
                vertices,
                faces,
                tuple(signs),
            )
        )
        cursor = stop

    # Face rows are in cell/face encounter order. Sorting the three actual
    # global vertex IDs identifies incidence without coordinate welding.
    face_count = int(offsets[-1])
    keys = np.empty((face_count, 3), dtype=np.int64)
    for start, stop, block, _element, _route, _vertices, faces, _signs in blocks:
        connectivity = np.asarray(block.vertices, dtype=np.int64)
        first = int(offsets[start])
        rows = keys[first : int(offsets[stop])].reshape(stop - start, len(faces), 3)
        for local_face, face in enumerate(faces):
            rows[:, local_face, :] = np.sort(vertex_ids[connectivity[:, face]], axis=1)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    equal = np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
    if np.any(equal[1:] & equal[:-1]):
        raise SourceCoverageUnavailable(
            "A target facet has more than two incident cells.", status="unresolved"
        )
    starts = (
        np.flatnonzero(np.r_[True, ~equal]) if face_count else np.empty(0, dtype=np.int64)
    )
    ends = np.r_[starts[1:], face_count] if face_count else starts
    first = order[starts]
    paired = ends - starts == 2
    second = np.full(first.shape, -1, dtype=np.int64)
    second[paired] = order[starts[paired] + 1]
    # Match original grouped insertion order, including first-owner selection.
    representative = np.minimum(first[paired], second[paired])
    second[paired] = np.maximum(first[paired], second[paired])
    first[paired] = representative
    first_cells = np.searchsorted(offsets, first, side="right") - 1
    required = ~paired
    required[paired] = (
        cell_regions[first_cells[paired]]
        != cell_regions[np.searchsorted(offsets, second[paired], side="right") - 1]
    )
    representatives = first[required]
    partners = second[required]
    boundary_order = np.argsort(representatives, kind="stable")
    representatives, partners = representatives[boundary_order], partners[boundary_order]
    del keys, order, sorted_keys, equal, starts, ends, first, second
    del paired, required, boundary_order, representative, first_cells

    def face_data(
        face_row: int, *, include_chart: bool = True
    ) -> tuple[int, int, tuple, tuple, int]:
        cell = int(np.searchsorted(offsets, face_row, side="right") - 1)
        local_face = int(face_row - offsets[cell])
        block_index = next(
            i for i, (start, stop, *_) in enumerate(blocks) if start <= cell < stop
        )
        start, _stop, block, element, route, vertices, faces, signs = blocks[block_index]
        row = cell - start
        local = geometry.source_coordinates(route[row])
        basis = coordinate_effective_basis(element, cell_row=row)
        if len(basis) != len(local):
            raise ValueError(
                "Exact coordinate basis and local source bank widths differ."
            )
        components = tuple(
            tuple(local[dof][axis] for dof in range(len(local)))
            for axis in range(len(local[0]))
        )
        expressions = linear_combinations(basis, components)
        slots = faces[local_face]
        ref = reference_cell_topology(block.cell_kind)
        points = tuple(
            _fraction_point(
                _reference_corner_image(expressions, ref.family, vertices[slot])
            )
            for slot in slots
        )
        if any(point is None for point in points):
            raise SourceCoverageUnavailable(
                "Target boundary map uses coefficients outside the rational exact plane owner.",
                status="unresolved",
            )
        chart = (
            tuple(
                _point3(tuple(evaluate_map(expressions, vertices[slot])))
                for slot in slots
            )
            if include_chart
            else ()
        )
        return cell, local_face, points, chart, signs[local_face]

    # A plain, cell-invariant finite element with a binary64 coefficient bank
    # has rational corner images whenever every basis value at every corner is
    # rational. Prove that once per block; other coordinate elements and exact
    # banks are checked cell by cell, including wholly interior cells.
    generic_ranges: list[tuple[int, int]] = []
    for start, stop, block, element, route, vertices, faces, _signs in blocks:
        fast = False
        if (
            isinstance(element, FiniteElementSpec)
            and geometry.exact_coordinate_bank is None
        ):
            try:
                basis = coordinate_effective_basis(element)
                coordinates = np.asarray(geometry.coordinates, dtype=np.float64)
                fast = (
                    route.ndim == 2
                    and route.shape == (stop - start, len(basis))
                    and (
                        not route.size
                        or (np.min(route) >= 0 and np.max(route) < len(coordinates))
                    )
                    and np.all(np.isfinite(coordinates))
                    and all(
                        isinstance(polynomial, dict)
                        and all(
                            isinstance(coefficient, (int, Fraction))
                            and not isinstance(coefficient, bool)
                            for coefficient in polynomial.values()
                        )
                        for polynomial in basis
                    )
                    and all(
                        _fraction_point(
                            tuple(evaluate_expression(value, point) for value in basis)
                        )
                        is not None
                        for point in vertices
                    )
                )
            except (TypeError, ValueError, IndexError, ArithmeticError):
                fast = False
        if not fast:
            generic_ranges.append((int(offsets[start]), int(offsets[stop])))
            for face_row in range(int(offsets[start]), int(offsets[stop])):
                try:
                    face_data(face_row, include_chart=False)
                except SourceCoverageUnavailable:
                    raise
                except (TypeError, ValueError, IndexError, ArithmeticError) as error:
                    raise SourceCoverageUnavailable(
                        f"Target boundary extraction failed: {error}", status="unresolved"
                    ) from error

    source_by_key = _SourceTriangles(source_vertices, source_face_rows)
    target_by_source = np.full(len(source_face_rows), -1, dtype=np.int64)
    unmatched = False
    for rep_raw, partner_raw in zip(representatives, partners, strict=True):
        rep, partner = int(rep_raw), int(partner_raw)
        try:
            cell, local_face, points, _chart, _sign = face_data(rep, include_chart=False)
            if partner >= 0:
                other_cell, other_face, other_points, _other_chart, _other_sign = (
                    face_data(partner, include_chart=False)
                )

                def ids(row: int, face: int) -> tuple[int, ...]:
                    start, _stop, block, _element, _route, _vertices, faces, _signs = (
                        next(b for b in blocks if b[0] <= row < b[1])
                    )
                    return tuple(
                        int(
                            vertex_ids[int(np.asarray(block.vertices)[row - start, slot])]
                        )
                        for slot in faces[face]
                    )

                if dict(zip(ids(cell, local_face), points, strict=True)) != dict(
                    zip(ids(other_cell, other_face), other_points, strict=True)
                ):
                    raise SourceCoverageUnavailable(
                        "Adjacent cells disagree on exact coordinates at their shared facet vertices.",
                        status="unresolved",
                    )
        except SourceCoverageUnavailable:
            raise
        except (TypeError, ValueError, IndexError, ArithmeticError) as error:
            raise SourceCoverageUnavailable(
                f"Target boundary extraction failed: {error}", status="unresolved"
            ) from error
        source_row = source_by_key.get(_triangle3(points))
        if source_row is None:
            unmatched = True
        elif target_by_source[source_row] != -1:
            raise SourceCoverageUnavailable(
                "Implicit target repeats a boundary triangle.", status="violated"
            )
        else:
            target_by_source[source_row] = rep
    if unmatched or np.any(target_by_source < 0):
        raise SourceCoverageUnavailable(
            "Implicit target boundary is not an exact conforming subdivision of the source-root scaffold.",
            status="violated",
        )
    for start, stop in generic_ranges:
        for face_row in range(start, stop):
            face_data(face_row)

    id_order = np.argsort(cell_ids, kind="stable")
    sorted_ids = cell_ids[id_order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise SourceCoverageUnavailable(
            "Target cell identities are not unique.", status="violated"
        )
    if (
        witness.implicit_cell_keys.shape != (witness.cell_global_ids.size, 4)
        or witness.implicit_root_edge_keys.shape
        != (witness.cell_global_ids.size, 3, 2, 3)
        or witness.cell_global_ids.size != len(representatives)
    ):
        raise SourceCoverageUnavailable(
            "Implicit witness omits target faces or retained root ancestry.",
            status="violated",
        )
    # Map either owner to the canonical representative, keeping one bit per
    # required face rather than a Python tuple/set per volume face.
    lookup_rows = np.concatenate((representatives, partners[partners >= 0]))
    lookup_owner = np.concatenate(
        (np.arange(len(representatives), dtype=np.int64), np.flatnonzero(partners >= 0))
    )
    perm = np.argsort(lookup_rows, kind="stable")
    lookup_rows, lookup_owner = lookup_rows[perm], lookup_owner[perm]
    seen = np.zeros(len(representatives), dtype=np.bool_)
    witnessed_triangles = np.zeros(len(source_face_rows), dtype=np.bool_)
    for witness_row, (cell_raw, face_raw) in enumerate(
        zip(witness.cell_global_ids, witness.local_entity_indices, strict=True)
    ):
        cell_id, local_face = int(cell_raw), int(face_raw)
        pos = int(np.searchsorted(sorted_ids, cell_id))
        cell = (
            int(id_order[pos])
            if pos < len(sorted_ids) and sorted_ids[pos] == cell_id
            else -1
        )
        face_row = (
            int(offsets[cell] + local_face)
            if cell >= 0 and 0 <= local_face < offsets[cell + 1] - offsets[cell]
            else -1
        )
        pos = int(np.searchsorted(lookup_rows, face_row))
        owner = (
            int(lookup_owner[pos])
            if pos < len(lookup_rows) and lookup_rows[pos] == face_row
            else -1
        )
        if owner < 0 or seen[owner]:
            raise SourceCoverageUnavailable(
                "Implicit witness repeats or omits a target boundary facet.",
                status="violated",
            )
        seen[owner] = True
        _cell, _face, _points, chart, sign = face_data(face_row)
        target_triangle_key = tuple(sorted(chart))
        source_row = source_by_key.get(target_triangle_key)
        if source_row is None or witnessed_triangles[source_row]:
            raise SourceCoverageUnavailable(
                "Implicit target facet has no unique source root triangle.",
                status="violated",
            )
        witnessed_triangles[source_row] = True
        source_cell = int(triangle_cells[source_row])
        if tuple(int(v) for v in witness.implicit_cell_keys[witness_row]) != tuple(
            int(v) for v in scaffold.cell_indices[source_cell]
        ):
            raise SourceCoverageUnavailable(
                "Implicit witness changes its source root-cell ancestry.",
                status="violated",
            )
        expected = {root_keys[int(triangle_roots[source_row, i])] for i in range(3)}
        edges = np.asarray(witness.implicit_root_edge_keys[witness_row], dtype=np.int64)
        if edges.shape != (3, 2, 3):
            raise SourceCoverageUnavailable(
                "Implicit witness root edges omit endpoint ancestry.", status="violated"
            )
        actual = set()
        for edge in edges:
            a, b = tuple(int(v) for v in edge[0]), tuple(int(v) for v in edge[1])
            actual.add((a, b) if a < b else (b, a))
        if actual != expected:
            raise SourceCoverageUnavailable(
                "Implicit witness changes its source root-edge ancestry.",
                status="violated",
            )
        if int(witness.orientations[witness_row]) * sign != 1:
            raise SourceCoverageUnavailable(
                "Implicit boundary chart reverses the source outward orientation.",
                status="violated",
            )
        source_order = _triangle3(
            tuple(
                tuple(
                    _fraction(v) for v in source_vertices[source_face_rows[source_row, i]]
                )
                for i in range(3)
            )
        )
        _c, _f, target_points, _chart, _s = face_data(int(target_by_source[source_row]))
        target_order = _triangle3(target_points)
        target_normal = _cross3(target_order)
        if _dot3(_cross3(source_order), target_normal) <= 0:
            raise SourceCoverageUnavailable(
                "Implicit target boundary winding reverses a source root triangle.",
                status="violated",
            )
        gradient = bounds.boxes(
            np.min(np.asarray(target_order, dtype=np.float64), axis=0)[None, :],
            np.max(np.asarray(target_order, dtype=np.float64), axis=0)[None, :],
        )
        if (
            _strict_sign(
                _interval_dot(
                    gradient.gradient_lower[0], gradient.gradient_upper[0], target_normal
                )
            )
            <= 0
        ):
            raise SourceCoverageUnavailable(
                "Implicit source boundary normal is not certified outward on a root triangle.",
                status="unresolved",
            )
    if not np.all(seen) or not np.all(witnessed_triangles):
        raise SourceCoverageUnavailable(
            "Implicit root chart witness does not cover the complete oriented boundary.",
            status="violated",
        )
