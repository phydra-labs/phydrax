#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Embedding of exact affine restrictions of independently injective source maps.

Scientific parent identity is declared by CellGeometryRestrictionSource, never
inferred from equal coefficients. Its declared identity is checked against the
actual source expressions, routes and affine charts. Rational pyramid children
are certified in the physical parent reference cell, retaining its apex, rather
than replacing a rational restriction by nodal polynomial interpolation.
"""

from __future__ import annotations

from fractions import Fraction
from typing import cast, TYPE_CHECKING

import numpy as np

from ..discretization._cell_geometry import (
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
)
from ..discretization._coordinate_enclosure import (
    _COORDINATE_BUDGET,
    affine_arguments,
    constant,
    coordinate_expressions,
    coordinate_reference_chain,
    CoordinateSourceBank,
    evaluate,
    Expression,
    expression_add,
    expression_node_count,
    expression_reference_evaluate,
    expression_scale,
    expression_sum,
    multi_affine_coordinates,
    RationalPolynomial,
    restrict_chart_expressions,
)
from ..discretization._reference_cell import reference_cell_topology
from ._mapped_embedding import (
    _bounds,
    _Cell,
    _domain,
    _injective,
    _pair_decision,
    _straight_boundary_images,
    _tensor_component_injectivity,
    _trace_continuity,
    _vertices,
)


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh
    from ._mesh_certificates import _EmbeddingState, MeshCertificateLimits


def _inside_reference(point: tuple[Fraction, ...], kind: str) -> bool:
    if any(value < 0 or value > 1 for value in point):
        return False
    if kind in ("triangle", "tetrahedron"):
        return sum(point) <= 1
    if kind == "prism":
        return point[0] + point[1] <= 1
    if kind == "pyramid":
        x, y, z = point
        return z / 2 <= x <= 1 - z / 2 and z / 2 <= y <= 1 - z / 2
    return True


def _root_traces_equal(first: _Cell, second: _Cell) -> bool:
    shared = set(first.vertices).intersection(second.vertices)
    if not shared:
        return True
    topologies = (
        reference_cell_topology(first.kind),
        reference_cell_topology(second.kind),
    )
    for identifier in shared:
        points = []
        for cell, topology in zip((first, second), topologies, strict=True):
            point = tuple(
                Fraction(float(value))
                for value in topology.vertices[cell.vertices.index(identifier)]
            )
            if cell.kind == "pyramid" and point[2] == 1:
                point = (Fraction(1, 2), Fraction(1, 2), Fraction(1))
            points.append(
                tuple(
                    expression_reference_evaluate(value, point, cell.domain)
                    for value in cell.coordinates
                )
            )
        if points[0] != points[1]:
            return False
    for dimension in range(1, first.dimension):
        for entity in topologies[0].entities[dimension]:
            identifiers = tuple(first.vertices[index] for index in entity)
            if not set(identifiers) <= shared:
                continue
            matches = [
                candidate
                for candidate in topologies[1].entities[dimension]
                if {second.vertices[index] for index in candidate} == set(identifiers)
            ]
            if len(matches) != 1:
                return False
            restrictions = []
            for cell, topology in zip((first, second), topologies, strict=True):
                corners = np.asarray(topology.vertices, dtype=np.float64)[
                    [cell.vertices.index(identifier) for identifier in identifiers]
                ]
                matrix = (
                    (corners[[1]] - corners[0]).T
                    if dimension == 1
                    else (corners[[1, 2 if len(identifiers) == 3 else 3]] - corners[0]).T
                )
                target_kind = (
                    "interval"
                    if dimension == 1
                    else "triangle"
                    if len(identifiers) == 3
                    else "quadrilateral"
                )
                restrictions.append(
                    restrict_chart_expressions(
                        cell.coordinates, cell.kind, target_kind, corners[0], matrix
                    )
                )
            if (
                any(value is None for value in restrictions)
                or restrictions[0] != restrictions[1]
            ):
                return False
    return True


type _Point2 = tuple[Fraction, Fraction]
type _Triangle2 = tuple[_Point2, _Point2, _Point2]


def _orient2(first: _Point2, second: _Point2, third: _Point2, /) -> Fraction:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _between(point: _Point2, first: _Point2, second: _Point2, /) -> bool:
    return all(
        min(a, b) <= value <= max(a, b)
        for value, a, b in zip(point, first, second, strict=True)
    )


def _affine_triangle_chart_points(cell: _Cell, /) -> _Triangle2 | None:
    if (
        cell.kind != "triangle"
        or cell.dimension != 2
        or len(cell.coordinates) != 2
        or any(
            isinstance(value, RationalPolynomial)
            or any(sum(index) > 1 for index in value)
            for value in cell.coordinates
        )
    ):
        return None
    return cast(
        _Triangle2,
        tuple(
            tuple(
                expression_reference_evaluate(value, reference, cell.domain)
                for value in cell.coordinates
            )
            for reference in cell.reference_vertices
        ),
    )


def _affine_triangle_chart_decision(
    first: _Cell,
    second: _Cell,
    /,
    *,
    prepared_points: tuple[_Triangle2, _Triangle2] | None = None,
) -> str | None:
    """Exact planar contact of affine child charts in their parent reference."""
    if prepared_points is None:
        first_points = _affine_triangle_chart_points(first)
        second_points = _affine_triangle_chart_points(second)
        if first_points is None or second_points is None:
            return None
        points = first_points, second_points
    else:
        points = prepared_points
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(220, 16384)
    triangle_orientations = tuple(_orient2(*triangle) for triangle in points)
    if any(value == 0 for value in triangle_orientations):
        return None
    edge_slots = ((0, 1), (1, 2), (2, 0))
    edge_signs = tuple(
        tuple(
            tuple(
                _orient2(
                    points[owner][edge[0]],
                    points[owner][edge[1]],
                    point,
                )
                for point in points[1 - owner]
            )
            for edge in edge_slots
        )
        for owner in range(2)
    )
    cells = (first, second)
    for first_edge, first_slots in enumerate(edge_slots):
        a, b = (points[0][slot] for slot in first_slots)
        first_ids = {cells[0].vertices[slot] for slot in first_slots}
        for second_edge, second_slots in enumerate(edge_slots):
            c, d = (points[1][slot] for slot in second_slots)
            second_ids = {cells[1].vertices[slot] for slot in second_slots}
            ab_c, ab_d = (edge_signs[0][first_edge][slot] for slot in second_slots)
            cd_a, cd_b = (edge_signs[1][second_edge][slot] for slot in first_slots)
            if ab_c * ab_d < 0 and cd_a * cd_b < 0:
                return "overlap"
            if ab_c == ab_d == cd_a == cd_b == 0:
                axis = int(abs(b[1] - a[1]) > abs(b[0] - a[0]))
                lower = max(min(a[axis], b[axis]), min(c[axis], d[axis]))
                upper = min(max(a[axis], b[axis]), max(c[axis], d[axis]))
                if lower < upper and first_ids != second_ids:
                    return "overlap"
            for point, sign, point_id, start, end in (
                (c, ab_c, cells[1].vertices[second_slots[0]], a, b),
                (d, ab_d, cells[1].vertices[second_slots[1]], a, b),
                (a, cd_a, cells[0].vertices[first_slots[0]], c, d),
                (b, cd_b, cells[0].vertices[first_slots[1]], c, d),
            ):
                if sign != 0 or not _between(point, start, end):
                    continue
                if point_id not in (first_ids & second_ids) or point not in (start, end):
                    return "overlap"
    for owner, other in ((0, 1), (1, 0)):
        orientation = triangle_orientations[other]
        for point_slot, (identifier, point) in enumerate(
            zip(cells[owner].vertices, points[owner], strict=True)
        ):
            signs = tuple(edge_signs[other][edge][point_slot] for edge in range(3))
            inside = (
                all(value > 0 for value in signs)
                if orientation > 0
                else all(value < 0 for value in signs)
            )
            if inside:
                return "overlap"
            enclosed = (
                all(value >= 0 for value in signs)
                if orientation > 0
                else all(value <= 0 for value in signs)
            )
            if enclosed and identifier not in cells[other].vertices:
                return "overlap"
    return "separated"


def certify_restricted_embedding(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
) -> None:
    from contextlib import nullcontext

    from ._mesh_certificates import _candidate_pairs

    record = geometry.restriction_source
    if record is None:
        raise ValueError("Restriction embedding requires its scientific source record.")
    ledger = _COORDINATE_BUDGET.get()
    elements, routes, _ = geometry.resolve(mesh)
    values = geometry.source_coordinates()
    parents: dict[int, tuple[_Cell, CoordinateSourceBank, str]] = {}
    children: dict[int, list[tuple[int, _Cell]]] = {}
    cursor = 0
    state.checks.extend(
        (
            "mapped_restriction_source",
            "mapped_restriction_reference_domain",
            "mapped_restriction_embedding",
        )
    )
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        source, reference_coordinates = coordinate_reference_chain(element)
        if ledger is not None:
            ledger.retain_basis(reference_coordinates)
        source_kind = source.cell_kind
        source_dimension = reference_cell_topology(source_kind).dimension
        parent_ids = np.asarray(record.block_parent_cell_ids[block.name], dtype=np.int64)
        vertex_ids = np.asarray(
            record.block_parent_vertex_ids[block.name], dtype=np.int64
        )
        # A multi-affine chart of a box child is the interpolant of its exact
        # reference corner images, shared by every row of the block.
        chart_corners = None
        if block.cell_kind in ("quadrilateral", "hexahedron") and all(
            isinstance(value, dict)
            and all(exponent <= 1 for index in value for exponent in index)
            for value in reference_coordinates
        ):
            chart_corners = tuple(
                tuple(
                    expression_reference_evaluate(value, point, _domain(block.cell_kind))
                    for value in reference_coordinates
                )
                for point in _vertices(block.cell_kind)
            )
            if ledger is not None:
                ledger.retain_basis(chart_corners)
        # The reference-domain proof depends only on the block chart; it is
        # decided once, at the block's first row, in the original check order.
        domain_proved = False
        for row, target_vertices, parent_id, source_vertices in zip(
            np.asarray(route),
            np.asarray(block.vertices),
            parent_ids.tolist(),
            vertex_ids,
            strict=True,
        ):
            source_vertices = np.asarray(source_vertices, dtype=np.int64)
            local = tuple(values[index] for index in row)
            parent_vertices = tuple(source_vertices[source_vertices >= 0].tolist())
            if len(parent_vertices) != len(reference_cell_topology(source_kind).vertices):
                state.add(
                    "restriction_source_corner_identity",
                    "unresolved",
                    "cell",
                    cell_ids[cursor : cursor + 1],
                )
                return
            if parent_id in parents:
                previous, old_local, element_id = parents[parent_id]
                if (
                    source.element_id != element_id
                    or previous.vertices != parent_vertices
                    or old_local != local
                ):
                    state.add(
                        "restriction_source_identity_mismatch",
                        "violated",
                        "cell",
                        cell_ids[cursor : cursor + 1],
                    )
                    return
            else:
                # Each root is proved once; its single-cell workspace is released
                # and only the exact records owned by the root cell are retained.
                with ledger.temporary_scope() if ledger is not None else nullcontext():
                    prepared = multi_affine_coordinates(source, local)
                    polynomials: tuple[Expression, ...] | None
                    if prepared is None:
                        polynomials, corner_images = (
                            coordinate_expressions(source, local),
                            None,
                        )
                    else:
                        polynomials, corner_images = prepared
                    if polynomials is None:
                        state.add(
                            "restriction_source_enclosure",
                            "unresolved",
                            "cell",
                            cell_ids[cursor : cursor + 1],
                        )
                        return
                    root = _Cell(
                        polynomials,
                        _domain(source_kind),
                        source_kind,
                        source_dimension,
                        parent_vertices,
                        _vertices(source_kind),
                        corner_images,
                    )
                    reason = _injective(root, limits, state)
                if ledger is not None:
                    ledger.retain_basis(root.coordinates)
                    ledger.retain_basis(root.vertices)
                    ledger.retain_basis(local)
                    if root.corner_images is not None:
                        ledger.retain_basis(root.corner_images)
                parents[parent_id] = (root, local, source.element_id)
                if reason is not None:
                    state.add(reason, "unresolved", "cell", cell_ids[cursor : cursor + 1])
            if not domain_proved:
                with ledger.temporary_scope() if ledger is not None else nullcontext():
                    if isinstance(
                        element,
                        (
                            PolynomialComposedCellGeometryElement,
                            RationalComposedCellGeometryElement,
                        ),
                    ) or (
                        isinstance(element, RestrictedCellGeometryElement)
                        and element.source_element.element_id != source.element_id
                    ):
                        from ._mapped_coverage import nonnegative, SubdivisionLedger

                        constraints = _reference_constraints(
                            reference_coordinates, source_kind, source_dimension
                        )
                        for constraint in constraints:
                            if (
                                expression_node_count(
                                    constraint, _domain(block.cell_kind), source_dimension
                                )
                                > limits.maximum_bernstein_nodes
                            ):
                                state.add(
                                    "restriction_bernstein_node_budget",
                                    "unresolved",
                                    "cell",
                                    cell_ids[cursor : cursor + 1],
                                )
                                return
                            work = SubdivisionLedger()
                            contained = nonnegative(
                                constraint,
                                _domain(block.cell_kind),
                                source_dimension,
                                limits.maximum_bernstein_nodes,
                                limits.maximum_subdivision_depth,
                                limits.maximum_subdivision_pieces
                                - state.subdivision_pieces,
                                work,
                            )
                            state.subdivision_pieces += work.pieces
                            if not contained:
                                state.add(
                                    "restriction_reference_domain",
                                    "unresolved",
                                    "cell",
                                    cell_ids[cursor : cursor + 1],
                                )
                                return
                    else:
                        matrix = (
                            np.asarray(element.matrix, dtype=np.float64)
                            if isinstance(element, RestrictedCellGeometryElement)
                            else np.eye(source_dimension, dtype=np.float64)
                        )
                        offset = (
                            np.asarray(element.offset, dtype=np.float64)
                            if isinstance(element, RestrictedCellGeometryElement)
                            else np.zeros((source_dimension,), dtype=np.float64)
                        )
                        arguments = affine_arguments(offset, matrix)
                        for vertex in reference_cell_topology(block.cell_kind).vertices:
                            point = tuple(Fraction(float(value)) for value in vertex)
                            image = tuple(evaluate(value, point) for value in arguments)
                            if not _inside_reference(image, source_kind):
                                state.add(
                                    "restriction_reference_domain",
                                    "unresolved",
                                    "cell",
                                    cell_ids[cursor : cursor + 1],
                                )
                                return
                domain_proved = True
            chart = _Cell(
                reference_coordinates,
                _domain(block.cell_kind),
                block.cell_kind,
                mesh.topological_dimension,
                tuple(target_vertices.tolist()),
                _vertices(block.cell_kind),
                chart_corners,
            )
            if ledger is not None:
                ledger.retain_basis(chart.vertices)
            children.setdefault(parent_id, []).append((cursor, chart))
            cursor += 1
    # Reference-cell contacts prove actual contacts because every root map is
    # independently injective. This includes source pyramid apex identifications.
    all_charts = [chart for members in children.values() for _, chart in members]
    ordered_charts: list[_Cell | None] = [None] * len(all_charts)
    groups = np.full((len(all_charts),), -1, dtype=np.int64)
    for parent, members in children.items():
        for target, chart in members:
            ordered_charts[target] = chart
            groups[target] = parent
    if any(chart is None for chart in ordered_charts):
        raise ValueError("Restriction source rows do not cover target cells.")
    _trace_continuity(
        state,
        mesh,
        [chart for chart in ordered_charts if chart is not None],
        groups,
        shared_charts=True,
    )
    # A chart's exact expressions are shared by its block. Its bounds, and a pair
    # separation proved for two charts with the same shared local corners, are
    # therefore the same exact proof for every parent that repeats them.
    chart_bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    chart_points: dict[int, _Triangle2] = {}
    separated: set[tuple[int, int, str, str, tuple[tuple[int, int], ...]]] = set()
    for members in children.values():
        if len(members) <= 1:
            continue
        for _, chart in members:
            chart_key = id(chart.coordinates)
            if chart_key not in chart_bounds:
                with ledger.temporary_scope() if ledger is not None else nullcontext():
                    chart_bounds[chart_key] = _bounds(
                        chart,
                        np.zeros((chart.dimension,), dtype=np.float64),
                        np.eye(chart.dimension, dtype=np.float64),
                    )
            if chart_key not in chart_points:
                with ledger.temporary_scope() if ledger is not None else nullcontext():
                    prepared_points = _affine_triangle_chart_points(chart)
                if prepared_points is not None:
                    if ledger is not None:
                        ledger.retain_basis(prepared_points)
                    chart_points[chart_key] = prepared_points
        boxes = [chart_bounds[id(chart.coordinates)] for _, chart in members]
        lower = np.stack([value[0] for value in boxes])
        upper = np.stack([value[1] for value in boxes])
        a_rows, b_rows, exceeded = _candidate_pairs(
            lower, upper, limits.maximum_candidate_pairs - state.candidate_pairs
        )
        state.candidate_pairs += a_rows.size
        if exceeded:
            state.add("restriction_candidate_pair_budget", "unresolved", "mesh")
        for a, b in zip(a_rows.tolist(), b_rows.tolist(), strict=True):
            target_a, first = members[a]
            target_b, second = members[b]
            pattern = tuple(
                sorted(
                    (first.vertices.index(vertex), second.vertices.index(vertex))
                    for vertex in set(first.vertices).intersection(second.vertices)
                )
            )
            key = (
                id(first.coordinates),
                id(second.coordinates),
                first.kind,
                second.kind,
                pattern,
            )
            if key in separated:
                continue
            with ledger.temporary_scope() if ledger is not None else nullcontext():
                first_points = chart_points.get(id(first.coordinates))
                second_points = chart_points.get(id(second.coordinates))
                decision = _affine_triangle_chart_decision(
                    first,
                    second,
                    prepared_points=(
                        (first_points, second_points)
                        if first_points is not None and second_points is not None
                        else None
                    ),
                )
                if decision is None:
                    decision, work = _pair_decision(
                        first,
                        second,
                        limits,
                        limits.maximum_subdivision_pieces - state.subdivision_pieces,
                    )
                else:
                    work = 0
            state.subdivision_pieces += work
            if decision == "separated":
                separated.add(key)
            elif decision == "overlap":
                state.add(
                    "restriction_reference_overlap",
                    "violated",
                    "cell",
                    cell_ids[[target_a, target_b]],
                )
            else:
                state.add(decision, "unresolved", "cell", cell_ids[[target_a, target_b]])
    roots = list(parents)
    root_cells = [parents[parent_id][0] for parent_id in roots]
    if len(root_cells) <= 1:
        return
    if _affine_tetrahedral_root_embedding(
        state, root_cells, roots, record.source_geometry_id, limits
    ):
        return
    with ledger.temporary_scope() if ledger is not None else nullcontext():
        straight = _straight_multi_affine_root_embedding(
            state, root_cells, roots, record.source_geometry_id, limits
        )
    if straight:
        return
    with ledger.temporary_scope() if ledger is not None else nullcontext():
        atlas_labels = _root_atlas_labels(
            state, root_cells, roots, record.source_geometry_id, limits
        )
        root_boxes = [
            _bounds(
                root,
                np.zeros((root.dimension,), dtype=np.float64),
                np.eye(root.dimension, dtype=np.float64),
            )
            for root in root_cells
        ]
    a_rows, b_rows, exceeded = _candidate_pairs(
        np.stack([box[0] for box in root_boxes]),
        np.stack([box[1] for box in root_boxes]),
        limits.maximum_candidate_pairs - state.candidate_pairs,
    )
    state.candidate_pairs += a_rows.size
    if exceeded:
        state.add("restriction_root_candidate_budget", "unresolved", "mesh")
    for a, b in zip(a_rows.tolist(), b_rows.tolist(), strict=True):
        if atlas_labels[a] >= 0 and atlas_labels[a] == atlas_labels[b]:
            continue
        with ledger.temporary_scope() if ledger is not None else nullcontext():
            traced = _root_traces_equal(root_cells[a], root_cells[b])
            decision, work = (
                _pair_decision(
                    root_cells[a],
                    root_cells[b],
                    limits,
                    limits.maximum_subdivision_pieces - state.subdivision_pieces,
                )
                if traced
                else ("restriction_root_trace_mismatch", 0)
            )
        if not traced:
            affected = [
                target for root in (roots[a], roots[b]) for target, _ in children[root]
            ]
            state.add(
                "restriction_root_trace_mismatch", "violated", "cell", cell_ids[affected]
            )
            continue
        state.subdivision_pieces += work
        if decision != "separated":
            affected = [
                target for root in (roots[a], roots[b]) for target, _ in children[root]
            ]
            # A root overlap need not lie in the retained restricted children.
            # It cannot be called a child violation without an inclusion witness.
            state.add(
                "restriction_root_embedding_premise"
                if decision == "overlap"
                else decision,
                "unresolved",
                "cell",
                cell_ids[affected],
            )


def _affine_tetrahedral_root_embedding(
    state: _EmbeddingState,
    roots: list[_Cell],
    identifiers: list[int],
    source_geometry_id: str,
    limits: MeshCertificateLimits,
    /,
) -> bool:
    """Use the linear theorem only for an exactly identical declared root image.

    Full affine expressions and exact binary64 corner images establish equality
    of whole tetrahedra, not a sampled corner surrogate. Shared scientific vertex
    identity establishes every full affine edge/face trace. All other source
    expressions retain the generic root theorem.
    """
    from ..discretization._cell_mesh import CellBlock, CellMesh
    from ._mesh_certificates import (
        _EmbeddingState,
        _facet_entity_ids,
        _mesh_facets,
        _orient3d,
        _pairing_findings,
        _volume_embedding,
    )

    if any(
        root.kind != "tetrahedron"
        or root.dimension != 3
        or len(root.coordinates) != 3
        or any(
            not isinstance(expression, dict)
            or any(len(index) != 3 or sum(index) > 1 for index in expression)
            for expression in root.coordinates
        )
        for root in roots
    ):
        return False
    positions: dict[int, tuple[Fraction, ...]] = {}
    owners: dict[int, int] = {}
    rounded: dict[int, tuple[float, ...]] = {}
    for root_id, root in zip(identifiers, roots, strict=True):
        for vertex_id, corner in zip(root.vertices, root.reference_vertices, strict=True):
            image = tuple(
                expression_reference_evaluate(value, corner, root.domain)
                for value in root.coordinates
            )
            try:
                values = tuple(float(value) for value in image)
            except OverflowError:
                return False
            if any(
                not np.isfinite(value) or Fraction.from_float(value) != exact
                for value, exact in zip(values, image, strict=True)
            ):
                return False
            known = positions.get(vertex_id)
            if known is not None and known != image:
                state.add(
                    "restriction_source_corner_trace_mismatch",
                    "violated",
                    "source_cell",
                    tuple(sorted({owners[vertex_id], root_id})),
                )
                return True
            positions[vertex_id], rounded[vertex_id], owners[vertex_id] = (
                image,
                values,
                root_id,
            )
    vertex_ids = np.asarray(sorted(positions), dtype=np.int64)
    rows = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    coordinates = np.asarray(
        [rounded[int(identifier)] for identifier in vertex_ids], dtype=np.float64
    )
    cells = np.asarray(
        [tuple(rows[identifier] for identifier in root.vertices) for root in roots],
        dtype=np.int64,
    )
    corners = coordinates[cells]
    orientation, certain = _orient3d(
        corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
    )
    if not np.all(certain & (orientation > 0)):
        return False
    view = CellMesh(
        coordinates,
        (
            CellBlock(
                "declared_affine_source_roots",
                "tetrahedron",
                cells,
                global_ids=np.asarray(identifiers, dtype=np.int64),
            ),
        ),
        vertex_global_ids=vertex_ids,
        numeric_version=source_geometry_id,
    )
    facets = _mesh_facets(view)
    premise = _EmbeddingState(
        [],
        [],
        candidate_pairs=state.candidate_pairs,
        ray_tests=state.ray_tests,
        subdivision_pieces=state.subdivision_pieces,
    )
    _pairing_findings(premise, view, facets, np.empty((0,), dtype=np.int64))
    _volume_embedding(premise, view, coordinates, facets, limits)
    state.candidate_pairs = premise.candidate_pairs
    state.ray_tests = premise.ray_tests
    state.subdivision_pieces = premise.subdivision_pieces
    state.subdivision_depth = max(state.subdivision_depth, premise.subdivision_depth)
    state.checks.extend(f"restriction_affine_root:{check}" for check in premise.checks)
    facet_ids = _facet_entity_ids(view, facets.rows)
    root_ids = np.asarray(identifiers, dtype=np.int64)
    for finding in premise.findings:
        if finding.entity_kind == "cell":
            affected = np.asarray(finding.entity_ids, dtype=np.int64)
        elif finding.entity_kind == "facet":
            selected = np.isin(facet_ids, np.asarray(finding.entity_ids, dtype=np.int64))
            affected = np.unique(root_ids[facets.cells[selected]])
        else:
            affected = root_ids
        # Failed full roots are not a retained-child overlap witness.
        state.add(
            f"restriction_affine_root:{finding.check}",
            "unresolved",
            "source_cell",
            affected,
        )
    return True


def _straight_multi_affine_root_embedding(
    state: _EmbeddingState,
    roots: list[_Cell],
    identifiers: list[int],
    source_geometry_id: str,
    limits: MeshCertificateLimits,
    /,
) -> bool:
    """Reuse the straight-boundary degree theorem of direct mapped meshes on roots.

    Every root is a multi-affine quadrilateral or hexahedron whose local
    injectivity is already proved. When the exact root corner images are binary64
    points, shared facets have identical traces and every boundary facet image is
    its straight or planar carrier, the root union is embedded exactly when the
    polygonal view of those points is. Any other case returns ``False`` and keeps
    the generic root atlas and pair proofs.
    """
    from ..discretization._cell_mesh import CellBlock, CellMesh
    from ._mesh_certificates import (
        _EmbeddingState,
        _facet_entity_ids,
        _mesh_facets,
        _pairing_findings,
        _volume_embedding,
    )

    if not roots or not state.clean:
        return False
    kind, dimension = roots[0].kind, roots[0].dimension
    if (
        kind not in ("quadrilateral", "hexahedron")
        or dimension not in (2, 3)
        or any(
            root.corner_images is None
            or root.kind != kind
            or root.dimension != dimension
            or len(root.coordinates) != dimension
            for root in roots
        )
    ):
        return False
    positions: dict[int, tuple[Fraction, ...]] = {}
    for root in roots:
        if root.corner_images is None:
            return False
        for vertex_id, image in zip(root.vertices, root.corner_images, strict=True):
            known = positions.get(vertex_id)
            if known is not None and known != image:
                return False
            positions[vertex_id] = image
    vertex_ids = np.asarray(sorted(positions), dtype=np.int64)
    rows = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    rounded = []
    for identifier in vertex_ids.tolist():
        try:
            values = tuple(float(value) for value in positions[identifier])
        except OverflowError:
            return False
        if any(
            not np.isfinite(value) or Fraction.from_float(value) != exact
            for value, exact in zip(values, positions[identifier], strict=True)
        ):
            return False
        rounded.append(values)
    coordinates = np.asarray(rounded, dtype=np.float64)
    vertices = np.asarray(
        [tuple(rows[identifier] for identifier in root.vertices) for root in roots],
        dtype=np.int64,
    )
    view = CellMesh(
        coordinates,
        (
            CellBlock(
                "declared_multi_affine_source_roots",
                kind,
                vertices,
                global_ids=np.asarray(identifiers, dtype=np.int64),
            ),
        ),
        vertex_global_ids=vertex_ids,
        numeric_version=source_geometry_id,
    )
    cells = [
        _Cell(
            root.coordinates,
            root.domain,
            kind,
            dimension,
            tuple(vertices[row].tolist()),
            root.reference_vertices,
            root.corner_images,
        )
        for row, root in enumerate(roots)
    ]
    facets = _mesh_facets(view)
    premise = _EmbeddingState(
        [],
        [],
        candidate_pairs=state.candidate_pairs,
        ray_tests=state.ray_tests,
        subdivision_pieces=state.subdivision_pieces,
    )
    _pairing_findings(premise, view, facets, np.empty((0,), dtype=np.int64))
    _trace_continuity(premise, view, cells)
    if not premise.clean or not _straight_boundary_images(view, cells):
        return False
    premise.checks.append("mapped_straight_boundary_degree")
    _volume_embedding(premise, view, coordinates, facets, limits)
    state.candidate_pairs = premise.candidate_pairs
    state.ray_tests = premise.ray_tests
    state.subdivision_pieces = premise.subdivision_pieces
    state.subdivision_depth = max(state.subdivision_depth, premise.subdivision_depth)
    state.checks.extend(
        f"restriction_multi_affine_root:{check}" for check in premise.checks
    )
    facet_ids = _facet_entity_ids(view, facets.rows)
    root_ids = np.asarray(identifiers, dtype=np.int64)
    for finding in premise.findings:
        if finding.entity_kind == "cell":
            affected = np.asarray(finding.entity_ids, dtype=np.int64)
        elif finding.entity_kind == "facet":
            selected = np.isin(facet_ids, np.asarray(finding.entity_ids, dtype=np.int64))
            affected = np.unique(root_ids[facets.cells[selected]])
        else:
            affected = root_ids
        # A failed root union is not a retained-child overlap witness.
        state.add(
            f"restriction_multi_affine_root:{finding.check}",
            "unresolved",
            "source_cell",
            affected,
        )
    return True


def _reference_constraints(
    coordinates: tuple[Expression, ...], kind: str, dimension: int
) -> tuple[Expression, ...]:
    """Exact convex source inequalities, retaining genuine rational denominators."""
    if kind in ("triangle", "tetrahedron") or kind.startswith("simplex:"):
        return (
            *coordinates,
            expression_add(
                constant(1, dimension), expression_scale(expression_sum(coordinates), -1)
            ),
        )
    if kind == "prism":
        return (
            coordinates[0],
            coordinates[1],
            expression_add(
                constant(1, dimension),
                expression_scale(expression_add(coordinates[0], coordinates[1]), -1),
            ),
            coordinates[2],
            expression_add(constant(1, dimension), expression_scale(coordinates[2], -1)),
        )
    if kind == "pyramid":
        x, y, z = coordinates
        return (
            z,
            expression_add(constant(1, dimension), expression_scale(z, -1)),
            expression_add(x, expression_scale(z, Fraction(-1, 2))),
            expression_add(y, expression_scale(z, Fraction(-1, 2))),
            expression_add(
                constant(1, dimension),
                expression_scale(
                    expression_add(x, expression_scale(z, Fraction(1, 2))), -1
                ),
            ),
            expression_add(
                constant(1, dimension),
                expression_scale(
                    expression_add(y, expression_scale(z, Fraction(1, 2))), -1
                ),
            ),
        )
    return (
        *coordinates,
        *(
            expression_add(constant(1, dimension), expression_scale(value, -1))
            for value in coordinates
        ),
    )


def _root_atlas_labels(
    state: _EmbeddingState,
    roots: list[_Cell],
    identifiers: list[int],
    source_geometry_id: str,
    limits: MeshCertificateLimits,
    /,
) -> np.ndarray:
    """Reuse the full-map injectivity theorem on declared scientific root topology.

    The temporary corner view supplies incidence only. Exact original root
    expressions, not these rounded coordinates, supply every atlas map and
    global extension. No physical coordinate map is reconstructed or replaced.
    """
    from ..discretization._cell_mesh import CellBlock, CellMesh

    absent = np.full((len(roots),), -1, dtype=np.int64)
    if (
        not roots
        or not state.clean
        or roots[0].kind not in ("quadrilateral", "hexahedron")
    ):
        return absent
    kind, dimension = roots[0].kind, roots[0].dimension
    if any(root.kind != kind or root.dimension != dimension for root in roots):
        return absent
    positions: dict[int, tuple[Fraction, ...]] = {}
    owners: dict[int, int] = {}
    for root_id, root in zip(identifiers, roots, strict=True):
        for identifier, corner in zip(
            root.vertices, root.reference_vertices, strict=True
        ):
            image = tuple(
                expression_reference_evaluate(value, corner, root.domain)
                for value in root.coordinates
            )
            previous = positions.get(identifier)
            if previous is not None and previous != image:
                state.add(
                    "restriction_source_corner_trace_mismatch",
                    "violated",
                    "source_cell",
                    tuple(sorted({owners[identifier], root_id})),
                )
                return absent
            positions[identifier] = image
            owners[identifier] = root_id
    vertex_ids = np.asarray(sorted(positions), dtype=np.int64)
    rows = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    coordinates = np.asarray(
        [
            tuple(float(value) for value in positions[int(identifier)])
            for identifier in vertex_ids
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(coordinates)):
        state.add("restriction_root_corner_range", "unresolved", "mesh")
        return absent
    vertices = np.asarray(
        [tuple(rows[identifier] for identifier in root.vertices) for root in roots],
        dtype=np.int64,
    )
    view = CellMesh(
        coordinates,
        (
            CellBlock(
                "declared_source_roots",
                kind,
                vertices,
                global_ids=np.asarray(identifiers, dtype=np.int64),
            ),
        ),
        vertex_global_ids=vertex_ids,
        numeric_version=source_geometry_id,
    )
    cells = [
        _Cell(
            root.coordinates,
            root.domain,
            root.kind,
            root.dimension,
            tuple(vertices[row].tolist()),
            root.reference_vertices,
            root.corner_images,
        )
        for row, root in enumerate(roots)
    ]
    _trace_continuity(state, view, cells)
    if not state.clean:
        return absent
    return _tensor_component_injectivity(state, view, cells, limits)
