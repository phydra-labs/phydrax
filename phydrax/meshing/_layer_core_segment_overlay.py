#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact original source-edge support through target face interiors."""

from __future__ import annotations

from fractions import Fraction
from typing import cast, TYPE_CHECKING

import numpy as np

from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_transition_types import (
    _ExactCellMaps,
    _geometry_map_expressions,
)
from ..discretization._coordinate_enclosure import (
    add,
    axes,
    compose_expression,
    constant,
    evaluate_expression,
    expression_equal,
    scale,
)
from ..discretization._nested_reference import _chart_from_corners
from ..geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
    SourceBoundaryFaceImageWitness,
    SourceBoundaryImagePullback,
)
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from ._layer_core_association import _core_source_identity
from .providers._native_sources import NativeLayerCoreSource


if TYPE_CHECKING:
    from .._bvh import PackedBVH
    from ..geometry._planar_primitives import PolygonBoxes
    from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry


type _Interval = tuple[Fraction, Fraction]
type _Point = tuple[Fraction, ...]
type _Sides = tuple[str | None, str | None]
type _Candidate = tuple[
    Fraction, Fraction, int, int, int, str, _Point, _Point, _Sides, str | None
]


def _subtract(interval: _Interval, covered: list[_Interval], /) -> tuple[_Interval, ...]:
    pending = [interval]
    for lower, upper in sorted(covered):
        pieces = []
        for start, end in pending:
            if upper <= start or lower >= end:
                pieces.append((start, end))
            else:
                if start < lower:
                    pieces.append((start, lower))
                if upper < end:
                    pieces.append((upper, end))
        pending = pieces
    return tuple(pending)


def _reference_point(
    point: _Point, physical: tuple[_Point, ...], reference: tuple[_Point, ...], /
) -> _Point:
    from ._adaptation import _simplex_chart_weights

    weights = _simplex_chart_weights(point, physical, 2)
    if weights is None:
        raise ValueError("A source-edge overlay point leaves its exact target chart.")
    return tuple(
        sum(
            (
                weight * vertex[axis]
                for weight, vertex in zip(weights, reference, strict=True)
            ),
            Fraction(0),
        )
        for axis in range(len(reference[0]))
    )


def _affine_witness_triangle(
    cell_id: int,
    reference: tuple[_Point, ...],
    maps: _ExactCellMaps,
    /,
) -> tuple[_Point, ...]:
    if len(reference) != 3:
        raise ValueError("A source-face support chart must be triangular.")
    chart = _chart_from_corners("triangle", reference)
    images = tuple(
        compose_expression(value, chart) for value in maps.expressions(cell_id)
    )
    coordinates = tuple(
        tuple(evaluate_expression(value, point) for value in images)
        for point in (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
    )
    if any(not isinstance(value, Fraction) for point in coordinates for value in point):
        raise ValueError(
            "Source-edge face overlays require rational physical chart coordinates."
        )
    rational_coordinates = cast(tuple[_Point, ...], coordinates)
    u, v = axes(2)
    for axis, expression in enumerate(images):
        first, second, third = (point[axis] for point in rational_coordinates)
        expected = add(
            add(constant(first, 2), scale(u, second - first)),
            scale(v, third - first),
        )
        if not expression_equal(expression, expected):
            raise ValueError(
                "Source-edge face overlay requires an exact affine target chart."
            )
    return rational_coordinates


def _physical_source_edge(
    source: NativeLayerCoreSource, edge: int, /
) -> tuple[_Point, _Point]:
    """Prove the full original source edge is an affine physical segment."""
    authored = source if source.reference_source is None else source.reference_source
    if source.mapped_domain is None:
        if authored.complex.geometry is not None:
            raise ValueError("Image edge support requires an exact source curve chart.")
        first, last = authored.complex.plc_edge_vertices[edge]
        vertices = np.asarray(authored.complex.vertices, dtype=np.float64)
        return (
            tuple(Fraction.from_float(float(value)) for value in vertices[first]),
            tuple(Fraction.from_float(float(value)) for value in vertices[last]),
        )
    from ._layer_core_strata_replay import mapped_core_edge_curve, source_line_interval

    curve = mapped_core_edge_curve(source, edge)
    values = tuple(
        tuple(evaluate_expression(value, (Fraction(parameter),)) for value in curve)
        for parameter in (0, 1)
    )
    if any(not isinstance(value, Fraction) for point in values for value in point):
        raise ValueError(
            "Image edge source endpoints require exact rational coordinates."
        )
    endpoints = cast(tuple[_Point, _Point], values)
    if source_line_interval(curve, endpoints[0], endpoints[1]) != (
        Fraction(0),
        Fraction(1),
    ):
        raise ValueError("Image edge support requires an exact affine source segment.")
    return endpoints


def _unmapped_face_edge_candidates(
    key: tuple[str, tuple[str, ...]],
    relevant: dict[int, _Sides],
    missing: dict[int, tuple[_Interval, ...]],
    points: tuple[_Point, ...],
    edges: tuple[tuple[int, int], ...],
    triangle: tuple[_Point, ...],
    prepared: dict[
        tuple[str, tuple[str, ...]],
        tuple[tuple[int, ...], tuple[PolygonBoxes, PolygonBoxes], PackedBVH | None]
        | None,
    ],
    /,
) -> set[int] | None:
    """Conservatively project exact affine supports onto two coordinate planes."""
    from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
    from ..geometry._planar_primitives import (
        _DENSE_BOX_PAIRS,
        box_overlap_pairs,
        polygon_boxes,
    )

    try:
        if key not in prepared:
            rows = tuple(edge for edge in relevant if missing[edge])
            first_boxes = polygon_boxes(
                tuple(
                    tuple(
                        (points[vertex][0], points[vertex][1]) for vertex in edges[edge]
                    )
                    for edge in rows
                )
            )
            second_boxes = polygon_boxes(
                tuple(
                    tuple(
                        (points[vertex][1], points[vertex][2]) for vertex in edges[edge]
                    )
                    for edge in rows
                )
            )
            if first_boxes is None or second_boxes is None:
                prepared[key] = None
            else:
                tree = None
                if len(rows) > _DENSE_BOX_PAIRS:
                    lower = np.column_stack((first_boxes[0], second_boxes[0][:, 1]))
                    upper = np.column_stack((first_boxes[1], second_boxes[1][:, 1]))
                    limit = float(np.finfo(np.float32).max) / 2
                    if np.max(np.abs(lower)) <= limit and np.max(np.abs(upper)) <= limit:
                        tree = prepare_bvh(lower, upper, dtype=np.float32)
                prepared[key] = (rows, (first_boxes, second_boxes), tree)
        item = prepared[key]
        if item is None:
            return None
        rows, boxes, tree = item
        targets = tuple(
            polygon_boxes((tuple((point[i], point[j]) for point in triangle),))
            for i, j in ((0, 1), (1, 2))
        )
        if targets[0] is None or targets[1] is None:
            return None
        valid_targets: tuple[PolygonBoxes, PolygonBoxes] = (targets[0], targets[1])
        if len(rows) > _DENSE_BOX_PAIRS:
            lower = np.column_stack((targets[0][0], targets[1][0][:, 1]))
            upper = np.column_stack((targets[0][1], targets[1][1][:, 1]))
            limit = float(np.finfo(np.float32).max) / 2
            if (
                tree is not None
                and np.max(np.abs(lower)) <= limit
                and np.max(np.abs(upper)) <= limit
            ):
                target_tree = prepare_bvh(lower, upper, dtype=np.float32)
                return {
                    rows[int(index)]
                    for source_block, _ in bvh_overlap_pair_blocks(
                        tree, target_tree, include_touching=True
                    )
                    for index in source_block
                }
            # Coordinates outside float32 remain filtered by the exact owner's
            # outward binary64 boxes, without narrowing or an unbounded pair bank.
            keep = np.ones(len(rows), dtype=bool)
            for source_boxes, target_boxes in zip(boxes, valid_targets, strict=True):
                keep &= np.all(
                    (source_boxes[0] <= target_boxes[1])
                    & (target_boxes[0] <= source_boxes[1]),
                    axis=1,
                )
            return {rows[int(index)] for index in np.flatnonzero(keep)}
        candidates: set[int] | None = None
        for source_boxes, target_boxes in zip(boxes, valid_targets, strict=True):
            pairs = box_overlap_pairs(
                len(rows), 1, source_boxes, target_boxes, self_pairs=False
            )
            found = {rows[int(index)] for index in pairs[:, 0]}
            candidates = found if candidates is None else candidates & found
        return candidates
    except (ArithmeticError, IndexError, TypeError, ValueError):
        # The exact owner retains unsupported inputs and ordered refusals.
        return None


def replay_source_edge_face_overlays(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    tokens: tuple[ValidatedSourceFidelityWitness, ...],
    edge_tables: tuple[GeometryAssociation, ...],
    original_edge_rows: tuple[int, ...],
    root: LayerCoreRootGeometry | None,
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> tuple[GeometryAssociation, ...]:
    """Cover source-edge portions not carried by target edges with exact face segments.

    Each input token must already bind the actual source query and target geometry.
    Mapped sources use independently reproved original-reference charts; a curved
    image is never replaced by a physical chord approximation.
    """
    from ._layer_core_boundary_replay import (
        _core_facet_sides,
        _require_layer_fidelity_token,
        core_facet_source_key,
    )
    from ._layer_core_strata_replay import _triangle_line_interval

    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    if tuple(sorted(set(original_edge_rows))) != original_edge_rows or any(
        row < 0 or row >= len(complex_.plc_edge_ids) for row in original_edge_rows
    ):
        raise ValueError("Original source-edge inventory is malformed.")
    identities = {
        row: _core_source_identity(
            authored,
            1,
            str(complex_.plc_edge_ids[row]),
            tuple(complex_.plc_edge_occurrence_paths[row]),
        )
        for row in original_edge_rows
    }
    row_by_identity = {identity: row for row, identity in identities.items()}
    if len(row_by_identity) != len(identities):
        raise ValueError(
            "Original source edges need unique occurrence-qualified identities."
        )
    covered: dict[int, list[_Interval]] = {row: [] for row in original_edge_rows}
    edge_set = mesh.entity_set(1).entity_set_id
    for table in edge_tables:
        if table.target_entity_set_id != edge_set:
            continue
        for identity, path, role, interval in zip(
            table.source_entity_ids,
            table.source_occurrence_paths,
            table.source_roles,
            table.source_parameter_intervals,
            strict=True,
        ):
            if role is not GeometrySourceEntityRole.EDGE:
                continue
            source_row = row_by_identity.get((identity, path))
            if source_row is None or interval is None:
                raise ValueError(
                    "A source-edge carrier lost its exact original interval."
                )
            covered[source_row].append(interval)
    for edge, intervals in covered.items():
        ordered = sorted(intervals)
        if any(not (0 <= lower < upper <= 1) for lower, upper in ordered):
            raise ValueError(
                f"Original source edge {edge} has out-of-range source support."
            )
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(
                f"Original source edge {edge} has overlapping target-edge support."
            )
    missing = {
        row: _subtract((Fraction(0), Fraction(1)), values)
        for row, values in covered.items()
    }
    if not any(missing.values()):
        return ()
    reference_maps = None
    mapped_edges = None
    if source.mapped_domain is not None:
        from ._layer_core_mapped_segment_overlay import (
            mapped_local_reference_maps,
            mapped_local_source_edges,
        )

        points: tuple[_Point, ...] = ()
        reference_maps = mapped_local_reference_maps(
            source, mesh, geometry, root, computed_root=computed_root
        )
        mapped_edges = mapped_local_source_edges(source)
    else:
        if complex_.geometry is not None:
            raise ValueError(
                "Curved source edges require an exact original-reference chart."
            )
        points = tuple(
            tuple(Fraction(float(value)) for value in point)
            for point in np.asarray(complex_.vertices, dtype=np.float64)
        )
    facet_edges: dict[tuple[str, tuple[str, ...]], dict[int, _Sides]] = {}
    sides_by_facet = _core_facet_sides(source)
    for edge in original_edge_rows:
        for facet in complex_.plc_edge_facets[edge]:
            facet = int(facet)
            if (
                authored.source_boundary_scope is not None
                and authored.core_facet_source_ids[facet] is None
            ):
                positive, negative = (
                    int(value) for value in complex_.facet_regions[facet]
                )
                if positive < 0 or positive != negative:
                    continue
            key = core_facet_source_key(authored, facet)
            previous = facet_edges.setdefault(key, {}).setdefault(
                edge, sides_by_facet[facet]
            )
            if previous != sides_by_facet[facet]:
                raise ValueError("A source facet has conflicting material-side identity.")
    maps = None
    if reference_maps is None:
        resolved = geometry.resolve(mesh)
        maps = _geometry_map_expressions(mesh, geometry, (resolved[0], resolved[1]))
    mapped_face_maps = (
        None
        if reference_maps is None
        else {cell: value[1] for cell, value in reference_maps.items()}
    )
    candidates: dict[int, list[_Candidate]] = {row: [] for row in original_edge_rows}
    physical_edges: dict[int, tuple[_Point, _Point]] = {}
    prepared_boxes: dict[
        tuple[str, tuple[str, ...]],
        tuple[tuple[int, ...], tuple[PolygonBoxes, PolygonBoxes], PackedBVH | None]
        | None,
    ] = {}
    for token in tokens:
        witness = token.witness
        if not isinstance(
            witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        ):
            continue
        if (
            isinstance(witness, SourceBoundaryChartWitness)
            and witness.chart_dimension != 2
        ):
            continue
        if not token.binds(mesh, geometry, token.source):
            raise ValueError(
                "Source-edge overlay token is stale for the target geometry."
            )
        authenticated = False
        for index, (cell, face, local, identity, path) in enumerate(
            zip(
                witness.cell_global_ids,
                witness.entity_global_ids,
                witness.local_entity_indices,
                witness.source_entity_ids,
                witness.occurrence_paths,
                strict=True,
            )
        ):
            relevant = facet_edges.get((identity, tuple(path)), {})
            if not relevant or not any(missing[row] for row in relevant):
                continue
            if not authenticated:
                _require_layer_fidelity_token(mesh, geometry, token, source)
                authenticated = True
            image_witness = isinstance(witness, SourceBoundaryFaceImageWitness)
            if image_witness:
                assert isinstance(witness, SourceBoundaryFaceImageWitness)
                physical = witness.image_triangle(index)
                reference = ()
            else:
                assert isinstance(witness, SourceBoundaryChartWitness)
                reference = witness.target_reference_chart(index)
                if reference_maps is None:
                    assert maps is not None
                    physical = _affine_witness_triangle(int(cell), reference, maps)
                else:
                    from ._layer_core_mapped_segment_overlay import (
                        affine_mapped_reference_triangle,
                    )

                    assert mapped_face_maps is not None
                    physical = affine_mapped_reference_triangle(
                        mapped_face_maps, int(cell), reference
                    )
            face_candidates: set[int] | None = None
            triangle_checked = False
            candidates_prepared = False
            for edge, sides in relevant.items():
                if not missing[edge]:
                    continue
                if image_witness:
                    assert isinstance(witness, SourceBoundaryFaceImageWitness)
                    if witness.source_region_sides[index] != sides:
                        raise ValueError(
                            "Image edge support changes its source facet material sides."
                        )
                    if edge not in physical_edges:
                        physical_edges[edge] = _physical_source_edge(source, edge)
                    first, last = physical_edges[edge]
                elif reference_maps is None:
                    first_row, last_row = complex_.plc_edge_vertices[edge]
                    first, last = points[first_row], points[last_row]
                else:
                    assert mapped_edges is not None
                    root_id = reference_maps[int(cell)][0]
                    endpoints = mapped_edges.get(root_id, {}).get(edge)
                    if endpoints is None:
                        continue
                    first, last = endpoints
                if reference_maps is None and triangle_checked:
                    if not candidates_prepared:
                        face_candidates = _unmapped_face_edge_candidates(
                            (identity, tuple(path)),
                            relevant,
                            missing,
                            points,
                            complex_.plc_edge_vertices,
                            physical,
                            prepared_boxes,
                        )
                        candidates_prepared = True
                    if face_candidates is not None and edge not in face_candidates:
                        continue
                interval = _triangle_line_interval(first, last, physical)
                triangle_checked = True
                if interval is None or interval[0] >= interval[1]:
                    continue
                for lower, upper in missing[edge]:
                    start, end = max(lower, interval[0]), min(upper, interval[1])
                    if start >= end:
                        continue
                    endpoints = tuple(
                        tuple(
                            a + parameter * (b - a)
                            for a, b in zip(first, last, strict=True)
                        )
                        for parameter in (start, end)
                    )
                    if image_witness:
                        start_ref, end_ref = endpoints
                        chart_id = witness.source_chart_ids[index]
                    else:
                        start_ref, end_ref = tuple(
                            _reference_point(point, physical, reference)
                            for point in endpoints
                        )
                        chart_id = None
                    candidates[edge].append(
                        (
                            start,
                            end,
                            int(face),
                            int(cell),
                            int(local),
                            witness.witness_id,
                            start_ref,
                            end_ref,
                            sides,
                            chart_id,
                        )
                    )
    output = []
    for edge in original_edge_rows:
        selected: list[_Candidate] = []
        occupied = list(covered[edge])
        for candidate in sorted(candidates[edge], key=repr):
            start, end, face, cell, local, witness_id, first, last, sides, chart_id = (
                candidate
            )
            for lower, upper in _subtract((start, end), occupied):
                alpha = (lower - start) / (end - start)
                beta = (upper - start) / (end - start)
                restricted = tuple(
                    tuple(
                        a + fraction * (b - a) for a, b in zip(first, last, strict=True)
                    )
                    for fraction in (alpha, beta)
                )
                selected.append(
                    (
                        lower,
                        upper,
                        face,
                        cell,
                        local,
                        witness_id,
                        restricted[0],
                        restricted[1],
                        sides,
                        chart_id,
                    )
                )
                occupied.append((lower, upper))
        cursor = Fraction(0)
        for lower, upper in sorted(occupied):
            if lower != cursor or upper <= lower:
                raise ValueError(
                    f"Original source edge {edge} has a gap or overlap in its edge/face support."
                )
            cursor = upper
        if cursor != Fraction(1):
            raise ValueError(
                f"Original source edge {edge} has incomplete edge/face support."
            )
        identity, path = identities[edge]
        for (
            lower,
            upper,
            face,
            cell,
            local,
            witness_id,
            first,
            last,
            sides,
            chart_id,
        ) in selected:
            tokens_row = rational_chart_tokens(((first, last),))[0]
            output.append(
                GeometryAssociation(
                    GeometryAssociationKind.PIECEWISE_LINEAR,
                    authored.source_id,
                    authored.source_revision,
                    mesh.entity_set(2).entity_set_id,
                    np.asarray([face], dtype=np.int64),
                    (identity,),
                    np.zeros((1,), dtype=np.float64),
                    exact=True,
                    source_roles=(GeometrySourceEntityRole.EDGE,),
                    source_occurrence_paths=(path,),
                    source_region_sides=(sides,),
                    source_parameter_intervals=((lower, upper),),
                    target_reference_pieces=(
                        (
                            (cell, local, (), tokens_row)
                            if chart_id is None
                            else SourceBoundaryImagePullback(
                                cell, local, tokens_row, chart_id
                            ),
                        ),
                    ),
                    source_boundary_witness_id=witness_id,
                    orientations=np.asarray([1], dtype=np.int8),
                    provenance=GeometryAssociationProvenance.CLASSIFICATION,
                )
            )
    return tuple(output)


def _validate_image_source_edge_segments(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    association: GeometryAssociation,
    /,
) -> None:
    from ._layer_core_boundary_replay import (
        _core_facet_sides,
        _require_layer_image_token,
        core_facet_source_key,
    )
    from ._layer_core_strata_replay import _triangle_line_interval

    witness = _require_layer_image_token(mesh, geometry, token, source)
    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    if (
        association.target_entity_set_id != mesh.entity_set(2).entity_set_id
        or association.source_boundary_witness_id != witness.witness_id
        or association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
        or (association.source_id, association.source_revision)
        != (authored.source_id, authored.source_revision)
    ):
        raise ValueError("Image edge support binds different source or target owners.")
    association.validate_target(mesh.entity_set(2))
    edge_rows = {
        _core_source_identity(authored, 1, str(identifier), tuple(path)): row
        for row, (identifier, path) in enumerate(
            zip(complex_.plc_edge_ids, complex_.plc_edge_occurrence_paths, strict=True)
        )
    }
    if len(edge_rows) != len(complex_.plc_edge_ids):
        raise ValueError("Original source edge identities are ambiguous.")
    facet_sides = _core_facet_sides(source)
    charts: dict[
        tuple[int, int, int, str, str, tuple[str, ...], _Sides],
        list[tuple[_Point, ...]],
    ] = {}
    for index, (face, cell, local, chart, identity, path, sides) in enumerate(
        zip(
            witness.entity_global_ids,
            witness.cell_global_ids,
            witness.local_entity_indices,
            witness.source_chart_ids,
            witness.source_entity_ids,
            witness.occurrence_paths,
            witness.source_region_sides,
            strict=True,
        )
    ):
        charts.setdefault(
            (
                int(face),
                int(cell),
                int(local),
                chart,
                identity,
                tuple(path),
                tuple(sides),
            ),
            [],
        ).append(witness.image_triangle(index))
    orientations = np.asarray(association.orientations, dtype=np.int8)
    for row, face in enumerate(association.target_global_ids):
        if association.source_roles[row] is not GeometrySourceEntityRole.EDGE:
            continue
        edge = edge_rows.get(
            (association.source_entity_ids[row], association.source_occurrence_paths[row])
        )
        pieces = association.target_reference_pieces[row]
        expected_interval = association.source_parameter_intervals[row]
        if edge is None or pieces is None or expected_interval is None:
            raise ValueError(
                "Image edge support lacks its original exact source ancestry."
            )
        source_first, source_last = _physical_source_edge(source, edge)
        direction = tuple(b - a for a, b in zip(source_first, source_last, strict=True))
        axis = next((index for index, value in enumerate(direction) if value), None)
        if axis is None:
            raise ValueError("An original source edge is degenerate.")
        valid_facets: set[tuple[str, tuple[str, ...], _Sides]] = set()
        for facet_raw in complex_.plc_edge_facets[edge]:
            facet = int(facet_raw)
            if (
                authored.source_boundary_scope is not None
                and authored.core_facet_source_ids[facet] is None
            ):
                positive, negative = (
                    int(value) for value in complex_.facet_regions[facet]
                )
                if positive < 0 or positive != negative:
                    continue
            identity, path = core_facet_source_key(authored, facet)
            valid_facets.add((identity, path, facet_sides[facet]))
        intervals: list[_Interval] = []
        for piece in pieces:
            if (
                not isinstance(piece, SourceBoundaryImagePullback)
                or len(piece.image_rational_tokens) != 2
            ):
                raise ValueError(
                    "Image edge support requires an exact two-point image pullback."
                )
            first, last = tuple(
                tuple(Fraction(*coordinate) for coordinate in point)
                for point in piece.image_rational_tokens
            )
            containing = any(
                _triangle_line_interval(first, last, triangle)
                == (Fraction(0), Fraction(1))
                for identity, path, sides in valid_facets
                if sides == association.source_region_sides[row]
                for triangle in charts.get(
                    (
                        int(face),
                        piece.cell_global_id,
                        piece.local_entity_index,
                        piece.source_chart_id,
                        identity,
                        path,
                        sides,
                    ),
                    (),
                )
            )
            if not containing:
                raise ValueError(
                    "Image edge segment lacks its exact incident source-facet image."
                )
            start = (first[axis] - source_first[axis]) / direction[axis]
            end = (last[axis] - source_first[axis]) / direction[axis]
            if (
                start == end
                or not (0 <= start <= 1 and 0 <= end <= 1)
                or (1 if end > start else -1) != int(orientations[row])
                or any(
                    point[component]
                    != source_first[component] + parameter * direction[component]
                    for point, parameter in ((first, start), (last, end))
                    for component in range(3)
                )
            ):
                raise ValueError(
                    "Image edge segment changes its exact source curve or orientation."
                )
            intervals.append((min(start, end), max(start, end)))
        cursor = expected_interval[0]
        for lower, upper in sorted(intervals):
            if lower != cursor or upper <= lower:
                raise ValueError(
                    "Image edge pieces overlap or leave a source parameter gap."
                )
            cursor = upper
        if cursor != expected_interval[1]:
            raise ValueError(
                "Image edge pieces do not cover their declared source interval."
            )


def validate_source_edge_face_segments(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    token: ValidatedSourceFidelityWitness,
    association: GeometryAssociation,
    /,
    *,
    root: LayerCoreRootGeometry | None = None,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> None:
    """Reprove each source-edge segment against its actual source and face chart."""
    if isinstance(token.witness, SourceBoundaryFaceImageWitness):
        _validate_image_source_edge_segments(source, mesh, geometry, token, association)
        return
    from ._layer_core_boundary_replay import (
        _core_facet_sides,
        _require_layer_chart_token,
        core_facet_source_key,
    )
    from ._layer_core_strata_replay import _triangle_line_interval, source_line_interval

    witness = _require_layer_chart_token(mesh, geometry, token, source)
    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    if source.mapped_domain is None and complex_.geometry is not None:
        raise ValueError("Curved source edges require an exact original-reference chart.")
    if (
        association.target_entity_set_id != mesh.entity_set(2).entity_set_id
        or association.source_boundary_witness_id != witness.witness_id
        or association.association_kind is not GeometryAssociationKind.PIECEWISE_LINEAR
        or (association.source_id, association.source_revision)
        != (authored.source_id, authored.source_revision)
    ):
        raise ValueError(
            "Source-edge face segments bind different source or target owners."
        )
    association.validate_target(mesh.entity_set(2))
    edge_rows = {
        _core_source_identity(authored, 1, str(identifier), tuple(path)): row
        for row, (identifier, path) in enumerate(
            zip(complex_.plc_edge_ids, complex_.plc_edge_occurrence_paths, strict=True)
        )
    }
    if len(edge_rows) != len(complex_.plc_edge_ids):
        raise ValueError("Original source edge identities are ambiguous.")
    facet_sides = _core_facet_sides(source)
    reference_maps = None
    mapped_edges = None
    if source.mapped_domain is not None:
        from ._layer_core_mapped_segment_overlay import (
            mapped_local_reference_maps,
            mapped_local_source_edges,
        )

        source_points: tuple[_Point, ...] = ()
        reference_maps = mapped_local_reference_maps(
            source, mesh, geometry, root, computed_root=computed_root
        )
        mapped_edges = mapped_local_source_edges(source)
    else:
        source_points = tuple(
            tuple(Fraction(float(value)) for value in point)
            for point in np.asarray(complex_.vertices, dtype=np.float64)
        )
    source_charts: dict[
        tuple[int, int, int, str, tuple[str, ...]], list[tuple[_Point, ...]]
    ] = {}
    for row, (face, cell, local, identity, path) in enumerate(
        zip(
            witness.entity_global_ids,
            witness.cell_global_ids,
            witness.local_entity_indices,
            witness.source_entity_ids,
            witness.occurrence_paths,
            strict=True,
        )
    ):
        source_charts.setdefault(
            (int(face), int(cell), int(local), identity, tuple(path)), []
        ).append(witness.target_reference_chart(row))
    maps = None
    if reference_maps is None:
        resolved = geometry.resolve(mesh)
        maps = _geometry_map_expressions(mesh, geometry, (resolved[0], resolved[1]))
    orientations = np.asarray(association.orientations, dtype=np.int8)
    for row, face in enumerate(association.target_global_ids):
        if association.source_roles[row] is not GeometrySourceEntityRole.EDGE:
            continue
        edge = edge_rows.get(
            (association.source_entity_ids[row], association.source_occurrence_paths[row])
        )
        pieces = association.target_reference_pieces[row]
        expected_interval = association.source_parameter_intervals[row]
        if edge is None or pieces is None or expected_interval is None:
            raise ValueError("A source-edge face segment lacks exact original ancestry.")
        first_row, last_row = complex_.plc_edge_vertices[edge]
        valid_facets: set[tuple[str, tuple[str, ...], _Sides]] = set()
        for facet_raw in complex_.plc_edge_facets[edge]:
            facet = int(facet_raw)
            if (
                authored.source_boundary_scope is not None
                and authored.core_facet_source_ids[facet] is None
            ):
                positive, negative = (
                    int(value) for value in complex_.facet_regions[facet]
                )
                if positive < 0 or positive != negative:
                    continue
            identity, path = core_facet_source_key(authored, facet)
            valid_facets.add((identity, path, facet_sides[facet]))
        intervals: list[_Interval] = []
        for piece in pieces:
            if not isinstance(piece, tuple) or len(piece) != 4:
                raise ValueError(
                    "Source-edge face support requires an explicit reference segment."
                )
            cell, local, vertices, chart_tokens = piece
            if vertices or len(chart_tokens) != 2:
                raise ValueError(
                    "Source-edge face support requires a standard exact segment chart."
                )
            chart = tuple(
                tuple(Fraction(*value) for value in point) for point in chart_tokens
            )
            containing = any(
                _triangle_line_interval(chart[0], chart[1], triangle)
                == (Fraction(0), Fraction(1))
                for identity, path, sides in valid_facets
                if sides == association.source_region_sides[row]
                for triangle in source_charts.get(
                    (int(face), cell, local, identity, path), ()
                )
            )
            if not containing:
                raise ValueError(
                    "A source-edge segment lacks its exact incident source-facet chart."
                )
            arguments = _chart_from_corners("interval", chart)
            if reference_maps is None:
                assert maps is not None
                expressions = maps.expressions(cell)
                source_first, source_last = (
                    source_points[first_row],
                    source_points[last_row],
                )
            else:
                assert mapped_edges is not None
                root_id, expressions = reference_maps[cell]
                endpoints = mapped_edges.get(root_id, {}).get(edge)
                if endpoints is None:
                    raise ValueError(
                        "Source-edge segment lacks its declared original root incidence."
                    )
                source_first, source_last = endpoints
            physical = tuple(
                compose_expression(value, arguments) for value in expressions
            )
            interval = source_line_interval(physical, source_first, source_last)
            if interval is None or (1 if interval[1] > interval[0] else -1) != int(
                orientations[row]
            ):
                raise ValueError(
                    "A source-edge segment changes its exact source curve or orientation."
                )
            intervals.append((min(interval), max(interval)))
        cursor = expected_interval[0]
        for lower, upper in sorted(intervals):
            if lower != cursor or upper <= lower:
                raise ValueError(
                    "Source-edge face pieces overlap or leave a parameter gap."
                )
            cursor = upper
        if cursor != expected_interval[1]:
            raise ValueError(
                "Source-edge face pieces do not cover their declared source interval."
            )


__all__ = ["replay_source_edge_face_overlays", "validate_source_edge_face_segments"]
