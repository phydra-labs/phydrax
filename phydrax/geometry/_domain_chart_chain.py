#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Oriented chart-chain validation and interval-bounded trim homotopies.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "source_trim_arc_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The input boundary and each refined proof cover have at most 65,536 arcs before source-arc storage is admitted."
      }
    ],
    "conclusion": "A larger boundary or proof cover leaves the source-trim certification unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "MeshingDomain trim arc allowance",
      "locator": "phydrax.geometry._domain_chart_chain._prepare_chart_chain and _bound_trim_loop",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "source_trim_pair_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The chart-chain, per-loop topology, and cross-loop separation searches share a remaining candidate-pair allowance bounded by min(maximum_pairs, 200,000)."
      }
    ],
    "conclusion": "Pair exhaustion leaves the source-trim certification unresolved, retaining consumed pair work in resource_counts.",
    "citation": {
      "kind": "definition",
      "reference": "MeshingDomain trim pair allowance",
      "locator": "phydrax.geometry._domain_chart_chain.verify_chart_chain",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "source_chart_restriction_authority"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Restriction vertices and edges index the chart topology, and each rational affine root is bound to the same source revision, patch, and parent edge."
      }
    ],
    "conclusion": "Invalid or missing restriction definitions leave chart orientation and source coverage unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Source-bound rational chart restrictions",
      "locator": "phydrax.geometry._chart_restriction.validate_chart_restrictions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "source_trim_topology"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A CurveTrimLoop topology certificate resolves every refined source/chord arc relation while subdivisions retain their unchanged source chord endpoints."
      },
      {
        "tag": "X",
        "statement": "Refinement remains within depth 32 and the per-loop arc and candidate-pair allowances."
      }
    ],
    "conclusion": "The absence of a certified proof cover leaves the source/chord trim relation unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Bounded source/chord trim proof admission",
      "locator": "phydrax.geometry._domain_chart_chain._certified_trim_cover and _bound_trim_loop",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  }
]
End mesh certificate checks
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, final

import equinox as eqx
import numpy as np

import phydrax.geometry._domain_numerics as domain_numerics_owner
import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._meshing_domain as domain_owner
import phydrax.geometry.brep._constructors as constructors_owner
import phydrax.geometry.brep._intersection_curve as intersection_curve_owner

from .._meshcore import exact_orient2d
from .._strict import StrictModule
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ._atlas import (
    AbstractTrimCurve,
    CurveTrimLoop,
    overlapping_box_pairs,
    PolygonTrimLoop,
    TrimDomain,
    TrimLoop,
)
from ._chart_restriction import validate_chart_restrictions
from .brep._intersection_curve import (
    CurveTrimSegment,
    original_trim_intersection_preparation,
)
from .brep._patches import (
    AbstractCurve,
    BernsteinRestrictionBank,
    LineCurve,
    MeridianOverlaps,
)
from .brep._request_preparations import TrimRequestPreparations


def _polygon_sign(points: np.ndarray, /) -> int:
    area = Fraction(0)
    for first, second in zip(points, np.roll(points, -1, axis=0), strict=True):
        area += Fraction(float(first[0])) * Fraction(float(second[1]))
        area -= Fraction(float(first[1])) * Fraction(float(second[0]))
    return (area > 0) - (area < 0)


def _winding(point: np.ndarray, polygon: np.ndarray, /) -> int:
    following = np.roll(polygon, -1, axis=0)
    signs = exact_orient2d(polygon, following, np.broadcast_to(point, polygon.shape))
    up = (polygon[:, 1] <= point[1]) & (following[:, 1] > point[1]) & (signs > 0)
    down = (polygon[:, 1] > point[1]) & (following[:, 1] <= point[1]) & (signs < 0)
    return int(np.sum(up) - np.sum(down))


def _simple_boundary(
    charts: np.ndarray, boundary: np.ndarray, maximum_pairs: int, /
) -> tuple[bool, int, bool]:
    """simplicity of a chart boundary; returns validity, pair work, exhaustion."""
    boxes = charts[boundary]
    candidates, work = overlapping_box_pairs(
        np.min(boxes, axis=1), np.max(boxes, axis=1), maximum_pairs
    )
    if candidates is None:
        return False, work, True
    first, second = boundary[candidates[:, 0]].T
    others = boundary[candidates[:, 1]]
    selected = np.all((others != first[:, None]) & (others != second[:, None]), axis=1)
    if not np.any(selected):
        return True, work, False
    aa, bb = charts[first[selected]], charts[second[selected]]
    c, d = charts[others[selected, 0]], charts[others[selected, 1]]
    s0, s1 = exact_orient2d(aa, bb, c), exact_orient2d(aa, bb, d)
    s2, s3 = exact_orient2d(c, d, aa), exact_orient2d(c, d, bb)
    contact = (s0 * s1 < 0) & (s2 * s3 < 0)
    for sign, point, start, end in (
        (s0, c, aa, bb),
        (s1, d, aa, bb),
        (s2, aa, c, d),
        (s3, bb, c, d),
    ):
        contact |= (sign == 0) & np.all(
            (point >= np.minimum(start, end)) & (point <= np.maximum(start, end)),
            axis=1,
        )
    return not bool(np.any(contact)), work, False


def trim_curve_interval_bounds(
    domain: domain_owner.MeshingDomain,
    patch: int,
    curves: tuple[AbstractTrimCurve, ...],
    firsts: np.ndarray,
    lasts: np.ndarray,
    chart_edges: np.ndarray,
    point_edges: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
    restriction_bank: BernsteinRestrictionBank | None = None,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> np.ndarray:
    """Bound each source-curve arc against only its own chord segment."""
    count = len(curves)
    if (
        firsts.shape != (count,)
        or lasts.shape != (count,)
        or chart_edges.shape != (count, 2, 2)
        or point_edges.shape != (count, 2, 3)
    ):
        raise ValueError(
            "MeshingDomain.trim_ribbon_arrays must align source intervals with chart and physical edges; got inconsistent array shapes"
        )
    source_boxes = np.asarray(
        [
            curve.enclosure(
                float(first), float(last), budget=budget, preparations=preparations
            )
            for curve, first, last in zip(curves, firsts, lasts, strict=True)
        ],
        dtype=np.float64,
    )
    boxes = np.stack(
        (
            np.minimum(source_boxes[:, 0], np.min(chart_edges, axis=1)),
            np.maximum(source_boxes[:, 1], np.max(chart_edges, axis=1)),
        ),
        axis=1,
    )
    surface = domain.patches[patch].surface
    first_jet = np.zeros((count,), dtype=np.bool_)
    if domain_numerics_owner.continuous_spline_carrier(surface, budget=budget):
        first_jet = np.asarray(
            [
                not surface.is_c1_on(box, budget=budget, preparations=preparations)
                for box in boxes
            ],
            dtype=np.bool_,
        )
    second_lower = np.zeros((count, 3, 2, 2), dtype=np.float64)
    second_upper = np.zeros((count, 3, 2, 2), dtype=np.float64)
    second_rows = np.flatnonzero(~first_jet)
    if second_rows.size:
        lower, upper = surface.derivative_bounds_batch(
            boxes[second_rows],
            order=2,
            budget=budget,
            preparations=preparations,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        second_lower[second_rows], second_upper[second_rows] = lower, upper
    first_lower, first_upper = surface.derivative_bounds_batch(
        boxes,
        order=1,
        budget=budget,
        preparations=preparations,
        restriction_bank=restriction_bank,
        meridian_jets=meridian_jets,
    )
    mapped_edges = domain.evaluate(
        np.full((2 * count,), patch), chart_edges.reshape((-1, 2)), budget=budget
    ).reshape((-1, 2, 3))
    result = []
    for arc, (curve, first_, last_) in enumerate(zip(curves, firsts, lasts, strict=True)):
        first, last = float(first_), float(last_)
        nodal = float(
            np.max(np.linalg.norm(mapped_edges[arc] - point_edges[arc], axis=1))
        )
        jl, ju = first_lower[arc], first_upper[arc]
        hessian = np.linalg.norm(
            np.maximum(np.abs(second_lower[arc]), np.abs(second_upper[arc])),
            axis=0,
        )
        width = boxes[arc, 1] - boxes[arc, 0]
        endpoint_error = np.maximum(
            np.max(
                np.abs(
                    np.asarray(
                        curve.enclosure(
                            first, first, budget=budget, preparations=preparations
                        )
                    )
                    - chart_edges[arc, 0]
                ),
                axis=0,
            ),
            np.max(
                np.abs(
                    np.asarray(
                        curve.enclosure(
                            last, last, budget=budget, preparations=preparations
                        )
                    )
                    - chart_edges[arc, 1]
                ),
                axis=0,
            ),
        )
        chart_error = domain_numerics_owner.curve_chord_error(
            curve,
            first,
            last,
            budget=budget,
            preparations=preparations,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        ) + np.nextafter(endpoint_error, np.inf)
        jacobian = np.linalg.norm(np.maximum(np.abs(jl), np.abs(ju)), axis=0)
        if first_jet[arc]:
            carrier_error = domain_numerics_owner.first_jet_chord_remainder(
                jl, ju, width, budget=budget
            )
        else:
            carrier_error = float(width @ hessian @ width / 8)
        bound = carrier_error + float(jacobian @ chart_error) + nodal
        result.append(
            np.nextafter(
                bound * (1 + 256 * np.finfo(np.float64).eps)
                + 256 * np.finfo(np.float64).eps * domain.scale,
                np.inf,
            )
        )
    return np.asarray(result, dtype=np.float64)


def _trim_ribbon_bounds(
    domain: domain_owner.MeshingDomain,
    patch: int,
    cover: CurveTrimLoop,
    charts: np.ndarray,
    points: np.ndarray,
    edges: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
    restriction_bank: BernsteinRestrictionBank | None = None,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> np.ndarray:
    """Bound the whole source/chord homotopy per source-curve arc."""
    count = edges.shape[0]
    return trim_curve_interval_bounds(
        domain,
        patch,
        tuple(cover.curves[int(curve)] for curve in cover.arc_curves[:count]),
        np.asarray(cover.arc_first[:count]),
        np.asarray(cover.arc_last[:count]),
        charts[edges],
        points[edges],
        budget=budget,
        preparations=preparations,
        restriction_bank=restriction_bank,
        meridian_jets=meridian_jets,
    )


def boundary_cell_ribbon_bounds(
    cells: np.ndarray,
    boundary: np.ndarray,
    ribbon_bounds: np.ndarray,
    /,
) -> np.ndarray:
    """Assign each local boundary ribbon only to its incident source cell."""
    if ribbon_bounds.shape != (boundary.shape[0],):
        raise ValueError(
            f"MeshingDomain.boundary_ribbon_bounds must align with source edges; got {ribbon_bounds.shape} bounds for {boundary.shape[0]} edges"
        )
    by_edge = {
        tuple(sorted((int(edge[0]), int(edge[1])))): float(bound)
        for edge, bound in zip(boundary, ribbon_bounds, strict=True)
    }
    result = np.zeros((cells.shape[0],), dtype=np.float64)
    for row, cell in enumerate(cells):
        result[row] = max(
            (
                by_edge.get(
                    tuple(sorted((int(cell[first]), int(cell[second])))),
                    0.0,
                )
                for first, second in ((0, 1), (1, 2), (2, 0))
            ),
            default=0.0,
        )
    return result


def _disjoint_trim_covers(
    covers: list[tuple[np.ndarray, np.ndarray]], maximum_pairs: int, /
) -> tuple[bool, int, bool]:
    """Bounded cross-loop homotopy separation, preserving every source hole."""
    if len(covers) < 2:
        return True, 0, False
    lower = np.concatenate([cover[0] for cover in covers])
    upper = np.concatenate([cover[1] for cover in covers])
    owners = np.concatenate(
        [np.full((cover[0].shape[0],), index) for index, cover in enumerate(covers)]
    )
    pairs, work = overlapping_box_pairs(lower, upper, maximum_pairs)
    if pairs is None:
        return False, work, True
    return not bool(np.any(owners[pairs[:, 0]] != owners[pairs[:, 1]])), work, False


@final
class _TrimCoverProof(StrictModule):
    """source chord cover, topology proof cover, and consumed subdivision allowance."""

    cover: CurveTrimLoop
    proof: CurveTrimLoop | None
    pairs: int = eqx.field(static=True)
    arcs: int = eqx.field(static=True)
    exhausted: bool = eqx.field(static=True)


def _trim_chord_point(
    rows: list[tuple[int, float, float]],
    polygon: np.ndarray,
    arc: int,
    parameter: float,
) -> np.ndarray:
    """Evaluate a proof endpoint on its unchanged affine source chord."""
    _, start, end = rows[arc]
    if parameter == start:
        return polygon[arc]
    return polygon[arc] + (parameter - start) / (end - start) * (
        polygon[(arc + 1) % polygon.shape[0]] - polygon[arc]
    )


def _certified_trim_cover(
    trim_curves: list[AbstractTrimCurve],
    partitions: list[np.ndarray],
    polygon: np.ndarray,
    relative_closure_tolerance: float,
    maximum_pairs: int,
    reserve_queries: Callable[[int], None] | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
    restriction_bank: BernsteinRestrictionBank | None = None,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> _TrimCoverProof:
    """Prove source/chord isotopy of the source chord chain.

    Returns the source cover, its topology proof cover, the pair work, the
    largest proof arc count, and pair exhaustion. Only arcs named by an
    unresolved topology finding are bisected. Inserted chord vertices lie on
    their source chord segment at the affine image of the source parameter,
    so the proof binds exactly the source chord polygon; subdivision only
    tightens arc and homotopy boxes. Every subdivided cover is charged before
    construction, whether or not it certifies: per arc two enclosures, one
    endpoint evaluation, and one derivative bound.
    """
    count = polygon.shape[0]
    # source arc rows: (curve, parameter start, parameter end), loop order.
    rows = [
        (curve, float(first), float(last))
        for curve, values in enumerate(partitions)
        for first, last in zip(values[:-1], values[1:], strict=True)
    ]
    if len(rows) != count:
        raise ValueError(
            f"MeshingDomain.trim_partitions must align with the chord chain; got {len(rows)} arcs for {count} chords"
        )

    # Proof arcs: (source arc, first, last, bisection depth), loop order.
    arcs = [(arc, start, end, 0) for arc, (_, start, end) in enumerate(rows)]
    original: CurveTrimLoop | None = None
    used = 0
    while True:
        if original is not None and reserve_queries is not None:
            reserve_queries(4 * len(arcs))

        parameters = [
            np.asarray(
                [first for arc, first, _, _ in arcs if rows[arc][0] == curve]
                + [float(values[-1])]
            )
            for curve, values in enumerate(partitions)
        ]
        boxes = [
            trim_curves[rows[arc][0]].enclosure(
                first, last, budget=budget, preparations=preparations
            )
            for arc, first, last, _ in arcs
        ]
        tolerance = float(
            np.nextafter(max(np.linalg.norm(box[1] - box[0]) for box in boxes), np.inf)
        )
        cover = CurveTrimLoop(
            trim_curves,
            tolerance=max(tolerance, np.finfo(np.float64).tiny),
            maximum_arcs=65_536,
            arc_parameters=parameters,
            chord_vertices=np.asarray(
                [
                    _trim_chord_point(rows, polygon, arc, first)
                    for arc, first, _, _ in arcs
                ]
            ),
            relative_closure_tolerance=relative_closure_tolerance,
            budget=budget,
            preparations=preparations,
        )
        if original is None:
            original = cover
        if maximum_pairs - used < 1:
            return _TrimCoverProof(original, None, used, len(arcs), True)
        topology = cover.certify_topology(
            maximum_pairs=maximum_pairs - used, budget=budget, preparations=preparations
        )
        used += topology.pairs_checked
        if topology.certified:
            return _TrimCoverProof(original, cover, used, len(arcs), False)
        if topology.budget_exhausted:
            return _TrimCoverProof(original, None, used, len(arcs), True)
        split = _trim_proof_splits(cover.chords, topology.unresolved_pairs)
        if any(arcs[arc][3] >= 32 for arc in split) or len(arcs) + len(split) > 65_536:
            return _TrimCoverProof(original, None, used, len(arcs) + len(split), False)
        refined: list[tuple[int, float, float, int]] = []
        for index, (arc, first, last, depth) in enumerate(arcs):
            if index in split:
                middle = 0.5 * (first + last)
                refined.extend(
                    ((arc, first, middle, depth + 1), (arc, middle, last, depth + 1))
                )
            else:
                refined.append((arc, first, last, depth))
        arcs = refined


def _trim_proof_splits(
    chords: np.ndarray, unresolved: tuple[tuple[int, int], ...], /
) -> set[int]:
    """Proof arcs to bisect for each unresolved topology finding.

    A self-monotonicity or nonadjacent-overlap finding needs every named arc
    tighter. An adjacent pair across a junction is decided by its two chord
    lengths together; bisecting both keeps their ratio and can never resolve
    a sharp corner, so only the longer chord is bisected.
    """
    count = chords.shape[0]
    lengths = np.linalg.norm(np.roll(chords, -1, axis=0) - chords, axis=1)
    split: set[int] = set()
    for first, second in unresolved:
        if first != second and (second - first) % count in (1, count - 1):
            split.add(first if lengths[first] >= lengths[second] else second)
        else:
            split.update((first, second))
    return split


@final
class _ChartChain(StrictModule):
    """Arrays of one source-bound oriented chart triangulation."""

    charts: np.ndarray
    points: np.ndarray
    cells: np.ndarray
    boundary: np.ndarray
    provenance: np.ndarray
    restriction_required: np.ndarray
    restriction_vertices: np.ndarray
    restriction_edges: np.ndarray
    restriction_parameters: np.ndarray


@final
class _PreparedChain(StrictModule):
    """Restriction identities and remaining pair allowance after chain validation."""

    restriction_ids: tuple[str, ...] = eqx.field(static=True)
    remaining_pairs: int = eqx.field(static=True)


@final
class _PreparedTrimLoop(StrictModule):
    """Source arcs partitioning one oriented boundary polygon."""

    polygon: np.ndarray
    edges: np.ndarray
    curves: list[AbstractTrimCurve]
    partitions: list[np.ndarray]
    bounded: bool = eqx.field(static=True)


@final
class _TrimLoopRibbon(StrictModule):
    """Physical ribbons and separated source-arc boxes of one trim loop."""

    images: np.ndarray
    bounds: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    remaining_pairs: int = eqx.field(static=True)


def _prepare_chart_chain(
    chain: _ChartChain,
    source_revision: str,
    patch: int,
    findings: list[embedding_state_owner.MeshCertificateFinding],
    resource_counts: list[tuple[str, int, int]],
    maximum_pairs: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PreparedChain | None:
    """Validate shapes, source restrictions, orientation, and boundary cancellation."""
    charts, points, cells, boundary, provenance = (
        chain.charts,
        chain.points,
        chain.cells,
        chain.boundary,
        chain.provenance,
    )
    (
        restriction_required,
        restriction_vertices,
        restriction_edges,
        restriction_parameters,
    ) = (
        chain.restriction_required,
        chain.restriction_vertices,
        chain.restriction_edges,
        chain.restriction_parameters,
    )
    if (
        charts.ndim != 2
        or charts.shape[1] != 2
        or points.shape != (charts.shape[0], 3)
        or cells.ndim != 2
        or cells.shape[1] != 3
        or boundary.ndim != 2
        or boundary.shape[1] != 2
        or provenance.shape != (boundary.shape[0], 4)
        or not np.all(np.isfinite(charts))
        or not np.all(np.isfinite(points))
        or not cells.size
        or np.any(cells < 0)
        or np.any(cells >= charts.shape[0])
        or np.any(boundary < 0)
        or np.any(boundary >= charts.shape[0])
    ):
        return None
    resource_counts.append((f"source_trim_arcs:{patch}", boundary.shape[0], 65_536))
    if boundary.shape[0] > 65_536:
        findings.append(
            embedding_state_owner.MeshCertificateFinding(
                "source_trim_arc_capacity", "unresolved", "source_facet", (patch,)
            )
        )
        return None
    if (
        restriction_edges.shape != (restriction_vertices.size, 2)
        or np.any(restriction_edges < 0)
        or np.any(restriction_edges >= charts.shape[0])
    ):
        findings.append(
            embedding_state_owner.MeshCertificateFinding(
                "source_chart_restriction_authority",
                "unresolved",
                "source_facet",
                (patch,),
            )
        )
        return None
    try:
        with budget.measurement():
            exact, restrictions, _ = validate_chart_restrictions(
                charts,
                restriction_required,
                restriction_vertices,
                restriction_edges,
                charts[restriction_edges]
                if restriction_vertices.size
                else np.empty((0, 2, 2), dtype=np.float64),
                restriction_parameters,
                source_revision,
                patch,
            )
        resource_counts.append(
            (
                f"source_chart_restrictions:{patch}",
                len(restrictions),
                charts.shape[0],
            )
        )
    except (TypeError, ValueError):
        findings.append(
            embedding_state_owner.MeshCertificateFinding(
                "source_chart_restriction_authority",
                "unresolved",
                "source_facet",
                (patch,),
            )
        )
        return None
    signs = exact_orient2d(charts[cells[:, 0]], charts[cells[:, 1]], charts[cells[:, 2]])
    for row in np.flatnonzero(np.any(restriction_required[cells], axis=1)):
        first, second, third = cells[row]
        ax, ay = exact[first]
        bx, by = exact[second]
        cx, cy = exact[third]
        determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        signs[row] = (determinant > 0) - (determinant < 0)
    if np.any(signs <= 0):
        return None
    chain: dict[tuple[int, int], int] = {}
    for first, second in np.concatenate(
        (cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]])
    ):
        key = (int(first), int(second))
        reverse = (key[1], key[0])
        if chain.get(reverse, 0):
            chain[reverse] -= 1
        else:
            chain[key] = chain.get(key, 0) + 1
    remaining = {edge: count for edge, count in chain.items() if count}
    expected = {tuple(map(int, edge)): 1 for edge in boundary}
    simple, pair_work, exhausted = _simple_boundary(charts, boundary, maximum_pairs)
    resource_counts.append((f"source_trim_chain_pairs:{patch}", pair_work, maximum_pairs))
    if exhausted:
        findings.append(
            embedding_state_owner.MeshCertificateFinding(
                "source_trim_pair_capacity", "unresolved", "source_facet", (patch,)
            )
        )
    if len(expected) != boundary.shape[0] or remaining != expected or not simple:
        return None
    return _PreparedChain(
        tuple(r.restriction_id for r in restrictions), maximum_pairs - pair_work
    )


def _prepare_trim_loop(
    domain: domain_owner.MeshingDomain,
    patch: int,
    loop_index: int,
    chain: _ChartChain,
) -> _PreparedTrimLoop | None:
    """Validate source-use intervals and prepare their normalized trim partitions."""
    charts, boundary, provenance = chain.charts, chain.boundary, chain.provenance
    patch_source = domain.patches[patch]
    loop = patch_source.loops[loop_index]
    selected = provenance[:, 0] == loop_index
    loop_edges = boundary[selected]
    if not loop_edges.size:
        return None
    polygon = charts[loop_edges[:, 0]]
    if not np.array_equal(loop_edges[:, 1], np.roll(loop_edges[:, 0], -1)):
        return None
    if _polygon_sign(polygon) != (1 if loop_index == 0 else -1):
        return None
    trim_curves: list[AbstractTrimCurve] = []
    partitions: list[np.ndarray] = []
    bounded_loop = False
    for use_index, use in enumerate(loop):
        mask = selected & (provenance[:, 1] == use_index)
        intervals = provenance[mask, 2:]
        edges = boundary[mask]
        if not intervals.size:
            return None
        first, last = (
            (use.first, use.last)
            if isinstance(use, domain_owner.PatchCurveUse)
            else (0.0, 1.0)
        )
        if intervals[0, 0] != first or intervals[-1, 1] != last:
            return None
        if not np.array_equal(intervals[:-1, 1], intervals[1:, 0]) or np.any(
            (intervals[:, 1] - intervals[:, 0]) * (last - first) <= 0
        ):
            return None
        bounded_ribbon = isinstance(use, domain_owner.PatchCurveUse) and (
            not isinstance(use.pcurve, LineCurve)
            or use.first_root is not None
            or use.last_root is not None
        )
        if not bounded_ribbon:
            source_ends = use.charts(intervals.reshape(-1)).reshape((-1, 2, 2))
            scale = max(1.0, float(np.max(np.abs(source_ends))))
            roundoff = 128 * np.finfo(np.float64).eps * scale
            gap = float(np.max(np.abs(source_ends - charts[edges])))
            if gap > domain.tolerance * scale:
                return None
            bounded_ribbon = gap > roundoff
        bounded_loop |= bounded_ribbon
        match use:
            case domain_owner.PatchCurveUse():
                curve = domain_owner.domain_trim_curve(patch_source, use)
            case domain_owner.PatchPoleUse() if use.trim_curve is not None:
                curve = use.trim_curve
            case _:
                curve = CurveTrimSegment(
                    LineCurve(
                        np.asarray(use.start), np.asarray(use.end) - np.asarray(use.start)
                    ),
                    0.0,
                    1.0,
                )
        trim_curves.append(curve)
        partition = (np.concatenate((intervals[:, 0], intervals[-1:, 1])) - first) / (
            last - first
        )
        partition[0], partition[-1] = 0.0, 1.0
        partitions.append(partition)
        if np.any(np.diff(partition) <= 0):
            return None
    return _PreparedTrimLoop(
        polygon, loop_edges, trim_curves, partitions, bool(bounded_loop)
    )


def _bound_trim_loop(
    domain: domain_owner.MeshingDomain,
    patch: int,
    loop_index: int,
    chain: _ChartChain,
    loop: _PreparedTrimLoop,
    remaining_pairs: int,
    findings: list[embedding_state_owner.MeshCertificateFinding],
    resource_counts: list[tuple[str, int, int]],
    maximum_pairs: int,
    reserve_queries: Callable[[int], None] | None,
    *,
    topology_only: bool,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
    restriction_bank: BernsteinRestrictionBank | None,
    meridian_jets: list[MeridianOverlaps] | None,
) -> _TrimLoopRibbon | None:
    """Certify source/chord isotopy before bounding its physical homotopy ribbon."""
    charts, points = chain.charts, chain.points
    polygon, loop_edges = loop.polygon, loop.edges
    trim_curves, partitions = loop.curves, loop.partitions
    if not loop.bounded:
        line_images = charts[loop_edges]
        return _TrimLoopRibbon(
            points[loop_edges[:, [0, 1, 1]]],
            np.zeros((loop_edges.shape[0],), dtype=np.float64),
            np.min(line_images, axis=1),
            np.max(line_images, axis=1),
            remaining_pairs,
        )
    if remaining_pairs < 1:
        findings.append(
            embedding_state_owner.MeshCertificateFinding(
                "source_trim_pair_capacity",
                "unresolved",
                "source_facet",
                (patch,),
            )
        )
        return None
    with original_trim_intersection_preparation(
        tuple(trim_curves), preparations=preparations
    ) as preparations:
        certified = _certified_trim_cover(
            trim_curves,
            partitions,
            polygon,
            domain.tolerance,
            remaining_pairs,
            reserve_queries,
            budget=budget,
            preparations=preparations,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        cover, proof = certified.cover, certified.proof
        pairs, arcs, exhausted = certified.pairs, certified.arcs, certified.exhausted
        remaining_pairs -= pairs
        resource_counts.append(
            (
                f"source_trim_loop_pairs:{patch}:{loop_index}",
                pairs,
                maximum_pairs,
            )
        )
        resource_counts.append(
            (
                f"source_trim_proof_arcs:{patch}:{loop_index}",
                arcs,
                65_536,
            )
        )
        if exhausted:
            findings.append(
                embedding_state_owner.MeshCertificateFinding(
                    "source_trim_pair_capacity",
                    "unresolved",
                    "source_facet",
                    (patch,),
                )
            )
        if arcs > 65_536:
            findings.append(
                embedding_state_owner.MeshCertificateFinding(
                    "source_trim_arc_capacity",
                    "unresolved",
                    "source_facet",
                    (patch,),
                )
            )
        if proof is None:
            findings.append(
                embedding_state_owner.MeshCertificateFinding(
                    "source_trim_topology",
                    "unresolved",
                    "source_facet",
                    (patch,),
                )
            )
            return None
        if topology_only:
            bound = np.zeros((loop_edges.shape[0],), dtype=np.float64)

        else:
            if reserve_queries is not None:
                reserve_queries(2 * loop_edges.shape[0])
            bound = _trim_ribbon_bounds(
                domain,
                patch,
                cover,
                charts,
                points,
                loop_edges,
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )
    if not np.all(np.isfinite(bound)):
        return None
    return _TrimLoopRibbon(
        points[loop_edges[:, [0, 1, 1]]],
        bound,
        proof.arc_lower,
        proof.arc_upper,
        remaining_pairs,
    )


def verify_chart_chain(
    domain: domain_owner.MeshingDomain,
    patch: int,
    charts: np.ndarray,
    points: np.ndarray,
    cells: np.ndarray,
    boundary: np.ndarray,
    provenance: np.ndarray,
    restriction_required: np.ndarray,
    restriction_vertices: np.ndarray,
    restriction_edges: np.ndarray,
    restriction_parameters: np.ndarray,
    findings: list[embedding_state_owner.MeshCertificateFinding],
    resource_counts: list[tuple[str, int, int]],
    /,
    *,
    maximum_pairs: int = 200000,
    reserve_queries: Callable[[int], None] | None = None,
    topology_only: bool = False,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
    restriction_bank: BernsteinRestrictionBank | None = None,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> tuple[bool, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Verify positive chart coverage and retain trim/resource evidence in phase order."""
    maximum_pairs = min(200_000, maximum_pairs)
    empty = (False, np.empty((0, 3, 3)), np.empty((0,)), ())
    chain = _ChartChain(
        charts,
        points,
        cells,
        boundary,
        provenance,
        restriction_required,
        restriction_vertices,
        restriction_edges,
        restriction_parameters,
    )
    prepared = _prepare_chart_chain(
        chain,
        domain.source_revision,
        patch,
        findings,
        resource_counts,
        maximum_pairs,
        budget=budget,
    )
    if prepared is None:
        return empty
    patch_source = domain.patches[patch]
    polygons, ribbons, ribbon_bounds = [], [], []
    covers: list[tuple[np.ndarray, np.ndarray]] = []
    remaining_pairs = prepared.remaining_pairs
    for loop_index in range(len(patch_source.loops)):
        loop = _prepare_trim_loop(domain, patch, loop_index, chain)
        if loop is None:
            return empty
        polygons.append(loop.polygon)
        ribbon = _bound_trim_loop(
            domain,
            patch,
            loop_index,
            chain,
            loop,
            remaining_pairs,
            findings,
            resource_counts,
            maximum_pairs,
            reserve_queries,
            topology_only=topology_only,
            budget=budget,
            preparations=preparations,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        if ribbon is None:
            return empty
        remaining_pairs = ribbon.remaining_pairs
        covers.append((ribbon.lower, ribbon.upper))
        ribbons.append(ribbon.images)
        ribbon_bounds.append(ribbon.bounds)
    if np.any((provenance[:, 0] < 0) | (provenance[:, 0] >= len(patch_source.loops))):
        return empty
    for index, hole in enumerate(polygons[1:], start=1):
        if _winding(hole[0], polygons[0]) != 1:
            return empty
        if any(
            _winding(hole[0], other) != 0
            for j, other in enumerate(polygons[1:], start=1)
            if j != index
        ):
            return empty
    separated, separation_work, exhausted = _disjoint_trim_covers(covers, remaining_pairs)
    resource_counts.append(
        (f"source_trim_cross_loop_pairs:{patch}", separation_work, maximum_pairs)
    )
    if exhausted:
        findings.append(
            embedding_state_owner.MeshCertificateFinding(
                "source_trim_pair_capacity", "unresolved", "source_facet", (patch,)
            )
        )
    if not separated:
        return empty
    return (
        True,
        np.concatenate(ribbons) if ribbons else np.empty((0, 3, 3)),
        np.concatenate(ribbon_bounds) if ribbon_bounds else np.empty((0,)),
        prepared.restriction_ids,
    )


def _sampling_affine_loop(
    source: domain_owner.MeshingSurfacePatch,
    loop: tuple[domain_owner.PatchBoundaryUse, ...],
    /,
) -> np.ndarray | None:
    """affine-image polygon, never a sampled fit of the source curves."""

    starts: list[tuple[Fraction, ...]] = []
    ends: list[tuple[Fraction, ...]] = []
    for use in loop:
        if isinstance(use, domain_owner.PatchPoleUse):
            if use.trim_curve is not None:
                return None
            starts.append(tuple(Fraction(value) for value in use.start))
            ends.append(tuple(Fraction(value) for value in use.end))
            continue
        if any(
            root is not None
            for root in (
                use.first_root,
                use.last_root,
                use.start_vertex_root,
                use.end_vertex_root,
            )
        ):
            return None
        carrier = domain_owner.domain_trim_curve(source, use)
        lower = (Fraction(0), Fraction(0))
        extent = (Fraction(1), Fraction(1))
        if isinstance(carrier, constructors_owner.NormalizedTrimCurve):
            if any(
                root is not None
                for root in (
                    carrier.start_root,
                    carrier.end_root,
                    carrier.start_endpoint,
                    carrier.end_endpoint,
                )
            ):
                return None
            origin = np.asarray(carrier.lower, dtype=np.float64)
            scale = np.asarray(carrier.extent, dtype=np.float64)
            if (
                origin.ndim != 1
                or origin.shape != (2,)
                or scale.ndim != 1
                or scale.shape != (2,)
            ):
                return None
            lower = (Fraction(float(origin[0])), Fraction(float(origin[1])))
            extent = (Fraction(float(scale[0])), Fraction(float(scale[1])))
            carrier = carrier.curve
        if (
            not isinstance(carrier, CurveTrimSegment)
            or carrier.first_root is not None
            or carrier.last_root is not None
            or not isinstance(carrier.curve, AbstractCurve)
            or len(lower) != 2
            or len(extent) != 2
            or any(value == 0 for value in extent)
        ):
            return None
        coefficients = intersection_curve_owner.affine_curve_coefficients(
            carrier.curve, np.zeros((2,), dtype=np.float64)
        )
        if coefficients is None:
            return None
        parameters = (
            (carrier.last, carrier.first)
            if carrier.reversed
            else (carrier.first, carrier.last)
        )
        endpoints = [
            tuple(
                (constant + Fraction(parameter) * slope - offset) / scale
                for constant, slope, offset, scale in zip(
                    coefficients[0],
                    coefficients[1],
                    lower,
                    extent,
                    strict=True,
                )
            )
            for parameter in parameters
        ]
        starts.append(endpoints[0])
        ends.append(endpoints[1])
    if any(last != starts[(index + 1) % len(starts)] for index, last in enumerate(ends)):
        return None
    # Polygon predicates must see the same endpoint coordinates, not a
    # rounded replacement of a non-binary-rational affine source endpoint.
    vertices = np.asarray(starts, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape != (len(starts), 2):
        return None
    if not np.all(np.isfinite(vertices)) or any(
        Fraction(float(vertices[row, axis])) != starts[row][axis]
        for row in range(vertices.shape[0])
        for axis in range(vertices.shape[1])
    ):
        return None
    return vertices


def sampling_trim(
    source: domain_owner.MeshingSurfacePatch,
    intervals: int,
    relative_closure_tolerance: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None = None,
    restriction_bank: BernsteinRestrictionBank | None = None,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> TrimDomain:
    """Retain authored trims and their validated bounded joins for source queries."""
    loops: list[TrimLoop] = []
    for loop in source.loops:
        affine = _sampling_affine_loop(source, loop)
        if affine is not None:
            loops.append(PolygonTrimLoop(affine))
            continue
        curves: list[AbstractTrimCurve] = []
        for use in loop:
            match use:
                case domain_owner.PatchCurveUse():
                    curves.append(domain_owner.domain_trim_curve(source, use))
                case domain_owner.PatchPoleUse() if use.trim_curve is not None:
                    curves.append(use.trim_curve)
                case _:
                    curves.append(
                        CurveTrimSegment(
                            LineCurve(
                                np.asarray(use.start),
                                np.asarray(use.end) - np.asarray(use.start),
                            ),
                            0.0,
                            1.0,
                        )
                    )
        partitions = [
            np.linspace(*curve.parameter_interval, intervals + 1, dtype=np.float64)
            for curve in curves
        ]
        diameter = max(
            np.linalg.norm(box[1] - box[0])
            for curve, parameters in zip(curves, partitions, strict=True)
            for first, last in zip(parameters[:-1], parameters[1:], strict=True)
            for box in (
                curve.enclosure(
                    float(first), float(last), budget=budget, preparations=preparations
                ),
            )
        )
        loops.append(
            CurveTrimLoop(
                curves,
                tolerance=float(
                    np.nextafter(max(diameter, np.finfo(np.float64).tiny), np.inf)
                ),
                maximum_arcs=len(curves) * intervals,
                arc_parameters=partitions,
                relative_closure_tolerance=relative_closure_tolerance,
                budget=budget,
                preparations=preparations,
            )
        )
    return TrimDomain(loops[0], loops[1:])
