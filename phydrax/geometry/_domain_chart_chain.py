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

import math
from dataclasses import dataclass, field, fields, is_dataclass
from fractions import Fraction
from typing import Callable, cast, final, Literal, overload

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.geometry._domain_numerics as domain_numerics_owner
import phydrax.geometry._domain_preparation as domain_preparation_owner
import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._meshing_domain as domain_owner
import phydrax.geometry.brep._constructors as constructors_owner
import phydrax.geometry.brep._intersection_curve as intersection_curve_owner

from .._fingerprint import canonical_fingerprint, tree_content_equal
from .._issued_proofs import IssuedProofs
from .._meshcore import exact_orient2d
from .._strict import StrictModule
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
)
from ._atlas import (
    AbstractTrimCurve,
    CurveTrimLoop,
    overlapping_box_pairs,
    PolygonTrimLoop,
    TrimDomain,
    TrimLoop,
)
from ._chart_restriction import validate_chart_restrictions
from ._embedding_cells import SurfaceMapEnclosures
from ._interval_enclosure import (
    Interval,
    interval_add,
    interval_multiply,
    interval_subtract,
)
from .brep import _correspondence as correspondence_owner, _model as model_owner
from .brep._carrier_pieces import (
    admit_carrier_fingerprint,
    carrier_admission_current,
    carrier_payload,
    CarrierAdmission,
)
from .brep._intersection import NativePeriodEndpoint, TrimRootEndpoint
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
from .brep._placed import PlacedCurve
from .brep._request_preparations import TrimRequestPreparations
from .brep._root_bindings import BRepVertexRoot
from .brep._trim_helpers import period_scalar_bounds, PeriodOffset


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
    # Adjacent arcs share chord endpoints; each source point is enclosed once.
    point_boxes: dict[tuple[int, str], np.ndarray] = {}

    def point_box(curve: AbstractTrimCurve, value: float, /) -> np.ndarray:
        key = (id(curve), float(value).hex())
        box = point_boxes.get(key)
        if box is None:
            box = np.asarray(
                curve.enclosure(value, value, budget=budget, preparations=preparations)
            )
            point_boxes[key] = box
        return box

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
            np.max(np.abs(point_box(curve, first) - chart_edges[arc, 0]), axis=0),
            np.max(np.abs(point_box(curve, last) - chart_edges[arc, 1]), axis=0),
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


class SourceChartMapError(ValueError):
    """An actual source premise needed by the continuous chart map is unproved."""

    def __init__(self, reason: str, /) -> None:
        self.reason = reason
        super().__init__(reason)


def _source_rational_bounds(value: Fraction) -> tuple[float, float]:
    """Outward binary64 bounds of a rational, exact when it is representable."""
    rounded = float(value)
    exact = Fraction(rounded)
    return (
        math.nextafter(rounded, -math.inf) if exact > value else rounded,
        math.nextafter(rounded, math.inf) if exact < value else rounded,
    )


def _source_rational_interval(value: Fraction) -> tuple[np.ndarray, np.ndarray]:
    """Convert a rational without replacing it by its nearest binary value."""
    lower, upper = _source_rational_bounds(value)
    return np.asarray(lower), np.asarray(upper)


def _source_scalar_interval(value: int | float) -> tuple[np.ndarray, np.ndarray]:
    scalar = np.asarray(value)
    return scalar, scalar


def _source_atom(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


# Unlike scalar interval arithmetic, every batched row remains an ndarray,
# including one-row and point-reference requests.
type _ArrayInterval = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class _SourceTrimArc:
    carrier: AbstractTrimCurve
    first: Fraction
    last: Fraction
    identity: str
    preparations: TrimRequestPreparations
    direction: int
    affine: bool
    # Issued only after request-local native continuity and finite whole-use
    # first-jet proofs. A finite derivative box alone does not prove C0.
    mean_value_certified: bool = False
    # Endpoint enclosures are request-constant; at most the two arc ends are
    # retained, never interior query points.
    points: dict[Fraction, Interval] = field(default_factory=dict, compare=False, repr=False)

    def interval(self, first: Fraction, last: Fraction) -> tuple[float, float]:
        start = self.first + (self.last - self.first) * first
        end = self.first + (self.last - self.first) * last
        low, high = min(start, end), max(start, end)
        return float(_source_rational_interval(low)[0]), float(
            _source_rational_interval(high)[1]
        )

    def bounds(
        self, first: Fraction, last: Fraction, budget: CoordinateEnclosureBudget
    ) -> Interval:
        if first == last and first in self.points:
            return self.points[first]
        low, high = self.interval(first, last)
        box = type(self.carrier).enclosure(
            self.carrier, low, high, preparations=self.preparations, budget=budget
        )
        if not np.all(np.isfinite(box)):
            raise SourceChartMapError("source_trim_enclosure_unsupported")
        result: Interval = (box[0], box[1])
        if first == last and first in (0, 1):
            self.points[first] = result
        return result

    def derivative(
        self, first: Fraction, last: Fraction, budget: CoordinateEnclosureBudget
    ) -> Interval:
        low, high = self.interval(first, last)
        jet = type(self.carrier).derivative_bounds(
            self.carrier, low, high, preparations=self.preparations, budget=budget
        )
        if not all(np.all(np.isfinite(value)) for value in jet):
            raise SourceChartMapError("source_trim_derivative_singular")
        return interval_multiply(jet, _source_rational_interval(self.last - self.first))


@dataclass(frozen=True, slots=True)
class _SourceChartVertex:
    identity: str
    exact: tuple[Fraction | PeriodOffset, Fraction | PeriodOffset] | None = None
    arc: _SourceTrimArc | None = None
    parameter: Fraction = Fraction(0)
    parents: tuple[_SourceChartVertex, _SourceChartVertex] | None = None

    def bounds(self, budget: CoordinateEnclosureBudget) -> Interval:
        if self.arc is not None:
            return self.arc.bounds(self.parameter, self.parameter, budget)
        if self.parents is not None:
            first, second = (value.bounds(budget) for value in self.parents)
            return interval_add(
                interval_multiply(first, _source_rational_interval(1 - self.parameter)),
                interval_multiply(second, _source_rational_interval(self.parameter)),
            )
        if self.exact is None:
            raise SourceChartMapError("source_chart_endpoint_unsupported")
        pairs = [period_scalar_bounds(value) for value in self.exact]
        return np.asarray([value[0] for value in pairs]), np.asarray(
            [value[1] for value in pairs]
        )


@dataclass(frozen=True, slots=True)
class _AuthoredPointLayout:
    layout: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _SourceAuthoredVertex:
    point: np.ndarray
    root: BRepVertexRoot | None
    identity: str
    preparations: TrimRequestPreparations
    # One request-constant root point enclosure.
    memo: dict[str, Interval | _AuthoredPointLayout] = field(default_factory=dict, compare=False, repr=False)

    def require_point_layout(self) -> None:
        from ..discretization._coordinate_operands import _immutable_array_layout

        issued = self.memo.get("layout")
        if not isinstance(issued, _AuthoredPointLayout) or (
            _immutable_array_layout(self.point) != issued.layout
        ):
            raise SourceChartMapError("source_authored_point_layout_changed")

    def bounds(self, budget: CoordinateEnclosureBudget) -> Interval:
        self.require_point_layout()
        if self.root is None:
            return self.point, self.point
        cached = self.memo.get("point")
        if cached is None:
            box = self.root.point_enclosure(preparations=self.preparations, budget=budget)
            cached = box[0], box[1]
            self.memo["point"] = cached
        if isinstance(cached, _AuthoredPointLayout):
            raise SourceChartMapError("source_authored_point_enclosure_changed")
        return cached


@dataclass(frozen=True, slots=True)
class _SourceClosedEdge:
    """Original native edge plus its exact endpoint-affine vertex correction."""

    curve: AbstractCurve | intersection_curve_owner.IntersectionCurve
    parameters: tuple[float, float]
    roots: tuple[intersection_curve_owner.RootEndpoint | None, intersection_curve_owner.RootEndpoint | None]
    vertices: tuple[_SourceAuthoredVertex, _SourceAuthoredVertex]
    endpoint_bounds: tuple[float, float]
    identity: str
    preparations: TrimRequestPreparations
    # Root parameter ends and endpoint offsets are request-constant: one entry
    # each, recomputed never per reference piece.
    memo: dict[str, tuple[Interval, Interval]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def _parameter_ends(self, budget: CoordinateEnclosureBudget) -> tuple[Interval, Interval]:
        if "ends" in self.memo:
            return self.memo["ends"]
        ends: list[Interval] = []
        for parameter, root in zip(self.parameters, self.roots, strict=True):
            if root is None:
                ends.append(_source_scalar_interval(parameter))
            else:
                lower, upper = root.parameter_enclosure(
                    preparations=self.preparations, budget=budget
                )
                ends.append((np.asarray(lower), np.asarray(upper)))
        self.memo["ends"] = ends[0], ends[1]
        return ends[0], ends[1]

    def _native(
        self, first: Fraction, last: Fraction, budget: CoordinateEnclosureBudget,
        *, derivative: bool = False,
    ) -> Interval:
        ends = self._parameter_ends(budget)
        span: Interval
        values: list[Interval]
        if all(root is None for root in self.roots):
            start, end = Fraction(self.parameters[0]), Fraction(self.parameters[1])
            span = _source_rational_interval(end - start)
            values = [_source_rational_interval(start + t * (end - start)) for t in (first, last)]
        else:
            span = interval_subtract(ends[1], ends[0])
            values = [
                ends[int(t)] if t in (0, 1) else
                interval_add(ends[0], interval_multiply(_source_rational_interval(t), span))
                for t in (first, last)
            ]
        lower = min(float(value[0]) for value in values)
        upper = max(float(value[1]) for value in values)
        roots = (
            self.roots if self.parameters[0] < self.parameters[1]
            else (self.roots[1], self.roots[0])
        )
        if derivative:
            if isinstance(self.curve, intersection_curve_owner.IntersectionCurve):
                jet = self.curve.derivative_bounds(
                    lower, upper, endpoint_roots=roots,
                    preparations=self.preparations, budget=budget,
                )
            else:
                jet = self.curve.derivative_bounds(
                    lower, upper, preparations=self.preparations, budget=budget
                )
            result = interval_multiply(jet, span)
        else:
            if isinstance(self.curve, intersection_curve_owner.IntersectionCurve):
                box = self.curve.bounding_box(
                    lower, upper, endpoint_roots=roots,
                    preparations=self.preparations, budget=budget,
                )
            else:
                box = self.curve.bounding_box(
                    lower, upper, preparations=self.preparations, budget=budget
                )
            result = box[0], box[1]
        if not all(np.all(np.isfinite(value)) for value in result):
            raise SourceChartMapError("source_gap_native_edge_jet_unsupported")
        return result

    def _offsets(self, budget: CoordinateEnclosureBudget) -> tuple[Interval, Interval]:
        if "offsets" in self.memo:
            return self.memo["offsets"]
        result: list[Interval] = []
        for i, (vertex, bound) in enumerate(zip(self.vertices, self.endpoint_bounds, strict=True)):
            if bound == 0.0:
                result.append((np.zeros(3), np.zeros(3)))
                continue
            offset = interval_subtract(
                vertex.bounds(budget), self._native(Fraction(i), Fraction(i), budget)
            )
            lower, upper = np.maximum(offset[0], -bound), np.minimum(offset[1], bound)
            if np.any(lower > upper):
                raise SourceChartMapError("source_gap_endpoint_bound_inconsistent")
            result.append((lower, upper))
        self.memo["offsets"] = result[0], result[1]
        return result[0], result[1]

    def bounds(
        self, first: Fraction, last: Fraction, budget: CoordinateEnclosureBudget
    ) -> Interval:
        if first == last and first in (0, 1):
            return self.vertices[int(first)].bounds(budget)
        offsets = self._offsets(budget)
        t = (_source_rational_interval(min(first, last))[0],
             _source_rational_interval(max(first, last))[1])
        correction = interval_add(
            offsets[0], interval_multiply(t, interval_subtract(offsets[1], offsets[0]))
        )
        return interval_add(self._native(first, last, budget), correction)

    def derivative(
        self, first: Fraction, last: Fraction, budget: CoordinateEnclosureBudget
    ) -> Interval:
        offsets = self._offsets(budget)
        return interval_add(
            self._native(first, last, budget, derivative=True),
            interval_subtract(offsets[1], offsets[0]),
        )


@dataclass(frozen=True, slots=True)
class _SourceClosedEdgePoint:
    edge: _SourceClosedEdge
    parameter: Fraction

    def bounds(self, budget: CoordinateEnclosureBudget) -> Interval:
        return self.edge.bounds(self.parameter, self.parameter, budget)


@dataclass(frozen=True, slots=True)
class _SourceGapArc:
    """Whole native face-to-common-edge ribbon, not a scalar gap surrogate."""

    uv: _SourceTrimArc
    edge: _SourceClosedEdge
    first: Fraction
    last: Fraction
    surface: domain_owner.AbstractSurfacePatch
    distance_bound: float

    @overload
    def residuals(
        self,
        edge: _EdgeRows,
        cache: _ArcJetCache,
        budget: CoordinateEnclosureBudget,
        *,
        value: Literal[True],
        jacobian: Literal[True],
    ) -> tuple[_ArrayInterval, Interval]: ...

    @overload
    def residuals(
        self,
        edge: _EdgeRows,
        cache: _ArcJetCache,
        budget: CoordinateEnclosureBudget,
        *,
        value: Literal[True],
        jacobian: bool,
    ) -> tuple[_ArrayInterval, Interval | None]: ...

    @overload
    def residuals(
        self,
        edge: _EdgeRows,
        cache: _ArcJetCache,
        budget: CoordinateEnclosureBudget,
        *,
        value: bool,
        jacobian: bool,
    ) -> tuple[_ArrayInterval | None, Interval | None]: ...

    def residuals(
        self,
        edge: _EdgeRows,
        cache: _ArcJetCache,
        budget: CoordinateEnclosureBudget,
        *,
        value: bool,
        jacobian: bool,
    ) -> tuple[_ArrayInterval | None, Interval | None]:
        """Native residual E-F(C) (M,3) and its derivative on the active rows."""
        count = len(edge.lower)
        if self.distance_bound == 0.0:
            # Both native face correspondence and edge endpoint offsets were
            # independently certified exactly zero, so this residual is zero.
            zeros = np.zeros((count, 3))
            return ((zeros, zeros) if value else None), ((zeros, zeros) if jacobian else None)
        uv, uv_jet = _arc_jets(self.uv, edge, cache, budget, derivative=jacobian)
        span = self.last - self.first
        residual = (np.empty((count, 3)), np.empty((count, 3)))
        natives = []
        for index, (lower, upper) in enumerate(zip(edge.lower, edge.upper, strict=True)):
            a, b = self.first + lower * span, self.first + upper * span
            if jacobian:
                natives.append(self.edge.derivative(a, b, budget))
            if not value:
                continue
            physical = self.surface.bounding_box(
                np.stack((uv[0][index], uv[1][index])),
                preparations=self.uv.preparations, budget=budget,
            )
            result = interval_subtract(self.edge.bounds(a, b, budget), (physical[0], physical[1]))
            # Independently certified whole-edge norm bounds may tighten the
            # coordinate box, but never replace the native residual map.
            residual[0][index] = np.maximum(result[0], -self.distance_bound)
            residual[1][index] = np.minimum(result[1], self.distance_bound)
            if np.any(residual[0][index] > residual[1][index]):
                raise SourceChartMapError("source_gap_ribbon_bound_inconsistent")
        if uv_jet is None:
            return (residual if value else None), None
        native = interval_multiply(
            (np.stack([jet[0] for jet in natives]), np.stack([jet[1] for jet in natives])),
            _source_rational_interval(span),
        )
        surface_jet = self.surface.derivative_bounds_batch(
            np.stack(uv, axis=1), preparations=self.uv.preparations, budget=budget
        )
        physical: Interval = (np.zeros((count, 3)), np.zeros((count, 3)))
        for axis in range(2):
            physical = interval_add(
                physical,
                interval_multiply(
                    (surface_jet[0][:, :, axis], surface_jet[1][:, :, axis]),
                    (uv_jet[0][:, axis, None], uv_jet[1][:, axis, None]),
                ),
            )
        jet = interval_subtract(native, physical)
        if not all(np.all(np.isfinite(part)) for part in jet):
            raise SourceChartMapError("source_gap_ribbon_derivative_unsupported")
        return (residual if value else None), jet


@dataclass(frozen=True, slots=True)
class _SourceGapCompletion:
    edges: tuple[_SourceClosedEdge, ...]
    use_bounds: dict[tuple[int, int, int], float]
    endpoint_ids: dict[tuple[int, int], str]
    tolerance: float
    identity: str


type _Barycentric = tuple[
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
]
# Exact rational chart corners, or endpoint enclosures, and the affine columns.
type _ChartConstants = tuple[
    tuple[tuple[Fraction, Fraction], ...] | None,
    tuple[Interval, ...] | None,
    tuple[Interval, ...],
]
# Authored-vertex corner offsets and their affine columns.
type _CornerOffsets = tuple[tuple[Interval, ...], tuple[Interval, ...]]
type _EdgeRowCache = dict[tuple[int, int], _EdgeRows | None]
type _ArcJetCache = dict[tuple[int, Fraction, Fraction], list]
type _UseKey = tuple[int, int, int]
type _EndpointKey = tuple[int, int]
# One authenticated trim use: its carrier, native range and endpoint keys.
type _SourceTrimUse = tuple[
    constructors_owner.NormalizedTrimCurve,
    tuple[float, float],
    tuple[_EndpointKey, _EndpointKey],
]
# One original chart triangulation record from the boundary source.
type _ChartTriangulation = tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]
_GRADIENTS = ((-1, -1), (1, 0), (0, 1))


@dataclass(frozen=True, slots=True)
class _EdgeRows:
    """Exact edge coordinates s=lambda_i+lambda_j, t=lambda_j/s of active rows.

    Rows whose closed subtriangle lies on s=0 carry the proven removable zero
    and are omitted. ``lower``/``upper`` are the exact t extrema at nonzero-s
    vertices; ``t`` and ``s`` are their outward (M,1) binary enclosures.
    """

    rows: np.ndarray
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]
    t: _ArrayInterval
    s: _ArrayInterval


# The entire native arc t in [0, 1] on its edge, s in [0, 1].
_WHOLE_ARC = _EdgeRows(
    np.zeros(1, dtype=np.int64),
    (Fraction(0),),
    (Fraction(1),),
    (np.zeros((1, 1)), np.ones((1, 1))),
    (np.zeros((1, 1)), np.ones((1, 1))),
)


def _edge_rows(
    barycentric: list[_Barycentric],
    rows: dict[tuple[int, int], _EdgeRows | None],
    first: int,
    last: int,
) -> _EdgeRows | None:
    key = (first, last)
    if key in rows:
        return rows[key]
    active, lowers, uppers, t, s = [], [], [], [], []
    for index, corners in enumerate(barycentric):
        sums = [row[first] + row[last] for row in corners]
        maximum = max(sums)
        if maximum == 0:
            continue
        # Linear-fractional extrema on a simplex occur at its nonzero-s
        # vertices. Zero-s vertices contribute no new limiting ratios.
        ratios = [
            row[last] / total for row, total in zip(corners, sums, strict=True) if total > 0
        ]
        lower, upper = min(ratios), max(ratios)
        active.append(index)
        lowers.append(lower)
        uppers.append(upper)
        t.append((_source_rational_bounds(lower)[0], _source_rational_bounds(upper)[1]))
        s.append((_source_rational_bounds(min(sums))[0], _source_rational_bounds(maximum)[1]))
    result = None
    if active:
        t_bounds, s_bounds = np.asarray(t), np.asarray(s)
        result = _EdgeRows(
            np.asarray(active, dtype=np.int64),
            tuple(lowers),
            tuple(uppers),
            (t_bounds[:, :1], t_bounds[:, 1:]),
            (s_bounds[:, :1], s_bounds[:, 1:]),
        )
    rows[key] = result
    return result


def _arc_jets(
    arc: _SourceTrimArc,
    edge: _EdgeRows,
    cache: dict[tuple[int, Fraction, Fraction], list],
    budget: CoordinateEnclosureBudget,
    /,
    *,
    derivative: bool,
) -> tuple[_ArrayInterval, _ArrayInterval | None]:
    """Native trim enclosure (M,2) and first jet on each active row's t-interval.

    Chart and ribbon terms of one call request the identical native interval;
    it is enclosed once for that call only.
    """
    boxes, jets = [], []
    for lower, upper in zip(edge.lower, edge.upper, strict=True):
        key = (id(arc), lower, upper)
        entry = cache.get(key)
        if entry is None:
            entry = [arc.bounds(lower, upper, budget), None]
            cache[key] = entry
        if derivative and entry[1] is None:
            entry[1] = arc.derivative(lower, upper, budget)
        boxes.append(entry[0])
        jets.append(entry[1])
    box: _ArrayInterval = (np.stack([value[0] for value in boxes]), np.stack([value[1] for value in boxes]))
    if not derivative:
        return box, None
    return box, (np.stack([value[0] for value in jets]), np.stack([value[1] for value in jets]))


def _chord_deviation(
    arc: _SourceTrimArc,
    edge: _EdgeRows,
    curve: _ArrayInterval,
    jet: _ArrayInterval | None,
    cache: dict[tuple[int, Fraction, Fraction], list],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[_ArrayInterval, _ArrayInterval | None]:
    """Enclose delta=C-chord (M,2) and, with a jet, delta'=C'-(C(1)-C(0)).

    The mean-value intersection is used only for an arc whose native use has
    proved C0 joins, C1 source pieces and a finite whole-use first jet. These
    imply absolute continuity, including root-defined affine parameter ends.
    Then delta(t)-delta(m) is the integral of delta' over [m,t]. Unsupported
    or unproved carriers retain the direct enclosure, never a sampled proof.
    """
    start = arc.bounds(Fraction(0), Fraction(0), budget)
    end = arc.bounds(Fraction(1), Fraction(1), budget)
    span = interval_subtract(end, start)
    direct = cast(_ArrayInterval, interval_subtract(
        curve, interval_add(start, interval_multiply(edge.t, span))
    ))
    if jet is None:
        return direct, None
    slope = cast(_ArrayInterval, interval_subtract(jet, span))
    if not arc.mean_value_certified or all(
        lower == upper for lower, upper in zip(edge.lower, edge.upper, strict=True)
    ):
        return direct, slope
    middles, offsets, points = [], [], []
    for lower, upper in zip(edge.lower, edge.upper, strict=True):
        middle = (lower + upper) / 2
        key = (id(arc), middle, middle)
        entry = cache.get(key)
        if entry is None:
            entry = [arc.bounds(middle, middle, budget), None]
            cache[key] = entry
        points.append(entry[0])
        middles.append(_source_rational_bounds(middle))
        offsets.append(
            (_source_rational_bounds(lower - middle)[0], _source_rational_bounds(upper - middle)[1])
        )
    middle_bounds, offset_bounds = np.asarray(middles), np.asarray(offsets)
    center = interval_subtract(
        (np.stack([point[0] for point in points]), np.stack([point[1] for point in points])),
        interval_add(
            start, interval_multiply((middle_bounds[:, :1], middle_bounds[:, 1:]), span)
        ),
    )
    mean_value = interval_add(
        center, interval_multiply(slope, (offset_bounds[:, :1], offset_bounds[:, 1:]))
    )
    lower, upper = np.maximum(direct[0], mean_value[0]), np.minimum(direct[1], mean_value[1])
    if np.any(lower > upper):
        raise SourceChartMapError("source_trim_enclosure_inconsistent")
    return (lower, upper), slope


def _weight_bounds(barycentric: list[_Barycentric], /) -> _ArrayInterval:
    """Outward (N,corner,vertex) bounds of exact barycentric weights."""
    bounds = np.asarray(
        [
            [[_source_rational_bounds(weight) for weight in row] for row in corners]
            for corners in barycentric
        ]
    )
    return bounds[..., 0], bounds[..., 1]


def _scatter_add(value: _ArrayInterval, rows: np.ndarray, addend: Interval, /) -> _ArrayInterval:
    """Outward-add an (M,...) term to the active rows of an (N,...) interval."""
    lower, upper = value[0].copy(), value[1].copy()
    lower[rows], upper[rows] = interval_add((value[0][rows], value[1][rows]), addend)
    return lower, upper

@dataclass(frozen=True, slots=True)
class SourceChartMap:
    """Request-local native map on one original standard reference triangle.

    At an edge opposite vertex k, s=lambda_i+lambda_j and t=lambda_j/s.
    The correction is s² delta(t). Its one-sided/a.e. derivative is
    s [2 ds delta + (d lambda_j - t ds) delta']. Native continuous piecewise-C1
    trims with bounded first jets are Lipschitz, hence absolutely continuous
    on reference segments. Bounded delta and delta' prove the removable zero
    derivative at s=0. The returned hull includes both sides of every native
    chart join; it does not assert a classical derivative there. No inverse
    power of s occurs, including on point reference triangles.

    Under independently validated native gap authority the physical map is
    G=F(U)+sum(lambda_i*d_i)+sum(s²*(E-F(C)-linear(d_i,d_j))).
    E is the common native edge with endpoint-affine corrections to authored
    literal/root vertices. Positive gaps remain positive evidence: the bound
    covers the full raw face and the original native edges. The consumer must
    prove orientation with the raw/completed jet hull and embedding with G;
    merely constructing these request-local ribbons is not a certificate.
    """

    surface: domain_owner.AbstractSurfacePatch
    patch: int
    cell_row: int
    vertices: tuple[int, int, int]
    endpoints: tuple[_SourceChartVertex, _SourceChartVertex, _SourceChartVertex]
    arcs: tuple[tuple[int, int, _SourceTrimArc], ...]
    edge_ids: tuple[tuple[int, int, str], ...]
    map_id: str
    preparations: TrimRequestPreparations
    authored_deviation_bound: float = 0.0
    gap_vertices: tuple[_SourceClosedEdgePoint | None, ...] = (None, None, None)
    gap_vertex_bounds: tuple[float, ...] = (0.0, 0.0, 0.0)
    gap_arcs: tuple[tuple[int, int, _SourceGapArc], ...] = ()
    # Request-constant affine chart columns and corner offsets: one entry each,
    # never per reference piece, within the pre-admitted map record.
    memo: dict[str, object] = field(default_factory=dict, compare=False, repr=False)
    # Ephemeral issuance under the authenticated domain and original request.
    # No public protocol assertion or archived identifier grants cache reuse.
    enclosure_authority: tuple[str, str, int] | None = field(
        default=None, compare=False, repr=False
    )

    def trace_identity(self, local_vertices: tuple[int, ...], /) -> str | None:
        if len(local_vertices) == 1 and local_vertices[0] in (0, 1, 2):
            return self.endpoints[local_vertices[0]].identity
        if len(local_vertices) == 2:
            for first, last, identity in self.edge_ids:
                if local_vertices == (first, last):
                    return identity
        return None

    def _barycentric(self, references: np.ndarray, /) -> list[_Barycentric]:
        values = np.asarray(references, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[0] == 0
            or values.shape[1:] != (3, 2)
            or not np.all(np.isfinite(values))
        ):
            raise SourceChartMapError("source_chart_reference_invalid")
        result: list[_Barycentric] = []
        for reference in values.tolist():
            corners = []
            for x, y in reference:
                first, second = Fraction(x), Fraction(y)
                corners.append((1 - first - second, first, second))
            result.append((corners[0], corners[1], corners[2]))
        if any(value < 0 for corners in result for row in corners for value in row):
            raise SourceChartMapError("source_chart_reference_outside_simplex")
        return result

    def _chart_constants(self, budget: CoordinateEnclosureBudget) -> _ChartConstants:
        """Exact rational corners or endpoint enclosures, and the affine columns."""
        constants = self.memo.get("chart")
        if constants is not None:
            return cast(_ChartConstants, constants)
        rational: list[tuple[Fraction, Fraction]] = []
        for vertex in self.endpoints:
            coordinate = vertex.exact
            if coordinate is None:
                break
            first_coordinate, second_coordinate = coordinate
            if not isinstance(first_coordinate, Fraction) or not isinstance(second_coordinate, Fraction):
                break
            rational.append((first_coordinate, second_coordinate))
        if len(rational) == 3:
            columns = []
            for axis in range(2):
                entries = [
                    _source_rational_bounds(rational[axis + 1][k] - rational[0][k])
                    for k in range(2)
                ]
                columns.append(
                    (np.asarray([x[0] for x in entries]), np.asarray([x[1] for x in entries]))
                )
            constants = (tuple(rational), None, tuple(columns))
        else:
            endpoints = tuple(vertex.bounds(budget) for vertex in self.endpoints)
            constants = (
                None,
                endpoints,
                tuple(interval_subtract(endpoints[axis + 1], endpoints[0]) for axis in range(2)),
            )
        self.memo["chart"] = constants
        return constants

    def _chart(
        self,
        barycentric: list[_Barycentric],
        rows: _EdgeRowCache,
        cache: _ArcJetCache,
        budget: CoordinateEnclosureBudget,
        *,
        jacobian: bool,
    ) -> tuple[_ArrayInterval, _ArrayInterval | None]:
        """Endpoint-corrected UV boxes (N,2) and chart jets (N,2,2) if requested."""
        count = len(barycentric)
        rational, endpoints, constant_columns = self._chart_constants(budget)
        if rational is not None:
            lower = np.empty((count, 2))
            upper = np.empty((count, 2))
            for index, corners in enumerate(barycentric):
                images = [
                    tuple(
                        sum((row[i] * rational[i][axis] for i in range(3)), Fraction(0))
                        for axis in range(2)
                    )
                    for row in corners
                ]
                for k in range(2):
                    lower[index, k] = _source_rational_bounds(min(row[k] for row in images))[0]
                    upper[index, k] = _source_rational_bounds(max(row[k] for row in images))[1]
            value: _ArrayInterval = (lower, upper)
        else:
            # The affine chart image of a reference subtriangle is the convex
            # hull of its exact corner rows' images; independent per-weight
            # hulls would overestimate it by up to the sum of all vertices.
            weights = _weight_bounds(barycentric)
            corner_images = []
            for corner in range(3):
                image: Interval = (np.zeros((count, 2)), np.zeros((count, 2)))
                for vertex, point in enumerate(cast(tuple[Interval, ...], endpoints)):
                    image = interval_add(
                        image,
                        interval_multiply(
                            (weights[0][:, corner, vertex, None], weights[1][:, corner, vertex, None]),
                            point,
                        ),
                    )
                corner_images.append(image)
            value = (
                np.min(np.stack([image[0] for image in corner_images], axis=1), axis=1),
                np.max(np.stack([image[1] for image in corner_images], axis=1), axis=1),
            )
        columns = [
            (
                np.repeat(column[0][None], count, axis=0),
                np.repeat(column[1][None], count, axis=0),
            )
            for column in constant_columns
        ]
        for first, last, arc in self.arcs:
            if arc.affine:
                # Native affine coefficients prove delta identically zero;
                # this is not a small-residual or rounded-coordinate test.
                continue
            edge = _edge_rows(barycentric, rows, first, last)
            if edge is None:
                # The proven removable value and derivative are exactly zero.
                continue
            nondegenerate = any(
                lower != upper for lower, upper in zip(edge.lower, edge.upper, strict=True)
            )
            curve, jet = _arc_jets(
                arc, edge, cache, budget, derivative=jacobian or nondegenerate
            )
            delta, derivative = _chord_deviation(arc, edge, curve, jet, cache, budget)
            value = _scatter_add(
                value, edge.rows, interval_multiply(interval_multiply(edge.s, edge.s), delta)
            )
            if not jacobian or derivative is None:
                continue
            for axis in range(2):
                ds = _GRADIENTS[first][axis] + _GRADIENTS[last][axis]
                dt_numerator = interval_subtract(
                    _source_scalar_interval(_GRADIENTS[last][axis]),
                    interval_multiply(edge.t, _source_scalar_interval(ds)),
                )
                correction = interval_multiply(
                    edge.s,
                    interval_add(
                        interval_multiply(delta, _source_scalar_interval(2 * ds)),
                        interval_multiply(dt_numerator, derivative),
                    ),
                )
                columns[axis] = _scatter_add(columns[axis], edge.rows, correction)
        if not jacobian:
            return value, None
        return value, (
            np.stack([column[0] for column in columns], axis=2),
            np.stack([column[1] for column in columns], axis=2),
        )

    def _corner_offsets(self, budget: CoordinateEnclosureBudget) -> _CornerOffsets:
        """Request-constant authored-vertex offsets and their affine columns."""
        offsets = self.memo.get("offsets")
        if offsets is not None:
            return cast(_CornerOffsets, offsets)
        values: list[Interval] = []
        for endpoint, point, bound in zip(
            self.endpoints, self.gap_vertices, self.gap_vertex_bounds, strict=True
        ):
            if point is None or bound == 0.0:
                values.append((np.zeros(3), np.zeros(3)))
                continue
            raw = self.surface.bounding_box(
                np.stack(endpoint.bounds(budget)), preparations=self.preparations, budget=budget
            )
            offset = interval_subtract(point.bounds(budget), (raw[0], raw[1]))
            lower, upper = np.maximum(offset[0], -bound), np.minimum(offset[1], bound)
            if np.any(lower > upper):
                raise SourceChartMapError("source_gap_corner_bound_inconsistent")
            values.append((lower, upper))
        offsets = (
            tuple(values),
            tuple(interval_subtract(values[axis + 1], values[0]) for axis in range(2)),
        )
        self.memo["offsets"] = offsets
        return offsets

    def _completion(
        self,
        barycentric: list[_Barycentric],
        rows: _EdgeRowCache,
        cache: _ArcJetCache,
        budget: CoordinateEnclosureBudget,
        *,
        jacobian: bool,
    ) -> tuple[_ArrayInterval, _ArrayInterval | None]:
        """Affine corner correction plus homogeneous physical ribbon residuals."""
        count = len(barycentric)
        offsets, constant_columns = self._corner_offsets(budget)
        weights = _weight_bounds(barycentric)
        # Exact corner rows bound the affine corner term by their convex hull.
        corner_images = []
        for corner in range(3):
            image: Interval = (np.zeros((count, 3)), np.zeros((count, 3)))
            for vertex, offset in enumerate(offsets):
                image = interval_add(
                    image,
                    interval_multiply(
                        (weights[0][:, corner, vertex, None], weights[1][:, corner, vertex, None]),
                        offset,
                    ),
                )
            corner_images.append(image)
        value: _ArrayInterval = (
            np.min(np.stack([image[0] for image in corner_images], axis=1), axis=1),
            np.max(np.stack([image[1] for image in corner_images], axis=1), axis=1),
        )
        columns = [
            (
                np.repeat(column[0][None], count, axis=0),
                np.repeat(column[1][None], count, axis=0),
            )
            for column in constant_columns
        ]
        for first, last, gap in self.gap_arcs:
            edge = _edge_rows(barycentric, rows, first, last)
            if edge is None:
                continue
            slope = interval_subtract(offsets[last], offsets[first])
            chord = interval_add(offsets[first], interval_multiply(edge.t, slope))
            residual, residual_jet = gap.residuals(edge, cache, budget, value=True, jacobian=jacobian)
            delta = interval_subtract(residual, chord)
            value = _scatter_add(
                value, edge.rows, interval_multiply(interval_multiply(edge.s, edge.s), delta)
            )
            if residual_jet is None:
                continue
            derivative = interval_subtract(residual_jet, slope)
            for axis in range(2):
                ds = _GRADIENTS[first][axis] + _GRADIENTS[last][axis]
                dt = interval_subtract(
                    _source_scalar_interval(_GRADIENTS[last][axis]),
                    interval_multiply(edge.t, _source_scalar_interval(ds)),
                )
                columns[axis] = _scatter_add(columns[axis], edge.rows, interval_multiply(
                    edge.s, interval_add(
                        interval_multiply(delta, _source_scalar_interval(2 * ds)),
                        interval_multiply(dt, derivative),
                    )
                ))
        if not jacobian:
            return value, None
        return value, (
            np.stack([column[0] for column in columns], axis=2),
            np.stack([column[1] for column in columns], axis=2),
        )

    @classmethod
    def enclosures(
        cls,
        requests: list[tuple[SourceChartMap, np.ndarray]],
        /,
        *,
        image: bool,
        jacobian: bool,
        budget: CoordinateEnclosureBudget,
    ) -> list[SurfaceMapEnclosures]:
        """Enclose (map, (N,3,2) references) requests in one native evaluation.

        Every row is the outward enclosure of its own closed subtriangle: exact
        barycentric rows, the endpoint-corrected chart, native trim/edge/surface
        enclosures and the physical ribbon terms. Rows share only request-constant
        chart columns/corner offsets, one row-ordered surface image bank per map,
        one native surface jet call per surface, and, within one map, identical
        native trim intervals; nothing is interpolated between rows. The
        orientation jet is the hull of the raw authored and completed jets.
        """
        if not image and not jacobian:
            raise SourceChartMapError("source_chart_request_empty")
        prepared = [
            (source, source._barycentric(references), {}, {}) for source, references in requests
        ]
        count = sum(len(item[1]) for item in prepared)
        kinds = int(image) + int(jacobian)
        charts: list[tuple[_ArrayInterval, _ArrayInterval | None]] = []
        raw_images: list[_ArrayInterval | None] = [None] * len(prepared)
        raw_jacobians: list[_ArrayInterval | None] = [None] * len(prepared)
        with budget.temporary_scope():
            budget.reserve(kinds * count, 4096 * kinds * count)
            for index, (source, barycentric, rows, cache) in enumerate(prepared):
                try:
                    uv, chart_jet = source._chart(
                        barycentric, rows, cache, budget, jacobian=jacobian
                    )
                    charts.append((uv, chart_jet))
                    if image:
                        budget.reserve(0, 128 + 32 * len(barycentric))
                        boxes = source.surface.bounding_boxes(
                            np.stack(uv, axis=1),
                            preparations=source.preparations, budget=budget,
                        )
                        # The bank stops after its first nonfinite row.
                        if not np.all(np.isfinite(boxes)):
                            raise SourceChartMapError("source_surface_enclosure_unsupported")
                        raw_images[index] = boxes[:, 0].copy(), boxes[:, 1].copy()
                        del boxes
                except (CoordinateEnclosureResourceError, SourceChartMapError):
                    raise
                except (TypeError, ValueError, NotImplementedError) as error:
                    label = "enclosure" if image else "derivative"
                    raise SourceChartMapError(
                        f"source_surface_{label}_unsupported:{error}"
                    ) from error
            if jacobian:
                surfaces: dict[tuple[int, int], list[int]] = {}
                for index, (source, *_) in enumerate(prepared):
                    surfaces.setdefault(
                        (id(source.surface), id(source.preparations)), []
                    ).append(index)
                for indices in surfaces.values():
                    source = prepared[indices[0]][0]
                    try:
                        surface_jet = source.surface.derivative_bounds_batch(
                            np.concatenate([np.stack(charts[index][0], axis=1) for index in indices]),
                            preparations=source.preparations, budget=budget,
                        )
                        if not all(np.all(np.isfinite(value)) for value in surface_jet):
                            raise SourceChartMapError("source_surface_derivative_singular")
                        chart_jet = (
                            np.concatenate([cast(_ArrayInterval, charts[index][1])[0] for index in indices]),
                            np.concatenate([cast(_ArrayInterval, charts[index][1])[1] for index in indices]),
                        )
                        total = len(chart_jet[0])
                        raw: Interval = (np.zeros((total, 3, 2)), np.zeros((total, 3, 2)))
                        for axis in range(2):
                            raw = interval_add(
                                raw,
                                interval_multiply(
                                    (surface_jet[0][:, :, axis, None], surface_jet[1][:, :, axis, None]),
                                    (chart_jet[0][:, None, axis, :], chart_jet[1][:, None, axis, :]),
                                ),
                            )
                    except (CoordinateEnclosureResourceError, SourceChartMapError):
                        raise
                    except (TypeError, ValueError, NotImplementedError) as error:
                        raise SourceChartMapError(
                            f"source_surface_derivative_unsupported:{error}"
                        ) from error
                    raw_arrays = cast(_ArrayInterval, raw)
                    start = 0
                    for index in indices:
                        stop = start + len(prepared[index][1])
                        raw_jacobians[index] = raw_arrays[0][start:stop], raw_arrays[1][start:stop]
                        start = stop
        results = []
        for (source, barycentric, rows, cache), raw_image, raw_jacobian in zip(
            prepared, raw_images, raw_jacobians, strict=True
        ):
            if source.authored_deviation_bound == 0.0:
                results.append(SurfaceMapEnclosures(raw_image, raw_jacobian, raw_jacobian))
                continue
            try:
                with budget.temporary_scope():
                    budget.reserve(kinds * len(barycentric), 8192 * kinds * len(barycentric))
                    correction, correction_jet = source._completion(
                        barycentric, rows, cache, budget, jacobian=jacobian
                    )
            except (CoordinateEnclosureResourceError, SourceChartMapError):
                raise
            except (TypeError, ValueError, NotImplementedError) as error:
                raise SourceChartMapError(f"source_gap_ribbon_unsupported:{error}") from error
            completed_image = None if raw_image is None else cast(_ArrayInterval, interval_add(raw_image, correction))
            if raw_jacobian is None or correction_jet is None:
                results.append(SurfaceMapEnclosures(completed_image, None, None))
                continue
            completed = cast(_ArrayInterval, interval_add(raw_jacobian, correction_jet))
            results.append(SurfaceMapEnclosures(
                completed_image,
                completed,
                (np.minimum(raw_jacobian[0], completed[0]), np.maximum(raw_jacobian[1], completed[1])),
            ))
        return results


def _source_use_signature(
    source: domain_owner.MeshingSurfacePatch,
    use: domain_owner.PatchCurveUse,
    budget: CoordinateEnclosureBudget,
) -> (
    tuple[str, tuple[tuple[tuple[int, int], ...], ...]]
    | tuple[str, domain_owner.AbstractSurfacePatch, AbstractCurve | AbstractTrimCurve]
):
    """Exact native affine lift, or identical authored surface/pcurve expression."""
    coefficients = (
        intersection_curve_owner.affine_curve_coefficients(use.pcurve, np.zeros(2))
        if isinstance(use.pcurve, AbstractCurve)
        else None
    )
    if isinstance(source.surface, domain_owner.PlanePatch) and coefficients is not None:
        budget.reserve(1, 256)
        origin = np.asarray(source.surface.origin)
        axes = np.stack((source.surface.first_axis, source.surface.second_axis))
        lifted = tuple(
            tuple(
                (Fraction(float(origin[k])) if order == 0 else Fraction(0))
                + sum(
                    (Fraction(float(axes[j, k])) * coefficients[order][j] for j in range(2)),
                    Fraction(0),
                )
                for k in range(3)
            )
            for order in range(2)
        )
        return ("affine", tuple(tuple(_source_atom(x) for x in row) for row in lifted))
    return ("composition", source.surface, use.pcurve)


def _source_intersection_continuity(
    curve: intersection_curve_owner.IntersectionCurve,
    first: float,
    last: float,
    head: int,
    tail: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    verified: dict[int, intersection_curve_owner.IntersectionCurve],
    /,
) -> None:
    """Reprove shared physical roots and smooth pieces on actual query support."""
    if id(curve) not in verified:
        # Reconstruct through the canonical owner: it re-certifies shared
        # node roots, graph inclusions and transitions from source equations.
        # Neither a stored fully_certified bit nor node samples prove C0.
        with budget.temporary_scope():
            budget.reserve(0, 4 * budget.live_object_storage_upper(curve) + 8192)
            rebuilt = intersection_curve_owner.IntersectionCurve(
                curve.first,
                curve.second,
                chart_axes=curve.chart_axes,
                chart_start=curve.chart_start,
                chart_end=curve.chart_end,
                chart_pieces=curve.chart_pieces,
                box_lower=curve.box_lower,
                box_upper=curve.box_upper,
                preconditioners=curve.preconditioners,
                contraction=curve.contraction,
                certified=curve.certified,
                transition_shifts=curve.transition_shifts,
                closed=curve.closed,
                start_kind=curve.start_kind,
                end_kind=curve.end_kind,
                node_axes=curve.node_axes,
                node_values=curve.node_values,
                node_pieces=curve.node_pieces,
                node_references=curve.node_references,
                node_period_shifts=curve.node_period_shifts,
                preparations=preparations,
                budget=budget,
            )
            if not rebuilt.fully_certified or not tree_content_equal(rebuilt, curve):
                raise SourceChartMapError("source_trim_atlas_continuity_unproved")
            del rebuilt
        budget.reserve(1, 128)
        verified[id(curve)] = curve
    boxes = curve.parameter_enclosures(
        first, last, minimum_chart=head, maximum_chart=tail,
        source_extension=True, preparations=preparations, budget=budget,
    )
    if not np.all(np.isfinite(boxes)) or any(
        not curve.first.patch.is_c1_on(
            box[:, :2], preparations=preparations, budget=budget
        )
        or not curve.second.patch.is_c1_on(
            box[:, 2:], preparations=preparations, budget=budget
        )
        for box in boxes
    ):
        raise SourceChartMapError("source_trim_atlas_piece_regularity_unproved")


def _source_piecewise_trim_continuity(
    segment: CurveTrimSegment,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    verified: dict[int, intersection_curve_owner.IntersectionCurve],
    /,
) -> bool:
    """C0 in UV, not merely physical equality modulo a native period.

    Exact affine/periodic wrappers preserve continuity of a continuous lift.
    Periodic atlas jumps need a different chart lift and are refused.
    """
    pcurve, _, _ = intersection_curve_owner._pcurve_affine_source(segment.curve)
    if not isinstance(pcurve, intersection_curve_owner.IntersectionPCurve):
        return False
    curve = pcurve.curve
    lower, upper = segment.carrier_parameter_enclosure(
        0.0, 1.0, preparations=preparations, budget=budget
    )
    # Match the carrier API's supporting charts of this *restricted* use.
    # Root uncertainty may extend those charts, but cannot select a neighbor
    # outside the nominal trim support.
    support_head, support_tail = intersection_curve_owner.pcurve_chart_support(
        pcurve, segment.first, segment.last
    )
    first, last = sorted(
        Fraction(pcurve.first) + Fraction(pcurve.last) - Fraction(float(value))
        if pcurve.reversed else Fraction(float(value))
        for value in (lower, upper)
    )
    head = max(support_head, min(first.numerator // first.denominator, support_tail))
    # Include an upper-end node only when the restricted carrier API actually
    # selects its right-hand chart; otherwise the left chart owns that end.
    tail = max(head, min(last.numerator // last.denominator, support_tail))
    columns = slice(0, 2) if pcurve.side == "first" else slice(2, 4)
    if np.any(np.asarray(curve.transition_shifts)[head:tail, columns] != 0):
        raise SourceChartMapError("source_trim_uv_transition_discontinuous")
    _source_intersection_continuity(
        curve, _source_rational_bounds(first)[0], _source_rational_bounds(last)[1],
        support_head, support_tail, preparations, budget, verified,
    )
    # The caller next requires finite native order-one bounds over the whole
    # use. Together with these re-proved finite C0 joins they establish the
    # absolute continuity needed by interval mean-value/SPD arguments.
    return True


def _source_closed_edge_continuity(
    edge: _SourceClosedEdge,
    budget: CoordinateEnclosureBudget,
    verified: dict[int, intersection_curve_owner.IntersectionCurve],
) -> None:
    """Prove AC of the native edge with its fixed, root-defined affine ends.

    Endpoint root enclosures are not point samples or new parameter choices:
    the value and derivative APIs enclose the same fixed source atoms. The
    endpoint-affine closure correction preserves absolute continuity.
    """
    ends = edge._parameter_ends(budget)
    lower = min(float(value[0]) for value in ends)
    upper = max(float(value[1]) for value in ends)
    # Gap completion admits only exact identity placements. Check their actual
    # definition: AbstractCurve's default C1 method is not a wrapper theorem.
    curve = _source_identity_definition(edge.curve)
    if isinstance(curve, intersection_curve_owner.IntersectionCurve):
        _source_intersection_continuity(
            curve, lower, upper, 0, curve.num_charts - 1,
            edge.preparations, budget, verified,
        )
    elif not curve.is_c1_on(
        lower, upper, preparations=edge.preparations, budget=budget
    ):
        raise SourceChartMapError("source_gap_native_edge_continuity_unproved")
    # This query includes the complete root-expanded support, not nominal
    # binary endpoint representatives. No endpoint or seam samples suffice.
    jet = edge.derivative(Fraction(0), Fraction(1), budget)
    if not all(np.all(np.isfinite(value)) for value in jet):
        raise SourceChartMapError("source_gap_native_edge_jet_unsupported")


def _source_gap_allowance(
    domain: domain_owner.MeshingDomain, patch: int, loop: int, position: int
) -> float:
    """Use declared native rows, or the unchanged original import allowance."""
    authority = domain.brep_authority
    geometry = authority.geometry if isinstance(authority, model_owner.BRepModel) else authority
    if geometry is not None and geometry.edge_face_gap_bounds is not None:
        face = domain.source_indices[2][patch]
        coedge = geometry.face_loops[face][loop][position]
        allowance = float(geometry.edge_face_gap_bounds[coedge])
    else:
        allowance = (
            float(authority.report.curve_surface_tolerance)
            if isinstance(authority, model_owner.BRepModel) else 0.0
        )
    if not np.isfinite(allowance) or allowance < 0:
        raise SourceChartMapError("source_gap_original_tolerance_invalid")
    return allowance


def _root_spatial_carrier(
    root: intersection_curve_owner.RootEndpoint, /
) -> intersection_curve_owner.CurveEvaluator | intersection_curve_owner.IntersectionCurve | None:
    """Return the native carrier on which ``root``'s scalar atom is a parameter.

    A ``TrimRootEndpoint`` p-curve lift was proved at construction to trace its
    spatial root curve at the same parameter. Affine/periodic UV layers do not
    move the scalar, so such a lift keeps the unlifted spatial carrier. Lifts
    that reparameterize (trim segments, intersection p-curves) have none here.
    """
    if isinstance(root, TrimRootEndpoint) and root.source_pcurve is not None:
        lift = root.source_pcurve
        while isinstance(
            lift, (intersection_curve_owner.AffinePCurve, intersection_curve_owner.PeriodicPCurve)
        ):
            lift = (
                lift.curve
                if isinstance(lift, intersection_curve_owner.AffinePCurve)
                else lift.source_curve
            )
        if isinstance(lift, (CurveTrimSegment, intersection_curve_owner.IntersectionPCurve)):
            return None
        root = TrimRootEndpoint(
            root.root,
            root.operand,
            affine_parameter_offset=root.affine_parameter_offset,
            affine_parameter_scale=root.affine_parameter_scale,
            affine_transforms=root.affine_transforms,
        )
    return root.carrier


def _source_endpoint_gap_bounds(
    geometry: model_owner.BRepGeometry,
    points: np.ndarray,
    edge: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:
    """Prove missing endpoint rows from literal/root source atoms, never samples."""
    curve = geometry.curves[geometry.edge_curves[edge]]
    roots = geometry.edge_endpoint_roots[edge]
    ranges = np.asarray(geometry.edge_ranges)[edge]
    vertices = geometry.edge_vertices[edge]
    bounds: list[float] = []
    for end in (0, 1):
        budget.reserve(1, 512)
        parameter = float(ranges[end])
        root = roots[end]
        vertex = vertices[end]
        vertex_root = geometry.vertex_roots[vertex]
        literal = points[vertex]
        scalar: Fraction | PeriodOffset | None = None
        if root is None:
            scalar = Fraction(parameter)
            lower, upper = parameter, parameter
        else:
            if isinstance(root, NativePeriodEndpoint):
                scalar = _source_period_scalar(
                    root, curve, parameter, coedge=False,
                    preparations=preparations, budget=budget,
                )
            elif not tree_content_equal(_root_spatial_carrier(root), curve):
                raise SourceChartMapError("source_gap_endpoint_carrier_unproved")
            lower, upper = root.parameter_enclosure(preparations=preparations, budget=budget)
            if not lower <= parameter <= upper:
                raise SourceChartMapError("source_gap_endpoint_representative_unproved")
        if (
            vertex_root is not None and root is not None
            and not isinstance(root, NativePeriodEndpoint)
            and intersection_curve_owner._root_parameter_affine(root) == (Fraction(1), Fraction(0))
            and vertex_root.supports_endpoint(root, preparations=preparations, budget=budget)
        ):
            # Same native root with the identity scalar transport is an exact
            # incidence, independently of its binary execution realization.
            box = vertex_root.point_enclosure(preparations=preparations, budget=budget)
            if not np.all(np.isfinite(box)):
                raise SourceChartMapError("source_gap_vertex_root_unproved")
            bounds.append(0.0)
            continue
        exact_point = (
            None if scalar is None else intersection_curve_owner.exact_curve_parameter_point(
                curve, scalar, coordinate_budget=budget
            )
        )
        if exact_point is not None:
            if len(exact_point) != 3:
                raise SourceChartMapError("source_gap_endpoint_dimension")
            if vertex_root is None:
                squared = Fraction(0)
                for coordinate, target in zip(exact_point, literal, strict=True):
                    difference = coordinate - Fraction(float(target))
                    if not isinstance(difference, Fraction):
                        break
                    squared += difference * difference
                else:
                    bounds.append(model_owner.square_root_interval(squared)[1])
                    continue
            coordinates = [period_scalar_bounds(value) for value in exact_point]
            image = np.asarray(
                ([value[0] for value in coordinates], [value[1] for value in coordinates])
            )
        elif isinstance(curve, intersection_curve_owner.IntersectionCurve):
            image = curve.bounding_box(
                lower, upper, endpoint_roots=roots, preparations=preparations, budget=budget
            )
        else:
            image = curve.bounding_box(lower, upper, preparations=preparations, budget=budget)
        target = (
            np.stack((literal, literal)) if vertex_root is None else
            vertex_root.point_enclosure(preparations=preparations, budget=budget)
        )
        difference = interval_subtract((image[0], image[1]), (target[0], target[1]))
        bound = model_owner.norm_upper(np.maximum(np.abs(difference[0]), np.abs(difference[1])))
        if not np.isfinite(bound):
            raise SourceChartMapError("source_gap_endpoint_bound_unproved")
        bounds.append(bound)
    return bounds[0], bounds[1]


@overload
def _source_identity_definition(
    carrier: domain_owner.AbstractSurfacePatch,
) -> domain_owner.AbstractSurfacePatch: ...


@overload
def _source_identity_definition(
    carrier: AbstractCurve | intersection_curve_owner.IntersectionCurve,
) -> AbstractCurve | intersection_curve_owner.IntersectionCurve: ...


def _source_identity_definition(
    carrier: (
        domain_owner.AbstractSurfacePatch
        | AbstractCurve
        | intersection_curve_owner.IntersectionCurve
    ),
) -> (
    domain_owner.AbstractSurfacePatch
    | AbstractCurve
    | intersection_curve_owner.IntersectionCurve
):
    """Remove only an exactly identity spatial operation, never a nearby pose."""
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    while isinstance(carrier, (PlacedCurve, domain_owner.PlacedSurface)):
        if (
            not np.array_equal(np.asarray(carrier.rotation), identity)
            or not np.array_equal(np.asarray(carrier.translation), (0.0, 0.0, 0.0))
        ):
            raise SourceChartMapError("source_gap_nonidentity_placement_unsupported")
        carrier = carrier.definition
    return carrier


def _source_identity_occurrences(
    domain: domain_owner.MeshingDomain,
    authority: model_owner.BRepModel,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Bind every requested stratum to canonical qualified identity placement."""
    geometry = authority.geometry
    if geometry is None or domain.source_revision != authority.source_revision:
        raise SourceChartMapError("source_gap_occurrence_authority_mismatch")
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if any(
        occurrence.rotation != identity or occurrence.translation != (0.0, 0.0, 0.0)
        for occurrence in geometry.occurrences
    ):
        raise SourceChartMapError("source_gap_nonidentity_placement_unsupported")
    with budget.temporary_scope():
        storage = budget.live_object_storage_upper(
            (geometry, domain.source_indices, domain.source_occurrences)
        )
        budget.reserve(sum(len(row) for row in domain.source_indices), 16 * storage + 8192)
        records = geometry.qualified_entity_incidence(authority.source_revision)
        qualified = {
            (record.member.kind, record.member.index, record.member.occurrence_path)
            for record in records
        }
        for dimension, kind in enumerate(("vertex", "edge", "face")):
            for index, path in zip(
                domain.source_indices[dimension], domain.source_occurrences[dimension], strict=True
            ):
                if (kind, index, path) not in qualified:
                    raise SourceChartMapError("source_gap_occurrence_authority_mismatch")
        del qualified, records


def _prepare_source_gap_completion(
    domain: domain_owner.MeshingDomain,
    source_id: str,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    proved_use_bounds: dict[_UseKey, float],
    verified_intersections: dict[int, intersection_curve_owner.IntersectionCurve],
) -> _SourceGapCompletion | None:
    """Revalidate native gap authority before constructing any closed ribbon."""
    authority = domain.brep_authority
    geometry = authority.geometry if isinstance(authority, model_owner.BRepModel) else authority
    if geometry is None:
        return None
    if not isinstance(authority, model_owner.BRepModel):
        if domain_preparation_owner.has_brep_gaps(geometry):
            raise SourceChartMapError("source_gap_original_import_tolerance_unavailable")
        return None
    physical = domain.curve_atlas.mapping
    if not isinstance(physical, domain_owner.PhysicalCurveMap):
        raise SourceChartMapError("source_gap_native_curve_authority_unavailable")
    tolerance = float(authority.report.curve_surface_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise SourceChartMapError("source_gap_original_tolerance_invalid")
    from ..discretization._coordinate_operands import (
        _immutable_array_layout,
        seal_coordinate_operand_array,
    )

    # One current-source host read and one genuine immutable backing copy.
    # Row views share that backing; no per-vertex JAX gather or query-time sync.
    host_points = seal_coordinate_operand_array(
        np.asarray(geometry.vertex_points), budget=budget, copy=True,
    )
    domain_preparation_owner.certify_brep_gap_bounds(
        geometry, authority.patches, budget=budget
    )
    with budget.temporary_scope():
        budget.reserve(0, 128 * len(domain.curves) + 128 * len(proved_use_bounds))
        endpoint_bounds = tuple(
            _source_endpoint_gap_bounds(geometry, host_points, edge, preparations, budget)
            if geometry.edge_vertex_gap_bounds is None else geometry.edge_vertex_gap_bounds[edge]
            for edge in domain.source_indices[1]
        )
        # Each proved row was certified against its declared native row (or the
        # unchanged import allowance), so it never exceeds that declaration.
        use_bounds = dict(proved_use_bounds)
        if not any(any(row) for row in endpoint_bounds) and not any(use_bounds.values()):
            return None
        if any(max(row) > tolerance for row in endpoint_bounds):
            raise SourceChartMapError("source_gap_vertex_bound_exceeds_original_tolerance")
        if any(bound > tolerance for bound in use_bounds.values()):
            raise SourceChartMapError("source_gap_face_bound_exceeds_original_tolerance")
    _source_identity_occurrences(domain, authority, budget)
    budget.reserve(
        len(domain.curves) + domain.corner_count,
        1024 * len(domain.curves) + 512 * domain.corner_count,
    )
    identity = canonical_fingerprint({
        "source": source_id, "completion": "endpoint-affine-native-edges-and-homogeneous-face-ribbons",
        "original_curve_surface_tolerance": tolerance,
        "edge_vertex_bounds": endpoint_bounds,
        "edge_face_bounds": tuple(use_bounds.items()),
        "placement": "exact-identity-qualified-incidence",
    })
    budget.reserve(domain.corner_count, 1024 * domain.corner_count)
    authored = []
    for corner, source_vertex in enumerate(domain.source_indices[0]):
        point = host_points[source_vertex]
        budget.retain_owner(point)
        authored.append(_SourceAuthoredVertex(
            point,
            geometry.vertex_roots[source_vertex],
            canonical_fingerprint({
                "completion": identity, "authored_vertex": source_vertex,
                "occurrence": domain.source_occurrences[0][corner],
            }),
            preparations,
            {"layout": _AuthoredPointLayout(_immutable_array_layout(point))},
        ))
    vertices = tuple(authored)
    edges = []
    endpoint_ids = {}
    for index, source_edge in enumerate(domain.source_indices[1]):
        source_curve = geometry.curves[geometry.edge_curves[source_edge]]
        if not tree_content_equal(
            _source_identity_definition(source_curve),
            _source_identity_definition(physical.curves[index]),
        ):
            raise SourceChartMapError("source_gap_native_edge_authority_mismatch")
        source_range = np.asarray(geometry.edge_ranges)[source_edge]
        original_range = float(source_range[0]), float(source_range[1])
        owner_range = np.asarray(physical.ranges)[index]
        parameters = float(owner_range[0]), float(owner_range[1])
        forward = parameters == original_range
        if not forward and parameters[::-1] != original_range:
            raise SourceChartMapError("source_gap_native_edge_parameter_mismatch")
        source_vertices = geometry.edge_vertices[source_edge]
        roots = geometry.edge_endpoint_roots[source_edge]
        bounds = endpoint_bounds[index]
        if max(bounds) > tolerance:
            raise SourceChartMapError("source_gap_vertex_bound_exceeds_original_tolerance")
        if not forward:
            source_vertices = source_vertices[1], source_vertices[0]
            roots = roots[1], roots[0]
            bounds = bounds[1], bounds[0]
        curve_vertices = domain.curves[index].start, domain.curves[index].end
        if (
            tuple(domain.source_indices[0][vertex] for vertex in curve_vertices)
            != tuple(source_vertices)
            or not tree_content_equal(roots, physical.endpoint_roots[index])
        ):
            raise SourceChartMapError("source_gap_authored_vertex_binding_mismatch")
        endpoints = vertices[curve_vertices[0]], vertices[curve_vertices[1]]
        edge_id = canonical_fingerprint({
            "completion": identity, "authored_edge": source_edge,
            "occurrence": domain.source_occurrences[1][index],
            "endpoints": tuple(vertex.identity for vertex in endpoints),
            "native_range": parameters, "endpoint_bounds": bounds,
        })
        edges.append(_SourceClosedEdge(
            source_curve, parameters, roots, endpoints, bounds, edge_id, preparations
        ))
        _source_closed_edge_continuity(edges[-1], budget, verified_intersections)
        for end, vertex in enumerate(endpoints):
            endpoint_ids[index, end] = vertex.identity
    for patch, source in enumerate(domain.patches):
        face = domain.source_indices[2][patch]
        if not tree_content_equal(
            _source_identity_definition(source.surface),
            _source_identity_definition(authority.patches[face]),
        ):
            raise SourceChartMapError("source_gap_face_authority_mismatch")
        if len(source.loops) != len(geometry.face_loops[face]):
            raise SourceChartMapError("source_gap_face_loop_authority_mismatch")
        for loop, (members, coedges) in enumerate(zip(source.loops, geometry.face_loops[face], strict=True)):
            if len(members) != len(coedges):
                raise SourceChartMapError("source_gap_coedge_authority_mismatch")
            directed_endpoints = []
            for position, (use, coedge) in enumerate(zip(members, coedges, strict=True)):
                if (
                    not isinstance(use, domain_owner.PatchCurveUse)
                    or domain.source_indices[1][use.curve] != geometry.coedge_edges[coedge]
                    or not tree_content_equal(use.pcurve, geometry.pcurves[coedge])
                ):
                    raise SourceChartMapError("source_gap_coedge_authority_mismatch")
                first = 0 if use.first == edges[use.curve].parameters[0] else 1
                directed_endpoints.append((
                    edges[use.curve].vertices[first].identity,
                    edges[use.curve].vertices[1 - first].identity,
                ))
            if any(
                pair[1] != directed_endpoints[(index + 1) % len(directed_endpoints)][0]
                for index, pair in enumerate(directed_endpoints)
            ):
                raise SourceChartMapError("source_gap_oriented_corner_incidence_mismatch")
    return _SourceGapCompletion(tuple(edges), use_bounds, endpoint_ids, tolerance, identity)


def _source_period_scalar(
    endpoint: NativePeriodEndpoint,
    carrier: AbstractCurve | AbstractTrimCurve | intersection_curve_owner.IntersectionCurve,
    parameter: float,
    /,
    *,
    coedge: bool,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> Fraction | PeriodOffset:
    """Authenticate one carrier binding without confusing it with its scalar."""
    if coedge:
        if (
            isinstance(carrier, intersection_curve_owner.IntersectionCurve)
            or intersection_curve_owner._endpoint_source_transport(carrier, endpoint) is None
        ):
            raise SourceChartMapError("source_period_coedge_binding_unproved")
    else:
        native = carrier
        bound = endpoint.carrier
        # A spatial placement preserves its source's scalar parameter exactly.
        while isinstance(native, PlacedCurve):
            native = native.definition
        while isinstance(bound, PlacedCurve):
            bound = bound.definition
        if not tree_content_equal(native, bound):
            raise SourceChartMapError("source_period_native_binding_unproved")
    with budget.temporary_scope():
        budget.reserve(1, 512)
        authenticated = NativePeriodEndpoint(
            endpoint.curve, endpoint.patch, endpoint.axis,
            rational=endpoint.rational, turns=endpoint.turns,
            preparations=preparations, budget=budget,
        )
        lower, upper = authenticated.parameter_enclosure(
            preparations=preparations, budget=budget
        )
        if not lower <= parameter <= upper:
            raise SourceChartMapError("source_period_representative_outside_atom")
        scalar = authenticated.exact_parameter
        del authenticated
        return scalar


def _source_endpoint_parameters_equal(
    native_roots: tuple[
        intersection_curve_owner.RootEndpoint | None,
        intersection_curve_owner.RootEndpoint | None,
    ],
    coedge_roots: tuple[
        intersection_curve_owner.RootEndpoint | None,
        intersection_curve_owner.RootEndpoint | None,
    ],
    native_curve: AbstractCurve | intersection_curve_owner.IntersectionCurve,
    pcurve: AbstractCurve | AbstractTrimCurve,
    parameters: tuple[float, float],
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Equal Q+Q*2pi atoms may have different, separately proved carriers."""
    for native, coedge, parameter in zip(native_roots, coedge_roots, parameters, strict=True):
        if isinstance(native, NativePeriodEndpoint) or isinstance(coedge, NativePeriodEndpoint):
            if native is None:
                native_scalar = Fraction(parameter)
            elif isinstance(native, NativePeriodEndpoint):
                native_scalar = _source_period_scalar(
                    native, native_curve, parameter, coedge=False,
                    preparations=preparations, budget=budget,
                )
            else:
                return False
            if coedge is None:
                coedge_scalar = Fraction(parameter)
            elif isinstance(coedge, NativePeriodEndpoint):
                coedge_scalar = _source_period_scalar(
                    coedge, pcurve, parameter, coedge=True,
                    preparations=preparations, budget=budget,
                )
            else:
                return False
            if native_scalar != coedge_scalar:
                return False
        elif not (
            tree_content_equal(native, coedge)
            or (
                isinstance(native, TrimRootEndpoint)
                and isinstance(coedge, TrimRootEndpoint)
                and tree_content_equal(_trim_root_scalar(native), _trim_root_scalar(coedge))
            )
        ):
            return False
    return True


def _trim_root_scalar(endpoint: TrimRootEndpoint, /) -> tuple[object, ...]:
    """Return every field that ``TrimRootEndpoint`` reads to form its scalar atom.

    Outer ``PeriodicPCurve`` lifts select only the UV period sheet of the
    returned carrier; the scalar interval passes through them unchanged. The
    same root, side, affine chain and remaining lift therefore define the same
    exact scalar even when coedges sit on different seam sheets.
    """
    pcurve = endpoint.source_pcurve
    while isinstance(pcurve, intersection_curve_owner.PeriodicPCurve):
        pcurve = pcurve.source_curve
    return (
        endpoint.root,
        endpoint.operand,
        endpoint.affine_parameter_offset,
        endpoint.affine_parameter_scale,
        endpoint.affine_transforms,
        pcurve,
    )


def _source_native_uses(
    domain: domain_owner.MeshingDomain,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
    source_id: str,
    verified_intersections: dict[int, intersection_curve_owner.IntersectionCurve],
) -> tuple[
    str,
    dict[_UseKey, _SourceTrimUse],
    dict[_EndpointKey, str],
    dict[_UseKey, float],
]:
    """Authenticate continuous coedges and exact joins, not constructor samples."""
    physical = domain.curve_atlas.mapping
    trims: dict[_UseKey, _SourceTrimUse] = {}
    proved_use_bounds: dict[_UseKey, float] = {}
    endpoint_parent: dict[_EndpointKey, _EndpointKey] = {}

    def find(key: _EndpointKey) -> _EndpointKey:
        endpoint_parent.setdefault(key, key)
        while endpoint_parent[key] != key:
            key = endpoint_parent[key]
        return key

    def join(first: _EndpointKey, last: _EndpointKey) -> None:
        first, last = find(first), find(last)
        if first != last:
            endpoint_parent[max(first, last)] = min(first, last)

    for patch, source in enumerate(domain.patches):
        for loop_index, loop in enumerate(source.loops):
            members = []
            for position, use in enumerate(loop):
                budget.reserve(1, 512)
                if not isinstance(use, domain_owner.PatchCurveUse):
                    raise SourceChartMapError("source_chart_pole_singular")
                if use.trim_curve is not None:
                    carrier = use.trim_curve
                else:
                    forward_roots = (use.first_root, use.last_root)
                    if use.first > use.last:
                        forward_roots = forward_roots[::-1]
                    segment = CurveTrimSegment(
                        use.pcurve, min(use.first, use.last), max(use.first, use.last),
                        reversed=use.first > use.last,
                        first_root=forward_roots[0], last_root=forward_roots[1],
                        preparations=preparations, budget=budget,
                    )
                    carrier = constructors_owner.NormalizedTrimCurve(
                        segment, jnp.zeros(2, dtype=jnp.float64), jnp.ones(2, dtype=jnp.float64),
                        source.surface.definition
                        if isinstance(source.surface, domain_owner.PlacedSurface)
                        else source.surface,
                        use.start_vertex_root, use.end_vertex_root,
                        use.first_root, use.last_root,
                    )
                if not isinstance(carrier, constructors_owner.NormalizedTrimCurve):
                    raise SourceChartMapError("source_trim_authority_unsupported")
                segment = carrier.curve
                ordered_roots = (use.first_root, use.last_root)
                if use.first > use.last:
                    ordered_roots = ordered_roots[::-1]
                if (
                    not isinstance(segment, CurveTrimSegment)
                    or not tree_content_equal(segment.curve, use.pcurve)
                    or (segment.first, segment.last) != tuple(sorted((use.first, use.last)))
                    or segment.reversed != (use.first > use.last)
                    or not tree_content_equal(
                        (segment.first_root, segment.last_root), ordered_roots
                    )
                    or not tree_content_equal(
                        (carrier.start_root, carrier.end_root,
                         carrier.start_endpoint, carrier.end_endpoint),
                        (use.start_vertex_root, use.end_vertex_root,
                         use.first_root, use.last_root),
                    )
                    or np.any(np.asarray(carrier.lower) != 0)
                    or np.any(np.asarray(carrier.extent) != 1)
                    or not tree_content_equal(
                        carrier.patch,
                        source.surface.definition
                        if isinstance(source.surface, domain_owner.PlacedSurface)
                        else source.surface,
                    )
                ):
                    raise SourceChartMapError("source_trim_authority_mismatch")
                pcurve, _, _ = intersection_curve_owner._pcurve_affine_source(segment.curve)
                if isinstance(pcurve, intersection_curve_owner.IntersectionPCurve):
                    continuous = _source_piecewise_trim_continuity(
                        segment, preparations, budget, verified_intersections
                    )
                else:
                    continuous = carrier.is_c1_on(
                        0.0, 1.0, preparations=preparations, budget=budget
                    )
                if not continuous:
                    raise SourceChartMapError("source_trim_continuity_unproved")
                # These finite whole-use bounds are the removable-limit proof
                # prerequisites, even when a later query is exactly s=0.
                whole_box = carrier.enclosure(
                    0.0, 1.0, preparations=preparations, budget=budget
                )
                whole_jet = carrier.derivative_bounds(
                    0.0, 1.0, preparations=preparations, budget=budget
                )
                if not np.all(np.isfinite(whole_box)) or not all(
                    np.all(np.isfinite(value)) for value in whole_jet
                ):
                    raise SourceChartMapError("source_trim_derivative_singular")
                if isinstance(physical, domain_owner.PhysicalCurveMap):
                    range_row = np.asarray(physical.ranges)[use.curve]
                    native_range = float(range_row[0]), float(range_row[1])
                    forward = (use.first, use.last) == native_range
                    if not forward and (use.last, use.first) != native_range:
                        raise SourceChartMapError("source_native_parameter_mismatch")
                    native_roots = physical.endpoint_roots[use.curve]
                    roots = (use.first_root, use.last_root)
                    if not forward:
                        roots = roots[::-1]
                    curve = physical.curves[use.curve]
                    if not _source_endpoint_parameters_equal(
                        native_roots, roots, curve, use.pcurve, native_range,
                        preparations=preparations, budget=budget,
                    ):
                        raise SourceChartMapError("source_native_endpoint_root_mismatch")
                    low, high = sorted(native_range)
                    correspondence_roots = ordered_roots
                    if not tree_content_equal(roots, native_roots):
                        # Equal mathematical scalar atoms do not make their
                        # native/coedge carrier bindings interchangeable.
                        supports = [
                            (parameter, parameter) if root is None else root.parameter_enclosure(
                                preparations=preparations, budget=budget
                            )
                            for parameter, root in zip(native_range, native_roots, strict=True)
                        ]
                        low = min(support[0] for support in supports)
                        high = max(support[1] for support in supports)
                        if isinstance(curve, intersection_curve_owner.IntersectionCurve):
                            physical_roots = (
                                native_roots if native_range[0] < native_range[1]
                                else (native_roots[1], native_roots[0])
                            )
                            image = curve.bounding_box(
                                low, high, endpoint_roots=physical_roots,
                                preparations=preparations, budget=budget,
                            )
                        else:
                            image = curve.bounding_box(
                                low, high, preparations=preparations, budget=budget
                            )
                        if not np.all(np.isfinite(image)):
                            raise SourceChartMapError("source_native_endpoint_enclosure_unproved")
                        # The trim's whole_box above used ONLY coedge roots.
                        # Prove correspondence over this outward scalar support
                        # without handing either carrier the other's root.
                        correspondence_roots = None
                    # A native B-Rep domain's coedge uses carry its BRepPCurve lifts.
                    native_pcurve = cast(model_owner.BRepPCurve, use.pcurve)
                    squared = model_owner._edge_face_gap_squared(
                        curve, native_pcurve, source.surface, low, high, budget=budget
                    )
                    allowance = _source_gap_allowance(domain, patch, loop_index, position)
                    if squared is not None and squared > Fraction(allowance) ** 2:
                        raise SourceChartMapError("source_native_coedge_gap_exceeds_authority")
                    if squared is None:
                        box = whole_box
                        # Composed chart rows charge up to four ribbon/corner
                        # bounds against the SAME original allowance, so first
                        # seek a quarter-allowance certificate of the actual
                        # residual. The unchanged original target remains the
                        # fallback; neither attempt relaxes the allowance.
                        witness = None
                        for target in (allowance / 4.0, allowance) if allowance > 0.0 else (0.0,):
                            witness = correspondence_owner.certify_curve_surface(
                                curve, native_pcurve, source.surface, box, low, high,
                                point=np.zeros(3), tolerance=target,
                                endpoint_roots=correspondence_roots,
                                preparations=preparations, budget=budget,
                            )
                            if witness.complete and witness.deviation_bound <= target:
                                break
                        if (
                            witness is None
                            or not witness.complete
                            or witness.deviation_bound > allowance
                        ):
                            raise SourceChartMapError("source_native_coedge_identity_unproved")
                        proved_use_bounds[patch, loop_index, position] = witness.deviation_bound
                    else:
                        proved_use_bounds[patch, loop_index, position] = model_owner.square_root_interval(squared)[1]
                else:
                    owner_patch, owner_loop, owner_position = domain.curve_owners[use.curve]
                    owner_source = domain.patches[owner_patch]
                    owner = cast(
                        domain_owner.PatchCurveUse,
                        owner_source.loops[owner_loop][owner_position],
                    )
                    native_range = (owner.first, owner.last)
                    forward = (use.first, use.last) == native_range
                    if (
                        (not forward and (use.last, use.first) != native_range)
                        or not tree_content_equal(
                            _source_use_signature(source, use, budget),
                            _source_use_signature(owner_source, owner, budget),
                        )
                        or any(root is not None for root in (use.first_root, use.last_root))
                    ):
                        raise SourceChartMapError("source_coedge_identity_unproved")
                ends = ((use.curve, 0), (use.curve, 1))
                if not forward:
                    ends = ends[::-1]
                for end in ends:
                    find(end)
                trims[patch, loop_index, position] = (carrier, native_range, ends)
                members.append((carrier, ends))
            for index, (carrier, ends) in enumerate(members):
                following, next_ends = members[(index + 1) % len(members)]
                if not carrier.shares_endpoint(following, preparations=preparations, budget=budget):
                    raise SourceChartMapError("source_native_trim_join_unproved")
                join(ends[1], next_ends[0])
    endpoint_ids = {
        key: canonical_fingerprint({"source": source_id, "native_endpoint_component": find(key)})
        for key in endpoint_parent
    }
    return source_id, trims, endpoint_ids, proved_use_bounds


@dataclass(frozen=True, slots=True)
class SourceChartPreparations:
    """One request's native coedge proof, bound to all source fields and budget."""

    domain: domain_owner.MeshingDomain
    domain_identity: str
    native: TrimRequestPreparations
    source_id: str
    uses: dict
    endpoint_ids: dict
    budget: CoordinateEnclosureBudget
    completion: _SourceGapCompletion | None


def require_source_chart_point_layouts(
    source: SourceChartMap, budget: CoordinateEnclosureBudget, /
) -> None:
    """Authenticate private view metadata before any memoized source callback."""
    if not source.gap_arcs and all(point is None for point in source.gap_vertices):
        return
    with budget.temporary_scope():
        budget.reserve(0, 256 + 128 * (len(source.gap_vertices) + len(source.gap_arcs)))
        edges = (
            *(point.edge for point in source.gap_vertices if point is not None),
            *(arc.edge for _, _, arc in source.gap_arcs),
        )
        seen = set()
        for edge in edges:
            for vertex in edge.vertices:
                if id(vertex) not in seen:
                    budget.reserve(1, 128)
                    vertex.require_point_layout()
                    seen.add(id(vertex))


@dataclass(slots=True, weakref_slot=True)
class _SourceChartCreation:
    """Request-owned authority for one genuinely prepared, unchanged map."""
    source: SourceChartMap
    definition: str
    revoked: bool = False


_ISSUED_SOURCE_CHARTS: IssuedProofs[
    CoordinateEnclosureBudget, _SourceChartCreation
] = IssuedProofs()

# One proof-local definition record: its dict slot and key, entry pair and
# hexadecimal digest string.
_SOURCE_DEFINITION_BYTES = 512
# A bound receipt: its three-slot object and two reference-tuple headers, plus
# one pointer per bank summary or nested receipt it references.
_BOUND_SOURCE_BYTES = 256
_BOUND_SOURCE_REFERENCE_BYTES = 8


@dataclass(frozen=True, slots=True)
class _BoundSourceDefinition:
    """A proof-local definition digest that depends on byte-backed host banks.

    The values of an immutable byte-backed bank cannot change, but its NumPy
    dtype, shape and strides remain assignable in place. The receipt retains
    distinct bank admissions, in first-occurrence order, with their earned
    layouts. Shared definition DAGs therefore validate each dependency once
    per reuse, without a byte scan or a cache across identity checks.
    """

    owner: object
    banks: tuple[CarrierAdmission, ...]

    def current(self) -> bool:
        return all(carrier_admission_current(bank) for bank in self.banks)


def source_chart_enclosure_identity(
    source: SourceChartMap,
    budget: CoordinateEnclosureBudget,
    carriers: dict[int, tuple[object, str]],
    /,
    *,
    admissions: dict[int, CarrierAdmission] | None = None,
    retain_admission: Callable[[int], None] | None = None,
    retain_definition: Callable[[int], None] | None = None,
) -> str | None:
    if (
        type(source) is not SourceChartMap or source.enclosure_authority is None
        or source.enclosure_authority[2] != id(budget)
    ):
        return None
    creation = _ISSUED_SOURCE_CHARTS.get(budget, str(id(source)))
    if creation is None or creation.source is not source or creation.revoked:
        raise SourceChartMapError("source_chart_requires_genuine_reprepare")
    try:
        current = _source_chart_current_identity(
            source, budget, carriers, admissions=admissions,
            retain_admission=retain_admission, retain_definition=retain_definition,
        )
    except SourceChartMapError:
        # A changed bank layout or authored point revokes this preparation;
        # its derived enclosures must never be relabelled as current.
        creation.revoked = True
        raise
    if current != creation.definition:
        # Derived gap/root/continuity caches belong to the original source.
        # A new digest must never relabel their old enclosures as current.
        creation.revoked = True
        raise SourceChartMapError("source_chart_definition_changed_requires_reprepare")
    return current


def _source_chart_current_identity(
    source: SourceChartMap,
    budget: CoordinateEnclosureBudget,
    carriers: dict[int, tuple[object, str]],
    /,
    *,
    admissions: dict[int, CarrierAdmission] | None = None,
    retain_admission: Callable[[int], None] | None = None,
    retain_definition: Callable[[int], None] | None = None,
) -> str | None:
    """Bind reusable native enclosures to actual source state, not asserted IDs.

    Query preparations and derived memo dictionaries are not map definitions.
    Every endpoint, trim, fixed root, placement, physical correction and native
    coefficient tree is included. Carrier and immutable definition-container
    digests are shared only within one proof and recomputed for the next proof,
    so geometry mutations miss. A digest that depends on byte-backed host banks
    is reused only through its ``_BoundSourceDefinition`` receipt, whose bank
    layouts are rechecked first; a changed layout discards every bound receipt
    of the proof and refuses, so the caller must reprepare. The receipt's bank
    summaries live in ``admissions``; ``retain_admission`` and
    ``retain_definition`` receive the bytes of each retained summary and
    definition record before it is created. Without them, the caller owns the
    storage of both dictionaries. No candidate or scientific verdict is
    retained.
    """
    if (
        type(source) is not SourceChartMap
        or source.enclosure_authority is None
        or source.enclosure_authority[2] != id(budget)
    ):
        return None
    require_source_chart_point_layouts(source, budget)
    if admissions is None:
        admissions = {}

    def remember(
        value: object,
        identity: str,
        immutable: bool,
        banks: list[CarrierAdmission] | tuple[()],
        parts: tuple[_BoundSourceDefinition, ...],
    ) -> tuple[str, bool, _BoundSourceDefinition | None]:
        if not immutable:
            return identity, False, None
        if not banks and not parts:
            if retain_definition is not None:
                retain_definition(_SOURCE_DEFINITION_BYTES)
            carriers[id(value)] = (value, identity)
            return identity, True, None
        with budget.temporary_scope():
            count = sum(
                1 if bank.layout is not None else len(bank.banks)
                for bank in banks
            ) + sum(len(part.banks) for part in parts)
            budget.reserve(count, 512 + count * 256)
            unique: dict[int, CarrierAdmission] = {}
            for bank in banks:
                for leaf in (bank,) if bank.layout is not None else bank.banks:
                    unique.setdefault(id(leaf), leaf)
            for part in parts:
                for leaf in part.banks:
                    unique.setdefault(id(leaf), leaf)
            size = _SOURCE_DEFINITION_BYTES
            if unique:
                size += _BOUND_SOURCE_BYTES + _BOUND_SOURCE_REFERENCE_BYTES * len(unique)
            if retain_definition is not None:
                retain_definition(size)
            bound = _BoundSourceDefinition(value, tuple(unique.values())) if unique else None
            del unique
            carriers[id(value)] = (value if bound is None else bound, identity)
            return identity, True, bound

    def payload(
        value: object, *, private_host: bool = False
    ) -> tuple[object, bool, _BoundSourceDefinition | None]:
        budget.reserve(1, 512)
        structured = (
            isinstance(value, (StrictModule, tuple))
            or (is_dataclass(value) and not isinstance(value, type))
        )
        if structured:
            previous = carriers.get(id(value))
            if previous is not None:
                held = previous[0]
                if held is value:
                    return previous[1], True, None
                if type(held) is _BoundSourceDefinition and held.owner is value:
                    if held.current():
                        return previous[1], True, held
                    # A bank's dtype, shape or strides changed in place. Other
                    # receipts may share it: none of them remains authority.
                    for key in [
                        key for key, (owner, _) in carriers.items()
                        if type(owner) is _BoundSourceDefinition
                    ]:
                        del carriers[key]
                    raise SourceChartMapError(
                        "source_chart_definition_changed_requires_reprepare"
                    )
        if isinstance(value, StrictModule):
            banks: list[CarrierAdmission] = []
            with budget.temporary_scope():
                immutable = admit_carrier_fingerprint(
                    value, budget, memo=admissions, retain_memo=retain_admission,
                    banks=banks,
                )
                identity = canonical_fingerprint(carrier_payload(value))
            return remember(value, identity, immutable, banks, ())
        if isinstance(value, Fraction):
            return _source_atom(value), True, None
        if private_host:
            # Only genuine pre-proof private point-bank rows are eligible.
            # Their immutable bytes and full view layout were checked above.
            # Other public NumPy leaves still require fresh authentication.
            return value, True, None
        if is_dataclass(value) and not isinstance(value, type):
            members = tuple(
                (item.name, payload(
                    getattr(value, item.name),
                    private_host=type(value) is _SourceAuthoredVertex and item.name == "point",
                ))
                for item in fields(value)
                if item.name not in ("preparations", "memo", "points", "enclosure_authority")
            )
            definition = (
                type(value).__name__,
                tuple((name, child[0]) for name, child in members),
            )
            immutable = (
                getattr(type(value), "__dataclass_params__").frozen
                and all(child[1] for _, child in members)
            )
            parts = tuple(child[2] for _, child in members if child[2] is not None)
        elif isinstance(value, tuple):
            children = tuple(payload(item) for item in value)
            definition = tuple(child[0] for child in children)
            immutable = all(child[1] for child in children)
            parts = tuple(child[2] for child in children if child[2] is not None)
        else:
            return value, type(value) in (
                type(None), bool, int, float, complex, str, bytes
            ) or isinstance(value, Array), None
        return remember(value, canonical_fingerprint(definition), immutable, (), parts)

    with budget.temporary_scope():
        definition, _, _ = payload(source)
        return canonical_fingerprint({
            "map": source.map_id,
            "authority": source.enclosure_authority[:2],
            "definition": definition,
        })


def _source_chart_domain_identity(
    domain: domain_owner.MeshingDomain,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    expected: domain_owner.MeshingDomain | None = None,
) -> str:
    """Pre-admit identity workspace, retaining only the resulting digest."""
    with budget.temporary_scope():
        source_bytes = budget.live_object_storage_upper(
            domain if expected is None else (domain, expected)
        )
        # Simultaneous tree/array-path records, escaped static metadata and
        # canonical JSON/UTF-8 buffers. This is a byte bound, not hash work.
        budget.reserve(0, 16 * source_bytes + 8192)
        if expected is not None and not tree_content_equal(domain, expected):
            raise SourceChartMapError("source_chart_preparation_authority_mismatch")
        return canonical_fingerprint(carrier_payload(domain))


def prepare_source_chart_domain(
    domain: domain_owner.MeshingDomain, /, *, budget: CoordinateEnclosureBudget
) -> SourceChartPreparations:
    """Authenticate the native domain once before preparing its chart patches."""
    try:
        budget.reserve(1, 512)
        # The canonical carrier owner includes family AND static tree metadata,
        # unlike an array-only digest (which cannot distinguish scalar roots).
        identity = _source_chart_domain_identity(domain, budget)
        native = TrimRequestPreparations()
        verified_intersections: dict[int, intersection_curve_owner.IntersectionCurve] = {}
        source_id, uses, endpoint_ids, proved_use_bounds = _source_native_uses(
            domain, budget, native, identity, verified_intersections
        )
        completion = _prepare_source_gap_completion(
            domain, identity, native, budget, proved_use_bounds, verified_intersections
        )
        if completion is not None:
            source_id, endpoint_ids = completion.identity, completion.endpoint_ids
        return SourceChartPreparations(
            domain, identity, native, source_id, uses, endpoint_ids, budget, completion
        )
    except (CoordinateEnclosureResourceError, SourceChartMapError):
        raise
    except (TypeError, ValueError, NotImplementedError) as error:
        raise SourceChartMapError(f"source_chart_native_unsupported:{error}") from error


def _prepare_source_chart_maps(
    domain: domain_owner.MeshingDomain,
    chart_data: _ChartTriangulation,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    prepared: SourceChartPreparations,
    identity_memo: CoordinateLiveStorage,
) -> tuple[SourceChartMap, ...]:
    """Prepare one endpoint-corrected native map per original chart cell.

    The caller first proves the full chain with boundary_chart_cover. This
    independent preparation proves native authority, exact joins and rational
    restrictions; it never interprets a generation flag as a geometric proof.
    """
    patch, charts, points, cells, boundary, provenance, required, vertices, edges, parameters = chart_data
    patch = int(patch)
    preparations = prepared.native
    source_id, uses, endpoint_ids = prepared.source_id, prepared.uses, prepared.endpoint_ids
    completion = prepared.completion
    charts, cells, boundary, provenance = (
        np.asarray(value) for value in (charts, cells, boundary, provenance)
    )
    budget.reserve(len(charts) + len(cells) + len(boundary) + len(vertices),
                   512 * len(charts) + 1024 * len(cells) + 512 * len(boundary))
    if completion is not None:
        budget.reserve(3 * len(boundary), 1024 * len(boundary) + 128 * len(charts))
    try:
        with budget.measurement():
            exact, restrictions, _ = validate_chart_restrictions(
                charts, required, vertices, edges, charts[np.asarray(edges)],
                parameters, domain.source_revision, patch,
            )
    except CoordinateEnclosureResourceError:
        raise
    except (TypeError, ValueError) as error:
        raise SourceChartMapError("source_chart_restriction_authority") from error
    chain_id = canonical_fingerprint({
        "source": source_id, "patch": patch, "charts": charts, "cells": cells,
        "boundary": boundary, "provenance": provenance,
        "restrictions": tuple(value.restriction_id for value in restrictions),
    })
    # A closed single-use loop meets its seam vertex at both use ends, so one
    # chart vertex legitimately carries both normalized parameters 0 and 1.
    # Each boundary edge therefore reads its endpoints from its own record.
    recorded: dict[tuple[int, int, int], set[Fraction]] = {}
    for edge, record in zip(boundary, provenance, strict=True):
        loop, position = (int(x) for x in record[:2])
        # Prepared native uses refused every pole use (source_chart_pole_singular).
        use = cast(domain_owner.PatchCurveUse, domain.patches[patch].loops[loop][position])
        for vertex, parameter in zip(edge, record[2:], strict=True):
            recorded.setdefault((loop, position, int(vertex)), set()).add(
                (Fraction(float(parameter)) - Fraction(use.first))
                / (Fraction(use.last) - Fraction(use.first))
            )
    if any(
        len(values) > 2 or (len(values) == 2 and values != {Fraction(0), Fraction(1)})
        for values in recorded.values()
    ):
        raise SourceChartMapError("source_chart_trim_partition_invalid")
    # A rational restriction denotes its actual source subdivision, not the
    # rounded scalar written to the execution/provenance bank.
    use_keys = {(int(record[0]), int(record[1])) for record in provenance}
    restricted: dict[tuple[int, int, int], Fraction] = {}
    for vertex, edge, parameter in zip(vertices, edges, parameters, strict=True):
        fraction = Fraction(int(parameter[0]), int(parameter[1]))
        for loop, position in use_keys:
            first_key, last_key = (loop, position, int(edge[0])), (loop, position, int(edge[1]))
            first_values = (
                {restricted[first_key]} if first_key in restricted else recorded.get(first_key)
            )
            last_values = (
                {restricted[last_key]} if last_key in restricted else recorded.get(last_key)
            )
            if not first_values or not last_values:
                continue
            candidates = {
                (first_value, last_value)
                for first_value in first_values
                for last_value in last_values
            }
            if len(candidates) > 1:
                # Only a seam parent is two-valued. Its rounded child row selects
                # the parent interval by order, which rounding preserves; a row
                # equal to the other parent cannot identify a side and refuses.
                rounded = recorded.get((loop, position, int(vertex)), set())
                candidates = {
                    pair for pair in candidates
                    if len(rounded) == 1 and min(pair) < next(iter(rounded)) < max(pair)
                }
                if len(candidates) != 1:
                    raise SourceChartMapError("source_chart_restriction_seam_side_unproved")
            first_value, last_value = next(iter(candidates))
            restricted[loop, position, int(vertex)] = (
                (1 - fraction) * first_value + fraction * last_value
            )

    def source_parameter(loop: int, position: int, vertex: int, value: float) -> Fraction:
        key = loop, position, vertex
        if key in restricted:
            return restricted[key]
        use = cast(domain_owner.PatchCurveUse, domain.patches[patch].loops[loop][position])
        return (Fraction(float(value)) - Fraction(use.first)) / (
            Fraction(use.last) - Fraction(use.first)
        )

    prepared_vertices = [
        _SourceChartVertex(
            canonical_fingerprint({"source": source_id, "patch": patch,
                                   "uv": tuple(_source_atom(x) for x in row)}),
            exact=(row[0], row[1]),
        )
        for row in exact
    ]
    boundary_arcs = {}
    boundary_gap_arcs = {}
    gap_points = [None] * len(charts)
    gap_point_bounds = [0.0] * len(charts)
    assigned = {}
    assigned_uv = {}
    for edge, record in zip(boundary, provenance, strict=True):
        first, last = (int(x) for x in edge)
        loop, position = (int(x) for x in record[:2])
        use = cast(domain_owner.PatchCurveUse, domain.patches[patch].loops[loop][position])
        carrier, native_range, ends = uses[patch, loop, position]
        use_start, use_finish = Fraction(use.first), Fraction(use.last)
        a = source_parameter(loop, position, first, record[2])
        b = source_parameter(loop, position, last, record[3])
        if not 0 <= a < b <= 1:
            raise SourceChartMapError("source_chart_trim_partition_invalid")
        native_first, native_last = Fraction(float(native_range[0])), Fraction(float(native_range[1]))
        native_parameters = (
            (use_start + a * (use_finish - use_start) - native_first) / (native_last - native_first),
            (use_start + b * (use_finish - use_start) - native_first) / (native_last - native_first),
        )
        arc_id = canonical_fingerprint({
            "source": source_id, "curve": use.curve,
            "parameters": tuple(_source_atom(x) for x in sorted(native_parameters)),
        })
        affine = (
            isinstance(use.pcurve, AbstractCurve)
            and intersection_curve_owner.affine_curve_coefficients(use.pcurve, np.zeros(2))
            is not None
        )
        direction = 1 if native_parameters[1] > native_parameters[0] else -1
        arc = _SourceTrimArc(
            carrier, a, b, arc_id, preparations, direction, affine,
            mean_value_certified=True,
        )
        boundary_arcs[first, last] = arc
        gap_arc = None
        if completion is not None:
            closed_edge = completion.edges[use.curve]
            distance = float(_source_rational_interval(
                Fraction(completion.use_bounds[patch, loop, position])
                + Fraction(max(closed_edge.endpoint_bounds))
            )[1])
            gap_arc = _SourceGapArc(
                arc, closed_edge, native_parameters[0], native_parameters[1],
                domain.patches[patch].surface, distance,
            )
            uv_box = np.stack(arc.bounds(Fraction(0), Fraction(1), budget))
            if not gap_arc.surface.is_c1_on(
                uv_box, preparations=preparations, budget=budget
            ):
                raise SourceChartMapError("source_gap_surface_continuity_unproved")
            # Established AC of both traces plus finite whole-arc first jets
            # proves the physical s=0 removable limit, even for point queries.
            gap_arc.residuals(_WHOLE_ARC, {}, budget, value=False, jacobian=True)
            boundary_gap_arcs[first, last] = gap_arc
        for vertex, normalized, native_parameter, end_parameter in (
            (first, a, native_parameters[0], Fraction(0)),
            (last, b, native_parameters[1], Fraction(1)),
        ):
            identity = (
                endpoint_ids[ends[0]] if normalized == 0 else
                endpoint_ids[ends[1]] if normalized == 1 else
                canonical_fingerprint({"source": source_id, "curve": use.curve,
                                       "parameter": _source_atom(native_parameter)})
            )
            if vertex in assigned_uv:
                previous_loop, previous_position, previous_parameter = assigned_uv[vertex]
                loop_size = len(domain.patches[patch].loops[loop])
                same_parameter = previous_position == position and previous_parameter == normalized
                following_join = (
                    previous_parameter == 1 and normalized == 0
                    and (previous_position + 1) % loop_size == position
                )
                preceding_join = (
                    normalized == 1 and previous_parameter == 0
                    and (position + 1) % loop_size == previous_position
                )
                # Only the exact UV joins authenticated above may share an
                # affine corner. A common 3D root does not identify UV sheets.
                if previous_loop != loop or not (
                    same_parameter or following_join or preceding_join
                ):
                    raise SourceChartMapError("source_chart_uv_endpoint_identity_unproved")
            assigned_uv[vertex] = (loop, position, normalized)
            if vertex in assigned and assigned[vertex] != identity:
                raise SourceChartMapError("source_chart_endpoint_identity_mismatch")
            assigned[vertex] = identity
            if gap_arc is not None:
                gap_point_bounds[vertex] = (
                    gap_arc.distance_bound if gap_points[vertex] is None else
                    min(gap_point_bounds[vertex], gap_arc.distance_bound)
                )
                gap_points[vertex] = _SourceClosedEdgePoint(gap_arc.edge, native_parameter)
            coordinate: tuple[Fraction | PeriodOffset, Fraction | PeriodOffset] | None = None
            if use.first_root is None and use.last_root is None:
                point = intersection_curve_owner.exact_curve_parameter_point(
                    use.pcurve, use_start + normalized * (use_finish - use_start),
                    coordinate_budget=budget,
                )
                if point is not None:
                    if len(point) != 2:
                        raise SourceChartMapError("source_chart_endpoint_dimension")
                    coordinate = point[0], point[1]
            prepared_vertices[vertex] = (
                _SourceChartVertex(identity, exact=coordinate)
                if coordinate is not None else
                _SourceChartVertex(identity, arc=arc, parameter=end_parameter)
            )
    # Propagate global endpoint corrections through exact rational restrictions.
    # Boundary vertices instead retain their exact native trim parameters.
    for vertex, edge, parameter in zip(vertices, edges, parameters, strict=True):
        vertex = int(vertex)
        if vertex in assigned:
            continue
        fraction = Fraction(int(parameter[0]), int(parameter[1]))
        parents = prepared_vertices[int(edge[0])], prepared_vertices[int(edge[1])]
        identity = canonical_fingerprint({
            "source": source_id, "patch": patch,
            "restriction_parents": tuple(parent.identity for parent in parents),
            "parameter": _source_atom(fraction),
        })
        first_exact, last_exact = parents[0].exact, parents[1].exact
        if first_exact is not None and last_exact is not None:
            coordinate = (
                first_exact[0] * (1 - fraction) + last_exact[0] * fraction,
                first_exact[1] * (1 - fraction) + last_exact[1] * fraction,
            )
            prepared_vertices[vertex] = _SourceChartVertex(identity, exact=coordinate)
        else:
            prepared_vertices[vertex] = _SourceChartVertex(identity, parameter=fraction, parents=parents)
    result = []
    creation_carriers: dict[int, tuple[object, str]] = {}
    creation_admissions: dict[int, CarrierAdmission] = {}

    def retain_creation_memo(size: int) -> None:
        # Definition records, receipts and admission summaries shared by this
        # preparation's maps stay live until all are issued; each is admitted
        # before it is created.
        identity_memo.set_bound(identity_memo.bound + size)

    for row, cell in enumerate(cells):
        indices = int(cell[0]), int(cell[1]), int(cell[2])
        endpoints = (
            prepared_vertices[indices[0]],
            prepared_vertices[indices[1]],
            prepared_vertices[indices[2]],
        )
        arcs, traces = [], []
        physical_arcs = []
        physical_points = tuple(gap_points[index] for index in indices)
        physical_bounds = tuple(gap_point_bounds[index] for index in indices)
        for first, last in ((0, 1), (1, 2), (2, 0)):
            pair = indices[first], indices[last]
            arc = boundary_arcs.get(pair)
            if arc is None and pair[::-1] in boundary_arcs:
                original = boundary_arcs[pair[::-1]]
                arc = _SourceTrimArc(
                    original.carrier, original.last, original.first,
                    original.identity, preparations, -original.direction, original.affine,
                    mean_value_certified=original.mean_value_certified,
                )
            if arc is not None:
                arcs.append((first, last, arc))
                physical_arc = boundary_gap_arcs.get(pair)
                if physical_arc is None and pair[::-1] in boundary_gap_arcs:
                    original_gap = boundary_gap_arcs[pair[::-1]]
                    physical_arc = _SourceGapArc(
                        arc, original_gap.edge, original_gap.last, original_gap.first,
                        original_gap.surface, original_gap.distance_bound,
                    )
                if physical_arc is not None:
                    physical_arcs.append((first, last, physical_arc))
            for i, j in ((first, last), (last, first)):
                identity = canonical_fingerprint({
                    "source": source_id,
                    "trace": (
                        "native" if completion is None else "completed-native-edge", arc.identity,
                        arc.direction if (i, j) == (first, last) else -arc.direction,
                    ) if arc is not None else (
                        "affine", chain_id, indices[i], indices[j]
                    ),
                    "endpoints": (endpoints[i].identity, endpoints[j].identity),
                })
                traces.append((i, j, identity))
        # With s_e=lambda_i+lambda_j and the chord removed from each ribbon,
        # G-F(U) = sum_v c_v d_v + sum_e s_e^2 R_e(t_e) exactly, where
        # c_v = lambda_v (1 - sum_{boundary e containing v} s_e), |d_v| <= M_v
        # and |R_e| <= D_e. No boundary edge gives the convex sum of d_v. One
        # edge (i,j) opposite k gives c_i=lambda_i lambda_k, c_j=lambda_j lambda_k,
        # c_k=lambda_k and s^2: nonnegative weights summing to one. A vertex on
        # two boundary edges has c_v=-lambda_v^2; bounding each remaining
        # coefficient by its simplex maximum keeps every case rigorous.
        vertex_bounds = [Fraction(value) for value in physical_bounds]
        ribbon_bounds = [
            (first, last, Fraction(physical_arc.distance_bound))
            for first, last, physical_arc in physical_arcs
        ]
        if len(ribbon_bounds) <= 1:
            face_displacement = max(
                (*vertex_bounds, *(bound for _, _, bound in ribbon_bounds))
            )
        else:
            face_displacement = sum(
                (bound for _, _, bound in ribbon_bounds), Fraction(0)
            )
            if len(ribbon_bounds) == 2:
                incidence = [
                    sum(vertex in (first, last) for first, last, _ in ribbon_bounds)
                    for vertex in range(3)
                ]
                corner = incidence.index(2)
                # |c_corner| <= 1 and the other two coefficients are
                # lambda_j lambda_k <= 1/4; each s_e^2 <= 1.
                face_displacement += vertex_bounds[corner] + sum(
                    (vertex_bounds[vertex] for vertex in range(3) if vertex != corner),
                    Fraction(0),
                ) / 4
            else:
                # Every c_v=-lambda_v^2 and sum lambda_v^2 <= 1.
                face_displacement += max(vertex_bounds)
        # Original lower-dimensional source edges must remain covered even
        # when endpoint-affine corrections cancel the face displacement.
        edge_displacement = max(
            (max(point.edge.endpoint_bounds) for point in physical_points if point is not None),
            default=0.0,
        )
        authored_bound = float(_source_rational_interval(
            max(face_displacement, Fraction(edge_displacement))
        )[1])
        if completion is not None and authored_bound > completion.tolerance:
            raise SourceChartMapError("source_gap_completion_exceeds_original_tolerance")
        map_id = canonical_fingerprint({
            "source": source_id, "patch": patch, "cell": row,
            "chain": chain_id,
            "exact_chart": tuple(tuple(_source_atom(x) for x in exact[index]) for index in indices),
            "endpoints": tuple(point.identity for point in endpoints),
            "arcs": tuple((i, j, arc.identity) for i, j, arc in arcs),
            "restrictions": tuple(value.restriction_id for value in restrictions),
            "authored_deviation_bound": authored_bound,
            "completion": None if completion is None else completion.identity,
            "physical_vertices": tuple(
                None if point is None else (point.edge.identity, _source_atom(point.parameter))
                for point in physical_points
            ),
            "physical_ribbons": tuple(
                (i, j, arc.edge.identity, _source_atom(arc.first), _source_atom(arc.last), arc.distance_bound)
                for i, j, arc in physical_arcs
            ),
        })
        source_map = SourceChartMap(
            domain.patches[patch].surface, patch, row, indices,
            endpoints, tuple(arcs), tuple(traces), map_id, preparations,
            authored_bound, physical_points, physical_bounds, tuple(physical_arcs),
            enclosure_authority=(prepared.domain_identity, prepared.source_id, id(budget)),
        )
        definition = _source_chart_current_identity(
            source_map, budget, creation_carriers,
            admissions=creation_admissions,
            retain_admission=retain_creation_memo,
            retain_definition=retain_creation_memo,
        )
        if definition is None:
            raise SourceChartMapError("source_chart_requires_genuine_reprepare")
        budget.reserve(1, 1024)
        creation = _SourceChartCreation(source_map, definition)
        budget.retain_owner(creation)
        _ISSUED_SOURCE_CHARTS.record(budget, str(id(source_map)), creation)
        result.append(source_map)
    return tuple(result)


def prepare_source_chart_maps(
    domain: domain_owner.MeshingDomain,
    chart_data: _ChartTriangulation,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: SourceChartPreparations | None = None,
) -> tuple[SourceChartMap, ...]:
    """Return native source maps in unchanged original chart-cell vertex order.

    A supplied request preparation reuses all native coedge/root queries. Its
    complete source binding is checked independently, including static fields,
    so a replaced or mutated source cannot borrow another domain's proof.
    """
    try:
        if preparations is None:
            preparations = prepare_source_chart_domain(domain, budget=budget)
        elif (
            preparations.budget is not budget
            or _source_chart_domain_identity(
                domain, budget, expected=preparations.domain
            ) != preparations.domain_identity
        ):
            raise SourceChartMapError("source_chart_preparation_authority_mismatch")
        # The creation identity memo lives only while these maps are issued.
        with budget.live_storage() as identity_memo:
            return _prepare_source_chart_maps(
                domain, chart_data, budget=budget, prepared=preparations,
                identity_memo=identity_memo,
            )
    except (CoordinateEnclosureResourceError, SourceChartMapError):
        raise
    except (TypeError, ValueError, NotImplementedError) as error:
        raise SourceChartMapError(f"source_chart_native_unsupported:{error}") from error
