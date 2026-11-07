#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Clipped-domain discovery: an open zero set bounded by certified face traces.

The declared domain box is a clipping set, not a container: the zero set may
cross its faces. Each of the six faces is first resolved by the
parametric-chart crease discovery (``discover_parametric_chart_curve``) at the
discovery target level; its root bank then supplies every face crossing of the
volume polygonization verbatim (``polygonize(..., clipped=...)``), so the
contour's boundary edges are exactly the crease segments. A tangential or
otherwise uncertified face contact is a decided chart event and the volume is
not polygonized. Typical use is a field composed with an envelope
parametrization, whose box faces are the envelope's surface charts.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import final, Literal, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked, parse
from .._contracts import CompiledGeometry
from ._adaptive_discovery import (
    _cell_diagonal,
    _cover_identity,
    _FieldEnclosures,
    _PACKED_OWNERSHIP,
    _state_identity,
    _target_level,
    _validate_discovery_inputs,
    AdaptiveImplicitCover,
    AdaptiveImplicitScaffold,
    AdaptiveImplicitSurfaceEvidence,
)
from ._adaptive_packed_polygon import _records, ClippedBoundary
from ._curve_discovery import (
    _CurvatureEnclosures,
    _discover_chart_face,
    ParametricChartCurve,
)
from ._policy import (
    AdaptiveImplicitBoxIssue,
    AdaptiveImplicitSurfacePolicy,
    AdaptiveImplicitSurfaceStatus,
)


ClippedDiscoveryOutcome: TypeAlias = Literal[
    "resolved", "tangency", "lattice_vertex", "binary64_resolution", "unresolved"
]
"""Decided outcome of clipped discovery.

``resolved``: every face chart and every volume band cell is certified and the
open contour exists. ``tangency``/``lattice_vertex``/``binary64_resolution``:
a face chart event (see ``ParametricChartOutcome``); the volume is not
polygonized. ``unresolved``: volume band cells stay unresolved at
``root_tolerance`` or with a lattice vertex on the zero set.
"""

# Volume issues a finer lattice resolves for a regular zero set. A tangency
# box hugs a lattice face no deeper than ``2**-8`` of the cell, and a finer
# lattice halves that slab.
_REFINABLE = int(
    AdaptiveImplicitBoxIssue.EDGE
    | AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS
    | AdaptiveImplicitBoxIssue.SINGULAR
    | AdaptiveImplicitBoxIssue.FACE
    | AdaptiveImplicitBoxIssue.TANGENCY
)
_FACES = tuple((axis, side) for axis in range(3) for side in range(2))


def clipped_boundary(charts: tuple[ParametricChartCurve, ...], /) -> ClippedBoundary:
    """Merge the six face root banks; a shared box-edge root must agree bitwise."""
    ids = np.concatenate([chart.root_edge_ids for chart in charts]).reshape(-1, 6)
    unresolved = [
        np.column_stack(
            (
                np.full(chart.unresolved_cells.shape[0], chart.axis, np.int64),
                np.full(chart.unresolved_cells.shape[0], chart.side, np.int64),
                chart.unresolved_cells,
            )
        )
        for chart in charts
    ]
    unresolved_rows = np.concatenate(unresolved).reshape(-1, 4)
    if not ids.shape[0]:
        return ClippedBoundary(
            np.empty((0, 2, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int8),
            unresolved_rows,
        )
    columns = {
        name: np.concatenate([getattr(chart, name) for chart in charts])
        for name in (
            "root_points",
            "root_errors",
            "root_issues",
            "root_brackets",
            "root_signs",
        )
    }
    unique, first, inverse = np.unique(
        np.ascontiguousarray(ids), axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.reshape(-1)
    for name, values in columns.items():
        kept = values[first]
        if name in ("root_points", "root_errors", "root_brackets"):
            same = np.array_equal(
                np.asarray(values, dtype=np.float64).view(np.uint64),
                np.asarray(kept[inverse], dtype=np.float64).view(np.uint64),
            )
        else:
            same = np.array_equal(values, kept[inverse])
        if not same:
            raise ValueError("Face charts disagree on a shared box-edge crease root.")
        columns[name] = kept
    order = np.argsort(_records(np.ascontiguousarray(unique, dtype=np.int64)))
    return ClippedBoundary(
        np.ascontiguousarray(unique[order]).reshape(-1, 2, 3),
        columns["root_points"][order],
        columns["root_errors"][order],
        columns["root_issues"][order],
        columns["root_brackets"][order],
        columns["root_signs"][order],
        unresolved_rows,
    )


@final
class ClippedImplicitSurface(StrictModule, NonTrainableState):
    """Open zero set of a region field clipped by its discovery box.

    ``charts`` are the six certified face traces in ``(axis, side)`` order.
    ``points``/``faces`` are the open contour (``None`` arrays are empty when
    not ``resolved``), oriented with the field gradient, and
    ``vertex_root_rows`` names each vertex's scaffold root edge. Contour edges
    used by one triangle lie in a domain face and are exactly the chart
    crease segments. ``cover`` is the complete certified cover of the box.
    """

    cover: AdaptiveImplicitCover | None
    evidence: AdaptiveImplicitSurfaceEvidence | None
    scaffold: AdaptiveImplicitScaffold | None
    charts: tuple[ParametricChartCurve, ...]
    domain: np.ndarray
    points: np.ndarray
    faces: np.ndarray
    vertex_root_rows: np.ndarray
    feature_edges: np.ndarray
    boundary_witness_boxes: np.ndarray
    boundary_witnesses: np.ndarray
    boundary_errors: np.ndarray
    target_level: int = eqx.field(static=True)
    outcome: ClippedDiscoveryOutcome = eqx.field(static=True)
    policy: AdaptiveImplicitSurfacePolicy = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    refinement_history: tuple[tuple[int, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        cover: AdaptiveImplicitCover | None,
        evidence: AdaptiveImplicitSurfaceEvidence | None,
        scaffold: AdaptiveImplicitScaffold | None,
        charts: tuple[ParametricChartCurve, ...],
        domain: np.ndarray,
        arrays: dict[str, np.ndarray],
        target_level: int,
        outcome: ClippedDiscoveryOutcome,
        policy: AdaptiveImplicitSurfacePolicy,
        source_id: str,
        refinement_history: tuple[tuple[int, str], ...] = (),
    ) -> None:
        def frozen(name: str, dtype: type, shape: tuple[int, ...]) -> np.ndarray:
            value = np.array(arrays[name], dtype=dtype, copy=True).reshape(shape)
            value.setflags(write=False)
            return value

        if len(charts) != len(_FACES) or any(
            (chart.axis, chart.side) != face
            for chart, face in zip(charts, _FACES, strict=True)
        ):
            raise ValueError("Clipped discovery needs one chart per box face in order.")
        self.cover = cover
        self.evidence = evidence
        self.scaffold = scaffold
        self.charts = charts
        box = np.array(domain, dtype=np.float64, copy=True).reshape((2, 3))
        box.setflags(write=False)
        self.domain = box
        self.points = frozen("points", np.float64, (-1, 3))
        self.faces = frozen("faces", np.int64, (-1, 3))
        self.vertex_root_rows = frozen("vertex_root_rows", np.int64, (-1,))
        self.feature_edges = frozen("feature_edges", np.int64, (-1, 2))
        self.boundary_witness_boxes = frozen("witness_boxes", np.float64, (-1, 2, 3))
        self.boundary_witnesses = frozen("witness_values", np.float64, (-1, 3))
        self.boundary_errors = frozen("witness_errors", np.float64, (-1,))
        self.target_level = int(target_level)
        self.outcome = parse(outcome, ClippedDiscoveryOutcome, "outcome")
        self.policy = policy
        self.source_id = source_id
        self.refinement_history = tuple(
            (int(level), str(reason)) for level, reason in refinement_history
        )

    @property
    def resolved(self) -> bool:
        return self.outcome == "resolved"

    @property
    def mesh(self) -> None:
        """Clipped discovery retains no closed triangle mesh (replay protocol)."""
        return None

    @property
    def boundary(self) -> ClippedBoundary:
        """The merged crease root bank and uncertified face cells."""
        return clipped_boundary(self.charts)

    def replay(self, geometry: CompiledGeometry, /) -> bool:
        """Cold-replay the retained polygonization under the clipped boundary."""
        from ._adaptive_polygon_replay import replay_polygon

        if self.scaffold is None:
            return False
        bounds = _FieldEnclosures(geometry, self.policy.enclosure)
        return replay_polygon(self, bounds, clipped=self.boundary)


def _empty_arrays() -> dict[str, np.ndarray]:
    return {
        "points": np.empty((0, 3)),
        "faces": np.empty((0, 3)),
        "vertex_root_rows": np.empty((0,)),
        "feature_edges": np.empty((0, 2)),
        "witness_boxes": np.empty((0, 2, 3)),
        "witness_values": np.empty((0, 3)),
        "witness_errors": np.empty((0,)),
    }


def _policy_at_level(
    policy: AdaptiveImplicitSurfacePolicy, domain: np.ndarray, level: int, /
) -> AdaptiveImplicitSurfacePolicy:
    """The policy whose physical resolution selects discovery ``level``."""
    current = policy
    while _target_level(domain, current.relative_cell_diameter) < level:
        following = current.relative_cell_diameter * 0.5
        if not math.isfinite(following) or following <= 0.0:
            raise ValueError("The clipped discovery resolution is not representable.")
        current = replace(current, relative_cell_diameter=following)
    return current


@checked
def discover_clipped_implicit_surface(
    geometry: CompiledGeometry,
    /,
    *,
    domain: ArrayLike,
    policy: AdaptiveImplicitSurfacePolicy = AdaptiveImplicitSurfacePolicy(),
    source_id: str,
) -> ClippedImplicitSurface:
    """Discover the zero set of ``geometry`` clipped to ``domain`` (open-boundary mode).

    This is the explicit, opt-in open-boundary request: the closed-domain
    ``discover_adaptive_implicit_surface`` keeps its ``DOMAIN_BOUNDARY``
    refusal. The face charts are resolved first, refining the common target
    level until every chart is certified or decides an event; the volume is
    then polygonized with the merged crease root bank and refined while its
    unresolved cells are refinable and wider than ``root_tolerance``. The
    lattice is never re-anchored: the domain faces are the clipping surfaces.
    """
    from ._adaptive_packed_frontier import discover_leaves, PackedPolygonLeaves
    from ._adaptive_packed_polygon import polygonize

    _validate_discovery_inputs(geometry, policy, source_id)
    if policy.enclosure != "interval":
        raise ValueError("Clipped discovery requires the interval enclosure policy.")
    box = np.array(domain, dtype=np.float64, copy=True)
    if box.shape != (2, 3) or not np.all(np.isfinite(box)) or np.any(box[0] >= box[1]):
        raise ValueError("domain must be a finite ordered (2, 3) box.")
    bounds = _FieldEnclosures(geometry, policy.enclosure)
    if not bounds.gradient_rigorous:
        raise ValueError("Clipped discovery requires rigorous gradient enclosures.")
    curvature = _CurvatureEnclosures(geometry)
    level = _target_level(box, policy.relative_cell_diameter)
    tolerance = float(policy.root_tolerance)
    history: list[tuple[int, str]] = []
    while True:
        current = _policy_at_level(policy, box, level)
        charts = tuple(
            _discover_chart_face(
                bounds, curvature, box, axis, side, level, tolerance, source_id
            )
            for axis, side in _FACES
        )
        terminal = [
            chart.outcome
            for chart in charts
            if chart.outcome not in ("resolved", "refine")
        ]
        if terminal:
            outcome: ClippedDiscoveryOutcome = terminal[0]
            return ClippedImplicitSurface(
                cover=None,
                evidence=None,
                scaffold=None,
                charts=charts,
                domain=box,
                arrays=_empty_arrays(),
                target_level=level,
                outcome=outcome,
                policy=current,
                source_id=source_id,
                refinement_history=tuple(history),
            )
        required = max(chart.required_level for chart in charts)
        if required > level:
            history.extend(
                (
                    level,
                    f"chart {chart.axis}{'-+'[chart.side]}: {chart.unresolved_cells.shape[0]} cells",
                )
                for chart in charts
                if chart.outcome == "refine"
            )
            level = required
            continue
        boundary = clipped_boundary(charts)
        leaves = discover_leaves(bounds, box, level)
        leaf_count = len(leaves.level)
        surface_count = len(leaves.surface_rows)
        deepest = leaves.deepest_level
        status = leaves.status
        cover_arrays = (
            leaves.boxes,
            leaves.value_lower,
            leaves.value_upper,
            leaves.gradient_lower,
            leaves.gradient_upper,
        )
        polygon_leaves = PackedPolygonLeaves.from_leaves(leaves)
        del leaves
        polygon = polygonize(
            bounds,
            box,
            level,
            current,
            polygon_leaves,
            source_id,
            False,
            clipped=boundary,
        )
        del polygon_leaves
        status |= polygon.status
        issues = np.asarray(polygon.unresolved_issues, dtype=np.int64)
        if issues.size:
            status |= int(AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES)
        if not surface_count:
            status |= int(AdaptiveImplicitSurfaceStatus.NO_SURFACE)
        vertex_on_zero = bool(np.any(polygon.scaffold_arrays["corner_signs"] == 0))
        widest = max(
            (_cell_diagonal(item[0], item[1]) for item in polygon.unresolved_boxes),
            default=0.0,
        )
        if (
            issues.size
            and not vertex_on_zero
            and not np.any(issues & ~_REFINABLE)
            and widest > tolerance
        ):
            history.append(
                (
                    level,
                    f"volume: {issues.size} cells, issues {int(np.bitwise_or.reduce(issues))}",
                )
            )
            level += 1
            del polygon, cover_arrays
            continue
        break
    for values in (*cover_arrays, polygon.unresolved_boxes, issues):
        values.setflags(write=False)
    state_id = _state_identity(geometry)
    cover = AdaptiveImplicitCover(
        *cover_arrays,
        box,
        source_id,
        state_id,
        current.enclosure,
        True,
        True,
        _cover_identity(
            cover_arrays[0],
            cover_arrays[1],
            cover_arrays[2],
            box,
            source_id,
            state_id,
            current.enclosure,
            True,
        ),
        _ownership=_PACKED_OWNERSHIP,
    )
    evidence = AdaptiveImplicitSurfaceEvidence(
        polygon.unresolved_boxes,
        issues,
        "enclosed",
        current.enclosure,
        status,
        leaf_count,
        surface_count,
        deepest,
        polygon.max_surface_diagonal,
        canonical_fingerprint(
            {
                "kind": "clipped-implicit-evidence",
                "status": status,
                "unresolved_boxes": array_tree_fingerprint(polygon.unresolved_boxes),
                "unresolved_issues": array_tree_fingerprint(issues),
                "cover": cover.cover_id,
                "leaf_count": leaf_count,
            }
        ),
        declared_domain=box,
        lattice_anchor=0,
        _ownership=_PACKED_OWNERSHIP,
    )
    for values in polygon.scaffold_arrays.values():
        values.setflags(write=False)
    scaffold = AdaptiveImplicitScaffold(
        domain=box,
        target_level=level,
        _ownership=_PACKED_OWNERSHIP,
        **polygon.scaffold_arrays,
    )
    contour = polygon.clipped
    arrays = _empty_arrays()
    arrays.update(
        witness_boxes=polygon.witness_boxes,
        witness_values=polygon.witness_values,
        witness_errors=polygon.witness_errors,
    )
    if contour is not None:
        arrays.update(
            points=contour.points,
            faces=contour.faces,
            vertex_root_rows=contour.vertex_root_rows,
        )
    return ClippedImplicitSurface(
        cover=cover,
        evidence=evidence,
        scaffold=scaffold,
        charts=charts,
        domain=box,
        arrays=arrays,
        target_level=level,
        outcome="unresolved" if issues.size else "resolved",
        policy=current,
        source_id=source_id,
        refinement_history=tuple(history),
    )


__all__ = [
    "clipped_boundary",
    "ClippedDiscoveryOutcome",
    "ClippedImplicitSurface",
    "discover_clipped_implicit_surface",
]
