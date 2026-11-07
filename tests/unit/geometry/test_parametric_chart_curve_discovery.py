#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import PeriodicCell
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    AdaptiveImplicitSurfaceStatus,
    discover_adaptive_implicit_surface,
)
from phydrax.geometry.implicit._adaptive_packed_polygon import ClippedBoundary
from phydrax.geometry.implicit._clipped_discovery import (
    discover_clipped_implicit_surface,
)
from phydrax.geometry.implicit._curve_discovery import (
    discover_parametric_chart_curve,
)


_G = phx.geometry
_POLICY = AdaptiveImplicitSurfacePolicy(
    relative_cell_diameter=0.125, root_tolerance=1e-10
)


def _schwarz_p() -> phx.geometry.TrigonometricLevelSet:
    return _G.TrigonometricLevelSet(
        _G.TrigonometricTable.family("schwarz_p"), PeriodicCell(np.eye(3))
    )


def _schwarz_p_value(points: np.ndarray) -> np.ndarray:
    angles = 2.0 * math.pi * np.asarray(points, dtype=np.float64)
    return np.sum(np.cos(angles), axis=-1)


def _loops(segments: np.ndarray) -> list[list[int]]:
    """Closed vertex cycles of an unordered segment set (every degree two)."""
    neighbours: dict[int, list[int]] = {}
    for a, b in segments.tolist():
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    assert all(len(items) == 2 for items in neighbours.values())
    loops, seen = [], set()
    for start in neighbours:
        if start in seen:
            continue
        loop, previous, current = [start], None, start
        seen.add(start)
        while True:
            following = [v for v in neighbours[current] if v != previous][0]
            if following == start:
                break
            loop.append(following)
            seen.add(following)
            previous, current = current, following
        loops.append(loop)
    return loops


def test_chart_crease_matches_the_analytic_face_trace() -> None:
    # On the face z = 0.1 the Schwarz P zero set is cos X + cos Y = -cos(0.2 pi):
    # one convex loop around (1/2, 1/2) (X = 2 pi x).
    domain = np.asarray(((0.05, 0.05, 0.1), (0.95, 0.95, 0.6)))
    chart = discover_parametric_chart_curve(
        _schwarz_p().compile(),
        domain=domain,
        axis=2,
        side=0,
        target_level=5,
        policy=_POLICY,
        source_id="schwarz-p-face",
    )

    assert chart.outcome == "resolved"
    assert chart.unresolved_cells.shape == (0, 2)
    points = np.asarray(chart.root_points)
    # Crease vertices lie exactly on the chart face and on the zero set to the
    # root enclosure (the field is 2 pi sqrt(3)-Lipschitz in x).
    assert np.all(points[:, 2] == 0.1)
    lipschitz = 2.0 * math.pi * math.sqrt(3.0)
    assert np.all(
        np.abs(_schwarz_p_value(points))
        <= lipschitz * np.asarray(chart.root_errors) + 1.0e-12
    )
    (loop,) = _loops(np.asarray(chart.segments))
    polygon = points[loop][:, :2]
    area = 0.5 * abs(
        float(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )
    )
    # Independent oracle: for each x, cos(2 pi y) <= b holds on a y-arc of
    # length 1 - arccos(b) / pi; Gauss-Legendre in x over the loop's span.
    level = -math.cos(0.2 * math.pi)
    nodes, weights = np.polynomial.legendre.leggauss(4000)
    x = 0.05 + 0.45 * (nodes + 1.0)
    bound = np.clip(level - np.cos(2.0 * math.pi * x), -1.0, 1.0)
    length = np.where(bound <= -1.0, 0.0, 1.0 - np.arccos(bound) / math.pi)
    oracle = 0.45 * float(np.sum(weights * length))
    assert area == pytest.approx(oracle, abs=2.0e-3)


def test_chart_tangency_is_a_certified_unresolved_event() -> None:
    # z = 1/4 is tangent to the Schwarz P surface at (0, 1/2, 1/4), where the
    # face trace cos X + cos Y has a saddle on its zero set (the crossing lines
    # x = +-(y - 1/2)): no lattice can certify it. The face lattice avoids
    # both lines at its vertices.
    domain = np.asarray(((-0.213, 0.327, 0.25), (0.171, 0.707, 0.6)))
    chart = discover_parametric_chart_curve(
        _schwarz_p().compile(),
        domain=domain,
        axis=2,
        side=0,
        target_level=4,
        policy=_POLICY,
        source_id="schwarz-p-tangent",
    )

    assert chart.outcome == "tangency"
    assert not chart.resolved
    boxes = np.asarray(chart.event_boxes)
    assert boxes.shape[0] > 0
    width = np.max(boxes[:, 1, :2] - boxes[:, 0, :2])
    assert width <= 1.0e-9
    contact = np.asarray((0.0, 0.5))
    assert np.any(
        np.all((boxes[:, 0, :2] <= contact) & (contact <= boxes[:, 1, :2]), axis=1)
    )


def test_chart_lattice_vertex_on_the_crease_is_refused() -> None:
    domain = np.asarray(((-0.5, 0.0, 0.25), (0.5, 1.0, 0.6)))
    chart = discover_parametric_chart_curve(
        _schwarz_p().compile(),
        domain=domain,
        axis=2,
        side=0,
        target_level=3,
        policy=_POLICY,
        source_id="schwarz-p-vertex",
    )

    assert chart.outcome == "lattice_vertex"
    assert chart.event_boxes.shape[0] > 0


def test_clipped_discovery_bounds_the_contour_by_the_chart_creases() -> None:
    geometry = _schwarz_p().compile()
    domain = np.asarray(((0.07, 0.11, 0.13), (0.83, 0.79, 0.91)))
    surface = discover_clipped_implicit_surface(
        geometry, domain=domain, policy=_POLICY, source_id="schwarz-p-clipped"
    )

    assert surface.resolved
    assert surface.faces.shape[0] > 0
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in np.asarray(surface.faces).tolist():
        for first, second in ((a, b), (b, c), (c, a)):
            key = (min(first, second), max(first, second))
            edges[key] = edges.get(key, 0) + 1
    assert set(edges.values()) <= {1, 2}
    assert surface.scaffold is not None
    scaffold_edges = np.asarray(surface.scaffold.root_edge_ids)
    rows = np.asarray(surface.vertex_root_rows)

    def edge_key(ids: np.ndarray) -> tuple[int, ...]:
        return tuple(int(value) for value in ids.reshape(-1))

    trace = {
        frozenset((edge_key(scaffold_edges[rows[a]]), edge_key(scaffold_edges[rows[b]])))
        for (a, b), count in edges.items()
        if count == 1
    }
    creases = {
        frozenset((edge_key(chart.root_edge_ids[a]), edge_key(chart.root_edge_ids[b])))
        for chart in surface.charts
        for a, b in np.asarray(chart.segments).tolist()
    }
    assert trace == creases
    # Every boundary vertex is the chart bank's root, bit for bit.
    boundary = surface.boundary
    for vertex in {v for (a, b), n in edges.items() if n == 1 for v in (a, b)}:
        ids = scaffold_edges[rows[vertex]].reshape(1, 6)
        position = int(
            np.flatnonzero(np.all(boundary.root_edge_ids.reshape(-1, 6) == ids, axis=1))[
                0
            ]
        )
        assert np.array_equal(
            surface.points[vertex].view(np.uint64),
            boundary.root_points[position].view(np.uint64),
        )
    assert surface.replay(geometry)
    tampered = ClippedBoundary(
        boundary.root_edge_ids,
        np.nextafter(boundary.root_points, np.inf),
        boundary.root_errors,
        boundary.root_issues,
        boundary.root_brackets,
        boundary.root_signs,
        boundary.unresolved_faces,
    )
    from phydrax.geometry.implicit._adaptive_discovery import _FieldEnclosures
    from phydrax.geometry.implicit._adaptive_polygon_replay import replay_polygon

    assert not replay_polygon(
        surface, _FieldEnclosures(geometry, "interval"), clipped=tampered
    )


def test_closed_domain_discovery_still_refuses_a_boundary_crossing() -> None:
    domain = np.asarray(((0.07, 0.11, 0.13), (0.83, 0.79, 0.91)))
    surface = discover_adaptive_implicit_surface(
        _schwarz_p().compile(), domain=domain, policy=_POLICY, source_id="closed"
    )

    assert surface.mesh is None
    assert (
        surface.evidence.status_flags
        & AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING
    )
