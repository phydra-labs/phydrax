#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.geometry._approximate_source_coverage import SourceCoverageUnavailable
from phydrax.geometry._implicit_cube_topology import prove_implicit_cube_topology
from phydrax.geometry.implicit._adaptive_discovery import _CORNER_OFFSETS


class _Plane:
    def boxes(
        self, lower: NDArray[np.float64], upper: NDArray[np.float64]
    ) -> SimpleNamespace:
        return SimpleNamespace(
            value_lower=np.sum(lower, axis=1) - 0.5,
            value_upper=np.sum(upper, axis=1) - 0.5,
            gradient_lower=np.ones_like(lower),
            gradient_upper=np.ones_like(upper),
        )


class _HiddenSphere:
    def boxes(
        self, lower: NDArray[np.float64], upper: NDArray[np.float64]
    ) -> SimpleNamespace:
        lo = lower - 0.5
        hi = upper - 0.5
        nearest = np.where((lo <= 0) & (hi >= 0), 0, np.minimum(lo * lo, hi * hi))
        farthest = np.maximum(lo * lo, hi * hi)
        return SimpleNamespace(
            value_lower=np.sum(nearest, axis=1) - 0.04,
            value_upper=np.sum(farthest, axis=1) - 0.04,
            gradient_lower=2 * lo,
            gradient_upper=2 * hi,
        )


def _surface() -> SimpleNamespace:
    corners = np.asarray(sorted(map(tuple, _CORNER_OFFSETS)), dtype=np.int64)
    edges = np.asarray(
        sorted((tuple((0, 0, 0)), tuple(p)) for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        dtype=np.int64,
    )
    points = np.mean(edges, axis=1)
    # Edge root order is z, y, x; reverse it to orient toward (1,1,1).
    scaffold = SimpleNamespace(
        cell_boxes=np.asarray([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]),
        cell_corner_ids=_CORNER_OFFSETS[None, :, :].copy(),
        corner_ids=corners,
        corner_coordinates=corners.astype(np.float64),
        corner_signs=np.where(np.sum(corners, axis=1) == 0, -1, 1),
        root_edge_ids=edges,
        root_brackets=np.stack((points, points), axis=1),
        triangle_cell_rows=np.asarray([0]),
        triangle_root_edge_rows=np.asarray([[2, 1, 0]]),
    )
    return SimpleNamespace(
        scaffold=scaffold, mesh=SimpleNamespace(faces=np.zeros((1, 3), dtype=int))
    )


def test_plane_cube_has_independent_regular_disk_proof() -> None:
    prove_implicit_cube_topology(_surface(), _Plane())


def test_parent_gradient_enclosure_avoids_redundant_subbox_queries() -> None:
    class CountingPlane(_Plane):
        def __init__(self) -> None:
            self.calls: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

        def boxes(
            self, lower: NDArray[np.float64], upper: NDArray[np.float64]
        ) -> SimpleNamespace:
            self.calls.append((lower.copy(), upper.copy()))
            return super().boxes(lower, upper)

    bounds = CountingPlane()
    prove_implicit_cube_topology(_surface(), bounds)
    assert len(bounds.calls) == 1
    np.testing.assert_array_equal(bounds.calls[0][0], [[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(bounds.calls[0][1], [[1.0, 1.0, 1.0]])


def test_quad_face_arcs_and_internal_diagonal_are_checked() -> None:
    class PlaneXY:
        def boxes(
            self, lower: NDArray[np.float64], upper: NDArray[np.float64]
        ) -> SimpleNamespace:
            gradient = np.broadcast_to([1.0, 1.0, 0.0], lower.shape)
            return SimpleNamespace(
                value_lower=lower[:, 0] + lower[:, 1] - 0.5,
                value_upper=upper[:, 0] + upper[:, 1] - 0.5,
                gradient_lower=gradient,
                gradient_upper=gradient,
            )

    surface = _surface()
    scaffold = surface.scaffold
    scaffold.corner_signs = np.where(
        np.sum(scaffold.corner_ids[:, :2], axis=1) == 0, -1, 1
    )
    pairs = [((0, 0, z), endpoint) for z in (0, 1) for endpoint in ((1, 0, z), (0, 1, z))]
    scaffold.root_edge_ids = np.asarray(sorted(pairs), dtype=np.int64)
    points = np.mean(scaffold.root_edge_ids, axis=1)
    scaffold.root_brackets = np.stack((points, points), axis=1)
    root = {
        tuple(point): row for row, point in enumerate(np.asarray(points).reshape(-1, 3))
    }
    a, b, c, d = (
        root[p]
        for p in ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.5, 1.0), (0.5, 0.0, 1.0))
    )
    scaffold.triangle_cell_rows = np.asarray([0, 0])
    scaffold.triangle_root_edge_rows = np.asarray([[a, b, c], [a, c, d]])
    surface.mesh.faces = np.zeros((2, 3), dtype=int)
    prove_implicit_cube_topology(surface, PlaneXY())
    scaffold.triangle_root_edge_rows[1] = (a, b, d)
    with pytest.raises(SourceCoverageUnavailable):
        prove_implicit_cube_topology(surface, PlaneXY())


@pytest.mark.parametrize("mutation", ["edge", "root", "orientation", "omission"])
def test_wrong_cube_ancestry_or_orientation_is_refused(mutation: str) -> None:
    surface = _surface()
    scaffold = surface.scaffold
    if mutation == "edge":
        scaffold.root_edge_ids[0, 1] = (1, 1, 1)
    elif mutation == "root":
        scaffold.triangle_root_edge_rows[0] = (2, 1, 1)
    elif mutation == "orientation":
        scaffold.triangle_root_edge_rows[0] = (0, 1, 2)
    else:
        scaffold.triangle_cell_rows = np.empty(0, dtype=int)
        scaffold.triangle_root_edge_rows = np.empty((0, 3), dtype=int)
        surface.mesh.faces = np.empty((0, 3), dtype=int)
    with pytest.raises(SourceCoverageUnavailable):
        prove_implicit_cube_topology(surface, _Plane())


def test_hidden_closed_component_is_not_inferred_absent_from_edge_signs() -> None:
    surface = _surface()
    scaffold = surface.scaffold
    scaffold.corner_signs[:] = 1
    scaffold.root_edge_ids = np.empty((0, 2, 3), dtype=np.int64)
    scaffold.root_brackets = np.empty((0, 2, 3))
    scaffold.triangle_cell_rows = np.empty(0, dtype=int)
    scaffold.triangle_root_edge_rows = np.empty((0, 3), dtype=int)
    surface.mesh.faces = np.empty((0, 3), dtype=int)
    with pytest.raises(SourceCoverageUnavailable) as exc:
        prove_implicit_cube_topology(surface, _HiddenSphere())
    assert exc.value.status == "unresolved"


def test_unresolved_gradient_is_not_accepted() -> None:
    class Unresolved(_Plane):
        def boxes(
            self, lower: NDArray[np.float64], upper: NDArray[np.float64]
        ) -> SimpleNamespace:
            result = super().boxes(lower, upper)
            result.gradient_lower[:] = -1
            result.gradient_upper[:] = 1
            return result

    with pytest.raises(SourceCoverageUnavailable) as exc:
        prove_implicit_cube_topology(_surface(), Unresolved())
    assert exc.value.status == "unresolved"
