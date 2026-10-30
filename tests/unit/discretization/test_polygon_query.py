from __future__ import annotations

import numpy as np
import pytest

from phydrax.discretization import FieldQueryStatus
from phydrax.discretization._polygon_query import PreparedPolygonMeshLocator


def _winding_contains(polygon: np.ndarray, point: np.ndarray, /) -> bool:
    winding = 0
    for row in range(polygon.shape[0]):
        start = polygon[row]
        stop = polygon[(row + 1) % polygon.shape[0]]
        cross = (stop[0] - start[0]) * (point[1] - start[1]) - (stop[1] - start[1]) * (
            point[0] - start[0]
        )
        if start[1] <= point[1] < stop[1] and cross > 0.0:
            winding += 1
        elif stop[1] <= point[1] < start[1] and cross < 0.0:
            winding -= 1
    return winding != 0


def test_nonconvex_polygon_location_matches_independent_winding_oracle() -> None:
    polygon = np.asarray(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ),
        dtype=np.float64,
    )
    query = np.asarray(
        [
            (x, y)
            for y in np.arange(0.125, 2.0, 0.25)
            for x in np.arange(-0.375, 2.5, 0.25)
        ],
        dtype=np.float64,
    )
    cells = np.arange(polygon.shape[0], dtype=np.int32)[None, :]
    locator = PreparedPolygonMeshLocator(
        polygon,
        cells,
        np.asarray((polygon.shape[0],), dtype=np.int32),
        np.asarray((True,), dtype=np.bool_),
        tolerance=1e-12,
        maximum_condition=1e12,
    )

    location = locator.locate(query)

    expected = np.asarray([_winding_contains(polygon, point) for point in query])
    observed = np.asarray(location.count) > 0
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(
        np.asarray(location.status) == int(FieldQueryStatus.VALID), expected
    )
    np.testing.assert_array_equal(
        np.asarray(location.candidate_cells) >= 0,
        expected[:, None],
    )
    for row in np.flatnonzero(expected):
        candidate = int(np.flatnonzero(np.asarray(location.candidate_cells[row]) >= 0)[0])
        triangle = np.asarray(location.candidate_triangle_vertices[row, candidate])
        barycentric = np.asarray(location.candidate_barycentric[row, candidate])
        reconstructed = barycentric @ polygon[triangle]
        np.testing.assert_allclose(reconstructed, query[row], rtol=0.0, atol=2e-14)


def test_polygon_locator_rejects_a_self_intersecting_loop() -> None:
    polygon = np.asarray(
        ((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)),
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="simple boundary loop"):
        PreparedPolygonMeshLocator(
            polygon,
            np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            np.asarray((4,), dtype=np.int32),
            np.asarray((True,), dtype=np.bool_),
            tolerance=1e-12,
            maximum_condition=1e12,
        )
