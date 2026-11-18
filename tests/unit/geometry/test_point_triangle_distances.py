# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available, point_triangle_distances


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(), reason="native phydrax-meshcore library is unavailable"
    ),
]


def test_point_triangle_distances_batch_and_empty() -> None:
    points = np.array([[0.25, 0.25, 2], [1, 1, 0]], dtype=np.float64)
    triangles = np.array(
        [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 0], [1, 0, 0]]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(point_triangle_distances(points, triangles), [2, 1])
    assert point_triangle_distances(np.empty((0, 3)), np.empty((0, 3, 3))).shape == (0,)
    np.testing.assert_array_equal(
        point_triangle_distances(points[::-1], triangles[::-1]), [1, 2]
    )


@pytest.mark.parametrize(
    ("points", "triangles", "exception"),
    [
        (np.zeros((3,)), np.zeros((1, 3, 3)), ValueError),
        (np.zeros((1, 3)), np.zeros((1, 9)), ValueError),
        (np.zeros((2, 3)), np.zeros((1, 3, 3)), ValueError),
        (np.zeros((1, 3), dtype=np.bool_), np.zeros((1, 3, 3)), TypeError),
        (np.zeros((1, 3)), np.zeros((1, 3, 3), dtype=np.complex128), TypeError),
        (np.array([[2**53 + 1, 0, 0]], dtype=np.int64), np.zeros((1, 3, 3)), ValueError),
        (np.array([[np.nan, 0, 0]]), np.zeros((1, 3, 3)), ValueError),
        (np.zeros((1, 3)), np.full((1, 3, 3), np.inf), ValueError),
        (np.array([[1e308, 0, 0]]), np.zeros((1, 3, 3)), ValueError),
    ],
)
def test_point_triangle_distances_refuse_invalid_inputs(
    points: np.ndarray, triangles: np.ndarray, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        point_triangle_distances(points, triangles)
