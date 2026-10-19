# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Closed-form controls for the independent authored-material cover reference."""

import numpy as np
import pytest

from tests._support.surface_metric import material_triangle_weights


def test_material_triangle_weights_preserve_first_cover_and_detect_corrupted_weights() -> (
    None
):
    triangle = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    triangles = np.stack((triangle, triangle))
    point = np.asarray((0.25, 0.25), dtype=np.float64)
    row, weights = material_triangle_weights(triangles, point)

    def require_authored_weights(actual_row: int, actual: np.ndarray) -> None:
        assert actual_row == 0
        np.testing.assert_array_equal(
            actual, np.asarray((0.5, 0.25, 0.25), dtype=np.float64)
        )

    require_authored_weights(row, weights)
    corrupted = weights.copy()
    corrupted[0] += 0.125
    with pytest.raises(AssertionError):
        require_authored_weights(row, corrupted)
    with pytest.raises(AssertionError):
        require_authored_weights(1, weights)
    with pytest.raises(ValueError, match=r"material_reference\.coverage"):
        material_triangle_weights(triangles, np.asarray((2.0, 2.0), dtype=np.float64))
