# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Preserve local vertex ordering when orienting independent mesh fixtures."""

import numpy as np
import pytest

from tests._support.cell_meshes import kuhn_cube, unit_square


@pytest.mark.parametrize(
    ("swap", "expected"),
    (
        (
            (0, 1),
            (
                (0, 4, 6, 7),
                (4, 0, 5, 7),
                (2, 0, 6, 7),
                (0, 2, 3, 7),
                (0, 1, 5, 7),
                (1, 0, 3, 7),
            ),
        ),
        (
            (1, 2),
            (
                (0, 4, 6, 7),
                (0, 5, 4, 7),
                (0, 6, 2, 7),
                (0, 2, 3, 7),
                (0, 1, 5, 7),
                (0, 3, 1, 7),
            ),
        ),
    ),
    ids=("first-pair", "middle-pair"),
)
def test_kuhn_cube_retains_requested_positive_local_order(
    swap: tuple[int, int], expected: tuple[tuple[int, ...], ...]
) -> None:
    # The table enumerates the six monotone coordinate-axis paths through the
    # binary cube. Odd axis permutations receive only the requested swap.
    mesh = kuhn_cube(orientation_swap=swap)
    np.testing.assert_array_equal(mesh.blocks[0].vertices, expected)
    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    determinants = np.linalg.det(corners[:, 1:] - corners[:, :1])
    np.testing.assert_array_equal(determinants, 1.0)


def test_square_fixture_refuses_an_identity_orientation_swap() -> None:
    with pytest.raises(ValueError, match="cell_meshes.orientation_swap"):
        unit_square(orientation_swap=(1, 1))
