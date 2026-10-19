#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Binary64 unit cubes and exhaustive cross-operand arrangement contacts."""

import numpy as np


CUBE_VERTICES = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    )
)
CUBE_FACES = np.asarray(
    (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ),
    dtype=np.int64,
)


def crossing_cubes(
    surfaces: tuple[int, int] = (0, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return two crossing cubes with every cross-operand source-face pair."""
    vertices = np.concatenate((CUBE_VERTICES, CUBE_VERTICES + (0.5, 0.25, 0.25)))
    faces = np.concatenate((CUBE_FACES, CUBE_FACES + 8))
    labels = np.repeat(np.asarray(surfaces, dtype=np.int32), 12)
    first, second = np.meshgrid(np.arange(12), np.arange(12, 24), indexing="ij")
    pairs = np.stack((first.reshape(-1), second.reshape(-1)), axis=1)
    return vertices, faces, labels, pairs


def assert_cross_operand_pairs(labels: np.ndarray, pairs: np.ndarray) -> None:
    """Require one contact candidate per source face pair of distinct operands."""
    expected = {
        (first, second)
        for first in range(labels.size)
        for second in range(first + 1, labels.size)
        if labels[first] != labels[second]
    }
    observed = {tuple(row) for row in pairs.tolist()}
    assert observed == expected, "cross-operand source-face coverage"
    assert pairs.shape[0] == len(observed), "duplicate cross-operand source-face pair"
