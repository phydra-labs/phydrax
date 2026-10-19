# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Fault adequacy of shared arrangement and rational-volume references."""

import numpy as np
import pytest

from tests._support.surface_arrangements import (
    assert_cross_operand_pairs,
    crossing_cubes,
    CUBE_FACES,
    CUBE_VERTICES,
)
from tests._support.surface_boolean_reference import signed_volume_reference


def test_crossing_cube_builder_covers_every_distinct_operand_face_pair() -> None:
    vertices, faces, labels, pairs = crossing_cubes()
    assert vertices.shape == (16, 3)
    assert faces.shape == (24, 3)
    assert_cross_operand_pairs(labels, pairs)


def test_contact_coverage_reference_rejects_a_missing_face_pair() -> None:
    _, _, labels, pairs = crossing_cubes()
    with pytest.raises(AssertionError, match="cross-operand source-face coverage"):
        assert_cross_operand_pairs(labels, pairs[:-1])


def test_contact_coverage_reference_rejects_a_duplicate_face_pair() -> None:
    _, _, labels, pairs = crossing_cubes()
    duplicated = np.concatenate((pairs, pairs[:1]))
    with pytest.raises(AssertionError, match="duplicate cross-operand source-face pair"):
        assert_cross_operand_pairs(labels, duplicated)


def test_rational_volume_is_translation_invariant_and_orientation_sensitive() -> None:
    translated = CUBE_VERTICES + np.asarray((2.0**50, -(2.0**50), 2.0**50))
    assert signed_volume_reference(CUBE_VERTICES, CUBE_FACES) == 1.0
    assert signed_volume_reference(translated, CUBE_FACES) == 1.0
    assert signed_volume_reference(translated, CUBE_FACES[:, ::-1]) == -1.0


def test_rational_volume_reference_detects_an_inverted_boundary_face() -> None:
    broken = CUBE_FACES.copy()
    broken[2] = broken[2, ::-1]
    assert signed_volume_reference(CUBE_VERTICES, broken) != 1.0
