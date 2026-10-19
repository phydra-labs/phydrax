#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Membership and refusal contracts of the classified arrangement boundary."""

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available, MeshcoreStatus
from phydrax.geometry.surface import arrange_triangle_surfaces, SurfaceArrangementLimits
from phydrax.geometry.surface._arrangement import arrange_classified_surfaces
from tests._support.surface_arrangements import CUBE_FACES, CUBE_VERTICES


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]


def test_classified_publication_preserves_unclassified_fragment_geometry() -> None:
    inputs = (
        (CUBE_VERTICES, CUBE_FACES),
        (CUBE_VERTICES + (0.5, 0.25, 0.25), CUBE_FACES),
    )
    limits = SurfaceArrangementLimits()
    result = arrange_classified_surfaces(
        inputs, limits, limits.maximum_winding_evaluations
    )
    plain = arrange_triangle_surfaces(*inputs[0], *inputs[1], limits=limits)

    assert result.classification.status is MeshcoreStatus.OK
    np.testing.assert_array_equal(result.arrangement.vertices, plain.vertices)
    np.testing.assert_array_equal(result.arrangement.triangles, plain.triangles)
    np.testing.assert_array_equal(
        result.arrangement.vertex_features, plain.vertex_features
    )
    np.testing.assert_array_equal(result.arrangement.source_surface, plain.source_surface)
    np.testing.assert_array_equal(result.arrangement.source_face, plain.source_face)
    assert result.classification.fragment_components.shape == (
        result.arrangement.triangles.shape[0],
    )


def test_classification_capacity_retains_a_membership_refusal_record() -> None:
    inputs = (
        (CUBE_VERTICES, CUBE_FACES),
        (CUBE_VERTICES + (0.5, 0.25, 0.25), CUBE_FACES),
    )
    result = arrange_classified_surfaces(inputs, SurfaceArrangementLimits(), 1)

    assert result.classification.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert result.classification.fragment_components.size == 0
    assert result.classification.component_windings.shape == (0, 2)
    assert result.classification.pair_scans + result.classification.matrix_entries > 1
    assert result.arrangement.triangles.shape[0] > 0
