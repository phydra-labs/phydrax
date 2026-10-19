"""Curving preserves placed B-Rep occurrence identity and ambiguous-node refusal."""

from __future__ import annotations

import numpy as np

from tests._support.cad_curving import CURVING_STATUS, repeated_definition_curving


# Unchanged empirical allowance from the pre-refactor straight-edge, seam, plane-distance and zero-mean reference assertions.
CURVING_COORDINATE_ROUNDOFF_ATOL = 1e-12
# Unchanged empirical allowance from the pre-refactor placed occurrence-node residual assertions.
INSTANCE_NODE_RESIDUAL_ATOL = 1e-10


def test_curving_repeated_definition_preserves_authored_occurrence_projection() -> None:
    instances, definition, result = repeated_definition_curving(
        tuple((instance, corner) for instance in (0, 1) for corner in range(3)),
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
    )
    assert result.status is CURVING_STATUS.ROLLED_BACK_CERTIFICATION
    assert result.candidate is not None
    assert result.candidate.certificate.all_certified
    assert result.candidate.maximum_residual < INSTANCE_NODE_RESIDUAL_ATOL
    paths = result.candidate.node_occurrence_paths
    assert set(paths) == {instance.path for instance in instances}
    # Every degree-three node, including the face-interior diagonal's, carries
    # its own triangle's instance and lies on that instance's placed face.
    straight = np.asarray(result.straight.coordinates)
    for instance in instances:
        rows = [row for row, path in enumerate(paths) if path == instance.path]
        placed = instance.place(definition)
        normal = np.cross(placed[1] - placed[0], placed[2] - placed[0])
        assert len(rows) == 10
        np.testing.assert_allclose(
            (straight[rows] - placed[0]) @ normal,
            0.0,
            atol=CURVING_COORDINATE_ROUNDOFF_ATOL,
            rtol=0,
        )
    # Omitting occurrence identity would snap the translated/rotated instance
    # onto the definition, although both rows name identical definition indices.
    np.testing.assert_allclose(
        result.geometry.coordinates, result.straight.coordinates, atol=0, rtol=0
    )


def test_curving_mixed_occurrence_corners_remain_unresolved() -> None:
    instances, _, result = repeated_definition_curving(
        ((0, 0), (0, 1), (1, 2)), np.asarray([[0, 1, 2]], dtype=np.int32)
    )
    # No occurrence path is common to the corners, so neither the cell nor its
    # mixed edges derive a class; the refusal keeps the straight geometry.
    assert result.status is CURVING_STATUS.UNRESOLVED_ASSOCIATION
    assert result.candidate is None
    dimensions = np.asarray(result.evidence.node_dimensions)
    paths = result.evidence.node_occurrence_paths
    assert np.any(dimensions < 0)
    assert all(paths[row] == () for row in np.flatnonzero(dimensions < 0))
    assert {instance.path for instance in instances} <= set(paths)
    np.testing.assert_allclose(
        result.geometry.coordinates, result.straight.coordinates, atol=0, rtol=0
    )
