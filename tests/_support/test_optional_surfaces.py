"""Independent geometric invariants for the optional-provider surface fixture."""

import numpy as np

from tests._support.optional_surfaces import boolean_cube


def test_boolean_cube_has_closed_outward_unit_boundary() -> None:
    surface = boolean_cube((0.0, 0.0, 0.0))
    faces = np.asarray(surface.mesh.blocks[0].vertices, dtype=np.int64)
    points = np.asarray(surface.mesh.coordinates, dtype=np.float64)
    edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    np.testing.assert_array_equal(counts, np.full(18, 2, dtype=np.int64))
    corners = points[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    assert np.all(np.sum(normals * (np.mean(corners, axis=1) - 0.5), axis=1) > 0.0)
    volume = (
        np.sum(np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1))
        / 6.0
    )
    assert volume == 1.0


def test_boolean_cube_translation_retains_unit_bounds_and_source_tag() -> None:
    offset = (0.5, 0.25, -0.5)
    surface = boolean_cube(offset)
    points = np.asarray(surface.mesh.coordinates, dtype=np.float64)
    np.testing.assert_array_equal(np.min(points, axis=0), offset)
    np.testing.assert_array_equal(np.max(points, axis=0), np.asarray(offset) + 1.0)
    assert surface.metadata.source_id == str(offset)
