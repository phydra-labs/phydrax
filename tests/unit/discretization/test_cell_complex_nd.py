#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._cell_complex import (
    cubical_cell_complex,
    simplicial_cell_complex,
    simplicial_cell_geometry,
)
from phydrax.discretization._cochain_orientation import (
    reorient_cell_complex,
    reorient_cochain,
)


def test_four_simplex_boundary_and_reorientation() -> None:
    levels = tuple(
        np.asarray(tuple(combinations(range(5), degree + 1)), dtype=np.int32)
        for degree in range(5)
    )
    topology = simplicial_cell_complex(levels)
    expected_counts = (5, 10, 10, 5, 1)
    assert tuple(entity.count for entity in topology.entity_sets) == expected_counts
    rows, orientation = simplicial_cell_geometry(topology)
    for actual, expected in zip(rows, levels, strict=True):
        np.testing.assert_array_equal(actual, expected)
    changes = tuple(
        np.ones(count) if degree == 0 else np.where(np.arange(count) % 2, -1.0, 1.0)
        for degree, count in enumerate(expected_counts)
    )
    changed = reorient_cell_complex(topology, changes)
    changed_rows, changed_orientation = simplicial_cell_geometry(changed)
    for degree in range(5):
        np.testing.assert_array_equal(changed_rows[degree], rows[degree])
        np.testing.assert_array_equal(
            changed_orientation[degree], changes[degree] * orientation[degree]
        )
    for degree in range(1, 5):
        boundary = topology.incidences[degree - 1].scipy_boundary().toarray()
        expected = changes[degree - 1][:, None] * boundary * changes[degree][None, :]
        np.testing.assert_array_equal(
            changed.incidences[degree - 1].scipy_boundary().toarray(), expected
        )
    for lower, upper in zip(changed.incidences[:-1], changed.incidences[1:], strict=True):
        np.testing.assert_array_equal(
            (lower.scipy_boundary() @ upper.scipy_boundary()).toarray(), 0.0
        )


def test_simplicial_builder_refuses_missing_face_and_repeated_vertices() -> None:
    vertices = np.arange(3, dtype=np.int32)[:, None]
    with pytest.raises(ValueError, match="face closed"):
        simplicial_cell_complex(
            (vertices, np.asarray([[0, 1], [1, 2]]), np.asarray([[0, 1, 2]]))
        )
    with pytest.raises(ValueError, match="increasing"):
        simplicial_cell_complex((vertices, np.asarray([[1, 1]])))


def test_periodic_single_site_four_torus_has_zero_boundary() -> None:
    complex_ = cubical_cell_complex((1, 1, 1, 1), periodic=True)
    assert tuple(entity.count for entity in complex_.topology.entity_sets) == (
        1,
        4,
        6,
        4,
        1,
    )
    for incidence in complex_.topology.incidences:
        np.testing.assert_array_equal(incidence.scipy_boundary().toarray(), 0.0)
    assert complex_.orientations[2] == tuple(combinations(range(4), 2))
    with pytest.raises(ValueError, match="nonperiodic"):
        cubical_cell_complex((1, 2), periodic=(False, True))


def test_cubical_two_dimensional_boundary_has_outward_signs() -> None:
    complex_ = cubical_cell_complex((2, 2))
    np.testing.assert_array_equal(
        complex_.topology.incidences[1].scipy_boundary().toarray()[:, 0],
        np.asarray([1.0, -1.0, -1.0, 1.0]),
    )
    np.testing.assert_array_equal(
        complex_.topology.incidences[0].scipy_boundary().toarray(),
        np.asarray([[-1, 0, -1, 0], [0, -1, 1, 0], [1, 0, 0, -1], [0, 1, 0, 1]]),
    )


def test_reorientation_uses_explicit_cell_axis_with_equal_extents() -> None:
    values = jnp.arange(27.0).reshape((3, 3, 3))
    signs = jnp.asarray([-1.0, 1.0, -1.0])
    changed = jax.jit(lambda value: reorient_cochain(value, signs, cell_axis=1))(values)
    np.testing.assert_array_equal(
        changed, np.asarray(values) * np.asarray(signs)[None, :, None]
    )
    np.testing.assert_array_equal(reorient_cochain(changed, signs, cell_axis=-2), values)
    with pytest.raises(ValueError, match="out of range"):
        reorient_cochain(values, signs, cell_axis=3)
