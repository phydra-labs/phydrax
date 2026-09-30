#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import combinations
from math import comb

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import (
    CellMesh,
    reference_cell_topology,
    SimplicialConnectivity,
)
from phydrax.discretization.fem import FiniteElementDeRhamComplex


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


@pytest.mark.parametrize(
    "dimension",
    (1, 2, 4, 5),
    ids=("dimension-1", "dimension-2", "dimension-4", "dimension-5"),
)
def test_declared_simplex_mesh_retains_oriented_boundary(dimension: int) -> None:
    points = np.concatenate(
        (np.zeros((1, dimension), dtype=np.float64), np.eye(dimension, dtype=np.float64))
    )
    vertices = np.arange(dimension + 1, dtype=np.int32)
    vertices[:2] = vertices[1::-1]
    mesh = CellMesh.from_simplices(points, vertices[None], dimension=dimension)
    assert isinstance(mesh.connectivity, SimplicialConnectivity)
    assert mesh.topological_dimension == dimension
    np.testing.assert_array_equal(mesh.connectivity.cells, vertices[None])
    np.testing.assert_array_equal(mesh.connectivity.boundary_masks[-1], [False])
    for degree, entities in enumerate(mesh.topology.entity_sets):
        count = entities.count
        assert count == comb(dimension + 1, degree + 1), f"degree={degree}"
    for degree in range(1, dimension):
        lower = mesh.topology.incidences[degree - 1].exterior_derivative()
        upper = mesh.topology.incidences[degree].exterior_derivative()
        values = jnp.arange(lower.source.size, dtype=jnp.float64)
        np.testing.assert_allclose(upper.mv(lower.mv(values)), 0.0, atol=1e-14)


def test_shared_four_dimensional_facet_is_not_a_support_boundary() -> None:
    points = np.concatenate(
        (
            np.zeros((1, 4), dtype=np.float64),
            np.eye(4, dtype=np.float64),
            -np.eye(4, dtype=np.float64)[3:4],
        )
    )
    cells = np.asarray(((0, 1, 2, 3, 4), (0, 2, 1, 3, 5)), dtype=np.int32)
    mesh = CellMesh.from_simplices(points, cells, dimension=4)
    assert isinstance(mesh.connectivity, SimplicialConnectivity)
    assert int(jnp.sum(mesh.connectivity.boundary_masks[3])) == 8
    complex_ = FiniteElementDeRhamComplex(
        mesh, family="trimmed", order=1, twist="untwisted"
    )
    geometry = complex_.reconstruction(0).support_geometry
    queries = jnp.asarray(((0.1, 0.1, 0.1, 0.0), (0.5, 0.5, 0.5, 0.0)), dtype=jnp.float64)
    np.testing.assert_array_equal(geometry.contains(queries), [True, False])
    np.testing.assert_allclose(
        eqx.filter_jit(geometry.boundary_field)(queries),
        np.asarray((-0.1, np.sqrt(1.0 / 12.0)), dtype=np.float64),
        rtol=1e-10,
        atol=1e-12,
    )


def test_dimension_qualified_reference_has_explicit_entities_and_capacity() -> None:
    reference = reference_cell_topology("simplex:4")
    assert reference.dimension == 4
    assert reference.entities[2] == tuple(combinations(range(5), 3))
    cube = reference_cell_topology("tensor:4")
    assert tuple(len(level) for level in cube.entities) == (16, 32, 24, 8, 1)
    with pytest.raises(ValueError, match="maximum_entities"):
        reference_cell_topology("tensor:4", maximum_entities=80)
    with pytest.raises(ValueError):
        CellMesh.from_simplices(
            np.eye(4, dtype=np.float64),
            np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            dimension=4,
        )
