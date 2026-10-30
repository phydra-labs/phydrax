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
    CellBlock,
    CellMesh,
    PolyhedralConnectivity,
    reference_cell_topology,
    SimplicialConnectivity,
)
from phydrax.discretization.fem import FiniteElementDeRhamComplex
from phydrax.geometry import CompiledGeometry


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


def test_mixed_qualified_simplex_keeps_conforming_interface_and_source_kinds() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 1.0),
            (0.5, -1.0, 0.25),
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "layer",
                "pyramid",
                np.asarray(((0, 1, 2, 3, 4),), dtype=np.int32),
                global_ids=(31,),
            ),
            CellBlock(
                "core",
                "simplex:3",
                np.asarray(((0, 1, 4, 5),), dtype=np.int32),
                global_ids=(47,),
            ),
        ),
    )
    assert isinstance(mesh.connectivity, PolyhedralConnectivity)
    assert mesh.block("core").cell_kind == "simplex:3"
    assert mesh.block("layer").cell_kind == "pyramid"
    np.testing.assert_array_equal(mesh.block("core").global_ids, (47,))
    boundary = mesh.topology.incidences[-1]
    relation = boundary.relation
    valid = np.asarray(relation.valid)
    faces = np.asarray(relation.source_indices)[valid]
    counts = np.bincount(faces, minlength=mesh.entity_set(2).count)
    (interior,) = np.flatnonzero(counts == 2)
    np.testing.assert_array_equal(counts[counts != 2], np.ones(7, dtype=np.int64))
    np.testing.assert_allclose(
        np.sum(np.asarray(boundary.signs)[valid][faces == interior]), 0.0, atol=0.0
    )
    from phydrax.meshing import evaluate_cell_quality

    quality = evaluate_cell_quality(mesh)
    measures = dict(
        zip(
            np.asarray(quality.cell_global_ids).tolist(),
            np.asarray(quality.measures).tolist(),
            strict=True,
        )
    )
    np.testing.assert_allclose((measures[31], measures[47]), (1.0 / 3.0, 3.0 / 16.0))


@pytest.mark.parametrize(
    ("kind", "family"),
    (
        ("simplex:1", "interval"),
        ("simplex:2", "triangle"),
        ("simplex:3", "tetrahedron"),
        ("tensor:1", "interval"),
        ("tensor:2", "quadrilateral"),
        ("tensor:3", "hexahedron"),
    ),
    ids=("simplex-1", "simplex-2", "simplex-3", "tensor-1", "tensor-2", "tensor-3"),
)
def test_named_cell_family_preserves_qualified_reference_identity(
    kind: str, family: str
) -> None:
    qualified = reference_cell_topology(kind)
    named = reference_cell_topology(family)
    assert qualified.family == named.family == family
    assert qualified.name == kind
    assert qualified.topology_id != named.topology_id
    assert qualified.vertices == named.vertices
    assert qualified.entities == named.entities


def test_higher_dimensional_reference_is_not_a_lower_dimensional_family() -> None:
    reference = reference_cell_topology("simplex:4")
    assert reference.family == "simplex:4"
    assert reference.family != reference_cell_topology("tetrahedron").family


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
    assert isinstance(geometry, CompiledGeometry)
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
