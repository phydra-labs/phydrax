#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._integration_guardrails import CoreAbstractionRegistry
from tests.unit.topology._fixtures import (
    filled_triangle_topology,
    projective_plane_topology,
)


def test_full_compaction_excludes_inactive_capacity():
    vertices = phx.discretization.EntitySet(
        "vertices",
        0,
        np.arange(3),
        active_mask=np.asarray([True, True, False]),
    )
    edges = phx.discretization.EntitySet(
        "edges",
        1,
        np.arange(2),
        active_mask=np.asarray([True, False]),
    )
    relation = phx.sparse.EdgeRelation(
        np.asarray([0, 1, 1, 2]),
        np.asarray([0, 0, 1, 1]),
        source_size=3,
        target_size=2,
    )
    incidence = phx.discretization.OrientedIncidence(
        1,
        vertices,
        edges,
        relation,
        np.asarray([-1.0, 1.0, -1.0, 1.0]),
    )
    topology = phx.discretization.CellComplexTopology((vertices, edges), (incidence,))
    complex = phx.topology.CellSubcomplex.full(topology)

    assert complex.layout.counts == (2, 1)
    np.testing.assert_array_equal(complex.layout.ambient_to_compact[0], [0, 1, -1])
    np.testing.assert_array_equal(complex.layout.compact_to_ambient[1], [0])
    result = phx.topology.compute_homology(
        complex,
        coefficients=phx.topology.PrimeField(2),
    )
    assert result.dimensions == (1, 0)


def test_nonclosed_active_masks_fail_before_algebra():
    vertices = phx.discretization.EntitySet(
        "vertices",
        0,
        np.arange(2),
        active_mask=np.asarray([True, False]),
    )
    edges = phx.discretization.EntitySet("edges", 1, np.arange(1))
    relation = phx.sparse.EdgeRelation(
        np.asarray([0, 1]),
        np.asarray([0, 0]),
        source_size=2,
        target_size=1,
    )
    incidence = phx.discretization.OrientedIncidence(
        1,
        vertices,
        edges,
        relation,
        np.asarray([-1.0, 1.0]),
    )
    topology = phx.discretization.CellComplexTopology((vertices, edges), (incidence,))

    with pytest.raises(ValueError, match="do not form a subcomplex"):
        phx.topology.CellSubcomplex.full(topology)


def test_boundary_subcomplex_and_relative_disk_homology():
    topology = filled_triangle_topology()
    disk = phx.topology.CellSubcomplex.full(topology)
    boundary = phx.topology.CellSubcomplex.from_subsets(topology, "boundary")
    pair = phx.topology.CellComplexPair(disk, boundary)
    result = phx.topology.compute_homology(
        pair,
        coefficients=phx.topology.PrimeField(3),
        representatives="cycles",
    )
    rational = phx.topology.compute_betti_dimensions(
        pair,
        coefficients=phx.topology.RationalField(),
    )

    assert result.dimensions == (0, 0, 1)
    assert rational.dimensions == (0, 0, 1)
    representative = result.degree(2).cycles
    assert representative is not None
    assert representative.generator_count == 1
    assert representative.nonzero_count == 1


def test_relative_subcomplex_must_be_contained_in_ambient():
    topology = filled_triangle_topology()
    full = phx.topology.CellSubcomplex.full(topology)
    empty = phx.topology.CellSubcomplex(
        topology,
        tuple(jnp.zeros_like(value, dtype=bool) for value in full.masks),
    )
    with pytest.raises(ValueError, match="contained"):
        phx.topology.CellComplexPair(empty, full)


def test_reduced_homology_uses_explicit_augmentation():
    vertex = phx.discretization.EntitySet("vertices", 0, np.asarray([0]))
    point = phx.discretization.CellComplexTopology((vertex,), ())
    result = phx.topology.compute_homology(
        point,
        coefficients=phx.topology.PrimeField(2),
        reduced=True,
    )

    assert tuple(value.degree for value in result.degrees) == (-1, 0)
    assert result.dimensions == (0, 0)


def test_empty_reduced_complex_has_degree_minus_one_class():
    vertex = phx.discretization.EntitySet(
        "vertices",
        0,
        np.asarray([0]),
        active_mask=np.asarray([False]),
    )
    empty = phx.discretization.CellComplexTopology((vertex,), ())
    result = phx.topology.compute_homology(
        empty,
        coefficients=phx.topology.PrimeField(5),
        reduced=True,
    )

    assert result.degree(-1).dimension == 1
    assert result.degree(0).dimension == 0


def test_reduced_relative_homology_is_rejected():
    topology = filled_triangle_topology()
    pair = phx.topology.CellComplexPair(
        phx.topology.CellSubcomplex.full(topology),
        phx.topology.CellSubcomplex.from_subsets(topology, "boundary"),
    )
    with pytest.raises(ValueError, match="Reduced relative"):
        phx.topology.compute_homology(
            pair,
            coefficients=phx.topology.PrimeField(2),
            reduced=True,
        )


def test_prime_field_is_explicit_and_exact():
    field = phx.topology.PrimeField(2_147_483_647)
    assert field.multiply(field.modulus - 1, field.modulus - 1) == 1
    assert field.divide(7, 7) == 1
    with pytest.raises(ValueError, match="prime"):
        phx.topology.PrimeField(15)
    with pytest.raises(ZeroDivisionError):
        field.inverse(0)


def test_projective_plane_homology_depends_on_coefficient_field():
    topology = projective_plane_topology()
    mod_two = phx.topology.compute_homology(
        topology,
        coefficients=phx.topology.PrimeField(2),
    )
    mod_three = phx.topology.compute_homology(
        topology,
        coefficients=phx.topology.PrimeField(3),
    )
    rational = phx.topology.compute_betti_dimensions(
        topology,
        coefficients=phx.topology.RationalField(),
    )

    assert mod_two.dimensions == (1, 1, 1)
    assert mod_three.dimensions == (1, 0, 0)
    assert rational.dimensions == (1, 0, 0)


def test_cycle_and_cocycle_representatives_match_betti_dimensions():
    topology = projective_plane_topology()
    result = phx.topology.compute_homology(
        topology,
        coefficients=phx.topology.PrimeField(2),
        representatives="both",
    )

    for degree in result.degrees:
        assert degree.cycles is not None
        assert degree.cocycles is not None
        assert degree.cycles.generator_count == degree.dimension
        assert degree.cocycles.generator_count == degree.dimension


def test_reorientation_preserves_dimensions():
    topology = filled_triangle_topology()
    entity_sets = topology.entity_sets
    incidences = tuple(
        phx.discretization.OrientedIncidence(
            incidence.degree,
            entity_sets[incidence.degree - 1],
            entity_sets[incidence.degree],
            incidence.relation,
            -np.asarray(incidence.signs),
        )
        for incidence in topology.incidences
    )
    reoriented = phx.discretization.CellComplexTopology(entity_sets, incidences)
    original = phx.topology.compute_homology(
        topology,
        coefficients=phx.topology.PrimeField(3),
    )
    transformed = phx.topology.compute_homology(
        reoriented,
        coefficients=phx.topology.PrimeField(3),
    )
    assert transformed.dimensions == original.dimensions


def test_resource_limit_fails_without_partial_result():
    with pytest.raises(phx.topology.TopologyResourceError, match="max_cells"):
        phx.topology.compute_homology(
            filled_triangle_topology(),
            coefficients=phx.topology.PrimeField(2),
            resources=phx.topology.TopologyResourcePolicy(max_cells=2),
        )


def test_canonical_ownership_is_explicit():
    registry = CoreAbstractionRegistry()
    assert registry.owner("cell_complex") == "phydrax.discretization.CellComplexTopology"
    assert registry.owner("homology") == "phydrax.topology.compute_homology"
    assert registry.owner("persistence") == "phydrax.topology.compute_persistence"
