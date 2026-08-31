#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tests.unit.topology._fixtures import (
    filled_triangle_filtration,
    filled_triangle_topology,
    filled_triangle_vertex_support,
    projective_plane_topology,
)


def test_exact_integer_matrix_and_identity_chain_map():
    matrix = phx.topology.ExactIntegerCOO(
        2,
        2,
        np.asarray([0, 1, 0]),
        np.asarray([0, 1, 0]),
        (2, 3, -1),
        source_id="x",
        target_id="y",
    )
    assert matrix.entries() == ((0, 0, 1), (1, 1, 3))
    assert matrix.apply_integer((2, 4)) == (2, 12)

    complex = phx.topology.CellSubcomplex.full(filled_triangle_topology())
    identity = phx.topology.CellularChainMap.identity(complex)
    assert identity.compose(identity).map_id
    assert all(
        value.nonzero_count == count
        for value, count in zip(identity.degree_maps, complex.layout.counts, strict=True)
    )


def test_induced_identity_and_mapping_cone_are_exact():
    complex = phx.topology.CellSubcomplex.full(filled_triangle_topology())
    identity = phx.topology.CellularChainMap.identity(complex)
    homology = phx.topology.compute_homology(
        complex,
        coefficients=phx.topology.PrimeField(2),
        representatives="both",
    )
    induced = phx.topology.compute_induced_topology_map(identity, homology, homology)
    cone = phx.topology.compute_mapping_cone_homology(
        identity,
        coefficients=phx.topology.PrimeField(2),
    )

    np.testing.assert_array_equal(induced.homology_maps[0].matrix, [[1]])
    assert cone.acyclic


def test_filtered_identity_has_zero_shift():
    _, complex, filtration = filled_triangle_filtration()
    filtered = phx.topology.FilteredCellularChainMap(
        phx.topology.CellularChainMap.identity(complex),
        filtration,
        filtration,
    )
    assert filtered.epsilon == 0.0
    homotopies = tuple(
        phx.topology.ExactIntegerCOO.zero(
            complex.layout.counts[degree + 1] if degree < complex.max_degree else 0,
            complex.layout.counts[degree],
            source_id=phx.topology.chain_coordinate_id(complex.subcomplex_id, degree),
            target_id=phx.topology.chain_coordinate_id(complex.subcomplex_id, degree + 1),
        )
        for degree in range(complex.max_degree + 1)
    )
    contraction = phx.topology.CellularChainContraction(
        complex,
        complex,
        phx.topology.CellularChainMap.identity(complex),
        phx.topology.CellularChainMap.identity(complex),
        homotopies,
    )
    filtered_contraction = phx.topology.FilteredCellularChainContraction(
        contraction,
        filtration,
        filtration,
    )
    assert filtered_contraction.epsilon == 0.0


def test_extended_persistence_and_persistent_cohomology():
    _, _, filtration = filled_triangle_filtration()
    extended = phx.topology.compute_extended_persistence(
        filtration,
        coefficients=phx.topology.PrimeField(2),
    )
    cohomology = phx.topology.compute_persistent_cohomology(
        filtration,
        coefficients=phx.topology.PrimeField(2),
    )

    assert extended.ordinary.interval_count == 1
    assert extended.extended_positive.interval_count >= 1
    assert cohomology.terminal_cocycles[0].generator_count == 1
    assert cohomology.annotations[0].essential_pair_indices.shape == (1,)


def test_field_snapshot_series_and_ensemble_summary():
    topology = filled_triangle_topology()
    complex = phx.topology.CellSubcomplex.full(topology)
    plan = phx.topology.FieldTopologyPlan(
        complex,
        filled_triangle_vertex_support(topology),
        phx.topology.PrimeField(2),
        jnp.asarray([0.25, 0.75, 1.5]),
    )
    first = plan.snapshot(jnp.asarray([0.0, 0.5, 1.0]), field_id="first")
    second = plan.snapshot(jnp.asarray([0.1, 0.6, 1.1]), field_id="second")
    series = phx.topology.FieldTopologySeries((first, second), jnp.asarray([0.0, 1.0]))
    summary = phx.uq.TopologyEnsembleSummary((first, second))

    assert series.betti_history.shape == (2, 3, 3)
    assert summary.mean_betti.shape == (3, 3)


def test_diagram_distances_are_zero_on_identity():
    _, _, filtration = filled_triangle_filtration()
    diagram = phx.topology.compute_persistence(
        filtration,
        coefficients=phx.topology.PrimeField(2),
    ).diagram()
    wasserstein = phx.topology.diagram_wasserstein_distance(diagram, diagram)
    bottleneck = phx.topology.diagram_bottleneck_distance(diagram, diagram)
    sliced = phx.topology.diagram_sliced_wasserstein_distance(
        diagram,
        diagram,
        degree=1,
        num_directions=8,
    )

    assert bool(wasserstein.valid)
    assert float(wasserstein.distance) == pytest.approx(0.0)
    assert bool(bottleneck.valid)
    assert float(bottleneck.distance) == pytest.approx(0.0)
    assert bool(sliced.valid)
    assert float(sliced.distance) == pytest.approx(0.0)


def test_rational_and_integral_homology_distinguish_torsion():
    complex = phx.topology.CellSubcomplex.full(projective_plane_topology())
    rational = phx.topology.compute_rational_homology_basis(complex)
    integral = phx.topology.compute_integral_homology(complex)

    assert rational.degree(1).generator_count == 0
    assert integral.degree(1).free_rank == 0
    assert integral.degree(1).torsion_invariants == (2,)


def test_elementary_morse_cancellation_preserves_interval_homology():
    boundary_zero = phx.topology.ExactIntegerCOO.zero(
        0,
        2,
        source_id="interval:degree:0",
        target_id="interval:degree:-1",
    )
    boundary_one = phx.topology.ExactIntegerCOO(
        2,
        1,
        np.asarray([0, 1]),
        np.asarray([0, 0]),
        (-1, 1),
        source_id="interval:degree:1",
        target_id="interval:degree:0",
    )
    chain = phx.topology.ExactChainComplex(
        (boundary_zero, boundary_one), complex_id="interval"
    )
    reduced = phx.topology.cancel_unit_pair(chain, 1, 0, 0)
    assert reduced.reduced.counts == (1, 0)


def test_vineyard_and_zigzag_topology_evolution():
    topology, complex, filtration = filled_triangle_filtration()
    vineyard = phx.topology.compute_vineyard(
        (filtration, filtration),
        jnp.asarray([0.0, 1.0]),
        coefficients=phx.topology.PrimeField(2),
    )
    empty_masks = tuple(
        np.zeros_like(np.asarray(mask), dtype=bool) for mask in complex.masks
    )
    operations = (
        phx.topology.ZigzagCellOperation("insert", 0, 0),
        phx.topology.ZigzagCellOperation("remove", 0, 0),
    )
    zigzag = phx.topology.compute_zigzag_topology(
        complex,
        empty_masks,
        operations,
        coefficients=phx.topology.PrimeField(2),
    )
    insertion_operations = tuple(
        phx.topology.ZigzagCellOperation("insert", degree, cell)
        for degree, count in enumerate(complex.layout.counts)
        for cell in range(count)
    )
    monotone = phx.topology.compute_monotone_zigzag_intervals(
        complex,
        empty_masks,
        insertion_operations,
        coefficients=phx.topology.PrimeField(2),
    )

    assert len(vineyard.snapshots) == 2
    np.testing.assert_array_equal(zigzag.betti_history[:, 0], [0, 1, 0])
    assert topology.topology_id == complex.topology.topology_id
    assert monotone.persistence.pairing.pair_count >= 1


def test_local_homology_and_certified_implicit_evidence():
    complex = phx.topology.CellSubcomplex.full(filled_triangle_topology())
    local = phx.topology.compute_cell_local_homology(
        complex,
        2,
        0,
        coefficients=phx.topology.PrimeField(2),
    )
    cover = phx.geometry.CertifiedImplicitCover(
        jnp.asarray([[[0.0], [1.0]], [[1.0], [2.0]]]),
        jnp.asarray([-1.0, 1.0]),
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([0.5, 0.0]),
    )
    certified = phx.geometry.CertifiedImplicitTopology(
        cover,
        complex.topology,
        theorem="regular-value-box-cover",
    )

    assert local.homology.degree(2).dimension == 1
    assert bool(certified.certified)
