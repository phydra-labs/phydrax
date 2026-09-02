#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_point_cloud_complexes_are_canonical_face_closed_and_bounded():
    points = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.2]])
    policy = phx.topology.PointCloudComplexPolicy(
        maximum_dimension=2, maximum_simplices=64
    )
    vr = phx.topology.vietoris_rips_complex(points, 1.1, policy=policy)
    cech = phx.topology.cech_complex(points, 0.8, policy=policy)
    alpha = phx.topology.alpha_complex(points, 0.8, policy=policy)
    for result in (vr, cech, alpha):
        assert bool(result.certified)
        assert result.topology.dimension >= 1
        assert result.simplices[0].shape == (4, 1)

    with pytest.raises(ValueError, match="ambiguous"):
        phx.topology.vietoris_rips_complex(points[:2], 1.0, policy=policy)
    with pytest.raises(ValueError, match="maximum_simplices"):
        phx.topology.vietoris_rips_complex(
            points,
            2.0,
            policy=phx.topology.PointCloudComplexPolicy(
                maximum_dimension=2, maximum_simplices=4
            ),
        )


def test_alpha_complex_admits_faces_before_their_delaunay_coface():
    points = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    result = phx.topology.alpha_complex(
        points,
        1.1,
        policy=phx.topology.PointCloudComplexPolicy(maximum_dimension=2),
    )

    assert {tuple(simplex) for simplex in result.simplices[1].tolist()} == {
        (0, 1),
        (0, 2),
    }
    assert len(result.simplices) == 2
    assert result.simplices[0].shape == (3, 1)

    obtuse = phx.topology.alpha_complex(
        jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.5, 0.2]]),
        1.1,
        policy=phx.topology.PointCloudComplexPolicy(maximum_dimension=2),
    )
    assert {tuple(simplex) for simplex in obtuse.simplices[1].tolist()} == {
        (0, 2),
        (1, 2),
    }
    assert len(obtuse.simplices) == 2


def test_finite_multiparameter_module_does_not_claim_a_barcode():
    field = phx.topology.PrimeField(2)
    module = phx.topology.FinitePersistenceModule(
        jnp.asarray([1, 2, 1]),
        jnp.asarray([[0, 1], [2, 1]]),
        (jnp.asarray([[1], [0]]), jnp.asarray([[0], [1]])),
        field=field,
    )
    result = phx.topology.compute_multiparameter_persistence(module, rank_edges=(0, 1))
    assert jnp.array_equal(result.hilbert_dimensions, jnp.asarray([1, 2, 1]))
    assert jnp.array_equal(result.rank_queries, jnp.asarray([1, 1]))
    assert not result.barcode_claimed


def test_mixed_zigzag_interval_decomposition_reconstructs_dimensions_and_ranks():
    field = phx.topology.PrimeField(3)
    result = phx.topology.compute_zigzag_intervals(
        (1, 2, 1),
        (jnp.asarray([[1], [0]]), jnp.asarray([[0], [1]])),
        ("forward", "backward"),
        coefficients=field,
    )
    assert bool(result.valid)
    assert jnp.array_equal(result.reconstructed_dimensions, jnp.asarray([1, 2, 1]))
    assert jnp.array_equal(result.reconstructed_edge_ranks, jnp.asarray([1, 1]))
    assert {tuple(value) for value in result.intervals.tolist()} == {(0, 1), (1, 2)}


def test_explicit_diagonal_cup_product_and_constant_cellular_sheaf():
    points = jnp.asarray([[0.0], [1.0]])
    interval = phx.topology.vietoris_rips_complex(
        points,
        1.1,
        policy=phx.topology.PointCloudComplexPolicy(maximum_dimension=1),
    )
    diagonal = phx.topology.CellDiagonalApproximation(
        interval.topology,
        1,
        0,
        1,
        jnp.asarray([0]),
        jnp.asarray([0]),
        jnp.asarray([0]),
        jnp.asarray([1]),
    )
    product = phx.topology.cup_product(
        jnp.asarray([1, 0]),
        jnp.asarray([2]),
        diagonal,
        coefficients=phx.topology.PrimeField(3),
    )
    assert jnp.array_equal(product, jnp.asarray([2]))

    sheaf = phx.topology.CellularSheaf(
        interval.topology,
        (jnp.asarray([1, 1]), jnp.asarray([1])),
        (jnp.ones((1, 1), dtype=int), jnp.ones((1, 1), dtype=int)),
        field=phx.topology.PrimeField(2),
    )
    assert jnp.array_equal(sheaf.cohomology_dimensions(), jnp.asarray([1, 0]))


def test_cellular_sheaf_rejects_incompatible_composed_restrictions():
    triangle = phx.topology.vietoris_rips_complex(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        2.0,
        policy=phx.topology.PointCloudComplexPolicy(maximum_dimension=2),
    )
    dimensions = (
        jnp.ones((3,), dtype=int),
        jnp.ones((3,), dtype=int),
        jnp.ones((1,), dtype=int),
    )
    restrictions = [jnp.ones((1, 1), dtype=int) for _ in range(9)]
    restrictions[0] = jnp.zeros((1, 1), dtype=int)

    with pytest.raises(ValueError, match="incompatible restriction routes"):
        phx.topology.CellularSheaf(
            triangle.topology,
            dimensions,
            restrictions,
            field=phx.topology.PrimeField(2),
        )

    valid = phx.topology.CellularSheaf(
        triangle.topology,
        dimensions,
        tuple(jnp.ones((1, 1), dtype=int) for _ in range(9)),
        field=phx.topology.PrimeField(2),
    )
    assert jnp.array_equal(valid.cohomology_dimensions(), jnp.asarray([1, 0, 0]))


def test_cellular_sheaf_rejects_nonzero_assembled_coboundary():
    vertices = phx.discretization.EntitySet("sheaf-vertices", 0, jnp.asarray([0]))
    edges = phx.discretization.EntitySet("sheaf-edges", 1, jnp.asarray([0]))
    faces = phx.discretization.EntitySet("sheaf-faces", 2, jnp.asarray([0]))
    vertex_edge = phx.discretization.OrientedIncidence(
        1,
        vertices,
        edges,
        phx.sparse.EdgeRelation(
            jnp.asarray([0]), jnp.asarray([0]), source_size=1, target_size=1
        ),
        jnp.asarray([1]),
    )
    edge_face = phx.discretization.OrientedIncidence(
        2,
        edges,
        faces,
        phx.sparse.EdgeRelation(
            jnp.asarray([0]), jnp.asarray([0]), source_size=1, target_size=1
        ),
        jnp.asarray([1]),
    )
    invalid_topology = phx.discretization.CellComplexTopology(
        (vertices, edges, faces),
        (vertex_edge, edge_face),
        validate=False,
    )

    with pytest.raises(ValueError, match="nonzero consecutive coboundary"):
        phx.topology.CellularSheaf(
            invalid_topology,
            tuple(jnp.asarray([1]) for _ in range(3)),
            tuple(jnp.ones((1, 1), dtype=int) for _ in range(2)),
            field=phx.topology.PrimeField(2),
        )


def test_filtered_chain_spectral_page_is_e1_homology_and_flags_extensions_unresolved():
    field = phx.topology.PrimeField(2)
    complex = phx.topology.FilteredChainComplex(
        (jnp.asarray([[1], [1]]),),
        (jnp.asarray([0, 0]), jnp.asarray([0])),
        field=field,
    )
    result = phx.topology.compute_spectral_sequence(complex, maximum_page=2)
    assert bool(result.convergence_certified)
    assert result.stabilized_page == 1
    assert not result.extension_resolved
    assert jnp.array_equal(result.page_dimensions[1, 0], jnp.asarray([1, 0]))


def test_filtered_chain_spectral_sequence_computes_an_induced_d2():
    complex = phx.topology.FilteredChainComplex(
        (jnp.asarray([[1, 0], [1, 1]]),),
        (jnp.asarray([0, 2]), jnp.asarray([2, 2])),
        field=phx.topology.PrimeField(2),
    )

    result = phx.topology.compute_spectral_sequence(complex, maximum_page=3)

    assert jnp.array_equal(result.page_dimensions[0], jnp.asarray([[1, 0], [1, 2]]))
    assert jnp.array_equal(result.page_dimensions[1], jnp.asarray([[1, 0], [0, 1]]))
    assert jnp.array_equal(result.page_dimensions[2], result.page_dimensions[1])
    assert jnp.array_equal(result.page_dimensions[3], jnp.zeros((2, 2), dtype=int))
    assert result.differential_ranks[0, 1, 1] == 1
    assert result.differential_ranks[2, 1, 1] == 1
    assert jnp.sum(result.differential_ranks[1]) == 0
    assert result.stabilized_page == 3
    assert bool(result.convergence_certified)

    truncated = phx.topology.compute_spectral_sequence(complex, maximum_page=2)
    assert truncated.stabilized_page == -1
    assert not bool(truncated.convergence_certified)


def test_zero_filtered_differential_is_certified_stable_at_e0():
    complex = phx.topology.FilteredChainComplex(
        (jnp.zeros((1, 1), dtype=int),),
        (jnp.asarray([0]), jnp.asarray([3])),
        field=phx.topology.PrimeField(3),
    )

    result = phx.topology.compute_spectral_sequence(complex, maximum_page=1)

    assert result.stabilized_page == 0
    assert bool(result.convergence_certified)
    assert jnp.array_equal(result.page_dimensions[0], result.page_dimensions[1])
    assert jnp.sum(result.differential_ranks) == 0
