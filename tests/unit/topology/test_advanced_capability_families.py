#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

import phydrax as phx


def test_advanced_capability_families_scenario_1() -> None:
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
    field = phx.topology.PrimeField(2)

    with pytest.raises(ValueError, match="noncommuting path maps"):
        phx.topology.FinitePersistenceModule(
            jnp.asarray([1, 1, 1]),
            jnp.asarray([[0, 1], [1, 2], [0, 2]]),
            (
                jnp.asarray([[1]]),
                jnp.asarray([[1]]),
                jnp.asarray([[0]]),
            ),
            field=field,
        )

    with pytest.raises(ValueError, match="noncommuting path maps"):
        phx.topology.FinitePersistenceModule(
            jnp.asarray([1, 1, 1, 1]),
            jnp.asarray([[0, 1], [1, 2], [2, 3], [0, 3]]),
            (
                jnp.asarray([[1]]),
                jnp.asarray([[1]]),
                jnp.asarray([[1]]),
                jnp.asarray([[0]]),
            ),
            field=field,
        )


def test_advanced_capability_families_scenario_2() -> None:
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
        left_topology_id=interval.topology.topology_id,
        right_topology_id=interval.topology.topology_id,
    )
    assert jnp.array_equal(product, jnp.asarray([2]))

    empty_diagonal = phx.topology.CellDiagonalApproximation(
        interval.topology,
        0,
        0,
        0,
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int32),
    )
    zero_product = phx.topology.cup_product(
        jnp.asarray([1, 2]),
        jnp.asarray([2, 1]),
        empty_diagonal,
        coefficients=phx.topology.PrimeField(3),
        left_topology_id=interval.topology.topology_id,
        right_topology_id=interval.topology.topology_id,
    )
    assert jnp.array_equal(zero_product, jnp.zeros((2,), dtype=jnp.int32))
    with pytest.raises(ValueError, match="Left cochain length"):
        phx.topology.cup_product(
            jnp.asarray([1]),
            jnp.asarray([2, 1]),
            empty_diagonal,
            coefficients=phx.topology.PrimeField(3),
            left_topology_id=interval.topology.topology_id,
            right_topology_id=interval.topology.topology_id,
        )

    sheaf = phx.topology.CellularSheaf(
        interval.topology,
        (jnp.asarray([1, 1]), jnp.asarray([1])),
        (jnp.ones((1, 1), dtype="int64"), jnp.ones((1, 1), dtype="int64")),
        field=phx.topology.PrimeField(2),
    )
    assert jnp.array_equal(sheaf.cohomology_dimensions(), jnp.asarray([1, 0]))
    triangle = phx.topology.vietoris_rips_complex(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        2.0,
        policy=phx.topology.PointCloudComplexPolicy(maximum_dimension=2),
    )
    dimensions = (
        jnp.ones((3,), dtype="int64"),
        jnp.ones((3,), dtype="int64"),
        jnp.ones((1,), dtype="int64"),
    )
    restrictions = [jnp.ones((1, 1), dtype="int64") for _ in range(9)]
    restrictions[0] = jnp.zeros((1, 1), dtype="int64")

    with pytest.raises(ValueError, match="nonzero consecutive coboundary"):
        phx.topology.CellularSheaf(
            triangle.topology,
            dimensions,
            restrictions,
            field=phx.topology.PrimeField(2),
        )

    valid = phx.topology.CellularSheaf(
        triangle.topology,
        dimensions,
        tuple(jnp.ones((1, 1), dtype="int64") for _ in range(9)),
        field=phx.topology.PrimeField(2),
    )
    assert jnp.array_equal(valid.cohomology_dimensions(), jnp.asarray([1, 0, 0]))
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
            tuple(jnp.ones((1, 1), dtype="int64") for _ in range(2)),
            field=phx.topology.PrimeField(2),
        )


def test_advanced_capability_families_scenario_3() -> None:
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
    complex = phx.topology.FilteredChainComplex(
        (jnp.asarray([[1, 0], [1, 1]]),),
        (jnp.asarray([0, 2]), jnp.asarray([2, 2])),
        field=phx.topology.PrimeField(2),
    )

    result = phx.topology.compute_spectral_sequence(complex, maximum_page=3)

    assert jnp.array_equal(result.page_dimensions[0], jnp.asarray([[1, 0], [1, 2]]))
    assert jnp.array_equal(result.page_dimensions[1], jnp.asarray([[1, 0], [0, 1]]))
    assert jnp.array_equal(result.page_dimensions[2], result.page_dimensions[1])
    assert jnp.array_equal(result.page_dimensions[3], jnp.zeros((2, 2), dtype="int64"))
    assert result.differential_ranks[0, 1, 1] == 1
    assert result.differential_ranks[2, 1, 1] == 1
    assert jnp.sum(result.differential_ranks[1]) == 0
    assert result.stabilized_page == 3
    assert bool(result.convergence_certified)

    truncated = phx.topology.compute_spectral_sequence(complex, maximum_page=2)
    assert truncated.stabilized_page == -1
    assert not bool(truncated.convergence_certified)
    complex = phx.topology.FilteredChainComplex(
        (jnp.zeros((1, 1), dtype="int64"),),
        (jnp.asarray([0]), jnp.asarray([3])),
        field=phx.topology.PrimeField(3),
    )

    result = phx.topology.compute_spectral_sequence(complex, maximum_page=1)

    assert result.stabilized_page == 0
    assert bool(result.convergence_certified)
    assert jnp.array_equal(result.page_dimensions[0], result.page_dimensions[1])
    assert jnp.sum(result.differential_ranks) == 0


@pytest.mark.parametrize(
    "builder",
    (
        phx.topology.vietoris_rips_complex,
        phx.topology.cech_complex,
        phx.topology.alpha_complex,
    ),
)
def test_point_cloud_identity_includes_content_and_filtration(
    builder: Callable[[ArrayLike, float], phx.topology.PointCloudComplexResult],
) -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.2, 0.8]], dtype=np.float64)
    original = builder(points, 1.7)
    identical = builder(points.copy(), 1.7)
    translated = builder(points + np.asarray([2.0, -3.0]), 1.7)
    changed_radius = builder(points, 1.8)
    assert original.topology.topology_id == identical.topology.topology_id
    assert original.topology.topology_id != translated.topology.topology_id
    assert original.topology.topology_id != changed_radius.topology.topology_id


def test_cup_large_prime_matches_python_integer_oracle() -> None:
    prime = 2_147_483_629
    interval = phx.topology.vietoris_rips_complex(
        np.asarray([[0.0], [1.0]], dtype=np.float64), 1.1
    )
    topology = interval.topology
    coefficient_values = np.asarray(
        [prime - 1, prime - 2, -7, 1_073_741_824, -prime + 3], dtype=np.int64
    )
    left = np.asarray([prime - 2, -prime - 5], dtype=np.int64)
    right = np.asarray([prime - 3, 2 * prime - 1], dtype=np.int64)
    left_cells = np.asarray([0, 1, 0, 1, 0], dtype=np.int32)
    right_cells = np.asarray([1, 0, 0, 1, 1], dtype=np.int32)
    source_cells = np.asarray([0, 0, 1, 1, 0], dtype=np.int32)
    diagonal = phx.topology.CellDiagonalApproximation(
        topology, 0, 0, 0, source_cells, left_cells, right_cells, coefficient_values
    )
    expected = [0, 0]
    for source, a, b, c in zip(
        source_cells, left_cells, right_cells, coefficient_values, strict=True
    ):
        expected[int(source)] = (
            expected[int(source)] + int(c) * int(left[a]) * int(right[b])
        ) % prime
    result = phx.topology.cup_product(
        left,
        right,
        diagonal,
        coefficients=phx.topology.PrimeField(prime),
        left_topology_id=topology.topology_id,
        right_topology_id=topology.topology_id,
    )
    np.testing.assert_array_equal(result, np.asarray(expected, dtype=np.int32))


def test_cup_refuses_equal_size_distinct_topologies() -> None:
    first = phx.topology.vietoris_rips_complex(
        np.asarray([[0.0], [1.0]], dtype=np.float64), 1.1
    )
    second = phx.topology.vietoris_rips_complex(
        np.asarray([[2.0], [3.0]], dtype=np.float64), 1.1
    )
    diagonal = phx.topology.CellDiagonalApproximation(
        first.topology,
        0,
        0,
        0,
        np.asarray([0]),
        np.asarray([0]),
        np.asarray([0]),
        np.asarray([1]),
    )
    for left_id, right_id in (
        (first.topology.topology_id, second.topology.topology_id),
        (second.topology.topology_id, second.topology.topology_id),
    ):
        with pytest.raises(ValueError, match="topology_id"):
            phx.topology.cup_product(
                np.asarray([1, 2]),
                np.asarray([3, 4]),
                diagonal,
                coefficients=phx.topology.PrimeField(5),
                left_topology_id=left_id,
                right_topology_id=right_id,
            )


def test_one_vertex_circle_preserves_each_restriction_occurrence() -> None:
    vertices = phx.discretization.EntitySet("circle:vertices", 0, np.asarray([0]))
    edges = phx.discretization.EntitySet("circle:edges", 1, np.asarray([0]))
    incidence = phx.discretization.OrientedIncidence(
        1,
        vertices,
        edges,
        phx.sparse.EdgeRelation(
            np.asarray([0, 0]),
            np.asarray([0, 0]),
            source_size=1,
            target_size=1,
        ),
        np.asarray([-1, 1], dtype=np.int32),
    )
    topology = phx.discretization.CellComplexTopology((vertices, edges), (incidence,))
    stalks = (np.asarray([1]), np.asarray([1]))
    constant = phx.topology.CellularSheaf(
        topology,
        stalks,
        (np.asarray([[1]]), np.asarray([[1]])),
        field=phx.topology.PrimeField(5),
    )
    monodromy = phx.topology.CellularSheaf(
        topology,
        stalks,
        (np.asarray([[1]]), np.asarray([[2]])),
        field=phx.topology.PrimeField(5),
    )
    np.testing.assert_array_equal(constant.cohomology_dimensions(), np.asarray([1, 1]))
    np.testing.assert_array_equal(monodromy.cohomology_dimensions(), np.asarray([0, 0]))


def test_klein_bottle_route_orientation_system_has_top_cohomology() -> None:
    vertices = phx.discretization.EntitySet("klein:v", 0, np.asarray([0]))
    edges = phx.discretization.EntitySet("klein:e", 1, np.asarray([0, 1]))
    faces = phx.discretization.EntitySet("klein:f", 2, np.asarray([0]))
    vertex_edge = phx.discretization.OrientedIncidence(
        1,
        vertices,
        edges,
        phx.sparse.EdgeRelation(
            np.asarray([0, 0, 0, 0]),
            np.asarray([0, 0, 1, 1]),
            source_size=1,
            target_size=2,
        ),
        np.asarray([-1, 1, -1, 1]),
    )
    # The attaching word a b a^-1 b reverses orientation along a.
    edge_face = phx.discretization.OrientedIncidence(
        2,
        edges,
        faces,
        phx.sparse.EdgeRelation(
            np.asarray([0, 1, 0, 1]),
            np.asarray([0, 0, 0, 0]),
            source_size=2,
            target_size=1,
        ),
        np.asarray([1, 1, -1, 1]),
    )
    topology = phx.discretization.CellComplexTopology(
        (vertices, edges, faces), (vertex_edge, edge_face)
    )
    stalks = (np.asarray([1]), np.asarray([1, 1]), np.asarray([1]))
    field = phx.topology.PrimeField(5)
    constant = phx.topology.CellularSheaf(
        topology,
        stalks,
        tuple(np.asarray([[1]]) for _ in range(8)),
        field=field,
    )
    orientation = phx.topology.CellularSheaf(
        topology,
        stalks,
        tuple(np.asarray([[value]]) for value in (1, -1, 1, 1, 1, 1, 1, -1)),
        field=field,
    )
    np.testing.assert_array_equal(constant.cohomology_dimensions(), np.asarray([1, 1, 0]))
    np.testing.assert_array_equal(
        orientation.cohomology_dimensions(), np.asarray([0, 1, 1])
    )
