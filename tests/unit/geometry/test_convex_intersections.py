import numpy as np
import pytest

from phydrax.geometry._convex_intersections import (
    intersect_convex_polygons,
    IntersectionStatus,
)


SQUARE = np.asarray(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float64,
)
TRIANGLE = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float64)


def test_identity_returns_canonical_polygon_area_and_centroid():
    result = intersect_convex_polygons(
        TRIANGLE, TRIANGLE, source_id="left", target_id="right"
    )

    assert result.status is IntersectionStatus.SUCCESS
    assert result.vertices.shape == (3, 2)
    np.testing.assert_allclose(result.vertices, TRIANGLE)
    assert result.area == pytest.approx(2.0)
    np.testing.assert_allclose(result.centroid, [2.0 / 3.0, 2.0 / 3.0])
    assert result.source_pair_id == "left"
    assert result.target_pair_id == "right"
    assert (
        result.pair_id
        == intersect_convex_polygons(
            TRIANGLE, TRIANGLE, source_id="left", target_id="right"
        ).pair_id
    )


def test_containment_and_analytic_partial_triangle_overlap():
    inner = np.asarray([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]])
    contained = intersect_convex_polygons(SQUARE, inner)
    assert contained.status is IntersectionStatus.SUCCESS
    assert contained.area == pytest.approx(0.125)

    # The intersection is the triangle (0, 0), (1, 0), (0, 1) clipped by the
    # x+y <= 1/2 half-plane, whose area is 1/8.
    partial = intersect_convex_polygons(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]),
    )
    assert partial.status is IntersectionStatus.SUCCESS
    assert partial.area == pytest.approx(0.125)
    np.testing.assert_allclose(partial.centroid, [1.0 / 6.0, 1.0 / 6.0])


def test_partial_quad_overlap_and_permutation_canonicalization():
    left = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    right = np.asarray([[1.0, -0.5], [3.0, -0.5], [3.0, 0.5], [1.0, 0.5]])
    expected = np.asarray([[1.0, 0.0], [2.0, 0.0], [2.0, 0.5], [1.0, 0.5]])

    reference = intersect_convex_polygons(left, right)
    rotated = intersect_convex_polygons(
        np.roll(left, 2, axis=0), np.roll(right, -1, axis=0)
    )
    reversed_order = intersect_convex_polygons(left[::-1], right[::-1])

    assert reference.status is IntersectionStatus.SUCCESS
    assert reference.area == pytest.approx(0.5)
    np.testing.assert_allclose(reference.vertices, expected)
    np.testing.assert_allclose(rotated.vertices, reference.vertices)
    np.testing.assert_allclose(reversed_order.vertices, reference.vertices)
    assert rotated.area == reference.area == reversed_order.area


def test_shared_edge_and_vertex_are_explicit_zero_measure_contacts():
    edge = intersect_convex_polygons(SQUARE, SQUARE + [1.0, 0.0])
    vertex = intersect_convex_polygons(SQUARE, SQUARE + [1.0, 1.0])

    assert edge.status is IntersectionStatus.ZERO_MEASURE
    assert edge.area == 0.0
    np.testing.assert_allclose(edge.vertices, [[1.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(edge.centroid, [1.0, 0.5])
    assert vertex.status is IntersectionStatus.ZERO_MEASURE
    assert vertex.area == 0.0
    np.testing.assert_allclose(vertex.vertices, [[1.0, 1.0]])
    np.testing.assert_allclose(vertex.centroid, [1.0, 1.0])


def test_near_degenerate_predicate_fails_closed():
    thin = np.asarray([[0.0, 0.0], [1.0, 1.0e-16], [1.0, 0.0]])
    result = intersect_convex_polygons(thin, thin)

    assert result.status is IntersectionStatus.UNCERTAIN_PREDICATE
    assert result.predicate_evidence.uncertain
    assert result.predicate_evidence.uncertain_count > 0


def test_nonconvex_and_nonfinite_inputs_are_rejected():
    nonconvex = np.asarray([[0.0, 0.0], [2.0, 0.0], [1.0, 0.5], [2.0, 2.0], [0.0, 2.0]])
    crossing = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])

    assert (
        intersect_convex_polygons(nonconvex, SQUARE).status
        is IntersectionStatus.NONCONVEX_INPUT
    )
    assert (
        intersect_convex_polygons(crossing, SQUARE).status
        is IntersectionStatus.SELF_INTERSECTING
    )
    assert (
        intersect_convex_polygons(
            np.asarray([[0.0, 0.0], [1.0, 0.0], [np.inf, 1.0]]), SQUARE
        ).status
        is IntersectionStatus.NONFINITE_INPUT
    )


def test_intersection_areas_conserve_a_partition_without_jit():
    lower = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    upper = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    first = intersect_convex_polygons(lower, SQUARE)
    second = intersect_convex_polygons(upper, SQUARE)

    assert first.status is IntersectionStatus.SUCCESS
    assert second.status is IntersectionStatus.SUCCESS
    assert first.area + second.area == pytest.approx(1.0)

    repeat = intersect_convex_polygons(lower, SQUARE)
    assert repeat.status is first.status
    assert repeat.area == first.area
    np.testing.assert_array_equal(repeat.vertices, first.vertices)
    np.testing.assert_array_equal(repeat.centroid, first.centroid)
    assert repeat.pair_id == first.pair_id
