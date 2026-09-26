#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from itertools import permutations

import numpy as np
import pytest

from phydrax._geometry_precision import GeometryPrecisionPolicy
from phydrax._meshcore import meshcore_available
from phydrax.geometry._predicates import PredicateMode
from phydrax.geometry._tetra_intersections import (
    intersect_tetrahedra,
    TetraIntersectionLimits,
    TetraIntersectionStatus,
    TetraIntersectionTolerance,
)


@pytest.fixture
def unit_tetrahedron() -> np.ndarray:
    return np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def test_identity_has_canonical_polytope_and_compensated_volume(unit_tetrahedron):
    result = intersect_tetrahedra(unit_tetrahedron, unit_tetrahedron)

    assert result.status is TetraIntersectionStatus.SUCCESS
    assert result.volume == pytest.approx(1.0 / 6.0, rel=0.0, abs=2.0e-14)
    np.testing.assert_array_equal(
        result.vertices,
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
            )
        ),
    )
    assert len(result.faces) == 4
    assert result.vertices.flags.writeable is False
    assert result.evidence.volume_error < 1.0e-12


def test_containment_returns_inner_tetrahedron(unit_tetrahedron):
    inner = 0.5 * unit_tetrahedron
    result = intersect_tetrahedra(unit_tetrahedron, inner)

    assert result.status is TetraIntersectionStatus.SUCCESS
    assert result.volume == pytest.approx(1.0 / 48.0)
    np.testing.assert_array_equal(
        result.vertices,
        np.asarray(sorted(tuple(float(value) for value in row) for row in inner)),
    )


def test_partial_overlap_matches_analytic_shifted_tetrahedron(unit_tetrahedron):
    translated = unit_tetrahedron + 0.25
    result = intersect_tetrahedra(unit_tetrahedron, translated)

    assert result.status is TetraIntersectionStatus.SUCCESS
    assert result.volume == pytest.approx(1.0 / 384.0, rel=0.0, abs=2.0e-14)
    assert result.evidence.vertex_count == 4


@pytest.mark.meshcore
@pytest.mark.skipif(not meshcore_available(), reason="phydrax-meshcore unavailable")
def test_shared_face_edge_and_vertex_are_zero_measure_contacts(unit_tetrahedron):
    # Three contact planes meeting in a shared edge form an exactly singular
    # triple; certifying that zero requires the exact predicates.
    exact = GeometryPrecisionPolicy(predicate_mode=PredicateMode.EXACT)
    shared_face = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )
    shared_edge = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        )
    )
    shared_vertex = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, -1.0, 0.0),
        )
    )

    for other in (shared_face, shared_edge, shared_vertex):
        result = intersect_tetrahedra(unit_tetrahedron, other, precision=exact)
        assert result.status is TetraIntersectionStatus.ZERO_MEASURE_CONTACT
        assert result.volume == 0.0

    face_result = intersect_tetrahedra(unit_tetrahedron, shared_face, precision=exact)
    edge_result = intersect_tetrahedra(unit_tetrahedron, shared_edge, precision=exact)
    vertex_result = intersect_tetrahedra(unit_tetrahedron, shared_vertex, precision=exact)
    assert face_result.evidence.vertex_count == 3
    assert edge_result.evidence.vertex_count == 2
    assert vertex_result.evidence.vertex_count == 1


def test_even_vertex_permutations_have_identical_canonical_result(unit_tetrahedron):
    reference = intersect_tetrahedra(unit_tetrahedron, unit_tetrahedron)
    for order in permutations(range(4)):
        if (
            sum(
                order[index] > order[j] for index in range(4) for j in range(index + 1, 4)
            )
            % 2
        ):
            continue
        result = intersect_tetrahedra(unit_tetrahedron[list(order)], unit_tetrahedron)
        assert result.status is TetraIntersectionStatus.SUCCESS
        assert result.volume == reference.volume
        np.testing.assert_array_equal(result.vertices, reference.vertices)
        assert result.faces == reference.faces


def test_inverted_near_degenerate_and_nonfinite_tetrahedra_fail_closed(unit_tetrahedron):
    inverted = unit_tetrahedron[[0, 1, 3, 2]]
    near_degenerate = unit_tetrahedron.copy()
    near_degenerate[3, 2] = 1.0e-15
    nonfinite = unit_tetrahedron.copy()
    nonfinite[2, 1] = np.inf

    assert (
        intersect_tetrahedra(inverted, unit_tetrahedron).status
        is TetraIntersectionStatus.INVERTED_TETRAHEDRON
    )
    assert (
        intersect_tetrahedra(near_degenerate, unit_tetrahedron).status
        is TetraIntersectionStatus.DEGENERATE_TETRAHEDRON
    )
    assert (
        intersect_tetrahedra(nonfinite, unit_tetrahedron).status
        is TetraIntersectionStatus.NONFINITE_INPUT
    )


def test_disjoint_and_volume_conservation_bounds(unit_tetrahedron):
    disjoint = unit_tetrahedron + np.asarray((2.0, 0.0, 0.0))
    result = intersect_tetrahedra(unit_tetrahedron, disjoint)

    assert result.status is TetraIntersectionStatus.DISJOINT
    assert result.volume == 0.0
    assert result.volume <= 1.0 / 6.0


def test_candidate_and_topology_limits_are_explicit(unit_tetrahedron):
    candidate_limited = intersect_tetrahedra(
        unit_tetrahedron,
        unit_tetrahedron,
        limits=TetraIntersectionLimits(max_candidates=3, max_vertices=64, max_faces=16),
    )
    topology_limited = intersect_tetrahedra(
        unit_tetrahedron,
        unit_tetrahedron,
        limits=TetraIntersectionLimits(max_candidates=64, max_vertices=3, max_faces=16),
    )

    assert candidate_limited.status is TetraIntersectionStatus.CANDIDATE_LIMIT
    assert topology_limited.status is TetraIntersectionStatus.CANDIDATE_LIMIT


def test_volume_only_and_repeated_results_are_deterministic(unit_tetrahedron):
    tolerance = TetraIntersectionTolerance(absolute=1.0e-13, relative=1.0e-11)
    first = intersect_tetrahedra(
        unit_tetrahedron,
        unit_tetrahedron + 0.25,
        source_id="left",
        target_id="right",
        tolerance=tolerance,
        volume_only=True,
    )
    second = intersect_tetrahedra(
        unit_tetrahedron,
        unit_tetrahedron + 0.25,
        source_id="left",
        target_id="right",
        tolerance=tolerance,
        volume_only=True,
    )

    assert first.status is TetraIntersectionStatus.SUCCESS
    assert first.pair_id == "left:right"
    assert first.evidence.volume_only is True
    assert first.vertices.shape == (0, 3)
    assert first.faces == ()
    assert first.volume == second.volume
    assert first.evidence == second.evidence


# Positively oriented sliver whose orientation (exactly 11.5 * 2**-48) is
# below the floating-point filter bound.
SLIVER = np.asarray(
    (
        (0.5, 0.5, 0.0),
        (12.0, 12.0, 0.0),
        (24.0, 24.0 + 2.0**-48, 0.0),
        (0.0, 0.0, 1.0),
    )
)
ZERO_TOLERANCE = TetraIntersectionTolerance(absolute=0.0, relative=0.0)


def test_unresolved_filtered_orientation_fails_closed(unit_tetrahedron):
    result = intersect_tetrahedra(
        SLIVER,
        unit_tetrahedron + 100.0,
        tolerance=ZERO_TOLERANCE,
        precision=GeometryPrecisionPolicy(predicate_mode=PredicateMode.FILTERED),
    )

    assert result.status is TetraIntersectionStatus.UNCERTAIN_PREDICATE
    assert result.evidence.predicate_uncertain
    assert result.evidence.predicate_mode is PredicateMode.FILTERED


@pytest.mark.meshcore
@pytest.mark.skipif(not meshcore_available(), reason="phydrax-meshcore unavailable")
def test_exact_orientation_certifies_the_sliver(unit_tetrahedron):
    result = intersect_tetrahedra(
        SLIVER,
        unit_tetrahedron + 100.0,
        tolerance=ZERO_TOLERANCE,
        precision=GeometryPrecisionPolicy(predicate_mode=PredicateMode.EXACT),
    )

    assert result.status is TetraIntersectionStatus.DISJOINT
    assert not result.evidence.predicate_uncertain
    assert result.evidence.predicate_mode is PredicateMode.EXACT
