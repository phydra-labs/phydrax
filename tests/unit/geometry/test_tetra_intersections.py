#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from itertools import permutations
from typing import Any

import numpy as np
import pytest

from phydrax._geometry_precision import GeometryPrecisionPolicy
from phydrax._geometry_predicates import PredicateMode
from phydrax._meshcore import meshcore_available
from phydrax.geometry._tetra_intersections import (
    intersect_tetrahedra,
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


def test_identity_has_canonical_polytope_and_compensated_volume(
    unit_tetrahedron: Any,
) -> None:
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


def test_containment_returns_inner_tetrahedron(unit_tetrahedron: Any) -> None:
    inner = 0.5 * unit_tetrahedron
    result = intersect_tetrahedra(unit_tetrahedron, inner)

    assert result.status is TetraIntersectionStatus.SUCCESS
    assert result.volume == pytest.approx(1.0 / 48.0)
    np.testing.assert_array_equal(
        result.vertices,
        np.asarray(sorted(tuple(float(value) for value in row) for row in inner)),
    )


def test_partial_overlap_matches_analytic_shifted_tetrahedron(
    unit_tetrahedron: Any,
) -> None:
    translated = unit_tetrahedron + 0.25
    result = intersect_tetrahedra(unit_tetrahedron, translated)

    assert result.status is TetraIntersectionStatus.SUCCESS
    assert result.volume == pytest.approx(1.0 / 384.0, rel=0.0, abs=2.0e-14)
    assert result.evidence.vertex_count == 4


def _hex_rows(rows: tuple[tuple[str, str, str], ...]) -> np.ndarray:
    return np.asarray(
        [[float.fromhex(value) for value in row] for row in rows], dtype=np.float64
    )


def _exact_tetrahedron_volume(vertices: np.ndarray) -> Fraction:
    points = [[Fraction(float(value)) for value in row] for row in vertices]
    edges = [
        [points[row][axis] - points[0][axis] for axis in range(3)] for row in (1, 2, 3)
    ]
    determinant = (
        edges[0][0] * (edges[1][1] * edges[2][2] - edges[1][2] * edges[2][1])
        - edges[0][1] * (edges[1][0] * edges[2][2] - edges[1][2] * edges[2][0])
        + edges[0][2] * (edges[1][0] * edges[2][1] - edges[1][1] * edges[2][0])
    )
    return abs(determinant) / 6


# A cube-inscribed tetrahedron at scale 2**21 whose triple products sit near the
# Hadamard bound; it lies strictly inside its 4x scaling, so the exact
# intersection is the tetrahedron itself.
CUBE_INSCRIBED = _hex_rows(
    (
        ("0x1.ebb2848c75b00p+21", "0x1.da5d9b6f930a7p+21", "0x1.f3cf9c693e9e3p+21"),
        ("-0x1.d751e354c6d3fp+21", "0x1.d2d58fc23d707p+21", "-0x1.ce27817798058p+21"),
        ("0x1.e6d34eb733dabp+21", "-0x1.f1119ab215ba8p+21", "-0x1.d960d46151ec0p+21"),
        ("-0x1.ebb6de2d1d79bp+21", "-0x1.d046159daac38p+21", "0x1.e5b162d222062p+21"),
    )
)
UNIT = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)
# Targets whose only active face cuts the unit tetrahedron along x <= 1/2 (three
# vertices kept) or x + y <= 1/2 (two vertices kept).
HALF_X_TARGET = np.asarray(
    ((0.5, -10.0, -10.0), (0.5, -10.0, 30.0), (0.5, 30.0, -10.0), (-20.0, 0.0, 0.0)),
    dtype=np.float64,
)
HALF_XY_TARGET = np.asarray(
    (
        (-19.75, 20.25, -10.0),
        (20.25, -19.75, -10.0),
        (0.25, 0.25, 40.0),
        (-30.0, -30.0, 5.0),
    ),
    dtype=np.float64,
)


@pytest.mark.parametrize(
    ("source", "target", "exact"),
    (
        (CUBE_INSCRIBED, 4.0 * CUBE_INSCRIBED, _exact_tetrahedron_volume(CUBE_INSCRIBED)),
        (UNIT, UNIT + 0.25, Fraction(1, 384)),
        (UNIT, HALF_X_TARGET, Fraction(1, 6) - Fraction(1, 48)),
        (UNIT, HALF_XY_TARGET, Fraction(1, 12)),
    ),
)
def test_volume_error_encloses_the_exact_intersection_volume(
    source: np.ndarray, target: np.ndarray, exact: Fraction
) -> None:
    result = intersect_tetrahedra(source, target)

    assert result.status is TetraIntersectionStatus.SUCCESS
    assert abs(Fraction(result.volume) - exact) <= Fraction(result.evidence.volume_error)


@pytest.mark.meshcore
@pytest.mark.skipif(not meshcore_available(), reason="phydrax-meshcore unavailable")
def test_shared_face_edge_and_vertex_are_zero_measure_contacts(
    unit_tetrahedron: Any,
) -> None:
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


def test_even_vertex_permutations_have_identical_canonical_result(
    unit_tetrahedron: Any,
) -> None:
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


def test_inverted_near_degenerate_and_nonfinite_tetrahedra_fail_closed(
    unit_tetrahedron: Any,
) -> None:
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


def test_disjoint_and_volume_conservation_bounds(unit_tetrahedron: Any) -> None:
    disjoint = unit_tetrahedron + np.asarray((2.0, 0.0, 0.0))
    result = intersect_tetrahedra(unit_tetrahedron, disjoint)

    assert result.status is TetraIntersectionStatus.DISJOINT
    assert result.volume == 0.0
    assert result.volume <= 1.0 / 6.0


def test_volume_only_and_repeated_results_are_deterministic(
    unit_tetrahedron: Any,
) -> None:
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


def test_unresolved_filtered_orientation_fails_closed(unit_tetrahedron: Any) -> None:
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
def test_exact_orientation_certifies_the_sliver(unit_tetrahedron: Any) -> None:
    result = intersect_tetrahedra(
        SLIVER,
        unit_tetrahedron + 100.0,
        tolerance=ZERO_TOLERANCE,
        precision=GeometryPrecisionPolicy(predicate_mode=PredicateMode.EXACT),
    )

    assert result.status is TetraIntersectionStatus.DISJOINT
    assert not result.evidence.predicate_uncertain
    assert result.evidence.predicate_mode is PredicateMode.EXACT


# A unit-sized tetrahedron at offset 2**30 and its exact translate by
# (3/4)(FAR[1] - FAR[2]).  In barycentric coordinates of FAR the overlap is
# lambda_1 >= 3/4: the homothet of FAR at FAR[1] with ratio 1/4, of exact volume
# vol(FAR)/64.  The rounded face residuals at this offset exceed the coordinate
# tolerance, so the floating-point vertex enumeration constructs no vertex.
FAR = _hex_rows(
    (
        ("0x1.262802e44c000p+30", "0x1.f7082049c4000p+30", "0x1.66d1a1b140000p+30"),
        ("0x1.262802e2c0000p+30", "0x1.f7082046a0000p+30", "0x1.66d1a1b084000p+30"),
        ("0x1.262802e3e8000p+30", "0x1.f708204668000p+30", "0x1.66d1a1b070000p+30"),
        ("0x1.262802e50c000p+30", "0x1.f708204748000p+30", "0x1.66d1a1b190000p+30"),
    )
)
FAR_TRANSLATED = FAR + 0.75 * (FAR[1] - FAR[2])


def _assert_far_translate_is_exact() -> None:
    for row in range(4):
        for axis in range(3):
            assert Fraction(float(FAR_TRANSLATED[row, axis])) == Fraction(
                float(FAR[row, axis])
            ) + Fraction(3, 4) * (
                Fraction(float(FAR[1, axis])) - Fraction(float(FAR[2, axis]))
            )


def test_far_corner_overlap_is_certified_instead_of_disjoint() -> None:
    _assert_far_translate_is_exact()
    exact = _exact_tetrahedron_volume(FAR) / 64

    result = intersect_tetrahedra(FAR, FAR_TRANSLATED, volume_only=True)

    assert result.status is TetraIntersectionStatus.SUCCESS
    assert abs(Fraction(result.volume) - exact) <= Fraction(result.evidence.volume_error)


def test_far_corner_overlap_lost_by_the_constructed_topology_fails_closed() -> None:
    _assert_far_translate_is_exact()

    result = intersect_tetrahedra(FAR, FAR_TRANSLATED)

    assert result.status is TetraIntersectionStatus.UNCERTAIN_PREDICATE
    assert result.evidence.predicate_uncertain


def test_gap_below_the_coordinate_tolerance_is_disjoint(unit_tetrahedron: Any) -> None:
    other = unit_tetrahedron + np.asarray((1.0 + 2.0**-40, 0.0, 0.0))
    # The plane x = 1 + 2**-41 separates the closed tetrahedra exactly.
    assert max(Fraction(float(value)) for value in unit_tetrahedron[:, 0]) < min(
        Fraction(float(value)) for value in other[:, 0]
    )

    result = intersect_tetrahedra(unit_tetrahedron, other)

    assert result.status is TetraIntersectionStatus.DISJOINT
    assert result.volume == 0.0
    assert result.vertices.shape == (0, 3)
