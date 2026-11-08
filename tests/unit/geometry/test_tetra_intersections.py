#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from itertools import combinations, permutations
from typing import Any

import numpy as np
import pytest

from phydrax._geometry_precision import GeometryPrecisionPolicy
from phydrax._geometry_predicates import PredicateMode, resolve_host_predicate_mode
from phydrax._meshcore import meshcore_available
from phydrax.geometry import _tetra_intersections as tetra_module
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


@pytest.mark.parametrize("mode", (PredicateMode.FILTERED, PredicateMode.EXACT))
def test_sliver_orientation_is_decided_in_every_predicate_mode(
    unit_tetrahedron: Any, mode: PredicateMode
) -> None:
    # The sliver's exact determinant is nonzero, so its orientation is decided
    # even where the filtered orient3d bound cannot certify the sign.
    result = intersect_tetrahedra(
        SLIVER,
        unit_tetrahedron + 100.0,
        tolerance=ZERO_TOLERANCE,
        precision=GeometryPrecisionPolicy(predicate_mode=mode),
    )

    assert result.status is TetraIntersectionStatus.DISJOINT
    assert not result.evidence.predicate_uncertain
    assert result.evidence.predicate_mode is resolve_host_predicate_mode(mode)


# A unit-sized tetrahedron at offset 2**30 and its exact translate by
# (3/4)(FAR[1] - FAR[2]).  In barycentric coordinates of FAR the overlap is
# lambda_1 >= 3/4: the homothet of FAR at FAR[1] with ratio 1/4, of exact volume
# vol(FAR)/64, with vertices FAR[1], FAR_TRANSLATED[2] = (3/4) FAR[1] + (1/4) FAR[2],
# and (3/4) FAR[1] + (1/4) FAR[k] for k = 0, 3.  In rational arithmetic on these
# binary64 inputs every face edge difference and cross product is exact; FAR[1]
# lies exactly on the two FAR_TRANSLATED faces through its edge 1-2 (residual 0)
# and strictly inside the other two, FAR_TRANSLATED[2] likewise for FAR, and all
# four overlap vertices are binary64 numbers.  Rounded plane offsets n . o at this
# offset carry errors up to 6e-8, so admitting the corners through them depends on
# the BLAS accumulation order of the dot product.
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


def _far_overlap_vertices() -> np.ndarray:
    """The four exact overlap vertices, which are all binary64 numbers."""
    rows = []
    for other in (0, 2, 3):
        exact = [
            Fraction(3, 4) * Fraction(float(FAR[1, axis]))
            + Fraction(1, 4) * Fraction(float(FAR[other, axis]))
            for axis in range(3)
        ]
        assert all(Fraction(float(value)) == value for value in exact)
        rows.append([float(value) for value in exact])
    rows.append([float(value) for value in FAR[1]])
    return np.asarray(sorted(rows), dtype=np.float64)


def test_far_corner_overlap_is_certified_instead_of_disjoint() -> None:
    _assert_far_translate_is_exact()
    exact = _exact_tetrahedron_volume(FAR) / 64

    result = intersect_tetrahedra(FAR, FAR_TRANSLATED, volume_only=True)

    assert result.status is TetraIntersectionStatus.SUCCESS
    assert abs(Fraction(result.volume) - exact) <= Fraction(result.evidence.volume_error)


def _assert_far_overlap_topology(result: Any) -> None:
    assert result.status is TetraIntersectionStatus.SUCCESS
    assert not result.evidence.predicate_uncertain
    np.testing.assert_array_equal(result.vertices, _far_overlap_vertices())
    assert result.faces == ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))


def test_far_corner_overlap_constructs_the_exact_polytope() -> None:
    _assert_far_translate_is_exact()

    _assert_far_overlap_topology(intersect_tetrahedra(FAR, FAR_TRANSLATED))


def test_far_corner_overlap_lost_by_the_constructed_topology_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_far_translate_is_exact()
    # The exact construction loses its closure: it refuses loudly instead of
    # reporting a SUCCESS with an unclosed topology.
    monkeypatch.setattr(tetra_module, "_closed_surface", lambda *arguments: False)

    with pytest.raises(RuntimeError, match="does not close"):
        intersect_tetrahedra(FAR, FAR_TRANSLATED)


# beta = 1 + 2**-30: det[(1, 0, 0), (0, beta, 1), (0, 1, beta)] = beta**2 - 1 is
# exactly 2**-29 + 2**-60, a binary64 number, but rounding beta**2 gives 2**-29.
_THRESHOLD_BETA = 1.0 + 2.0**-30
THRESHOLD_TETRAHEDRON = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, _THRESHOLD_BETA, 1.0),
        (0.0, 1.0, _THRESHOLD_BETA),
    ),
    dtype=np.float64,
)
_EXACT_DETERMINANT = 2.0**-29 + 2.0**-60


@pytest.mark.parametrize(
    ("threshold", "degenerate"),
    (
        (_EXACT_DETERMINANT, True),
        (float(np.nextafter(_EXACT_DETERMINANT, np.inf)), True),
        (float(np.nextafter(_EXACT_DETERMINANT, -np.inf)), False),
        (2.0**-29, False),
    ),
    ids=("equal", "ulp-above", "ulp-below", "rounded-determinant"),
)
def test_degeneracy_compares_the_exact_determinant_with_the_threshold(
    threshold: float, degenerate: bool
) -> None:
    assert _exact_tetrahedron_volume(THRESHOLD_TETRAHEDRON) * 6 == Fraction(
        _EXACT_DETERMINANT
    )
    assert _THRESHOLD_BETA * _THRESHOLD_BETA - 1.0 == 2.0**-29
    tolerance = TetraIntersectionTolerance(absolute=threshold, relative=0.0)

    result = intersect_tetrahedra(
        THRESHOLD_TETRAHEDRON,
        THRESHOLD_TETRAHEDRON,
        tolerance=tolerance,
        volume_only=True,
    )

    assert result.status is (
        TetraIntersectionStatus.DEGENERATE_TETRAHEDRON
        if degenerate
        else TetraIntersectionStatus.ZERO_MEASURE_CONTACT
    )


def _fused(left: float, right: float, addend: float) -> float:
    """``fma(left, right, addend)``: one rounding of the exact value."""
    if not all(np.isfinite((left, right, addend))):
        return left * right + addend
    return float(Fraction(left) * Fraction(right) + Fraction(addend))


_FUSED = np.frompyfunc(_fused, 3, 1)


def _reassociated(first: int, second: int, third: int) -> Any:
    def dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (
            left[..., first] * right[..., first] + left[..., second] * right[..., second]
        ) + left[..., third] * right[..., third]

    return dot


def _aarch64_openblas(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # fl(fma(a2, b2, a0 * b0) + a1 * b1), the Linux aarch64 OpenBLAS ddot.
    fused = _FUSED(left[..., 2], right[..., 2], left[..., 0] * right[..., 0])
    return np.asarray(fused, dtype=np.float64) + left[..., 1] * right[..., 1]


@pytest.mark.parametrize(
    "dot",
    (
        _reassociated(0, 1, 2),
        _reassociated(0, 2, 1),
        _reassociated(2, 0, 1),
        _aarch64_openblas,
    ),
    ids=("(x0+x1)+x2", "(x0+x2)+x1", "(x2+x0)+x1", "fma(x2,x0)+x1"),
)
def test_far_corner_decisions_do_not_depend_on_the_summation_order(
    monkeypatch: pytest.MonkeyPatch, dot: Any
) -> None:
    # Each order or contraction a BLAS kernel may use for a three-term dot
    # product is injected into both routes a dot product can take: NumPy's
    # ``np.dot`` and the module's fixed-order ``_dot3``.
    blas_dot = np.dot
    calls = [0]

    def counted(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        calls[0] += 1
        return dot(left, right)

    def ordered_dot(left: Any, right: Any) -> Any:
        left_ = np.asarray(left)
        right_ = np.asarray(right)
        if left_.shape == right_.shape == (3,):
            return counted(left_, right_)
        return blas_dot(left, right)

    monkeypatch.setattr(np, "dot", ordered_dot)
    monkeypatch.setattr(tetra_module, "_dot3", counted, raising=False)

    _assert_far_overlap_topology(intersect_tetrahedra(FAR, FAR_TRANSLATED))
    assert calls[0] > 0


# A far-offset pair whose distinct exact vertices round to one binary64: the edge
# P-Q of the second cell passes u/4 above the first cell's edge along x at
# y = z = C (u = ulp(C)).  It enters the first cell through y = C and leaves
# through z = C at two exact points within u/4 of (C + 1, C, C).
_C = 2.0**30
_U = float(np.spacing(_C))
COINCIDENT_SOURCE = np.asarray(
    ((_C, _C, _C), (_C + 4, _C, _C), (_C, _C + 4, _C), (_C, _C, _C + 4)),
    dtype=np.float64,
)
COINCIDENT_TARGET = np.asarray(
    (
        (_C + 1, _C - 3, _C + 3 + _U),
        (_C + 1, _C + 1, _C - 1),
        (_C - 2, _C + 3, _C + 3),
        (_C + 4, _C + 3, _C + 3),
    ),
    dtype=np.float64,
)


def _determinant(rows: list[list[Fraction]]) -> Fraction:
    return (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )


def _rounded_exact_vertices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Correctly rounded vertices of the exact intersection, independently."""
    halfspaces: list[tuple[list[Fraction], Fraction]] = []
    for cell in (source, target):
        points = [[Fraction(float(value)) for value in row] for row in cell]
        for opposite in range(4):
            origin, first, second = (points[i] for i in range(4) if i != opposite)
            u = [a - b for a, b in zip(first, origin, strict=True)]
            v = [a - b for a, b in zip(second, origin, strict=True)]
            normal = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ]
            offset = -sum(
                (a * b for a, b in zip(normal, origin, strict=True)), Fraction(0)
            )
            if (
                sum(a * b for a, b in zip(normal, points[opposite], strict=True))
                < -offset
            ):
                normal, offset = [-a for a in normal], -offset
            halfspaces.append((normal, offset))
    exact: set[tuple[Fraction, ...]] = set()
    for triple in combinations(halfspaces, 3):
        matrix = [normal for normal, _ in triple]
        determinant = _determinant(matrix)
        if determinant == 0:
            continue
        point = tuple(
            _determinant(
                [
                    [-offset if column == axis else row[column] for column in range(3)]
                    for row, (_, offset) in zip(matrix, triple, strict=True)
                ]
            )
            / determinant
            for axis in range(3)
        )
        if all(
            sum(a * b for a, b in zip(normal, point, strict=True)) + offset >= 0
            for normal, offset in halfspaces
        ):
            exact.add(point)
    return np.asarray(
        [[float(value) for value in point] for point in sorted(exact)], dtype=np.float64
    )


# A generic overlap: its exact vertices differ by up to 119 ulps from the
# intersections of the rounded np.cross normal planes.
GENERIC_SOURCE = _hex_rows(
    (
        ("0x1.911924f1561c9p-1", "0x1.e57dc3e1048e2p-1", "0x1.1103a5bb45834p-3"),
        ("0x1.6881b3714921fp-1", "0x1.78bb935de7596p-2", "0x1.46d7ca30d4f8fp-1"),
        ("0x1.3fa44de5cfc68p-4", "0x1.9713a804e7131p-1", "0x1.13030f41ab727p-1"),
        ("0x1.c6ff5711523a0p-6", "0x1.c21d8d593b5c8p-3", "0x1.c691bdc876e70p-5"),
    )
)
GENERIC_TARGET = _hex_rows(
    (
        ("0x1.0500f6325897ep-1", "0x1.59bbaf1345c4ap-1", "0x1.e3a5140f733cdp-3"),
        ("0x1.492e59a1fe5bfp-1", "0x1.2dd650754f7bbp-1", "0x1.d51a48c47185cp-2"),
        ("0x1.1677bac319ca0p-3", "0x1.0e14c047f5b22p-1", "0x1.7482020b09f68p-2"),
        ("0x1.0ab117c8232e6p-1", "0x1.556152c47049ep-1", "0x1.02ff23208440cp-2"),
    )
)


@pytest.mark.parametrize(
    ("source", "target"),
    ((GENERIC_SOURCE, GENERIC_TARGET), (FAR, FAR_TRANSLATED)),
    ids=("generic", "far"),
)
def test_success_vertices_are_the_correctly_rounded_exact_vertices(
    source: np.ndarray, target: np.ndarray
) -> None:
    result = intersect_tetrahedra(source, target)

    assert result.status is TetraIntersectionStatus.SUCCESS
    np.testing.assert_array_equal(
        result.vertices, _rounded_exact_vertices(source, target)
    )


def test_closed_surface_requires_consistently_oriented_faces() -> None:
    oriented = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    # One face reversed: every edge still borders two faces, but a directed edge
    # repeats and its reverse is missing.
    reversed_face = ((0, 1, 2), (0, 1, 3), (0, 3, 2), (1, 2, 3))

    assert tetra_module._closed_surface(4, oriented)
    assert not tetra_module._closed_surface(4, reversed_face)


def test_distinct_exact_vertices_may_round_to_one_coordinate() -> None:
    first, second = (
        [Fraction(float(value)) for value in COINCIDENT_TARGET[row]] for row in (0, 1)
    )
    # The exact points where P-Q crosses y = C and z = C.
    crossings = []
    for axis in (1, 2):
        weight = (Fraction(_C) - first[axis]) / (second[axis] - first[axis])
        crossings.append(
            tuple(a + weight * (b - a) for a, b in zip(first, second, strict=True))
        )
    assert crossings[0] != crossings[1]
    rounded = (_C + 1, _C, _C)
    assert all(tuple(float(value) for value in point) == rounded for point in crossings)

    result = intersect_tetrahedra(COINCIDENT_SOURCE, COINCIDENT_TARGET)

    assert result.status is TetraIntersectionStatus.SUCCESS
    # The topology is the exact polytope's: closed and oriented over every vertex
    # index, while two of its vertices share a correctly rounded coordinate.
    assert tetra_module._closed_surface(len(result.vertices), result.faces)
    matches = np.flatnonzero(np.all(result.vertices == np.asarray(rounded), axis=1))
    assert len(matches) == 2
    assert len(np.unique(result.vertices, axis=0)) == len(result.vertices) - 1


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
