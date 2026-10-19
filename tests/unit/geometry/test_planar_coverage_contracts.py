from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.discretization._coordinate_enclosure import axes, constant
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._mapped_coverage import SubdivisionLedger
from phydrax.geometry._planar_coverage import (
    affine_segment_containment,
    affine_simplex_corner_containment,
    AffineSourceSimplex,
    mapped_containment,
    prepare_affine_exact_source_simplex,
    source_groups,
)
from phydrax.geometry._planar_primitives import (
    _prepared_separated,
    intersection,
    intersection_polygon,
    Polygon,
    PolygonSidePreparation,
    prepare_polygon_intersection,
    signed_measure,
)
from tests._support.coordinate_references import coordinate_reference_budget


@pytest.mark.parametrize(
    "reverse", (False, True), ids=("forward-regions", "reverse-regions")
)
def test_planar_source_union_preserves_oriented_chart_area(reverse: bool) -> None:
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=np.float64)
    facets = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    regions = np.asarray(((0, -1), (0, -1)), dtype=np.int64)
    with coordinate_reference_budget() as budget:
        groups, overlaps, used, exceeded = source_groups(
            vertices, facets, regions, 16, budget=budget
        )
        assert len(groups) == 1 and overlaps == () and not exceeded
        assert groups[0].members == (0, 1)
        assert groups[0].measures == (Fraction(1, 2), Fraction(1, 2))
        x, y = axes(2, budget=budget)
        chart = (x, y, constant(0, 2, budget=budget))
        work = SubdivisionLedger(candidate_pairs=used)
        own, other = (-1, 0) if reverse else (0, -1)
        outcome, measure, candidates = mapped_containment(
            chart, "simplex", groups[0], own, other, 64, 4, 128, 16, work, budget=budget
        )
    assert outcome == ("absent" if reverse else "proven")
    assert measure == (Fraction(0) if reverse else Fraction(1, 2))
    assert candidates == (() if reverse else (0, 1))


def test_planar_source_group_reports_positive_area_overlap() -> None:
    vertices = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    facets = np.asarray(((0, 1, 2), (0, 1, 2)), dtype=np.int64)
    regions = np.asarray(((0, -1), (0, -1)), dtype=np.int64)
    with coordinate_reference_budget() as budget:
        groups, overlaps, used, exceeded = source_groups(
            vertices, facets, regions, 16, budget=budget
        )
    assert len(groups) == 1
    assert overlaps == ((0, 1),)
    assert used == 1 and not exceeded


def test_affine_source_simplex_retains_plane_projection_and_boundary_containment() -> (
    None
):
    corners = tuple(
        tuple(Fraction(value) for value in point)
        for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    )
    with coordinate_reference_budget() as budget:
        source = prepare_affine_exact_source_simplex(corners, budget=budget)
        assert isinstance(source, AffineSourceSimplex)
        assert source.axes == (0, 1)
        assert source.orientation == 1
        assert affine_simplex_corner_containment(
            (corners[0], corners[1], corners[2]), source, budget=budget
        ) == (Fraction(1, 2), 1)
        assert affine_segment_containment((corners[0], corners[1]), source, budget=budget)
        outside = (Fraction(2), Fraction(0), Fraction(0))
        assert not affine_segment_containment(
            (corners[0], outside), source, budget=budget
        )


def test_affine_source_simplex_refuses_degenerate_source_plane() -> None:
    points = tuple(
        tuple(Fraction(value) for value in point)
        for point in ((0, 0, 0), (1, 0, 0), (2, 0, 0))
    )
    with coordinate_reference_budget() as budget:
        with pytest.raises(
            ValueError, match="prepare_affine_exact_source_simplex.points"
        ):
            prepare_affine_exact_source_simplex(points, budget=budget)


@pytest.mark.parametrize("reverse_subject", (False, True))
@pytest.mark.parametrize("reverse_target", (False, True))
@pytest.mark.parametrize(
    ("target_rows", "expected"),
    (
        (((0, 0), (1, 0), (0, 1)), Fraction(1, 2)),
        (((Fraction(1, 2), 0), (Fraction(3, 2), 0), (Fraction(1, 2), 1)), Fraction(1, 8)),
        (((1, 0), (1, 1), (0, 1)), Fraction(0)),
        (((1, 0), (2, 0), (1, 1)), Fraction(0)),
        (
            (
                (1 + Fraction(1, 2**60), 0),
                (1 + Fraction(1, 2**60), 1),
                (Fraction(1, 2**60), 1),
            ),
            Fraction(0),
        ),
        (
            (
                (1 - Fraction(1, 2**60), 0),
                (1 - Fraction(1, 2**60), 1),
                (-Fraction(1, 2**60), 1),
            ),
            Fraction(1, 2**60) * (1 - Fraction(1, 2**60)),
        ),
    ),
    ids=(
        "identical",
        "positive-overlap",
        "edge-contact",
        "vertex-contact",
        "exact-gap",
        "thin-overlap",
    ),
)
def test_prepared_clipping_preserves_exact_measure_and_polygon(
    target_rows: tuple[tuple[int | Fraction, int | Fraction], ...],
    expected: Fraction,
    reverse_subject: bool,
    reverse_target: bool,
) -> None:
    subject: Polygon = tuple(
        tuple(Fraction(value) for value in point) for point in ((0, 0), (1, 0), (0, 1))
    )
    target: Polygon = tuple(
        tuple(Fraction(value) for value in point) for point in target_rows
    )
    if reverse_subject:
        subject = subject[::-1]
    if reverse_target:
        target = target[::-1]
    with coordinate_reference_budget() as budget:
        subject_preparation = prepare_polygon_intersection(
            subject, -1 if reverse_subject else 1, budget=budget
        )
        target_preparation = prepare_polygon_intersection(
            target, -1 if reverse_target else 1, budget=budget
        )
        with budget.temporary_scope():
            raw_polygon = intersection_polygon(subject, target, budget=budget)
            prepared_polygon = intersection_polygon(
                subject, target, budget=budget, target_preparation=target_preparation
            )
            assert prepared_polygon == raw_polygon
            assert abs(signed_measure(prepared_polygon, budget=budget)) == expected
        raw_measure = intersection(subject, target, budget=budget)
        prepared_measure = intersection(
            subject,
            target,
            budget=budget,
            subject_preparation=subject_preparation,
            target_preparation=target_preparation,
        )
    assert prepared_measure == raw_measure == expected


def test_prepared_clipping_reduces_repeated_positive_overlap_work() -> None:
    polygon: Polygon = tuple(
        tuple(Fraction(value) for value in point) for point in ((0, 0), (1, 0), (0, 1))
    )
    with coordinate_reference_budget() as raw_budget:
        raw = tuple(intersection(polygon, polygon, budget=raw_budget) for _ in range(16))
        raw_work = raw_budget.work_units
    with coordinate_reference_budget() as prepared_budget:
        preparation = prepare_polygon_intersection(polygon, 1, budget=prepared_budget)
        prepared = tuple(
            intersection(
                polygon,
                polygon,
                budget=prepared_budget,
                subject_preparation=preparation,
                target_preparation=preparation,
            )
            for _ in range(16)
        )
        prepared_work = prepared_budget.work_units
    assert prepared == raw == (Fraction(1, 2),) * 16
    assert prepared_work < raw_work


@pytest.mark.parametrize("prepared", (False, True), ids=("raw", "prepared"))
def test_clipping_refuses_exhausted_original_work_allowance(prepared: bool) -> None:
    polygon: Polygon = tuple(
        tuple(Fraction(value) for value in point) for point in ((0, 0), (1, 0), (0, 1))
    )
    limits = MeshCertificateLimits(maximum_work_units=1000)
    with coordinate_reference_budget(limits) as budget:
        preparation = (
            prepare_polygon_intersection(polygon, 1, budget=budget) if prepared else None
        )
        budget.reserve(limits.maximum_work_units - budget.work_units - 1)
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            intersection(
                polygon,
                polygon,
                budget=budget,
                subject_preparation=preparation,
                target_preparation=preparation,
            )
        assert caught.value.resource == "coefficient_work"
        assert budget.work_units <= limits.maximum_work_units


def test_prepared_clipping_refuses_a_different_polygon_owner() -> None:
    polygon: Polygon = tuple(
        tuple(Fraction(value) for value in point) for point in ((0, 0), (1, 0), (0, 1))
    )
    other_owner = tuple(list(polygon))
    with coordinate_reference_budget() as budget:
        preparation = prepare_polygon_intersection(polygon, 1, budget=budget)
        with pytest.raises(ValueError, match="must bind the target polygon"):
            intersection(
                polygon, other_owner, budget=budget, target_preparation=preparation
            )


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    ("endpoints", "expected"),
    (
        ((Fraction(0), Fraction(1)), Fraction(1)),
        ((Fraction(1, 2), Fraction(3, 2)), Fraction(1, 2)),
        ((Fraction(1), Fraction(2)), Fraction(0)),
        ((Fraction(3, 2), Fraction(2)), Fraction(0)),
    ),
    ids=("identical", "overlap", "contact", "gap"),
)
def test_prepared_interval_clipping_preserves_exact_endpoints(
    endpoints: tuple[Fraction, Fraction],
    expected: Fraction,
    reverse: bool,
) -> None:
    subject: Polygon = ((Fraction(0),), (Fraction(1),))
    target: Polygon = tuple((value,) for value in endpoints)
    if reverse:
        target = target[::-1]
    with coordinate_reference_budget() as budget:
        subject_preparation = prepare_polygon_intersection(subject, 1, budget=budget)
        target_preparation = prepare_polygon_intersection(
            target, -1 if reverse else 1, budget=budget
        )
        with budget.temporary_scope():
            assert intersection_polygon(
                subject, target, budget=budget, target_preparation=target_preparation
            ) == intersection_polygon(subject, target, budget=budget)
        assert (
            intersection(
                subject,
                target,
                budget=budget,
                subject_preparation=subject_preparation,
                target_preparation=target_preparation,
            )
            == intersection(subject, target, budget=budget)
            == expected
        )


@pytest.mark.parametrize(
    ("point", "expected"),
    (
        ((Fraction(0), Fraction(0)), False),
        ((Fraction(1), Fraction(0)), True),
        ((Fraction(1) - Fraction(1, 2**100), Fraction(0)), False),
        ((Fraction(1) + Fraction(1, 2**100), Fraction(0)), True),
    ),
    ids=("inside", "contact", "exact-inside", "exact-outside"),
)
def test_prepared_supporting_line_reuses_only_equal_exact_content(
    point: tuple[Fraction, ...], expected: bool
) -> None:
    plane = Fraction(1), Fraction(1), Fraction(1)
    with coordinate_reference_budget() as budget:
        with budget.live_storage() as storage:
            sides = PolygonSidePreparation(budget, storage)
            assert sides.outside(plane, point, 128) is expected
            first_work = budget.work_units
            retained = budget.retained_basis_bytes
            live_bytes = storage.bound
            repeated = tuple(
                sides.outside(tuple(list(plane)), tuple(list(point)), 128)
                for _ in range(16)
            )
            assert repeated == (expected,) * 16
            assert budget.work_units - first_work == 16
            assert budget.retained_basis_bytes == retained
            assert storage.bound == live_bytes > 0
            opposite = tuple(-value for value in plane)
            assert sides.outside(opposite, point, 128) is (
                plane[0] * point[0] + plane[1] * point[1] <= plane[2]
            )
        assert storage.bound == 0


def test_prepared_supporting_lines_preserve_exact_contact_and_overlap() -> None:
    first: Polygon = tuple(
        tuple(Fraction(value) for value in point) for point in ((0, 0), (2, 0), (0, 2))
    )
    second: Polygon = tuple(
        tuple(Fraction(value) for value in point) for point in ((2, 0), (2, 2), (0, 2))
    )
    with coordinate_reference_budget() as budget:
        a = prepare_polygon_intersection(first, 1, budget=budget)
        b = prepare_polygon_intersection(second, 1, budget=budget)
        with budget.live_storage() as storage:
            sides = PolygonSidePreparation(budget, storage)
            assert _prepared_separated(a, b, budget=budget, sides=sides)
            assert not _prepared_separated(a, a, budget=budget, sides=sides)
            assert _prepared_separated(b, a, budget=budget, sides=sides)


def test_prepared_supporting_lines_refuse_a_foreign_request() -> None:
    polygon: Polygon = tuple(
        tuple(Fraction(value) for value in point) for point in ((0, 0), (1, 0), (0, 1))
    )
    with (
        coordinate_reference_budget() as first,
        coordinate_reference_budget() as second,
        first.live_storage() as storage,
    ):
        sides = PolygonSidePreparation(first, storage)
        prepared = prepare_polygon_intersection(polygon, 1, budget=first)
        with pytest.raises(ValueError, match="must match the current request"):
            _prepared_separated(prepared, prepared, budget=second, sides=sides)


def test_prepared_supporting_line_hit_refuses_exhausted_work() -> None:
    limits = MeshCertificateLimits(maximum_work_units=1000)
    plane = Fraction(1), Fraction(1), Fraction(1)
    point = Fraction(1), Fraction(0)
    with coordinate_reference_budget(limits) as budget:
        with budget.live_storage() as storage:
            sides = PolygonSidePreparation(budget, storage)
            assert sides.outside(plane, point, 2)
            budget.reserve(limits.maximum_work_units - budget.work_units)
            with pytest.raises(CoordinateEnclosureResourceError) as caught:
                sides.outside(plane, point, 2)
            assert caught.value.resource == "coefficient_work"
            assert budget.work_units == limits.maximum_work_units


@pytest.mark.parametrize("scale", (Fraction(3), Fraction(1, 8)), ids=("long", "short"))
def test_prepared_supporting_lines_reuse_positive_scaled_coefficients(
    scale: Fraction,
) -> None:
    plane = Fraction(1), Fraction(1), Fraction(1)
    point = Fraction(1) - Fraction(1, 2**100), Fraction(0)
    with coordinate_reference_budget() as budget:
        with budget.live_storage() as storage:
            sides = PolygonSidePreparation(budget, storage)
            assert not sides.outside(plane, point, 128)
            before = budget.work_units
            scaled = tuple(scale * value for value in plane)
            assert not sides.outside(scaled, point, 128)
            # Three normalization divisions, one new line index and one query;
            # the already earned point-side decision does not execute again.
            assert budget.work_units - before == 5
            opposite = tuple(-value for value in scaled)
            assert sides.outside(opposite, point, 128)


@pytest.mark.parametrize(
    ("axis", "sign"),
    ((0, 1), (0, -1), (1, 1), (1, -1)),
    ids=("positive-x", "negative-x", "positive-y", "negative-y"),
)
def test_axis_supporting_lines_omit_unit_and_zero_arithmetic(
    axis: int, sign: int
) -> None:
    plane = (
        Fraction(sign if axis == 0 else 0),
        Fraction(sign if axis == 1 else 0),
        Fraction(sign),
    )
    coordinates = [Fraction(0), Fraction(0)]
    coordinates[axis] = Fraction(1) - Fraction(1, 2**100)
    point = tuple(coordinates)
    expected = plane[0] * point[0] + plane[1] * point[1] >= plane[2]
    with coordinate_reference_budget() as budget:
        with budget.live_storage() as storage:
            sides = PolygonSidePreparation(budget, storage)
            before = budget.work_units
            assert sides.outside(plane, point, 128) is expected
            # Query, line index and decision index cost three units; only a
            # scalar comparison and, for a negative normal, one negation remain.
            assert budget.work_units - before == (4 if sign == 1 else 5)
