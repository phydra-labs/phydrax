from collections.abc import Callable, Mapping
from fractions import Fraction
from inspect import getclosurevars
from itertools import permutations
from math import isqrt

import numpy as np
import pytest

from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
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
    plane_key,
    Polygon,
    PolygonSidePreparation,
    prepare_polygon_intersection,
    prepared_plane_key,
    prepared_point_inside,
    reserve_fraction_work,
    signed_measure,
)
from phydrax.meshing import MeshingLimits
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


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("scale", (Fraction(1), Fraction(7, 5), Fraction(1, 1 << 1200)))
def test_normalized_point_membership_preserves_exact_closed_halfplanes(
    reverse: bool, scale: Fraction
) -> None:
    polygon = tuple(
        (scale * x, scale * y)
        for x, y in ((Fraction(0), Fraction(0)), (Fraction(2), Fraction(0)), (Fraction(0), Fraction(3)))
    )
    if reverse:
        polygon = polygon[::-1]
    points = tuple(
        (scale * Fraction(x, 3), scale * Fraction(y, 3))
        for x in range(-3, 10)
        for y in range(-3, 13)
    )
    points += polygon + (
        (scale, scale * Fraction(3, 2)),
        (scale, scale * (Fraction(3, 2) + Fraction(1, 1 << 1200))),
        (scale, scale * (Fraction(3, 2) - Fraction(1, 1 << 1200))),
    )
    orientation = -1 if reverse else 1
    with coordinate_reference_budget() as budget:
        start = budget.work_units
        reference = prepare_polygon_intersection(polygon, orientation, budget=budget)
        expected = tuple(prepared_point_inside(reference, point, budget=budget) for point in points)
        reference_work = budget.work_units - start
    with coordinate_reference_budget() as budget:
        start = budget.work_units
        prepared = prepare_polygon_intersection(
            polygon, orientation, budget=budget, point_membership=True
        )
        actual = tuple(prepared_point_inside(prepared, point, budget=budget) for point in points)
        normalized_work = budget.work_units - start
    assert prepared.point_membership
    assert actual == expected
    assert actual[-3:] == (True, False, True)
    assert normalized_work < reference_work


@pytest.mark.parametrize("pivot", (0, 1, 2))
@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("scale", (Fraction(1), Fraction(7, 5), Fraction(1, 1 << 1200), Fraction(1 << 1200, 3)))
def test_axis_fraction_plane_circuit_matches_complete_generic_plane(
    pivot: int, reverse: bool, scale: Fraction,
) -> None:
    projection = ((1, 2), (2, 0), (0, 1))[pivot]
    shift = Fraction(-7, 11)
    rows = []
    for x, y in ((0, 0), (2, 0), (0, 3)):
        row = [Fraction(0), Fraction(0), Fraction(0)]
        row[pivot] = shift
        row[projection[0]] = shift + scale * x
        row[projection[1]] = -shift + scale * y
        rows.append(tuple(row))
    polygon = tuple(reversed(rows)) if reverse else tuple(rows)
    expected = prepared_plane_key(polygon)
    with coordinate_reference_budget() as budget:
        reserve_fraction_work(polygon, 21, 13, 51, budget=budget)
        reference_work = budget.work_units
    with coordinate_reference_budget() as budget:
        actual = plane_key(polygon, budget=budget)
        actual_work = budget.work_units
    assert actual == expected
    assert actual is not None and actual[1:] == (projection, 1, -1 if reverse else 1)
    assert actual_work < reference_work


@pytest.mark.parametrize(
    "raw",
    (
        ((0, 0, 0), (0, 1, 2), (0, 2, 4)),
        ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        ((0, 0, 0), (1, 2, 3), (2, 4, 6)),
        ((0, 0, 5), (1, 0, 5), (0, 1, 5), (0, 0, 6)),
        ((0, 0), (2, 3)),
    ),
)
def test_axis_plane_circuit_preserves_degeneracy_and_first_plane_semantics(
    raw: tuple[tuple[int, ...], ...],
) -> None:
    polygon = tuple(tuple(Fraction(value) for value in row) for row in raw)
    with coordinate_reference_budget() as budget:
        actual = plane_key(polygon, budget=budget)
    assert actual == prepared_plane_key(polygon)


def test_axis_plane_circuit_does_not_round_a_near_axis_plane() -> None:
    delta = Fraction(1, 1 << 1200)
    polygon = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (delta, Fraction(0), Fraction(1)),
    )
    with coordinate_reference_budget() as budget:
        actual = plane_key(polygon, budget=budget)
    assert actual == prepared_plane_key(polygon)
    assert actual is not None and actual[0][2] == -delta


def test_axis_plane_circuit_preserves_radical_generic_path() -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    with coordinate_reference_budget() as budget:
        root = RadicalElement.sqrt(Fraction(2), budget=budget)
        polygon = (
            (root, Fraction(0), Fraction(0)),
            (root, Fraction(1), Fraction(0)),
            (root, Fraction(0), Fraction(1)),
        )
        actual = plane_key(polygon, budget=budget)
    assert actual == prepared_plane_key(polygon)


@pytest.mark.parametrize("resource", ("work", "memory"))
def test_axis_plane_circuit_admits_its_actual_bound_and_refuses_one_less(
    resource: str,
) -> None:
    polygon = (
        (Fraction(7, 11), Fraction(0), Fraction(0)),
        (Fraction(7, 11), Fraction(2), Fraction(0)),
        (Fraction(7, 11), Fraction(0), Fraction(3)),
    )
    with coordinate_reference_budget() as budget:
        expected = plane_key(polygon, budget=budget)
        work, memory = budget.work_units, budget.peak_bytes_upper
    for slack in (0, -1):
        limits = MeshCertificateLimits(
            maximum_work_units=work + slack if resource == "work" else work,
            maximum_scratch_bytes=memory + slack if resource == "memory" else memory,
        )
        with coordinate_reference_budget(limits) as budget:
            if slack == 0:
                assert plane_key(polygon, budget=budget) == expected
            else:
                with pytest.raises(CoordinateEnclosureResourceError):
                    plane_key(polygon, budget=budget)


def _prepared_affine_pair(
    first: tuple[tuple[Fraction, ...], ...],
    second: tuple[tuple[Fraction, ...], ...],
    *,
    limit: int | None = None,
    repeated: bool = False,
    physical: bool = False,
) -> tuple[Fraction, int, int]:
    from phydrax.geometry import _affine_chart_coverage as owner

    limits = MeshingLimits(maximum_work_units=10**7, maximum_scratch_bytes=10**8)
    with NativeExecutionScope(limits.native_limits()) as scope:
        with scope.batch(10**7, bytes_upper=10**8) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=10**7, maximum_memory_bytes=10**8,
            )
            with (
                budget.activate(), ticket.native_target(),
                budget.live_storage() as cache, budget.live_storage() as sides,
            ):
                preparation = owner._Preparation(
                    budget=budget, cache_storage=cache, side_storage=sides,
                )
                preparation.prepare_points(first)
                preparation.prepare_points(second)
                if repeated:
                    owner._intersection(first, second, preparation)
                before = budget.work_units
                with budget.bound_stage(10**7 if limit is None else limit, 10**8):
                    area = (
                        owner._physical_intersection(first, second, preparation)[2]
                        if physical else owner._intersection(first, second, preparation)
                    )
                return area, budget.required_work_units - before, len(preparation.polygon_cache)


@pytest.mark.parametrize(
    "offset", (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(2)),
    ids=("equal", "overlap", "contact", "separated"),
)
def test_affine_lazy_supports_match_independent_exact_clipping(offset: Fraction) -> None:
    first = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    second = tuple((x + offset, y) for x, y in first)
    with coordinate_reference_budget() as budget:
        expected = intersection(first, second, budget=budget)
    actual, required, polygons = _prepared_affine_pair(first, second)
    assert actual == expected
    assert required > 0
    assert polygons == (0 if offset == 0 or offset >= 1 else 2)
    assert _prepared_affine_pair(first, second, limit=required)[0] == expected
    with pytest.raises(CoordinateEnclosureResourceError):
        _prepared_affine_pair(first, second, limit=required - 1)


def test_affine_lazy_bounds_do_not_cross_requests_after_input_mutation() -> None:
    first = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    second = ((Fraction(1), Fraction(0)), (Fraction(2), Fraction(0)), (Fraction(1), Fraction(1)))
    assert _prepared_affine_pair(first, second)[0] == 0
    object.__setattr__(second[0][0], "_numerator", -1)
    with coordinate_reference_budget() as budget:
        expected = intersection(first, second, budget=budget)
    assert expected > 0
    assert _prepared_affine_pair(first, second)[0] == expected


@pytest.mark.parametrize("order", ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)))
def test_affine_identity_intersection_preserves_cyclic_orientation_and_work(
    order: tuple[int, int, int],
) -> None:
    first = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    second = tuple(
        tuple(Fraction(value.numerator, value.denominator) for value in first[index])
        for index in order
    )
    assert second is not first
    assert all(second[slot] is not first[index] for slot, index in enumerate(order))
    actual, required, polygons = _prepared_affine_pair(first, second)
    with coordinate_reference_budget() as budget:
        expected = intersection(first, second, budget=budget)
    assert actual == expected == Fraction(1, 2)
    assert polygons == 0
    assert _prepared_affine_pair(first, second, limit=required)[0] == expected
    with pytest.raises(CoordinateEnclosureResourceError):
        _prepared_affine_pair(first, second, limit=required - 1)
    reused, reused_work, _ = _prepared_affine_pair(first, second, repeated=True)
    assert reused == expected
    assert 0 < reused_work < required
    assert _prepared_affine_pair(
        first, second, repeated=True, limit=reused_work
    )[0] == expected
    with pytest.raises(CoordinateEnclosureResourceError):
        _prepared_affine_pair(first, second, repeated=True, limit=reused_work - 1)


def test_affine_identity_intersection_rejects_rank_loss_and_rounded_equality() -> None:
    singular = tuple((Fraction(value), Fraction(value)) for value in (0, 1, 2))
    with pytest.raises(ValueError, match="singular source transform"):
        _prepared_affine_pair(singular, singular)
    first = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    shifted = tuple((x + Fraction(1, 2**100), y) for x, y in first)
    actual, _, polygons = _prepared_affine_pair(first, shifted)
    with coordinate_reference_budget() as budget:
        expected = intersection(first, shifted, budget=budget)
    assert actual == expected < Fraction(1, 2)
    assert polygons == 2


type _ExactPolygon = tuple[tuple[Fraction, ...], ...]


def _native_affine_reference(
    first: _ExactPolygon, second: _ExactPolygon,
) -> tuple[_ExactPolygon, tuple[int, ...], Fraction]:
    """Independent former inverse/native-support construction, used only as oracle."""
    from phydrax._meshcore import clip_reference_triangle_exact

    def twice_area(points: _ExactPolygon) -> Fraction:
        return sum(
            (
                a[0] * b[1] - a[1] * b[0]
                for a, b in zip(points, (*points[1:], points[0]), strict=True)
            ),
            Fraction(0),
        )

    def planes(points: _ExactPolygon) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
        orientation = 1 if twice_area(points) > 0 else -1
        return tuple(
            (orientation * (b[1] - a[1]), orientation * (a[0] - b[0]),
             orientation * ((b[1] - a[1]) * a[0] - (b[0] - a[0]) * a[1]))
            for a, b in zip(points, (*points[1:], points[0]), strict=True)
        )

    a, b = tuple(first[1][i] - first[0][i] for i in range(2)), tuple(first[2][i] - first[0][i] for i in range(2))
    determinant = a[0] * b[1] - a[1] * b[0]
    mapped = []
    for point in second:
        x, y = point[0] - first[0][0], point[1] - first[0][1]
        mapped.append(((x * b[1] - y * b[0]) / determinant, (a[0] * y - a[1] * x) / determinant))
    halfplanes = (
        (Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(-1), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1)),
        *planes(tuple(mapped)),
    )
    supports, _, _ = clip_reference_triangle_exact(
        halfplanes, maximum_scratch_bytes=10**8, maximum_work_units=10**7,
    )
    polygon = []
    for left, right in supports:
        p, q = halfplanes[int(left)], halfplanes[int(right)]
        divisor = p[0] * q[1] - p[1] * q[0]
        u, v = (p[2] * q[1] - p[1] * q[2]) / divisor, (p[0] * q[2] - p[2] * q[0]) / divisor
        polygon.append(tuple(first[0][i] + a[i] * u + b[i] * v for i in range(2)))
    if len(polygon) < 3:
        return (), (), Fraction(0)
    if twice_area(tuple(polygon)) < 0:
        polygon.reverse()
    start = polygon.index(min(polygon))
    canonical = tuple(polygon[start:] + polygon[:start])
    physical = planes(first)
    physical = (physical[2], physical[0], physical[1], *planes(second))
    labels = tuple(
        next(
            index for index, plane in enumerate(physical)
            if all(plane[0] * point[0] + plane[1] * point[1] == plane[2] for point in (a, b))
        )
        for a, b in zip(canonical, (*canonical[1:], canonical[0]), strict=True)
    )
    return canonical, labels, twice_area(canonical) / 2


@pytest.mark.parametrize("case", ("equal", "disjoint", "contact", "edge", "contained", "partial", "thin"))
def test_physical_affine_clip_matches_native_oracle_for_all_orientations(case: str) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    first = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    eps = Fraction(1, 2**100)
    variants = {
        "equal": first,
        "disjoint": tuple((x + 2, y) for x, y in first),
        "contact": tuple((x + 1, y) for x, y in first),
        "edge": tuple((x, -y) for x, y in first),
        "contained": tuple((x / 2, y / 2) for x, y in first),
        "partial": tuple((x + Fraction(1, 3), y) for x, y in first),
        "thin": ((Fraction(0), eps), (Fraction(1), eps), (Fraction(0), 2 * eps)),
    }
    value, required, _ = _prepared_affine_pair(first, variants[case], physical=True)
    assert _prepared_affine_pair(
        first, variants[case], physical=True, limit=required
    )[0] == value
    with pytest.raises(CoordinateEnclosureResourceError):
        _prepared_affine_pair(first, variants[case], physical=True, limit=required - 1)
    limits = MeshingLimits(maximum_work_units=10**7, maximum_scratch_bytes=10**8)
    for a in permutations(first):
        for b in permutations(variants[case]):
            with NativeExecutionScope(limits.native_limits()) as scope:
                with scope.batch(10**7, bytes_upper=10**8) as ticket:
                    budget = CoordinateEnclosureBudget(ticket, maximum_work_units=10**7, maximum_memory_bytes=10**8)
                    with budget.activate(), ticket.native_target(), budget.live_storage() as cache, budget.live_storage() as sides:
                        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
                        preparation.prepare_points(a)
                        preparation.prepare_points(b)
                        actual = owner._physical_intersection(a, b, preparation)
                        expected = _native_affine_reference(a, b)
                        assert actual == expected


def test_physical_affine_clip_refuses_malformed_or_singular_operands() -> None:
    valid = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    singular = tuple((Fraction(value), Fraction(value)) for value in (0, 1, 2))
    with pytest.raises(ValueError, match="singular source transform"):
        _prepared_affine_pair(valid, singular, physical=True)
    with pytest.raises(ValueError, match="three exact planar corners"):
        _prepared_affine_pair(valid, valid[:2], physical=True)


@pytest.mark.parametrize("case", ("equal", "contact", "edge", "partial", "thin"))
def test_affine_output_bounds_reuse_matches_fresh_exact_operands(case: str) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner
    from phydrax.geometry._planar_primitives import polygon_bounds

    first = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    eps = Fraction(1, 2**100)
    second = {
        "equal": first,
        "contact": tuple((x + 1, y) for x, y in first),
        "edge": tuple((x, -y) for x, y in first),
        "partial": tuple((x + Fraction(1, 3), y) for x, y in first),
        "thin": ((Fraction(0), eps), (Fraction(1), eps), (Fraction(0), 2 * eps)),
    }[case]
    for a in permutations(first):
        for b in permutations(second):
            with coordinate_reference_budget() as reference:
                expected_bounds = tuple(polygon_bounds(p, budget=reference) for p in (a, b))
                expected = intersection(a, b, budget=reference)
                expected_measures = tuple(signed_measure(p, budget=reference) for p in (a, b))
            with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
                preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
                assert tuple(preparation.bounds(p) for p in (a, b)) == expected_bounds
                before = budget.work_units
                assert tuple(preparation.bounds(p) for p in (a, b)) == expected_bounds
                assert budget.work_units - before == 2 * (1 + 6)
                for _ in range(2):
                    assert owner._intersection(a, b, preparation) == expected
                    assert tuple(preparation.signed_measure(p) for p in (a, b)) == expected_measures
                assert preparation.earned_areas == {}


def test_affine_output_bounds_first_visit_pays_minmax_and_equal_owner_is_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    first = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    equal = tuple(tuple(value for value in point) for point in first)
    original = owner.polygon_bounds
    charged: list[int] = []

    def measured(
        points: tuple[tuple[Fraction, ...], ...], /, *, budget: CoordinateEnclosureBudget,
    ) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
        before = budget.work_units
        result = original(points, budget=budget)
        charged.append(budget.work_units - before)
        return result

    monkeypatch.setattr(owner, "polygon_bounds", measured)
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        first_bounds = preparation.bounds(first)
        assert preparation.bounds(first) is first_bounds
        assert preparation.polygon(first).lower is first_bounds[0]
        assert preparation.bounds(equal) == first_bounds
        assert charged == [16, 16]
        assert len(preparation.bounds_cache) == 2
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        assert preparation.bounds(first) == first_bounds
        assert charged == [16, 16, 16]


@pytest.mark.parametrize("field", ("_numerator", "_denominator"))
def test_affine_output_bounds_refuses_mutated_scalar_owner(field: str) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    points = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        preparation.bounds(points)
        object.__setattr__(points[1][0], field, 2)
        with pytest.raises(ValueError, match="mutated coefficients"):
            preparation.bounds(points)
        with pytest.raises(ValueError, match="mutated coefficients"):
            owner._intersection(points, points, preparation)
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        assert preparation.bounds(points) == ((Fraction(0), Fraction(0)), (points[1][0], Fraction(1)))


def test_affine_output_bounds_refuses_transplanted_request_or_polygon_record() -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    points = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    other = tuple(reversed(points))
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        preparation.bounds(points)
        record = preparation.bounds_cache[id(points)]
        preparation.bounds_cache[id(other)] = record
        with pytest.raises(ValueError, match="current operation and polygon owner"):
            preparation.bounds(other)
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        preparation.bounds_cache[id(points)] = record
        with pytest.raises(ValueError, match="current operation and polygon owner"):
            preparation.bounds(points)


@pytest.mark.parametrize("resource", ("work", "memory"))
@pytest.mark.parametrize("repeated", (False, True))
def test_affine_output_bounds_exact_resource_boundary(resource: str, repeated: bool) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    points = ((Fraction(0), Fraction(0)), (Fraction(7, 11), Fraction(0)), (Fraction(0), Fraction(3)))

    def exercise(limits: MeshCertificateLimits) -> tuple[int, int]:
        with coordinate_reference_budget(limits) as budget, budget.live_storage() as cache, budget.live_storage() as sides:
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            bounds = preparation.bounds(points)
            if repeated:
                assert preparation.bounds(points) is bounds
            return budget.required_work_units, budget.peak_bytes_upper

    work, memory = exercise(MeshCertificateLimits())
    assert exercise(MeshCertificateLimits(maximum_work_units=work, maximum_scratch_bytes=memory)) == (work, memory)
    with pytest.raises(CoordinateEnclosureResourceError):
        exercise(MeshCertificateLimits(
            maximum_work_units=work - (resource == "work"),
            maximum_scratch_bytes=memory - (resource == "memory"),
        ))


def test_affine_point_profile_exact_permutations_and_lookup_saving() -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    points = ((Fraction(0), Fraction(0)), (Fraction(7, 11), Fraction(0)), (Fraction(0), Fraction(3)))
    for triangle in permutations(points):
        with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            assert preparation.bits(triangle) == 4
            for point in triangle:
                preparation.prepare_points(point)
            before = budget.work_units
            assert preparation.bits(triangle) == 4
            assert budget.work_units - before == 3
            preparation.prepare_points(triangle)
            retained = preparation.cache_bytes_upper
            peak = budget.peak_bytes_upper
            before = budget.work_units
            assert preparation.bits(triangle) == 4
            assert budget.work_units - before == 1
            equal = tuple(point for point in triangle)
            assert equal is not triangle
            before = budget.work_units
            assert preparation.bits(equal) == 4
            assert budget.work_units - before == 3
            assert preparation.cache_bytes_upper == retained
            assert budget.peak_bytes_upper == peak


@pytest.mark.parametrize("field", ("_numerator", "_denominator"))
def test_affine_point_profile_refuses_replaced_fraction_integer(field: str) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    scalar = Fraction(1001, 1003)
    points = ((scalar, Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        preparation.prepare_points(points)
        original = getattr(scalar, field)
        replacement = int(str(original))
        assert replacement == original and replacement is not original
        object.__setattr__(scalar, field, replacement)
        with pytest.raises(ValueError, match="mutated coefficients"):
            preparation.bits(points)
        with pytest.raises(ValueError, match="mutated coefficients"):
            preparation.prepare_points(points)
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        assert preparation.bits(points) == 10


def test_affine_point_profile_refuses_transplanted_owner_and_request() -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    points = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    other = tuple(reversed(points))
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        preparation.prepare_points(points)
        record = preparation.point_profiles[id(points)]
        preparation.point_profiles[id(other)] = record
        with pytest.raises(ValueError, match="current operation and point owner"):
            preparation.bits(other)
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        preparation.point_profiles[id(points)] = record
        with pytest.raises(ValueError, match="current operation and point owner"):
            preparation.bits(points)


def test_affine_point_profile_exact_work_boundary() -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    points = ((Fraction(0), Fraction(0)), (Fraction(7, 11), Fraction(0)), (Fraction(0), Fraction(3)))

    def exercise(limits: MeshCertificateLimits) -> int:
        with coordinate_reference_budget(limits) as budget, budget.live_storage() as cache, budget.live_storage() as sides:
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            for point in points:
                preparation.prepare_points(point)
            preparation.prepare_points(points)
            assert preparation.bits(points) == 4
            return budget.required_work_units

    work = exercise(MeshCertificateLimits())
    assert exercise(MeshCertificateLimits(maximum_work_units=work)) == work
    with pytest.raises(CoordinateEnclosureResourceError):
        exercise(MeshCertificateLimits(maximum_work_units=work - 1))


@pytest.mark.parametrize("point_bits", (1, 53, 1075))
def test_affine_accumulator_bound_independent_integer_products(point_bits: int) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    # Worst-sized integers, not clipped values or a copy of the production
    # shoelace algorithm. Distinct odd denominators defeat a common-dyadic
    # denominator assumption; every partial sum must fit independently.
    individual_bits = 32 * point_bits + 256
    total = Fraction(0)
    for count in range(1, 129):
        value = Fraction(
            (1 << individual_bits) - 1,
            (1 << individual_bits) - 2 * count - 1,
        )
        total += value
        bound = owner._chain_accumulator_bits(point_bits, count)
        assert abs(total.numerator).bit_length() <= bound
        assert total.denominator.bit_length() <= bound
        assert max(value.numerator.bit_length(), value.denominator.bit_length()) <= bound


@pytest.mark.parametrize("resource", ("work", "memory"))
def test_affine_accumulator_exact_tiling_resource_boundary(resource: str) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    points = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=np.float64)
    targets = tuple(points[list(indices)] for indices in ((0, 1, 2), (0, 2, 3)))
    sources = tuple(points[list(indices)] for indices in ((0, 1, 3), (1, 2, 3)))
    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))

    def exercise(limits: MeshCertificateLimits) -> tuple[int, int]:
        with (
            coordinate_reference_budget(limits) as budget,
            budget.live_storage() as cache,
            budget.live_storage() as sides,
        ):
            preparation = owner._Preparation(
                budget=budget, cache_storage=cache, side_storage=sides,
            )
            prepared = owner._prepare_chain_records(targets, sources, preparation)
            assert owner._chain_tiling(targets, sources, pairs, prepared, preparation)
            assert set(preparation.earned_areas.values()) == {Fraction(1, 4)}
            return budget.required_work_units, budget.peak_bytes_upper

    work, memory = exercise(MeshCertificateLimits())
    assert exercise(MeshCertificateLimits(
        maximum_work_units=work, maximum_scratch_bytes=memory,
    )) == (work, memory)
    with pytest.raises(CoordinateEnclosureResourceError):
        exercise(MeshCertificateLimits(
            maximum_work_units=work - (resource == "work"),
            maximum_scratch_bytes=memory - (resource == "memory"),
        ))


@pytest.mark.parametrize("offset", (0.0, 0.5, 1.0, 2.0))
def test_affine_accumulator_matches_profiled_additions(
    offset: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    targets = (points,)
    sources = (points + np.asarray((offset, 0, 0)),)
    original_equal = owner._chain_areas_equal
    captured: list[tuple[Fraction, ...]] = []

    def capture(
        rows: tuple[np.ndarray, ...],
        areas: list[Fraction],
        records: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
        preparation: owner._Preparation,
    ) -> bool:
        captured.append(tuple(areas))
        return original_equal(rows, areas, records, preparation)

    monkeypatch.setattr(owner, "_chain_areas_equal", capture)
    results = []
    charges = []
    for legacy in (True, False):
        with (
            coordinate_reference_budget() as budget,
            budget.live_storage() as cache,
            budget.live_storage() as sides,
        ):
            preparation = owner._Preparation(
                budget=budget, cache_storage=cache, side_storage=sides,
            )
            prepared = owner._prepare_chain_records(targets, sources, preparation)
            before = budget.work_units
            if legacy:
                target_area, source_area = [Fraction(0)], [Fraction(0)]
                budget.reserve(2)  # The two retained accumulator owners.
                target, plane = prepared.records[id(targets[0])]
                source, source_plane = prepared.records[id(sources[0])]
                assert plane is not None and source_plane is not None
                with budget.temporary_scope():
                    area = preparation.area(
                        preparation.projected(target, plane[1]),
                        preparation.projected(source, plane[1]),
                    )

                    def addition() -> tuple[None, int]:
                        target_area[0] += area
                        source_area[0] += area
                        return None, 2

                    preparation.stage(
                        2, 2,
                        8 * preparation.bits((area, target_area[0], source_area[0])) + 64,
                        addition,
                    )
                results.append(
                    capture(targets, target_area, prepared.records, preparation)
                    and capture(sources, source_area, prepared.records, preparation)
                )
            else:
                results.append(owner._chain_tiling(
                    targets, sources, ((0, 0),), prepared, preparation,
                ))
            charges.append(budget.work_units - before)
    assert results == [offset == 0.0, offset == 0.0]
    middle = len(captured) // 2
    assert captured[:middle] == captured[middle:]
    assert charges[1] - charges[0] == (1 if offset < 1.0 else -4)


def test_affine_accumulator_admits_capped_dimensions_dynamically(
    record_property: Callable[[str, object], None],
) -> None:
    from phydrax._host_storage import declared_integer_storage_bytes
    from phydrax.geometry import _affine_chart_coverage as owner

    # Measured capped-cylinder r4 dimensions, but a synthetic repeated triangle:
    # this checks storage admission, not a capped-cylinder scientific result.
    points = np.asarray(((0, 0, 0), (2.0**-54, 0, 0), (0, 1, 0)))
    rows = (points,) * 778
    pairs = tuple((index, index) for index in range(778))
    limits = MeshCertificateLimits(maximum_scratch_bytes=81_920_000)
    bits = owner._chain_accumulator_bits(107, 778)
    assert bits == 2_863_050
    assert 1024 + 1556 * (512 + 2 * declared_integer_storage_bytes(bits)) == 4_456_111_168
    with (
        coordinate_reference_budget(limits) as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(
            budget=budget, cache_storage=cache, side_storage=sides,
        )
        prepared = owner._prepare_chain_records(rows, rows, preparation)
        assert prepared.point_bits == 107
        scratch_before = budget.temporary_bytes_upper - budget._live_bytes_upper
        assert owner._chain_tiling(rows, rows, pairs, prepared, preparation)
        assert 0 < budget.peak_bytes_upper < limits.maximum_scratch_bytes
        record_property("dynamic_peak_bytes", budget.peak_bytes_upper)
        assert budget.temporary_bytes_upper - budget._live_bytes_upper == scratch_before


@pytest.mark.parametrize("exponent", (0, 52, 1023))
def test_affine_accumulator_bounds_extreme_dyadic_intersections(exponent: int) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    large = Fraction(2**exponent)
    small = Fraction(1, 2**min(1074, exponent + 51))
    zero = Fraction(0)
    first = ((-large, zero), (large, zero), (zero, small))
    second = ((zero, -small), (large, small), (-large, small))
    point_bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for triangle in (first, second) for point in triangle for value in point
    )
    with coordinate_reference_budget() as budget:
        area = intersection(first, second, budget=budget)
    assert area > 0
    bound = owner._chain_accumulator_bits(point_bits, 1)
    assert max(area.numerator.bit_length(), area.denominator.bit_length()) <= bound


@pytest.mark.parametrize("denominators", ("coprime", "dyadic"))
def test_affine_accumulator_dynamic_long_sums_and_exact_live_storage(
    denominators: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._host_storage import declared_integer_storage_bytes
    from phydrax.geometry import _affine_chart_coverage as owner

    count = 192
    primes = tuple(
        value for value in range(1009, 4000)
        if all(value % divisor for divisor in range(2, isqrt(value) + 1))
    )
    values = tuple(
        Fraction((1 << 257) - 1, primes[index] if denominators == "coprime" else 1 << (17 * index + 1))
        for index in range(count)
    )
    pairs = tuple((index % 2, (index // 2) % 3) for index in range(count))
    targets = tuple(np.zeros((3, 3)) for _ in range(2))
    sources = tuple(np.zeros((3, 3)) for _ in range(3))
    triangle = ((Fraction(0), Fraction(0)),) * 3
    plane: owner._PlaneKey = ((Fraction(1), Fraction(0)), (0, 1), 0, 0)
    prepared = owner._PreparedChains(
        (), (), {id(row): (triangle, plane) for row in (*targets, *sources)}, 4096,
    )
    expected_target = [Fraction(0)] * len(targets)
    expected_source = [Fraction(0)] * len(sources)
    expected_inputs = []
    expected_states = []
    for (first, second), area in zip(pairs, values, strict=True):
        expected_inputs.append((area, expected_target[first], expected_source[second]))
        expected_target[first] += area
        expected_source[second] += area
        expected_states.append((tuple(expected_target), tuple(expected_source)))

    def storage(value: Fraction) -> int:
        return (
            512
            + declared_integer_storage_bytes(value.numerator.bit_length())
            + declared_integer_storage_bytes(value.denominator.bit_length())
        )

    additions: list[tuple[Fraction, Fraction]] = []
    original_add = Fraction.__add__

    def add(left: Fraction, right: Fraction) -> Fraction:
        additions.append((left, right))
        return original_add(left, right)

    monkeypatch.setattr(Fraction, "__add__", add)
    with (
        coordinate_reference_budget() as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        baseline = budget.temporary_bytes_upper
        starting_work = budget.work_units
        original_bits = preparation.bits
        original_stage = preparation.stage
        schedule = 1024 + 5 * (
            32 + declared_integer_storage_bytes(
                prepared.point_bits.bit_length() + len(pairs).bit_length() + 16
            )
        )
        actual_bytes = schedule + 5 * storage(Fraction(0))
        peak = baseline + actual_bytes
        visited = 0
        comparisons = 0
        stream = iter(values)

        def area(first: owner._Triangle, second: owner._Triangle) -> Fraction:
            return next(stream)

        def projected(points: owner._Triangle, axes: tuple[int, ...]) -> owner._Triangle:
            return points

        def bits(current: tuple[object, ...]) -> int:
            nonlocal visited
            assert current == expected_inputs[visited]
            assert budget.temporary_bytes_upper == baseline + actual_bytes
            before = budget.work_units
            result = original_bits(current)
            assert budget.work_units - before == 3
            visited += 1
            return result

        def stage(
            bound: int, terms: int, bit_bound: int,
            operation: Callable[[], tuple[None, int]],
        ) -> None:
            nonlocal actual_bytes, peak
            captured = getclosurevars(operation).nonlocals
            live = captured["area_storage"]
            target_area = captured["target_area"]
            source_area = captured["source_area"]
            assert live.bound == actual_bytes
            first, second = pairs[visited - 1]
            current_bits = max(
                max(value.numerator.bit_length(), value.denominator.bit_length(), 1)
                for value in expected_inputs[visited - 1]
            )
            upper = (
                512 + declared_integer_storage_bytes(2 * current_bits + 1)
                + declared_integer_storage_bytes(2 * current_bits)
            )
            simultaneous = (
                actual_bytes - storage(target_area[first])
                - storage(source_area[second]) + 2 * upper
            )
            scratch = 256 + 2 * (
                512 + 32 * 3 + 2 * declared_integer_storage_bytes(8 * current_bits + 64)
            )
            assert (bound, terms, bit_bound) == (2, 2, 8 * current_bits + 64)
            original_stage(bound, terms, bit_bound, operation)
            assert live.bound == simultaneous
            assert budget.temporary_bytes_upper == baseline + simultaneous + scratch
            assert (tuple(target_area), tuple(source_area)) == expected_states[visited - 1]
            actual_bytes = schedule + sum(storage(value) for value in (*target_area, *source_area))
            assert actual_bytes <= simultaneous
            peak = max(peak, baseline + simultaneous + scratch)

        def equal(
            rows: tuple[np.ndarray, ...], accumulated: list[Fraction],
            records: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
            current: owner._Preparation,
        ) -> bool:
            nonlocal comparisons
            assert accumulated == (expected_target if rows is targets else expected_source)
            assert budget.temporary_bytes_upper == baseline + actual_bytes
            comparisons += 1
            return True

        monkeypatch.setattr(preparation, "area", area)
        monkeypatch.setattr(preparation, "projected", projected)
        monkeypatch.setattr(preparation, "bits", bits)
        monkeypatch.setattr(preparation, "stage", stage)
        monkeypatch.setattr(owner, "_chain_areas_equal", equal)
        assert owner._chain_tiling(targets, sources, pairs, prepared, preparation)
        assert visited == count and comparisons == 2
        assert budget.work_units - starting_work == 5 + 6 * count
        assert budget.peak_bytes_upper == peak
        assert budget.temporary_bytes_upper == baseline
        assert budget._live_bytes_upper == baseline
    assert additions == [
        addition
        for area, target, source in expected_inputs
        for addition in ((target, area), (source, area))
    ]


@pytest.mark.parametrize("boundary", ("exact", "one-byte-short", "second-result-short"))
def test_affine_accumulator_atomic_preadd_boundary_and_cleanup(
    boundary: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._host_storage import declared_integer_storage_bytes
    from phydrax.geometry import _affine_chart_coverage as owner

    targets, sources = (np.zeros((3, 3)),), (np.zeros((3, 3)),)
    triangle = ((Fraction(0), Fraction(0)),) * 3
    plane: owner._PlaneKey = ((Fraction(1), Fraction(0)), (0, 1), 0, 0)
    prepared = owner._PreparedChains(
        (), (), {id(row): (triangle, plane) for row in (*targets, *sources)}, 401,
    )
    value = Fraction((1 << 400) - 1, (1 << 401) + 1)
    current_bits = max(value.numerator.bit_length(), value.denominator.bit_length())
    zero_bytes = 512 + declared_integer_storage_bytes(0) + declared_integer_storage_bytes(1)
    result_bytes = (
        512 + declared_integer_storage_bytes(2 * current_bits + 1)
        + declared_integer_storage_bytes(2 * current_bits)
    )
    schedule = 1024 + 2 * (
        32 + declared_integer_storage_bytes(prepared.point_bits.bit_length() + 1 + 16)
    )
    scratch = 256 + 2 * (
        512 + 32 * 3 + 2 * declared_integer_storage_bytes(8 * current_bits + 64)
    )
    captured_rows: list[list[Fraction]] = []
    additions: list[tuple[Fraction, Fraction]] = []
    original_add = Fraction.__add__

    def add(left: Fraction, right: Fraction) -> Fraction:
        additions.append((left, right))
        return original_add(left, right)

    monkeypatch.setattr(Fraction, "__add__", add)
    with (
        coordinate_reference_budget() as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        baseline = budget.temporary_bytes_upper
        before_work = budget.work_units
        original_stage = preparation.stage
        required = baseline + schedule + 2 * result_bytes + scratch
        shortfall = (
            0 if boundary == "exact" else
            1 if boundary == "one-byte-short" else result_bytes - zero_bytes
        )

        def area(first: owner._Triangle, second: owner._Triangle) -> Fraction:
            return value

        def projected(points: owner._Triangle, axes: tuple[int, ...]) -> owner._Triangle:
            return points

        def stage(
            bound: int, terms: int, bits: int,
            operation: Callable[[], tuple[None, int]],
        ) -> None:
            captured = getclosurevars(operation).nonlocals
            captured_rows.extend((captured["target_area"], captured["source_area"]))
            original_stage(bound, terms, bits, operation)

        def equal(
            rows: tuple[np.ndarray, ...], accumulated: list[Fraction],
            records: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
            current: owner._Preparation,
        ) -> bool:
            return accumulated == [value]

        monkeypatch.setattr(preparation, "area", area)
        monkeypatch.setattr(preparation, "projected", projected)
        monkeypatch.setattr(preparation, "stage", stage)
        monkeypatch.setattr(owner, "_chain_areas_equal", equal)
        with budget.bound_stage(100, required - shortfall):
            if shortfall:
                with pytest.raises(CoordinateEnclosureResourceError):
                    owner._chain_tiling(targets, sources, ((0, 0),), prepared, preparation)
                assert captured_rows == [[Fraction(0)], [Fraction(0)]]
                assert additions == []
                assert budget.work_units - before_work == 6
            else:
                assert owner._chain_tiling(targets, sources, ((0, 0),), prepared, preparation)
                assert captured_rows == [[value], [value]]
                assert additions == [(Fraction(0), value), (Fraction(0), value)]
                assert budget.work_units - before_work == 8
                assert budget.peak_bytes_upper == required
            assert budget.temporary_bytes_upper == baseline
            assert budget._live_bytes_upper == baseline


@pytest.mark.parametrize("axes", tuple(permutations(range(3))))
@pytest.mark.parametrize("reverse", (False, True))
def test_affine_plane_filter_matches_unfiltered_areas(
    axes: tuple[int, ...], reverse: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    base = np.asarray(((0, 0, 0), (2, 0, 0), (0, 2, 0)), dtype=np.float64)
    crossing = np.asarray(((0, 0, -1), (2, 0, 1), (0, 2, 1)), dtype=np.float64)
    variants = (
        base, base + (0, 0, 1), crossing,
        base + (2, 0, 0), base + (0.5, 0, 0),
    )
    targets = (base[:, axes],)
    sources = tuple(row[::-1 if reverse else 1, axes] for row in variants)
    pairs = tuple((0, index) for index in (2, 0, 3, 1, 4))
    expected_target = [Fraction(0)]
    expected_source = [Fraction(0)] * len(sources)
    decisions = []
    original_equal = owner._chain_areas_equal
    with (
        coordinate_reference_budget() as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(targets, sources, preparation)
        # Independent old path: every candidate adds its exact area, including
        # the zeros proved by distinct supporting planes.
        for first, second in pairs:
            target, plane = prepared.records[id(targets[first])]
            source, other = prepared.records[id(sources[second])]
            assert plane is not None and other is not None
            area = Fraction(0) if plane[0] != other[0] else preparation.area(
                preparation.projected(target, plane[1]),
                preparation.projected(source, plane[1]),
            )
            expected_target[first] += area
            expected_source[second] += area
        expected_decision = (
            original_equal(targets, expected_target, prepared.records, preparation)
            and original_equal(sources, expected_source, prepared.records, preparation)
        )

    def equal(
        rows: tuple[np.ndarray, ...], areas: list[Fraction],
        records: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
        preparation: owner._Preparation,
    ) -> bool:
        assert areas == (expected_target if rows is targets else expected_source)
        decisions.append(original_equal(rows, areas, records, preparation))
        return True  # Inspect both sides even when the first does not tile.

    monkeypatch.setattr(owner, "_chain_areas_equal", equal)
    with (
        coordinate_reference_budget() as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(targets, sources, preparation)
        original_area = preparation.area
        visited = []

        def area(first: owner._Triangle, second: owner._Triangle) -> Fraction:
            visited.append(second)
            return original_area(first, second)

        monkeypatch.setattr(preparation, "area", area)
        assert owner._chain_tiling(targets, sources, pairs, prepared, preparation)
        assert all(decisions) == expected_decision
        assert preparation.plane_comparisons == 5
        assert preparation.plane_mismatches == 2
        assert preparation.zero_area_pairs == 1
        expected_visited = []
        for index in (0, 3, 4):
            points, plane = prepared.source[index]
            assert plane is not None
            expected_visited.append(preparation.projected(points, plane[1]))
        assert visited == expected_visited


@pytest.mark.parametrize("resource", ("work", "memory"))
def test_affine_plane_filter_mismatch_resource_boundary(resource: str) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    base = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    targets, sources = (base,), (base + (0, 0, 1),)
    pairs = ((0, 0),) * 31

    def exercise(limits: MeshCertificateLimits) -> tuple[int, int]:
        with (
            coordinate_reference_budget(limits) as budget,
            budget.live_storage() as cache,
            budget.live_storage() as sides,
        ):
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            prepared = owner._prepare_chain_records(targets, sources, preparation)
            assert not owner._chain_tiling(targets, sources, pairs, prepared, preparation)
            assert preparation.plane_comparisons == preparation.plane_mismatches == 31
            assert preparation.earned_areas == {}
            return budget.required_work_units, budget.peak_bytes_upper

    work, memory = exercise(MeshCertificateLimits())
    assert exercise(MeshCertificateLimits(
        maximum_work_units=work, maximum_scratch_bytes=memory,
    )) == (work, memory)
    with pytest.raises(CoordinateEnclosureResourceError):
        exercise(MeshCertificateLimits(
            maximum_work_units=work - (resource == "work"),
            maximum_scratch_bytes=memory - (resource == "memory"),
        ))


@pytest.mark.parametrize("kind", ("collision", "mutation", "none"))
def test_affine_plane_filter_uses_current_exact_keys(
    kind: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from phydrax.geometry import _affine_chart_coverage as owner

    targets, sources = (np.zeros((3, 3)),), (np.zeros((3, 3)),)
    coefficient = Fraction(0)
    other = Fraction(sys.hash_info.modulus if kind == "collision" else 0)
    assert hash(coefficient) == hash(other)
    plane: owner._PlaneKey = ((coefficient,), (0, 1), 0, 0)
    source_plane: owner._PlaneKey | None = (
        None if kind == "none" else ((other,), (0, 1), 0, 0)
    )
    triangle = ((Fraction(0), Fraction(0)),) * 3
    prepared = owner._PreparedChains(
        (), (), {
            id(targets[0]): (triangle, plane),
            id(sources[0]): (triangle, source_plane),
        }, 1,
    )
    if kind == "mutation":
        # Deliberately corrupt a Fraction after preparation. Streaming equality
        # must not reuse a stale grouping/hash verdict.
        object.__setattr__(other, "_numerator", 1)

    def forbidden(*args: object) -> Fraction:
        raise AssertionError("a distinct or absent plane reached exact clipping")

    monkeypatch.setattr(owner._Preparation, "area", forbidden)
    monkeypatch.setattr(owner, "_chain_areas_equal", lambda *args: False)
    with (
        coordinate_reference_budget() as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        before = budget.work_units
        assert not owner._chain_tiling(targets, sources, ((0, 0),), prepared, preparation)
        comparisons = int(kind != "none")
        assert preparation.plane_comparisons == preparation.plane_mismatches == comparisons
        assert budget.work_units - before == 2 + comparisons


def test_affine_plane_filter_does_not_relax_candidate_cap() -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    target = np.asarray(((0, 0, 0), (2, 0, 0), (0, 2, 0)), dtype=np.float64)
    source = np.asarray(((0, 0, -1), (2, 0, 1), (0, 2, 1)), dtype=np.float64)
    targets, sources = (target,), (source, source.copy())
    assert owner._chain_candidates(
        targets, sources, MeshCertificateLimits(maximum_candidate_pairs=2),
    ) is None
    candidates = owner._chain_candidates(
        targets, sources, MeshCertificateLimits(maximum_candidate_pairs=3),
    )
    assert candidates is not None
    assert len(candidates.cross) == 2


def test_affine_plane_filter_measures_avoided_zero_additions(
    monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, object], None],
) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    targets, sources = (np.zeros((3, 3)),), (np.zeros((3, 3)),)
    triangle = ((Fraction(0), Fraction(0)),) * 3
    first_plane: owner._PlaneKey = ((Fraction(0),), (0, 1), 0, 0)
    second_plane: owner._PlaneKey = ((Fraction(1),), (0, 1), 0, 0)
    prepared = owner._PreparedChains(
        (), (), {
            id(targets[0]): (triangle, first_plane),
            id(sources[0]): (triangle, second_plane),
        }, 1,
    )
    count = 101
    charges = []
    monkeypatch.setattr(owner, "_chain_areas_equal", lambda *args: True)
    for legacy in (True, False):
        with (
            coordinate_reference_budget() as budget,
            budget.live_storage() as cache,
            budget.live_storage() as sides,
        ):
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            before = budget.work_units
            if legacy:
                budget.reserve(2)
                target_area, source_area = [Fraction(0)], [Fraction(0)]
                for _ in range(count):
                    with budget.temporary_scope():
                        area = Fraction(0)
                        bits = preparation.bits((area, target_area[0], source_area[0]))

                        def addition() -> tuple[None, int]:
                            target_area[0] += area
                            source_area[0] += area
                            return None, 2

                        preparation.stage(2, 2, 8 * bits + 64, addition)
                assert target_area == source_area == [Fraction(0)]
            else:
                assert owner._chain_tiling(
                    targets, sources, ((0, 0),) * count, prepared, preparation,
                )
                assert preparation.plane_mismatches == count
            charges.append(budget.work_units - before)
    assert charges == [2 + 5 * count, 2 + count]
    record_property("plane_mismatch_pairs", count)
    record_property("plane_filter_saved_work", charges[0] - charges[1])
    record_property("plane_filter_saved_additions", 2 * count)


@pytest.mark.parametrize("offset", (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(2)))
@pytest.mark.parametrize("resource", ("work", "memory"))
def test_affine_symmetric_area_reuse_exact_decisions_and_bounds(
    offset: Fraction, resource: str,
) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    first = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    second = tuple((x + offset, y) for x, y in first)
    with coordinate_reference_budget() as budget:
        expected = intersection(first, second, budget=budget)

    def exercise(limits: MeshCertificateLimits) -> tuple[int, int]:
        with coordinate_reference_budget(limits) as budget, budget.live_storage() as cache, budget.live_storage() as sides:
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            preparation.prepare_points(first)
            preparation.prepare_points(second)
            assert preparation.area(first, second) == expected
            calls = preparation.clip_calls
            count = len(preparation.earned_areas)
            before = budget.work_units
            assert preparation.area(second, first) == expected
            assert budget.work_units - before == (1 if first == second else 2)
            assert preparation.clip_calls == calls
            assert len(preparation.earned_areas) == count
            return budget.required_work_units, budget.peak_bytes_upper

    work, memory = exercise(MeshCertificateLimits())
    assert exercise(MeshCertificateLimits(maximum_work_units=work, maximum_scratch_bytes=memory)) == (work, memory)
    with pytest.raises(CoordinateEnclosureResourceError):
        exercise(MeshCertificateLimits(
            maximum_work_units=work - (resource == "work"),
            maximum_scratch_bytes=memory - (resource == "memory"),
        ))


def test_affine_symmetric_area_rejects_mutation_and_recomputes_new_request() -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    first = ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    second = ((Fraction(1, 2), Fraction(0)), (Fraction(3, 2), Fraction(0)), (Fraction(1, 2), Fraction(1)))
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        preparation.prepare_points(first)
        preparation.prepare_points(second)
        assert preparation.area(first, second) == Fraction(1, 8)
        for point in second:
            value = point[0]
            object.__setattr__(value, "_numerator", value.numerator + 4 * value.denominator)
        with pytest.raises(ValueError, match="mutated coefficients"):
            preparation.area(second, first)
    with coordinate_reference_budget() as budget:
        expected = intersection(second, first, budget=budget)
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        preparation.prepare_points(first)
        preparation.prepare_points(second)
        assert preparation.area(second, first) == expected
        assert len(preparation.earned_areas) == 1


@pytest.mark.parametrize("order", tuple(permutations(range(3))))
def test_affine_chain_bijection_matches_independent_overlay(
    order: tuple[int, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    square = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=float)
    targets = (square[[0, 1, 2]], square[[0, 2, 3]])
    sources = tuple(row[list(order)].copy() for row in reversed(targets))
    captured: list[tuple[Fraction, ...]] = []
    original = owner._chain_areas_equal

    def capture(
        rows: tuple[np.ndarray, ...], areas: list[Fraction],
        records: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
        preparation: owner._Preparation,
        *,
        matched_areas: Mapping[int, Fraction] | None = None,
    ) -> bool:
        captured.append(tuple(areas))
        return original(
            rows, areas, records, preparation, matched_areas=matched_areas,
        )

    monkeypatch.setattr(owner, "_chain_areas_equal", capture)
    for identity in (False, True):
        with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            prepared = owner._prepare_chain_records(targets, sources, preparation)
            if identity:
                with budget.temporary_scope():
                    assert owner._chain_identity_mapping(targets, sources, prepared, preparation) == (1, 0)
                    assert owner._chain_identity_mapping(targets, sources, prepared, preparation) == (1, 0)

                def forbidden(*args: object) -> Fraction:
                    raise AssertionError("An exact bijection must not intersect triangle pairs")

                monkeypatch.setattr(preparation, "area", forbidden)
                assert owner._chain_identity_tiling(targets, sources, prepared, preparation)
                assert preparation.earned_areas == {}
                assert preparation.clip_calls == 0
            else:
                assert owner._chain_tiling(
                    targets, sources, ((0, 0), (0, 1), (1, 0), (1, 1)),
                    prepared, preparation,
                )
    assert captured == [(Fraction(1, 2), Fraction(1, 2))] * 4


@pytest.mark.parametrize(
    "change", ("duplicate", "missing", "extra", "corner", "contact", "partial", "plane", "mutable", "mutable-target"),
)
def test_affine_chain_bijection_refuses_incomplete_or_changed_owners(change: str) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    first = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    second = first + np.asarray((2, 0, 0))
    targets = (first, second)
    sources = (second.copy(), first.copy())
    match change:
        case "duplicate":
            targets = (first, first.copy())
            sources = (first.copy(), first.copy())
        case "missing":
            sources = sources[:1]
        case "extra":
            sources = (*sources, first.copy())
        case "corner":
            sources[0][0, 0] += 0.25
        case "contact":
            sources = tuple(row + np.asarray((1, 0, 0)) for row in sources)
        case "partial":
            sources = tuple(row + np.asarray((0.25, 0, 0)) for row in sources)
        case "plane":
            sources = tuple(row + np.asarray((0, 0, 1)) for row in sources)
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(targets, sources, preparation)
        if change in ("mutable", "mutable-target"):
            (sources if change == "mutable" else targets)[0][0, 0] += 0.25
            with pytest.raises(ValueError, match="unchanged row owners"):
                owner._chain_identity_tiling(targets, sources, prepared, preparation)
        else:
            assert owner._chain_identity_tiling(targets, sources, prepared, preparation) is None
            assert preparation.earned_areas == {}
            pairs = tuple((i, j) for i in range(len(targets)) for j in range(len(sources)))
            assert not owner._chain_tiling(targets, sources, pairs, prepared, preparation)


@pytest.mark.parametrize("operand", ("corner", "plane"))
@pytest.mark.parametrize("role", ("target", "source"))
def test_affine_chain_bijection_rejects_mutated_fraction_owner(operand: str, role: str) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    row = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    targets, sources = (row,), (row.copy(),)
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(targets, sources, preparation)
        points, plane = (prepared.target if role == "target" else prepared.source)[0]
        assert plane is not None
        value = points[0][0] if operand == "corner" else plane[0][0]
        object.__setattr__(value, "_numerator", value.numerator + value.denominator)
        with pytest.raises(ValueError, match="mutated coefficients"):
            owner._chain_identity_tiling(targets, sources, prepared, preparation)


@pytest.mark.parametrize("resource", ("work", "memory"))
def test_affine_chain_bijection_exact_resource_boundary(resource: str) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    row = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    targets = tuple(row + np.asarray((2 * index, 0, 0)) for index in range(8))
    sources = tuple(triangle[::-1].copy() for triangle in reversed(targets))

    def exercise(limits: MeshCertificateLimits) -> tuple[int, int]:
        with coordinate_reference_budget(limits) as budget, budget.live_storage() as cache, budget.live_storage() as sides:
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            prepared = owner._prepare_chain_records(targets, sources, preparation)
            assert owner._chain_identity_tiling(targets, sources, prepared, preparation)
            return budget.required_work_units, budget.peak_bytes_upper

    work, memory = exercise(MeshCertificateLimits())
    assert exercise(MeshCertificateLimits(
        maximum_work_units=work, maximum_scratch_bytes=memory,
    )) == (work, memory)
    with pytest.raises(CoordinateEnclosureResourceError):
        exercise(MeshCertificateLimits(
            maximum_work_units=work - (resource == "work"),
            maximum_scratch_bytes=memory - (resource == "memory"),
        ))


def test_affine_chain_bijection_capped_size_work_and_storage(
    record_property: Callable[[str, object], None],
) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    # The measured chain/pair dimensions, not a capped-cylinder fidelity claim.
    count = 778
    row = np.asarray(((0, 0, 0), (2.0**-54, 0, 0), (0, 1, 0)))
    targets = tuple(row + np.asarray((0, 2 * index, 0)) for index in range(count))
    sources = tuple(triangle[::-1].copy() for triangle in reversed(targets))
    pairs = tuple(
        (index, count - 1 - (index + offset) % count)
        for offset in range(14) for index in range(count)
    )[:10758]
    charges = []
    for identity in (False, True):
        limits = MeshCertificateLimits(maximum_work_units=10_000_000, maximum_scratch_bytes=81_920_000)
        with coordinate_reference_budget(limits) as budget, budget.live_storage() as cache, budget.live_storage() as sides:
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            prepared = owner._prepare_chain_records(targets, sources, preparation)
            before = budget.work_units
            if identity:
                assert owner._chain_identity_tiling(targets, sources, prepared, preparation)
                assert preparation.earned_areas == {}
            else:
                assert owner._chain_tiling(targets, sources, pairs, prepared, preparation)
            charges.append(budget.work_units - before)
            assert budget.peak_bytes_upper < 81_920_000
            record_property(f"identity_{identity}_peak_bytes", budget.peak_bytes_upper)
    record_property("bijection_saved_work", charges[0] - charges[1])
    assert charges[0] - charges[1] > 45382


def test_affine_chain_bijection_rejects_projection_support_mismatch() -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    row = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    targets, sources = (row,), (row.copy(),)
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(targets, sources, preparation)
        points, plane = prepared.source[0]
        assert plane is not None
        changed = (points, (plane[0], (0, 2), plane[2], plane[3]))
        records = dict(prepared.records)
        records[id(sources[0])] = changed
        mismatched = owner._PreparedChains(
            prepared.target, (changed,), records, prepared.point_bits,
        )
        assert owner._chain_identity_tiling(targets, sources, mismatched, preparation) is None


@pytest.mark.parametrize("limit", ("maximum_candidate_pairs", "maximum_distance_evaluations", "maximum_source_samples"))
def test_affine_chain_early_bijection_skips_cross_candidates_and_retains_limits(
    limit: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartCover

    square = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=float)
    targets = (square[[0, 1, 2]], square[[0, 2, 3]])
    cover = SourceBoundaryChartCover(
        np.asarray([row[::-1] for row in reversed(targets)]),
        np.zeros(2), "certified", True, "scoped-chain", "r1",
    )
    original = owner.embedding_state_owner.candidate_pairs
    calls = []
    original_identity = owner._chain_identity_tiling
    identity_completed = False

    def identity(
        target_rows: tuple[np.ndarray, ...], source_rows: tuple[np.ndarray, ...],
        prepared: owner._PreparedChains, preparation: owner._Preparation,
        *, source_keys: owner._SourceIdentityOperands | None = None,
    ) -> bool | None:
        nonlocal identity_completed
        result = original_identity(
            target_rows, source_rows, prepared, preparation, source_keys=source_keys,
        )
        identity_completed = result is True
        return result

    def candidates(
        lower: np.ndarray, upper: np.ndarray, capacity: int, /,
        *, other: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        assert other is None, "Exact bijection must not prepare cross-chain candidates"
        assert identity_completed
        calls.append(capacity)
        return original(lower, upper, capacity)

    monkeypatch.setattr(owner.embedding_state_owner, "candidate_pairs", candidates)
    monkeypatch.setattr(owner, "_chain_identity_tiling", identity)
    required = 2 if limit == "maximum_source_samples" else 4
    for allowance in (required, required - 1):
        identity_completed = False
        result, proven, _ = owner.prove_affine_chart_chain(
            [(row, 0.0, 0.0) for row in targets], cover,
            MeshCertificateLimits(**{limit: allowance}), 0.0,
            source_identity=("scoped-chain", "r1"),
        )
        if allowance == required:
            assert result == 0.0
            counts = {
                row.metric: row.used for row in proven.affine_chain_resources
                if row.role == "target" and row.owner_id == cover.cover_id
            }
            assert counts["identity_rows"] == 2
            assert counts["identity_key_bytes_upper"] > 0
        else:
            assert result is None
    assert calls


def test_affine_chain_ineligible_bijection_preserves_general_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartCover

    square = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=float)
    targets = (square[[0, 1, 2]], square[[0, 2, 3]])
    cover = SourceBoundaryChartCover(
        np.asarray((square[[0, 1, 3]], square[[1, 2, 3]])),
        np.zeros(2), "certified", True, "scoped-chain", "r1",
    )
    original = owner.embedding_state_owner.candidate_pairs
    cross_calls = []

    def candidates(
        lower: np.ndarray, upper: np.ndarray, capacity: int, /,
        *, other: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if other is not None:
            cross_calls.append(capacity)
        return original(lower, upper, capacity, other=other)

    monkeypatch.setattr(owner.embedding_state_owner, "candidate_pairs", candidates)
    areas = {}
    result, proven, _ = owner.prove_affine_chart_chain(
        [(row, 0.0, 0.0) for row in targets], cover, MeshCertificateLimits(), 0.0,
        source_identity=("scoped-chain", "r1"), earned_areas=areas,
    )
    assert result == 0.0 and len(cross_calls) == 1
    assert sum(area for area in areas.values() if area > 0) == Fraction(1)
    assert {
        row.metric: row.used for row in proven.affine_chain_resources
        if row.role == "target" and row.owner_id == cover.cover_id
    }["identity_rows"] == 0


@pytest.mark.parametrize("fault", ("overlap", "duplicate", "seam", "scope", "revision"))
def test_affine_chain_early_bijection_does_not_bypass_partition_or_scope(fault: str) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartCover

    row = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    targets = (row, row + np.asarray((0.25, 0, 0)))
    if fault == "duplicate":
        targets = (row, row.copy())
    elif fault != "overlap":
        targets = (row, row + np.asarray((2, 0, 0)))
    sources = np.asarray(targets).copy()
    if fault == "seam":
        sources[1, 0, 0] += 2.0**-40
    cover = SourceBoundaryChartCover(sources, np.zeros(2), "certified", True, "scope-a", "r1")
    proxies = [(triangle, 0.0, 0.0) for triangle in targets]
    if fault in ("scope", "revision"):
        identity = ("scope-b", "r1") if fault == "scope" else ("scope-a", "r2")
        with pytest.raises(ValueError, match="authenticated scoped source owner"):
            owner.prove_affine_chart_chain(
                proxies, cover, MeshCertificateLimits(), 0.0, source_identity=identity,
            )
    else:
        result, _, _ = owner.prove_affine_chart_chain(
            proxies, cover, MeshCertificateLimits(), 0.0, source_identity=("scope-a", "r1"),
        )
        assert result is None


@pytest.mark.parametrize("resource", ("work", "memory"))
def test_affine_chain_early_bijection_public_resource_boundary(
    resource: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax import _meshcore
    from phydrax.geometry import _affine_chart_coverage as owner
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartCover

    row = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    targets = (row, row + np.asarray((2, 0, 0)))
    cover = SourceBoundaryChartCover(
        np.asarray(targets[::-1]), np.zeros(2), "certified", True, "scoped-chain", "r1",
    )
    original = owner._prepare_chain_records
    budgets: list[CoordinateEnclosureBudget] = []

    def prepare(
        target_rows: tuple[np.ndarray, ...], source_rows: tuple[np.ndarray, ...],
        preparation: owner._Preparation,
    ) -> owner._PreparedChains:
        budgets.append(preparation.budget)
        return original(target_rows, source_rows, preparation)

    monkeypatch.setattr(owner, "_prepare_chain_records", prepare)

    def exercise(limits: MeshCertificateLimits) -> tuple[int, int]:
        result, proven, _ = owner.prove_affine_chart_chain(
            [(triangle, 0.0, 0.0) for triangle in targets], cover, limits, 0.0,
            source_identity=("scoped-chain", "r1"),
        )
        assert result == 0.0
        counts = {
            row.metric: row.used for row in proven.affine_chain_resources
            if row.role == "target" and row.owner_id == cover.cover_id
        }
        return (
            counts["total_work"]
            + budgets[-1].required_work_units - budgets[-1].work_units,
            counts["coefficient_bytes_upper"],
        )

    work, memory = exercise(MeshCertificateLimits())
    assert exercise(MeshCertificateLimits(
        maximum_work_units=work, maximum_scratch_bytes=memory,
    )) == (work, memory)
    with pytest.raises((CoordinateEnclosureResourceError, _meshcore.MeshcoreError)):
        exercise(MeshCertificateLimits(
            maximum_work_units=work - (resource == "work"),
            maximum_scratch_bytes=memory - (resource == "memory"),
        ))


def test_affine_chain_shared_measure_matches_separate_exact_sums(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    from phydrax.discretization._coordinate_budget import reserve_polynomial
    from phydrax.geometry import _affine_chart_coverage as owner

    row = np.asarray(((0, 0, 0), (2.0**-54, 0, 0), (0, 1, 0)))
    targets = tuple(row + np.asarray((0, 2 * index, 0)) for index in range(778))
    sources = tuple(triangle[::-1].copy() for triangle in reversed(targets))
    captured: list[tuple[Fraction, ...]] = []
    original = owner._chain_areas_equal

    def compare(
        rows: tuple[np.ndarray, ...], areas: list[Fraction],
        records: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
        preparation: owner._Preparation,
        *, matched_areas: Mapping[int, Fraction] | None = None,
    ) -> bool:
        captured.append(tuple(areas))
        if matched_areas is not None:
            assert all(area is matched_areas[id(row)] for row, area in zip(rows, areas, strict=True))
        return original(rows, areas, records, preparation, matched_areas=matched_areas)

    monkeypatch.setattr(owner, "_chain_areas_equal", compare)
    charges = []
    for shared in (False, True):
        with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            prepared = owner._prepare_chain_records(targets, sources, preparation)
            assert all(target is not source for target, source in zip(prepared.target, reversed(prepared.source), strict=True))
            before = budget.work_units
            if shared:
                assert owner._chain_identity_tiling(targets, sources, prepared, preparation)
                assert len(preparation.projections) == len(targets)
            else:
                # Independent former algorithm: each operand computes its own
                # measure and each row compares its own exact accumulated sum.
                with budget.temporary_scope():
                    mapping = owner._chain_identity_mapping(targets, sources, prepared, preparation)
                    assert mapping is not None
                    budget.reserve(2 * len(mapping), 512 + 32 * len(mapping))
                    target_area = [Fraction(0)] * len(mapping)
                    source_area = [Fraction(0)] * len(mapping)
                    for first, second in enumerate(mapping):
                        for records, areas, index in (
                            (prepared.target, target_area, first),
                            (prepared.source, source_area, second),
                        ):
                            points, plane = records[index]
                            assert plane is not None
                            projected = preparation.projected(points, plane[1])
                            measure = preparation.signed_measure(projected)
                            bits = preparation.bits((measure,))
                            reserve_polynomial(1, 1, 1, bits, budget=budget)
                            areas[index] = abs(measure)
                        budget.reserve(2)
                        assert target_area[first] > 0 and target_area[first] == source_area[second]
                    assert owner._chain_areas_equal(targets, target_area, prepared.records, preparation)
                    assert owner._chain_areas_equal(sources, source_area, prepared.records, preparation)
                assert len(preparation.projections) == len(targets) + len(sources)
            assert preparation.earned_areas == {}
            charges.append((budget.work_units - before, budget.peak_bytes_upper))
    assert captured[0] == captured[1] == captured[2] == captured[3]
    assert all(area == Fraction(1, 2**55) for area in captured[0])
    assert charges[1][0] < charges[0][0]
    # The unchanged complete-key preparation can dominate the lifetime peak.
    assert charges[1][1] <= charges[0][1]
    record_property("separate_exact_measure_work", charges[0][0])
    record_property("shared_exact_measure_work", charges[1][0])
    record_property("shared_exact_measure_saved_work", charges[0][0] - charges[1][0])
    record_property("separate_exact_measure_peak_bytes", charges[0][1])
    record_property("shared_exact_measure_peak_bytes", charges[1][1])


@pytest.mark.parametrize("role", ("target", "source"))
def test_affine_chain_shared_measure_retains_both_accumulator_checks(
    role: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _affine_chart_coverage as owner

    row = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
    targets, sources = (row,), (row[::-1].copy(),)
    original = owner._chain_areas_equal
    checked_roles = []

    def corrupt(
        rows: tuple[np.ndarray, ...], areas: list[Fraction],
        records: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
        preparation: owner._Preparation,
        *, matched_areas: Mapping[int, Fraction] | None = None,
    ) -> bool:
        checked_role = "target" if rows is targets else "source"
        checked_roles.append(checked_role)
        assert matched_areas is not None
        if checked_role == role:
            areas[0] += 1
        return original(rows, areas, records, preparation, matched_areas=matched_areas)

    monkeypatch.setattr(owner, "_chain_areas_equal", corrupt)
    with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(targets, sources, preparation)
        assert owner._chain_identity_tiling(targets, sources, prepared, preparation) is False
    assert checked_roles == (["target"] if role == "target" else ["target", "source"])
