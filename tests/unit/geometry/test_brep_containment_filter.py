#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rational-reference containment decisions and deterministic filtering work counts."""

from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest

import phydrax.geometry.brep._query_membership as membership
from phydrax._bvh import BVHBuildPolicy, prepare_bvh
from phydrax.geometry.brep._affine_query import QueryBroadphase
from phydrax.geometry.brep._constructors import BRepTessellationPolicy
from phydrax.geometry.brep._exact import square_root_interval
from phydrax.geometry.brep._query import (
    BRepQueryPolicy,
    prepare_brep_query,
)
from phydrax.geometry.brep._query_types import BRepQueryBudget
from phydrax.meshing.qualification import corpus_route_points, corpus_route_query
from tests._support.cad_models import native_solid


class _FractionCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, value: float | int = 0) -> Fraction:
        self.count += 1
        return Fraction(value)


def _exact_reference(
    point: np.ndarray,
    boxes: np.ndarray,
    fraction: Callable[[float | int], Fraction] = Fraction,
) -> np.ndarray:
    # This is the pre-filter reference: rational binary endpoint differences,
    # squared sums, and the historical directed square-root classifier.
    bounds = []
    for box in boxes:
        delta = tuple(fraction(float(value)) for value in point)
        first = tuple(fraction(float(value)) for value in box[0])
        last = tuple(fraction(float(value)) for value in box[1])
        gap = tuple(
            max(fraction(0), a - q, q - b)
            for a, b, q in zip(first, last, delta, strict=True)
        )
        far = tuple(
            max(abs(a - q), abs(b - q))
            for a, b, q in zip(first, last, delta, strict=True)
        )
        bounds.append(
            (
                square_root_interval(sum((value * value for value in gap), fraction(0)))[
                    0
                ],
                square_root_interval(sum((value * value for value in far), fraction(0)))[
                    1
                ],
            )
        )
    return np.asarray(bounds, dtype=np.float64)


def _boundary_boxes(scale: float) -> tuple[np.ndarray, float]:
    classifier = scale / 8.0
    next_distance = np.nextafter(classifier, np.inf)
    distances = np.asarray(
        (
            0.0,
            classifier / 2.0,
            np.nextafter(classifier, -np.inf),
            classifier,
            next_distance,
            np.nextafter(next_distance, np.inf),
            classifier * 2.0,
        ),
        dtype=np.float64,
    )
    distances = np.concatenate((distances, scale * np.arange(4, 68, dtype=np.float64)))
    lower = np.column_stack(
        (
            distances,
            np.full(distances.shape, -scale / 8),
            np.full(distances.shape, -scale / 8),
        )
    )
    upper = lower + np.asarray((scale / 4, scale / 4, scale / 4), dtype=np.float64)
    return np.stack((lower, upper), axis=1), classifier


@pytest.mark.parametrize(
    "scale",
    (2.0**-600, 2.0**-500, 2.0**-40, 1.0, 2.0**40, 2.0**500),
    ids=("square-underflow", "tiny", "small", "unit", "large", "huge"),
)
def test_directed_float_box_evidence_encloses_exact_binary_distances(
    scale: float,
) -> None:
    boxes, _ = _boundary_boxes(scale)
    point = np.asarray((scale / 16, -scale / 32, scale / 64), dtype=np.float64)
    bounds = membership._containment_box_bounds(point, boxes)
    reference = _exact_reference(point, boxes)
    assert np.all(bounds[:, 0] <= reference[:, 0])
    assert np.all(bounds[:, 2] >= reference[:, 1])
    for index, box in enumerate(boxes):
        coordinates = tuple(Fraction(float(value)) for value in point)
        first = tuple(Fraction(float(value)) for value in box[0])
        last = tuple(Fraction(float(value)) for value in box[1])
        gap_square = sum(
            max(Fraction(0), a - q, q - b) ** 2
            for a, b, q in zip(first, last, coordinates, strict=True)
        )
        far_square = sum(
            max(abs(a - q), abs(b - q)) ** 2
            for a, b, q in zip(first, last, coordinates, strict=True)
        )
        assert Fraction(float(bounds[index, 0])) ** 2 <= gap_square, index
        assert Fraction(float(bounds[index, 1])) ** 2 >= gap_square, index
        assert Fraction(float(bounds[index, 2])) ** 2 >= far_square, index


@pytest.mark.parametrize(
    "scale",
    (2.0**-600, 2.0**-500, 2.0**-40, 1.0, 2.0**40, 2.0**500),
    ids=("square-underflow", "tiny", "small", "unit", "large", "huge"),
)
def test_near_boundaries_match_exact_reference_with_fewer_fraction_constructions(
    scale: float,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    boxes, classifier = _boundary_boxes(scale)
    point = np.zeros((3,), dtype=np.float64)
    before = _FractionCounter()
    reference = _exact_reference(point, boxes, before)
    bounds = membership._containment_box_bounds(point, boxes)
    uncertain = np.count_nonzero(
        (bounds[:, 0] <= classifier) & (bounds[:, 1] > classifier)
    )
    after = _FractionCounter()
    monkeypatch.setattr(membership, "Fraction", after)
    near = membership._containment_near_mask(point, boxes, bounds, classifier)
    np.testing.assert_array_equal(near, reference[:, 0] <= classifier)
    assert before.count == 14 * len(boxes)
    assert after.count == (3 + 10 * uncertain if uncertain else 0)
    assert after.count < before.count
    record_property("fraction_constructions_before", before.count)
    record_property("fraction_constructions_after", after.count)


@pytest.mark.parametrize(
    ("gap", "classifier", "expected"),
    (
        ((1.0, 2.0**-26, 0.0), 1.0, True),
        ((1.0, 0.0, 0.0), np.nextafter(1.0, -np.inf), False),
        ((np.nextafter(1.0, np.inf), 0.0, 0.0), 1.0, False),
        ((np.nextafter(1.0, np.inf), 2.0**-26, 0.0), 1.0, False),
        ((0.0, 0.0, 0.0), 0.0, True),
    ),
    ids=(
        "rounded-down-root",
        "classifier-predecessor",
        "gap-successor",
        "diagonal-successor",
        "zero",
    ),
)
def test_near_criterion_preserves_adjacent_square_root_rounding(
    gap: tuple[float, float, float], classifier: float, expected: bool
) -> None:
    point = np.zeros((3,), dtype=np.float64)
    corner = np.asarray(gap, dtype=np.float64)
    boxes = np.stack((corner, corner))[None]
    reference = _exact_reference(point, boxes)
    bounds = membership._containment_box_bounds(point, boxes)
    near = membership._containment_near_mask(point, boxes, bounds, classifier)
    assert bool(reference[0, 0] <= classifier) is expected
    assert bool(near[0]) is expected
    if gap == (1.0, 2.0**-26, 0.0):
        # Replacing the historical predicate with geometric distance <= tolerance
        # incorrectly rejects this point, although its directed lower root is 1.
        assert sum(Fraction(value) ** 2 for value in gap) > Fraction(classifier) ** 2


@pytest.mark.parametrize("translated", (False, True), ids=("origin", "large-translation"))
@pytest.mark.parametrize(
    "select_near", (False, True), ids=("other-solid-near", "selected-solid-near")
)
def test_bvh_filter_preserves_selected_face_order_and_exact_near_set(
    translated: bool, select_near: bool
) -> None:
    boxes, classifier = _boundary_boxes(1.0)
    point = np.zeros((3,), dtype=np.float64)
    if translated:
        point = np.asarray((2.0**40, -(2.0**40), 2.0**39), dtype=np.float64)
        boxes = boxes + point
    broadphase = QueryBroadphase.prepare(
        prepare_bvh(
            boxes[:, 0], boxes[:, 1], policy=BVHBuildPolicy(leaf_size=2), dtype=np.float64
        )
    )
    faces = (5, 3, 1, 0, 2, 6, 4, *range(8, len(boxes))) if select_near else (40, 20, 30)
    selected = boxes[list(faces)]
    reference = _exact_reference(point, selected)
    expected = tuple(
        faces[index] for index in np.flatnonzero(reference[:, 0] <= classifier)
    )
    near, lower, upper, far_lower = membership._containment_face_filter(
        point,
        selected,
        faces,
        {face: index for index, face in enumerate(faces)},
        broadphase,
        classifier,
    )
    assert near == expected
    assert lower <= np.min(reference[:, 0])
    assert upper >= np.min(reference[:, 1])
    assert far_lower > classifier
    if not select_near:
        assert lower > classifier


def test_bvh_separation_prunes_all_face_arithmetic_and_fraction_work(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    boxes, classifier = _boundary_boxes(1.0)
    point = np.asarray((-100.0, -100.0, -100.0), dtype=np.float64)
    broadphase = QueryBroadphase.prepare(
        prepare_bvh(
            boxes[:, 0], boxes[:, 1], policy=BVHBuildPolicy(leaf_size=2), dtype=np.float64
        )
    )
    faces = tuple(range(len(boxes)))
    before = _FractionCounter()
    reference = _exact_reference(point, boxes, before)
    after = _FractionCounter()
    evaluated_boxes = 0
    directed_bounds = membership._containment_box_bounds

    def counted_bounds(point_: np.ndarray, boxes_: np.ndarray, /) -> np.ndarray:
        nonlocal evaluated_boxes
        evaluated_boxes += len(boxes_)
        return directed_bounds(point_, boxes_)

    monkeypatch.setattr(membership, "_containment_box_bounds", counted_bounds)
    monkeypatch.setattr(membership, "Fraction", after)
    near, lower, upper, far_lower = membership._containment_face_filter(
        point, boxes, faces, {face: face for face in faces}, broadphase, classifier
    )
    assert near == ()
    assert np.all(reference[:, 0] > classifier)
    assert classifier < lower <= np.min(reference[:, 0])
    assert far_lower == lower
    assert upper >= np.min(reference[:, 1])
    assert evaluated_boxes == 1
    assert after.count == 0 < before.count
    record_property("fraction_constructions_before", before.count)
    record_property("fraction_constructions_after", after.count)
    record_property("box_rows_before", len(boxes))
    record_property("box_rows_after", evaluated_boxes)


class _RecordingQueryBudget(BRepQueryBudget):
    def __init__(self, maximum_operations: int) -> None:
        super().__init__(maximum_operations)
        self.charges: list[tuple[int, int, int]] = []

    def consume(
        self, operations: int, /, *, points: int = 0, subdivisions: int = 0
    ) -> None:
        self.charges.append((operations, points, subdivisions))
        super().consume(operations, points=points, subdivisions=subdivisions)


@pytest.mark.parametrize(
    ("name", "refuse_refinement"),
    (("cylinder", False), ("extruded_plate", False), ("cylinder", True)),
    ids=("cylinder", "plate-with-hole", "refinement-exhaustion"),
)
def test_public_corpus_containment_matches_exact_filter_decisions_and_charge_order(
    name: str,
    refuse_refinement: bool,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    model = native_solid(
        name, catalog="geometry", tessellation=BRepTessellationPolicy(realize=False)
    )
    policy = (
        BRepQueryPolicy(maximum_subdivisions=0, maximum_depth=0)
        if refuse_refinement
        else BRepQueryPolicy()
    )
    query = prepare_brep_query(model, policy=policy)
    points = np.asarray(
        ((0.0, 0.0, 1.0), (1.2, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.5, 2.0))
        if name == "cylinder"
        else ((1.5, 1.5, 0.25), (0.0, 0.0, 0.25), (0.5, 0.5, 0.25), (2.0, 1.0, 0.25)),
        dtype=np.float64,
    )
    before = _FractionCounter()

    def exact_filter(
        point: np.ndarray,
        boxes: np.ndarray,
        faces: tuple[int, ...],
        face_indices: dict[int, int],
        broadphase: QueryBroadphase,
        classifier: float,
        /,
    ) -> tuple[tuple[int, ...], float, float, float]:
        del face_indices, broadphase
        bounds = _exact_reference(point, boxes, before)
        mask = bounds[:, 0] <= classifier
        near = tuple(faces[index] for index in np.flatnonzero(mask))
        return (
            near,
            float(np.min(bounds[:, 0])),
            float(np.min(bounds[:, 1])),
            float(np.min(bounds[~mask, 0], initial=np.inf)),
        )

    old_budget = _RecordingQueryBudget(policy.maximum_operations)
    with monkeypatch.context() as patch:
        patch.setattr(membership, "_containment_face_filter", exact_filter)
        reference = query.contains(points, budget=old_budget)
    after = _FractionCounter()
    filtered = membership._containment_face_filter

    def counted_filter(
        point: np.ndarray,
        boxes: np.ndarray,
        faces: tuple[int, ...],
        face_indices: dict[int, int],
        broadphase: QueryBroadphase,
        classifier: float,
        /,
    ) -> tuple[tuple[int, ...], float, float, float]:
        with monkeypatch.context() as patch:
            patch.setattr(membership, "Fraction", after)
            return filtered(point, boxes, faces, face_indices, broadphase, classifier)

    new_budget = _RecordingQueryBudget(policy.maximum_operations)
    monkeypatch.setattr(membership, "_containment_face_filter", counted_filter)
    result = query.contains(points, budget=new_budget)
    np.testing.assert_array_equal(result.inside, reference.inside)
    np.testing.assert_array_equal(result.status, reference.status)
    np.testing.assert_array_equal(result.unresolved, reference.unresolved)
    np.testing.assert_array_equal(result.resource_exhausted, reference.resource_exhausted)
    np.testing.assert_array_equal(result.query_operations, reference.query_operations)
    np.testing.assert_array_equal(result.winding, reference.winding)
    assert new_budget.charges == old_budget.charges
    assert (new_budget.operations, new_budget.points, new_budget.subdivisions) == (
        old_budget.operations,
        old_budget.points,
        old_budget.subdivisions,
    )
    assert np.all(np.asarray(result.distance_lower_bounds) >= 0.0)
    assert np.all(
        np.asarray(result.distance_lower_bounds)
        <= np.asarray(result.distance_upper_bounds)
    )
    assert before.count > after.count
    if refuse_refinement:
        assert np.all(np.asarray(result.unresolved))
        assert np.all(np.asarray(result.resource_exhausted))
    else:
        np.testing.assert_array_equal(
            result.inside,
            (True, False, True, True)
            if name == "cylinder"
            else (True, False, False, True),
        )
        np.testing.assert_array_equal(
            result.unresolved,
            (False, False, True, True)
            if name == "cylinder"
            else (False, False, False, True),
        )
        assert not np.any(np.asarray(result.resource_exhausted))
    if name == "cylinder":
        # These radial/axial distances use binary64 differences.
        distance = np.asarray((1.0, points[1, 0] - 1.0, 0.0, 0.0), dtype=np.float64)
        assert np.all(np.asarray(result.distance_lower_bounds) <= distance)
        assert np.all(np.asarray(result.distance_upper_bounds) >= distance)
    record_property("fraction_constructions_before", before.count)
    record_property("fraction_constructions_after", after.count)


def _reference_face_filter(
    point: np.ndarray,
    boxes: np.ndarray,
    faces: tuple[int, ...],
    face_indices: dict[int, int],
    broadphase: QueryBroadphase,
    classifier: float,
    /,
    *,
    fraction: Callable[[float | int], Fraction] = Fraction,
) -> tuple[tuple[int, ...], float, float, float]:
    del face_indices, broadphase
    bounds = _exact_reference(point, boxes, fraction)
    mask = bounds[:, 0] <= classifier
    return (
        tuple(faces[index] for index in np.flatnonzero(mask)),
        float(np.min(bounds[:, 0])),
        float(np.min(bounds[:, 1])),
        float(np.min(bounds[~mask, 0], initial=np.inf)),
    )


@pytest.mark.parametrize(
    "route",
    ("sphere", "revolution", "affine", "nurbs"),
    ids=("sphere", "revolution", "affine", "general-nurbs"),
)
def test_public_route_boundary_corpus_matches_rational_reference(
    route: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    from functools import partial

    query = corpus_route_query(route)
    points = corpus_route_points(query, route)
    before = _FractionCounter()
    old_budget = _RecordingQueryBudget(query.policy.maximum_operations)
    with monkeypatch.context() as patch:
        patch.setattr(
            membership,
            "_containment_face_filter",
            partial(_reference_face_filter, fraction=before),
        )
        reference = query.contains(points, budget=old_budget)
    after = _FractionCounter()
    directed_filter = membership._containment_face_filter

    def counted_filter(
        point: np.ndarray,
        boxes: np.ndarray,
        faces: tuple[int, ...],
        face_indices: dict[int, int],
        broadphase: QueryBroadphase,
        classifier: float,
        /,
    ) -> tuple[tuple[int, ...], float, float, float]:
        with monkeypatch.context() as patch:
            patch.setattr(membership, "Fraction", after)
            return directed_filter(
                point, boxes, faces, face_indices, broadphase, classifier
            )

    new_budget = _RecordingQueryBudget(query.policy.maximum_operations)
    monkeypatch.setattr(membership, "_containment_face_filter", counted_filter)
    result = query.contains(points, budget=new_budget)
    np.testing.assert_array_equal(result.inside, reference.inside)
    np.testing.assert_array_equal(result.status, reference.status)
    np.testing.assert_array_equal(result.unresolved, reference.unresolved)
    np.testing.assert_array_equal(result.resource_exhausted, reference.resource_exhausted)
    np.testing.assert_array_equal(result.query_operations, reference.query_operations)
    np.testing.assert_array_equal(result.winding, reference.winding)
    np.testing.assert_array_equal(result.inside[:2], (True, False))
    np.testing.assert_array_equal(result.unresolved[:2], (False, False))
    assert new_budget.charges == old_budget.charges
    assert after.count <= before.count
    if route == "nurbs":
        assert before.count == 14 * len(query.solid_faces[0]) * len(points)
        assert after.count < before.count
    record_property("fraction_constructions_before", before.count)
    record_property("fraction_constructions_after", after.count)
