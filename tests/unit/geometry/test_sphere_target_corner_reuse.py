# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from fractions import Fraction

import pytest

import phydrax.discretization._sphere_chart_deformation as owner
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
)
from phydrax.geometry._sphere_material_atlas import (
    SphereProjectiveActionBank,
    SphereProjectiveReferenceMap,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.sphere_material import accepted_sphere_material


type _Point = tuple[Fraction, Fraction]
type _CornerKey = tuple[int, bytes]
type _NativePolygon = tuple[
    tuple[_Point, ...],
    tuple[int, ...],
    tuple[frozenset[frozenset[bytes]], ...],
    tuple[int | None, ...],
    int,
]
type _IncidentCorners = tuple[
    SphereProjectiveReferenceMap,
    tuple[frozenset[bytes], ...],
    tuple[bytes, ...],
    _CornerKey,
    _Point,
    _NativePolygon,
]


@pytest.fixture(scope="module")
def incident_target_corners() -> _IncidentCorners:
    """Find independently clipped cones sharing a ray but not its support pair."""
    _, _, source = accepted_sphere_material(0.4)
    _, _, target = accepted_sphere_material(0.32)
    seen = {}
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(
            source, target, budget=budget, storage=storage
        )
        source_rows, target_rows, _ = owner._candidates(
            source, target, 10**6, 10**9, budget=budget
        )
        for source_row, target_row in zip(source_rows, target_rows, strict=True):
            source_row, target_row = int(source_row), int(target_row)
            mapping = source.projective_reference_map(
                int(source.cell_global_ids[source_row]), target,
                int(target.cell_global_ids[target_row]), budget=budget, actions=actions,
            )
            edges = owner._support_edges(actions, source_row, target_row)
            rays = tuple(actions.corner_key(True, target_row, slot) for slot in range(3))
            result = owner._native_polygon(
                mapping, 10**8, 10**7, edges, {}, storage,
                source_row=source_row, target_corner_keys=rays,
                target_corner_points={}, budget=budget,
            )
            polygon, _, supports, corners, _ = result
            for point, support, corner in zip(polygon, supports, corners, strict=True):
                if corner is None or point in owner._SOURCE_CORNERS.values():
                    continue
                key = (source_row, rays[corner])
                if key in seen and seen[key][0] != support:
                    previous_support, previous_point = seen[key]
                    assert previous_support != support
                    assert previous_point == point
                    return mapping, edges, rays, key, previous_point, result
                seen[key] = support, point
    raise AssertionError("fixture must contain incident target cones with distinct supports")


def _polygon(
    fixture: _IncidentCorners,
    budget: CoordinateEnclosureBudget,
    storage: CoordinateLiveStorage,
    cache: dict[_CornerKey, _Point],
    *,
    source_row: int | None = None,
    rays: tuple[bytes, ...] | None = None,
) -> _NativePolygon:
    mapping, edges, original_rays, key, _, _ = fixture
    return owner._native_polygon(
        mapping, 10**8, 10**7, edges, {}, storage,
        source_row=key[0] if source_row is None else source_row,
        target_corner_keys=original_rays if rays is None else rays,
        target_corner_points=cache, budget=budget,
    )


def test_incident_target_cones_reuse_actual_point(
    incident_target_corners: _IncidentCorners,
) -> None:
    fixture = incident_target_corners
    _, _, _, key, previous_point, expected = fixture
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        result = _polygon(fixture, budget, storage, {key: previous_point})
        reused_work = budget.work_units
    assert result[:4] == expected[:4]
    assert any(point is previous_point for point in result[0])
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        independent = _polygon(fixture, budget, storage, {})
        assert budget.work_units > reused_work
    assert independent[:4] == result[:4]
    assert independent[4] > result[4]


@pytest.mark.parametrize("separation", ["source_row", "binary_ray"])
def test_corner_reuse_does_not_merge_distinct_keys(
    incident_target_corners: _IncidentCorners, separation: str
) -> None:
    fixture = incident_target_corners
    _, _, rays, key, _, expected = fixture
    poison = (Fraction(-1), Fraction(-1))
    cache_key = (key[0] + 1, key[1]) if separation == "source_row" else (key[0], key[1] + b"noncanonical")
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        result = _polygon(fixture, budget, storage, {cache_key: poison})
    assert result[:4] == expected[:4]


def test_wrong_cached_point_still_requires_native_polygon_support(
    incident_target_corners: _IncidentCorners,
) -> None:
    fixture = incident_target_corners
    key = fixture[3]
    # Feasible in the source triangle, but not the named target corner.
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        with pytest.raises(ValueError, match="Native"):
            _polygon(fixture, budget, storage, {key: (Fraction(1, 7), Fraction(1, 11))})


def test_corner_reuse_observed_work_boundary(
    incident_target_corners: _IncidentCorners,
) -> None:
    fixture = incident_target_corners
    key, point = fixture[3:5]
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        expected = _polygon(fixture, budget, storage, {key: point})
        required = budget.required_work_units
    assert required > 0
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        with budget.bound_stage(required, 10**8):
            assert _polygon(fixture, budget, storage, {key: point}) == expected
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        with pytest.raises(CoordinateEnclosureResourceError):
            with budget.bound_stage(required - 1, 10**8):
                _polygon(fixture, budget, storage, {key: point})
