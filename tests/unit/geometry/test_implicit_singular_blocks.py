#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.implicit import _source_charts
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.units import METER
from tests.unit.geometry.test_implicit_source_charts import _adaptive_source


def _exact_intersection(profile: AnalyticImplicitProfile, box: np.ndarray) -> bool:
    lower = tuple(Fraction.from_float(float(value)) for value in box[0])
    upper = tuple(Fraction.from_float(float(value)) for value in box[1])
    center = tuple(Fraction.from_float(float(value)) for value in profile.center)
    contains = tuple(lower[axis] <= center[axis] <= upper[axis] for axis in range(3))
    if profile.family == "sphere":
        return all(contains)
    if contains[0] and contains[1]:
        return True
    if not contains[2]:
        return False
    nearest = tuple(
        Fraction(0)
        if contains[axis]
        else min(abs(lower[axis] - center[axis]), abs(upper[axis] - center[axis]))
        for axis in range(2)
    )
    farthest = tuple(
        max(abs(lower[axis] - center[axis]), abs(upper[axis] - center[axis]))
        for axis in range(2)
    )
    radius = Fraction.from_float(profile.major_radius)
    return (
        sum(value * value for value in nearest)
        <= radius * radius
        <= sum(value * value for value in farthest)
    )


@pytest.mark.parametrize("family", ["sphere", "torus"])
def test_block_scan_matches_exact_rational_reference(family: str) -> None:
    center = (0.125, -0.25, 0.5)
    geometry = (
        phx.geometry.Sphere(center, 0.5)
        if family == "sphere"
        else phx.geometry.Torus(center, 0.3, 0.9)
    ).compile()
    profile = AnalyticImplicitProfile(geometry, SpatialCoordinateContract(METER))
    rng = np.random.default_rng(3891)
    endpoints = rng.uniform(-2.0, 2.0, size=(256, 2, 3))
    boxes = np.stack((endpoints.min(axis=1), endpoints.max(axis=1)), axis=1)
    radius = profile.radius if family == "sphere" else profile.major_radius
    boundary = center[0] + radius
    tiny = np.nextafter(0.0, 1.0)
    special = np.asarray(
        [
            [center, center],
            [(center[0], -1.0, -1.0), (1.0, 1.0, 1.0)],
            [(boundary, center[1], center[2]), (boundary, center[1], center[2])],
            [
                (np.nextafter(boundary, -np.inf), center[1], center[2]),
                (np.nextafter(boundary, np.inf), center[1], center[2]),
            ],
            [(1.0e308, tiny, center[2]), (1.0e308, tiny, center[2])],
            [(-1.0e308, -tiny, center[2]), (-1.0e308, -tiny, center[2])],
            [(tiny, tiny, tiny), (tiny, tiny, tiny)],
        ],
        dtype=np.float64,
    )
    boxes = np.concatenate((boxes, special))
    expected = [_exact_intersection(profile, box) for box in boxes]
    with np.errstate(over="ignore", invalid="ignore"):
        for box, result in zip(boxes, expected, strict=True):
            assert profile._field_singular_set_intersects_boxes(box[None]) is result
        for start in range(0, len(boxes), 17):
            assert profile._field_singular_set_intersects_boxes(
                boxes[start : start + 17]
            ) is any(expected[start : start + 17])


def test_touching_torus_circle_requires_exact_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry.implicit import _analytic_profile

    k = 1.0 + 2.0**-50
    profile = AnalyticImplicitProfile(
        phx.geometry.Torus((0.0, 0.0, 0.0), 4.0 * k, 6.0 * k).compile(),
        SpatialCoordinateContract(METER),
    )
    point = np.asarray((3.0 * k, 4.0 * k, 0.0))
    box = np.stack((point, point))[None]
    assert _exact_intersection(profile, box[0])
    original = _analytic_profile.interval_multiply

    def widened(
        first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = original(first, second)
        return np.full_like(lo, -np.inf), np.full_like(hi, np.inf)

    monkeypatch.setattr(_analytic_profile, "interval_multiply", widened)
    assert profile._field_singular_set_intersects_boxes(box)


def test_singularity_contact_refuses_cover_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    surface, source, contract, _, _, _ = _adaptive_source(
        smooth=False, authoring_profile=True
    )
    assert _source_charts._profile_covers_singularities_away_from_zero_set(
        source, surface, contract.spatial_id
    )
    touching = np.zeros((1, 2, 3), dtype=np.float64)
    monkeypatch.setattr(
        _source_charts, "_possible_zero_box_blocks", lambda _: iter((touching,))
    )
    assert not _source_charts._profile_covers_singularities_away_from_zero_set(
        source, surface, contract.spatial_id
    )


@pytest.mark.parametrize(
    "boxes",
    [
        np.empty((2, 3)),
        np.zeros((1, 2, 2)),
        np.full((1, 2, 3), np.inf),
        np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]),
    ],
)
def test_block_scan_refuses_invalid_boxes(boxes: np.ndarray) -> None:
    profile = AnalyticImplicitProfile(
        phx.geometry.Sphere((0.0, 0.0, 0.0), 0.5).compile(),
        SpatialCoordinateContract(METER),
    )
    with pytest.raises(ValueError, match="finite ordered 3D box"):
        profile._field_singular_set_intersects_boxes(boxes)
