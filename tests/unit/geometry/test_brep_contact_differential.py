"""Differential checks of analytic contact and certified general discovery."""

import math

import numpy as np
import pytest

from phydrax.geometry.brep import (
    intersect_surface_regions,
    ParametricIntersectionPolicy,
    PlanePatch,
    SpherePatch,
    SurfaceRegion,
)
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._intersection import _execute_surface_intersection
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations


def _sphere(center: tuple[float, float, float]) -> SurfaceRegion:
    # Signed unit axes keep the contact away from poles and the longitude seam.
    patch = SpherePatch(center, (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)
    return SurfaceRegion(patch, ((0.0, -math.pi / 2), (math.tau, math.pi / 2)))


@pytest.fixture(params=("sphere-plane", "sphere-sphere"))
def contact_pair(request: pytest.FixtureRequest) -> tuple[SurfaceRegion, SurfaceRegion]:
    sphere = _sphere((0.0, 0.0, 0.0))
    match request.param:
        case "sphere-plane":
            partner = SurfaceRegion(
                PlanePatch((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ((-2.0, -2.0), (2.0, 2.0)),
            )
        case "sphere-sphere":
            partner = _sphere((2.0, 0.0, 0.0))
        case _:
            raise ValueError(
                f"contact_pair.kind must select a contact family; got {request.param!r}"
            )
    return sphere, partner


def test_analytic_and_general_contact_certificates_agree(
    contact_pair: tuple[SurfaceRegion, SurfaceRegion],
) -> None:
    first, second = contact_pair
    policy = ParametricIntersectionPolicy(maximum_boxes=4096, batch=32)
    analytic = intersect_surface_regions(first, second, policy=policy)
    with coordinate_source_request((first, second)) as budget:
        general = _execute_surface_intersection(
            first,
            second,
            policy,
            preparations=TrimRequestPreparations(),
            coordinate_budget=budget,
        )
    assert analytic.complete
    assert len(analytic.curves) == len(general.curves) == 0
    assert (
        [point.kind for point in analytic.points]
        == [point.kind for point in general.points]
        == ["tangent_point"]
    )
    shortcut, discovered = analytic.points[0], general.points[0]
    assert np.all(shortcut.parameter_lower <= discovered.parameter_upper)
    assert np.all(discovered.parameter_lower <= shortcut.parameter_upper)
    # The independent reference is the represented unit-sphere support at x=1.
    # Gap bounds plus outward binary64 rounding enclose both realizations.
    reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    for point in (shortcut, discovered):
        lower = np.nextafter(point.point - point.gap_bound, -np.inf)
        upper = np.nextafter(point.point + point.gap_bound, np.inf)
        assert np.all(lower <= reference) and np.all(reference <= upper)


@pytest.mark.parametrize(
    "family,separation",
    [
        ("sphere-plane", 0.5),
        ("sphere-plane", 3.0),
        ("sphere-sphere", 1.0),
        ("sphere-sphere", 3.0),
        ("sphere-sphere", 0.0),
    ],
    ids=[
        "plane-secant",
        "plane-disjoint",
        "sphere-secant",
        "sphere-disjoint",
        "concentric",
    ],
)
def test_public_and_general_surface_routes_agree(family: str, separation: float) -> None:
    first = _sphere((0.0, 0.0, 0.0))
    if family == "sphere-plane":
        second = SurfaceRegion(
            PlanePatch((separation, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ((-2.0, -2.0), (2.0, 2.0)),
        )
    else:
        second = _sphere((separation, 0.0, 0.0))
    policy = ParametricIntersectionPolicy(maximum_boxes=4096, batch=32)
    public = intersect_surface_regions(first, second, policy=policy)
    with coordinate_source_request((first, second)) as budget:
        general = _execute_surface_intersection(
            first,
            second,
            policy,
            preparations=TrimRequestPreparations(),
            coordinate_budget=budget,
        )
    # These non-tangent inputs exercise the shortcuts' refusal boundary:
    # the general owner retains secants, disjointness and concentric coincidence.
    assert public.complete == general.complete
    assert len(public.curves) == len(general.curves)
    assert len(public.coincident) == len(general.coincident)
    assert [point.kind for point in public.points] == [
        point.kind for point in general.points
    ]
    for shortcut, discovered in zip(public.curves, general.curves, strict=True):
        assert shortcut.start_kind == discovered.start_kind
        assert shortcut.end_kind == discovered.end_kind
        left = np.asarray(shortcut.bounding_box(0.0, shortcut.num_charts))
        right = np.asarray(discovered.bounding_box(0.0, discovered.num_charts))
        assert np.all(left[0] <= right[1])
        assert np.all(right[0] <= left[1])
