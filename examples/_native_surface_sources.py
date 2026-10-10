#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Parametric meshing domains with analytic measures for surface-meshing tests.

Each builder returns a `MeshingDomain` over exact analytic patches together
with the independent reference quantities of the represented surface (area,
distance to the surface), so tests never check a mesh against the generator's
own bookkeeping.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from phydrax.geometry import (
    CircleCurve,
    CylinderPatch,
    LineCurve,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSurfacePatch,
    PatchCurveUse,
    PatchPoleUse,
    PlanePatch,
    SpherePatch,
    TorusPatch,
)
from phydrax.geometry.brep import NativePeriodEndpoint


_TAU = 2.0 * np.pi
_HALF_PI = 0.5 * np.pi


@dataclass(frozen=True, slots=True)
class AnalyticSurface:
    """A meshing domain with its exact area and point-to-surface distance."""

    domain: MeshingDomain
    area: float
    distance: Callable[[np.ndarray], np.ndarray]
    closed: bool
    euler_characteristic: int | None


def _side(
    curve: int,
    origin: tuple[float, float],
    direction: tuple[float, float],
    first: float,
    last: float,
) -> PatchCurveUse:
    return PatchCurveUse(curve, LineCurve(origin, direction), first, last)


def sphere(radius: float = 1.0, *, revision: str = "r1") -> AnalyticSurface:
    """Latitude/longitude sphere: one seam curve and two collapsed pole sides."""

    patch = SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), radius)
    loop = (
        PatchPoleUse(0, (0.0, -_HALF_PI), (_TAU, -_HALF_PI)),
        _side(0, (_TAU, 0.0), (0.0, 1.0), -_HALF_PI, _HALF_PI),
        PatchPoleUse(1, (_TAU, _HALF_PI), (0.0, _HALF_PI)),
        _side(0, (0.0, 0.0), (0.0, 1.0), _HALF_PI, -_HALF_PI),
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        (MeshingDomainCurve(0, 1),),
        2,
        source_id="sphere",
        source_revision=revision,
        regions=(MeshingDomainRegion("ball", ((0, 1),)),),
    )
    return AnalyticSurface(
        domain,
        4.0 * np.pi * radius**2,
        lambda points: np.abs(np.linalg.norm(points, axis=1) - radius),
        True,
        2,
    )


def torus(major: float = 2.0, minor: float = 0.6) -> AnalyticSurface:
    """Doubly periodic torus: two closed seam curves meeting at one corner."""

    patch = TorusPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), major, minor)
    loop = (
        _side(0, (0.0, 0.0), (1.0, 0.0), 0.0, _TAU),
        _side(1, (_TAU, 0.0), (0.0, 1.0), 0.0, _TAU),
        _side(0, (0.0, _TAU), (1.0, 0.0), _TAU, 0.0),
        _side(1, (0.0, 0.0), (0.0, 1.0), _TAU, 0.0),
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        (MeshingDomainCurve(0, 0), MeshingDomainCurve(0, 0)),
        1,
        source_id="torus",
        source_revision="r1",
        regions=(MeshingDomainRegion("ring", ((0, 1),)),),
    )

    def distance(points: np.ndarray, /) -> np.ndarray:
        ring = np.hypot(points[:, 0], points[:, 1]) - major
        return np.abs(np.hypot(ring, points[:, 2]) - minor)

    return AnalyticSurface(domain, 4.0 * np.pi**2 * major * minor, distance, True, 0)


def capped_cylinder(radius: float = 0.8, height: float = 1.5) -> AnalyticSurface:
    """Cylinder side and disk caps whose circular coedges retain exact periods."""

    side = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), radius)
    side_loop = (
        _side(0, (0.0, 0.0), (1.0, 0.0), 0.0, _TAU),
        _side(2, (_TAU, 0.0), (0.0, 1.0), 0.0, height),
        _side(1, (0.0, height), (1.0, 0.0), _TAU, 0.0),
        _side(2, (0.0, 0.0), (0.0, 1.0), height, 0.0),
    )
    rim = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), radius)
    bottom = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        (
            (
                PatchCurveUse(
                    0,
                    rim,
                    0.0,
                    _TAU,
                    first_root=NativePeriodEndpoint(rim, turns=0),
                    last_root=NativePeriodEndpoint(rim, turns=1),
                ),
            ),
        ),
        reversed=True,
    )
    top = MeshingSurfacePatch(
        PlanePatch((0, 0, height), (1, 0, 0), (0, 1, 0)),
        (
            (
                PatchCurveUse(
                    1,
                    rim,
                    0.0,
                    _TAU,
                    first_root=NativePeriodEndpoint(rim, turns=0),
                    last_root=NativePeriodEndpoint(rim, turns=1),
                ),
            ),
        ),
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(side, (side_loop,)), bottom, top),
        (MeshingDomainCurve(0, 0), MeshingDomainCurve(1, 1), MeshingDomainCurve(0, 1)),
        2,
        source_id="can",
        source_revision="r1",
        regions=(MeshingDomainRegion("interior", ((0, 1), (1, 1), (2, 1))),),
    )

    def distance(points: np.ndarray, /) -> np.ndarray:
        radial = np.hypot(points[:, 0], points[:, 1])
        wall = np.where(
            (points[:, 2] >= 0.0) & (points[:, 2] <= height),
            np.abs(radial - radius),
            np.inf,
        )
        caps = np.where(
            radial <= radius,
            np.minimum(np.abs(points[:, 2]), np.abs(points[:, 2] - height)),
            np.inf,
        )
        return np.minimum(wall, caps)

    area = _TAU * radius * height + 2.0 * np.pi * radius**2
    return AnalyticSurface(domain, area, distance, True, 2)


def annulus(outer: float = 1.0, inner: float = 0.4) -> AnalyticSurface:
    """Open planar sheet with exact source-period outer and hole circle coedges."""

    outer_curve = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), outer)
    inner_curve = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), inner)
    patch = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        (
            (
                PatchCurveUse(
                    0,
                    outer_curve,
                    0.0,
                    _TAU,
                    first_root=NativePeriodEndpoint(outer_curve, turns=0),
                    last_root=NativePeriodEndpoint(outer_curve, turns=1),
                ),
            ),
            (
                PatchCurveUse(
                    1,
                    inner_curve,
                    _TAU,
                    0.0,
                    first_root=NativePeriodEndpoint(inner_curve, turns=1),
                    last_root=NativePeriodEndpoint(inner_curve, turns=0),
                ),
            ),
        ),
    )
    domain = MeshingDomain(
        (patch,),
        (MeshingDomainCurve(0, 0), MeshingDomainCurve(1, 1)),
        2,
        source_id="annulus",
        source_revision="r1",
    )
    return AnalyticSurface(
        domain,
        np.pi * (outer**2 - inner**2),
        lambda points: np.abs(points[:, 2]),
        False,
        0,
    )


def cylinder_sheet(radius: float = 1.0, height: float = 1.0) -> AnalyticSurface:
    """Open rectangular quarter-cylinder patch bounded by four curves."""

    patch = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), radius)
    angle = _HALF_PI
    loop = (
        _side(0, (0.0, 0.0), (1.0, 0.0), 0.0, angle),
        _side(1, (angle, 0.0), (0.0, 1.0), 0.0, height),
        _side(2, (0.0, height), (1.0, 0.0), angle, 0.0),
        _side(3, (0.0, 0.0), (0.0, 1.0), height, 0.0),
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(1, 2),
            MeshingDomainCurve(2, 3),
            MeshingDomainCurve(3, 0),
        ),
        4,
        source_id="sheet",
        source_revision="r1",
    )
    return AnalyticSurface(
        domain,
        radius * angle * height,
        lambda points: np.abs(np.hypot(points[:, 0], points[:, 1]) - radius),
        False,
        1,
    )


def folded_plates() -> AnalyticSurface:
    """Two unit squares meeting at a right angle along one shared curve."""

    first = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        (
            (
                _side(1, (0.0, 0.0), (1.0, 0.0), -1.0, 0.0),
                _side(0, (0.0, 0.0), (0.0, 1.0), 0.0, 1.0),
                _side(2, (0.0, 1.0), (1.0, 0.0), 0.0, -1.0),
                _side(3, (-1.0, 0.0), (0.0, 1.0), 1.0, 0.0),
            ),
        ),
    )
    second = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (0, 0, 1), (0, 1, 0)),
        (
            (
                _side(4, (0.0, 0.0), (1.0, 0.0), 0.0, 1.0),
                _side(5, (1.0, 0.0), (0.0, 1.0), 0.0, 1.0),
                _side(6, (0.0, 1.0), (1.0, 0.0), 1.0, 0.0),
                _side(0, (0.0, 0.0), (0.0, 1.0), 1.0, 0.0),
            ),
        ),
    )
    domain = MeshingDomain(
        (first, second),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(2, 0),
            MeshingDomainCurve(1, 3),
            MeshingDomainCurve(3, 2),
            MeshingDomainCurve(0, 4),
            MeshingDomainCurve(4, 5),
            MeshingDomainCurve(5, 1),
        ),
        6,
        source_id="plates",
        source_revision="r1",
    )

    def distance(points: np.ndarray, /) -> np.ndarray:
        on_first = np.where(points[:, 0] <= 0.0, np.abs(points[:, 2]), np.inf)
        on_second = np.where(points[:, 2] >= 0.0, np.abs(points[:, 0]), np.inf)
        return np.minimum(on_first, on_second)

    return AnalyticSurface(domain, 2.0, distance, False, 1)


__all__ = [
    "AnalyticSurface",
    "annulus",
    "capped_cylinder",
    "cylinder_sheet",
    "folded_plates",
    "sphere",
    "torus",
]
