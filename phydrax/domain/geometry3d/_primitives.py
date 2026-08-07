#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence

import numpy as np
from jaxtyping import ArrayLike

from ...geometry import (
    Box as BoxSource,
    Cone as ConeSource,
    Cylinder as CylinderSource,
    Ellipsoid as EllipsoidSource,
    Sphere as SphereSource,
    Torus as TorusSource,
    Wedge as WedgeSource,
)
from .._geometry import GeometryDomain


def Sphere(
    center: tuple[float, float, float],
    radius: float,
) -> GeometryDomain:
    """Construct an analytic solid sphere."""
    return GeometryDomain(
        SphereSource(center, radius, feature_id="domain-sphere").compile()
    )


def Ellipsoid(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> GeometryDomain:
    """Construct an analytic axis-aligned solid ellipsoid."""
    return GeometryDomain(
        EllipsoidSource(center, radii, feature_id="domain-ellipsoid").compile()
    )


def Cuboid(
    center: tuple[float, float, float],
    dimensions: tuple[float, float, float],
) -> GeometryDomain:
    """Construct an analytic axis-aligned box."""
    return GeometryDomain(
        BoxSource(center, dimensions, feature_id="domain-box").compile()
    )


def Cube(
    center: tuple[float, float, float],
    side: float,
) -> GeometryDomain:
    """Construct an analytic axis-aligned cube."""
    return GeometryDomain(
        BoxSource(center, (side, side, side), feature_id="domain-cube").compile()
    )


def Cylinder(
    face_center: Sequence | ArrayLike,
    axis: Sequence | ArrayLike,
    radius: float,
    angle: float = 2.0 * np.pi,
) -> GeometryDomain:
    """Construct an analytic oriented cylinder or cylindrical sector."""
    return GeometryDomain(
        CylinderSource(
            face_center,
            axis,
            radius,
            angle,
            feature_id="domain-cylinder",
        ).compile()
    )


def Cone(
    base_center: Sequence | ArrayLike,
    axis: Sequence | ArrayLike,
    radius0: float,
    radius1: float = 0.0,
    angle: float = 2.0 * np.pi,
) -> GeometryDomain:
    """Construct an analytic cone or conical frustum."""
    return GeometryDomain(
        ConeSource(
            base_center,
            axis,
            radius0,
            radius1,
            angle,
            feature_id="domain-cone",
        ).compile()
    )


def Torus(
    center: Sequence | ArrayLike,
    inner_radius: float,
    outer_radius: float,
    angle: float = 2.0 * np.pi,
) -> GeometryDomain:
    """Construct an analytic torus or toroidal sector."""
    return GeometryDomain(
        TorusSource(
            center,
            inner_radius,
            outer_radius,
            angle,
            feature_id="domain-torus",
        ).compile()
    )


def Wedge(
    x0: Sequence | ArrayLike,
    extends: Sequence | ArrayLike,
    top_extent: float,
) -> GeometryDomain:
    """Construct an analytic right wedge."""
    return GeometryDomain(
        WedgeSource(
            x0,
            extends,
            top_extent,
            feature_id="domain-wedge",
        ).compile()
    )


__all__ = [
    "Cone",
    "Cube",
    "Cuboid",
    "Cylinder",
    "Ellipsoid",
    "Sphere",
    "Torus",
    "Wedge",
]
