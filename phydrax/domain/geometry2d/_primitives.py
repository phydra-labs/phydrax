#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence

from ...geometry import (
    Circle as CircleSource,
    Ellipse as EllipseSource,
    Polygon as PolygonSource,
    Rectangle as RectangleSource,
)
from .._geometry import GeometryDomain


def Circle(center: tuple[float, float], radius: float) -> GeometryDomain:
    """Construct an analytic filled circle."""
    return GeometryDomain(
        CircleSource(center, radius, feature_id="domain-circle").compile()
    )


def Ellipse(
    center: tuple[float, float],
    x_radius: float,
    y_radius: float,
) -> GeometryDomain:
    """Construct an analytic axis-aligned filled ellipse."""
    return GeometryDomain(
        EllipseSource(
            center,
            (x_radius, y_radius),
            feature_id="domain-ellipse",
        ).compile()
    )


def Rectangle(
    center: tuple[float, float],
    width: float,
    height: float,
) -> GeometryDomain:
    """Construct an analytic axis-aligned filled rectangle."""
    return GeometryDomain(
        RectangleSource(
            center,
            (width, height),
            feature_id="domain-rectangle",
        ).compile()
    )


def Square(center: tuple[float, float], side: float) -> GeometryDomain:
    """Construct an analytic axis-aligned square."""
    return GeometryDomain(
        RectangleSource(
            center,
            (side, side),
            feature_id="domain-square",
        ).compile()
    )


def Polygon(vertices: Sequence[tuple[float, float]]) -> GeometryDomain:
    """Construct an analytic simple polygonal region."""
    return GeometryDomain(PolygonSource(vertices, feature_id="domain-polygon").compile())


def Triangle(vertices: Sequence[tuple[float, float]]) -> GeometryDomain:
    """Construct an analytic triangular region."""
    if len(vertices) != 3:
        raise ValueError("Triangle must have exactly 3 vertices.")
    return GeometryDomain(PolygonSource(vertices, feature_id="domain-triangle").compile())


__all__ = ["Circle", "Ellipse", "Polygon", "Rectangle", "Square", "Triangle"]
