#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import build123d as bd
import jax.numpy as jnp
from shapely.geometry import Polygon as ShapelyPolygon

from ._from_cad import Geometry2DFromCAD


def Circle(
    center: tuple[float, float],
    radius: float,
) -> Geometry2DFromCAD:
    # Build a 2D Face on XY and export to a temporary STL, then load via Geometry2DFromCAD
    r"""A disk (filled circle) geometry.

    Defines the set

    $$
    \Omega = \{x\in\mathbb{R}^2 : \|x-c\|_2 \le r\},
    $$

    with center $c\in\mathbb{R}^2$ and radius $r>0$.

    **Arguments:**

    - `center`: The center $c=(c_x,c_y)$.
    - `radius`: The radius $r$.

    **Returns:**

    A `Geometry2DFromCAD` representing the circle.
    """
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    with bd.BuildSketch(bd.Plane.XY) as sk:
        with bd.Locations((cx, cy)):
            bd.Circle(r)
    face = sk.sketch
    tmp = Path(f"/tmp/bd2d_circle_{uuid4().hex}.stl").resolve()
    # Export the Face as STL and construct the geometry from it
    bd.export_stl(face, tmp, tolerance=1e-1, angular_tolerance=1e-1)
    geom = Geometry2DFromCAD(tmp, recenter=False)
    tmp.unlink(missing_ok=True)
    return geom


def Ellipse(
    center: tuple[float, float],
    x_radius: float,
    y_radius: float,
) -> Geometry2DFromCAD:
    r"""A filled ellipse geometry.

    Defines the set

    $$
    \Omega = \left\{(x,y)\in\mathbb{R}^2 :
      \left(\frac{x-c_x}{r_x}\right)^2 + \left(\frac{y-c_y}{r_y}\right)^2 \le 1
    \right\}.
    $$

    **Arguments:**

    - `center`: The center $c=(c_x,c_y)$.
    - `x_radius`: Semi-axis length $r_x>0$.
    - `y_radius`: Semi-axis length $r_y>0$.

    **Returns:**

    A `Geometry2DFromCAD` representing the ellipse.
    """
    cx, cy = float(center[0]), float(center[1])
    rx, ry = float(x_radius), float(y_radius)
    with bd.BuildSketch(bd.Plane.XY) as sk:
        with bd.Locations((cx, cy)):
            bd.Ellipse(rx, ry)
    face = sk.sketch
    tmp = Path(f"/tmp/bd2d_ellipse_{uuid4().hex}.stl").resolve()
    bd.export_stl(face, tmp, tolerance=1e-4, angular_tolerance=1e-2)
    geom = Geometry2DFromCAD(tmp, recenter=False)
    tmp.unlink(missing_ok=True)
    return geom


def Rectangle(
    center: tuple[float, float],
    width: float,
    height: float,
) -> Geometry2DFromCAD:
    r"""An axis-aligned rectangle geometry.

    Defines the set

    $$
    \Omega = \{(x,y)\in\mathbb{R}^2 :
      |x-c_x|\le \tfrac{w}{2},\ |y-c_y|\le \tfrac{h}{2}\}.
    $$

    **Arguments:**

    - `center`: The center $c=(c_x,c_y)$.
    - `width`: Width $w>0$.
    - `height`: Height $h>0$.

    **Returns:**

    A `Geometry2DFromCAD` representing the rectangle.
    """
    cx, cy = float(center[0]), float(center[1])
    w, h = float(width), float(height)
    with bd.BuildSketch(bd.Plane.XY) as sk:
        with bd.Locations((cx, cy)):
            bd.Rectangle(w, h)
    face = sk.sketch
    tmp = Path(f"/tmp/bd2d_rectangle_{uuid4().hex}.stl").resolve()
    bd.export_stl(face, tmp, tolerance=1e-4, angular_tolerance=0.001)
    geom = Geometry2DFromCAD(tmp, recenter=False)
    tmp.unlink(missing_ok=True)
    return geom


def Square(
    center: tuple[float, float],
    side: float,
) -> Geometry2DFromCAD:
    r"""An axis-aligned square geometry.

    This is the special case of `Rectangle` with width = height = `side`.

    **Arguments:**

    - `center`: The center $c=(c_x,c_y)$.
    - `side`: Side length $s>0$.

    **Returns:**

    A `Geometry2DFromCAD` representing the square.
    """
    return Rectangle(center, side, side)


def Polygon(
    vertices: Sequence[tuple[float, float]],
) -> Geometry2DFromCAD:
    r"""A polygonal planar region defined by its vertices.

    Given vertices $(v_1,\dots,v_n)$ ordered around the boundary (clockwise or
    counter-clockwise), this constructs a planar region whose boundary is the
    piecewise-linear closed curve connecting consecutive vertices.

    **Arguments:**

    - `vertices`: Sequence of vertices $v_i\in\mathbb{R}^2$.

    **Returns:**

    A `Geometry2DFromCAD` representing the polygon.
    """
    vertices_ = jnp.array(vertices, dtype=float)
    vertices_unique = jnp.unique(vertices_, axis=0)
    if vertices_.shape != vertices_unique.shape:
        sorted_indices = jnp.lexsort((vertices_[:, 1], vertices_[:, 0]))
        sorted_array = vertices_[sorted_indices]

        row_diffs = jnp.diff(sorted_array, axis=0)

        duplicate_indices = jnp.where(jnp.all(row_diffs == 0, axis=1))[0]

        non_unique_rows = sorted_array[duplicate_indices].tolist()
        non_unique_vertices = [tuple(row) for row in non_unique_rows]

        raise ValueError(f"Non-unique vertices: {non_unique_vertices}.")

    shapely_polygon = ShapelyPolygon(vertices_)
    if not shapely_polygon.is_valid:
        raise ValueError("Self-intersection(s) detected.")

    if not shapely_polygon.exterior.is_ccw:
        vertices_ = vertices_[::-1]

    # Build a Face from the polygon and export as STL.
    # Note: build123d's Polygon centers geometry in the sketch frame.
    # To preserve absolute coordinates, place a local frame at the vertex centroid
    # and express vertices relative to that frame.
    min_xy = jnp.min(vertices_, axis=0)
    max_xy = jnp.max(vertices_, axis=0)
    center = 0.5 * (min_xy + max_xy)
    rel = vertices_ - center
    with bd.BuildSketch(bd.Plane.XY) as sk:
        with bd.Locations((float(center[0]), float(center[1]))):
            bd.Polygon(*(tuple(map(float, v)) for v in rel))
    face = sk.sketch
    tmp = Path(f"/tmp/bd2d_polygon_{uuid4().hex}.stl").resolve()
    bd.export_stl(face, tmp, tolerance=1e-4, angular_tolerance=0.001)
    geom = Geometry2DFromCAD(tmp, recenter=False)
    tmp.unlink(missing_ok=True)
    return geom


def Triangle(
    vertices: Sequence[tuple[float, float]],
) -> Geometry2DFromCAD:
    r"""A triangular planar region defined by three vertices.

    **Arguments:**

    - `vertices`: The three vertices $v_1,v_2,v_3\in\mathbb{R}^2$.

    **Returns:**

    A `Geometry2DFromCAD` representing the triangle.
    """
    if len(vertices) != 3:
        raise ValueError("Triangle must have exactly 3 vertices.")
    return Polygon(vertices)
