#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import build123d as bd
import numpy as np
from jaxtyping import ArrayLike

from ._mesh import Geometry3DFromCAD


def Sphere(
    center: tuple[float, float, float],
    radius: float,
) -> Geometry3DFromCAD:
    r"""A solid sphere geometry.

    Defines

    $$
    \Omega = \{x\in\mathbb{R}^3:\|x-c\|_2 \le r\}.
    $$

    **Arguments:**

    - `center`: The center $c\in\mathbb{R}^3$.
    - `radius`: The radius $r>0$.

    **Returns:**

    A `Geometry3DFromCAD` representing the sphere.
    """
    c = np.asarray(center, dtype=float)
    r = float(radius)
    shape = bd.Sphere(r).moved(bd.Location(tuple(c)))
    tmp = Path(f"/tmp/bd3d_sphere_{uuid4().hex}.stl").resolve()
    bd.export_stl(shape, tmp, tolerance=1e-3, angular_tolerance=5e-2)
    geom = Geometry3DFromCAD(tmp, recenter=False)
    tmp.unlink(missing_ok=True)
    return geom


def Ellipsoid(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> Geometry3DFromCAD:
    r"""A solid ellipsoid geometry.

    Defines

    $$
    \Omega = \left\{x\in\mathbb{R}^3:
      \left(\frac{x_1-c_1}{r_1}\right)^2+
      \left(\frac{x_2-c_2}{r_2}\right)^2+
      \left(\frac{x_3-c_3}{r_3}\right)^2 \le 1
    \right\}.
    $$

    **Arguments:**

    - `center`: The center $c\in\mathbb{R}^3$.
    - `radii`: Semi-axis lengths $(r_1,r_2,r_3)$.

    **Returns:**

    A `Geometry3DFromCAD` representing the ellipsoid.
    """
    c = np.asarray(center, dtype=float)
    rx, ry, rz = [float(x) for x in radii]
    # Scale a unit sphere to create an ellipsoid, then move to center
    shp = bd.Sphere(1.0)
    shp = bd.scale(shp, (rx, ry, rz))
    shp = shp.moved(bd.Location(tuple(c)))
    tmp = Path(f"/tmp/bd3d_ellipsoid_{uuid4().hex}.stl").resolve()
    bd.export_stl(shp, tmp, tolerance=1e-3, angular_tolerance=5e-2)
    geom = Geometry3DFromCAD(tmp, recenter=False)
    tmp.unlink(missing_ok=True)
    return geom


def Cuboid(
    center: tuple[float, float, float],
    dimensions: tuple[float, float, float],
) -> Geometry3DFromCAD:
    r"""An axis-aligned box (cuboid) geometry.
    
    Defines
    
    $$
    \Omega = \{x\in\mathbb{R}^3:
      |x_1-c_1|\le \tfrac{d_x}{2},\
      |x_2-c_2|\le \tfrac{d_y}{2},\
      |x_3-c_3|\le \tfrac{d_z}{2}\}.
    $$
    
    **Arguments:**
    
    - `center`: The center $c\in\mathbb{R}^3$.
    - `dimensions`: Side lengths $(d_x,d_y,d_z)$.
    
    **Returns:**
    
    A `Geometry3DFromCAD` representing the cuboid.
    """
    c = np.asarray(center, dtype=float)
    dx, dy, dz = [float(x) for x in dimensions]
    shp = bd.Box(dx, dy, dz).moved(bd.Location(tuple(c)))
    tmp = Path(f"/tmp/bd3d_cuboid_{uuid4().hex}.stl").resolve()
    bd.export_stl(shp, tmp, tolerance=1e-3, angular_tolerance=5e-2)
    geom = Geometry3DFromCAD(tmp, recenter=False)
    tmp.unlink(missing_ok=True)
    return geom


def Cube(
    center: tuple[float, float, float],
    side: float,
) -> Geometry3DFromCAD:
    r"""An axis-aligned cube geometry.

    This is the special case of `Cuboid` with all side lengths equal to `side`.

    **Arguments:**

    - `center`: The center $c\in\mathbb{R}^3$.
    - `side`: Side length $s>0$.

    **Returns:**

    A `Geometry3DFromCAD` representing the cube.
    """
    dims = (side, side, side)
    return Cuboid(center, dims)


def Cylinder(
    face_center: Sequence | ArrayLike,
    axis: Sequence | ArrayLike,
    radius: float,
    angle: float = 2 * np.pi,
) -> Geometry3DFromCAD:
    r"""A solid cylinder geometry.

    Constructs a cylinder by extruding a disk of radius $r$ along an axis vector
    $a\in\mathbb{R}^3$ of length $h=\|a\|_2$ starting from `face_center`.

    **Arguments:**

    - `face_center`: Center of the starting face.
    - `axis`: Extrusion vector $a$ (its direction sets the cylinder axis).
    - `radius`: Radius $r>0$.
    - `angle`: Angular opening in radians (currently full cylinders are the common use).

    **Returns:**

    A `Geometry3DFromCAD` representing the cylinder.
    """
    fc = np.asarray(face_center, dtype=float)
    ax = np.asarray(axis, dtype=float)
    r = float(radius)
    # Orient a workplane such that +Z aligns with the axis; extrude a circle by |axis|
    h = float(np.linalg.norm(ax))
    if h == 0:
        raise ValueError("Cylinder axis must have non-zero length.")
    z_dir = tuple((ax / h).tolist())
    wp = bd.Plane(origin=tuple(fc), z_dir=z_dir)
    with bd.BuildPart(wp) as bp:
        with bd.BuildSketch():
            bd.Circle(r)
        bd.extrude(amount=h)
    shp = bp.part
    tmp = Path(f"/tmp/bd3d_cylinder_{uuid4().hex}.stl").resolve()
    bd.export_stl(shp, tmp, tolerance=1e-3, angular_tolerance=5e-2)
    geom = Geometry3DFromCAD(tmp, recenter=False)
    tmp.unlink(missing_ok=True)
    return geom


def Cone(
    base_center: Sequence | ArrayLike,
    axis: Sequence | ArrayLike,
    radius0: float,
    radius1: float = 0.0,
    angle: float = 2 * np.pi,
) -> Geometry3DFromCAD:
    r"""A solid cone (or conical frustum) geometry.

    Constructs a cone/frustum by extruding a circle of radius `radius0` along an axis
    vector, linearly varying the radius to `radius1` at the end.

    **Arguments:**

    - `base_center`: Center of the base face.
    - `axis`: Extrusion vector (its length sets the height).
    - `radius0`: Base radius $r_0$.
    - `radius1`: Top radius $r_1$ (use $0$ for a pointed cone).
    - `angle`: Angular opening in radians.

    **Returns:**

    A `Geometry3DFromCAD` representing the cone.
    """
    bc = np.asarray(base_center, dtype=float)
    ax = np.asarray(axis, dtype=float)
    r0 = float(radius0)
    r1 = float(radius1)
    h = float(np.linalg.norm(ax))
    if h == 0:
        raise ValueError("Cone axis must have non-zero length.")
    z_dir = tuple((ax / h).tolist())
    wp = bd.Plane(origin=tuple(bc), z_dir=z_dir)
    with bd.BuildPart(wp) as bp:
        bd.Cone(r0, r1, h)
    shp = bp.part
    tmp = Path(f"/tmp/bd3d_cone_{uuid4().hex}.stl").resolve()
    bd.export_stl(shp, tmp, tolerance=1e-3, angular_tolerance=5e-2)
    geom = Geometry3DFromCAD(tmp, recenter=False)
    tmp.unlink(missing_ok=True)
    return geom


def Torus(
    center: Sequence | ArrayLike,
    inner_radius: float,
    outer_radius: float,
    angle: float = 2 * np.pi,
) -> Geometry3DFromCAD:
    r"""A solid torus geometry.

    This helper interprets `inner_radius`/`outer_radius` as the inner/outer radii of
    the torus tube. It converts them to the usual major/minor radii via
    $R=\tfrac{1}{2}(r_{\texttt{in}}+r_{\texttt{out}})$ and
    $r=\tfrac{1}{2}(r_{\texttt{out}}-r_{\texttt{in}})$.

    **Arguments:**

    - `center`: Center translation of the torus.
    - `inner_radius`: Inner tube radius $r_{\texttt{in}}$.
    - `outer_radius`: Outer tube radius $r_{\texttt{out}}$.
    - `angle`: Angular opening in radians.

    **Returns:**

    A `Geometry3DFromCAD` representing the torus.
    """
    c = np.asarray(center, dtype=float)
    ir = float(inner_radius)
    or_ = float(outer_radius)
    major = 0.5 * (ir + or_)
    minor = 0.5 * (or_ - ir)
    # Build full torus. Note: partial-angle torus not exported reliably as STL with watertight volume.
    shp = bd.Torus(major_radius=major, minor_radius=minor)
    shp = shp.moved(bd.Location(tuple(c)))
    tmp = Path(f"/tmp/bd3d_torus_{uuid4().hex}.stl").resolve()
    bd.export_stl(shp, tmp, tolerance=5e-2, angular_tolerance=8e-2)
    geom = Geometry3DFromCAD(tmp)
    tmp.unlink(missing_ok=True)
    return geom


def Wedge(
    x0: Sequence | ArrayLike,
    extends: Sequence | ArrayLike,
    top_extent: float,
) -> Geometry3DFromCAD:
    r"""A right angular wedge geometry.

    Constructs a wedge-like solid defined by a right-angle corner and edge extents.
    This is useful for simple ramp/wedge test geometries in mechanics and CFD.

    **Arguments:**

    - `x0`: Coordinates of the reference corner.
    - `extends`: Edge extents along each axis.
    - `top_extent`: Controls the top face extent along the $x$ direction.

    **Returns:**

    A `Geometry3DFromCAD` representing the wedge.
    """
    x0 = np.asarray(x0, dtype=float)
    ex = np.asarray(extends, dtype=float)
    te = float(top_extent)
    # Map to build123d Wedge parameters
    xsize, ysize, zsize = float(ex[0]), float(ex[1]), float(ex[2])
    xmin, xmax = 0.0, te
    zmin, zmax = 0.0, zsize
    shp = bd.Wedge(
        xsize,
        ysize,
        zsize,
        xmin,
        zmin,
        xmax,
        zmax,
        align=(bd.Align.MIN, bd.Align.MIN, bd.Align.MIN),
    )
    # Place the wedge so that its right-angle near corner is at x0
    shp = shp.moved(bd.Location(tuple(x0)))
    tmp = Path(f"/tmp/bd3d_wedge_{uuid4().hex}.stl").resolve()
    bd.export_stl(shp, tmp, tolerance=1e-3, angular_tolerance=5e-2)
    geom = Geometry3DFromCAD(tmp, recenter=False)
    tmp.unlink(missing_ok=True)
    return geom
