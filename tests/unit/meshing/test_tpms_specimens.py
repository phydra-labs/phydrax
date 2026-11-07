#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools
import math

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import PeriodicCell
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    AdaptiveImplicitSurfaceStatus,
    discover_adaptive_implicit_surface,
)
from phydrax.meshing import LatticeSpecimenSurface, LatticeSpecimenVolume


_G = phx.geometry
_M = phx.meshing
_POLICY = AdaptiveImplicitSurfacePolicy(relative_cell_diameter=0.125, root_tolerance=1e-9)
type _Vector = tuple[float, float, float]


def _schwarz_d() -> phx.geometry.TrigonometricLevelSet:
    return _G.TrigonometricLevelSet(
        _G.TrigonometricTable.family("schwarz_d"), PeriodicCell(np.eye(3))
    )


def _network_length(
    x: np.ndarray, y: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Exact length of ``{F <= 0}`` on the vertical segments ``(x, y, [low, high])``.

    ``F = cos(X - Y) sin Z + sin(X + Y) cos Z = R sin(Z + phi)`` with
    ``(X, Y, Z) = 2 pi (x, y, z)``, ``R cos phi = cos(X - Y)`` and
    ``R sin phi = sin(X + Y)``; ``sin <= 0`` on ``[pi, 2 pi]`` of each period.
    """
    angle_x, angle_y = 2.0 * math.pi * x, 2.0 * math.pi * y
    phase = np.arctan2(np.sin(angle_x + angle_y), np.cos(angle_x - angle_y))

    def below(theta: np.ndarray) -> np.ndarray:
        turns = np.floor(theta / (2.0 * math.pi))
        rest = theta - 2.0 * math.pi * turns
        return math.pi * turns + np.clip(rest - math.pi, 0.0, math.pi)

    return (below(2.0 * math.pi * high + phase) - below(2.0 * math.pi * low + phase)) / (
        2.0 * math.pi
    )


def _disc_quadrature(
    radius: float, count: int = 384
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre in the radius and the periodic trapezoid in the angle."""
    nodes, weights = np.polynomial.legendre.leggauss(count)
    radii = 0.5 * radius * (nodes + 1.0)
    angles = 2.0 * math.pi * np.arange(2 * count) / (2 * count)
    r, theta = np.meshgrid(radii, angles, indexing="ij")
    w = (0.5 * radius * weights * radii)[:, None] * (2.0 * math.pi / (2 * count))
    return r * np.cos(theta), r * np.sin(theta), np.broadcast_to(w, r.shape)


def _cylinder_oracle(center: _Vector, height: float, radius: float) -> float:
    x, y, weight = _disc_quadrature(radius)
    low = np.full(x.shape, center[2])
    return float(
        np.sum(weight * _network_length(center[0] + x, center[1] + y, low, low + height))
    )


def _box_oracle(center: _Vector, size: _Vector, count: int = 384) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(count)
    x = center[0] + 0.5 * size[0] * nodes
    y = center[1] + 0.5 * size[1] * nodes
    gx, gy = np.meshgrid(x, y, indexing="ij")
    w = np.outer(weights, weights) * 0.25 * size[0] * size[1]
    low = np.full(gx.shape, center[2] - 0.5 * size[2])
    return float(np.sum(w * _network_length(gx, gy, low, low + size[2])))


def _sphere_oracle(center: _Vector, radius: float, count: int = 384) -> float:
    # r = radius sin t keeps the half chord radius cos t smooth at the rim.
    nodes, weights = np.polynomial.legendre.leggauss(count)
    t = 0.25 * math.pi * (nodes + 1.0)
    radii = radius * np.sin(t)
    half = radius * np.cos(t)
    angles = 2.0 * math.pi * np.arange(2 * count) / (2 * count)
    r, theta = np.meshgrid(radii, angles, indexing="ij")
    h = np.broadcast_to(half[:, None], r.shape)
    w = (0.25 * math.pi * weights * radii * radius * np.cos(t))[:, None] * (
        2.0 * math.pi / (2 * count)
    )
    return float(
        np.sum(
            w
            * _network_length(
                center[0] + r * np.cos(theta),
                center[1] + r * np.sin(theta),
                center[2] - h,
                center[2] + h,
            )
        )
    )


def _boundary_area(surface: LatticeSpecimenSurface) -> float:
    corners = np.asarray(surface.vertices)[np.asarray(surface.triangles)]
    return float(
        0.5
        * np.sum(
            np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            )
        )
    )


def _assert_certified_specimen(volume: LatticeSpecimenVolume, oracle: float) -> None:
    surface = volume.surface
    assert volume.certified_solid
    assert volume.boundary_matches_surface
    assert volume.result.audit.passed
    assert volume.covers_surface
    # Creases are protected: every crease vertex is a mesh vertex.
    assert surface.crease_segments.shape[0] > 0
    # Measure versus the independent exact-per-line oracle, within the
    # certified fidelity of the boundary (tube of the bound over its area).
    fidelity = max(surface.implicit_face_distance, surface.cap_deviation)
    assert math.isfinite(fidelity)
    assert abs(volume.measure - oracle) <= _boundary_area(surface) * fidelity
    assert 0.0 < volume.porosity < 1.0


@pytest.fixture(scope="module")
def cylinder_volume() -> LatticeSpecimenVolume:
    envelope = _G.Cylinder((0.13, 0.21, 0.07), (0.0, 0.0, 0.4), 0.25)
    surface = _M.construct_lattice_specimen_surface(
        _G.Intersection(_schwarz_d(), envelope), policy=_POLICY, source_id="cylinder"
    )
    return _M.mesh_lattice_specimen(surface)


def test_cylinder_specimen_is_a_certified_solid_matching_the_exact_volume(
    cylinder_volume: LatticeSpecimenVolume,
) -> None:
    surface = cylinder_volume.surface
    assert surface.facet_names == ("lattice", "side", "base", "top")
    assert surface.cap_triangle_count > 0 and surface.implicit_triangle_count > 0
    _assert_certified_specimen(
        cylinder_volume, _cylinder_oracle((0.13, 0.21, 0.07), 0.4, 0.25)
    )


def test_box_specimen_is_a_certified_solid_matching_the_exact_volume() -> None:
    center, size = (0.31, -0.12, 0.27), (0.5, 0.4, 0.35)
    surface = _M.construct_lattice_specimen_surface(
        _G.Intersection(_schwarz_d(), _G.Box(center, size)),
        policy=_POLICY,
        source_id="box",
    )

    volume = _M.mesh_lattice_specimen(surface)

    _assert_certified_specimen(volume, _box_oracle(center, size))
    assert surface.envelope_edge_segments.shape[0] > 0


def test_sphere_specimen_is_a_certified_solid_matching_the_exact_volume() -> None:
    center, radius = (0.11, 0.37, -0.08), 0.27
    surface = _M.construct_lattice_specimen_surface(
        _G.Intersection(_schwarz_d(), _G.Sphere(center, radius)),
        policy=_POLICY,
        source_id="sphere",
    )

    volume = _M.mesh_lattice_specimen(surface)

    _assert_certified_specimen(volume, _sphere_oracle(center, radius))


def _octant_sphere_triangles(radius: float, count: int) -> np.ndarray:
    """Flat triangles of the projected octahedron, every vertex on the sphere."""
    rows = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        corners = np.diag(signs)

        def point(i: int, j: int, corners: np.ndarray = corners) -> np.ndarray:
            value = i * corners[0] + j * corners[1] + (count - i - j) * corners[2]
            return radius * value / np.linalg.norm(value)

        for i in range(count):
            for j in range(count - i):
                rows.append((point(i, j), point(i + 1, j), point(i, j + 1)))
                if i + j < count - 1:
                    rows.append((point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)))
    return np.asarray(rows)


def _origin_distance(triangles: np.ndarray) -> np.ndarray:
    """Exact distance from the origin to each triangle whose circumcentre is inside.

    For vertices on a sphere about the origin the foot of the origin on the
    triangle's plane is the circumcentre; the tests use acute triangles only.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normal = np.cross(b - a, c - a)
    foot = (
        normal
        * (np.einsum("ij,ij->i", normal, a) / np.sum(normal * normal, axis=1))[:, None]
    )
    for first, second in ((a, b), (b, c), (c, a)):
        side = np.einsum("ij,ij->i", np.cross(second - first, foot - first), normal)
        assert np.all(side > 0.0)
    return np.linalg.norm(foot, axis=1)


def test_implicit_face_bound_dominates_the_exact_sphere_distance() -> None:
    from phydrax.meshing._surface_construction import _implicit_face_distances

    radius = 0.3
    triangles = _octant_sphere_triangles(radius, 6)
    witness = np.full((triangles.shape[0], 3), 1e-12)

    bounds = _implicit_face_distances(
        _G.Sphere((0.0, 0.0, 0.0), radius).compile(), triangles, witness
    )

    # The field is ||x|| - r; a flat triangle with its vertices on the sphere
    # lies inside it, at most r - (distance of the origin to the triangle).
    exact = radius - _origin_distance(triangles)
    assert np.all(np.isfinite(bounds))
    assert np.all(bounds >= exact)
    # The gradient-flow bound is the sharper one: below the root witness.
    longest = np.max(
        np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2), axis=1
    )
    assert np.all(bounds < longest / math.sqrt(3.0))


def test_implicit_face_bound_falls_back_to_the_root_witness() -> None:
    from phydrax.meshing._surface_construction import _implicit_face_distances

    # An equatorial triangle through the centre: the gradient of ||x|| - r has
    # no positive lower bound on any box around the origin, which is at the
    # exact distance r from the sphere. The root witness, e / sqrt(3) plus the
    # vertex root distance with e = r sqrt(3), is still finite and certified.
    radius = 0.3
    angles = 2.0 * math.pi * np.arange(3) / 3.0
    triangle = radius * np.stack((np.cos(angles), np.sin(angles), np.zeros(3)), axis=1)
    vertex_distance = 1e-12

    (bound,) = _implicit_face_distances(
        _G.Sphere((0.0, 0.0, 0.0), radius).compile(),
        triangle[None],
        np.full((1, 3), vertex_distance),
    )

    assert radius <= bound <= radius * (1.0 + 1e-12) + vertex_distance


def test_crease_tangent_to_the_envelope_is_a_certified_unresolved_event() -> None:
    # The Schwarz P surface cos X + cos Y + cos Z = 0 is tangent to z = 1/4 at
    # (0, 1/2, 1/4), where the face trace has a saddle on its zero set. The
    # box's lower face is that plane; its chart lattice avoids the saddle.
    schwarz_p = _G.TrigonometricLevelSet(
        _G.TrigonometricTable.family("schwarz_p"), PeriodicCell(np.eye(3))
    )
    low = np.asarray((-0.213, 0.327, 0.25))
    high = np.asarray((0.171, 0.707, 0.6))
    box = _G.Box(0.5 * (low + high), high - low)

    with pytest.raises(_M.LatticeSpecimenEventError) as caught:
        _M.construct_lattice_specimen_surface(
            _G.Intersection(schwarz_p, box), policy=_POLICY, source_id="tangent"
        )

    discovery = caught.value.discovery
    assert discovery.outcome == "tangency"
    assert discovery.faces.shape == (0, 3)
    (chart,) = [chart for chart in discovery.charts if chart.outcome == "tangency"]
    assert (chart.axis, chart.side) == (2, 0)
    contact = (np.asarray((0.0, 0.5, 0.25)) - 0.5 * (low + high)) / (0.5 * (high - low))
    boxes = np.asarray(chart.event_boxes)
    assert np.any(np.all((boxes[:, 0] <= contact) & (contact <= boxes[:, 1]), axis=1))


def test_closed_domain_discovery_default_is_unchanged() -> None:
    # Without the declared clipped mode, the same field and box keep the
    # closed-domain refusal of a zero set crossing the domain boundary.
    surface = discover_adaptive_implicit_surface(
        _schwarz_d().compile(),
        domain=np.asarray(((0.07, 0.11, 0.13), (0.83, 0.79, 0.91))),
        policy=_POLICY,
        source_id="closed",
    )

    assert surface.mesh is None
    assert (
        surface.evidence.status_flags
        & AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING
    )
