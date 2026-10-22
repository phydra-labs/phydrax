#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.implicit import _analytic_profile
from phydrax.geometry.implicit._analytic_profile import AnalyticDiscoveryShape
from phydrax.units import METER


_G = phx.geometry
_CENTER = np.asarray([0.0123, -0.0071, 0.0173])
_UNIT = np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])


def _sphere() -> Any:
    return _G.Sphere(tuple(_CENTER), 0.75).compile(coordinate_unit=METER)


def _torus() -> Any:
    # Minor radius 0.3 about the major radius 0.6.
    return _G.Torus(tuple(_CENTER), 0.3, 0.9).compile(coordinate_unit=METER)


def _gradients(family: str, points: np.ndarray) -> np.ndarray:
    # Independent closed forms of the exact signed-distance gradients.
    relative = points - _CENTER
    if family == "sphere":
        return relative / np.linalg.norm(relative, axis=-1, keepdims=True)
    rho = np.hypot(relative[..., 0], relative[..., 1])
    radial, height = rho - 0.6, relative[..., 2]
    tube = np.hypot(radial, height)
    return np.stack(
        (
            radial / tube * relative[..., 0] / rho,
            radial / tube * relative[..., 1] / rho,
            height / tube,
        ),
        axis=-1,
    )


def _surface_boxes(family: str, count: int, size: float) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(7)
    first = generator.uniform(0.0, 2.0 * math.pi, count)
    second = generator.uniform(-0.5 * math.pi, 0.5 * math.pi, count)
    if family == "sphere":
        second = np.arcsin(generator.uniform(-1.0, 1.0, count))
        points = 0.75 * np.stack(
            (
                np.cos(second) * np.cos(first),
                np.cos(second) * np.sin(first),
                np.sin(second),
            ),
            axis=-1,
        )
    else:
        second = generator.uniform(0.0, 2.0 * math.pi, count)
        ring = 0.6 + 0.3 * np.cos(second)
        points = np.stack(
            (ring * np.cos(first), ring * np.sin(first), 0.3 * np.sin(second)),
            axis=-1,
        )
    offset = generator.uniform(-1.0, 0.0, (count, 3)) * size
    lower = _CENTER + points + offset
    return lower, lower + size


def _sampled(family: str, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 7)
    weights = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    return _gradients(family, lower[:, None] + weights[None] * (upper - lower)[:, None])


@pytest.mark.parametrize(
    ("family", "geometry"), [("sphere", _sphere), ("ring-torus", _torus)]
)
def test_analytic_normal_cone_encloses_and_tracks_sampled_gradients(
    family: str, geometry: Any
) -> None:
    cone = AnalyticDiscoveryShape.from_geometry(geometry())
    assert cone is not None and cone.family == family
    lower, upper = _surface_boxes(family, 64, 0.08)

    spread = cone.gradient_spread(lower, upper)

    gradients = _sampled(family, lower, upper)
    chords = np.linalg.norm(gradients[:, :, None] - gradients[:, None, :], axis=-1)
    diameter = np.max(chords, axis=(1, 2))
    assert np.all(spread < 1.0)
    # Two gradients within ``spread`` of one unit axis are at most twice apart,
    # and no enclosing ball is narrower than half the sampled diameter; the
    # analytic cone stays within 1.5 times that floor.
    assert np.all(diameter <= 2.0 * spread)
    assert np.all(spread <= 0.75 * diameter)


def test_analytic_normal_cone_keeps_unit_spread_on_singular_loci() -> None:
    sphere = AnalyticDiscoveryShape.from_geometry(_sphere())
    torus = AnalyticDiscoveryShape.from_geometry(_torus())
    assert sphere is not None and torus is not None
    core = _CENTER + np.asarray([0.6, 0.0, 0.0])
    axis = _CENTER + np.asarray([0.0, 0.0, 0.2])

    assert sphere.gradient_spread(_CENTER[None] - 0.05, _CENTER[None] + 0.05)[0] == 1.0
    assert torus.gradient_spread(core[None] - 0.05, core[None] + 0.05)[0] == 1.0
    assert torus.gradient_spread(axis[None] - 0.05, axis[None] + 0.05)[0] == 1.0


def test_analytic_normal_cone_refuses_unestablished_kernels() -> None:
    sector = _G.Torus(tuple(_CENTER), 0.3, 0.9, angle=math.pi).compile(
        coordinate_unit=METER
    )
    union = (_G.Sphere((0.0, 0.0, 0.0), 0.4) | _G.Sphere((0.5, 0.0, 0.0), 0.4)).compile(
        coordinate_unit=METER
    )

    assert AnalyticDiscoveryShape.from_geometry(sector) is None
    assert AnalyticDiscoveryShape.from_geometry(union) is None


def _discover_torus() -> Any:
    policy = _G.AdaptiveImplicitSurfacePolicy(maximum_level=7, flatness_tolerance=0.03)
    return _G.discover_adaptive_implicit_surface(
        _torus(), domain=1.3 * _UNIT, policy=policy, source_id="cone"
    )


def test_discovery_uses_the_analytic_cone_and_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analytic = _discover_torus()
    monkeypatch.setattr(
        _analytic_profile.AnalyticDiscoveryShape,
        "from_geometry",
        classmethod(lambda cls, geometry: None),
    )
    interval = _discover_torus()

    for result in (analytic, interval):
        assert result.evidence.accuracy == "certified"
        assert result.evidence.maximum_flatness_bound <= 0.03
        assert result.mesh.topology.euler_characteristic == 0
    assert analytic.evidence.leaf_count < interval.evidence.leaf_count


def test_support_discovery_covers_the_authored_domain_with_excluded_slabs() -> None:
    domain = 1.3 * _UNIT
    result = _discover_torus()

    cover = result.cover
    leaves = result.evidence.leaf_count
    assert cover.complete and cover.certified and cover.uncovered_volume == 0.0
    assert np.array_equal(cover.domain, domain)
    assert np.array_equal(result.volume.domain, domain)
    assert cover.boxes.shape[0] == leaves + 6
    # The six slabs outside the refined support exclude the zero set.
    assert np.all(cover.value_lower[leaves:] > 0.0)
    hull = np.stack(
        (np.min(result.volume.leaf_lower, axis=0), np.max(result.volume.leaf_upper, axis=0))
    )
    assert np.all(hull[0] > domain[0]) and np.all(hull[1] < domain[1])
    corner = result.volume.classify_points(domain[:1] + 0.01)
    assert corner.certified and corner.classes.tolist() == [0]
    restored = type(result).from_stored_fields(result)
    assert restored.cover.cover_id == cover.cover_id
    assert restored.evidence.evidence_id == result.evidence.evidence_id


def test_unproved_support_refines_the_whole_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    # A box cutting through the torus leaves surface in its slabs, so the
    # slab enclosures cannot exclude zero and the support is not used.
    def clipped(self: Any, domain: np.ndarray) -> np.ndarray:
        return np.stack((_CENTER - [1.0, 1.0, 0.1], _CENTER + [1.0, 1.0, 0.1]))

    monkeypatch.setattr(_analytic_profile.AnalyticDiscoveryShape, "computational_box", clipped)
    result = _discover_torus()

    assert result.evidence.accuracy == "certified"
    assert result.cover.boxes.shape[0] == result.evidence.leaf_count
    assert np.array_equal(np.min(result.volume.leaf_lower, axis=0), 1.3 * _UNIT[0])
    assert np.array_equal(np.max(result.volume.leaf_upper, axis=0), 1.3 * _UNIT[1])
