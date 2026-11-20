#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.units import METER


_G = phx.geometry


def test_sphere_profile_binds_reach_and_whole_boundary_cover() -> None:
    geometry = _G.Sphere((0.013, -0.021, 0.007), 0.4).compile()
    profile = AnalyticImplicitProfile(geometry, SpatialCoordinateContract(METER))

    assert profile.family == "sphere"
    assert profile.component_count == 1
    assert profile.genus == 0
    assert 0.0 < profile.tube_radius < profile.reach_lower
    sample_cover = profile.boundary_cover(0.15)

    assert sample_cover.semantics == "certified" and sample_cover.complete
    assert np.max(sample_cover.covering_radius) <= 0.15
    assert np.all(np.isfinite(sample_cover.covering_radius))
    assert np.max(sample_cover.covering_radius) < profile.radius
    measured = profile.boundary_distance(sample_cover.points)
    np.testing.assert_array_equal(measured.upper, sample_cover.point_error)
    assert np.all(measured.lower <= measured.upper)


def test_sphere_boundary_cover_is_the_first_count_whose_every_cell_is_covered() -> None:
    geometry = _G.Sphere((0.013, -0.021, 0.007), 0.4).compile()
    profile = AnalyticImplicitProfile(geometry, SpatialCoordinateContract(METER))
    radius = 0.02
    cover = profile.boundary_cover(radius)
    count = round(np.sqrt(cover.points.shape[0] // 6))
    assert 6 * count * count == cover.points.shape[0]
    initial, _ = profile._cover_counts(radius)
    # The diagonal probe refutes counts; it must not move the admissible one.
    assert initial < count
    for candidate in range(initial, count + 1):
        first, second = np.meshgrid(
            np.arange(candidate, dtype=np.int64),
            np.arange(candidate, dtype=np.int64),
            indexing="ij",
        )
        points, radii = profile._cell_radii(
            candidate, candidate, first.reshape((-1,)), second.reshape((-1,))
        )
        if candidate < count:
            assert np.max(radii) > radius
        else:
            np.testing.assert_array_equal(points, cover.points)
            np.testing.assert_array_equal(radii, cover.covering_radius)


def test_full_ring_torus_profile_establishes_positive_reach_and_certified_queries() -> (
    None
):
    geometry = _G.Torus((0.012, -0.007, 0.017), 0.3, 0.9).compile()
    profile = AnalyticImplicitProfile(geometry, SpatialCoordinateContract(METER))

    assert profile.family == "ring-torus"
    assert profile.component_count == 1
    assert profile.genus == 1
    assert profile.reach_lower > 0.0
    query = profile.boundary_distance(np.asarray(((0.012 + 0.9, -0.007, 0.017),)))
    assert query.semantics == "certified"
    assert 0.0 <= query.lower[0] <= query.upper[0] <= 1.0e-12
    samples = profile.boundary_cover(0.15)
    assert samples.semantics == "certified" and samples.complete
    assert np.max(samples.covering_radius) <= 0.15
