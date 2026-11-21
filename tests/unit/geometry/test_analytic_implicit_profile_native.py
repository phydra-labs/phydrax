#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

import phydrax as phx
from benchmarks._meshing_surfaces import icosphere
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.units import METER


_G = phx.geometry


def test_sphere_profile_binds_reach_and_certifies_a_surface_both_ways() -> None:
    center = np.asarray((0.013, -0.021, 0.007), dtype=np.float64)
    geometry = _G.Sphere(tuple(center), 0.4).compile()
    profile = AnalyticImplicitProfile(geometry, SpatialCoordinateContract(METER))

    assert profile.family == "sphere"
    assert profile.component_count == 1
    assert profile.genus == 0
    assert 0.0 < profile.tube_radius < profile.reach_lower
    unit, faces = icosphere(3)
    vertices = center + 0.4 * unit
    reverse, proof = profile.certify_zero_set_cover(vertices, faces, 0.02)

    assert proof in ("tube", "cover")
    assert 0.0 <= reverse <= 0.02
    # Every sphere point lies within the certified reverse distance of a vertex
    # Facets the certificate would have to repair are refused, never repaired.
    with pytest.raises(ValueError, match="zero-set certificate"):
        profile.certify_zero_set_cover(vertices, faces, 1.0e-5)
    # Inward orientation is not the field's: refused.
    with pytest.raises(ValueError, match="zero-set certificate"):
        profile.certify_zero_set_cover(vertices, faces[:, ::-1], 0.02)


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
