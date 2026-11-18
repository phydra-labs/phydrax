#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import numpy as np

import phydrax as phx
from phydrax.discretization import PeriodicCell
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    discover_periodic_trigonometric_surface,
)
from phydrax.geometry.implicit._periodic_discovery import QuotientZeroSet


def test_projection_decides_exact_reach_boundary() -> None:
    """On this fixture, exact squared distance is
    418399173461941767479947330913/81129638414606681695789005144064.

    The outside and inside reach squares are respectively
    26777547101564269689882063722689/5192296858534827628530496329220096
    and 6694386775391070009821820912481/1298074214633706907132624082305024.
    """
    source = phx.geometry.TrigonometricLevelSet(
        phx.geometry.TrigonometricTable.family("schwarz_p"), PeriodicCell(np.eye(3))
    )
    surface = discover_periodic_trigonometric_surface(
        source,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.1, root_tolerance=1.0e-8
        ),
        source_id="reach-gate-schwarz-p",
    )
    assert surface.resolved
    zero_set = QuotientZeroSet(surface)
    # The endpoints and threshold are binary64 rationals. Exact squared
    # distance, rather than a rounded norm, decides whether reach is exceeded.
    start = np.asarray(
        [
            [
                float.fromhex("0x1.a226192c277c7p-1"),
                float.fromhex("0x1.36c3e329f4dd2p-1"),
                float.fromhex("0x1.684994cb4b324p-1"),
            ]
        ],
        dtype=np.float64,
    )
    reach = float.fromhex("0x1.2625d56e29a61p-4")
    inside_reach = float.fromhex("0x1.2625d56e29a62p-4")
    axis = np.asarray([-1], dtype=np.int64)
    point, root_error, _, accepted = zero_set.project(start, axis, 10.0)
    assert accepted[0] and np.isfinite(root_error[0])
    exact_squared = sum(
        (Fraction(float(point[0, i])) - Fraction(float(start[0, i]))) ** 2
        for i in range(3)
    )
    # The adjacent thresholds squared are exactly these rational values.
    assert Fraction(reach) ** 2 == Fraction(
        26777547101564269689882063722689, 5192296858534827628530496329220096
    )
    assert Fraction(inside_reach) ** 2 == Fraction(
        6694386775391070009821820912481, 1298074214633706907132624082305024
    )
    assert Fraction(reach) ** 2 < exact_squared <= Fraction(inside_reach) ** 2

    # The first bracket has opposite certified signs even at zero reach, so
    # the refusal is caused by distance admission, not bracket discovery.
    newton, _, gradient, zero_reach = zero_set.project(start, axis, 0.0)
    assert not zero_reach[0]
    direction = gradient[0] / np.linalg.norm(gradient, axis=1)[0]
    offset = (2.0 * zero_set.tolerance) * direction
    signs = zero_set.signs(np.stack((newton[0] - offset, newton[0] + offset)))
    assert signs[0] != 0 and signs[0] == -signs[1]
    _, refused_error, _, refused = zero_set.project(start, axis, reach)
    assert not refused[0]
    assert np.isinf(refused_error[0])
    inside_point, inside_error, _, inside = zero_set.project(start, axis, inside_reach)
    assert inside[0] and np.isfinite(inside_error[0])
    np.testing.assert_array_equal(inside_point, point)
