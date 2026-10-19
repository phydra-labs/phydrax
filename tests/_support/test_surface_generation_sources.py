# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent authored-net anchors for the shared rational spline source."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry import PatchCurveUse
from tests._support.surface_generation_sources import accounting_spline_domain


# Rational weighted averages require only a few float64 multiply/add operations.
_SOURCE_POINT_ROUNDING = 32 * np.finfo(np.float64).eps


def test_accounting_spline_domain_retains_authored_rational_net_and_trim() -> None:
    domain = accounting_spline_domain()
    assert domain.source_id == "native-spline-normal-owner"
    assert domain.source_revision == "authored"
    assert len(domain.patches) == 1
    assert len(domain.curves) == 4
    patch = domain.patches[0]
    sites = jnp.asarray(
        ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.5)),
        dtype=jnp.float64,
    )
    expected = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.05),
            (1.0, 0.0, 0.1),
            (1.0, 1.0, 0.2),
            (0.525, 0.475, 0.08875),
        ),
        dtype=np.float64,
    )

    # At the center all four tensor Bernstein factors equal 1/4, so the
    # rational point is the weighted control average, not their plain mean.
    def require_authored_points(points: np.ndarray) -> None:
        np.testing.assert_allclose(
            points, expected, rtol=0.0, atol=_SOURCE_POINT_ROUNDING
        )

    evaluated = np.asarray(patch.surface.evaluate(sites))
    require_authored_points(evaluated)
    corrupted = evaluated.copy()
    corrupted[-1] = (0.5, 0.5, 0.0875)
    with pytest.raises(AssertionError):
        require_authored_points(corrupted)
    assert len(patch.loops) == 1
    assert len(patch.loops[0]) == 4
    corners = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    for row, use in enumerate(patch.loops[0]):
        assert isinstance(use, PatchCurveUse)
        assert use.curve == row
        assert (use.first, use.last) == (0.0, 1.0)
        endpoints = use.pcurve.evaluate(jnp.asarray((0.0, 1.0), dtype=jnp.float64))
        np.testing.assert_array_equal(endpoints, corners[[row, (row + 1) % 4]])
