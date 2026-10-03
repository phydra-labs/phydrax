# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.meshfree import (
    MeshfreeExteriorCalculusPlan,
    SurfaceMeshShift,
    SurfaceShiftPolicy,
)


_POINTS = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
_NORMALS = np.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])


def _line_shift(strength: float = 1.0) -> SurfaceMeshShift:
    exterior = MeshfreeExteriorCalculusPlan(
        _POINTS[:, :1],
        1.1,
        3,
        node_volumes=np.asarray([1.0, 1.0, 1.0]),
        dirichlet=np.asarray([True, False, True]),
    ).prepare()
    return SurfaceMeshShift(
        SurfaceShiftPolicy(strength=strength, target_separation=1.5), exterior
    )


def test_shift_velocity_is_tangential_repulsion_below_the_target_separation() -> None:
    policy = SurfaceShiftPolicy(strength=2.0, target_separation=1.5)
    tilted = _POINTS + np.asarray([0.0, 0.3])[None, :] * np.asarray([[0.0], [1.0], [0.0]])
    velocity = np.asarray(
        policy.velocity(tilted, _NORMALS, np.asarray([[0, 1], [1, 2]], dtype=np.int32))
    )
    np.testing.assert_allclose(velocity[:, 1], 0.0, atol=0)
    assert velocity[0, 0] < 0 < velocity[2, 0]
    relaxed = SurfaceShiftPolicy(strength=2.0, target_separation=0.5).velocity(
        _POINTS, _NORMALS, np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    )
    np.testing.assert_allclose(relaxed, 0.0, atol=0)


def test_relative_transport_is_conservative_upwind_opposite_to_the_mesh() -> None:
    shift = _line_shift()
    mesh = np.asarray(shift.velocity(_POINTS, _NORMALS))
    # The end nodes move outward; material, at rest, crosses the mesh inward.
    assert mesh[0, 0] < 0 < mesh[2, 0]
    concentration = np.asarray([1.0, 2.0, 3.0])
    rate = shift.rate(concentration, _POINTS, -mesh)
    np.testing.assert_allclose(jnp.sum(rate.content_rate), 0.0, atol=1e-14)
    np.testing.assert_allclose(rate.conservation_residual, 0.0, atol=1e-14)
    content_rate = np.asarray(rate.content_rate)
    assert content_rate[0] < 0 and content_rate[2] < 0 and content_rate[1] > 0
    # Donor cell: each end node exports its own concentration.
    np.testing.assert_allclose(
        content_rate[0] / content_rate[2],
        concentration[0]
        * rate.node_outflow[0]
        / (concentration[2] * rate.node_outflow[2]),
        rtol=1e-12,
    )
    assert shift.metric_nonnegative
    assert float(rate.stable_step) > 0


def test_relative_transport_vanishes_without_relative_velocity() -> None:
    rate = _line_shift().rate(
        np.asarray([1.0, 2.0, 3.0]), _POINTS, np.zeros_like(_POINTS)
    )
    np.testing.assert_allclose(rate.content_rate, 0.0, atol=0)
    np.testing.assert_allclose(rate.node_outflow, 0.0, atol=0)
