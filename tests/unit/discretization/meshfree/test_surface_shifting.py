# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.meshfree._shifting import (
    surface_relative_advection,
    SurfaceShiftPolicy,
)
from phydrax.graph import graph_to_cochain_complex, GraphIR
from phydrax.sparse import SparseCoordinateOperator


def _incidence() -> SparseCoordinateOperator:
    graph = GraphIR(
        nodes=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        edges={"weight": np.asarray([1.0])},
        senders=np.asarray([0], dtype=np.int32),
        receivers=np.asarray([1], dtype=np.int32),
        n_node=np.asarray([2]),
        n_edge=np.asarray([1]),
    )
    native = graph_to_cochain_complex(
        graph,
        edge_weight_key="weight",
        node_measure=np.asarray([1.0, 1.0]),
        edge_semantics="undirected_once",
    )
    incidence = native.hilbert_complex().differential(0)
    if not isinstance(incidence, SparseCoordinateOperator):
        raise TypeError(
            "The native graph differential must retain sparse incidence structure."
        )
    return incidence


def test_pure_mesh_motion_upwinds_opposite_to_mesh_and_conserves() -> None:
    x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    mesh = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    result = surface_relative_advection(
        np.asarray([1.0, 2.0], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        x,
        _incidence(),
        np.asarray([1.0]),
        jnp.zeros_like(x),
        mesh,
        0.1,
        require_positivity=True,
    )
    np.testing.assert_allclose(result.oriented_volume_flux, [-1.0], atol=1e-12)
    np.testing.assert_allclose(result.content, [1.2, 1.8], atol=1e-12)
    np.testing.assert_allclose(result.conservation_residual, 0.0, atol=1e-12)
    assert bool(result.positivity_admitted) and bool(result.successful)


def test_positivity_refuses_cfl_and_signed_metrics_without_clamping() -> None:
    x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    velocity = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    incidence = _incidence()
    for metric, dt in ((np.asarray([1.0]), 2.0), (np.asarray([-1.0]), 0.1)):
        result = surface_relative_advection(
            np.asarray([1.0, 2.0], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
            x,
            incidence,
            metric,
            velocity,
            jnp.zeros_like(x),
            dt,
            require_positivity=True,
        )
        assert not bool(result.positivity_admitted)
        assert not bool(result.successful)
        np.testing.assert_allclose(result.content, [1.0, 2.0], atol=1e-12)


def test_shift_projects_repulsion_and_refuses_large_displacement() -> None:
    x = jnp.array([[0.0, 0.0], [0.1, 0.0]])
    normal = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    policy = SurfaceShiftPolicy(
        target_separation=1.0, maximum_displacement=0.01, strength=1.0
    )
    result = policy.propose(
        x,
        normal,
        np.asarray([[0, 1]], dtype=np.int32),
        1.0,
        lambda p: p.at[:, 1].set(0),
        lambda p: p[:, 1],
    )
    assert not bool(result.successful)
    np.testing.assert_allclose(result.points, x, atol=0)
    accepted = policy.propose(
        x,
        normal,
        np.asarray([[0, 1]], dtype=np.int32),
        0.001,
        lambda p: p.at[:, 1].set(0),
        lambda p: p[:, 1],
    )
    assert bool(accepted.successful)
    assert float(accepted.points[1, 0] - accepted.points[0, 0]) > 0.1
    np.testing.assert_allclose(accepted.points[:, 1], 0.0, atol=0)
