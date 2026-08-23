#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_intrinsic_transport_cost_and_riemannian_flow_metric():
    sphere = phx.metrix.SphereManifold(3)
    cost = phx.transport.IntrinsicSquaredDistanceCost(sphere)
    points = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    matrix = cost.matrix(points, points)
    assert jnp.allclose(jnp.diag(matrix), 0.0)
    assert jnp.allclose(matrix[0, 1], (jnp.pi / 2.0) ** 2)

    chart = phx.metrix.CoordinateChart("plane", ("x", "y"))
    metric = phx.metrix.diagonal_metric(
        lambda q: jnp.asarray([2.0, 3.0]),
        chart=chart,
    )
    flow_metric = phx.terms.RiemannianFlowMatchingMetric(metric)
    value = flow_metric(
        jnp.asarray([0.2, -0.1]),
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([0.0, 0.0]),
    )
    assert jnp.allclose(value, 14.0)
