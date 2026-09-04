#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.optics.geometric._nonsequential import NonSequentialSurfaceTable
from phydrax.optics.transport._tissue import (
    prepare_tissue_transport,
    simulate_tissue_transport,
    TissueTransportCoefficients,
    TissueTransportPlan,
)


def test_layered_nonscattering_slab_matches_piecewise_beer_lambert_reference():
    count = 32_768
    vertices = jnp.asarray(
        [
            [-20.0, -20.0, 0.4],
            [20.0, -20.0, 0.4],
            [20.0, 20.0, 0.4],
            [-20.0, 20.0, 0.4],
            [-20.0, -20.0, 1.0],
            [20.0, -20.0, 1.0],
            [20.0, 20.0, 1.0],
            [-20.0, 20.0, 1.0],
        ]
    )
    triangles = jnp.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=jnp.int32)
    surfaces = NonSequentialSurfaceTable(
        vertices,
        triangles,
        jnp.asarray([0, 0, 1, 1]),
        jnp.asarray([1, 1, 2, 2]),
        jnp.asarray([1.0, 1.0, 1.0]),
        surface_ids=jnp.asarray([0, 0, 1, 1]),
    )
    coefficients = TissueTransportCoefficients(
        jnp.asarray([0.2, 0.7, 0.0]),
        jnp.zeros((3,)),
        jnp.zeros((3,)),
        jnp.ones((3,)),
    )
    prepared = prepare_tissue_transport(
        TissueTransportPlan(surfaces, coefficients, maximum_interactions=3)
    )
    origins = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 0.0]), (count, 3))
    directions = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (count, 3))
    result = simulate_tissue_transport(
        prepared,
        origins,
        directions,
        jnp.zeros((count,), dtype=jnp.int32),
        jr.PRNGKey(4401),
        photon_ids=jnp.arange(count),
    )

    reference = jnp.exp(-(0.2 * 0.4 + 0.7 * 0.6))
    uncertainty = result.standard_errors.escape
    assert abs(float(result.tallies.escape - reference)) < max(
        4.0 * float(uncertainty), 0.01
    )
    np.testing.assert_allclose(
        result.tallies.absorption.sum() + result.tallies.escape,
        1.0,
        atol=2e-6,
    )
    assert float(result.maximum_absolute_ledger_residual) < 2e-6
