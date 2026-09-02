#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.equations.fem._embedded_coupling import (
    ConservativeOversetPlan,
    CutCellConservationPlan,
    OversetConnectivity,
    SlidingMortarPlan,
)
from phydrax.equations.fem._moving_conservation import MovingTraceRoute
from phydrax.equations.fem._trace_routes import PreparedDGTraceRoute


def test_overset_transfer_preserves_constants_and_accounts_content():
    connectivity = OversetConnectivity(
        jnp.asarray(((0, 1), (1, 2))),
        jnp.asarray((3, 4)),
        jnp.asarray(((0.25, 0.75), (0.5, 0.5))),
        jnp.asarray(((0.2, 0.3), (0.1, 0.4))),
        jnp.asarray((0.5, 0.5)),
        jnp.asarray((True, True)),
    )
    plan = ConservativeOversetPlan(connectivity)
    constant = jnp.ones((5, 2))
    constant_result = plan.transfer(constant)
    np.testing.assert_allclose(constant_result.receptor_state, 1.0)
    assert constant_result.successful
    state = jnp.asarray(((1.0,), (2.0,), (4.0,), (0.0,), (0.0,)))
    result = plan.transfer(state)
    assert result.conservation_defect <= 1.0e-12
    np.testing.assert_allclose(
        jnp.sum(result.donor_content_correction, axis=0),
        jnp.sum(
            jnp.asarray(((0.2, 0.3), (0.1, 0.4)))
            * state[jnp.asarray(((0, 1), (1, 2)))][..., 0],
            axis=(0, 1),
        )
        - jnp.sum(result.receptor_content),
        atol=3.0e-12,
    )


def test_cut_cell_merge_and_sliding_mortar_are_exactly_conservative():
    cut = CutCellConservationPlan(
        jnp.asarray((0.02, 0.8, 1.0)),
        jnp.ones((3, 2)),
        jnp.asarray((1, -1, -1)),
    )
    contents = jnp.asarray(((1.0,), (2.0,), (3.0,)))
    merged = cut.merge_small_cell_contents(contents)
    np.testing.assert_allclose(jnp.sum(merged), jnp.sum(contents))
    np.testing.assert_allclose(merged[:, 0], (0.0, 3.0, 3.0))

    current = PreparedDGTraceRoute(
        "conforming",
        jnp.asarray((0,)),
        neighbour_dofs=jnp.asarray((1,)),
        owner_basis=jnp.asarray(((1.0,),)),
        neighbour_basis=jnp.asarray(((1.0,),)),
        physical_points=jnp.asarray(((0.0, 0.0),)),
        physical_weights=jnp.asarray((1.0,)),
        normal=jnp.asarray(((1.0, 0.0),)),
        route_id="current",
    )
    next_route = PreparedDGTraceRoute(
        "conforming",
        jnp.asarray((0,)),
        neighbour_dofs=jnp.asarray((1,)),
        owner_basis=jnp.asarray(((1.0,),)),
        neighbour_basis=jnp.asarray(((1.0,),)),
        physical_points=jnp.asarray(((0.1, 0.0),)),
        physical_weights=jnp.asarray((1.2,)),
        normal=jnp.asarray(((1.0, 0.0),)),
        route_id="next",
    )
    sliding = SlidingMortarPlan(
        (MovingTraceRoute(current, next_route),), jnp.asarray((0.75,))
    )
    owner, neighbour = sliding.flux_contributions(0.5, (jnp.asarray(((2.0,),)),))[0]
    np.testing.assert_allclose(owner + neighbour, 0.0, atol=2.0e-12)
