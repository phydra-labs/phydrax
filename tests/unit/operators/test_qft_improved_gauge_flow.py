#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.discretization import polygonal_cell_complex, prepare_cell_boundary_paths
from phydrax.graph import MatrixGaugeLinkSpace
from phydrax.metrix import SpecialUnitaryGroup
from phydrax.operators.path_integral._improved_gauge import (
    gauge_flow_observables,
    GaugeGradientFlowPlan,
    GaugeLoopTerm,
    ImprovedGaugeAction,
)


def _action_and_links():
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    boundary = prepare_cell_boundary_paths(topology)
    space = MatrixGaugeLinkSpace(topology, SpecialUnitaryGroup(2))
    action = ImprovedGaugeAction(
        space,
        (GaugeLoopTerm(boundary.paths, 1.7, name="plaquette"),),
    )
    coordinates = jnp.asarray(
        [[0.28, -0.11, 0.04], [-0.12, 0.19, 0.07], [0.08, 0.03, -0.17]]
    )
    links = jax.vmap(lambda value: space.group.exp(space.group.hat(value)))(coordinates)
    return action, links


def test_local_loop_action_is_the_exact_incident_subset_of_full_action():
    action, links = _action_and_links()
    contributions = action.loop_contributions(links)

    assert contributions.shape == (1,)
    assert jnp.allclose(action.action(links), contributions[0])
    for edge in range(action.link_space.num_edges):
        assert jnp.allclose(action.local_action(links, edge), action.action(links))
    assert jnp.allclose(action.canonical_action(action.link_space.identity()), 0.0)


def test_group_gradient_flow_preserves_membership_and_descends_action():
    action, links = _action_and_links()
    flow = GaugeGradientFlowPlan(
        action,
        step_size=0.2,
        steps=5,
        maximum_backtracks=8,
    )
    result = jax.jit(lambda values: flow.flow(values))(links)
    observables = gauge_flow_observables(action, result.links)

    assert result.evidence.successful
    assert action.link_space.contains(result.links)
    assert jnp.all(result.action_history[1:] <= result.action_history[:-1] + 1e-8)
    assert result.action_history[-1] <= result.action_history[0]
    assert observables.membership
    assert observables.finite
