#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _action(dimension=2, beta=1.4):
    topology = phx.discretization.polygonal_cell_complex(
        jnp.asarray([[0, 1, 2]]), None, 3
    )
    boundary = phx.discretization.prepare_cell_boundary_paths(topology)
    group = phx.metrix.SpecialUnitaryGroup(dimension)
    space = phx.graph.MatrixGaugeLinkSpace(topology, group)
    action = phx.operators.path_integral.WilsonGaugeAction(
        space,
        boundary,
        plaquette_couplings=beta,
    )
    return action


def _perturbed_links(action):
    dimension = action.link_space.group.algebra_shape[0]
    coordinates = 0.1 * jnp.reshape(
        jnp.arange(action.link_space.num_edges * dimension, dtype=float) + 1.0,
        (action.link_space.num_edges, dimension),
    )
    return action.geometry.retract(action.link_space.identity(), coordinates)


def test_wilson_action_uses_normalized_trace_and_explicit_constant():
    for dimension in (2, 3):
        action = _action(dimension)
        identity = action.link_space.identity()

        assert jnp.allclose(action.action(identity), -1.4)
        assert jnp.allclose(action.canonical_action(identity), 0.0)
        assert action.evidence.reference_measure == "product-haar"
        assert action.link_space.contains(identity)


def test_wilson_action_is_gauge_invariant():
    action = _action(2)
    links = _perturbed_links(action)
    group = action.link_space.group
    vertex_coordinates = jnp.asarray(
        [[0.05, 0.02, -0.01], [-0.03, 0.04, 0.02], [0.01, -0.02, 0.03]]
    )
    vertices = jax.vmap(lambda value: group.exp(group.hat(value)))(vertex_coordinates)
    transformed = phx.graph.gauge_transform_links(
        action.link_space,
        links,
        vertices,
    )

    assert jnp.allclose(action.action(links), action.action(transformed), atol=1e-10)


def test_wilson_incremental_cache_matches_full_recomputation():
    action = _action(2)
    links = _perturbed_links(action)
    value, cache = action.initialize_incremental(links)
    edge = 1
    displacement = jnp.asarray([0.07, -0.03, 0.02])
    proposed = links.at[edge].set(
        action.link_space.group.compose(
            links[edge],
            action.link_space.group.exp(action.link_space.group.hat(displacement)),
        )
    )
    payload = phx.operators.path_integral.GaugeLinkProposalPayload(
        edge=jnp.asarray(edge, dtype=jnp.int32)
    )
    delta, candidate, valid = action.propose_incremental(
        links,
        cache,
        proposed,
        payload,
    )
    refreshed, refreshed_cache = action.refresh_incremental(proposed)

    assert valid
    assert jnp.allclose(value, action.action(links))
    assert jnp.allclose(delta, action.action(proposed) - action.action(links))
    assert jnp.allclose(candidate.action, refreshed)
    assert jnp.allclose(
        candidate.plaquette_holonomies,
        refreshed_cache.plaquette_holonomies,
    )


def test_wilson_local_coordinate_gradient_matches_directional_difference():
    action = _action(2)
    links = _perturbed_links(action)
    direction = jnp.asarray(
        [[0.03, -0.02, 0.01], [0.01, 0.04, -0.03], [-0.02, 0.01, 0.02]]
    )
    gradient = phx.operators.path_integral.lattice_action_local_gradient(action, links)
    step = 1e-5
    forward = action.geometry.retract(links, step * direction)
    backward = action.geometry.retract(links, -step * direction)
    finite_difference = (action.action(forward) - action.action(backward)) / (2.0 * step)

    assert jnp.allclose(jnp.vdot(gradient, direction), finite_difference, rtol=1e-4)
