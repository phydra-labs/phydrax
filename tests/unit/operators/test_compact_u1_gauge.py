#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _measure(beta=0.8):
    topology = phx.discretization.polygonal_cell_complex(
        jnp.asarray([[0, 1, 2]]), None, 3
    )
    return phx.operators.path_integral.CompactU1GaugeMeasure(topology, beta=beta)


def test_compact_u1_uses_topology_and_explicit_action_offset():
    measure = _measure()
    links = jnp.asarray([0.2, -0.4, 0.1])
    phases = jnp.asarray([0.3, -0.2, 0.1])
    transformed = measure.gauge_transform(links, phases)

    assert measure.topology_id == measure.topology.topology_id
    assert measure.geometry.contains(measure.geometry.wrap(links))
    assert jnp.allclose(measure.action(links), measure.action(transformed))
    assert jnp.allclose(
        measure.canonical_action(links),
        measure.action(links) + measure.beta * measure.num_plaquettes,
    )


def test_compact_u1_incremental_cache_matches_full_action():
    measure = _measure()
    links = jnp.asarray([jnp.pi - 0.05, -0.4, 0.1])
    proposal = phx.sampling.SingleCoordinatePeriodicProposal(2.0 * jnp.pi, 0.4)
    move = proposal.propose(jax.random.key(3), links)
    value, cache = measure.initialize_incremental(links)
    delta, candidate, valid = measure.propose_incremental(
        links, cache, move.position, move.payload
    )
    refreshed, refreshed_cache = measure.refresh_incremental(move.position)

    assert valid
    assert jnp.allclose(value, measure.action(links))
    assert jnp.allclose(delta, measure.action(move.position) - measure.action(links))
    assert jnp.allclose(candidate.action, refreshed)
    assert jnp.allclose(candidate.plaquette_angles, refreshed_cache.plaquette_angles)


def test_compact_u1_runs_through_incremental_markov_sampling():
    measure = _measure(beta=0.5)
    target = phx.operators.path_integral.incremental_target_from_lattice_action(
        measure,
        refresh_cadence=4,
    )
    kernel = phx.sampling.MetropolisHastings(
        phx.sampling.SingleCoordinatePeriodicProposal(2.0 * jnp.pi, 0.6)
    )
    state = kernel.initialize(
        target,
        jnp.zeros((2, measure.num_edges)),
    )
    result = phx.sampling.sample_markov(
        target,
        kernel,
        state,
        key=jax.random.key(8),
        warmup_steps=4,
        num_draws=8,
    )

    assert result.samples.shape == (2, 8, measure.num_edges)
    assert jnp.all(jnp.isfinite(result.log_target))
    assert jnp.all(result.final_state.valid)
