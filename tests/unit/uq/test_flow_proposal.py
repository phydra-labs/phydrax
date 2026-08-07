#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.uq._flow_proposal import (
    _build_default_flow,
    _fit_flow,
    _FlowProposalState,
    _independence_mh_scan,
    _initialize_replay,
    _replay_data,
    _update_replay,
)


def _normal_log_density(value, *, location=0.0):
    return -0.5 * (value - location) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi)


def test_independence_mh_uses_both_proposal_density_terms_and_rejects_nonfinite():
    current = jnp.asarray([0.25])
    proposal = jnp.asarray([[1.5], [jnp.nan]])
    current_target = _normal_log_density(current[0])
    current_proposal = _normal_log_density(current[0], location=1.0)
    proposed_target = jnp.asarray([_normal_log_density(1.5), -1.0])
    proposed_density = jnp.asarray([_normal_log_density(1.5, location=1.0), -1.0])
    expected = jnp.minimum(
        0.0,
        proposed_target[0] - current_target + current_proposal - proposed_density[0],
    )

    final, info = _independence_mh_scan(
        _FlowProposalState(current, current_target, current_proposal),
        proposal,
        proposed_target,
        proposed_density,
        jnp.asarray([-100.0, -100.0]),
    )

    assert jnp.allclose(info.log_acceptance_ratio[0], expected)
    assert info.accepted[0]
    assert not info.accepted[1]
    assert info.nonfinite[1]
    assert jnp.array_equal(final.position, proposal[0])


def test_asymmetric_independence_mh_recovers_the_target_not_the_proposal():
    count = 50_000
    proposal_key, acceptance_key = jr.split(jr.key(4))
    proposed_positions = 1.0 + jr.normal(proposal_key, (count, 1))
    proposed_log_targets = jax.vmap(lambda value: _normal_log_density(value[0]))(
        proposed_positions
    )
    proposed_log_densities = jax.vmap(
        lambda value: _normal_log_density(value[0], location=1.0)
    )(proposed_positions)
    log_uniforms = jnp.log(jr.uniform(acceptance_key, (count,)))
    initial = _FlowProposalState(
        jnp.zeros((1,)),
        _normal_log_density(0.0),
        _normal_log_density(0.0, location=1.0),
    )

    def transition(state, item):
        proposed, log_target, log_proposal, log_uniform = item
        next_state, _ = _independence_mh_scan(
            state,
            proposed[None, :],
            log_target[None],
            log_proposal[None],
            log_uniform[None],
        )
        return next_state, next_state.position[0]

    _, samples = jax.jit(
        lambda: jax.lax.scan(
            transition,
            initial,
            (
                proposed_positions,
                proposed_log_targets,
                proposed_log_densities,
                log_uniforms,
            ),
        )
    )()
    retained = samples[5_000:]

    assert jnp.abs(jnp.mean(retained)) < 0.05
    assert jnp.abs(jnp.var(retained) - 1.0) < 0.08
    assert jnp.abs(jnp.mean(retained) - 1.0) > 0.8


def test_chain_stratified_reservoir_is_bounded_and_reproducible():
    replay = _initialize_replay(
        num_chains=2,
        capacity_per_chain=3,
        dimension=1,
        dtype=jnp.float32,
    )
    samples = jnp.arange(10, dtype=jnp.float32).reshape((2, 5, 1))
    keys = jax.vmap(lambda key: jr.split(key, 5))(jr.split(jr.key(5), 2))

    first = jax.jit(_update_replay)(replay, samples, keys)
    second = jax.jit(_update_replay)(replay, samples, keys)

    assert jnp.array_equal(first.values, second.values)
    assert jnp.array_equal(first.size, jnp.asarray([3, 3]))
    assert jnp.array_equal(first.seen, jnp.asarray([5, 5]))
    assert _replay_data(first).shape == (6, 1)


def test_default_flow_supports_scalar_and_vector_events():
    scalar_data = jnp.linspace(-2.0, 2.0, 16)[:, None]
    vector_data = jnp.stack((scalar_data[:, 0], scalar_data[:, 0] ** 2), axis=1)
    scalar = _build_default_flow(
        jr.key(6),
        scalar_data,
        flow_layers=1,
        num_knots=4,
        nn_width=8,
        nn_depth=1,
    )
    vector = _build_default_flow(
        jr.key(7),
        vector_data,
        flow_layers=1,
        num_knots=4,
        nn_width=8,
        nn_depth=1,
    )

    scalar_sample, scalar_log_density = scalar.sample_and_log_prob(jr.key(8))
    vector_sample, vector_log_density = vector.sample_and_log_prob(jr.key(9))

    assert scalar.shape == (1,)
    assert vector.shape == (2,)
    assert scalar_sample.shape == (1,)
    assert vector_sample.shape == (2,)
    assert jnp.isfinite(scalar_log_density)
    assert jnp.isfinite(vector_log_density)


def test_flow_training_caps_oversized_batches_to_the_training_split():
    data = jnp.linspace(-2.0, 2.0, 10)[:, None]
    flow = _build_default_flow(
        jr.key(10),
        data,
        flow_layers=1,
        num_knots=4,
        nn_width=8,
        nn_depth=1,
    )

    trained, training_loss, validation_loss = _fit_flow(
        jr.key(11),
        flow,
        data,
        learning_rate=1e-3,
        max_epochs=1,
        max_patience=1,
        batch_size=64,
        validation_fraction=0.2,
    )

    assert trained.shape == (1,)
    assert training_loss.shape == (1,)
    assert validation_loss.shape == (1,)
    assert jnp.all(jnp.isfinite(training_loss))
    assert jnp.all(jnp.isfinite(validation_loss))
