#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_single_coordinate_gaussian_proposal_is_normalized_and_symmetric():
    proposal = phx.sampling.SingleCoordinateGaussianProposal(0.3)
    current = jnp.asarray([0.1, -0.2, 0.4, 0.7])
    move = proposal.propose(jax.random.key(1), current)

    assert move.valid
    assert jnp.sum(move.position != current) == 1
    assert jnp.allclose(move.log_forward, move.log_reverse)
    assert move.payload.index >= 0
    assert move.payload.index < current.size
    assert jnp.allclose(
        move.payload.displacement,
        move.position[move.payload.index] - current[move.payload.index],
    )


def test_single_coordinate_periodic_proposal_crosses_principal_seam():
    proposal = phx.sampling.SingleCoordinatePeriodicProposal(2.0 * jnp.pi, 1.2)
    current = jnp.asarray([jnp.pi - 0.01])
    move = proposal.propose(jax.random.key(8), current)

    assert move.valid
    assert jnp.all(move.position >= -jnp.pi)
    assert jnp.all(move.position < jnp.pi)
    assert jnp.allclose(move.log_forward, move.log_reverse)
    assert jnp.abs(move.payload.displacement) <= 1.2


def test_single_coordinate_proposals_reject_nonlocal_transitions():
    proposal = phx.sampling.SingleCoordinateGaussianProposal(0.4)
    current = jnp.zeros((3,))
    proposed = jnp.asarray([0.2, -0.1, 0.0])

    assert jnp.isneginf(proposal.log_prob(proposed, current))
    with pytest.raises(TypeError, match="real inexact array"):
        proposal.sample(jax.random.key(0), jnp.arange(3))
    with pytest.raises(ValueError, match="period / 2"):
        phx.sampling.SingleCoordinatePeriodicProposal(2.0, 1.1)


def test_single_coordinate_proposals_are_jittable():
    current = jnp.zeros((8,))
    gaussian = phx.sampling.SingleCoordinateGaussianProposal(0.1)
    periodic = phx.sampling.SingleCoordinatePeriodicProposal(2.0 * jnp.pi, 0.5)

    gaussian_move = jax.jit(gaussian.propose)(jax.random.key(2), current)
    periodic_move = jax.jit(periodic.propose)(jax.random.key(3), current)

    assert gaussian_move.valid
    assert periodic_move.valid
