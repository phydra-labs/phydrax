# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

from phydrax.operators.quantum import (
    apply_finite_cptp,
    compose_finite_cptp,
    finite_cptp_from_kraus,
    finite_cptp_from_local_kraus_operation,
    finite_cptp_from_unitary,
    LocalKrausChannelOperation,
)


def _amplitude_damping(probability=0.25):
    return jnp.array(
        [
            [[1.0, 0.0], [0.0, jnp.sqrt(1.0 - probability)]],
            [[0.0, jnp.sqrt(probability)], [0.0, 0.0]],
        ],
        dtype=jnp.complex64,
    )


def test_finite_cptp_representations_apply_and_compose_without_state_repair():
    damping = finite_cptp_from_kraus(_amplitude_damping())
    identity = finite_cptp_from_unitary(jnp.eye(2, dtype=jnp.complex64))
    excited = jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)
    output = apply_finite_cptp(damping, excited)
    assert bool(damping.valid)
    assert jnp.allclose(jnp.trace(output), 1.0)
    assert jnp.allclose(jnp.real(output[0, 0]), 0.25)
    composed = compose_finite_cptp(damping, identity)
    assert jnp.allclose(apply_finite_cptp(composed, excited), output)


def test_rectangular_finite_channel_has_explicit_input_output_dimensions():
    kraus = jnp.zeros((3, 2, 3), dtype=jnp.complex64)
    kraus = kraus.at[0, 0, 0].set(1.0)
    kraus = kraus.at[1, 1, 1].set(1.0)
    kraus = kraus.at[2, 0, 2].set(1.0)
    channel = finite_cptp_from_kraus(kraus)
    assert channel.input_dimension == 3
    assert channel.output_dimension == 2
    assert bool(channel.valid)
    assert apply_finite_cptp(channel, jnp.eye(3, dtype=jnp.complex64) / 3).shape == (2, 2)


def test_local_kraus_operation_adapter_preserves_channel_evidence():
    operation = LocalKrausChannelOperation(_amplitude_damping(), ("q",))
    adapted = finite_cptp_from_local_kraus_operation(operation)
    assert bool(adapted.valid)
