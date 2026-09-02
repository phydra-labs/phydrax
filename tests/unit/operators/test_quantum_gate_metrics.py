#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


q = phx.operators.quantum


def test_unitary_gate_quality_separates_leakage_and_conditional_fidelity():
    angle = jnp.asarray(0.4)
    rotation = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, jnp.cos(angle), -jnp.sin(angle)],
            [0.0, jnp.sin(angle), jnp.cos(angle)],
        ],
        dtype=jnp.complex128,
    )
    subspace = q.BasisStateSubspace(3, (0, 1))
    result = q.unitary_gate_quality(rotation, jnp.eye(2), subspace)
    expected_survival = 0.5 * (1.0 + jnp.cos(angle) ** 2)
    expected_average = (1.0 + jnp.cos(angle) ** 2 + (1.0 + jnp.cos(angle)) ** 2) / 6.0

    assert bool(result.diagnostics.valid)
    assert result.representation == "effective-operator"
    assert jnp.allclose(result.survival, expected_survival)
    assert jnp.allclose(result.leakage, 1.0 - expected_survival)
    assert jnp.allclose(result.average_fidelity, expected_average)
    assert jnp.allclose(
        result.conditional_fidelity,
        expected_average / expected_survival,
    )


def test_unitary_and_channel_gate_quality_agree_for_a_unitary_channel():
    unitary = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    subspace = q.BasisStateSubspace(2, (0, 1))
    unitary_result = q.unitary_gate_quality(unitary, unitary, subspace)
    channel_result = q.finite_channel_gate_quality(
        q.finite_cptp_from_unitary(unitary),
        unitary,
        subspace,
    )

    assert jnp.allclose(unitary_result.survival, 1.0)
    assert jnp.allclose(unitary_result.average_fidelity, 1.0)
    assert jnp.allclose(channel_result.survival, unitary_result.survival)
    assert jnp.allclose(channel_result.average_fidelity, unitary_result.average_fidelity)
    assert bool(channel_result.diagnostics.valid)


def test_finite_channel_gate_quality_matches_amplitude_damping_formula():
    probability = jnp.asarray(0.3)
    kraus = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, jnp.sqrt(1.0 - probability)]],
            [[0.0, jnp.sqrt(probability)], [0.0, 0.0]],
        ],
        dtype=jnp.complex128,
    )
    channel = q.finite_cptp_from_kraus(kraus)
    subspace = q.BasisStateSubspace(2, (0, 1))
    result = q.finite_channel_gate_quality(channel, jnp.eye(2), subspace)
    expected = (2.0 + (1.0 + jnp.sqrt(1.0 - probability)) ** 2) / 6.0

    assert jnp.allclose(result.survival, 1.0)
    assert jnp.allclose(result.leakage, 0.0)
    assert jnp.allclose(result.average_fidelity, expected)
    assert jnp.allclose(result.conditional_fidelity, expected)


def test_coherent_pauli_expansion_reconstructs_without_rate_semantics():
    operator = jnp.asarray(
        [[jnp.exp(0.2j), 0.0], [0.0, jnp.exp(-0.2j)]],
        dtype=jnp.complex128,
    )
    expansion = q.coherent_pauli_expansion(operator)

    assert bool(expansion.valid)
    assert expansion.qubit_count == 1
    assert float(expansion.reconstruction_residual) < 1e-12
    assert jnp.allclose(jnp.sum(expansion.weights), 1.0)
    with pytest.raises(ValueError, match="power of two"):
        q.coherent_pauli_expansion(jnp.eye(3))
