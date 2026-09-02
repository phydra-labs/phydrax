# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp

import phydrax as phx


def test_public_finite_chain_and_tensor_train_reference_oracles():
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))
    state = phx.tensor_network.product_mps(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128)
    )
    operator = phx.tensor_network.product_mpo(jnp.stack((z, z)))
    assert jnp.allclose(phx.tensor_network.mps_mpo_expectation(state, operator), -1.0)

    dense = jnp.arange(24.0, dtype=jnp.float64).reshape((2, 3, 4))
    decomposition = phx.tensor_train.tt_svd(
        dense,
        max_ranks=(2, 4),
        relative_tolerance=0.0,
    )
    assert jnp.allclose(
        decomposition.tensor.to_dense(max_entries=dense.size), dense, atol=1e-10
    )
    assert decomposition.evidence.frobenius_error_bound >= 0.0


def test_public_network_and_symmetry_reference_oracles():
    local = jnp.asarray([1.0, 0.0], dtype=jnp.float64).reshape((1, 1, 1, 1, 2))
    state = phx.tensor_network.PEPS((local,), 1, 1)
    contraction = phx.tensor_network.contract_peps_exact(state)
    assert contraction.evidence.exact
    assert jnp.allclose(contraction.value, 1.0)

    _, _, recoupling = phx.tensor_network.su2_recoupling_matrix(1, 1, 1, 1)
    assert jnp.allclose(
        recoupling @ recoupling.T,
        jnp.eye(recoupling.shape[0]),
        atol=1e-10,
    )
    assert phx.tensor_network.su2_pentagon_residual(1, 1, 1, 1, 0) < 1e-10


def test_public_quantum_instrument_probability_oracle():
    zero = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex64)
    one = jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)
    instrument = phx.solver.QuantumInstrument(
        jnp.stack((zero, one))[:, None, :, :],
        jnp.ones((2, 1), dtype=bool),
        tolerance=1e-5,
    )
    plus = jnp.asarray([1.0, 1.0], dtype=jnp.complex64) / jnp.sqrt(2.0)
    result = phx.solver.apply_dense_quantum_instrument(instrument, plus)
    assert result.valid
    assert jnp.allclose(result.probabilities, 0.5, atol=1e-6)
    assert result.probability_sum_residual < 1e-6
