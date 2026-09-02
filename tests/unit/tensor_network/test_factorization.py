import jax.numpy as jnp
import pytest

import phydrax as phx


tn = phx.tensor_network


def test_two_site_truncation_uses_factorization_precision_and_reports_loss():
    precision = tn.TensorNetworkPrecisionPolicy(
        storage_dtype="complex64",
        contraction_dtype="complex64",
        factorization_dtype="complex128",
        accumulation_dtype="complex128",
        decision_dtype="float64",
    )
    state = tn.product_mps(
        jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.complex64),
        precision=precision,
    )
    root_two = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    hadamard = jnp.asarray([[1.0, 1.0], [1.0, -1.0]]) / root_two
    cnot = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=jnp.complex64,
    )
    gate = (cnot @ jnp.kron(hadamard, jnp.eye(2))).reshape((2, 2, 2, 2))

    result, evidence = tn.apply_two_site_gate(
        state,
        0,
        gate,
        maximum_bond_dimension=1,
        normalize=False,
    )

    assert evidence.retained_rank == 1
    assert evidence.available_rank == 2
    assert evidence.discarded_weight.dtype == jnp.dtype("float64")
    assert jnp.allclose(evidence.discarded_weight, 0.5, atol=1e-6)
    assert all(tensor.dtype == jnp.dtype("complex64") for tensor in result.tensors)
    assert jnp.allclose(result.norm() ** 2, 0.5, atol=1e-6)


def test_two_site_gate_rejects_nonpositive_capacity():
    state = tn.product_mps(jnp.asarray([[1.0, 0.0], [1.0, 0.0]]))
    gate = jnp.eye(4).reshape((2, 2, 2, 2))
    with pytest.raises(ValueError, match="positive"):
        tn.apply_two_site_gate(state, 0, gate, maximum_bond_dimension=0)


def test_canonicalization_preserves_state_and_precision():
    tensors = (
        jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.complex128),
        jnp.asarray([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=jnp.complex128),
    )
    state = tn.MatrixProductState(tensors)
    canonical, evidence = tn.canonicalize_mps(state, center=1)
    assert evidence.valid
    assert jnp.allclose(canonical.to_dense(), state.to_dense() / state.norm())
    assert canonical.precision.policy_id == state.precision.policy_id
