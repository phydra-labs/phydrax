import jax.numpy as jnp

import phydrax as phx


tn = phx.tensor_network


def test_mps_mpo_environments_and_expectation_match_dense_reference():
    identity = jnp.eye(2, dtype=jnp.complex128)
    pauli_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    operator = tn.product_mpo(jnp.stack((pauli_z, identity)))
    state = tn.product_mps(jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128))

    left, right = tn.build_mps_mpo_environments(state, operator, state)
    expected = jnp.vdot(state.to_dense(), operator.to_dense() @ state.to_dense())

    assert len(left) == state.site_count + 1
    assert len(right) == state.site_count + 1
    assert left[0].shape == right[-1].shape == (1, 1, 1)
    assert jnp.allclose(tn.mps_mpo_expectation(state, operator), expected)
    assert jnp.allclose(tn.mps_mpo_inner(state, operator, state), expected)


def test_mpo_inner_norm_and_hermiticity_are_network_native():
    identity = jnp.eye(2, dtype=jnp.complex128)
    pauli_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    lowering = jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex128)
    hermitian = tn.product_mpo(jnp.stack((pauli_x, identity)))
    nonhermitian = tn.product_mpo(jnp.stack((lowering, identity)))

    dense = hermitian.to_dense()
    assert jnp.allclose(tn.mpo_inner(hermitian, hermitian), jnp.vdot(dense, dense))
    assert jnp.allclose(tn.mpo_norm(hermitian), jnp.linalg.norm(dense))
    assert jnp.allclose(tn.mpo_hermiticity_residual(hermitian), 0.0, atol=1e-12)
    assert tn.mpo_hermiticity_residual(nonhermitian) > 0.0
