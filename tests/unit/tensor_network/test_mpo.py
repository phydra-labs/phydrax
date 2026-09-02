import jax.numpy as jnp
import pytest

import phydrax as phx


tn = phx.tensor_network


def _operators():
    identity = jnp.eye(2, dtype=jnp.complex128)
    pauli_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    return identity, pauli_x


def test_product_mpo_action_composition_and_addition_match_dense_reference():
    identity, pauli_x = _operators()
    operator = tn.product_mpo(jnp.stack((pauli_x, identity)))
    state = tn.product_mps(jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.complex128))

    applied, action_evidence = tn.apply_mpo(operator, state, maximum_bond_dimension=4)
    composed, composition_evidence = tn.compose_mpo(
        operator, operator, maximum_bond_dimension=4
    )
    summed = tn.add_mpo(operator, operator)

    assert jnp.allclose(applied.to_dense(), operator.to_dense() @ state.to_dense())
    assert jnp.allclose(composed.to_dense(), jnp.eye(4))
    assert jnp.allclose(summed.to_dense(), 2.0 * operator.to_dense())
    assert jnp.allclose(action_evidence.accumulated_discarded_weight, 0.0)
    assert jnp.allclose(composition_evidence.accumulated_discarded_weight, 0.0)


def test_mpo_adjoint_and_compression_preserve_exact_bond_one_operator():
    identity, _ = _operators()
    phase = jnp.asarray([[1.0, 0.0], [0.0, 1.0j]], dtype=jnp.complex128)
    operator = tn.product_mpo(jnp.stack((phase, identity)))
    adjoint = tn.adjoint_mpo(operator)
    compressed, evidence = tn.compress_mpo(operator, maximum_bond_dimension=1)

    assert jnp.allclose(adjoint.to_dense(), jnp.conj(operator.to_dense().T))
    assert jnp.allclose(compressed.to_dense(), operator.to_dense())
    assert jnp.allclose(evidence.accumulated_discarded_weight, 0.0)


def test_chain_structure_ids_track_shapes_not_values():
    identity, pauli_x = _operators()
    first = tn.product_mpo(jnp.stack((identity, identity)))
    second = tn.product_mpo(jnp.stack((pauli_x, identity)))
    assert first.structure_id == second.structure_id
    assert first.bond_dimensions == (1,)


def test_dense_mpo_and_lpdo_materialization_are_capacity_bounded():
    identity, _ = _operators()
    operator = tn.product_mpo(jnp.stack((identity, identity)))
    purification = tn.LocallyPurifiedDensity(
        (
            jnp.asarray([[[[1.0]], [[0.0]]]], dtype=jnp.complex128),
            jnp.asarray([[[[1.0]], [[0.0]]]], dtype=jnp.complex128),
        )
    )
    with pytest.raises(ValueError, match="capacity"):
        operator.to_dense(maximum_elements=15)
    with pytest.raises(ValueError, match="capacity"):
        purification.to_dense_density(maximum_elements=15)
