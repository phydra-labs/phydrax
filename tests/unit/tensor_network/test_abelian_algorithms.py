import jax.numpy as jnp

import phydrax as phx


tn = phx.tensor_network


def _charged_bell_state():
    group = tn.AbelianGroup((None,))
    physical = tn.AbelianLeg(group, ((0,), (1,)), (1, 1), orientation=1)
    left_boundary = tn.AbelianLeg(group, ((0,),), (1,), orientation=1)
    middle_left = tn.AbelianLeg(group, ((0,), (1,)), (1, 1), orientation=-1)
    middle_right = middle_left.dual()
    right_boundary = tn.AbelianLeg(group, ((1,),), (1,), orientation=-1)
    left_layout = tn.AbelianTensorLayout((left_boundary, physical, middle_left))
    right_layout = tn.AbelianTensorLayout((middle_right, physical, right_boundary))
    root_two = jnp.sqrt(2.0)
    left_dense = jnp.zeros((1, 2, 2), dtype=jnp.complex128)
    left_dense = left_dense.at[0, 0, 0].set(1.0 / root_two)
    left_dense = left_dense.at[0, 1, 1].set(1.0 / root_two)
    right_dense = jnp.zeros((2, 2, 1), dtype=jnp.complex128)
    right_dense = right_dense.at[0, 1, 0].set(1.0)
    right_dense = right_dense.at[1, 0, 0].set(1.0)
    state = tn.AbelianMatrixProductState(
        (
            tn.AbelianTensor.from_dense(left_layout, left_dense),
            tn.AbelianTensor.from_dense(right_layout, right_dense),
        )
    )
    return state, physical


def test_global_cross_sector_truncation_reports_discarded_weight():
    state, _ = _charged_bell_state()
    gate = jnp.eye(4, dtype=jnp.complex128).reshape((2, 2, 2, 2))
    truncated, evidence = tn.apply_abelian_two_site_gate(
        state,
        0,
        gate,
        maximum_bond_dimension=1,
        normalize=True,
    )

    assert evidence.retained_rank == 1
    assert evidence.available_rank == 2
    assert tuple(evidence.per_sector_retained_ranks.tolist()) == (1, 0)
    assert jnp.allclose(evidence.discarded_weight, 0.5, atol=1e-9)
    assert jnp.allclose(truncated.norm(), 1.0)
    assert jnp.allclose(
        truncated.to_dense(), jnp.asarray([0.0, 1.0, 0.0, 0.0]), atol=1e-9
    )


def test_abelian_canonicalization_and_zero_tebd_preserve_state():
    state, physical = _charged_bell_state()
    canonical = tn.canonicalize_abelian_mps(state, center=1)
    zero = jnp.zeros((4, 4), dtype=jnp.complex128)
    hamiltonian = tn.AbelianNearestNeighborHamiltonian(
        (zero,), (physical, physical), hamiltonian_id="zero-u1"
    )
    evolved, evidence = tn.abelian_tebd_step(
        canonical,
        hamiltonian,
        0.1,
        maximum_bond_dimension=2,
    )

    assert evidence.valid
    assert jnp.allclose(canonical.to_dense(), state.to_dense(), atol=1e-9)
    assert jnp.allclose(evolved.to_dense(), state.to_dense(), atol=1e-9)
    assert jnp.allclose(
        tn.abelian_mps_one_site_expectation(
            evolved, 0, jnp.diag(jnp.asarray([1.0, -1.0]))
        ),
        0.0,
        atol=1e-9,
    )
