#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.tensor_network._gauge_models import (
    AbelianFusionBasis,
    build_gauge_invariant_mps,
    build_gauge_invariant_peps,
    build_gauge_projector_mpo,
    gauge_invariant_peps_tensor,
    SU2FusionBasis,
)


def test_fusion_mps_and_projector_mpo_match_dense_sector():
    basis = AbelianFusionBasis(((-1, 0, 1), (-1, 0, 1)))
    state = build_gauge_invariant_mps(basis)
    projector = build_gauge_projector_mpo(basis)
    dense_state = state.state.to_dense()
    expected_state = (
        jnp.zeros((9,), dtype="complex128")
        .at[jnp.asarray((2, 4, 6))]
        .set(1.0 / jnp.sqrt(3.0))
    )
    assert state.evidence.valid
    assert projector.evidence.valid
    assert jnp.allclose(dense_state, expected_state)
    assert jnp.allclose(projector.operator.to_dense(), basis.dense_projector())
    assert jnp.allclose(projector.operator.to_dense() @ dense_state, dense_state)
    structural = build_gauge_projector_mpo(
        basis,
        maximum_reference_elements=1,
    )
    assert structural.evidence.valid
    assert structural.evidence.structurally_proven
    assert structural.evidence.verification_method == "deterministic-charge-flow"


def test_su2_fusion_basis_constructs_exact_singlet_multiplicities():
    basis = SU2FusionBasis((1, 1, 1, 1), 0)
    state = basis.state(jnp.ones((len(basis.fusion_paths),)))
    assert len(basis.fusion_paths) == 2
    assert jnp.allclose(state.norm(), 1.0)
    assert all(path[-1] == 0 for path in basis.fusion_paths)


def test_peps_local_tensors_have_only_zero_gauss_residual_entries():
    tensor = gauge_invariant_peps_tensor(
        (-1, 0, 1),
        (-1, 0, 1),
        (-1, 0, 1),
        (-1, 0, 1),
        (-1, 0, 1),
    )
    assert tensor.evidence.valid
    assert jnp.all(tensor.evidence.constraint_values == 0)
    assert tensor.evidence.allowed_entry_count == jnp.count_nonzero(tensor.tensor)

    state = build_gauge_invariant_peps(2, 2, (-1, 0, 1), (-1, 0, 1))
    assert state.state.rows == 2
    assert state.state.columns == 2
    assert state.evidence.valid
    assert jnp.all(state.evidence.local_constraint_residuals == 0)
