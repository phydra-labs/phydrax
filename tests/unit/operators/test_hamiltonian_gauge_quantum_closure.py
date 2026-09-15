#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.operators.quantum._fermionic_fock import (
    car_evidence,
    CARMonomial,
    CARPolynomial,
    fermion_ladder_matrix,
    fermion_mode_permutation_matrix,
    FermionicFockBasis,
    FermionModeOrder,
)
from phydrax.operators.quantum._gauge_constraints import (
    GaussConstraintNetwork,
    GaussSectorResourcePolicy,
    prepare_exact_physical_sector,
    prepare_sparse_physical_sector,
)
from phydrax.operators.quantum._gauge_link_hilbert import (
    cyclic_group_link_hilbert,
    SU2IrrepTruncatedLinkHilbertSpace,
    TruncatedU1LinkHilbertSpace,
)


def test_canonical_car_and_mode_permutation_signs():
    order = FermionModeOrder(("a", "b", "c"))
    basis = FermionicFockBasis(order)
    annihilation = tuple(
        fermion_ladder_matrix(basis, mode, "annihilate") for mode in order.labels
    )
    identity = jnp.eye(basis.dimension)
    for left, first in enumerate(annihilation):
        for right, second in enumerate(annihilation):
            expected = identity if left == right else jnp.zeros_like(identity)
            assert jnp.allclose(first @ second + second @ first, 0.0)
            assert jnp.allclose(
                first @ jnp.conj(second.T) + jnp.conj(second.T) @ first,
                expected,
            )
    evidence = car_evidence(order)
    target = FermionModeOrder(("c", "b", "a"))
    permutation = fermion_mode_permutation_matrix(order, target)
    source_double = basis.basis_index((1, 0, 1))
    target_double = FermionicFockBasis(target).basis_index((1, 0, 1))
    assert evidence.valid
    assert permutation[target_double, source_double] == -1.0
    assert jnp.allclose(jnp.conj(permutation.T) @ permutation, identity)


def test_car_polynomial_obeys_declared_operator_order():
    order = FermionModeOrder(("left", "right"))
    basis = FermionicFockBasis(order)
    hopping = CARMonomial(order, (("left", "create"), ("right", "annihilate")))
    polynomial = CARPolynomial(order, ((2.0, hopping),))
    expected = 2.0 * (
        fermion_ladder_matrix(basis, "left", "create")
        @ fermion_ladder_matrix(basis, "right", "annihilate")
    )
    assert jnp.allclose(polynomial.dense_matrix(), expected)


def test_finite_and_truncated_link_algebras_have_explicit_cutoff_defects():
    z3 = cyclic_group_link_hilbert(3)
    assert z3.evidence.valid
    assert jnp.allclose(
        z3.left_translation(1) @ z3.right_translation(2),
        z3.right_translation(2) @ z3.left_translation(1),
    )

    u1 = TruncatedU1LinkHilbertSpace(2)
    assert u1.algebra.valid
    assert jnp.allclose(
        u1.electric_field @ u1.link_operator - u1.link_operator @ u1.electric_field,
        u1.link_operator,
    )
    center = jnp.zeros((u1.dimension,), dtype=complex).at[2].set(1.0)
    boundary = jnp.zeros((u1.dimension,), dtype=complex).at[-1].set(1.0)
    assert u1.cutoff_evidence(center).valid
    assert not u1.cutoff_evidence(boundary).valid
    assert u1.cutoff_evidence(boundary).state_defect_norm == 1.0

    su2 = SU2IrrepTruncatedLinkHilbertSpace(1)
    assert su2.dimension == 5
    assert su2.algebra.valid
    assert su2.algebra.link_covariance_residual <= 1e-12
    singlet = jnp.zeros((su2.dimension,), dtype=complex).at[0].set(1.0)
    doublet = jnp.zeros((su2.dimension,), dtype=complex).at[-1].set(1.0)
    assert su2.cutoff_evidence(singlet).boundary_probability == 0.0
    assert su2.cutoff_evidence(doublet).boundary_probability == 1.0


def test_exact_and_sparse_gauss_sectors_have_the_same_dimension():
    u1 = TruncatedU1LinkHilbertSpace(1)
    electric = u1.electric_field[None]
    network = GaussConstraintNetwork(jnp.asarray([[1], [-1]]), (electric,), (electric,))
    policy = GaussSectorResourcePolicy(
        maximum_hilbert_dimension=16,
        maximum_dense_elements=1024,
        maximum_basis_states=16,
    )
    sparse = prepare_sparse_physical_sector(network, resources=policy)
    exact = prepare_exact_physical_sector(network, resources=policy)
    assert sparse.subspace.logical_dimension == 1
    assert exact.subspace.logical_dimension == 1
    assert sparse.evidence.valid
    assert exact.evidence.valid
