import jax.numpy as jnp
import pytest

from phydrax.tensor_network._abelian import (
    AbelianGroup,
    AbelianLeg,
    AbelianTensor,
    AbelianTensorLayout,
)
from phydrax.tensor_network._abelian_evolution import (
    abelian_product_mps,
    apply_abelian_two_site_gate,
)
from phydrax.tensor_network._abelian_open import (
    AbelianKrausOperator,
    AbelianLPDO,
    apply_charge_covariant_kraus,
    ChargeCovariantKrausMap,
)
from phydrax.tensor_network._abelian_plan import (
    contract_abelian_tensors,
    prepare_abelian_contraction,
)
from phydrax.tensor_network._fermion import (
    fermionic_swap_gate,
    FermionModeOrder,
    jordan_wigner_monomial_mpo,
)
from phydrax.tensor_network._graded import (
    contract_graded_closed_network,
    FermionGrading,
    GradedLeg,
    GradedTensor,
)
from phydrax.tensor_network._representation_category import (
    FusionChannel,
    FusionTree,
    Irrep,
    RepresentationCategory,
)
from phydrax.tensor_network._su2 import (
    su2_clebsch_gordan,
    su2_finite_dmrg,
    su2_fusion,
    su2_mps_dmrg,
    su2_pentagon_residual,
    su2_recoupling_matrix,
    SU2InvariantOperator,
    SU2MatrixProductOperator,
    SU2MatrixProductState,
    truncate_su2_multiplets,
)


def test_abelian_unreachable_support_and_prepared_block_contraction():
    group = AbelianGroup((None,))
    positive = AbelianLeg(group, ((1,),), (2,), orientation=1)
    unreachable = AbelianTensorLayout((positive, positive), total_charge=(0,))
    structural_zero = AbelianTensor(unreachable, ())
    assert structural_zero.blocks == ()
    assert jnp.all(structural_zero.to_dense() == 0)

    incoming = AbelianLeg(group, ((0,),), (2,), orientation=1)
    outgoing = incoming.dual()
    matrix_layout = AbelianTensorLayout((incoming, outgoing))
    vector_layout = AbelianTensorLayout((incoming,))
    matrix = AbelianTensor(matrix_layout, (jnp.asarray(((1.0, 2.0), (3.0, 4.0))),))
    vector = AbelianTensor(vector_layout, (jnp.asarray((5.0, 6.0)),))
    plan = prepare_abelian_contraction(matrix_layout, vector_layout, (1,), (0,))
    result = contract_abelian_tensors(plan, matrix, vector)
    assert isinstance(result, AbelianTensor)
    assert jnp.allclose(result.blocks[0], jnp.asarray((17.0, 39.0)))


def test_abelian_gate_reports_sector_capacity_overflow():
    group = AbelianGroup((2,))
    physical = AbelianLeg(group, ((0,),), (2,), orientation=1)
    state = abelian_product_mps(
        (jnp.asarray((1.0, 0.0)), jnp.asarray((1.0, 0.0))),
        (physical, physical),
        (0, 0),
    )
    root_two = jnp.sqrt(2.0)
    gate_matrix = jnp.asarray(
        (
            (1.0 / root_two, 0.0, 0.0, 1.0 / root_two),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (1.0 / root_two, 0.0, 0.0, -1.0 / root_two),
        ),
        dtype=jnp.complex128,
    )
    _, evidence = apply_abelian_two_site_gate(
        state,
        0,
        gate_matrix.reshape((2, 2, 2, 2)),
        maximum_bond_dimension=1,
        normalize=False,
    )
    assert evidence.available_rank == 2
    assert jnp.allclose(evidence.overflow_discarded_weight, 0.5)
    assert jnp.allclose(evidence.discarded_weight, 0.5)


def test_charge_covariant_kraus_lpdo_amplitude_damping_oracle():
    group = AbelianGroup((None,))
    physical = AbelianLeg(group, ((0,), (1,)), (1, 1), orientation=1)
    gamma = 0.25
    no_jump = AbelianKrausOperator(
        physical,
        physical,
        (0,),
        ((0, 0), (1, 1)),
        (jnp.ones((1, 1)), jnp.sqrt(1.0 - gamma) * jnp.ones((1, 1))),
    )
    jump = AbelianKrausOperator(
        physical,
        physical,
        (-1,),
        ((0, 1),),
        (jnp.sqrt(gamma) * jnp.ones((1, 1)),),
    )
    channel = ChargeCovariantKrausMap((no_jump, jump))
    excited = AbelianLPDO(
        physical,
        (1, 1),
        (jnp.zeros((1, 1)), jnp.ones((1, 1))),
    )
    result, evidence = apply_charge_covariant_kraus(
        channel,
        excited,
        maximum_purification_dimension=2,
    )
    populations = tuple(jnp.real(block[0, 0]) for block in result.density_blocks())
    assert jnp.allclose(jnp.asarray(populations), jnp.asarray((gamma, 1.0 - gamma)))
    assert jnp.allclose(evidence.output_trace, 1.0)
    assert evidence.valid


def test_grading_is_explicit_homomorphism_and_odd_zn_rejects_parity():
    odd = AbelianGroup((3,))
    with pytest.raises(ValueError, match="odd-order"):
        FermionGrading(odd, (1,))
    group = AbelianGroup((None, 4))
    grading = FermionGrading(group, (1, 1))
    charges = ((0, 0), (1, 0), (0, 1), (2, 3))
    assert grading.verify_homomorphism(charges)
    assert grading.parity((3, 1)) == 0


def test_fswap_and_jordan_wigner_anticommutation_dense_oracles():
    fswap = fermionic_swap_gate().reshape((4, 4))
    assert fswap[3, 3] == -1
    assert jnp.allclose(fswap @ fswap, jnp.eye(4))

    order = FermionModeOrder(("a", "b"))
    annihilate_a = jordan_wigner_monomial_mpo(order, (("a", "annihilate"),)).to_dense()
    annihilate_b = jordan_wigner_monomial_mpo(order, (("b", "annihilate"),)).to_dense()
    assert jnp.allclose(
        annihilate_a @ annihilate_b + annihilate_b @ annihilate_a,
        jnp.zeros((4, 4)),
        atol=1e-12,
    )


def test_graded_closed_contraction_is_independent_of_input_path_order():
    group = AbelianGroup((2,))
    grading = FermionGrading(group, (1,))
    outward = AbelianLeg(group, ((0,), (1,)), (1, 1), orientation=1)
    inward = outward.dual()
    left_layout = AbelianTensorLayout((outward,), total_charge=(1,))
    right_layout = AbelianTensorLayout((inward,), total_charge=(1,))
    left = GradedTensor(
        AbelianTensor(left_layout, (jnp.asarray((2.0,)),)),
        (GradedLeg(outward, grading, mode_label="m"),),
    )
    right = GradedTensor(
        AbelianTensor(right_layout, (jnp.asarray((3.0,)),)),
        (GradedLeg(inward, grading, mode_label="m"),),
    )
    first = contract_graded_closed_network((left, right), ("m",))
    second = contract_graded_closed_network((right, left), ("m",))
    assert jnp.allclose(first, second)
    assert jnp.allclose(first, 6.0)


def test_representation_category_fusion_multiplicity_and_tree():
    irreps = (Irrep("1", 1, dual_label="1"), Irrep("x", 1, dual_label="x"))
    rules = (
        ("1", "1", (("1", 1),)),
        ("1", "x", (("x", 1),)),
        ("x", "1", (("x", 1),)),
        ("x", "x", (("1", 1),)),
    )
    category = RepresentationCategory(irreps, rules, unit_label="1")
    first = FusionChannel(category, "x", "x", "1")
    second = FusionChannel(category, "1", "x", "x")
    tree = FusionTree(category, ("x", "x", "x"), (first, second))
    assert tree.output == "x"
    assert category.multiplicity("x", "x", "1") == 1


def test_su2_cg_recoupling_pentagon_and_multiplet_truncation():
    singlet = su2_clebsch_gordan(1, 1, 0)[:, :, 0]
    triplet = su2_clebsch_gordan(1, 1, 2)
    assert jnp.allclose(jnp.sum(jnp.abs(singlet) ** 2), 1.0)
    assert jnp.allclose(jnp.sum(jnp.abs(triplet) ** 2, axis=(0, 1)), jnp.ones(3))
    assert su2_fusion(1, 1) == (0, 2)

    _, _, recoupling = su2_recoupling_matrix(1, 1, 1, 1)
    assert jnp.allclose(recoupling @ recoupling.T, jnp.eye(2), atol=1e-10)
    assert su2_pentagon_residual(1, 1, 1, 1, 0) < 1e-10

    masks, evidence = truncate_su2_multiplets(
        (0, 2),
        (jnp.asarray((0.8, 0.2)), jnp.asarray((0.7,))),
        maximum_dimension=3,
        protected_twice_spins=(2,),
    )
    assert tuple(mask.tolist() for mask in masks) == ([False, False], [True])
    assert evidence.retained_multiplet_dimension == 3
    assert evidence.protected_multiplets_satisfied


def test_su2_protected_sector_dmrg_singlet_triplet_oracle():
    operator = SU2InvariantOperator(
        (0, 2),
        (jnp.asarray(((-0.75,),)), jnp.asarray(((0.25,),))),
    )
    singlet, singlet_evidence = su2_finite_dmrg(
        operator, protected_twice_spin=0, maximum_sweeps=2
    )
    triplet, triplet_evidence = su2_finite_dmrg(
        operator, protected_twice_spin=2, maximum_sweeps=2
    )
    assert singlet.twice_spin == 0
    assert triplet.twice_spin == 2
    assert jnp.allclose(singlet_evidence.energy, -0.75)
    mps = SU2MatrixProductState((1, 1), 0, jnp.asarray((1.0,)))
    mpo = SU2MatrixProductOperator((1, 1), operator)
    optimized, route_evidence = su2_mps_dmrg(mps, mpo, maximum_sweeps=2)
    assert optimized.total_twice_spin == 0
    assert jnp.allclose(route_evidence.energy, -0.75)
    assert jnp.allclose(triplet_evidence.energy, 0.25)
    assert singlet_evidence.converged & triplet_evidence.converged
