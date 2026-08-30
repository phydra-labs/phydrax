#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.discretization.fem._fast_diagonalization import (
    TensorFastDiagonalizationBuilder,
)
from phydrax.discretization.fem._high_order import ReferenceNodalFamily
from phydrax.discretization.fem._low_order_auxiliary import (
    low_order_auxiliary_preconditioner_builder,
    LowOrderAuxiliaryOperatorPlan,
)
from phydrax.discretization.fem._multigrid import (
    finite_element_p_transfer,
    quadrilateral_p_transfer,
)
from phydrax.discretization.fem._p_multigrid import (
    _local_dof_count,
    finite_element_p_multigrid_plan,
    FiniteElementPMultigridPolicy,
)
from phydrax.discretization.fem._patch_preconditioning import (
    FiniteElementPatchPlan,
    FiniteElementPatchPreconditionerBuilder,
)
from phydrax.linalg import (
    ArraySpace,
    DenseLinearOperator,
    DiagonalLinearOperator,
    factorize,
    GalerkinHierarchyBuilder,
    IdentityLinearOperator,
    IdentityPreconditioner,
    MaterializationPolicy,
    materialize,
    MultigridHierarchyBuilder,
    OperatorProperties,
)


def _positive_properties(*, diagonal: bool = False) -> OperatorProperties:
    return OperatorProperties(
        diagonal=diagonal,
        self_adjoint=True,
        positive_definite=True,
        evidence={
            **({"diagonal": "construction"} if diagonal else {}),
            "self_adjoint": "verified",
            "positive_definite": "verified",
        },
    )


def _dense_endomorphism(matrix, space):
    return DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        properties=_positive_properties(),
    )


def test_p_transfer_keeps_dual_pairing_and_physical_mass_roles_distinct():
    coarse = ReferenceNodalFamily("quadrilateral", 1).finite_element()
    fine = ReferenceNodalFamily("quadrilateral", 2).finite_element()
    coarse_pairing = jnp.diag(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    fine_pairing = jnp.diag(jnp.linspace(1.0, 2.0, fine.local_dof_count))
    coarse_mass = jnp.diag(jnp.asarray([2.0, 3.0, 5.0, 7.0]))
    fine_mass = jnp.diag(jnp.linspace(2.0, 4.0, fine.local_dof_count))
    transfer = finite_element_p_transfer(
        coarse,
        fine,
        coarse_pairing=coarse_pairing,
        fine_pairing=fine_pairing,
        coarse_mass=coarse_mass,
        fine_mass=fine_mass,
    )
    coarse_value = jnp.arange(1.0, coarse.local_dof_count + 1.0)
    fine_value = jnp.linspace(-1.0, 1.0, fine.local_dof_count)
    fine_dual = jnp.linspace(0.5, 2.5, fine.local_dof_count)

    prolonged = transfer.apply("primal-prolongation", coarse_value)
    pulled = transfer.apply("dual-pullback", fine_dual)
    paired = transfer.apply("pairing-adjoint", fine_value)
    projected = transfer.apply("mass-projection", fine_value)

    assert jnp.allclose(jnp.vdot(prolonged, fine_dual), jnp.vdot(coarse_value, pulled))
    assert jnp.allclose(
        jnp.vdot(prolonged, fine_pairing @ fine_value),
        jnp.vdot(coarse_value, coarse_pairing @ paired),
    )
    assert jnp.allclose(
        coarse_mass @ projected,
        transfer.dual_pullback @ (fine_mass @ fine_value),
    )
    assert not jnp.allclose(transfer.pairing_adjoint, transfer.mass_projection)


def test_anisotropic_p_transfer_accepts_nested_axes_and_rejects_axis_coarsening():
    coarse = ReferenceNodalFamily("quadrilateral", (2, 3))
    fine = ReferenceNodalFamily("quadrilateral", (2, 5))
    transfer = quadrilateral_p_transfer(coarse, fine)

    assert transfer.primal_prolongation.shape == (18, 12)
    with pytest.raises(ValueError, match="nested axis orders"):
        quadrilateral_p_transfer(
            coarse,
            ReferenceNodalFamily("quadrilateral", (1, 5)),
        )


def test_p_multigrid_selects_direct_or_galerkin_coarse_operators():
    fine_space = ArraySpace((4,))
    coarse_space = ArraySpace((2,))
    fine_matrix = jnp.diag(jnp.asarray([2.0, 3.0, 4.0, 5.0]))
    fine_operator = _dense_endomorphism(fine_matrix, fine_space)
    direct_matrix = jnp.diag(jnp.asarray([11.0, 13.0]))
    direct_operator = _dense_endomorphism(direct_matrix, coarse_space)
    prolongation_matrix = jnp.asarray([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 1.0]])
    prolongation = DenseLinearOperator(
        prolongation_matrix,
        source=coarse_space,
        target=fine_space,
    )
    restriction = DenseLinearOperator(
        prolongation_matrix.T,
        source=fine_space,
        target=coarse_space,
    )
    smoothers = (
        IdentityPreconditioner(fine_space),
        IdentityPreconditioner(coarse_space),
    )
    direct = finite_element_p_multigrid_plan(
        "quadrilateral",
        2,
        (fine_operator, direct_operator),
        smoothers,
        (restriction,),
        (prolongation,),
        coarse_operator_source="direct",
    )
    galerkin = finite_element_p_multigrid_plan(
        "quadrilateral",
        2,
        (fine_operator,),
        smoothers,
        (restriction,),
        (prolongation,),
        coarse_operator_source="galerkin",
    )

    assert isinstance(direct.hierarchy_builder, MultigridHierarchyBuilder)
    assert jnp.allclose(
        direct.hierarchy_builder.levels[-1].operator.matrix, direct_matrix
    )
    assert isinstance(galerkin.hierarchy_builder, GalerkinHierarchyBuilder)
    prepared = galerkin.hierarchy_builder.prepare(
        fine_operator,
        materialization=MaterializationPolicy(),
    )
    coarse = materialize(
        prepared.hierarchy.levels[-1].operator,
        MaterializationPolicy(),
    )
    assert jnp.allclose(
        coarse,
        prolongation_matrix.T @ fine_matrix @ prolongation_matrix,
    )


def test_tensor_fast_diagonalization_matches_dense_separable_solve():
    axis = ArraySpace((2,))
    mass_x_values = jnp.asarray([2.0, 3.0])
    mass_y_values = jnp.asarray([5.0, 7.0])
    stiffness_x_values = jnp.asarray([[3.0, -1.0], [-1.0, 2.0]])
    stiffness_y_values = jnp.asarray([[4.0, -1.0], [-1.0, 3.0]])
    mass_x = DiagonalLinearOperator(
        mass_x_values,
        space=axis,
        properties=_positive_properties(diagonal=True),
    )
    mass_y = DiagonalLinearOperator(
        mass_y_values,
        space=axis,
        properties=_positive_properties(diagonal=True),
    )
    stiffness_x = _dense_endomorphism(stiffness_x_values, axis)
    stiffness_y = _dense_endomorphism(stiffness_y_values, axis)
    reaction = 0.75
    diffusion = (1.25, 0.5)
    physical_matrix = (
        reaction * jnp.kron(jnp.diag(mass_x_values), jnp.diag(mass_y_values))
        + diffusion[0] * jnp.kron(stiffness_x_values, jnp.diag(mass_y_values))
        + diffusion[1] * jnp.kron(jnp.diag(mass_x_values), stiffness_y_values)
    )
    tensor_space = ArraySpace((4,))
    setup_operator = _dense_endomorphism(physical_matrix, tensor_space)
    builder = TensorFastDiagonalizationBuilder(
        (mass_x, mass_y),
        (stiffness_x, stiffness_y),
        diffusion=diffusion,
        reaction=reaction,
    )
    rejected = builder.eligibility_for(setup_operator)
    policy = MaterializationPolicy()
    evidence = builder.eligibility_for(setup_operator, materialization=policy)
    preconditioner = builder.prepare(setup_operator, materialization=policy)
    right_hand_side = jnp.asarray([1.0, -2.0, 3.0, 0.5])
    dense = factorize(setup_operator).solve(right_hand_side)

    assert evidence.eligible
    assert not rejected.eligible
    assert rejected.reasons == ("an explicit materialization policy is required",)
    assert dense.successful
    assert jnp.allclose(preconditioner.apply(right_hand_side), dense.value)


def test_one_ring_schwarz_weights_form_partition_of_unity():
    plan = FiniteElementPatchPlan(
        jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32),
        jnp.ones((2, 2), dtype=bool),
        jnp.asarray([[1.0, 0.5], [0.5, 1.0]]),
        3,
    )
    space = ArraySpace((3,))
    setup_operator = IdentityLinearOperator(space)
    builder = FiniteElementPatchPreconditionerBuilder(plan)
    preconditioner = builder.prepare(
        setup_operator,
        materialization=MaterializationPolicy(),
    )
    value = jnp.asarray([2.0, -1.0, 4.0])

    assert plan.partition_residual == 0.0
    assert jnp.allclose(preconditioner.apply(value), value)


def test_low_order_auxiliary_builder_uses_generic_subspace_correction():
    high = ArraySpace((3,))
    low = ArraySpace((3,))
    high_to_low = DenseLinearOperator(
        jnp.eye(3),
        source=high,
        target=low,
    )
    low_to_high = DenseLinearOperator(
        jnp.eye(3),
        source=low,
        target=high,
    )
    plan = LowOrderAuxiliaryOperatorPlan(
        high_to_low,
        low_to_high,
        jnp.ones((3,)),
    )
    builder = low_order_auxiliary_preconditioner_builder(
        plan,
        IdentityPreconditioner(low),
    )
    preconditioner = builder.prepare(
        IdentityLinearOperator(high),
        materialization=MaterializationPolicy(),
    )
    value = jnp.asarray([1.0, -2.0, 3.0])

    assert jnp.allclose(preconditioner.apply(value), value)


def test_anisotropic_half_dof_hierarchy_is_strict_and_bounded():
    fine_order = (15, 5, 3)
    sequence = FiniteElementPMultigridPolicy("half-dofs").degree_sequence(
        "hexahedron",
        fine_order,
    )
    counts = tuple(_local_dof_count("hexahedron", order) for order in sequence)

    assert sequence[0] == fine_order
    assert sequence[-1] == (1, 1, 1)
    assert all(left > right for left, right in zip(counts, counts[1:]))
    assert len(sequence) <= 1 + sum(value - 1 for value in fine_order)
