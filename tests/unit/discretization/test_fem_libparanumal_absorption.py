#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _mesh():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = jnp.asarray([[0, 1, 3], [1, 2, 3]], dtype=jnp.int32)
    return phx.discretization.CellMesh.from_triangles(vertices, cells)


def test_exact_diagonal_matches_sparse_diffusion():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(), phx.discretization.FiniteElementFieldSpec("u", element)
    ).prepare()
    form = phx.equations.FiniteElementForm(
        "poisson", "u", (phx.equations.DiffusionAction("u"),)
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    diagonal = compiled.exact_diagonal()
    matrix = compiled.to_scipy_csr().toarray()

    assert diagonal.method == "workset"
    assert jnp.allclose(diagonal.diagonal, jnp.diag(jnp.asarray(matrix)))


def test_generic_p_transfer_preserves_pairing():
    coarse = phx.discretization.lagrange_element("triangle", 1)
    fine = phx.discretization.lagrange_element("triangle", 3)
    transfer = phx.discretization.fem.finite_element_p_transfer(coarse, fine)
    coarse_value = jnp.asarray([1.0, 2.0, 3.0])
    fine_dual = jnp.arange(float(fine.local_dof_count))

    prolonged = transfer.apply("primal-prolongation", coarse_value)
    pulled = transfer.apply("dual-pullback", fine_dual)

    assert jnp.allclose(jnp.vdot(prolonged, fine_dual), jnp.vdot(coarse_value, pulled))


def test_p_degree_coarsening_policies_are_deterministic():
    assert phx.discretization.fem.FiniteElementPMultigridPolicy(
        "all-degrees"
    ).degree_sequence("hexahedron", 5) == (5, 4, 3, 2, 1)
    assert phx.discretization.fem.FiniteElementPMultigridPolicy(
        "half-degrees"
    ).degree_sequence("hexahedron", 8) == (8, 4, 2, 1)
    assert phx.discretization.fem.FiniteElementPMultigridPolicy(
        "half-dofs"
    ).degree_sequence("hexahedron", 8) == (8, 6, 4, 2, 1)


def test_collocated_tensor_mass_path_is_identity_for_unit_data():
    derivative = jnp.zeros((2, 2))
    metric = jnp.ones((1, 2, 2, 3))
    mass = jnp.ones((1, 2, 2))
    gathers = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
    operator = phx.equations.fem.CollocatedTensorProductOperator(
        derivative, metric, mass, gathers, 4
    )
    value = jnp.arange(1.0, 5.0)

    assert jnp.allclose(operator.mv(value), value)
    assert jnp.allclose(operator.transpose_mv(value), value)


def test_dg_trace_staging_builds_conservative_jet():
    plus = jnp.asarray([[1.0, 2.0]])
    minus = jnp.asarray([[3.0, 4.0]])
    basis = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    gradients = jnp.asarray([[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]])
    normal = jnp.asarray([[[1.0, 0.0], [1.0, 0.0]]])
    measure = jnp.ones((1, 2))

    staged = phx.equations.fem.DGTraceBatch(
        plus, minus, basis, basis, gradients, gradients, normal, measure
    )

    assert jnp.allclose(staged.jet.jump, jnp.asarray([[-2.0, -2.0]]))
    assert jnp.allclose(staged.jet.average, jnp.asarray([[2.0, 3.0]]))


def test_quadrature_policy_requires_nonpolynomial_evidence():
    policy = phx.equations.fem.QuadratureAccuracyPolicy("exact-polynomial")
    assert (
        policy.resolve_degree(2, 2, coefficient_order=1, kernel_polynomial_degree=2) == 7
    )

    with pytest.raises(ValueError, match="requires declared"):
        policy.resolve_degree(2, 2, coefficient_order=None)


def test_one_ring_patch_weights_form_partition_of_unity():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(), phx.discretization.FiniteElementFieldSpec("u", element)
    ).prepare()
    plan = phx.discretization.fem.one_ring_patch_plan(discretization, "u")
    width = plan.gathers.shape[1]
    local_inverse = jnp.broadcast_to(
        jnp.eye(width), (plan.gathers.shape[0], width, width)
    )
    preconditioner = phx.discretization.fem.FiniteElementPatchPreconditioner(
        plan, local_inverse, discretization.field_spaces[0].vector_space
    )
    value = jnp.arange(1.0, 5.0)

    assert jnp.allclose(preconditioner.apply(value), value)


def test_low_order_auxiliary_identity_path():
    high = phx.linalg.ArraySpace((3,))
    low = phx.linalg.ArraySpace((3,))
    identity = phx.linalg.IdentityLinearOperator(high)
    plan = phx.discretization.fem.LowOrderAuxiliaryOperatorPlan(
        identity, identity, jnp.ones((3,))
    )
    preconditioner = phx.discretization.fem.LowOrderAuxiliaryPreconditioner(
        plan, phx.linalg.IdentityPreconditioner(low)
    )
    value = jnp.arange(1.0, 4.0)

    assert jnp.allclose(preconditioner.apply(value), value)
