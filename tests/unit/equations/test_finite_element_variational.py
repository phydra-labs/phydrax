#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe

import phydrax as phx
from phydrax.discretization.fem._high_order import (
    ReferenceNodalFamily,
    SumFactorizationPlan,
)
from phydrax.discretization.fem._precision import FiniteElementPrecisionPolicy
from phydrax.discretization.fem._reference_operator import PreparedFiniteElementReference
from phydrax.equations._finite_element_variational import _action_payload
from phydrax.equations._variational import _rule_id
from phydrax.equations.fem._execution import TensorProductPartialAssemblyOperator
from phydrax.equations.fem._operators import FiniteElementMetricData


def _triangle_discretization(degree=1):
    mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
    )
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", degree)
        ),
    ).prepare()


def _quadrilateral_discretization(degree=2):
    mesh = phx.discretization.CellMesh(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        (
            phx.discretization.CellBlock(
                "quads",
                "quadrilateral",
                jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32),
            ),
        ),
    )
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("quadrilateral", degree)
        ),
    ).prepare()


def _compiled(discretization, action, realization="matrix_free", local_kernel="auto"):
    return phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(action.action_id, "u", (action,)),
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization=realization,
            local_kernel=local_kernel,
        ),
    )


def test_scalar_action_payload_remains_legacy_exact():
    scalar_payload = _action_payload(phx.equations.DiffusionAction("u"))
    tensor_payload = _action_payload(phx.equations.TensorDiffusionAction("u"))

    assert tuple(scalar_payload) == (
        "kind",
        "action_id",
        "output_fields",
        "coefficient_id",
        "input_fields",
        "domain_id",
        "rules",
        "penalty_policy",
        "boundary",
    )
    assert "tensor_axes" not in scalar_payload
    assert tensor_payload["tensor_axes"] == ["flux", "gradient"]
    assert "operator_properties" in tensor_payload


def test_p1_tensor_diffusion_matches_direct_element_reference_and_axis_order():
    discretization = _triangle_discretization()
    diffusivity = jnp.asarray([[2.0, 0.3], [0.1, 1.4]])
    state = jnp.asarray([0.2, -0.4, 1.1])
    action = phx.equations.TensorDiffusionAction("u", diffusivity)
    reversed_action = phx.equations.TensorDiffusionAction(
        "u",
        diffusivity.T,
        tensor_axes=("gradient", "flux"),
        action_id="reversed-tensor-axes",
    )

    residual = _compiled(discretization, action).full_residual(state, None)
    reversed_residual = _compiled(discretization, reversed_action).full_residual(
        state, None
    )
    gradients = jnp.asarray([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    element_matrix = 0.5 * oe.contract("id,de,je->ij", gradients, diffusivity, gradients)

    assert jnp.allclose(residual, element_matrix @ state, atol=2.0e-6)
    assert jnp.allclose(reversed_residual, residual, atol=2.0e-6)


def test_callable_constant_tensor_is_coerced_before_point_axes():
    discretization = _triangle_discretization()
    tensor = jnp.asarray([[1.6, 0.2], [-0.1, 0.7]])
    callable_coefficient = phx.equations.coefficient(
        lambda points, args: tensor,
        coefficient_id="constant-callable-tensor",
    )
    state = jnp.asarray([-0.3, 0.4, 0.9])
    constant = _compiled(
        discretization,
        phx.equations.TensorDiffusionAction("u", tensor, action_id="constant-tensor"),
    )
    callable_problem = _compiled(
        discretization,
        phx.equations.TensorDiffusionAction(
            "u", callable_coefficient, action_id="callable-tensor"
        ),
    )

    assert jnp.allclose(
        callable_problem.full_residual(state, None),
        constant.full_residual(state, None),
        atol=2.0e-6,
    )


def test_scalar_isotropic_tensor_action_preserves_scalar_diffusion_behavior():
    discretization = _triangle_discretization(degree=2)
    state = jnp.linspace(-0.3, 0.9, discretization.dof_maps[0].global_dof_count)
    scalar = _compiled(
        discretization,
        phx.equations.DiffusionAction("u", 2.5, action_id="scalar-diffusion"),
    )
    isotropic = _compiled(
        discretization,
        phx.equations.TensorDiffusionAction(
            "u", 2.5, action_id="isotropic-tensor-diffusion"
        ),
    )

    assert jnp.allclose(
        isotropic.full_residual(state, None),
        scalar.full_residual(state, None),
        atol=2.0e-6,
    )


def test_cell_dof_and_quadrature_tensor_coefficients_agree_for_constant_data():
    discretization = _triangle_discretization(degree=2)
    state = jnp.linspace(-0.7, 0.8, discretization.dof_maps[0].global_dof_count)
    tensor = jnp.asarray([[1.8, 0.25], [0.25, 0.9]])
    support_id = discretization.support.support_id
    entity_set_id = discretization.cell_domain.entity_set_id
    field_space_id = discretization.field_spaces[0].field_space_id
    rule = phx.integration.ReferenceTriangleRule()
    quadrature_count = phx.integration.reference_rule_data(rule).points.shape[0]
    coefficients = (
        phx.equations.coefficient(
            tensor[None],
            location="cell",
            support_id=support_id,
            entity_set_id=entity_set_id,
        ),
        phx.equations.coefficient(
            jnp.broadcast_to(
                tensor,
                (discretization.dof_maps[0].global_dof_count, 2, 2),
            ),
            location="dof",
            support_id=support_id,
            field_space_id=field_space_id,
        ),
        phx.equations.coefficient(
            jnp.broadcast_to(tensor, (1, quadrature_count, 2, 2)),
            location="quadrature",
            support_id=support_id,
            entity_set_id=entity_set_id,
            rule_id=_rule_id(rule),
        ),
    )
    residuals = []
    for index, coefficient in enumerate(coefficients):
        action = phx.equations.TensorDiffusionAction(
            "u",
            coefficient,
            action_id=f"located-tensor-{index}",
            rules={"triangles": rule},
        )
        matrix_free = _compiled(discretization, action)
        residuals.append(matrix_free.full_residual(state, None))
        sparse = _compiled(discretization, action, realization="sparse")
        assert jnp.allclose(
            sparse.affine_operator().mv(state), residuals[-1], atol=2.0e-6
        )

    assert jnp.allclose(residuals[0], residuals[1], atol=2.0e-6)
    assert jnp.allclose(residuals[0], residuals[2], atol=2.0e-6)


def test_high_order_tensor_action_sparse_matrix_free_jvp_vjp_and_properties():
    discretization = _quadrilateral_discretization(degree=3)
    properties = phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={"self_adjoint": "asserted", "positive_semidefinite": "asserted"},
    )
    action = phx.equations.TensorDiffusionAction(
        "u",
        jnp.asarray([[2.0, 0.4], [0.4, 1.1]]),
        properties=properties,
        action_id="high-order-tensor",
    )
    matrix_free = _compiled(discretization, action, local_kernel="sum_factorized")
    sparse = _compiled(discretization, action, realization="sparse")
    state = jnp.linspace(-0.5, 0.7, discretization.dof_maps[0].global_dof_count)
    direction = jnp.linspace(0.8, -0.2, state.size)
    cotangent = jnp.linspace(-0.4, 0.6, state.size)
    sparse_operator = sparse.affine_operator()

    expected = matrix_free.full_residual(state, None)
    _, tangent = jax.jvp(
        lambda value: matrix_free.full_residual(value, None),
        (state,),
        (direction,),
    )
    _, pullback = jax.vjp(lambda value: matrix_free.full_residual(value, None), state)

    assert jnp.allclose(sparse_operator.mv(state), expected, atol=2.0e-6)
    assert jnp.allclose(tangent, sparse_operator.mv(direction), atol=2.0e-6)
    assert jnp.allclose(
        pullback(cotangent)[0], sparse_operator.transpose_mv(cotangent), atol=2.0e-6
    )
    assert sparse_operator.properties.self_adjoint
    assert sparse_operator.properties.positive_semidefinite
    assert sparse.to_scipy_csr().shape == (state.size, state.size)


def test_tensor_product_partial_diffusion_has_exact_sparse_and_transpose_lowering():
    family = ReferenceNodalFamily("quadrilateral", 2)
    axis_rule = phx.integration.GaussLobattoLegendreRule(3)
    reference = PreparedFiniteElementReference(
        family.finite_element(),
        phx.integration.ReferenceQuadrilateralRule(axis_rule),
        (phx.integration.ReferenceIntervalRule(axis_rule),) * 4,
        (
            "interpolate",
            "interpolate_transpose",
            "gradient",
            "gradient_transpose",
            "trace",
            "trace_transpose",
        ),
        FiniteElementPrecisionPolicy(),
        tensor_family=family,
    )
    coordinate_element = phx.discretization.lagrange_element("quadrilateral", 1)
    basis, gradients = coordinate_element.tabulate(reference.volume_rule.points)
    metric = FiniteElementMetricData(
        basis,
        gradients,
        coordinate_element.reference_nodes[None],
        reference.weights,
    )
    plan = SumFactorizationPlan(reference.tensor_tabulation)
    physical_tensor = jnp.asarray([[1.7, 0.35], [-0.2, 0.8]])
    reference_tensor = oe.contract(
        "cqrd,de,cqse,cq->cqrs",
        metric.inverse_jacobian,
        physical_tensor,
        metric.inverse_jacobian,
        metric.weighted_measure,
    ).reshape((1,) + plan.tabulation.evaluation_shape + (2, 2))
    width = int(np.prod(family.nodal_shape))
    operator = TensorProductPartialAssemblyOperator(
        plan,
        reference_tensor,
        jnp.arange(width, dtype=jnp.int32)[None],
        width,
        action_kind="diffusion",
        properties=phx.linalg.OperatorProperties(),
    )
    state = jnp.linspace(-0.6, 1.0, width)
    cotangent = jnp.linspace(0.7, -0.3, width)
    sparse = operator.as_sparse_coordinate()
    _, pullback = jax.vjp(operator.mv, state)

    assert jnp.allclose(operator.mv(state), sparse.mv(state), atol=2.0e-6)
    assert jnp.allclose(
        operator.transpose_mv(cotangent), sparse.transpose_mv(cotangent), atol=2.0e-6
    )
    assert jnp.allclose(
        pullback(cotangent)[0], operator.transpose_mv(cotangent), atol=2.0e-6
    )
