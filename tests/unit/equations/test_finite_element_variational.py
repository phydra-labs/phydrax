#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import pytest
from jax import Array

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


def _triangle_discretization(degree: Any = 1) -> Any:
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


def _quadrilateral_discretization(degree: Any = 2) -> Any:
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


def _compiled(
    discretization: Any,
    action: Any,
    realization: Any = "matrix_free",
    local_kernel: Any = "auto",
) -> Any:
    return phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(action.action_id, "u", (action,)),
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization=realization,
            local_kernel=local_kernel,
        ),
    )


def test_finite_element_variational_scenario_1() -> None:
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


def test_finite_element_variational_scenario_2() -> None:
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
    # ty: ignore[invalid-argument-type]
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


# Runtime parameters (diffusivity scale, source amplitude, Dirichlet amplitude).
_RUNTIME_PARAMETERS = jnp.asarray([1.3, 0.7, 0.4])
_RUNTIME_ARGUMENT_IDS = ("diffusivity", "source", "dirichlet-lift")
_RUNTIME_CONFIGURATIONS = (
    ("scalar", "matrix_free"),
    ("tensor", "matrix_free"),
    ("tensor", "sparse"),
)
_RUNTIME_CONFIGURATION_IDS = ("scalar-matrix-free", "tensor-matrix-free", "tensor-sparse")
_CENTRAL_STEP = 1.0e-5
# Central differences of exact dense solves agree with implicit derivatives up
# to O(step^2) truncation and O(eps / step) roundoff.
_RTOL = 1.0e-6
_ATOL = 1.0e-8


def _square_discretization() -> Any:
    vertices = jnp.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ]
    )
    cells = jnp.asarray(
        [[0, 5, 4], [5, 1, 4], [1, 6, 4], [6, 2, 4], [2, 7, 4], [7, 3, 4], [3, 8, 4]]
        + [[8, 0, 4]],
        dtype=jnp.int32,
    )
    return phx.discretization.FiniteElementPlan(
        phx.discretization.CellMesh.from_triangles(vertices, cells),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 2)
        ),
    ).prepare()


# FE coefficient callables receive the execution context; runtime data is user_args.
def _runtime_diffusivity(points: Array, context: Any) -> Array:
    return context.user_args["kappa"] * (1.0 + 0.5 * points[..., 0])


def _runtime_tensor(points: Array, context: Any) -> Array:
    anisotropy = jnp.asarray([[1.0, 0.2], [0.2, 0.8]])
    return _runtime_diffusivity(points, context)[..., None, None] * anisotropy


def _runtime_source(points: Array, context: Any) -> Array:
    return context.user_args["source"] * (1.0 + points[..., 1])


def _runtime_actions(action_kind: str) -> tuple[Any, Any]:
    diffusion = (
        phx.equations.DiffusionAction(
            "u",
            phx.equations.coefficient(
                _runtime_diffusivity, coefficient_id="runtime-diffusivity"
            ),
        )
        if action_kind == "scalar"
        else phx.equations.TensorDiffusionAction(
            "u",
            phx.equations.coefficient(_runtime_tensor, coefficient_id="runtime-tensor"),
        )
    )
    source = phx.equations.SourceAction(
        "u", phx.equations.coefficient(_runtime_source, coefficient_id="runtime-source")
    )
    return diffusion, source


def _runtime_problem(action_kind: str, realization: str) -> tuple[Any, Any, Any]:
    discretization = _square_discretization()
    constraint = phx.discretization.dirichlet_constraint(discretization, "u")
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            f"runtime-{action_kind}", "u", _runtime_actions(action_kind)
        ),
        discretization,
        constraint=constraint,
        dirichlet_values=0.0,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization=realization
        ),
    )
    return discretization, constraint, compiled


def _runtime_context(discretization: Any, constraint: Any, theta: Array) -> Any:
    return phx.equations.FiniteElementExecutionContext(
        discretization.default_runtime,
        lift=constraint.lift(
            lambda points: theta[2] * (points[..., 0] + 2.0 * points[..., 1])
        ),
        user_args={"kappa": theta[0], "source": theta[1]},
    )


def _mathematical_policy() -> Any:
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
    )


def _runtime_solution(action_kind: str, realization: str) -> Callable[[Array], Array]:
    """Full solved state as a function of all runtime parameters."""
    discretization, constraint, compiled = _runtime_problem(action_kind, realization)
    policy = _mathematical_policy()

    def solution(theta: Array) -> Array:
        context = _runtime_context(discretization, constraint, theta)
        system, right_hand_side = compiled.linear_system(context)
        result = phx.linalg.solve(system, right_hand_side, policy=policy)
        return compiled.expand(result.value, context)

    return solution


def _structural_solution() -> Callable[[Array], Array]:
    """Unconstrained sparse tensor-diffusion-reaction solve on assembled storage."""
    diffusion, source = _runtime_actions("tensor")
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "runtime-structural",
            "u",
            (diffusion, phx.equations.MassAction("u", 1.0), source),
        ),
        _square_discretization(),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(realization="sparse"),
    )
    policy = _mathematical_policy()

    def solution(theta: Array) -> Array:
        system, right_hand_side = compiled.linear_system(
            {"kappa": theta[0], "source": theta[1]}
        )
        return phx.linalg.solve(system, right_hand_side, policy=policy).value

    return solution


def _central_difference(
    function: Callable[[Array], Array], theta: Array, direction: Array
) -> np.ndarray:
    plus = np.asarray(function(theta + _CENTRAL_STEP * direction))
    minus = np.asarray(function(theta - _CENTRAL_STEP * direction))
    return (plus - minus) / (2.0 * _CENTRAL_STEP)


def _assert_solution_derivative(
    solution: Callable[[Array], Array], theta: Array, argument: int
) -> None:
    # Tracing under jit refuses host synchronization on runtime data.
    compiled_solution = jax.jit(solution)
    direction = jnp.zeros_like(theta).at[argument].set(1.0)
    reference = _central_difference(compiled_solution, theta, direction)
    _, tangent = jax.jvp(compiled_solution, (theta,), (direction,))
    _, pullback = jax.vjp(compiled_solution, theta)
    cotangent = jnp.linspace(-0.4, 0.9, reference.size)

    assert np.linalg.norm(reference) > 1.0e-2
    np.testing.assert_allclose(tangent, reference, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        pullback(cotangent)[0][argument],
        np.dot(np.asarray(cotangent), reference),
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize("argument", range(3), ids=_RUNTIME_ARGUMENT_IDS)
@pytest.mark.parametrize(
    ("action_kind", "realization"),
    _RUNTIME_CONFIGURATIONS,
    ids=_RUNTIME_CONFIGURATION_IDS,
)
def test_runtime_solve_derivatives_match_central_differences(
    action_kind: str, realization: str, argument: int
) -> None:
    _assert_solution_derivative(
        _runtime_solution(action_kind, realization), _RUNTIME_PARAMETERS, argument
    )


@pytest.mark.parametrize("argument", range(2), ids=_RUNTIME_ARGUMENT_IDS[:2])
def test_structural_sparse_solve_derivatives_match_central_differences(
    argument: int,
) -> None:
    _assert_solution_derivative(_structural_solution(), _RUNTIME_PARAMETERS[:2], argument)


@pytest.mark.parametrize(
    ("action_kind", "realization"),
    _RUNTIME_CONFIGURATIONS,
    ids=_RUNTIME_CONFIGURATION_IDS,
)
def test_runtime_objective_gradient_compiles_under_jit(
    action_kind: str, realization: str
) -> None:
    solution = _runtime_solution(action_kind, realization)

    def objective(theta: Array) -> Array:
        state = solution(theta)
        return jnp.sum(jnp.linspace(0.5, 1.5, state.size) * state**2)

    gradient = jax.jit(jax.grad(objective))(_RUNTIME_PARAMETERS)
    compiled_objective = jax.jit(objective)
    reference = [
        _central_difference(compiled_objective, _RUNTIME_PARAMETERS, direction)
        for direction in jnp.eye(3)
    ]

    np.testing.assert_allclose(gradient, reference, rtol=_RTOL, atol=_ATOL)


def test_adjoint_solve_reproduces_runtime_parameter_gradient() -> None:
    discretization, constraint, compiled = _runtime_problem("tensor", "matrix_free")
    policy = _mathematical_policy()

    def reduced_state(theta: Array) -> Array:
        context = _runtime_context(discretization, constraint, theta)
        system, right_hand_side = compiled.linear_system(context)
        return phx.linalg.solve(system, right_hand_side, policy=policy).value

    def objective(state: Array) -> Array:
        return jnp.sum(jnp.linspace(0.5, 1.5, state.size) * state**2)

    @jax.jit
    def adjoint_gradient(theta: Array) -> Array:
        state = reduced_state(theta)
        multiplier = compiled.solve_adjoint(
            state,
            jax.grad(objective)(state),
            _runtime_context(discretization, constraint, theta),
            linear_policy=policy,
        )
        _, pullback = jax.vjp(
            lambda value: compiled.residual(
                state, _runtime_context(discretization, constraint, value)
            ),
            theta,
        )
        return -pullback(multiplier.value)[0]

    compiled_objective = jax.jit(lambda theta: objective(reduced_state(theta)))
    reference = [
        _central_difference(compiled_objective, _RUNTIME_PARAMETERS, direction)
        for direction in jnp.eye(3)
    ]

    np.testing.assert_allclose(
        adjoint_gradient(_RUNTIME_PARAMETERS), reference, rtol=_RTOL, atol=_ATOL
    )


@pytest.mark.parametrize(
    ("action_kind", "realization"),
    _RUNTIME_CONFIGURATIONS,
    ids=_RUNTIME_CONFIGURATION_IDS,
)
def test_runtime_linearization_transpose_matches_forward_action(
    action_kind: str, realization: str
) -> None:
    discretization, constraint, compiled = _runtime_problem(action_kind, realization)
    context = _runtime_context(discretization, constraint, _RUNTIME_PARAMETERS)
    state = jnp.linspace(-0.3, 0.8, compiled.state_space.size)
    direction = jnp.cos(jnp.arange(state.size, dtype=jnp.float64))
    covector = jnp.linspace(0.9, -0.6, state.size)
    operator = compiled.linearization_operator(state, context)
    reference = _central_difference(
        lambda value: compiled.residual(value, context), state, direction
    )

    np.testing.assert_allclose(operator.mv(direction), reference, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        jnp.dot(covector, operator.mv(direction)),
        jnp.dot(direction, operator.transpose_mv(covector)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
