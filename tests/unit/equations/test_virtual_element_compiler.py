#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


def _space(degree: Any = 1) -> Any:
    coordinates = jnp.asarray(
        (
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.0, 0.5),
            (0.5, 0.5),
            (1.0, 0.5),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 1.0),
        )
    )
    mesh = phx.discretization.CellMesh.from_polygons(
        coordinates,
        # ty: ignore[invalid-argument-type]
        (
            (0, 1, 4, 3),
            (1, 2, 5, 4),
            (3, 4, 7, 6),
            (4, 5, 8, 7),
        ),
    )
    field = phx.discretization.VirtualElementFieldSpec(
        "u", phx.discretization.conforming_h1_virtual_element(degree)
    )
    return phx.discretization.VirtualElementPlan(mesh, field).prepare()


def _single_cell_space(factory: Any) -> Any:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    # ty: ignore[invalid-argument-type]
    mesh = phx.discretization.CellMesh.from_polygons(coordinates, ((0, 1, 2, 3),))
    field = phx.discretization.VirtualElementFieldSpec("u", factory(1))
    return phx.discretization.VirtualElementPlan(mesh, field).prepare()


def _vector_polynomial_state(space: Any, differential_kind: Any) -> Any:
    projection = space.default_runtime.projections[0]
    geometry = space.default_runtime.geometries[0]
    exponents = [tuple(row) for row in projection.basis.exponents]
    constant = exponents.index((0, 0))
    x_term = exponents.index((1, 0))
    y_term = exponents.index((0, 1))
    polynomial_count = projection.basis.feature_count
    coefficients = jnp.zeros((2 * polynomial_count,))
    centroid = geometry.centroids[0]
    scale = geometry.characteristic_lengths[0]
    if differential_kind == "divergence":
        coefficients = coefficients.at[constant].set(centroid[0])
        coefficients = coefficients.at[x_term].set(scale)
        coefficients = coefficients.at[polynomial_count + constant].set(centroid[1])
        coefficients = coefficients.at[polynomial_count + y_term].set(scale)
    else:
        coefficients = coefficients.at[constant].set(-centroid[1])
        coefficients = coefficients.at[y_term].set(-scale)
        coefficients = coefficients.at[polynomial_count + constant].set(centroid[0])
        coefficients = coefficients.at[polynomial_count + x_term].set(scale)
    local = projection.dof_matrix[0] @ coefficients
    routes = space.dof_map.cell_dofs[0][0]
    orientation = space.dof_map.orientations[0][0]
    state = jnp.zeros((space.dof_map.global_dof_count,))
    return state.at[routes].set(local * orientation)


def _l2_polynomial_state(space: Any) -> Any:
    projection = space.default_runtime.projections[0]
    geometry = space.default_runtime.geometries[0]
    exponents = [tuple(row) for row in projection.basis.exponents]
    coefficients = jnp.zeros((projection.basis.feature_count,))
    coefficients = coefficients.at[exponents.index((0, 0))].set(
        1.0 + geometry.centroids[0, 0]
    )
    coefficients = coefficients.at[exponents.index((1, 0))].set(
        geometry.characteristic_lengths[0]
    )
    local = projection.dof_matrix[0] @ coefficients
    routes = space.dof_map.cell_dofs[0][0]
    return jnp.zeros((space.dof_map.global_dof_count,)).at[routes].set(local)


def _compiled(realization: Any = "matrix_free", degree: Any = 1) -> Any:
    space = _space(degree)
    constraint = phx.discretization.virtual_element_dirichlet_constraint(space, "u")
    form = phx.equations.VirtualElementForm(
        "poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", 0.0),
        ),
    )
    return phx.equations.compile_virtual_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=lambda points: points[:, 0] + points[:, 1],
        execution_policy=phx.equations.VirtualElementExecutionPolicy(
            realization=realization
        ),
    )


def test_virtual_element_compiler_scenario_1() -> None:
    matrix_free = _compiled("matrix_free")
    sparse = _compiled("sparse")
    value = jnp.linspace(-0.5, 0.5, matrix_free.state_space.size)

    assert jnp.allclose(
        matrix_free.affine_operator().mv(value),
        sparse.affine_operator().mv(value),
        atol=1.0e-11,
    )
    assert jnp.allclose(
        matrix_free.affine_operator().transpose_mv(value),
        sparse.affine_operator().transpose_mv(value),
        atol=1.0e-11,
    )
    compiled = _compiled("matrix_free", degree=2)
    problem, rhs = compiled.linear_system()
    solution = phx.linalg.solve(problem, rhs)
    full = compiled.expand(solution.value)

    assert jnp.sqrt(jnp.sum(compiled.residual(solution.value) ** 2)) < 1.0e-9
    assert jnp.allclose(full[4], 1.0, atol=1.0e-9)
    space = _space(1)
    form = phx.equations.VirtualElementForm(
        "neumann",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.BoundaryLoadAction("u", 0.0),
        ),
    )
    compiled = phx.equations.compile_virtual_element_problem(form, space)
    problem, rhs = compiled.linear_system()

    assert problem.nullspace_policy is not None
    assert problem.nullspace_policy.right is not None
    assert jnp.allclose(rhs, 0.0)
    space = _space(1)
    robin = phx.equations.VirtualElementRobinAction(
        "u", 2.0, 0.0, space.exterior_facet_domain
    )
    form = phx.equations.VirtualElementForm(
        "reaction-robin",
        "u",
        (phx.equations.MassAction("u", 1.0), robin),
    )
    compiled = phx.equations.compile_virtual_element_problem(form, space)
    operator = compiled.affine_operator()
    value = jnp.arange(space.dof_map.global_dof_count, dtype="float64")

    assert jnp.allclose(operator.mv(value), operator.transpose_mv(value), atol=1.0e-11)
    assert jnp.all(jnp.isfinite(operator.mv(value)))
    linear = _space(1)
    quadratic = _space(2)
    state = jnp.zeros((quadratic.dof_map.global_dof_count,))

    with pytest.raises(ValueError, match="runtime is incompatible"):
        phx.equations.project_virtual_element_field(
            quadratic,
            state,
            runtime=linear.default_runtime,
        )

    compiled = _compiled("matrix_free", degree=2)
    context = phx.equations.VirtualElementExecutionContext(linear.default_runtime)
    with pytest.raises(ValueError, match="context is incompatible"):
        compiled.expand(jnp.zeros((compiled.state_space.size,)), context)
    for factory, differential_kind, expected_rhs_pairing in (
        (phx.discretization.conforming_hdiv_virtual_element, "divergence", 3.5),
        (phx.discretization.conforming_hcurl_virtual_element, "curl", 2.5),
    ):
        space = _single_cell_space(factory)
        state = _vector_polynomial_state(space, differential_kind)
        differential_form = phx.equations.VirtualElementForm(
            f"{differential_kind}-form",
            "u",
            (phx.equations.DiffusionAction("u", 1.0),),
        )
        matrix_free = phx.equations.compile_virtual_element_problem(
            differential_form, space
        )
        sparse = phx.equations.compile_virtual_element_problem(
            differential_form,
            space,
            execution_policy=phx.equations.VirtualElementExecutionPolicy(
                realization="sparse"
            ),
        )
        matrix_free_action = matrix_free.affine_operator()
        sparse_action = sparse.affine_operator()

        np.testing.assert_allclose(
            matrix_free_action.mv(state), sparse_action.mv(state), atol=2.0e-9
        )
        np.testing.assert_allclose(state @ matrix_free_action.mv(state), 4.0, atol=2.0e-9)

        source = jnp.asarray((1.0, 2.0))
        mass_and_load = phx.equations.VirtualElementForm(
            f"{differential_kind}-mass-load",
            "u",
            (
                phx.equations.MassAction("u", 1.0),
                phx.equations.SourceAction("u", source),
                phx.equations.BoundaryLoadAction("u", 1.0),
            ),
        )
        compiled = phx.equations.compile_virtual_element_problem(mass_and_load, space)
        mass = compiled.affine_operator()
        robin_form = phx.equations.VirtualElementForm(
            f"{differential_kind}-robin",
            "u",
            (
                phx.equations.VirtualElementRobinAction(
                    "u", 1.0, 0.0, space.exterior_facet_domain
                ),
            ),
        )
        robin = phx.equations.compile_virtual_element_problem(
            robin_form, space
        ).affine_operator()
        np.testing.assert_allclose(state @ robin.mv(state), 2.0, atol=2.0e-9)
        rhs = compiled.full_right_hand_side()

        np.testing.assert_allclose(state @ mass.mv(state), 2.0 / 3.0, atol=2.0e-9)
        np.testing.assert_allclose(state @ rhs, expected_rhs_pairing, atol=2.0e-9)


def test_l2_vem_contracts() -> None:
    space = _single_cell_space(phx.discretization.discontinuous_l2_virtual_element)
    state = _l2_polynomial_state(space)
    form = phx.equations.VirtualElementForm(
        "l2-cell-form",
        "u",
        (
            phx.equations.MassAction("u", 1.0),
            phx.equations.SourceAction("u", 2.0),
        ),
    )
    compiled = phx.equations.compile_virtual_element_problem(form, space)

    np.testing.assert_allclose(
        state @ compiled.affine_operator().mv(state), 7.0 / 3.0, atol=2.0e-9
    )
    np.testing.assert_allclose(state @ compiled.full_right_hand_side(), 3.0, atol=2.0e-9)
    space = _single_cell_space(phx.discretization.discontinuous_l2_virtual_element)
    diffusion = phx.equations.VirtualElementForm(
        "undefined-l2-diffusion",
        "u",
        (phx.equations.DiffusionAction("u", jnp.asarray((1.0, 2.0))),),
    )
    with pytest.raises(ValueError, match="Diffusion is undefined"):
        phx.equations.compile_virtual_element_problem(diffusion, space)

    boundary = phx.equations.VirtualElementForm(
        "undefined-l2-boundary",
        "u",
        (phx.equations.BoundaryLoadAction("u", jnp.ones((3, 4, 5))),),
    )
    with pytest.raises(ValueError, match="no boundary trace"):
        phx.equations.compile_virtual_element_problem(boundary, space)


_REALIZATIONS = ("matrix_free", "sparse")
_CENTRAL_STEP = 1.0e-5
# Central differences of exact dense solves agree with implicit derivatives up
# to O(step^2) truncation and O(eps / step) roundoff.
_RTOL = 1.0e-6
_ATOL = 1.0e-8
# Diffusivity scale, source amplitude, Dirichlet amplitude.
_DIRICHLET_PARAMETERS = jnp.asarray([1.3, 0.7, 0.4])
_DIRICHLET_ARGUMENT_IDS = ("diffusivity", "source", "dirichlet-lift")
# Reaction, Robin coefficient, Robin value, boundary load.
_BOUNDARY_PARAMETERS = jnp.asarray([0.6, 1.3, 0.7, 0.4])
_BOUNDARY_ARGUMENT_IDS = ("reaction", "robin-coefficient", "robin-value", "load")


# VEM coefficient callables receive the runtime user_args directly.
def _runtime_coefficient(
    name: str, profile: Callable[[Array], Array]
) -> phx.equations.VariationalCoefficient:
    def evaluate(points: Array, args: Any) -> Array:
        return args[name] * profile(points)

    return phx.equations.coefficient(evaluate, coefficient_id=f"runtime-{name}")


def _policy(realization: str) -> Any:
    return phx.equations.VirtualElementExecutionPolicy(realization=realization)


def _mathematical_policy() -> Any:
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
    )


def _dirichlet_solution(realization: str) -> Callable[[Array], Array]:
    """Full solved state as a function of runtime diffusivity, source, and lift."""
    space = _space(2)
    constraint = phx.discretization.virtual_element_dirichlet_constraint(space, "u")
    form = phx.equations.VirtualElementForm(
        "runtime-poisson",
        "u",
        (
            phx.equations.DiffusionAction(
                "u", _runtime_coefficient("kappa", lambda p: 1.0 + 0.5 * p[..., 0])
            ),
            phx.equations.SourceAction(
                "u", _runtime_coefficient("source", lambda p: 1.0 + p[..., 1])
            ),
        ),
    )
    compiled = phx.equations.compile_virtual_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=0.0,
        execution_policy=_policy(realization),
    )
    policy = _mathematical_policy()

    def solution(theta: Array) -> Array:
        context = phx.equations.VirtualElementExecutionContext(
            space.default_runtime,
            lift=constraint.lift(lambda p: theta[2] * (p[:, 0] + 2.0 * p[:, 1])),
            user_args={"kappa": theta[0], "source": theta[1]},
        )
        system, right_hand_side = compiled.linear_system(context)
        result = phx.linalg.solve(system, right_hand_side, policy=policy)
        return compiled.expand(result.value, context)

    return solution


def _boundary_problem(realization: str) -> Any:
    space = _space(1)
    form = phx.equations.VirtualElementForm(
        "runtime-reaction-robin",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.MassAction(
                "u", _runtime_coefficient("reaction", lambda p: 1.0 + 0.0 * p[..., 0])
            ),
            phx.equations.VirtualElementRobinAction(
                "u",
                _runtime_coefficient("alpha", lambda p: 1.0 + p[..., 0]),
                _runtime_coefficient("value", lambda p: 1.0 + p[..., 1]),
                space.exterior_facet_domain,
            ),
            phx.equations.BoundaryLoadAction(
                "u", _runtime_coefficient("load", lambda p: 1.0 + 0.0 * p[..., 0])
            ),
        ),
    )
    return phx.equations.compile_virtual_element_problem(
        form, space, execution_policy=_policy(realization)
    )


def _boundary_arguments(theta: Array) -> dict[str, Array]:
    return {
        "reaction": theta[0],
        "alpha": theta[1],
        "value": theta[2],
        "load": theta[3],
    }


def _boundary_solution(realization: str) -> Callable[[Array], Array]:
    """Solved state as a function of runtime reaction, Robin, and load data."""
    compiled = _boundary_problem(realization)
    policy = _mathematical_policy()

    def solution(theta: Array) -> Array:
        system, right_hand_side = compiled.linear_system(_boundary_arguments(theta))
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


@pytest.mark.parametrize("argument", range(3), ids=_DIRICHLET_ARGUMENT_IDS)
@pytest.mark.parametrize("realization", _REALIZATIONS)
def test_runtime_dirichlet_solve_derivatives_match_central_differences(
    realization: str, argument: int
) -> None:
    _assert_solution_derivative(
        _dirichlet_solution(realization), _DIRICHLET_PARAMETERS, argument
    )


@pytest.mark.parametrize("argument", range(4), ids=_BOUNDARY_ARGUMENT_IDS)
@pytest.mark.parametrize("realization", _REALIZATIONS)
def test_runtime_boundary_solve_derivatives_match_central_differences(
    realization: str, argument: int
) -> None:
    _assert_solution_derivative(
        _boundary_solution(realization), _BOUNDARY_PARAMETERS, argument
    )


@pytest.mark.parametrize("realization", _REALIZATIONS)
def test_runtime_objective_gradient_compiles_under_jit(realization: str) -> None:
    solution = _dirichlet_solution(realization)

    def objective(theta: Array) -> Array:
        state = solution(theta)
        return jnp.sum(jnp.linspace(0.5, 1.5, state.size) * state**2)

    gradient = jax.jit(jax.grad(objective))(_DIRICHLET_PARAMETERS)
    compiled_objective = jax.jit(objective)
    reference = [
        _central_difference(compiled_objective, _DIRICHLET_PARAMETERS, direction)
        for direction in jnp.eye(3)
    ]

    np.testing.assert_allclose(gradient, reference, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("realization", _REALIZATIONS)
def test_runtime_affine_operator_transpose_matches_forward_action(
    realization: str,
) -> None:
    compiled = _boundary_problem(realization)
    arguments = _boundary_arguments(_BOUNDARY_PARAMETERS)
    size = compiled.state_space.size
    state = jnp.linspace(-0.3, 0.8, size)
    direction = jnp.cos(jnp.arange(size, dtype=jnp.float64))
    covector = jnp.linspace(0.9, -0.6, size)
    operator = compiled.affine_operator(arguments)
    reference = _central_difference(
        lambda value: compiled.residual(value, arguments), state, direction
    )

    np.testing.assert_allclose(operator.mv(direction), reference, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        jnp.dot(covector, operator.mv(direction)),
        jnp.dot(direction, operator.transpose_mv(covector)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
