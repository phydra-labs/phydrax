#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _space(degree=1):
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


def _single_cell_space(factory):
    coordinates = jnp.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    mesh = phx.discretization.CellMesh.from_polygons(
        coordinates, ((0, 1, 2, 3),)
    )
    field = phx.discretization.VirtualElementFieldSpec("u", factory(1))
    return phx.discretization.VirtualElementPlan(mesh, field).prepare()


def _vector_polynomial_state(space, differential_kind):
    projection = space.default_runtime.projections[0]
    geometry = space.default_runtime.geometries[0]
    exponents = [
        tuple(int(value) for value in row) for row in projection.basis.exponents
    ]
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


def _l2_polynomial_state(space):
    projection = space.default_runtime.projections[0]
    geometry = space.default_runtime.geometries[0]
    exponents = [
        tuple(int(value) for value in row) for row in projection.basis.exponents
    ]
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


def _compiled(realization="matrix_free", degree=1):
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


def test_matrix_free_and_sparse_vem_actions_match():
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


def test_vem_linear_patch_and_constraint_lift():
    compiled = _compiled("matrix_free", degree=2)
    problem, rhs = compiled.linear_system()
    solution = phx.linalg.solve(problem, rhs)
    full = compiled.expand(solution.value)

    assert jnp.sqrt(jnp.sum(compiled.residual(solution.value) ** 2)) < 1.0e-9
    assert jnp.allclose(full[4], 1.0, atol=1.0e-9)


def test_vem_neumann_problem_declares_constant_nullspace():
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


def test_vem_robin_and_mass_are_symmetric():
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
    value = jnp.arange(space.dof_map.global_dof_count, dtype=float)

    assert jnp.allclose(operator.mv(value), operator.transpose_mv(value), atol=1.0e-11)
    assert jnp.all(jnp.isfinite(operator.mv(value)))


@pytest.mark.parametrize(
    ("factory", "differential_kind", "expected_rhs_pairing"),
    (
        (phx.discretization.conforming_hdiv_virtual_element, "divergence", 3.5),
        (phx.discretization.conforming_hcurl_virtual_element, "curl", 2.5),
    ),
)
def test_vector_vem_assembles_differential_mass_source_and_trace_forms(
    factory, differential_kind, expected_rhs_pairing
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
    np.testing.assert_allclose(
        state @ matrix_free_action.mv(state), 4.0, atol=2.0e-9
    )

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
    np.testing.assert_allclose(
        state @ rhs, expected_rhs_pairing, atol=2.0e-9
    )


def test_l2_vem_assembles_only_cell_mass_and_source_forms():
    space = _single_cell_space(
        phx.discretization.discontinuous_l2_virtual_element
    )
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
    np.testing.assert_allclose(
        state @ compiled.full_right_hand_side(), 3.0, atol=2.0e-9
    )


def test_l2_vem_rejects_undefined_operators_before_evaluation():
    space = _single_cell_space(
        phx.discretization.discontinuous_l2_virtual_element
    )
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
