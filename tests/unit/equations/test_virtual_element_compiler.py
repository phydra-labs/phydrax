#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

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
