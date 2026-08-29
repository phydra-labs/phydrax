#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _mesh():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = jnp.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=jnp.int32)
    return phx.discretization.CellMesh.from_triangles(vertices, cells)


def test_darcy_compiles_as_one_product_space_problem():
    mesh = _mesh()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        (
            phx.discretization.FiniteElementFieldSpec(
                "q", phx.discretization.raviart_thomas_element("triangle")
            ),
            phx.discretization.FiniteElementFieldSpec(
                "p", phx.discretization.discontinuous_element("triangle", 0)
            ),
        ),
    ).prepare()
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.fem.darcy_form("q", "p"), discretization
    )
    residual = compiled.residual(compiled.state_space.zeros())

    assert type(compiled) is phx.equations.CompiledFiniteElementProblem
    assert compiled.state_space.names == ("q", "p")
    assert tuple(value.shape for value in residual) == ((8,), (4,))
    assert not hasattr(compiled, "subproblems")


def test_stokes_has_nonzero_off_diagonal_jvp_and_adjoint_identity():
    mesh = _mesh()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        (
            phx.discretization.FiniteElementFieldSpec(
                "u",
                phx.discretization.lagrange_element("triangle", 2),
                component_shape=(2,),
            ),
            phx.discretization.FiniteElementFieldSpec(
                "p", phx.discretization.lagrange_element("triangle", 1)
            ),
        ),
    ).prepare()
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.fem.stokes_form("u", "p"), discretization
    )
    zero = compiled.state_space.zeros()
    pressure_direction = (jnp.zeros_like(zero[0]), jnp.arange(5.0))
    _, image = jax.jvp(compiled.residual, (zero,), (pressure_direction,))
    block = compiled.block_linearization_operator(zero)
    block_image = block.mv(pressure_direction)

    assert jnp.linalg.norm(image[0]) > 0.0
    assert jnp.linalg.norm(image[1]) == 0.0
    assert compiled.block_dependency_graph() == ((True, True), (True, False))
    assert jnp.allclose(block_image[0], image[0])
    assert jnp.allclose(block_image[1], image[1])


def test_upwind_constant_state_is_preserved_with_matching_inflow():
    mesh = _mesh()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "c", phx.discretization.discontinuous_element("triangle", 1)
        ),
    ).prepare()
    form = phx.equations.fem.upwind_advection_form(
        "c",
        jnp.asarray([1.0, 0.0]),
        interior_domain=discretization.interior_facet_domain,
        boundary_domain=discretization.exterior_facet_domain,
        inflow=1.0,
    )
    residual = phx.equations.compile_finite_element_problem(
        form, discretization
    ).residual(jnp.ones((12,)))

    assert jnp.linalg.norm(residual) < 1.0e-12


def test_hdg_local_system_is_generated_and_reconstructed():
    mesh = _mesh()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.discontinuous_element("triangle", 1)
        ),
    ).prepare()
    result = phx.equations.fem.solve_hdg_poisson(
        discretization,
        "u",
        0.0,
        lambda points: points[:, 0] + points[:, 1],
    )

    assert bool(result.successful)
    assert result.local.shape == (4, 6)
    assert result.trace.shape == (8,)
