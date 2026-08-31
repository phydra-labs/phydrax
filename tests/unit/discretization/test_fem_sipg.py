#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _sipg_discretization():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = jnp.asarray([[0, 1, 3], [1, 2, 3]], dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(vertices, cells)
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.discontinuous_element("triangle", 1)
    )
    return phx.discretization.FiniteElementPlan(mesh, field).prepare()


def test_sipg_nitsche_reproduces_affine_solution_with_reversed_neighbour():
    discretization = _sipg_discretization()
    data = phx.equations.coefficient(
        lambda points, context: points[..., 0] + points[..., 1],
        coefficient_id="sipg-affine-boundary",
    )
    boundary = phx.equations.fem.sipg_dirichlet(
        discretization.exterior_facet_domain, data
    )
    form = phx.equations.fem.sipg_poisson_form(
        "u",
        1.0,
        phx.equations.fem.SIPGPenaltyPolicy(12.0),
        discretization.cell_domain,
        discretization.interior_facet_domain,
        (boundary,),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    state = discretization.project(
        "u", lambda points, args: points[..., 0] + points[..., 1]
    )

    assert jnp.linalg.norm(compiled.full_residual(state)) < 1.0e-12


def test_sipg_operator_is_symmetric_with_harmonic_cell_coefficient():
    discretization = _sipg_discretization()
    cells = discretization.mesh.topology.entity_sets[
        discretization.mesh.topological_dimension
    ]
    coefficient = phx.equations.coefficient(
        jnp.asarray([1.0, 10.0]),
        location="cell",
        support_id=discretization.support.support_id,
        entity_set_id=cells.entity_set_id,
    )
    form = phx.equations.fem.sipg_poisson_form(
        "u",
        coefficient,
        phx.equations.fem.SIPGPenaltyPolicy(20.0),
        discretization.cell_domain,
        discretization.interior_facet_domain,
        (),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    operator = compiled.linearization_operator(jnp.zeros((6,)))
    left = jnp.arange(1.0, 7.0)
    right = jnp.arange(6.0, 0.0, -1.0)

    defect = jnp.vdot(left, operator.mv(right)) - jnp.vdot(operator.mv(left), right)
    assert jnp.abs(defect) < 1.0e-10


def test_pure_neumann_sipg_attaches_verified_component_nullspace():
    discretization = _sipg_discretization()
    form = phx.equations.fem.sipg_poisson_form(
        "u",
        1.0,
        phx.equations.fem.SIPGPenaltyPolicy(12.0),
        discretization.cell_domain,
        discretization.interior_facet_domain,
        (),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    system, right_hand_side = compiled.linear_system()

    assert int(system.nullspace_policy.right.dimension) == 1
    assert bool(system.nullspace_policy.certificate.valid)
    assert jnp.linalg.norm(system.operator.mv(jnp.ones((6,)))) < 1.0e-12
    assert jnp.linalg.norm(right_hand_side) < 1.0e-12
