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
from phydrax.discretization.fem._reference_operator import (
    PreparedFiniteElementReference,
)
from phydrax.equations._finite_element_variational import (
    FiniteElementForm,
    InteriorFacetAction,
    MassAction,
    PairwiseVolumeFluxAction,
)
from phydrax.equations._variational import VariationalCoefficient
from phydrax.equations.fem._execution import (
    FiniteElementMassPolicy,
    TensorProductPartialAssemblyOperator,
)
from phydrax.equations.fem._lowering import (
    compile_workset_program,
    kernel_table_from_form,
    lower_finite_element_form,
)
from phydrax.equations.fem._operators import (
    FiniteElementFacetMetricData,
    FiniteElementMetricData,
    PreparedFacetTrace,
)
from phydrax.integration._rules import (
    GaussLobattoLegendreRule,
    ReferenceHexahedronRule,
    ReferenceIntervalRule,
    ReferenceQuadrilateralRule,
)


def _prepared(cell, order):
    family = ReferenceNodalFamily(cell, order)
    axis_rule = GaussLobattoLegendreRule(order + 1)
    if cell == "quadrilateral":
        volume = ReferenceQuadrilateralRule(axis_rule)
        facets = (ReferenceIntervalRule(axis_rule),) * 4
    else:
        volume = ReferenceHexahedronRule(axis_rule)
        facets = (ReferenceQuadrilateralRule(axis_rule),) * 6
    reference = PreparedFiniteElementReference(
        family.finite_element(),
        volume,
        facets,
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
    return family, reference


def _affine_metric(cell, reference):
    coordinate_element = phx.discretization.lagrange_element(cell, 1)
    basis, gradients = coordinate_element.tabulate(reference.volume_rule.points)
    coordinates = coordinate_element.reference_nodes[None]
    return FiniteElementMetricData(
        basis,
        gradients,
        coordinates,
        reference.weights,
    )


def test_quad_and_hex_dense_factorized_mass_diffusion_jvp_vjp_agree():
    for cell, order in (("quadrilateral", 3), ("hexahedron", 2)):
        family, reference = _prepared(cell, order)
        metric = _affine_metric(cell, reference)
        plan = SumFactorizationPlan(reference.tensor_tabulation)
        width = int(np.prod(family.nodal_shape))
        route = jnp.arange(width, dtype=jnp.int32)[None]
        mass = TensorProductPartialAssemblyOperator(
            plan,
            metric.weighted_measure.reshape((1,) + plan.tabulation.evaluation_shape),
            route,
            width,
            action_kind="mass",
        )
        diffusion = TensorProductPartialAssemblyOperator(
            plan,
            metric.weighted_metric.reshape(
                (1,)
                + plan.tabulation.evaluation_shape
                + (plan.tabulation.dimension, plan.tabulation.dimension)
            ),
            route,
            width,
            action_kind="diffusion",
        )
        basis = reference.basis_values
        gradients = reference.basis_gradients
        dense_mass = oe.contract("q,qi,qj->ij", metric.weighted_measure[0], basis, basis)
        dense_diffusion = oe.contract(
            "qab,qia,qjb->ij", metric.weighted_metric[0], gradients, gradients
        )
        state = jnp.linspace(-0.4, 1.1, width)
        direction = jnp.linspace(0.9, -0.3, width)
        cotangent = jnp.linspace(-0.2, 0.7, width)
        for operator, matrix in ((mass, dense_mass), (diffusion, dense_diffusion)):
            assert jnp.allclose(operator.mv(state), matrix @ state, atol=2.0e-11)
            _, tangent = jax.jvp(operator.mv, (state,), (direction,))
            _, pullback = jax.vjp(operator.mv, state)
            assert jnp.allclose(tangent, matrix @ direction, atol=2.0e-11)
            assert jnp.allclose(
                pullback(cotangent)[0], matrix.T @ cotangent, atol=2.0e-11
            )


def test_prepared_quad_and_hex_trace_lift_are_exact_adjoints_for_permutations():
    for cell, order in (("quadrilateral", 4), ("hexahedron", 2)):
        _, reference = _prepared(cell, order)
        for facet in reference.facets:
            count = facet.points.shape[0]
            permutation = jnp.arange(count - 1, -1, -1, dtype=jnp.int32)
            trace = PreparedFacetTrace(facet.basis_values, permutation)
            coefficients = jnp.linspace(-0.8, 1.2, reference.element.local_dof_count)
            facet_values = jnp.linspace(0.5, -0.4, count)
            assert jnp.allclose(
                jnp.vdot(trace.trace(coefficients), facet_values),
                jnp.vdot(coefficients, trace.lift(facet_values)),
                atol=2.0e-12,
            )


def test_curved_facet_metric_uses_pointwise_normals_and_weights():
    _, reference = _prepared("quadrilateral", 3)
    facet = reference.facets[1]
    coordinate_family = ReferenceNodalFamily("quadrilateral", 2)
    coordinate_element = coordinate_family.finite_element()
    basis, gradients = coordinate_element.tabulate(facet.points)
    coordinates = coordinate_element.reference_nodes.at[:, 0].add(
        0.15 * coordinate_element.reference_nodes[:, 1] ** 2
    )[None]
    metric = FiniteElementMetricData(basis, gradients, coordinates, facet.weights)
    facet_metric = FiniteElementFacetMetricData(metric, facet.normals, facet.weights)

    assert facet_metric.normal.shape == facet_metric.physical_points.shape
    assert jnp.all(facet_metric.physical_weights > 0.0)
    assert jnp.allclose(jnp.linalg.norm(facet_metric.normal, axis=-1), 1.0)


def _single_tensor_discretization(cell, order):
    if cell == "quadrilateral":
        coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        vertices = ((0, 1, 2, 3),)
    else:
        coordinates = jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            )
        )
        vertices = ((0, 1, 2, 3, 4, 5, 6, 7),)
    mesh = phx.discretization.CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "quads" if cell == "quadrilateral" else "hexes",
                cell,
                jnp.asarray(vertices, dtype=jnp.int32),
            ),
        ),
    )
    element = ReferenceNodalFamily(cell, order).finite_element()
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("u", element),
    ).prepare()


def _single_quad_discretization():
    return _single_tensor_discretization("quadrilateral", 2)


def test_high_order_quadrilateral_dirichlet_constraint_routes_boundary_dofs():
    discretization = _single_tensor_discretization("quadrilateral", 3)
    constraint = phx.discretization.dirichlet_constraint(discretization, "u")
    points = np.asarray(discretization.dof_maps[0].dof_coordinates)
    expected = np.flatnonzero(
        np.isclose(points[:, 0], 0.0)
        | np.isclose(points[:, 0], 1.0)
        | np.isclose(points[:, 1], 0.0)
        | np.isclose(points[:, 1], 1.0)
    )

    assert np.array_equal(np.asarray(constraint.constrained_dofs), expected)
    assert constraint.free_dofs.shape == (4,)


def test_high_order_quad_and_hex_boundary_loads_use_tensor_trace_lifts():
    for cell, order, expected_measure in (
        ("quadrilateral", 3, 4.0),
        ("hexahedron", 2, 6.0),
    ):
        discretization = _single_tensor_discretization(cell, order)
        form = phx.equations.FiniteElementForm(
            "boundary-load",
            "u",
            (phx.equations.BoundaryLoadAction("u", 1.0),),
        )
        compiled = phx.equations.compile_finite_element_problem(
            form,
            discretization,
            execution_policy=phx.equations.FiniteElementExecutionPolicy(
                realization="matrix_free"
            ),
        )
        state = jnp.zeros((discretization.dof_maps[0].global_dof_count,))
        load = -compiled.full_residual(state, None)

        assert jnp.allclose(jnp.sum(load), expected_measure, atol=2.0e-12)


def test_exact_and_lumped_mass_integrate_high_order_tensor_polynomials():
    discretization = _single_tensor_discretization("quadrilateral", 4)
    form = FiniteElementForm(
        "mass-carrier",
        "u",
        (MassAction("u", 1.0),),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free"
        ),
    )
    coordinates = discretization.dof_maps[0].dof_coordinates
    polynomial = coordinates[:, 0] ** 4
    for kind in ("exact", "lumped"):
        mass, _ = compiled._mass_operators(
            compiled._execution_context(None),
            jnp.asarray(1.0),
            FiniteElementMassPolicy(kind),
        )
        assert jnp.allclose(
            jnp.vdot(jnp.ones_like(polynomial), mass.mv(polynomial)),
            0.2,
            atol=2.0e-12,
        )


def test_compiler_keys_bind_reference_layout_and_semantics_not_coefficient_values():
    discretization = _single_quad_discretization()
    field_space_id = discretization.field_spaces[0].field_space_id
    rule = ReferenceQuadrilateralRule(GaussLobattoLegendreRule(3))
    programs = []
    tables = []
    for offset in (0.0, 7.0):
        coefficient = VariationalCoefficient(
            jnp.arange(9.0) + offset,
            coefficient_id="nodal-density",
            location="dof",
            support_id=discretization.support.support_id,
            field_space_id=field_space_id,
        )
        form = FiniteElementForm(
            "nodal-mass",
            "u",
            (MassAction("u", coefficient, rules={"quads": rule}),),
        )
        ir = lower_finite_element_form(form, discretization)
        program = compile_workset_program(
            ir,
            form,
            discretization,
            local_kernel="collocated",
        )
        table = kernel_table_from_form(form, ir, program, discretization)
        programs.append(program)
        tables.append(table)
    first = programs[0].worksets[0].signature
    second = programs[1].worksets[0].signature

    assert first.signature_id == second.signature_id
    assert first.reference_action_ids == second.reference_action_ids
    assert first.field_layout_ids == second.field_layout_ids
    assert first.geometry_action_id == second.geometry_action_id
    assert first.coefficient_layout_ids == second.coefficient_layout_ids
    assert first.local_kernel == "collocated"
    assert (
        tables[0].bindings[0].reference_action_ids
        == tables[1].bindings[0].reference_action_ids
    )
    assert tables[0].bindings[0].ir_semantics_id == programs[0].ir.actions[0].action_id


def test_mass_policy_identities_distinguish_exact_collocated_and_lumped():
    policies = tuple(
        FiniteElementMassPolicy(kind)
        for kind in ("exact", "collocated_diagonal", "lumped")
    )
    assert tuple(policy.kind for policy in policies) == (
        "exact",
        "collocated_diagonal",
        "lumped",
    )
    assert len({policy.policy_id for policy in policies}) == 3


def test_pairwise_volume_flux_lowers_to_collocated_authoritative_action():
    discretization = _single_quad_discretization()
    rule = ReferenceQuadrilateralRule(GaussLobattoLegendreRule(3))

    def central_flux(left, right, left_points, right_points, context):
        value = 0.5 * (left + right)
        return jnp.broadcast_to(value[..., None], value.shape + (2,))

    action = PairwiseVolumeFluxAction(
        "u",
        central_flux,
        rules={"quads": rule},
        action_id="pairwise-central-flux",
    )
    form = FiniteElementForm("pairwise", "u", (action,))
    ir = lower_finite_element_form(form, discretization)
    program = compile_workset_program(
        ir,
        form,
        discretization,
        local_kernel="collocated",
    )
    table = kernel_table_from_form(form, ir, program, discretization)

    assert ir.actions[0].action_kind == "pairwise-volume-flux"
    assert program.worksets[0].signature.local_kernel == "collocated"
    assert table.bindings[0].kernel_kind == "pairwise-volume-flux"
    assert table.bindings[0].local_kernel == "collocated"


def test_tensor_interior_facet_reads_multiple_fields_and_writes_one():
    coordinates = jnp.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        )
    )
    mesh = phx.discretization.CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "quads",
                "quadrilateral",
                jnp.asarray(((0, 1, 4, 3), (1, 2, 5, 4)), dtype=jnp.int32),
            ),
        ),
    )
    element = phx.discretization.discontinuous_element("quadrilateral", 2)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        (
            phx.discretization.FiniteElementFieldSpec("u", element),
            phx.discretization.FiniteElementFieldSpec("g", element, component_shape=(2,)),
        ),
    ).prepare()

    def flux(plus, minus, points, weights, normal, context):
        del points, weights, normal, context
        value = (plus[0] - minus[0]) + jnp.sum(plus[1] - minus[1], axis=-1)
        return value, -value

    action = InteriorFacetAction(
        "u",
        ("u", "g"),
        flux,
        domain=discretization.interior_facet_domain,
        action_id="tensor-cross-field-facet",
    )
    compiled = phx.equations.compile_finite_element_problem(
        FiniteElementForm("tensor-cross-field", ("u", "g"), (action,)),
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="auto",
        ),
    )
    state = compiled.state_space.zeros()
    state = (
        jnp.linspace(0.0, 1.0, state[0].size).reshape(state[0].shape),
        jnp.linspace(-0.5, 0.75, state[1].size).reshape(state[1].shape),
    )
    residual = compiled.residual(state)
    assert jnp.linalg.norm(residual[0]) > 0.0
    assert jnp.allclose(jnp.sum(residual[0]), 0.0, atol=2.0e-12)
    assert jnp.array_equal(residual[1], jnp.zeros_like(residual[1]))
