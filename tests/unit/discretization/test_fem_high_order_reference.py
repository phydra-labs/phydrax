#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import pytest

from phydrax.discretization.fem._generic import _degree_aware_reference_rule
from phydrax.discretization.fem._high_order import (
    lagrange_1d_tabulation,
    ReferenceNodalFamily,
    SimplexNodalFamily,
    SumFactorizationPlan,
    TensorProductTabulation,
)
from phydrax.discretization.fem._precision import FiniteElementPrecisionPolicy
from phydrax.discretization.fem._reference_operator import (
    PreparedFiniteElementReference,
)
from phydrax.integration._rules import (
    GaussLegendreRule,
    GaussLobattoLegendreRule,
    ReferenceHexahedronRule,
    ReferenceIntervalRule,
    ReferenceQuadrilateralRule,
    ReferenceTetrahedronRule,
    ReferenceTriangleRule,
)


@pytest.mark.parametrize(
    ("rule", "order", "exact_degree"),
    (
        (GaussLegendreRule, 5, 9),
        (GaussLobattoLegendreRule, 6, 9),
    ),
)
def test_legendre_rules_are_positive_and_polynomial_exact(rule, order, exact_degree):
    prepared_rule = rule(order)
    data = prepared_rule.data()
    nodes = np.asarray(data.nodes)
    weights = np.asarray(data.weights)

    assert prepared_rule.exact_degree == exact_degree
    assert data.degree == exact_degree
    assert np.all(np.diff(nodes) > 0.0)
    assert np.all(weights > 0.0)
    assert np.isclose(np.sum(weights), 2.0)
    reference = ReferenceIntervalRule(prepared_rule).materialize()
    assert reference.points.shape == (order, 1)
    assert np.all(np.asarray(reference.weights) > 0.0)
    assert np.isclose(np.sum(np.asarray(reference.weights)), 1.0)
    if rule is GaussLobattoLegendreRule:
        assert np.array_equal(
            np.asarray(reference.points)[[0, -1], 0],
            np.asarray((0.0, 1.0)),
        )
    for degree in range(exact_degree + 1):
        exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
        assert np.isclose(
            np.sum(weights * nodes**degree),
            exact,
            rtol=2.0e-11,
            atol=2.0e-11,
        )


def test_tetrahedral_degree_aware_rule_integrates_mass_degree_polynomial():
    polynomial_degree = 4
    points, weights = _degree_aware_reference_rule(
        "tetrahedron",
        polynomial_degree,
    )
    monomial_degree = 2 * polynomial_degree
    expected = 1.0 / (
        (monomial_degree + 1) * (monomial_degree + 2) * (monomial_degree + 3)
    )

    assert jnp.allclose(
        jnp.sum(weights * points[:, 0] ** monomial_degree),
        expected,
        atol=2.0e-13,
    )


def test_gll_family_uses_stable_barycentric_polynomial_reproduction():
    family = ReferenceNodalFamily("quadrilateral", 8)
    assert family.orders == (8, 8)
    nodes = family.nodes_by_axis[0]
    barycentric = family.barycentric_weights_by_axis[0]
    points = jnp.linspace(0.0, 1.0, 31)
    values, gradients = lagrange_1d_tabulation(
        nodes,
        points,
        barycentric_weights=barycentric,
    )
    nodal_polynomial = nodes**8 - 2.0 * nodes**5 + 0.25 * nodes**2 - 1.0
    expected = points**8 - 2.0 * points**5 + 0.25 * points**2 - 1.0
    expected_gradient = 8.0 * points**7 - 10.0 * points**4 + 0.5 * points
    values_at_nodes, _ = lagrange_1d_tabulation(
        nodes,
        nodes,
        barycentric_weights=barycentric,
    )

    assert jnp.allclose(values_at_nodes, jnp.eye(9), atol=2.0e-10)
    assert jnp.allclose(values @ nodal_polynomial, expected, atol=2.0e-9)
    assert jnp.allclose(
        gradients @ nodal_polynomial,
        expected_gradient,
        atol=2.0e-8,
    )
    assert jnp.all(family.quadrature_weights_by_axis[0] > 0.0)
    assert jnp.allclose(jnp.sum(family.quadrature_weights_by_axis[0]), 1.0)


@pytest.mark.parametrize(
    ("cell", "orders", "points_by_axis"),
    (
        (
            "quadrilateral",
            (2, 4),
            (jnp.asarray([0.1, 0.4, 0.9]), jnp.asarray([0.2, 0.8])),
        ),
        (
            "hexahedron",
            (2, 3, 1),
            (
                jnp.asarray([0.15, 0.65]),
                jnp.asarray([0.1, 0.5, 0.9]),
                jnp.asarray([0.25, 0.75]),
            ),
        ),
    ),
)
def test_anisotropic_dense_and_factorized_reference_actions_agree(
    cell, orders, points_by_axis
):
    family = ReferenceNodalFamily(cell, orders)
    tabulation = TensorProductTabulation(family, points_by_axis)
    plan = SumFactorizationPlan(tabulation)
    points = jnp.stack(
        jnp.meshgrid(*points_by_axis, indexing="ij"),
        axis=-1,
    ).reshape((-1, len(orders)))
    dense_values, dense_gradients = family.tabulate(points)
    coefficients = jnp.linspace(
        -0.75,
        1.25,
        math.prod(family.nodal_shape),
    ).reshape(family.nodal_shape)
    evaluation_shape = tuple(len(points) for points in points_by_axis)
    value_seed = jnp.linspace(-1.0, 0.8, math.prod(evaluation_shape)).reshape(
        evaluation_shape
    )
    gradient_seed = jnp.linspace(
        -0.5,
        1.5,
        math.prod(evaluation_shape) * len(orders),
    ).reshape(evaluation_shape + (len(orders),))

    dense_interpolation = oe.contract(
        "qd,d->q", dense_values, coefficients.reshape((-1,))
    )
    dense_gradient = oe.contract(
        "qdk,d->qk", dense_gradients, coefficients.reshape((-1,))
    )
    dense_interpolation_transpose = oe.contract(
        "qd,q->d", dense_values, value_seed.reshape((-1,))
    )
    dense_gradient_transpose = oe.contract(
        "qdk,qk->d",
        dense_gradients,
        gradient_seed.reshape((-1, len(orders))),
    )

    assert family.orders == orders
    assert tabulation.nodal_shape == family.nodal_shape
    assert tabulation.evaluation_shape == evaluation_shape
    assert jnp.allclose(
        plan.interpolate(coefficients).reshape((-1,)),
        dense_interpolation,
        atol=2.0e-10,
    )
    assert jnp.allclose(
        plan.gradient(coefficients).reshape((-1, len(orders))),
        dense_gradient,
        atol=2.0e-10,
    )
    assert jnp.allclose(
        plan.interpolate_transpose(value_seed).reshape((-1,)),
        dense_interpolation_transpose,
        atol=2.0e-10,
    )
    assert jnp.allclose(
        plan.gradient_transpose(gradient_seed).reshape((-1,)),
        dense_gradient_transpose,
        atol=2.0e-10,
    )


def _flatten_entity_dofs(element):
    return tuple(
        dof for dimension in element.entity_dofs for entity in dimension for dof in entity
    )


def test_anisotropic_quadrilateral_entity_partition_and_orientation():
    element = ReferenceNodalFamily("quadrilateral", (2, 3)).finite_element()

    assert tuple(entity[0] for entity in element.entity_dofs[0]) == (0, 8, 11, 3)
    assert element.entity_dofs[1] == ((4,), (9, 10), (7,), (2, 1))
    assert tuple(len(entity) for entity in element.entity_dofs[1]) == (1, 2, 1, 2)
    assert len(element.entity_dofs[2][0]) == 2
    assert tuple(sorted(_flatten_entity_dofs(element))) == tuple(range(12))


def test_anisotropic_hexahedron_entity_partition_and_orientation():
    family = ReferenceNodalFamily("hexahedron", (2, 3, 4))
    element = family.finite_element()

    assert tuple(entity[0] for entity in element.entity_dofs[0]) == (
        0,
        40,
        55,
        15,
        4,
        44,
        59,
        19,
    )
    assert tuple(len(entity) for entity in element.entity_dofs[1]) == (
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
        3,
        3,
        3,
        3,
    )
    assert tuple(len(entity) for entity in element.entity_dofs[2]) == (
        2,
        2,
        3,
        6,
        3,
        6,
    )
    assert len(element.entity_dofs[3][0]) == 6
    assert tuple(sorted(_flatten_entity_dofs(element))) == tuple(range(60))


def _all_actions():
    return (
        "interpolate",
        "interpolate_transpose",
        "gradient",
        "gradient_transpose",
        "trace",
        "trace_transpose",
    )


def _prepare_quad(family, *, actions=None, precision=None, volume_order=4):
    return PreparedFiniteElementReference(
        family.finite_element(),
        ReferenceQuadrilateralRule(GaussLegendreRule(volume_order)),
        (ReferenceIntervalRule(GaussLobattoLegendreRule(4)),) * 4,
        _all_actions() if actions is None else actions,
        FiniteElementPrecisionPolicy() if precision is None else precision,
        tensor_family=family,
    )


def _prepare_hex(family):
    return PreparedFiniteElementReference(
        family.finite_element(),
        ReferenceHexahedronRule(GaussLegendreRule(3)),
        (ReferenceQuadrilateralRule(GaussLobattoLegendreRule(3)),) * 6,
        _all_actions(),
        FiniteElementPrecisionPolicy(),
        tensor_family=family,
    )


@pytest.mark.parametrize(
    ("family", "prepare"),
    (
        (ReferenceNodalFamily("quadrilateral", (3, 2)), _prepare_quad),
        (ReferenceNodalFamily("hexahedron", (2, 2, 2)), _prepare_hex),
    ),
)
def test_prepared_quad_and_hex_trace_and_transpose_consistency(family, prepare):
    prepared = prepare(family)
    coefficients = jnp.linspace(-0.6, 1.1, family.finite_element().local_dof_count)
    volume_values = prepared.interpolate(coefficients)
    volume_gradients = prepared.gradient(coefficients)
    value_seed = jnp.linspace(-0.3, 0.7, volume_values.shape[0])
    gradient_seed = jnp.linspace(
        -0.9,
        0.4,
        volume_gradients.size,
    ).reshape(volume_gradients.shape)
    factorized = SumFactorizationPlan(prepared.tensor_tabulation)

    assert prepared.report.tensor_factorized
    assert prepared.report.point_count == prepared.weights.shape[0]
    assert jnp.allclose(
        factorized.interpolate(coefficients.reshape(family.nodal_shape)).reshape((-1,)),
        volume_values,
        atol=2.0e-10,
    )
    assert jnp.allclose(
        factorized.gradient(coefficients.reshape(family.nodal_shape)).reshape(
            volume_gradients.shape
        ),
        volume_gradients,
        atol=2.0e-10,
    )
    assert jnp.allclose(
        jnp.vdot(volume_values, value_seed),
        jnp.vdot(coefficients, prepared.interpolate_transpose(value_seed)),
        atol=2.0e-10,
    )
    assert jnp.allclose(
        jnp.vdot(volume_gradients, gradient_seed),
        jnp.vdot(coefficients, prepared.gradient_transpose(gradient_seed)),
        atol=2.0e-10,
    )
    for facet in prepared.facets:
        direct_values, direct_gradients = family.tabulate(facet.points)
        trace_seed = jnp.linspace(-0.25, 0.75, facet.points.shape[0])
        trace = prepared.trace(facet.facet_index, coefficients)

        assert jnp.all(facet.weights > 0.0)
        assert jnp.allclose(jnp.linalg.norm(facet.normals, axis=-1), 1.0)
        assert jnp.allclose(facet.basis_values, direct_values, atol=2.0e-10)
        assert jnp.allclose(facet.basis_gradients, direct_gradients, atol=2.0e-10)
        assert jnp.allclose(trace, direct_values @ coefficients, atol=2.0e-10)
        assert jnp.allclose(
            jnp.vdot(trace, trace_seed),
            jnp.vdot(
                coefficients,
                prepared.trace_transpose(facet.facet_index, trace_seed),
            ),
            atol=2.0e-10,
        )


def test_prepared_reference_identity_binds_rules_actions_and_precision():
    family = ReferenceNodalFamily("quadrilateral", 3)
    baseline = _prepare_quad(family)
    reordered = _prepare_quad(family, actions=tuple(reversed(_all_actions())))
    fewer_actions = _prepare_quad(family, actions=("interpolate", "gradient"))
    other_rule = _prepare_quad(family, volume_order=5)
    other_precision = _prepare_quad(
        family,
        precision=FiniteElementPrecisionPolicy(
            geometry_dtype="float32",
            evaluation_dtype="float32",
        ),
    )

    assert baseline.prepared_id == reordered.prepared_id
    assert baseline.prepared_id != fewer_actions.prepared_id
    assert baseline.prepared_id != other_rule.prepared_id
    assert baseline.prepared_id != other_precision.prepared_id
    assert baseline.report.report_id != fewer_actions.report.report_id
    assert baseline.report.precision_id != other_precision.report.precision_id


@pytest.mark.parametrize(
    ("cell", "order"),
    (("triangle", 5), ("tetrahedron", 3)),
)
def test_modepy_simplex_family_and_prepared_reference_reproduce_polynomials(cell, order):
    family = SimplexNodalFamily(cell, order)
    dimension = family.nodes.shape[1]
    nodal_values, _nodal_gradients = family.tabulate(family.nodes)
    np.testing.assert_allclose(nodal_values, np.eye(family.nodes.shape[0]), atol=2.0e-10)
    points = np.asarray(family.nodes) * 0.73 + 0.05
    values, gradients = family.tabulate(points)
    polynomial_nodes = np.asarray(family.nodes)[:, 0] ** order + 0.25 * np.asarray(
        family.nodes
    )[:, -1] ** min(order, 2)
    expected = points[:, 0] ** order + 0.25 * points[:, -1] ** min(order, 2)
    np.testing.assert_allclose(values @ polynomial_nodes, expected, atol=3.0e-9)
    assert gradients.shape == (
        points.shape[0],
        family.nodes.shape[0],
        dimension,
    )

    axis_rule = GaussLegendreRule(order + 2)
    volume_rule = (
        ReferenceTriangleRule(axis_rule)
        if cell == "triangle"
        else ReferenceTetrahedronRule(axis_rule)
    )
    facet_rule = (
        ReferenceIntervalRule(axis_rule)
        if cell == "triangle"
        else ReferenceTriangleRule(axis_rule)
    )
    facet_count = 3 if cell == "triangle" else 4
    reference = PreparedFiniteElementReference(
        family.finite_element(),
        volume_rule,
        (facet_rule,) * facet_count,
        (
            "interpolate",
            "interpolate_transpose",
            "gradient",
            "gradient_transpose",
            "trace",
            "trace_transpose",
        ),
        FiniteElementPrecisionPolicy(),
    )
    mass = oe.contract(
        "q,qi,qj->ij",
        reference.weights,
        reference.basis_values,
        reference.basis_values,
    )
    assert np.min(np.linalg.eigvalsh(np.asarray(mass))) > 0.0
    for facet in reference.facets:
        assert facet.normals.shape == facet.points.shape
        assert jnp.all(facet.weights > 0.0)
    assert family.condition_number < 1.0e5


def test_facet_orientation_groups_have_exact_inverses_and_composition():
    from phydrax.discretization.fem._reference_topology import (
        facet_orientation_actions,
        facet_orientation_between,
    )

    for shape, size in (
        ("point", 1),
        ("edge", 2),
        ("triangle", 3),
        ("quadrilateral", 4),
    ):
        identity = facet_orientation_actions(shape)[0]
        values = jnp.arange(size)
        for action in facet_orientation_actions(shape):
            assert action.compose(action.inverse) == identity
            assert action.inverse.compose(action) == identity
            np.testing.assert_array_equal(
                action.inverse.apply(action.apply(values)), values
            )
    action = facet_orientation_between((10, 20, 30), (20, 30, 10))
    assert action.permutation == (2, 0, 1)


def test_triangular_physical_mortar_reproduces_total_degree_space():
    from phydrax.discretization.fem._mortar import (
        serial_finite_element_mortar_plan,
    )
    from phydrax.integration import GaussLegendreRule, ReferenceTriangleRule

    family = SimplexNodalFamily("triangle", 2)
    rule = ReferenceTriangleRule(GaussLegendreRule(4)).materialize()
    mortar = serial_finite_element_mortar_plan(
        family.nodes,
        family.nodes,
        family.nodes,
        rule.points,
        rule.weights,
        facet_shape="triangle",
        declared_reproduction_degree=2,
        left_physical_coordinates=rule.points,
        right_physical_coordinates=rule.points,
        interface_id="triangle-mortar",
    )
    assert mortar.facet_shape == "triangle"
    assert mortar.evidence.declared_polynomials_reproduced
    flux = jnp.linspace(-0.4, 0.6, rule.points.shape[0])
    np.testing.assert_allclose(mortar.conservation_residual(flux), 0.0, atol=3.0e-12)


def test_entropy_reference_operators_close_generalized_sbp_identity():
    from phydrax.equations.fem._entropy_stability import (
        prepare_entropy_reference_operator,
    )
    from phydrax.integration import (
        GaussLegendreRule,
        ReferenceIntervalRule,
        ReferenceTriangleRule,
    )

    family = SimplexNodalFamily("triangle", 3)
    axis = GaussLegendreRule(6)
    operator = prepare_entropy_reference_operator(
        family.finite_element(),
        ReferenceTriangleRule(axis),
        (ReferenceIntervalRule(axis),) * 3,
        tolerance=3.0e-9,
    )
    assert operator.formal_sbp
    assert operator.minimum_mass_eigenvalue > 0.0
    assert operator.sbp_defect <= 3.0e-9
    assert operator.constant_defect <= 3.0e-9
