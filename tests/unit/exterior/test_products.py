#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.discretization as D
from phydrax.discretization._cell_complex import (
    cubical_cell_complex,
    simplicial_cell_complex,
    simplicial_cell_geometry,
)
from phydrax.discretization._cochain import CochainDiscretization
from phydrax.discretization._cochain_hodge import DiagonalHodge
from phydrax.discretization._cubical_whitney import CubicalSplineWhitneyKernel
from phydrax.discretization._structured_cochain import StructuredCochainBridge
from phydrax.discretization.fem._simplicial_whitney_chains import SimplicialWhitneyKernel
from phydrax.exterior._complex import DiscreteForm
from phydrax.exterior._form_type import FormType
from phydrax.exterior._products import (
    cochain_cup_product,
    interior_product,
    lie_derivative,
    whitney_wedge,
    WhitneyProductPlan,
)
from phydrax.topology._diagonals import serre_diagonal


def _bridge(count: int, /, *, periodic: bool = False) -> StructuredCochainBridge:
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(count, periodic=periodic),
            D.UniformCellAxisSpec(count, periodic=periodic),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=jnp.float64))
    return StructuredCochainBridge(grid)


def _plan(bridge: StructuredCochainBridge, /) -> WhitneyProductPlan:
    return WhitneyProductPlan(
        bridge.cochain, CubicalSplineWhitneyKernel(bridge, 1), quadrature_order=4
    )


def _form(plan: WhitneyProductPlan, degree: int, values: Array, /) -> DiscreteForm:
    return DiscreteForm(
        plan.complex.realization_id, FormType(plan.complex.dimension, degree), values
    )


def _constant_one(plan: WhitneyProductPlan, vector: Array, /) -> Array:
    vertices = plan.vertices[1]
    return (vertices[:, 1] - vertices[:, 0]) @ vector * plan.orientations[1]


def _tetrahedron_plan() -> WhitneyProductPlan:
    coordinates = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=jnp.float64,
    )
    levels = tuple(
        np.asarray(tuple(combinations(range(4), degree + 1)), dtype=np.int32)
        for degree in range(4)
    )
    topology = simplicial_cell_complex(levels)
    centers = tuple(jnp.mean(coordinates[jnp.asarray(level)], axis=1) for level in levels)
    complex = CochainDiscretization(
        topology,
        tuple(
            DiagonalHodge(jnp.ones((len(level),), dtype=jnp.float64)) for level in levels
        ),
        coordinates=centers,
    )
    mesh = D.CellMesh(
        coordinates,
        (
            D.CellBlock(
                "tet", "tetrahedron", jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)
            ),
        ),
    )
    finite_element = D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("u", D.lagrange_element("tetrahedron", 1))
    ).prepare()
    locator = D.PreparedSimplicialCellLocator(
        D.fem.prepare_finite_element_cell_map(finite_element, 0),
        coordinates,
        D.SimplicialLocationPolicy(1, 8, 4),
    )
    return WhitneyProductPlan(
        complex, SimplicialWhitneyKernel(complex, locator), quadrature_order=4
    )


@pytest.mark.parametrize("family", ("cubical", "simplicial"))
def test_whitney_constant_one_forms_integrate_their_geometric_wedge(family: str) -> None:
    plan = _plan(_bridge(3)) if family == "cubical" else _tetrahedron_plan()
    dimension = plan.complex.dimension
    dx = jnp.eye(dimension, dtype=jnp.float64)[0]
    dy = jnp.eye(dimension, dtype=jnp.float64)[1]
    a = _form(plan, 1, _constant_one(plan, dx))
    b = _form(plan, 1, _constant_one(plan, dy))
    result = whitney_wedge(plan, a, b)
    vertices = plan.vertices[2]
    first = vertices[:, 1] - vertices[:, 0]
    second = vertices[:, 2] - vertices[:, 0]
    signed_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    if not plan.simplex:
        # Cubical corner order is 00,01,10,11.
        signed_area = -signed_area
    else:
        signed_area = signed_area / 2
    expected = signed_area * plan.orientations[2]
    np.testing.assert_allclose(result.values, expected, atol=2e-12, rtol=2e-12)
    assert result.form_type.degree == 2


def test_whitney_wedge_is_graded_commutative_for_nonconstant_cochains() -> None:
    plan = _plan(_bridge(3))
    a = _form(
        plan,
        1,
        jax.random.normal(
            jax.random.key(31), (plan.complex.cell_counts[1],), dtype=jnp.float64
        ),
    )
    b = _form(
        plan,
        1,
        jax.random.normal(
            jax.random.key(32), (plan.complex.cell_counts[1],), dtype=jnp.float64
        ),
    )
    np.testing.assert_allclose(
        whitney_wedge(plan, a, b).values,
        -whitney_wedge(plan, b, a).values,
        rtol=2e-12,
        atol=2e-12,
    )


def _exponential_product_error(count: int, /) -> Array:
    plan = _plan(_bridge(count))
    nodes = plan.vertices[0][:, 0]
    scalar = _form(plan, 0, jnp.exp(nodes[:, 0]))
    dx = _form(plan, 1, _constant_one(plan, jnp.asarray((1.0, 0.0), dtype=jnp.float64)))
    actual = whitney_wedge(plan, scalar, dx).values
    edges = plan.vertices[1]
    expected = jnp.exp(edges[:, 1, 0]) - jnp.exp(edges[:, 0, 0])
    lengths = jnp.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    return jnp.sqrt(jnp.sum((actual - expected) ** 2 / lengths))


def test_whitney_product_converges_to_smooth_chain_integrals() -> None:
    coarse = _exponential_product_error(8)
    fine = _exponential_product_error(16)
    # There are O(h^-2) parallel edge samples in this two-dimensional norm.
    assert fine < coarse / 2.7
    assert coarse > 1e-5


def test_cartan_lie_derivative_matches_translation_of_x_dy() -> None:
    plan = _plan(_bridge(3))
    vertices = plan.vertices[1]
    values = vertices[:, 0, 0] * (vertices[:, 1, 1] - vertices[:, 0, 1])
    a = _form(plan, 1, values)
    vector = jnp.asarray((1.0, 0.0), dtype=jnp.float64)
    actual = lie_derivative(plan, vector, a)
    expected = _constant_one(plan, jnp.asarray((0.0, 1.0), dtype=jnp.float64))
    np.testing.assert_allclose(actual.values, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(interior_product(plan, vector, a).values, 0, atol=2e-12)


@pytest.mark.parametrize("left_degree", (0, 1))
def test_cup_product_satisfies_leibniz_with_the_incidence_derivative(
    left_degree: int,
) -> None:
    bridge = _bridge(3, periodic=True)
    plan = _plan(bridge)
    diagonals = serre_diagonal(cubical_cell_complex((3, 3), periodic=True))
    right_degree = 1 - left_degree
    a = _form(
        plan,
        left_degree,
        jax.random.normal(
            jax.random.key(4), (plan.complex.cell_counts[left_degree],), dtype=jnp.float64
        ),
    )
    b = _form(
        plan,
        right_degree,
        jax.random.normal(
            jax.random.key(5),
            (plan.complex.cell_counts[right_degree],),
            dtype=jnp.float64,
        ),
    )
    product = cochain_cup_product(plan.complex, a, b, diagonal=diagonals)
    da = _form(
        plan, left_degree + 1, plan.complex.exterior_derivative(left_degree, a.values)
    )
    db = _form(
        plan, right_degree + 1, plan.complex.exterior_derivative(right_degree, b.values)
    )
    expected = (
        cochain_cup_product(plan.complex, da, b, diagonal=diagonals).values
        + (-1) ** left_degree
        * cochain_cup_product(plan.complex, a, db, diagonal=diagonals).values
    )
    np.testing.assert_allclose(
        plan.complex.exterior_derivative(1, product.values),
        expected,
        atol=2e-12,
        rtol=2e-12,
    )


def test_torus_cup_product_has_nonzero_fundamental_period() -> None:
    plan = _plan(_bridge(3, periodic=True))
    diagonals = serre_diagonal(cubical_cell_complex((3, 3), periodic=True))
    a = _form(plan, 1, _constant_one(plan, jnp.asarray((1.0, 0.0), dtype=jnp.float64)))
    b = _form(plan, 1, _constant_one(plan, jnp.asarray((0.0, 1.0), dtype=jnp.float64)))
    product = cochain_cup_product(plan.complex, a, b, diagonal=diagonals)
    np.testing.assert_allclose(jnp.sum(product.values), 1, atol=2e-12)


def test_cup_product_is_associative_for_nonconstant_factors() -> None:
    plan = _plan(_bridge(2, periodic=True))
    diagonal = serre_diagonal(cubical_cell_complex((2, 2), periodic=True))
    a = _form(plan, 0, jnp.asarray((2.0, -1.0, 3.0, 4.0), dtype=jnp.float64))
    b = _form(plan, 1, jnp.arange(plan.complex.cell_counts[1], dtype=jnp.float64) + 1)
    c = _form(
        plan, 1, jnp.arange(plan.complex.cell_counts[1], dtype=jnp.float64) ** 2 - 2
    )
    ab = cochain_cup_product(plan.complex, a, b, diagonal=diagonal)
    bc = cochain_cup_product(plan.complex, b, c, diagonal=diagonal)
    left = cochain_cup_product(plan.complex, ab, c, diagonal=diagonal)
    right = cochain_cup_product(plan.complex, a, bc, diagonal=diagonal)
    np.testing.assert_allclose(left.values, right.values, atol=2e-12)


def test_semilagrangian_one_form_integrates_across_backtracked_cells() -> None:
    plan = _plan(_bridge(4, periodic=True))
    nodes = plan.vertices[0][:, 0]
    potential = jnp.sin(2 * jnp.pi * nodes[:, 0])
    a = _form(plan, 1, plan.complex.exterior_derivative(0, potential))
    vector = jnp.asarray((1.0, 0.0), dtype=jnp.float64)
    step = 0.37
    actual = lie_derivative(plan, vector, a, method="semi-lagrangian", step=step)
    endpoints = plan.vertices[1] - step * vector

    # Independent periodic piecewise-linear nodal interpolation, not a chain query.
    def interpolate(x: Array, /) -> Array:
        q = 4 * jnp.mod(x, 1)
        index = jnp.floor(q).astype(jnp.int32)
        alpha = q - index
        samples = jnp.sin(2 * jnp.pi * jnp.arange(4, dtype=jnp.float64) / 4)
        return (1 - alpha) * samples[index] + alpha * samples[(index + 1) % 4]

    pulled = interpolate(endpoints[:, 1, 0]) - interpolate(endpoints[:, 0, 0])
    np.testing.assert_allclose(
        actual.values, (a.values - pulled) / step, rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("degree", (0, 1, 2, 3))
def test_simplicial_semilagrangian_pullback_includes_chain_jacobian(degree: int) -> None:
    plan = _tetrahedron_plan()
    step = 0.1

    def dilation(point: Array, /) -> Array:
        return point

    if degree == 0:
        values = jnp.ones((plan.complex.cell_counts[0],), dtype=jnp.float64)
    elif degree == 1:
        values = _constant_one(plan, jnp.asarray((1.0, 2.0, -1.0), dtype=jnp.float64))
    else:
        rows, signs = simplicial_cell_geometry(plan.complex.topology)
        coordinates = (
            plan.kernel.locator.coordinates
            if isinstance(plan.kernel, SimplicialWhitneyKernel)
            else plan.vertices[0][:, 0]
        )
        cells = coordinates[jnp.asarray(rows[degree])]
        edges = cells[:, 1:] - cells[:, :1]
        if degree == 2:
            values = (
                (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
                / 2
                * jnp.asarray(signs[degree])
            )
        else:
            # The fixture is the positively oriented unit reference tetrahedron.
            values = jnp.asarray((1 / 6,), dtype=jnp.float64)
    a = _form(plan, degree, values)
    actual = lie_derivative(plan, dilation, a, method="semi-lagrangian", step=step)
    expected = (1 - (1 - step) ** degree) / step * values
    np.testing.assert_allclose(actual.values, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("dynamic_plan", (False, True))
def test_matrix_fiber_product_uses_noncommutative_matrix_multiplication(
    dynamic_plan: bool,
) -> None:
    plan = _plan(_bridge(2))
    left = jnp.asarray(((0.0, 1.0), (0.0, 0.0)), dtype=jnp.float64)
    right = jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=jnp.float64)
    dx = _constant_one(plan, jnp.asarray((1.0, 0.0), dtype=jnp.float64))
    dy = _constant_one(plan, jnp.asarray((0.0, 1.0), dtype=jnp.float64))

    def multiply(p: WhitneyProductPlan, a: Array, b: Array, /) -> Array:
        return p.wedge(a, b, 1, 1, product="matrix")

    def captured(a: Array, b: Array, /) -> Array:
        return multiply(plan, a, b)

    a, b = dx[:, None, None] * left, dy[:, None, None] * right
    if dynamic_plan:
        compiled = eqx.filter_jit(multiply)
        actual, reverse = compiled(plan, a, b), compiled(plan, b, a)
    else:
        constant_compiled = jax.jit(captured)
        actual, reverse = constant_compiled(a, b), constant_compiled(b, a)
    # Each square has area 1/4. E12 E21 = E11, whereas E21 E12 = E22.
    # Reversing dy ^ dx additionally changes the orientation sign.
    expected = np.broadcast_to(np.asarray(((0.25, 0.0), (0.0, 0.0))), (4, 2, 2))
    expected_reverse = np.broadcast_to(np.asarray(((0.0, 0.0), (0.0, -0.25))), (4, 2, 2))
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    np.testing.assert_allclose(reverse, expected_reverse, atol=2e-12)
    assert not jnp.allclose(actual, -reverse)


def test_matrix_wedge_compiled_plan_uses_differentiable_prepared_geometry() -> None:
    plan = _plan(_bridge(2))
    left = jnp.asarray(((0.0, 1.0), (0.0, 0.0)), dtype=jnp.float64)
    right = left.T
    dx = _constant_one(plan, jnp.asarray((1.0, 0.0), dtype=jnp.float64))
    dy = _constant_one(plan, jnp.asarray((0.0, 1.0), dtype=jnp.float64))
    a, b = dx[:, None, None] * left, dy[:, None, None] * right

    @eqx.filter_jit
    def multiply(p: WhitneyProductPlan, a: Array, b: Array, /) -> Array:
        return p.wedge(a, b, 1, 1, product="matrix")

    def blade_leaf(p: WhitneyProductPlan, /) -> Array:
        return p.blades[2]

    def coefficient_leaf(p: WhitneyProductPlan, /) -> Array:
        return p.reconstruction_queries[2][1].coefficients

    def area_action(scale: Array, /) -> Array:
        scaled = eqx.tree_at(blade_leaf, plan, plan.blades[2] * scale)
        return jnp.sum(multiply(scaled, a, b))

    value, derivative = jax.value_and_grad(area_action)(jnp.asarray(1.5))
    # The four constant dx ^ dy cells cover total oriented area one.
    np.testing.assert_allclose(value, 1.5, atol=2e-12)
    np.testing.assert_allclose(derivative, 1.0, atol=2e-12)
    coefficients = plan.reconstruction_queries[2][1].coefficients
    rescaled_query = eqx.tree_at(
        coefficient_leaf,
        plan,
        2 * coefficients,
    )
    expected = np.broadcast_to(np.asarray(((1.0, 0.0), (0.0, 0.0))), (4, 2, 2))
    np.testing.assert_allclose(multiply(rescaled_query, a, b), expected, atol=2e-12)


def test_matrix_wedge_compiled_plan_propagates_query_failure() -> None:
    plan = _plan(_bridge(2))
    values = jnp.ones((plan.complex.cell_counts[1], 2, 2), dtype=jnp.float64)

    def success_leaf(p: WhitneyProductPlan, /) -> Array:
        return p.reconstruction_queries[2][1].successful

    unsuccessful = eqx.tree_at(
        success_leaf,
        plan,
        jnp.zeros_like(plan.reconstruction_queries[2][1].successful),
    )

    @eqx.filter_jit
    def multiply(p: WhitneyProductPlan, a: Array, /) -> Array:
        return p.wedge(a, a, 1, 1, product="matrix")

    with pytest.raises(eqx.EquinoxRuntimeError):
        multiply(unsuccessful, values).block_until_ready()


def test_dual_placement_is_refused_even_when_coordinate_counts_match() -> None:
    plan = _plan(_bridge(3))
    values = jnp.ones((plan.complex.cell_counts[1],), dtype=jnp.float64)
    dual = DiscreteForm(
        plan.complex.realization_id, FormType(2, 1, twist="twisted"), values
    )
    with pytest.raises(ValueError):
        interior_product(plan, jnp.asarray((1.0, 0.0), dtype=jnp.float64), dual)


def test_relative_contraction_restricts_and_zero_extends_cell_coordinates() -> None:
    bridge = _bridge(4)
    plan = WhitneyProductPlan(
        bridge.cochain, CubicalSplineWhitneyKernel(bridge, 1), boundary="relative"
    )
    vector = jnp.asarray((0.0, 1.0), dtype=jnp.float64)
    values = _constant_one(plan, vector)
    poisoned = jnp.where(plan.complex.boundary_masks[1], 1e6, values)
    actual = interior_product(plan, vector, _form(plan, 1, poisoned)).values
    expected = (~plan.complex.boundary_masks[0]).astype(jnp.float64)
    np.testing.assert_allclose(actual, expected, atol=2e-12)


def test_simplicial_cartan_derivative_translates_an_angular_whitney_form() -> None:
    plan = _tetrahedron_plan()
    edges = plan.vertices[1]
    delta = edges[:, 1] - edges[:, 0]
    midpoint = jnp.mean(edges, axis=1)
    # x dy - y dx lies in the affine Whitney-1 space.
    values = (
        midpoint[:, 0] * delta[:, 1] - midpoint[:, 1] * delta[:, 0]
    ) * plan.orientations[1]
    vector = jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.float64)
    actual = lie_derivative(plan, vector, _form(plan, 1, values))
    expected = _constant_one(plan, jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.float64))
    np.testing.assert_allclose(actual.values, expected, rtol=2e-12, atol=2e-12)


def test_semilagrangian_zero_form_uses_exact_periodic_point_pullback() -> None:
    plan = _plan(_bridge(4, periodic=True))
    points = np.asarray(plan.vertices[0][:, 0])
    values = jnp.asarray(np.sin(2 * np.pi * points[:, 0]), dtype=jnp.float64)
    step = 0.37
    a = _form(plan, 0, values)
    vector = jnp.asarray((1.0, 0.0), dtype=jnp.float64)
    actual = lie_derivative(plan, vector, a, method="semi-lagrangian", step=step)
    nodes = np.arange(4, dtype=np.float64) / 4
    pulled = np.interp(points[:, 0] - step, nodes, np.sin(2 * np.pi * nodes), period=1)
    np.testing.assert_allclose(
        actual.values, (np.asarray(values) - pulled) / step, rtol=2e-12, atol=2e-12
    )


def test_bridge_product_preserves_bridge_scientific_ownership() -> None:
    bridge = _bridge(2)
    plan = WhitneyProductPlan(bridge, CubicalSplineWhitneyKernel(bridge, 1))
    a = _form(plan, 1, _constant_one(plan, jnp.asarray((1.0, 0.0), dtype=jnp.float64)))
    b = _form(plan, 1, _constant_one(plan, jnp.asarray((0.0, 1.0), dtype=jnp.float64)))
    product = whitney_wedge(bridge, a, b)
    np.testing.assert_allclose(product.values, 0.25, atol=2e-12)
    assert product.realization_id == bridge.realization_id
    cochain_view = DiscreteForm(bridge.cochain.realization_id, a.form_type, a.values)
    with pytest.raises(ValueError):
        whitney_wedge(bridge, cochain_view, b)
