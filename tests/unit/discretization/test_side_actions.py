#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared side actions: traces, dual pullbacks, fluxes, and impositions."""

from math import factorial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import (
    BoundaryImposition,
    FacetTraceRule,
    IntegrationDomain,
    PreparedTraceAction,
    SideActionDescriptor,
    SideGatherRoute,
)
from phydrax.linalg import (
    ArraySpace,
    DenseCholesky,
    DenseLinearOperator,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    OperatorPairing,
    OperatorProperties,
    prepare,
)


jax.config.update("jax_enable_x64", True)


def _monomial_integral(shape: str, exponents: tuple[int, ...]) -> float:
    """Exact integral of one monomial over the unit reference facet."""
    match shape:
        case "edge":
            return 1.0 / (exponents[0] + 1)
        case "quadrilateral":
            return 1.0 / ((exponents[0] + 1) * (exponents[1] + 1))
        case "triangle":
            first, second = exponents
            return factorial(first) * factorial(second) / factorial(first + second + 2)
    raise AssertionError(shape)


@pytest.mark.parametrize(
    ("family", "points", "shape"),
    (
        ("gauss-legendre", 3, "edge"),
        ("gauss-lobatto-legendre", 4, "edge"),
        ("gauss-legendre", 3, "quadrilateral"),
        ("gauss-lobatto-legendre", 3, "quadrilateral"),
        ("gauss-legendre", 4, "triangle"),
    ),
    ids=("gl-edge", "gll-edge", "gl-quadrilateral", "gll-quadrilateral", "gl-triangle"),
)
def test_facet_rule_is_exact_through_its_declared_degree(
    family: Any, points: int, shape: Any
) -> None:
    rule = FacetTraceRule(family, points=points)
    parameters, weights = rule.reference(shape)
    degree = rule.exact_degree(shape)
    assert degree is not None
    dimension = parameters.shape[1]

    for total in range(degree + 1):
        for first in range(total + 1):
            exponents = (first,) if dimension == 1 else (first, total - first)
            integrand = np.prod(parameters ** np.asarray(exponents), axis=1)
            np.testing.assert_allclose(
                weights @ integrand, _monomial_integral(shape, exponents), atol=1e-14
            )
    beyond = (degree + 1,) + (0,) * (dimension - 1)
    integrand = np.prod(parameters ** np.asarray(beyond), axis=1)
    assert abs(weights @ integrand - _monomial_integral(shape, beyond)) > 1e-8


def test_facet_rule_refuses_unsupported_families_and_evaluates_points() -> None:
    parameters, weights = FacetTraceRule(points=2).reference("point")

    assert parameters.shape == (1, 0) and weights.tolist() == [1.0]
    assert FacetTraceRule(points=2).exact_degree("point") is None
    with pytest.raises(ValueError, match="tensor rules"):
        FacetTraceRule("gauss-lobatto-legendre", points=3).reference("triangle")
    with pytest.raises(ValueError, match="two points"):
        FacetTraceRule("gauss-lobatto-legendre", points=1)


# A synthetic P1 boundary: three nodes on [0, 2] x {0}, two facets, two
# Gauss-Legendre sites per facet. The route follows the declared gather
# contract; every reference below is an analytic P1 integral.
_GAUSS = 0.5 + np.asarray((-1.0, 1.0)) / (2.0 * np.sqrt(3.0))


def _synthetic_trace(
    *,
    components: tuple[int, ...] = (),
    quantity: Any = "value",
    orientation: Any = "unoriented",
) -> PreparedTraceAction:
    domain = IntegrationDomain(
        "exterior_facet",
        np.arange(2),
        "synthetic-support",
        "synthetic-facets",
        owner_cells=np.asarray((0, 1)),
    )
    rule = FacetTraceRule(points=2)
    descriptor = SideActionDescriptor(
        owner_id="synthetic-owner",
        field_space_id="synthetic-field",
        quantity=quantity,
        representation="quadrature-values",
        orientation=orientation,
        approximation="exact",
        side="owner",
        domain=domain,
        revision_id="synthetic-revision",
        rule=rule,
        trace_degree=1,
        quadrature_exact_degree=rule.exact_degree("edge"),
    )
    hats = np.stack((1.0 - _GAUSS, _GAUSS), axis=-1)
    basis = np.broadcast_to(hats, (2, 2, 2))
    if quantity == "value":
        route = SideGatherRoute(
            ((0, 1), (1, 2)),
            basis,
            coefficient_shape=(3, *components),
            value_shape=components,
        )
    else:
        normal = np.asarray((0.0, -1.0))
        route = SideGatherRoute(
            ((0, 1), (1, 2)),
            basis[..., None] * normal,
            coefficient_shape=(3, 2),
            mode="contracted",
        )
    sites = np.stack((np.stack((_GAUSS, 1.0 + _GAUSS)), np.zeros((2, 2))), axis=-1)
    return PreparedTraceAction(
        descriptor,
        route,
        ArraySpace(route.coefficient_shape, dtype=np.float64),
        sites=sites,
        weights=np.full((2, 2), 0.5),
        normals=np.broadcast_to((0.0, -1.0), (2, 2, 2)),
        support_rows=np.arange(3),
    )


_P1_MASS = np.asarray(
    (
        (1.0 / 3.0, 1.0 / 6.0, 0.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        (0.0, 1.0 / 6.0, 1.0 / 3.0),
    )
)


def _mass_pairing() -> OperatorPairing:
    space = ArraySpace((3,), dtype=np.float64)
    riesz = DenseLinearOperator(
        jnp.asarray(_P1_MASS),
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    return OperatorPairing(
        riesz,
        prepared_inverse=prepare(
            LinearSystem(riesz),
            LinearSolvePolicy(DenseCholesky(), failure=FailurePolicy("error")),
        ),
    )


def test_trace_action_separates_pullback_measure_load_and_hilbert_adjoint() -> None:
    trace = _synthetic_trace()
    coefficients = jnp.asarray((1.0, -2.0, 0.5))
    sites = np.asarray(trace.sites)[..., 0]
    covector = jnp.asarray(np.random.default_rng(5).normal(size=(2, 2)))

    values = trace.apply(coefficients)
    load = trace.inject_load(jnp.asarray(sites))
    pullback = trace.dual_pullback(covector)
    adjoint = trace.hilbert_adjoint(_mass_pairing())

    np.testing.assert_allclose(values, np.interp(sites, (0.0, 1.0, 2.0), coefficients))
    # Work of the load density g(x) = x against the P1 hat functions on [0, 2].
    np.testing.assert_allclose(load, (1.0 / 6.0, 1.0, 5.0 / 6.0), atol=1e-14)
    np.testing.assert_allclose(
        jnp.vdot(values, covector), jnp.vdot(coefficients, pullback), atol=1e-14
    )
    np.testing.assert_allclose(
        trace.dual_pullback_operator().mv(covector), pullback, atol=1e-14
    )
    # The two-point rule integrates P1 x P1 exactly, so T* T is the identity
    # in the mass pairing while the Euclidean load pullback is not.
    np.testing.assert_allclose(adjoint.mv(values), coefficients, atol=1e-12)
    np.testing.assert_allclose(
        trace.inject_load(values), _P1_MASS @ np.asarray(coefficients), atol=1e-14
    )
    np.testing.assert_allclose(
        jnp.sum(0.5 * values * covector),
        coefficients @ _P1_MASS @ adjoint.mv(covector),
        atol=1e-12,
    )


def test_contracted_routes_trace_vector_components_against_the_normal() -> None:
    trace = _synthetic_trace(quantity="normal", orientation="outward")
    coefficients = jnp.asarray(((1.0, 2.0), (0.0, -1.0), (3.0, 4.0)))
    sites = np.asarray(trace.sites)[..., 0]
    covector = jnp.asarray(np.random.default_rng(11).normal(size=(2, 2)))

    values = trace.apply(coefficients)

    np.testing.assert_allclose(
        values, -np.interp(sites, (0.0, 1.0, 2.0), coefficients[:, 1]), atol=1e-14
    )
    np.testing.assert_allclose(
        jnp.vdot(values, covector),
        jnp.vdot(coefficients, trace.dual_pullback(covector)),
        atol=1e-14,
    )
    assert trace.value_shape == ()


def test_side_records_refuse_inconsistent_declarations() -> None:
    domain = IntegrationDomain(
        "exterior_facet",
        np.arange(2),
        "support",
        "facets",
        owner_cells=np.asarray((0, 1)),
    )
    common: dict[str, Any] = dict(
        owner_id="owner",
        field_space_id="field",
        representation="quadrature-values",
        approximation="exact",
        side="owner",
        domain=domain,
        revision_id="revision",
        rule=None,
        trace_degree=None,
        quadrature_exact_degree=None,
    )
    flux = SideActionDescriptor(quantity="conormal-flux", orientation="outward", **common)
    route = SideGatherRoute(((0, 1), (1, 2)), np.ones((2, 1, 2)), coefficient_shape=(3,))
    arrays: dict[str, Any] = dict(
        sites=np.zeros((2, 1, 2)),
        weights=np.ones((2, 1)),
        normals=np.zeros((2, 1, 2)),
        support_rows=np.arange(3),
    )

    with pytest.raises(ValueError, match="declare its normal orientation"):
        SideActionDescriptor(quantity="conormal-flux", orientation="unoriented", **common)
    with pytest.raises(ValueError, match="compiled physics owners"):
        PreparedTraceAction(flux, route, ArraySpace((3,)), **arrays)
    value = SideActionDescriptor(quantity="value", orientation="unoriented", **common)
    with pytest.raises(ValueError, match="weights positive"):
        PreparedTraceAction(
            value, route, ArraySpace((3,)), **{**arrays, "weights": -np.ones((2, 1))}
        )
    with pytest.raises(ValueError, match="sorted, unique"):
        PreparedTraceAction(
            value, route, ArraySpace((3,)), **{**arrays, "support_rows": (2, 0)}
        )
    with pytest.raises(ValueError, match="outside the coefficient rows"):
        SideGatherRoute(((0, 3),), np.ones((1, 1, 2)), coefficient_shape=(3,))
    cells: dict[str, Any] = {
        **common,
        "domain": IntegrationDomain("cell", np.zeros(1, np.int32), "support", "cells"),
    }
    with pytest.raises(ValueError, match="exterior or interior facets"):
        SideActionDescriptor(quantity="value", orientation="unoriented", **cells)


def test_boundary_impositions_locate_their_rows_or_facets() -> None:
    trace = _synthetic_trace()
    strong = BoundaryImposition(
        "strong", field_space_id="synthetic-field", source_id="dirichlet", rows=(2,)
    )
    natural = BoundaryImposition(
        "natural",
        field_space_id="synthetic-field",
        source_id="load",
        entity_set_id="synthetic-facets",
        facets=(4, 5),
    )
    elsewhere = BoundaryImposition(
        "strong", field_space_id="other-field", source_id="dirichlet", rows=(0, 1)
    )

    assert strong.overlaps(trace)
    assert not natural.overlaps(trace)
    assert not elsewhere.overlaps(trace)
    with pytest.raises(ValueError, match="constrained rows"):
        BoundaryImposition(
            "strong",
            field_space_id="field",
            source_id="dirichlet",
            entity_set_id="facets",
            facets=(0,),
        )
    with pytest.raises(ValueError, match="entity_set_id"):
        BoundaryImposition("natural", field_space_id="f", source_id="s", facets=(0,))


# --- Finite elements ---
# Unit-square meshes; every reference is an analytic polynomial, normal, or
# boundary integral evaluated on the host independently of the trace routes.


def _fe_square(kind: str, cells_per_side: int) -> Any:
    axis = np.linspace(0.0, 1.0, cells_per_side + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)
    cells: list[tuple[int, ...]] = []
    for row in range(cells_per_side):
        for column in range(cells_per_side):
            first = row * (cells_per_side + 1) + column
            second, third = first + 1, first + cells_per_side + 2
            fourth = first + cells_per_side + 1
            if kind == "triangle":
                cells += [(first, second, third), (first, third, fourth)]
            else:
                cells.append((first, second, third, fourth))
    block = phx.discretization.CellBlock("cells", kind, np.asarray(cells))
    return phx.discretization.CellMesh(points, (block,))


def _fe_discretization(
    kind: str,
    element: Any,
    *,
    components: tuple[int, ...] = (),
    cells_per_side: int = 3,
) -> Any:
    field = phx.discretization.FiniteElementFieldSpec(
        "u", element, component_shape=components
    )
    return phx.discretization.FiniteElementPlan(
        _fe_square(kind, cells_per_side), field
    ).prepare()


def _square_normals(sites: np.ndarray) -> np.ndarray:
    """Analytic outward unit-square normals of each facet, from its mean site."""
    center = sites.mean(axis=1)
    normals = np.zeros(center.shape)
    normals[np.isclose(center[:, 1], 0.0), 1] = -1.0
    normals[np.isclose(center[:, 1], 1.0), 1] = 1.0
    normals[np.isclose(center[:, 0], 0.0), 0] = -1.0
    normals[np.isclose(center[:, 0], 1.0), 0] = 1.0
    return np.broadcast_to(normals[:, None, :], sites.shape)


def _square_boundary_integral(integrand: Any) -> float:
    """Gauss--Legendre integral of `integrand(points)` over the unit square edges."""
    nodes, weights = np.polynomial.legendre.leggauss(8)
    t = 0.5 * (nodes + 1.0)
    edges = (
        np.stack((t, 0.0 * t), -1),
        np.stack((t, 1.0 + 0.0 * t), -1),
        np.stack((0.0 * t, t), -1),
        np.stack((1.0 + 0.0 * t, t), -1),
    )
    return float(sum(0.5 * weights @ integrand(points) for points in edges))


def _fe_polynomial(points: Any, degree: int) -> Any:
    x, y = points[..., 0], points[..., 1]
    return x**degree - 0.5 * y**degree + x * y + 0.25


_FE_VALUE_CASES = (
    ("triangle", 1, "gauss-legendre"),
    ("triangle", 2, "gauss-legendre"),
    ("triangle", 3, "gauss-legendre"),
    ("quadrilateral", 3, "gauss-lobatto-legendre"),
)


@pytest.mark.parametrize(
    ("kind", "degree", "family"),
    _FE_VALUE_CASES,
    ids=("p1-triangle", "p2-triangle", "p3-triangle", "gll-q3-quadrilateral"),
)
def test_fe_exterior_value_trace_is_exact_with_physical_measure_and_normals(
    kind: str, degree: int, family: Any
) -> None:
    discretization = _fe_discretization(
        kind, phx.discretization.lagrange_element(kind, degree)
    )
    coefficients = discretization.project(
        "u", lambda points, args: _fe_polynomial(points, degree)
    )
    domain = discretization.exterior_facet_domain
    trace = discretization.prepare_side_trace(
        "u", domain, rule=FacetTraceRule(family, points=4)
    )
    sites = np.asarray(trace.sites)
    weights = np.asarray(trace.weights)
    normals = np.asarray(trace.normals)

    np.testing.assert_allclose(
        trace.apply(coefficients), _fe_polynomial(sites, degree), atol=1e-13
    )
    # int_{boundary} (x^2 + y) ds = 1/3 + 4/3 + 1/2 + 3/2 on the unit square.
    np.testing.assert_allclose(
        np.sum(weights * (sites[..., 0] ** 2 + sites[..., 1])), 11.0 / 3.0, rtol=1e-13
    )
    np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1.0, rtol=1e-14)
    np.testing.assert_allclose(normals, _square_normals(sites), atol=1e-14)
    assert trace.descriptor.orientation == "unoriented"
    assert trace.descriptor.trace_degree == degree
    assert trace.descriptor.owner_id == discretization.prepared_id
    assert trace.descriptor.field_space_id == (
        discretization.field_spaces[0].field_space_id
    )
    np.testing.assert_array_equal(trace.descriptor.facets, domain.entity_indices)


def test_fe_trace_route_is_local_to_facets() -> None:
    degree = 3
    discretization = _fe_discretization(
        "triangle",
        phx.discretization.lagrange_element("triangle", degree),
        cells_per_side=6,
    )
    trace = discretization.prepare_side_trace(
        "u", discretization.exterior_facet_domain, rule=FacetTraceRule(points=4)
    )
    facets = trace.descriptor.facets.shape[0]
    rows = trace.coefficient_space.shape[0]
    route = trace.route
    assert isinstance(route, SideGatherRoute)

    # One edge closure holds degree + 1 DOFs; nothing scales with the rows.
    assert route.dofs.shape == (facets, degree + 1)
    assert route.weights.shape == (facets, 4, degree + 1)
    assert route.weights.size < rows * trace.output_shape[1]
    # The boundary closure rows of a P3 square boundary: 24 edges, 3 DOFs each.
    assert trace.support_rows.shape == (facets * degree,)


@pytest.mark.parametrize(
    ("element", "continuous"),
    (
        (phx.discretization.lagrange_element("triangle", 2), True),
        (phx.discretization.discontinuous_element("triangle", 1), False),
    ),
    ids=("h1-p2", "l2-p1"),
)
def test_fe_interior_traces_share_sites_continuous_h1_and_jumping_l2(
    element: Any, continuous: bool
) -> None:
    discretization = _fe_discretization("triangle", element)
    domain = discretization.interior_facet_domain
    rule = FacetTraceRule(points=3)
    owner = discretization.prepare_side_trace("u", domain, rule=rule)
    neighbor = discretization.prepare_side_trace("u", domain, rule=rule, side="neighbor")
    coefficients = jnp.asarray(
        np.random.default_rng(7).normal(
            size=discretization.field_spaces[0].vector_space.shape
        )
    )
    jump = np.max(np.abs(owner.apply(coefficients) - neighbor.apply(coefficients)))

    np.testing.assert_array_equal(owner.sites, neighbor.sites)
    np.testing.assert_allclose(owner.weights, neighbor.weights, rtol=1e-14)
    np.testing.assert_allclose(neighbor.normals, -owner.normals, atol=1e-14)
    assert neighbor.descriptor.side == "neighbor"
    if continuous:
        assert jump < 1e-12
    else:
        assert jump > 1e-2


def test_fe_multiblock_traces_pad_mixed_cell_blocks() -> None:
    axis_x, axis_y = np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 3)
    points = np.stack(np.meshgrid(axis_x, axis_y, indexing="xy"), -1).reshape(-1, 2)
    triangles, quadrilaterals = [], []
    for row in range(2):
        for column in range(4):
            first, second = row * 5 + column, row * 5 + column + 1
            third, fourth = second + 5, first + 5
            if column < 2:
                triangles += [(first, second, third), (first, third, fourth)]
            else:
                quadrilaterals.append((first, second, third, fourth))
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "triangles",
                "triangle",
                np.asarray(triangles),
                global_ids=np.arange(len(triangles)),
            ),
            phx.discretization.CellBlock(
                "quadrilaterals",
                "quadrilateral",
                np.asarray(quadrilaterals),
                global_ids=len(triangles) + np.arange(len(quadrilaterals)),
            ),
        ),
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            "triangles": phx.discretization.lagrange_element("triangle", 2),
            "quadrilaterals": phx.discretization.lagrange_element("quadrilateral", 2),
        },
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    coefficients = discretization.project(
        "u", lambda points, args: _fe_polynomial(points, 2)
    )
    rule = FacetTraceRule(points=3)
    exterior = discretization.prepare_side_trace(
        "u", discretization.exterior_facet_domain, rule=rule
    )
    interior = discretization.interior_facet_domain
    owner = discretization.prepare_side_trace("u", interior, rule=rule)
    neighbor = discretization.prepare_side_trace(
        "u", interior, rule=rule, side="neighbor"
    )
    random = jnp.asarray(np.random.default_rng(2).normal(size=coefficients.shape))

    np.testing.assert_allclose(
        exterior.apply(coefficients),
        _fe_polynomial(np.asarray(exterior.sites), 2),
        atol=1e-13,
    )
    np.testing.assert_allclose(jnp.sum(exterior.weights), 4.0, rtol=1e-14)
    np.testing.assert_allclose(owner.apply(random), neighbor.apply(random), atol=1e-12)
    assert exterior.descriptor.trace_degree == 2


def test_fe_trace_load_pairing_and_mass_hilbert_adjoint() -> None:
    discretization = _fe_discretization(
        "triangle", phx.discretization.lagrange_element("triangle", 2)
    )
    trace = discretization.prepare_side_trace(
        "u", discretization.exterior_facet_domain, rule=FacetTraceRule(points=3)
    )

    def field(points: Any) -> Any:
        return points[..., 0] ** 2 + points[..., 1]

    def load(points: Any) -> Any:
        return 1.0 + points[..., 0]

    coefficients = discretization.project("u", lambda points, args: field(points))
    density = jnp.asarray(load(np.asarray(trace.sites)))
    work = jnp.sum(trace.weights * density * trace.apply(coefficients))

    np.testing.assert_allclose(
        work, jnp.vdot(coefficients, trace.inject_load(density)), rtol=1e-13
    )
    np.testing.assert_allclose(
        work,
        _square_boundary_integral(lambda points: load(points) * field(points)),
        rtol=1e-13,
    )

    mass = jax.vmap(discretization.mass.mv)(jnp.eye(coefficients.shape[0]))
    space = ArraySpace(coefficients.shape, dtype=np.float64)
    riesz = DenseLinearOperator(
        mass,
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    pairing = OperatorPairing(
        riesz,
        prepared_inverse=prepare(
            LinearSystem(riesz),
            LinearSolvePolicy(DenseCholesky(), failure=FailurePolicy("error")),
        ),
    )
    covector = jnp.asarray(np.random.default_rng(3).normal(size=trace.output_shape))
    adjoint = trace.hilbert_adjoint(pairing).mv(covector)

    np.testing.assert_allclose(
        jnp.sum(trace.weights * trace.apply(coefficients) * covector),
        coefficients @ mass @ adjoint,
        rtol=1e-11,
    )
    assert not np.allclose(adjoint, trace.dual_pullback(covector))


def test_fe_vector_normal_and_tangential_traces_contract_the_outward_normal() -> None:
    discretization = _fe_discretization(
        "triangle",
        phx.discretization.lagrange_element("triangle", 2),
        components=(2,),
    )

    def field(points: Any) -> Any:
        x, y = points[..., 0], points[..., 1]
        return np.stack((x**2 + y, x * y - 1.0), axis=-1)

    coefficients = discretization.project(
        "u", lambda points, args: jnp.asarray(field(np.asarray(points)))
    )
    rule = FacetTraceRule(points=3)
    domain = discretization.exterior_facet_domain
    normal = discretization.prepare_side_trace("u", domain, rule=rule, quantity="normal")
    tangential = discretization.prepare_side_trace(
        "u", domain, rule=rule, quantity="tangential"
    )
    values = field(np.asarray(normal.sites))
    normals = _square_normals(np.asarray(normal.sites))
    tangents = np.stack((-normals[..., 1], normals[..., 0]), axis=-1)

    np.testing.assert_allclose(
        normal.apply(coefficients), np.sum(values * normals, axis=-1), atol=1e-13
    )
    np.testing.assert_allclose(
        tangential.apply(coefficients), np.sum(values * tangents, axis=-1), atol=1e-13
    )
    assert normal.descriptor.orientation == "outward"
    assert normal.value_shape == () and tangential.value_shape == ()


def _bottom_and_side_domains(discretization: Any) -> tuple[Any, Any]:
    exterior = discretization.exterior_facet_domain
    trace = discretization.prepare_side_trace(
        "u", exterior, rule=FacetTraceRule(points=2)
    )
    on_bottom = np.all(np.isclose(np.asarray(trace.sites)[..., 1], 0.0), axis=1)
    facets = np.asarray(exterior.entity_indices)
    entities = discretization.mesh.topology.entity_sets[1]
    domains = []
    for selected in (on_bottom, ~on_bottom):
        mask = np.zeros((entities.count,), dtype=np.bool_)
        mask[facets[selected]] = True
        domains.append(
            discretization.integration_domain(
                "exterior_facet",
                phx.discretization.EntitySelection(entities, mask),
            )
        )
    return domains[0], domains[1]


def _poisson_solution(points: Any) -> Any:
    x, y = points[..., 0], points[..., 1]
    return x**2 + 3.0 * x * y + 2.0 * y**2 + x


def _poisson_gradient(points: Any) -> Any:
    x, y = points[..., 0], points[..., 1]
    return jnp.stack((2.0 * x + 3.0 * y + 1.0, 3.0 * x + 4.0 * y), axis=-1)


def _neumann_data(points: Any, args: Any) -> Any:
    normal = jnp.where(
        jnp.isclose(points[..., 1], 1.0)[..., None],
        jnp.asarray((0.0, 1.0)),
        jnp.where(
            jnp.isclose(points[..., 0], 1.0)[..., None],
            jnp.asarray((1.0, 0.0)),
            jnp.asarray((-1.0, 0.0)),
        ),
    )
    return jnp.sum(_poisson_gradient(points) * normal, axis=-1)


def _bottom_reaction_reference(coordinates: np.ndarray) -> np.ndarray:
    """int_{y=0} (du/dn) phi_i dx with du/dn = -3x and 1-D quadratic P2 hats."""
    nodes, weights = np.polynomial.legendre.leggauss(6)
    reference = np.zeros((coordinates.shape[0],))
    on_bottom = np.isclose(coordinates[:, 1], 0.0)
    vertices = np.unique(coordinates[on_bottom, 0])[::2]
    for start, stop in zip(vertices[:-1], vertices[1:], strict=True):
        middle = 0.5 * (start + stop)
        x = middle + 0.5 * (stop - start) * nodes
        local = (start, middle, stop)
        for index, node in enumerate(local):
            hat = np.prod(
                [(x - other) / (node - other) for other in local if other != node],
                axis=0,
            )
            row = np.flatnonzero(on_bottom & np.isclose(coordinates[:, 0], node))[0]
            reference[row] += 0.5 * (stop - start) * weights @ (-3.0 * x * hat)
    return reference


def test_fe_reaction_flux_matches_the_conormal_integral_on_dirichlet_rows() -> None:
    discretization = _fe_discretization(
        "triangle", phx.discretization.lagrange_element("triangle", 2)
    )
    bottom, sides = _bottom_and_side_domains(discretization)
    coordinates = np.asarray(discretization.dof_maps[0].dof_coordinates)
    dirichlet = phx.discretization.dirichlet_constraint(
        discretization, "u", boundary_mask=np.isclose(coordinates[:, 1], 0.0)
    )
    # -Laplace(u) = -6 with Neumann data du/dn on the three other sides.
    form = phx.equations.FiniteElementForm(
        "manufactured-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u"),
            phx.equations.SourceAction("u", -6.0),
            phx.equations.BoundaryLoadAction(
                "u",
                phx.equations.coefficient(_neumann_data, coefficient_id="neumann"),
                domain=sides,
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form, discretization, constraint=dirichlet, dirichlet_values=_poisson_solution
    )
    system, rhs = compiled.linear_system()
    result = phx.linalg.solve(
        system,
        rhs,
        policy=LinearSolvePolicy(
            phx.linalg.GMRES(),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1e-13, absolute=1e-14, max_steps=500
            ),
        ),
    )
    state = compiled.expand(result.value)
    trace = discretization.prepare_side_trace("u", bottom, rule=FacetTraceRule(points=3))
    flux = compiled.prepare_conormal_flux(trace)
    reference = _bottom_reaction_reference(coordinates)

    assert bool(result.successful)
    np.testing.assert_allclose(state, _poisson_solution(coordinates), atol=1e-10)
    np.testing.assert_allclose(
        flux.evaluate(state), reference[np.asarray(trace.support_rows)], atol=1e-10
    )
    assert flux.descriptor.representation == "residual-reaction"
    assert flux.descriptor.approximation == "variational-reaction"
    assert flux.descriptor.orientation == "outward"
    assert flux.descriptor.revision_id == trace.descriptor.revision_id

    strong, natural = compiled.boundary_impositions()
    assert strong.kind == "strong" and natural.kind == "natural"
    np.testing.assert_array_equal(
        strong.rows, np.flatnonzero(np.isclose(coordinates[:, 1], 0.0))
    )
    np.testing.assert_array_equal(natural.facets, np.sort(sides.entity_indices))
    assert natural.entity_set_id == sides.entity_set_id
    assert strong.overlaps(trace)
    assert not natural.overlaps(trace)


def test_fe_side_traces_and_reactions_refuse_unsupported_requests() -> None:
    discretization = _fe_discretization(
        "triangle", phx.discretization.lagrange_element("triangle", 2)
    )
    rule = FacetTraceRule(points=2)
    exterior = discretization.exterior_facet_domain
    interior = discretization.interior_facet_domain
    other = _fe_discretization(
        "quadrilateral", phx.discretization.lagrange_element("quadrilateral", 1)
    )
    compatible = _fe_discretization(
        "triangle", phx.discretization.raviart_thomas_element("triangle")
    )
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "diffusion", "u", (phx.equations.DiffusionAction("u"),)
        ),
        discretization,
    )

    with pytest.raises(ValueError, match="no neighbor side"):
        discretization.prepare_side_trace("u", exterior, rule=rule, side="neighbor")
    with pytest.raises(ValueError, match="compose the owner and neighbor"):
        discretization.prepare_side_trace("u", interior, rule=rule, side="average")
    with pytest.raises(ValueError, match="Piola-mapped"):
        compatible.prepare_side_trace("u", compatible.exterior_facet_domain, rule=rule)
    with pytest.raises(ValueError, match="not produced by this finite-element"):
        discretization.prepare_side_trace("u", other.exterior_facet_domain, rule=rule)
    with pytest.raises(ValueError, match="vector field"):
        discretization.prepare_side_trace("u", exterior, rule=rule, quantity="normal")
    with pytest.raises(ValueError, match="exterior facets"):
        compiled.prepare_conormal_flux(
            discretization.prepare_side_trace("u", interior, rule=rule)
        )
    with pytest.raises(ValueError, match="another discretization"):
        compiled.prepare_conormal_flux(
            other.prepare_side_trace("u", other.exterior_facet_domain, rule=rule)
        )


# --- Explicit polygon H1 and virtual elements ---

_POLYGON_COORDINATES = np.asarray(
    (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.55, 0.5),
        (0.5, 1.0),
    )
)
# A pentagon, a triangle, and a quadrilateral tiling the unit square.
_POLYGON_CELLS = tuple(
    np.asarray(loop, dtype=np.int32)
    for loop in ((0, 4, 5, 6, 3), (4, 1, 5), (1, 2, 6, 5))
)


def _polygon_mesh() -> Any:
    return phx.discretization.CellMesh.from_polygons(
        jnp.asarray(_POLYGON_COORDINATES), _POLYGON_CELLS
    )


def _vem_space(element: Any) -> Any:
    return phx.discretization.VirtualElementPlan(
        _polygon_mesh(), phx.discretization.VirtualElementFieldSpec("u", element)
    ).prepare()


def _cell_loops(mesh: Any) -> list[np.ndarray]:
    """Counter-clockwise vertex loops in global cell order (blocks by arity)."""
    return [loop for block in mesh.blocks for loop in np.asarray(block.vertices)]


def _convex_polygon_mean(vertices: np.ndarray, function: Any) -> Any:
    """Mean over a convex polygon; the edge-midpoint rule is exact for quadratics."""
    total = 0.0
    area = 0.0
    for index in range(1, vertices.shape[0] - 1):
        a, b, c = vertices[0], vertices[index], vertices[index + 1]
        measure = 0.5 * abs((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0])
        midpoints = np.stack((0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)))
        total = total + measure * np.mean(
            function(midpoints[:, 0], midpoints[:, 1]), axis=0
        )
        area += measure
    return total / area


def _vem_quadratic_dofs(mesh: Any, function: Any) -> Any:
    """Degree-2 H1 VEM DOFs: vertex values, edge midpoints, and cell means."""
    points = _POLYGON_COORDINATES
    edges = np.asarray(mesh.connectivity.edges)
    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    means = [_convex_polygon_mean(points[loop], function) for loop in _cell_loops(mesh)]
    return jnp.asarray(
        np.concatenate(
            (
                function(points[:, 0], points[:, 1]),
                function(midpoints[:, 0], midpoints[:, 1]),
                np.asarray(means),
            )
        )
    )


def _vem_moment_dofs(mesh: Any, field: Any, quantity: str) -> Any:
    """Degree-1 H(div)/H(curl) DOFs of a vector field.

    Edge DOFs are Legendre moments `int_0^1 (v . d)(x(s)) P_m(2s - 1) ds` along
    the edge from its lower to its higher vertex index, with `d` the tangent
    rotated clockwise (normal traces) or the tangent itself (tangential
    traces); cell DOFs are the component means.
    """
    points = _POLYGON_COORDINATES
    edges = np.asarray(mesh.connectivity.edges)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    s = 0.5 * (nodes + 1.0)
    moments = []
    for start, stop in points[edges]:
        tangent = (stop - start) / np.linalg.norm(stop - start)
        direction = (
            np.asarray((tangent[1], -tangent[0])) if quantity == "normal" else tangent
        )
        samples = field(start[None, :] + s[:, None] * (stop - start)[None, :]) @ direction
        legendre = np.stack((np.ones_like(s), 2.0 * s - 1.0))
        moments.append(legendre @ (0.5 * weights * samples))
    means = [
        _convex_polygon_mean(points[loop], lambda x, y: field(np.stack((x, y), axis=-1)))
        for loop in _cell_loops(mesh)
    ]
    return jnp.asarray(np.concatenate((np.ravel(moments), np.ravel(means))))


def _outward_normals(mesh: Any, facets: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Unit normals of each facet pointing away from the (convex) side cell."""
    loops = _cell_loops(mesh)
    edges = np.asarray(mesh.connectivity.edges)[facets]
    normals = []
    for (start, stop), cell in zip(_POLYGON_COORDINATES[edges], cells, strict=True):
        tangent = stop - start
        normal = np.asarray((tangent[1], -tangent[0])) / np.linalg.norm(tangent)
        centroid = np.mean(_POLYGON_COORDINATES[loops[cell]], axis=0)
        normals.append(
            normal if normal @ (0.5 * (start + stop) - centroid) > 0 else -normal
        )
    return np.asarray(normals)


def _canonical_parameters(mesh: Any, facets: np.ndarray, sites: np.ndarray) -> Any:
    """Edge endpoints (lower, higher vertex index) and each site's parameter."""
    edges = np.asarray(mesh.connectivity.edges)[facets]
    start = _POLYGON_COORDINATES[edges[:, 0]]
    stop = _POLYGON_COORDINATES[edges[:, 1]]
    tangent = stop - start
    s = (
        np.sum((sites - start[:, None]) * tangent[:, None], axis=-1)
        / np.sum(tangent * tangent, axis=-1)[:, None]
    )
    np.testing.assert_allclose(start[:, None] + s[..., None] * tangent[:, None], sites)
    return edges, s


def test_explicit_polygon_h1_edge_trace_is_linear_and_pulls_back_through_the_measure() -> (
    None
):
    space = phx.discretization.ExplicitPolygonH1Plan(
        _polygon_mesh(), phx.discretization.ExplicitPolygonH1FieldSpec("u")
    ).prepare()
    state = np.random.default_rng(2).normal(size=7)
    trace = space.prepare_side_trace(
        "u", space.exterior_facet_domain, rule=FacetTraceRule(points=3)
    )
    facets = np.asarray(trace.descriptor.facets)
    edges, s = _canonical_parameters(space.mesh, facets, np.asarray(trace.sites))
    lengths = np.linalg.norm(
        _POLYGON_COORDINATES[edges[:, 1]] - _POLYGON_COORDINATES[edges[:, 0]], axis=1
    )
    _, gauss = np.polynomial.legendre.leggauss(3)

    assert trace.descriptor.approximation == "exact"
    assert trace.descriptor.orientation == "unoriented"
    assert trace.descriptor.trace_degree == 1
    assert trace.support_rows.tolist() == [0, 1, 2, 3, 4, 6]
    np.testing.assert_allclose(
        trace.apply(jnp.asarray(state)),
        (1.0 - s) * state[edges[:, 0], None] + s * state[edges[:, 1], None],
        atol=1e-14,
    )
    np.testing.assert_allclose(trace.weights, 0.5 * lengths[:, None] * gauss, atol=1e-15)
    np.testing.assert_allclose(
        trace.normals[:, 0],
        _outward_normals(
            space.mesh, facets, np.asarray(space.exterior_facet_domain.owner_cells)
        ),
        atol=1e-14,
    )

    density = np.random.default_rng(3).normal(size=trace.output_shape)
    work = np.sum(np.asarray(trace.weights) * density * np.asarray(trace.apply(state)))
    np.testing.assert_allclose(state @ trace.inject_load(density), work, atol=1e-12)
    np.testing.assert_allclose(
        state @ trace.dual_pullback(density),
        np.sum(density * np.asarray(trace.apply(state))),
        atol=1e-12,
    )
    # A unit load puts half of every incident boundary edge on each vertex.
    expected = np.zeros(7)
    np.add.at(expected, edges[:, 0], 0.5 * lengths)
    np.add.at(expected, edges[:, 1], 0.5 * lengths)
    np.testing.assert_allclose(
        trace.inject_load(np.ones(trace.output_shape)), expected, atol=1e-14
    )


def test_explicit_polygon_h1_interior_sides_share_sites_and_oppose_normals() -> None:
    space = phx.discretization.ExplicitPolygonH1Plan(
        _polygon_mesh(), phx.discretization.ExplicitPolygonH1FieldSpec("u")
    ).prepare()
    domain = space.interior_facet_domain
    rule = FacetTraceRule("gauss-lobatto-legendre", points=3)
    owner = space.prepare_side_trace("u", domain, rule=rule)
    neighbor = space.prepare_side_trace("u", domain, rule=rule, side="neighbor")
    state = jnp.asarray(np.random.default_rng(4).normal(size=7))
    facets = np.asarray(domain.entity_indices)

    np.testing.assert_allclose(owner.sites, neighbor.sites, atol=0.0)
    np.testing.assert_allclose(owner.apply(state), neighbor.apply(state), atol=1e-14)
    np.testing.assert_allclose(
        neighbor.normals[:, 0],
        _outward_normals(space.mesh, facets, np.asarray(domain.neighbor_cells)),
        atol=1e-14,
    )
    np.testing.assert_allclose(owner.normals, -neighbor.normals, atol=1e-14)
    assert owner.descriptor.revision_id == neighbor.descriptor.revision_id
    assert owner.descriptor.quadrature_exact_degree == 3


def test_virtual_element_value_trace_is_exact_and_distinct_from_the_projection() -> None:
    space = _vem_space(phx.discretization.conforming_h1_virtual_element(2))
    trace = space.prepare_side_trace(
        "u", space.exterior_facet_domain, rule=FacetTraceRule(points=4)
    )
    facets = np.asarray(trace.descriptor.facets)
    sites = np.asarray(trace.sites)
    edges, s = _canonical_parameters(space.mesh, facets, sites)
    start = _POLYGON_COORDINATES[edges[:, 0]]
    stop = _POLYGON_COORDINATES[edges[:, 1]]

    # The cell means of this smooth function are only approximate; traces use
    # the vertex and edge DOFs alone.
    def smooth(x: Any, y: Any) -> Any:
        return np.sin(2.0 * x + y) + np.cos(3.0 * y)

    state = _vem_quadratic_dofs(space.mesh, smooth)
    # Quadratic Lagrange interpolation on the Gauss-Lobatto nodes s = 0, 1/2, 1.
    nodal = np.stack(
        (
            smooth(start[:, 0], start[:, 1]),
            smooth(*(0.5 * (start + stop)).T),
            smooth(stop[:, 0], stop[:, 1]),
        ),
        axis=1,
    )
    lagrange = np.stack(
        (2.0 * (s - 0.5) * (s - 1.0), 4.0 * s * (1.0 - s), 2.0 * s * (s - 0.5)), axis=-1
    )
    expected = np.sum(lagrange * nodal[:, None, :], axis=-1)
    projection = phx.equations.vem.prepare_virtual_element_field_reconstruction(
        space, channel="h1-projection"
    )
    projected = projection.prepare_query(sites.reshape((-1, 2))).apply(state)

    assert trace.descriptor.approximation == "exact"
    assert trace.descriptor.trace_degree == 2
    assert projection.approximation == "h1-projection"
    np.testing.assert_allclose(trace.apply(state), expected, atol=1e-13)
    assert np.max(np.abs(np.asarray(projected) - expected.ravel())) > 1e-3

    def quadratic(x: Any, y: Any) -> Any:
        return 1.0 + x - 2.0 * y + 0.5 * x**2 + x * y - y**2

    polynomial = _vem_quadratic_dofs(space.mesh, quadratic)
    exact = quadratic(sites[..., 0], sites[..., 1])
    np.testing.assert_allclose(trace.apply(polynomial), exact, atol=1e-13)
    np.testing.assert_allclose(
        projection.prepare_query(sites.reshape((-1, 2))).apply(polynomial),
        exact.ravel(),
        atol=1e-11,
    )


def _linear_vector_field(points: np.ndarray) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    return np.stack((0.3 + 1.2 * x - 0.7 * y, -0.4 + 0.5 * x + 0.9 * y), axis=-1)


@pytest.mark.parametrize(
    ("factory", "quantity"),
    [
        (phx.discretization.conforming_hdiv_virtual_element, "normal"),
        (phx.discretization.conforming_hcurl_virtual_element, "tangential"),
    ],
    ids=["hdiv-normal", "hcurl-tangential"],
)
def test_virtual_element_moment_traces_are_outward_traces_of_linear_fields(
    factory: Any, quantity: Any
) -> None:
    space = _vem_space(factory(1))
    state = _vem_moment_dofs(space.mesh, _linear_vector_field, quantity)
    rule = FacetTraceRule(points=3)
    interior = space.interior_facet_domain
    for side, domain, cells in (
        ("owner", space.exterior_facet_domain, space.exterior_facet_domain.owner_cells),
        ("owner", interior, interior.owner_cells),
        ("neighbor", interior, interior.neighbor_cells),
    ):
        trace = space.prepare_side_trace(
            "u", domain, rule=rule, quantity=quantity, side=side
        )
        normals = _outward_normals(
            space.mesh, np.asarray(domain.entity_indices), np.asarray(cells)
        )
        # The tangential trace uses the side cell's counter-clockwise tangent.
        direction = (
            normals
            if quantity == "normal"
            else np.stack((-normals[:, 1], normals[:, 0]), axis=1)
        )
        expected = np.sum(
            _linear_vector_field(np.asarray(trace.sites)) * direction[:, None, :],
            axis=-1,
        )

        assert trace.descriptor.quantity == quantity
        assert trace.descriptor.orientation == "outward"
        assert trace.descriptor.approximation == "exact"
        np.testing.assert_allclose(trace.normals[:, 0], normals, atol=1e-14)
        np.testing.assert_allclose(trace.apply(state), expected, atol=1e-12)


def _unit_square_normals(sites: np.ndarray) -> np.ndarray:
    """Outward normals of the unit square at boundary points."""
    normals = np.zeros_like(sites)
    normals[np.isclose(sites[..., 0], 0.0)] = (-1.0, 0.0)
    normals[np.isclose(sites[..., 0], 1.0)] = (1.0, 0.0)
    normals[np.isclose(sites[..., 1], 0.0)] = (0.0, -1.0)
    normals[np.isclose(sites[..., 1], 1.0)] = (0.0, 1.0)
    return normals


def _manufactured(points: np.ndarray) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    return x**2 + x * y - 0.5 * y**2 + x - 2.0 * y + 0.25


def _manufactured_gradient(points: np.ndarray) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    return np.stack((2.0 * x + y + 1.0, x - y - 2.0), axis=-1)


def test_virtual_element_reaction_flux_is_the_edge_integral_of_the_physical_flux() -> (
    None
):
    space = _vem_space(phx.discretization.conforming_h1_virtual_element(2))
    kappa = 2.0
    constraint = phx.discretization.virtual_element_dirichlet_constraint(space, "u")
    # -div(kappa grad u) = -kappa * (2 - 1) for the manufactured quadratic.
    form = phx.equations.VirtualElementForm(
        "poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", kappa),
            phx.equations.SourceAction("u", -kappa),
        ),
    )
    compiled = phx.equations.compile_virtual_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=lambda points: _manufactured(np.asarray(points)),
    )
    problem, rhs = compiled.linear_system()
    state = compiled.expand(phx.linalg.solve(problem, rhs).value)
    trace = space.prepare_side_trace(
        "u", space.exterior_facet_domain, rule=FacetTraceRule(points=3)
    )
    reaction = compiled.prepare_conormal_flux(trace)
    projected = compiled.prepare_projected_flux(trace)

    # Independent edge integrals of kappa du/dn against the quadratic edge basis.
    edges = np.asarray(space.mesh.connectivity.edges)
    nodes, weights = np.polynomial.legendre.leggauss(4)
    s = 0.5 * (nodes + 1.0)
    basis = np.stack(
        (2.0 * (s - 0.5) * (s - 1.0), 4.0 * s * (1.0 - s), 2.0 * s * (s - 0.5))
    )
    expected = np.zeros(space.dof_map.global_dof_count)
    for edge in np.asarray(trace.descriptor.facets):
        start, stop = _POLYGON_COORDINATES[edges[edge]]
        points = start[None, :] + s[:, None] * (stop - start)[None, :]
        flux = kappa * np.sum(
            _manufactured_gradient(points) * _unit_square_normals(points), axis=-1
        )
        moments = basis @ (0.5 * weights * np.linalg.norm(stop - start) * flux)
        for row, moment in zip((edges[edge, 0], 7 + edge, edges[edge, 1]), moments):
            expected[row] += moment
    sites = np.asarray(trace.sites)

    assert reaction.descriptor.representation == "residual-reaction"
    assert reaction.descriptor.approximation == "variational-reaction"
    assert reaction.descriptor.orientation == "outward"
    np.testing.assert_allclose(
        reaction.evaluate(state), expected[np.asarray(trace.support_rows)], atol=1e-8
    )
    np.testing.assert_allclose(
        reaction.embed(reaction.evaluate(state)), expected, atol=1e-8
    )
    assert projected.descriptor.approximation == "h1-projection"
    assert projected.descriptor.representation == "quadrature-values"
    np.testing.assert_allclose(
        projected.evaluate(state),
        kappa * np.sum(_manufactured_gradient(sites) * _unit_square_normals(sites), -1),
        atol=1e-8,
    )
    boundary_rows = np.concatenate(
        (
            np.flatnonzero(np.asarray(space.mesh.connectivity.boundary_vertices)),
            7 + np.flatnonzero(np.asarray(space.mesh.connectivity.boundary_edges)),
        )
    )
    (imposition,) = compiled.boundary_impositions()
    assert imposition.kind == "strong"
    assert imposition.rows is not None
    assert imposition.rows.tolist() == boundary_rows.tolist()


def test_virtual_element_projected_flux_uses_the_side_cell_tensor_diffusivity() -> None:
    space = _vem_space(phx.discretization.conforming_h1_virtual_element(2))
    tensor = np.asarray(((2.0, 0.5), (0.5, 1.0)))
    compiled = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm(
            "anisotropic",
            "u",
            (phx.equations.DiffusionAction("u", jnp.asarray(tensor)),),
        ),
        space,
    )
    state = _vem_quadratic_dofs(
        space.mesh, lambda x, y: _manufactured(np.stack((x, y), axis=-1))
    )
    domain = space.interior_facet_domain
    rule = FacetTraceRule(points=2)
    owner = compiled.prepare_projected_flux(
        space.prepare_side_trace("u", domain, rule=rule)
    )
    neighbor = compiled.prepare_projected_flux(
        space.prepare_side_trace("u", domain, rule=rule, side="neighbor")
    )
    sites = np.asarray(owner.trace.sites)
    normals = _outward_normals(
        space.mesh, np.asarray(domain.entity_indices), np.asarray(domain.neighbor_cells)
    )
    expected = np.sum(
        normals[:, None, :] * (_manufactured_gradient(sites) @ tensor.T), axis=-1
    )

    np.testing.assert_allclose(neighbor.evaluate(state), expected, atol=1e-10)
    np.testing.assert_allclose(owner.evaluate(state), -expected, atol=1e-10)
    assert neighbor.descriptor.side == "neighbor"


def test_virtual_element_boundary_impositions_follow_the_form_order() -> None:
    space = _vem_space(phx.discretization.conforming_h1_virtual_element(1))
    exterior = space.exterior_facet_domain
    form = phx.equations.VirtualElementForm(
        "robin-and-load",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.BoundaryLoadAction("u", 1.0, action_id="heat-flux"),
            phx.equations.VirtualElementRobinAction(
                "u", 2.0, 0.5, exterior, action_id="convection"
            ),
        ),
    )
    compiled = phx.equations.compile_virtual_element_problem(form, space)

    natural, robin = compiled.boundary_impositions()

    assert (natural.kind, natural.source_id) == ("natural", "heat-flux")
    assert (robin.kind, robin.source_id) == ("robin", "convection")
    for imposition in (natural, robin):
        assert imposition.entity_set_id == exterior.entity_set_id
        assert imposition.facets is not None
        assert imposition.facets.tolist() == sorted(
            np.asarray(exterior.entity_indices).tolist()
        )
        assert imposition.rows is None


def test_polygon_side_traces_refuse_undefined_traces_and_sides() -> None:
    h1 = phx.discretization.ExplicitPolygonH1Plan(
        _polygon_mesh(), phx.discretization.ExplicitPolygonH1FieldSpec("u")
    ).prepare()
    vem = _vem_space(phx.discretization.conforming_h1_virtual_element(1))
    hdiv = _vem_space(phx.discretization.conforming_hdiv_virtual_element(1))
    discontinuous = _vem_space(phx.discretization.discontinuous_l2_virtual_element(1))
    rule = FacetTraceRule(points=2)
    other = phx.discretization.ExplicitPolygonH1Plan(
        phx.discretization.CellMesh.from_polygons(
            jnp.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))),
            (np.asarray((0, 1, 2, 3), dtype=np.int32),),
        ),
        phx.discretization.ExplicitPolygonH1FieldSpec("u"),
    ).prepare()

    with pytest.raises(ValueError, match="no L2 boundary trace"):
        discontinuous.prepare_side_trace(
            "u", discontinuous.exterior_facet_domain, rule=rule
        )
    with pytest.raises(ValueError, match="value traces only"):
        h1.prepare_side_trace("u", h1.exterior_facet_domain, rule=rule, quantity="normal")
    with pytest.raises(ValueError, match="publish 'value' traces"):
        vem.prepare_side_trace(
            "u", vem.exterior_facet_domain, rule=rule, quantity="tangential"
        )
    with pytest.raises(ValueError, match="publish 'normal' traces"):
        hdiv.prepare_side_trace("u", hdiv.exterior_facet_domain, rule=rule)
    for space in (h1, vem):
        with pytest.raises(ValueError, match="no neighbor side"):
            space.prepare_side_trace(
                "u", space.exterior_facet_domain, rule=rule, side="neighbor"
            )
        with pytest.raises(ValueError, match="one-sided"):
            space.prepare_side_trace(
                "u", space.interior_facet_domain, rule=rule, side="average"
            )
    with pytest.raises(ValueError, match="another owner's support"):
        vem.prepare_side_trace("u", other.exterior_facet_domain, rule=rule)


def test_virtual_element_fluxes_refuse_foreign_or_undefined_sides() -> None:
    space = _vem_space(phx.discretization.conforming_h1_virtual_element(1))
    form = phx.equations.VirtualElementForm(
        "diffusion", "u", (phx.equations.DiffusionAction("u", 1.0),)
    )
    compiled = phx.equations.compile_virtual_element_problem(form, space)
    rule = FacetTraceRule(points=2)
    interior = space.prepare_side_trace("u", space.interior_facet_domain, rule=rule)
    other = _vem_space(phx.discretization.conforming_h1_virtual_element(2))
    foreign = other.prepare_side_trace("u", other.exterior_facet_domain, rule=rule)
    hdiv = _vem_space(phx.discretization.conforming_hdiv_virtual_element(1))
    mixed = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm(
            "div", "u", (phx.equations.DiffusionAction("u", 1.0),)
        ),
        hdiv,
    )

    with pytest.raises(ValueError, match="exterior facets"):
        compiled.prepare_conormal_flux(interior)
    with pytest.raises(ValueError, match="another discretization field"):
        compiled.prepare_conormal_flux(foreign)
    with pytest.raises(ValueError, match="scalar ConformingH1"):
        mixed.prepare_projected_flux(
            hdiv.prepare_side_trace(
                "u", hdiv.exterior_facet_domain, rule=rule, quantity="normal"
            )
        )
    moved = space.prepare_side_trace(
        "u",
        space.exterior_facet_domain,
        rule=rule,
        runtime=space.prepare_runtime(
            _POLYGON_COORDINATES * 1.5, numeric_version="moved"
        ),
    )
    with pytest.raises(ValueError, match="another geometry revision"):
        compiled.prepare_projected_flux(moved)


# --- Finite differences and global spectral ---
# Nodal FD traces are measured by the tangential factors of a tensor SBP norm;
# spectral traces synthesize the field on bounded faces. References are host
# numpy polynomials, analytic face integrals, and numpy Gauss rules.


def _sbp_fd(
    shape: tuple[int, int],
    bounds: Any,
    interior_order: phx.discretization.SBPInteriorOrder,
) -> tuple[Any, Any]:
    d = phx.discretization
    grid = d.TensorGridPlan(
        tuple(d.UniformAxisSpec(count) for count in shape), axis_names=("x", "y")
    ).prepare(jnp.asarray(bounds, dtype=jnp.float64))
    requests = tuple(
        d.DerivativeRequest(f"d{name}", grid, name, derivative_order=1, accuracy_order=2)
        for name in ("x", "y")
    )
    discretization = d.FiniteDifferencePlan(grid, requests, field_name="u").prepare()
    norm = d.SBPGridNorm(
        tuple(
            d.SBPDerivativePlan(grid, name, interior_order=interior_order).prepare()
            for name in ("x", "y")
        )
    )
    return discretization, norm


def _flat_nodes(discretization: Any) -> np.ndarray:
    return np.asarray(discretization.grid.points)


@pytest.mark.parametrize("interior_order", (2, 4, 6))
def test_fd_sbp_boundary_norm_integrates_face_polynomials_to_its_order(
    interior_order: phx.discretization.SBPInteriorOrder,
) -> None:
    discretization, norm = _sbp_fd((21, 23), ((0.0, -1.0), (2.0, 1.5)), interior_order)
    lower_x = discretization.integration_domain(
        "exterior_facet", discretization.boundary_face_selection("x", "lower")
    )
    trace = discretization.prepare_side_trace("u", lower_x, norm=norm)
    nodes = _flat_nodes(discretization)
    field = np.exp(nodes[:, 0]) * np.cos(nodes[:, 1])
    sites = np.asarray(trace.sites)[:, 0]
    weights = np.asarray(trace.weights)[:, 0]
    # Diagonal-norm SBP with boundary closure order p integrates degree 2p - 1.
    exact = 2 * (interior_order // 2) - 1

    def moment(degree: int) -> float:
        return (1.5 ** (degree + 1) - (-1.0) ** (degree + 1)) / (degree + 1)

    np.testing.assert_allclose(
        trace.apply(jnp.asarray(field))[:, 0],
        np.exp(sites[:, 0]) * np.cos(sites[:, 1]),
        atol=1e-14,
    )
    np.testing.assert_allclose(sites[:, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(sites[:, 1], np.linspace(-1.0, 1.5, 23), atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(trace.normals)[:, 0], np.broadcast_to((-1.0, 0.0), (23, 2))
    )
    for degree in range(exact + 1):
        np.testing.assert_allclose(
            weights @ sites[:, 1] ** degree, moment(degree), rtol=1e-12
        )
    assert abs(weights @ sites[:, 1] ** (exact + 1) - moment(exact + 1)) > 1e-8
    assert trace.descriptor.quadrature_exact_degree == exact
    assert trace.descriptor.approximation == "exact"
    assert trace.descriptor.rule_id is None
    assert trace.support_rows.tolist() == list(range(23))
    assert trace.coefficient_space.shape == (21 * 23,)


def test_fd_trace_load_duality_and_sbp_hilbert_adjoint() -> None:
    discretization, norm = _sbp_fd((13, 15), ((0.0, 0.0), (1.0, 2.0)), 4)
    domain = discretization.exterior_facet_domain
    trace = discretization.prepare_side_trace("u", domain, norm=norm)
    rng = np.random.default_rng(17)
    state = jnp.asarray(rng.normal(size=13 * 15))
    covector = jnp.asarray(rng.normal(size=trace.output_shape))
    sites = np.asarray(trace.sites)[:, 0]
    pairing = norm.pairing(layout="rows")
    adjoint = trace.hilbert_adjoint(pairing)
    vector = discretization.prepare_side_trace(
        "u", domain, norm=norm, quantity="normal", component_shape=(2,)
    )
    nodes = _flat_nodes(discretization)
    velocity = np.stack((nodes[:, 0] ** 2, nodes[:, 0] * nodes[:, 1]), axis=-1)
    normals = np.asarray(vector.normals)[:, 0]

    assert trace.output_shape == (2 * (13 + 15), 1)
    np.testing.assert_allclose(
        jnp.vdot(trace.apply(state), covector),
        jnp.vdot(state, trace.dual_pullback(covector)),
        atol=1e-12,
    )
    # The load of g = y^2 pulled back through the SBP boundary quadrature sums
    # to the exact boundary integral: 8/3 on each x face, 0 + 4 on the y faces.
    load = trace.inject_load(jnp.asarray(sites[:, 1:2] ** 2))
    np.testing.assert_allclose(jnp.sum(load), 2.0 * 8.0 / 3.0 + 4.0, rtol=1e-12)
    np.testing.assert_allclose(
        jnp.sum(trace.weights * trace.apply(state) * covector),
        jnp.sum(norm.weights.reshape(-1) * state * adjoint.mv(covector)),
        atol=1e-12,
    )
    assert not np.allclose(adjoint.mv(covector), trace.dual_pullback(covector))
    np.testing.assert_allclose(
        vector.apply(jnp.asarray(velocity))[:, 0],
        np.sum(
            np.stack((sites[:, 0] ** 2, sites[:, 0] * sites[:, 1]), axis=-1) * normals,
            axis=-1,
        ),
        atol=1e-13,
    )
    assert vector.descriptor.orientation == "outward"


def test_fd_side_traces_refuse_rules_sides_periodic_faces_and_fluxes() -> None:
    discretization, norm = _sbp_fd((13, 13), ((0.0, 0.0), (1.0, 1.0)), 4)
    other, other_norm = _sbp_fd((13, 13), ((0.0, 0.0), (2.0, 1.0)), 4)
    domain = discretization.exterior_facet_domain
    d = phx.discretization
    periodic_grid = d.TensorGridPlan(
        (d.UniformAxisSpec(12, periodic=True, endpoint=False), d.UniformAxisSpec(9)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    periodic = d.FiniteDifferencePlan(
        periodic_grid,
        (
            d.DerivativeRequest(
                "dy", periodic_grid, "y", derivative_order=1, accuracy_order=2
            ),
        ),
        field_name="u",
    ).prepare()

    with pytest.raises(ValueError, match="rule must be None"):
        discretization.prepare_side_trace(
            "u", domain, norm=norm, rule=FacetTraceRule(points=2)
        )
    with pytest.raises(ValueError, match="no neighbor side"):
        discretization.prepare_side_trace("u", domain, norm=norm, side="neighbor")
    with pytest.raises(ValueError, match="compose the owner and neighbor"):
        discretization.prepare_side_trace("u", domain, norm=norm, side="average")
    with pytest.raises(ValueError, match="compiled physics owners"):
        discretization.prepare_side_trace(
            "u", domain, norm=norm, quantity="conormal-flux"
        )
    with pytest.raises(ValueError, match="no interior facets"):
        discretization.integration_domain("interior_facet")
    with pytest.raises(ValueError, match="another grid"):
        discretization.prepare_side_trace("u", domain, norm=other_norm)
    with pytest.raises(ValueError, match="another owner"):
        discretization.prepare_side_trace("u", other.exterior_facet_domain, norm=norm)
    with pytest.raises(ValueError, match="component_shape"):
        discretization.prepare_side_trace("u", domain, norm=norm, quantity="normal")
    with pytest.raises(ValueError, match="periodic"):
        periodic.boundary_face_selection("x", "lower")
    assert set(np.asarray(periodic.exterior_facet_domain.owner_local_entities)) == {
        2,
        3,
    }


def _chebyshev_legendre() -> Any:
    d = phx.discretization
    return d.TensorSpectralPlan(
        (d.ChebyshevBasisPlan(20), d.LegendreBasisPlan(16)),
        axis_names=("x", "y"),
        field_name="u",
    ).prepare((d.AxisDomain.interval(0.0, 2.0), d.AxisDomain.interval(-1.0, 1.0)))


def _smooth(x: Any, y: Any) -> Any:
    return np.exp(0.5 * x) * np.sin(1.3 * y + 0.2)


def _tensor_samples(space: Any, field: Any) -> Any:
    axes = tuple(np.asarray(axis.nodes) for axis in space.axes)
    return jnp.asarray(field(*np.meshgrid(*axes, indexing="ij")))


def test_spectral_face_traces_are_spectrally_accurate_with_native_measures() -> None:
    space = _chebyshev_legendre()
    coefficients = space.project(_tensor_samples(space, _smooth))
    x_faces = space.boundary_face_selection("x", "lower").union(
        space.boundary_face_selection("x", "upper")
    )
    x_trace = space.prepare_side_trace(
        "u", space.integration_domain("exterior_facet", x_faces)
    )
    y_trace = space.prepare_side_trace(
        "u",
        space.integration_domain(
            "exterior_facet", space.boundary_face_selection("y", "upper")
        ),
    )
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(16)
    lobatto = 1.0 + np.cos(np.pi * np.arange(19, -1, -1) / 19)

    x_sites = np.asarray(x_trace.sites)
    np.testing.assert_allclose(
        x_sites[..., 0], np.broadcast_to(((0.0,), (2.0,)), (2, 16)), atol=1e-15
    )
    np.testing.assert_allclose(x_sites[0, :, 1], gauss_nodes, atol=1e-14)
    np.testing.assert_allclose(np.asarray(x_trace.weights)[1], gauss_weights, atol=1e-14)
    np.testing.assert_allclose(
        x_trace.apply(coefficients), _smooth(x_sites[..., 0], x_sites[..., 1]), atol=1e-13
    )
    face_integral = (np.cos(-1.1) - np.cos(1.5)) / 1.3
    np.testing.assert_allclose(
        jnp.sum(x_trace.weights * x_trace.apply(coefficients), axis=1),
        (face_integral, np.exp(1.0) * face_integral),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(x_trace.normals)[:, 0], ((-1.0, 0.0), (1.0, 0.0))
    )
    assert x_trace.descriptor.trace_degree == 15
    assert x_trace.descriptor.quadrature_exact_degree == 31
    # Clenshaw--Curtis weights at the Chebyshev--Lobatto nodes of [0, 2].
    y_sites = np.asarray(y_trace.sites)[0]
    y_weights = np.asarray(y_trace.weights)[0]
    np.testing.assert_allclose(y_sites[:, 0], lobatto, atol=1e-14)
    np.testing.assert_allclose(y_sites[:, 1], 1.0)
    for degree in range(20):
        np.testing.assert_allclose(
            y_weights @ y_sites[:, 0] ** degree, 2.0 ** (degree + 1) / (degree + 1)
        )
    np.testing.assert_allclose(
        y_trace.apply(coefficients)[0], _smooth(y_sites[:, 0], 1.0), atol=1e-13
    )
    np.testing.assert_allclose(np.asarray(y_trace.normals)[0, 0], (0.0, 1.0))
    assert y_trace.descriptor.quadrature_exact_degree == 19
    assert y_trace.support_rows.tolist() == list(range(20))


def test_spectral_rule_traces_pull_loads_back_to_complex_modes() -> None:
    d = phx.discretization
    space = d.TensorSpectralPlan(
        (d.FourierBasisPlan(16), d.ChebyshevBasisPlan(8)),
        axis_names=("x", "y"),
        field_name="u",
    ).prepare((d.AxisDomain.periodic(0.0, 1.0), d.AxisDomain.interval(0.0, 1.0)))

    def field(x: Any, y: Any) -> Any:
        return np.cos(2.0 * np.pi * x) * (1.0 + y**2) + 0.25 * y

    def swirl(x: Any, y: Any) -> Any:
        return np.sin(2.0 * np.pi * x) * y**3

    coefficients = space.project(_tensor_samples(space, field))
    domain = space.exterior_facet_domain
    trace = space.prepare_side_trace("u", domain, rule=FacetTraceRule(points=5))
    native = space.prepare_side_trace("u", domain)
    velocity = jnp.stack(
        (coefficients, space.project(_tensor_samples(space, swirl))), axis=-1
    )
    normal = space.prepare_side_trace(
        "u", domain, quantity="normal", component_shape=(2,)
    )
    covector = jnp.asarray(np.random.default_rng(23).normal(size=trace.output_shape))
    sites = np.asarray(trace.sites)
    points, weights = np.polynomial.legendre.leggauss(5)

    assert np.asarray(domain.entity_indices).tolist() == [2, 3]
    np.testing.assert_allclose(sites[0, :, 0], 0.5 * (points + 1.0), atol=1e-15)
    np.testing.assert_allclose(np.asarray(trace.weights)[1], 0.5 * weights, atol=1e-15)
    np.testing.assert_allclose(
        trace.apply(coefficients), field(sites[..., 0], sites[..., 1]), atol=1e-12
    )
    # Uniform periodic weights integrate the trigonometric trace exactly:
    # int_0^1 u(x, 0) dx = 0 and int_0^1 u(x, 1) dx = 0.25.
    np.testing.assert_allclose(
        jnp.real(
            jnp.sum(coefficients * native.inject_load(jnp.ones(native.output_shape)))
        ),
        0.25,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        jnp.sum(trace.apply(coefficients) * covector),
        jnp.real(jnp.sum(coefficients * trace.dual_pullback(covector))),
        atol=1e-12,
    )
    normal_sites = np.asarray(normal.sites)
    np.testing.assert_allclose(
        normal.apply(velocity),
        np.where(normal_sites[..., 1] > 0.5, 1.0, -1.0)
        * swirl(normal_sites[..., 0], normal_sites[..., 1]),
        atol=1e-12,
    )
    for action in (trace, native):
        assert action.descriptor.trace_degree is None
        assert action.descriptor.quadrature_exact_degree is None
    assert native.output_shape == (2, 16)


def test_spectral_side_traces_refuse_periodic_sine_and_undefined_requests() -> None:
    d = phx.discretization
    mixed = _chebyshev_legendre()
    periodic = d.TensorSpectralPlan(
        (d.FourierBasisPlan(8), d.SineBasisPlan(8)), axis_names=("x", "y"), field_name="u"
    ).prepare((d.AxisDomain.periodic(0.0, 1.0), d.AxisDomain.interval(0.0, 1.0)))
    torus = d.TensorSpectralPlan(
        (d.FourierBasisPlan(8), d.FourierBasisPlan(8)),
        axis_names=("x", "y"),
        field_name="u",
    ).prepare((d.AxisDomain.periodic(0.0, 1.0), d.AxisDomain.periodic(0.0, 1.0)))
    base = periodic.exterior_facet_domain
    forged = IntegrationDomain(
        "exterior_facet",
        np.asarray((0,)),
        base.support_id,
        base.entity_set_id,
        owner_cells=np.zeros(1, np.int32),
        owner_local_entities=np.asarray((0,)),
    )
    domain = mixed.exterior_facet_domain
    fd, _ = _sbp_fd((13, 13), ((0.0, 0.0), (1.0, 1.0)), 4)

    with pytest.raises(ValueError, match="periodic"):
        periodic.boundary_face_selection("x", "upper")
    with pytest.raises(ValueError, match="periodic"):
        periodic.prepare_side_trace("u", forged)
    with pytest.raises(ValueError, match="sine mode vanishes"):
        periodic.prepare_side_trace("u", base)
    with pytest.raises(ValueError, match="no boundary faces"):
        torus.exterior_facet_domain
    with pytest.raises(ValueError, match="no interior facets"):
        mixed.integration_domain("interior_facet")
    with pytest.raises(ValueError, match="different site counts"):
        mixed.prepare_side_trace("u", domain)
    with pytest.raises(ValueError, match="compiled physics owners"):
        mixed.prepare_side_trace(
            "u", domain, rule=FacetTraceRule(points=3), quantity="conormal-flux"
        )
    with pytest.raises(ValueError, match="no neighbor side"):
        mixed.prepare_side_trace(
            "u", domain, rule=FacetTraceRule(points=3), side="neighbor"
        )
    with pytest.raises(ValueError, match="another owner"):
        mixed.prepare_side_trace(
            "u", fd.exterior_facet_domain, rule=FacetTraceRule(points=3)
        )
    with pytest.raises(KeyError):
        mixed.prepare_side_trace("v", domain, rule=FacetTraceRule(points=3))


# --- Isogeometric analysis ---
# An exact rational quarter annulus 1 <= r <= 2 (first quadrant) with one
# interior knot per axis. References are analytic arc and edge lengths,
# outward normals, boundary integrals of physical-linear fields, and the
# divergence theorem for v(x) = x.

_IGA_HALF = np.sqrt(0.5)


def _iga_annulus(*, vector: bool = False) -> tuple[Any, np.ndarray]:
    """Prepared quarter annulus with an isoparametric scalar or vector field."""
    from phydrax.discretization.iga._basis import (
        IsogeometricFieldSpec,
        TensorSplineBasisSpec,
    )

    iga = phx.discretization.iga
    # Boehm insertion of theta = 1/2 into the quadratic quarter circle.
    homogeneous = np.asarray(((1.0, 0.0, 1.0), (_IGA_HALF, _IGA_HALF, _IGA_HALF)))
    homogeneous = np.concatenate((homogeneous, ((0.0, 1.0, 1.0),)))
    refined = np.stack(
        (
            homogeneous[0],
            0.5 * (homogeneous[0] + homogeneous[1]),
            0.5 * (homogeneous[1] + homogeneous[2]),
            homogeneous[2],
        )
    )
    arc, arc_weights = refined[:, :2] / refined[:, 2:], refined[:, 2]
    radii = 1.0 + np.asarray((0.0, 0.25, 0.75, 1.0))
    controls = radii[:, None, None] * arc[None, :, :]
    knots = jnp.asarray((0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0))
    basis = TensorSplineBasisSpec(
        (iga.BSplineGrid(knots, 2), iga.BSplineGrid(knots, 2)), axis_names=("r", "t")
    )
    geometry = iga.NURBSGeometryState(
        jnp.asarray(controls), jnp.asarray(np.ones(4)[:, None] * arc_weights)
    )
    field = IsogeometricFieldSpec(
        "v" if vector else "u",
        basis,
        component_shape=(2,) if vector else (),
        weights_from_geometry=True,
    )
    plan = iga.IsogeometricPlan(
        basis,
        geometry,
        field,
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )
    return plan.prepare(numeric_version="annulus"), controls


def _iga_annulus_normals(sites: np.ndarray) -> np.ndarray:
    """Analytic outward normals of the quarter-annulus boundary at its sites."""
    radius = np.linalg.norm(sites, axis=-1, keepdims=True)
    radial = sites / radius
    normals = np.where(np.abs(radius - 2.0) < 1e-12, radial, -radial)
    normals = np.where(
        (np.abs(sites[..., 1:2]) < 1e-12), np.asarray((0.0, -1.0)), normals
    )
    return np.where(np.abs(sites[..., 0:1]) < 1e-12, np.asarray((-1.0, 0.0)), normals)


def test_isogeometric_exterior_traces_have_analytic_lengths_normals_and_values() -> None:
    prepared, controls = _iga_annulus()
    domain = prepared.exterior_facet_domain
    trace = prepared.prepare_side_trace("u", domain, rule=FacetTraceRule(points=6))
    a, b, c = 0.5, 2.0, -3.0
    # Coefficients on the compiled field's public control layout (4, 4).
    state = jnp.asarray(a + b * controls[..., 0] + c * controls[..., 1])
    sites = np.asarray(trace.sites)
    lengths = np.sum(np.asarray(trace.weights), axis=1)

    np.testing.assert_allclose(
        trace.apply(state), a + b * sites[..., 0] + c * sites[..., 1], atol=1e-12
    )
    # Facets: inner arc, outer arc, edge y = 0, edge x = 0 (two spans each).
    np.testing.assert_allclose(
        lengths.reshape((4, 2)).sum(axis=1), (0.5 * np.pi, np.pi, 1.0, 1.0), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.norm(sites, axis=-1)[:4],
        ((1.0,) * 6, (1.0,) * 6, (2.0,) * 6, (2.0,) * 6),
    )
    np.testing.assert_allclose(trace.normals, _iga_annulus_normals(sites), atol=1e-12)
    # Boundary integral of a + b x + c y, and the equal load/trace work pairing.
    load = jnp.ones(trace.output_shape)
    work = jnp.sum(trace.weights * trace.apply(state))
    exact = a * (1.5 * np.pi + 2.0) + (b + c) * (1.0 + 4.0 + 1.5)
    np.testing.assert_allclose(work, exact, rtol=1e-10)
    np.testing.assert_allclose(jnp.vdot(state, trace.inject_load(load)), work, rtol=1e-13)
    # Only boundary control rows carry a nonzero trace.
    boundary = np.ones((4, 4), dtype=bool)
    boundary[1:-1, 1:-1] = False
    assert trace.support_rows.tolist() == np.flatnonzero(boundary).tolist()
    descriptor = trace.descriptor
    assert descriptor.orientation == "unoriented"
    assert descriptor.approximation == "exact"
    assert descriptor.trace_degree is None  # rational along the arcs
    assert descriptor.quadrature_exact_degree == 11


def test_isogeometric_vector_traces_contract_the_outward_normal() -> None:
    prepared, controls = _iga_annulus(vector=True)
    domain = prepared.exterior_facet_domain
    rule = FacetTraceRule(points=6)
    state = jnp.asarray(controls)  # v(x) = x on the (4, 4, 2) control layout
    normal = prepared.prepare_side_trace("v", domain, rule=rule, quantity="normal")
    tangential = prepared.prepare_side_trace(
        "v", domain, rule=rule, quantity="tangential"
    )
    radius = np.linalg.norm(np.asarray(normal.sites), axis=-1)

    np.testing.assert_allclose(
        normal.apply(state).reshape((4, -1)),
        np.stack((-radius[:2].reshape(-1), radius[2:4].reshape(-1), *np.zeros((2, 12)))),
        atol=1e-12,
    )
    # Divergence theorem: the outward flux of x is twice the area 3 pi / 4.
    np.testing.assert_allclose(
        jnp.sum(normal.weights * normal.apply(state)), 1.5 * np.pi, rtol=1e-10
    )
    # tau = (-n_y, n_x): zero on the arcs, x on y = 0, and -y on x = 0.
    flux = jnp.sum(tangential.weights * tangential.apply(state), axis=1)
    np.testing.assert_allclose(
        flux.reshape((4, 2)).sum(axis=1), (0.0, 0.0, 1.5, -1.5), atol=1e-10
    )
    assert normal.descriptor.orientation == "outward"
    with pytest.raises(ValueError, match="vector field"):
        _iga_annulus()[0].prepare_side_trace("u", domain, rule=rule, quantity="normal")


def test_isogeometric_interior_knot_faces_share_sites_and_oppose_normals() -> None:
    prepared, _ = _iga_annulus()
    domain = prepared.integration_domain("interior_facet")
    rule = FacetTraceRule(points=6)
    owner = prepared.prepare_side_trace("u", domain, rule=rule)
    neighbor = prepared.prepare_side_trace("u", domain, rule=rule, side="neighbor")
    rng = np.random.default_rng(4)
    state = jnp.asarray(rng.normal(size=(4, 4)))
    sites = np.asarray(owner.sites)
    radius = np.linalg.norm(sites, axis=-1, keepdims=True)
    radial = sites / radius
    angular = np.stack((-radial[..., 1], radial[..., 0]), axis=-1)

    np.testing.assert_allclose(neighbor.sites, sites, atol=1e-14)
    np.testing.assert_allclose(neighbor.weights, owner.weights, atol=1e-14)
    np.testing.assert_allclose(neighbor.apply(state), owner.apply(state), atol=1e-12)
    # Knot faces r = 3/2 (owner r < 3/2) and theta = pi/4 (owner theta < pi/4).
    np.testing.assert_allclose(radius[:2], 1.5, atol=1e-12)
    np.testing.assert_allclose(owner.normals[:2], radial[:2], atol=1e-12)
    np.testing.assert_allclose(owner.normals[2:], angular[2:], atol=1e-12)
    np.testing.assert_allclose(neighbor.normals, -np.asarray(owner.normals), atol=1e-14)
    np.testing.assert_allclose(
        np.sum(np.asarray(owner.weights), axis=1).reshape((2, 2)).sum(axis=1),
        (0.75 * np.pi, 1.0),
        rtol=1e-10,
    )
    # Coordinate pullback versus the Hilbert adjoint of a diagonal Riesz map.
    covector = jnp.asarray(rng.normal(size=owner.output_shape))
    riesz = jnp.asarray(rng.uniform(0.5, 2.0, (4, 4)))
    adjoint = owner.hilbert_adjoint(phx.linalg.DiagonalPairing(riesz)).mv(covector)
    np.testing.assert_allclose(
        jnp.vdot(owner.apply(state), covector),
        jnp.vdot(state, owner.dual_pullback(covector)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.sum(owner.weights * owner.apply(state) * covector),
        jnp.vdot(state, riesz * adjoint),
        atol=1e-12,
    )
    np.testing.assert_allclose(adjoint, owner.inject_load(covector) / riesz, atol=1e-12)
    assert not np.allclose(adjoint, owner.dual_pullback(covector))


def test_isogeometric_side_traces_refuse_undefined_sides_and_foreign_domains() -> None:
    prepared, controls = _iga_annulus()
    rule = FacetTraceRule(points=3)
    exterior = prepared.exterior_facet_domain
    foreign = IntegrationDomain(
        "exterior_facet",
        np.arange(2),
        "foreign-support",
        exterior.entity_set_id,
        owner_cells=np.asarray(exterior.owner_cells)[:2],
        owner_local_entities=np.asarray(exterior.owner_local_entities)[:2],
    )
    trace = prepared.prepare_side_trace("u", exterior, rule=rule)
    moved = prepared.prepare_runtime(
        phx.discretization.iga.NURBSGeometryState(
            2.0 * jnp.asarray(controls), prepared.default_runtime.weights
        ),
        numeric_version="moved",
    )
    refreshed = prepared.prepare_side_trace("u", exterior, rule=rule, runtime=moved)

    np.testing.assert_allclose(refreshed.weights, 2.0 * np.asarray(trace.weights))
    trace.require_revision(trace.descriptor.revision_id)
    with pytest.raises(ValueError, match="another geometry revision"):
        trace.require_revision(refreshed.descriptor.revision_id)
    with pytest.raises(ValueError, match="no neighbor side"):
        prepared.prepare_side_trace("u", exterior, rule=rule, side="neighbor")
    with pytest.raises(ValueError, match="compose the owner and neighbor"):
        prepared.prepare_side_trace(
            "u", prepared.interior_facet_domain, rule=rule, side="average"
        )
    with pytest.raises(ValueError, match="compiled physics owners"):
        prepared.prepare_side_trace("u", exterior, rule=rule, quantity="conormal-flux")
    with pytest.raises(ValueError, match="exterior or interior facet"):
        prepared.prepare_side_trace("u", prepared.cell_domain, rule=rule)
    with pytest.raises(ValueError, match="not produced by this isogeometric"):
        prepared.prepare_side_trace("u", foreign, rule=rule)
    with pytest.raises(ValueError, match="S1 IGA supports only cell and exterior"):
        prepared.prepare_local_regions(
            prepared.interior_facet_domain,
            field_names=("u",),
            maximum_derivative_order=1,
            kernel_mode="sum_factorized",
        )


# --- Finite volumes ---


_FV_EDGES = (
    np.asarray((0.0, 0.1, 0.35, 0.55, 0.8, 1.0)),
    np.asarray((0.0, 0.2, 0.45, 0.7, 1.0)),
    np.asarray((0.0, 0.3, 0.5, 0.65, 1.0)),
)
_FV_KUHN_PATHS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def _fv_structured(dimension: int) -> Any:
    d = phx.discretization
    grid = d.TensorGridPlan(
        tuple(d.NonuniformCellAxisSpec(_FV_EDGES[axis]) for axis in range(dimension)),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.stack((jnp.zeros(dimension), jnp.ones(dimension))))
    return d.FiniteVolumePlan(grid, component_names=("rho", "energy")).prepare()


def _fv_square_triangles() -> tuple[np.ndarray, np.ndarray]:
    """Skewed right-diagonal triangulation of the unit square."""
    resolution = 4
    axis = np.linspace(0.0, 1.0, resolution + 1)
    vertices = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape((-1, 2))
    inside = np.all((vertices > 0.0) & (vertices < 1.0), axis=1)
    vertices[inside] += 0.04 * np.sin(5.0 * vertices[inside][:, ::-1] + 1.0)
    triangles = []
    for j in range(resolution):
        for i in range(resolution):
            corner = j * (resolution + 1) + i
            triangles.append((corner, corner + 1, corner + resolution + 2))
            triangles.append((corner, corner + resolution + 2, corner + resolution + 1))
    return vertices, np.asarray(triangles, dtype=np.int32)


def _fv_cube_tetrahedra() -> tuple[np.ndarray, np.ndarray]:
    """Kuhn triangulation of the unit cube on a 2 x 2 x 2 brick grid."""
    count = 3
    axis = np.linspace(0.0, 1.0, count)
    vertices = np.asarray([(x, y, z) for z in axis for y in axis for x in axis])
    tetrahedra = []
    for k in range(count - 1):
        for j in range(count - 1):
            for i in range(count - 1):
                for path in _FV_KUHN_PATHS:
                    corner = [i, j, k]
                    cells = [(corner[2] * count + corner[1]) * count + corner[0]]
                    for step in path:
                        corner[step] += 1
                        cells.append((corner[2] * count + corner[1]) * count + corner[0])
                    tetrahedra.append(cells)
    return vertices, np.asarray(tetrahedra, dtype=np.int32)


def _fv_owner(kind: str) -> Any:
    d = phx.discretization
    components = ("rho", "energy")
    match kind:
        case "structured-2d":
            return _fv_structured(2)
        case "structured-3d":
            return _fv_structured(3)
        case "triangles":
            vertices, triangles = _fv_square_triangles()
            return d.UnstructuredFiniteVolumePlan(
                vertices, triangles=triangles, component_names=components
            ).prepare()
        case "triangle-fv":
            vertices, triangles = _fv_square_triangles()
            return d.TriangleFiniteVolumePlan(
                vertices, triangles, component_names=components
            ).prepare()
        case "tetrahedra":
            vertices, tetrahedra = _fv_cube_tetrahedra()
            return d.UnstructuredFiniteVolumePlan(
                vertices, tetrahedra=tetrahedra, component_names=components
            ).prepare()
    raise AssertionError(kind)


def _fv_contains(kind: str, cell: int, point: np.ndarray) -> bool:
    """Host containment of a point in one cell of the reference meshes."""
    if kind.startswith("structured"):
        edges = _FV_EDGES[: point.size]
        index = np.unravel_index(cell, tuple(values.size - 1 for values in edges))
        return all(
            values[index[axis]] <= point[axis] <= values[index[axis] + 1]
            for axis, values in enumerate(edges)
        )
    vertices, cells = (
        _fv_cube_tetrahedra() if kind == "tetrahedra" else _fv_square_triangles()
    )
    simplex = vertices[cells[cell]]
    barycentric = np.linalg.solve((simplex[1:] - simplex[0]).T, point - simplex[0])
    return bool(np.all(barycentric >= -1e-9) and np.sum(barycentric) <= 1.0 + 1e-9)


@pytest.mark.parametrize(
    "kind", ("structured-2d", "structured-3d", "triangles", "triangle-fv", "tetrahedra")
)
def test_fv_face_measures_normals_and_sides_follow_the_cell_geometry(kind: str) -> None:
    owner = _fv_owner(kind)
    name = owner.cell_space.name
    rule = FacetTraceRule(points=2)
    dimension = 3 if kind in ("structured-3d", "tetrahedra") else 2
    exterior = owner.prepare_side_trace(
        name, owner.integration_domain("exterior_facet"), rule=rule
    )
    sites = np.asarray(exterior.sites)
    weights = np.asarray(exterior.weights)
    normals = np.asarray(exterior.normals)

    # The unit square/cube: boundary face measures sum to the boundary measure,
    # and the divergence theorem gives oint x_a n_a dS = |Omega| = 1 per axis.
    np.testing.assert_allclose(weights.sum(), 2.0 * dimension, rtol=1e-12)
    np.testing.assert_allclose(
        np.einsum("fq,fqa,fqa->a", weights, sites, normals), np.ones(dimension)
    )
    expected = np.isclose(sites, 1.0).astype(float) - np.isclose(sites, 0.0)
    np.testing.assert_allclose(normals, expected, atol=1e-12)

    interior = owner.integration_domain("interior_facet")
    owner_side = owner.prepare_side_trace(name, interior, rule=rule)
    neighbor_side = owner.prepare_side_trace(name, interior, rule=rule, side="neighbor")
    np.testing.assert_allclose(owner_side.sites, neighbor_side.sites)
    np.testing.assert_allclose(owner_side.weights, neighbor_side.weights)
    np.testing.assert_allclose(owner_side.normals, -np.asarray(neighbor_side.normals))
    np.testing.assert_allclose(np.linalg.norm(owner_side.normals, axis=-1), 1.0)

    # Cell averages u_c = (c, -c) identify the side cell, which lies behind
    # every facet along its outward normal.
    shape = owner.state_shape
    ids = np.arange(np.prod(shape[:-1]), dtype=np.float64).reshape(shape[:-1])
    coefficients = jnp.asarray(np.stack((ids, -ids), axis=-1))
    for trace in (exterior, owner_side, neighbor_side):
        assert trace.descriptor.representation == "cell-average"
        assert trace.descriptor.trace_degree == 0
        values = np.asarray(trace.apply(coefficients))
        np.testing.assert_allclose(values[..., 1], -values[..., 0])
        facet_sites = np.asarray(trace.sites)
        facet_normals = np.asarray(trace.normals)
        for facet in range(values.shape[0]):
            cell = int(round(values[facet, 0, 0]))
            np.testing.assert_allclose(values[facet, :, 0], cell)
            probe = facet_sites[facet].mean(axis=0) - 1e-7 * facet_normals[facet, 0]
            assert _fv_contains(kind, cell, probe)


def _fv_linear(points: np.ndarray) -> np.ndarray:
    return 0.3 + 1.5 * points[..., 0] - 0.7 * points[..., 1]


def _fv_quadratic(points: np.ndarray) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    return 0.2 + x - 2.0 * y + 1.5 * x * x - 0.5 * x * y + 0.8 * y * y


def _fv_triangle_averages(
    vertices: np.ndarray, triangles: np.ndarray, function: Any
) -> np.ndarray:
    """Exact cell averages of polynomials through degree two (edge midpoints)."""
    corners = vertices[triangles]
    midpoints = 0.5 * (corners + np.roll(corners, -1, axis=1))
    return np.mean(function(midpoints), axis=1)


@pytest.mark.parametrize(
    ("case", "degree"),
    (("k-exact-1", 1), ("k-exact-2", 2), ("triangle-k-exact", 2), ("triangle-muscl", 1)),
)
def test_fv_linear_reconstruction_face_states_are_exact_polynomial_traces(
    case: str, degree: int
) -> None:
    d = phx.discretization
    vertices, triangles = _fv_square_triangles()
    function = _fv_linear if degree == 1 else _fv_quadratic
    reconstruction: Any
    if case.startswith("k-exact"):
        owner: Any = d.UnstructuredFiniteVolumePlan(
            vertices, triangles=triangles
        ).prepare()
        reconstruction = d.CellPolynomialReconstructionPlan(degree).prepare(owner)
    else:
        owner = d.TriangleFiniteVolumePlan(vertices, triangles).prepare()
        reconstruction = (
            d.TriangleKExactReconstructionPlan(d.PreparedTriangleQuadratic(owner))
            if case == "triangle-k-exact"
            else d.TriangleMUSCLReconstructionPlan(
                d.PreparedTriangleWLSQ(owner), limiter="unlimited"
            )
        )
    averages = jnp.asarray(_fv_triangle_averages(vertices, triangles, function)[:, None])
    rule = FacetTraceRule(points=3)
    for kind, sides in (
        ("exterior_facet", ("owner",)),
        ("interior_facet", ("owner", "neighbor")),
    ):
        domain = owner.integration_domain(kind)
        for side in sides:
            trace = owner.prepare_side_trace(
                owner.cell_space.name,
                domain,
                rule=rule,
                side=side,
                reconstruction=reconstruction,
            )
            assert trace.descriptor.representation == "face-state"
            assert trace.descriptor.approximation == "exact"
            assert trace.descriptor.trace_degree == degree
            np.testing.assert_allclose(
                trace.apply(averages)[..., 0],
                function(np.asarray(trace.sites)),
                atol=1e-10,
            )


def _fv_interior_selection(owner: Any, keep: Any) -> Any:
    """Interior-facet domain restricted to the facets accepted by `keep`."""
    interior = owner.integration_domain("interior_facet")
    total = (
        interior.entity_indices.size
        + owner.integration_domain("exterior_facet").entity_indices.size
    )
    mask = np.zeros((total,), dtype=np.bool_)
    mask[np.asarray(interior.entity_indices)[keep(interior)]] = True
    selection = phx.discretization.EntitySelection(
        interior.entity_set_id, mask, active_mask=np.ones((total,), dtype=np.bool_)
    )
    return owner.integration_domain("interior_facet", selection)


def _fv_face_axes(owner: Any, domain: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normal axis and lower cell index of structured interior facets."""
    first = np.stack(
        np.unravel_index(np.asarray(domain.owner_cells), owner.cell_shape), axis=-1
    )
    second = np.stack(
        np.unravel_index(np.asarray(domain.neighbor_cells), owner.cell_shape), axis=-1
    )
    axis = np.argmax(first != second, axis=1)
    rows = np.arange(axis.size)
    return axis, np.minimum(first[rows, axis], second[rows, axis])


def test_fv_structured_unlimited_muscl_face_state_is_the_linear_face_average() -> None:
    d = phx.discretization
    owner = _fv_structured(2)

    def keep(domain: Any) -> Any:
        axis, lower = _fv_face_axes(owner, domain)
        return (lower >= 1) & (lower <= np.asarray(owner.cell_shape)[axis] - 3)

    domain = _fv_interior_selection(owner, keep)
    muscl = d.MUSCLReconstruction(d.UnlimitedLimiter())
    centers = [0.5 * (edges[1:] + edges[:-1]) for edges in _FV_EDGES[:2]]
    x, y = np.meshgrid(*centers, indexing="ij")
    linear = _fv_linear(np.stack((x, y), axis=-1))
    averages = jnp.asarray(np.stack((linear, -2.0 * linear), axis=-1))
    for side in ("owner", "neighbor"):
        trace = owner.prepare_side_trace(
            "state",
            domain,
            rule=FacetTraceRule(points=3),
            side=side,
            reconstruction=muscl,
        )
        assert trace.descriptor.representation == "face-state"
        assert trace.descriptor.trace_degree == 0
        midpoints = np.asarray(trace.sites).mean(axis=1, keepdims=True)
        expected = np.broadcast_to(_fv_linear(midpoints), trace.output_shape[:2])
        values = trace.apply(averages)
        np.testing.assert_allclose(values[..., 0], expected, atol=1e-12)
        np.testing.assert_allclose(values[..., 1], -2.0 * expected, atol=1e-12)


def test_fv_face_state_routes_are_local_with_exact_transposes_and_volume_adjoints() -> (
    None
):
    d = phx.discretization
    vertices, triangles = _fv_square_triangles()
    owner = d.UnstructuredFiniteVolumePlan(
        vertices, triangles=triangles, component_names=("rho", "energy")
    ).prepare()
    reconstruction = d.CellPolynomialReconstructionPlan(1).prepare(owner)
    trace = owner.prepare_side_trace(
        owner.cell_space.name,
        owner.integration_domain("interior_facet"),
        rule=FacetTraceRule(points=2),
        side="neighbor",
        reconstruction=reconstruction,
    )
    rng = np.random.default_rng(3)
    coefficients = jnp.asarray(rng.normal(size=owner.state_shape))
    covector = jnp.asarray(rng.normal(size=trace.output_shape))
    values = trace.apply(coefficients)

    np.testing.assert_allclose(
        jnp.vdot(values, covector),
        jnp.vdot(coefficients, trace.dual_pullback(covector)),
        rtol=1e-12,
    )
    # Partial assembly: each facet gathers its side cell and that cell's stencil.
    assert trace.route.dofs.shape == (
        trace.output_shape[0],
        1 + reconstruction.report.stencil_capacity,
    )
    corners = vertices[triangles]
    edges = corners[:, 1:] - corners[:, :1]
    areas = 0.5 * np.abs(
        edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
    )
    pairing = phx.linalg.DiagonalPairing(
        jnp.asarray(np.broadcast_to(areas[:, None], owner.state_shape))
    )
    adjoint = trace.hilbert_adjoint(pairing).mv(covector)
    measure = np.asarray(trace.weights)[..., None]
    np.testing.assert_allclose(
        adjoint, np.asarray(trace.inject_load(covector)) / areas[:, None], atol=1e-10
    )
    np.testing.assert_allclose(
        np.sum(measure * values * covector),
        np.sum(areas[:, None] * coefficients * adjoint),
        rtol=1e-10,
    )


def test_fv_structured_cell_average_load_injects_the_exposed_face_area() -> None:
    owner = _fv_structured(3)
    trace = owner.prepare_side_trace(
        "state", owner.integration_domain("exterior_facet"), rule=FacetTraceRule(points=2)
    )
    load = np.asarray(trace.inject_load(jnp.ones(trace.output_shape)))
    widths = [np.diff(edges) for edges in _FV_EDGES]
    exposed = np.zeros(owner.cell_shape)
    for axis in range(3):
        others = [widths[other] for other in range(3) if other != axis]
        area = np.multiply.outer(*others)
        for end in (0, -1):
            index: list[Any] = [slice(None)] * 3
            index[axis] = end
            exposed[tuple(index)] += area
    assert load.shape == owner.state_shape
    np.testing.assert_allclose(load[..., 0], exposed, rtol=1e-12)
    np.testing.assert_allclose(load[..., 1], exposed, rtol=1e-12)
    coefficients = jnp.asarray(np.random.default_rng(9).normal(size=owner.state_shape))
    covector = jnp.asarray(np.random.default_rng(10).normal(size=trace.output_shape))
    np.testing.assert_allclose(
        jnp.vdot(trace.apply(coefficients), covector),
        jnp.vdot(coefficients, trace.dual_pullback(covector)),
        rtol=1e-12,
    )


def _fv_check_linearization(trace: Any, state: Any, seed: int) -> None:
    """The prepared linearization matches central differences and its own VJP."""
    rng = np.random.default_rng(seed)
    tangent = jnp.asarray(rng.normal(size=state.shape))
    cotangent = jnp.asarray(rng.normal(size=trace.output_shape))
    linearization = trace.linearize(state)
    step = 1e-6
    difference = (
        trace.apply(state + step * tangent) - trace.apply(state - step * tangent)
    ) / (2.0 * step)
    pushed = linearization.jvp(tangent)
    np.testing.assert_allclose(pushed, difference, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        jnp.vdot(pushed, cotangent),
        jnp.vdot(tangent, linearization.vjp(cotangent)),
        rtol=1e-10,
    )


def _fv_front(points: np.ndarray) -> np.ndarray:
    return np.tanh(8.0 * (points[..., 0] - 0.45)) + 0.3 * points[..., 1]


@pytest.mark.parametrize("limiter", ("none", "cell_extrema"))
def test_fv_weno_z_face_states_publish_a_local_linearization(limiter: Any) -> None:
    d = phx.discretization
    vertices, triangles = _fv_square_triangles()
    owner = d.UnstructuredFiniteVolumePlan(vertices, triangles=triangles).prepare()
    weno = d.UnstructuredWENOZReconstructionPlan(2, limiter=limiter).prepare(owner)
    state = jnp.asarray(_fv_triangle_averages(vertices, triangles, _fv_front)[:, None])
    domain = owner.integration_domain("interior_facet")
    facets = np.asarray(domain.entity_indices)
    left, right = weno.reconstruct_at(state, owner.face_centers[:, None, :])
    rule = FacetTraceRule(points=1)
    for side, expected in (("owner", left), ("neighbor", right)):
        trace = owner.prepare_nonlinear_face_trace(
            owner.cell_space.name, domain, rule=rule, reconstruction=weno, side=side
        )
        assert trace.descriptor.representation == "face-state"
        np.testing.assert_allclose(
            trace.apply(state), np.asarray(expected)[facets], atol=1e-12
        )
        _fv_check_linearization(trace, state, 21)
    with pytest.raises(ValueError, match="nonlinear in the cell averages"):
        owner.prepare_side_trace(
            owner.cell_space.name, domain, rule=rule, reconstruction=weno
        )


def test_fv_limited_muscl_and_structured_weno_face_states_match_their_owners() -> None:
    d = phx.discretization
    vertices, triangles = _fv_square_triangles()
    owner = d.TriangleFiniteVolumePlan(vertices, triangles).prepare()
    muscl = d.TriangleMUSCLReconstructionPlan(
        d.PreparedTriangleWLSQ(owner), limiter="barth_jespersen"
    )
    state = jnp.asarray(_fv_triangle_averages(vertices, triangles, _fv_front)[:, None])
    domain = owner.integration_domain("interior_facet")
    facets = np.asarray(domain.entity_indices)
    left, right = muscl.reconstruct(state)
    for side, expected in (("owner", left), ("neighbor", right)):
        trace = owner.prepare_nonlinear_face_trace(
            owner.cell_space.name,
            domain,
            rule=FacetTraceRule(points=1),
            reconstruction=muscl,
            side=side,
        )
        np.testing.assert_allclose(
            trace.apply(state)[:, 0], np.asarray(expected)[facets], atol=1e-12
        )
        _fv_check_linearization(trace, state, 22)

    grid = d.TensorGridPlan(
        (d.UniformCellAxisSpec(8, periodic=True), d.NonuniformCellAxisSpec(_FV_EDGES[1])),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    structured = d.FiniteVolumePlan(grid).prepare()

    def keep(interior: Any) -> Any:
        axis, _ = _fv_face_axes(structured, interior)
        return (axis == 0) & ~np.asarray(interior.periodic_face_mask)

    selected = _fv_interior_selection(structured, keep)
    x = (np.arange(8) + 0.5) / 8.0
    y = 0.5 * (_FV_EDGES[1][1:] + _FV_EDGES[1][:-1])
    cells = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    values = jnp.asarray(_fv_front(cells)[..., None])
    plan = d.WENOReconstructionPlan(5)
    upper, lower = plan.reconstruct(values)
    rows = np.unravel_index(np.asarray(selected.owner_cells), structured.cell_shape)
    for side, expected in (("owner", upper), ("neighbor", lower)):
        trace = structured.prepare_nonlinear_face_trace(
            "state",
            selected,
            rule=FacetTraceRule(points=2),
            reconstruction=plan,
            side=side,
        )
        np.testing.assert_allclose(
            trace.apply(values)[:, :, 0],
            np.broadcast_to(np.asarray(expected)[rows][:, None, 0], (rows[0].size, 2)),
            atol=1e-12,
        )
        _fv_check_linearization(trace, values, 23)


class _FVStateFlux(phx.discretization.AbstractSideFluxEvaluator):
    """Identity flux evaluator used to probe representation checks."""

    space: ArraySpace

    @property
    def state_space(self) -> ArraySpace:
        return self.space

    def evaluate(self, state: Any, args: Any) -> Any:
        return state


def test_fv_cell_average_and_face_state_descriptors_are_distinct_identities() -> None:
    d = phx.discretization
    vertices, triangles = _fv_square_triangles()
    owner = d.UnstructuredFiniteVolumePlan(vertices, triangles=triangles).prepare()
    name = owner.cell_space.name
    domain = owner.integration_domain("exterior_facet")
    rule = FacetTraceRule(points=2)
    average = owner.prepare_side_trace(name, domain, rule=rule)
    first = owner.prepare_side_trace(
        name,
        domain,
        rule=rule,
        reconstruction=d.CellPolynomialReconstructionPlan(1).prepare(owner),
    )
    second = owner.prepare_side_trace(
        name,
        domain,
        rule=rule,
        reconstruction=d.CellPolynomialReconstructionPlan(1, weight_power=1.0).prepare(
            owner
        ),
    )
    assert average.descriptor.representation == "cell-average"
    assert first.descriptor.representation == "face-state"
    assert len({average.action_id, first.action_id, second.action_id}) == 3
    reconstruction = phx.discretization.prepare_finite_volume_field_reconstruction(
        owner, d.PiecewiseConstantReconstruction()
    )
    assert average.descriptor.field_space_id == reconstruction.field_space_id
    assert average.descriptor.owner_id == owner.prepared_id

    flux = SideActionDescriptor(
        owner_id=owner.prepared_id,
        field_space_id=average.descriptor.field_space_id,
        quantity="conormal-flux",
        representation="cell-average",
        orientation="outward",
        approximation="exact",
        side="owner",
        domain=domain,
        revision_id=average.descriptor.revision_id,
        rule=rule,
        trace_degree=0,
        quadrature_exact_degree=3,
    )
    with pytest.raises(ValueError, match="cell-average and face-state data are traces"):
        phx.discretization.PreparedFluxAction(
            flux, average, _FVStateFlux(average.coefficient_space)
        )


def test_fv_side_traces_refuse_undefined_sides_domains_and_reconstructions() -> None:
    d = phx.discretization
    vertices, triangles = _fv_square_triangles()
    owner = d.UnstructuredFiniteVolumePlan(vertices, triangles=triangles).prepare()
    name = owner.cell_space.name
    rule = FacetTraceRule(points=2)
    exterior = owner.integration_domain("exterior_facet")
    interior = owner.integration_domain("interior_facet")
    structured = _fv_structured(2)
    weno = d.UnstructuredWENOZReconstructionPlan(2).prepare(owner)
    with pytest.raises(ValueError, match="compose the owner and neighbor"):
        owner.prepare_side_trace(name, interior, rule=rule, side="average")
    with pytest.raises(ValueError, match="no neighbor side"):
        owner.prepare_side_trace(name, exterior, rule=rule, side="neighbor")
    with pytest.raises(ValueError, match="another owner"):
        owner.prepare_side_trace(
            name, structured.integration_domain("exterior_facet"), rule=rule
        )
    with pytest.raises(ValueError, match="value face states"):
        owner.prepare_side_trace(name, exterior, rule=rule, quantity="normal")
    with pytest.raises(KeyError):
        owner.prepare_side_trace("pressure", exterior, rule=rule)
    tampered = IntegrationDomain(
        "interior_facet",
        interior.entity_indices,
        interior.support_id,
        interior.entity_set_id,
        owner_cells=interior.neighbor_cells,
        neighbor_cells=interior.owner_cells,
        owner_local_entities=interior.neighbor_local_entities,
        neighbor_local_entities=interior.owner_local_entities,
    )
    with pytest.raises(ValueError, match="do not match"):
        owner.prepare_side_trace(name, tampered, rule=rule)
    with pytest.raises(ValueError, match="linear in the cell averages"):
        owner.prepare_nonlinear_face_trace(
            name,
            exterior,
            rule=rule,
            reconstruction=d.CellPolynomialReconstructionPlan(1).prepare(owner),
        )
    refreshed = d.UnstructuredFiniteVolumePlan(vertices, triangles=triangles).prepare(
        numeric_version="1"
    )
    with pytest.raises(ValueError, match="different finite-volume"):
        refreshed.prepare_nonlinear_face_trace(
            name, exterior, rule=rule, reconstruction=weno
        )
    with pytest.raises(ValueError, match="structured grids"):
        owner.prepare_side_trace(
            name, exterior, rule=rule, reconstruction=d.MUSCLReconstruction()
        )
    with pytest.raises(ValueError, match="leaves the grid"):
        structured.prepare_nonlinear_face_trace(
            "state",
            structured.integration_domain("exterior_facet"),
            rule=rule,
            reconstruction=d.MUSCLReconstruction(d.MinmodLimiter()),
        )
    periodic_grid = d.TensorGridPlan(
        (d.UniformCellAxisSpec(6, periodic=True), d.UniformCellAxisSpec(4)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    periodic = d.FiniteVolumePlan(periodic_grid).prepare()
    with pytest.raises(ValueError, match="Periodic faces"):
        periodic.prepare_side_trace(
            "state", periodic.integration_domain("interior_facet"), rule=rule
        )

    def interior_x(domain: Any) -> Any:
        axis, lower = _fv_face_axes(structured, domain)
        return (axis == 0) & (lower == 2)

    with pytest.raises(ValueError, match="uniform cells"):
        structured.prepare_nonlinear_face_trace(
            "state",
            _fv_interior_selection(structured, interior_x),
            rule=rule,
            reconstruction=d.WENOReconstructionPlan(3),
        )
