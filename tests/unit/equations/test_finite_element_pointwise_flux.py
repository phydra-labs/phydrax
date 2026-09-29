#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact pointwise conormal fluxes and certified trace-inverse constants of FE owners.

References are independent of the implementation: nodal interpolants of
polynomials the spaces contain, their analytic gradients contracted with the
declared diffusivity tensor, and the analytic P1 trace-inverse constant
``|F| n.K n / |K|`` (the gradient of a P1 function is one constant vector, so
the Rayleigh quotient ``|F| (n.K g)^2 / (|K| g.K g)`` peaks at ``g = n``).
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain


_KAPPA = np.asarray([[2.0, 0.5], [0.5, 1.0]])


def _square(
    cells: int, degree: int, scale: float = 1.0
) -> phx.discretization.FiniteElementDiscretization:
    """Structured right-triangle mesh of ``[0, scale]^2``."""
    xs = np.linspace(0.0, scale, cells + 1)
    points = np.stack(np.meshgrid(xs, xs, indexing="xy"), -1).reshape(-1, 2)
    triangles = []
    for j in range(cells):
        for i in range(cells):
            a = j * (cells + 1) + i
            triangles += [(a, a + 1, a + cells + 2), (a, a + cells + 2, a + cells + 1)]
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", degree)
        ),
    ).prepare()


def _facets(
    space: phx.discretization.FiniteElementDiscretization,
    selected: Callable[[np.ndarray], np.ndarray],
    /,
) -> IntegrationDomain:
    """Exterior facets whose two end points both satisfy ``selected``."""
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    sites = np.asarray(probe.sites)
    on = np.all(selected(sites), axis=1)
    entities = space.mesh.topology.entity_sets[1]
    mask = np.zeros((entities.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[on]] = True
    return space.integration_domain("exterior_facet", EntitySelection(entities, mask))


def _compiled(
    space: phx.discretization.FiniteElementDiscretization,
    *actions: phx.equations.FiniteElementAction,
) -> phx.equations.CompiledFiniteElementProblem:
    form = phx.equations.FiniteElementForm("physics", "u", actions)
    return phx.equations.compile_finite_element_problem(form, space)


def _quadratic(points: np.ndarray) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    return x**2 + 3.0 * x * y - 2.0 * y**2 + x


def _quadratic_gradient(points: np.ndarray) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    return np.stack((2.0 * x + 3.0 * y + 1.0, 3.0 * x - 4.0 * y), axis=-1)


def test_tensor_flux_equals_the_analytic_conormal_flux_at_the_trace_sites() -> None:
    """P2 interpolant of a quadratic: ``q = n . K grad(u)`` at every site of x = 1."""
    space = _square(3, 2)
    compiled = _compiled(space, phx.equations.TensorDiffusionAction("u", _KAPPA))
    right = _facets(space, lambda sites: np.isclose(sites[..., 0], 1.0))
    trace = space.prepare_side_trace(
        "u", right, rule=FacetTraceRule("gauss-lobatto-legendre", points=4)
    )
    flux = compiled.prepare_pointwise_flux(trace)
    coefficients = jnp.asarray(_quadratic(np.asarray(space.dof_maps[0].dof_coordinates)))
    exact = np.einsum(
        "fqi,ij,fqj->fq",
        np.asarray(trace.normals),
        _KAPPA,
        _quadratic_gradient(np.asarray(trace.sites)),
    )

    assert flux.descriptor.representation == "quadrature-values"
    assert flux.descriptor.approximation == "exact"
    assert flux.descriptor.orientation == "outward"
    assert flux.descriptor.trace_degree == 1
    np.testing.assert_allclose(np.asarray(flux.evaluate(coefficients)), exact, atol=1e-12)


def test_owner_and_neighbor_fluxes_of_an_interior_facet_cancel() -> None:
    """Outward fluxes of a smooth discrete field through one interior facet balance."""
    space = _square(3, 2)
    compiled = _compiled(space, phx.equations.DiffusionAction("u", 3.0))
    interior = space.interior_facet_domain
    rule = FacetTraceRule(points=3)
    owner = compiled.prepare_pointwise_flux(
        space.prepare_side_trace("u", interior, rule=rule, side="owner")
    )
    neighbor = compiled.prepare_pointwise_flux(
        space.prepare_side_trace("u", interior, rule=rule, side="neighbor")
    )
    coefficients = jnp.asarray(_quadratic(np.asarray(space.dof_maps[0].dof_coordinates)))
    sites = np.asarray(owner.trace.sites)
    exact = 3.0 * np.sum(
        np.asarray(owner.trace.normals) * _quadratic_gradient(sites), axis=-1
    )

    np.testing.assert_allclose(
        np.asarray(owner.evaluate(coefficients)), exact, atol=1e-11
    )
    np.testing.assert_allclose(
        np.asarray(owner.evaluate(coefficients) + neighbor.evaluate(coefficients)),
        0.0,
        atol=1e-11,
    )


def test_p1_constants_equal_the_analytic_trace_inverse_bound() -> None:
    """``C_F = |F| n.K n / |K|`` and corner cells shared by two selected facets."""
    cells = 4
    space = _square(cells, 1)
    compiled = _compiled(space, phx.equations.TensorDiffusionAction("u", _KAPPA))
    # Right (x = 1) and bottom (y = 0) boundaries; one triangle touches both.
    boundary = _facets(
        space,
        lambda sites: np.isclose(sites[..., 0], 1.0) | np.isclose(sites[..., 1], 0.0),
    )
    trace = space.prepare_side_trace("u", boundary, rule=FacetTraceRule(points=2))
    evidence = compiled.certify_flux_stability(compiled.prepare_pointwise_flux(trace))
    normals = np.asarray(trace.normals)[:, 0]
    h = 1.0 / cells
    exact = h * np.einsum("fi,ij,fj->f", normals, _KAPPA, normals) / (0.5 * h * h)

    np.testing.assert_allclose(np.asarray(evidence.constants), exact, rtol=1e-12)
    assert np.all(np.asarray(evidence.relative_residuals) < 1e-10)
    assert sorted(np.asarray(evidence.cell_multiplicity).tolist()).count(2) == 2
    np.testing.assert_array_equal(
        np.asarray(evidence.facets), np.asarray(trace.descriptor.facets)
    )


def test_p2_constants_scale_with_diffusivity_over_mesh_size() -> None:
    """Dimensional identities: ``C`` scales like ``kappa`` and like ``1 / h``."""
    rule = FacetTraceRule("gauss-lobatto-legendre", points=3)
    constants = []
    for scale, kappa in ((1.0, 1.0), (1.0, 7.0), (0.25, 1.0)):
        space = _square(2, 2, scale)
        compiled = _compiled(space, phx.equations.DiffusionAction("u", kappa))
        left = _facets(space, lambda sites: np.isclose(sites[..., 0], 0.0))
        trace = space.prepare_side_trace("u", left, rule=rule)
        evidence = compiled.certify_flux_stability(compiled.prepare_pointwise_flux(trace))
        constants.append(np.asarray(evidence.constants))

    assert np.all(constants[0] > 0.0)
    np.testing.assert_allclose(constants[1], 7.0 * constants[0], rtol=1e-11)
    np.testing.assert_allclose(constants[2], 4.0 * constants[0], rtol=1e-11)


def test_forms_without_a_declared_flux_law_are_refused() -> None:
    """Mass-only forms define no flux; undeclared operator terms have no flux law."""
    space = _square(2, 1)
    left = _facets(space, lambda sites: np.isclose(sites[..., 0], 0.0))
    trace = space.prepare_side_trace("u", left, rule=FacetTraceRule(points=2))
    mass = _compiled(space, phx.equations.MassAction("u", 1.0))
    operator = phx.linalg.IdentityLinearOperator(trace.coefficient_space)
    undeclared = _compiled(
        space,
        phx.equations.DiffusionAction("u", 1.0),
        phx.equations.PreparedOperatorAction("u", operator, action_id="black-box"),
    )

    with pytest.raises(ValueError, match="defines no conormal flux"):
        mass.prepare_pointwise_flux(trace)
    with pytest.raises(ValueError, match="'black-box'.*without a declared conormal flux"):
        undeclared.prepare_pointwise_flux(trace)


def test_callable_diffusivity_flux_publishes_no_degree_and_is_not_certified() -> None:
    """A callable diffusivity is not polynomial along facets: no exact certification."""
    space = _square(2, 1)

    def kappa(points: Array, args: object) -> Array:
        del args
        return 1.0 + points[..., 0] ** 2

    compiled = _compiled(
        space,
        phx.equations.DiffusionAction(
            "u", phx.equations.coefficient(kappa, coefficient_id="kappa")
        ),
    )
    left = _facets(space, lambda sites: np.isclose(sites[..., 0], 1.0))
    trace = space.prepare_side_trace("u", left, rule=FacetTraceRule(points=2))
    flux = compiled.prepare_pointwise_flux(trace)
    coefficients = jnp.asarray(np.asarray(space.dof_maps[0].dof_coordinates)[:, 0])

    assert flux.descriptor.trace_degree is None
    # u = x: q = kappa(1, y) * 1 = 2 on x = 1.
    np.testing.assert_allclose(np.asarray(flux.evaluate(coefficients)), 2.0, atol=1e-12)
    with pytest.raises(ValueError, match="polynomial along the facets"):
        compiled.certify_flux_stability(flux)


def test_flux_transpose_loads_rows_whose_pairing_is_the_analytic_flux_moment() -> None:
    """``u . Q^T (w g) = int_{x=1} g n.K grad(u) ds = 61 / 12`` for ``g = y``.

    On ``x = 1``: ``n.K grad(u) = 2 (3 + 3 y) + 0.5 (3 - 4 y) = 7.5 + 4 y``, so
    ``int_0^1 y (7.5 + 4 y) dy = 61 / 12``; the GLL sites integrate it exactly.
    """
    space = _square(3, 2)
    compiled = _compiled(space, phx.equations.TensorDiffusionAction("u", _KAPPA))
    right = _facets(space, lambda sites: np.isclose(sites[..., 0], 1.0))
    trace = space.prepare_side_trace(
        "u", right, rule=FacetTraceRule("gauss-lobatto-legendre", points=4)
    )
    flux = compiled.prepare_pointwise_flux(trace)
    coefficients = jnp.asarray(_quadratic(np.asarray(space.dof_maps[0].dof_coordinates)))
    density = trace.weights * trace.sites[..., 1]
    rows = flux.linearize(jnp.zeros_like(coefficients)).vjp(density)

    assert float(jnp.vdot(coefficients, rows)) == pytest.approx(61.0 / 12.0, rel=1e-12)


def test_nonsymmetric_tensor_constant_uses_the_flux_tensor_and_the_symmetric_energy() -> (
    None
):
    """P1: ``C_F = |F| w.S^-1 w / |K|`` with ``w = K^T n`` and ``S = (K + K^T) / 2``.

    The gradient ``g`` of a P1 function is one constant vector, so
    ``||q||^2_F / a_K = |F| (w.g)^2 / (|K| g.S g)``, maximized at ``g = S^-1 w``.
    """
    cells = 3
    tensor = np.asarray([[2.0, 1.0], [0.0, 1.0]])
    space = _square(cells, 1)
    compiled = _compiled(space, phx.equations.TensorDiffusionAction("u", tensor))
    right = _facets(space, lambda sites: np.isclose(sites[..., 0], 1.0))
    trace = space.prepare_side_trace("u", right, rule=FacetTraceRule(points=2))
    evidence = compiled.certify_flux_stability(compiled.prepare_pointwise_flux(trace))
    conormal = tensor.T @ np.asarray([1.0, 0.0])
    symmetric = 0.5 * (tensor + tensor.T)
    h = 1.0 / cells
    exact = h * conormal @ np.linalg.solve(symmetric, conormal) / (0.5 * h * h)

    np.testing.assert_allclose(np.asarray(evidence.constants), exact, rtol=1e-12)
