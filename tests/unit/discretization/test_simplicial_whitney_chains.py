from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.discretization import PreparedSimplicialCellLocator
from phydrax.discretization.pic import UnstructuredWhitneyCurrentPlan


def _locator(
    coordinates: Array, cells: Array, kind: str
) -> PreparedSimplicialCellLocator:
    discretization = phx.discretization
    mesh = discretization.CellMesh(
        coordinates, (discretization.CellBlock("cells", kind, cells),)
    )
    prepared = discretization.FiniteElementPlan(
        mesh,
        discretization.FiniteElementFieldSpec(
            "u", discretization.lagrange_element(kind, 1)
        ),
    ).prepare()
    return discretization.PreparedSimplicialCellLocator(
        discretization.fem.prepare_finite_element_cell_map(prepared, 0),
        prepared.default_runtime.coordinates,
        discretization.SimplicialLocationPolicy(cells.shape[0], 8, cells.shape[1]),
    )


def _square_plan(capacity: int = 2) -> UnstructuredWhitneyCurrentPlan:
    return UnstructuredWhitneyCurrentPlan(
        _locator(
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))),
            jnp.asarray(((0, 1, 2), (1, 3, 2)), dtype=jnp.int32),
            "triangle",
        ),
        maximum_segments=capacity,
    )


def test_facet_split_continuity_and_positive_uniform_field_work() -> None:
    plan = _square_plan()
    start, end = jnp.asarray(((0.1, 0.2),)), jnp.asarray(((0.9, 0.8),))
    deposited = plan.deposit(start, end, jnp.asarray((2.0,)), jnp.asarray((True,)), 0.4)
    assert deposited.successful
    np.testing.assert_allclose(deposited.start_charge, (1.4, 0.2, 0.4, 0.0), atol=1e-13)
    np.testing.assert_allclose(deposited.end_charge, (0.0, 0.4, 0.2, 1.4), atol=1e-13)
    np.testing.assert_allclose(deposited.continuity_residual, 0.0, atol=1e-12)
    electric = jnp.asarray((2.0, -3.0))
    vertices = plan.locator.coordinates
    edge_cochain = (vertices[plan.edges[:, 1]] - vertices[plan.edges[:, 0]]) @ electric
    np.testing.assert_allclose(
        edge_cochain @ deposited.edge_current,
        2.0 * ((end - start) @ electric)[0] / 0.4,
        atol=1e-12,
    )
    exhausted = _square_plan(1).deposit(
        start, end, jnp.asarray((2.0,)), jnp.asarray((True,)), 0.4
    )
    assert exhausted.route_overflow
    assert not exhausted.successful


def test_phase_segment_moment_and_complex_adjoint_work() -> None:
    plan = _square_plan()
    start, end = jnp.asarray(((0.1, 0.2),)), jnp.asarray(((0.9, 0.8),))
    rate = 7.3
    query = plan.kernel.integrate_segments(
        start, end, weight="phase", phase_rate=rate, maximum_segments=2
    )
    electric = jnp.asarray((0.8, -0.3))
    vertices = plan.locator.coordinates
    cochain = (vertices[plan.edges[:, 1]] - vertices[plan.edges[:, 0]]) @ electric
    expected = (
        ((end - start) @ electric)[0] * np.exp(0.5j * rate) * np.sinc(rate / (2 * np.pi))
    )
    np.testing.assert_allclose(query.gather(cochain)[0], expected, atol=2e-13)
    complex_cochain = cochain + 1j * jnp.arange(cochain.size)
    weights = jnp.asarray((0.3 + 0.7j,))
    np.testing.assert_allclose(
        jnp.vdot(query.gather(complex_cochain), weights),
        jnp.vdot(complex_cochain, query.deposit(weights)),
        atol=2e-13,
    )


def test_nonuniform_whitney_two_reconstructs_physical_affine_flux() -> None:
    coordinates = jnp.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))
    )
    cells = jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)
    plan = UnstructuredWhitneyCurrentPlan(_locator(coordinates, cells, "tetrahedron"))
    faces = np.asarray(tuple(itertools.combinations(range(4), 3)), dtype=np.int32)
    triangles = np.asarray(coordinates)[faces]
    normals = 0.5 * np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    constant = np.asarray((1.0, -2.0, 0.5))
    slope = 0.7
    flux = np.sum(normals * (constant + slope * triangles.mean(axis=1)), axis=1)
    points = jnp.asarray(((0.2, 0.3, 0.4), (0.8, 0.3, 0.4)))
    query = plan.kernel.evaluate(points, 2, proxy="flux")
    assert query.successful.all()
    np.testing.assert_allclose(
        query.gather(flux), constant + slope * np.asarray(points), atol=2e-13
    )
    density = plan.kernel.evaluate(points, 3).gather(jnp.asarray((8.0,)))
    np.testing.assert_allclose(density, 2.0, atol=2e-13)


def test_stationary_current_and_inactive_outside_particle() -> None:
    plan = _square_plan()
    positions = jnp.asarray(((0.2, 0.2), (3.0, 4.0)))
    result = plan.deposit(
        positions, positions, jnp.asarray((1.0, 8.0)), jnp.asarray((True, False)), 0.1
    )
    assert result.successful
    np.testing.assert_allclose(result.edge_current, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.start_charge, result.end_charge, atol=1e-14)


def test_sparse_electrostatic_dirichlet_solve_matches_closed_form() -> None:
    d = phx.discretization
    coordinates = jnp.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))
    )
    cells = jnp.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=jnp.int32)
    locator = _locator(coordinates, cells, "triangle")
    charge_model = d.pic.PICChargeModelPlan(
        1.0,
        "ions",
        minimum_charge_number=0,
        maximum_charge_number=1,
        initial_charge_number=1,
    )
    plan = d.pic.UnstructuredElectrostaticPICPlan(
        locator,
        charge_model,
        jnp.asarray((True, True, True, True, False)),
        permittivity=2.0,
    )
    potential, residual, solved = plan.solve_field(
        jnp.asarray((0.0, 0.0, 0.0, 0.0, 16.0))
    )
    assert solved.successful
    # Four quarter-area cells each contribute epsilon*area*|grad(lambda_center)|²=2.
    np.testing.assert_allclose(potential, (0.0, 0.0, 0.0, 0.0, 2.0), atol=2e-12)
    np.testing.assert_allclose(residual, 0.0, atol=2e-12)
    location = locator.locate(
        jnp.asarray(((0.5, 0.1), (0.9, 0.5), (0.5, 0.9), (0.1, 0.5)))
    )
    np.testing.assert_allclose(
        plan.gather_electric(location, potential),
        ((0.0, -4.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (-4.0, 0.0, 0.0)),
        atol=2e-12,
    )
