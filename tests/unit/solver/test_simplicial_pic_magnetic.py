from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.solver import UnstructuredMaxwellPICFieldSolver
from phydrax.solver.maxwell import CompatibleMaxwellState, MaxwellPrimaryState


def test_pic_gathers_nonuniform_physical_b_without_permeability_rescaling() -> None:
    d = phx.discretization
    coordinates = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 4.0),
            (0.5, 0.75, 1.0),
        )
    )
    cells = jnp.asarray(
        ((4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)), dtype=jnp.int32
    )
    mesh = d.CellMesh(coordinates, (d.CellBlock("tet", "tetrahedron", cells),))
    prepared = d.FiniteElementPlan(
        mesh, d.FiniteElementFieldSpec("u", d.lagrange_element("tetrahedron", 1))
    ).prepare()
    locator = d.PreparedSimplicialCellLocator(
        d.fem.prepare_finite_element_cell_map(prepared, 0),
        prepared.default_runtime.coordinates,
        d.SimplicialLocationPolicy(4, 8, 4),
    )
    complex_ = d.FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
    maxwell = phx.solver.maxwell.UnstructuredMaxwellPlan(
        complex_,
        phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(permeability=3.0),
        spectral_upper_bound=100.0,
        courant_factor=0.9,
        boundary="relative",
    ).prepare()
    solver = UnstructuredMaxwellPICFieldSolver(
        maxwell, d.pic.UnstructuredWhitneyCurrentPlan(locator, maximum_segments=4)
    )
    faces = np.asarray(d.tetrahedral_connectivity(cells, coordinates.shape[0]).faces)
    triangles = np.asarray(coordinates)[faces]
    normal = 0.5 * np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    constant = np.asarray((1.0, -0.2, 0.7))
    slope = 0.4
    flux = jnp.asarray(
        np.sum(normal * (constant + slope * triangles.mean(axis=1)), axis=1)
    )
    initial = maxwell.initialize()
    field = CompatibleMaxwellState(
        MaxwellPrimaryState(
            initial.primary.electric_displacement, flux, initial.primary.charge
        ),
        initial.auxiliary,
        initial.observations,
    )
    points = jnp.asarray(((0.2, 0.3, 0.2), (0.8, 0.6, 0.7)))
    electric, magnetic, supported = solver.gather_fields(
        0, points, jnp.asarray((True, True)), field
    )
    assert supported.all()
    np.testing.assert_allclose(electric, 0.0, atol=2e-12)
    np.testing.assert_allclose(
        magnetic, constant + slope * np.asarray(points), atol=2e-12
    )
    velocity = jnp.asarray(((0.3, 0.2, -0.1), (0.1, -0.2, 0.4)))
    np.testing.assert_allclose(
        jnp.cross(velocity, magnetic),
        np.cross(np.asarray(velocity), constant + slope * np.asarray(points)),
        atol=2e-12,
    )
