#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from phydrax.discretization._cell_ordering import meshio_reference_nodes
from phydrax.discretization._conservation_boundary import ExtrapolationBoundary
from phydrax.discretization.fem._boundary import FiniteElementBoundarySet
from phydrax.discretization.fem._generic import FiniteElementFieldSpec, FiniteElementPlan
from phydrax.discretization.fem._reference import discontinuous_element
from phydrax.discretization.fem._spectral_hp_io import read_finite_element_mesh
from phydrax.discretization.finite_volume._riemann import RusanovFluxPlan
from phydrax.equations._conservation import (
    compile_conservation_problem,
    ConservationProblemIR,
)
from phydrax.equations._hyperbolic_systems import EulerSystem
from phydrax.equations.fem._nodal_conservation import (
    NodalDGConservationMethodPlan,
)


def test_meshio_quadratic_triangle_preserves_coordinate_dofs(tmp_path):
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, -0.1, 0.0),
            (0.55, 0.55, 0.0),
            (-0.1, 0.5, 0.0),
        )
    )
    path = tmp_path / "curved-triangle.vtu"
    meshio.write(path, meshio.Mesh(points, [("triangle6", np.arange(6)[None, :])]))

    imported = read_finite_element_mesh(path)
    assert imported.report.curved
    assert imported.report.cell_kinds == ("triangle",)
    assert imported.coordinate_spec.coordinates.shape == (6, 2)
    routes = np.asarray(imported.coordinate_spec.geometry_dofs[0])
    assert routes.shape == (1, 6)
    coordinate_element = imported.coordinate_spec.elements[0]
    physical_nodes = np.asarray(imported.coordinate_spec.coordinates)[routes[0]]
    reference_nodes = np.asarray(coordinate_element.reference_nodes)
    curved_index = int(np.argmin(np.sum((reference_nodes - (0.5, 0.0)) ** 2, axis=1)))
    np.testing.assert_allclose(physical_nodes[curved_index], (0.5, -0.1))

    discretization = FiniteElementPlan(
        imported.mesh,
        FiniteElementFieldSpec(
            "state", discontinuous_element("triangle", 2), component_shape=(4,)
        ),
        coordinate_spec=imported.coordinate_spec,
    ).prepare()
    system = EulerSystem(2)
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = FiniteElementBoundarySet(
        discretization,
        {"outflow": (exterior, ExtrapolationBoundary())},
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR("curved-triangle", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(RusanovFluxPlan()),
    )
    quality = compiled.dynamics.geometry_quality
    assert quality.passed
    assert jnp.all(quality.minimum_jacobian > quality.determinant_floor)
    assert jnp.all(jnp.isfinite(quality.maximum_condition_number))
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.1, -0.05, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=3.0e-9)
    assert discretization.coordinate_elements[0].degree == 2


def test_curved_mixed_mortar_uses_both_high_order_coordinate_traces(tmp_path):
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.5, 0.0, 0.0),
            (1.1, 0.5, 0.0),
            (0.5, 0.5, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.5, 0.0, 0.0),
            (2.0, 0.5, 0.0),
            (1.5, 1.0, 0.0),
            (1.55, 0.5, 0.0),
        )
    )
    cells = [
        ("triangle6", np.asarray(((0, 1, 2, 3, 4, 5),), dtype=np.int32)),
        ("quad9", np.asarray(((1, 6, 7, 2, 8, 9, 10, 4, 11),), dtype=np.int32)),
    ]
    path = tmp_path / "curved-mixed.vtu"
    meshio.write(path, meshio.Mesh(points, cells))
    imported = read_finite_element_mesh(path)
    system = EulerSystem(2)
    discretization = FiniteElementPlan(
        imported.mesh,
        FiniteElementFieldSpec(
            "state",
            {
                imported.mesh.blocks[0].name: discontinuous_element("triangle", 2),
                imported.mesh.blocks[1].name: discontinuous_element("quadrilateral", 2),
            },
            component_shape=(system.component_count,),
        ),
        coordinate_spec=imported.coordinate_spec,
    ).prepare()
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = FiniteElementBoundarySet(
        discretization,
        {"outflow": (exterior, ExtrapolationBoundary())},
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR("curved-mixed", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(RusanovFluxPlan()),
    )
    assert len(compiled.dynamics.mortar_routes) == 1
    mortar = compiled.dynamics.mortar_routes[0].mortar
    assert mortar.evidence.coordinates_compatible
    assert float(jnp.max(mortar.physical_coordinates[:, 0])) > 1.05
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.1, -0.05, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=5.0e-9)


@pytest.mark.parametrize("cell_type", ("hexahedron27", "wedge18", "pyramid14"))
def test_meshio_quadratic_hybrid_cells_preserve_all_geometry_nodes(tmp_path, cell_type):
    points = meshio_reference_nodes(cell_type)
    path = tmp_path / f"{cell_type}.msh"
    meshio.write(
        path,
        meshio.Mesh(points, [(cell_type, np.arange(points.shape[0])[None, :])]),
        file_format="gmsh22",
    )
    imported = read_finite_element_mesh(path)
    assert imported.report.geometry_orders == (2,)
    assert imported.report.curved
    routes = np.asarray(imported.coordinate_spec.geometry_dofs[0])
    assert routes.shape == (1, points.shape[0])
    element = imported.coordinate_spec.elements[0]
    basis = np.asarray(element.tabulate(element.reference_nodes)[0])
    physical = basis @ np.asarray(imported.coordinate_spec.coordinates)[routes[0]]
    np.testing.assert_allclose(
        physical,
        np.asarray(imported.coordinate_spec.coordinates)[routes[0]],
        atol=3.0e-10,
    )
