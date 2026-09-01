#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import meshio
import numpy as np

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
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.1, -0.05, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=3.0e-9)
    assert discretization.coordinate_elements[0].degree == 2
