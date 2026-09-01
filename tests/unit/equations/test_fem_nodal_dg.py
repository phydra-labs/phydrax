#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._conservation_boundary import ExtrapolationBoundary
from phydrax.discretization.fem._boundary import FiniteElementBoundarySet
from phydrax.discretization.fem._generic import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization.fem._reference import discontinuous_element
from phydrax.discretization.finite_volume._physical_boundaries import (
    NoSlipAdiabaticWallBoundary,
)
from phydrax.discretization.finite_volume._riemann import RusanovFluxPlan
from phydrax.equations._conservation import (
    compile_conservation_problem,
    ConservationProblemIR,
)
from phydrax.equations._hyperbolic_systems import (
    CompressibleNavierStokesSystem,
    EulerSystem,
)
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.equations.fem._nodal_conservation import (
    NodalDGConservationMethodPlan,
)
from phydrax.equations.fem._viscous_conservation import LDGViscousFluxPlan


def _triangle_problem(order=2):
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    mesh = CellMesh.from_triangles(
        vertices,
        np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int32),
    )
    system = EulerSystem(2)
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state",
            discontinuous_element("triangle", order),
            component_shape=(system.component_count,),
        ),
    ).prepare()
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = FiniteElementBoundarySet(
        discretization,
        {"outflow": (exterior, ExtrapolationBoundary())},
    )
    method = NodalDGConservationMethodPlan(RusanovFluxPlan())
    compiled = compile_conservation_problem(
        ConservationProblemIR("triangle-euler", "state", system, boundaries),
        discretization,
        method,
    )
    return compiled, system, discretization


def test_triangle_nodal_dg_preserves_free_stream_and_conservation():
    compiled, system, discretization = _triangle_problem()
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.2, -0.1, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    rate, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    np.testing.assert_allclose(rate, 0.0, atol=2.0e-10)
    np.testing.assert_allclose(diagnostics.conservation_rate, 0.0, atol=2.0e-10)
    assert compiled.dynamics.mass_inverse.evidence.positive_definite
    assert not compiled.dynamics.report.volume_quadrature.exact


def test_triangle_nodal_dg_interface_is_conservative_and_linearizable():
    compiled, system, discretization = _triangle_problem(order=1)
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 0.0, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    routes = discretization.dof_maps[0].cell_dofs[0]
    perturbed = state.at[routes[1], 0].multiply(1.01)
    rate, pushforward, pullback = compiled.linearize(0.0, perturbed)
    direction = jnp.linspace(-0.05, 0.05, perturbed.size).reshape(perturbed.shape)
    covector = jnp.linspace(0.04, -0.03, perturbed.size).reshape(perturbed.shape)
    tangent = pushforward(direction)
    cotangent = pullback(covector)[0]
    assert jnp.all(jnp.isfinite(rate))
    np.testing.assert_allclose(
        jnp.vdot(covector, tangent),
        jnp.vdot(cotangent, direction),
        rtol=4.0e-5,
        atol=4.0e-5,
    )


def test_tetrahedron_nodal_dg_preserves_free_stream():
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "cells",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            ),
        ),
    )
    system = EulerSystem(3)
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state",
            discontinuous_element("tetrahedron", 1),
            component_shape=(system.component_count,),
        ),
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
        ConservationProblemIR("tetrahedron-euler", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(RusanovFluxPlan()),
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.1, -0.05, 0.02, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=3.0e-10)


def test_mixed_triangle_quadrilateral_nodal_dg_uses_conservative_mortar():
    vertices = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (2.0, 0.0),
            (2.0, 1.0),
        )
    )
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "triangles",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
            CellBlock(
                "quads",
                "quadrilateral",
                np.asarray(((1, 3, 4, 2),), dtype=np.int32),
                global_ids=np.asarray((1,), dtype=np.int64),
            ),
        ),
    )
    system = EulerSystem(2)
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state",
            {
                "triangles": discontinuous_element("triangle", 2),
                "quads": discontinuous_element("quadrilateral", 2),
            },
            component_shape=(system.component_count,),
        ),
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
        ConservationProblemIR("mixed-euler", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(RusanovFluxPlan()),
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.1, -0.05, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    rate, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    np.testing.assert_allclose(rate, 0.0, atol=5.0e-10)
    np.testing.assert_allclose(diagnostics.conservation_rate, 0.0, atol=5.0e-10)
    assert len(compiled.dynamics.mortar_routes) == 1
    mortar = compiled.dynamics.mortar_routes[0].mortar
    flux = jnp.ones((mortar.physical_weights.shape[0], system.component_count))
    np.testing.assert_allclose(mortar.conservation_residual(flux), 0.0, atol=3.0e-12)


def test_prism_and_pyramid_nodal_dg_preserve_free_stream():
    cells = {
        "prism": np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 1.0),
            )
        ),
        "pyramid": np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.5, 0.5, 1.0),
            )
        ),
    }
    system = EulerSystem(3)
    for kind, points in cells.items():
        mesh = CellMesh(
            points,
            (
                CellBlock(
                    "cells",
                    kind,
                    np.arange(points.shape[0], dtype=np.int32)[None, :],
                ),
            ),
        )
        discretization = FiniteElementPlan(
            mesh,
            FiniteElementFieldSpec(
                "state",
                discontinuous_element(kind, 1),
                component_shape=(system.component_count,),
            ),
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
            ConservationProblemIR(f"{kind}-euler", "state", system, boundaries),
            discretization,
            NodalDGConservationMethodPlan(RusanovFluxPlan()),
        )
        state = jnp.broadcast_to(
            system.primitive_to_conserved(jnp.asarray((1.0, 0.1, -0.05, 0.02, 1.0))),
            discretization.field_spaces[0].vector_space.shape,
        )
        np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=2.0e-9)


def test_tetrahedron_nodal_ldg_preserves_stationary_rest_state():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "cells",
                "tetrahedron",
                np.arange(4, dtype=np.int32)[None, :],
            ),
        ),
    )
    system = CompressibleNavierStokesSystem(ConstantTransport(0.1, 0.2), 3)
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state",
            discontinuous_element("tetrahedron", 1),
            component_shape=(system.component_count,),
        ),
    ).prepare()
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = FiniteElementBoundarySet(
        discretization,
        {
            "walls": (
                exterior,
                NoSlipAdiabaticWallBoundary(jnp.zeros((3,))),
            )
        },
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR("tetrahedron-viscous", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(
            RusanovFluxPlan(),
            viscous=LDGViscousFluxPlan(),
        ),
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 0.0, 0.0, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=3.0e-9)


def test_polyhedral_three_dimensional_interface_is_conservative():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 1.0),
            (0.5, 0.5, -1.0),
        )
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "upper",
                "pyramid",
                np.asarray(((0, 1, 2, 3, 4),), dtype=np.int32),
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
            CellBlock(
                "lower",
                "pyramid",
                np.asarray(((0, 3, 2, 1, 5),), dtype=np.int32),
                global_ids=np.asarray((1,), dtype=np.int64),
            ),
        ),
    )
    system = EulerSystem(3)
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state",
            {
                "upper": discontinuous_element("pyramid", 1),
                "lower": discontinuous_element("pyramid", 1),
            },
            component_shape=(system.component_count,),
        ),
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
        ConservationProblemIR("polyhedral-euler", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(RusanovFluxPlan()),
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.1, -0.05, 0.02, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    rate, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    np.testing.assert_allclose(rate, 0.0, atol=3.0e-9)
    np.testing.assert_allclose(diagnostics.conservation_rate, 0.0, atol=3.0e-9)
    assert len(compiled.dynamics.three_dimensional_interface_routes) == 1
