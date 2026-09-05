#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import opt_einsum as oe

import phydrax as phx
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._conservation_boundary import ExtrapolationBoundary
from phydrax.discretization._reference_cell import (
    facet_orientation_actions,
)
from phydrax.discretization.fem._boundary import (
    FiniteElementBoundarySet,
    FiniteElementPeriodicFacetPair,
    FiniteElementPeriodicTransform,
)
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
    ScalarConservationSystem,
)
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.equations.fem._nodal_conservation import (
    NodalDGConservationMethodPlan,
)
from phydrax.equations.fem._viscous_conservation import ViscousDGPlan


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
    evidence = phx.equations.fem.certify_conservation_sensitivity(
        compiled.dynamics,
        0.0,
        perturbed,
        direction,
        covector,
        epsilon=1.0e-5,
        tolerance=5.0e-4,
    )
    assert evidence.valid


def test_discontinuous_mass_strategies_share_affine_semantics():
    from phydrax.equations.fem._mass_inverse import (
        PreparedDiscontinuousMassInverse,
    )
    from phydrax.integration import GaussLegendreRule, ReferenceTriangleRule

    compiled, _system, discretization = _triangle_problem(order=2)
    auto = compiled.dynamics.mass_inverse
    assert auto.strategies == ("affine_scaled",)
    rule = ReferenceTriangleRule(GaussLegendreRule(4))
    exact = PreparedDiscontinuousMassInverse(
        discretization, "state", rule, strategy="exact_batched"
    )
    weight_adjusted = PreparedDiscontinuousMassInverse(
        discretization, "state", rule, strategy="weight_adjusted"
    )
    residual = jnp.linspace(
        -0.2,
        0.3,
        discretization.field_spaces[0].vector_space.size,
    ).reshape(discretization.field_spaces[0].vector_space.shape)
    expected = exact.apply(residual)
    np.testing.assert_allclose(auto.apply(residual), expected, rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(
        weight_adjusted.apply(residual), expected, rtol=3e-10, atol=3e-10
    )
    assert auto.evidence.resident_factor_bytes < exact.evidence.resident_factor_bytes


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
        entropy_pair = phx.equations.ideal_gas_euler_entropy_pair(system)
        entropy_compiled = compile_conservation_problem(
            ConservationProblemIR(f"{kind}-entropy-euler", "state", system, boundaries),
            discretization,
            NodalDGConservationMethodPlan(
                RusanovFluxPlan(),
                entropy_stability=phx.equations.fem.EntropyStableDGPlan(
                    phx.discretization.EntropyConservativeEulerFluxPlan(),
                    entropy_pair,
                    tolerance=3.0e-8,
                    boundary_contracts=(
                        phx.equations.fem.PhysicalBoundaryEntropyContract.transparent(
                            ExtrapolationBoundary().boundary_id
                        ),
                    ),
                ),
            ),
            entropy_pair=entropy_pair,
        )
        assert entropy_compiled.dynamics.entropy_operators[0] is not None
        np.testing.assert_allclose(entropy_compiled(0.0, state), 0.0, atol=8.0e-8)


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
            viscous=ViscousDGPlan(
                boundary_closures=(
                    phx.equations.fem.ViscousBoundaryClosure(
                        boundaries.patches[0].boundary.boundary_id
                    ),
                )
            ),
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


def test_interval_p_zero_nodal_dg_preserves_constant_state():
    mesh = CellMesh(
        np.asarray(((0.0,), (1.0,))),
        (
            CellBlock(
                "cells",
                "interval",
                np.asarray(((0, 1),), dtype=np.int32),
            ),
        ),
    )
    system = ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="interval-advection",
    )
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state",
            discontinuous_element("interval", 0),
            component_shape=(1,),
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
        ConservationProblemIR("interval-p0", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(RusanovFluxPlan()),
    )
    state = jnp.ones(discretization.field_spaces[0].vector_space.shape)
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=2.0e-12)


def test_hexahedron_general_nodal_dg_preserves_free_stream():
    points = np.asarray(
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
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "cells",
                "hexahedron",
                np.arange(8, dtype=np.int32)[None, :],
            ),
        ),
    )
    system = EulerSystem(3)
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state",
            discontinuous_element("hexahedron", 1),
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
        ConservationProblemIR("hexahedron-nodal", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(RusanovFluxPlan()),
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.1, -0.05, 0.02, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=3.0e-10)


def test_nodal_dg_transformed_periodicity_is_conservative():
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "cells",
                "quadrilateral",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            ),
        ),
    )
    system = EulerSystem(2)
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state",
            discontinuous_element("quadrilateral", 2),
            component_shape=(system.component_count,),
        ),
    ).prepare()
    exterior = np.asarray(discretization.exterior_facet_domain.entity_indices)
    local_facets = np.asarray(discretization.exterior_facet_domain.owner_local_entities)
    by_local = {
        int(local): int(facet)
        for facet, local in zip(exterior, local_facets, strict=True)
    }
    reverse = facet_orientation_actions("edge")[1]
    identity_components = np.eye(system.component_count)
    boundaries = FiniteElementBoundarySet(
        discretization,
        {},
        periodic_pairs=(
            FiniteElementPeriodicFacetPair(
                by_local[0],
                by_local[2],
                transform=FiniteElementPeriodicTransform(
                    np.eye(2),
                    np.asarray((0.0, 1.0)),
                    reverse,
                    component_matrix=identity_components,
                ),
            ),
            FiniteElementPeriodicFacetPair(
                by_local[3],
                by_local[1],
                transform=FiniteElementPeriodicTransform(
                    np.eye(2),
                    np.asarray((1.0, 0.0)),
                    reverse,
                    component_matrix=identity_components,
                ),
            ),
        ),
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR("periodic-nodal", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(RusanovFluxPlan()),
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.1, -0.05, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    faces = compiled.dynamics.face_fluxes(0.0, state)
    assert faces.route_kinds == ("periodic", "periodic")
    assert compiled.dynamics.stable_step_evidence(state).positive
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=3.0e-10)
    coordinates = discretization.dof_maps[0].dof_coordinates
    perturbed = state.at[:, 0].multiply(
        1.0 + 0.05 * jnp.sin(2.0 * jnp.pi * coordinates[:, 0])
    )
    _rate, diagnostics = compiled.residual_with_diagnostics(0.0, perturbed)
    np.testing.assert_allclose(diagnostics.conservation_rate, 0.0, atol=5.0e-9)


def test_nodal_entropy_plan_prepares_formal_simplex_sbp_operator():
    _compiled, system, discretization = _triangle_problem(order=2)
    entropy_pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    entropy_plan = phx.equations.fem.EntropyStableDGPlan(
        phx.discretization.EntropyConservativeEulerFluxPlan(),
        entropy_pair,
        formulation="generalized_sbp",
        tolerance=3.0e-8,
        boundary_contracts=(
            phx.equations.fem.PhysicalBoundaryEntropyContract.transparent(
                ExtrapolationBoundary().boundary_id
            ),
        ),
    )
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = FiniteElementBoundarySet(
        discretization,
        {"outflow": (exterior, ExtrapolationBoundary())},
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR("entropy-triangle", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(RusanovFluxPlan(), entropy_stability=entropy_plan),
        entropy_pair=entropy_pair,
    )
    assert len(compiled.dynamics.entropy_operators) == 1
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.1, -0.05, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=5.0e-9)
    assert compiled.dynamics.entropy_operators[0].formal_sbp
    filter_ = phx.equations.fem.EntropyFilterPlan(
        density_floor=1.0e-6,
        pressure_floor=1.0e-6,
    ).prepare(compiled.dynamics)
    troubled = state.at[0, 0].set(1.0e-5).at[0, -1].set(1.0e-5)
    filtered, filter_evidence = filter_.filter(0.0, troubled)
    assert filter_evidence.successful
    assert filter_evidence.applied
    weights = jnp.sum(compiled.dynamics.mass_inverse.mass_matrices[0], axis=1)
    routes = compiled.dynamics.mass_inverse.routes[0]
    before = oe.contract("ci,civ->v", weights, troubled[routes])
    after = oe.contract("ci,civ->v", weights, filtered[routes])
    np.testing.assert_allclose(after, before, atol=3.0e-9)


def test_executable_equilibrium_family_cancels_discrete_source_imbalance():
    _compiled, system, discretization = _triangle_problem(order=2)

    def equilibrium_state(time, coordinates, args):
        del time, args
        density = 1.0 + 0.1 * coordinates[:, 0]
        primitive = jnp.stack(
            (
                density,
                jnp.zeros_like(density),
                jnp.zeros_like(density),
                jnp.ones_like(density),
            ),
            axis=-1,
        )
        return system.primitive_to_conserved(primitive)

    equilibrium = phx.equations.fem.WellBalancedEquilibriumPlan(
        equilibrium_state,
        equilibrium_id="linear-density-equilibrium",
    )
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = FiniteElementBoundarySet(
        discretization,
        {"outflow": (exterior, ExtrapolationBoundary())},
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR("well-balanced", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(RusanovFluxPlan(), equilibrium=equilibrium),
    )
    state = equilibrium_state(
        jnp.asarray(0.0), discretization.dof_maps[0].dof_coordinates, None
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=3.0e-10)
    perturbed = state.at[0, 0].multiply(1.01)
    assert jnp.linalg.norm(compiled(0.0, perturbed)) > 0.0


def test_mixed_triangle_viscous_dg_preserves_stationary_rest_state():
    _euler_compiled, _euler, discretization = _triangle_problem(order=2)
    system = CompressibleNavierStokesSystem(ConstantTransport(0.1, 0.2), 2)
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = FiniteElementBoundarySet(
        discretization,
        {
            "walls": (
                exterior,
                NoSlipAdiabaticWallBoundary(jnp.zeros((2,))),
            )
        },
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR("triangle-viscous", "state", system, boundaries),
        discretization,
        NodalDGConservationMethodPlan(
            RusanovFluxPlan(),
            viscous=ViscousDGPlan(
                formulation="entropy_br1",
                boundary_closures=(
                    phx.equations.fem.ViscousBoundaryClosure(
                        boundaries.patches[0].boundary.boundary_id
                    ),
                ),
            ),
        ),
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 0.0, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=5.0e-9)
    assert compiled.dynamics.stable_step_evidence(state).maximum_diffusive_rate > 0.0
    residual, pushforward, pullback = compiled.dynamics.viscous_linearize(0.0, state)
    direction = jnp.linspace(-0.02, 0.02, state.size).reshape(state.shape)
    cotangent = jnp.linspace(0.03, -0.01, state.size).reshape(state.shape)
    tangent = pushforward(direction)
    adjoint = pullback(cotangent)[0]
    assert jnp.all(jnp.isfinite(residual))
    np.testing.assert_allclose(
        jnp.vdot(cotangent, tangent),
        jnp.vdot(adjoint, direction),
        rtol=5.0e-5,
        atol=5.0e-5,
    )
