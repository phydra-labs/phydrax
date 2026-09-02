import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization.fem._boundary import (
    FiniteElementBoundarySet,
    FiniteElementPeriodicFacetPair,
)
from phydrax.discretization.fem._generic import (
    FiniteElementCoordinateSpec,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization.fem._high_order import ReferenceNodalFamily
from phydrax.discretization.fem._reference import discontinuous_element
from phydrax.discretization.fem._sbp import (
    MappedTensorMetricPlan,
    MetricFacePair,
    TensorGLLSBPPlan,
)
from phydrax.discretization.finite_difference import TensorSBPDiscretization
from phydrax.discretization.finite_volume._boundary import FiniteVolumeBoundarySet
from phydrax.discretization.finite_volume._physical_boundaries import (
    NoSlipAdiabaticWallBoundary,
    SlipWallBoundary,
)
from phydrax.discretization.finite_volume._riemann import (
    EntropyConservativeEulerFluxPlan,
    EntropyStableEulerFluxPlan,
    RusanovFluxPlan,
)
from phydrax.equations._conservation import (
    compile_conservation_problem,
    ConservationProblemIR,
)
from phydrax.equations._entropy_pair import ideal_gas_euler_entropy_pair
from phydrax.equations._hyperbolic_systems import (
    CompressibleNavierStokesSystem,
    EulerSystem,
    ScalarConservationSystem,
)
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.equations.fem._conservation import (
    DGSEMConservationMethodPlan,
    PreparedDGSEMConservationDynamics,
    sample_dgsem_flux_compatibility,
)
from phydrax.equations.fem._entropy_filter import EntropyFilterPlan
from phydrax.equations.fem._viscous_conservation import (
    entropy_diffusion_evidence,
    ViscousDGPlan,
)
from phydrax.solver._fixed_step import SSPRK33FixedStepMethod


def _quad_discretization(order=2, *, curved=False):
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    mesh = CellMesh(
        vertices,
        (CellBlock("cells", "quadrilateral", np.asarray(((0, 1, 2, 3),))),),
    )
    system = EulerSystem(2)
    field = FiniteElementFieldSpec(
        "state",
        discontinuous_element("quadrilateral", order),
        component_shape=(system.component_count,),
    )
    coordinate_spec = None
    if curved:
        family = ReferenceNodalFamily("quadrilateral", order)
        coordinate_element = family.finite_element()
        xi, eta = np.meshgrid(
            np.asarray(family.nodes_by_axis[0]),
            np.asarray(family.nodes_by_axis[1]),
            indexing="ij",
        )
        bubble = xi * (1.0 - xi) * eta * (1.0 - eta)
        coordinates = np.stack((xi + 0.2 * bubble, eta - 0.15 * bubble), axis=-1).reshape(
            (-1, 2)
        )
        coordinate_spec = FiniteElementCoordinateSpec(
            {"cells": coordinate_element},
            {"cells": np.arange(coordinates.shape[0]).reshape((1, -1))},
            coordinates,
        )
    discretization = FiniteElementPlan(
        mesh,
        field,
        coordinate_spec=coordinate_spec,
    ).prepare()
    return system, discretization


def _hex_discretization(order=1, *, curved=False):
    vertices = np.asarray(
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
        vertices,
        (
            CellBlock(
                "cells",
                "hexahedron",
                np.asarray(((0, 1, 2, 3, 4, 5, 6, 7),)),
            ),
        ),
    )
    system = EulerSystem(3)
    field = FiniteElementFieldSpec(
        "state",
        discontinuous_element("hexahedron", order),
        component_shape=(system.component_count,),
    )
    coordinate_spec = None
    if curved:
        family = ReferenceNodalFamily("hexahedron", order)
        coordinate_element = family.finite_element()
        xi, eta, zeta = np.meshgrid(
            *(np.asarray(nodes) for nodes in family.nodes_by_axis),
            indexing="ij",
        )
        bubble = xi * (1.0 - xi) * eta * (1.0 - eta) * zeta * (1.0 - zeta)
        coordinates = np.stack(
            (xi + 0.1 * bubble, eta - 0.08 * bubble, zeta + 0.06 * bubble),
            axis=-1,
        ).reshape((-1, 3))
        coordinate_spec = FiniteElementCoordinateSpec(
            {"cells": coordinate_element},
            {"cells": np.arange(coordinates.shape[0]).reshape((1, -1))},
            coordinates,
        )
    return system, FiniteElementPlan(
        mesh, field, coordinate_spec=coordinate_spec
    ).prepare()


def _sampled_evidence(system, interface_flux):
    entropy_pair = ideal_gas_euler_entropy_pair(system)
    left_primitive = jnp.asarray(
        (
            (1.0,) + (0.2,) + (0.05,) * (system.dimension - 1) + (1.0,),
            (0.8,) + (-0.1,) + (0.03,) * (system.dimension - 1) + (0.9,),
        )
    )
    right_primitive = jnp.asarray(
        (
            (0.9,) + (0.1,) + (-0.04,) * (system.dimension - 1) + (0.95,),
            (1.1,) + (0.15,) + (0.02,) * (system.dimension - 1) + (1.2,),
        )
    )
    volume_flux = EntropyConservativeEulerFluxPlan()
    certificate = sample_dgsem_flux_compatibility(
        system,
        volume_flux,
        interface_flux,
        entropy_pair,
        system.primitive_to_conserved(left_primitive),
        system.primitive_to_conserved(right_primitive),
        tolerance=2.0e-5,
    )
    return entropy_pair, volume_flux, certificate


def _compiled(*, curved=False, stable=True, entropy=True):
    system, discretization = _quad_discretization(curved=curved)
    interface_flux = (
        EntropyStableEulerFluxPlan() if stable else EntropyConservativeEulerFluxPlan()
    )
    pair = None
    compatibility = None
    volume_flux = EntropyConservativeEulerFluxPlan()
    if entropy:
        pair, volume_flux, compatibility = _sampled_evidence(system, interface_flux)
    method = DGSEMConservationMethodPlan(
        volume_flux,
        interface_flux,
        compatibility=compatibility,
    )
    problem = ConservationProblemIR("periodic-euler", "state", system, None)
    compiled = compile_conservation_problem(
        problem,
        discretization,
        method,
        entropy_pair=pair,
    )
    return compiled, system, discretization


def _constant_state(system, discretization):
    primitive = jnp.asarray((1.0,) + (0.15,) * system.dimension + (1.0,))
    conserved = system.primitive_to_conserved(primitive)
    return jnp.broadcast_to(
        conserved,
        discretization.field_spaces[0].vector_space.shape,
    )


def test_element_local_gll_sbp_has_positive_norm_and_exact_defect_evidence():
    sbp = TensorGLLSBPPlan(5).prepare()
    np.testing.assert_array_less(0.0, np.asarray(sbp.norm_weights))
    np.testing.assert_allclose(
        np.asarray(sbp.derivative_matrix @ jnp.ones((6,))),
        0.0,
        atol=sbp.report.tolerance,
    )
    boundary = (
        sbp.restriction.T @ sbp.boundary_weights @ sbp.boundary_normals @ sbp.restriction
    )
    np.testing.assert_allclose(
        np.asarray(
            sbp.norm_matrix @ sbp.derivative_matrix
            + sbp.derivative_matrix.T @ sbp.norm_matrix
        ),
        np.asarray(boundary),
        atol=sbp.report.tolerance,
    )
    assert sbp.report.passed
    assert sbp.report.sbp_identity_defect.shape == (6, 6)


@pytest.mark.parametrize("dimension", (2, 3))
def test_mapped_tensor_metric_plan_has_discrete_gcl_and_free_stream_evidence(dimension):
    sbp = TensorGLLSBPPlan(3).prepare()
    axes = np.meshgrid(*(np.asarray(sbp.nodes),) * dimension, indexing="ij")
    bubble = np.ones_like(axes[0])
    for axis in axes:
        bubble = bubble * axis * (1.0 - axis)
    coordinates = np.stack(
        tuple(
            axis + (0.08 * (-1.0) ** index) * bubble for index, axis in enumerate(axes)
        ),
        axis=-1,
    )[None, ...]
    metrics = MappedTensorMetricPlan(sbp, dimension).prepare(coordinates)
    assert metrics.report.passed
    assert float(metrics.report.determinant_margin) > 0.0
    np.testing.assert_allclose(
        np.asarray(metrics.report.metric_identity_defect),
        0.0,
        atol=metrics.report.tolerance,
    )
    np.testing.assert_allclose(
        np.asarray(metrics.report.free_stream_residual),
        0.0,
        atol=metrics.report.tolerance,
    )


@pytest.mark.parametrize("dimension", (2, 3))
def test_mapped_tensor_shared_face_is_watertight_with_opposite_scaled_normals(
    dimension,
):
    sbp = TensorGLLSBPPlan(2).prepare()
    axes = np.meshgrid(*(np.asarray(sbp.nodes),) * dimension, indexing="ij")
    first = np.stack(tuple(axes), axis=-1)
    translation = np.zeros((dimension,))
    translation[0] = 1.0
    second = first + translation
    pair = MetricFacePair(0, 0, 1, 1, 0, 0)
    metrics = MappedTensorMetricPlan(sbp, dimension).prepare(
        np.stack((first, second), axis=0),
        face_pairs=(pair,),
    )
    assert metrics.report.watertight_face_position_defect.shape == (1,)
    np.testing.assert_allclose(
        metrics.report.watertight_face_position_defect,
        0.0,
        atol=metrics.report.tolerance,
    )
    np.testing.assert_allclose(
        metrics.report.opposite_scaled_normal_defect,
        0.0,
        atol=metrics.report.tolerance,
    )


def test_arbitrary_normal_flux_has_conservative_orientation():
    system = EulerSystem(2)
    flux = RusanovFluxPlan()
    left = system.primitive_to_conserved(jnp.asarray(((1.0, 0.3, -0.1, 1.0),)))
    right = system.primitive_to_conserved(jnp.asarray(((0.8, -0.2, 0.15, 0.9),)))
    normal = jnp.asarray(((0.6, 0.8),))
    forward = flux.normal_face_flux(system, left, right, normal)
    reverse = flux.normal_face_flux(system, right, left, -normal)
    np.testing.assert_allclose(
        np.asarray(forward.normal_flux),
        -np.asarray(reverse.normal_flux),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(forward.max_speed, reverse.max_speed)


def test_sampled_flux_compatibility_separates_entropy_evidence():
    system = EulerSystem(2)
    pair, volume, certificate = _sampled_evidence(system, EntropyStableEulerFluxPlan())
    assert certificate.system_id == system.system_id
    assert certificate.entropy_pair_id == pair.pair_id
    assert certificate.volume_flux_id == volume.flux_id
    assert certificate.volume_entropy_conservative
    assert certificate.interface_entropy_stable
    assert certificate.boundary_evidence == "periodic_pair_cancellation"
    assert certificate.source_evidence == "absent"
    assert certificate.viscous_evidence == "absent"
    assert certificate.sampled_periodic_entropy_compatibility
    assert float(jnp.max(certificate.interface_entropy_residual)) <= certificate.tolerance


@pytest.mark.parametrize("curved", (False, True))
def test_periodic_quad_free_stream_is_real_fem_compiler_execution(curved):
    compiled, system, discretization = _compiled(curved=curved)
    state = _constant_state(system, discretization)
    weak = compiled.weak_residual(0.0, state)
    rate = compiled.mass_inverted_rate(0.0, state)
    np.testing.assert_allclose(weak, 0.0, atol=3.0e-5)
    np.testing.assert_allclose(rate, 0.0, atol=3.0e-5)
    report = compiled.dynamics.report
    np.testing.assert_allclose(
        compiled.dynamics.metrics.report.periodic_face_translation_defect,
        0.0,
        atol=compiled.dynamics.metrics.report.tolerance,
    )
    assert report.passed
    assert report.finite_element_compilation_id == (
        compiled.dynamics.compiled_finite_element_problem.compilation_id
    )
    assert "pairwise-volume-flux" in tuple(
        action.action_kind
        for action in compiled.dynamics.compiled_finite_element_problem._action_ir.actions
    )
    facet_domain = compiled.dynamics.compiled_finite_element_problem.form.actions[
        1
    ].domain
    assert facet_domain.neighbour_trace_permutations.shape[0] == report.facet_route_count
    assert jnp.all(facet_domain.periodic_face_mask)
    assert not isinstance(compiled.discretization, TensorSBPDiscretization)
    assert (
        compiled.discretization_bundle.records[0].artifact_id
        == discretization.prepared_id
    )


@pytest.mark.parametrize("curved", (False, True))
def test_periodic_hex_free_stream_and_opposite_face_normals(curved):
    system, discretization = _hex_discretization(order=2 if curved else 1, curved=curved)
    method = DGSEMConservationMethodPlan(
        EntropyConservativeEulerFluxPlan(),
        RusanovFluxPlan(),
    )
    problem = ConservationProblemIR("periodic-hex-euler", "state", system, None)
    compiled = compile_conservation_problem(problem, discretization, method)
    state = _constant_state(system, discretization)
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=4.0e-5)
    np.testing.assert_allclose(
        compiled.dynamics.metrics.report.opposite_scaled_normal_defect,
        0.0,
        atol=compiled.dynamics.metrics.report.tolerance,
    )
    assert compiled.dynamics.report.facet_route_count == 3


def test_pair_fluxes_cancel_and_global_conservation_rate_is_zero():
    compiled, system, discretization = _compiled(entropy=False)
    coordinates = discretization.dof_maps[0].dof_coordinates
    primitive = jnp.stack(
        (
            1.0 + 0.08 * jnp.sin(2.0 * jnp.pi * coordinates[:, 0]),
            0.2 + 0.03 * jnp.cos(2.0 * jnp.pi * coordinates[:, 1]),
            -0.1 + 0.02 * jnp.sin(2.0 * jnp.pi * coordinates[:, 1]),
            1.0 + 0.05 * jnp.cos(2.0 * jnp.pi * coordinates[:, 0]),
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    weak = compiled.weak_residual(0.0, state)
    rate, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    np.testing.assert_allclose(
        compiled.dynamics.mass_operator.mv(rate),
        -weak,
        rtol=3.0e-6,
        atol=3.0e-6,
    )
    np.testing.assert_allclose(jnp.sum(weak, axis=0), 0.0, atol=5.0e-5)
    np.testing.assert_allclose(diagnostics.conservation_rate, 0.0, atol=5.0e-5)
    faces = compiled.face_fluxes(0.0, state)
    assert faces.integrated_flux.shape == (2, system.component_count)
    np.testing.assert_allclose(
        jnp.sum(compiled.dynamics.scalar_mass_weights[:, None] * rate, axis=0),
        0.0,
        atol=5.0e-5,
    )


@pytest.mark.parametrize("stable", (False, True))
def test_sampled_entropy_rate_is_conservative_or_dissipative(stable):
    compiled, system, discretization = _compiled(stable=stable)
    coordinates = discretization.dof_maps[0].dof_coordinates
    primitive = jnp.stack(
        (
            1.0 + 0.04 * jnp.sin(2.0 * jnp.pi * coordinates[:, 0]),
            0.15 + 0.02 * jnp.cos(2.0 * jnp.pi * coordinates[:, 1]),
            0.03 * jnp.sin(2.0 * jnp.pi * coordinates[:, 1]),
            1.0 + 0.03 * jnp.cos(2.0 * jnp.pi * coordinates[:, 0]),
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    _rate, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    assert diagnostics.sampled_entropy_inequality
    assert diagnostics.admissible
    if stable:
        assert float(diagnostics.convective_entropy_rate) <= 7.0e-5
        assert float(diagnostics.interface_entropy_production) <= 7.0e-5
    else:
        np.testing.assert_allclose(diagnostics.convective_entropy_rate, 0.0, atol=7.0e-5)


def test_rate_and_weak_residual_linearizations_have_exact_jvp_vjp_pairing():
    compiled, system, discretization = _compiled()
    state = _constant_state(system, discretization)
    state = state.at[0].set(
        system.primitive_to_conserved(jnp.asarray((1.02, 0.17, -0.03, 1.01)))
    )
    direction = jnp.linspace(-0.1, 0.1, state.size).reshape(state.shape)
    covector = jnp.linspace(0.07, -0.04, state.size).reshape(state.shape)
    residual, pushforward, pullback = compiled.linearize(0.0, state)
    tangent = pushforward(direction)
    cotangent = pullback(covector)[0]
    np.testing.assert_allclose(residual, compiled(0.0, state), atol=2.0e-6)
    np.testing.assert_allclose(
        jnp.vdot(covector, tangent),
        jnp.vdot(cotangent, direction),
        rtol=3.0e-5,
        atol=3.0e-5,
    )
    weak, weak_jvp, weak_vjp = compiled.dynamics.linearize_weak_residual(0.0, state)
    np.testing.assert_allclose(weak, compiled.weak_residual(0.0, state), atol=2.0e-6)
    np.testing.assert_allclose(
        jnp.vdot(covector, weak_jvp(direction)),
        jnp.vdot(weak_vjp(covector)[0], direction),
        rtol=3.0e-5,
        atol=3.0e-5,
    )


def test_stable_step_exposes_positive_rate_evidence():
    compiled, system, discretization = _compiled(entropy=False)
    evidence = compiled.dynamics.stable_step_evidence(
        _constant_state(system, discretization), cfl=0.3
    )
    assert evidence.positive
    assert float(evidence.step) > 0.0
    assert float(evidence.maximum_nodal_rate) > 0.0
    np.testing.assert_allclose(
        compiled.stable_step(_constant_state(system, discretization), cfl=0.3),
        evidence.step,
    )


def test_compiler_rejects_unsupported_system_cell_metric_and_entropy_scope():
    system, discretization = _quad_discretization()
    method = DGSEMConservationMethodPlan(
        EntropyConservativeEulerFluxPlan(), RusanovFluxPlan()
    )
    scalar = ScalarConservationSystem(
        2,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="unsupported-dgsem-scalar",
    )
    bad_problem = ConservationProblemIR("scalar", "state", scalar, None)
    with pytest.raises(TypeError, match="EulerSystem"):
        compile_conservation_problem(bad_problem, discretization, method)
    bounded_problem = ConservationProblemIR(
        "bounded", "state", system, FiniteVolumeBoundarySet(("x", "y"), (None, None))
    )
    with pytest.raises(TypeError, match="FiniteElementBoundarySet"):
        compile_conservation_problem(bounded_problem, discretization, method)

    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    triangle_mesh = CellMesh(
        vertices,
        (CellBlock("cells", "triangle", np.asarray(((0, 1, 2),))),),
    )
    triangle_field = FiniteElementFieldSpec(
        "state",
        discontinuous_element("triangle", 1),
        component_shape=(system.component_count,),
    )
    triangle = FiniteElementPlan(triangle_mesh, triangle_field).prepare()
    problem = ConservationProblemIR("triangle", "state", system, None)
    with pytest.raises(ValueError, match="quadrilateral and hexahedron"):
        compile_conservation_problem(problem, triangle, method)

    sbp = TensorGLLSBPPlan(2).prepare()
    xi, eta = np.meshgrid(np.asarray(sbp.nodes), np.asarray(sbp.nodes), indexing="ij")
    inverted = np.stack((1.0 - xi, eta), axis=-1)[None, ...]
    with pytest.raises(ValueError, match="nonpositive determinant"):
        MappedTensorMetricPlan(sbp, 2).prepare(inverted)

    pair = ideal_gas_euler_entropy_pair(system)
    with pytest.raises(ValueError, match="supplied together"):
        compile_conservation_problem(
            ConservationProblemIR("uncertified", "state", system, None),
            discretization,
            method,
            entropy_pair=pair,
        )


def test_prepared_dgsem_is_not_constructible_on_fd_sbp_discretization():
    with pytest.raises(TypeError, match="FiniteElementDiscretization"):
        PreparedDGSEMConservationDynamics(
            EulerSystem(2),
            object(),
            DGSEMConservationMethodPlan(
                EntropyConservativeEulerFluxPlan(), RusanovFluxPlan()
            ),
        )


def test_finite_element_boundary_set_requires_exact_exterior_ownership():
    _system, discretization = _quad_discretization()
    exterior = np.asarray(
        discretization.exterior_facet_domain.entity_indices, dtype=np.int32
    )
    local = np.asarray(
        discretization.exterior_facet_domain.owner_local_entities, dtype=np.int32
    )
    by_local = {int(local_id): int(facet) for facet, local_id in zip(exterior, local)}
    periodic = FiniteElementPeriodicFacetPair(by_local[0], by_local[2])
    boundaries = FiniteElementBoundarySet(
        discretization,
        {"walls": ((by_local[1], by_local[3]), SlipWallBoundary())},
        periodic_pairs=(periodic,),
    )
    assert boundaries.patch_names == ("walls",)
    assert boundaries.patch("walls").domain.entity_indices.shape == (2,)
    assert boundaries.periodic_pairs == (periodic,)

    with pytest.raises(ValueError, match="exhaustive"):
        FiniteElementBoundarySet(
            discretization,
            {"wall": ((int(exterior[0]),), SlipWallBoundary())},
        )
    with pytest.raises(ValueError, match="overlap"):
        FiniteElementBoundarySet(
            discretization,
            {
                "left": ((int(exterior[0]), int(exterior[1])), SlipWallBoundary()),
                "right": (
                    (int(exterior[1]), int(exterior[2]), int(exterior[3])),
                    SlipWallBoundary(),
                ),
            },
        )


def test_physical_slip_wall_disables_sampled_periodic_evidence():
    system, discretization = _quad_discretization()
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    boundaries = FiniteElementBoundarySet(
        discretization,
        {"walls": (exterior, SlipWallBoundary())},
    )
    interface_flux = EntropyStableEulerFluxPlan()
    entropy_pair, volume_flux, compatibility = _sampled_evidence(system, interface_flux)
    method = DGSEMConservationMethodPlan(
        volume_flux,
        interface_flux,
        compatibility=compatibility,
    )
    problem = ConservationProblemIR("wall-euler", "state", system, boundaries)
    compiled = compile_conservation_problem(
        problem,
        discretization,
        method,
        entropy_pair=entropy_pair,
    )
    primitive = jnp.asarray((1.0, 0.0, 0.0, 1.0))
    state = jnp.broadcast_to(
        system.primitive_to_conserved(primitive),
        discretization.field_spaces[0].vector_space.shape,
    )
    rate, diagnostics = compiled.residual_with_diagnostics(0.0, state)
    faces = compiled.dynamics.face_fluxes(0.0, state)
    np.testing.assert_allclose(rate, 0.0, atol=3.0e-6)
    np.testing.assert_allclose(diagnostics.conservation_balance_defect, 0.0, atol=3.0e-6)
    np.testing.assert_allclose(diagnostics.boundary_flux_rate[0], 0.0, atol=3.0e-6)
    assert np.all(np.asarray(faces.is_boundary))
    assert np.all(np.asarray(faces.neighbour_cells) == -1)
    assert not diagnostics.sampled_entropy_inequality
    assert compiled.dynamics.stable_step_evidence(state).positive


def test_entropy_filter_preserves_weighted_mean_and_repairs_pressure():
    compiled, system, discretization = _compiled()
    filter_ = EntropyFilterPlan(
        density_floor=1.0e-6,
        pressure_floor=1.0e-6,
    ).prepare(compiled.dynamics)
    state = _constant_state(system, discretization)
    state = state.at[0, 0].set(1.0e-4)
    state = state.at[0, -1].set(1.0e-5)
    before = jnp.sum(compiled.dynamics.scalar_mass_weights[:, None] * state, axis=0)
    filtered, evidence = eqx.filter_jit(filter_.filter)(jnp.asarray(0.0), state, None)
    after = jnp.sum(compiled.dynamics.scalar_mass_weights[:, None] * filtered, axis=0)
    np.testing.assert_allclose(after, before, atol=3.0e-9)
    assert evidence.successful
    assert evidence.applied
    assert evidence.minimum_density >= filter_.density_floor
    assert evidence.minimum_pressure >= filter_.pressure_floor
    method = SSPRK33FixedStepMethod(
        lambda time, value, args: jnp.zeros_like(value),
        stage_transform=filter_,
    )
    step_result = method.step(
        jnp.asarray(0),
        jnp.asarray(0.0),
        state,
        jnp.asarray(0.01),
        None,
    )
    assert step_result.successful
    assert step_result.transform_applied
    assert jnp.min(step_result.accepted_state[:, 0]) >= filter_.density_floor
    assert jnp.min(system.pressure(step_result.accepted_state)) >= filter_.pressure_floor


def test_entropy_filter_is_identity_on_smooth_state_and_rejects_invalid_mean():
    compiled, system, discretization = _compiled()
    filter_ = EntropyFilterPlan().prepare(compiled.dynamics)
    state = _constant_state(system, discretization)
    filtered, evidence = filter_.filter(0.0, state)
    np.testing.assert_allclose(filtered, state, atol=3.0e-10)
    assert evidence.successful
    assert not evidence.applied

    invalid = state.at[:, 0].set(-1.0)
    rejected, rejected_evidence = filter_.filter(0.0, invalid)
    assert not rejected_evidence.successful
    assert jnp.array_equal(rejected, invalid)
    method = SSPRK33FixedStepMethod(
        lambda time, value, args: jnp.zeros_like(value),
        stage_transform=filter_,
    )
    result = method.step(
        jnp.asarray(0),
        jnp.asarray(0.0),
        invalid,
        jnp.asarray(0.01),
        None,
    )
    assert not result.successful
    assert jnp.array_equal(result.accepted_state, invalid)


def test_compressible_navier_stokes_constitutive_gradient_and_flux():
    system = CompressibleNavierStokesSystem(ConstantTransport(0.2, 0.3), 2)

    def conserved(coordinates):
        density = jnp.asarray(1.0)
        velocity = jnp.asarray((coordinates[0], 2.0 * coordinates[1]))
        temperature = 1.0 + 3.0 * coordinates[0]
        pressure = density * system.material.gas_constant * temperature
        primitive = jnp.concatenate((density[None], velocity, pressure[None]))
        return system.primitive_to_conserved(primitive)

    coordinates = jnp.asarray((0.0, 0.0))
    state = conserved(coordinates)
    gradient = jax.jacfwd(conserved)(coordinates)
    velocity_gradient, temperature_gradient = system.primitive_gradients(state, gradient)
    np.testing.assert_allclose(
        velocity_gradient, jnp.diag(jnp.asarray((1.0, 2.0))), atol=2.0e-12
    )
    np.testing.assert_allclose(
        temperature_gradient, jnp.asarray((3.0, 0.0)), atol=2.0e-12
    )
    flux = system.viscous_flux(state, gradient)
    assert flux.shape == (system.component_count, system.dimension)
    np.testing.assert_allclose(flux[0], 0.0, atol=2.0e-12)
    diffusion = entropy_diffusion_evidence(system, state, gradient)
    assert diffusion.nonnegative


def test_tensor_ldg_constant_state_is_zero_and_has_positive_stability_step():
    _euler, discretization = _quad_discretization()
    system = CompressibleNavierStokesSystem(ConstantTransport(0.2, 0.1), 2)
    method = DGSEMConservationMethodPlan(
        EntropyConservativeEulerFluxPlan(),
        RusanovFluxPlan(),
        viscous=ViscousDGPlan(beta=0.0, penalty=1.0),
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR("periodic-navier-stokes", "state", system, None),
        discretization,
        method,
    )
    primitive = jnp.asarray((1.0, 0.0, 0.0, 1.0))
    state = jnp.broadcast_to(
        system.primitive_to_conserved(primitive),
        discretization.field_spaces[0].vector_space.shape,
    )
    gradient = compiled.dynamics.viscous_operator.corrected_gradient(0.0, state)
    rate = compiled(0.0, state)
    np.testing.assert_allclose(gradient, 0.0, atol=3.0e-11)
    np.testing.assert_allclose(rate, 0.0, atol=3.0e-10)
    evidence = compiled.dynamics.stable_step_evidence(state)
    assert evidence.positive
    assert evidence.maximum_diffusive_rate > 0.0


def test_tensor_ldg_stationary_no_slip_wall_preserves_rest_state():
    _euler, discretization = _quad_discretization()
    system = CompressibleNavierStokesSystem(ConstantTransport(0.2, 0.1), 2)
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
    method = DGSEMConservationMethodPlan(
        EntropyConservativeEulerFluxPlan(),
        RusanovFluxPlan(),
        viscous=ViscousDGPlan(
            boundary_closures=(
                phx.equations.fem.ViscousBoundaryClosure(
                    boundaries.patches[0].boundary.boundary_id
                ),
            )
        ),
    )
    compiled = compile_conservation_problem(
        ConservationProblemIR("wall-navier-stokes", "state", system, boundaries),
        discretization,
        method,
    )
    state = jnp.broadcast_to(
        system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 0.0, 1.0))),
        discretization.field_spaces[0].vector_space.shape,
    )
    np.testing.assert_allclose(compiled(0.0, state), 0.0, atol=3.0e-10)
