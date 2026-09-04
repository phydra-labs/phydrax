#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import io

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.incompressible_flow._boundary_turbulence import (
    VectorEquilibriumWallStressPlan,
)
from phydrax.applications.incompressible_flow._immersed_les import (
    compile_fixed_immersed_mac_les_flow,
    FixedImmersedMACLESPlan,
    ImmersedMACLESStageResult,
    PreparedFixedImmersedMACLES,
)
from phydrax.equations._les_closures import (
    LESParameterProvenance,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)
from phydrax.equations._mac_les import MACAlgebraicLESPlan


def _route(*, count=4, coefficient=0.17, fraction=None, wall=False):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [2.0 * jnp.pi] * 3]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=boundaries
    ).prepare()
    pressure = phx.solver.MACPressureProjectionPlan(
        operators, boundaries=boundaries, solve_method="transform"
    )
    position = jnp.asarray([[1.7, 2.1, 2.4]])
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.asarray([7]), position, jnp.asarray([1.0])
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    immersed = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators,
        transfer,
        boundaries=boundaries,
        tolerance=2.0e-7,
        maximum_iterations=300,
    )
    resolved_filter = ResolvedLESFilter(
        "fixed-immersed-mac-cell-fluid-volume",
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="volume-equivalent",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id,
        "incompressible-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    les = MACAlgebraicLESPlan(SmagorinskyLESPlan(coefficient).prepare(provenance))
    if fraction is None:
        fraction = jnp.ones(discretization.cell_shape)
    kinematics = markers.kinematics(position, jnp.zeros_like(position))
    wall_model = VectorEquilibriumWallStressPlan().prepare(3) if wall else None
    plan = FixedImmersedMACLESPlan(
        les,
        immersed,
        kinematics,
        fraction,
        geometry_id="fixed-one-marker-body",
        wall_stress=wall_model,
        marker_wall_normal=jnp.asarray([[1.0, 0.0, 0.0]]) if wall else None,
        marker_sample_distance=jnp.asarray([0.2]) if wall else None,
    )
    problem = phx.equations.IncompressibleFlowProblem(3, 0.01)
    dynamics = compile_fixed_immersed_mac_les_flow(problem, momentum, pressure, plan)
    return discretization, operators, momentum, pressure, plan, dynamics


def _taylor_green(discretization):
    x_faces, y_faces, z_faces = discretization.face_centers
    return (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]) * jnp.cos(x_faces[..., 2]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]) * jnp.cos(y_faces[..., 2]),
        jnp.zeros(z_faces.shape[:-1], dtype=z_faces.dtype),
    )


def test_fixed_immersed_les_binds_owner_ids_and_zero_coefficient_parity():
    discretization, operators, momentum, pressure, plan, immersed_dynamics = _route(
        coefficient=0.0
    )
    dns = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.01), momentum, pressure
    )
    velocity = _taylor_green(discretization)
    immersed_state = immersed_dynamics.pack_velocity(velocity)
    dns_state = dns.pack_velocity(velocity)

    immersed_components = immersed_dynamics.rate_components(0.0, immersed_state)
    dns_components = dns.rate_components(0.0, dns_state)
    prepared = immersed_dynamics.algebraic_les

    assert isinstance(prepared, PreparedFixedImmersedMACLES)
    assert prepared.geometry_id == plan.geometry_id
    assert prepared.marker_id == plan.projection.transfer.markers.prepared_id
    assert (
        prepared.filter_id
        == plan.algebraic_les.prepared_model.provenance.resolved_filter.filter_id
    )
    assert prepared.model_id == plan.algebraic_les.prepared_model.prepared_id
    assert prepared.boundary_id == momentum.boundaries.prepared_id
    assert prepared.solver_id == plan.projection.plan_id
    regime = plan.admission_regime()
    assert regime.owner_plan_id == plan.projection.plan_id
    assert regime.marker_set_id == prepared.marker_id
    assert regime.geometry_id == prepared.geometry_id
    assert isinstance(immersed_components.les_stage, ImmersedMACLESStageResult)
    assert jnp.all(immersed_components.les_stage.model_result.kinematic_viscosity == 0.0)
    for immersed_rate, dns_rate in zip(
        immersed_components.unconstrained, dns_components.unconstrained, strict=True
    ):
        np.testing.assert_allclose(immersed_rate, dns_rate, rtol=0.0, atol=0.0)
    assert operators.prepared_id == plan.projection.operators.prepared_id


def test_fixed_immersed_les_uses_fluid_volume_filter_and_zero_solid_stress():
    fraction = jnp.ones((4, 4, 4)).at[0, 0, 0].set(0.0).at[1, 1, 1].set(0.125)
    discretization, _, _, _, _, dynamics = _route(fraction=fraction)
    velocity = _taylor_green(discretization)
    state = dynamics.pack_velocity(velocity)

    stage = dynamics.rate_components(0.0, state).les_stage
    base_width = dynamics.algebraic_les.base.filter_scale().directional_widths

    assert stage.successful
    np.testing.assert_allclose(
        stage.filter_scale.directional_widths[1, 1, 1],
        0.5 * base_width[1, 1, 1],
    )
    np.testing.assert_allclose(
        stage.filter_scale.directional_widths[0, 0, 0], base_width[0, 0, 0]
    )
    assert stage.model_result.kinematic_viscosity[0, 0, 0] == 0.0
    assert jnp.all(stage.model_result.specific_deviatoric_stress[0, 0, 0] == 0.0)
    assert stage.model_result.kinematic_viscosity[1, 1, 1] >= 0.0


def test_fixed_immersed_les_vector_wall_traction_is_tangent_dissipative_and_applied():
    discretization, _, _, _, plan, dynamics = _route(wall=True)
    state = dynamics.pack_velocity(_taylor_green(discretization))

    components = dynamics.rate_components(0.0, state)
    stage = components.les_stage

    assert stage.wall_stress is not None
    assert plan.admission_regime().marker_constraint_count == 1
    assert jnp.all(stage.wall_stress.successful)
    np.testing.assert_allclose(stage.wall_traction_density[:, 0], 0.0, atol=2.0e-12)
    assert stage.modeled_wall_power <= 0.0
    assert max(float(jnp.max(jnp.abs(value))) for value in stage.wall_rate) > 0.0
    for total, sgs, wall in zip(
        components.sgs, stage.sgs_rate, stage.wall_rate, strict=True
    ):
        np.testing.assert_allclose(total, sgs + wall, rtol=2.0e-12, atol=2.0e-12)


def test_vector_wall_stress_changes_normal_constrained_step_trajectory():
    discretization, _, _, _, baseline_plan, baseline = _route(wall=False)
    _, _, _, _, wall_plan, wall_dynamics = _route(wall=True)
    state = baseline.pack_velocity(
        tuple(0.05 * value for value in _taylor_green(discretization))
    )
    baseline_method = phx.solver.MACImmersedBoundaryIMEXEulerMethod(
        baseline,
        baseline_plan.projection,
        baseline_plan.marker_motion,
        motion_id=baseline_plan.marker_motion.motion_id,
        fixed_step_size=1.0e-4,
        marker_constraint_normals=wall_plan.marker_wall_normal,
    )
    wall_method = wall_plan.imex_euler_method(wall_dynamics, fixed_step_size=1.0e-4)

    baseline_step = baseline_method.step(0.0, state)
    wall_step = wall_method.step(0.0, state)

    assert baseline_step.accepted
    assert wall_step.accepted
    assert baseline_step.projection.constraint_mode == "normal"
    assert wall_step.projection.constraint_mode == "normal"
    assert baseline_step.projection.differentiation_certified
    assert wall_step.projection.differentiation_certified
    assert jnp.linalg.norm(baseline_step.projection.marker_slip) < 2.0e-7
    assert jnp.linalg.norm(wall_step.projection.marker_slip) < 2.0e-7
    assert jnp.linalg.norm(wall_step.state - baseline_step.state) > 1.0e-12


def test_immersed_methods_apply_sgs_project_and_restart_sbdf2_history():
    discretization, operators, _, _, plan, dynamics = _route(coefficient=0.12)
    velocity = tuple(0.03 * value for value in _taylor_green(discretization))
    state = dynamics.project_state(velocity)
    initial_components = dynamics.rate_components(0.0, state)
    method = plan.imex_euler_method(dynamics, fixed_step_size=1.0e-3)
    result = method.step(0.0, state)

    assert result.accepted
    assert jnp.linalg.norm(result.projection.divergence_after) < 2.0e-7
    assert jnp.linalg.norm(result.projection.marker_slip) < 2.0e-7
    assert max(float(jnp.max(jnp.abs(value))) for value in initial_components.sgs) > 0.0

    sbdf = plan.sbdf2_method(dynamics, 1.0e-3)
    startup = sbdf.initialize(0.0, state)
    startup_ledger = dynamics.algebraic_les.balance_ledger(dynamics, startup)
    payload = io.BytesIO()
    eqx.tree_serialise_leaves(payload, startup.history)
    payload.seek(0)
    restored = eqx.tree_deserialise_leaves(payload, startup.history)
    advanced = sbdf.step(restored)
    advanced_ledger = dynamics.algebraic_les.balance_ledger(
        dynamics, advanced, history=restored
    )
    current_stage = dynamics.rate_components(restored.time, restored.state).les_stage
    previous_stage = dynamics.rate_components(
        restored.time - advanced.step_size, restored.previous_state
    ).les_stage
    extrapolated_sgs = tuple(
        2.0 * current - previous
        for current, previous in zip(
            current_stage.sgs_rate, previous_stage.sgs_rate, strict=True
        )
    )
    expected_sgs_work = advanced.step_size * jnp.real(
        operators.velocity_space.inner(advanced.velocity, extrapolated_sgs)
    )
    with pytest.raises(ValueError, match="requires its input history"):
        dynamics.algebraic_les.balance_ledger(dynamics, advanced)

    assert startup.accepted
    assert advanced.accepted
    assert result.projection.constraint_mode == "full-vector"
    assert advanced.history.accepted_steps == 2
    assert advanced.history.method_id == sbdf.method_id
    assert startup_ledger.successful
    assert advanced_ledger.successful
    assert max(float(jnp.max(jnp.abs(value))) for value in extrapolated_sgs) > 0.0
    assert jnp.abs(expected_sgs_work) > 1.0e-14
    np.testing.assert_allclose(
        advanced_ledger.sgs_bulk_work,
        expected_sgs_work,
        rtol=2.0e-11,
        atol=2.0e-14,
    )
    assert jnp.linalg.norm(advanced.projection.divergence_after) < 2.0e-7
    assert jnp.linalg.norm(advanced.projection.marker_slip) < 2.0e-7


def test_immersed_les_step_ledger_closes_impulse_and_transfer_work():
    discretization, _, _, _, plan, dynamics = _route(coefficient=0.08)
    velocity = tuple(0.05 * value for value in _taylor_green(discretization))
    state = dynamics.pack_velocity(velocity)
    components = dynamics.rate_components(0.0, state)
    method = plan.imex_euler_method(dynamics, fixed_step_size=1.0e-4)
    step = method.step(0.0, state)
    ledger = dynamics.algebraic_les.balance_ledger(dynamics, step)
    assert max(float(jnp.max(jnp.abs(value))) for value in components.sgs) > 0.0
    for actual, advective, sgs, forcing in zip(
        step.explicit_rate,
        components.convection,
        components.sgs,
        components.forcing,
        strict=True,
    ):
        np.testing.assert_allclose(
            actual, -advective + sgs + forcing, rtol=2.0e-12, atol=2.0e-12
        )

    assert step.accepted
    assert ledger.successful
    np.testing.assert_allclose(
        ledger.fluid_impulse + ledger.body_impulse,
        ledger.impulse_balance_residual,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        ledger.impulse_balance_residual, 0.0, rtol=2.0e-7, atol=2.0e-10
    )
    np.testing.assert_allclose(
        ledger.fluid_stress_work,
        ledger.marker_stress_work,
        rtol=2.0e-7,
        atol=2.0e-10,
    )
    assert jnp.isfinite(ledger.sgs_bulk_work)
    assert ledger.body_mechanical_work == 0.0


def test_fixed_immersed_les_stage_is_jittable_and_has_velocity_jvp():
    discretization, operators, momentum, _, _, dynamics = _route(coefficient=0.1)
    state = dynamics.pack_velocity(_taylor_green(discretization))
    stage_data = momentum.boundaries.homogeneous_stage()
    prepared = dynamics.algebraic_les

    def rate(coordinates):
        velocity = operators.velocity_space.unflatten(coordinates)
        stage = prepared.evaluate(tuple(velocity), stage_data)
        return operators.velocity_space.flatten(stage.physical_rate)

    compiled = jax.jit(rate)(state)
    primal, tangent = jax.jvp(rate, (state,), (jnp.ones_like(state),))

    np.testing.assert_allclose(compiled, primal, rtol=2.0e-11, atol=2.0e-11)
    assert compiled.shape == state.shape
    assert tangent.shape == state.shape
    assert jnp.all(jnp.isfinite(compiled))
    assert jnp.all(jnp.isfinite(tangent))


@pytest.mark.parametrize("motion", ("moving", "deforming"))
def test_fixed_immersed_les_prepare_refuses_nonfixed_geometry(motion):
    _, _, momentum, _, plan, _ = _route()
    refused = FixedImmersedMACLESPlan(
        plan.algebraic_les,
        plan.projection,
        plan.marker_motion.kinematics,
        plan.cell_fluid_fraction,
        geometry_id=plan.geometry_id,
        motion=motion,
    )
    with pytest.raises(ValueError, match="stationary fixed geometry"):
        refused.prepare(momentum, molecular_viscosity=0.01)


def test_fixed_immersed_les_prepare_refuses_distributed_and_wrong_wall_route():
    _, _, momentum, _, plan, _ = _route()
    distributed = FixedImmersedMACLESPlan(
        plan.algebraic_les,
        plan.projection,
        plan.marker_motion.kinematics,
        plan.cell_fluid_fraction,
        geometry_id=plan.geometry_id,
        distributed=True,
    )
    with pytest.raises(ValueError, match="Distributed immersed MAC LES"):
        distributed.prepare(momentum, molecular_viscosity=0.01)

    wrong_wall = FixedImmersedMACLESPlan(
        plan.algebraic_les,
        plan.projection,
        plan.marker_motion.kinematics,
        plan.cell_fluid_fraction,
        geometry_id=plan.geometry_id,
        wall_stress=VectorEquilibriumWallStressPlan().prepare(2),
        marker_wall_normal=jnp.asarray([[1.0, 0.0, 0.0]]),
        marker_sample_distance=jnp.asarray([0.2]),
    )
    with pytest.raises(ValueError, match="prepared in 3D"):
        wrong_wall.prepare(momentum, molecular_viscosity=0.01)


def test_fixed_immersed_les_refuses_open_outer_boundary_at_prepare():
    count = 4
    specs = (
        phx.discretization.UniformCellAxisSpec(count, periodic=True),
        phx.discretization.UniformCellAxisSpec(count, periodic=True),
        phx.discretization.UniformCellAxisSpec(count),
    )
    grid = phx.discretization.TensorGridPlan(specs, axis_names=("x", "y", "z")).prepare(
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    )
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide("z", "lower", "pressure-outlet"),
            phx.discretization.MACBoundarySide("z", "upper", "pressure-outlet"),
        ),
    ).prepare()
    momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=boundaries
    ).prepare()
    position = jnp.asarray([[0.5, 0.5, 0.5]])
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.asarray([1]), position, jnp.asarray([1.0])
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    immersed = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators, transfer, boundaries=boundaries
    )
    resolved_filter = ResolvedLESFilter(
        "open-immersed-grid",
        family="implicit-grid-volume",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="wall-bounded",
        scale_rule="volume-equivalent",
        commutation_status="unmodeled",
        repeated_filter_semantics="unmodeled",
    )
    provenance = LESParameterProvenance(
        resolved_filter,
        discretization.prepared_id,
        "incompressible-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    les = MACAlgebraicLESPlan(SmagorinskyLESPlan(0.1).prepare(provenance))
    plan = FixedImmersedMACLESPlan(
        les,
        immersed,
        markers.kinematics(position, jnp.zeros_like(position)),
        jnp.ones(discretization.cell_shape),
        geometry_id="open-refusal",
    )

    with pytest.raises(ValueError, match="no-slip, open, inflow"):
        plan.prepare(momentum, molecular_viscosity=0.01)
