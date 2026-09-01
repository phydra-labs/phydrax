#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import phydrax as phx


def test_shared_ring_sheet_and_midpoint_wake_preserve_circulation():
    topology = phx.discretization.VortexRingSheetTopology(
        4,
        (0, 1, 2, 3),
        (1, 2, 3, 0),
        ((0, 1, 2, 3),),
        ((1, 1, 1, 1),),
    )
    state = phx.discretization.VortexRingSheetState(
        topology,
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))),
        jnp.asarray((1.0,)),
        jnp.full((4,), 0.05),
    )
    target = phx.discretization.VortexTargetState(jnp.asarray(((0.5, 0.5, 1.0),)))
    field = phx.operators.PreparedRingSheetField3D(state).evaluate(
        target,
        request=phx.discretization.VortexFieldRequest(
            velocity=True,
            velocity_gradient=True,
            vorticity=True,
        ),
    )
    step = phx.solver.VortexWakeIntegratorPlan(
        "midpoint",
        core_diffusivity=0.01,
    ).step(
        state,
        lambda targets, time, args: jnp.broadcast_to(
            jnp.asarray((0.1, 0.0, 0.0)),
            targets.positions.shape,
        ),
        0.0,
        0.1,
    )

    assert field.velocity_gradient.shape == (1, 3, 3)
    np.testing.assert_allclose(step.evidence.circulation_residual, 0.0)
    assert jnp.all(step.accepted.edge_core_radius > state.edge_core_radius)
    assert bool(step.successful)


def _component(name, y_offset):
    span = jnp.linspace(-1.0, 1.0, 4)
    leading = jnp.stack((jnp.zeros_like(span), span, jnp.zeros_like(span)), axis=-1)
    surface = phx.discretization.LiftingSurfacePlan(
        leading,
        leading + jnp.asarray((1.0, 0.0, 0.0)),
    )
    frame = phx.discretization.LiftingFrame3D(
        jnp.eye(3),
        jnp.asarray((0.0, y_offset, 0.0)),
    )
    return phx.discretization.LiftingComponentPlan(name, surface, frame)


def test_multi_surface_lifting_has_explicit_kelvin_and_load_evidence():
    surface = phx.discretization.MultiLiftingSurfacePlan(
        (_component("left", -1.5), _component("right", 1.5))
    ).prepare()
    plan = phx.solver.CompleteLiftingSystemPlan(
        surface,
        "horseshoe-vlm",
        wake_length=30.0,
        core_radius=0.02,
    )
    alpha = jnp.deg2rad(4.0)
    result = plan.solve(jnp.asarray((jnp.cos(alpha), 0.0, jnp.sin(alpha))))

    assert result.circulation.shape == (surface.panel_count,)
    assert jnp.linalg.norm(result.constraints.normal_residual) < 1e-8
    assert jnp.linalg.norm(result.constraints.kelvin_residual) < 1e-8
    assert result.load.total_force[2] > 0.0
    assert bool(result.successful)


def test_multiaxis_polar_dynamic_stall_and_compressibility_are_explicit():
    angle = jnp.deg2rad(jnp.asarray((-10.0, 0.0, 10.0)))
    reynolds = jnp.asarray((1.0e5, 1.0e6))
    mach = jnp.asarray((0.0, 0.5))
    flap = jnp.asarray((0.0, 0.2))
    shape = (3, 2, 2, 2)
    lift = jnp.broadcast_to(2.0 * jnp.pi * angle[:, None, None, None], shape)
    drag = 0.01 + 0.02 * lift**2
    polar = phx.solver.MultiAxisAirfoilPolar(
        angle,
        reynolds,
        mach,
        flap,
        lift,
        drag,
        endpoint="clamp",
    )
    evaluation = polar.evaluate(0.0, 5.0e5, 0.2, 0.1)
    stall = phx.solver.DynamicStallPlan(0.1, 0.1, jnp.deg2rad(12.0), 0.2)
    stall_state = stall.initialize(jnp.asarray((0.0,)), jnp.asarray((0.0,)))
    stall_result = stall.step(
        stall_state,
        jnp.asarray((jnp.deg2rad(15.0),)),
        jnp.asarray((1.0,)),
        0.01,
    )
    corrected = phx.solver.CompressibilityCorrectionPlan().apply(
        evaluation.lift,
        jnp.asarray(0.5),
        jnp.asarray(0.3),
    )

    assert evaluation.finite
    assert stall_result.finite
    assert corrected[2]


def test_native_boundary_panel_adapter_and_wall_flux_close_slip():
    geometry = phx.geometry.Circle((0.0, 0.0), 1.0).compile()
    panelization = phx.operators.BoundaryPanelization2D(
        geometry.boundary_atlas,
        panels_per_chart=12,
        quadrature_order=4,
        geometry=geometry,
    )
    native = phx.operators.NativePanelGeometry2D.from_panelization(panelization)
    flux = phx.solver.BoundaryIntegralVorticityFluxPlan2D(native.straight).solve(
        jnp.broadcast_to(
            jnp.asarray((1.0, 0.0)),
            native.straight.control.shape,
        ),
        jnp.zeros_like(native.straight.control),
        0.01,
    )

    assert flux.vortex_sheet_strength.shape == (native.straight.length.size,)
    assert jnp.all(jnp.isfinite(flux.vortex_sheet_strength))
    assert flux.evidence.slip_norm < 1e-5


def test_wall_corrected_pse_reports_flux_ledger():
    source = phx.discretization.VortexSourceState(
        jnp.asarray(((0.0, 0.1), (0.3, 0.2))),
        jnp.asarray((1.0, -1.0)),
        volume=jnp.full((2,), 0.1),
    )
    evaluation, evidence = phx.discretization.WallCorrectedPSEPlan(
        0.3,
        policy="mirror",
    ).evaluate(
        source,
        0.01,
        jnp.asarray(((0.0, 0.0), (0.3, 0.0))),
        jnp.asarray(((0.0, 1.0), (0.0, 1.0))),
    )

    assert evaluation.rate.shape == source.strength.shape
    assert evidence.conservative_with_flux
    assert bool(evaluation.successful)


def test_random_vortex_antithetic_noise_has_zero_weak_mean():
    direct = phx.operators.GaussianDirectVortexPlan2D(
        maximum_sources=2,
    ).prepare(source_capacity=2, target_capacity=2)
    source = phx.discretization.VortexSourceState(
        jnp.asarray(((-0.5, 0.0), (0.5, 0.0))),
        jnp.asarray((1.0, -1.0)),
        core_radius=jnp.full((2,), 0.1),
        volume=jnp.ones((2,)),
    )
    plan = phx.applications.vortex_flow.RandomVortexSolverPlan(
        direct,
        0.01,
        4,
        antithetic=True,
    )
    result = plan.step(plan.initialize(source), jax.random.key(2), 0.01)

    assert result.evidence.weak_moment_residual < 1e-12
    assert bool(result.successful)


def test_assimilation_and_constrained_closure_enforce_contracts():
    observation = phx.applications.vortex_flow.VortexObservationSet(
        jnp.eye(2),
        jnp.asarray((1.0, -1.0)),
        jnp.ones((2,)),
        kind="vorticity",
    )
    assimilation = phx.applications.vortex_flow.VortexDataAssimilationPlan(
        (observation,),
        jnp.ones((2,)),
    ).assimilate(jnp.zeros((2,)))

    class ZeroModel(eqx.Module):
        def __call__(self, value):
            return jnp.zeros_like(value[:1])

    closure = phx.applications.vortex_flow.ConstrainedLearnedClosure(
        ZeroModel(),
        jnp.zeros((2,)),
        jnp.ones((2,)),
        closure_id="zero-closure",
    ).evaluate(jnp.zeros((2, 2)), jnp.asarray((1.0, -1.0)))

    assert assimilation.weighted_residual_norm < 2.0
    assert jnp.all(closure.circulation_residual == 0.0)
    assert closure.dissipation <= 0.0


def test_sharding_preflight_is_real_on_available_devices():
    mesh = Mesh(np.asarray(jax.devices()), ("vortex",))
    policy = phx.operators.VortexShardingPolicy(
        mesh,
        strategy="target-sharded",
    )
    source = phx.discretization.VortexSourceState(
        jnp.zeros((2, 2)),
        jnp.asarray((1.0, -1.0)),
        core_radius=jnp.ones((2,)),
    )
    target = phx.discretization.VortexTargetState(jnp.zeros((3, 2)))
    evidence = policy.preflight(source, target, payload_components=2)

    assert int(evidence.device_count) == len(jax.devices())
    assert bool(evidence.supported)


def test_native_three_dimensional_panel_field_is_finite_off_surface():
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile()
    panelization = phx.operators.SurfacePanelization3D(
        geometry.boundary_atlas,
        quadrature_order=2,
        geometry=geometry,
    )
    native = phx.operators.NativePanelGeometry3D.from_panelization(panelization)
    result = phx.operators.NativePanelFieldPlan3D(native).evaluate(
        jnp.asarray(((2.0, 0.0, 0.0),)),
        jnp.ones((native.panel_count,)),
        kind="source",
        target_side="exterior",
    )

    assert result.velocity.shape == (1, 3)
    assert bool(result.successful)


def test_native_rigid_vortex_coupling_uses_prepared_body_dynamics():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((0,)),
        jnp.asarray((1.0,)),
        ambient_dimension=2,
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.asarray((0,)),
        jnp.asarray((1.0,)),
    ).prepare(particles)
    kinematics = phx.discretization.RigidBodyKinematics(
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
    )
    load = phx.discretization.RigidBodyLoad(
        jnp.zeros((1, 2)),
        jnp.zeros((1, 1)),
    )

    def coupler(time, fluid, body_state, args):
        del time, body_state, args
        return fluid, load, jnp.asarray(0.0)

    result = phx.applications.vortex_flow.VortexRigidCouplingPlan(
        bodies,
        "loose",
    ).step(
        jnp.asarray((0.0,)),
        kinematics,
        load,
        0.0,
        0.01,
        coupler,
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.kinematics.position, 0.0)


def test_mac_immersed_vortex_hybrid_preserves_zero_total_strength():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(6, periodic=True) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(operators).prepare()
    momentum = phx.discretization.MACMomentumPlan(
        operators,
        boundaries=boundaries,
    ).prepare()
    dynamics = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        momentum,
        phx.solver.MACPressureProjectionPlan(
            operators,
            solve_method="transform",
        ),
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((0, 1)),
        jnp.ones((2,)),
        ambient_dimension=2,
    ).prepare()
    transfer = phx.solver.MACVortexParticleTransferPlan(
        particles,
        dynamics,
        degree=1,
    )
    marker_position = jnp.asarray(
        ((0.35, 0.35), (0.65, 0.35), (0.65, 0.65), (0.35, 0.65))
    )
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(4),
        marker_position,
        jnp.full((4,), 0.25),
    ).prepare()
    marker_transfer = phx.discretization.MACMarkerTransferPlan(
        operators,
        markers,
    ).prepare()
    projection = phx.solver.MACImmersedBoundaryProjectionPlan(
        operators,
        marker_transfer,
        boundaries=boundaries,
        tolerance=1.0e-6,
        maximum_iterations=200,
    )
    source = phx.discretization.VortexSourceState(
        jnp.asarray(((0.25, 0.5), (0.75, 0.5))),
        jnp.asarray((0.0, 0.0)),
        core_radius=jnp.full((2,), 0.1),
        volume=jnp.full((2,), 0.5),
    )
    kinematics = markers.kinematics(
        marker_position,
        jnp.zeros_like(marker_position),
    )
    result = phx.solver.VortexImmersedHybridPlan(
        transfer,
        projection,
    ).step(source, kinematics, 0.01)

    assert jnp.linalg.norm(result.projection.divergence_after) < 1e-5
    assert jnp.all(jnp.isfinite(result.source.strength))
    assert bool(result.successful), (
        result.projection.successful,
        result.projection.status,
        result.grid.evidence.transfer_successful,
        result.grid.evidence.finite,
        result.work_residual,
    )
