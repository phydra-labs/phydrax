import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications import incompressible_flow as flow
from phydrax.applications.incompressible_flow._production import (
    PeriodicDynamicLESProductionState,
    PreparedPeriodicDynamicETDRKMethod,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import SupportDependency
from phydrax.solver._production_runtime import ArtifactCheckpointStore


def _periodic_production_inputs():
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(4),
            phx.discretization.FourierBasisPlan(4),
        ),
        axis_names=("x", "y"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
        )
    )
    compiled = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        space,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
    )
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        space, component_shape=(2,)
    )
    method = phx.solver.ETDRKMethod(2).prepare(
        compiled.semilinear_drift,
        coordinates=coordinates,
    )
    forcing = flow.ConstantPowerFourierForcingPlan(
        compiled.projector,
        maximum_wavenumber=1.1,
        power_input=0.1,
        minimum_forced_energy=1.0e-10,
    )
    statistics = flow.PeriodicModalTurbulenceStatisticsPlan(
        compiled,
        jnp.linspace(0.0, 4.0, 5),
    )
    basis = flow.SolenoidalHermitianFourierBasis(
        compiled.projector,
        maximum_wavenumber=1.1,
    )
    initial = basis.evaluate(jnp.linspace(0.2, 0.8, basis.coordinate_size))
    case = flow.PeriodicSpectralProductionCase(
        compiled,
        initial,
        case_id="periodic-production-case",
    )
    return compiled, method, forcing, statistics, case, initial


def _periodic_plan(*, end_time=0.1, output_times=None):
    dynamics, method, forcing, statistics, case, initial = _periodic_production_inputs()
    plan = flow.PeriodicSpectralProductionPlan(
        dynamics,
        method,
        statistics,
        case,
        start_time=0.0,
        end_time=end_time,
        step_size=0.05,
        checkpoint_interval=1,
        constant_power_forcing=forcing,
        constant_power_wiring="adapter",
        output_times=output_times,
    )
    return plan, method, forcing, initial


def _channel_production_inputs():
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(4),
            phx.discretization.ChebyshevBasisPlan(6),
            phx.discretization.FourierBasisPlan(4),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            phx.discretization.AxisDomain.interval(-1.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
        )
    )
    stokes = phx.discretization.ChannelStokesPlan(
        space,
        0.1,
        lower_wall_velocity=(-1.0, 0.0, 0.0),
        upper_wall_velocity=(1.0, 0.0, 0.0),
    )
    dynamics = phx.equations.compile_channel_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.1),
        stokes,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
    )
    method = phx.solver.ChannelSBDF2Method().prepare(dynamics, 0.01)
    velocity_coordinates = phx.discretization.HermitianSpectralCoordinates(
        space, component_shape=(3,)
    )
    pressure_coordinates = phx.discretization.HermitianSpectralCoordinates(space)
    statistics = flow.SpectralChannelStatisticsPlan(
        space,
        density=1.0,
        kinematic_viscosity=0.1,
    )
    return method, velocity_coordinates, pressure_coordinates, statistics


def _mac_production_inputs():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=False),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, -1.0, 0.0], [2.0 * jnp.pi, 1.0, 2.0 * jnp.pi]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    pressure_gradient = flow.MACConstantPressureGradientForcing(
        operators,
        jnp.asarray((-1.0, 0.0, 0.0)),
        density=1.0,
    )
    problem = phx.equations.IncompressibleFlowProblem(
        3,
        0.01,
        forcing=pressure_gradient,
        forcing_id=pressure_gradient.forcing_id,
    )
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    dynamics = phx.equations.compile_mac_incompressible_flow(
        problem,
        momentum,
        phx.solver.MACPressureProjectionPlan(
            operators,
            solve_method="hybrid",
            hybrid_line_axis=1,
        ),
    )
    method = phx.solver.SSPRK33FixedStepMethod(dynamics)
    statistics = flow.MACPlaneWallStatisticsPlan(
        operators,
        density=1.0,
        kinematic_viscosity=0.01,
        wall_normal_axis=1,
        streamwise_axis=0,
    )
    return discretization, operators, pressure_gradient, dynamics, method, statistics


def _periodic_les_production_inputs():
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(4) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi) for _ in range(3))
    )
    resolved_filter = phx.equations.ResolvedLESFilter(
        "retained Fourier grid",
        family="sharp-fourier-projection",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        space.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    les = phx.equations.PeriodicAlgebraicLESPlan(
        phx.equations.SmagorinskyLESPlan(0.12).prepare(provenance),
        phx.equations.PeriodicFourierGridFilterPlan(resolved_filter),
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.OversamplingDealiasingPlan(1.5)
        ),
    )
    dynamics = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.01),
        space,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
        algebraic_les=les,
    )
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        space,
        component_shape=(3,),
    )
    method = phx.solver.LESStabilityGuardedETDRKMethod(
        phx.solver.ETDRKMethod(2),
        safety_factor=0.9,
    ).prepare(dynamics, coordinates=coordinates)
    statistics = flow.PeriodicModalTurbulenceStatisticsPlan(
        dynamics,
        jnp.linspace(0.0, 4.0, 5),
    )
    basis = flow.SolenoidalHermitianFourierBasis(
        dynamics.projector,
        maximum_wavenumber=1.1,
    )
    initial = basis.evaluate(jnp.linspace(0.2, 0.8, basis.coordinate_size))
    case = flow.PeriodicSpectralProductionCase(
        dynamics,
        initial,
        case_id="periodic-les-production-case",
    )
    return dynamics, method, statistics, case, initial


def _periodic_dynamic_production_inputs():
    resolved = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(8) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi) for _ in range(3))
    )
    test = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(4) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi) for _ in range(3))
    )
    resolved_filter = phx.equations.ResolvedLESFilter(
        "dynamic retained Fourier grid",
        family="sharp-fourier-projection",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    test_filter = phx.equations.ResolvedLESFilter(
        "dynamic coarse Fourier grid",
        family="sharp-fourier-projection",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    provenance = phx.equations.LESParameterProvenance(
        resolved_filter,
        resolved.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    dynamic_model = phx.equations.DynamicSmagorinskyPlan(
        phx.equations.LagrangianDynamicLESAveraging(0.25),
        phx.equations.AdditiveDenominatorRegularization(1.0e-8),
        phx.equations.NonnegativeBackscatterClip(),
    ).prepare(
        phx.equations.DynamicLESProvenance(
            provenance,
            test_filter,
            (2.0, 2.0, 2.0),
        )
    )
    dynamic_plan = phx.equations.PeriodicDynamicLESPlan(
        dynamic_model,
        phx.equations.PeriodicFourierGridFilterPlan(resolved_filter),
        phx.equations.PeriodicFourierTestFilterPlan(test_filter),
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.OversamplingDealiasingPlan(1.5)
        ),
        energy_tolerance=2.0e-8,
    )
    dynamics = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.01),
        resolved,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
        dynamic_les=dynamic_plan,
        dynamic_test_discretization=test,
    )
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        resolved, component_shape=(3,)
    )
    method = phx.solver.ETDRKMethod(2).prepare(
        dynamics.semilinear_drift,
        coordinates=coordinates,
    )
    statistics = flow.PeriodicModalTurbulenceStatisticsPlan(
        dynamics,
        jnp.linspace(0.0, 7.0, 8),
    )
    basis = flow.SolenoidalHermitianFourierBasis(
        dynamics.projector,
        maximum_wavenumber=1.1,
    )
    initial = basis.evaluate(jnp.linspace(0.2, 0.8, basis.coordinate_size))
    case = flow.PeriodicSpectralProductionCase(
        dynamics,
        initial,
        case_id="periodic-dynamic-les-production-case",
    )
    return dynamics, method, statistics, case, initial


def test_periodic_dynamic_production_commits_only_accepted_state_and_restarts(
    tmp_path,
):
    dynamics, method, statistics, case, initial_velocity = (
        _periodic_dynamic_production_inputs()
    )
    plan = flow.PeriodicSpectralProductionPlan(
        dynamics,
        method,
        statistics,
        case,
        start_time=0.0,
        end_time=1.0e-6,
        step_size=1.0e-6,
        checkpoint_interval=1,
    )
    assert isinstance(plan.method, PreparedPeriodicDynamicETDRKMethod)
    assert plan.method.method_id != method.method_id
    assert plan.checkpoint_encoding.bindings == ()
    prepared = plan.prepare(tmp_path / "periodic-dynamic")
    initial = prepared.initialize(initial_velocity)
    assert isinstance(initial.accepted_state, PeriodicDynamicLESProductionState)

    initial_dynamic_stage = dynamics.dynamic_les_stage(
        initial.accepted_state.velocity,
        initial.accepted_state.continuation_state,
    )
    rejected_step = (
        2.0
        * dynamics.step_restriction(
            initial.accepted_state.velocity,
            dynamic_les_stage=initial_dynamic_stage,
        ).etdrk_selected
    )
    rejected = plan.method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        initial.accepted_state,
        rejected_step,
        None,
    )
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(
        rejected.accepted_state.velocity,
        initial.accepted_state.velocity,
    )
    np.testing.assert_array_equal(
        rejected.accepted_state.continuation_state.averaged_numerator,
        initial.accepted_state.continuation_state.averaged_numerator,
    )
    assert int(rejected.accepted_state.continuation_state.accepted_updates) == int(
        initial.accepted_state.continuation_state.accepted_updates
    )

    following, transition = prepared.step(initial)
    assert bool(transition.successful)
    assert int(following.accepted_state.continuation_state.accepted_updates) > int(
        initial.accepted_state.continuation_state.accepted_updates
    )
    checkpointed = prepared.checkpoint(following)
    resumed = prepared.resume(checkpointed)
    np.testing.assert_array_equal(
        resumed.accepted_state.velocity,
        following.accepted_state.velocity,
    )
    np.testing.assert_array_equal(
        resumed.accepted_state.continuation_state.averaged_numerator,
        following.accepted_state.continuation_state.averaged_numerator,
    )
    np.testing.assert_array_equal(
        resumed.accepted_state.continuation_state.averaged_denominator,
        following.accepted_state.continuation_state.averaged_denominator,
    )
    snapshot = prepared.statistics_snapshot(
        following.time,
        following.accepted_state,
    )
    assert snapshot.sgs_dynamic_provenance_id is not None
    assert snapshot.sgs_averaging_id is not None
    assert snapshot.sgs_backscatter_id is not None
    assert bool(snapshot.sgs_regularization_available)
    assert bool(snapshot.sgs_stability_available)
    assert jnp.isfinite(snapshot.sgs_dynamic_coefficient_mean)
    np.testing.assert_allclose(
        snapshot.sgs_transfer_shells.total,
        snapshot.sgs_energy_rate,
    )


def test_periodic_plan_identity_and_checkpoint_binding_change_with_runtime():
    first, base_method, forcing, _ = _periodic_plan(end_time=0.1)
    second, _, _, _ = _periodic_plan(end_time=0.15)

    assert first.plan_id != second.plan_id
    assert first.manifest.manifest_id != second.manifest.manifest_id
    assert first.runtime_plan.plan_id != second.runtime_plan.plan_id
    assert first.method.method_id != base_method.method_id
    assert first.constant_power_forcing.forcing_id == forcing.forcing_id
    assert tuple(
        binding.leaf_index for binding in first.checkpoint_encoding.bindings
    ) == (0,)
    assert first.checkpoint_encoding.bindings[0].coordinates.coordinate_id == (
        first.method.coordinates.coordinate_id
    )


def test_periodic_les_production_uses_guarded_first_stage_and_statistics(tmp_path):
    dynamics, method, statistics, case, initial_velocity = (
        _periodic_les_production_inputs()
    )
    step_size = jnp.asarray(1.0e-4)
    stage = dynamics.stage(jnp.asarray(0.0), initial_velocity)
    guarded_result = method.step(
        jnp.asarray(0),
        jnp.asarray(0.0),
        initial_velocity,
        step_size,
        None,
    )
    reused_result = method.base_method._step_with_first_nonlinear(
        jnp.asarray(0),
        jnp.asarray(0.0),
        initial_velocity,
        step_size,
        None,
        stage.rates.nonlinear_rate,
    )
    plan = flow.PeriodicSpectralProductionPlan(
        dynamics,
        method,
        statistics,
        case,
        start_time=0.0,
        end_time=1.0e-4,
        step_size=1.0e-4,
        checkpoint_interval=1,
    )
    prepared = plan.prepare(tmp_path / "periodic-les")
    initial = prepared.initialize(initial_velocity)
    following, transition = prepared.step(initial)
    snapshot = prepared.statistics_snapshot(
        following.time,
        following.accepted_state,
    )

    assert bool(guarded_result.successful)
    assert bool(transition.successful)
    np.testing.assert_allclose(
        guarded_result.accepted_state,
        reused_result.accepted_state,
        atol=np.finfo(float).eps,
    )
    assert bool(snapshot.sgs_available)
    assert bool(snapshot.sgs_stability_available)
    assert snapshot.compilation_id == dynamics.compilation_id
    assert snapshot.sgs_prepared_action_id == dynamics.algebraic_les.prepared_id
    np.testing.assert_allclose(
        snapshot.sgs_transfer_shells.total,
        snapshot.sgs_energy_rate,
    )


def test_periodic_constant_power_adapter_executes_through_generic_runtime(tmp_path):
    plan, base_method, forcing, initial_velocity = _periodic_plan(end_time=0.05)
    prepared = plan.prepare(tmp_path / "periodic")
    initial = prepared.initialize(initial_velocity)
    following, transition = prepared.step(initial)
    base = base_method.step(
        jnp.asarray(0),
        jnp.asarray(0.0),
        initial_velocity,
        jnp.asarray(0.05),
        None,
    )
    statistics = prepared.statistics_snapshot(following.time, following.accepted_state)

    assert bool(transition.successful)
    assert bool(forcing.evaluate(initial_velocity).successful)
    assert not np.allclose(following.accepted_state, base.accepted_state)
    assert bool(statistics.forcing_available)
    assert statistics.plan_id == plan.statistics.plan_id
    with pytest.raises(ValueError, match="bound initial condition"):
        prepared.initialize(2.0 * initial_velocity)


def test_periodic_artifact_checkpoint_restart_preserves_exact_case(tmp_path):
    plan, _, _, initial_velocity = _periodic_plan(end_time=0.05)
    profile = HPCFilesystemProfile(
        "periodic-production-posix",
        "test-filesystem",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    repository_policy = POSIXRepositoryPolicy(
        profile,
        maximum_chunk_bytes=256,
        maximum_metadata_bytes=1024 * 1024,
    )
    repository = POSIXArtifactRepository(
        tmp_path / "periodic-artifact",
        repository_policy,
    )
    checkpoint_policy = phx.solver.CheckpointGenerationPolicy(plan.checkpoint_retention)
    dependency = SupportDependency(
        "repository-profile",
        repository.support_tuple.support_tuple_id,
    )
    resolved = ResolvedRunSpec(
        (),
        (dependency,),
        release_index_id="release-index",
        profile_ids=(dependency.profile_id,),
        trust_policy_id="trust-policy",
        valid_at=10,
        valid_from=0,
        valid_until=20,
        prepared_configuration_id=plan.plan_id,
        precision_policy_id=plan.manifest.precision_id,
        resource_policy_id="resource-policy",
        checkpoint_policy_id=checkpoint_policy.policy_id,
        output_policy_id="output-policy",
        repository_id=repository.provider_id,
        scheduler_id="scheduler",
        auth_policy_id="auth-policy",
    )
    store = ArtifactCheckpointStore(
        repository,
        plan.manifest,
        checkpoint_policy,
        resolved,
        writer_id="periodic-production-worker",
        encoding_plan=plan.checkpoint_encoding,
    )
    prepared = plan.prepare(store)
    initial = prepared.initialize(initial_velocity)
    following, transition = prepared.step(initial)
    checkpointed = prepared.checkpoint(following)
    resumed = prepared.resume(checkpointed)

    assert bool(transition.successful)
    coordinates = plan.method.coordinates
    assert coordinates is not None
    np.testing.assert_array_equal(
        coordinates.to_real_coordinates(resumed.accepted_state),
        coordinates.to_real_coordinates(following.accepted_state),
    )
    assert resumed.last_checkpoint_id == checkpointed.last_checkpoint_id
    mismatched_plan, _, _, _ = _periodic_plan(end_time=0.1)
    with pytest.raises(ValueError, match="exactly bind"):
        mismatched_plan.prepare(store)


def test_rejected_periodic_attempt_does_not_update_statistics(tmp_path):
    dynamics, method, forcing, statistics, _, initial_velocity = (
        _periodic_production_inputs()
    )
    invalid_velocity = initial_velocity.at[1, 0, 0].add(0.25j)
    case = flow.PeriodicSpectralProductionCase(
        dynamics,
        invalid_velocity,
        case_id="periodic-rejected-attempt",
    )
    plan = flow.PeriodicSpectralProductionPlan(
        dynamics,
        method,
        statistics,
        case,
        start_time=0.0,
        end_time=0.05,
        step_size=0.05,
        checkpoint_interval=1,
        constant_power_forcing=forcing,
        constant_power_wiring="adapter",
    )
    prepared = plan.prepare(tmp_path / "periodic-rejected")
    initial = prepared.initialize(invalid_velocity)
    following, transition = prepared.step(initial)

    assert not bool(transition.successful)
    assert int(following.step_index) == 0
    np.testing.assert_array_equal(
        following.moment_states[0].weight,
        jnp.asarray(0.0),
    )


def test_channel_binds_complete_continuation_leaves_and_rejects_off_lattice():
    method, velocity_coordinates, pressure_coordinates, statistics = (
        _channel_production_inputs()
    )
    with pytest.raises(ValueError, match="step lattice"):
        flow.SpectralChannelProductionPlan(
            method,
            velocity_coordinates,
            pressure_coordinates,
            statistics,
            problem_id="channel-production-case",
            start_time=0.0,
            end_time=0.025,
            checkpoint_interval=1,
        )

    plan = flow.SpectralChannelProductionPlan(
        method,
        velocity_coordinates,
        pressure_coordinates,
        statistics,
        problem_id="channel-production-case",
        start_time=0.0,
        end_time=0.02,
        checkpoint_interval=1,
        output_times=jnp.asarray((0.01, 0.02)),
    )
    bindings = plan.checkpoint_encoding.bindings
    assert tuple(binding.leaf_index for binding in bindings) == (0, 1, 2, 3, 4)
    assert all(
        binding.coordinates.coordinate_id == velocity_coordinates.coordinate_id
        for binding in bindings[:4]
    )
    assert bindings[4].coordinates.coordinate_id == pressure_coordinates.coordinate_id
    assert plan.runtime_plan.method is method
    assert plan.runtime_plan.retry_policy.maximum_retries == 0


def test_mac_pressure_gradient_and_native_statistics_are_bound(tmp_path):
    discretization, operators, pressure_gradient, dynamics, method, statistics = (
        _mac_production_inputs()
    )
    plan = flow.StructuredMACProductionPlan(
        method,
        dynamics,
        statistics,
        start_time=0.0,
        end_time=0.01,
        step_size=0.01,
        checkpoint_interval=1,
        constant_pressure_gradient=pressure_gradient,
    )
    prepared = plan.prepare(tmp_path / "mac")
    velocity = tuple(
        jnp.ones(layout.shape) if axis == 0 else jnp.zeros(layout.shape)
        for axis, layout in enumerate(discretization.face_layouts)
    )
    initial = prepared.initialize(velocity)
    snapshot = prepared.statistics_snapshot(initial.time, initial.accepted_state)

    assert plan.constant_pressure_gradient.forcing_id == dynamics.problem.forcing_id
    assert plan.checkpoint_encoding.bindings == ()
    assert snapshot.operators_id == operators.prepared_id
    assert snapshot.face_to_cell_convention == "adjacent-face arithmetic average"
    assert snapshot.plane_weight_convention.startswith("exact cell-volume")
    assert bool(jnp.isfinite(snapshot.forcing_power))
    assert snapshot.forcing_power > 0.0
    assert snapshot.mean_velocity.shape == (4, 3)
    assert snapshot.raw_second_moment.shape == (4, 3, 3)


def test_mac_statistics_use_declared_moving_wall_velocity():
    discretization, operators, _, _, _, _ = _mac_production_inputs()
    statistics = flow.MACPlaneWallStatisticsPlan(
        operators,
        density=1.0,
        kinematic_viscosity=0.01,
        wall_normal_axis=1,
        streamwise_axis=0,
        upper_wall_velocity=jnp.asarray((1.0, 0.0, 0.0)),
    )
    x_faces, y_faces, z_faces = discretization.face_centers
    velocity = (
        0.5 * (x_faces[..., 1] + 1.0),
        jnp.zeros(y_faces.shape[:-1], dtype=y_faces.dtype),
        jnp.zeros(z_faces.shape[:-1], dtype=z_faces.dtype),
    )

    snapshot = statistics.evaluate(velocity)

    np.testing.assert_allclose(snapshot.lower_wall_shear[0], 0.005)
    np.testing.assert_allclose(snapshot.upper_wall_shear[0], 0.005)
    assert (
        statistics.plan_id
        != flow.MACPlaneWallStatisticsPlan(
            operators,
            density=1.0,
            kinematic_viscosity=0.01,
            wall_normal_axis=1,
            streamwise_axis=0,
        ).plan_id
    )


def test_ou_forced_periodic_production_couples_and_restarts(tmp_path):
    dynamics, method, _, statistics, case, initial_velocity = (
        _periodic_production_inputs()
    )
    basis = flow.SolenoidalHermitianFourierBasis(
        statistics.projector,
        maximum_wavenumber=1.1,
    )
    forcing = flow.SolenoidalOUForcingPlan(
        basis,
        correlation_time=0.7,
        rms_acceleration=0.2,
    )
    realization = phx.stochastic.OrnsteinUhlenbeckRealization(
        jax.random.key(29),
        (basis.coordinate_size,),
        support=(0.0, 1.0),
        tolerance=1.0e-6,
    )
    plan = flow.PeriodicSpectralProductionPlan(
        dynamics,
        method,
        statistics,
        case,
        start_time=0.0,
        end_time=0.1,
        step_size=0.05,
        checkpoint_interval=1,
        ou_forcing=forcing,
        ou_realization=realization,
    )
    prepared = plan.prepare(tmp_path / "periodic-ou")
    initial = prepared.initialize(initial_velocity)

    following, transition = prepared.step(initial)
    checkpointed = prepared.checkpoint(following)
    resumed = prepared.resume(checkpointed)
    snapshot = prepared.statistics_snapshot(following.time, following.accepted_state)

    assert isinstance(following.accepted_state, flow.OUForcedPeriodicState)
    assert isinstance(plan.method, flow.PreparedOUForcedETDRKMethod)
    assert bool(transition.successful)
    np.testing.assert_allclose(following.accepted_state.forcing_state.time, 0.05)
    np.testing.assert_allclose(
        resumed.accepted_state.velocity,
        following.accepted_state.velocity,
    )
    np.testing.assert_allclose(
        resumed.accepted_state.forcing_state.coefficients,
        following.accepted_state.forcing_state.coefficients,
    )
    assert tuple(binding.leaf_index for binding in plan.checkpoint_encoding.bindings) == (
        0,
    )
    assert bool(snapshot.forcing_available)


def test_ou_method_uses_continuation_time_with_fixed_step_scheduler_roundoff():
    _, method, _, statistics, _, initial_velocity = _periodic_production_inputs()
    basis = flow.SolenoidalHermitianFourierBasis(
        statistics.projector,
        maximum_wavenumber=1.1,
    )
    forcing = flow.SolenoidalOUForcingPlan(
        basis,
        correlation_time=0.7,
        rms_acceleration=0.2,
    )
    realization = phx.stochastic.OrnsteinUhlenbeckRealization(
        jax.random.key(31),
        (basis.coordinate_size,),
        support=(0.0, 200.0),
        tolerance=1.0e-6,
    )
    prepared = flow.prepare_ou_forced_periodic_method(method, forcing, realization)
    initial = prepared.initial_state(initial_velocity, 0.0)

    def body(step_index, carry):
        current, successful = carry
        result = prepared.step(
            step_index,
            step_index.astype(initial_velocity.real.dtype) * 0.1,
            current,
            jnp.asarray(0.1),
            None,
        )
        return result.accepted_state, successful & result.successful

    state, successful = jax.lax.fori_loop(0, 1024, body, (initial, jnp.asarray(True)))
    assert bool(successful)
    np.testing.assert_allclose(state.forcing_state.time, 102.4)
    mismatched = prepared.step(
        jnp.asarray(1024, dtype=jnp.int32),
        state.forcing_state.time + 1.0e-6,
        state,
        jnp.asarray(0.1),
        None,
    )
    assert not bool(mismatched.successful)
    np.testing.assert_array_equal(mismatched.accepted_state.velocity, state.velocity)
    np.testing.assert_array_equal(
        mismatched.accepted_state.forcing_state.coefficients,
        state.forcing_state.coefficients,
    )
