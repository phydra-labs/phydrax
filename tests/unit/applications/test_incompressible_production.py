import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications import incompressible_flow as flow


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
        compiled.projector,
        jnp.linspace(0.0, 4.0, 5),
        viscosity=0.01,
    )
    basis = flow.SolenoidalHermitianFourierBasis(
        compiled.projector,
        maximum_wavenumber=1.1,
    )
    initial = basis.evaluate(jnp.linspace(0.2, 0.8, basis.coordinate_size))
    return method, forcing, statistics, initial


def _periodic_plan(*, end_time=0.1, output_times=None):
    method, forcing, statistics, initial = _periodic_production_inputs()
    plan = flow.PeriodicSpectralProductionPlan(
        method,
        statistics,
        problem_id="periodic-production-case",
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
    method, _, statistics, initial_velocity = _periodic_production_inputs()
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
        method,
        statistics,
        problem_id="periodic-ou-production-case",
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
    method, _, statistics, initial_velocity = _periodic_production_inputs()
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
