#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def _periodic_euler_runtime(shape):
    dimension = len(shape)
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for count in shape
        ),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    system = phx.equations.EulerSystem(dimension)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "differentiable-hydrodynamics-test",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    return grid, system, discretization, runtime


def _periodic_mhd_transport(count=3):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    system = phx.equations.IdealMHDSystem(3)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "constant-mhd",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x", "y", "z")),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLDFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    face_shape = (count, count, count)
    magnetic_flux = bridge.pack_face_flux(
        (
            jnp.full(face_shape, 0.2),
            jnp.zeros(face_shape),
            jnp.zeros(face_shape),
        )
    )
    primitive = jnp.zeros(face_shape + (8,))
    primitive = primitive.at[..., 0].set(1.0)
    primitive = primitive.at[..., 4].set(1.0)
    primitive = primitive.at[..., 5].set(0.2)
    full = system.primitive_to_conserved(primitive)
    spatial = phx.discretization.UpwindConstrainedTransportPlan(dynamics, bridge)
    integrator = phx.solver.ConstrainedMHDSSPRK3Plan(spatial, cfl=0.2)
    return grid, system, integrator, full, magnetic_flux


class _PreparedLinearSource(phx.solver.AbstractPreparedBalanceLawProcess):
    maximum_step: float = eqx.field(static=True)

    def __init__(self, maximum_step: float, /):
        self.maximum_step = float(maximum_step)
        self.process_id = "adaptive-linear-source"
        self.requires_realization = False
        self.realization_name = None
        self.differentiability = "smooth_discrete"
        self.modified_components = ("total_energy",)

    def initialize(self, source_view, args: Any = None, /):
        del source_view, args
        return phx.solver.BalanceLawProcessState(
            self.process_id,
            ("accepted_duration",),
            (jnp.asarray(0.0),),
        )

    def step_limit(self, time, cell_average, process_state, args: Any = None, /):
        del time, cell_average, process_state, args
        return jnp.asarray(self.maximum_step)

    def advance(
        self,
        start_time,
        end_time,
        cell_average,
        process_state,
        realization=None,
        args: Any = None,
        /,
    ):
        del realization
        step = end_time - start_time
        rate = jnp.asarray(args["rate"], dtype=cell_average.dtype)
        candidate = cell_average.at[..., -1].add(rate * step)
        elapsed = process_state.field("accepted_duration") + step
        next_state = phx.solver.BalanceLawProcessState(
            self.process_id,
            ("accepted_duration",),
            (elapsed,),
        )
        return phx.solver.BalanceLawProcessAdvance(
            cell_average=candidate,
            process_state=next_state,
            successful=jnp.asarray(True),
            source_change=candidate - cell_average,
            diagnostics=elapsed,
        )


class _PreparedMagneticMutation(phx.solver.AbstractPreparedBalanceLawProcess):
    def __init__(self, modified_components=("total_energy",)):
        self.process_id = "undeclared-magnetic-mutation"
        self.requires_realization = False
        self.realization_name = None
        self.differentiability = "invalid"
        self.modified_components = tuple(modified_components)

    def initialize(self, source_view, args: Any = None, /):
        del source_view, args
        return phx.solver.BalanceLawProcessState.empty(self.process_id)

    def step_limit(self, time, cell_average, process_state, args: Any = None, /):
        del time, cell_average, process_state, args
        return jnp.asarray(jnp.inf)

    def advance(
        self,
        start_time,
        end_time,
        cell_average,
        process_state,
        realization=None,
        args: Any = None,
        /,
    ):
        del start_time, end_time, realization, args
        candidate = cell_average.at[..., 5].add(1e-3)
        return phx.solver.BalanceLawProcessAdvance(
            cell_average=candidate,
            process_state=process_state,
            successful=jnp.asarray(True),
            source_change=candidate - cell_average,
            diagnostics=jnp.asarray(True),
        )


def test_adaptive_balance_law_records_rolls_back_replays_and_checkpoints(tmp_path):
    _, system, _, runtime = _periodic_euler_runtime((4,))
    primitive = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 1.0]), (4, 3))
    transport_state = runtime.initialize_state(
        system.primitive_to_conserved(primitive),
        0.0,
        2e-3,
    )
    transport = phx.solver.prepare_balance_law_transport(runtime)
    balance = phx.solver.PreparedBalanceLawRuntime(
        transport,
        (_PreparedLinearSource(1e-3),),
    )
    initial = balance.initialize_state(transport_state)
    policy = phx.solver.BalanceLawAdaptivePolicy(
        3,
        maximum_retries=2,
        safety_factor=1.0,
        growth_factor=1.0,
    )
    adaptive = phx.solver.AdaptiveBalanceLawRolloutPlan(
        balance,
        3e-3,
        policy,
    )

    realized = adaptive.rollout(initial, {"rate": jnp.asarray(0.2)})

    assert bool(realized.completed)
    assert int(realized.status) == int(phx.solver.BalanceLawAdaptiveStatus.SUCCESS)
    assert int(realized.journal.attempt_count) == 4
    assert int(realized.journal.accepted_count) == 3
    assert not bool(realized.journal.accepted[0])
    assert int(realized.journal.limiting_process_indices[0]) == 0
    np.testing.assert_allclose(
        realized.realized_mesh.accepted_times,
        jnp.asarray([1e-3, 2e-3, 3e-3]),
    )
    np.testing.assert_allclose(
        realized.final_state.process_states[0].field("accepted_duration"),
        3e-3,
    )

    replay_results = []
    replay_policies = (
        phx.solver.FiniteVolumeReplayPolicy("full"),
        phx.solver.FiniteVolumeReplayPolicy("step"),
        phx.solver.FiniteVolumeReplayPolicy("block", block_size=2),
    )
    for replay_policy in replay_policies:
        scheduled = phx.solver.ScheduledBalanceLawRolloutPlan.from_realized_mesh(
            balance,
            realized.realized_mesh,
            replay=replay_policy,
        )
        replayed = scheduled.rollout(initial, {"rate": jnp.asarray(0.2)})
        replay_results.append(replayed)
        np.testing.assert_allclose(
            replayed.final_state.transport_state.cell_average(),
            realized.final_state.transport_state.cell_average(),
        )
        assert bool(jnp.all(replayed.accepted))

    def loss(rate, replay_policy):
        scheduled = phx.solver.ScheduledBalanceLawRolloutPlan.from_realized_mesh(
            balance,
            realized.realized_mesh,
            replay=replay_policy,
        )
        replayed = scheduled.rollout(initial, {"rate": rate})
        return jnp.sum(replayed.final_state.transport_state.cell_average()[..., -1])

    def gradient(replay_policy):
        return jax.grad(lambda rate: loss(rate, replay_policy))(jnp.asarray(0.2))

    gradients = tuple(gradient(replay_policy) for replay_policy in replay_policies)
    np.testing.assert_allclose(gradients, gradients[0], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(gradients[0], 4 * 3e-3, rtol=1e-6, atol=1e-8)

    scheduled = phx.solver.ScheduledBalanceLawRolloutPlan.from_realized_mesh(
        balance,
        realized.realized_mesh,
    )
    checkpoint_plan = phx.solver.BalanceLawCheckpointPlan(
        balance,
        scheduled.temporal_mesh.mesh_id,
    )
    path = tmp_path / "adaptive-balance-law.phxckpt"
    written = phx.solver.write_balance_law_checkpoint(
        path,
        checkpoint_plan,
        realized.final_state,
    )
    restored = phx.solver.read_balance_law_checkpoint(path, checkpoint_plan)
    assert restored.payload_id == written.payload_id
    np.testing.assert_array_equal(
        restored.runtime_state.transport_state.content_state.conservative_content,
        realized.final_state.transport_state.content_state.conservative_content,
    )


def test_ou_realization_is_subdivision_consistent_and_antithetic():
    realization = phx.stochastic.OrnsteinUhlenbeckRealization(
        jr.key(19),
        (3,),
        support=(0.0, 1.0),
        tolerance=1e-6,
        noise_id="ou-semigroup",
    )
    correlation = jnp.asarray(0.35)
    first = realization.innovations(0.0, 0.4, correlation)
    second = realization.innovations(0.4, 1.0, correlation)
    full = realization.innovations(0.0, 1.0, correlation)
    composed = realization.decay(0.4, 1.0, correlation) * first + second

    np.testing.assert_allclose(full, composed, rtol=1e-6, atol=1e-7)
    variance = 1.0 - jnp.exp(-2.0 / correlation)
    assert jnp.all(jnp.isfinite(full))
    assert variance > 0.0

    antithetic = phx.stochastic.OrnsteinUhlenbeckRealization.antithetic(
        jr.key(23),
        (3,),
        support=(0.0, 1.0),
        num_pairs=1,
        tolerance=1e-6,
    )
    paired = antithetic.innovations(0.1, 0.7, correlation)
    np.testing.assert_allclose(paired[0] + paired[1], 0.0, atol=1e-8)
    composite = phx.stochastic.CompositeStochasticRealization({"forcing": antithetic})
    assert composite.component("forcing") is antithetic
    assert phx.stochastic.is_stochastic_realization(composite)
    assert composite.independence_labels[0] == composite.independence_labels[1]


def test_gravity_balance_runtime_preserves_kick_internal_energy_and_checkpoints(tmp_path):
    grid, system, _, runtime = _periodic_euler_runtime((16,))
    x = grid.structured_axes[0].interval_centers
    density = 1.0 + 0.05 * jnp.sin(2.0 * jnp.pi * x)
    primitive = jnp.stack(
        (density, jnp.zeros_like(density), jnp.ones_like(density)), axis=-1
    )
    state = runtime.initialize_state(system.primitive_to_conserved(primitive), 0.0, 1e-4)
    transport = phx.solver.prepare_balance_law_transport(runtime)
    gravity = phx.solver.NewtonianSelfGravityPlan(0.2).prepare(transport)
    balance = phx.solver.PreparedBalanceLawRuntime(transport, (gravity,))
    balance_state = balance.initialize_state(state)

    advanced = balance.advance_prescribed(balance_state, 0.0, 1e-4)

    assert bool(advanced.accepted)
    first_diagnostics = advanced.process_diagnostics[0]
    assert first_diagnostics.internal_energy_defect < 1e-10
    assert first_diagnostics.poisson_residual < 1e-8
    assert jnp.abs(first_diagnostics.gauge_defect) < 1e-10
    plan = phx.solver.BalanceLawCheckpointPlan(balance, "gravity-test-mesh")
    path = tmp_path / "gravity.phxckpt"
    written = phx.solver.write_balance_law_checkpoint(path, plan, advanced.runtime_state)
    restored = phx.solver.read_balance_law_checkpoint(path, plan)
    assert written.payload_id == restored.payload_id
    np.testing.assert_array_equal(
        restored.runtime_state.transport_state.content_state.conservative_content,
        advanced.runtime_state.transport_state.content_state.conservative_content,
    )


def test_particle_mesh_deposition_and_kick_drift_kick_are_finite():
    grid, _, _, runtime = _periodic_euler_runtime((8,))
    gravity = phx.solver.NewtonianSelfGravityPlan(0.1).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), jnp.ones((4,)), ambient_dimension=1
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    particle_gravity = phx.solver.ParticleMeshGravityPlan(gravity, transfer)
    position = jnp.asarray([[0.125], [0.375], [0.625], [0.875]])
    state = particle_gravity.initialize(position, velocity=jnp.zeros_like(position))

    deposited, _ = particle_gravity.density(position)
    result = particle_gravity.step(state, 0.0, 1e-3)

    assert deposited.balance.closed_domain_conservation_valid
    assert bool(result.successful)
    assert jnp.all(jnp.isfinite(result.state.position))
    assert jnp.all(jnp.isfinite(result.state.momentum))
    assert result.diagnostics.mass_balance_defect < 1e-10


def test_spectral_ou_replays_real_zero_mean_forcing():
    _, system, _, runtime = _periodic_euler_runtime((4, 4))
    primitive = jnp.zeros((4, 4, 4)).at[..., 0].set(1.0).at[..., -1].set(1.0)
    transport_state = runtime.initialize_state(
        system.primitive_to_conserved(primitive), 0.0, 1e-3
    )
    transport = phx.solver.prepare_balance_law_transport(runtime)
    process = phx.solver.SpectralOUForcingPlan(
        kmin=1.0,
        kmax=2.0,
        solenoidal_fraction=1.0,
        correlation_time=0.2,
        rms_acceleration=0.1,
    ).prepare(transport)
    process_state = process.initialize(transport.source_view(transport_state))
    realization = phx.stochastic.OrnsteinUhlenbeckRealization(
        jr.key(7),
        (4, 4, 2),
        support=(0.0, 1.0),
        noise_id="ou-test",
    )
    first = process.advance(
        0.0,
        1e-3,
        transport_state.cell_average(),
        process_state,
        realization,
    )
    replay = process.advance(
        0.0,
        1e-3,
        transport_state.cell_average(),
        process_state,
        realization,
    )

    assert bool(first.successful)
    np.testing.assert_array_equal(first.cell_average, replay.cell_average)
    np.testing.assert_array_equal(
        first.process_state.values[0], replay.process_state.values[0]
    )
    assert jnp.max(jnp.abs(first.diagnostics.mean_acceleration)) < 1e-6
    assert jnp.all(jnp.isfinite(first.diagnostics.acceleration))


def test_implicit_radiative_cooling_decreases_energy_without_clipping():
    _, system, _, runtime = _periodic_euler_runtime((4,))
    primitive = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 1.0]), (4, 3))
    transport_state = runtime.initialize_state(
        system.primitive_to_conserved(primitive), 0.0, 1e-3
    )
    curve = phx.equations.TabulatedCoolingCurve(
        jnp.asarray([-6.0, 6.0]),
        jnp.asarray([-3.0, -3.0]),
        bounds_policy="power_law_extrapolate",
    )
    transport = phx.solver.prepare_balance_law_transport(runtime)
    cooling = phx.solver.RadiativeCoolingProcessPlan(
        curve,
        amplitude=1.0,
        accuracy_fraction=1.0,
        tolerance=1e-10,
    ).prepare(transport)
    process_state = cooling.initialize(transport.source_view(transport_state))

    result = cooling.advance(
        0.0,
        1e-3,
        transport_state.cell_average(),
        process_state,
    )

    assert bool(result.successful)
    assert jnp.all(result.diagnostics.energy_change < 0.0)
    assert result.diagnostics.maximum_residual < 1e-8
    assert jnp.all(system.admissible(result.cell_average.reshape((4, 3))))


def test_shared_face_closure_is_conservative_and_equal_state_consistent():
    grid, system, discretization, _ = _periodic_euler_runtime((8,))
    closure = phx.discretization.ConservativeFaceClosurePlan(
        lambda system, left, right, baseline, axis, args: args["scale"] * (right - left),
        closure_id="linear-jump-correction",
    )
    problem = phx.equations.ConservationProblemIR(
        "closure-conservation",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
        closure=closure,
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    x = grid.structured_axes[0].interval_centers
    primitive = jnp.stack(
        (jnp.ones_like(x), 0.1 * jnp.sin(2.0 * jnp.pi * x), jnp.ones_like(x)),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)

    residual = dynamics(0.0, state, {"scale": jnp.asarray(0.02)})
    constant = system.primitive_to_conserved(
        jnp.broadcast_to(jnp.asarray([1.0, 0.0, 1.0]), (8, 3))
    )

    np.testing.assert_allclose(jnp.sum(residual, axis=0), 0.0, atol=1e-11)
    np.testing.assert_allclose(
        dynamics(0.0, constant, {"scale": jnp.asarray(0.02)}), 0.0, atol=1e-12
    )


def test_hlld_and_constrained_transport_preserve_constant_mhd_state():
    _, system, integrator, full, magnetic_flux = _periodic_mhd_transport()
    hlld = phx.discretization.HLLDFluxPlan().face_flux(system, full, full, 0)
    state = integrator.initialize(full, magnetic_flux, step_size=1e-4)

    result = integrator.advance(state, 0.0, 1e-4)

    assert not jnp.any(hlld.fallback_activated)
    np.testing.assert_allclose(
        hlld.normal_flux, system.physical_flux(full, 0), rtol=1e-10, atol=1e-10
    )
    assert bool(result.accepted)
    np.testing.assert_allclose(result.state.cell_state, state.cell_state, atol=1e-10)
    np.testing.assert_allclose(
        result.state.magnetic_flux, state.magnetic_flux, atol=1e-10
    )
    assert result.diagnostics.magnetic_constraint_change < 1e-12


def test_unified_mhd_balance_replays_and_checkpoints_cooling(tmp_path):
    _, _, integrator, full, magnetic_flux = _periodic_mhd_transport()
    state = integrator.initialize(full, magnetic_flux, step_size=1e-4)
    transport = phx.solver.prepare_balance_law_transport(integrator)
    curve = phx.equations.TabulatedCoolingCurve(
        jnp.asarray([-6.0, 6.0]),
        jnp.asarray([-3.0, -3.0]),
        bounds_policy="power_law_extrapolate",
    )
    cooling = phx.solver.RadiativeCoolingProcessPlan(
        curve,
        accuracy_fraction=1.0,
        tolerance=1e-10,
    ).prepare(transport)
    runtime = phx.solver.PreparedBalanceLawRuntime(transport, (cooling,))
    initial = runtime.initialize_state(state)
    adaptive = phx.solver.AdaptiveBalanceLawRolloutPlan(
        runtime,
        2e-4,
        phx.solver.BalanceLawAdaptivePolicy(
            2,
            maximum_retries=1,
            safety_factor=1.0,
            growth_factor=1.0,
        ),
    )

    realized = adaptive.rollout(initial)
    scheduled = phx.solver.ScheduledBalanceLawRolloutPlan.from_realized_mesh(
        runtime,
        realized.realized_mesh,
        replay=phx.solver.FiniteVolumeReplayPolicy("block", block_size=1),
    )
    replayed = scheduled.rollout(initial)

    assert bool(realized.completed)
    assert bool(jnp.all(replayed.accepted))
    np.testing.assert_allclose(
        replayed.final_state.transport_state.cell_state,
        realized.final_state.transport_state.cell_state,
    )
    np.testing.assert_array_equal(
        replayed.final_state.transport_state.magnetic_flux,
        realized.final_state.transport_state.magnetic_flux,
    )
    np.testing.assert_array_equal(
        replayed.retained_transport_auxiliary,
        jnp.broadcast_to(
            magnetic_flux,
            replayed.retained_transport_auxiliary.shape,
        ),
    )
    initial_energy = transport.source_view(state).cell_average[..., 4]
    final_energy = transport.source_view(
        realized.final_state.transport_state
    ).cell_average[..., 4]
    assert jnp.all(final_energy < initial_energy)

    checkpoint_plan = phx.solver.BalanceLawCheckpointPlan(
        runtime,
        scheduled.temporal_mesh.mesh_id,
    )
    path = tmp_path / "mhd-balance-law.phxckpt"
    written = phx.solver.write_balance_law_checkpoint(
        path,
        checkpoint_plan,
        realized.final_state,
    )
    restored = phx.solver.read_balance_law_checkpoint(path, checkpoint_plan)
    assert restored.payload_id == written.payload_id
    np.testing.assert_array_equal(
        restored.runtime_state.transport_state.cell_state,
        realized.final_state.transport_state.cell_state,
    )
    np.testing.assert_array_equal(
        restored.runtime_state.transport_state.magnetic_flux,
        realized.final_state.transport_state.magnetic_flux,
    )


def test_mhd_balance_rejects_declared_and_undeclared_magnetic_sources():
    _, _, integrator, full, magnetic_flux = _periodic_mhd_transport()
    state = integrator.initialize(full, magnetic_flux, step_size=1e-4)
    transport = phx.solver.prepare_balance_law_transport(integrator)
    with pytest.raises(ValueError, match="transport-owned"):
        phx.solver.PreparedBalanceLawRuntime(
            transport,
            (_PreparedMagneticMutation(("magnetic_x",)),),
        )

    runtime = phx.solver.PreparedBalanceLawRuntime(
        transport,
        (_PreparedMagneticMutation(),),
    )
    initial = runtime.initialize_state(state)
    result = runtime.advance_prescribed(initial, 0.0, 1e-4)

    assert not bool(result.accepted)
    assert int(result.status) == 3
    np.testing.assert_array_equal(
        result.runtime_state.transport_state.cell_state, state.cell_state
    )
    np.testing.assert_array_equal(
        result.runtime_state.transport_state.magnetic_flux,
        state.magnetic_flux,
    )


def test_mhd_balance_composes_gravity_cooling_and_ou_forcing():
    grid, system, integrator, full, magnetic_flux = _periodic_mhd_transport()
    x = grid.structured_axes[0].interval_centers[:, None, None]
    primitive = system.conserved_to_primitive(full)
    primitive = primitive.at[..., 0].set(
        jnp.broadcast_to(1.0 + 0.01 * jnp.sin(2.0 * jnp.pi * x), primitive.shape[:-1])
    )
    state = integrator.initialize(
        system.primitive_to_conserved(primitive),
        magnetic_flux,
        step_size=1e-5,
    )
    transport = phx.solver.prepare_balance_law_transport(integrator)
    gravity = phx.solver.NewtonianSelfGravityPlan(0.1).prepare(transport)
    forcing = phx.solver.SpectralOUForcingPlan(
        kmin=1.0,
        kmax=1.0,
        correlation_time=0.2,
        rms_acceleration=1e-3,
    ).prepare(transport)
    curve = phx.equations.TabulatedCoolingCurve(
        jnp.asarray([-6.0, 6.0]),
        jnp.asarray([-4.0, -4.0]),
        bounds_policy="power_law_extrapolate",
    )
    cooling = phx.solver.RadiativeCoolingProcessPlan(
        curve,
        accuracy_fraction=1.0,
    ).prepare(transport)
    runtime = phx.solver.PreparedBalanceLawRuntime(
        transport,
        (gravity, forcing, cooling),
    )
    initial = runtime.initialize_state(state)
    realization = phx.stochastic.OrnsteinUhlenbeckRealization(
        jr.key(31),
        integrator.spatial.cell_shape + (3,),
        support=(0.0, 1.0),
        noise_id="mhd-balance-forcing",
    )

    result = runtime.advance_prescribed(
        initial,
        0.0,
        1e-5,
        None,
        realization,
    )

    assert bool(result.accepted)
    assert len(result.process_diagnostics) == 6
    constraint_before = integrator.spatial.magnetic_constraint(magnetic_flux)
    constraint_after = integrator.spatial.magnetic_constraint(
        result.runtime_state.transport_state.magnetic_flux
    )
    np.testing.assert_allclose(constraint_after, constraint_before, atol=1e-12)
    assert jnp.all(
        system.admissible(
            transport.source_view(result.runtime_state.transport_state).cell_average
        )
    )
