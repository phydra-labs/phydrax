import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _compiled_channel():
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(4),
            phx.discretization.ChebyshevBasisPlan(8),
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
    plan = phx.discretization.ChannelStokesPlan(
        space,
        0.1,
        lower_wall_velocity=(-1.0, 0.0, 0.0),
        upper_wall_velocity=(1.0, 0.0, 0.0),
    )
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    problem = phx.equations.IncompressibleFlowProblem(3, 0.1)
    return space, phx.equations.compile_channel_flow(problem, plan, method)


def test_channel_sbdf2_preserves_steady_couette_profile():
    space, dynamics = _compiled_channel()
    y = space.axes[1].nodes
    couette = jnp.zeros(space.physical_shape + (3,)).at[..., 0].set(y[None, :, None])
    initial = dynamics.project_state(couette)
    solution = phx.solver.solve_channel_sbdf2(
        dynamics,
        initial,
        jnp.asarray([0.0, 0.01, 0.02]),
    )
    final = dynamics.reconstruct_state(solution.velocity[-1])

    assert bool(solution.successful)
    np.testing.assert_allclose(np.asarray(final), np.asarray(couette), atol=1e-10)
    assert jnp.nanmax(solution.diagnostics.divergence_norm) < 1e-10
    assert jnp.nanmax(solution.diagnostics.wall_residual) < 1e-10
    assert jnp.nanmax(solution.diagnostics.pressure_gauge_residual) < 1e-10


def test_channel_compiler_rejects_mismatched_problem_viscosity():
    space, dynamics = _compiled_channel()
    with pytest.raises(ValueError, match="viscosities"):
        phx.equations.compile_channel_flow(
            phx.equations.IncompressibleFlowProblem(3, 0.2),
            dynamics.stokes_plan,
            phx.discretization.PseudospectralMethodPlan(
                dealiasing=phx.discretization.PaddingDealiasingPlan(2)
            ),
        )
    assert space.prepared_id == dynamics.discretization.prepared_id


def test_channel_sbdf2_rejects_constraint_invalid_initial_state_without_advancing():
    _, dynamics = _compiled_channel()
    initial = jnp.zeros(dynamics.state_shape, dtype=complex)
    solution = phx.solver.solve_channel_sbdf2(
        dynamics,
        initial,
        jnp.asarray([0.0, 0.01, 0.02]),
    )

    assert not bool(solution.successful)
    assert (
        int(solution.diagnostics.status[0]) == phx.solver.CHANNEL_FLOW_INITIAL_CONSTRAINT
    )
    assert jnp.all(~solution.diagnostics.valid)
    np.testing.assert_allclose(solution.velocity, 0.0, atol=0.0)


def _perturbed_channel_state(space, dynamics):
    y = space.axes[1].nodes
    z = space.axes[2].nodes
    streamwise = y[None, :, None] + 0.05 * (1.0 - y[None, :, None] ** 2) * jnp.cos(
        z[None, None, :]
    )
    physical = (
        jnp.zeros(space.physical_shape + (3,))
        .at[..., 0]
        .set(jnp.broadcast_to(streamwise, space.physical_shape))
    )
    return dynamics.project_state(physical)


def test_channel_prepared_restart_matches_uninterrupted_history():
    space, dynamics = _compiled_channel()
    initial = _perturbed_channel_state(space, dynamics)
    step = 0.01
    method = phx.solver.ChannelSBDF2Method()
    prepared = method.prepare(dynamics, step)
    state0 = prepared.initialize(initial, 0.0, None)
    first = prepared.step(0, 0.0, state0, step, None).accepted_state
    second = prepared.step(1, step, first, step, None).accepted_state
    third = prepared.step(2, 2.0 * step, second, step, None).accepted_state
    restarted_after_startup = (
        method.prepare(dynamics, step)
        .step(
            1,
            step,
            first,
            step,
            None,
        )
        .accepted_state
    )
    restarted_later = (
        method.prepare(dynamics, step)
        .step(
            2,
            2.0 * step,
            second,
            step,
            None,
        )
        .accepted_state
    )
    solution = phx.solver.solve_channel_sbdf2(
        dynamics,
        initial,
        jnp.asarray([0.0, step, 2.0 * step, 3.0 * step]),
        method=method,
    )

    assert prepared.required_step_size == step
    assert not prepared.allows_step_reduction
    assert int(first.history_count) == 1
    assert int(second.history_count) == 2
    assert int(third.history_count) == 3
    for uninterrupted, restarted in (
        (second, restarted_after_startup),
        (third, restarted_later),
    ):
        for left, right in zip(
            jax.tree.leaves(uninterrupted),
            jax.tree.leaves(restarted),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right))
    np.testing.assert_allclose(
        np.asarray(solution.velocity[-1]),
        np.asarray(third.current_velocity),
    )
    np.testing.assert_allclose(
        np.asarray(solution.pressure[-1]),
        np.asarray(third.current_pressure),
    )
    np.testing.assert_allclose(
        np.asarray(solution.pressure_gradient[-1]),
        np.asarray(third.pressure_gradient),
    )


def test_channel_prepared_failure_preserves_history_and_rejects_changed_step():
    space, dynamics = _compiled_channel()
    step = 0.01
    prepared = phx.solver.ChannelSBDF2Method().prepare(dynamics, step)
    state = prepared.step(
        0,
        0.0,
        prepared.initialize(_perturbed_channel_state(space, dynamics), 0.0, None),
        step,
        None,
    ).accepted_state
    invalid = phx.solver.ChannelSBDF2State(
        state.previous_velocity,
        jnp.zeros_like(state.current_velocity),
        state.previous_nonlinear_rhs,
        state.current_nonlinear_rhs,
        state.current_pressure,
        state.pressure_gradient,
        state.history_count,
    )
    failed = prepared.step(1, step, invalid, step, None)

    assert not bool(failed.successful)
    for incoming, accepted in zip(
        jax.tree.leaves(invalid),
        jax.tree.leaves(failed.accepted_state),
        strict=True,
    ):
        assert jnp.array_equal(incoming, accepted)
    with pytest.raises(Exception, match="exactly equal"):
        prepared.step(1, step, state, 0.5 * step, None)


def test_bounded_observer_reports_overflow_without_growing_state():
    layout = phx.dynamics.StateLayout((1,))
    system = phx.dynamics.DiscreteSystem(
        lambda step, state, args: 0.5 * state,
        state_layout=layout,
        system_id="bounded-observer-decay",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    plan = phx.solver.BoundedEvolutionObservationPlan(
        lambda coordinate, state, args: state,
        (1,),
        3,
        observer_id="bounded-state-observer",
    )
    result = phx.solver.observe_evolution_bounded(
        evolution,
        jnp.asarray([1.0]),
        jnp.arange(6.0),
        plan,
    )

    assert int(result.count) == 3
    assert bool(result.overflow)
    np.testing.assert_allclose(np.asarray(result.values[:, 0]), [1.0, 0.5, 0.25])
    np.testing.assert_allclose(np.asarray(result.final_state), [0.03125])


def test_bounded_observer_latches_nonfinite_observable_status():
    layout = phx.dynamics.StateLayout((1,))
    system = phx.dynamics.DiscreteSystem(
        lambda step, state, args: state,
        state_layout=layout,
        system_id="bounded-observer-nonfinite-map",
    )
    plan = phx.solver.BoundedEvolutionObservationPlan(
        lambda coordinate, state, args: jnp.where(coordinate > 0.0, jnp.nan, state),
        (1,),
        3,
        observer_id="bounded-nonfinite-observer",
    )
    result = phx.solver.observe_evolution_bounded(
        phx.dynamics.DiscreteEvolution(system),
        jnp.asarray([1.0]),
        jnp.arange(3.0),
        plan,
    )

    assert not bool(result.final_valid)
    assert int(result.final_status) == phx.solver.OBSERVATION_NONFINITE


def test_spectral_seed_and_fixed_step_checkpoint_roundtrip(tmp_path):
    state = jnp.asarray([1.0 + 0.5j, 2.0 - 0.25j])
    artifact = phx.solver.SpectralStateArtifact(
        state,
        0.5,
        5,
        discretization_id="spectral-space",
        compilation_id="compiled-flow",
        method_id="etdrk4",
        source_hash="problem-source",
        step_size=0.1,
        restartable=True,
        extra={"constraint": "mean-zero"},
    )
    with pytest.raises(ValueError, match="artifact_id"):
        phx.solver.SpectralStateArtifact(
            state,
            0.5,
            5,
            discretization_id="spectral-space",
            compilation_id="compiled-flow",
            method_id="etdrk4",
            source_hash="problem-source",
            step_size=0.1,
            restartable=True,
            artifact_id="0" * 64,
        )
    path = phx.solver.write_spectral_state_artifact(
        tmp_path / "state.phx",
        artifact,
    )
    restored = phx.solver.read_spectral_state_artifact(
        path,
        expected_discretization_id="spectral-space",
        expected_compilation_id="compiled-flow",
    )

    assert restored.restartable
    assert restored.artifact_id == artifact.artifact_id
    assert restored.extra == {"constraint": "mean-zero"}
    np.testing.assert_allclose(np.asarray(restored.state), np.asarray(state))
    np.testing.assert_allclose(np.asarray(restored.step_size), 0.1)


def test_hermitian_spectral_artifact_uses_minimal_real_storage(tmp_path):
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(8),)
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    state = space.project(jnp.cos(2.0 * jnp.pi * space.axes[0].nodes))
    coordinates = phx.discretization.HermitianSpectralCoordinates(space)
    artifact = phx.solver.SpectralStateArtifact(
        state,
        0.0,
        0,
        discretization_id=space.prepared_id,
        compilation_id="packed-dns",
        method_id="fixed-step",
        source_hash="packed-dns-source",
    )
    path = phx.solver.write_spectral_state_artifact(
        tmp_path / "packed.phx",
        artifact,
        state_coordinates=coordinates,
    )
    restored = phx.solver.read_spectral_state_artifact(
        path,
        state_coordinates=coordinates,
        expected_discretization_id=space.prepared_id,
    )

    np.testing.assert_allclose(restored.state, state)
    assert restored.coordinate_evidence.evidence_id == coordinates.evidence.evidence_id
    assert restored.stored_state_bytes < restored.full_state_bytes
    assert restored.fixed_coordinate_count == coordinates.fixed_mode_count
    assert restored.conjugate_pair_count == coordinates.conjugate_pair_count
