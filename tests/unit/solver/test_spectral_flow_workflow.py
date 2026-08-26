import jax.numpy as jnp
import numpy as np

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
    ).prepare(jnp.asarray([[0.0, -1.0, 0.0], [2.0 * jnp.pi, 1.0, 2.0 * jnp.pi]]))
    plan = phx.discretization.ChannelStokesPlan(
        space,
        0.1,
        lower_wall_velocity=(-1.0, 0.0, 0.0),
        upper_wall_velocity=(1.0, 0.0, 0.0),
    )
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    return space, phx.equations.compile_channel_flow(plan, method)


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
