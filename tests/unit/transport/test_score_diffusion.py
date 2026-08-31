import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


class _GaussianMarginalScore(eqx.Module):
    process: phx.stochastic.VariancePreservingDiffusion
    mean: jnp.ndarray
    variance: jnp.ndarray

    def __call__(self, state, time):
        slope = (
            self.process.beta_maximum - self.process.beta_minimum
        ) / self.process.terminal_time
        integrated = self.process.beta_minimum * time + 0.5 * slope * time**2
        mean_scale = jnp.exp(-0.5 * integrated)
        noise_variance = -jnp.expm1(-integrated)
        marginal_mean = mean_scale * self.mean
        marginal_variance = mean_scale**2 * self.variance + noise_variance
        return -(state - marginal_mean) / marginal_variance


def _score_function(process, mean, variance):
    state = phx.domain.HyperRectangle(
        jnp.full(process.state_shape, -100.0),
        jnp.full(process.state_shape, 100.0),
        label="x",
    )
    domain = state @ phx.domain.TimeInterval(0.0, process.terminal_time)
    return domain.Function("x", "t")(_GaussianMarginalScore(process, mean, variance))


def _problem():
    process = phx.stochastic.VariancePreservingDiffusion(
        1,
        beta_minimum=0.1,
        beta_maximum=3.0,
    )
    mean = jnp.asarray([0.6])
    variance = jnp.asarray([0.8])
    score = _score_function(process, mean, variance)
    mean_scale = process.transition_mean_scale(0.0, process.terminal_time)
    noise_scale = process.transition_scale(0.0, process.terminal_time)
    terminal = phx.stochastic.DiffusionTerminalReference(
        phx.uq.DiagonalNormalLaw(
            mean_scale * mean,
            jnp.sqrt(mean_scale**2 * variance + noise_scale**2),
            event_shape=(1,),
        ),
        relationship="exact",
        residual_signal_scale=mean_scale,
        reference_id="exact-vp-terminal",
        process_id=process.process_id,
    )
    reverse = phx.transport.ReverseDiffusion(
        process,
        score,
        terminal,
        score_id="analytic-vp-score",
        dt0=0.02,
        wiener_tolerance=1e-4,
    )
    return process, mean, variance, score, terminal, reverse


def test_reverse_diffusion_recovers_gaussian_moments_and_replays():
    process, mean, variance, _, _, reverse = _problem()
    realization = reverse.realize(jr.key(0), (256,))
    first = reverse.solve(realization)
    replay = reverse.solve(realization)
    samples = first.final_states[:, 0]

    assert first.successful
    assert jnp.array_equal(first.final_states, replay.final_states)
    assert jnp.abs(jnp.mean(samples) - mean[0]) < 0.15
    assert jnp.abs(jnp.var(samples) - variance[0]) < 0.2
    assert first.process_id == process.process_id
    assert first.terminal_relationship == "exact"
    trajectory = first.to_stochastic_trajectory()
    assert trajectory.states.shape == (256, 2, 1)
    assert jnp.array_equal(trajectory.states[:, 0], realization.terminal_states)


def test_probability_flow_reuses_continuous_density_contract():
    process, mean, variance, score, terminal, _ = _problem()
    system = phx.transport.probability_flow_system(
        process,
        score,
        state_layout=phx.dynamics.StateLayout((1,)),
        score_id="analytic-vp-score",
    )
    transport = phx.transport.ContinuousTransport(
        terminal.law,
        phx.solver.DiffraxEvolution(system, rtol=1e-9, atol=1e-11),
        source_coordinate=0.0,
        target_coordinate=process.terminal_time,
    )
    flow = phx.transport.ContinuousFlowLaw(transport, max_exact_dimension=4)
    target = phx.uq.DiagonalNormalLaw(
        mean,
        jnp.sqrt(variance),
        event_shape=(1,),
    )
    points = jnp.asarray([[-0.2], [0.5], [1.4]])
    result = flow.log_prob_with_diagnostics(points)

    assert result.successful
    assert jnp.allclose(result.log_prob, target.log_prob(points), atol=2e-7, rtol=2e-7)


def test_reverse_diffusion_rejects_invalid_realization_and_save_grid():
    process, _, _, score, terminal, reverse = _problem()
    realization = reverse.realize(jr.key(1), (2,))
    with pytest.raises(ValueError, match="save_times"):
        reverse.solve(realization, save_times=jnp.asarray([0.0, 1.0]))

    other = phx.transport.ReverseDiffusion(
        process,
        score,
        terminal,
        score_id="different-score",
        dt0=0.02,
        wiener_tolerance=1e-4,
    )
    with pytest.raises(ValueError, match="does not match"):
        other.solve(realization)


def test_asymptotic_terminal_reference_remains_explicit():
    process = phx.stochastic.VariancePreservingDiffusion(2)
    reference = process.asymptotic_terminal_reference()
    assert reference.relationship == "asymptotic"
    assert reference.residual_signal_scale > 0.0
    assert reference.law.event_shape == (2,)
