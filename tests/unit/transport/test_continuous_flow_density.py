import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


class _LinearField(eqx.Module):
    matrix: jnp.ndarray
    shift: jnp.ndarray

    def __init__(self, matrix, shift=None):
        self.matrix = jnp.asarray(matrix, dtype=float)
        self.shift = (
            jnp.zeros((self.matrix.shape[0],))
            if shift is None
            else jnp.asarray(shift, dtype=float)
        )

    def __call__(self, time, state, args):
        del time, args
        return self.matrix @ state + self.shift


def _normal(location, covariance):
    location = jnp.asarray(location, dtype=float)
    family = phx.uq.MultivariateNormalFamily(int(location.shape[0]))
    return family.law_from_location_covariance(location, covariance)


def _flow(matrix, *, shift=None, max_exact_dimension=32):
    matrix = jnp.asarray(matrix, dtype=float)
    dimension = int(matrix.shape[0])
    system = phx.dynamics.ContinuousSystem(
        _LinearField(matrix, shift),
        state_layout=phx.dynamics.StateLayout((dimension,)),
        system_id="continuous-flow-density-test-system",
    )
    evolution = phx.solver.DiffraxEvolution(
        system,
        rtol=2e-9,
        atol=2e-11,
        max_steps=2048,
    )
    transport = phx.transport.ContinuousTransport(
        _normal(jnp.zeros((dimension,)), jnp.eye(dimension)),
        evolution,
    )
    return phx.transport.ContinuousFlowLaw(
        transport,
        max_exact_dimension=max_exact_dimension,
    )


def test_zero_and_translation_flows_preserve_base_log_density():
    zero = _flow(jnp.zeros((2, 2)))
    values = jnp.asarray([[0.2, -0.4], [1.0, 0.5]])
    expected = _normal(jnp.zeros((2,)), jnp.eye(2)).log_prob(values)

    assert jnp.allclose(zero.log_prob(values), expected, atol=2e-8)

    translation = _flow(jnp.zeros((2, 2)), shift=[1.0, -2.0])
    translated = values + jnp.asarray([1.0, -2.0])
    assert jnp.allclose(translation.log_prob(translated), expected, atol=3e-7)


def test_diagonal_linear_flow_matches_analytic_gaussian_and_log_volume():
    rates = jnp.asarray([0.3, -0.2])
    flow = _flow(jnp.diag(rates))
    data = jnp.asarray([[0.4, -0.7], [1.2, 0.1], [-0.5, 0.9]])
    target = _normal(jnp.zeros((2,)), jnp.diag(jnp.exp(2.0 * rates)))

    result = eqx.filter_jit(flow.log_prob_with_diagnostics)(data)

    assert result.successful
    assert jnp.allclose(result.log_prob, target.log_prob(data), atol=4e-7)
    assert jnp.allclose(result.log_volume, -jnp.sum(rates), atol=3e-8)
    assert jnp.all(result.accepted_steps > 0)


def test_sample_and_log_prob_agrees_with_inverse_evaluation_and_gradients():
    flow = _flow(jnp.asarray([[0.2, 0.0], [0.0, -0.1]]))
    samples, sampled_log_prob = flow.sample_and_log_prob(jr.key(11), (5,))
    inverse_log_prob = flow.log_prob(samples)

    assert samples.shape == (5, 2)
    assert sampled_log_prob.shape == (5,)
    assert jnp.allclose(sampled_log_prob, inverse_log_prob, atol=5e-7)

    value = jnp.asarray([0.3, -0.2])
    gradient = eqx.filter_grad(lambda current: current.log_prob(value))(flow)
    leaves = [leaf for leaf in jax.tree.leaves(gradient) if eqx.is_inexact_array(leaf)]
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_one_dimensional_flow_density_integrates_to_one():
    flow = _flow(jnp.asarray([[0.25]]))
    grid = jnp.linspace(-8.0, 8.0, 65)[:, None]
    density = jnp.exp(flow.log_prob(grid))
    mass = jnp.trapezoid(density, grid[:, 0])

    assert jnp.allclose(mass, 1.0, atol=2e-5)


def test_stochastic_log_density_reports_replayable_probe_uncertainty():
    matrix = jnp.asarray([[0.1, 0.8], [-0.4, -0.2]])
    flow = _flow(matrix)
    value = jnp.asarray([0.25, -0.5])
    policy = phx.operators.StochasticTracePolicy(128, distribution="rademacher")

    first = phx.transport.estimate_continuous_flow_log_prob(
        flow.transport, value, jr.key(17), policy=policy
    )
    replay = phx.transport.estimate_continuous_flow_log_prob(
        flow.transport, value, jr.key(17), policy=policy
    )
    changed = phx.transport.estimate_continuous_flow_log_prob(
        flow.transport, value, jr.key(18), policy=policy
    )
    exact = flow.log_prob(value)

    assert first.num_probes == 128
    assert jnp.array_equal(first.probe_log_volumes, replay.probe_log_volumes)
    assert not jnp.array_equal(first.probe_log_volumes, changed.probe_log_volumes)
    assert jnp.abs(first.log_prob - exact) <= 4.0 * first.standard_error + 2e-5
    assert first.standard_error > 0.0


def test_continuous_flow_density_rejects_unsupported_contracts():
    with pytest.raises(ValueError, match="exceeds cap"):
        _flow(jnp.zeros((3, 3)), max_exact_dimension=2)

    layout = phx.dynamics.StateLayout(())
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, args: jnp.zeros_like(state),
        state_layout=layout,
        system_id="counting-flow-test",
    )
    transport = phx.transport.ContinuousTransport(
        phx.uq.EmpiricalDistribution(jnp.asarray([0.0, 1.0])),
        phx.solver.DiffraxEvolution(system),
    )
    with pytest.raises(ValueError, match="Lebesgue"):
        phx.transport.ContinuousFlowLaw(transport)


def test_piecewise_density_reduces_validity_over_active_tape_slots():
    event = phx.solver.HybridEventPlan(
        lambda time, state, args: state[0] - 0.5,
        lambda time, state, args: state,
        lambda time, state, args: jnp.ones_like(state),
        lambda time, state, args: 2.0 * jnp.ones_like(state),
        event_kind="velocity-change",
        plan_id="piecewise-density-velocity-change",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (phx.solver.ScheduledHybridEvent(event),),
        maximum_events=2,
    )
    prepared = phx.solver.prepare_hybrid_schedule(schedule, jnp.asarray([0.0]))
    schedule_result = phx.solver.execute_hybrid_schedule(
        prepared,
        lambda time, args: jnp.asarray([time]),
        jnp.asarray([[0.0, 1.0]]),
    )
    flow = _flow(jnp.zeros((1, 1)))
    law = phx.transport.PiecewiseContinuousFlowLaw(
        flow.transport,
        prepared,
        forward_event_map=lambda state, prepared: state,
        inverse_event_map=lambda state, prepared: state,
        tape_provider=lambda data, pre_event, prepared: schedule_result.tape,
    )

    result = law.log_prob_with_diagnostics(jnp.asarray([0.2]))

    assert result.valid.shape == ()
    assert result.valid
    assert jnp.isclose(result.event_log_abs_determinant, jnp.log(2.0))


def test_piecewise_density_binds_preparation_and_replay_policy_identity():
    event = phx.solver.HybridEventPlan(
        lambda time, state, args: state[0] - 0.5,
        lambda time, state, args: state,
        lambda time, state, args: jnp.ones_like(state),
        lambda time, state, args: 2.0 * jnp.ones_like(state),
        event_kind="velocity-change",
        plan_id="piecewise-density-policy-identity",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (phx.solver.ScheduledHybridEvent(event),),
        maximum_events=2,
    )
    prepared = phx.solver.prepare_hybrid_schedule(schedule, jnp.asarray([0.0]))
    schedule_result = phx.solver.execute_hybrid_schedule(
        prepared,
        lambda time, args: jnp.asarray([time]),
        jnp.asarray([[0.0, 1.0]]),
    )
    flow = _flow(jnp.zeros((1, 1)))

    def density_law(current_prepared):
        return phx.transport.PiecewiseContinuousFlowLaw(
            flow.transport,
            current_prepared,
            forward_event_map=lambda state, prepared: state,
            inverse_event_map=lambda state, prepared: state,
            tape_provider=lambda data, pre_event, prepared: schedule_result.tape,
            law_id="shared-requested-density-law",
        )

    law = density_law(prepared)
    matching = law.log_prob_with_diagnostics(jnp.asarray([0.2]))
    assert matching.valid.shape == ()
    assert matching.valid
    assert jnp.isfinite(matching.log_prob)

    policy = prepared.replay_policy
    alternate_policies = (
        phx.solver.HybridReplayPolicy(
            policy.maximum_events,
            grazing_tolerance=2.0 * policy.grazing_tolerance,
            simultaneous_tolerance=policy.simultaneous_tolerance,
            event_tolerance=policy.event_tolerance,
            failure=policy.failure,
        ),
        phx.solver.HybridReplayPolicy(
            policy.maximum_events,
            grazing_tolerance=policy.grazing_tolerance,
            simultaneous_tolerance=policy.simultaneous_tolerance,
            event_tolerance=policy.event_tolerance,
            failure=policy.failure - 1,
        ),
    )
    for alternate_policy in alternate_policies:
        alternate_prepared = phx.solver.prepare_hybrid_schedule(
            schedule,
            jnp.asarray([0.0]),
            replay_policy=alternate_policy,
        )
        alternate_law = density_law(alternate_prepared)

        assert alternate_prepared.schedule_id == prepared.schedule_id
        assert alternate_prepared.preparation_id != prepared.preparation_id
        assert alternate_law.law_id != law.law_id
        with pytest.raises(ValueError, match="replay-policy identity"):
            alternate_law.log_prob_with_diagnostics(jnp.asarray([0.2]))
