import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _gaussian_process():
    return phx.stochastic.LatentGaussianCoefficientProcess(
        jnp.asarray([0.2, -0.1]),
        jnp.asarray([[0.4, 0.1], [0.0, 0.3]]),
        label="latent-coefficients",
    )


def test_gaussian_coefficient_realization_replays_across_query_schedules():
    process = _gaussian_process()
    initial = jnp.asarray([1.0, -1.0])
    realization = process.realize(
        jr.key(0),
        initial,
        support=(0.0, 1.0),
        sample_shape=(128,),
        tolerance=1e-4,
        label="shared-driver",
    )
    coarse_times = jnp.asarray([0.0, 0.5, 1.0])
    fine_times = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    coarse = process.evaluate(realization, coarse_times)
    replay = process.evaluate(realization, coarse_times)
    consistency = phx.stochastic.process_query_consistency(
        process,
        realization,
        coarse_times,
        fine_times,
    )

    assert coarse.states.shape == (128, 3, 2)
    assert coarse.realization_axes == ("process_0",)
    assert coarse.metadata["uncertainty_source"] == "process"
    assert coarse.metadata["process_realization_id"] == realization.realization_id
    assert jnp.array_equal(coarse.states, replay.states)
    assert consistency.shared_times == 3
    assert consistency.consistent
    assert consistency.max_absolute_error == 0.0


def test_gaussian_pathwise_cocycle_and_marginal_semigroup_contracts():
    process = _gaussian_process()
    state = jnp.asarray([1.0, -1.0])
    first = jnp.asarray([0.2, -0.1])
    second = jnp.asarray([0.1, 0.3])
    cocycle_loss = phx.stochastic.cocycle_objective(
        process,
        state,
        t0=0.0,
        tmid=0.4,
        t1=1.0,
        first_driver_segment=first,
        second_driver_segment=second,
    )
    semigroup_loss = phx.stochastic.semigroup_objective(
        process,
        state,
        t0=0.0,
        tmid=0.4,
        t1=1.0,
        key=jr.key(1),
        num_samples=4096,
    )
    marginal = process.marginal_transition(state, t0=0.0, t1=1.0)

    assert isinstance(process, phx.stochastic.AbstractPathwiseTransition)
    assert isinstance(process, phx.stochastic.AbstractMarginalTransitionLaw)
    assert isinstance(marginal, phx.stochastic.GaussianProcessDistribution)
    assert marginal.uncertainty_source == "process"
    assert cocycle_loss < 1e-24
    assert semigroup_loss < 5e-3
    assert jnp.isfinite(marginal.log_prob(marginal.mean))


def test_gaussian_process_diagnostics_match_marginal_moments():
    process = _gaussian_process()
    realization = process.realize(
        jr.key(2),
        jnp.asarray([1.0, -1.0]),
        support=(0.0, 1.0),
        sample_shape=(2048,),
        tolerance=1e-4,
    )
    diagnostics = phx.stochastic.gaussian_process_diagnostics(
        process,
        realization,
        jnp.asarray([0.0, 0.4, 1.0]),
    )

    assert diagnostics.uncertainty_source == "process"
    assert diagnostics.num_samples == 2048
    assert diagnostics.mean_relative_error < 0.04
    assert diagnostics.covariance_relative_error < 0.06
    assert diagnostics.query_max_absolute_error == 0.0
    assert diagnostics.cocycle_max_absolute_error < 1e-12
    assert diagnostics.replay_exact


def test_process_realization_keeps_input_and_process_uncertainty_separate():
    process = _gaussian_process()
    with pytest.raises(ValueError, match="separate input-uncertainty axis"):
        process.realize(
            jr.key(3),
            jnp.ones((4, 2)),
            support=(0.0, 1.0),
            sample_shape=(4,),
        )

    first = process.realize(
        jr.key(4),
        jnp.zeros((2,)),
        support=(0.0, 1.0),
        sample_shape=(4,),
    )
    replay = process.realize(
        jr.key(4),
        jnp.zeros((2,)),
        support=(0.0, 1.0),
        sample_shape=(4,),
    )
    changed = process.realize(
        jr.key(5),
        jnp.zeros((2,)),
        support=(0.0, 1.0),
        sample_shape=(4,),
    )
    assert first.uncertainty_source == "process"
    assert first.realization_id == replay.realization_id
    assert first.realization_id != changed.realization_id


def test_flowjax_coefficient_process_is_a_marginal_law_not_a_path_claim():
    process = phx.nn.models.conditional_coupling_flow_process(
        jr.key(6),
        state_shape=(2,),
        flow_layers=2,
        nn_width=8,
        label="learned-transition",
    )
    state = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [-1.0, 0.5]])
    distribution = process.marginal_transition(state, t0=0.0, t1=0.2)
    samples = distribution.sample(jr.key(7), (32,))
    statistics = phx.stochastic.process_sample_statistics(distribution, samples)

    assert isinstance(process, phx.stochastic.AbstractMarginalTransitionLaw)
    assert not isinstance(process, phx.stochastic.AbstractPathwiseTransition)
    assert isinstance(distribution, phx.nn.models.FlowJAXProcessDistribution)
    assert distribution.uncertainty_source == "process"
    assert distribution.batch_shape == (3,)
    assert distribution.event_shape == (2,)
    assert samples.shape == (32, 3, 2)
    assert distribution.log_prob(samples[0]).shape == (3,)
    assert statistics.mean.shape == (3, 2)
    assert statistics.covariance.shape == (3, 2, 2)
    assert statistics.uncertainty_source == "process"
    assert statistics.finite_fraction == 1.0
    assert jnp.isfinite(statistics.average_log_prob)


def test_flowjax_semigroup_objective_is_differentiable_but_not_assumed_satisfied():
    process = phx.nn.models.conditional_coupling_flow_process(
        jr.key(8),
        state_shape=(2,),
        flow_layers=2,
        nn_width=8,
    )

    def objective(model):
        return phx.stochastic.semigroup_objective(
            model,
            jnp.asarray([0.2, -0.1]),
            t0=0.0,
            tmid=0.3,
            t1=0.8,
            key=jr.key(9),
            num_samples=16,
        )

    value, gradient = eqx.filter_value_and_grad(objective)(process)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(gradient) if eqx.is_array(leaf)]

    assert jnp.isfinite(value)
    assert any(jnp.all(jnp.isfinite(leaf)) for leaf in leaves if leaf is not None)


def test_process_query_consistency_rejects_unmatched_schedules():
    process = _gaussian_process()
    realization = process.realize(
        jr.key(10),
        jnp.zeros((2,)),
        support=(0.0, 1.0),
        sample_shape=(2,),
    )
    with pytest.raises(ValueError, match="exactly once"):
        phx.stochastic.process_query_consistency(
            process,
            realization,
            jnp.asarray([0.0, 0.5, 1.0]),
            jnp.asarray([0.0, 1.0]),
        )
