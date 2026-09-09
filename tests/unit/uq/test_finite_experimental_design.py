#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _space(values):
    return phx.optim.FiniteProductSpace(phx.optim.FiniteAxis(jnp.asarray(values)))


def _table_problem(table, *, design_mask=None, context=None):
    logs = jnp.asarray(table)
    d, p, o = logs.shape

    def channel(parameters, design, outcomes, context):
        return logs[design, parameters[:, None], outcomes[None, :]]

    return phx.uq.FiniteExperimentalDesignProblem(
        _space(np.arange(p)),
        _space(np.arange(d)),
        _space(np.arange(o)),
        channel,
        likelihood_id="finite-channel",
        design_mask=design_mask,
        context=context,
    )


def _dense_eig(probabilities, prior):
    predictive = np.sum(prior[None, :, None] * probabilities, axis=1)
    joint = prior[None, :, None] * probabilities
    supported = joint > 0
    log_ratio = np.log(np.where(supported, probabilities, 1)) - np.log(
        np.where(supported, predictive[:, None, :], 1)
    )
    return np.sum(joint * log_ratio, axis=(1, 2))


def test_exact_channel_selection_observation_and_next_design():
    # Design d asks whether the unknown parameter equals d.
    probability = np.stack(
        [np.stack((np.arange(3) != d, np.arange(3) == d), axis=1) for d in range(3)]
    ).astype(float)
    logs = np.full(probability.shape, -np.inf)
    np.log(probability, out=logs, where=probability > 0)
    problem = _table_problem(logs, context={"sensor_temperature": 295.0})
    prior = np.asarray([0.6, 0.3, 0.1])
    belief = phx.uq.FiniteDesignBelief(problem.parameters, np.log(prior))
    policy = phx.uq.ExpectedInformationGain(candidate_batch_size=2, outcome_batch_size=1)
    score = phx.uq.evaluate_finite_experimental_design(problem, belief, 0, policy=policy)
    assert bool(score.valid)
    assert float(score.expected_information_gain) == pytest.approx(
        _dense_eig(probability, prior)[0]
    )
    first = phx.uq.select_finite_experimental_design(problem, belief, policy=policy)
    assert int(first.design_flat_index) == 0
    assert bool(first.search_result.exact)
    assert first.search_result.landscape_scores is None
    assert first.design_space_id == problem.designs.space_id

    observed = phx.uq.bind_finite_design_experiment(
        problem,
        belief,
        first,
        0,
        experiment_id="real-observation-1",
    )
    update = phx.uq.update_finite_design_belief(problem, belief, observed, policy=policy)
    assert bool(update.accepted)
    np.testing.assert_allclose(jnp.exp(update.belief.log_masses), [0.0, 0.75, 0.25])
    assert np.isneginf(update.belief.log_masses[0])
    assert update.belief.history[0] is observed
    assert belief.history == ()
    assert float(update.log_predictive_probability) == pytest.approx(np.log(0.4))
    next_design = phx.uq.select_finite_experimental_design(
        problem, update.belief, policy=policy
    )
    assert int(next_design.design_flat_index) == 1
    assert next_design.belief_id != first.belief_id
    assert int(observed.observations) == 0
    assert float(observed.conditions["context:sensor_temperature"]) == 295.0


def test_chunked_complete_support_matches_dense_oracle_and_jit():
    probability = np.random.default_rng(17).uniform(0.05, 1.0, (4, 3, 5))
    probability /= probability.sum(axis=2, keepdims=True)
    problem = _table_problem(np.log(probability))
    prior = np.asarray([0.08, 0.71, 0.21])
    belief = phx.uq.FiniteDesignBelief(problem.parameters, np.log(prior))
    expected = _dense_eig(probability, prior)
    for candidates, outcomes in ((1, 1), (3, 3), (4, 5)):
        policy = phx.uq.ExpectedInformationGain(
            candidate_batch_size=candidates, outcome_batch_size=outcomes
        )
        score = jax.jit(
            lambda index: phx.uq.evaluate_finite_experimental_design(
                problem,
                belief,
                index,
                policy=policy,
            )
        )(jnp.asarray(2))
        assert bool(score.valid)
        assert float(score.expected_information_gain) == pytest.approx(
            expected[2], abs=1e-12
        )
        selection = phx.uq.select_finite_experimental_design(
            problem, belief, policy=policy
        )
        assert int(selection.design_flat_index) == int(np.argmax(expected))
        assert float(selection.expected_information_gain) == pytest.approx(
            np.max(expected), abs=1e-12
        )


def test_flat_identity_masks_zero_prior_rows_and_stable_ties():
    parameters = _space(np.arange(3))
    designs = phx.optim.FiniteProductSpace(
        {
            "first": phx.optim.FiniteAxis(jnp.asarray([10.0, 10.0])),
            "second": phx.optim.FiniteAxis(jnp.asarray([20.0, 20.0])),
        }
    )

    def channel(parameters, design, outcomes, context):
        finite = jnp.where(parameters[:, None] == outcomes[None, :], 0.0, -jnp.inf)
        return jnp.where(parameters[:, None] == 2, jnp.nan, finite)

    problem = phx.uq.FiniteExperimentalDesignProblem(
        parameters,
        designs,
        _space([0, 1]),
        channel,
        likelihood_id="masked-perfect-channel",
        design_mask=[False, False, True, True],
    )
    belief = phx.uq.FiniteDesignBelief(
        parameters, [0.0, 0.0, jnp.nan], parameter_mask=[True, True, False]
    )
    selection = phx.uq.select_finite_experimental_design(
        problem,
        belief,
        policy=phx.uq.ExpectedInformationGain(
            candidate_batch_size=3, outcome_batch_size=1
        ),
    )
    assert int(selection.design_flat_index) == 2
    assert tuple(int(index[0]) for index in selection.search_result.product_indices) == (
        1,
        0,
    )
    assert float(selection.expected_information_gain) == pytest.approx(np.log(2))
    inactive = phx.uq.evaluate_finite_experimental_design(problem, belief, 0)
    assert not bool(inactive.valid)
    assert int(inactive.status) == int(phx.uq.FiniteDesignStatus.INACTIVE_DESIGN)
    # A finite but underflowing mass is NOT a zero-prior row: its NaN is invalid.
    tiny = phx.uq.FiniteDesignBelief(parameters, [0.0, -2.0, -1000.0])
    invalid = phx.uq.evaluate_finite_experimental_design(problem, tiny, 2)
    assert bool(tiny.active_parameters[2])
    assert not bool(invalid.valid)
    assert int(invalid.status) == int(phx.uq.FiniteDesignStatus.INVALID_LIKELIHOOD)


def test_zero_mass_parameter_payload_is_not_evaluated():
    parameters = _space([0, 1, 2])

    def strict_channel(parameter_values, design, outcomes, context):
        del design, context
        checked = eqx.error_if(
            parameter_values,
            jnp.any(parameter_values == 2),
            "zero-mass parameter reached likelihood",
        )
        return jnp.where(
            checked[:, None] == outcomes[None, :],
            0.0,
            -jnp.inf,
        )

    problem = phx.uq.FiniteExperimentalDesignProblem(
        parameters,
        _space([0]),
        _space([0, 1]),
        strict_channel,
        likelihood_id="strict-masked-channel",
    )
    belief = phx.uq.FiniteDesignBelief(
        parameters,
        [0.0, 0.0, -jnp.inf],
    )
    score = phx.uq.evaluate_finite_experimental_design(problem, belief, 0)
    assert bool(score.valid)
    assert float(score.expected_information_gain) == pytest.approx(np.log(2.0))


@pytest.mark.parametrize(
    "bad_row",
    [
        [np.log(0.2), np.log(0.3)],  # Missing half the declared outcome mass.
        [np.inf, -np.inf],
        [np.nan, 0.0],
    ],
)
def test_active_likelihood_rows_are_rejected_not_renormalized(bad_row):
    problem = _table_problem(np.asarray([[bad_row, [np.log(0.5), np.log(0.5)]]]))
    belief = phx.uq.FiniteDesignBelief(problem.parameters, [0.0, 0.0])
    score = phx.uq.evaluate_finite_experimental_design(
        problem,
        belief,
        0,
        policy=phx.uq.ExpectedInformationGain(outcome_batch_size=1),
    )
    assert not bool(score.valid)
    assert int(score.status) == int(phx.uq.FiniteDesignStatus.INVALID_LIKELIHOOD)
    assert np.isnan(score.expected_information_gain)
    selection = phx.uq.select_finite_experimental_design(problem, belief)
    assert not bool(selection.valid)
    assert selection.design is None
    assert int(selection.status) == int(phx.uq.FiniteDesignStatus.NO_VALID_DESIGNS)


def test_log_masses_and_rare_observations_never_round_trip_through_probabilities():
    perfect = _table_problem([[[0.0, -np.inf], [-np.inf, 0.0]]])
    offset = phx.uq.FiniteDesignBelief(perfect.parameters, [1e300, 1e300])
    np.testing.assert_allclose(offset.log_masses, [-np.log(2), -np.log(2)], atol=1e-15)
    rare = phx.uq.FiniteDesignBelief(perfect.parameters, [0.0, -1000.0])
    selected = phx.uq.evaluate_finite_experimental_design(perfect, rare, 0)
    observed = phx.uq.bind_finite_design_experiment(
        perfect, rare, selected, 1, experiment_id="rare-event"
    )
    updated = phx.uq.update_finite_design_belief(perfect, rare, observed)
    assert bool(updated.accepted)
    np.testing.assert_array_equal(updated.belief.log_masses, [-np.inf, 0.0])
    assert float(updated.log_predictive_probability) == -1000.0

    # A common huge event log likelihood must not erase the prior log odds.
    common_rare = _table_problem([[[0.0, -1e300], [0.0, -1e300]]])
    prior = phx.uq.FiniteDesignBelief(common_rare.parameters, np.log([0.9, 0.1]))
    score = phx.uq.evaluate_finite_experimental_design(common_rare, prior, 0)
    event = phx.uq.bind_finite_design_experiment(
        common_rare, prior, score, 1, experiment_id="common-rare-event"
    )
    posterior = phx.uq.update_finite_design_belief(common_rare, prior, event)
    assert bool(posterior.accepted)
    np.testing.assert_allclose(posterior.belief.log_masses, prior.log_masses, atol=1e-15)
    assert float(posterior.log_predictive_probability) == -1e300


def test_impossible_stale_replayed_and_tampered_observations_preserve_belief():
    logs = [[[0.0, -np.inf, -np.inf], [-np.inf, 0.0, -np.inf]]]
    problem = _table_problem(logs, context={"calibration": 1.0})
    belief = phx.uq.FiniteDesignBelief(problem.parameters, [0.0, 0.0])
    selection = phx.uq.select_finite_experimental_design(problem, belief)
    impossible = phx.uq.bind_finite_design_experiment(
        problem, belief, selection, 2, experiment_id="impossible"
    )
    failure = phx.uq.update_finite_design_belief(problem, belief, impossible)
    assert not bool(failure.accepted)
    assert failure.belief is belief
    assert int(failure.status) == int(phx.uq.FiniteDesignStatus.IMPOSSIBLE_OBSERVATION)
    assert np.isneginf(failure.log_predictive_probability)

    event = phx.uq.bind_finite_design_experiment(
        problem, belief, selection, 1, experiment_id="observed"
    )
    changed_context = _table_problem(logs, context={"calibration": 2.0})
    stale = phx.uq.update_finite_design_belief(changed_context, belief, event)
    assert stale.belief is belief
    assert int(stale.status) == int(phx.uq.FiniteDesignStatus.STALE_IDENTITY)
    tampered = phx.uq.Experiment(
        "observed",
        jnp.asarray(0),
        conditions=event.conditions,
        likelihood_id=event.likelihood_id,
    )
    rejected = phx.uq.update_finite_design_belief(problem, belief, tampered)
    assert rejected.belief is belief
    assert int(rejected.status) == int(phx.uq.FiniteDesignStatus.STALE_IDENTITY)
    accepted = phx.uq.update_finite_design_belief(problem, belief, event)
    replayed = phx.uq.update_finite_design_belief(problem, accepted.belief, event)
    assert replayed.belief is accepted.belief
    assert int(replayed.status) == int(phx.uq.FiniteDesignStatus.STALE_IDENTITY)
    assert len(replayed.belief.history) == 1


def test_resource_refusal_precedes_likelihood_tracing_and_search_allocation():
    calls = []

    def channel(parameters, design, outcomes, context):
        calls.append(outcomes.shape)
        return jnp.full((parameters.shape[0], outcomes.shape[0]), -jnp.log(2.0))

    designs = phx.optim.FiniteProductSpace(
        tuple(phx.optim.FiniteAxis(jnp.arange(1000)) for _ in range(3))
    )
    problem = phx.uq.FiniteExperimentalDesignProblem(
        _space([0, 1]),
        designs,
        _space([0, 1]),
        channel,
        likelihood_id="uniform",
    )
    belief = phx.uq.FiniteDesignBelief(problem.parameters, [0.0, 0.0])
    policy = phx.uq.ExpectedInformationGain(maximum_bytes=1)
    with pytest.raises(MemoryError):
        phx.uq.evaluate_finite_experimental_design(problem, belief, 0, policy=policy)
    valid = phx.uq.evaluate_finite_experimental_design(problem, belief, 0)
    experiment = phx.uq.bind_finite_design_experiment(
        problem, belief, valid, 0, experiment_id="resource-preflight"
    )
    calls.clear()
    with pytest.raises(MemoryError):
        phx.uq.update_finite_design_belief(problem, belief, experiment, policy=policy)
    with pytest.raises(MemoryError):
        phx.uq.select_finite_experimental_design(problem, belief, policy=policy)
    assert calls == []
