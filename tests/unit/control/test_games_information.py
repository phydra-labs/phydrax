#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.control.games._information import (
    CentralizedObservationInformation,
    CommonInformationEquilibriumSelector,
    FiniteStateCommonInformationGame,
    FullStateInformation,
    GaussianBelief,
    solve_common_information_game,
)


def _first_selector():
    return CommonInformationEquilibriumSelector(
        lambda equilibria: int(equilibria.equilibrium_indices[0]),
        selector_id="lexicographic-first",
    )


def _last_selector():
    return CommonInformationEquilibriumSelector(
        lambda equilibria: int(equilibria.equilibrium_indices[-1]),
        selector_id="lexicographic-last",
    )


def _coordination_game():
    horizon = 1
    common_states = 1
    joint_types = 2
    joint_actions = 4
    beliefs = jnp.asarray([[[0.5, 0.5]], [[0.5, 0.5]]])
    type_transitions = jnp.zeros(
        (horizon, common_states, joint_types, joint_actions, joint_types)
    )
    for private_type in range(joint_types):
        type_transitions = type_transitions.at[:, :, private_type, :, private_type].set(
            1.0
        )
    observations = jnp.ones(
        (horizon, common_states, joint_types, joint_actions, joint_types, 1)
    )
    stage_costs = jnp.zeros((horizon, common_states, joint_types, joint_actions, 2))
    for action_index, (row_action, column_action) in enumerate(
        ((0, 0), (0, 1), (1, 0), (1, 1))
    ):
        if row_action != column_action:
            stage_costs = stage_costs.at[:, :, :, action_index].set(1.0)
    return FiniteStateCommonInformationGame(
        ("row", "column"),
        (2, 1),
        (2, 2),
        beliefs,
        type_transitions,
        observations,
        stage_costs,
        jnp.zeros((common_states, joint_types, 2)),
        game_id="private-type-coordination",
    )


def _bayes_game(*, zero_probability_observation=False):
    beliefs = jnp.asarray(
        [
            [[0.25, 0.75], [0.25, 0.75]],
            [[0.5, 0.5], [0.0, 1.0]],
        ]
    )
    type_transitions = jnp.zeros((1, 2, 2, 1, 2))
    type_transitions = type_transitions.at[:, :, 0, :, 0].set(1.0)
    type_transitions = type_transitions.at[:, :, 1, :, 1].set(1.0)
    observations = jnp.empty((1, 2, 2, 1, 2, 2))
    if zero_probability_observation:
        observations = observations.at[:].set(jnp.asarray([1.0, 0.0]))
        beliefs = beliefs.at[:].set(jnp.asarray([1.0, 0.0]))
    else:
        observations = observations.at[:, :, :, :, 0].set(jnp.asarray([1.0, 0.0]))
        observations = observations.at[:, :, :, :, 1].set(
            jnp.asarray([1.0 / 3.0, 2.0 / 3.0])
        )
    return FiniteStateCommonInformationGame(
        ("controller",),
        (2,),
        (1,),
        beliefs,
        type_transitions,
        observations,
        jnp.zeros((1, 2, 2, 1, 1)),
        jnp.zeros((2, 2, 1)),
        game_id=(
            "zero-support-observation"
            if zero_probability_observation
            else "bayes-normalization"
        ),
    )


def _two_stage_dominance_game():
    stage_costs = jnp.zeros((2, 1, 1, 4, 2))
    for action_index, (first_action, second_action) in enumerate(
        ((0, 0), (0, 1), (1, 0), (1, 1))
    ):
        stage_costs = stage_costs.at[0, 0, 0, action_index].set(
            jnp.asarray([3.0 + 2.0 * first_action, 4.0 + 2.0 * second_action])
        )
        stage_costs = stage_costs.at[1, 0, 0, action_index].set(
            jnp.asarray([1.0 + 2.0 * first_action, 2.0 + 2.0 * second_action])
        )
    return FiniteStateCommonInformationGame(
        ("first", "second"),
        (1, 1),
        (2, 2),
        jnp.ones((3, 1, 1)),
        jnp.ones((2, 1, 1, 4, 1)),
        jnp.ones((2, 1, 1, 4, 1, 1)),
        stage_costs,
        jnp.asarray([[[5.0, 7.0]]]),
        game_id="two-stage-dominance",
    )


def test_information_values_preserve_only_the_declared_identity():
    state = object()
    observation = object()
    full_state = FullStateInformation(information_id="plant-state")
    centralized = CentralizedObservationInformation(
        information_id="fusion-center-observation"
    )

    assert full_state.policy_input(state) is state
    assert centralized.policy_input(observation) is observation
    assert full_state.information_id == "plant-state"
    assert centralized.information_id == "fusion-center-observation"
    assert full_state.timing == "pre-action"
    assert centralized.timing == "pre-action"
    assert tuple(inspect.signature(centralized.policy_input).parameters) == (
        "observation",
    )
    with pytest.raises(TypeError):
        centralized.policy_input(observation, latent_private_state=object())


def test_gaussian_belief_validates_shape_finiteness_symmetry_and_semidefiniteness():
    belief = GaussianBelief(
        jnp.asarray([1.0, -2.0]),
        jnp.asarray([[2.0, 0.0], [0.0, 0.0]]),
        belief_id="possibly-singular",
    )
    assert belief.dimension == 2
    assert belief.belief_id == "possibly-singular"
    np.testing.assert_array_equal(belief.mean, [1.0, -2.0])

    with pytest.raises(ValueError, match="shape"):
        GaussianBelief(jnp.zeros(2), jnp.eye(3))
    with pytest.raises(ValueError, match="finite"):
        GaussianBelief(jnp.asarray([jnp.nan]), jnp.ones((1, 1)))
    with pytest.raises(ValueError, match="symmetric"):
        GaussianBelief(jnp.zeros(2), jnp.asarray([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianBelief(jnp.zeros(2), jnp.asarray([[1.0, 0.0], [0.0, -1.0]]))


def test_common_information_policy_cannot_receive_other_latent_types_or_raw_keys():
    result = solve_common_information_game(_coordination_game(), _first_selector())

    assert int(result.policy.action("row", 0, 0, 0)) == 0
    assert int(result.policy.action("row", 0, 0, 1)) == 0
    assert int(result.policy.action("column", 0, 0, 0)) == 0
    np.testing.assert_array_equal(result.policy.player_prescription("column", 0, 0), [0])
    with pytest.raises(TypeError):
        result.policy.action("column", 0, 0, 0, latent_private_state=1)
    with pytest.raises(TypeError):
        result.policy.action("column", 0, 0, 0, key=jnp.asarray([0, 1]))


def test_bayes_evidence_is_normalized_on_positive_probability_support():
    result = solve_common_information_game(_bayes_game(), _first_selector())

    np.testing.assert_allclose(result.bayes_normalizers[0, 0], [0.5, 0.5])
    np.testing.assert_array_equal(result.bayes_support[0, 0], [True, True])
    np.testing.assert_allclose(
        result.bayes_posteriors[0, 0], [[0.5, 0.5], [0.0, 1.0]], atol=1.0e-7
    )
    np.testing.assert_allclose(
        jnp.sum(result.bayes_posteriors[0, 0], axis=-1), [1.0, 1.0]
    )
    np.testing.assert_allclose(
        result.bayes_evidence.normalization_residuals[0, 0], 0.0, atol=1.0e-7
    )
    assert bool(jnp.all(result.bayes_evidence.common_belief_consistent))


def test_zero_probability_observation_has_explicit_unsupported_zero_posterior():
    result = solve_common_information_game(
        _bayes_game(zero_probability_observation=True), _first_selector()
    )

    np.testing.assert_allclose(result.bayes_normalizers[0, 0], [1.0, 0.0])
    np.testing.assert_array_equal(result.bayes_support[0, 0], [True, False])
    np.testing.assert_array_equal(result.bayes_posteriors[0, 0, 0], [1.0, 0.0])
    np.testing.assert_array_equal(result.bayes_posteriors[0, 0, 1], [0.0, 0.0])
    assert result.bayes_evidence.normalization_residuals[0, 0, 1] == 0.0
    assert result.bayes_evidence.common_belief_residuals[0, 0, 1] == 0.0
    assert bool(result.bayes_evidence.common_belief_consistent[0, 0, 1])
    np.testing.assert_array_equal(
        result.bayes_evidence.private_type_support[0][0, 0], [True, False]
    )


def test_selector_identity_and_branch_change_the_selected_equilibrium():
    first = solve_common_information_game(_coordination_game(), _first_selector())
    last = solve_common_information_game(_coordination_game(), _last_selector())

    assert first.selector_id == "lexicographic-first"
    assert last.selector_id == "lexicographic-last"
    assert first.branch_id != last.branch_id
    assert int(first.policy.action("row", 0, 0, 0)) == 0
    assert int(last.policy.action("row", 0, 0, 0)) == 1
    assert int(first.policy.action("column", 0, 0, 0)) == 0
    assert int(last.policy.action("column", 0, 0, 0)) == 1
    assert int(first.equilibrium_candidate_counts[0, 0]) == 2
    assert int(last.equilibrium_candidate_counts[0, 0]) == 2


def test_tiny_common_information_game_has_analytic_backward_values():
    result = solve_common_information_game(_two_stage_dominance_game(), _last_selector())

    assert bool(result.valid)
    assert result.result_label == "COMMON_INFORMATION_MARKOV_PERFECT_CANDIDATE"
    np.testing.assert_array_equal(result.selected_profile_indices, [[0], [0]])
    np.testing.assert_array_equal(result.equilibrium_candidate_counts, [[1], [1]])
    np.testing.assert_allclose(
        result.value_tables[:, 0, 0],
        [[9.0, 13.0], [6.0, 9.0], [5.0, 7.0]],
    )
    assert int(result.policy.action("first", 0, 0, 0)) == 0
    assert int(result.policy.action("second", 1, 0, 0)) == 0


def test_prescription_enumeration_rejects_capacity_above_the_declared_bound():
    with pytest.raises(ValueError, match="capacity 8 exceeds"):
        solve_common_information_game(
            _coordination_game(),
            _first_selector(),
            maximum_prescription_profiles=7,
        )
