"""Select a finite-state common-information candidate by backward induction."""

from itertools import product

import jax.numpy as jnp

import phydrax as phx


PLAYER_IDS = ("row", "column")
PRIVATE_TYPE_COUNTS = (2, 2)
ACTION_COUNTS = (2, 2)
HORIZON = 2
COMMON_STATE_COUNT = 2
joint_types = tuple(product(*(range(count) for count in PRIVATE_TYPE_COUNTS)))
joint_actions = tuple(product(*(range(count) for count in ACTION_COUNTS)))

# Public state 0 means the private types agree; public state 1 means they differ.
# It does not reveal either player's private type.
common_beliefs = jnp.zeros((HORIZON + 1, COMMON_STATE_COUNT, len(joint_types)))
common_beliefs = common_beliefs.at[:, 0, (0, 3)].set(0.5)
common_beliefs = common_beliefs.at[:, 1, (1, 2)].set(0.5)

type_transitions = jnp.zeros(
    (
        HORIZON,
        COMMON_STATE_COUNT,
        len(joint_types),
        len(joint_actions),
        len(joint_types),
    )
)
for private_type_index in range(len(joint_types)):
    type_transitions = type_transitions.at[
        :, :, private_type_index, :, private_type_index
    ].set(1.0)

# The next common observation preserves the declared public agree/differ state.
observation_transitions = jnp.zeros(type_transitions.shape + (COMMON_STATE_COUNT,))
for common_state in range(COMMON_STATE_COUNT):
    observation_transitions = observation_transitions.at[
        :, common_state, ..., common_state
    ].set(1.0)

# Each player has a strict private-type-matching action. Costs depend on the joint
# latent type only while constructing the finite Bayesian game, never while a
# player evaluates the resulting decentralized prescription.
stage_costs = jnp.zeros(
    (
        HORIZON,
        COMMON_STATE_COUNT,
        len(joint_types),
        len(joint_actions),
        len(PLAYER_IDS),
    )
)
for private_type_index, private_types in enumerate(joint_types):
    for action_index, actions in enumerate(joint_actions):
        costs = jnp.asarray(
            [float(actions[player] != private_types[player]) for player in range(2)]
        )
        stage_costs = stage_costs.at[:, :, private_type_index, action_index].set(costs)

game = phx.control.games.FiniteStateCommonInformationGame(
    PLAYER_IDS,
    PRIVATE_TYPE_COUNTS,
    ACTION_COUNTS,
    common_beliefs,
    type_transitions,
    observation_transitions,
    stage_costs,
    jnp.zeros((COMMON_STATE_COUNT, len(joint_types), len(PLAYER_IDS))),
    game_id="agree-differ-common-information",
)
selector = phx.control.games.CommonInformationEquilibriumSelector(
    lambda equilibria: int(equilibria.equilibrium_indices[0]),
    selector_id="lexicographic-first-bayes-consistent-branch",
)
result = phx.control.games.solve_common_information_game(
    game,
    selector,
    maximum_prescription_profiles=16,
)

if not bool(result.valid):
    raise RuntimeError("common-information backward selection returned invalid evidence")
if not bool(jnp.all(result.bayes_evidence.common_belief_consistent)):
    raise RuntimeError("selected branch is inconsistent with the declared common beliefs")

# A policy query receives only public information and that player's own private
# type. It has no parameter through which another player's latent type can leak.
stage = 0
public_common_state = 1
row_private_type = 0
column_private_type = 1
row_action = int(
    result.policy.action(
        "row",
        stage,
        public_common_state,
        row_private_type,
    )
)
column_action = int(
    result.policy.action(
        "column",
        stage,
        public_common_state,
        column_private_type,
    )
)
if (row_action, column_action) != (row_private_type, column_private_type):
    raise RuntimeError(
        "selected private prescriptions do not match the analytic solution"
    )

print(
    {
        "result_label": result.result_label,
        "method": result.method_id,
        "selector": result.selector_id,
        "selected_branch": result.branch_id,
        "public_common_state": public_common_state,
        "row": {"own_private_type": row_private_type, "action": row_action},
        "column": {
            "own_private_type": column_private_type,
            "action": column_action,
        },
        "selected_profile_indices": result.selected_profile_indices.tolist(),
        "equilibrium_candidate_counts": result.equilibrium_candidate_counts.tolist(),
        "bayes_normalizers": result.bayes_normalizers.tolist(),
        "bayes_support": result.bayes_support.tolist(),
        "claim_boundary": "finite pure-prescription common-information candidate",
    }
)
