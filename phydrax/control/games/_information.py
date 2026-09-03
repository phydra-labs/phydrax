#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit information values and finite common-information games."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from operator import index

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


_COMMON_INFORMATION_METHOD_ID = "finite-pure-common-information-backward-induction-v1"
_COMMON_INFORMATION_RESULT_LABEL = "COMMON_INFORMATION_MARKOV_PERFECT_CANDIDATE"
_DEFAULT_MAXIMUM_PRESCRIPTION_PROFILES = 65_536


def _name(value: str, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _count(value, *, owner: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{owner} must be a positive integer, not a boolean.")
    result = index(value)
    if result <= 0:
        raise ValueError(f"{owner} must be positive.")
    return result


def _tolerance(value, *, owner: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return result


def _real_array(value: ArrayLike, *, owner: str) -> Array:
    result = jnp.asarray(value)
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must be real-valued.")
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    return result


def _host(value: Array, /) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _probability_table(
    value: ArrayLike,
    shape: tuple[int, ...],
    *,
    owner: str,
    tolerance: float,
) -> Array:
    result = _real_array(value, owner=owner)
    if result.shape != shape:
        raise ValueError(f"{owner} must have shape {shape}; got {result.shape}.")
    host = _host(result)
    if not np.all(np.isfinite(host)) or np.any(host < 0.0):
        raise ValueError(f"{owner} must contain finite nonnegative probabilities.")
    if not np.allclose(np.sum(host, axis=-1), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(f"Every row of {owner} must sum to one.")
    return result


class FullStateInformation(StrictModule):
    """A policy information value that exposes exactly the supplied full state."""

    information_id: str = eqx.field(static=True)
    timing: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        information_id: str = "full-state",
        timing: str = "pre-action",
    ):
        self.information_id = _name(information_id, owner="information_id")
        self.timing = _name(timing, owner="timing")

    def policy_input(self, state, /):
        """Return the declared full state without copying or transforming it."""
        return state


class CentralizedObservationInformation(StrictModule):
    """A policy information value containing one declared centralized observation."""

    information_id: str = eqx.field(static=True)
    timing: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        information_id: str = "centralized-observation",
        timing: str = "pre-action",
    ):
        self.information_id = _name(information_id, owner="information_id")
        self.timing = _name(timing, owner="timing")

    def policy_input(self, observation, /):
        """Return only the observation explicitly supplied to the policy."""
        return observation


class GaussianBelief(StrictModule):
    """Finite-dimensional Gaussian belief with explicit semidefinite covariance."""

    mean: Array
    covariance: Array
    dimension: int = eqx.field(static=True)
    belief_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        belief_id: str = "gaussian-belief",
        validation_tolerance: float = 1.0e-10,
    ):
        tolerance = _tolerance(validation_tolerance, owner="validation_tolerance")
        mean_array = _real_array(mean, owner="mean")
        if mean_array.ndim != 1 or mean_array.shape[0] < 1:
            raise ValueError("mean must be a nonempty vector.")
        covariance_array = _real_array(covariance, owner="covariance")
        dimension = int(mean_array.shape[0])
        if covariance_array.shape != (dimension, dimension):
            raise ValueError(
                "covariance must have shape "
                f"({dimension}, {dimension}); got {covariance_array.shape}."
            )
        mean_host = _host(mean_array)
        covariance_host = _host(covariance_array)
        if not np.all(np.isfinite(mean_host)):
            raise ValueError("mean must be finite.")
        if not np.all(np.isfinite(covariance_host)) or not np.allclose(
            covariance_host,
            covariance_host.T,
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError("covariance must be finite and symmetric.")
        if np.min(np.linalg.eigvalsh(covariance_host)) < -tolerance:
            raise ValueError("covariance must be positive semidefinite.")
        self.mean = mean_array
        self.covariance = covariance_array
        self.dimension = dimension
        self.belief_id = _name(belief_id, owner="belief_id")


class FiniteStateCommonInformationGame(StrictModule):
    """A finite-horizon Bayesian game on a declared finite common state.

    Joint private types and joint actions use lexicographic Cartesian-product
    order induced by ``private_type_counts`` and ``action_counts``. Costs have a
    trailing player payload axis; players are never interpreted as independent
    cases. The transition tables have shapes

    - type transitions: ``(H, C, T, A, T)``;
    - public-observation transitions: ``(H, C, T, A, T, C)``.

    Thus the public-observation probability may depend on current type, joint
    action, and next type. ``common_beliefs[k, c]`` declares the common belief
    represented by finite common state ``c`` at stage ``k``.
    """

    common_beliefs: Array
    type_transition_probabilities: Array
    observation_transition_probabilities: Array
    stage_costs: Array
    terminal_costs: Array
    player_ids: tuple[str, ...] = eqx.field(static=True)
    private_type_counts: tuple[int, ...] = eqx.field(static=True)
    action_counts: tuple[int, ...] = eqx.field(static=True)
    joint_private_types: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    joint_actions: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    num_joint_private_types: int = eqx.field(static=True)
    num_joint_actions: int = eqx.field(static=True)
    num_common_states: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    probability_tolerance: float = eqx.field(static=True)
    game_id: str = eqx.field(static=True)

    def __init__(
        self,
        player_ids: Sequence[str],
        private_type_counts: Sequence[int],
        action_counts: Sequence[int],
        common_beliefs: ArrayLike,
        type_transition_probabilities: ArrayLike,
        observation_transition_probabilities: ArrayLike,
        stage_costs: ArrayLike,
        terminal_costs: ArrayLike,
        /,
        *,
        game_id: str = "finite-state-common-information-game",
        probability_tolerance: float = 1.0e-7,
    ):
        if isinstance(player_ids, str):
            raise TypeError("player_ids must be a sequence of identifiers.")
        players = tuple(player_ids)
        if not players or any(
            not isinstance(player, str) or not player for player in players
        ):
            raise ValueError("player_ids must contain non-empty strings.")
        if len(set(players)) != len(players):
            raise ValueError("player_ids must be unique.")
        if isinstance(private_type_counts, (str, bytes)) or isinstance(
            action_counts, (str, bytes)
        ):
            raise TypeError("private_type_counts and action_counts must be sequences.")
        type_counts = tuple(
            _count(value, owner=f"private_type_counts[{position}]")
            for position, value in enumerate(private_type_counts)
        )
        actions = tuple(
            _count(value, owner=f"action_counts[{position}]")
            for position, value in enumerate(action_counts)
        )
        if len(type_counts) != len(players) or len(actions) != len(players):
            raise ValueError(
                "private_type_counts and action_counts must provide one entry per player."
            )

        tolerance = _tolerance(probability_tolerance, owner="probability_tolerance")
        beliefs = _real_array(common_beliefs, owner="common_beliefs")
        if beliefs.ndim != 3 or beliefs.shape[0] < 2 or beliefs.shape[1] < 1:
            raise ValueError(
                "common_beliefs must have shape "
                "(horizon + 1, common states, joint types)."
            )
        horizon = int(beliefs.shape[0] - 1)
        common_states = int(beliefs.shape[1])
        joint_types = int(np.prod(type_counts, dtype=object))
        joint_action_count = int(np.prod(actions, dtype=object))
        beliefs = _probability_table(
            beliefs,
            (horizon + 1, common_states, joint_types),
            owner="common_beliefs",
            tolerance=tolerance,
        )
        type_probabilities = _probability_table(
            type_transition_probabilities,
            (horizon, common_states, joint_types, joint_action_count, joint_types),
            owner="type_transition_probabilities",
            tolerance=tolerance,
        )
        observation_probabilities = _probability_table(
            observation_transition_probabilities,
            (
                horizon,
                common_states,
                joint_types,
                joint_action_count,
                joint_types,
                common_states,
            ),
            owner="observation_transition_probabilities",
            tolerance=tolerance,
        )
        stage = _real_array(stage_costs, owner="stage_costs")
        expected_stage = (
            horizon,
            common_states,
            joint_types,
            joint_action_count,
            len(players),
        )
        if stage.shape != expected_stage:
            raise ValueError(
                f"stage_costs must have shape {expected_stage}; got {stage.shape}."
            )
        terminal = _real_array(terminal_costs, owner="terminal_costs")
        expected_terminal = (common_states, joint_types, len(players))
        if terminal.shape != expected_terminal:
            raise ValueError(
                "terminal_costs must have shape "
                f"{expected_terminal}; got {terminal.shape}."
            )
        if not np.all(np.isfinite(_host(stage))) or not np.all(
            np.isfinite(_host(terminal))
        ):
            raise ValueError("stage_costs and terminal_costs must be finite.")

        self.common_beliefs = beliefs
        self.type_transition_probabilities = type_probabilities
        self.observation_transition_probabilities = observation_probabilities
        self.stage_costs = stage
        self.terminal_costs = terminal
        self.player_ids = players
        self.private_type_counts = type_counts
        self.action_counts = actions
        self.joint_private_types = tuple(product(*(range(size) for size in type_counts)))
        self.joint_actions = tuple(product(*(range(size) for size in actions)))
        self.num_players = len(players)
        self.num_joint_private_types = joint_types
        self.num_joint_actions = joint_action_count
        self.num_common_states = common_states
        self.horizon = horizon
        self.probability_tolerance = tolerance
        self.game_id = _name(game_id, owner="game_id")


class CommonInformationStageEquilibria(StrictModule):
    """Pure Bayesian stage-Nash candidates presented to a selector."""

    prescriptions: tuple[Array, ...]
    conditional_costs: tuple[Array, ...]
    incentive_gains: tuple[Array, ...]
    private_type_probabilities: tuple[Array, ...]
    private_type_support: tuple[Array, ...]
    expected_costs: Array
    nash_indices: Array
    equilibrium_indices: Array
    bayes_consistency_residuals: Array
    bayes_consistent: Array
    stage: int = eqx.field(static=True)
    common_state: int = eqx.field(static=True)
    candidate_count: int = eqx.field(static=True)


class CommonInformationEquilibriumSelector(StrictModule):
    """Identified deterministic selector for finite pure stage equilibria.

    The callback receives only a finite, already ordered candidate table and is
    not given a random key. It must return one entry of ``equilibrium_indices``.
    """

    selection: Callable[[CommonInformationStageEquilibria], int] = eqx.field(static=True)
    selector_id: str = eqx.field(static=True)

    def __init__(
        self,
        selection: Callable[[CommonInformationStageEquilibria], int],
        /,
        *,
        selector_id: str,
    ):
        if not callable(selection):
            raise TypeError("selection must be callable.")
        self.selection = selection
        self.selector_id = _name(selector_id, owner="selector_id")

    def select_equilibrium(self, equilibria: CommonInformationStageEquilibria, /) -> int:
        selected = self.selection(equilibria)
        if isinstance(selected, bool):
            raise TypeError(
                "The equilibrium selector must return an integer profile index."
            )
        selected_index = index(selected)
        available = _host(equilibria.equilibrium_indices)
        if not np.any(available == selected_index):
            raise ValueError(
                "The equilibrium selector returned a profile that is not an available "
                "Bayesian Nash common-information candidate."
            )
        return selected_index


class CommonInformationPolicy(StrictModule):
    """Pure prescriptions indexed only by public state and own private type."""

    prescriptions: tuple[Array, ...]
    player_ids: tuple[str, ...] = eqx.field(static=True)
    private_type_counts: tuple[int, ...] = eqx.field(static=True)
    action_counts: tuple[int, ...] = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    num_common_states: int = eqx.field(static=True)

    def __init__(
        self,
        prescriptions: tuple[Array, ...],
        player_ids: tuple[str, ...],
        private_type_counts: tuple[int, ...],
        action_counts: tuple[int, ...],
        horizon: int,
        num_common_states: int,
        /,
    ):
        if len(prescriptions) != len(player_ids):
            raise ValueError("prescriptions must provide one table per player.")
        tables = tuple(jnp.asarray(table, dtype=jnp.int32) for table in prescriptions)
        for player, (table, private_types, actions) in enumerate(
            zip(tables, private_type_counts, action_counts, strict=True)
        ):
            expected = (horizon, num_common_states, private_types)
            if table.shape != expected:
                raise ValueError(
                    f"Player {player_ids[player]!r} prescription must have shape "
                    f"{expected}; got {table.shape}."
                )
            host = _host(table)
            if np.any(host < 0) or np.any(host >= actions):
                raise ValueError(
                    "Prescription actions must lie in the declared action set."
                )
        self.prescriptions = tables
        self.player_ids = player_ids
        self.private_type_counts = private_type_counts
        self.action_counts = action_counts
        self.horizon = horizon
        self.num_common_states = num_common_states

    def player_prescription(
        self, player_id: str, stage: int, common_state: int, /
    ) -> Array:
        """Return one player's action table over that player's declared type only."""
        player = self._player_index(player_id)
        stage_index = self._bounded_index(stage, self.horizon, owner="stage")
        state_index = self._bounded_index(
            common_state, self.num_common_states, owner="common_state"
        )
        return self.prescriptions[player][stage_index, state_index]

    def action(
        self,
        player_id: str,
        stage: int,
        common_state: int,
        private_type: int,
        /,
    ) -> Array:
        """Evaluate a prescription without accepting other latent types or keys."""
        player = self._player_index(player_id)
        stage_index = self._bounded_index(stage, self.horizon, owner="stage")
        state_index = self._bounded_index(
            common_state, self.num_common_states, owner="common_state"
        )
        type_index = self._bounded_index(
            private_type,
            self.private_type_counts[player],
            owner="private_type",
        )
        return self.prescriptions[player][stage_index, state_index, type_index]

    def _player_index(self, player_id: str, /) -> int:
        if not isinstance(player_id, str):
            raise TypeError("player_id must be a declared player identifier.")
        if player_id not in self.player_ids:
            raise ValueError(f"Unknown player_id {player_id!r}.")
        return self.player_ids.index(player_id)

    @staticmethod
    def _bounded_index(value, bound: int, *, owner: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{owner} must be an integer, not a boolean.")
        result = index(value)
        if result < 0 or result >= bound:
            raise IndexError(f"{owner} must lie in [0, {bound}).")
        return result


class CommonInformationBayesEvidence(StrictModule):
    """Selected-branch Bayes normalizers, supports, and posterior evidence."""

    normalizers: Array
    support: Array
    posteriors: Array
    normalization_residuals: Array
    common_belief_residuals: Array
    common_belief_consistent: Array
    private_type_probabilities: tuple[Array, ...]
    private_type_support: tuple[Array, ...]


class CommonInformationGameResult(StrictModule):
    """Finite pure-prescription common-information Markov-perfect candidate."""

    game: FiniteStateCommonInformationGame
    policy: CommonInformationPolicy
    values: Array
    bayes_evidence: CommonInformationBayesEvidence
    selected_profile_indices: Array
    nash_candidate_counts: Array
    equilibrium_candidate_counts: Array
    valid: Array
    result_label: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    game_id: str = eqx.field(static=True)
    selector_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    maximum_prescription_profiles: int = eqx.field(static=True)
    incentive_tolerance: float = eqx.field(static=True)
    bayes_tolerance: float = eqx.field(static=True)

    @property
    def prescription_profile(self) -> tuple[Array, ...]:
        return self.policy.prescriptions

    @property
    def value_tables(self) -> Array:
        return self.values

    @property
    def bayes_normalizers(self) -> Array:
        return self.bayes_evidence.normalizers

    @property
    def bayes_support(self) -> Array:
        return self.bayes_evidence.support

    @property
    def bayes_posteriors(self) -> Array:
        return self.bayes_evidence.posteriors


def _flat_index(values: tuple[int, ...], sizes: tuple[int, ...], /) -> int:
    result = 0
    for value, size in zip(values, sizes, strict=True):
        result = result * size + value
    return result


def _prescription_catalog(
    game: FiniteStateCommonInformationGame,
    maximum_prescription_profiles: int,
    /,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    per_player_counts = tuple(
        action_count**type_count
        for type_count, action_count in zip(
            game.private_type_counts, game.action_counts, strict=True
        )
    )
    capacity = int(np.prod(per_player_counts, dtype=object))
    if capacity > maximum_prescription_profiles:
        raise ValueError(
            "Pure prescription profile capacity "
            f"{capacity} exceeds maximum_prescription_profiles="
            f"{maximum_prescription_profiles}."
        )
    per_player = tuple(
        tuple(product(range(action_count), repeat=type_count))
        for type_count, action_count in zip(
            game.private_type_counts, game.action_counts, strict=True
        )
    )
    profiles = tuple(product(*per_player))
    prescriptions = tuple(
        np.asarray(
            [profile[player] for profile in profiles],
            dtype=np.int32,
        )
        for player in range(game.num_players)
    )
    profile_actions = np.empty((capacity, game.num_joint_private_types), dtype=np.int32)
    for profile_index in range(capacity):
        for type_index, joint_type in enumerate(game.joint_private_types):
            actions = tuple(
                int(prescriptions[player][profile_index, joint_type[player]])
                for player in range(game.num_players)
            )
            profile_actions[profile_index, type_index] = _flat_index(
                actions, game.action_counts
            )
    return prescriptions, profile_actions


def _bayes_update(
    belief: np.ndarray,
    type_probabilities: np.ndarray,
    observation_probabilities: np.ndarray,
    profile_actions: np.ndarray,
    next_beliefs: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joint_types = belief.shape[0]
    common_states = observation_probabilities.shape[-1]
    mass = np.zeros((joint_types, common_states), dtype=np.result_type(belief, float))
    for current_type in range(joint_types):
        action = int(profile_actions[current_type])
        weighted = (
            belief[current_type]
            * type_probabilities[current_type, action, :, None]
            * observation_probabilities[current_type, action]
        )
        mass += weighted
    normalizers = np.sum(mass, axis=0)
    support = normalizers > 0.0
    posteriors = np.zeros_like(mass.T)
    posteriors[support] = mass.T[support] / normalizers[support, None]
    normalization_residuals = np.zeros(common_states, dtype=mass.dtype)
    normalization_residuals[support] = np.abs(np.sum(posteriors[support], axis=-1) - 1.0)
    belief_residuals = np.zeros(common_states, dtype=mass.dtype)
    if np.any(support):
        belief_residuals[support] = np.max(
            np.abs(posteriors[support] - next_beliefs[support]), axis=-1
        )
    return normalizers, support, posteriors, normalization_residuals, belief_residuals


def _stage_equilibria(
    game: FiniteStateCommonInformationGame,
    stage: int,
    common_state: int,
    q_values: np.ndarray,
    prescriptions: tuple[np.ndarray, ...],
    profile_actions: np.ndarray,
    incentive_tolerance: float,
    bayes_tolerance: float,
    /,
) -> CommonInformationStageEquilibria:
    belief = _host(game.common_beliefs[stage, common_state])
    candidate_count = profile_actions.shape[0]
    players = game.num_players
    expected_costs = np.empty((candidate_count, players), dtype=q_values.dtype)
    conditional_costs = tuple(
        np.zeros((candidate_count, type_count), dtype=q_values.dtype)
        for type_count in game.private_type_counts
    )
    incentive_gains = tuple(
        np.zeros((candidate_count, type_count, action_count), dtype=q_values.dtype)
        for type_count, action_count in zip(
            game.private_type_counts, game.action_counts, strict=True
        )
    )
    private_probabilities = []
    private_support = []
    for player, type_count in enumerate(game.private_type_counts):
        probabilities = np.zeros(type_count, dtype=belief.dtype)
        for joint_type_index, joint_type in enumerate(game.joint_private_types):
            probabilities[joint_type[player]] += belief[joint_type_index]
        private_probabilities.append(probabilities)
        private_support.append(probabilities > 0.0)

    nash = np.ones(candidate_count, dtype=bool)
    bayes_consistency = np.zeros((candidate_count, game.num_common_states), dtype=float)
    bayes_consistent = np.ones(candidate_count, dtype=bool)
    type_probabilities = _host(game.type_transition_probabilities[stage, common_state])
    observation_probabilities = _host(
        game.observation_transition_probabilities[stage, common_state]
    )
    next_beliefs = _host(game.common_beliefs[stage + 1])

    for profile_index in range(candidate_count):
        actual_actions = profile_actions[profile_index]
        expected_costs[profile_index] = np.sum(
            belief[:, None]
            * q_values[np.arange(game.num_joint_private_types), actual_actions],
            axis=0,
        )
        for player, type_count in enumerate(game.private_type_counts):
            for private_type in range(type_count):
                probability = private_probabilities[player][private_type]
                if probability == 0.0:
                    continue
                chosen_cost = 0.0
                deviation_costs = np.zeros(
                    game.action_counts[player], dtype=q_values.dtype
                )
                for joint_type_index, joint_type in enumerate(game.joint_private_types):
                    if joint_type[player] != private_type:
                        continue
                    conditional_weight = belief[joint_type_index] / probability
                    chosen_cost += (
                        conditional_weight
                        * q_values[
                            joint_type_index, actual_actions[joint_type_index], player
                        ]
                    )
                    other_actions = [
                        int(prescriptions[other][profile_index, joint_type[other]])
                        for other in range(players)
                    ]
                    for deviation in range(game.action_counts[player]):
                        other_actions[player] = deviation
                        deviation_index = _flat_index(
                            tuple(other_actions), game.action_counts
                        )
                        deviation_costs[deviation] += (
                            conditional_weight
                            * q_values[joint_type_index, deviation_index, player]
                        )
                conditional_costs[player][profile_index, private_type] = chosen_cost
                gains = chosen_cost - deviation_costs
                incentive_gains[player][profile_index, private_type] = gains
                if np.max(gains) > incentive_tolerance:
                    nash[profile_index] = False

        payload = _bayes_update(
            belief,
            type_probabilities,
            observation_probabilities,
            actual_actions,
            next_beliefs,
        )
        support = payload[1]
        bayes_consistency[profile_index] = payload[4]
        bayes_consistent[profile_index] = bool(
            np.all(payload[4][support] <= bayes_tolerance)
        )

    nash_indices = np.flatnonzero(nash).astype(np.int32)
    equilibrium_indices = np.flatnonzero(nash & bayes_consistent).astype(np.int32)
    stage_equilibria = CommonInformationStageEquilibria(
        prescriptions=tuple(jnp.asarray(value) for value in prescriptions),
        conditional_costs=tuple(jnp.asarray(value) for value in conditional_costs),
        incentive_gains=tuple(jnp.asarray(value) for value in incentive_gains),
        private_type_probabilities=tuple(
            jnp.asarray(value) for value in private_probabilities
        ),
        private_type_support=tuple(jnp.asarray(value) for value in private_support),
        expected_costs=jnp.asarray(expected_costs),
        nash_indices=jnp.asarray(nash_indices),
        equilibrium_indices=jnp.asarray(equilibrium_indices),
        bayes_consistency_residuals=jnp.asarray(bayes_consistency),
        bayes_consistent=jnp.asarray(bayes_consistent),
        stage=stage,
        common_state=common_state,
        candidate_count=candidate_count,
    )
    return stage_equilibria


def solve_common_information_game(
    game: FiniteStateCommonInformationGame,
    selector: CommonInformationEquilibriumSelector,
    /,
    *,
    maximum_prescription_profiles: int = _DEFAULT_MAXIMUM_PRESCRIPTION_PROFILES,
    incentive_tolerance: float = 1.0e-10,
    bayes_tolerance: float = 1.0e-7,
) -> CommonInformationGameResult:
    """Solve a finite game by pure-prescription Bayesian backward induction.

    Only pure prescriptions are enumerated. Every selected stage profile must be
    both a simultaneous Bayesian Nash equilibrium on supported private types and
    Bayes-consistent with the next declared finite common state. A missing pure
    candidate is reported as an error rather than replaced by a mixed strategy,
    regularized game, or approximate fallback.
    """
    if not isinstance(game, FiniteStateCommonInformationGame):
        raise TypeError("game must be a FiniteStateCommonInformationGame.")
    if not isinstance(selector, CommonInformationEquilibriumSelector):
        raise TypeError("selector must be a CommonInformationEquilibriumSelector.")
    maximum_profiles = _count(
        maximum_prescription_profiles, owner="maximum_prescription_profiles"
    )
    incentive_tolerance_ = _tolerance(incentive_tolerance, owner="incentive_tolerance")
    bayes_tolerance_ = _tolerance(bayes_tolerance, owner="bayes_tolerance")
    prescriptions, profile_actions = _prescription_catalog(game, maximum_profiles)

    costs = np.result_type(_host(game.stage_costs), _host(game.terminal_costs), float)
    values = np.empty(
        (
            game.horizon + 1,
            game.num_common_states,
            game.num_joint_private_types,
            game.num_players,
        ),
        dtype=costs,
    )
    values[-1] = _host(game.terminal_costs)
    selected_indices = np.empty((game.horizon, game.num_common_states), dtype=np.int32)
    nash_counts = np.empty_like(selected_indices)
    equilibrium_counts = np.empty_like(selected_indices)
    policy_tables = tuple(
        np.empty((game.horizon, game.num_common_states, type_count), dtype=np.int32)
        for type_count in game.private_type_counts
    )
    normalizers = np.empty(
        (game.horizon, game.num_common_states, game.num_common_states), dtype=costs
    )
    support = np.empty(normalizers.shape, dtype=bool)
    posteriors = np.empty(
        normalizers.shape + (game.num_joint_private_types,), dtype=costs
    )
    normalization_residuals = np.empty(normalizers.shape, dtype=costs)
    common_belief_residuals = np.empty(normalizers.shape, dtype=costs)
    common_belief_consistent = np.empty(normalizers.shape, dtype=bool)
    private_probabilities = tuple(
        np.empty((game.horizon, game.num_common_states, type_count), dtype=costs)
        for type_count in game.private_type_counts
    )
    private_support = tuple(
        np.empty((game.horizon, game.num_common_states, type_count), dtype=bool)
        for type_count in game.private_type_counts
    )

    stage_costs = _host(game.stage_costs)
    type_transitions = _host(game.type_transition_probabilities)
    observation_transitions = _host(game.observation_transition_probabilities)
    for stage in range(game.horizon - 1, -1, -1):
        for common_state in range(game.num_common_states):
            q_values = np.empty(
                (
                    game.num_joint_private_types,
                    game.num_joint_actions,
                    game.num_players,
                ),
                dtype=costs,
            )
            for current_type in range(game.num_joint_private_types):
                for joint_action in range(game.num_joint_actions):
                    continuation = np.zeros(game.num_players, dtype=costs)
                    for next_type in range(game.num_joint_private_types):
                        type_probability = type_transitions[
                            stage,
                            common_state,
                            current_type,
                            joint_action,
                            next_type,
                        ]
                        if type_probability == 0.0:
                            continue
                        for next_state in range(game.num_common_states):
                            probability = (
                                type_probability
                                * observation_transitions[
                                    stage,
                                    common_state,
                                    current_type,
                                    joint_action,
                                    next_type,
                                    next_state,
                                ]
                            )
                            continuation += (
                                probability * values[stage + 1, next_state, next_type]
                            )
                    q_values[current_type, joint_action] = (
                        stage_costs[stage, common_state, current_type, joint_action]
                        + continuation
                    )
            if not np.all(np.isfinite(q_values)):
                raise FloatingPointError(
                    "Nonfinite backward values at "
                    f"stage {stage}, common state {common_state}."
                )

            equilibria = _stage_equilibria(
                game,
                stage,
                common_state,
                q_values,
                prescriptions,
                profile_actions,
                incentive_tolerance_,
                bayes_tolerance_,
            )
            nash_count = int(equilibria.nash_indices.shape[0])
            equilibrium_count = int(equilibria.equilibrium_indices.shape[0])
            if equilibrium_count == 0:
                if nash_count == 0:
                    reason = "no pure simultaneous Bayesian Nash equilibrium exists"
                else:
                    reason = (
                        "no pure Bayesian Nash equilibrium is Bayes-consistent with "
                        "the declared next common-state beliefs"
                    )
                raise ValueError(
                    f"At stage {stage}, common state {common_state}, {reason}."
                )
            selected = selector.select_equilibrium(equilibria)
            selected_indices[stage, common_state] = selected
            nash_counts[stage, common_state] = nash_count
            equilibrium_counts[stage, common_state] = equilibrium_count
            actual_actions = profile_actions[selected]
            values[stage, common_state] = q_values[
                np.arange(game.num_joint_private_types), actual_actions
            ]
            for player in range(game.num_players):
                policy_tables[player][stage, common_state] = prescriptions[player][
                    selected
                ]
                private_probabilities[player][stage, common_state] = _host(
                    equilibria.private_type_probabilities[player]
                )
                private_support[player][stage, common_state] = _host(
                    equilibria.private_type_support[player]
                )

            payload = _bayes_update(
                _host(game.common_beliefs[stage, common_state]),
                type_transitions[stage, common_state],
                observation_transitions[stage, common_state],
                actual_actions,
                _host(game.common_beliefs[stage + 1]),
            )
            normalizers[stage, common_state] = payload[0]
            support[stage, common_state] = payload[1]
            posteriors[stage, common_state] = payload[2]
            normalization_residuals[stage, common_state] = payload[3]
            common_belief_residuals[stage, common_state] = payload[4]
            common_belief_consistent[stage, common_state] = (~payload[1]) | (
                payload[4] <= bayes_tolerance_
            )

    branch_id = "common-information-branch:" + canonical_fingerprint(
        {
            "game_id": game.game_id,
            "selected_profile_indices": selected_indices.tolist(),
        }
    )
    policy = CommonInformationPolicy(
        tuple(jnp.asarray(value) for value in policy_tables),
        game.player_ids,
        game.private_type_counts,
        game.action_counts,
        game.horizon,
        game.num_common_states,
    )
    evidence = CommonInformationBayesEvidence(
        normalizers=jnp.asarray(normalizers),
        support=jnp.asarray(support),
        posteriors=jnp.asarray(posteriors),
        normalization_residuals=jnp.asarray(normalization_residuals),
        common_belief_residuals=jnp.asarray(common_belief_residuals),
        common_belief_consistent=jnp.asarray(common_belief_consistent),
        private_type_probabilities=tuple(
            jnp.asarray(value) for value in private_probabilities
        ),
        private_type_support=tuple(jnp.asarray(value) for value in private_support),
    )
    return CommonInformationGameResult(
        game=game,
        policy=policy,
        values=jnp.asarray(values),
        bayes_evidence=evidence,
        selected_profile_indices=jnp.asarray(selected_indices),
        nash_candidate_counts=jnp.asarray(nash_counts),
        equilibrium_candidate_counts=jnp.asarray(equilibrium_counts),
        valid=jnp.asarray(True),
        result_label=_COMMON_INFORMATION_RESULT_LABEL,
        method_id=_COMMON_INFORMATION_METHOD_ID,
        game_id=game.game_id,
        selector_id=selector.selector_id,
        branch_id=branch_id,
        maximum_prescription_profiles=maximum_profiles,
        incentive_tolerance=incentive_tolerance_,
        bayes_tolerance=bayes_tolerance_,
    )


__all__ = [
    "CentralizedObservationInformation",
    "CommonInformationBayesEvidence",
    "CommonInformationEquilibriumSelector",
    "CommonInformationGameResult",
    "CommonInformationPolicy",
    "CommonInformationStageEquilibria",
    "FiniteStateCommonInformationGame",
    "FullStateInformation",
    "GaussianBelief",
    "solve_common_information_game",
]
