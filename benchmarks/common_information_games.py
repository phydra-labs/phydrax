#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import jax.numpy as jnp

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_repeated,
)
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint


_HOST_COMPILATION_REASON = (
    "finite pure-prescription enumeration and backward induction are host algorithms "
    "with no JAX lowering or executable-compilation boundary"
)


def _game(
    name: str,
    horizon: int,
    common_state_count: int,
    private_type_counts: tuple[int, ...],
    action_counts: tuple[int, ...],
):
    player_count = len(private_type_counts)
    player_ids = tuple(f"player-{player}" for player in range(player_count))
    joint_types = tuple(product(*(range(count) for count in private_type_counts)))
    joint_actions = tuple(product(*(range(count) for count in action_counts)))
    joint_type_count = len(joint_types)
    joint_action_count = len(joint_actions)

    common_beliefs = jnp.full(
        (horizon + 1, common_state_count, joint_type_count),
        1.0 / joint_type_count,
    )
    identity_types = jnp.eye(joint_type_count)
    type_transitions = jnp.broadcast_to(
        identity_types[None, None, :, None, :],
        (
            horizon,
            common_state_count,
            joint_type_count,
            joint_action_count,
            joint_type_count,
        ),
    )
    public_state_identity = jnp.eye(common_state_count)
    observation_transitions = jnp.broadcast_to(
        public_state_identity[None, :, None, None, None, :],
        (
            horizon,
            common_state_count,
            joint_type_count,
            joint_action_count,
            joint_type_count,
            common_state_count,
        ),
    )

    stage_costs = jnp.zeros(
        (
            horizon,
            common_state_count,
            joint_type_count,
            joint_action_count,
            player_count,
        )
    )
    for stage in range(horizon):
        for common_state in range(common_state_count):
            constant = 0.01 * stage + 0.001 * common_state
            for private_type_index, private_types in enumerate(joint_types):
                for action_index, actions in enumerate(joint_actions):
                    costs = jnp.asarray(
                        [
                            float(
                                actions[player]
                                != private_types[player] % action_counts[player]
                            )
                            + constant
                            for player in range(player_count)
                        ]
                    )
                    stage_costs = stage_costs.at[
                        stage,
                        common_state,
                        private_type_index,
                        action_index,
                    ].set(costs)

    terminal_costs = jnp.zeros((common_state_count, joint_type_count, player_count))
    for private_type_index, private_types in enumerate(joint_types):
        terminal_costs = terminal_costs.at[:, private_type_index].set(
            jnp.asarray([0.1 * value for value in private_types])
        )

    return phx.control.games.FiniteStateCommonInformationGame(
        player_ids,
        private_type_counts,
        action_counts,
        common_beliefs,
        type_transitions,
        observation_transitions,
        stage_costs,
        terminal_costs,
        game_id=f"benchmark:{name}",
    )


def _selector():
    return phx.control.games.CommonInformationEquilibriumSelector(
        lambda equilibria: int(equilibria.equilibrium_indices[0]),
        selector_id="benchmark-lexicographic-first",
    )


def _compiler_record() -> dict[str, Any]:
    evidence = compiler_evidence(
        None,
        None,
        source="host-finite-common-information-solver",
        unavailable_reason=_HOST_COMPILATION_REASON,
    )
    return {
        "flops": evidence.flops,
        "bytes_accessed": evidence.bytes_accessed,
        "argument_bytes": evidence.argument_bytes,
        "output_bytes": evidence.output_bytes,
        "temporary_bytes": evidence.temporary_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
        "source": evidence.source,
        "unavailable_reason": evidence.unavailable_reason,
    }


def _bayes_evidence(result) -> dict[str, Any]:
    support = result.bayes_support
    normalizers = result.bayes_normalizers
    posterior_mass = jnp.sum(result.bayes_posteriors, axis=-1)
    supported_normalizers = normalizers[support]
    posterior_mass_defect = jnp.where(support, jnp.abs(posterior_mass - 1.0), 0.0)
    return {
        "support_count": int(jnp.sum(support)),
        "total_observation_branches": int(support.size),
        "minimum_supported_normalizer": float(jnp.min(supported_normalizers)),
        "maximum_normalization_residual": float(
            jnp.max(result.bayes_evidence.normalization_residuals)
        ),
        "maximum_common_belief_residual": float(
            jnp.max(result.bayes_evidence.common_belief_residuals)
        ),
        "maximum_supported_posterior_mass_defect": float(jnp.max(posterior_mass_defect)),
        "all_common_beliefs_consistent": bool(
            jnp.all(result.bayes_evidence.common_belief_consistent)
        ),
        "normalizers": normalizers.tolist(),
        "support": support.tolist(),
    }


def _branch_evidence(result) -> dict[str, Any]:
    return {
        "branch_id": result.branch_id,
        "selector_id": result.selector_id,
        "selected_profile_indices": result.selected_profile_indices.tolist(),
        "selected_profile_fingerprint": array_tree_fingerprint(
            result.selected_profile_indices
        ),
        "nash_candidate_counts": result.nash_candidate_counts.tolist(),
        "equilibrium_candidate_counts": result.equilibrium_candidate_counts.tolist(),
        "minimum_nash_candidate_count": int(jnp.min(result.nash_candidate_counts)),
        "minimum_equilibrium_candidate_count": int(
            jnp.min(result.equilibrium_candidate_counts)
        ),
    }


def _case(
    name: str,
    horizon: int,
    common_state_count: int,
    private_type_counts: tuple[int, ...],
    action_counts: tuple[int, ...],
    maximum_prescription_profiles: int,
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    game = _game(
        name,
        horizon,
        common_state_count,
        private_type_counts,
        action_counts,
    )
    selector = _selector()
    operation = lambda: phx.control.games.solve_common_information_game(
        game,
        selector,
        maximum_prescription_profiles=maximum_prescription_profiles,
    )
    result, execution = measure_repeated(
        operation,
        warmup=warmup,
        repeats=repeats,
    )

    per_player_prescriptions = tuple(
        actions**private_types
        for private_types, actions in zip(
            private_type_counts,
            action_counts,
            strict=True,
        )
    )
    prescription_profile_count = math.prod(per_player_prescriptions)
    configuration = {
        "name": name,
        "horizon": horizon,
        "common_state_count": common_state_count,
        "private_type_counts": list(private_type_counts),
        "action_counts": list(action_counts),
        "maximum_prescription_profiles": maximum_prescription_profiles,
    }
    arrays = (
        game.common_beliefs,
        game.type_transition_probabilities,
        game.observation_transition_probabilities,
        game.stage_costs,
        game.terminal_costs,
    )
    return {
        **configuration,
        "player_count": game.num_players,
        "joint_private_type_count": game.num_joint_private_types,
        "joint_action_count": game.num_joint_actions,
        "input_fingerprint": {
            "configuration_sha256": canonical_fingerprint(configuration),
            "arrays": array_tree_fingerprint(arrays),
        },
        "lower": {
            "seconds": None,
            "source": "host-finite-common-information-solver",
            "unavailable_reason": _HOST_COMPILATION_REASON,
        },
        "compile": _compiler_record(),
        "run": execution.to_milliseconds_dict(),
        "memory": {
            "logical_input_bytes": logical_array_bytes(game),
            "logical_output_bytes": logical_array_bytes(result),
            "compiler_estimated_device_bytes": None,
            "compiler_estimate_unavailable_reason": _HOST_COMPILATION_REASON,
        },
        "work": {
            "backward_stage_common_nodes": horizon * common_state_count,
            "q_value_entries": (
                horizon
                * common_state_count
                * game.num_joint_private_types
                * game.num_joint_actions
                * game.num_players
            ),
            "transition_terms": (
                horizon
                * common_state_count
                * game.num_joint_private_types
                * game.num_joint_actions
                * game.num_joint_private_types
                * common_state_count
            ),
            "prescriptions_per_player": list(per_player_prescriptions),
            "prescription_profile_count": prescription_profile_count,
            "prescription_profile_capacity": maximum_prescription_profiles,
            "stage_profile_candidates": (
                horizon * common_state_count * prescription_profile_count
            ),
        },
        "bayes_evidence": _bayes_evidence(result),
        "branch_evidence": _branch_evidence(result),
        "certificate": {
            "valid": bool(result.valid),
            "result_label": result.result_label,
            "method_id": result.method_id,
            "game_id": result.game_id,
            "selector_id": result.selector_id,
            "branch_id": result.branch_id,
            "maximum_prescription_profiles": result.maximum_prescription_profiles,
            "claim_boundary": (
                "finite pure-prescription common-information Markov-perfect candidate"
            ),
        },
    }


def _specifications():
    return (
        ("baseline", 2, 2, (2, 2), (2, 2), 16),
        ("common-states-8", 2, 8, (2, 2), (2, 2), 16),
        ("private-types-3", 2, 2, (3, 3), (2, 2), 64),
        ("actions-3", 2, 2, (2, 2), (3, 3), 81),
        ("prescription-profiles-256", 2, 2, (4, 4), (2, 2), 256),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive.")

    cases = [
        _case(
            name,
            horizon,
            common_state_count,
            private_type_counts,
            action_counts,
            maximum_prescription_profiles,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for (
            name,
            horizon,
            common_state_count,
            private_type_counts,
            action_counts,
            maximum_prescription_profiles,
        ) in _specifications()
    ]
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "all_valid": all(case["certificate"]["valid"] for case in cases),
        "all_bayes_consistent": all(
            case["bayes_evidence"]["all_common_beliefs_consistent"] for case in cases
        ),
    }
    if arguments.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
