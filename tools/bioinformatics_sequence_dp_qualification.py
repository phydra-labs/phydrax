#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unit qualification for sequence lowering and differentiable dynamic programs."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics import sequence
from tools.bioinformatics_common_qualification import (
    emit_report,
    fingerprint,
    method_contract_evidence,
    qualification_report,
)


_MATCH = 0
_INSERT = 1
_DELETE = 2


def _brute_affine_global(
    query: tuple[int, ...],
    target: tuple[int, ...],
    scores: np.ndarray,
    gap_open: float,
    gap_extend: float,
) -> float:
    """Enumerate every legal three-state global-alignment path."""

    @lru_cache(maxsize=None)
    def best(query_index: int, target_index: int, state: int) -> float:
        if query_index == len(query) and target_index == len(target):
            return 0.0
        candidates: list[float] = []
        if query_index < len(query) and target_index < len(target):
            candidates.append(
                float(scores[query[query_index], target[target_index]])
                + best(query_index + 1, target_index + 1, _MATCH)
            )
        if query_index < len(query) and state in (_MATCH, _INSERT):
            penalty = gap_extend if state == _INSERT else gap_open
            candidates.append(penalty + best(query_index + 1, target_index, _INSERT))
        if target_index < len(target) and state in (_MATCH, _DELETE):
            penalty = gap_extend if state == _DELETE else gap_open
            candidates.append(penalty + best(query_index, target_index + 1, _DELETE))
        return max(candidates, default=-np.inf)

    return best(0, 0, _MATCH)


def _affine_alignment_case() -> dict[str, object]:
    query_values = (0, 1, 2, 3)
    target_values = (0, 2, 3)
    query = jnp.asarray(query_values, dtype=jnp.int32)
    target = jnp.asarray(target_values, dtype=jnp.int32)
    scoring = sequence.identity_substitution_table(
        ("A", "C", "G", "T"), match_score=2.0, mismatch_score=-1.0
    )
    penalties = sequence.AffineGapPenalties(-2.0, -0.5)
    plan = sequence.AlignmentExecutionPlan.full(4, 3, traceback_capacity=7)
    result = sequence.align_affine(query, target, scoring, penalties, plan)
    oracle = _brute_affine_global(
        query_values,
        target_values,
        np.asarray(scoring.encoded_scores),
        -2.0,
        -0.5,
    )
    score = float(np.asarray(result.score))
    score_error = abs(score - oracle)

    insufficient = sequence.AlignmentExecutionPlan.full(4, 3, traceback_capacity=1)
    rejected = sequence.align_affine(query, target, scoring, penalties, insufficient)
    capacity_rejected = (
        not bool(np.asarray(rejected.valid))
        and not bool(np.asarray(rejected.evidence.capacity_sufficient))
        and int(np.asarray(rejected.status)) != 0
    )
    contract = method_contract_evidence(result.method_contract)
    inputs = {
        "query": query,
        "target": target,
        "substitution_scores": scoring.encoded_scores,
        "gap_open": -2.0,
        "gap_extend": -0.5,
        "traceback_capacity": 7,
    }
    return {
        "scope": "unit_qualification",
        "oracle": "exhaustive three-state affine path enumeration",
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "observed_score": score,
        "oracle_score": oracle,
        "absolute_error": score_error,
        "status": int(np.asarray(result.status)),
        "valid": bool(np.asarray(result.valid)),
        "exact": bool(np.asarray(result.exact)),
        "traceback_complete": bool(np.asarray(result.evidence.traceback_complete)),
        "score_consistent": bool(np.asarray(result.evidence.score_consistent)),
        "capacity_check": {
            "configured_traceback_capacity": 1,
            "required_upper_bound": len(query_values) + len(target_values),
            "status": int(np.asarray(rejected.status)),
            "rejected": capacity_rejected,
        },
        "passed": bool(
            np.asarray(result.valid)
            and np.asarray(result.exact)
            and score_error <= 2.0e-6
            and capacity_rejected
        ),
    }


def _pair_hmm_gradient_case() -> dict[str, object]:
    model = sequence.PairHMM(
        jnp.zeros((3,)),
        jnp.zeros((3, 3)),
        jnp.zeros((3,)),
        jnp.zeros((2, 2)),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
    )
    match = jnp.asarray(((0.20, -0.10), (0.00, 0.30)))
    insertion = jnp.asarray((0.10, -0.20))
    deletion = jnp.asarray((-0.10, 0.05))
    plan = sequence.PairHMMExecutionPlan.full(2, 2, traceback_capacity=4)

    def objective(match_potential):
        return sequence.pair_hmm_forward_backward_from_potentials(
            model,
            match_potential,
            insertion,
            deletion,
            plan,
        ).log_partition

    result = sequence.pair_hmm_forward_backward_from_potentials(
        model, match, insertion, deletion, plan
    )
    gradient = jax.grad(objective)(match)
    occupancy = result.state_marginals[1:, 1:, _MATCH]
    gradient_error = float(np.max(np.abs(np.asarray(gradient - occupancy))))
    contract = method_contract_evidence(result.method_contract)
    inputs = {
        "initial_logits": model.initial_logits,
        "transition_logits": model.transition_logits,
        "terminal_logits": model.terminal_logits,
        "match_log_potentials": match,
        "insertion_log_potentials": insertion,
        "deletion_log_potentials": deletion,
        "plan_id": plan.plan_id,
    }
    return {
        "scope": "unit_qualification",
        "oracle": "d(log partition)/d(match log potential) equals match occupancy",
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "log_partition": float(np.asarray(result.log_partition)),
        "maximum_gradient_identity_error": gradient_error,
        "forward_backward_error": float(np.asarray(result.forward_backward_error)),
        "posterior_conservation_error": float(
            np.asarray(result.posterior_conservation_error)
        ),
        "status": int(np.asarray(result.status)),
        "valid": bool(np.asarray(result.valid)),
        "exact": bool(np.asarray(result.exact)),
        "passed": bool(
            np.asarray(result.valid)
            and np.asarray(result.exact)
            and gradient_error <= 2.0e-5
            and float(np.asarray(result.forward_backward_error)) <= 2.0e-5
            and float(np.asarray(result.posterior_conservation_error)) <= 2.0e-5
        ),
    }


def qualification() -> dict[str, object]:
    return qualification_report(
        "sequence_dp",
        {
            "affine_global_alignment": _affine_alignment_case(),
            "pair_hmm_gradient_identity": _pair_hmm_gradient_case(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify public sequence and dynamic-programming APIs."
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    return emit_report(qualification(), arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
