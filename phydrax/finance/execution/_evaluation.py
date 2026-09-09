#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Disjoint holdout policy evaluation with independent ledger replay."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._semantics import (
    execution_cash_inventory_pnl,
    ExecutionAccounting,
    ExecutionLedger,
    ExecutionReplayEvidence,
    replay_execution_ledger,
)
from ._signature_policy import (
    CausalSignaturePolicy,
    evaluate_causal_signature_policy,
    SignaturePolicyEvaluation,
    SignaturePolicySampleSet,
)


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


class ExecutionPolicyEvaluationPlan(StrictModule):
    """A holdout-only replay tolerance and semantic evaluation identity."""

    replay_tolerance: float = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)

    def __init__(self, *, replay_tolerance: float, evaluation_id: str):
        tolerance = float(replay_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("replay_tolerance must be finite and nonnegative.")
        self.replay_tolerance = tolerance
        self.evaluation_id = _identifier(evaluation_id, "evaluation_id")


class ExecutionPolicyEvaluationResult(StrictModule):
    """Policy actions plus separately reconstructed cash, inventory, and PnL."""

    policy_evaluation: SignaturePolicyEvaluation
    accounting: tuple[ExecutionAccounting, ...]
    replay_evidence: tuple[ExecutionReplayEvidence, ...]
    cash: Array
    inventory: Array
    net_pnl: Array
    action_valid: Array
    replay_valid: Array
    successful: Array
    evaluation_id: str = eqx.field(static=True)
    evidence_scope: str = eqx.field(static=True)
    use_claim: str = eqx.field(static=True)


def evaluate_execution_policy_holdout(
    policy: CausalSignaturePolicy,
    holdout_sample: SignaturePolicySampleSet,
    ledgers: tuple[ExecutionLedger, ...],
    initial_mark_prices: ArrayLike,
    terminal_mark_prices: ArrayLike,
    plan: ExecutionPolicyEvaluationPlan,
    /,
) -> ExecutionPolicyEvaluationResult:
    """Evaluate actions and independently replay exogenous holdout event ledgers.

    Ledgers contain caller-supplied or model-sampled fills. This routine deliberately
    does not infer matching, fill probability, execution quality, or venue behavior.
    """

    if not isinstance(policy, CausalSignaturePolicy):
        raise TypeError("policy must be a CausalSignaturePolicy.")
    if not isinstance(holdout_sample, SignaturePolicySampleSet):
        raise TypeError("holdout_sample must be a SignaturePolicySampleSet.")
    if holdout_sample.sample_role != "holdout":
        raise ValueError("Execution policy evaluation requires a holdout sample.")
    if not isinstance(plan, ExecutionPolicyEvaluationPlan):
        raise TypeError("plan must be an ExecutionPolicyEvaluationPlan.")
    ledger_values = tuple(ledgers)
    if len(ledger_values) != holdout_sample.num_paths:
        raise ValueError("ledgers must contain one independent ledger per holdout path.")
    if any(not isinstance(ledger, ExecutionLedger) for ledger in ledger_values):
        raise TypeError("ledgers must contain only ExecutionLedger values.")
    initial_marks = jnp.asarray(initial_mark_prices)
    terminal_marks = jnp.asarray(terminal_mark_prices)
    expected = (holdout_sample.num_paths,)
    if initial_marks.shape != expected or terminal_marks.shape != expected:
        raise ValueError(f"mark price arrays must have shape {expected}.")
    if not bool(jnp.all(jnp.isfinite(initial_marks))) or bool(
        jnp.any(initial_marks <= 0.0)
    ):
        raise ValueError("initial_mark_prices must be finite and positive.")
    if not bool(jnp.all(jnp.isfinite(terminal_marks))) or bool(
        jnp.any(terminal_marks <= 0.0)
    ):
        raise ValueError("terminal_mark_prices must be finite and positive.")

    policy_evaluation = evaluate_causal_signature_policy(policy, holdout_sample)
    accounting = tuple(
        execution_cash_inventory_pnl(
            ledger,
            initial_mark_price=initial_marks[index],
            terminal_mark_price=terminal_marks[index],
        )
        for index, ledger in enumerate(ledger_values)
    )
    replay = tuple(
        replay_execution_ledger(
            ledger,
            initial_mark_price=initial_marks[index],
            terminal_mark_price=terminal_marks[index],
            tolerance=plan.replay_tolerance,
        )
        for index, ledger in enumerate(ledger_values)
    )
    cash = jnp.stack(tuple(value.cash for value in accounting))
    inventory = jnp.stack(tuple(value.inventory for value in accounting))
    pnl = jnp.stack(tuple(value.net_pnl for value in accounting))
    replay_valid = jnp.stack(tuple(value.passed for value in replay))
    action_valid = policy_evaluation.valid
    successful = (
        jnp.all(action_valid) & jnp.all(replay_valid) & jnp.all(jnp.isfinite(pnl))
    )
    return ExecutionPolicyEvaluationResult(
        policy_evaluation=policy_evaluation,
        accounting=accounting,
        replay_evidence=replay,
        cash=cash,
        inventory=inventory,
        net_pnl=pnl,
        action_valid=action_valid,
        replay_valid=replay_valid,
        successful=successful,
        evaluation_id=plan.evaluation_id,
        evidence_scope=(
            "disjoint-holdout-actions-and-independent-exogenous-event-ledger-replay"
        ),
        use_claim="technical-candidate-evaluation-only",
    )


__all__ = [
    "ExecutionPolicyEvaluationPlan",
    "ExecutionPolicyEvaluationResult",
    "evaluate_execution_policy_holdout",
]
