#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._posterior import PosteriorProblem


ExpectedUtilityTarget = Literal[
    "parameter",
    "predictive",
    "model_discrimination",
]
RetrospectiveDesignStrategy = Literal[
    "random",
    "space_filling",
    "uncertainty_only",
    "domain_heuristic",
    "proposed_design",
]

_UTILITY_TARGETS = ("parameter", "predictive", "model_discrimination")
_RETROSPECTIVE_STRATEGIES = (
    "random",
    "space_filling",
    "uncertainty_only",
    "domain_heuristic",
    "proposed_design",
)
_EXACT_SELECTION_LIMIT = 24


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _identifiers(
    values: Sequence[str],
    name: str,
    /,
    *,
    nonempty: bool = True,
) -> tuple[str, ...]:
    result = tuple(_identifier(value, name) for value in values)
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique.")
    return result


def _nonnegative_finite(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _positive_integer(value: int, name: str, /, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result != value or result < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return result


def _utility_target(value: ExpectedUtilityTarget, /) -> ExpectedUtilityTarget:
    if value not in _UTILITY_TARGETS:
        raise ValueError(
            "utility_target must be 'parameter', 'predictive', or 'model_discrimination'."
        )
    return value


class ExperimentalDesignCandidate(StrictModule, NonTrainableState):
    """One predeclared experiment with stable identity and planning constraints.

    ``cost`` is the per-candidate cost. A non-empty ``setup_id`` declares a
    shared setup whose ``setup_cost`` is charged once per selected batch.
    ``feasibility_group`` supports an allowed-group constraint without implying
    that experiments in the same group are mutually exclusive. Mutual
    exclusions are declared explicitly on :class:`ExperimentalBatchConstraints`.
    """

    candidate_id: str = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    cost: float = eqx.field(static=True)
    feasibility_group: str = eqx.field(static=True)
    prediction_source_id: str = eqx.field(static=True)
    setup_id: str = eqx.field(static=True)
    setup_cost: float = eqx.field(static=True)
    diversity_group: str = eqx.field(static=True)
    mandatory_control: bool = eqx.field(static=True)
    candidate_content_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate_id: str,
        condition_id: str,
        cost: float,
        feasibility_group: str,
        prediction_source_id: str,
        /,
        *,
        setup_id: str = "",
        setup_cost: float = 0.0,
        diversity_group: str | None = None,
        mandatory_control: bool = False,
    ):
        identifier = _identifier(candidate_id, "candidate_id")
        condition = _identifier(condition_id, "condition_id")
        group = _identifier(feasibility_group, "feasibility_group")
        source = _identifier(prediction_source_id, "prediction_source_id")
        candidate_cost = _nonnegative_finite(cost, "cost")
        if not isinstance(setup_id, str):
            raise TypeError("setup_id must be a string.")
        setup = setup_id.strip()
        shared_cost = _nonnegative_finite(setup_cost, "setup_cost")
        if not setup and shared_cost != 0.0:
            raise ValueError("A positive setup_cost requires a non-empty setup_id.")
        diversity = (
            identifier
            if diversity_group is None
            else _identifier(diversity_group, "diversity_group")
        )
        if not isinstance(mandatory_control, bool):
            raise TypeError("mandatory_control must be boolean.")
        payload = {
            "kind": "experimental-design-candidate-v1",
            "candidate_id": identifier,
            "condition_id": condition,
            "cost": candidate_cost.hex(),
            "feasibility_group": group,
            "prediction_source_id": source,
            "setup_id": setup,
            "setup_cost": shared_cost.hex(),
            "diversity_group": diversity,
            "mandatory_control": mandatory_control,
        }
        self.candidate_id = identifier
        self.condition_id = condition
        self.cost = candidate_cost
        self.feasibility_group = group
        self.prediction_source_id = source
        self.setup_id = setup
        self.setup_cost = shared_cost
        self.diversity_group = diversity
        self.mandatory_control = mandatory_control
        self.candidate_content_id = canonical_fingerprint(payload)


class ExpectedUtilityResult(StrictModule):
    """Candidate utilities bound to candidate, prediction, and model content."""

    expected_utility: Array
    estimator_standard_error: Array
    estimator_bias_bound: Array
    valid: Array
    candidate_ids: tuple[str, ...] = eqx.field(static=True)
    candidate_content_ids: tuple[str, ...] = eqx.field(static=True)
    prediction_source_ids: tuple[str, ...] = eqx.field(static=True)
    model_ids: tuple[str, ...] = eqx.field(static=True)
    utility_target: ExpectedUtilityTarget = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    error_basis: str = eqx.field(static=True)
    bound_direction: Literal["none", "lower", "upper"] = eqx.field(static=True)
    outer_sample_count: int = eqx.field(static=True)
    inner_sample_count: int = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        expected_utility: ArrayLike,
        estimator_standard_error: ArrayLike,
        estimator_bias_bound: ArrayLike,
        valid: ArrayLike,
        candidates: Sequence[ExperimentalDesignCandidate],
        model_ids: Sequence[str],
        utility_target: ExpectedUtilityTarget,
        method_id: str,
        approximation: str,
        error_basis: str,
        bound_direction: Literal["none", "lower", "upper"] = "none",
        outer_sample_count: int = 0,
        inner_sample_count: int = 0,
        unit_id: str = "nat",
    ):
        candidate_values = _candidate_tuple(candidates)
        values = jnp.asarray(expected_utility, dtype=float)
        errors = jnp.asarray(estimator_standard_error, dtype=float)
        biases = jnp.asarray(estimator_bias_bound, dtype=float)
        validity = jnp.asarray(valid, dtype=bool)
        expected_shape = (len(candidate_values),)
        if (
            values.shape != expected_shape
            or errors.shape != expected_shape
            or biases.shape != expected_shape
            or validity.shape != expected_shape
        ):
            raise ValueError(
                "Expected utility, uncertainty, bias, and validity must have one "
                "rank-1 entry per candidate."
            )
        if bool(jnp.any(jnp.isinf(errors))) or bool(jnp.any(jnp.isinf(biases))):
            raise ValueError("Estimator error fields may be finite or NaN, not infinite.")
        if bool(jnp.any(validity & ~jnp.isfinite(values))):
            raise ValueError("Valid expected utilities must be finite.")
        if bool(jnp.any(validity & (~jnp.isfinite(errors) | (errors < 0.0)))):
            raise ValueError(
                "Valid expected utilities require finite non-negative standard errors."
            )
        finite_bias = jnp.isfinite(biases)
        if bool(jnp.any(finite_bias & (biases < 0.0))):
            raise ValueError("Finite estimator bias bounds must be non-negative.")
        if bound_direction not in ("none", "lower", "upper"):
            raise ValueError("bound_direction must be 'none', 'lower', or 'upper'.")
        self.expected_utility = values
        self.estimator_standard_error = errors
        self.estimator_bias_bound = biases
        self.valid = validity
        self.candidate_ids = tuple(
            candidate.candidate_id for candidate in candidate_values
        )
        self.candidate_content_ids = tuple(
            candidate.candidate_content_id for candidate in candidate_values
        )
        self.prediction_source_ids = tuple(
            candidate.prediction_source_id for candidate in candidate_values
        )
        self.model_ids = tuple(sorted(_identifiers(model_ids, "model_ids")))
        self.utility_target = _utility_target(utility_target)
        self.method_id = _identifier(method_id, "method_id")
        self.approximation = _identifier(approximation, "approximation")
        self.error_basis = _identifier(error_basis, "error_basis")
        self.bound_direction = bound_direction
        self.outer_sample_count = _positive_integer(
            outer_sample_count, "outer_sample_count", minimum=0
        )
        self.inner_sample_count = _positive_integer(
            inner_sample_count, "inner_sample_count", minimum=0
        )
        self.unit_id = _identifier(unit_id, "unit_id")


class ExperimentalBatchConstraints(StrictModule, NonTrainableState):
    """Host-side cost, control, exclusion, feasibility, and diversity contract."""

    budget: float = eqx.field(static=True)
    minimum_batch_size: int = eqx.field(static=True)
    maximum_batch_size: int = eqx.field(static=True)
    required_candidate_ids: tuple[str, ...] = eqx.field(static=True)
    mutually_exclusive_candidate_groups: tuple[tuple[str, ...], ...] = eqx.field(
        static=True
    )
    allowed_feasibility_groups: tuple[str, ...] = eqx.field(static=True)
    minimum_diversity_groups: int = eqx.field(static=True)
    maximum_per_diversity_group: int | None = eqx.field(static=True)
    constraints_id: str = eqx.field(static=True)

    def __init__(
        self,
        budget: float,
        maximum_batch_size: int,
        /,
        *,
        minimum_batch_size: int = 1,
        required_candidate_ids: Sequence[str] = (),
        mutually_exclusive_candidate_groups: Sequence[Sequence[str]] = (),
        allowed_feasibility_groups: Sequence[str] = (),
        minimum_diversity_groups: int = 1,
        maximum_per_diversity_group: int | None = None,
    ):
        budget_value = _nonnegative_finite(budget, "budget")
        minimum_size = _positive_integer(minimum_batch_size, "minimum_batch_size")
        maximum_size = _positive_integer(maximum_batch_size, "maximum_batch_size")
        if minimum_size > maximum_size:
            raise ValueError("minimum_batch_size cannot exceed maximum_batch_size.")
        required = tuple(
            sorted(
                _identifiers(
                    required_candidate_ids,
                    "required_candidate_ids",
                    nonempty=False,
                )
            )
        )
        exclusion_groups: list[tuple[str, ...]] = []
        for group in mutually_exclusive_candidate_groups:
            values = tuple(sorted(_identifiers(group, "mutual exclusion group")))
            if len(values) < 2:
                raise ValueError(
                    "Every mutual exclusion group must contain at least two candidates."
                )
            exclusion_groups.append(values)
        exclusions = tuple(sorted(set(exclusion_groups)))
        allowed = tuple(
            sorted(
                _identifiers(
                    allowed_feasibility_groups,
                    "allowed_feasibility_groups",
                    nonempty=False,
                )
            )
        )
        minimum_diversity = _positive_integer(
            minimum_diversity_groups,
            "minimum_diversity_groups",
            minimum=0,
        )
        if minimum_diversity > maximum_size:
            raise ValueError("minimum_diversity_groups cannot exceed maximum_batch_size.")
        maximum_diversity = None
        if maximum_per_diversity_group is not None:
            maximum_diversity = _positive_integer(
                maximum_per_diversity_group,
                "maximum_per_diversity_group",
            )
        payload = {
            "kind": "experimental-batch-constraints-v1",
            "budget": budget_value.hex(),
            "minimum_batch_size": minimum_size,
            "maximum_batch_size": maximum_size,
            "required_candidate_ids": list(required),
            "mutually_exclusive_candidate_groups": [list(group) for group in exclusions],
            "allowed_feasibility_groups": list(allowed),
            "minimum_diversity_groups": minimum_diversity,
            "maximum_per_diversity_group": maximum_diversity,
        }
        self.budget = budget_value
        self.minimum_batch_size = minimum_size
        self.maximum_batch_size = maximum_size
        self.required_candidate_ids = required
        self.mutually_exclusive_candidate_groups = exclusions
        self.allowed_feasibility_groups = allowed
        self.minimum_diversity_groups = minimum_diversity
        self.maximum_per_diversity_group = maximum_diversity
        self.constraints_id = canonical_fingerprint(payload)


class ExperimentalBatchPlan(StrictModule, NonTrainableState):
    """Immutable content-addressed batch registered before acquisition."""

    selected_candidate_ids: tuple[str, ...] = eqx.field(static=True)
    candidate_ids: tuple[str, ...] = eqx.field(static=True)
    candidate_content_ids: tuple[str, ...] = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)
    budget: float = eqx.field(static=True)
    planned_total_cost: float = eqx.field(static=True)
    objective_value: float = eqx.field(static=True)
    model_ids: tuple[str, ...] = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)
    constraints_id: str = eqx.field(static=True)
    estimator_method_id: str = eqx.field(static=True)
    estimator_approximation: str = eqx.field(static=True)
    selection_policy_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        selected_candidate_ids: Sequence[str],
        objective_id: str,
        budget: float,
        model_ids: Sequence[str],
        /,
        *,
        candidate_ids: Sequence[str],
        candidate_content_ids: Sequence[str],
        analysis_id: str,
        constraints_id: str,
        estimator_method_id: str,
        estimator_approximation: str,
        selection_policy_id: str,
        planned_total_cost: float,
        objective_value: float,
    ):
        candidates = _identifiers(candidate_ids, "candidate_ids")
        contents = _identifiers(candidate_content_ids, "candidate_content_ids")
        if len(contents) != len(candidates):
            raise ValueError(
                "candidate_content_ids must contain one entry per candidate ID."
            )
        records = tuple(sorted(zip(candidates, contents, strict=True)))
        canonical_candidates = tuple(record[0] for record in records)
        canonical_contents = tuple(record[1] for record in records)
        selected = tuple(
            sorted(_identifiers(selected_candidate_ids, "selected_candidate_ids"))
        )
        if not set(selected).issubset(canonical_candidates):
            raise ValueError("Every selected candidate must belong to the frozen panel.")
        budget_value = _nonnegative_finite(budget, "budget")
        total_cost = _nonnegative_finite(planned_total_cost, "planned_total_cost")
        tolerance = 64.0 * jnp.finfo(float).eps * max(1.0, budget_value)
        if total_cost > budget_value + float(tolerance):
            raise ValueError("planned_total_cost exceeds budget.")
        value = float(objective_value)
        if not isfinite(value):
            raise ValueError("objective_value must be finite.")
        models = tuple(sorted(_identifiers(model_ids, "model_ids")))
        payload = {
            "kind": "prospective-experimental-batch-plan-v1",
            "selected_candidate_ids": list(selected),
            "candidates": [list(record) for record in records],
            "objective_id": _identifier(objective_id, "objective_id"),
            "budget": budget_value.hex(),
            "planned_total_cost": total_cost.hex(),
            "objective_value": value.hex(),
            "model_ids": list(models),
            "analysis_id": _identifier(analysis_id, "analysis_id"),
            "constraints_id": _identifier(constraints_id, "constraints_id"),
            "estimator_method_id": _identifier(
                estimator_method_id, "estimator_method_id"
            ),
            "estimator_approximation": _identifier(
                estimator_approximation, "estimator_approximation"
            ),
            "selection_policy_id": _identifier(
                selection_policy_id, "selection_policy_id"
            ),
        }
        self.selected_candidate_ids = selected
        self.candidate_ids = canonical_candidates
        self.candidate_content_ids = canonical_contents
        self.objective_id = payload["objective_id"]
        self.budget = budget_value
        self.planned_total_cost = total_cost
        self.objective_value = value
        self.model_ids = models
        self.analysis_id = payload["analysis_id"]
        self.constraints_id = payload["constraints_id"]
        self.estimator_method_id = payload["estimator_method_id"]
        self.estimator_approximation = payload["estimator_approximation"]
        self.selection_policy_id = payload["selection_policy_id"]
        self.plan_id = canonical_fingerprint(payload)

    def to_record(self) -> dict[str, Any]:
        """Return the complete canonical prospective registration record."""
        return {
            "kind": "prospective-experimental-batch-plan-v1",
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "candidate_ids": list(self.candidate_ids),
            "candidate_content_ids": list(self.candidate_content_ids),
            "objective_id": self.objective_id,
            "budget": self.budget.hex(),
            "planned_total_cost": self.planned_total_cost.hex(),
            "objective_value": self.objective_value.hex(),
            "model_ids": list(self.model_ids),
            "analysis_id": self.analysis_id,
            "constraints_id": self.constraints_id,
            "estimator_method_id": self.estimator_method_id,
            "estimator_approximation": self.estimator_approximation,
            "selection_policy_id": self.selection_policy_id,
            "plan_id": self.plan_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> ExperimentalBatchPlan:
        """Reconstruct a plan and reject any content-identity mismatch."""
        expected_keys = {
            "kind",
            "selected_candidate_ids",
            "candidate_ids",
            "candidate_content_ids",
            "objective_id",
            "budget",
            "planned_total_cost",
            "objective_value",
            "model_ids",
            "analysis_id",
            "constraints_id",
            "estimator_method_id",
            "estimator_approximation",
            "selection_policy_id",
            "plan_id",
        }
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            raise ValueError("Experimental batch plan record has an invalid schema.")
        if record["kind"] != "prospective-experimental-batch-plan-v1":
            raise ValueError("Experimental batch plan record has an invalid kind.")
        plan = cls(
            record["selected_candidate_ids"],
            record["objective_id"],
            float.fromhex(record["budget"]),
            record["model_ids"],
            candidate_ids=record["candidate_ids"],
            candidate_content_ids=record["candidate_content_ids"],
            analysis_id=record["analysis_id"],
            constraints_id=record["constraints_id"],
            estimator_method_id=record["estimator_method_id"],
            estimator_approximation=record["estimator_approximation"],
            selection_policy_id=record["selection_policy_id"],
            planned_total_cost=float.fromhex(record["planned_total_cost"]),
            objective_value=float.fromhex(record["objective_value"]),
        )
        if plan.plan_id != record["plan_id"]:
            raise ValueError("Experimental batch plan content identity is corrupt.")
        return plan


class RetrospectiveDesignResult(StrictModule, NonTrainableState):
    """Historical replay normalized by realized planned cost, not budget ceiling."""

    plans: tuple[ExperimentalBatchPlan, ...]
    realized_utility: Array
    realized_valid: Array
    planned_total_costs: Array
    selected_batch_sizes: Array
    cost_normalized_realized_utility: Array
    cost_normalized_valid: Array
    strategy_ids: tuple[RetrospectiveDesignStrategy, ...] = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    evaluation_kind: str = eqx.field(static=True)
    comparison_basis: str = eqx.field(static=True)
    matched_planned_total_cost: bool = eqx.field(static=True)
    matched_batch_size: bool = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)

    def __init__(
        self,
        plans: Sequence[ExperimentalBatchPlan],
        realized_utility: ArrayLike,
        realized_valid: ArrayLike,
        /,
        *,
        metric_id: str,
    ):
        plan_values = tuple(plans)
        if len(plan_values) != len(_RETROSPECTIVE_STRATEGIES) or any(
            not isinstance(plan, ExperimentalBatchPlan) for plan in plan_values
        ):
            raise ValueError(
                "plans must contain random, space-filling, uncertainty-only, "
                "domain-heuristic, and proposed plans in that order."
            )
        values = jnp.asarray(realized_utility, dtype=float)
        validity = jnp.asarray(realized_valid, dtype=bool)
        expected_shape = (len(_RETROSPECTIVE_STRATEGIES),)
        if values.shape != expected_shape or validity.shape != expected_shape:
            raise ValueError("Retrospective metrics must have one value per strategy.")
        if bool(jnp.any(validity & ~jnp.isfinite(values))):
            raise ValueError("Valid retrospective utilities must be finite.")
        comparison_bindings = {
            (
                plan.budget,
                plan.constraints_id,
                plan.candidate_ids,
                plan.candidate_content_ids,
                plan.model_ids,
                plan.analysis_id,
            )
            for plan in plan_values
        }
        if len(comparison_bindings) != 1:
            raise ValueError(
                "Retrospective plans must share one budget ceiling, constraint set, "
                "candidate panel, model set, and analysis."
            )
        costs = jnp.asarray(
            tuple(plan.planned_total_cost for plan in plan_values), dtype=float
        )
        sizes = jnp.asarray(
            tuple(len(plan.selected_candidate_ids) for plan in plan_values), dtype=int
        )
        positive_cost = jnp.isfinite(costs) & (costs > 0.0)
        normalized_valid = validity & positive_cost
        normalized = jnp.where(normalized_valid, values / costs, jnp.nan)
        cost_tolerance = (
            64.0
            * jnp.finfo(costs.dtype).eps
            * max(1.0, max(plan.budget for plan in plan_values))
        )
        matched_cost = bool(jnp.max(costs) - jnp.min(costs) <= cost_tolerance)
        matched_size = (
            len({len(plan.selected_candidate_ids) for plan in plan_values}) == 1
        )
        metric = _identifier(metric_id, "metric_id")
        self.plans = plan_values
        self.realized_utility = values
        self.realized_valid = validity
        self.planned_total_costs = costs
        self.selected_batch_sizes = sizes
        self.cost_normalized_realized_utility = normalized
        self.cost_normalized_valid = normalized_valid
        self.strategy_ids = _RETROSPECTIVE_STRATEGIES
        self.metric_id = metric
        self.evaluation_kind = "retrospective_cost_normalized_replay"
        self.comparison_basis = "realized_utility_per_planned_total_cost"
        self.matched_planned_total_cost = matched_cost
        self.matched_batch_size = matched_size
        self.evaluation_id = canonical_fingerprint(
            {
                "kind": "retrospective-experimental-design-evaluation-v2",
                "strategy_ids": list(_RETROSPECTIVE_STRATEGIES),
                "plan_ids": [plan.plan_id for plan in plan_values],
                "metric_id": metric,
                "realized": array_tree_fingerprint((values, validity)),
                "planned_total_costs": array_tree_fingerprint(costs),
                "selected_batch_sizes": array_tree_fingerprint(sizes),
                "comparison_basis": self.comparison_basis,
            }
        )


def exact_finite_expected_utility(
    conditional_observation_probabilities: ArrayLike,
    target_probabilities: ArrayLike,
    /,
    *,
    candidates: Sequence[ExperimentalDesignCandidate],
    model_ids: Sequence[str],
    utility_target: ExpectedUtilityTarget,
    utility_values: ArrayLike | None = None,
) -> ExpectedUtilityResult:
    """Enumerate a finite target and finite observations exactly.

    Without ``utility_values`` this computes Shannon mutual information in nats.
    Supplying a ``(candidate, target, observation)`` utility table computes its
    exact expectation under the declared joint distribution instead.
    """
    target = _utility_target(utility_target)
    candidate_values = _candidate_tuple(candidates)
    conditional = jnp.asarray(conditional_observation_probabilities, dtype=float)
    priors = jnp.asarray(target_probabilities, dtype=float)
    if conditional.ndim != 3:
        raise ValueError(
            "conditional_observation_probabilities must have shape "
            "(candidate, target, observation)."
        )
    if conditional.shape[0] != len(candidate_values):
        raise ValueError("Conditional probabilities do not align with candidates.")
    if priors.shape != (conditional.shape[1],):
        raise ValueError("target_probabilities must have one entry per target value.")
    if bool(jnp.any(~jnp.isfinite(conditional))) or bool(jnp.any(~jnp.isfinite(priors))):
        raise ValueError("Finite expected utility requires finite probabilities.")
    if bool(jnp.any(conditional < 0.0)) or bool(jnp.any(priors < 0.0)):
        raise ValueError("Probabilities must be non-negative.")
    tolerance = (
        64.0
        * jnp.finfo(conditional.dtype).eps
        * max(conditional.shape[1], conditional.shape[2])
    )
    if bool(jnp.any(jnp.abs(jnp.sum(conditional, axis=-1) - 1.0) > tolerance)):
        raise ValueError("Every conditional observation distribution must sum to one.")
    if bool(jnp.abs(jnp.sum(priors) - 1.0) > tolerance):
        raise ValueError("target_probabilities must sum to one.")
    joint = conditional * priors[None, :, None]
    if utility_values is None:
        marginal = jnp.sum(joint, axis=1)
        positive_joint = joint > 0.0
        safe_conditional = jnp.where(positive_joint, conditional, 1.0)
        safe_marginal = jnp.where(positive_joint, marginal[:, None, :], 1.0)
        pointwise = jnp.log(safe_conditional) - jnp.log(safe_marginal)
        values = jnp.sum(jnp.where(positive_joint, joint * pointwise, 0.0), axis=(1, 2))
        method_id = "exact_finite_mutual_information"
        unit_id = "nat"
    else:
        utilities = jnp.asarray(utility_values, dtype=float)
        if utilities.shape != conditional.shape:
            raise ValueError(
                "utility_values must match the candidate-target-observation shape."
            )
        if bool(jnp.any(~jnp.isfinite(utilities))):
            raise ValueError("utility_values must be finite.")
        values = jnp.sum(joint * utilities, axis=(1, 2))
        method_id = "exact_finite_expected_utility"
        unit_id = "declared_utility"
    valid = jnp.isfinite(values)
    return ExpectedUtilityResult(
        expected_utility=jnp.where(valid, values, jnp.nan),
        estimator_standard_error=jnp.zeros_like(values),
        estimator_bias_bound=jnp.zeros_like(values),
        valid=valid,
        candidates=candidate_values,
        model_ids=model_ids,
        utility_target=target,
        method_id=method_id,
        approximation="exact_finite_enumeration",
        error_basis="no Monte Carlo error; declared finite support is enumerated",
        outer_sample_count=0,
        inner_sample_count=0,
        unit_id=unit_id,
    )


def _log_mean_exp(values: Array, axis: int, /) -> Array:
    return logsumexp(values, axis=axis) - jnp.log(
        jnp.asarray(values.shape[axis], dtype=values.dtype)
    )


def _nested_result(
    contributions: Array,
    /,
    *,
    candidates: Sequence[ExperimentalDesignCandidate],
    model_ids: Sequence[str],
    utility_target: ExpectedUtilityTarget,
    method_id: str,
    approximation: str,
    inner_sample_count: int,
    error_qualifier: str = "",
) -> ExpectedUtilityResult:
    if contributions.ndim != 2:
        raise ValueError(
            "Nested utility contributions must have candidate and outer axes."
        )
    outer_count = int(contributions.shape[1])
    if outer_count < 2:
        raise ValueError("At least two outer samples are required to estimate error.")
    valid = jnp.all(jnp.isfinite(contributions), axis=1)
    means = jnp.mean(contributions, axis=1)
    errors = jnp.std(contributions, axis=1, ddof=1) / jnp.sqrt(
        jnp.asarray(outer_count, dtype=contributions.dtype)
    )
    return ExpectedUtilityResult(
        expected_utility=jnp.where(valid, means, jnp.nan),
        estimator_standard_error=jnp.where(valid, errors, jnp.nan),
        estimator_bias_bound=jnp.full_like(means, jnp.nan),
        valid=valid,
        candidates=candidates,
        model_ids=model_ids,
        utility_target=utility_target,
        method_id=method_id,
        approximation=approximation,
        error_basis=(
            f"{error_qualifier}IID outer-sample standard error; finite-inner "
            "log-mixture bias is not bounded and must be assessed by increasing "
            "inner_sample_count"
        ),
        outer_sample_count=outer_count,
        inner_sample_count=inner_sample_count,
        unit_id="nat",
    )


def nested_monte_carlo_expected_utility(
    target_conditioned_log_probability_samples: ArrayLike,
    marginal_log_probability_samples: ArrayLike,
    /,
    *,
    candidates: Sequence[ExperimentalDesignCandidate],
    model_ids: Sequence[str],
    utility_target: ExpectedUtilityTarget,
) -> ExpectedUtilityResult:
    """Estimate ``E[log p(y|target) - log p(y)]`` by nested Monte Carlo.

    Arrays have shape ``(candidate, outer, inner)``. The conditioned and
    marginal inner draws must be independent of the outer joint draws. For a
    parameter target, the conditioned axis may have length one when the
    conditional log density is evaluated exactly. For predictive targets the
    caller must supply a scientifically justified ``p(y|prediction)`` mixture;
    a shared-parameter conditional-independence shortcut is not assumed here.
    """
    target = _utility_target(utility_target)
    candidate_values = _candidate_tuple(candidates)
    conditioned = jnp.asarray(target_conditioned_log_probability_samples, dtype=float)
    marginal = jnp.asarray(marginal_log_probability_samples, dtype=float)
    if conditioned.ndim != 3 or marginal.ndim != 3:
        raise ValueError("Nested log-probability samples must be rank-3 arrays.")
    if conditioned.shape[:2] != marginal.shape[:2]:
        raise ValueError("Conditioned and marginal candidate/outer axes must match.")
    if conditioned.shape[0] != len(candidate_values):
        raise ValueError("Nested samples do not align with candidates.")
    if conditioned.shape[2] < 1 or marginal.shape[2] < 1:
        raise ValueError("Nested estimators require non-empty inner sample axes.")
    allowed_conditioned = jnp.isfinite(conditioned) | jnp.isneginf(conditioned)
    allowed_marginal = jnp.isfinite(marginal) | jnp.isneginf(marginal)
    if bool(jnp.any(~allowed_conditioned)) or bool(jnp.any(~allowed_marginal)):
        raise ValueError("Log probabilities may be finite or negative infinity only.")
    contributions = _log_mean_exp(conditioned, 2) - _log_mean_exp(marginal, 2)
    return _nested_result(
        contributions,
        candidates=candidate_values,
        model_ids=model_ids,
        utility_target=target,
        method_id="nested_monte_carlo_expected_information_gain",
        approximation="nested_monte_carlo_finite_inner_log_mixture",
        inner_sample_count=max(conditioned.shape[2], marginal.shape[2]),
    )


def _position_sample_count(samples: PyTree[ArrayLike], /) -> tuple[PyTree[Array], int]:
    arrays = jax.tree_util.tree_map(jnp.asarray, samples)
    leaves = jax.tree_util.tree_leaves(arrays)
    if not leaves:
        raise ValueError("posterior_position_samples must contain array leaves.")
    if any(value.ndim < 1 for value in leaves):
        raise ValueError("Every posterior-position leaf requires a leading sample axis.")
    counts = {int(value.shape[0]) for value in leaves}
    if len(counts) != 1:
        raise ValueError("Posterior-position leaves must share their leading axis.")
    count = counts.pop()
    if count < 1:
        raise ValueError("posterior_position_samples must not be empty.")
    return arrays, count


def _tree_index(tree: PyTree[Array], index: int, /) -> PyTree[Array]:
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def _scalar_log_probability(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value, dtype=float)
    if result.ndim != 0:
        raise ValueError(f"{name} must return a scalar log probability.")
    if not bool(jnp.isfinite(result) | jnp.isneginf(result)):
        raise ValueError(f"{name} may return finite values or negative infinity only.")
    return result


def _candidate_tuple(
    candidates: Sequence[ExperimentalDesignCandidate],
    /,
) -> tuple[ExperimentalDesignCandidate, ...]:
    values = tuple(candidates)
    if not values or any(
        not isinstance(value, ExperimentalDesignCandidate) for value in values
    ):
        raise TypeError(
            "candidates must be a non-empty sequence of ExperimentalDesignCandidate."
        )
    identifiers = tuple(value.candidate_id for value in values)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Candidate IDs must be unique.")
    return values


def posterior_parameter_expected_utility(
    problem: PosteriorProblem,
    key: Array,
    posterior_position_samples: PyTree[ArrayLike],
    candidates: Sequence[ExperimentalDesignCandidate],
    observation_log_probability: Callable[
        [Any, PyTree[Array], ExperimentalDesignCandidate], ArrayLike
    ],
    /,
    *,
    model_ids: Sequence[str],
    num_outer_samples: int,
    num_inner_samples: int,
) -> ExpectedUtilityResult:
    """Estimate parameter information using ``PosteriorProblem`` observations."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    if problem.sample_observation_fn is None:
        raise ValueError("problem must declare sample_observation.")
    if not callable(observation_log_probability):
        raise TypeError("observation_log_probability must be callable.")
    positions, position_count = _position_sample_count(posterior_position_samples)
    candidate_values = _candidate_tuple(candidates)
    outer_count = _positive_integer(num_outer_samples, "num_outer_samples", minimum=2)
    inner_count = _positive_integer(num_inner_samples, "num_inner_samples")
    candidate_keys = jr.split(key, len(candidate_values))
    conditioned_rows: list[Array] = []
    marginal_rows: list[Array] = []
    for candidate, candidate_key in zip(candidate_values, candidate_keys, strict=True):
        outer_index_key, observation_key, inner_index_key = jr.split(candidate_key, 3)
        outer_indices = jr.randint(outer_index_key, (outer_count,), 0, position_count)
        observation_keys = jr.split(observation_key, outer_count)
        conditioned_values: list[Array] = []
        marginal_values: list[Array] = []
        for outer_number, (outer_index, draw_key) in enumerate(
            zip(outer_indices, observation_keys, strict=True)
        ):
            position = _tree_index(positions, int(outer_index))
            observation = problem.sample_observation(draw_key, position, candidate)
            conditioned_values.append(
                _scalar_log_probability(
                    observation_log_probability(observation, position, candidate),
                    "observation_log_probability",
                )
            )
            inner_key = jr.fold_in(inner_index_key, outer_number)
            inner_indices = jr.randint(inner_key, (inner_count,), 0, position_count)
            marginal_values.append(
                jnp.stack(
                    tuple(
                        _scalar_log_probability(
                            observation_log_probability(
                                observation,
                                _tree_index(positions, int(inner_index)),
                                candidate,
                            ),
                            "observation_log_probability",
                        )
                        for inner_index in inner_indices
                    )
                )
            )
        conditioned_rows.append(jnp.stack(tuple(conditioned_values))[:, None])
        marginal_rows.append(jnp.stack(tuple(marginal_values)))
    result = nested_monte_carlo_expected_utility(
        jnp.stack(tuple(conditioned_rows)),
        jnp.stack(tuple(marginal_rows)),
        candidates=candidate_values,
        model_ids=model_ids,
        utility_target="parameter",
    )
    return ExpectedUtilityResult(
        expected_utility=result.expected_utility,
        estimator_standard_error=result.estimator_standard_error,
        estimator_bias_bound=result.estimator_bias_bound,
        valid=result.valid,
        candidates=candidate_values,
        model_ids=model_ids,
        utility_target="parameter",
        method_id="posterior_problem_nested_parameter_information",
        approximation=(
            "empirical_posterior_pool_with_replacement; "
            "nested_monte_carlo_finite_inner_log_mixture"
        ),
        error_basis=(f"conditional on the supplied posterior pool; {result.error_basis}"),
        outer_sample_count=outer_count,
        inner_sample_count=inner_count,
        unit_id="nat",
    )


def posterior_predictive_expected_utility(
    problem: PosteriorProblem,
    key: Array,
    posterior_position_samples: PyTree[ArrayLike],
    candidates: Sequence[ExperimentalDesignCandidate],
    observation_log_probability: Callable[
        [Any, PyTree[Array], ExperimentalDesignCandidate], ArrayLike
    ],
    sample_prediction: Callable[[Array, PyTree[Array]], Any],
    prediction_log_probability: Callable[[Any, PyTree[Array]], ArrayLike],
    /,
    *,
    model_ids: Sequence[str],
    num_outer_samples: int,
    num_inner_samples: int,
) -> ExpectedUtilityResult:
    """Estimate information about a future prediction under one posterior.

    Acquisition observations and the target prediction must be conditionally
    independent given the posterior position. This routine estimates their
    joint mixture and both marginal mixtures; it does not equate parameter
    information with predictive information.
    """
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    if problem.sample_observation_fn is None:
        raise ValueError("problem must declare sample_observation.")
    for function, name in (
        (observation_log_probability, "observation_log_probability"),
        (sample_prediction, "sample_prediction"),
        (prediction_log_probability, "prediction_log_probability"),
    ):
        if not callable(function):
            raise TypeError(f"{name} must be callable.")
    positions, position_count = _position_sample_count(posterior_position_samples)
    candidate_values = _candidate_tuple(candidates)
    outer_count = _positive_integer(num_outer_samples, "num_outer_samples", minimum=2)
    inner_count = _positive_integer(num_inner_samples, "num_inner_samples")
    contribution_rows: list[Array] = []
    for candidate, candidate_key in zip(
        candidate_values, jr.split(key, len(candidate_values)), strict=True
    ):
        outer_key, observation_key, prediction_key, inner_key = jr.split(candidate_key, 4)
        outer_indices = jr.randint(outer_key, (outer_count,), 0, position_count)
        observation_keys = jr.split(observation_key, outer_count)
        prediction_keys = jr.split(prediction_key, outer_count)
        contributions: list[Array] = []
        for outer_number, (outer_index, y_key, z_key) in enumerate(
            zip(outer_indices, observation_keys, prediction_keys, strict=True)
        ):
            position = _tree_index(positions, int(outer_index))
            observation = problem.sample_observation(y_key, position, candidate)
            prediction = sample_prediction(z_key, position)
            indices = jr.randint(
                jr.fold_in(inner_key, outer_number),
                (inner_count,),
                0,
                position_count,
            )
            observation_logs: list[Array] = []
            prediction_logs: list[Array] = []
            for inner_index in indices:
                inner_position = _tree_index(positions, int(inner_index))
                observation_logs.append(
                    _scalar_log_probability(
                        observation_log_probability(
                            observation, inner_position, candidate
                        ),
                        "observation_log_probability",
                    )
                )
                prediction_logs.append(
                    _scalar_log_probability(
                        prediction_log_probability(prediction, inner_position),
                        "prediction_log_probability",
                    )
                )
            observation_values = jnp.stack(tuple(observation_logs))
            prediction_values = jnp.stack(tuple(prediction_logs))
            log_joint = _log_mean_exp(observation_values + prediction_values, 0)
            log_observation = _log_mean_exp(observation_values, 0)
            log_prediction = _log_mean_exp(prediction_values, 0)
            contributions.append(log_joint - log_observation - log_prediction)
        contribution_rows.append(jnp.stack(tuple(contributions)))
    return _nested_result(
        jnp.stack(tuple(contribution_rows)),
        candidates=candidate_values,
        model_ids=model_ids,
        utility_target="predictive",
        method_id="posterior_problem_nested_predictive_information",
        approximation=(
            "conditionally_independent_observation_and_prediction; "
            "empirical_posterior_pool_with_replacement; finite_inner_mixtures"
        ),
        inner_sample_count=inner_count,
        error_qualifier="conditional on the supplied posterior pool; ",
    )


def posterior_model_discrimination_expected_utility(
    problems: Sequence[PosteriorProblem],
    key: Array,
    posterior_position_samples: Sequence[PyTree[ArrayLike]],
    candidates: Sequence[ExperimentalDesignCandidate],
    observation_log_probabilities: Sequence[
        Callable[[Any, PyTree[Array], ExperimentalDesignCandidate], ArrayLike]
    ],
    model_probabilities: ArrayLike,
    /,
    *,
    model_ids: Sequence[str],
    num_outer_samples: int,
    num_inner_samples: int,
) -> ExpectedUtilityResult:
    """Estimate finite-model discrimination while integrating model parameters."""
    problem_values = tuple(problems)
    position_values = tuple(posterior_position_samples)
    log_probability_values = tuple(observation_log_probabilities)
    identifiers = _identifiers(model_ids, "model_ids")
    model_count = len(identifiers)
    if (
        len(problem_values) != model_count
        or len(position_values) != model_count
        or len(log_probability_values) != model_count
    ):
        raise ValueError(
            "Problems, position samples, log-probability functions, and model IDs "
            "must have identical lengths."
        )
    if any(not isinstance(problem, PosteriorProblem) for problem in problem_values):
        raise TypeError("Every problems entry must be a PosteriorProblem.")
    if any(problem.sample_observation_fn is None for problem in problem_values):
        raise ValueError("Every model problem must declare sample_observation.")
    if any(not callable(function) for function in log_probability_values):
        raise TypeError("Every observation log-probability entry must be callable.")
    prepared_positions: list[PyTree[Array]] = []
    position_counts: list[int] = []
    for samples in position_values:
        positions, count = _position_sample_count(samples)
        prepared_positions.append(positions)
        position_counts.append(count)
    probabilities = jnp.asarray(model_probabilities, dtype=float)
    if probabilities.shape != (model_count,):
        raise ValueError("model_probabilities must have one entry per model.")
    if bool(jnp.any(~jnp.isfinite(probabilities))) or bool(jnp.any(probabilities <= 0.0)):
        raise ValueError("Model probabilities must be finite and strictly positive.")
    tolerance = 64.0 * jnp.finfo(probabilities.dtype).eps * model_count
    if bool(jnp.abs(jnp.sum(probabilities) - 1.0) > tolerance):
        raise ValueError("model_probabilities must sum to one.")
    candidate_values = _candidate_tuple(candidates)
    outer_count = _positive_integer(num_outer_samples, "num_outer_samples", minimum=2)
    inner_count = _positive_integer(num_inner_samples, "num_inner_samples")
    log_model_probabilities = jnp.log(probabilities)
    contribution_rows: list[Array] = []
    for candidate, candidate_key in zip(
        candidate_values, jr.split(key, len(candidate_values)), strict=True
    ):
        (
            model_key,
            position_key,
            observation_key,
            numerator_key,
            marginal_key,
        ) = jr.split(candidate_key, 5)
        contributions: list[Array] = []
        for outer_number in range(outer_count):
            model_index = int(
                jr.categorical(
                    jr.fold_in(model_key, outer_number), log_model_probabilities
                )
            )
            outer_position_index = int(
                jr.randint(
                    jr.fold_in(position_key, outer_number),
                    (),
                    0,
                    position_counts[model_index],
                )
            )
            outer_position = _tree_index(
                prepared_positions[model_index], outer_position_index
            )
            observation = problem_values[model_index].sample_observation(
                jr.fold_in(observation_key, outer_number),
                outer_position,
                candidate,
            )
            numerator_indices = jr.randint(
                jr.fold_in(numerator_key, outer_number),
                (inner_count,),
                0,
                position_counts[model_index],
            )
            numerator_logs = jnp.stack(
                tuple(
                    _scalar_log_probability(
                        log_probability_values[model_index](
                            observation,
                            _tree_index(
                                prepared_positions[model_index], int(inner_index)
                            ),
                            candidate,
                        ),
                        "observation_log_probabilities entry",
                    )
                    for inner_index in numerator_indices
                )
            )
            log_conditioned = _log_mean_exp(numerator_logs, 0)
            model_marginal_logs: list[Array] = []
            for inner_model_index in range(model_count):
                indices = jr.randint(
                    jr.fold_in(jr.fold_in(marginal_key, outer_number), inner_model_index),
                    (inner_count,),
                    0,
                    position_counts[inner_model_index],
                )
                values = jnp.stack(
                    tuple(
                        _scalar_log_probability(
                            log_probability_values[inner_model_index](
                                observation,
                                _tree_index(
                                    prepared_positions[inner_model_index],
                                    int(inner_index),
                                ),
                                candidate,
                            ),
                            "observation_log_probabilities entry",
                        )
                        for inner_index in indices
                    )
                )
                model_marginal_logs.append(
                    log_model_probabilities[inner_model_index] + _log_mean_exp(values, 0)
                )
            log_marginal = logsumexp(jnp.stack(tuple(model_marginal_logs)))
            contributions.append(log_conditioned - log_marginal)
        contribution_rows.append(jnp.stack(tuple(contributions)))
    return _nested_result(
        jnp.stack(tuple(contribution_rows)),
        candidates=candidate_values,
        model_ids=identifiers,
        utility_target="model_discrimination",
        method_id="posterior_problem_nested_model_discrimination_information",
        approximation=(
            "finite_model_prior; empirical_within_model_posterior_pools; "
            "independent_conditioned_and_marginal_inner_mixtures"
        ),
        inner_sample_count=inner_count,
        error_qualifier="conditional on supplied within-model posterior pools; ",
    )


def _candidate_order(
    candidates: Sequence[ExperimentalDesignCandidate],
    /,
) -> tuple[
    tuple[ExperimentalDesignCandidate, ...],
    tuple[ExperimentalDesignCandidate, ...],
    tuple[int, ...],
]:
    original = _candidate_tuple(candidates)
    ordered = tuple(sorted(original, key=lambda candidate: candidate.candidate_id))
    original_indices = {
        candidate.candidate_id: index for index, candidate in enumerate(original)
    }
    order = tuple(original_indices[candidate.candidate_id] for candidate in ordered)
    setup_costs: dict[str, float] = {}
    for candidate in ordered:
        if candidate.setup_id:
            if (
                candidate.setup_id in setup_costs
                and setup_costs[candidate.setup_id] != candidate.setup_cost
            ):
                raise ValueError(
                    "Candidates sharing setup_id must declare the same setup_cost."
                )
            setup_costs[candidate.setup_id] = candidate.setup_cost
    return original, ordered, order


def _pairwise_matrix(
    value: ArrayLike | None,
    size: int,
    order: tuple[int, ...],
    name: str,
    /,
) -> Array:
    if value is None:
        return jnp.zeros((size, size), dtype=float)
    matrix = jnp.asarray(value, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have one square entry per candidate.")
    if bool(jnp.any(~jnp.isfinite(matrix))) or bool(jnp.any(matrix < 0.0)):
        raise ValueError(f"{name} must be finite and non-negative.")
    tolerance = 64.0 * jnp.finfo(matrix.dtype).eps * max(1, size)
    if bool(jnp.any(jnp.abs(matrix - matrix.T) > tolerance)):
        raise ValueError(f"{name} must be symmetric.")
    if bool(jnp.any(jnp.abs(jnp.diag(matrix)) > tolerance)):
        raise ValueError(f"{name} must have a zero diagonal.")
    indices = jnp.asarray(order, dtype=int)
    return matrix[indices[:, None], indices[None, :]]


def _batch_cost(
    selected: tuple[int, ...],
    candidates: tuple[ExperimentalDesignCandidate, ...],
    /,
) -> float:
    candidate_cost = sum(candidates[index].cost for index in selected)
    setup_costs = {
        candidates[index].setup_id: candidates[index].setup_cost
        for index in selected
        if candidates[index].setup_id
    }
    return candidate_cost + sum(setup_costs.values())


def _feasible_batch(
    selected: tuple[int, ...],
    candidates: tuple[ExperimentalDesignCandidate, ...],
    constraints: ExperimentalBatchConstraints,
    /,
) -> tuple[bool, float]:
    selected_ids = {candidates[index].candidate_id for index in selected}
    size = len(selected)
    cost = _batch_cost(selected, candidates)
    tolerance = 64.0 * jnp.finfo(float).eps * max(1.0, constraints.budget)
    if (
        size < constraints.minimum_batch_size
        or size > constraints.maximum_batch_size
        or cost > constraints.budget + float(tolerance)
        or not set(constraints.required_candidate_ids).issubset(selected_ids)
    ):
        return False, cost
    if constraints.allowed_feasibility_groups and any(
        candidates[index].feasibility_group not in constraints.allowed_feasibility_groups
        for index in selected
    ):
        return False, cost
    if any(
        len(selected_ids.intersection(group)) > 1
        for group in constraints.mutually_exclusive_candidate_groups
    ):
        return False, cost
    diversity_counts: dict[str, int] = {}
    for index in selected:
        group = candidates[index].diversity_group
        diversity_counts[group] = diversity_counts.get(group, 0) + 1
    if len(diversity_counts) < constraints.minimum_diversity_groups:
        return False, cost
    if constraints.maximum_per_diversity_group is not None and any(
        count > constraints.maximum_per_diversity_group
        for count in diversity_counts.values()
    ):
        return False, cost
    return True, cost


def select_experimental_batch(
    candidates: Sequence[ExperimentalDesignCandidate],
    utility: ExpectedUtilityResult,
    constraints: ExperimentalBatchConstraints,
    /,
    *,
    objective_id: str,
    model_ids: Sequence[str],
    analysis_id: str,
    pairwise_redundancy: ArrayLike | None = None,
    redundancy_weight: float = 1.0,
    pairwise_diversity: ArrayLike | None = None,
    diversity_weight: float = 0.0,
) -> ExperimentalBatchPlan:
    """Select the exact best modest batch with stable deterministic tie-breaking."""
    if not isinstance(utility, ExpectedUtilityResult):
        raise TypeError("utility must be an ExpectedUtilityResult.")
    if not isinstance(constraints, ExperimentalBatchConstraints):
        raise TypeError("constraints must be ExperimentalBatchConstraints.")
    original, ordered, order = _candidate_order(candidates)
    if len(ordered) > _EXACT_SELECTION_LIMIT:
        raise ValueError(
            f"Exact deterministic selection supports at most {_EXACT_SELECTION_LIMIT} "
            "candidates; use an explicitly qualified campaign-scale optimizer."
        )
    if set(utility.candidate_ids) != {candidate.candidate_id for candidate in original}:
        raise ValueError("Utility candidate IDs must exactly match the candidate panel.")
    utility_bindings = {
        candidate_id: (content_id, source_id)
        for candidate_id, content_id, source_id in zip(
            utility.candidate_ids,
            utility.candidate_content_ids,
            utility.prediction_source_ids,
            strict=True,
        )
    }
    if any(
        utility_bindings[candidate.candidate_id]
        != (candidate.candidate_content_id, candidate.prediction_source_id)
        for candidate in original
    ):
        raise ValueError(
            "Utility candidate content and prediction sources must exactly match "
            "the candidate panel."
        )
    models = tuple(sorted(_identifiers(model_ids, "model_ids")))
    if models != utility.model_ids:
        raise ValueError(
            "Selected model IDs must exactly match the utility model sources."
        )
    utility_index = {
        identifier: index for index, identifier in enumerate(utility.candidate_ids)
    }
    raw_values = jnp.asarray(
        [
            utility.expected_utility[utility_index[candidate.candidate_id]]
            for candidate in ordered
        ]
    )
    mandatory_control = jnp.asarray(
        [candidate.mandatory_control for candidate in ordered], dtype=bool
    )
    values = jnp.where(mandatory_control, 0.0, raw_values)
    valid = jnp.asarray(
        [utility.valid[utility_index[candidate.candidate_id]] for candidate in ordered]
    )
    required_ids = set(constraints.required_candidate_ids)
    required_ids.update(
        candidate.candidate_id for candidate in ordered if candidate.mandatory_control
    )
    panel_ids = {candidate.candidate_id for candidate in ordered}
    if not required_ids.issubset(panel_ids):
        raise ValueError("Required candidate IDs must belong to the candidate panel.")
    exclusion_ids = {
        candidate_id
        for group in constraints.mutually_exclusive_candidate_groups
        for candidate_id in group
    }
    if not exclusion_ids.issubset(panel_ids):
        raise ValueError(
            "Mutual exclusion groups may reference only frozen panel candidates."
        )
    if any(
        not candidate.mandatory_control and not bool(valid[index])
        for index, candidate in enumerate(ordered)
        if candidate.candidate_id in required_ids
    ):
        raise ValueError("Every required non-control candidate needs valid utility.")
    effective_constraints = ExperimentalBatchConstraints(
        constraints.budget,
        constraints.maximum_batch_size,
        minimum_batch_size=constraints.minimum_batch_size,
        required_candidate_ids=tuple(sorted(required_ids)),
        mutually_exclusive_candidate_groups=(
            constraints.mutually_exclusive_candidate_groups
        ),
        allowed_feasibility_groups=constraints.allowed_feasibility_groups,
        minimum_diversity_groups=constraints.minimum_diversity_groups,
        maximum_per_diversity_group=constraints.maximum_per_diversity_group,
    )
    redundancy = _pairwise_matrix(
        pairwise_redundancy,
        len(ordered),
        order,
        "pairwise_redundancy",
    )
    diversity = _pairwise_matrix(
        pairwise_diversity,
        len(ordered),
        order,
        "pairwise_diversity",
    )
    redundancy_scale = _nonnegative_finite(redundancy_weight, "redundancy_weight")
    diversity_scale = _nonnegative_finite(diversity_weight, "diversity_weight")
    eligible = tuple(
        index
        for index, candidate in enumerate(ordered)
        if bool(valid[index]) or candidate.mandatory_control
    )
    best_selected: tuple[int, ...] | None = None
    best_score = float("-inf")
    best_cost = float("inf")
    best_ids: tuple[str, ...] | None = None
    maximum_size = min(effective_constraints.maximum_batch_size, len(eligible))
    for size in range(effective_constraints.minimum_batch_size, maximum_size + 1):
        for selected in combinations(eligible, size):
            feasible, cost = _feasible_batch(selected, ordered, effective_constraints)
            if not feasible:
                continue
            indices = jnp.asarray(selected, dtype=int)
            score = float(jnp.sum(values[indices]))
            if len(selected) > 1:
                selected_redundancy = redundancy[indices[:, None], indices[None, :]]
                selected_diversity = diversity[indices[:, None], indices[None, :]]
                score -= redundancy_scale * float(
                    jnp.sum(jnp.triu(selected_redundancy, k=1))
                )
                score += diversity_scale * float(
                    jnp.sum(jnp.triu(selected_diversity, k=1))
                )
            identifiers = tuple(ordered[index].candidate_id for index in selected)
            if score > best_score or (
                score == best_score
                and (
                    cost < best_cost
                    or (
                        cost == best_cost and (best_ids is None or identifiers < best_ids)
                    )
                )
            ):
                best_selected = selected
                best_score = score
                best_cost = cost
                best_ids = identifiers
    if best_selected is None or best_ids is None:
        raise ValueError("No batch satisfies all declared experimental constraints.")
    selection_policy_id = canonical_fingerprint(
        {
            "kind": "exact-modest-experimental-batch-selection-v2",
            "constraints_id": effective_constraints.constraints_id,
            "estimator_method_id": utility.method_id,
            "estimator_approximation": utility.approximation,
            "candidate_order": [candidate.candidate_id for candidate in ordered],
            "candidate_content_ids": [
                candidate.candidate_content_id for candidate in ordered
            ],
            "candidate_prediction_source_ids": [
                candidate.prediction_source_id for candidate in ordered
            ],
            "model_ids": list(models),
            "mandatory_control_utility_contribution": "zero",
            "pairwise_redundancy": array_tree_fingerprint(redundancy),
            "redundancy_weight": redundancy_scale.hex(),
            "pairwise_diversity": array_tree_fingerprint(diversity),
            "utility_target": utility.utility_target,
            "utility_unit": utility.unit_id,
            "utility_error_basis": utility.error_basis,
            "utility_outer_sample_count": utility.outer_sample_count,
            "utility_inner_sample_count": utility.inner_sample_count,
            "utility_values": array_tree_fingerprint(
                (
                    utility.expected_utility,
                    utility.estimator_standard_error,
                    utility.estimator_bias_bound,
                    utility.valid,
                )
            ),
            "diversity_weight": diversity_scale.hex(),
            "tie_break": "objective_then_lower_cost_then_lexicographic_candidate_ids",
        }
    )
    return ExperimentalBatchPlan(
        best_ids,
        objective_id,
        effective_constraints.budget,
        models,
        candidate_ids=tuple(candidate.candidate_id for candidate in ordered),
        candidate_content_ids=tuple(
            candidate.candidate_content_id for candidate in ordered
        ),
        analysis_id=analysis_id,
        constraints_id=effective_constraints.constraints_id,
        estimator_method_id=utility.method_id,
        estimator_approximation=utility.approximation,
        selection_policy_id=selection_policy_id,
        planned_total_cost=best_cost,
        objective_value=best_score,
    )


def _score_result(
    scores: ArrayLike,
    candidates: tuple[ExperimentalDesignCandidate, ...],
    /,
    *,
    method_id: str,
    model_ids: Sequence[str],
) -> ExpectedUtilityResult:
    values = jnp.asarray(scores, dtype=float)
    if values.shape != (len(candidates),):
        raise ValueError(f"{method_id} scores must have one entry per candidate.")
    valid = jnp.isfinite(values)
    return ExpectedUtilityResult(
        expected_utility=jnp.where(valid, values, jnp.nan),
        estimator_standard_error=jnp.zeros_like(values),
        estimator_bias_bound=jnp.full_like(values, jnp.nan),
        valid=valid,
        candidates=candidates,
        model_ids=model_ids,
        utility_target="predictive",
        method_id=method_id,
        approximation="retrospective_ranking_score_not_expected_information",
        error_basis="no estimator uncertainty claimed for supplied ranking scores",
        outer_sample_count=0,
        inner_sample_count=0,
        unit_id="ranking_score",
    )


def evaluate_retrospective_design(
    candidates: Sequence[ExperimentalDesignCandidate],
    constraints: ExperimentalBatchConstraints,
    proposed_utility: ExpectedUtilityResult,
    /,
    *,
    random_key: Array,
    space_filling_distances: ArrayLike,
    uncertainty_scores: ArrayLike,
    domain_heuristic_scores: ArrayLike,
    realized_utility: Callable[[tuple[str, ...]], ArrayLike],
    metric_id: str,
    model_ids: Sequence[str],
    analysis_id: str,
    objective_id: str,
    proposed_pairwise_redundancy: ArrayLike | None = None,
    proposed_redundancy_weight: float = 1.0,
) -> RetrospectiveDesignResult:
    """Replay fixed baselines under one budget ceiling and normalize by cost.

    ``realized_utility`` is evaluated only after every plan is frozen and should
    return the realized reduction in predictive loss, model ambiguity, or a
    declared decision loss. Raw costs and batch sizes remain recorded; the
    cross-strategy comparison is explicitly per realized planned cost.
    """
    candidate_values = _candidate_tuple(candidates)
    if not isinstance(constraints, ExperimentalBatchConstraints):
        raise TypeError("constraints must be ExperimentalBatchConstraints.")
    if not isinstance(proposed_utility, ExpectedUtilityResult):
        raise TypeError("proposed_utility must be an ExpectedUtilityResult.")
    if not callable(realized_utility):
        raise TypeError("realized_utility must be callable.")
    distances = _pairwise_matrix(
        space_filling_distances,
        len(candidate_values),
        tuple(range(len(candidate_values))),
        "space_filling_distances",
    )
    random_scores = jr.uniform(random_key, (len(candidate_values),))
    random_result = _score_result(
        random_scores,
        candidate_values,
        method_id="common_budget_ceiling_random_ranking",
        model_ids=model_ids,
    )
    space_result = _score_result(
        jnp.ones((len(candidate_values),), dtype=float),
        candidate_values,
        method_id="common_budget_ceiling_space_filling",
        model_ids=model_ids,
    )
    uncertainty_result = _score_result(
        uncertainty_scores,
        candidate_values,
        method_id="common_budget_ceiling_uncertainty_only",
        model_ids=model_ids,
    )
    heuristic_result = _score_result(
        domain_heuristic_scores,
        candidate_values,
        method_id="common_budget_ceiling_domain_heuristic",
        model_ids=model_ids,
    )
    maximum_distance = float(jnp.max(distances))
    normalized_distances = (
        distances if maximum_distance == 0.0 else distances / maximum_distance
    )
    space_diversity_weight = 1.0 / (2.0 * max(1, constraints.maximum_batch_size) ** 2)
    plan_inputs = (
        ("random", random_result, None, 0.0, None, 0.0),
        (
            "space_filling",
            space_result,
            None,
            0.0,
            normalized_distances,
            space_diversity_weight,
        ),
        ("uncertainty_only", uncertainty_result, None, 0.0, None, 0.0),
        ("domain_heuristic", heuristic_result, None, 0.0, None, 0.0),
        (
            "proposed_design",
            proposed_utility,
            proposed_pairwise_redundancy,
            proposed_redundancy_weight,
            None,
            0.0,
        ),
    )
    plans: list[ExperimentalBatchPlan] = []
    for (
        strategy,
        result,
        redundancy,
        redundancy_weight,
        diversity,
        diversity_weight,
    ) in plan_inputs:
        plans.append(
            select_experimental_batch(
                candidate_values,
                result,
                constraints,
                objective_id=f"{objective_id}:{strategy}",
                model_ids=model_ids,
                analysis_id=analysis_id,
                pairwise_redundancy=redundancy,
                redundancy_weight=redundancy_weight,
                pairwise_diversity=diversity,
                diversity_weight=diversity_weight,
            )
        )
    realized_values: list[Array] = []
    realized_validity: list[Array] = []
    for plan in plans:
        value = jnp.asarray(realized_utility(plan.selected_candidate_ids), dtype=float)
        if value.ndim != 0:
            raise ValueError("realized_utility must return one scalar per plan.")
        valid = jnp.isfinite(value)
        realized_values.append(jnp.where(valid, value, jnp.nan))
        realized_validity.append(valid)
    return RetrospectiveDesignResult(
        plans,
        jnp.stack(tuple(realized_values)),
        jnp.stack(tuple(realized_validity)),
        metric_id=metric_id,
    )


__all__ = [
    "ExpectedUtilityTarget",
    "RetrospectiveDesignStrategy",
    "ExperimentalDesignCandidate",
    "ExpectedUtilityResult",
    "ExperimentalBatchConstraints",
    "ExperimentalBatchPlan",
    "RetrospectiveDesignResult",
    "exact_finite_expected_utility",
    "nested_monte_carlo_expected_utility",
    "posterior_parameter_expected_utility",
    "posterior_predictive_expected_utility",
    "posterior_model_discrimination_expected_utility",
    "select_experimental_batch",
    "evaluate_retrospective_design",
]
