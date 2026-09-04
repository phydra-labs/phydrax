#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sampled constraint and KKT evaluation for mean-field-game candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...stochastic import EmpiricalMeanField
from ._constraints import (
    GameConstraintBlock,
    GameConstraintLayout,
    GameConstraintScope,
    GameFeasibilityEvidence,
    GameMultiplierLayout,
    OpenLoopGameConstraints,
)
from ._mean_field import FrozenLawBestResponseResult
from ._mean_field_fixed_point import (
    MeanFieldGameFixedPointPlan,
    MeanFieldGameFixedPointProblem,
    MeanFieldGameFixedPointResult,
    MeanFieldGameFixedPointStatus,
    solve_mean_field_game_fixed_point,
)


CONSTRAINED_MEAN_FIELD_GAME_KKT_CANDIDATE = "CONSTRAINED_MEAN_FIELD_GAME_KKT_CANDIDATE"


class MeanFieldConstraintConcept(IntEnum):
    """Stable constrained-MFG concepts and their aggregate multiplier convention."""

    INDIVIDUAL = 0
    AGGREGATE_GENERIC = 1
    AGGREGATE_VARIATIONAL = 2


class ConstrainedMeanFieldGameStatus(IntEnum):
    """Stable outcomes for constrained mean-field-game candidate evaluation."""

    SUCCESS = 0
    INVALID_BEST_RESPONSE = 1
    INVALID_INDUCED_LAW = 2
    LOW_EFFECTIVE_SAMPLE_SIZE = 3
    INVALID_LAW_DISTANCE = 4
    INVALID_INDIVIDUAL_EVIDENCE = 5
    INDIVIDUAL_INFEASIBLE = 6
    INVALID_AGGREGATE_RESIDUAL = 7
    POPULATION_INFEASIBLE = 8
    INVALID_MULTIPLIERS = 9
    DUAL_INFEASIBLE = 10
    COMPLEMENTARITY_FAILURE = 11
    INDIVIDUAL_KKT_FAILURE = 12
    MAX_ITERATIONS = 13
    INVALID_AGGREGATE_DERIVATIVE_EVIDENCE = 14
    INVALID_LAW_MIXTURE = 15


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    result = tuple(_identifier(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _nonnegative_tolerance(value: float, name: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


class MeanFieldIndividualConstraintEvidence(StrictModule):
    """Original-scale feasibility and stationarity for one frozen-law response.

    ``feasibility`` is the existing sampled open-loop feasibility evidence for the
    individual (non-shared) constraint blocks. ``original_stationarity`` is the
    original-equation representative-agent stationarity vector, including any
    individual-constraint multiplier terms but excluding aggregate-law multiplier
    terms. Aggregate primal, dual, complementarity, and derivative evidence are
    composed by the outer evaluator because they depend on the newly induced law
    and the exact aggregate multiplier vector.
    """

    feasibility: GameFeasibilityEvidence
    original_stationarity: Array
    original_stationarity_residual: Array
    valid: Array
    finite: Array
    best_response_flow_id: str = eqx.field(static=True)
    best_response_path_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    feasibility_scope: str = eqx.field(static=True)
    sampled_only: bool = eqx.field(static=True)
    certified: bool = eqx.field(static=True)

    def __init__(
        self,
        feasibility: GameFeasibilityEvidence,
        original_stationarity: ArrayLike,
        /,
        *,
        best_response_flow_id: str,
        best_response_path_id: str,
        evidence_id: str,
        valid: ArrayLike = True,
    ):
        if not isinstance(feasibility, GameFeasibilityEvidence):
            raise TypeError("feasibility must be GameFeasibilityEvidence.")
        if feasibility.case_shape != ():
            raise ValueError(
                "Mean-field individual feasibility evidence must have scalar case shape."
            )
        stationarity = jnp.asarray(original_stationarity)
        if jnp.issubdtype(stationarity.dtype, jnp.complexfloating):
            raise TypeError("original_stationarity must be real-valued.")
        if stationarity.ndim == 0:
            stationarity = stationarity.reshape((1,))
        if stationarity.ndim != 1 or stationarity.size == 0:
            raise ValueError("original_stationarity must be a nonempty vector.")
        stationarity_residual = jnp.max(jnp.abs(stationarity))
        validity = jnp.asarray(valid, dtype=bool)
        if validity.shape != ():
            raise ValueError("valid must be scalar.")
        finite = (
            bool(feasibility.finite)
            and bool(jnp.all(jnp.isfinite(stationarity)))
            and bool(validity)
        )
        self.feasibility = feasibility
        self.original_stationarity = stationarity
        self.original_stationarity_residual = stationarity_residual
        self.valid = validity
        self.finite = jnp.asarray(finite)
        self.best_response_flow_id = _identifier(
            best_response_flow_id, "best_response_flow_id"
        )
        self.best_response_path_id = _identifier(
            best_response_path_id, "best_response_path_id"
        )
        self.evidence_id = _identifier(evidence_id, "evidence_id")
        self.feasibility_scope = feasibility.feasibility_scope
        self.sampled_only = feasibility.sampled_only
        self.certified = False


class MeanFieldAggregateConstraintDerivativeEvidence(StrictModule):
    """Identified aggregate-constraint Jacobian at one law and price vector.

    Rows of ``aggregate_jacobian`` follow the complete declared multiplier layout.
    Rows belonging to non-shared constraints must be zero because their multiplier
    terms are already included in ``original_stationarity``. Each column follows
    the representative-agent stationarity coordinates.
    """

    aggregate_jacobian: Array
    multipliers: Array
    valid: Array
    finite: Array
    best_response_flow_id: str = eqx.field(static=True)
    best_response_path_id: str = eqx.field(static=True)
    induced_flow_id: str = eqx.field(static=True)
    aggregate_law_constraints_id: str = eqx.field(static=True)
    multiplier_ids: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    certified: bool = eqx.field(static=True)

    def __init__(
        self,
        aggregate_jacobian: ArrayLike,
        multipliers: ArrayLike,
        /,
        *,
        best_response_flow_id: str,
        best_response_path_id: str,
        induced_flow_id: str,
        aggregate_law_constraints_id: str,
        multiplier_ids: Sequence[str],
        evidence_id: str,
        valid: ArrayLike = True,
    ):
        jacobian = jnp.asarray(aggregate_jacobian)
        prices = jnp.asarray(multipliers)
        if jnp.issubdtype(jacobian.dtype, jnp.complexfloating):
            raise TypeError("aggregate_jacobian must be real-valued.")
        if jnp.issubdtype(prices.dtype, jnp.complexfloating):
            raise TypeError("multipliers must be real-valued.")
        if jacobian.ndim != 2 or jacobian.shape[1] == 0:
            raise ValueError("aggregate_jacobian must be a nonempty matrix.")
        if prices.ndim != 1 or jacobian.shape[0] != prices.size:
            raise ValueError("aggregate_jacobian rows must match the multiplier vector.")
        validity = jnp.asarray(valid, dtype=bool)
        if validity.shape != ():
            raise ValueError("valid must be scalar.")
        finite = bool(
            jnp.all(jnp.isfinite(jacobian)) & jnp.all(jnp.isfinite(prices)) & validity
        )
        self.aggregate_jacobian = jacobian
        self.multipliers = prices
        self.valid = validity
        self.finite = jnp.asarray(finite)
        self.best_response_flow_id = _identifier(
            best_response_flow_id, "best_response_flow_id"
        )
        self.best_response_path_id = _identifier(
            best_response_path_id, "best_response_path_id"
        )
        self.induced_flow_id = _identifier(induced_flow_id, "induced_flow_id")
        self.aggregate_law_constraints_id = _identifier(
            aggregate_law_constraints_id, "aggregate_law_constraints_id"
        )
        self.multiplier_ids = _identifiers(multiplier_ids, "multiplier_ids")
        self.evidence_id = _identifier(evidence_id, "evidence_id")
        self.certified = False


class ConstrainedMeanFieldGameProblem(StrictModule):
    """A state-law MFG fixed point with explicitly classified constraints.

    Non-shared ``GameConstraintBlock`` values are individual constraints and are
    checked through ``individual_evidence(response, args)``. Shared blocks are
    aggregate law constraints and are evaluated exactly once through
    ``aggregate_law_residuals(induced_flow, args)``. The latter callback returns
    one raw equality or ``residual <= 0`` array per aggregate block, in declaration
    order and with the corresponding ``GameConstraintLayout`` output shape.

    ``aggregate_derivative_evidence(response, induced_flow, multipliers, args)``
    must return the identified aggregate Jacobian evaluated at that exact induced
    law and complete multiplier vector. Its rows follow ``multiplier_layout`` so
    the outer evaluator can assemble the complete original-equation stationarity
    vector before accepting a KKT candidate.

    ``multipliers(response, induced_flow, args)`` returns one flat vector in the
    supplied ``multiplier_layout``. Generic aggregate constraints require the
    nonvariational layout, which allocates a separate shared-row copy to every
    participating population. Variational aggregate constraints require the
    variational layout, which allocates exactly one common shared-row copy.
    """

    fixed_point_problem: MeanFieldGameFixedPointProblem
    constraints: OpenLoopGameConstraints
    individual_constraints: OpenLoopGameConstraints
    aggregate_constraints: OpenLoopGameConstraints
    constraint_layout: GameConstraintLayout
    individual_constraint_layout: GameConstraintLayout
    aggregate_constraint_layout: GameConstraintLayout
    multiplier_layout: GameMultiplierLayout
    individual_evidence: (
        Callable[
            [FrozenLawBestResponseResult, Any], MeanFieldIndividualConstraintEvidence
        ]
        | None
    ) = eqx.field(static=True)
    aggregate_law_residuals: (
        Callable[[EmpiricalMeanField, Any], Sequence[ArrayLike]] | None
    ) = eqx.field(static=True)
    aggregate_derivative_evidence: (
        Callable[
            [
                FrozenLawBestResponseResult,
                EmpiricalMeanField,
                Array,
                Any,
            ],
            MeanFieldAggregateConstraintDerivativeEvidence,
        ]
        | None
    ) = eqx.field(static=True)
    multipliers: (
        Callable[[FrozenLawBestResponseResult, EmpiricalMeanField, Any], ArrayLike] | None
    ) = eqx.field(static=True)
    concept: MeanFieldConstraintConcept = eqx.field(static=True)
    multiplier_ids: tuple[str, ...] = eqx.field(static=True)
    individual_evidence_id: str | None = eqx.field(static=True)
    aggregate_law_constraints_id: str | None = eqx.field(static=True)
    aggregate_derivative_evidence_id: str | None = eqx.field(static=True)
    multiplier_callback_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        fixed_point_problem: MeanFieldGameFixedPointProblem,
        constraints: OpenLoopGameConstraints,
        /,
        *,
        concept: MeanFieldConstraintConcept,
        multiplier_layout: GameMultiplierLayout,
        multiplier_ids: Sequence[str] = (),
        individual_evidence: Callable[
            [FrozenLawBestResponseResult, Any],
            MeanFieldIndividualConstraintEvidence,
        ]
        | None = None,
        aggregate_law_residuals: Callable[[EmpiricalMeanField, Any], Sequence[ArrayLike]]
        | None = None,
        aggregate_derivative_evidence: Callable[
            [
                FrozenLawBestResponseResult,
                EmpiricalMeanField,
                Array,
                Any,
            ],
            MeanFieldAggregateConstraintDerivativeEvidence,
        ]
        | None = None,
        multipliers: Callable[
            [FrozenLawBestResponseResult, EmpiricalMeanField, Any], ArrayLike
        ]
        | None = None,
        individual_evidence_id: str | None = None,
        aggregate_law_constraints_id: str | None = None,
        aggregate_derivative_evidence_id: str | None = None,
        multiplier_callback_id: str | None = None,
        problem_id: str,
    ):
        if not isinstance(fixed_point_problem, MeanFieldGameFixedPointProblem):
            raise TypeError(
                "fixed_point_problem must be a MeanFieldGameFixedPointProblem."
            )
        if not isinstance(constraints, OpenLoopGameConstraints):
            raise TypeError("constraints must be OpenLoopGameConstraints.")
        if not isinstance(concept, MeanFieldConstraintConcept):
            raise TypeError("concept must be a MeanFieldConstraintConcept.")
        if not isinstance(multiplier_layout, GameMultiplierLayout):
            raise TypeError("multiplier_layout must be a GameMultiplierLayout.")

        num_path_sites = int(fixed_point_problem.initial_flow.times.size) - 1
        layout = constraints.layout(num_path_sites=num_path_sites)
        individual_blocks = tuple(
            block
            for block in constraints.blocks
            if block.scope is not GameConstraintScope.SHARED
        )
        aggregate_blocks = tuple(
            block
            for block in constraints.blocks
            if block.scope is GameConstraintScope.SHARED
        )
        individual_constraints = OpenLoopGameConstraints(
            constraints.partition, individual_blocks
        )
        aggregate_constraints = OpenLoopGameConstraints(
            constraints.partition, aggregate_blocks
        )
        individual_layout = individual_constraints.layout(num_path_sites=num_path_sites)
        aggregate_layout = aggregate_constraints.layout(num_path_sites=num_path_sites)

        if concept is MeanFieldConstraintConcept.INDIVIDUAL:
            if aggregate_blocks:
                raise ValueError(
                    "INDIVIDUAL constrained MFGs cannot contain aggregate shared "
                    "constraint blocks."
                )
            variational = False
        else:
            if not aggregate_blocks:
                raise ValueError(
                    "Aggregate constrained MFG concepts require at least one "
                    "shared constraint block."
                )
            variational = concept is MeanFieldConstraintConcept.AGGREGATE_VARIATIONAL
        expected_multiplier_layout = layout.multiplier_layout(variational=variational)
        if (
            multiplier_layout.constraint_layout.layout_id != layout.layout_id
            or multiplier_layout.layout_id != expected_multiplier_layout.layout_id
            or multiplier_layout.variational != variational
        ):
            convention = "common variational" if variational else "population-specific"
            raise ValueError(
                "multiplier_layout does not match the declared constraint concept; "
                f"expected the {convention} layout."
            )

        has_constraints = bool(constraints.blocks)
        if has_constraints:
            if not callable(individual_evidence):
                raise TypeError(
                    "Constrained problems require an individual_evidence callback "
                    "for frozen best-response feasibility/KKT evidence."
                )
            if not callable(multipliers):
                raise TypeError("Constrained problems require a multipliers callback.")
            evidence_identifier = _identifier(
                individual_evidence_id, "individual_evidence_id"
            )
            multiplier_callback_identifier = _identifier(
                multiplier_callback_id, "multiplier_callback_id"
            )
        else:
            if individual_evidence is not None or multipliers is not None:
                raise ValueError(
                    "Unconstrained problems must not supply constraint evidence or "
                    "multiplier callbacks."
                )
            if individual_evidence_id is not None or multiplier_callback_id is not None:
                raise ValueError(
                    "Unconstrained problems must not supply constraint callback IDs."
                )
            evidence_identifier = None
            multiplier_callback_identifier = None

        if aggregate_blocks:
            if not callable(aggregate_law_residuals):
                raise TypeError(
                    "Aggregate constraints require an aggregate_law_residuals callback."
                )
            if not callable(aggregate_derivative_evidence):
                raise TypeError(
                    "Aggregate constraints require an "
                    "aggregate_derivative_evidence callback."
                )
            aggregate_identifier = _identifier(
                aggregate_law_constraints_id,
                "aggregate_law_constraints_id",
            )
            aggregate_derivative_identifier = _identifier(
                aggregate_derivative_evidence_id,
                "aggregate_derivative_evidence_id",
            )
        else:
            if (
                aggregate_law_residuals is not None
                or aggregate_derivative_evidence is not None
            ):
                raise ValueError(
                    "Aggregate residual and derivative callbacks require aggregate "
                    "shared constraint blocks."
                )
            if (
                aggregate_law_constraints_id is not None
                or aggregate_derivative_evidence_id is not None
            ):
                raise ValueError("Aggregate callback IDs require aggregate constraints.")
            aggregate_identifier = None
            aggregate_derivative_identifier = None

        identifiers = _identifiers(multiplier_ids, "multiplier_ids")
        if len(identifiers) != multiplier_layout.num_multipliers:
            raise ValueError(
                "multiplier_ids must identify every multiplier slot in the declared "
                "layout."
            )

        self.fixed_point_problem = fixed_point_problem
        self.constraints = constraints
        self.individual_constraints = individual_constraints
        self.aggregate_constraints = aggregate_constraints
        self.constraint_layout = layout
        self.individual_constraint_layout = individual_layout
        self.aggregate_constraint_layout = aggregate_layout
        self.multiplier_layout = multiplier_layout
        self.individual_evidence = individual_evidence
        self.aggregate_law_residuals = aggregate_law_residuals
        self.aggregate_derivative_evidence = aggregate_derivative_evidence
        self.multipliers = multipliers
        self.concept = concept
        self.multiplier_ids = identifiers
        self.individual_evidence_id = evidence_identifier
        self.aggregate_law_constraints_id = aggregate_identifier
        self.aggregate_derivative_evidence_id = aggregate_derivative_identifier
        self.multiplier_callback_id = multiplier_callback_identifier
        self.problem_id = _identifier(problem_id, "problem_id")

    @property
    def population_ids(self) -> tuple[str, ...]:
        return self.constraints.partition.player_ids

    @property
    def individual_constraint_ids(self) -> tuple[str, ...]:
        return self.individual_constraint_layout.block_ids

    @property
    def aggregate_constraint_ids(self) -> tuple[str, ...]:
        return self.aggregate_constraint_layout.block_ids


class ConstrainedMeanFieldGamePlan(StrictModule):
    """Fixed outer capacity and acceptance thresholds for constrained candidates."""

    maximum_iterations: int = eqx.field(static=True)
    consistency_tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    minimum_effective_sample_size: float = eqx.field(static=True)
    feasibility_tolerance: float = eqx.field(static=True)
    kkt_tolerance: float = eqx.field(static=True)
    dual_feasibility_tolerance: float = eqx.field(static=True)
    complementarity_tolerance: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int,
        consistency_tolerance: float,
        feasibility_tolerance: float,
        kkt_tolerance: float,
        damping: float = 1.0,
        minimum_effective_sample_size: float = 2.0,
        dual_feasibility_tolerance: float | None = None,
        complementarity_tolerance: float | None = None,
        problem_id: str,
    ):
        if not isinstance(maximum_iterations, int) or maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be a positive integer.")
        consistency = _nonnegative_tolerance(
            consistency_tolerance, "consistency_tolerance"
        )
        feasibility = _nonnegative_tolerance(
            feasibility_tolerance, "feasibility_tolerance"
        )
        kkt = _nonnegative_tolerance(kkt_tolerance, "kkt_tolerance")
        damping_value = float(damping)
        minimum_ess = float(minimum_effective_sample_size)
        if not isfinite(damping_value) or not 0.0 < damping_value <= 1.0:
            raise ValueError("damping must be finite and in (0, 1].")
        if not isfinite(minimum_ess) or minimum_ess <= 0.0:
            raise ValueError("minimum_effective_sample_size must be finite and positive.")
        dual = _nonnegative_tolerance(
            kkt if dual_feasibility_tolerance is None else dual_feasibility_tolerance,
            "dual_feasibility_tolerance",
        )
        complementarity = _nonnegative_tolerance(
            kkt if complementarity_tolerance is None else complementarity_tolerance,
            "complementarity_tolerance",
        )
        identifier = _identifier(problem_id, "problem_id")
        self.maximum_iterations = maximum_iterations
        self.consistency_tolerance = consistency
        self.damping = damping_value
        self.minimum_effective_sample_size = minimum_ess
        self.feasibility_tolerance = feasibility
        self.kkt_tolerance = kkt
        self.dual_feasibility_tolerance = dual
        self.complementarity_tolerance = complementarity
        self.problem_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "constrained-mean-field-game-plan",
                "problem": identifier,
                "maximum_iterations": maximum_iterations,
                "consistency_tolerance": consistency,
                "damping": damping_value,
                "minimum_effective_sample_size": minimum_ess,
                "feasibility_tolerance": feasibility,
                "kkt_tolerance": kkt,
                "dual_feasibility_tolerance": dual,
                "complementarity_tolerance": complementarity,
            }
        )


class ConstrainedMeanFieldGameResult(StrictModule):
    """Frozen-response, law, sampled-feasibility, and original-KKT evidence."""

    problem: ConstrainedMeanFieldGameProblem
    plan: ConstrainedMeanFieldGamePlan
    fixed_point_result: MeanFieldGameFixedPointResult | None
    fixed_point_results: tuple[MeanFieldGameFixedPointResult, ...]
    flow: EmpiricalMeanField
    induced_flow: EmpiricalMeanField | None
    best_response_result: FrozenLawBestResponseResult | None
    individual_evidence: MeanFieldIndividualConstraintEvidence | None
    aggregate_derivative_evidence: MeanFieldAggregateConstraintDerivativeEvidence | None
    aggregate_raw_residuals: tuple[Array, ...]
    multipliers: Array
    physical_constraint_residuals: Array
    multiplier_residuals: Array
    law_distance_history: Array
    current_effective_sample_size_history: Array
    induced_effective_sample_size_history: Array
    individual_primal_violation_history: Array
    population_primal_violation_history: Array
    stationarity_residual_history: Array
    dual_violation_history: Array
    complementarity_residual_history: Array
    original_kkt_residual_history: Array
    physical_constraint_residual_history: Array
    multiplier_history: Array
    best_response_validity_history: Array
    induced_law_validity_history: Array
    law_consistency_history: Array
    individual_evidence_validity_history: Array
    population_feasibility_history: Array
    dual_feasibility_history: Array
    complementarity_validity_history: Array
    kkt_validity_history: Array
    iteration_validity_history: Array
    acceptance_history: Array
    iterations: Array
    accepted_iterations: Array
    accepted_iteration: Array
    final_law_distance: Array
    final_individual_primal_violation: Array
    final_population_primal_violation: Array
    final_stationarity_residual: Array
    final_dual_violation: Array
    final_complementarity_residual: Array
    final_original_kkt_residual: Array
    converged: Array
    valid: Array
    status: Array
    current_flow_ids: tuple[str | None, ...] = eqx.field(static=True)
    induced_flow_ids: tuple[str | None, ...] = eqx.field(static=True)
    current_source_path_ids: tuple[str | None, ...] = eqx.field(static=True)
    induced_source_path_ids: tuple[str | None, ...] = eqx.field(static=True)
    best_response_path_ids: tuple[str | None, ...] = eqx.field(static=True)
    individual_evidence_ids: tuple[str | None, ...] = eqx.field(static=True)
    aggregate_derivative_evidence_ids: tuple[str | None, ...] = eqx.field(static=True)
    population_ids: tuple[str, ...] = eqx.field(static=True)
    constraint_ids: tuple[str, ...] = eqx.field(static=True)
    individual_constraint_ids: tuple[str, ...] = eqx.field(static=True)
    aggregate_constraint_ids: tuple[str, ...] = eqx.field(static=True)
    multiplier_ids: tuple[str, ...] = eqx.field(static=True)
    current_flow_id: str = eqx.field(static=True)
    induced_flow_id: str | None = eqx.field(static=True)
    current_source_path_id: str | None = eqx.field(static=True)
    induced_source_path_id: str | None = eqx.field(static=True)
    best_response_path_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    fixed_point_problem_id: str = eqx.field(static=True)
    constraints_id: str = eqx.field(static=True)
    constraint_layout_id: str = eqx.field(static=True)
    multiplier_layout_id: str = eqx.field(static=True)
    individual_evidence_callback_id: str | None = eqx.field(static=True)
    aggregate_law_constraints_id: str | None = eqx.field(static=True)
    aggregate_derivative_evidence_callback_id: str | None = eqx.field(static=True)
    multiplier_callback_id: str | None = eqx.field(static=True)
    concept: MeanFieldConstraintConcept = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    sampling_scope: str = eqx.field(static=True)
    candidate_evaluation_only: bool = eqx.field(static=True)
    sampled_only: bool = eqx.field(static=True)
    frozen_law_best_response_evaluated: bool = eqx.field(static=True)
    law_consistency_evaluated: bool = eqx.field(static=True)
    best_response_kkt_evaluated: bool = eqx.field(static=True)
    best_response_optimality_evaluated: bool = eqx.field(static=True)
    individual_feasibility_evaluated: bool = eqx.field(static=True)
    aggregate_feasibility_evaluated: bool = eqx.field(static=True)
    continuous_safety_claimed: bool = eqx.field(static=True)
    mean_field_game_equilibrium_claimed: bool = eqx.field(static=True)
    generalized_mean_field_equilibrium_claimed: bool = eqx.field(static=True)
    mean_field_control_optimum_claimed: bool = eqx.field(static=True)
    master_equation_claimed: bool = eqx.field(static=True)
    finite_population_game_claimed: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid

    @property
    def mean_field(self) -> EmpiricalMeanField:
        return self.flow

    @property
    def final_best_response(self) -> FrozenLawBestResponseResult | None:
        return self.best_response_result

    @property
    def population_multipliers(self) -> tuple[Array, ...]:
        return tuple(
            self.multipliers[start:stop]
            for start, stop in self.problem.multiplier_layout.player_slices
        )

    @property
    def common_multipliers(self) -> Array:
        start, stop = self.problem.multiplier_layout.shared_slice
        return self.multipliers[start:stop]


def _one_step_problem(
    base: MeanFieldGameFixedPointProblem,
    flow: EmpiricalMeanField,
    iteration: int,
    /,
) -> MeanFieldGameFixedPointProblem:
    mixture_builder = base.law_mixture

    def law_mixture(
        current: EmpiricalMeanField,
        induced: EmpiricalMeanField,
        damping: float,
        local_iteration: int,
        args: Any,
        /,
    ) -> EmpiricalMeanField:
        del local_iteration
        if mixture_builder is None:
            raise RuntimeError("Missing law_mixture after validation.")
        return mixture_builder(current, induced, damping, iteration, args)

    return MeanFieldGameFixedPointProblem(
        flow,
        base.best_response,
        base.induced_flow,
        base.law_distance,
        law_mixture=(None if mixture_builder is None else law_mixture),
        law_mixture_id=base.law_mixture_id,
        best_response_id=base.best_response_id,
        induced_flow_id=base.induced_flow_id,
        law_distance_id=base.law_distance_id,
        problem_id=base.problem_id,
    )


def _one_step_plan(
    problem: MeanFieldGameFixedPointProblem,
    plan: ConstrainedMeanFieldGamePlan,
    /,
) -> MeanFieldGameFixedPointPlan:
    return MeanFieldGameFixedPointPlan(
        maximum_iterations=1,
        consistency_tolerance=plan.consistency_tolerance,
        damping=plan.damping,
        minimum_effective_sample_size=plan.minimum_effective_sample_size,
        problem_id=problem.problem_id,
    )


def _fixed_point_failure_status(
    result: MeanFieldGameFixedPointResult,
    /,
) -> ConstrainedMeanFieldGameStatus:
    status = int(result.status)
    if status == int(MeanFieldGameFixedPointStatus.INVALID_BEST_RESPONSE):
        return ConstrainedMeanFieldGameStatus.INVALID_BEST_RESPONSE
    if status == int(MeanFieldGameFixedPointStatus.INVALID_INDUCED_LAW):
        return ConstrainedMeanFieldGameStatus.INVALID_INDUCED_LAW
    if status == int(MeanFieldGameFixedPointStatus.INVALID_LAW_MIXTURE):
        return ConstrainedMeanFieldGameStatus.INVALID_LAW_MIXTURE
    if status == int(MeanFieldGameFixedPointStatus.LOW_EFFECTIVE_SAMPLE_SIZE):
        return ConstrainedMeanFieldGameStatus.LOW_EFFECTIVE_SAMPLE_SIZE
    return ConstrainedMeanFieldGameStatus.INVALID_LAW_DISTANCE


def _validate_individual_evidence(
    problem: ConstrainedMeanFieldGameProblem,
    response: FrozenLawBestResponseResult,
    evidence: Any,
    /,
) -> bool:
    return (
        isinstance(evidence, MeanFieldIndividualConstraintEvidence)
        and evidence.evidence_id == problem.individual_evidence_id
        and evidence.best_response_flow_id == response.flow_id
        and evidence.best_response_path_id == response.paths.path_id
        and evidence.feasibility.layout.layout_id
        == problem.individual_constraint_layout.layout_id
        and bool(evidence.valid)
        and bool(evidence.finite)
        and bool(evidence.feasibility.valid)
    )


def _aggregate_multiplier_mask(
    problem: ConstrainedMeanFieldGameProblem,
    /,
) -> Array:
    mask = [False] * problem.multiplier_layout.num_multipliers
    groups = zip(
        problem.multiplier_layout.player_block_indices,
        problem.multiplier_layout.player_multiplier_slices,
        strict=True,
    )
    for block_indices, destinations in groups:
        for block_index, (left, right) in zip(block_indices, destinations, strict=True):
            if (
                problem.constraints.blocks[block_index].scope
                is GameConstraintScope.SHARED
            ):
                mask[left:right] = [True] * (right - left)
    for left, right in problem.multiplier_layout.shared_multiplier_slices:
        mask[left:right] = [True] * (right - left)
    return jnp.asarray(mask, dtype=bool)


def _validate_aggregate_derivative_evidence(
    problem: ConstrainedMeanFieldGameProblem,
    response: FrozenLawBestResponseResult,
    induced: EmpiricalMeanField,
    multipliers: Array,
    individual: MeanFieldIndividualConstraintEvidence,
    evidence: Any,
    /,
) -> bool:
    if not isinstance(evidence, MeanFieldAggregateConstraintDerivativeEvidence):
        return False
    expected_shape = (
        problem.multiplier_layout.num_multipliers,
        individual.original_stationarity.size,
    )
    if (
        evidence.aggregate_jacobian.shape != expected_shape
        or evidence.multipliers.shape != multipliers.shape
    ):
        return False
    aggregate_rows = _aggregate_multiplier_mask(problem)
    nonaggregate_derivatives = jnp.where(
        aggregate_rows[:, None],
        0.0,
        evidence.aggregate_jacobian,
    )
    return (
        evidence.evidence_id == problem.aggregate_derivative_evidence_id
        and evidence.best_response_flow_id == response.flow_id
        and evidence.best_response_path_id == response.paths.path_id
        and evidence.induced_flow_id == induced.mean_field_id
        and evidence.aggregate_law_constraints_id == problem.aggregate_law_constraints_id
        and evidence.multiplier_ids == problem.multiplier_ids
        and bool(jnp.array_equal(evidence.multipliers, multipliers))
        and bool(jnp.all(nonaggregate_derivatives == 0.0))
        and bool(evidence.valid)
        and bool(evidence.finite)
    )


def _aggregate_residuals(
    problem: ConstrainedMeanFieldGameProblem,
    induced: EmpiricalMeanField,
    args: Any,
    /,
) -> tuple[Array, ...] | None:
    if problem.aggregate_law_residuals is None:
        return ()
    values = problem.aggregate_law_residuals(induced, args)
    if isinstance(values, (str, bytes)):
        return None
    try:
        result = tuple(jnp.asarray(value) for value in values)
    except (TypeError, ValueError):
        return None
    if len(result) != problem.aggregate_constraint_layout.num_blocks:
        return None
    for value, expected in zip(
        result,
        problem.aggregate_constraint_layout.block_output_shapes,
        strict=True,
    ):
        if value.shape != expected or jnp.issubdtype(value.dtype, jnp.complexfloating):
            return None
    return result


def _multiplier_vector(
    problem: ConstrainedMeanFieldGameProblem,
    response: FrozenLawBestResponseResult,
    induced: EmpiricalMeanField,
    args: Any,
    dtype: jnp.dtype,
    /,
) -> Array | None:
    if problem.multipliers is None:
        return jnp.zeros((0,), dtype=dtype)
    value = jnp.asarray(problem.multipliers(response, induced, args))
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        return None
    if value.shape != (problem.multiplier_layout.num_multipliers,):
        return None
    return value.astype(dtype)


def _ordered_physical_residuals(
    problem: ConstrainedMeanFieldGameProblem,
    evidence: MeanFieldIndividualConstraintEvidence | None,
    aggregate: tuple[Array, ...],
    dtype: jnp.dtype,
    /,
) -> tuple[Array, tuple[Array, ...]]:
    individual_values = () if evidence is None else evidence.feasibility.raw_residuals
    individual_iterator = iter(individual_values)
    aggregate_iterator = iter(aggregate)
    ordered: list[Array] = []
    for block in problem.constraints.blocks:
        value = (
            next(aggregate_iterator)
            if block.scope is GameConstraintScope.SHARED
            else next(individual_iterator)
        )
        ordered.append(jnp.asarray(value, dtype=dtype))
    if not ordered:
        return jnp.zeros((0,), dtype=dtype), ()
    return jnp.concatenate(tuple(value.reshape((-1,)) for value in ordered)), tuple(
        ordered
    )


def _block_primal_violation(
    values: Sequence[Array],
    blocks: Sequence[GameConstraintBlock],
    dtype: jnp.dtype,
    /,
) -> Array:
    maxima: list[Array] = []
    for value, block in zip(values, blocks, strict=True):
        residual = jnp.asarray(value, dtype=dtype)
        violation = jnp.abs(residual) if block.equality else jnp.maximum(residual, 0.0)
        maxima.append(jnp.max(violation))
    return jnp.max(jnp.stack(maxima)) if maxima else jnp.asarray(0.0, dtype=dtype)


def _multiplier_metadata(
    layout: GameMultiplierLayout,
    /,
) -> tuple[Array, Array]:
    rows = [-1] * layout.num_multipliers
    equality = [False] * layout.num_multipliers
    constraint_layout = layout.constraint_layout
    groups = zip(
        layout.player_block_indices,
        layout.player_multiplier_slices,
        layout.player_residual_slices,
        strict=True,
    )
    for block_indices, destinations, sources in groups:
        for block_index, (left, right), (source_left, source_right) in zip(
            block_indices, destinations, sources, strict=True
        ):
            rows[left:right] = range(source_left, source_right)
            equality[left:right] = [constraint_layout.equalities[block_index]] * (
                right - left
            )
    for block_index, (left, right), (source_left, source_right) in zip(
        layout.shared_block_indices,
        layout.shared_multiplier_slices,
        layout.shared_residual_slices,
        strict=True,
    ):
        rows[left:right] = range(source_left, source_right)
        equality[left:right] = [constraint_layout.equalities[block_index]] * (
            right - left
        )
    if any(row < 0 for row in rows):
        raise RuntimeError("Multiplier layout did not cover every multiplier slot.")
    return jnp.asarray(rows, dtype=jnp.int32), jnp.asarray(equality, dtype=bool)


def _maximum_or_zero(value: Array, dtype: jnp.dtype, /) -> Array:
    return jnp.max(value) if value.size else jnp.asarray(0.0, dtype=dtype)


def _last(history: Array, index: int | None, dtype: jnp.dtype, /) -> Array:
    return jnp.asarray(jnp.nan, dtype=dtype) if index is None else history[index]


def solve_constrained_mean_field_game(
    problem: ConstrainedMeanFieldGameProblem,
    plan: ConstrainedMeanFieldGamePlan,
    /,
    *,
    args: Any = None,
) -> ConstrainedMeanFieldGameResult:
    """Evaluate a constrained state-law MFG candidate to fixed outer capacity.

    Each outer step delegates frozen best-response, independently induced-law,
    empirical-law validity, ESS, and distance checks to the existing MFG fixed-point
    evaluator. Acceptance additionally requires original-scale individual
    feasibility/stationarity evidence, aggregate-law primal feasibility, identified
    aggregate derivative evidence evaluated at the exact induced law and multiplier
    vector, inequality dual feasibility, and complementarity. The
    multiplier-weighted aggregate Jacobian is added to the original
    representative-agent stationarity vector before the KKT gate. The result is a
    sampled KKT candidate; it is not a continuous-safety, global-equilibrium, MFC,
    master-equation, or finite-population claim.
    """
    if not isinstance(problem, ConstrainedMeanFieldGameProblem):
        raise TypeError("problem must be a ConstrainedMeanFieldGameProblem.")
    if not isinstance(plan, ConstrainedMeanFieldGamePlan):
        raise TypeError("plan must be a ConstrainedMeanFieldGamePlan.")
    if plan.problem_id != problem.problem_id:
        raise ValueError("plan and problem IDs must match.")

    capacity = plan.maximum_iterations
    dtype = jnp.result_type(problem.fixed_point_problem.initial_flow.particles, float)
    nan_history = lambda: jnp.full((capacity,), jnp.nan, dtype=dtype)
    law_distance_history = nan_history()
    current_ess_history = nan_history()
    induced_ess_history = nan_history()
    individual_primal_history = nan_history()
    population_primal_history = nan_history()
    stationarity_history = nan_history()
    dual_history = nan_history()
    complementarity_history = nan_history()
    original_kkt_history = nan_history()
    physical_history = jnp.full(
        (capacity, problem.constraint_layout.num_residuals), jnp.nan, dtype=dtype
    )
    multiplier_history = jnp.full(
        (capacity, problem.multiplier_layout.num_multipliers),
        jnp.nan,
        dtype=dtype,
    )
    best_response_validity = jnp.zeros((capacity,), dtype=bool)
    induced_law_validity = jnp.zeros((capacity,), dtype=bool)
    law_consistency = jnp.zeros((capacity,), dtype=bool)
    evidence_validity = jnp.zeros((capacity,), dtype=bool)
    population_feasibility = jnp.zeros((capacity,), dtype=bool)
    dual_feasibility = jnp.zeros((capacity,), dtype=bool)
    complementarity_validity = jnp.zeros((capacity,), dtype=bool)
    kkt_validity = jnp.zeros((capacity,), dtype=bool)
    iteration_validity = jnp.zeros((capacity,), dtype=bool)
    acceptance = jnp.zeros((capacity,), dtype=bool)

    current_flow_ids: list[str | None] = [None] * capacity
    induced_flow_ids: list[str | None] = [None] * capacity
    current_source_ids: list[str | None] = [None] * capacity
    induced_source_ids: list[str | None] = [None] * capacity
    response_path_ids: list[str | None] = [None] * capacity
    evidence_ids: list[str | None] = [None] * capacity
    aggregate_derivative_evidence_ids: list[str | None] = [None] * capacity

    current = problem.fixed_point_problem.initial_flow
    used_flow_ids = {current.mean_field_id}
    used_source_ids = (
        set() if current.source_path_id is None else {current.source_path_id}
    )
    fixed_point_results: list[MeanFieldGameFixedPointResult] = []
    final_fixed_point: MeanFieldGameFixedPointResult | None = None
    final_response: FrozenLawBestResponseResult | None = None
    final_induced: EmpiricalMeanField | None = None
    final_evidence: MeanFieldIndividualConstraintEvidence | None = None
    final_aggregate_derivative_evidence: (
        MeanFieldAggregateConstraintDerivativeEvidence | None
    ) = None
    final_aggregate: tuple[Array, ...] = ()
    final_multipliers = jnp.zeros(
        (problem.multiplier_layout.num_multipliers,), dtype=dtype
    )
    final_physical = jnp.zeros((problem.constraint_layout.num_residuals,), dtype=dtype)
    final_multiplier_residuals = jnp.zeros(
        (problem.multiplier_layout.num_multipliers,), dtype=dtype
    )
    status = ConstrainedMeanFieldGameStatus.MAX_ITERATIONS
    iterations = 0
    accepted_iteration = -1
    last_evaluated_index: int | None = None
    multiplier_rows, multiplier_equalities = _multiplier_metadata(
        problem.multiplier_layout
    )

    for index in range(capacity):
        iterations = index + 1
        current_flow_ids[index] = current.mean_field_id
        current_source_ids[index] = current.source_path_id
        step_problem = _one_step_problem(
            problem.fixed_point_problem,
            current,
            index,
        )
        step_result = solve_mean_field_game_fixed_point(
            step_problem,
            _one_step_plan(step_problem, plan),
            args=args,
        )
        fixed_point_results.append(step_result)
        final_fixed_point = step_result
        final_response = step_result.best_response_result
        final_induced = step_result.induced_flow
        law_distance_history = law_distance_history.at[index].set(
            step_result.final_distance
        )
        current_ess_history = current_ess_history.at[index].set(
            step_result.current_effective_sample_size_history[0]
        )
        induced_ess_history = induced_ess_history.at[index].set(
            step_result.induced_effective_sample_size_history[0]
        )
        best_response_validity = best_response_validity.at[index].set(
            step_result.best_response_validity_history[0]
        )
        induced_law_validity = induced_law_validity.at[index].set(
            step_result.induced_flow_validity_history[0]
        )
        if final_response is not None:
            response_path_ids[index] = final_response.paths.path_id
        if final_induced is not None:
            induced_flow_ids[index] = final_induced.mean_field_id
            induced_source_ids[index] = final_induced.source_path_id

        fixed_point_status = int(step_result.status)
        base_iteration_valid = bool(
            step_result.iteration_validity_history[0]
        ) and fixed_point_status in (
            int(MeanFieldGameFixedPointStatus.SUCCESS),
            int(MeanFieldGameFixedPointStatus.MAX_ITERATIONS),
        )
        if not base_iteration_valid:
            status = _fixed_point_failure_status(step_result)
            break
        if final_response is None or final_induced is None:
            status = ConstrainedMeanFieldGameStatus.INVALID_INDUCED_LAW
            break
        if (
            final_induced.mean_field_id in used_flow_ids
            or final_induced.source_path_id is None
            or final_induced.source_path_id in used_source_ids
        ):
            status = ConstrainedMeanFieldGameStatus.INVALID_INDUCED_LAW
            break

        is_law_consistent = bool(step_result.converged)
        law_consistency = law_consistency.at[index].set(is_law_consistent)

        evidence: MeanFieldIndividualConstraintEvidence | None = None
        if problem.individual_evidence is not None:
            candidate_evidence = problem.individual_evidence(final_response, args)
            if not _validate_individual_evidence(
                problem, final_response, candidate_evidence
            ):
                status = ConstrainedMeanFieldGameStatus.INVALID_INDIVIDUAL_EVIDENCE
                break
            evidence = candidate_evidence
            evidence_ids[index] = evidence.evidence_id
        evidence_validity = evidence_validity.at[index].set(True)

        aggregate = _aggregate_residuals(problem, final_induced, args)
        if aggregate is None or any(
            not bool(jnp.all(jnp.isfinite(value))) for value in aggregate
        ):
            status = ConstrainedMeanFieldGameStatus.INVALID_AGGREGATE_RESIDUAL
            break
        multiplier_vector = _multiplier_vector(
            problem, final_response, final_induced, args, dtype
        )
        if multiplier_vector is None or not bool(
            jnp.all(jnp.isfinite(multiplier_vector))
        ):
            status = ConstrainedMeanFieldGameStatus.INVALID_MULTIPLIERS
            break

        derivative_evidence: MeanFieldAggregateConstraintDerivativeEvidence | None = None
        if problem.aggregate_derivative_evidence is not None:
            candidate_derivative_evidence = problem.aggregate_derivative_evidence(
                final_response,
                final_induced,
                multiplier_vector,
                args,
            )
            if evidence is None or not _validate_aggregate_derivative_evidence(
                problem,
                final_response,
                final_induced,
                multiplier_vector,
                evidence,
                candidate_derivative_evidence,
            ):
                status = (
                    ConstrainedMeanFieldGameStatus.INVALID_AGGREGATE_DERIVATIVE_EVIDENCE
                )
                break
            derivative_evidence = candidate_derivative_evidence
            aggregate_derivative_evidence_ids[index] = derivative_evidence.evidence_id

        physical, ordered = _ordered_physical_residuals(
            problem, evidence, aggregate, dtype
        )
        multiplier_residual = jnp.take(physical, multiplier_rows, axis=0)
        inequality_multiplier = jnp.where(multiplier_equalities, 0.0, multiplier_vector)
        inequality_residual = jnp.where(multiplier_equalities, 0.0, multiplier_residual)
        dual_violation = _maximum_or_zero(jnp.maximum(-inequality_multiplier, 0.0), dtype)
        complementarity = _maximum_or_zero(
            jnp.abs(inequality_multiplier * inequality_residual), dtype
        )

        individual_ordered = tuple(
            value
            for value, block in zip(ordered, problem.constraints.blocks, strict=True)
            if block.scope is not GameConstraintScope.SHARED
        )
        individual_primal = _block_primal_violation(
            individual_ordered,
            problem.individual_constraints.blocks,
            dtype,
        )
        population_primal = _block_primal_violation(
            aggregate,
            problem.aggregate_constraints.blocks,
            dtype,
        )
        if evidence is None:
            stationarity = jnp.asarray(0.0, dtype=dtype)
        elif derivative_evidence is None:
            stationarity = evidence.original_stationarity_residual.astype(dtype)
        else:
            aggregate_stationarity = ein.contract(
                "ms,m->s",
                derivative_evidence.aggregate_jacobian.astype(dtype),
                multiplier_vector,
            )
            complete_stationarity = (
                evidence.original_stationarity.astype(dtype) + aggregate_stationarity
            )
            stationarity = jnp.max(jnp.abs(complete_stationarity))
        original_kkt = jnp.max(
            jnp.stack(
                (
                    individual_primal,
                    population_primal,
                    stationarity,
                    dual_violation,
                    complementarity,
                )
            )
        )

        individual_feasible = individual_primal <= plan.feasibility_tolerance
        population_feasible = population_primal <= plan.feasibility_tolerance
        dual_feasible = dual_violation <= plan.dual_feasibility_tolerance
        complementary = complementarity <= plan.complementarity_tolerance
        stationary = stationarity <= plan.kkt_tolerance
        kkt_ok = (
            individual_feasible
            & population_feasible
            & dual_feasible
            & complementary
            & stationary
        )
        all_finite = bool(
            jnp.all(jnp.isfinite(physical))
            & jnp.isfinite(stationarity)
            & jnp.isfinite(original_kkt)
        )

        individual_primal_history = individual_primal_history.at[index].set(
            individual_primal
        )
        population_primal_history = population_primal_history.at[index].set(
            population_primal
        )
        stationarity_history = stationarity_history.at[index].set(stationarity)
        dual_history = dual_history.at[index].set(dual_violation)
        complementarity_history = complementarity_history.at[index].set(complementarity)
        original_kkt_history = original_kkt_history.at[index].set(original_kkt)
        physical_history = physical_history.at[index].set(physical)
        multiplier_history = multiplier_history.at[index].set(multiplier_vector)
        population_feasibility = population_feasibility.at[index].set(population_feasible)
        dual_feasibility = dual_feasibility.at[index].set(dual_feasible)
        complementarity_validity = complementarity_validity.at[index].set(complementary)
        kkt_validity = kkt_validity.at[index].set(kkt_ok)
        iteration_validity = iteration_validity.at[index].set(all_finite)
        accepted = all_finite and is_law_consistent and bool(kkt_ok)
        acceptance = acceptance.at[index].set(accepted)

        final_evidence = evidence
        final_aggregate_derivative_evidence = derivative_evidence
        final_aggregate = aggregate
        final_multipliers = multiplier_vector
        final_physical = physical
        final_multiplier_residuals = multiplier_residual
        last_evaluated_index = index
        used_flow_ids.add(final_induced.mean_field_id)
        used_source_ids.add(final_induced.source_path_id)

        if accepted:
            status = ConstrainedMeanFieldGameStatus.SUCCESS
            accepted_iteration = index
            break
        if is_law_consistent:
            if not bool(individual_feasible):
                status = ConstrainedMeanFieldGameStatus.INDIVIDUAL_INFEASIBLE
            elif not bool(population_feasible):
                status = ConstrainedMeanFieldGameStatus.POPULATION_INFEASIBLE
            elif not bool(dual_feasible):
                status = ConstrainedMeanFieldGameStatus.DUAL_INFEASIBLE
            elif not bool(complementary):
                status = ConstrainedMeanFieldGameStatus.COMPLEMENTARITY_FAILURE
            else:
                status = ConstrainedMeanFieldGameStatus.INDIVIDUAL_KKT_FAILURE
            break
        if index + 1 == capacity:
            status = ConstrainedMeanFieldGameStatus.MAX_ITERATIONS
            break

        next_flow = step_result.flow
        if plan.damping < 1.0 and next_flow.mean_field_id in used_flow_ids:
            status = ConstrainedMeanFieldGameStatus.INVALID_LAW_MIXTURE
            break
        current = next_flow
        used_flow_ids.add(current.mean_field_id)
    successful = status is ConstrainedMeanFieldGameStatus.SUCCESS
    final_index = last_evaluated_index
    return ConstrainedMeanFieldGameResult(
        problem=problem,
        plan=plan,
        fixed_point_result=final_fixed_point,
        fixed_point_results=tuple(fixed_point_results),
        flow=current,
        induced_flow=final_induced,
        best_response_result=final_response,
        individual_evidence=final_evidence,
        aggregate_derivative_evidence=final_aggregate_derivative_evidence,
        aggregate_raw_residuals=final_aggregate,
        multipliers=final_multipliers,
        physical_constraint_residuals=final_physical,
        multiplier_residuals=final_multiplier_residuals,
        law_distance_history=law_distance_history,
        current_effective_sample_size_history=current_ess_history,
        induced_effective_sample_size_history=induced_ess_history,
        individual_primal_violation_history=individual_primal_history,
        population_primal_violation_history=population_primal_history,
        stationarity_residual_history=stationarity_history,
        dual_violation_history=dual_history,
        complementarity_residual_history=complementarity_history,
        original_kkt_residual_history=original_kkt_history,
        physical_constraint_residual_history=physical_history,
        multiplier_history=multiplier_history,
        best_response_validity_history=best_response_validity,
        induced_law_validity_history=induced_law_validity,
        law_consistency_history=law_consistency,
        individual_evidence_validity_history=evidence_validity,
        population_feasibility_history=population_feasibility,
        dual_feasibility_history=dual_feasibility,
        complementarity_validity_history=complementarity_validity,
        kkt_validity_history=kkt_validity,
        iteration_validity_history=iteration_validity,
        acceptance_history=acceptance,
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        accepted_iterations=jnp.asarray(int(successful), dtype=jnp.int32),
        accepted_iteration=jnp.asarray(accepted_iteration, dtype=jnp.int32),
        final_law_distance=(
            jnp.asarray(jnp.nan, dtype=dtype)
            if final_fixed_point is None
            else final_fixed_point.final_distance
        ),
        final_individual_primal_violation=_last(
            individual_primal_history, final_index, dtype
        ),
        final_population_primal_violation=_last(
            population_primal_history, final_index, dtype
        ),
        final_stationarity_residual=_last(stationarity_history, final_index, dtype),
        final_dual_violation=_last(dual_history, final_index, dtype),
        final_complementarity_residual=_last(complementarity_history, final_index, dtype),
        final_original_kkt_residual=_last(original_kkt_history, final_index, dtype),
        converged=jnp.asarray(successful),
        valid=jnp.asarray(successful),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        current_flow_ids=tuple(current_flow_ids),
        induced_flow_ids=tuple(induced_flow_ids),
        current_source_path_ids=tuple(current_source_ids),
        induced_source_path_ids=tuple(induced_source_ids),
        best_response_path_ids=tuple(response_path_ids),
        individual_evidence_ids=tuple(evidence_ids),
        aggregate_derivative_evidence_ids=tuple(aggregate_derivative_evidence_ids),
        population_ids=problem.population_ids,
        constraint_ids=problem.constraint_layout.block_ids,
        individual_constraint_ids=problem.individual_constraint_ids,
        aggregate_constraint_ids=problem.aggregate_constraint_ids,
        multiplier_ids=problem.multiplier_ids,
        current_flow_id=current.mean_field_id,
        induced_flow_id=(None if final_induced is None else final_induced.mean_field_id),
        current_source_path_id=current.source_path_id,
        induced_source_path_id=(
            None if final_induced is None else final_induced.source_path_id
        ),
        best_response_path_id=(
            None if final_response is None else final_response.paths.path_id
        ),
        problem_id=problem.problem_id,
        plan_id=plan.plan_id,
        fixed_point_problem_id=problem.fixed_point_problem.problem_id,
        constraints_id=problem.constraints.constraints_id,
        constraint_layout_id=problem.constraint_layout.layout_id,
        multiplier_layout_id=problem.multiplier_layout.layout_id,
        individual_evidence_callback_id=problem.individual_evidence_id,
        aggregate_law_constraints_id=problem.aggregate_law_constraints_id,
        aggregate_derivative_evidence_callback_id=(
            problem.aggregate_derivative_evidence_id
        ),
        multiplier_callback_id=problem.multiplier_callback_id,
        concept=problem.concept,
        certificate_label=CONSTRAINED_MEAN_FIELD_GAME_KKT_CANDIDATE,
        sampling_scope=(
            "empirical-induced-law-and-sampled-best-response-trajectory-sites"
            if problem.constraints.blocks
            else "empirical-frozen-and-induced-law-support"
        ),
        candidate_evaluation_only=True,
        sampled_only=True,
        frozen_law_best_response_evaluated=True,
        law_consistency_evaluated=True,
        best_response_kkt_evaluated=bool(problem.constraints.blocks),
        best_response_optimality_evaluated=False,
        individual_feasibility_evaluated=bool(problem.individual_constraints.blocks),
        aggregate_feasibility_evaluated=bool(problem.aggregate_constraints.blocks),
        continuous_safety_claimed=False,
        mean_field_game_equilibrium_claimed=False,
        generalized_mean_field_equilibrium_claimed=False,
        mean_field_control_optimum_claimed=False,
        master_equation_claimed=False,
        finite_population_game_claimed=False,
    )


__all__ = [
    "CONSTRAINED_MEAN_FIELD_GAME_KKT_CANDIDATE",
    "ConstrainedMeanFieldGamePlan",
    "ConstrainedMeanFieldGameProblem",
    "ConstrainedMeanFieldGameResult",
    "ConstrainedMeanFieldGameStatus",
    "MeanFieldConstraintConcept",
    "MeanFieldAggregateConstraintDerivativeEvidence",
    "MeanFieldIndividualConstraintEvidence",
    "solve_constrained_mean_field_game",
]
