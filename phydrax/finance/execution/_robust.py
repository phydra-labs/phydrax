#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Robust HJBI and mean-field execution candidate orchestration."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
from jaxtyping import Array

from ..._strict import StrictModule
from ...control.games._hjbi import (
    DiscreteZeroSumHJBIProblem,
    DiscreteZeroSumHJBIResult,
    solve_discrete_hjbi_reference,
)
from ...control.games._mean_field_fixed_point import (
    MeanFieldGameFixedPointPlan,
    MeanFieldGameFixedPointProblem,
    MeanFieldGameFixedPointResult,
    solve_mean_field_game_fixed_point,
)
from ..core import InstrumentReference, PhysicalLaw, StressLaw


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _nonnegative(value: float, owner: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return resolved


class RobustExecutionDefinition(StrictModule):
    """Zero-sum execution/stress game with P-law and stress law kept distinct."""

    instrument: InstrumentReference
    physical_law: PhysicalLaw
    stress_law: StressLaw
    problem: DiscreteZeroSumHJBIProblem
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        instrument: InstrumentReference,
        physical_law: PhysicalLaw,
        stress_law: StressLaw,
        problem: DiscreteZeroSumHJBIProblem,
        /,
        *,
        definition_id: str,
    ):
        if not isinstance(instrument, InstrumentReference):
            raise TypeError("instrument must be an InstrumentReference.")
        if not isinstance(physical_law, PhysicalLaw):
            raise TypeError("physical_law must be a PhysicalLaw.")
        if not isinstance(stress_law, StressLaw):
            raise TypeError("stress_law must be a StressLaw.")
        if physical_law.law_id == stress_law.law_id:
            raise ValueError("Physical and stress law identities must remain distinct.")
        if physical_law.factor_layout_id != stress_law.factor_layout_id:
            raise ValueError("Physical and stress laws must use the same factor layout.")
        if physical_law.filtration_id != stress_law.filtration_id:
            raise ValueError("Physical and stress laws must use the same filtration.")
        if not isinstance(problem, DiscreteZeroSumHJBIProblem):
            raise TypeError("problem must be a DiscreteZeroSumHJBIProblem.")
        self.instrument = instrument
        self.physical_law = physical_law
        self.stress_law = stress_law
        self.problem = problem
        self.definition_id = _identifier(definition_id, "definition_id")


class RobustExecutionPlan(StrictModule):
    """Residual, refinement, and Isaacs-gap thresholds for a robust candidate."""

    residual_tolerance: float = eqx.field(static=True)
    refinement_absolute_tolerance: float = eqx.field(static=True)
    refinement_relative_tolerance: float = eqx.field(static=True)
    isaacs_absolute_tolerance: float = eqx.field(static=True)
    isaacs_relative_tolerance: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual_tolerance: float,
        refinement_absolute_tolerance: float,
        refinement_relative_tolerance: float,
        isaacs_absolute_tolerance: float,
        isaacs_relative_tolerance: float,
        definition_id: str,
        plan_id: str,
    ):
        self.residual_tolerance = _nonnegative(residual_tolerance, "residual_tolerance")
        self.refinement_absolute_tolerance = _nonnegative(
            refinement_absolute_tolerance, "refinement_absolute_tolerance"
        )
        self.refinement_relative_tolerance = _nonnegative(
            refinement_relative_tolerance, "refinement_relative_tolerance"
        )
        self.isaacs_absolute_tolerance = _nonnegative(
            isaacs_absolute_tolerance, "isaacs_absolute_tolerance"
        )
        self.isaacs_relative_tolerance = _nonnegative(
            isaacs_relative_tolerance, "isaacs_relative_tolerance"
        )
        self.definition_id = _identifier(definition_id, "definition_id")
        self.plan_id = _identifier(plan_id, "plan_id")


class RobustExecutionResult(StrictModule):
    """Independent lower/upper value tables and bounded Isaacs evidence."""

    definition: RobustExecutionDefinition
    plan: RobustExecutionPlan
    reference: DiscreteZeroSumHJBIResult
    robust_claim_supported: Array
    result_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def solve_robust_execution_reference(
    definition: RobustExecutionDefinition,
    plan: RobustExecutionPlan,
    /,
) -> RobustExecutionResult:
    """Evaluate both action orders; support a robust label only when HJBI gates pass."""

    if not isinstance(definition, RobustExecutionDefinition):
        raise TypeError("definition must be a RobustExecutionDefinition.")
    if not isinstance(plan, RobustExecutionPlan):
        raise TypeError("plan must be a RobustExecutionPlan.")
    if plan.definition_id != definition.definition_id:
        raise ValueError("Robust execution plan is bound to a different definition.")
    reference = solve_discrete_hjbi_reference(
        definition.problem,
        residual_tolerance=plan.residual_tolerance,
        refinement_absolute_tolerance=plan.refinement_absolute_tolerance,
        refinement_relative_tolerance=plan.refinement_relative_tolerance,
        isaacs_absolute_tolerance=plan.isaacs_absolute_tolerance,
        isaacs_relative_tolerance=plan.isaacs_relative_tolerance,
    )
    return RobustExecutionResult(
        definition=definition,
        plan=plan,
        reference=reference,
        robust_claim_supported=reference.saddle,
        result_id=(f"robust-execution-result:{definition.definition_id}:{plan.plan_id}"),
        scope="declared-bounded-grid-discrete-lower-upper-isaacs-evidence-only",
    )


class MeanFieldExecutionDefinition(StrictModule):
    """Crowding statistic interpretation of a generic induced-law fixed point."""

    instrument: InstrumentReference
    physical_law: PhysicalLaw
    problem: MeanFieldGameFixedPointProblem
    crowding_statistic_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        instrument: InstrumentReference,
        physical_law: PhysicalLaw,
        problem: MeanFieldGameFixedPointProblem,
        /,
        *,
        crowding_statistic_id: str,
        definition_id: str,
    ):
        if not isinstance(instrument, InstrumentReference):
            raise TypeError("instrument must be an InstrumentReference.")
        if not isinstance(physical_law, PhysicalLaw):
            raise TypeError("mean-field execution requires a PhysicalLaw.")
        if not isinstance(problem, MeanFieldGameFixedPointProblem):
            raise TypeError("problem must be a MeanFieldGameFixedPointProblem.")
        self.instrument = instrument
        self.physical_law = physical_law
        self.problem = problem
        self.crowding_statistic_id = _identifier(
            crowding_statistic_id, "crowding_statistic_id"
        )
        self.definition_id = _identifier(definition_id, "definition_id")


class MeanFieldExecutionPlan(StrictModule):
    """Bounded induced-law iteration plan tied to an execution definition."""

    fixed_point_plan: MeanFieldGameFixedPointPlan
    definition_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fixed_point_plan: MeanFieldGameFixedPointPlan,
        /,
        *,
        definition_id: str,
        plan_id: str,
    ):
        if not isinstance(fixed_point_plan, MeanFieldGameFixedPointPlan):
            raise TypeError("fixed_point_plan must be a MeanFieldGameFixedPointPlan.")
        self.fixed_point_plan = fixed_point_plan
        self.definition_id = _identifier(definition_id, "definition_id")
        self.plan_id = _identifier(plan_id, "plan_id")


class MeanFieldExecutionResult(StrictModule):
    """Fixed-point residual history without an unsupported equilibrium claim."""

    definition: MeanFieldExecutionDefinition
    plan: MeanFieldExecutionPlan
    reference: MeanFieldGameFixedPointResult
    fixed_point_candidate: Array
    equilibrium_claimed: bool = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def solve_mean_field_execution_candidate(
    definition: MeanFieldExecutionDefinition,
    plan: MeanFieldExecutionPlan,
    /,
) -> MeanFieldExecutionResult:
    """Run bounded induced-law iteration and retain all consistency evidence."""

    if not isinstance(definition, MeanFieldExecutionDefinition):
        raise TypeError("definition must be a MeanFieldExecutionDefinition.")
    if not isinstance(plan, MeanFieldExecutionPlan):
        raise TypeError("plan must be a MeanFieldExecutionPlan.")
    if plan.definition_id != definition.definition_id:
        raise ValueError("Mean-field execution plan is bound to a different definition.")
    if plan.fixed_point_plan.problem_id != definition.problem.problem_id:
        raise ValueError("Fixed-point plan is bound to a different generic problem.")
    reference = solve_mean_field_game_fixed_point(
        definition.problem, plan.fixed_point_plan
    )
    candidate = reference.valid & reference.converged
    return MeanFieldExecutionResult(
        definition=definition,
        plan=plan,
        reference=reference,
        fixed_point_candidate=candidate,
        equilibrium_claimed=False,
        result_id=(
            f"mean-field-execution-result:{definition.definition_id}:{plan.plan_id}"
        ),
        scope="bounded-induced-law-fixed-point-candidate-evidence-only",
    )


__all__ = [
    "MeanFieldExecutionDefinition",
    "MeanFieldExecutionPlan",
    "MeanFieldExecutionResult",
    "RobustExecutionDefinition",
    "RobustExecutionPlan",
    "RobustExecutionResult",
    "solve_mean_field_execution_candidate",
    "solve_robust_execution_reference",
]
