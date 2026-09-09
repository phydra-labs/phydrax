#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Route-specific controlled-jump, HJB, and impulse-QVI execution orchestration."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...control.stochastic._controlled_jump import (
    ControlledJumpPathBatch,
    ControlledJumpPlan,
    ControlledJumpProblem,
    rollout_controlled_jumps_reference,
)
from ...control.stochastic._hjb import (
    DiscreteHJBProblem,
    DiscreteHJBResult,
    solve_discrete_hjb_reference,
)
from ...control.stochastic._impulse_qvi import (
    BoundedImpulseQVIProblem,
    ImpulseQVIPlan,
    ImpulseQVIResult,
    solve_impulse_qvi_reference,
)
from ...stochastic._jump import PoissonClockRealization
from ..core import InstrumentReference, PhysicalLaw


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


class ExecutionFeedbackPolicy(StrictModule):
    """Explicit affine state-feedback parameters and hard action bounds."""

    gain: Array
    bias: Array
    lower_bounds: Array
    upper_bounds: Array
    state_size: int = eqx.field(static=True)
    action_size: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        gain: ArrayLike,
        bias: ArrayLike,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        /,
        *,
        policy_id: str,
    ):
        matrix = jnp.asarray(gain)
        offset = jnp.asarray(bias)
        lower = jnp.asarray(lower_bounds)
        upper = jnp.asarray(upper_bounds)
        if matrix.ndim != 2 or 0 in matrix.shape:
            raise ValueError("gain must be a nonempty action-by-state matrix.")
        action_size, state_size = map(int, matrix.shape)
        if (
            offset.shape != (action_size,)
            or lower.shape != (action_size,)
            or upper.shape != (action_size,)
        ):
            raise ValueError(
                "bias and bounds must have one entry per action row of gain."
            )
        for owner, value in (
            ("gain", matrix),
            ("bias", offset),
            ("lower_bounds", lower),
            ("upper_bounds", upper),
        ):
            if jnp.issubdtype(value.dtype, jnp.complexfloating):
                raise TypeError(f"{owner} must be real-valued.")
            if not bool(jnp.all(jnp.isfinite(value))):
                raise ValueError(f"{owner} must be finite.")
        if bool(jnp.any(lower > upper)):
            raise ValueError("lower_bounds cannot exceed upper_bounds.")
        self.gain = matrix.astype(jnp.result_type(matrix, float))
        self.bias = offset.astype(jnp.result_type(offset, float))
        self.lower_bounds = lower.astype(jnp.result_type(lower, float))
        self.upper_bounds = upper.astype(jnp.result_type(upper, float))
        self.state_size = state_size
        self.action_size = action_size
        self.policy_id = _identifier(policy_id, "policy_id")

    def action(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        """Evaluate one affine action and reject rather than project bound violations."""

        del time
        values = jnp.asarray(state)
        if values.size != self.state_size:
            raise ValueError(
                f"state must contain exactly {self.state_size} scalar entries."
            )
        action = self.gain @ values.reshape((self.state_size,)) + self.bias
        if not bool(jnp.all(jnp.isfinite(action))):
            raise ValueError("Execution feedback produced a nonfinite action.")
        if bool(jnp.any((action < self.lower_bounds) | (action > self.upper_bounds))):
            raise ValueError("Execution feedback action violates its declared bounds.")
        return action


class JumpExecutionDefinition(StrictModule):
    """Physical-law binding for one generic controlled jump problem."""

    instrument: InstrumentReference
    physical_law: PhysicalLaw
    problem: ControlledJumpProblem
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        instrument: InstrumentReference,
        physical_law: PhysicalLaw,
        problem: ControlledJumpProblem,
        /,
        *,
        definition_id: str,
    ):
        if not isinstance(instrument, InstrumentReference):
            raise TypeError("instrument must be an InstrumentReference.")
        if not isinstance(physical_law, PhysicalLaw):
            raise TypeError("controlled jump execution requires a PhysicalLaw.")
        if not isinstance(problem, ControlledJumpProblem):
            raise TypeError("problem must be a ControlledJumpProblem.")
        self.instrument = instrument
        self.physical_law = physical_law
        self.problem = problem
        self.definition_id = _identifier(definition_id, "definition_id")


class JumpExecutionPlan(StrictModule):
    """Controlled-jump numerical plan tied to a definition identity."""

    control_plan: ControlledJumpPlan
    definition_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        control_plan: ControlledJumpPlan,
        /,
        *,
        definition_id: str,
        plan_id: str,
    ):
        if not isinstance(control_plan, ControlledJumpPlan):
            raise TypeError("control_plan must be a ControlledJumpPlan.")
        self.control_plan = control_plan
        self.definition_id = _identifier(definition_id, "definition_id")
        self.plan_id = _identifier(plan_id, "plan_id")


class PreparedJumpExecution(StrictModule):
    """Definition/plan/randomness binding prepared before policy evaluation."""

    definition: JumpExecutionDefinition
    plan: JumpExecutionPlan
    realization: PoissonClockRealization
    prepared_id: str = eqx.field(static=True)


def prepare_jump_execution(
    definition: JumpExecutionDefinition,
    plan: JumpExecutionPlan,
    realization: PoissonClockRealization,
    /,
) -> PreparedJumpExecution:
    if not isinstance(definition, JumpExecutionDefinition):
        raise TypeError("definition must be a JumpExecutionDefinition.")
    if not isinstance(plan, JumpExecutionPlan):
        raise TypeError("plan must be a JumpExecutionPlan.")
    if plan.definition_id != definition.definition_id:
        raise ValueError("Jump execution plan is bound to a different definition.")
    if not isinstance(realization, PoissonClockRealization):
        raise TypeError("realization must be a PoissonClockRealization.")
    if realization.process_id != definition.problem.process.process_id:
        raise ValueError("realization process identity does not match the definition.")
    return PreparedJumpExecution(
        definition=definition,
        plan=plan,
        realization=realization,
        prepared_id=(
            f"prepared-jump-execution:{definition.definition_id}:"
            f"{plan.plan_id}:{realization.realization_id}"
        ),
    )


class JumpExecutionResult(StrictModule):
    """Controlled jump path evidence under one declared physical law."""

    prepared: PreparedJumpExecution
    policy: ExecutionFeedbackPolicy
    paths: ControlledJumpPathBatch
    successful: Array
    physical_law_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def evaluate_jump_execution(
    prepared: PreparedJumpExecution,
    policy: ExecutionFeedbackPolicy,
    /,
) -> JumpExecutionResult:
    if not isinstance(prepared, PreparedJumpExecution):
        raise TypeError("prepared must be a PreparedJumpExecution.")
    if not isinstance(policy, ExecutionFeedbackPolicy):
        raise TypeError("policy must be an ExecutionFeedbackPolicy.")
    expected_state_size = int(
        jnp.prod(jnp.asarray(prepared.definition.problem.state_shape))
    )
    expected_action_size = int(
        jnp.prod(jnp.asarray(prepared.definition.problem.action_shape))
    )
    if (
        policy.state_size != expected_state_size
        or policy.action_size != expected_action_size
    ):
        raise ValueError("Policy state/action dimensions do not match the jump problem.")

    def feedback(time, state, args):
        del args
        return policy.action(time, state).reshape(
            prepared.definition.problem.action_shape
        )

    paths = rollout_controlled_jumps_reference(
        prepared.definition.problem,
        prepared.plan.control_plan,
        prepared.realization,
        feedback,
        policy_id=policy.policy_id,
    )
    return JumpExecutionResult(
        prepared=prepared,
        policy=policy,
        paths=paths,
        successful=jnp.all(paths.valid),
        physical_law_id=prepared.definition.physical_law.law_id,
        result_id=f"jump-execution-result:{prepared.prepared_id}:{policy.policy_id}",
        scope="supplied-physical-law-fixed-capacity-reference-paths-only",
    )


class HJBExecutionDefinition(StrictModule):
    """Execution interpretation of one bounded generic HJB problem."""

    instrument: InstrumentReference
    physical_law: PhysicalLaw
    problem: DiscreteHJBProblem
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        instrument: InstrumentReference,
        physical_law: PhysicalLaw,
        problem: DiscreteHJBProblem,
        /,
        *,
        definition_id: str,
    ):
        if not isinstance(instrument, InstrumentReference):
            raise TypeError("instrument must be an InstrumentReference.")
        if not isinstance(physical_law, PhysicalLaw):
            raise TypeError("HJB execution requires a PhysicalLaw.")
        if not isinstance(problem, DiscreteHJBProblem):
            raise TypeError("problem must be a DiscreteHJBProblem.")
        self.instrument = instrument
        self.physical_law = physical_law
        self.problem = problem
        self.definition_id = _identifier(definition_id, "definition_id")


class HJBExecutionPlan(StrictModule):
    """Residual/refinement thresholds for one HJB execution reference."""

    residual_tolerance: float = eqx.field(static=True)
    refinement_absolute_tolerance: float = eqx.field(static=True)
    refinement_relative_tolerance: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual_tolerance: float,
        refinement_absolute_tolerance: float,
        refinement_relative_tolerance: float,
        definition_id: str,
        plan_id: str,
    ):
        values = (
            residual_tolerance,
            refinement_absolute_tolerance,
            refinement_relative_tolerance,
        )
        if any(not jnp.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("HJB execution tolerances must be finite and nonnegative.")
        self.residual_tolerance = float(residual_tolerance)
        self.refinement_absolute_tolerance = float(refinement_absolute_tolerance)
        self.refinement_relative_tolerance = float(refinement_relative_tolerance)
        self.definition_id = _identifier(definition_id, "definition_id")
        self.plan_id = _identifier(plan_id, "plan_id")


class HJBExecutionResult(StrictModule):
    """Bounded HJB result with its financial-law interpretation retained."""

    definition: HJBExecutionDefinition
    plan: HJBExecutionPlan
    reference: DiscreteHJBResult
    successful: Array
    result_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def solve_hjb_execution_reference(
    definition: HJBExecutionDefinition,
    plan: HJBExecutionPlan,
    /,
) -> HJBExecutionResult:
    if not isinstance(definition, HJBExecutionDefinition):
        raise TypeError("definition must be an HJBExecutionDefinition.")
    if not isinstance(plan, HJBExecutionPlan):
        raise TypeError("plan must be an HJBExecutionPlan.")
    if plan.definition_id != definition.definition_id:
        raise ValueError("HJB execution plan is bound to a different definition.")
    reference = solve_discrete_hjb_reference(
        definition.problem,
        residual_tolerance=plan.residual_tolerance,
        refinement_absolute_tolerance=plan.refinement_absolute_tolerance,
        refinement_relative_tolerance=plan.refinement_relative_tolerance,
    )
    return HJBExecutionResult(
        definition=definition,
        plan=plan,
        reference=reference,
        successful=reference.successful,
        result_id=f"hjb-execution-result:{definition.definition_id}:{plan.plan_id}",
        scope="declared-bounded-grid-discrete-hjb-reference-only",
    )


class ImpulseExecutionDefinition(StrictModule):
    """Execution interpretation of one bounded generic impulse-QVI problem."""

    instrument: InstrumentReference
    physical_law: PhysicalLaw
    problem: BoundedImpulseQVIProblem
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        instrument: InstrumentReference,
        physical_law: PhysicalLaw,
        problem: BoundedImpulseQVIProblem,
        /,
        *,
        definition_id: str,
    ):
        if not isinstance(instrument, InstrumentReference):
            raise TypeError("instrument must be an InstrumentReference.")
        if not isinstance(physical_law, PhysicalLaw):
            raise TypeError("impulse execution requires a PhysicalLaw.")
        if not isinstance(problem, BoundedImpulseQVIProblem):
            raise TypeError("problem must be a BoundedImpulseQVIProblem.")
        self.instrument = instrument
        self.physical_law = physical_law
        self.problem = problem
        self.definition_id = _identifier(definition_id, "definition_id")


class ImpulseExecutionResult(StrictModule):
    """Bounded impulse reference retaining complementarity/refinement evidence."""

    definition: ImpulseExecutionDefinition
    plan: ImpulseQVIPlan
    reference: ImpulseQVIResult
    successful: Array
    result_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)


def solve_impulse_execution_reference(
    definition: ImpulseExecutionDefinition,
    plan: ImpulseQVIPlan,
    /,
) -> ImpulseExecutionResult:
    if not isinstance(definition, ImpulseExecutionDefinition):
        raise TypeError("definition must be an ImpulseExecutionDefinition.")
    if not isinstance(plan, ImpulseQVIPlan):
        raise TypeError("plan must be an ImpulseQVIPlan.")
    reference = solve_impulse_qvi_reference(definition.problem, plan)
    return ImpulseExecutionResult(
        definition=definition,
        plan=plan,
        reference=reference,
        successful=reference.successful,
        result_id=(f"impulse-execution-result:{definition.definition_id}:{plan.plan_id}"),
        scope="declared-bounded-grid-single-impulse-qvi-reference-only",
    )


__all__ = [
    "ExecutionFeedbackPolicy",
    "HJBExecutionDefinition",
    "HJBExecutionPlan",
    "HJBExecutionResult",
    "ImpulseExecutionDefinition",
    "ImpulseExecutionResult",
    "JumpExecutionDefinition",
    "JumpExecutionPlan",
    "JumpExecutionResult",
    "PreparedJumpExecution",
    "evaluate_jump_execution",
    "prepare_jump_execution",
    "solve_hjb_execution_reference",
    "solve_impulse_execution_reference",
]
