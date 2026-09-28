#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared holonomic mechanics with native SHAKE/RATTLE projections."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    DenseSVD,
    FailurePolicy,
    FunctionLinearOperator,
    JacobianLinearOperator,
    LinearSolvePolicy,
    LinearSolveStatus,
    MaterializationPolicy,
    MinimumNormProblem,
    prepare_linearization,
    RankPolicy,
    solve,
)


class ConstrainedMechanicsStatus(IntEnum):
    """Outcome of one bounded SHAKE/RATTLE step."""

    SUCCESS = 0
    INVALID_STEP = 1
    POSITION_PROJECTION_FAILED = 2
    VELOCITY_PROJECTION_FAILED = 3
    RANK_DEFICIENT = 4
    CONDITION_LIMIT_REACHED = 5
    NONFINITE = 6


@final
class ConstrainedMechanicalState(StrictModule):
    """Configuration and canonical momentum of one constrained system."""

    configuration: Array
    momentum: Array

    def __init__(self, configuration: ArrayLike, momentum: ArrayLike, /) -> None:
        configuration_ = jnp.asarray(configuration)
        momentum_ = jnp.asarray(momentum)
        if configuration_.ndim != 1 or momentum_.shape != configuration_.shape:
            raise ValueError("Configuration and momentum must be aligned vectors.")
        if not jnp.issubdtype(configuration_.dtype, jnp.floating):
            raise TypeError("Configuration must have a floating dtype.")
        if momentum_.dtype != configuration_.dtype:
            raise TypeError("Configuration and momentum must have the same dtype.")
        self.configuration = configuration_
        self.momentum = momentum_


@final
class ConstrainedMechanicsEvidence(StrictModule):
    """Residual, projection-work, rank, conditioning, and rollback evidence."""

    status: Array
    position_residual: Array
    velocity_residual: Array
    position_iterations: Array
    projection_work: Array
    constraint_rank: Array
    constraint_condition: Array
    position_linear_status: Array
    velocity_linear_status: Array
    finite: Array
    accepted: Array
    derivative_available: Array
    prepared_id: str = eqx.field(static=True)


@final
class ConstrainedMechanicalStep(StrictModule):
    """Accepted state or the unchanged source state, plus complete evidence."""

    state: ConstrainedMechanicalState
    evidence: ConstrainedMechanicsEvidence

    @property
    def accepted(self) -> Array:
        """Whether the candidate committed."""
        return self.evidence.accepted


@final
class SHAKERATTLEPlan(StrictModule):
    """Static tolerances and resource bounds for prepared SHAKE/RATTLE."""

    maximum_projection_steps: int = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    maximum_constraint_entries: int = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_projection_steps: int = 8,
        constraint_tolerance: float = 1.0e-10,
        rank_tolerance: float = 1.0e-12,
        condition_limit: float = 1.0e12,
        maximum_constraint_entries: int = 1_000_000,
    ) -> None:
        steps = int(maximum_projection_steps)
        tolerance = float(constraint_tolerance)
        rank = float(rank_tolerance)
        condition = float(condition_limit)
        entries = int(maximum_constraint_entries)
        if steps <= 0:
            raise ValueError("maximum_projection_steps must be positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("constraint_tolerance must be finite and positive.")
        if not np.isfinite(rank) or rank <= 0.0:
            raise ValueError("rank_tolerance must be finite and positive.")
        if not np.isfinite(condition) or condition <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one.")
        if entries < 1:
            raise ValueError("maximum_constraint_entries must be positive.")
        materialization = MaterializationPolicy(
            max_entries=entries,
            max_bytes=max(8, 8 * entries),
        )
        self.maximum_projection_steps = steps
        self.constraint_tolerance = tolerance
        self.rank_tolerance = rank
        self.condition_limit = condition
        self.maximum_constraint_entries = entries
        self.linear_policy = LinearSolvePolicy(
            DenseSVD(),
            rank=RankPolicy(relative_cutoff=rank, require_full_rank=True),
            materialization=materialization,
            failure=FailurePolicy("status"),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shake-rattle-plan",
                "maximum_projection_steps": steps,
                "constraint_tolerance": tolerance.hex(),
                "rank_tolerance": rank.hex(),
                "condition_limit": condition.hex(),
                "maximum_constraint_entries": entries,
            }
        )

    def prepare(
        self,
        inverse_mass: ArrayLike,
        potential_gradient: Callable[[Array, object], Array],
        constraint: Callable[[Array, object], Array],
        /,
    ) -> PreparedSHAKERATTLEPlan:
        """Bind masses and differentiable mechanics callbacks."""
        return PreparedSHAKERATTLEPlan(
            self,
            inverse_mass,
            potential_gradient,
            constraint,
        )


@final
class PreparedSHAKERATTLEPlan(StrictModule):
    """Mass-bound SHAKE/RATTLE with operator Jacobian and transpose actions."""

    plan: SHAKERATTLEPlan
    inverse_mass: Array
    potential_gradient: Callable[[Array, object], Array]
    constraint: Callable[[Array, object], Array]
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SHAKERATTLEPlan,
        inverse_mass: ArrayLike,
        potential_gradient: Callable[[Array, object], Array],
        constraint: Callable[[Array, object], Array],
        /,
    ) -> None:
        if not isinstance(plan, SHAKERATTLEPlan):
            raise TypeError("plan must be a SHAKERATTLEPlan.")
        inverse_mass_ = jnp.asarray(inverse_mass)
        if (
            inverse_mass_.ndim != 1
            or inverse_mass_.size == 0
            or not jnp.issubdtype(inverse_mass_.dtype, jnp.floating)
        ):
            raise ValueError("inverse_mass must be a nonempty floating vector.")
        host_mass = np.asarray(inverse_mass_)
        if not np.all(np.isfinite(host_mass)) or np.any(host_mass <= 0.0):
            raise ValueError("inverse_mass must be finite and strictly positive.")
        if not callable(potential_gradient) or not callable(constraint):
            raise TypeError("potential_gradient and constraint must be callable.")
        self.plan = plan
        self.inverse_mass = inverse_mass_
        self.potential_gradient = potential_gradient
        self.constraint = constraint
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-shake-rattle",
                "plan": plan.plan_id,
                "dimension": inverse_mass_.size,
                "dtype": inverse_mass_.dtype.str,
            }
        )

    def constraint_operator(
        self, configuration: Array, args: object = None, /
    ) -> AbstractLinearOperator:
        """Native matrix-free Jacobian with exact transpose action."""
        point = jnp.asarray(configuration)
        if point.shape != self.inverse_mass.shape:
            raise ValueError("configuration shape does not match inverse_mass.")
        linearization = prepare_linearization(
            lambda value: jnp.asarray(self.constraint(value, args)).reshape((-1,)),
            point,
            linearization_id=f"{self.prepared_id}:constraint",
        )
        return JacobianLinearOperator(
            linearization,
            operator_id=f"{self.prepared_id}:constraint-jacobian",
        )

    def _gram_solve(
        self,
        operator: AbstractLinearOperator,
        right_hand_side: Array,
        /,
    ) -> tuple[Array, Array, Array, Array, Array]:
        target = operator.target
        if target.size == 0:
            zero = jnp.zeros_like(self.inverse_mass)
            return (
                zero,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(1.0, dtype=self.inverse_mass.dtype),
                jnp.asarray(int(LinearSolveStatus.SUCCESS), dtype=jnp.int32),
                jnp.asarray(True),
            )
        if target.size**2 > self.plan.maximum_constraint_entries:
            return (
                jnp.zeros_like(self.inverse_mass),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(jnp.inf, dtype=self.inverse_mass.dtype),
                jnp.asarray(int(LinearSolveStatus.CAPABILITY_REJECTED), dtype=jnp.int32),
                jnp.asarray(False),
            )

        def gram_action(multiplier: Array) -> Array:
            transpose = operator.transpose_mv(multiplier)
            return operator.mv(self.inverse_mass * transpose)

        gram = FunctionLinearOperator(
            gram_action,
            source=target,
            target=target,
            operator_id=f"{operator.operator_id}:mass-gram",
        )
        result = solve(
            MinimumNormProblem(gram),
            right_hand_side,
            policy=self.plan.linear_policy,
        )
        rank = result.diagnostics.rank
        condition = result.diagnostics.condition_estimate
        full_rank = rank == target.size
        conditioned = jnp.isfinite(condition) & (condition <= self.plan.condition_limit)
        successful = result.successful & full_rank & conditioned
        correction = self.inverse_mass * operator.transpose_mv(result.value)
        return correction, rank, condition, result.status, successful

    def _position_projection(
        self, trial: Array, args: object, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        dtype = trial.dtype
        integer_zero = jnp.asarray(0, dtype=jnp.int32)
        initial = (
            trial,
            jnp.asarray(True),
            integer_zero,
            integer_zero,
            jnp.asarray(1.0, dtype=dtype),
            jnp.asarray(int(LinearSolveStatus.SUCCESS), dtype=jnp.int32),
        )

        def project(
            _: int,
            carry: tuple[Array, Array, Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array, Array, Array]:
            configuration, projection_ok, used, rank, condition, linear_status = carry
            residual = jnp.asarray(self.constraint(configuration, args)).reshape((-1,))
            residual_norm = jnp.linalg.norm(residual)
            execute = (
                projection_ok
                & jnp.isfinite(residual_norm)
                & (residual_norm > self.plan.constraint_tolerance)
            )
            operator = self.constraint_operator(configuration, args)
            correction, solved_rank, solved_condition, solved_status, solved = (
                self._gram_solve(operator, -residual)
            )
            candidate = configuration + correction
            candidate_finite = jnp.all(jnp.isfinite(candidate))
            accept_correction = execute & solved & candidate_finite
            configuration = jnp.where(
                accept_correction,
                candidate,
                configuration,
            )
            failed = execute & ~(solved & candidate_finite)
            return (
                configuration,
                projection_ok & ~failed,
                used + execute.astype(jnp.int32),
                jnp.where(projection_ok, solved_rank, rank),
                jnp.where(projection_ok, solved_condition, condition),
                jnp.where(projection_ok, solved_status, linear_status),
            )

        configuration, ok, used, rank, condition, status = jax.lax.fori_loop(
            0,
            self.plan.maximum_projection_steps,
            project,
            initial,
        )
        residual = jnp.linalg.norm(
            jnp.asarray(self.constraint(configuration, args)).reshape((-1,))
        )
        return configuration, residual, used, rank, condition, status

    def step(
        self,
        state: ConstrainedMechanicalState,
        step_size: ArrayLike,
        /,
        *,
        args: object = None,
    ) -> ConstrainedMechanicalStep:
        """Advance one bounded kick-drift-kick step and atomically commit it."""
        if not isinstance(state, ConstrainedMechanicalState):
            raise TypeError("state must be ConstrainedMechanicalState.")
        if state.configuration.shape != self.inverse_mass.shape:
            raise ValueError("State dimension does not match inverse_mass.")
        if state.configuration.dtype != self.inverse_mass.dtype:
            raise TypeError("State and inverse_mass must have the same dtype.")
        step = jnp.asarray(step_size, dtype=state.configuration.dtype)
        if step.ndim != 0:
            raise ValueError("step_size must be scalar.")
        first_gradient = jnp.asarray(
            self.potential_gradient(state.configuration, args),
            dtype=state.configuration.dtype,
        )
        if first_gradient.shape != state.configuration.shape:
            raise ValueError("potential_gradient changed the configuration shape.")
        half_momentum = state.momentum - 0.5 * step * first_gradient
        trial = state.configuration + step * self.inverse_mass * half_momentum
        (
            configuration,
            position_residual,
            iterations,
            position_rank,
            position_condition,
            position_linear_status,
        ) = self._position_projection(trial, args)
        second_gradient = jnp.asarray(
            self.potential_gradient(configuration, args),
            dtype=state.configuration.dtype,
        )
        if second_gradient.shape != state.configuration.shape:
            raise ValueError("potential_gradient changed the configuration shape.")
        trial_momentum = half_momentum - 0.5 * step * second_gradient
        operator = self.constraint_operator(configuration, args)
        velocity_defect = operator.mv(self.inverse_mass * trial_momentum)
        (
            momentum_correction,
            velocity_rank,
            velocity_condition,
            velocity_linear_status,
            velocity_solved,
        ) = self._gram_solve(operator, velocity_defect)
        momentum = trial_momentum - momentum_correction / self.inverse_mass
        velocity_residual = jnp.linalg.norm(operator.mv(self.inverse_mass * momentum))
        trial_kinetic = (
            0.5
            * jnp.vdot(
                trial_momentum,
                self.inverse_mass * trial_momentum,
            ).real
        )
        projected_kinetic = (
            0.5
            * jnp.vdot(
                momentum,
                self.inverse_mass * momentum,
            ).real
        )
        projection_work = projected_kinetic - trial_kinetic
        constraint_rank = jnp.minimum(position_rank, velocity_rank)
        constraint_condition = jnp.maximum(
            position_condition,
            velocity_condition,
        )
        finite = (
            jnp.all(jnp.isfinite(configuration))
            & jnp.all(jnp.isfinite(momentum))
            & jnp.isfinite(position_residual)
            & jnp.isfinite(velocity_residual)
            & jnp.isfinite(projection_work)
        )
        valid_step = jnp.isfinite(step) & (step > 0.0)
        position_ok = (position_residual <= self.plan.constraint_tolerance) & (
            position_linear_status == int(LinearSolveStatus.SUCCESS)
        )
        velocity_ok = (
            velocity_solved
            & (velocity_residual <= self.plan.constraint_tolerance)
            & (velocity_linear_status == int(LinearSolveStatus.SUCCESS))
        )
        target_size = operator.target.size
        full_rank = constraint_rank == target_size
        conditioned = jnp.isfinite(constraint_condition) & (
            constraint_condition <= self.plan.condition_limit
        )
        accepted = (
            valid_step & finite & position_ok & velocity_ok & full_rank & conditioned
        )
        status = jnp.asarray(int(ConstrainedMechanicsStatus.SUCCESS), dtype=jnp.int32)
        status = jnp.where(
            valid_step & finite & full_rank & conditioned & ~velocity_ok,
            int(ConstrainedMechanicsStatus.VELOCITY_PROJECTION_FAILED),
            status,
        )
        status = jnp.where(
            valid_step & finite & full_rank & conditioned & ~position_ok,
            int(ConstrainedMechanicsStatus.POSITION_PROJECTION_FAILED),
            status,
        )
        status = jnp.where(
            valid_step & finite & full_rank & ~conditioned,
            int(ConstrainedMechanicsStatus.CONDITION_LIMIT_REACHED),
            status,
        )
        status = jnp.where(
            valid_step & finite & ~full_rank,
            int(ConstrainedMechanicsStatus.RANK_DEFICIENT),
            status,
        )
        status = jnp.where(
            valid_step & ~finite,
            int(ConstrainedMechanicsStatus.NONFINITE),
            status,
        )
        status = jnp.where(
            ~valid_step,
            int(ConstrainedMechanicsStatus.INVALID_STEP),
            status,
        ).astype(jnp.int32)
        safe_state = ConstrainedMechanicalState(
            jnp.where(accepted, configuration, state.configuration),
            jnp.where(accepted, momentum, state.momentum),
        )
        evidence = ConstrainedMechanicsEvidence(
            status=status,
            position_residual=position_residual,
            velocity_residual=velocity_residual,
            position_iterations=iterations,
            projection_work=projection_work,
            constraint_rank=constraint_rank,
            constraint_condition=constraint_condition,
            position_linear_status=position_linear_status,
            velocity_linear_status=velocity_linear_status,
            finite=finite,
            accepted=accepted,
            derivative_available=accepted,
            prepared_id=self.prepared_id,
        )
        return ConstrainedMechanicalStep(safe_state, evidence)


__all__ = [
    "ConstrainedMechanicalState",
    "ConstrainedMechanicalStep",
    "ConstrainedMechanicsEvidence",
    "ConstrainedMechanicsStatus",
    "PreparedSHAKERATTLEPlan",
    "SHAKERATTLEPlan",
]
