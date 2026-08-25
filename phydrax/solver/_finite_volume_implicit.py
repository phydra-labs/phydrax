#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    PreparedFiniteVolumeDynamics,
    PreparedTriangleFiniteVolumeDynamics,
    PreparedUnstructuredFiniteVolumeDynamics,
)
from ..equations._hyperbolic_systems import AbstractAdmissibleSystem
from ..linalg import ArraySpace, DiagonalPairing
from ..nonlinear import (
    NewtonKrylov,
    NewtonTrustRegion,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
    solve_prepared_nonlinear,
)


ImplicitFVDynamics = (
    PreparedFiniteVolumeDynamics
    | PreparedTriangleFiniteVolumeDynamics
    | PreparedUnstructuredFiniteVolumeDynamics
)
ImplicitFVMethod = NewtonKrylov | NewtonTrustRegion


def _termination_identity(termination: NonlinearTermination, /) -> tuple[Any, ...]:
    return (
        termination.absolute_residual,
        termination.relative_residual,
        termination.absolute_step,
        termination.relative_step,
        termination.maximum_steps,
        termination.maximum_evaluations,
        termination.maximum_linear_iterations,
        termination.divergence_factor,
    )


class FiniteVolumeImplicitStage(StrictModule):
    """Dynamic data for one backward-Euler finite-volume stage."""

    previous_state: Array
    time: Array
    step_size: Array
    dynamics_args: Any

    def __init__(
        self,
        previous_state: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike,
        dynamics_args: Any = None,
        /,
    ):
        self.previous_state = jnp.asarray(previous_state)
        self.time = jnp.asarray(time).reshape(())
        self.step_size = jnp.asarray(step_size).reshape(())
        self.dynamics_args = dynamics_args


class _FiniteVolumeBackwardEulerResidual(StrictModule, NonTrainableState):
    dynamics: ImplicitFVDynamics

    def __init__(self, dynamics: ImplicitFVDynamics, /):
        self.dynamics = dynamics

    def __call__(self, candidate: Array, stage: FiniteVolumeImplicitStage, /) -> Array:
        if not isinstance(stage, FiniteVolumeImplicitStage):
            raise TypeError("Backward-Euler residual requires FiniteVolumeImplicitStage.")
        precision = self.dynamics.precision
        value = precision.storage(candidate)
        difference = precision.reduction(value) - precision.reduction(
            stage.previous_state
        )
        rate = precision.reduction(
            self.dynamics(
                precision.decision(stage.time + stage.step_size),
                value,
                stage.dynamics_args,
            )
        )
        return precision.storage(difference - precision.decision(stage.step_size) * rate)

    def valid(
        self,
        candidate: Array,
        residual: Array,
        auxiliary: Any,
        stage: FiniteVolumeImplicitStage,
        /,
    ) -> Array:
        del residual, auxiliary, stage
        if isinstance(self.dynamics.system, AbstractAdmissibleSystem):
            return jnp.all(self.dynamics.system.admissible(candidate))
        return jnp.asarray(True)


class FiniteVolumeImplicitStepResult(StrictModule):
    """Fail-closed accepted state plus complete nonlinear evidence."""

    state: Array
    time: Array
    attempted_step_size: Array
    accepted_step_size: Array
    nonlinear: NonlinearResult
    temporal_method_id: str = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.nonlinear.successful


class FiniteVolumeBackwardEulerPlan(StrictModule, NonTrainableState):
    """Matrix-free backward Euler using the canonical nonlinear solver stack."""

    dynamics: ImplicitFVDynamics
    residual_operator: _FiniteVolumeBackwardEulerResidual
    problem: NonlinearSystemProblem
    method: ImplicitFVMethod
    termination: NonlinearTermination
    nonlinear_precision: NonlinearPrecisionPolicy
    temporal_method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: ImplicitFVDynamics,
        /,
        *,
        method: ImplicitFVMethod | None = None,
        termination: NonlinearTermination | None = None,
        nonlinear_precision: NonlinearPrecisionPolicy | None = None,
    ):
        if not isinstance(
            dynamics,
            (
                PreparedFiniteVolumeDynamics,
                PreparedTriangleFiniteVolumeDynamics,
                PreparedUnstructuredFiniteVolumeDynamics,
            ),
        ):
            raise TypeError("dynamics must be prepared finite-volume dynamics.")
        method_ = NewtonKrylov() if method is None else method
        if not isinstance(method_, (NewtonKrylov, NewtonTrustRegion)):
            raise TypeError("method must be NewtonKrylov or NewtonTrustRegion.")
        termination_ = NonlinearTermination() if termination is None else termination
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination.")
        fv_precision = dynamics.precision
        nonlinear_precision_ = (
            NonlinearPrecisionPolicy(
                state_dtype=fv_precision.storage_dtype,
                residual_dtype=fv_precision.storage_dtype,
                direction_dtype=fv_precision.storage_dtype,
                accumulation_dtype=fv_precision.reduction_dtype,
                decision_dtype=fv_precision.reduction_dtype,
                certificate_dtype=fv_precision.reduction_dtype,
                output_dtype=fv_precision.storage_dtype,
            )
            if nonlinear_precision is None
            else nonlinear_precision
        )
        if not isinstance(nonlinear_precision_, NonlinearPrecisionPolicy):
            raise TypeError("nonlinear_precision must be NonlinearPrecisionPolicy.")
        residual = _FiniteVolumeBackwardEulerResidual(dynamics)
        problem_id = canonical_fingerprint(
            {
                "kind": "finite-volume-backward-euler-residual",
                "dynamics": dynamics.dynamics_id,
            }
        )
        state_shape = dynamics.discretization.state_shape
        cell_weights = jnp.broadcast_to(
            fv_precision.reduction(dynamics.discretization.cell_volumes[:, None]),
            state_shape,
        )
        state_space = ArraySpace(
            state_shape,
            dtype=jnp.dtype(fv_precision.storage_dtype),
            pairing=DiagonalPairing(cell_weights),
        )
        problem = NonlinearSystemProblem(
            residual,
            state_space=state_space,
            residual_space=state_space,
            validity=residual.valid,
            problem_id=problem_id,
        )
        self.dynamics = dynamics
        self.residual_operator = residual
        self.problem = problem
        self.method = method_
        self.termination = termination_
        self.nonlinear_precision = nonlinear_precision_
        self.temporal_method_id = "temporal:backward-euler"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-volume-backward-euler-plan",
                "dynamics": dynamics.dynamics_id,
                "method": method_.method_id,
                "problem": problem_id,
                "termination": _termination_identity(termination_),
                "precision": nonlinear_precision_.policy_id,
            }
        )

    def prepare(
        self,
        previous_state: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        initial_guess: ArrayLike | None = None,
        args: Any = None,
    ) -> "PreparedFiniteVolumeBackwardEulerStep":
        previous = jnp.asarray(previous_state)
        self.dynamics.precision.validate_state(previous)
        if previous.shape != self.dynamics.discretization.state_shape:
            raise ValueError(
                f"Implicit FV state must have shape {self.dynamics.discretization.state_shape}."
            )
        time_ = self.dynamics.precision.decision(time)
        step = self.dynamics.precision.decision(step_size)
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Implicit FV step_size must be positive and finite.",
        )
        stage = FiniteVolumeImplicitStage(previous, time_, step, args)
        guess = previous if initial_guess is None else jnp.asarray(initial_guess)
        self.dynamics.precision.validate_state(guess)
        nonlinear = prepare_nonlinear(
            self.problem,
            guess,
            method=self.method,
            termination=self.termination,
            args=stage,
            precision=self.nonlinear_precision,
        )
        return PreparedFiniteVolumeBackwardEulerStep(self, stage, nonlinear)

    def advance(
        self,
        previous_state: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        initial_guess: ArrayLike | None = None,
        args: Any = None,
    ) -> FiniteVolumeImplicitStepResult:
        return self.prepare(
            previous_state,
            time,
            step_size,
            initial_guess=initial_guess,
            args=args,
        ).solve()


class PreparedFiniteVolumeBackwardEulerStep(StrictModule, NonTrainableState):
    """Reusable symbolic Newton plan bound to one implicit FV stage."""

    plan: FiniteVolumeBackwardEulerPlan
    stage: FiniteVolumeImplicitStage
    nonlinear: PreparedNonlinearSolve

    def __init__(
        self,
        plan: FiniteVolumeBackwardEulerPlan,
        stage: FiniteVolumeImplicitStage,
        nonlinear: PreparedNonlinearSolve,
        /,
    ):
        if not isinstance(plan, FiniteVolumeBackwardEulerPlan):
            raise TypeError("plan must be FiniteVolumeBackwardEulerPlan.")
        if not isinstance(stage, FiniteVolumeImplicitStage):
            raise TypeError("stage must be FiniteVolumeImplicitStage.")
        if not isinstance(nonlinear, PreparedNonlinearSolve):
            raise TypeError("nonlinear must be PreparedNonlinearSolve.")
        self.plan = plan
        self.stage = stage
        self.nonlinear = nonlinear

    def refresh(
        self,
        previous_state: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        initial_guess: ArrayLike | None = None,
        args: Any = None,
    ) -> "PreparedFiniteVolumeBackwardEulerStep":
        previous = jnp.asarray(previous_state)
        self.plan.dynamics.precision.validate_state(previous)
        if previous.shape != self.plan.dynamics.discretization.state_shape:
            raise ValueError(
                "Implicit FV previous state must have shape "
                f"{self.plan.dynamics.discretization.state_shape}."
            )
        time_ = self.plan.dynamics.precision.decision(time)
        step = self.plan.dynamics.precision.decision(step_size)
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Implicit FV step_size must be positive and finite.",
        )
        stage = FiniteVolumeImplicitStage(previous, time_, step, args)
        guess = previous if initial_guess is None else jnp.asarray(initial_guess)
        self.plan.dynamics.precision.validate_state(guess)
        nonlinear = refresh_nonlinear(
            self.nonlinear,
            self.plan.problem,
            guess,
            args=stage,
        )
        return PreparedFiniteVolumeBackwardEulerStep(self.plan, stage, nonlinear)

    def solve(self, /) -> FiniteVolumeImplicitStepResult:
        nonlinear = solve_prepared_nonlinear(self.nonlinear)
        successful = nonlinear.successful
        accepted_state = jnp.where(successful, nonlinear.state, self.stage.previous_state)
        accepted_step = jnp.where(successful, self.stage.step_size, 0.0)
        accepted_time = self.stage.time + accepted_step
        return FiniteVolumeImplicitStepResult(
            state=self.plan.dynamics.precision.storage(accepted_state),
            time=self.plan.dynamics.precision.decision(accepted_time),
            attempted_step_size=self.plan.dynamics.precision.decision(
                self.stage.step_size
            ),
            accepted_step_size=self.plan.dynamics.precision.decision(accepted_step),
            nonlinear=nonlinear,
            temporal_method_id=self.plan.temporal_method_id,
            precision_evidence=self.plan.dynamics.precision.evidence(),
        )


__all__ = [
    "FiniteVolumeBackwardEulerPlan",
    "FiniteVolumeImplicitStage",
    "FiniteVolumeImplicitStepResult",
    "PreparedFiniteVolumeBackwardEulerStep",
]
