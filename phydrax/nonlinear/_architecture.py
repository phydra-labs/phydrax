#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._tree_math import tree_allfinite
from ..linalg import (
    AbstractLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    solve as solve_linear,
)
from ._linearization import JacobianPolicy, prepare_jacobian
from ._precision import NonlinearPrecisionPolicy
from ._types import NonlinearSystemProblem, NonlinearTermination
from ._work import NonlinearWork, NonlinearWorkBudget


class DirectionStatus(IntEnum):
    SUCCESS = 0
    LINEAR_FAILURE = 1
    NONDESCENT = 2
    NONFINITE = 3
    BUDGET_EXHAUSTED = 4


class GlobalizationStatus(IntEnum):
    ACCEPTED = 0
    NO_PROGRESS = 1
    DOMAIN_FAILURE = 2
    NONFINITE = 3
    BUDGET_EXHAUSTED = 4


class NonlinearModel(StrictModule):
    """Physical point, residual, linear model, and exact preparation work."""

    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    operator: AbstractLinearOperator
    residual_norm: Array
    merit: Array
    work: NonlinearWork
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state: PyTree[Any],
        residual: PyTree[Any],
        auxiliary: Any,
        operator: AbstractLinearOperator,
        residual_norm: Any,
        merit: Any,
        work: NonlinearWork,
        precision_evidence: PrecisionEvidenceEnvelope,
        precision_policy_id: str,
        model_id: str,
    ):
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be AbstractLinearOperator.")
        if not isinstance(work, NonlinearWork):
            raise TypeError("work must be NonlinearWork.")
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        precision_identifier = str(precision_policy_id)
        if not precision_identifier:
            raise ValueError("precision_policy_id must be non-empty.")
        identifier = str(model_id)
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        self.state = operator.source.validate(state)
        self.residual = operator.target.validate(residual)
        self.auxiliary = auxiliary
        self.operator = operator
        self.residual_norm = jnp.asarray(residual_norm)
        self.merit = jnp.asarray(merit)
        self.work = work
        self.precision_evidence = precision_evidence
        self.precision_policy_id = precision_identifier
        self.model_id = identifier


class DirectionResult(StrictModule):
    direction: PyTree[Array]
    model_image: PyTree[Array]
    slope: Array
    predicted_reduction: Array
    status: Array
    work: NonlinearWork
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    direction_id: str = eqx.field(static=True)

    @property
    def usable(self) -> Array:
        return self.status == int(DirectionStatus.SUCCESS)


class GlobalizationResult(StrictModule):
    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    residual_norm: Array
    step_norm: Array
    rate: Array
    status: Array
    work: NonlinearWork
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    globalization_id: str = eqx.field(static=True)

    @property
    def accepted(self) -> Array:
        return self.status == int(GlobalizationStatus.ACCEPTED)


class NonlinearCertificate(StrictModule):
    residual_norm: Array
    threshold: Array
    finite: Array
    valid: Array
    certified: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)


class AbstractNonlinearModelPolicy(StrictModule):
    @abc.abstractmethod
    def prepare(
        self,
        problem: NonlinearSystemProblem,
        state: PyTree[Any],
        args: Any,
        /,
    ) -> NonlinearModel:
        raise NotImplementedError


class AbstractDirectionPolicy(StrictModule):
    @abc.abstractmethod
    def compute(
        self,
        model: NonlinearModel,
        budget: NonlinearWorkBudget,
        /,
    ) -> DirectionResult:
        raise NotImplementedError


class AbstractGlobalizationPolicy(StrictModule):
    @abc.abstractmethod
    def apply(
        self,
        problem: NonlinearSystemProblem,
        model: NonlinearModel,
        direction: DirectionResult,
        args: Any,
        budget: NonlinearWorkBudget,
        /,
    ) -> GlobalizationResult:
        raise NotImplementedError


class AbstractNonlinearCertificate(StrictModule):
    @abc.abstractmethod
    def certify(
        self,
        problem: NonlinearSystemProblem,
        state: PyTree[Any],
        residual: PyTree[Any],
        auxiliary: Any,
        termination: NonlinearTermination,
        initial_residual_norm: Any,
        args: Any,
        /,
    ) -> NonlinearCertificate:
        raise NotImplementedError


class RootLinearModelPolicy(AbstractNonlinearModelPolicy):
    jacobian: JacobianPolicy
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        jacobian: JacobianPolicy | None = None,
        /,
        *,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        policy = JacobianPolicy() if jacobian is None else jacobian
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(policy, JacobianPolicy):
            raise TypeError("jacobian must be JacobianPolicy or None.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.jacobian = policy
        self.precision = precision_

    def prepare(self, problem, state, args, /) -> NonlinearModel:
        prepared = prepare_jacobian(problem, state, self.jacobian, args)
        self.precision.validate_trees(state, prepared.residual)
        self.precision.validate_accumulation_space(prepared.operator.target)
        residual_norm = self.precision.norm(
            prepared.operator.target,
            prepared.residual,
        )
        return NonlinearModel(
            state=state,
            residual=prepared.residual,
            auxiliary=prepared.auxiliary,
            operator=prepared.operator,
            residual_norm=residual_norm,
            merit=self.precision.decision(0.5 * residual_norm * residual_norm),
            work=NonlinearWork(
                residual_evaluations=prepared.residual_evaluations,
                jacobian_preparations=1,
            ),
            precision_evidence=self.precision.evidence_for(
                state,
                prepared.residual,
            ),
            precision_policy_id=self.precision.policy_id,
            model_id=f"root-linear/{prepared.derivative_id}",
        )


class NewtonDirectionPolicy(AbstractDirectionPolicy):
    linear: LinearSolvePolicy
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        linear: LinearSolvePolicy | None = None,
        /,
        *,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        policy = LinearSolvePolicy() if linear is None else linear
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.linear = policy
        self.precision = precision_

    def compute(self, model, budget, /) -> DirectionResult:
        if not isinstance(model, NonlinearModel):
            raise TypeError("model must be NonlinearModel.")
        if not isinstance(budget, NonlinearWorkBudget):
            raise TypeError("budget must be NonlinearWorkBudget.")
        right_hand_side = jax.tree.map(jnp.negative, model.residual)
        linear_result = solve_linear(
            LinearSystem(model.operator),
            right_hand_side,
            policy=self.precision.bind_linear(self.linear),
        )
        direction = linear_result.value
        image = model.operator.mv(direction)
        slope = self.precision.decision(
            jnp.real(
                self.precision.inner(
                    model.operator.target,
                    model.residual,
                    image,
                )
            )
        )
        iterations = jnp.sum(
            linear_result.diagnostics.iterations,
            dtype=jnp.int32,
        )
        work = NonlinearWork(
            jvp_evaluations=jnp.sum(
                linear_result.diagnostics.matvec_count,
                dtype=jnp.int32,
            )
            + 1,
            vjp_evaluations=jnp.sum(
                linear_result.diagnostics.adjoint_matvec_count,
                dtype=jnp.int32,
            ),
            linear_setups=1,
            linear_solves=1,
            linear_iterations=iterations,
        )
        finite = tree_allfinite(direction) & jnp.isfinite(slope)
        permitted = budget.permits(work)
        status = jnp.where(
            ~permitted,
            int(DirectionStatus.BUDGET_EXHAUSTED),
            jnp.where(
                ~linear_result.diagnostics.converged,
                int(DirectionStatus.LINEAR_FAILURE),
                jnp.where(
                    ~finite,
                    int(DirectionStatus.NONFINITE),
                    jnp.where(
                        slope >= 0.0,
                        int(DirectionStatus.NONDESCENT),
                        int(DirectionStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return DirectionResult(
            direction=direction,
            model_image=image,
            slope=slope,
            predicted_reduction=-slope,
            status=status,
            work=work,
            precision_evidence=model.precision_evidence,
            precision_policy_id=self.precision.policy_id,
            direction_id="newton",
        )


class ResidualArmijoPolicy(AbstractGlobalizationPolicy):
    initial_rate: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    sufficient_decrease: float = eqx.field(static=True)
    minimum_rate: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        *,
        initial_rate: float = 1.0,
        contraction: float = 0.5,
        sufficient_decrease: float = 1e-4,
        minimum_rate: float = 1e-10,
        maximum_steps: int = 24,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        values = tuple(
            float(value)
            for value in (
                initial_rate,
                contraction,
                sufficient_decrease,
                minimum_rate,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Armijo values must be finite and positive.")
        if not values[1] < 1.0 or not values[2] < 1.0:
            raise ValueError("Armijo contraction and decrease must be below one.")
        steps = int(maximum_steps)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        (
            self.initial_rate,
            self.contraction,
            self.sufficient_decrease,
            self.minimum_rate,
        ) = values
        self.maximum_steps = steps
        self.precision = precision_

    def apply(self, problem, model, direction, args, budget, /):
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be NonlinearSystemProblem.")
        if not isinstance(model, NonlinearModel):
            raise TypeError("model must be NonlinearModel.")
        if not isinstance(direction, DirectionResult):
            raise TypeError("direction must be DirectionResult.")
        if not isinstance(budget, NonlinearWorkBudget):
            raise TypeError("budget must be NonlinearWorkBudget.")
        self.precision.validate_accumulation_space(model.operator.source)
        self.precision.validate_accumulation_space(model.operator.target)

        class _Search(StrictModule):
            state: PyTree[Array]
            residual: PyTree[Array]
            auxiliary: Any
            norm: Array
            rate: Array
            evaluations: Array
            accepted: Array
            finite_seen: Array
            valid_seen: Array

        search = _Search(
            state=model.state,
            residual=model.residual,
            auxiliary=model.auxiliary,
            norm=model.residual_norm,
            rate=self.precision.decision(self.initial_rate),
            evaluations=jnp.asarray(0, dtype=jnp.int32),
            accepted=jnp.asarray(False),
            finite_seen=jnp.asarray(False),
            valid_seen=jnp.asarray(False),
        )

        def condition(item):
            trial_work = NonlinearWork(
                residual_evaluations=item.evaluations + 1,
                validity_evaluations=item.evaluations + 1,
            )
            return (
                direction.usable
                & ~item.accepted
                & (item.evaluations < self.maximum_steps)
                & (item.rate >= self.minimum_rate)
                & budget.permits(trial_work)
            )

        def body(item):
            candidate = jax.tree.map(
                lambda value, delta: jnp.asarray(
                    value + item.rate * delta,
                    dtype=value.dtype,
                ),
                model.state,
                direction.direction,
            )
            residual, auxiliary = problem.evaluate(candidate, args)
            norm = self.precision.norm(model.operator.target, residual)
            finite = tree_allfinite(candidate) & tree_allfinite(residual)
            valid = problem.valid(candidate, residual, auxiliary, args)
            accepted = (
                finite
                & valid
                & (
                    self.precision.decision(0.5 * norm * norm)
                    <= model.merit
                    + self.sufficient_decrease * item.rate * direction.slope
                )
            )
            return _Search(
                state=jax.tree.map(
                    lambda proposed, old: jnp.where(accepted, proposed, old),
                    candidate,
                    item.state,
                ),
                residual=jax.tree.map(
                    lambda proposed, old: jnp.where(accepted, proposed, old),
                    residual,
                    item.residual,
                ),
                auxiliary=jax.tree.map(
                    lambda proposed, old: jnp.where(accepted, proposed, old),
                    auxiliary,
                    item.auxiliary,
                ),
                norm=jnp.where(accepted, norm, item.norm),
                rate=jnp.where(
                    accepted,
                    item.rate,
                    self.contraction * item.rate,
                ),
                evaluations=item.evaluations + 1,
                accepted=accepted,
                finite_seen=item.finite_seen | finite,
                valid_seen=item.valid_seen | (finite & valid),
            )

        search = jax.lax.while_loop(condition, body, search)
        work = NonlinearWork(
            residual_evaluations=search.evaluations,
            validity_evaluations=search.evaluations,
        )
        status = jnp.where(
            search.accepted,
            int(GlobalizationStatus.ACCEPTED),
            jnp.where(
                ~budget.permits(work),
                int(GlobalizationStatus.BUDGET_EXHAUSTED),
                jnp.where(
                    search.valid_seen,
                    int(GlobalizationStatus.NO_PROGRESS),
                    jnp.where(
                        search.finite_seen,
                        int(GlobalizationStatus.DOMAIN_FAILURE),
                        int(GlobalizationStatus.NONFINITE),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        step = jax.tree.map(
            lambda new, old: new - old,
            search.state,
            model.state,
        )
        return GlobalizationResult(
            state=search.state,
            residual=search.residual,
            auxiliary=search.auxiliary,
            residual_norm=search.norm,
            step_norm=self.precision.norm(model.operator.source, step),
            rate=search.rate,
            status=status,
            work=work,
            precision_evidence=self.precision.evidence_for(
                search.state,
                search.residual,
            ),
            precision_policy_id=self.precision.policy_id,
            globalization_id="residual-armijo",
        )


class RootResidualCertificate(AbstractNonlinearCertificate):
    precision: NonlinearPrecisionPolicy

    def __init__(self, precision: NonlinearPrecisionPolicy | None = None, /):
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.precision = precision_

    def certify(
        self,
        problem,
        state,
        residual,
        auxiliary,
        termination,
        initial_residual_norm,
        args,
        /,
    ) -> NonlinearCertificate:
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be NonlinearSystemProblem.")
        if problem.residual_space is None:
            raise ValueError("Root certification requires a bound residual space.")
        self.precision.validate_trees(state, residual)
        self.precision.validate_accumulation_space(problem.residual_space)
        self.precision.validate_tolerance(termination.absolute_residual)
        norm = self.precision.norm(problem.residual_space, residual)
        finite = tree_allfinite(state) & tree_allfinite(residual)
        valid = problem.valid(state, residual, auxiliary, args)
        threshold = self.precision.decision(
            termination.residual_threshold(initial_residual_norm)
        )
        return NonlinearCertificate(
            residual_norm=norm,
            threshold=threshold,
            finite=finite,
            valid=valid,
            certified=finite & valid & (norm <= threshold),
            precision_evidence=self.precision.evidence_for(state, residual),
            precision_policy_id=self.precision.policy_id,
            certificate_id="physical-root-residual",
        )


__all__ = [
    "AbstractDirectionPolicy",
    "AbstractGlobalizationPolicy",
    "AbstractNonlinearCertificate",
    "AbstractNonlinearModelPolicy",
    "DirectionResult",
    "DirectionStatus",
    "GlobalizationResult",
    "GlobalizationStatus",
    "NewtonDirectionPolicy",
    "NonlinearCertificate",
    "NonlinearModel",
    "ResidualArmijoPolicy",
    "RootLinearModelPolicy",
    "RootResidualCertificate",
]
