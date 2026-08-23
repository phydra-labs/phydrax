#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._tree_math import tree_allfinite, validate_inexact_tree
from ..linalg import AbstractVectorSpace
from ._newton import NewtonKrylov, NewtonTrustRegion
from ._prepared import (
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
    step_prepared_nonlinear,
)
from ._types import (
    AbstractNonlinearMethod,
    NonlinearDiagnostics,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._work import (
    NonlinearAttemptEvidence,
    NonlinearWork,
    NonlinearWorkBudget,
)


def _space_norm(space: AbstractVectorSpace, vector: PyTree[Any], /) -> Array:
    squared = jnp.real(space.inner(vector, vector))
    return jnp.sqrt(jnp.maximum(squared, 0.0))


class NonlinearUpdateStatus(IntEnum):
    """Terminal status for one bounded nonlinear update application."""

    APPLIED = 0
    NO_PROGRESS = 1
    DOMAIN_REJECTED = 2
    NONFINITE_INPUT = 3
    NONFINITE_EVALUATION = 4
    INNER_FAILURE = 5
    LINEAR_FAILURE = 6
    BUDGET_EXHAUSTED = 7


_UPDATE_STATUS_MESSAGES = {
    NonlinearUpdateStatus.APPLIED: "nonlinear update applied",
    NonlinearUpdateStatus.NO_PROGRESS: "nonlinear update made no certified progress",
    NonlinearUpdateStatus.DOMAIN_REJECTED: "nonlinear update left the physical domain",
    NonlinearUpdateStatus.NONFINITE_INPUT: "nonlinear update input is non-finite",
    NonlinearUpdateStatus.NONFINITE_EVALUATION: "nonlinear update evaluation is non-finite",
    NonlinearUpdateStatus.INNER_FAILURE: "inner nonlinear method failed",
    NonlinearUpdateStatus.LINEAR_FAILURE: "inner linear solve failed",
    NonlinearUpdateStatus.BUDGET_EXHAUSTED: "remaining work cannot fund the nonlinear update",
}


def nonlinear_update_status_message(status: int | NonlinearUpdateStatus, /) -> str:
    return _UPDATE_STATUS_MESSAGES[NonlinearUpdateStatus(int(status))]


class NonlinearUpdateCapabilities(StrictModule):
    """Static execution capabilities of one finite nonlinear update."""

    jit: bool = eqx.field(static=True)
    prepared_refresh: bool = eqx.field(static=True)
    differentiable_action: bool = eqx.field(static=True)
    exposes_linearization: bool = eqx.field(static=True)
    counts_complete: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        jit: bool,
        prepared_refresh: bool,
        differentiable_action: bool,
        exposes_linearization: bool = False,
        counts_complete: bool = True,
    ):
        self.jit = bool(jit)
        self.prepared_refresh = bool(prepared_refresh)
        self.differentiable_action = bool(differentiable_action)
        self.exposes_linearization = bool(exposes_linearization)
        self.counts_complete = bool(counts_complete)


class NonlinearUpdateControl(StrictModule):
    """Dynamic remaining work available to one nonlinear update."""

    budget: NonlinearWorkBudget

    def __init__(
        self,
        *,
        budget: NonlinearWorkBudget | None = None,
        maximum_residual_evaluations: Any = None,
        maximum_validity_evaluations: Any = None,
        maximum_jvp_evaluations: Any = None,
        maximum_vjp_evaluations: Any = None,
        maximum_jacobian_preparations: Any = None,
        maximum_linear_setups: Any = None,
        maximum_linear_refreshes: Any = None,
        maximum_linear_solves: Any = None,
        maximum_linear_iterations: Any = None,
        maximum_preconditioner_applications: Any = None,
        maximum_local_updates: Any = None,
    ):
        if budget is not None:
            if not isinstance(budget, NonlinearWorkBudget):
                raise TypeError("budget must be NonlinearWorkBudget or None.")
            if any(
                value is not None
                for value in (
                    maximum_residual_evaluations,
                    maximum_validity_evaluations,
                    maximum_jvp_evaluations,
                    maximum_vjp_evaluations,
                    maximum_jacobian_preparations,
                    maximum_linear_setups,
                    maximum_linear_refreshes,
                    maximum_linear_solves,
                    maximum_linear_iterations,
                    maximum_preconditioner_applications,
                    maximum_local_updates,
                )
            ):
                raise ValueError(
                    "Pass either budget or individual maximum work limits, not both."
                )
            self.budget = budget
            return

        def limit(value: Any) -> Any:
            return -1 if value is None else value

        self.budget = NonlinearWorkBudget(
            residual_evaluations=limit(maximum_residual_evaluations),
            validity_evaluations=limit(maximum_validity_evaluations),
            jvp_evaluations=limit(maximum_jvp_evaluations),
            vjp_evaluations=limit(maximum_vjp_evaluations),
            jacobian_preparations=limit(maximum_jacobian_preparations),
            linear_setups=limit(maximum_linear_setups),
            linear_refreshes=limit(maximum_linear_refreshes),
            linear_solves=limit(maximum_linear_solves),
            linear_iterations=limit(maximum_linear_iterations),
            preconditioner_applications=limit(maximum_preconditioner_applications),
            local_updates=limit(maximum_local_updates),
        )

    def permits(
        self,
        work: NonlinearWork | None = None,
        /,
        **legacy_counts: Any,
    ) -> Array:
        if work is not None and legacy_counts:
            raise ValueError("Pass work or count keywords, not both.")
        requested = NonlinearWork(**legacy_counts) if work is None else work
        if not isinstance(requested, NonlinearWork):
            raise TypeError("work must be NonlinearWork or None.")
        return self.budget.permits(requested)

    def consume(self, work: NonlinearWork, /) -> NonlinearUpdateControl:
        return NonlinearUpdateControl(budget=self.budget.consume(work))

    def split(
        self,
        count: int,
        /,
        *,
        reserve: NonlinearWork | None = None,
    ) -> NonlinearUpdateControl:
        return NonlinearUpdateControl(budget=self.budget.split(count, reserve=reserve))


class NonlinearUpdateDiagnostics(StrictModule):
    """Physical progress and exact work from one update application."""

    initial_residual_norm: Array
    final_residual_norm: Array
    step_norm: Array
    work: NonlinearWork
    residual_evaluations: Array
    validity_evaluations: Array
    jvp_evaluations: Array
    vjp_evaluations: Array
    jacobian_preparations: Array
    linear_setups: Array
    linear_refreshes: Array
    linear_solves: Array
    linear_iterations: Array
    preconditioner_applications: Array
    local_updates: Array
    accepted_steps: Array
    rejected_steps: Array
    domain_failures: Array
    nonfinite_trials: Array
    counts_complete: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_residual_norm: Any,
        final_residual_norm: Any,
        step_norm: Any,
        work: NonlinearWork | None = None,
        residual_evaluations: Any = 0,
        validity_evaluations: Any = 0,
        jvp_evaluations: Any = 0,
        vjp_evaluations: Any = 0,
        jacobian_preparations: Any = 0,
        linear_setups: Any = 0,
        linear_refreshes: Any = 0,
        linear_solves: Any = 0,
        linear_iterations: Any = 0,
        preconditioner_applications: Any = 0,
        local_updates: Any = 0,
        accepted_steps: Any = 0,
        rejected_steps: Any = 0,
        domain_failures: Any = 0,
        nonfinite_trials: Any = 0,
        counts_complete: bool = True,
    ):
        self.initial_residual_norm = jnp.asarray(initial_residual_norm)
        self.final_residual_norm = jnp.asarray(final_residual_norm)
        self.step_norm = jnp.asarray(step_norm)
        work_ = (
            NonlinearWork(
                residual_evaluations=residual_evaluations,
                validity_evaluations=validity_evaluations,
                jvp_evaluations=jvp_evaluations,
                vjp_evaluations=vjp_evaluations,
                jacobian_preparations=jacobian_preparations,
                linear_setups=linear_setups,
                linear_refreshes=linear_refreshes,
                linear_solves=linear_solves,
                linear_iterations=linear_iterations,
                preconditioner_applications=preconditioner_applications,
                local_updates=local_updates,
                complete=counts_complete,
            )
            if work is None
            else work
        )
        if not isinstance(work_, NonlinearWork):
            raise TypeError("work must be NonlinearWork or None.")
        self.work = work_
        self.residual_evaluations = work_.residual_evaluations
        self.validity_evaluations = work_.validity_evaluations
        self.jvp_evaluations = work_.jvp_evaluations
        self.vjp_evaluations = work_.vjp_evaluations
        self.jacobian_preparations = work_.jacobian_preparations
        self.linear_setups = work_.linear_setups
        self.linear_refreshes = work_.linear_refreshes
        self.linear_solves = work_.linear_solves
        self.linear_iterations = work_.linear_iterations
        self.preconditioner_applications = work_.preconditioner_applications
        self.local_updates = work_.local_updates
        values = tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (
                accepted_steps,
                rejected_steps,
                domain_failures,
                nonfinite_trials,
            )
        )
        (
            self.accepted_steps,
            self.rejected_steps,
            self.domain_failures,
            self.nonfinite_trials,
        ) = values
        self.counts_complete = work_.complete

    @classmethod
    def from_nonlinear(
        cls,
        diagnostics: NonlinearDiagnostics,
        /,
        *,
        step_norm: Any | None = None,
    ) -> NonlinearUpdateDiagnostics:
        if not isinstance(diagnostics, NonlinearDiagnostics):
            raise TypeError("diagnostics must be NonlinearDiagnostics.")
        return cls(
            initial_residual_norm=diagnostics.initial_residual_norm,
            final_residual_norm=diagnostics.final_residual_norm,
            step_norm=(diagnostics.final_step_norm if step_norm is None else step_norm),
            residual_evaluations=diagnostics.residual_evaluations,
            validity_evaluations=diagnostics.residual_evaluations,
            jvp_evaluations=diagnostics.jvp_evaluations,
            vjp_evaluations=diagnostics.vjp_evaluations,
            jacobian_preparations=diagnostics.jacobian_preparations,
            linear_setups=diagnostics.setup_refreshes,
            linear_refreshes=diagnostics.numeric_refreshes,
            linear_solves=diagnostics.linear_solves,
            linear_iterations=diagnostics.linear_iterations,
            preconditioner_applications=0,
            local_updates=0,
            accepted_steps=diagnostics.accepted_steps,
            rejected_steps=diagnostics.rejected_steps,
            domain_failures=diagnostics.domain_failures,
            nonfinite_trials=diagnostics.nonfinite_trials,
            counts_complete=diagnostics.counts_complete,
        )


class NonlinearUpdateProvenance(StrictModule):
    """Static identity of one update application and its bound plan."""

    problem_id: str = eqx.field(static=True)
    update_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    notes: str = eqx.field(static=True)

    def __init__(self, *, problem_id: str, update_id: str, plan_id: str, notes: str = ""):
        identifiers = tuple(str(value) for value in (problem_id, update_id, plan_id))
        if any(not value for value in identifiers):
            raise ValueError("Nonlinear update provenance identifiers must be non-empty.")
        self.problem_id, self.update_id, self.plan_id = identifiers
        self.notes = str(notes)


class NonlinearUpdateResult(StrictModule):
    """One physical state proposal with application-level evidence."""

    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    status: Array
    inner_status: Array
    diagnostics: NonlinearUpdateDiagnostics
    provenance: NonlinearUpdateProvenance
    components: tuple[NonlinearUpdateResult, ...]
    evidence: NonlinearAttemptEvidence

    def __init__(
        self,
        *,
        state: PyTree[Any],
        residual: PyTree[Any],
        auxiliary: Any,
        status: Any,
        diagnostics: NonlinearUpdateDiagnostics,
        provenance: NonlinearUpdateProvenance,
        inner_status: Any = -1,
        components: tuple[NonlinearUpdateResult, ...] = (),
        evidence: NonlinearAttemptEvidence | None = None,
    ):
        if not isinstance(diagnostics, NonlinearUpdateDiagnostics):
            raise TypeError("diagnostics must be NonlinearUpdateDiagnostics.")
        if not isinstance(provenance, NonlinearUpdateProvenance):
            raise TypeError("provenance must be NonlinearUpdateProvenance.")
        components_ = tuple(components)
        if not all(isinstance(value, NonlinearUpdateResult) for value in components_):
            raise TypeError("components must contain NonlinearUpdateResult values.")
        self.state = validate_inexact_tree(state, name="nonlinear update state")
        self.residual = validate_inexact_tree(residual, name="nonlinear update residual")
        self.auxiliary = auxiliary
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.inner_status = jnp.asarray(inner_status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.components = components_
        evidence_ = (
            NonlinearAttemptEvidence(
                component_id=provenance.update_id,
                status=self.status,
                accepted=self.applied,
                input_residual_norm=diagnostics.initial_residual_norm,
                output_residual_norm=diagnostics.final_residual_norm,
                work=diagnostics.work,
                failure_origin="update-status",
                children=tuple(value.evidence for value in components_),
            )
            if evidence is None
            else evidence
        )
        if not isinstance(evidence_, NonlinearAttemptEvidence):
            raise TypeError("evidence must be NonlinearAttemptEvidence or None.")
        self.evidence = evidence_

    @property
    def applied(self) -> Array:
        return self.status == int(NonlinearUpdateStatus.APPLIED)


class NonlinearUpdatePlan(StrictModule):
    """Symbolic binding of one update to physical vector spaces."""

    state_space: AbstractVectorSpace
    residual_space: AbstractVectorSpace
    problem_id: str = eqx.field(static=True)
    update_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_space: AbstractVectorSpace,
        residual_space: AbstractVectorSpace,
        /,
        *,
        problem_id: str,
        update_id: str,
    ):
        if not isinstance(state_space, AbstractVectorSpace) or not isinstance(
            residual_space, AbstractVectorSpace
        ):
            raise TypeError("Update plans require declared state and residual spaces.")
        problem_id_ = str(problem_id)
        update_id_ = str(update_id)
        if not problem_id_ or not update_id_:
            raise ValueError("Update plan identifiers must be non-empty.")
        plan_id = canonical_fingerprint(
            {
                "kind": "nonlinear-update-plan",
                "problem": problem_id_,
                "update": update_id_,
                "state_space": state_space.space_id,
                "residual_space": residual_space.space_id,
            }
        )
        self.state_space = state_space
        self.residual_space = residual_space
        self.problem_id = problem_id_
        self.update_id = update_id_
        self.plan_id = plan_id


class PreparedNonlinearUpdate(StrictModule):
    """Prepared finite-work update with reusable numerical state."""

    problem: NonlinearSystemProblem
    update: AbstractNonlinearUpdate
    plan: NonlinearUpdatePlan
    internal_state: Any
    reference_state: PyTree[Array]
    reference_residual: PyTree[Array]
    reference_auxiliary: Any
    numeric_version: Array

    def __init__(
        self,
        problem: NonlinearSystemProblem,
        update: AbstractNonlinearUpdate,
        plan: NonlinearUpdatePlan,
        internal_state: Any,
        reference_state: PyTree[Any],
        reference_residual: PyTree[Any],
        reference_auxiliary: Any,
        /,
        *,
        numeric_version: Any,
    ):
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be NonlinearSystemProblem.")
        if not isinstance(update, AbstractNonlinearUpdate):
            raise TypeError("update must be AbstractNonlinearUpdate.")
        if not isinstance(plan, NonlinearUpdatePlan):
            raise TypeError("plan must be NonlinearUpdatePlan.")
        if plan.problem_id != problem.problem_id or plan.update_id != update.update_id:
            raise ValueError("Prepared update identity does not match its plan.")
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        version = eqx.error_if(
            version, version < 0, "numeric_version must be non-negative."
        )
        self.problem = problem
        self.update = update
        self.plan = plan
        self.internal_state = internal_state
        self.numeric_version = version

        self.reference_state = plan.state_space.validate(reference_state)
        self.reference_residual = plan.residual_space.validate(reference_residual)
        self.reference_auxiliary = reference_auxiliary


class AbstractNonlinearUpdate(StrictModule):
    """Finite nonlinear work that proposes, but need not solve for, a physical state."""

    @property
    @abc.abstractmethod
    def update_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> NonlinearUpdateCapabilities:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def maximum_work(self) -> NonlinearWork:
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_internal(
        self,
        problem: NonlinearSystemProblem,
        state: PyTree[Any],
        args: Any,
        /,
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def _refresh_internal(
        self,
        internal_state: Any,
        problem: NonlinearSystemProblem,
        state: PyTree[Any],
        args: Any,
        /,
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def _apply(
        self,
        prepared: PreparedNonlinearUpdate,
        state: PyTree[Any],
        args: Any,
        control: NonlinearUpdateControl,
        /,
    ) -> tuple[NonlinearUpdateResult, Any]:
        raise NotImplementedError


class FunctionNonlinearUpdate(AbstractNonlinearUpdate):
    """Explicit callable boundary for one physical state proposal."""

    function: Any
    update_name: str = eqx.field(static=True)

    def __init__(self, function: Any, /, *, update_id: str = "function-update"):
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(update_id)
        if not identifier:
            raise ValueError("update_id must be non-empty.")
        self.function = function
        self.update_name = identifier

    @property
    def update_id(self) -> str:
        return self.update_name

    @property
    def capabilities(self) -> NonlinearUpdateCapabilities:
        return NonlinearUpdateCapabilities(
            jit=True,
            prepared_refresh=True,
            differentiable_action=True,
        )

    @property
    def maximum_work(self) -> NonlinearWork:
        return NonlinearWork(
            residual_evaluations=2,
            validity_evaluations=1,
        )

    def _prepare_internal(
        self,
        problem: NonlinearSystemProblem,
        state: PyTree[Any],
        args: Any,
        /,
    ) -> None:
        del problem, state, args
        return None

    def _refresh_internal(
        self,
        internal_state: Any,
        problem: NonlinearSystemProblem,
        state: PyTree[Any],
        args: Any,
        /,
    ) -> Any:
        del problem, state, args
        return internal_state

    def _apply(
        self,
        prepared: PreparedNonlinearUpdate,
        state: PyTree[Any],
        args: Any,
        control: NonlinearUpdateControl,
        /,
    ) -> tuple[NonlinearUpdateResult, Any]:
        problem = prepared.problem
        state_ = prepared.plan.state_space.validate(state)

        def skipped(_):
            diagnostics = NonlinearUpdateDiagnostics(
                initial_residual_norm=jnp.asarray(jnp.nan),
                final_residual_norm=jnp.asarray(jnp.nan),
                step_norm=0.0,
                work=NonlinearWork.zero(),
            )
            return (
                NonlinearUpdateResult(
                    state=state_,
                    residual=prepared.plan.residual_space.zeros(),
                    auxiliary=prepared.reference_auxiliary,
                    status=NonlinearUpdateStatus.BUDGET_EXHAUSTED,
                    diagnostics=diagnostics,
                    provenance=_provenance(prepared),
                ),
                prepared.internal_state,
            )

        def execute(_):
            initial_residual, _ = problem.evaluate(state_, args)
            initial_norm = _space_norm(
                prepared.plan.residual_space,
                initial_residual,
            )
            candidate = prepared.plan.state_space.validate(self.function(state_, args))
            residual, auxiliary = problem.evaluate(candidate, args)
            final_norm = _space_norm(prepared.plan.residual_space, residual)
            finite = tree_allfinite(candidate) & tree_allfinite(residual)
            valid = problem.valid(candidate, residual, auxiliary, args)
            status = jnp.where(
                ~finite,
                int(NonlinearUpdateStatus.NONFINITE_EVALUATION),
                jnp.where(
                    ~valid,
                    int(NonlinearUpdateStatus.DOMAIN_REJECTED),
                    int(NonlinearUpdateStatus.APPLIED),
                ),
            ).astype(jnp.int32)
            diagnostics = NonlinearUpdateDiagnostics(
                initial_residual_norm=initial_norm,
                final_residual_norm=final_norm,
                step_norm=_space_norm(
                    prepared.plan.state_space,
                    jax.tree.map(
                        lambda new, old: new - old,
                        candidate,
                        state_,
                    ),
                ),
                work=self.maximum_work,
                accepted_steps=(status == int(NonlinearUpdateStatus.APPLIED)).astype(
                    jnp.int32
                ),
                rejected_steps=(status != int(NonlinearUpdateStatus.APPLIED)).astype(
                    jnp.int32
                ),
                domain_failures=(finite & ~valid).astype(jnp.int32),
                nonfinite_trials=(~finite).astype(jnp.int32),
            )
            return (
                NonlinearUpdateResult(
                    state=candidate,
                    residual=residual,
                    auxiliary=auxiliary,
                    status=status,
                    diagnostics=diagnostics,
                    provenance=_provenance(prepared),
                ),
                prepared.internal_state,
            )

        return jax.lax.cond(
            control.permits(self.maximum_work),
            execute,
            skipped,
            operand=None,
        )


class NewtonStepUpdate(AbstractNonlinearUpdate):
    """One globalized Newton step exposed independently of root convergence."""

    method: NewtonKrylov | NewtonTrustRegion
    termination: NonlinearTermination
    require_decrease: bool = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractNonlinearMethod | None = None,
        /,
        *,
        termination: NonlinearTermination | None = None,
        require_decrease: bool = True,
    ):
        method_ = NewtonKrylov() if method is None else method
        termination_ = NonlinearTermination() if termination is None else termination
        if not isinstance(method_, (NewtonKrylov, NewtonTrustRegion)):
            raise TypeError(
                "NewtonStepUpdate requires NewtonKrylov or NewtonTrustRegion."
            )
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        self.method = method_
        self.termination = _single_step_termination(termination_)
        self.require_decrease = bool(require_decrease)

    @property
    def update_id(self) -> str:
        return f"newton-step/{self.method.method_id}"

    @property
    def capabilities(self) -> NonlinearUpdateCapabilities:
        return NonlinearUpdateCapabilities(
            jit=True,
            prepared_refresh=True,
            differentiable_action=False,
            exposes_linearization=True,
        )

    @property
    def maximum_work(self) -> NonlinearWork:
        globalization_evaluations = (
            self.method.line_search.maximum_steps + 2
            if isinstance(self.method, NewtonKrylov)
            else self.method.trust_region.maximum_attempts + 2
        )
        residual_evaluations = (
            globalization_evaluations
            if self.termination.maximum_evaluations is None
            else min(
                self.termination.maximum_evaluations,
                globalization_evaluations,
            )
        )
        linear_iterations = (
            0
            if self.termination.maximum_linear_iterations is None
            else self.termination.maximum_linear_iterations
        )
        return NonlinearWork(
            residual_evaluations=residual_evaluations,
            validity_evaluations=max(residual_evaluations - 1, 0),
            jvp_evaluations=linear_iterations,
            vjp_evaluations=(
                linear_iterations if isinstance(self.method, NewtonTrustRegion) else 0
            ),
            jacobian_preparations=1,
            linear_setups=1,
            linear_refreshes=1,
            linear_solves=1,
            linear_iterations=linear_iterations,
            complete=self.termination.maximum_linear_iterations is not None,
        )

    def _prepare_internal(
        self,
        problem: NonlinearSystemProblem,
        state: PyTree[Any],
        args: Any,
        /,
    ) -> PreparedNonlinearSolve:
        return prepare_nonlinear(
            problem,
            state,
            method=self.method,
            termination=self.termination,
            args=args,
        )

    def _refresh_internal(
        self,
        internal_state: Any,
        problem: NonlinearSystemProblem,
        state: PyTree[Any],
        args: Any,
        /,
    ) -> PreparedNonlinearSolve:
        if not isinstance(internal_state, PreparedNonlinearSolve):
            raise TypeError("Prepared Newton update state is invalid.")
        return refresh_nonlinear(internal_state, problem, state, args=args)

    def _apply(
        self,
        prepared: PreparedNonlinearUpdate,
        state: PyTree[Any],
        args: Any,
        control: NonlinearUpdateControl,
        /,
    ) -> tuple[NonlinearUpdateResult, Any]:
        state_ = prepared.plan.state_space.validate(state)
        internal_dynamic, internal_static = eqx.partition(
            prepared.internal_state,
            eqx.is_array,
        )

        def skipped(_):
            diagnostics = NonlinearUpdateDiagnostics(
                initial_residual_norm=jnp.asarray(jnp.nan),
                final_residual_norm=jnp.asarray(jnp.nan),
                step_norm=0.0,
                work=NonlinearWork.zero(complete=self.maximum_work.complete),
            )
            return (
                NonlinearUpdateResult(
                    state=state_,
                    residual=prepared.plan.residual_space.zeros(),
                    auxiliary=prepared.reference_auxiliary,
                    status=NonlinearUpdateStatus.BUDGET_EXHAUSTED,
                    diagnostics=diagnostics,
                    provenance=_provenance(prepared),
                ),
                internal_dynamic,
            )

        def execute(_):
            combined = eqx.combine(internal_dynamic, internal_static)
            internal = self._refresh_internal(
                combined,
                prepared.problem,
                state_,
                args,
            )
            initial_norm = internal.run.residual_norm
            result, next_internal = step_prepared_nonlinear(
                internal,
                termination=self.termination,
            )
            accepted = result.diagnostics.accepted_steps > 0
            decreased = result.diagnostics.final_residual_norm < initial_norm
            finite = tree_allfinite(result.state) & tree_allfinite(result.residual)
            valid = prepared.problem.valid(
                result.state,
                result.residual,
                result.auxiliary,
                args,
            )
            usable = accepted & finite & valid & (decreased | (not self.require_decrease))
            status = jnp.where(
                usable,
                int(NonlinearUpdateStatus.APPLIED),
                _update_status_from_inner(result.status),
            ).astype(jnp.int32)
            diagnostics = NonlinearUpdateDiagnostics.from_nonlinear(result.diagnostics)
            return (
                NonlinearUpdateResult(
                    state=result.state,
                    residual=result.residual,
                    auxiliary=result.auxiliary,
                    status=status,
                    inner_status=result.status,
                    diagnostics=diagnostics,
                    provenance=_provenance(prepared),
                ),
                eqx.partition(next_internal, eqx.is_array)[0],
            )

        result, next_dynamic = jax.lax.cond(
            control.permits(self.maximum_work),
            execute,
            skipped,
            operand=None,
        )
        return result, eqx.combine(next_dynamic, internal_static)


def _single_step_termination(value: NonlinearTermination, /) -> NonlinearTermination:
    return NonlinearTermination(
        absolute_residual=value.absolute_residual,
        relative_residual=value.relative_residual,
        absolute_step=value.absolute_step,
        relative_step=value.relative_step,
        maximum_steps=1,
        maximum_evaluations=value.maximum_evaluations,
        maximum_linear_iterations=value.maximum_linear_iterations,
        divergence_factor=value.divergence_factor,
    )


def _update_status_from_inner(status: Any, /) -> Array:
    value = jnp.asarray(status, dtype=jnp.int32)
    return jnp.where(
        value == int(NonlinearStatus.NONFINITE_INPUT),
        int(NonlinearUpdateStatus.NONFINITE_INPUT),
        jnp.where(
            value == int(NonlinearStatus.NONFINITE_EVALUATION),
            int(NonlinearUpdateStatus.NONFINITE_EVALUATION),
            jnp.where(
                (value == int(NonlinearStatus.RECOVERABLE_DOMAIN_FAILURE))
                | (value == int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE)),
                int(NonlinearUpdateStatus.DOMAIN_REJECTED),
                jnp.where(
                    (value == int(NonlinearStatus.LINEAR_SOLVE_FAILED))
                    | (value == int(NonlinearStatus.SINGULAR_JACOBIAN))
                    | (value == int(NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED)),
                    int(NonlinearUpdateStatus.LINEAR_FAILURE),
                    int(NonlinearUpdateStatus.NO_PROGRESS),
                ),
            ),
        ),
    ).astype(jnp.int32)


def _provenance(
    prepared: PreparedNonlinearUpdate, /, *, notes: str = ""
) -> NonlinearUpdateProvenance:
    return NonlinearUpdateProvenance(
        problem_id=prepared.problem.problem_id,
        update_id=prepared.update.update_id,
        plan_id=prepared.plan.plan_id,
        notes=notes,
    )


def skipped_nonlinear_update_result(
    prepared: PreparedNonlinearUpdate,
    state: PyTree[Any],
    /,
    *,
    status: NonlinearUpdateStatus = NonlinearUpdateStatus.INNER_FAILURE,
    failure_origin: str = "skipped-after-component-failure",
) -> NonlinearUpdateResult:
    if not isinstance(prepared, PreparedNonlinearUpdate):
        raise TypeError("prepared must be PreparedNonlinearUpdate.")
    state_ = prepared.plan.state_space.validate(state)
    children = ()
    if isinstance(prepared.internal_state, tuple) and all(
        isinstance(child, PreparedNonlinearUpdate) for child in prepared.internal_state
    ):
        children = tuple(
            skipped_nonlinear_update_result(
                child,
                child.reference_state,
                status=status,
                failure_origin=failure_origin,
            )
            for child in prepared.internal_state
        )
    work = NonlinearWork.zero(complete=prepared.update.capabilities.counts_complete)
    diagnostics = NonlinearUpdateDiagnostics(
        initial_residual_norm=jnp.asarray(jnp.nan),
        final_residual_norm=jnp.asarray(jnp.nan),
        step_norm=0.0,
        work=work,
    )
    provenance = NonlinearUpdateProvenance(
        problem_id=prepared.problem.problem_id,
        update_id=prepared.update.update_id,
        plan_id=prepared.plan.plan_id,
        notes="",
    )
    evidence = NonlinearAttemptEvidence(
        component_id=prepared.update.update_id,
        status=status,
        accepted=False,
        skipped=True,
        input_residual_norm=jnp.asarray(jnp.nan),
        output_residual_norm=jnp.asarray(jnp.nan),
        work=work,
        failure_origin="update-status",
        children=tuple(child.evidence for child in children),
    )
    return NonlinearUpdateResult(
        state=state_,
        residual=prepared.plan.residual_space.zeros(),
        auxiliary=prepared.reference_auxiliary,
        status=status,
        diagnostics=diagnostics,
        provenance=provenance,
        components=children,
        evidence=evidence,
    )


def plan_nonlinear_update(
    problem: NonlinearSystemProblem,
    initial_state: PyTree[Any],
    update: AbstractNonlinearUpdate,
    /,
    *,
    args: Any = None,
) -> tuple[NonlinearSystemProblem, PyTree[Array], NonlinearUpdatePlan]:
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError("problem must be NonlinearSystemProblem.")
    if not isinstance(update, AbstractNonlinearUpdate):
        raise TypeError("update must be AbstractNonlinearUpdate.")
    state = problem.validate_state(initial_state)
    residual, _ = problem.evaluate(state, args)
    problem_ = problem.bind_spaces(state, residual)
    if problem_.state_space is None or problem_.residual_space is None:
        raise ValueError("A nonlinear update plan requires bound vector spaces.")
    return (
        problem_,
        state,
        NonlinearUpdatePlan(
            problem_.state_space,
            problem_.residual_space,
            problem_id=problem_.problem_id,
            update_id=update.update_id,
        ),
    )


def prepare_nonlinear_update(
    problem: NonlinearSystemProblem,
    initial_state: PyTree[Any],
    update: AbstractNonlinearUpdate,
    /,
    *,
    args: Any = None,
) -> PreparedNonlinearUpdate:
    problem_, state, plan = plan_nonlinear_update(
        problem,
        initial_state,
        update,
        args=args,
    )
    internal = update._prepare_internal(problem_, state, args)
    residual, auxiliary = problem_.evaluate(state, args)
    return PreparedNonlinearUpdate(
        problem_,
        update,
        plan,
        internal,
        state,
        residual,
        auxiliary,
        numeric_version=0,
    )


def refresh_nonlinear_update(
    prepared: PreparedNonlinearUpdate,
    problem: NonlinearSystemProblem,
    state: PyTree[Any],
    /,
    *,
    args: Any = None,
) -> PreparedNonlinearUpdate:
    if not isinstance(prepared, PreparedNonlinearUpdate):
        raise TypeError("prepared must be PreparedNonlinearUpdate.")
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError("problem must be NonlinearSystemProblem.")
    if problem.problem_id != prepared.problem.problem_id:
        raise ValueError("Nonlinear update refresh must preserve problem_id.")
    state_ = problem.validate_state(state)
    residual, auxiliary = problem.evaluate(state_, args)
    problem_ = problem.bind_spaces(state_, residual)
    if problem_.state_space is None or problem_.residual_space is None:
        raise ValueError("Refreshed nonlinear update requires bound spaces.")
    if not prepared.plan.state_space.compatible(problem_.state_space):
        raise ValueError("Nonlinear update refresh changed the state space.")
    if not prepared.plan.residual_space.compatible(problem_.residual_space):
        raise ValueError("Nonlinear update refresh changed the residual space.")
    internal = prepared.update._refresh_internal(
        prepared.internal_state,
        problem_,
        state_,
        args,
    )
    return PreparedNonlinearUpdate(
        problem_,
        prepared.update,
        prepared.plan,
        internal,
        state_,
        residual,
        auxiliary,
        numeric_version=prepared.numeric_version + 1,
    )


def apply_prepared_nonlinear_update(
    prepared: PreparedNonlinearUpdate,
    state: PyTree[Any],
    /,
    *,
    args: Any = None,
    control: NonlinearUpdateControl | None = None,
) -> tuple[NonlinearUpdateResult, PreparedNonlinearUpdate]:
    if not isinstance(prepared, PreparedNonlinearUpdate):
        raise TypeError("prepared must be PreparedNonlinearUpdate.")
    control_ = NonlinearUpdateControl() if control is None else control
    if not isinstance(control_, NonlinearUpdateControl):
        raise TypeError("control must be NonlinearUpdateControl or None.")
    state_ = prepared.plan.state_space.validate(state)
    result, internal = prepared.update._apply(prepared, state_, args, control_)
    next_prepared = PreparedNonlinearUpdate(
        prepared.problem,
        prepared.update,
        prepared.plan,
        internal,
        result.state,
        result.residual,
        result.auxiliary,
        numeric_version=prepared.numeric_version,
    )
    return result, next_prepared


__all__ = [
    "AbstractNonlinearUpdate",
    "FunctionNonlinearUpdate",
    "NewtonStepUpdate",
    "NonlinearUpdateCapabilities",
    "NonlinearUpdateControl",
    "NonlinearUpdateDiagnostics",
    "NonlinearUpdatePlan",
    "NonlinearUpdateProvenance",
    "NonlinearUpdateResult",
    "NonlinearUpdateStatus",
    "PreparedNonlinearUpdate",
    "apply_prepared_nonlinear_update",
    "nonlinear_update_status_message",
    "plan_nonlinear_update",
    "prepare_nonlinear_update",
    "refresh_nonlinear_update",
]
