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
    solve_prepared_nonlinear,
)
from ._types import (
    AbstractNonlinearMethod,
    NonlinearDiagnostics,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
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
    """Remaining aggregate work available to one indivisible update."""

    maximum_residual_evaluations: int | None = eqx.field(static=True)
    maximum_jacobian_preparations: int | None = eqx.field(static=True)
    maximum_linear_solves: int | None = eqx.field(static=True)
    maximum_linear_iterations: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_residual_evaluations: int | None = None,
        maximum_jacobian_preparations: int | None = None,
        maximum_linear_solves: int | None = None,
        maximum_linear_iterations: int | None = None,
    ):
        values = (
            maximum_residual_evaluations,
            maximum_jacobian_preparations,
            maximum_linear_solves,
            maximum_linear_iterations,
        )
        normalized = tuple(None if value is None else int(value) for value in values)
        if any(value is not None and value < 0 for value in normalized):
            raise ValueError(
                "Nonlinear update work allowances must be non-negative or None."
            )
        (
            self.maximum_residual_evaluations,
            self.maximum_jacobian_preparations,
            self.maximum_linear_solves,
            self.maximum_linear_iterations,
        ) = normalized

    def permits(
        self,
        *,
        residual_evaluations: int,
        jacobian_preparations: int,
        linear_solves: int,
        linear_iterations: int,
    ) -> bool:
        requested = (
            int(residual_evaluations),
            int(jacobian_preparations),
            int(linear_solves),
            int(linear_iterations),
        )
        available = (
            self.maximum_residual_evaluations,
            self.maximum_jacobian_preparations,
            self.maximum_linear_solves,
            self.maximum_linear_iterations,
        )
        return all(
            limit is None or need <= limit
            for need, limit in zip(requested, available, strict=True)
        )


class NonlinearUpdateDiagnostics(StrictModule):
    """Physical progress and exact work from one update application."""

    initial_residual_norm: Array
    final_residual_norm: Array
    step_norm: Array
    residual_evaluations: Array
    jacobian_preparations: Array
    linear_solves: Array
    linear_iterations: Array
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
        residual_evaluations: Any = 0,
        jacobian_preparations: Any = 0,
        linear_solves: Any = 0,
        linear_iterations: Any = 0,
        accepted_steps: Any = 0,
        rejected_steps: Any = 0,
        domain_failures: Any = 0,
        nonfinite_trials: Any = 0,
        counts_complete: bool = True,
    ):
        self.initial_residual_norm = jnp.asarray(initial_residual_norm)
        self.final_residual_norm = jnp.asarray(final_residual_norm)
        self.step_norm = jnp.asarray(step_norm)
        values = tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (
                residual_evaluations,
                jacobian_preparations,
                linear_solves,
                linear_iterations,
                accepted_steps,
                rejected_steps,
                domain_failures,
                nonfinite_trials,
            )
        )
        (
            self.residual_evaluations,
            self.jacobian_preparations,
            self.linear_solves,
            self.linear_iterations,
            self.accepted_steps,
            self.rejected_steps,
            self.domain_failures,
            self.nonfinite_trials,
        ) = values
        self.counts_complete = bool(counts_complete)

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
            jacobian_preparations=diagnostics.jacobian_preparations,
            linear_solves=diagnostics.linear_solves,
            linear_iterations=diagnostics.linear_iterations,
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
    numeric_version: Array

    def __init__(
        self,
        problem: NonlinearSystemProblem,
        update: AbstractNonlinearUpdate,
        plan: NonlinearUpdatePlan,
        internal_state: Any,
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
        initial_residual, initial_auxiliary = problem.evaluate(state_, args)
        initial_norm = _space_norm(prepared.plan.residual_space, initial_residual)
        if not control.permits(
            residual_evaluations=2,
            jacobian_preparations=0,
            linear_solves=0,
            linear_iterations=0,
        ):
            diagnostics = NonlinearUpdateDiagnostics(
                initial_residual_norm=initial_norm,
                final_residual_norm=initial_norm,
                step_norm=0.0,
                residual_evaluations=1,
            )
            return (
                NonlinearUpdateResult(
                    state=state_,
                    residual=initial_residual,
                    auxiliary=initial_auxiliary,
                    status=NonlinearUpdateStatus.BUDGET_EXHAUSTED,
                    diagnostics=diagnostics,
                    provenance=_provenance(prepared),
                ),
                prepared.internal_state,
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
                jax.tree.map(lambda new, old: new - old, candidate, state_),
            ),
            residual_evaluations=2,
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
        internal = self._refresh_internal(
            prepared.internal_state,
            prepared.problem,
            state,
            args,
        )
        initial_norm = internal.run.residual_norm
        maximum_evaluations = self.termination.maximum_evaluations
        maximum_linear = self.termination.maximum_linear_iterations
        if not control.permits(
            residual_evaluations=(
                2 if maximum_evaluations is None else maximum_evaluations
            ),
            jacobian_preparations=1,
            linear_solves=1,
            linear_iterations=(0 if maximum_linear is None else maximum_linear),
        ):
            diagnostics = NonlinearUpdateDiagnostics(
                initial_residual_norm=initial_norm,
                final_residual_norm=initial_norm,
                step_norm=0.0,
                residual_evaluations=internal.run.residual_evaluations,
                jacobian_preparations=internal.run.jacobian_preparations,
            )
            return (
                NonlinearUpdateResult(
                    state=internal.state,
                    residual=internal.run.residual,
                    auxiliary=internal.run.auxiliary,
                    status=NonlinearUpdateStatus.BUDGET_EXHAUSTED,
                    diagnostics=diagnostics,
                    provenance=_provenance(prepared),
                ),
                internal,
            )
        result = solve_prepared_nonlinear(internal, termination=self.termination)
        accepted = result.diagnostics.accepted_steps > 0
        decreased = result.diagnostics.final_residual_norm < initial_norm
        finite = tree_allfinite(result.state) & tree_allfinite(result.residual)
        valid = prepared.problem.valid(
            result.state, result.residual, result.auxiliary, args
        )
        usable = accepted & finite & valid & (decreased | ~self.require_decrease)
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
            internal,
        )


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
    return PreparedNonlinearUpdate(
        problem_,
        update,
        plan,
        internal,
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
    residual, _ = problem.evaluate(state_, args)
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
