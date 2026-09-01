#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import AbstractAttribute, StrictModule
from ..linalg import (
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    PyTreeSpace,
    solve as solve_linear,
    TolerancePolicy,
    transpose,
)
from ._bounds import _projected_displacement
from ._iterative._base import AbstractLeastSquaresMethod
from ._iterative._globalization import ArmijoLineSearch
from ._iterative._types import (
    _tree_add_scaled,
    _tree_allfinite,
    _tree_inner,
    _tree_negative,
    _tree_norm,
    _validate_real_inexact_tree,
    Bounds,
    ConstrainedOptimalityCertificate,
    NonlinearConstraint,
    NonlinearLeastSquaresProblem,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._least_squares import LevenbergMarquardt


class StateAcceptanceEvidence(StrictModule):
    """Numerical evidence for accepting one realized state equation."""

    residual_norm: Array
    reference_norm: Array
    threshold: Array
    normalized_residual: Array
    finite: Array
    admissible: Array
    realization_matches: Array
    status_accepted: Array
    accepted: Array

    def __init__(
        self,
        residual_norm: Any,
        reference_norm: Any,
        threshold: Any,
        finite: Any,
        admissible: Any,
        realization_matches: Any,
        status_accepted: Any,
        /,
    ):
        scalars = tuple(
            jnp.asarray(value)
            for value in (
                residual_norm,
                reference_norm,
                threshold,
                finite,
                admissible,
                realization_matches,
                status_accepted,
            )
        )
        if any(value.shape != () for value in scalars):
            raise ValueError("State acceptance evidence values must be scalar.")
        self.residual_norm = scalars[0]
        self.reference_norm = scalars[1]
        self.threshold = scalars[2]
        tiny = jnp.asarray(
            jnp.finfo(self.residual_norm.dtype).tiny,
            dtype=self.residual_norm.dtype,
        )
        self.normalized_residual = self.residual_norm / jnp.maximum(
            self.threshold,
            tiny,
        )
        self.finite = jnp.asarray(scalars[3], dtype=bool)
        self.admissible = jnp.asarray(scalars[4], dtype=bool)
        self.realization_matches = jnp.asarray(scalars[5], dtype=bool)
        self.status_accepted = jnp.asarray(scalars[6], dtype=bool)
        self.accepted = (
            self.status_accepted
            & self.finite
            & self.admissible
            & self.realization_matches
            & (self.residual_norm <= self.threshold)
        )


class AdjointAcceptanceEvidence(StrictModule):
    """Numerical evidence for accepting one realized transpose solve."""

    transpose_defect_norm: Array
    right_hand_side_norm: Array
    threshold: Array
    normalized_transpose_defect: Array
    finite: Array
    admissible: Array
    realization_matches: Array
    status_accepted: Array
    accepted: Array

    def __init__(
        self,
        transpose_defect_norm: Any,
        right_hand_side_norm: Any,
        threshold: Any,
        finite: Any,
        admissible: Any,
        realization_matches: Any,
        status_accepted: Any,
        /,
    ):
        scalars = tuple(
            jnp.asarray(value)
            for value in (
                transpose_defect_norm,
                right_hand_side_norm,
                threshold,
                finite,
                admissible,
                realization_matches,
                status_accepted,
            )
        )
        if any(value.shape != () for value in scalars):
            raise ValueError("Adjoint acceptance evidence values must be scalar.")
        self.transpose_defect_norm = scalars[0]
        self.right_hand_side_norm = scalars[1]
        self.threshold = scalars[2]
        tiny = jnp.asarray(
            jnp.finfo(self.transpose_defect_norm.dtype).tiny,
            dtype=self.transpose_defect_norm.dtype,
        )
        self.normalized_transpose_defect = self.transpose_defect_norm / jnp.maximum(
            self.threshold, tiny
        )
        self.finite = jnp.asarray(scalars[3], dtype=bool)
        self.admissible = jnp.asarray(scalars[4], dtype=bool)
        self.realization_matches = jnp.asarray(scalars[5], dtype=bool)
        self.status_accepted = jnp.asarray(scalars[6], dtype=bool)
        self.accepted = (
            self.status_accepted
            & self.finite
            & self.admissible
            & self.realization_matches
            & (self.transpose_defect_norm <= self.threshold)
        )


class StateAcceptancePolicy(StrictModule):
    """Explicit residual, status, admissibility, and realization acceptance."""

    state_relative_tolerance: float = eqx.field(static=True)
    state_absolute_tolerance: float = eqx.field(static=True)
    adjoint_relative_tolerance: float = eqx.field(static=True)
    adjoint_absolute_tolerance: float = eqx.field(static=True)
    accepted_state_statuses: tuple[int, ...] = eqx.field(static=True)
    accepted_adjoint_statuses: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_relative_tolerance: float = 1e-8,
        state_absolute_tolerance: float = 1e-10,
        adjoint_relative_tolerance: float = 1e-7,
        adjoint_absolute_tolerance: float = 1e-10,
        accepted_state_statuses: Sequence[Any] = (OptimizationStatus.SUCCESS,),
        accepted_adjoint_statuses: Sequence[Any] = (LinearSolveStatus.SUCCESS,),
    ):
        state_relative = float(state_relative_tolerance)
        state_absolute = float(state_absolute_tolerance)
        adjoint_relative = float(adjoint_relative_tolerance)
        adjoint_absolute = float(adjoint_absolute_tolerance)
        tolerances = (
            state_relative,
            state_absolute,
            adjoint_relative,
            adjoint_absolute,
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Acceptance tolerances must be finite and non-negative.")
        state_statuses = tuple(int(value) for value in accepted_state_statuses)
        adjoint_statuses = tuple(int(value) for value in accepted_adjoint_statuses)
        if not state_statuses or len(set(state_statuses)) != len(state_statuses):
            raise ValueError("accepted_state_statuses must be non-empty and unique.")
        if not adjoint_statuses or len(set(adjoint_statuses)) != len(adjoint_statuses):
            raise ValueError("accepted_adjoint_statuses must be non-empty and unique.")
        self.state_relative_tolerance = state_relative
        self.state_absolute_tolerance = state_absolute
        self.adjoint_relative_tolerance = adjoint_relative
        self.adjoint_absolute_tolerance = adjoint_absolute
        self.accepted_state_statuses = state_statuses
        self.accepted_adjoint_statuses = adjoint_statuses

    def state_evidence(
        self,
        state: PyTree[Any],
        residual: PyTree[Any],
        status: Any,
        /,
        *,
        reference_norm: Any,
        admissible: Any,
        realization_matches: Any,
    ) -> StateAcceptanceEvidence:
        residual_norm = _tree_norm(residual)
        reference = jnp.asarray(reference_norm, dtype=residual_norm.dtype)
        if reference.shape != ():
            raise TypeError("reference_norm must be one scalar.")
        threshold = (
            jnp.asarray(self.state_absolute_tolerance, dtype=residual_norm.dtype)
            + jnp.asarray(self.state_relative_tolerance, dtype=residual_norm.dtype)
            * reference
        )
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if status_.shape != ():
            raise ValueError("State solver status must be scalar.")
        status_accepted = jnp.any(
            status_ == jnp.asarray(self.accepted_state_statuses, dtype=jnp.int32)
        )
        finite = (
            _tree_allfinite(state)
            & _tree_allfinite(residual)
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(reference)
            & (reference >= 0.0)
        )
        admissible_ = jnp.all(jnp.asarray(admissible, dtype=bool))
        realization_ = jnp.all(jnp.asarray(realization_matches, dtype=bool))
        return StateAcceptanceEvidence(
            residual_norm,
            reference,
            threshold,
            finite,
            admissible_,
            realization_,
            status_accepted,
        )

    def adjoint_evidence(
        self,
        adjoint: PyTree[Any],
        transpose_image: PyTree[Any],
        right_hand_side: PyTree[Any],
        status: Any,
        /,
        *,
        admissible: Any,
        realization_matches: Any,
    ) -> AdjointAcceptanceEvidence:
        defect = jax.tree.map(
            lambda image, right: image - right,
            transpose_image,
            right_hand_side,
        )
        defect_norm = _tree_norm(defect)
        right_norm = _tree_norm(right_hand_side)
        threshold = (
            jnp.asarray(self.adjoint_absolute_tolerance, dtype=defect_norm.dtype)
            + jnp.asarray(self.adjoint_relative_tolerance, dtype=defect_norm.dtype)
            * right_norm
        )
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if status_.shape != ():
            raise ValueError("Adjoint solver status must be scalar.")
        status_accepted = jnp.any(
            status_ == jnp.asarray(self.accepted_adjoint_statuses, dtype=jnp.int32)
        )
        finite = (
            _tree_allfinite(adjoint)
            & _tree_allfinite(transpose_image)
            & _tree_allfinite(right_hand_side)
            & jnp.isfinite(defect_norm)
            & jnp.isfinite(right_norm)
        )
        admissible_ = jnp.all(jnp.asarray(admissible, dtype=bool))
        realization_ = jnp.all(jnp.asarray(realization_matches, dtype=bool))
        return AdjointAcceptanceEvidence(
            defect_norm,
            right_norm,
            threshold,
            finite,
            admissible_,
            realization_,
            status_accepted,
        )


class StateEquationResult(StrictModule):
    """State-equation solution with explicit numerical acceptance evidence."""

    state: PyTree[Array]
    residual: PyTree[Array]
    residual_norm: Array
    status: Array
    diagnostics: OptimizationDiagnostics
    acceptance: StateAcceptanceEvidence

    def __init__(
        self,
        state: PyTree[Any],
        residual: PyTree[Any],
        status: Any,
        diagnostics: OptimizationDiagnostics,
        acceptance: StateAcceptanceEvidence,
        /,
    ):
        self.state = _validate_real_inexact_tree(state, name="state")
        self.residual = _validate_real_inexact_tree(residual, name="state residual")
        self.residual_norm = _tree_norm(self.residual)
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if status_.shape != ():
            raise ValueError("State equation status must be scalar.")
        self.status = status_
        if not isinstance(diagnostics, OptimizationDiagnostics):
            raise TypeError("diagnostics must be OptimizationDiagnostics.")
        if not isinstance(acceptance, StateAcceptanceEvidence):
            raise TypeError("acceptance must be StateAcceptanceEvidence.")
        self.diagnostics = diagnostics
        self.acceptance = acceptance

    @property
    def successful(self) -> Array:
        return self.acceptance.accepted


class AbstractStateSolver(StrictModule):
    """Solver for one frozen state equation at a declared design."""

    method_id: AbstractAttribute[str]

    @abc.abstractmethod
    def solve(
        self,
        problem: "StateDesignProblem",
        design: PyTree[Any],
        initial_state: PyTree[Any],
        /,
        *,
        args: Any,
    ) -> StateEquationResult:
        raise NotImplementedError


class LeastSquaresStateSolver(AbstractStateSolver):
    """State solver backed by a native nonlinear least-squares method."""

    method: AbstractLeastSquaresMethod
    termination: OptimizationTermination

    def __init__(
        self,
        *,
        method: AbstractLeastSquaresMethod | None = None,
        termination: OptimizationTermination | None = None,
    ):
        method_ = LevenbergMarquardt() if method is None else method
        termination_ = (
            OptimizationTermination(
                absolute_optimality=1e-14,
                relative_optimality=0.0,
                maximum_steps=100,
            )
            if termination is None
            else termination
        )
        if not isinstance(method_, AbstractLeastSquaresMethod):
            raise TypeError("method must be an AbstractLeastSquaresMethod or None.")
        if not isinstance(termination_, OptimizationTermination):
            raise TypeError("termination must be an OptimizationTermination or None.")
        self.method = method_
        self.termination = termination_

    @property
    def method_id(self) -> str:
        return f"least-squares-state/{self.method.method_id}"

    def solve(
        self,
        problem: "StateDesignProblem",
        design: PyTree[Any],
        initial_state: PyTree[Any],
        /,
        *,
        args: Any,
    ) -> StateEquationResult:
        reference_residual = problem.residual(initial_state, design, args)
        residual_problem = NonlinearLeastSquaresProblem(
            lambda state, dynamic_args: problem.residual(state, design, dynamic_args),
            problem_id=f"{problem.problem_id}/state-equation",
        )
        result = self.method.solve(
            residual_problem,
            initial_state,
            termination=self.termination,
            args=args,
        )
        diagnostics = eqx.tree_at(
            lambda item: item.residual_evaluations,
            result.diagnostics,
            result.diagnostics.residual_evaluations + 1,
        )
        acceptance = problem.state_evidence(
            result.parameters,
            design,
            result.residual,
            result.status,
            reference_norm=_tree_norm(reference_residual),
            args=args,
        )
        return StateEquationResult(
            result.parameters,
            result.residual,
            result.status,
            diagnostics,
            acceptance,
        )


class StateDesignConstraint(StrictModule):
    """Bound-form constraint evaluated on one accepted state/design pair."""

    function: Callable
    lower: Any
    upper: Any
    constraint_id: str = eqx.field(static=True)
    depends_on_state: bool = eqx.field(static=True)

    def __init__(
        self,
        function: Callable,
        /,
        *,
        lower: Any = -jnp.inf,
        upper: Any = jnp.inf,
        constraint_id: str,
        depends_on_state: bool = True,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(constraint_id)
        if not identifier:
            raise ValueError("constraint_id must be non-empty.")
        self.function = function
        self.lower = lower
        self.upper = upper
        self.constraint_id = identifier
        self.depends_on_state = bool(depends_on_state)

    def value(
        self,
        state: PyTree[Any],
        design: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        return _validate_real_inexact_tree(
            self.function(state, design, args),
            name="state-design constraint value",
        )

    def bounds(
        self,
        value: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        adapter = NonlinearConstraint(
            lambda _, __: value,
            lower=self.lower,
            upper=self.upper,
            constraint_id=self.constraint_id,
        )
        return adapter.bounds(value)


class StateDesignProblem(StrictModule):
    """PDE/state-constrained objective with explicit state and design roles."""

    state_residual: Any
    objective: Any
    state_solver: AbstractStateSolver
    acceptance_policy: StateAcceptancePolicy
    state_admissibility: Callable | None = eqx.field(static=True)
    state_realization: Callable | None = eqx.field(static=True)
    design_bounds: Bounds | None
    constraints: tuple[StateDesignConstraint, ...]
    has_aux: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_residual,
        objective,
        /,
        *,
        state_solver: AbstractStateSolver | None = None,
        acceptance_policy: StateAcceptancePolicy | None = None,
        state_admissibility: Callable | None = None,
        state_realization: Callable | None = None,
        design_bounds: Bounds | None = None,
        constraints: Sequence[StateDesignConstraint] = (),
        has_aux: bool = False,
        problem_id: str = "state-design",
    ):
        if not callable(state_residual) or not callable(objective):
            raise TypeError("state_residual and objective must be callable.")
        solver = LeastSquaresStateSolver() if state_solver is None else state_solver
        acceptance = (
            StateAcceptancePolicy() if acceptance_policy is None else acceptance_policy
        )
        if not isinstance(solver, AbstractStateSolver):
            raise TypeError("state_solver must be an AbstractStateSolver or None.")
        if not isinstance(acceptance, StateAcceptancePolicy):
            raise TypeError("acceptance_policy must be a StateAcceptancePolicy or None.")
        if state_admissibility is not None and not callable(state_admissibility):
            raise TypeError("state_admissibility must be callable or None.")
        if state_realization is not None and not callable(state_realization):
            raise TypeError("state_realization must be callable or None.")
        if design_bounds is not None and not isinstance(design_bounds, Bounds):
            raise TypeError("design_bounds must be a Bounds or None.")
        constraints_ = tuple(constraints)
        if any(
            not isinstance(constraint, StateDesignConstraint)
            for constraint in constraints_
        ):
            raise TypeError("constraints must contain StateDesignConstraint values.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.state_residual = state_residual
        self.objective = objective
        self.state_solver = solver
        self.acceptance_policy = acceptance
        self.state_admissibility = state_admissibility
        self.state_realization = state_realization
        self.design_bounds = design_bounds
        self.constraints = constraints_
        self.has_aux = bool(has_aux)
        self.problem_id = identifier

    def residual(
        self,
        state: PyTree[Any],
        design: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        return _validate_real_inexact_tree(
            self.state_residual(state, design, args),
            name="state residual",
        )

    def value(
        self,
        state: PyTree[Any],
        design: PyTree[Any],
        args: Any = None,
        /,
    ) -> tuple[Array, Any]:
        output = self.objective(state, design, args)
        if self.has_aux:
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError(
                    "A state-design objective with has_aux=True must return "
                    "(value, auxiliary)."
                )
            raw_value, auxiliary = output
        else:
            raw_value, auxiliary = output, None
        value = jnp.asarray(raw_value)
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("A state-design objective must return one real scalar array.")
        return value, auxiliary

    def constraint_values(
        self,
        state: PyTree[Any],
        design: PyTree[Any],
        args: Any = None,
        /,
    ) -> tuple[PyTree[Array], ...]:
        """Evaluate every state/design constraint in stable declaration order."""

        return tuple(
            constraint.value(state, design, args) for constraint in self.constraints
        )

    def state_evidence(
        self,
        state: PyTree[Any],
        design: PyTree[Any],
        residual: PyTree[Any],
        status: Any,
        /,
        *,
        reference_norm: Any,
        args: Any = None,
    ) -> StateAcceptanceEvidence:
        """Evaluate one realized state against the declared acceptance policy."""

        admissible = (
            jnp.asarray(True)
            if self.state_admissibility is None
            else self.state_admissibility(state, design, args)
        )
        realization_matches = (
            jnp.asarray(True)
            if self.state_realization is None
            else self.state_realization(state, design, args)
        )
        return self.acceptance_policy.state_evidence(
            state,
            residual,
            status,
            reference_norm=reference_norm,
            admissible=admissible,
            realization_matches=realization_matches,
        )

    def solve_state(
        self,
        design: PyTree[Any],
        initial_state: PyTree[Any],
        /,
        *,
        args: Any = None,
    ) -> StateEquationResult:
        """Solve and independently certify the realized state residual."""
        result = self.state_solver.solve(
            self,
            design,
            initial_state,
            args=args,
        )
        if not isinstance(result, StateEquationResult):
            raise TypeError("A state solver must return StateEquationResult.")
        reference_residual = self.residual(initial_state, design, args)
        realized_residual = self.residual(result.state, design, args)
        acceptance = self.state_evidence(
            result.state,
            design,
            realized_residual,
            result.status,
            reference_norm=_tree_norm(reference_residual),
            args=args,
        )
        diagnostics = eqx.tree_at(
            lambda item: item.residual_evaluations,
            result.diagnostics,
            result.diagnostics.residual_evaluations + 2,
        )
        return StateEquationResult(
            result.state,
            realized_residual,
            result.status,
            diagnostics,
            acceptance,
        )


class StateDesignResult(StrictModule):
    """Optimized state/design pair with KKT and solve provenance."""

    state: PyTree[Array]
    design: PyTree[Array]
    objective: Array
    auxiliary: Any
    adjoint: PyTree[Array] | None
    status: Array
    diagnostics: OptimizationDiagnostics
    provenance: OptimizationProvenance
    certificate: ConstrainedOptimalityCertificate | None
    method_evidence: Any
    state_acceptance: StateAcceptanceEvidence
    adjoint_acceptance: AdjointAcceptanceEvidence | None

    def __init__(
        self,
        state: PyTree[Any],
        design: PyTree[Any],
        objective: Any,
        auxiliary: Any,
        adjoint: PyTree[Any] | None,
        status: Any,
        diagnostics: OptimizationDiagnostics,
        provenance: OptimizationProvenance,
        /,
        *,
        state_acceptance: StateAcceptanceEvidence,
        adjoint_acceptance: AdjointAcceptanceEvidence | None = None,
        certificate: ConstrainedOptimalityCertificate | None = None,
        method_evidence: Any = None,
    ):
        self.state = _validate_real_inexact_tree(state, name="state")
        self.design = _validate_real_inexact_tree(design, name="design")
        self.objective = jnp.asarray(objective)
        self.auxiliary = auxiliary
        self.adjoint = adjoint
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if status_.shape != ():
            raise ValueError("State-design status must be scalar.")
        if not isinstance(diagnostics, OptimizationDiagnostics):
            raise TypeError("diagnostics must be OptimizationDiagnostics.")
        if not isinstance(provenance, OptimizationProvenance):
            raise TypeError("provenance must be OptimizationProvenance.")
        if certificate is not None and not isinstance(
            certificate, ConstrainedOptimalityCertificate
        ):
            raise TypeError(
                "certificate must be a ConstrainedOptimalityCertificate or None."
            )
        if not isinstance(state_acceptance, StateAcceptanceEvidence):
            raise TypeError("state_acceptance must be StateAcceptanceEvidence.")
        if adjoint_acceptance is not None and not isinstance(
            adjoint_acceptance, AdjointAcceptanceEvidence
        ):
            raise TypeError(
                "adjoint_acceptance must be AdjointAcceptanceEvidence or None."
            )
        acceptance_certified = state_acceptance.accepted & (
            jnp.asarray(True)
            if adjoint_acceptance is None
            else adjoint_acceptance.accepted
        )
        self.status = jnp.where(
            (status_ == int(OptimizationStatus.SUCCESS)) & ~acceptance_certified,
            int(OptimizationStatus.CERTIFICATION_FAILED),
            status_,
        ).astype(jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.certificate = certificate
        self.method_evidence = method_evidence
        self.state_acceptance = state_acceptance
        self.adjoint_acceptance = adjoint_acceptance

    @property
    def successful(self) -> Array:
        return self.status == int(OptimizationStatus.SUCCESS)


class AbstractStateDesignMethod(StrictModule):
    """Complete method for a state/design optimization problem."""

    method_id: AbstractAttribute[str]

    @abc.abstractmethod
    def solve(
        self,
        problem: StateDesignProblem,
        initial_state: PyTree[Any],
        initial_design: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> StateDesignResult:
        raise NotImplementedError


def _default_adjoint_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        GMRES(),
        tolerance=TolerancePolicy(relative=1e-7, absolute=1e-10),
    )


def _adjoint_gradient(
    problem: StateDesignProblem,
    state: PyTree[Any],
    design: PyTree[Any],
    args: Any,
    linear_policy: LinearSolvePolicy,
    state_acceptance: StateAcceptanceEvidence,
    /,
):
    def residual_function(current_state):
        return problem.residual(
            current_state,
            design,
            args,
        )

    residual, state_linearization = jax.linearize(residual_function, state)
    _, state_pullback = jax.vjp(residual_function, state)

    def state_action(tangent):
        return state_linearization(tangent)

    def state_transpose_action(cotangent):
        return state_pullback(cotangent)[0]

    state_jacobian = FunctionLinearOperator(
        state_action,
        source=PyTreeSpace(state),
        target=PyTreeSpace(residual),
        transpose_action=state_transpose_action,
        operator_id="state-jacobian",
        closure_convert=False,
    )
    state_objective_gradient = jax.grad(
        lambda current_state: problem.value(current_state, design, args)[0]
    )(state)
    adjoint_result = solve_linear(
        LinearSystem(transpose(state_jacobian)),
        state_objective_gradient,
        policy=linear_policy,
    )
    adjoint = adjoint_result.value
    transpose_image = state_transpose_action(adjoint)
    adjoint_acceptance = problem.acceptance_policy.adjoint_evidence(
        adjoint,
        transpose_image,
        state_objective_gradient,
        adjoint_result.status,
        admissible=state_acceptance.admissible & state_acceptance.finite,
        realization_matches=state_acceptance.realization_matches,
    )
    design_objective_gradient = jax.grad(
        lambda current_design: problem.value(state, current_design, args)[0]
    )(design)
    _, design_pullback = jax.vjp(
        lambda current_design: problem.residual(state, current_design, args),
        design,
    )
    residual_design_adjoint = design_pullback(adjoint)[0]
    reduced_gradient = jax.tree.map(
        lambda objective_part, residual_part: objective_part - residual_part,
        design_objective_gradient,
        residual_design_adjoint,
    )
    return reduced_gradient, adjoint, adjoint_result, adjoint_acceptance


def _state_design_line_search(
    problem: StateDesignProblem,
    policy: ArmijoLineSearch,
    state_result: StateEquationResult,
    design: PyTree[Any],
    value: Array,
    direction: PyTree[Any],
    directional_derivative: Array,
    args: Any,
    /,
):
    """Backtrack over fully solved states without exposing rejected trials."""

    scalar_dtype = jnp.result_type(value, directional_derivative, float)
    initial_rate = jnp.asarray(policy.initial_rate, dtype=scalar_dtype)
    minimum_rate = jnp.asarray(policy.minimum_rate, dtype=scalar_dtype)
    contraction = jnp.asarray(policy.contraction, dtype=scalar_dtype)
    sufficient_decrease = jnp.asarray(
        policy.sufficient_decrease,
        dtype=scalar_dtype,
    )

    def condition(carry):
        trials, rate, accepted, *_ = carry
        return (
            (trials < policy.maximum_steps)
            & (~accepted)
            & (rate >= minimum_rate)
            & jnp.isfinite(rate)
        )

    def body(carry):
        (
            trials,
            rate,
            _,
            candidate_state_result,
            candidate_design,
            candidate_value,
            objective_evaluations,
            residual_evaluations,
            jvp_evaluations,
            vjp_evaluations,
            hvp_evaluations,
            setup_refreshes,
            numeric_refreshes,
            linear_solves,
            linear_iterations,
            nested_globalization_evaluations,
        ) = carry
        trial_design = _tree_add_scaled(design, direction, rate)
        if problem.design_bounds is not None:
            trial_design = problem.design_bounds.project(trial_design)
        trial_state_result = problem.solve_state(
            trial_design,
            state_result.state,
            args=args,
        )
        trial_value, _ = problem.value(
            trial_state_result.state,
            trial_design,
            args,
        )
        sufficient = trial_value <= (
            value + sufficient_decrease * rate * directional_derivative
        )
        accepted = (
            trial_state_result.acceptance.accepted
            & jnp.isfinite(trial_value)
            & sufficient
        )

        def accept_trial(_):
            return trial_state_result, trial_design, trial_value

        def keep_accepted(_):
            return candidate_state_result, candidate_design, candidate_value

        (
            next_state_result,
            next_design,
            next_value,
        ) = jax.lax.cond(
            accepted,
            accept_trial,
            keep_accepted,
            None,
        )
        next_rate = jnp.where(accepted, rate, rate * contraction)
        return (
            trials + 1,
            next_rate,
            accepted,
            next_state_result,
            next_design,
            next_value,
            objective_evaluations
            + trial_state_result.diagnostics.objective_evaluations
            + 1,
            residual_evaluations + trial_state_result.diagnostics.residual_evaluations,
            jvp_evaluations + trial_state_result.diagnostics.jvp_evaluations,
            vjp_evaluations + trial_state_result.diagnostics.vjp_evaluations,
            hvp_evaluations + trial_state_result.diagnostics.hvp_evaluations,
            setup_refreshes + trial_state_result.diagnostics.setup_refreshes,
            numeric_refreshes + trial_state_result.diagnostics.numeric_refreshes,
            linear_solves + trial_state_result.diagnostics.linear_solves,
            linear_iterations + trial_state_result.diagnostics.linear_iterations,
            nested_globalization_evaluations
            + trial_state_result.diagnostics.globalization_evaluations,
        )

    initial = (
        jnp.asarray(0, dtype=jnp.int32),
        initial_rate,
        jnp.asarray(False),
        state_result,
        design,
        jnp.asarray(value),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    (
        trials,
        rate,
        accepted,
        candidate_state_result,
        candidate_design,
        candidate_value,
        objective_evaluations,
        residual_evaluations,
        jvp_evaluations,
        vjp_evaluations,
        hvp_evaluations,
        setup_refreshes,
        numeric_refreshes,
        linear_solves,
        linear_iterations,
        nested_globalization_evaluations,
    ) = jax.lax.while_loop(condition, body, initial)
    accepted_rate = jnp.where(accepted, rate, jnp.zeros_like(rate))
    return (
        candidate_state_result,
        candidate_design,
        candidate_value,
        accepted,
        accepted_rate,
        trials,
        objective_evaluations,
        residual_evaluations,
        jvp_evaluations,
        vjp_evaluations,
        hvp_evaluations,
        setup_refreshes,
        numeric_refreshes,
        linear_solves,
        linear_iterations,
        nested_globalization_evaluations + trials,
    )


class ReducedAdjoint(AbstractStateDesignMethod):
    """Reduced-space projected gradient using one matrix-free adjoint per iterate."""

    linear_policy: LinearSolvePolicy
    line_search: ArmijoLineSearch

    def __init__(
        self,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        line_search: ArmijoLineSearch | None = None,
    ):
        policy = _default_adjoint_policy() if linear_policy is None else linear_policy
        search = ArmijoLineSearch() if line_search is None else line_search
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if not isinstance(search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch or None.")
        self.linear_policy = policy
        self.line_search = search

    @property
    def method_id(self) -> str:
        return "reduced-adjoint"

    def solve(
        self,
        problem: StateDesignProblem,
        initial_state: PyTree[Any],
        initial_design: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> StateDesignResult:
        return _solve_reduced_adjoint(
            self,
            problem,
            initial_state,
            initial_design,
            termination=termination,
            args=args,
        )


class SimultaneousKKT(AbstractStateDesignMethod):
    """All-at-once state, design, and adjoint KKT residual method."""

    method: AbstractLeastSquaresMethod

    def __init__(self, *, method: AbstractLeastSquaresMethod | None = None):
        method_ = LevenbergMarquardt() if method is None else method
        if not isinstance(method_, AbstractLeastSquaresMethod):
            raise TypeError("method must be an AbstractLeastSquaresMethod or None.")
        self.method = method_

    @property
    def method_id(self) -> str:
        return "simultaneous-kkt"

    def solve(
        self,
        problem: StateDesignProblem,
        initial_state: PyTree[Any],
        initial_design: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> StateDesignResult:
        return _solve_simultaneous_kkt(
            self,
            problem,
            initial_state,
            initial_design,
            termination=termination,
            args=args,
        )


def _solve_reduced_adjoint(
    method: ReducedAdjoint,
    problem: StateDesignProblem,
    initial_state: PyTree[Any],
    initial_design: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> StateDesignResult:
    if not isinstance(problem, StateDesignProblem):
        raise TypeError("problem must be a StateDesignProblem.")
    if problem.constraints:
        raise ValueError(
            "ReducedAdjoint does not support StateDesignProblem constraints; "
            "use ReducedMMA."
        )
    state = _validate_real_inexact_tree(initial_state, name="initial_state")
    design = _validate_real_inexact_tree(initial_design, name="initial_design")
    if problem.design_bounds is not None:
        design = problem.design_bounds.project(design)
    state_result = problem.solve_state(design, state, args=args)
    initial_status = jnp.where(
        state_result.acceptance.accepted,
        int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.BACKEND_FAILED),
    ).astype(jnp.int32)
    initial_adjoint = jax.tree.map(jnp.zeros_like, state_result.residual)
    design_scalar = _tree_norm(design)
    initial_carry = (
        state_result,
        design,
        initial_status,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        state_result.diagnostics.objective_evaluations,
        jnp.asarray(0, dtype=jnp.int32),
        state_result.diagnostics.residual_evaluations,
        state_result.diagnostics.jvp_evaluations,
        state_result.diagnostics.vjp_evaluations,
        state_result.diagnostics.hvp_evaluations,
        state_result.diagnostics.linear_solves,
        state_result.diagnostics.linear_iterations,
        state_result.diagnostics.setup_refreshes,
        state_result.diagnostics.numeric_refreshes,
        state_result.diagnostics.globalization_evaluations,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros_like(design_scalar),
        jnp.zeros_like(design_scalar),
        jnp.full_like(design_scalar, jnp.nan),
        initial_adjoint,
    )

    def condition(carry):
        (
            _,
            _,
            status,
            iterations,
            _,
            objective_evaluations,
            *_,
        ) = carry
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else objective_evaluations < termination.maximum_evaluations
        )
        return (
            (status == int(OptimizationStatus.ITERATING))
            & (iterations < termination.maximum_steps)
            & within_evaluations
        )

    def body(carry):
        (
            current_state_result,
            current_design,
            status,
            iterations,
            state_solves,
            objective_evaluations,
            gradient_evaluations,
            residual_evaluations,
            jvp_evaluations,
            vjp_evaluations,
            hvp_evaluations,
            linear_solves,
            linear_iterations,
            setup_refreshes,
            numeric_refreshes,
            globalization_evaluations,
            accepted_steps,
            rejected_steps,
            final_step_norm,
            accepted_rate,
            initial_optimality,
            _,
        ) = carry
        current_state = current_state_result.state
        value, _ = problem.value(current_state, current_design, args)
        (
            reduced_gradient,
            adjoint,
            adjoint_result,
            adjoint_acceptance,
        ) = _adjoint_gradient(
            problem,
            current_state,
            current_design,
            args,
            method.linear_policy,
            current_state_result.acceptance,
        )
        projected_gradient = (
            reduced_gradient
            if problem.design_bounds is None
            else problem.design_bounds.projected_gradient(
                current_design,
                reduced_gradient,
            )
        )
        optimality = _tree_norm(projected_gradient)
        next_initial_optimality = jnp.where(
            gradient_evaluations == 0,
            optimality,
            initial_optimality,
        )
        model_carry = (
            current_state_result,
            current_design,
            status,
            iterations,
            state_solves,
            objective_evaluations + 3,
            gradient_evaluations + 1,
            residual_evaluations + 3,
            jvp_evaluations + adjoint_result.diagnostics.adjoint_matvec_count,
            vjp_evaluations + adjoint_result.diagnostics.matvec_count + 1,
            hvp_evaluations,
            linear_solves + 1,
            linear_iterations + adjoint_result.diagnostics.iterations,
            setup_refreshes + 1,
            numeric_refreshes + 1,
            globalization_evaluations,
            accepted_steps,
            rejected_steps,
            final_step_norm,
            accepted_rate,
            next_initial_optimality,
            adjoint,
        )

        def fail_model(status_code):
            return (
                current_state_result,
                current_design,
                jnp.asarray(status_code, dtype=jnp.int32),
                iterations,
                state_solves,
                objective_evaluations + 3,
                gradient_evaluations + 1,
                residual_evaluations + 3,
                jvp_evaluations + adjoint_result.diagnostics.adjoint_matvec_count,
                vjp_evaluations + adjoint_result.diagnostics.matvec_count + 1,
                hvp_evaluations,
                linear_solves + 1,
                linear_iterations + adjoint_result.diagnostics.iterations,
                setup_refreshes + 1,
                numeric_refreshes + 1,
                globalization_evaluations,
                accepted_steps,
                rejected_steps + 1,
                final_step_norm,
                accepted_rate,
                next_initial_optimality,
                adjoint,
            )

        def evaluate_direction(_):
            direction = _tree_negative(reduced_gradient)
            if problem.design_bounds is not None:
                direction = _projected_displacement(
                    problem.design_bounds,
                    current_design,
                    direction,
                )
            directional = _tree_inner(reduced_gradient, direction)
            valid_direction = (
                _tree_allfinite(direction)
                & jnp.isfinite(directional)
                & (directional < 0.0)
            )

            def search(_):
                (
                    candidate_state_result,
                    candidate_design,
                    _,
                    accepted,
                    rate,
                    trials,
                    trial_objective_evaluations,
                    trial_residual_evaluations,
                    trial_jvp_evaluations,
                    trial_vjp_evaluations,
                    trial_hvp_evaluations,
                    trial_setup_refreshes,
                    trial_numeric_refreshes,
                    trial_linear_solves,
                    trial_linear_iterations,
                    trial_globalization_evaluations,
                ) = _state_design_line_search(
                    problem,
                    method.line_search,
                    current_state_result,
                    current_design,
                    value,
                    direction,
                    directional,
                    args,
                )
                step_norm = rate * _tree_norm(direction)
                stagnated = accepted & (
                    step_norm <= termination.step_threshold(_tree_norm(candidate_design))
                )
                next_status = jnp.where(
                    accepted,
                    jnp.where(
                        stagnated,
                        int(OptimizationStatus.STAGNATION),
                        int(OptimizationStatus.ITERATING),
                    ),
                    int(OptimizationStatus.LINE_SEARCH_FAILED),
                ).astype(jnp.int32)
                return (
                    candidate_state_result,
                    candidate_design,
                    next_status,
                    iterations + 1,
                    state_solves + trials,
                    objective_evaluations + 3 + trial_objective_evaluations,
                    gradient_evaluations + 1,
                    residual_evaluations + 3 + trial_residual_evaluations,
                    jvp_evaluations
                    + adjoint_result.diagnostics.adjoint_matvec_count
                    + trial_jvp_evaluations,
                    vjp_evaluations
                    + adjoint_result.diagnostics.matvec_count
                    + 1
                    + trial_vjp_evaluations,
                    hvp_evaluations + trial_hvp_evaluations,
                    linear_solves + 1 + trial_linear_solves,
                    linear_iterations
                    + adjoint_result.diagnostics.iterations
                    + trial_linear_iterations,
                    setup_refreshes + 1 + trial_setup_refreshes,
                    numeric_refreshes + 1 + trial_numeric_refreshes,
                    globalization_evaluations + trial_globalization_evaluations,
                    accepted_steps + accepted.astype(jnp.int32),
                    rejected_steps + (~accepted).astype(jnp.int32),
                    step_norm,
                    rate,
                    next_initial_optimality,
                    adjoint,
                )

            return jax.lax.cond(
                valid_direction,
                search,
                lambda _: fail_model(OptimizationStatus.INVALID_DIRECTION),
                None,
            )

        def evaluate_finite_model(_):
            converged = optimality <= termination.optimality_threshold(
                next_initial_optimality
            )
            return jax.lax.cond(
                converged,
                lambda _: (
                    current_state_result,
                    current_design,
                    jnp.asarray(
                        int(OptimizationStatus.SUCCESS),
                        dtype=jnp.int32,
                    ),
                    *model_carry[3:],
                ),
                evaluate_direction,
                None,
            )

        finite_model = (
            jnp.isfinite(value)
            & jnp.isfinite(optimality)
            & _tree_allfinite(reduced_gradient)
        )
        return jax.lax.cond(
            adjoint_acceptance.accepted,
            lambda _: jax.lax.cond(
                finite_model,
                evaluate_finite_model,
                lambda _: fail_model(OptimizationStatus.NONFINITE_EVALUATION),
                None,
            ),
            lambda _: fail_model(OptimizationStatus.LINEAR_SOLVE_FAILED),
            None,
        )

    (
        state_result,
        design,
        status,
        iterations,
        state_solves,
        objective_evaluations,
        gradient_evaluations,
        residual_evaluations,
        jvp_evaluations,
        vjp_evaluations,
        hvp_evaluations,
        linear_solves,
        linear_iterations,
        setup_refreshes,
        numeric_refreshes,
        globalization_evaluations,
        accepted_steps,
        rejected_steps,
        final_step_norm,
        accepted_rate,
        initial_optimality,
        adjoint,
    ) = jax.lax.while_loop(condition, body, initial_carry)
    if termination.maximum_evaluations is not None:
        status = jnp.where(
            (status == int(OptimizationStatus.ITERATING))
            & (objective_evaluations >= termination.maximum_evaluations),
            int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
            status,
        )
    status = jnp.where(
        status == int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        status,
    ).astype(jnp.int32)

    state = state_result.state
    final_value, auxiliary = problem.value(state, design, args)
    objective_evaluations = objective_evaluations + 3
    (
        final_gradient,
        adjoint,
        adjoint_result,
        adjoint_acceptance,
    ) = _adjoint_gradient(
        problem,
        state,
        design,
        args,
        method.linear_policy,
        state_result.acceptance,
    )
    gradient_evaluations = gradient_evaluations + 1
    residual_evaluations = residual_evaluations + 3
    jvp_evaluations = jvp_evaluations + adjoint_result.diagnostics.adjoint_matvec_count
    vjp_evaluations = vjp_evaluations + adjoint_result.diagnostics.matvec_count + 1
    linear_solves = linear_solves + 1
    linear_iterations = linear_iterations + adjoint_result.diagnostics.iterations
    setup_refreshes = setup_refreshes + 1
    numeric_refreshes = numeric_refreshes + 1
    projected_final_gradient = (
        final_gradient
        if problem.design_bounds is None
        else problem.design_bounds.projected_gradient(design, final_gradient)
    )
    final_optimality = _tree_norm(projected_final_gradient)
    status_allows_final_success = (
        (status == int(OptimizationStatus.ITERATING))
        | (status == int(OptimizationStatus.MAXIMUM_STEPS_REACHED))
        | (status == int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED))
        | (status == int(OptimizationStatus.STAGNATION))
    )
    status = jnp.where(
        status_allows_final_success
        & state_result.acceptance.accepted
        & adjoint_acceptance.accepted
        & (final_optimality <= termination.optimality_threshold(initial_optimality)),
        int(OptimizationStatus.SUCCESS),
        status,
    ).astype(jnp.int32)
    primal = state_result.residual_norm
    if problem.design_bounds is not None:
        primal = jnp.maximum(primal, problem.design_bounds.violation(design))
    diagnostics = OptimizationDiagnostics(
        iterations=iterations,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        objective_evaluations=objective_evaluations,
        gradient_evaluations=gradient_evaluations,
        residual_evaluations=residual_evaluations,
        jvp_evaluations=jvp_evaluations,
        vjp_evaluations=vjp_evaluations,
        hvp_evaluations=hvp_evaluations,
        constraint_evaluations=state_solves,
        linear_solves=linear_solves,
        setup_refreshes=setup_refreshes,
        numeric_refreshes=numeric_refreshes,
        linear_iterations=linear_iterations,
        globalization_evaluations=globalization_evaluations,
        initial_optimality_norm=initial_optimality,
        final_optimality_norm=final_optimality,
        final_step_norm=final_step_norm,
        accepted_step_size=accepted_rate,
        primal_feasibility=primal,
        dual_feasibility=final_optimality,
        complementarity=0.0,
        counts_complete=False,
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax",
        backend_method=problem.state_solver.method_id,
        globalization="reduced-objective-armijo",
        matrix_free=True,
        notes="Forward state solves are warm-started; gradients use transpose state solves.",
    )
    return StateDesignResult(
        state,
        design,
        final_value,
        auxiliary,
        adjoint,
        status,
        diagnostics,
        provenance,
        state_acceptance=state_result.acceptance,
        adjoint_acceptance=adjoint_acceptance,
    )


def _solve_simultaneous_kkt(
    method: SimultaneousKKT,
    problem: StateDesignProblem,
    initial_state: PyTree[Any],
    initial_design: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> StateDesignResult:
    if problem.constraints:
        raise ValueError(
            "SimultaneousKKT does not support StateDesignProblem constraints."
        )
    if problem.design_bounds is not None:
        raise ValueError(
            "SimultaneousKKT currently requires an unconstrained design; use "
            "ReducedAdjoint for bound-constrained designs."
        )
    state = _validate_real_inexact_tree(initial_state, name="initial_state")
    design = _validate_real_inexact_tree(initial_design, name="initial_design")
    initial_residual = problem.residual(state, design, args)
    adjoint = jax.tree.map(jnp.zeros_like, initial_residual)

    def kkt_residual(variables, dynamic_args):
        current_state, current_design, current_adjoint = variables
        residual = problem.residual(current_state, current_design, dynamic_args)
        state_objective_gradient = jax.grad(
            lambda value: problem.value(value, current_design, dynamic_args)[0]
        )(current_state)
        design_objective_gradient = jax.grad(
            lambda value: problem.value(current_state, value, dynamic_args)[0]
        )(current_design)
        _, pullback = jax.vjp(
            lambda state_value, design_value: problem.residual(
                state_value,
                design_value,
                dynamic_args,
            ),
            current_state,
            current_design,
        )
        state_residual_adjoint, design_residual_adjoint = pullback(current_adjoint)
        state_stationarity = jax.tree.map(
            lambda objective_part, residual_part: objective_part - residual_part,
            state_objective_gradient,
            state_residual_adjoint,
        )
        design_stationarity = jax.tree.map(
            lambda objective_part, residual_part: objective_part - residual_part,
            design_objective_gradient,
            design_residual_adjoint,
        )
        return state_stationarity, design_stationarity, residual

    kkt_problem = NonlinearLeastSquaresProblem(
        kkt_residual,
        problem_id=f"{problem.problem_id}/simultaneous-kkt",
    )
    result = method.method.solve(
        kkt_problem,
        (state, design, adjoint),
        termination=termination,
        args=args,
    )
    state, design, adjoint = result.parameters
    state_stationarity, design_stationarity, residual = result.residual
    primal = _tree_norm(residual)
    dual = jnp.sqrt(
        _tree_norm(state_stationarity) ** 2 + _tree_norm(design_stationarity) ** 2
    )
    state_acceptance = problem.state_evidence(
        state,
        design,
        residual,
        result.status,
        reference_norm=_tree_norm(initial_residual),
        args=args,
    )
    state_objective_gradient = jax.grad(
        lambda current_state: problem.value(current_state, design, args)[0]
    )(state)
    transpose_image = jax.tree.map(
        lambda right, defect: right - defect,
        state_objective_gradient,
        state_stationarity,
    )
    adjoint_status = jnp.where(
        result.status == int(OptimizationStatus.SUCCESS),
        int(LinearSolveStatus.SUCCESS),
        jnp.where(
            result.status == int(OptimizationStatus.STAGNATION),
            int(LinearSolveStatus.STAGNATION),
            int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
        ),
    ).astype(jnp.int32)
    adjoint_acceptance = problem.acceptance_policy.adjoint_evidence(
        adjoint,
        transpose_image,
        state_objective_gradient,
        adjoint_status,
        admissible=state_acceptance.admissible & state_acceptance.finite,
        realization_matches=state_acceptance.realization_matches,
    )
    status = jnp.where(
        (result.status == int(OptimizationStatus.SUCCESS))
        & (~state_acceptance.accepted | ~adjoint_acceptance.accepted),
        int(OptimizationStatus.CERTIFICATION_FAILED),
        result.status,
    ).astype(jnp.int32)
    final_value, auxiliary = problem.value(state, design, args)
    diagnostics = eqx.tree_at(
        lambda item: (
            item.objective_evaluations,
            item.primal_feasibility,
            item.dual_feasibility,
            item.complementarity,
        ),
        result.diagnostics,
        (
            result.diagnostics.objective_evaluations + 1,
            primal,
            dual,
            jnp.asarray(0.0),
        ),
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax",
        backend_method=method.method.method_id,
        globalization="least-squares-trust-region",
        matrix_free=True,
        notes="State, design, and adjoint are solved together from the full KKT residual.",
    )
    return StateDesignResult(
        state,
        design,
        final_value,
        auxiliary,
        adjoint,
        status,
        diagnostics,
        provenance,
        state_acceptance=state_acceptance,
        adjoint_acceptance=adjoint_acceptance,
    )


def solve_state_design(
    problem: StateDesignProblem,
    initial_state: PyTree[Any],
    initial_design: PyTree[Any],
    /,
    *,
    method: AbstractStateDesignMethod,
    termination: OptimizationTermination | None = None,
    args: Any = None,
) -> StateDesignResult:
    """Solve one state-constrained optimization problem."""

    if not isinstance(problem, StateDesignProblem):
        raise TypeError("problem must be a StateDesignProblem.")
    if not isinstance(method, AbstractStateDesignMethod):
        raise TypeError("method must be an AbstractStateDesignMethod.")
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination or None.")
    return method.solve(
        problem,
        initial_state,
        initial_design,
        termination=termination_,
        args=args,
    )


__all__ = [
    "AbstractStateDesignMethod",
    "AdjointAcceptanceEvidence",
    "StateAcceptanceEvidence",
    "StateAcceptancePolicy",
    "AbstractStateSolver",
    "LeastSquaresStateSolver",
    "ReducedAdjoint",
    "SimultaneousKKT",
    "StateDesignConstraint",
    "StateDesignProblem",
    "StateDesignResult",
    "StateEquationResult",
    "solve_state_design",
]
