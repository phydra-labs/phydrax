#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...continuation import ContinuationResult
from ...linalg import LinearSolveStatus
from ...optim import (
    AbstractStateSolver,
    AdjointAcceptanceEvidence,
    OptimizationDiagnostics,
    OptimizationStatus,
    StateAcceptanceEvidence,
    StateDesignProblem,
    StateEquationResult,
)


def _tree_allfinite(tree: PyTree[Any], /) -> Array:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError("A mechanics state must contain at least one array leaf.")
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in leaves)))


def _tree_norm(tree: PyTree[Any], /) -> Array:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError("A mechanics residual must contain at least one array leaf.")
    squares = tuple(jnp.real(jnp.vdot(value, value)) for value in leaves)
    return jnp.sqrt(sum(squares[1:], start=squares[0]))


def _tree_select(predicate: Array, accepted: PyTree[Any], fallback: PyTree[Any], /):
    if jax.tree.structure(accepted) != jax.tree.structure(fallback):
        raise ValueError(
            "Proposed and warm mechanics states must share one PyTree structure."
        )
    return jax.tree.map(
        lambda candidate, previous: jnp.where(predicate, candidate, previous),
        accepted,
        fallback,
    )


class MechanicsStateCandidate(StrictModule):
    """One state returned by an executed finite-element root method."""

    state: PyTree[Array]
    status: Array
    diagnostics: OptimizationDiagnostics

    def __init__(
        self,
        state: PyTree[Any],
        /,
        *,
        status: Any = OptimizationStatus.SUCCESS,
        diagnostics: OptimizationDiagnostics | None = None,
    ):
        leaves = jax.tree.leaves(state)
        if not leaves:
            raise ValueError("state must contain at least one array leaf.")
        state_ = jax.tree.map(jnp.asarray, state)
        if any(
            not jnp.issubdtype(value.dtype, jnp.floating)
            for value in jax.tree.leaves(state_)
        ):
            raise TypeError("Mechanics state leaves must be real inexact arrays.")
        status_ = jnp.asarray(status, dtype=jnp.int32)
        if status_.shape != ():
            raise ValueError("Mechanics state status must be scalar.")
        diagnostics_ = OptimizationDiagnostics() if diagnostics is None else diagnostics
        if not isinstance(diagnostics_, OptimizationDiagnostics):
            raise TypeError("diagnostics must be OptimizationDiagnostics or None.")
        self.state = state_
        self.status = status_
        self.diagnostics = diagnostics_


class FiniteElementStateSolver(AbstractStateSolver):
    """Authoritative FE state root backed by a caller-supplied executed solve."""

    solve_function: Callable = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)

    def __init__(self, solve: Callable, /, *, solver_id: str = "finite-element-state"):
        if not callable(solve):
            raise TypeError("solve must be callable.")
        identifier = str(solver_id)
        if not identifier:
            raise ValueError("solver_id must be non-empty.")
        self.solve_function = solve
        self.solver_id = identifier

    @property
    def method_id(self) -> str:
        return f"finite-element/{self.solver_id}"

    @property
    def authoritative(self) -> bool:
        return True

    def solve(
        self,
        problem: StateDesignProblem,
        design: PyTree[Any],
        initial_state: PyTree[Any],
        /,
        *,
        args: Any,
    ) -> StateEquationResult:
        if not isinstance(problem, StateDesignProblem):
            raise TypeError("problem must be a StateDesignProblem.")
        reference = problem.residual(initial_state, design, args)
        candidate = self.solve_function(problem, design, initial_state, args)
        if not isinstance(candidate, MechanicsStateCandidate):
            raise TypeError("An FE solve must return MechanicsStateCandidate.")
        residual = problem.residual(candidate.state, design, args)
        acceptance = problem.state_evidence(
            candidate.state,
            design,
            residual,
            candidate.status,
            reference_norm=_tree_norm(reference),
            args=args,
        )
        return StateEquationResult(
            candidate.state,
            residual,
            candidate.status,
            candidate.diagnostics,
            acceptance,
        )


class NeuralProposalEvidence(StrictModule):
    """Support, residual, and exact warm-state rollback for one learned proposal."""

    proposed_state: PyTree[Array]
    selected_initial_state: PyTree[Array]
    residual_norm: Array
    reference_norm: Array
    threshold: Array
    supported: Array
    finite: Array
    accepted: Array
    rollback_applied: Array


class NeuralVariationalRoot(StrictModule):
    """Learned proposal evidence followed by an authoritative FE root."""

    proposal: NeuralProposalEvidence
    state_equation: StateEquationResult
    final_fe_reanalysis: Array
    authoritative_solver_id: str = eqx.field(static=True)


class NeuralVariationalStateSolver(AbstractStateSolver):
    """Use a learned operator only as an FE variational-root initial proposal.

    A proposal is never returned as a mechanics state. Unsupported, nonfinite, or
    excessively imbalanced proposals roll back exactly to the supplied warm state,
    after which the authoritative FE solver always executes and certifies its root.
    """

    proposal_function: Callable = eqx.field(static=True)
    support_function: Callable | None = eqx.field(static=True)
    finite_element_solver: FiniteElementStateSolver
    relative_residual_limit: float = eqx.field(static=True)
    absolute_residual_limit: float = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        proposal: Callable,
        finite_element_solver: FiniteElementStateSolver,
        /,
        *,
        support: Callable | None = None,
        relative_residual_limit: float = 10.0,
        absolute_residual_limit: float = 1.0e-8,
        solver_id: str = "neural-variational-state",
    ):
        if not callable(proposal):
            raise TypeError("proposal must be callable.")
        if not isinstance(finite_element_solver, FiniteElementStateSolver):
            raise TypeError("finite_element_solver must be FiniteElementStateSolver.")
        if support is not None and not callable(support):
            raise TypeError("support must be callable or None.")
        relative = float(relative_residual_limit)
        absolute = float(absolute_residual_limit)
        identifier = str(solver_id)
        if (
            not isfinite(relative)
            or relative < 0.0
            or not isfinite(absolute)
            or absolute < 0.0
        ):
            raise ValueError("Proposal residual limits must be finite and non-negative.")
        if not identifier:
            raise ValueError("solver_id must be non-empty.")
        self.proposal_function = proposal
        self.support_function = support
        self.finite_element_solver = finite_element_solver
        self.relative_residual_limit = relative
        self.absolute_residual_limit = absolute
        self.solver_id = identifier

    @property
    def method_id(self) -> str:
        return f"neural-proposal/{self.solver_id}+{self.finite_element_solver.method_id}"

    @property
    def authoritative(self) -> bool:
        return True

    def prepare_initial_state(
        self,
        problem: StateDesignProblem,
        design: PyTree[Any],
        initial_state: PyTree[Any],
        /,
        *,
        args: Any,
    ) -> NeuralProposalEvidence:
        support_value = (
            jnp.asarray(True)
            if self.support_function is None
            else jnp.asarray(self.support_function(design, args), dtype=bool)
        )
        if support_value.shape != ():
            raise ValueError("Neural proposal support evidence must be scalar.")
        supported_static = bool(support_value)
        proposed = (
            jax.tree.map(jnp.asarray, initial_state)
            if not supported_static
            else jax.tree.map(
                jnp.asarray,
                self.proposal_function(problem, design, initial_state, args),
            )
        )
        if jax.tree.structure(proposed) != jax.tree.structure(initial_state):
            raise ValueError(
                "A neural proposal must preserve the mechanics state structure."
            )
        reference_residual = problem.residual(initial_state, design, args)
        proposed_residual = problem.residual(proposed, design, args)
        reference_norm = _tree_norm(reference_residual)
        residual_norm = _tree_norm(proposed_residual)
        threshold = jnp.maximum(
            jnp.asarray(self.absolute_residual_limit, dtype=residual_norm.dtype),
            jnp.asarray(self.relative_residual_limit, dtype=residual_norm.dtype)
            * reference_norm,
        )
        supported = jnp.asarray(supported_static)
        finite = _tree_allfinite(proposed) & jnp.isfinite(residual_norm)
        accepted = supported & finite & (residual_norm <= threshold)
        selected = _tree_select(accepted, proposed, initial_state)
        return NeuralProposalEvidence(
            proposed,
            selected,
            residual_norm,
            reference_norm,
            threshold,
            supported,
            finite,
            accepted,
            ~accepted,
        )

    def solve_root(
        self,
        problem: StateDesignProblem,
        design: PyTree[Any],
        initial_state: PyTree[Any],
        /,
        *,
        args: Any,
    ) -> NeuralVariationalRoot:
        proposal = self.prepare_initial_state(
            problem,
            design,
            initial_state,
            args=args,
        )
        state_equation = self.finite_element_solver.solve(
            problem,
            design,
            proposal.selected_initial_state,
            args=args,
        )
        return NeuralVariationalRoot(
            proposal,
            state_equation,
            jnp.asarray(True),
            self.finite_element_solver.method_id,
        )

    def solve(
        self,
        problem: StateDesignProblem,
        design: PyTree[Any],
        initial_state: PyTree[Any],
        /,
        *,
        args: Any,
    ) -> StateEquationResult:
        return self.solve_root(
            problem,
            design,
            initial_state,
            args=args,
        ).state_equation


class StateAdjointEvidence(StrictModule):
    """Independently recomputed state and transpose-root acceptance evidence."""

    state: StateAcceptanceEvidence
    adjoint: AdjointAcceptanceEvidence
    accepted: Array


def certify_state_adjoint(
    problem: StateDesignProblem,
    state: PyTree[Any],
    design: PyTree[Any],
    adjoint: PyTree[Any],
    /,
    *,
    reference_state: PyTree[Any],
    state_status: Any = OptimizationStatus.SUCCESS,
    adjoint_status: Any = LinearSolveStatus.SUCCESS,
    args: Any = None,
) -> StateAdjointEvidence:
    """Recompute the primal residual and transpose equation without trusting a solver."""

    if not isinstance(problem, StateDesignProblem):
        raise TypeError("problem must be a StateDesignProblem.")
    residual = problem.residual(state, design, args)
    reference = problem.residual(reference_state, design, args)
    state_evidence = problem.state_evidence(
        state,
        design,
        residual,
        state_status,
        reference_norm=_tree_norm(reference),
        args=args,
    )
    objective_state_gradient = jax.grad(
        lambda current: problem.value(current, design, args)[0]
    )(state)
    _, pullback = jax.vjp(lambda current: problem.residual(current, design, args), state)
    transpose_image = pullback(adjoint)[0]
    adjoint_evidence = problem.acceptance_policy.adjoint_evidence(
        adjoint,
        transpose_image,
        objective_state_gradient,
        adjoint_status,
        admissible=state_evidence.admissible & state_evidence.finite,
        realization_matches=state_evidence.realization_matches,
    )
    return StateAdjointEvidence(
        state_evidence,
        adjoint_evidence,
        state_evidence.accepted & adjoint_evidence.accepted,
    )


class BranchGateEvidence(StrictModule):
    """Branch identity and topology-event decision for one mechanics load case."""

    continuation_accepted: Array
    branch_matches: Array
    forbidden_continuation_event: Array
    contact_event: Array
    fracture_event: Array
    accepted: Array
    branch_id: str = eqx.field(static=True)


class MechanicsBranchGate(StrictModule, NonTrainableState):
    """Reject undeclared nonlinear branches and contact/fracture topology events."""

    accepted_branch_ids: tuple[str, ...] = eqx.field(static=True)
    forbidden_event_kinds: tuple[str, ...] = eqx.field(static=True)
    reject_contact_events: bool = eqx.field(static=True)
    reject_fracture_events: bool = eqx.field(static=True)

    def __init__(
        self,
        accepted_branch_ids: Sequence[str],
        /,
        *,
        forbidden_event_kinds: Sequence[str] = (
            "fold-candidate",
            "hopf-candidate",
            "stability-real-crossing",
            "stability-analysis-failure",
        ),
        reject_contact_events: bool = True,
        reject_fracture_events: bool = True,
    ):
        branches = tuple(str(value) for value in accepted_branch_ids)
        events = tuple(str(value) for value in forbidden_event_kinds)
        if not branches or any(not value for value in branches):
            raise ValueError("accepted_branch_ids must be non-empty identifiers.")
        if len(set(branches)) != len(branches):
            raise ValueError("accepted_branch_ids must be unique.")
        if any(not value for value in events) or len(set(events)) != len(events):
            raise ValueError("forbidden_event_kinds must be non-empty and unique.")
        self.accepted_branch_ids = branches
        self.forbidden_event_kinds = events
        self.reject_contact_events = bool(reject_contact_events)
        self.reject_fracture_events = bool(reject_fracture_events)

    def evaluate(
        self,
        branch_id: str,
        /,
        *,
        continuation: ContinuationResult | None = None,
        contact_event: Any = False,
        fracture_event: Any = False,
    ) -> BranchGateEvidence:
        identifier = str(branch_id)
        if not identifier:
            raise ValueError("branch_id must be non-empty.")
        if continuation is not None and not isinstance(continuation, ContinuationResult):
            raise TypeError("continuation must be a ContinuationResult or None.")
        continuation_accepted = (
            jnp.asarray(True) if continuation is None else continuation.successful
        )
        forbidden = (
            jnp.asarray(False)
            if continuation is None
            else jnp.asarray(
                any(
                    event.kind in self.forbidden_event_kinds
                    for event in continuation.events
                )
            )
        )
        contact = jnp.asarray(contact_event, dtype=bool)
        fracture = jnp.asarray(fracture_event, dtype=bool)
        if contact.shape != () or fracture.shape != ():
            raise ValueError("Contact and fracture event indicators must be scalar.")
        branch_matches = jnp.asarray(
            identifier in self.accepted_branch_ids
            and (continuation is None or continuation.branch.branch_id == identifier)
        )
        accepted = (
            continuation_accepted
            & branch_matches
            & ~forbidden
            & (~contact if self.reject_contact_events else jnp.asarray(True))
            & (~fracture if self.reject_fracture_events else jnp.asarray(True))
        )
        return BranchGateEvidence(
            continuation_accepted,
            branch_matches,
            forbidden,
            contact,
            fracture,
            accepted,
            identifier,
        )


__all__ = [
    "BranchGateEvidence",
    "FiniteElementStateSolver",
    "MechanicsBranchGate",
    "MechanicsStateCandidate",
    "NeuralProposalEvidence",
    "NeuralVariationalRoot",
    "NeuralVariationalStateSolver",
    "StateAdjointEvidence",
    "certify_state_adjoint",
]
