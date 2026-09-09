#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree

from .._strict import StrictModule
from ._iterative._types import (
    _tree_norm,
    _validate_real_inexact_tree,
    Bounds,
    OptimizationDiagnostics,
    OptimizationStatus,
)
from ._pde_constrained import (
    AbstractStateSolver,
    AdjointAcceptanceEvidence,
    StateAcceptanceEvidence,
    StateAcceptancePolicy,
    StateDesignConstraint,
    StateDesignProblem,
    StateEquationResult,
)


class StateDesignCase(StrictModule):
    """One independent state equation and its shared/local design binding.

    ``design_binding(shared, own_local, args)`` returns the child's complete
    design. ``args`` is this case's operating data, also supplied to the child
    problem. A binding never receives another case's state or local variables.
    Cross-case responses belong in the multipoint objective or constraints.
    """

    case_id: str = eqx.field(static=True)
    problem: StateDesignProblem
    initial_state: PyTree[Any]
    design_binding: Callable
    args: Any

    def __init__(
        self,
        case_id: str,
        problem: StateDesignProblem,
        initial_state: PyTree[Any],
        design_binding: Callable,
        /,
        *,
        args: Any = None,
    ):
        identifier = str(case_id)
        if not identifier:
            raise ValueError("case_id must be non-empty.")
        if not isinstance(problem, StateDesignProblem):
            raise TypeError("problem must be a StateDesignProblem.")
        if not callable(design_binding):
            raise TypeError("design_binding must be callable.")
        self.case_id = identifier
        self.problem = problem
        self.initial_state = _validate_real_inexact_tree(
            initial_state, name="case initial_state"
        )
        self.design_binding = design_binding
        self.args = args

    def bind_design(self, shared, own_local, /):
        """Bind only this case's design at its declared operating point."""
        return _validate_real_inexact_tree(
            self.design_binding(shared, own_local, self.args),
            name=f"case {self.case_id!r} design",
        )


def _check_design(design, case_count):
    if not isinstance(design, dict) or set(design) != {"shared", "local"}:
        raise TypeError("Multipoint design must be {'shared': tree, 'local': tuple}.")
    if not isinstance(design["local"], tuple) or len(design["local"]) != case_count:
        raise ValueError("Multipoint local design must have one tuple entry per case.")


def _check_states(state, case_count):
    if not isinstance(state, tuple) or len(state) != case_count:
        raise ValueError("Multipoint state must have one tuple entry per case.")


class _MultipointResidual(StrictModule):
    cases: tuple[StateDesignCase, ...]

    def __call__(self, state, design, args):
        del args
        _check_design(design, len(self.cases))
        _check_states(state, len(self.cases))
        return tuple(
            case.problem.residual(
                state[index],
                case.bind_design(design["shared"], design["local"][index]),
                case.args,
            )
            for index, case in enumerate(self.cases)
        )


class _MultipointObjective(StrictModule):
    cases: tuple[StateDesignCase, ...]

    def __call__(self, state, design, args):
        del args
        _check_design(design, len(self.cases))
        _check_states(state, len(self.cases))
        return sum(
            case.problem.value(
                state[index],
                case.bind_design(design["shared"], design["local"][index]),
                case.args,
            )[0]
            for index, case in enumerate(self.cases)
        )


class _CaseResponse(StrictModule):
    case: StateDesignCase
    index: int = eqx.field(static=True)
    case_count: int = eqx.field(static=True)
    constraint: StateDesignConstraint | None

    def __call__(self, state, design, args):
        del args
        _check_design(design, self.case_count)
        bound_design = self.case.bind_design(
            design["shared"], design["local"][self.index]
        )
        if self.constraint is None:
            return bound_design
        _check_states(state, self.case_count)
        return self.constraint.value(state[self.index], bound_design, self.case.args)


class _MultipointStateCertification(StrictModule):
    cases: tuple[StateDesignCase, ...]

    def __call__(
        self,
        state,
        design,
        residual,
        status,
        /,
        *,
        reference_norm,
        args=None,
        solver_acceptance=None,
    ):
        # References are repeatable per operating point, never an aggregate norm
        # or the changing warm start of an outer optimizer.
        del reference_norm, args
        _check_design(design, len(self.cases))
        _check_states(state, len(self.cases))
        _check_states(residual, len(self.cases))
        if solver_acceptance is not None and solver_acceptance.block_ids != tuple(
            case.case_id for case in self.cases
        ):
            raise ValueError("Multipoint solver evidence must identify every case.")
        blocks = []
        for index, case in enumerate(self.cases):
            bound_design = case.bind_design(design["shared"], design["local"][index])
            reference = case.problem.residual(case.initial_state, bound_design, case.args)
            previous = (
                None if solver_acceptance is None else solver_acceptance.blocks[index]
            )
            evidence = case.problem.state_evidence(
                state[index],
                bound_design,
                residual[index],
                status,
                reference_norm=_tree_norm(reference),
                args=case.args,
                solver_acceptance=previous,
            )
            if previous is not None:
                evidence = _with_solver_status(evidence, previous)
            blocks.append(evidence)
        return StateAcceptanceEvidence.from_blocks(
            tuple(case.case_id for case in self.cases), tuple(blocks)
        )


def _with_solver_status(evidence, previous):
    """Keep actual child status gates while independently recertifying physics."""
    if evidence.block_ids != previous.block_ids:
        raise ValueError("Recertified state evidence changed block identities.")
    blocks = tuple(
        _with_solver_status(current, old)
        for current, old in zip(evidence.blocks, previous.blocks, strict=True)
    )
    return StateAcceptanceEvidence(
        evidence.residual_norm,
        evidence.reference_norm,
        evidence.threshold,
        evidence.finite,
        evidence.admissible,
        evidence.realization_matches,
        previous.status_accepted,
        blocks=blocks,
        block_ids=evidence.block_ids,
    )


class _MultipointAdjointCertification(StrictModule):
    cases: tuple[StateDesignCase, ...]

    def __call__(
        self,
        adjoint,
        transpose_image,
        right_hand_side,
        status,
        /,
        *,
        admissible,
        realization_matches,
    ):
        for value in (adjoint, transpose_image, right_hand_side):
            _check_states(value, len(self.cases))
        return AdjointAcceptanceEvidence.from_blocks(
            tuple(case.case_id for case in self.cases),
            tuple(
                case.problem.acceptance_policy.adjoint_evidence(
                    adjoint[index],
                    transpose_image[index],
                    right_hand_side[index],
                    status,
                    admissible=admissible,
                    realization_matches=realization_matches,
                )
                for index, case in enumerate(self.cases)
            ),
        )


class _MultipointStateSolver(AbstractStateSolver):
    cases: tuple[StateDesignCase, ...]

    def __init__(self, cases, /):
        self.cases = tuple(cases)

    @property
    def method_id(self):
        return "multipoint-independent-state"

    def solve(self, problem, design, initial_state, /, *, args):
        del problem, args
        _check_design(design, len(self.cases))
        _check_states(initial_state, len(self.cases))
        results = tuple(
            case.problem.solve_state(
                case.bind_design(design["shared"], design["local"][index]),
                initial_state[index],
                args=case.args,
            )
            for index, case in enumerate(self.cases)
        )
        state = tuple(result.state for result in results)
        residual = tuple(result.residual for result in results)
        acceptance = StateAcceptanceEvidence.from_blocks(
            tuple(case.case_id for case in self.cases),
            tuple(result.acceptance for result in results),
        )
        # All blocks belong to this design, including failures. The native outer
        # line search commits or rejects this complete tuple, never individual
        # successful children from a rejected trial.
        status = jnp.where(
            acceptance.status_accepted,
            int(OptimizationStatus.SUCCESS),
            int(OptimizationStatus.CERTIFICATION_FAILED),
        ).astype(jnp.int32)

        # Only work counters add across heterogeneous child solvers. Quantities
        # such as damping, step size and optimality norms have no common scale.
        def total(select):
            return sum(select(result.diagnostics) for result in results)

        diagnostics = OptimizationDiagnostics(
            iterations=total(lambda value: value.iterations),
            accepted_steps=total(lambda value: value.accepted_steps),
            rejected_steps=total(lambda value: value.rejected_steps),
            objective_evaluations=total(lambda value: value.objective_evaluations),
            gradient_evaluations=total(lambda value: value.gradient_evaluations),
            residual_evaluations=total(lambda value: value.residual_evaluations),
            jvp_evaluations=total(lambda value: value.jvp_evaluations),
            vjp_evaluations=total(lambda value: value.vjp_evaluations),
            hvp_evaluations=total(lambda value: value.hvp_evaluations),
            jacobian_evaluations=total(lambda value: value.jacobian_evaluations),
            constraint_evaluations=total(lambda value: value.constraint_evaluations),
            linear_solves=total(lambda value: value.linear_solves),
            setup_refreshes=total(lambda value: value.setup_refreshes),
            numeric_refreshes=total(lambda value: value.numeric_refreshes),
            linear_iterations=total(lambda value: value.linear_iterations),
            globalization_evaluations=total(
                lambda value: value.globalization_evaluations
            ),
            direction_fallbacks=total(lambda value: value.direction_fallbacks),
            primal_feasibility=jnp.max(
                jnp.stack(
                    tuple(result.acceptance.normalized_residual for result in results)
                )
            ),
            counts_complete=False,
        )
        return StateEquationResult(state, residual, status, diagnostics, acceptance)


class MultipointStateDesignProblem(StrictModule):
    """Independent heterogeneous states with shared and case-local designs.

    Lower with :meth:`to_state_design_problem`, then use the ordinary reduced
    methods or structured all-at-once compiler. The state is a case-ordered
    tuple; the design is ``{'shared': tree, 'local': tuple}``. The default
    objective sums child objectives. An explicit objective and the additional
    constraints see the complete state/design and the outer solver's ``args``.

    Child constraints and child design bounds are always retained. A bound on
    an arbitrary child design binding becomes a nonlinear constraint, not a
    projected bound on unrelated shared coordinates. Vector and equality
    constraints require the structured NLP path; ReducedMMA retains its native
    scalar-inequality contract. ReducedAdjoint requires no composed constraints.

    No state coupling or execution pool is introduced: every child residual
    sees only its own state, bound design, and declared operating arguments.
    """

    cases: tuple[StateDesignCase, ...]
    objective: Callable | None
    constraints: tuple[StateDesignConstraint, ...]
    design_bounds: Bounds | None
    has_aux: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        cases: Sequence[StateDesignCase],
        /,
        *,
        objective: Callable | None = None,
        constraints: Sequence[StateDesignConstraint] = (),
        design_bounds: Bounds | None = None,
        has_aux: bool = False,
        problem_id: str = "multipoint-state-design",
    ):
        cases_ = tuple(cases)
        if not cases_ or any(not isinstance(case, StateDesignCase) for case in cases_):
            raise TypeError(
                "cases must be a non-empty sequence of StateDesignCase values."
            )
        identifiers = tuple(case.case_id for case in cases_)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Multipoint case_ids must be unique.")
        if objective is not None and not callable(objective):
            raise TypeError("objective must be callable or None.")
        if objective is None and has_aux:
            raise ValueError("has_aux=True requires an explicit multipoint objective.")
        constraints_ = tuple(constraints)
        if any(not isinstance(item, StateDesignConstraint) for item in constraints_):
            raise TypeError("constraints must contain StateDesignConstraint values.")
        if design_bounds is not None and not isinstance(design_bounds, Bounds):
            raise TypeError("design_bounds must be Bounds or None.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.cases = cases_
        self.objective = objective
        self.constraints = constraints_
        self.design_bounds = design_bounds
        self.has_aux = bool(has_aux)
        self.problem_id = identifier

    @property
    def case_ids(self):
        return tuple(case.case_id for case in self.cases)

    @property
    def initial_state(self):
        return tuple(case.initial_state for case in self.cases)

    def to_state_design_problem(self) -> StateDesignProblem:
        """Lower without dropping bounds, constraints, or block acceptance."""
        constraints = []
        for index, case in enumerate(self.cases):
            for constraint in case.problem.constraints:
                constraints.append(
                    StateDesignConstraint(
                        _CaseResponse(case, index, len(self.cases), constraint),
                        lower=constraint.lower,
                        upper=constraint.upper,
                        constraint_id=f"{case.case_id}:{constraint.constraint_id}",
                        depends_on_state=constraint.depends_on_state,
                    )
                )
            if case.problem.design_bounds is not None:
                constraints.append(
                    StateDesignConstraint(
                        _CaseResponse(case, index, len(self.cases), None),
                        lower=case.problem.design_bounds.lower,
                        upper=case.problem.design_bounds.upper,
                        constraint_id=f"{case.case_id}:design-bounds",
                        depends_on_state=False,
                    )
                )
        constraints.extend(self.constraints)
        identifiers = tuple(item.constraint_id for item in constraints)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(
                "Composed constraint_ids must be unique, including case bounds."
            )
        return StateDesignProblem(
            _MultipointResidual(self.cases),
            _MultipointObjective(self.cases)
            if self.objective is None
            else self.objective,
            state_solver=_MultipointStateSolver(self.cases),
            acceptance_policy=StateAcceptancePolicy(
                adjoint_certification=_MultipointAdjointCertification(self.cases),
                accepted_adjoint_statuses=tuple(
                    sorted(
                        {
                            status
                            for case in self.cases
                            for status in case.problem.acceptance_policy.accepted_adjoint_statuses
                        }
                    )
                ),
            ),
            state_certification=_MultipointStateCertification(self.cases),
            design_bounds=self.design_bounds,
            constraints=tuple(constraints),
            has_aux=self.has_aux,
            problem_id=self.problem_id,
        )


__all__ = ["MultipointStateDesignProblem", "StateDesignCase"]
