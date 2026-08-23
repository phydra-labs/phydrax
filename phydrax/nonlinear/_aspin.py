#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from ..linalg import (
    DenseLinearOperator,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare as prepare_linear_solve,
    solve as solve_linear_system,
)
from ._decomposition import NonlinearAdditiveSchwarz
from ._linearization import JacobianPolicy
from ._newton import NewtonKrylov
from ._preconditioning import (
    FunctionLeftNonlinearPreconditioner,
    LeftPreconditionedSystem,
)
from ._types import (
    AbstractNonlinearMethod,
    NonlinearCapabilities,
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._updates import (
    apply_prepared_nonlinear_update,
    prepare_nonlinear_update,
    PreparedNonlinearUpdate,
    refresh_nonlinear_update,
)


class ASPIN(AbstractNonlinearMethod):
    """Additive-Schwarz preconditioned Newton with prepared local solves."""

    schwarz: NonlinearAdditiveSchwarz
    outer: NewtonKrylov
    local_linear_policy: LinearSolvePolicy

    def __init__(
        self,
        schwarz: NonlinearAdditiveSchwarz,
        /,
        *,
        outer: NewtonKrylov | None = None,
        local_linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(schwarz, NonlinearAdditiveSchwarz):
            raise TypeError("schwarz must be NonlinearAdditiveSchwarz.")
        outer_ = NewtonKrylov() if outer is None else outer
        if not isinstance(outer_, NewtonKrylov):
            raise TypeError("outer must be NewtonKrylov or None.")
        local_policy = (
            LinearSolvePolicy() if local_linear_policy is None else local_linear_policy
        )
        if not isinstance(local_policy, LinearSolvePolicy):
            raise TypeError("local_linear_policy must be LinearSolvePolicy or None.")
        self.schwarz = schwarz
        self.outer = outer_
        self.local_linear_policy = local_policy

    @property
    def method_id(self) -> str:
        return "aspin"

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return NonlinearCapabilities(
            matrix_free=True,
            prepared_refresh=False,
            jit=False,
            implicit_differentiation=True,
            nonlinear_preconditioning=True,
        )

    def solve(
        self,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any],
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
    ) -> NonlinearResult:
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be NonlinearSystemProblem.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination.")
        if (
            termination.maximum_evaluations is not None
            and termination.maximum_evaluations < 2
        ):
            raise ValueError(
                "ASPIN requires at least two residual evaluations to solve and "
                "certify the physical system."
            )
        initial = problem.validate_state(initial_state)
        initial_residual, _ = problem.evaluate(initial, args)
        problem_ = problem.bind_spaces(initial, initial_residual)
        if problem_.state_space is None or problem_.residual_space is None:
            raise ValueError("ASPIN requires bound physical vector spaces.")
        if problem_.state_space.size != problem_.residual_space.size:
            raise ValueError(
                "ASPIN requires equal physical state and residual dimensions."
            )
        prepared = prepare_nonlinear_update(
            problem_,
            initial,
            self.schwarz,
            args=args,
        )

        def preconditioned_residual(state, residual, current_args):
            del residual
            result, _ = apply_prepared_nonlinear_update(
                prepared,
                state,
                args=current_args,
            )
            return jax.tree.map(
                lambda value, corrected: value - corrected,
                state,
                result.state,
            )

        preconditioner = FunctionLeftNonlinearPreconditioner(
            preconditioned_residual,
            state_space=problem_.state_space,
            source=problem_.residual_space,
            target=problem_.state_space,
            preconditioner_id=f"aspin/{self.schwarz.update_id}",
        )
        transformed = LeftPreconditionedSystem(problem_, preconditioner)

        def operator(state, current_args):
            return _aspin_operator(
                problem_,
                self.schwarz,
                prepared,
                state,
                current_args,
                self.local_linear_policy,
            )

        outer = NewtonKrylov(
            jacobian_policy=JacobianPolicy("explicit", operator=operator),
            linear_policy=self.outer.linear_policy,
            forcing_policy=self.outer.forcing_policy,
            jacobian_refresh=self.outer.jacobian_refresh,
            line_search=self.outer.line_search,
        )
        inner_termination = NonlinearTermination(
            absolute_residual=0.01 * termination.absolute_residual,
            relative_residual=0.01 * termination.relative_residual,
            absolute_step=termination.absolute_step,
            relative_step=termination.relative_step,
            maximum_steps=termination.maximum_steps,
            maximum_evaluations=(
                None
                if termination.maximum_evaluations is None
                else termination.maximum_evaluations - 1
            ),
            maximum_linear_iterations=termination.maximum_linear_iterations,
            divergence_factor=termination.divergence_factor,
        )
        transformed_result = outer.solve(
            transformed.problem,
            initial,
            termination=inner_termination,
            args=args,
        )
        result = transformed.finalize_result(
            transformed_result,
            initial,
            termination,
            args=args,
        )
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=result.diagnostics.initial_residual_norm,
            final_residual_norm=result.diagnostics.final_residual_norm,
            final_step_norm=result.diagnostics.final_step_norm,
            iterations=result.diagnostics.iterations,
            residual_evaluations=result.diagnostics.residual_evaluations,
            jvp_evaluations=result.diagnostics.jvp_evaluations,
            vjp_evaluations=result.diagnostics.vjp_evaluations,
            jacobian_preparations=result.diagnostics.jacobian_preparations,
            linear_solves=result.diagnostics.linear_solves,
            linear_iterations=result.diagnostics.linear_iterations,
            accepted_steps=result.diagnostics.accepted_steps,
            rejected_steps=result.diagnostics.rejected_steps,
            domain_failures=result.diagnostics.domain_failures,
            nonfinite_trials=result.diagnostics.nonfinite_trials,
            acceleration_restarts=result.diagnostics.acceleration_restarts,
            setup_refreshes=result.diagnostics.setup_refreshes,
            numeric_refreshes=result.diagnostics.numeric_refreshes,
            final_forcing=result.diagnostics.final_forcing,
            final_trust_radius=result.diagnostics.final_trust_radius,
            final_linear_status=result.diagnostics.final_linear_status,
            final_linear_rank=result.diagnostics.final_linear_rank,
            final_linear_condition_estimate=(
                result.diagnostics.final_linear_condition_estimate
            ),
            final_linear_residual_norm=(result.diagnostics.final_linear_residual_norm),
            final_linear_converged=result.diagnostics.final_linear_converged,
            counts_complete=False,
        )
        provenance = NonlinearProvenance(
            problem_id=problem_.problem_id,
            method_id=self.method_id,
            derivative_id="aspin-local-inverse-jacobians",
            globalization_id=result.provenance.globalization_id,
            linear_plan_id=result.provenance.linear_plan_id,
            notes=(
                f"subdomains={len(self.schwarz.subdomains)};"
                "prepared local linear solves reused within each ASPIN Jacobian"
            ),
        )
        return NonlinearResult(
            state=result.state,
            residual=result.residual,
            auxiliary=result.auxiliary,
            status=result.status,
            diagnostics=diagnostics,
            provenance=provenance,
            transformation_evidence=result.transformation_evidence,
            attempts=result.attempts,
        )


def _aspin_operator(
    problem: NonlinearSystemProblem,
    schwarz: NonlinearAdditiveSchwarz,
    prepared: PreparedNonlinearUpdate,
    state: PyTree[Any],
    args: Any,
    local_linear_policy: LinearSolvePolicy,
    /,
) -> FunctionLinearOperator:
    if problem.state_space is None or problem.residual_space is None:
        raise ValueError("ASPIN operator requires bound physical spaces.")
    state_ = problem.state_space.validate(state)
    local_data = []
    for child, subdomain in zip(
        prepared.internal_state,
        schwarz.subdomains,
        strict=True,
    ):
        local_state = subdomain.state_space.validate(subdomain.restrict_state(state_))
        local_problem = subdomain.local_problem()
        refreshed = refresh_nonlinear_update(
            child,
            local_problem,
            local_state,
            args=(state_, args),
        )
        local_result, _ = apply_prepared_nonlinear_update(
            refreshed,
            local_state,
            args=(state_, args),
        )
        point = subdomain.state_space.validate(local_result.state)
        coordinates = subdomain.state_space.flatten(point)

        def local_coordinates_residual(
            value, *, subdomain=subdomain, local_problem=local_problem
        ):
            local_value = subdomain.state_space.unflatten(value)
            residual = local_problem.residual(local_value, (state_, args))
            return subdomain.residual_space.flatten(residual)

        matrix = jax.jacfwd(local_coordinates_residual)(coordinates)
        local_operator = DenseLinearOperator(
            matrix,
            operator_id=f"aspin-local-jacobian/{subdomain.subdomain_id}",
        )
        local_linear = prepare_linear_solve(
            LinearSystem(
                local_operator,
                problem_id=f"aspin-local-system/{subdomain.subdomain_id}",
            ),
            local_linear_policy,
        )
        local_data.append((subdomain, local_linear, local_result.applied))

    def action(direction):
        direction_ = problem.state_space.validate(direction)
        _, global_action = jax.jvp(
            lambda value: problem.residual(value, args),
            (state_,),
            (direction_,),
        )
        result = problem.state_space.zeros()
        for subdomain, local_linear, local_applied in local_data:
            restricted = subdomain.residual_space.validate(
                subdomain.restrict_residual(global_action)
            )
            linear_result = solve_linear_system(
                local_linear,
                subdomain.residual_space.flatten(restricted),
            )
            local_coordinates = jnp.where(
                local_applied & linear_result.diagnostics.converged,
                linear_result.value,
                jnp.full_like(linear_result.value, jnp.nan),
            )
            local_correction = subdomain.state_space.unflatten(local_coordinates)
            prolonged = problem.state_space.validate(
                subdomain.prolong_correction(local_correction)
            )
            result = jax.tree.map(
                lambda value, correction, weight=subdomain.weight: (
                    value + weight * correction
                ),
                result,
                prolonged,
            )
        return result

    return FunctionLinearOperator(
        action,
        source=problem.state_space,
        target=problem.state_space,
        properties=OperatorProperties(),
        operator_id=f"aspin-jacobian/{schwarz.update_id}",
        closure_convert=False,
    )


__all__ = ["ASPIN"]
