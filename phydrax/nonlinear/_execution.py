#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._tree_math import tree_allfinite
from ._newton import NewtonKrylov, NewtonTrustRegion
from ._precision import NonlinearPrecisionPolicy
from ._types import (
    AbstractNonlinearMethod,
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    NonlinearTransformationEvidence,
)


NormReduction: TypeAlias = Literal["local-l2", "global-l2"]


def _precision_tree_norm(
    value: PyTree[Any],
    precision: NonlinearPrecisionPolicy,
    /,
) -> Array:
    terms = tuple(
        jnp.sum(jnp.real(jnp.conj(leaf_) * leaf_))
        for leaf in jax.tree.leaves(value)
        if (leaf_ := precision.accumulation(leaf)).size
    )
    if not terms:
        raise ValueError("A nonlinear residual requires at least one non-empty leaf.")
    squared = terms[0]
    for term in terms[1:]:
        squared = squared + term
    return precision.decision(jnp.sqrt(jnp.maximum(squared, 0.0)))


class ShardedNonlinearPolicy(StrictModule):
    """Explicit state/residual sharding and norm-reduction semantics."""

    state_sharding: Any
    residual_sharding: Any
    axis_name: str | None = eqx.field(static=True)
    norm_reduction: NormReduction = eqx.field(static=True)
    replicated_status: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_sharding: Any,
        residual_sharding: Any,
        axis_name: str | None = None,
        norm_reduction: NormReduction = "global-l2",
        replicated_status: bool = True,
    ):
        if not isinstance(state_sharding, jax.sharding.Sharding):
            raise TypeError("state_sharding must implement jax.sharding.Sharding.")
        if not isinstance(residual_sharding, jax.sharding.Sharding):
            raise TypeError("residual_sharding must implement jax.sharding.Sharding.")
        if norm_reduction not in ("local-l2", "global-l2"):
            raise ValueError("Unknown norm_reduction.")
        self.state_sharding = state_sharding
        self.residual_sharding = residual_sharding
        self.axis_name = None if axis_name is None else str(axis_name)
        self.norm_reduction = norm_reduction
        self.replicated_status = bool(replicated_status)

    def place_state(self, state: PyTree[Any], /):
        return jax.tree.map(
            lambda value: jax.device_put(value, self.state_sharding),
            state,
        )

    def place_residual(self, residual: PyTree[Any], /):
        return jax.tree.map(
            lambda value: jax.device_put(value, self.residual_sharding),
            residual,
        )

    def residual_norm(self, residual: PyTree[Any], /) -> Array:
        local_squared = sum(
            jnp.real(jnp.vdot(value, value)) for value in jax.tree.leaves(residual)
        )
        if self.norm_reduction == "global-l2" and self.axis_name is not None:
            squared = jax.lax.psum(local_squared, self.axis_name)
        else:
            squared = local_squared
        return jnp.sqrt(jnp.maximum(squared, 0.0))


class MixedPrecisionRootExecution(StrictModule):
    """Model-precision solve followed by higher-precision physical certification."""

    precision: NonlinearPrecisionPolicy

    def __init__(self, precision: NonlinearPrecisionPolicy | None = None, /):
        policy = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(policy, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.precision = policy

    def solve(
        self,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any],
        method: AbstractNonlinearMethod,
        termination: NonlinearTermination,
        /,
        *,
        args: Any = None,
    ) -> NonlinearResult:
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be NonlinearSystemProblem.")
        if not isinstance(method, AbstractNonlinearMethod):
            raise TypeError("method must be AbstractNonlinearMethod.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination.")
        self.precision.validate_tolerance(termination.absolute_residual)
        model_initial = self.precision.state(initial_state)

        def model_residual(state, current_args):
            residual, auxiliary = problem.residual_function(state, current_args), None
            if problem.has_aux:
                residual, auxiliary = residual
            return self.precision.residual(residual), auxiliary

        def model_validity(state, residual, auxiliary, current_args):
            assert problem.validity_function is not None
            return problem.validity_function(
                state,
                residual,
                auxiliary,
                current_args,
            )

        model_problem = NonlinearSystemProblem(
            model_residual,
            has_aux=True,
            validity=(None if problem.validity_function is None else model_validity),
            problem_id=f"{problem.problem_id}/mixed-precision",
        )
        if isinstance(method, (NewtonKrylov, NewtonTrustRegion)):
            result = method.solve(
                model_problem,
                model_initial,
                termination=termination,
                args=args,
                precision=self.precision,
            )
        else:
            result = method.solve(
                model_problem,
                model_initial,
                termination=termination,
                args=args,
            )
        model_state = self.precision.state(result.state)
        model_residual_value = self.precision.residual(result.residual)
        physical_state = self.precision.certificate(result.state)
        physical_residual, physical_auxiliary = problem.evaluate(
            physical_state,
            args,
        )
        physical_residual = self.precision.certificate(physical_residual)
        initial_residual, _ = problem.evaluate(
            self.precision.certificate(initial_state),
            args,
        )
        initial_residual = self.precision.certificate(initial_residual)
        physical_norm = _precision_tree_norm(physical_residual, self.precision)
        initial_norm = _precision_tree_norm(initial_residual, self.precision)
        finite = tree_allfinite(physical_state) & tree_allfinite(physical_residual)
        valid = problem.valid(
            physical_state,
            physical_residual,
            physical_auxiliary,
            args,
        )
        threshold = self.precision.decision(termination.residual_threshold(initial_norm))
        certified = finite & valid & (physical_norm <= threshold)
        status = jnp.where(
            result.successful & ~certified,
            int(NonlinearStatus.TRANSFORMATION_CERTIFICATION_FAILED),
            result.status,
        ).astype(jnp.int32)
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=initial_norm,
            final_residual_norm=physical_norm,
            final_step_norm=self.precision.decision(result.diagnostics.final_step_norm),
            iterations=result.diagnostics.iterations,
            residual_evaluations=result.diagnostics.residual_evaluations + 2,
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
            counts_complete=result.diagnostics.counts_complete,
        )
        provenance = NonlinearProvenance(
            problem_id=problem.problem_id,
            method_id=result.provenance.method_id,
            derivative_id=result.provenance.derivative_id,
            globalization_id=result.provenance.globalization_id,
            linear_plan_id=result.provenance.linear_plan_id,
            precision_policy_id=self.precision.policy_id,
            notes=(
                f"model={self.precision.model_dtype};"
                f"direction={self.precision.direction_dtype};"
                f"certificate={self.precision.certificate_dtype}"
            ),
        )
        children = (
            {}
            if result.precision_evidence is None
            else {"model-solve": result.precision_evidence}
        )
        output_state = jax.tree.map(self.precision.output, physical_state)
        precision_evidence = self.precision.evidence_for(
            model_state,
            model_residual_value,
            children=children,
            output_value=output_state,
        )
        return NonlinearResult(
            state=output_state,
            residual=physical_residual,
            auxiliary=physical_auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=provenance,
            transformation_evidence=NonlinearTransformationEvidence(
                state=model_state,
                residual=model_residual_value,
                auxiliary=result.auxiliary,
            ),
            precision_evidence=precision_evidence,
            attempts=result.attempts,
        )


__all__ = [
    "MixedPrecisionRootExecution",
    "NormReduction",
    "ShardedNonlinearPolicy",
]
