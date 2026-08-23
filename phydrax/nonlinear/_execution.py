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
from ._scaling import NonlinearPrecisionPolicy
from ._types import (
    AbstractNonlinearMethod,
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)


NormReduction: TypeAlias = Literal["local-l2", "global-l2"]


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
        model_initial = self.precision.model(initial_state)

        def model_residual(state, current_args):
            residual, auxiliary = problem.evaluate(state, current_args)
            return self.precision.model(residual), auxiliary

        model_problem = NonlinearSystemProblem(
            model_residual,
            has_aux=True,
            problem_id=f"{problem.problem_id}/mixed-precision",
        )
        result = method.solve(
            model_problem,
            model_initial,
            termination=termination,
            args=args,
        )
        physical_state = self.precision.certificate(result.state)
        physical_residual, physical_auxiliary = problem.evaluate(
            physical_state,
            args,
        )
        initial_residual, _ = problem.evaluate(
            self.precision.certificate(initial_state),
            args,
        )
        physical_norm = jnp.sqrt(
            sum(
                jnp.real(jnp.vdot(value, value))
                for value in jax.tree.leaves(physical_residual)
            )
        )
        initial_norm = jnp.sqrt(
            sum(
                jnp.real(jnp.vdot(value, value))
                for value in jax.tree.leaves(initial_residual)
            )
        )
        finite = tree_allfinite(physical_state) & tree_allfinite(physical_residual)
        valid = problem.valid(
            physical_state,
            physical_residual,
            physical_auxiliary,
            args,
        )
        certified = (
            finite
            & valid
            & (physical_norm <= termination.residual_threshold(initial_norm))
        )
        status = jnp.where(
            result.successful & ~certified,
            int(NonlinearStatus.TRANSFORMATION_CERTIFICATION_FAILED),
            result.status,
        ).astype(jnp.int32)
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=initial_norm,
            final_residual_norm=physical_norm,
            final_step_norm=result.diagnostics.final_step_norm,
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
            notes=(
                f"model={self.precision.model_dtype};"
                f"direction={self.precision.direction_dtype};"
                f"certificate={self.precision.certificate_dtype}"
            ),
        )
        return NonlinearResult(
            state=physical_state,
            residual=physical_residual,
            auxiliary=physical_auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=provenance,
            attempts=result.attempts,
        )


__all__ = [
    "MixedPrecisionRootExecution",
    "NormReduction",
    "ShardedNonlinearPolicy",
]
