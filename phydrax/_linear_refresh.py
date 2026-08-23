#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._strict import StrictModule
from .linalg import (
    AbstractLinearProblem,
    AbstractPreconditioner,
    LinearSolvePolicy,
    LinearSolveTemplate,
    prepare as prepare_linear,
    PreparedLinearSolve,
    PreparedPreconditioner,
    refresh as refresh_linear,
)


class LinearRefreshState(StrictModule):
    """Problem-free state for reusing one symbolic linear-solve template."""

    template_arrays: LinearSolveTemplate
    template_static: LinearSolveTemplate = eqx.field(static=True)
    preconditioner_arrays: AbstractPreconditioner | None
    preconditioner_static: AbstractPreconditioner | None = eqx.field(static=True)
    numeric_version: Array
    preconditioner_numeric_version: Array
    preconditioner_built_numeric_version: Array
    preconditioner_refresh_kind: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedLinearSolve,
        /,
        *,
        template: LinearSolveTemplate | None = None,
        preconditioner_refresh_kind: str | None = None,
    ):
        if not isinstance(prepared, PreparedLinearSolve):
            raise TypeError("prepared must be a PreparedLinearSolve.")
        template_ = prepared.template if template is None else template
        if not isinstance(template_, LinearSolveTemplate):
            raise TypeError("template must be a LinearSolveTemplate or None.")
        if template_.template_id != prepared.template.template_id:
            raise ValueError("Refresh state must preserve its symbolic template.")
        previous = prepared.preconditioning_state
        template_arrays, template_static = eqx.partition(template_, eqx.is_array)
        action = None if previous is None else previous.action
        preconditioner_arrays, preconditioner_static = eqx.partition(
            action,
            eqx.is_array,
        )
        self.template_arrays = template_arrays
        self.template_static = template_static
        self.preconditioner_arrays = preconditioner_arrays
        self.preconditioner_static = preconditioner_static
        self.numeric_version = jnp.asarray(prepared.numeric_version, dtype=jnp.int32)
        self.preconditioner_numeric_version = jnp.asarray(
            -1 if previous is None else previous.numeric_version,
            dtype=jnp.int32,
        )
        self.preconditioner_built_numeric_version = jnp.asarray(
            -1 if previous is None else previous.built_numeric_version,
            dtype=jnp.int32,
        )
        self.preconditioner_refresh_kind = (
            ("none" if previous is None else previous.refresh_kind)
            if preconditioner_refresh_kind is None
            else str(preconditioner_refresh_kind)
        )

    @property
    def template(self) -> LinearSolveTemplate:
        return eqx.combine(self.template_arrays, self.template_static)

    @property
    def preconditioner(self) -> AbstractPreconditioner | None:
        return eqx.combine(
            self.preconditioner_arrays,
            self.preconditioner_static,
        )

    def _previous_preconditioner(
        self,
        problem: AbstractLinearProblem,
        /,
    ) -> PreparedPreconditioner | None:
        plan = self.template.plan.preconditioner_plan
        if plan is None:
            return None
        if self.preconditioner is None:
            raise ValueError("Prepared refresh state is missing its preconditioner.")
        policy = plan.policy
        setup_operator = (
            None
            if policy.preconditioner is not None
            else policy.resolve_setup_operator(problem.operator)
        )
        return PreparedPreconditioner(
            self.preconditioner,
            setup_operator,
            plan,
            numeric_version=self.preconditioner_numeric_version,
            built_numeric_version=self.preconditioner_built_numeric_version,
            refresh_kind=self.preconditioner_refresh_kind,
        )

    def refresh(
        self,
        problem: AbstractLinearProblem,
        /,
    ) -> tuple[PreparedLinearSolve, "LinearRefreshState"]:
        if not isinstance(problem, AbstractLinearProblem):
            raise TypeError("problem must be an AbstractLinearProblem.")
        seed = PreparedLinearSolve(
            problem,
            self.template,
            None,
            preconditioning_state=self._previous_preconditioner(problem),
            numeric_version=self.numeric_version,
        )
        prepared = refresh_linear(seed, problem)
        return prepared, LinearRefreshState(
            prepared,
            template=self.template,
            preconditioner_refresh_kind=self.preconditioner_refresh_kind,
        )


def prepare_refresh_state(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy,
    /,
) -> tuple[PreparedLinearSolve, LinearRefreshState]:
    """Prepare one numerical solve and retain only loop-safe refresh state."""
    prepared = prepare_linear(problem, policy)
    return prepared, LinearRefreshState(prepared)


__all__ = ["LinearRefreshState", "prepare_refresh_state"]
