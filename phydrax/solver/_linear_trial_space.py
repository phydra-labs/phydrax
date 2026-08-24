#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._trainable import combine_trainable, partition_trainable
from ..domain._model_function import ConcatenatedModelEvaluator
from ..equations.trefftz import trial_space_certificate
from ..integration import FixedIntegration
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveResult,
    solve as solve_linear,
)
from ..terms import ResidualPenalty
from ._kfac_problem import (
    frozen_term_residual_vector,
    materialize_frozen_residual_terms,
)


class LinearTrialSpaceResult(StrictModule):
    """Updated functional solver plus affine lowering and linear-solve evidence."""

    solver: Any
    linear_result: LinearSolveResult
    initial_residual_norm: Array
    final_residual_norm: Array
    affine_audit_residual: Array
    affine_audit_tolerance: Array
    coefficient_count: int = eqx.field(static=True)
    residual_count: int = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        *,
        solver: Any,
        linear_result: LinearSolveResult,
        initial_residual_norm: Array,
        final_residual_norm: Array,
        affine_audit_residual: Array,
        affine_audit_tolerance: Array,
        coefficient_count: int,
        residual_count: int,
    ):
        if not isinstance(linear_result, LinearSolveResult):
            raise TypeError("linear_result must be a LinearSolveResult.")
        affine_residual = jnp.asarray(affine_audit_residual)
        affine_tolerance = jnp.asarray(affine_audit_tolerance)
        self.solver = solver
        self.linear_result = linear_result
        self.initial_residual_norm = jnp.asarray(initial_residual_norm)
        self.final_residual_norm = jnp.asarray(final_residual_norm)
        self.affine_audit_residual = affine_residual
        self.affine_audit_tolerance = affine_tolerance
        self.coefficient_count = int(coefficient_count)
        self.residual_count = int(residual_count)
        self.valid = (
            linear_result.successful
            & linear_result.diagnostics.finite
            & (affine_residual <= affine_tolerance)
            & jnp.isfinite(self.final_residual_norm)
        )


def _validate_solver(solver) -> None:
    from ._functional_solver import FunctionalSolver

    if not isinstance(solver, FunctionalSolver):
        raise TypeError("solve_linear_trial_space requires a FunctionalSolver.")
    if solver.enforcement is not None:
        raise ValueError("Linear trial-space solves do not accept hard enforcement.")
    if not solver.terms:
        raise ValueError("Linear trial-space solves require at least one training term.")
    for term in solver.terms:
        if not isinstance(term, ResidualPenalty):
            raise TypeError("Linear trial-space solves require ResidualPenalty terms only.")
        if not isinstance(term.source, FixedIntegration):
            raise TypeError(
                "Linear trial-space solves require fixed integration realizations."
            )
    for name, field in solver.functions.items():
        certificate = trial_space_certificate(field)
        if not certificate.linear_in_coefficients:
            raise TypeError(
                f"Field {name!r} is not linear in its declared coefficients."
            )
        if not isinstance(field.func, ConcatenatedModelEvaluator):
            raise TypeError(
                f"Field {name!r} must bind its certified model directly."
            )


def solve_linear_trial_space(
    solver,
    /,
    *,
    linear: LinearSolvePolicy | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    affine_tolerance: float | None = None,
) -> LinearTrialSpaceResult:
    """Assemble and solve one fixed affine boundary-residual problem."""

    _validate_solver(solver)
    policy = LinearSolvePolicy(DenseSVD()) if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be a LinearSolvePolicy or None.")
    if affine_tolerance is not None:
        affine_tolerance_ = float(affine_tolerance)
        if not math.isfinite(affine_tolerance_) or affine_tolerance_ < 0.0:
            raise ValueError("affine_tolerance must be finite and nonnegative.")

    params, non_trainable = partition_trainable(solver.functions)
    flat_params, unravel = ravel_pytree(params)
    if flat_params.size == 0:
        raise ValueError("Linear trial-space solver found no trainable coefficients.")
    if jnp.iscomplexobj(flat_params):
        raise TypeError("Linear trial-space coefficients must be real-valued.")

    evaluation_key, sampling_key = jr.split(key)
    prepared = solver.objective.prepare_training(
        range(len(solver.terms)),
        scale=1.0,
        evaluation_key=evaluation_key,
        sampling_key=sampling_key,
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )
    frozen_terms = materialize_frozen_residual_terms(prepared)
    if len(frozen_terms) != len(solver.terms):
        raise RuntimeError("Failed to materialize every linear trial-space residual term.")

    def residual_vector(flat):
        current = unravel(flat)
        pieces = tuple(
            frozen_term_residual_vector(
                current,
                non_trainable,
                solver,
                term,
                iter_=prepared.iteration,
            )
            for term in frozen_terms
        )
        if not pieces:
            return jnp.zeros((0,), dtype=flat.dtype)
        return jnp.concatenate(pieces, axis=0)

    zero = jnp.zeros_like(flat_params)
    offset = residual_vector(zero)
    if offset.size == 0:
        raise ValueError("Linear trial-space terms produced no residual roots.")
    design = jax.jacfwd(residual_vector)(zero)
    if design.shape != (offset.size, flat_params.size):
        raise RuntimeError("Linear trial-space design matrix has an invalid shape.")
    if not bool(jnp.all(jnp.isfinite(design))) or not bool(jnp.all(jnp.isfinite(offset))):
        raise ValueError("Linear trial-space design matrix and offset must be finite.")

    direction = jnp.linspace(-0.75, 0.75, int(flat_params.size), dtype=flat_params.dtype)
    first_actual = residual_vector(direction)
    second_actual = residual_vector(-0.5 * direction)
    first_error = jnp.max(jnp.abs(first_actual - (design @ direction + offset)))
    second_error = jnp.max(
        jnp.abs(second_actual - (design @ (-0.5 * direction) + offset))
    )
    audit_residual = jnp.maximum(first_error, second_error)
    audit_scale = jnp.maximum(
        jnp.max(
            jnp.abs(
                jnp.concatenate((first_actual, second_actual, offset), axis=0)
            )
        ),
        1.0,
    )
    if affine_tolerance is None:
        epsilon = np.finfo(np.dtype(flat_params.dtype)).eps
        audit_tolerance = (
            4096.0
            * epsilon
            * max(int(flat_params.size), int(offset.size), 1)
            * audit_scale
        )
    else:
        audit_tolerance = jnp.asarray(affine_tolerance_, dtype=flat_params.dtype)
    if not bool(jnp.isfinite(audit_residual)) or bool(audit_residual > audit_tolerance):
        raise ValueError(
            "Training residual is not affine in the trial-space coefficients: "
            f"audit={float(audit_residual):.3e}, "
            f"tolerance={float(audit_tolerance):.3e}."
        )

    problem = LeastSquaresProblem(
        DenseLinearOperator(design),
        problem_id="linear-trefftz-boundary-fit",
    )
    linear_result = solve_linear(problem, -offset, policy=policy)
    solved_params = unravel(jnp.asarray(linear_result.value))
    updated_functions = combine_trainable(solved_params, non_trainable)
    updated_solver = eqx.tree_at(lambda value: value.functions, solver, updated_functions)
    initial_residual = jnp.linalg.norm(residual_vector(flat_params))
    final_residual = jnp.linalg.norm(residual_vector(jnp.asarray(linear_result.value)))
    return LinearTrialSpaceResult(
        solver=updated_solver,
        linear_result=linear_result,
        initial_residual_norm=initial_residual,
        final_residual_norm=final_residual,
        affine_audit_residual=audit_residual,
        affine_audit_tolerance=audit_tolerance,
        coefficient_count=int(flat_params.size),
        residual_count=int(offset.size),
    )


__all__ = ["LinearTrialSpaceResult", "solve_linear_trial_space"]
