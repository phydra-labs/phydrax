#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..optim import FiniteAxis, FiniteExhaustiveSearch, FiniteProductSpace
from ..optim._finite import _exhaustive_minimum
from ._global_search import _evaluate_control
from ._parameterization import AbstractControlParameterization
from ._problem import ControlProblem
from ._trajectory import ControlResult, ControlTrajectory


_CONTROL_CANDIDATE_METHOD_ID = "finite-exhaustive-control-candidate-search-v1"
_DEFAULT_FINITE_SEARCH = FiniteExhaustiveSearch()


class _ControlCandidateEvaluator(StrictModule):
    problem: ControlProblem
    parameterization: AbstractControlParameterization
    solver_options: dict[str, Any]

    def __init__(
        self,
        problem: ControlProblem,
        parameterization: AbstractControlParameterization,
        solver_options: dict[str, Any],
        /,
    ):
        self.problem = problem
        self.parameterization = parameterization
        self.solver_options = solver_options

    def __call__(self, coefficients: Array, /) -> tuple[Array, Array]:
        result = _evaluate_control(
            self.problem,
            self.parameterization,
            coefficients,
            self.solver_options,
        )
        objective = jnp.sum(result.sampled_loss.total)
        valid = (
            jnp.all(result.successful)
            & jnp.all(result.feasibility.feasible)
            & jnp.isfinite(objective)
        )
        return objective, valid


class ControlCandidateSearchResult(StrictModule):
    """Exact finite control-catalog minimum and rollout evidence."""

    problem: ControlProblem
    parameterization: AbstractControlParameterization
    evaluation: ControlResult | None
    coefficients: Array | None
    objective: Array
    search: FiniteExhaustiveSearch
    valid: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    flat_index: int = eqx.field(static=True)
    product_index: tuple[int, ...] = eqx.field(static=True)
    axis_paths: tuple[str, ...] = eqx.field(static=True)
    product_shape: tuple[int, ...] = eqx.field(static=True)
    candidate_count: int = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    valid_evaluations: int = eqx.field(static=True)
    invalid_candidates: int = eqx.field(static=True)
    winner_evaluations: int = eqx.field(static=True)
    effective_batch_size: int = eqx.field(static=True)
    candidate_signature: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    control_id: str | None = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    parameter_shape: tuple[int, ...] = eqx.field(static=True)
    coefficient_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: ControlProblem,
        parameterization: AbstractControlParameterization,
        evaluation: ControlResult | None,
        coefficients: Array | None,
        objective: ArrayLike,
        search: FiniteExhaustiveSearch,
        valid: bool,
        flat_index: int,
        product_index: tuple[int, ...],
        axis_paths: tuple[str, ...],
        product_shape: tuple[int, ...],
        candidate_count: int,
        objective_evaluations: int,
        invalid_candidates: int,
        winner_evaluations: int,
        effective_batch_size: int,
        candidate_signature: str,
    ):
        valid_ = bool(valid)
        complete_winner = evaluation is not None and coefficients is not None
        if valid_ != complete_winner:
            raise ValueError(
                "A valid control candidate result requires coefficients and evaluation."
            )
        evaluations = int(objective_evaluations)
        invalid = int(invalid_candidates)
        winner_count = int(winner_evaluations)
        if evaluations != int(candidate_count):
            raise ValueError(
                "Control candidate evaluation count must equal candidate_count."
            )
        if invalid < 0 or invalid > evaluations:
            raise ValueError("invalid_candidates must lie within the evaluation count.")
        if winner_count != int(valid_):
            raise ValueError("winner_evaluations must be one exactly when valid.")

        if valid_:
            if flat_index < 0 or any(index < 0 for index in product_index):
                raise ValueError("Valid control candidate indices must be nonnegative.")
            coefficients_ = jnp.asarray(coefficients)
            objective_ = jnp.asarray(objective, dtype=float).reshape(())
            termination_reason = "finite_minimum"
            assert evaluation is not None
            control_id = evaluation.trajectory.control_id
        else:
            if flat_index != -1 or any(index != -1 for index in product_index):
                raise ValueError(
                    "Invalid control candidate indices must use -1 sentinels."
                )
            coefficients_ = None
            objective_ = jnp.asarray(jnp.nan, dtype=float)
            termination_reason = "no_finite_candidates"
            control_id = None

        self.problem = problem
        self.parameterization = parameterization
        self.evaluation = evaluation
        self.coefficients = coefficients_
        self.objective = objective_
        self.search = search
        self.valid = valid_
        self.termination_reason = termination_reason
        self.flat_index = int(flat_index)
        self.product_index = tuple(int(index) for index in product_index)
        self.axis_paths = tuple(str(path) for path in axis_paths)
        self.product_shape = tuple(int(size) for size in product_shape)
        self.candidate_count = int(candidate_count)
        self.objective_evaluations = evaluations
        self.valid_evaluations = evaluations - invalid
        self.invalid_candidates = invalid
        self.winner_evaluations = winner_count
        self.effective_batch_size = int(effective_batch_size)
        self.candidate_signature = str(candidate_signature)
        self.problem_id = problem.problem_id
        self.dynamics_id = problem.dynamics.dynamics_id
        self.time_id = problem.time_grid.time_id
        self.control_id = control_id
        self.parameterization_id = parameterization.parameterization_id
        self.approximation_id = parameterization.approximation_id
        self.result_id = (
            f"control-candidate-search:{problem.problem_id}:"
            f"{parameterization.parameterization_id}:{candidate_signature}"
        )
        self.method_id = _CONTROL_CANDIDATE_METHOD_ID
        self.control_shape = parameterization.control_shape
        self.case_shape = problem.case_shape
        self.parameter_shape = parameterization.parameter_shape
        self.coefficient_shape = problem.case_shape + parameterization.parameter_shape

    @property
    def total_control_evaluations(self) -> int:
        """Return search evaluations plus full winner reconstruction."""
        return self.objective_evaluations + self.winner_evaluations

    @property
    def successful(self) -> Array:
        """Whether the selected candidate has a valid feasible rollout."""
        if self.evaluation is None:
            return jnp.asarray(False)
        return (
            jnp.all(self.evaluation.successful)
            & jnp.all(self.evaluation.feasibility.feasible)
            & jnp.isfinite(self.objective)
        )

    @property
    def trajectory(self) -> ControlTrajectory:
        """Return the selected trajectory or reject an all-invalid search."""
        if self.evaluation is None:
            raise RuntimeError("Control candidate search has no valid trajectory.")
        return self.evaluation.trajectory

    @property
    def controls(self) -> Array:
        """Return the selected sampled controls."""
        return self.trajectory.controls


def _validate_control_candidate_space(
    problem: ControlProblem,
    parameterization: AbstractControlParameterization,
    candidates: FiniteProductSpace,
    /,
) -> None:
    if parameterization.control_shape != problem.control_shape:
        raise ValueError(
            "Control parameterization control_shape does not match the problem."
        )
    candidate_spec = candidates.point_spec()
    if not isinstance(candidate_spec, jax.ShapeDtypeStruct):
        raise ValueError(
            "Control candidate points must be one coefficient array; use one "
            "correlated FiniteAxis catalog."
        )
    expected_shape = problem.case_shape + parameterization.parameter_shape
    if tuple(candidate_spec.shape) != expected_shape:
        raise ValueError(
            f"Control candidate points must have coefficient shape {expected_shape}, "
            f"got {candidate_spec.shape}."
        )
    if not np.issubdtype(np.dtype(candidate_spec.dtype), np.floating):
        raise TypeError("Control candidate coefficients must be real floating-point.")

    axis_blocks = jax.tree_util.tree_leaves(
        candidates.axes,
        is_leaf=lambda value: isinstance(value, FiniteAxis),
    )
    for axis in axis_blocks:
        for values in jax.tree_util.tree_leaves(axis.values):
            if not jnp.issubdtype(values.dtype, jnp.floating):
                raise TypeError(
                    "Control candidate coefficients must be real floating-point."
                )
            if bool(jnp.any(~jnp.isfinite(values))):
                raise ValueError("Control candidate coefficients must be finite.")


def search_control_candidates(
    problem: ControlProblem,
    parameterization: AbstractControlParameterization,
    candidates: FiniteProductSpace,
    /,
    *,
    search: FiniteExhaustiveSearch = _DEFAULT_FINITE_SEARCH,
    **solver_options: Any,
) -> ControlCandidateSearchResult:
    """Find the exact feasible minimum over a declared control catalog."""
    if not isinstance(problem, ControlProblem):
        raise TypeError("problem must be a ControlProblem.")
    if not isinstance(parameterization, AbstractControlParameterization):
        raise TypeError(
            "parameterization must implement AbstractControlParameterization."
        )
    if not isinstance(candidates, FiniteProductSpace):
        raise TypeError("candidates must be a FiniteProductSpace.")
    if not isinstance(search, FiniteExhaustiveSearch):
        raise TypeError("search must be a FiniteExhaustiveSearch.")
    _validate_control_candidate_space(problem, parameterization, candidates)

    options = dict(solver_options)
    evidence = _exhaustive_minimum(
        _ControlCandidateEvaluator(problem, parameterization, options),
        candidates,
        search,
    )
    valid = bool(evidence.valid)
    flat_index = int(evidence.flat_index) if valid else -1
    product_index = (
        tuple(int(index) for index in evidence.product_index)
        if valid
        else (-1,) * len(candidates.product_shape)
    )
    coefficients = jax.lax.stop_gradient(candidates.take(flat_index)) if valid else None
    evaluation = (
        _evaluate_control(problem, parameterization, coefficients, options)
        if coefficients is not None
        else None
    )
    signature = candidates.signature()

    return ControlCandidateSearchResult(
        problem=problem,
        parameterization=parameterization,
        evaluation=evaluation,
        coefficients=coefficients,
        objective=evidence.minimum,
        search=search,
        valid=valid,
        flat_index=flat_index,
        product_index=product_index,
        axis_paths=candidates.axis_paths,
        product_shape=candidates.product_shape,
        candidate_count=candidates.size,
        objective_evaluations=int(evidence.attempted_evaluations),
        invalid_candidates=int(evidence.invalid_evaluations),
        winner_evaluations=int(valid),
        effective_batch_size=search.effective_batch_size(candidates.size),
        candidate_signature=signature,
    )


__all__ = ["ControlCandidateSearchResult", "search_control_candidates"]
