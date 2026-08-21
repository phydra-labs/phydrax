#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._bounds import ProjectedLBFGS
from ._iterative._base import AbstractMinimizationMethod
from ._iterative._globalization import armijo_backtracking, ArmijoLineSearch
from ._iterative._types import (
    _tree_allfinite,
    _tree_norm,
    ConstrainedOptimalityCertificate,
    MinimizationProblem,
    MinimizationResult,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._newton_krylov import NewtonKrylov
from ._quadratic_program import (
    QP_SUCCESS,
    QuadraticProgram,
    solve_quadratic_program,
)


@dataclass(frozen=True, slots=True)
class _ConstraintLayout:
    """Fixed canonical ordering for nonlinear constraints and parameter bounds."""

    lower: Array
    upper: Array
    equality_indices: Array
    lower_indices: Array
    upper_indices: Array
    equality_sources: tuple[str, ...]
    inequality_sources: tuple[str, ...]


def _flat_constraint_values(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    /,
) -> Array:
    values = []
    for constraint in problem.constraints:
        value = constraint.value(parameters, args)
        flat, _ = ravel_pytree(value)
        values.append(flat)
    flat_parameters, _ = ravel_pytree(parameters)
    if problem.bounds is not None:
        values.append(flat_parameters)
    if not values:
        return jnp.empty((0,), dtype=flat_parameters.dtype)
    return jnp.concatenate(values)


def _constraint_layout(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    args: Any,
    /,
) -> _ConstraintLayout:
    def materialize_dynamic(bound, template, *, name):
        template_structure = jax.tree.structure(template)
        if jax.tree.structure(bound) == template_structure:
            bound_leaves = jax.tree.leaves(bound)
        else:
            scalar = jnp.asarray(bound)
            if scalar.shape != ():
                raise ValueError(
                    f"{name} must be scalar or have the constrained PyTree structure."
                )
            bound_leaves = [scalar] * template_structure.num_leaves
        arrays = [
            jnp.broadcast_to(
                jnp.asarray(bound_leaf, dtype=specification.dtype),
                specification.shape,
            ).reshape(-1)
            for bound_leaf, specification in zip(
                bound_leaves,
                jax.tree.leaves(template),
                strict=True,
            )
        ]
        if not arrays:
            raise ValueError("A constrained value must contain at least one array leaf.")
        return jnp.concatenate(arrays)

    def materialize_metadata(metadata, bound, template, *, name):
        if metadata is None:
            return None
        metadata_leaves = [
            np.asarray(values, dtype=np.dtype(dtype)).reshape(shape)
            for shape, dtype, values in metadata
        ]
        template_structure = jax.tree.structure(template)
        if jax.tree.structure(bound) == template_structure:
            selected = metadata_leaves
        else:
            if len(metadata_leaves) != 1 or metadata_leaves[0].shape != ():
                raise ValueError(
                    f"{name} must be scalar or have the constrained PyTree structure."
                )
            selected = metadata_leaves * template_structure.num_leaves
        arrays = [
            np.broadcast_to(
                bound_leaf.astype(np.dtype(specification.dtype), copy=False),
                specification.shape,
            ).reshape(-1)
            for bound_leaf, specification in zip(
                selected,
                jax.tree.leaves(template),
                strict=True,
            )
        ]
        return np.concatenate(arrays)

    lower_values = []
    upper_values = []
    equality_indices: list[int] = []
    lower_indices: list[int] = []
    upper_indices: list[int] = []
    equality_sources: list[str] = []
    lower_sources: list[str] = []
    upper_sources: list[str] = []
    offset = 0

    def append_bounds(
        lower,
        upper,
        lower_metadata,
        upper_metadata,
        template,
        segment_sources,
        *,
        lower_name,
        upper_name,
    ):
        nonlocal offset
        lower_flat = materialize_dynamic(lower, template, name=lower_name)
        upper_flat = materialize_dynamic(upper, template, name=upper_name)
        valid = jnp.all(lower_flat <= upper_flat)
        lower_flat = eqx.error_if(
            lower_flat,
            ~valid,
            f"{lower_name} must not exceed {upper_name}.",
        )
        lower_static = materialize_metadata(
            lower_metadata,
            lower,
            template,
            name=lower_name,
        )
        upper_static = materialize_metadata(
            upper_metadata,
            upper,
            template,
            name=upper_name,
        )
        size = int(lower_flat.size)
        if lower_static is not None and upper_static is not None:
            local_equal = (
                np.isfinite(lower_static)
                & np.isfinite(upper_static)
                & (lower_static == upper_static)
            )
        else:
            local_equal = np.zeros((size,), dtype=bool)
        local_lower = (
            np.ones((size,), dtype=bool)
            if lower_static is None
            else np.isfinite(lower_static)
        ) & ~local_equal
        local_upper = (
            np.ones((size,), dtype=bool)
            if upper_static is None
            else np.isfinite(upper_static)
        ) & ~local_equal
        for local_index in np.flatnonzero(local_equal):
            equality_indices.append(offset + int(local_index))
            equality_sources.append(f"{segment_sources[int(local_index)]}:equality")
        for local_index in np.flatnonzero(local_lower):
            lower_indices.append(offset + int(local_index))
            lower_sources.append(f"{segment_sources[int(local_index)]}:lower")
        for local_index in np.flatnonzero(local_upper):
            upper_indices.append(offset + int(local_index))
            upper_sources.append(f"{segment_sources[int(local_index)]}:upper")
        lower_values.append(lower_flat)
        upper_values.append(upper_flat)
        offset += size

    for constraint_index, constraint in enumerate(problem.constraints):
        value_shape = jax.eval_shape(
            lambda candidate: constraint.value(candidate, args),
            parameters,
        )
        coordinate_count = sum(
            int(specification.size) for specification in jax.tree.leaves(value_shape)
        )
        append_bounds(
            constraint.lower,
            constraint.upper,
            constraint._lower_metadata,
            constraint._upper_metadata,
            value_shape,
            tuple(
                f"constraint:{constraint_index}:{coordinate}"
                for coordinate in range(coordinate_count)
            ),
            lower_name="constraint lower",
            upper_name="constraint upper",
        )

    flat_parameters, _ = ravel_pytree(parameters)
    if problem.bounds is not None:
        parameter_shape = jax.tree.map(
            lambda parameter: jax.ShapeDtypeStruct(
                parameter.shape,
                parameter.dtype,
            ),
            parameters,
        )
        append_bounds(
            problem.bounds.lower,
            problem.bounds.upper,
            problem.bounds._lower_metadata,
            problem.bounds._upper_metadata,
            parameter_shape,
            tuple(f"bound:{coordinate}" for coordinate in range(flat_parameters.size)),
            lower_name="parameter lower",
            upper_name="parameter upper",
        )

    if not lower_values:
        empty = jnp.empty((0,), dtype=flat_parameters.dtype)
        empty_indices = jnp.empty((0,), dtype=jnp.int32)
        return _ConstraintLayout(
            empty,
            empty,
            empty_indices,
            empty_indices,
            empty_indices,
            (),
            (),
        )
    return _ConstraintLayout(
        lower=jnp.concatenate(lower_values).astype(flat_parameters.dtype),
        upper=jnp.concatenate(upper_values).astype(flat_parameters.dtype),
        equality_indices=jnp.asarray(equality_indices, dtype=jnp.int32),
        lower_indices=jnp.asarray(lower_indices, dtype=jnp.int32),
        upper_indices=jnp.asarray(upper_indices, dtype=jnp.int32),
        equality_sources=tuple(equality_sources),
        inequality_sources=tuple(lower_sources + upper_sources),
    )


def _canonical_constraints(
    problem: MinimizationProblem,
    layout: _ConstraintLayout,
    parameters: PyTree[Any],
    args: Any,
    /,
) -> tuple[Array, Array]:
    values = _flat_constraint_values(problem, parameters, args)
    equality = values[layout.equality_indices] - layout.lower[layout.equality_indices]
    lower_bounds = layout.lower[layout.lower_indices]
    upper_bounds = layout.upper[layout.upper_indices]
    lower = jnp.where(
        jnp.isfinite(lower_bounds),
        lower_bounds - values[layout.lower_indices],
        -jnp.ones_like(lower_bounds),
    )
    upper = jnp.where(
        jnp.isfinite(upper_bounds),
        values[layout.upper_indices] - upper_bounds,
        -jnp.ones_like(upper_bounds),
    )
    return equality, jnp.concatenate((lower, upper))


def _max_abs(value: Array, /) -> Array:
    if value.size == 0:
        return jnp.asarray(0.0, dtype=value.dtype)
    return jnp.max(jnp.abs(value))


def _max_positive(value: Array, /) -> Array:
    if value.size == 0:
        return jnp.asarray(0.0, dtype=value.dtype)
    return jnp.max(jnp.maximum(value, 0.0))


def _constraint_violation(equality: Array, inequality: Array, /) -> Array:
    return jnp.maximum(_max_abs(equality), _max_positive(inequality))


def _constraint_l1(equality: Array, inequality: Array, /) -> Array:
    return jnp.sum(jnp.abs(equality)) + jnp.sum(jnp.maximum(inequality, 0.0))


def _derivatives(
    problem: MinimizationProblem,
    layout: _ConstraintLayout,
    parameters: PyTree[Any],
    args: Any,
    /,
):
    flat_parameters, unravel = ravel_pytree(parameters)

    def flat_value(candidate):
        return problem.value(unravel(candidate), args)[0]

    def flat_constraints(candidate):
        return _canonical_constraints(problem, layout, unravel(candidate), args)

    value, gradient = jax.value_and_grad(flat_value)(flat_parameters)
    equality, inequality = flat_constraints(flat_parameters)
    equality_jacobian, inequality_jacobian = jax.jacrev(flat_constraints)(flat_parameters)
    return (
        flat_parameters,
        unravel,
        value,
        gradient,
        equality,
        inequality,
        equality_jacobian,
        inequality_jacobian,
    )


def _lagrangian_gradient(
    objective_gradient: Array,
    equality_jacobian: Array,
    inequality_jacobian: Array,
    equality_multipliers: Array,
    inequality_multipliers: Array,
    /,
) -> Array:
    return (
        objective_gradient
        + equality_jacobian.T @ equality_multipliers
        + inequality_jacobian.T @ inequality_multipliers
    )


def _stationarity_norm(lagrangian_gradient: Array, /) -> Array:
    return jnp.linalg.norm(lagrangian_gradient)


def _kkt_metrics(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    unravel,
    objective_gradient: Array,
    equality: Array,
    inequality: Array,
    equality_jacobian: Array,
    inequality_jacobian: Array,
    equality_multipliers: Array,
    inequality_multipliers: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    lagrangian_gradient = _lagrangian_gradient(
        objective_gradient,
        equality_jacobian,
        inequality_jacobian,
        equality_multipliers,
        inequality_multipliers,
    )
    stationarity = _stationarity_norm(lagrangian_gradient)
    primal = _constraint_violation(equality, inequality)
    if problem.bounds is not None:
        primal = jnp.maximum(primal, problem.bounds.violation(parameters))
    dual = jnp.maximum(
        stationarity,
        _max_positive(-inequality_multipliers),
    )
    complementarity = (
        _max_abs(inequality_multipliers * inequality)
        if inequality.size
        else jnp.asarray(0.0, dtype=objective_gradient.dtype)
    )
    optimality = jnp.maximum(jnp.maximum(primal, dual), complementarity)
    return primal, dual, complementarity, optimality


def _active_constraint_count(
    inequality: Array,
    /,
    *,
    tolerance: float,
) -> Array:
    return jnp.sum(inequality >= -tolerance, dtype=jnp.int32)


def _constraint_certificate(
    layout: _ConstraintLayout,
    unravel,
    lagrangian_gradient: Array,
    equality_multipliers: Array,
    inequality_multipliers: Array,
    inequality: Array,
    primal_feasibility: Array,
    dual_feasibility: Array,
    complementarity: Array,
    /,
    *,
    active_tolerance: float,
) -> ConstrainedOptimalityCertificate:
    slacks = jnp.maximum(-inequality, 0.0)
    return ConstrainedOptimalityCertificate(
        equality_multipliers=equality_multipliers,
        inequality_multipliers=inequality_multipliers,
        slacks=slacks,
        active_mask=slacks <= active_tolerance,
        stationarity_residual=unravel(lagrangian_gradient),
        primal_feasibility=primal_feasibility,
        dual_feasibility=dual_feasibility,
        complementarity=complementarity,
        equality_sources=layout.equality_sources,
        inequality_sources=layout.inequality_sources,
    )


class FilterGlobalization(StrictModule):
    """Objective-feasibility filter with margin-based dominance rejection."""

    objective_margin: float = eqx.field(static=True)
    violation_margin: float = eqx.field(static=True)
    correction_regularization: float = eqx.field(static=True)
    correction_limit: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        objective_margin: float = 1e-4,
        violation_margin: float = 1e-4,
        correction_regularization: float = 1e-10,
        correction_limit: float = 2.0,
    ):
        objective = float(objective_margin)
        violation = float(violation_margin)
        regularization = float(correction_regularization)
        limit = float(correction_limit)
        if (
            not isfinite(objective)
            or not isfinite(violation)
            or not 0.0 < objective < 1.0
            or not 0.0 < violation < 1.0
        ):
            raise ValueError("Filter margins must be finite and lie in (0, 1).")
        if not isfinite(regularization) or regularization < 0.0:
            raise ValueError("correction_regularization must be finite and non-negative.")
        if not isfinite(limit) or limit <= 0.0:
            raise ValueError("correction_limit must be finite and positive.")
        self.objective_margin = objective
        self.violation_margin = violation
        self.correction_regularization = regularization
        self.correction_limit = limit

    def acceptable(
        self,
        objective: Any,
        violation: Any,
        filter_objectives: Array,
        filter_violations: Array,
        filter_size: Any,
        /,
    ) -> Array:
        """Return whether a finite pair clears every active filter entry."""

        objective_ = jnp.asarray(objective)
        violation_ = jnp.asarray(violation)
        indices = jnp.arange(filter_objectives.size, dtype=jnp.int32)
        active = indices < jnp.asarray(filter_size, dtype=jnp.int32)
        clears_entry = (
            violation_ <= (1.0 - self.violation_margin) * filter_violations
        ) | (
            objective_
            <= filter_objectives
            - self.objective_margin * jnp.maximum(filter_violations, violation_)
        )
        return (
            jnp.isfinite(objective_)
            & jnp.isfinite(violation_)
            & jnp.all((~active) | clears_entry)
        )


class AugmentedLagrangian(AbstractMinimizationMethod):
    """Powell--Hestenes augmented Lagrangian for equality and inequality constraints."""

    inner_method: AbstractMinimizationMethod | None
    initial_penalty: float = eqx.field(static=True)
    penalty_increase: float = eqx.field(static=True)
    maximum_penalty: float = eqx.field(static=True)
    required_feasibility_reduction: float = eqx.field(static=True)
    maximum_outer_steps: int = eqx.field(static=True)
    inner_maximum_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        inner_method: AbstractMinimizationMethod | None = None,
        initial_penalty: float = 10.0,
        penalty_increase: float = 10.0,
        maximum_penalty: float = 1e10,
        required_feasibility_reduction: float = 0.25,
        maximum_outer_steps: int = 20,
        inner_maximum_steps: int = 100,
    ):
        penalty = float(initial_penalty)
        increase = float(penalty_increase)
        maximum = float(maximum_penalty)
        reduction = float(required_feasibility_reduction)
        outer_steps = int(maximum_outer_steps)
        inner_steps = int(inner_maximum_steps)
        if inner_method is not None and not isinstance(
            inner_method,
            AbstractMinimizationMethod,
        ):
            raise TypeError("inner_method must be an AbstractMinimizationMethod or None.")
        if not isfinite(penalty) or penalty <= 0.0:
            raise ValueError("initial_penalty must be positive and finite.")
        if not isfinite(increase) or increase <= 1.0:
            raise ValueError("penalty_increase must be finite and greater than one.")
        if not isfinite(maximum) or maximum < penalty:
            raise ValueError(
                "maximum_penalty must be finite and at least initial_penalty."
            )
        if not isfinite(reduction) or not 0.0 < reduction < 1.0:
            raise ValueError("required_feasibility_reduction must lie in (0, 1).")
        if outer_steps < 1 or inner_steps < 1:
            raise ValueError("Outer and inner step limits must be positive.")
        self.inner_method = inner_method
        self.initial_penalty = penalty
        self.penalty_increase = increase
        self.maximum_penalty = maximum
        self.required_feasibility_reduction = reduction
        self.maximum_outer_steps = outer_steps
        self.inner_maximum_steps = inner_steps

    @property
    def method_id(self) -> str:
        return "augmented-lagrangian"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=True,
            prepared_refresh=False,
            implicit_differentiation=True,
        )

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        return _solve_augmented_lagrangian(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


class SQP(AbstractMinimizationMethod):
    """Dense quasi-Newton SQP with convex QP subproblems and restoration."""

    line_search: ArmijoLineSearch
    filter_globalization: FilterGlobalization | None
    second_order_correction: bool = eqx.field(static=True)
    merit_penalty: float = eqx.field(static=True)
    hessian_scale: float = eqx.field(static=True)
    qp_tolerance: float = eqx.field(static=True)
    qp_maximum_steps: int = eqx.field(static=True)
    qp_regularization: float = eqx.field(static=True)
    max_dense_dimension: int = eqx.field(static=True)
    active_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        line_search: ArmijoLineSearch | None = None,
        filter_globalization: FilterGlobalization | None = None,
        second_order_correction: bool = True,
        merit_penalty: float = 10.0,
        hessian_scale: float = 1.0,
        qp_tolerance: float = 1e-8,
        qp_maximum_steps: int = 100,
        qp_regularization: float = 1e-8,
        max_dense_dimension: int = 512,
        active_tolerance: float = 1e-8,
    ):
        search = ArmijoLineSearch() if line_search is None else line_search
        scalars = tuple(
            float(value)
            for value in (
                merit_penalty,
                hessian_scale,
                qp_tolerance,
                qp_regularization,
                active_tolerance,
            )
        )
        qp_steps = int(qp_maximum_steps)
        dense_dimension = int(max_dense_dimension)
        if not isinstance(search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch or None.")
        if filter_globalization is not None and not isinstance(
            filter_globalization, FilterGlobalization
        ):
            raise TypeError("filter_globalization must be a FilterGlobalization or None.")
        if any(not isfinite(value) or value <= 0.0 for value in scalars[:3]):
            raise ValueError(
                "Merit, Hessian, and QP tolerances must be positive and finite."
            )
        if any(not isfinite(value) or value < 0.0 for value in scalars[3:]):
            raise ValueError("Regularization and active tolerance must be non-negative.")
        if qp_steps < 1 or dense_dimension < 1:
            raise ValueError("QP step and dense-dimension limits must be positive.")
        self.line_search = search
        self.filter_globalization = filter_globalization
        self.second_order_correction = bool(second_order_correction)
        (
            self.merit_penalty,
            self.hessian_scale,
            self.qp_tolerance,
            self.qp_regularization,
            self.active_tolerance,
        ) = scalars
        self.qp_maximum_steps = qp_steps
        self.max_dense_dimension = dense_dimension

    @property
    def method_id(self) -> str:
        return (
            "sqp/filter-soc"
            if self.filter_globalization is not None and self.second_order_correction
            else "sqp/filter"
            if self.filter_globalization is not None
            else "sqp/merit"
        )

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=True,
        )

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        return _solve_sqp(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


class _AugmentedLagrangianState(eqx.Module):
    """Array-only carry for the staged augmented-Lagrangian outer iteration."""

    outer: Array
    iterations: Array
    parameters: PyTree[Any]
    status: Array
    equality_multipliers: Array
    inequality_multipliers: Array
    penalty: Array
    previous_feasibility: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    residual_evaluations: Array
    jvp_evaluations: Array
    vjp_evaluations: Array
    hvp_evaluations: Array
    jacobian_evaluations: Array
    constraint_evaluations: Array
    accepted_steps: Array
    rejected_steps: Array
    linear_solves: Array
    linear_iterations: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    globalization_evaluations: Array
    direction_fallbacks: Array
    final_step_norm: Array
    initial_optimality: Array


def _solve_augmented_lagrangian(
    method: AugmentedLagrangian,
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> MinimizationResult:
    if not problem.constraints:
        raise ValueError("AugmentedLagrangian requires nonlinear constraints.")
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")
    parameters = initial_parameters
    if problem.bounds is not None:
        parameters = problem.bounds.project(parameters)
    initial_status = jnp.where(
        _tree_allfinite(parameters),
        int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.NONFINITE_INPUT),
    ).astype(jnp.int32)
    layout = _constraint_layout(problem, parameters, args)
    equality, inequality = _canonical_constraints(problem, layout, parameters, args)
    inner_method = method.inner_method
    if inner_method is None:
        inner_method = ProjectedLBFGS() if problem.bounds is not None else NewtonKrylov()
    maximum_outer_steps = min(method.maximum_outer_steps, termination.maximum_steps)
    inner_tolerance = max(
        termination.absolute_optimality,
        min(1e-2, 1.0 / method.maximum_penalty),
    )
    inner_termination = OptimizationTermination(
        absolute_optimality=inner_tolerance,
        relative_optimality=0.0,
        absolute_step=termination.absolute_step,
        relative_step=termination.relative_step,
        maximum_steps=method.inner_maximum_steps,
        maximum_evaluations=termination.maximum_evaluations,
    )

    def augmented_objective(candidate, augmented_args):
        (
            dynamic_args,
            equality_multipliers,
            inequality_multipliers,
            penalty,
        ) = augmented_args
        value = problem.value(candidate, dynamic_args)[0]
        equality_value, inequality_value = _canonical_constraints(
            problem,
            layout,
            candidate,
            dynamic_args,
        )
        shifted = jnp.maximum(
            0.0,
            inequality_multipliers + penalty * inequality_value,
        )
        return (
            value
            + jnp.vdot(equality_multipliers, equality_value).real
            + 0.5 * penalty * jnp.vdot(equality_value, equality_value).real
            + 0.5
            / penalty
            * (
                jnp.vdot(shifted, shifted).real
                - jnp.vdot(inequality_multipliers, inequality_multipliers).real
            )
        )

    inner_problem = MinimizationProblem(
        augmented_objective,
        bounds=problem.bounds,
        problem_id=f"{problem.problem_id}/augmented-subproblem",
    )
    integer_zero = jnp.asarray(0, dtype=jnp.int32)
    scalar_dtype = jnp.result_type(equality, inequality)
    initial_state = _AugmentedLagrangianState(
        outer=integer_zero,
        iterations=integer_zero,
        parameters=parameters,
        status=initial_status,
        equality_multipliers=jnp.zeros_like(equality),
        inequality_multipliers=jnp.zeros_like(inequality),
        penalty=jnp.asarray(method.initial_penalty, dtype=scalar_dtype),
        previous_feasibility=jnp.asarray(jnp.inf, dtype=scalar_dtype),
        objective_evaluations=integer_zero,
        gradient_evaluations=integer_zero,
        residual_evaluations=integer_zero,
        jvp_evaluations=integer_zero,
        vjp_evaluations=integer_zero,
        hvp_evaluations=integer_zero,
        jacobian_evaluations=integer_zero,
        constraint_evaluations=jnp.asarray(
            len(problem.constraints),
            dtype=jnp.int32,
        ),
        accepted_steps=integer_zero,
        rejected_steps=integer_zero,
        linear_solves=integer_zero,
        linear_iterations=integer_zero,
        setup_refreshes=integer_zero,
        numeric_refreshes=integer_zero,
        globalization_evaluations=integer_zero,
        direction_fallbacks=integer_zero,
        final_step_norm=jnp.asarray(0.0, dtype=scalar_dtype),
        initial_optimality=jnp.asarray(jnp.nan, dtype=scalar_dtype),
    )

    def condition(state):
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else state.objective_evaluations < termination.maximum_evaluations
        )
        return (
            (state.status == int(OptimizationStatus.ITERATING))
            & (state.outer < maximum_outer_steps)
            & within_evaluations
        )

    def body(state):
        inner = inner_method.solve(
            inner_problem,
            state.parameters,
            termination=inner_termination,
            args=(
                args,
                state.equality_multipliers,
                state.inequality_multipliers,
                state.penalty,
            ),
        )
        inner_diagnostics = inner.diagnostics

        def observed(count):
            return jnp.maximum(jnp.asarray(count, dtype=jnp.int32), 0)

        objective_evaluations = state.objective_evaluations + observed(
            inner_diagnostics.objective_evaluations
        )
        gradient_evaluations = state.gradient_evaluations + observed(
            inner_diagnostics.gradient_evaluations
        )
        residual_evaluations = state.residual_evaluations + observed(
            inner_diagnostics.residual_evaluations
        )
        jvp_evaluations = state.jvp_evaluations + observed(
            inner_diagnostics.jvp_evaluations
        )
        vjp_evaluations = state.vjp_evaluations + observed(
            inner_diagnostics.vjp_evaluations
        )
        hvp_evaluations = state.hvp_evaluations + observed(
            inner_diagnostics.hvp_evaluations
        )
        jacobian_evaluations = state.jacobian_evaluations + observed(
            inner_diagnostics.jacobian_evaluations
        )
        constraint_evaluations = state.constraint_evaluations + len(
            problem.constraints
        ) * observed(inner_diagnostics.objective_evaluations)
        linear_solves = state.linear_solves + observed(inner_diagnostics.linear_solves)
        linear_iterations = state.linear_iterations + observed(
            inner_diagnostics.linear_iterations
        )
        setup_refreshes = state.setup_refreshes + observed(
            inner_diagnostics.setup_refreshes
        )
        numeric_refreshes = state.numeric_refreshes + observed(
            inner_diagnostics.numeric_refreshes
        )
        globalization_evaluations = state.globalization_evaluations + observed(
            inner_diagnostics.globalization_evaluations
        )
        direction_fallbacks = state.direction_fallbacks + observed(
            inner_diagnostics.direction_fallbacks
        )
        inner_status = jnp.asarray(inner.status, dtype=jnp.int32)
        inner_failed = (
            (inner_status == int(OptimizationStatus.NONFINITE_INPUT))
            | (inner_status == int(OptimizationStatus.NONFINITE_EVALUATION))
            | (inner_status == int(OptimizationStatus.BACKEND_FAILED))
            | (inner_status == int(OptimizationStatus.DIVERGENCE))
        )

        def reject_inner(_):
            return _AugmentedLagrangianState(
                outer=state.outer + 1,
                iterations=state.iterations,
                parameters=state.parameters,
                status=jnp.asarray(
                    int(OptimizationStatus.BACKEND_FAILED),
                    dtype=jnp.int32,
                ),
                equality_multipliers=state.equality_multipliers,
                inequality_multipliers=state.inequality_multipliers,
                penalty=state.penalty,
                previous_feasibility=state.previous_feasibility,
                objective_evaluations=objective_evaluations,
                gradient_evaluations=gradient_evaluations,
                residual_evaluations=residual_evaluations,
                jvp_evaluations=jvp_evaluations,
                vjp_evaluations=vjp_evaluations,
                hvp_evaluations=hvp_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                constraint_evaluations=constraint_evaluations,
                accepted_steps=state.accepted_steps,
                rejected_steps=state.rejected_steps + 1,
                linear_solves=linear_solves,
                linear_iterations=linear_iterations,
                setup_refreshes=setup_refreshes,
                numeric_refreshes=numeric_refreshes,
                globalization_evaluations=globalization_evaluations,
                direction_fallbacks=direction_fallbacks,
                final_step_norm=state.final_step_norm,
                initial_optimality=state.initial_optimality,
            )

        def accept_inner(_):
            candidate_parameters = inner.parameters
            final_step_norm = _tree_norm(
                jax.tree.map(
                    lambda current, previous: current - previous,
                    candidate_parameters,
                    state.parameters,
                )
            )
            (
                _,
                unravel,
                _,
                objective_gradient,
                candidate_equality,
                candidate_inequality,
                equality_jacobian,
                inequality_jacobian,
            ) = _derivatives(problem, layout, candidate_parameters, args)
            next_objective_evaluations = objective_evaluations + 1
            next_gradient_evaluations = gradient_evaluations + 1
            next_constraint_evaluations = constraint_evaluations + 2 * len(
                problem.constraints
            )
            equality_multipliers = (
                state.equality_multipliers + state.penalty * candidate_equality
            )
            inequality_multipliers = jnp.maximum(
                0.0,
                state.inequality_multipliers + state.penalty * candidate_inequality,
            )
            primal, _, _, optimality = _kkt_metrics(
                problem,
                candidate_parameters,
                unravel,
                objective_gradient,
                candidate_equality,
                candidate_inequality,
                equality_jacobian,
                inequality_jacobian,
                equality_multipliers,
                inequality_multipliers,
            )
            initial_optimality = jnp.where(
                state.outer == 0,
                optimality,
                state.initial_optimality,
            )
            converged = optimality <= termination.optimality_threshold(initial_optimality)
            increase_penalty = (
                primal
                > method.required_feasibility_reduction * state.previous_feasibility
            )
            penalty = jnp.where(
                increase_penalty,
                jnp.minimum(
                    method.maximum_penalty,
                    state.penalty * method.penalty_increase,
                ),
                state.penalty,
            )
            return _AugmentedLagrangianState(
                outer=state.outer + 1,
                iterations=state.outer + 1,
                parameters=candidate_parameters,
                status=jnp.where(
                    converged,
                    int(OptimizationStatus.SUCCESS),
                    int(OptimizationStatus.ITERATING),
                ).astype(jnp.int32),
                equality_multipliers=equality_multipliers,
                inequality_multipliers=inequality_multipliers,
                penalty=penalty,
                previous_feasibility=primal,
                objective_evaluations=next_objective_evaluations,
                gradient_evaluations=next_gradient_evaluations,
                residual_evaluations=residual_evaluations,
                jvp_evaluations=jvp_evaluations,
                vjp_evaluations=vjp_evaluations,
                hvp_evaluations=hvp_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                constraint_evaluations=next_constraint_evaluations,
                accepted_steps=state.accepted_steps + 1,
                rejected_steps=state.rejected_steps,
                linear_solves=linear_solves,
                linear_iterations=linear_iterations,
                setup_refreshes=setup_refreshes,
                numeric_refreshes=numeric_refreshes,
                globalization_evaluations=globalization_evaluations,
                direction_fallbacks=direction_fallbacks,
                final_step_norm=final_step_norm,
                initial_optimality=initial_optimality,
            )

        return jax.lax.cond(inner_failed, reject_inner, accept_inner, None)

    state = jax.lax.while_loop(condition, body, initial_state)
    if termination.maximum_evaluations is None:
        exhausted_status = int(OptimizationStatus.MAXIMUM_STEPS_REACHED)
    else:
        exhausted_status = jnp.where(
            state.objective_evaluations >= termination.maximum_evaluations,
            int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
            int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        )
    status = jnp.where(
        state.status == int(OptimizationStatus.ITERATING),
        exhausted_status,
        state.status,
    ).astype(jnp.int32)

    (
        _,
        unravel,
        final_value,
        objective_gradient,
        equality,
        inequality,
        equality_jacobian,
        inequality_jacobian,
    ) = _derivatives(problem, layout, state.parameters, args)
    objective_evaluations = state.objective_evaluations + 1
    gradient_evaluations = state.gradient_evaluations + 1
    constraint_evaluations = state.constraint_evaluations + 2 * len(problem.constraints)
    primal, dual, complementarity, final_optimality = _kkt_metrics(
        problem,
        state.parameters,
        unravel,
        objective_gradient,
        equality,
        inequality,
        equality_jacobian,
        inequality_jacobian,
        state.equality_multipliers,
        state.inequality_multipliers,
    )
    eligible_for_success = (
        (status == int(OptimizationStatus.ITERATING))
        | (status == int(OptimizationStatus.MAXIMUM_STEPS_REACHED))
        | (status == int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED))
    )
    status = jnp.where(
        eligible_for_success
        & (
            final_optimality <= termination.optimality_threshold(state.initial_optimality)
        ),
        int(OptimizationStatus.SUCCESS),
        status,
    ).astype(jnp.int32)
    lagrangian_gradient = _lagrangian_gradient(
        objective_gradient,
        equality_jacobian,
        inequality_jacobian,
        state.equality_multipliers,
        state.inequality_multipliers,
    )
    active_constraints = _active_constraint_count(
        inequality,
        tolerance=termination.absolute_optimality,
    )
    _, auxiliary = problem.value(state.parameters, args)
    objective_evaluations = objective_evaluations + 1
    diagnostics = OptimizationDiagnostics(
        iterations=state.iterations,
        accepted_steps=state.accepted_steps,
        rejected_steps=state.rejected_steps,
        objective_evaluations=objective_evaluations,
        gradient_evaluations=gradient_evaluations,
        residual_evaluations=state.residual_evaluations,
        jvp_evaluations=state.jvp_evaluations,
        vjp_evaluations=state.vjp_evaluations,
        hvp_evaluations=state.hvp_evaluations,
        jacobian_evaluations=state.jacobian_evaluations,
        constraint_evaluations=constraint_evaluations,
        linear_solves=state.linear_solves,
        linear_iterations=state.linear_iterations,
        setup_refreshes=state.setup_refreshes,
        numeric_refreshes=state.numeric_refreshes,
        globalization_evaluations=state.globalization_evaluations,
        initial_optimality_norm=state.initial_optimality,
        final_optimality_norm=final_optimality,
        final_step_norm=state.final_step_norm,
        damping=state.penalty,
        direction_fallbacks=state.direction_fallbacks,
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        active_constraints=active_constraints,
        counts_complete=False,
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax",
        backend_method=inner_method.method_id,
        globalization="powell-hestenes",
        matrix_free=inner_method.capabilities.matrix_free,
        implicit_differentiation=True,
        notes=(
            "Inequalities use projected multiplier updates; diagnostics report "
            "primal, dual, and complementarity KKT residuals."
        ),
    )
    certificate = _constraint_certificate(
        layout,
        unravel,
        lagrangian_gradient,
        state.equality_multipliers,
        state.inequality_multipliers,
        inequality,
        primal,
        dual,
        complementarity,
        active_tolerance=termination.absolute_optimality,
    )
    return MinimizationResult(
        state.parameters,
        final_value,
        auxiliary,
        status,
        diagnostics,
        provenance,
        certificate=certificate,
    )


def _bfgs_update(matrix: Array, step: Array, gradient_change: Array, /) -> Array:
    matrix_step = matrix @ step
    step_matrix_step = jnp.vdot(step, matrix_step).real
    step_gradient = jnp.vdot(step, gradient_change).real
    finite_curvature = (
        jnp.isfinite(step_matrix_step)
        & jnp.isfinite(step_gradient)
        & (step_matrix_step > 1e-30)
    )
    safe_step_matrix_step = jnp.where(
        finite_curvature,
        step_matrix_step,
        jnp.ones_like(step_matrix_step),
    )
    theta = jnp.where(
        step_gradient >= 0.2 * safe_step_matrix_step,
        1.0,
        0.8
        * safe_step_matrix_step
        / jnp.maximum(safe_step_matrix_step - step_gradient, 1e-30),
    )
    damped_change = theta * gradient_change + (1.0 - theta) * matrix_step
    step_damped = jnp.vdot(step, damped_change).real
    usable_update = finite_curvature & jnp.isfinite(step_damped) & (step_damped > 1e-30)
    safe_step_damped = jnp.where(
        usable_update,
        step_damped,
        jnp.ones_like(step_damped),
    )
    updated = (
        matrix
        - jnp.outer(matrix_step, matrix_step) / safe_step_matrix_step
        + jnp.outer(damped_change, damped_change) / safe_step_damped
    )
    symmetric = 0.5 * (updated + updated.T)
    return jnp.where(usable_update, symmetric, matrix)


class _FilterSearchResult(eqx.Module):
    parameters: Array
    value: Array
    violation: Array
    rate: Array
    evaluations: Array
    derivative_evaluations: Array
    accepted: Array
    finite_candidate_seen: Array
    correction_used: Array


def _filter_backtracking(
    problem: MinimizationProblem,
    layout: _ConstraintLayout,
    unravel,
    flat_parameters: Array,
    value: Array,
    equality: Array,
    inequality: Array,
    direction: Array,
    objective_directional: Array,
    filter_objectives: Array,
    filter_violations: Array,
    filter_size: Array,
    args: Any,
    /,
    *,
    line_search: ArmijoLineSearch,
    policy: FilterGlobalization,
    second_order_correction: bool,
) -> _FilterSearchResult:
    """Backtrack against an objective-feasibility filter, optionally applying SOC."""

    current_violation = _constraint_l1(equality, inequality)
    switching_threshold = jnp.sqrt(jnp.finfo(flat_parameters.dtype).eps) * (
        1.0 + jnp.abs(value)
    )
    require_objective_decrease = current_violation <= switching_threshold
    insertion = jnp.minimum(
        jnp.asarray(filter_size, dtype=jnp.int32),
        filter_objectives.size - 1,
    )
    local_objectives = filter_objectives.at[insertion].set(value)
    local_violations = filter_violations.at[insertion].set(current_violation)
    local_size = jnp.minimum(filter_size + 1, filter_objectives.size)
    initial_rate = jnp.asarray(line_search.initial_rate, dtype=flat_parameters.dtype)

    def evaluate(candidate, candidate_rate):
        candidate_parameters = unravel(candidate)
        candidate_value = problem.value(candidate_parameters, args)[0]
        candidate_equality, candidate_inequality = _canonical_constraints(
            problem,
            layout,
            candidate_parameters,
            args,
        )
        candidate_violation = _constraint_l1(
            candidate_equality,
            candidate_inequality,
        )
        finite = (
            jnp.isfinite(candidate_value)
            & jnp.isfinite(candidate_violation)
            & jnp.all(jnp.isfinite(candidate))
            & (
                jnp.asarray(True)
                if problem.bounds is None
                else problem.bounds.contains(candidate_parameters)
            )
        )
        filter_acceptable = policy.acceptable(
            candidate_value,
            candidate_violation,
            local_objectives,
            local_violations,
            local_size,
        )
        objective_armijo = (
            jnp.isfinite(objective_directional)
            & (objective_directional < 0.0)
            & (
                candidate_value
                <= value
                + line_search.sufficient_decrease * candidate_rate * objective_directional
            )
        )
        acceptable = (
            finite
            & filter_acceptable
            & ((~require_objective_decrease) | objective_armijo)
        )
        return (
            candidate_value,
            candidate_equality,
            candidate_inequality,
            candidate_violation,
            finite,
            acceptable,
        )

    def condition(carry):
        trial, rate, _, _, accepted, *_ = carry
        return (
            (trial < line_search.maximum_steps)
            & (~accepted)
            & (rate >= line_search.minimum_rate)
        )

    def body(carry):
        (
            trial,
            rate,
            evaluations,
            derivative_evaluations,
            _,
            accepted_parameters,
            accepted_value,
            accepted_violation,
            accepted_rate,
            finite_seen,
            correction_used,
        ) = carry
        candidate = flat_parameters + rate * direction
        (
            candidate_value,
            candidate_equality,
            candidate_inequality,
            candidate_violation,
            candidate_finite,
            candidate_acceptable,
        ) = evaluate(candidate, rate)

        if second_order_correction:

            def canonical(vector):
                return _canonical_constraints(
                    problem,
                    layout,
                    unravel(vector),
                    args,
                )

            equality_jacobian, inequality_jacobian = jax.jacrev(canonical)(candidate)
            active = (candidate_inequality > 0.0).astype(candidate.dtype)
            correction_matrix = jnp.concatenate(
                (
                    equality_jacobian,
                    active[:, None] * inequality_jacobian,
                ),
                axis=0,
            )
            correction_rhs = jnp.concatenate(
                (
                    -candidate_equality,
                    -active * candidate_inequality,
                )
            )
            normal = (
                correction_matrix @ correction_matrix.T
                + policy.correction_regularization
                * jnp.eye(
                    correction_matrix.shape[0],
                    dtype=correction_matrix.dtype,
                )
            )
            correction = correction_matrix.T @ jnp.linalg.solve(
                normal,
                correction_rhs,
            )
            correction_finite = jnp.all(jnp.isfinite(correction)) & (
                jnp.linalg.norm(correction)
                <= policy.correction_limit
                * jnp.maximum(
                    rate * jnp.linalg.norm(direction),
                    1e-30,
                )
            )
            corrected = candidate + correction
            (
                corrected_value,
                _,
                _,
                corrected_violation,
                corrected_finite,
                corrected_acceptable,
            ) = evaluate(corrected, rate)
            use_correction = (
                (~candidate_acceptable) & correction_finite & corrected_acceptable
            )
            accepted = candidate_acceptable | use_correction
            selected_parameters = jnp.where(
                use_correction,
                corrected,
                candidate,
            )
            selected_value = jnp.where(
                use_correction,
                corrected_value,
                candidate_value,
            )
            selected_violation = jnp.where(
                use_correction,
                corrected_violation,
                candidate_violation,
            )
            evaluations_increment = jnp.asarray(2, dtype=jnp.int32)
            derivative_increment = jnp.asarray(1, dtype=jnp.int32)
            finite = candidate_finite | corrected_finite
        else:
            accepted = candidate_acceptable
            selected_parameters = candidate
            selected_value = candidate_value
            selected_violation = candidate_violation
            use_correction = jnp.asarray(False)
            evaluations_increment = jnp.asarray(1, dtype=jnp.int32)
            derivative_increment = jnp.asarray(0, dtype=jnp.int32)
            finite = candidate_finite

        return (
            trial + 1,
            jnp.where(
                accepted,
                rate,
                rate * line_search.contraction,
            ),
            evaluations + evaluations_increment,
            derivative_evaluations + derivative_increment,
            accepted,
            jnp.where(
                accepted,
                selected_parameters,
                accepted_parameters,
            ),
            jnp.where(accepted, selected_value, accepted_value),
            jnp.where(
                accepted,
                selected_violation,
                accepted_violation,
            ),
            jnp.where(accepted, rate, accepted_rate),
            finite_seen | finite,
            correction_used | (accepted & use_correction),
        )

    (
        _,
        _,
        evaluations,
        derivative_evaluations,
        accepted,
        accepted_parameters,
        accepted_value,
        accepted_violation,
        accepted_rate,
        finite_seen,
        correction_used,
    ) = jax.lax.while_loop(
        condition,
        body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            initial_rate,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            flat_parameters,
            jnp.asarray(value),
            current_violation,
            jnp.asarray(0.0, dtype=flat_parameters.dtype),
            jnp.asarray(False),
            jnp.asarray(False),
        ),
    )
    return _FilterSearchResult(
        parameters=accepted_parameters,
        value=accepted_value,
        violation=accepted_violation,
        rate=accepted_rate,
        evaluations=evaluations,
        derivative_evaluations=derivative_evaluations,
        accepted=accepted,
        finite_candidate_seen=finite_seen,
        correction_used=correction_used,
    )


class _SQPState(eqx.Module):
    """Array-only carry for staged SQP iteration and globalization."""

    iteration: Array
    iterations: Array
    parameters: PyTree[Any]
    status: Array
    hessian: Array
    equality_multipliers: Array
    inequality_multipliers: Array
    merit_penalty: Array
    filter_objectives: Array
    filter_violations: Array
    filter_size: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    constraint_evaluations: Array
    accepted_steps: Array
    rejected_steps: Array
    globalization_evaluations: Array
    direction_fallbacks: Array
    final_step_norm: Array
    accepted_rate: Array
    initial_optimality: Array


def _solve_sqp(
    method: SQP,
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> MinimizationResult:
    if not problem.constraints and problem.bounds is None:
        raise ValueError("SQP requires nonlinear constraints or parameter bounds.")
    parameters = initial_parameters
    if problem.bounds is not None:
        parameters = problem.bounds.project(parameters)
    initial_status = jnp.where(
        _tree_allfinite(parameters),
        int(OptimizationStatus.ITERATING),
        int(OptimizationStatus.NONFINITE_INPUT),
    ).astype(jnp.int32)
    layout = _constraint_layout(problem, parameters, args)
    flat_parameters, _ = ravel_pytree(parameters)
    dimension = int(flat_parameters.size)
    if dimension > method.max_dense_dimension:
        raise ValueError(
            f"SQP has {dimension} variables, exceeding max_dense_dimension="
            f"{method.max_dense_dimension}."
        )
    integer_zero = jnp.asarray(0, dtype=jnp.int32)
    scalar_zero = jnp.asarray(0.0, dtype=flat_parameters.dtype)
    inequality_count = layout.lower_indices.size + layout.upper_indices.size
    initial_state = _SQPState(
        iteration=integer_zero,
        iterations=integer_zero,
        parameters=parameters,
        status=initial_status,
        hessian=method.hessian_scale * jnp.eye(dimension, dtype=flat_parameters.dtype),
        equality_multipliers=jnp.zeros(
            (layout.equality_indices.size,),
            dtype=flat_parameters.dtype,
        ),
        inequality_multipliers=jnp.zeros(
            (inequality_count,),
            dtype=flat_parameters.dtype,
        ),
        merit_penalty=jnp.asarray(
            method.merit_penalty,
            dtype=flat_parameters.dtype,
        ),
        filter_objectives=jnp.full(
            (termination.maximum_steps + 1,),
            jnp.inf,
            dtype=flat_parameters.dtype,
        ),
        filter_violations=jnp.full(
            (termination.maximum_steps + 1,),
            jnp.inf,
            dtype=flat_parameters.dtype,
        ),
        filter_size=integer_zero,
        objective_evaluations=integer_zero,
        gradient_evaluations=integer_zero,
        constraint_evaluations=jnp.asarray(
            len(problem.constraints),
            dtype=jnp.int32,
        ),
        accepted_steps=integer_zero,
        rejected_steps=integer_zero,
        globalization_evaluations=integer_zero,
        direction_fallbacks=integer_zero,
        final_step_norm=scalar_zero,
        accepted_rate=scalar_zero,
        initial_optimality=jnp.asarray(jnp.nan, dtype=flat_parameters.dtype),
    )

    def condition(state):
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else state.objective_evaluations < termination.maximum_evaluations
        )
        return (
            (state.status == int(OptimizationStatus.ITERATING))
            & (state.iteration < termination.maximum_steps)
            & within_evaluations
        )

    def body(state):
        (
            flat_parameters,
            unravel,
            value,
            gradient,
            equality,
            inequality,
            equality_jacobian,
            inequality_jacobian,
        ) = _derivatives(problem, layout, state.parameters, args)
        objective_evaluations = state.objective_evaluations + 1
        gradient_evaluations = state.gradient_evaluations + 1
        constraint_evaluations = state.constraint_evaluations + 2 * len(
            problem.constraints
        )
        _, _, _, optimality = _kkt_metrics(
            problem,
            state.parameters,
            unravel,
            gradient,
            equality,
            inequality,
            equality_jacobian,
            inequality_jacobian,
            state.equality_multipliers,
            state.inequality_multipliers,
        )
        initial_optimality = jnp.where(
            state.iteration == 0,
            optimality,
            state.initial_optimality,
        )
        finite_model = (
            jnp.isfinite(value)
            & jnp.isfinite(optimality)
            & jnp.all(jnp.isfinite(gradient))
        )
        converged = optimality <= termination.optimality_threshold(initial_optimality)

        def finish_model(_):
            return _SQPState(
                iteration=state.iteration,
                iterations=state.iterations,
                parameters=state.parameters,
                status=jnp.where(
                    finite_model,
                    int(OptimizationStatus.SUCCESS),
                    int(OptimizationStatus.NONFINITE_EVALUATION),
                ).astype(jnp.int32),
                hessian=state.hessian,
                equality_multipliers=state.equality_multipliers,
                inequality_multipliers=state.inequality_multipliers,
                merit_penalty=state.merit_penalty,
                filter_objectives=state.filter_objectives,
                filter_violations=state.filter_violations,
                filter_size=state.filter_size,
                objective_evaluations=objective_evaluations,
                gradient_evaluations=gradient_evaluations,
                constraint_evaluations=constraint_evaluations,
                accepted_steps=state.accepted_steps,
                rejected_steps=(state.rejected_steps + (~finite_model).astype(jnp.int32)),
                globalization_evaluations=state.globalization_evaluations,
                direction_fallbacks=state.direction_fallbacks,
                final_step_norm=state.final_step_norm,
                accepted_rate=state.accepted_rate,
                initial_optimality=initial_optimality,
            )

        def take_step(_):
            qp = QuadraticProgram(
                state.hessian,
                gradient,
                equality_matrix=equality_jacobian,
                equality_rhs=-equality,
                inequality_matrix=inequality_jacobian,
                inequality_rhs=-inequality,
            )
            qp_result = solve_quadratic_program(
                qp,
                tolerance=method.qp_tolerance,
                max_iterations=method.qp_maximum_steps,
                regularization=method.qp_regularization,
                max_dense_dimension=method.max_dense_dimension,
            )
            qp_success = (
                (qp_result.status == QP_SUCCESS)
                & qp_result.valid
                & jnp.all(jnp.isfinite(qp_result.primal))
            )
            equality_trial_multipliers = jnp.where(
                qp_success,
                qp_result.equality_dual,
                state.equality_multipliers,
            )
            inequality_trial_multipliers = jnp.where(
                qp_success,
                qp_result.inequality_dual,
                state.inequality_multipliers,
            )
            multiplier_scale = jnp.maximum(
                jnp.max(
                    jnp.abs(qp_result.equality_dual),
                    initial=0.0,
                ),
                jnp.max(
                    qp_result.inequality_dual,
                    initial=0.0,
                ),
            )
            merit_penalty = jnp.where(
                qp_success,
                jnp.maximum(state.merit_penalty, 1.0 + multiplier_scale),
                state.merit_penalty,
            )
            qp_direction = jnp.where(
                qp_success,
                qp_result.primal,
                jnp.zeros_like(flat_parameters),
            )

            def merit(candidate):
                candidate_parameters = unravel(candidate)
                candidate_value = problem.value(candidate_parameters, args)[0]
                candidate_equality, candidate_inequality = _canonical_constraints(
                    problem,
                    layout,
                    candidate_parameters,
                    args,
                )
                return candidate_value + merit_penalty * _constraint_l1(
                    candidate_equality,
                    candidate_inequality,
                )

            merit_value, merit_gradient = jax.value_and_grad(merit)(flat_parameters)
            next_objective_evaluations = objective_evaluations + 1
            next_gradient_evaluations = gradient_evaluations + 1
            next_constraint_evaluations = constraint_evaluations + len(
                problem.constraints
            )
            qp_directional = jnp.vdot(merit_gradient, qp_direction).real
            if method.filter_globalization is None:
                qp_direction_usable = (
                    qp_success
                    & jnp.all(jnp.isfinite(qp_direction))
                    & jnp.isfinite(qp_directional)
                    & (qp_directional < 0.0)
                )
            else:
                qp_direction_usable = (
                    qp_success
                    & jnp.all(jnp.isfinite(qp_direction))
                    & jnp.isfinite(qp_directional)
                )
            fallback_direction = -merit_gradient
            if problem.bounds is not None:
                projected = problem.bounds.project(
                    unravel(flat_parameters + fallback_direction)
                )
                projected_flat, _ = ravel_pytree(projected)
                fallback_direction = projected_flat - flat_parameters
            direction = jnp.where(
                qp_direction_usable,
                qp_direction,
                fallback_direction,
            )
            directional = jnp.vdot(merit_gradient, direction).real
            direction_fallbacks = state.direction_fallbacks + (
                ~qp_direction_usable
            ).astype(jnp.int32)
            usable_direction = (
                jnp.isfinite(merit_value)
                & jnp.all(jnp.isfinite(merit_gradient))
                & jnp.all(jnp.isfinite(direction))
                & jnp.isfinite(directional)
                & (
                    (directional < 0.0)
                    if method.filter_globalization is None
                    else jnp.asarray(True)
                )
            )

            def reject_direction(_):
                return _SQPState(
                    iteration=state.iteration,
                    iterations=state.iterations,
                    parameters=state.parameters,
                    status=jnp.asarray(
                        int(OptimizationStatus.RESTORATION_FAILED),
                        dtype=jnp.int32,
                    ),
                    hessian=state.hessian,
                    equality_multipliers=state.equality_multipliers,
                    inequality_multipliers=state.inequality_multipliers,
                    merit_penalty=merit_penalty,
                    filter_objectives=state.filter_objectives,
                    filter_violations=state.filter_violations,
                    filter_size=state.filter_size,
                    objective_evaluations=next_objective_evaluations,
                    gradient_evaluations=next_gradient_evaluations,
                    constraint_evaluations=next_constraint_evaluations,
                    accepted_steps=state.accepted_steps,
                    rejected_steps=state.rejected_steps + 1,
                    globalization_evaluations=state.globalization_evaluations,
                    direction_fallbacks=direction_fallbacks,
                    final_step_norm=state.final_step_norm,
                    accepted_rate=state.accepted_rate,
                    initial_optimality=initial_optimality,
                )

            def globalize(_):
                if method.filter_globalization is None:
                    search = armijo_backtracking(
                        merit,
                        flat_parameters,
                        merit_value,
                        direction,
                        directional,
                        step=lambda base, tangent, rate: base + rate * tangent,
                        contains=lambda candidate: (
                            jnp.all(jnp.isfinite(candidate))
                            if problem.bounds is None
                            else problem.bounds.contains(unravel(candidate))
                        ),
                        policy=method.line_search,
                    )
                    derivative_evaluations = jnp.asarray(0, dtype=jnp.int32)
                else:
                    search = _filter_backtracking(
                        problem,
                        layout,
                        unravel,
                        flat_parameters,
                        value,
                        equality,
                        inequality,
                        direction,
                        jnp.vdot(gradient, direction).real,
                        state.filter_objectives,
                        state.filter_violations,
                        state.filter_size,
                        args,
                        line_search=method.line_search,
                        policy=method.filter_globalization,
                        second_order_correction=method.second_order_correction,
                    )
                    derivative_evaluations = search.derivative_evaluations
                evaluations = search.evaluations
                search_objective_evaluations = next_objective_evaluations + evaluations
                search_constraint_evaluations = next_constraint_evaluations + (
                    evaluations + derivative_evaluations
                ) * len(problem.constraints)
                globalization_evaluations = state.globalization_evaluations + evaluations
                final_step_norm = jnp.linalg.norm(search.parameters - flat_parameters)

                def reject_search(_):
                    return _SQPState(
                        iteration=state.iteration + 1,
                        iterations=state.iteration + 1,
                        parameters=state.parameters,
                        status=jnp.where(
                            qp_success,
                            int(OptimizationStatus.LINE_SEARCH_FAILED),
                            int(OptimizationStatus.RESTORATION_FAILED),
                        ).astype(jnp.int32),
                        hessian=state.hessian,
                        equality_multipliers=state.equality_multipliers,
                        inequality_multipliers=state.inequality_multipliers,
                        merit_penalty=merit_penalty,
                        filter_objectives=state.filter_objectives,
                        filter_violations=state.filter_violations,
                        filter_size=state.filter_size,
                        objective_evaluations=search_objective_evaluations,
                        gradient_evaluations=next_gradient_evaluations,
                        constraint_evaluations=search_constraint_evaluations,
                        accepted_steps=state.accepted_steps,
                        rejected_steps=state.rejected_steps + 1,
                        globalization_evaluations=globalization_evaluations,
                        direction_fallbacks=direction_fallbacks,
                        final_step_norm=final_step_norm,
                        accepted_rate=search.rate,
                        initial_optimality=initial_optimality,
                    )

                def accept_search(_):
                    previous_lagrangian_gradient = _lagrangian_gradient(
                        gradient,
                        equality_jacobian,
                        inequality_jacobian,
                        equality_trial_multipliers,
                        inequality_trial_multipliers,
                    )
                    candidate_flat = search.parameters
                    candidate_parameters = unravel(candidate_flat)
                    equality_multipliers = equality_trial_multipliers
                    inequality_multipliers = jnp.maximum(
                        0.0,
                        inequality_trial_multipliers,
                    )
                    (
                        _,
                        _,
                        _,
                        next_gradient,
                        _,
                        _,
                        next_equality_jacobian,
                        next_inequality_jacobian,
                    ) = _derivatives(
                        problem,
                        layout,
                        candidate_parameters,
                        args,
                    )
                    accepted_objective_evaluations = search_objective_evaluations + 1
                    accepted_gradient_evaluations = next_gradient_evaluations + 1
                    accepted_constraint_evaluations = (
                        search_constraint_evaluations + 2 * len(problem.constraints)
                    )
                    next_lagrangian_gradient = _lagrangian_gradient(
                        next_gradient,
                        next_equality_jacobian,
                        next_inequality_jacobian,
                        equality_multipliers,
                        inequality_multipliers,
                    )
                    hessian = _bfgs_update(
                        state.hessian,
                        candidate_flat - flat_parameters,
                        (next_lagrangian_gradient - previous_lagrangian_gradient),
                    )
                    stagnated = final_step_norm <= termination.step_threshold(
                        jnp.linalg.norm(candidate_flat)
                    )
                    if method.filter_globalization is None:
                        accepted_filter_objectives = state.filter_objectives
                        accepted_filter_violations = state.filter_violations
                        accepted_filter_size = state.filter_size
                    else:
                        filter_insertion = jnp.minimum(
                            state.filter_size,
                            state.filter_objectives.size - 1,
                        )
                        accepted_filter_objectives = state.filter_objectives.at[
                            filter_insertion
                        ].set(search.value)
                        accepted_filter_violations = state.filter_violations.at[
                            filter_insertion
                        ].set(search.violation)
                        accepted_filter_size = jnp.minimum(
                            state.filter_size + 1,
                            state.filter_objectives.size,
                        )
                    return _SQPState(
                        iteration=state.iteration + 1,
                        iterations=state.iteration + 1,
                        parameters=candidate_parameters,
                        status=jnp.where(
                            stagnated,
                            int(OptimizationStatus.STAGNATION),
                            int(OptimizationStatus.ITERATING),
                        ).astype(jnp.int32),
                        hessian=hessian,
                        equality_multipliers=equality_multipliers,
                        inequality_multipliers=inequality_multipliers,
                        merit_penalty=merit_penalty,
                        filter_objectives=accepted_filter_objectives,
                        filter_violations=accepted_filter_violations,
                        filter_size=accepted_filter_size,
                        objective_evaluations=accepted_objective_evaluations,
                        gradient_evaluations=accepted_gradient_evaluations,
                        constraint_evaluations=accepted_constraint_evaluations,
                        accepted_steps=state.accepted_steps + 1,
                        rejected_steps=state.rejected_steps,
                        globalization_evaluations=globalization_evaluations,
                        direction_fallbacks=direction_fallbacks,
                        final_step_norm=final_step_norm,
                        accepted_rate=search.rate,
                        initial_optimality=initial_optimality,
                    )

                return jax.lax.cond(
                    search.accepted,
                    accept_search,
                    reject_search,
                    None,
                )

            return jax.lax.cond(
                usable_direction,
                globalize,
                reject_direction,
                None,
            )

        return jax.lax.cond(
            (~finite_model) | converged,
            finish_model,
            take_step,
            None,
        )

    state = jax.lax.while_loop(condition, body, initial_state)
    if termination.maximum_evaluations is None:
        exhausted_status = int(OptimizationStatus.MAXIMUM_STEPS_REACHED)
    else:
        exhausted_status = jnp.where(
            state.objective_evaluations >= termination.maximum_evaluations,
            int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
            int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
        )
    status = jnp.where(
        state.status == int(OptimizationStatus.ITERATING),
        exhausted_status,
        state.status,
    ).astype(jnp.int32)

    (
        _,
        unravel,
        final_value,
        gradient,
        equality,
        inequality,
        equality_jacobian,
        inequality_jacobian,
    ) = _derivatives(problem, layout, state.parameters, args)
    objective_evaluations = state.objective_evaluations + 1
    gradient_evaluations = state.gradient_evaluations + 1
    constraint_evaluations = state.constraint_evaluations + 2 * len(problem.constraints)
    primal, dual, complementarity, final_optimality = _kkt_metrics(
        problem,
        state.parameters,
        unravel,
        gradient,
        equality,
        inequality,
        equality_jacobian,
        inequality_jacobian,
        state.equality_multipliers,
        state.inequality_multipliers,
    )
    eligible_for_success = (
        (status == int(OptimizationStatus.ITERATING))
        | (status == int(OptimizationStatus.MAXIMUM_STEPS_REACHED))
        | (status == int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED))
        | (status == int(OptimizationStatus.STAGNATION))
    )
    status = jnp.where(
        eligible_for_success
        & (
            final_optimality <= termination.optimality_threshold(state.initial_optimality)
        ),
        int(OptimizationStatus.SUCCESS),
        status,
    ).astype(jnp.int32)
    lagrangian_gradient = _lagrangian_gradient(
        gradient,
        equality_jacobian,
        inequality_jacobian,
        state.equality_multipliers,
        state.inequality_multipliers,
    )
    active_constraints = _active_constraint_count(
        inequality,
        tolerance=method.active_tolerance,
    )
    _, auxiliary = problem.value(state.parameters, args)
    objective_evaluations = objective_evaluations + 1
    diagnostics = OptimizationDiagnostics(
        iterations=state.iterations,
        accepted_steps=state.accepted_steps,
        rejected_steps=state.rejected_steps,
        objective_evaluations=objective_evaluations,
        gradient_evaluations=gradient_evaluations,
        constraint_evaluations=constraint_evaluations,
        globalization_evaluations=state.globalization_evaluations,
        initial_optimality_norm=state.initial_optimality,
        final_optimality_norm=final_optimality,
        final_step_norm=state.final_step_norm,
        accepted_step_size=state.accepted_rate,
        direction_fallbacks=state.direction_fallbacks,
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        active_constraints=active_constraints,
        counts_complete=False,
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax",
        backend_method="dense-primal-dual",
        globalization=(
            "objective-feasibility-filter-with-soc"
            if method.filter_globalization is not None and method.second_order_correction
            else "objective-feasibility-filter"
            if method.filter_globalization is not None
            else "l1-merit-armijo"
        ),
        matrix_free=False,
        implicit_differentiation=True,
        notes=(
            "Powell-damped BFGS QP Hessian; filter mode rejects pairs dominated "
            "in objective-feasibility space and optionally applies a linearized "
            "second-order constraint correction before rate contraction."
            if method.filter_globalization is not None
            else "Powell-damped BFGS QP Hessian with L1-merit Armijo restoration."
        ),
    )
    certificate = _constraint_certificate(
        layout,
        unravel,
        lagrangian_gradient,
        state.equality_multipliers,
        state.inequality_multipliers,
        inequality,
        primal,
        dual,
        complementarity,
        active_tolerance=method.active_tolerance,
    )
    return MinimizationResult(
        state.parameters,
        final_value,
        auxiliary,
        status,
        diagnostics,
        provenance,
        certificate=certificate,
    )


__all__ = ["AugmentedLagrangian", "FilterGlobalization", "SQP"]
