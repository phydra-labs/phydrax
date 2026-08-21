#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from ..linalg import (
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    MINRES,
    OperatorProperties,
    PyTreeSpace,
    solve as solve_linear,
    TolerancePolicy,
)
from ._iterative._base import AbstractMinimizationMethod
from ._iterative._globalization import armijo_backtracking, ArmijoLineSearch
from ._iterative._types import (
    _tree_add_scaled,
    _tree_allfinite,
    _tree_inner,
    _tree_negative,
    _tree_norm,
    _tree_scale,
    _tree_where,
    _validate_real_inexact_tree,
    Bounds,
    ConstrainedOptimalityCertificate,
    MinimizationProblem,
    MinimizationResult,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)


def _default_active_set_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        MINRES(),
        tolerance=TolerancePolicy(relative=1e-3, absolute=1e-10),
    )


def _usable_linear_status(status: Any, /) -> Array:
    status_ = jnp.asarray(status, dtype=jnp.int32)
    return (
        (status_ == int(LinearSolveStatus.SUCCESS))
        | (status_ == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED))
        | (status_ == int(LinearSolveStatus.STAGNATION))
        | (status_ == int(LinearSolveStatus.CONDITION_LIMIT_REACHED))
    )


def _tree_zeros_on_mask(
    vector: PyTree[Any],
    mask: PyTree[Any],
    /,
) -> PyTree[Array]:
    return jax.tree.map(lambda value, active: jnp.where(active, 0.0, value), vector, mask)


def _tree_mask_equal(left: PyTree[Any], right: PyTree[Any], /) -> Array:
    equal = jnp.asarray(True)
    for x, y in zip(
        jax.tree.leaves(left),
        jax.tree.leaves(right),
        strict=True,
    ):
        equal = equal & jnp.all(x == y)
    return equal


def _active_count(mask: PyTree[Any], /) -> Array:
    count = jnp.asarray(0, dtype=jnp.int32)
    for leaf in jax.tree.leaves(mask):
        count = count + jnp.sum(leaf, dtype=jnp.int32)
    return count


def _projected_displacement(
    bounds: Bounds,
    parameters: PyTree[Any],
    direction: PyTree[Any],
    /,
) -> PyTree[Array]:
    candidate = _tree_add_scaled(parameters, direction, 1.0)
    projected = bounds.project(candidate)
    return jax.tree.map(lambda new, old: new - old, projected, parameters)


def _complementarity_residual(
    bounds: Bounds,
    parameters: PyTree[Any],
    gradient: PyTree[Any],
    /,
    *,
    tolerance: float,
) -> Array:
    lower, upper = bounds.materialize(parameters)
    residual = jnp.asarray(0.0)
    for value, grad, lo, hi in zip(
        jax.tree.leaves(parameters),
        jax.tree.leaves(gradient),
        jax.tree.leaves(lower),
        jax.tree.leaves(upper),
        strict=True,
    ):
        lower_active = jnp.isfinite(lo) & (value <= lo + tolerance * (1.0 + jnp.abs(lo)))
        upper_active = jnp.isfinite(hi) & (value >= hi - tolerance * (1.0 + jnp.abs(hi)))
        lower_multiplier = jnp.where(lower_active, jnp.maximum(grad, 0.0), 0.0)
        upper_multiplier = jnp.where(upper_active, jnp.maximum(-grad, 0.0), 0.0)
        lower_gap = jnp.where(jnp.isfinite(lo), value - lo, 0.0)
        upper_gap = jnp.where(jnp.isfinite(hi), hi - value, 0.0)
        residual = jnp.maximum(
            residual,
            jnp.max(
                jnp.maximum(
                    jnp.abs(lower_gap * lower_multiplier),
                    jnp.abs(upper_gap * upper_multiplier),
                )
            ),
        )
    return residual


class ProjectedGradient(AbstractMinimizationMethod):
    """Projected-gradient baseline with feasible Armijo steps."""

    line_search: ArmijoLineSearch
    project_initial: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        line_search: ArmijoLineSearch | None = None,
        project_initial: bool = True,
    ):
        search = ArmijoLineSearch() if line_search is None else line_search
        if not isinstance(search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch or None.")
        self.line_search = search
        self.project_initial = bool(project_initial)

    @property
    def method_id(self) -> str:
        return "projected-gradient"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return _bound_capabilities()

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        return _solve_bound_constrained(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


class ActiveSetNewton(AbstractMinimizationMethod):
    """Matrix-free reduced Newton method over the current free-variable space."""

    linear_policy: LinearSolvePolicy
    line_search: ArmijoLineSearch
    active_tolerance: float = eqx.field(static=True)
    project_initial: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        line_search: ArmijoLineSearch | None = None,
        active_tolerance: float = 1e-10,
        project_initial: bool = True,
    ):
        policy = (
            _default_active_set_linear_policy()
            if linear_policy is None
            else linear_policy
        )
        search = ArmijoLineSearch() if line_search is None else line_search
        tolerance = float(active_tolerance)
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if not isinstance(search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch or None.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("active_tolerance must be finite and non-negative.")
        self.linear_policy = policy
        self.line_search = search
        self.active_tolerance = tolerance
        self.project_initial = bool(project_initial)

    @property
    def method_id(self) -> str:
        return "active-set-newton"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return _bound_capabilities()

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        return _solve_bound_constrained(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


class ProjectedLBFGS(AbstractMinimizationMethod):
    """Projected limited-memory BFGS with activity-aware memory resets."""

    line_search: ArmijoLineSearch
    history_size: int = eqx.field(static=True)
    curvature_tolerance: float = eqx.field(static=True)
    active_tolerance: float = eqx.field(static=True)
    project_initial: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        line_search: ArmijoLineSearch | None = None,
        history_size: int = 10,
        curvature_tolerance: float = 1e-10,
        active_tolerance: float = 1e-10,
        project_initial: bool = True,
    ):
        search = ArmijoLineSearch() if line_search is None else line_search
        history = int(history_size)
        curvature = float(curvature_tolerance)
        active = float(active_tolerance)
        if not isinstance(search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch or None.")
        if history < 1:
            raise ValueError("history_size must be positive.")
        if not isfinite(curvature) or curvature < 0.0:
            raise ValueError("curvature_tolerance must be finite and non-negative.")
        if not isfinite(active) or active < 0.0:
            raise ValueError("active_tolerance must be finite and non-negative.")
        self.line_search = search
        self.history_size = history
        self.curvature_tolerance = curvature
        self.active_tolerance = active
        self.project_initial = bool(project_initial)

    @property
    def method_id(self) -> str:
        return "projected-lbfgs"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return _bound_capabilities()

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        return _solve_bound_constrained(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )


def _bound_capabilities() -> OptimizationCapabilities:
    return OptimizationCapabilities(
        scalar_objective=True,
        residual_objective=False,
        matrix_free=True,
        prepared_refresh=False,
        implicit_differentiation=True,
    )


def _bound_layout_is_static(bounds: Bounds, /) -> bool:
    leaves = jax.tree.leaves(bounds.lower) + jax.tree.leaves(bounds.upper)
    return all(not isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _static_flat_bound(
    bound: Any,
    parameters: PyTree[Any],
    /,
    *,
    name: str,
) -> np.ndarray:
    """Materialize certificate layout without converting traced iterates."""

    parameter_structure = jax.tree.structure(parameters)
    parameter_leaves = jax.tree.leaves(parameters)
    if jax.tree.structure(bound) == parameter_structure:
        bound_leaves = jax.tree.leaves(bound)
        arrays = tuple(
            np.broadcast_to(
                np.asarray(bound_leaf, dtype=np.dtype(parameter.dtype)),
                parameter.shape,
            ).reshape(-1)
            for bound_leaf, parameter in zip(
                bound_leaves,
                parameter_leaves,
                strict=True,
            )
        )
    else:
        scalar = np.asarray(bound)
        if scalar.shape != ():
            raise ValueError(
                f"{name} must be scalar or have the parameter PyTree structure."
            )
        arrays = tuple(
            np.broadcast_to(
                scalar.astype(np.dtype(parameter.dtype)),
                parameter.shape,
            ).reshape(-1)
            for parameter in parameter_leaves
        )
    return np.concatenate(arrays)


def _bound_certificate(
    bounds: Bounds,
    parameters: PyTree[Any],
    gradient: PyTree[Any],
    /,
    *,
    active_tolerance: float,
) -> ConstrainedOptimalityCertificate:
    lower_numpy = _static_flat_bound(
        bounds.lower,
        parameters,
        name="lower",
    )
    upper_numpy = _static_flat_bound(
        bounds.upper,
        parameters,
        name="upper",
    )
    equal = (
        np.isfinite(lower_numpy) & np.isfinite(upper_numpy) & (lower_numpy == upper_numpy)
    )
    lower_indices = np.flatnonzero(np.isfinite(lower_numpy) & ~equal)
    upper_indices = np.flatnonzero(np.isfinite(upper_numpy) & ~equal)
    equality_indices = np.flatnonzero(equal)
    flat_parameters, unravel = ravel_pytree(parameters)
    flat_gradient, _ = ravel_pytree(gradient)
    lower, upper = bounds.materialize(parameters)
    flat_lower, _ = ravel_pytree(lower)
    flat_upper, _ = ravel_pytree(upper)
    equality_multipliers = -flat_gradient[equality_indices]
    lower_slacks = flat_parameters[lower_indices] - flat_lower[lower_indices]
    upper_slacks = flat_upper[upper_indices] - flat_parameters[upper_indices]
    lower_multipliers = jnp.where(
        lower_slacks <= active_tolerance,
        jnp.maximum(flat_gradient[lower_indices], 0.0),
        0.0,
    )
    upper_multipliers = jnp.where(
        upper_slacks <= active_tolerance,
        jnp.maximum(-flat_gradient[upper_indices], 0.0),
        0.0,
    )
    inequality_multipliers = jnp.concatenate((lower_multipliers, upper_multipliers))
    slacks = jnp.concatenate((lower_slacks, upper_slacks))
    stationarity = flat_gradient
    stationarity = stationarity.at[equality_indices].add(equality_multipliers)
    stationarity = stationarity.at[lower_indices].add(-lower_multipliers)
    stationarity = stationarity.at[upper_indices].add(upper_multipliers)
    primal_feasibility = bounds.violation(parameters)
    dual_feasibility = (
        jnp.max(jnp.maximum(-inequality_multipliers, 0.0))
        if inequality_multipliers.size
        else jnp.asarray(0.0, dtype=flat_parameters.dtype)
    )
    complementarity = (
        jnp.max(jnp.abs(inequality_multipliers * slacks))
        if inequality_multipliers.size
        else jnp.asarray(0.0, dtype=flat_parameters.dtype)
    )
    return ConstrainedOptimalityCertificate(
        equality_multipliers=equality_multipliers,
        inequality_multipliers=inequality_multipliers,
        slacks=slacks,
        active_mask=slacks <= active_tolerance,
        stationarity_residual=unravel(stationarity),
        primal_feasibility=primal_feasibility,
        dual_feasibility=dual_feasibility,
        complementarity=complementarity,
        equality_sources=tuple(f"bound:{index}:equality" for index in equality_indices),
        inequality_sources=tuple(
            [f"bound:{index}:lower" for index in lower_indices]
            + [f"bound:{index}:upper" for index in upper_indices]
        ),
    )


class _BoundState(NamedTuple):
    parameters: PyTree[Any]
    status: Array
    iterations: Array
    initial_optimality: Array
    final_step_norm: Array
    accepted_rate: Array
    accepted_steps: Array
    rejected_steps: Array
    objective_evaluations: Array
    gradient_evaluations: Array
    hvp_evaluations: Array
    linear_solves: Array
    linear_iterations: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    globalization_evaluations: Array
    direction_fallbacks: Array


class _LBFGSState(NamedTuple):
    previous_parameters: PyTree[Any]
    previous_gradient: PyTree[Any]
    previous_active: PyTree[Any]
    has_previous: Array
    steps: PyTree[Any]
    gradient_changes: PyTree[Any]
    inverse_curvatures: Array
    history_count: Array


def _lbfgs_direction(
    gradient: PyTree[Any],
    steps: PyTree[Any],
    gradient_changes: PyTree[Any],
    inverse_curvatures: Array,
    history_count: Array,
    /,
) -> PyTree[Array]:
    """Apply two-loop L-BFGS recursion over fixed-capacity staged buffers."""

    history_size = inverse_curvatures.shape[0]
    alphas = jnp.zeros_like(inverse_curvatures)

    def reverse_body(index, carry):
        q, stored_alphas = carry
        buffer_index = jnp.maximum(history_count - 1 - index, 0)
        step = jax.tree.map(lambda values: values[buffer_index], steps)
        change = jax.tree.map(
            lambda values: values[buffer_index],
            gradient_changes,
        )
        inverse_curvature = inverse_curvatures[buffer_index]
        active = index < history_count
        alpha = jnp.where(
            active,
            inverse_curvature * _tree_inner(step, q),
            0.0,
        )
        next_q = _tree_add_scaled(q, change, -alpha)
        return next_q, stored_alphas.at[index].set(alpha)

    q, alphas = jax.lax.fori_loop(
        0,
        history_size,
        reverse_body,
        (gradient, alphas),
    )
    latest_index = jnp.maximum(history_count - 1, 0)
    latest_step = jax.tree.map(lambda values: values[latest_index], steps)
    latest_change = jax.tree.map(
        lambda values: values[latest_index],
        gradient_changes,
    )
    scale = _tree_inner(latest_step, latest_change) / jnp.maximum(
        _tree_inner(latest_change, latest_change),
        1e-30,
    )
    scale = jnp.where(history_count > 0, scale, 1.0)
    initial_result = _tree_scale(scale, q)

    def forward_body(index, result):
        step = jax.tree.map(lambda values: values[index], steps)
        change = jax.tree.map(
            lambda values: values[index],
            gradient_changes,
        )
        active = index < history_count
        alpha_index = jnp.maximum(history_count - 1 - index, 0)
        alpha = jnp.where(active, alphas[alpha_index], 0.0)
        beta = jnp.where(
            active,
            inverse_curvatures[index] * _tree_inner(change, result),
            0.0,
        )
        return _tree_add_scaled(result, step, alpha - beta)

    result = jax.lax.fori_loop(
        0,
        history_size,
        forward_body,
        initial_result,
    )
    return _tree_negative(result)


def _append_lbfgs_history(
    state: _LBFGSState,
    step: PyTree[Any],
    change: PyTree[Any],
    inverse_curvature: Array,
    /,
) -> _LBFGSState:
    history_size = state.inverse_curvatures.shape[0]

    def append_open_slot(_):
        updated_steps = jax.tree.map(
            lambda values, new: values.at[state.history_count].set(new),
            state.steps,
            step,
        )
        updated_changes = jax.tree.map(
            lambda values, new: values.at[state.history_count].set(new),
            state.gradient_changes,
            change,
        )
        updated_inverse = state.inverse_curvatures.at[state.history_count].set(
            inverse_curvature
        )
        return updated_steps, updated_changes, updated_inverse, state.history_count + 1

    def replace_oldest(_):
        updated_steps = jax.tree.map(
            lambda values, new: jnp.concatenate(
                (values[1:], jnp.expand_dims(new, 0)),
                axis=0,
            ),
            state.steps,
            step,
        )
        updated_changes = jax.tree.map(
            lambda values, new: jnp.concatenate(
                (values[1:], jnp.expand_dims(new, 0)),
                axis=0,
            ),
            state.gradient_changes,
            change,
        )
        updated_inverse = jnp.concatenate(
            (
                state.inverse_curvatures[1:],
                jnp.reshape(inverse_curvature, (1,)),
            ),
            axis=0,
        )
        return (
            updated_steps,
            updated_changes,
            updated_inverse,
            state.history_count,
        )

    steps, changes, inverse, count = jax.lax.cond(
        state.history_count < history_size,
        append_open_slot,
        replace_oldest,
        None,
    )
    return state._replace(
        steps=steps,
        gradient_changes=changes,
        inverse_curvatures=inverse,
        history_count=count,
    )


def _active_set_newton_direction(
    method: ActiveSetNewton,
    parameters: PyTree[Any],
    gradient: PyTree[Any],
    linearized,
    active: PyTree[Any],
    /,
) -> tuple[PyTree[Array], Any, Array]:
    def hessian_action(vector):
        free_vector = _tree_zeros_on_mask(vector, active)
        _, hessian_vector = linearized(free_vector)
        free_hessian_vector = _tree_zeros_on_mask(hessian_vector, active)
        return jax.tree.map(
            lambda free_value, original, is_active: jnp.where(
                is_active,
                original,
                free_value,
            ),
            free_hessian_vector,
            vector,
            active,
        )

    space = PyTreeSpace(parameters)
    hessian = FunctionLinearOperator(
        hessian_action,
        source=space,
        target=space,
        transpose_action=hessian_action,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "asserted"},
        ),
        operator_id="active-set-reduced-hessian",
        closure_convert=False,
    )
    right_hand_side = _tree_negative(_tree_zeros_on_mask(gradient, active))
    linear_result = solve_linear(
        LinearSystem(hessian),
        right_hand_side,
        policy=method.linear_policy,
    )
    proposed = _tree_zeros_on_mask(linear_result.value, active)
    directional = _tree_inner(gradient, proposed)
    usable = (
        _usable_linear_status(linear_result.status)
        & _tree_allfinite(proposed)
        & jnp.isfinite(directional)
        & (directional < 0.0)
    )
    return proposed, linear_result, usable


def _initial_bound_state(
    parameters: PyTree[Any],
    initial_status: Any,
    scalar_dtype: Any,
    /,
) -> _BoundState:
    zero = jnp.asarray(0.0, dtype=scalar_dtype)
    zero_count = jnp.asarray(0, dtype=jnp.int32)
    return _BoundState(
        parameters=parameters,
        status=jnp.asarray(initial_status, dtype=jnp.int32),
        iterations=zero_count,
        initial_optimality=jnp.asarray(jnp.nan, dtype=scalar_dtype),
        final_step_norm=zero,
        accepted_rate=zero,
        accepted_steps=zero_count,
        rejected_steps=zero_count,
        objective_evaluations=zero_count,
        gradient_evaluations=zero_count,
        hvp_evaluations=zero_count,
        linear_solves=zero_count,
        linear_iterations=zero_count,
        setup_refreshes=zero_count,
        numeric_refreshes=zero_count,
        globalization_evaluations=zero_count,
        direction_fallbacks=zero_count,
    )


def _evaluate_bound_state(
    state: _BoundState,
    value: Array,
    gradient: PyTree[Any],
    bounds: Bounds,
    termination: OptimizationTermination,
    /,
) -> tuple[_BoundState, PyTree[Array], Array]:
    projected_gradient = bounds.projected_gradient(state.parameters, gradient)
    optimality = _tree_norm(projected_gradient)
    initial_optimality = jnp.where(
        state.objective_evaluations == 0,
        optimality,
        state.initial_optimality,
    )
    finite = (
        jnp.isfinite(value)
        & jnp.isfinite(optimality)
        & _tree_allfinite(state.parameters)
        & _tree_allfinite(gradient)
    )
    converged = optimality <= termination.optimality_threshold(initial_optimality)
    status = jnp.where(
        ~finite,
        int(OptimizationStatus.NONFINITE_EVALUATION),
        jnp.where(
            converged,
            int(OptimizationStatus.SUCCESS),
            int(OptimizationStatus.ITERATING),
        ),
    ).astype(jnp.int32)
    evaluated = state._replace(
        status=status,
        initial_optimality=initial_optimality,
        rejected_steps=state.rejected_steps + (~finite).astype(jnp.int32),
        objective_evaluations=state.objective_evaluations + 1,
        gradient_evaluations=state.gradient_evaluations + 1,
    )
    return evaluated, projected_gradient, optimality


def _take_bound_step(
    state: _BoundState,
    value_function,
    value: Array,
    gradient: PyTree[Any],
    projected_gradient: PyTree[Any],
    raw_direction: PyTree[Any],
    usable: Array,
    bounds: Bounds,
    line_search: ArmijoLineSearch,
    termination: OptimizationTermination,
    /,
) -> _BoundState:
    proposed_direction = _projected_displacement(
        bounds,
        state.parameters,
        raw_direction,
    )
    proposed_directional = _tree_inner(gradient, proposed_direction)
    usable_direction = (
        jnp.asarray(usable, dtype=bool)
        & _tree_allfinite(proposed_direction)
        & jnp.isfinite(proposed_directional)
        & (proposed_directional < 0.0)
    )
    fallback_direction = _tree_negative(projected_gradient)
    fallback_directional = _tree_inner(gradient, fallback_direction)
    direction = _tree_where(
        usable_direction,
        proposed_direction,
        fallback_direction,
    )
    directional = jnp.where(
        usable_direction,
        proposed_directional,
        fallback_directional,
    )
    prepared = state._replace(
        direction_fallbacks=(
            state.direction_fallbacks + (~usable_direction).astype(jnp.int32)
        )
    )
    valid_direction = (
        _tree_allfinite(direction) & jnp.isfinite(directional) & (directional < 0.0)
    )

    def invalid_direction(_):
        return prepared._replace(
            status=jnp.asarray(
                int(OptimizationStatus.INVALID_DIRECTION),
                dtype=jnp.int32,
            ),
            rejected_steps=prepared.rejected_steps + 1,
        )

    def line_search_step(_):
        search = armijo_backtracking(
            value_function,
            prepared.parameters,
            value,
            direction,
            directional,
            step=_tree_add_scaled,
            contains=lambda candidate: (
                bounds.contains(candidate) & _tree_allfinite(candidate)
            ),
            policy=line_search,
        )
        objective_evaluations = prepared.objective_evaluations + search.evaluations
        exhausted = (
            jnp.asarray(False)
            if termination.maximum_evaluations is None
            else objective_evaluations >= termination.maximum_evaluations
        )
        final_step_norm = search.rate * _tree_norm(direction)
        stagnated = final_step_norm <= termination.step_threshold(
            _tree_norm(search.parameters)
        )
        status = jnp.where(
            ~search.accepted,
            jnp.where(
                search.finite_candidate_seen,
                int(OptimizationStatus.LINE_SEARCH_FAILED),
                int(OptimizationStatus.NONFINITE_EVALUATION),
            ),
            jnp.where(
                stagnated,
                int(OptimizationStatus.STAGNATION),
                jnp.where(
                    exhausted,
                    int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                    int(OptimizationStatus.ITERATING),
                ),
            ),
        ).astype(jnp.int32)
        return prepared._replace(
            parameters=search.parameters,
            status=status,
            iterations=prepared.iterations + 1,
            final_step_norm=final_step_norm,
            accepted_rate=search.rate,
            accepted_steps=(prepared.accepted_steps + search.accepted.astype(jnp.int32)),
            rejected_steps=(
                prepared.rejected_steps + (~search.accepted).astype(jnp.int32)
            ),
            objective_evaluations=objective_evaluations,
            globalization_evaluations=(
                prepared.globalization_evaluations + search.evaluations
            ),
        )

    return jax.lax.cond(
        valid_direction,
        line_search_step,
        invalid_direction,
        None,
    )


def _run_bound_iterations(
    initial_state: _BoundState,
    initial_method_state: Any,
    iteration_body,
    termination: OptimizationTermination,
    /,
) -> _BoundState:
    def condition(carry):
        state, _ = carry
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else state.objective_evaluations < termination.maximum_evaluations
        )
        return (
            (state.status == int(OptimizationStatus.ITERATING))
            & (state.iterations < termination.maximum_steps)
            & within_evaluations
        )

    state, _ = jax.lax.while_loop(
        condition,
        iteration_body,
        (initial_state, initial_method_state),
    )
    return state._replace(
        status=jnp.where(
            state.status == int(OptimizationStatus.ITERATING),
            int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
            state.status,
        ).astype(jnp.int32)
    )


def _run_projected_gradient(
    method: ProjectedGradient,
    value_function,
    bounds: Bounds,
    initial_state: _BoundState,
    termination: OptimizationTermination,
    /,
) -> _BoundState:
    value_and_gradient = jax.value_and_grad(value_function)

    def iteration_body(carry):
        state, method_state = carry
        value, gradient = value_and_gradient(state.parameters)
        evaluated, projected_gradient, _ = _evaluate_bound_state(
            state,
            value,
            gradient,
            bounds,
            termination,
        )

        def take_step(_):
            return _take_bound_step(
                evaluated,
                value_function,
                value,
                gradient,
                projected_gradient,
                _tree_negative(gradient),
                jnp.asarray(True),
                bounds,
                method.line_search,
                termination,
            )

        next_state = jax.lax.cond(
            evaluated.status == int(OptimizationStatus.ITERATING),
            take_step,
            lambda _: evaluated,
            None,
        )
        return next_state, method_state

    return _run_bound_iterations(
        initial_state,
        (),
        iteration_body,
        termination,
    )


def _run_active_set_newton(
    method: ActiveSetNewton,
    value_function,
    bounds: Bounds,
    initial_state: _BoundState,
    termination: OptimizationTermination,
    /,
) -> _BoundState:
    value_and_gradient = jax.value_and_grad(value_function)

    def iteration_body(carry):
        state, method_state = carry
        (value, gradient), linearized = jax.linearize(
            value_and_gradient,
            state.parameters,
        )
        evaluated, projected_gradient, _ = _evaluate_bound_state(
            state,
            value,
            gradient,
            bounds,
            termination,
        )

        def take_step(_):
            active = bounds.active_mask(
                evaluated.parameters,
                gradient,
                tolerance=method.active_tolerance,
            )
            raw_direction, linear_result, usable = _active_set_newton_direction(
                method,
                evaluated.parameters,
                gradient,
                linearized,
                active,
            )
            prepared = evaluated._replace(
                hvp_evaluations=(
                    evaluated.hvp_evaluations
                    + jnp.asarray(
                        linear_result.diagnostics.matvec_count,
                        dtype=jnp.int32,
                    )
                ),
                linear_solves=evaluated.linear_solves + 1,
                linear_iterations=(
                    evaluated.linear_iterations
                    + jnp.asarray(
                        linear_result.diagnostics.iterations,
                        dtype=jnp.int32,
                    )
                ),
                setup_refreshes=evaluated.setup_refreshes + 1,
                numeric_refreshes=evaluated.numeric_refreshes + 1,
            )
            return _take_bound_step(
                prepared,
                value_function,
                value,
                gradient,
                projected_gradient,
                raw_direction,
                usable,
                bounds,
                method.line_search,
                termination,
            )

        next_state = jax.lax.cond(
            evaluated.status == int(OptimizationStatus.ITERATING),
            take_step,
            lambda _: evaluated,
            None,
        )
        return next_state, method_state

    return _run_bound_iterations(
        initial_state,
        (),
        iteration_body,
        termination,
    )


def _initial_lbfgs_state(
    parameters: PyTree[Any],
    history_size: int,
    /,
) -> _LBFGSState:
    zero_gradient = jax.tree.map(jnp.zeros_like, parameters)
    inactive = jax.tree.map(
        lambda value: jnp.zeros(value.shape, dtype=bool),
        parameters,
    )
    steps = jax.tree.map(
        lambda value: jnp.zeros((history_size, *value.shape), dtype=value.dtype),
        parameters,
    )
    return _LBFGSState(
        previous_parameters=parameters,
        previous_gradient=zero_gradient,
        previous_active=inactive,
        has_previous=jnp.asarray(False),
        steps=steps,
        gradient_changes=steps,
        inverse_curvatures=jnp.zeros(
            (history_size,),
            dtype=_tree_inner(parameters, parameters).dtype,
        ),
        history_count=jnp.asarray(0, dtype=jnp.int32),
    )


def _update_lbfgs_history(
    method: ProjectedLBFGS,
    parameters: PyTree[Any],
    gradient: PyTree[Any],
    active: PyTree[Any],
    state: _LBFGSState,
    /,
) -> _LBFGSState:
    same_active = (~state.has_previous) | _tree_mask_equal(
        active,
        state.previous_active,
    )
    empty_steps = jax.tree.map(jnp.zeros_like, state.steps)
    retained = state._replace(
        steps=_tree_where(same_active, state.steps, empty_steps),
        gradient_changes=_tree_where(
            same_active,
            state.gradient_changes,
            empty_steps,
        ),
        inverse_curvatures=jnp.where(
            same_active,
            state.inverse_curvatures,
            jnp.zeros_like(state.inverse_curvatures),
        ),
        history_count=jnp.where(
            same_active,
            state.history_count,
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    step = jax.tree.map(
        lambda current, previous: current - previous,
        parameters,
        state.previous_parameters,
    )
    change = jax.tree.map(
        lambda current, previous: current - previous,
        gradient,
        state.previous_gradient,
    )
    curvature = _tree_inner(step, change)
    threshold = method.curvature_tolerance * _tree_norm(step) * _tree_norm(change)
    add_pair = (
        state.has_previous
        & same_active
        & jnp.isfinite(curvature)
        & (curvature > threshold)
    )
    return jax.lax.cond(
        add_pair,
        lambda current: _append_lbfgs_history(
            current,
            step,
            change,
            1.0 / curvature,
        ),
        lambda current: current,
        retained,
    )


def _run_projected_lbfgs(
    method: ProjectedLBFGS,
    value_function,
    bounds: Bounds,
    initial_state: _BoundState,
    termination: OptimizationTermination,
    /,
) -> _BoundState:
    value_and_gradient = jax.value_and_grad(value_function)
    initial_history = _initial_lbfgs_state(
        initial_state.parameters,
        method.history_size,
    )

    def iteration_body(carry):
        state, history = carry
        value, gradient = value_and_gradient(state.parameters)
        evaluated, projected_gradient, _ = _evaluate_bound_state(
            state,
            value,
            gradient,
            bounds,
            termination,
        )

        def take_step(_):
            active = bounds.active_mask(
                evaluated.parameters,
                gradient,
                tolerance=method.active_tolerance,
            )
            updated_history = _update_lbfgs_history(
                method,
                evaluated.parameters,
                gradient,
                active,
                history,
            )
            raw_direction = _lbfgs_direction(
                gradient,
                updated_history.steps,
                updated_history.gradient_changes,
                updated_history.inverse_curvatures,
                updated_history.history_count,
            )
            next_state = _take_bound_step(
                evaluated,
                value_function,
                value,
                gradient,
                projected_gradient,
                raw_direction,
                jnp.asarray(True),
                bounds,
                method.line_search,
                termination,
            )
            accepted = next_state.accepted_steps > evaluated.accepted_steps
            accepted_history = updated_history._replace(
                previous_parameters=evaluated.parameters,
                previous_gradient=gradient,
                previous_active=active,
                has_previous=jnp.asarray(True),
            )
            next_history = _tree_where(
                accepted,
                accepted_history,
                history,
            )
            return next_state, next_history

        return jax.lax.cond(
            evaluated.status == int(OptimizationStatus.ITERATING),
            take_step,
            lambda _: (evaluated, history),
            None,
        )

    return _run_bound_iterations(
        initial_state,
        initial_history,
        iteration_body,
        termination,
    )


def _solve_bound_constrained(
    method: ProjectedGradient | ActiveSetNewton | ProjectedLBFGS,
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
) -> MinimizationResult:
    if not isinstance(problem, MinimizationProblem):
        raise TypeError("problem must be a MinimizationProblem.")
    if problem.bounds is None:
        raise ValueError("A bound-constrained method requires problem.bounds.")
    if problem.constraints:
        raise ValueError(
            "Bound-constrained methods do not accept general nonlinear constraints."
        )
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")

    bounds = problem.bounds
    parameters = _validate_real_inexact_tree(initial_parameters, name="parameters")
    PyTreeSpace(parameters)
    if method.project_initial:
        parameters = bounds.project(parameters)
        initial_status = jnp.asarray(
            int(OptimizationStatus.ITERATING),
            dtype=jnp.int32,
        )
    else:
        initial_feasible = bounds.contains(parameters)
        if (
            not isinstance(initial_feasible, jax.core.Tracer)
            and not np.asarray(initial_feasible).item()
        ):
            return _infeasible_result(method, problem, parameters)
        initial_status = jnp.where(
            initial_feasible,
            int(OptimizationStatus.ITERATING),
            int(OptimizationStatus.INFEASIBLE),
        ).astype(jnp.int32)

    def value_function(candidate):
        return problem.value(candidate, args)[0]

    abstract_value = jax.eval_shape(value_function, parameters)
    scalar_dtype = jnp.result_type(
        abstract_value.dtype,
        *(leaf.dtype for leaf in jax.tree.leaves(parameters)),
        float,
    )
    initial_state = _initial_bound_state(
        parameters,
        initial_status,
        scalar_dtype,
    )

    # Dispatch is static: no method object or per-iteration callable enters the
    # staged carry.
    if isinstance(method, ProjectedGradient):
        state = _run_projected_gradient(
            method,
            value_function,
            bounds,
            initial_state,
            termination,
        )
        active_tolerance = 1e-10
    elif isinstance(method, ActiveSetNewton):
        state = _run_active_set_newton(
            method,
            value_function,
            bounds,
            initial_state,
            termination,
        )
        active_tolerance = method.active_tolerance
    elif isinstance(method, ProjectedLBFGS):
        state = _run_projected_lbfgs(
            method,
            value_function,
            bounds,
            initial_state,
            termination,
        )
        active_tolerance = method.active_tolerance
    else:
        raise TypeError("Unsupported bound-constrained minimization method.")

    (final_value, auxiliary), final_gradient = problem.value_and_gradient(
        state.parameters,
        args,
    )
    objective_evaluations = state.objective_evaluations + 1
    gradient_evaluations = state.gradient_evaluations + 1
    final_projected_gradient = bounds.projected_gradient(
        state.parameters,
        final_gradient,
    )
    final_optimality = _tree_norm(final_projected_gradient)
    eligible_for_final_success = (
        (state.status == int(OptimizationStatus.ITERATING))
        | (state.status == int(OptimizationStatus.MAXIMUM_STEPS_REACHED))
        | (state.status == int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED))
        | (state.status == int(OptimizationStatus.STAGNATION))
    )
    status = jnp.where(
        eligible_for_final_success
        & (
            final_optimality <= termination.optimality_threshold(state.initial_optimality)
        ),
        int(OptimizationStatus.SUCCESS),
        state.status,
    ).astype(jnp.int32)
    final_active = bounds.active_mask(
        state.parameters,
        final_gradient,
        tolerance=active_tolerance,
    )
    diagnostics = OptimizationDiagnostics(
        iterations=state.iterations,
        accepted_steps=state.accepted_steps,
        rejected_steps=state.rejected_steps,
        objective_evaluations=objective_evaluations,
        gradient_evaluations=gradient_evaluations,
        hvp_evaluations=state.hvp_evaluations,
        linear_solves=state.linear_solves,
        linear_iterations=state.linear_iterations,
        setup_refreshes=state.setup_refreshes,
        numeric_refreshes=state.numeric_refreshes,
        globalization_evaluations=state.globalization_evaluations,
        initial_optimality_norm=state.initial_optimality,
        final_optimality_norm=final_optimality,
        final_step_norm=state.final_step_norm,
        accepted_step_size=state.accepted_rate,
        direction_fallbacks=state.direction_fallbacks,
        primal_feasibility=bounds.violation(state.parameters),
        dual_feasibility=final_optimality,
        complementarity=_complementarity_residual(
            bounds,
            state.parameters,
            final_gradient,
            tolerance=active_tolerance,
        ),
        active_constraints=_active_count(final_active),
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax",
        globalization="projected-armijo",
        matrix_free=True,
        implicit_differentiation=True,
        notes="Feasible iterates maintained by projection.",
    )
    certificate = (
        _bound_certificate(
            bounds,
            state.parameters,
            final_gradient,
            active_tolerance=active_tolerance,
        )
        if _bound_layout_is_static(bounds)
        else None
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


def _infeasible_result(
    method: ProjectedGradient | ActiveSetNewton | ProjectedLBFGS,
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    /,
) -> MinimizationResult:
    assert problem.bounds is not None
    diagnostics = OptimizationDiagnostics(
        primal_feasibility=problem.bounds.violation(parameters),
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax",
        globalization="projected-armijo",
        matrix_free=True,
        implicit_differentiation=True,
        notes="Initial point rejected by the feasibility policy.",
    )
    return MinimizationResult(
        parameters,
        jnp.asarray(jnp.nan),
        None,
        OptimizationStatus.INFEASIBLE,
        diagnostics,
        provenance,
    )


__all__ = ["ActiveSetNewton", "ProjectedGradient", "ProjectedLBFGS"]
