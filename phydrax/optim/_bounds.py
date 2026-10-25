#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from math import isfinite, isnan
from typing import Any, final, NamedTuple, Protocol, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike, DTypeLike
from jaxtyping import PyTree

from .._dtype_names import inexact_result_type
from .._strict import StrictModule
from .._tree_math import (
    tree_add_scaled as _tree_add_scaled,
    tree_allfinite as _tree_allfinite,
    tree_inner as _tree_inner,
    tree_negative as _tree_negative,
    tree_norm as _tree_norm,
    tree_where as _tree_where,
    validate_real_inexact_tree as _validate_real_inexact_tree,
)
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
from ..typing import Bool, Float, Int32
from ._group_layout import OptimizerGroupDim, OptimizerGroupLayout
from ._iterative._base import AbstractMinimizationMethod
from ._iterative._globalization import armijo_backtracking, ArmijoLineSearch
from ._iterative._types import (
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
from ._lbfgs_kernel import (
    admit_lbfgs_pair,
    append_lbfgs_pair,
    initial_lbfgs_history,
    lbfgs_direction,
    lbfgs_pair_curvature,
    LBFGSHistory,
    reset_lbfgs_history,
)
from ._trust_region import (
    solve_trust_region_subproblem,
    SteihaugToint,
    TrustRegionQuadraticProblem,
)


_SINGLE_GROUP = OptimizerGroupLayout()
_ITERATING = int(OptimizationStatus.ITERATING)
_SUCCESS = int(OptimizationStatus.SUCCESS)
_NONFINITE_EVALUATION = int(OptimizationStatus.NONFINITE_EVALUATION)


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
        # Infinite bounds are inactive; never form inf - inf arithmetic for them.
        lower_finite = jnp.isfinite(lo)
        upper_finite = jnp.isfinite(hi)
        lo_ = jnp.where(lower_finite, lo, 0.0)
        hi_ = jnp.where(upper_finite, hi, 0.0)
        lower_active = lower_finite & (value <= lo_ + tolerance * (1.0 + jnp.abs(lo_)))
        upper_active = upper_finite & (value >= hi_ - tolerance * (1.0 + jnp.abs(hi_)))
        lower_multiplier = jnp.where(lower_active, jnp.maximum(grad, 0.0), 0.0)
        upper_multiplier = jnp.where(upper_active, jnp.maximum(-grad, 0.0), 0.0)
        lower_gap = jnp.where(lower_finite, value - lo_, 0.0)
        upper_gap = jnp.where(upper_finite, hi_ - value, 0.0)
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
    ) -> None:
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
    ) -> None:
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


class BoundedNewtonTrustRegion(AbstractMinimizationMethod):
    """Projected active-set Newton method with a matrix-free trust region."""

    subproblem: SteihaugToint
    initial_radius: float = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)
    acceptance_ratio: float = eqx.field(static=True)
    shrink_ratio: float = eqx.field(static=True)
    expansion_ratio: float = eqx.field(static=True)
    shrink_factor: float = eqx.field(static=True)
    expansion_factor: float = eqx.field(static=True)
    active_tolerance: float = eqx.field(static=True)
    project_initial: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        subproblem: SteihaugToint | None = None,
        initial_radius: float = 1.0,
        maximum_radius: float = 1e4,
        minimum_radius: float = 1e-12,
        acceptance_ratio: float = 1e-4,
        shrink_ratio: float = 0.25,
        expansion_ratio: float = 0.75,
        shrink_factor: float = 0.25,
        expansion_factor: float = 2.0,
        active_tolerance: float = 1e-10,
        project_initial: bool = True,
    ) -> None:
        subproblem_ = SteihaugToint() if subproblem is None else subproblem
        if not isinstance(subproblem_, SteihaugToint):
            raise TypeError("subproblem must be SteihaugToint or None.")
        values = tuple(
            float(value)
            for value in (
                initial_radius,
                maximum_radius,
                minimum_radius,
                acceptance_ratio,
                shrink_ratio,
                expansion_ratio,
                shrink_factor,
                expansion_factor,
                active_tolerance,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values[:-1]):
            raise ValueError("Bounded trust-region controls must be positive and finite.")
        if not isfinite(values[-1]) or values[-1] < 0.0:
            raise ValueError("active_tolerance must be finite and non-negative.")
        if not values[2] <= values[0] <= values[1]:
            raise ValueError(
                "Radii must satisfy minimum_radius <= initial_radius <= maximum_radius."
            )
        if not 0.0 < values[3] < values[4] < values[5] < 1.0:
            raise ValueError("Ratios must satisfy acceptance < shrink < expansion < one.")
        if not 0.0 < values[6] < 1.0 or values[7] <= 1.0:
            raise ValueError("Radius factors must shrink below and expand above one.")
        self.subproblem = subproblem_
        (
            self.initial_radius,
            self.maximum_radius,
            self.minimum_radius,
            self.acceptance_ratio,
            self.shrink_ratio,
            self.expansion_ratio,
            self.shrink_factor,
            self.expansion_factor,
            self.active_tolerance,
        ) = values
        self.project_initial = bool(project_initial)

    @property
    def method_id(self) -> str:
        return "bounded-newton-trust-region/steihaug-toint"

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
    ) -> None:
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


@final
class TrustScaledLBFGSState(StrictModule):
    """Complete stepwise state of `TrustScaledLBFGS` for every optimizer group.

    ``parameters`` is the point the caller evaluates next. ``accepted_*`` holds the
    last accepted evaluation (point, gradient, bound activity and value) from which
    ``parameters`` was proposed; it is the reported iterate, so its value and
    gradient always belong to one evaluated point. ``direction``,
    ``step_length``, ``directional_derivative`` and ``step_norm`` describe the
    latest step a group took; the boolean flags describe the latest evaluation a
    group consumed; the integer fields count events over the run. ``status`` is
    an `OptimizationStatus` per group: ``ITERATING``, ``SUCCESS`` (the accepted
    evaluation satisfied convergence and the group did not move) or
    ``NONFINITE_EVALUATION`` (the evaluation was rejected and the group returned
    to its accepted point).
    """

    __strict_contract__ = True

    parameters: PyTree[Array]
    accepted_parameters: PyTree[Array]
    accepted_gradient: PyTree[Array]
    accepted_activity: PyTree[Array]
    accepted_value: Float[OptimizerGroupDim]
    has_accepted: Bool[OptimizerGroupDim]
    history: LBFGSHistory
    direction: PyTree[Array]
    step_length: Float[OptimizerGroupDim]
    directional_derivative: Float[OptimizerGroupDim]
    optimality: Float[OptimizerGroupDim]
    maximum_site_gradient: Float[OptimizerGroupDim]
    initial_optimality: Float[OptimizerGroupDim]
    step_norm: Float[OptimizerGroupDim]
    status: Int32[OptimizerGroupDim]
    evaluation_accepted: Bool[OptimizerGroupDim]
    curvature_rejected: Bool[OptimizerGroupDim]
    direction_reset: Bool[OptimizerGroupDim]
    trust_limited: Bool[OptimizerGroupDim]
    bound_limited: Bool[OptimizerGroupDim]
    evaluations: Int32[OptimizerGroupDim]
    iterations: Int32[OptimizerGroupDim]
    curvature_rejections: Int32[OptimizerGroupDim]
    direction_resets: Int32[OptimizerGroupDim]
    trust_limited_steps: Int32[OptimizerGroupDim]
    nonfinite_evaluations: Int32[OptimizerGroupDim]
    layout: OptimizerGroupLayout

    @property
    def accepted_step_size(self) -> Array:
        return self.step_length

    @property
    def damping(self) -> Array:
        return jnp.full_like(self.step_length, jnp.nan)

    @property
    def direction_fallbacks(self) -> Array:
        return self.direction_resets

    @property
    def rejected_steps(self) -> Array:
        return self.nonfinite_evaluations


@final
class TrustScaledLBFGS(AbstractMinimizationMethod):
    """Force-driven limited-memory BFGS with a trust-scaled step and no line search.

    Every finite evaluation is an accepted point. It forms one secant pair with the
    previous accepted point of its group, and the two-loop direction
    ``d = -H g`` is taken with step length ``min(1, maximum_step / max_site |d|)``,
    so the largest site displacement never exceeds ``maximum_step``. No energy is
    compared: the method does not claim monotone decrease, and `ProjectedLBFGS`
    remains the line-search method for a true scalar objective.

    A pair is admitted only when ``s·y > curvature_tolerance |s| |y|`` (a weak or
    negative curvature pair is rejected and counted); without an explicit
    tolerance the floor is ``1e-6`` for single-precision and ``1e-10`` for wider
    parameters. A two-loop direction that is not a descent direction
    (``g·d >= 0``) discards the group's history and the group restarts along the
    unit steepest-descent direction, as does a group without history.

    With bounds, activity follows `Bounds.active_mask`, the direction is
    restricted to free variables, a change of active set forgets that group's
    history, and the trust-scaled trial point is projected onto the bounds.
    """

    history_size: int = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    curvature_tolerance: float | None = eqx.field(static=True)
    active_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        history_size: int = 6,
        maximum_step: float = 0.2,
        curvature_tolerance: float | None = None,
        active_tolerance: float = 1e-10,
    ) -> None:
        if isinstance(history_size, bool) or not isinstance(history_size, int):
            raise TypeError("history_size must be an int.")
        if history_size < 1:
            raise ValueError("history_size must be positive.")
        step = float(maximum_step)
        if isnan(step) or step <= 0.0:
            raise ValueError(
                "maximum_step must be positive; inf disables the trust scale."
            )
        curvature = None if curvature_tolerance is None else float(curvature_tolerance)
        if curvature is not None and (not isfinite(curvature) or curvature < 0.0):
            raise ValueError("curvature_tolerance must be finite and non-negative.")
        active = float(active_tolerance)
        if not isfinite(active) or active < 0.0:
            raise ValueError("active_tolerance must be finite and non-negative.")
        self.history_size = history_size
        self.maximum_step = step
        self.curvature_tolerance = curvature
        self.active_tolerance = active

    @property
    def method_id(self) -> str:
        return "trust-scaled-lbfgs"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return _bound_capabilities()

    def init(
        self,
        parameters: PyTree[ArrayLike],
        /,
        *,
        layout: OptimizerGroupLayout | None = None,
        value_dtype: DTypeLike | None = None,
    ) -> TrustScaledLBFGSState:
        """Prepare a state whose ``parameters`` are the first point to evaluate."""

        parameters_, layout_, dtype, value_dtype_ = _force_driven_admission(
            parameters, layout, value_dtype
        )
        groups = layout_.group_count
        zero = jnp.zeros((groups,), dtype=dtype)
        unset = jnp.full((groups,), jnp.nan, dtype=dtype)
        no = jnp.zeros((groups,), dtype=jnp.bool_)
        none = jnp.zeros((groups,), dtype=jnp.int32)
        zero_tree = jax.tree.map(jnp.zeros_like, parameters_)
        return TrustScaledLBFGSState(
            parameters=parameters_,
            accepted_parameters=parameters_,
            accepted_gradient=zero_tree,
            accepted_activity=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, dtype=jnp.bool_), parameters_
            ),
            accepted_value=jnp.full((groups,), jnp.nan, dtype=value_dtype_),
            has_accepted=no,
            history=initial_lbfgs_history(parameters_, self.history_size, layout_, dtype),
            direction=zero_tree,
            step_length=zero,
            directional_derivative=unset,
            optimality=unset,
            maximum_site_gradient=unset,
            initial_optimality=unset,
            step_norm=zero,
            status=jnp.full((groups,), _ITERATING, dtype=jnp.int32),
            evaluation_accepted=no,
            curvature_rejected=no,
            direction_reset=no,
            trust_limited=no,
            bound_limited=no,
            evaluations=none,
            iterations=none,
            curvature_rejections=none,
            direction_resets=none,
            trust_limited_steps=none,
            nonfinite_evaluations=none,
            layout=layout_,
        )

    def step(
        self,
        state: TrustScaledLBFGSState,
        value: ArrayLike,
        gradient: PyTree[ArrayLike],
        /,
        *,
        bounds: Bounds | None = None,
        termination: OptimizationTermination | None = None,
        converged: ArrayLike | None = None,
        active: ArrayLike | None = None,
    ) -> TrustScaledLBFGSState:
        """Consume one evaluation at ``state.parameters`` and propose the next point.

        ``value`` holds one objective value per group and ``gradient`` the
        gradient at ``state.parameters``. Groups outside ``active`` are left
        unchanged. An active group whose evaluation is finite becomes accepted; it
        stays in place with status ``SUCCESS`` when ``converged`` selects it or its
        projected-gradient norm meets ``termination``, and otherwise takes one
        trust-scaled step. A nonfinite evaluation returns the group to its accepted
        point with status ``NONFINITE_EVALUATION``.
        """

        if not isinstance(state, TrustScaledLBFGSState):
            raise TypeError("state must be a TrustScaledLBFGSState.")
        layout = state.layout
        x = state.parameters
        value_, gradient_, active_, converged_ = _force_driven_inputs(
            layout, x, state.accepted_value.dtype, value, gradient, active, converged
        )
        evaluation = _evaluate_force_driven(
            layout, bounds, x, value_, gradient_, self.active_tolerance
        )
        activity = (
            state.accepted_activity
            if evaluation.activity is None
            else evaluation.activity
        )
        accepted = active_ & evaluation.finite
        initial_optimality = jnp.where(
            accepted & ~state.has_accepted,
            evaluation.optimality,
            state.initial_optimality,
        )
        converged_ = _group_converged(
            termination, evaluation.optimality, initial_optimality, converged_
        )
        moving = accepted & ~converged_

        dtype = state.step_length.dtype
        same_activity = (
            layout.count_true(
                jax.tree.map(jnp.not_equal, activity, state.accepted_activity)
            )
            == 0
        )
        continuing = moving & state.has_accepted
        pair_step = jax.tree.map(jnp.subtract, x, state.accepted_parameters)
        pair_change = jax.tree.map(jnp.subtract, gradient_, state.accepted_gradient)
        curvature, step_squared, change_squared = lbfgs_pair_curvature(
            pair_step, pair_change, layout
        )
        formed = continuing & same_activity
        admitted = formed & admit_lbfgs_pair(
            curvature,
            step_squared,
            change_squared,
            _curvature_floor(self.curvature_tolerance, dtype),
        )
        history = reset_lbfgs_history(state.history, continuing & ~same_activity)
        history = append_lbfgs_pair(
            history, pair_step, pair_change, curvature, change_squared, admitted, layout
        )

        free_gradient = evaluation.free_gradient
        direction = lbfgs_direction(free_gradient, history, layout)
        if evaluation.activity is not None:
            direction = _tree_zeros_on_mask(direction, activity)
        slope = layout.inner(free_gradient, direction).astype(dtype)
        descent = (history.count > 0) & jnp.isfinite(slope) & (slope < 0.0)
        restart = moving & ~descent
        direction_reset = restart & (history.count > 0)
        history = reset_lbfgs_history(history, restart)
        gradient_norm = layout.norm(free_gradient).astype(dtype)
        has_force = gradient_norm > 0.0
        inverse_norm = jnp.where(
            has_force, 1.0 / jnp.where(has_force, gradient_norm, 1.0), 0.0
        )
        steepest = layout.scale(-inverse_norm, free_gradient)
        direction = layout.select(restart, steepest, direction)
        slope = jnp.where(restart, -gradient_norm, slope)

        largest = layout.site_norm_max(direction).astype(dtype)
        trust_limited = moving & (largest > self.maximum_step)
        step_length = jnp.where(
            trust_limited,
            self.maximum_step / jnp.where(trust_limited, largest, 1.0),
            jnp.ones_like(largest),
        ).astype(dtype)
        trial = jax.tree.map(jnp.add, x, layout.scale(step_length, direction))
        trial, bound_limited = _admit_trial(layout, bounds, trial)
        bound_limited = moving & bound_limited

        retreat = active_ & ~evaluation.finite & state.has_accepted
        held = layout.select(retreat, state.accepted_parameters, x)
        proposal = layout.select(moving, trial, held)
        step_norm = layout.norm(jax.tree.map(jnp.subtract, trial, x)).astype(dtype)
        curvature_rejected = formed & ~admitted
        return dataclasses.replace(
            state,
            parameters=proposal,
            accepted_parameters=layout.select(accepted, x, state.accepted_parameters),
            accepted_gradient=layout.select(accepted, gradient_, state.accepted_gradient),
            accepted_activity=layout.select(accepted, activity, state.accepted_activity),
            accepted_value=jnp.where(accepted, value_, state.accepted_value),
            has_accepted=state.has_accepted | accepted,
            history=history,
            direction=layout.select(moving, direction, state.direction),
            step_length=jnp.where(moving, step_length, state.step_length),
            directional_derivative=jnp.where(moving, slope, state.directional_derivative),
            optimality=jnp.where(accepted, evaluation.optimality, state.optimality),
            maximum_site_gradient=jnp.where(
                accepted,
                evaluation.maximum_site_gradient,
                state.maximum_site_gradient,
            ),
            initial_optimality=initial_optimality,
            step_norm=jnp.where(moving, step_norm, state.step_norm),
            status=_group_status(state.status, active_, evaluation.finite, converged_),
            evaluation_accepted=jnp.where(active_, accepted, state.evaluation_accepted),
            curvature_rejected=jnp.where(
                active_, curvature_rejected, state.curvature_rejected
            ),
            direction_reset=jnp.where(active_, direction_reset, state.direction_reset),
            trust_limited=jnp.where(active_, trust_limited, state.trust_limited),
            bound_limited=jnp.where(active_, bound_limited, state.bound_limited),
            evaluations=state.evaluations + active_.astype(jnp.int32),
            iterations=state.iterations + moving.astype(jnp.int32),
            curvature_rejections=(
                state.curvature_rejections + curvature_rejected.astype(jnp.int32)
            ),
            direction_resets=state.direction_resets + direction_reset.astype(jnp.int32),
            trust_limited_steps=(
                state.trust_limited_steps + trust_limited.astype(jnp.int32)
            ),
            nonfinite_evaluations=(
                state.nonfinite_evaluations
                + (active_ & ~evaluation.finite).astype(jnp.int32)
            ),
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
        return _solve_force_driven(
            self,
            problem,
            initial_parameters,
            termination=termination,
            args=args,
            globalization="displacement-trust-scale",
            notes=(
                "Force-driven step without line search; objective decrease is not "
                "enforced. Reported iterate is the last accepted evaluation."
            ),
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
    return all(not isinstance(leaf, core.Tracer) for leaf in leaves)


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
    trust_radius: Array
    reduction_ratio: Array


class _LBFGSState(NamedTuple):
    previous_parameters: PyTree[Any]
    previous_gradient: PyTree[Any]
    previous_active: PyTree[Any]
    has_previous: Array
    history: LBFGSHistory


_ValueFunction: TypeAlias = Callable[[PyTree[Any]], Array]
_Linearized: TypeAlias = Callable[[PyTree[Any]], tuple[Array, PyTree[Array]]]
_MethodState = TypeVar("_MethodState")
_EmptyCarry: TypeAlias = tuple[_BoundState, tuple[()]]
_LBFGSCarry: TypeAlias = tuple[_BoundState, _LBFGSState]


def _active_set_newton_direction(
    method: ActiveSetNewton,
    parameters: PyTree[Any],
    gradient: PyTree[Any],
    linearized: _Linearized,
    active: PyTree[Any],
    /,
) -> tuple[PyTree[Array], Any, Array]:
    def hessian_action(vector: PyTree[Any]) -> PyTree[Array]:
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
        trust_radius=jnp.asarray(jnp.nan, dtype=scalar_dtype),
        reduction_ratio=jnp.asarray(jnp.nan, dtype=scalar_dtype),
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
    value_function: _ValueFunction,
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
        jnp.asarray(usable, dtype=jnp.bool_)
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

    def invalid_direction(_: None) -> _BoundState:
        return prepared._replace(
            status=jnp.asarray(
                int(OptimizationStatus.INVALID_DIRECTION),
                dtype=jnp.int32,
            ),
            rejected_steps=prepared.rejected_steps + 1,
        )

    def line_search_step(_: None) -> _BoundState:
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
    initial_method_state: _MethodState,
    iteration_body: Callable[
        [tuple[_BoundState, _MethodState]], tuple[_BoundState, _MethodState]
    ],
    termination: OptimizationTermination,
    /,
) -> _BoundState:
    def condition(carry: tuple[_BoundState, _MethodState]) -> Array:
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
    exhausted_evaluations = (
        jnp.asarray(False)
        if termination.maximum_evaluations is None
        else state.objective_evaluations >= termination.maximum_evaluations
    )
    return state._replace(
        status=jnp.where(
            state.status == int(OptimizationStatus.ITERATING),
            jnp.where(
                exhausted_evaluations,
                int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
            ),
            state.status,
        ).astype(jnp.int32)
    )


def _run_projected_gradient(
    method: ProjectedGradient,
    value_function: _ValueFunction,
    bounds: Bounds,
    initial_state: _BoundState,
    termination: OptimizationTermination,
    /,
) -> _BoundState:
    value_and_gradient = jax.value_and_grad(value_function)

    def iteration_body(carry: _EmptyCarry) -> _EmptyCarry:
        state, method_state = carry
        value, gradient = value_and_gradient(state.parameters)
        evaluated, projected_gradient, _ = _evaluate_bound_state(
            state,
            value,
            gradient,
            bounds,
            termination,
        )

        def take_step(_: None) -> _BoundState:
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
    value_function: _ValueFunction,
    bounds: Bounds,
    initial_state: _BoundState,
    termination: OptimizationTermination,
    /,
) -> _BoundState:
    value_and_gradient = jax.value_and_grad(value_function)

    def iteration_body(carry: _EmptyCarry) -> _EmptyCarry:
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

        def take_step(_: None) -> _BoundState:
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


def _run_bounded_newton_trust_region(
    method: BoundedNewtonTrustRegion,
    value_function: _ValueFunction,
    bounds: Bounds,
    initial_state: _BoundState,
    termination: OptimizationTermination,
    /,
) -> _BoundState:
    value_and_gradient = jax.value_and_grad(value_function)
    seeded = initial_state._replace(
        trust_radius=jnp.asarray(
            method.initial_radius,
            dtype=initial_state.final_step_norm.dtype,
        )
    )

    def iteration_body(carry: _EmptyCarry) -> _EmptyCarry:
        state, method_state = carry
        (value, gradient), linearized = jax.linearize(
            value_and_gradient,
            state.parameters,
        )
        evaluated, _, optimality = _evaluate_bound_state(
            state,
            value,
            gradient,
            bounds,
            termination,
        )

        def take_step(_: None) -> _BoundState:
            active = bounds.active_mask(
                evaluated.parameters,
                gradient,
                tolerance=method.active_tolerance,
            )
            free_gradient = jax.tree.map(
                lambda value, mask: jnp.where(mask, 0.0, value),
                gradient,
                active,
            )
            space = PyTreeSpace(evaluated.parameters)

            def hessian_action(direction: PyTree[Any]) -> PyTree[Array]:
                free_direction = jax.tree.map(
                    lambda value, mask: jnp.where(mask, 0.0, value),
                    direction,
                    active,
                )
                _, action = linearized(free_direction)
                return jax.tree.map(
                    lambda value, mask: jnp.where(mask, 0.0, value),
                    action,
                    active,
                )

            hessian = FunctionLinearOperator(
                hessian_action,
                source=space,
                target=space,
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "construction"},
                ),
                operator_id="bounded-objective-free-hessian",
                closure_convert=False,
            )
            subproblem = solve_trust_region_subproblem(
                TrustRegionQuadraticProblem(
                    hessian,
                    free_gradient,
                    evaluated.trust_radius,
                ),
                method=method.subproblem,
            )
            unprojected = _tree_add_scaled(
                evaluated.parameters,
                subproblem.step,
                1.0,
            )
            candidate = bounds.project(unprojected)
            displacement = jax.tree.map(
                lambda new, old: new - old,
                candidate,
                evaluated.parameters,
            )
            hessian_displacement = hessian.mv(displacement)
            predicted = -(
                _tree_inner(gradient, displacement)
                + 0.5 * _tree_inner(displacement, hessian_displacement)
            )
            candidate_value, candidate_gradient = value_and_gradient(candidate)
            actual = value - candidate_value
            ratio = actual / jnp.maximum(predicted, 1e-30)
            candidate_finite = (
                jnp.isfinite(candidate_value)
                & _tree_allfinite(candidate)
                & _tree_allfinite(candidate_gradient)
            )
            accepted = (
                subproblem.successful
                & candidate_finite
                & jnp.isfinite(predicted)
                & (predicted > 0.0)
                & jnp.isfinite(ratio)
                & (ratio >= method.acceptance_ratio)
            )
            step_norm = _tree_norm(displacement)
            shrink = (
                ~subproblem.successful
                | ~candidate_finite
                | ~jnp.isfinite(ratio)
                | (ratio < method.shrink_ratio)
            )
            expand = (
                accepted
                & (ratio > method.expansion_ratio)
                & subproblem.diagnostics.boundary_hit
            )
            next_radius = jnp.where(
                shrink,
                jnp.maximum(
                    method.minimum_radius,
                    method.shrink_factor * evaluated.trust_radius,
                ),
                jnp.where(
                    expand,
                    jnp.minimum(
                        method.maximum_radius,
                        method.expansion_factor * evaluated.trust_radius,
                    ),
                    evaluated.trust_radius,
                ),
            )
            accepted_parameters = _tree_where(
                accepted,
                candidate,
                evaluated.parameters,
            )
            accepted_gradient = _tree_where(
                accepted,
                candidate_gradient,
                gradient,
            )
            accepted_projected = bounds.projected_gradient(
                accepted_parameters,
                accepted_gradient,
            )
            accepted_optimality = _tree_norm(accepted_projected)
            stagnated = (
                accepted
                & (
                    step_norm
                    <= termination.step_threshold(_tree_norm(accepted_parameters))
                )
                & (
                    accepted_optimality
                    > termination.optimality_threshold(evaluated.initial_optimality)
                )
            )
            failed = (~accepted) & (next_radius <= method.minimum_radius)
            status = jnp.where(
                stagnated,
                int(OptimizationStatus.STAGNATION),
                jnp.where(
                    failed,
                    int(OptimizationStatus.TRUST_REGION_FAILED),
                    int(OptimizationStatus.ITERATING),
                ),
            ).astype(jnp.int32)
            return evaluated._replace(
                parameters=accepted_parameters,
                status=status,
                iterations=evaluated.iterations + 1,
                final_step_norm=step_norm,
                accepted_rate=jnp.where(accepted, 1.0, 0.0),
                accepted_steps=evaluated.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=evaluated.rejected_steps + (~accepted).astype(jnp.int32),
                objective_evaluations=evaluated.objective_evaluations + 1,
                gradient_evaluations=evaluated.gradient_evaluations + 1,
                hvp_evaluations=(
                    evaluated.hvp_evaluations + subproblem.diagnostics.hessian_actions + 1
                ),
                linear_solves=evaluated.linear_solves + 1,
                linear_iterations=(
                    evaluated.linear_iterations + subproblem.diagnostics.iterations
                ),
                setup_refreshes=evaluated.setup_refreshes + 1,
                numeric_refreshes=evaluated.numeric_refreshes + 1,
                globalization_evaluations=evaluated.globalization_evaluations + 1,
                trust_radius=next_radius,
                reduction_ratio=ratio,
            )

        next_state = jax.lax.cond(
            evaluated.status == int(OptimizationStatus.ITERATING),
            take_step,
            lambda _: evaluated,
            None,
        )
        return next_state, method_state

    return _run_bound_iterations(
        seeded,
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
        lambda value: jnp.zeros(value.shape, dtype=jnp.bool_),
        parameters,
    )
    return _LBFGSState(
        previous_parameters=parameters,
        previous_gradient=zero_gradient,
        previous_active=inactive,
        has_previous=jnp.asarray(False),
        history=initial_lbfgs_history(
            parameters,
            history_size,
            _SINGLE_GROUP,
            _tree_inner(parameters, parameters).dtype,
        ),
    )


def _update_lbfgs_history(
    method: ProjectedLBFGS,
    parameters: PyTree[Any],
    gradient: PyTree[Any],
    active: PyTree[Any],
    state: _LBFGSState,
    /,
) -> LBFGSHistory:
    same_active = (~state.has_previous) | _tree_mask_equal(
        active,
        state.previous_active,
    )
    retained = reset_lbfgs_history(state.history, jnp.reshape(~same_active, (1,)))
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
    curvature, step_squared, change_squared = lbfgs_pair_curvature(
        step, change, _SINGLE_GROUP
    )
    admitted = (
        state.has_previous
        & same_active
        & admit_lbfgs_pair(
            curvature,
            step_squared,
            change_squared,
            method.curvature_tolerance,
        )
    )
    return append_lbfgs_pair(
        retained,
        step,
        change,
        curvature,
        change_squared,
        admitted,
        _SINGLE_GROUP,
    )


def _run_projected_lbfgs(
    method: ProjectedLBFGS,
    value_function: _ValueFunction,
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

    def iteration_body(carry: _LBFGSCarry) -> _LBFGSCarry:
        state, history = carry
        value, gradient = value_and_gradient(state.parameters)
        evaluated, projected_gradient, _ = _evaluate_bound_state(
            state,
            value,
            gradient,
            bounds,
            termination,
        )

        def take_step(_: None) -> _LBFGSCarry:
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
            raw_direction = lbfgs_direction(gradient, updated_history, _SINGLE_GROUP)
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
            accepted_history = _LBFGSState(
                previous_parameters=evaluated.parameters,
                previous_gradient=gradient,
                previous_active=active,
                has_previous=jnp.asarray(True),
                history=updated_history,
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
    method: (
        ProjectedGradient | ActiveSetNewton | BoundedNewtonTrustRegion | ProjectedLBFGS
    ),
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
            not isinstance(initial_feasible, core.Tracer)
            and not np.asarray(initial_feasible).item()
        ):
            return _infeasible_result(method, problem, parameters)
        initial_status = jnp.where(
            initial_feasible,
            int(OptimizationStatus.ITERATING),
            int(OptimizationStatus.INFEASIBLE),
        ).astype(jnp.int32)

    def value_function(candidate: PyTree[Any]) -> Array:
        return problem.value(candidate, args)[0]

    abstract_value = jax.eval_shape(value_function, parameters)
    scalar_dtype = inexact_result_type(
        abstract_value.dtype, *(leaf.dtype for leaf in jax.tree.leaves(parameters))
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
    elif isinstance(method, BoundedNewtonTrustRegion):
        state = _run_bounded_newton_trust_region(
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
        damping=state.trust_radius,
        reduction_ratio=state.reduction_ratio,
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
        globalization=(
            "projected-trust-region"
            if isinstance(method, BoundedNewtonTrustRegion)
            else "projected-armijo"
        ),
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


class _ForceDrivenState(Protocol):
    parameters: PyTree[Array]
    accepted_parameters: PyTree[Array]
    initial_optimality: Array
    step_norm: Array
    status: Array
    evaluations: Array
    iterations: Array

    @property
    def accepted_step_size(self) -> Array: ...

    @property
    def damping(self) -> Array: ...

    @property
    def direction_fallbacks(self) -> Array: ...

    @property
    def rejected_steps(self) -> Array: ...


_ForceDrivenStateT = TypeVar("_ForceDrivenStateT", bound=_ForceDrivenState)


class _ForceDrivenMethod(Protocol[_ForceDrivenStateT]):
    active_tolerance: float

    @property
    def method_id(self) -> str: ...

    @property
    def capabilities(self) -> OptimizationCapabilities: ...

    def init(
        self,
        parameters: PyTree[ArrayLike],
        /,
        *,
        layout: OptimizerGroupLayout | None = None,
        value_dtype: DTypeLike | None = None,
    ) -> _ForceDrivenStateT: ...

    def step(
        self,
        state: _ForceDrivenStateT,
        value: ArrayLike,
        gradient: PyTree[ArrayLike],
        /,
        *,
        bounds: Bounds | None = None,
        termination: OptimizationTermination | None = None,
        converged: ArrayLike | None = None,
        active: ArrayLike | None = None,
    ) -> _ForceDrivenStateT: ...


class _ForceDrivenEvaluation(NamedTuple):
    free_gradient: PyTree[Array]
    activity: PyTree[Array] | None
    optimality: Array
    maximum_site_gradient: Array
    finite: Array


def _force_driven_admission(
    parameters: PyTree[ArrayLike],
    layout: OptimizerGroupLayout | None,
    value_dtype: DTypeLike | None,
    /,
) -> tuple[PyTree[Array], OptimizerGroupLayout, np.dtype, np.dtype]:
    parameters_ = _validate_real_inexact_tree(parameters, name="parameters")
    layout_ = _SINGLE_GROUP if layout is None else layout
    if not isinstance(layout_, OptimizerGroupLayout):
        raise TypeError("layout must be an OptimizerGroupLayout or None.")
    layout_.validate(parameters_)
    dtype = inexact_result_type(*(leaf.dtype for leaf in jax.tree.leaves(parameters_)))
    value_dtype_ = dtype if value_dtype is None else jnp.dtype(value_dtype)
    if not jnp.issubdtype(value_dtype_, jnp.floating):
        raise TypeError("value_dtype must be a real floating dtype.")
    return parameters_, layout_, dtype, value_dtype_


def _group_mask(
    mask: ArrayLike | None, groups: int, /, *, default: bool, name: str
) -> Array:
    if mask is None:
        return jnp.full((groups,), default, dtype=jnp.bool_)
    mask_ = jnp.asarray(mask)
    if mask_.dtype != jnp.bool_ or mask_.shape != (groups,):
        raise ValueError(f"{name} must be a boolean array with one entry per group.")
    return mask_


def _force_driven_inputs(
    layout: OptimizerGroupLayout,
    parameters: PyTree[Array],
    value_dtype: DTypeLike,
    value: ArrayLike,
    gradient: PyTree[ArrayLike],
    active: ArrayLike | None,
    converged: ArrayLike | None,
    /,
) -> tuple[Array, PyTree[Array], Array, Array]:
    groups = layout.group_count
    value_ = jnp.asarray(value)
    if value_.shape != (groups,) or not jnp.issubdtype(value_.dtype, jnp.floating):
        raise ValueError("value must hold one real objective value per optimizer group.")
    gradient_ = _validate_real_inexact_tree(gradient, name="gradient")
    if jax.tree.structure(gradient_) != jax.tree.structure(parameters) or any(
        leaf.shape != parameter.shape
        for leaf, parameter in zip(
            jax.tree.leaves(gradient_), jax.tree.leaves(parameters), strict=True
        )
    ):
        raise ValueError("gradient must match the parameter PyTree and leaf shapes.")
    # Stored gradients keep the parameter leaf dtypes so staged state is stable.
    gradient_ = jax.tree.map(
        lambda leaf, parameter: leaf.astype(parameter.dtype), gradient_, parameters
    )
    return (
        value_.astype(value_dtype),
        gradient_,
        _group_mask(active, groups, default=True, name="active"),
        _group_mask(converged, groups, default=False, name="converged"),
    )


def _evaluate_force_driven(
    layout: OptimizerGroupLayout,
    bounds: Bounds | None,
    parameters: PyTree[Array],
    value: Array,
    gradient: PyTree[Array],
    active_tolerance: float,
    /,
) -> _ForceDrivenEvaluation:
    dtype = inexact_result_type(*(leaf.dtype for leaf in jax.tree.leaves(parameters)))
    finite = (
        jnp.isfinite(value)
        & layout.group_all_finite(gradient)
        & layout.group_all_finite(parameters)
    )
    if bounds is None:
        activity = None
        free_gradient = gradient
        projected_gradient = gradient
    else:
        if not isinstance(bounds, Bounds):
            raise TypeError("bounds must be Bounds or None.")
        activity = bounds.active_mask(parameters, gradient, tolerance=active_tolerance)
        free_gradient = _tree_zeros_on_mask(gradient, activity)
        projected_gradient = bounds.projected_gradient(parameters, gradient)
    return _ForceDrivenEvaluation(
        free_gradient=free_gradient,
        activity=activity,
        optimality=layout.norm(projected_gradient).astype(dtype),
        maximum_site_gradient=layout.site_norm_max(projected_gradient).astype(dtype),
        finite=finite,
    )


def _group_converged(
    termination: OptimizationTermination | None,
    optimality: Array,
    initial_optimality: Array,
    converged: Array,
    /,
) -> Array:
    if termination is None:
        return converged
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination or None.")
    return converged | (
        optimality <= termination.optimality_threshold(initial_optimality)
    )


def _group_status(
    previous: Array,
    active: Array,
    finite: Array,
    converged: Array,
    /,
) -> Array:
    return jnp.where(
        ~active,
        previous,
        jnp.where(
            ~finite,
            _NONFINITE_EVALUATION,
            jnp.where(converged, _SUCCESS, _ITERATING),
        ),
    ).astype(jnp.int32)


def _admit_trial(
    layout: OptimizerGroupLayout,
    bounds: Bounds | None,
    trial: PyTree[Array],
    /,
) -> tuple[PyTree[Array], Array]:
    """Project a trial point onto the bounds; report groups the bounds shortened."""

    if bounds is None:
        return trial, jnp.zeros((layout.group_count,), dtype=jnp.bool_)
    projected = bounds.project(trial)
    changed = layout.count_true(jax.tree.map(jnp.not_equal, projected, trial)) > 0
    return projected, changed


def _curvature_floor(tolerance: float | None, dtype: DTypeLike, /) -> float:
    if tolerance is not None:
        return tolerance
    return 1e-6 if jnp.finfo(dtype).bits <= 32 else 1e-10


def _solve_force_driven(
    method: _ForceDrivenMethod[_ForceDrivenStateT],
    problem: MinimizationProblem,
    initial_parameters: PyTree[ArrayLike],
    /,
    *,
    termination: OptimizationTermination,
    args: Any,
    globalization: str,
    notes: str,
) -> MinimizationResult:
    """Run one single-group force-driven method to a terminal status.

    Each loop iteration evaluates the pending point once and consumes it with
    ``method.step``. The result reports the last accepted evaluation, never an
    unevaluated proposal; its value, auxiliary output and gradient are taken from
    one final evaluation of that point.
    """

    if not isinstance(problem, MinimizationProblem):
        raise TypeError("problem must be a MinimizationProblem.")
    if problem.constraints:
        raise ValueError(
            f"{method.method_id} does not accept general nonlinear constraints."
        )
    if problem.derivative_execution != "automatic-jax":
        raise ValueError(f"{method.method_id} requires automatic JAX derivatives.")
    if not isinstance(termination, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination.")
    bounds = problem.bounds
    parameters = _validate_real_inexact_tree(initial_parameters, name="parameters")
    PyTreeSpace(parameters)
    if bounds is not None:
        parameters = bounds.project(parameters)

    def value_function(candidate: PyTree[Array]) -> Array:
        return problem.value(candidate, args)[0]

    value_and_gradient = jax.value_and_grad(value_function)
    abstract_value = jax.eval_shape(value_function, parameters)
    state = method.init(parameters, value_dtype=abstract_value.dtype)
    maximum_evaluations = termination.maximum_evaluations
    evaluation_limited = (
        maximum_evaluations is not None
        and maximum_evaluations <= termination.maximum_steps
    )
    budget = maximum_evaluations if evaluation_limited else termination.maximum_steps

    def condition(current: _ForceDrivenStateT) -> Array:
        return (current.status[0] == _ITERATING) & (current.evaluations[0] < budget)

    def body(current: _ForceDrivenStateT) -> _ForceDrivenStateT:
        value, gradient = value_and_gradient(current.parameters)
        return method.step(
            current,
            jnp.reshape(value, (1,)),
            gradient,
            bounds=bounds,
            termination=termination,
        )

    state = jax.lax.while_loop(condition, body, state)
    exhausted = int(
        OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
        if evaluation_limited
        else OptimizationStatus.MAXIMUM_STEPS_REACHED
    )
    status = jnp.where(state.status[0] == _ITERATING, exhausted, state.status[0]).astype(
        jnp.int32
    )

    accepted = state.accepted_parameters
    (final_value, auxiliary), final_gradient = problem.value_and_gradient(accepted, args)
    if bounds is None:
        final_optimality = _tree_norm(final_gradient)
        primal_feasibility = jnp.zeros_like(final_optimality)
        complementarity = jnp.zeros_like(final_optimality)
        active_constraints = jnp.asarray(0, dtype=jnp.int32)
        certificate = None
    else:
        final_optimality = _tree_norm(bounds.projected_gradient(accepted, final_gradient))
        primal_feasibility = bounds.violation(accepted)
        complementarity = _complementarity_residual(
            bounds,
            accepted,
            final_gradient,
            tolerance=method.active_tolerance,
        )
        active_constraints = _active_count(
            bounds.active_mask(
                accepted, final_gradient, tolerance=method.active_tolerance
            )
        )
        certificate = (
            _bound_certificate(
                bounds,
                accepted,
                final_gradient,
                active_tolerance=method.active_tolerance,
            )
            if _bound_layout_is_static(bounds)
            else None
        )
    evaluations = state.evaluations[0] + 1
    diagnostics = OptimizationDiagnostics(
        iterations=state.iterations[0],
        accepted_steps=state.iterations[0],
        rejected_steps=state.rejected_steps[0],
        objective_evaluations=evaluations,
        gradient_evaluations=evaluations,
        initial_optimality_norm=state.initial_optimality[0],
        final_optimality_norm=final_optimality,
        final_step_norm=state.step_norm[0],
        accepted_step_size=state.accepted_step_size[0],
        damping=state.damping[0],
        direction_fallbacks=state.direction_fallbacks[0],
        primal_feasibility=primal_feasibility,
        dual_feasibility=final_optimality,
        complementarity=complementarity,
        active_constraints=active_constraints,
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax",
        globalization=globalization,
        matrix_free=True,
        implicit_differentiation=method.capabilities.implicit_differentiation,
        notes=notes,
    )
    return MinimizationResult(
        accepted,
        final_value,
        auxiliary,
        status,
        diagnostics,
        provenance,
        certificate=certificate,
        method_evidence=state,
    )


def _infeasible_result(
    method: (
        ProjectedGradient | ActiveSetNewton | BoundedNewtonTrustRegion | ProjectedLBFGS
    ),
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


__all__ = [
    "ActiveSetNewton",
    "BoundedNewtonTrustRegion",
    "ProjectedGradient",
    "ProjectedLBFGS",
    "TrustScaledLBFGS",
    "TrustScaledLBFGSState",
]
