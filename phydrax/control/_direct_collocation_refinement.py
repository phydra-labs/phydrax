#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import TemporalMesh
from ..dynamics import StateLayout
from ..optim import AbstractMinimizationMethod, Bounds, OptimizationTermination
from ._direct_collocation import (
    compile_direct_collocation,
    DirectCollocationBounds,
    DirectCollocationDecision,
    DirectCollocationPlan,
    DirectCollocationResult,
    prepare_direct_collocation,
    solve_prepared_direct_collocation,
)
from ._trajectory_optimization import TrajectoryOptimizationView


DIRECT_REFINEMENT_CONVERGED = 0
DIRECT_REFINEMENT_NO_REFINEMENT = 1
DIRECT_REFINEMENT_MAXIMUM_LEVELS = 2
DIRECT_REFINEMENT_CAPACITY_EXCEEDED = 3
DIRECT_REFINEMENT_SOURCE_FAILED = 4
DIRECT_REFINEMENT_TARGET_FAILED = 5
DIRECT_REFINEMENT_INSUFFICIENT_REDUCTION = 6
DIRECT_REFINEMENT_NONFINITE = 7

DirectCollocationRefinementMode: TypeAlias = Literal["uniform", "bulk-defect"]
DirectCollocationRefinementFailureMode: TypeAlias = Literal["status", "error"]
DirectCollocationBoundProvider: TypeAlias = Callable[
    [DirectCollocationPlan, DirectCollocationDecision], DirectCollocationBounds
]


def _nonnegative(value: float, owner: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{owner} must be finite and non-negative.")
    return resolved


def _maximum_absolute(value: Array, /) -> Array:
    return jnp.max(jnp.abs(value), initial=jnp.asarray(0.0, dtype=value.dtype))


def _maximum_state_error(
    state_layout: StateLayout,
    references: Array,
    points: Array,
    /,
) -> Array:
    if references.shape != points.shape:
        raise ValueError("State error operands must have identical shapes.")
    flat_references = references.reshape((-1,) + state_layout.shape)
    flat_points = points.reshape((-1,) + state_layout.shape)
    local_errors = jax.vmap(
        lambda reference, point: jnp.asarray(
            state_layout.geometry.inverse_retract(reference, point)
        ).reshape((state_layout.local_size,))
    )(flat_references, flat_points)
    return _maximum_absolute(local_errors)


def _tree_error(left: Any, right: Any, /) -> Array:
    if left is None and right is None:
        return jnp.asarray(0.0)
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if left_structure != right_structure:
        raise ValueError("Transferred parameter structures do not match.")
    errors = [
        _maximum_absolute(jnp.asarray(a) - jnp.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    ]
    if not errors:
        return jnp.asarray(0.0)
    return jnp.max(jnp.stack(errors))


class DirectCollocationRefinementPolicy(StrictModule):
    """Explicit h-refinement selection, convergence, and capacity contract."""

    mode: DirectCollocationRefinementMode = eqx.field(static=True)
    maximum_levels: int = eqx.field(static=True)
    maximum_intervals: int = eqx.field(static=True)
    bulk_fraction: float = eqx.field(static=True)
    off_grid_defect_tolerance: float = eqx.field(static=True)
    relative_objective_tolerance: float = eqx.field(static=True)
    state_tolerance: float = eqx.field(static=True)
    control_tolerance: float = eqx.field(static=True)
    minimum_defect_reduction: float = eqx.field(static=True)
    failure_mode: DirectCollocationRefinementFailureMode = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        mode: DirectCollocationRefinementMode = "bulk-defect",
        maximum_levels: int = 4,
        maximum_intervals: int = 4096,
        bulk_fraction: float = 0.5,
        off_grid_defect_tolerance: float = 1.0e-5,
        relative_objective_tolerance: float = 1.0e-6,
        state_tolerance: float = 1.0e-5,
        control_tolerance: float = 1.0e-5,
        minimum_defect_reduction: float = 0.05,
        failure_mode: DirectCollocationRefinementFailureMode = "status",
        policy_id: str = "control:direct-collocation:refinement",
    ):
        if mode not in ("uniform", "bulk-defect"):
            raise ValueError("mode must be 'uniform' or 'bulk-defect'.")
        levels = int(maximum_levels)
        capacity = int(maximum_intervals)
        fraction = float(bulk_fraction)
        reduction = float(minimum_defect_reduction)
        if levels < 1 or capacity < 1:
            raise ValueError("Refinement level and interval capacities must be positive.")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("bulk_fraction must lie in (0, 1].")
        if not 0.0 <= reduction <= 1.0:
            raise ValueError("minimum_defect_reduction must lie in [0, 1].")
        if failure_mode not in ("status", "error"):
            raise ValueError("failure_mode must be 'status' or 'error'.")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be a non-empty string.")
        self.mode = mode
        self.maximum_levels = levels
        self.maximum_intervals = capacity
        self.bulk_fraction = fraction
        self.off_grid_defect_tolerance = _nonnegative(
            off_grid_defect_tolerance, "off_grid_defect_tolerance"
        )
        self.relative_objective_tolerance = _nonnegative(
            relative_objective_tolerance, "relative_objective_tolerance"
        )
        self.state_tolerance = _nonnegative(state_tolerance, "state_tolerance")
        self.control_tolerance = _nonnegative(control_tolerance, "control_tolerance")
        self.minimum_defect_reduction = reduction
        self.failure_mode = failure_mode
        self.policy_id = policy_id


class DirectCollocationRefinementSelection(StrictModule):
    """Per-interval defect metric and one explicit bisection mask."""

    interval_defects: Array
    selected_mask: Array
    selected_indices: Array
    threshold: Array
    target_intervals: int = eqx.field(static=True)
    capacity_exceeded: bool = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)


class DirectCollocationPrimalTransfer(StrictModule):
    """Nested-mesh primal transfer with explicit absence of dual transfer."""

    source_result: DirectCollocationResult
    selection: DirectCollocationRefinementSelection
    target_plan: DirectCollocationPlan
    decision: DirectCollocationDecision
    bounds: DirectCollocationBounds
    old_node_state_error: Array
    control_representation_error: Array
    parameter_error: Array
    duration_error: Array
    dual_transferred: bool = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)


class DirectCollocationRefinementLevel(StrictModule):
    """One source-selection-transfer-target refinement transition."""

    index: int = eqx.field(static=True)
    source_result: DirectCollocationResult
    selection: DirectCollocationRefinementSelection
    transfer: DirectCollocationPrimalTransfer
    target_result: DirectCollocationResult
    relative_objective_change: Array
    state_change: Array
    control_change: Array
    defect_reduction: Array
    status: Array
    level_id: str = eqx.field(static=True)


class DirectCollocationRefinementStudy(StrictModule):
    """Ordered refinement evidence and one explicit terminal reason."""

    initial_result: DirectCollocationResult
    levels: tuple[DirectCollocationRefinementLevel, ...]
    final_result: DirectCollocationResult
    policy: DirectCollocationRefinementPolicy
    status: Array
    converged: Array
    capacity_exhausted: Array
    study_id: str = eqx.field(static=True)


def select_direct_collocation_intervals(
    result: DirectCollocationResult,
    policy: DirectCollocationRefinementPolicy,
    /,
) -> DirectCollocationRefinementSelection:
    if not isinstance(result, DirectCollocationResult):
        raise TypeError("result must be a DirectCollocationResult.")
    if not isinstance(policy, DirectCollocationRefinementPolicy):
        raise TypeError("policy must be DirectCollocationRefinementPolicy.")
    if not bool(np.asarray(result.successful)):
        raise ValueError("Interval selection requires a successful source result.")
    defects = np.asarray(result.diagnostics.off_grid.interval_defects, dtype=float)
    if defects.ndim == 0 or defects.shape[-1] != result.compilation.plan.mesh.num_steps:
        raise ValueError("Source off-grid defects do not match the source mesh.")
    axes = tuple(range(defects.ndim - 1))
    metric = np.sqrt(np.sum(defects**2, axis=axes)) if axes else np.abs(defects)
    selected = np.zeros(metric.shape, dtype=bool)
    if policy.mode == "uniform":
        selected[:] = True
    else:
        squared = metric**2
        total = float(np.sum(squared))
        if total > 0.0:
            order = np.argsort(-metric, kind="stable")
            target = policy.bulk_fraction * total
            accumulated = 0.0
            for index in order:
                selected[index] = True
                accumulated += float(squared[index])
                if accumulated >= target:
                    break
    indices = np.flatnonzero(selected)
    target_intervals = int(metric.size + indices.size)
    threshold = 0.0 if indices.size == 0 else float(np.min(metric[indices]))
    return DirectCollocationRefinementSelection(
        jnp.asarray(metric),
        jnp.asarray(selected),
        jnp.asarray(indices, dtype=jnp.int32),
        jnp.asarray(threshold),
        target_intervals=target_intervals,
        capacity_exceeded=target_intervals > policy.maximum_intervals,
        selection_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-refinement-selection",
                "result": result.result_id,
                "policy": policy.policy_id,
                "selected": indices.tolist(),
                "target_intervals": target_intervals,
            }
        ),
    )


def _bound_is_reusable(bound: Bounds | None, event_shape: tuple[int, ...], /) -> bool:
    if bound is None:
        return True
    for value in (bound.lower, bound.upper):
        shape = np.asarray(value).shape
        if shape not in ((), event_shape):
            return False
    return True


def _refined_bounds(
    result: DirectCollocationResult,
    target_plan: DirectCollocationPlan,
    decision: DirectCollocationDecision,
    provider: DirectCollocationBoundProvider | None,
    /,
) -> DirectCollocationBounds:
    if provider is not None:
        bounds = provider(target_plan, decision)
        if not isinstance(bounds, DirectCollocationBounds):
            raise TypeError("bounds provider must return DirectCollocationBounds.")
        return bounds
    source = result.compilation.bounds
    problem = result.compilation.problem
    if not _bound_is_reusable(source.states, problem.state_shape):
        raise ValueError("Mesh-shaped state bounds require a refinement bound provider.")
    if not _bound_is_reusable(source.controls, problem.control_shape):
        raise ValueError(
            "Mesh-shaped control bounds require a refinement bound provider."
        )
    return source


def _view(result: DirectCollocationResult, /) -> TrajectoryOptimizationView:
    problem = result.compilation.problem
    return TrajectoryOptimizationView(
        result.trajectory.time_grid.times,
        result.decision.states,
        result.decision.controls,
        case_shape=problem.case_shape,
        state_shape=problem.state_shape,
        control_shape=problem.control_shape,
        state_geometry=problem.state_layout.geometry,
    )


def refine_direct_collocation(
    result: DirectCollocationResult,
    policy: DirectCollocationRefinementPolicy,
    /,
    *,
    selection: DirectCollocationRefinementSelection | None = None,
    bounds_provider: DirectCollocationBoundProvider | None = None,
) -> DirectCollocationPrimalTransfer:
    """Bisect selected intervals and transfer only physical primal decisions."""
    if not bool(np.asarray(result.successful)):
        raise ValueError("Refinement requires a successful source result.")
    selection_ = (
        select_direct_collocation_intervals(result, policy)
        if selection is None
        else selection
    )
    if not isinstance(selection_, DirectCollocationRefinementSelection):
        raise TypeError("selection must be DirectCollocationRefinementSelection or None.")
    if selection_.capacity_exceeded:
        raise ValueError("Refinement selection exceeds the interval capacity.")
    selected = np.asarray(selection_.selected_indices, dtype=int)
    if selected.size == 0:
        raise ValueError("Refinement selection contains no intervals.")
    source_plan = result.compilation.plan
    source_nodes = np.asarray(source_plan.mesh.nodes)
    refined_nodes = list(source_nodes)
    for index in selected:
        refined_nodes.append(0.5 * (source_nodes[index] + source_nodes[index + 1]))
    target_nodes = np.asarray(sorted(refined_nodes), dtype=source_nodes.dtype)
    target_mesh = TemporalMesh(
        target_nodes,
        role="collocation",
        mesh_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-refined-mesh",
                "source": source_plan.mesh.mesh_id,
                "selection": selection_.selection_id,
            }
        ),
    )
    target_plan = DirectCollocationPlan(
        target_mesh,
        method=source_plan.method,
        variable_duration=source_plan.variable_duration,
        scaling=source_plan.scaling,
        derivatives=source_plan.derivatives,
        audit=source_plan.audit,
        plan_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-refined-plan",
                "source": source_plan.plan_id,
                "mesh": target_mesh.mesh_id,
            }
        ),
    )
    source_view = _view(result)
    if source_plan.variable_duration:
        duration = result.duration
        target_times = (
            target_mesh.t0
            + (target_mesh.nodes - target_mesh.t0) * duration / target_mesh.duration
        )
    else:
        duration = None
        target_times = target_mesh.nodes
    states = source_view.evaluate_state(target_times)
    theta = target_plan.method.theta
    stage_times = (1.0 - theta) * target_times[:-1] + theta * target_times[1:]
    controls = source_view.evaluate_control(stage_times)
    decision = DirectCollocationDecision(
        states,
        controls,
        result.decision.parameters,
        duration,
    )
    target_positions = np.searchsorted(target_nodes, source_nodes)
    old_node_states = jnp.take(
        states,
        jnp.asarray(target_positions),
        axis=len(result.compilation.problem.case_shape),
    )
    old_node_error = _maximum_state_error(
        result.compilation.problem.state_layout,
        result.decision.states,
        old_node_states,
    )
    transferred_control = source_view.evaluate_control(stage_times)
    control_error = _maximum_absolute(controls - transferred_control)
    parameter_error = _tree_error(decision.parameters, result.decision.parameters)
    duration_error = (
        jnp.asarray(0.0, dtype=states.dtype)
        if duration is None
        else jnp.abs(duration - result.duration)
    )
    bounds = _refined_bounds(result, target_plan, decision, bounds_provider)
    return DirectCollocationPrimalTransfer(
        result,
        selection_,
        target_plan,
        decision,
        bounds,
        old_node_error,
        control_error,
        parameter_error,
        duration_error,
        dual_transferred=False,
        transfer_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-primal-transfer",
                "source": result.result_id,
                "selection": selection_.selection_id,
                "target": target_plan.plan_id,
            }
        ),
    )


def _study(
    initial: DirectCollocationResult,
    levels: list[DirectCollocationRefinementLevel],
    final: DirectCollocationResult,
    policy: DirectCollocationRefinementPolicy,
    status: int,
    /,
) -> DirectCollocationRefinementStudy:
    return DirectCollocationRefinementStudy(
        initial,
        tuple(levels),
        final,
        policy,
        jnp.asarray(status, dtype=jnp.int32),
        jnp.asarray(status == DIRECT_REFINEMENT_CONVERGED),
        jnp.asarray(status == DIRECT_REFINEMENT_CAPACITY_EXCEEDED),
        study_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-refinement-study",
                "initial": initial.result_id,
                "levels": [level.level_id for level in levels],
                "status": status,
                "policy": policy.policy_id,
            }
        ),
    )


def solve_refined_direct_collocation(
    initial_result: DirectCollocationResult,
    policy: DirectCollocationRefinementPolicy,
    /,
    *,
    method: AbstractMinimizationMethod,
    termination: OptimizationTermination,
    bounds_provider: DirectCollocationBoundProvider | None = None,
    args: Any = None,
) -> DirectCollocationRefinementStudy:
    """Solve an explicit sequence of nested h-refined direct transcriptions."""
    if not isinstance(initial_result, DirectCollocationResult):
        raise TypeError("initial_result must be a DirectCollocationResult.")
    if not isinstance(policy, DirectCollocationRefinementPolicy):
        raise TypeError("policy must be DirectCollocationRefinementPolicy.")
    if not bool(np.asarray(initial_result.successful)):
        if policy.failure_mode == "error":
            raise ValueError("Initial direct-collocation result is not successful.")
        return _study(
            initial_result,
            [],
            initial_result,
            policy,
            DIRECT_REFINEMENT_SOURCE_FAILED,
        )
    current = initial_result
    levels: list[DirectCollocationRefinementLevel] = []
    if (
        float(current.diagnostics.maximum_off_grid_defect)
        <= policy.off_grid_defect_tolerance
    ):
        return _study(
            initial_result,
            levels,
            current,
            policy,
            DIRECT_REFINEMENT_NO_REFINEMENT,
        )
    runtime_args = current.compilation.problem.args if args is None else args
    for level_index in range(policy.maximum_levels):
        selection = select_direct_collocation_intervals(current, policy)
        if selection.capacity_exceeded:
            if policy.failure_mode == "error":
                raise ValueError("Refinement interval capacity was exceeded.")
            return _study(
                initial_result,
                levels,
                current,
                policy,
                DIRECT_REFINEMENT_CAPACITY_EXCEEDED,
            )
        if int(selection.selected_indices.size) == 0:
            return _study(
                initial_result,
                levels,
                current,
                policy,
                DIRECT_REFINEMENT_NO_REFINEMENT,
            )
        transfer = refine_direct_collocation(
            current,
            policy,
            selection=selection,
            bounds_provider=bounds_provider,
        )
        compilation = compile_direct_collocation(
            current.compilation.problem,
            transfer.target_plan,
            transfer.decision.states,
            transfer.decision.controls,
            parameter_guess=transfer.decision.parameters,
            duration_guess=transfer.decision.duration,
            bounds=transfer.bounds,
        )
        prepared = prepare_direct_collocation(
            compilation,
            method=method,
            termination=termination,
        )
        target = solve_prepared_direct_collocation(prepared, args=runtime_args)
        source_view = _view(current)
        target_view = _view(target)
        common_times = current.trajectory.time_grid.times
        state_change = _maximum_state_error(
            current.compilation.problem.state_layout,
            source_view.states,
            target_view.evaluate_state(common_times),
        )
        theta = current.compilation.plan.method.theta
        source_stage_times = (1.0 - theta) * common_times[:-1] + theta * common_times[1:]
        control_change = _maximum_absolute(
            target_view.evaluate_control(source_stage_times)
            - source_view.evaluate_control(source_stage_times)
        )
        objective_scale = jnp.maximum(jnp.abs(current.objective), 1.0)
        objective_change = jnp.abs(target.objective - current.objective) / objective_scale
        source_defect = current.diagnostics.maximum_off_grid_defect
        target_defect = target.diagnostics.maximum_off_grid_defect
        defect_reduction = (source_defect - target_defect) / jnp.maximum(
            source_defect,
            jnp.finfo(source_defect.dtype).eps,
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        objective_change,
                        state_change,
                        control_change,
                        defect_reduction,
                    )
                )
            )
        )
        level_status = jnp.where(
            ~finite,
            DIRECT_REFINEMENT_NONFINITE,
            jnp.where(
                ~target.successful,
                DIRECT_REFINEMENT_TARGET_FAILED,
                DIRECT_REFINEMENT_CONVERGED,
            ),
        ).astype(jnp.int32)
        level = DirectCollocationRefinementLevel(
            level_index,
            current,
            selection,
            transfer,
            target,
            objective_change,
            state_change,
            control_change,
            defect_reduction,
            level_status,
            level_id=canonical_fingerprint(
                {
                    "kind": "direct-collocation-refinement-level",
                    "index": level_index,
                    "source": current.result_id,
                    "target": target.result_id,
                    "selection": selection.selection_id,
                }
            ),
        )
        levels.append(level)
        if not bool(np.asarray(finite)):
            return _study(
                initial_result,
                levels,
                target,
                policy,
                DIRECT_REFINEMENT_NONFINITE,
            )
        if not bool(np.asarray(target.successful)):
            if policy.failure_mode == "error":
                raise ValueError("A refined direct-collocation solve failed.")
            return _study(
                initial_result,
                levels,
                target,
                policy,
                DIRECT_REFINEMENT_TARGET_FAILED,
            )
        converged = (
            float(target_defect) <= policy.off_grid_defect_tolerance
            and float(objective_change) <= policy.relative_objective_tolerance
            and float(state_change) <= policy.state_tolerance
            and float(control_change) <= policy.control_tolerance
        )
        if converged:
            return _study(
                initial_result,
                levels,
                target,
                policy,
                DIRECT_REFINEMENT_CONVERGED,
            )
        if float(defect_reduction) < policy.minimum_defect_reduction:
            if policy.failure_mode == "error":
                raise ValueError("Refinement produced insufficient defect reduction.")
            return _study(
                initial_result,
                levels,
                target,
                policy,
                DIRECT_REFINEMENT_INSUFFICIENT_REDUCTION,
            )
        current = target
    return _study(
        initial_result,
        levels,
        current,
        policy,
        DIRECT_REFINEMENT_MAXIMUM_LEVELS,
    )


__all__ = [
    "DIRECT_REFINEMENT_CAPACITY_EXCEEDED",
    "DIRECT_REFINEMENT_CONVERGED",
    "DIRECT_REFINEMENT_INSUFFICIENT_REDUCTION",
    "DIRECT_REFINEMENT_MAXIMUM_LEVELS",
    "DIRECT_REFINEMENT_NONFINITE",
    "DIRECT_REFINEMENT_NO_REFINEMENT",
    "DIRECT_REFINEMENT_SOURCE_FAILED",
    "DIRECT_REFINEMENT_TARGET_FAILED",
    "DirectCollocationBoundProvider",
    "DirectCollocationPrimalTransfer",
    "DirectCollocationRefinementFailureMode",
    "DirectCollocationRefinementLevel",
    "DirectCollocationRefinementMode",
    "DirectCollocationRefinementPolicy",
    "DirectCollocationRefinementSelection",
    "DirectCollocationRefinementStudy",
    "refine_direct_collocation",
    "select_direct_collocation_intervals",
    "solve_refined_direct_collocation",
]
