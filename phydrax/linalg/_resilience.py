#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._binding import LinearSolveTemplate
from ._materialization import MaterializationPolicy, materialize
from ._policies import FailurePolicy, LinearSolvePolicy, TolerancePolicy
from ._problems import LinearSystem
from ._results import LinearSolveResult, LinearSolveStatus
from ._runtime import _pack_rhs, _unpack_value, bind_numeric, prepare_template, solve
from ._spaces import _coordinate_dtype, RHSLayout
from ._structured_operators import TwoSidedScaledLinearOperator


EquilibrationMode = Literal["none", "ruiz", "symmetric-ruiz", "explicit"]


class ResilientSolveStatus(IntEnum):
    """Terminal state of residual-verified iterative refinement."""

    SUCCESS = 0
    BASE_SOLVE_FAILED = 1
    STAGNATED = 2
    MAX_STEPS_REACHED = 3
    NONFINITE_OUTPUT = 4


class EquilibrationPolicy(StrictModule):
    """Bounded two-sided diagonal equilibration policy."""

    mode: EquilibrationMode = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    minimum_scale: float = eqx.field(static=True)
    maximum_scale: float = eqx.field(static=True)
    diagnose_condition: bool = eqx.field(static=True)
    materialization: MaterializationPolicy
    left_scale: Array | None
    right_scale: Array | None

    def __init__(
        self,
        mode: EquilibrationMode = "none",
        *,
        max_steps: int = 8,
        tolerance: float = 1e-2,
        minimum_scale: float = 1e-12,
        maximum_scale: float = 1e12,
        diagnose_condition: bool = False,
        materialization: MaterializationPolicy | None = None,
        left_scale: ArrayLike | None = None,
        right_scale: ArrayLike | None = None,
    ):
        if mode not in ("none", "ruiz", "symmetric-ruiz", "explicit"):
            raise ValueError("Unknown equilibration mode.")
        steps = int(max_steps)
        tolerance_ = float(tolerance)
        minimum = float(minimum_scale)
        maximum = float(maximum_scale)
        if steps < 1:
            raise ValueError("max_steps must be positive.")
        if not math.isfinite(tolerance_) or tolerance_ <= 0:
            raise ValueError("tolerance must be finite and positive.")
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum <= 0
            or maximum < minimum
        ):
            raise ValueError("Scale bounds must be finite, positive, and ordered.")
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        if mode == "explicit":
            if left_scale is None or right_scale is None:
                raise ValueError("Explicit equilibration requires both scaling arrays.")
            left = jnp.asarray(left_scale)
            right = jnp.asarray(right_scale)
        else:
            if left_scale is not None or right_scale is not None:
                raise ValueError("Scaling arrays are only valid for explicit mode.")
            left = right = None
        self.mode = mode
        self.max_steps = steps
        self.tolerance = tolerance_
        self.minimum_scale = minimum
        self.maximum_scale = maximum
        self.diagnose_condition = bool(diagnose_condition)
        self.materialization = materialization_
        self.left_scale = left
        self.right_scale = right


class RefinementPolicy(StrictModule):
    """Residual tolerance and monotonicity policy for iterative refinement."""

    max_steps: int = eqx.field(static=True)
    tolerance: TolerancePolicy
    minimum_improvement: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_steps: int = 3,
        tolerance: TolerancePolicy | None = None,
        minimum_improvement: float = 0.999,
    ):
        steps = int(max_steps)
        tolerance_ = TolerancePolicy(relative=1e-10) if tolerance is None else tolerance
        improvement = float(minimum_improvement)
        if steps < 0:
            raise ValueError("max_steps must be non-negative.")
        if not isinstance(tolerance_, TolerancePolicy):
            raise TypeError("tolerance must be a TolerancePolicy.")
        if not math.isfinite(improvement) or not 0 < improvement <= 1:
            raise ValueError("minimum_improvement must lie in (0, 1].")
        self.max_steps = steps
        self.tolerance = tolerance_
        self.minimum_improvement = improvement


class ResilienceResourcePolicy(StrictModule):
    """Hard cap for transformation and refinement workspace."""

    max_workspace_bytes: int = eqx.field(static=True)

    def __init__(self, *, max_workspace_bytes: int = 512 * 1024 * 1024):
        limit = int(max_workspace_bytes)
        if limit < 1:
            raise ValueError("max_workspace_bytes must be positive.")
        self.max_workspace_bytes = limit


class ResilientSolvePolicy(StrictModule):
    """Base solve plus equilibration, refinement, failure, and resource controls."""

    base: LinearSolvePolicy
    equilibration: EquilibrationPolicy
    refinement: RefinementPolicy
    failure: FailurePolicy
    resources: ResilienceResourcePolicy

    def __init__(
        self,
        base: LinearSolvePolicy | None = None,
        *,
        equilibration: EquilibrationPolicy | None = None,
        refinement: RefinementPolicy | None = None,
        failure: FailurePolicy | None = None,
        resources: ResilienceResourcePolicy | None = None,
    ):
        base_ = LinearSolvePolicy() if base is None else base
        equilibration_ = EquilibrationPolicy() if equilibration is None else equilibration
        refinement_ = RefinementPolicy() if refinement is None else refinement
        failure_ = FailurePolicy("error") if failure is None else failure
        resources_ = ResilienceResourcePolicy() if resources is None else resources
        if not isinstance(base_, LinearSolvePolicy):
            raise TypeError("base must be a LinearSolvePolicy.")
        if not isinstance(equilibration_, EquilibrationPolicy):
            raise TypeError("equilibration must be an EquilibrationPolicy.")
        if not isinstance(refinement_, RefinementPolicy):
            raise TypeError("refinement must be a RefinementPolicy.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy.")
        if not isinstance(resources_, ResilienceResourcePolicy):
            raise TypeError("resources must be a ResilienceResourcePolicy.")
        self.base = base_
        self.equilibration = equilibration_
        self.refinement = refinement_
        self.failure = failure_
        self.resources = resources_


class ResilientCostEstimate(StrictModule):
    """Static storage and workspace accounting for a resilient solve."""

    dimension: int = eqx.field(static=True)
    transformation_storage_bytes: int = eqx.field(static=True)
    materialization_bytes: int = eqx.field(static=True)
    refinement_workspace_bytes_per_rhs: int = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        itemsize: int,
        materializes: bool,
        /,
    ):
        n, size = int(dimension), int(itemsize)
        self.dimension = n
        self.transformation_storage_bytes = 2 * n * size
        self.materialization_bytes = n * n * size if materializes else 0
        self.refinement_workspace_bytes_per_rhs = 6 * n * size


class DiagonalSystemTransform(StrictModule):
    """Auditable map between original and diagonally transformed coordinates."""

    left_scale: Array
    right_scale: Array
    converged: Array
    steps: Array
    row_spread: Array
    column_spread: Array
    mode: EquilibrationMode = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        left_scale: Array,
        right_scale: Array,
        converged: Array,
        steps: Array,
        row_spread: Array,
        column_spread: Array,
        mode: EquilibrationMode,
        transform_id: str,
    ):
        self.left_scale = jnp.asarray(left_scale)
        self.right_scale = jnp.asarray(right_scale)
        self.converged = jnp.asarray(converged, dtype=bool)
        self.steps = jnp.asarray(steps, dtype=jnp.int32)
        self.row_spread = jnp.asarray(row_spread)
        self.column_spread = jnp.asarray(column_spread)
        self.mode = mode
        self.transform_id = str(transform_id)


class ResilientSolvePlan(StrictModule):
    """Coefficient-independent plan for transformed solve and refinement."""

    policy: ResilientSolvePolicy
    base_template: LinearSolveTemplate
    cost: ResilientCostEstimate
    problem_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    space_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: LinearSystem,
        policy: ResilientSolvePolicy,
        base_template: LinearSolveTemplate,
        cost: ResilientCostEstimate,
    ):
        self.policy = policy
        self.base_template = base_template
        self.cost = cost
        self.problem_id = problem.problem_id
        self.operator_id = problem.operator.operator_id
        self.space_id = problem.operator.source.space_id
        self.dimension = problem.operator.source.size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "resilient-solve-plan",
                "problem": problem.problem_id,
                "operator": problem.operator.operator_id,
                "space": problem.operator.source.space_id,
                "dimension": self.dimension,
                "base_template": base_template.template_id,
                "equilibration": {
                    "mode": policy.equilibration.mode,
                    "steps": policy.equilibration.max_steps,
                    "tolerance": policy.equilibration.tolerance,
                    "minimum_scale": policy.equilibration.minimum_scale,
                    "maximum_scale": policy.equilibration.maximum_scale,
                    "diagnose_condition": policy.equilibration.diagnose_condition,
                },
                "refinement": {
                    "steps": policy.refinement.max_steps,
                    "relative": policy.refinement.tolerance.relative,
                    "absolute": policy.refinement.tolerance.absolute,
                    "minimum_improvement": policy.refinement.minimum_improvement,
                },
                "failure": policy.failure.mode,
                "workspace_limit": policy.resources.max_workspace_bytes,
            }
        )


class PreparedResilientSolve(StrictModule):
    """Numerical transformation and reusable prepared solve state."""

    problem: LinearSystem
    transformed_operator: TwoSidedScaledLinearOperator
    plan: ResilientSolvePlan
    base_prepared: Any
    transform: DiagonalSystemTransform
    condition_before: Array
    condition_after: Array
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: LinearSystem,
        transformed_operator: TwoSidedScaledLinearOperator,
        plan: ResilientSolvePlan,
        base_prepared: Any,
        transform: DiagonalSystemTransform,
        condition_before: Array,
        condition_after: Array,
        numeric_version: Any,
    ):
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        self.problem = problem
        self.transformed_operator = transformed_operator
        self.plan = plan
        self.base_prepared = base_prepared
        self.transform = transform
        self.condition_before = jnp.asarray(condition_before)
        self.condition_after = jnp.asarray(condition_after)
        self.numeric_version = version
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-resilient-solve",
                "plan": plan.plan_id,
                "state": "numeric",
            }
        )


class ResilientSolveDiagnostics(StrictModule):
    """Per-RHS forward evidence and shared equilibration diagnostics."""

    initial_residual_norm: Array
    residual_norm: Array
    relative_residual: Array
    backward_error: Array
    refinement_steps: Array
    base_status: Array
    base_iterations: Array
    base_matvec_count: Array
    condition_before: Array
    condition_after: Array
    row_spread: Array
    column_spread: Array

    def __init__(self, **values: Any):
        self.initial_residual_norm = jnp.asarray(values["initial_residual_norm"])
        self.residual_norm = jnp.asarray(values["residual_norm"])
        self.relative_residual = jnp.asarray(values["relative_residual"])
        self.backward_error = jnp.asarray(values["backward_error"])
        self.refinement_steps = jnp.asarray(values["refinement_steps"], dtype=jnp.int32)
        self.base_status = jnp.asarray(values["base_status"], dtype=jnp.int32)
        self.base_iterations = jnp.asarray(values["base_iterations"], dtype=jnp.int32)
        self.base_matvec_count = jnp.asarray(values["base_matvec_count"], dtype=jnp.int32)
        self.condition_before = jnp.asarray(values["condition_before"])
        self.condition_after = jnp.asarray(values["condition_after"])
        self.row_spread = jnp.asarray(values["row_spread"])
        self.column_spread = jnp.asarray(values["column_spread"])


class ResilientSolveProvenance(StrictModule):
    """Static plan identities and dynamic numerical version."""

    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)
    base_plan_id: str = eqx.field(static=True)
    base_template_id: str = eqx.field(static=True)
    numeric_version: Array
    base_numeric_version: Array

    def __init__(self, prepared: PreparedResilientSolve, /):
        self.plan_id = prepared.plan.plan_id
        self.prepared_id = prepared.prepared_id
        self.operator_id = prepared.problem.operator.operator_id
        self.transform_id = prepared.transform.transform_id
        self.base_plan_id = prepared.base_prepared.plan.plan_id
        self.base_template_id = prepared.base_prepared.template.template_id
        self.numeric_version = prepared.numeric_version
        self.base_numeric_version = prepared.base_prepared.numeric_version


class ResilientSolveResult(StrictModule):
    """Residual-verified value, status, diagnostics, provenance, and initial solve."""

    value: PyTree[Array]
    status: Array
    diagnostics: ResilientSolveDiagnostics
    provenance: ResilientSolveProvenance
    base_result: LinearSolveResult

    def __init__(
        self,
        value: PyTree[Array],
        status: Array,
        diagnostics: ResilientSolveDiagnostics,
        provenance: ResilientSolveProvenance,
        base_result: LinearSolveResult,
        /,
    ):
        self.value = value
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.base_result = base_result

    @property
    def successful(self) -> Array:
        return self.status == int(ResilientSolveStatus.SUCCESS)


def plan_resilient_solve(
    problem: LinearSystem,
    policy: ResilientSolvePolicy | None = None,
    /,
) -> ResilientSolvePlan:
    """Plan transformation and solve structure without inspecting coefficients."""
    _validate_problem(problem)
    policy_ = ResilientSolvePolicy() if policy is None else policy
    if not isinstance(policy_, ResilientSolvePolicy):
        raise TypeError("policy must be a ResilientSolvePolicy or None.")
    _validate_explicit_scales(problem, policy_.equilibration)
    n = problem.operator.source.size
    dtype = _coordinate_dtype(problem.operator.source)
    materializes = _requires_materialization(policy_.equilibration)
    cost = ResilientCostEstimate(n, dtype.itemsize, materializes)
    if materializes:
        entries = n * n
        materialization = policy_.equilibration.materialization
        if entries > materialization.max_entries:
            raise ValueError("Equilibration matrix exceeds max_entries.")
        if cost.materialization_bytes > materialization.max_bytes:
            raise ValueError("Equilibration matrix exceeds max_bytes.")
        if not problem.operator.capabilities.materialize:
            raise ValueError(
                "Automatic equilibration requires materialization capability."
            )
    workspace = cost.materialization_bytes + cost.refinement_workspace_bytes_per_rhs
    if workspace > policy_.resources.max_workspace_bytes:
        raise ValueError("Resilient solve workspace exceeds max_workspace_bytes.")
    placeholder = _placeholder_transform(problem, policy_.equilibration)
    transformed = _transformed_problem(problem, placeholder)
    base_template = prepare_template(transformed, policy_.base)
    return ResilientSolvePlan(
        problem=problem,
        policy=policy_,
        base_template=base_template,
        cost=cost,
    )


def prepare_resilient_solve(
    problem: LinearSystem,
    policy: ResilientSolvePolicy | ResilientSolvePlan | None = None,
    /,
    *,
    numeric_version: Any = 0,
) -> PreparedResilientSolve:
    """Build equilibration and bind reusable numerical solver state."""
    plan = (
        policy
        if isinstance(policy, ResilientSolvePlan)
        else plan_resilient_solve(problem, policy)
    )
    if not isinstance(plan, ResilientSolvePlan):
        raise TypeError(
            "policy must be a ResilientSolvePolicy, ResilientSolvePlan, or None."
        )
    _validate_plan_problem(plan, problem)
    return _prepare_with_plan(problem, plan, numeric_version=numeric_version)


def refresh_resilient_solve(
    prepared: PreparedResilientSolve,
    problem: LinearSystem,
    /,
) -> PreparedResilientSolve:
    """Re-equilibrate changed coefficients while preserving the symbolic plan."""
    if not isinstance(prepared, PreparedResilientSolve):
        raise TypeError("prepared must be a PreparedResilientSolve.")
    _validate_plan_problem(prepared.plan, problem)
    version = prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32)
    return _prepare_with_plan(problem, prepared.plan, numeric_version=version)


def solve_resilient(
    problem_or_prepared: LinearSystem | PreparedResilientSolve,
    rhs: PyTree[Any],
    policy: ResilientSolvePolicy | ResilientSolvePlan | None = None,
    /,
    *,
    rhs_layout: RHSLayout | None = None,
) -> ResilientSolveResult:
    """Solve, verify the original residual, and monotonically refine corrections."""
    if isinstance(problem_or_prepared, PreparedResilientSolve):
        if policy is not None:
            raise ValueError("policy must be omitted when solving prepared state.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, LinearSystem):
        prepared = prepare_resilient_solve(problem_or_prepared, policy)
    else:
        raise TypeError("Expected a LinearSystem or PreparedResilientSolve.")
    if rhs_layout is not None and not isinstance(rhs_layout, RHSLayout):
        raise TypeError("rhs_layout must be an RHSLayout or None.")

    problem = prepared.problem
    operator = problem.operator
    canonical_rhs, layout = _pack_rhs(operator.target, (), rhs, rhs_layout)
    transformed_rhs = prepared.transform.left_scale[:, None] * canonical_rhs
    transformed_rhs_tree = _unpack_value(operator.target, transformed_rhs, layout)
    base_result = solve(
        prepared.base_prepared,
        transformed_rhs_tree,
        rhs_layout=rhs_layout,
    )
    transformed_value, result_layout = _pack_rhs(operator.source, (), base_result.value)
    if result_layout.rhs_shape != layout.rhs_shape:
        raise ValueError("The transformed solve changed the right-hand-side layout.")
    coordinates = prepared.transform.right_scale[:, None] * transformed_value
    residual, residual_norm, applied_norm = _residual(
        operator, coordinates, canonical_rhs
    )
    initial_residual_norm = residual_norm
    rhs_norm = _column_norm(operator.target, canonical_rhs)
    tolerance = prepared.plan.policy.refinement.tolerance
    converged = residual_norm <= tolerance.absolute + tolerance.relative * rhs_norm
    finite = jnp.all(jnp.isfinite(coordinates), axis=0) & jnp.isfinite(residual_norm)
    base_status = jnp.asarray(base_result.status, dtype=jnp.int32).reshape(-1)
    reported_failure = base_status != int(LinearSolveStatus.SUCCESS)
    stagnated = jnp.zeros_like(converged)
    refinement_steps = jnp.zeros_like(base_status)
    base_iterations = jnp.asarray(
        base_result.diagnostics.iterations, dtype=jnp.int32
    ).reshape(-1)
    base_matvec_count = jnp.asarray(
        base_result.diagnostics.matvec_count, dtype=jnp.int32
    ).reshape(-1)

    def refinement_step(_, state):
        (
            current,
            current_residual,
            current_norm,
            current_applied_norm,
            current_converged,
            current_finite,
            current_reported_failure,
            current_stagnated,
            steps,
            iterations,
            matvecs,
        ) = state
        active = ~current_converged & current_finite & ~current_stagnated
        correction_rhs = jnp.where(active[None, :], current_residual, 0)
        transformed_correction_rhs = (
            prepared.transform.left_scale[:, None] * correction_rhs
        )
        correction_rhs_tree = _unpack_value(
            operator.target,
            transformed_correction_rhs,
            layout,
        )
        correction_result = solve(
            prepared.base_prepared,
            correction_rhs_tree,
            rhs_layout=rhs_layout,
        )
        transformed_correction, _ = _pack_rhs(
            operator.source, (), correction_result.value
        )
        correction = prepared.transform.right_scale[:, None] * transformed_correction
        candidate = current + correction
        candidate_residual, candidate_norm, candidate_applied_norm = _residual(
            operator, candidate, canonical_rhs
        )
        candidate_finite = jnp.all(jnp.isfinite(candidate), axis=0) & jnp.isfinite(
            candidate_norm
        )
        candidate_converged = candidate_norm <= (
            tolerance.absolute + tolerance.relative * rhs_norm
        )
        improved = candidate_norm <= (
            prepared.plan.policy.refinement.minimum_improvement * current_norm
        )
        accept = active & candidate_finite & (improved | candidate_converged)
        updated = jnp.where(accept[None, :], candidate, current)
        updated_residual = jnp.where(
            accept[None, :], candidate_residual, current_residual
        )
        updated_norm = jnp.where(accept, candidate_norm, current_norm)
        updated_applied_norm = jnp.where(
            accept, candidate_applied_norm, current_applied_norm
        )
        correction_status = jnp.asarray(
            correction_result.status, dtype=jnp.int32
        ).reshape(-1)
        correction_iterations = jnp.asarray(
            correction_result.diagnostics.iterations, dtype=jnp.int32
        ).reshape(-1)
        correction_matvecs = jnp.asarray(
            correction_result.diagnostics.matvec_count, dtype=jnp.int32
        ).reshape(-1)
        return (
            updated,
            updated_residual,
            updated_norm,
            updated_applied_norm,
            current_converged | (accept & candidate_converged),
            current_finite & (~active | candidate_finite),
            current_reported_failure
            | (active & (correction_status != int(LinearSolveStatus.SUCCESS))),
            current_stagnated
            | (active & candidate_finite & ~improved & ~candidate_converged),
            steps + active.astype(jnp.int32),
            iterations + jnp.where(active, correction_iterations, 0),
            matvecs + jnp.where(active, correction_matvecs, 0),
        )

    state = (
        coordinates,
        residual,
        residual_norm,
        applied_norm,
        converged,
        finite,
        reported_failure,
        stagnated,
        refinement_steps,
        base_iterations,
        base_matvec_count,
    )
    state = jax.lax.fori_loop(
        0,
        prepared.plan.policy.refinement.max_steps,
        refinement_step,
        state,
    )
    (
        coordinates,
        _,
        residual_norm,
        applied_norm,
        converged,
        finite,
        reported_failure,
        stagnated,
        refinement_steps,
        base_iterations,
        base_matvec_count,
    ) = state
    relative_residual = residual_norm / jnp.maximum(
        rhs_norm, jnp.finfo(residual_norm.dtype).tiny
    )
    backward_error = residual_norm / jnp.maximum(
        applied_norm + rhs_norm,
        jnp.finfo(residual_norm.dtype).tiny,
    )
    status = jnp.full(
        residual_norm.shape,
        int(ResilientSolveStatus.MAX_STEPS_REACHED),
        dtype=jnp.int32,
    )
    status = jnp.where(
        reported_failure,
        int(ResilientSolveStatus.BASE_SOLVE_FAILED),
        status,
    )
    status = jnp.where(
        stagnated,
        int(ResilientSolveStatus.STAGNATED),
        status,
    )
    status = jnp.where(
        ~finite,
        int(ResilientSolveStatus.NONFINITE_OUTPUT),
        status,
    )
    status = jnp.where(converged & finite, int(ResilientSolveStatus.SUCCESS), status)
    value = _unpack_value(operator.source, coordinates, layout)
    status_out = _restore_axes(status, layout)
    diagnostics = ResilientSolveDiagnostics(
        initial_residual_norm=_restore_axes(initial_residual_norm, layout),
        residual_norm=_restore_axes(residual_norm, layout),
        relative_residual=_restore_axes(relative_residual, layout),
        backward_error=_restore_axes(backward_error, layout),
        refinement_steps=_restore_axes(refinement_steps, layout),
        base_status=_restore_axes(base_status, layout),
        base_iterations=_restore_axes(base_iterations, layout),
        base_matvec_count=_restore_axes(base_matvec_count, layout),
        condition_before=prepared.condition_before,
        condition_after=prepared.condition_after,
        row_spread=prepared.transform.row_spread,
        column_spread=prepared.transform.column_spread,
    )
    if prepared.plan.policy.failure.mode == "error":
        value = jax.tree.map(
            lambda leaf: eqx.error_if(
                leaf,
                jnp.any(status_out != int(ResilientSolveStatus.SUCCESS)),
                "Resilient solve failed; inspect status-mode diagnostics.",
            ),
            value,
        )
    return ResilientSolveResult(
        value,
        status_out,
        diagnostics,
        ResilientSolveProvenance(prepared),
        base_result,
    )


def _prepare_with_plan(
    problem: LinearSystem,
    plan: ResilientSolvePlan,
    *,
    numeric_version: Any,
) -> PreparedResilientSolve:
    transform, matrix = _build_transform(problem, plan.policy.equilibration)
    transformed_problem = _transformed_problem(problem, transform)
    base_prepared = bind_numeric(
        plan.base_template,
        transformed_problem,
        numeric_version=numeric_version,
    )
    transformed_operator = transformed_problem.operator
    if not isinstance(transformed_operator, TwoSidedScaledLinearOperator):
        raise TypeError("Internal transformed operator construction failed.")
    real_dtype = np.empty((), dtype=_coordinate_dtype(problem.operator.source)).real.dtype
    if matrix is None:
        condition_before = condition_after = jnp.asarray(jnp.nan, dtype=real_dtype)
    else:
        condition_before = jnp.linalg.cond(matrix)
        transformed_matrix = (
            transform.left_scale[:, None] * matrix * transform.right_scale[None, :]
        )
        condition_after = jnp.linalg.cond(transformed_matrix)
    return PreparedResilientSolve(
        problem=problem,
        transformed_operator=transformed_operator,
        plan=plan,
        base_prepared=base_prepared,
        transform=transform,
        condition_before=condition_before,
        condition_after=condition_after,
        numeric_version=numeric_version,
    )


def _build_transform(
    problem: LinearSystem,
    policy: EquilibrationPolicy,
    /,
) -> tuple[DiagonalSystemTransform, Array | None]:
    n = problem.operator.source.size
    dtype = _coordinate_dtype(problem.operator.source)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    one = jnp.ones((n,), dtype=dtype)
    nan = jnp.asarray(jnp.nan, dtype=real_dtype)
    if policy.mode == "none":
        transform = DiagonalSystemTransform(
            left_scale=one,
            right_scale=one,
            converged=jnp.asarray(True),
            steps=jnp.asarray(0, dtype=jnp.int32),
            row_spread=jnp.asarray(1.0, dtype=real_dtype),
            column_spread=jnp.asarray(1.0, dtype=real_dtype),
            mode=policy.mode,
            transform_id=_transform_id(problem, policy),
        )
        matrix = (
            materialize(problem.operator, policy.materialization)
            if policy.diagnose_condition
            else None
        )
        return transform, matrix
    if policy.mode == "explicit":
        left = jnp.asarray(policy.left_scale, dtype=dtype)
        right = jnp.asarray(policy.right_scale, dtype=dtype)
        coefficients = jnp.concatenate((left, right))
        coefficients = eqx.error_if(
            coefficients,
            jnp.any(~jnp.isfinite(coefficients))
            | jnp.any(jnp.real(coefficients) <= 0)
            | jnp.any(jnp.imag(coefficients) != 0),
            "Explicit equilibration scales must be finite and positive real values.",
        )
        matrix = (
            materialize(problem.operator, policy.materialization)
            if policy.diagnose_condition
            else None
        )
        if matrix is None:
            row_spread = column_spread = nan
        else:
            scaled = left[:, None] * matrix * right[None, :]
            row_spread, column_spread = _matrix_spreads(scaled)
        return (
            DiagonalSystemTransform(
                left_scale=coefficients[:n],
                right_scale=coefficients[n:],
                converged=jnp.asarray(True),
                steps=jnp.asarray(0, dtype=jnp.int32),
                row_spread=row_spread,
                column_spread=column_spread,
                mode=policy.mode,
                transform_id=_transform_id(problem, policy),
            ),
            matrix,
        )
    matrix = materialize(problem.operator, policy.materialization)
    left, right, scaled, converged, steps = _ruiz_equilibrate(matrix, policy)
    row_spread, column_spread = _matrix_spreads(scaled)
    return (
        DiagonalSystemTransform(
            left_scale=left,
            right_scale=right,
            converged=converged,
            steps=steps,
            row_spread=row_spread,
            column_spread=column_spread,
            mode=policy.mode,
            transform_id=_transform_id(problem, policy),
        ),
        matrix,
    )


def _ruiz_equilibrate(
    matrix: Array,
    policy: EquilibrationPolicy,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    n = matrix.shape[0]
    dtype = matrix.dtype
    real_dtype = matrix.real.dtype
    left = jnp.ones((n,), dtype=real_dtype)
    right = jnp.ones((n,), dtype=real_dtype)
    converged = jnp.asarray(False)
    steps = jnp.asarray(0, dtype=jnp.int32)

    def step(_, state):
        left_, right_, scaled, converged_, steps_ = state
        row_norm = jnp.max(jnp.abs(scaled), axis=1)
        if policy.mode == "symmetric-ruiz":
            raw = jnp.where(row_norm > 0, jax.lax.rsqrt(row_norm), 1.0)
            left_step = right_step = raw
        else:
            column_norm = jnp.max(jnp.abs(scaled), axis=0)
            left_step = jnp.where(row_norm > 0, jax.lax.rsqrt(row_norm), 1.0)
            right_step = jnp.where(column_norm > 0, jax.lax.rsqrt(column_norm), 1.0)
        active = ~converged_
        left_step = jnp.where(active, left_step, 1.0)
        right_step = jnp.where(active, right_step, 1.0)
        candidate_left = jnp.clip(
            left_ * left_step,
            policy.minimum_scale,
            policy.maximum_scale,
        )
        candidate_right = jnp.clip(
            right_ * right_step,
            policy.minimum_scale,
            policy.maximum_scale,
        )
        scaled_ = candidate_left[:, None] * matrix * candidate_right[None, :]
        row_after = jnp.max(jnp.abs(scaled_), axis=1)
        column_after = jnp.max(jnp.abs(scaled_), axis=0)
        positive = jnp.concatenate((row_after, column_after))
        error = jnp.max(
            jnp.abs(jnp.log(jnp.maximum(positive, jnp.finfo(real_dtype).tiny)))
        )
        converged_after = converged_ | (error <= policy.tolerance)
        return (
            candidate_left,
            candidate_right,
            scaled_,
            converged_after,
            steps_ + active.astype(jnp.int32),
        )

    left, right, scaled, converged, steps = jax.lax.fori_loop(
        0,
        policy.max_steps,
        step,
        (left, right, matrix, converged, steps),
    )
    return left.astype(dtype), right.astype(dtype), scaled, converged, steps


def _placeholder_transform(
    problem: LinearSystem,
    policy: EquilibrationPolicy,
    /,
) -> DiagonalSystemTransform:
    n = problem.operator.source.size
    dtype = _coordinate_dtype(problem.operator.source)
    real_dtype = np.empty((), dtype=dtype).real.dtype
    one = jnp.ones((n,), dtype=dtype)
    return DiagonalSystemTransform(
        left_scale=one,
        right_scale=one,
        converged=jnp.asarray(True),
        steps=jnp.asarray(0, dtype=jnp.int32),
        row_spread=jnp.asarray(1.0, dtype=real_dtype),
        column_spread=jnp.asarray(1.0, dtype=real_dtype),
        mode=policy.mode,
        transform_id=_transform_id(problem, policy),
    )


def _transformed_problem(
    problem: LinearSystem,
    transform: DiagonalSystemTransform,
    /,
) -> LinearSystem:
    congruence = transform.mode in ("none", "symmetric-ruiz")
    operator = TwoSidedScaledLinearOperator(
        problem.operator,
        transform.left_scale,
        None if congruence else transform.right_scale,
        congruence=congruence,
        operator_id=f"{problem.operator.operator_id}:equilibrated:{transform.mode}",
    )
    return LinearSystem(
        operator,
        nullspace_policy=problem.nullspace_policy,
        problem_id=f"{problem.problem_id}:equilibrated:{transform.mode}",
    )


def _residual(operator, coordinates: Array, rhs: Array, /):
    applied = _operator_columns(operator, coordinates)
    residual = rhs - applied
    return (
        residual,
        _column_norm(operator.target, residual),
        _column_norm(operator.target, applied),
    )


def _operator_columns(operator, coordinates: Array, /) -> Array:
    def apply(column):
        value = operator.mv(operator.source.unflatten(column))
        return operator.target.flatten(value)

    return jax.vmap(apply, in_axes=1, out_axes=1)(coordinates)


def _column_norm(space, coordinates: Array, /) -> Array:
    def norm(column):
        value = space.unflatten(column)
        squared = jnp.real(space.inner(value, value))
        return jnp.sqrt(jnp.maximum(squared, 0.0))

    return jax.vmap(norm, in_axes=1)(coordinates)


def _matrix_spreads(matrix: Array, /) -> tuple[Array, Array]:
    tiny = jnp.finfo(matrix.real.dtype).tiny
    row = jnp.max(jnp.abs(matrix), axis=1)
    column = jnp.max(jnp.abs(matrix), axis=0)
    row_spread = jnp.max(row) / jnp.maximum(jnp.min(row), tiny)
    column_spread = jnp.max(column) / jnp.maximum(jnp.min(column), tiny)
    return row_spread, column_spread


def _requires_materialization(policy: EquilibrationPolicy, /) -> bool:
    return policy.mode in ("ruiz", "symmetric-ruiz") or policy.diagnose_condition


def _validate_problem(problem: LinearSystem, /) -> None:
    if not isinstance(problem, LinearSystem):
        raise TypeError("problem must be a LinearSystem.")
    operator = problem.operator
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Resilient solves require an unbatched operator endomorphism.")


def _validate_explicit_scales(
    problem: LinearSystem,
    policy: EquilibrationPolicy,
    /,
) -> None:
    if policy.mode != "explicit":
        return
    left_scale = policy.left_scale
    right_scale = policy.right_scale
    if left_scale is None or right_scale is None:
        raise ValueError("Explicit equilibration requires both scaling arrays.")
    n = problem.operator.source.size
    dtype = _coordinate_dtype(problem.operator.source)
    if left_scale.shape != (n,) or right_scale.shape != (n,):
        raise ValueError("Explicit equilibration scales must match the system dimension.")
    if np.dtype(left_scale.dtype) != dtype or np.dtype(right_scale.dtype) != dtype:
        raise TypeError("Explicit equilibration dtype must match system coordinates.")


def _validate_plan_problem(
    plan: ResilientSolvePlan,
    problem: LinearSystem,
    /,
) -> None:
    _validate_problem(problem)
    if (
        problem.problem_id != plan.problem_id
        or problem.operator.operator_id != plan.operator_id
        or problem.operator.source.space_id != plan.space_id
        or problem.operator.source.size != plan.dimension
    ):
        raise ValueError("Resilient numeric binding changed symbolic problem structure.")


def _transform_id(problem: LinearSystem, policy: EquilibrationPolicy, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "diagonal-system-transform",
            "problem": problem.problem_id,
            "operator": problem.operator.operator_id,
            "mode": policy.mode,
        }
    )


def _restore_axes(value: Array, layout, /) -> Array:
    return jnp.asarray(value).reshape(layout.rhs_shape)


__all__ = [
    "DiagonalSystemTransform",
    "EquilibrationMode",
    "EquilibrationPolicy",
    "PreparedResilientSolve",
    "RefinementPolicy",
    "ResilienceResourcePolicy",
    "ResilientCostEstimate",
    "ResilientSolveDiagnostics",
    "ResilientSolvePlan",
    "ResilientSolvePolicy",
    "ResilientSolveProvenance",
    "ResilientSolveResult",
    "ResilientSolveStatus",
    "plan_resilient_solve",
    "prepare_resilient_solve",
    "refresh_resilient_solve",
    "solve_resilient",
]
