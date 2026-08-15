#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from ._binding import LinearSolveTemplate
from ._gcrodr import refresh_recycling, solve_recycled
from ._operators import AbstractLinearOperator, adjoint, transpose
from ._plans import LinearSolvePlan, plan as make_plan
from ._policies import GeneralizedLSMR, LinearSolvePolicy, LSMR
from ._preconditioning import prepare_preconditioner, PreparedPreconditioner
from ._prepared import PreparedLinearSolve
from ._problems import (
    _problem_structure,
    AbstractLinearProblem,
    LeastSquaresProblem,
    LinearSystem,
    MinimumNormProblem,
)
from ._results import (
    LinearSolveDiagnostics,
    LinearSolveProvenance,
    LinearSolveResult,
    LinearSolveStatus,
)
from ._spaces import RHSLayout
from ._subspaces import KernelCertificate, LinearSubspace, NullspacePolicy
from .backends._jax_dense import (
    DenseCholeskyState,
    DenseLUState,
    DenseQRState,
    DenseSVDState,
)
from .backends._jax_sparse import HostSparseState
from .backends._native_block_krylov import NativeBlockKrylovBackendOutput
from .backends._provider import provider_for


_ProblemT = TypeVar("_ProblemT", bound=AbstractLinearProblem)


class _PackedRHSLayout(NamedTuple):
    rhs_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    broadcast_batch: bool


def prepare_template(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy | LinearSolvePlan | None = None,
    /,
    *,
    rhs_layout: RHSLayout | None = None,
) -> LinearSolveTemplate:
    """Plan and analyze coefficient-independent solve structure."""
    selected_plan = _select_plan(problem, policy, rhs_layout=rhs_layout)
    return _template_for_plan(problem, selected_plan)


def bind_numeric(
    template: LinearSolveTemplate,
    problem: AbstractLinearProblem,
    /,
    *,
    numeric_version: Any = 0,
) -> PreparedLinearSolve:
    """Bind current coefficients to a reusable symbolic solve template."""
    return _bind_for_template(
        template,
        problem,
        numeric_version=numeric_version,
    )


def prepare(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy | LinearSolvePlan | None = None,
    /,
    *,
    rhs_layout: RHSLayout | None = None,
) -> PreparedLinearSolve:
    """Compose symbolic analysis and numerical binding for one problem."""
    selected_plan = _select_plan(problem, policy, rhs_layout=rhs_layout)
    return _prepare_for_plan(problem, selected_plan)


def _template_for_plan(
    problem: AbstractLinearProblem,
    selected_plan: LinearSolvePlan,
    /,
) -> LinearSolveTemplate:
    provider = provider_for(selected_plan.backend)
    symbolic_state = provider.analyze(problem, selected_plan)
    device_bindable = selected_plan.backend not in ("host-sparse", "lineax")
    rejection_reason = (
        None
        if device_bindable
        else f"backend {selected_plan.backend!r} requires host-side numerical setup"
    )
    return LinearSolveTemplate(
        selected_plan,
        symbolic_state,
        device_bindable=device_bindable,
        source_space_id=problem.operator.source.space_id,
        target_space_id=problem.operator.target.space_id,
        batch_shape=problem.operator.batch_shape,
        rejection_reason=rejection_reason,
    )


def _prepare_for_plan(
    problem: AbstractLinearProblem,
    selected_plan: LinearSolvePlan,
    /,
) -> PreparedLinearSolve:
    template = _template_for_plan(problem, selected_plan)
    return _bind_for_template(template, problem)


def _select_plan(
    problem: AbstractLinearProblem,
    policy: LinearSolvePolicy | LinearSolvePlan | None,
    /,
    *,
    rhs_layout: RHSLayout | None,
) -> LinearSolvePlan:
    if not isinstance(problem, AbstractLinearProblem):
        raise TypeError("problem must be an AbstractLinearProblem.")
    if rhs_layout is not None and not isinstance(rhs_layout, RHSLayout):
        raise TypeError("rhs_layout must be an RHSLayout or None.")
    if not isinstance(policy, LinearSolvePlan):
        return make_plan(problem, policy, rhs_layout=rhs_layout)
    selected_plan = policy
    if rhs_layout is not None and (
        selected_plan.rhs_layout is None
        or rhs_layout.layout_id != selected_plan.rhs_layout.layout_id
    ):
        raise ValueError("rhs_layout must match the supplied plan exactly.")
    if selected_plan.problem_id != problem.problem_id:
        raise ValueError("Plan and problem IDs must match.")
    if selected_plan.problem_signature != _problem_structure(problem):
        raise ValueError(
            "Plan reuse cannot change operator structure, weights or regularizers, "
            "or nullspace policies."
        )
    refreshed_plan = make_plan(
        problem,
        selected_plan.policy,
        rhs_layout=selected_plan.rhs_layout,
    )
    if refreshed_plan.plan_id != selected_plan.plan_id:
        raise ValueError("Plan does not match the problem's symbolic structure.")
    return selected_plan


def refresh(
    prepared: PreparedLinearSolve,
    problem: AbstractLinearProblem,
    /,
    *,
    setup_operator: AbstractLinearOperator | None = None,
) -> PreparedLinearSolve:
    """Rebind numerical state while preserving one symbolic solve template."""
    if not isinstance(prepared, PreparedLinearSolve):
        raise TypeError("prepared must be a PreparedLinearSolve.")
    if not isinstance(problem, AbstractLinearProblem):
        raise TypeError("problem must be an AbstractLinearProblem.")
    if problem.problem_id != prepared.problem.problem_id:
        raise ValueError("Numeric refreshes must preserve problem_id.")
    policy = prepared.plan.policy
    preconditioning = policy.preconditioning
    if setup_operator is not None:
        if preconditioning is None:
            raise ValueError("setup_operator requires a preconditioning policy.")
        replacement = preconditioning.with_setup_operator(setup_operator)
        policy = eqx.tree_at(
            lambda selected: selected.preconditioning,
            policy,
            replacement,
        )
    elif (
        preconditioning is not None
        and preconditioning.builder is not None
        and preconditioning.setup_operator is not None
        and preconditioning.refresh_policy != "frozen"
    ):
        raise ValueError(
            "Refreshing a distinct setup operator requires setup_operator=...; "
            "silent reuse after coefficient changes is forbidden."
        )
    refreshed_plan = make_plan(
        problem,
        policy,
        rhs_layout=prepared.plan.rhs_layout,
    )
    if refreshed_plan.plan_id != prepared.plan.plan_id:
        raise ValueError("Numeric refreshes must preserve the symbolic solve plan.")
    refreshed_template = LinearSolveTemplate(
        refreshed_plan,
        prepared.template.symbolic_state,
        device_bindable=prepared.template.device_bindable,
        source_space_id=prepared.template.source_space_id,
        target_space_id=prepared.template.target_space_id,
        batch_shape=prepared.template.batch_shape,
        rejection_reason=prepared.template.rejection_reason,
        schema_version=prepared.template.schema_version,
    )
    if refreshed_template.template_id != prepared.template.template_id:
        raise ValueError("Numeric refreshes must preserve the symbolic solve template.")
    return _bind_for_template(
        refreshed_template,
        problem,
        previous_preconditioner=prepared.preconditioning_state,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def _bind_for_template(
    template: LinearSolveTemplate,
    problem: AbstractLinearProblem,
    /,
    *,
    previous_preconditioner: PreparedPreconditioner | None = None,
    numeric_version: Any = 0,
) -> PreparedLinearSolve:
    if not isinstance(template, LinearSolveTemplate):
        raise TypeError("template must be a LinearSolveTemplate.")
    if not isinstance(problem, AbstractLinearProblem):
        raise TypeError("problem must be an AbstractLinearProblem.")
    selected_plan = template.plan
    if selected_plan.problem_id != problem.problem_id:
        raise ValueError("Template and problem IDs must match.")
    if (
        problem.operator.source.space_id != template.source_space_id
        or problem.operator.target.space_id != template.target_space_id
        or problem.operator.batch_shape != template.batch_shape
    ):
        raise ValueError("Numerical binding cannot change symbolic problem structure.")
    if template.problem_signature != _problem_structure(problem):
        raise ValueError("Numerical binding cannot change symbolic problem structure.")
    stop_arrays = selected_plan.policy.differentiation.mode in ("rhs-only", "none")
    execution_problem = _stop_problem_arrays(problem) if stop_arrays else problem
    preparation_plan = (
        jax.tree.map(
            lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
            selected_plan,
        )
        if stop_arrays
        else selected_plan
    )
    preconditioning_state = prepare_preconditioner(
        preparation_plan.preconditioner_plan,
        execution_problem.operator,
        materialization=preparation_plan.policy.materialization,
        previous=previous_preconditioner,
        numeric_version=numeric_version,
    )
    action = None if preconditioning_state is None else preconditioning_state.action
    provider = provider_for(selected_plan.backend)
    state = provider.bind(
        template.symbolic_state,
        execution_problem,
        preparation_plan,
        preconditioner=action,
    )
    return PreparedLinearSolve(
        problem,
        template,
        state,
        preconditioning_state=preconditioning_state,
        numeric_version=numeric_version,
    )


def _preconditioner_provenance(
    prepared: PreparedLinearSolve,
    /,
) -> dict[str, Any]:
    state = prepared.preconditioning_state
    if state is None:
        return {
            "preconditioner_plan_id": None,
            "preconditioner_id": None,
            "preconditioning_side": None,
            "preconditioner_refresh": None,
            "preconditioner_numeric_version": -1,
            "preconditioner_built_numeric_version": -1,
            "preconditioner_storage_bytes": 0,
            "preconditioner_preparation_workspace_bytes": 0,
            "preconditioner_apply_workspace_bytes_per_rhs": 0,
            "preconditioner_setup_matvec_count": 0,
        }
    return {
        "preconditioner_plan_id": state.plan.plan_id,
        "preconditioner_id": state.action.preconditioner_id,
        "preconditioning_side": state.plan.side,
        "preconditioner_refresh": state.refresh_kind,
        "preconditioner_numeric_version": state.numeric_version,
        "preconditioner_built_numeric_version": state.built_numeric_version,
        "preconditioner_storage_bytes": state.plan.cost.storage_bytes,
        "preconditioner_preparation_workspace_bytes": (
            state.plan.cost.preparation_workspace_bytes
        ),
        "preconditioner_apply_workspace_bytes_per_rhs": (
            state.plan.cost.apply_workspace_bytes_per_rhs
        ),
        "preconditioner_setup_matvec_count": state.plan.cost.setup_matvec_count,
    }


def _execution_rhs_layout(
    prepared: PreparedLinearSolve,
    rhs_layout: RHSLayout | None,
    /,
) -> RHSLayout | None:
    if rhs_layout is not None and not isinstance(rhs_layout, RHSLayout):
        raise TypeError("rhs_layout must be an RHSLayout or None.")
    planned_layout = prepared.plan.rhs_layout
    if planned_layout is None:
        return rhs_layout
    if rhs_layout is not None and rhs_layout.layout_id != planned_layout.layout_id:
        raise ValueError("Execution rhs_layout must match the prepared plan exactly.")
    return planned_layout


def solve(
    problem_or_prepared: AbstractLinearProblem | PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    policy: LinearSolvePolicy | LinearSolvePlan | None = None,
    rhs_layout: RHSLayout | None = None,
    initial_guess: PyTree[Any] | None = None,
) -> LinearSolveResult:
    """Solve one or many right-hand sides with explicit status evidence."""
    if isinstance(problem_or_prepared, PreparedLinearSolve):
        if policy is not None:
            raise ValueError("policy must be omitted when solving prepared state.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, AbstractLinearProblem):
        if isinstance(policy, LinearSolvePlan):
            planned_layout = policy.rhs_layout if rhs_layout is None else rhs_layout
            if planned_layout is None:
                canonical_rhs, _ = _pack_rhs(
                    problem_or_prepared.operator.target,
                    problem_or_prepared.operator.batch_shape,
                    rhs,
                )
                _require_rhs_resources(policy, int(canonical_rhs.shape[-1]))
        else:
            planned_layout = rhs_layout
            if planned_layout is None:
                _, inferred_layout = _pack_rhs(
                    problem_or_prepared.operator.target,
                    problem_or_prepared.operator.batch_shape,
                    rhs,
                )
                if inferred_layout.rhs_shape:
                    planned_layout = RHSLayout(inferred_layout.rhs_shape)
        prepared = prepare(
            problem_or_prepared,
            policy,
            rhs_layout=planned_layout,
        )
        rhs_layout = planned_layout
    else:
        raise TypeError("Expected an AbstractLinearProblem or PreparedLinearSolve.")
    declared_layout = _execution_rhs_layout(prepared, rhs_layout)

    problem = (
        _stop_problem_arrays(prepared.problem)
        if prepared.plan.policy.differentiation.mode in ("rhs-only", "none")
        else prepared.problem
    )
    canonical_rhs, layout = _pack_rhs(
        problem.operator.target,
        problem.operator.batch_shape,
        rhs,
        declared_layout,
    )
    _require_rhs_resources(prepared.plan, int(canonical_rhs.shape[-1]))
    canonical_rhs, compatibility_residual = _apply_nullspace_compatibility(
        problem,
        canonical_rhs,
        prepared.plan,
    )
    canonical_guess = None
    if initial_guess is not None:
        if not provider_for(prepared.plan.backend).accepts_initial_guess:
            raise ValueError("This provider does not accept an initial_guess.")
        canonical_guess, guess_layout = _pack_rhs(
            problem.operator.source,
            problem.operator.batch_shape,
            initial_guess,
            (
                None
                if not layout.rhs_shape
                else (
                    declared_layout
                    if declared_layout is not None
                    else RHSLayout(layout.rhs_shape)
                )
            ),
        )
        if guess_layout.rhs_shape != layout.rhs_shape:
            raise ValueError("initial_guess RHS axes must match rhs.")
        if isinstance(problem, MinimumNormProblem):
            canonical_guess = eqx.error_if(
                canonical_guess,
                jnp.any(canonical_guess != 0),
                "MinimumNormProblem initial_guess must be zero.",
            )
    provider = provider_for(prepared.plan.backend)
    backend = provider.solve(
        prepared.state,
        canonical_rhs,
        prepared.plan,
        initial_guess=canonical_guess,
    )

    if (
        prepared.plan.policy.differentiation.mode in ("mathematical", "rhs-only")
        and provider.supports_implicit_differentiation
        and isinstance(
            problem,
            (LinearSystem, LeastSquaresProblem, MinimumNormProblem),
        )
    ):
        backend = eqx.tree_at(
            lambda output: output.value,
            backend,
            _implicit_root_value(prepared, problem, canonical_rhs, backend.value),
        )

    canonical_value = (
        jax.lax.stop_gradient(backend.value)
        if prepared.plan.policy.differentiation.mode == "none"
        else backend.value
    )
    canonical_value, gauge_residual, nullity = _apply_nullspace_gauge(
        problem,
        canonical_value,
    )
    canonical_residual = (
        _canonical_action(prepared, problem, canonical_value) - canonical_rhs
    )
    residual_norm = _coordinate_norm(problem.operator.target, canonical_residual)
    rhs_norm = _coordinate_norm(problem.operator.target, canonical_rhs)
    relative_residual = jnp.where(rhs_norm > 0.0, residual_norm / rhs_norm, residual_norm)
    roundoff_relative = (
        10.0
        * jnp.finfo(canonical_rhs.real.dtype).eps
        * float(max(problem.operator.source.size, problem.operator.target.size))
    )
    effective_relative = jnp.maximum(
        prepared.plan.policy.tolerance.relative,
        roundoff_relative,
    )
    threshold = prepared.plan.policy.tolerance.absolute + effective_relative * rhs_norm

    status = backend.status
    normal_residual = jnp.full_like(residual_norm, jnp.nan)
    convergence_measure = residual_norm
    convergence_threshold = threshold
    if isinstance(problem, LeastSquaresProblem):
        normal_residual, normal_reference = _normal_residual(
            prepared,
            problem,
            canonical_rhs,
            canonical_value,
        )
        convergence_measure = normal_residual
        convergence_threshold = (
            prepared.plan.policy.tolerance.absolute
            + effective_relative * normal_reference
        )
    status = jnp.where(
        (status == int(LinearSolveStatus.SUCCESS))
        & (convergence_measure > convergence_threshold),
        int(LinearSolveStatus.RESIDUAL_TOO_LARGE),
        status,
    )

    rhs_finite = jnp.all(jnp.isfinite(canonical_rhs), axis=-2)
    value_finite = (
        jnp.all(jnp.isfinite(canonical_value), axis=-2)
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(convergence_measure)
    )
    finite = rhs_finite & value_finite
    status = jnp.where(
        ~rhs_finite,
        int(LinearSolveStatus.NONFINITE_INPUT),
        status,
    )
    status = jnp.where(
        rhs_finite & ~value_finite & (status == int(LinearSolveStatus.SUCCESS)),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    )
    converged = status == int(LinearSolveStatus.SUCCESS)
    status_out = _restore_rhs_axes(status, layout)
    residual_out = _restore_rhs_axes(residual_norm, layout)
    relative_out = _restore_rhs_axes(relative_residual, layout)
    normal_out = _restore_rhs_axes(normal_residual, layout)
    finite_out = _restore_rhs_axes(finite, layout)
    converged_out = _restore_rhs_axes(converged, layout)
    iterations_out = _restore_rhs_axes(
        jnp.broadcast_to(backend.iterations, status.shape), layout
    )
    rank_out = _restore_rhs_axes(_rhs_broadcast(backend.rank, status.shape), layout)
    condition_out = _restore_rhs_axes(
        _rhs_broadcast(backend.condition_estimate, status.shape), layout
    )
    if prepared.plan.backend in (
        "native-krylov",
        "native-block-krylov",
        "matfree",
        "lineax",
    ):
        matvec_count_out = _restore_rhs_axes(
            _rhs_broadcast(backend.matvec_count, status.shape), layout
        )
        adjoint_matvec_count_out = _restore_rhs_axes(
            _rhs_broadcast(backend.adjoint_matvec_count, status.shape), layout
        )
    else:
        zero_counts = jnp.zeros(status.shape, dtype=jnp.int32)
        matvec_count_out = _restore_rhs_axes(zero_counts, layout)
        adjoint_matvec_count_out = matvec_count_out
    if isinstance(backend, NativeBlockKrylovBackendOutput):
        effective_block_rank_out = _restore_rhs_axes(
            _rhs_broadcast(backend.effective_block_rank, status.shape),
            layout,
        )
        deflated_rhs_count_out = _restore_rhs_axes(
            _rhs_broadcast(backend.deflated_rhs_count, status.shape),
            layout,
        )
    else:
        effective_block_rank_out = _restore_rhs_axes(
            jnp.full(status.shape, -1, dtype=jnp.int32),
            layout,
        )
        deflated_rhs_count_out = _restore_rhs_axes(
            jnp.zeros(status.shape, dtype=jnp.int32),
            layout,
        )
    value = _unpack_value(problem.operator.source, canonical_value, layout)
    diagnostics = LinearSolveDiagnostics(
        residual_norm=residual_out,
        relative_residual=relative_out,
        normal_residual_norm=normal_out,
        iterations=iterations_out,
        rank=rank_out,
        condition_estimate=condition_out,
        finite=finite_out,
        converged=converged_out,
        compatibility_residual=_restore_rhs_axes(
            _rhs_broadcast(compatibility_residual, status.shape),
            layout,
        ),
        gauge_residual=_restore_rhs_axes(
            _rhs_broadcast(gauge_residual, status.shape),
            layout,
        ),
        nullity=_restore_rhs_axes(
            _rhs_broadcast(nullity, status.shape),
            layout,
        ),
        matvec_count=matvec_count_out,
        adjoint_matvec_count=adjoint_matvec_count_out,
        effective_block_rank=effective_block_rank_out,
        deflated_rhs_count=deflated_rhs_count_out,
        singular_values=backend.singular_values,
    )
    provenance = LinearSolveProvenance(
        backend=prepared.plan.backend,
        method=prepared.plan.method,
        plan_id=prepared.plan.plan_id,
        problem_id=problem.problem_id,
        reason=prepared.plan.reason,
        rejected=prepared.plan.rejected,
        prepared=True,
        rhs_mode=(
            "true-block"
            if prepared.plan.backend == "native-block-krylov"
            else "pseudo-block"
            if layout.rhs_shape
            else "single"
        ),
        operator_numeric_version=prepared.numeric_version,
        recycling_capacity=prepared.plan.recycling_capacity,
        recycling_state_bytes=prepared.plan.recycling_state_bytes,
        **_preconditioner_provenance(prepared),
    )
    if prepared.plan.policy.failure.mode == "error":
        value = _error_on_failure(value, status_out)
    return LinearSolveResult(value, status_out, diagnostics, provenance)


def solve_many(
    prepared: PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    initial_guess: PyTree[Any] | None = None,
) -> LinearSolveResult:
    """Solve shared trailing RHS axes and broadcast them over operator batches."""
    inferred_layout = _shared_rhs_layout(prepared.problem.operator.target, rhs)
    planned_layout = prepared.plan.rhs_layout
    if planned_layout is not None and planned_layout.shape != inferred_layout.shape:
        raise ValueError("solve_many RHS axes must match the prepared plan exactly.")
    rhs_layout = inferred_layout if planned_layout is None else planned_layout
    return solve(
        prepared,
        rhs,
        rhs_layout=rhs_layout,
        initial_guess=initial_guess,
    )


def solve_transpose(
    problem_or_prepared: LinearSystem | PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    policy: LinearSolvePolicy | None = None,
    rhs_layout: RHSLayout | None = None,
) -> LinearSolveResult:
    """Solve a transposed system with explicitly declared trailing RHS axes.

    When prepared state has a planned RHS layout, ``rhs_layout`` must match it
    exactly and the planned layout remains authoritative.
    """
    declared_layout = rhs_layout
    if isinstance(problem_or_prepared, PreparedLinearSolve):
        if policy is not None:
            raise ValueError("policy must be omitted for prepared transformed solves.")
        declared_layout = _execution_rhs_layout(problem_or_prepared, rhs_layout)
        if not problem_or_prepared.problem.operator.batch_shape and provider_for(
            problem_or_prepared.plan.backend
        ).supports_transformed(problem_or_prepared.state):
            return _solve_prepared_transformed(
                problem_or_prepared,
                rhs,
                rhs_layout=declared_layout,
                adjoint_mode=False,
            )
    problem, selected_policy = _transformed_problem(problem_or_prepared, policy)
    transformed = _transformed_linear_system(problem, adjoint_mode=False)
    return solve(
        transformed,
        rhs,
        policy=selected_policy,
        rhs_layout=declared_layout,
    )


def solve_adjoint(
    problem_or_prepared: LinearSystem | PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    policy: LinearSolvePolicy | None = None,
    rhs_layout: RHSLayout | None = None,
) -> LinearSolveResult:
    """Solve an adjoint system with explicitly declared trailing RHS axes.

    When prepared state has a planned RHS layout, ``rhs_layout`` must match it
    exactly and the planned layout remains authoritative.
    """
    declared_layout = rhs_layout
    if isinstance(problem_or_prepared, PreparedLinearSolve):
        if policy is not None:
            raise ValueError("policy must be omitted for prepared transformed solves.")
        declared_layout = _execution_rhs_layout(problem_or_prepared, rhs_layout)
        if not problem_or_prepared.problem.operator.batch_shape and provider_for(
            problem_or_prepared.plan.backend
        ).supports_transformed(problem_or_prepared.state):
            return _solve_prepared_transformed(
                problem_or_prepared,
                rhs,
                rhs_layout=declared_layout,
                adjoint_mode=True,
            )
    problem, selected_policy = _transformed_problem(problem_or_prepared, policy)
    transformed = _transformed_linear_system(problem, adjoint_mode=True)
    return solve(
        transformed,
        rhs,
        policy=selected_policy,
        rhs_layout=declared_layout,
    )


def _solve_prepared_transformed(
    prepared: PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    rhs_layout: RHSLayout | None,
    adjoint_mode: bool,
) -> LinearSolveResult:
    problem = (
        _stop_problem_arrays(prepared.problem)
        if prepared.plan.policy.differentiation.mode in ("rhs-only", "none")
        else prepared.problem
    )
    if not isinstance(problem, LinearSystem):
        raise TypeError("Prepared transpose and adjoint reuse requires a LinearSystem.")
    original = problem.operator
    transformed_problem = _transformed_linear_system(
        problem,
        adjoint_mode=adjoint_mode,
    )
    transformed_operator = transformed_problem.operator
    canonical_rhs, layout = _pack_rhs(
        transformed_operator.target,
        transformed_operator.batch_shape,
        rhs,
        rhs_layout,
    )
    _require_rhs_resources(prepared.plan, int(canonical_rhs.shape[-1]))
    canonical_rhs, compatibility_residual = _apply_nullspace_compatibility(
        transformed_problem,
        canonical_rhs,
        prepared.plan,
    )
    backend_rhs = canonical_rhs
    metric_transform = adjoint_mode and isinstance(
        prepared.state, (DenseLUState, HostSparseState)
    )
    if metric_transform:
        backend_rhs = _riesz_coordinates(original.source, backend_rhs)
    provider = provider_for(prepared.plan.backend)
    backend = provider.solve_transformed(
        prepared.state,
        backend_rhs,
        prepared.plan,
        adjoint=adjoint_mode,
    )
    canonical_value = backend.value
    if metric_transform:
        canonical_value = _inverse_riesz_coordinates(
            original.target,
            canonical_value,
        )
    canonical_value, gauge_residual, nullity = _apply_nullspace_gauge(
        transformed_problem,
        canonical_value,
    )
    if prepared.plan.policy.differentiation.mode == "none":
        canonical_value = jax.lax.stop_gradient(canonical_value)

    def one_column(coordinates):
        vector = transformed_operator.source.unflatten(coordinates)
        return transformed_operator.target.flatten(transformed_operator.mv(vector))

    image = jax.vmap(one_column, in_axes=1, out_axes=1)(canonical_value)
    residual = image - canonical_rhs
    residual_norm = _coordinate_norm(transformed_operator.target, residual)
    rhs_norm = _coordinate_norm(transformed_operator.target, canonical_rhs)
    relative = jnp.where(rhs_norm > 0.0, residual_norm / rhs_norm, residual_norm)
    roundoff_relative = (
        10.0
        * jnp.finfo(canonical_rhs.real.dtype).eps
        * float(max(original.source.size, original.target.size))
    )
    effective_relative = jnp.maximum(
        prepared.plan.policy.tolerance.relative,
        roundoff_relative,
    )
    threshold = prepared.plan.policy.tolerance.absolute + effective_relative * rhs_norm
    status = jnp.where(
        (backend.status == int(LinearSolveStatus.SUCCESS)) & (residual_norm > threshold),
        int(LinearSolveStatus.RESIDUAL_TOO_LARGE),
        backend.status,
    )
    rhs_finite = jnp.all(jnp.isfinite(canonical_rhs), axis=-2)
    value_finite = jnp.all(jnp.isfinite(canonical_value), axis=-2) & jnp.isfinite(
        residual_norm
    )
    finite = rhs_finite & value_finite
    status = jnp.where(
        ~rhs_finite,
        int(LinearSolveStatus.NONFINITE_INPUT),
        status,
    )
    status = jnp.where(
        rhs_finite & ~value_finite & (status == int(LinearSolveStatus.SUCCESS)),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    )
    status_out = _restore_rhs_axes(status, layout)
    diagnostics = LinearSolveDiagnostics(
        residual_norm=_restore_rhs_axes(residual_norm, layout),
        relative_residual=_restore_rhs_axes(relative, layout),
        iterations=_restore_rhs_axes(
            jnp.broadcast_to(backend.iterations, status.shape), layout
        ),
        rank=_restore_rhs_axes(_rhs_broadcast(backend.rank, status.shape), layout),
        condition_estimate=_restore_rhs_axes(
            _rhs_broadcast(backend.condition_estimate, status.shape), layout
        ),
        finite=_restore_rhs_axes(finite, layout),
        converged=_restore_rhs_axes(status == int(LinearSolveStatus.SUCCESS), layout),
        compatibility_residual=_restore_rhs_axes(
            _rhs_broadcast(compatibility_residual, status.shape),
            layout,
        ),
        gauge_residual=_restore_rhs_axes(
            _rhs_broadcast(gauge_residual, status.shape),
            layout,
        ),
        nullity=_restore_rhs_axes(
            _rhs_broadcast(nullity, status.shape),
            layout,
        ),
        singular_values=backend.singular_values,
    )
    value = _unpack_value(transformed_operator.source, canonical_value, layout)
    if prepared.plan.policy.failure.mode == "error":
        value = _error_on_failure(value, status_out)
    provenance = LinearSolveProvenance(
        backend=prepared.plan.backend,
        method=(
            f"{prepared.plan.method}-adjoint"
            if adjoint_mode
            else f"{prepared.plan.method}-transpose"
        ),
        plan_id=prepared.plan.plan_id,
        problem_id=prepared.problem.problem_id,
        reason="reused prepared direct factorization",
        rejected=prepared.plan.rejected,
        prepared=True,
        rhs_mode="pseudo-block" if layout.rhs_shape else "single",
        operator_numeric_version=prepared.numeric_version,
        recycling_capacity=prepared.plan.recycling_capacity,
        recycling_state_bytes=prepared.plan.recycling_state_bytes,
        **_preconditioner_provenance(prepared),
    )
    return LinearSolveResult(value, status_out, diagnostics, provenance)


def _transformed_problem(
    value: LinearSystem | PreparedLinearSolve,
    policy: LinearSolvePolicy | None,
    /,
) -> tuple[LinearSystem, LinearSolvePolicy]:
    if isinstance(value, PreparedLinearSolve):
        if policy is not None:
            raise ValueError("policy must be omitted for prepared transformed solves.")
        problem = value.problem
        selected = value.plan.policy
    else:
        problem = value
        selected = LinearSolvePolicy() if policy is None else policy
    if not isinstance(problem, LinearSystem):
        raise TypeError("Transpose and adjoint solves require a LinearSystem.")
    return problem, selected


def _transformed_linear_system(
    problem: LinearSystem,
    /,
    *,
    adjoint_mode: bool,
) -> LinearSystem:
    operator = adjoint(problem.operator) if adjoint_mode else transpose(problem.operator)
    policy = problem.nullspace_policy
    if policy is None:
        return LinearSystem(operator)
    if adjoint_mode:
        right = policy.left
        left = policy.right
    else:
        right = _transpose_right_subspace(policy.left, problem.operator.target)
        left = _transpose_left_subspace(policy.right, problem.operator.source)
    certificate = (
        None
        if policy.certificate is None
        else KernelCertificate(
            operator,
            right,
            left=left,
            evidence=policy.certificate.evidence,
            scope=policy.certificate.scope,
            complete=policy.certificate.complete,
            tolerance=policy.certificate.tolerance,
        )
    )
    return LinearSystem(
        operator,
        nullspace_policy=NullspacePolicy(
            right=right,
            left=left,
            certificate=certificate,
            compatibility=policy.compatibility,
            gauge=policy.gauge,
        ),
    )


def _transpose_right_subspace(
    subspace: LinearSubspace | None,
    space,
    /,
) -> LinearSubspace | None:
    if subspace is None:
        return None

    def transform(column):
        vector = space.unflatten(column)
        return jnp.conj(space.flatten(space.riesz(vector)))

    basis = jax.vmap(transform, in_axes=1, out_axes=1)(subspace.basis)
    return LinearSubspace(
        space,
        basis,
        dimension=subspace.dimension,
        subspace_id=f"{subspace.subspace_id}:transpose-right",
    )


def _transpose_left_subspace(
    subspace: LinearSubspace | None,
    space,
    /,
) -> LinearSubspace | None:
    if subspace is None:
        return None

    def transform(column):
        covector = space.unflatten(jnp.conj(column))
        return space.flatten(space.inverse_riesz(covector))

    basis = jax.vmap(transform, in_axes=1, out_axes=1)(subspace.basis)
    return LinearSubspace(
        space,
        basis,
        dimension=subspace.dimension,
        subspace_id=f"{subspace.subspace_id}:transpose-left",
    )


def _require_rhs_resources(
    plan: LinearSolvePlan,
    rhs_count: int,
    /,
) -> None:
    estimate = plan.candidates[-1]
    required_krylov = rhs_count * estimate.krylov_basis_bytes_per_rhs
    available_krylov = plan.policy.resources.krylov_basis_bytes
    if required_krylov > available_krylov:
        raise ValueError(
            f"Selected {estimate.method} requires {required_krylov} Krylov basis "
            f"bytes for {rhs_count} right-hand sides, exceeding the policy budget "
            f"{available_krylov}."
        )
    required_workspace = rhs_count * (
        estimate.solve_workspace_bytes_per_rhs
        + estimate.preconditioner_apply_workspace_bytes_per_rhs
    )
    available_workspace = plan.policy.resources.workspace_bytes
    if required_workspace > available_workspace:
        raise ValueError(
            f"Selected {estimate.method} requires {required_workspace} workspace "
            f"bytes for {rhs_count} right-hand sides, exceeding the policy budget "
            f"{available_workspace}."
        )


def _pack_rhs(
    space,
    batch_shape: tuple[int, ...],
    rhs: PyTree[Any],
    declared_layout: RHSLayout | None = None,
    /,
):
    specifications, expected_tree = jax.tree.flatten(space.structure())
    values, actual_tree = jax.tree.flatten(rhs)
    if actual_tree != expected_tree:
        raise ValueError("Right-hand-side PyTree structure does not match target space.")
    arrays = tuple(jnp.asarray(value) for value in values)
    if declared_layout is not None and not isinstance(declared_layout, RHSLayout):
        raise TypeError("rhs_layout must be an RHSLayout or None.")
    for array, specification in zip(arrays, specifications, strict=True):
        if np.dtype(array.dtype) != np.dtype(specification.dtype):
            raise TypeError(
                f"Right-hand-side dtype must be {specification.dtype}; got {array.dtype}."
            )

    if declared_layout is None:
        modes = ((True, False), (True, True), (False, False), (False, True))
        if not batch_shape:
            modes = ((False, False), (False, True))
    else:
        multiple = bool(declared_layout.shape)
        modes = (
            ((True, multiple), (False, multiple)) if batch_shape else ((False, multiple),)
        )
    selected: tuple[bool, tuple[int, ...]] | None = None
    for batched, multiple in modes:
        trailing: tuple[int, ...] | None = None
        valid = True
        for array, specification in zip(arrays, specifications, strict=True):
            prefix = (batch_shape if batched else ()) + tuple(specification.shape)
            if array.shape[: len(prefix)] != prefix:
                valid = False
                break
            remainder = tuple(int(size) for size in array.shape[len(prefix) :])
            if declared_layout is not None and remainder != declared_layout.shape:
                valid = False
                break
            if (not multiple and remainder) or (multiple and not remainder):
                valid = False
                break
            if trailing is None:
                trailing = remainder
            elif trailing != remainder:
                valid = False
                break
        if valid:
            selected = (batched, () if trailing is None else trailing)
            break
    if selected is None:
        raise ValueError(
            "Right-hand sides must have event shape, optional operator batch axes, "
            "and optional shared trailing RHS axes."
        )
    batched, rhs_shape = selected
    rhs_count = prod(rhs_shape) if rhs_shape else 1
    flattened = []
    for array, specification in zip(arrays, specifications, strict=True):
        event_shape = tuple(specification.shape)
        target_shape = batch_shape + event_shape + rhs_shape
        if not batched and batch_shape:
            array = jnp.broadcast_to(array, target_shape)
        flattened.append(array.reshape(batch_shape + (prod(event_shape), rhs_count)))
    canonical = (
        flattened[0] if len(flattened) == 1 else jnp.concatenate(flattened, axis=-2)
    )
    return canonical, _PackedRHSLayout(rhs_shape, batch_shape, not batched)


def _unpack_value(space, value: Array, layout: _PackedRHSLayout, /) -> PyTree[Array]:
    specifications, tree = jax.tree.flatten(space.structure())
    leaves = []
    offset = 0
    for specification in specifications:
        count = prod(specification.shape)
        shape = layout.batch_shape + tuple(specification.shape) + layout.rhs_shape
        leaf = value[..., offset : offset + count, :].reshape(shape)
        leaves.append(leaf.astype(specification.dtype))
        offset += count
    return jax.tree.unflatten(tree, leaves)


def _shared_rhs_layout(space, rhs: PyTree[Any], /) -> RHSLayout:
    specifications, expected_tree = jax.tree.flatten(space.structure())
    values, actual_tree = jax.tree.flatten(rhs)
    if actual_tree != expected_tree:
        raise ValueError("Right-hand-side PyTree structure does not match target space.")
    trailing: tuple[int, ...] | None = None
    for value, specification in zip(values, specifications, strict=True):
        shape = tuple(jnp.shape(value))
        event_shape = tuple(specification.shape)
        if shape[: len(event_shape)] != event_shape:
            raise ValueError(
                "solve_many expects unbatched event axes followed by shared RHS axes."
            )
        remainder = shape[len(event_shape) :]
        if not remainder:
            raise ValueError("solve_many requires at least one trailing RHS axis.")
        if trailing is None:
            trailing = remainder
        elif trailing != remainder:
            raise ValueError("All solve_many leaves must share trailing RHS axes.")
    if trailing is None:
        raise ValueError("solve_many requires at least one right-hand-side leaf.")
    return RHSLayout(trailing)


def _canonical_action(
    prepared: PreparedLinearSolve,
    problem: AbstractLinearProblem,
    value: Array,
    /,
) -> Array:
    state = prepared.state
    if isinstance(state, (DenseLUState, DenseCholeskyState)):
        return jnp.matmul(state.matrix, value)
    if isinstance(state, (DenseQRState, DenseSVDState)):
        return jnp.matmul(state.original_matrix, value)
    operator = problem.operator

    def one_column(coordinates):
        vector = operator.source.unflatten(coordinates)
        return operator.target.flatten(operator.mv(vector))

    return jax.vmap(one_column, in_axes=1, out_axes=1)(value)


def _normal_residual(
    prepared: PreparedLinearSolve,
    problem: LeastSquaresProblem,
    rhs: Array,
    value: Array,
    /,
) -> tuple[Array, Array]:
    normal = _least_squares_stationarity(
        problem,
        value,
        rhs,
        prepared.plan,
    )
    reference = _least_squares_stationarity(
        problem,
        jnp.zeros_like(value),
        rhs,
        prepared.plan,
    )
    if isinstance(prepared.state, DenseSVDState):
        normal = _dense_svd_active_projection(prepared.state, normal)
        reference = _dense_svd_active_projection(prepared.state, reference)
    source = problem.operator.source
    return _coordinate_norm(source, normal), _coordinate_norm(source, reference)


def _coordinate_norm(space, coordinates: Array, /) -> Array:
    flattened, output_shape = _flatten_coordinate_columns(space, coordinates)

    def norm(column):
        vector = space.unflatten(column)
        return jnp.sqrt(jnp.maximum(jnp.real(space.inner(vector, vector)), 0.0))

    return jax.vmap(norm)(flattened).reshape(output_shape)


def _dual_coordinate_norm(space, coordinates: Array, /) -> Array:
    flattened, output_shape = _flatten_coordinate_columns(space, coordinates)

    def norm(column):
        covector = space.unflatten(column)
        primal = space.inverse_riesz(covector)
        return jnp.sqrt(jnp.maximum(jnp.real(space.inner(primal, primal)), 0.0))

    return jax.vmap(norm)(flattened).reshape(output_shape)


def _riesz_coordinates(space, coordinates: Array, /) -> Array:
    return _map_coordinate_columns(space, coordinates, space.riesz)


def _inverse_riesz_coordinates(space, coordinates: Array, /) -> Array:
    return _map_coordinate_columns(space, coordinates, space.inverse_riesz)


def _map_coordinate_columns(space, coordinates: Array, transform, /) -> Array:
    array = jnp.asarray(coordinates)
    flattened, _ = _flatten_coordinate_columns(space, array)
    mapped = jax.vmap(lambda column: space.flatten(transform(space.unflatten(column))))(
        flattened
    )
    moved_shape = array.shape[:-2] + (array.shape[-1], space.size)
    return jnp.moveaxis(mapped.reshape(moved_shape), -1, -2)


def _project_coordinate_columns(subspace, coordinates: Array, /) -> Array:
    array = jnp.asarray(coordinates)
    flattened, _ = _flatten_coordinate_columns(subspace.space, array)
    projected = jax.vmap(subspace.project_coordinates)(flattened)
    moved_shape = array.shape[:-2] + (array.shape[-1], subspace.space.size)
    return jnp.moveaxis(projected.reshape(moved_shape), -1, -2)


def _flatten_coordinate_columns(
    space,
    coordinates: Array,
    /,
) -> tuple[Array, tuple[int, ...]]:
    array = jnp.asarray(coordinates)
    if array.ndim < 2 or array.shape[-2] != space.size:
        raise ValueError(
            "Canonical coordinate batches must end in (space.size, rhs_count)."
        )
    moved = jnp.moveaxis(array, -2, -1)
    return moved.reshape((prod(moved.shape[:-1]), space.size)), moved.shape[:-1]


def _rhs_broadcast(value: Array, shape: tuple[int, ...], /) -> Array:
    array = jnp.asarray(value)
    if array.shape == shape:
        return array
    if array.shape == shape[:-1]:
        array = array[..., None]
    return jnp.broadcast_to(array, shape)


def _restore_rhs_axes(value: Array, layout: _PackedRHSLayout, /) -> Array:
    target = layout.batch_shape + layout.rhs_shape
    return jnp.asarray(value).reshape(target)


def _apply_nullspace_compatibility(
    problem: AbstractLinearProblem,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
) -> tuple[Array, Array]:
    policy = problem.nullspace_policy
    if policy is None or policy.left is None:
        zero_shape = rhs.shape[:-2] + rhs.shape[-1:]
        return rhs, jnp.zeros(zero_shape, dtype=rhs.real.dtype)

    projections = _project_coordinate_columns(policy.left, rhs)
    residual = _coordinate_norm(problem.operator.target, projections)
    if policy.compatibility == "error":
        threshold = (
            plan.policy.tolerance.absolute
            + plan.policy.tolerance.relative
            * _coordinate_norm(problem.operator.target, rhs)
        )
        rhs = eqx.error_if(
            rhs,
            jnp.any(residual > threshold),
            "Right-hand side is incompatible with the declared left nullspace.",
        )
        return rhs, residual
    return rhs - projections, residual


def _apply_nullspace_gauge(
    problem: AbstractLinearProblem,
    value: Array,
    /,
) -> tuple[Array, Array, Array]:
    policy = problem.nullspace_policy
    if policy is None or policy.right is None:
        zero_shape = value.shape[:-2] + value.shape[-1:]
        return (
            value,
            jnp.zeros(zero_shape, dtype=value.real.dtype),
            jnp.asarray(-1, dtype=jnp.int32),
        )
    projections = _project_coordinate_columns(policy.right, value)
    value = value - projections
    remaining = _project_coordinate_columns(policy.right, value)
    residual = _coordinate_norm(problem.operator.source, remaining)
    return value, residual, policy.right.dimension


def _error_on_failure(value: PyTree[Array], status: Array, /) -> PyTree[Array]:
    leaves, tree = jax.tree.flatten(value)
    leaves[0] = eqx.error_if(
        leaves[0],
        jnp.any(status != int(LinearSolveStatus.SUCCESS)),
        "Linear solve failed; inspect status-mode diagnostics for the failure class.",
    )
    return jax.tree.unflatten(tree, leaves)


def _implicit_root_value(
    prepared: PreparedLinearSolve,
    problem: LinearSystem | LeastSquaresProblem | MinimumNormProblem,
    rhs: Array,
    initial: Array,
    /,
) -> Array:
    if isinstance(problem, MinimumNormProblem):
        return _implicit_minimum_norm_value(prepared, problem, rhs, initial)
    initial = jax.lax.stop_gradient(initial)
    if isinstance(problem, LinearSystem):

        def residual(value):
            return _operator_action(problem.operator, value) - rhs

    else:

        def residual(value):
            return _least_squares_root_residual(
                prepared,
                problem,
                value,
                rhs,
            )

    return _implicit_custom_root(residual, initial, prepared.plan)


def _implicit_minimum_norm_value(
    prepared: PreparedLinearSolve,
    problem: MinimumNormProblem,
    rhs: Array,
    initial: Array,
    /,
) -> Array:
    operator = problem.operator
    operator_adjoint = adjoint(operator)

    def dual_action(value):
        return _operator_action(
            operator,
            _operator_action(operator_adjoint, value),
        )

    multiplier = _solve_independent_columns(dual_action, rhs, prepared.plan)
    source_size = operator.source.size
    augmented_initial = jax.lax.stop_gradient(
        jnp.concatenate((initial, multiplier), axis=-2)
    )

    def residual(augmented):
        value = augmented[..., :source_size, :]
        dual = augmented[..., source_size:, :]
        stationarity = value - _operator_action(operator_adjoint, dual)
        constraint = _operator_action(operator, value) - rhs
        return jnp.concatenate((stationarity, constraint), axis=-2)

    augmented_value = _implicit_custom_root(
        residual,
        augmented_initial,
        prepared.plan,
    )
    return augmented_value[..., :source_size, :]


def _implicit_custom_root(
    residual,
    initial: Array,
    plan: LinearSolvePlan,
    /,
) -> Array:
    def tangent_solve(linearized, target):
        return jax.lax.custom_linear_solve(
            linearized,
            target,
            solve=lambda action, rhs: _solve_independent_columns(
                action,
                rhs,
                plan,
            ),
            transpose_solve=lambda action, rhs: _solve_independent_columns(
                action,
                rhs,
                plan,
            ),
        )

    return jax.lax.custom_root(
        residual,
        initial,
        solve=lambda _, value: value,
        tangent_solve=tangent_solve,
    )


def _solve_independent_columns(
    action,
    right_hand_side: Array,
    plan: LinearSolvePlan,
    /,
) -> Array:
    if right_hand_side.size == 0:
        return jnp.zeros_like(right_hand_side)
    batch_shape = right_hand_side.shape[:-2]
    dimension = right_hand_side.shape[-2]
    rhs_count = right_hand_side.shape[-1]
    batch_count = prod(batch_shape) if batch_shape else 1
    flattened_rhs = right_hand_side.reshape((batch_count, dimension, rhs_count))

    def solve_instance(instance_index):
        batch_index = instance_index // rhs_count
        rhs_index = instance_index % rhs_count
        column = flattened_rhs[batch_index, :, rhs_index]

        def vector_action(vector):
            embedded = (
                jnp.zeros_like(flattened_rhs)
                .at[
                    batch_index,
                    :,
                    rhs_index,
                ]
                .set(vector)
            )
            image = action(embedded.reshape(right_hand_side.shape))
            return image.reshape(flattened_rhs.shape)[batch_index, :, rhs_index]

        return _callable_gmres(vector_action, column, plan)

    solved = jax.vmap(solve_instance)(jnp.arange(batch_count * rhs_count))
    solved = solved.reshape((batch_count, rhs_count, dimension))
    solved = jnp.swapaxes(solved, -1, -2)
    return solved.reshape(batch_shape + (dimension, rhs_count))


def _operator_action(operator, value: Array, /) -> Array:
    if operator.batch_shape:
        vector = value.reshape(
            operator.batch_shape + operator.source.shape + value.shape[-1:]
        )
        image = operator.mv(vector)
        return jnp.asarray(image).reshape(
            operator.batch_shape + (operator.target.size, value.shape[-1])
        )

    def one_column(coordinates):
        vector = operator.source.unflatten(coordinates)
        return operator.target.flatten(operator.mv(vector))

    return jax.vmap(one_column, in_axes=1, out_axes=1)(value)


def _least_squares_stationarity(
    problem: LeastSquaresProblem,
    value: Array,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
) -> Array:
    operator = problem.operator
    residual = _operator_action(operator, value) - rhs
    if problem.weights is not None:
        weights = jnp.asarray(problem.weights, dtype=residual.real.dtype)
        target_size = operator.target.size
        if weights.size == target_size:
            weights = jnp.broadcast_to(
                weights.reshape((target_size,)),
                operator.batch_shape + (target_size,),
            )
        elif weights.size == prod(operator.batch_shape or (1,)) * target_size:
            weights = weights.reshape(operator.batch_shape + (target_size,))
        else:
            raise ValueError(
                "Least-squares weights must have one entry per target coordinate."
            )
        residual = weights[..., :, None] * residual
    gradient = _operator_action(adjoint(operator), residual)
    if problem.regularizer is not None:
        regularized = _operator_action(problem.regularizer, value)
        gradient = gradient + _operator_action(
            adjoint(problem.regularizer),
            regularized,
        )
    method = plan.policy.method
    damping = method.damping if isinstance(method, (LSMR, GeneralizedLSMR)) else 0.0
    if damping:
        gradient = gradient + damping**2 * value
    return gradient


def _least_squares_root_residual(
    prepared: PreparedLinearSolve,
    problem: LeastSquaresProblem,
    value: Array,
    rhs: Array,
    /,
) -> Array:
    normal = _least_squares_stationarity(problem, value, rhs, prepared.plan)
    if not isinstance(prepared.state, DenseSVDState):
        return normal
    projected_normal = _dense_svd_active_projection(prepared.state, normal)
    projected_value = _dense_svd_active_projection(prepared.state, value)
    return projected_normal + value - projected_value


def _dense_svd_active_projection(
    state: DenseSVDState,
    coordinates: Array,
    /,
) -> Array:
    synthesis = jnp.conj(jnp.swapaxes(state.vh, -1, -2))
    basis = synthesis * state.retained[..., None, :]
    if state.source_projection is not None:
        basis = jnp.matmul(state.source_projection, basis)
    basis = jax.lax.stop_gradient(basis)
    coefficients = jnp.matmul(
        jnp.conj(jnp.swapaxes(basis, -1, -2)),
        coordinates,
    )
    return jnp.matmul(basis, coefficients)


def _callable_gmres(
    action,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
) -> Array:
    from .backends._native_krylov import _fgmres_raw

    dimension = int(rhs.shape[0])
    if plan.backend == "native-block-krylov":
        # A full, unrestarted Krylov basis is the conservative fixed-capacity
        # contract for exact implicit tangents. The primal block iteration
        # limit cannot safely bound a lower-rank independent-column tangent.
        max_steps = dimension
        restart = dimension
        stagnation_iterations = dimension + 1
        relative = 0.0
        absolute = 0.0
    else:
        max_steps = plan.policy.tolerance.max_steps or dimension
        restart = min(30, max_steps, dimension)
        stagnation_iterations = max_steps
        relative = plan.policy.tolerance.relative
        absolute = plan.policy.tolerance.absolute

    def inner(left, right):
        return jnp.vdot(left, right)

    def identity(vector, _):
        return vector

    value, _ = _fgmres_raw(
        action,
        rhs,
        jnp.zeros_like(rhs),
        inner,
        identity,
        max_steps,
        restart,
        stagnation_iterations,
        relative,
        absolute,
    )
    return value


def _stop_problem_arrays(problem: _ProblemT, /) -> _ProblemT:
    return jax.tree.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        problem,
    )


__all__ = [
    "bind_numeric",
    "prepare",
    "prepare_template",
    "refresh",
    "refresh_recycling",
    "solve",
    "solve_adjoint",
    "solve_many",
    "solve_recycled",
    "solve_transpose",
]
