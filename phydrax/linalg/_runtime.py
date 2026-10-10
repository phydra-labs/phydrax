#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._admissibility import guard_derivative_validity
from .._custom_root import custom_root
from .._iteration import (
    bind_iteration_scope,
    finalize_iteration,
    initialize_iteration,
    IterationCapabilities,
    IterationCoordinates,
    IterationEvidence,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    IterationRuntimeState,
    IterationScope,
)
from ._binding import LinearSolveTemplate
from ._dense_pseudoinverse import fixed_rank_pseudoinverse_action
from ._gcrodr import initialize_recycling, refresh_recycling, solve_recycled
from ._initial_guess import _select_proposal, AbstractInitialGuessProvider
from ._operators import AbstractLinearOperator, adjoint, transpose
from ._plans import (
    _derivative_restart,
    _preconditioned_operator,
    LinearSolvePlan,
    plan as make_plan,
)
from ._policies import (
    DenseSVD,
    FGMRES,
    GeneralizedLSMR,
    GMRES,
    LinearDerivativeSolvePolicy,
    LinearSolveCheckPolicy,
    LinearSolveControl,
    LinearSolvePolicy,
    LSMR,
)
from ._preconditioners import AbstractPreconditioner
from ._preconditioning import prepare_preconditioner, PreparedPreconditioner
from ._prepared import PreparedLinearSolve
from ._problems import (
    _problem_structure,
    AbstractLinearProblem,
    LeastSquaresProblem,
    LinearSystem,
    MinimumNormProblem,
)
from ._rank import exact_spectrum_fixed_rank, singular_value_backward_error
from ._rectangular_rank import RectangularRankCertificate
from ._results import (
    InitialGuessDiagnostics,
    LinearIterationMetrics,
    LinearPrecisionEvidence,
    LinearSolveCheckEvidence,
    LinearSolveCheckKind,
    LinearSolveDiagnostics,
    LinearSolveProvenance,
    LinearSolveResult,
    LinearSolveStatus,
    MinimumNormEvidence,
)
from ._spaces import _coordinate_dtype, AbstractVectorSpace, RHSLayout
from ._subspaces import KernelCertificate, LinearSubspace, NullspacePolicy
from ._tree import _implicit_tree_value, _prepare_tree, TreeLinearOperator
from .backends._jax_dense import (
    DenseBackendOutput,
    DenseCholeskyState,
    DenseLUState,
    DenseMixedPrecisionLUState,
    DenseQRState,
    DenseSVDState,
)
from .backends._jax_sparse import HostSparseState
from .backends._native_block_krylov import NativeBlockKrylovBackendOutput
from .backends._native_krylov import _minimum_norm_lsmr, NativeKrylovBackendOutput
from .backends._provider import AbstractLinearProvider, provider_for
from .krylov._results import KrylovBreakdownStatus


_TreeT = TypeVar("_TreeT")


class _PackedRHSLayout(NamedTuple):
    rhs_shape: tuple[int, ...]
    batch_shape: tuple[int, ...]
    broadcast_batch: bool


class _CallableLinearSolve(NamedTuple):
    value: Array
    residual_norm: Array
    iterations: Array
    breakdown: Array
    valid: Array


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
            "Plan reuse cannot change operator structure, weights or regularizers, or nullspace policies."
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
    previous_operator = prepared.problem.operator
    current_operator = problem.operator
    if type(current_operator) is not type(
        previous_operator
    ) or _operator_symbolic_contract(current_operator) != _operator_symbolic_contract(
        previous_operator
    ):
        raise ValueError("Numeric refreshes must preserve the symbolic solve plan.")
    policy = prepared.plan.policy
    preconditioning = policy.preconditioning
    refreshed_template = prepared.template
    if setup_operator is not None:
        if preconditioning is None:
            raise ValueError("setup_operator requires a preconditioning policy.")
        replacement = preconditioning.with_setup_operator(setup_operator)
        policy = eqx.tree_at(
            lambda selected: selected.preconditioning,
            policy,
            replacement,
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
        )
        if refreshed_template.template_id != prepared.template.template_id:
            raise ValueError(
                "Numeric refreshes must preserve the symbolic solve template."
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
    return _bind_for_template(
        refreshed_template,
        problem,
        previous_preconditioner=prepared.preconditioning_state,
        previous_state=prepared.state,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def release(prepared: PreparedLinearSolve, /) -> bool:
    """Explicitly release provider-owned resources for one prepared solve."""
    if not isinstance(prepared, PreparedLinearSolve):
        raise TypeError("prepared must be a PreparedLinearSolve.")
    return provider_for(prepared.plan.backend).release(prepared.state)


def _operator_symbolic_contract(operator: AbstractLinearOperator, /) -> tuple[Any, ...]:
    properties = operator.properties
    capabilities = operator.capabilities
    return (
        properties.diagonal,
        properties.triangular,
        properties.self_adjoint,
        properties.positive_definite,
        properties.positive_semidefinite,
        properties.block_diagonal,
        properties.rank,
        properties.evidence,
        capabilities.transpose,
        capabilities.adjoint,
        capabilities.materialize,
        capabilities.diagonal_assembly,
    )


def _bind_for_template(
    template: LinearSolveTemplate,
    problem: AbstractLinearProblem,
    /,
    *,
    previous_preconditioner: PreparedPreconditioner | None = None,
    previous_state: Any = None,
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
    execution_problem = _stop_arrays(problem) if stop_arrays else problem
    # Concrete plan arrays carry no tangent; evaluating stop_gradient eagerly
    # keeps host symbolic data (sparse setup patterns) readable under traces.
    with jax.ensure_compile_time_eval():
        preparation_plan = (
            jax.tree.map(
                lambda value: (
                    jax.lax.stop_gradient(value) if eqx.is_array(value) else value
                ),
                selected_plan,
            )
            if stop_arrays
            else selected_plan
        )
    preconditioning_state = prepare_preconditioner(
        preparation_plan.preconditioner_plan,
        _preconditioned_operator(
            execution_problem, preparation_plan.method, preparation_plan.policy
        ),
        materialization=preparation_plan.policy.materialization,
        previous=previous_preconditioner,
        numeric_version=numeric_version,
    )
    action = None if preconditioning_state is None else preconditioning_state.action
    provider = provider_for(selected_plan.backend)
    state = (
        provider.bind(
            template.symbolic_state,
            execution_problem,
            preparation_plan,
            preconditioner=action,
        )
        if previous_state is None
        else provider.refresh(
            template.symbolic_state,
            previous_state,
            execution_problem,
            preparation_plan,
            preconditioner=action,
        )
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


def _precision_provenance(prepared: PreparedLinearSolve, /) -> dict[str, Any]:
    requested = prepared.plan.policy.precision
    if requested is None:
        return {}
    state = prepared.state
    if isinstance(state, DenseMixedPrecisionLUState):
        operator_dtype = state.matrix.dtype
        factorization_dtype = state.factor.dtype.name
        preconditioner_dtype = None
        condition_limit = state.condition_limit
        maximum_refinement_steps = state.maximum_refinement_steps
    elif isinstance(state, DenseLUState):
        operator_dtype = state.matrix.dtype
        factorization_dtype = state.factor.dtype.name
        preconditioner_dtype = None
        condition_limit = None
        maximum_refinement_steps = 0
    else:
        operator_dtype = _coordinate_dtype(prepared.problem.operator.source)
        factorization_dtype = None
        preconditioner_dtype = (
            None
            if prepared.preconditioning_state is None
            else prepared.preconditioning_state.plan.compute_dtype
        )
        condition_limit = None
        maximum_refinement_steps = 0
    evidence = LinearPrecisionEvidence(
        operator_dtype=operator_dtype.name,
        factorization_dtype=factorization_dtype,
        preconditioner_dtype=preconditioner_dtype,
        krylov_dtype=requested.krylov_dtype,
        residual_dtype=operator_dtype.name,
        accumulation_dtype=operator_dtype.name,
        condition_limit=condition_limit,
        maximum_refinement_steps=maximum_refinement_steps,
    )
    return {
        "requested_precision": requested,
        "effective_precision": evidence,
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


def _linear_iteration_record(
    phase: IterationPhase,
    ordinal: ArrayLike,
    status: ArrayLike,
    metrics: LinearIterationMetrics,
    /,
    *,
    active: ArrayLike = True,
    committed: ArrayLike = False,
    terminal: ArrayLike = False,
) -> IterationRecord:
    accepted = jnp.asarray(ordinal, dtype=jnp.int32)
    return IterationRecord(
        IterationCoordinates(
            phase,
            ordinal,
            attempt=ordinal,
            accepted=accepted,
            active=active,
            committed=committed,
            terminal=terminal,
        ),
        status,
        metrics,
    )


def _attach_terminal_linear_iteration(
    result: LinearSolveResult,
    iteration: IterationPlan | None,
    algorithm_id: str,
    /,
) -> LinearSolveResult:
    if iteration is None:
        return result
    capabilities = IterationCapabilities.terminal_only()
    scope = bind_iteration_scope(iteration, capabilities, algorithm_id)
    diagnostics = result.diagnostics
    metrics = LinearIterationMetrics(
        residual_norm=diagnostics.residual_norm,
        relative_residual=diagnostics.relative_residual,
        normal_residual_norm=diagnostics.normal_residual_norm,
        iterations=diagnostics.iterations,
        matvec_count=diagnostics.matvec_count,
        adjoint_matvec_count=diagnostics.adjoint_matvec_count,
        condition_estimate=diagnostics.condition_estimate,
        breakdown_status=result.status,
    )
    initial = _linear_iteration_record(
        IterationPhase.START,
        0,
        result.status,
        metrics,
        active=jnp.ones_like(result.status, dtype=jnp.bool_),
    )
    terminal = _linear_iteration_record(
        IterationPhase.TERMINAL,
        diagnostics.iterations,
        result.status,
        metrics,
        active=jnp.ones_like(result.status, dtype=jnp.bool_),
        committed=result.successful,
        terminal=True,
    )
    evidence = finalize_iteration(
        iteration,
        scope,
        capabilities,
        initialize_iteration(iteration, initial),
        terminal,
    )
    return eqx.tree_at(
        lambda value: value.iteration_evidence,
        result,
        evidence,
        is_leaf=lambda value: value is None,
    )


def _initial_linear_iteration(
    prepared: PreparedLinearSolve,
    problem: AbstractLinearProblem,
    canonical_rhs: Array,
    canonical_guess: Array | None,
    layout: _PackedRHSLayout,
    iteration: IterationPlan,
    provider: AbstractLinearProvider,
) -> tuple[IterationScope, IterationCapabilities, IterationRuntimeState]:
    capabilities = provider.iteration_capabilities
    if (
        iteration.granularity == "inner-iteration"
        and prepared.plan.backend == "native-krylov"
        and canonical_rhs.shape[-1] != 1
    ):
        raise ValueError(
            "Native pseudo-block solves require one RHS for inner-iteration evidence."
        )
    scope = bind_iteration_scope(
        iteration,
        capabilities,
        f"{prepared.plan.backend}:{prepared.plan.method}",
    )
    initial = (
        jnp.zeros(
            (
                *canonical_rhs.shape[:-2],
                problem.operator.source.size,
                canonical_rhs.shape[-1],
            ),
            dtype=canonical_rhs.dtype,
        )
        if canonical_guess is None
        else canonical_guess
    )
    residual = _canonical_action(prepared, problem, initial) - canonical_rhs
    residual_norm = _coordinate_norm(problem.operator.target, residual)
    rhs_norm = _coordinate_norm(problem.operator.target, canonical_rhs)
    relative = jnp.where(rhs_norm > 0.0, residual_norm / rhs_norm, residual_norm)
    normal = jnp.full_like(residual_norm, jnp.nan)
    adjoint_count = jnp.zeros_like(residual_norm, dtype=jnp.int32)
    if isinstance(problem, LeastSquaresProblem):
        normal, _ = _normal_residual(
            prepared,
            problem,
            canonical_rhs,
            initial,
        )
        adjoint_count = jnp.ones_like(residual_norm, dtype=jnp.int32)
    residual_out = _restore_rhs_axes(residual_norm, layout)
    metrics = LinearIterationMetrics(
        residual_norm=residual_out,
        relative_residual=_restore_rhs_axes(relative, layout),
        normal_residual_norm=_restore_rhs_axes(normal, layout),
        iterations=jnp.zeros_like(residual_out, dtype=jnp.int32),
        matvec_count=jnp.ones_like(residual_out, dtype=jnp.int32),
        adjoint_matvec_count=_restore_rhs_axes(adjoint_count, layout),
        condition_estimate=jnp.full_like(residual_out, jnp.nan),
        breakdown_status=jnp.zeros_like(residual_out, dtype=jnp.int32),
    )
    initial_record = _linear_iteration_record(
        IterationPhase.START,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros_like(residual_out, dtype=jnp.int32),
        metrics,
        active=jnp.ones_like(residual_out, dtype=jnp.bool_),
    )
    return scope, capabilities, initialize_iteration(iteration, initial_record)


def solve(
    problem_or_prepared: AbstractLinearProblem | PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    policy: LinearSolvePolicy | LinearSolvePlan | None = None,
    rhs_layout: RHSLayout | None = None,
    initial_guess: PyTree[Any] | AbstractInitialGuessProvider | None = None,
    control: LinearSolveControl | None = None,
    iteration: IterationPlan | None = None,
) -> LinearSolveResult:
    """Solve one or many right-hand sides with explicit status evidence.

    `initial_guess` is either a raw guess, used as given (its derivative follows
    the differentiation policy), or an `AbstractInitialGuessProvider`. A
    provider's proposal is checked on device against the native zero guess
    before dispatch: it is used only when finite with a strictly smaller
    residual, the selected guess is stopped, and `result.initial_guess` reports
    the branch evidence per right-hand side.
    """
    if control is not None and not isinstance(control, LinearSolveControl):
        raise TypeError("control must be a LinearSolveControl or None.")
    if iteration is not None and not isinstance(iteration, IterationPlan):
        raise TypeError("iteration must be IterationPlan or None.")
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
                _require_rhs_resources(policy, canonical_rhs.shape[-1])
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
    execute = (
        _compiled_solve_prepared
        if provider_for(prepared.plan.backend).compiled_execution
        else _solve_prepared
    )
    return execute(prepared, rhs, rhs_layout, initial_guess, control, iteration)


def _solve_prepared(
    prepared: PreparedLinearSolve,
    rhs: PyTree[Any],
    rhs_layout: RHSLayout | None,
    initial_guess: PyTree[Any] | AbstractInitialGuessProvider | None,
    control: LinearSolveControl | None,
    iteration: IterationPlan | None,
    /,
) -> LinearSolveResult:
    has_runtime_overrides = control is not None and any(
        value is not None
        for value in (
            control.relative_tolerance,
            control.absolute_tolerance,
            control.maximum_steps,
        )
    )
    if has_runtime_overrides and prepared.plan.backend != "native-krylov":
        raise ValueError(
            "LinearSolveControl overrides require the native-krylov backend."
        )
    declared_layout = _execution_rhs_layout(prepared, rhs_layout)

    problem = (
        _stop_arrays(prepared.problem)
        if prepared.plan.policy.differentiation.mode in ("rhs-only", "none")
        else prepared.problem
    )
    canonical_rhs, layout = _pack_rhs(
        problem.operator.target,
        problem.operator.batch_shape,
        rhs,
        declared_layout,
    )
    _require_rhs_resources(prepared.plan, canonical_rhs.shape[-1])
    canonical_rhs, compatibility_residual = _apply_nullspace_compatibility(
        problem,
        canonical_rhs,
        prepared.plan,
    )
    canonical_guess = None
    initial_guess_evidence: InitialGuessDiagnostics | None = None
    if isinstance(initial_guess, AbstractInitialGuessProvider):
        canonical_guess, initial_guess_evidence = _provider_initial_guess(
            prepared,
            problem,
            canonical_rhs,
            layout,
            initial_guess,
        )
    elif initial_guess is not None:
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
    iteration_scope = None
    iteration_capabilities = None
    iteration_state: IterationRuntimeState | None = None
    if iteration is not None:
        (
            iteration_scope,
            iteration_capabilities,
            iteration_state,
        ) = _initial_linear_iteration(
            prepared,
            problem,
            canonical_rhs,
            canonical_guess,
            layout,
            iteration,
            provider,
        )
    backend = provider.solve(
        prepared.state,
        canonical_rhs,
        prepared.plan,
        initial_guess=canonical_guess,
        control=control,
        iteration=iteration,
        iteration_state=iteration_state,
    )
    if isinstance(backend, (NativeKrylovBackendOutput, NativeBlockKrylovBackendOutput)):
        iteration_state = backend.iteration_state

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
            _implicit_root_value(
                prepared,
                problem,
                canonical_rhs,
                backend.value,
                (
                    backend.multiplier
                    if isinstance(backend, NativeKrylovBackendOutput)
                    else None
                ),
            ),
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
    relative_tolerance = (
        prepared.plan.policy.tolerance.relative
        if control is None or control.relative_tolerance is None
        else control.relative_tolerance
    )
    absolute_tolerance = (
        prepared.plan.policy.tolerance.absolute
        if control is None or control.absolute_tolerance is None
        else control.absolute_tolerance
    )
    effective_relative = jnp.maximum(
        relative_tolerance,
        roundoff_relative,
    )
    threshold = absolute_tolerance + effective_relative * rhs_norm

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
        convergence_threshold = absolute_tolerance + effective_relative * normal_reference
        if (
            isinstance(backend, NativeKrylovBackendOutput)
            and backend.normal_residual_floor is not None
        ):
            # LSMR's attainable stationarity: below this roundoff floor of the
            # returned point no further reduction is meaningful, even when
            # ||A* b|| << ||A|| ||b|| makes the relative reference smaller.
            convergence_threshold = jnp.maximum(
                convergence_threshold,
                _rhs_broadcast(backend.normal_residual_floor, residual_norm.shape),
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
    if iteration_state is not None:
        certified = finite & (convergence_measure <= convergence_threshold)
        status = jnp.where(
            iteration_state.stop_requested & certified,
            int(LinearSolveStatus.SUCCESS),
            jnp.where(
                iteration_state.stop_requested,
                int(LinearSolveStatus.USER_STOPPED),
                status,
            ),
        )
    minimum_norm_audit: _MinimumNormAudit | None = None
    if isinstance(problem, MinimumNormProblem):
        minimum_norm_audit = _audit_minimum_norm(
            prepared,
            problem,
            (
                backend.stationarity_residual
                if isinstance(backend, NativeKrylovBackendOutput)
                else None
            ),
            canonical_rhs,
            canonical_value,
            -canonical_residual,
            residual_norm,
            threshold,
            absolute_tolerance,
            effective_relative,
            (
                None
                if not isinstance(backend, NativeKrylovBackendOutput)
                or backend.left_null_direction is None
                or backend.least_squares_stationary is None
                else (backend.left_null_direction, backend.least_squares_stationary)
            ),
        )
        status = _minimum_norm_status(status, minimum_norm_audit, finite)
        normal_residual = minimum_norm_audit.normal_residual
    derivative_regular: Array | None = None
    if minimum_norm_audit is not None:
        derivative_regular = minimum_norm_audit.derivative_regular
    elif _dense_svd_least_squares_route(prepared, problem):
        derivative_regular = _rhs_broadcast(
            _dense_svd_least_squares_regularity(prepared), status.shape
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
    if isinstance(backend, DenseBackendOutput):
        refinement_steps_out = _restore_rhs_axes(
            jnp.broadcast_to(backend.refinement_steps, status.shape),
            layout,
        )
    else:
        refinement_steps_out = _restore_rhs_axes(
            jnp.zeros(status.shape, dtype=jnp.int32),
            layout,
        )
    if prepared.plan.backend in (
        "native-krylov",
        "native-block-krylov",
        "lineax",
    ):
        matvec_count = _rhs_broadcast(backend.matvec_count, status.shape)
        adjoint_matvec_count = _rhs_broadcast(backend.adjoint_matvec_count, status.shape)
        if minimum_norm_audit is not None:
            # Residual recomputation plus the left-null and normal-reference
            # actions; a backend-supplied witness adds its own left-null action.
            matvec_count = matvec_count + 1
            adjoint_matvec_count = adjoint_matvec_count + (
                3
                if isinstance(backend, NativeKrylovBackendOutput)
                and backend.left_null_direction is not None
                else 2
            )
        matvec_count_out = _restore_rhs_axes(matvec_count, layout)
        adjoint_matvec_count_out = _restore_rhs_axes(adjoint_matvec_count, layout)
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
    if prepared.plan.policy.differentiation.mode in ("mathematical", "rhs-only"):
        canonical_value = guard_derivative_validity(
            canonical_value,
            (converged if derivative_regular is None else converged & derivative_regular)[
                ..., None, :
            ],
            failure=prepared.plan.policy.failure.mode,
            message=(
                "A failed linear solve, or a pseudoinverse solve without fixed-rank "
                "evidence, has no valid mathematical derivative; inspect "
                "status-mode diagnostics."
            ),
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
        refinement_steps=refinement_steps_out,
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
        **_precision_provenance(prepared),
    )
    iteration_evidence: IterationEvidence | None = None
    if iteration is not None:
        assert iteration_scope is not None
        assert iteration_capabilities is not None
        assert iteration_state is not None
        terminal_metrics = LinearIterationMetrics(
            residual_norm=residual_out,
            relative_residual=relative_out,
            normal_residual_norm=normal_out,
            iterations=iterations_out,
            matvec_count=matvec_count_out,
            adjoint_matvec_count=adjoint_matvec_count_out,
            condition_estimate=condition_out,
            breakdown_status=status_out,
        )
        terminal_record = _linear_iteration_record(
            IterationPhase.TERMINAL,
            iterations_out,
            status_out,
            terminal_metrics,
            active=jnp.ones_like(status_out, dtype=jnp.bool_),
            committed=converged_out,
            terminal=True,
        )
        iteration_evidence = finalize_iteration(
            iteration,
            iteration_scope,
            iteration_capabilities,
            iteration_state,
            terminal_record,
        )
    if prepared.plan.policy.failure.mode == "error":
        value = _error_on_failure(value, status_out)
    return LinearSolveResult(
        value,
        status_out,
        diagnostics,
        provenance,
        differentiation=prepared.plan.policy.differentiation,
        iteration_evidence=iteration_evidence,
        initial_guess=initial_guess_evidence,
        minimum_norm=(
            None
            if minimum_norm_audit is None
            else _minimum_norm_evidence(
                problem.operator.target, minimum_norm_audit, residual_out, layout
            )
        ),
        derivative_regular=(
            None
            if derivative_regular is None
            else _restore_rhs_axes(derivative_regular, layout)
        ),
    )


# One stable compiled entry for device-executable providers: repeated solves
# and numeric refreshes of one prepared structure trace and compile the
# provider loops, audits and implicit-derivative rules once instead of on
# every eager call. Prepared arrays and right-hand sides are dynamic arguments.
_compiled_solve_prepared = eqx.filter_jit(_solve_prepared)


class _MinimumNormAudit(NamedTuple):
    normal_residual: Array
    stationarity_residual: Array
    stationarity_available: bool
    stationarity_verified: Array
    left_null_residual: Array
    incompatibility_margin: Array
    incompatibility_radius: Array
    incompatible: Array
    witness: Array
    rank_certificate: RectangularRankCertificate | None
    derivative_regular: Array


def _audit_minimum_norm(
    prepared: PreparedLinearSolve,
    problem: MinimumNormProblem,
    stationarity_residual: Array | None,
    rhs: Array,
    value: Array,
    residual: Array,
    residual_norm: Array,
    constraint_threshold: Array,
    absolute_tolerance: float | Array,
    relative_tolerance: Array,
    left_null_candidate: tuple[Array, Array] | None,
    /,
) -> _MinimumNormAudit:
    """Audit the original constraint, stationarity, and compatibility of one solve.

    Any target vector ``z`` of unit ``N`` norm is a left-null witness: for every
    source vector ``x'``, ``||b - A x'||_N >= |<z, b>|_N - ||A* z||_M ||x'||_M``.
    By default ``z = r / ||r||_N`` and least-squares stationarity is
    ``||A* r|| <= absolute + relative ||A* b||``. A backend may supply its own
    candidate direction with its confirmed stationarity (Craig's preconditioned
    residual ``M r``, which lies in ``null(A*)`` at its weighted least-squares
    point). A finite exclusion radius is only a bounded infeasibility claim.
    Global numerical-range incompatibility additionally needs an exact null
    action or independent certified rank and retained-spectrum evidence.
    """
    operator = problem.operator
    source, target = operator.source, operator.target
    positive = residual_norm > 0.0
    residual_witness = (
        residual
        / jnp.where(positive, residual_norm, 1.0).astype(residual.dtype)[..., None, :]
    )
    normal_residual = jnp.where(
        positive,
        residual_norm
        * _coordinate_norm(source, operator.adjoint_mv_block(residual_witness)),
        0.0,
    )
    normal_reference = _coordinate_norm(source, operator.adjoint_mv_block(rhs))
    if left_null_candidate is None:
        witness = residual_witness
        witness_valid = positive
        least_squares_stationary = normal_residual <= (
            absolute_tolerance + relative_tolerance * normal_reference
        )
    else:
        direction, confirmed = left_null_candidate
        direction_norm = _coordinate_norm(target, direction)
        witness_valid = direction_norm > 0.0
        witness = (
            direction
            / jnp.where(witness_valid, direction_norm, 1.0).astype(direction.dtype)[
                ..., None, :
            ]
        )
        least_squares_stationary = _rhs_broadcast(confirmed, residual_norm.shape)
    left_null = jnp.where(
        witness_valid,
        _coordinate_norm(source, operator.adjoint_mv_block(witness)),
        jnp.nan,
    )
    margin = jnp.abs(_coordinate_inner(target, witness, rhs))
    value_norm = _coordinate_norm(source, value)
    radius = jnp.where(
        witness_valid & (left_null > 0.0),
        (margin - constraint_threshold)
        / jnp.where(witness_valid & (left_null > 0.0), left_null, 1.0),
        jnp.where(witness_valid & (margin > constraint_threshold), jnp.inf, 0.0),
    )
    certificate, regular = _minimum_norm_regularity(prepared, problem)
    rank, retained_lower = _certified_minimum_norm_spectrum(
        operator,
        certificate,
        prepared.state if isinstance(prepared.state, DenseSVDState) else None,
    )
    rank = _rhs_broadcast(rank, residual_norm.shape)
    retained_lower = _rhs_broadcast(retained_lower, residual_norm.shape)
    # A certified retained singular-value lower bound gives
    # ||P_range z|| <= ||A* z|| / sigma_min. This separates an out-of-range RHS
    # from an unresolved small retained direction; a finite ball cannot do so.
    range_leakage = left_null / jnp.where(retained_lower > 0.0, retained_lower, 1.0)
    certified_separation = (
        (rank >= 0)
        & (rank < target.size)
        & (retained_lower > 0.0)
        & (margin - range_leakage * _coordinate_norm(target, rhs) > constraint_threshold)
    )
    incompatible = (
        jnp.isfinite(margin)
        & jnp.isfinite(normal_residual)
        & (residual_norm > constraint_threshold)
        & least_squares_stationary
        & (margin > constraint_threshold)
        & (((left_null == 0.0) & (rank < target.size)) | certified_separation)
    )
    stationarity_available = stationarity_residual is not None
    if stationarity_residual is not None:
        stationarity = _rhs_broadcast(stationarity_residual, residual_norm.shape)
        stationarity_verified = jnp.isfinite(stationarity) & (
            stationarity <= absolute_tolerance + relative_tolerance * value_norm
        )
    else:
        stationarity = jnp.full_like(residual_norm, jnp.nan)
        stationarity_verified = jnp.zeros(residual_norm.shape, dtype=jnp.bool_)
    return _MinimumNormAudit(
        normal_residual=normal_residual,
        stationarity_residual=stationarity,
        stationarity_available=stationarity_available,
        stationarity_verified=stationarity_verified,
        left_null_residual=left_null,
        incompatibility_margin=margin,
        incompatibility_radius=radius,
        incompatible=incompatible,
        witness=witness,
        rank_certificate=certificate,
        derivative_regular=_rhs_broadcast(regular, residual_norm.shape),
    )


def _certified_minimum_norm_spectrum(
    operator: AbstractLinearOperator,
    certificate: RectangularRankCertificate | None,
    state: DenseSVDState | None = None,
    /,
) -> tuple[Array, Array]:
    """Independent rank and retained-spectrum lower bound, never a Krylov estimate."""
    if isinstance(state, DenseSVDState):
        values = state.reported_singular_values
        largest = values[..., 0]
        error = singular_value_backward_error(
            largest, operator.target.size, operator.source.size
        )
        smallest = jnp.min(
            jnp.where(state.retained, state.singular_values, jnp.inf), axis=-1
        )
        return state.rank, smallest - error
    if certificate is not None:
        matched = certificate.fixed_rank & certificate.matches(operator)
        return (
            jnp.where(matched, certificate.rank, -1),
            jnp.where(matched, certificate.smallest_retained_singular_value, jnp.nan),
        )
    return jnp.asarray(-1, dtype=jnp.int32), jnp.asarray(
        jnp.nan, dtype=jnp.finfo(_coordinate_dtype(operator.source)).dtype
    )


def _minimum_norm_status(
    status: Array,
    audit: _MinimumNormAudit,
    finite: Array,
    /,
) -> Array:
    preserved = (
        (status == int(LinearSolveStatus.NONFINITE_INPUT))
        | (status == int(LinearSolveStatus.NONFINITE_OUTPUT))
        | (status == int(LinearSolveStatus.USER_STOPPED))
    )
    status = jnp.where(
        audit.incompatible & finite & ~preserved,
        int(LinearSolveStatus.INCOMPATIBLE_RHS),
        status,
    )
    if not audit.stationarity_available:
        return status
    return jnp.where(
        (status == int(LinearSolveStatus.SUCCESS)) & ~audit.stationarity_verified,
        int(LinearSolveStatus.STATIONARITY_RESIDUAL_TOO_LARGE),
        status,
    )


def _minimum_norm_regularity(
    prepared: PreparedLinearSolve,
    problem: MinimumNormProblem,
    /,
) -> tuple[RectangularRankCertificate | None, Array]:
    """Rank evidence and implicit-derivative regularity of one minimum-norm solve.

    A dense SVD route decides fixed rank from the complete spectrum it executes
    (reported in the diagnostics); an iterative route needs the problem's
    certificate bound to this operator revision. Only the operator derivative
    needs fixed rank: the right-hand-side derivative of the pseudoinverse
    solution is the linear map itself. Accepting an out-of-range action still
    needs independent evidence that its retained-range residual was resolved.
    """
    state = prepared.state
    certificate = problem.rank_certificate
    match prepared.plan.policy.differentiation.mode:
        case "mathematical":
            if isinstance(state, DenseSVDState):
                rows, columns = state.design.shape[-2:]
                regular = exact_spectrum_fixed_rank(
                    state.reported_singular_values,
                    rows,
                    columns,
                    prepared.plan.policy.rank,
                )
            elif certificate is not None:
                regular = certificate.fixed_rank & certificate.matches(problem.operator)
            else:
                regular = jnp.asarray(False)
        case "rhs-only" | "algorithmic" | "none":
            regular = jnp.asarray(True)
        case mode:
            raise ValueError(f"Unknown differentiation mode {mode!r}.")
    return certificate, regular


def _minimum_norm_evidence(
    target: AbstractVectorSpace,
    audit: _MinimumNormAudit,
    constraint_residual: Array,
    layout: _PackedRHSLayout,
    /,
) -> MinimumNormEvidence:
    return MinimumNormEvidence(
        constraint_residual=constraint_residual,
        normal_residual=_restore_rhs_axes(audit.normal_residual, layout),
        stationarity_residual=_restore_rhs_axes(audit.stationarity_residual, layout),
        stationarity_verified=_restore_rhs_axes(audit.stationarity_verified, layout),
        left_null_residual=_restore_rhs_axes(audit.left_null_residual, layout),
        incompatibility_margin=_restore_rhs_axes(audit.incompatibility_margin, layout),
        incompatibility_radius=_restore_rhs_axes(audit.incompatibility_radius, layout),
        incompatible=_restore_rhs_axes(audit.incompatible, layout),
        left_null_witness=_unpack_value(target, audit.witness, layout),
        rank_certificate=audit.rank_certificate,
    )


def _provider_initial_guess(
    prepared: PreparedLinearSolve,
    problem: AbstractLinearProblem,
    canonical_rhs: Array,
    layout: _PackedRHSLayout,
    provider: AbstractInitialGuessProvider,
    /,
) -> tuple[Array, InitialGuessDiagnostics]:
    if not provider_for(prepared.plan.backend).accepts_initial_guess:
        raise ValueError("This provider does not accept an initial_guess.")
    if isinstance(problem, MinimumNormProblem):
        raise ValueError("MinimumNormProblem solves start from the zero guess.")
    if layout.batch_shape:
        raise ValueError("Initial-guess providers require an unbatched operator.")
    source = problem.operator.source
    target = problem.operator.target
    baseline_state = source.zeros()

    def propose(column: Array) -> Array:
        proposal = provider.propose(target.unflatten(column), baseline_state)
        return source.flatten(source.validate(proposal))

    proposal = jax.lax.stop_gradient(
        jax.vmap(propose, in_axes=-1, out_axes=-1)(canonical_rhs)
    )
    proposal_residual = _coordinate_norm(
        target,
        _canonical_action(prepared, problem, proposal) - canonical_rhs,
    )
    selected, evidence = _select_proposal(
        proposal,
        jnp.zeros_like(proposal),
        proposal_residual,
        _coordinate_norm(target, canonical_rhs),
        jnp.all(jnp.isfinite(proposal), axis=-2),
        provider_id=provider.provider_id,
    )
    return selected, jax.tree.map(
        lambda value: _restore_rhs_axes(value, layout), evidence
    )


def solve_many(
    prepared: PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    initial_guess: PyTree[Any] | AbstractInitialGuessProvider | None = None,
    control: LinearSolveControl | None = None,
    iteration: IterationPlan | None = None,
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
        control=control,
        iteration=iteration,
    )


def solve_transpose(
    problem_or_prepared: LinearSystem | PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    policy: LinearSolvePolicy | None = None,
    rhs_layout: RHSLayout | None = None,
    iteration: IterationPlan | None = None,
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
        if (
            iteration is None
            and not problem_or_prepared.problem.operator.batch_shape
            and provider_for(problem_or_prepared.plan.backend).supports_transformed(
                problem_or_prepared.state
            )
        ):
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
        iteration=iteration,
    )


def solve_adjoint(
    problem_or_prepared: LinearSystem | PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    policy: LinearSolvePolicy | None = None,
    rhs_layout: RHSLayout | None = None,
    iteration: IterationPlan | None = None,
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
        if (
            iteration is None
            and not problem_or_prepared.problem.operator.batch_shape
            and provider_for(problem_or_prepared.plan.backend).supports_transformed(
                problem_or_prepared.state
            )
        ):
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
        iteration=iteration,
    )


def solve_checked(
    problem_or_prepared: LinearSystem | PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    policy: LinearSolvePolicy | LinearSolvePlan | None = None,
    check_policy: LinearSolveCheckPolicy | None = None,
    rhs_layout: RHSLayout | None = None,
    initial_guess: PyTree[Any] | AbstractInitialGuessProvider | None = None,
    control: LinearSolveControl | None = None,
    iteration: IterationPlan | None = None,
) -> tuple[LinearSolveResult, LinearSolveCheckEvidence]:
    """Solve and independently assess the declared primal system."""
    problem = _checked_linear_system(problem_or_prepared)
    checks = LinearSolveCheckPolicy() if check_policy is None else check_policy
    if not isinstance(checks, LinearSolveCheckPolicy):
        raise TypeError("check_policy must be a LinearSolveCheckPolicy or None.")
    result = solve(
        problem_or_prepared,
        rhs,
        policy=policy,
        rhs_layout=rhs_layout,
        initial_guess=initial_guess,
        control=control,
        iteration=iteration,
    )
    evidence = _assess_linear_system_result(
        problem,
        rhs,
        result,
        checks,
        kind="primal",
        stability_operator=problem.operator,
        primal_valid=True,
    )
    return result, evidence


def solve_adjoint_checked(
    problem_or_prepared: LinearSystem | PreparedLinearSolve,
    rhs: PyTree[Any],
    /,
    *,
    primal_evidence: LinearSolveCheckEvidence,
    policy: LinearSolvePolicy | None = None,
    check_policy: LinearDerivativeSolvePolicy | None = None,
    rhs_layout: RHSLayout | None = None,
    iteration: IterationPlan | None = None,
) -> tuple[LinearSolveResult, LinearSolveCheckEvidence]:
    """Solve the declared adjoint and assess it under derivative requirements."""
    problem = _checked_linear_system(problem_or_prepared)
    if not isinstance(primal_evidence, LinearSolveCheckEvidence):
        raise TypeError("primal_evidence must be LinearSolveCheckEvidence.")
    if (
        primal_evidence.kind != "primal"
        or primal_evidence.operator_id != problem.operator.operator_id
    ):
        raise ValueError(
            "primal_evidence must assess this system's declared primal operator."
        )
    checks = LinearDerivativeSolvePolicy() if check_policy is None else check_policy
    if not isinstance(checks, LinearDerivativeSolvePolicy):
        raise TypeError("check_policy must be a LinearDerivativeSolvePolicy or None.")
    result = solve_adjoint(
        problem_or_prepared,
        rhs,
        policy=policy,
        rhs_layout=rhs_layout,
        iteration=iteration,
    )
    transformed = _transformed_linear_system(problem, adjoint_mode=True)
    evidence = _assess_linear_system_result(
        transformed,
        rhs,
        result,
        checks,
        kind="adjoint",
        stability_operator=problem.operator,
        primal_valid=jnp.all(primal_evidence.valid),
    )
    return result, evidence


def _checked_linear_system(
    value: LinearSystem | PreparedLinearSolve,
    /,
) -> LinearSystem:
    problem = value.problem if isinstance(value, PreparedLinearSolve) else value
    if not isinstance(problem, LinearSystem):
        raise TypeError("Checked solves require a LinearSystem.")
    return problem


def _assess_linear_system_result(
    problem: LinearSystem,
    rhs: PyTree[Any],
    result: LinearSolveResult,
    check_policy: LinearSolveCheckPolicy | LinearDerivativeSolvePolicy,
    /,
    *,
    kind: LinearSolveCheckKind,
    stability_operator: AbstractLinearOperator,
    primal_valid: Any,
) -> LinearSolveCheckEvidence:
    operator = problem.operator
    canonical_rhs, layout = _pack_rhs(
        operator.target,
        operator.batch_shape,
        rhs,
    )
    declared_layout = None if not layout.rhs_shape else RHSLayout(layout.rhs_shape)
    canonical_value, _ = _pack_rhs(
        operator.source,
        operator.batch_shape,
        result.value,
        declared_layout,
    )
    residual = operator.mv_block(canonical_value) - canonical_rhs
    residual_norm = _coordinate_norm(operator.target, residual)
    rhs_norm = _coordinate_norm(operator.target, canonical_rhs)
    threshold = (
        check_policy.absolute_tolerance + check_policy.relative_tolerance * rhs_norm
    )
    finite = (
        jnp.all(jnp.isfinite(canonical_rhs), axis=-2)
        & jnp.all(jnp.isfinite(canonical_value), axis=-2)
        & jnp.all(jnp.isfinite(residual), axis=-2)
        & jnp.isfinite(residual_norm)
        & _operator_arrays_finite(operator)
    )
    finite_out = _restore_rhs_axes(finite, layout) & jnp.asarray(
        result.diagnostics.finite, dtype=jnp.bool_
    )
    converged = jnp.asarray(result.diagnostics.converged, dtype=jnp.bool_) & jnp.asarray(
        result.successful, dtype=jnp.bool_
    )
    (
        stability_checked,
        stability_bound,
        stability_ok,
        stability_certificate_id,
        stability_evidence,
        stability_scope,
    ) = _check_stability_certificate(
        check_policy,
        stability_operator,
        operator,
    )
    (
        compatibility_residual,
        gauge_residual,
        nullspace_ok,
        nullspace_certificate_id,
    ) = _check_nullspace_evidence(
        problem,
        canonical_rhs,
        canonical_value,
        check_policy,
        layout,
        result,
    )
    return LinearSolveCheckEvidence(
        kind=kind,
        operator_id=operator.operator_id,
        status=result.status,
        true_residual_norm=_restore_rhs_axes(residual_norm, layout),
        rhs_norm=_restore_rhs_axes(rhs_norm, layout),
        residual_threshold=_restore_rhs_axes(threshold, layout),
        finite=finite_out,
        converged=converged,
        stability_lower_bound=stability_bound,
        stability_checked=stability_checked,
        stability_ok=stability_ok,
        stability_certificate_id=stability_certificate_id,
        stability_evidence=stability_evidence,
        stability_scope=stability_scope,
        compatibility_residual=compatibility_residual,
        gauge_residual=gauge_residual,
        nullspace_checked=check_policy.require_nullspace,
        nullspace_ok=nullspace_ok,
        nullspace_certificate_id=nullspace_certificate_id,
        primal_valid=primal_valid,
    )


def _operator_arrays_finite(operator: AbstractLinearOperator, /) -> Array:
    finite = jnp.asarray(True)
    for leaf in jax.tree.leaves(operator):
        if eqx.is_array(leaf):
            finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def _check_stability_certificate(
    policy: LinearSolveCheckPolicy | LinearDerivativeSolvePolicy,
    original_operator: AbstractLinearOperator,
    checked_operator: AbstractLinearOperator,
    /,
) -> tuple[bool, Array, Array, str | None, str | None, str | None]:
    certificate = policy.stability_lower_bound
    if certificate is None:
        return (
            False,
            jnp.asarray(jnp.nan),
            jnp.asarray(True),
            None,
            None,
            None,
        )
    matches = certificate.matches(original_operator) or certificate.matches(
        checked_operator
    )
    valid = (
        jnp.asarray(matches)
        & jnp.asarray(certificate.valid)
        & jnp.isfinite(certificate.lower_bound)
        & (certificate.lower_bound > 0.0)
    )
    return (
        True,
        certificate.lower_bound,
        valid,
        certificate.certificate_id,
        certificate.evidence,
        certificate.scope,
    )


def _check_nullspace_evidence(
    problem: LinearSystem,
    rhs: Array,
    value: Array,
    policy: LinearSolveCheckPolicy | LinearDerivativeSolvePolicy,
    layout: _PackedRHSLayout,
    result: LinearSolveResult,
    /,
) -> tuple[Array, Array, Array, str | None]:
    if not policy.require_nullspace:
        return (
            jnp.asarray(result.diagnostics.compatibility_residual),
            jnp.asarray(result.diagnostics.gauge_residual),
            jnp.asarray(True),
            None,
        )
    declared = problem.nullspace_policy
    if (
        declared is None
        or declared.certificate is None
        or declared.right is None
        or declared.left is None
    ):
        missing = jnp.full(jnp.asarray(result.status).shape, jnp.nan)
        return missing, missing, jnp.asarray(False), None
    certificate = declared.certificate
    projected_rhs = _project_coordinate_columns(declared.left, rhs)
    projected_value = _project_coordinate_columns(declared.right, value)
    compatibility = _coordinate_norm(declared.left.space, projected_rhs)
    gauge = _coordinate_norm(declared.right.space, projected_value)
    certificate_ok = certificate.complete and certificate.matches(problem.operator)
    nullspace_ok = (
        jnp.asarray(certificate_ok)
        & jnp.asarray(certificate.valid)
        & jnp.isfinite(compatibility)
        & jnp.isfinite(gauge)
        & (compatibility <= policy.nullspace_tolerance)
        & (gauge <= policy.nullspace_tolerance)
    )
    return (
        _restore_rhs_axes(compatibility, layout),
        _restore_rhs_axes(gauge, layout),
        _restore_rhs_axes(nullspace_ok, layout),
        certificate.certificate_id,
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
        _stop_arrays(prepared.problem)
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
    _require_rhs_resources(prepared.plan, canonical_rhs.shape[-1])
    canonical_rhs, compatibility_residual = _apply_nullspace_compatibility(
        transformed_problem,
        canonical_rhs,
        prepared.plan,
    )
    backend_rhs = canonical_rhs
    metric_transform = adjoint_mode and isinstance(
        prepared.state,
        (DenseLUState, DenseMixedPrecisionLUState, HostSparseState),
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

    image = transformed_operator.mv_block(canonical_value)
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
    transformed_refinement_steps = (
        backend.refinement_steps
        if isinstance(backend, DenseBackendOutput)
        else jnp.zeros(status.shape, dtype=jnp.int32)
    )
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
        refinement_steps=_restore_rhs_axes(
            jnp.broadcast_to(transformed_refinement_steps, status.shape),
            layout,
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
        **_precision_provenance(prepared),
    )
    return LinearSolveResult(
        value,
        status_out,
        diagnostics,
        provenance,
        differentiation=prepared.plan.policy.differentiation,
    )


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
    space: AbstractVectorSpace,
    /,
) -> LinearSubspace | None:
    if subspace is None:
        return None

    def transform(column: Array) -> Array:
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
    space: AbstractVectorSpace,
    /,
) -> LinearSubspace | None:
    if subspace is None:
        return None

    def transform(column: Array) -> Array:
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
    space: AbstractVectorSpace,
    batch_shape: tuple[int, ...],
    rhs: PyTree[Any],
    declared_layout: RHSLayout | None = None,
    /,
) -> tuple[Array, _PackedRHSLayout]:
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
            remainder = array.shape[len(prefix) :]
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


def _unpack_value(
    space: AbstractVectorSpace, value: Array, layout: _PackedRHSLayout, /
) -> PyTree[Array]:
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


def _shared_rhs_layout(space: AbstractVectorSpace, rhs: PyTree[Any], /) -> RHSLayout:
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
    return problem.operator.mv_block(value)


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


def _coordinate_norm(space: AbstractVectorSpace, coordinates: Array, /) -> Array:
    flattened, output_shape = _flatten_coordinate_columns(space, coordinates)

    def norm(column: Array) -> Array:
        vector = space.unflatten(column)
        return jnp.sqrt(jnp.maximum(jnp.real(space.inner(vector, vector)), 0.0))

    return jax.vmap(norm)(flattened).reshape(output_shape)


def _coordinate_inner(space: AbstractVectorSpace, left: Array, right: Array, /) -> Array:
    left_columns, output_shape = _flatten_coordinate_columns(space, left)
    right_columns, _ = _flatten_coordinate_columns(space, right)

    def inner(left_column: Array, right_column: Array) -> Array:
        return space.inner(space.unflatten(left_column), space.unflatten(right_column))

    return jax.vmap(inner)(left_columns, right_columns).reshape(output_shape)


def _dual_coordinate_norm(space: AbstractVectorSpace, coordinates: Array, /) -> Array:
    flattened, output_shape = _flatten_coordinate_columns(space, coordinates)

    def norm(column: Array) -> Array:
        covector = space.unflatten(column)
        primal = space.inverse_riesz(covector)
        return jnp.sqrt(jnp.maximum(jnp.real(space.inner(primal, primal)), 0.0))

    return jax.vmap(norm)(flattened).reshape(output_shape)


def _riesz_coordinates(space: AbstractVectorSpace, coordinates: Array, /) -> Array:
    return _map_coordinate_columns(space, coordinates, space.riesz)


def _inverse_riesz_coordinates(
    space: AbstractVectorSpace, coordinates: Array, /
) -> Array:
    return _map_coordinate_columns(space, coordinates, space.inverse_riesz)


def _map_coordinate_columns(
    space: AbstractVectorSpace,
    coordinates: Array,
    transform: Callable[[PyTree[Any]], PyTree[Array]],
    /,
) -> Array:
    array = jnp.asarray(coordinates)
    flattened, _ = _flatten_coordinate_columns(space, array)
    mapped = jax.vmap(lambda column: space.flatten(transform(space.unflatten(column))))(
        flattened
    )
    moved_shape = array.shape[:-2] + (array.shape[-1], space.size)
    return jnp.moveaxis(mapped.reshape(moved_shape), -1, -2)


def _project_coordinate_columns(subspace: LinearSubspace, coordinates: Array, /) -> Array:
    array = jnp.asarray(coordinates)
    flattened, _ = _flatten_coordinate_columns(subspace.space, array)
    projected = jax.vmap(subspace.project_coordinates)(flattened)
    moved_shape = array.shape[:-2] + (array.shape[-1], subspace.space.size)
    return jnp.moveaxis(projected.reshape(moved_shape), -1, -2)


def _flatten_coordinate_columns(
    space: AbstractVectorSpace,
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
    multiplier: Array | None,
    /,
) -> Array:
    route = prepared.plan.policy.derivative_solve.route
    match route:
        case "primal-factors":
            if (
                not isinstance(problem, LinearSystem)
                or problem.nullspace_policy is not None
                or not isinstance(
                    prepared.state,
                    (DenseLUState, DenseCholeskyState, DenseMixedPrecisionLUState),
                )
            ):
                raise TypeError(
                    "route='primal-factors' requires prepared square direct factors "
                    "of a LinearSystem without a nullspace policy."
                )
            return _implicit_factored_value(prepared, problem, rhs, initial)
        case "krylov":
            pass
        case _:
            raise ValueError(f"Unsupported derivative solve route {route!r}.")
    if _dense_svd_least_squares_route(prepared, problem):
        return _dense_svd_least_squares_value(prepared, rhs, initial)
    if isinstance(problem, MinimumNormProblem):
        return _implicit_minimum_norm_value(prepared, problem, rhs, initial, multiplier)
    if isinstance(problem, LinearSystem) and isinstance(
        problem.operator, TreeLinearOperator
    ):
        factor = (
            prepared.state.prepared
            if prepared.plan.backend == "jax-structured"
            else _prepare_tree(problem.operator)
        )
        return _implicit_tree_value(
            problem.operator,
            rhs,
            initial,
            factor,
            failure_mode=prepared.plan.policy.failure.mode,
        )
    initial = jax.lax.stop_gradient(initial)
    if isinstance(problem, LinearSystem):

        def residual(value: Array) -> Array:
            return _operator_action(problem.operator, value) - rhs

    else:

        def residual(value: Array) -> Array:
            return _least_squares_root_residual(
                prepared,
                problem,
                value,
                rhs,
            )

    return _implicit_custom_root(
        residual,
        initial,
        prepared.plan,
        _derivative_preconditioner(prepared, problem),
    )


def _derivative_preconditioner(
    prepared: PreparedLinearSolve,
    problem: LinearSystem | LeastSquaresProblem,
    /,
) -> AbstractPreconditioner | None:
    """The prepared primal accelerator reused by implicit derivative solves.

    The tangent solve ``A dx = r`` shares the primal operator, so the prepared
    preconditioner of ``A`` accelerates it exactly as it does the primal. The
    cotangent solve ``Aᵀ y = g`` reuses the same action: preconditioners expose
    no transpose, and ``P ≈ A⁻¹`` is also ``P ≈ A⁻ᵀ`` for the self-adjoint
    systems that dominate implicit differentiation. Both solves apply it as a
    flexible right preconditioner and accept only the true residual of the
    unpreconditioned system, so the accelerator never changes the accepted
    derivative. Least-squares stationarity differentiates ``AᵀA``, which the
    prepared action does not approximate. Its arrays are stopped: an
    accelerator carries no derivative.
    """
    state = prepared.preconditioning_state
    if (
        state is None
        or not isinstance(problem, LinearSystem)
        or problem.operator.batch_shape
    ):
        return None
    return _stop_arrays(state.action)


def _implicit_minimum_norm_value(
    prepared: PreparedLinearSolve,
    problem: MinimumNormProblem,
    rhs: Array,
    initial: Array,
    multiplier: Array | None,
    /,
) -> Array:
    """Implicit derivative of ``x = A^+ b`` through the minimum-norm KKT residual.

    The residual ``F(x, y) = (x - A* y, A x - b)`` has a singular multiplier block
    when rows are redundant, so tangents use the generalized inverse
    ``G(p, q) = (p + t, (A*)^+ t)`` with ``t = A^+ (q - A p)`` and cotangents its
    exact coordinate transpose, both through zero-start LSMR on the fixed
    operator in the declared pairings; no KKT matrix or nullspace is formed. On the
    certified fixed-rank manifold this is the derivative of the pseudoinverse
    solution; the caller guards the operator derivative with that rank evidence.
    """
    operator = problem.operator
    fixed = _stop_operator_arrays(operator)
    source, target = operator.source, operator.target
    source_size = source.size
    plan = prepared.plan
    spectrum = jax.lax.stop_gradient(
        _certified_minimum_norm_spectrum(
            fixed,
            problem.rank_certificate,
            prepared.state if isinstance(prepared.state, DenseSVDState) else None,
        )
    )
    stopped_initial = jax.lax.stop_gradient(initial)
    dual = (
        _pseudoinverse_columns(fixed, stopped_initial, plan, spectrum, adjoint=True)
        if multiplier is None
        else jax.lax.stop_gradient(multiplier)
    )
    augmented_initial = jnp.concatenate((stopped_initial, dual), axis=-2)

    def residual(augmented: Array) -> Array:
        value = augmented[..., :source_size, :]
        dual_value = augmented[..., source_size:, :]
        stationarity = value - operator.adjoint_mv_block(dual_value)
        constraint = _operator_action(operator, value) - rhs
        return jnp.concatenate((stationarity, constraint), axis=-2)

    def solve_tangent(_: Callable[[Array], Array], right: Array) -> Array:
        primal, constraint = right[..., :source_size, :], right[..., source_size:, :]
        range_component = _pseudoinverse_columns(
            fixed, constraint - fixed.mv_block(primal), plan, spectrum, adjoint=False
        )
        dual_direction = _pseudoinverse_columns(
            fixed, range_component, plan, spectrum, adjoint=True
        )
        return jnp.concatenate((primal + range_component, dual_direction), axis=-2)

    def solve_cotangent(_: Callable[[Array], Array], right: Array) -> Array:
        # Coordinate transpose of `solve_tangent`: G^T c = conj(G^H conj(c)), with
        # G^H (c_x, c_y) = (c_x - M A* v, N v),
        # v = (A*)^+ A^+ (A M^-1 c_x + N^-1 c_y) for source/target pairings M, N,
        # using (A*)^+ = (A*)^+ A^+ A so that (A*)^+ acts only on A^+ images, as in
        # the tangent.
        conjugated = jnp.conj(right)
        primal, constraint = (
            conjugated[..., :source_size, :],
            conjugated[..., source_size:, :],
        )
        lifted = _pseudoinverse_columns(
            fixed,
            fixed.mv_block(_inverse_riesz_coordinates(source, primal))
            + _inverse_riesz_coordinates(target, constraint),
            plan,
            spectrum,
            adjoint=False,
        )
        direction = _pseudoinverse_columns(
            fixed,
            lifted,
            plan,
            spectrum,
            adjoint=True,
        )
        transposed = jnp.concatenate(
            (
                primal - _riesz_coordinates(source, fixed.adjoint_mv_block(direction)),
                _riesz_coordinates(target, direction),
            ),
            axis=-2,
        )
        return jnp.conj(transposed)

    def tangent_solve(linearized: Callable[[Array], Array], target_: Array) -> Array:
        return jax.lax.custom_linear_solve(
            linearized,
            target_,
            solve=solve_tangent,
            transpose_solve=solve_cotangent,
        )

    augmented_value = custom_root(
        residual,
        augmented_initial,
        solve=lambda _, value: value,
        tangent_solve=tangent_solve,
    )
    return augmented_value[..., :source_size, :]


def _pseudoinverse_columns(
    operator: AbstractLinearOperator,
    right_hand_side: Array,
    plan: LinearSolvePlan,
    spectrum: tuple[Array, Array],
    /,
    *,
    adjoint: bool,
) -> Array:
    """Apply ``A^+`` (or ``(A*)^+``) to canonical coordinate columns.

    Each operator-batch and right-hand-side column runs one undamped zero-start
    LSMR in the declared pairings, accepted by its true residual against the
    derivative-solve tolerances. Least-squares stationarity alone is insufficient:
    an out-of-range column also needs an exact null residual or independent rank
    evidence bounding its unresolved retained-range component.
    """
    if right_hand_side.size == 0:
        return jnp.zeros(
            right_hand_side.shape[:-2]
            + (
                operator.target.size if adjoint else operator.source.size,
                right_hand_side.shape[-1],
            ),
            dtype=right_hand_side.dtype,
        )
    domain, codomain = (
        (operator.target, operator.source)
        if adjoint
        else (operator.source, operator.target)
    )
    forward_block = operator.adjoint_mv_block if adjoint else operator.mv_block
    backward_block = operator.mv_block if adjoint else operator.adjoint_mv_block
    batch_shape = right_hand_side.shape[:-2]
    rhs_count = right_hand_side.shape[-1]
    batch_count = prod(batch_shape) if batch_shape else 1
    flattened = right_hand_side.reshape((batch_count, codomain.size, rhs_count))
    policy = plan.policy.derivative_solve
    dtype = right_hand_side.real.dtype
    relative = jnp.asarray(policy.relative_tolerance, dtype=dtype)
    absolute = jnp.asarray(policy.absolute_tolerance, dtype=dtype)
    max_steps = policy.maximum_steps or max(domain.size, codomain.size)
    ranks = jnp.broadcast_to(spectrum[0], batch_shape).reshape((batch_count,))
    retained_bounds = jnp.broadcast_to(spectrum[1], batch_shape).reshape((batch_count,))

    def domain_inner(left: Array, right: Array) -> Array:
        return domain.inner(domain.unflatten(left), domain.unflatten(right))

    def codomain_inner(left: Array, right: Array) -> Array:
        return codomain.inner(codomain.unflatten(left), codomain.unflatten(right))

    def solve_instance(instance: Array) -> Array:
        batch_index = instance // rhs_count
        rhs_index = instance % rhs_count
        rank = ranks[batch_index]
        retained_lower = retained_bounds[batch_index]
        codomain_rank_deficient = (rank >= 0) & (rank < codomain.size)
        column = flattened[batch_index, :, rhs_index]

        def embedded(vector: Array, size: int) -> Array:
            block = jnp.zeros((batch_count, size, 1), dtype=right_hand_side.dtype)
            return (
                block.at[batch_index, :, 0].set(vector).reshape(batch_shape + (size, 1))
            )

        def action(vector: Array) -> Array:
            image = forward_block(embedded(vector, domain.size))
            return image.reshape((batch_count, codomain.size))[batch_index]

        def adjoint_action(vector: Array) -> Array:
            image = backward_block(embedded(vector, codomain.size))
            return image.reshape((batch_count, domain.size))[batch_index]

        column_norm = jnp.sqrt(jnp.maximum(jnp.real(codomain_inner(column, column)), 0.0))
        constraint_threshold = absolute + relative * column_norm
        normal_limit = jnp.where(
            codomain_rank_deficient & (retained_lower > 0.0),
            jnp.where(jnp.isfinite(retained_lower), retained_lower, 1.0)
            * constraint_threshold,
            0.0,
        )
        result = _minimum_norm_lsmr(
            action,
            adjoint_action,
            column,
            domain.size,
            domain_inner,
            codomain_inner,
            max_steps=max_steps,
            relative=relative,
            absolute=absolute,
            condition_limit=float("inf"),
            normal_residual_limit=normal_limit,
        )
        # Stationarity can hide unresolved retained singular directions. Only a
        # certified range bound (or an exactly null residual) admits a nonzero
        # least-squares residual as a completed pseudoinverse action.
        range_residual_bound = result.normal_residual_norm / jnp.where(
            retained_lower > 0.0, retained_lower, 1.0
        )
        range_resolved = (
            (result.normal_residual_norm == 0.0) & ((rank < 0) | codomain_rank_deficient)
        ) | (
            codomain_rank_deficient
            & (retained_lower > 0.0)
            & (range_residual_bound <= constraint_threshold)
        )
        valid = (
            jnp.all(jnp.isfinite(result.value))
            & jnp.isfinite(result.residual_norm)
            & jnp.isfinite(result.normal_residual_norm)
            & (
                (result.residual_norm <= constraint_threshold)
                | (
                    (result.breakdown == int(KrylovBreakdownStatus.STAGNATION))
                    & range_resolved
                )
            )
        )
        return _checked_callable_value(
            _CallableLinearSolve(
                result.value,
                result.residual_norm,
                result.iterations,
                result.breakdown,
                valid,
            ),
            plan.policy.failure.mode,
            message=(
                "Implicit minimum-norm derivative solve failed its pseudoinverse "
                "residual contract."
            ),
        )

    solved = jax.vmap(solve_instance)(jnp.arange(batch_count * rhs_count))
    solved = solved.reshape((batch_count, rhs_count, domain.size))
    return jnp.swapaxes(solved, -1, -2).reshape(batch_shape + (domain.size, rhs_count))


def _stop_operator_arrays(operator: AbstractLinearOperator, /) -> AbstractLinearOperator:
    return jax.tree.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        operator,
    )


def _implicit_factored_value(
    prepared: PreparedLinearSolve,
    problem: LinearSystem,
    rhs: Array,
    initial: Array,
    /,
) -> Array:
    """Mathematical root derivative whose tangent and transpose solves reuse the
    primal square direct factors (``route='primal-factors'``).

    The linearized residual of a linear system is its operator, so the prepared
    factors solve every tangent (and their algebraic transpose every cotangent)
    without iteration. Each derivative column is accepted by its true residual
    against the declared derivative-solve tolerances.
    """
    provider = provider_for(prepared.plan.backend)
    fixed_state = jax.tree.map(jax.lax.stop_gradient, prepared.state)
    failure_mode = prepared.plan.policy.failure.mode
    derivative_policy = prepared.plan.policy.derivative_solve

    def residual(value: Array) -> Array:
        return _operator_action(problem.operator, value) - rhs

    def solve_factored(
        action: Callable[[Array], Array],
        right: Array,
        *,
        transposed: bool,
    ) -> Array:
        output = (
            provider.solve_transformed(fixed_state, right, prepared.plan, adjoint=False)
            if transposed
            else provider.solve(fixed_state, right, prepared.plan)
        )
        dtype = right.real.dtype
        right_norm = jnp.linalg.norm(right, axis=-2)
        residual_norm = jnp.linalg.norm(action(output.value) - right, axis=-2)
        threshold = (
            jnp.asarray(derivative_policy.absolute_tolerance, dtype=dtype)
            + jnp.asarray(derivative_policy.relative_tolerance, dtype=dtype) * right_norm
        )
        valid = (
            jnp.all(output.status == int(LinearSolveStatus.SUCCESS))
            & jnp.all(jnp.isfinite(output.value), axis=-2)
            & jnp.isfinite(residual_norm)
            & (residual_norm <= threshold)
        )
        if failure_mode == "error":
            return eqx.error_if(
                output.value,
                ~valid,
                "Implicit linear derivative solve failed its primal-factor "
                "residual contract.",
            )
        return jnp.where(
            valid[..., None, :], output.value, jnp.full_like(output.value, jnp.nan)
        )

    def tangent_solve(linearized: Callable[[Array], Array], target: Array) -> Array:
        return jax.lax.custom_linear_solve(
            linearized,
            target,
            solve=lambda action, right: solve_factored(action, right, transposed=False),
            transpose_solve=lambda action, right: solve_factored(
                action, right, transposed=True
            ),
        )

    return custom_root(
        residual,
        jax.lax.stop_gradient(initial),
        solve=lambda _, value: value,
        tangent_solve=tangent_solve,
    )


def _implicit_custom_root(
    residual: Callable[[Array], Array],
    initial: Array,
    plan: LinearSolvePlan,
    preconditioner: AbstractPreconditioner | None,
    /,
) -> Array:
    def solve_columns(action: Callable[[Array], Array], rhs: Array) -> Array:
        return _solve_independent_columns(action, rhs, plan, preconditioner)

    def tangent_solve(linearized: Callable[[Array], Array], target: Array) -> Array:
        return jax.lax.custom_linear_solve(
            linearized,
            target,
            solve=solve_columns,
            transpose_solve=solve_columns,
        )

    return custom_root(
        residual,
        initial,
        solve=lambda _, value: value,
        tangent_solve=tangent_solve,
    )


def _solve_independent_columns(
    action: Callable[[Array], Array],
    right_hand_side: Array,
    plan: LinearSolvePlan,
    preconditioner: AbstractPreconditioner | None,
    /,
) -> Array:
    if right_hand_side.size == 0:
        return jnp.zeros_like(right_hand_side)
    batch_shape = right_hand_side.shape[:-2]
    dimension = right_hand_side.shape[-2]
    rhs_count = right_hand_side.shape[-1]
    batch_count = prod(batch_shape) if batch_shape else 1
    flattened_rhs = right_hand_side.reshape((batch_count, dimension, rhs_count))

    def solve_instance(instance_index: Array) -> Array:
        batch_index = instance_index // rhs_count
        rhs_index = instance_index % rhs_count
        column = flattened_rhs[batch_index, :, rhs_index]

        def vector_action(vector: Array) -> Array:
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

        return _callable_gmres(vector_action, column, plan, preconditioner)

    solved = jax.vmap(solve_instance)(jnp.arange(batch_count * rhs_count))
    solved = solved.reshape((batch_count, rhs_count, dimension))
    solved = jnp.swapaxes(solved, -1, -2)
    return solved.reshape(batch_shape + (dimension, rhs_count))


def _operator_action(operator: AbstractLinearOperator, value: Array, /) -> Array:
    return operator.mv_block(value)


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
    damping = (
        method.damping if isinstance(method, (DenseSVD, LSMR, GeneralizedLSMR)) else 0.0
    )
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
    method = prepared.plan.policy.method
    if isinstance(method, DenseSVD) and method.damping > 0.0:
        return normal
    if not isinstance(prepared.state, DenseSVDState):
        return normal
    projected_normal = _dense_svd_active_projection(prepared.state, normal)
    projected_value = _dense_svd_active_projection(prepared.state, value)
    return projected_normal + value - projected_value


def _dense_svd_least_squares_route(
    prepared: PreparedLinearSolve,
    problem: AbstractLinearProblem,
    /,
) -> bool:
    """Whether an undamped dense SVD least-squares solve is ``A_w^+ b_w``.

    Damped solves are smooth Tikhonov maps differentiated through their normal
    equations; a rank-cutoff regularized design keeps its projected root.
    """
    method = prepared.plan.policy.method
    return (
        isinstance(problem, LeastSquaresProblem)
        and isinstance(prepared.state, DenseSVDState)
        and prepared.state.source_projection is None
        and not (isinstance(method, DenseSVD) and method.damping > 0.0)
    )


def _dense_svd_state(prepared: PreparedLinearSolve, /) -> DenseSVDState:
    state = prepared.state
    if not isinstance(state, DenseSVDState):
        raise TypeError("The dense SVD least-squares route requires DenseSVDState.")
    return state


def _dense_svd_least_squares_regularity(prepared: PreparedLinearSolve, /) -> Array:
    """Fixed-rank evidence of the executed weighted design for an operator derivative.

    ``A_w^+`` is differentiable exactly where its numerical rank is locally
    constant; the factorization's singular-value gap certifies that.
    """
    match prepared.plan.policy.differentiation.mode:
        case "mathematical":
            state = _dense_svd_state(prepared)
            rows, columns = state.design.shape[-2:]
            return exact_spectrum_fixed_rank(
                state.singular_values, rows, columns, prepared.plan.policy.rank
            )
        case "rhs-only" | "algorithmic" | "none":
            return jnp.asarray(True)
        case mode:
            raise ValueError(f"Unknown differentiation mode {mode!r}.")


def _dense_svd_least_squares_value(
    prepared: PreparedLinearSolve,
    rhs: Array,
    initial: Array,
    /,
) -> Array:
    """Primal ``initial`` carrying the exact fixed-rank tangent of ``A_w^+ b_w``.

    The weighted design ``D = W^(1/2) A M^(-1/2)`` (with any stacked regularizer
    rows) gives ``x = M^(-1/2) D^+ [W^(1/2) b; 0]``. Its complete fixed-rank
    tangent, including range and nullspace terms, is applied through economy
    factors by ``fixed_rank_pseudoinverse_action``; the caller refuses it unless
    the rank is certified fixed.
    """
    state = _dense_svd_state(prepared)
    operator_derivative = prepared.plan.policy.differentiation.mode == "mathematical"
    design = state.design if operator_derivative else jax.lax.stop_gradient(state.design)
    weights = state.square_root_weights
    if weights is not None and not operator_derivative:
        weights = jax.lax.stop_gradient(weights)
    left, singular_values, right_adjoint, retained = jax.lax.stop_gradient(
        (state.u, state.singular_values, state.vh, state.retained)
    )
    weighted_rhs = rhs if weights is None else weights[..., :, None] * rhs
    padding = design.shape[-2] - weighted_rhs.shape[-2]
    if padding:
        weighted_rhs = jnp.concatenate(
            (
                weighted_rhs,
                jnp.zeros(
                    weighted_rhs.shape[:-2] + (padding, weighted_rhs.shape[-1]),
                    dtype=weighted_rhs.dtype,
                ),
            ),
            axis=-2,
        )
    solution = fixed_rank_pseudoinverse_action(
        design,
        left,
        singular_values,
        right_adjoint,
        retained,
        weighted_rhs,
        state.hermitian,
    )
    if state.source_inverse_square_root is not None:
        solution = state.source_inverse_square_root[..., :, None] * solution
    # The executed value is kept bit-for-bit; only its tangent comes from A_w^+.
    return jax.lax.stop_gradient(initial) + (solution - jax.lax.stop_gradient(solution))


def _dense_svd_active_projection(
    state: DenseSVDState,
    coordinates: Array,
    /,
) -> Array:
    synthesis = jnp.conj(jnp.swapaxes(state.vh, -1, -2))
    basis = synthesis * state.retained.astype(synthesis.dtype)[..., None, :]
    if state.source_projection is not None:
        basis = jnp.matmul(state.source_projection, basis)
    basis = jax.lax.stop_gradient(basis)
    coefficients = jnp.matmul(
        jnp.conj(jnp.swapaxes(basis, -1, -2)),
        coordinates,
    )
    return jnp.matmul(basis, coefficients)


def _checked_callable_value(
    result: _CallableLinearSolve,
    failure_mode: str,
    /,
    *,
    message: str,
) -> Array:
    if failure_mode == "error":
        return eqx.error_if(result.value, ~result.valid, message)
    return jnp.where(
        result.valid,
        result.value,
        jnp.full_like(result.value, jnp.nan),
    )


def _callable_gmres(
    action: Callable[[Array], Array],
    rhs: Array,
    plan: LinearSolvePlan,
    preconditioner: AbstractPreconditioner | None,
    /,
) -> Array:
    dimension = rhs.shape[0]
    policy = plan.policy.derivative_solve
    max_steps = policy.maximum_steps or dimension
    restart = _derivative_restart(plan.policy.method, max_steps, dimension)
    result = _run_callable_gmres(
        action,
        rhs,
        max_steps=max_steps,
        restart=restart,
        stagnation_iterations=max_steps + 1,
        relative=policy.relative_tolerance,
        absolute=policy.absolute_tolerance,
        preconditioner=preconditioner,
    )
    return _checked_callable_value(
        result,
        plan.policy.failure.mode,
        message=(
            "Implicit linear derivative solve failed its independent residual contract."
        ),
    )


def _callable_gmres_for_policy(
    action: Callable[[Array], Array],
    rhs: Array,
    policy: LinearSolvePolicy,
    /,
) -> Array:
    method = policy.method
    if not isinstance(method, (GMRES, FGMRES)):
        raise TypeError("Callable GMRES requires an explicit GMRES or FGMRES policy.")
    dimension = rhs.shape[0]
    max_steps = policy.tolerance.max_steps or dimension
    result = _run_callable_gmres(
        action,
        rhs,
        max_steps=max_steps,
        restart=min(method.restart, max_steps, dimension),
        stagnation_iterations=method.stagnation_iterations,
        relative=policy.tolerance.relative,
        absolute=policy.tolerance.absolute,
    )
    return _checked_callable_value(
        result,
        policy.failure.mode,
        message="Callable linear derivative solve failed its residual contract.",
    )


def _run_callable_gmres(
    action: Callable[[Array], Array],
    rhs: Array,
    /,
    *,
    max_steps: int,
    restart: int,
    stagnation_iterations: int,
    relative: float,
    absolute: float,
    preconditioner: AbstractPreconditioner | None = None,
) -> _CallableLinearSolve:
    from .backends._native_krylov import _fgmres_raw, _preconditioner_action

    def inner(left: Array, right: Array) -> Array:
        return jnp.vdot(left, right)

    def identity(vector: Array, _: Array) -> Array:
        return vector

    precondition = (
        identity
        if preconditioner is None
        else _preconditioner_action(preconditioner, preconditioner.space)
    )

    value, auxiliary, _ = _fgmres_raw(
        action,
        rhs,
        jnp.zeros_like(rhs),
        inner,
        precondition,
        max_steps,
        restart,
        stagnation_iterations,
        jnp.asarray(relative, dtype=rhs.real.dtype),
        jnp.asarray(absolute, dtype=rhs.real.dtype),
        identity_preconditioner=preconditioner is None,
    )
    iterations = auxiliary[0]
    residual_norm = auxiliary[1]
    breakdown = auxiliary[4]
    rhs_norm = jnp.linalg.norm(rhs)
    threshold = (
        jnp.asarray(absolute, dtype=rhs.real.dtype)
        + jnp.asarray(
            relative,
            dtype=rhs.real.dtype,
        )
        * rhs_norm
    )
    finite = (
        jnp.all(jnp.isfinite(rhs))
        & jnp.all(jnp.isfinite(value))
        & jnp.isfinite(residual_norm)
    )
    valid = finite & (residual_norm <= threshold)
    return _CallableLinearSolve(
        value,
        residual_norm,
        iterations,
        breakdown,
        valid,
    )


def _stop_arrays(tree: _TreeT, /) -> _TreeT:
    return jax.tree.map(
        lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
        tree,
    )


__all__ = [
    "bind_numeric",
    "initialize_recycling",
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
