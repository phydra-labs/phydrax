#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast, Literal, NamedTuple, TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax.typing import ArrayLike

from ..._iteration import (
    IterationCoordinates,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    IterationRuntimeState,
    update_iteration,
)
from ..._strict import StrictModule
from .._certificates import KernelCertificate
from .._operators import AbstractLinearOperator
from .._plans import _certified_rank, LinearSolvePlan
from .._policies import (
    Craig,
    FGMRES,
    GeneralizedLSMR,
    GMRES,
    LinearSolveControl,
    LSMR,
    MINRES,
    PCG,
    PrecisionDType,
    ProjectedPCG,
)
from .._preconditioners import AbstractPreconditioner
from .._problems import AbstractLinearProblem, LeastSquaresProblem, MinimumNormProblem
from .._results import LinearIterationMetrics, LinearSolveStatus
from .._spaces import AbstractVectorSpace
from .._structured_operators import RowGramLinearOperator
from .._subspaces import LinearSubspace, NullspacePolicy
from ..krylov._decompositions import _gated_input, _norm_from_squared
from ..krylov._results import KrylovBreakdownStatus


# Loop structure of one native Krylov execution. `"early-exit"` stops at the
# first converged or broken-down step through data-dependent loops; it is the
# only route of every non-algorithmic solve. `"fixed-trip"` executes the same
# gated steps inside static-length scans so the executed algorithm is
# reverse-differentiable: inactive steps are exact identities, masked
# orthogonalization/rotation entries contribute exact zeros, and active work
# keeps the early-exit floating-point order.
KrylovLoopDriver: TypeAlias = Literal["early-exit", "fixed-trip"]

# Coordinate-vector callbacks shared by the scalar Krylov kernels.
_Action: TypeAlias = Callable[[Array], Array]
_Inner: TypeAlias = Callable[[Array, Array], Array]
_Precondition: TypeAlias = Callable[[Array, Array], Array]

# Least-squares targets pair operator-range and regularizer-range coordinates.
_TargetPair: TypeAlias = tuple[Array, Array]
_TargetAction: TypeAlias = Callable[[Array], _TargetPair]
_TargetAdjoint: TypeAlias = Callable[[_TargetPair], Array]
_TargetInner: TypeAlias = Callable[[_TargetPair, _TargetPair], Array]

# (iterations, residual norm, normal residual norm, condition, breakdown).
_KrylovAuxiliary: TypeAlias = tuple[Array, Array, Array, Array, Array]
_KrylovResult: TypeAlias = tuple[Array, _KrylovAuxiliary, IterationRuntimeState | None]
# `_KrylovAuxiliary` followed by the executed true-residual confirmation count.
_PCGResult: TypeAlias = tuple[
    Array,
    tuple[Array, Array, Array, Array, Array, Array],
    IterationRuntimeState | None,
]
# Batched PCG returns the number of executed batched operator actions directly.
_PCGBatchedResult: TypeAlias = tuple[
    Array,
    tuple[Array, Array, Array, Array, Array, Array],
]
# `_KrylovAuxiliary` followed by the executed restart cycle count.
_FGMRESResult: TypeAlias = tuple[
    Array,
    tuple[Array, Array, Array, Array, Array, Array],
    IterationRuntimeState | None,
]
# `_KrylovAuxiliary` followed by true-quantity confirmations and the
# least-squares stationarity roundoff floor of the returned point.
_LSMRResult: TypeAlias = tuple[
    Array,
    tuple[Array, Array, Array, Array, Array, Array, Array],
    IterationRuntimeState | None,
]
# `_KrylovAuxiliary` followed by forward and adjoint matvec counts and the
# least-squares stationarity roundoff floor (NaN for square methods).
_SolveAuxiliary: TypeAlias = tuple[Array, Array, Array, Array, Array, Array, Array, Array]
_SolveResult: TypeAlias = tuple[Array, _SolveAuxiliary, IterationRuntimeState | None]

# (x, r, z, p, rho, iterations, active, breakdown, confirmations, observed).
_PCGCarry: TypeAlias = tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    IterationRuntimeState | None,
]
# (x, r, z, p, rho, iterations, active, breakdown, batched action count).
_PCGBatchedCarry: TypeAlias = tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]
# (x, r1, r2, y, old beta, beta, dbar, epsilon, phibar, cosine, sine, w, w2,
# squared Lanczos Frobenius norm, confirmations, iterations, active, breakdown,
# observed).
_MINRESCarry: TypeAlias = tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    IterationRuntimeState | None,
]
# (x, residual, residual norm, iterations, breakdown, best norm, stagnant steps,
# active, executed cycles, observed).
_FGMRESCycleCarry: TypeAlias = tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    IterationRuntimeState | None,
]
# (basis, preconditioned basis, Hessenberg, cosines, sines, reduced RHS,
# iterations, active, breakdown, best norm, stagnant steps, observed).
_ArnoldiCarry: TypeAlias = tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    IterationRuntimeState | None,
]


def _loop_driver(plan: LinearSolvePlan, /) -> KrylovLoopDriver:
    match plan.policy.differentiation.mode:
        case "algorithmic":
            return "fixed-trip"
        case "mathematical" | "rhs-only" | "none":
            return "early-exit"
        case mode:
            raise ValueError(f"Unknown differentiation mode {mode!r}.")


def _fixed_trip(driver: KrylovLoopDriver, /) -> bool:
    match driver:
        case "fixed-trip":
            return True
        case "early-exit":
            return False
        case _:
            raise ValueError(f"Unknown native Krylov loop driver {driver!r}.")


def _pcg_checkpoint_block(max_steps: int, /) -> int:
    # Square-root block checkpointing bounds reverse-mode storage at
    # O(sqrt(max_steps)) block carries plus one recomputed block.
    return max(1, math.isqrt(max_steps))


class NativeKrylovState(StrictModule):
    problem: Any
    preconditioner: AbstractPreconditioner | None


class NativeKrylovBackendOutput(StrictModule):
    """Native Krylov value and evidence.

    Minimum-norm LSMR solves also return the zero-start multiplier ``y`` of the
    adjoint witness solve ``min ||A* y - x||_M`` (target coordinates per
    right-hand side) and the true stationarity residual ``||x - A* y||_M``; both
    are ``None`` for every other problem. Least-squares LSMR solves return the
    roundoff floor ``normal_residual_floor`` below which their stationarity
    residual is accepted (``None`` for every other problem).

    Craig solves also return ``left_null_direction``, the preconditioned
    residual ``M (b - B z)`` per right-hand side, and ``least_squares_stationary``,
    whether the preconditioned least-squares stationarity of that residual was
    confirmed; on an inconsistent system that direction lies in ``null(B*)``.
    """

    value: Array
    status: Array
    iterations: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None
    iteration_state: IterationRuntimeState | None
    multiplier: Array | None
    stationarity_residual: Array | None
    normal_residual_floor: Array | None
    left_null_direction: Array | None
    least_squares_stationary: Array | None


class _LSMRState(NamedTuple):
    iteration: Array
    alpha: Array
    u_operator: Array
    u_regularizer: Array
    v: Array
    alphabar: Array
    rho: Array
    rhobar: Array
    zeta: Array
    sbar: Array
    cbar: Array
    zetabar: Array
    hbar: Array
    h: Array
    x: Array
    betadd: Array
    thetatilde: Array
    rhodold: Array
    betad: Array
    tautildeold: Array
    accumulated_residual: Array
    norm_a_squared: Array
    maximum_rbar: Array
    minimum_rbar: Array
    normal_residual: Array
    residual: Array
    norm_a: Array
    condition: Array
    active: Array
    breakdown: Array
    confirmations: Array
    iteration_state: IterationRuntimeState | None


def _gated_loop[CarryT](
    step: Callable[[Array, CarryT], CarryT],
    state: CarryT,
    max_steps: int,
    running: Callable[[Array, CarryT], Array],
    driver: KrylovLoopDriver,
    /,
) -> CarryT:
    """Run ``fori_loop(0, max_steps, step, state)`` for steps that are identities
    once ``running(index, state)`` is false.

    ``"fixed-trip"`` keeps the static-length loop that reverse mode
    differentiates. ``"early-exit"`` stops the loop at the first inactive step:
    every later step would be an identity, so the result is identical while the
    cost follows executed work instead of ``max_steps``, also under ``vmap``
    (where a batched predicate runs until every lane is inactive).
    """
    if _fixed_trip(driver):
        return jax.lax.fori_loop(0, max_steps, step, state)

    def condition(carry: tuple[Array, CarryT]) -> Array:
        index, current = carry
        return (index < max_steps) & running(index, current)

    def body(carry: tuple[Array, CarryT]) -> tuple[Array, CarryT]:
        index, current = carry
        return index + 1, step(index, current)

    _, final = jax.lax.while_loop(
        condition, body, (jnp.asarray(0, dtype=jnp.int32), state)
    )
    return final


def _iteration_stop(state: IterationRuntimeState | None, /) -> Array:
    return jnp.asarray(False) if state is None else state.stop_requested


def _update_krylov_iteration(
    plan: IterationPlan | None,
    state: IterationRuntimeState | None,
    iteration: ArrayLike,
    residual_norm: Array,
    rhs_norm: Array,
    breakdown: Array,
    /,
    *,
    matvec_count: ArrayLike,
    adjoint_matvec_count: ArrayLike = 0,
    normal_residual_norm: ArrayLike = jnp.nan,
    condition_estimate: ArrayLike = jnp.nan,
) -> IterationRuntimeState | None:
    if plan is None or state is None:
        return state
    blocking_breakdown = (breakdown != int(KrylovBreakdownStatus.NONE)) & (
        breakdown != int(KrylovBreakdownStatus.HAPPY)
    )
    status = jnp.where(
        breakdown == int(KrylovBreakdownStatus.HAPPY),
        int(LinearSolveStatus.SUCCESS),
        jnp.where(
            blocking_breakdown,
            int(LinearSolveStatus.BREAKDOWN),
            int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
        ),
    )
    record = IterationRecord(
        IterationCoordinates(
            IterationPhase.COMMIT,
            iteration,
            attempt=iteration,
            accepted=iteration,
            active=True,
            committed=True,
        ),
        status,
        LinearIterationMetrics(
            residual_norm=residual_norm,
            relative_residual=jnp.where(
                rhs_norm > 0.0, residual_norm / rhs_norm, residual_norm
            ),
            normal_residual_norm=normal_residual_norm,
            iterations=iteration,
            matvec_count=matvec_count,
            adjoint_matvec_count=adjoint_matvec_count,
            condition_estimate=condition_estimate,
            breakdown_status=breakdown,
        ),
    )
    return update_iteration(plan, state, record)


def prepare_native_krylov(
    problem: Any,
    plan: LinearSolvePlan,
    /,
    *,
    preconditioner: AbstractPreconditioner | None = None,
) -> NativeKrylovState:
    if plan.backend != "native-krylov":
        raise ValueError("Native Krylov preparation requires a native-krylov plan.")
    if problem.operator.batch_shape:
        raise ValueError("Native Krylov preparation requires an unbatched operator.")
    if (plan.preconditioner_plan is None) != (preconditioner is None):
        raise ValueError("Prepared preconditioning must match the symbolic solve plan.")
    return NativeKrylovState(problem, preconditioner)


def _runtime_controls(
    problem: Any,
    plan: LinearSolvePlan,
    control: LinearSolveControl | None,
    dtype: Any,
    /,
) -> tuple[Array, Array, Array, int]:
    # The planned bound fixes loop/basis structure; each invocation may only
    # tighten it.
    tolerance = plan.policy.tolerance
    structural_max_steps = tolerance.max_steps or (
        max(problem.operator.source.size, problem.operator.target.size)
        if isinstance(problem, (LeastSquaresProblem, MinimumNormProblem))
        else max(1, problem.operator.source.size)
    )
    relative = jnp.asarray(
        (
            tolerance.relative
            if control is None or control.relative_tolerance is None
            else control.relative_tolerance
        ),
        dtype=dtype,
    )
    absolute = jnp.asarray(
        (
            tolerance.absolute
            if control is None or control.absolute_tolerance is None
            else control.absolute_tolerance
        ),
        dtype=dtype,
    )
    max_steps = jnp.asarray(
        (
            structural_max_steps
            if control is None or control.maximum_steps is None
            else control.maximum_steps
        ),
    )
    max_steps = eqx.error_if(
        max_steps,
        max_steps > structural_max_steps,
        "Runtime maximum_steps cannot exceed the prepared structural limit.",
    ).astype(jnp.int32)
    return relative, absolute, max_steps, structural_max_steps


def solve_native_krylov(
    state: NativeKrylovState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
    *,
    initial_guess: Array | None = None,
    control: LinearSolveControl | None = None,
    iteration: IterationPlan | None = None,
    iteration_state: IterationRuntimeState | None = None,
) -> NativeKrylovBackendOutput:
    if rhs.ndim != 2:
        raise ValueError("Native Krylov right-hand sides must have shape (m, k).")
    if control is not None and not isinstance(control, LinearSolveControl):
        raise TypeError("control must be a LinearSolveControl or None.")
    problem = state.problem
    if plan.policy.differentiation.mode == "rhs-only":
        problem = jax.tree.map(
            lambda value: jax.lax.stop_gradient(value) if eqx.is_array(value) else value,
            problem,
        )
    method = plan.policy.method
    method_name = plan.method if method.name == "auto" else method.name
    (
        relative_tolerance,
        absolute_tolerance,
        maximum_steps,
        structural_maximum_steps,
    ) = _runtime_controls(problem, plan, control, rhs.real.dtype)
    guesses = (
        jnp.zeros((problem.operator.source.size, rhs.shape[1]), dtype=rhs.dtype)
        if initial_guess is None
        else initial_guess
    )
    if guesses.shape != (problem.operator.source.size, rhs.shape[1]):
        raise ValueError("initial_guess must match canonical solution and RHS axes.")
    inner_plan = (
        iteration
        if iteration is not None and iteration.granularity == "inner-iteration"
        else None
    )
    if inner_plan is not None and rhs.shape[1] != 1:
        raise ValueError("Inner native Krylov observation requires one right-hand side.")

    def solve_column(
        target: Array, guess: Array, observed_state: IterationRuntimeState | None
    ) -> _SolveResult:
        if method_name == PCG().name:
            return _square_solve(
                problem,
                target,
                guess,
                plan,
                "pcg",
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
                iteration=inner_plan,
                iteration_state=observed_state,
            )
        if method_name == ProjectedPCG().name:
            return _square_solve(
                problem,
                target,
                guess,
                plan,
                "projected-pcg",
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
                iteration=inner_plan,
                iteration_state=observed_state,
            )
        if method_name == MINRES().name:
            return _square_solve(
                problem,
                target,
                guess,
                plan,
                "minres",
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
                iteration=inner_plan,
                iteration_state=observed_state,
            )
        if method_name == FGMRES().name:
            return _square_solve(
                problem,
                target,
                guess,
                plan,
                "fgmres",
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
                iteration=inner_plan,
                iteration_state=observed_state,
            )
        if method_name == GMRES().name:
            return _square_solve(
                problem,
                target,
                guess,
                plan,
                "gmres",
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
                iteration=inner_plan,
                iteration_state=observed_state,
            )
        if method_name in (GeneralizedLSMR().name, LSMR().name):
            return _least_squares_solve(
                problem,
                target,
                guess,
                plan,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
                iteration=inner_plan,
                iteration_state=observed_state,
            )
        if method_name == Craig().name:
            return _craig_solve(
                problem,
                target,
                plan,
                preconditioner=state.preconditioner,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=maximum_steps,
                structural_max_steps=structural_maximum_steps,
                iteration=inner_plan,
                iteration_state=observed_state,
            )
        raise ValueError(f"Unsupported native Krylov method {method_name!r}.")

    if rhs.shape[1] == 1:
        value_column, auxiliary_column, updated_iteration_state = solve_column(
            rhs[:, 0], guesses[:, 0], iteration_state
        )
        value = value_column[:, None]
        auxiliary = jax.tree.map(lambda item: item[None], auxiliary_column)
    elif method_name in (PCG().name, ProjectedPCG().name):
        value, auxiliary, _ = _square_pcg_batched(
            problem,
            rhs,
            guesses,
            "projected-pcg" if method_name == ProjectedPCG().name else "pcg",
            preconditioner=state.preconditioner,
            relative=relative_tolerance,
            absolute=absolute_tolerance,
            max_steps=maximum_steps,
            structural_max_steps=structural_maximum_steps,
            driver=_loop_driver(plan),
        )
        updated_iteration_state = iteration_state
    else:

        def solve_unobserved(
            target: Array, guess: Array
        ) -> tuple[Array, _SolveAuxiliary]:
            value_, auxiliary_, _ = solve_column(target, guess, None)
            return value_, auxiliary_

        value, auxiliary = jax.vmap(solve_unobserved, in_axes=(1, 1), out_axes=(1, 0))(
            rhs, guesses
        )
        updated_iteration_state = iteration_state
    (
        iterations,
        residual,
        normal_residual,
        condition,
        breakdown,
        matvec_count,
        adjoint_matvec_count,
        normal_residual_floor,
    ) = auxiliary
    tolerance = (relative_tolerance, absolute_tolerance)
    rhs_norms = jax.vmap(
        lambda column: _space_norm(problem.operator.target, column), in_axes=1
    )(rhs)
    converged = residual <= tolerance[1] + tolerance[0] * rhs_norms
    if isinstance(problem, LeastSquaresProblem):
        # LSMR reports HAPPY only after its true stationarity residual met
        # absolute + relative ||A* b|| or its roundoff floor at the returned point.
        converged = breakdown == int(KrylovBreakdownStatus.HAPPY)
    status = jnp.full(rhs_norms.shape, int(LinearSolveStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~converged,
        int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
        status,
    )
    if method_name in (GeneralizedLSMR().name, LSMR().name):
        selected_lsmr = (
            method if isinstance(method, (GeneralizedLSMR, LSMR)) else GeneralizedLSMR()
        )
        status = jnp.where(
            (~converged) & (condition >= selected_lsmr.condition_limit),
            int(LinearSolveStatus.CONDITION_LIMIT_REACHED),
            status,
        )
    status = jnp.where(
        breakdown == int(KrylovBreakdownStatus.NONFINITE_ACTION),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        status,
    )
    status = jnp.where(
        breakdown == int(KrylovBreakdownStatus.STAGNATION),
        int(LinearSolveStatus.STAGNATION),
        status,
    )
    status = jnp.where(
        (breakdown != int(KrylovBreakdownStatus.NONE))
        & (breakdown != int(KrylovBreakdownStatus.HAPPY))
        & (breakdown != int(KrylovBreakdownStatus.NONFINITE_ACTION))
        & (breakdown != int(KrylovBreakdownStatus.STAGNATION)),
        int(LinearSolveStatus.BREAKDOWN),
        status,
    )
    craig_multiplier: Array | None = None
    left_null_direction: Array | None = None
    least_squares_stationary: Array | None = None
    if method_name == Craig().name:
        # Craig iterates on the multiplier y of B B* y = b; z = B* y lies in
        # range(B*) exactly, so its minimum-norm stationarity residual is zero.
        craig_multiplier = value
        value = problem.operator.adjoint_mv_block(value)
        constraint_residual = jax.lax.stop_gradient(
            rhs - problem.operator.mv_block(value)
        )
        precondition = _preconditioner_action(
            state.preconditioner, problem.operator.target
        )
        left_null_direction = jax.vmap(
            lambda column: precondition(column, jnp.asarray(0, dtype=jnp.int32)),
            in_axes=1,
            out_axes=1,
        )(constraint_residual)
        least_squares_stationary = breakdown == int(KrylovBreakdownStatus.STAGNATION)
        matvec_count = matvec_count + 1
    if plan.policy.differentiation.mode == "none":
        value = jax.lax.stop_gradient(value)
    multiplier: Array | None = None
    stationarity: Array | None = None
    if craig_multiplier is not None:
        multiplier = jax.lax.stop_gradient(craig_multiplier)
        stationarity = jnp.zeros(rhs.shape[1], dtype=rhs.real.dtype)
    elif isinstance(problem, MinimumNormProblem):
        selected_lsmr = (
            method if isinstance(method, (GeneralizedLSMR, LSMR)) else GeneralizedLSMR()
        )

        def witness_column(column: Array) -> tuple[Array, Array, Array, Array]:
            return _stationarity_witness(
                problem.operator,
                column,
                relative=relative_tolerance,
                absolute=absolute_tolerance,
                max_steps=structural_maximum_steps,
                condition_limit=selected_lsmr.condition_limit,
            )

        # Evidence only: the witness is not differentiated.
        multiplier, stationarity, witness_iterations, witness_confirmations = jax.vmap(
            witness_column, in_axes=1, out_axes=(1, 0, 0, 0)
        )(jax.lax.stop_gradient(value))
        # The witness applies A* as its forward action and A as its adjoint; each
        # true-quantity confirmation applies both.
        matvec_count = matvec_count + witness_iterations + 2 + witness_confirmations
        adjoint_matvec_count = (
            adjoint_matvec_count + witness_iterations + 3 + witness_confirmations
        )
    if method_name == ProjectedPCG().name:
        rank = (
            problem.operator.source.size
            - problem.nullspace_policy.certificate.right.dimension
        ).astype(jnp.int32)
    elif isinstance(problem, MinimumNormProblem) and problem.rank_certificate is not None:
        certificate = problem.rank_certificate
        rank = jnp.where(certificate.matches(problem.operator), certificate.rank, -1)
    else:
        certified_rank = _certified_rank(problem.operator)
        rank = jnp.asarray(
            -1 if certified_rank is None else certified_rank,
            dtype=jnp.int32,
        )
    return NativeKrylovBackendOutput(
        value=value,
        status=status,
        iterations=iterations,
        matvec_count=matvec_count,
        adjoint_matvec_count=adjoint_matvec_count,
        rank=rank,
        condition_estimate=condition,
        singular_values=None,
        iteration_state=updated_iteration_state,
        multiplier=multiplier,
        stationarity_residual=stationarity,
        normal_residual_floor=(
            normal_residual_floor if isinstance(problem, LeastSquaresProblem) else None
        ),
        left_null_direction=left_null_direction,
        least_squares_stationary=least_squares_stationary,
    )


def _square_solve(
    problem: AbstractLinearProblem,
    rhs: Array,
    initial: Array,
    plan: LinearSolvePlan,
    method: str,
    /,
    *,
    preconditioner: AbstractPreconditioner | None,
    relative: Array,
    absolute: Array,
    max_steps: Array,
    structural_max_steps: int,
    iteration: IterationPlan | None = None,
    iteration_state: IterationRuntimeState | None = None,
) -> _SolveResult:
    operator = problem.operator
    action = lambda vector: _action_coordinates(operator, vector)
    inner = lambda left, right: _space_inner(operator.source, left, right)
    precondition = _preconditioner_action(preconditioner, operator.source)
    tolerance = (relative, absolute)
    driver = _loop_driver(plan)
    projected = method == "projected-pcg"
    if projected:
        # ProjectedPCG planning (`_projected_pcg_rejection`) certifies a right
        # nullspace and kernel certificate before a native plan exists.
        nullspace = cast(NullspacePolicy, problem.nullspace_policy)
        certificate = cast(KernelCertificate, nullspace.certificate)
        right_nullspace = cast(LinearSubspace, nullspace.right)
        complement = lambda vector: vector - right_nullspace.project_coordinates(vector)
        rhs = eqx.error_if(
            complement(rhs),
            (~certificate.valid) | (certificate.right.dimension < 1),
            "ProjectedPCG requires a valid nonempty kernel certificate.",
        )
        initial = complement(initial)
        unprojected_action = action
        action = lambda vector: complement(unprojected_action(complement(vector)))
        unprojected_precondition = precondition
        precondition = lambda vector, iteration: complement(
            unprojected_precondition(complement(vector), iteration)
        )
        method = "pcg"

    def run(selected_action: _Action, target: Array) -> _SolveResult:
        if method == "pcg":
            value, pcg_auxiliary, next_iteration_state = _pcg_raw(
                selected_action,
                target,
                initial,
                inner,
                precondition,
                structural_max_steps,
                tolerance[0],
                tolerance[1],
                step_limit=max_steps,
                driver=driver,
                iteration=iteration,
                iteration_state=iteration_state,
            )
            auxiliary = pcg_auxiliary[:5]
            matvec_count = auxiliary[0] + pcg_auxiliary[5] + 2
        elif method == "minres":
            value, minres_auxiliary, next_iteration_state = _minres_raw(
                selected_action,
                target,
                initial,
                inner,
                precondition,
                structural_max_steps,
                tolerance[0],
                tolerance[1],
                step_limit=max_steps,
                driver=driver,
                iteration=iteration,
                iteration_state=iteration_state,
            )
            auxiliary = minres_auxiliary[:5]
            matvec_count = auxiliary[0] + minres_auxiliary[5] + 2
        else:
            if method == "gmres":
                selected_method = (
                    plan.policy.method
                    if isinstance(plan.policy.method, GMRES)
                    else GMRES()
                )
                selected_preconditioner = lambda vector, _: precondition(
                    vector, jnp.asarray(0, dtype=jnp.int32)
                )
            else:
                selected_method = (
                    plan.policy.method
                    if isinstance(plan.policy.method, FGMRES)
                    else FGMRES()
                )
                selected_preconditioner = precondition
            restart = min(selected_method.restart, structural_max_steps)
            value, cycle_auxiliary, next_iteration_state = _fgmres_raw(
                selected_action,
                target,
                initial,
                inner,
                selected_preconditioner,
                structural_max_steps,
                restart,
                selected_method.stagnation_iterations,
                tolerance[0],
                tolerance[1],
                step_limit=max_steps,
                driver=driver,
                identity_preconditioner=preconditioner is None,
                basis_dtype=(
                    None
                    if plan.policy.precision is None
                    else plan.policy.precision.krylov_dtype
                ),
                iteration=iteration,
                iteration_state=iteration_state,
            )
            auxiliary = cycle_auxiliary[:5]
            executed_cycles = cycle_auxiliary[5]
            matvec_count = auxiliary[0] + executed_cycles + 1
        return (
            value,
            (
                *auxiliary,
                jnp.asarray(matvec_count, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(jnp.nan, dtype=auxiliary[1].dtype),
            ),
            next_iteration_state,
        )

    value, auxiliary, next_iteration_state = run(action, rhs)
    return (
        complement(value) if projected else value,
        auxiliary,
        next_iteration_state,
    )


def _square_pcg_batched(
    problem: AbstractLinearProblem,
    rhs: Array,
    initial: Array,
    method: Literal["pcg", "projected-pcg"],
    *,
    preconditioner: AbstractPreconditioner | None,
    relative: Array,
    absolute: Array,
    max_steps: Array,
    structural_max_steps: int,
    driver: KrylovLoopDriver,
) -> _SolveResult:
    """Solve independent PCG lanes with one explicitly batched operator action."""
    operator = problem.operator
    action = lambda vector: _action_coordinates(operator, vector)
    inner = lambda left, right: _space_inner(operator.source, left, right)
    precondition = _preconditioner_action(preconditioner, operator.source)
    projected = method == "projected-pcg"
    if projected:
        nullspace = cast(NullspacePolicy, problem.nullspace_policy)
        certificate = cast(KernelCertificate, nullspace.certificate)
        right_nullspace = cast(LinearSubspace, nullspace.right)
        complement = lambda vector: vector - right_nullspace.project_coordinates(vector)
        batched_complement = jax.vmap(complement, in_axes=1, out_axes=1)
        rhs = eqx.error_if(
            batched_complement(rhs),
            (~certificate.valid) | (certificate.right.dimension < 1),
            "ProjectedPCG requires a valid nonempty kernel certificate.",
        )
        initial = batched_complement(initial)
        unprojected_action = action
        action = lambda vector: complement(unprojected_action(complement(vector)))
        unprojected_precondition = precondition
        precondition = lambda vector, iteration: complement(
            unprojected_precondition(complement(vector), iteration)
        )

    value, pcg_auxiliary = _pcg_batched_raw(
        action,
        rhs,
        initial,
        inner,
        precondition,
        structural_max_steps,
        relative,
        absolute,
        step_limit=max_steps,
        driver=driver,
    )
    iterations, residual, normal_residual, condition, breakdown, action_count = (
        pcg_auxiliary
    )
    value = batched_complement(value) if projected else value
    return (
        value,
        (
            iterations,
            residual,
            normal_residual,
            condition,
            breakdown,
            action_count,
            jnp.zeros_like(action_count),
            jnp.full_like(residual, jnp.nan),
        ),
        None,
    )


def _pcg_batched_raw(
    action: _Action,
    rhs: Array,
    initial: Array,
    inner: _Inner,
    precondition: _Precondition,
    max_steps: int,
    relative: Array,
    absolute: Array,
    *,
    step_limit: Array,
    driver: KrylovLoopDriver,
) -> _PCGBatchedResult:
    """Run independent PCG lanes while gating confirmation as one batch."""
    batched_action = jax.vmap(action, in_axes=1, out_axes=1)
    batched_inner = jax.vmap(inner, in_axes=(1, 1), out_axes=0)
    batched_precondition = jax.vmap(precondition, in_axes=(1, None), out_axes=1)
    residual = rhs - batched_action(initial)
    transformed = batched_precondition(residual, jnp.asarray(0, dtype=jnp.int32))
    rho = jnp.real(batched_inner(residual, transformed))
    rhs_norm = _norm(rhs, batched_inner)
    threshold = absolute + relative * rhs_norm
    residual_norm = _norm(residual, batched_inner)
    lane_count = rhs.shape[1]
    state: _PCGBatchedCarry = (
        initial,
        residual,
        transformed,
        transformed,
        rho,
        jnp.zeros((lane_count,), dtype=jnp.int32),
        residual_norm > threshold,
        jnp.full(
            (lane_count,),
            int(KrylovBreakdownStatus.NONE),
            dtype=jnp.int32,
        ),
        jnp.asarray(1, dtype=jnp.int32),
    )
    epsilon = jnp.finfo(rhs.real.dtype).eps

    def step(index: Array, current: _PCGBatchedCarry) -> _PCGBatchedCarry:
        executing = current[6] & (index < step_limit)

        def execute(operand: _PCGBatchedCarry) -> _PCGBatchedCarry:
            x, r, z, p, rho_, iterations, active, breakdown, action_count = operand
            image = batched_action(_gated_input(executing[None, :], p, initial))
            denominator = jnp.real(batched_inner(p, image))
            invalid = (
                ~jnp.isfinite(denominator)
                | (
                    jnp.abs(denominator)
                    <= epsilon * _norm(p, batched_inner) * _norm(image, batched_inner)
                )
                | (rho_ <= 0.0)
            )
            safe_denominator = jnp.where(invalid, 1.0, denominator)
            # A broken-down step keeps the last valid iterate.
            alpha = jnp.where(invalid, 0.0, rho_ / safe_denominator)
            candidate_x = x + alpha.astype(p.dtype)[None, :] * p
            recursive_r = r - alpha.astype(image.dtype)[None, :] * image
            recursive_norm = _norm(recursive_r, batched_inner)
            nominated = executing & (recursive_norm <= threshold)
            any_nominated = jnp.any(nominated)

            def confirm(_: None) -> tuple[Array, Array]:
                true_residual = rhs - batched_action(
                    _gated_input(executing[None, :], candidate_x, initial)
                )
                return true_residual, _norm(true_residual, batched_inner)

            confirmed_r, confirmed_norm = jax.lax.cond(
                any_nominated,
                confirm,
                lambda _: (recursive_r, recursive_norm),
                operand=None,
            )
            candidate_r = jnp.where(nominated[None, :], confirmed_r, recursive_r)
            norm = jnp.where(nominated, confirmed_norm, recursive_norm)
            converged = executing & (norm <= threshold)
            replaced = nominated & ~converged
            candidate_z = batched_precondition(
                _gated_input(executing[None, :], candidate_r, residual),
                jnp.asarray(index + 1, dtype=jnp.int32),
            )
            next_rho = jnp.real(batched_inner(candidate_r, candidate_z))
            beta = jnp.where(
                replaced,
                0.0,
                next_rho / jnp.where(rho_ == 0.0, 1.0, rho_),
            )
            candidate_p = candidate_z + beta.astype(p.dtype)[None, :] * p
            finite = jnp.all(jnp.isfinite(candidate_x), axis=0) & jnp.isfinite(norm)
            candidate_breakdown = jnp.where(
                finite,
                jnp.where(
                    invalid & ~converged,
                    int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
                    jnp.where(
                        converged,
                        int(KrylovBreakdownStatus.HAPPY),
                        int(KrylovBreakdownStatus.NONE),
                    ),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            next_active = finite & ~invalid & ~converged
            select_vector = lambda candidate, previous: jnp.where(
                executing[None, :], candidate, previous
            )
            select_lane = lambda candidate, previous: jnp.where(
                executing, candidate, previous
            )
            return (
                select_vector(candidate_x, x),
                select_vector(candidate_r, r),
                select_vector(candidate_z, z),
                select_vector(candidate_p, p),
                select_lane(next_rho, rho_),
                select_lane(index + 1, iterations),
                select_lane(next_active, active),
                select_lane(candidate_breakdown, breakdown),
                action_count
                + jnp.asarray(1, dtype=jnp.int32)
                + any_nominated.astype(jnp.int32),
            )

        return jax.lax.cond(jnp.any(executing), execute, lambda operand: operand, current)

    if _fixed_trip(driver):
        from ..._numerics._checkpointed_scan import checkpointed_scan

        final, _ = checkpointed_scan(
            lambda current, index: (step(index, current), None),
            state,
            jnp.arange(max_steps, dtype=jnp.int32),
            length=max_steps,
            mode="block",
            block_size=_pcg_checkpoint_block(max_steps),
        )
    else:
        final = _gated_loop(
            step,
            state,
            max_steps,
            lambda index, current: jnp.any(current[6]) & (index < step_limit),
            driver,
        )
    x, _, _, _, _, iterations, _, breakdown, action_count = final
    true_residual = rhs - batched_action(x)
    residual_norm = _norm(true_residual, batched_inner)
    action_count = action_count + jnp.asarray(1, dtype=jnp.int32)
    action_counts = jnp.full((lane_count,), action_count, dtype=jnp.int32)
    auxiliary = (
        iterations,
        residual_norm,
        jnp.full_like(residual_norm, jnp.nan),
        jnp.full_like(residual_norm, jnp.nan),
        breakdown,
        action_counts,
    )
    return x, auxiliary


def _pcg_raw(
    action: _Action,
    rhs: Array,
    initial: Array,
    inner: _Inner,
    precondition: _Precondition,
    max_steps: int,
    relative: Array,
    absolute: Array,
    *,
    finite_all: Callable[[Array], Array] | None = None,
    step_limit: Array | None = None,
    driver: KrylovLoopDriver = "early-exit",
    iteration: IterationPlan | None = None,
    iteration_state: IterationRuntimeState | None = None,
) -> _PCGResult:
    """Preconditioned CG whose stop test is the certified true residual.

    The recurrence residual only nominates convergence: when its norm meets
    ``absolute + relative * ||rhs||`` the true residual ``rhs - A x`` is
    evaluated (one extra action), and the iteration stops only if that meets
    the same threshold.  Otherwise the recurrence is restarted from the true
    residual (residual replacement, van der Vorst & Ye 2000, SIAM J. Sci.
    Comput. 22:1035) and continues.  Stopping and acceptance therefore apply
    one test to one quantity; rounding drift of the recurrence can cost
    iterations but never a stop that the final certification rejects.
    """

    if step_limit is None:
        step_limit = jnp.asarray(max_steps, dtype=jnp.int32)
    residual = rhs - action(initial)
    transformed = precondition(residual, jnp.asarray(0, dtype=jnp.int32))
    direction = transformed
    rho = jnp.real(inner(residual, transformed))
    rhs_norm = _norm(rhs, inner)
    threshold = absolute + relative * rhs_norm
    residual_norm = _norm(residual, inner)
    state: _PCGCarry = (
        initial,
        residual,
        transformed,
        direction,
        rho,
        jnp.asarray(0, dtype=jnp.int32),
        (residual_norm > threshold) & ~_iteration_stop(iteration_state),
        jnp.asarray(int(KrylovBreakdownStatus.NONE), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        iteration_state,
    )
    epsilon = jnp.finfo(rhs.real.dtype).eps

    def step(index: Array, current: _PCGCarry) -> _PCGCarry:
        (
            x,
            r,
            z,
            p,
            rho_,
            iterations,
            active,
            breakdown,
            confirmations,
            observed,
        ) = current
        executing = active & (index < step_limit)

        def execute(operand: _PCGCarry) -> _PCGCarry:
            x_, r_, z_, p_, rho_i, _, _, _, confirmations_i, observed_i = operand
            image = action(_gated_input(executing, p_, initial))
            denominator = jnp.real(inner(p_, image))
            invalid = (
                ~jnp.isfinite(denominator)
                | (
                    jnp.abs(denominator)
                    <= epsilon * _norm(p_, inner) * _norm(image, inner)
                )
                | (rho_i <= 0.0)
            )
            safe_denominator = jnp.where(invalid, 1.0, denominator)
            # A broken-down step keeps the last valid iterate.
            alpha = jnp.where(invalid, 0.0, rho_i / safe_denominator)
            candidate_x = x_ + alpha.astype(p_.dtype) * p_
            recursive_r = r_ - alpha.astype(image.dtype) * image
            recursive_norm = _norm(recursive_r, inner)
            nominated = recursive_norm <= threshold
            candidate_r, norm = jax.lax.cond(
                nominated,
                lambda: _confirmed_residual(
                    action, inner, rhs, _gated_input(executing, candidate_x, initial)
                ),
                lambda: (recursive_r, recursive_norm),
            )
            converged = norm <= threshold
            replaced = nominated & ~converged
            next_confirmations = confirmations_i + nominated.astype(jnp.int32)
            candidate_z = precondition(
                _gated_input(executing, candidate_r, residual),
                jnp.asarray(index + 1, dtype=jnp.int32),
            )
            next_rho = jnp.real(inner(candidate_r, candidate_z))
            beta = jnp.where(
                replaced, 0.0, next_rho / jnp.where(rho_i == 0.0, 1.0, rho_i)
            )
            candidate_p = candidate_z + beta.astype(p_.dtype) * p_
            finite_local = jnp.all(jnp.isfinite(candidate_x)) & jnp.isfinite(norm)
            finite = finite_local if finite_all is None else finite_all(finite_local)
            breakdown_i = jnp.where(
                finite,
                jnp.where(
                    invalid & ~converged,
                    int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
                    jnp.where(
                        converged,
                        int(KrylovBreakdownStatus.HAPPY),
                        int(KrylovBreakdownStatus.NONE),
                    ),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            next_iteration = _update_krylov_iteration(
                iteration,
                observed_i,
                index + 1,
                norm,
                rhs_norm,
                breakdown_i,
                matvec_count=index + 2 + next_confirmations,
            )
            return (
                candidate_x,
                candidate_r,
                candidate_z,
                candidate_p,
                next_rho,
                jnp.asarray(index + 1, dtype=jnp.int32),
                finite & ~invalid & ~converged & ~_iteration_stop(next_iteration),
                breakdown_i,
                next_confirmations,
                next_iteration,
            )

        return jax.lax.cond(executing, execute, lambda operand: operand, current)

    if _fixed_trip(driver):
        # Lazy: phydrax._numerics imports sampling, which imports linalg.
        from ..._numerics._checkpointed_scan import checkpointed_scan

        final, _ = checkpointed_scan(
            lambda current, index: (step(index, current), None),
            state,
            jnp.arange(max_steps, dtype=jnp.int32),
            length=max_steps,
            mode="block",
            block_size=_pcg_checkpoint_block(max_steps),
        )
    else:
        final = _gated_loop(
            step,
            state,
            max_steps,
            lambda index, current: current[6] & (index < step_limit),
            driver,
        )
    x, residual, _, _, _, iterations, _, breakdown, confirmations, iteration_state = final
    residual_norm = _norm(rhs - action(x), inner)
    auxiliary = (
        iterations,
        residual_norm,
        jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        breakdown,
        confirmations,
    )
    return x, auxiliary, iteration_state


def _confirmed_residual(
    action: _Action, inner: _Inner, rhs: Array, value: Array, /
) -> tuple[Array, Array]:
    residual = rhs - action(value)
    return residual, _norm(residual, inner)


def _minres_raw(
    action: _Action,
    rhs: Array,
    initial: Array,
    inner: _Inner,
    precondition: _Precondition,
    max_steps: int,
    relative: Array,
    absolute: Array,
    *,
    step_limit: Array | None = None,
    driver: KrylovLoopDriver = "early-exit",
    iteration: IterationPlan | None = None,
    iteration_state: IterationRuntimeState | None = None,
    row_gram_stop: bool = False,
) -> _PCGResult:
    """Preconditioned MINRES whose stops are confirmed with true quantities.

    A recurrence residual estimate within ``absolute + relative ||rhs||``
    nominates convergence; the true residual ``rhs - A x`` (one action) must meet
    the same threshold, otherwise the iteration continues.

    ``row_gram_stop`` declares ``A = B B*`` (Craig's multiplier system for
    ``B z = b``) and also stops, consistent or not, at a weighted least-squares
    point of ``B z = b``: the witness ``d = M r`` satisfies
    ``||B* d|| <= tol ||M^1/2 B|| ||r||_M`` with ``tol = max(relative,
    sqrt(4 sqrt(m) eps))``, ``||B* d||^2 = <d, A d>`` and ``||M^1/2 B||^2 =
    ||A_hat||`` (``A_hat = M^1/2 A M^1/2``, estimated by the Lanczos Frobenius
    norm ``||T_k||_F``). This is LSMR's stationarity test for the preconditioned
    ``M^1/2 B``. Its floor is the square root of LSMR's roundoff floor
    ``4 sqrt(m) eps`` (``m`` rows) because Craig reaches ``B`` only through
    ``B B*``. The Paige-Saunders recurrence
    ``||A_hat r_hat_(k-1)|| = phibar_(k-1) sqrt(gbar_k^2 + dbar_(k+1)^2)`` never
    exceeds ``||M^1/2 B|| ||B* d||``, so ``sqrt(gbar_k^2 + dbar_(k+1)^2) <= tol
    ||T_k||_F`` is a necessary condition that nominates the previous iterate;
    two true actions confirm it (``STAGNATION``) and the previous iterate is
    returned. Past that point an inconsistent singular iteration only amplifies
    roundoff. A relatively vanishing Lanczos vector closes the Krylov space and
    ends the iteration (``NEAR_BREAKDOWN``) unless a confirmation succeeds.
    Auxiliary entries are ``_KrylovAuxiliary`` followed by the confirmation
    action count.
    """
    if step_limit is None:
        step_limit = jnp.asarray(max_steps, dtype=jnp.int32)
    residual = rhs - action(initial)
    y = precondition(residual, jnp.asarray(0, dtype=jnp.int32))
    beta_one_squared = jnp.real(inner(residual, y))
    beta_one = _norm_from_squared(beta_one_squared)
    rhs_norm = _norm(rhs, inner)
    threshold = absolute + relative * rhs_norm
    real_dtype = rhs.real.dtype
    epsilon = jnp.finfo(real_dtype).eps
    stationarity_tolerance = jnp.maximum(
        relative, jnp.sqrt(4.0 * jnp.sqrt(float(rhs.size)) * epsilon)
    )
    state: _MINRESCarry = (
        initial,
        residual,
        residual,
        y,
        jnp.asarray(0.0, dtype=real_dtype),
        beta_one,
        jnp.asarray(0.0, dtype=real_dtype),
        jnp.asarray(0.0, dtype=real_dtype),
        beta_one,
        jnp.asarray(-1.0, dtype=real_dtype),
        jnp.asarray(0.0, dtype=real_dtype),
        jnp.zeros_like(rhs),
        jnp.zeros_like(rhs),
        jnp.asarray(0.0, dtype=real_dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        (_norm(residual, inner) > threshold) & ~_iteration_stop(iteration_state),
        jnp.where(
            beta_one_squared >= 0.0,
            int(KrylovBreakdownStatus.NONE),
            int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
        ).astype(jnp.int32),
        iteration_state,
    )

    def gram_stationarity(
        point: Array, index: Array, executing: Array
    ) -> tuple[Array, Array]:
        """True ``||r||_M = sqrt(<r, M r>)`` and ``||B* M r|| = sqrt(<M r, A M r>)``."""
        true_residual = rhs - action(_gated_input(executing, point, initial))
        witness = precondition(_gated_input(executing, true_residual, residual), index)
        image = action(_gated_input(executing, witness, initial))
        return (
            _norm_from_squared(inner(true_residual, witness)),
            _norm_from_squared(inner(witness, image)),
        )

    def step(index: Array, current: _MINRESCarry) -> _MINRESCarry:
        (
            x,
            r1,
            r2,
            y_,
            old_beta,
            beta,
            dbar,
            epsln,
            phibar,
            cosine,
            sine,
            w,
            w2,
            tnorm2,
            confirmations,
            iterations,
            active,
            breakdown,
            observed,
        ) = current
        executing = active & (index < step_limit)

        def execute(operand: _MINRESCarry) -> _MINRESCarry:
            (
                x_,
                r1_,
                r2_,
                y_i,
                old_beta_i,
                beta_i,
                dbar_i,
                epsln_i,
                phibar_i,
                cosine_i,
                sine_i,
                w_i,
                w2_i,
                tnorm2_i,
                confirmations_i,
                _,
                _,
                _,
                observed_i,
            ) = operand
            # Lanczos normalizations divide by any positive norm; an absolute
            # floor would discard the basis of a small-norm operator or rhs.
            safe_beta = jnp.where(beta_i > 0.0, beta_i, 1.0)
            v = y_i / safe_beta.astype(y_i.dtype)
            next_y = action(_gated_input(executing, v, initial))
            next_y = jax.lax.cond(
                index > 0,
                lambda value: (
                    value
                    - (beta_i / jnp.where(old_beta_i > 0.0, old_beta_i, 1.0)).astype(
                        r1_.dtype
                    )
                    * r1_
                ),
                lambda value: value,
                next_y,
            )
            alpha = jnp.real(inner(v, next_y))
            next_y = next_y - (alpha / safe_beta).astype(r2_.dtype) * r2_
            next_r1 = r2_
            next_r2 = next_y
            preconditioned = precondition(
                _gated_input(executing, next_r2, residual),
                jnp.asarray(index + 1, dtype=jnp.int32),
            )
            beta_squared = jnp.real(inner(next_r2, preconditioned))
            next_beta = _norm_from_squared(beta_squared)
            old_epsilon = epsln_i
            delta = cosine_i * dbar_i + sine_i * alpha
            gbar = sine_i * dbar_i - cosine_i * alpha
            next_epsln = sine_i * next_beta
            next_dbar = -cosine_i * next_beta
            gamma = _norm_from_squared(gbar * gbar + next_beta * next_beta)
            safe_gamma = jnp.where(gamma > 0.0, gamma, epsilon)
            next_cosine = gbar / safe_gamma
            next_sine = next_beta / safe_gamma
            phi = next_cosine * phibar_i
            next_phibar = next_sine * phibar_i
            w1 = w2_i
            next_w2 = w_i
            next_w = (
                v
                - old_epsilon.astype(w1.dtype) * w1
                - delta.astype(next_w2.dtype) * next_w2
            ) / safe_gamma.astype(v.dtype)
            next_x = x_ + phi.astype(next_w.dtype) * next_w
            # beta_0 normalizes the RHS, not the operator's Lanczos matrix.
            operator_beta = jnp.where(index > 0, beta_i, 0.0)
            next_tnorm2 = tnorm2_i + alpha * alpha + operator_beta**2 + next_beta**2
            anorm = _norm_from_squared(next_tnorm2)
            residual_estimate = jnp.abs(next_phibar)
            nominated = residual_estimate <= threshold
            infinity = jnp.asarray(jnp.inf, dtype=real_dtype)
            true_norm = jax.lax.cond(
                nominated,
                lambda: _norm(
                    rhs - action(_gated_input(executing, next_x, initial)), inner
                ),
                lambda: infinity,
            )
            converged = nominated & (true_norm <= threshold)
            confirmation_actions = jnp.where(nominated, 1, 0)
            if row_gram_stop:
                stationary_nominated = ~converged & (
                    jnp.sqrt(gbar * gbar + next_dbar * next_dbar)
                    <= stationarity_tolerance * anorm
                )
                previous_norm_m, previous_adjoint = jax.lax.cond(
                    stationary_nominated,
                    lambda: gram_stationarity(
                        x_, jnp.asarray(index, dtype=jnp.int32), executing
                    ),
                    lambda: (infinity, infinity),
                )
                stationary = stationary_nominated & (
                    previous_adjoint
                    <= stationarity_tolerance * jnp.sqrt(anorm) * previous_norm_m
                )
                confirmation_actions = confirmation_actions + jnp.where(
                    stationary_nominated, 2, 0
                )
            else:
                stationary = jnp.asarray(False)
            next_confirmations = confirmations_i + confirmation_actions.astype(jnp.int32)
            invariant = next_beta <= epsilon * anorm
            output_x = jnp.where(stationary, x_, next_x)
            invalid = ~stationary & (
                (beta_squared < 0.0)
                | ~jnp.isfinite(residual_estimate)
                | jnp.any(~jnp.isfinite(next_x))
            )
            status = jnp.where(
                converged,
                int(KrylovBreakdownStatus.HAPPY),
                jnp.where(
                    stationary,
                    int(KrylovBreakdownStatus.STAGNATION),
                    jnp.where(
                        invalid,
                        int(KrylovBreakdownStatus.NONFINITE_ACTION),
                        jnp.where(
                            invariant,
                            int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
                            int(KrylovBreakdownStatus.NONE),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            completed = jnp.where(stationary, index, index + 1).astype(jnp.int32)
            next_iteration = _update_krylov_iteration(
                iteration,
                observed_i,
                completed,
                residual_estimate,
                rhs_norm,
                status,
                matvec_count=index + 2 + next_confirmations,
            )
            return (
                output_x,
                next_r1,
                next_r2,
                preconditioned,
                beta_i,
                next_beta,
                next_dbar,
                next_epsln,
                next_phibar,
                next_cosine,
                next_sine,
                next_w,
                next_w2,
                next_tnorm2,
                next_confirmations,
                completed,
                ~invalid
                & ~converged
                & ~stationary
                & ~invariant
                & ~_iteration_stop(next_iteration),
                status,
                next_iteration,
            )

        return jax.lax.cond(executing, execute, lambda operand: operand, current)

    result = _gated_loop(
        step,
        state,
        max_steps,
        lambda index, current: current[16] & (index < step_limit),
        driver,
    )
    x, *_, confirmations, iterations, _, breakdown, iteration_state = result
    residual_norm = _norm(rhs - action(x), inner)
    return (
        x,
        (
            iterations,
            residual_norm,
            jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
            jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
            breakdown,
            confirmations,
        ),
        iteration_state,
    )


def _fgmres_raw(
    action: _Action,
    rhs: Array,
    initial: Array,
    inner: _Inner,
    precondition: _Precondition,
    max_steps: int,
    restart: int,
    stagnation_iterations: int,
    relative: Array,
    absolute: Array,
    *,
    step_limit: Array | None = None,
    driver: KrylovLoopDriver = "early-exit",
    identity_preconditioner: bool = False,
    basis_dtype: PrecisionDType | None = None,
    iteration: IterationPlan | None = None,
    iteration_state: IterationRuntimeState | None = None,
) -> _FGMRESResult:
    if step_limit is None:
        step_limit = jnp.asarray(max_steps, dtype=jnp.int32)
    fixed_trip = _fixed_trip(driver)
    if fixed_trip:
        # Lazy: phydrax._numerics imports sampling, which imports linalg.
        from ..._numerics._checkpointed_scan import checkpointed_scan
    rhs_norm = _norm(rhs, inner)
    threshold = absolute + relative * rhs_norm
    residual = rhs - action(initial)
    residual_norm = _norm(residual, inner)
    # Gated-off Arnoldi steps precondition the first Krylov vector of the solve.
    first_basis = residual / jnp.where(residual_norm > 0.0, residual_norm, 1.0).astype(
        residual.dtype
    )
    finite = jnp.isfinite(residual_norm) & jnp.all(jnp.isfinite(initial))
    active = finite & (residual_norm > threshold) & ~_iteration_stop(iteration_state)
    initial_breakdown = jnp.where(
        finite,
        int(KrylovBreakdownStatus.NONE),
        int(KrylovBreakdownStatus.NONFINITE_ACTION),
    ).astype(jnp.int32)
    initial_state: _FGMRESCycleCarry = (
        initial,
        residual,
        residual_norm,
        jnp.asarray(0, dtype=jnp.int32),
        initial_breakdown,
        residual_norm,
        jnp.asarray(0, dtype=jnp.int32),
        active,
        jnp.asarray(0, dtype=jnp.int32),
        iteration_state,
    )
    # A projected convergence nomination can shorten a cycle without satisfying
    # the physical residual. Such a cycle consumes only its actual Arnoldi
    # steps, so the maximum number of cycles is the iteration budget itself.
    cycles = max_steps
    columns = jnp.arange(restart, dtype=jnp.int32)

    def reduced_solve(hessenberg: Array, reduced_rhs: Array, steps: Array) -> Array:
        active_columns = columns < steps
        upper = jnp.where(
            active_columns[:, None] & active_columns[None, :],
            hessenberg[:-1],
            0,
        )
        upper = upper.at[columns, columns].add((~active_columns).astype(rhs.dtype))
        target = jnp.where(active_columns, reduced_rhs[:-1], 0)
        coefficients = jsp.linalg.solve_triangular(
            upper,
            target,
            lower=False,
        )
        return jnp.where(active_columns, coefficients, 0)

    def cycle_step(_: None, state: _FGMRESCycleCarry) -> _FGMRESCycleCarry:
        (
            x,
            current_residual,
            current_norm,
            iterations,
            breakdown,
            best_norm,
            stagnant_steps,
            cycle_active,
            executed_cycles,
            observed,
        ) = state
        cycle_executing = (
            cycle_active & (iterations < step_limit) & (iterations < max_steps)
        )

        def execute_cycle(operand: _FGMRESCycleCarry) -> _FGMRESCycleCarry:
            (
                cycle_base,
                cycle_residual,
                cycle_norm,
                starting_iterations,
                _,
                previous_best,
                previous_stagnant,
                _,
                previous_cycles,
                observed_outer,
            ) = operand
            stored_basis_dtype = (
                rhs.dtype if basis_dtype is None else jnp.dtype(basis_dtype)
            )
            safe_norm = jnp.where(cycle_norm > 0.0, cycle_norm, 1.0)
            basis = jnp.zeros(
                (restart + 1, rhs.size),
                dtype=stored_basis_dtype,
            )
            basis = basis.at[0].set(
                (cycle_residual / safe_norm.astype(cycle_residual.dtype)).astype(
                    stored_basis_dtype
                )
            )
            preconditioned_basis = jnp.zeros(
                (0 if identity_preconditioner else restart, rhs.size),
                dtype=stored_basis_dtype,
            )
            hessenberg = jnp.zeros((restart + 1, restart), dtype=rhs.dtype)
            cosines = jnp.zeros((restart,), dtype=rhs.real.dtype)
            sines = jnp.zeros((restart,), dtype=rhs.dtype)
            reduced_rhs = jnp.zeros((restart + 1,), dtype=rhs.dtype)
            reduced_rhs = reduced_rhs.at[0].set(cycle_norm.astype(rhs.dtype))
            inner_state: _ArnoldiCarry = (
                basis,
                preconditioned_basis,
                hessenberg,
                cosines,
                sines,
                reduced_rhs,
                starting_iterations,
                jnp.asarray(True),
                jnp.asarray(int(KrylovBreakdownStatus.NONE), dtype=jnp.int32),
                previous_best,
                previous_stagnant,
                observed_outer,
            )

            def arnoldi_step(local_index: Array, current: _ArnoldiCarry) -> _ArnoldiCarry:
                (
                    basis_,
                    preconditioned_,
                    hessenberg_,
                    cosines_,
                    sines_,
                    reduced_rhs_,
                    iteration_,
                    inner_active,
                    inner_breakdown,
                    inner_best,
                    inner_stagnant,
                    observed_inner,
                ) = current
                can_execute = (
                    inner_active & (iteration_ < step_limit) & (iteration_ < max_steps)
                )
                step_executing = cycle_executing & can_execute

                def execute_arnoldi(inner_operand: _ArnoldiCarry) -> _ArnoldiCarry:
                    (
                        basis_i,
                        preconditioned_i,
                        hessenberg_i,
                        cosines_i,
                        sines_i,
                        reduced_rhs_i,
                        iteration_i,
                        _,
                        _,
                        best_i,
                        stagnant_i,
                        observed_i,
                    ) = inner_operand
                    vector = _gated_input(
                        step_executing,
                        basis_i[local_index].astype(rhs.dtype),
                        first_basis,
                    )
                    transformed = (
                        vector
                        if identity_preconditioner
                        else precondition(vector, iteration_i)
                    )
                    image = action(_gated_input(step_executing, transformed, initial))
                    projection = jnp.zeros((restart,), dtype=rhs.dtype)

                    # Early exit bounds the modified Gram-Schmidt passes by the
                    # dynamic basis size; fixed trip runs the static restart
                    # length and masks entries beyond `local_index` to exact
                    # zeros, so active work keeps the same order.
                    def orthogonalize(
                        index: Array, state: tuple[Array, Array]
                    ) -> tuple[Array, Array]:
                        remainder, coefficients = state
                        basis_vector = basis_i[index].astype(rhs.dtype)
                        coefficient = inner(basis_vector, remainder)
                        reduced = remainder - coefficient * basis_vector
                        if fixed_trip:
                            active = index <= local_index
                            coefficient = jnp.where(active, coefficient, 0)
                            reduced = jnp.where(active, reduced, remainder)
                        return reduced, coefficients.at[index].add(coefficient)

                    basis_bound = restart if fixed_trip else local_index + 1
                    orthogonal, projection = jax.lax.fori_loop(
                        0,
                        basis_bound,
                        orthogonalize,
                        (image, projection),
                    )
                    orthogonal, projection = jax.lax.fori_loop(
                        0,
                        basis_bound,
                        orthogonalize,
                        (orthogonal, projection),
                    )
                    next_norm = _norm(orthogonal, inner)
                    # Relative to the Hessenberg column scale ||A v||: an
                    # absolute floor would flag every small-norm operator as a
                    # near-invariant subspace. A near-invariant subspace ends the
                    # cycle as a restart; the cycle then decides from the true
                    # residual whether it was lucky, progressing, or stagnant.
                    near_breakdown = next_norm <= jnp.sqrt(
                        jnp.finfo(rhs.real.dtype).eps
                    ) * _norm(image, inner)
                    basis_i = basis_i.at[local_index + 1].set(
                        (
                            orthogonal
                            / jnp.where(near_breakdown, 1.0, next_norm).astype(
                                orthogonal.dtype
                            )
                        ).astype(stored_basis_dtype)
                    )
                    if not identity_preconditioner:
                        preconditioned_i = preconditioned_i.at[local_index].set(
                            transformed.astype(stored_basis_dtype)
                        )
                    column = hessenberg_i[:, local_index]
                    column = column.at[:-1].set(projection)
                    column = column.at[local_index + 1].set(
                        next_norm.astype(column.dtype)
                    )

                    def apply_previous_rotation(index: Array, value: Array) -> Array:
                        upper = value[index]
                        lower = value[index + 1]
                        cosine = cosines_i[index].astype(upper.dtype)
                        sine = sines_i[index]
                        rotated = value.at[index].set(cosine * upper + sine * lower)
                        rotated = rotated.at[index + 1].set(
                            -jnp.conj(sine) * upper + cosine * lower
                        )
                        if fixed_trip:
                            return jnp.where(index < local_index, rotated, value)
                        return rotated

                    column = jax.lax.fori_loop(
                        0,
                        restart if fixed_trip else local_index,
                        apply_previous_rotation,
                        column,
                    )
                    upper = column[local_index]
                    lower = column[local_index + 1]
                    upper_abs = _safe_abs(upper)
                    rotation_scale = jnp.hypot(upper_abs, _safe_abs(lower))
                    safe_scale = jnp.where(rotation_scale > 0.0, rotation_scale, 1.0)
                    safe_upper_abs = jnp.where(upper_abs > 0.0, upper_abs, 1.0)
                    phase = upper / safe_upper_abs.astype(upper.dtype)
                    phase = jnp.where(
                        upper_abs > 0.0,
                        phase,
                        jnp.ones((), dtype=rhs.dtype),
                    )
                    cosine = jnp.where(
                        rotation_scale > 0.0,
                        upper_abs / safe_scale,
                        1.0,
                    )
                    sine = jnp.where(
                        rotation_scale > 0.0,
                        phase * jnp.conj(lower) / safe_scale.astype(phase.dtype),
                        jnp.zeros((), dtype=rhs.dtype),
                    )
                    column = column.at[local_index].set(
                        cosine.astype(upper.dtype) * upper + sine * lower
                    )
                    column = column.at[local_index + 1].set(0)
                    hessenberg_i = hessenberg_i.at[:, local_index].set(column)
                    cosines_i = cosines_i.at[local_index].set(cosine)
                    sines_i = sines_i.at[local_index].set(sine)
                    reduced_upper = reduced_rhs_i[local_index]
                    reduced_lower = reduced_rhs_i[local_index + 1]
                    reduced_rhs_i = reduced_rhs_i.at[local_index].set(
                        cosine.astype(reduced_upper.dtype) * reduced_upper
                        + sine * reduced_lower
                    )
                    reduced_rhs_i = reduced_rhs_i.at[local_index + 1].set(
                        -jnp.conj(sine) * reduced_upper
                        + cosine.astype(reduced_lower.dtype) * reduced_lower
                    )
                    estimated_norm = _safe_abs(reduced_rhs_i[local_index + 1])
                    finite_step = (
                        jnp.isfinite(next_norm)
                        & jnp.isfinite(rotation_scale)
                        & jnp.isfinite(estimated_norm)
                        & jnp.all(jnp.isfinite(image))
                        & jnp.all(jnp.isfinite(transformed))
                        & jnp.all(jnp.isfinite(column))
                    )
                    converged = estimated_norm <= threshold
                    improvement = estimated_norm < best_i * (
                        1.0 - jnp.sqrt(jnp.finfo(rhs.real.dtype).eps)
                    )
                    next_best = jnp.minimum(best_i, estimated_norm)
                    next_stagnant = jnp.where(
                        improvement,
                        jnp.asarray(0, dtype=jnp.int32),
                        stagnant_i + 1,
                    )
                    stagnated = next_stagnant >= stagnation_iterations
                    step_breakdown = jnp.where(
                        finite_step,
                        jnp.where(
                            converged,
                            int(KrylovBreakdownStatus.HAPPY),
                            jnp.where(
                                stagnated,
                                int(KrylovBreakdownStatus.STAGNATION),
                                jnp.where(
                                    near_breakdown,
                                    int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
                                    int(KrylovBreakdownStatus.NONE),
                                ),
                            ),
                        ),
                        int(KrylovBreakdownStatus.NONFINITE_ACTION),
                    ).astype(jnp.int32)
                    next_iteration = _update_krylov_iteration(
                        iteration,
                        observed_i,
                        iteration_i + 1,
                        estimated_norm,
                        rhs_norm,
                        # A near-invariant step is a restart, not yet a failure;
                        # the cycle reports genuine breakdown from true residuals.
                        jnp.where(
                            step_breakdown == int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
                            int(KrylovBreakdownStatus.NONE),
                            step_breakdown,
                        ),
                        matvec_count=iteration_i + 2,
                    )
                    return (
                        basis_i,
                        preconditioned_i,
                        hessenberg_i,
                        cosines_i,
                        sines_i,
                        reduced_rhs_i,
                        iteration_i + 1,
                        finite_step
                        & ~converged
                        & ~near_breakdown
                        & ~stagnated
                        & ~_iteration_stop(next_iteration),
                        step_breakdown,
                        next_best,
                        next_stagnant,
                        next_iteration,
                    )

                return jax.lax.cond(
                    step_executing,
                    execute_arnoldi,
                    lambda inner_operand: inner_operand,
                    current,
                )

            def inner_condition(current: _ArnoldiCarry) -> Array:
                iteration = current[6]
                return (
                    cycle_executing
                    & current[7]
                    & (iteration - starting_iterations < restart)
                    & (iteration < step_limit)
                    & (iteration < max_steps)
                )

            def inner_body(current: _ArnoldiCarry) -> _ArnoldiCarry:
                return arnoldi_step(
                    current[6] - starting_iterations,
                    current,
                )

            if fixed_trip:
                # Each Arnoldi step is rematerialized in reverse mode: storage is
                # the step carries of one cycle plus one step's residuals.
                final_inner, _ = checkpointed_scan(
                    lambda current, local_index: (
                        arnoldi_step(local_index, current),
                        None,
                    ),
                    inner_state,
                    jnp.arange(restart, dtype=jnp.int32),
                    length=restart,
                    mode="step",
                )
            else:
                final_inner = jax.lax.while_loop(inner_condition, inner_body, inner_state)
            (
                basis,
                preconditioned_basis,
                hessenberg,
                _,
                _,
                reduced_rhs,
                final_iterations,
                _,
                inner_breakdown,
                inner_best,
                inner_stagnant,
                inner_iteration_state,
            ) = final_inner
            update_basis = basis[:-1] if identity_preconditioner else preconditioned_basis
            cycle_steps = final_iterations - starting_iterations
            coefficients = reduced_solve(
                hessenberg,
                reduced_rhs,
                cycle_steps,
            )
            candidate = cycle_base + jnp.sum(
                coefficients[:, None] * update_basis,
                axis=0,
            )
            candidate_residual = rhs - action(
                _gated_input(cycle_executing, candidate, initial)
            )
            candidate_norm = _norm(candidate_residual, inner)
            finite_candidate = (
                jnp.isfinite(candidate_norm)
                & jnp.all(jnp.isfinite(candidate))
                & jnp.all(jnp.isfinite(coefficients))
            )
            converged = candidate_norm <= threshold
            next_best = jnp.minimum(inner_best, candidate_norm)
            next_stagnant = inner_stagnant
            stagnated = next_stagnant >= stagnation_iterations
            # A near-invariant Krylov subspace that still reduced the true
            # residual is restarted from the new residual. Only an invariant
            # direction without true-residual progress is a genuine breakdown.
            progressing_restart = (
                inner_breakdown == int(KrylovBreakdownStatus.NEAR_BREAKDOWN)
            ) & (
                candidate_norm
                < cycle_norm * (1.0 - jnp.sqrt(jnp.finfo(rhs.real.dtype).eps))
            )
            final_breakdown = jnp.where(
                finite_candidate,
                jnp.where(
                    converged,
                    int(KrylovBreakdownStatus.HAPPY),
                    jnp.where(
                        (inner_breakdown != int(KrylovBreakdownStatus.NONE))
                        & (inner_breakdown != int(KrylovBreakdownStatus.HAPPY))
                        & ~progressing_restart,
                        inner_breakdown,
                        jnp.where(
                            stagnated,
                            int(KrylovBreakdownStatus.STAGNATION),
                            int(KrylovBreakdownStatus.NONE),
                        ),
                    ),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            blocking_breakdown = (final_breakdown != int(KrylovBreakdownStatus.NONE)) & (
                final_breakdown != int(KrylovBreakdownStatus.HAPPY)
            )
            next_active = (
                finite_candidate
                & ~converged
                & ~blocking_breakdown
                & (final_iterations < step_limit)
                & (final_iterations < max_steps)
                & ~_iteration_stop(inner_iteration_state)
            )
            return (
                candidate,
                candidate_residual,
                candidate_norm,
                final_iterations,
                final_breakdown,
                next_best,
                next_stagnant,
                next_active,
                previous_cycles + 1,
                inner_iteration_state,
            )

        def execute_lanes(operand: _FGMRESCycleCarry) -> _FGMRESCycleCarry:
            executed = execute_cycle(operand)
            return jax.tree.map(
                lambda new, old: jnp.where(cycle_executing, new, old),
                executed,
                operand,
            )

        # The cond nests the checkpointed Arnoldi scan, so its predicate must
        # stay unbatched. A batched predicate turns the cond into a select whose
        # reverse mode rematerializes the scan on zero-filled residuals for
        # gated-off lanes, including the operator's closure constants. The
        # cycle instead runs while any lane executes, as a select-converted cond
        # would, and gated-off lanes keep their carry. Unbatched, the cond is
        # unchanged and skips every cycle after the solve stops.
        return jax.lax.cond(
            eqxi.unvmap_any(cycle_executing),
            execute_lanes,
            lambda operand: operand,
            state,
        )

    def cycle_condition(state: _FGMRESCycleCarry) -> Array:
        return (
            state[7]
            & (state[3] < step_limit)
            & (state[3] < max_steps)
            & (state[8] < cycles)
        )

    if fixed_trip:
        # One checkpoint per restart cycle: reverse mode stores cycle carries
        # and recomputes at most one cycle of Arnoldi work at a time.
        final_state, _ = checkpointed_scan(
            lambda state, _: (cycle_step(None, state), None),
            initial_state,
            None,
            length=cycles,
            mode="step",
        )
    else:
        final_state = jax.lax.while_loop(
            cycle_condition,
            lambda state: cycle_step(None, state),
            initial_state,
        )
    (
        x,
        _,
        residual_norm,
        iterations,
        breakdown,
        _,
        _,
        _,
        executed_cycles,
        iteration_state,
    ) = final_state
    return (
        x,
        (
            iterations,
            residual_norm,
            jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
            jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
            breakdown,
            executed_cycles,
        ),
        iteration_state,
    )


def _craig_solve(
    problem: AbstractLinearProblem,
    rhs: Array,
    plan: LinearSolvePlan,
    *,
    preconditioner: AbstractPreconditioner | None,
    relative: Array,
    absolute: Array,
    max_steps: Array,
    structural_max_steps: int,
    iteration: IterationPlan | None = None,
    iteration_state: IterationRuntimeState | None = None,
) -> _SolveResult:
    """Preconditioned Krylov solve of ``B B* y = b`` from zero, returning ``y``.

    The iteration is preconditioned MINRES: on a consistent system it converges
    like conjugate gradients (``B B*`` is positive semidefinite and every
    preconditioner is positive definite), and its residual ``b - B B* y`` is the
    true constraint residual of ``z = B* y``, confirmed with true actions before
    every stop. A redundant consistent system has a singular ``B B*`` whose range
    contains ``b``. An inconsistent ``b`` is never declared converged: MINRES
    approaches the preconditioned least-squares point, where the weighted
    residual ``M r`` lies in ``null(B*)`` and is a left-null witness. Its
    stationarity ``||B* M r|| <= max(relative, sqrt(eps)) ||M^1/2 B|| ||r||_M`` is
    confirmed with true actions and reported as ``STAGNATION``; the iteration
    stops there instead of amplifying roundoff along ``null(B*)``. Conjugate
    gradients have no such point on an inconsistent singular system.
    """
    operator = problem.operator
    gram = RowGramLinearOperator(operator)
    target = operator.target
    y, minres_auxiliary, next_iteration_state = _minres_raw(
        lambda vector: _action_coordinates(gram, vector),
        rhs,
        jnp.zeros_like(rhs),
        lambda left, right: _space_inner(target, left, right),
        _preconditioner_action(preconditioner, target),
        structural_max_steps,
        relative,
        absolute,
        step_limit=max_steps,
        driver=_loop_driver(plan),
        iteration=iteration,
        iteration_state=iteration_state,
        row_gram_stop=True,
    )
    iterations, residual, normal, condition, breakdown, confirmations = minres_auxiliary
    # B B* actions: one per step plus the initial and final residuals, and the
    # confirmation actions; z = B* y adds one adjoint action.
    gram_actions = iterations + confirmations + 2
    return (
        y,
        (
            iterations,
            residual,
            normal,
            condition,
            breakdown,
            jnp.asarray(gram_actions, dtype=jnp.int32),
            jnp.asarray(gram_actions + 1, dtype=jnp.int32),
            jnp.asarray(jnp.nan, dtype=residual.dtype),
        ),
        next_iteration_state,
    )


def _least_squares_solve(
    problem: AbstractLinearProblem,
    rhs: Array,
    initial: Array,
    plan: LinearSolvePlan,
    *,
    relative: Array,
    absolute: Array,
    max_steps: Array,
    structural_max_steps: int,
    iteration: IterationPlan | None = None,
    iteration_state: IterationRuntimeState | None = None,
) -> _SolveResult:
    action, adjoint, target_inner, source_inner, right = _least_squares_actions(
        problem, rhs
    )
    selected = plan.policy.method
    method = (
        selected if isinstance(selected, (GeneralizedLSMR, LSMR)) else GeneralizedLSMR()
    )
    max_steps_ = structural_max_steps
    value, auxiliary, next_iteration_state = _lsmr_raw(
        action,
        adjoint,
        right,
        initial,
        source_inner,
        target_inner,
        max_steps_,
        relative,
        absolute,
        method.condition_limit,
        method.damping,
        constraint=isinstance(problem, MinimumNormProblem),
        step_limit=max_steps,
        driver=_loop_driver(plan),
        iteration=iteration,
        iteration_state=iteration_state,
    )
    iterations, confirmations = auxiliary[0], auxiliary[5]
    # A least-squares solve spends one adjoint action on its acceptance reference.
    reference_actions = 0 if isinstance(problem, MinimumNormProblem) else 1
    return (
        value,
        (
            *auxiliary[:5],
            jnp.asarray(iterations + 3 + confirmations, dtype=jnp.int32),
            jnp.asarray(
                iterations + 2 + confirmations + reference_actions, dtype=jnp.int32
            ),
            auxiliary[6],
        ),
        next_iteration_state,
    )


def _lsmr_raw(
    action: _TargetAction,
    adjoint: _TargetAdjoint,
    rhs: _TargetPair,
    initial: Array,
    source_inner: _Inner,
    target_inner: _TargetInner,
    max_steps: int,
    relative: Array,
    absolute: Array,
    condition_limit: float,
    damping: float,
    *,
    constraint: bool = False,
    normal_residual_limit: Array | None = None,
    step_limit: Array | None = None,
    driver: KrylovLoopDriver = "early-exit",
    iteration: IterationPlan | None = None,
    iteration_state: IterationRuntimeState | None = None,
) -> _LSMRResult:
    """Native LSMR returning ``_KrylovAuxiliary``, true-quantity confirmations, and
    the stationarity roundoff floor of the returned point.

    Both modes stop only on true quantities; recurrence estimates merely schedule
    one confirming forward and adjoint action. ``constraint=True`` solves an exact
    (minimum-norm) constraint: it stops on a confirmed true residual or a
    scale-free least-squares stationary point (reported as ``STAGNATION``), never
    on an absolute normal-residual threshold. Otherwise the least-squares
    stationarity residual ``||A*(b - A x) - d^2 x||`` is accepted (``HAPPY``)
    against the reference of the solve status, ``absolute + relative ||A* b||``,
    or against its evaluation roundoff floor ``4 sqrt(max(m, n)) eps ||A|| ||r||``
    (``||A||`` the recurrence Frobenius estimate): the probabilistic rounding scale
    of ``A* r`` for a large residual, below which no iterate can be distinguished
    from a stationary point.
    A supplied ``normal_residual_limit`` additionally bounds the confirmed normal
    residual before declaring stationarity (used by certified derivative actions).
    """
    if step_limit is None:
        step_limit = jnp.asarray(max_steps, dtype=jnp.int32)
    residual = _target_subtract(rhs, action(initial))
    beta = _target_norm(residual, target_inner)
    u_operator, u_regularizer = _target_scale(
        residual, 1.0 / jnp.where(beta > 0.0, beta, 1.0)
    )
    v = adjoint((u_operator, u_regularizer))
    alpha = _norm(v, source_inner)
    v = v / jnp.where(alpha > 0.0, alpha, 1.0).astype(v.dtype)
    norm_b = _target_norm(rhs, target_inner)
    state = _LSMRState(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        alpha=alpha,
        u_operator=u_operator,
        u_regularizer=u_regularizer,
        v=v,
        alphabar=alpha,
        rho=jnp.asarray(1.0, dtype=rhs[0].real.dtype),
        rhobar=jnp.asarray(1.0, dtype=rhs[0].real.dtype),
        zeta=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        sbar=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        cbar=jnp.asarray(1.0, dtype=rhs[0].real.dtype),
        zetabar=alpha * beta,
        hbar=jnp.zeros_like(initial),
        h=v,
        x=initial,
        betadd=beta,
        thetatilde=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        rhodold=jnp.asarray(1.0, dtype=rhs[0].real.dtype),
        betad=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        tautildeold=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        accumulated_residual=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        norm_a_squared=alpha * alpha,
        maximum_rbar=jnp.asarray(0.0, dtype=rhs[0].real.dtype),
        minimum_rbar=jnp.asarray(jnp.inf, dtype=rhs[0].real.dtype),
        normal_residual=alpha * beta,
        residual=beta,
        norm_a=alpha,
        condition=jnp.asarray(1.0, dtype=rhs[0].real.dtype),
        active=(norm_b > 0.0) & (alpha * beta > 0.0) & ~_iteration_stop(iteration_state),
        breakdown=jnp.asarray(int(KrylovBreakdownStatus.NONE), dtype=jnp.int32),
        confirmations=jnp.asarray(0, dtype=jnp.int32),
        iteration_state=iteration_state,
    )
    roundoff = (
        4.0
        * math.sqrt(max(initial.size, rhs[0].size + rhs[1].size))
        * float(jnp.finfo(rhs[0].real.dtype).eps)
    )
    # Least-squares acceptance reference of the solve status (one adjoint action).
    normal_target = (
        absolute
        if constraint
        else absolute + relative * _norm(adjoint(rhs), source_inner)
    )

    def true_quantities(point: Array) -> tuple[Array, Array]:
        true_residual = _target_subtract(rhs, action(point))
        gradient = adjoint(true_residual)
        if damping:
            gradient = gradient - damping**2 * point
        return _target_norm(true_residual, target_inner), _norm(gradient, source_inner)

    def stationarity_floor(norm_a: Array, residual_norm: Array) -> Array:
        return roundoff * norm_a * residual_norm

    def step(index: Array, current: _LSMRState) -> _LSMRState:
        executing = current.active & (index < step_limit)

        def execute(value: _LSMRState) -> _LSMRState:
            image_operator, image_regularizer = action(
                _gated_input(executing, value.v, initial)
            )
            next_u_operator = (
                image_operator
                - value.alpha.astype(value.u_operator.dtype) * value.u_operator
            )
            next_u_regularizer = (
                image_regularizer
                - value.alpha.astype(value.u_regularizer.dtype) * value.u_regularizer
            )
            next_beta = _target_norm((next_u_operator, next_u_regularizer), target_inner)
            next_u_operator, next_u_regularizer = _target_scale(
                (next_u_operator, next_u_regularizer),
                1.0 / jnp.where(next_beta > 0.0, next_beta, 1.0),
            )
            next_v = (
                adjoint(
                    _gated_input(
                        executing,
                        (next_u_operator, next_u_regularizer),
                        (u_operator, u_regularizer),
                    )
                )
                - next_beta.astype(value.v.dtype) * value.v
            )
            next_alpha = _norm(next_v, source_inner)
            next_v = next_v / jnp.where(next_alpha > 0.0, next_alpha, 1.0).astype(
                next_v.dtype
            )
            chat, shat, alphahat = _symmetric_orthogonalization(
                value.alphabar, jnp.asarray(damping, dtype=value.alphabar.dtype)
            )
            cosine, sine, rho = _symmetric_orthogonalization(alphahat, next_beta)
            theta_new = sine * next_alpha
            alpha_bar = cosine * next_alpha
            theta_bar = value.sbar * rho
            rho_temp = value.cbar * rho
            cbar, sbar, rho_bar = _symmetric_orthogonalization(rho_temp, theta_new)
            zeta = cbar * value.zetabar
            zeta_bar = -sbar * value.zetabar
            safe_denominator = jnp.where(
                value.rho * value.rhobar == 0.0,
                1.0,
                value.rho * value.rhobar,
            )
            hbar = value.h - value.hbar * (theta_bar * rho / safe_denominator).astype(
                value.hbar.dtype
            )
            x = (
                value.x
                + (zeta / jnp.where(rho * rho_bar == 0.0, 1.0, rho * rho_bar)).astype(
                    hbar.dtype
                )
                * hbar
            )
            h = next_v - value.h * (theta_new / jnp.where(rho == 0.0, 1.0, rho)).astype(
                value.h.dtype
            )
            beta_acute = chat * value.betadd
            beta_check = -shat * value.betadd
            beta_hat = cosine * beta_acute
            beta_dd = -sine * beta_acute
            ctilde, stilde, rho_tilde = _symmetric_orthogonalization(
                value.rhodold, theta_bar
            )
            theta_tilde = stilde * rho_bar
            rho_d = ctilde * rho_bar
            beta_d = -stilde * value.betad + ctilde * beta_hat
            tau_tilde = (value.zeta - value.thetatilde * value.tautildeold) / jnp.where(
                rho_tilde == 0.0, 1.0, rho_tilde
            )
            tau_d = (zeta - theta_tilde * tau_tilde) / jnp.where(rho_d == 0.0, 1.0, rho_d)
            accumulated = value.accumulated_residual + beta_check * beta_check
            residual_norm = jnp.sqrt(
                accumulated + (beta_d - tau_d) ** 2 + beta_dd * beta_dd
            )
            norm_a_squared = value.norm_a_squared + next_beta**2
            norm_a = jnp.sqrt(norm_a_squared)
            norm_a_squared = norm_a_squared + next_alpha**2
            maximum = jnp.maximum(value.maximum_rbar, value.rhobar)
            minimum = jnp.minimum(value.minimum_rbar, jnp.abs(value.rhobar))
            condition = jnp.maximum(maximum, jnp.abs(rho_temp)) / jnp.maximum(
                jnp.minimum(minimum, jnp.abs(rho_temp)),
                jnp.finfo(rho_temp.dtype).tiny,
            )
            normal_residual = jnp.abs(zeta_bar)
            next_iteration_index = value.iteration + 1
            if constraint:
                # An exact constraint stops only on true quantities; recurrence
                # estimates merely schedule one forward plus one adjoint action.
                # It stops on a true residual within tolerance, or on a
                # least-squares stationary point (r orthogonal to the range) by
                # the scale-free backward-error test ||A* r|| <= tol ||A|| ||r||
                # (||A|| the recurrence Frobenius estimate), never on an absolute
                # normal-residual threshold, which would stop early by a factor of
                # the smallest singular value. Estimated normal residuals drift
                # from true ones without reorthogonalization, so a stationary
                # estimate alone never stops.
                residual_threshold = absolute + relative * norm_b
                stationarity_tolerance = jnp.maximum(
                    relative, jnp.finfo(residual_norm.dtype).eps
                )
                triggered = (residual_norm <= residual_threshold) | (
                    normal_residual <= stationarity_tolerance * norm_a * residual_norm
                )

                infinity = jnp.asarray(jnp.inf, dtype=residual_norm.dtype)
                confirmed_residual, confirmed_normal = jax.lax.cond(
                    triggered,
                    lambda: true_quantities(_gated_input(executing, x, initial)),
                    lambda: (infinity, infinity),
                )
                converged = confirmed_residual <= residual_threshold
                stationary = (
                    triggered
                    & ~converged
                    & (
                        confirmed_normal
                        <= stationarity_tolerance * norm_a * confirmed_residual
                    )
                )
                if normal_residual_limit is not None:
                    stationary = stationary & (confirmed_normal <= normal_residual_limit)
                confirmations = value.confirmations + triggered.astype(jnp.int32)
            else:
                residual_threshold = absolute + relative * norm_b
                triggered = (
                    normal_residual
                    <= jnp.maximum(
                        normal_target, stationarity_floor(norm_a, residual_norm)
                    )
                ) | (residual_norm <= residual_threshold)
                infinity = jnp.asarray(jnp.inf, dtype=residual_norm.dtype)
                confirmed_residual, confirmed_normal = jax.lax.cond(
                    triggered,
                    lambda: true_quantities(_gated_input(executing, x, initial)),
                    lambda: (infinity, infinity),
                )
                # Unconfirmed steps carry infinite placeholders (and an infinite
                # floor), so acceptance is gated on the confirmation itself.
                converged = triggered & (
                    confirmed_normal
                    <= jnp.maximum(
                        normal_target,
                        stationarity_floor(norm_a, confirmed_residual),
                    )
                )
                stationary = jnp.asarray(False)
                confirmations = value.confirmations + triggered.astype(jnp.int32)
            condition_limited = condition >= condition_limit
            finite = (
                jnp.all(jnp.isfinite(x))
                & jnp.isfinite(residual_norm)
                & jnp.isfinite(normal_residual)
            )
            recurrence_breakdown = (rho == 0.0) | (rho_bar == 0.0)
            breakdown = jnp.where(
                finite,
                jnp.where(
                    converged,
                    int(KrylovBreakdownStatus.HAPPY),
                    jnp.where(
                        stationary,
                        int(KrylovBreakdownStatus.STAGNATION),
                        jnp.where(
                            recurrence_breakdown,
                            int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
                            int(KrylovBreakdownStatus.NONE),
                        ),
                    ),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            next_iteration = _update_krylov_iteration(
                iteration,
                value.iteration_state,
                next_iteration_index,
                residual_norm,
                norm_b,
                breakdown,
                matvec_count=next_iteration_index + 2 + confirmations,
                adjoint_matvec_count=next_iteration_index + 2 + confirmations,
                normal_residual_norm=normal_residual,
                condition_estimate=condition,
            )
            return _LSMRState(
                next_iteration_index,
                next_alpha,
                next_u_operator,
                next_u_regularizer,
                next_v,
                alpha_bar,
                rho,
                rho_bar,
                zeta,
                sbar,
                cbar,
                zeta_bar,
                hbar,
                h,
                x,
                beta_dd,
                theta_tilde,
                rho_d,
                beta_d,
                tau_tilde,
                accumulated,
                norm_a_squared,
                maximum,
                minimum,
                normal_residual,
                residual_norm,
                norm_a,
                condition,
                finite
                & ~converged
                & ~stationary
                & ~condition_limited
                & ~recurrence_breakdown
                & ~_iteration_stop(next_iteration),
                breakdown,
                confirmations,
                next_iteration,
            )

        return jax.lax.cond(executing, execute, lambda value: value, current)

    state = _gated_loop(
        step,
        state,
        max_steps,
        lambda index, current: current.active & (index < step_limit),
        driver,
    )
    true_residual, true_normal = true_quantities(state.x)
    floor = stationarity_floor(state.norm_a, true_residual)
    breakdown = state.breakdown
    if not constraint:
        # Final classification from the returned point's true stationarity, so
        # an unexecuted, recurrence-ended, or condition-limited solve reports
        # HAPPY exactly when the solve status accepts it.
        accepted = jnp.isfinite(true_normal) & (
            true_normal <= jnp.maximum(normal_target, floor)
        )
        breakdown = jnp.where(
            accepted,
            int(KrylovBreakdownStatus.HAPPY),
            jnp.where(
                breakdown == int(KrylovBreakdownStatus.HAPPY),
                int(KrylovBreakdownStatus.NONE),
                breakdown,
            ),
        ).astype(jnp.int32)
    return (
        state.x,
        (
            state.iteration,
            true_residual,
            true_normal,
            state.condition,
            breakdown,
            state.confirmations,
            floor,
        ),
        state.iteration_state,
    )


class MinimumNormLSMRResult(NamedTuple):
    value: Array
    iterations: Array
    residual_norm: Array
    normal_residual_norm: Array
    breakdown: Array
    confirmations: Array


def _minimum_norm_lsmr(
    action: _Action,
    adjoint: _Action,
    rhs: Array,
    domain_size: int,
    domain_inner: _Inner,
    codomain_inner: _Inner,
    /,
    *,
    max_steps: int,
    relative: Array,
    absolute: Array,
    condition_limit: float,
    normal_residual_limit: Array | None = None,
) -> MinimumNormLSMRResult:
    """Undamped zero-start LSMR for one coordinate column.

    Every iterate lies in the range of ``adjoint``, so the limit is the
    pseudoinverse (minimum-norm least-squares) solution in the declared pairings.
    It stops on a confirmed true residual or a scale-free least-squares stationary
    point subject to ``normal_residual_limit`` when supplied; residual norms are
    recomputed with true actions.
    """
    empty = jnp.zeros((0,), dtype=rhs.dtype)

    def pair_action(vector: Array) -> _TargetPair:
        return action(vector), empty

    def pair_adjoint(pair: _TargetPair) -> Array:
        return adjoint(pair[0])

    def pair_inner(left: _TargetPair, right: _TargetPair) -> Array:
        return codomain_inner(left[0], right[0])

    value, auxiliary, _ = _lsmr_raw(
        pair_action,
        pair_adjoint,
        (rhs, empty),
        jnp.zeros((domain_size,), dtype=rhs.dtype),
        domain_inner,
        pair_inner,
        max_steps,
        relative,
        absolute,
        condition_limit,
        0.0,
        constraint=True,
        normal_residual_limit=normal_residual_limit,
    )
    iterations, residual, normal, _, breakdown, confirmations, _ = auxiliary
    return MinimumNormLSMRResult(
        value, iterations, residual, normal, breakdown, confirmations
    )


def _stationarity_witness(
    operator: AbstractLinearOperator,
    value: Array,
    /,
    *,
    relative: Array,
    absolute: Array,
    max_steps: int,
    condition_limit: float,
) -> tuple[Array, Array, Array, Array]:
    """Multiplier ``y`` of ``min ||A* y - x||_M`` and the true ``||x - A* y||_M``.

    The residual bounds the distance of ``x`` from ``range(A*)``, i.e. the
    component of ``x`` in ``null(A)``, without materializing any nullspace.
    """
    result = _minimum_norm_lsmr(
        lambda vector: _adjoint_coordinates(operator, vector),
        lambda vector: _action_coordinates(operator, vector),
        value,
        operator.target.size,
        lambda left, right: _space_inner(operator.target, left, right),
        lambda left, right: _space_inner(operator.source, left, right),
        max_steps=max_steps,
        relative=relative,
        absolute=absolute,
        condition_limit=condition_limit,
    )
    return result.value, result.residual_norm, result.iterations, result.confirmations


def _least_squares_actions(
    problem: AbstractLinearProblem, rhs: Array
) -> tuple[_TargetAction, _TargetAdjoint, _TargetInner, _Inner, _TargetPair]:
    operator = problem.operator
    regularizer = (
        problem.regularizer if isinstance(problem, LeastSquaresProblem) else None
    )
    weights = None
    if isinstance(problem, LeastSquaresProblem) and problem.weights is not None:
        weights = jnp.asarray(problem.weights).reshape((-1,))
        if weights.shape != (operator.target.size,):
            raise ValueError(
                "GeneralizedLSMR weights must have one target coordinate entry."
            )

    def action(vector: Array) -> _TargetPair:
        primary = _action_coordinates(operator, vector)
        secondary = (
            jnp.zeros((0,), dtype=primary.dtype)
            if regularizer is None
            else _action_coordinates(regularizer, vector)
        )
        return primary, secondary

    def adjoint(value: _TargetPair) -> Array:
        primary, secondary = value
        weighted = primary if weights is None else weights * primary
        result = _adjoint_coordinates(operator, weighted)
        if regularizer is not None:
            result = result + _adjoint_coordinates(regularizer, secondary)
        return result

    def target_inner(left: _TargetPair, right: _TargetPair) -> Array:
        left_primary, left_secondary = left
        right_primary, right_secondary = right
        weighted_right = right_primary if weights is None else weights * right_primary
        value = _space_inner(operator.target, left_primary, weighted_right)
        if regularizer is not None:
            value = value + _space_inner(
                regularizer.target, left_secondary, right_secondary
            )
        return value

    source_inner = lambda left, right: _space_inner(operator.source, left, right)
    right = (
        rhs,
        jnp.zeros(
            (0 if regularizer is None else regularizer.target.size,), dtype=rhs.dtype
        ),
    )
    return action, adjoint, target_inner, source_inner, right


def _preconditioner_action(
    preconditioner: AbstractPreconditioner | None, space: AbstractVectorSpace
) -> _Precondition:
    if preconditioner is None:
        return lambda vector, iteration: vector

    def apply(vector: Array, iteration: Array) -> Array:
        return space.flatten(
            preconditioner.apply(space.unflatten(vector), iteration=iteration)
        )

    return apply


def _action_coordinates(operator: AbstractLinearOperator, vector: Array) -> Array:
    return operator.target.flatten(operator.mv(operator.source.unflatten(vector)))


def _adjoint_coordinates(operator: AbstractLinearOperator, vector: Array) -> Array:
    return operator.source.flatten(operator.adjoint_mv(operator.target.unflatten(vector)))


def _space_inner(space: AbstractVectorSpace, left: Array, right: Array) -> Array:
    return space.inner(space.unflatten(left), space.unflatten(right))


def _space_norm(space: AbstractVectorSpace, vector: Array) -> Array:
    return _norm(vector, lambda left, right: _space_inner(space, left, right))


def _norm(vector: Array, inner: _Inner) -> Array:
    return _norm_from_squared(inner(vector, vector))


def _safe_abs(value: Array) -> Array:
    # Value-identical to |z|; complex |z| otherwise has no derivative at zero.
    nonzero = value != 0
    return jnp.where(nonzero, jnp.abs(jnp.where(nonzero, value, 1)), 0)


def _target_norm(value: _TargetPair, inner: _TargetInner) -> Array:
    return _norm_from_squared(inner(value, value))


def _target_subtract(left: _TargetPair, right: _TargetPair) -> _TargetPair:
    return left[0] - right[0], left[1] - right[1]


def _target_scale(value: _TargetPair, scalar: Array) -> _TargetPair:
    return (
        scalar.astype(value[0].dtype) * value[0],
        scalar.astype(value[1].dtype) * value[1],
    )


def _symmetric_orthogonalization(left: Array, right: Array) -> tuple[Array, Array, Array]:
    radius = jnp.hypot(left, right)
    safe = jnp.where(radius == 0.0, 1.0, radius)
    return left / safe, right / safe, radius


__all__ = [
    "NativeKrylovBackendOutput",
    "NativeKrylovState",
    "prepare_native_krylov",
    "solve_native_krylov",
]
