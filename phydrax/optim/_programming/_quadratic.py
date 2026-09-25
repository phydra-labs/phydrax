#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._bounds import Bounds
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    DenseSVD,
    DifferentiationPolicy,
    FactorizationPolicy,
    factorize,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSystem,
    prepare_linearization,
    RankPolicy,
    solve as solve_linear,
)
from ._policy import (
    ConvexDifferentiationPolicy,
    ConvexSolvePolicy,
    DensePrimalDualQP,
    MPAXr2HPDHG,
    MPAXraPDHG,
)
from ._types import (
    ConvexProgramCertificate,
    ConvexProgramProvenance,
    ConvexProgramStatus,
    ConvexWarmStart,
)


def _batch_shape(value: Sequence[int], /) -> tuple[int, ...]:
    return tuple(value)


def _canonical_matrix(
    value: ArrayLike | None,
    /,
    *,
    rows_from: ArrayLike | None,
    variables: int,
    name: str,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    matrix = None if value is None else jnp.asarray(value)
    rhs = None if rows_from is None else jnp.asarray(rows_from)
    if (matrix is not None and jnp.issubdtype(matrix.dtype, jnp.complexfloating)) or (
        rhs is not None and jnp.issubdtype(rhs.dtype, jnp.complexfloating)
    ):
        raise TypeError("QuadraticProgram data must be real-valued.")
    if matrix is None:
        if rhs is not None:
            raise ValueError(f"{name} and its right-hand side must be provided together.")
        return (
            jnp.empty((0, variables), dtype=dtype),
            jnp.empty((0,), dtype=dtype),
        )
    if rhs is None:
        raise ValueError(f"{name} and its right-hand side must be provided together.")
    matrix = matrix.astype(dtype)
    rhs = rhs.astype(dtype)
    if matrix.ndim < 2:
        raise ValueError(
            f"{name} must have shape batch_shape + (constraints, variables)."
        )
    if matrix.shape[-1] != variables:
        raise ValueError(
            f"{name} has {matrix.shape[-1]} variables; expected {variables}."
        )
    if rhs.ndim < 1 or rhs.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            f"The right-hand side for {name} must end in shape ({matrix.shape[-2]},)."
        )
    return matrix, rhs


def _static_bound_values(
    bounds: Bounds,
    batch_shape: tuple[int, ...],
    variables: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    lower_metadata = bounds._lower_metadata
    upper_metadata = bounds._upper_metadata
    if (
        lower_metadata is None
        or upper_metadata is None
        or len(lower_metadata) != 1
        or len(upper_metadata) != 1
    ):
        raise ValueError(
            "QuadraticProgram bounds require static scalar or array role metadata."
        )
    metadata = (lower_metadata, upper_metadata)
    arrays = []
    for bound_metadata in metadata:
        shape, dtype, values = bound_metadata[0]
        raw = np.asarray(values, dtype=np.dtype(dtype)).reshape(shape)
        arrays.append(np.broadcast_to(raw, batch_shape + (variables,)))
    return arrays[0], arrays[1]


def _append_bound_rows(
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    lower: Array,
    upper: Array,
    /,
    *,
    batch_shape: tuple[int, ...],
    fixed_indices: tuple[int, ...],
    lower_indices: tuple[int, ...],
    upper_indices: tuple[int, ...],
) -> tuple[Array, Array, Array, Array]:
    """Append fixed-bound equality rows and one-sided bound inequality rows."""
    variables = lower.shape[-1]
    identity = jnp.eye(variables, dtype=equality_matrix.dtype)
    fixed = jnp.asarray(fixed_indices, dtype=jnp.int32)
    lower_rows = jnp.asarray(lower_indices, dtype=jnp.int32)
    upper_rows = jnp.asarray(upper_indices, dtype=jnp.int32)
    fixed_matrix = jnp.broadcast_to(
        identity[fixed],
        batch_shape + (len(fixed_indices), variables),
    )
    lower_matrix = jnp.broadcast_to(
        -identity[lower_rows],
        batch_shape + (len(lower_indices), variables),
    )
    upper_matrix = jnp.broadcast_to(
        identity[upper_rows],
        batch_shape + (len(upper_indices), variables),
    )
    return (
        jnp.concatenate((equality_matrix, fixed_matrix), axis=-2),
        jnp.concatenate((equality_rhs, jnp.take(lower, fixed, axis=-1)), axis=-1),
        jnp.concatenate((inequality_matrix, lower_matrix, upper_matrix), axis=-2),
        jnp.concatenate(
            (
                inequality_rhs,
                -jnp.take(lower, lower_rows, axis=-1),
                jnp.take(upper, upper_rows, axis=-1),
            ),
            axis=-1,
        ),
    )


class QuadraticProgram(StrictModule):
    r"""A convex quadratic program in canonical equality/inequality form.

    The program is ``min 1/2 xᵀQx + qᵀx`` subject to ``Ax=b``, ``Gx<=h``,
    and optional native variable bounds. Bound roles are fixed across a batch.
    Finite fixed bounds become equality rows for numerical execution; one-sided
    finite bounds become inequality rows while retaining their public provenance.
    """

    quadratic: Array
    linear: Array
    equality_matrix: Array
    equality_rhs: Array
    inequality_matrix: Array
    inequality_rhs: Array
    lower_bounds: Array
    upper_bounds: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_equalities: int = eqx.field(static=True)
    num_inequalities: int = eqx.field(static=True)
    num_user_equalities: int = eqx.field(static=True)
    num_user_inequalities: int = eqx.field(static=True)
    fixed_bound_indices: tuple[int, ...] = eqx.field(static=True)
    lower_bound_indices: tuple[int, ...] = eqx.field(static=True)
    upper_bound_indices: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    convexity_evidence: str = eqx.field(static=True)

    def __init__(
        self,
        quadratic: ArrayLike,
        linear: ArrayLike,
        /,
        *,
        equality_matrix: ArrayLike | None = None,
        equality_rhs: ArrayLike | None = None,
        inequality_matrix: ArrayLike | None = None,
        inequality_rhs: ArrayLike | None = None,
        bounds: Bounds | None = None,
        problem_id: str = "canonical-quadratic-program",
        convexity_evidence: str = "asserted",
    ):
        quadratic_value = jnp.asarray(quadratic)
        linear_value = jnp.asarray(linear)
        if (
            quadratic_value.ndim < 2
            or quadratic_value.shape[-2] != quadratic_value.shape[-1]
        ):
            raise ValueError(
                "quadratic must have shape batch_shape + (variables, variables)."
            )
        variables = quadratic_value.shape[-1]
        if variables < 1:
            raise ValueError("QuadraticProgram requires at least one decision variable.")
        if linear_value.ndim < 1 or linear_value.shape[-1] != variables:
            raise ValueError(f"linear must have shape batch_shape + ({variables},).")
        dtype = jnp.result_type(quadratic_value, linear_value, jnp.float32)
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("QuadraticProgram data must be real-valued.")
        quadratic_value = quadratic_value.astype(dtype)
        quadratic_value = 0.5 * quadratic_value + 0.5 * jnp.swapaxes(
            quadratic_value, -1, -2
        )
        linear_value = linear_value.astype(dtype)
        equality_value, equality_rhs_value = _canonical_matrix(
            equality_matrix,
            rows_from=equality_rhs,
            variables=variables,
            name="equality_matrix",
            dtype=dtype,
        )
        inequality_value, inequality_rhs_value = _canonical_matrix(
            inequality_matrix,
            rows_from=inequality_rhs,
            variables=variables,
            name="inequality_matrix",
            dtype=dtype,
        )
        batch = np.broadcast_shapes(
            quadratic_value.shape[:-2],
            linear_value.shape[:-1],
            equality_value.shape[:-2],
            equality_rhs_value.shape[:-1],
            inequality_value.shape[:-2],
            inequality_rhs_value.shape[:-1],
        )
        user_equalities = equality_value.shape[-2]
        user_inequalities = inequality_value.shape[-2]
        quadratic_value = jnp.broadcast_to(
            quadratic_value, batch + (variables, variables)
        )
        linear_value = jnp.broadcast_to(linear_value, batch + (variables,))
        equality_value = jnp.broadcast_to(
            equality_value, batch + (user_equalities, variables)
        )
        equality_rhs_value = jnp.broadcast_to(
            equality_rhs_value, batch + (user_equalities,)
        )
        inequality_value = jnp.broadcast_to(
            inequality_value, batch + (user_inequalities, variables)
        )
        inequality_rhs_value = jnp.broadcast_to(
            inequality_rhs_value, batch + (user_inequalities,)
        )
        bounds_ = Bounds() if bounds is None else bounds
        if not isinstance(bounds_, Bounds):
            raise TypeError("bounds must be a Bounds or None.")
        lower, upper = bounds_.materialize(linear_value)
        lower = jnp.asarray(lower, dtype=dtype)
        upper = jnp.asarray(upper, dtype=dtype)
        static_lower, static_upper = _static_bound_values(bounds_, batch, variables)
        flat_lower = static_lower.reshape((-1, variables))
        flat_upper = static_upper.reshape((-1, variables))
        lower_finite = np.isfinite(flat_lower)
        upper_finite = np.isfinite(flat_upper)
        fixed = lower_finite & upper_finite & (flat_lower == flat_upper)
        roles = np.stack((lower_finite, upper_finite, fixed), axis=-1)
        if not np.all(roles == roles[:1]):
            raise ValueError(
                "QuadraticProgram bounds must have one shared finite/fixed role pattern across the batch."
            )
        fixed_indices = tuple(np.flatnonzero(fixed[0]))
        lower_indices = tuple(np.flatnonzero(lower_finite[0] & ~fixed[0]))
        upper_indices = tuple(np.flatnonzero(upper_finite[0] & ~fixed[0]))
        (
            equality_value,
            equality_rhs_value,
            inequality_value,
            inequality_rhs_value,
        ) = _append_bound_rows(
            equality_value,
            equality_rhs_value,
            inequality_value,
            inequality_rhs_value,
            lower,
            upper,
            batch_shape=batch,
            fixed_indices=fixed_indices,
            lower_indices=lower_indices,
            upper_indices=upper_indices,
        )
        identifier = str(problem_id)
        evidence = str(convexity_evidence)
        if not identifier or not evidence:
            raise ValueError("problem_id and convexity_evidence must be non-empty.")
        self.quadratic = quadratic_value
        self.linear = linear_value
        self.equality_matrix = equality_value
        self.equality_rhs = equality_rhs_value
        self.inequality_matrix = inequality_value
        self.inequality_rhs = inequality_rhs_value
        self.lower_bounds = lower
        self.upper_bounds = upper
        self.batch_shape = _batch_shape(batch)
        self.num_variables = variables
        self.num_equalities = user_equalities + len(fixed_indices)
        self.num_inequalities = (
            user_inequalities + len(lower_indices) + len(upper_indices)
        )
        self.num_user_equalities = user_equalities
        self.num_user_inequalities = user_inequalities
        self.fixed_bound_indices = fixed_indices
        self.lower_bound_indices = lower_indices
        self.upper_bound_indices = upper_indices
        self.problem_id = identifier
        self.convexity_evidence = evidence
        self.structure_id = canonical_fingerprint(
            {
                "kind": "quadratic-program",
                "problem_id": identifier,
                "batch_shape": list(batch),
                "variables": variables,
                "user_equalities": user_equalities,
                "user_inequalities": user_inequalities,
                "fixed_bounds": list(fixed_indices),
                "lower_bounds": list(lower_indices),
                "upper_bounds": list(upper_indices),
                "dtype": str(dtype),
            }
        )


def _rebind_quadratic_program(
    template: QuadraticProgram,
    /,
    *,
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    lower_bounds: Array,
    upper_bounds: Array,
) -> QuadraticProgram:
    """Rebind numeric QP data to a template's fixed static structure.

    The user rows and bounds are folded exactly as `QuadraticProgram` folds
    them, but the finite/fixed bound roles are the template's static roles, so
    the rebinding is traceable and affine in every input. Callers own the
    invariant that the new bounds keep the template's role pattern.
    """
    batch = template.batch_shape
    variables = template.num_variables
    expected = {
        "quadratic": (quadratic, batch + (variables, variables)),
        "linear": (linear, batch + (variables,)),
        "equality_matrix": (
            equality_matrix,
            batch + (template.num_user_equalities, variables),
        ),
        "equality_rhs": (equality_rhs, batch + (template.num_user_equalities,)),
        "inequality_matrix": (
            inequality_matrix,
            batch + (template.num_user_inequalities, variables),
        ),
        "inequality_rhs": (
            inequality_rhs,
            batch + (template.num_user_inequalities,),
        ),
        "lower_bounds": (lower_bounds, batch + (variables,)),
        "upper_bounds": (upper_bounds, batch + (variables,)),
    }
    for name, (value, shape) in expected.items():
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}; got {value.shape}.")
    dtype = template.linear.dtype
    quadratic = quadratic.astype(dtype)
    lower = lower_bounds.astype(dtype)
    upper = upper_bounds.astype(dtype)
    folded = _append_bound_rows(
        equality_matrix.astype(dtype),
        equality_rhs.astype(dtype),
        inequality_matrix.astype(dtype),
        inequality_rhs.astype(dtype),
        lower,
        upper,
        batch_shape=batch,
        fixed_indices=template.fixed_bound_indices,
        lower_indices=template.lower_bound_indices,
        upper_indices=template.upper_bound_indices,
    )
    return eqx.tree_at(
        lambda program: (
            program.quadratic,
            program.linear,
            program.equality_matrix,
            program.equality_rhs,
            program.inequality_matrix,
            program.inequality_rhs,
            program.lower_bounds,
            program.upper_bounds,
        ),
        template,
        (
            0.5 * quadratic + 0.5 * jnp.swapaxes(quadratic, -1, -2),
            linear.astype(dtype),
            *folded,
            lower,
            upper,
        ),
    )


class PreparedQPSensitivity(StrictModule):
    """Reusable primal solution and linear sensitivity actions for one dense QP.

    `regular` has the program's batch shape: one flag per batched QP case.
    """

    primal: Array
    pushforward: Callable[[QuadraticProgram], Array]
    pullback: Callable[[Array], QuadraticProgram]
    regular: Array
    differentiation: ConvexDifferentiationPolicy = eqx.field(static=True)

    def jvp(self, tangent: QuadraticProgram, /) -> Array:
        return self.pushforward(tangent)

    def vjp(self, cotangent: ArrayLike, /) -> QuadraticProgram:
        value = jnp.asarray(cotangent, dtype=self.primal.dtype)
        if value.shape != self.primal.shape:
            raise ValueError("QP cotangent shape must match the primal solution.")
        return self.pullback(value)


class ConvexProgramResult(StrictModule):
    """Primal/dual solution, KKT diagnostics, and complete solver provenance."""

    primal: Array
    equality_dual: Array
    inequality_dual: Array
    inequality_slack: Array
    cone_slack: Array
    cone_dual: Array
    cone_primal_residual: Array
    cone_violation: Array
    cone_dual_violation: Array
    cone_complementarity: Array
    lower_bound_dual: Array
    upper_bound_dual: Array
    objective: Array
    stationarity_residual: Array
    solver_stationarity_residual: Array
    equality_residual: Array
    inequality_residual: Array
    inequality_violation: Array
    complementarity_residual: Array
    primal_residual_norm: Array
    dual_residual_norm: Array
    solver_dual_residual_norm: Array
    complementarity_gap: Array
    kkt_residual_norm: Array
    iterations: Array
    backend_converged: Array
    valid: Array
    status: Array
    certificate: ConvexProgramCertificate
    provenance: ConvexProgramProvenance
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    max_iterations: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(ConvexProgramStatus.OPTIMAL))


def _apply_failure_policy(
    result: ConvexProgramResult,
    policy: ConvexSolvePolicy,
    /,
) -> ConvexProgramResult:
    if policy.failure.mode == "status":
        return result
    checked_primal = eqx.error_if(
        result.primal,
        jnp.any(~result.successful),
        "Convex program failed its audited numerical contract.",
    )
    return eqx.tree_at(
        lambda candidate: candidate.primal,
        result,
        checked_primal,
    )


def _validate_quadratic_materialization(
    problem: QuadraticProgram,
    policy: ConvexSolvePolicy,
    /,
) -> tuple[int, int]:
    arrays = (
        problem.quadratic,
        problem.linear,
        problem.equality_matrix,
        problem.equality_rhs,
        problem.inequality_matrix,
        problem.inequality_rhs,
        problem.lower_bounds,
        problem.upper_bounds,
    )
    input_entries = sum(array.size for array in arrays)
    input_bytes = sum(array.size * array.dtype.itemsize for array in arrays)
    if input_entries > policy.materialization.max_entries:
        raise ValueError(
            f"Quadratic program requires {input_entries} materialized entries, "
            f"exceeding the policy limit {policy.materialization.max_entries}."
        )
    if input_bytes > policy.materialization.max_bytes:
        raise ValueError(
            f"Quadratic program requires {input_bytes} materialized bytes, "
            f"exceeding the policy limit {policy.materialization.max_bytes}."
        )
    return input_entries, input_bytes


def _validate_quadratic_resources(
    problem: QuadraticProgram,
    policy: ConvexSolvePolicy,
    /,
    *,
    max_dense_dimension: int,
) -> None:
    input_entries, input_bytes = _validate_quadratic_materialization(problem, policy)
    kkt_dimension = (
        problem.num_variables + problem.num_equalities + 2 * problem.num_inequalities
    )
    if kkt_dimension > max_dense_dimension:
        raise ValueError(
            f"Dense QP dimension {kkt_dimension} exceeds max_dense_dimension={max_dense_dimension}."
        )
    batch_count = prod(problem.batch_shape or (1,))
    itemsize = problem.linear.dtype.itemsize
    kkt_entries = batch_count * kkt_dimension * kkt_dimension
    kkt_bytes = kkt_entries * itemsize
    materialization_entries = max(input_entries, kkt_entries)
    materialization_bytes = max(input_bytes, kkt_bytes)
    if materialization_entries > policy.materialization.max_entries:
        raise ValueError(
            f"Dense QP requires {materialization_entries} materialized entries, "
            f"exceeding the policy limit {policy.materialization.max_entries}."
        )
    if materialization_bytes > policy.materialization.max_bytes:
        raise ValueError(
            f"Dense QP requires {materialization_bytes} materialized bytes, "
            f"exceeding the policy limit {policy.materialization.max_bytes}."
        )
    if kkt_bytes > policy.resources.factorization_bytes:
        raise ValueError(
            f"Dense QP factorization estimate {kkt_bytes} bytes exceeds "
            f"the resource limit {policy.resources.factorization_bytes}."
        )
    workspace_bytes = kkt_bytes + 4 * batch_count * kkt_dimension * itemsize
    if workspace_bytes > policy.resources.workspace_bytes:
        raise ValueError(
            f"Dense QP workspace estimate {workspace_bytes} bytes exceeds "
            f"the resource limit {policy.resources.workspace_bytes}."
        )


def _max_abs(value: Array, /) -> Array:
    if value.shape[-1] == 0:
        return jnp.zeros(value.shape[:-1], dtype=value.dtype)
    return jnp.max(jnp.abs(value), axis=-1)


def _min_value(value: Array, /) -> Array:
    if value.shape[-1] == 0:
        return jnp.full(value.shape[:-1], jnp.inf, dtype=value.dtype)
    return jnp.min(value, axis=-1)


def _fraction_to_boundary(value: Array, direction: Array, step_fraction: float) -> Array:
    candidates = jnp.where(direction < 0, -value / direction, jnp.inf)
    return jnp.minimum(
        jnp.asarray(1.0, dtype=value.dtype),
        step_fraction * jnp.min(candidates, initial=jnp.inf),
    )


def _solve_saddle(
    matrix: Array, constraint: Array, primal_rhs: Array, constraint_rhs: Array
) -> tuple[Array, Array]:
    variables = matrix.shape[0]
    equalities = constraint.shape[0]
    zeros = jnp.zeros((equalities, equalities), dtype=matrix.dtype)
    kkt = jnp.block([[matrix, constraint.T], [constraint, zeros]])
    solution = solve_linear(
        LeastSquaresProblem(DenseLinearOperator(kkt)),
        jnp.concatenate((primal_rhs, constraint_rhs)),
        policy=LinearSolvePolicy(DenseSVD()),
    ).value
    return solution[:variables], solution[variables:]


def _newton_direction(
    quadratic: Array,
    equality_matrix: Array,
    inequality_matrix: Array,
    slack: Array,
    inequality_dual: Array,
    dual_residual: Array,
    equality_residual: Array,
    inequality_residual: Array,
    centering_residual: Array,
    regularization: float,
) -> tuple[Array, Array, Array, Array]:
    variables = quadratic.shape[0]
    equalities = equality_matrix.shape[0]
    inequalities = inequality_matrix.shape[0]
    dtype = quadratic.dtype
    zeros_variables_inequalities = jnp.zeros((variables, inequalities), dtype=dtype)
    zeros_equalities = jnp.zeros((equalities, equalities), dtype=dtype)
    zeros_equalities_inequalities = jnp.zeros((equalities, inequalities), dtype=dtype)
    zeros_inequalities_equalities = jnp.zeros((inequalities, equalities), dtype=dtype)
    zeros_inequalities = jnp.zeros((inequalities, inequalities), dtype=dtype)
    zeros_inequalities_variables = jnp.zeros((inequalities, variables), dtype=dtype)
    regularized = quadratic + regularization * jnp.eye(variables, dtype=dtype)
    kkt = jnp.block(
        [
            [
                regularized,
                equality_matrix.T,
                inequality_matrix.T,
                zeros_variables_inequalities,
            ],
            [
                equality_matrix,
                zeros_equalities,
                zeros_equalities_inequalities,
                zeros_equalities_inequalities,
            ],
            [
                inequality_matrix,
                zeros_inequalities_equalities,
                zeros_inequalities,
                jnp.eye(inequalities, dtype=dtype),
            ],
            [
                zeros_inequalities_variables,
                zeros_inequalities_equalities,
                jnp.diag(slack),
                jnp.diag(inequality_dual),
            ],
        ]
    )
    solution = solve_linear(
        LinearSystem(DenseLinearOperator(kkt)),
        -jnp.concatenate(
            (
                dual_residual,
                equality_residual,
                inequality_residual,
                centering_residual,
            )
        ),
        policy=LinearSolvePolicy(DenseLU()),
    ).value
    equality_end = variables + equalities
    inequality_end = equality_end + inequalities
    primal_direction = solution[:variables]
    equality_dual_direction = solution[variables:equality_end]
    inequality_dual_direction = solution[equality_end:inequality_end]
    slack_direction = solution[inequality_end:]
    return (
        primal_direction,
        slack_direction,
        inequality_dual_direction,
        equality_dual_direction,
    )


def _farkas_certificate_single(
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    equality_dual: Array,
    inequality_dual: Array,
    tolerance: float,
) -> Array:
    stationarity = (
        equality_matrix.T @ equality_dual + inequality_matrix.T @ inequality_dual
    )
    scale = jnp.maximum(
        1.0,
        jnp.maximum(_max_abs(equality_dual), _max_abs(inequality_dual)),
    )
    certificate_tolerance = jnp.sqrt(
        jnp.asarray(tolerance, dtype=inequality_matrix.dtype)
    )
    value = (equality_rhs @ equality_dual + inequality_rhs @ inequality_dual) / scale
    return (
        (_max_abs(stationarity) / scale <= certificate_tolerance)
        & (value < -certificate_tolerance)
        & (_min_value(inequality_dual) >= -certificate_tolerance)
        & jnp.isfinite(value)
    )


def _dense_constrained_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    initial_primal: Array,
    initial_slack: Array,
    initial_inequality_dual: Array,
    initial_equality_dual: Array,
    *,
    use_warm_start: bool,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    variables = quadratic.shape[0]
    inequalities = inequality_matrix.shape[0]
    equality_candidate = (
        solve_linear(
            LeastSquaresProblem(DenseLinearOperator(equality_matrix)),
            equality_rhs,
            policy=LinearSolvePolicy(DenseSVD()),
        ).value
        if equality_matrix.shape[0]
        else jnp.zeros((variables,), dtype=quadratic.dtype)
    )
    default_primal = jnp.where(
        equality_matrix.shape[0] == 0,
        jnp.zeros((variables,), dtype=quadratic.dtype),
        equality_candidate,
    )
    raw_slack = inequality_rhs - inequality_matrix @ default_primal
    default_slack = jnp.where(raw_slack > 0, raw_slack, jnp.ones_like(raw_slack))
    warm = jnp.asarray(use_warm_start)
    primal = jnp.where(warm, initial_primal, default_primal)
    slack = jnp.where(warm, initial_slack, default_slack)
    inequality_dual = jnp.where(
        warm,
        initial_inequality_dual,
        jnp.ones((inequalities,), dtype=quadratic.dtype),
    )
    equality_dual = jnp.where(
        warm,
        initial_equality_dual,
        jnp.zeros((equality_matrix.shape[0],), dtype=quadratic.dtype),
    )
    initial = (
        primal,
        slack,
        inequality_dual,
        equality_dual,
        jnp.asarray(0, dtype=jnp.int32),
    )

    def residuals(primal, slack, inequality_dual, equality_dual):
        dual_residual = (
            quadratic @ primal
            + regularization * primal
            + linear
            + equality_matrix.T @ equality_dual
            + inequality_matrix.T @ inequality_dual
        )
        equality_residual = equality_matrix @ primal - equality_rhs
        inequality_residual = inequality_matrix @ primal + slack - inequality_rhs
        complementarity = slack * inequality_dual
        residual_norm = jnp.maximum(
            jnp.maximum(_max_abs(dual_residual), _max_abs(equality_residual)),
            jnp.maximum(_max_abs(inequality_residual), _max_abs(complementarity)),
        )
        return (
            dual_residual,
            equality_residual,
            inequality_residual,
            complementarity,
            residual_norm,
        )

    def condition(state):
        primal, slack, inequality_dual, equality_dual, iterations = state
        *_, residual_norm = residuals(primal, slack, inequality_dual, equality_dual)
        infeasible = _farkas_certificate_single(
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            equality_dual,
            inequality_dual,
            tolerance,
        )
        return (iterations < max_iterations) & (residual_norm > tolerance) & ~infeasible

    def body(state):
        primal, slack, inequality_dual, equality_dual, iterations = state
        (
            dual_residual,
            equality_residual,
            inequality_residual,
            complementarity,
            _,
        ) = residuals(primal, slack, inequality_dual, equality_dual)
        affine = _newton_direction(
            quadratic,
            equality_matrix,
            inequality_matrix,
            slack,
            inequality_dual,
            dual_residual,
            equality_residual,
            inequality_residual,
            complementarity,
            regularization,
        )
        _, slack_affine, dual_affine, _ = affine
        alpha_primal_affine = _fraction_to_boundary(slack, slack_affine, 1.0)
        alpha_dual_affine = _fraction_to_boundary(inequality_dual, dual_affine, 1.0)
        mean_complementarity = jnp.mean(complementarity)
        affine_complementarity = jnp.mean(
            (slack + alpha_primal_affine * slack_affine)
            * (inequality_dual + alpha_dual_affine * dual_affine)
        )
        centering = jnp.where(
            mean_complementarity > 0,
            (affine_complementarity / mean_complementarity) ** 3,
            jnp.asarray(0.0, dtype=quadratic.dtype),
        )
        corrected_complementarity = (
            complementarity
            + slack_affine * dual_affine
            - centering * mean_complementarity
        )
        direction = _newton_direction(
            quadratic,
            equality_matrix,
            inequality_matrix,
            slack,
            inequality_dual,
            dual_residual,
            equality_residual,
            inequality_residual,
            corrected_complementarity,
            regularization,
        )
        primal_direction, slack_direction, dual_direction, equality_direction = direction
        alpha_primal = _fraction_to_boundary(slack, slack_direction, step_fraction)
        alpha_dual = _fraction_to_boundary(inequality_dual, dual_direction, step_fraction)
        return (
            primal + alpha_primal * primal_direction,
            slack + alpha_primal * slack_direction,
            inequality_dual + alpha_dual * dual_direction,
            equality_dual + alpha_dual * equality_direction,
            iterations + 1,
        )

    final = jax.lax.while_loop(condition, body, initial)
    primal, slack, inequality_dual, equality_dual, iterations = final
    *_, final_residual_norm = residuals(primal, slack, inequality_dual, equality_dual)
    return (
        primal,
        slack,
        inequality_dual,
        equality_dual,
        final_residual_norm <= tolerance,
        iterations,
    )


def _dense_finite_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    initial_primal: Array,
    initial_slack: Array,
    initial_inequality_dual: Array,
    initial_equality_dual: Array,
    *,
    use_warm_start: bool,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    variables = quadratic.shape[0]
    inequalities = inequality_matrix.shape[0]
    regularized = quadratic + regularization * jnp.eye(variables, dtype=quadratic.dtype)
    if inequalities == 0:
        primal, equality_dual = _solve_saddle(
            regularized,
            equality_matrix,
            -linear,
            equality_rhs,
        )
        direct_residual_norm = jnp.maximum(
            _max_abs(regularized @ primal + linear + equality_matrix.T @ equality_dual),
            _max_abs(equality_matrix @ primal - equality_rhs),
        )
        return (
            primal,
            jnp.empty((0,), dtype=quadratic.dtype),
            jnp.empty((0,), dtype=quadratic.dtype),
            equality_dual,
            jnp.isfinite(direct_residual_norm) & (direct_residual_norm <= tolerance),
            jnp.asarray(0, dtype=jnp.int32),
        )
    return _dense_constrained_single(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        initial_primal,
        initial_slack,
        initial_inequality_dual,
        initial_equality_dual,
        tolerance=tolerance,
        use_warm_start=use_warm_start,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
    )


def _nonfinite_single(
    quadratic: Array,
    equality_matrix: Array,
    inequality_matrix: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    dtype = quadratic.dtype
    return (
        jnp.full((quadratic.shape[0],), jnp.nan, dtype=dtype),
        jnp.full((inequality_matrix.shape[0],), jnp.nan, dtype=dtype),
        jnp.full((inequality_matrix.shape[0],), jnp.nan, dtype=dtype),
        jnp.full((equality_matrix.shape[0],), jnp.nan, dtype=dtype),
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
    )


def _dense_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    initial_primal: Array,
    initial_slack: Array,
    initial_inequality_dual: Array,
    initial_equality_dual: Array,
    *,
    use_warm_start: bool,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    input_finite = jnp.all(
        jnp.stack(
            (
                jnp.all(jnp.isfinite(quadratic)),
                jnp.all(jnp.isfinite(linear)),
                jnp.all(jnp.isfinite(equality_matrix)),
                jnp.all(jnp.isfinite(equality_rhs)),
                jnp.all(jnp.isfinite(inequality_matrix)),
                jnp.all(jnp.isfinite(inequality_rhs)),
            )
        )
    )
    return jax.lax.cond(
        input_finite,
        lambda _: _dense_finite_single(
            quadratic,
            linear,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            initial_primal,
            initial_slack,
            initial_inequality_dual,
            initial_equality_dual,
            tolerance=tolerance,
            use_warm_start=use_warm_start,
            max_iterations=max_iterations,
            regularization=regularization,
            step_fraction=step_fraction,
        ),
        lambda _: _nonfinite_single(quadratic, equality_matrix, inequality_matrix),
        operand=None,
    )


def _flatten_problem(
    problem: QuadraticProgram, /
) -> tuple[Array, Array, Array, Array, Array, Array]:
    count = prod(problem.batch_shape or (1,))
    return (
        problem.quadratic.reshape((count, problem.num_variables, problem.num_variables)),
        problem.linear.reshape((count, problem.num_variables)),
        problem.equality_matrix.reshape(
            (count, problem.num_equalities, problem.num_variables)
        ),
        problem.equality_rhs.reshape((count, problem.num_equalities)),
        problem.inequality_matrix.reshape(
            (count, problem.num_inequalities, problem.num_variables)
        ),
        problem.inequality_rhs.reshape((count, problem.num_inequalities)),
    )


def _warm_start_arrays(
    problem: QuadraticProgram,
    warm_start: ConvexWarmStart | None,
    /,
) -> tuple[Array | None, Array | None, Array | None, Array | None]:
    if warm_start is None:
        return None, None, None, None
    if not isinstance(warm_start, ConvexWarmStart):
        raise TypeError("warm_start must be a ConvexWarmStart or None.")
    if warm_start.structure_id != problem.structure_id:
        raise ValueError("Warm start does not match the quadratic-program structure.")
    expected = {
        "primal": problem.batch_shape + (problem.num_variables,),
        "equality_dual": problem.batch_shape + (problem.num_user_equalities,),
        "inequality_dual": problem.batch_shape + (problem.num_user_inequalities,),
        "inequality_slack": problem.batch_shape + (problem.num_user_inequalities,),
        "lower_bound_dual": problem.batch_shape + (problem.num_variables,),
        "upper_bound_dual": problem.batch_shape + (problem.num_variables,),
    }
    values = {
        "primal": warm_start.primal,
        "equality_dual": warm_start.equality_dual,
        "inequality_dual": warm_start.inequality_dual,
        "inequality_slack": warm_start.inequality_slack,
        "lower_bound_dual": warm_start.lower_bound_dual,
        "upper_bound_dual": warm_start.upper_bound_dual,
    }
    for name, value in values.items():
        if tuple(value.shape) != expected[name]:
            raise ValueError(
                f"Warm-start {name} must have shape {expected[name]}; got {value.shape}."
            )
    dtype = problem.linear.dtype
    primal = jnp.asarray(warm_start.primal, dtype=dtype)
    fixed_indices = jnp.asarray(problem.fixed_bound_indices, dtype=jnp.int32)
    lower_indices = jnp.asarray(problem.lower_bound_indices, dtype=jnp.int32)
    upper_indices = jnp.asarray(problem.upper_bound_indices, dtype=jnp.int32)
    fixed_dual = jnp.take(warm_start.upper_bound_dual, fixed_indices, axis=-1) - jnp.take(
        warm_start.lower_bound_dual, fixed_indices, axis=-1
    )
    equality_dual = jnp.concatenate(
        (jnp.asarray(warm_start.equality_dual, dtype=dtype), fixed_dual),
        axis=-1,
    )
    lower_dual = jnp.take(warm_start.lower_bound_dual, lower_indices, axis=-1)
    upper_dual = jnp.take(warm_start.upper_bound_dual, upper_indices, axis=-1)
    inequality_dual = jnp.concatenate(
        (
            jnp.asarray(warm_start.inequality_dual, dtype=dtype),
            lower_dual,
            upper_dual,
        ),
        axis=-1,
    )
    lower_slack = jnp.take(
        primal - problem.lower_bounds,
        lower_indices,
        axis=-1,
    )
    upper_slack = jnp.take(
        problem.upper_bounds - primal,
        upper_indices,
        axis=-1,
    )
    slack = jnp.concatenate(
        (
            jnp.asarray(warm_start.inequality_slack, dtype=dtype),
            lower_slack,
            upper_slack,
        ),
        axis=-1,
    )
    finite = (
        jnp.all(jnp.isfinite(primal))
        & jnp.all(jnp.isfinite(equality_dual))
        & jnp.all(jnp.isfinite(inequality_dual))
        & jnp.all(jnp.isfinite(slack))
    )
    interior = jnp.all(inequality_dual > 0.0) & jnp.all(slack > 0.0)
    primal = eqx.error_if(
        primal,
        ~(finite & interior),
        "Dense QP warm starts require finite data and strictly positive slacks/duals.",
    )
    count = prod(problem.batch_shape or (1,))
    return (
        primal.reshape((count, problem.num_variables)),
        slack.reshape((count, problem.num_inequalities)),
        inequality_dual.reshape((count, problem.num_inequalities)),
        equality_dual.reshape((count, problem.num_equalities)),
    )


# One compiled interior-point kernel per static layout (shapes, dtype, and
# solver configuration): repeated solves at fixed structure, such as MPC
# windows of one prediction topology, reuse it instead of recompiling.
@partial(
    jax.jit,
    static_argnames=(
        "use_warm_start",
        "tolerance",
        "max_iterations",
        "regularization",
        "step_fraction",
    ),
)
def _solve_dense_arrays(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    *,
    initial_primal: Array | None = None,
    initial_slack: Array | None = None,
    initial_inequality_dual: Array | None = None,
    initial_equality_dual: Array | None = None,
    use_warm_start: bool = False,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    count, variables = linear.shape
    inequalities = inequality_rhs.shape[-1]
    equalities = equality_rhs.shape[-1]
    initial_primal = (
        jnp.zeros((count, variables), dtype=linear.dtype)
        if initial_primal is None
        else initial_primal
    )
    initial_slack = (
        jnp.ones((count, inequalities), dtype=linear.dtype)
        if initial_slack is None
        else initial_slack
    )
    initial_inequality_dual = (
        jnp.ones((count, inequalities), dtype=linear.dtype)
        if initial_inequality_dual is None
        else initial_inequality_dual
    )
    initial_equality_dual = (
        jnp.zeros((count, equalities), dtype=linear.dtype)
        if initial_equality_dual is None
        else initial_equality_dual
    )
    solve_one = partial(
        _dense_single,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
        use_warm_start=use_warm_start,
    )
    return jax.lax.map(
        lambda arrays: solve_one(*arrays),
        (
            quadratic,
            linear,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            initial_primal,
            initial_slack,
            initial_inequality_dual,
            initial_equality_dual,
        ),
    )


# The independent audit (including its least-squares infeasibility candidates)
# is traced and compiled once per static program layout and configuration.
@eqx.filter_jit
def _diagnostics(
    problem: QuadraticProgram,
    primal: Array,
    slack: Array,
    inequality_dual: Array,
    equality_dual: Array,
    input_backend_converged: Array,
    iterations: Array,
    *,
    method: str,
    backend: str,
    tolerance: float,
    relative_tolerance: float,
    max_iterations: int,
    regularization: float,
    policy_id: str,
    primal_infeasible_tolerance: float,
    dual_infeasible_tolerance: float,
) -> ConvexProgramResult:
    from ._audit import audit_dual_infeasibility_ray, audit_primal_recession_ray

    quadratic = problem.quadratic
    linear = problem.linear
    equality_matrix = problem.equality_matrix
    equality_rhs = problem.equality_rhs
    inequality_matrix = problem.inequality_matrix
    inequality_rhs = problem.inequality_rhs
    objective = 0.5 * ein.contract(
        "...i,...ij,...j->...", primal, quadratic, primal
    ) + ein.contract("...i,...i->...", linear, primal)
    stationarity = (
        ein.contract("...ij,...j->...i", quadratic, primal)
        + linear
        + ein.contract("...ji,...j->...i", equality_matrix, equality_dual)
        + ein.contract("...ji,...j->...i", inequality_matrix, inequality_dual)
    )
    solver_stationarity = stationarity + regularization * primal
    equality_residual = (
        ein.contract("...ij,...j->...i", equality_matrix, primal) - equality_rhs
    )
    inequality_residual = (
        ein.contract("...ij,...j->...i", inequality_matrix, primal)
        + slack
        - inequality_rhs
    )
    inequality_violation = jnp.maximum(
        ein.contract("...ij,...j->...i", inequality_matrix, primal) - inequality_rhs,
        0.0,
    )
    complementarity = slack * inequality_dual
    equality_norm = _max_abs(equality_residual)
    inequality_norm = _max_abs(inequality_residual)
    violation_norm = _max_abs(inequality_violation)
    primal_norm = jnp.maximum(jnp.maximum(equality_norm, inequality_norm), violation_norm)
    dual_norm = _max_abs(stationarity)
    solver_dual_norm = _max_abs(solver_stationarity)
    complementarity_norm = _max_abs(complementarity)
    complementarity_gap = (
        jnp.sum(complementarity, axis=-1)
        if problem.num_inequalities
        else jnp.zeros(problem.batch_shape, dtype=quadratic.dtype)
    )
    kkt_norm = jnp.maximum(
        jnp.maximum(primal_norm, solver_dual_norm), complementarity_norm
    )
    optimality_scale = jnp.maximum(
        1.0,
        jnp.maximum(
            jnp.abs(objective),
            jnp.maximum(
                _max_abs(linear),
                jnp.maximum(_max_abs(equality_rhs), _max_abs(inequality_rhs)),
            ),
        ),
    )
    audit_tolerance = tolerance + relative_tolerance * optimality_scale
    input_finite = (
        jnp.all(jnp.isfinite(quadratic), axis=(-2, -1))
        & jnp.all(jnp.isfinite(linear), axis=-1)
        & jnp.all(jnp.isfinite(equality_matrix), axis=(-2, -1))
        & jnp.all(jnp.isfinite(equality_rhs), axis=-1)
        & jnp.all(jnp.isfinite(inequality_matrix), axis=(-2, -1))
        & jnp.all(jnp.isfinite(inequality_rhs), axis=-1)
    )
    output_finite = (
        jnp.all(jnp.isfinite(primal), axis=-1)
        & jnp.all(jnp.isfinite(slack), axis=-1)
        & jnp.all(jnp.isfinite(inequality_dual), axis=-1)
        & jnp.all(jnp.isfinite(equality_dual), axis=-1)
        & jnp.isfinite(kkt_norm)
    )
    nonnegative = (_min_value(slack) >= -audit_tolerance) & (
        _min_value(inequality_dual) >= -audit_tolerance
    )
    converged = output_finite & nonnegative & (kkt_norm <= audit_tolerance)

    user_equalities = problem.num_user_equalities
    user_inequalities = problem.num_user_inequalities
    fixed_dual = equality_dual[..., user_equalities:]
    lower_start = user_inequalities
    lower_stop = lower_start + len(problem.lower_bound_indices)
    upper_stop = lower_stop + len(problem.upper_bound_indices)
    lower_dual = inequality_dual[..., lower_start:lower_stop]
    upper_dual = inequality_dual[..., lower_stop:upper_stop]
    lower_bound_dual = jnp.zeros_like(primal)
    upper_bound_dual = jnp.zeros_like(primal)
    if problem.lower_bound_indices:
        lower_bound_dual = lower_bound_dual.at[
            ..., jnp.asarray(problem.lower_bound_indices, dtype=jnp.int32)
        ].set(lower_dual)
    if problem.upper_bound_indices:
        upper_bound_dual = upper_bound_dual.at[
            ..., jnp.asarray(problem.upper_bound_indices, dtype=jnp.int32)
        ].set(upper_dual)
    if problem.fixed_bound_indices:
        fixed_indices = jnp.asarray(problem.fixed_bound_indices, dtype=jnp.int32)
        lower_bound_dual = lower_bound_dual.at[..., fixed_indices].set(
            jnp.maximum(-fixed_dual, 0.0)
        )
        upper_bound_dual = upper_bound_dual.at[..., fixed_indices].set(
            jnp.maximum(fixed_dual, 0.0)
        )

    def equality_infeasibility_candidate(matrix: Array, rhs: Array) -> Array:
        if matrix.shape[0] == 0:
            return jnp.empty((0,), dtype=matrix.dtype)
        candidate = solve_linear(
            LeastSquaresProblem(DenseLinearOperator(matrix)),
            rhs,
            policy=LinearSolvePolicy(DenseSVD()),
        ).value
        return matrix @ candidate - rhs

    flat_count = prod(problem.batch_shape or (1,))
    equality_candidate = jax.vmap(equality_infeasibility_candidate)(
        equality_matrix.reshape(
            (flat_count, problem.num_equalities, problem.num_variables)
        ),
        equality_rhs.reshape((flat_count, problem.num_equalities)),
    ).reshape(problem.batch_shape + (problem.num_equalities,))
    candidate_lower = jnp.zeros_like(primal)
    candidate_upper = jnp.zeros_like(primal)
    if problem.fixed_bound_indices:
        fixed_indices = jnp.asarray(problem.fixed_bound_indices, dtype=jnp.int32)
        fixed_candidate = equality_candidate[..., user_equalities:]
        candidate_lower = candidate_lower.at[..., fixed_indices].set(
            jnp.maximum(-fixed_candidate, 0.0)
        )
        candidate_upper = candidate_upper.at[..., fixed_indices].set(
            jnp.maximum(fixed_candidate, 0.0)
        )
    equality_ray_audit = audit_dual_infeasibility_ray(
        problem,
        equality_candidate[..., :user_equalities],
        jnp.zeros(problem.batch_shape + (user_inequalities,), dtype=primal.dtype),
        candidate_lower,
        candidate_upper,
        tolerance=primal_infeasible_tolerance,
    )
    solver_ray_audit = audit_dual_infeasibility_ray(
        problem,
        equality_dual[..., :user_equalities],
        inequality_dual[..., :user_inequalities],
        lower_bound_dual,
        upper_bound_dual,
        tolerance=primal_infeasible_tolerance,
    )
    use_equality_ray = equality_ray_audit.valid
    dual_equality_ray = jnp.where(
        use_equality_ray[..., None],
        equality_ray_audit.equality_ray,
        solver_ray_audit.equality_ray,
    )
    dual_inequality_ray = jnp.where(
        use_equality_ray[..., None],
        equality_ray_audit.inequality_ray,
        solver_ray_audit.inequality_ray,
    )
    lower_bound_dual_ray = jnp.where(
        use_equality_ray[..., None],
        equality_ray_audit.lower_bound_ray,
        solver_ray_audit.lower_bound_ray,
    )
    upper_bound_dual_ray = jnp.where(
        use_equality_ray[..., None],
        equality_ray_audit.upper_bound_ray,
        solver_ray_audit.upper_bound_ray,
    )
    dual_ray_residual = jnp.where(
        use_equality_ray,
        equality_ray_audit.residual_norm,
        solver_ray_audit.residual_norm,
    )
    dual_ray_objective = jnp.where(
        use_equality_ray,
        equality_ray_audit.objective,
        solver_ray_audit.objective,
    )
    dual_ray_valid = equality_ray_audit.valid | solver_ray_audit.valid

    linear_ray_audit = audit_primal_recession_ray(
        problem,
        -linear,
        tolerance=dual_infeasible_tolerance,
    )
    iterate_ray_audit = audit_primal_recession_ray(
        problem,
        primal,
        tolerance=dual_infeasible_tolerance,
    )
    use_linear_ray = linear_ray_audit.valid
    primal_ray = jnp.where(
        use_linear_ray[..., None],
        linear_ray_audit.ray,
        iterate_ray_audit.ray,
    )
    primal_ray_residual = jnp.where(
        use_linear_ray,
        linear_ray_audit.residual_norm,
        iterate_ray_audit.residual_norm,
    )
    primal_ray_objective = jnp.where(
        use_linear_ray,
        linear_ray_audit.objective,
        iterate_ray_audit.objective,
    )
    primal_ray_valid = linear_ray_audit.valid | iterate_ray_audit.valid
    status = jnp.where(
        ~input_finite,
        int(ConvexProgramStatus.NONFINITE_INPUT),
        jnp.where(
            converged,
            int(ConvexProgramStatus.OPTIMAL),
            jnp.where(
                dual_ray_valid,
                int(ConvexProgramStatus.PRIMAL_INFEASIBLE),
                jnp.where(
                    primal_ray_valid,
                    int(ConvexProgramStatus.DUAL_INFEASIBLE),
                    jnp.where(
                        ~output_finite,
                        int(ConvexProgramStatus.NONFINITE_OUTPUT),
                        int(ConvexProgramStatus.ITERATION_LIMIT),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    valid = status == int(ConvexProgramStatus.OPTIMAL)
    certificate = ConvexProgramCertificate(
        primal_ray=primal_ray,
        equality_dual_ray=dual_equality_ray,
        inequality_dual_ray=dual_inequality_ray,
        lower_bound_dual_ray=lower_bound_dual_ray,
        upper_bound_dual_ray=upper_bound_dual_ray,
        primal_ray_residual_norm=primal_ray_residual,
        dual_ray_residual_norm=dual_ray_residual,
        primal_ray_objective=primal_ray_objective,
        dual_ray_objective=dual_ray_objective,
        primal_ray_valid=primal_ray_valid,
        dual_ray_valid=dual_ray_valid,
    )
    if backend == "phydrax":
        backend_name, backend_version = "phydrax", "native"
    else:
        backend_name, _, backend_version = backend.partition("-")
    provenance = ConvexProgramProvenance(
        numeric_version=0,
        problem_id=problem.problem_id,
        structure_id=problem.structure_id,
        policy_id=policy_id,
        method_id=method,
        backend=backend_name,
        backend_version=backend_version,
        convexity_evidence=problem.convexity_evidence,
        regularization=regularization,
    )
    return ConvexProgramResult(
        primal=primal,
        equality_dual=equality_dual[..., :user_equalities],
        cone_slack=jnp.concatenate(
            (
                jnp.zeros(problem.batch_shape + (user_equalities,), dtype=primal.dtype),
                slack[..., :user_inequalities],
            ),
            axis=-1,
        ),
        cone_dual=jnp.concatenate(
            (
                equality_dual[..., :user_equalities],
                inequality_dual[..., :user_inequalities],
            ),
            axis=-1,
        ),
        cone_primal_residual=jnp.concatenate(
            (
                equality_residual[..., :user_equalities],
                inequality_residual[..., :user_inequalities],
            ),
            axis=-1,
        ),
        cone_violation=jnp.concatenate(
            (
                jnp.abs(equality_residual[..., :user_equalities]),
                inequality_violation[..., :user_inequalities],
            ),
            axis=-1,
        ),
        cone_dual_violation=jnp.concatenate(
            (
                jnp.zeros(problem.batch_shape + (user_equalities,), dtype=primal.dtype),
                jnp.maximum(
                    -inequality_dual[..., :user_inequalities],
                    0.0,
                ),
            ),
            axis=-1,
        ),
        cone_complementarity=jnp.concatenate(
            (
                jnp.zeros(problem.batch_shape + (user_equalities,), dtype=primal.dtype),
                complementarity[..., :user_inequalities],
            ),
            axis=-1,
        ),
        inequality_dual=inequality_dual[..., :user_inequalities],
        inequality_slack=slack[..., :user_inequalities],
        lower_bound_dual=lower_bound_dual,
        upper_bound_dual=upper_bound_dual,
        objective=objective,
        stationarity_residual=stationarity,
        solver_stationarity_residual=solver_stationarity,
        equality_residual=equality_residual[..., :user_equalities],
        inequality_residual=inequality_residual[..., :user_inequalities],
        inequality_violation=inequality_violation[..., :user_inequalities],
        complementarity_residual=complementarity[..., :user_inequalities],
        primal_residual_norm=primal_norm,
        dual_residual_norm=dual_norm,
        solver_dual_residual_norm=solver_dual_norm,
        complementarity_gap=complementarity_gap,
        kkt_residual_norm=kkt_norm,
        iterations=iterations,
        backend_converged=jnp.asarray(input_backend_converged, dtype=jnp.bool_),
        valid=valid,
        status=status,
        certificate=certificate,
        provenance=provenance,
        batch_shape=problem.batch_shape,
        method=method,
        backend=backend,
        regularization=regularization,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _validate_solver_configuration(
    problem: QuadraticProgram,
    /,
    *,
    tolerance: float,
    max_iterations: int,
    regularization: float,
    step_fraction: float,
    max_dense_dimension: int,
) -> tuple[float, int, float, float, int]:
    if not isinstance(problem, QuadraticProgram):
        raise TypeError("problem must be a QuadraticProgram.")
    tolerance_value = float(tolerance)
    iterations_value = int(max_iterations)
    regularization_value = float(regularization)
    step_value = float(step_fraction)
    dense_limit = int(max_dense_dimension)
    if not np.isfinite(tolerance_value) or tolerance_value <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if iterations_value < 1:
        raise ValueError("max_iterations must be positive.")
    if not np.isfinite(regularization_value) or regularization_value < 0:
        raise ValueError("regularization must be finite and nonnegative.")
    if not np.isfinite(step_value) or not 0 < step_value < 1:
        raise ValueError(
            "step_fraction must be finite and strictly between zero and one."
        )
    if dense_limit < 1:
        raise ValueError("max_dense_dimension must be positive.")
    dimension = (
        problem.num_variables + problem.num_equalities + 2 * problem.num_inequalities
    )
    if dimension > dense_limit:
        raise ValueError(
            f"Dense QP dimension {dimension} exceeds max_dense_dimension={dense_limit}."
        )
    return (
        tolerance_value,
        iterations_value,
        regularization_value,
        step_value,
        dense_limit,
    )


def solve_quadratic_program(
    problem: QuadraticProgram,
    /,
    *,
    policy: ConvexSolvePolicy | None = None,
    warm_start: ConvexWarmStart | None = None,
) -> ConvexProgramResult:
    """Solve a convex QP through one explicit typed policy and independent audit."""

    selected = ConvexSolvePolicy() if policy is None else policy
    if not isinstance(selected, ConvexSolvePolicy):
        raise TypeError("policy must be a ConvexSolvePolicy or None.")
    method = selected.method
    if isinstance(method, (MPAXraPDHG, MPAXr2HPDHG)):
        _validate_quadratic_materialization(problem, selected)
        from ._mpax import solve_mpax_program

        result = solve_mpax_program(
            problem,
            selected,
            warm_start=warm_start,
        )
        return _apply_failure_policy(result, selected)
    if not isinstance(method, DensePrimalDualQP):
        raise TypeError(
            f"Method {type(method).__name__!r} does not solve QuadraticProgram."
        )
    method_id = method.method_id
    step_fraction = method.step_fraction
    max_dimension = method.max_kkt_dimension
    _validate_quadratic_resources(
        problem,
        selected,
        max_dense_dimension=max_dimension,
    )
    tolerance, maximum_steps, regularization, step_fraction, _ = (
        _validate_solver_configuration(
            problem,
            tolerance=selected.termination.absolute + selected.termination.relative,
            max_iterations=selected.termination.maximum_steps,
            regularization=selected.regularization,
            step_fraction=step_fraction,
            max_dense_dimension=max_dimension,
        )
    )
    arrays = _flatten_problem(problem)
    warm_arrays = _warm_start_arrays(problem, warm_start)
    primal, slack, inequality_dual, equality_dual, backend_converged, iterations = (
        _solve_dense_arrays(
            *arrays,
            initial_primal=warm_arrays[0],
            initial_slack=warm_arrays[1],
            initial_inequality_dual=warm_arrays[2],
            initial_equality_dual=warm_arrays[3],
            use_warm_start=warm_start is not None,
            tolerance=tolerance,
            max_iterations=maximum_steps,
            regularization=regularization,
            step_fraction=step_fraction,
        )
    )
    backend = method.backend
    primal = primal.reshape(problem.batch_shape + (problem.num_variables,))
    slack = slack.reshape(problem.batch_shape + (problem.num_inequalities,))
    inequality_dual = inequality_dual.reshape(
        problem.batch_shape + (problem.num_inequalities,)
    )
    equality_dual = equality_dual.reshape(problem.batch_shape + (problem.num_equalities,))
    result = _diagnostics(
        problem,
        primal,
        slack,
        inequality_dual,
        equality_dual,
        backend_converged.reshape(problem.batch_shape),
        iterations.reshape(problem.batch_shape),
        method=method_id,
        backend=backend,
        tolerance=selected.termination.absolute,
        relative_tolerance=selected.termination.relative,
        max_iterations=maximum_steps,
        regularization=regularization,
        policy_id=selected.policy_id,
        primal_infeasible_tolerance=selected.termination.primal_infeasible,
        dual_infeasible_tolerance=selected.termination.dual_infeasible,
    )
    return _apply_failure_policy(result, selected)


def _barrier_residuals(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    primal: Array,
    slack: Array,
    inequality_dual: Array,
    equality_dual: Array,
    /,
    *,
    barrier: float,
    regularization: float,
) -> tuple[Array, Array, Array, Array]:
    stationarity = (
        quadratic @ primal
        + regularization * primal
        + linear
        + equality_matrix.T @ equality_dual
        + inequality_matrix.T @ inequality_dual
    )
    equality_residual = equality_matrix @ primal - equality_rhs
    inequality_residual = inequality_matrix @ primal + slack - inequality_rhs
    complementarity = slack * inequality_dual - barrier
    return (
        stationarity,
        equality_residual,
        inequality_residual,
        complementarity,
    )


def _barrier_residual_norm(residuals: tuple[Array, ...], /) -> Array:
    value = jnp.asarray(0.0, dtype=residuals[0].dtype)
    for residual in residuals:
        value = jnp.maximum(value, _max_abs(residual))
    return value


def _center_barrier_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    primal: Array,
    slack: Array,
    inequality_dual: Array,
    equality_dual: Array,
    /,
    *,
    barrier: float,
    regularization: float,
    step_fraction: float,
    tolerance: float,
    maximum_steps: int,
) -> tuple[Array, Array, Array, Array, Array]:
    interior = jnp.sqrt(jnp.asarray(barrier, dtype=quadratic.dtype))
    slack = jnp.maximum(slack, interior)
    inequality_dual = jnp.maximum(inequality_dual, interior)

    def body(_, carry):
        primal_, slack_, inequality_dual_, equality_dual_, completed = carry
        residuals = _barrier_residuals(
            quadratic,
            linear,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            primal_,
            slack_,
            inequality_dual_,
            equality_dual_,
            barrier=barrier,
            regularization=regularization,
        )
        direction = _newton_direction(
            quadratic,
            equality_matrix,
            inequality_matrix,
            slack_,
            inequality_dual_,
            residuals[0],
            residuals[1],
            residuals[2],
            residuals[3],
            regularization,
        )
        primal_step, slack_step, inequality_dual_step, equality_dual_step = direction
        step = jnp.minimum(
            _fraction_to_boundary(slack_, slack_step, step_fraction),
            _fraction_to_boundary(
                inequality_dual_,
                inequality_dual_step,
                step_fraction,
            ),
        )
        active = ~completed
        primal_ = jnp.where(active, primal_ + step * primal_step, primal_)
        slack_ = jnp.where(active, slack_ + step * slack_step, slack_)
        inequality_dual_ = jnp.where(
            active,
            inequality_dual_ + step * inequality_dual_step,
            inequality_dual_,
        )
        equality_dual_ = jnp.where(
            active,
            equality_dual_ + step * equality_dual_step,
            equality_dual_,
        )
        next_residuals = _barrier_residuals(
            quadratic,
            linear,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
            primal_,
            slack_,
            inequality_dual_,
            equality_dual_,
            barrier=barrier,
            regularization=regularization,
        )
        completed = completed | (_barrier_residual_norm(next_residuals) <= tolerance)
        return primal_, slack_, inequality_dual_, equality_dual_, completed

    initial_residuals = _barrier_residuals(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        primal,
        slack,
        inequality_dual,
        equality_dual,
        barrier=barrier,
        regularization=regularization,
    )
    initial_completed = _barrier_residual_norm(initial_residuals) <= tolerance
    state = jax.lax.fori_loop(
        0,
        maximum_steps,
        body,
        (
            primal,
            slack,
            inequality_dual,
            equality_dual,
            initial_completed,
        ),
    )
    primal, slack, inequality_dual, equality_dual, completed = state
    finite = (
        jnp.all(jnp.isfinite(primal))
        & jnp.all(jnp.isfinite(slack))
        & jnp.all(jnp.isfinite(inequality_dual))
        & jnp.all(jnp.isfinite(equality_dual))
    )
    interior_valid = (_min_value(slack) > 0.0) & (_min_value(inequality_dual) > 0.0)
    return (
        primal,
        slack,
        inequality_dual,
        equality_dual,
        completed & finite & interior_valid,
    )


def _barrier_kkt(
    quadratic: Array,
    equality_matrix: Array,
    inequality_matrix: Array,
    slack: Array,
    inequality_dual: Array,
    /,
    *,
    regularization: float,
) -> Array:
    variables = quadratic.shape[0]
    equalities = equality_matrix.shape[0]
    inequalities = inequality_matrix.shape[0]
    dtype = quadratic.dtype
    return jnp.block(
        [
            [
                quadratic + regularization * jnp.eye(variables, dtype=dtype),
                equality_matrix.T,
                inequality_matrix.T,
                jnp.zeros((variables, inequalities), dtype=dtype),
            ],
            [
                equality_matrix,
                jnp.zeros((equalities, equalities), dtype=dtype),
                jnp.zeros((equalities, inequalities), dtype=dtype),
                jnp.zeros((equalities, inequalities), dtype=dtype),
            ],
            [
                inequality_matrix,
                jnp.zeros((inequalities, equalities), dtype=dtype),
                jnp.zeros((inequalities, inequalities), dtype=dtype),
                jnp.eye(inequalities, dtype=dtype),
            ],
            [
                jnp.zeros((inequalities, variables), dtype=dtype),
                jnp.zeros((inequalities, equalities), dtype=dtype),
                jnp.diag(slack),
                jnp.diag(inequality_dual),
            ],
        ]
    )


# Traced once per KKT shape: every prepared sensitivity of one layout (for
# example each MPC window of one topology) reuses the factorization program.
@jax.jit
def _regular_kkt_solve(matrix: Array, rhs: Array, /) -> tuple[Array, Array]:
    factorization = factorize(
        DenseLinearOperator(matrix),
        FactorizationPolicy(
            "svd",
            rank=RankPolicy(require_full_rank=True),
            differentiation=DifferentiationPolicy("rhs-only"),
        ),
    )
    result = factorization.solve(rhs)
    regular = (
        (factorization.rank() == matrix.shape[0])
        & jnp.all(jnp.isfinite(matrix))
        & jnp.all(jnp.isfinite(factorization.singular_values()))
    )
    return result.value, regular


def _barrier_tangent_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    quadratic_tangent: Array,
    linear_tangent: Array,
    equality_matrix_tangent: Array,
    equality_rhs_tangent: Array,
    inequality_matrix_tangent: Array,
    inequality_rhs_tangent: Array,
    primal: Array,
    slack: Array,
    inequality_dual: Array,
    equality_dual: Array,
    primal_valid: Array,
    /,
    *,
    barrier: float,
    regularization: float,
) -> Array:
    quadratic_tangent = 0.5 * (
        quadratic_tangent + jnp.swapaxes(quadratic_tangent, -1, -2)
    )

    def equation(q, c, a, b, g, h):
        return jnp.concatenate(
            _barrier_residuals(
                q,
                c,
                a,
                b,
                g,
                h,
                primal,
                slack,
                inequality_dual,
                equality_dual,
                barrier=barrier,
                regularization=regularization,
            )
        )

    _, forcing = jax.jvp(
        equation,
        (
            quadratic,
            linear,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
        ),
        (
            quadratic_tangent,
            linear_tangent,
            equality_matrix_tangent,
            equality_rhs_tangent,
            inequality_matrix_tangent,
            inequality_rhs_tangent,
        ),
    )
    kkt = _barrier_kkt(
        quadratic,
        equality_matrix,
        inequality_matrix,
        slack,
        inequality_dual,
        regularization=regularization,
    )
    safe_kkt = jnp.where(
        primal_valid,
        kkt,
        jnp.eye(kkt.shape[0], dtype=kkt.dtype),
    )
    safe_rhs = -forcing
    tangent, linear_valid = _regular_kkt_solve(safe_kkt, safe_rhs)
    valid = primal_valid & linear_valid
    tangent_scale = jnp.where(
        valid,
        jnp.asarray(1.0, dtype=primal.dtype),
        jnp.asarray(jnp.nan, dtype=primal.dtype),
    )
    return tangent_scale * tangent[: primal.shape[0]]


@partial(jax.custom_jvp, nondiff_argnums=(6, 7, 8, 9, 10, 11, 12))
def _barrier_primal_implicit(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    max_iterations: int,
    tolerance: float,
    regularization: float,
    step_fraction: float,
    barrier: float,
    centering_tolerance: float,
    maximum_centering_steps: int,
) -> Array:
    primal, slack, inequality_dual, equality_dual, _, _ = _solve_dense_arrays(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
    )
    center = partial(
        _center_barrier_single,
        barrier=barrier,
        regularization=regularization,
        step_fraction=step_fraction,
        tolerance=centering_tolerance,
        maximum_steps=maximum_centering_steps,
    )
    primal, _, _, _, _ = jax.vmap(center)(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        primal,
        slack,
        inequality_dual,
        equality_dual,
    )
    return primal


@_barrier_primal_implicit.defjvp
def _barrier_primal_implicit_jvp(
    max_iterations,
    tolerance,
    regularization,
    step_fraction,
    barrier,
    centering_tolerance,
    maximum_centering_steps,
    primals,
    tangents,
):
    (
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
    ) = primals
    (
        quadratic_tangent,
        linear_tangent,
        equality_matrix_tangent,
        equality_rhs_tangent,
        inequality_matrix_tangent,
        inequality_rhs_tangent,
    ) = tangents
    primal, slack, inequality_dual, equality_dual, _, _ = _solve_dense_arrays(
        *primals,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
    )
    center = partial(
        _center_barrier_single,
        barrier=barrier,
        regularization=regularization,
        step_fraction=step_fraction,
        tolerance=centering_tolerance,
        maximum_steps=maximum_centering_steps,
    )
    primal, slack, inequality_dual, equality_dual, valid = jax.vmap(center)(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        primal,
        slack,
        inequality_dual,
        equality_dual,
    )
    tangent = jax.vmap(
        partial(
            _barrier_tangent_single,
            barrier=barrier,
            regularization=regularization,
        )
    )(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        quadratic_tangent,
        linear_tangent,
        equality_matrix_tangent,
        equality_rhs_tangent,
        inequality_matrix_tangent,
        inequality_rhs_tangent,
        primal,
        slack,
        inequality_dual,
        equality_dual,
        valid,
    )
    return primal, tangent


def _active_set_kkt(
    quadratic: Array,
    equality_matrix: Array,
    inequality_matrix: Array,
    active: Array,
    /,
    *,
    regularization: float,
) -> Array:
    variables = quadratic.shape[0]
    equalities = equality_matrix.shape[0]
    active_scalar = active.astype(quadratic.dtype)
    active_matrix = active_scalar[:, None] * inequality_matrix
    constraint = jnp.concatenate((equality_matrix, active_matrix), axis=0)
    inactive_diagonal = jnp.concatenate(
        (
            jnp.zeros((equalities,), dtype=quadratic.dtype),
            1.0 - active_scalar,
        )
    )
    regularized = quadratic + regularization * jnp.eye(
        variables,
        dtype=quadratic.dtype,
    )
    return jnp.block(
        [
            [regularized, constraint.T],
            [constraint, -jnp.diag(inactive_diagonal)],
        ]
    )


def _active_set_equation(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    primal: Array,
    equality_dual: Array,
    inequality_dual: Array,
    active: Array,
    /,
    *,
    regularization: float,
) -> Array:
    active_dual = jnp.where(active, inequality_dual, 0.0)
    stationarity = (
        quadratic @ primal
        + regularization * primal
        + linear
        + equality_matrix.T @ equality_dual
        + inequality_matrix.T @ active_dual
    )
    equality = equality_matrix @ primal - equality_rhs
    inequality = jnp.where(
        active,
        inequality_matrix @ primal - inequality_rhs,
        inequality_dual,
    )
    return jnp.concatenate((stationarity, equality, inequality))


def _active_set_tangent_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    quadratic_tangent: Array,
    linear_tangent: Array,
    equality_matrix_tangent: Array,
    equality_rhs_tangent: Array,
    inequality_matrix_tangent: Array,
    inequality_rhs_tangent: Array,
    primal: Array,
    equality_dual: Array,
    inequality_dual: Array,
    slack: Array,
    primal_valid: Array,
    /,
    *,
    regularization: float,
    active_tolerance: float,
    strict_complementarity_tolerance: float,
) -> Array:
    active = (slack <= active_tolerance) & (
        inequality_dual >= strict_complementarity_tolerance
    )
    inactive = (slack >= strict_complementarity_tolerance) & (
        inequality_dual <= active_tolerance
    )
    unambiguous = jnp.all(active | inactive)
    quadratic_tangent = 0.5 * (
        quadratic_tangent + jnp.swapaxes(quadratic_tangent, -1, -2)
    )

    def equation(q, c, a, b, g, h):
        return _active_set_equation(
            q,
            c,
            a,
            b,
            g,
            h,
            primal,
            equality_dual,
            inequality_dual,
            active,
            regularization=regularization,
        )

    _, forcing = jax.jvp(
        equation,
        (
            quadratic,
            linear,
            equality_matrix,
            equality_rhs,
            inequality_matrix,
            inequality_rhs,
        ),
        (
            quadratic_tangent,
            linear_tangent,
            equality_matrix_tangent,
            equality_rhs_tangent,
            inequality_matrix_tangent,
            inequality_rhs_tangent,
        ),
    )
    kkt = _active_set_kkt(
        quadratic,
        equality_matrix,
        inequality_matrix,
        active,
        regularization=regularization,
    )
    eligible = primal_valid & unambiguous
    safe_kkt = jnp.where(eligible, kkt, jnp.eye(kkt.shape[0], dtype=kkt.dtype))
    safe_rhs = -forcing
    tangent, linear_valid = _regular_kkt_solve(safe_kkt, safe_rhs)
    valid = eligible & linear_valid
    tangent_scale = jnp.where(
        valid,
        jnp.asarray(1.0, dtype=primal.dtype),
        jnp.asarray(jnp.nan, dtype=primal.dtype),
    )
    return tangent_scale * tangent[: primal.shape[0]]


def _active_set_primal_valid_single(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    primal: Array,
    slack: Array,
    inequality_dual: Array,
    equality_dual: Array,
    backend_converged: Array,
    /,
    *,
    tolerance: float,
    regularization: float,
) -> Array:
    stationarity = (
        quadratic @ primal
        + regularization * primal
        + linear
        + equality_matrix.T @ equality_dual
        + inequality_matrix.T @ inequality_dual
    )
    equality = equality_matrix @ primal - equality_rhs
    inequality = inequality_matrix @ primal + slack - inequality_rhs
    complementarity = slack * inequality_dual
    residual = jnp.maximum(
        jnp.maximum(_max_abs(stationarity), _max_abs(equality)),
        jnp.maximum(_max_abs(inequality), _max_abs(complementarity)),
    )
    return (
        backend_converged
        & jnp.all(jnp.isfinite(primal))
        & jnp.all(jnp.isfinite(slack))
        & jnp.all(jnp.isfinite(inequality_dual))
        & jnp.all(jnp.isfinite(equality_dual))
        & (_min_value(slack) >= -tolerance)
        & (_min_value(inequality_dual) >= -tolerance)
        & (residual <= tolerance)
    )


@partial(jax.custom_jvp, nondiff_argnums=(6, 7, 8, 9, 10, 11))
def _dense_primal_implicit(
    quadratic: Array,
    linear: Array,
    equality_matrix: Array,
    equality_rhs: Array,
    inequality_matrix: Array,
    inequality_rhs: Array,
    max_iterations: int,
    tolerance: float,
    regularization: float,
    step_fraction: float,
    active_set_tolerance: float,
    strict_complementarity_tolerance: float,
) -> Array:
    del active_set_tolerance, strict_complementarity_tolerance
    primal, _, _, _, _, _ = _solve_dense_arrays(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
    )
    return primal


@_dense_primal_implicit.defjvp
def _dense_primal_implicit_jvp(
    max_iterations,
    tolerance,
    regularization,
    step_fraction,
    active_set_tolerance,
    strict_complementarity_tolerance,
    primals,
    tangents,
):
    (
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
    ) = primals
    (
        quadratic_tangent,
        linear_tangent,
        equality_matrix_tangent,
        equality_rhs_tangent,
        inequality_matrix_tangent,
        inequality_rhs_tangent,
    ) = tangents
    primal, slack, inequality_dual, equality_dual, converged, _ = _solve_dense_arrays(
        *primals,
        tolerance=tolerance,
        max_iterations=max_iterations,
        regularization=regularization,
        step_fraction=step_fraction,
    )
    valid = jax.vmap(
        partial(
            _active_set_primal_valid_single,
            tolerance=tolerance,
            regularization=regularization,
        )
    )(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        primal,
        slack,
        inequality_dual,
        equality_dual,
        converged,
    )
    tangent = jax.vmap(
        partial(
            _active_set_tangent_single,
            regularization=regularization,
            active_tolerance=active_set_tolerance,
            strict_complementarity_tolerance=strict_complementarity_tolerance,
        )
    )(
        quadratic,
        linear,
        equality_matrix,
        equality_rhs,
        inequality_matrix,
        inequality_rhs,
        quadratic_tangent,
        linear_tangent,
        equality_matrix_tangent,
        equality_rhs_tangent,
        inequality_matrix_tangent,
        inequality_rhs_tangent,
        primal,
        equality_dual,
        inequality_dual,
        slack,
        valid,
    )
    return primal, tangent


def solve_quadratic_program_primal(
    problem: QuadraticProgram,
    /,
    *,
    policy: ConvexSolvePolicy | None = None,
    differentiation: ConvexDifferentiationPolicy | None = None,
) -> Array:
    """Return one explicitly differentiated regular convex-QP solution map."""

    selected = ConvexSolvePolicy() if policy is None else policy
    if not isinstance(selected, ConvexSolvePolicy):
        raise TypeError("policy must be a ConvexSolvePolicy or None.")
    method = selected.method
    derivative = (
        ConvexDifferentiationPolicy() if differentiation is None else differentiation
    )
    if not isinstance(derivative, ConvexDifferentiationPolicy):
        raise TypeError("differentiation must be a ConvexDifferentiationPolicy or None.")
    if isinstance(method, MPAXraPDHG):
        _validate_quadratic_materialization(problem, selected)
        if derivative.mode != "algorithmic" or not method.plan.unroll:
            raise ValueError(
                "MPAX QP differentiation requires an unrolled method and ConvexDifferentiationPolicy('algorithmic')."
            )
        from ._mpax import solve_mpax_program

        return solve_mpax_program(problem, selected).primal
    if isinstance(method, MPAXr2HPDHG):
        raise TypeError("MPAXr2HPDHG supports LinearProgram only.")
    if derivative.mode == "algorithmic":
        raise ValueError(
            "Dense primal-dual QP does not expose algorithmic differentiation."
        )
    if derivative.mode == "none":
        raise ValueError("Use solve_quadratic_program when differentiation is disabled.")
    if not isinstance(method, DensePrimalDualQP):
        raise TypeError(
            f"Method {type(method).__name__!r} does not solve QuadraticProgram."
        )
    step_fraction = method.step_fraction
    max_dimension = method.max_kkt_dimension
    _validate_quadratic_resources(
        problem,
        selected,
        max_dense_dimension=max_dimension,
    )
    tolerance, maximum_steps, regularization, step_fraction, _ = (
        _validate_solver_configuration(
            problem,
            tolerance=selected.termination.absolute + selected.termination.relative,
            max_iterations=selected.termination.maximum_steps,
            regularization=selected.regularization,
            step_fraction=step_fraction,
            max_dense_dimension=max_dimension,
        )
    )
    arrays = _flatten_problem(problem)
    if derivative.mode == "active-set-kkt":
        primal = _dense_primal_implicit(
            *arrays,
            maximum_steps,
            tolerance,
            regularization,
            step_fraction,
            derivative.active_tolerance,
            derivative.strict_complementarity_tolerance,
        )
    else:
        assert derivative.mode == "barrier-kkt"
        assert derivative.barrier is not None
        primal = _barrier_primal_implicit(
            *arrays,
            maximum_steps,
            tolerance,
            regularization,
            step_fraction,
            derivative.barrier,
            derivative.centering_tolerance,
            derivative.maximum_centering_steps,
        )
    return primal.reshape(problem.batch_shape + (problem.num_variables,))


def prepare_qp_sensitivity(
    problem: QuadraticProgram,
    /,
    *,
    policy: ConvexSolvePolicy | None = None,
    differentiation: ConvexDifferentiationPolicy | None = None,
) -> PreparedQPSensitivity:
    """Prepare reusable primal pullback and transposed-pullback actions."""
    if not isinstance(problem, QuadraticProgram):
        raise TypeError("problem must be a QuadraticProgram.")
    derivative = (
        ConvexDifferentiationPolicy() if differentiation is None else differentiation
    )
    if not isinstance(derivative, ConvexDifferentiationPolicy):
        raise TypeError("differentiation must be a ConvexDifferentiationPolicy or None.")

    def solution(candidate):
        return solve_quadratic_program_primal(
            candidate,
            policy=policy,
            differentiation=derivative,
        )

    linearization = prepare_linearization(
        solution,
        problem,
        linearization_id="quadratic-program-sensitivity",
    )
    primal = linearization.primal

    def pushforward(tangent):
        return linearization.jvp(tangent)

    def pullback(cotangent):
        return linearization.vjp(cotangent)

    zero_tangent = jax.tree.map(
        lambda value: jnp.zeros_like(value) if eqx.is_inexact_array(value) else value,
        problem,
    )
    regular = jnp.all(jnp.isfinite(pushforward(zero_tangent)), axis=-1)

    return PreparedQPSensitivity(
        primal,
        pushforward,
        pullback,
        regular,
        derivative,
    )


__all__ = [
    "ConvexProgramResult",
    "PreparedQPSensitivity",
    "QuadraticProgram",
    "prepare_qp_sensitivity",
    "solve_quadratic_program",
    "solve_quadratic_program_primal",
]
