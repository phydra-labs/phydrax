#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, PyTree

from ..._strict import StrictModule
from ...linalg import (
    adjoint,
    ArraySpace,
    DenseSVD,
    DifferentiationPolicy,
    FailurePolicy,
    FunctionLinearOperator,
    LeastSquaresProblem,
    LinearSolveDiagnostics,
    LinearSolvePolicy,
    RankPolicy,
    solve as solve_linear,
)
from ._cones import AbstractConvexCone, NonnegativeCone, ProductCone, ZeroCone
from ._lifecycle import ConvexProgramExecution, PreparedConvexProgram
from ._problem import _conic_bound_indices, ConicProgram


class ConicProgramData(StrictModule):
    """Differentiable numerical coordinates of one fixed-topology conic program."""

    quadratic: Array | None
    linear: Array
    constraint_matrix: Array
    constraint_rhs: Array
    lower_bounds: Array
    upper_bounds: Array

    def __init__(
        self,
        quadratic: ArrayLike | None,
        linear: ArrayLike,
        constraint_matrix: ArrayLike,
        constraint_rhs: ArrayLike,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        /,
    ):
        self.quadratic = None if quadratic is None else jnp.asarray(quadratic)
        self.linear = jnp.asarray(linear)
        self.constraint_matrix = jnp.asarray(constraint_matrix)
        self.constraint_rhs = jnp.asarray(constraint_rhs)
        self.lower_bounds = jnp.asarray(lower_bounds)
        self.upper_bounds = jnp.asarray(upper_bounds)

    @classmethod
    def zeros_like(cls, program: ConicProgram, /) -> ConicProgramData:
        """Return the zero tangent in one conic program's numerical data space."""

        if not isinstance(program, ConicProgram):
            raise TypeError("program must be a ConicProgram.")
        quadratic = (
            None if program.quadratic is None else jnp.zeros_like(program.quadratic)
        )
        return cls(
            quadratic,
            jnp.zeros_like(program.linear),
            jnp.zeros_like(program.constraint_matrix),
            jnp.zeros_like(program.constraint_rhs),
            jnp.zeros_like(program.lower_bounds),
            jnp.zeros_like(program.upper_bounds),
        )


class PreparedConicSensitivity(StrictModule):
    """Audited numerical state for reusable projection-KKT sensitivities."""

    original_data: ConicProgramData
    quadratic: Array
    linear: Array
    constraint_matrix: Array
    constraint_rhs: Array
    state: Array
    cone: AbstractConvexCone
    forward_valid: Array
    projection_margin: Array
    projection_regular: Array
    root_residual_norm: Array
    lower_tangent_mask: Array
    upper_tangent_mask: Array
    linear_policy: LinearSolvePolicy
    numeric_version: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    num_cases: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_original_constraints: int = eqx.field(static=True)
    num_constraints: int = eqx.field(static=True)
    fixed_indices: tuple[int, ...] = eqx.field(static=True)
    lower_indices: tuple[int, ...] = eqx.field(static=True)
    upper_indices: tuple[int, ...] = eqx.field(static=True)
    quadratic_present: bool = eqx.field(static=True)
    regularity_tolerance: float = eqx.field(static=True)
    failure_mode: Literal["status", "error"] = eqx.field(static=True)
    convex_plan_id: str = eqx.field(static=True)
    numeric_binding_id: str = eqx.field(static=True)


class ConicSensitivityResult(StrictModule):
    """First-order value with forward, projection, and linear regularity evidence."""

    value: PyTree[Array]
    forward_valid: Array
    projection_margin: Array
    projection_regular: Array
    root_residual_norm: Array
    linear_status: Array
    linear_diagnostics: LinearSolveDiagnostics
    regular: Array
    numeric_version: Array
    convex_plan_id: str = eqx.field(static=True)
    linear_plan_id: str = eqx.field(static=True)
    numeric_binding_id: str = eqx.field(static=True)


def _max_abs(value: Array, /) -> Array:
    if value.shape[-1] == 0:
        return jnp.zeros(value.shape[:-1], dtype=value.dtype)
    return jnp.max(jnp.abs(value), axis=-1)


def _linear_policy(linear: LinearSolvePolicy | None, /) -> tuple[LinearSolvePolicy, str]:
    selected = LinearSolvePolicy(DenseSVD()) if linear is None else linear
    if not isinstance(selected, LinearSolvePolicy):
        raise TypeError("linear must be a LinearSolvePolicy or None.")
    if not isinstance(selected.method, DenseSVD):
        raise TypeError(
            "Conic sensitivity currently requires DenseSVD so full-rank and "
            "condition evidence are available."
        )
    if selected.method.damping != 0.0:
        raise ValueError(
            "Conic sensitivity requires zero derivative-solver damping; "
            "use ConvexSolvePolicy.regularization for the forward program."
        )
    failure_mode = selected.failure.mode
    return (
        LinearSolvePolicy(
            selected.method,
            tolerance=selected.tolerance,
            rank=RankPolicy(
                relative_cutoff=selected.rank.relative_cutoff,
                require_full_rank=True,
            ),
            materialization=selected.materialization,
            preconditioning=selected.preconditioning,
            recycling=selected.recycling,
            differentiation=DifferentiationPolicy("none"),
            failure=FailurePolicy("status"),
            resources=selected.resources,
            precision=selected.precision,
            require_device_binding=selected.require_device_binding,
        ),
        failure_mode,
    )


def _cone_blocks(cone: AbstractConvexCone, /) -> tuple[AbstractConvexCone, ...]:
    return cone.cones if isinstance(cone, ProductCone) else (cone,)


def _restore_cases(value: Array, batch_shape: tuple[int, ...], /) -> Array:
    return value.reshape(batch_shape + value.shape[1:])


def _restore_tree_cases(value: PyTree[Array], batch_shape: tuple[int, ...], /):
    return jax.tree.map(lambda leaf: _restore_cases(leaf, batch_shape), value)


def _mask_cases(value: Array, regular: Array, /) -> Array:
    mask = regular.reshape((regular.shape[0],) + (1,) * (value.ndim - 1))
    return jnp.where(mask, value, jnp.full_like(value, jnp.nan))


def _kkt_residual(
    state: Array,
    quadratic: Array,
    linear: Array,
    matrix: Array,
    rhs: Array,
    cone: AbstractConvexCone,
    num_variables: int,
    /,
) -> Array:
    primal = state[:num_variables]
    dual = state[num_variables:]
    projection_point = dual + matrix @ primal - rhs
    stationarity = quadratic @ primal + linear + jnp.conj(matrix.T) @ dual
    complementarity = dual - cone.project_dual(projection_point)
    return jnp.concatenate((stationarity, complementarity))


def _state_operator(
    quadratic: Array,
    linear: Array,
    matrix: Array,
    rhs: Array,
    state: Array,
    cone: AbstractConvexCone,
    num_variables: int,
    /,
) -> FunctionLinearOperator:
    space = ArraySpace(state.shape, dtype=state.dtype)

    def residual(candidate):
        return _kkt_residual(
            candidate,
            quadratic,
            linear,
            matrix,
            rhs,
            cone,
            num_variables,
        )

    def action(direction):
        return jax.jvp(residual, (state,), (direction,))[1]

    def transpose_action(cotangent):
        _, pullback = jax.vjp(residual, state)
        return pullback(cotangent)[0]

    return FunctionLinearOperator(
        action,
        source=space,
        target=space,
        transpose_action=transpose_action,
        operator_id="conic-projection-kkt-jacobian",
        closure_convert=False,
    )


def _data_residual(
    quadratic: Array,
    linear: Array,
    matrix: Array,
    rhs: Array,
    state: Array,
    cone: AbstractConvexCone,
    num_variables: int,
    /,
) -> Array:
    return _kkt_residual(
        state,
        quadratic,
        linear,
        matrix,
        rhs,
        cone,
        num_variables,
    )


def _jvp_case(
    quadratic: Array,
    linear: Array,
    matrix: Array,
    rhs: Array,
    state: Array,
    tangent_quadratic: Array,
    tangent_linear: Array,
    tangent_matrix: Array,
    tangent_rhs: Array,
    *,
    cone: AbstractConvexCone,
    num_variables: int,
    linear_policy: LinearSolvePolicy,
):
    operator = _state_operator(
        quadratic,
        linear,
        matrix,
        rhs,
        state,
        cone,
        num_variables,
    )
    _, data_action = jax.jvp(
        lambda p, q, a, b: _data_residual(
            p,
            q,
            a,
            b,
            state,
            cone,
            num_variables,
        ),
        (quadratic, linear, matrix, rhs),
        (tangent_quadratic, tangent_linear, tangent_matrix, tangent_rhs),
    )
    return solve_linear(
        LeastSquaresProblem(operator, problem_id="conic-projection-kkt-system"),
        -data_action,
        policy=linear_policy,
    )


def _vjp_case(
    quadratic: Array,
    linear: Array,
    matrix: Array,
    rhs: Array,
    state: Array,
    cotangent: Array,
    *,
    cone: AbstractConvexCone,
    num_variables: int,
    linear_policy: LinearSolvePolicy,
):
    operator = _state_operator(
        quadratic,
        linear,
        matrix,
        rhs,
        state,
        cone,
        num_variables,
    )
    state_cotangent = jnp.concatenate(
        (cotangent, jnp.zeros(state.shape[0] - num_variables, dtype=state.dtype))
    )
    linear_result = solve_linear(
        LeastSquaresProblem(
            adjoint(operator),
            problem_id="conic-projection-kkt-adjoint-system",
        ),
        state_cotangent,
        policy=linear_policy,
    )
    _, pullback = jax.vjp(
        lambda p, q, a, b: _data_residual(
            p,
            q,
            a,
            b,
            state,
            cone,
            num_variables,
        ),
        quadratic,
        linear,
        matrix,
        rhs,
    )
    gradients = jax.tree.map(jnp.negative, pullback(linear_result.value))
    return linear_result, gradients


def _result_regularity(prepared: PreparedConicSensitivity, linear_result, /) -> Array:
    diagnostics = linear_result.diagnostics
    condition = diagnostics.condition_estimate
    condition_ok = jnp.isfinite(condition)
    precision = prepared.linear_policy.precision
    if precision is not None and precision.condition_limit is not None:
        condition_ok = condition_ok & (condition <= precision.condition_limit)
    return (
        prepared.forward_valid
        & prepared.projection_regular
        & linear_result.successful
        & diagnostics.finite
        & diagnostics.converged
        & condition_ok
    )


def _guard_result(value: PyTree[Array], regular: Array, message: str, /):
    leaves, structure = jax.tree.flatten(value)
    leaves[0] = eqx.error_if(leaves[0], jnp.any(~regular), message)
    return jax.tree.unflatten(structure, leaves)


def _validate_tangent(
    prepared: PreparedConicSensitivity,
    tangent: ConicProgramData,
    /,
) -> ConicProgramData:
    if not isinstance(tangent, ConicProgramData):
        raise TypeError("tangent must be a ConicProgramData.")
    original = prepared.original_data
    if prepared.quadratic_present != (tangent.quadratic is not None):
        raise ValueError("Tangent quadratic presence must match the conic program.")
    pairs = (
        (tangent.linear, original.linear, "linear"),
        (tangent.constraint_matrix, original.constraint_matrix, "constraint_matrix"),
        (tangent.constraint_rhs, original.constraint_rhs, "constraint_rhs"),
        (tangent.lower_bounds, original.lower_bounds, "lower_bounds"),
        (tangent.upper_bounds, original.upper_bounds, "upper_bounds"),
    )
    if tangent.quadratic is not None and original.quadratic is not None:
        pairs = ((tangent.quadratic, original.quadratic, "quadratic"), *pairs)
    for value, reference, name in pairs:
        if value.shape != reference.shape:
            raise ValueError(
                f"Tangent {name} shape must be {reference.shape}; got {value.shape}."
            )
        if value.dtype != reference.dtype:
            raise TypeError(
                f"Tangent {name} dtype must be {reference.dtype}; got {value.dtype}."
            )
    fixed = jnp.asarray(prepared.fixed_indices, dtype=jnp.int32)
    finite = jnp.all(
        jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value, _, _ in pairs))
    )
    fixed_mismatch = jnp.any(
        tangent.lower_bounds[..., fixed] != tangent.upper_bounds[..., fixed]
    )
    invalid_lower = jnp.any((~prepared.lower_tangent_mask) & (tangent.lower_bounds != 0))
    invalid_upper = jnp.any((~prepared.upper_tangent_mask) & (tangent.upper_bounds != 0))
    guarded = eqx.error_if(
        tangent.linear,
        ~finite | fixed_mismatch | invalid_lower | invalid_upper,
        "Conic tangent must be finite and preserve fixed and infinite bound roles.",
    )
    return eqx.tree_at(lambda value: value.linear, tangent, guarded)


def _lower_tangent(
    prepared: PreparedConicSensitivity,
    tangent: ConicProgramData,
    /,
) -> tuple[Array, Array, Array, Array]:
    tangent = _validate_tangent(prepared, tangent)
    count = prepared.num_cases
    variables = prepared.num_variables
    original_constraints = prepared.num_original_constraints
    fixed = jnp.asarray(prepared.fixed_indices, dtype=jnp.int32)
    lower = jnp.asarray(prepared.lower_indices, dtype=jnp.int32)
    upper = jnp.asarray(prepared.upper_indices, dtype=jnp.int32)
    if tangent.quadratic is None:
        tangent_quadratic = jnp.zeros(
            (count, variables, variables), dtype=prepared.quadratic.dtype
        )
    else:
        tangent_quadratic = tangent.quadratic.reshape((count, variables, variables))
        tangent_quadratic = 0.5 * (
            tangent_quadratic + jnp.swapaxes(tangent_quadratic, -1, -2)
        )
    tangent_linear = tangent.linear.reshape((count, variables))
    tangent_matrix = tangent.constraint_matrix.reshape(
        (count, original_constraints, variables)
    )
    zero_bound_rows = jnp.zeros(
        (
            count,
            len(prepared.fixed_indices)
            + len(prepared.lower_indices)
            + len(prepared.upper_indices),
            variables,
        ),
        dtype=tangent_matrix.dtype,
    )
    tangent_matrix = jnp.concatenate((tangent_matrix, zero_bound_rows), axis=1)
    tangent_rhs = tangent.constraint_rhs.reshape((count, original_constraints))
    lower_tangent = tangent.lower_bounds.reshape((count, variables))
    upper_tangent = tangent.upper_bounds.reshape((count, variables))
    tangent_rhs = jnp.concatenate(
        (
            tangent_rhs,
            0.5 * (lower_tangent[:, fixed] + upper_tangent[:, fixed]),
            -lower_tangent[:, lower],
            upper_tangent[:, upper],
        ),
        axis=1,
    )
    return tangent_quadratic, tangent_linear, tangent_matrix, tangent_rhs


def _pullback_data(
    prepared: PreparedConicSensitivity,
    quadratic: Array,
    linear: Array,
    matrix: Array,
    rhs: Array,
    /,
) -> ConicProgramData:
    count = prepared.num_cases
    variables = prepared.num_variables
    original_constraints = prepared.num_original_constraints
    fixed = jnp.asarray(prepared.fixed_indices, dtype=jnp.int32)
    lower = jnp.asarray(prepared.lower_indices, dtype=jnp.int32)
    upper = jnp.asarray(prepared.upper_indices, dtype=jnp.int32)
    quadratic = 0.5 * (quadratic + jnp.swapaxes(quadratic, -1, -2))
    original_matrix = matrix[:, :original_constraints, :]
    original_rhs = rhs[:, :original_constraints]
    cursor = original_constraints
    fixed_rhs = rhs[:, cursor : cursor + len(prepared.fixed_indices)]
    cursor += len(prepared.fixed_indices)
    lower_rhs = rhs[:, cursor : cursor + len(prepared.lower_indices)]
    cursor += len(prepared.lower_indices)
    upper_rhs = rhs[:, cursor : cursor + len(prepared.upper_indices)]
    lower_bounds = jnp.zeros((count, variables), dtype=rhs.dtype)
    upper_bounds = jnp.zeros((count, variables), dtype=rhs.dtype)
    lower_bounds = lower_bounds.at[:, fixed].add(0.5 * fixed_rhs)
    upper_bounds = upper_bounds.at[:, fixed].add(0.5 * fixed_rhs)
    lower_bounds = lower_bounds.at[:, lower].add(-lower_rhs)
    upper_bounds = upper_bounds.at[:, upper].add(upper_rhs)
    batch_shape = prepared.batch_shape
    return ConicProgramData(
        None
        if not prepared.quadratic_present
        else _restore_cases(quadratic, batch_shape),
        _restore_cases(linear, batch_shape),
        _restore_cases(original_matrix, batch_shape),
        _restore_cases(original_rhs, batch_shape),
        _restore_cases(lower_bounds, batch_shape),
        _restore_cases(upper_bounds, batch_shape),
    )


def prepare_conic_sensitivity(
    prepared: PreparedConvexProgram,
    execution: ConvexProgramExecution,
    /,
    *,
    linear: LinearSolvePolicy | None = None,
    regularity_tolerance: float = 1e-7,
) -> PreparedConicSensitivity:
    """Bind an audited conic execution to a reusable projection-KKT derivative."""

    if not isinstance(prepared, PreparedConvexProgram):
        raise TypeError("prepared must be a PreparedConvexProgram.")
    if not isinstance(execution, ConvexProgramExecution):
        raise TypeError("execution must be a ConvexProgramExecution.")
    if not isinstance(prepared.program, ConicProgram):
        raise TypeError("Prepared conic sensitivity requires a ConicProgram.")
    program = prepared.program
    result = execution.result
    provenance = result.provenance
    if execution.plan_id != prepared.plan.plan_id:
        raise ValueError("Execution plan does not match the prepared conic program.")
    if int(execution.numeric_version) != int(prepared.numeric_version):
        raise ValueError("Execution numeric version does not match prepared conic data.")
    if execution.numeric_binding_id != prepared.numeric_binding_id:
        raise ValueError("Execution numeric binding does not match prepared conic data.")
    if (
        provenance.problem_id != program.problem_id
        or provenance.structure_id != program.structure_id
        or provenance.policy_id != prepared.plan.policy.policy_id
        or int(provenance.numeric_version) != int(prepared.numeric_version)
        or provenance.numeric_binding_id != prepared.numeric_binding_id
    ):
        raise ValueError(
            "Execution provenance does not match the prepared conic program."
        )
    tolerance = float(regularity_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("regularity_tolerance must be finite and positive.")
    linear_policy, failure_mode = _linear_policy(linear)
    fixed_array, lower_array, upper_array = _conic_bound_indices(
        program.bounds,
        program.batch_shape,
        program.num_variables,
    )
    fixed_indices = tuple(int(index) for index in fixed_array)
    lower_indices = tuple(int(index) for index in lower_array)
    upper_indices = tuple(int(index) for index in upper_array)
    count = int(np.prod(program.batch_shape)) if program.batch_shape else 1
    variables = program.num_variables
    original_constraints = program.num_constraints
    dtype = program.linear.dtype
    fixed = jnp.asarray(fixed_indices, dtype=jnp.int32)
    lower = jnp.asarray(lower_indices, dtype=jnp.int32)
    upper = jnp.asarray(upper_indices, dtype=jnp.int32)
    identity = jnp.eye(variables, dtype=dtype)
    bound_matrix = jnp.concatenate(
        (identity[fixed], -identity[lower], identity[upper]), axis=0
    )
    bound_matrix = jnp.broadcast_to(
        bound_matrix,
        (count, bound_matrix.shape[0], variables),
    )
    matrix = program.constraint_matrix.reshape((count, original_constraints, variables))
    matrix = jnp.concatenate((matrix, bound_matrix), axis=1)
    lower_values = program.lower_bounds.reshape((count, variables))
    upper_values = program.upper_bounds.reshape((count, variables))
    rhs = program.constraint_rhs.reshape((count, original_constraints))
    rhs = jnp.concatenate(
        (
            rhs,
            0.5 * (lower_values[:, fixed] + upper_values[:, fixed]),
            -lower_values[:, lower],
            upper_values[:, upper],
        ),
        axis=1,
    )
    linear_values = program.linear.reshape((count, variables))
    quadratic_present = program.quadratic is not None
    quadratic = (
        jnp.zeros((count, variables, variables), dtype=dtype)
        if program.quadratic is None
        else program.quadratic.reshape((count, variables, variables))
    )
    regularization = prepared.plan.policy.regularization
    quadratic = quadratic + regularization * jnp.eye(variables, dtype=dtype)
    primal = result.primal.reshape((count, variables))
    original_dual = result.cone_dual.reshape((count, original_constraints))
    lower_dual = result.lower_bound_dual.reshape((count, variables))
    upper_dual = result.upper_bound_dual.reshape((count, variables))
    bound_dual = jnp.concatenate(
        (
            upper_dual[:, fixed] - lower_dual[:, fixed],
            lower_dual[:, lower],
            upper_dual[:, upper],
        ),
        axis=1,
    )
    dual = jnp.concatenate((original_dual, bound_dual), axis=1)
    state = jnp.concatenate((primal, dual), axis=1)
    blocks = _cone_blocks(program.cone)
    if fixed_indices:
        blocks = (*blocks, ZeroCone(len(fixed_indices)))
    if lower_indices:
        blocks = (*blocks, NonnegativeCone(len(lower_indices)))
    if upper_indices:
        blocks = (*blocks, NonnegativeCone(len(upper_indices)))
    cone = ProductCone(blocks)
    residual = jax.vmap(
        lambda p, q, a, b, u: _kkt_residual(
            u,
            p,
            q,
            a,
            b,
            cone,
            variables,
        )
    )(quadratic, linear_values, matrix, rhs, state)
    root_residual_norm = _max_abs(residual)
    data_scale = jnp.maximum(
        1.0,
        jnp.maximum(
            _max_abs(linear_values),
            jnp.maximum(_max_abs(rhs), _max_abs(state)),
        ),
    )
    termination = prepared.plan.policy.termination
    root_tolerance = termination.absolute + termination.relative * data_scale
    projection_point = dual + oe.contract("bij,bj->bi", matrix, primal) - rhs
    projection_margin = cone.dual_projection_smoothness_margin(projection_point)
    projection_scale = jnp.maximum(1.0, _max_abs(projection_point))
    projection_regular = projection_margin > tolerance * projection_scale
    finite = (
        jnp.all(jnp.isfinite(state), axis=-1)
        & jnp.all(jnp.isfinite(residual), axis=-1)
        & jnp.isfinite(root_residual_norm)
    )
    forward_valid = (
        result.successful.reshape((count,))
        & finite
        & (root_residual_norm <= root_tolerance)
    )
    lower_tangent_mask = jnp.zeros((variables,), dtype=bool)
    upper_tangent_mask = jnp.zeros((variables,), dtype=bool)
    lower_tangent_mask = lower_tangent_mask.at[fixed].set(True)
    lower_tangent_mask = lower_tangent_mask.at[lower].set(True)
    upper_tangent_mask = upper_tangent_mask.at[fixed].set(True)
    upper_tangent_mask = upper_tangent_mask.at[upper].set(True)
    original_data = ConicProgramData(
        program.quadratic,
        program.linear,
        program.constraint_matrix,
        program.constraint_rhs,
        program.lower_bounds,
        program.upper_bounds,
    )
    return PreparedConicSensitivity(
        original_data,
        quadratic,
        linear_values,
        matrix,
        rhs,
        state,
        cone,
        forward_valid,
        projection_margin,
        projection_regular,
        root_residual_norm,
        lower_tangent_mask,
        upper_tangent_mask,
        linear_policy,
        prepared.numeric_version,
        batch_shape=program.batch_shape,
        num_cases=count,
        num_variables=variables,
        num_original_constraints=original_constraints,
        num_constraints=matrix.shape[1],
        fixed_indices=fixed_indices,
        lower_indices=lower_indices,
        upper_indices=upper_indices,
        quadratic_present=quadratic_present,
        regularity_tolerance=tolerance,
        failure_mode=failure_mode,
        convex_plan_id=prepared.plan.plan_id,
        numeric_binding_id=prepared.numeric_binding_id,
    )


@eqx.filter_jit
def conic_primal_jvp(
    prepared: PreparedConicSensitivity,
    tangent: ConicProgramData,
    /,
) -> ConicSensitivityResult:
    """Apply the regular conic primal solution derivative to one data tangent."""

    if not isinstance(prepared, PreparedConicSensitivity):
        raise TypeError("prepared must be a PreparedConicSensitivity.")
    tangent_quadratic, tangent_linear, tangent_matrix, tangent_rhs = _lower_tangent(
        prepared,
        tangent,
    )
    solve_cases = eqx.filter_vmap(
        lambda p, q, a, b, u, dp, dq, da, db: _jvp_case(
            p,
            q,
            a,
            b,
            u,
            dp,
            dq,
            da,
            db,
            cone=prepared.cone,
            num_variables=prepared.num_variables,
            linear_policy=prepared.linear_policy,
        )
    )
    linear_result = solve_cases(
        prepared.quadratic,
        prepared.linear,
        prepared.constraint_matrix,
        prepared.constraint_rhs,
        prepared.state,
        tangent_quadratic,
        tangent_linear,
        tangent_matrix,
        tangent_rhs,
    )
    regular = _result_regularity(prepared, linear_result)
    primal_tangent = _mask_cases(
        linear_result.value[:, : prepared.num_variables],
        regular,
    )
    value = _restore_cases(primal_tangent, prepared.batch_shape)
    if prepared.failure_mode == "error":
        value = _guard_result(
            value,
            regular,
            "Conic primal JVP requires a successful regular projection-KKT system.",
        )
    return ConicSensitivityResult(
        value,
        _restore_cases(prepared.forward_valid, prepared.batch_shape),
        _restore_cases(prepared.projection_margin, prepared.batch_shape),
        _restore_cases(prepared.projection_regular, prepared.batch_shape),
        _restore_cases(prepared.root_residual_norm, prepared.batch_shape),
        _restore_cases(linear_result.status, prepared.batch_shape),
        _restore_tree_cases(linear_result.diagnostics, prepared.batch_shape),
        _restore_cases(regular, prepared.batch_shape),
        prepared.numeric_version,
        convex_plan_id=prepared.convex_plan_id,
        linear_plan_id=linear_result.provenance.plan_id,
        numeric_binding_id=prepared.numeric_binding_id,
    )


@eqx.filter_jit
def conic_primal_vjp(
    prepared: PreparedConicSensitivity,
    cotangent: ArrayLike,
    /,
) -> ConicSensitivityResult:
    """Apply the adjoint regular conic primal derivative to one primal cotangent."""

    if not isinstance(prepared, PreparedConicSensitivity):
        raise TypeError("prepared must be a PreparedConicSensitivity.")
    cotangent_ = jnp.asarray(cotangent)
    expected = prepared.batch_shape + (prepared.num_variables,)
    if cotangent_.shape != expected:
        raise ValueError(f"cotangent must have shape {expected}; got {cotangent_.shape}.")
    if cotangent_.dtype != prepared.linear.dtype:
        raise TypeError(
            f"cotangent dtype must be {prepared.linear.dtype}; got {cotangent_.dtype}."
        )
    cotangent_ = eqx.error_if(
        cotangent_,
        jnp.any(~jnp.isfinite(cotangent_)),
        "Conic primal cotangent must be finite.",
    ).reshape((prepared.num_cases, prepared.num_variables))
    solve_cases = eqx.filter_vmap(
        lambda p, q, a, b, u, dx: _vjp_case(
            p,
            q,
            a,
            b,
            u,
            dx,
            cone=prepared.cone,
            num_variables=prepared.num_variables,
            linear_policy=prepared.linear_policy,
        )
    )
    linear_result, gradients = solve_cases(
        prepared.quadratic,
        prepared.linear,
        prepared.constraint_matrix,
        prepared.constraint_rhs,
        prepared.state,
        cotangent_,
    )
    regular = _result_regularity(prepared, linear_result)
    gradients = jax.tree.map(lambda value: _mask_cases(value, regular), gradients)
    value = _pullback_data(prepared, *gradients)
    if prepared.failure_mode == "error":
        value = _guard_result(
            value,
            regular,
            "Conic primal VJP requires a successful regular projection-KKT system.",
        )
    return ConicSensitivityResult(
        value,
        _restore_cases(prepared.forward_valid, prepared.batch_shape),
        _restore_cases(prepared.projection_margin, prepared.batch_shape),
        _restore_cases(prepared.projection_regular, prepared.batch_shape),
        _restore_cases(prepared.root_residual_norm, prepared.batch_shape),
        _restore_cases(linear_result.status, prepared.batch_shape),
        _restore_tree_cases(linear_result.diagnostics, prepared.batch_shape),
        _restore_cases(regular, prepared.batch_shape),
        prepared.numeric_version,
        convex_plan_id=prepared.convex_plan_id,
        linear_plan_id=linear_result.provenance.plan_id,
        numeric_binding_id=prepared.numeric_binding_id,
    )


__all__ = [
    "ConicProgramData",
    "ConicSensitivityResult",
    "PreparedConicSensitivity",
    "conic_primal_jvp",
    "conic_primal_vjp",
    "prepare_conic_sensitivity",
]
