#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Reduced quasi-definite Newton systems for the sparse homogeneous embedding.

With ``v = (x, z, s, tau, kappa)`` the embedding Newton system eliminates
``ds = -R_p - A dx + b dtau`` and every non-zero-cone multiplier
``dz_C = W (A_C dx - b_C dtau) + W R_p,C - T R_c,C`` (``W = z/s`` and ``T = 1/s`` on
orthant rows, ``W = mu ∇²F(s)`` and ``T = I`` on barrier-form rows). The remaining
unknowns ``(dx, dz_Z)`` satisfy the symmetric quasi-definite system

    [ Q + A_Cᵀ W A_C   A_Zᵀ ] [dx  ]   =  f + dtau h,
    [ A_Z               0   ] [dz_Z]

solved for both right-hand sides by one refreshed sparse factorization of its
statically regularized form plus iterative refinement on the exact operator.
An unresolved full direction receives one additive correction of its original
equation defect; bounded native GMRES may resolve that correction's inner solve.
Accepted directions are retained, and correction structure is host-prepared.
The scalar ``dtau`` follows from the linearized gap row and ``dkappa`` from the
``tau kappa`` row. Each assembled direction is accepted only by its residual in
the full embedding linearization.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...linalg import (
    AbstractSparseLinearOperator,
    ArraySpace,
    DifferentiationPolicy,
    FailurePolicy,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSolveTemplate,
    LinearSystem,
    PreconditioningPolicy,
    prepare_template,
    PreparedPreconditioner,
    PreparedSparseFactorization,
    refresh_sparse_factorization_values,
    RHSLayout,
    solve_many,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationPreconditioner,
    SparseFactorizationPreconditionerBuilder,
    SparseFactorizationStatus,
    TolerancePolicy,
)
from ...linalg._norm import scaled_norm
from ...linalg._runtime import _bind_for_template
from ...sparse import EdgeRelation, SparseCoordinateOperator, SparseLinearMap
from ._barrier import cone_barrier_oracle
from ._cones import AbstractConvexCone, NonnegativeCone, ProductCone, ZeroCone
from ._policy import ConvexSolvePolicy, NativeHomogeneousConic
from ._problem import (
    _conic_matrix_mv,
    _conic_matrix_transpose_mv,
    _conic_quadratic_perspective,
    ConicProgram,
)


# Maximum iterative-refinement sweeps against the unregularized reduced operator.
_REFINEMENT_STEPS = 32
# Inexact-Newton forcing term relative to the row-equilibrated embedding residual.
_INEXACT_FORCING = 1e-4


class FactorizedNewtonPlan(StrictModule):
    """Symbolic reduced-KKT pattern and assembly routes for one program structure.

    Entry arrays index the factorization's CSR value order. ``*_edges`` index the
    flattened coefficient vectors of the program operators; ``product_weights``
    indexes the flattened block weights ``W`` in cone order.
    """

    factorization: SparseFactorizationPlan
    entry_rows: Array
    entry_columns: Array
    quadratic_positions: Array
    quadratic_edges: Array
    product_positions: Array
    product_left: Array
    product_right: Array
    product_weights: Array
    constraint_positions: Array
    constraint_edges: Array
    primal_diagonal: Array
    dual_diagonal: Array
    zero_rows: Array
    structure_id: str = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_zero_rows: int = eqx.field(static=True)
    num_entries: int = eqx.field(static=True)
    multiplier_independent: bool = eqx.field(static=True)
    operator: SparseCoordinateOperator
    correction: LinearSolveTemplate


class FactorizedNewtonFactor(StrictModule):
    """One numerically refreshed reduced-KKT factor and its inertia evidence."""

    factor: PreparedSparseFactorization
    exact_values: Array
    regularization: Array
    inertia_valid: Array


class _ReducedNewtonStructure(NamedTuple):
    operator: SparseCoordinateOperator
    correction: LinearSolveTemplate
    factorization: SparseFactorizationPlan


class _ReducedNewtonRhs(NamedTuple):
    residual: Array
    rhs: Array
    offset: Array
    weighted_rhs: Array
    weighted_rhs_image: Array


class _AuditedDirection(NamedTuple):
    value: Array
    accepted: Array
    status: Array
    defect: Array


def _split(cone: AbstractConvexCone, value: Array) -> tuple[Array, ...]:
    return cone.split(value) if isinstance(cone, ProductCone) else (value,)


def _blocks(cone: AbstractConvexCone) -> tuple[AbstractConvexCone, ...]:
    return cone.cones if isinstance(cone, ProductCone) else (cone,)


def _host_entries(
    operator: Array | AbstractSparseLinearOperator | None, rows: int, columns: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return host ``(row, column, coefficient-index)`` triples of one operator."""

    if operator is None:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty, empty
    if isinstance(operator, (SparseCoordinateOperator, SparseLinearMap)):
        relation = operator.relation
        if not isinstance(relation, EdgeRelation) or (
            isinstance(operator, SparseCoordinateOperator)
            and operator.block_shape is not None
        ):
            raise TypeError(
                "Factorized native conic Newton systems require scalar EdgeRelation "
                "operators; select NativeHomogeneousConic(newton='matrix-free')."
            )
        valid = np.asarray(relation.valid, dtype=np.bool_).reshape(-1)
        edges = np.flatnonzero(valid)
        return (
            np.asarray(relation.target_indices, dtype=np.int64).reshape(-1)[edges],
            np.asarray(relation.source_indices, dtype=np.int64).reshape(-1)[edges],
            edges,
        )
    if isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError(
            "Factorized native conic Newton systems require explicit sparse "
            "coefficients; select NativeHomogeneousConic(newton='matrix-free')."
        )
    grid_rows, grid_columns = np.indices((rows, columns))
    flat_rows = grid_rows.reshape(-1).astype(np.int64)
    return (
        flat_rows,
        grid_columns.reshape(-1).astype(np.int64),
        np.arange(rows * columns, dtype=np.int64),
    )


def _coefficients(operator: Array | AbstractSparseLinearOperator, /) -> Array:
    if isinstance(operator, (SparseCoordinateOperator, SparseLinearMap)):
        return operator.coefficients.reshape(-1)
    if isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError("Factorized native conic Newton systems require coefficients.")
    return operator.reshape(-1)


def _row_groups(rows: np.ndarray, count: int, /) -> list[np.ndarray]:
    order = np.argsort(rows, kind="stable")
    bounds = np.searchsorted(rows[order], np.arange(count + 1))
    return [order[bounds[row] : bounds[row + 1]] for row in range(count)]


def _prepare_reduced_structure(
    program: ConicProgram,
    policy: ConvexSolvePolicy,
    factorization_policy: SparseFactorizationPolicy,
    rows: np.ndarray,
    columns: np.ndarray,
    size: int,
) -> _ReducedNewtonStructure:
    """Own canonical sparse storage, shared factor analysis, and admission."""
    relation = EdgeRelation(
        jnp.asarray(columns, dtype=jnp.int32),
        jnp.asarray(rows, dtype=jnp.int32),
        source_size=size,
        target_size=size,
    )
    coordinates = ArraySpace(
        (size,),
        dtype=program.linear.dtype,
        space_id=f"native-hsd:{program.structure_id}:reduced-coordinates",
    )
    operator = SparseCoordinateOperator(
        relation,
        jnp.arange(rows.size, dtype=program.linear.dtype),
        source=coordinates,
        target=coordinates,
        operator_id=f"native-hsd:{program.structure_id}:reduced",
    )
    # Preserve the row-major assembly order required by numeric factor refresh.
    storage = operator.sparse_storage()
    if not np.array_equal(
        np.asarray(storage.values), np.arange(rows.size, dtype=np.float64)
    ):
        raise RuntimeError("Reduced KKT pattern is not in canonical CSR order.")
    linear_policy = LinearSolvePolicy(
        GMRES(restart=_REFINEMENT_STEPS),
        tolerance=TolerancePolicy(
            relative=16.0 * np.finfo(operator.coefficients.dtype).eps,
            absolute=0.0,
            max_steps=_REFINEMENT_STEPS,
        ),
        preconditioning=PreconditioningPolicy(
            SparseFactorizationPreconditionerBuilder(factorization_policy),
            side="right",
            refresh="frozen",
        ),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
        materialization=policy.materialization,
        resources=policy.resources,
    )
    correction = prepare_template(
        LinearSystem(operator, problem_id=operator.operator_id),
        linear_policy,
        rhs_layout=RHSLayout((2,)),
    )
    preconditioner_plan = correction.plan.preconditioner_plan
    if preconditioner_plan is None or preconditioner_plan.setup is None:
        raise RuntimeError("Reduced Newton correction lost its sparse factor plan.")
    factorization = preconditioner_plan.setup.construction_for(operator.operator_id)
    if not isinstance(factorization, SparseFactorizationPlan):
        raise RuntimeError("Reduced Newton correction requires sparse factorization.")
    if factorization.factor_bytes > policy.resources.factorization_bytes:
        raise ValueError("Reduced Newton factor exceeds the factorization byte budget.")
    return _ReducedNewtonStructure(operator, correction, factorization)


def prepare_factorized_newton(
    program: ConicProgram, policy: ConvexSolvePolicy, /
) -> FactorizedNewtonPlan:
    """Analyze the reduced quasi-definite Newton pattern once per structure.

    Host-only immutable preparation over concrete sparse topology. The pattern
    and its fill-reducing symbolic factorization are bounded by ``policy``;
    exceeding a declared budget refuses instead of switching routes.
    Explicit block/row/pair enumeration preserves canonical pattern identities
    and the order of symbolic-budget refusals.
    """

    if program.batch_shape:
        raise ValueError("Factorized homogeneous Newton systems require one case.")
    method = policy.method
    if not isinstance(method, NativeHomogeneousConic):
        raise TypeError("Reduced Newton preparation requires NativeHomogeneousConic.")
    n, m = program.num_variables, program.num_constraints
    a_rows, a_columns, a_edges = _host_entries(program.constraint_matrix, m, n)
    q_rows, q_columns, q_edges = _host_entries(program.quadratic, n, n)
    blocks = _blocks(program.cone)
    slices = (
        program.cone.slices
        if isinstance(program.cone, ProductCone)
        else (slice(0, program.cone.dimension),)
    )
    zero_rows = np.concatenate(
        [
            np.arange(block_slice.start, block_slice.stop, dtype=np.int64)
            for block, block_slice in zip(blocks, slices, strict=True)
            if isinstance(block, ZeroCone)
        ]
        or [np.empty((0,), dtype=np.int64)]
    )
    zero_index = np.full((m,), -1, dtype=np.int64)
    zero_index[zero_rows] = np.arange(zero_rows.size)
    size = n + zero_rows.size
    groups = _row_groups(a_rows, m)
    product_keys: list[np.ndarray] = []
    product_left: list[np.ndarray] = []
    product_right: list[np.ndarray] = []
    product_weights: list[np.ndarray] = []
    weight_cursor = 0
    work = 0
    for block, block_slice in zip(blocks, slices, strict=True):
        if isinstance(block, ZeroCone):
            continue
        rows = range(block_slice.start, block_slice.stop)
        dimension = block_slice.stop - block_slice.start
        orthant = isinstance(block, NonnegativeCone)
        for local_left, row_left in enumerate(rows):
            partners = ((local_left, row_left),) if orthant else tuple(enumerate(rows))
            for local_right, row_right in partners:
                left = groups[row_left]
                right = groups[row_right]
                work += left.size * right.size
                if work > method.factorization.max_symbolic_work:
                    raise ValueError(
                        "Reduced homogeneous KKT assembly exceeds max_symbolic_work; "
                        "select NativeHomogeneousConic(newton='matrix-free') or raise "
                        "the factorization budget."
                    )
                if left.size == 0 or right.size == 0:
                    continue
                pair_left = np.repeat(left, right.size)
                pair_right = np.tile(right, left.size)
                product_keys.append(a_columns[pair_left] * size + a_columns[pair_right])
                product_left.append(a_edges[pair_left])
                product_right.append(a_edges[pair_right])
                weight = (
                    weight_cursor + local_left
                    if orthant
                    else weight_cursor + local_left * dimension + local_right
                )
                product_weights.append(np.full(pair_left.size, weight, dtype=np.int64))
        weight_cursor += dimension if orthant else dimension * dimension
    in_zero = zero_index[a_rows] >= 0
    zero_edges = a_edges[in_zero]
    zero_targets = n + zero_index[a_rows[in_zero]]
    zero_columns = a_columns[in_zero]
    constraint_keys = np.concatenate(
        (zero_targets * size + zero_columns, zero_columns * size + zero_targets)
    )
    constraint_edges = np.concatenate((zero_edges, zero_edges))
    quadratic_keys = q_rows * size + q_columns
    diagonal = np.arange(size, dtype=np.int64)
    diagonal_keys = diagonal * size + diagonal

    def flat(values: list[np.ndarray]) -> np.ndarray:
        return (
            np.concatenate(values).astype(np.int64)
            if values
            else np.empty((0,), dtype=np.int64)
        )

    product_key_array = flat(product_keys)
    keys = np.unique(
        np.concatenate(
            (quadratic_keys, product_key_array, constraint_keys, diagonal_keys)
        )
    )
    entry_rows = keys // size
    entry_columns = keys % size
    structure = _prepare_reduced_structure(
        program, policy, method.factorization, entry_rows, entry_columns, size
    )

    def positions(selected: np.ndarray) -> Array:
        return jnp.asarray(np.searchsorted(keys, selected), dtype=jnp.int32)

    multiplier_independent = all(
        isinstance(block, (ZeroCone, NonnegativeCone)) for block in blocks
    )
    return FactorizedNewtonPlan(
        structure.factorization,
        jnp.asarray(entry_rows, dtype=jnp.int32),
        jnp.asarray(entry_columns, dtype=jnp.int32),
        positions(quadratic_keys),
        jnp.asarray(q_edges, dtype=jnp.int32),
        positions(product_key_array),
        jnp.asarray(flat(product_left), dtype=jnp.int32),
        jnp.asarray(flat(product_right), dtype=jnp.int32),
        jnp.asarray(flat(product_weights), dtype=jnp.int32),
        positions(constraint_keys),
        jnp.asarray(constraint_edges, dtype=jnp.int32),
        positions(diagonal_keys[:n]),
        positions(diagonal_keys[n:]),
        jnp.asarray(zero_rows, dtype=jnp.int32),
        structure_id=program.structure_id,
        num_variables=n,
        num_zero_rows=zero_rows.size,
        num_entries=keys.size,
        multiplier_independent=multiplier_independent,
        operator=structure.operator,
        correction=structure.correction,
    )


def _block_weights(
    program: ConicProgram, slack: Array, dual: Array, mu: Array, /
) -> Array:
    pieces: list[Array] = []
    for block, slack_block, dual_block in zip(
        _blocks(program.cone),
        _split(program.cone, slack),
        _split(program.cone, dual),
        strict=True,
    ):
        if isinstance(block, ZeroCone):
            continue
        if isinstance(block, NonnegativeCone):
            pieces.append(dual_block / slack_block)
        else:
            hessian = cone_barrier_oracle(block).hessian(slack_block)
            pieces.append((mu * hessian).reshape(-1))
    if not pieces:
        return jnp.empty((0,), dtype=slack.dtype)
    return jnp.concatenate(pieces)


def _apply_weights(
    program: ConicProgram, slack: Array, dual: Array, mu: Array, value: Array, /
) -> Array:
    """Apply ``W`` on non-zero-cone rows; zero-cone rows map to zero."""

    pieces: list[Array] = []
    for block, slack_block, dual_block, value_block in zip(
        _blocks(program.cone),
        _split(program.cone, slack),
        _split(program.cone, dual),
        _split(program.cone, value),
        strict=True,
    ):
        if isinstance(block, ZeroCone):
            pieces.append(jnp.zeros_like(value_block))
        elif isinstance(block, NonnegativeCone):
            pieces.append(dual_block / slack_block * value_block)
        else:
            hessian = cone_barrier_oracle(block).hessian(slack_block)
            pieces.append(mu * (hessian @ value_block))
    return jnp.concatenate(pieces)


def _scale_centrality(program: ConicProgram, slack: Array, value: Array, /) -> Array:
    """Apply ``T`` on non-zero-cone centrality rows; zero-cone rows map to zero."""

    pieces: list[Array] = []
    for block, slack_block, value_block in zip(
        _blocks(program.cone),
        _split(program.cone, slack),
        _split(program.cone, value),
        strict=True,
    ):
        if isinstance(block, ZeroCone):
            pieces.append(jnp.zeros_like(value_block))
        elif isinstance(block, NonnegativeCone):
            pieces.append(value_block / slack_block)
        else:
            pieces.append(value_block)
    return jnp.concatenate(pieces)


def factor_reduced_newton(
    plan: FactorizedNewtonPlan, program: ConicProgram, vector: Array, mu: Array, /
) -> FactorizedNewtonFactor:
    """Assemble and refresh the regularized reduced KKT factor at one iterate."""

    if plan.structure_id != program.structure_id:
        raise ValueError("Factorized Newton plan does not match the program structure.")
    n, m = program.num_variables, program.num_constraints
    dual = vector[n : n + m]
    slack = vector[n + m : n + 2 * m]
    a = _coefficients(program.constraint_matrix)
    weights = _block_weights(program, slack, dual, mu)
    contributions = [
        a[plan.product_left] * a[plan.product_right] * weights[plan.product_weights],
        a[plan.constraint_edges],
    ]
    targets = [plan.product_positions, plan.constraint_positions]
    if program.quadratic is not None:
        contributions.append(_coefficients(program.quadratic)[plan.quadratic_edges])
        targets.append(plan.quadratic_positions)
    exact = jax.ops.segment_sum(
        jnp.concatenate(contributions),
        jnp.concatenate(targets),
        num_segments=plan.num_entries,
    )
    # Static primal-dual regularization makes the factored matrix quasi-definite
    # for every symmetric ordering; refinement restores the exact system. Each
    # primal diagonal is perturbed by sqrt(eps) relative to itself, so active
    # rows with large z/s weights keep the same relative accuracy as the rest,
    # and the sqrt(eps) dual shift bounds the 1/delta growth of dual-first pivots.
    dtype = vector.dtype
    root_eps = jnp.sqrt(jnp.finfo(dtype).eps)
    primal_shift = root_eps * jnp.maximum(1.0, jnp.abs(exact[plan.primal_diagonal]))
    regularization = jnp.asarray(root_eps, dtype=dtype)
    regularized = exact.at[plan.primal_diagonal].add(primal_shift)
    regularized = regularized.at[plan.dual_diagonal].add(-regularization)
    factor = refresh_sparse_factorization_values(plan.factorization, regularized)
    # A static-pivot factorization of a symmetric matrix has U's diagonal equal to
    # the LDLᵀ pivots; quasi-definiteness requires exactly n positive pivots.
    pivots = factor.factor_values[plan.factorization.diagonal_positions]
    positive = jnp.sum(pivots > 0.0)
    negative = jnp.sum(pivots < 0.0)
    inertia_valid = (positive == n) & (negative == plan.num_zero_rows)
    return FactorizedNewtonFactor(factor, exact, regularization, inertia_valid)


def _exact_action(plan: FactorizedNewtonPlan, values: Array, vectors: Array, /) -> Array:
    return jax.ops.segment_sum(
        values[:, None] * vectors[plan.entry_columns],
        plan.entry_rows,
        num_segments=vectors.shape[0],
        indices_are_sorted=True,
    )


def _reduced_solve(
    plan: FactorizedNewtonPlan, factor: FactorizedNewtonFactor, rhs: Array, /
) -> Array:
    """Regularized solve refined toward the exact reduced system.

    Refinement continues while the exact reduced residual halves and stops at
    rounding level or stagnation, retaining the best iterate. Dependent
    zero-cone rows make the exact system singular but consistent; the
    regularized factor then selects the proximal multiplier and refinement keeps
    the null component bounded. Acceptance is decided in the full embedding.
    """
    rhs_norm = scaled_norm(rhs)
    floor = 16.0 * jnp.finfo(rhs.dtype).eps * rhs_norm

    def defect(value: Array) -> Array:
        return scaled_norm(rhs - _exact_action(plan, factor.exact_values, value))

    initial = factor.factor.solve(rhs).value
    initial_defect = defect(initial)

    def keep_refining(state: tuple[Array, Array, Array, Array]) -> Array:
        _, residual, improving, step = state
        return improving & (residual > floor) & (step < _REFINEMENT_STEPS)

    def refine(
        state: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        value, residual, _, step = state
        correction = factor.factor.solve(
            rhs - _exact_action(plan, factor.exact_values, value)
        ).value
        candidate = value + correction
        candidate_defect = defect(candidate)
        improved = jnp.isfinite(candidate_defect) & (candidate_defect < residual)
        return (
            jnp.where(improved, candidate, value),
            jnp.where(improved, candidate_defect, residual),
            improved & (candidate_defect <= 0.5 * residual),
            step + 1,
        )

    value, _, _, _ = jax.lax.while_loop(
        keep_refining,
        refine,
        (
            initial,
            initial_defect,
            jnp.isfinite(initial_defect),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return value


def _correct_reduced_solve(
    plan: FactorizedNewtonPlan,
    factor: FactorizedNewtonFactor,
    rhs: Array,
    initial: Array,
) -> LinearSolveResult:
    """Correct with the existing factor without weakening the roundoff target."""
    operator = eqx.tree_at(
        lambda value: value.coefficients, plan.operator, factor.exact_values
    )
    preconditioner_plan = plan.correction.plan.preconditioner_plan
    if preconditioner_plan is None:
        raise RuntimeError("Reduced Newton correction lost its preconditioner plan.")
    preconditioner = SparseFactorizationPreconditioner(
        operator,
        factor.factor,
        properties=preconditioner_plan.properties,
        preconditioner_id=f"native-hsd:{plan.structure_id}:regularized-lu",
    )
    # The host builder owns symbolic admission. Supply the genuinely refreshed
    # factor, then reuse that action through the canonical numeric binding.
    supplied = PreparedPreconditioner(
        preconditioner,
        operator,
        preconditioner_plan,
        numeric_version=0,
        built_numeric_version=0,
        refresh_kind="supplied",
    )
    prepared = _bind_for_template(
        plan.correction,
        LinearSystem(operator, problem_id=operator.operator_id),
        previous_preconditioner=supplied,
    )
    return solve_many(prepared, rhs, initial_guess=initial)


def _reduced_newton_rhs(
    plan: FactorizedNewtonPlan,
    program: ConicProgram,
    vector: Array,
    mu: Array,
    residual: Array,
) -> _ReducedNewtonRhs:
    """Eliminate explicit residual data while retaining the original vector."""

    n, m = program.num_variables, program.num_constraints
    dual = vector[n : n + m]
    slack = vector[n + m : n + 2 * m]
    stationarity = residual[:n]
    primal = residual[n : n + m]
    centrality = residual[n + m : n + 2 * m]
    zero = plan.zero_rows
    b = program.constraint_rhs
    offset = _apply_weights(program, slack, dual, mu, primal) - _scale_centrality(
        program, slack, centrality
    )
    weighted_rhs = _apply_weights(program, slack, dual, mu, b)
    weighted_rhs_image = _conic_matrix_transpose_mv(
        program.constraint_matrix, weighted_rhs
    )
    first = jnp.concatenate(
        (
            -stationarity - _conic_matrix_transpose_mv(program.constraint_matrix, offset),
            (centrality - primal)[zero],
        )
    )
    second = jnp.concatenate((weighted_rhs_image - program.linear, b[zero]))
    return _ReducedNewtonRhs(
        residual,
        jnp.stack((first, second), axis=1),
        offset,
        weighted_rhs,
        weighted_rhs_image,
    )


def _audit_direction(
    factor: FactorizedNewtonFactor,
    vector: Array,
    residual_function: Callable[[Array], Array],
    residual: Array,
    row_scale: Array,
    relative_tolerance: float,
    absolute_tolerance: float,
    direction: Array,
) -> _AuditedDirection:
    """Keep the original Jacobian, requested accuracy, and forcing gate."""
    _, action = jax.jvp(residual_function, (vector,), (direction,))
    defect = action + residual
    norm = scaled_norm(row_scale * defect)
    scaled_residual = scaled_norm(row_scale * residual)
    target = absolute_tolerance + relative_tolerance * scaled_residual
    forcing = target + _INEXACT_FORCING * scaled_residual
    finite = (
        jnp.all(jnp.isfinite(direction))
        & jnp.isfinite(norm)
        & jnp.isfinite(scaled_residual)
    )
    factor_ok = (
        factor.factor.status == int(SparseFactorizationStatus.SUCCESS)
    ) & factor.inertia_valid
    accepted = finite & factor_ok & (norm <= forcing)
    status = jnp.where(
        ~finite,
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        jnp.where(
            ~factor_ok,
            int(LinearSolveStatus.SINGULAR),
            jnp.where(
                norm <= target,
                int(LinearSolveStatus.SUCCESS),
                jnp.where(
                    norm <= forcing,
                    int(LinearSolveStatus.STAGNATION),
                    int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return _AuditedDirection(direction, accepted, status, defect)


def _direction_from_reduced(
    plan: FactorizedNewtonPlan,
    factor: FactorizedNewtonFactor,
    program: ConicProgram,
    vector: Array,
    mu: Array,
    residual_function: Callable[[Array], Array],
    row_scale: Array,
    relative_tolerance: float,
    absolute_tolerance: float,
    system: _ReducedNewtonRhs,
    solution: Array,
) -> _AuditedDirection:
    """Recover and audit the original full embedding direction."""
    n, m = program.num_variables, program.num_constraints
    x = vector[:n]
    dual = vector[n : n + m]
    slack = vector[n + m : n + 2 * m]
    tau, kappa = vector[-2], vector[-1]
    residual = system.residual
    primal = residual[n : n + m]
    gap, scalar = residual[-2], residual[-1]
    zero = plan.zero_rows
    b = program.constraint_rhs
    offset, weighted_rhs, weighted_rhs_image = (
        system.offset,
        system.weighted_rhs,
        system.weighted_rhs_image,
    )
    dx_first, dz_first = solution[:n, 0], solution[n:, 0]
    dx_second, dz_second = solution[:n, 1], solution[n:, 1]
    _, _, quadratic_gradient, tau_gradient = _conic_quadratic_perspective(
        program.quadratic, x, tau
    )
    gap_gradient = quadratic_gradient + program.linear + weighted_rhs_image
    constant = (
        jnp.sum(gap_gradient * dx_first)
        + jnp.sum(b[zero] * dz_first)
        + jnp.sum(b * offset)
        - scalar / tau
    )
    coefficient = (
        jnp.sum(gap_gradient * dx_second)
        + jnp.sum(b[zero] * dz_second)
        - jnp.sum(b * weighted_rhs)
        + tau_gradient
        - kappa / tau
    )
    dtau = (-gap - constant) / coefficient
    dx = dx_first + dtau * dx_second
    image = _conic_matrix_mv(program.constraint_matrix, dx)
    dz = _apply_weights(program, slack, dual, mu, image - b * dtau) + offset
    dz = dz.at[zero].set(dz_first + dtau * dz_second)
    ds = -primal - image + b * dtau
    dkappa = (-scalar - kappa * dtau) / tau
    direction = jnp.concatenate((dx, dz, ds, dtau[None], dkappa[None]))
    return _audit_direction(
        factor,
        vector,
        residual_function,
        residual,
        row_scale,
        relative_tolerance,
        absolute_tolerance,
        direction,
    )


def factorized_direction(
    plan: FactorizedNewtonPlan,
    factor: FactorizedNewtonFactor,
    program: ConicProgram,
    vector: Array,
    mu: Array,
    residual_function: Callable[[Array], Array],
    row_scale: Array,
    relative_tolerance: float,
    absolute_tolerance: float,
    /,
) -> tuple[Array, Array, Array]:
    """Retain accepted directions; apply at most one full-equation correction."""
    system = _reduced_newton_rhs(plan, program, vector, mu, residual_function(vector))
    solution = _reduced_solve(plan, factor, system.rhs)

    def recover(value: Array, rhs: _ReducedNewtonRhs) -> _AuditedDirection:
        return _direction_from_reduced(
            plan,
            factor,
            program,
            vector,
            mu,
            residual_function,
            row_scale,
            relative_tolerance,
            absolute_tolerance,
            rhs,
            value,
        )

    initial = recover(solution, system)

    def correct() -> tuple[Array, Array, Array]:
        # Refine the full original equation, not two large absolute solutions.
        correction_rhs = _reduced_newton_rhs(plan, program, vector, mu, initial.defect)
        correction_solution = _reduced_solve(plan, factor, correction_rhs.rhs)
        delta = recover(correction_solution, correction_rhs)

        def refine_delta() -> _AuditedDirection:
            reduced = _correct_reduced_solve(
                plan, factor, correction_rhs.rhs, correction_solution
            )
            audited = recover(reduced.value, correction_rhs)
            status = jnp.where(
                (audited.status == int(LinearSolveStatus.SUCCESS))
                & ~jnp.all(reduced.successful),
                int(LinearSolveStatus.STAGNATION),
                audited.status,
            ).astype(jnp.int32)
            return _AuditedDirection(
                audited.value, audited.accepted, status, audited.defect
            )

        delta = jax.lax.cond(
            ~delta.accepted
            & jnp.all(jnp.isfinite(correction_solution))
            & jnp.all(jnp.isfinite(correction_rhs.rhs)),
            refine_delta,
            lambda: delta,
        )
        total = _audit_direction(
            factor,
            vector,
            residual_function,
            system.residual,
            row_scale,
            relative_tolerance,
            absolute_tolerance,
            initial.value + delta.value,
        )
        # A strict inner miss is not promoted by a successful total-direction gate.
        status = jnp.where(
            (total.status == int(LinearSolveStatus.SUCCESS))
            & (delta.status != int(LinearSolveStatus.SUCCESS)),
            int(LinearSolveStatus.STAGNATION),
            total.status,
        ).astype(jnp.int32)
        return total.value, total.accepted, status

    eligible = (
        ~initial.accepted
        & factor.inertia_valid
        & (factor.factor.status == int(SparseFactorizationStatus.SUCCESS))
        & jnp.all(jnp.isfinite(solution))
        & jnp.all(jnp.isfinite(system.rhs))
        & jnp.all(jnp.isfinite(initial.defect))
    )
    return jax.lax.cond(
        eligible,
        correct,
        lambda: (initial.value, initial.accepted, initial.status),
    )


__all__ = [
    "FactorizedNewtonFactor",
    "FactorizedNewtonPlan",
    "factor_reduced_newton",
    "factorized_direction",
    "prepare_factorized_newton",
]
