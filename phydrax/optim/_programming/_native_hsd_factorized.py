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
statically regularized form plus iterative refinement on the exact operator. The
scalar ``dtau`` follows from the linearized gap row and ``dkappa`` from the
``tau kappa`` row. Each assembled direction is accepted only by its residual in
the full embedding linearization.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...linalg import (
    AbstractSparseLinearOperator,
    LinearSolveStatus,
    prepare_sparse_factorization,
    PreparedSparseFactorization,
    refresh_sparse_factorization_values,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationStatus,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator, SparseLinearMap
from ._barrier import cone_barrier_oracle
from ._cones import AbstractConvexCone, NonnegativeCone, ProductCone, ZeroCone
from ._problem import (
    _conic_matrix_mv,
    _conic_matrix_transpose_mv,
    _conic_quadratic_mv,
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


class FactorizedNewtonFactor(StrictModule):
    """One numerically refreshed reduced-KKT factor and its inertia evidence."""

    factor: PreparedSparseFactorization
    exact_values: Array
    regularization: Array
    inertia_valid: Array


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


def prepare_factorized_newton(
    program: ConicProgram, policy: SparseFactorizationPolicy, /
) -> FactorizedNewtonPlan:
    """Analyze the reduced quasi-definite Newton pattern once per structure.

    Host-only immutable preparation over concrete sparse topology. The pattern
    and its fill-reducing symbolic factorization are bounded by ``policy``;
    exceeding a declared budget refuses instead of switching routes.
    """

    if program.batch_shape:
        raise ValueError("Factorized homogeneous Newton systems require one case.")
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
                if work > policy.max_symbolic_work:
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
    relation = EdgeRelation(
        jnp.asarray(entry_columns, dtype=jnp.int32),
        jnp.asarray(entry_rows, dtype=jnp.int32),
        source_size=size,
        target_size=size,
    )
    pattern = SparseLinearMap(relation, jnp.arange(keys.size, dtype=program.linear.dtype))
    # Row-major unique keys are already canonical CSR order; verify it so the
    # assembled values can be passed directly to the numeric refresh.
    storage = pattern.sparse_storage()
    if not np.array_equal(
        np.asarray(storage.values), np.arange(keys.size, dtype=np.float64)
    ):
        raise RuntimeError("Reduced KKT pattern is not in canonical CSR order.")
    factorization = prepare_sparse_factorization(pattern, policy)

    def positions(selected: np.ndarray) -> Array:
        return jnp.asarray(np.searchsorted(keys, selected), dtype=jnp.int32)

    multiplier_independent = all(
        isinstance(block, (ZeroCone, NonnegativeCone)) for block in blocks
    )
    return FactorizedNewtonPlan(
        factorization,
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
    rhs_norm = jnp.linalg.norm(rhs)
    floor = 16.0 * jnp.finfo(rhs.dtype).eps * rhs_norm

    def defect(value: Array) -> Array:
        return jnp.linalg.norm(rhs - _exact_action(plan, factor.exact_values, value))

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
    """Return one embedding Newton direction, acceptance flag, and linear status."""

    n, m = program.num_variables, program.num_constraints
    x = vector[:n]
    dual = vector[n : n + m]
    slack = vector[n + m : n + 2 * m]
    tau, kappa = vector[-2], vector[-1]
    residual = residual_function(vector)
    stationarity = residual[:n]
    primal = residual[n : n + m]
    centrality = residual[n + m : n + 2 * m]
    gap, scalar = residual[-2], residual[-1]
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
    solution = _reduced_solve(plan, factor, jnp.stack((first, second), axis=1))
    dx_first, dz_first = solution[:n, 0], solution[n:, 0]
    dx_second, dz_second = solution[:n, 1], solution[n:, 1]
    quadratic_x = _conic_quadratic_mv(program.quadratic, x)
    quadratic_energy = jnp.sum(x * quadratic_x)
    gap_gradient = 2.0 * quadratic_x / tau + program.linear + weighted_rhs_image
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
        - quadratic_energy / tau**2
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
    # Acceptance in the original embedding linearization, row-equilibrated like
    # the iterative route. ``SUCCESS`` certifies the requested Newton accuracy;
    # a direction that only meets the inexact-Newton forcing bound (dependent or
    # numerically rank-deficient zero-cone rows) is still a descent step and is
    # reported as ``STAGNATION``. Optimality and certificates stay with the
    # independent original-coordinate audit, never with the Newton solve.
    _, action = jax.jvp(residual_function, (vector,), (direction,))
    defect = jnp.linalg.norm(row_scale * (action + residual))
    scaled_residual = jnp.linalg.norm(row_scale * residual)
    target = absolute_tolerance + relative_tolerance * scaled_residual
    forcing = target + _INEXACT_FORCING * scaled_residual
    finite = jnp.all(jnp.isfinite(direction))
    factor_ok = (
        factor.factor.status == int(SparseFactorizationStatus.SUCCESS)
    ) & factor.inertia_valid
    accepted = finite & factor_ok & (defect <= forcing)
    status = jnp.where(
        ~finite,
        int(LinearSolveStatus.NONFINITE_OUTPUT),
        jnp.where(
            ~factor_ok,
            int(LinearSolveStatus.SINGULAR),
            jnp.where(
                defect <= target,
                int(LinearSolveStatus.SUCCESS),
                jnp.where(
                    defect <= forcing,
                    int(LinearSolveStatus.STAGNATION),
                    int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return direction, accepted, status


__all__ = [
    "FactorizedNewtonFactor",
    "FactorizedNewtonPlan",
    "factor_reduced_newton",
    "factorized_direction",
    "prepare_factorized_newton",
]
