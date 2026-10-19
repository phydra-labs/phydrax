#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Target rank-statistic proposals on free quotient roots; publication is independent.

A hard zero-tolerance quantile is met only by edges whose *published* float
length equals the target exactly. The proposal therefore has three owners:

1. A matrix-free native augmented-Lagrangian solve pushes every mutable quotient
   edge towards the target from below (``length <= target``) under positive
   normalized cell areas and bounded root moves. Its extreme point makes many
   upper constraints active at once.
2. Native primal-dual KKT polishing enforces the retained target-rank equalities
   and the area, inactive-edge, minimum-size and move bounds as hard
   constraints, with the mean-length objective. Finite AL termination
   is not an active-cardinality certificate.
3. Conflict-directed representable placement moves each free root through a bounded
   representable neighborhood until every kept edge rounds to the target with
   the publication's lift and length arithmetic, without any other edge
   exceeding it. A root has at most two kept edges to earlier roots, so it is a
   circle intersection, a circle slide or a free point; conflicts back-jump to
   the latest placed root that constrains it.

Protected roots (explicit corners and native constrained-segment incidence)
never move. Orbit identity is preserved because only quotient representatives
move; cell connectivity and lattice shifts are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from heapq import heapify, heappop, heappush
from typing import final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.geometry._rational import dyadic_integers
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._sizing import size_statistic_quantile, SizeStatistic

from ..._bounds import Bounds
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import (
    NativeBatchTicket,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from ..._strict import StrictModule
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    BlockSpace,
    coordinate_dtype,
    FunctionLinearOperator,
    JacobiPreconditionerBuilder,
    LinearResourceLimitError,
    LinearSolvePolicy,
    LinearSolveStatus,
    MINRES,
    OperatorProperties,
    plan_sparse_assembly,
    PreconditioningPolicy,
    prepare_sparse_assembly,
    prepare_sparse_factorization,
    SparseAssemblyPlan,
    SparseAssemblyPolicy,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationPreconditionerBuilder,
    TolerancePolicy,
)
from ...optim import (
    AbstractPrimalDualKKTSetup,
    AugmentedLagrangian,
    constraint_layout,
    ConstraintLayout,
    MinimizationProblem,
    minimize,
    NonlinearConstraint,
    OptimizationTermination,
    PrimalDualEvidence,
    PrimalDualKKTSetupResult,
    PrimalDualNewtonKrylov,
    ProjectedLBFGS,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ...typing import checked


# Normalized (target-squared) cell area kept by the continuous solve. It is far
# above placement perturbations, so representable placement cannot invert a cell.
_AREA_MARGIN = 2.0**-10
# Normalized length gap identifying an upper constraint as active.
_ACTIVE_GAP = 2.0**-20
# Relative strict margin of demoted active edges; representable placement moves roots by
# at most ``_SLIDES * _SLIDE_STEP`` target lengths, far below this margin.
_DEMOTION = 2.0**-30
_SLIDE_STEP = 2.0**-44
_SLIDES = 512
_FREE_RINGS = 16
_GRID_RADIUS = 3
_BATCH = 16
_EXTREME_INNER_METHOD = ProjectedLBFGS()
_EXTREME_METHOD = AugmentedLagrangian(
    inner_method=_EXTREME_INNER_METHOD,
    maximum_outer_steps=12,
    inner_maximum_steps=256,
)
_EXTREME_EVALUATIONS = 4096
_EXTREME_TERMINATION = OptimizationTermination(
    absolute_optimality=1.0e-12,
    relative_optimality=0.0,
    absolute_step=0.0,
    relative_step=0.0,
    maximum_steps=12,
    maximum_evaluations=_EXTREME_EVALUATIONS,
)
_POLISH_LINEAR_ITERATIONS_PER_TRIAL = 64
_POLISH_TRIALS = 4
_POLISH_METHOD = PrimalDualNewtonKrylov(
    linear_policy=LinearSolvePolicy(
        MINRES(),
        tolerance=TolerancePolicy(
            relative=1.0e-6,
            absolute=1.0e-10,
            max_steps=_POLISH_LINEAR_ITERATIONS_PER_TRIAL,
        ),
        preconditioning=PreconditioningPolicy(JacobiPreconditionerBuilder()),
    ),
    maximum_line_search_steps=_POLISH_TRIALS,
    maximum_restoration_steps=0,
)
_POLISH_TERMINATION = OptimizationTermination(
    absolute_optimality=0.0,
    relative_optimality=0.0,
    absolute_step=0.0,
    relative_step=0.0,
    maximum_steps=16,
    maximum_evaluations=64,
)


class _SizeArgs(NamedTuple):
    origins: Array
    free: Array
    edges: Array
    edge_offsets: Array
    cells: Array
    cell_offsets: Array
    target: Array
    mutable: Array
    minimum: Array


class _PolishArgs(NamedTuple):
    size: _SizeArgs
    active: Array
    goals: Array
    mutable_edges: Array


class _PolishDerivativeSource(NamedTuple):
    polish: _PolishArgs
    lower: Array
    upper: Array
    equality_indices: Array
    lower_indices: Array
    upper_indices: Array


class _PolishDerivativeArgs(NamedTuple):
    polish: _PolishArgs
    lower: Array
    upper: Array
    equality_indices: Array
    lower_indices: Array
    upper_indices: Array
    equality_multipliers: Array
    inequality_multipliers: Array


def _polish_derivative_source(
    args: _PolishArgs, layout: ConstraintLayout, /
) -> _PolishDerivativeSource:
    return _PolishDerivativeSource(
        args,
        layout.lower,
        layout.upper,
        layout.equality_indices,
        layout.lower_indices,
        layout.upper_indices,
    )


def _same_derivative_source(
    original: _PolishDerivativeSource,
    current: _PolishDerivativeSource,
    /,
) -> Array:
    matches = []
    for first, second in zip(
        jax.tree.leaves(original), jax.tree.leaves(current), strict=True
    ):
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError(
                "periodic.derivative_source.fields must retain the prepared shapes and dtypes; got a changed field signature"
            )
        if jnp.issubdtype(first.dtype, jnp.floating):
            bits = np.dtype(f"u{first.dtype.itemsize}")
            first = jax.lax.bitcast_convert_type(first, bits)
            second = jax.lax.bitcast_convert_type(second, bits)
        matches.append(jnp.all(first == second))
    return jnp.all(jnp.stack(matches))


def _polish_derivative_args(
    args: _PolishArgs,
    layout: ConstraintLayout,
    equality_multipliers: Array,
    inequality_multipliers: Array,
    /,
) -> _PolishDerivativeArgs:
    return _PolishDerivativeArgs(
        args,
        layout.lower,
        layout.upper,
        layout.equality_indices,
        layout.lower_indices,
        layout.upper_indices,
        equality_multipliers,
        inequality_multipliers,
    )


class _PeriodicLocalDerivativeState(NamedTuple):
    source: _PolishDerivativeArgs
    edge_gradients: Array
    edge_hessians: Array
    cell_gradients: Array
    lagrangian_weights: Array
    edge_positions: Array
    cell_positions: Array


def _local_edge_length(delta: Array, target: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(delta * delta)) / target


def _local_cell_area(corners: Array, /) -> Array:
    first, second = corners[1] - corners[0], corners[2] - corners[0]
    return first[0] * second[1] - first[1] * second[0]


def _local_role_pullback(
    source: _PolishDerivativeArgs,
    cotangent: tuple[Array, Array],
    /,
) -> Array:
    equality, inequality = cotangent
    lower_size = source.lower_indices.size
    weights = jnp.zeros_like(source.lower).at[source.equality_indices].add(equality)
    weights = weights.at[source.lower_indices].add(
        jnp.where(
            jnp.isfinite(source.lower[source.lower_indices]),
            -inequality[:lower_size],
            0.0,
        )
    )
    return weights.at[source.upper_indices].add(
        jnp.where(
            jnp.isfinite(source.upper[source.upper_indices]),
            inequality[lower_size:],
            0.0,
        )
    )


def _prepare_local_derivative_state(
    parameters: Array,
    source: _PolishDerivativeArgs,
    edge_positions: Array,
    cell_positions: Array,
    /,
) -> _PeriodicLocalDerivativeState:
    args = source.polish
    points = _points(parameters.reshape((args.size.free.size, 2)), args.size)
    edges = args.size.edges[args.mutable_edges]
    delta = (
        points[edges[:, 1]]
        - points[edges[:, 0]]
        + args.size.edge_offsets[args.mutable_edges]
    )
    gradient = (
        jax.vmap(jax.grad(_local_edge_length), in_axes=(0, None))(
            delta,
            args.size.target,
        )
        * args.size.target
    )
    hessian = (
        jax.vmap(jax.hessian(_local_edge_length), in_axes=(0, None))(
            delta,
            args.size.target,
        )
        * args.size.target
        * args.size.target
    )
    corners = (points[args.size.cells] + args.size.cell_offsets) / args.size.target
    cell_gradient = jax.vmap(jax.grad(_local_cell_area))(corners)
    weights = _local_role_pullback(
        source,
        (source.equality_multipliers, source.inequality_multipliers),
    )
    # Exactly the mutable-edge mean objective, not a squared norm.
    weights = weights.at[: args.mutable_edges.size].add(
        -jnp.reciprocal(jnp.sum(args.size.mutable).astype(parameters.dtype)),
    )
    return _PeriodicLocalDerivativeState(
        source,
        gradient,
        hessian,
        cell_gradient,
        weights,
        edge_positions,
        cell_positions,
    )


def _local_constraint_mv(
    state: _PeriodicLocalDerivativeState,
    tangent: Array,
    /,
) -> tuple[Array, Array]:
    source, args = state.source, state.source.polish
    field = jnp.concatenate(
        (
            tangent.reshape((args.size.free.size, 2)),
            jnp.zeros((1, 2), dtype=tangent.dtype),
        )
    )
    edges = state.edge_positions
    edge_values = jnp.sum(
        state.edge_gradients * (field[edges[:, 1]] - field[edges[:, 0]]),
        axis=1,
    )
    cell_values = jnp.sum(state.cell_gradients * field[state.cell_positions], axis=(1, 2))
    values = jnp.concatenate((edge_values, cell_values, tangent))
    lower = jnp.where(
        jnp.isfinite(source.lower[source.lower_indices]),
        -values[source.lower_indices],
        0.0,
    )
    upper = jnp.where(
        jnp.isfinite(source.upper[source.upper_indices]),
        values[source.upper_indices],
        0.0,
    )
    return values[source.equality_indices], jnp.concatenate((lower, upper))


def _local_constraint_vjp(
    state: _PeriodicLocalDerivativeState,
    cotangent: tuple[Array, Array],
    /,
) -> Array:
    args = state.source.polish
    weights = _local_role_pullback(state.source, cotangent)
    mutable, cells = args.mutable_edges.size, args.size.cells.shape[0]
    edges = state.edge_positions
    edge_force = weights[:mutable, None] * state.edge_gradients
    cell_force = weights[mutable : mutable + cells, None, None] * state.cell_gradients
    field = jnp.zeros((args.size.free.size + 1, 2), dtype=edge_force.dtype)
    field = field.at[edges[:, 0]].add(-edge_force).at[edges[:, 1]].add(edge_force)
    field = field.at[state.cell_positions.reshape(-1)].add(cell_force.reshape((-1, 2)))
    return field[:-1].reshape(-1) + weights[mutable + cells :]


def _local_cell_hessian_force(weights: Array, tangent: Array, /) -> Array:
    # Oriented cross-product minors avoid dense 6x6 zero products.
    differences = jnp.stack(
        (
            tangent[:, 1] - tangent[:, 2],
            tangent[:, 2] - tangent[:, 0],
            tangent[:, 0] - tangent[:, 1],
        ),
        axis=1,
    )
    rotated = jnp.stack((differences[..., 1], -differences[..., 0]), axis=2)
    return weights[:, None, None] * rotated


def _local_hessian_mv(state: _PeriodicLocalDerivativeState, tangent: Array, /) -> Array:
    args = state.source.polish
    mutable, cells = args.mutable_edges.size, args.size.cells.shape[0]
    field = jnp.concatenate(
        (
            tangent.reshape((args.size.free.size, 2)),
            jnp.zeros((1, 2), dtype=tangent.dtype),
        )
    )
    edges = state.edge_positions
    delta = field[edges[:, 1]] - field[edges[:, 0]]
    edge_force = jnp.sum(state.edge_hessians * delta[:, None, :], axis=2)
    edge_force = state.lagrangian_weights[:mutable, None] * edge_force
    cell_tangent = field[state.cell_positions]
    cell_force = _local_cell_hessian_force(
        state.lagrangian_weights[mutable : mutable + cells],
        cell_tangent,
    )
    result = jnp.zeros_like(field)
    result = result.at[edges[:, 0]].add(-edge_force).at[edges[:, 1]].add(edge_force)
    result = result.at[state.cell_positions.reshape(-1)].add(cell_force.reshape((-1, 2)))
    return result[:-1].reshape(-1)


class _PeriodicLocalKKTState(NamedTuple):
    derivative: _PeriodicLocalDerivativeState
    edge_blocks: Array
    weighted_cell_gradients: Array
    bound_diagonal: Array


def _prepare_local_kkt_state(
    state: _PeriodicLocalDerivativeState,
    barrier_weights: Array,
    regularization: float,
    /,
) -> _PeriodicLocalKKTState:
    source, args = state.source, state.source.polish
    lower_size = source.lower_indices.size
    gram = (
        jnp.zeros_like(source.lower)
        .at[source.lower_indices]
        .add(
            jnp.where(
                jnp.isfinite(source.lower[source.lower_indices]),
                barrier_weights[:lower_size],
                0.0,
            )
        )
    )
    gram = gram.at[source.upper_indices].add(
        jnp.where(
            jnp.isfinite(source.upper[source.upper_indices]),
            barrier_weights[lower_size:],
            0.0,
        )
    )
    mutable, cells = args.mutable_edges.size, args.size.cells.shape[0]
    edge_blocks = (
        state.lagrangian_weights[:mutable, None, None] * state.edge_hessians
        + gram[:mutable, None, None]
        * state.edge_gradients[:, :, None]
        * state.edge_gradients[:, None, :]
    )
    return _PeriodicLocalKKTState(
        state,
        edge_blocks,
        gram[mutable : mutable + cells, None, None] * state.cell_gradients,
        gram[mutable + cells :] + regularization,
    )


def _local_kkt_mv(
    prepared: _PeriodicLocalKKTState,
    vector: tuple[Array, Array],
    /,
) -> tuple[Array, Array]:
    tangent, multiplier = vector
    state, args = prepared.derivative, prepared.derivative.source.polish
    mutable, cells = args.mutable_edges.size, args.size.cells.shape[0]
    field = jnp.concatenate(
        (
            tangent.reshape((args.size.free.size, 2)),
            jnp.zeros((1, 2), dtype=tangent.dtype),
        )
    )
    edges = state.edge_positions
    delta = field[edges[:, 1]] - field[edges[:, 0]]
    edge_values = jnp.sum(state.edge_gradients * delta, axis=1)
    edge_force = jnp.sum(prepared.edge_blocks * delta[:, None, :], axis=2)
    equality_rows = state.source.equality_indices
    edge_force = edge_force.at[equality_rows].add(
        multiplier[:, None] * state.edge_gradients[equality_rows],
    )
    cell_tangent = field[state.cell_positions]
    cell_values = jnp.sum(state.cell_gradients * cell_tangent, axis=(1, 2))
    cell_force = (
        _local_cell_hessian_force(
            state.lagrangian_weights[mutable : mutable + cells],
            cell_tangent,
        )
        + prepared.weighted_cell_gradients * cell_values[:, None, None]
    )
    result = jnp.zeros_like(field)
    result = result.at[edges[:, 0]].add(-edge_force).at[edges[:, 1]].add(edge_force)
    result = result.at[state.cell_positions.reshape(-1)].add(cell_force.reshape((-1, 2)))
    return (
        result[:-1].reshape(-1) + prepared.bound_diagonal * tangent,
        edge_values[equality_rows],
    )


def _local_constraint_state(
    operator: AbstractLinearOperator, /
) -> _PeriodicLocalDerivativeState:
    match operator:
        case FunctionLinearOperator(function=eqx.Partial() as action):
            if action.func is _local_constraint_mv:
                match action.args:
                    case (_PeriodicLocalDerivativeState() as state,):
                        return state
            raise TypeError(
                "periodic.derivative_operator must retain its prepared local action; got a different operator or action"
            )
        case _:
            raise TypeError(
                "periodic.derivative_operator must retain its prepared local action; got a different operator or action"
            )


@final
class _PeriodicKKTAssemblyData(StrictModule):
    template: SparseCoordinateOperator
    edge_indices: Array
    edge_signs: Array
    cell_indices: Array
    cell_minors: Array
    equality_indices: Array
    equality_signs: Array


def _assemble_local_kkt(
    data: _PeriodicKKTAssemblyData,
    prepared: _PeriodicLocalKKTState,
    /,
) -> SparseCoordinateOperator:
    state = prepared.derivative
    mutable = state.source.polish.mutable_edges.size
    edge = data.edge_indices
    cell = data.cell_indices
    equality = data.equality_indices
    edge_values = (
        prepared.edge_blocks[edge[:, 0], edge[:, 1], edge[:, 2]] * data.edge_signs
    )
    cell_values = (
        prepared.weighted_cell_gradients[cell[:, 0], cell[:, 1] // 2, cell[:, 1] % 2]
        * state.cell_gradients[cell[:, 0], cell[:, 2] // 2, cell[:, 2] % 2]
        + state.lagrangian_weights[mutable + cell[:, 0]] * data.cell_minors
    )
    equality_values = (
        state.edge_gradients[equality[:, 0], equality[:, 1]] * data.equality_signs
    )
    values = jnp.concatenate(
        (
            prepared.bound_diagonal,
            edge_values,
            cell_values,
            equality_values,
            jnp.zeros(
                state.source.equality_indices.size, dtype=prepared.bound_diagonal.dtype
            ),
        )
    )
    return eqx.tree_at(lambda item: item.coefficients, data.template, values)


@final
class _KKTContributions(StrictModule):
    """Retain ordered sparse contributions for one physical incidence family."""

    rows: np.ndarray
    columns: np.ndarray
    indices: np.ndarray
    coefficients: np.ndarray


def _edge_kkt_contributions(
    edge_positions: np.ndarray, dimension: int, /
) -> _KKTContributions:
    rows: list[int] = []
    columns: list[int] = []
    edge_indices: list[tuple[int, int, int]] = []
    edge_signs: list[int] = []
    for index, endpoints in enumerate(edge_positions):
        for first, sign_first in ((int(endpoints[0]), -1), (int(endpoints[1]), 1)):
            for second, sign_second in ((int(endpoints[0]), -1), (int(endpoints[1]), 1)):
                if 2 * first >= dimension or 2 * second >= dimension:
                    continue
                for axis_first in range(2):
                    for axis_second in range(2):
                        rows.append(2 * first + axis_first)
                        columns.append(2 * second + axis_second)
                        edge_indices.append((index, axis_first, axis_second))
                        edge_signs.append(sign_first * sign_second)
    return _KKTContributions(
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(edge_indices, dtype=np.int32).reshape((-1, 3)),
        np.asarray(edge_signs, dtype=np.int32),
    )


def _cell_kkt_contributions(
    cell_positions: np.ndarray, dimension: int, /
) -> _KKTContributions:
    rows: list[int] = []
    columns: list[int] = []
    cell_indices: list[tuple[int, int, int]] = []
    cell_minors: list[int] = []
    for index, roots in enumerate(cell_positions):
        for first, root_first in enumerate(roots):
            for second, root_second in enumerate(roots):
                if 2 * root_first >= dimension or 2 * root_second >= dimension:
                    continue
                orientation = (
                    1
                    if (first, second) in ((0, 1), (1, 2), (2, 0))
                    else -1
                    if (second, first) in ((0, 1), (1, 2), (2, 0))
                    else 0
                )
                for axis_first in range(2):
                    for axis_second in range(2):
                        rows.append(2 * int(root_first) + axis_first)
                        columns.append(2 * int(root_second) + axis_second)
                        cell_indices.append(
                            (index, 2 * first + axis_first, 2 * second + axis_second)
                        )
                        cell_minors.append(orientation * (axis_second - axis_first))
    return _KKTContributions(
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(cell_indices, dtype=np.int32).reshape((-1, 3)),
        np.asarray(cell_minors, dtype=np.int32),
    )


def _equality_kkt_contributions(
    edge_positions: np.ndarray, equality_rows: np.ndarray, dimension: int, /
) -> _KKTContributions:
    rows: list[int] = []
    columns: list[int] = []
    equality_indices: list[tuple[int, int]] = []
    equality_signs: list[int] = []
    for row, edge in enumerate(equality_rows):
        for root, sign in (
            (int(edge_positions[edge, 0]), -1),
            (int(edge_positions[edge, 1]), 1),
        ):
            if 2 * root >= dimension:
                continue
            for axis in range(2):
                rows.extend((dimension + row, 2 * root + axis))
                columns.extend((2 * root + axis, dimension + row))
                equality_indices.extend(((int(edge), axis), (int(edge), axis)))
                equality_signs.extend((sign, sign))
    return _KKTContributions(
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(equality_indices, dtype=np.int32).reshape((-1, 2)),
        np.asarray(equality_signs, dtype=np.int32),
    )


def _prepare_kkt_assembly_data(
    source: _PolishDerivativeSource,
    edge_positions: np.ndarray,
    cell_positions: np.ndarray,
    source_id: str,
    maximum_bytes: int,
    /,
) -> _PeriodicKKTAssemblyData:
    dimension = 2 * source.polish.size.free.size
    equality_rows = np.asarray(source.equality_indices)
    equalities = equality_rows.size
    if np.any(equality_rows >= edge_positions.shape[0]):
        raise ValueError(
            "periodic.kkt.equalities must retain their mutable-edge rows; got an out-of-range equality row"
        )
    contribution_upper = (
        dimension
        + 16 * edge_positions.shape[0]
        + 36 * cell_positions.shape[0]
        + 8 * equalities
        + equalities
    )
    if 64 * contribution_upper > maximum_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "periodic.kkt.incidence must fit the remaining scratch allowance; got a request beyond the allowance",
            requested=(("kkt_incidence_bytes", 64 * contribution_upper),),
            achieved=(("remaining_bytes", maximum_bytes),),
        )
    edge = _edge_kkt_contributions(edge_positions, dimension)
    cell = _cell_kkt_contributions(cell_positions, dimension)
    equality = _equality_kkt_contributions(edge_positions, equality_rows, dimension)
    primal_diagonal = np.arange(dimension, dtype=np.int32)
    equality_diagonal = np.arange(dimension, dimension + equalities, dtype=np.int32)
    rows = np.concatenate(
        (primal_diagonal, edge.rows, cell.rows, equality.rows, equality_diagonal)
    )
    columns = np.concatenate(
        (primal_diagonal, edge.columns, cell.columns, equality.columns, equality_diagonal)
    )
    space = BlockSpace(
        (
            ArraySpace((dimension,), dtype=jnp.float64),
            ArraySpace((equalities,), dtype=jnp.float64),
        )
    )
    template = SparseCoordinateOperator(
        EdgeRelation(
            np.asarray(columns, dtype=np.int32),
            np.asarray(rows, dtype=np.int32),
            source_size=space.size,
            target_size=space.size,
        ),
        jax.device_put(np.zeros(rows.size, dtype=np.float64)),
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "construction"}
        ),
        operator_id=f"{source_id}/actual-sparse-saddle",
    )
    return _PeriodicKKTAssemblyData(
        template,
        jax.device_put(edge.indices),
        jax.device_put(edge.coefficients),
        jax.device_put(cell.indices),
        jax.device_put(cell.coefficients),
        jax.device_put(equality.indices),
        jax.device_put(equality.coefficients),
    )


@final
class _PeriodicSparseKKTSetup(AbstractPrimalDualKKTSetup):
    """Prepare sparse saddle coefficients for native LU congruence."""

    root_ids: Array
    kept_edge_keys: Array
    kkt_assembly: _PeriodicKKTAssemblyData
    assembly_plan: SparseAssemblyPlan
    source_binding_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    derivative_source: _PolishDerivativeSource
    edge_positions: Array
    cell_positions: Array
    local_derivative_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        root_ids: Array,
        kept_edge_keys: Array,
        kkt_assembly: _PeriodicKKTAssemblyData,
        assembly_plan: SparseAssemblyPlan,
        source_binding_id: str,
        operator_id: str,
        derivative_source: _PolishDerivativeSource,
        edge_positions: Array,
        cell_positions: Array,
        local_derivative_id: str,
        /,
    ) -> None:
        if not source_binding_id or not operator_id:
            raise ValueError(
                "periodic.kkt.identity must have nonempty source and operator IDs; got an empty ID"
            )
        self.root_ids = root_ids
        self.kept_edge_keys = kept_edge_keys
        self.kkt_assembly = kkt_assembly
        self.assembly_plan = assembly_plan
        self.source_binding_id = source_binding_id
        self.operator_id = operator_id
        self.derivative_source = derivative_source
        self.edge_positions = edge_positions
        self.cell_positions = cell_positions
        self.local_derivative_id = local_derivative_id

    @property
    def primal_space(self) -> ArraySpace:
        match self.kkt_assembly.template.source:
            case BlockSpace(spaces=(ArraySpace() as primal, *_)):
                return primal
            case _:
                raise TypeError(
                    "periodic.kkt.primal_space must be the prepared ArraySpace primal block; got a different space kind"
                )

    @checked
    def prepare_derivatives(
        self,
        problem: MinimizationProblem,
        layout: ConstraintLayout,
        args: _PolishArgs,
        primal: Array,
        equality_multipliers: Array,
        inequality_multipliers: Array,
        /,
    ) -> tuple[AbstractLinearOperator, AbstractLinearOperator]:
        if problem.problem_id != "periodic-quotient-size-hard-polish":
            raise ValueError(
                "periodic.kkt.problem must bind the periodic hard-polish problem; got a different problem ID"
            )
        primal = eqx.error_if(
            primal,
            ~_same_derivative_source(
                self.derivative_source,
                _polish_derivative_source(args, layout),
            ),
            "periodic.derivative_source must match its prepared scientific binding; got changed source fields",
        )
        current = _polish_derivative_args(
            args, layout, equality_multipliers, inequality_multipliers
        )
        state = _prepare_local_derivative_state(
            primal,
            current,
            self.edge_positions,
            self.cell_positions,
        )
        target = BlockSpace(
            (
                ArraySpace(equality_multipliers.shape, dtype=primal.dtype),
                ArraySpace(inequality_multipliers.shape, dtype=primal.dtype),
            )
        )
        constraint = FunctionLinearOperator(
            eqx.Partial(_local_constraint_mv, state),
            transpose_action=eqx.Partial(_local_constraint_vjp, state),
            source=self.primal_space,
            target=target,
            operator_id=f"{self.local_derivative_id}/canonical-constraints",
            closure_convert=False,
        )
        hessian = FunctionLinearOperator(
            eqx.Partial(_local_hessian_mv, state),
            transpose_action=eqx.Partial(_local_hessian_mv, state),
            source=self.primal_space,
            target=self.primal_space,
            properties=OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
            operator_id=f"{self.local_derivative_id}/true-lagrangian-hessian",
            closure_convert=False,
        )
        return constraint, hessian

    @checked
    def prepare_kkt_operator(
        self,
        derivative: AbstractLinearOperator,
        hessian: AbstractLinearOperator,
        barrier_weights: Array,
        regularization: float,
        space: BlockSpace,
        /,
    ) -> AbstractLinearOperator:
        prepared = _prepare_local_kkt_state(
            _local_constraint_state(derivative),
            barrier_weights,
            regularization,
        )
        return FunctionLinearOperator(
            eqx.Partial(_local_kkt_mv, prepared),
            source=space,
            target=space,
            transpose_action=eqx.Partial(_local_kkt_mv, prepared),
            properties=OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
            operator_id=f"{self.local_derivative_id}/exact-local-saddle-action",
            closure_convert=False,
        )

    def kkt_action_work_upper(self) -> int:
        return (
            24 * self.edge_positions.shape[0]
            + 50 * self.cell_positions.shape[0]
            + 8 * self.primal_space.size
            + 6 * self.derivative_source.equality_indices.size
        )

    def kkt_preparation_work_upper(self) -> int:
        source = self.derivative_source
        return (
            source.lower.size
            + 4 * (source.lower_indices.size + source.upper_indices.size)
            + 16 * self.edge_positions.shape[0]
            + 6 * self.cell_positions.shape[0]
            + self.primal_space.size
        )

    def derivative_preparation_work_upper(self) -> int:
        source = self.derivative_source
        return (
            128 * self.edge_positions.shape[0]
            + 64 * self.cell_positions.shape[0]
            + 16 * self.primal_space.size
            + 4
            * (
                source.equality_indices.size
                + source.lower_indices.size
                + source.upper_indices.size
            )
            + 4 * source.polish.size.origins.size
            + 3 * sum(leaf.size for leaf in jax.tree.leaves(source))
        )

    def sparse_setup_work_upper(self) -> int:
        data, cost = self.kkt_assembly, self.assembly_plan.cost
        return (
            self.primal_space.size
            + 3 * data.edge_indices.shape[0]
            + 8 * data.cell_indices.shape[0]
            + 3 * data.equality_indices.shape[0]
            + self.derivative_source.equality_indices.size
            + cost.maximum_contributions
            + cost.maximum_intermediate_nnz
        )

    def derivative_workspace_bytes_upper(self, itemsize: int) -> int:
        return (
            itemsize
            * (
                40 * self.edge_positions.shape[0]
                + 48 * self.cell_positions.shape[0]
                + 14 * self.primal_space.size
                + 8 * self.derivative_source.lower.size
                + 2 * self.kkt_assembly.template.coefficients.size
            )
            + self.assembly_plan.cost.numeric_workspace_bytes
        )

    @checked
    def prepare(
        self,
        derivative: AbstractLinearOperator,
        barrier_weights: Array,
        regularization: float,
        primal: Array,
        equality: Array,
        space: BlockSpace,
        kkt_operator: AbstractLinearOperator,
        /,
    ) -> PrimalDualKKTSetupResult:
        if not derivative.source.compatible(self.primal_space):
            raise ValueError(
                "periodic.kkt.primal_space must match the prepared primal coordinates; got an incompatible space"
            )
        match kkt_operator:
            case FunctionLinearOperator():
                action = kkt_operator.function
            case _:
                raise TypeError(
                    "periodic.kkt.operator must be the current local saddle FunctionLinearOperator; got a different operator kind"
                )
        match action:
            case eqx.Partial():
                if action.func is not _local_kkt_mv:
                    raise TypeError(
                        "periodic.kkt.action must retain the current local saddle blocks; got a different action"
                    )
            case _:
                raise TypeError(
                    "periodic.kkt.action must retain the current local saddle blocks; got a different action"
                )
        match action.args:
            case (_PeriodicLocalKKTState() as current,):
                raw = _assemble_local_kkt(self.kkt_assembly, current)
            case _:
                raise TypeError(
                    "periodic.kkt.action must retain the current local saddle blocks; got a different action"
                )
        match prepare_sparse_assembly(self.assembly_plan, raw).operator:
            case SparseCoordinateOperator() as assembled:
                pass
            case _:
                raise TypeError(
                    "periodic.kkt.assembly must retain SparseCoordinateOperator coordinates; got a different operator kind"
                )
        status = jnp.where(
            jnp.all(jnp.isfinite(assembled.coefficients)),
            int(LinearSolveStatus.SUCCESS),
            int(LinearSolveStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return PrimalDualKKTSetupResult(
            assembled,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(self.sparse_setup_work_upper(), dtype=jnp.int64),
            status,
        )


def _prepare_polish_derivatives(
    source_binding_id: str,
    free: np.ndarray,
    edges: np.ndarray,
    kept: np.ndarray,
    mutable: np.ndarray,
    args: _PolishArgs,
    maximum_work: int,
    maximum_bytes: int,
    /,
) -> tuple[_PolishDerivativeSource, Array, Array, str, int]:
    """Retain constraint roles and compact physical-root incidence."""
    dimension = 2 * free.size
    cells = np.asarray(args.size.cells)
    mutable_edges = np.flatnonzero(mutable)
    preparation_upper = 32 * (free.size + edges.shape[0] + 3 * cells.shape[0] + dimension)
    storage_upper = 64 * (2 * mutable_edges.size + 3 * cells.shape[0] + dimension)
    if preparation_upper > maximum_work or storage_upper > maximum_bytes:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "periodic.derivative_incidence.resources must fit the remaining work and storage allowances; got a request beyond the allowances",
            requested=(
                ("derivative_work", preparation_upper),
                ("derivative_bytes", storage_upper),
            ),
            achieved=(
                ("remaining_work", maximum_work),
                ("remaining_bytes", maximum_bytes),
            ),
        )
    point = jax.device_put(np.zeros((free.size, 2), dtype=np.float64))
    problem = _polish_problem(mutable_edges, kept, float(np.asarray(args.size.minimum)))
    layout = constraint_layout(problem, point, args)
    position = {int(root): index for index, root in enumerate(free)}
    edge_positions = np.asarray(
        [
            [position.get(int(root), free.size) for root in edge[:2]]
            for edge in edges[mutable_edges]
        ],
        dtype=np.int32,
    ).reshape((-1, 2))
    cell_positions = np.asarray(
        [[position.get(int(root), free.size) for root in cell] for cell in cells],
        dtype=np.int32,
    ).reshape((-1, 3))
    source = _polish_derivative_source(args, layout)
    identity = canonical_fingerprint(
        {
            "kind": "periodic-local-exact-canonical-bound-derivatives",
            "source_binding": source_binding_id,
            "free_roots": array_tree_fingerprint(free),
            "quotient_edges": array_tree_fingerprint(edges),
            "source_cells": array_tree_fingerprint(cells),
            "canonical_equality_rows": array_tree_fingerprint(
                np.asarray(layout.equality_indices)
            ),
            "canonical_lower_rows": array_tree_fingerprint(
                np.asarray(layout.lower_indices)
            ),
            "canonical_upper_rows": array_tree_fingerprint(
                np.asarray(layout.upper_indices)
            ),
            "edge_root_routes": array_tree_fingerprint(edge_positions),
            "cell_root_routes": array_tree_fingerprint(cell_positions),
        }
    )
    return (
        source,
        jax.device_put(edge_positions),
        jax.device_put(cell_positions),
        identity,
        preparation_upper,
    )


def _prepare_polish_method(
    source_binding_id: str,
    free: np.ndarray,
    edges: np.ndarray,
    kept: np.ndarray,
    mutable: np.ndarray,
    maximum_work: int,
    maximum_bytes: int,
    args: _PolishArgs,
    /,
) -> tuple[PrimalDualNewtonKrylov, int]:
    """Prepare sparse saddle support and native LU congruence before JIT."""
    if not source_binding_id:
        raise ValueError(
            "periodic.kkt.source_binding_id must be nonempty; got an empty source binding ID"
        )
    if maximum_work <= 0 or maximum_bytes <= 0:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "periodic.kkt.resources must have positive remaining work and storage allowances; got an exhausted allowance",
        )
    equality_edges = np.flatnonzero(kept)
    if np.any(kept & ~mutable):
        raise ValueError(
            "periodic.kkt.equalities must name mutable quotient edges; got an immutable equality edge"
        )
    (
        derivative_source,
        edge_positions,
        cell_positions,
        local_derivative_id,
        derivative_work,
    ) = _prepare_polish_derivatives(
        source_binding_id,
        free,
        edges,
        kept,
        mutable,
        args,
        maximum_work,
        maximum_bytes,
    )
    identity = canonical_fingerprint(
        {
            "kind": "periodic-hard-current-saddle-lu-congruence",
            "source_binding": source_binding_id,
            "native_free_root_ids": array_tree_fingerprint(free),
            "native_kept_quotient_edge_keys": array_tree_fingerprint(
                edges[equality_edges]
            ),
            "exact_local_derivative_source": local_derivative_id,
        }
    )
    kkt_data = _prepare_kkt_assembly_data(
        derivative_source,
        np.asarray(edge_positions),
        np.asarray(cell_positions),
        identity,
        maximum_bytes,
    )
    assembly = plan_sparse_assembly(
        kkt_data.template,
        SparseAssemblyPolicy(
            max_nnz=max(1, maximum_bytes // 32),
            max_bytes=maximum_bytes,
            max_contributions=max(1, maximum_work // 32),
            max_workspace_bytes=maximum_bytes,
        ),
    )
    assembly_work = (
        32
        * (assembly.cost.maximum_contributions + assembly.nnz)
        * max(1, assembly.nnz.bit_length())
        + 32 * kkt_data.template.relation.capacity
    )
    remaining_work = maximum_work - assembly_work - derivative_work
    if remaining_work <= 0:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "periodic.kkt.preparation.work must fit the remaining work allowance; got a symbolic request beyond the allowance",
            achieved=(("sparse_saddle_symbolic_work_upper", assembly_work),),
        )
    prototype = SparseCoordinateOperator(
        EdgeRelation(
            assembly.column_indices,
            assembly.row_indices,
            source_size=kkt_data.template.source.size,
            target_size=kkt_data.template.target.size,
        ),
        jax.device_put(np.zeros(assembly.nnz, dtype=np.float64)),
        source=kkt_data.template.source,
        target=kkt_data.template.target,
        properties=assembly.properties,
        operator_id=f"{assembly.plan_id}:operator",
    )
    factor_policy = SparseFactorizationPolicy(
        "lu",
        ordering="natural",
        max_factor_nnz=max(1, maximum_bytes // 32),
        max_factor_bytes=maximum_bytes,
        max_symbolic_work=remaining_work,
    )
    factor_plan = prepare_sparse_factorization(prototype, factor_policy)
    builder = SparseFactorizationPreconditionerBuilder(
        factor_policy,
        prepared_plan=factor_plan,
        setup_operator=prototype,
        form="lu-congruence",
    )
    setup = _PeriodicSparseKKTSetup(
        jax.device_put(np.asarray(free, dtype=np.int64)),
        jax.device_put(np.asarray(edges[equality_edges], dtype=np.int64)),
        kkt_data,
        assembly,
        source_binding_id,
        identity,
        derivative_source,
        edge_positions,
        cell_positions,
        local_derivative_id,
    )
    policy = eqx.tree_at(
        lambda item: item.preconditioning,
        _POLISH_METHOD.linear_policy,
        PreconditioningPolicy(builder),
    )
    method = eqx.tree_at(
        lambda item: (item.linear_policy, item.kkt_setup),
        _POLISH_METHOD,
        (policy, setup),
        is_leaf=lambda item: item is None,
    )
    return method, assembly_work + factor_plan.symbolic_work + derivative_work


def _points(parameters: Array, args: _SizeArgs, /) -> Array:
    return args.origins.at[args.free].add(args.target * parameters)


def _lengths(parameters: Array, args: _SizeArgs, /) -> Array:
    points = _points(parameters, args)
    delta = points[args.edges[:, 1]] - points[args.edges[:, 0]] + args.edge_offsets
    return jax.vmap(_local_edge_length, in_axes=(0, None))(delta, args.target)


def _mutable_lengths(parameters: Array, args: _SizeArgs, /) -> Array:
    return jnp.where(args.mutable, _lengths(parameters, args), 0.0)


def _mutable_minimum_gaps(parameters: Array, args: _SizeArgs, /) -> Array:
    return jnp.where(args.mutable, _lengths(parameters, args) - args.minimum, 0.0)


def _cell_areas(parameters: Array, args: _SizeArgs, /) -> Array:
    points = _points(parameters, args)
    corners = (points[args.cells] + args.cell_offsets) / args.target
    return jax.vmap(_local_cell_area)(corners)


def _negative_mean_length(parameters: Array, args: _SizeArgs, /) -> Array:
    return -jnp.sum(_mutable_lengths(parameters, args)) / jnp.sum(args.mutable)


_EXTREME_PROBLEM = MinimizationProblem(
    _negative_mean_length,
    bounds=Bounds(-1.0, 1.0),
    constraints=(
        NonlinearConstraint(
            _mutable_lengths, upper=1.0, constraint_id="periodic-size-target-ceiling"
        ),
        NonlinearConstraint(
            _mutable_minimum_gaps, lower=0.0, constraint_id="periodic-size-minimum"
        ),
        NonlinearConstraint(
            _cell_areas, lower=_AREA_MARGIN, constraint_id="periodic-positive-cell-areas"
        ),
    ),
    problem_id="periodic-quotient-size-extreme-point",
)


def _polish_objective(parameters: Array, args: _PolishArgs, /) -> Array:
    return _negative_mean_length(parameters, args.size)


def _polish_edge_lengths(parameters: Array, args: _PolishArgs, /) -> Array:
    points = _points(parameters, args.size)
    rows = args.size.edges[args.mutable_edges]
    delta = (
        points[rows[:, 1]]
        - points[rows[:, 0]]
        + args.size.edge_offsets[args.mutable_edges]
    )
    return jax.vmap(_local_edge_length, in_axes=(0, None))(delta, args.size.target)


def _polish_areas(parameters: Array, args: _PolishArgs, /) -> Array:
    return _cell_areas(parameters, args.size)


def _polish_problem(
    mutable: np.ndarray, kept: np.ndarray, minimum: float, /
) -> MinimizationProblem:
    # Each row is an mutable quotient-edge owner. Kept target rows have
    # equal bounds; all other rows retain both size inequalities.
    return MinimizationProblem(
        _polish_objective,
        bounds=Bounds(-1.0, 1.0),
        constraints=(
            NonlinearConstraint(
                _polish_edge_lengths,
                lower=np.where(kept[mutable], 1.0, minimum),
                upper=np.where(kept[mutable], 1.0, 1.0 - 2.0 * _DEMOTION),
                constraint_id="periodic-quotient-size-edge-bounds",
            ),
            NonlinearConstraint(
                _polish_areas,
                lower=0.5 * _AREA_MARGIN,
                constraint_id="periodic-positive-cell-areas",
            ),
        ),
        problem_id="periodic-quotient-size-hard-polish",
    )


class _SolveRecord(NamedTuple):
    parameters: Array
    status: Array
    iterations: Array
    value_evaluations: Array
    actions: Array
    damping: Array
    accepted_steps: Array
    rejected_steps: Array
    linear_solves: Array
    linear_iterations: Array
    evidence: PrimalDualEvidence | None
    counts_complete: bool
    primal_feasibility: Array
    dual_feasibility: Array
    complementarity: Array


@eqx.filter_jit
def _solve_extreme(seed: Array, args: _SizeArgs, /) -> _SolveRecord:
    result = minimize(
        _EXTREME_PROBLEM,
        seed,
        method=_EXTREME_METHOD,
        termination=_EXTREME_TERMINATION,
        args=args,
    )
    diagnostics = result.diagnostics
    values = diagnostics.objective_evaluations + diagnostics.constraint_evaluations
    # Gradients and line-search derivatives are counted as three actions each.
    actions = diagnostics.objective_evaluations + 3 * (
        diagnostics.gradient_evaluations + diagnostics.hvp_evaluations
    )
    actions += diagnostics.jvp_evaluations + diagnostics.vjp_evaluations
    return _SolveRecord(
        result.parameters,
        result.status,
        diagnostics.iterations,
        values,
        actions,
        diagnostics.damping,
        diagnostics.accepted_steps,
        diagnostics.rejected_steps,
        diagnostics.linear_solves,
        diagnostics.linear_iterations,
        None,
        diagnostics.counts_complete,
        diagnostics.primal_feasibility,
        diagnostics.dual_feasibility,
        diagnostics.complementarity,
    )


@eqx.filter_jit
def _solve_polish(
    parameters: Array,
    args: _PolishArgs,
    problem: MinimizationProblem,
    method: PrimalDualNewtonKrylov,
    /,
) -> _SolveRecord:
    result = minimize(
        problem,
        parameters,
        method=method,
        termination=_POLISH_TERMINATION,
        args=args,
    )
    diagnostics = result.diagnostics
    values = diagnostics.objective_evaluations + diagnostics.constraint_evaluations
    actions = values + 3 * (
        diagnostics.gradient_evaluations + diagnostics.hvp_evaluations
    )
    actions += diagnostics.jvp_evaluations + diagnostics.vjp_evaluations
    match result.method_evidence:
        case PrimalDualEvidence() as evidence:
            pass
        case _:
            raise TypeError(
                "periodic.polish.evidence must be PrimalDualEvidence; got missing or incompatible method evidence"
            )
    return _SolveRecord(
        result.parameters,
        result.status,
        diagnostics.iterations,
        values,
        actions,
        diagnostics.damping,
        diagnostics.accepted_steps,
        diagnostics.rejected_steps,
        diagnostics.linear_solves,
        diagnostics.linear_iterations,
        evidence,
        diagnostics.counts_complete,
        diagnostics.primal_feasibility,
        diagnostics.dual_feasibility,
        diagnostics.complementarity,
    )


def periodic_reference_lifts(
    simplices: np.ndarray, shifts: np.ndarray, count: int, /
) -> np.ndarray:
    """Lattice image published as each root's quotient representative.

    This is the representative rule of ``publish_periodic_simplices``: the
    zero-shift corner when present, else the lexicographically smallest one.
    Length evidence is evaluated from these published coordinates.
    """
    dimension = shifts.shape[-1]
    corners = np.unique(
        np.concatenate(
            (simplices.reshape((-1, 1)), shifts.reshape((-1, dimension))), axis=1
        ),
        axis=0,
    )
    reference = np.searchsorted(corners[:, 0], np.arange(count))
    zero = np.flatnonzero(np.all(corners[:, 1:] == 0, axis=1))
    reference[corners[zero, 0]] = zero
    return corners[reference, 1:]


def quotient_edge_lengths(
    points: np.ndarray, edges: np.ndarray, lifts: np.ndarray, vectors: np.ndarray, /
) -> np.ndarray:
    """Quotient edge lengths with the published coordinate and shift arithmetic."""
    coordinates = points + lifts.astype(np.float64) @ vectors
    relative = edges[:, 2:] - lifts[edges[:, 1]] + lifts[edges[:, 0]]
    return np.linalg.norm(
        coordinates[edges[:, 1]] - coordinates[edges[:, 0]] + relative @ vectors, axis=1
    )


def _positive_scientific_cells(
    points: np.ndarray, cells: np.ndarray, shifts: np.ndarray, vectors: np.ndarray, /
) -> bool:
    packed, _ = dyadic_integers(np.concatenate((points, vectors)))
    corners = packed[: points.shape[0]][cells] + shifts @ packed[points.shape[0] :]
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    return bool(np.all(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0] > 0))


def _required_exact_edges(
    lengths: np.ndarray,
    immutable: np.ndarray,
    target: float,
    statistics: tuple[SizeStatistic, ...],
    /,
) -> int | None:
    """Mutable target-length edges needed when every mutable edge is at most the target.

    Every sorted position between the lowest and highest requested rank must
    hold the target value; immutable over-target edges must fit above them.
    """
    count = lengths.size
    ranks = [size_statistic_quantile(name) * (count - 1) for name in statistics]
    lowest, highest = int(np.floor(min(ranks))), int(np.ceil(max(ranks)))
    fixed = lengths[immutable]
    if np.count_nonzero(fixed > target) > count - 1 - highest:
        return None
    mutable = count - fixed.size
    required = mutable + np.count_nonzero(fixed < target) - lowest
    return max(int(required), 0) if required <= mutable else None


def _fixed_rank_feasible(
    lengths: np.ndarray,
    fixed: np.ndarray,
    target: float,
    quantiles: tuple[float, ...],
    /,
) -> bool:
    """Necessary fixed-edge condition under the unmodified floating reducer.

    When the requested rank intervals are separated, equal quantiles force
    their intervening sorted rows to the common target. Replacing every free
    row by that target is therefore an outward feasibility bank, including
    interpolation that rounds to the target from an adjacent endpoint.
    This is not a certificate that any geometric placement exists.
    """
    ranks = [quantile * (lengths.size - 1) for quantile in quantiles]
    if np.ceil(min(ranks)) > np.floor(max(ranks)):
        return True
    optimistic = np.where(fixed, lengths, target)
    return bool(np.all(np.quantile(optimistic, quantiles) == target))


def _rank_window_quantiles(
    total_count: int,
    unchanged: np.ndarray,
    candidates: np.ndarray,
    quantiles: tuple[float, ...],
    charge: Callable[[int], None],
    /,
) -> np.ndarray:
    """Evaluate the full linear quantiles from a proved local rank window.

    With ``k`` replaced edges, an unchanged row at global rank ``r`` can only
    come from base ranks ``r-k`` through ``r``. All earlier unchanged rows
    precede the requested rank, regardless of the replacement values. Retain
    those windows and every changed row, then use NumPy's unmodified two-value
    quantile interpolation for the full-bank fractional rank.
    """
    changed, proposals = candidates.shape
    values = []
    for quantile in quantiles:
        rank = quantile * (total_count - 1)
        lower, upper = int(np.floor(rank)), int(np.ceil(rank))
        start = max(0, lower - changed)
        stop = min(unchanged.size, upper + 1)
        rows = stop - start + changed
        charge(4 * proposals * (rows + 2) * (max(1, rows.bit_length()) + 2))
        window = np.concatenate(
            (
                np.broadcast_to(unchanged[start:stop, None], (stop - start, proposals)),
                candidates,
            ),
            axis=0,
        )
        selected = np.partition(window, (lower - start, upper - start), axis=0)
        pair = np.stack((selected[lower - start], selected[upper - start]))
        values.append(np.quantile(pair, rank - lower, axis=0))
    return np.stack(values)


def _placement_order(
    free: np.ndarray,
    edges: np.ndarray,
    active: np.ndarray,
    closeness: np.ndarray,
    protected: np.ndarray,
    /,
) -> tuple[list[int], np.ndarray]:
    """Order roots so each keeps at most two active edges to earlier roots."""
    incident: list[list[int]] = [[] for _ in range(protected.size)]
    for edge in np.flatnonzero(active):
        first, second = (int(value) for value in edges[edge, :2])
        incident[first].append(int(edge))
        incident[second].append(int(edge))
    placed = protected.copy()
    kept = np.zeros(edges.shape[0], dtype=np.bool_)
    remaining = sorted(int(value) for value in free)
    order: list[int] = []
    # A newly placed root is the only owner that can add predecessors. Keep
    # those incidence lists instead of rescanning every remaining root's whole
    # adjacency at every placement. Selection below retains the root
    # tie order and the (closeness, edge) ordering.
    backs = {
        root: [
            edge
            for edge in incident[root]
            if placed[edges[edge, 1] if edges[edge, 0] == root else edges[edge, 0]]
        ]
        for root in remaining
    }
    # Priorities retain the stable ascending root tie break of the former
    # scan. Predecessor counts only increase, so an obsolete heap entry is
    # identified by its count without a separate generation bank.
    priorities = [
        (-min(len(backs[root]), 2), len(backs[root]), root) for root in remaining
    ]
    heapify(priorities)
    while priorities:
        _, predecessors, root = heappop(priorities)
        if placed[root] or predecessors != len(backs[root]):
            continue
        selected = sorted(backs.pop(root), key=lambda edge: (closeness[edge], edge))[:2]
        kept[selected] = True
        placed[root] = True
        order.append(root)
        for edge in incident[root]:
            other = int(edges[edge, 1] if edges[edge, 0] == root else edges[edge, 0])
            if not placed[other]:
                backs[other].append(edge)
                count = len(backs[other])
                heappush(priorities, (-min(count, 2), count, other))
    return order, kept


def _rank_placement_edges(
    free: np.ndarray,
    edges: np.ndarray,
    normalized: np.ndarray,
    mutable: np.ndarray,
    protected: np.ndarray,
    required: int,
    /,
    *,
    admit_bank: Callable[[int], None] | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Select a realizable target-rank bank, not an optimizer stopping tolerance.

    A finite AL termination can leave required upper constraints just below the
    target. Rank those mutable constraints by distance, then enlarge the
    polish bank until its two-predecessor graph can realize the requested
    cardinality. LM and representable placement still own all numerical proof.
    """
    closeness = np.abs(normalized - 1.0)
    active = mutable & (normalized >= 1.0 - _ACTIVE_GAP)
    if admit_bank is not None:
        admit_bank(int(np.count_nonzero(active)))
    order, kept = _placement_order(free, edges, active, closeness, protected)
    if np.count_nonzero(kept) >= required:
        return order, active, kept
    candidates = np.flatnonzero(mutable & ~active)
    ranked = candidates[np.lexsort((candidates, closeness[candidates]))]
    for edge in ranked:
        active[edge] = True
        if admit_bank is not None:
            admit_bank(int(np.count_nonzero(active)))
        order, kept = _placement_order(free, edges, active, closeness, protected)
        if np.count_nonzero(kept) >= required:
            break
    return order, active, kept


class _ExactPlacement:
    """Conflict-directed search taking ownership of one private point carrier."""

    def __init__(
        self,
        points: np.ndarray,
        edges: np.ndarray,
        lifts: np.ndarray,
        vectors: np.ndarray,
        protected: np.ndarray,
        kept: np.ndarray,
        target: float,
        minimum: float,
        maximum_work: int,
        check: Callable[[], None],
        /,
        *,
        edge_lower: np.ndarray | None = None,
        edge_upper: np.ndarray | None = None,
    ) -> None:
        self.points, self.edges, self.protected = points, edges, protected
        self.kept, self.target, self.minimum = kept, target, minimum
        self.edge_lower = edge_lower
        self.edge_upper = edge_upper
        self.lift = lifts.astype(np.float64) @ vectors
        self.relative = (edges[:, 2:] - lifts[edges[:, 1]] + lifts[edges[:, 0]]) @ vectors
        self.coordinates = points + self.lift
        # Slides and grids are always anchored at the polished position, so
        # repeated back-jumps cannot accumulate drift beyond the demotion margin.
        self.base = self.coordinates.copy()
        self.placed = protected.copy()
        self.incident: list[list[int]] = [[] for _ in range(points.shape[0])]
        for edge, (first, second) in enumerate(edges[:, :2]):
            if first != second:
                self.incident[int(first)].append(edge)
                self.incident[int(second)].append(edge)
        self.maximum_work, self.check = maximum_work, check
        self.work = 0
        self.nodes = 0
        self.rank_quantiles: tuple[float, ...] = ()
        self.rank_error = np.inf
        self.rank_exact = False

    def _other(self, edge: int, root: int, /) -> int:
        first, second = (int(value) for value in self.edges[edge, :2])
        return second if first == root else first

    def _center(self, edge: int, root: int, /) -> np.ndarray:
        first, second = (int(value) for value in self.edges[edge, :2])
        if first == root:
            return self.coordinates[second] + self.relative[edge]
        return self.coordinates[first] - self.relative[edge]

    def _targets(self, root: int, /) -> Iterator[np.ndarray]:
        """Continuous lifted positions: circle intersection, circle slides or free offsets."""
        base = self.base[root]
        kept = [
            edge
            for edge in self.incident[root]
            if self.kept[edge] and self.placed[self._other(edge, root)]
        ]
        step = _SLIDE_STEP * self.target
        if len(kept) == 2:
            first, second = self._center(kept[0], root), self._center(kept[1], root)
            chord = second - first
            distance = float(np.linalg.norm(chord))
            if 0.0 < distance < 2.0 * self.target:
                normal = (
                    np.asarray((-chord[1], chord[0]))
                    / distance
                    * np.sqrt(self.target * self.target - 0.25 * distance * distance)
                )
                roots = (
                    0.5 * (first + second) + normal,
                    0.5 * (first + second) - normal,
                )
                yield min(roots, key=lambda value: float(np.linalg.norm(value - base)))
            return
        if len(kept) == 1:
            center = self._center(kept[0], root)
            radial = (base - center) / np.linalg.norm(base - center)
            tangent = np.asarray((-radial[1], radial[0]))
            yield center + self.target * radial
            for slide in range(1, _SLIDES + 1):
                for sign in (1.0, -1.0):
                    angle = sign * slide * step / self.target
                    yield center + self.target * (
                        np.cos(angle) * radial + np.sin(angle) * tangent
                    )
            return
        yield base
        for ring in range(1, _FREE_RINGS + 1):
            for x in range(-ring, ring + 1):
                for y in (-ring, ring) if abs(x) < ring else range(-ring, ring + 1):
                    yield base + step * np.asarray((x, y), dtype=np.float64)

    def _grid(self, root: int, targets: list[np.ndarray], /) -> np.ndarray:
        """Representable representatives around each target, lifted as published."""
        representatives = np.stack(targets) - self.lift[root]
        lower, upper = representatives.copy(), representatives.copy()
        axes = [representatives]
        for _ in range(_GRID_RADIUS):
            lower, upper = np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)
            axes.extend((lower, upper))
        values = np.stack(axes, axis=1)
        x = np.broadcast_to(values[:, :, None, 0], (len(targets), len(axes), len(axes)))
        y = np.broadcast_to(values[:, None, :, 1], (len(targets), len(axes), len(axes)))
        return np.stack((x, y), axis=-1).reshape((-1, 2))

    def _spend(self, cost: int, /) -> None:
        if self.work + cost > self.maximum_work:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "periodic.placement.work must fit the remaining work budget; got a candidate request beyond the budget",
                requested=(("remaining_work", self.maximum_work),),
                achieved=(
                    ("placement_work", self.work),
                    ("placement_request", cost),
                    ("placement_nodes", self.nodes),
                ),
            )
        self.work += cost

    def _admissible(
        self,
        root: int,
        representatives: np.ndarray,
        /,
        *,
        record_lengths: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        constrained = [
            edge for edge in self.incident[root] if self.placed[self._other(edge, root)]
        ]
        self._spend(representatives.shape[0] * (len(constrained) + 1))
        lifted = representatives + self.lift[root]
        valid = np.ones(lifted.shape[0], dtype=np.bool_)
        for edge in constrained:
            other = self._other(edge, root)
            if int(self.edges[edge, 0]) == root:
                delta = self.coordinates[other] - lifted + self.relative[edge]
            else:
                delta = lifted - self.coordinates[other] + self.relative[edge]
            length = np.linalg.norm(delta, axis=1)
            if record_lengths is not None:
                record_lengths.append(length)
            if self.kept[edge]:
                valid &= length == self.target
            else:
                lower = self.minimum if self.edge_lower is None else self.edge_lower[edge]
                upper = self.target if self.edge_upper is None else self.edge_upper[edge]
                valid &= (length <= upper) & (length >= lower)
        return valid

    def _candidates(self, root: int, /) -> Iterator[np.ndarray]:
        targets = self._targets(root)
        while True:
            batch = [value for _, value in zip(range(_BATCH), targets, strict=False)]
            if not batch:
                return
            self.check()
            grid = self._grid(root, batch)
            yield from grid[self._admissible(root, grid)]

    def _neighbors(self, root: int, /) -> set[int]:
        return {
            other
            for edge in self.incident[root]
            if not self.protected[other := self._other(edge, root)] and self.placed[other]
        }

    def search(self, order: list[int], /) -> bool:
        positions = {root: index for index, root in enumerate(order)}
        domains: list[Iterator[np.ndarray] | None] = [None] * len(order)
        conflicts: list[set[int]] = [set() for _ in order]
        index = 0
        while index < len(order):
            root = order[index]
            domain = domains[index]
            if domain is None:
                domain = domains[index] = self._candidates(root)
                conflicts[index] = {positions[other] for other in self._neighbors(root)}
            representative = next(domain, None)
            if representative is not None:
                self.nodes += 1
                self.points[root] = representative
                self.coordinates[root] = representative + self.lift[root]
                self.placed[root] = True
                index += 1
                continue
            if not conflicts[index]:
                return False
            jump = max(conflicts[index])
            conflicts[jump] |= conflicts[index] - {jump}
            for later in range(jump, index + 1):
                self.placed[order[later]] = False
                if later > jump:
                    domains[later] = None
            index = jump
        return True

    def _local_circle(
        self,
        root: int,
        edge: int,
        lengths: np.ndarray,
        unchanged: np.ndarray,
        below: int,
        above: int,
        allowed_above: int,
        lowest: int,
        anchor: int | None,
        /,
    ) -> tuple[int, int] | None:
        incident = self.incident[root]
        self._spend(8 * len(incident) + 2)
        old_below = int(np.count_nonzero(lengths[incident] < self.target))
        old_above = int(np.count_nonzero(lengths[incident] > self.target))
        self.placed[root] = False
        self.kept[edge] = True
        if anchor is not None:
            self.kept[anchor] = True
        try:
            targets = self._targets(root)
            batch = [value for _, value in zip(range(_BATCH), targets, strict=False)]
            if not batch:
                return None
            self.check()
            grid = self._grid(root, batch)
            bank: list[np.ndarray] = []
            valid = self._admissible(root, grid, record_lengths=bank)
            self._spend(4 * grid.shape[0] * (len(incident) + 1))
            values = np.stack(bank)
            candidate_below = (
                below - old_below + np.count_nonzero(values < self.target, axis=0)
            )
            candidate_above = (
                above - old_above + np.count_nonzero(values > self.target, axis=0)
            )
            unmatched = candidate_below + candidate_above
            deficit = np.maximum(0, candidate_below - lowest) + np.maximum(
                0, candidate_above - allowed_above
            )
            previous_deficit = max(0, below - lowest) + max(0, above - allowed_above)
            # One representative can repair only its incident edges. Permit a
            # monotone unresolved tail while the existing strict merit improves;
            # final admission still requires both unmodified quantile targets.
            valid &= candidate_above <= max(above, allowed_above)
            valid &= (
                np.max(np.abs(grid + self.lift[root] - self.base[root]), axis=1)
                <= _SLIDES * _SLIDE_STEP * self.target
            )
            choices = np.flatnonzero(valid)
            if not choices.size:
                return None
            measured = _rank_window_quantiles(
                lengths.size,
                unchanged,
                values[:, choices],
                self.rank_quantiles,
                self._spend,
            )
            errors = np.sum(((measured - self.target) / self.target) ** 2, axis=0)
            improves = (
                (errors < self.rank_error)
                | ((errors == self.rank_error) & (deficit[choices] < previous_deficit))
                | (
                    (errors == self.rank_error)
                    & (deficit[choices] == previous_deficit)
                    & (unmatched[choices] < below + above)
                )
            )
            selected = np.flatnonzero(improves)
            if not selected.size:
                return None
            selected_index = min(
                selected,
                key=lambda index: (
                    errors[index],
                    deficit[choices[index]],
                    unmatched[choices[index]],
                    candidate_below[choices[index]],
                    candidate_above[choices[index]],
                    index,
                ),
            )
            chosen = choices[selected_index]
            self.points[root] = grid[chosen]
            self.coordinates[root] = grid[chosen] + self.lift[root]
            lengths[incident] = values[:, chosen]
            self.nodes += 1
            self.rank_error = float(errors[selected_index])
            self.rank_exact = bool(np.all(measured[:, selected_index] == self.target))
            return int(candidate_below[chosen]), int(candidate_above[chosen])
        finally:
            self.placed[root] = True
            self.kept[edge] = False
            if anchor is not None:
                self.kept[anchor] = False

    def search_local_ranks(
        self,
        lengths: np.ndarray,
        lowest: int,
        highest: int,
        /,
        *,
        statistics: tuple[SizeStatistic, ...],
    ) -> bool:
        """Correct target-rank ties with the common circle/representable owner."""
        below = int(np.count_nonzero(lengths < self.target))
        above = int(np.count_nonzero(lengths > self.target))
        allowed_above = lengths.size - 1 - highest
        self.placed[:] = True
        self.rank_quantiles = tuple(size_statistic_quantile(name) for name in statistics)
        measured = np.quantile(lengths, self.rank_quantiles)
        self.rank_error = float(np.sum(((measured - self.target) / self.target) ** 2))
        self.rank_exact = bool(np.all(measured == self.target))
        while not self.rank_exact:
            self._spend(lengths.size * (max(1, lengths.size.bit_length()) + 1))
            closeness = np.abs(lengths - self.target)
            measured = np.quantile(lengths, self.rank_quantiles)
            unchanged_banks: dict[int, np.ndarray] = {}
            priority = ((lengths < self.target) & np.any(measured < self.target)) | (
                (lengths > self.target) & np.any(measured > self.target)
            )
            order = np.lexsort(
                (np.arange(lengths.size, dtype=np.int64), closeness, ~priority)
            )
            changed = False
            for edge in order:
                limit = self.target * _ACTIVE_GAP
                if not 0.0 < closeness[edge] <= limit:
                    continue
                roots = sorted(
                    (int(value) for value in self.edges[edge, :2]),
                    key=lambda root: (
                        float(np.max(np.abs(np.spacing(self.points[root])))),
                        root,
                    ),
                )
                for root in roots:
                    if self.protected[root] or self.edges[edge, 0] == self.edges[edge, 1]:
                        continue
                    if root not in unchanged_banks:
                        self._spend(
                            4 * lengths.size * (max(1, lengths.size.bit_length()) + 2)
                        )
                        mask = np.ones(lengths.size, dtype=np.bool_)
                        mask[self.incident[root]] = False
                        bank = np.sort(lengths[mask], kind="stable")
                        bank.setflags(write=False)
                        unchanged_banks[root] = bank
                    anchors: list[int | None] = [
                        candidate
                        for candidate in self.incident[root]
                        if candidate != edge and lengths[candidate] == self.target
                    ]
                    anchors.append(None)
                    for anchor in anchors:
                        found = self._local_circle(
                            root,
                            int(edge),
                            lengths,
                            unchanged_banks[root],
                            below,
                            above,
                            allowed_above,
                            lowest,
                            anchor,
                        )
                        if found is not None:
                            below, above = found
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if not changed:
                return False
        return True


def _seed(points: np.ndarray, free: np.ndarray, vectors: np.ndarray, /) -> np.ndarray:
    """Geometry-addressed symmetry-breaking start inside the bounded move box."""
    fractional = np.linalg.solve(vectors.T, points[free].T).T
    return 0.01 * np.stack(
        (
            np.sin(2.0 * np.pi * (fractional[:, 0] + 2.0 * fractional[:, 1])),
            np.cos(2.0 * np.pi * (2.0 * fractional[:, 0] - fractional[:, 1])),
        ),
        axis=1,
    )


class PeriodicPolishRecord(NamedTuple):
    method_id: str
    status: int
    iterations: int
    value_evaluations: int
    reported_actions: int
    accepted_steps: int
    rejected_steps: int
    linear_solves: int
    linear_iterations: int
    linear_status: int
    linear_rank: int
    linear_condition_estimate: float
    linear_residual_norm: float
    last_linear_matvec_count: int
    counts_complete: bool
    barrier: float
    primal_feasibility: float
    dual_feasibility: float
    complementarity: float
    preconditioner_id: str | None
    setup_id: str | None
    setup_jvp_evaluations: int
    setup_work_units_upper: int
    setup_status: int
    factorization_status: int
    factor_minimum_pivot: float
    factor_replaced_pivots: float
    factor_dropped_entries: float
    factor_nonzeros: float
    factor_finite: float


class PeriodicSizeRecord(NamedTuple):
    method_id: str
    status: int
    iterations: int
    value_evaluations: int
    reported_actions: int
    charged_work_bound: int
    polish_status: int
    required_exact_edges: int
    kept_exact_edges: int
    placement_nodes: int
    polish: PeriodicPolishRecord | None


@final
class PeriodicSizeProposal(StrictModule):
    points: np.ndarray | None
    work_units: int
    method_id: str
    status: int
    iterations: int
    value_evaluations: int
    reported_actions: int
    polish_status: int
    required_exact_edges: int
    kept_exact_edges: int
    placement_nodes: int
    polish: PeriodicPolishRecord | None

    @checked
    def __init__(
        self,
        points: np.ndarray | None,
        work_units: int,
        method_id: str,
        status: int,
        iterations: int,
        value_evaluations: int,
        reported_actions: int,
        polish_status: int,
        required_exact_edges: int,
        kept_exact_edges: int,
        placement_nodes: int,
        polish: PeriodicPolishRecord | None,
    ) -> None:
        self.points, self.work_units, self.method_id = points, work_units, method_id
        self.status, self.iterations = status, iterations
        self.value_evaluations, self.reported_actions = (
            value_evaluations,
            reported_actions,
        )
        self.polish_status = polish_status
        self.required_exact_edges, self.kept_exact_edges = (
            required_exact_edges,
            kept_exact_edges,
        )
        self.placement_nodes, self.polish = placement_nodes, polish

    @property
    def record(self) -> PeriodicSizeRecord:
        return PeriodicSizeRecord(
            self.method_id,
            self.status,
            self.iterations,
            self.value_evaluations,
            self.reported_actions,
            self.work_units,
            self.polish_status,
            self.required_exact_edges,
            self.kept_exact_edges,
            self.placement_nodes,
            self.polish,
        )


PERIODIC_SIZE_EVIDENCE_FIELDS = (
    "status",
    "iterations",
    "value_evaluations",
    "reported_actions",
    "charged_work_bound",
    "polish_status",
    "required_exact_edges",
    "kept_exact_edges",
    "placement_nodes",
)


def _polish_record(
    result: _SolveRecord, method: PrimalDualNewtonKrylov, /
) -> PeriodicPolishRecord:
    evidence = result.evidence
    if evidence is None:
        raise ValueError(
            "periodic.polish.evidence must contain KKT solve evidence; got no evidence"
        )
    preconditioning = method.linear_policy.preconditioning
    builder = None if preconditioning is None else preconditioning.builder
    setup = method.kkt_setup
    diagnostics = evidence.factorization_diagnostics
    return PeriodicPolishRecord(
        method.method_id,
        int(result.status),
        int(result.iterations),
        int(result.value_evaluations),
        int(result.actions),
        int(result.accepted_steps),
        int(result.rejected_steps),
        int(result.linear_solves),
        int(result.linear_iterations),
        int(evidence.linear_status),
        int(evidence.linear_rank),
        float(evidence.linear_condition_estimate),
        float(evidence.linear_residual_norm),
        int(evidence.linear_matvec_count),
        result.counts_complete,
        float(result.damping),
        float(result.primal_feasibility),
        float(result.dual_feasibility),
        float(result.complementarity),
        None if builder is None else builder.builder_id,
        None if not isinstance(setup, _PeriodicSparseKKTSetup) else setup.operator_id,
        int(evidence.setup_jvp_evaluations),
        int(evidence.setup_work_units_upper),
        int(evidence.setup_status),
        int(evidence.factorization_status),
        np.nan if diagnostics is None else float(diagnostics.minimum_pivot),
        np.nan if diagnostics is None else float(diagnostics.replaced_pivots),
        np.nan if diagnostics is None else float(diagnostics.dropped_entries),
        np.nan if diagnostics is None else float(diagnostics.factor_nonzeros),
        np.nan if diagnostics is None else float(diagnostics.finite),
    )


def _polish_sparse_owners(
    method: PrimalDualNewtonKrylov,
    /,
) -> tuple[_PeriodicSparseKKTSetup, SparseFactorizationPlan]:
    setup = method.kkt_setup
    preconditioning = method.linear_policy.preconditioning
    if not isinstance(setup, _PeriodicSparseKKTSetup) or preconditioning is None:
        raise TypeError(
            "periodic.polish.preparation must retain the sparse setup and preconditioning policy; got an absent or different setup"
        )
    builder = preconditioning.builder
    if (
        not isinstance(builder, SparseFactorizationPreconditionerBuilder)
        or builder.form != "lu-congruence"
    ):
        raise TypeError(
            "periodic.polish.preconditioner must be the prepared LU congruence builder; got a different builder or form"
        )
    plan = builder.prepared_plan
    if plan is None:
        raise ValueError(
            "periodic.polish.factor_plan must be prepared before execution admission; got no prepared plan"
        )
    return setup, plan


def _polish_work_bounds(
    points: int,
    edges: int,
    cells: int,
    dimension: int,
    mutable: int,
    equalities: int,
    method: PrimalDualNewtonKrylov,
    /,
) -> tuple[tuple[str, int], ...]:
    """Disjoint worst-case phases, excluding symbolically empty pivot windows.

    Numeric factors still refresh at every current saddle: geometry,
    multipliers and barrier weights change their values. Only the immutable
    lower-column support earns the skipped rank-one updates; neither observed
    early termination nor a prior factor action earns reusable solve work.
    """
    inequalities = cells + 2 * (mutable - equalities) + 2 * dimension
    kkt_dimension = dimension + equalities
    nonlinear_work = edges + mutable + 4 * cells + points
    constraint_work = mutable + 4 * cells + points + equalities + inequalities
    # The fused KKT owner applies one prepared HVP, JVP and VJP. Retain
    # worst-case confirmation of every MINRES iterate, not observed early stops.
    steps = _POLISH_LINEAR_ITERATIONS_PER_TRIAL
    iterations = _POLISH_TERMINATION.maximum_steps
    refreshes = iterations + 1
    setup, plan = _polish_sparse_owners(method)
    action_work = setup.kkt_action_work_upper()
    derivative_preparation_work = (
        setup.derivative_preparation_work_upper() + setup.kkt_preparation_work_upper()
    )
    solve_work = plan.lu_congruence_solve_work_units_upper_for(
        setup.kkt_assembly.template.coefficients.dtype,
        coordinate_dtype(setup.primal_space),
    )
    # Empty strictly-lower columns bypass the numeric rank-one window. Keep
    # scalar/pivot and row work for every pivot, including those skipped columns.
    active_columns = int(np.count_nonzero(np.diff(np.asarray(plan.column_offsets))))
    factor_work = 16 * (
        active_columns * (plan.row_width * max(1, plan.column_width) + plan.column_width)
        + plan.shape[0] * (plan.row_width + 1)
    )
    return (
        (
            "initial",
            32 * nonlinear_work
            + 16 * constraint_work
            + 16 * (kkt_dimension + inequalities + dimension),
        ),
        ("kkt_actions", iterations * (2 * steps + 3) * action_work),
        (
            "minres_recurrence",
            iterations * ((19 * steps + 12) * kkt_dimension + 128 * (steps + 1)),
        ),
        ("trial_batches", iterations * 5 * nonlinear_work * (_POLISH_TRIALS + 1)),
        ("prepared_source", iterations * (3 * nonlinear_work + constraint_work)),
        (
            "state",
            iterations
            * (
                (20 + 16 * _POLISH_TRIALS) * inequalities + 16 * kkt_dimension + dimension
            ),
        ),
        ("prepared_derivatives", refreshes * derivative_preparation_work),
        ("setup_sparse_saddle", refreshes * setup.sparse_setup_work_upper()),
        ("native_factor", refreshes * factor_work),
        (
            "native_substitution_prepare",
            refreshes
            * (
                plan.numeric_substitution_preparation_work_units_upper
                + plan.lu_congruence_preparation_work_units_upper
            ),
        ),
        ("native_congruence_correction", iterations * (steps + 1) * solve_work),
    )


def _placement_order_work_bound(
    sites: int,
    edges: int,
    free: int,
    mutable: int,
    /,
) -> int:
    """Bound one admitted predecessor bank and its priority-heap traversal."""
    # Each call builds incidence once, initializes predecessor lists once and
    # updates them once per placed endpoint. The priority heap contains at most
    # free + 2*mutable entries; each entry is pushed and popped at most once.
    # Selected predecessor sorts collectively touch at most mutable entries.
    ordering = 4 * (
        2 * edges
        + sites
        + 6 * mutable
        + 2 * (free + 2 * mutable) * max(1, (free + 2 * mutable).bit_length())
        + 4 * free
        + mutable * max(1, mutable.bit_length())
    )
    return ordering


def _proposal_scratch_bound(points: int, edges: int, cells: int, /) -> int:
    """Conservative live numerical, prepared-action and host-graph storage."""
    dimension = 2 * points
    # Before selecting the bank, every edge can be a mutable row and every
    # selected row can be an equality. Bounds are on the same pool.
    kkt_dimension = dimension + edges
    inequalities = cells + 2 * edges + 2 * dimension
    coefficients = 64 * (2 * edges + 3 * cells + dimension)
    state = 32 * (kkt_dimension + inequalities + dimension)
    # Local coefficients, a prepared metric column and Jacobi inverse coexist
    # with the Newton/Krylov state, never a dense derivative bank.
    metric = 4 * dimension + 3 * inequalities + 4 * edges + 2 * kkt_dimension
    source = 24 * points + 16 * (edges + 3 * cells)
    grid = 16 * _BATCH * (2 * _GRID_RADIUS + 1) ** 2
    host_graph = 128 * (points + edges) + 64 * points * points
    sparse_plans = 256 * edges * edges + 512 * (dimension + edges)
    return 8 * (coefficients + state + source + grid + metric) + host_graph + sparse_plans


class _Charge:
    """Admitted conservative host/device work of one proposal."""

    def __init__(
        self, maximum_work: int, stack: ExitStack, /, *, scope: NativeExecutionScope
    ) -> None:
        self.maximum_work = maximum_work
        self.work = 0
        self.scope, self.stack = scope, stack
        self.ticket: NativeBatchTicket | None = None
        self.phases: dict[str, int] = {}

    def admit(
        self,
        bound: int,
        achieved: tuple[tuple[str, float], ...],
        /,
        *,
        reserve: bool = True,
    ) -> None:
        if self.work + bound > self.maximum_work:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "periodic.proposal.work must fit the remaining work envelope; got a phase request beyond the envelope",
                requested=(("remaining_work", self.maximum_work),),
                achieved=(
                    ("proposal_work", self.work),
                    ("stage_work_bound", bound),
                    ("proposal_remaining_work", self.maximum_work - self.work),
                    *(
                        (f"proposal_phase_work:{name}", work)
                        for name, work in self.phases.items()
                    ),
                    *achieved,
                ),
            )
        if self.ticket is not None:
            self.ticket.settle()
            self.ticket = None
        if reserve:
            self.ticket = self.stack.enter_context(self.scope.batch(bound))

    def charge(self, work: int, phase: str, /) -> None:
        self.work += work
        self.phases[phase] = self.phases.get(phase, 0) + work
        if self.ticket is None:
            raise RuntimeError(
                "periodic.proposal.ticket must be admitted before consumption; got no stage ticket"
            )
        self.ticket.consume(work)


@final
class _RelocationDomain(StrictModule):
    """Bind the quotient geometry and hard placement bounds for one proposal."""

    points: np.ndarray
    edges: np.ndarray
    cells: np.ndarray
    shifts: np.ndarray
    vectors: np.ndarray
    protected: np.ndarray
    target: float
    minimum_size: float
    maximum_size: float


@final
class _RelocationPreparation(StrictModule):
    """Retain free-root routes, publication lifts, and immutable rank evidence."""

    domain: _RelocationDomain
    free: np.ndarray
    immutable: np.ndarray
    lifts: np.ndarray
    required: int | None
    action_work: int
    eligible: bool


@final
class _ExtremePhase(StrictModule):
    """Carry the continuous size solve and its materialized admissibility data."""

    args: _SizeArgs
    result: _SolveRecord
    normalized: np.ndarray
    areas: np.ndarray
    edge_offsets: np.ndarray
    cell_offsets: np.ndarray


@final
class _RankPhase(StrictModule):
    """Retain a deterministic predecessor order and the selected equality bank."""

    order: list[int]
    kept: np.ndarray


@final
class _PolishPreparation(StrictModule):
    """Retain the admitted sparse hard method and compiled-cache storage bound."""

    args: _PolishArgs
    method: PrimalDualNewtonKrylov
    problem: MinimizationProblem
    work_bound: int
    cache_bytes_upper: int
    goals: np.ndarray


@final
class _PolishPhase(StrictModule):
    """Carry the hard solve, diagnostics, and host placement coordinates."""

    result: _SolveRecord
    record: PeriodicPolishRecord
    parameters: np.ndarray
    admitted: bool


@final
class _PlacementPhase(StrictModule):
    """Carry the independently certified representable proposal and search work."""

    points: np.ndarray | None
    nodes: int


@final
class _HostRelocationMetrics(StrictModule):
    """Carry host edge lengths and cell areas for the prepared quotient state."""

    normalized: np.ndarray
    areas: np.ndarray


def _host_relocation_metrics(
    prepared: _RelocationPreparation,
    parameters: np.ndarray,
    /,
    *,
    edge_offsets: np.ndarray,
    cell_offsets: np.ndarray,
) -> _HostRelocationMetrics:
    """Evaluate the size kernel's formulas on materialized solver parameters."""
    domain = prepared.domain
    points = domain.points.copy()
    points[prepared.free] += domain.target * parameters
    delta = points[domain.edges[:, 1]] - points[domain.edges[:, 0]] + edge_offsets
    lengths = np.sqrt(np.sum(delta * delta, axis=1)) / domain.target
    normalized = np.where(~prepared.immutable, lengths, 0.0)
    corners = (points[domain.cells] + cell_offsets) / domain.target
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    areas = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    return _HostRelocationMetrics(normalized, areas)


def _prepare_relocation_domain(
    domain: _RelocationDomain, statistics: tuple[SizeStatistic, ...], /
) -> _RelocationPreparation:
    points, edges, cells = domain.points, domain.edges, domain.cells
    shifts, vectors, protected = domain.shifts, domain.vectors, domain.protected
    target, minimum_size, maximum_size = (
        domain.target,
        domain.minimum_size,
        domain.maximum_size,
    )
    free = np.flatnonzero(~protected)
    immutable = np.all(protected[edges[:, :2]], axis=1) | (edges[:, 0] == edges[:, 1])
    lifts = periodic_reference_lifts(cells, shifts, points.shape[0])
    lengths = quotient_edge_lengths(points, edges, lifts, vectors)
    required = _required_exact_edges(lengths, immutable, target, statistics)
    dimension = 2 * free.size
    action_work = edges.shape[0] + 4 * cells.shape[0] + dimension
    eligible = not (
        free.size == 0
        or required is None
        or np.any(lengths[immutable] > maximum_size)
        or np.any(lengths[immutable] < minimum_size)
    )
    return _RelocationPreparation(
        domain, free, immutable, lifts, required, action_work, eligible
    )


def _execute_extreme_phase(
    prepared: _RelocationPreparation,
    charge: _Charge,
    sites: tuple[tuple[str, float], ...],
    check: Callable[[], None],
    /,
) -> _ExtremePhase:
    domain = prepared.domain
    points, edges, cells = domain.points, domain.edges, domain.cells
    shifts, vectors = domain.shifts, domain.vectors
    target, minimum_size = domain.target, domain.minimum_size
    free, immutable = prepared.free, prepared.immutable
    action_work = prepared.action_work
    # A projected line search can finish its bounded batch beyond the nominal
    # evaluation cutoff. Include each outer batch and the final certificates.
    extreme_evaluations = (
        _EXTREME_EVALUATIONS
        + _EXTREME_METHOD.maximum_outer_steps
        * (_EXTREME_INNER_METHOD.line_search.maximum_steps + 2)
        + 2
    )
    extreme_bound = (
        4 * extreme_evaluations + 3 * (_EXTREME_METHOD.maximum_outer_steps + 1)
    ) * action_work
    charge.admit(extreme_bound, sites)
    edge_offsets = edges[:, 2:] @ vectors
    cell_offsets = shifts @ vectors
    args = _SizeArgs(
        jax.device_put(points),
        jax.device_put(np.asarray(free, dtype=np.int32)),
        jax.device_put(np.asarray(edges[:, :2], dtype=np.int32)),
        jax.device_put(edge_offsets),
        jax.device_put(np.asarray(cells, dtype=np.int32)),
        jax.device_put(cell_offsets),
        jax.device_put(np.asarray(target, dtype=np.float64)),
        jax.device_put(~immutable),
        jax.device_put(np.asarray(minimum_size / target, dtype=np.float64)),
    )
    # Diagnostics report partial optimization counts. Charge the admitted
    # bound rather than treating partial totals as the work performed.
    charge.charge(extreme_bound, "extreme")
    extreme = _solve_extreme(jax.device_put(_seed(points, free, vectors)), args)
    check()

    parameters = np.asarray(jax.device_get(extreme.parameters))
    metrics = _host_relocation_metrics(
        prepared, parameters, edge_offsets=edge_offsets, cell_offsets=cell_offsets
    )
    return _ExtremePhase(
        args, extreme, metrics.normalized, metrics.areas, edge_offsets, cell_offsets
    )


def _prepare_relocation_rank(
    prepared: _RelocationPreparation,
    extreme: _ExtremePhase,
    needed: int,
    charge: _Charge,
    sites: tuple[tuple[str, float], ...],
    /,
) -> _RankPhase:
    domain = prepared.domain
    points, edges, protected = domain.points, domain.edges, domain.protected
    free, immutable = prepared.free, prepared.immutable
    normalized = extreme.normalized
    mutable = ~immutable
    # Rank preparation is reserved once, then each executed bank is admitted
    # independently. Unexecuted enlargement banks never consume proposal work.
    # All charges remain proved host bounds, not partial optimizer diagnostics.
    rank_bound = 4 * edges.shape[0] * (max(1, edges.shape[0].bit_length()) + 2)
    charge.admit(rank_bound, sites)
    charge.charge(rank_bound, "rank_selection")

    def admit_bank(active_edges: int) -> None:
        bound = _placement_order_work_bound(
            points.shape[0], edges.shape[0], free.size, active_edges
        )
        charge.admit(bound, sites)
        charge.charge(bound, "rank_graph")

    order, _, kept = _rank_placement_edges(
        free, edges, normalized, mutable, protected, needed, admit_bank=admit_bank
    )
    return _RankPhase(order, kept)


def _prepare_relocation_polish(
    prepared: _RelocationPreparation,
    extreme: _ExtremePhase,
    rank: _RankPhase,
    charge: _Charge,
    sites: tuple[tuple[str, float], ...],
    source_binding_id: str,
    maximum_scratch_bytes: int,
    workspace: NativeHostStorageWorkspace,
    /,
) -> _PolishPreparation:
    domain = prepared.domain
    points, edges, cells = domain.points, domain.edges, domain.cells
    target, minimum_size = domain.target, domain.minimum_size
    free = prepared.free
    kept, mutable, args = rank.kept, ~prepared.immutable, extreme.args
    dimension, maximum_work = 2 * free.size, charge.maximum_work
    goals = np.where(kept, 1.0, 1.0 - 2.0 * _DEMOTION)
    mutable_edges = np.flatnonzero(mutable)
    polish_args = _PolishArgs(
        args,
        jax.device_put(kept),
        jax.device_put(goals),
        jax.device_put(np.asarray(mutable_edges, dtype=np.int32)),
    )
    remaining_work = maximum_work - charge.work
    # Native symbolic owners enforce the same remaining limits
    # before growing their incidence, product and factor patterns.
    charge.admit(remaining_work, sites, reserve=False)
    try:
        method, symbolic_work_upper = _prepare_polish_method(
            source_binding_id,
            free,
            edges,
            kept,
            mutable,
            remaining_work,
            maximum_scratch_bytes,
            polish_args,
        )
    except LinearResourceLimitError as error:
        observed = (
            ("resource_request", error.requested),
            ("completed_proposal_work", charge.work),
            ("kept_native_quotient_equalities", int(np.count_nonzero(kept))),
            ("free_native_roots", free.size),
        )
        extra = tuple(
            (name, value)
            for name, value in (
                ("resource_completed", error.completed),
                ("sparse_symbolic_work", error.symbolic_work),
                ("sparse_storage_bytes_upper", error.storage_bytes_upper),
            )
            if value is not None
        )
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            str(error),
            requested=((f"{error.resource}:limit", error.limit),),
            achieved=(*observed, *extra),
        ) from error
    charge.admit(symbolic_work_upper, sites)
    charge.charge(symbolic_work_upper, "sparse_symbolic")
    workspace.retain_owner(method)
    workspace.set_bound(
        workspace.bound
        - (256 * edges.shape[0] ** 2 + 512 * (2 * points.shape[0] + edges.shape[0]))
    )
    polish_phases = _polish_work_bounds(
        points.size,
        edges.shape[0],
        cells.shape[0],
        dimension,
        mutable_edges.size,
        int(np.count_nonzero(kept)),
        method,
    )
    polish_bound = sum(work for _, work in polish_phases)
    charge.admit(
        polish_bound,
        (
            *sites,
            *((f"polish_work_upper:{name}", work) for name, work in polish_phases),
        ),
    )
    problem = _polish_problem(mutable_edges, kept, minimum_size / target)
    setup, factor_plan = _polish_sparse_owners(method)
    itemsize = setup.kkt_assembly.template.coefficients.dtype.itemsize
    cache_logical_upper = (
        factor_plan.numeric_substitution_storage_bytes_upper(itemsize)
        + factor_plan.numeric_substitution_refresh_workspace_bytes_upper(itemsize)
        + factor_plan.lu_congruence_storage_bytes_upper(itemsize)
        + factor_plan.lu_congruence_refresh_workspace_bytes_upper(itemsize)
        + setup.derivative_workspace_bytes_upper(itemsize)
    )
    return _PolishPreparation(
        polish_args, method, problem, polish_bound, cache_logical_upper, goals
    )


def _execute_relocation_polish(
    prepared: _RelocationPreparation,
    extreme: _ExtremePhase,
    rank: _RankPhase,
    polish: _PolishPreparation,
    charge: _Charge,
    check: Callable[[], None],
    /,
    *,
    scope: NativeExecutionScope,
) -> _PolishPhase:
    kept, goals = rank.kept, polish.goals
    mutable = ~prepared.immutable
    target, minimum_size = prepared.domain.target, prepared.domain.minimum_size
    # Compiled cache owners are not host-observable. Reserve both refresh
    # epochs and preparation temporaries in the same pool until
    # the existing result materialization completes; no callback/new sync
    # and no device-measured or host-owner claim.
    with scope.host_workspace() as cache_workspace:
        cache_workspace.set_bound(polish.cache_bytes_upper)
        charge.charge(polish.work_bound, "hard_solve")
        polished = _solve_polish(
            extreme.result.parameters,
            polish.args,
            polish.problem,
            polish.method,
        )
        polish_record = _polish_record(polished, polish.method)
    check()
    parameters = np.asarray(jax.device_get(polished.parameters))
    metrics = _host_relocation_metrics(
        prepared,
        parameters,
        edge_offsets=extreme.edge_offsets,
        cell_offsets=extreme.cell_offsets,
    )
    normalized, areas = metrics.normalized, metrics.areas
    admitted = bool(
        np.all(np.isfinite(normalized))
        and np.max(np.abs(normalized - goals)[kept], initial=0.0) <= _DEMOTION / 4.0
        and np.max(normalized[mutable & ~kept], initial=0.0) <= 1.0 - _DEMOTION
        and np.min(areas) > 0.5 * _AREA_MARGIN
        and np.min(normalized[mutable], initial=np.inf) >= minimum_size / target
        and np.max(np.abs(parameters), initial=0.0) <= 1.0
    )
    return _PolishPhase(polished, polish_record, parameters, admitted)


def _place_relocated_sites(
    prepared: _RelocationPreparation,
    rank: _RankPhase,
    polished: _PolishPhase,
    needed: int,
    charge: _Charge,
    sites: tuple[tuple[str, float], ...],
    check: Callable[[], None],
    /,
    *,
    scope: NativeExecutionScope,
) -> _PlacementPhase:
    domain = prepared.domain
    points, edges, cells = domain.points, domain.edges, domain.cells
    shifts, vectors, protected = domain.shifts, domain.vectors, domain.protected
    target, minimum_size = domain.target, domain.minimum_size
    free, lifts, mutable = prepared.free, prepared.lifts, ~prepared.immutable
    kept, order = rank.kept, rank.order
    maximum_work, action_work = charge.maximum_work, prepared.action_work
    if charge.ticket is not None:
        charge.ticket.settle()
        charge.ticket = None
    relocated = scope.allocate_host_array(points.shape, np.float64)
    relocated[...] = points
    relocated[free] += target * np.asarray(polished.parameters, dtype=np.float64)
    charge.admit(maximum_work - charge.work, sites)
    placement = _ExactPlacement(
        relocated,
        edges,
        lifts,
        vectors,
        protected,
        kept,
        target,
        minimum_size,
        maximum_work - charge.work,
        check,
    )
    try:
        found = placement.search(order)
    finally:
        charge.charge(placement.work, "exact_placement")
    candidate = placement.points if found else None
    if candidate is not None:
        certification_bound = 8 * action_work
        charge.admit(certification_bound, sites)
        charge.charge(certification_bound, "certification")
        moves = np.abs(candidate[free] - points[free])
        final = quotient_edge_lengths(candidate, edges, lifts, vectors)
        if (
            np.any(moves > target)
            or not np.array_equal(candidate[protected], points[protected])
            or np.any(final[mutable] > target)
            or np.any(final[mutable] < minimum_size)
            or np.count_nonzero(final[mutable] == target) < needed
            or not _positive_scientific_cells(candidate, cells, shifts, vectors)
        ):
            candidate = None
    return _PlacementPhase(candidate, placement.nodes)


def _relocation_proposal(
    points: np.ndarray | None,
    work: int,
    extreme: _SolveRecord | None,
    needed: int,
    polish_status: int,
    kept: int,
    nodes: int,
    polish: PeriodicPolishRecord | None = None,
) -> PeriodicSizeProposal:
    return PeriodicSizeProposal(
        points,
        work,
        _EXTREME_METHOD.method_id,
        -1 if extreme is None else int(extreme.status),
        0 if extreme is None else int(extreme.iterations),
        0 if extreme is None else int(extreme.value_evaluations),
        0 if extreme is None else int(extreme.actions),
        polish_status,
        needed,
        kept,
        nodes,
        polish,
    )


def propose_periodic_rank_placement(
    points: np.ndarray,
    edges: np.ndarray,
    cells: np.ndarray,
    shifts: np.ndarray,
    vectors: np.ndarray,
    protected: np.ndarray,
    target: float,
    statistics: tuple[SizeStatistic, ...],
    maximum_work: int,
    maximum_scratch_bytes: int,
    check: Callable[[], None],
    /,
    *,
    scope: NativeExecutionScope,
    maximum_size: float = np.inf,
    minimum_size: float = 0.0,
) -> PeriodicSizeProposal:
    """Finish a representable target-rank continuation without a KKT solve.

    Tail edges retain their admitted bounds. Protected roots remain fixed and
    every returned candidate must meet the frozen quantile targets and pass
    the positive-cell certificate; nonlocal geometric searches are not used.
    """
    sites, edge_count, cell_count = points.shape[0], edges.shape[0], cells.shape[0]
    maximum_work = min(maximum_work, scope.remaining().remaining_work_units)
    scratch = (
        8 * (32 * sites + 24 * edge_count + 8 * cell_count)
        + 128 * (sites + edge_count)
        + 64 * sites * sites
    )
    available = min(maximum_scratch_bytes, scope.remaining().remaining_managed_bytes)
    if scratch > available:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "periodic.rank_placement.scratch must fit the remaining scratch envelope; got an unadmitted placement bank",
            requested=(("remaining_scratch_bytes", available),),
            achieved=(("placement_scratch_bound", scratch),),
        )
    method_id = canonical_fingerprint({"kind": "periodic-representable-rank-placement"})
    work, needed, kept_count, nodes = 0, -1, 0, 0
    candidate: np.ndarray | None = None
    with scope.host_workspace() as workspace:
        workspace.set_bound(scratch)
        preparation = 8 * (sites + edge_count + cell_count) + (
            8 * len(statistics) * edge_count * max(1, edge_count.bit_length())
        )
        with scope.batch(min(preparation, maximum_work)) as ticket:
            if preparation > maximum_work:
                ticket.refuse(preparation)
            ticket.consume(preparation)
            work += preparation
            lifts = periodic_reference_lifts(cells, shifts, sites)
            lengths = quotient_edge_lengths(points, edges, lifts, vectors)
            fixed = np.all(protected[edges[:, :2]], axis=1) | (edges[:, 0] == edges[:, 1])
            ranks = [
                size_statistic_quantile(name) * (edge_count - 1) for name in statistics
            ]
            lowest, highest = int(np.floor(min(ranks))), int(np.ceil(max(ranks)))
            fixed_lengths = lengths[fixed]
            eligible = _fixed_rank_feasible(
                lengths,
                fixed,
                target,
                tuple(size_statistic_quantile(name) for name in statistics),
            )
            needed = max(
                0, highest - lowest + 1 - int(np.count_nonzero(fixed_lengths == target))
            )
            measured = tuple(
                float(np.quantile(lengths, size_statistic_quantile(name)))
                for name in statistics
            )
            near = all(abs(value - target) <= target * _ACTIVE_GAP for value in measured)
            free = np.flatnonzero(~protected)
            incidence = np.bincount(edges[:, :2].reshape(-1), minlength=sites)
            maximum_incident = int(np.max(incidence, initial=0))
        if eligible and near and free.size and np.all(lengths <= maximum_size):
            rank_scratch = (
                16 * _BATCH * (2 * _GRID_RADIUS + 1) ** 2 * (4 * maximum_incident + 24)
            )
            if scratch + rank_scratch > available:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "periodic.rank_placement.ranks must fit the remaining scratch envelope; got an unadmitted local rank bank",
                    requested=(("remaining_scratch_bytes", available),),
                    achieved=(("placement_scratch_bound", scratch + rank_scratch),),
                )
            scratch += rank_scratch
            workspace.set_bound(scratch)
            move_bound = _SLIDES * _SLIDE_STEP * target
            tail = lengths > target * (1.0 + _ACTIVE_GAP)
            lower = np.where(
                tail,
                np.maximum(
                    target * (1.0 + _ACTIVE_GAP / 2.0), lengths - 4.0 * move_bound
                ),
                minimum_size,
            )
            upper = np.minimum(
                maximum_size,
                np.where(tail, lengths + 4.0 * move_bound, target + 4.0 * move_bound),
            )
            relocated = scope.allocate_host_array(points.shape, np.float64)
            relocated[...] = points
            placement = _ExactPlacement(
                relocated,
                edges,
                lifts,
                vectors,
                protected,
                np.zeros(edge_count, dtype=np.bool_),
                target,
                minimum_size,
                maximum_work - work,
                check,
                edge_lower=lower,
                edge_upper=upper,
            )
            with scope.batch(maximum_work - work) as ticket:
                try:
                    found = placement.search_local_ranks(
                        lengths, lowest, highest, statistics=statistics
                    )
                finally:
                    ticket.consume(placement.work)
                    work += placement.work
            nodes = placement.nodes
            kept_count = int(np.count_nonzero(lengths[~fixed] == target))
            if found:
                certification = 8 * (sites + edge_count + cell_count) + (
                    4 * len(statistics) * edge_count * max(1, edge_count.bit_length())
                )
                with scope.batch(min(certification, maximum_work - work)) as ticket:
                    if certification > maximum_work - work:
                        ticket.refuse(certification)
                    ticket.consume(certification)
                    work += certification
                    final = quotient_edge_lengths(relocated, edges, lifts, vectors)
                    final_statistics = tuple(
                        float(np.quantile(final, size_statistic_quantile(name)))
                        for name in statistics
                    )
                    admitted = (
                        np.max(np.abs(relocated - points), initial=0.0) <= move_bound
                        and np.array_equal(relocated[protected], points[protected])
                        and np.all(final >= minimum_size)
                        and np.all(final <= maximum_size)
                        and all(value == target for value in final_statistics)
                        and _positive_scientific_cells(relocated, cells, shifts, vectors)
                    )
                kept_count = int(np.count_nonzero(final[~fixed] == target))
                if admitted:
                    candidate = relocated
    return PeriodicSizeProposal(
        candidate,
        work,
        method_id,
        -1,
        0,
        0,
        0,
        -1,
        needed,
        kept_count,
        nodes,
        None,
    )


@checked
def propose_periodic_site_relocation(
    points: np.ndarray,
    edges: np.ndarray,
    cells: np.ndarray,
    shifts: np.ndarray,
    vectors: np.ndarray,
    protected: np.ndarray,
    target: float,
    statistics: tuple[SizeStatistic, ...],
    maximum_work: int,
    maximum_scratch_bytes: int,
    check: Callable[[], None],
    /,
    *,
    source_binding_id: str,
    scope: NativeExecutionScope,
    maximum_size: float = np.inf,
    minimum_size: float = 0.0,
) -> PeriodicSizeProposal:
    """Target-statistic relocation of unprotected roots, or ``points=None`` with status."""
    scratch = _proposal_scratch_bound(points.shape[0], edges.shape[0], cells.shape[0])
    available_scratch = min(
        maximum_scratch_bytes, scope.remaining().remaining_managed_bytes
    )
    if scratch > available_scratch:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "periodic.proposal.scratch must fit the remaining scratch envelope; got a proposal request beyond the envelope",
            requested=(
                ("maximum_scratch_bytes", maximum_scratch_bytes),
                ("remaining_scratch_bytes", available_scratch),
            ),
            achieved=(
                ("proposal_scratch_bound", scratch),
                ("free_sites", float(protected.size - np.count_nonzero(protected))),
            ),
        )
    with ExitStack() as stack, scope.host_workspace() as workspace:
        workspace.set_bound(scratch)
        prepared = _prepare_relocation_domain(
            _RelocationDomain(
                points,
                edges,
                cells,
                shifts,
                vectors,
                protected,
                target,
                minimum_size,
                maximum_size,
            ),
            statistics,
        )
        if not prepared.eligible or prepared.required is None:
            return _relocation_proposal(
                None,
                0,
                None,
                -1 if prepared.required is None else prepared.required,
                -1,
                0,
                0,
            )
        needed = prepared.required
        charge = _Charge(maximum_work, stack, scope=scope)
        sites = (
            ("quotient_sites", float(points.shape[0])),
            ("free_sites", float(prepared.free.size)),
        )
        extreme = _execute_extreme_phase(prepared, charge, sites, check)
        if not (
            np.all(np.isfinite(extreme.normalized))
            and np.max(extreme.normalized) <= 1.0 + _ACTIVE_GAP
            and np.min(extreme.areas) > 0.5 * _AREA_MARGIN
        ):
            return _relocation_proposal(
                None, charge.work, extreme.result, needed, -1, 0, 0
            )
        rank = _prepare_relocation_rank(prepared, extreme, needed, charge, sites)
        kept = int(np.count_nonzero(rank.kept))
        if kept < needed:
            return _relocation_proposal(
                None, charge.work, extreme.result, needed, -1, kept, 0
            )
        polish_preparation = _prepare_relocation_polish(
            prepared,
            extreme,
            rank,
            charge,
            sites,
            source_binding_id,
            maximum_scratch_bytes,
            workspace,
        )
        polished = _execute_relocation_polish(
            prepared,
            extreme,
            rank,
            polish_preparation,
            charge,
            check,
            scope=scope,
        )
        if not polished.admitted:
            return _relocation_proposal(
                None,
                charge.work,
                extreme.result,
                needed,
                int(polished.result.status),
                kept,
                0,
                polished.record,
            )
        placement = _place_relocated_sites(
            prepared, rank, polished, needed, charge, sites, check, scope=scope
        )
        return _relocation_proposal(
            placement.points,
            charge.work,
            extreme.result,
            needed,
            int(polished.result.status),
            kept,
            placement.nodes,
            polished.record,
        )
