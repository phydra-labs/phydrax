#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Vector-field jet hierarchy preconditioning the exact signed metric solve.

The exact signed metric is the minimum-norm solution of ``B z = c`` with
``B = D^-1 A S``. Its Craig solve iterates on ``B B^T``, whose low spectrum is
known in closed form. Let ``g`` be any smooth vector field. On every node with
moment rows, take its multipliers to be the Taylor coefficients of the line
integral of ``g``:

    y_i^alpha = (1/|alpha|) sum_c [alpha_c >= 1] d^(alpha - e_c) g_c(x_i) / (alpha - e_c)!

For an edge ``d = x_j - x_i`` the adjoint ``A^T y`` evaluates
``p_i(d) + p_j(-d)``, where ``p_i(d)`` is the truncated Taylor expansion of
``integral_0^1 g(x_i + t d) . d dt``. Both endpoints expand the same segment
integral, so the sum is a Taylor remainder: ``O(h^4 |grad^3 g|)`` at moment
degree 2. ``g`` is not constrained to be a gradient, and it has to vanish only
where degree-one rows are absent (boundary-closure nodes under the
``"second-moment"`` closure). Measured on jittered clouds, these jets carry the
smallest eigenvalues of ``B B^T``. They are degree-one dominated and spread over
the cloud, so the condition number grows like ``h^-6``. Row or column scaling
cannot remove that: it is the conditioning of a sixth-order operator acting on
``g``.

The coarse spaces are therefore jets of tensor-product B-spline vector fields
of degree ``moment_degree + 1`` on dyadically nested grids over the cloud's
coordinate box (periodic splines along periodic axes). The spline spaces are
conforming, i.e. smooth across cells. This matters because piecewise-polynomial
aggregates pay jump energy in this high-order operator. Transfers are exact:
the jets at the nodes give the finest prolongation, and B-spline knot insertion
gives every coarser one. The preconditioner only changes the iteration count:
Craig returns ``z = B^T y`` in ``range(B^T)``, which is the exact minimum-norm
solution for any symmetric positive-definite preconditioner.
"""

from __future__ import annotations

from math import factorial

import jax.numpy as jnp
import numpy as np
import scipy.interpolate as si
import scipy.sparse as sp

from ...linalg import ArraySpace
from ...sparse import EdgeRelation, SparseCoordinateOperator


def _axis_knots(intervals: int, degree: int, periodic: bool, /) -> np.ndarray:
    if periodic:
        return np.arange(-degree, intervals + degree + 1, dtype=np.float64) / intervals
    interior = np.linspace(0.0, 1.0, intervals + 1)
    return np.concatenate(
        (np.zeros(degree), interior, np.ones(degree)),
    )


def _axis_count(intervals: int, degree: int, periodic: bool, /) -> int:
    return intervals if periodic else intervals + degree


def _axis_tables(
    unit: np.ndarray, intervals: int, degree: int, periodic: bool, orders: int, /
) -> list[np.ndarray]:
    """Dense B-spline basis derivatives ``0..orders`` at unit coordinates.

    Periodic bases evaluate the uniform extension on ``[0, 1)`` and fold every
    translate by one period onto the same coefficient.
    """
    knots = _axis_knots(intervals, degree, periodic)
    count = knots.size - degree - 1
    x = np.mod(unit, 1.0) if periodic else np.clip(unit, 0.0, 1.0)
    basis = si.BSpline(knots, np.eye(count), degree, extrapolate=False)
    tables: list[np.ndarray] = []
    for order in range(orders + 1):
        values = np.asarray((basis.derivative(order) if order else basis)(x))
        if not np.all(np.isfinite(values)):
            raise RuntimeError("B-spline evaluation left the coordinate box.")
        if periodic:
            folded = np.zeros((values.shape[0], intervals))
            for column in range(count):
                folded[:, column % intervals] += values[:, column]
            values = folded
        tables.append(values)
    return tables


def _axis_refinement(intervals: int, degree: int, periodic: bool, /) -> np.ndarray:
    """Exact knot-insertion map from ``intervals`` to ``2 intervals`` cells."""
    samples = np.linspace(
        0.0, 1.0, 4 * (degree + 1) * 2 * intervals + 1, endpoint=not periodic
    )
    fine = _axis_tables(samples, 2 * intervals, degree, periodic, 0)[0]
    coarse = _axis_tables(samples, intervals, degree, periodic, 0)[0]
    refinement, *_ = np.linalg.lstsq(fine, coarse, rcond=None)
    refinement[np.abs(refinement) < 1e-13 * np.abs(refinement).max()] = 0.0
    if not np.allclose(fine @ refinement, coarse, atol=1e-11):
        raise RuntimeError(
            "B-spline knot insertion failed to reproduce the coarse basis."
        )
    return refinement


def _finest_intervals(width: float, spacing: float, /) -> int:
    """Largest power of two whose cells are at least two node spacings."""
    intervals = 1
    while width / (2 * intervals) >= 2.0 * spacing:
        intervals *= 2
    return intervals


def _jet_prolongation(
    unit: np.ndarray,
    widths: np.ndarray,
    exponents: np.ndarray,
    row_nodes: np.ndarray,
    row_scaling: np.ndarray,
    intervals: tuple[int, ...],
    degree: int,
    periodic: tuple[bool, ...],
    /,
) -> sp.csr_matrix:
    """Rows of ``D y_phys``: scaled line-integral jets of spline vector fields."""
    dimension = unit.shape[1]
    orders = int(exponents.sum(axis=1).max()) - 1
    tables = [
        _axis_tables(unit[:, axis], intervals[axis], degree, periodic[axis], orders)
        for axis in range(dimension)
    ]
    counts = tuple(table[0].shape[1] for table in tables)
    tensor = int(np.prod(counts))
    # Each node touches degree + 1 consecutive basis functions per axis.
    spans = []
    for axis in range(dimension):
        cells = intervals[axis]
        scaled = unit[:, axis] * cells
        if periodic[axis] and cells <= degree:
            local = np.broadcast_to(np.arange(cells), (unit.shape[0], cells))
        elif periodic[axis]:
            first = np.floor(np.mod(scaled, cells)).astype(np.int64)
            local = (first[:, None] + np.arange(degree + 1)[None, :]) % cells
        else:
            first = np.clip(np.floor(scaled).astype(np.int64), 0, cells - 1)
            local = first[:, None] + np.arange(degree + 1)[None, :]
        spans.append(local)
    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    vals_out: list[np.ndarray] = []
    for row_pattern in np.unique(exponents, axis=0):
        selected = np.flatnonzero(np.all(exponents == row_pattern, axis=1))
        total = int(row_pattern.sum())
        nodes = row_nodes[selected]
        for component in range(dimension):
            if row_pattern[component] < 1:
                continue
            beta = row_pattern.copy()
            beta[component] -= 1
            weight = 1.0 / (total * np.prod([factorial(int(b)) for b in beta]))
            local_values = np.ones((selected.size, 1))
            local_index = np.zeros((selected.size, 1), dtype=np.int64)
            for axis in range(dimension):
                index = spans[axis][nodes]
                values = np.take_along_axis(
                    tables[axis][beta[axis]][nodes], index, axis=1
                )
                values = values / widths[axis] ** beta[axis]
                local_values = (local_values[:, :, None] * values[:, None, :]).reshape(
                    selected.size, -1
                )
                local_index = (
                    local_index[:, :, None] * counts[axis] + index[:, None, :]
                ).reshape(selected.size, -1)
            scale = weight * row_scaling[selected]
            rows_out.append(np.repeat(selected, local_index.shape[1]))
            cols_out.append((component * tensor + local_index).reshape(-1))
            vals_out.append((scale[:, None] * local_values).reshape(-1))
    matrix = sp.coo_matrix(
        (np.concatenate(vals_out), (np.concatenate(rows_out), np.concatenate(cols_out))),
        shape=(row_scaling.size, dimension * tensor),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _coordinate(
    matrix: sp.csr_matrix,
    source: ArraySpace,
    target: ArraySpace,
    operator_id: str,
    /,
) -> SparseCoordinateOperator:
    coo = matrix.tocoo()
    return SparseCoordinateOperator(
        EdgeRelation(
            coo.col.astype(np.int32),
            coo.row.astype(np.int32),
            source_size=matrix.shape[1],
            target_size=matrix.shape[0],
        ),
        jnp.asarray(coo.data, dtype=jnp.float64),
        source=source,
        target=target,
        operator_id=operator_id,
    )


_NULL_ENERGY = 1e-20
"""Relative ``|B^T P e_k|^2 / (lambda_max(B B^T) |P e_k|^2)`` below which a coarse
column is an exact null direction of ``B B^T`` up to rounding. One example is a
constant vector field on a periodic cloud. Such a column carries no coarse
correction and would leave a zero Galerkin diagonal, so it is dropped."""


def metric_jet_transfers(
    points: np.ndarray,
    exponents: np.ndarray,
    row_nodes: np.ndarray,
    row_scaling: np.ndarray,
    design: sp.csr_matrix,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    periodic: tuple[bool, ...],
    spacing: float,
    moment_degree: int,
    target: ArraySpace,
) -> tuple[tuple[SparseCoordinateOperator, SparseCoordinateOperator], ...]:
    """Restriction/prolongation pairs from moment rows to nested spline jets.

    ``points`` are the moment-node coordinates in the frame of the moment
    displacements, wrapped into ``[lower, upper)`` along periodic axes.
    ``exponents`` holds the multi-index of each row and ``row_nodes`` its moment
    node. ``row_scaling`` is the declared ``D``, and ``design`` is the prepared
    ``B = D^-1 A S``. On each level, columns that are null for ``B^T`` are
    dropped. These include B-splines that vanish at every node and exact null
    fields.
    """
    points = np.asarray(points, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    widths = upper - lower
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Metric jet hierarchy needs a positive finite coordinate box.")
    if not (np.isfinite(spacing) and spacing > 0):
        raise ValueError("Metric jet hierarchy needs a positive node spacing.")
    dimension = points.shape[1]
    degree = moment_degree + 1
    unit = (points - lower) / widths
    intervals = tuple(
        _finest_intervals(float(widths[axis]), spacing) for axis in range(dimension)
    )
    adjoint = sp.csr_matrix(design.T)
    gram = sp.csr_matrix(design @ adjoint)
    bound = float(np.asarray(abs(gram).sum(axis=1)).max())

    def energetic(columns: sp.csr_matrix, /) -> np.ndarray:
        image = adjoint @ columns
        energy = np.asarray(image.multiply(image).sum(axis=0)).ravel()
        size = np.asarray(columns.multiply(columns).sum(axis=0)).ravel()
        return np.flatnonzero(energy > _NULL_ENERGY * bound * size)

    # Rows of B that vanish leave the jets: they decouple in B B^T.
    live = np.asarray(design.multiply(design).sum(axis=1)).ravel() > 0
    finest = sp.csr_matrix(
        sp.diags(live.astype(np.float64))
        @ _jet_prolongation(
            unit,
            widths,
            np.asarray(exponents),
            np.asarray(row_nodes),
            np.asarray(row_scaling, dtype=np.float64),
            intervals,
            degree,
            periodic,
        )
    )
    finest.eliminate_zeros()
    active = energetic(finest)
    if active.size == 0:
        return ()
    cumulative = sp.csr_matrix(finest[:, active])
    matrices = [cumulative]
    while any(cells > 1 for cells in intervals):
        coarser = tuple(max(cells // 2, 1) for cells in intervals)
        factors = [
            _axis_refinement(coarser[axis], degree, periodic[axis])
            if coarser[axis] < intervals[axis]
            else np.eye(_axis_count(intervals[axis], degree, periodic[axis]))
            for axis in range(dimension)
        ]
        tensor = sp.csr_matrix(np.ones((1, 1)))
        for factor in factors:
            tensor = sp.kron(tensor, sp.csr_matrix(factor), format="csr")
        refinement = sp.csr_matrix(
            sp.kron(sp.identity(dimension), tensor, format="csr")[active]
        )
        refinement.eliminate_zeros()
        candidate = sp.csr_matrix(cumulative @ refinement)
        coarse_active = energetic(candidate)
        if coarse_active.size == 0 or coarse_active.size >= active.size:
            break
        matrices.append(sp.csr_matrix(refinement[:, coarse_active]))
        cumulative = sp.csr_matrix(candidate[:, coarse_active])
        active = coarse_active
        intervals = coarser
    spaces = [target] + [
        ArraySpace(
            (matrix.shape[1],),
            dtype=jnp.float64,
            space_id=f"{target.space_id}:metric-jets-{level}",
        )
        for level, matrix in enumerate(matrices)
    ]
    return tuple(
        (
            _coordinate(
                matrix.T.tocsr(),
                spaces[level],
                spaces[level + 1],
                f"{spaces[level + 1].space_id}:restriction",
            ),
            _coordinate(
                matrix,
                spaces[level + 1],
                spaces[level],
                f"{spaces[level + 1].space_id}:prolongation",
            ),
        )
        for level, matrix in enumerate(matrices)
    )
