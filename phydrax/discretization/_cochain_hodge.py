#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Metric-only dynamic diagonal and symmetric sparse cochain Hodges."""

from __future__ import annotations

import math
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    LinearSolvePolicy,
    OperatorProperties,
    prepare_sparse_factorization,
    refresh_sparse_factorization_values,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationStatus,
)
from ..sparse import EdgeRelation, SparseCoordinateOperator
from ..sparse._linear import _SparseStoragePlan
from ..typing import Dim, Float64, parse
from ._gram import diagonal_gram_space, gram_solve_policy, sparse_gram_space
from ._topology import CellComplexTopology


class _HodgeCoordinateDim(Dim):
    """Coordinates of one metric-paired cochain space."""


class _HodgeRouteDim(Dim):
    """Upper-triangular metric routes."""


DualCellPolicy: TypeAlias = Literal["barycentric", "circumcentric"]


def _metric_values(values: ArrayLike, /) -> Array:
    array = jnp.asarray(values)
    if array.ndim != 1 or jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValueError("Hodge metric values must be a rank-one real vector.")
    return array.astype(jnp.float64)


def _active_indices(active: ArrayLike, size: int, /) -> np.ndarray:
    host = np.asarray(active)
    if host.ndim != 1:
        raise ValueError("Active coordinates must be one static vector.")
    if host.dtype == np.dtype(np.bool_):
        if host.shape != (size,):
            raise ValueError("Active mask must match the Hodge coordinate count.")
        return np.flatnonzero(host).astype(np.int32)
    if not np.issubdtype(host.dtype, np.integer):
        raise TypeError("Active coordinates must be a Boolean mask or integer indices.")
    if np.any((host < 0) | (host >= size)) or np.any(host[1:] <= host[:-1]):
        raise ValueError("Active indices must be sorted, unique, and in range.")
    return host.astype(np.int32)


@final
class DiagonalHodge(StrictModule):
    """A real metric diagonal; numerical validity is device evidence."""

    __strict_contract__ = True
    weights: Float64[_HodgeCoordinateDim]
    size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, weights: ArrayLike, /) -> None:
        values = _metric_values(weights)
        layout = canonical_fingerprint(
            {"kind": "diagonal-hodge", "size": values.shape[0]}
        )
        self.weights = values
        self.size = values.shape[0]
        self.layout_id = layout

    @property
    def valid(self) -> Array:
        return jnp.all(jnp.isfinite(self.weights) & (self.weights > 0.0))

    def admit(self) -> DiagonalHodge:
        if not bool(np.asarray(self.valid)):
            raise ValueError("A Hodge metric must be finite and positive definite.")
        return self

    def restrict(self, active: ArrayLike, /) -> DiagonalHodge:
        indices = _active_indices(active, self.size)
        return DiagonalHodge(self.weights[indices])

    def refresh(self, values: ArrayLike, /) -> DiagonalHodge:
        refreshed = _metric_values(values)
        if refreshed.shape != self.weights.shape:
            raise ValueError("Refreshed weights must preserve the Hodge layout.")
        return eqx.tree_at(lambda hodge: hodge.weights, self, refreshed)

    def make_space(
        self, *, space_id: str, dtype: DTypeLike = jnp.float64
    ) -> tuple[ArraySpace, AbstractLinearOperator]:
        return diagonal_gram_space(self.weights, dtype=dtype, space_id=space_id)


def _admission_pattern(
    rows: tuple[int, ...], columns: tuple[int, ...], mirror: tuple[int, ...], size: int, /
) -> tuple[SparseFactorizationPlan | None, tuple[int, ...], _SparseStoragePlan]:
    """Prepare reusable storage and native SPD evidence from immutable routes."""
    row = np.asarray(rows, dtype=np.int32)
    column = np.asarray(columns, dtype=np.int32)
    mirrored = np.asarray(mirror, dtype=np.int32)
    full_rows = np.concatenate((row, column[mirrored]))
    full_columns = np.concatenate((column, row[mirrored]))
    order = np.lexsort((full_columns, full_rows))
    identifier = canonical_fingerprint(
        {"kind": "hodge-admission", "rows": rows, "columns": columns, "size": size}
    )
    # Symbolic preparation sees only the host pattern, even when numerical
    # construction occurs inside jit. No numerical metric is read here.
    with jax.ensure_compile_time_eval():
        coordinates = ArraySpace((size,), dtype=jnp.float64, space_id=identifier)
        relation = EdgeRelation(
            full_columns, full_rows, source_size=size, target_size=size
        )
        storage_plan = _SparseStoragePlan(relation)
        operator = SparseCoordinateOperator(
            relation,
            jnp.ones((full_rows.shape[0],), dtype=jnp.float64),
            source=coordinates,
            target=coordinates,
            properties=OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
            operator_id=f"{identifier}:pattern",
            storage_plan=storage_plan,
        )
        plan = (
            None
            if size == 0
            else prepare_sparse_factorization(
                operator,
                SparseFactorizationPolicy("cholesky", ordering="reverse-cuthill-mckee"),
            )
        )
    return plan, tuple(int(index) for index in order), storage_plan


@final
class SparseHodge(StrictModule):
    """A fixed upper pattern and dynamic real SPD metric, symmetric by construction."""

    __strict_contract__ = True
    rows: tuple[int, ...] = eqx.field(static=True)
    columns: tuple[int, ...] = eqx.field(static=True)
    upper_values: Float64[_HodgeRouteDim]
    size: int = eqx.field(static=True)
    policy: LinearSolvePolicy
    layout_id: str = eqx.field(static=True)
    _mirror: tuple[int, ...] = eqx.field(static=True)
    _admission_plan: SparseFactorizationPlan | None
    _admission_order: tuple[int, ...] = eqx.field(static=True)
    _storage_plan: _SparseStoragePlan

    def __init__(
        self,
        rows: ArrayLike,
        columns: ArrayLike,
        upper_values: ArrayLike,
        size: int,
        /,
        *,
        policy: LinearSolvePolicy | None = None,
    ) -> None:
        row = np.asarray(rows)
        column = np.asarray(columns)
        values = _metric_values(upper_values)
        if row.ndim != 1 or column.shape != row.shape or values.shape != row.shape:
            raise ValueError(
                "Hodge rows, columns, and values must be equal-size vectors."
            )
        if not np.issubdtype(row.dtype, np.integer) or not np.issubdtype(
            column.dtype, np.integer
        ):
            raise TypeError("Hodge pattern must contain integer coordinates.")
        if size < 0 or np.any((row < 0) | (column < row) | (column >= size)):
            raise ValueError("Hodge pattern must be upper triangular and in range.")
        pairs = tuple((int(r), int(c)) for r, c in zip(row, column, strict=True))
        if len(set(pairs)) != len(pairs):
            raise ValueError("Hodge upper-triangular routes must be unique.")
        rows_ = tuple(pair[0] for pair in pairs)
        columns_ = tuple(pair[1] for pair in pairs)
        selected = gram_solve_policy(size) if policy is None else policy
        if not isinstance(selected, LinearSolvePolicy):
            raise TypeError("policy must be a LinearSolvePolicy or None.")
        mirror = tuple(index for index, (r, c) in enumerate(pairs) if r != c)
        admission_plan, admission_order, storage_plan = _admission_pattern(
            rows_, columns_, mirror, size
        )
        layout = canonical_fingerprint(
            {
                "kind": "sparse-hodge",
                "size": size,
                "rows": rows_,
                "columns": columns_,
                "policy_structure": str(jax.tree.structure(selected)),
            }
        )
        self.rows = rows_
        self.columns = columns_
        self.upper_values = values
        self.size = size
        self.policy = selected
        self.layout_id = layout
        self._mirror = mirror
        self._admission_plan = admission_plan
        self._admission_order = admission_order
        self._storage_plan = storage_plan

    @property
    def valid(self) -> Array:
        plan = self._admission_plan
        if plan is None:
            return jnp.asarray(True)
        mirror = jnp.asarray(self._mirror, dtype=jnp.int32)
        order = jnp.asarray(self._admission_order, dtype=jnp.int32)
        values = jnp.concatenate((self.upper_values, self.upper_values[mirror]))[order]
        evidence = refresh_sparse_factorization_values(plan, values)
        return (
            (evidence.status == int(SparseFactorizationStatus.SUCCESS))
            & evidence.diagnostics.finite
            & jnp.all(jnp.isfinite(self.upper_values))
        )

    def admit(self) -> SparseHodge:
        if not bool(np.asarray(self.valid)):
            raise ValueError("A Hodge metric must be finite and positive definite.")
        return self

    def restrict(self, active: ArrayLike, /) -> SparseHodge:
        indices = _active_indices(active, self.size)
        inverse = np.full((self.size,), -1, dtype=np.int32)
        inverse[indices] = np.arange(indices.shape[0], dtype=np.int32)
        row = inverse[np.asarray(self.rows, dtype=np.int32)]
        column = inverse[np.asarray(self.columns, dtype=np.int32)]
        keep = np.flatnonzero((row >= 0) & (column >= 0))
        return SparseHodge(
            row[keep],
            column[keep],
            self.upper_values[keep],
            indices.shape[0],
            policy=self.policy,
        )

    def refresh(self, values: ArrayLike, /) -> SparseHodge:
        refreshed = _metric_values(values)
        if refreshed.shape != self.upper_values.shape:
            raise ValueError("Refreshed values must preserve the Hodge layout.")
        return eqx.tree_at(lambda hodge: hodge.upper_values, self, refreshed)

    def make_space(
        self, *, space_id: str, dtype: DTypeLike = jnp.float64
    ) -> tuple[ArraySpace, AbstractLinearOperator]:
        mirror = np.asarray(self._mirror, dtype=np.int32)
        rows = np.asarray(self.rows, dtype=np.int32)
        columns = np.asarray(self.columns, dtype=np.int32)
        values = eqx.error_if(
            self.upper_values,
            ~self.valid,
            "A Hodge metric must be finite and positive definite.",
        )
        return sparse_gram_space(
            np.concatenate((rows, columns[mirror])),
            np.concatenate((columns, rows[mirror])),
            jnp.concatenate((values, values[mirror])),
            size=self.size,
            dtype=dtype,
            space_id=space_id,
            policy=self.policy,
            storage_plan=self._storage_plan,
        )


type CochainHodge = DiagonalHodge | SparseHodge


def _simplex_measure(points: Array, /) -> Array:
    degree = points.shape[0] - 1
    if degree == 0:
        return jnp.ones((), dtype=points.dtype)
    edges = points[1:] - points[:1]
    return jnp.sqrt(jnp.maximum(jnp.linalg.det(edges @ edges.T), 0.0)) / math.factorial(
        degree
    )


def _simplex_center(points: Array, dual: DualCellPolicy, /) -> tuple[Array, Array]:
    if dual == "barycentric" or points.shape[0] == 1:
        return jnp.mean(points, axis=0), jnp.asarray(True)
    edges = points[1:] - points[:1]
    gram = edges @ edges.T
    coordinates = jnp.linalg.solve(gram, jnp.diag(gram) / 2.0)
    barycentric = jnp.concatenate(
        (1.0 - jnp.sum(coordinates, keepdims=True), coordinates)
    )
    return points[0] + coordinates @ edges, jnp.all(barycentric > 0.0)


def simplicial_dual_hodges(
    topology: CellComplexTopology, vertices: ArrayLike, /, *, dual: DualCellPolicy
) -> tuple[DiagonalHodge, ...]:
    """Build n-D dual/primal metric ratios from barycentric or well-centered flags.

    Every chain of nested cofaces contributes its geometric elementary dual
    simplex. In two dimensions this is exactly area/3 at vertices and the sum
    of face-barycenter to edge-midpoint lengths at edges.
    """
    from ._cell_complex import simplicial_cell_geometry

    policy = parse(dual, DualCellPolicy, "dual")
    if not isinstance(topology, CellComplexTopology):
        raise TypeError("topology must be a CellComplexTopology.")
    cells, _ = simplicial_cell_geometry(topology)
    points = jnp.asarray(vertices, dtype=jnp.float64)
    if (
        points.ndim != 2
        or points.shape[0] != topology.entities(0).count
        or points.shape[1] < topology.dimension
    ):
        raise ValueError(
            "vertices must give ambient coordinates for every topology vertex."
        )
    measures: list[Array] = []
    centers: list[Array] = []
    valid = jnp.all(jnp.isfinite(points))
    for degree_cells in cells:
        local = points[degree_cells]
        measure = jax.vmap(_simplex_measure)(local)
        center, centered = jax.vmap(lambda simplex: _simplex_center(simplex, policy))(
            local
        )
        measures.append(measure)
        centers.append(center)
        valid = (
            valid & jnp.all(jnp.isfinite(measure) & (measure > 0.0)) & jnp.all(centered)
        )
    dual_measures: list[Array] = []
    for degree, degree_cells in enumerate(cells):
        routes: list[tuple[int, ...]] = []
        for index in range(degree_cells.shape[0]):
            frontier = [(index,)]
            for next_degree in range(degree + 1, topology.dimension + 1):
                expanded: list[tuple[int, ...]] = []
                for flag in frontier:
                    previous = set(cells[next_degree - 1][flag[-1]].tolist())
                    for next_index, coface in enumerate(cells[next_degree].tolist()):
                        if previous.issubset(coface):
                            expanded.append((*flag, next_index))
                frontier = expanded
            routes.extend(frontier)
        if degree == topology.dimension:
            dual_measures.append(jnp.ones((degree_cells.shape[0],), dtype=points.dtype))
            continue
        route_array = np.asarray(routes, dtype=np.int32).reshape(
            (-1, topology.dimension - degree + 1)
        )
        flag_points = jnp.stack(
            tuple(
                centers[k][route_array[:, k - degree]]
                for k in range(degree, topology.dimension + 1)
            ),
            axis=1,
        )
        volumes = jax.vmap(_simplex_measure)(flag_points)
        total = (
            jnp.zeros((degree_cells.shape[0],), dtype=points.dtype)
            .at[route_array[:, 0]]
            .add(volumes)
        )
        dual_measures.append(total)
    result = tuple(
        DiagonalHodge(dual_measure / primal)
        for dual_measure, primal in zip(dual_measures, measures, strict=True)
    )
    validity = valid & jnp.all(jnp.stack(tuple(hodge.valid for hodge in result)))
    message = (
        "Circumcentric dual requires a nondegenerate well-centered simplicial mesh."
        if policy == "circumcentric"
        else "Barycentric dual requires nondegenerate simplices with positive dual measures."
    )
    return tuple(
        hodge.refresh(eqx.error_if(hodge.weights, ~validity, message)) for hodge in result
    )


__all__ = [
    "CochainHodge",
    "DiagonalHodge",
    "DualCellPolicy",
    "SparseHodge",
    "simplicial_dual_hodges",
]
