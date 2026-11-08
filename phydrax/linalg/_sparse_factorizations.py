#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from hashlib import sha256
from math import isfinite, prod
from numbers import Integral
from typing import Any, Literal, NoReturn, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax._src import dispatch
from jax._src.core import ShapedArray
from jax.extend import core as jex_core
from jax.interpreters import ad, batching, mlir
from jax.typing import ArrayLike, DTypeLike

from .._strict import StrictModule
from ..typing import parse
from . import _sparse_supernodal as _supernodal
from ._properties import LinearCapabilityError
from ._sparse_contract import AbstractSparseLinearOperator, SparseStorage
from ._sparse_ordering import (
    _order_pattern,
    _pattern_identifier,
    _validated_pattern,
    _WorkMeter,
    PreparedSparseOrdering,
    SparseOrdering,
    SparseOrderingPolicy,
)
from ._sparse_supernodal import SupernodalAnalysis
from ._sparse_triangular import (
    analyze_sparse_triangular,
    solve_sparse_triangular,
    SparseTriangularAnalysis,
    SparseTriangularStatus,
)


SparseFactorizationKind: TypeAlias = Literal["auto", "lu", "cholesky"]


class SparseFactorizationStatus(IntEnum):
    SUCCESS = 0
    ZERO_PIVOT = 1
    NONPOSITIVE_PIVOT = 2
    NONFINITE = 3


class SparseFactorizationPolicy(StrictModule):
    """Symbolic fill, ordering, dropping, and explicit pivot-replacement policy.

    ``ordering`` selects the symmetric fill-reducing ordering owned by
    `prepare_sparse_ordering`; its work is charged to ``max_symbolic_work``.
    """

    kind: SparseFactorizationKind = eqx.field(static=True)
    ordering: SparseOrdering = eqx.field(static=True)
    fill_level: int | None = eqx.field(static=True)
    drop_tolerance: float = eqx.field(static=True)
    maximum_fill_per_row: int | None = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    diagonal_shift: float = eqx.field(static=True)
    allow_pivot_replacement: bool = eqx.field(static=True)
    replacement_value: float = eqx.field(static=True)
    max_factor_nnz: int = eqx.field(static=True)
    max_factor_bytes: int = eqx.field(static=True)
    max_symbolic_work: int = eqx.field(static=True)

    def __init__(
        self,
        kind: SparseFactorizationKind = "auto",
        /,
        *,
        ordering: SparseOrdering = "natural",
        fill_level: int | None = None,
        drop_tolerance: float = 0.0,
        maximum_fill_per_row: int | None = None,
        pivot_tolerance: float = 0.0,
        diagonal_shift: float = 0.0,
        allow_pivot_replacement: bool = False,
        replacement_value: float = 1e-12,
        max_factor_nnz: int = 2_000_000,
        max_factor_bytes: int = 256 * 1024 * 1024,
        max_symbolic_work: int = 512_000_000,
    ) -> None:
        kind = parse(kind, SparseFactorizationKind, "kind")
        ordering = parse(ordering, SparseOrdering, "ordering")
        fill = None if fill_level is None else int(fill_level)
        maximum_fill = None if maximum_fill_per_row is None else int(maximum_fill_per_row)
        resource_limits = {
            "max_factor_nnz": max_factor_nnz,
            "max_factor_bytes": max_factor_bytes,
            "max_symbolic_work": max_symbolic_work,
        }
        for name, value in resource_limits.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a host integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if fill is not None and fill < 0:
            raise ValueError("fill_level must be non-negative or None.")
        if maximum_fill is not None and maximum_fill < 0:
            raise ValueError("maximum_fill_per_row must be non-negative or None.")
        numeric = tuple(
            float(value)
            for value in (
                drop_tolerance,
                pivot_tolerance,
                diagonal_shift,
                replacement_value,
            )
        )
        if any(not isfinite(value) for value in numeric):
            raise ValueError("Sparse factorization numeric policies must be finite.")
        if numeric[0] < 0.0 or numeric[1] < 0.0 or numeric[2] < 0.0:
            raise ValueError("Drop, pivot, and shift tolerances must be non-negative.")
        if numeric[3] <= 0.0:
            raise ValueError("replacement_value must be positive.")
        self.kind = kind
        self.ordering = ordering
        self.fill_level = fill
        self.drop_tolerance = numeric[0]
        self.maximum_fill_per_row = maximum_fill
        self.pivot_tolerance = numeric[1]
        self.diagonal_shift = numeric[2]
        self.allow_pivot_replacement = bool(allow_pivot_replacement)
        self.replacement_value = numeric[3]
        self.max_factor_nnz = int(max_factor_nnz)
        self.max_factor_bytes = int(max_factor_bytes)
        self.max_symbolic_work = int(max_symbolic_work)


class SparseEliminationSchedule(StrictModule):
    """Pivot-by-pivot factor pattern of incomplete and static-pivot LU plans.

    It stores only the factor pattern: CSR rows of the combined factor and a
    column-major index of its strictly lower entries. The numeric kernel
    derives every elimination target at runtime from that pattern, so stored
    bytes scale with the factor nonzeros, not with the number of elimination
    updates. ``row_width``, ``column_width`` and ``upper_width`` bound the
    fixed per-pivot windows: the longest factor row, the longest strictly
    lower column, and the longest strictly upper row.
    """

    factor_indices: Array
    factor_indptr: Array
    factor_rows: Array
    input_positions: Array
    input_conjugate: Array
    column_positions: Array
    column_offsets: Array
    lower_positions: Array
    upper_positions: Array | None
    lower_analysis: SparseTriangularAnalysis
    upper_analysis: SparseTriangularAnalysis | None
    row_width: int = eqx.field(static=True)
    column_width: int = eqx.field(static=True)
    upper_width: int = eqx.field(static=True)


class SparseFactorizationPlan(StrictModule):
    """Immutable host symbolic plan for refreshable sparse LU or Cholesky values.

    Complete Cholesky plans (``fill_level=None``, no dropping) carry a
    ``supernodal`` analysis: elimination tree, column counts and relaxed
    supernodes, factored by dense multifrontal blocks into column-major
    supernode panels. Incomplete and LU plans carry an ``elimination``
    schedule factored pivot by pivot on the combined CSR factor pattern.
    ``factor_size`` is the length of the numeric factor values,
    ``factor_nnz`` the structural nonzeros of the factor, and
    ``factor_flops`` the floating-point operations of one numeric refresh
    (zero where the pattern alone does not determine them).
    ``diagonal_positions[k]`` locates pivot ``k`` in the factor values.
    Supernodal plans also carry ``entry_routes`` (rows: input position,
    original row, original column) and ``entry_conjugate`` for the one stored
    entry per symmetric pair that the factor reads, so solves differentiate
    through the factored operator itself.
    ``ordering_id`` identifies the symmetric ordering ``permutation`` and
    ``ordering_work`` is its share of ``symbolic_work``.
    ``value_dtype`` fixes the numeric dtype admitted by the resource estimate;
    numeric refresh cannot widen retained factor or workspace allocations.
    """

    permutation: Array
    inverse_permutation: Array
    diagonal_positions: Array
    elimination: SparseEliminationSchedule | None
    entry_routes: Array | None
    entry_conjugate: Array | None
    shape: tuple[int, int] = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    kind: Literal["lu", "cholesky"] = eqx.field(static=True)
    policy: SparseFactorizationPolicy = eqx.field(static=True)
    input_pattern_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    input_nnz: int = eqx.field(static=True)
    factor_nnz: int = eqx.field(static=True)
    factor_size: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)
    factor_flops: int = eqx.field(static=True)
    symbolic_work: int = eqx.field(static=True)
    ordering_id: str = eqx.field(static=True)
    ordering_work: int = eqx.field(static=True)
    value_dtype: str = eqx.field(static=True)
    supernodal: SupernodalAnalysis | None = eqx.field(static=True, default=None)
    storage_plan: Any = None


class SparseFactorizationDiagnostics(StrictModule):
    """Numerical refresh evidence for one fixed symbolic factor pattern.

    ``rejected_pivots`` counts pivots that failed the acceptance test
    (non-positive or non-finite for Cholesky, below tolerance for LU), whether
    or not they were replaced; a positive count refutes definiteness.
    """

    minimum_pivot: Array
    replaced_pivots: Array
    rejected_pivots: Array
    dropped_entries: Array
    input_nonzeros: Array
    factor_nonzeros: Array
    fill_ratio: Array
    finite: Array


class SparseFactorizationSolveResult(StrictModule):
    """Sparse-factor solve with factor and triangular status evidence."""

    value: Array
    status: Array
    factorization_status: Array
    lower_status: Array
    upper_status: Array

    @property
    def success(self) -> Array:
        return self.status == int(SparseFactorizationStatus.SUCCESS)


class PreparedSparseFactorization(StrictModule):
    """Refreshable sparse factor values paired with immutable symbolic analysis.

    Supernodal factors keep the refreshed ``input_values``: their solves are
    implicit linear solves whose derivatives with respect to the values and
    right-hand sides are solves with the same prepared panels, in forward and
    reverse mode and at every order, so no derivative refactors the values.
    The factor panels and minimum pivot differentiate in forward mode at every
    order from the stored panels; reverse mode through them raises
    `SparseFactorDerivativeError`.
    """

    plan: SparseFactorizationPlan
    factor_values: Array
    status: Array
    diagnostics: SparseFactorizationDiagnostics
    factorization_id: str = eqx.field(static=True)
    input_values: Array | None = None

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.plan.batch_shape

    def solve(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        adjoint: bool = False,
    ) -> SparseFactorizationSolveResult:
        """Solve with the prepared factors, optionally for the conjugate transpose."""
        rhs = jnp.asarray(right_hand_side)
        size = self.plan.shape[0]
        shared_vector = rhs.shape == (size,)
        batched_vector = rhs.shape == self.batch_shape + (size,)
        shared_matrix = rhs.ndim == 2 and rhs.shape[0] == size
        batched_matrix = (
            rhs.ndim == len(self.batch_shape) + 2
            and rhs.shape[: len(self.batch_shape)] == self.batch_shape
            and rhs.shape[-2] == size
        )
        if shared_vector:
            rhs = jnp.broadcast_to(rhs, self.batch_shape + (size,))[..., None]
            vector_input = True
        elif batched_vector:
            rhs = rhs[..., None]
            vector_input = True
        elif shared_matrix:
            rhs = jnp.broadcast_to(rhs, self.batch_shape + rhs.shape)
            vector_input = False
        elif batched_matrix:
            vector_input = False
        else:
            raise ValueError(
                "right_hand_side must have shape (n,), (n, k), batch_shape + (n,), or batch_shape + (n, k)."
            )
        batch_count = int(np.prod(self.batch_shape)) if self.batch_shape else 1
        factors = self.factor_values.reshape((batch_count, -1))
        statuses = self.status.reshape((batch_count,))
        right_hand_sides = rhs.reshape((batch_count, size, rhs.shape[-1]))
        if self.plan.supernodal is not None:
            if self.input_values is None:
                raise ValueError(
                    "A supernodal factorization must retain its input values."
                )
            # A Cholesky factor is Hermitian: the adjoint solve is the solve.
            plan = self.plan
            solution, (lower_status, upper_status) = jax.vmap(
                lambda factor_values, values, value: _supernodal_solve(
                    plan, factor_values, values, value
                )
            )(
                factors,
                self.input_values.reshape((batch_count, plan.input_nnz)),
                right_hand_sides,
            )
        else:
            schedule = self.plan.elimination
            if schedule is None:
                raise ValueError("Sparse factorization plan has no elimination schedule.")
            plan = self.plan
            solution, lower_status, upper_status = jax.vmap(
                lambda factor_values, value: _elimination_solve(
                    plan, schedule, factor_values, value, adjoint=adjoint
                )
            )(factors, right_hand_sides)
        status = _solve_status(statuses, lower_status, upper_status)
        value = solution.reshape(self.batch_shape + (size, rhs.shape[-1]))
        if vector_input:
            value = value[..., 0]
        return SparseFactorizationSolveResult(
            value=value,
            status=status.reshape(self.batch_shape),
            factorization_status=self.status,
            lower_status=lower_status.reshape(self.batch_shape),
            upper_status=upper_status.reshape(self.batch_shape),
        )


def _elimination_solve(
    plan: SparseFactorizationPlan,
    schedule: SparseEliminationSchedule,
    factor_values: Array,
    value: Array,
    /,
    *,
    adjoint: bool,
) -> tuple[Array, Array, Array]:
    """Level-scheduled triangular sweeps of one pivot-by-pivot factor."""
    permuted_rhs = value[plan.permutation]
    lower_values = factor_values[schedule.lower_positions]
    tolerance = plan.policy.pivot_tolerance
    if plan.kind == "lu":
        lower_values = jnp.where(
            schedule.lower_analysis.row_indices == schedule.lower_analysis.indices,
            jnp.ones((), dtype=lower_values.dtype),
            lower_values,
        )
        if schedule.upper_analysis is None or schedule.upper_positions is None:
            raise ValueError("LU plan is missing upper triangular analysis.")
        upper_values = factor_values[schedule.upper_positions]
        if adjoint:
            upper = solve_sparse_triangular(
                schedule.upper_analysis,
                upper_values,
                permuted_rhs,
                pivot_tolerance=tolerance,
                adjoint=True,
            )
            lower = solve_sparse_triangular(
                schedule.lower_analysis,
                lower_values,
                upper.value,
                pivot_tolerance=tolerance,
                adjoint=True,
            )
            permuted_solution = lower.value
        else:
            lower = solve_sparse_triangular(
                schedule.lower_analysis,
                lower_values,
                permuted_rhs,
                pivot_tolerance=tolerance,
            )
            upper = solve_sparse_triangular(
                schedule.upper_analysis,
                upper_values,
                lower.value,
                pivot_tolerance=tolerance,
            )
            permuted_solution = upper.value
    else:
        lower = solve_sparse_triangular(
            schedule.lower_analysis,
            lower_values,
            permuted_rhs,
            pivot_tolerance=tolerance,
        )
        upper = solve_sparse_triangular(
            schedule.lower_analysis,
            lower_values,
            lower.value,
            pivot_tolerance=tolerance,
            adjoint=True,
        )
        permuted_solution = upper.value
    solution = (
        jnp.zeros_like(permuted_solution).at[plan.permutation].set(permuted_solution)
    )
    return solution, lower.status, upper.status


def _solve_status(
    factor_status: Array, lower_status: Array, upper_status: Array, /
) -> Array:
    """Combine factor and triangular sweep statuses; non-finite dominates."""
    triangular_success = (lower_status == int(SparseTriangularStatus.SUCCESS)) & (
        upper_status == int(SparseTriangularStatus.SUCCESS)
    )
    triangular_nonfinite = (lower_status == int(SparseTriangularStatus.NONFINITE)) | (
        upper_status == int(SparseTriangularStatus.NONFINITE)
    )
    # Any other non-success sweep status is a zero triangular pivot.
    triangular_status = jnp.where(
        triangular_nonfinite,
        int(SparseFactorizationStatus.NONFINITE),
        jnp.where(
            triangular_success,
            int(SparseFactorizationStatus.SUCCESS),
            int(SparseFactorizationStatus.ZERO_PIVOT),
        ),
    )
    any_nonfinite = (factor_status == int(SparseFactorizationStatus.NONFINITE)) | (
        triangular_status == int(SparseFactorizationStatus.NONFINITE)
    )
    return jnp.where(
        any_nonfinite,
        int(SparseFactorizationStatus.NONFINITE),
        jnp.where(
            factor_status != int(SparseFactorizationStatus.SUCCESS),
            factor_status,
            triangular_status,
        ),
    ).astype(jnp.int32)


@dataclass(frozen=True)
class _SupernodalSolveCallback:
    """Host supernodal sweeps over any leading batch axes of factors and RHS."""

    analysis: SupernodalAnalysis

    def __call__(
        self, factors: np.ndarray, right_hand_sides: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        factors = np.asarray(factors)
        right_hand_sides = np.asarray(right_hand_sides)
        batch = np.broadcast_shapes(factors.shape[:-1], right_hand_sides.shape[:-2])
        factors = np.broadcast_to(factors, batch + factors.shape[-1:])
        right_hand_sides = np.broadcast_to(
            right_hand_sides, batch + right_hand_sides.shape[-2:]
        )
        dtype = np.result_type(factors.dtype, right_hand_sides.dtype)
        solutions = np.empty(batch + right_hand_sides.shape[-2:], dtype=dtype)
        lower = np.empty(batch, dtype=np.int32)
        upper = np.empty(batch, dtype=np.int32)
        for index in np.ndindex(*batch):
            solutions[index], lower[index], upper[index] = _supernodal.solve_values(
                self.analysis, factors[index], right_hand_sides[index]
            )
        return solutions, lower, upper


def _supernodal_matvec(
    plan: SparseFactorizationPlan, values: Array, value: Array, /
) -> Array:
    """``A x`` for the shifted Hermitian operator the supernodal factor reads."""
    if plan.entry_routes is None or plan.entry_conjugate is None:
        raise ValueError("A supernodal plan must carry its factored entry routes.")
    sources, rows, columns = plan.entry_routes
    entries = values[sources]
    entries = jnp.where(plan.entry_conjugate, jnp.conj(entries), entries)
    mirrored = jnp.where(rows == columns, jnp.zeros((), entries.dtype), jnp.conj(entries))
    result = jnp.zeros_like(value).at[rows].add(entries[:, None] * value[columns])
    result = result.at[columns].add(mirrored[:, None] * value[rows])
    return result + plan.policy.diagonal_shift * value


def _supernodal_solve(
    plan: SparseFactorizationPlan,
    factor_values: Array,
    values: Array,
    right_hand_side: Array,
    /,
) -> tuple[Array, tuple[Array, Array]]:
    """Supernodal sweeps of one ``(n, k)`` block as an implicit linear solve.

    The host sweeps apply the stored panels; `jax.lax.custom_linear_solve`
    differentiates the solution through the factored operator's action on the
    input values, so tangents and cotangents reuse the same host sweeps.
    """
    analysis = plan.supernodal
    if analysis is None:
        raise ValueError("A supernodal solve requires a supernodal plan.")
    panels = jax.lax.stop_gradient(factor_values)
    dtype = jnp.result_type(factor_values.dtype, right_hand_side.dtype)
    complex_ = jnp.issubdtype(dtype, jnp.complexfloating)
    callback = _SupernodalSolveCallback(analysis)

    def sweep(block: Array, /) -> tuple[Array, tuple[Array, Array]]:
        solution, lower, upper = jax.pure_callback(
            callback,
            (
                jax.ShapeDtypeStruct(block.shape, dtype),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
            ),
            panels,
            block,
            vmap_method="broadcast_all",
        )
        return solution, (lower, upper)

    def solve(_: Any, block: Array, /) -> tuple[Array, tuple[Array, Array]]:
        return sweep(block)

    def transpose_solve(_: Any, block: Array, /) -> tuple[Array, tuple[Array, Array]]:
        # The linear transpose of a Hermitian operator is its conjugate.
        if not complex_:
            return sweep(block)
        solution, statuses = sweep(jnp.conj(block))
        return jnp.conj(solution), statuses

    return jax.lax.custom_linear_solve(
        lambda value: _supernodal_matvec(plan, values, value),
        right_hand_side.astype(dtype),
        solve,
        transpose_solve,
        has_aux=True,
    )


@dataclass
class _SymbolicResourceTracker:
    size: int
    kind: Literal["lu", "cholesky"]
    batch_count: int
    value_itemsize: int
    index_itemsize: int
    max_factor_nnz: int
    max_factor_bytes: int
    max_symbolic_work: int
    base_bytes: int
    supernodal: bool = False
    factor_nnz: int = 0
    lower_nnz: int = 0
    strictly_lower_nnz: int = 0
    upper_nnz: int = 0
    factor_bytes: int = 0
    symbolic_work: int = 0

    def _refuse(
        self,
        metric: str,
        required: int,
        limit: int,
        /,
        *,
        factor_nnz: int | None = None,
        factor_bytes: int | None = None,
        symbolic_work: int | None = None,
    ) -> NoReturn:
        observed_nnz = self.factor_nnz if factor_nnz is None else factor_nnz
        observed_bytes = self.factor_bytes if factor_bytes is None else factor_bytes
        observed_work = self.symbolic_work if symbolic_work is None else symbolic_work
        raise LinearCapabilityError(
            "Sparse symbolic factorization refused before allocation: "
            f"{metric} requires {required}, exceeding limit {limit}; "
            f"factor_nnz={observed_nnz}/{self.max_factor_nnz}, "
            f"factor_bytes={observed_bytes}/{self.max_factor_bytes}, "
            f"symbolic_work={observed_work}/{self.max_symbolic_work}."
        )

    def _retained_bytes(
        self,
        factor_nnz: int,
        lower_nnz: int,
        strictly_lower_nnz: int,
        upper_nnz: int,
        /,
    ) -> int:
        index = self.index_itemsize
        if self.supernodal:
            # Permutation, inverse permutation and pivot positions; panels and
            # host indices are admitted in bulk by `admit_supernodal`.
            return self.base_bytes + 3 * index * self.size
        triangular_fixed = index * (4 * self.size + 2) + 8 * self.size
        fixed = self.base_bytes + index * (5 * self.size + 2) + triangular_fixed
        if self.kind == "lu":
            fixed += triangular_fixed
        return (
            fixed
            + self.batch_count * factor_nnz * self.value_itemsize
            + factor_nnz * (3 * index + 1)
            + strictly_lower_nnz * index
            + lower_nnz * 6 * index
            + upper_nnz * 6 * index
        )

    def reserve_fixed_bytes(self, /) -> None:
        required = self._retained_bytes(0, 0, 0, 0)
        if required > self.max_factor_bytes:
            self._refuse(
                "factor_bytes",
                required,
                self.max_factor_bytes,
                factor_bytes=required,
            )
        self.factor_bytes = required

    def add_fixed_bytes(self, count: int, /) -> None:
        projected = self.factor_bytes + count
        if projected > self.max_factor_bytes:
            self._refuse(
                "factor_bytes",
                projected,
                self.max_factor_bytes,
                factor_bytes=projected,
            )
        self.factor_bytes = projected

    def add_work(self, count: int = 1, /) -> None:
        projected = self.symbolic_work + count
        if projected > self.max_symbolic_work:
            self._refuse(
                "symbolic_work",
                projected,
                self.max_symbolic_work,
                symbolic_work=projected,
            )
        self.symbolic_work = projected

    def admit_supernodal(
        self, structural_nnz: int, panel_entries: int, index_bytes: int, /
    ) -> None:
        """Admit a counted supernodal factor before any panel is allocated."""
        if structural_nnz > self.max_factor_nnz:
            self._refuse(
                "factor_nnz",
                structural_nnz,
                self.max_factor_nnz,
                factor_nnz=structural_nnz,
            )
        required = (
            self._retained_bytes(0, 0, 0, 0)
            + self.batch_count * panel_entries * self.value_itemsize
            + index_bytes
        )
        if required > self.max_factor_bytes:
            self._refuse(
                "factor_bytes",
                required,
                self.max_factor_bytes,
                factor_nnz=structural_nnz,
                factor_bytes=required,
            )
        self.factor_nnz = structural_nnz
        self.lower_nnz = structural_nnz
        self.factor_bytes = required

    def add_factor_entry(self, row: int, column: int, /) -> None:
        projected_nnz = self.factor_nnz + 1
        projected_lower = self.lower_nnz + (column <= row)
        projected_strictly_lower = self.strictly_lower_nnz + (column < row)
        projected_upper = self.upper_nnz + (self.kind == "lu" and column >= row)
        projected_bytes = self._retained_bytes(
            projected_nnz,
            projected_lower,
            projected_strictly_lower,
            projected_upper,
        )
        if projected_nnz > self.max_factor_nnz:
            self._refuse(
                "factor_nnz",
                projected_nnz,
                self.max_factor_nnz,
                factor_nnz=projected_nnz,
                factor_bytes=projected_bytes,
            )
        if projected_bytes > self.max_factor_bytes:
            self._refuse(
                "factor_bytes",
                projected_bytes,
                self.max_factor_bytes,
                factor_nnz=projected_nnz,
                factor_bytes=projected_bytes,
            )
        self.factor_nnz = projected_nnz
        self.lower_nnz = projected_lower
        self.strictly_lower_nnz = projected_strictly_lower
        self.upper_nnz = projected_upper
        self.factor_bytes = projected_bytes


def _symbolic_resource_tracker(
    storage: SparseStorage,
    policy: SparseFactorizationPolicy,
    kind: Literal["lu", "cholesky"],
    base_bytes: int,
    /,
    *,
    supernodal: bool,
) -> _SymbolicResourceTracker:
    tracker = _SymbolicResourceTracker(
        size=storage.shape[0],
        kind=kind,
        batch_count=prod(storage.batch_shape or (1,)),
        value_itemsize=storage.values.dtype.itemsize,
        index_itemsize=storage.indices.dtype.itemsize,
        max_factor_nnz=policy.max_factor_nnz,
        max_factor_bytes=policy.max_factor_bytes,
        max_symbolic_work=policy.max_symbolic_work,
        base_bytes=base_bytes,
        supernodal=supernodal,
    )
    tracker.reserve_fixed_bytes()
    return tracker


def _resolved_ordering(
    storage: SparseStorage,
    indices: np.ndarray,
    indptr: np.ndarray,
    pattern_id: str,
    policy: SparseFactorizationPolicy,
    ordering: PreparedSparseOrdering | None,
    tracker: _SymbolicResourceTracker,
    /,
) -> PreparedSparseOrdering:
    """Charge a prepared ordering, or prepare the policy's, before any fill.

    Ordering work is symbolic work: it is charged to ``max_symbolic_work``
    before diagonal or fill entries are reserved, whether it was prepared here
    or earlier, so equal orderings give equal plans.
    """
    if ordering is None:
        return _order_pattern(
            storage.shape[0],
            indices,
            indptr,
            SparseOrderingPolicy(policy.ordering),
            None,
            _WorkMeter(None, tracker.add_work),
        )
    if not isinstance(ordering, PreparedSparseOrdering):
        raise TypeError("ordering must be a PreparedSparseOrdering or None.")
    if ordering.method != policy.ordering:
        raise ValueError(
            f"Prepared ordering method {ordering.method!r} differs from the "
            f"policy ordering {policy.ordering!r}."
        )
    if ordering.pattern_id != pattern_id:
        raise ValueError("Prepared ordering belongs to a different sparse pattern.")
    tracker.add_work(ordering.work)
    return ordering


def _permuted_entries(
    indices: np.ndarray,
    indptr: np.ndarray,
    permutation: np.ndarray,
    tracker: _SymbolicResourceTracker,
    /,
) -> dict[tuple[int, int], tuple[int, bool]]:
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    entries: dict[tuple[int, int], tuple[int, bool]] = {}
    for old_row in range(permutation.size):
        new_row = int(inverse[old_row])
        for position in range(indptr[old_row], indptr[old_row + 1]):
            tracker.add_work()
            new_column = int(inverse[indices[position]])
            if tracker.kind == "lu":
                coordinate = (new_row, new_column)
                conjugate = False
            else:
                coordinate = (
                    max(new_row, new_column),
                    min(new_row, new_column),
                )
                conjugate = new_row < new_column
            previous = entries.get(coordinate)
            if previous is None:
                if coordinate[0] != coordinate[1]:
                    tracker.add_factor_entry(*coordinate)
                entries[coordinate] = (position, conjugate)
            elif previous[1] and not conjugate:
                entries[coordinate] = (position, False)
    return entries


def _insert_symbolic_entry(
    rows: list[dict[int, int]],
    column_rows: list[set[int]],
    row: int,
    column: int,
    level: int,
    tracker: _SymbolicResourceTracker,
    /,
) -> None:
    previous = rows[row].get(column)
    if previous is None:
        tracker.add_factor_entry(row, column)
        rows[row][column] = level
        column_rows[column].add(row)
    elif level < previous:
        rows[row][column] = level


def _lu_symbolic_rows(
    size: int,
    entries: dict[tuple[int, int], tuple[int, bool]],
    fill_level: int | None,
    tracker: _SymbolicResourceTracker,
    /,
) -> list[dict[int, int]]:
    rows = [dict() for _ in range(size)]
    column_rows = [set() for _ in range(size)]
    for row, column in entries:
        rows[row][column] = 0
        column_rows[column].add(row)
    for row in range(size):
        rows[row].setdefault(row, 0)
        column_rows[row].add(row)
    if fill_level == 0:
        return rows
    for pivot in range(size):
        upper = [
            (column, level)
            for column, level in sorted(rows[pivot].items())
            if column > pivot
        ]
        below = [row for row in sorted(column_rows[pivot]) if row > pivot]
        for row in below:
            lower_level = rows[row][pivot]
            for column, upper_level in upper:
                tracker.add_work()
                level = lower_level + upper_level + 1
                if fill_level is None or level <= fill_level:
                    _insert_symbolic_entry(
                        rows,
                        column_rows,
                        row,
                        column,
                        level,
                        tracker,
                    )
    return rows


def _cholesky_symbolic_rows(
    size: int,
    entries: dict[tuple[int, int], tuple[int, bool]],
    fill_level: int | None,
    tracker: _SymbolicResourceTracker,
    /,
) -> list[dict[int, int]]:
    rows = [dict() for _ in range(size)]
    column_rows = [set() for _ in range(size)]
    for row, column in entries:
        rows[row][column] = 0
        column_rows[column].add(row)
    for row in range(size):
        rows[row].setdefault(row, 0)
        column_rows[row].add(row)
    if fill_level == 0:
        return rows
    for pivot in range(size):
        neighbors = [row for row in sorted(column_rows[pivot]) if row > pivot]
        for left_index, row in enumerate(neighbors):
            left_level = rows[row][pivot]
            for column in neighbors[: left_index + 1]:
                tracker.add_work()
                right_level = rows[column][pivot]
                level = left_level + right_level + 1
                if fill_level is None or level <= fill_level:
                    _insert_symbolic_entry(
                        rows,
                        column_rows,
                        row,
                        column,
                        level,
                        tracker,
                    )
    return rows


def _csr_from_rows(
    rows: list[dict[int, int]],
    tracker: _SymbolicResourceTracker,
    /,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    indices: list[int] = []
    indptr = [0]
    positions: dict[tuple[int, int], int] = {}
    for row, columns in enumerate(rows):
        for column in sorted(columns):
            tracker.add_work()
            positions[(row, column)] = len(indices)
            indices.append(column)
        indptr.append(len(indices))
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(indptr, dtype=np.int64),
        positions,
    )


def _column_index(
    indices: np.ndarray,
    indptr: np.ndarray,
    diagonal: np.ndarray,
    tracker: _SymbolicResourceTracker,
    /,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Return the column-major index of strictly lower factor entries and widths.

    ``column_positions[column_offsets[k] : column_offsets[k + 1]]`` are the CSR
    positions of the entries ``(i, k)`` with ``i > k`` in ascending row order.
    The widths are the longest factor row, strictly lower column and strictly
    upper row.
    """
    tracker.add_work(indices.size)
    size = indptr.size - 1
    rows = np.repeat(np.arange(size, dtype=np.int64), np.diff(indptr))
    strictly_lower = np.flatnonzero(indices < rows)
    columns = indices[strictly_lower]
    # A stable sort keeps CSR (ascending-row) order inside every column.
    column_positions = strictly_lower[np.argsort(columns, kind="stable")]
    counts = np.bincount(columns, minlength=size)
    column_offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return (
        column_positions,
        column_offsets,
        int(np.diff(indptr).max(initial=0)),
        int(counts.max(initial=0)),
        int((indptr[1:] - diagonal - 1).max(initial=0)),
    )


def _triangular_pattern(
    rows: list[dict[int, int]],
    combined_positions: dict[tuple[int, int], int],
    triangle: Literal["lower", "upper"],
    index_dtype: DTypeLike,
    tracker: _SymbolicResourceTracker,
    /,
    *,
    unit_diagonal: bool,
) -> tuple[SparseTriangularAnalysis, np.ndarray]:
    selected: list[list[int]] = []
    factor_positions: list[int] = []
    for row, columns in enumerate(rows):
        kept: list[int] = []
        for column in sorted(columns):
            tracker.add_work()
            if column <= row if triangle == "lower" else column >= row:
                kept.append(column)
        selected.append(kept)
        factor_positions.extend(combined_positions[(row, column)] for column in kept)
    triangular_nnz = len(factor_positions)
    tracker.add_work(2 * triangular_nnz)
    indices = np.asarray([column for row in selected for column in row], dtype=np.int64)
    indptr = np.concatenate(
        ([0], np.cumsum([len(row) for row in selected], dtype=np.int64))
    )
    storage = SparseStorage(
        jnp.ones((indices.size,), dtype=jnp.float64),
        jnp.asarray(indices, dtype=index_dtype),
        jnp.asarray(indptr, dtype=index_dtype),
        shape=(len(rows), len(rows)),
    )
    return (
        analyze_sparse_triangular(
            storage,
            triangle=triangle,
            unit_diagonal=unit_diagonal,
        ),
        np.asarray(factor_positions, dtype=np.int64),
    )


# Symbolic planning reads concrete host patterns. Evaluate it eagerly even
# when a caller prepares inside an ambient trace such as a guarded cond.
@jax.ensure_compile_time_eval()
def prepare_sparse_factorization(
    operator: AbstractSparseLinearOperator,
    policy: SparseFactorizationPolicy | None = None,
    /,
    *,
    ordering: PreparedSparseOrdering | None = None,
) -> SparseFactorizationPlan:
    """Build a bounded host symbolic factorization plan without reading values.

    Factor fill is bounded by the policy's own ``max_factor_nnz``,
    ``max_factor_bytes`` and ``max_symbolic_work``. Factoring stored sparse
    values is not a dense materialization, so ``MaterializationPolicy`` does not
    apply; solve plans charge the retained factor to
    ``SolveResourcePolicy.preconditioner_bytes``.

    ``ordering`` reuses a `PreparedSparseOrdering` of this exact pattern whose
    method matches ``policy.ordering``; otherwise the policy's ordering is
    prepared with default capacities. Ordering work is charged first, so a
    work cap below it refuses before any factor entry is reserved.
    """
    policy_ = SparseFactorizationPolicy() if policy is None else policy
    if not isinstance(policy_, SparseFactorizationPolicy):
        raise TypeError("policy must be SparseFactorizationPolicy or None.")
    storage, input_indices, input_indptr = _validated_pattern(operator)
    # Canonicalization is symbolic work. Retain its route-to-CSR scatter for
    # numeric Jacobians carried through cond/while_loop, where even invariant
    # relation indices are traced arrays and cannot be converted to NumPy.
    from ..sparse._linear import (
        _SparseStoragePlan,
        SparseCoordinateOperator,
        SparseLinearMap,
    )

    if (
        isinstance(operator, SparseCoordinateOperator)
        and operator._storage_plan is not None
    ):
        storage_plan = operator._storage_plan
    elif isinstance(operator, (SparseCoordinateOperator, SparseLinearMap)):
        storage_plan = _SparseStoragePlan(
            operator.relation,
            block_shape=operator.block_shape
            if isinstance(operator, SparseCoordinateOperator)
            else None,
        )
    else:
        storage_plan = None
    storage_plan_arrays = (
        {}
        if storage_plan is None
        else {
            id(leaf): leaf for leaf in jax.tree.leaves(storage_plan) if eqx.is_array(leaf)
        }
    )
    base_bytes = sum(
        array.size * array.dtype.itemsize for array in storage_plan_arrays.values()
    )
    kind: Literal["lu", "cholesky"]
    if policy_.kind == "auto":
        kind = "cholesky" if operator.properties.certifies("positive_definite") else "lu"
    else:
        kind = policy_.kind
    if kind == "cholesky" and not operator.properties.certifies("self_adjoint"):
        raise ValueError("Sparse Cholesky requires a certified self-adjoint operator.")
    supernodal = _supernodal_route(kind, policy_)
    tracker = _symbolic_resource_tracker(
        storage, policy_, kind, base_bytes, supernodal=supernodal
    )
    input_pattern_id = _pattern_identifier(storage.shape, input_indices, input_indptr)
    prepared_ordering = _resolved_ordering(
        storage,
        input_indices,
        input_indptr,
        input_pattern_id,
        policy_,
        ordering,
        tracker,
    )
    ordering_work = tracker.symbolic_work
    index_dtype = storage.indices.dtype
    analysis: SupernodalAnalysis | None = None
    elimination: SparseEliminationSchedule | None = None
    entry_routes: Array | None = None
    entry_conjugate: Array | None = None
    if supernodal:
        analysis, routes, conjugate = _supernodal_analysis(
            storage, input_indices, input_indptr, prepared_ordering, tracker
        )
        entry_routes = jnp.asarray(routes, dtype=index_dtype)
        entry_conjugate = jnp.asarray(conjugate)
        diagonal = analysis.diagonal_targets
        factor_size = analysis.factor_size
        factor_flops = analysis.factor_flops
        pattern_payload = (analysis.analysis_id.encode(),)
    else:
        elimination, diagonal, factor_flops, pattern_payload = _elimination_schedule(
            storage,
            input_indices,
            input_indptr,
            prepared_ordering,
            kind,
            policy_,
            tracker,
        )
        factor_size = int(elimination.factor_indices.size)
    plan_payload = b"|".join(
        (
            input_pattern_id.encode(),
            kind.encode(),
            prepared_ordering.ordering_id.encode(),
            str(policy_.fill_level).encode(),
            str(storage.batch_shape).encode(),
            str(storage.index_width).encode(),
            str(policy_.max_factor_nnz).encode(),
            str(policy_.max_factor_bytes).encode(),
            str(policy_.max_symbolic_work).encode(),
            str(tracker.factor_bytes).encode(),
            str(tracker.symbolic_work).encode(),
            *pattern_payload,
        )
    )
    if elimination is not None:
        # Level schedules are sized by the analyzed level histogram, so they
        # are charged once both analyses exist; the plan identity keeps the
        # pattern-determined bytes counted during symbolic construction.
        tracker.add_fixed_bytes(
            sum(
                triangular.level_schedule.nbytes
                + triangular.transpose_level_schedule.nbytes
                for triangular in (
                    elimination.lower_analysis,
                    elimination.upper_analysis,
                )
                if triangular is not None
            )
        )
    return SparseFactorizationPlan(
        permutation=jnp.asarray(prepared_ordering.permutation, dtype=index_dtype),
        inverse_permutation=jnp.asarray(
            prepared_ordering.inverse_permutation, dtype=index_dtype
        ),
        diagonal_positions=jnp.asarray(diagonal, dtype=index_dtype),
        elimination=elimination,
        entry_routes=entry_routes,
        entry_conjugate=entry_conjugate,
        shape=storage.shape,
        batch_shape=storage.batch_shape,
        kind=kind,
        policy=policy_,
        input_pattern_id=input_pattern_id,
        plan_id=sha256(plan_payload).hexdigest(),
        input_nnz=input_indices.size,
        factor_nnz=tracker.factor_nnz,
        factor_size=factor_size,
        factor_bytes=tracker.factor_bytes,
        factor_flops=factor_flops,
        symbolic_work=tracker.symbolic_work,
        ordering_id=prepared_ordering.ordering_id,
        ordering_work=ordering_work,
        value_dtype=storage.values.dtype.name,
        supernodal=analysis,
        storage_plan=storage_plan,
    )


def _supernodal_route(
    kind: Literal["lu", "cholesky"], policy: SparseFactorizationPolicy, /
) -> bool:
    """Complete Cholesky factors are supernodal; incomplete and LU are not."""
    return (
        kind == "cholesky"
        and policy.fill_level is None
        and policy.drop_tolerance == 0.0
        and policy.maximum_fill_per_row is None
    )


def _supernodal_analysis(
    storage: SparseStorage,
    indices: np.ndarray,
    indptr: np.ndarray,
    ordering: PreparedSparseOrdering,
    tracker: _SymbolicResourceTracker,
    /,
) -> tuple[SupernodalAnalysis, np.ndarray, np.ndarray]:
    """Counted supernodal symbolic factorization, admitted before allocation.

    The elimination tree, postorder and column counts read every stored entry
    a bounded number of times and are charged up front. Column counts give
    the exact factor nonzeros and the amalgamated panel sizes before any row
    structure or numeric panel exists, so caps refuse at that point. Retained
    bytes cover the panels, the host analysis, the factored entry routes, and
    the input values a prepared factorization keeps for its solves.

    Returns the analysis, the ``(3, entries)`` routes (input position,
    original row, original column) of the factored entries, and their
    conjugation flags.
    """
    size = storage.shape[0]
    tracker.add_work(2 * indices.size + 2 * size)
    counts = _supernodal.symbolic_counts(
        size, indices, indptr, ordering.inverse_permutation
    )
    boundaries, heights = _supernodal.relaxed_supernodes(
        counts.parent, counts.column_counts
    )
    widths = np.diff(boundaries)
    entries = int(counts.source_rows.size)
    rows_total = int(heights.sum())
    index = storage.indices.dtype.itemsize
    tracker.admit_supernodal(
        int(counts.column_counts.sum()),
        int(np.sum(widths * heights)) + indices.size,
        _supernodal.retained_index_bytes(size, widths.size, rows_total, entries)
        + entries * (3 * index + 1),
    )
    tracker.add_work(2 * rows_total + entries)
    analysis = _supernodal.analyze_supernodal(
        size,
        indices,
        indptr,
        ordering.permutation,
        counts,
        boundaries,
        heights,
    )
    routes = np.stack(
        (
            counts.source_positions,
            analysis.elimination[counts.source_rows],
            analysis.elimination[counts.source_columns],
        )
    )
    return analysis, routes, counts.source_conjugate


def _elimination_schedule(
    storage: SparseStorage,
    input_indices: np.ndarray,
    input_indptr: np.ndarray,
    ordering: PreparedSparseOrdering,
    kind: Literal["lu", "cholesky"],
    policy: SparseFactorizationPolicy,
    tracker: _SymbolicResourceTracker,
    /,
) -> tuple[SparseEliminationSchedule, np.ndarray, int, tuple[bytes, ...]]:
    """Pivot-by-pivot symbolic fill of incomplete and static-pivot LU plans.

    Returns the schedule, the pivot positions, the pattern operation count
    (exact for complete factors, an upper bound under dropping) and the
    pattern bytes that enter the plan identity.
    """
    size = storage.shape[0]
    for row in range(size):
        tracker.add_work()
        tracker.add_factor_entry(row, row)
    entries = _permuted_entries(
        input_indices, input_indptr, ordering.permutation, tracker
    )
    rows = (
        _lu_symbolic_rows(size, entries, policy.fill_level, tracker)
        if kind == "lu"
        else _cholesky_symbolic_rows(size, entries, policy.fill_level, tracker)
    )
    factor_indices, factor_indptr, positions = _csr_from_rows(rows, tracker)
    input_positions = np.full(factor_indices.size, -1, dtype=np.int64)
    input_conjugate = np.zeros(factor_indices.size, dtype=np.bool_)
    for coordinate, factor_position in positions.items():
        tracker.add_work()
        route = entries.get(coordinate)
        if route is not None:
            input_positions[factor_position] = route[0]
            input_conjugate[factor_position] = route[1]
    diagonal = np.asarray([positions[(row, row)] for row in range(size)], dtype=np.int64)
    (
        column_positions,
        column_offsets,
        row_width,
        column_width,
        upper_width,
    ) = _column_index(factor_indices, factor_indptr, diagonal, tracker)
    lower_analysis, lower_positions = _triangular_pattern(
        rows,
        positions,
        "lower",
        storage.indices.dtype,
        tracker,
        unit_diagonal=kind == "lu",
    )
    if kind == "lu":
        upper_analysis, upper_positions = _triangular_pattern(
            rows,
            positions,
            "upper",
            storage.indices.dtype,
            tracker,
            unit_diagonal=False,
        )
    else:
        upper_analysis = None
        upper_positions = None
    below = np.diff(column_offsets)
    # One division per multiplier and one multiply-add per pattern update:
    # the lower triangle of an outer product for Cholesky, the full one for LU.
    updates = (
        below * (below + 1) // 2
        if kind == "cholesky"
        else below * (factor_indptr[1:] - diagonal - 1)
    )
    flops = int(np.sum(below + 2 * updates))
    index_dtype = storage.indices.dtype
    schedule = SparseEliminationSchedule(
        factor_indices=jnp.asarray(factor_indices, dtype=index_dtype),
        factor_indptr=jnp.asarray(factor_indptr, dtype=index_dtype),
        factor_rows=jnp.asarray(
            np.repeat(np.arange(size, dtype=np.int64), np.diff(factor_indptr)),
            dtype=index_dtype,
        ),
        input_positions=jnp.asarray(input_positions, dtype=index_dtype),
        input_conjugate=jnp.asarray(input_conjugate),
        column_positions=jnp.asarray(column_positions, dtype=index_dtype),
        column_offsets=jnp.asarray(column_offsets, dtype=index_dtype),
        lower_positions=jnp.asarray(lower_positions, dtype=index_dtype),
        upper_positions=(
            None
            if upper_positions is None
            else jnp.asarray(upper_positions, dtype=index_dtype)
        ),
        lower_analysis=lower_analysis,
        upper_analysis=upper_analysis,
        row_width=row_width,
        column_width=column_width,
        upper_width=upper_width,
    )
    return (
        schedule,
        diagonal,
        flops,
        (factor_indices.tobytes(), factor_indptr.tobytes()),
    )


def _window(start: Array, stop: Array, width: int, /) -> tuple[Array, Array]:
    """Return ``width`` consecutive positions from ``start`` masked below ``stop``.

    Masked positions are replaced by zero so gathers stay in bounds; scatters
    route them out of bounds and drop them.
    """
    positions = start + jnp.arange(width, dtype=start.dtype)
    valid = positions < stop
    return jnp.where(valid, positions, 0), valid


def _prune_row(
    values: Array,
    plan: SparseFactorizationPlan,
    schedule: SparseEliminationSchedule,
    row: Array,
    /,
) -> tuple[Array, Array]:
    safe_positions, valid = _window(
        schedule.factor_indptr[row], schedule.factor_indptr[row + 1], schedule.row_width
    )
    row_values = values[safe_positions]
    diagonal = safe_positions == plan.diagonal_positions[row]
    row_scale = jnp.max(jnp.where(valid, jnp.abs(row_values), 0.0))
    threshold_keep = jnp.abs(row_values) >= (plan.policy.drop_tolerance * row_scale)
    candidate = valid & ~diagonal & threshold_keep
    if plan.policy.maximum_fill_per_row is None:
        selected = candidate
    elif plan.policy.maximum_fill_per_row == 0:
        selected = jnp.zeros_like(candidate)
    else:
        count = min(plan.policy.maximum_fill_per_row, schedule.row_width)
        scores = jnp.where(candidate, jnp.abs(row_values), -jnp.inf)
        _, selected_indices = jax.lax.top_k(scores, count)
        selected = (
            jnp.any(
                jnp.arange(scores.size)[:, None] == selected_indices[None, :],
                axis=1,
            )
            & candidate
        )
    removed = valid & ~(diagonal | selected)
    pruned = values.at[jnp.where(removed, safe_positions, values.size)].set(
        jnp.zeros((), values.dtype), mode="drop"
    )
    return pruned, jnp.sum(removed, dtype=jnp.int32)


def _eliminate_pivot(
    values: Array,
    marker: Array,
    plan: SparseFactorizationPlan,
    schedule: SparseEliminationSchedule,
    pivot: Array,
    denominator: Array,
    /,
) -> tuple[Array, Array]:
    """Divide pivot column ``k`` and apply its rank-one update on the pattern.

    Targets come from the stored factor rows at runtime: ``marker`` maps every
    column of the pivot's update row (``U[k, j > k]`` for LU, the conjugate
    column ``L[j > k, k]`` for Cholesky) to its window slot. Each target row's
    entries right of its pivot-column entry look up the marker, so an update
    is applied exactly where the symbolic pattern holds ``(i, j)``. Targets are
    unique within one pivot, which keeps the scatter deterministic.
    """
    size = values.size
    below, below_valid = _window(
        schedule.column_offsets[pivot],
        schedule.column_offsets[pivot + 1],
        schedule.column_width,
    )
    below = schedule.column_positions[below]
    multipliers = values[below] / denominator
    values = values.at[jnp.where(below_valid, below, size)].set(multipliers, mode="drop")
    if plan.kind == "lu" and schedule.upper_width == 0:
        return values, marker
    if plan.kind == "lu":
        right, right_valid = _window(
            plan.diagonal_positions[pivot] + 1,
            schedule.factor_indptr[pivot + 1],
            schedule.upper_width,
        )
        keys = schedule.factor_indices[right]
        right_values = values[right]
    else:
        right_valid = below_valid
        keys = schedule.factor_rows[below]
        right_values = jnp.conj(multipliers)
    marked = jnp.where(right_valid, keys, marker.size)
    marker = marker.at[marked].set(jnp.arange(keys.size, dtype=marker.dtype), mode="drop")
    rows = schedule.factor_rows[below]
    tail = below[:, None] + 1 + jnp.arange(schedule.row_width, dtype=below.dtype)[None, :]
    tail_valid = below_valid[:, None] & (tail < schedule.factor_indptr[rows + 1][:, None])
    tail = jnp.where(tail_valid, tail, 0)
    slot = marker[schedule.factor_indices[tail]]
    hit = tail_valid & (slot >= 0)
    updates = multipliers[:, None] * right_values[jnp.where(hit, slot, 0)]
    values = values.at[jnp.where(hit, tail, size)].add(-updates, mode="drop")
    return values, marker.at[marked].set(-1, mode="drop")


def _refresh_workspace_bytes(plan: SparseFactorizationPlan, itemsize: int, /) -> int:
    """Transient bytes of one numeric refresh besides the factor values.

    A supernodal refresh holds the largest dense front, the multifrontal
    stack of pending update matrices at its peak, and the gathered input
    pairs. A pivot-by-pivot refresh holds the dense column marker and a
    ``column_width x row_width`` target window (positions, columns, slots,
    update values and masks).
    """
    if plan.supernodal is not None:
        analysis = plan.supernodal
        return (
            analysis.front_entries + analysis.stack_entries + analysis.input_sources.size
        ) * itemsize
    schedule = plan.elimination
    if schedule is None:
        raise ValueError("Sparse factorization plan has no elimination schedule.")
    index = schedule.factor_indices.dtype.itemsize
    window = schedule.column_width * schedule.row_width * (3 * index + itemsize + 2)
    return plan.shape[0] * index + window


def refresh_sparse_factorization(
    plan: SparseFactorizationPlan,
    operator: AbstractSparseLinearOperator,
    /,
) -> PreparedSparseFactorization:
    """Refresh independent numeric factors under one shared CSR pattern."""
    if not isinstance(plan, SparseFactorizationPlan):
        raise TypeError("plan must be a SparseFactorizationPlan.")
    from ..sparse._linear import SparseCoordinateOperator, SparseLinearMap

    routed = isinstance(operator, (SparseCoordinateOperator, SparseLinearMap))
    traced_routes = routed and any(
        isinstance(leaf, jax_core.Tracer)
        for leaf in jax.tree_util.tree_leaves(operator.relation)
    )
    if (
        isinstance(operator, (SparseCoordinateOperator, SparseLinearMap))
        and traced_routes
        and plan.storage_plan is not None
    ):
        storage = plan.storage_plan.apply(
            operator.coefficients, relation=operator.relation
        )
    else:
        storage, indices, indptr = _validated_pattern(operator)
        if _pattern_identifier(storage.shape, indices, indptr) != plan.input_pattern_id:
            raise ValueError(
                "Sparse factorization refresh requires an unchanged CSR pattern."
            )
    if storage.batch_shape != plan.batch_shape:
        raise ValueError(
            "Sparse factorization refresh requires an unchanged value batch shape."
        )
    return refresh_sparse_factorization_values(plan, storage.values)


def refresh_sparse_factorization_values(
    plan: SparseFactorizationPlan,
    values: ArrayLike,
    /,
) -> PreparedSparseFactorization:
    """Refresh traced numeric values in the plan's original CSR entry order.

    Symbolic analysis and pattern validation belong to
    :func:`prepare_sparse_factorization`. This numeric-only boundary accepts
    exactly ``plan.batch_shape + (plan.input_nnz,)`` values without inspecting
    traced CSR indices on the host.
    """
    if not isinstance(plan, SparseFactorizationPlan):
        raise TypeError("plan must be a SparseFactorizationPlan.")
    numeric_values = jnp.asarray(values)
    if numeric_values.shape != plan.batch_shape + (plan.input_nnz,):
        raise ValueError("Sparse factorization values must match the plan's input shape.")
    if not jnp.issubdtype(numeric_values.dtype, jnp.inexact):
        raise TypeError("Sparse factorization values must have an inexact dtype.")
    if numeric_values.dtype.name != plan.value_dtype:
        raise ValueError(
            "Sparse factorization refresh requires an unchanged value dtype "
            f"({plan.value_dtype}); got {numeric_values.dtype.name}."
        )

    if plan.supernodal is not None:
        return _refresh_supernodal(plan, plan.supernodal, numeric_values)
    schedule = plan.elimination
    if schedule is None:
        raise ValueError("Sparse factorization plan has no elimination schedule.")

    def factor_one(
        input_values: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        safe_input = jnp.maximum(schedule.input_positions, 0)
        gathered = input_values[safe_input]
        gathered = jnp.where(schedule.input_conjugate, jnp.conj(gathered), gathered)
        values = jnp.where(
            schedule.input_positions >= 0,
            gathered,
            jnp.zeros((), dtype=input_values.dtype),
        )
        values = values.at[plan.diagonal_positions].add(plan.policy.diagonal_shift)
        initial = (
            values,
            jnp.full((plan.shape[0],), -1, dtype=schedule.factor_indices.dtype),
            jnp.asarray(int(SparseFactorizationStatus.SUCCESS), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=values.real.dtype),
        )

        def factor_step(
            pivot_index: Array,
            carry: tuple[Array, Array, Array, Array, Array, Array, Array],
        ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
            current, marker, status, replaced, rejected, dropped, minimum_pivot = carry
            if (
                plan.policy.drop_tolerance > 0.0
                or plan.policy.maximum_fill_per_row is not None
            ):
                current, row_dropped = _prune_row(current, plan, schedule, pivot_index)
                dropped = dropped + row_dropped
            pivot_position = plan.diagonal_positions[pivot_index]
            pivot = current[pivot_position]
            finite = jnp.isfinite(pivot)
            if plan.kind == "cholesky":
                real_pivot = jnp.real(pivot)
                real_dtype = real_pivot.dtype
                hermitian_roundoff = (
                    64.0
                    * jnp.finfo(real_dtype).eps
                    * jnp.maximum(
                        jnp.ones((), dtype=real_dtype),
                        jnp.abs(real_pivot),
                    )
                )
                imaginary_tolerance = jnp.maximum(
                    jnp.asarray(plan.policy.pivot_tolerance, dtype=real_dtype),
                    hermitian_roundoff,
                )
                acceptable = (
                    finite
                    & (real_pivot > plan.policy.pivot_tolerance)
                    & (jnp.abs(jnp.imag(pivot)) <= imaginary_tolerance)
                )
                failure_status = int(SparseFactorizationStatus.NONPOSITIVE_PIVOT)
                replacement = jnp.asarray(
                    max(
                        plan.policy.replacement_value,
                        plan.policy.pivot_tolerance,
                    ),
                    dtype=current.dtype,
                )
            else:
                acceptable = finite & (jnp.abs(pivot) > plan.policy.pivot_tolerance)
                failure_status = int(SparseFactorizationStatus.ZERO_PIVOT)
                phase = jnp.where(
                    jnp.abs(pivot) > 0.0,
                    pivot / jnp.abs(pivot),
                    jnp.ones((), dtype=current.dtype),
                )
                replacement = phase * jnp.asarray(
                    max(
                        plan.policy.replacement_value,
                        plan.policy.pivot_tolerance,
                    ),
                    dtype=current.dtype,
                )
            bad = ~acceptable
            rejected = rejected + bad.astype(jnp.int32)
            status = jnp.where(
                ~finite,
                int(SparseFactorizationStatus.NONFINITE),
                jnp.where(
                    (status == int(SparseFactorizationStatus.SUCCESS)) & bad,
                    failure_status,
                    status,
                ),
            ).astype(jnp.int32)
            use_replacement = bad & plan.policy.allow_pivot_replacement
            effective_pivot = jnp.where(
                acceptable,
                pivot,
                jnp.where(
                    use_replacement,
                    replacement,
                    jnp.ones((), dtype=current.dtype),
                ),
            )
            replaced = replaced + use_replacement.astype(jnp.int32)
            status = jnp.where(
                use_replacement & (status == failure_status),
                int(SparseFactorizationStatus.SUCCESS),
                status,
            ).astype(jnp.int32)
            minimum_pivot = jnp.minimum(minimum_pivot, jnp.abs(effective_pivot))
            if plan.kind == "cholesky":
                factor_pivot = jnp.sqrt(jnp.real(effective_pivot)).astype(current.dtype)
                current = current.at[pivot_position].set(factor_pivot)
                denominator = factor_pivot
            else:
                current = current.at[pivot_position].set(effective_pivot)
                denominator = effective_pivot
            current, marker = _eliminate_pivot(
                current, marker, plan, schedule, pivot_index, denominator
            )
            return current, marker, status, replaced, rejected, dropped, minimum_pivot

        values, _, status, replaced, rejected, dropped, minimum_pivot = jax.lax.fori_loop(
            0, plan.shape[0], factor_step, initial
        )
        finite = jnp.all(jnp.isfinite(values))
        status = jnp.where(
            ~finite,
            int(SparseFactorizationStatus.NONFINITE),
            status,
        ).astype(jnp.int32)
        factor_nonzeros = jnp.count_nonzero(values).astype(jnp.int32)
        return (
            values,
            status,
            minimum_pivot,
            replaced,
            rejected,
            dropped,
            factor_nonzeros,
            finite,
        )

    batch_count = int(np.prod(plan.batch_shape)) if plan.batch_shape else 1
    flattened_input = numeric_values.reshape((batch_count, plan.input_nnz))
    (
        values,
        status,
        minimum_pivot,
        replaced,
        rejected,
        dropped,
        factor_nonzeros,
        finite,
    ) = jax.vmap(factor_one)(flattened_input)
    return _prepared(
        plan,
        values.reshape(plan.batch_shape + (plan.factor_size,)),
        status.reshape(plan.batch_shape),
        minimum_pivot.reshape(plan.batch_shape),
        replaced.reshape(plan.batch_shape),
        rejected.reshape(plan.batch_shape),
        dropped.reshape(plan.batch_shape),
        factor_nonzeros.reshape(plan.batch_shape),
        finite.reshape(plan.batch_shape),
    )


def _prepared(
    plan: SparseFactorizationPlan,
    values: Array,
    status: Array,
    minimum_pivot: Array,
    replaced: Array,
    rejected: Array,
    dropped: Array,
    factor_nonzeros: Array,
    finite: Array,
    /,
    *,
    input_values: Array | None = None,
) -> PreparedSparseFactorization:
    diagnostics = SparseFactorizationDiagnostics(
        minimum_pivot=minimum_pivot,
        replaced_pivots=replaced,
        rejected_pivots=rejected,
        dropped_entries=dropped,
        input_nonzeros=jnp.full(plan.batch_shape, plan.input_nnz, dtype=jnp.int32),
        factor_nonzeros=factor_nonzeros,
        fill_ratio=factor_nonzeros / max(plan.input_nnz, 1),
        finite=finite,
    )
    return PreparedSparseFactorization(
        plan=plan,
        factor_values=values,
        status=status,
        diagnostics=diagnostics,
        factorization_id=f"{plan.plan_id}/numeric",
        input_values=input_values,
    )


@dataclass(frozen=True)
class _SupernodalFactorCallback:
    """Host multifrontal refresh over any leading batch axes of the values.

    Returns panels, status, minimum pivot, replaced and rejected pivot counts,
    realized nonzeros, finiteness, the minimum pivot's panel position and the
    per-column pivot kinds that `_panel_tangent_p` reads to differentiate the
    panels without refactoring.
    """

    analysis: SupernodalAnalysis
    diagonal_shift: float
    pivot_tolerance: float
    allow_replacement: bool
    replacement_value: float

    def result_shapes(self, values: Array, /) -> tuple[jax.ShapeDtypeStruct, ...]:
        batch = values.shape[:-1]
        return (
            jax.ShapeDtypeStruct(batch + (self.analysis.factor_size,), values.dtype),
            jax.ShapeDtypeStruct(batch, jnp.int32),
            jax.ShapeDtypeStruct(batch, jnp.finfo(values.dtype).dtype),
            jax.ShapeDtypeStruct(batch, jnp.int32),
            jax.ShapeDtypeStruct(batch, jnp.int32),
            jax.ShapeDtypeStruct(batch, jnp.int32),
            jax.ShapeDtypeStruct(batch, jnp.bool_),
            jax.ShapeDtypeStruct(batch, jnp.int32),
            jax.ShapeDtypeStruct(batch + (self.analysis.size,), jnp.int8),
        )

    def __call__(self, values: np.ndarray, /) -> tuple[np.ndarray, ...]:
        values = np.asarray(values)
        batch = values.shape[:-1]
        panels = np.empty(batch + (self.analysis.factor_size,), dtype=values.dtype)
        status = np.empty(batch, dtype=np.int32)
        minimum_pivot = np.empty(batch, dtype=np.finfo(values.dtype).dtype)
        replaced = np.empty(batch, dtype=np.int32)
        rejected = np.empty(batch, dtype=np.int32)
        nonzeros = np.empty(batch, dtype=np.int32)
        finite = np.empty(batch, dtype=bool)
        position = np.empty(batch, dtype=np.int32)
        kinds = np.empty(batch + (self.analysis.size,), dtype=np.int8)
        for index in np.ndindex(*batch):
            result = _supernodal.factor_values(
                self.analysis,
                values[index],
                self.diagonal_shift,
                self.pivot_tolerance,
                self.allow_replacement,
                self.replacement_value,
            )
            panels[index] = result.panels
            status[index] = result.status
            minimum_pivot[index] = result.minimum_pivot
            replaced[index] = result.replaced
            rejected[index] = result.rejected
            nonzeros[index] = np.count_nonzero(result.panels)
            finite[index] = result.status != _supernodal.STATUS_NONFINITE
            position[index] = result.minimum_position
            kinds[index] = result.pivot_kinds
        return (
            panels,
            status,
            minimum_pivot,
            replaced,
            rejected,
            nonzeros,
            finite,
            position,
            kinds,
        )


class SparseFactorDerivativeError(LinearCapabilityError):
    """A reverse-mode cotangent reached supernodal factor panels.

    Supernodal solves differentiate at every order in forward and reverse mode
    through the retained input values. The factor panels and minimum pivot
    carry exact forward-mode derivatives at every order, but their tangents
    have no transpose, so reverse mode through them is refused at trace time.
    """


@dataclass(frozen=True)
class _SupernodalPanelCallback:
    """Host panel tangent or symmetric panel product over broadcast batch axes."""

    analysis: SupernodalAnalysis
    product: bool

    def __call__(self, *operands: np.ndarray) -> np.ndarray:
        arrays = tuple(np.asarray(operand) for operand in operands)
        batch = np.broadcast_shapes(*(array.shape[:-1] for array in arrays))
        floating = (arrays[0], arrays[-1])
        dtype = np.result_type(*floating)
        arrays = tuple(
            np.broadcast_to(array, batch + array.shape[-1:]) for array in arrays
        )
        kernel = (
            _supernodal.symmetric_product if self.product else _supernodal.tangent_values
        )
        result = np.empty(batch + (self.analysis.factor_size,), dtype=dtype)
        for index in np.ndindex(*batch):
            result[index] = kernel(self.analysis, *(array[index] for array in arrays))
        return result


def _panel_abstract(*operands: ShapedArray, analysis: SupernodalAnalysis) -> ShapedArray:
    batch = np.broadcast_shapes(*(operand.shape[:-1] for operand in operands))
    dtype = jnp.result_type(operands[0].dtype, operands[-1].dtype)
    return ShapedArray(batch + (analysis.factor_size,), dtype)


def _panel_host(*operands: Array, analysis: SupernodalAnalysis, product: bool) -> Array:
    result = _panel_abstract(*map(jax.typeof, operands), analysis=analysis)
    return jax.pure_callback(
        _SupernodalPanelCallback(analysis, product),
        jax.ShapeDtypeStruct(result.shape, result.dtype),
        *operands,
        vmap_method="broadcast_all",
    )


def _panel_batch(
    primitive: jex_core.Primitive,
    operands: tuple[Array, ...],
    axes: tuple[int | None, ...],
    *,
    analysis: SupernodalAnalysis,
) -> tuple[Array, int]:
    """Lead with the mapped axis; unmapped operands broadcast against it."""
    rank = max(operand.ndim - (axis is not None) for operand, axis in zip(operands, axes))
    aligned = []
    for operand, axis in zip(operands, axes):
        leading = operand[None] if axis is None else jnp.moveaxis(operand, axis, 0)
        padding = (1,) * (rank + 1 - leading.ndim)
        aligned.append(leading.reshape(leading.shape[:1] + padding + leading.shape[1:]))
    return primitive.bind(*aligned, analysis=analysis), 0


def _refuse_panel_transpose(
    cotangent: Array | ad.Zero,
    *operands: Array | ad.UndefinedPrimal,
    analysis: SupernodalAnalysis,
) -> tuple[ad.Zero | None, ...]:
    del analysis
    if not isinstance(cotangent, ad.Zero):
        raise SparseFactorDerivativeError(
            "Reverse-mode differentiation of supernodal factor panels or the "
            "minimum pivot is unsupported; differentiate the solve, whose "
            "derivatives are solves with the prepared factor, or use forward mode."
        )
    return tuple(
        ad.Zero(operand.aval.to_tangent_aval())
        if ad.is_undefined_primal(operand)
        else None
        for operand in operands
    )


# ``tril(X Yᴴ + Y Xᴴ)`` of two panel-stored lower factors (bilinear).
_panel_product_p = jex_core.Primitive("phydrax_supernodal_panel_product")
_panel_product_p.def_impl(partial(dispatch.apply_primitive, _panel_product_p))
_panel_product_p.def_abstract_eval(_panel_abstract)
mlir.register_lowering(
    _panel_product_p,
    mlir.lower_fun(partial(_panel_host, product=True), multiple_results=False),
)

# Panel tangent ``dL`` solving ``tril(dL Lᴴ + L dLᴴ) = seed`` with stored panels
# ``L`` and their pivot kinds (linear in the seed).
_panel_tangent_p = jex_core.Primitive("phydrax_supernodal_panel_tangent")
_panel_tangent_p.def_impl(partial(dispatch.apply_primitive, _panel_tangent_p))
_panel_tangent_p.def_abstract_eval(_panel_abstract)
mlir.register_lowering(
    _panel_tangent_p,
    mlir.lower_fun(partial(_panel_host, product=False), multiple_results=False),
)


def _panel_product_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Array | ad.Zero, Array | ad.Zero],
    *,
    analysis: SupernodalAnalysis,
) -> tuple[Array, Array | ad.Zero]:
    (left, right), (left_tangent, right_tangent) = primals, tangents
    result = _panel_product_p.bind(left, right, analysis=analysis)
    terms = []
    if not isinstance(left_tangent, ad.Zero):
        terms.append(_panel_product_p.bind(left_tangent, right, analysis=analysis))
    if not isinstance(right_tangent, ad.Zero):
        terms.append(_panel_product_p.bind(left, right_tangent, analysis=analysis))
    if not terms:
        return result, ad.Zero(jax.typeof(result))
    return result, terms[0] if len(terms) == 1 else terms[0] + terms[1]


def _panel_tangent_jvp(
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array | ad.Zero, Array | ad.Zero, Array | ad.Zero],
    *,
    analysis: SupernodalAnalysis,
) -> tuple[Array, Array | ad.Zero]:
    """Differentiate ``tril(X Lᴴ + L Xᴴ) = seed`` implicitly in ``L`` and seed.

    ``dX`` solves the same identity for ``d seed − tril(dL Xᴴ + X dLᴴ)``, so
    every order reuses the stored panels and this primitive.
    """
    (panels, kinds, seed), (panel_tangent, _, seed_tangent) = primals, tangents
    result = _panel_tangent_p.bind(panels, kinds, seed, analysis=analysis)
    right = seed_tangent
    if not isinstance(panel_tangent, ad.Zero):
        coupling = _panel_product_p.bind(panel_tangent, result, analysis=analysis)
        right = -coupling if isinstance(right, ad.Zero) else right - coupling
    if isinstance(right, ad.Zero):
        return result, ad.Zero(jax.typeof(result))
    return result, _panel_tangent_p.bind(panels, kinds, right, analysis=analysis)


ad.primitive_jvps[_panel_product_p] = _panel_product_jvp
ad.primitive_transposes[_panel_product_p] = _refuse_panel_transpose
batching.primitive_batchers[_panel_product_p] = partial(_panel_batch, _panel_product_p)
ad.primitive_jvps[_panel_tangent_p] = _panel_tangent_jvp
ad.primitive_transposes[_panel_tangent_p] = _refuse_panel_transpose
batching.primitive_batchers[_panel_tangent_p] = partial(_panel_batch, _panel_tangent_p)


def _panel_seed(analysis: SupernodalAnalysis, direction: Array, /) -> Array:
    """Scatter a value direction onto the panel pattern, as the refresh does."""
    gathered = direction[..., analysis.input_sources]
    if jnp.iscomplexobj(direction):
        gathered = jnp.where(analysis.input_conjugate, jnp.conj(gathered), gathered)
    seed = jnp.zeros(direction.shape[:-1] + (analysis.factor_size,), direction.dtype)
    return seed.at[..., analysis.input_targets].set(gathered)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _supernodal_refresh(
    callback: _SupernodalFactorCallback, values: Array, /
) -> tuple[Array, ...]:
    return jax.pure_callback(
        callback, callback.result_shapes(values), values, vmap_method="broadcast_all"
    )


@_supernodal_refresh.defjvp
def _supernodal_refresh_jvp(
    callback: _SupernodalFactorCallback,
    primals: tuple[Array],
    tangents: tuple[Array],
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    """Exact panel tangents from the refreshed panels, at every order.

    The panel tangent solves the linearized factor identity with the stored
    panels (`_panel_tangent_p`), so no tangent refactors the values and every
    higher derivative recurses through the same primitive. The minimum pivot
    ``|L_kk|²`` moves with ``2 Re(conj(L_kk) dL_kk)`` when it is an accepted
    pivot and is constant when replaced or continued. Integer and boolean
    evidence has no tangent.
    """
    (values,), (direction,) = primals, tangents
    primal = _supernodal_refresh(callback, values)
    panels, _, minimum_pivot, *_, position, kinds = primal
    analysis = callback.analysis
    panel_tangent = _panel_tangent_p.bind(
        panels, kinds, _panel_seed(analysis, direction), analysis=analysis
    )
    safe = jnp.maximum(position, 0)[..., None]
    pivot = jnp.take_along_axis(panels, safe, axis=-1)[..., 0]
    pivot_tangent = jnp.take_along_axis(panel_tangent, safe, axis=-1)[..., 0]
    minimum_tangent = jnp.where(
        position >= 0, 2.0 * jnp.real(jnp.conj(pivot) * pivot_tangent), 0.0
    ).astype(minimum_pivot.dtype)
    integer_tangents = tuple(
        np.zeros(output.shape, dtype=jax.dtypes.float0)
        for output in (
            primal[1],
            primal[3],
            primal[4],
            primal[5],
            primal[6],
            primal[7],
            primal[8],
        )
    )
    return primal, (
        panel_tangent,
        integer_tangents[0],
        minimum_tangent,
        *integer_tangents[1:],
    )


def _refresh_supernodal(
    plan: SparseFactorizationPlan, analysis: SupernodalAnalysis, values: Array, /
) -> PreparedSparseFactorization:
    """Multifrontal numeric refresh through one host callback per value set."""
    policy = plan.policy
    panels, status, minimum_pivot, replaced, rejected, nonzeros, finite, _, _ = (
        _supernodal_refresh(
            _SupernodalFactorCallback(
                analysis,
                policy.diagonal_shift,
                policy.pivot_tolerance,
                policy.allow_pivot_replacement,
                policy.replacement_value,
            ),
            values,
        )
    )
    return _prepared(
        plan,
        panels,
        status,
        minimum_pivot,
        replaced,
        rejected,
        jnp.zeros(plan.batch_shape, dtype=jnp.int32),
        nonzeros,
        finite,
        input_values=values,
    )


def factorize_sparse(
    operator: AbstractSparseLinearOperator,
    policy: SparseFactorizationPolicy | None = None,
    /,
) -> PreparedSparseFactorization:
    """Symbolically plan and numerically factor one sparse operator."""
    plan = prepare_sparse_factorization(operator, policy)
    return refresh_sparse_factorization(plan, operator)


__all__ = [
    "PreparedSparseFactorization",
    "SparseFactorDerivativeError",
    "SparseFactorizationDiagnostics",
    "SparseFactorizationKind",
    "SparseFactorizationPlan",
    "SparseFactorizationPolicy",
    "SparseFactorizationSolveResult",
    "SparseFactorizationStatus",
    "factorize_sparse",
    "prepare_sparse_factorization",
    "refresh_sparse_factorization",
    "refresh_sparse_factorization_values",
]
