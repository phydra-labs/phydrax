#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse native programs for bounded local coupled relocation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import auto, Enum
from operator import index
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from ..config import x64_enabled
from ..linalg import ArraySpace, OperatorProperties
from ..linalg._sparse_factorizations import _refresh_workspace_bytes
from ..optim import (
    ConicProgram,
    ConvexSolvePolicy,
    NativeHomogeneousConic,
    NonnegativeCone,
    ProductCone,
    ZeroCone,
)
from ..optim._programming._lifecycle import (
    bind_convex_numeric,
    ConvexProgramTemplate,
    prepare_convex_template,
    PreparedConvexProgram,
)
from ..optim._programming._native_conic import PreparedNativeConic
from ..sparse import EdgeRelation, SparseCoordinateOperator
from ..sparse._linear import _storage_index_dtype


class _RowRole(Enum):
    SOURCE = auto()
    CARRIER = auto()
    ORBIT = auto()
    METRIC_LOWER = auto()
    METRIC_UPPER = auto()
    NATIVE = auto()


@dataclass(frozen=True, eq=False, init=False)
class _CoupledRow:
    """Immutable host coefficients, including explicitly supported zeros."""

    columns: tuple[int, ...]
    values: np.ndarray
    role: _RowRole
    identity: tuple[int, ...]

    def __init__(
        self,
        columns: tuple[int, ...],
        values: np.ndarray,
        role: _RowRole,
        *,
        identity: tuple[int, ...],
    ) -> None:
        if not isinstance(columns, tuple):
            raise TypeError("Row columns must be a tuple of indices.")
        if any(isinstance(column, (bool, np.bool_)) for column in columns):
            raise TypeError("Row columns must be integer indices, not booleans.")
        columns = tuple(index(column) for column in columns)
        if any(column < 0 for column in columns) or any(
            left >= right for left, right in zip(columns, columns[1:])
        ):
            raise ValueError("Row columns must be nonnegative, sorted, and unique.")
        if not isinstance(values, np.ndarray):
            raise TypeError("Row values must be a NumPy array.")
        if values.dtype != np.dtype(np.float64):
            raise TypeError("Row values must have float64 dtype.")
        if values.shape != (len(columns),):
            raise ValueError("Row values must match the column vector.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Row values must be finite.")
        if not isinstance(role, _RowRole):
            raise TypeError("Row role must be a _RowRole.")
        if not isinstance(identity, tuple) or not identity:
            raise TypeError("A coupled row requires an explicit nonempty identity.")
        if any(isinstance(item, (bool, np.bool_)) for item in identity):
            raise TypeError("Row identity components must be integer identifiers.")
        identity = tuple(index(item) for item in identity)
        # An immutable bytes owner also prevents re-enabling NumPy's write flag.
        values = np.frombuffer(values.tobytes(), dtype=np.float64)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "identity", identity)

    def block(self, start: int) -> np.ndarray:
        if isinstance(start, bool):
            raise TypeError("Block start must be an integer.")
        first = index(start)
        if first < 0:
            raise ValueError("Block start must be nonnegative.")
        result = np.zeros(3, dtype=np.float64)
        for column, value in zip(self.columns, self.values, strict=True):
            if first <= column < first + 3:
                result[column - first] = value
        return result

    def negated(self, role: _RowRole) -> _CoupledRow:
        return _CoupledRow(self.columns, -self.values, role, identity=self.identity)


class _ProgramDemand(NamedTuple):
    variables: int
    equalities: int
    inequalities: int
    dimension: int
    input_entries: int
    input_bytes: int
    factor_nnz: int | None
    factor_bytes: int | None
    workspace_bytes: int | None
    symbolic_work: int | None


class _PreparedCoupledProgram(NamedTuple):
    program: ConicProgram
    prepared: PreparedConvexProgram
    demand: _ProgramDemand


class _ProgramCache:
    """Two structure-and-policy keyed templates owned by one search."""

    def __init__(self) -> None:
        self.primary: ConvexProgramTemplate | None = None
        self.certificate: ConvexProgramTemplate | None = None


def _demand(n: int, e: int, m: int, entries: int, *, quadratic: bool) -> _ProgramDemand:
    # Canonical accounting uses CSR values, indices and row pointers, linear,
    # rhs, and both materialized bound vectors. The diagonal Q has n entries.
    constraints = e + m
    input_entries = 3 * n + 2 * constraints + 2 * entries + 1
    input_bytes = (3 * n + entries + constraints) * np.dtype(np.float64).itemsize
    input_bytes += (entries + constraints + 1) * _storage_index_dtype(
        (constraints, n), entries
    ).itemsize
    if quadratic:
        input_entries += 3 * n + 1
        input_bytes += n * np.dtype(np.float64).itemsize
        input_bytes += (2 * n + 1) * _storage_index_dtype((n, n), n).itemsize
    return _ProgramDemand(
        n, e, m, n + e, input_entries, input_bytes, None, None, None, None
    )


def primary_demand(
    rows: list[_CoupledRow], equalities: list[_CoupledRow], variables: int
) -> _ProgramDemand:
    if isinstance(variables, bool):
        raise TypeError("variables must be an integer.")
    n = index(variables)
    if n < 1:
        raise ValueError("A coupled program needs at least one variable.")
    for row in (*equalities, *rows):
        if not isinstance(row, _CoupledRow):
            raise TypeError("Coupled constraints must be _CoupledRow values.")
        if row.columns and row.columns[-1] >= n:
            raise ValueError("A row column exceeds the variable space.")
    return _demand(
        n,
        len(equalities),
        len(rows),
        sum(len(row.columns) for row in (*equalities, *rows)),
        quadratic=True,
    )


def admit_inputs(demand: _ProgramDemand, policy: ConvexSolvePolicy) -> None:
    """Admit either native program before allocating its sparse input arrays."""
    materialization = policy.materialization
    if demand.input_entries > materialization.max_entries:
        raise ValueError(
            f"Canonical program stores {demand.input_entries} entries, exceeding the "
            f"materialization limit {materialization.max_entries}."
        )
    if demand.input_bytes > materialization.max_bytes:
        raise ValueError(
            f"Canonical program stores {demand.input_bytes} bytes, exceeding the "
            f"materialization limit {materialization.max_bytes}."
        )


def certificate_demand(primary: _PreparedCoupledProgram) -> _ProgramDemand:
    matrix = primary.program.constraint_matrix
    if not isinstance(matrix, SparseCoordinateOperator) or not isinstance(
        matrix.relation, EdgeRelation
    ):
        raise TypeError("The primary constraint matrix must have sparse coordinates.")
    n, m = primary.program.num_variables, primary.program.num_constraints
    return _demand(m, n, 2 * m, matrix.relation.capacity + 2 * m, quadratic=False)


def _operator(
    columns: np.ndarray,
    rows: np.ndarray,
    values: np.ndarray,
    source: ArraySpace,
    target: ArraySpace,
    *,
    properties: OperatorProperties | None = None,
) -> SparseCoordinateOperator:
    return SparseCoordinateOperator(
        EdgeRelation(
            jnp.asarray(columns, dtype=jnp.int32),
            jnp.asarray(rows, dtype=jnp.int32),
            source_size=source.size,
            target_size=target.size,
        ),
        jnp.asarray(values, dtype=jnp.float64),
        source=source,
        target=target,
        properties=properties,
    )


def _prepare(
    program: ConicProgram,
    demand: _ProgramDemand,
    policy: ConvexSolvePolicy,
    cache: _ProgramCache,
    *,
    certificate: bool,
) -> _PreparedCoupledProgram:
    method = policy.method
    if not isinstance(method, NativeHomogeneousConic) or method.newton != "factorized":
        raise TypeError("Coupled programs require factorized NativeHomogeneousConic.")
    template = cache.certificate if certificate else cache.primary
    if (
        template is None
        or template.plan.problem_signature != program.structure_id
        or template.plan.policy.policy_id != policy.policy_id
    ):
        template = prepare_convex_template(program, policy)
        if certificate:
            cache.certificate = template
        else:
            cache.primary = template
    symbolic = template.symbolic_state
    if not isinstance(symbolic, PreparedNativeConic) or symbolic.newton is None:
        raise RuntimeError("Coupled program has no prepared factorized native plan.")
    factor = symbolic.newton.factorization
    workspace = _refresh_workspace_bytes(factor, np.dtype(np.float64).itemsize)
    if factor.factor_bytes > policy.resources.factorization_bytes:
        raise ValueError("Reduced Newton factor exceeds the factorization byte budget.")
    if workspace > min(
        policy.resources.workspace_bytes, policy.materialization.max_workspace_bytes
    ):
        raise ValueError(
            "Reduced Newton factor refresh exceeds the workspace byte budget."
        )
    admitted = demand._replace(
        factor_nnz=factor.factor_nnz,
        factor_bytes=factor.factor_bytes,
        workspace_bytes=workspace,
        symbolic_work=factor.symbolic_work,
    )
    return _PreparedCoupledProgram(
        program, bind_convex_numeric(template, program), admitted
    )


def prepare_primary(
    rows: list[_CoupledRow],
    rhs: list[float],
    equalities: list[_CoupledRow],
    variables: int,
    identity: str,
    policy: ConvexSolvePolicy,
    cache: _ProgramCache,
) -> _PreparedCoupledProgram:
    demand = primary_demand(rows, equalities, variables)
    admit_inputs(demand, policy)
    if not isinstance(identity, str) or not identity:
        raise ValueError("Coupled program identity must be a nonempty string.")
    right = np.asarray(rhs, dtype=np.float64)
    if right.shape != (len(rows),) or not np.all(np.isfinite(right)):
        raise ValueError("Coupled inequality rhs must be a finite vector matching rows.")
    if not x64_enabled():
        raise ValueError("Coupled programs require binary64 JAX inputs.")
    all_rows = (*equalities, *rows)
    structure = canonical_fingerprint(
        {
            "kind": "coupled-primary-structure",
            "namespace": identity,
            "dtype": np.dtype(np.float64).str,
            "variables": demand.variables,
            "equalities": demand.equalities,
            "inequalities": demand.inequalities,
            "structural_entries": sum(len(row.columns) for row in all_rows),
            "rows": [(row.role.name, row.identity, row.columns) for row in all_rows],
        }
    )
    n, count = demand.variables, len(all_rows)
    source = ArraySpace((n,), dtype=jnp.float64, space_id=f"{identity}:displacements")
    target = ArraySpace(
        (count,), dtype=jnp.float64, space_id=f"{structure}:constraint-rows"
    )
    columns = np.asarray(
        [column for row in all_rows for column in row.columns], dtype=np.int32
    )
    row_indices = np.repeat(
        np.arange(count, dtype=np.int32), [len(row.columns) for row in all_rows]
    )
    values = (
        np.concatenate([row.values for row in all_rows])
        if all_rows
        else np.empty(0, dtype=np.float64)
    )
    matrix = _operator(columns, row_indices, values, source, target)
    diagonal = np.arange(n, dtype=np.int32)
    quadratic = _operator(
        diagonal,
        diagonal,
        np.ones(n, dtype=np.float64),
        source,
        source,
        properties=OperatorProperties(
            diagonal=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={"positive_definite": "construction"},
        ),
    )
    program = ConicProgram(
        quadratic,
        jnp.zeros(n, dtype=jnp.float64),
        matrix,
        jnp.asarray(np.concatenate((np.zeros(len(equalities), dtype=np.float64), right))),
        ProductCone((ZeroCone(len(equalities)), NonnegativeCone(len(rows)))),
        problem_id=structure,
        convexity_evidence="Identity quadratic is positive definite by construction.",
    )
    return _prepare(program, demand, policy, cache, certificate=False)


def prepare_certificate(
    primary: _PreparedCoupledProgram,
    policy: ConvexSolvePolicy,
    cache: _ProgramCache,
) -> _PreparedCoupledProgram:
    if primary.prepared.plan.policy.policy_id != policy.policy_id:
        raise ValueError("Certificate policy must match the primary program policy.")
    demand = certificate_demand(primary)
    admit_inputs(demand, policy)
    original = primary.program
    matrix = original.constraint_matrix
    if not isinstance(matrix, SparseCoordinateOperator) or not isinstance(
        matrix.relation, EdgeRelation
    ):
        raise TypeError("The primary constraint matrix must have sparse coordinates.")
    n, m = original.num_variables, original.num_constraints
    if m == 0:
        raise ValueError("A certificate requires at least one original constraint.")
    relation = matrix.relation
    original_rows = np.asarray(relation.target_indices, dtype=np.int32)
    original_columns = np.asarray(relation.source_indices, dtype=np.int32)
    original_values = np.asarray(matrix.coefficients, dtype=np.float64)
    diagonal = np.arange(m, dtype=np.int32)
    rows = np.concatenate((original_columns, n + diagonal, n + m + diagonal))
    columns = np.concatenate((original_rows, diagonal, diagonal))
    values = np.concatenate(
        (original_values, -np.ones(m, dtype=np.float64), np.ones(m, dtype=np.float64))
    )
    order = np.lexsort((columns, rows))
    identity = f"{original.structure_id}:bounded-farkas"
    source = ArraySpace((m,), dtype=jnp.float64, space_id=f"{identity}:dual-variables")
    target = ArraySpace(
        (n + 2 * m,),
        dtype=jnp.float64,
        space_id=f"{identity}:stationarity-lower-upper-rows",
    )
    operator = _operator(columns[order], rows[order], values[order], source, target)
    lower_rhs = np.zeros(m, dtype=np.float64)
    lower_rhs[: primary.demand.equalities] = 1.0
    rhs = np.concatenate(
        (np.zeros(n, dtype=np.float64), lower_rhs, np.ones(m, dtype=np.float64))
    )
    program = ConicProgram(
        None,
        original.constraint_rhs,
        operator,
        jnp.asarray(rhs),
        ProductCone((ZeroCone(n), NonnegativeCone(m), NonnegativeCone(m))),
        problem_id=identity,
        convexity_evidence="Linear objective, original transpose equalities, and explicit dual box rows.",
    )
    return _prepare(program, demand, policy, cache, certificate=True)
