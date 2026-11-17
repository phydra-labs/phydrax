#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native programs for bounded local coupled relocation.

A program whose power-of-two capacity class the declared budgets admit
executes as bounded dense data on the reduced dense Newton route, so one
compiled solve serves every program of that class. Larger programs execute as
sparse programs with a per-structure factorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import auto, Enum
from operator import index
from typing import Final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from ..config import x64_enabled
from ..linalg import ArraySpace, OperatorProperties
from ..linalg._sparse_factorizations import _refresh_workspace_bytes
from ..optim import (
    ConicProgram,
    ConvexProgramResult,
    ConvexSolvePolicy,
    NativeHomogeneousConic,
    NonnegativeCone,
    ProductCone,
    solve_prepared_convex_program,
    ZeroCone,
)
from ..optim._programming._audit import audit_conic_dual_infeasibility_ray, DualRayAudit
from ..optim._programming._lifecycle import (
    bind_convex_numeric,
    ConvexProgramTemplate,
    prepare_convex_template,
    PreparedConvexProgram,
)
from ..optim._programming._native_conic import PreparedNativeConic
from ..optim._programming._types import ConvexProgramProvenance
from ..sparse import EdgeRelation, SparseCoordinateOperator
from ..sparse._linear import _storage_index_dtype


# Smallest capacity class, so tiny programs do not fragment into many classes.
_MINIMUM_CAPACITY: Final = 8


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


class _CapacityLayout(NamedTuple):
    """Original sizes of one program inside its power-of-two capacity class."""

    variables: int
    blocks: tuple[int, ...]
    zero_blocks: tuple[bool, ...]
    variable_capacity: int
    block_capacities: tuple[int, ...]

    @property
    def rows(self) -> np.ndarray:
        """Execution row of every original constraint row, in original order."""
        offsets = np.concatenate(([0], np.cumsum(self.block_capacities[:-1])))
        return np.concatenate(
            [
                offset + np.arange(size, dtype=np.int64)
                for offset, size in zip(offsets, self.blocks, strict=True)
            ]
        ).astype(np.int64)


class _PreparedCoupledProgram(NamedTuple):
    """Original-coordinate program with its bound execution program.

    ``layout`` is present when ``prepared`` binds the dense capacity-class
    program rather than ``program`` itself; ``policy`` is the declared policy.
    """

    program: ConicProgram
    prepared: PreparedConvexProgram
    demand: _ProgramDemand
    layout: _CapacityLayout | None
    policy: ConvexSolvePolicy


class _ProgramCache:
    """Structure-and-policy keyed templates owned by one search."""

    def __init__(self) -> None:
        self.primary: ConvexProgramTemplate | None = None
        self.certificate: ConvexProgramTemplate | None = None
        self.capacity: dict[tuple[str, str], ConvexProgramTemplate] = {}
        self.execution_policies: dict[str, ConvexSolvePolicy] = {}


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
        program, bind_convex_numeric(template, program), admitted, None, policy
    )


def _capacity(count: int, /) -> int:
    """Power-of-two capacity class that holds ``count`` entries."""
    return max(_MINIMUM_CAPACITY, 1 << max(count - 1, 0).bit_length())


def _layout(
    variables: int, blocks: tuple[int, ...], zero_blocks: tuple[bool, ...], /
) -> _CapacityLayout:
    return _CapacityLayout(
        variables,
        blocks,
        zero_blocks,
        _capacity(variables),
        tuple(_capacity(block) for block in blocks),
    )


def _capacity_demand(layout: _CapacityLayout, /, *, quadratic: bool) -> tuple[int, int]:
    """Dense input and reduced-factor entries of one capacity class."""
    n, rows = layout.variable_capacity, sum(layout.block_capacities)
    zero = sum(
        capacity
        for capacity, is_zero in zip(
            layout.block_capacities, layout.zero_blocks, strict=True
        )
        if is_zero
    )
    inputs = rows * n + rows + 3 * n + (n * n if quadratic else 0)
    return inputs, n * n + zero * zero


def _capacity_admitted(
    layout: _CapacityLayout, policy: ConvexSolvePolicy, /, *, quadratic: bool
) -> bool:
    """Whether the declared budgets admit dense execution of this class.

    Capacity classes trade a bounded dense program for one compiled solve per
    class. A class beyond the materialization or factorization budget keeps
    the sparse program and its per-structure factorization.
    """
    inputs, factor = _capacity_demand(layout, quadratic=quadratic)
    itemsize = np.dtype(np.float64).itemsize
    reduced = layout.variable_capacity + sum(
        capacity
        for capacity, is_zero in zip(
            layout.block_capacities, layout.zero_blocks, strict=True
        )
        if is_zero
    )
    materialization, resources = policy.materialization, policy.resources
    return (
        inputs <= materialization.max_entries
        and reduced * reduced <= materialization.max_entries
        and inputs * itemsize <= materialization.max_bytes
        and factor * itemsize <= resources.factorization_bytes
        and (inputs + reduced * reduced) * itemsize
        <= min(resources.workspace_bytes, materialization.max_workspace_bytes)
    )


def _capacity_program(
    layout: _CapacityLayout,
    row_indices: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    rhs: np.ndarray,
    linear: np.ndarray,
    /,
    *,
    quadratic: bool,
    kind: str,
    evidence: str,
) -> ConicProgram:
    """Embed one program in its capacity class as bounded dense data.

    Padding variables have zero cost and no coefficients, and padding rows have
    zero coefficients and right-hand side, so the padded program has the same
    solutions on the original coordinates. Its identity names only the class:
    every program of one class shares one compiled solve.
    """
    n = layout.variable_capacity
    capacities = layout.block_capacities
    rows = layout.rows
    matrix = np.zeros((sum(capacities), n), dtype=np.float64)
    np.add.at(matrix, (rows[row_indices], columns), values)
    right = np.zeros(sum(capacities), dtype=np.float64)
    right[rows] = rhs
    objective = np.zeros(n, dtype=np.float64)
    objective[: layout.variables] = linear
    cone = ProductCone(
        tuple(
            ZeroCone(capacity) if is_zero else NonnegativeCone(capacity)
            for capacity, is_zero in zip(capacities, layout.zero_blocks, strict=True)
        )
    )
    sizes = "x".join(str(capacity) for capacity in capacities)
    return ConicProgram(
        jnp.asarray(np.eye(n, dtype=np.float64)) if quadratic else None,
        jnp.asarray(objective),
        jnp.asarray(matrix),
        jnp.asarray(right),
        cone,
        problem_id=f"coupled-{kind}-capacity:{n}:{sizes}",
        convexity_evidence=evidence,
    )


def _prepare_capacity(
    program: ConicProgram,
    execution: ConicProgram,
    layout: _CapacityLayout,
    demand: _ProgramDemand,
    policy: ConvexSolvePolicy,
    cache: _ProgramCache,
) -> _PreparedCoupledProgram:
    method = policy.method
    if not isinstance(method, NativeHomogeneousConic) or method.newton != "factorized":
        raise TypeError("Coupled programs require factorized NativeHomogeneousConic.")
    selected = cache.execution_policies.get(policy.policy_id)
    if selected is None:
        selected = ConvexSolvePolicy(
            NativeHomogeneousConic(
                primal_step=method.primal_step,
                dual_step=method.dual_step,
                extrapolation=method.extrapolation,
                newton="reduced",
                factorization=method.factorization,
            ),
            termination=policy.termination,
            regularization=policy.regularization,
            materialization=policy.materialization,
            resources=policy.resources,
            failure=policy.failure,
        )
        cache.execution_policies[policy.policy_id] = selected
    key = (execution.structure_id, selected.policy_id)
    template = cache.capacity.get(key)
    if template is None:
        template = prepare_convex_template(execution, selected)
        cache.capacity[key] = template
    inputs, factor = _capacity_demand(layout, quadratic=execution.quadratic is not None)
    itemsize = np.dtype(np.float64).itemsize
    admitted = demand._replace(
        factor_nnz=factor,
        factor_bytes=factor * itemsize,
        workspace_bytes=inputs * itemsize,
        symbolic_work=0,
    )
    return _PreparedCoupledProgram(
        program, bind_convex_numeric(template, execution), admitted, layout, policy
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
    evidence = "Identity quadratic is positive definite by construction."
    program = ConicProgram(
        quadratic,
        jnp.zeros(n, dtype=jnp.float64),
        matrix,
        jnp.asarray(np.concatenate((np.zeros(len(equalities), dtype=np.float64), right))),
        ProductCone((ZeroCone(len(equalities)), NonnegativeCone(len(rows)))),
        problem_id=structure,
        convexity_evidence=evidence,
    )
    layout = _layout(n, (len(equalities), len(rows)), (True, False))
    if not _capacity_admitted(layout, policy, quadratic=True):
        return _prepare(program, demand, policy, cache, certificate=False)
    execution = _capacity_program(
        layout,
        row_indices,
        columns,
        values,
        np.concatenate((np.zeros(len(equalities), dtype=np.float64), right)),
        np.zeros(n, dtype=np.float64),
        quadratic=True,
        kind="primary",
        evidence=evidence,
    )
    return _prepare_capacity(program, execution, layout, demand, policy, cache)


def prepare_certificate(
    primary: _PreparedCoupledProgram,
    policy: ConvexSolvePolicy,
    cache: _ProgramCache,
) -> _PreparedCoupledProgram:
    if primary.policy.policy_id != policy.policy_id:
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
    evidence = (
        "Linear objective, original transpose equalities, and explicit dual box rows."
    )
    program = ConicProgram(
        None,
        original.constraint_rhs,
        operator,
        jnp.asarray(rhs),
        ProductCone((ZeroCone(n), NonnegativeCone(m), NonnegativeCone(m))),
        problem_id=identity,
        convexity_evidence=evidence,
    )
    layout = _layout(m, (n, m, m), (True, False, False))
    if not _capacity_admitted(layout, policy, quadratic=False):
        return _prepare(program, demand, policy, cache, certificate=True)
    execution = _capacity_program(
        layout,
        rows[order],
        columns[order],
        values[order],
        rhs,
        np.asarray(original.constraint_rhs, dtype=np.float64),
        quadratic=False,
        kind="certificate",
        evidence=evidence,
    )
    return _prepare_capacity(program, execution, layout, demand, policy, cache)


def _original_result(
    result: ConvexProgramResult, prepared: _PreparedCoupledProgram, /
) -> ConvexProgramResult:
    """Restore one capacity-class result to the original program's coordinates."""
    layout = prepared.layout
    if layout is None:
        return result
    n, capacity, rows = layout.variables, layout.variable_capacity, layout.rows
    blocks = len(layout.blocks)
    complementarity = np.concatenate(
        (
            np.arange(blocks),
            blocks + np.arange(n),
            blocks + capacity + np.arange(n),
        )
    )

    def variables(value: Array) -> Array:
        return jnp.asarray(np.asarray(value)[:n])

    def constraints(value: Array) -> Array:
        return jnp.asarray(np.asarray(value)[rows])

    executed = result.provenance
    provenance = ConvexProgramProvenance(
        numeric_version=executed.numeric_version,
        problem_id=prepared.program.problem_id,
        structure_id=prepared.program.structure_id,
        policy_id=executed.policy_id,
        method_id=executed.method_id,
        backend=executed.backend,
        backend_version=executed.backend_version,
        convexity_evidence=prepared.program.convexity_evidence,
        regularization=executed.regularization,
        numeric_binding_id=executed.numeric_binding_id,
    )
    return eqx.tree_at(
        lambda value: (
            value.primal,
            value.lower_bound_dual,
            value.upper_bound_dual,
            value.stationarity_residual,
            value.solver_stationarity_residual,
            value.certificate.primal_ray,
            value.certificate.lower_bound_dual_ray,
            value.certificate.upper_bound_dual_ray,
            value.cone_slack,
            value.cone_dual,
            value.cone_primal_residual,
            value.cone_violation,
            value.cone_dual_violation,
            value.certificate.inequality_dual_ray,
            value.complementarity_residual,
            value.provenance,
        ),
        result,
        (
            variables(result.primal),
            variables(result.lower_bound_dual),
            variables(result.upper_bound_dual),
            variables(result.stationarity_residual),
            variables(result.solver_stationarity_residual),
            variables(result.certificate.primal_ray),
            variables(result.certificate.lower_bound_dual_ray),
            variables(result.certificate.upper_bound_dual_ray),
            constraints(result.cone_slack),
            constraints(result.cone_dual),
            constraints(result.cone_primal_residual),
            constraints(result.cone_violation),
            constraints(result.cone_dual_violation),
            constraints(result.certificate.inequality_dual_ray),
            jnp.asarray(np.asarray(result.complementarity_residual)[complementarity]),
            provenance,
        ),
    )


def solve_coupled(prepared: _PreparedCoupledProgram, /) -> ConvexProgramResult:
    """Execute one prepared coupled program; report in original coordinates."""
    result = solve_prepared_convex_program(prepared.prepared).result
    return _original_result(result, prepared)


_AUDIT_RAY = eqx.filter_jit(audit_conic_dual_infeasibility_ray)


def audit_dual_ray(
    prepared: _PreparedCoupledProgram,
    candidate: Array,
    lower_candidate: Array,
    upper_candidate: Array,
    /,
    *,
    tolerance: float,
) -> DualRayAudit:
    """Audit an original-coordinate dual ray against the prepared program.

    A capacity-class program holds the original coefficients with zero
    padding, so auditing the zero-extended ray on it is the same audit and
    shares the class's compiled kernel.
    """
    layout = prepared.layout
    if layout is None:
        return _AUDIT_RAY(
            prepared.program,
            candidate,
            lower_candidate,
            upper_candidate,
            tolerance=tolerance,
        )
    n, rows = layout.variables, layout.rows

    def extended(value: Array, capacity: int, index: np.ndarray) -> Array:
        host = np.zeros(capacity, dtype=np.float64)
        host[index] = np.asarray(value, dtype=np.float64)
        return jnp.asarray(host)

    execution = prepared.prepared.program
    if not isinstance(execution, ConicProgram):
        raise TypeError("A capacity-class binding holds a conic execution program.")
    variables = np.arange(n)
    audit = _AUDIT_RAY(
        execution,
        extended(candidate, sum(layout.block_capacities), rows),
        extended(lower_candidate, layout.variable_capacity, variables),
        extended(upper_candidate, layout.variable_capacity, variables),
        tolerance=tolerance,
    )
    return DualRayAudit(
        audit.equality_ray,
        jnp.asarray(np.asarray(audit.inequality_ray)[rows]),
        jnp.asarray(np.asarray(audit.lower_bound_ray)[:n]),
        jnp.asarray(np.asarray(audit.upper_bound_ray)[:n]),
        audit.residual_norm,
        audit.objective,
        audit.valid,
    )
