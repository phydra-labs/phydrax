#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Equal-input cuDSS phase measurements under an existing execution owner.

The caller supplies a selected method/group and, for MGMN, the already bound
site-MPI lease. This module never initializes another distributed world. Records
are rank-local campaign inputs, not fabricated all-rank qualification results.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    DurationDistribution,
    measure_host,
    measure_repeated,
    measure_synchronized,
)
from phydrax import _jax_config
from phydrax._execution_collectives import (
    CollectiveLease,
    CollectiveProgram,
    prepare_collective_program,
)
from phydrax._execution_device import device_platform


@dataclass(frozen=True)
class _ReferenceData:
    matrix: np.ndarray
    rhs: np.ndarray
    digest: str
    rows: np.ndarray
    columns: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class _PreparedCase:
    reference: _ReferenceData
    selected: jax.Device
    problem: phx.linalg.LinearSystem
    forcing: jax.Array
    start: int
    stop: int


def _select_device(
    method: phx.linalg.SparseLDLT, local_device: jax.Device | None
) -> jax.Device:
    mode = method.execution.mode
    group = method.execution.group
    devices = {
        (device.process_index, device.id): device for device in jax.local_devices()
    }
    if mode == "local":
        return jax.local_devices()[0] if local_device is None else local_device
    if group is None:
        raise ValueError("multi-device execution requires a declared group")
    key = (
        group.device_keys[0]
        if mode == "mg"
        else next(key for key in group.device_keys if key[0] == jax.process_index())
    )
    if key not in devices:
        raise ValueError("execution group references an unavailable local device")
    return devices[key]


def _reference_data(
    size: int, rhs_columns: int, dtype: str, indefinite: bool, selected: jax.Device
) -> _ReferenceData:
    dtype_ = np.dtype(dtype)
    diagonal = 4.0 + np.arange(size, dtype=dtype_) / np.asarray(1000, dtype=dtype_)
    if indefinite:
        diagonal *= np.where(np.arange(size) % 2, -1, 1).astype(dtype_)
    if device_platform(selected) != "cuda":
        raise ValueError(
            "cuDSS execution benchmarks require an explicitly selected CUDA device"
        )
    matrix = np.diag(diagonal)
    off_diagonal = np.full(size - 1, 0.05, dtype=dtype_)
    matrix += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    rhs = np.sin(
        np.arange(size, dtype=dtype_)[:, None] / np.asarray(10, dtype=dtype_)
        + np.arange(rhs_columns, dtype=dtype_)[None, :]
    ).astype(dtype_)
    digest = hashlib.sha256(matrix.tobytes() + rhs.tobytes()).hexdigest()
    rows, columns = np.nonzero(matrix)
    values = matrix[rows, columns]
    return _ReferenceData(matrix, rhs, digest, rows, columns, values)


def _prepare_case(
    method: phx.linalg.SparseLDLT,
    reference: _ReferenceData,
    selected: jax.Device,
    lease: CollectiveLease | None,
) -> _PreparedCase:
    size = reference.matrix.shape[0]
    rows, columns, values = reference.rows, reference.columns, reference.values
    start, stop = 0, size
    if method.execution.mode == "mgmn":
        group = method.execution.group
        if lease is None or group is None:
            raise ValueError("MGMN requires its live group lease")
        lease.spec.require_group(group)
        partitions = len(group.process_indices)
        if size < partitions:
            raise ValueError("benchmark cannot insert empty or padded rank-local rows")
        ranges = tuple(
            (size * rank // partitions, size * (rank + 1) // partitions)
            for rank in range(partitions)
        )
        layout = phx.linalg.DistributedCoordinateLayout(size, ranges)
        partition = group.process_indices.index(jax.process_index())
        start, stop = ranges[partition]
        owned = (rows >= start) & (rows < stop)
        local_rows = rows[owned] - start
        pointers = np.concatenate(
            (
                np.zeros(1, dtype=np.int32),
                np.cumsum(
                    np.bincount(local_rows, minlength=stop - start), dtype=np.int32
                ),
            )
        )
        storage = phx.linalg.DistributedCSRStorage(
            jnp.asarray(values[owned]),
            jnp.asarray(columns[owned], dtype=jnp.int32),
            jnp.asarray(pointers),
            global_shape=(size, size),
            row_layout=layout,
            partition_index=partition,
            structure_id=f"cudss-benchmark-{size}-tridiagonal",
            self_adjoint=True,
        )
        space = phx.linalg.ArraySpace(
            (stop - start,),
            dtype=reference.matrix.dtype,
            space_id=f"cudss-benchmark-{reference.digest}-owned-{partition}",
        )
        operator = phx.linalg.DistributedCSRLinearOperator(
            storage, space, space, binding=lease.binding
        )
    else:
        relation = phx.sparse.EdgeRelation(
            columns.astype(np.int32),
            rows.astype(np.int32),
            source_size=size,
            target_size=size,
        )
        operator = phx.sparse.SparseLinearMap(
            relation,
            jnp.asarray(values),
            operator_id=f"cudss-benchmark-{size}-tridiagonal",
            properties=phx.linalg.OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "asserted"}
            ),
        )
        operator = phx.linalg.assemble_sparse(operator)
    problem = phx.linalg.LinearSystem(operator)
    forcing = jnp.asarray(reference.rhs[start:stop])
    return _PreparedCase(reference, selected, problem, forcing, start, stop)


def _refreshed_problem(case: _PreparedCase) -> phx.linalg.LinearSystem:
    operator = case.problem.operator
    if isinstance(operator, phx.linalg.DistributedCSRLinearOperator):
        coefficients = operator.storage.values * np.asarray(
            1.1, dtype=case.reference.matrix.dtype
        )
        return eqx.tree_at(
            lambda item: item.operator.storage.values, case.problem, coefficients
        )
    if not isinstance(operator, phx.sparse.SparseCoordinateOperator):
        raise TypeError("Local benchmark refresh requires its prepared sparse operator.")
    coefficients = operator.coefficients * np.asarray(
        1.1, dtype=case.reference.matrix.dtype
    )
    return eqx.tree_at(
        lambda item: item.operator.coefficients, case.problem, coefficients
    )


def _case_evidence(
    method: phx.linalg.SparseLDLT,
    case: _PreparedCase,
    result: phx.linalg.LinearSolveResult,
    changed: phx.linalg.LinearSolveResult,
    relative: float,
    absolute: float,
    times: dict[str, float],
    steady: DurationDistribution,
) -> dict[str, Any]:
    matrix, rhs = case.reference.matrix, case.reference.rhs
    expected = np.linalg.solve(matrix, rhs)[case.start : case.stop]
    changed_expected = np.linalg.solve(matrix * 1.1, rhs * 0.75)[case.start : case.stop]
    error = float(np.max(np.abs(np.asarray(result.value) - expected)))
    changed_error = float(np.max(np.abs(np.asarray(changed.value) - changed_expected)))
    memory = result.diagnostics.external_memory
    group = method.execution.group
    return {
        "mode": method.execution.mode,
        "memory_mode": method.memory_mode,
        "input_sha256": case.reference.digest,
        "global_shape": list(matrix.shape),
        "owned_rows": [case.start, case.stop],
        "rhs_columns": rhs.shape[1],
        "dtype": matrix.dtype.name,
        "relative_tolerance": relative,
        "absolute_tolerance": absolute,
        "process_index": jax.process_index(),
        "group_id": None if group is None else group.group_id,
        "device_keys": [[case.selected.process_index, case.selected.id]]
        if group is None
        else [list(key) for key in group.device_keys],
        "plan_seconds": times["plan_seconds"],
        "first_analysis_seconds": times["first_analysis_seconds"],
        "first_factorization_seconds": times["first_factorization_seconds"],
        "first_solve_seconds": times["first_solve_seconds"],
        "steady_solve": steady.to_seconds_dict(),
        "first_refresh_seconds": times["first_refresh_seconds"],
        "changed_solve_seconds": times["changed_solve_seconds"],
        "solution_max_abs_error_local": error,
        "changed_max_abs_error_local": changed_error,
        "residual_norm": np.asarray(result.diagnostics.residual_norm).tolist(),
        "native_rebuilds": int(np.asarray(result.provenance.native_rebuilds)),
        "memory_estimates": None if memory is None else memory.host_estimates().tolist(),
        "memory_basis": "provider_estimate",
        "certified_memory_bounds": False,
        "host_estimates_additive": False,
        "analysis_rhs_columns": 1,
        "successful": bool(np.all(np.asarray(result.successful)))
        and bool(np.all(np.asarray(changed.successful)))
        and bool(
            np.allclose(np.asarray(result.value), expected, rtol=relative, atol=absolute)
        )
        and bool(
            np.allclose(
                np.asarray(changed.value), changed_expected, rtol=relative, atol=absolute
            )
        ),
    }


def _execute_case(
    method: phx.linalg.SparseLDLT,
    case: _PreparedCase,
    dtype: str,
    repeats: int,
    lease: CollectiveLease | None,
) -> dict[str, Any]:
    problem, forcing = case.problem, case.forcing
    relative = 3e-5 if dtype == "float32" else 1e-10
    absolute = 3e-6 if dtype == "float32" else 1e-12
    policy = phx.linalg.LinearSolvePolicy(
        method, tolerance=phx.linalg.TolerancePolicy(relative=relative, absolute=absolute)
    )
    times: dict[str, float] = {}
    planned, times["plan_seconds"] = measure_host(
        lambda: phx.linalg.plan(
            problem, policy, rhs_layout=phx.linalg.RHSLayout((forcing.shape[1],))
        )
    )
    analyze = eqx.filter_jit(phx.linalg.prepare_template)
    bind = eqx.filter_jit(phx.linalg.bind_numeric)
    solve = eqx.filter_jit(phx.linalg.solve)
    refresh = eqx.filter_jit(phx.linalg.refresh)
    release = eqx.filter_jit(phx.linalg.release)

    def stage(
        function: Callable[..., Any], *arguments: Any
    ) -> tuple[CollectiveProgram | Callable[..., Any], float]:
        if lease is None:
            return function, 0.0
        return measure_host(
            lambda: prepare_collective_program(function, *arguments, lease_key=lease.spec)
        )

    def invoke(program: CollectiveProgram | Callable[..., Any], *arguments: Any) -> Any:
        if lease is None:
            if isinstance(program, CollectiveProgram):
                raise TypeError("A collective program requires its owning lease.")
            return program(*arguments)
        if not isinstance(program, CollectiveProgram):
            raise TypeError("MGMN dispatch requires an explicitly staged program.")
        return lease.invoke(program, *arguments)

    analysis_program, analysis_staging = stage(analyze, problem, planned)
    template, times["first_analysis_seconds"] = measure_synchronized(
        lambda: invoke(analysis_program, problem, planned)
    )
    times["first_analysis_seconds"] += analysis_staging
    factor_program, factor_staging = stage(bind, template, problem)
    prepared, times["first_factorization_seconds"] = measure_synchronized(
        lambda: invoke(factor_program, template, problem)
    )
    times["first_factorization_seconds"] += factor_staging
    solve_program, solve_staging = stage(solve, prepared, forcing)
    release_program, release_staging = stage(release, prepared)
    refreshed = None
    try:
        result, times["first_solve_seconds"] = measure_synchronized(
            lambda: invoke(solve_program, prepared, forcing)
        )
        times["first_solve_seconds"] += solve_staging
        result, steady = measure_repeated(
            lambda: invoke(solve_program, prepared, forcing), warmup=1, repeats=repeats
        )
        current = _refreshed_problem(case)
        refresh_program, refresh_staging = stage(refresh, prepared, current)
        refreshed, times["first_refresh_seconds"] = measure_synchronized(
            lambda: invoke(refresh_program, prepared, current)
        )
        times["first_refresh_seconds"] += refresh_staging
        changed, times["changed_solve_seconds"] = measure_synchronized(
            lambda: invoke(solve_program, refreshed, forcing * 0.75)
        )
        record = _case_evidence(
            method, case, result, changed, relative, absolute, times, steady
        )
    finally:
        if refreshed is not None:
            invoke(release_program, refreshed)
        _, release_seconds = measure_synchronized(
            lambda: invoke(release_program, prepared)
        )
        release_seconds += release_staging
    record["release_seconds"] = release_seconds
    return record


def cudss_execution_case(
    method: phx.linalg.SparseLDLT,
    /,
    *,
    size: int = 128,
    rhs_columns: int = 4,
    dtype: str = "float64",
    indefinite: bool = False,
    repeats: int = 5,
    lease: CollectiveLease | None = None,
    local_device: jax.Device | None = None,
) -> dict[str, Any]:
    """Measure plan/analyze/bind/solve/refresh/release on the same global model.

    Compare only equal input hashes, precision, RHS counts, and tolerances.
    First phase times include first lowering/compilation; steady solves are
    separately synchronized. The independent host oracle remains outside all
    measured phases, not a numerical execution fallback.
    """
    if not isinstance(method, phx.linalg.SparseLDLT):
        raise TypeError("method must be SparseLDLT")
    if (
        type(size) is not int
        or size < 2
        or type(rhs_columns) is not int
        or rhs_columns < 1
    ):
        raise ValueError("size >= 2 and rhs_columns >= 1 must be explicit integers")
    if dtype not in ("float32", "float64") or type(repeats) is not int or repeats < 1:
        raise ValueError("dtype must be float32/float64 and repeats must be positive")
    mode = method.execution.mode
    if dtype == "float64" and not _jax_config.x64_enabled():
        raise ValueError("float64 benchmark requires the existing runtime's x64 policy")
    if (mode == "mgmn") != (lease is not None):
        raise ValueError("only MGMN requires an existing live collective lease")
    selected = _select_device(method, local_device)
    reference = _reference_data(size, rhs_columns, dtype, indefinite, selected)
    with jax.default_device(selected):
        case, preparation_seconds = measure_synchronized(
            lambda: _prepare_case(method, reference, selected, lease),
            synchronizer=lambda prepared: jax.block_until_ready(
                (prepared.problem, prepared.forcing)
            ),
        )
        record = _execute_case(method, case, dtype, repeats, lease)
        record["operator_preparation_seconds"] = preparation_seconds
        return record


__all__ = ["cudss_execution_case"]
