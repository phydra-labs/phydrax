# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real site-MPI/cuDSS qualification under authenticated coupled supervision.

The common collective worker initializes MPI/JAX and owns the real lease. This
module never constructs a pretend communicator or substitutes a dense executor.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from functools import partial
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

from phydrax._cuda_native import cuda_component_function
from phydrax._execution_collectives import (
    CollectiveBinding,
    CollectiveLease,
    prepare_collective_program,
)
from phydrax._execution_launch import (
    launch_local_processes,
    LocalProcessLaunchError,
    LocalProcessLaunchPlan,
)
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax.linalg._distributed import DistributedCoordinateLayout
from phydrax.linalg._distributed_sparse import (
    distributed_sparse_action,
    DistributedCSRLinearOperator,
    DistributedCSRStorage,
)
from phydrax.linalg._plans import LinearSolvePlan
from phydrax.linalg._policies import (
    CudssExecutionPolicy,
    LinearDerivativeSolvePolicy,
    LinearSolvePolicy,
    SparseLDLT,
    TolerancePolicy,
)
from phydrax.linalg._problems import LinearSystem
from phydrax.linalg._spaces import ArraySpace, RHSLayout
from phydrax.linalg.backends import _cudss_mgmn as mgmn
from phydrax.linalg.backends._cudss import CudssBackendOutput
from phydrax.linalg.backends._provider import provider_for


_ROOT = Path(__file__).resolve().parents[2]
_WORKER = _ROOT / "tests" / "_support" / "cuda_collective_worker.py"


def _launch(dtype: str, failure_case: str | None = None) -> tuple[dict[str, Any], ...]:
    executable = os.environ.get("PHYDRAX_TEST_CUDSS_MGMN_MPIEXEC")
    if executable is None:
        pytest.skip(
            "cuDSS MGMN qualification requires an explicit site MPI executable and site-built components"
        )
    if not Path(executable).is_absolute() or not Path(executable).is_file():
        raise ValueError(
            "PHYDRAX_TEST_CUDSS_MGMN_MPIEXEC must name the exact absolute site mpiexec"
        )
    ranks = int(os.environ.get("PHYDRAX_TEST_CUDSS_MGMN_RANKS", "2"))
    if ranks not in (2, 4, 8):
        raise ValueError("MGMN qualification requires 2, 4, or 8 real GPU processes")
    with socket.socket() as coordinator:
        coordinator.bind(("127.0.0.1", 0))
        port = coordinator.getsockname()[1]
    arguments = [
        sys.executable,
        str(_WORKER),
        "--case",
        "mgmn",
        "--dtype",
        dtype,
        "--rounds",
        "2",
    ]
    if failure_case is not None:
        arguments.extend(("--failure-case", failure_case))
    results = launch_local_processes(
        arguments,
        LocalProcessLaunchPlan(ranks, f"127.0.0.1:{port}", devices_per_process=1),
        mpi_launcher=(executable,),
        cwd=str(_ROOT),
        timeout_seconds=240,
        max_output_bytes=1024 * 1024,
        environment={
            **os.environ,
            "JAX_ENABLE_X64": "true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
    )
    records: list[dict[str, Any]] = []
    for result in results:
        for line in result.stdout.decode().splitlines():
            if line.startswith("{"):
                record = json.loads(line)
                if record.get("case") == "mgmn":
                    records.append(record)
    if len(records) != ranks:
        raise AssertionError(
            f"expected one MGMN qualification record per rank, received {records}"
        )
    return tuple(records)


@pytest.mark.cuda_native
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_mgmn_real_lifecycle_owned_residual_and_implicit_ad(dtype: str) -> None:
    records = _launch(dtype)
    for record in records:
        assert record["dtype"] == dtype
        rounds = record["rounds"]
        assert len(rounds) == 2
        for generation, round_record in enumerate(rounds):
            assert round_record["generation"] == generation
            assert round_record["native_lease_id"] > 0
            assert round_record["owner_thread_refused"] is True
            assert round_record["mpi_provenance"]
            evidence = round_record["evidence"]
            assert evidence["global_n"] > evidence["owned_rows"]
            assert evidence["nrhs"] == 3
            assert evidence["unequal_owned_rows"] is True
            assert evidence["owned_residual"] is True
            assert evidence["ordered_wrapper_cache_reuse"] is True
            assert evidence["refresh"] is True
            assert evidence["released_rebuilt"] is True
            assert evidence["asymmetric_memo_miss_rebuilt"] is True
            assert evidence["sparse_transpose"] is True
            assert evidence["values_rhs_jvp_vjp"] is True
            assert evidence["transformed_value_vjp"] is True
            assert evidence["zero_cotangent"] is True
            assert evidence["second_order"] is True
            assert evidence["inertia_unknown"] is True
            assert evidence["hybrid_estimates_unknown"] is True


@pytest.mark.cuda_native
@pytest.mark.parametrize(
    "failure_case",
    (
        "mgmn-asymmetric-symbolic-activity",
        "mgmn-one-rank-invalid-values",
        "mgmn-structure-change",
        "mgmn-stale-ticket",
        "mgmn-singular-refresh",
    ),
)
def test_mgmn_collective_refusal_has_no_orphan_ranks(failure_case: str) -> None:
    with pytest.raises(LocalProcessLaunchError) as failure:
        _launch("float64", failure_case)
    assert not failure.value.timed_out, (
        "refusal must be coupled, not a parent timeout recovery"
    )
    assert any(result.returncode != 0 for result in failure.value.results)


def _text(field: str) -> str:
    result = cuda_component_function(
        "cudss_mgmn", f"phx_cuda_cudss_mgmn_communication_{field}"
    )()
    if not isinstance(result, bytes):
        raise TypeError("site cuDSS provenance getter did not return bytes")
    return result.decode()


def _case(
    lease: CollectiveLease,
    dtype_name: str,
) -> tuple[LinearSystem, LinearSolvePlan, NDArray[Any], NDArray[Any], tuple[int, int]]:
    dtype = np.dtype(dtype_name)
    count = len(lease.spec.process_indices)
    rank = lease.spec.process_indices.index(jax.process_index())
    n = 2 * count + 1
    ranges = tuple(
        (2 * index, 2 * index + 2 if index < count - 1 else n) for index in range(count)
    )
    start, stop = ranges[rank]
    matrix = np.diag(np.full((n,), 4, dtype=dtype))
    matrix += np.diag(np.full((n - 1,), -1, dtype=dtype), 1)
    matrix += np.diag(np.full((n - 1,), -1, dtype=dtype), -1)
    columns: list[int] = []
    values: list[float] = []
    offsets = [0]
    for row in range(start, stop):
        for column in range(max(0, row - 1), min(n, row + 2)):
            columns.append(column)
            values.append(float(matrix[row, column]))
        offsets.append(len(columns))
    storage = DistributedCSRStorage(
        np.asarray(values, dtype=dtype),
        np.asarray(columns, dtype=np.int32),
        np.asarray(offsets, dtype=np.int32),
        global_shape=(n, n),
        row_layout=DistributedCoordinateLayout(n, ranges),
        partition_index=rank,
        structure_id=lease.spec.preparation_id,
        self_adjoint=True,
    )
    space = ArraySpace((stop - start,), dtype=dtype)
    operator = DistributedCSRLinearOperator(storage, space, space, binding=lease.binding)
    problem = LinearSystem(operator)
    group = ExecutionGroupSpec(
        lease.spec.group_id, lease.spec.process_indices, lease.spec.device_keys
    )
    method = SparseLDLT(
        execution=CudssExecutionPolicy(
            "mgmn",
            group=group,
            communication_library_path=_text("library_path"),
            communication_library_sha256=_text("library_sha256"),
            communication_build_identity=_text("build_identity"),
        ),
        refinement_steps=1,
        factor_cache_capacity=2,
    )
    relative = 2e-5 if dtype == np.dtype(np.float32) else 1e-10
    policy = LinearSolvePolicy(
        method,
        tolerance=TolerancePolicy(relative=relative, absolute=relative * 0.01),
        derivative_solve=LinearDerivativeSolvePolicy(
            relative_tolerance=relative, absolute_tolerance=relative * 0.01
        ),
    )
    plan = LinearSolvePlan(
        problem=problem,
        policy=policy,
        rhs_layout=RHSLayout((3,)),
        backend="cudss-mgmn",
        preconditioner_plan=None,
        method="sparse-ldlt",
        reason="explicit real MGMN qualification",
    )
    coordinates = np.arange(n, dtype=dtype)
    rhs = np.stack(
        (coordinates + 1, (-1) ** np.arange(n), np.ones(n, dtype=dtype)), axis=1
    ).astype(dtype)
    return problem, plan, matrix, rhs, (start, stop)


def run_mgmn_cases(
    lease: CollectiveLease, source_communicator: Any, dtype_name: str
) -> dict[str, object]:
    """Called only by the common authenticated, actual-MPI collective worker."""
    if source_communicator.Get_size() != len(lease.spec.process_indices):
        raise ValueError("MGMN qualification refuses independent one-rank MPI worlds")
    problem, plan, matrix, global_rhs, (start, stop) = _case(lease, dtype_name)
    operator = problem.operator
    if not isinstance(operator, DistributedCSRLinearOperator):
        raise TypeError("MGMN qualification requires its distributed CSR operator")
    storage = operator.storage
    rhs = jnp.asarray(global_rhs[start:stop])
    tolerance = 5e-5 if dtype_name == "float32" else 2e-9

    @eqx.filter_jit
    def prepare(
        current: LinearSystem,
    ) -> tuple[mgmn.CudssMGMNSymbolicState, mgmn.CudssMGMNFactorState]:
        symbolic = mgmn.analyze_cudss_mgmn(current, plan)
        return symbolic, mgmn.bind_cudss_mgmn(symbolic, current, plan)

    @eqx.filter_jit
    def execute(current: mgmn.CudssMGMNFactorState, right: Array) -> CudssBackendOutput:
        return mgmn.solve_cudss_mgmn(current, right, plan)

    preparation_program = prepare_collective_program(
        prepare, problem, lease_key=lease.spec
    )
    symbolic, state = lease.invoke(preparation_program, problem)
    solve_program = prepare_collective_program(execute, state, rhs, lease_key=lease.spec)
    expected = np.linalg.solve(matrix, global_rhs)
    output = lease.invoke(solve_program, state, rhs)
    np.testing.assert_allclose(
        output.value, expected[start:stop], rtol=tolerance, atol=tolerance
    )
    np.testing.assert_array_equal(output.status, np.zeros(3, dtype=np.int32))
    # Exercise the same compiled outer-jit/Equinox/scientific-primitive stack
    # again, with changed numerical leaves. One-shot callback observers cannot
    # detect an AOT fast path that omits the ordered runtime token on reuse.
    repeated_output = lease.invoke(solve_program, state, rhs * 2)
    np.testing.assert_allclose(
        repeated_output.value,
        expected[start:stop] * 2,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_array_equal(repeated_output.status, np.zeros(3, dtype=np.int32))
    for inertia in (state.positive_inertia, state.negative_inertia, state.zero_inertia):
        np.testing.assert_array_equal(inertia, -1)
    assert output.memory_evidence.byte_words.shape == (1, 6, 2)
    np.testing.assert_array_equal(
        output.memory_evidence.byte_words[:, 4:, :],
        np.full((1, 2, 2), np.iinfo(np.uint32).max, dtype=np.uint32),
    )
    assert output.memory_evidence.host_estimates()[0, 4:].tolist() == [None, None]

    @eqx.filter_jit
    def action(current: mgmn.CudssMGMNFactorState, vector: Array) -> tuple[Array, Array]:
        return (
            distributed_sparse_action(current.storage, current.binding, vector),
            distributed_sparse_action(
                current.storage, current.binding, vector, transpose=True
            ),
        )

    expected_local = jnp.asarray(expected[start:stop])
    action_program = prepare_collective_program(
        action, state, expected_local, lease_key=lease.spec
    )
    forward, transpose = lease.invoke(action_program, state, expected_local)
    np.testing.assert_allclose(
        forward, global_rhs[start:stop], rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        transpose, (matrix.T @ expected)[start:stop], rtol=tolerance, atol=tolerance
    )
    positions = np.arange(storage.nnz, dtype=np.int32)
    local_rows = np.searchsorted(np.asarray(storage.indptr)[1:], positions, side="right")
    global_rows = local_rows + start
    columns = np.asarray(storage.indices)
    dvalues = jnp.asarray(
        np.where(global_rows == columns, 0.05, 0), dtype=storage.values.dtype
    )
    global_dr = np.full_like(global_rhs, 0.02)
    global_dr[:2] = 0
    drhs = jnp.asarray(global_dr[start:stop])

    @eqx.filter_jit
    def directional(
        current: mgmn.CudssMGMNFactorState, right: Array, dv: Array, db: Array
    ) -> Array:
        def solve(values: Array, forcing: Array) -> Array:
            rebound = eqx.tree_at(lambda item: item.storage.values, current, values)
            return mgmn.solve_cudss_mgmn(rebound, forcing, plan).value

        return jax.jvp(solve, (current.storage.values, right), (dv, db))[1]

    directional_program = prepare_collective_program(
        directional, state, rhs, dvalues, drhs, lease_key=lease.spec
    )
    tangent = lease.invoke(directional_program, state, rhs, dvalues, drhs)
    tangent_reference = np.linalg.solve(matrix, global_dr - 0.05 * expected)
    np.testing.assert_allclose(
        tangent, tangent_reference[start:stop], rtol=5 * tolerance, atol=tolerance
    )

    @eqx.filter_jit
    def reverse(
        current: mgmn.CudssMGMNFactorState, right: Array, cotangent: Array
    ) -> tuple[Array, Array]:
        def solve(values: Array, forcing: Array) -> Array:
            rebound = eqx.tree_at(lambda item: item.storage.values, current, values)
            return mgmn.solve_cudss_mgmn(rebound, forcing, plan).value

        _, pullback = jax.vjp(solve, current.storage.values, right)
        return pullback(cotangent)

    global_cotangent = np.ones_like(global_rhs)
    global_cotangent[:2] = 0
    local_cotangent = jnp.asarray(global_cotangent[start:stop])
    reverse_program = prepare_collective_program(
        reverse, state, rhs, local_cotangent, lease_key=lease.spec
    )
    values_bar, rhs_bar = lease.invoke(reverse_program, state, rhs, local_cotangent)
    adjoint = np.linalg.solve(matrix.T, global_cotangent)
    expected_values_bar = -np.sum(adjoint[global_rows] * expected[columns], axis=1)
    np.testing.assert_allclose(
        values_bar, expected_values_bar, rtol=5 * tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        rhs_bar, adjoint[start:stop], rtol=5 * tolerance, atol=tolerance
    )
    zero_values, zero_rhs = lease.invoke(reverse_program, state, rhs, jnp.zeros_like(rhs))
    np.testing.assert_array_equal(zero_values, jnp.zeros_like(zero_values))
    np.testing.assert_array_equal(zero_rhs, jnp.zeros_like(zero_rhs))

    def transformed_reverse(
        current: mgmn.CudssMGMNFactorState,
        right: Array,
        cotangent: Array,
        *,
        adjoint: bool,
    ) -> tuple[Array, Array]:
        def solve(values: Array, forcing: Array) -> Array:
            rebound = eqx.tree_at(lambda item: item.storage.values, current, values)
            return (
                provider_for(plan.backend)
                .solve_transformed(
                    rebound,
                    forcing,
                    plan,
                    adjoint=adjoint,
                )
                .value
            )

        _, pullback = jax.vjp(solve, current.storage.values, right)
        return pullback(cotangent)

    expected_transpose_values_bar = -np.sum(
        expected[global_rows] * adjoint[columns], axis=1
    )
    for use_adjoint in (False, True):
        transformed_program = prepare_collective_program(
            partial(transformed_reverse, adjoint=use_adjoint),
            state,
            rhs,
            local_cotangent,
            lease_key=lease.spec,
        )
        transformed_values_bar, transformed_rhs_bar = lease.invoke(
            transformed_program,
            state,
            rhs,
            local_cotangent,
        )
        np.testing.assert_allclose(
            transformed_values_bar,
            expected_transpose_values_bar,
            rtol=5 * tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            transformed_rhs_bar,
            adjoint[start:stop],
            rtol=5 * tolerance,
            atol=tolerance,
        )

    @eqx.filter_jit
    def second_order(
        current: mgmn.CudssMGMNFactorState, right: Array, direction: Array
    ) -> Array:
        def objective(values: Array) -> Array:
            rebound = eqx.tree_at(lambda item: item.storage.values, current, values)
            return jnp.sum(jnp.square(mgmn.solve_cudss_mgmn(rebound, right, plan).value))

        return jax.jvp(jax.grad(objective), (current.storage.values,), (direction,))[1]

    second_order_program = prepare_collective_program(
        second_order, state, rhs, dvalues, lease_key=lease.spec
    )
    hessian_direction = lease.invoke(second_order_program, state, rhs, dvalues)
    dense_x = np.linalg.solve(matrix, global_rhs)
    dense_dx = np.linalg.solve(matrix, -0.05 * dense_x)
    dense_adjoint = np.linalg.solve(matrix.T, 2 * dense_x)
    dense_dadjoint = np.linalg.solve(matrix.T, 2 * dense_dx - 0.05 * dense_adjoint)
    expected_hessian = -np.sum(
        dense_dadjoint[global_rows] * dense_x[columns]
        + dense_adjoint[global_rows] * dense_dx[columns],
        axis=1,
    )
    np.testing.assert_allclose(
        hessian_direction, expected_hessian, rtol=10 * tolerance, atol=5 * tolerance
    )

    # A rank-asymmetric absent numeric hint exercises all-rank cache-miss
    # consensus without destroying a communicator on an unsynchronized rank.
    missing = eqx.tree_at(
        lambda item: item.id,
        state,
        jnp.asarray(-1 if start == 0 else state.id, dtype=np.int64),
    )
    rebuilt = lease.invoke(solve_program, missing, rhs)
    assert np.asarray(rebuilt.native_rebuilds) > 0
    np.testing.assert_allclose(
        rebuilt.value, expected[start:stop], rtol=tolerance, atol=tolerance
    )

    new_values = storage.values + jnp.asarray(
        np.where(global_rows == columns, 0.4, 0), dtype=storage.values.dtype
    )
    new_problem = eqx.tree_at(
        lambda item: item.operator.storage.values, problem, new_values
    )

    @eqx.filter_jit
    def refresh(
        symbolic_state: mgmn.CudssMGMNSymbolicState,
        previous: mgmn.CudssMGMNFactorState,
        current: LinearSystem,
    ) -> mgmn.CudssMGMNFactorState:
        return mgmn.refresh_cudss_mgmn(symbolic_state, previous, current, plan)

    refresh_program = prepare_collective_program(
        refresh, symbolic, state, new_problem, lease_key=lease.spec
    )
    refreshed = lease.invoke(refresh_program, symbolic, state, new_problem)
    new_expected = np.linalg.solve(
        matrix + np.eye(matrix.shape[0], dtype=matrix.dtype) * 0.4, global_rhs
    )
    refreshed_output = lease.invoke(solve_program, refreshed, rhs)
    np.testing.assert_allclose(
        refreshed_output.value, new_expected[start:stop], rtol=tolerance, atol=tolerance
    )
    release_program = prepare_collective_program(
        mgmn.release_cudss_mgmn, refreshed, lease_key=lease.spec
    )
    released = lease.invoke(release_program, refreshed)
    assert bool(released)
    after_release = lease.invoke(solve_program, refreshed, rhs)
    assert np.asarray(after_release.native_rebuilds) > 0
    np.testing.assert_allclose(
        after_release.value, new_expected[start:stop], rtol=tolerance, atol=tolerance
    )
    lease.invoke(release_program, refreshed)
    return {
        "global_n": matrix.shape[0],
        "owned_rows": stop - start,
        "nrhs": 3,
        "unequal_owned_rows": True,
        "owned_residual": True,
        "ordered_wrapper_cache_reuse": True,
        "refresh": True,
        "released_rebuilt": True,
        "asymmetric_memo_miss_rebuilt": True,
        "sparse_transpose": True,
        "values_rhs_jvp_vjp": True,
        "transformed_value_vjp": True,
        "zero_cotangent": True,
        "second_order": True,
        "inertia_unknown": True,
        "hybrid_estimates_unknown": True,
    }


def run_mgmn_failure_case(
    lease: CollectiveLease, source_communicator: Any, dtype_name: str, case: str
) -> dict[str, object]:
    del source_communicator
    problem, plan, _, global_rhs, (start, stop) = _case(lease, dtype_name)

    @eqx.filter_jit
    def prepare(
        current: LinearSystem,
    ) -> tuple[mgmn.CudssMGMNSymbolicState, mgmn.CudssMGMNFactorState]:
        symbolic = mgmn.analyze_cudss_mgmn(current, plan)
        return symbolic, mgmn.bind_cudss_mgmn(symbolic, current, plan)

    preparation_program = prepare_collective_program(
        prepare, problem, lease_key=lease.spec
    )
    symbolic, state = lease.invoke(preparation_program, problem)
    rhs = jnp.asarray(global_rhs[start:stop])

    @eqx.filter_jit
    def execute_invalid(current: mgmn.CudssMGMNFactorState, right: Array) -> Array:
        return mgmn.solve_cudss_mgmn(current, right, plan).value

    invalid_program = prepare_collective_program(
        execute_invalid, state, rhs, lease_key=lease.spec
    )

    match case:
        case "mgmn-asymmetric-symbolic-activity":

            @eqx.filter_jit
            def asymmetric(current: mgmn.CudssMGMNFactorState, forcing: Array) -> Array:
                def solve(right: Array) -> Array:
                    return mgmn.solve_cudss_mgmn(current, right, plan).value

                if start == 0:
                    return solve(jax.lax.stop_gradient(forcing))
                return jax.jvp(solve, (forcing,), (jnp.ones_like(forcing),))[1]

            asymmetric_program = prepare_collective_program(
                asymmetric, state, rhs, lease_key=lease.spec
            )
            lease.invoke(asymmetric_program, state, rhs)
        case "mgmn-one-rank-invalid-values":
            values = (
                state.storage.values.at[0].set(jnp.nan)
                if start == 0
                else state.storage.values
            )
            invalid = eqx.tree_at(lambda item: item.storage.values, state, values)
            lease.invoke(invalid_program, invalid, rhs)
        case "mgmn-structure-change":
            columns = (
                state.storage.indices.at[0].set(state.storage.global_shape[1] + 1)
                if start == 0
                else state.storage.indices
            )
            invalid = eqx.tree_at(lambda item: item.storage.indices, state, columns)
            lease.invoke(invalid_program, invalid, rhs)
        case "mgmn-stale-ticket":
            # A binding closed over by an executable is not the invocation leaf
            # that lease.invoke authorizes and rewrites.
            captured = state

            @eqx.filter_jit
            def stale(binding: CollectiveBinding, forcing: Array) -> Array:
                del binding
                return mgmn.solve_cudss_mgmn(captured, forcing, plan).value

            stale_program = prepare_collective_program(
                stale, lease.binding, rhs, lease_key=lease.spec
            )
            lease.invoke(stale_program, lease.binding, rhs)
        case "mgmn-singular-refresh":
            zero_problem = eqx.tree_at(
                lambda item: item.operator.storage.values,
                problem,
                jnp.zeros_like(state.storage.values),
            )

            @eqx.filter_jit
            def singular(
                symbolic_state: mgmn.CudssMGMNSymbolicState,
                previous: mgmn.CudssMGMNFactorState,
                current: LinearSystem,
            ) -> mgmn.CudssMGMNFactorState:
                return mgmn.refresh_cudss_mgmn(symbolic_state, previous, current, plan)

            singular_program = prepare_collective_program(
                singular, symbolic, state, zero_problem, lease_key=lease.spec
            )
            lease.invoke(singular_program, symbolic, state, zero_problem)
        case _:
            raise ValueError(f"unknown MGMN failure case {case}")
    raise AssertionError("invalid collective invocation was not refused")
