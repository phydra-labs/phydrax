#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

import phydrax as phx
from phydrax import config as pxconfig
from phydrax._execution_device import device_platform
from phydrax._execution_resources import ExecutionGroupSpec


la = phx.linalg
pytestmark = pytest.mark.cuda_native


def _reservation(count: int) -> tuple[ExecutionGroupSpec, tuple[jax.Device, ...]]:
    availability = phx.backends.cudss_availability(execution_mode="mg")
    if not availability.available:
        pytest.skip(availability.reason)
    devices = tuple(
        device for device in jax.local_devices() if device_platform(device) == "cuda"
    )
    if len(devices) < count:
        pytest.skip(f"cuDSS MG case requires {count} local CUDA devices")
    start = min(count, len(devices) - count)
    selected = devices[start : start + count]
    return ExecutionGroupSpec(
        f"cudss-mg-{count}-{start}",
        (jax.process_index(),),
        tuple((device.process_index, device.id) for device in selected),
    ), selected


def _problem(matrix: jax.Array, identity: str) -> la.LinearSystem:
    size = matrix.shape[-1]
    rows, columns = np.indices((size, size), dtype=np.int32)
    relation = phx.sparse.EdgeRelation(
        columns.reshape(-1), rows.reshape(-1), source_size=size, target_size=size
    )
    return la.LinearSystem(
        phx.sparse.SparseLinearMap(
            relation,
            matrix.reshape(matrix.shape[:-2] + (-1,)),
            operator_id=identity,
            properties=la.OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "asserted"}
            ),
        )
    )


@pytest.mark.parametrize("count", [2, 4, 8])
@pytest.mark.parametrize("indefinite", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_full_input_mg_single_solve_refresh_rhs_and_release(
    count: int, indefinite: bool, dtype: DTypeLike
) -> None:
    group, devices = _reservation(count)
    with (
        pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)),
        jax.default_device(devices[0]),
    ):
        size = 32
        diagonal = jnp.arange(size, dtype=dtype) * 0.1 + 4.0
        if indefinite:
            diagonal = diagonal * jnp.where(jnp.arange(size) % 2, -1.0, 1.0)
        matrix = jnp.diag(diagonal) + jnp.full((size, size), 0.025, dtype=dtype)
        rhs = jnp.stack(
            (jnp.arange(size, dtype=dtype) + 1, jnp.ones(size, dtype=dtype)), axis=-1
        )
        problem = _problem(matrix, group.group_id)
        method = la.SparseLDLT(execution=la.CudssExecutionPolicy("mg", group=group))
        prepared = la.prepare(
            problem, la.LinearSolvePolicy(method), rhs_layout=la.RHSLayout((2,))
        )
        try:
            tolerance = 2e-5 if dtype == jnp.float32 else 1e-11
            result = eqx.filter_jit(la.solve)(prepared, rhs)
            assert jnp.all(result.successful)
            assert jnp.allclose(
                result.value,
                jnp.linalg.solve(matrix, rhs),
                rtol=tolerance,
                atol=tolerance,
            )
            assert result.value.devices() == {devices[0]}
            memory = result.diagnostics.external_memory
            assert memory is not None
            assert memory.device_indices == tuple(
                device.local_hardware_id for device in devices
            )
            assert memory.byte_words.shape == (count, 6, 2)
            assert memory.group_id == group.group_id
            assert not la.sparse_provider_capabilities("cudss").reliable_zero_inertia
            transpose = eqx.filter_jit(la.solve_transpose)(prepared, rhs)
            assert jnp.allclose(
                transpose.value, result.value, rtol=tolerance, atol=tolerance
            )
            changed_matrix = matrix + jnp.eye(size, dtype=dtype) * 0.2
            changed = eqx.tree_at(
                lambda item: item.operator.coefficients,
                problem,
                changed_matrix.reshape(-1),
            )
            refreshed = la.refresh(prepared, changed)
            try:
                changed_result = eqx.filter_jit(la.solve)(refreshed, rhs * 1.5)
                assert jnp.all(changed_result.successful)
                assert jnp.allclose(
                    changed_result.value,
                    jnp.linalg.solve(changed_matrix, rhs * 1.5),
                    rtol=tolerance,
                    atol=tolerance,
                )
            finally:
                la.release(refreshed)
            la.release(prepared)
            rebuilt = eqx.filter_jit(la.solve)(prepared, rhs)
            assert jnp.all(rebuilt.successful)
            assert jnp.allclose(
                rebuilt.value,
                jnp.linalg.solve(matrix, rhs),
                rtol=tolerance,
                atol=tolerance,
            )
        finally:
            la.release(prepared)


@pytest.mark.parametrize("count", [2, 4, 8])
def test_mg_mathematical_value_rhs_derivatives_and_zero_cotangent(count: int) -> None:
    group, devices = _reservation(count)
    with (
        pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)),
        jax.default_device(devices[0]),
    ):
        size = 16
        matrix = jnp.eye(size, dtype=jnp.float64) * 4 + jnp.full(
            (size, size), 0.05, dtype=jnp.float64
        )
        rhs = jnp.arange(size, dtype=jnp.float64) + 1
        problem = _problem(matrix, f"{group.group_id}-derivative")
        prepared = la.prepare(
            problem,
            la.LinearSolvePolicy(
                la.SparseLDLT(execution=la.CudssExecutionPolicy("mg", group=group))
            ),
        )
        try:

            def solve(values: jax.Array, forcing: jax.Array) -> jax.Array:
                current = eqx.tree_at(
                    lambda item: item.operator.coefficients, problem, values.reshape(-1)
                )
                return la.solve(la.refresh(prepared, current), forcing).value

            tangent_matrix = jnp.eye(size, dtype=jnp.float64) * 0.1
            tangent_rhs = jnp.ones(size, dtype=jnp.float64) * 0.2
            actual = eqx.filter_jit(
                lambda a, b: jax.jvp(solve, (a, b), (tangent_matrix, tangent_rhs))[1]
            )(matrix, rhs)
            expected = jax.jvp(
                jnp.linalg.solve, (matrix, rhs), (tangent_matrix, tangent_rhs)
            )[1]
            assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-11)
            _, pullback = jax.vjp(solve, matrix, rhs)
            _, reference = jax.vjp(jnp.linalg.solve, matrix, rhs)
            for cotangent in (jnp.ones_like(rhs), jnp.zeros_like(rhs)):
                for actual, expected in zip(
                    pullback(cotangent), reference(cotangent), strict=True
                ):
                    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-11)
        finally:
            la.release(prepared)


def test_mg_refuses_matrix_batch_and_peer_resident_rhs() -> None:
    group, devices = _reservation(2)
    with (
        pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)),
        jax.default_device(devices[0]),
    ):
        matrix = jnp.eye(8, dtype=jnp.float64) * 4
        policy = la.LinearSolvePolicy(
            la.SparseLDLT(execution=la.CudssExecutionPolicy("mg", group=group))
        )
        with pytest.raises(ValueError, match="refuses matrix batches"):
            la.plan(_problem(jnp.stack((matrix, matrix)), "mg-batch-refusal"), policy)
        prepared = la.prepare(_problem(matrix, "mg-placement-refusal"), policy)
        try:
            rhs = jax.device_put(jnp.ones(8, dtype=jnp.float64), devices[1])
            with pytest.raises(ValueError, match="initiating device"):
                la.solve(prepared, rhs)
        finally:
            la.release(prepared)
