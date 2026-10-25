# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""MGMN mathematical/ownership contracts; CPU observers do not qualify MPI."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax import config as pxconfig
from phydrax._execution_collective_ffi import collective_schedule
from phydrax._execution_collectives import CollectiveBinding, CollectiveLeaseSpec
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._jax_config import default_device
from phydrax.linalg import _distributed_sparse as sparse
from phydrax.linalg._distributed import DistributedCoordinateLayout
from phydrax.linalg._policies import CudssExecutionPolicy, CudssMemoryPolicy, SparseLDLT
from phydrax.linalg._results import LinearSolveStatus
from phydrax.linalg.backends import _cudss_mgmn as mgmn


@pytest.fixture(autouse=True)
def cpu_numerics() -> Iterator[None]:
    with (
        default_device(jax.local_devices(backend="cpu")[0]),
        pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)),
    ):
        yield


def _storage() -> sparse.DistributedCSRStorage:
    return sparse.DistributedCSRStorage(
        np.asarray([4, 1, 1, 3], dtype=np.float64),
        np.asarray([0, 1, 0, 1], dtype=np.int32),
        np.asarray([0, 2, 4], dtype=np.int32),
        global_shape=(2, 2),
        row_layout=DistributedCoordinateLayout(2, ((0, 2),)),
        partition_index=0,
        structure_id="global-two-by-two",
        self_adjoint=True,
    )


def _configuration(storage: sparse.DistributedCSRStorage) -> mgmn._Configuration:
    return mgmn._Configuration(
        global_n=2,
        row_starts=(0, 2),
        partition_index=0,
        structure_id=storage.structure_id,
        partition_id=storage.partition_id,
        storage_treedef=jax.tree.structure(storage),
        capacity=2,
        reordering=0,
        communication_library_path="/qualified-site/libcudss_commlayer_openmpi.so",
        communication_library_sha256="a" * 64,
        communication_build_identity="site-mpi-abi",
        refinement_steps=1,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        derivative_relative_tolerance=1e-10,
        derivative_absolute_tolerance=1e-12,
        failure_mode="status",
    )


@pytest.fixture
def binding(monkeypatch: pytest.MonkeyPatch) -> CollectiveBinding:
    device = jax.local_devices(backend="cpu")[0]
    spec = CollectiveLeaseSpec(
        group_id="mgmn-mathematical-observer",
        provider_id="cudss-mgmn",
        configuration_id="global-configuration",
        preparation_id="global-two-by-two",
        process_indices=(device.process_index,),
        device_keys=((device.process_index, device.id),),
        mpi_build_identity="cpu-observer-no-mpi",
    )

    def admit(key: CollectiveLeaseSpec) -> None:
        if key != spec:
            raise ValueError("unexpected observer lease")

    monkeypatch.setattr(mgmn, "require_collective_lease", admit)
    monkeypatch.setattr(sparse, "require_collective_lease", admit)
    return CollectiveBinding(
        jnp.asarray(1, dtype=np.int64), jnp.asarray(1, dtype=np.int64), spec
    )


@pytest.fixture
def numerical_observer(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Observe the true single-partition math; production native coverage is separate."""
    transposes: list[bool] = []

    def solve(
        lease: Array,
        ticket: Array,
        id: Array,
        values: Array,
        offsets: Array,
        columns: Array,
        rhs: Array,
        *,
        configuration: mgmn._Configuration,
        lease_key: CollectiveLeaseSpec,
        transpose: bool,
    ) -> tuple[Array, Array, Array]:
        del lease, ticket, id, configuration, lease_key
        transposes.append(transpose)
        positions = jnp.arange(values.size, dtype=np.int32)
        rows = jnp.searchsorted(offsets[1:], positions, side="right")
        matrix = jnp.zeros((2, 2), dtype=values.dtype).at[rows, columns].add(values)
        result = jnp.linalg.solve(matrix.T if transpose else matrix, rhs)
        return (
            result,
            jnp.asarray(0, dtype=np.int32),
            jnp.zeros((rhs.shape[1],), dtype=np.int32),
        )

    def tangent(*args: Array, **params: Any) -> tuple[Array]:
        return (solve(*args, **params)[0],)

    def exchange(
        lease: Array,
        ticket: Array,
        indices: Array,
        indptr: Array,
        vector: Array,
        *,
        lease_key: CollectiveLeaseSpec,
        row_starts: tuple[int, ...],
        partition_index: int,
        structure_id: str,
        transpose: bool,
    ) -> tuple[Array]:
        del lease, ticket, indptr, lease_key, partition_index, structure_id
        if row_starts != (0, 2):
            raise ValueError(
                "single-partition observer refuses distributed qualification"
            )
        if transpose:
            return (
                jnp.zeros((2, *vector.shape[1:]), dtype=vector.dtype)
                .at[indices]
                .add(vector),
            )
        return (vector[indices],)

    monkeypatch.setattr(mgmn._solve_p, "impl", solve)
    monkeypatch.setattr(mgmn._tangent_p, "impl", tangent)
    monkeypatch.setattr(sparse._exchange_p, "impl", exchange)
    return transposes


def _solve(
    storage: sparse.DistributedCSRStorage,
    binding: CollectiveBinding,
    values: Array,
    rhs: Array,
    *,
    transpose: bool = False,
) -> Array:
    value, _, _ = mgmn._solve_p.bind(
        binding.lease_id,
        binding.invocation_ticket,
        jnp.asarray(7, dtype=np.int64),
        values,
        storage.indptr,
        storage.indices,
        rhs,
        configuration=_configuration(storage),
        lease_key=binding.spec,
        transpose=transpose,
    )
    return value


def test_numeric_leaves_and_binding_are_reconstructible(
    binding: CollectiveBinding,
) -> None:
    storage = _storage()
    configuration = _configuration(storage)
    values = storage.values * 2
    restored = mgmn._restore_storage(
        configuration, values, storage.indptr, storage.indices
    )
    assert restored.values is values
    assert restored.structure_id == storage.structure_id
    assert restored.partition_id == storage.partition_id
    state = mgmn.CudssMGMNSymbolicState(
        jnp.asarray(7, dtype=np.int64),
        restored,
        binding,
        jnp.asarray(0, dtype=np.int32),
        jnp.full((1, 6, 2), np.iinfo(np.uint32).max, dtype=np.uint32),
        configuration,
    )
    leaves = jax.tree.leaves(state)
    assert any(leaf is values for leaf in leaves)
    assert any(leaf is binding.lease_id for leaf in leaves)
    assert any(leaf is binding.invocation_ticket for leaf in leaves)
    assert all(isinstance(leaf, Array) for leaf in leaves)


@pytest.mark.parametrize("nrhs", (1, 3))
def test_implicit_values_and_rhs_jvp_vjp(
    binding: CollectiveBinding,
    numerical_observer: list[bool],
    nrhs: int,
) -> None:
    storage = _storage()
    rhs = jnp.arange(2 * nrhs, dtype=np.float64).reshape(2, nrhs) + 1
    dvalues = jnp.asarray([0.2, -0.1, -0.1, 0.3], dtype=np.float64)
    drhs = jnp.ones_like(rhs) * 0.15

    def solve(values: Array, right: Array) -> Array:
        return _solve(storage, binding, values, right)

    def reference(values: Array, right: Array) -> Array:
        return jnp.linalg.solve(values.reshape(2, 2), right)

    actual = jax.jvp(solve, (storage.values, rhs), (dvalues, drhs))
    expected = jax.jvp(reference, (storage.values, rhs), (dvalues, drhs))
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    _, pullback = jax.vjp(solve, storage.values, rhs)
    _, dense_pullback = jax.vjp(reference, storage.values, rhs)
    for left, right in zip(
        pullback(jnp.ones_like(rhs)), dense_pullback(jnp.ones_like(rhs)), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    assert any(numerical_observer)


def test_zero_cotangent_enters_inverse_solve_and_second_order_is_mathematical(
    binding: CollectiveBinding,
    numerical_observer: list[bool],
) -> None:
    storage = _storage()
    rhs = jnp.asarray([[1.0], [-2.0]], dtype=np.float64)

    def solve(values: Array, right: Array) -> Array:
        return _solve(storage, binding, values, right)

    _, pullback = jax.vjp(solve, storage.values, rhs)
    before = len(numerical_observer)
    gradients = pullback(jnp.zeros_like(rhs))
    assert len(numerical_observer) > before
    for gradient in gradients:
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))

    def objective(values: Array) -> Array:
        return jnp.sum(jnp.square(solve(values, rhs)))

    def reference(values: Array) -> Array:
        return jnp.sum(jnp.square(jnp.linalg.solve(values.reshape(2, 2), rhs)))

    direction = jnp.asarray([0.2, -0.1, -0.1, 0.3], dtype=np.float64)
    actual = jax.jvp(jax.grad(objective), (storage.values,), (direction,))[1]
    expected = jax.jvp(jax.grad(reference), (storage.values,), (direction,))[1]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_schedule_expands_true_block_rhs_and_zero_tangents(
    binding: CollectiveBinding,
) -> None:
    storage = _storage()
    rhs = jnp.ones((2, 3), dtype=np.float64)

    def solve(values: Array, right: Array) -> Array:
        return _solve(storage, binding, values, right)

    primal = jax.make_jaxpr(solve)(storage.values, rhs)
    schedule = collective_schedule(primal, lease_key=binding.spec)
    # one block solve + (block exchange, correction solve) + final exchange/norm
    assert len(schedule) == 5

    def tangent(values: Array, right: Array) -> Array:
        return jax.jvp(
            solve, (values, right), (jnp.zeros_like(values), jnp.zeros_like(right))
        )[1]

    transformed = jax.make_jaxpr(tangent)(storage.values, rhs)
    assert len(collective_schedule(transformed, lease_key=binding.spec)) > len(schedule)
    with pytest.raises(ValueError, match="matrix-batch vmap"):
        jax.make_jaxpr(jax.vmap(solve))(
            jnp.stack((storage.values, storage.values)), jnp.stack((rhs, rhs))
        )
    with pytest.raises(
        (ValueError, NotImplementedError), match="remat|effect|checkpoint"
    ):
        jax.make_jaxpr(
            jax.grad(lambda values: jnp.sum(jax.checkpoint(solve)(values, rhs)))
        )(storage.values)


def _method(path: Path, digest: str) -> SparseLDLT:
    return SparseLDLT(
        execution=CudssExecutionPolicy(
            "mgmn",
            group=ExecutionGroupSpec("two-ranks", (0, 1), ((0, 0), (1, 1))),
            communication_library_path=str(path),
            communication_library_sha256=digest,
            communication_build_identity="qualified-site-mpi-shim",
        )
    )


def test_exact_communication_layer_provenance_precedes_vendor_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "libcudss_commlayer_openmpi.so"
    path.write_bytes(b"test-provenance-not-a-vendor-library")
    method = _method(path, "0" * 64)

    def forbidden(name: str) -> None:
        raise AssertionError(f"provider loaded before provenance validation: {name}")

    monkeypatch.setattr(mgmn, "load_cuda_component", forbidden)
    with pytest.raises(ValueError, match="SHA256"):
        mgmn._provenance(method)


def test_communication_build_mismatch_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"test-only-identity-material"
    path = tmp_path / "libcudss_commlayer_openmpi.so"
    path.write_bytes(content)
    method = _method(path, sha256(content).hexdigest())

    def load(name: str) -> None:
        if name != "cudss_mgmn":
            raise ValueError("wrong provider")

    def function(name: str, symbol: str) -> Any:
        del name, symbol

        def text() -> bytes:
            return b"different-site-build"

        return text

    monkeypatch.setattr(mgmn, "load_cuda_component", load)
    monkeypatch.setattr(mgmn, "cuda_component_function", function)
    with pytest.raises(ValueError, match="site component build"):
        mgmn._provenance(method)


def test_mgmn_refuses_hybrid_and_certified_host_bounds() -> None:
    method = _method(Path("/explicit/site/libcudss_commlayer_openmpi.so"), "a" * 64)
    with pytest.raises(ValueError, match="hybrid"):
        SparseLDLT(execution=method.execution, memory_mode="hybrid")
    with pytest.raises(ValueError, match="bounds|host"):
        SparseLDLT(
            execution=method.execution,
            memory_policy=CudssMemoryPolicy(host_limit_bytes=64),
        )
    with pytest.raises(ValueError, match="bounds|host"):
        SparseLDLT(
            execution=method.execution,
            memory_policy=CudssMemoryPolicy(require_certified_bounds=True),
        )
    changed = replace(_configuration(_storage()), derivative_relative_tolerance=1e-6)
    assert changed.derivative_relative_tolerance != changed.relative_tolerance


def test_transpose_value_vjp_preserves_independent_csr_orientation(
    binding: CollectiveBinding,
    numerical_observer: list[bool],
) -> None:
    storage = _storage()
    rhs = jnp.asarray([[1.0], [2.0]], dtype=np.float64)
    cotangent = jnp.asarray([[1.0], [0.0]], dtype=np.float64)

    def ordinary(values: Array) -> Array:
        return _solve(storage, binding, values, rhs)

    def transposed(values: Array) -> Array:
        return _solve(storage, binding, values, rhs, transpose=True)

    ordinary_value, ordinary_pullback = jax.vjp(ordinary, storage.values)
    transpose_value, transpose_pullback = jax.vjp(transposed, storage.values)
    np.testing.assert_allclose(ordinary_value, transpose_value, rtol=1e-12, atol=1e-12)
    (ordinary_gradient,) = ordinary_pullback(cotangent)
    (transpose_gradient,) = transpose_pullback(cotangent)
    np.testing.assert_allclose(ordinary_gradient[1], -21 / 121, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(transpose_gradient[1], 1 / 121, rtol=1e-12, atol=1e-12)
    assert any(numerical_observer)


def test_derivative_residual_failure_preserves_status_policy(
    binding: CollectiveBinding,
    numerical_observer: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject failed collective statistics, not a replacement derivative rule."""
    del numerical_observer
    storage = _storage()
    configuration = _configuration(storage)
    rhs = jnp.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    require_success: list[int] = []

    def load(component: str) -> None:
        if component != "cudss_mgmn":
            raise ValueError("unexpected observed component")

    def native_phase(
        target: str,
        results: tuple[jax.ShapeDtypeStruct, ...],
        operands: tuple[Array, ...],
        *,
        lease_key: CollectiveLeaseSpec,
        attributes: dict[str, Any],
    ) -> tuple[Array, ...]:
        del results
        if lease_key != binding.spec:
            raise ValueError("unexpected observed lease")
        if target == "phydrax_cudss_mgmn_solve_f64":
            values, right = operands[3], operands[-1]
            solved = jnp.linalg.solve(values.reshape((2, 2)), right.T).T
            return solved, jnp.asarray(0, dtype=np.int32), operands[2]
        if target != "phydrax_cudss_mgmn_norm_f64":
            raise ValueError(f"unexpected observed collective {target}")
        required = int(attributes["require_success"])
        require_success.append(required)
        if required:
            raise RuntimeError("observed all-rank derivative residual refusal")
        return (jnp.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),)

    monkeypatch.setattr(mgmn, "load_cuda_component", load)
    monkeypatch.setattr(mgmn, "collective_ffi_call", native_phase)
    operands = (
        binding.lease_id,
        binding.invocation_ticket,
        jnp.asarray(7, dtype=np.int64),
        storage.values,
        storage.indptr,
        storage.indices,
        rhs,
    )
    primal, _, status = mgmn._numerical_solve(
        *operands,
        configuration=configuration,
        lease_key=binding.spec,
        transpose=False,
        derivative=False,
    )
    np.testing.assert_array_equal(
        status,
        [int(LinearSolveStatus.RESIDUAL_TOO_LARGE), int(LinearSolveStatus.SUCCESS)],
    )
    np.testing.assert_allclose(
        primal, jnp.linalg.solve(storage.values.reshape((2, 2)), rhs)
    )
    derivative, _, derivative_status = mgmn._numerical_solve(
        *operands,
        configuration=configuration,
        lease_key=binding.spec,
        transpose=False,
        derivative=True,
    )
    np.testing.assert_array_equal(derivative_status, status)
    assert bool(jnp.all(jnp.isnan(derivative[:, 0])))
    np.testing.assert_array_equal(derivative[:, 1], jnp.zeros((2,), dtype=np.float64))
    assert require_success == [0, 0]
    with pytest.raises(RuntimeError, match="all-rank derivative residual refusal"):
        mgmn._numerical_solve(
            *operands,
            configuration=replace(configuration, failure_mode="error"),
            lease_key=binding.spec,
            transpose=False,
            derivative=True,
        )
    assert require_success == [0, 0, 1]
