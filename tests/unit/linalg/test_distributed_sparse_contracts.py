#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owned-CSR contracts and mathematical AD; CPU observers do not qualify MPI."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import DTypeLike

from phydrax import config as pxconfig
from phydrax._execution_collective_ffi import collective_schedule
from phydrax._execution_collectives import CollectiveBinding, CollectiveLeaseSpec
from phydrax.execution import default_device
from phydrax.linalg import _distributed_sparse as sparse
from phydrax.linalg._distributed import DistributedCoordinateLayout
from phydrax.linalg._properties import LinearCapabilityError
from phydrax.linalg._spaces import ArraySpace
from phydrax.linalg._sparse_contract import SparseStorage


@pytest.fixture(autouse=True)
def _cpu_numerics() -> Iterator[None]:
    with (
        default_device(jax.local_devices(backend="cpu")[0]),
        pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)),
    ):
        yield


def _storage(
    *, partition: int = 0, unequal: bool = False
) -> sparse.DistributedCSRStorage:
    layout = DistributedCoordinateLayout(3, ((0, 1), (1, 3)) if unequal else ((0, 3),))
    columns = np.asarray([0, 2, 0, 1, 1, 2], dtype=np.int32)
    values = np.asarray([4, 2, -1, 3, 5, 6], dtype=np.float64)
    pointers = np.asarray([0, 2, 4, 6], dtype=np.int32)
    start, stop = layout.owned_ranges[partition]
    begin, end = pointers[start], pointers[stop]
    return sparse.DistributedCSRStorage(
        values[begin:end],
        columns[begin:end],
        pointers[start : stop + 1] - begin,
        global_shape=(3, 3),
        row_layout=layout,
        partition_index=partition,
        structure_id="explicit-global-pattern",
        self_adjoint=False,
    )


@pytest.fixture
def binding(monkeypatch: pytest.MonkeyPatch) -> CollectiveBinding:
    device = jax.local_devices(backend="cpu")[0]
    spec = CollectiveLeaseSpec(
        group_id="csr-math-observer",
        provider_id="csr-observer",
        configuration_id="csr-math-config",
        preparation_id="explicit-global-pattern",
        process_indices=(device.process_index,),
        device_keys=((device.process_index, device.id),),
        mpi_build_identity="compiler-test-no-mpi",
    )

    def admit(key: CollectiveLeaseSpec) -> None:
        if key != spec:
            raise ValueError("unexpected test lease")

    monkeypatch.setattr(sparse, "require_collective_lease", admit)
    return CollectiveBinding(
        jnp.asarray(1, dtype=jnp.int64), jnp.asarray(1, dtype=jnp.int64), spec
    )


@pytest.fixture
def local_coordinate_observer(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Execute the exact single-partition communication map, not a fake solver.

    Production CUDA/MPI is independently qualified by coupled GPU launch tests.
    This observer lets CPU jobs exercise the actual mathematical primitive's JVP,
    transpose, forced linearization and higher composition, not a replacement AD.
    """
    observed: list[bool] = []

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
    ) -> tuple[Array, ...]:
        del lease, ticket, indptr, lease_key, partition_index, structure_id
        if row_starts != (0, 3):
            raise ValueError("CPU coordinate observer is single-partition only")
        observed.append(transpose)
        if transpose:
            return (
                jnp.zeros((3, *vector.shape[1:]), dtype=vector.dtype)
                .at[indices]
                .add(vector),
            )
        return (vector[indices],)

    def reduce(
        lease: Array,
        ticket: Array,
        value: Array,
        *,
        lease_key: CollectiveLeaseSpec,
    ) -> tuple[Array, ...]:
        del lease, ticket, lease_key
        return (value,)

    monkeypatch.setattr(sparse._exchange_p, "impl", exchange)
    monkeypatch.setattr(sparse._sum_p, "impl", reduce)
    return observed


@pytest.mark.parametrize("partition,rows,nnz", ((0, 1, 2), (1, 2, 4)))
def test_unequal_owned_rows_are_not_padded(partition: int, rows: int, nnz: int) -> None:
    storage = _storage(partition=partition, unequal=True)
    assert storage.local_row_count == rows
    assert storage.indptr.shape == (rows + 1,)
    assert storage.values.shape == storage.indices.shape == (nnz,)
    assert storage.global_shape == (3, 3)
    leaves = jax.tree.leaves(storage)
    assert len(leaves) == 3
    assert leaves[0] is storage.values
    assert leaves[1] is storage.indices
    assert leaves[2] is storage.indptr
    np.testing.assert_array_equal(
        storage.indices, [0, 2] if partition == 0 else [0, 1, 1, 2]
    )


def test_owned_partition_identity_is_explicit_and_values_remain_dynamic() -> None:
    first = _storage(partition=0, unequal=True)
    second = _storage(partition=1, unequal=True)
    assert first.structure_id == second.structure_id
    assert first.partition_id != second.partition_id
    updated = eqx.tree_at(lambda item: item.values, first, first.values * 2)
    assert updated.partition_id == first.partition_id
    assert updated.structure_id == first.structure_id
    np.testing.assert_array_equal(updated.values, first.values * 2)


def test_empty_owned_rows_preserve_exact_topology() -> None:
    storage = sparse.DistributedCSRStorage(
        np.empty((0,), dtype=np.float64),
        np.empty((0,), dtype=np.int32),
        np.zeros((3,), dtype=np.int32),
        global_shape=(3, 3),
        row_layout=DistributedCoordinateLayout(3, ((0, 1), (1, 3))),
        partition_index=1,
        structure_id="explicit-empty-pattern",
        self_adjoint=False,
    )
    assert storage.local_row_count == 2
    assert storage.nnz == 0
    np.testing.assert_array_equal(storage.indptr, [0, 0, 0])


def test_global_row_sharding_is_not_a_csr_partitioner() -> None:
    devices = jax.local_devices(backend="cpu")
    if len(devices) < 2:
        pytest.skip("global-array refusal needs two addressable CPU devices")
    mesh = jax.sharding.Mesh(np.asarray(devices[:2]), ("ranks",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("ranks"))
    values = jax.device_put(np.ones(6, dtype=np.float64), sharding)
    with pytest.raises(ValueError, match="real CSR partitioner"):
        sparse.DistributedCSRStorage(
            values,
            np.asarray([0, 2, 0, 1, 1, 2], dtype=np.int32),
            np.asarray([0, 2, 4, 6], dtype=np.int32),
            global_shape=(3, 3),
            row_layout=DistributedCoordinateLayout(3, ((0, 3),)),
            partition_index=0,
            structure_id="pattern",
            self_adjoint=False,
        )


@pytest.mark.parametrize(
    "columns,pointers,message",
    (
        ([0, 0], [0, 2], "sorted and unique"),
        ([2, 0], [0, 2], "sorted and unique"),
        ([-1, 2], [0, 2], "outside global_shape"),
        ([0, 3], [0, 2], "outside global_shape"),
        ([0, 2], [1, 2], "monotone"),
        ([0, 2], [0, 1], "monotone"),
        ([0, 2], [0, 2, 2], "without padding"),
    ),
)
def test_owned_csr_rejects_invalid_structure(
    columns: list[int],
    pointers: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sparse.DistributedCSRStorage(
            np.ones(2, dtype=np.float64),
            np.asarray(columns, dtype=np.int32),
            np.asarray(pointers, dtype=np.int32),
            global_shape=(3, 3),
            row_layout=DistributedCoordinateLayout(3, ((0, 1), (1, 3))),
            partition_index=0,
            structure_id="pattern",
            self_adjoint=True,
        )


@pytest.mark.parametrize("dtype", (np.int32, np.complex64, np.float16))
def test_owned_values_require_supported_real_dtype(dtype: DTypeLike) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        sparse.DistributedCSRStorage(
            np.ones(1, dtype=dtype),
            np.asarray([0], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
            global_shape=(1, 1),
            row_layout=DistributedCoordinateLayout(1, ((0, 1),)),
            partition_index=0,
            structure_id="pattern",
            self_adjoint=True,
        )


def test_wide_indices_are_not_silently_downcast() -> None:
    with pytest.raises(TypeError, match="int32"):
        sparse.DistributedCSRStorage(
            np.ones(1, dtype=np.float64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            global_shape=(1, 1),
            row_layout=DistributedCoordinateLayout(1, ((0, 1),)),
            partition_index=0,
            structure_id="pattern",
            self_adjoint=True,
        )


def test_topology_preparation_is_not_hidden_inside_jit() -> None:
    layout = DistributedCoordinateLayout(1, ((0, 1),))

    def prepare(columns: Array) -> Array:
        return sparse.DistributedCSRStorage(
            jnp.ones((1,), dtype=jnp.float64),
            columns,
            jnp.asarray([0, 1], dtype=jnp.int32),
            global_shape=(1, 1),
            row_layout=layout,
            partition_index=0,
            structure_id="pattern",
            self_adjoint=True,
        ).values

    with pytest.raises(ValueError, match="outside transformations"):
        jax.make_jaxpr(prepare)(jax.ShapeDtypeStruct((1,), np.int32))


def test_operator_requires_canonical_owned_spaces_and_not_local_sparse(
    binding: CollectiveBinding,
) -> None:
    storage = _storage()
    space = ArraySpace((3,), dtype=np.float64, space_id="explicit-owned-coordinates")
    operator = sparse.DistributedCSRLinearOperator(storage, space, space, binding=binding)
    assert operator.source is space
    assert operator.target is space
    assert not operator.properties.self_adjoint
    assert not operator.capabilities.materialize
    assert len(jax.tree.leaves(operator.binding)) == 2
    with pytest.raises(LinearCapabilityError, match="dense materialization"):
        operator._materialize()
    with pytest.raises(ValueError, match="owned rows"):
        sparse.DistributedCSRLinearOperator(
            storage,
            ArraySpace((4,), dtype=np.float64),
            space,
            binding=binding,
        )
    local = SparseStorage(storage.values, storage.indices, storage.indptr, shape=(3, 3))
    with pytest.raises(TypeError, match="storage and a collective binding"):
        sparse.DistributedCSRLinearOperator(local, space, space, binding=binding)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("block", (False, True))
@pytest.mark.parametrize("transpose", (False, True))
def test_sparse_action_values_vector_jvp_vjp_and_second_order(
    binding: CollectiveBinding,
    local_coordinate_observer: list[bool],
    transpose: bool,
    block: bool,
) -> None:
    storage = _storage()
    vector = jnp.asarray([2, -3, 1], dtype=jnp.float64)
    dvalues = jnp.asarray([0.2, -0.1, 0.4, 0.3, 0.1, -0.2], dtype=jnp.float64)
    dvector = jnp.asarray([-0.3, 0.2, 0.1], dtype=jnp.float64)
    if block:
        vector = jnp.stack((vector, vector * 2 - 1), axis=1)
        dvector = jnp.stack((dvector, -dvector / 2), axis=1)
    row_indices = jnp.asarray([0, 0, 1, 1, 2, 2], dtype=jnp.int32)

    def action(values: Array, argument: Array) -> Array:
        current = eqx.tree_at(lambda item: item.values, storage, values)
        return sparse.distributed_sparse_action(
            current, binding, argument, transpose=transpose
        )

    def reference(values: Array, argument: Array) -> Array:
        dense = (
            jnp.zeros((3, 3), dtype=values.dtype)
            .at[row_indices, storage.indices]
            .set(values)
        )
        return (dense.T if transpose else dense) @ argument

    for actual, expected in zip(
        jax.jvp(action, (storage.values, vector), (dvalues, dvector)),
        jax.jvp(reference, (storage.values, vector), (dvalues, dvector)),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    _, pullback = jax.vjp(action, storage.values, vector)
    _, expected_pullback = jax.vjp(reference, storage.values, vector)
    cotangent = jnp.asarray([0.5, -0.25, 0.75], dtype=jnp.float64)
    if block:
        cotangent = jnp.stack((cotangent, cotangent * 2), axis=1)
    for actual, expected in zip(
        pullback(cotangent), expected_pullback(cotangent), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def objective(
        function: Callable[[Array, Array], Array], values: Array, argument: Array
    ) -> Array:
        output = function(values, argument)
        return jnp.sum(output * output) / 2

    def gradient(values: Array, argument: Array) -> tuple[Array, Array]:
        return jax.grad(lambda a, b: objective(action, a, b), argnums=(0, 1))(
            values, argument
        )

    def reference_gradient(values: Array, argument: Array) -> tuple[Array, Array]:
        return jax.grad(lambda a, b: objective(reference, a, b), argnums=(0, 1))(
            values, argument
        )

    for actual, expected in zip(
        jax.jvp(gradient, (storage.values, vector), (dvalues, dvector))[1],
        jax.jvp(reference_gradient, (storage.values, vector), (dvalues, dvector))[1],
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert False in local_coordinate_observer
    assert True in local_coordinate_observer


def test_zero_cotangent_still_executes_communication_transpose(
    binding: CollectiveBinding,
    local_coordinate_observer: list[bool],
) -> None:
    storage = _storage()

    def action(value: Array) -> Array:
        return sparse.distributed_sparse_action(storage, binding, value)

    _, pullback = jax.vjp(action, jnp.ones((3,), dtype=jnp.float64))
    local_coordinate_observer.clear()
    (gradient,) = pullback(jnp.zeros((3,), dtype=jnp.float64))
    np.testing.assert_array_equal(gradient, np.zeros(3, dtype=np.float64))
    assert local_coordinate_observer == [True]


def test_true_rhs_blocks_use_one_exchange(
    binding: CollectiveBinding,
    local_coordinate_observer: list[bool],
) -> None:
    storage = _storage()
    space = ArraySpace((3,), dtype=np.float64)
    operator = sparse.DistributedCSRLinearOperator(storage, space, space, binding=binding)
    vectors = jnp.asarray([[1, 2], [3, -1], [-2, 4]], dtype=jnp.float64)
    dense = jnp.asarray([[4, 0, 2], [-1, 3, 0], [0, 5, 6]], dtype=jnp.float64)
    np.testing.assert_allclose(operator.mv_block(vectors), dense @ vectors)
    np.testing.assert_allclose(operator.transpose_mv_block(vectors), dense.T @ vectors)
    np.testing.assert_allclose(operator.adjoint_mv_block(vectors), dense.T @ vectors)
    assert local_coordinate_observer == [False, True, True]


def test_true_rhs_block_schedule_has_one_collective(binding: CollectiveBinding) -> None:
    storage = _storage()
    space = ArraySpace((3,), dtype=np.float64)
    operator = sparse.DistributedCSRLinearOperator(storage, space, space, binding=binding)
    program = jax.make_jaxpr(operator.mv_block)(jax.ShapeDtypeStruct((3, 4), np.float64))
    assert len(collective_schedule(program.jaxpr, lease_key=binding.spec)) == 1


def test_mathematical_sum_supports_zero_cotangents_and_composed_derivatives(
    binding: CollectiveBinding,
    local_coordinate_observer: list[bool],
) -> None:
    del local_coordinate_observer
    value = jnp.asarray([2, -3], dtype=jnp.float64)

    def objective(argument: Array) -> Array:
        total = sparse.distributed_sum(binding, argument)
        return jnp.sum(total * total) / 2

    np.testing.assert_allclose(jax.grad(objective)(value), value)
    _, derivative = jax.jvp(jax.grad(objective), (value,), (jnp.ones_like(value),))
    np.testing.assert_allclose(derivative, jnp.ones_like(value))
    _, pullback = jax.vjp(lambda x: sparse.distributed_sum(binding, x), value)
    np.testing.assert_array_equal(
        pullback(jnp.zeros_like(value))[0], jnp.zeros_like(value)
    )


def test_compiled_derivative_schedule_retains_zero_cotangent_collective(
    binding: CollectiveBinding,
) -> None:
    storage = _storage()

    def reverse(value: Array) -> Array:
        _, pullback = jax.vjp(
            lambda x: sparse.distributed_sparse_action(storage, binding, x), value
        )
        return pullback(jnp.zeros_like(value))[0]

    program = jax.make_jaxpr(reverse)(jax.ShapeDtypeStruct((3,), np.float64))
    schedule = collective_schedule(program.jaxpr, lease_key=binding.spec)
    assert len(program.effects) == 1
    assert len(schedule) == 2
    assert schedule[0] != schedule[1]


def test_asymmetric_symbolic_activity_changes_admitted_schedule(
    binding: CollectiveBinding,
) -> None:
    storage = _storage()

    def active(value: Array) -> Array:
        return sparse.distributed_sparse_action(storage, binding, value)

    def inactive(value: Array) -> Array:
        return sparse.distributed_sparse_action(
            storage, binding, jax.lax.stop_gradient(value)
        )

    def forward(function: Callable[[Array], Array], value: Array) -> Array:
        return jax.jvp(function, (value,), (jnp.zeros_like(value),))[1]

    argument = jax.ShapeDtypeStruct((3,), np.float64)
    active_program = jax.make_jaxpr(lambda x: forward(active, x))(argument)
    inactive_program = jax.make_jaxpr(lambda x: forward(inactive, x))(argument)
    active_schedule = collective_schedule(active_program.jaxpr, lease_key=binding.spec)
    inactive_schedule = collective_schedule(
        inactive_program.jaxpr, lease_key=binding.spec
    )
    # The invocation owner's schedule agreement must reject these rank programs
    # before minting a numeric ticket; it must not rely on a later MPI timeout.
    assert len(active_schedule) == 2
    assert len(inactive_schedule) == 1
    assert active_schedule != inactive_schedule


def test_vmap_and_rematerialization_are_refused(binding: CollectiveBinding) -> None:
    storage = _storage()

    def action(value: Array) -> Array:
        return sparse.distributed_sparse_action(storage, binding, value)

    with pytest.raises(ValueError, match="vmap"):
        jax.make_jaxpr(jax.vmap(action))(jax.ShapeDtypeStruct((2, 3), np.float64))
    with pytest.raises(NotImplementedError, match="Effects not supported.*checkpoint"):
        jax.make_jaxpr(jax.grad(lambda x: jnp.sum(jax.checkpoint(action)(x))))(
            jax.ShapeDtypeStruct((3,), np.float64),
        )
