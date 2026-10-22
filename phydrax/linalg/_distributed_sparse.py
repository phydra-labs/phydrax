#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Unpadded rank-local CSR and mathematical sparse coordinate communication.

Construction prepares and validates topology on the host; numerical values and
vectors remain dynamic device leaves. Global JAX arrays are not a partitioner.
The communication primitive is linear, with its actual scatter-add transpose;
CSR multiplication and owned-value derivatives use ordinary JAX arithmetic.
All ranks must execute the same transformed program. Rank-asymmetric symbolic
AD activity is not supported (materialized numerical zeros are supported).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, final, NoReturn

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, ShapeDtypeStruct
from jax._src import core as jax_core, dispatch
from jax._src.interpreters import ad as _ad_internal
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from jax.typing import ArrayLike

from .._cuda_native import load_cuda_component
from .._execution_collective_ffi import (
    collective_call_identity,
    collective_effectful_abstract_eval,
    collective_ffi_call,
    register_collective_schedule_rule,
)
from .._execution_collectives import (
    CollectiveBinding,
    CollectiveLeaseSpec,
    require_collective_lease,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._distributed import DistributedCoordinateLayout
from ._operators import AbstractLinearOperator
from ._pairings import EuclideanPairing
from ._properties import LinearCapabilityError, OperatorCapabilities, OperatorProperties
from ._spaces import ArraySpace


@final
class DistributedCSRStorage(StrictModule):
    """Exactly one owned row interval, with sorted unique global int32 columns.

    ``structure_id`` identifies the global pattern across all partitions. Its
    producer must change that identity when the pattern changes. ``self_adjoint``
    is an explicit scientific certification, not a numerical inference.
    """

    values: Array
    indices: Array
    indptr: Array
    global_shape: tuple[int, int] = eqx.field(static=True)
    row_layout: DistributedCoordinateLayout = eqx.field(static=True)
    partition_index: int = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    self_adjoint: bool = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        indices: ArrayLike,
        indptr: ArrayLike,
        /,
        *,
        global_shape: tuple[int, int],
        row_layout: DistributedCoordinateLayout,
        partition_index: int,
        structure_id: str,
        self_adjoint: bool,
    ) -> None:
        if not isinstance(row_layout, DistributedCoordinateLayout):
            raise TypeError("row_layout must be a DistributedCoordinateLayout")
        shape = tuple(global_shape)
        if (
            len(shape) != 2
            or any(type(size) is not int for size in shape)
            or shape != (row_layout.global_size, row_layout.global_size)
        ):
            raise ValueError(
                "distributed CSR requires an explicit square global shape matching row_layout"
            )
        if (
            type(partition_index) is not int
            or not 0 <= partition_index < row_layout.partition_count
        ):
            raise ValueError("partition_index must identify one owned row interval")
        if (
            not isinstance(structure_id, str)
            or not structure_id.strip()
            or "\x00" in structure_id
        ):
            raise ValueError(
                "structure_id must be a nonempty NUL-free global pattern identity"
            )
        if type(self_adjoint) is not bool:
            raise TypeError("self_adjoint must be an explicit bool certification")
        values_ = jnp.asarray(values)
        indices_ = jnp.asarray(indices)
        indptr_ = jnp.asarray(indptr)
        start, stop = row_layout.owned_ranges[partition_index]
        if values_.ndim != 1 or indices_.shape != values_.shape:
            raise ValueError(
                "distributed CSR values and indices must be equal one-dimensional arrays"
            )
        if indptr_.shape != (stop - start + 1,):
            raise ValueError(
                "indptr must describe exactly the owned rows, without padding"
            )
        if values_.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("distributed CSR values must use float32 or float64")
        if indices_.dtype != np.dtype(np.int32) or indptr_.dtype != np.dtype(np.int32):
            raise TypeError("distributed CSR global columns and indptr must use int32")
        if shape[0] > np.iinfo(np.int32).max or values_.size > np.iinfo(np.int32).max:
            raise ValueError("distributed CSR dimensions and local nnz must fit int32")
        for array in (values_, indices_, indptr_):
            if not isinstance(array, jax_core.Tracer) and (
                not array.is_fully_addressable or len(array.devices()) != 1
            ):
                raise ValueError(
                    "global/sharded arrays require a real CSR partitioner; pass rank-local arrays"
                )
        if isinstance(indices_, jax_core.Tracer) or isinstance(indptr_, jax_core.Tracer):
            raise ValueError("prepare distributed CSR topology outside transformations")
        columns = np.asarray(indices_)
        pointers = np.asarray(indptr_)
        if (
            pointers[0] != 0
            or pointers[-1] != values_.size
            or np.any(pointers[1:] < pointers[:-1])
        ):
            raise ValueError("CSR indptr must be monotone from zero through local nnz")
        if np.any(columns < 0) or np.any(columns >= shape[1]):
            raise ValueError("CSR global column is outside global_shape")
        for row in range(stop - start):
            row_columns = columns[pointers[row] : pointers[row + 1]]
            if np.any(row_columns[1:] <= row_columns[:-1]):
                raise ValueError(
                    "CSR columns must be sorted and unique within each owned row"
                )
        self.values = values_
        self.indices = indices_
        self.indptr = indptr_
        self.global_shape = (shape[0], shape[1])
        self.row_layout = row_layout
        self.partition_index = partition_index
        self.structure_id = structure_id
        self.partition_id = canonical_fingerprint(
            {
                "kind": "distributed-csr-partition",
                "structure_id": structure_id,
                "layout_id": row_layout.layout_id,
                "partition_index": partition_index,
            }
        )
        self.self_adjoint = self_adjoint

    @property
    def local_row_count(self) -> int:
        start, stop = self.row_layout.owned_ranges[self.partition_index]
        return stop - start

    @property
    def nnz(self) -> int:
        return self.values.shape[0]


def _exchange_impl(
    lease_id: Array,
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
    load_cuda_component("cudss_mgmn")
    count = (
        row_starts[partition_index + 1] - row_starts[partition_index]
        if transpose
        else indices.shape[0]
    )
    suffix = "f32" if vector.dtype == np.dtype(np.float32) else "f64"
    return collective_ffi_call(
        f"phydrax_cudss_mgmn_exchange_{suffix}",
        (ShapeDtypeStruct((count, *vector.shape[1:]), vector.dtype),),
        (lease_id, ticket, indices, indptr, vector),
        lease_key=lease_key,
        attributes={
            "row_starts": np.asarray(row_starts, dtype=np.int64),
            "partition_index": partition_index,
            "structure_id": structure_id,
            "operation_id": "csr-coordinate-scatter"
            if transpose
            else "csr-coordinate-gather",
            "transpose": int(transpose),
        },
    )


def _exchange_abstract(
    *avals: jax_core.AbstractValue,
    lease_key: CollectiveLeaseSpec,
    row_starts: tuple[int, ...],
    partition_index: int,
    structure_id: str,
    transpose: bool,
) -> tuple[tuple[jax_core.ShapedArray, ...], set[Any]]:
    del structure_id
    vector = avals[-1]
    indices = avals[-3]
    if not isinstance(vector, jax_core.ShapedArray) or not isinstance(
        indices, jax_core.ShapedArray
    ):
        raise TypeError("coordinate exchange requires array operands")
    count = (
        row_starts[partition_index + 1] - row_starts[partition_index]
        if transpose
        else indices.shape[0]
    )
    return collective_effectful_abstract_eval(
        *avals,
        result_avals=(jax_core.ShapedArray((count, *vector.shape[1:]), vector.dtype),),
        lease_key=lease_key,
    )


def _communication_jvp(
    primals: tuple[Array, ...],
    tangents: tuple[Array | ad.Zero, ...],
    *,
    primitive: core.Primitive,
    **params: Any,
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    primal = tuple(primitive.bind(*primals, **params))
    # Always materialize numerical zeros: all ranks execute the same tangent call.
    tangent = ad.instantiate_zeros(tangents[-1])
    tangent_out = tuple(primitive.bind(*primals[:-1], tangent, **params))
    return primal, tangent_out


def _communication_linearize(
    is_vjp: bool,
    tangent_nzs: Sequence[bool],
    *primals: Array,
    primitive: core.Primitive,
    **params: Any,
) -> tuple[
    tuple[Array, ...],
    tuple[bool, ...],
    tuple[Array, ...],
    None,
    Callable[..., tuple[Array, ...]],
]:
    del is_vjp, tangent_nzs
    primal = tuple(primitive.bind(*primals, **params))

    def linearized(
        residuals: tuple[Array, ...],
        structured: None,
        *tangents: Array | ad.Zero,
    ) -> tuple[Array, ...]:
        del structured
        tangent = ad.instantiate_zeros(tangents[-1])
        return tuple(primitive.bind(*residuals, tangent, **params))

    # A local symbolic zero can receive nonzero remote tangents. Never mark the
    # communication output symbolically zero or prune its effectful transpose.
    return primal, (True,), tuple(primals[:-1]), None, linearized


def _exchange_transpose(
    cotangents: tuple[Array | ad.Zero, ...],
    lease_id: Array,
    ticket: Array,
    indices: Array,
    indptr: Array,
    vector: Array | ad.UndefinedPrimal,
    **params: Any,
) -> tuple[None, None, None, None, Array | None]:
    cotangent = ad.instantiate_zeros(cotangents[0])
    reverse = dict(params, transpose=not params["transpose"])
    (result,) = _exchange_p.bind(lease_id, ticket, indices, indptr, cotangent, **reverse)
    return None, None, None, None, result if ad.is_undefined_primal(vector) else None


def _exchange_batch(*args: Any, **params: Any) -> NoReturn:
    del args, params
    raise ValueError("distributed CSR coordinate exchange does not support vmap")


_exchange_p = core.Primitive("phydrax_distributed_csr_exchange")
_exchange_p.multiple_results = True
dispatch.simple_impl(_exchange_p)
_exchange_p.def_effectful_abstract_eval(_exchange_abstract)
ad.primitive_jvps[_exchange_p] = partial(_communication_jvp, primitive=_exchange_p)
_ad_internal.primitive_linearizations[_exchange_p] = partial(
    _communication_linearize, primitive=_exchange_p
)
ad.primitive_transposes[_exchange_p] = _exchange_transpose
_ad_internal.linearize_on_zero_tangents.add(_exchange_p)
batching.fancy_primitive_batchers[_exchange_p] = _exchange_batch
mlir.register_lowering(
    _exchange_p, mlir.lower_fun(_exchange_impl, multiple_results=True), platform="cuda"
)


def _exchange_schedule(
    equation: core.JaxprEqn,
) -> tuple[tuple[CollectiveLeaseSpec, str], ...]:
    spec = equation.params["lease_key"]
    value = equation.invars[-1].aval
    if not isinstance(value, jax_core.ShapedArray):
        raise TypeError("coordinate exchange schedule requires an array operand")
    suffix = "f32" if value.dtype == np.dtype(np.float32) else "f64"
    operation = (
        "csr-coordinate-scatter"
        if equation.params["transpose"]
        else "csr-coordinate-gather"
    )
    return (
        (
            spec,
            collective_call_identity(
                f"phydrax_cudss_mgmn_exchange_{suffix}", spec, operation
            ),
        ),
    )


register_collective_schedule_rule(_exchange_p, _exchange_schedule)


def _reduction_ffi(
    lease_id: Array,
    ticket: Array,
    value: Array,
    *,
    lease_key: CollectiveLeaseSpec,
    operation: int,
) -> tuple[Array, ...]:
    load_cuda_component("cudss_mgmn")
    suffix = {
        np.dtype(np.float32): "f32",
        np.dtype(np.float64): "f64",
        np.dtype(np.int32): "i32",
    }[value.dtype]
    return collective_ffi_call(
        f"phydrax_cudss_mgmn_reduce_{suffix}",
        (ShapeDtypeStruct(value.shape, value.dtype),),
        (lease_id, ticket, value),
        lease_key=lease_key,
        attributes={
            "operation": operation,
            "operation_id": "owned-sum" if operation == 0 else "owned-status-max",
        },
    )


def _sum_impl(
    lease_id: Array,
    ticket: Array,
    value: Array,
    *,
    lease_key: CollectiveLeaseSpec,
) -> tuple[Array, ...]:
    return _reduction_ffi(lease_id, ticket, value, lease_key=lease_key, operation=0)


def _sum_abstract(
    *avals: jax_core.AbstractValue,
    lease_key: CollectiveLeaseSpec,
) -> tuple[tuple[jax_core.ShapedArray, ...], set[Any]]:
    value = avals[-1]
    if not isinstance(value, jax_core.ShapedArray):
        raise TypeError("distributed sum requires an array operand")
    return collective_effectful_abstract_eval(
        *avals, result_avals=(value,), lease_key=lease_key
    )


def _sum_transpose(
    cotangents: tuple[Array | ad.Zero, ...],
    lease_id: Array,
    ticket: Array,
    value: Array | ad.UndefinedPrimal,
    *,
    lease_key: CollectiveLeaseSpec,
) -> tuple[None, None, Array | None]:
    cotangent = ad.instantiate_zeros(cotangents[0])
    (result,) = _sum_p.bind(lease_id, ticket, cotangent, lease_key=lease_key)
    return None, None, result if ad.is_undefined_primal(value) else None


def _sum_schedule(equation: core.JaxprEqn) -> tuple[tuple[CollectiveLeaseSpec, str], ...]:
    spec = equation.params["lease_key"]
    value = equation.invars[-1].aval
    if not isinstance(value, jax_core.ShapedArray):
        raise TypeError("distributed sum schedule requires an array operand")
    suffix = "f32" if value.dtype == np.dtype(np.float32) else "f64"
    return (
        (
            spec,
            collective_call_identity(
                f"phydrax_cudss_mgmn_reduce_{suffix}", spec, "owned-sum"
            ),
        ),
    )


_sum_p = core.Primitive("phydrax_distributed_sum")
_sum_p.multiple_results = True
dispatch.simple_impl(_sum_p)
_sum_p.def_effectful_abstract_eval(_sum_abstract)
ad.primitive_jvps[_sum_p] = partial(_communication_jvp, primitive=_sum_p)
_ad_internal.primitive_linearizations[_sum_p] = partial(
    _communication_linearize, primitive=_sum_p
)
ad.primitive_transposes[_sum_p] = _sum_transpose
_ad_internal.linearize_on_zero_tangents.add(_sum_p)
batching.fancy_primitive_batchers[_sum_p] = _exchange_batch
mlir.register_lowering(
    _sum_p, mlir.lower_fun(_sum_impl, multiple_results=True), platform="cuda"
)
register_collective_schedule_rule(_sum_p, _sum_schedule)


def distributed_sum(binding: CollectiveBinding, value: ArrayLike) -> Array:
    """Sum owned scalar/RHS-vector contributions; the transpose is allreduce sum."""
    if not isinstance(binding, CollectiveBinding):
        raise TypeError("distributed sum requires a collective binding")
    require_collective_lease(binding.spec)
    array = jnp.asarray(value)
    if array.ndim > 1 or array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(
            "distributed sum requires a real float32/float64 scalar or RHS vector"
        )
    (result,) = _sum_p.bind(
        binding.lease_id, binding.invocation_ticket, array, lease_key=binding.spec
    )
    return result


def distributed_max(binding: CollectiveBinding, value: ArrayLike) -> Array:
    """Reach one all-rank int32 status authority without numerical AD."""
    if not isinstance(binding, CollectiveBinding):
        raise TypeError("distributed status maximum requires a collective binding")
    require_collective_lease(binding.spec)
    array = jnp.asarray(value)
    if array.ndim > 1 or array.dtype != np.dtype(np.int32):
        raise ValueError(
            "distributed status maximum requires an int32 scalar or RHS vector"
        )
    (result,) = _reduction_ffi(
        binding.lease_id,
        binding.invocation_ticket,
        array,
        lease_key=binding.spec,
        operation=1,
    )
    return result


def distributed_sparse_action(
    storage: DistributedCSRStorage,
    binding: CollectiveBinding,
    vector: ArrayLike,
    *,
    transpose: bool = False,
) -> Array:
    """Apply owned CSR to a vector or true RHS block in one coordinate exchange."""
    if not isinstance(storage, DistributedCSRStorage) or not isinstance(
        binding, CollectiveBinding
    ):
        raise TypeError(
            "distributed sparse action requires owned CSR and a collective binding"
        )
    if type(transpose) is not bool:
        raise TypeError("transpose must be a bool")
    require_collective_lease(binding.spec)
    if len(binding.spec.process_indices) != storage.row_layout.partition_count:
        raise ValueError("CSR partitions do not match the collective process membership")
    value = jnp.asarray(vector)
    if value.ndim not in (1, 2) or value.shape[0] != storage.local_row_count:
        raise ValueError(
            "vector/block must have owned unpadded rows and at most one true RHS axis"
        )
    if value.dtype != storage.values.dtype:
        raise TypeError("vector must have the CSR value dtype")
    if value.ndim == 2 and value.shape[1] == 0:
        return jnp.zeros_like(value)
    coefficients = storage.values.reshape((storage.nnz, *(1 for _ in value.shape[1:])))
    rows = jnp.searchsorted(
        storage.indptr[1:], jnp.arange(storage.nnz, dtype=jnp.int32), side="right"
    )
    starts = tuple(start for start, _ in storage.row_layout.owned_ranges) + (
        storage.global_shape[0],
    )
    operands = (
        binding.lease_id,
        binding.invocation_ticket,
        storage.indices,
        storage.indptr,
    )
    params = dict(
        lease_key=binding.spec,
        row_starts=starts,
        partition_index=storage.partition_index,
        structure_id=storage.structure_id,
        transpose=transpose,
    )
    if transpose:
        (result,) = _exchange_p.bind(*operands, coefficients * value[rows], **params)
        return result
    (coordinates,) = _exchange_p.bind(*operands, value, **params)
    return jnp.zeros_like(value).at[rows].add(coefficients * coordinates)


@final
class DistributedCSRLinearOperator(AbstractLinearOperator):
    """An owned CSR operator; deliberately not an AbstractSparseLinearOperator.

    Local SparseStorage and its factorization consumers have a different
    ownership contract. Distributed dispatch must inspect this exact type.
    """

    storage: DistributedCSRStorage
    binding: CollectiveBinding

    def __init__(
        self,
        storage: DistributedCSRStorage,
        source: ArraySpace,
        target: ArraySpace,
        *,
        binding: CollectiveBinding,
    ) -> None:
        if not isinstance(storage, DistributedCSRStorage) or not isinstance(
            binding, CollectiveBinding
        ):
            raise TypeError(
                "distributed CSR operator requires storage and a collective binding"
            )
        for space in (source, target):
            if not isinstance(space, ArraySpace):
                raise TypeError(
                    "distributed CSR requires canonical rank-local ArraySpace values"
                )
            if (
                space.shape != (storage.local_row_count,)
                or space.dtype != storage.values.dtype
            ):
                raise ValueError(
                    "source and target must describe exactly the owned rows and CSR dtype"
                )
            if not isinstance(space.pairing, EuclideanPairing):
                raise ValueError(
                    "distributed CSR currently requires Euclidean coordinate pairings"
                )
        if not source.compatible(target):
            raise ValueError(
                "source and target must share the canonical owned coordinate identity"
            )
        if len(binding.spec.process_indices) != storage.row_layout.partition_count:
            raise ValueError(
                "CSR partitions do not match the collective process membership"
            )
        self.storage = storage
        self.binding = binding
        self.source = source
        self.target = target
        self.properties = OperatorProperties(
            self_adjoint=storage.self_adjoint,
            evidence={"self_adjoint": "asserted"} if storage.self_adjoint else {},
        )
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=False
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "distributed-csr",
                "partition_id": storage.partition_id,
                "source": source.space_id,
                "target": target.space_id,
                "configuration_id": binding.spec.configuration_id,
            }
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        return distributed_sparse_action(self.storage, self.binding, vector)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        return distributed_sparse_action(
            self.storage, self.binding, vector, transpose=True
        )

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        return self.transpose_mv(vector)

    def _block_action(self, vectors: ArrayLike, *, transpose: bool) -> Array:
        values = jnp.asarray(vectors)
        if values.ndim != 2 or values.shape[0] != self.storage.local_row_count:
            raise ValueError(
                "distributed block coordinates must have owned rows and one RHS axis"
            )
        return distributed_sparse_action(
            self.storage, self.binding, values, transpose=transpose
        )

    def mv_block(self, vectors: ArrayLike, /) -> Array:
        return self._block_action(vectors, transpose=False)

    def transpose_mv_block(self, vectors: ArrayLike, /) -> Array:
        return self._block_action(vectors, transpose=True)

    def adjoint_mv_block(self, vectors: ArrayLike, /) -> Array:
        return self.transpose_mv_block(vectors)

    def _materialize(self, /) -> NoReturn:
        raise LinearCapabilityError(
            "distributed CSR does not admit local or global dense materialization"
        )


__all__ = [
    "DistributedCSRStorage",
    "DistributedCSRLinearOperator",
    "distributed_sparse_action",
    "distributed_sum",
    "distributed_max",
]
