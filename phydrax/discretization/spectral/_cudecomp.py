#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit cuDecomp Y/X slab redistribution beneath the native Fourier owner."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from functools import partial
from math import prod
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, ShapeDtypeStruct
from jax._src import core as jax_core, dispatch
from jax._src.interpreters import ad
from jax.core import Tracer
from jax.extend import core
from jax.interpreters import batching, mlir
from jax.sharding import Mesh, NamedSharding, PartitionSpec, use_abstract_mesh

from ..._cuda_native import cuda_component_function, load_cuda_component
from ..._execution_collective_ffi import (
    collective_call_identity,
    collective_effectful_abstract_eval,
    collective_ffi_call,
    register_collective_schedule_rule,
)
from ..._execution_collectives import (
    CollectiveBinding,
    CollectiveLeaseSpec,
    require_collective_lease,
)
from ..._fingerprint import canonical_fingerprint
from ..._jax_context import inside_jax_transformation
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


@dataclass(frozen=True, slots=True)
class _TransposeConfiguration:
    lease_spec: CollectiveLeaseSpec
    preparation_id: str
    shape: tuple[int, int, int]
    direction: int
    components: int

    @property
    def source_shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.shape
        ranks = len(self.lease_spec.device_keys)
        return (nx // ranks, ny, nz) if self.direction == 0 else (nx, ny // ranks, nz)

    @property
    def target_shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.shape
        ranks = len(self.lease_spec.device_keys)
        return (nx, ny // ranks, nz) if self.direction == 0 else (nx // ranks, ny, nz)

    @property
    def operation_id(self) -> str:
        return f"{self.preparation_id}:transpose:{self.direction}:components:{self.components}"


def _raw_transpose(
    lease_id: Array,
    ticket: Array,
    values: Array,
    *,
    configuration: _TransposeConfiguration,
) -> Array:
    suffix = "c64" if values.dtype == np.dtype(np.complex64) else "c128"
    shape = (values.shape[0], *configuration.target_shape)
    return collective_ffi_call(
        f"phydrax_cudecomp_transpose_{suffix}",
        (ShapeDtypeStruct(shape, values.dtype),),
        (lease_id, ticket, values),
        lease_key=configuration.lease_spec,
        attributes={
            "operation_id": configuration.operation_id,
            "preparation_id": configuration.preparation_id,
            "direction": configuration.direction,
        },
    )[0]


def _transpose_abstract(
    lease_id: jax_core.AbstractValue,
    ticket: jax_core.AbstractValue,
    values: jax_core.AbstractValue,
    *,
    configuration: _TransposeConfiguration,
) -> tuple[jax_core.ShapedArray, set[jax_core.Effect]]:
    if not isinstance(values, jax_core.ShapedArray):
        raise TypeError("cuDecomp input must be an ordinary complex array.")
    if values.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError("cuDecomp admits complex64 and complex128 only.")
    if (
        values.ndim != 4
        or values.shape[0] != configuration.components
        or values.shape[1:] != configuration.source_shape
    ):
        raise ValueError("cuDecomp input does not match its validated source pencil.")
    outputs, effects = collective_effectful_abstract_eval(
        lease_id,
        ticket,
        values,
        result_avals=(
            values.update(
                shape=(values.shape[0], *configuration.target_shape), weak_type=False
            ),
        ),
        lease_key=configuration.lease_spec,
    )
    return outputs[0], effects


def _transpose_lowering(
    context: mlir.LoweringRuleContext,
    *operands: mlir.ir.Value,
    configuration: _TransposeConfiguration,
) -> Any:
    return mlir.lower_fun(
        partial(_raw_transpose, configuration=configuration), multiple_results=False
    )(context, *operands)


def _with_primal_variance(value: Array, expected: jax_core.AbstractValue) -> Array:
    if not isinstance(expected, jax_core.ShapedArray):
        raise TypeError("A transpose derivative requires its ordinary primal array type.")
    missing = expected.mat.varying - jax.typeof(value).mat.varying
    if not missing:
        return value
    # Weakening invariance to the declared primal variance is an identity, not
    # a claim of replication. Its JAX transpose retains the required reduction.
    with use_abstract_mesh(expected.sharding.mesh):
        return jax.lax.pcast(value, tuple(sorted(missing)), to="varying")


def _transpose_jvp(
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array | ad.Zero, Array | ad.Zero, Array | ad.Zero],
    *,
    configuration: _TransposeConfiguration,
) -> tuple[Array, Array]:
    lease_id, ticket, values = primals
    result = _transpose_primitive.bind(
        lease_id, ticket, values, configuration=configuration
    )
    tangent = _with_primal_variance(
        ad.instantiate_zeros(tangents[2]), jax.typeof(values).to_tangent_aval()
    )
    return result, _transpose_primitive.bind(
        lease_id, ticket, tangent, configuration=configuration
    )


def _transpose_adjoint(
    cotangent: Array | ad.Zero,
    lease_id: Array | ad.UndefinedPrimal,
    ticket: Array | ad.UndefinedPrimal,
    values: Array | ad.UndefinedPrimal,
    *,
    configuration: _TransposeConfiguration,
) -> tuple[None, None, Array]:
    if isinstance(lease_id, ad.UndefinedPrimal) or isinstance(ticket, ad.UndefinedPrimal):
        raise ValueError(
            "A transpose derivative requires its dynamic collective binding."
        )
    primal_aval = (
        values.aval if isinstance(values, ad.UndefinedPrimal) else jax.typeof(values)
    )
    cotangent = _with_primal_variance(
        ad.instantiate_zeros(cotangent), primal_aval.to_ct_aval()
    )
    # This is a real permutation, including for complex arrays: its bilinear
    # transpose and complex Hilbert adjoint are both the exact inverse map.
    return (
        None,
        None,
        _transpose_primitive.bind(
            lease_id,
            ticket,
            cotangent,
            configuration=replace(configuration, direction=1 - configuration.direction),
        ),
    )


def _reject_batching(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise ValueError(
        "cuDecomp vmap is unsupported; admit an explicit trailing payload instead."
    )


def _collective_schedule(
    equation: core.JaxprEqn,
) -> tuple[tuple[CollectiveLeaseSpec, str], ...]:
    configuration = equation.params["configuration"]
    if not isinstance(configuration, _TransposeConfiguration):
        raise TypeError(
            "cuDecomp schedule requires its immutable transpose configuration."
        )
    aval = equation.invars[2].aval
    if not isinstance(aval, jax_core.ShapedArray):
        raise TypeError("cuDecomp schedule requires a numeric input array.")
    suffix = "c64" if aval.dtype == np.dtype(np.complex64) else "c128"
    return (
        (
            configuration.lease_spec,
            collective_call_identity(
                f"phydrax_cudecomp_transpose_{suffix}",
                configuration.lease_spec,
                configuration.operation_id,
            ),
        ),
    )


_transpose_primitive = core.Primitive("phydrax_cudecomp_slab_transpose")
dispatch.simple_impl(_transpose_primitive)
_transpose_primitive.def_effectful_abstract_eval(_transpose_abstract)
mlir.register_lowering(_transpose_primitive, _transpose_lowering, platform="cuda")
ad.primitive_jvps[_transpose_primitive] = _transpose_jvp
ad.primitive_transposes[_transpose_primitive] = _transpose_adjoint
ad.linearize_on_zero_tangents.add(_transpose_primitive)
batching.primitive_batchers[_transpose_primitive] = _reject_batching
register_collective_schedule_rule(_transpose_primitive, _collective_schedule)


def _expected_pencil_metadata(
    shape: tuple[int, int, int], ranks: int, rank: int
) -> tuple[int, ...]:
    nx, ny, nz = shape
    source_shape = (nz, ny, nx // ranks)
    source_lo = (0, 0, rank * (nx // ranks))
    target_shape = (nz, ny // ranks, nx)
    target_lo = (0, rank * (ny // ranks), 0)
    return (
        *source_shape,
        *source_lo,
        *(lo + size - 1 for lo, size in zip(source_lo, source_shape, strict=True)),
        2,
        1,
        0,
        *target_shape,
        *target_lo,
        *(lo + size - 1 for lo, size in zip(target_lo, target_shape, strict=True)),
        2,
        1,
        0,
    )


@dataclass(frozen=True, slots=True)
class _PreparedSlab:
    shape: tuple[int, int, int]
    preparation_id: str
    scratch_bytes: int
    endpoint_proof_id: str


def _prepare_slab(
    binding: CollectiveBinding,
    shape: tuple[int, int, int],
    dtype: np.dtype[Any],
    maximum_scratch_bytes: int,
    device_ordinal: int,
    component_build: str,
) -> _PreparedSlab:
    if maximum_scratch_bytes <= 0:
        raise MemoryError(
            "cuDecomp retained scratch exceeds its explicit per-rank budget."
        )
    if (
        inside_jax_transformation()
        or isinstance(binding.lease_id, Tracer)
        or isinstance(binding.invocation_ticket, Tracer)
    ):
        raise ValueError(
            "cuDecomp descriptor preparation is an explicit host phase, not a JIT operation."
        )
    lease = require_collective_lease(binding.spec)
    preparation_id = canonical_fingerprint(
        {
            "kind": "cudecomp-direct-slab-preparation",
            "binding": binding.spec.identity,
            "component_build": component_build,
            "shape": shape,
            "dtype": dtype.str,
            "maximum_scratch_bytes": maximum_scratch_bytes,
            "process_grid": (len(binding.spec.device_keys), 1),
            "rank_order": "row-major",
            "memory_order": (2, 1, 0),
            "transport": "MPI_A2A",
        }
    )
    return lease.prepare(
        partial(
            _prepare_slab_host,
            shape=shape,
            dtype=dtype,
            maximum_scratch_bytes=maximum_scratch_bytes,
            device_ordinal=device_ordinal,
            preparation_id=preparation_id,
        ),
        binding,
        target_name="phydrax_cudecomp_prepare",
        operation_id=preparation_id,
    )


def _prepare_slab_host(
    binding: CollectiveBinding,
    *,
    shape: tuple[int, int, int],
    dtype: np.dtype[Any],
    maximum_scratch_bytes: int,
    device_ordinal: int,
    preparation_id: str,
) -> _PreparedSlab:
    shape_array = np.asarray(shape, dtype=np.int64)
    metadata = np.empty((25,), dtype=np.int64)
    prepare = cuda_component_function("cudecomp", "phx_cuda_cudecomp_prepare")
    # Descriptor preparation is deliberately a host synchronization boundary.
    status = prepare(
        int(jax.device_get(binding.lease_id)),
        int(jax.device_get(binding.invocation_ticket)),
        binding.spec.identity.encode(),
        collective_call_identity(
            "phydrax_cudecomp_prepare", binding.spec, preparation_id
        ).encode(),
        preparation_id.encode(),
        shape_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        dtype.itemsize * 8,
        device_ordinal,
        maximum_scratch_bytes,
        metadata.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if status != 0:
        error = cuda_component_function("cudecomp", "phx_cuda_last_error")()
        message = (
            error.decode("utf-8", errors="replace")
            if isinstance(error, bytes)
            else str(error)
        )
        raise RuntimeError(f"cuDecomp descriptor preparation refused: {message}")
    ranks = len(binding.spec.device_keys)
    rank = binding.spec.process_indices.index(jax.process_index())
    expected = _expected_pencil_metadata(shape, ranks, rank)
    if tuple(metadata[1:]) != expected:
        raise ValueError(
            "cuDecomp GetPencilInfo does not match the exact source/target slab mapping."
        )
    scratch_bytes = int(metadata[0])
    if scratch_bytes <= 0 or scratch_bytes > maximum_scratch_bytes:
        raise MemoryError(
            "cuDecomp returned scratch outside the admitted per-rank bound."
        )
    proof_id = canonical_fingerprint(
        {
            "kind": "cudecomp-pencil-endpoint-proof",
            "preparation": preparation_id,
            "ordered_device_keys": binding.spec.device_keys,
            "rank_pencil_metadata": tuple(
                _expected_pencil_metadata(shape, ranks, participant)
                for participant in range(ranks)
            ),
        }
    )
    return _PreparedSlab(shape, preparation_id, scratch_bytes, proof_id)


@final
class CudecompTransposeEvidence(StrictModule, NonTrainableState):
    """Preparation evidence, separating known scratch from opaque vendor storage."""

    component_build: str = eqx.field(static=True)
    mpi_build_identity: str = eqx.field(static=True)
    binding_identity: str = eqx.field(static=True)
    endpoint_proof_ids: tuple[str, ...] = eqx.field(static=True)
    external_device_scratch_bytes_per_rank: int = eqx.field(static=True)
    maximum_external_scratch_bytes_per_rank: int = eqx.field(static=True)
    external_resident_device_bytes: None = eqx.field(static=True)
    external_host_bytes: None = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)


@final
class PreparedCudecompExecution(StrictModule, NonTrainableState):
    binding: CollectiveBinding
    canonical: _PreparedSlab = eqx.field(static=True)
    padded: _PreparedSlab = eqx.field(static=True)
    evidence: CudecompTransposeEvidence

    def apply_local(self, values: Array, /, *, inverse: bool, padded: bool) -> Array:
        slab = self.padded if padded else self.canonical
        payload = values.shape[3:]
        components = prod(payload)
        configuration = _TransposeConfiguration(
            self.binding.spec, slab.preparation_id, slab.shape, int(inverse), components
        )
        packed = jnp.moveaxis(values.reshape((*values.shape[:3], components)), -1, 0)
        result = _transpose_primitive.bind(
            self.binding.lease_id,
            self.binding.invocation_ticket,
            packed,
            configuration=configuration,
        )
        return jnp.moveaxis(result, 0, -1).reshape(
            (*configuration.target_shape, *payload)
        )


def prepare_cudecomp_execution(
    binding: CollectiveBinding,
    shape: tuple[int, ...],
    padded_shape: tuple[int, ...],
    mesh: Mesh,
    dtype: np.dtype[Any],
    maximum_scratch_bytes: int,
) -> PreparedCudecompExecution:
    """Prepare exact slab endpoints in individually admitted host lease phases."""
    if (
        inside_jax_transformation()
        or isinstance(binding.lease_id, Tracer)
        or isinstance(binding.invocation_ticket, Tracer)
    ):
        raise ValueError(
            "cuDecomp descriptor preparation is an explicit host phase, not a JIT operation."
        )
    if len(shape) != 3 or len(padded_shape) != 3:
        raise ValueError("cuDecomp admits direct three-dimensional slabs only.")
    if dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(
            "cuDecomp admits complex64 and complex128 transform buffers only."
        )
    device_keys = tuple((device.process_index, device.id) for device in mesh.devices.flat)
    if binding.spec.device_keys != device_keys:
        raise ValueError(
            "cuDecomp topology does not match the ordered collective lease devices."
        )
    ranks = len(device_keys)
    if ranks < 2 or any(
        count % ranks for grid in (shape, padded_shape) for count in grid[:2]
    ):
        raise ValueError(
            "cuDecomp requires regular divisible slabs over at least two ranks."
        )
    binding = require_collective_lease(binding.spec).binding_for(
        NamedSharding(mesh, PartitionSpec())
    )
    local = tuple(
        device
        for device in jax.local_devices()
        if (device.process_index, device.id) in device_keys
    )
    if len(local) != 1 or local[0].local_hardware_id is None:
        raise ValueError(
            "cuDecomp requires exactly one explicitly mapped local GPU per rank."
        )
    component = load_cuda_component("cudecomp")
    raw_build = cuda_component_function("cudecomp", "phx_cuda_cudecomp_build_identity")()
    if not isinstance(raw_build, bytes) or not raw_build:
        raise RuntimeError(
            "cuDecomp site component did not report its exact vendor build identity."
        )
    component_build = canonical_fingerprint(
        {"component": component.build_hash, "vendor_and_site": raw_build.decode("utf-8")}
    )
    canonical_shape = (shape[0], shape[1], shape[2])
    padded_grid = (padded_shape[0], padded_shape[1], padded_shape[2])
    canonical = _prepare_slab(
        binding,
        canonical_shape,
        dtype,
        maximum_scratch_bytes,
        local[0].local_hardware_id,
        component_build,
    )
    padded = (
        canonical
        if padded_grid == canonical_shape
        else _prepare_slab(
            binding,
            padded_grid,
            dtype,
            maximum_scratch_bytes - canonical.scratch_bytes,
            local[0].local_hardware_id,
            component_build,
        )
    )
    scratch = canonical.scratch_bytes + (
        0 if padded is canonical else padded.scratch_bytes
    )
    proof_ids = (canonical.endpoint_proof_id, padded.endpoint_proof_id)
    execution_id = canonical_fingerprint(
        {
            "kind": "cudecomp-slab-execution",
            "component_build": component_build,
            "binding": binding.spec.identity,
            "endpoint_proofs": proof_ids,
            "external_device_scratch_bytes_per_rank": scratch,
            "maximum_external_scratch_bytes_per_rank": maximum_scratch_bytes,
            "external_resident_device_bytes": None,
            "external_host_bytes": None,
            "payload_packing": "explicit-leading-component-serial-3d",
            "derivative": "linear-permutation-inverse-transpose",
            "manual_axis_variance": "preserve-primal-sharding-and-tangent-cotangent-MAT",
        }
    )
    evidence = CudecompTransposeEvidence(
        component_build,
        binding.spec.mpi_build_identity,
        binding.spec.identity,
        proof_ids,
        scratch,
        maximum_scratch_bytes,
        None,
        None,
        execution_id,
    )
    return PreparedCudecompExecution(binding, canonical, padded, evidence)
