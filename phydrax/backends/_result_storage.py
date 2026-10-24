# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Public result storage reserved ahead of private native work.

XLA buffer assignment may place any temporary value in a live-out allocation
whose live range does not interfere with it. Results produced at the end of a
long private computation therefore share their allocations with scratch
values that die earlier, and a caller retaining an earlier result forces every
graph node touching that scratch to be rebound on the next execution. A
kernel or fusion boundary alone does not change this: it privatizes only the
values inside the boundary.

`evaluate_with_result_storage` makes every published result allocation live
for the whole private computation instead. On an admitted CUDA target the
reservation primitive defines the result storage with one Mosaic GPU kernel
bound to the gated dynamic operands and orders it before every
operand-dependent value of the private computation with
``stablehlo.optimization_barrier``. XLA removes such barriers only after
scheduling, so the schedule seen by buffer assignment defines the storage
before any operand-dependent work. The publication primitive then copies the
final values into that storage in place with one Mosaic GPU kernel whose
operands alias its outputs. Every private temporary interferes with the
storage and cannot be assigned to it. The storage is created inside each
execution, so results never alias caller buffers or earlier results.

The reservation kernel never reads its operands; it binds every input any of
its results depends on, so each storage definition is an instruction of that
request's own inputs. Its kernel symbol carries the canonical SHA-256 of the
request: the traced computation (``_identity.execution_metadata_payload`` with
dynamic captures, the content of concrete captures, and the positional map of
program inputs and captures onto deduplicated inputs), the original
publication slots it still owns, and the original input roles it binds with
the axis every applied batching layer maps on each of them. Dead-code
elimination prunes slots and inputs without renumbering them. That symbol is
part of the serialized Mosaic module in the custom call's backend
configuration, which XLA compares when eliminating common subexpressions.
Two reservations therefore merge only when they own the same results of the
same computation over the same input roles read in the same batched layout,
whose values are identical; every other request, such as successive,
alternative, differently projected or differently batched queries of one
compiled step, owns storage that interferes with its own private work.
Computations the canonical compiler identity does not admit fail closed when
traced.

A leaf is published when it is a computed, operand-dependent, nonempty array
of a supported dtype whose every possible dynamic dependency also has a
supported (bindable) dtype; each distinct computed variable is published once
however often it occurs in the result tree. Forwarded operands, constants,
values independent of every dynamic operand, and leaves that may depend on an
input of another dtype (complex, extended or narrow types) are returned
exactly as the computation returns them, with ordinary allocation. Both
primitives prune unused results and their private inputs during dead-code
elimination, so caller projections keep their meaning. On any other target,
or when a result exceeds the native bounds, the storage is dead and
publication is the identity; numerical results are identical everywhere.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, cast, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import Array, lax
from jax._src import core as jax_core
from jax._src.core import ShapedArray
from jax._src.interpreters import partial_eval as pe
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from ._mosaic import lower_mosaic_or_reference


_T = TypeVar("_T")

# One warpgroup per CTA; one thread per published element.
_TILE = 128
# Lane indices ``block * _TILE + thread`` of every launched thread fit int32.
_ELEMENT_LIMIT = (1 << 31) - _TILE
_PUBLISHED_DTYPES = frozenset(
    np.dtype(dtype)
    for dtype in (
        jnp.bool_,
        jnp.int8,
        jnp.int16,
        jnp.int32,
        jnp.int64,
        jnp.uint8,
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        jnp.float16,
        jnp.bfloat16,
        jnp.float32,
        jnp.float64,
    )
)


def _publishable(aval: object, /) -> bool:
    """Nonempty array of a published dtype within the int32 lane range."""
    if not isinstance(aval, ShapedArray):
        return False
    size = math.prod(aval.shape)
    return aval.dtype in _PUBLISHED_DTYPES and 0 < size <= _ELEMENT_LIMIT


def _bindable(aval: object, /) -> bool:
    """Array type a reservation kernel binds without reading, of any extent."""
    return isinstance(aval, ShapedArray) and aval.dtype in _PUBLISHED_DTYPES


def _native_storage(avals: Sequence[ShapedArray], /) -> bool:
    """Native reservation and publication bounds for one storage set."""
    return bool(avals) and all(_publishable(aval) for aval in avals)


def _structs(values: Sequence[Any], /) -> tuple[jax.ShapeDtypeStruct, ...]:
    return tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values)


# ---------------------------------------------------------------------------
# Native kernels. Built per lowering inside its MLIR context.
# ---------------------------------------------------------------------------


def _reservation_kernel(
    operands: tuple[jax.ShapeDtypeStruct, ...],
    storage: tuple[jax.ShapeDtypeStruct, ...],
    identity: str,
    /,
) -> Callable[..., tuple[Array, ...]]:
    """One kernel defining the storage allocations of one reservation request.

    It reads no operand and writes nothing: the operands bind the reservation
    to every input its results depend on, the request identity names its
    kernel symbol inside the serialized module that forms the custom call's
    backend configuration, and publication overwrites every storage element
    before any value of the storage is observable.
    """

    from jax.experimental.mosaic import gpu as mgpu

    def body(_: object, *refs: object) -> None:
        del refs

    return mgpu.as_gpu_kernel(
        body,
        grid=(1, 1, 1),
        block=(_TILE, 1, 1),
        in_shape=operands,
        out_shape=storage,
        smem_scratch_shape=(),
        kernel_name=f"phydrax_result_reservation_{identity}",
    )


def _publication_kernel(
    values: tuple[jax.ShapeDtypeStruct, ...], /
) -> Callable[..., tuple[Array, ...]]:
    """One kernel copying ``values`` into aliased storage, one thread per element."""

    from jax.experimental.mosaic import gpu as mgpu
    from jax.experimental.mosaic.gpu import utils as mgpu_utils
    from jax.extend.mlir import ir
    from jax.extend.mlir.dialects import arith, memref

    count = len(values)
    lanes = max(math.prod(value.shape) for value in values)

    def body(_: object, *refs: ir.Value) -> None:
        # Inputs, then in-place storage; the trailing shared-memory tree is empty.
        sources, targets = refs[:count], refs[count : 2 * count]
        i32 = ir.IntegerType.get_signless(32)
        index = ir.IndexType.get()
        lane = arith.addi(
            arith.muli(mgpu_utils.block_idx(), mgpu.c(_TILE, i32)), mgpu.thread_idx()
        )
        for source, target, value in zip(sources, targets, values, strict=True):
            size = math.prod(value.shape)
            with mgpu.when(arith.cmpi(arith.CmpIPredicate.slt, lane, mgpu.c(size, i32))):
                remainder = arith.index_cast(index, lane)
                indices: list[ir.Value] = []
                for extent in reversed(value.shape):
                    width = mgpu.c(extent, index)
                    indices.append(arith.remui(remainder, width))
                    remainder = arith.divui(remainder, width)
                indices.reverse()
                memref.store(memref.load(source, indices), target, indices)

    return mgpu.as_gpu_kernel(
        body,
        grid=(-(-lanes // _TILE), 1, 1),
        block=(_TILE, 1, 1),
        in_shape=values,
        out_shape=(),
        smem_scratch_shape=(),
        inout_shape=values,
        kernel_name="phydrax_result_publication",
    )


# ---------------------------------------------------------------------------
# Reservation: ``(*storage, *operands) = reserve(*operands)``.
# ---------------------------------------------------------------------------


# Parameters: ``storage`` (non-weak avals), ``dependencies`` (per storage, the
# current operand positions it depends on), ``domain`` (computation digest),
# ``slots`` (per storage, its original publication slot in the computation),
# ``origins`` (per operand, its original deduplicated input position) and
# ``batching`` (applied batching layers, innermost first: the axis size and,
# per current operand, the axis that layer maps or ``None``; each axis indexes
# the operand without the dimensions of later layers). Batching forwards the
# physical operands unmoved, so ``batching`` is their complete interpretation;
# storage is mapped at axis 0 of a layer exactly when a dependency is mapped
# by it. Dead-code elimination prunes storage, operands and per-operand axes
# but never renumbers ``slots`` or ``origins``, so a pruned reservation still
# names exactly which results of which computation over which input roles and
# layouts it owns.


def _reserve_reference(
    *operands: Array, storage: tuple[ShapedArray, ...], **params: Any
) -> list[Array]:
    """Dead zero storage and forwarded operands, without ordering."""
    del params
    return [jnp.zeros(aval.shape, aval.dtype) for aval in storage] + list(operands)


def _reservation_identity(
    domain: str,
    slots: tuple[int, ...],
    origins: tuple[int, ...],
    batching: tuple[tuple[int, tuple[int | None, ...]], ...],
    bound_positions: Sequence[int],
    /,
) -> str:
    """Canonical name of one reservation request; never object identities."""
    from .._fingerprint import canonical_json

    payload = {
        "kind": "result-storage-reservation",
        "domain": domain,
        "slots": list(slots),
        "layer_sizes": [size for size, _ in batching],
        "bound_inputs": [
            {
                "origin": origins[position],
                "mapped_axes": [axes[position] for _, axes in batching],
            }
            for position in bound_positions
        ],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _reserve_native(
    *operands: Array,
    storage: tuple[ShapedArray, ...],
    dependencies: tuple[tuple[int, ...], ...],
    domain: str,
    slots: tuple[int, ...],
    origins: tuple[int, ...],
    batching: tuple[tuple[int, tuple[int | None, ...]], ...],
) -> list[Array]:
    if not _native_storage(storage):
        return _reserve_reference(*operands, storage=storage)
    # Every operand any reserved result depends on is bound. Evaluation admits
    # a result only when all its dependencies have a bindable type, and the
    # kernel never indexes them, so their extents are unrestricted.
    bound_positions = sorted({position for group in dependencies for position in group})
    bound = [operands[position] for position in bound_positions]
    identity = _reservation_identity(domain, slots, origins, batching, bound_positions)
    kernel = _reservation_kernel(_structs(bound), _structs(storage), identity)
    reserved = list(kernel(*bound))
    # Every operand-dependent value is ordered after the storage definition.
    ordered, forwarded = lax.optimization_barrier((reserved, list(operands)))
    return [*ordered, *forwarded]


def _reserve_abstract(
    *operands: ShapedArray,
    storage: tuple[ShapedArray, ...],
    dependencies: tuple[tuple[int, ...], ...],
    domain: str,
    slots: tuple[int, ...],
    origins: tuple[int, ...],
    batching: tuple[tuple[int, tuple[int | None, ...]], ...],
) -> list[ShapedArray]:
    if len(dependencies) != len(storage) or any(
        not 0 <= position < len(operands) for group in dependencies for position in group
    ):
        raise ValueError("Result storage dependencies must name reserved operands.")
    if len(domain) != 64 or any(digit not in "0123456789abcdef" for digit in domain):
        raise ValueError("Result storage domain must be a SHA-256 hexadecimal digest.")
    if len(slots) != len(storage) or len(set(slots)) != len(slots):
        raise ValueError(
            "Result storage needs one distinct publication slot per storage."
        )
    if len(origins) != len(operands) or len(set(origins)) != len(origins):
        raise ValueError("Result storage needs one distinct input origin per operand.")
    if any(
        not _bindable(operands[position]) for group in dependencies for position in group
    ):
        raise ValueError("Result storage dependencies must have bindable types.")
    for layer, (size, axes) in enumerate(batching):
        if size < 0 or len(axes) != len(operands):
            raise ValueError("Result storage batching needs one axis per operand.")
        for position, axis in enumerate(axes):
            if axis is None:
                continue
            # The operand at this layer includes the layer's own mapped
            # dimension but none of the later layers' dimensions.
            later = sum(outer[position] is not None for _, outer in batching[layer + 1 :])
            if not 0 <= axis < operands[position].ndim - later:
                raise ValueError("Result storage batching axis is out of range.")
    return [*storage, *operands]


def _reserve_batch(
    operands: Sequence[Array],
    axes: Sequence[int | None],
    *,
    storage: tuple[ShapedArray, ...],
    dependencies: tuple[tuple[int, ...], ...],
    batching: tuple[tuple[int, tuple[int | None, ...]], ...],
    **params: Any,
) -> tuple[list[Array], list[int | None]]:
    size = next(
        (
            operand.shape[axis]
            for operand, axis in zip(operands, axes, strict=True)
            if axis is not None
        ),
        None,
    )
    if size is None:
        result = _reserve_p.bind(
            *operands,
            storage=storage,
            dependencies=dependencies,
            batching=batching,
            **params,
        )
        return list(result), [None] * len(result)
    batched = tuple(
        any(axes[position] is not None for position in group) for group in dependencies
    )
    mapped_storage = tuple(
        aval.update(shape=(size, *aval.shape)) if mapped else aval
        for aval, mapped in zip(storage, batched, strict=True)
    )
    # Operands stay physically unmoved; the layer records how they are read.
    layer = (
        size,
        tuple(
            None if axis is None else axis % operand.ndim
            for operand, axis in zip(operands, axes, strict=True)
        ),
    )
    result = _reserve_p.bind(
        *operands,
        storage=mapped_storage,
        dependencies=dependencies,
        batching=(*batching, layer),
        **params,
    )
    return list(result), [0 if mapped else None for mapped in batched] + list(axes)


def _reserve_jvp(
    primals: Sequence[Array],
    tangents: Sequence[Any],
    *,
    storage: tuple[ShapedArray, ...],
    **params: Any,
) -> tuple[list[Array], list[Any]]:
    result = _reserve_p.bind(*primals, storage=storage, **params)
    zeros = [ad.Zero(aval.to_tangent_aval()) for aval in storage]
    return list(result), [*zeros, *tangents]


def _reserve_transpose(
    cotangents: Sequence[Any],
    *operands: Any,
    storage: tuple[ShapedArray, ...],
    **params: Any,
) -> list[Any]:
    del params
    forwarded = cotangents[len(storage) :]
    return [
        cotangent if ad.is_undefined_primal(operand) else None
        for cotangent, operand in zip(forwarded, operands, strict=True)
    ]


def _reserve_dce(
    used_outputs: list[bool], live_inputs: list[bool], equation: core.JaxprEqn
) -> tuple[list[bool], core.JaxprEqn | None]:
    del live_inputs
    if not any(used_outputs):
        return [False] * len(equation.invars), None
    params = equation.params
    storage = params["storage"]
    used_storage = used_outputs[: len(storage)]
    used_forwarded = used_outputs[len(storage) :]
    kept = [position for position, used in enumerate(used_forwarded) if used]
    renumbered = {position: new for new, position in enumerate(kept)}

    def surviving[V](values: Sequence[V], /) -> tuple[V, ...]:
        return tuple(
            value for value, used in zip(values, used_storage, strict=True) if used
        )

    pruned = equation.replace(
        invars=[equation.invars[position] for position in kept],
        outvars=[
            var for var, used in zip(equation.outvars, used_outputs, strict=True) if used
        ],
        params=dict(
            params,
            storage=surviving(storage),
            dependencies=tuple(
                tuple(
                    renumbered[position] for position in group if position in renumbered
                )
                for group in surviving(params["dependencies"])
            ),
            # Original slot and input-role numbers survive pruning unchanged.
            slots=surviving(params["slots"]),
            origins=tuple(params["origins"][position] for position in kept),
            batching=tuple(
                (size, tuple(axes[position] for position in kept))
                for size, axes in params["batching"]
            ),
        ),
    )
    return list(used_forwarded), pruned


_reserve_p = core.Primitive("phydrax_result_storage_reserve")
_reserve_p.multiple_results = True
_reserve_p.def_impl(_reserve_reference)
_reserve_p.def_abstract_eval(_reserve_abstract)
mlir.register_lowering(
    _reserve_p, mlir.lower_fun(_reserve_reference, multiple_results=True)
)
mlir.register_lowering(
    _reserve_p,
    partial(lower_mosaic_or_reference, _reserve_native, _reserve_reference),
    platform="cuda",
)
batching.primitive_batchers[_reserve_p] = _reserve_batch
ad.primitive_jvps[_reserve_p] = _reserve_jvp
ad.primitive_transposes[_reserve_p] = _reserve_transpose
pe.dce_rules[_reserve_p] = _reserve_dce


# ---------------------------------------------------------------------------
# Publication: ``values = publish(*storage, *values)``.
# ---------------------------------------------------------------------------


def _halves[V](operands: Sequence[V], /) -> tuple[Sequence[V], Sequence[V]]:
    count = len(operands) // 2
    return operands[:count], operands[count:]


def _publish_reference(*operands: Array) -> list[Array]:
    return list(_halves(operands)[1])


def _publish_native(*operands: Array) -> list[Array]:
    storage, values = _halves(operands)
    avals = [ShapedArray(value.shape, value.dtype) for value in values]
    if not _native_storage(avals):
        return list(values)
    return list(_publication_kernel(_structs(values))(*values, *storage))


def _publish_abstract(*operands: ShapedArray) -> list[ShapedArray]:
    storage, values = _halves(operands)
    if len(operands) % 2 or any(
        target.shape != value.shape or target.dtype != value.dtype
        for target, value in zip(storage, values, strict=True)
    ):
        raise ValueError("Published values must match their reserved storage.")
    return list(values)


def _leading(value: Array, axis: int | None, /) -> Array:
    return value if axis is None else jnp.moveaxis(value, axis, 0)


def _publish_batch(
    operands: Sequence[Array], axes: Sequence[int | None]
) -> tuple[list[Array], list[int | None]]:
    storage, values = _halves(operands)
    storage_axes, value_axes = _halves(axes)
    results: list[Array] = list(values)
    result_axes: list[int | None] = list(value_axes)
    # A leaf whose storage and value disagree on batching keeps its value
    # unpublished, preserving the value's own batching.
    published = [
        position
        for position, (target, value) in enumerate(
            zip(storage_axes, value_axes, strict=True)
        )
        if (target is None) == (value is None)
    ]
    if published:
        bound = _publish_p.bind(
            *(_leading(storage[index], storage_axes[index]) for index in published),
            *(_leading(values[index], value_axes[index]) for index in published),
        )
        for position, value in zip(published, bound, strict=True):
            results[position] = value
            result_axes[position] = None if value_axes[position] is None else 0
    return results, result_axes


def _publish_jvp(
    primals: Sequence[Array], tangents: Sequence[Any]
) -> tuple[list[Array], list[Any]]:
    return list(_publish_p.bind(*primals)), list(_halves(tangents)[1])


def _publish_transpose(cotangents: Sequence[Any], *operands: Any) -> list[Any]:
    storage, values = _halves(operands)
    return [None] * len(storage) + [
        cotangent if ad.is_undefined_primal(value) else None
        for cotangent, value in zip(cotangents, values, strict=True)
    ]


def _publish_dce(
    used_outputs: list[bool], live_inputs: list[bool], equation: core.JaxprEqn
) -> tuple[list[bool], core.JaxprEqn | None]:
    del live_inputs
    if not any(used_outputs):
        return [False] * len(equation.invars), None
    storage, values = _halves(equation.invars)
    kept = [position for position, used in enumerate(used_outputs) if used]
    pruned = equation.replace(
        invars=[storage[position] for position in kept]
        + [values[position] for position in kept],
        outvars=[equation.outvars[position] for position in kept],
    )
    return used_outputs + used_outputs, pruned


_publish_p = core.Primitive("phydrax_result_storage_publish")
_publish_p.multiple_results = True
_publish_p.def_impl(_publish_reference)
_publish_p.def_abstract_eval(_publish_abstract)
mlir.register_lowering(
    _publish_p, mlir.lower_fun(_publish_reference, multiple_results=True)
)
mlir.register_lowering(
    _publish_p,
    partial(lower_mosaic_or_reference, _publish_native, _publish_reference),
    platform="cuda",
)
batching.primitive_batchers[_publish_p] = _publish_batch
ad.primitive_jvps[_publish_p] = _publish_jvp
ad.primitive_transposes[_publish_p] = _publish_transpose
pe.dce_rules[_publish_p] = _publish_dce


# ---------------------------------------------------------------------------
# Public evaluation boundary.
# ---------------------------------------------------------------------------


def _is_tracer(value: object, /) -> bool:
    return isinstance(value, jax_core.Tracer)


def _computation_domain(
    jaxpr: jax_core.Jaxpr,
    consts: Sequence[Any],
    input_binding: Sequence[int],
    capture_binding: Sequence[int | None],
    /,
) -> str:
    """Canonical SHA-256 of one traced computation, independent of object IDs.

    The program is the canonical compiler payload with captures declared
    dynamic; a traced capture contributes only its position and type, a
    concrete capture also its content. The bindings give, for every program
    input and capture, the deduplicated reservation input it reads (``None``
    for a concrete capture), so the same program over aliased arguments such
    as ``f(a, a, b)`` and ``f(a, b, b)`` has distinct domains. Computations
    the canonical compiler identity does not support fail closed.
    """

    from .._fingerprint import array_tree_fingerprint, canonical_json
    from .._identity import execution_metadata_payload

    constants: list[Any] = []
    for value in consts:
        if _is_tracer(value):
            constants.append(None)
        elif jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
            constants.append(
                {
                    "key_type": str(value.dtype),
                    "data": array_tree_fingerprint(
                        np.asarray(jax.random.key_data(value))
                    ),
                }
            )
        elif jnp.issubdtype(value.dtype, jax.dtypes.extended):
            raise TypeError(
                f"Result storage cannot identify a {value.dtype} capture by content."
            )
        else:
            constants.append(array_tree_fingerprint(np.asarray(value)))
    payload = {
        "kind": "result-storage-computation",
        "program": execution_metadata_payload(
            jaxpr, "result_storage", dynamic_captures=True
        ),
        "constants": constants,
        "input_binding": list(input_binding),
        "capture_binding": list(capture_binding),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def evaluate_with_result_storage(compute: Callable[..., _T], /, *operands: Any) -> _T:
    """Evaluate ``compute(*operands)`` with result storage owned by the evaluation.

    Traced array leaves of ``operands`` and every tracer ``compute`` closes
    over are the dynamic inputs; concrete arrays and other leaves reach
    ``compute`` unchanged. The result has the structure, values, dtypes,
    batching and derivatives of ``compute(*operands)``. Outside any trace the
    call is exactly ``compute(*operands)``.
    """

    if jax_core.trace_state_clean():
        return compute(*operands)

    leaves, treedef = jtu.tree_flatten(operands)
    dynamic_positions = [
        position for position, leaf in enumerate(leaves) if _is_tracer(leaf)
    ]
    structure: list[tuple[jtu.PyTreeDef, Any]] = []

    def flat(*dynamic: Array) -> list[Array]:
        filled = list(leaves)
        for position, value in zip(dynamic_positions, dynamic, strict=True):
            filled[position] = value
        result = compute(*jtu.tree_unflatten(treedef, filled))
        traced, static = eqx.partition(result, _is_tracer)
        traced_leaves, traced_treedef = jtu.tree_flatten(traced)
        structure.append((traced_treedef, static))
        return traced_leaves

    jaxpr = jax.make_jaxpr(flat)(*(leaves[position] for position in dynamic_positions))
    ((traced_treedef, static),) = structure
    consts = list(jaxpr.consts)

    # Deduplicated dynamic inputs: operand tracers, then closed-over tracers,
    # numbered by first occurrence.
    inputs: list[Array] = []
    input_positions: dict[int, int] = {}

    def slot(value: Array, /) -> int:
        if id(value) not in input_positions:
            input_positions[id(value)] = len(inputs)
            inputs.append(value)
        return input_positions[id(value)]

    input_binding = [slot(leaves[position]) for position in dynamic_positions]
    capture_binding = [slot(value) if _is_tracer(value) else None for value in consts]
    masks: dict[jax_core.Var, int] = {}
    for var, position in zip(jaxpr.invars, input_binding, strict=True):
        masks[var] = 1 << position
    for var, position in zip(jaxpr.constvars, capture_binding, strict=True):
        if position is not None:
            masks[var] = 1 << position
    # Conservative forward dependence on the dynamic inputs.
    for equation in jaxpr.eqns:
        mask = 0
        for var in equation.invars:
            if isinstance(var, jax_core.Var):
                mask |= masks.get(var, 0)
        if mask:
            for var in equation.outvars:
                masks[var] = mask
    # A result is published only when the reservation can bind every input it
    # may depend on; any other result keeps its value and the plain execution.
    unbindable = sum(
        1 << position
        for position, value in enumerate(inputs)
        if not _bindable(jax.typeof(value))
    )

    forwarded: dict[jax_core.Var, Any] = {
        var: leaves[position]
        for var, position in zip(jaxpr.invars, dynamic_positions, strict=True)
    }
    forwarded.update(zip(jaxpr.constvars, consts, strict=True))
    published: dict[jax_core.Var, int] = {}
    published_avals: list[ShapedArray] = []
    for var in jaxpr.outvars:
        if (
            not isinstance(var, jax_core.Var)
            or var in forwarded
            or var in published
            or not masks.get(var, 0)
            or masks[var] & unbindable
        ):
            continue
        aval = var.aval
        if isinstance(aval, ShapedArray) and _publishable(aval):
            published[var] = len(published)
            published_avals.append(aval)

    if published:
        dependencies = tuple(
            tuple(bit for bit in range(len(inputs)) if masks[var] >> bit & 1)
            for var in published
        )
        storage = tuple(ShapedArray(aval.shape, aval.dtype) for aval in published_avals)
        reserved = _reserve_p.bind(
            *inputs,
            storage=storage,
            dependencies=dependencies,
            domain=_computation_domain(jaxpr, consts, input_binding, capture_binding),
            slots=tuple(range(len(storage))),
            origins=tuple(range(len(inputs))),
            batching=(),
        )
        held, ordered = reserved[: len(storage)], reserved[len(storage) :]
    else:
        held, ordered = [], list(inputs)

    evaluation_consts = [
        value if position is None else ordered[position]
        for value, position in zip(consts, capture_binding, strict=True)
    ]
    evaluation_inputs = [ordered[position] for position in input_binding]
    values = jax_core.eval_jaxpr(jaxpr, evaluation_consts, *evaluation_inputs)

    first = {
        var: index
        for index, var in reversed(list(enumerate(jaxpr.outvars)))
        if isinstance(var, jax_core.Var)
    }
    finals = (
        _publish_p.bind(*held, *(values[first[var]] for var in published))
        if published
        else []
    )
    results = [
        forwarded[var]
        if isinstance(var, jax_core.Var) and var in forwarded
        else finals[published[var]]
        if isinstance(var, jax_core.Var) and var in published
        else value
        for var, value in zip(jaxpr.outvars, values, strict=True)
    ]
    traced = jtu.tree_unflatten(traced_treedef, results)
    return cast(_T, eqx.combine(traced, static))


__all__ = ["evaluate_with_result_storage"]
