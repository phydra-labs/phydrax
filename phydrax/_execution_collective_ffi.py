#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Ordered CUDA collectives admitted by an explicitly live execution lease.

The FFI ABI is ``(token, int64 lease_id, int64 ticket, *arrays) -> (token, *arrays)``.
Both integer scalars must be dynamic arguments from ``CollectiveBinding`` under
``CollectiveLease.invoke``; closing over them does not authorize later calls.
Every native handler must use the common collective call guard before its vendor:
validate the live lease and active ticket, match configuration/call/sequence on
all ranks, and retain group serialization through CUDA stream completion. Static
attributes and a trace-time lease check cannot replace that runtime guard.

JAX's ordered-effect runtime carries each group's token between separately
compiled, asynchronously dispatched executables on the invoking host thread.
The lease owns host-thread admission and native authorization, including cached
executables that never execute this Python entry point again. Numerical outputs
may be discarded without discarding the collective. No communicator is created,
initialized, or retired here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import jax
import numpy as np
from jax import Array, ShapeDtypeStruct
from jax._src import core as _core, dispatch, effects, ffi as _ffi
from jax._src.shard_map import shard_map_p
from jax.extend import core
from jax.extend.core import primitives
from jax.interpreters import ad, batching, mlir
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ._execution_collectives import CollectiveLeaseSpec, require_collective_lease
from ._fingerprint import canonical_fingerprint


@dataclass(frozen=True, slots=True)
class _CollectiveEffect(effects.Effect):
    # Different provider/configuration leases of one group must share its order.
    group_id: str

    def __str__(self) -> str:
        return f"Collective[{self.group_id}]"


# These are the only private effect registrations. In particular remat is not
# admitted: duplicating a communicator operation is not recomputation. Owners
# may supply mathematical custom derivatives, but raw FFI has no derivative.
for _registry in (
    effects.ordered_effects,
    effects.shardable_ordered_effects,
    effects.lowerable_effects,
    effects.control_flow_allowed_effects,
    effects.custom_derivatives_allowed_effects,
    effects.partial_eval_kept_effects,
):
    _registry.add_type(_CollectiveEffect)


type _ScheduleRule = Callable[
    [core.JaxprEqn], tuple[tuple[CollectiveLeaseSpec, str], ...]
]

_schedule_rules: dict[core.Primitive, _ScheduleRule] = {}
_transparent_calls = frozenset(
    (
        primitives.jit_p,
        primitives.call_p,
        primitives.closed_call_p,
        primitives.custom_jvp_call_p,
        primitives.custom_vjp_call_p,
        shard_map_p,
    )
)


def collective_call_identity(
    target_name: str, lease_key: CollectiveLeaseSpec, operation_id: str
) -> str:
    """Identify one rank-independent native operation, never its local buffers."""
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("collective target_name must be a nonempty string")
    if not isinstance(operation_id, str):
        raise TypeError("collective operation_id must be a shared semantic string")
    if not operation_id.strip() or "\x00" in operation_id:
        raise ValueError("collective operation_id must be nonempty and NUL-free")
    return canonical_fingerprint(
        {
            "target": target_name,
            "lease_specification": lease_key.identity,
            "operation_id": operation_id,
        }
    )


def register_collective_schedule_rule(
    primitive: core.Primitive, rule: _ScheduleRule
) -> None:
    """Register a mathematical primitive's exact native call expansion."""
    if primitive in _schedule_rules:
        raise ValueError(f"collective schedule already registered for {primitive.name}")
    _schedule_rules[primitive] = rule
    # JAX otherwise intentionally erases the device assignment before lowering.
    # Mathematical wrappers must opt in too: their raw FFI is hidden until MLIR.
    dispatch.prim_requires_devices_during_lowering.add(primitive)


def collective_schedule(
    jaxpr: core.Jaxpr, *, lease_key: CollectiveLeaseSpec
) -> tuple[str, ...]:
    """Extract the actual transformed schedule before authorizing numeric entry.

    Mathematical owners register their lowering's semantic expansion. Opaque
    collective primitives, dynamic control flow, and rematerialization refuse
    rather than guessing a schedule from source text, HLO, or primal activity.
    Transparent calls preserve order; fixed-length scans repeat a proved body
    schedule. The invocation owner agrees the resulting IDs on all ranks before
    minting a ticket and checks their consumption through the common native guard.
    """
    schedule: list[str] = []
    for equation in jaxpr.eqns:
        if not any(isinstance(effect, _CollectiveEffect) for effect in equation.effects):
            continue
        rule = _schedule_rules.get(equation.primitive)
        if rule is not None:
            entries = rule(equation)
            if not entries:
                raise ValueError(
                    "collective schedule rule cannot erase an effectful operation"
                )
            for spec, call_id in entries:
                if spec != lease_key:
                    raise ValueError(
                        "collective schedule captures a different lease specification"
                    )
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError(
                        "collective schedule rule returned an invalid call identity"
                    )
                schedule.append(call_id)
            continue
        nested = tuple(core.jaxprs_in_params(equation.params))
        if equation.primitive in _transparent_calls and len(nested) == 1:
            schedule.extend(collective_schedule(nested[0], lease_key=lease_key))
        elif equation.primitive is primitives.scan_p and len(nested) == 1:
            length = equation.params["length"]
            if type(length) is not int or length < 0:
                raise ValueError("collective scan requires a fixed nonnegative length")
            body = collective_schedule(nested[0], lease_key=lease_key)
            schedule.extend(body * length)
        else:
            raise ValueError(
                "cannot establish a fixed collective schedule for "
                f"{equation.primitive.name}; register the mathematical owner's "
                "exact lowering or remove unsupported control flow/rematerialization"
            )
    return tuple(schedule)


def _raw_schedule(
    equation: core.JaxprEqn,
) -> tuple[tuple[CollectiveLeaseSpec, str], ...]:
    return ((equation.params["lease_key"], equation.params["call_id"]),)


def _require_invocation_operands(
    avals: Sequence[_core.AbstractValue],
) -> tuple[_core.ShapedArray, ...]:
    if len(avals) < 2:
        raise ValueError("collective operands must begin with a lease ID and ticket")
    arrays: list[_core.ShapedArray] = []
    for aval in avals:
        if not isinstance(aval, _core.ShapedArray) or not isinstance(
            aval.dtype, np.dtype
        ):
            raise TypeError("collective operands must be ordinary numeric arrays")
        arrays.append(aval)
    for aval in arrays[:2]:
        if aval.shape != () or aval.dtype != np.dtype(np.int64):
            raise ValueError("collective lease ID and ticket must be scalar int64 arrays")
    return tuple(arrays)


def collective_effectful_abstract_eval(
    *avals: _core.AbstractValue,
    result_avals: tuple[_core.ShapedArray, ...],
    lease_key: CollectiveLeaseSpec,
    **params: Any,
) -> tuple[tuple[_core.ShapedArray, ...], set[effects.Effect]]:
    """Validate invocation/effect identity, preserving the owner's exact result types."""
    del params
    _require_invocation_operands(avals)
    # A mathematical owner may prove replication (for example a reduction) or
    # carry a permutation's variance across an AD boundary outside its lexical
    # mesh scope. Neither proof may be overwritten by generic FFI assumptions.
    return result_avals, {_CollectiveEffect(lease_key.group_id)}


def _raw_abstract(
    *avals: _core.AbstractValue,
    result_avals: tuple[_core.ShapedArray, ...],
    lease_key: CollectiveLeaseSpec,
    **params: Any,
) -> tuple[tuple[_core.ShapedArray, ...], set[effects.Effect]]:
    arrays = _require_invocation_operands(avals)
    sharding = _core.get_cur_mesh_sharding()
    current_axes = frozenset(sharding.mesh.manual_axes)
    carried_axes = frozenset().union(*(aval.mat.varying for aval in arrays))
    if not carried_axes.issubset(current_axes):
        # A carried manual mesh is real type metadata, unlike axes guessed from
        # array extents or process counts. JAX validates the combined MAT against
        # that mesh and the current named-axis environment.
        for aval in arrays:
            if aval.mat.varying:
                sharding = NamedSharding(aval.sharding.mesh, PartitionSpec())
                break
    manual_axes = _core.ManualAxisType(varying=current_axes | carried_axes)
    outputs = tuple(
        aval.update(sharding=sharding, manual_axis_type=manual_axes)
        for aval in result_avals
    )
    return collective_effectful_abstract_eval(
        *avals, result_avals=outputs, lease_key=lease_key, **params
    )


def _require_local_lowering(
    ctx: mlir.LoweringRuleContext, lease_key: CollectiveLeaseSpec
) -> None:
    axis_context = ctx.module_context.axis_context
    if isinstance(axis_context, mlir.SPMDAxisContext):
        mesh = axis_context.mesh
        if not isinstance(mesh, Mesh):
            raise ValueError("collective lowering requires a concrete execution mesh")
        if frozenset(axis_context.manual_axes) != frozenset(mesh.axis_names):
            raise ValueError("collective lowering requires a fully manual shard_map")
        device_keys = tuple(
            (device.process_index, device.id) for device in mesh.devices.flat
        )
        if device_keys != lease_key.device_keys:
            raise ValueError("collective mesh does not match the ordered lease devices")
    elif isinstance(axis_context, mlir.ShardingContext):
        if axis_context.num_devices != 1 or axis_context.device_assignment is None:
            raise ValueError(
                "collective arrays must be rank-local or in a manual shard_map"
            )
        device = axis_context.device_assignment[0]
        if (device.process_index, device.id) not in lease_key.device_keys:
            raise ValueError("collective device is not a participant in the live lease")
    else:
        raise ValueError("unsupported collective execution axis context")


def _lowering(
    ctx: mlir.LoweringRuleContext,
    *operands: mlir.ir.Value,
    target_name: str,
    result_avals: tuple[_core.ShapedArray, ...],
    lease_key: CollectiveLeaseSpec,
    attributes: tuple[tuple[str, Any], ...],
    call_id: str,
) -> Sequence[mlir.ir.Value]:
    del result_avals
    _require_local_lowering(ctx, lease_key)
    effect = _CollectiveEffect(lease_key.group_id)
    token = ctx.tokens_in.get(effect)
    native_attributes = _ffi._unwrap_kwargs_hashable(attributes)
    native_attributes.update(
        collective_configuration_id=lease_key.identity,
        collective_call_id=call_id,
    )
    # The token is a real custom-call operand/result, not an unrelated token
    # returned beside a pure call. Native handlers bind ffi::Token first on both
    # sides; all following array buffers have explicit row-major layouts.
    token_ctx = ctx.replace(
        avals_in=[_core.abstract_token, *ctx.avals_in],
        avals_out=[_core.abstract_token, *ctx.avals_out],
    )
    outputs = jax.ffi.ffi_lowering(target_name, has_side_effect=True)(
        token_ctx, token, *operands, **native_attributes
    )
    flat_outputs: list[mlir.ir.Value] = []
    for output in outputs:
        if not isinstance(output, mlir.ir.Value):
            raise TypeError(
                "collective token and array results must each lower to one IR value"
            )
        flat_outputs.append(output)
    if len(flat_outputs) != len(ctx.avals_out) + 1:
        raise ValueError("collective FFI results do not match the token-plus-array ABI")
    ctx.set_tokens_out(mlir.TokenSet({effect: flat_outputs[0]}))
    return flat_outputs[1:]


def _reject_ad(*args: Any, **params: Any) -> NoReturn:
    del args, params
    raise ValueError(
        "raw collective FFI cannot be differentiated; the scientific operation "
        "must define its mathematical derivative"
    )


def _reject_batching(*args: Any, **params: Any) -> NoReturn:
    del args, params
    raise ValueError("raw collective FFI does not support vmap")


_collective_p = core.Primitive("phydrax_collective_ffi")
_collective_p.multiple_results = True
dispatch.simple_impl(_collective_p)
_collective_p.def_effectful_abstract_eval(_raw_abstract)
ad.primitive_jvps[_collective_p] = _reject_ad
ad.primitive_transposes[_collective_p] = _reject_ad
batching.fancy_primitive_batchers[_collective_p] = _reject_batching
mlir.register_lowering(_collective_p, _lowering, platform="cuda")
register_collective_schedule_rule(_collective_p, _raw_schedule)


def collective_ffi_call(
    target_name: str,
    result_specs: Sequence[ShapeDtypeStruct],
    operands: Sequence[Array],
    *,
    lease_key: CollectiveLeaseSpec,
    attributes: Mapping[str, Any],
) -> tuple[Array, ...]:
    """Bind a guarded rank-local collective with a dynamic invocation ticket.

    ``operands[:2]`` are the dynamic scalar int64 lease ID and invocation ticket
    from the binding passed to the invoked function; remaining operands and all
    results are ordinary numeric arrays. Results use
    local shapes and row-major storage. Multi-device execution is admitted only
    inside a fully manual ``shard_map`` over the exact ordered lease mesh.
    ``attributes["operation_id"]`` is a required shared semantic identity, consumed
    here rather than forwarded to the handler. Call identity combines it with
    target and complete lease specification. Rank-local shapes and attributes
    may legitimately differ and must never enter an all-rank agreement digest.
    Global structure/layout identity belongs to the lease's agreed configuration
    and preparation. Runtime ticket and sequence validation belongs to the guard.
    """
    require_collective_lease(lease_key)
    if any(not isinstance(key, str) for key in attributes):
        raise TypeError("collective FFI attribute names must be strings")
    if any(key.startswith("collective_") for key in attributes):
        raise ValueError("collective_ attributes are reserved for the live lease")
    if "operation_id" not in attributes:
        raise ValueError("collective FFI requires a shared operation_id attribute")
    operation_id = attributes["operation_id"]
    _require_invocation_operands(tuple(map(jax.typeof, operands)))
    result_avals = tuple(
        _core.ShapedArray(spec.shape, spec.dtype) for spec in result_specs
    )
    native_attributes = dict(attributes)
    del native_attributes["operation_id"]
    frozen_attributes = tuple(_ffi._wrap_kwargs_hashable(native_attributes))
    call_id = collective_call_identity(target_name, lease_key, operation_id)
    return tuple(
        _collective_p.bind(
            *operands,
            target_name=target_name,
            result_avals=result_avals,
            lease_key=lease_key,
            attributes=frozen_attributes,
            call_id=call_id,
        )
    )
