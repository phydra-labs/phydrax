#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._precision import (
    dequantize_mx,
    MicroscaledArray,
    MicroscalingFormat,
    precision_dtype_name,
    PrecisionFormat,
    quantize_mx,
)
from .._strict import StrictModule


OptimizerStateLeafRole: TypeAlias = Literal[
    "exact",
    "first-moment",
    "second-moment",
    "trace",
    "accumulator",
]

_ROLES = frozenset({"exact", "first-moment", "second-moment", "trace", "accumulator"})


class OptimizerStateCompressionPolicy(StrictModule):
    """Deterministic local resident-state compression policy."""

    format: PrecisionFormat = eqx.field(static=True)
    block_axes: tuple[int, ...] = eqx.field(static=True)
    exact_roles: tuple[str, ...] = eqx.field(static=True)
    overflow: Literal["error", "saturate"] = eqx.field(static=True)
    differentiation: Literal["none"] = eqx.field(static=True)

    def __init__(
        self,
        format: Any = "float16",
        /,
        *,
        block_axes: Sequence[int] = (-1,),
        exact_roles: Sequence[str] = (),
        overflow: Literal["error", "saturate"] = "error",
        differentiation: Literal["none"] = "none",
    ):
        if isinstance(format, MicroscalingFormat):
            format_ = format
        else:
            format_ = precision_dtype_name(format)
            if format_ not in (
                "float8_e4m3fn",
                "float8_e5m2",
                "float8_e4m3fnuz",
                "float8_e5m2fnuz",
                "float16",
                "bfloat16",
            ):
                raise ValueError(
                    "Optimizer resident compression requires FP8, FP16, BF16, or MX."
                )
        axes = tuple(int(axis) for axis in block_axes)
        exact = tuple(str(path) for path in exact_roles)
        if any(not path for path in exact) or len(set(exact)) != len(exact):
            raise ValueError("exact_roles paths must be unique and non-empty.")
        if overflow not in ("error", "saturate"):
            raise ValueError("overflow must be error or saturate.")
        if differentiation != "none":
            raise ValueError("Optimizer state compression is nondifferentiable.")
        self.format = format_
        self.block_axes = axes
        self.exact_roles = exact
        self.overflow = overflow
        self.differentiation = "none"


class OptimizerStateCompressionPlan(StrictModule):
    """Exact optimizer treedef, roles, formats, and transformation identity."""

    treedef: jax.tree_util.PyTreeDef = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dtypes: tuple[str, ...] = eqx.field(static=True)
    leaf_roles: tuple[OptimizerStateLeafRole, ...] = eqx.field(static=True)
    compressed_indices: tuple[int, ...] = eqx.field(static=True)
    exact_indices: tuple[int, ...] = eqx.field(static=True)
    policy: OptimizerStateCompressionPolicy = eqx.field(static=True)
    transformation_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class OptimizerStateLeafDiagnostics(StrictModule):
    maximum_absolute_error: Array
    saturation_count: Array
    nonfinite_count: Array
    payload_bytes: Array
    scale_minimum: Array
    scale_maximum: Array
    path: str = eqx.field(static=True)
    role: str = eqx.field(static=True)


class CompressedOptimizerState(StrictModule):
    """Fixed-shape compressed payloads plus exact semantic leaves."""

    payloads: tuple[Array | MicroscaledArray, ...]
    exact_leaves: tuple[Array, ...]
    diagnostics: tuple[OptimizerStateLeafDiagnostics, ...]
    plan_id: str = eqx.field(static=True)


def _role_tree(state: Any, /):
    if isinstance(state, optax.ScaleByAdamState):
        return optax.ScaleByAdamState(
            count="exact",
            mu=jax.tree.map(lambda _: "first-moment", state.mu),
            nu=jax.tree.map(lambda _: "second-moment", state.nu),
        )
    if isinstance(state, optax.ScaleByLionState):
        return optax.ScaleByLionState(
            count="exact",
            mu=jax.tree.map(lambda _: "first-moment", state.mu),
        )
    if isinstance(state, optax.ScaleByRmsState):
        return optax.ScaleByRmsState(nu=jax.tree.map(lambda _: "second-moment", state.nu))
    if isinstance(state, optax.TraceState):
        return optax.TraceState(trace=jax.tree.map(lambda _: "trace", state.trace))
    if isinstance(state, optax.ScaleByScheduleState):
        return optax.ScaleByScheduleState(count="exact")
    if isinstance(state, optax.EmptyState):
        return optax.EmptyState()
    if isinstance(state, optax.MaskedState):
        return optax.MaskedState(inner_state=_role_tree(state.inner_state))
    if isinstance(state, tuple):
        return tuple(_role_tree(item) for item in state)
    if isinstance(state, list):
        return [_role_tree(item) for item in state]
    raise TypeError(
        "Unsupported optimizer state; supply an explicit exact-treedef leaf_roles tree."
    )


def _validated_role_layout(state: Any, leaf_roles: Any | None, /):
    roles = _role_tree(state) if leaf_roles is None else leaf_roles
    leaves, treedef = jax.tree.flatten(state)
    role_leaves, role_treedef = jax.tree.flatten(roles)
    if role_treedef != treedef or len(role_leaves) != len(leaves):
        raise ValueError("Optimizer leaf_roles must exactly match the state treedef.")
    normalized = tuple(str(role) for role in role_leaves)
    if any(role not in _ROLES for role in normalized):
        raise ValueError("Optimizer leaf_roles contains an unknown role.")
    return tuple(leaves), treedef, normalized


def prepare_optimizer_state_compression(
    state: Any,
    policy: OptimizerStateCompressionPolicy,
    /,
    *,
    transformation_id: str,
    leaf_roles: Any | None = None,
) -> OptimizerStateCompressionPlan:
    """Prepare explicit optimizer roles without field-name inference."""
    if not isinstance(policy, OptimizerStateCompressionPolicy):
        raise TypeError("policy must be an OptimizerStateCompressionPolicy.")
    transformation = str(transformation_id)
    if not transformation:
        raise ValueError("transformation_id must be non-empty.")
    leaves, treedef, roles = _validated_role_layout(state, leaf_roles)
    paths = tuple(
        jax.tree_util.keystr(path) or "<root>"
        for path, _ in jax.tree_util.tree_flatten_with_path(state)[0]
    )
    arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
    if len(arrays) != len(paths):
        raise ValueError("Optimizer state must be an array-only PyTree.")
    effective_roles = []
    compressed = []
    exact = []
    for index, (path, array, role) in enumerate(zip(paths, arrays, roles, strict=True)):
        force_exact = (
            path in policy.exact_roles
            or role == "exact"
            or not jnp.issubdtype(array.dtype, jnp.inexact)
        )
        if force_exact:
            effective_roles.append("exact")
            exact.append(index)
        else:
            if jnp.issubdtype(array.dtype, jnp.complexfloating):
                raise TypeError(
                    "Complex optimizer leaves require explicit exact storage or a "
                    "complex training-state layout."
                )
            effective_roles.append(role)
            compressed.append(index)
    payload = {
        "kind": "optimizer-state-compression-plan",
        "paths": paths,
        "shapes": [list(array.shape) for array in arrays],
        "dtypes": [array.dtype.name for array in arrays],
        "roles": effective_roles,
        "format": (
            policy.format.to_dict()
            if isinstance(policy.format, MicroscalingFormat)
            else policy.format
        ),
        "transformation": transformation,
    }
    return OptimizerStateCompressionPlan(
        treedef=treedef,
        paths=paths,
        shapes=tuple(tuple(int(size) for size in array.shape) for array in arrays),
        dtypes=tuple(array.dtype.name for array in arrays),
        leaf_roles=tuple(effective_roles),
        compressed_indices=tuple(compressed),
        exact_indices=tuple(exact),
        policy=policy,
        transformation_id=transformation,
        plan_id=canonical_fingerprint(payload),
    )


def _compress_scalar(array: Array, plan: OptimizerStateCompressionPlan, /):
    format_ = plan.policy.format
    if not isinstance(format_, str):
        raise TypeError("Scalar compression requires a scalar format.")
    dtype = jnp.dtype(format_)
    maximum = jnp.asarray(jnp.finfo(dtype).max, dtype=array.dtype)
    invalid = ~jnp.isfinite(array)
    overflowed = jnp.abs(array) > maximum
    saturation = jnp.sum(invalid | overflowed, dtype=jnp.int32)
    value = array
    if plan.policy.overflow == "error":
        value = eqx.error_if(
            value,
            saturation != 0,
            "Optimizer state is nonfinite or outside the compression range.",
        )
    value = jnp.nan_to_num(value, nan=0.0, posinf=maximum, neginf=-maximum)
    value = jnp.clip(value, -maximum, maximum).astype(dtype)
    return value, saturation


def compress_optimizer_state(
    plan: OptimizerStateCompressionPlan,
    state: Any,
    /,
) -> CompressedOptimizerState:
    """Pure deterministic local compression with per-leaf evidence."""
    if not isinstance(plan, OptimizerStateCompressionPlan):
        raise TypeError("plan must be an OptimizerStateCompressionPlan.")
    leaves, treedef = jax.tree.flatten(state)
    if treedef != plan.treedef or len(leaves) != len(plan.paths):
        raise ValueError("Optimizer state does not match its compression plan.")
    arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
    for array, shape, dtype in zip(arrays, plan.shapes, plan.dtypes, strict=True):
        if array.shape != shape or array.dtype.name != dtype:
            raise ValueError("Optimizer state shape/dtype changed after preparation.")
    payloads = []
    exact_leaves = tuple(arrays[index] for index in plan.exact_indices)
    diagnostics = []
    for index in plan.compressed_indices:
        array = jax.lax.stop_gradient(arrays[index])
        if isinstance(plan.policy.format, MicroscalingFormat):
            payload = quantize_mx(
                array,
                plan.policy.format,
                overflow=plan.policy.overflow,
            )
            reconstructed = dequantize_mx(payload, dtype=array.dtype)
            saturation = payload.saturation_count
            scale_values = jnp.exp2(
                payload.scales.astype(jnp.int32).astype(jnp.float32) - 127.0
            )
            scale_minimum = jnp.min(
                jnp.concatenate(
                    (jnp.ones((1,), dtype=jnp.float32), scale_values.reshape((-1,)))
                )
            )
            scale_maximum = jnp.max(
                jnp.concatenate(
                    (jnp.ones((1,), dtype=jnp.float32), scale_values.reshape((-1,)))
                )
            )
            payload_bytes = payload.payload_bytes
        else:
            payload, saturation = _compress_scalar(array, plan)
            reconstructed = payload.astype(array.dtype)
            scale_minimum = jnp.asarray(1.0, dtype=jnp.float32)
            scale_maximum = jnp.asarray(1.0, dtype=jnp.float32)
            payload_bytes = int(payload.size * payload.dtype.itemsize)
        error = jnp.max(
            jnp.concatenate(
                (
                    jnp.zeros((1,), dtype=array.real.dtype),
                    jnp.abs(reconstructed - array).reshape((-1,)),
                )
            )
        )
        diagnostics.append(
            OptimizerStateLeafDiagnostics(
                maximum_absolute_error=error,
                saturation_count=saturation,
                nonfinite_count=jnp.sum(~jnp.isfinite(array), dtype=jnp.int32),
                payload_bytes=jnp.asarray(payload_bytes, dtype=jnp.int64),
                scale_minimum=scale_minimum,
                scale_maximum=scale_maximum,
                path=plan.paths[index],
                role=plan.leaf_roles[index],
            )
        )
        payloads.append(payload)
    return CompressedOptimizerState(
        payloads=tuple(payloads),
        exact_leaves=exact_leaves,
        diagnostics=tuple(diagnostics),
        plan_id=plan.plan_id,
    )


def decompress_optimizer_state(
    plan: OptimizerStateCompressionPlan,
    state: CompressedOptimizerState,
    /,
):
    """Reconstruct the exact prepared optimizer treedef in compute dtype."""
    if not isinstance(plan, OptimizerStateCompressionPlan):
        raise TypeError("plan must be an OptimizerStateCompressionPlan.")
    if not isinstance(state, CompressedOptimizerState) or state.plan_id != plan.plan_id:
        raise ValueError("Compressed optimizer state does not match its plan.")
    if len(state.payloads) != len(plan.compressed_indices) or len(
        state.exact_leaves
    ) != len(plan.exact_indices):
        raise ValueError("Compressed optimizer payload cardinality changed.")
    leaves: list[Array | None] = [None] * len(plan.paths)
    for index, exact in zip(plan.exact_indices, state.exact_leaves, strict=True):
        leaves[index] = exact
    for index, payload in zip(plan.compressed_indices, state.payloads, strict=True):
        dtype = jnp.dtype(plan.dtypes[index])
        if isinstance(payload, MicroscaledArray):
            value = dequantize_mx(payload, dtype=dtype)
        else:
            value = jnp.asarray(payload).astype(dtype)
        if value.shape != plan.shapes[index]:
            raise ValueError("Decompressed optimizer leaf shape changed.")
        leaves[index] = value
    if any(leaf is None for leaf in leaves):
        raise ValueError("Compressed optimizer state is incomplete.")
    return jax.tree.unflatten(plan.treedef, leaves)


class PreparedCompressedOptimizer(StrictModule):
    """Optax-compatible transformation with resident compressed state."""

    transformation: Any = eqx.field(static=True)
    plan: OptimizerStateCompressionPlan = eqx.field(static=True)

    def init(self, params: Any, /) -> CompressedOptimizerState:
        state = self.transformation.init(params)
        return compress_optimizer_state(self.plan, state)

    def update(
        self,
        updates: Any,
        state: CompressedOptimizerState,
        params: Any | None = None,
        /,
        **extra_args: Any,
    ):
        decompressed = decompress_optimizer_state(self.plan, state)
        transformed, next_state = self.transformation.update(
            updates,
            decompressed,
            params,
            **extra_args,
        )
        return transformed, compress_optimizer_state(self.plan, next_state)


def prepare_compressed_optimizer(
    transformation: Any,
    params: Any,
    policy: OptimizerStateCompressionPolicy,
    /,
    *,
    transformation_id: str,
    leaf_roles: Any | None = None,
) -> PreparedCompressedOptimizer:
    """Bind an Optax transform to one explicit compression plan."""
    if not isinstance(
        transformation,
        (optax.GradientTransformation, optax.GradientTransformationExtraArgs),
    ):
        raise TypeError("transformation must be a public Optax transformation.")
    initial = transformation.init(params)
    plan = prepare_optimizer_state_compression(
        initial,
        policy,
        transformation_id=transformation_id,
        leaf_roles=leaf_roles,
    )
    return PreparedCompressedOptimizer(transformation, plan)


__all__ = [
    "CompressedOptimizerState",
    "OptimizerStateCompressionPlan",
    "OptimizerStateCompressionPolicy",
    "OptimizerStateLeafDiagnostics",
    "OptimizerStateLeafRole",
    "PreparedCompressedOptimizer",
    "compress_optimizer_state",
    "decompress_optimizer_state",
    "prepare_compressed_optimizer",
    "prepare_optimizer_state_compression",
]
