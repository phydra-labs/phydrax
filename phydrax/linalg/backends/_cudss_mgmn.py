#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lease-owned cuDSS 0.8 MGMN, with owned residuals and implicit derivatives.

All ranks invoke the same transformed collective schedule. Numerical zero
partials/cotangents participate; different symbolic AD activity is refused by
``CollectiveLease.invoke`` before any numerical entry. Matrix batches, global
JAX-array reinterpretation, hybrid execution, graph capture, rematerialization,
and algorithmic differentiation are not admitted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from hashlib import file_digest
from pathlib import Path
from typing import Any, final, NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax, ShapeDtypeStruct
from jax._src import core as jax_core, dispatch
from jax._src.interpreters import ad as _ad_internal
from jax.extend import core
from jax.interpreters import ad, batching, mlir

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
from ..._strict import StrictModule
from .._distributed_sparse import (
    distributed_sparse_action,
    DistributedCSRLinearOperator,
    DistributedCSRStorage,
)
from .._plans import LinearSolvePlan
from .._policies import FailureMode, SparseLDLT
from .._results import CudssMemoryEvidence, LinearSolveStatus
from ._cudss import CudssBackendOutput


@dataclass(frozen=True, slots=True)
class _Configuration:
    global_n: int
    row_starts: tuple[int, ...]
    partition_index: int
    structure_id: str
    partition_id: str
    storage_treedef: jax.tree_util.PyTreeDef
    capacity: int
    reordering: int
    communication_library_path: str
    communication_library_sha256: str
    communication_build_identity: str
    refinement_steps: int
    relative_tolerance: float
    absolute_tolerance: float
    derivative_relative_tolerance: float
    derivative_absolute_tolerance: float
    failure_mode: FailureMode


@final
class CudssMGMNSymbolicState(StrictModule):
    id: Array
    storage: DistributedCSRStorage
    binding: CollectiveBinding
    native_rebuilds: Array
    analysis_memory_words: Array
    configuration: _Configuration = eqx.field(static=True)


@final
class CudssMGMNFactorState(StrictModule):
    id: Array
    storage: DistributedCSRStorage
    binding: CollectiveBinding
    native_rebuilds: Array
    analysis_memory_words: Array
    positive_inertia: Array
    negative_inertia: Array
    zero_inertia: Array
    configuration: _Configuration = eqx.field(static=True)


def _suffix(dtype: np.dtype, /) -> str:
    if dtype == np.dtype(np.float32):
        return "f32"
    if dtype == np.dtype(np.float64):
        return "f64"
    raise ValueError("cuDSS MGMN supports only real float32/float64")


def _provenance(method: SparseLDLT, /) -> tuple[str, str, str]:
    execution = method.execution
    path = execution.communication_library_path
    sha256 = execution.communication_library_sha256
    identity = execution.communication_build_identity
    if path is None or sha256 is None or identity is None:
        raise ValueError("MGMN requires exact supplied communication layer provenance")
    library = Path(path)
    if not library.is_absolute() or library.resolve() != library:
        raise ValueError(
            "MGMN communication library must be an absolute resolved site path"
        )
    with library.open("rb") as stream:
        observed = file_digest(stream, "sha256").hexdigest()
    if observed != sha256:
        raise ValueError(
            "MGMN communication library SHA256 does not match supplied provenance"
        )
    load_cuda_component("cudss_mgmn")
    for field, expected in (
        ("library_path", path),
        ("library_sha256", sha256),
        ("build_identity", identity),
    ):
        observed_native = cuda_component_function(
            "cudss_mgmn", f"phx_cuda_cudss_mgmn_communication_{field}"
        )()
        if (
            not isinstance(observed_native, bytes)
            or observed_native.decode("utf-8") != expected
        ):
            raise ValueError(
                "MGMN communication provenance does not match the site component build"
            )
    return path, sha256, identity


def _operator(
    problem: Any, plan: LinearSolvePlan, /
) -> tuple[DistributedCSRLinearOperator, SparseLDLT]:
    operator = problem.operator
    method = plan.policy.method
    if not isinstance(operator, DistributedCSRLinearOperator):
        raise TypeError("cuDSS MGMN requires an explicit DistributedCSRLinearOperator")
    if not isinstance(method, SparseLDLT) or method.execution.mode != "mgmn":
        raise TypeError("cuDSS MGMN requires SparseLDLT with execution mode='mgmn'")
    if not operator.storage.self_adjoint:
        raise ValueError("cuDSS MGMN requires explicit certified self-adjoint CSR")
    if method.memory_mode != "device":
        raise ValueError(
            "cuDSS MGMN hybrid memory/execute combinations are not qualified"
        )
    if (
        method.memory_policy.require_certified_bounds
        or method.memory_policy.host_limit_bytes is not None
    ):
        raise ValueError("cuDSS MGMN cannot certify all-phase host/device memory bounds")
    if plan.policy.differentiation.mode == "algorithmic":
        raise ValueError("cuDSS MGMN refuses algorithmic differentiation")
    group = method.execution.group
    if group is None:
        raise ValueError("cuDSS MGMN requires an explicit one-GPU-per-process group")
    operator.binding.spec.require_group(group)
    if len(group.process_indices) != operator.storage.row_layout.partition_count:
        raise ValueError("MGMN CSR row partitions must match ordered process ownership")
    require_collective_lease(operator.binding.spec)
    return operator, method


def _reordering(method: SparseLDLT, /) -> int:
    match method.reordering:
        case "default":
            return 0
        case "amd":
            return 3
        case "nested_dissection":
            return 4
        case _:
            raise ValueError("cuDSS 0.8 MGMN refuses COLAMD/BTF_COLAMD")


def _configuration(
    storage: DistributedCSRStorage, method: SparseLDLT, plan: LinearSolvePlan, /
) -> _Configuration:
    path, sha256, identity = _provenance(method)
    derivative = plan.policy.derivative_solve
    if derivative.maximum_steps is not None or derivative.require_nullspace:
        raise ValueError(
            "cuDSS MGMN implicit factors do not admit iterative/nullspace derivative contracts"
        )
    ranges = storage.row_layout.owned_ranges
    return _Configuration(
        storage.global_shape[0],
        tuple(start for start, _ in ranges) + (ranges[-1][1],),
        storage.partition_index,
        storage.structure_id,
        storage.partition_id,
        jax.tree.structure(storage),
        method.factor_cache_capacity,
        _reordering(method),
        path,
        sha256,
        identity,
        method.refinement_steps,
        plan.policy.tolerance.relative,
        plan.policy.tolerance.absolute,
        max(1e-8, 10 * np.finfo(storage.values.dtype).eps)
        if derivative.relative_tolerance is None
        else derivative.relative_tolerance,
        1e-10 if derivative.absolute_tolerance is None else derivative.absolute_tolerance,
        plan.policy.failure.mode,
    )


def _attributes(
    configuration: _Configuration, binding: CollectiveBinding, operation: str, /
) -> dict[str, Any]:
    return {
        "global_n": np.int64(configuration.global_n),
        "row_starts": np.asarray(configuration.row_starts, dtype=np.int64),
        "partition_index": np.int64(configuration.partition_index),
        "capacity": np.int64(configuration.capacity),
        "reordering": np.int64(configuration.reordering),
        "preparation_id": binding.spec.preparation_id,
        "structure_id": configuration.structure_id,
        "communication_library_path": configuration.communication_library_path,
        "communication_library_sha256": configuration.communication_library_sha256,
        "communication_build_identity": configuration.communication_build_identity,
        "operation_id": operation,
    }


def _restore_storage(
    configuration: _Configuration, values: Array, offsets: Array, columns: Array, /
) -> DistributedCSRStorage:
    # PyTree metadata is reconstructible static structure, not captured arrays.
    storage = jax.tree.unflatten(
        configuration.storage_treedef, (values, columns, offsets)
    )
    if not isinstance(storage, DistributedCSRStorage):
        raise TypeError(
            "MGMN storage reconstruction requires its exact registered CSR type"
        )
    return storage


def _lifecycle(
    operation: str,
    id: Array,
    storage: DistributedCSRStorage,
    binding: CollectiveBinding,
    configuration: _Configuration,
    /,
) -> tuple[Array, Array, Array]:
    load_cuda_component("cudss_mgmn")
    result = collective_ffi_call(
        f"phydrax_cudss_mgmn_{operation}_{_suffix(storage.values.dtype)}",
        (
            ShapeDtypeStruct((), np.int64),
            ShapeDtypeStruct((), np.int32),
            ShapeDtypeStruct((1, 6, 2), np.uint32),
        ),
        (
            binding.lease_id,
            binding.invocation_ticket,
            lax.stop_gradient(id),
            lax.stop_gradient(storage.values),
            storage.indptr,
            storage.indices,
        ),
        lease_key=binding.spec,
        attributes=_attributes(configuration, binding, operation),
    )
    return result[0], result[1], result[2]


def analyze_cudss_mgmn(problem: Any, plan: LinearSolvePlan, /) -> CudssMGMNSymbolicState:
    operator, method = _operator(problem, plan)
    configuration = _configuration(operator.storage, method, plan)
    id, rebuilt, memory = _lifecycle(
        "analyze",
        jnp.asarray(0, dtype=np.int64),
        operator.storage,
        operator.binding,
        configuration,
    )
    return CudssMGMNSymbolicState(
        id, operator.storage, operator.binding, rebuilt, memory, configuration
    )


def _bind(
    symbolic: CudssMGMNSymbolicState,
    previous: CudssMGMNFactorState | None,
    problem: Any,
    plan: LinearSolvePlan,
    /,
) -> CudssMGMNFactorState:
    operator, method = _operator(problem, plan)
    configuration = _configuration(operator.storage, method, plan)
    if (
        configuration != symbolic.configuration
        or operator.binding.spec != symbolic.binding.spec
    ):
        raise ValueError(
            "MGMN binding cannot change structure, ownership, lease, provenance, or configuration"
        )
    if previous is not None and (
        previous.configuration != configuration
        or previous.binding.spec != operator.binding.spec
    ):
        raise ValueError(
            "MGMN refresh cannot change the prepared distributed structure or lease"
        )
    id, rebuilt, memory = _lifecycle(
        "factorize" if previous is None else "refactorize",
        symbolic.id if previous is None else previous.id,
        operator.storage,
        operator.binding,
        configuration,
    )
    unknown = jnp.asarray(-1, dtype=np.int32)
    return CudssMGMNFactorState(
        id,
        operator.storage,
        operator.binding,
        rebuilt,
        memory,
        unknown,
        unknown,
        unknown,
        configuration,
    )


def bind_cudss_mgmn(
    symbolic: CudssMGMNSymbolicState, problem: Any, plan: LinearSolvePlan, /
) -> CudssMGMNFactorState:
    if not isinstance(symbolic, CudssMGMNSymbolicState):
        raise TypeError("MGMN binding requires CudssMGMNSymbolicState")
    return _bind(symbolic, None, problem, plan)


def refresh_cudss_mgmn(
    symbolic: CudssMGMNSymbolicState,
    previous: CudssMGMNFactorState,
    problem: Any,
    plan: LinearSolvePlan,
    /,
) -> CudssMGMNFactorState:
    if not isinstance(symbolic, CudssMGMNSymbolicState) or not isinstance(
        previous, CudssMGMNFactorState
    ):
        raise TypeError("MGMN refresh requires its own symbolic and factor states")
    return _bind(symbolic, previous, problem, plan)


def _numerical_solve(
    lease_id: Array,
    ticket: Array,
    id: Array,
    values: Array,
    offsets: Array,
    columns: Array,
    rhs: Array,
    *,
    configuration: _Configuration,
    lease_key: CollectiveLeaseSpec,
    transpose: bool,
    derivative: bool,
) -> tuple[Array, Array, Array]:
    load_cuda_component("cudss_mgmn")
    binding = CollectiveBinding(lease_id, ticket, lease_key)
    storage = _restore_storage(configuration, values, offsets, columns)
    operation = ("implicit-" if derivative else "") + (
        "solve-transpose" if transpose else "solve"
    )
    attributes = _attributes(configuration, binding, operation)
    memo_id = id

    def solve_one(right: Array) -> tuple[Array, Array]:
        nonlocal memo_id
        result, count, memo_id = collective_ffi_call(
            f"phydrax_cudss_mgmn_solve_{_suffix(values.dtype)}",
            (
                ShapeDtypeStruct((right.shape[1], right.shape[0]), right.dtype),
                ShapeDtypeStruct((), np.int32),
                ShapeDtypeStruct((), np.int64),
            ),
            (
                lease_id,
                ticket,
                memo_id,
                values,
                offsets,
                columns,
                jnp.swapaxes(right, 0, 1),
            ),
            lease_key=lease_key,
            attributes=attributes,
        )
        return jnp.swapaxes(result, 0, 1), count

    value, rebuilt = solve_one(rhs)
    for _ in range(configuration.refinement_steps):
        residual = rhs - distributed_sparse_action(
            storage, binding, value, transpose=transpose
        )
        correction, count = solve_one(residual)
        value = value + correction
        rebuilt = rebuilt + count
    residual = rhs - distributed_sparse_action(
        storage, binding, value, transpose=transpose
    )
    local = jnp.stack(
        (
            jnp.sum(jnp.square(residual), axis=0),
            jnp.sum(jnp.square(rhs), axis=0),
            (
                jnp.all(jnp.isfinite(value), axis=0)
                & jnp.all(jnp.isfinite(residual), axis=0)
                & jnp.all(jnp.isfinite(rhs), axis=0)
            ).astype(rhs.dtype),
        )
    )
    relative = (
        configuration.derivative_relative_tolerance
        if derivative
        else configuration.relative_tolerance
    )
    absolute = (
        configuration.derivative_absolute_tolerance
        if derivative
        else configuration.absolute_tolerance
    )
    (statistics,) = collective_ffi_call(
        f"phydrax_cudss_mgmn_norm_{_suffix(values.dtype)}",
        (ShapeDtypeStruct(local.shape, local.dtype),),
        (lease_id, ticket, local),
        lease_key=lease_key,
        attributes={
            "operation_id": ("implicit-" if derivative else "")
            + ("owned-transpose-residual" if transpose else "owned-residual"),
            "require_success": np.int64(
                derivative and configuration.failure_mode == "error"
            ),
            "relative_tolerance": np.float64(relative),
            "absolute_tolerance": np.float64(absolute),
        },
    )
    finite = (statistics[2] == 1) & jnp.all(jnp.isfinite(statistics[:2]), axis=0)
    accepted = jnp.sqrt(statistics[0]) <= absolute + relative * jnp.sqrt(statistics[1])
    status = jnp.where(
        finite,
        jnp.where(
            accepted,
            int(LinearSolveStatus.SUCCESS),
            int(LinearSolveStatus.RESIDUAL_TOO_LARGE),
        ),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
    ).astype(np.int32)
    if derivative:
        value = jnp.where(
            status[None, :] == int(LinearSolveStatus.SUCCESS), value, jnp.nan
        )
    return value, rebuilt, status


def _solve_impl(
    lease_id: Array,
    ticket: Array,
    id: Array,
    values: Array,
    offsets: Array,
    columns: Array,
    rhs: Array,
    *,
    configuration: _Configuration,
    lease_key: CollectiveLeaseSpec,
    transpose: bool,
) -> tuple[Array, Array, Array]:
    return _numerical_solve(
        lease_id,
        ticket,
        id,
        values,
        offsets,
        columns,
        rhs,
        configuration=configuration,
        lease_key=lease_key,
        transpose=transpose,
        derivative=False,
    )


def _tangent_impl(
    lease_id: Array,
    ticket: Array,
    id: Array,
    values: Array,
    offsets: Array,
    columns: Array,
    rhs: Array,
    *,
    configuration: _Configuration,
    lease_key: CollectiveLeaseSpec,
    transpose: bool,
) -> tuple[Array]:
    value, _, _ = _numerical_solve(
        lease_id,
        ticket,
        id,
        values,
        offsets,
        columns,
        rhs,
        configuration=configuration,
        lease_key=lease_key,
        transpose=transpose,
        derivative=True,
    )
    return (value,)


def _abstract(
    *avals: jax_core.AbstractValue,
    configuration: _Configuration,
    lease_key: CollectiveLeaseSpec,
    transpose: bool,
    tangent: bool,
) -> tuple[tuple[jax_core.ShapedArray, ...], set[Any]]:
    del configuration, transpose
    rhs = avals[-1]
    if not isinstance(rhs, jax_core.ShapedArray):
        raise TypeError("MGMN right-hand side must be an ordinary numeric array")
    result = (
        (rhs,)
        if tangent
        else (
            rhs,
            jax_core.ShapedArray((), np.int32),
            jax_core.ShapedArray((rhs.shape[1],), np.int32),
        )
    )
    return collective_effectful_abstract_eval(
        *avals, result_avals=result, lease_key=lease_key
    )


def _solve_abstract(
    *avals: jax_core.AbstractValue, **params: Any
) -> tuple[tuple[jax_core.ShapedArray, ...], set[Any]]:
    return _abstract(*avals, tangent=False, **params)


def _tangent_abstract(
    *avals: jax_core.AbstractValue, **params: Any
) -> tuple[tuple[jax_core.ShapedArray, ...], set[Any]]:
    return _abstract(*avals, tangent=True, **params)


def _differential(
    primals: Sequence[Array],
    tangents: Sequence[Array | ad.Zero],
    value: Array,
    *,
    configuration: _Configuration,
    lease_key: CollectiveLeaseSpec,
    transpose: bool,
) -> Array:
    storage = _restore_storage(
        configuration, ad.instantiate_zeros(tangents[3]), primals[4], primals[5]
    )
    binding = CollectiveBinding(primals[0], primals[1], lease_key)
    right = ad.instantiate_zeros(tangents[6]) - distributed_sparse_action(
        storage, binding, value, transpose=transpose
    )
    (tangent,) = _tangent_p.bind(
        *primals[:6],
        right,
        configuration=configuration,
        lease_key=lease_key,
        transpose=transpose,
    )
    return tangent


def _jvp(
    primals: Sequence[Array],
    tangents: Sequence[Array | ad.Zero],
    *,
    primitive: core.Primitive,
    **params: Any,
) -> tuple[tuple[Array, ...], tuple[Array | ad.Zero, ...]]:
    primal = tuple(primitive.bind(*primals, **params))
    tangent = _differential(primals, tangents, primal[0], **params)
    return primal, (
        tangent,
        *(ad.Zero(jax.typeof(item).to_tangent_aval()) for item in primal[1:]),
    )


def _linearize(
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
    Callable[..., tuple[Array | ad.Zero, ...]],
]:
    del is_vjp, tangent_nzs
    primal = tuple(primitive.bind(*primals, **params))

    def linearized(
        residuals: tuple[Array, ...], structured: None, *tangents: Array | ad.Zero
    ) -> tuple[Array | ad.Zero, ...]:
        del structured
        tangent = _differential(residuals[:7], tangents, residuals[7], **params)
        return (
            tangent,
            *(ad.Zero(jax.typeof(item).to_tangent_aval()) for item in residuals[8:]),
        )

    # Always active: a zero LOCAL tangent may receive a nonzero remote one.
    return (
        primal,
        (True, *((False,) * (len(primal) - 1))),
        (*primals, *primal),
        None,
        linearized,
    )


def _transpose(
    cotangents: Sequence[Array | ad.Zero],
    lease_id: Array,
    ticket: Array,
    id: Array,
    values: Array,
    offsets: Array,
    columns: Array,
    rhs: Array | ad.UndefinedPrimal,
    **params: Any,
) -> tuple[Array | None, ...]:
    if any(
        ad.is_undefined_primal(value)
        for value in (lease_id, ticket, id, values, offsets, columns)
    ):
        raise ValueError(
            "MGMN transpose requires retained coefficients and the live collective binding"
        )
    # Direct registration intentionally does not use deflinear2, which skips
    # symbolic-zero cotangents and would orphan other ranks' inverse solve.
    cotangent = ad.instantiate_zeros(cotangents[0])
    reverse = dict(params, transpose=not params["transpose"])
    (right,) = _tangent_p.bind(
        lease_id, ticket, id, values, offsets, columns, cotangent, **reverse
    )
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        right if ad.is_undefined_primal(rhs) else None,
    )


def _reject_batching(*args: Any, **params: Any) -> NoReturn:
    del args, params
    raise ValueError("cuDSS MGMN refuses matrix-batch vmap; use multiple RHS")


_solve_p = core.Primitive("phydrax_cudss_mgmn_solve")
_tangent_p = core.Primitive("phydrax_cudss_mgmn_tangent_solve")
for _primitive, _abstract_rule, _implementation in (
    (_solve_p, _solve_abstract, _solve_impl),
    (_tangent_p, _tangent_abstract, _tangent_impl),
):
    _primitive.multiple_results = True
    dispatch.simple_impl(_primitive)
    _primitive.def_effectful_abstract_eval(_abstract_rule)
    ad.primitive_jvps[_primitive] = partial(_jvp, primitive=_primitive)
    _ad_internal.primitive_linearizations[_primitive] = partial(
        _linearize, primitive=_primitive
    )
    _ad_internal.linearize_on_zero_tangents.add(_primitive)
    batching.fancy_primitive_batchers[_primitive] = _reject_batching
    mlir.register_lowering(
        _primitive,
        mlir.lower_fun(_implementation, multiple_results=True),
        platform="cuda",
    )
ad.primitive_transposes[_tangent_p] = _transpose


def _schedule(equation: core.JaxprEqn) -> tuple[tuple[CollectiveLeaseSpec, str], ...]:
    configuration = equation.params["configuration"]
    lease_key = equation.params["lease_key"]
    transpose = equation.params["transpose"]
    values = equation.invars[3].aval
    if not isinstance(values, jax_core.ShapedArray):
        raise TypeError("MGMN schedule requires ordinary numeric CSR values")
    suffix = _suffix(values.dtype)
    prefix = "implicit-" if equation.primitive is _tangent_p else ""
    solve_id = collective_call_identity(
        f"phydrax_cudss_mgmn_solve_{suffix}",
        lease_key,
        prefix + ("solve-transpose" if transpose else "solve"),
    )
    exchange_id = collective_call_identity(
        f"phydrax_cudss_mgmn_exchange_{suffix}",
        lease_key,
        "csr-coordinate-scatter" if transpose else "csr-coordinate-gather",
    )
    norm_id = collective_call_identity(
        f"phydrax_cudss_mgmn_norm_{suffix}",
        lease_key,
        prefix + ("owned-transpose-residual" if transpose else "owned-residual"),
    )
    action = ((lease_key, exchange_id),)
    solve = ((lease_key, solve_id),)
    return (
        solve
        + (action + solve) * configuration.refinement_steps
        + action
        + ((lease_key, norm_id),)
    )


register_collective_schedule_rule(_solve_p, _schedule)
register_collective_schedule_rule(_tangent_p, _schedule)


def solve_cudss_mgmn(
    state: CudssMGMNFactorState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
    *,
    transpose: bool = False,
) -> CudssBackendOutput:
    if not isinstance(state, CudssMGMNFactorState):
        raise TypeError("MGMN solve requires CudssMGMNFactorState")
    method = plan.policy.method
    if not isinstance(method, SparseLDLT) or method.execution.mode != "mgmn":
        raise TypeError("MGMN solve requires SparseLDLT execution mode='mgmn'")
    group = method.execution.group
    if group is None or method.memory_mode != "device":
        raise ValueError("MGMN solve requires its prepared device-memory execution group")
    state.binding.spec.require_group(group)
    configuration = state.configuration
    if (
        method.execution.communication_library_path
        != configuration.communication_library_path
        or method.execution.communication_library_sha256
        != configuration.communication_library_sha256
        or method.execution.communication_build_identity
        != configuration.communication_build_identity
        or method.refinement_steps != configuration.refinement_steps
        or method.factor_cache_capacity != configuration.capacity
        or _reordering(method) != configuration.reordering
        or plan.policy.tolerance.relative != configuration.relative_tolerance
        or plan.policy.tolerance.absolute != configuration.absolute_tolerance
        or plan.policy.failure.mode != configuration.failure_mode
    ):
        raise ValueError("MGMN solve configuration differs from its prepared factors")
    if (
        rhs.ndim != 2
        or rhs.shape[0] != state.storage.local_row_count
        or rhs.shape[1] < 1
        or rhs.dtype != state.storage.values.dtype
    ):
        raise ValueError(
            "MGMN RHS must be unpadded rank-local (owned_rows,nrhs) with the CSR dtype"
        )
    require_collective_lease(state.binding.spec)
    values = state.storage.values
    match plan.policy.differentiation.mode:
        case "mathematical":
            right = rhs
        case "rhs-only":
            values = lax.stop_gradient(values)
            right = rhs
        case "none":
            values, right = lax.stop_gradient(values), lax.stop_gradient(rhs)
        case "algorithmic":
            raise ValueError("cuDSS MGMN refuses algorithmic differentiation")
        case _:
            raise ValueError("Unsupported MGMN differentiation mode")
    operands = (
        state.binding.lease_id,
        state.binding.invocation_ticket,
        lax.stop_gradient(state.id),
        values,
        state.storage.indptr,
        state.storage.indices,
        right,
    )
    if plan.policy.differentiation.mode == "none":
        value, rebuilt, status = _numerical_solve(
            *operands,
            configuration=state.configuration,
            lease_key=state.binding.spec,
            transpose=transpose,
            derivative=False,
        )
    else:
        value, rebuilt, status = _solve_p.bind(
            *operands,
            configuration=state.configuration,
            lease_key=state.binding.spec,
            transpose=transpose,
        )
    return CudssBackendOutput(
        value=value,
        status=status,
        iterations=jnp.full(status.shape, method.refinement_steps, dtype=np.int32),
        refinement_steps=jnp.full(status.shape, method.refinement_steps, dtype=np.int32),
        rank=jnp.asarray(-1, dtype=np.int32),
        condition_estimate=jnp.full(status.shape, jnp.nan, dtype=rhs.dtype),
        singular_values=None,
        native_rebuilds=state.native_rebuilds + rebuilt,
        memory_evidence=CudssMemoryEvidence(
            state.analysis_memory_words,
            device_indices=(
                state.binding.spec.device_keys[state.storage.partition_index][1],
            ),
            memory_mode="device",
            group_id=state.binding.spec.group_id,
            process_index=jax.process_index(),
            group_device_count=len(state.binding.spec.device_keys),
        ),
    )


def release_cudss_mgmn(state: CudssMGMNFactorState, /) -> Array:
    if not isinstance(state, CudssMGMNFactorState):
        raise TypeError("MGMN release requires CudssMGMNFactorState")
    load_cuda_component("cudss_mgmn")
    (released,) = collective_ffi_call(
        "phydrax_cudss_mgmn_release",
        (ShapeDtypeStruct((), np.int32),),
        (
            state.binding.lease_id,
            state.binding.invocation_ticket,
            lax.stop_gradient(state.id),
        ),
        lease_key=state.binding.spec,
        attributes={
            "preparation_id": state.binding.spec.preparation_id,
            "operation_id": "release",
        },
    )
    return released > 0


__all__ = [
    "CudssMGMNFactorState",
    "CudssMGMNSymbolicState",
    "analyze_cudss_mgmn",
    "bind_cudss_mgmn",
    "refresh_cudss_mgmn",
    "release_cudss_mgmn",
    "solve_cudss_mgmn",
]
