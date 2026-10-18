#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import IntEnum
from math import isfinite, prod
from typing import Any, Literal, TYPE_CHECKING, TypeAlias, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.tree_util import PyTreeDef
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg._problems import LinearSystem
from ..linalg._spaces import AbstractVectorSpace
from ..linalg._sparse_contract import AbstractSparseLinearOperator, SparseStorage
from ._availability import import_backend_module, probe_backend, probe_cuda_component
from ._types import (
    AbstractExternalBackend,
    BackendAvailability,
    BackendCapabilities,
    BackendDistributionCapabilities,
    BackendTransferEvidence,
)


if TYPE_CHECKING:
    import scipy.sparse as sp

    from ..sparse._linear import _SparseStoragePlan


AMGBackendName: TypeAlias = Literal["pyamgcl", "amgx"]
CanonicalConfig: TypeAlias = tuple[Literal["mapping"], tuple[tuple[str, Any], ...]]


PYAMGCL_CAPABILITIES = BackendCapabilities(
    backend="pyamgcl",
    problem_kinds=("linear-system",),
    execution="host",
    host_only=True,
    supports_matrix_free=False,
    supports_assembled=True,
    coordinate_dtypes=("float32", "float64"),
)
AMGX_CAPABILITIES = BackendCapabilities(
    backend="amgx",
    problem_kinds=("linear-system",),
    execution="device",
    host_only=False,
    supports_matrix_free=False,
    supports_assembled=True,
    coordinate_dtypes=("float32", "float64"),
    requires_explicit_release=False,
    distribution=BackendDistributionCapabilities(
        array_model="global",
        scopes=("single_device",),
        platforms=("cuda",),
        transformations=("jit",),
    ),
)


_DEFAULT_PYAMGCL_CONFIG = {
    "solver": {"tol": 1.0e-8, "maxiter": 100},
}
_DEFAULT_AMGX_CONFIG = {
    "config_version": 2,
    "solver": {
        "solver": "FGMRES",
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            # AmgX defaults to the classical PMIS selector, which aggregation lacks.
            "selector": "SIZE_2",
            "max_iters": 1,
        },
        "max_iters": 100,
        "tolerance": 1.0e-8,
        "norm": "L2",
        "monitor_residual": 1,
        "convergence": "RELATIVE_INI_CORE",
    },
}


def _canonical_value(value: Any, /) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("Backend configuration keys must be non-empty strings.")
        items = tuple(
            sorted((key, _canonical_value(item)) for key, item in value.items())
        )
        return ("mapping", items)
    if isinstance(value, (tuple, list)):
        return ("sequence", tuple(_canonical_value(item) for item in value))
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Backend floating configuration values must be finite.")
        return value
    raise TypeError(
        "Backend configuration values must be nested mappings/sequences of scalar JSON values."
    )


def _canonical_config(config: Mapping[str, Any], /) -> CanonicalConfig:
    value = _canonical_value(config)
    if not isinstance(value, tuple) or len(value) != 2 or value[0] != "mapping":
        raise TypeError("Backend configuration must be a mapping.")
    return value


def _config_dict(value: Any, /) -> Any:
    if isinstance(value, tuple) and len(value) == 2:
        tag, payload = value
        if tag == "mapping":
            return {key: _config_dict(item) for key, item in payload}
        if tag == "sequence":
            return [_config_dict(item) for item in payload]
    return value


class PyAMGCLPolicy(StrictModule):
    """Immutable PyAMGCL solver and parameter mapping."""

    solver: str = eqx.field(static=True)
    config: CanonicalConfig = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        solver: str = "bicgstab",
        config: Mapping[str, Any] | None = None,
    ) -> None:
        solver_ = str(solver)
        if not solver_:
            raise ValueError("PyAMGCL solver must be non-empty.")
        config_ = _canonical_config(_DEFAULT_PYAMGCL_CONFIG if config is None else config)
        self.solver = solver_
        self.config = config_
        self.policy_id = canonical_fingerprint(
            {"kind": "pyamgcl-policy", "solver": solver_, "config": config_}
        )


class AmgXPolicy(StrictModule):
    """Immutable canonical AmgX configuration."""

    config: CanonicalConfig = eqx.field(static=True)
    hierarchy_cache_capacity: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        /,
        *,
        hierarchy_cache_capacity: int = 2,
    ) -> None:
        if isinstance(hierarchy_cache_capacity, bool) or not isinstance(
            hierarchy_cache_capacity, int
        ):
            raise TypeError("AmgX hierarchy_cache_capacity must be an integer.")
        if hierarchy_cache_capacity < 1:
            raise ValueError("AmgX hierarchy_cache_capacity must be positive.")
        config_ = _canonical_config(_DEFAULT_AMGX_CONFIG if config is None else config)
        self.config = config_
        self.hierarchy_cache_capacity = hierarchy_cache_capacity
        self.policy_id = canonical_fingerprint(
            {
                "kind": "amgx-policy",
                "config": config_,
                "hierarchy_cache_capacity": hierarchy_cache_capacity,
            }
        )


class AMGSolveStatus(IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    PROVIDER_FAILURE = 2
    RELEASED = 3


class AMGSolveDiagnostics(StrictModule):
    """Backend and independently recomputed residual evidence."""

    residual_norm: Array
    relative_residual_norm: Array
    iterations: Array | None
    provider_reasons: tuple[str | None, ...] = eqx.field(static=True)
    native_status: Array | None = None
    native_residual_norm: Array | None = None
    native_rebuilds: Array | None = None


class AMGProvenance(StrictModule):
    """Symbolic and numeric identity of one external AMG result."""

    numeric_version: Array
    backend: AMGBackendName = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    promoted_to_float64: bool = eqx.field(static=True, default=False)


class AMGSolveResult(StrictModule):
    """PyTree-preserving AMG result with transfer and residual evidence."""

    value: PyTree[Array]
    status: Array
    diagnostics: AMGSolveDiagnostics
    provenance: AMGProvenance
    transfers: BackendTransferEvidence

    @property
    def success(self) -> Array:
        return self.status == int(AMGSolveStatus.SUCCESS)


class PyAMGCLPlan(StrictModule):
    policy: PyAMGCLPolicy
    shape: tuple[int, int] = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    pattern_id: str = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    index_width: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class AmgXPlan(StrictModule):
    policy: AmgXPolicy
    storage_plan: _SparseStoragePlan | None
    shape: tuple[int, int] = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    pattern_id: str = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    index_width: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class _PyAMGCLRuntime:
    def __init__(self, module: Any, matrix: Any, solver: Any) -> None:
        self.module = module
        self.matrix = matrix
        self.solver = solver


class PreparedPyAMGCL(StrictModule):
    problem: LinearSystem
    plan: PyAMGCLPlan
    numeric_version: Array
    transfers: BackendTransferEvidence
    runtime: Any = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PreparedAmgX(StrictModule):
    problem: LinearSystem
    plan: AmgXPlan
    numeric_version: Array
    transfers: BackendTransferEvidence
    native_id: Array
    indptr: Array
    indices: Array
    values: Array
    native_rebuilds: Array
    partition: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PyAMGCLBackend(AbstractExternalBackend):
    @property
    def name(self) -> str:
        return "pyamgcl"

    @property
    def capabilities(self) -> BackendCapabilities:
        return PYAMGCL_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return probe_backend(
            self.capabilities,
            module="pyamgcl",
            requirement="install the optional pyamgcl provider",
            distributions=("pyamgcl",),
        )

    def plan(
        self, problem: LinearSystem, policy: PyAMGCLPolicy | None = None, /
    ) -> PyAMGCLPlan:
        return plan_pyamgcl(problem, policy)

    def prepare(self, problem: LinearSystem, plan: PyAMGCLPlan, /) -> PreparedPyAMGCL:
        return prepare_pyamgcl(problem, plan)

    def solve(
        self,
        prepared: PreparedPyAMGCL,
        right_hand_side: PyTree[Any],
        /,
        **kwargs: Any,
    ) -> AMGSolveResult:
        return solve_pyamgcl(prepared, right_hand_side, **kwargs)

    def refresh(
        self, prepared: PreparedPyAMGCL, problem: LinearSystem, /
    ) -> PreparedPyAMGCL:
        return refresh_pyamgcl(prepared, problem)


class AmgXBackend(AbstractExternalBackend):
    @property
    def name(self) -> str:
        return "amgx"

    @property
    def capabilities(self) -> BackendCapabilities:
        return AMGX_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return probe_cuda_component(
            self.capabilities,
            component="amgx",
            requirement="install phydrax[cuda13] for JAX CUDA13",
        )

    def plan(
        self, problem: LinearSystem, policy: AmgXPolicy | None = None, /
    ) -> AmgXPlan:
        return plan_amgx(problem, policy)

    def prepare(self, problem: LinearSystem, plan: AmgXPlan, /) -> PreparedAmgX:
        return prepare_amgx(problem, plan)

    def solve(
        self,
        prepared: PreparedAmgX,
        right_hand_side: PyTree[Any],
        /,
        **kwargs: Any,
    ) -> AMGSolveResult:
        return solve_amgx(prepared, right_hand_side, **kwargs)

    def refresh(self, prepared: PreparedAmgX, problem: LinearSystem, /) -> PreparedAmgX:
        return refresh_amgx(prepared, problem)

    def release(self, prepared: PreparedAmgX, /) -> None:
        release_amgx(prepared)


def pyamgcl_availability() -> BackendAvailability:
    """Probe PyAMGCL without retaining or importing it at package import time."""
    return PyAMGCLBackend().availability()


def amgx_availability() -> BackendAvailability:
    """Probe the AmgX companion without loading it at package import time."""
    return AmgXBackend().availability()


def _storage(
    problem: LinearSystem, /, *, storage_plan: _SparseStoragePlan | None = None
) -> SparseStorage:
    from ..sparse._linear import SparseCoordinateOperator, SparseLinearMap

    if not isinstance(problem, LinearSystem):
        raise TypeError("AMG backends require a LinearSystem.")
    operator = problem.operator
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError("AMG backends require AbstractSparseLinearOperator storage.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("AMG backends require an unbatched square endomorphism.")
    storage = (
        storage_plan.apply(operator.coefficients, relation=operator.relation)
        if storage_plan is not None
        and isinstance(operator, (SparseCoordinateOperator, SparseLinearMap))
        else operator.sparse_storage()
    )
    if storage.shape[0] != storage.shape[1]:
        raise ValueError("AMG backends require square CSR storage.")
    if not storage.canonical or not storage.sorted_indices:
        raise ValueError("AMG backends require canonical sorted CSR storage.")
    dtype = np.dtype(storage.values.dtype)
    if dtype.name not in ("float32", "float64"):
        raise TypeError("AMG backends support real float32 and float64 coordinates.")
    return storage


def _pattern_id(storage: SparseStorage, /) -> str:
    return canonical_fingerprint(
        {
            "shape": storage.shape,
            "indices": np.asarray(storage.indices).tolist(),
            "indptr": np.asarray(storage.indptr).tolist(),
        }
    )


class _AMGPlanFields(TypedDict):
    shape: tuple[int, int]
    source_space_id: str
    target_space_id: str
    operator_id: str
    pattern_id: str
    coordinate_dtype: str
    index_width: int
    plan_id: str


def _plan_fields(
    problem: LinearSystem,
    policy_id: str,
    backend: AMGBackendName,
    /,
    *,
    storage_plan: _SparseStoragePlan | None = None,
) -> _AMGPlanFields:
    storage = _storage(problem, storage_plan=storage_plan)
    pattern_id = _pattern_id(storage)
    coordinate_dtype = np.dtype(storage.values.dtype).name
    payload = {
        "kind": f"{backend}-plan",
        "policy": policy_id,
        "shape": storage.shape,
        "source": problem.operator.source.space_id,
        "target": problem.operator.target.space_id,
        "operator": problem.operator.operator_id,
        "pattern": pattern_id,
        "dtype": coordinate_dtype,
        "index_width": storage.index_width,
    }
    return _AMGPlanFields(
        shape=storage.shape,
        source_space_id=problem.operator.source.space_id,
        target_space_id=problem.operator.target.space_id,
        operator_id=problem.operator.operator_id,
        pattern_id=pattern_id,
        coordinate_dtype=coordinate_dtype,
        index_width=storage.index_width,
        plan_id=canonical_fingerprint(payload),
    )


def plan_pyamgcl(
    problem: LinearSystem,
    policy: PyAMGCLPolicy | None = None,
    /,
) -> PyAMGCLPlan:
    policy_ = PyAMGCLPolicy() if policy is None else policy
    if not isinstance(policy_, PyAMGCLPolicy):
        raise TypeError("policy must be PyAMGCLPolicy or None.")
    return PyAMGCLPlan(
        policy=policy_, **_plan_fields(problem, policy_.policy_id, "pyamgcl")
    )


def plan_amgx(
    problem: LinearSystem,
    policy: AmgXPolicy | None = None,
    /,
) -> AmgXPlan:
    from ..sparse._linear import (
        _SparseStoragePlan,
        SparseCoordinateOperator,
        SparseLinearMap,
    )

    policy_ = AmgXPolicy() if policy is None else policy
    if not isinstance(policy_, AmgXPolicy):
        raise TypeError("policy must be AmgXPolicy or None.")
    operator = problem.operator
    if (
        isinstance(operator, SparseCoordinateOperator)
        and operator._storage_plan is not None
    ):
        storage_plan = operator._storage_plan
    elif isinstance(operator, (SparseCoordinateOperator, SparseLinearMap)):
        storage_plan = _SparseStoragePlan(
            operator.relation,
            block_shape=operator.block_shape
            if isinstance(operator, SparseCoordinateOperator)
            else None,
        )
    else:
        storage_plan = None
    return AmgXPlan(
        policy=policy_,
        storage_plan=storage_plan,
        **_plan_fields(problem, policy_.policy_id, "amgx", storage_plan=storage_plan),
    )


def _validate_plan_problem(
    plan: Any, problem: LinearSystem, /, *, check_pattern: bool = True
) -> SparseStorage:
    storage = _storage(
        problem, storage_plan=plan.storage_plan if isinstance(plan, AmgXPlan) else None
    )
    if (
        storage.shape != plan.shape
        or problem.operator.source.space_id != plan.source_space_id
        or problem.operator.target.space_id != plan.target_space_id
        or problem.operator.operator_id != plan.operator_id
    ):
        raise ValueError(
            "AMG numeric binding must preserve its symbolic operator contract."
        )
    if check_pattern and _pattern_id(storage) != plan.pattern_id:
        raise ValueError("AMG numeric refresh requires an unchanged CSR pattern.")
    if np.dtype(storage.values.dtype).name != plan.coordinate_dtype:
        raise TypeError("AMG numeric refresh must preserve coordinate dtype.")
    return storage


def _scipy_csr(storage: SparseStorage, /) -> sp.csr_matrix:
    import scipy.sparse as sp

    return sp.csr_matrix(
        (
            np.asarray(storage.values),
            np.asarray(storage.indices),
            np.asarray(storage.indptr),
        ),
        shape=storage.shape,
    )


def _version(value: Any, /) -> Array:
    version = jnp.asarray(value, dtype=jnp.int32)
    if version.shape != ():
        raise ValueError("AMG numeric_version must be scalar.")
    return eqx.error_if(version, version < 0, "AMG numeric_version must be nonnegative.")


def prepare_pyamgcl(
    problem: LinearSystem,
    plan: PyAMGCLPlan,
    /,
    *,
    numeric_version: Any = 0,
) -> PreparedPyAMGCL:
    if not isinstance(plan, PyAMGCLPlan):
        raise TypeError("plan must be PyAMGCLPlan.")
    storage = _validate_plan_problem(plan, problem)
    availability = PyAMGCLBackend().availability()
    module = import_backend_module(availability, "linear-system", "pyamgcl")
    matrix = _scipy_csr(storage)
    solver = module.make_solver(
        matrix,
        solver=plan.policy.solver,
        prm=_config_dict(plan.policy.config),
    )
    return PreparedPyAMGCL(
        problem=problem,
        plan=plan,
        numeric_version=_version(numeric_version),
        transfers=BackendTransferEvidence(),
        runtime=_PyAMGCLRuntime(module, matrix, solver),
        prepared_id=canonical_fingerprint(
            {"kind": "prepared-pyamgcl", "plan": plan.plan_id}
        ),
    )


def _amgx_attributes(plan: AmgXPlan, partition: int, /) -> dict[str, Any]:
    return {
        "partition": np.int64(partition),
        "capacity": np.int64(plan.policy.hierarchy_cache_capacity),
        "config": json.dumps(
            _config_dict(plan.policy.config), sort_keys=True, separators=(",", ":")
        ),
    }


def _amgx_storage(storage: SparseStorage, /) -> tuple[Array, Array, Array]:
    if max(*storage.shape, storage.nnz) > np.iinfo(np.int32).max:
        raise ValueError("AmgX dDDI requires CSR dimensions and capacity to fit int32.")
    values = storage.values.astype(jnp.float64)
    if values.dtype != jnp.float64:
        raise ValueError("AmgX dDDI requires JAX float64 support to be enabled.")
    return (
        storage.indptr.astype(jnp.int32),
        storage.indices.astype(jnp.int32),
        values,
    )


def prepare_amgx(
    problem: LinearSystem,
    plan: AmgXPlan,
    /,
    *,
    numeric_version: Any = 0,
) -> PreparedAmgX:
    from .._cuda_native import cuda_partition_key, load_cuda_component

    if not isinstance(plan, AmgXPlan):
        raise TypeError("plan must be AmgXPlan.")
    storage = _validate_plan_problem(plan, problem)
    indptr, indices, values = _amgx_storage(storage)
    version = _version(numeric_version)
    load_cuda_component("amgx")
    partition = cuda_partition_key(plan.plan_id)
    native_id = jax.ffi.ffi_call(
        "phydrax_amgx_setup",
        jax.ShapeDtypeStruct((1,), jnp.int32),
        has_side_effect=True,
        vmap_method="sequential",
    )(indptr, indices, values, **_amgx_attributes(plan, partition))
    return PreparedAmgX(
        problem=problem,
        plan=plan,
        numeric_version=version,
        transfers=BackendTransferEvidence(synchronization_count=2),
        native_id=native_id,
        indptr=indptr,
        indices=indices,
        values=values,
        native_rebuilds=jnp.asarray(0, dtype=jnp.int32),
        partition=partition,
        prepared_id=canonical_fingerprint(
            {"kind": "prepared-amgx", "plan": plan.plan_id}
        ),
    )


def refresh_pyamgcl(
    prepared: PreparedPyAMGCL,
    problem: LinearSystem,
    /,
) -> PreparedPyAMGCL:
    if not isinstance(prepared, PreparedPyAMGCL):
        raise TypeError("prepared must be PreparedPyAMGCL.")
    return prepare_pyamgcl(
        problem,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def refresh_amgx(prepared: PreparedAmgX, problem: LinearSystem, /) -> PreparedAmgX:
    if not isinstance(prepared, PreparedAmgX):
        raise TypeError("prepared must be PreparedAmgX.")
    from .._cuda_native import load_cuda_component

    storage = _validate_plan_problem(prepared.plan, problem, check_pattern=False)
    indptr, indices, values = _amgx_storage(storage)
    if indptr.shape != prepared.indptr.shape or indices.shape != prepared.indices.shape:
        raise ValueError("AMG numeric refresh requires an unchanged CSR pattern.")
    values = eqx.error_if(
        values,
        jnp.any(indptr != prepared.indptr) | jnp.any(indices != prepared.indices),
        "AMG numeric refresh requires an unchanged CSR pattern.",
    )
    load_cuda_component("amgx")
    native_id, rebuilt = jax.ffi.ffi_call(
        "phydrax_amgx_refresh",
        (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        has_side_effect=True,
        vmap_method="sequential",
    )(
        prepared.native_id,
        indptr,
        indices,
        values,
        **_amgx_attributes(prepared.plan, prepared.partition),
    )
    return PreparedAmgX(
        problem=problem,
        plan=prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        transfers=BackendTransferEvidence(synchronization_count=2),
        native_id=native_id,
        indptr=indptr,
        indices=indices,
        values=values,
        native_rebuilds=rebuilt[0],
        partition=prepared.partition,
        prepared_id=prepared.prepared_id,
    )


def _pack_rhs(
    space: AbstractVectorSpace, value: PyTree[Any], /
) -> tuple[Array, tuple[int, ...], PyTreeDef, tuple[jax.ShapeDtypeStruct, ...]]:
    leaves, treedef = jax.tree.flatten(value)
    specifications, expected_treedef = jax.tree.flatten(space.structure())
    if treedef != expected_treedef or len(leaves) != len(specifications):
        raise ValueError(
            "Right-hand side PyTree does not match the operator target space."
        )
    arrays = tuple(jnp.asarray(leaf) for leaf in leaves)
    rhs_shape: tuple[int, ...] | None = None
    columns: list[Array] = []
    for array, specification in zip(arrays, specifications, strict=True):
        event_shape = tuple(specification.shape)
        if array.shape[: len(event_shape)] != event_shape:
            raise ValueError("Right-hand-side leaves must begin with their space shape.")
        suffix = tuple(array.shape[len(event_shape) :])
        if rhs_shape is None:
            rhs_shape = suffix
        elif rhs_shape != suffix:
            raise ValueError("Every right-hand-side leaf must share the same RHS axes.")
        if np.dtype(array.dtype) != np.dtype(specification.dtype):
            raise TypeError("Right-hand-side dtype must match the canonical space dtype.")
        count = prod(event_shape) if event_shape else 1
        rhs_count = prod(suffix) if suffix else 1
        columns.append(array.reshape((count, rhs_count)))
    coordinates = columns[0] if len(columns) == 1 else jnp.concatenate(columns, axis=0)
    return coordinates, rhs_shape or (), expected_treedef, tuple(specifications)


def _unpack_rhs(
    coordinates: Array,
    rhs_shape: tuple[int, ...],
    treedef: PyTreeDef,
    specifications: tuple[jax.ShapeDtypeStruct, ...],
    /,
) -> PyTree[Array]:
    leaves: list[Array] = []
    offset = 0
    for specification in specifications:
        count = prod(specification.shape) if specification.shape else 1
        leaf = coordinates[offset : offset + count].reshape(
            tuple(specification.shape) + tuple(rhs_shape)
        )
        leaves.append(leaf.astype(specification.dtype))
        offset += count
    return jax.tree.unflatten(treedef, leaves)


def _initial_coordinates(
    space: AbstractVectorSpace,
    initial_guess: PyTree[Any] | None,
    rhs_shape: tuple[int, ...],
    /,
) -> Array | None:
    if initial_guess is None:
        return None
    coordinates, guess_shape, _, _ = _pack_rhs(space, initial_guess)
    if guess_shape != rhs_shape:
        raise ValueError("Initial guess and right-hand side RHS axes must match.")
    return coordinates


def _provider_metadata(
    solver: Any, output: Any, /
) -> tuple[int | None, tuple[str | None, ...]]:
    iterations = None
    reasons: tuple[str | None, ...] = ()
    info = output[1] if isinstance(output, tuple) and len(output) > 1 else None
    if isinstance(info, Mapping):
        if "iterations" in info:
            iterations = int(info["iterations"])
        reason = info.get("reason")
        reasons = (None if reason is None else str(reason),)
    elif isinstance(info, (int, np.integer)):
        iterations = int(info)
    if iterations is None and hasattr(solver, "get_iterations_number"):
        iterations = int(solver.get_iterations_number())
    if not reasons and hasattr(solver, "get_status"):
        reasons = (str(solver.get_status()),)
    return iterations, reasons


def _residual_diagnostics(
    problem: LinearSystem,
    rhs: Array,
    solution: Array,
    rhs_shape: tuple[int, ...],
    /,
) -> tuple[Array, Array]:
    count = rhs.shape[1]
    residuals = []
    relatives = []
    for index in range(count):
        value = problem.operator.source.unflatten(jnp.asarray(solution[:, index]))
        image = problem.operator.target.flatten(problem.operator.mv(value))
        residual = rhs[:, index] - image
        residual_norm = jnp.linalg.norm(residual)
        residuals.append(residual_norm)
        relatives.append(
            residual_norm / jnp.maximum(jnp.linalg.norm(rhs[:, index]), 1.0e-30)
        )
    target_shape = tuple(rhs_shape)
    residual_array = jnp.stack(residuals).reshape(target_shape or ())
    relative_array = jnp.stack(relatives).reshape(target_shape or ())
    return residual_array, relative_array


def _configured_tolerance(policy: Any, backend: AMGBackendName, /) -> float | None:
    config = _config_dict(policy.config)
    solver = config.get("solver")
    if not isinstance(solver, Mapping):
        return None
    key = "tol" if backend == "pyamgcl" else "tolerance"
    value = solver.get(key)
    return None if value is None else float(value)


def _solve_status(
    coordinates: Array,
    relative_residual: Array,
    rhs_shape: tuple[int, ...],
    tolerance: float | None,
    /,
) -> Array:
    finite = jnp.all(jnp.isfinite(coordinates), axis=0).reshape(rhs_shape or ())
    converged = (
        jnp.ones_like(finite) if tolerance is None else relative_residual <= tolerance
    )
    return jnp.where(
        ~finite,
        int(AMGSolveStatus.NONFINITE),
        jnp.where(
            converged,
            int(AMGSolveStatus.SUCCESS),
            int(AMGSolveStatus.PROVIDER_FAILURE),
        ),
    ).astype(jnp.int32)


def solve_pyamgcl(
    prepared: PreparedPyAMGCL,
    right_hand_side: PyTree[Any],
    /,
    *,
    initial_guess: PyTree[Any] | None = None,
) -> AMGSolveResult:
    if not isinstance(prepared, PreparedPyAMGCL):
        raise TypeError("prepared must be PreparedPyAMGCL.")
    space = prepared.problem.operator.target
    rhs, rhs_shape, treedef, specifications = _pack_rhs(space, right_hand_side)
    guess = _initial_coordinates(
        prepared.problem.operator.source, initial_guess, rhs_shape
    )
    solutions = []
    iteration_values = []
    reasons: list[str | None] = []
    for index in range(rhs.shape[1]):
        rhs_column = np.asarray(rhs[:, index])
        output = (
            prepared.runtime.solver(rhs_column)
            if guess is None
            else prepared.runtime.solver(rhs_column, np.asarray(guess[:, index]))
        )
        solution = output[0] if isinstance(output, tuple) else output
        solutions.append(np.asarray(solution))
        iterations, provider_reasons = _provider_metadata(prepared.runtime.solver, output)
        if iterations is not None:
            iteration_values.append(iterations)
        reasons.append(provider_reasons[0] if provider_reasons else None)
    coordinates = jnp.asarray(np.stack(solutions, axis=1), dtype=rhs.dtype)
    residual, relative = _residual_diagnostics(
        prepared.problem, rhs, coordinates, rhs_shape
    )
    status = _solve_status(
        coordinates,
        relative,
        rhs_shape,
        _configured_tolerance(prepared.plan.policy, "pyamgcl"),
    )
    iterations = (
        None
        if len(iteration_values) != rhs.shape[1]
        else jnp.asarray(iteration_values, dtype=jnp.int32).reshape(rhs_shape or ())
    )
    return AMGSolveResult(
        value=_unpack_rhs(coordinates, rhs_shape, treedef, specifications),
        status=status,
        diagnostics=AMGSolveDiagnostics(
            residual_norm=residual,
            relative_residual_norm=relative,
            iterations=iterations,
            provider_reasons=tuple(reasons),
        ),
        provenance=AMGProvenance(
            numeric_version=prepared.numeric_version,
            backend="pyamgcl",
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
        ),
        transfers=BackendTransferEvidence(),
    )


def solve_amgx(
    prepared: PreparedAmgX,
    right_hand_side: PyTree[Any],
    /,
    *,
    initial_guess: PyTree[Any] | None = None,
) -> AMGSolveResult:
    from .._cuda_native import load_cuda_component

    if not isinstance(prepared, PreparedAmgX):
        raise TypeError("prepared must be PreparedAmgX.")
    space = prepared.problem.operator.target
    rhs, rhs_shape, treedef, specifications = _pack_rhs(space, right_hand_side)
    guess = _initial_coordinates(
        prepared.problem.operator.source, initial_guess, rhs_shape
    )
    native_rhs = rhs.T.astype(jnp.float64)
    native_guess = (
        jnp.zeros_like(native_rhs) if guess is None else guess.T.astype(jnp.float64)
    )
    count = native_rhs.shape[0]
    load_cuda_component("amgx")
    solution, native_status, iterations, native_residual, rebuilt = jax.ffi.ffi_call(
        "phydrax_amgx_solve",
        (
            jax.ShapeDtypeStruct(native_rhs.shape, jnp.float64),
            jax.ShapeDtypeStruct((count,), jnp.int32),
            jax.ShapeDtypeStruct((count,), jnp.int32),
            jax.ShapeDtypeStruct((count,), jnp.float64),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ),
        has_side_effect=False,
        vmap_method="sequential",
    )(
        prepared.native_id,
        prepared.indptr,
        prepared.indices,
        prepared.values,
        native_rhs,
        native_guess,
        **_amgx_attributes(prepared.plan, prepared.partition),
    )
    coordinates = solution.T.astype(rhs.dtype)
    residual, relative = _residual_diagnostics(
        prepared.problem, rhs, coordinates, rhs_shape
    )
    status = _solve_status(
        coordinates,
        relative,
        rhs_shape,
        _configured_tolerance(prepared.plan.policy, "amgx"),
    )
    provider_status = native_status.reshape(rhs_shape or ())
    status = jnp.where(
        (status == int(AMGSolveStatus.SUCCESS)) & (provider_status != 0),
        int(AMGSolveStatus.PROVIDER_FAILURE),
        status,
    ).astype(jnp.int32)
    return AMGSolveResult(
        value=_unpack_rhs(coordinates, rhs_shape, treedef, specifications),
        status=status,
        diagnostics=AMGSolveDiagnostics(
            residual_norm=residual,
            relative_residual_norm=relative,
            iterations=iterations.reshape(rhs_shape or ()),
            provider_reasons=(),
            native_status=provider_status,
            native_residual_norm=native_residual.reshape(rhs_shape or ()),
            native_rebuilds=rebuilt[0],
        ),
        provenance=AMGProvenance(
            numeric_version=prepared.numeric_version,
            backend="amgx",
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            promoted_to_float64=prepared.plan.coordinate_dtype == "float32",
        ),
        transfers=BackendTransferEvidence(synchronization_count=2),
    )


def release_amgx(prepared: PreparedAmgX, /) -> None:
    """Idempotently free a memo entry; later use rebuilds from preparation leaves."""
    from .._cuda_native import release_cuda_entries

    if not isinstance(prepared, PreparedAmgX):
        raise TypeError("prepared must be PreparedAmgX.")
    # The id can be a closed-over eager array even when release is being traced.
    if isinstance(jnp.asarray(0, dtype=jnp.int32), Tracer):
        jax.ffi.ffi_call(
            "phydrax_amgx_release",
            jax.ShapeDtypeStruct((1,), jnp.int32),
            has_side_effect=True,
            vmap_method="sequential",
        )(prepared.native_id, partition=np.int64(prepared.partition))
    else:
        release_cuda_entries("amgx", prepared.partition, prepared.native_id)


__all__ = [
    "AMGX_CAPABILITIES",
    "PYAMGCL_CAPABILITIES",
    "AMGBackendName",
    "AMGProvenance",
    "AMGSolveDiagnostics",
    "AMGSolveResult",
    "AMGSolveStatus",
    "AmgXBackend",
    "AmgXPlan",
    "AmgXPolicy",
    "PreparedAmgX",
    "PreparedPyAMGCL",
    "PyAMGCLBackend",
    "PyAMGCLPlan",
    "PyAMGCLPolicy",
    "amgx_availability",
    "pyamgcl_availability",
    "plan_amgx",
    "plan_pyamgcl",
    "prepare_amgx",
    "prepare_pyamgcl",
    "refresh_amgx",
    "refresh_pyamgcl",
    "release_amgx",
    "solve_amgx",
    "solve_pyamgcl",
]
