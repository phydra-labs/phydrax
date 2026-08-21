#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from math import isfinite, prod
from threading import Lock
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg._problems import LinearSystem
from ..linalg._sparse_contract import AbstractSparseLinearOperator, SparseStorage
from ._availability import import_backend_module, probe_backend
from ._types import (
    AbstractExternalBackend,
    BackendAvailability,
    BackendCapabilities,
    BackendTransferEvidence,
)


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
    requires_explicit_release=True,
)


_DEFAULT_PYAMGCL_CONFIG = {
    "solver": {"tol": 1.0e-8, "maxiter": 100},
}
_DEFAULT_AMGX_CONFIG = {
    "config_version": 2,
    "solver": {
        "solver": "FGMRES",
        "preconditioner": {"solver": "AMG", "algorithm": "AGGREGATION"},
        "max_iters": 100,
        "tolerance": 1.0e-8,
        "norm": "L2",
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
    ):
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
    policy_id: str = eqx.field(static=True)

    def __init__(self, config: Mapping[str, Any] | None = None, /):
        config_ = _canonical_config(_DEFAULT_AMGX_CONFIG if config is None else config)
        self.config = config_
        self.policy_id = canonical_fingerprint({"kind": "amgx-policy", "config": config_})


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


class AMGProvenance(StrictModule):
    """Symbolic and numeric identity of one external AMG result."""

    numeric_version: Array
    backend: AMGBackendName = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


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
    shape: tuple[int, int] = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    pattern_id: str = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    index_width: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class _PyAMGCLRuntime:
    def __init__(self, module: Any, matrix: Any, solver: Any):
        self.module = module
        self.matrix = matrix
        self.solver = solver


class _AmgXRuntime:
    def __init__(
        self,
        module: Any,
        config: Any,
        resources: Any,
        matrix: Any,
        solver: Any,
    ):
        self.module = module
        self.config = config
        self.resources = resources
        self.matrix = matrix
        self.solver = solver
        self.released = False


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
    runtime: Any = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def released(self) -> bool:
        return bool(self.runtime.released)


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

    def plan(self, problem: LinearSystem, policy: PyAMGCLPolicy | None = None, /):
        return plan_pyamgcl(problem, policy)

    def prepare(self, problem: LinearSystem, plan: PyAMGCLPlan, /):
        return prepare_pyamgcl(problem, plan)

    def solve(self, prepared: PreparedPyAMGCL, right_hand_side: PyTree[Any], /, **kwargs):
        return solve_pyamgcl(prepared, right_hand_side, **kwargs)

    def refresh(self, prepared: PreparedPyAMGCL, problem: LinearSystem, /):
        return refresh_pyamgcl(prepared, problem)


class AmgXBackend(AbstractExternalBackend):
    @property
    def name(self) -> str:
        return "amgx"

    @property
    def capabilities(self) -> BackendCapabilities:
        return AMGX_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return probe_backend(
            self.capabilities,
            module="pyamgx",
            requirement="install NVIDIA AmgX and the optional pyamgx bindings",
            distributions=("pyamgx",),
        )

    def plan(self, problem: LinearSystem, policy: AmgXPolicy | None = None, /):
        return plan_amgx(problem, policy)

    def prepare(self, problem: LinearSystem, plan: AmgXPlan, /):
        return prepare_amgx(problem, plan)

    def solve(self, prepared: PreparedAmgX, right_hand_side: PyTree[Any], /, **kwargs):
        return solve_amgx(prepared, right_hand_side, **kwargs)

    def refresh(self, prepared: PreparedAmgX, problem: LinearSystem, /):
        return refresh_amgx(prepared, problem)

    def release(self, prepared: PreparedAmgX, /) -> None:
        release_amgx(prepared)


def pyamgcl_availability() -> BackendAvailability:
    """Probe PyAMGCL without retaining or importing it at package import time."""
    return PyAMGCLBackend().availability()


def amgx_availability() -> BackendAvailability:
    """Probe pyamgx without retaining or importing it at package import time."""
    return AmgXBackend().availability()


def _storage(problem: LinearSystem, /) -> SparseStorage:
    if not isinstance(problem, LinearSystem):
        raise TypeError("AMG backends require a LinearSystem.")
    operator = problem.operator
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError("AMG backends require AbstractSparseLinearOperator storage.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("AMG backends require an unbatched square endomorphism.")
    storage = operator.sparse_storage()
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


def _plan(problem: LinearSystem, policy: Any, backend: AMGBackendName, /):
    storage = _storage(problem)
    payload = {
        "kind": f"{backend}-plan",
        "policy": policy.policy_id,
        "shape": storage.shape,
        "source": problem.operator.source.space_id,
        "target": problem.operator.target.space_id,
        "operator": problem.operator.operator_id,
        "pattern": _pattern_id(storage),
        "dtype": np.dtype(storage.values.dtype).name,
        "index_width": storage.index_width,
    }
    fields = dict(
        policy=policy,
        shape=storage.shape,
        source_space_id=problem.operator.source.space_id,
        target_space_id=problem.operator.target.space_id,
        operator_id=problem.operator.operator_id,
        pattern_id=payload["pattern"],
        coordinate_dtype=payload["dtype"],
        index_width=storage.index_width,
        plan_id=canonical_fingerprint(payload),
    )
    return PyAMGCLPlan(**fields) if backend == "pyamgcl" else AmgXPlan(**fields)


def plan_pyamgcl(
    problem: LinearSystem,
    policy: PyAMGCLPolicy | None = None,
    /,
) -> PyAMGCLPlan:
    policy_ = PyAMGCLPolicy() if policy is None else policy
    if not isinstance(policy_, PyAMGCLPolicy):
        raise TypeError("policy must be PyAMGCLPolicy or None.")
    return _plan(problem, policy_, "pyamgcl")


def plan_amgx(
    problem: LinearSystem,
    policy: AmgXPolicy | None = None,
    /,
) -> AmgXPlan:
    policy_ = AmgXPolicy() if policy is None else policy
    if not isinstance(policy_, AmgXPolicy):
        raise TypeError("policy must be AmgXPolicy or None.")
    return _plan(problem, policy_, "amgx")


def _validate_plan_problem(plan: Any, problem: LinearSystem, /) -> SparseStorage:
    storage = _storage(problem)
    if (
        storage.shape != plan.shape
        or problem.operator.source.space_id != plan.source_space_id
        or problem.operator.target.space_id != plan.target_space_id
        or problem.operator.operator_id != plan.operator_id
    ):
        raise ValueError(
            "AMG numeric binding must preserve its symbolic operator contract."
        )
    if _pattern_id(storage) != plan.pattern_id:
        raise ValueError("AMG numeric refresh requires an unchanged CSR pattern.")
    if np.dtype(storage.values.dtype).name != plan.coordinate_dtype:
        raise TypeError("AMG numeric refresh must preserve coordinate dtype.")
    return storage


def _scipy_csr(storage: SparseStorage, /):
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


_AMGX_LOCK = Lock()
_AMGX_REFERENCES: dict[int, int] = {}


def _acquire_amgx(module: Any, /) -> None:
    key = id(module)
    with _AMGX_LOCK:
        count = _AMGX_REFERENCES.get(key, 0)
        if count == 0:
            module.initialize()
        _AMGX_REFERENCES[key] = count + 1


def _release_amgx_module(module: Any, /) -> None:
    key = id(module)
    with _AMGX_LOCK:
        count = _AMGX_REFERENCES.get(key, 0)
        if count <= 1:
            _AMGX_REFERENCES.pop(key, None)
            module.finalize()
        else:
            _AMGX_REFERENCES[key] = count - 1


def prepare_amgx(
    problem: LinearSystem,
    plan: AmgXPlan,
    /,
    *,
    numeric_version: Any = 0,
) -> PreparedAmgX:
    if not isinstance(plan, AmgXPlan):
        raise TypeError("plan must be AmgXPlan.")
    storage = _validate_plan_problem(plan, problem)
    availability = AmgXBackend().availability()
    module = import_backend_module(availability, "linear-system", "pyamgx")
    _acquire_amgx(module)
    config = resources = matrix = solver = None
    try:
        config = module.Config().create_from_dict(_config_dict(plan.policy.config))
        resources = module.Resources().create_simple(config)
        matrix = module.Matrix().create(resources)
        matrix.upload_CSR(_scipy_csr(storage))
        solver = module.Solver().create(resources, config)
        solver.setup(matrix)
    except Exception:
        for resource in (solver, matrix, resources, config):
            if resource is not None:
                resource.destroy()
        _release_amgx_module(module)
        raise
    matrix_bytes = int(
        storage.values.nbytes + storage.indices.nbytes + storage.indptr.nbytes
    )
    return PreparedAmgX(
        problem=problem,
        plan=plan,
        numeric_version=_version(numeric_version),
        transfers=BackendTransferEvidence(
            host_to_device_bytes=matrix_bytes,
        ),
        runtime=_AmgXRuntime(module, config, resources, matrix, solver),
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
    if prepared.released:
        raise ValueError("Cannot refresh a released AmgX preparation.")
    return prepare_amgx(
        problem,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def _pack_rhs(space: Any, value: PyTree[Any], /):
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
        suffix = tuple(int(size) for size in array.shape[len(event_shape) :])
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


def _unpack_rhs(coordinates: Array, rhs_shape, treedef, specifications, /):
    leaves = []
    offset = 0
    for specification in specifications:
        count = prod(specification.shape) if specification.shape else 1
        leaf = coordinates[offset : offset + count].reshape(
            tuple(specification.shape) + tuple(rhs_shape)
        )
        leaves.append(leaf.astype(specification.dtype))
        offset += count
    return jax.tree.unflatten(treedef, leaves)


def _initial_coordinates(space, initial_guess, rhs_shape, /):
    if initial_guess is None:
        return None
    coordinates, guess_shape, _, _ = _pack_rhs(space, initial_guess)
    if guess_shape != rhs_shape:
        raise ValueError("Initial guess and right-hand side RHS axes must match.")
    return coordinates


def _provider_metadata(solver: Any, output: Any, /):
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


def _residual_diagnostics(problem, rhs, solution, rhs_shape, /):
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
    if not isinstance(prepared, PreparedAmgX):
        raise TypeError("prepared must be PreparedAmgX.")
    if prepared.released:
        raise ValueError("Cannot solve with a released AmgX preparation.")
    space = prepared.problem.operator.target
    rhs, rhs_shape, treedef, specifications = _pack_rhs(space, right_hand_side)
    guess = _initial_coordinates(
        prepared.problem.operator.source, initial_guess, rhs_shape
    )
    solutions = []
    iteration_values = []
    reasons: list[str | None] = []
    for index in range(rhs.shape[1]):
        rhs_vector = prepared.runtime.module.Vector().create(prepared.runtime.resources)
        solution_vector = prepared.runtime.module.Vector().create(
            prepared.runtime.resources
        )
        try:
            rhs_vector.upload(np.asarray(rhs[:, index]))
            initial = (
                np.zeros((rhs.shape[0],), dtype=np.dtype(rhs.dtype))
                if guess is None
                else np.asarray(guess[:, index])
            )
            solution_vector.upload(initial)
            prepared.runtime.solver.solve(rhs_vector, solution_vector)
            solutions.append(np.asarray(solution_vector.download()))
            iterations, provider_reasons = _provider_metadata(
                prepared.runtime.solver, None
            )
            if iterations is not None:
                iteration_values.append(iterations)
            reasons.append(provider_reasons[0] if provider_reasons else None)
        finally:
            solution_vector.destroy()
            rhs_vector.destroy()
    coordinates = jnp.asarray(np.stack(solutions, axis=1), dtype=rhs.dtype)
    residual, relative = _residual_diagnostics(
        prepared.problem, rhs, coordinates, rhs_shape
    )
    status = _solve_status(
        coordinates,
        relative,
        rhs_shape,
        _configured_tolerance(prepared.plan.policy, "amgx"),
    )
    iterations = (
        None
        if len(iteration_values) != rhs.shape[1]
        else jnp.asarray(iteration_values, dtype=jnp.int32).reshape(rhs_shape or ())
    )
    bytes_per_columns = int(rhs.nbytes)
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
            backend="amgx",
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
        ),
        transfers=BackendTransferEvidence(
            host_to_device_bytes=2 * bytes_per_columns,
            device_to_host_bytes=bytes_per_columns,
            synchronization_count=rhs.shape[1],
        ),
    )


def release_amgx(prepared: PreparedAmgX, /) -> None:
    """Idempotently destroy one AmgX hierarchy and release global state."""
    if not isinstance(prepared, PreparedAmgX):
        raise TypeError("prepared must be PreparedAmgX.")
    runtime = prepared.runtime
    if runtime.released:
        return
    runtime.released = True
    failure: Exception | None = None
    for resource in (
        runtime.solver,
        runtime.matrix,
        runtime.resources,
        runtime.config,
    ):
        try:
            resource.destroy()
        except Exception as error:
            if failure is None:
                failure = error
    try:
        _release_amgx_module(runtime.module)
    except Exception as error:
        if failure is None:
            failure = error
    if failure is not None:
        raise failure


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
