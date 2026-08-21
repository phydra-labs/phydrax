#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    LinearSolveStatus,
    LinearSystem,
    PyTreeSpace,
)
from ..linalg._sparse_contract import AbstractSparseLinearOperator, SparseStorage
from ..nonlinear import NonlinearStatus, NonlinearSystemProblem
from ._availability import import_backend_module, probe_backend
from ._types import (
    AbstractExternalBackend,
    BackendAvailability,
    BackendCapabilities,
    BackendTransferEvidence,
)


PETScJacobianMode: TypeAlias = Literal["matrix-free", "dense-autodiff"]
PETScOptionValue: TypeAlias = str | int | float | bool | None

_CAPABILITIES = BackendCapabilities(
    backend="petsc",
    problem_kinds=("linear-system", "nonlinear-system"),
    execution="host",
    host_only=True,
    supports_matrix_free=True,
    supports_assembled=True,
    coordinate_dtypes=("float32", "float64", "complex64", "complex128"),
)
_REQUIREMENT = "petsc4py with a working PETSc runtime"


def _normalize_options(
    values: Mapping[str, PETScOptionValue] | Sequence[tuple[str, PETScOptionValue]], /
) -> tuple[tuple[str, str | None], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    result: list[tuple[str, str | None]] = []
    names: set[str] = set()
    for name, value in items:
        name_ = str(name).strip().lstrip("-")
        if not name_ or name_ in names:
            raise ValueError("PETSc option names must be non-empty and unique.")
        names.add(name_)
        result.append(
            (
                name_,
                None
                if value is None or value is True
                else "false"
                if value is False
                else str(value),
            )
        )
    return tuple(result)


class PETScKSPPolicy(StrictModule):
    """Explicit PETSc KSP, PC, tolerance, options, and reuse policy."""

    ksp_type: str = eqx.field(static=True)
    pc_type: str = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    divergence_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    reuse_preconditioner: bool = eqx.field(static=True)
    options: tuple[tuple[str, str | None], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        ksp_type: str = "gmres",
        pc_type: str = "none",
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-12,
        divergence_tolerance: float = 1e8,
        maximum_iterations: int = 1_000,
        reuse_preconditioner: bool = False,
        options: Mapping[str, PETScOptionValue]
        | Sequence[tuple[str, PETScOptionValue]] = (),
    ):
        ksp, pc = str(ksp_type), str(pc_type)
        relative, absolute, divergence = (
            float(relative_tolerance),
            float(absolute_tolerance),
            float(divergence_tolerance),
        )
        maximum = int(maximum_iterations)
        if not ksp or not pc:
            raise ValueError("PETSc KSP and PC types must be non-empty.")
        if any(not math.isfinite(value) or value < 0.0 for value in (relative, absolute)):
            raise ValueError("PETSc KSP tolerances must be finite and non-negative.")
        if not math.isfinite(divergence) or divergence <= 1.0 or maximum < 1:
            raise ValueError(
                "PETSc KSP divergence tolerance and maximum iterations are invalid."
            )
        self.ksp_type, self.pc_type = ksp, pc
        self.relative_tolerance, self.absolute_tolerance = relative, absolute
        self.divergence_tolerance, self.maximum_iterations = divergence, maximum
        self.reuse_preconditioner = bool(reuse_preconditioner)
        self.options = _normalize_options(options)


class PETScSNESPolicy(StrictModule):
    """Explicit PETSc SNES derivative policy and inner KSP contract."""

    jacobian_mode: PETScJacobianMode = eqx.field(static=True)
    snes_type: str = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    step_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    maximum_function_evaluations: int = eqx.field(static=True)
    maximum_dense_dimension: int = eqx.field(static=True)
    maximum_dense_bytes: int = eqx.field(static=True)
    ksp: PETScKSPPolicy
    options: tuple[tuple[str, str | None], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        jacobian_mode: PETScJacobianMode = "matrix-free",
        snes_type: str = "newtonls",
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-10,
        step_tolerance: float = 1e-12,
        maximum_iterations: int = 100,
        maximum_function_evaluations: int = 10_000,
        maximum_dense_dimension: int = 2_048,
        maximum_dense_bytes: int = 256 * 1024 * 1024,
        ksp: PETScKSPPolicy | None = None,
        options: Mapping[str, PETScOptionValue]
        | Sequence[tuple[str, PETScOptionValue]] = (),
    ):
        if jacobian_mode not in ("matrix-free", "dense-autodiff"):
            raise ValueError("jacobian_mode must be 'matrix-free' or 'dense-autodiff'.")
        snes = str(snes_type)
        tolerances = tuple(
            float(value)
            for value in (relative_tolerance, absolute_tolerance, step_tolerance)
        )
        limits = (
            int(maximum_iterations),
            int(maximum_function_evaluations),
            int(maximum_dense_dimension),
            int(maximum_dense_bytes),
        )
        if (
            not snes
            or any(not math.isfinite(value) or value < 0.0 for value in tolerances)
            or min(limits) < 1
        ):
            raise ValueError(
                "PETSc SNES type, tolerances, and resource limits are invalid."
            )
        ksp_ = PETScKSPPolicy() if ksp is None else ksp
        if not isinstance(ksp_, PETScKSPPolicy):
            raise TypeError("ksp must be a PETScKSPPolicy.")
        self.jacobian_mode, self.snes_type = jacobian_mode, snes
        self.relative_tolerance, self.absolute_tolerance, self.step_tolerance = tolerances
        (
            self.maximum_iterations,
            self.maximum_function_evaluations,
            self.maximum_dense_dimension,
            self.maximum_dense_bytes,
        ) = limits
        self.ksp, self.options = ksp_, _normalize_options(options)


class PETScLinearPlan(StrictModule):
    problem: LinearSystem
    preconditioner_operator: AbstractSparseLinearOperator
    policy: PETScKSPPolicy
    operator_pattern: tuple[tuple[int, ...], tuple[int, ...]] = eqx.field(static=True)
    preconditioner_pattern: tuple[tuple[int, ...], tuple[int, ...]] = eqx.field(
        static=True
    )
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: LinearSystem,
        preconditioner_operator: AbstractSparseLinearOperator,
        policy: PETScKSPPolicy,
        /,
    ):
        operator_storage = _canonical_storage(problem.operator, role="system operator")
        preconditioner_storage = _canonical_storage(
            preconditioner_operator, role="preconditioner operator"
        )
        if preconditioner_storage.shape != operator_storage.shape:
            raise ValueError("PETSc Amat and Pmat dimensions must match.")
        self.problem, self.preconditioner_operator, self.policy = (
            problem,
            preconditioner_operator,
            policy,
        )
        self.operator_pattern = _pattern(operator_storage)
        self.preconditioner_pattern = _pattern(preconditioner_storage)
        self.plan_id = canonical_fingerprint(
            {
                "backend": "petsc-ksp",
                "problem": problem.problem_id,
                "amat": problem.operator.operator_id,
                "pmat": preconditioner_operator.operator_id,
                "amat_pattern": self.operator_pattern,
                "pmat_pattern": self.preconditioner_pattern,
                "policy": _ksp_payload(policy),
            }
        )


class PETScNonlinearPlan(StrictModule):
    problem: NonlinearSystemProblem
    initial_state: PyTree[Array]
    args: Any
    space: PyTreeSpace
    policy: PETScSNESPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any],
        policy: PETScSNESPolicy,
        /,
        *,
        args: Any = None,
    ):
        if not isinstance(problem, NonlinearSystemProblem) or not isinstance(
            policy, PETScSNESPolicy
        ):
            raise TypeError(
                "PETSc SNES requires a NonlinearSystemProblem and PETScSNESPolicy."
            )
        space = PyTreeSpace(initial_state)
        state = space.validate(initial_state)
        residual, _ = problem.evaluate(state, args)
        _require_matching_tree(space, residual, "nonlinear residual")
        if policy.jacobian_mode == "dense-autodiff":
            itemsize = np.dtype(jax.tree.leaves(space.structure())[0].dtype).itemsize
            required = space.size * space.size * itemsize
            if (
                space.size > policy.maximum_dense_dimension
                or required > policy.maximum_dense_bytes
            ):
                raise ValueError(
                    f"dense-autodiff requires dimension {space.size} and {required} "
                    "bytes, exceeding the explicit policy limit."
                )
        self.problem, self.initial_state, self.args, self.space, self.policy = (
            problem,
            state,
            args,
            space,
            policy,
        )
        self.plan_id = canonical_fingerprint(
            {
                "backend": "petsc-snes",
                "problem": problem.problem_id,
                "space": space.space_id,
                "policy": _snes_payload(policy),
            }
        )


class PreparedPETScLinearSolve(StrictModule):
    plan: PETScLinearPlan
    setup_transfer: BackendTransferEvidence
    petsc: Any = eqx.field(static=True)
    operator_matrix: Any = eqx.field(static=True)
    preconditioner_matrix: Any = eqx.field(static=True)
    solver: Any = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)

    def __init__(
        self,
        plan,
        setup_transfer,
        /,
        *,
        petsc,
        operator_matrix,
        preconditioner_matrix,
        solver,
        prepared_id,
        numeric_version=0,
    ):
        self.plan, self.setup_transfer, self.petsc = plan, setup_transfer, petsc
        self.operator_matrix, self.preconditioner_matrix, self.solver = (
            operator_matrix,
            preconditioner_matrix,
            solver,
        )
        self.prepared_id, self.numeric_version = str(prepared_id), int(numeric_version)
        if not self.prepared_id or self.numeric_version < 0:
            raise ValueError("PETSc prepared identity or numeric version is invalid.")


class PreparedPETScNonlinearSolve(StrictModule):
    plan: PETScNonlinearPlan
    setup_transfer: BackendTransferEvidence
    petsc: Any = eqx.field(static=True)
    solver: Any = eqx.field(static=True)
    residual_vector: Any = eqx.field(static=True)
    jacobian: Any = eqx.field(static=True)
    callbacks: tuple[Any, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)

    def __init__(
        self,
        plan,
        setup_transfer,
        /,
        *,
        petsc,
        solver,
        residual_vector,
        jacobian,
        callbacks,
        prepared_id,
        numeric_version=0,
    ):
        self.plan, self.setup_transfer, self.petsc, self.solver = (
            plan,
            setup_transfer,
            petsc,
            solver,
        )
        self.residual_vector, self.jacobian, self.callbacks = (
            residual_vector,
            jacobian,
            tuple(callbacks),
        )
        self.prepared_id, self.numeric_version = str(prepared_id), int(numeric_version)
        if not self.prepared_id or self.numeric_version < 0:
            raise ValueError("PETSc prepared identity or numeric version is invalid.")


class PETScLinearDiagnostics(StrictModule):
    residual_norm: Array
    relative_residual: Array
    iterations: Array
    convergence_reason: Array
    converged: Array

    def __init__(
        self,
        *,
        residual_norm,
        relative_residual,
        iterations,
        convergence_reason,
        converged,
    ):
        self.residual_norm, self.relative_residual = (
            jnp.asarray(residual_norm),
            jnp.asarray(relative_residual),
        )
        self.iterations, self.convergence_reason = (
            jnp.asarray(iterations, dtype=jnp.int32),
            jnp.asarray(convergence_reason, dtype=jnp.int32),
        )
        self.converged = jnp.asarray(converged, dtype=bool)


class PETScNonlinearDiagnostics(StrictModule):
    initial_residual_norm: Array
    final_residual_norm: Array
    iterations: Array
    function_evaluations: Array
    linear_iterations: Array
    convergence_reason: Array
    converged: Array

    def __init__(
        self,
        *,
        initial_residual_norm,
        final_residual_norm,
        iterations,
        function_evaluations,
        linear_iterations,
        convergence_reason,
        converged,
    ):
        self.initial_residual_norm, self.final_residual_norm = (
            jnp.asarray(initial_residual_norm),
            jnp.asarray(final_residual_norm),
        )
        self.iterations, self.function_evaluations, self.linear_iterations = (
            jnp.asarray(value, dtype=jnp.int32)
            for value in (iterations, function_evaluations, linear_iterations)
        )
        self.convergence_reason, self.converged = (
            jnp.asarray(convergence_reason, dtype=jnp.int32),
            jnp.asarray(converged, dtype=bool),
        )


class PETScProvenance(StrictModule):
    setup_transfer: BackendTransferEvidence
    solve_transfer: BackendTransferEvidence
    backend: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    preconditioner: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    preconditioner_operator_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)
    reused_preconditioner: bool = eqx.field(static=True)
    host_only: bool = eqx.field(static=True)
    jit_compatible: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        method,
        preconditioner,
        operator_id,
        preconditioner_operator_id,
        plan_id,
        prepared_id,
        problem_id,
        numeric_version,
        reused_preconditioner,
        setup_transfer,
        solve_transfer,
    ):
        values = tuple(
            str(value)
            for value in (
                method,
                preconditioner,
                operator_id,
                preconditioner_operator_id,
                plan_id,
                prepared_id,
                problem_id,
            )
        )
        if any(not value for value in values):
            raise ValueError("PETSc provenance identifiers must be non-empty.")
        (
            self.method,
            self.preconditioner,
            self.operator_id,
            self.preconditioner_operator_id,
            self.plan_id,
            self.prepared_id,
            self.problem_id,
        ) = values
        self.backend, self.numeric_version, self.reused_preconditioner = (
            "petsc",
            int(numeric_version),
            bool(reused_preconditioner),
        )
        self.host_only, self.jit_compatible, self.differentiable = True, False, False
        self.setup_transfer, self.solve_transfer = setup_transfer, solve_transfer


class PETScLinearResult(StrictModule):
    value: PyTree[Array]
    status: Array
    diagnostics: PETScLinearDiagnostics
    provenance: PETScProvenance

    def __init__(self, value, status, diagnostics, provenance, /):
        self.value, self.status, self.diagnostics, self.provenance = (
            value,
            jnp.asarray(status, dtype=jnp.int32),
            diagnostics,
            provenance,
        )

    @property
    def successful(self) -> Array:
        return self.status == int(LinearSolveStatus.SUCCESS)


class PETScNonlinearResult(StrictModule):
    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    status: Array
    diagnostics: PETScNonlinearDiagnostics
    provenance: PETScProvenance

    def __init__(self, *, state, residual, auxiliary, status, diagnostics, provenance):
        self.state, self.residual, self.auxiliary = state, residual, auxiliary
        self.status, self.diagnostics, self.provenance = (
            jnp.asarray(status, dtype=jnp.int32),
            diagnostics,
            provenance,
        )

    @property
    def successful(self) -> Array:
        return self.status == int(NonlinearStatus.SUCCESS)


class PETScBackend(AbstractExternalBackend):
    """Lazy host-only petsc4py KSP/SNES provider; it never materializes implicitly."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "petsc"

    @property
    def capabilities(self) -> BackendCapabilities:
        return _CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return probe_backend(
            self.capabilities,
            module="petsc4py.PETSc",
            requirement=_REQUIREMENT,
            distributions=("petsc4py",),
        )

    def plan_linear(
        self,
        problem: LinearSystem,
        policy: PETScKSPPolicy | None = None,
        /,
        *,
        preconditioner_operator: AbstractSparseLinearOperator | None = None,
    ) -> PETScLinearPlan:
        if not isinstance(problem, LinearSystem):
            raise TypeError("PETSc KSP requires a LinearSystem.")
        operator = problem.operator
        _canonical_storage(operator, role="system operator")
        pmat = operator if preconditioner_operator is None else preconditioner_operator
        return PETScLinearPlan(
            problem, pmat, PETScKSPPolicy() if policy is None else policy
        )

    def prepare_linear(self, plan: PETScLinearPlan, /) -> PreparedPETScLinearSolve:
        if not isinstance(plan, PETScLinearPlan):
            raise TypeError("plan must be a PETScLinearPlan.")
        petsc = import_backend_module(
            self.availability(), "linear-system", "petsc4py.PETSc"
        )
        astorage, pstorage = (
            _canonical_storage(plan.problem.operator, role="system operator"),
            _canonical_storage(
                plan.preconditioner_operator, role="preconditioner operator"
            ),
        )
        amat = _create_matrix(petsc, astorage)
        pmat = (
            amat
            if plan.problem.operator is plan.preconditioner_operator
            else _create_matrix(petsc, pstorage)
        )
        solver = _prepare_ksp(petsc, amat, pmat, plan.policy, prefix=plan.plan_id[:16])
        transfer = BackendTransferEvidence(
            device_to_host_bytes=_storage_bytes(astorage)
            + (0 if pmat is amat else _storage_bytes(pstorage)),
            synchronization_count=3 + (0 if pmat is amat else 3),
        )
        return PreparedPETScLinearSolve(
            plan,
            transfer,
            petsc=petsc,
            operator_matrix=amat,
            preconditioner_matrix=pmat,
            solver=solver,
            prepared_id=canonical_fingerprint(
                {"kind": "prepared-petsc-ksp", "plan": plan.plan_id}
            ),
        )

    def solve_linear(
        self, prepared: PreparedPETScLinearSolve, rhs: PyTree[Any], /
    ) -> PETScLinearResult:
        if not isinstance(prepared, PreparedPETScLinearSolve):
            raise TypeError("prepared must be a PreparedPETScLinearSolve.")
        target, source = (
            prepared.plan.problem.operator.target,
            prepared.plan.problem.operator.source,
        )
        coordinates, rhs_shape = _pack_vectors(target, rhs)
        columns = coordinates.reshape((target.size, -1))
        solutions, iterations, reasons = [], [], []
        for column in np.asarray(jax.device_get(columns)).T:
            target_vector = prepared.petsc.Vec().createWithArray(
                np.asarray(column).copy(), comm=prepared.petsc.COMM_SELF
            )
            solution_vector = target_vector.duplicate()
            solution_vector.set(0.0)
            prepared.solver.solve(target_vector, solution_vector)
            solutions.append(solution_vector.getArray(readonly=True).copy())
            iterations.append(int(prepared.solver.getIterationNumber()))
            reasons.append(int(prepared.solver.getConvergedReason()))
        solution_coordinates = jnp.asarray(
            np.column_stack(solutions), dtype=coordinates.dtype
        ).reshape((source.size, *rhs_shape))
        value = _unpack_vectors(source, solution_coordinates, rhs_shape)
        residual_norm, relative_residual, rhs_norm = _linear_residuals(
            prepared.plan.problem, value, rhs, rhs_shape
        )
        shape = rhs_shape or ()
        reason_array, iteration_array = (
            jnp.asarray(reasons, dtype=jnp.int32).reshape(shape),
            jnp.asarray(iterations, dtype=jnp.int32).reshape(shape),
        )
        threshold = (
            prepared.plan.policy.absolute_tolerance
            + prepared.plan.policy.relative_tolerance * rhs_norm
        )
        converged = (reason_array > 0) & (residual_norm <= threshold)
        status = jnp.where(
            converged,
            int(LinearSolveStatus.SUCCESS),
            jnp.where(
                reason_array > 0,
                int(LinearSolveStatus.RESIDUAL_TOO_LARGE),
                jnp.where(
                    iteration_array >= prepared.plan.policy.maximum_iterations,
                    int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
                    int(LinearSolveStatus.BREAKDOWN),
                ),
            ),
        )
        coordinate_bytes = int(coordinates.size * coordinates.dtype.itemsize)
        solve_transfer = BackendTransferEvidence(
            host_to_device_bytes=coordinate_bytes,
            device_to_host_bytes=coordinate_bytes,
            synchronization_count=2 * columns.shape[1],
        )
        return PETScLinearResult(
            value,
            status,
            PETScLinearDiagnostics(
                residual_norm=residual_norm,
                relative_residual=relative_residual,
                iterations=iteration_array,
                convergence_reason=reason_array,
                converged=converged,
            ),
            _linear_provenance(prepared, solve_transfer),
        )

    def refresh_linear(
        self,
        prepared: PreparedPETScLinearSolve,
        problem: LinearSystem,
        /,
        *,
        preconditioner_operator: AbstractSparseLinearOperator | None = None,
    ) -> PreparedPETScLinearSolve:
        if not isinstance(prepared, PreparedPETScLinearSolve):
            raise TypeError("prepared must be a PreparedPETScLinearSolve.")
        if (
            problem.problem_id != prepared.plan.problem.problem_id
            or problem.operator.operator_id != prepared.plan.problem.operator.operator_id
        ):
            raise ValueError(
                "PETSc KSP refresh must preserve problem and Amat identities."
            )
        pmat_operator = (
            problem.operator
            if preconditioner_operator is None
            and prepared.plan.preconditioner_operator.operator_id
            == prepared.plan.problem.operator.operator_id
            else prepared.plan.preconditioner_operator
            if preconditioner_operator is None
            else preconditioner_operator
        )
        if pmat_operator.operator_id != prepared.plan.preconditioner_operator.operator_id:
            raise ValueError(
                "PETSc KSP refresh must preserve the immutable Pmat identity."
            )
        plan = self.plan_linear(
            problem, prepared.plan.policy, preconditioner_operator=pmat_operator
        )
        if (
            plan.operator_pattern != prepared.plan.operator_pattern
            or plan.preconditioner_pattern != prepared.plan.preconditioner_pattern
        ):
            raise ValueError(
                "PETSc KSP refresh requires unchanged Amat and Pmat CSR patterns."
            )
        astorage, pstorage = (
            _canonical_storage(problem.operator, role="system operator"),
            _canonical_storage(pmat_operator, role="preconditioner operator"),
        )
        _update_matrix(prepared.petsc, prepared.operator_matrix, astorage)
        if prepared.preconditioner_matrix is not prepared.operator_matrix:
            _update_matrix(prepared.petsc, prepared.preconditioner_matrix, pstorage)
        prepared.solver.setReusePreconditioner(prepared.plan.policy.reuse_preconditioner)
        prepared.solver.setOperators(
            prepared.operator_matrix, prepared.preconditioner_matrix
        )
        prepared.solver.setUp()
        transfer = BackendTransferEvidence(
            device_to_host_bytes=_storage_bytes(astorage)
            + (
                0
                if prepared.preconditioner_matrix is prepared.operator_matrix
                else _storage_bytes(pstorage)
            ),
            synchronization_count=3
            + (0 if prepared.preconditioner_matrix is prepared.operator_matrix else 3),
        )
        return PreparedPETScLinearSolve(
            plan,
            transfer,
            petsc=prepared.petsc,
            operator_matrix=prepared.operator_matrix,
            preconditioner_matrix=prepared.preconditioner_matrix,
            solver=prepared.solver,
            prepared_id=prepared.prepared_id,
            numeric_version=prepared.numeric_version + 1,
        )

    def plan_nonlinear(
        self,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any],
        policy: PETScSNESPolicy | None = None,
        /,
        *,
        args: Any = None,
    ) -> PETScNonlinearPlan:
        return PETScNonlinearPlan(
            problem,
            initial_state,
            PETScSNESPolicy() if policy is None else policy,
            args=args,
        )

    def prepare_nonlinear(
        self, plan: PETScNonlinearPlan, /
    ) -> PreparedPETScNonlinearSolve:
        if not isinstance(plan, PETScNonlinearPlan):
            raise TypeError("plan must be a PETScNonlinearPlan.")
        petsc = import_backend_module(
            self.availability(), "nonlinear-system", "petsc4py.PETSc"
        )
        solver = petsc.SNES().create(comm=petsc.COMM_SELF)
        residual_vector = petsc.Vec().createSeq(plan.space.size, comm=petsc.COMM_SELF)
        jacobian, callbacks = _configure_snes(
            petsc, solver, residual_vector, plan, prefix=plan.plan_id[:16]
        )
        return PreparedPETScNonlinearSolve(
            plan,
            BackendTransferEvidence(),
            petsc=petsc,
            solver=solver,
            residual_vector=residual_vector,
            jacobian=jacobian,
            callbacks=callbacks,
            prepared_id=canonical_fingerprint(
                {"kind": "prepared-petsc-snes", "plan": plan.plan_id}
            ),
        )

    def solve_nonlinear(
        self,
        prepared: PreparedPETScNonlinearSolve,
        initial_state: PyTree[Any] | None = None,
        /,
    ) -> PETScNonlinearResult:
        if not isinstance(prepared, PreparedPETScNonlinearSolve):
            raise TypeError("prepared must be a PreparedPETScNonlinearSolve.")
        plan = prepared.plan
        state = (
            plan.initial_state
            if initial_state is None
            else plan.space.validate(initial_state)
        )
        initial_residual, _ = plan.problem.evaluate(state, plan.args)
        initial_norm = jnp.linalg.norm(plan.space.flatten(initial_residual))
        coordinates = plan.space.flatten(state)
        value = prepared.petsc.Vec().createWithArray(
            np.asarray(jax.device_get(coordinates)).copy(), comm=prepared.petsc.COMM_SELF
        )
        prepared.solver.solve(None, value)
        final_state = plan.space.unflatten(
            jnp.asarray(value.getArray(readonly=True).copy(), dtype=coordinates.dtype)
        )
        final_residual, auxiliary = plan.problem.evaluate(final_state, plan.args)
        final_norm = jnp.linalg.norm(plan.space.flatten(final_residual))
        reason, iterations = (
            int(prepared.solver.getConvergedReason()),
            int(prepared.solver.getIterationNumber()),
        )
        function_evaluations, linear_iterations = (
            int(prepared.solver.getFunctionEvaluations()),
            int(prepared.solver.getLinearSolveIterations()),
        )
        threshold = (
            plan.policy.absolute_tolerance + plan.policy.relative_tolerance * initial_norm
        )
        converged = (reason > 0) & (final_norm <= threshold)
        status = jnp.where(
            converged,
            int(NonlinearStatus.SUCCESS),
            jnp.where(
                jnp.isfinite(final_norm),
                int(NonlinearStatus.MAXIMUM_STEPS_REACHED)
                if iterations >= plan.policy.maximum_iterations
                else int(NonlinearStatus.BACKEND_FAILED),
                int(NonlinearStatus.NONFINITE_EVALUATION),
            ),
        )
        coordinate_bytes = int(coordinates.size * coordinates.dtype.itemsize)
        callback_bytes = function_evaluations * coordinate_bytes
        transfer = BackendTransferEvidence(
            host_to_device_bytes=coordinate_bytes + callback_bytes,
            device_to_host_bytes=coordinate_bytes + callback_bytes,
            synchronization_count=2 + 2 * function_evaluations,
        )
        diagnostics = PETScNonlinearDiagnostics(
            initial_residual_norm=initial_norm,
            final_residual_norm=final_norm,
            iterations=iterations,
            function_evaluations=function_evaluations,
            linear_iterations=linear_iterations,
            convergence_reason=reason,
            converged=converged,
        )
        provenance = PETScProvenance(
            method=f"{plan.policy.snes_type}/{plan.policy.jacobian_mode}",
            preconditioner=plan.policy.ksp.pc_type,
            operator_id=f"{plan.problem.problem_id}/residual",
            preconditioner_operator_id=f"{plan.problem.problem_id}/{plan.policy.jacobian_mode}",
            plan_id=plan.plan_id,
            prepared_id=prepared.prepared_id,
            problem_id=plan.problem.problem_id,
            numeric_version=prepared.numeric_version,
            reused_preconditioner=plan.policy.ksp.reuse_preconditioner,
            setup_transfer=prepared.setup_transfer,
            solve_transfer=transfer,
        )
        return PETScNonlinearResult(
            state=final_state,
            residual=final_residual,
            auxiliary=auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=provenance,
        )

    def refresh_nonlinear(
        self,
        prepared: PreparedPETScNonlinearSolve,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any] | None = None,
        /,
        *,
        args: Any = None,
    ) -> PreparedPETScNonlinearSolve:
        if not isinstance(prepared, PreparedPETScNonlinearSolve):
            raise TypeError("prepared must be a PreparedPETScNonlinearSolve.")
        if problem.problem_id != prepared.plan.problem.problem_id:
            raise ValueError("PETSc SNES refresh must preserve problem_id.")
        plan = self.plan_nonlinear(
            problem,
            prepared.plan.initial_state if initial_state is None else initial_state,
            prepared.plan.policy,
            args=prepared.plan.args if args is None else args,
        )
        if not plan.space.compatible(prepared.plan.space):
            raise ValueError(
                "PETSc SNES refresh must preserve canonical state structure."
            )
        jacobian, callbacks = _configure_snes(
            prepared.petsc,
            prepared.solver,
            prepared.residual_vector,
            plan,
            prefix=plan.plan_id[:16],
            jacobian=prepared.jacobian,
        )
        return PreparedPETScNonlinearSolve(
            plan,
            BackendTransferEvidence(),
            petsc=prepared.petsc,
            solver=prepared.solver,
            residual_vector=prepared.residual_vector,
            jacobian=jacobian,
            callbacks=callbacks,
            prepared_id=prepared.prepared_id,
            numeric_version=prepared.numeric_version + 1,
        )


def _canonical_storage(
    operator: AbstractLinearOperator, /, *, role: str
) -> SparseStorage:
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise ValueError(
            f"PETSc {role} must be backed strictly by canonical CSR; no dense "
            "materialization or matrix-free fallback is permitted."
        )
    if not operator.source.compatible(operator.target):
        raise ValueError(f"PETSc {role} must be a square endomorphism.")
    storage = operator.sparse_storage()
    if (
        storage.format != "csr"
        or not storage.canonical
        or not storage.sorted_indices
        or storage.shape[0] != storage.shape[1]
    ):
        raise ValueError(f"PETSc {role} must expose sorted canonical square CSR storage.")
    return storage


def _pattern(storage: SparseStorage, /) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(
        int(value) for value in np.asarray(jax.device_get(storage.indptr))
    ), tuple(int(value) for value in np.asarray(jax.device_get(storage.indices)))


def _storage_bytes(storage: SparseStorage, /) -> int:
    return int(storage.values.nbytes + storage.indices.nbytes + storage.indptr.nbytes)


def _host_csr(storage: SparseStorage, petsc: Any, /):
    values = np.asarray(jax.device_get(storage.values))
    scalar_dtype = np.dtype(petsc.ScalarType)
    if values.dtype != scalar_dtype:
        raise TypeError(
            f"PETSc scalar dtype {scalar_dtype} does not match operator dtype "
            f"{values.dtype}; implicit precision conversion is disabled."
        )
    return (
        np.asarray(jax.device_get(storage.indptr), dtype=petsc.IntType),
        np.asarray(jax.device_get(storage.indices), dtype=petsc.IntType),
        values.copy(),
    )


def _create_matrix(petsc: Any, storage: SparseStorage, /):
    matrix = petsc.Mat().createAIJ(
        size=storage.shape, csr=_host_csr(storage, petsc), comm=petsc.COMM_SELF
    )
    matrix.assemble()
    return matrix


def _update_matrix(petsc: Any, matrix: Any, storage: SparseStorage, /) -> None:
    indptr, indices, values = _host_csr(storage, petsc)
    matrix.zeroEntries()
    matrix.setValuesCSR(indptr, indices, values)
    matrix.assemble()


def _prepare_ksp(
    petsc: Any, amat: Any, pmat: Any, policy: PETScKSPPolicy, /, *, prefix: str
):
    solver = petsc.KSP().create(comm=petsc.COMM_SELF)
    solver.setOperators(amat, pmat)
    solver.setType(policy.ksp_type)
    solver.getPC().setType(policy.pc_type)
    solver.setReusePreconditioner(policy.reuse_preconditioner)
    solver.setTolerances(
        rtol=policy.relative_tolerance,
        atol=policy.absolute_tolerance,
        divtol=policy.divergence_tolerance,
        max_it=policy.maximum_iterations,
    )
    solver.setInitialGuessNonzero(False)
    _apply_options(petsc, solver, policy.options, prefix=f"phydrax_ksp_{prefix}_")
    solver.setUp()
    return solver


def _configure_inner_ksp(
    petsc: Any, snes: Any, policy: PETScKSPPolicy, prefix: str
) -> None:
    solver = snes.getKSP()
    solver.setType(policy.ksp_type)
    solver.getPC().setType(policy.pc_type)
    solver.setReusePreconditioner(policy.reuse_preconditioner)
    solver.setTolerances(
        rtol=policy.relative_tolerance,
        atol=policy.absolute_tolerance,
        divtol=policy.divergence_tolerance,
        max_it=policy.maximum_iterations,
    )
    _apply_options(petsc, solver, policy.options, prefix=f"phydrax_snes_ksp_{prefix}_")


def _configure_snes(
    petsc: Any,
    solver: Any,
    residual_vector: Any,
    plan: PETScNonlinearPlan,
    /,
    *,
    prefix: str,
    jacobian: Any = None,
) -> tuple[Any, tuple[Any, ...]]:
    dtype = jax.tree.leaves(plan.space.structure())[0].dtype

    def residual_callback(snes, value, residual):
        del snes
        state = plan.space.unflatten(
            jnp.asarray(value.getArray(readonly=True), dtype=dtype)
        )
        physical, _ = plan.problem.evaluate(state, plan.args)
        residual.getArray()[:] = np.asarray(jax.device_get(plan.space.flatten(physical)))

    solver.setFunction(residual_callback, residual_vector)
    callbacks: tuple[Any, ...] = (residual_callback,)
    if plan.policy.jacobian_mode == "matrix-free":
        solver.setUseMF(True)
        jacobian_ = None
    else:
        dimension = plan.space.size
        jacobian_ = (
            petsc.Mat().createDense(size=(dimension, dimension), comm=petsc.COMM_SELF)
            if jacobian is None
            else jacobian
        )
        jacobian_.setUp()
        indices = np.arange(dimension, dtype=petsc.IntType)

        def coordinate_residual(coordinates):
            residual, _ = plan.problem.evaluate(
                plan.space.unflatten(coordinates), plan.args
            )
            return plan.space.flatten(residual)

        dense_jacobian = jax.jacfwd(coordinate_residual)

        def jacobian_callback(snes, value, operator, preconditioner):
            del snes
            dense = np.asarray(
                jax.device_get(
                    dense_jacobian(
                        jnp.asarray(value.getArray(readonly=True), dtype=dtype)
                    )
                )
            )
            operator.zeroEntries()
            operator.setValues(indices, indices, dense)
            operator.assemble()
            if preconditioner is not operator:
                preconditioner.zeroEntries()
                preconditioner.setValues(indices, indices, dense)
                preconditioner.assemble()

        solver.setJacobian(jacobian_callback, J=jacobian_, P=jacobian_)
        callbacks = (residual_callback, jacobian_callback, dense_jacobian)
    solver.setType(plan.policy.snes_type)
    solver.setTolerances(
        rtol=plan.policy.relative_tolerance,
        atol=plan.policy.absolute_tolerance,
        stol=plan.policy.step_tolerance,
        max_it=plan.policy.maximum_iterations,
        max_funcs=plan.policy.maximum_function_evaluations,
    )
    _configure_inner_ksp(petsc, solver, plan.policy.ksp, prefix)
    _apply_options(petsc, solver, plan.policy.options, prefix=f"phydrax_snes_{prefix}_")
    solver.setUp()
    return jacobian_, callbacks


def _apply_options(
    petsc: Any,
    solver: Any,
    options: tuple[tuple[str, str | None], ...],
    /,
    *,
    prefix: str,
) -> None:
    solver.setOptionsPrefix(prefix)
    database = petsc.Options()
    names: list[str] = []
    try:
        for name, value in options:
            key = f"{prefix}{name}"
            database[key] = value
            names.append(key)
        solver.setFromOptions()
    finally:
        for name in names:
            del database[name]


def _require_matching_tree(
    space: AbstractVectorSpace, value: PyTree[Any], name: str, /
) -> None:
    leaves, treedef = jax.tree.flatten(value)
    specs, specdef = jax.tree.flatten(space.structure())
    if treedef != specdef:
        raise ValueError(
            f"PETSc {name} PyTree structure must match canonical state coordinates."
        )
    for leaf, spec in zip(leaves, specs, strict=True):
        array = jnp.asarray(leaf)
        if array.shape != spec.shape or np.dtype(array.dtype) != np.dtype(spec.dtype):
            raise TypeError(
                f"PETSc {name} shapes and dtypes must match canonical state coordinates."
            )


def _pack_vectors(
    space: AbstractVectorSpace, vectors: PyTree[Any], /
) -> tuple[Array, tuple[int, ...]]:
    leaves, treedef = jax.tree.flatten(vectors)
    specs, specdef = jax.tree.flatten(space.structure())
    if treedef != specdef:
        raise ValueError("RHS PyTree structure does not match the operator target space.")
    rhs_shape: tuple[int, ...] | None = None
    flattened: list[Array] = []
    for leaf, spec in zip(leaves, specs, strict=True):
        array, event_rank = jnp.asarray(leaf), len(spec.shape)
        if tuple(array.shape[:event_rank]) != tuple(spec.shape) or np.dtype(
            array.dtype
        ) != np.dtype(spec.dtype):
            raise TypeError(
                "RHS leaf event shapes and dtypes must match the target space."
            )
        trailing = tuple(int(size) for size in array.shape[event_rank:])
        if rhs_shape is None:
            rhs_shape = trailing
        elif trailing != rhs_shape:
            raise ValueError("All RHS leaves must share the same trailing RHS axes.")
        flattened.append(array.reshape((prod(spec.shape), *trailing)))
    return jnp.concatenate(flattened, axis=0), rhs_shape or ()


def _unpack_vectors(
    space: AbstractVectorSpace, coordinates: Array, rhs_shape: tuple[int, ...], /
):
    specs, treedef = jax.tree.flatten(space.structure())
    leaves, offset = [], 0
    for spec in specs:
        size = prod(spec.shape)
        leaves.append(
            coordinates[offset : offset + size].reshape((*spec.shape, *rhs_shape))
        )
        offset += size
    return jax.tree.unflatten(treedef, leaves)


def _column(vectors, index: int, rhs_shape: tuple[int, ...]):
    if not rhs_shape:
        return vectors
    location = np.unravel_index(index, rhs_shape)
    return jax.tree.map(lambda value: value[(..., *location)], vectors)


def _linear_residuals(problem: LinearSystem, value, rhs, rhs_shape):
    norms, relatives, rhs_norms = [], [], []
    for index in range(prod(rhs_shape or (1,))):
        solution, target = (
            _column(value, index, rhs_shape),
            _column(rhs, index, rhs_shape),
        )
        residual = jax.tree.map(
            lambda image, expected: image - expected,
            problem.operator.mv(solution),
            target,
        )
        norm = jnp.linalg.norm(problem.operator.target.flatten(residual))
        rhs_norm = jnp.linalg.norm(problem.operator.target.flatten(target))
        norms.append(norm)
        rhs_norms.append(rhs_norm)
        relatives.append(norm / jnp.where(rhs_norm == 0, 1.0, rhs_norm))
    shape = rhs_shape or ()
    return (
        jnp.asarray(norms).reshape(shape),
        jnp.asarray(relatives).reshape(shape),
        jnp.asarray(rhs_norms).reshape(shape),
    )


def _linear_provenance(
    prepared: PreparedPETScLinearSolve, transfer: BackendTransferEvidence
) -> PETScProvenance:
    plan = prepared.plan
    return PETScProvenance(
        method=plan.policy.ksp_type,
        preconditioner=plan.policy.pc_type,
        operator_id=plan.problem.operator.operator_id,
        preconditioner_operator_id=plan.preconditioner_operator.operator_id,
        plan_id=plan.plan_id,
        prepared_id=prepared.prepared_id,
        problem_id=plan.problem.problem_id,
        numeric_version=prepared.numeric_version,
        reused_preconditioner=plan.policy.reuse_preconditioner,
        setup_transfer=prepared.setup_transfer,
        solve_transfer=transfer,
    )


def _ksp_payload(policy: PETScKSPPolicy) -> dict[str, Any]:
    return {
        "ksp": policy.ksp_type,
        "pc": policy.pc_type,
        "rtol": policy.relative_tolerance,
        "atol": policy.absolute_tolerance,
        "divtol": policy.divergence_tolerance,
        "max_it": policy.maximum_iterations,
        "reuse_preconditioner": policy.reuse_preconditioner,
        "options": policy.options,
    }


def _snes_payload(policy: PETScSNESPolicy) -> dict[str, Any]:
    return {
        "mode": policy.jacobian_mode,
        "snes": policy.snes_type,
        "rtol": policy.relative_tolerance,
        "atol": policy.absolute_tolerance,
        "stol": policy.step_tolerance,
        "max_it": policy.maximum_iterations,
        "max_funcs": policy.maximum_function_evaluations,
        "dense_dimension": policy.maximum_dense_dimension,
        "dense_bytes": policy.maximum_dense_bytes,
        "ksp": _ksp_payload(policy.ksp),
        "options": policy.options,
    }


def petsc_availability() -> BackendAvailability:
    """Probe the optional PETSc provider without importing it at package import time."""
    return PETScBackend().availability()


def plan_petsc_linear(problem, policy=None, /, *, preconditioner_operator=None):
    return PETScBackend().plan_linear(
        problem, policy, preconditioner_operator=preconditioner_operator
    )


def prepare_petsc_linear(plan, /):
    return PETScBackend().prepare_linear(plan)


def solve_petsc_linear(prepared, rhs, /):
    return PETScBackend().solve_linear(prepared, rhs)


def refresh_petsc_linear(prepared, problem, /, *, preconditioner_operator=None):
    return PETScBackend().refresh_linear(
        prepared, problem, preconditioner_operator=preconditioner_operator
    )


def plan_petsc_nonlinear(problem, initial_state, policy=None, /, *, args=None):
    return PETScBackend().plan_nonlinear(problem, initial_state, policy, args=args)


def prepare_petsc_nonlinear(plan, /):
    return PETScBackend().prepare_nonlinear(plan)


def solve_petsc_nonlinear(prepared, initial_state=None, /):
    return PETScBackend().solve_nonlinear(prepared, initial_state)


def refresh_petsc_nonlinear(prepared, problem, initial_state=None, /, *, args=None):
    return PETScBackend().refresh_nonlinear(prepared, problem, initial_state, args=args)


__all__ = [
    "PETScBackend",
    "PETScJacobianMode",
    "PETScKSPPolicy",
    "PETScLinearDiagnostics",
    "PETScLinearPlan",
    "PETScLinearResult",
    "PETScNonlinearDiagnostics",
    "PETScNonlinearPlan",
    "PETScNonlinearResult",
    "PETScOptionValue",
    "PETScProvenance",
    "PETScSNESPolicy",
    "PreparedPETScLinearSolve",
    "PreparedPETScNonlinearSolve",
    "petsc_availability",
    "plan_petsc_linear",
    "plan_petsc_nonlinear",
    "prepare_petsc_linear",
    "prepare_petsc_nonlinear",
    "refresh_petsc_linear",
    "refresh_petsc_nonlinear",
    "solve_petsc_linear",
    "solve_petsc_nonlinear",
]
