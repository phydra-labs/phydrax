#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Optional SLEPc EPS backend for general matrix-free and sparse eigenproblems.

The module itself does not import ``slepc4py`` or ``petsc4py``.  Provider imports
happen only while preparing an explicitly selected SLEPc plan.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg._operators import AbstractLinearOperator
from ..linalg._spaces import _coordinate_dtype
from ..linalg._sparse_contract import AbstractSparseLinearOperator
from ..linalg.eigen import (
    CayleyTransform,
    GeneralEigenproblem,
    GeneralEigenSelection,
    GeneralEigenSolveStatus,
    GeneralEigenTolerancePolicy,
    GeneralEigenTransform,
    ShiftInvertTransform,
    StandardTransform,
)
from ._availability import import_backend_module, probe_backend
from ._types import (
    AbstractExternalBackend,
    BackendAvailability,
    BackendCapabilities,
    BackendTransferEvidence,
)


SLEPcOperatorMode: TypeAlias = Literal["shell", "csr"]
SLEPcFailureMode: TypeAlias = Literal["status", "error"]

SLEPC_CAPABILITIES = BackendCapabilities(
    backend="slepc-eps",
    problem_kinds=("standard", "generalized"),
    execution="host",
    host_only=True,
    supports_matrix_free=True,
    supports_assembled=True,
    coordinate_dtypes=("float32", "float64", "complex64", "complex128"),
    requires_explicit_release=True,
)
_SLEPC_REQUIREMENT = "slepc4py and petsc4py built against compatible SLEPc/PETSc"


def slepc_availability() -> BackendAvailability:
    """Lazily probe SLEPc and its compatible PETSc provider."""
    slepc = probe_backend(
        SLEPC_CAPABILITIES,
        module="slepc4py",
        requirement=_SLEPC_REQUIREMENT,
        distributions=("slepc4py", "petsc4py"),
    )
    if not slepc.available:
        return slepc
    petsc = probe_backend(
        SLEPC_CAPABILITIES,
        module="petsc4py",
        requirement=_SLEPC_REQUIREMENT,
        distributions=("slepc4py", "petsc4py"),
    )
    if not petsc.available:
        return petsc
    versions = tuple(dict.fromkeys(slepc.versions + petsc.versions))
    return BackendAvailability(
        capabilities=SLEPC_CAPABILITIES,
        available=True,
        requirement=_SLEPC_REQUIREMENT,
        reason="slepc4py and petsc4py imported successfully",
        versions=versions,
    )


class SLEPcSTOptions(StrictModule):
    """Explicit spectral-transform and inner KSP choices for assembled solves."""

    st_type: Literal["sinvert", "cayley"] = eqx.field(static=True)
    ksp_type: str = eqx.field(static=True)
    pc_type: str = eqx.field(static=True)
    factor_solver_type: str | None = eqx.field(static=True)
    options_prefix: str | None = eqx.field(static=True)

    def __init__(
        self,
        st_type: Literal["sinvert", "cayley"],
        /,
        *,
        ksp_type: str,
        pc_type: str,
        factor_solver_type: str | None = None,
        options_prefix: str | None = None,
    ):
        if st_type not in ("sinvert", "cayley"):
            raise ValueError("st_type must be 'sinvert' or 'cayley'.")
        ksp = str(ksp_type)
        pc = str(pc_type)
        factor = None if factor_solver_type is None else str(factor_solver_type)
        prefix = None if options_prefix is None else str(options_prefix)
        if not ksp or not pc or factor == "" or prefix == "":
            raise ValueError("Declared SLEPc ST/KSP option strings must be non-empty.")
        self.st_type = st_type
        self.ksp_type = ksp
        self.pc_type = pc
        self.factor_solver_type = factor
        self.options_prefix = prefix


class SLEPcEigenPolicy(StrictModule):
    """Immutable EPS selection, transform, tolerance, and storage policy."""

    selection: GeneralEigenSelection
    transform: GeneralEigenTransform
    tolerance: GeneralEigenTolerancePolicy
    operator_mode: SLEPcOperatorMode = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    subspace_dimension: int | None = eqx.field(static=True)
    st_options: SLEPcSTOptions | None = eqx.field(static=True)
    failure_mode: SLEPcFailureMode = eqx.field(static=True)

    def __init__(
        self,
        selection: GeneralEigenSelection | None = None,
        /,
        *,
        transform: GeneralEigenTransform | None = None,
        tolerance: GeneralEigenTolerancePolicy | None = None,
        operator_mode: SLEPcOperatorMode = "shell",
        maximum_iterations: int = 300,
        subspace_dimension: int | None = None,
        st_options: SLEPcSTOptions | None = None,
        failure_mode: SLEPcFailureMode = "status",
    ):
        selected = (
            GeneralEigenSelection("largest-magnitude", count=1)
            if selection is None
            else selection
        )
        transformed = StandardTransform() if transform is None else transform
        tolerances = GeneralEigenTolerancePolicy() if tolerance is None else tolerance
        if not isinstance(selected, GeneralEigenSelection):
            raise TypeError("selection must be a GeneralEigenSelection.")
        if not isinstance(
            transformed, (StandardTransform, ShiftInvertTransform, CayleyTransform)
        ):
            raise TypeError("transform must be a general eigen transform.")
        if not isinstance(tolerances, GeneralEigenTolerancePolicy):
            raise TypeError("tolerance must be a GeneralEigenTolerancePolicy.")
        if operator_mode not in ("shell", "csr"):
            raise ValueError("operator_mode must be 'shell' or 'csr'.")
        iterations = int(maximum_iterations)
        dimension = None if subspace_dimension is None else int(subspace_dimension)
        if iterations < 1:
            raise ValueError("maximum_iterations must be positive.")
        if dimension is not None and dimension < 2:
            raise ValueError("subspace_dimension must be at least two or None.")
        if st_options is not None and not isinstance(st_options, SLEPcSTOptions):
            raise TypeError("st_options must be SLEPcSTOptions or None.")
        if failure_mode not in ("status", "error"):
            raise ValueError("failure_mode must be 'status' or 'error'.")
        self.selection = selected
        self.transform = transformed
        self.tolerance = tolerances
        self.operator_mode = operator_mode
        self.maximum_iterations = iterations
        self.subspace_dimension = dimension
        self.st_options = st_options
        self.failure_mode = failure_mode


class SLEPcEigenPlan(StrictModule):
    """Dependency-free symbolic EPS plan for one general eigenproblem."""

    policy: SLEPcEigenPolicy
    problem_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    mass_operator_id: str | None = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    requested_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedSLEPcEigenSolve(StrictModule):
    """Host PETSc/SLEPc numerical state bound to one immutable plan."""

    problem: GeneralEigenproblem
    plan: SLEPcEigenPlan
    eps: Any = eqx.field(static=True)
    matrix: Any = eqx.field(static=True)
    mass_matrix: Any = eqx.field(static=True)
    shell_contexts: tuple[Any, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: Array
    preparation_transfer: BackendTransferEvidence
    provider_versions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    scalar_is_complex: bool = eqx.field(static=True)


class SLEPcEigenDiagnostics(StrictModule):
    """Original-pencil verification and unmodified SLEPc termination evidence."""

    right_residual_norms: Array
    left_residual_norms: Array
    right_relative_residuals: Array
    left_relative_residuals: Array
    pairing_matrix: Array
    biorthogonality_error: Array
    converged_mask: Array
    converged_count: Array
    selected_count: Array
    available_count: Array
    requested_count: int = eqx.field(static=True)
    iteration_count: Array
    operator_action_count: Array
    slepc_reason: Array
    slepc_reason_name: str = eqx.field(static=True)


class SLEPcEigenProvenance(StrictModule):
    """External execution, identity, transfer, and no-fallback evidence."""

    backend: str = eqx.field(static=True)
    host_only: bool = eqx.field(static=True)
    operator_mode: SLEPcOperatorMode = eqx.field(static=True)
    algorithm: str = eqx.field(static=True)
    transform: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    mass_operator_id: str | None = eqx.field(static=True)
    numeric_version: Array
    provider_versions: tuple[tuple[str, str], ...] = eqx.field(static=True)
    no_fallback: bool = eqx.field(static=True)
    transfer: BackendTransferEvidence


class SLEPcEigenResult(StrictModule):
    """Selected paired eigenvectors with independently verified pencil residuals."""

    eigenvalues: Array
    alpha: Array
    beta: Array
    right_eigenvectors: PyTree[Array]
    left_eigenvectors: PyTree[Array]
    right_eigenvector_coordinates: Array
    left_eigenvector_coordinates: Array
    status: Array
    diagnostics: SLEPcEigenDiagnostics
    provenance: SLEPcEigenProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(GeneralEigenSolveStatus.SUCCESS)


class SLEPcBackend(AbstractExternalBackend):
    """Inspection and lifecycle facade for the optional SLEPc EPS provider."""

    @property
    def name(self) -> str:
        return SLEPC_CAPABILITIES.backend

    @property
    def capabilities(self) -> BackendCapabilities:
        return SLEPC_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return slepc_availability()

    def plan(
        self,
        problem: GeneralEigenproblem,
        policy: SLEPcEigenPolicy | None = None,
        /,
    ) -> SLEPcEigenPlan:
        return plan_slepc_eigensolve(problem, policy)

    def prepare(
        self,
        problem: GeneralEigenproblem,
        policy_or_plan: SLEPcEigenPolicy | SLEPcEigenPlan | None = None,
        /,
    ) -> PreparedSLEPcEigenSolve:
        return prepare_slepc_eigensolve(problem, policy_or_plan)

    def solve(self, prepared: PreparedSLEPcEigenSolve, /) -> SLEPcEigenResult:
        return slepc_eigensolve(prepared)

    def refresh(
        self,
        prepared: PreparedSLEPcEigenSolve,
        problem: GeneralEigenproblem,
        /,
    ) -> PreparedSLEPcEigenSolve:
        return refresh_slepc_eigensolve(prepared, problem)


def plan_slepc_eigensolve(
    problem: GeneralEigenproblem,
    policy: SLEPcEigenPolicy | None = None,
    /,
) -> SLEPcEigenPlan:
    """Validate and fingerprint an EPS plan without importing optional packages."""
    if not isinstance(problem, GeneralEigenproblem):
        raise TypeError("problem must be a GeneralEigenproblem.")
    selected = SLEPcEigenPolicy() if policy is None else policy
    if not isinstance(selected, SLEPcEigenPolicy):
        raise TypeError("policy must be SLEPcEigenPolicy or None.")
    selection = selected.selection
    if selection.kind not in ("largest-real", "largest-magnitude", "closest"):
        raise ValueError(
            "SLEPc EPS supports largest-real, largest-magnitude, or closest selection."
        )
    if selection.count is None:
        raise ValueError("SLEPc selection requires an explicit count.")
    if selection.count >= problem.dimension:
        raise ValueError("SLEPc partial eigensolves require count < problem dimension.")
    if (
        selected.subspace_dimension is not None
        and selected.subspace_dimension <= selection.count
    ):
        raise ValueError("subspace_dimension must exceed the requested eigenpair count.")
    transformed = not isinstance(selected.transform, StandardTransform)
    if transformed:
        if selected.operator_mode != "csr":
            raise ValueError(
                "SLEPc shift-invert and Cayley transforms require explicit CSR; "
                "shell operators are never materialized implicitly."
            )
        if selected.selection.kind != "closest":
            raise ValueError("SLEPc spectral transforms require closest selection.")
        options = selected.st_options
        expected = (
            "sinvert"
            if isinstance(selected.transform, ShiftInvertTransform)
            else "cayley"
        )
        if options is None or options.st_type != expected:
            raise ValueError(
                f"{selected.transform.name} requires declared SLEPcSTOptions({expected!r}, ...)."
            )
        delta = abs(selected.selection.target - selected.transform.shift)
        threshold = selected.tolerance.absolute + selected.tolerance.relative * max(
            1.0, abs(selected.transform.shift)
        )
        if delta > threshold:
            raise ValueError("Closest-selection target must equal the spectral shift.")
    elif selected.st_options is not None:
        raise ValueError("st_options may only be declared with shift-invert or Cayley.")
    coordinate_dtype = np.dtype(_coordinate_dtype(problem.operator.source))
    if not np.issubdtype(coordinate_dtype, np.complexfloating):
        if selection.kind == "closest" and selection.target.imag != 0.0:
            raise ValueError("A complex target requires complex coordinates.")
        if transformed and selected.transform.shift.imag != 0.0:
            raise ValueError("A complex spectral shift requires complex coordinates.")
    _validate_two_sided_operator(problem.operator, "operator")
    if problem.mass_operator is not None:
        _validate_two_sided_operator(problem.mass_operator, "mass_operator")
    if selected.operator_mode == "csr":
        _require_sparse(problem.operator, "operator")
        if problem.mass_operator is not None:
            _require_sparse(problem.mass_operator, "mass_operator")
    mass_id = None if problem.mass_operator is None else problem.mass_operator.operator_id
    plan_id = canonical_fingerprint(
        {
            "kind": "slepc-eigen-plan",
            "problem": problem.problem_id,
            "operator": problem.operator.operator_id,
            "mass": mass_id,
            "dimension": problem.dimension,
            "selection": selection.selection_id,
            "transform": selected.transform.name,
            "shift": (
                None
                if isinstance(selected.transform, StandardTransform)
                else [selected.transform.shift.real, selected.transform.shift.imag]
            ),
            "operator_mode": selected.operator_mode,
            "maximum_iterations": selected.maximum_iterations,
            "subspace_dimension": selected.subspace_dimension,
            "st": (
                None
                if selected.st_options is None
                else {
                    "type": selected.st_options.st_type,
                    "ksp": selected.st_options.ksp_type,
                    "pc": selected.st_options.pc_type,
                    "factor": selected.st_options.factor_solver_type,
                    "prefix": selected.st_options.options_prefix,
                }
            ),
        }
    )
    return SLEPcEigenPlan(
        policy=selected,
        problem_id=problem.problem_id,
        operator_id=problem.operator.operator_id,
        mass_operator_id=mass_id,
        dimension=problem.dimension,
        requested_count=selection.count,
        plan_id=plan_id,
    )


def prepare_slepc_eigensolve(
    problem: GeneralEigenproblem,
    policy_or_plan: SLEPcEigenPolicy | SLEPcEigenPlan | None = None,
    /,
) -> PreparedSLEPcEigenSolve:
    """Import SLEPc/PETSc lazily and bind an EPS context to the pencil."""
    plan = (
        policy_or_plan
        if isinstance(policy_or_plan, SLEPcEigenPlan)
        else plan_slepc_eigensolve(problem, policy_or_plan)
    )
    _validate_plan_problem(plan, problem)
    return _prepare_numeric(problem, plan, numeric_version=0)


def refresh_slepc_eigensolve(
    prepared: PreparedSLEPcEigenSolve,
    problem: GeneralEigenproblem,
    /,
) -> PreparedSLEPcEigenSolve:
    """Rebind current numeric values while retaining symbolic and prepared identity."""
    if not isinstance(prepared, PreparedSLEPcEigenSolve):
        raise TypeError("prepared must be PreparedSLEPcEigenSolve.")
    _validate_plan_problem(prepared.plan, problem)
    return _prepare_numeric(
        problem,
        prepared.plan,
        numeric_version=int(np.asarray(prepared.numeric_version)) + 1,
        prepared_id=prepared.prepared_id,
    )


def release_slepc_eigensolve(prepared: PreparedSLEPcEigenSolve, /) -> None:
    """Collectively destroy PETSc objects owned by one prepared solve."""
    if not isinstance(prepared, PreparedSLEPcEigenSolve):
        raise TypeError("prepared must be PreparedSLEPcEigenSolve.")
    prepared.eps.destroy()
    if prepared.mass_matrix is not None:
        prepared.mass_matrix.destroy()
    prepared.matrix.destroy()


def slepc_eigensolve(prepared: PreparedSLEPcEigenSolve, /) -> SLEPcEigenResult:
    """Execute EPS and verify every returned pair against the original pencil."""
    if not isinstance(prepared, PreparedSLEPcEigenSolve):
        raise TypeError("prepared must be PreparedSLEPcEigenSolve.")
    eps = prepared.eps
    before = tuple(context.snapshot() for context in prepared.shell_contexts)
    eps.solve()
    iterations = int(eps.getIterationNumber())
    available = max(int(eps.getConverged()), 0)
    reason_object = eps.getConvergedReason()
    reason = int(reason_object)
    reason_name = getattr(reason_object, "name", str(reason_object))
    values, right, left = _extract_slepc_pairs(prepared, available)
    values, right, left = _postprocess_eigenpairs(
        prepared.problem,
        values,
        right,
        left,
        prepared.plan.policy.selection,
        prepared.plan.requested_count,
    )
    pairing_actions = (
        values.size * values.size * (1 + int(prepared.problem.mass_operator is not None))
        if values.size > 1
        else 0
    )
    (
        right_residuals,
        left_residuals,
        right_relative,
        left_relative,
        pairing,
        right,
        left,
        verification_actions,
    ) = _verify_and_normalize_pairs(prepared.problem, values, right, left)
    tolerance = prepared.plan.policy.tolerance
    itemwise = (
        np.isfinite(values)
        & np.isfinite(right_residuals)
        & np.isfinite(left_residuals)
        & (
            right_residuals
            <= tolerance.absolute
            + tolerance.relative * _residual_scales(right_residuals, right_relative)
        )
        & (
            left_residuals
            <= tolerance.absolute
            + tolerance.relative * _residual_scales(left_residuals, left_relative)
        )
    )
    selected_count = values.size
    identity = np.eye(selected_count, dtype=pairing.dtype)
    pairing_error = float(np.max(np.abs(pairing - identity), initial=0.0))
    output_finite = bool(
        np.all(np.isfinite(values))
        and np.all(np.isfinite(right))
        and np.all(np.isfinite(left))
        and np.all(np.isfinite(right_residuals))
        and np.all(np.isfinite(left_residuals))
    )
    if not output_finite:
        status = GeneralEigenSolveStatus.NONFINITE_OUTPUT
    elif reason <= 0 or selected_count < prepared.plan.requested_count:
        status = GeneralEigenSolveStatus.PARTIAL_CONVERGENCE
    elif not bool(np.all(itemwise)):
        status = GeneralEigenSolveStatus.RESIDUAL_TOLERANCE_NOT_MET
    elif pairing_error > tolerance.biorthogonality:
        status = GeneralEigenSolveStatus.BIORTHOGONALITY_TOLERANCE_NOT_MET
    else:
        status = GeneralEigenSolveStatus.SUCCESS
    if (
        prepared.plan.policy.failure_mode == "error"
        and status != GeneralEigenSolveStatus.SUCCESS
    ):
        raise RuntimeError(
            "SLEPc EPS did not satisfy the general-eigen contract: "
            f"{status.name}; reason={reason_name}."
        )
    after = tuple(context.snapshot() for context in prepared.shell_contexts)
    shell_actions = sum(final[0] - initial[0] for initial, final in zip(before, after))
    shell_h2d = sum(final[1] - initial[1] for initial, final in zip(before, after))
    shell_d2h = sum(final[2] - initial[2] for initial, final in zip(before, after))
    coordinate_dtype = _complex_coordinate_dtype(prepared.problem)
    result_bytes = values.nbytes + right.nbytes + left.nbytes
    verification_bytes = (
        (verification_actions + pairing_actions)
        * prepared.problem.dimension
        * coordinate_dtype.itemsize
    )
    transfer = BackendTransferEvidence(
        host_to_device_bytes=(
            int(np.asarray(prepared.preparation_transfer.host_to_device_bytes))
            + shell_h2d
            + verification_bytes
            + result_bytes
        ),
        device_to_host_bytes=(
            int(np.asarray(prepared.preparation_transfer.device_to_host_bytes))
            + shell_d2h
            + verification_bytes
        ),
        synchronization_count=(
            int(np.asarray(prepared.preparation_transfer.synchronization_count))
            + shell_actions
            + verification_actions
            + pairing_actions
            + 1
        ),
    )
    complex_dtype = _complex_coordinate_dtype(prepared.problem)
    values_array = jnp.asarray(values, dtype=complex_dtype)
    right_array = jnp.asarray(right, dtype=complex_dtype)
    left_array = jnp.asarray(left, dtype=complex_dtype)
    alpha = values_array
    beta = jnp.ones_like(values_array)
    return SLEPcEigenResult(
        eigenvalues=values_array,
        alpha=alpha,
        beta=beta,
        right_eigenvectors=_unflatten_complex_columns(
            prepared.problem.operator.source, right_array
        ),
        left_eigenvectors=_unflatten_complex_columns(
            prepared.problem.operator.source, left_array
        ),
        right_eigenvector_coordinates=right_array,
        left_eigenvector_coordinates=left_array,
        status=jnp.asarray(int(status), dtype=jnp.int32),
        diagnostics=SLEPcEigenDiagnostics(
            right_residual_norms=jnp.asarray(right_residuals),
            left_residual_norms=jnp.asarray(left_residuals),
            right_relative_residuals=jnp.asarray(right_relative),
            left_relative_residuals=jnp.asarray(left_relative),
            pairing_matrix=jnp.asarray(pairing, dtype=complex_dtype),
            biorthogonality_error=jnp.asarray(pairing_error),
            converged_mask=jnp.asarray(itemwise),
            converged_count=jnp.asarray(np.count_nonzero(itemwise), dtype=jnp.int32),
            selected_count=jnp.asarray(selected_count, dtype=jnp.int32),
            available_count=jnp.asarray(available, dtype=jnp.int32),
            requested_count=prepared.plan.requested_count,
            iteration_count=jnp.asarray(iterations, dtype=jnp.int32),
            operator_action_count=jnp.asarray(
                shell_actions + verification_actions + pairing_actions,
                dtype=jnp.int32,
            ),
            slepc_reason=jnp.asarray(reason, dtype=jnp.int32),
            slepc_reason_name=reason_name,
        ),
        provenance=SLEPcEigenProvenance(
            backend=SLEPC_CAPABILITIES.backend,
            host_only=True,
            operator_mode=prepared.plan.policy.operator_mode,
            algorithm=str(eps.getType()),
            transform=prepared.plan.policy.transform.name,
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=prepared.problem.operator.operator_id,
            mass_operator_id=(
                None
                if prepared.problem.mass_operator is None
                else prepared.problem.mass_operator.operator_id
            ),
            numeric_version=prepared.numeric_version,
            provider_versions=prepared.provider_versions,
            no_fallback=True,
            transfer=transfer,
        ),
    )


def _prepare_numeric(
    problem: GeneralEigenproblem,
    plan: SLEPcEigenPlan,
    /,
    *,
    numeric_version: int,
    prepared_id: str | None = None,
) -> PreparedSLEPcEigenSolve:
    availability = slepc_availability()
    SLEPc = import_backend_module(availability, problem.kind, "slepc4py.SLEPc")
    PETSc = import_backend_module(availability, problem.kind, "petsc4py.PETSc")
    scalar_is_complex = np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating)
    coordinate_dtype = np.dtype(_coordinate_dtype(problem.operator.source))
    if np.issubdtype(coordinate_dtype, np.complexfloating) and not scalar_is_complex:
        raise ValueError(
            "This PETSc installation uses real scalars and cannot bind a complex pencil."
        )
    contexts: list[_ShellMatrixContext] = []
    transfer_bytes = 0
    if plan.policy.operator_mode == "shell":
        matrix, context = _create_shell_matrix(PETSc, problem.operator)
        contexts.append(context)
        if problem.mass_operator is None:
            mass = None
        else:
            mass, context = _create_shell_matrix(PETSc, problem.mass_operator)
            contexts.append(context)
    else:
        matrix, consumed = _create_csr_matrix(PETSc, problem.operator)
        transfer_bytes += consumed
        if problem.mass_operator is None:
            mass = None
        else:
            mass, consumed = _create_csr_matrix(PETSc, problem.mass_operator)
            transfer_bytes += consumed
    eps = SLEPc.EPS().create(comm=PETSc.COMM_SELF)
    eps.setOperators(matrix, mass)
    eps.setProblemType(
        SLEPc.EPS.ProblemType.NHEP if mass is None else SLEPc.EPS.ProblemType.GNHEP
    )
    eps.setTwoSided(True)
    eps.setDimensions(
        nev=plan.requested_count,
        ncv=(
            plan.policy.subspace_dimension
            if plan.policy.subspace_dimension is not None
            else min(plan.dimension, max(2 * plan.requested_count + 1, 20))
        ),
    )
    eps.setTolerances(
        tol=plan.policy.tolerance.relative,
        max_it=plan.policy.maximum_iterations,
    )
    selection = plan.policy.selection
    if selection.kind == "largest-real":
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    elif selection.kind == "largest-magnitude":
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    else:
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.setTarget(selection.target if scalar_is_complex else selection.target.real)
    transform = plan.policy.transform
    if not isinstance(transform, StandardTransform):
        options = plan.policy.st_options
        assert options is not None
        st = eps.getST()
        st.setType(
            SLEPc.ST.Type.SINVERT
            if isinstance(transform, ShiftInvertTransform)
            else SLEPc.ST.Type.CAYLEY
        )
        st.setShift(transform.shift if scalar_is_complex else transform.shift.real)
        if options.options_prefix is not None:
            st.setOptionsPrefix(options.options_prefix)
        ksp = st.getKSP()
        ksp.setType(options.ksp_type)
        pc = ksp.getPC()
        pc.setType(options.pc_type)
        if options.factor_solver_type is not None:
            pc.setFactorSolverType(options.factor_solver_type)
    eps.setUp()
    identifier = (
        canonical_fingerprint(
            {
                "kind": "prepared-slepc-eigen",
                "plan": plan.plan_id,
                "problem": problem.problem_id,
            }
        )
        if prepared_id is None
        else prepared_id
    )
    return PreparedSLEPcEigenSolve(
        problem=problem,
        plan=plan,
        eps=eps,
        matrix=matrix,
        mass_matrix=mass,
        shell_contexts=tuple(contexts),
        prepared_id=identifier,
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        preparation_transfer=BackendTransferEvidence(
            device_to_host_bytes=transfer_bytes,
            synchronization_count=int(transfer_bytes > 0),
        ),
        provider_versions=availability.versions,
        scalar_is_complex=scalar_is_complex,
    )


class _ShellMatrixContext:
    def __init__(self, operator: AbstractLinearOperator):
        self.operator = operator
        self.action_count = 0
        self.host_to_device_bytes = 0
        self.device_to_host_bytes = 0

    def snapshot(self) -> tuple[int, int, int]:
        return (
            self.action_count,
            self.host_to_device_bytes,
            self.device_to_host_bytes,
        )

    def _apply(self, x: Any, y: Any, *, adjoint: bool, transpose: bool = False) -> None:
        coordinates = np.asarray(x.getArray(readonly=True)).copy()
        self.action_count += 1
        self.host_to_device_bytes += coordinates.nbytes
        if transpose and np.iscomplexobj(coordinates):
            applied = np.conj(
                _operator_coordinate_action(
                    self.operator, np.conj(coordinates), adjoint=True
                )
            )
        else:
            applied = _operator_coordinate_action(
                self.operator, coordinates, adjoint=adjoint
            )
        output = np.asarray(applied, dtype=coordinates.dtype)
        self.device_to_host_bytes += output.nbytes
        y.getArray()[:] = output

    def mult(self, mat: Any, x: Any, y: Any) -> None:
        del mat
        self._apply(x, y, adjoint=False)

    def multTranspose(self, mat: Any, x: Any, y: Any) -> None:
        del mat
        self._apply(x, y, adjoint=True, transpose=True)

    def multHermitian(self, mat: Any, x: Any, y: Any) -> None:
        del mat
        self._apply(x, y, adjoint=True)

    def multHermitianTranspose(self, mat: Any, x: Any, y: Any) -> None:
        self.multHermitian(mat, x, y)


def _create_shell_matrix(PETSc: Any, operator: AbstractLinearOperator) -> tuple[Any, Any]:
    context = _ShellMatrixContext(operator)
    matrix = PETSc.Mat().createPython(
        [operator.target.size, operator.source.size],
        context=context,
        comm=PETSc.COMM_SELF,
    )
    matrix.setUp()
    return matrix, context


def _create_csr_matrix(PETSc: Any, operator: AbstractLinearOperator) -> tuple[Any, int]:
    _require_sparse(operator, "operator")
    storage = operator.sparse_storage()
    if not storage.canonical or not storage.sorted_indices:
        raise ValueError("SLEPc CSR mode requires canonical sorted CSR storage.")
    values = np.asarray(storage.values, dtype=PETSc.ScalarType)
    indices = np.asarray(storage.indices)
    indptr = np.asarray(storage.indptr)
    matrix = PETSc.Mat().createAIJ(
        size=storage.shape,
        csr=(indptr, indices, values),
        comm=PETSc.COMM_SELF,
    )
    return matrix, values.nbytes + indices.nbytes + indptr.nbytes


def _extract_slepc_pairs(
    prepared: PreparedSLEPcEigenSolve,
    available: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = prepared.eps
    matrix = prepared.matrix
    count = available
    dtype = _complex_coordinate_dtype(prepared.problem)
    values = np.zeros((count,), dtype=dtype)
    right = np.zeros((prepared.plan.dimension, count), dtype=dtype)
    left = np.zeros_like(right)
    for index in range(count):
        xr = matrix.createVecRight()
        yl = matrix.createVecLeft()
        if prepared.scalar_is_complex:
            value = eps.getEigenpair(index, xr)
            eps.getLeftEigenvector(index, yl)
            right[:, index] = np.asarray(xr.getArray(readonly=True))
            left[:, index] = np.asarray(yl.getArray(readonly=True))
        else:
            xi = matrix.createVecRight()
            yi = matrix.createVecLeft()
            value = eps.getEigenpair(index, xr, xi)
            eps.getLeftEigenvector(index, yl, yi)
            right[:, index] = np.asarray(xr.getArray(readonly=True)) + 1j * np.asarray(
                xi.getArray(readonly=True)
            )
            left[:, index] = np.asarray(yl.getArray(readonly=True)) + 1j * np.asarray(
                yi.getArray(readonly=True)
            )
            xi.destroy()
            yi.destroy()
        values[index] = complex(value)
        xr.destroy()
        yl.destroy()
    return values, right, left


def _postprocess_eigenpairs(
    problem: GeneralEigenproblem,
    values: np.ndarray,
    right: np.ndarray,
    left: np.ndarray,
    selection: GeneralEigenSelection,
    requested_count: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if values.size == 0:
        return values, right, left
    if selection.kind == "largest-real":
        primary = -np.real(values)
    elif selection.kind == "largest-magnitude":
        primary = -np.abs(values)
    else:
        primary = np.abs(values - selection.target)
    order = np.lexsort((np.imag(values), np.real(values), primary))
    order = order[:requested_count]
    values = values[order]
    right = right[:, order]
    unpaired_left = left[:, order]
    if values.size <= 1:
        return values, right, unpaired_left
    residual_cost = np.empty((values.size, values.size), dtype=np.float64)
    for row, value in enumerate(values):
        for column in range(values.size):
            residual_cost[row, column] = _left_residual_norm(
                problem, value, unpaired_left[:, column]
            )
    remaining = list(range(values.size))
    pairing: list[int] = []
    for row in range(values.size):
        selected = min(remaining, key=lambda column: (residual_cost[row, column], column))
        pairing.append(selected)
        remaining.remove(selected)
    return values, right, unpaired_left[:, np.asarray(pairing, dtype=np.int64)]


def _verify_and_normalize_pairs(
    problem: GeneralEigenproblem,
    values: np.ndarray,
    right: np.ndarray,
    left: np.ndarray,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    count = values.size
    right = np.asarray(right).copy()
    left = np.asarray(left).copy()
    for index in range(count):
        norm = np.linalg.norm(right[:, index])
        if norm > 0.0:
            right[:, index] /= norm
        mass_right = _mass_action(problem, right[:, index], adjoint=False)
        pairing = np.vdot(left[:, index], mass_right)
        if abs(pairing) > 0.0:
            left[:, index] /= np.conj(pairing)
    right_residuals = np.zeros((count,), dtype=np.float64)
    left_residuals = np.zeros((count,), dtype=np.float64)
    right_relative = np.zeros((count,), dtype=np.float64)
    left_relative = np.zeros((count,), dtype=np.float64)
    for index, value in enumerate(values):
        matrix_right = _operator_coordinate_action(
            problem.operator, right[:, index], adjoint=False
        )
        mass_right = _mass_action(problem, right[:, index], adjoint=False)
        matrix_left = _operator_coordinate_action(
            problem.operator, left[:, index], adjoint=True
        )
        mass_left = _mass_action(problem, left[:, index], adjoint=True)
        right_residuals[index] = np.linalg.norm(matrix_right - value * mass_right)
        left_residuals[index] = np.linalg.norm(matrix_left - np.conj(value) * mass_left)
        right_scale = np.linalg.norm(matrix_right) + abs(value) * np.linalg.norm(
            mass_right
        )
        left_scale = np.linalg.norm(matrix_left) + abs(value) * np.linalg.norm(mass_left)
        right_relative[index] = right_residuals[index] / max(
            right_scale, np.finfo(float).tiny
        )
        left_relative[index] = left_residuals[index] / max(
            left_scale, np.finfo(float).tiny
        )
    mass_right_columns = (
        np.column_stack(
            [
                _mass_action(problem, right[:, index], adjoint=False)
                for index in range(count)
            ]
        )
        if count
        else right.copy()
    )
    pairing_matrix = np.conj(left.T) @ mass_right_columns
    verification_actions = 2 * count * (1 + 2 * int(problem.mass_operator is not None))
    return (
        right_residuals,
        left_residuals,
        right_relative,
        left_relative,
        pairing_matrix,
        right,
        left,
        verification_actions,
    )


def _residual_scales(residual: np.ndarray, relative: np.ndarray, /) -> np.ndarray:
    scales = np.zeros_like(residual)
    np.divide(
        residual,
        relative,
        out=scales,
        where=relative > np.finfo(float).tiny,
    )
    return scales


def _left_residual_norm(
    problem: GeneralEigenproblem,
    value: complex,
    vector: np.ndarray,
    /,
) -> float:
    matrix = _operator_coordinate_action(problem.operator, vector, adjoint=True)
    mass = _mass_action(problem, vector, adjoint=True)
    return float(np.linalg.norm(matrix - np.conj(value) * mass))


def _mass_action(
    problem: GeneralEigenproblem,
    vector: np.ndarray,
    /,
    *,
    adjoint: bool,
) -> np.ndarray:
    if problem.mass_operator is None:
        return np.asarray(vector)
    return _operator_coordinate_action(problem.mass_operator, vector, adjoint=adjoint)


def _operator_coordinate_action(
    operator: AbstractLinearOperator,
    coordinates: Any,
    /,
    *,
    adjoint: bool,
) -> np.ndarray:
    input_space = operator.target if adjoint else operator.source
    output_space = operator.source if adjoint else operator.target
    dtype = np.dtype(_coordinate_dtype(input_space))

    def apply(component: Any) -> Array:
        tree = input_space.unflatten(jnp.asarray(component, dtype=dtype))
        if not adjoint:
            result = operator.mv(tree)
        elif operator.capabilities.adjoint:
            result = operator.adjoint_mv(tree)
        elif np.issubdtype(dtype, np.complexfloating):
            conjugated = input_space.unflatten(
                jnp.conj(jnp.asarray(component, dtype=dtype))
            )
            result = jax.tree.map(jnp.conj, operator.transpose_mv(conjugated))
        else:
            result = operator.transpose_mv(tree)
        return output_space.flatten(result)

    values = np.asarray(coordinates)
    if np.issubdtype(dtype, np.complexfloating):
        return np.asarray(apply(values))
    return np.asarray(apply(np.real(values))) + 1j * np.asarray(apply(np.imag(values)))


def _unflatten_complex_columns(space: Any, coordinates: Array, /) -> PyTree[Array]:
    dtype = np.dtype(_coordinate_dtype(space))
    if np.issubdtype(dtype, np.complexfloating):
        return jax.vmap(space.unflatten, in_axes=1, out_axes=-1)(
            coordinates.astype(dtype)
        )
    real = jax.vmap(space.unflatten, in_axes=1, out_axes=-1)(
        jnp.real(coordinates).astype(dtype)
    )
    imaginary = jax.vmap(space.unflatten, in_axes=1, out_axes=-1)(
        jnp.imag(coordinates).astype(dtype)
    )
    return jax.tree.map(lambda x, y: x + 1j * y, real, imaginary)


def _complex_coordinate_dtype(problem: GeneralEigenproblem, /) -> np.dtype:
    dtype = np.dtype(_coordinate_dtype(problem.operator.source))
    if np.issubdtype(dtype, np.complexfloating):
        return dtype
    return np.dtype(np.complex64 if dtype.itemsize <= 4 else np.complex128)


def _validate_two_sided_operator(operator: AbstractLinearOperator, name: str, /) -> None:
    if not operator.capabilities.adjoint and not operator.capabilities.transpose:
        raise ValueError(
            f"SLEPc two-sided EPS requires {name} adjoint/transpose actions."
        )


def _require_sparse(operator: AbstractLinearOperator, name: str, /) -> None:
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError(
            f"SLEPc operator_mode='csr' requires {name} to be an "
            "AbstractSparseLinearOperator."
        )


def _validate_plan_problem(plan: SLEPcEigenPlan, problem: GeneralEigenproblem, /) -> None:
    if not isinstance(plan, SLEPcEigenPlan):
        raise TypeError("plan must be SLEPcEigenPlan.")
    if not isinstance(problem, GeneralEigenproblem):
        raise TypeError("problem must be GeneralEigenproblem.")
    mass_id = None if problem.mass_operator is None else problem.mass_operator.operator_id
    if (
        plan.problem_id != problem.problem_id
        or plan.operator_id != problem.operator.operator_id
        or plan.mass_operator_id != mass_id
        or plan.dimension != problem.dimension
    ):
        raise ValueError("SLEPc plan belongs to a different symbolic eigenproblem.")


__all__ = [
    "PreparedSLEPcEigenSolve",
    "SLEPC_CAPABILITIES",
    "SLEPcBackend",
    "SLEPcEigenDiagnostics",
    "SLEPcEigenPlan",
    "SLEPcEigenPolicy",
    "SLEPcEigenProvenance",
    "SLEPcEigenResult",
    "SLEPcFailureMode",
    "SLEPcOperatorMode",
    "SLEPcSTOptions",
    "plan_slepc_eigensolve",
    "prepare_slepc_eigensolve",
    "refresh_slepc_eigensolve",
    "release_slepc_eigensolve",
    "slepc_availability",
    "slepc_eigensolve",
]
