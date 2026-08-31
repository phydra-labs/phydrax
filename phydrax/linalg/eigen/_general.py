#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as scipy_linalg
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from .._materialization import MaterializationPolicy, materialize
from .._operators import AbstractLinearOperator, IdentityLinearOperator
from .._policies import (
    DifferentiationPolicy,
    FailurePolicy,
    GMRES,
    LinearSolvePolicy,
    TolerancePolicy,
)
from .._prepared import PreparedLinearSolve
from .._problems import LinearSystem
from .._properties import OperatorCapabilities, OperatorProperties
from .._runtime import (
    prepare as prepare_linear_solve,
    refresh as refresh_linear_solve,
    solve as linear_solve,
)
from .._spaces import _coordinate_dtype
from ..krylov import block_arnoldi


GeneralEigenproblemKind: TypeAlias = Literal["standard", "generalized"]
GeneralEigenSelectionKind: TypeAlias = Literal[
    "all",
    "finite",
    "infinite",
    "closest",
    "largest-magnitude",
    "smallest-magnitude",
    "largest-real",
    "smallest-real",
    "largest-imaginary",
    "smallest-imaginary",
]
SingularMassPolicy: TypeAlias = Literal["report", "error"]


class GeneralEigenSolveStatus(IntEnum):
    """Portable status for a nonsymmetric standard or generalized eigensolve."""

    SUCCESS = 0
    PARTIAL_CONVERGENCE = 1
    NONFINITE_INPUT = 2
    NONFINITE_OUTPUT = 3
    RESIDUAL_TOLERANCE_NOT_MET = 4
    BIORTHOGONALITY_TOLERANCE_NOT_MET = 5
    INDETERMINATE_PENCIL = 6
    SELECTION_MISMATCH = 7


class GeneralEigenproblem(StrictModule):
    """General unbatched pencil ``A x = lambda B x`` in canonical coordinates.

    Omitting ``mass_operator`` defines the standard problem with ``B = I``.  No
    normality, definiteness, or invertibility claim is inferred from either
    operator.
    """

    operator: AbstractLinearOperator
    mass_operator: AbstractLinearOperator | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        mass_operator: AbstractLinearOperator | None = None,
        /,
        *,
        problem_id: str | None = None,
    ):
        _require_general_endomorphism(operator, "operator")
        if mass_operator is not None:
            _require_general_endomorphism(mass_operator, "mass_operator")
            if not operator.source.compatible(mass_operator.source):
                raise ValueError(
                    "The operator and mass operator must use one vector space."
                )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "general-eigenproblem",
                    "operator": operator.operator_id,
                    "mass": None if mass_operator is None else mass_operator.operator_id,
                    "source": operator.source.space_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.operator = operator
        self.mass_operator = mass_operator
        self.problem_id = identifier

    @property
    def dimension(self) -> int:
        return self.operator.source.size

    @property
    def kind(self) -> GeneralEigenproblemKind:
        return "standard" if self.mass_operator is None else "generalized"


class GeneralEigenSelection(StrictModule):
    """Explicit selection and deterministic ordering of general eigenmodes."""

    kind: GeneralEigenSelectionKind = eqx.field(static=True)
    count: int | None = eqx.field(static=True)
    target: complex = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: GeneralEigenSelectionKind = "all",
        /,
        *,
        count: int | None = None,
        target: complex = 0.0,
        selection_id: str | None = None,
    ):
        kinds = (
            "all",
            "finite",
            "infinite",
            "closest",
            "largest-magnitude",
            "smallest-magnitude",
            "largest-real",
            "smallest-real",
            "largest-imaginary",
            "smallest-imaginary",
        )
        if kind not in kinds:
            raise ValueError("Unknown general eigenvalue selection kind.")
        count_ = None if count is None else int(count)
        if count_ is not None and count_ < 1:
            raise ValueError("selection count must be positive or None.")
        target_ = complex(target)
        if not math.isfinite(target_.real) or not math.isfinite(target_.imag):
            raise ValueError("selection target must be finite.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "general-eigen-selection",
                    "selection_kind": kind,
                    "count": count_,
                    "target": [target_.real, target_.imag],
                }
            )
            if selection_id is None
            else str(selection_id)
        )
        if not identifier:
            raise ValueError("selection_id must be non-empty.")
        self.kind = kind
        self.count = count_
        self.target = target_
        self.selection_id = identifier

    @classmethod
    def all(cls, /) -> "GeneralEigenSelection":
        return cls("all")

    @classmethod
    def closest(cls, target: complex, count: int, /) -> "GeneralEigenSelection":
        return cls("closest", count=count, target=target)

    @classmethod
    def finite(cls, /, *, count: int | None = None) -> "GeneralEigenSelection":
        return cls("finite", count=count)

    @classmethod
    def infinite(cls, /, *, count: int | None = None) -> "GeneralEigenSelection":
        return cls("infinite", count=count)


class StandardTransform(StrictModule):
    """Use the pencil operator itself (or ``B^{-1} A`` when ``B`` is present)."""

    def __init__(self):
        return

    @property
    def name(self) -> str:
        return "standard"


class ShiftInvertTransform(StrictModule):
    """Use ``(A - sigma B)^{-1} B`` to expose modes near ``sigma``."""

    shift: complex = eqx.field(static=True)

    def __init__(self, shift: complex, /):
        shift_ = complex(shift)
        if not math.isfinite(shift_.real) or not math.isfinite(shift_.imag):
            raise ValueError("shift must be finite.")
        self.shift = shift_

    @property
    def name(self) -> str:
        return "shift-invert"


class CayleyTransform(StrictModule):
    """Use ``(A - sigma B)^{-1}(A + sigma B)`` near nonzero ``sigma``."""

    shift: complex = eqx.field(static=True)

    def __init__(self, shift: complex, /):
        shift_ = complex(shift)
        if not math.isfinite(shift_.real) or not math.isfinite(shift_.imag):
            raise ValueError("shift must be finite.")
        if shift_ == 0.0:
            raise ValueError("A Cayley transform requires a nonzero shift.")
        self.shift = shift_

    @property
    def name(self) -> str:
        return "cayley"


GeneralEigenTransform: TypeAlias = (
    StandardTransform | ShiftInvertTransform | CayleyTransform
)


class DenseSchurQZ(StrictModule):
    """Host LAPACK Schur/QZ eigenpairs, including homogeneous pencil values."""

    def __init__(self):
        return

    @property
    def name(self) -> str:
        return "dense-schur-qz"


class RestartedArnoldi(StrictModule):
    """Device-native thick-restarted block Arnoldi iteration."""

    subspace_dimension: int | None = eqx.field(static=True)

    def __init__(self, *, subspace_dimension: int | None = None):
        dimension = None if subspace_dimension is None else int(subspace_dimension)
        if dimension is not None and dimension < 3:
            raise ValueError("subspace_dimension must be at least three or None.")
        self.subspace_dimension = dimension

    @property
    def name(self) -> str:
        return "restarted-arnoldi"


GeneralEigenMethod: TypeAlias = DenseSchurQZ | RestartedArnoldi


class GeneralEigenTolerancePolicy(StrictModule):
    """Homogeneous classification, residual, pairing, and rank tolerances."""

    relative: float = eqx.field(static=True)
    absolute: float = eqx.field(static=True)
    biorthogonality: float = eqx.field(static=True)
    homogeneous_relative: float = eqx.field(static=True)
    mass_rank_relative: float = eqx.field(static=True)
    cluster_relative: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative: float = 1e-8,
        absolute: float = 1e-10,
        biorthogonality: float = 1e-7,
        homogeneous_relative: float = 1e-12,
        mass_rank_relative: float = 1e-12,
        cluster_relative: float = 1e-8,
    ):
        values = tuple(
            float(value)
            for value in (
                relative,
                absolute,
                biorthogonality,
                homogeneous_relative,
                mass_rank_relative,
                cluster_relative,
            )
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("General eigen tolerances must be finite and non-negative.")
        (
            self.relative,
            self.absolute,
            self.biorthogonality,
            self.homogeneous_relative,
            self.mass_rank_relative,
            self.cluster_relative,
        ) = values


class GeneralEigenResourcePolicy(StrictModule):
    """Hard dense dimension, retained-state, workspace, and Arnoldi budgets."""

    max_dimension: int = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    krylov_basis_bytes: int = eqx.field(static=True)
    operator_matvecs: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_dimension: int = 4096,
        preparation_bytes: int = 1024 * 1024 * 1024,
        workspace_bytes: int = 2 * 1024 * 1024 * 1024,
        krylov_basis_bytes: int = 1024 * 1024 * 1024,
        operator_matvecs: int = 10_000_000,
    ):
        values = tuple(
            int(value)
            for value in (
                max_dimension,
                preparation_bytes,
                workspace_bytes,
                krylov_basis_bytes,
                operator_matvecs,
            )
        )
        if values[0] < 1 or any(value < 0 for value in values[1:]):
            raise ValueError("max_dimension must be positive and budgets non-negative.")
        (
            self.max_dimension,
            self.preparation_bytes,
            self.workspace_bytes,
            self.krylov_basis_bytes,
            self.operator_matvecs,
        ) = values


class GeneralEigenSolvePolicy(StrictModule):
    """Method, transform, selection, resources, and singular-pencil contract."""

    method: GeneralEigenMethod
    transform: GeneralEigenTransform
    selection: GeneralEigenSelection
    max_steps: int = eqx.field(static=True)
    tolerance: GeneralEigenTolerancePolicy
    resources: GeneralEigenResourcePolicy
    materialization: MaterializationPolicy
    transform_solve: LinearSolvePolicy
    singular_mass: SingularMassPolicy = eqx.field(static=True)
    initial_vector: Array | None
    failure: FailurePolicy

    def __init__(
        self,
        method: GeneralEigenMethod | None = None,
        /,
        *,
        transform: GeneralEigenTransform | None = None,
        selection: GeneralEigenSelection | None = None,
        max_steps: int = 300,
        tolerance: GeneralEigenTolerancePolicy | None = None,
        resources: GeneralEigenResourcePolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        transform_solve: LinearSolvePolicy | None = None,
        singular_mass: SingularMassPolicy = "report",
        initial_vector: ArrayLike | None = None,
        failure: FailurePolicy | None = None,
    ):
        method_ = DenseSchurQZ() if method is None else method
        transform_ = StandardTransform() if transform is None else transform
        selection_ = GeneralEigenSelection.all() if selection is None else selection
        tolerance_ = GeneralEigenTolerancePolicy() if tolerance is None else tolerance
        resources_ = GeneralEigenResourcePolicy() if resources is None else resources
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(method_, (DenseSchurQZ, RestartedArnoldi)):
            raise TypeError("method must be DenseSchurQZ or RestartedArnoldi.")
        if not isinstance(
            transform_, (StandardTransform, ShiftInvertTransform, CayleyTransform)
        ):
            raise TypeError("transform must be a general eigen transform.")
        if not isinstance(selection_, GeneralEigenSelection):
            raise TypeError("selection must be a GeneralEigenSelection.")
        if not isinstance(tolerance_, GeneralEigenTolerancePolicy):
            raise TypeError("tolerance must be a GeneralEigenTolerancePolicy.")
        if not isinstance(resources_, GeneralEigenResourcePolicy):
            raise TypeError("resources must be a GeneralEigenResourcePolicy.")
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        if singular_mass not in ("report", "error"):
            raise ValueError("singular_mass must be 'report' or 'error'.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy.")
        if transform_solve is not None and not isinstance(
            transform_solve, LinearSolvePolicy
        ):
            raise TypeError("transform_solve must be a LinearSolvePolicy or None.")
        steps = int(max_steps)
        if steps < 1:
            raise ValueError("max_steps must be positive.")
        initial = None if initial_vector is None else jnp.asarray(initial_vector)
        if initial is not None:
            if initial.ndim != 1 or initial.size < 1:
                raise ValueError("initial_vector must be a nonempty coordinate vector.")
            if not jnp.issubdtype(initial.dtype, jnp.inexact):
                raise TypeError("initial_vector must use an inexact dtype.")
            initial = eqx.error_if(
                initial,
                jnp.any(~jnp.isfinite(initial)),
                "initial_vector entries must be finite.",
            )
        transform_solve_ = (
            LinearSolvePolicy(
                GMRES(restart=20, stagnation_iterations=20),
                tolerance=TolerancePolicy(
                    relative=tolerance_.relative,
                    absolute=tolerance_.absolute,
                    max_steps=steps,
                ),
                differentiation=DifferentiationPolicy("none"),
                failure=FailurePolicy("status"),
                require_device_binding=True,
            )
            if transform_solve is None
            else transform_solve
        )
        self.method = method_
        self.transform = transform_
        self.selection = selection_
        self.max_steps = steps
        self.tolerance = tolerance_
        self.resources = resources_
        self.materialization = materialization_
        self.transform_solve = transform_solve_
        self.singular_mass = singular_mass
        self.initial_vector = initial
        self.failure = failure_


class GeneralEigenCapabilities(StrictModule):
    """Backend capabilities fixed by one symbolic general-eigen plan."""

    backend: str = eqx.field(static=True)
    host_only: bool = eqx.field(static=True)
    supports_standard: bool = eqx.field(static=True)
    supports_generalized: bool = eqx.field(static=True)
    supports_singular_mass: bool = eqx.field(static=True)
    returns_left_eigenvectors: bool = eqx.field(static=True)
    returns_right_eigenvectors: bool = eqx.field(static=True)
    transforms: tuple[str, ...] = eqx.field(static=True)


class GeneralEigenCostEstimate(StrictModule):
    """Static retained-state, output, workspace, and matvec estimates."""

    backend: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    requested_count: int = eqx.field(static=True)
    input_matrix_bytes: int = eqx.field(static=True)
    output_bytes: int = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    krylov_basis_bytes: int = eqx.field(static=True)
    operator_matvecs: int = eqx.field(static=True)
    exact_preparation: bool = eqx.field(static=True)


class GeneralEigenSolvePlan(StrictModule):
    """Immutable symbolic plan for one standard or generalized general pencil."""

    policy: GeneralEigenSolvePolicy
    capabilities: GeneralEigenCapabilities = eqx.field(static=True)
    cost: GeneralEigenCostEstimate = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    mass_operator_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedGeneralEigenSolve(StrictModule):
    """Numeric dense pencil or matrix-free operators and prepared transform solves."""

    problem: GeneralEigenproblem
    matrix: Array
    mass_matrix: Array
    transform_solver: PreparedLinearSolve | None
    left_transform_solver: PreparedLinearSolve | None
    mass_rank: Array
    shifted_rank: Array
    plan: GeneralEigenSolvePlan
    prepared_id: str = eqx.field(static=True)
    operator_fingerprint: str = eqx.field(static=True)
    mass_operator_fingerprint: str | None = eqx.field(static=True)
    numeric_version: Array
    refresh_count: Array


class GeneralEigenSolveDiagnostics(StrictModule):
    """Homogeneous classification, paired residuals, and convergence evidence."""

    right_residual_norms: Array
    left_residual_norms: Array
    right_relative_residuals: Array
    left_relative_residuals: Array
    pairing_diagonal: Array
    pairing_matrix: Array
    biorthogonality_error: Array
    eigenvalue_condition_estimates: Array
    finite_mask: Array
    infinite_mask: Array
    indeterminate_mask: Array
    input_finite: Array
    output_finite: Array
    converged: Array
    converged_mask: Array
    converged_count: Array
    mass_rank: Array
    mass_singular: Array
    shifted_rank: Array
    selected_count: Array
    available_count: Array
    requested_count: int = eqx.field(static=True)
    arnoldi_action_count: Array
    transform_solve_count: Array
    backend_converged: Array
    decomposition_count: Array
    preparation_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class GeneralEigenSolveProvenance(StrictModule):
    """Host backend, identities, coordinate convention, and numeric version."""

    backend: str = eqx.field(static=True)
    host_only: bool = eqx.field(static=True)
    host_library: str = eqx.field(static=True)
    algorithm: str = eqx.field(static=True)
    transform: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    mass_operator_id: str | None = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    capabilities: GeneralEigenCapabilities = eqx.field(static=True)
    numeric_version: Array


class GeneralEigenSolveResult(StrictModule):
    """Selected homogeneous eigenvalues and paired left/right eigenvectors."""

    eigenvalues: Array
    alpha: Array
    beta: Array
    right_eigenvectors: PyTree[Array]
    left_eigenvectors: PyTree[Array]
    right_eigenvector_coordinates: Array
    left_eigenvector_coordinates: Array
    status: Array
    diagnostics: GeneralEigenSolveDiagnostics
    provenance: GeneralEigenSolveProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(GeneralEigenSolveStatus.SUCCESS)

    @property
    def finite_mask(self) -> Array:
        return self.diagnostics.finite_mask

    @property
    def infinite_mask(self) -> Array:
        return self.diagnostics.infinite_mask

    @property
    def indeterminate_mask(self) -> Array:
        return self.diagnostics.indeterminate_mask


def plan_general_eigensolve(
    problem: GeneralEigenproblem,
    policy: GeneralEigenSolvePolicy | None = None,
    /,
) -> GeneralEigenSolvePlan:
    """Select a host dense-QZ or device-native matrix-free Arnoldi solve."""
    if not isinstance(problem, GeneralEigenproblem):
        raise TypeError("problem must be a GeneralEigenproblem.")
    selected = GeneralEigenSolvePolicy() if policy is None else policy
    if not isinstance(selected, GeneralEigenSolvePolicy):
        raise TypeError("policy must be a GeneralEigenSolvePolicy or None.")
    dimension = problem.dimension
    if dimension > selected.resources.max_dimension:
        raise ValueError(
            f"General eigen dimension {dimension} exceeds limit "
            f"{selected.resources.max_dimension}."
        )
    if selected.initial_vector is not None and selected.initial_vector.shape != (
        dimension,
    ):
        raise ValueError("initial_vector must match the problem coordinate dimension.")
    if selected.selection.count is not None and selected.selection.count > dimension:
        raise ValueError("selection count cannot exceed the problem dimension.")
    if isinstance(selected.method, DenseSchurQZ):
        if not isinstance(selected.transform, StandardTransform):
            raise ValueError(
                "DenseSchurQZ uses the original pencil and StandardTransform."
            )
    else:
        count = selected.selection.count
        if count is None:
            raise ValueError("RestartedArnoldi requires an explicit selection count.")
        if selected.selection.kind in ("all", "finite", "infinite"):
            raise ValueError(
                "RestartedArnoldi requires an ordered finite-mode selection."
            )
        if count >= dimension - 1:
            raise ValueError("RestartedArnoldi requires count < dimension - 1.")
        if 2 * count > dimension:
            raise ValueError(
                "RestartedArnoldi requires room for at least two retained blocks "
                "(2 * count <= dimension)."
            )
        if selected.max_steps < 2 * count:
            raise ValueError(
                "RestartedArnoldi max_steps must admit at least two retained blocks."
            )
        if not problem.operator.capabilities.transpose or (
            problem.mass_operator is not None
            and not problem.mass_operator.capabilities.transpose
        ):
            raise ValueError("RestartedArnoldi requires operator transpose capabilities.")
        needs_transform_solve = problem.mass_operator is not None or not isinstance(
            selected.transform, StandardTransform
        )
        if needs_transform_solve and (
            not isinstance(selected.transform_solve.method, GMRES)
            or not selected.transform_solve.require_device_binding
        ):
            raise ValueError(
                "Matrix-free spectral transforms require a device-bound GMRES "
                "transform_solve policy."
            )
        if problem.mass_operator is not None:
            properties = problem.mass_operator.properties
            full_rank = properties.certifies("rank") and properties.rank == dimension
            if not properties.certifies("positive_definite") and not full_rank:
                raise ValueError(
                    "Matrix-free generalized Arnoldi requires a certified nonsingular "
                    "mass operator; use DenseSchurQZ to classify an uncertified or "
                    "singular mass pencil."
                )
        coordinate_dtype = np.dtype(_coordinate_dtype(problem.operator.source))
        if (
            not np.issubdtype(coordinate_dtype, np.complexfloating)
            and isinstance(selected.transform, (ShiftInvertTransform, CayleyTransform))
            and selected.transform.shift.imag != 0.0
        ):
            raise ValueError(
                "A complex spectral shift requires a complex coordinate space."
            )
        if isinstance(selected.transform, (ShiftInvertTransform, CayleyTransform)):
            if selected.selection.kind != "closest":
                raise ValueError(
                    "Shift-invert and Cayley transforms require selection kind 'closest'."
                )
            if abs(selected.selection.target - selected.transform.shift) > (
                selected.tolerance.absolute
                + selected.tolerance.relative * max(abs(selected.transform.shift), 1.0)
            ):
                raise ValueError(
                    "Closest-selection target must equal the transform shift."
                )
        elif selected.selection.kind == "closest" and selected.selection.target != 0.0:
            raise ValueError(
                "A nonzero closest target requires ShiftInvertTransform or "
                "CayleyTransform."
            )
        subspace = _arnoldi_subspace_dimension(selected.method, count, dimension)
        if subspace < 2 * count:
            raise ValueError(
                "Arnoldi subspace_dimension must admit at least two retained blocks."
            )
    capabilities = _capabilities(selected.method)
    cost = _general_eigen_cost(problem, selected, capabilities.backend)
    resources = selected.resources
    for value, limit, label in (
        (cost.preparation_bytes, resources.preparation_bytes, "preparation"),
        (cost.workspace_bytes, resources.workspace_bytes, "workspace"),
        (cost.krylov_basis_bytes, resources.krylov_basis_bytes, "Krylov basis"),
        (cost.operator_matvecs, resources.operator_matvecs, "operator matvec"),
    ):
        if value > limit:
            raise ValueError(
                f"General eigen {label} estimate {value} exceeds budget {limit}."
            )
    mass_id = None if problem.mass_operator is None else problem.mass_operator.operator_id
    return GeneralEigenSolvePlan(
        policy=selected,
        capabilities=capabilities,
        cost=cost,
        problem_id=problem.problem_id,
        operator_id=problem.operator.operator_id,
        mass_operator_id=mass_id,
        plan_id=canonical_fingerprint(
            {
                "kind": "general-eigen-plan",
                "problem": problem.problem_id,
                "operator": problem.operator.operator_id,
                "mass": mass_id,
                "method": selected.method.name,
                "transform": selected.transform.name,
                "selection": selected.selection.selection_id,
                "max_steps": selected.max_steps,
                "backend": capabilities.backend,
                "singular_mass": selected.singular_mass,
                "transform_solve": {
                    "method": selected.transform_solve.method.name,
                    "relative": selected.transform_solve.tolerance.relative,
                    "absolute": selected.transform_solve.tolerance.absolute,
                    "max_steps": selected.transform_solve.tolerance.max_steps,
                    "device": selected.transform_solve.require_device_binding,
                },
            }
        ),
    )


def prepare_general_eigensolve(
    problem: GeneralEigenproblem,
    policy: GeneralEigenSolvePolicy | GeneralEigenSolvePlan | None = None,
    /,
) -> PreparedGeneralEigenSolve:
    """Prepare dense QZ state or reusable matrix-free transform solves."""
    plan = (
        policy
        if isinstance(policy, GeneralEigenSolvePlan)
        else plan_general_eigensolve(problem, policy)
    )
    _validate_general_plan(problem, plan)
    return _prepare_general_numeric(problem, plan, numeric_version=0, refresh_count=0)


def refresh_general_eigensolve(
    prepared: PreparedGeneralEigenSolve,
    problem: GeneralEigenproblem,
    /,
) -> PreparedGeneralEigenSolve:
    """Refresh pencil values while preserving the symbolic plan and prepared ID."""
    if not isinstance(prepared, PreparedGeneralEigenSolve):
        raise TypeError("prepared must be a PreparedGeneralEigenSolve.")
    _validate_general_plan(problem, prepared.plan)
    return _prepare_general_numeric(
        problem,
        prepared.plan,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        refresh_count=prepared.refresh_count + jnp.asarray(1, dtype=jnp.int32),
        prepared_id=prepared.prepared_id,
        transform_solver=prepared.transform_solver,
        left_transform_solver=prepared.left_transform_solver,
    )


def general_eigensolve(
    problem_or_prepared: GeneralEigenproblem | PreparedGeneralEigenSolve,
    /,
    *,
    policy: GeneralEigenSolvePolicy | GeneralEigenSolvePlan | None = None,
) -> GeneralEigenSolveResult:
    """Solve a dense or matrix-free general standard or generalized pencil."""
    if isinstance(problem_or_prepared, PreparedGeneralEigenSolve):
        if policy is not None:
            raise ValueError("policy must be omitted for a prepared general eigensolve.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, GeneralEigenproblem):
        prepared = prepare_general_eigensolve(problem_or_prepared, policy)
    else:
        raise TypeError("Expected a GeneralEigenproblem or PreparedGeneralEigenSolve.")
    dense_method = isinstance(prepared.plan.policy.method, DenseSchurQZ)
    if not dense_method:
        return _general_eigensolve_native(prepared)
    matrix = np.asarray(prepared.matrix)
    mass = np.asarray(prepared.mass_matrix)
    input_finite = bool(np.all(np.isfinite(matrix)) and np.all(np.isfinite(mass)))
    requested = _requested_count(
        prepared.plan.policy.selection,
        prepared.problem.dimension,
    )
    host = (
        _dense_host_eigensolve(
            matrix,
            mass,
            prepared.problem.mass_operator is not None,
        )
        if input_finite
        else _nonfinite_host_result(prepared.problem.dimension, requested)
    )
    alpha_np, beta_np, right_np, left_np, available, backend_converged, matvecs = host
    finite_np, infinite_np, indeterminate_np = _classify_homogeneous(
        alpha_np,
        beta_np,
        prepared.plan.policy.tolerance,
    )
    requested = _selection_requested_count(
        prepared.plan.policy.selection,
        alpha_np.size,
        finite_np,
        infinite_np,
    )
    indices = _selection_indices(
        alpha_np,
        beta_np,
        finite_np,
        infinite_np,
        prepared.plan.policy.selection,
    )
    alpha_np = alpha_np[indices]
    beta_np = beta_np[indices]
    right_np = right_np[:, indices]
    left_np = left_np[:, indices]
    finite_np = finite_np[indices]
    infinite_np = infinite_np[indices]
    indeterminate_np = indeterminate_np[indices]
    right_np, left_np, pairing_matrix_np, condition_np = _normalize_paired_vectors(
        prepared,
        alpha_np,
        beta_np,
        right_np,
        left_np,
        finite_np,
        infinite_np,
        prepared.plan.policy.tolerance,
    )
    eigenvalues_np = _ordinary_eigenvalues(alpha_np, beta_np, finite_np, infinite_np)
    (
        right_residual_np,
        left_residual_np,
        right_relative_np,
        left_relative_np,
        right_scale_np,
        left_scale_np,
    ) = _homogeneous_residuals(
        prepared,
        alpha_np,
        beta_np,
        right_np,
        left_np,
    )
    selected_count = int(alpha_np.size)
    selection_satisfied = selected_count == requested
    tolerance = prepared.plan.policy.tolerance
    residual_ok = bool(
        np.all(
            (
                right_residual_np
                <= tolerance.absolute + tolerance.relative * right_scale_np
            )
            & (
                left_residual_np
                <= tolerance.absolute + tolerance.relative * left_scale_np
            )
        )
    )
    identity = np.eye(alpha_np.size, dtype=pairing_matrix_np.dtype)
    pairing_error = float(np.max(np.abs(pairing_matrix_np - identity), initial=0.0))
    pairing_ok = pairing_error <= tolerance.biorthogonality
    output_finite = bool(
        np.all(np.isfinite(alpha_np))
        and np.all(np.isfinite(beta_np))
        and np.all(np.isfinite(right_np))
        and np.all(np.isfinite(left_np))
        and np.all(np.isfinite(right_residual_np))
        and np.all(np.isfinite(left_residual_np))
    )
    if not input_finite:
        status = GeneralEigenSolveStatus.NONFINITE_INPUT
    elif np.any(indeterminate_np):
        status = GeneralEigenSolveStatus.INDETERMINATE_PENCIL
    elif not output_finite:
        status = GeneralEigenSolveStatus.NONFINITE_OUTPUT
    elif not backend_converged:
        status = GeneralEigenSolveStatus.PARTIAL_CONVERGENCE
    elif not selection_satisfied:
        status = GeneralEigenSolveStatus.SELECTION_MISMATCH
    elif not residual_ok:
        status = GeneralEigenSolveStatus.RESIDUAL_TOLERANCE_NOT_MET
    elif not pairing_ok:
        status = GeneralEigenSolveStatus.BIORTHOGONALITY_TOLERANCE_NOT_MET
    else:
        status = GeneralEigenSolveStatus.SUCCESS
    if (
        prepared.plan.policy.failure.mode == "error"
        and status != GeneralEigenSolveStatus.SUCCESS
    ):
        raise RuntimeError(
            f"General eigensolve did not satisfy its numerical contract: {status.name}."
        )
    complex_dtype = _complex_coordinate_dtype(prepared.problem)
    alpha = jnp.asarray(alpha_np, dtype=complex_dtype)
    beta = jnp.asarray(beta_np, dtype=complex_dtype)
    right = jnp.asarray(right_np, dtype=complex_dtype)
    left = jnp.asarray(left_np, dtype=complex_dtype)
    right_vectors = _unflatten_complex_columns(prepared.problem.operator.source, right)
    left_vectors = _unflatten_complex_columns(prepared.problem.operator.source, left)
    finite = jnp.asarray(finite_np)
    infinite = jnp.asarray(infinite_np)
    indeterminate = jnp.asarray(indeterminate_np)
    status_array = jnp.asarray(int(status), dtype=jnp.int32)
    diagnostics = GeneralEigenSolveDiagnostics(
        right_residual_norms=jnp.asarray(right_residual_np),
        left_residual_norms=jnp.asarray(left_residual_np),
        right_relative_residuals=jnp.asarray(right_relative_np),
        left_relative_residuals=jnp.asarray(left_relative_np),
        pairing_diagonal=jnp.asarray(
            np.diag(pairing_matrix_np),
            dtype=complex_dtype,
        ),
        pairing_matrix=jnp.asarray(pairing_matrix_np, dtype=complex_dtype),
        biorthogonality_error=jnp.asarray(pairing_error),
        eigenvalue_condition_estimates=jnp.asarray(condition_np),
        finite_mask=finite,
        infinite_mask=infinite,
        indeterminate_mask=indeterminate,
        input_finite=jnp.asarray(input_finite),
        output_finite=jnp.asarray(output_finite),
        converged=jnp.asarray(status == GeneralEigenSolveStatus.SUCCESS),
        mass_rank=prepared.mass_rank,
        mass_singular=prepared.mass_rank < prepared.problem.dimension,
        converged_mask=jnp.full(
            (selected_count,),
            status == GeneralEigenSolveStatus.SUCCESS,
            dtype=bool,
        ),
        converged_count=jnp.asarray(
            selected_count if status == GeneralEigenSolveStatus.SUCCESS else 0,
            dtype=jnp.int32,
        ),
        shifted_rank=prepared.shifted_rank,
        selected_count=jnp.asarray(selected_count, dtype=jnp.int32),
        available_count=jnp.asarray(available, dtype=jnp.int32),
        requested_count=requested,
        arnoldi_action_count=jnp.asarray(matvecs, dtype=jnp.int32),
        transform_solve_count=jnp.asarray(
            (matvecs + selected_count if prepared.transform_solver is not None else 0),
            dtype=jnp.int32,
        ),
        backend_converged=jnp.asarray(backend_converged),
        decomposition_count=jnp.asarray(
            1
            if dense_method
            else 2
            * _arnoldi_cycle_configuration(
                prepared.plan.policy,
                prepared.problem.dimension,
                requested,
            )[2],
            dtype=jnp.int32,
        ),
        preparation_bytes=prepared.plan.cost.preparation_bytes,
        workspace_bytes=prepared.plan.cost.workspace_bytes,
    )
    return GeneralEigenSolveResult(
        eigenvalues=jnp.asarray(eigenvalues_np, dtype=complex_dtype),
        alpha=alpha,
        beta=beta,
        right_eigenvectors=right_vectors,
        left_eigenvectors=left_vectors,
        right_eigenvector_coordinates=right,
        left_eigenvector_coordinates=left,
        status=status_array,
        diagnostics=diagnostics,
        provenance=GeneralEigenSolveProvenance(
            backend=prepared.plan.capabilities.backend,
            host_only=prepared.plan.capabilities.host_only,
            host_library=("scipy" if dense_method else "jax-native; numpy-host-pairing"),
            algorithm=prepared.plan.policy.method.name,
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
            source_space_id=prepared.problem.operator.source.space_id,
            target_space_id=prepared.problem.operator.target.space_id,
            selection_id=prepared.plan.policy.selection.selection_id,
            coordinate_convention=(
                "canonical-coordinate homogeneous alpha/beta; left vectors are "
                "canonical Euclidean covectors"
            ),
            capabilities=prepared.plan.capabilities,
            numeric_version=prepared.numeric_version,
        ),
    )


def _general_eigensolve_native(
    prepared: PreparedGeneralEigenSolve,
    /,
) -> GeneralEigenSolveResult:
    """Device-staged result assembly for the matrix-free Arnoldi backend."""
    policy = prepared.plan.policy
    count = policy.selection.count
    if count is None:
        raise ValueError("RestartedArnoldi requires an explicit selection count.")
    initial = _initial_arnoldi_block(prepared, count)
    (
        right_mu,
        right,
        _,
        right_locked,
        right_matvecs,
        right_valid,
    ) = _native_restarted_arnoldi_evidence(
        prepared,
        initial,
        adjoint_action=False,
    )
    (
        left_mu,
        left,
        _,
        left_locked,
        left_matvecs,
        left_valid,
    ) = _native_restarted_arnoldi_evidence(
        prepared,
        jnp.conj(initial),
        adjoint_action=True,
    )
    if prepared.left_transform_solver is not None:
        left = jax.vmap(
            lambda vector: _solve_complexified_coordinates(
                prepared.left_transform_solver,
                vector,
            ),
            in_axes=1,
            out_axes=1,
        )(left)
        left_valid = left_valid & jnp.all(jnp.isfinite(left))
    right_values = _jax_inverse_transformed_values(
        right_mu,
        policy.transform,
        adjoint=False,
    )
    left_values = jnp.conj(
        _jax_inverse_transformed_values(
            left_mu,
            policy.transform,
            adjoint=True,
        )
    )
    pairing = _jax_greedy_eigenvalue_pairing(right_values, left_values)
    left = left[:, pairing]
    left_locked = left_locked[pairing]
    mass_right = (
        right
        if prepared.problem.mass_operator is None
        else jax.vmap(
            lambda vector: _operator_coordinate_action(
                prepared.problem.mass_operator,
                vector,
                adjoint_action=False,
            ),
            in_axes=1,
            out_axes=1,
        )(right)
    )
    overlap = jnp.conj(left.T) @ mass_right
    value_scale = jnp.maximum(
        jnp.maximum(jnp.abs(right_values[:, None]), jnp.abs(right_values[None, :])),
        1,
    )
    same_cluster = (
        jnp.abs(right_values[:, None] - right_values[None, :])
        <= policy.tolerance.cluster_relative * value_scale
    )
    cluster_overlap = jnp.where(same_cluster, overlap, 0)
    left = left @ jnp.conj(jnp.linalg.pinv(cluster_overlap).T)
    mass_right = (
        right
        if prepared.problem.mass_operator is None
        else jax.vmap(
            lambda vector: _operator_coordinate_action(
                prepared.problem.mass_operator,
                vector,
                adjoint_action=False,
            ),
            in_axes=1,
            out_axes=1,
        )(right)
    )
    pairing_matrix = jnp.conj(left.T) @ mass_right
    right_residuals, right_scales = _jax_original_pencil_residuals(
        prepared,
        right_values,
        right,
        adjoint_action=False,
    )
    left_residuals, left_scales = _jax_original_pencil_residuals(
        prepared,
        jnp.conj(right_values),
        left,
        adjoint_action=True,
    )
    right_ok = (
        right_residuals
        <= policy.tolerance.absolute + policy.tolerance.relative * right_scales
    )
    left_ok = (
        left_residuals
        <= policy.tolerance.absolute + policy.tolerance.relative * left_scales
    )
    converged_mask = right_locked & left_locked & right_ok & left_ok
    converged_count = jnp.sum(converged_mask, dtype=jnp.int32)
    pairing_error = jnp.max(
        jnp.abs(pairing_matrix - jnp.eye(count, dtype=pairing_matrix.dtype)),
        initial=0,
    )
    pairing_ok = pairing_error <= policy.tolerance.biorthogonality
    output_finite = (
        jnp.all(jnp.isfinite(right_values))
        & jnp.all(jnp.isfinite(right))
        & jnp.all(jnp.isfinite(left))
        & jnp.all(jnp.isfinite(right_residuals))
        & jnp.all(jnp.isfinite(left_residuals))
    )
    backend_converged = right_valid & left_valid & (converged_count == count)
    residual_ok = jnp.all(right_ok & left_ok)
    status = jnp.asarray(int(GeneralEigenSolveStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~pairing_ok,
        int(GeneralEigenSolveStatus.BIORTHOGONALITY_TOLERANCE_NOT_MET),
        status,
    )
    status = jnp.where(
        ~residual_ok,
        int(GeneralEigenSolveStatus.RESIDUAL_TOLERANCE_NOT_MET),
        status,
    )
    status = jnp.where(
        ~backend_converged,
        int(GeneralEigenSolveStatus.PARTIAL_CONVERGENCE),
        status,
    )
    status = jnp.where(
        ~output_finite,
        int(GeneralEigenSolveStatus.NONFINITE_OUTPUT),
        status,
    ).astype(jnp.int32)
    alpha = right_values
    beta = jnp.ones_like(alpha)
    if policy.failure.mode == "error":
        alpha = eqx.error_if(
            alpha,
            status != int(GeneralEigenSolveStatus.SUCCESS),
            "General eigensolve did not satisfy its numerical contract.",
        )
    tiny = jnp.finfo(jnp.real(alpha).dtype).tiny
    right_relative = right_residuals / jnp.maximum(right_scales, tiny)
    left_relative = left_residuals / jnp.maximum(left_scales, tiny)
    pairing_diagonal = jnp.diag(pairing_matrix)
    conditions = (
        jnp.linalg.norm(right, axis=0)
        * jnp.linalg.norm(left, axis=0)
        / jnp.maximum(jnp.abs(pairing_diagonal), tiny)
    )
    matvecs = right_matvecs + left_matvecs
    _, _, restart_count = _arnoldi_cycle_configuration(
        policy,
        prepared.problem.dimension,
        count,
    )
    diagnostics = GeneralEigenSolveDiagnostics(
        right_residual_norms=right_residuals,
        left_residual_norms=left_residuals,
        right_relative_residuals=right_relative,
        left_relative_residuals=left_relative,
        pairing_diagonal=pairing_diagonal,
        pairing_matrix=pairing_matrix,
        biorthogonality_error=pairing_error,
        eigenvalue_condition_estimates=conditions,
        finite_mask=jnp.ones((count,), dtype=bool),
        infinite_mask=jnp.zeros((count,), dtype=bool),
        indeterminate_mask=jnp.zeros((count,), dtype=bool),
        input_finite=jnp.asarray(True),
        output_finite=output_finite,
        converged=status == int(GeneralEigenSolveStatus.SUCCESS),
        converged_mask=converged_mask,
        converged_count=converged_count,
        mass_rank=prepared.mass_rank,
        mass_singular=jnp.asarray(False),
        shifted_rank=prepared.shifted_rank,
        selected_count=jnp.asarray(count, dtype=jnp.int32),
        available_count=converged_count,
        requested_count=count,
        arnoldi_action_count=matvecs,
        transform_solve_count=jnp.where(
            prepared.transform_solver is not None,
            matvecs + count,
            0,
        ).astype(jnp.int32),
        backend_converged=backend_converged,
        decomposition_count=jnp.asarray(2 * restart_count, dtype=jnp.int32),
        preparation_bytes=prepared.plan.cost.preparation_bytes,
        workspace_bytes=prepared.plan.cost.workspace_bytes,
    )
    return GeneralEigenSolveResult(
        eigenvalues=alpha,
        alpha=alpha,
        beta=beta,
        right_eigenvectors=_unflatten_complex_columns(
            prepared.problem.operator.source,
            right,
        ),
        left_eigenvectors=_unflatten_complex_columns(
            prepared.problem.operator.source,
            left,
        ),
        right_eigenvector_coordinates=right,
        left_eigenvector_coordinates=left,
        status=status,
        diagnostics=diagnostics,
        provenance=GeneralEigenSolveProvenance(
            backend=prepared.plan.capabilities.backend,
            host_only=False,
            host_library="none",
            algorithm=policy.method.name,
            transform=policy.transform.name,
            problem_id=prepared.problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=prepared.problem.operator.operator_id,
            mass_operator_id=(
                None
                if prepared.problem.mass_operator is None
                else prepared.problem.mass_operator.operator_id
            ),
            source_space_id=prepared.problem.operator.source.space_id,
            target_space_id=prepared.problem.operator.target.space_id,
            selection_id=policy.selection.selection_id,
            coordinate_convention=(
                "canonical-coordinate homogeneous alpha/beta; left vectors are "
                "canonical Euclidean covectors"
            ),
            capabilities=prepared.plan.capabilities,
            numeric_version=prepared.numeric_version,
        ),
    )


def _jax_greedy_eigenvalue_pairing(right: Array, left: Array, /) -> Array:
    count = right.size
    initial = (
        jnp.zeros((count,), dtype=jnp.int32),
        jnp.zeros((count,), dtype=bool),
    )

    def pair(index, state):
        indices, used = state
        distance = jnp.abs(left - right[index])
        selected = jnp.argmin(jnp.where(used, jnp.inf, distance)).astype(indices.dtype)
        return indices.at[index].set(selected), used.at[selected].set(True)

    indices, _ = jax.lax.fori_loop(0, count, pair, initial)
    return indices


def _require_general_endomorphism(operator: object, name: str, /) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError(f"{name} must be an AbstractLinearOperator.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError(f"{name} must be an unbatched endomorphism.")
    if not jnp.issubdtype(_coordinate_dtype(operator.source), jnp.inexact):
        raise TypeError(f"{name} requires real or complex inexact coordinates.")


def _capabilities(method: GeneralEigenMethod, /) -> GeneralEigenCapabilities:
    dense = isinstance(method, DenseSchurQZ)
    return GeneralEigenCapabilities(
        backend="scipy-lapack-host" if dense else "phydrax-native-restarted-arnoldi",
        host_only=dense,
        supports_standard=True,
        supports_generalized=True,
        supports_singular_mass=dense,
        returns_left_eigenvectors=True,
        returns_right_eigenvectors=True,
        transforms=("standard",) if dense else ("standard", "shift-invert", "cayley"),
    )


def _complex_coordinate_dtype(problem: GeneralEigenproblem, /) -> np.dtype:
    dtype = np.dtype(_coordinate_dtype(problem.operator.source))
    if np.issubdtype(dtype, np.complexfloating):
        return dtype
    return np.dtype(np.complex64 if dtype.itemsize <= 4 else np.complex128)


def _requested_count(selection: GeneralEigenSelection, dimension: int, /) -> int:
    return dimension if selection.count is None else selection.count


def _selection_requested_count(
    selection: GeneralEigenSelection,
    available: int,
    finite: np.ndarray,
    infinite: np.ndarray,
    /,
) -> int:
    if selection.count is not None:
        return selection.count
    if selection.kind == "finite":
        return int(np.count_nonzero(finite))
    if selection.kind == "infinite":
        return int(np.count_nonzero(infinite))
    return available


def _arnoldi_subspace_dimension(
    method: RestartedArnoldi,
    count: int,
    dimension: int,
    /,
) -> int:
    if method.subspace_dimension is not None:
        return min(method.subspace_dimension, dimension)
    return min(dimension, max(2 * count + 2, 20))


def _general_eigen_cost(
    problem: GeneralEigenproblem,
    policy: GeneralEigenSolvePolicy,
    backend: str,
    /,
) -> GeneralEigenCostEstimate:
    dimension = problem.dimension
    count = _requested_count(policy.selection, dimension)
    itemsize = _complex_coordinate_dtype(problem).itemsize
    input_matrices = 1 if problem.mass_operator is None else 2
    dense_input_bytes = input_matrices * dimension * dimension * itemsize
    output_bytes = (2 * dimension * count + 3 * count) * itemsize
    if isinstance(policy.method, DenseSchurQZ):
        input_bytes = dense_input_bytes
        preparation = input_bytes
        workspace = 16 * dimension * dimension * itemsize
        krylov = 0
        matvecs = 0
        exact = True
    else:
        input_bytes = 0
        subspace = _arnoldi_subspace_dimension(policy.method, count, dimension)
        inner_steps = policy.transform_solve.tolerance.max_steps
        if inner_steps is None:
            inner_steps = max(10 * dimension, 1)
        needs_solve = problem.mass_operator is not None or not isinstance(
            policy.transform, StandardTransform
        )
        preparation = 2 * dimension * itemsize if needs_solve else 0
        workspace = (4 * dimension * subspace + 4 * subspace * subspace) * itemsize
        krylov = 2 * dimension * subspace * itemsize
        numerator_actions = (
            2
            if isinstance(policy.transform, CayleyTransform)
            and problem.mass_operator is not None
            else 1
        )
        action_cost = numerator_actions + (inner_steps if needs_solve else 0)
        _, _, restart_count = _arnoldi_cycle_configuration(
            policy,
            dimension,
            count,
        )
        residual_actions = 2 * (restart_count + 1) * count * input_matrices
        final_transform_solves = count * inner_steps if needs_solve else 0
        matvecs = (
            2 * policy.max_steps * action_cost + residual_actions + final_transform_solves
        )
        exact = False
    return GeneralEigenCostEstimate(
        backend=backend,
        dimension=dimension,
        requested_count=count,
        input_matrix_bytes=input_bytes,
        output_bytes=output_bytes,
        preparation_bytes=preparation,
        workspace_bytes=workspace,
        krylov_basis_bytes=krylov,
        operator_matvecs=matvecs,
        exact_preparation=exact,
    )


def _arnoldi_cycle_configuration(
    policy: GeneralEigenSolvePolicy,
    dimension: int,
    count: int,
    /,
) -> tuple[int, int, int]:
    requested_subspace = _arnoldi_subspace_dimension(
        policy.method,
        count,
        dimension,
    )
    cycle_dimension = min(requested_subspace, max(policy.max_steps, 2 * count))
    blocks = min(
        max(cycle_dimension // count, 2),
        dimension // count,
    )
    cycle_dimension = blocks * count
    restart_count = max(policy.max_steps // cycle_dimension, 1)
    return blocks, cycle_dimension, restart_count


def _validate_general_plan(
    problem: GeneralEigenproblem,
    plan: GeneralEigenSolvePlan,
    /,
) -> None:
    if not isinstance(problem, GeneralEigenproblem):
        raise TypeError("problem must be a GeneralEigenproblem.")
    if not isinstance(plan, GeneralEigenSolvePlan):
        raise TypeError("plan must be a GeneralEigenSolvePlan.")
    mass_id = None if problem.mass_operator is None else problem.mass_operator.operator_id
    if (
        problem.problem_id != plan.problem_id
        or problem.operator.operator_id != plan.operator_id
        or mass_id != plan.mass_operator_id
    ):
        raise ValueError("General eigen plan belongs to a different symbolic pencil.")


def _prepare_general_numeric(
    problem: GeneralEigenproblem,
    plan: GeneralEigenSolvePlan,
    *,
    numeric_version: Any,
    refresh_count: Any,
    prepared_id: str | None = None,
    transform_solver: PreparedLinearSolve | None = None,
    left_transform_solver: PreparedLinearSolve | None = None,
) -> PreparedGeneralEigenSolve:
    dimension = problem.dimension
    coordinate_dtype = np.dtype(_coordinate_dtype(problem.operator.source))
    mass_fingerprint = (
        None
        if problem.mass_operator is None
        else canonical_fingerprint(array_tree_fingerprint(problem.mass_operator))
    )
    if isinstance(plan.policy.method, DenseSchurQZ):
        matrix = jnp.asarray(materialize(problem.operator, plan.policy.materialization))
        if matrix.shape != (dimension, dimension):
            raise ValueError("Materialized operator must be a square coordinate matrix.")
        if problem.mass_operator is None:
            mass = jnp.eye(dimension, dtype=matrix.dtype)
        else:
            mass = jnp.asarray(
                materialize(problem.mass_operator, plan.policy.materialization)
            )
            if mass.shape != (dimension, dimension):
                raise ValueError(
                    "Materialized mass operator must match the pencil dimension."
                )
        common_dtype = jnp.result_type(matrix.dtype, mass.dtype)
        matrix = matrix.astype(common_dtype)
        mass = mass.astype(common_dtype)
        mass_rank = _numerical_rank(
            np.asarray(mass),
            plan.policy.tolerance.mass_rank_relative,
        )
        if mass_rank < dimension and plan.policy.singular_mass == "error":
            raise ValueError(
                "The generalized mass matrix is singular under singular_mass='error'."
            )
        right_solver = None
        left_solver = None
        shifted_rank = dimension
    else:
        empty_dtype = (
            coordinate_dtype
            if np.issubdtype(coordinate_dtype, np.complexfloating)
            else np.dtype(
                np.complex64 if coordinate_dtype.itemsize <= 4 else np.complex128
            )
        )
        matrix = jnp.zeros((0, 0), dtype=empty_dtype)
        mass = jnp.zeros((0, 0), dtype=empty_dtype)
        denominator = _transform_denominator(problem, plan.policy.transform)
        if denominator is None:
            right_solver = None
            left_solver = None
        else:
            right_problem = LinearSystem(
                denominator,
                problem_id=f"{plan.plan_id}-right-transform",
            )
            left_problem = LinearSystem(
                _CanonicalCoordinateAdjoint(denominator),
                problem_id=f"{plan.plan_id}-left-transform",
            )
            right_solver = (
                prepare_linear_solve(right_problem, plan.policy.transform_solve)
                if transform_solver is None
                else refresh_linear_solve(transform_solver, right_problem)
            )
            left_solver = (
                prepare_linear_solve(left_problem, plan.policy.transform_solve)
                if left_transform_solver is None
                else refresh_linear_solve(left_transform_solver, left_problem)
            )
        mass_rank = dimension
        shifted_rank = -1
    operator_fingerprint = canonical_fingerprint(array_tree_fingerprint(problem.operator))
    identifier = (
        canonical_fingerprint(
            {
                "kind": "prepared-general-eigen",
                "plan": plan.plan_id,
                "problem": problem.problem_id,
            }
        )
        if prepared_id is None
        else prepared_id
    )
    return PreparedGeneralEigenSolve(
        problem=problem,
        matrix=jax.lax.stop_gradient(matrix),
        mass_matrix=jax.lax.stop_gradient(mass),
        transform_solver=right_solver,
        left_transform_solver=left_solver,
        mass_rank=jnp.asarray(mass_rank, dtype=jnp.int32),
        shifted_rank=jnp.asarray(shifted_rank, dtype=jnp.int32),
        plan=plan,
        prepared_id=identifier,
        operator_fingerprint=operator_fingerprint,
        mass_operator_fingerprint=mass_fingerprint,
        numeric_version=jnp.asarray(numeric_version, dtype=jnp.int32),
        refresh_count=jnp.asarray(refresh_count, dtype=jnp.int32),
    )


class _CanonicalCoordinateAdjoint(AbstractLinearOperator):
    """Matrix-free canonical-coordinate conjugate transpose of an endomorphism."""

    operator: AbstractLinearOperator

    def __init__(self, operator: AbstractLinearOperator, /):
        self.source = operator.target
        self.target = operator.source
        self.operator = operator
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=False,
            materialize=False,
        )
        self.batch_shape = operator.batch_shape
        self.operator_id = canonical_fingerprint(
            {
                "kind": "canonical-coordinate-adjoint",
                "operator": operator.operator_id,
            }
        )

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.source.flatten(vector)
        applied = _operator_coordinate_action(
            self.operator,
            coordinates,
            adjoint_action=True,
        )
        target_dtype = np.dtype(_coordinate_dtype(self.target))
        if not np.issubdtype(target_dtype, np.complexfloating):
            applied = jnp.real(applied)
        return self.target.unflatten(applied.astype(target_dtype))

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        coordinates = self.target.flatten(vector)
        applied = jnp.conj(
            _operator_coordinate_action(
                self.operator,
                jnp.conj(coordinates),
                adjoint_action=False,
            )
        )
        source_dtype = np.dtype(_coordinate_dtype(self.source))
        if not np.issubdtype(source_dtype, np.complexfloating):
            applied = jnp.real(applied)
        return self.source.unflatten(applied.astype(source_dtype))

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.operator.mv(vector)

    def _materialize(self, /) -> Array:
        raise ValueError(
            "Canonical coordinate adjoints used by native Arnoldi are matrix-free."
        )


def _numerical_rank(matrix: np.ndarray, relative_tolerance: float, /) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    threshold = relative_tolerance * float(singular_values[0])
    return int(np.count_nonzero(singular_values > threshold))


def _transform_denominator(
    problem: GeneralEigenproblem,
    transform: GeneralEigenTransform,
    /,
) -> AbstractLinearOperator | None:
    mass = (
        IdentityLinearOperator(problem.operator.source)
        if problem.mass_operator is None
        else problem.mass_operator
    )
    if isinstance(transform, StandardTransform):
        return None if problem.mass_operator is None else mass
    coordinate_dtype = np.dtype(_coordinate_dtype(problem.operator.source))
    shift = (
        transform.shift
        if np.issubdtype(coordinate_dtype, np.complexfloating)
        else transform.shift.real
    )
    return problem.operator - shift * mass


def _dense_host_eigensolve(
    matrix: np.ndarray,
    mass: np.ndarray,
    generalized: bool,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool, int]:
    if generalized:
        homogeneous, left, right = scipy_linalg.eig(
            matrix,
            mass,
            left=True,
            right=True,
            homogeneous_eigvals=True,
            check_finite=False,
        )
    else:
        homogeneous, left, right = scipy_linalg.eig(
            matrix,
            left=True,
            right=True,
            homogeneous_eigvals=True,
            check_finite=False,
        )
    alpha = np.asarray(homogeneous[0])
    beta = np.asarray(homogeneous[1])
    return alpha, beta, np.asarray(right), np.asarray(left), alpha.size, True, 0


def _arnoldi_host_eigensolve(
    prepared: PreparedGeneralEigenSolve,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool, int]:
    policy = prepared.plan.policy
    count = policy.selection.count
    if count is None:
        raise ValueError("RestartedArnoldi requires an explicit count.")
    initial = _initial_arnoldi_block(prepared, count)
    right_mu, right_vectors, right_matvecs, right_valid = _native_restarted_arnoldi(
        prepared, initial, adjoint_action=False
    )
    left_mu, left_vectors, left_matvecs, left_valid = _native_restarted_arnoldi(
        prepared,
        jnp.conj(initial),
        adjoint_action=True,
    )
    if prepared.left_transform_solver is not None:
        left_vectors = jax.vmap(
            lambda vector: _solve_complexified_coordinates(
                prepared.left_transform_solver,
                vector,
            ),
            in_axes=1,
            out_axes=1,
        )(left_vectors)
        left_valid = left_valid & jnp.all(jnp.isfinite(left_vectors))
    right_values = _inverse_transformed_values(
        np.asarray(right_mu),
        policy.transform,
    )
    left_values = np.conj(
        _inverse_transformed_values(
            np.asarray(left_mu),
            policy.transform,
            adjoint=True,
        )
    )
    right_vectors_np = np.asarray(right_vectors)
    left_vectors_np = np.asarray(left_vectors)
    paired_count = min(right_values.size, left_values.size)
    pairing = _greedy_eigenvalue_pairing(
        right_values[:paired_count],
        left_values[:paired_count],
    )
    values = right_values[:paired_count]
    right_vectors_np = right_vectors_np[:, :paired_count]
    left_vectors_np = left_vectors_np[:, pairing]
    alpha = np.asarray(values)
    beta = np.ones_like(alpha)
    converged = bool(np.asarray(right_valid) & np.asarray(left_valid))
    return (
        alpha,
        beta,
        right_vectors_np,
        left_vectors_np,
        paired_count,
        converged and paired_count == count,
        int(np.asarray(right_matvecs + left_matvecs)),
    )


def _initial_arnoldi_block(
    prepared: PreparedGeneralEigenSolve,
    count: int,
    /,
) -> Array:
    dimension = prepared.problem.dimension
    dtype = _complex_coordinate_dtype(prepared.problem)
    real_dtype = jnp.float32 if np.dtype(dtype).itemsize <= 8 else jnp.float64
    indices = jnp.arange(1, dimension + 1, dtype=real_dtype)
    columns = []
    for column in range(count):
        frequency = column + 1
        candidate = (
            jnp.cos(frequency * indices)
            + jnp.sin((frequency + math.sqrt(2.0)) * indices)
            + 1j * jnp.sin((frequency + math.sqrt(3.0)) * indices)
        )
        columns.append(candidate.astype(dtype))
    block = jnp.stack(columns, axis=1)
    if prepared.plan.policy.initial_vector is not None:
        supplied = jnp.asarray(
            prepared.plan.policy.initial_vector,
            dtype=dtype,
        )
        block = block.at[:, 0].set(supplied)
    basis, _ = jnp.linalg.qr(block, mode="reduced")
    return basis


def _native_restarted_arnoldi(
    prepared: PreparedGeneralEigenSolve,
    initial: Array,
    /,
    *,
    adjoint_action: bool,
) -> tuple[Array, Array, Array, Array]:
    values, vectors, _, _, matvecs, valid = _native_restarted_arnoldi_evidence(
        prepared,
        initial,
        adjoint_action=adjoint_action,
    )
    return values, vectors, matvecs, valid


def _native_restarted_arnoldi_evidence(
    prepared: PreparedGeneralEigenSolve,
    initial: Array,
    /,
    *,
    adjoint_action: bool,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Fixed-capacity restarted Arnoldi with residual-based hard retention."""
    policy = prepared.plan.policy
    count = initial.shape[1]
    dimension = prepared.problem.dimension
    blocks, cycle_dimension, restart_count = _arnoldi_cycle_configuration(
        policy,
        dimension,
        count,
    )

    def action(block):
        return jax.vmap(
            lambda vector: _transformed_coordinate_action(
                prepared,
                vector,
                adjoint_action=adjoint_action,
            ),
            in_axes=1,
            out_axes=1,
        )(block)

    initial_values = jnp.zeros((count,), dtype=initial.dtype)
    initial_residuals = jnp.full(
        (count,),
        jnp.inf,
        dtype=jnp.real(initial).dtype,
    )
    initial_state = (
        initial,
        initial,
        initial_values,
        initial_residuals,
        jnp.zeros((count,), dtype=bool),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(True),
    )

    def execute_cycle(state):
        (
            restart_basis,
            locked_vectors,
            locked_values,
            locked_residuals,
            locked_mask,
            matvecs,
            valid,
        ) = state
        decomposition = block_arnoldi(
            action,
            restart_basis,
            max_blocks=blocks,
            orthogonalization="double",
            breakdown_tolerance=policy.tolerance.absolute,
        )
        projected = decomposition.projected[:cycle_dimension, :cycle_dimension]
        active = (
            jnp.arange(cycle_dimension) % count
            < decomposition.block_ranks[jnp.arange(cycle_dimension) // count]
        )
        projected = jnp.where(active[:, None] & active[None, :], projected, 0)
        sentinel = jnp.asarray(1e3 + 1e3j, dtype=projected.dtype) * jnp.maximum(
            jnp.linalg.norm(projected),
            1,
        )
        projected = projected + jnp.diag((~active).astype(projected.dtype)) * sentinel
        transformed_values, projected_vectors = jnp.linalg.eig(projected)
        original_values = _jax_inverse_transformed_values(
            transformed_values,
            policy.transform,
            adjoint=adjoint_action,
        )
        active_weight = jnp.sum(
            jnp.where(
                active[:, None],
                jnp.abs(projected_vectors) ** 2,
                0,
            ),
            axis=0,
        )
        scoring_values = jnp.conj(original_values) if adjoint_action else original_values
        scores = _jax_selection_scores(scoring_values, policy.selection)
        scores = jnp.where(active_weight > 0.5, scores, jnp.inf)
        selected = jnp.argsort(scores)[:count]
        selected_vectors = projected_vectors[:, selected]
        ritz_vectors = decomposition.basis[:, :cycle_dimension] @ selected_vectors
        ritz_norms = jnp.linalg.norm(ritz_vectors, axis=0)
        ritz_vectors = (
            ritz_vectors
            / jnp.where(
                ritz_norms > 0,
                ritz_norms,
                1,
            )[None, :]
        )
        selected_transformed = transformed_values[selected]
        selected_original = original_values[selected]
        pencil_vectors = ritz_vectors
        if adjoint_action and prepared.left_transform_solver is not None:
            pencil_vectors = jax.vmap(
                lambda vector: _solve_complexified_coordinates(
                    prepared.left_transform_solver,
                    vector,
                ),
                in_axes=1,
                out_axes=1,
            )(ritz_vectors)
        residuals, scales = _jax_original_pencil_residuals(
            prepared,
            selected_original,
            pencil_vectors,
            adjoint_action=adjoint_action,
        )
        candidate_finite = (
            jnp.isfinite(selected_original)
            & (active_weight[selected] > 0.5)
            & jnp.isfinite(residuals)
            & jnp.isfinite(scales)
        )
        candidate_converged = candidate_finite & (
            residuals <= policy.tolerance.absolute + policy.tolerance.relative * scales
        )
        newly_locked = (~locked_mask) & candidate_converged
        updated_mask = locked_mask | candidate_converged
        updated_vectors = jnp.where(
            newly_locked[None, :],
            ritz_vectors,
            locked_vectors,
        )
        updated_values = jnp.where(
            newly_locked,
            selected_transformed,
            locked_values,
        )
        updated_residuals = jnp.where(
            newly_locked,
            residuals,
            locked_residuals,
        )
        retained_vectors = jnp.where(
            updated_mask[None, :],
            updated_vectors,
            ritz_vectors,
        )
        retained_values = jnp.where(
            updated_mask,
            updated_values,
            selected_transformed,
        )
        retained_residuals = jnp.where(
            updated_mask,
            updated_residuals,
            residuals,
        )
        next_basis, _ = jnp.linalg.qr(retained_vectors, mode="reduced")
        finite_cycle = (
            jnp.all(jnp.isfinite(next_basis))
            & (decomposition.breakdown_status != 3)
            & (decomposition.breakdown_status != 4)
        )
        return (
            next_basis,
            retained_vectors,
            retained_values,
            retained_residuals,
            updated_mask,
            matvecs + decomposition.matvec_count,
            valid & finite_cycle,
        )

    def restart_step(_, state):
        return jax.lax.cond(
            jnp.all(state[4]),
            lambda current: current,
            execute_cycle,
            state,
        )

    (
        _,
        ritz_vectors,
        transformed_values,
        residuals,
        converged_mask,
        matvecs,
        valid,
    ) = jax.lax.fori_loop(
        0,
        restart_count,
        restart_step,
        initial_state,
    )
    return (
        transformed_values,
        ritz_vectors,
        residuals,
        converged_mask,
        matvecs,
        valid,
    )


def _jax_original_pencil_residuals(
    prepared: PreparedGeneralEigenSolve,
    values: Array,
    vectors: Array,
    /,
    *,
    adjoint_action: bool,
) -> tuple[Array, Array]:
    problem = prepared.problem

    def residual(value, vector):
        matrix_action = _operator_coordinate_action(
            problem.operator,
            vector,
            adjoint_action=adjoint_action,
        )
        mass_action = (
            vector
            if problem.mass_operator is None
            else _operator_coordinate_action(
                problem.mass_operator,
                vector,
                adjoint_action=adjoint_action,
            )
        )
        residual_vector = matrix_action - value * mass_action
        scale = jnp.linalg.norm(matrix_action) + jnp.abs(value) * jnp.linalg.norm(
            mass_action
        )
        return jnp.linalg.norm(residual_vector), scale

    return jax.vmap(residual, in_axes=(0, 1), out_axes=(0, 0))(values, vectors)


def _transformed_coordinate_action(
    prepared: PreparedGeneralEigenSolve,
    vector: Array,
    /,
    *,
    adjoint_action: bool,
) -> Array:
    problem = prepared.problem
    transform = prepared.plan.policy.transform
    solver = (
        prepared.left_transform_solver if adjoint_action else prepared.transform_solver
    )
    operand = (
        _solve_complexified_coordinates(solver, vector)
        if adjoint_action and solver is not None
        else vector
    )
    if isinstance(transform, StandardTransform):
        numerator = _operator_coordinate_action(
            problem.operator,
            operand,
            adjoint_action=adjoint_action,
        )
    elif isinstance(transform, ShiftInvertTransform):
        numerator = (
            operand
            if problem.mass_operator is None
            else _operator_coordinate_action(
                problem.mass_operator,
                operand,
                adjoint_action=adjoint_action,
            )
        )
    else:
        matrix_action = _operator_coordinate_action(
            problem.operator,
            operand,
            adjoint_action=adjoint_action,
        )
        mass_action = (
            operand
            if problem.mass_operator is None
            else _operator_coordinate_action(
                problem.mass_operator,
                operand,
                adjoint_action=adjoint_action,
            )
        )
        shift = jnp.asarray(
            np.conj(transform.shift) if adjoint_action else transform.shift,
            dtype=vector.dtype,
        )
        numerator = matrix_action + shift * mass_action
    if adjoint_action or solver is None:
        return numerator
    return _solve_complexified_coordinates(solver, numerator)


def _operator_coordinate_action(
    operator: AbstractLinearOperator,
    vector: Array,
    /,
    *,
    adjoint_action: bool,
) -> Array:
    input_space = operator.target if adjoint_action else operator.source
    output_space = operator.source if adjoint_action else operator.target
    dtype = np.dtype(_coordinate_dtype(input_space))

    def apply_component(component):
        tree = input_space.unflatten(component.astype(dtype))
        action = operator.transpose_mv if adjoint_action else operator.mv
        return output_space.flatten(action(tree))

    if adjoint_action and np.issubdtype(dtype, np.complexfloating):
        return jnp.conj(apply_component(jnp.conj(vector)))
    if np.issubdtype(dtype, np.complexfloating):
        return apply_component(vector)
    return apply_component(jnp.real(vector)) + 1j * apply_component(jnp.imag(vector))


def _solve_complexified_coordinates(
    prepared: PreparedLinearSolve,
    right_hand_side: Array,
    /,
) -> Array:
    space = prepared.problem.operator.target
    dtype = np.dtype(_coordinate_dtype(space))

    def solve_component(component):
        tree = space.unflatten(component.astype(dtype))
        result = linear_solve(prepared, tree)
        coordinates = prepared.problem.operator.source.flatten(result.value)
        return jnp.where(result.successful, coordinates, jnp.nan)

    if np.issubdtype(dtype, np.complexfloating):
        return solve_component(right_hand_side)
    return solve_component(jnp.real(right_hand_side)) + 1j * solve_component(
        jnp.imag(right_hand_side)
    )


def _jax_inverse_transformed_values(
    values: Array,
    transform: GeneralEigenTransform,
    /,
    *,
    adjoint: bool,
) -> Array:
    if isinstance(transform, StandardTransform):
        return values
    shift = jnp.asarray(
        np.conj(transform.shift) if adjoint else transform.shift,
        dtype=values.dtype,
    )
    if isinstance(transform, ShiftInvertTransform):
        return shift + 1.0 / values
    return shift * (values + 1.0) / (values - 1.0)


def _jax_selection_scores(
    values: Array,
    selection: GeneralEigenSelection,
    /,
) -> Array:
    if selection.kind == "closest":
        return jnp.abs(values - selection.target)
    if selection.kind == "largest-magnitude":
        return -jnp.abs(values)
    if selection.kind == "smallest-magnitude":
        return jnp.abs(values)
    if selection.kind == "largest-real":
        return -jnp.real(values)
    if selection.kind == "smallest-real":
        return jnp.real(values)
    if selection.kind == "largest-imaginary":
        return -jnp.imag(values)
    return jnp.imag(values)


def _greedy_eigenvalue_pairing(
    right_values: np.ndarray,
    left_values: np.ndarray,
    /,
) -> np.ndarray:
    remaining = list(range(left_values.size))
    pairing: list[int] = []
    for value in right_values:
        selected = min(
            remaining,
            key=lambda index: (abs(value - left_values[index]), index),
        )
        pairing.append(selected)
        remaining.remove(selected)
    return np.asarray(pairing, dtype=np.int64)


def _inverse_transformed_values(
    values: np.ndarray,
    transform: GeneralEigenTransform,
    /,
    *,
    adjoint: bool = False,
) -> np.ndarray:
    if isinstance(transform, StandardTransform):
        return values
    shift = np.conj(transform.shift) if adjoint else transform.shift
    if isinstance(transform, ShiftInvertTransform):
        return shift + 1.0 / values
    return shift * (values + 1.0) / (values - 1.0)


def _nonfinite_host_result(
    dimension: int,
    requested: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool, int]:
    dtype = np.dtype(np.complex128)
    alpha = np.full((requested,), np.nan + 0j, dtype=dtype)
    beta = np.ones((requested,), dtype=dtype)
    vectors = np.full((dimension, requested), np.nan + 0j, dtype=dtype)
    return alpha, beta, vectors, vectors.copy(), requested, False, 0


def _classify_homogeneous(
    alpha: np.ndarray,
    beta: np.ndarray,
    tolerance: GeneralEigenTolerancePolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_magnitude = np.abs(alpha)
    beta_magnitude = np.abs(beta)
    scale = np.hypot(alpha_magnitude, beta_magnitude)
    indeterminate = scale == 0.0
    beta_small = beta_magnitude <= tolerance.homogeneous_relative * scale
    infinite = beta_small & ~indeterminate
    finite = ~beta_small & ~indeterminate
    return finite, infinite, indeterminate


def _ordinary_eigenvalues(
    alpha: np.ndarray,
    beta: np.ndarray,
    finite: np.ndarray,
    infinite: np.ndarray,
    /,
) -> np.ndarray:
    dtype = np.result_type(alpha.dtype, beta.dtype, np.complex64)
    values = np.full(alpha.shape, np.nan + 0j, dtype=dtype)
    values[finite] = alpha[finite] / beta[finite]
    values[infinite] = np.inf + 0j
    return values


def _selection_indices(
    alpha: np.ndarray,
    beta: np.ndarray,
    finite: np.ndarray,
    infinite: np.ndarray,
    selection: GeneralEigenSelection,
    /,
) -> np.ndarray:
    count = selection.count
    if selection.kind == "all":
        candidates = np.arange(alpha.size)
    elif selection.kind == "finite":
        candidates = np.flatnonzero(finite)
    elif selection.kind == "infinite":
        candidates = np.flatnonzero(infinite)
    else:
        candidates = np.flatnonzero(finite)
        values = alpha[candidates] / beta[candidates]
        if selection.kind == "closest":
            key = np.abs(values - selection.target)
        elif selection.kind == "largest-magnitude":
            key = -np.abs(values)
        elif selection.kind == "smallest-magnitude":
            key = np.abs(values)
        elif selection.kind == "largest-real":
            key = -np.real(values)
        elif selection.kind == "smallest-real":
            key = np.real(values)
        elif selection.kind == "largest-imaginary":
            key = -np.imag(values)
        else:
            key = np.imag(values)
        candidates = candidates[np.argsort(key, kind="stable")]
    if count is not None:
        candidates = candidates[:count]
    return np.asarray(candidates, dtype=np.int64)


def _cluster_groups(
    alpha: np.ndarray,
    beta: np.ndarray,
    finite: np.ndarray,
    infinite: np.ndarray,
    tolerance: GeneralEigenTolerancePolicy,
    /,
) -> tuple[np.ndarray, ...]:
    values = _ordinary_eigenvalues(alpha, beta, finite, infinite)
    remaining = list(range(alpha.size))
    groups: list[np.ndarray] = []
    while remaining:
        first = remaining.pop(0)
        group = [first]
        retained: list[int] = []
        for index in remaining:
            same_class = bool(
                finite[first] == finite[index] and infinite[first] == infinite[index]
            )
            if finite[first] and same_class:
                scale = max(abs(values[first]), abs(values[index]), 1.0)
                same = (
                    abs(values[first] - values[index])
                    <= tolerance.cluster_relative * scale
                )
            elif infinite[first] and same_class:
                same = True
            else:
                same = False
            if same:
                group.append(index)
            else:
                retained.append(index)
        remaining = retained
        groups.append(np.asarray(group, dtype=np.int64))
    return tuple(groups)


def _normalize_paired_vectors(
    prepared: PreparedGeneralEigenSolve,
    alpha: np.ndarray,
    beta: np.ndarray,
    right: np.ndarray,
    left: np.ndarray,
    finite: np.ndarray,
    infinite: np.ndarray,
    tolerance: GeneralEigenTolerancePolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if alpha.size == 0:
        empty = np.zeros((0, 0), dtype=alpha.dtype)
        return right, left, empty, np.zeros((0,))
    right = np.asarray(
        right,
        dtype=np.result_type(right.dtype, left.dtype, np.complex64),
    ).copy()
    left = np.asarray(left, dtype=right.dtype).copy()
    right_norms = np.linalg.norm(right, axis=0)
    left_norms = np.linalg.norm(left, axis=0)
    right /= np.where(right_norms > 0.0, right_norms, 1.0)[None, :]
    left /= np.where(left_norms > 0.0, left_norms, 1.0)[None, :]
    matrix_right, mass_right, _, _ = _pencil_vector_actions(
        prepared,
        right,
        left,
    )
    metric_columns = np.where(finite[None, :], mass_right, matrix_right)
    for group in _cluster_groups(alpha, beta, finite, infinite, tolerance):
        overlap = np.conj(left[:, group].T) @ metric_columns[:, group]
        left[:, group] = left[:, group] @ np.conj(np.linalg.pinv(overlap).T)
    pairing = np.conj(left.T) @ metric_columns
    diagonal = np.diag(pairing)
    condition = np.empty((alpha.size,), dtype=np.float64)
    for index in range(alpha.size):
        denominator = abs(diagonal[index])
        condition[index] = (
            np.linalg.norm(left[:, index]) * np.linalg.norm(right[:, index]) / denominator
            if denominator > 0.0
            else np.inf
        )
    return right, left, pairing, condition


def _homogeneous_residuals(
    prepared: PreparedGeneralEigenSolve,
    alpha: np.ndarray,
    beta: np.ndarray,
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
]:
    matrix_right, mass_right, matrix_left, mass_left = _pencil_vector_actions(
        prepared,
        right,
        left,
    )
    right_residual = beta[None, :] * matrix_right - alpha[None, :] * mass_right
    left_residual = (
        np.conj(beta)[None, :] * matrix_left - np.conj(alpha)[None, :] * mass_left
    )
    right_norms = np.linalg.norm(right_residual, axis=0)
    left_norms = np.linalg.norm(left_residual, axis=0)
    right_scale = np.abs(beta) * np.linalg.norm(matrix_right, axis=0) + np.abs(
        alpha
    ) * np.linalg.norm(mass_right, axis=0)
    left_scale = np.abs(beta) * np.linalg.norm(matrix_left, axis=0) + np.abs(
        alpha
    ) * np.linalg.norm(mass_left, axis=0)
    real_dtype = np.asarray(right_norms).dtype
    tiny = np.finfo(real_dtype).tiny
    return (
        right_norms,
        left_norms,
        right_norms / np.maximum(right_scale, tiny),
        left_norms / np.maximum(left_scale, tiny),
        right_scale,
        left_scale,
    )


def _pencil_vector_actions(
    prepared: PreparedGeneralEigenSolve,
    right: np.ndarray,
    left: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if right.shape[1] == 0:
        empty = np.zeros_like(right)
        return empty, empty, empty, empty
    if isinstance(prepared.plan.policy.method, DenseSchurQZ):
        matrix = np.asarray(prepared.matrix)
        mass = np.asarray(prepared.mass_matrix)
        return (
            matrix @ right,
            mass @ right,
            np.conj(matrix.T) @ left,
            np.conj(mass.T) @ left,
        )
    right_array = jnp.asarray(right)
    left_array = jnp.asarray(left)
    matrix_right = jax.vmap(
        lambda vector: _operator_coordinate_action(
            prepared.problem.operator,
            vector,
            adjoint_action=False,
        ),
        in_axes=1,
        out_axes=1,
    )(right_array)
    matrix_left = jax.vmap(
        lambda vector: _operator_coordinate_action(
            prepared.problem.operator,
            vector,
            adjoint_action=True,
        ),
        in_axes=1,
        out_axes=1,
    )(left_array)
    if prepared.problem.mass_operator is None:
        mass_right = right_array
        mass_left = left_array
    else:
        mass_right = jax.vmap(
            lambda vector: _operator_coordinate_action(
                prepared.problem.mass_operator,
                vector,
                adjoint_action=False,
            ),
            in_axes=1,
            out_axes=1,
        )(right_array)
        mass_left = jax.vmap(
            lambda vector: _operator_coordinate_action(
                prepared.problem.mass_operator,
                vector,
                adjoint_action=True,
            ),
            in_axes=1,
            out_axes=1,
        )(left_array)
    return tuple(
        np.asarray(value) for value in (matrix_right, mass_right, matrix_left, mass_left)
    )


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
    return jax.tree.map(
        lambda real_leaf, imag_leaf: real_leaf + 1j * imag_leaf, real, imaginary
    )


__all__ = [
    "CayleyTransform",
    "DenseSchurQZ",
    "GeneralEigenCapabilities",
    "GeneralEigenCostEstimate",
    "GeneralEigenMethod",
    "GeneralEigenproblem",
    "GeneralEigenproblemKind",
    "GeneralEigenResourcePolicy",
    "GeneralEigenSelection",
    "GeneralEigenSelectionKind",
    "GeneralEigenSolveDiagnostics",
    "GeneralEigenSolvePlan",
    "GeneralEigenSolvePolicy",
    "GeneralEigenSolveProvenance",
    "GeneralEigenSolveResult",
    "GeneralEigenSolveStatus",
    "GeneralEigenTolerancePolicy",
    "GeneralEigenTransform",
    "PreparedGeneralEigenSolve",
    "RestartedArnoldi",
    "ShiftInvertTransform",
    "SingularMassPolicy",
    "StandardTransform",
    "general_eigensolve",
    "plan_general_eigensolve",
    "prepare_general_eigensolve",
    "refresh_general_eigensolve",
]
