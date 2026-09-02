#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from numbers import Integral
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import HermitianPrecisionPolicy, HermitianSpectrum
from ...linalg.eigen import (
    HermitianEigenspaceTrackingPlan,
    HermitianEigenspaceTrackingPolicy,
    plan_hermitian_eigenspace_tracking,
    track_hermitian_eigenspaces,
)


class NamedModeOperator(StrictModule):
    """Named finite operator carried through one mode-basis reduction."""

    matrix: Array
    name: str = eqx.field(static=True)
    hermitian: bool = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        matrix: ArrayLike,
        /,
        *,
        hermitian: bool = False,
        operator_id: str | None = None,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("name must be nonempty.")
        value = jnp.asarray(matrix)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("matrix must be one square matrix.")
        self.matrix = value
        self.name = name_
        self.hermitian = bool(hermitian)
        self.operator_id = (
            canonical_fingerprint(
                {
                    "kind": "named-mode-operator",
                    "name": name_,
                    "hermitian": bool(hermitian),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not self.operator_id:
            raise ValueError("operator_id must be nonempty.")


class ModeReductionProblem(StrictModule):
    """Finite raw Hamiltonian and operators to reduce into low-energy coordinates."""

    hamiltonian: Array
    operators: tuple[NamedModeOperator, ...]
    hbar: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: ArrayLike,
        operators: tuple[NamedModeOperator, ...] | list[NamedModeOperator] = (),
        /,
        *,
        hbar: ArrayLike = 1.0,
        problem_id: str = "quantum-mode-reduction",
    ):
        matrix = jnp.asarray(hamiltonian)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("hamiltonian must be one nonempty square matrix.")
        operators_ = tuple(operators)
        if not all(isinstance(operator, NamedModeOperator) for operator in operators_):
            raise TypeError("operators must contain NamedModeOperator values.")
        names = tuple(operator.name for operator in operators_)
        if len(set(names)) != len(names):
            raise ValueError("operator names must be unique.")
        if any(operator.matrix.shape != matrix.shape for operator in operators_):
            raise ValueError("Every mode operator must match the Hamiltonian shape.")
        dtype = jnp.result_type(
            matrix, *(operator.matrix for operator in operators_), complex
        )
        matrix = matrix.astype(dtype)
        operators_ = tuple(
            NamedModeOperator(
                operator.name,
                operator.matrix.astype(dtype),
                hermitian=operator.hermitian,
                operator_id=operator.operator_id,
            )
            for operator in operators_
        )
        hbar_ = jnp.asarray(hbar, dtype=jnp.real(matrix).dtype)
        if hbar_.shape != ():
            raise ValueError("hbar must be scalar.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.hamiltonian = matrix
        self.operators = operators_
        self.hbar = hbar_
        self.problem_id = identifier

    def operator(self, name: str, /) -> NamedModeOperator:
        for operator in self.operators:
            if operator.name == name:
                return operator
        raise KeyError(f"Unknown mode operator {name!r}.")


class ModeReductionPolicy(StrictModule):
    """Resource and numerical validity contract for low-energy mode reduction."""

    precision: HermitianPrecisionPolicy
    tracking: HermitianEigenspaceTrackingPolicy
    retained_dimension: int = eqx.field(static=True)
    maximum_raw_dimension: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)
    eigen_residual_tolerance: float = eqx.field(static=True)
    orthogonality_tolerance: float = eqx.field(static=True)
    minimum_boundary_gap: float = eqx.field(static=True)

    def __init__(
        self,
        retained_dimension: int,
        /,
        *,
        maximum_raw_dimension: int = 4096,
        maximum_bytes: int = 1 << 30,
        hermiticity_tolerance: float = 1e-10,
        eigen_residual_tolerance: float = 1e-8,
        orthogonality_tolerance: float = 1e-8,
        minimum_boundary_gap: float = 0.0,
        precision: HermitianPrecisionPolicy | None = None,
        tracking: HermitianEigenspaceTrackingPolicy | None = None,
    ):
        for name, value in (
            ("retained_dimension", retained_dimension),
            ("maximum_raw_dimension", maximum_raw_dimension),
            ("maximum_bytes", maximum_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a positive integer.")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive.")
        tolerances = (
            hermiticity_tolerance,
            eigen_residual_tolerance,
            orthogonality_tolerance,
            minimum_boundary_gap,
        )
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in tolerances):
            raise ValueError("Mode-reduction tolerances must be finite and non-negative.")
        precision_ = HermitianPrecisionPolicy() if precision is None else precision
        tracking_ = HermitianEigenspaceTrackingPolicy() if tracking is None else tracking
        if not isinstance(precision_, HermitianPrecisionPolicy):
            raise TypeError("precision must be a HermitianPrecisionPolicy or None.")
        if not isinstance(tracking_, HermitianEigenspaceTrackingPolicy):
            raise TypeError(
                "tracking must be a HermitianEigenspaceTrackingPolicy or None."
            )
        self.precision = precision_
        self.tracking = tracking_
        self.retained_dimension = int(retained_dimension)
        self.maximum_raw_dimension = int(maximum_raw_dimension)
        self.maximum_bytes = int(maximum_bytes)
        self.hermiticity_tolerance = float(hermiticity_tolerance)
        self.eigen_residual_tolerance = float(eigen_residual_tolerance)
        self.orthogonality_tolerance = float(orthogonality_tolerance)
        self.minimum_boundary_gap = float(minimum_boundary_gap)


class ModeReductionCostEstimate(StrictModule):
    """Dense storage estimate for one mode reduction."""

    raw_dimension: int = eqx.field(static=True)
    retained_dimension: int = eqx.field(static=True)
    operator_count: int = eqx.field(static=True)
    persistent_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class ModeReductionPlan(StrictModule):
    """Content-addressed static plan for one mode-reduction structure."""

    policy: ModeReductionPolicy
    cost: ModeReductionCostEstimate
    operator_names: tuple[str, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ModeReductionDiagnostics(StrictModule):
    """Spectral, projection, and continuation evidence."""

    hermiticity_residual: Array
    eigen_residual: Array
    orthogonality_residual: Array
    minimum_internal_gap: Array
    boundary_gap: Array
    projected_hermiticity_residuals: Array
    tracking_valid: Array
    finite: Array
    valid: Array


class PreparedModeReduction(StrictModule):
    """Reduced mode coordinates, projected operators, and refresh state."""

    plan: ModeReductionPlan
    energies: Array
    isometry: Array
    operators: tuple[NamedModeOperator, ...]
    raw_eigenvalues: Array
    diagnostics: ModeReductionDiagnostics
    tracking_plan: HermitianEigenspaceTrackingPlan
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)

    def operator(self, name: str, /) -> NamedModeOperator:
        for operator in self.operators:
            if operator.name == name:
                return operator
        raise KeyError(f"Unknown reduced mode operator {name!r}.")


class ModeResolutionPolicy(StrictModule):
    """Acceptance thresholds for two independently prepared mode resolutions."""

    energy_absolute: float = eqx.field(static=True)
    energy_relative: float = eqx.field(static=True)
    operator_absolute: float = eqx.field(static=True)
    minimum_subspace_overlap: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        energy_absolute: float = 1e-8,
        energy_relative: float = 1e-6,
        operator_absolute: float = 1e-6,
        minimum_subspace_overlap: float = 1.0 - 1e-6,
    ):
        values = (
            energy_absolute,
            energy_relative,
            operator_absolute,
            minimum_subspace_overlap,
        )
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("Resolution thresholds must be finite and non-negative.")
        if float(minimum_subspace_overlap) > 1.0:
            raise ValueError("minimum_subspace_overlap must not exceed one.")
        self.energy_absolute = float(energy_absolute)
        self.energy_relative = float(energy_relative)
        self.operator_absolute = float(operator_absolute)
        self.minimum_subspace_overlap = float(minimum_subspace_overlap)


class ModeResolutionReport(StrictModule):
    """Cross-cutoff energy, operator, and optional subspace evidence."""

    energy_absolute_error: Array
    energy_relative_error: Array
    operator_absolute_error: Array
    minimum_subspace_overlap: Array
    subspace_overlap_available: Array
    finite: Array
    valid: Array


def plan_mode_reduction(
    problem: ModeReductionProblem,
    policy: ModeReductionPolicy,
    /,
) -> ModeReductionPlan:
    """Validate one raw mode structure and estimate its dense costs."""

    if not isinstance(problem, ModeReductionProblem):
        raise TypeError("problem must be a ModeReductionProblem.")
    if not isinstance(policy, ModeReductionPolicy):
        raise TypeError("policy must be a ModeReductionPolicy.")
    dimension = int(problem.hamiltonian.shape[0])
    retained = policy.retained_dimension
    if retained > dimension:
        raise ValueError("retained_dimension must not exceed the raw dimension.")
    if dimension > policy.maximum_raw_dimension:
        raise ValueError("Raw mode dimension exceeds maximum_raw_dimension.")
    itemsize = np.dtype(problem.hamiltonian.dtype).itemsize
    persistent = itemsize * (
        dimension * dimension
        + dimension
        + dimension * retained
        + len(problem.operators) * retained * retained
    )
    workspace = itemsize * (2 * dimension * dimension + dimension * retained)
    if persistent + workspace > policy.maximum_bytes:
        raise ValueError("Mode-reduction storage exceeds maximum_bytes.")
    names = tuple(operator.name for operator in problem.operators)
    cost = ModeReductionCostEstimate(
        dimension,
        retained,
        len(problem.operators),
        persistent,
        workspace,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "mode-reduction-plan",
            "problem_id": problem.problem_id,
            "dimension": dimension,
            "retained": retained,
            "operator_names": list(names),
            "operator_hermitian": [operator.hermitian for operator in problem.operators],
            "dtype": str(problem.hamiltonian.dtype),
            "policy": {
                "maximum_raw_dimension": policy.maximum_raw_dimension,
                "maximum_bytes": policy.maximum_bytes,
                "hermiticity_tolerance": policy.hermiticity_tolerance,
                "eigen_residual_tolerance": policy.eigen_residual_tolerance,
                "orthogonality_tolerance": policy.orthogonality_tolerance,
                "minimum_boundary_gap": policy.minimum_boundary_gap,
                "precision": policy.precision.policy_id,
                "tracking": {
                    "degeneracy_absolute": policy.tracking.degeneracy_absolute,
                    "degeneracy_relative": policy.tracking.degeneracy_relative,
                    "minimum_overlap": policy.tracking.minimum_overlap,
                    "minimum_assignment_margin": (
                        policy.tracking.minimum_assignment_margin
                    ),
                    "orthogonality_tolerance": (policy.tracking.orthogonality_tolerance),
                    "maximum_dimension": policy.tracking.maximum_dimension,
                },
            },
        }
    )
    return ModeReductionPlan(policy, cost, names, problem.problem_id, plan_id)


def _validate_problem_plan(
    problem: ModeReductionProblem, plan: ModeReductionPlan, /
) -> None:
    if not isinstance(problem, ModeReductionProblem):
        raise TypeError("problem must be a ModeReductionProblem.")
    if not isinstance(plan, ModeReductionPlan):
        raise TypeError("plan must be a ModeReductionPlan.")
    candidate = plan_mode_reduction(problem, plan.policy)
    if candidate.plan_id != plan.plan_id:
        raise ValueError("ModeReductionProblem does not match the plan structure.")


def _canonicalize_columns(vectors: Array, /) -> Array:
    anchor_rows = jnp.argmax(jnp.abs(vectors), axis=0)
    anchors = vectors[anchor_rows, jnp.arange(vectors.shape[1])]
    magnitude = jnp.abs(anchors)
    phase = jnp.where(
        magnitude > jnp.finfo(magnitude.dtype).tiny,
        jnp.conj(anchors) / magnitude,
        jnp.ones_like(anchors),
    )
    return vectors * phase[None, :]


def _prepare_mode_reduction(
    problem: ModeReductionProblem,
    plan: ModeReductionPlan,
    /,
    *,
    previous: PreparedModeReduction | None,
    numeric_version: Any,
) -> PreparedModeReduction:
    spectrum = HermitianSpectrum(
        problem.hamiltonian,
        tolerance=plan.policy.hermiticity_tolerance,
        precision=plan.policy.precision,
    )
    retained = plan.policy.retained_dimension
    if previous is None:
        energies = spectrum.eigenvalues[:retained]
        isometry = _canonicalize_columns(spectrum.eigenvectors[:, :retained])
        tracking_valid = jnp.asarray(True)
    else:
        tracking = track_hermitian_eigenspaces(
            previous.tracking_plan,
            previous.isometry,
            spectrum.eigenvalues,
            spectrum.eigenvectors,
        )
        energies = tracking.values
        isometry = tracking.vectors
        tracking_valid = tracking.successful

    reduced_operators = tuple(
        NamedModeOperator(
            operator.name,
            oe.contract(
                "ai,ab,bj->ij",
                jnp.conj(isometry),
                operator.matrix,
                isometry,
            ),
            hermitian=operator.hermitian,
            operator_id=canonical_fingerprint(
                {
                    "kind": "reduced-mode-operator",
                    "plan": plan.plan_id,
                    "source": operator.operator_id,
                }
            ),
        )
        for operator in problem.operators
    )
    residual = problem.hamiltonian @ isometry - isometry * energies[None, :]
    eigen_residual = jnp.max(jnp.abs(residual))
    gram = oe.contract("ai,aj->ij", jnp.conj(isometry), isometry)
    orthogonality = jnp.max(jnp.abs(gram - jnp.eye(retained, dtype=gram.dtype)))
    internal_gaps = jnp.abs(energies[1:] - energies[:-1])
    minimum_internal_gap = (
        jnp.min(internal_gaps)
        if retained > 1
        else jnp.asarray(jnp.inf, dtype=energies.dtype)
    )
    boundary_gap = (
        spectrum.eigenvalues[retained] - spectrum.eigenvalues[retained - 1]
        if retained < plan.cost.raw_dimension
        else jnp.asarray(jnp.inf, dtype=energies.dtype)
    )
    projected_residuals = jnp.stack(
        tuple(
            jnp.max(jnp.abs(operator.matrix - jnp.conj(operator.matrix.T)))
            if operator.hermitian
            else jnp.asarray(0.0, dtype=energies.dtype)
            for operator in reduced_operators
        )
        or (jnp.asarray(0.0, dtype=energies.dtype),)
    )
    finite = (
        jnp.all(jnp.isfinite(problem.hamiltonian))
        & jnp.isfinite(problem.hbar)
        & (problem.hbar > 0)
        & jnp.all(jnp.isfinite(energies))
        & jnp.all(jnp.isfinite(isometry))
        & jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(operator.matrix))
                    for operator in problem.operators
                )
                or (jnp.asarray(True),)
            )
        )
    )
    valid = (
        spectrum.valid
        & finite
        & tracking_valid
        & (eigen_residual <= plan.policy.eigen_residual_tolerance)
        & (orthogonality <= plan.policy.orthogonality_tolerance)
        & (boundary_gap >= plan.policy.minimum_boundary_gap)
        & jnp.all(
            projected_residuals
            <= jnp.asarray(plan.policy.hermiticity_tolerance, dtype=energies.dtype)
        )
    )
    diagnostics = ModeReductionDiagnostics(
        spectrum.hermiticity_residual,
        eigen_residual,
        orthogonality,
        minimum_internal_gap,
        boundary_gap,
        projected_residuals,
        tracking_valid,
        finite,
        valid,
    )
    tracking_plan = (
        plan_hermitian_eigenspace_tracking(
            np.asarray(energies),
            policy=plan.policy.tracking,
        )
        if previous is None
        else previous.tracking_plan
    )
    version = jnp.asarray(numeric_version, dtype=jnp.int32)
    if version.shape != ():
        raise ValueError("numeric_version must be scalar.")
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-mode-reduction", "plan": plan.plan_id}
    )
    return PreparedModeReduction(
        plan,
        energies,
        isometry,
        reduced_operators,
        spectrum.eigenvalues,
        diagnostics,
        tracking_plan,
        version,
        prepared_id,
    )


def prepare_mode_reduction(
    problem: ModeReductionProblem,
    plan: ModeReductionPlan | None = None,
    /,
    *,
    policy: ModeReductionPolicy | None = None,
) -> PreparedModeReduction:
    """Prepare a low-energy basis and projected operators."""

    if plan is None:
        if policy is None:
            raise ValueError("policy is required when plan is omitted.")
        selected = plan_mode_reduction(problem, policy)
    else:
        if policy is not None:
            raise ValueError("Specify plan or policy, not both.")
        selected = plan
        _validate_problem_plan(problem, selected)
    return _prepare_mode_reduction(
        problem,
        selected,
        previous=None,
        numeric_version=0,
    )


def refresh_mode_reduction(
    prepared: PreparedModeReduction,
    problem: ModeReductionProblem,
    /,
) -> PreparedModeReduction:
    """Refresh numerical mode data while preserving static structure and labels."""

    if not isinstance(prepared, PreparedModeReduction):
        raise TypeError("prepared must be a PreparedModeReduction.")
    _validate_problem_plan(problem, prepared.plan)
    return _prepare_mode_reduction(
        problem,
        prepared.plan,
        previous=prepared,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def compare_mode_resolutions(
    coarse: PreparedModeReduction,
    fine: PreparedModeReduction,
    /,
    *,
    policy: ModeResolutionPolicy | None = None,
    coarse_to_fine: ArrayLike | None = None,
) -> ModeResolutionReport:
    """Compare independently prepared mode cutoffs without assuming convergence."""

    if not isinstance(coarse, PreparedModeReduction) or not isinstance(
        fine, PreparedModeReduction
    ):
        raise TypeError("coarse and fine must be PreparedModeReduction values.")
    if coarse.energies.shape != fine.energies.shape:
        raise ValueError("Mode resolutions must retain the same dimension.")
    if coarse.plan.operator_names != fine.plan.operator_names:
        raise ValueError("Mode resolutions must contain the same named operators.")
    selected = ModeResolutionPolicy() if policy is None else policy
    if not isinstance(selected, ModeResolutionPolicy):
        raise TypeError("policy must be a ModeResolutionPolicy or None.")
    energy_difference = jnp.abs(coarse.energies - fine.energies)
    energy_absolute = jnp.max(energy_difference)
    energy_scale = jnp.maximum(jnp.max(jnp.abs(fine.energies)), 1.0)
    energy_relative = energy_absolute / energy_scale
    operator_error = jnp.max(
        jnp.stack(
            tuple(
                jnp.max(jnp.abs(jnp.abs(left.matrix) - jnp.abs(right.matrix)))
                for left, right in zip(coarse.operators, fine.operators, strict=True)
            )
            or (jnp.asarray(0.0, dtype=energy_absolute.dtype),)
        )
    )
    if coarse_to_fine is None:
        overlap = jnp.asarray(jnp.nan, dtype=energy_absolute.dtype)
        overlap_available = jnp.asarray(False)
        overlap_valid = jnp.asarray(True)
    else:
        embedding = jnp.asarray(coarse_to_fine)
        expected = (fine.isometry.shape[0], coarse.isometry.shape[0])
        if embedding.shape != expected:
            raise ValueError(f"coarse_to_fine must have shape {expected}.")
        embedded = embedding @ coarse.isometry
        cross = oe.contract("ai,aj->ij", jnp.conj(embedded), fine.isometry)
        singular_values = jnp.linalg.svd(cross, compute_uv=False)
        overlap = jnp.min(singular_values) ** 2
        overlap_available = jnp.asarray(True)
        overlap_valid = overlap >= selected.minimum_subspace_overlap
    finite = (
        jnp.isfinite(energy_absolute)
        & jnp.isfinite(energy_relative)
        & jnp.isfinite(operator_error)
        & (~overlap_available | jnp.isfinite(overlap))
    )
    valid = (
        finite
        & coarse.diagnostics.valid
        & fine.diagnostics.valid
        & (energy_absolute <= selected.energy_absolute)
        & (energy_relative <= selected.energy_relative)
        & (operator_error <= selected.operator_absolute)
        & overlap_valid
    )
    return ModeResolutionReport(
        energy_absolute,
        energy_relative,
        operator_error,
        overlap,
        overlap_available,
        finite,
        valid,
    )


__all__ = [
    "ModeReductionCostEstimate",
    "ModeReductionDiagnostics",
    "ModeReductionPlan",
    "ModeReductionPolicy",
    "ModeReductionProblem",
    "ModeResolutionPolicy",
    "ModeResolutionReport",
    "NamedModeOperator",
    "PreparedModeReduction",
    "compare_mode_resolutions",
    "plan_mode_reduction",
    "prepare_mode_reduction",
    "refresh_mode_reduction",
]
