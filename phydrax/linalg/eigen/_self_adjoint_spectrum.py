#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from .._materialization import MaterializationPolicy
from .._policies import FailurePolicy
from .._spaces import _coordinate_dtype
from ._plans import EigenCostEstimate, EigenSolvePlan
from ._policies import (
    DenseEigh,
    EigenResourcePolicy,
    EigenSolvePolicy,
    EigenTolerancePolicy,
)
from ._prepared import PreparedEigenSolve
from ._problems import Eigenproblem, EigenproblemLike, GeneralizedEigenproblem
from ._results import EigenSolveDiagnostics, EigenSolveStatus
from ._runtime import eigensolve, plan_eigensolve, prepare_eigensolve, refresh_eigensolve


class SelfAdjointSpectrumStatus(IntEnum):
    """Status of one reusable full self-adjoint spectrum."""

    SUCCESS = 0
    SOURCE_FAILURE = 1
    NONFINITE = 2
    NORMALIZATION_RESIDUAL_TOO_LARGE = 3


class SelfAdjointSpectrumPolicy(StrictModule):
    """Dense full-spectrum resources and verification requirements."""

    tolerance: EigenTolerancePolicy
    resources: EigenResourcePolicy
    materialization: MaterializationPolicy
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    max_retained_bytes: int = eqx.field(static=True)
    failure: FailurePolicy

    def __init__(
        self,
        *,
        tolerance: EigenTolerancePolicy | None = None,
        resources: EigenResourcePolicy | None = None,
        materialization: MaterializationPolicy | None = None,
        relative_tolerance: float = 1e-8,
        absolute_tolerance: float = 1e-10,
        max_retained_bytes: int = 512 * 1024 * 1024,
        failure: FailurePolicy | None = None,
    ):
        tolerance_ = EigenTolerancePolicy() if tolerance is None else tolerance
        resources_ = EigenResourcePolicy() if resources is None else resources
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(tolerance_, EigenTolerancePolicy):
            raise TypeError("tolerance must be an EigenTolerancePolicy or None.")
        if not isinstance(resources_, EigenResourcePolicy):
            raise TypeError("resources must be an EigenResourcePolicy or None.")
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy or None.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy or None.")
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        retained = int(max_retained_bytes)
        if any(not math.isfinite(value) or value < 0.0 for value in (relative, absolute)):
            raise ValueError("Spectrum tolerances must be finite and non-negative.")
        if retained < 0:
            raise ValueError("max_retained_bytes must be non-negative.")
        self.tolerance = tolerance_
        self.resources = resources_
        self.materialization = materialization_
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.max_retained_bytes = retained
        self.failure = failure_


class SelfAdjointSpectrumCostEstimate(StrictModule):
    """Full dense eigensolve and reusable spectral-state cost."""

    source: EigenCostEstimate
    retained_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        source: EigenCostEstimate,
        /,
        *,
        retained_bytes: int,
        workspace_bytes: int,
    ):
        if not isinstance(source, EigenCostEstimate):
            raise TypeError("source must be an EigenCostEstimate.")
        retained = int(retained_bytes)
        workspace = int(workspace_bytes)
        if retained < 0 or workspace < 0:
            raise ValueError("Spectrum cost estimates must be non-negative.")
        self.source = source
        self.retained_bytes = retained
        self.workspace_bytes = workspace


class SelfAdjointSpectrumPlan(StrictModule):
    """Symbolic full-spectrum plan backed by one dense eigen plan."""

    policy: SelfAdjointSpectrumPolicy
    eigen_plan: EigenSolvePlan
    cost: SelfAdjointSpectrumCostEstimate
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: EigenproblemLike,
        policy: SelfAdjointSpectrumPolicy,
        eigen_plan: EigenSolvePlan,
        cost: SelfAdjointSpectrumCostEstimate,
        /,
    ):
        _require_problem(problem)
        if not isinstance(policy, SelfAdjointSpectrumPolicy):
            raise TypeError("policy must be a SelfAdjointSpectrumPolicy.")
        if not isinstance(eigen_plan, EigenSolvePlan):
            raise TypeError("eigen_plan must be an EigenSolvePlan.")
        if not isinstance(eigen_plan.selected_method, DenseEigh):
            raise ValueError("A self-adjoint spectrum requires DenseEigh.")
        if eigen_plan.policy.count != problem.dimension:
            raise ValueError("A self-adjoint spectrum plan must retain the full spectrum.")
        if not isinstance(cost, SelfAdjointSpectrumCostEstimate):
            raise TypeError("cost must be a SelfAdjointSpectrumCostEstimate.")
        self.policy = policy
        self.eigen_plan = eigen_plan
        self.cost = cost
        self.problem_id = problem.problem_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "self-adjoint-spectrum-plan",
                "problem": problem.problem_id,
                "eigen_plan": eigen_plan.plan_id,
                "relative_tolerance": policy.relative_tolerance,
                "absolute_tolerance": policy.absolute_tolerance,
                "max_retained_bytes": policy.max_retained_bytes,
                "failure": policy.failure.mode,
            }
        )


class SelfAdjointSpectrumDiagnostics(StrictModule):
    """Full-spectrum residual, normalization, and resource evidence."""

    maximum_residual_norm: Array
    maximum_relative_residual: Array
    normalization_error: Array
    finite: Array
    converged: Array
    source_status: Array
    retained_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        maximum_residual_norm: ArrayLike,
        maximum_relative_residual: ArrayLike,
        normalization_error: ArrayLike,
        finite: ArrayLike,
        converged: ArrayLike,
        source_status: ArrayLike,
        /,
        *,
        retained_bytes: int,
        workspace_bytes: int,
    ):
        residual = jnp.asarray(maximum_residual_norm)
        relative = jnp.asarray(maximum_relative_residual)
        normalization = jnp.asarray(normalization_error)
        finite_ = jnp.asarray(finite, dtype=bool)
        converged_ = jnp.asarray(converged, dtype=bool)
        source = jnp.asarray(source_status, dtype=jnp.int32)
        batch_shape = residual.shape
        if any(
            value.shape != batch_shape
            for value in (relative, normalization, finite_, converged_, source)
        ):
            raise ValueError("Spectrum diagnostics must share one batch shape.")
        retained = int(retained_bytes)
        workspace = int(workspace_bytes)
        if retained < 0 or workspace < 0:
            raise ValueError("Spectrum diagnostic resource counts must be non-negative.")
        self.maximum_residual_norm = residual
        self.maximum_relative_residual = relative
        self.normalization_error = normalization
        self.finite = finite_
        self.converged = converged_
        self.source_status = source
        self.retained_bytes = retained
        self.workspace_bytes = workspace


class SelfAdjointSpectrumProvenance(StrictModule):
    """Problem, plan, and numerical-version identity for a full spectrum."""

    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    eigen_plan_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    symbolic_version: int = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)


class PreparedSelfAdjointSpectrum(StrictModule):
    """Reusable stopped full eigensystem with a live mathematical problem."""

    problem: EigenproblemLike
    plan: SelfAdjointSpectrumPlan
    eigen_prepared: PreparedEigenSolve
    eigenvalues: Array
    eigenvectors: Array
    paired_metric: Array
    inverse_basis: Array
    source_diagnostics: EigenSolveDiagnostics
    status: Array
    diagnostics: SelfAdjointSpectrumDiagnostics
    provenance: SelfAdjointSpectrumProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(SelfAdjointSpectrumStatus.SUCCESS)


def plan_self_adjoint_spectrum(
    problem: EigenproblemLike,
    policy: SelfAdjointSpectrumPolicy | None = None,
    /,
) -> SelfAdjointSpectrumPlan:
    """Plan one bounded reusable full dense self-adjoint spectrum."""
    _require_problem(problem)
    selected = SelfAdjointSpectrumPolicy() if policy is None else policy
    if not isinstance(selected, SelfAdjointSpectrumPolicy):
        raise TypeError("policy must be a SelfAdjointSpectrumPolicy or None.")
    if problem.constraints is not None and problem.constraints.capacity > 0:
        raise ValueError("Self-adjoint spectra do not support excluded constraints.")
    eigen_policy = EigenSolvePolicy(
        DenseEigh(),
        count=problem.dimension,
        max_steps=1,
        tolerance=selected.tolerance,
        resources=selected.resources,
        materialization=selected.materialization,
        differentiation="none",
        failure=selected.failure,
    )
    eigen_plan = plan_eigensolve(problem, eigen_policy)
    source_cost = next(estimate for estimate in eigen_plan.candidates if estimate.accepted)
    coordinate_dtype = np.dtype(_coordinate_dtype(problem.operator.source))
    real_dtype = np.empty((), dtype=coordinate_dtype).real.dtype
    n = problem.dimension
    batch_count = int(np.prod(problem.batch_shape)) if problem.batch_shape else 1
    extra_retained = batch_count * (
        3 * n * n * coordinate_dtype.itemsize + n * real_dtype.itemsize
    )
    retained = source_cost.storage_bytes + extra_retained
    workspace = (
        source_cost.preparation_workspace_bytes
        + 2 * batch_count * n * n * coordinate_dtype.itemsize
    )
    if retained > selected.max_retained_bytes:
        raise ValueError(
            f"Self-adjoint spectrum retained estimate {retained} exceeds limit "
            f"{selected.max_retained_bytes}."
        )
    cost = SelfAdjointSpectrumCostEstimate(
        source_cost,
        retained_bytes=retained,
        workspace_bytes=workspace,
    )
    return SelfAdjointSpectrumPlan(problem, selected, eigen_plan, cost)


def prepare_self_adjoint_spectrum(
    problem: EigenproblemLike,
    policy: SelfAdjointSpectrumPolicy | SelfAdjointSpectrumPlan | None = None,
    /,
) -> PreparedSelfAdjointSpectrum:
    """Prepare and retain one reusable full dense self-adjoint eigensystem."""
    _require_problem(problem)
    if isinstance(policy, SelfAdjointSpectrumPlan):
        plan = policy
        if plan.problem_id != problem.problem_id:
            raise ValueError("Spectrum plan and problem IDs must match.")
        replanned = plan_self_adjoint_spectrum(problem, plan.policy)
        if replanned.plan_id != plan.plan_id:
            raise ValueError("Spectrum plan does not match the problem structure.")
    else:
        plan = plan_self_adjoint_spectrum(problem, policy)
    eigen_prepared = prepare_eigensolve(problem, plan.eigen_plan)
    return _build_prepared_spectrum(problem, plan, eigen_prepared)


def refresh_self_adjoint_spectrum(
    prepared: PreparedSelfAdjointSpectrum,
    problem: EigenproblemLike,
    /,
) -> PreparedSelfAdjointSpectrum:
    """Refresh spectrum coefficients while preserving symbolic identities."""
    if not isinstance(prepared, PreparedSelfAdjointSpectrum):
        raise TypeError("prepared must be a PreparedSelfAdjointSpectrum.")
    _require_problem(problem)
    replanned = plan_self_adjoint_spectrum(problem, prepared.plan.policy)
    if replanned.plan_id != prepared.plan.plan_id:
        raise ValueError("Spectrum refresh must preserve the symbolic plan.")
    eigen_prepared = refresh_eigensolve(prepared.eigen_prepared, problem)
    return _build_prepared_spectrum(problem, prepared.plan, eigen_prepared)


def self_adjoint_spectrum(
    problem_or_prepared: EigenproblemLike | PreparedSelfAdjointSpectrum,
    /,
    *,
    policy: SelfAdjointSpectrumPolicy | SelfAdjointSpectrumPlan | None = None,
) -> PreparedSelfAdjointSpectrum:
    """Return reusable full-spectrum state for a prepared or unprepared problem."""
    if isinstance(problem_or_prepared, PreparedSelfAdjointSpectrum):
        if policy is not None:
            raise ValueError("policy must be omitted for prepared spectrum state.")
        return problem_or_prepared
    return prepare_self_adjoint_spectrum(problem_or_prepared, policy)


def _build_prepared_spectrum(
    problem: EigenproblemLike,
    plan: SelfAdjointSpectrumPlan,
    eigen_prepared: PreparedEigenSolve,
    /,
) -> PreparedSelfAdjointSpectrum:
    source = eigensolve(eigen_prepared)
    space = problem.operator.source
    values = jax.lax.stop_gradient(source.eigenvalues)
    if problem.batch_shape:
        vectors = jax.lax.stop_gradient(
            jnp.asarray(source.eigenvectors).reshape(
                problem.batch_shape + (problem.dimension, problem.dimension)
            )
        )
    else:
        vectors = jax.lax.stop_gradient(
            jax.vmap(space.flatten, in_axes=-1, out_axes=1)(source.eigenvectors)
        )
    if eigen_prepared.dense_state is None:
        raise ValueError("Prepared self-adjoint spectrum lost its dense state.")
    factor = eigen_prepared.dense_state.metric_factor
    paired_metric = jax.lax.stop_gradient(
        factor @ jnp.conj(jnp.swapaxes(factor, -1, -2))
    )
    inverse_basis = jax.lax.stop_gradient(
        jnp.conj(jnp.swapaxes(vectors, -1, -2)) @ paired_metric
    )
    identity = jnp.eye(problem.dimension, dtype=vectors.dtype)
    normalization = jnp.linalg.norm(
        inverse_basis @ vectors - identity,
        axis=(-2, -1),
    )
    maximum_residual = jnp.max(source.residual_norms, axis=-1)
    maximum_relative = jnp.max(source.relative_residuals, axis=-1)
    finite = (
        jnp.all(jnp.isfinite(values), axis=-1)
        & jnp.all(jnp.isfinite(vectors), axis=(-2, -1))
        & jnp.all(jnp.isfinite(paired_metric), axis=(-2, -1))
        & jnp.all(jnp.isfinite(inverse_basis), axis=(-2, -1))
        & jnp.isfinite(normalization)
    )
    normalization_tolerance = (
        plan.policy.absolute_tolerance
        + plan.policy.relative_tolerance * max(problem.dimension, 1)
    )
    source_success = source.status == int(EigenSolveStatus.SUCCESS)
    normalization_ok = normalization <= normalization_tolerance
    status = jnp.where(
        ~finite,
        int(SelfAdjointSpectrumStatus.NONFINITE),
        jnp.where(
            ~source_success,
            int(SelfAdjointSpectrumStatus.SOURCE_FAILURE),
            jnp.where(
                ~normalization_ok,
                int(SelfAdjointSpectrumStatus.NORMALIZATION_RESIDUAL_TOO_LARGE),
                int(SelfAdjointSpectrumStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    if plan.policy.failure.mode == "error":
        values = eqx.error_if(
            values,
            jnp.any(status != int(SelfAdjointSpectrumStatus.SUCCESS)),
            "Self-adjoint spectrum did not satisfy its numerical contract.",
        )
    diagnostics = SelfAdjointSpectrumDiagnostics(
        maximum_residual,
        maximum_relative,
        normalization,
        finite,
        status == int(SelfAdjointSpectrumStatus.SUCCESS),
        source.status,
        retained_bytes=plan.cost.retained_bytes,
        workspace_bytes=plan.cost.workspace_bytes,
    )
    provenance = SelfAdjointSpectrumProvenance(
        problem_id=problem.problem_id,
        plan_id=plan.plan_id,
        eigen_plan_id=plan.eigen_plan.plan_id,
        method="phydrax-native-dense-eigh",
        symbolic_version=eigen_prepared.symbolic_version,
        numeric_version=eigen_prepared.numeric_version,
    )
    return PreparedSelfAdjointSpectrum(
        problem=problem,
        plan=plan,
        eigen_prepared=eigen_prepared,
        eigenvalues=values,
        eigenvectors=vectors,
        paired_metric=paired_metric,
        inverse_basis=inverse_basis,
        source_diagnostics=source.diagnostics,
        status=status,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _require_problem(problem: object, /) -> None:
    if not isinstance(problem, (Eigenproblem, GeneralizedEigenproblem)):
        raise TypeError("problem must be an Eigenproblem or GeneralizedEigenproblem.")


__all__ = [
    "PreparedSelfAdjointSpectrum",
    "SelfAdjointSpectrumCostEstimate",
    "SelfAdjointSpectrumDiagnostics",
    "SelfAdjointSpectrumPlan",
    "SelfAdjointSpectrumPolicy",
    "SelfAdjointSpectrumProvenance",
    "SelfAdjointSpectrumStatus",
    "plan_self_adjoint_spectrum",
    "prepare_self_adjoint_spectrum",
    "refresh_self_adjoint_spectrum",
    "self_adjoint_spectrum",
]
