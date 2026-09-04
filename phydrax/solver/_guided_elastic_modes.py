#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    eigen as eigen_linalg,
    FailurePolicy,
    OperatorProperties,
)


class GuidedElasticModeStatus(IntEnum):
    """Portable status for a fixed-q guided elastic eigensolve."""

    SUCCESS = 0
    EIGENSOLVE_FAILURE = 1
    NONFINITE_OUTPUT = 2
    NEGATIVE_SQUARED_FREQUENCY = 3
    MASS_NORMALIZATION_FAILURE = 4


class GuidedElasticModeResult(StrictModule):
    """Mass-normalized guided elastic modes at one axial wavenumber."""

    axial_wavenumber: Array
    squared_angular_frequencies: Array
    angular_frequencies: Array
    displacement_modes: Array
    residuals: Array
    modal_masses: Array
    modal_stiffnesses: Array
    orthogonality_matrix: Array
    orthogonality_error: Array
    status: Array
    diagnostics: eigen_linalg.EigenSolveDiagnostics
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(GuidedElasticModeStatus.SUCCESS)


class GuidedElasticModePlan(StrictModule, NonTrainableState):
    """Generalized elastic pencil ``K(q) u = Ω² M u`` at fixed real q.

    Operators may be assembled from :func:`guided_elasticity_form` and a FEM
    mass form, or supplied as analytic dense fixtures. Operator inputs retain
    their Phydrax vector-space and property certificates.
    """

    stiffness_operator: AbstractLinearOperator
    mass_operator: AbstractLinearOperator
    eigen_policy: eigen_linalg.EigenSolvePolicy
    axial_wavenumber: Array
    mode_count: int = eqx.field(static=True)
    negative_eigenvalue_tolerance: float = eqx.field(static=True)
    orthogonality_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stiffness_operator: AbstractLinearOperator | ArrayLike,
        mass_operator: AbstractLinearOperator | ArrayLike,
        mode_count: int,
        /,
        *,
        axial_wavenumber: float,
        eigen_policy: eigen_linalg.EigenSolvePolicy | None = None,
        negative_eigenvalue_tolerance: float = 1e-10,
        orthogonality_tolerance: float = 1e-8,
        maximum_dofs: int = 4096,
    ):
        stiffness, mass = _guided_elastic_operators(stiffness_operator, mass_operator)
        if not isinstance(stiffness.source, ArraySpace):
            raise TypeError("Guided elastic mode coordinates must use an ArraySpace.")
        dimension = stiffness.source.size
        count = int(mode_count)
        maximum = int(maximum_dofs)
        if count < 1 or count > dimension:
            raise ValueError("mode_count is outside the guided elastic dimension.")
        if maximum < 1 or dimension > maximum:
            raise ValueError("Guided elastic solve exceeds maximum_dofs.")
        q = float(axial_wavenumber)
        if not math.isfinite(q):
            raise ValueError("axial_wavenumber must be finite.")
        negative = float(negative_eigenvalue_tolerance)
        orthogonality = float(orthogonality_tolerance)
        if any(
            not math.isfinite(value) or value < 0.0 for value in (negative, orthogonality)
        ):
            raise ValueError("Guided elastic tolerances must be finite and non-negative.")
        policy = (
            eigen_linalg.EigenSolvePolicy(
                eigen_linalg.DenseEigh(),
                count=count,
                which="smallest-algebraic",
                differentiation="eigenvalues",
                failure=FailurePolicy("status"),
            )
            if eigen_policy is None
            else eigen_policy
        )
        if not isinstance(policy, eigen_linalg.EigenSolvePolicy):
            raise TypeError("eigen_policy must be an EigenSolvePolicy or None.")
        if policy.count != count:
            raise ValueError("eigen_policy must request exactly mode_count eigenpairs.")
        self.stiffness_operator = stiffness
        self.mass_operator = mass
        self.eigen_policy = policy
        self.axial_wavenumber = jnp.asarray(q)
        self.mode_count = count
        self.negative_eigenvalue_tolerance = negative
        self.orthogonality_tolerance = orthogonality
        self.plan_id = canonical_fingerprint(
            {
                "kind": "guided-elastic-mode-plan",
                "stiffness": stiffness.operator_id,
                "mass": mass.operator_id,
                "axial_wavenumber": q,
                "mode_count": count,
                "which": policy.which,
                "negative_eigenvalue_tolerance": negative,
                "orthogonality_tolerance": orthogonality,
            }
        )

    def prepare(self, /) -> "PreparedGuidedElasticModes":
        return prepare_guided_elastic_modes(self)

    def solve(self, /) -> GuidedElasticModeResult:
        return solve_guided_elastic_modes(self.prepare())


class PreparedGuidedElasticModes(StrictModule):
    """Reusable eigensolver state for one fixed-q elastic mode plan."""

    plan: GuidedElasticModePlan
    eigen: eigen_linalg.PreparedEigenSolve
    prepared_id: str = eqx.field(static=True)


def prepare_guided_elastic_modes(
    plan: GuidedElasticModePlan,
    /,
) -> PreparedGuidedElasticModes:
    if not isinstance(plan, GuidedElasticModePlan):
        raise TypeError("plan must be a GuidedElasticModePlan.")
    problem = eigen_linalg.GeneralizedEigenproblem(
        plan.stiffness_operator,
        plan.mass_operator,
        problem_id=f"{plan.plan_id}:eigenproblem",
    )
    eigen = eigen_linalg.prepare_eigensolve(problem, plan.eigen_policy)
    return PreparedGuidedElasticModes(
        plan=plan,
        eigen=eigen,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-guided-elastic-modes",
                "plan": plan.plan_id,
                "eigen": eigen.plan.plan_id,
            }
        ),
    )


def solve_guided_elastic_modes(
    prepared: PreparedGuidedElasticModes,
    /,
) -> GuidedElasticModeResult:
    if not isinstance(prepared, PreparedGuidedElasticModes):
        raise TypeError("prepared must be PreparedGuidedElasticModes.")
    plan = prepared.plan
    solved = eigen_linalg.eigensolve(prepared.eigen)
    squared = jnp.real(solved.eigenvalues)
    modes = jnp.asarray(solved.eigenvectors)
    mass_images = plan.mass_operator.mv_block(modes)
    stiffness_images = plan.stiffness_operator.mv_block(modes)
    mass_matrix = jnp.conj(jnp.swapaxes(modes, -1, -2)) @ mass_images
    stiffness_matrix = jnp.conj(jnp.swapaxes(modes, -1, -2)) @ stiffness_images
    modal_masses = jnp.real(jnp.diag(mass_matrix))
    modal_stiffnesses = jnp.real(jnp.diag(stiffness_matrix))
    identity = jnp.eye(plan.mode_count, dtype=mass_matrix.dtype)
    orthogonality_error = jnp.max(jnp.abs(mass_matrix - identity))
    negative = squared < -plan.negative_eigenvalue_tolerance
    angular_frequencies = jnp.sqrt(jnp.maximum(squared, 0.0))
    finite = (
        jnp.all(jnp.isfinite(squared))
        & jnp.all(jnp.isfinite(modes))
        & jnp.all(jnp.isfinite(mass_matrix))
        & jnp.all(jnp.isfinite(stiffness_matrix))
    )
    mass_valid = jnp.all(modal_masses > 0.0) & (
        orthogonality_error <= plan.orthogonality_tolerance
    )
    status = jnp.where(
        solved.status != int(eigen_linalg.EigenSolveStatus.SUCCESS),
        int(GuidedElasticModeStatus.EIGENSOLVE_FAILURE),
        jnp.where(
            ~finite,
            int(GuidedElasticModeStatus.NONFINITE_OUTPUT),
            jnp.where(
                jnp.any(negative),
                int(GuidedElasticModeStatus.NEGATIVE_SQUARED_FREQUENCY),
                jnp.where(
                    ~mass_valid,
                    int(GuidedElasticModeStatus.MASS_NORMALIZATION_FAILURE),
                    int(GuidedElasticModeStatus.SUCCESS),
                ),
            ),
        ),
    ).astype(jnp.int32)
    mode_ids = tuple(f"{plan.plan_id}:mode:{index}" for index in range(plan.mode_count))
    return GuidedElasticModeResult(
        axial_wavenumber=plan.axial_wavenumber,
        squared_angular_frequencies=squared,
        angular_frequencies=angular_frequencies,
        displacement_modes=modes,
        residuals=solved.diagnostics.residual_norms,
        modal_masses=modal_masses,
        modal_stiffnesses=modal_stiffnesses,
        orthogonality_matrix=mass_matrix,
        orthogonality_error=orthogonality_error,
        status=status,
        diagnostics=solved.diagnostics,
        mode_ids=mode_ids,
        result_id=canonical_fingerprint(
            {
                "kind": "guided-elastic-mode-result",
                "prepared": prepared.prepared_id,
                "eigen_plan": solved.provenance.plan_id,
            }
        ),
    )


def _guided_elastic_operators(stiffness, mass):
    stiffness_is_operator = isinstance(stiffness, AbstractLinearOperator)
    mass_is_operator = isinstance(mass, AbstractLinearOperator)
    if stiffness_is_operator != mass_is_operator:
        raise TypeError(
            "stiffness_operator and mass_operator must both be operators or both be arrays."
        )
    if stiffness_is_operator:
        if (
            not stiffness.source.compatible(stiffness.target)
            or not mass.source.compatible(mass.target)
            or not stiffness.source.compatible(mass.source)
        ):
            raise ValueError(
                "Guided elastic operators must share one endomorphism space."
            )
        eigen_linalg.GeneralizedEigenproblem(stiffness, mass)
        return stiffness, mass
    stiffness_host = np.asarray(stiffness)
    mass_host = np.asarray(mass)
    if (
        stiffness_host.ndim != 2
        or stiffness_host.shape[0] != stiffness_host.shape[1]
        or mass_host.shape != stiffness_host.shape
        or stiffness_host.shape[0] < 1
    ):
        raise ValueError(
            "Guided elastic stiffness and mass must be equally sized square matrices."
        )
    if np.any(~np.isfinite(stiffness_host)) or np.any(~np.isfinite(mass_host)):
        raise ValueError("Guided elastic matrices must be finite.")
    scale = max(
        1.0, float(np.linalg.norm(stiffness_host)), float(np.linalg.norm(mass_host))
    )
    tolerance = (
        64.0 * np.finfo(np.result_type(stiffness_host.real, mass_host.real)).eps * scale
    )
    if not np.allclose(
        stiffness_host, stiffness_host.conj().T, rtol=1e-10, atol=tolerance
    ):
        raise ValueError("Guided elastic stiffness must be Hermitian.")
    if not np.allclose(mass_host, mass_host.conj().T, rtol=1e-10, atol=tolerance):
        raise ValueError("Guided elastic mass must be Hermitian.")
    if np.linalg.eigvalsh(mass_host)[0] <= tolerance:
        raise ValueError("Guided elastic mass must be positive definite.")
    dtype = jnp.result_type(stiffness_host, mass_host)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.dtype(float)
    space = ArraySpace((stiffness_host.shape[0],), dtype=dtype)
    stiffness_fingerprint = array_tree_fingerprint(stiffness_host)
    mass_fingerprint = array_tree_fingerprint(mass_host)
    stiffness_operator = DenseLinearOperator(
        jnp.asarray(stiffness_host, dtype=dtype),
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "verified"},
        ),
        operator_id=f"guided-elastic-stiffness:{stiffness_fingerprint}",
    )
    mass_operator = DenseLinearOperator(
        jnp.asarray(mass_host, dtype=dtype),
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "verified",
                "positive_definite": "verified",
            },
        ),
        operator_id=f"guided-elastic-mass:{mass_fingerprint}",
    )
    return stiffness_operator, mass_operator


__all__ = [
    "GuidedElasticModePlan",
    "GuidedElasticModeResult",
    "GuidedElasticModeStatus",
    "PreparedGuidedElasticModes",
    "prepare_guided_elastic_modes",
    "solve_guided_elastic_modes",
]
