#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, eigen as eigen_api, OperatorProperties
from ._perturbation import PerturbationStatus, SeparatedMode


def _cosine_coupling(ell: int, m: int, spin_weight: int, /) -> float:
    """Coefficient of the ``ell - 1`` harmonic in ``cos(theta) Y_ell``."""
    if ell == 0:
        return 0.0
    numerator = (ell * ell - m * m) * (ell * ell - spin_weight * spin_weight)
    if numerator <= 0:
        return 0.0
    denominator = ell * ell * (2 * ell - 1) * (2 * ell + 1)
    return math.sqrt(numerator / denominator)


def _cosine_diagonal(ell: int, m: int, spin_weight: int, /) -> float:
    if ell == 0:
        return 0.0
    return -(m * spin_weight) / (ell * (ell + 1))


class SpheroidalAngularPlan(StrictModule, NonTrainableState):
    """Fixed spin-spherical Galerkin basis for one spheroidal mode identity.

    The angular separation constant ``A`` is defined by

    ``[L_s + c**2 cos(theta)**2 - 2 c s cos(theta) + A] S = 0``,

    where ``c = a omega`` and ``A(c=0) = ell(ell+1)-s(s+1)``.  The stored
    ``cosine_squared_matrix`` is the direct projected ``cos(theta)**2`` action;
    it retains the path through the first mode above the truncation rather than
    incorrectly squaring the truncated cosine matrix.
    """

    mode: SeparatedMode
    ell_values: Array
    spherical_separation: Array
    cosine_matrix: Array
    cosine_squared_matrix: Array
    maximum_ell: int = eqx.field(static=True)
    target_index: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    isolation_tolerance: float = eqx.field(static=True)
    minimum_target_overlap: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: SeparatedMode,
        maximum_ell: int,
        /,
        *,
        residual_tolerance: float = 1.0e-6,
        isolation_tolerance: float = 1.0e-8,
        minimum_target_overlap: float = 1.0e-6,
        maximum_condition: float = 1.0e10,
    ):
        if not isinstance(mode, SeparatedMode):
            raise TypeError("mode must be a SeparatedMode.")
        if isinstance(maximum_ell, bool) or not isinstance(maximum_ell, Integral):
            raise TypeError("maximum_ell must be an integer.")
        maximum = int(maximum_ell)
        minimum = max(abs(mode.spin_weight), abs(mode.m))
        if maximum < mode.ell:
            raise ValueError("maximum_ell must include the target mode ell.")
        tolerances = tuple(
            float(value)
            for value in (
                residual_tolerance,
                isolation_tolerance,
                minimum_target_overlap,
                maximum_condition,
            )
        )
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in tolerances)
            or tolerances[2] > 1.0
            or tolerances[3] <= 1.0
        ):
            raise ValueError("Spheroidal tolerances and condition limit are invalid.")

        ell_values_host = np.arange(minimum, maximum + 1, dtype=np.int32)
        extended_ells = np.arange(minimum, maximum + 2, dtype=np.int32)
        extended_size = extended_ells.size
        cosine_extended = np.zeros((extended_size, extended_size), dtype=np.float64)
        for index, ell in enumerate(extended_ells):
            cosine_extended[index, index] = _cosine_diagonal(
                int(ell), mode.m, mode.spin_weight
            )
            if index + 1 < extended_size:
                coupling = _cosine_coupling(
                    int(ell) + 1,
                    mode.m,
                    mode.spin_weight,
                )
                cosine_extended[index, index + 1] = coupling
                cosine_extended[index + 1, index] = coupling
        size = ell_values_host.size
        cosine_host = cosine_extended[:size, :size]
        cosine_squared_host = (cosine_extended @ cosine_extended)[:size, :size]
        spherical_host = ell_values_host * (ell_values_host + 1) - mode.spin_weight * (
            mode.spin_weight + 1
        )
        target_index = mode.ell - minimum
        self.mode = mode
        self.ell_values = jnp.asarray(ell_values_host)
        self.spherical_separation = jnp.asarray(spherical_host, dtype=jnp.float64)
        self.cosine_matrix = jnp.asarray(cosine_host)
        self.cosine_squared_matrix = jnp.asarray(cosine_squared_host)
        self.maximum_ell = maximum
        self.target_index = target_index
        (
            self.residual_tolerance,
            self.isolation_tolerance,
            self.minimum_target_overlap,
            self.maximum_condition,
        ) = tolerances
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spin-weighted-spheroidal-angular-plan",
                "mode": mode.mode_id,
                "minimum_ell": minimum,
                "maximum_ell": maximum,
                "target_index": target_index,
                "residual_tolerance": tolerances[0],
                "isolation_tolerance": tolerances[1],
                "minimum_target_overlap": tolerances[2],
                "maximum_condition": tolerances[3],
                "basis": array_tree_fingerprint(
                    (
                        ell_values_host,
                        spherical_host,
                        cosine_host,
                        cosine_squared_host,
                    )
                ),
            }
        )

    @property
    def basis_size(self) -> int:
        return self.ell_values.size

    def matrix(self, spheroidicity: ArrayLike, /) -> Array:
        return spheroidal_angular_matrix(self, spheroidicity)

    def operator(self, spheroidicity: ArrayLike, /) -> DenseLinearOperator:
        return spheroidal_angular_operator(self, spheroidicity)


class SpheroidalAngularResult(StrictModule):
    """Selected angular mode and independent residual/branch evidence."""

    spheroidicity: Array
    separation_constant: Array
    coefficients: Array
    residual: Array
    relative_residual: Array
    normalization_defect: Array
    target_overlap: Array
    condition: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    mode_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return self.qualified


def spheroidal_angular_matrix(
    plan: SpheroidalAngularPlan,
    spheroidicity: ArrayLike,
    /,
) -> Array:
    """Construct the projected angular ``A`` matrix for ``c = a omega``."""
    if not isinstance(plan, SpheroidalAngularPlan):
        raise TypeError("plan must be a SpheroidalAngularPlan.")
    c = jnp.asarray(spheroidicity)
    if c.shape != ():
        raise ValueError("spheroidicity must be scalar.")
    dtype = jnp.result_type(c, plan.cosine_matrix)
    diagonal = jnp.diag(plan.spherical_separation.astype(dtype))
    cosine = plan.cosine_matrix.astype(dtype)
    cosine_squared = plan.cosine_squared_matrix.astype(dtype)
    return diagonal + 2.0 * plan.mode.spin_weight * c * cosine - c * c * cosine_squared


def spheroidal_angular_operator(
    plan: SpheroidalAngularPlan,
    spheroidicity: ArrayLike,
    /,
) -> DenseLinearOperator:
    """Return a dense operator, certifying self-adjointness only for real ``c``."""
    matrix = spheroidal_angular_matrix(plan, spheroidicity)
    properties = (
        OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
        )
        if not jnp.iscomplexobj(matrix)
        else OperatorProperties()
    )
    return DenseLinearOperator(
        matrix,
        properties=properties,
        operator_id=f"{plan.plan_id}:angular-operator",
    )


def spheroidal_angular_residual(
    plan: SpheroidalAngularPlan,
    spheroidicity: ArrayLike,
    separation_constant: ArrayLike,
    /,
) -> Array:
    """Return a dimensionless complex characteristic residual ``det(A(c)-A I)``."""
    matrix = spheroidal_angular_matrix(plan, spheroidicity)
    separation = jnp.asarray(separation_constant)
    if separation.shape != ():
        raise ValueError("separation_constant must be scalar.")
    pencil = matrix - separation * jnp.eye(plan.basis_size, dtype=matrix.dtype)
    scale = jnp.maximum(jnp.max(jnp.abs(pencil)), 1.0)
    return jnp.linalg.det(pencil / scale)


def _vector_norm(vector: Array, /) -> Array:
    squared = contract("i,i->", jnp.conj(vector), vector)
    return jnp.sqrt(jnp.maximum(jnp.real(squared), 0.0))


def _phase_normalize(plan: SpheroidalAngularPlan, coefficients: Array, /) -> Array:
    norm = _vector_norm(coefficients)
    normalized = coefficients / jnp.where(norm > 0.0, norm, 1.0)
    anchor = normalized[plan.target_index]
    magnitude = jnp.abs(anchor)
    phase = jnp.where(magnitude > 0.0, jnp.conj(anchor) / magnitude, 1.0)
    return normalized * phase


def _solve_real(
    plan: SpheroidalAngularPlan,
    spheroidicity: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    operator = spheroidal_angular_operator(plan, spheroidicity)
    solved = eigen_api.eigensolve(
        eigen_api.Eigenproblem(
            operator,
            problem_id=f"{plan.plan_id}:self-adjoint-angular",
        ),
        policy=eigen_api.EigenSolvePolicy(
            eigen_api.DenseEigh(),
            count=plan.basis_size,
            which="smallest-algebraic",
            differentiation="eigenvalues",
        ),
    )
    index = plan.target_index
    separation = solved.eigenvalues[index]
    coefficients = solved.eigenvectors[:, index]
    gap = solved.diagnostics.isolation_gaps[index]
    scale = jnp.maximum(jnp.abs(separation), 1.0)
    condition = scale / jnp.maximum(gap, jnp.finfo(separation.dtype).tiny)
    return separation, coefficients, solved.converged[index], condition


def _solve_complex(
    plan: SpheroidalAngularPlan,
    spheroidicity: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    operator = spheroidal_angular_operator(plan, spheroidicity)
    target = float(np.asarray(plan.spherical_separation[plan.target_index]))
    solved = eigen_api.general_eigensolve(
        eigen_api.GeneralEigenproblem(
            operator,
            problem_id=f"{plan.plan_id}:general-angular",
        ),
        policy=eigen_api.GeneralEigenSolvePolicy(
            eigen_api.DenseSchurQZ(),
            selection=eigen_api.GeneralEigenSelection.closest(target, 1),
            tolerance=eigen_api.GeneralEigenTolerancePolicy(
                relative=plan.residual_tolerance,
                absolute=0.01 * plan.residual_tolerance,
            ),
        ),
    )
    return (
        solved.eigenvalues[0],
        solved.right_eigenvector_coordinates[:, 0],
        solved.successful,
        solved.diagnostics.eigenvalue_condition_estimates[0],
    )


def solve_spheroidal_angular(
    plan: SpheroidalAngularPlan,
    spheroidicity: ArrayLike,
    /,
) -> SpheroidalAngularResult:
    """Solve and qualify the target spheroidal branch on the plan's fixed basis.

    Real ``c`` uses the JIT-compatible native self-adjoint eigensolver.  Complex
    ``c`` uses the native host QZ path and is deliberately marked
    ``derivative_valid=False`` even when its numerical residual is qualified.
    """
    if not isinstance(plan, SpheroidalAngularPlan):
        raise TypeError("plan must be a SpheroidalAngularPlan.")
    c = jnp.asarray(spheroidicity)
    if c.shape != ():
        raise ValueError("spheroidicity must be scalar.")
    real_problem = not jnp.iscomplexobj(c)
    separation, raw_coefficients, native_converged, condition = (
        _solve_real(plan, c) if real_problem else _solve_complex(plan, c)
    )
    coefficients = _phase_normalize(plan, raw_coefficients)
    matrix = spheroidal_angular_matrix(plan, c)
    vector_residual = (
        contract("ij,j->i", matrix, coefficients) - separation * coefficients
    )
    residual = _vector_norm(vector_residual)
    image_norm = _vector_norm(contract("ij,j->i", matrix, coefficients))
    denominator = image_norm + jnp.abs(separation) * _vector_norm(coefficients)
    relative_residual = residual / jnp.maximum(
        denominator,
        jnp.finfo(residual.dtype).tiny,
    )
    normalization_defect = jnp.abs(_vector_norm(coefficients) - 1.0)
    target_overlap = jnp.abs(coefficients[plan.target_index])
    finite = (
        jnp.isfinite(jnp.real(c))
        & jnp.isfinite(jnp.imag(c))
        & jnp.isfinite(jnp.real(separation))
        & jnp.isfinite(jnp.imag(separation))
        & jnp.all(jnp.isfinite(jnp.real(coefficients)))
        & jnp.all(jnp.isfinite(jnp.imag(coefficients)))
        & jnp.isfinite(residual)
        & jnp.isfinite(condition)
    )
    converged = jnp.asarray(native_converged, dtype=jnp.bool_)
    physically_valid = target_overlap >= plan.minimum_target_overlap
    residual_valid = relative_residual <= plan.residual_tolerance
    isolated = condition <= plan.maximum_condition
    qualified = finite & converged & physically_valid & residual_valid & isolated
    derivative_valid = (
        qualified & real_problem & (condition * plan.isolation_tolerance < 1.0)
    )
    status = jnp.where(
        ~finite,
        int(PerturbationStatus.NONFINITE),
        jnp.where(
            ~converged,
            int(PerturbationStatus.ANGULAR_NONCONVERGENCE),
            jnp.where(
                ~physically_valid,
                int(PerturbationStatus.INVALID_MODE),
                jnp.where(
                    ~residual_valid,
                    int(PerturbationStatus.RESIDUAL_TOLERANCE_NOT_MET),
                    jnp.where(
                        ~isolated,
                        int(PerturbationStatus.MODE_NOT_ISOLATED),
                        int(PerturbationStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    return SpheroidalAngularResult(
        c,
        separation,
        coefficients,
        residual,
        relative_residual,
        normalization_defect,
        target_overlap,
        condition,
        finite,
        converged,
        physically_valid,
        qualified,
        derivative_valid,
        status,
        plan.mode.mode_id,
        plan.plan_id,
        plan.mode.convention_id,
    )


__all__ = [
    "SpheroidalAngularPlan",
    "SpheroidalAngularResult",
    "solve_spheroidal_angular",
    "spheroidal_angular_matrix",
    "spheroidal_angular_operator",
    "spheroidal_angular_residual",
]
