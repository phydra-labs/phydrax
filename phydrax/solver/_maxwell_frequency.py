#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..discretization import CochainDiscretization
from ..linalg import DenseLinearOperator, eigen as eigen_linalg, OperatorProperties
from ._maxwell import AbstractPreparedMaxwellConstitutive


def _paired_matrix(metric: Array, matrix: Array, /) -> Array:
    return metric[:, None] * matrix if metric.ndim == 1 else metric @ matrix


def _verified_dense_operator(
    matrix: Array,
    name: str,
    /,
    *,
    positive_definite: bool = False,
) -> DenseLinearOperator:
    host = np.asarray(matrix)
    tolerance = (
        64.0
        * max(host.shape[0], 1)
        * np.finfo(host.real.dtype).eps
        * max(1.0, float(np.linalg.norm(host)))
    )
    if not np.allclose(host, host.conj().T, rtol=1e-10, atol=tolerance):
        raise ValueError(f"{name} must be Hermitian.")
    if (
        positive_definite
        and np.linalg.eigvalsh(0.5 * (host + host.conj().T))[0] <= tolerance
    ):
        raise ValueError(f"{name} must be positive definite.")
    evidence = {"self_adjoint": "verified"}
    if positive_definite:
        evidence["positive_definite"] = "verified"
    return DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=positive_definite,
            positive_semidefinite=not positive_definite,
            evidence=evidence,
        ),
        operator_id=canonical_fingerprint(
            {
                "kind": "verified-maxwell-eigen-operator",
                "name": name,
                "matrix": array_tree_fingerprint(matrix),
            }
        ),
    )


class FrequencyMaxwellSolveResult(StrictModule):
    electric: Array
    residual_norm: Array
    converged: Array
    iterations: Array


class FrequencyMaxwellEigenResult(StrictModule):
    angular_frequencies: Array
    modes: Array
    residuals: Array
    status: Array
    diagnostics: eigen_linalg.EigenSolveDiagnostics
    result_id: str = eqx.field(static=True)


class FrequencyMaxwellOperator(StrictModule):
    """Matrix-free electric curl-curl operator on compatible cochains."""

    cochain: CochainDiscretization
    constitutive: AbstractPreparedMaxwellConstitutive
    angular_frequency: Array
    material_state: Any
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        cochain: CochainDiscretization,
        constitutive: AbstractPreparedMaxwellConstitutive,
        angular_frequency: ArrayLike,
        /,
        *,
        material_state: Any = None,
    ):
        if not isinstance(cochain, CochainDiscretization) or cochain.max_degree != 3:
            raise TypeError("Frequency Maxwell requires a 3-D CochainDiscretization.")
        if not isinstance(constitutive, AbstractPreparedMaxwellConstitutive):
            raise TypeError("constitutive must be prepared Maxwell material data.")
        if not constitutive.capabilities.frequency_domain:
            raise ValueError("Constitutive law does not support frequency-domain use.")
        frequency = jnp.asarray(angular_frequency)
        if (
            frequency.shape != ()
            or bool(jnp.any(~jnp.isfinite(frequency)))
            or bool(frequency < 0.0)
        ):
            raise ValueError("angular_frequency must be a finite nonnegative scalar.")
        constitutive.validate_state(material_state)
        self.cochain = cochain
        self.constitutive = constitutive
        self.angular_frequency = frequency
        self.material_state = material_state
        self.operator_id = canonical_fingerprint(
            {
                "kind": "frequency-maxwell-operator",
                "cochain": cochain.prepared_id,
                "constitutive": constitutive.prepared_id,
                "angular_frequency": float(np.asarray(frequency)),
            }
        )

    @property
    def size(self) -> int:
        return self.cochain.cell_counts[1]

    def mv(self, electric: ArrayLike, /) -> Array:
        electric_ = jnp.asarray(electric)
        if electric_.shape != (self.size,):
            raise ValueError("Frequency Maxwell electric field has wrong shape.")
        curl = self.cochain.exterior_derivative(1, electric_)
        magnetic = self.constitutive.magnetic_field(curl, self.material_state)
        curl_curl = self.cochain.codifferential(2, magnetic)
        displacement = self.constitutive.electric_displacement(
            electric_, self.material_state
        )
        return curl_curl - self.angular_frequency**2 * displacement

    def adjoint_mv(self, electric: ArrayLike, /) -> Array:
        electric_ = jnp.asarray(electric)
        _, pullback = jax.vjp(self.mv, jnp.zeros_like(electric_))
        return pullback(electric_)[0]

    def solve(
        self,
        source: ArrayLike,
        /,
        *,
        tolerance: float = 1e-9,
        restart: int = 40,
        maxiter: int = 400,
    ) -> FrequencyMaxwellSolveResult:
        source_ = jnp.asarray(source)
        if source_.shape != (self.size,):
            raise ValueError("Frequency Maxwell source has wrong shape.")
        solution, info = gmres(
            self.mv,
            source_,
            tol=float(tolerance),
            restart=int(restart),
            maxiter=int(maxiter),
        )
        residual = jnp.linalg.norm(self.mv(solution) - source_)
        return FrequencyMaxwellSolveResult(
            solution,
            residual,
            info == 0,
            jnp.asarray(info),
        )

    def adjoint_solve(
        self,
        cotangent: ArrayLike,
        /,
        *,
        tolerance: float = 1e-9,
        restart: int = 40,
        maxiter: int = 400,
    ) -> FrequencyMaxwellSolveResult:
        cotangent_ = jnp.asarray(cotangent)
        solution, info = gmres(
            self.adjoint_mv,
            cotangent_,
            tol=float(tolerance),
            restart=int(restart),
            maxiter=int(maxiter),
        )
        residual = jnp.linalg.norm(self.adjoint_mv(solution) - cotangent_)
        return FrequencyMaxwellSolveResult(
            solution,
            residual,
            info == 0,
            jnp.asarray(info),
        )

    def materialize(self, /, *, maximum_dofs: int = 4096) -> Array:
        if self.size > int(maximum_dofs):
            raise ValueError("Frequency Maxwell materialization exceeds maximum_dofs.")
        basis = jnp.eye(self.size, dtype=complex)
        return jax.vmap(self.mv, in_axes=1, out_axes=1)(basis)

    def eigensystem(
        self,
        mode_count: int,
        /,
        *,
        maximum_dofs: int = 4096,
    ) -> FrequencyMaxwellEigenResult:
        count = int(mode_count)
        if count <= 0 or count > self.size:
            raise ValueError("mode_count is outside the operator dimension.")
        zero_frequency = FrequencyMaxwellOperator(
            self.cochain,
            self.constitutive,
            0.0,
            material_state=self.material_state,
        )
        stiffness = zero_frequency.materialize(maximum_dofs=maximum_dofs)
        identity = jnp.eye(self.size, dtype=stiffness.dtype)
        mass = jax.vmap(
            lambda vector: self.constitutive.electric_displacement(
                vector, self.material_state
            ),
            in_axes=1,
            out_axes=1,
        )(identity)
        hodge = self.cochain.hodge_metric(1)
        paired_stiffness = _paired_matrix(hodge, stiffness)
        paired_mass = _paired_matrix(hodge, mass)
        problem = eigen_linalg.GeneralizedEigenproblem(
            _verified_dense_operator(
                paired_stiffness,
                "paired Maxwell stiffness",
            ),
            _verified_dense_operator(
                paired_mass,
                "paired Maxwell mass",
                positive_definite=True,
            ),
        )
        solved = eigen_linalg.eigensolve(
            problem,
            policy=eigen_linalg.EigenSolvePolicy(
                eigen_linalg.DenseEigh(),
                count=count,
                which="smallest-algebraic",
            ),
        )
        values = jnp.real(solved.eigenvalues)
        return FrequencyMaxwellEigenResult(
            angular_frequencies=jnp.sqrt(jnp.maximum(values, 0.0)),
            modes=solved.eigenvectors,
            residuals=solved.diagnostics.residual_norms,
            status=solved.status,
            diagnostics=solved.diagnostics,
            result_id=canonical_fingerprint(
                {
                    "kind": "frequency-maxwell-eigensystem",
                    "operator": self.operator_id,
                    "eigen_plan": solved.provenance.plan_id,
                }
            ),
        )


class FrequencyMaxwellAdjointResult(StrictModule):
    solution: Array
    adjoint: Array
    objective: Array
    source_gradient: Array


def frequency_maxwell_adjoint(
    operator: FrequencyMaxwellOperator,
    source: ArrayLike,
    objective: Any,
    /,
) -> FrequencyMaxwellAdjointResult:
    if not callable(objective):
        raise TypeError("objective must be callable.")
    solved = operator.solve(source)
    value, pullback = jax.vjp(
        lambda field: jnp.asarray(objective(field)), solved.electric
    )
    if value.shape != () or jnp.iscomplexobj(value):
        raise ValueError("Frequency Maxwell objective must be a real scalar.")
    cotangent = pullback(jnp.asarray(1.0))[0]
    adjoint = operator.adjoint_solve(cotangent).electric
    return FrequencyMaxwellAdjointResult(
        solved.electric,
        adjoint,
        value,
        adjoint,
    )


def eigenspace_directional_derivative(
    spectrum: eigen_linalg.PreparedSelfAdjointSpectrum,
    selection: eigen_linalg.SpectralSelection,
    perturbation: ArrayLike,
    metric_perturbation: ArrayLike | None = None,
    /,
    *,
    policy: eigen_linalg.SelfAdjointSpectralSubspacePolicy | None = None,
) -> eigen_linalg.SelfAdjointSpectralDerivativeResult:
    """Differentiate an isolated Maxwell eigenspace as a basis-invariant projector."""
    return eigen_linalg.self_adjoint_spectral_projector_derivative(
        spectrum,
        selection,
        perturbation,
        metric_perturbation,
        policy=policy,
    )


__all__ = [
    "FrequencyMaxwellAdjointResult",
    "FrequencyMaxwellEigenResult",
    "FrequencyMaxwellOperator",
    "FrequencyMaxwellSolveResult",
    "eigenspace_directional_derivative",
    "frequency_maxwell_adjoint",
]
