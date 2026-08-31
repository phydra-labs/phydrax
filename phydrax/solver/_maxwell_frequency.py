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
from ._maxwell import (
    AbstractPreparedMaxwellConstitutive,
    CompatibleMaxwellState,
    MaxwellCochainLayout,
    PreparedCompatibleMaxwell,
)
from ._maxwell_sources import MaxwellSourceForcing


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
    layout: MaxwellCochainLayout
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        constitutive: AbstractPreparedMaxwellConstitutive,
        angular_frequency: ArrayLike,
        /,
        *,
        material_state: Any = None,
    ):
        if not isinstance(cochain, CochainDiscretization):
            raise TypeError("Frequency Maxwell requires a CochainDiscretization.")
        if not isinstance(layout, MaxwellCochainLayout):
            raise TypeError("Frequency Maxwell requires a MaxwellCochainLayout.")
        if not isinstance(constitutive, AbstractPreparedMaxwellConstitutive):
            raise TypeError("constitutive must be prepared Maxwell material data.")
        if not constitutive.capabilities.frequency_domain:
            raise ValueError("Constitutive law does not support frequency-domain use.")
        frequency = jnp.asarray(angular_frequency)
        if jnp.iscomplexobj(frequency):
            raise TypeError("angular_frequency must be real.")
        if (
            frequency.shape != ()
            or not jnp.issubdtype(frequency.dtype, jnp.inexact)
            or bool(jnp.any(~jnp.isfinite(frequency)))
            or bool(frequency < 0.0)
        ):
            raise ValueError("angular_frequency must be a finite nonnegative scalar.")
        constitutive.validate_state(material_state)
        self.cochain = cochain
        if constitutive.layout_id != layout.layout_id:
            raise ValueError("Frequency material and Maxwell layout do not match.")
        self.constitutive = constitutive
        self.layout = layout
        self.angular_frequency = frequency
        self.material_state = material_state
        self.operator_id = canonical_fingerprint(
            {
                "kind": "frequency-maxwell-operator",
                "cochain": cochain.prepared_id,
                "layout": layout.layout_id,
                "constitutive": constitutive.prepared_id,
                "angular_frequency": float(np.asarray(frequency)),
            }
        )

    @property
    def size(self) -> int:
        return self.layout.electric_count

    def mv(self, electric: ArrayLike, /) -> Array:
        electric_ = jnp.asarray(electric)
        if electric_.shape != (self.size,):
            raise ValueError("Frequency Maxwell electric field has wrong shape.")
        curl = self.cochain.exterior_derivative(self.layout.electric_degree, electric_)
        magnetic = self.constitutive.magnetic_field(curl, self.material_state)
        curl_curl = self.cochain.codifferential(self.layout.magnetic_degree, magnetic)
        displacement = self.constitutive.electric_displacement(
            electric_, self.material_state
        )
        return curl_curl - self.angular_frequency**2 * displacement

    def defect(
        self, electric: ArrayLike, source: ArrayLike, /
    ) -> MaxwellHarmonicDefectReport:
        electric_, source_ = jnp.asarray(electric), jnp.asarray(source)
        if electric_.shape != (self.size,) or source_.shape != (self.size,):
            raise ValueError("Frequency Maxwell defect vectors have the wrong shape.")
        applied = self.mv(electric_)
        residual = applied - source_
        metric = self.cochain.hodge_metric(self.layout.electric_degree)
        norm = lambda value: jnp.sqrt(
            jnp.maximum(
                jnp.real(jnp.vdot(value, _paired_matrix(metric, value[:, None])[:, 0])),
                0.0,
            )
        )
        absolute = norm(residual)
        denominator = norm(applied) + norm(source_)
        relative = absolute / jnp.maximum(denominator, jnp.finfo(absolute.dtype).tiny)
        zero = jnp.asarray(0.0, dtype=absolute.dtype)
        return MaxwellHarmonicDefectReport(
            absolute,
            relative,
            absolute,
            zero,
            zero,
            zero,
            jnp.asarray(True),
            "exp(-i*omega*t)",
        )

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
            self.layout,
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
        hodge = self.cochain.hodge_metric(self.layout.electric_degree)
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


class MaxwellHarmonicDefectReport(StrictModule):
    absolute_norm: Array
    relative_norm: Array
    electric_defect: Array
    magnetic_defect: Array
    charge_defect: Array
    auxiliary_defect: Array
    eligible: Array
    convention: str = eqx.field(static=True)


class MaxwellHarmonicSource(StrictModule):
    electric_current: Array
    magnetic_current: Array
    convention: str = eqx.field(static=True)

    def __init__(
        self,
        electric_current: ArrayLike,
        magnetic_current: ArrayLike,
        /,
        *,
        convention: str = "exp(-i*omega*t)",
    ):
        if convention != "exp(-i*omega*t)":
            raise ValueError(
                "Maxwell harmonic source convention must be exp(-i*omega*t)."
            )
        self.electric_current = jnp.asarray(electric_current)
        self.magnetic_current = jnp.asarray(magnetic_current)
        self.convention = convention


def _hodge_norm(metric: Array, value: Array, /) -> Array:
    paired = metric * value if metric.ndim == 1 else metric @ value
    return jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(value, paired)), 0.0))


def _tree_norm(tree: Any, /) -> Array:
    leaves = tuple(
        leaf for leaf in jax.tree_util.tree_leaves(tree) if isinstance(leaf, jax.Array)
    )
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.real(jnp.vdot(leaf, leaf)) for leaf in leaves))


def compatible_maxwell_harmonic_defect(
    runtime: PreparedCompatibleMaxwell,
    state_phasor: CompatibleMaxwellState,
    source_phasor: MaxwellHarmonicSource,
    angular_frequency: ArrayLike,
    step_size: ArrayLike,
    /,
) -> MaxwellHarmonicDefectReport:
    """Defect of the complete affine leapfrog map for exp(-i*omega*t)."""

    if not isinstance(runtime, PreparedCompatibleMaxwell):
        raise TypeError("Complete Maxwell harmonic defect requires a prepared runtime.")
    if not runtime.capabilities.linear_time_invariant or runtime.capabilities.nonlinear:
        raise ValueError(
            "Complete harmonic defect requires linear time-invariant dynamics."
        )
    state = runtime._state(state_phasor)
    if not isinstance(source_phasor, MaxwellHarmonicSource):
        raise TypeError("source_phasor must be MaxwellHarmonicSource.")
    if source_phasor.electric_current.shape != (runtime.layout.electric_count,):
        raise ValueError("Harmonic electric source has the wrong retained shape.")
    if source_phasor.magnetic_current.shape != (runtime.layout.magnetic_count,):
        raise ValueError("Harmonic magnetic source has the wrong retained shape.")
    omega, dt = jnp.asarray(angular_frequency), runtime._step_size(step_size)
    if omega.shape != () or jnp.iscomplexobj(omega):
        raise ValueError("Harmonic angular frequency must be a real scalar.")
    phase_half = jnp.exp(-0.5j * omega * dt)
    phase_full = phase_half**2
    base = MaxwellSourceForcing(
        source_phasor.electric_current,
        source_phasor.magnetic_current,
    )
    samples = (
        base,
        MaxwellSourceForcing(
            phase_half * source_phasor.electric_current,
            phase_half * source_phasor.magnetic_current,
        ),
        MaxwellSourceForcing(
            phase_full * source_phasor.electric_current,
            phase_full * source_phasor.magnetic_current,
        ),
    )
    coefficients = (
        None if runtime.pml is None else runtime.pml.bind_coefficients(dt, 0.5 * dt)
    )
    stepped = runtime._step_core(
        jnp.asarray(0.0),
        state,
        dt,
        None,
        cpml_coefficients=coefficients,
        source_samples=samples,
    )
    target_material = jax.tree_util.tree_map(
        lambda value: phase_full * value,
        state.auxiliary.material,
    )
    target_boundary = jax.tree_util.tree_map(
        lambda value: phase_full * value,
        state.auxiliary.boundary,
    )
    d_defect = (
        stepped.primary.electric_displacement
        - phase_full * state.primary.electric_displacement
    )
    b_defect = stepped.primary.magnetic_flux - phase_full * state.primary.magnetic_flux
    q_defect = stepped.primary.charge - phase_full * state.primary.charge
    material_defect = jax.tree_util.tree_map(
        lambda left, right: left - right,
        stepped.auxiliary.material,
        target_material,
    )
    boundary_defect = jax.tree_util.tree_map(
        lambda left, right: left - right,
        stepped.auxiliary.boundary,
        target_boundary,
    )
    electric_norm = _hodge_norm(
        runtime.plan.bridge.cochain.hodge_metric(runtime.layout.electric_degree),
        d_defect,
    )
    magnetic_norm = _hodge_norm(
        runtime.plan.bridge.cochain.hodge_metric(runtime.layout.magnetic_degree),
        b_defect,
    )
    charge_norm = jnp.linalg.norm(q_defect)
    auxiliary_norm = jnp.sqrt(
        _tree_norm(material_defect) ** 2 + _tree_norm(boundary_defect) ** 2
    )
    absolute = jnp.sqrt(
        electric_norm**2 + magnetic_norm**2 + charge_norm**2 + auxiliary_norm**2
    )
    state_scale = (
        _hodge_norm(
            runtime.plan.bridge.cochain.hodge_metric(runtime.layout.electric_degree),
            stepped.primary.electric_displacement,
        )
        + _hodge_norm(
            runtime.plan.bridge.cochain.hodge_metric(runtime.layout.magnetic_degree),
            stepped.primary.magnetic_flux,
        )
        + jnp.linalg.norm(stepped.primary.charge)
        + _tree_norm(stepped.auxiliary)
        + _hodge_norm(
            runtime.plan.bridge.cochain.hodge_metric(runtime.layout.electric_degree),
            state.primary.electric_displacement,
        )
        + _hodge_norm(
            runtime.plan.bridge.cochain.hodge_metric(runtime.layout.magnetic_degree),
            state.primary.magnetic_flux,
        )
    )
    relative = absolute / jnp.maximum(state_scale, jnp.finfo(absolute.dtype).tiny)
    return MaxwellHarmonicDefectReport(
        absolute,
        relative,
        electric_norm,
        magnetic_norm,
        charge_norm,
        auxiliary_norm,
        jnp.asarray(True),
        source_phasor.convention,
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
    "MaxwellHarmonicDefectReport",
    "MaxwellHarmonicSource",
    "compatible_maxwell_harmonic_defect",
    "eigenspace_directional_derivative",
    "frequency_maxwell_adjoint",
]
