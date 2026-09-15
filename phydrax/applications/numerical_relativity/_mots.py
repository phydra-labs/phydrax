#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Null expansions, marginally outer trapped surfaces, and stability evidence."""

from __future__ import annotations

from enum import IntEnum
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein
import phydrax.linalg as la
from phydrax.nonlinear import (
    root,
    NonlinearSystemProblem,
    NonlinearTermination,
    NewtonKrylov,
)

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._spacetime_conventions import RelativityConvention
from ._surfaces import SphericalSpectralSurface, SphericalSurfacePlan


_METRIC_SOLVE = la.SmallLinearSolvePlan(3)


class MOTSStatus(IntEnum):
    """Scientific status of a marginal-surface solve, never a horizon label."""

    SUCCESS = 0
    NONFINITE = 1
    NOT_CONVERGED = 2
    NONPHYSICAL_SURFACE = 3
    INVALID_STABILITY_EVIDENCE = 4


class NullExpansionEvidence(StrictModule):
    outgoing: Array
    ingoing: Array
    mean_curvature: Array
    trace_extrinsic_curvature: Array
    normal_extrinsic_curvature: Array
    normal_norm_error: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    convention_id: str = eqx.field(static=True)


class MOTSStabilityEvidence(StrictModule):
    """Finite-dimensional linearized outward-expansion operator evidence."""

    operator: Array
    symmetric_operator: Array
    eigenvalues: Array
    principal_eigenvalue: Array
    antisymmetry_norm: Array
    finite: Array
    self_adjoint: Array
    stable: Array
    derivative_valid: Array


class MOTSSolveResult(StrictModule):
    """A solved MOTS candidate.  This type deliberately is not a horizon result."""

    surface: SphericalSpectralSurface
    outgoing_expansion: Array
    residual_norm: Array
    iterations: Array
    stability: MOTSStabilityEvidence
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class MOTSSolvePlan(StrictModule, NonTrainableState):
    """Fixed-capacity nonlinear MOTS solve in sampled radial coordinates."""

    surface_plan: SphericalSurfacePlan
    termination: NonlinearTermination
    method: NewtonKrylov
    minimum_radius: float = eqx.field(static=True)
    stability_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_plan: SphericalSurfacePlan,
        /,
        *,
        residual_tolerance: float = 1.0e-9,
        maximum_steps: int = 40,
        minimum_radius: float = 1.0e-10,
        stability_tolerance: float = 1.0e-7,
    ):
        if not isinstance(surface_plan, SphericalSurfacePlan):
            raise TypeError("surface_plan must be a SphericalSurfacePlan.")
        residual_tolerance_ = float(residual_tolerance)
        minimum_radius_ = float(minimum_radius)
        stability_tolerance_ = float(stability_tolerance)
        maximum_steps_ = int(maximum_steps)
        if (
            residual_tolerance_ <= 0.0
            or maximum_steps_ < 1
            or minimum_radius_ <= 0.0
            or stability_tolerance_ <= 0.0
        ):
            raise ValueError("MOTS solve tolerances and capacities must be positive.")
        self.surface_plan = surface_plan
        self.termination = NonlinearTermination(
            absolute_residual=residual_tolerance_,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=maximum_steps_,
        )
        self.method = NewtonKrylov()
        self.minimum_radius = minimum_radius_
        self.stability_tolerance = stability_tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-mots-solve",
                "surface": surface_plan.plan_id,
                "residual_tolerance": residual_tolerance_,
                "maximum_steps": maximum_steps_,
                "minimum_radius": minimum_radius_,
                "stability_tolerance": stability_tolerance_,
            }
        )

    def solve(
        self,
        initial_surface: SphericalSpectralSurface,
        expansion_operator: Callable[[SphericalSpectralSurface], ArrayLike],
        /,
    ) -> MOTSSolveResult:
        """Solve ``Theta_out[r] = 0`` without assigning apparent-horizon status.

        The fixed-iteration root path is JIT compatible, while its accepted state
        does not claim implicit solution-map derivatives.  Stability-operator
        derivatives remain separately available in ``result.stability``.
        """
        initial_radius = self.surface_plan.radius(initial_surface)
        center = initial_surface.center
        sample_shape = self.surface_plan.sample_shape

        def sampled_residual(flat_radius: Array, _arguments: object) -> Array:
            surface = self.surface_plan.from_samples(
                flat_radius.reshape(sample_shape), center=center
            )
            expansion = jnp.asarray(expansion_operator(surface))
            if expansion.shape != sample_shape:
                raise ValueError("MOTS expansion operator must return one surface sample.")
            return jnp.real(expansion).reshape((-1,))

        nonlinear = root(
            NonlinearSystemProblem(
                sampled_residual,
                problem_id=f"mots:{self.plan_id}",
            ),
            initial_radius.reshape((-1,)),
            method=self.method,
            termination=self.termination,
        )
        radius = nonlinear.state.reshape(sample_shape)
        surface = self.surface_plan.from_samples(radius, center=center)
        outgoing = jnp.asarray(expansion_operator(surface))
        residual_norm = jnp.max(jnp.abs(outgoing))
        finite = (
            jnp.all(jnp.isfinite(radius))
            & jnp.all(jnp.isfinite(outgoing))
            & jnp.isfinite(residual_norm)
        )
        converged = (
            nonlinear.successful
            & finite
            & (residual_norm <= self.termination.absolute_residual)
        )
        physically_valid = finite & jnp.all(radius > self.minimum_radius)
        stability = self.stability(surface, expansion_operator)
        qualified = converged & physically_valid & stability.finite
        derivative_valid = jnp.asarray(False)
        status = jnp.select(
            (
                ~finite,
                ~converged,
                ~physically_valid,
                ~stability.finite,
            ),
            (
                int(MOTSStatus.NONFINITE),
                int(MOTSStatus.NOT_CONVERGED),
                int(MOTSStatus.NONPHYSICAL_SURFACE),
                int(MOTSStatus.INVALID_STABILITY_EVIDENCE),
            ),
            default=int(MOTSStatus.SUCCESS),
        ).astype(jnp.int32)
        return MOTSSolveResult(
            surface,
            outgoing,
            residual_norm,
            nonlinear.diagnostics.iterations,
            stability,
            finite,
            converged,
            physically_valid,
            qualified,
            derivative_valid,
            status,
            self.plan_id,
        )

    def stability(
        self,
        surface: SphericalSpectralSurface,
        expansion_operator: Callable[[SphericalSpectralSurface], ArrayLike],
        /,
    ) -> MOTSStabilityEvidence:
        """Linearize expansion under fixed-grid outward radial deformations.

        The self-adjoint part supplies a conservative principal stability value.
        ``self_adjoint`` separately records whether the sampled operator itself
        supports interpreting that value as the MOTS principal eigenvalue.
        """
        radius = self.surface_plan.radius(surface)
        center = surface.center
        sample_shape = self.surface_plan.sample_shape

        def expansion(flat_radius: Array) -> Array:
            candidate = self.surface_plan.from_samples(
                flat_radius.reshape(sample_shape), center=center
            )
            values = jnp.asarray(expansion_operator(candidate))
            if values.shape != sample_shape:
                raise ValueError("MOTS expansion operator must return one surface sample.")
            return jnp.real(values).reshape((-1,))

        operator = jax.jacfwd(expansion)(radius.reshape((-1,)))
        weights = self.surface_plan.solid_angle_weights.reshape((-1,))
        root_weights = jnp.sqrt(weights)
        weighted_operator = (
            root_weights[:, None] * operator / root_weights[None, :]
        )
        symmetric = 0.5 * (weighted_operator + weighted_operator.T)
        antisymmetry_norm = jnp.max(
            jnp.abs(weighted_operator - weighted_operator.T)
        )
        spectrum = la.HermitianSpectrum(
            symmetric, tolerance=self.stability_tolerance
        )
        scale = jnp.maximum(jnp.max(jnp.abs(symmetric)), 1.0)
        self_adjoint = antisymmetry_norm <= self.stability_tolerance * scale
        finite = (
            spectrum.valid
            & jnp.all(jnp.isfinite(operator))
            & jnp.isfinite(antisymmetry_norm)
        )
        represented_size = self.surface_plan.bandlimit**2
        magnitude_order = jnp.argsort(jnp.abs(spectrum.eigenvalues))[::-1]
        represented = jnp.arange(spectrum.eigenvalues.size) < represented_size
        represented_eigenvalues = jnp.where(
            represented,
            spectrum.eigenvalues[magnitude_order],
            jnp.inf,
        )
        principal = jnp.min(represented_eigenvalues)
        separated_from_sampling_nullspace = (
            jnp.min(jnp.abs(represented_eigenvalues))
            > self.stability_tolerance * scale
        )
        stable = (
            finite
            & self_adjoint
            & separated_from_sampling_nullspace
            & (principal > self.stability_tolerance)
        )
        return MOTSStabilityEvidence(
            operator,
            symmetric,
            spectrum.eigenvalues,
            principal,
            antisymmetry_norm,
            finite,
            self_adjoint,
            stable,
            finite & self_adjoint & separated_from_sampling_nullspace,
        )


def null_expansions(
    spatial_metric: ArrayLike,
    extrinsic_curvature: ArrayLike,
    outward_unit_normal: ArrayLike,
    surface_divergence: ArrayLike,
    /,
    *,
    convention: RelativityConvention | None = None,
) -> NullExpansionEvidence:
    """Evaluate future outgoing/ingoing expansions under an explicit convention.

    With ``ell = n + s`` and ``k = n - s``, canonical
    ``K=-L_n gamma/2`` gives ``Theta_ell = D_i s^i + K_ss - K`` and
    ``Theta_k = -D_i s^i + K_ss - K``.  The shared convention sign maps this
    formula to the supplied ``K``.  No zero is promoted beyond a MOTS candidate.
    """
    convention_ = RelativityConvention() if convention is None else convention
    if not isinstance(convention_, RelativityConvention):
        raise TypeError("convention must be a RelativityConvention or None.")
    metric = jnp.asarray(spatial_metric)
    curvature = jnp.asarray(extrinsic_curvature, dtype=metric.dtype)
    normal = jnp.asarray(outward_unit_normal, dtype=metric.dtype)
    divergence = jnp.asarray(surface_divergence, dtype=metric.dtype)
    if metric.shape[-2:] != (3, 3) or curvature.shape != metric.shape:
        raise ValueError("Spatial metric and extrinsic curvature must end in (3, 3).")
    if normal.shape != metric.shape[:-1] or divergence.shape != metric.shape[:-2]:
        raise ValueError("Normal or surface-divergence shapes are incompatible.")
    inverse_result = la.inverse_small_linear(_METRIC_SOLVE, metric)
    inverse_metric = inverse_result.value
    trace_curvature = ein.contract("...ij,...ij->...", inverse_metric, curvature)
    normal_curvature = ein.contract(
        "...i,...ij,...j->...", normal, curvature, normal
    )
    extrinsic_expansion = -float(convention_.extrinsic_curvature_sign) * (
        normal_curvature - trace_curvature
    )
    outgoing = divergence + extrinsic_expansion
    ingoing = -divergence + extrinsic_expansion
    normal_norm = ein.contract("...i,...ij,...j->...", normal, metric, normal)
    normal_error = jnp.abs(normal_norm - 1.0)
    finite = (
        jnp.all(jnp.isfinite(metric))
        & jnp.all(jnp.isfinite(curvature))
        & jnp.all(jnp.isfinite(normal))
        & jnp.all(jnp.isfinite(outgoing))
        & jnp.all(jnp.isfinite(ingoing))
    )
    metric_valid = jnp.all(inverse_result.successful)
    physically_valid = finite & metric_valid & jnp.all(normal_error <= 1.0e-6)
    return NullExpansionEvidence(
        outgoing,
        ingoing,
        divergence,
        trace_curvature,
        normal_curvature,
        normal_error,
        finite,
        physically_valid,
        physically_valid,
        finite & metric_valid,
        convention_.convention_id,
    )


def schwarzschild_isotropic_outgoing_expansion(
    isotropic_radius: ArrayLike,
    mass: ArrayLike,
    /,
) -> Array:
    """Analytic outgoing expansion on a time-symmetric isotropic Schwarzschild sphere."""
    radius = jnp.asarray(isotropic_radius)
    mass_ = jnp.asarray(mass, dtype=radius.dtype)
    conformal_factor = 1.0 + mass_ / (2.0 * radius)
    return (
        2.0
        / (radius * conformal_factor**2)
        * (1.0 - mass_ / (2.0 * radius))
        / (1.0 + mass_ / (2.0 * radius))
    )


__all__ = [
    "MOTSSolvePlan",
    "MOTSSolveResult",
    "MOTSStabilityEvidence",
    "MOTSStatus",
    "NullExpansionEvidence",
    "null_expansions",
    "schwarzschild_isotropic_outgoing_expansion",
]
