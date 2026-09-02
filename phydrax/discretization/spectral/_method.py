#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dealias import (
    AbstractDealiasingPlan,
    NoDealiasingPlan,
    PreparedDealiasingPlan,
)
from ._space import TensorSpectralDiscretization
from ._spherical import SphericalSpectralDiscretization


SpectralDifferentiabilityPolicy: TypeAlias = Literal[
    "smooth_discrete",
    "branchwise",
    "smooth_surrogate",
    "unsupported",
]


class PseudospectralMethodPlan(StrictModule, NonTrainableState):
    """Nonlinear realization, differentiability, and diagnostics policy."""

    dealiasing: AbstractDealiasingPlan | None
    differentiability: SpectralDifferentiabilityPolicy = eqx.field(static=True)
    diagnostics: bool = eqx.field(static=True)
    real_projection_tolerance: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        dealiasing: AbstractDealiasingPlan | None = None,
        differentiability: SpectralDifferentiabilityPolicy = "smooth_discrete",
        diagnostics: bool = False,
        real_projection_tolerance: float = 1e-10,
    ):
        if dealiasing is not None and not isinstance(dealiasing, AbstractDealiasingPlan):
            raise TypeError("dealiasing must be an AbstractDealiasingPlan or None.")
        if differentiability not in (
            "smooth_discrete",
            "branchwise",
            "smooth_surrogate",
            "unsupported",
        ):
            raise ValueError("Unknown spectral differentiability policy.")
        tolerance = float(real_projection_tolerance)
        if not jnp.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("real_projection_tolerance must be finite and non-negative.")
        self.dealiasing = dealiasing
        self.differentiability = differentiability
        self.diagnostics = bool(diagnostics)
        self.real_projection_tolerance = tolerance
        self.method_id = canonical_fingerprint(
            {
                "kind": "pseudospectral-method",
                "dealiasing": None if dealiasing is None else dealiasing.plan_id,
                "differentiability": differentiability,
                "diagnostics": bool(diagnostics),
                "real_projection_tolerance": tolerance,
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization | SphericalSpectralDiscretization,
        /,
        *,
        required_polynomial_degree: int | None,
        nonlinear: bool,
    ) -> "PreparedPseudospectralMethod":
        if not isinstance(
            discretization,
            (TensorSpectralDiscretization, SphericalSpectralDiscretization),
        ):
            raise TypeError("discretization must be a prepared spectral space.")
        if nonlinear and self.dealiasing is None:
            raise ValueError(
                "Nonlinear pseudospectral compilation requires an explicit "
                "dealiasing policy."
            )
        dealiasing = NoDealiasingPlan() if self.dealiasing is None else self.dealiasing
        prepared = dealiasing.prepare(
            discretization,
            required_polynomial_degree=required_polynomial_degree,
        )
        return PreparedPseudospectralMethod(
            self,
            discretization,
            prepared,
            nonlinear=nonlinear,
        )


class PreparedPseudospectralMethod(StrictModule, NonTrainableState):
    """Prepared nonlinear transform schedule for one spectral discretization."""

    plan: PseudospectralMethodPlan
    discretization: TensorSpectralDiscretization | SphericalSpectralDiscretization
    dealiasing: PreparedDealiasingPlan
    nonlinear: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PseudospectralMethodPlan,
        discretization: TensorSpectralDiscretization | SphericalSpectralDiscretization,
        dealiasing: PreparedDealiasingPlan,
        /,
        *,
        nonlinear: bool,
    ):
        if not isinstance(plan, PseudospectralMethodPlan):
            raise TypeError("plan must be a PseudospectralMethodPlan.")
        if not isinstance(
            discretization,
            (TensorSpectralDiscretization, SphericalSpectralDiscretization),
        ):
            raise TypeError("discretization must be a prepared spectral space.")
        if not isinstance(dealiasing, PreparedDealiasingPlan):
            raise TypeError("dealiasing must be a PreparedDealiasingPlan.")
        self.plan = plan
        self.discretization = discretization
        self.dealiasing = dealiasing
        self.nonlinear = bool(nonlinear)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-pseudospectral-method",
                "plan": plan.method_id,
                "discretization": discretization.prepared_id,
                "dealiasing": dealiasing.prepared_id,
                "nonlinear": self.nonlinear,
            }
        )

    def nonlinear_action(
        self,
        coefficients: ArrayLike,
        function: Callable[[Array], ArrayLike],
        /,
    ) -> Array:
        """Evaluate one declared nonlinearity on the prepared physical grid."""
        if not self.nonlinear:
            raise ValueError(
                "This method was not prepared for nonlinear spectral evaluation."
            )
        if not callable(function):
            raise TypeError("function must be callable.")
        physical = self.dealiasing.reconstruct(coefficients)
        return self.dealiasing.project(function(physical))


class SpectralResidualDiagnostics(StrictModule):
    """Conservation, modal-tail, reality, admissibility, and entropy evidence."""

    total_integral: Array
    semidiscrete_integral_rate: Array
    conservation_defect: Array
    low_mode_energy: Array
    tail_mode_energy: Array
    tail_energy_ratio: Array
    imaginary_leakage: Array
    admissible: Array
    total_entropy: Array | None
    semidiscrete_entropy_rate: Array | None
    precision_evidence: PrecisionEvidenceEnvelope
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        total_integral: ArrayLike,
        semidiscrete_integral_rate: ArrayLike,
        conservation_defect: ArrayLike,
        low_mode_energy: ArrayLike,
        tail_mode_energy: ArrayLike,
        tail_energy_ratio: ArrayLike,
        imaginary_leakage: ArrayLike,
        admissible: ArrayLike,
        total_entropy: ArrayLike | None,
        semidiscrete_entropy_rate: ArrayLike | None,
        precision_evidence: PrecisionEvidenceEnvelope,
        method_id: str,
    ):
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be a PrecisionEvidenceEnvelope.")
        identifier = str(method_id)
        if not identifier:
            raise ValueError("method_id must be non-empty.")
        self.total_integral = jnp.asarray(total_integral)
        self.semidiscrete_integral_rate = jnp.asarray(semidiscrete_integral_rate)
        self.conservation_defect = jnp.asarray(conservation_defect)
        self.low_mode_energy = jnp.asarray(low_mode_energy)
        self.tail_mode_energy = jnp.asarray(tail_mode_energy)
        self.tail_energy_ratio = jnp.asarray(tail_energy_ratio)
        self.imaginary_leakage = jnp.asarray(imaginary_leakage)
        self.admissible = jnp.asarray(admissible, dtype=bool)
        self.total_entropy = None if total_entropy is None else jnp.asarray(total_entropy)
        self.semidiscrete_entropy_rate = (
            None
            if semidiscrete_entropy_rate is None
            else jnp.asarray(semidiscrete_entropy_rate)
        )
        self.precision_evidence = precision_evidence
        self.method_id = identifier


__all__ = [
    "PreparedPseudospectralMethod",
    "PseudospectralMethodPlan",
    "SpectralDifferentiabilityPolicy",
    "SpectralResidualDiagnostics",
]
