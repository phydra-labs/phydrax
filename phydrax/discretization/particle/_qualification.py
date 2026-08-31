#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


ParticleDerivativeTier: TypeAlias = Literal["A", "B", "C", "D"]


class ParticleMethodMaturity(StrEnum):
    EXPERIMENTAL = "experimental"
    QUALIFIED = "qualified"
    PRODUCTION = "production"
    CERTIFIED = "certified"


class ParticleQualificationClaim(StrEnum):
    FINITE_EXECUTION = "finite-execution"
    MASS_CONSERVATIVE = "mass-conservative"
    LINEAR_MOMENTUM_CONSERVATIVE = "linear-momentum-conservative"
    ANGULAR_MOMENTUM_CONSERVATIVE = "angular-momentum-conservative"
    ENERGY_CONSERVATIVE = "energy-conservative"
    ENERGY_DISSIPATIVE = "energy-dissipative"
    CONTACT_HISTORY_CONTINUITY = "contact-history-continuity"
    FRICTION_CONE = "friction-cone"
    RESTITUTION_ACCURACY = "restitution-accuracy"
    DENSITY_CONSTRAINT = "density-constraint"
    DIVERGENCE_CONSTRAINT = "divergence-constraint"
    WALL_ACTION_REACTION = "wall-action-reaction"
    FREE_SURFACE_PRESSURE = "free-surface-pressure"
    MULTIPHASE_PRESSURE_BALANCE = "multiphase-pressure-balance"
    DIFFERENTIABLE_STATE = "differentiable-state"
    RESTART_EQUIVALENT = "restart-equivalent"
    DETERMINISTIC = "deterministic"
    DISTRIBUTED = "distributed"


class ParticleBenchmarkIdentity(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    benchmark_id: str = eqx.field(static=True)

    def __init__(self, name: str, configuration_id: str, source_id: str, /):
        values = tuple(str(value) for value in (name, configuration_id, source_id))
        if any(not value for value in values):
            raise ValueError("Particle benchmark identity fields must be non-empty.")
        self.name, self.configuration_id, self.source_id = values
        self.benchmark_id = canonical_fingerprint(
            {
                "kind": "particle-benchmark-identity",
                "name": values[0],
                "configuration": values[1],
                "source": values[2],
            }
        )


class ParticleClaimEvidence(StrictModule, NonTrainableState):
    claim: ParticleQualificationClaim = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    satisfied: Array

    def __init__(
        self,
        claim: ParticleQualificationClaim,
        evidence_id: str,
        satisfied: ArrayLike,
        /,
    ):
        if not isinstance(claim, ParticleQualificationClaim):
            raise TypeError("claim must be a ParticleQualificationClaim.")
        identifier = str(evidence_id)
        if not identifier:
            raise ValueError("Particle claim evidence ID must be non-empty.")
        self.claim = claim
        self.evidence_id = identifier
        self.satisfied = jnp.asarray(satisfied, dtype=bool)


class ParticleDerivativeQualification(StrictModule, NonTrainableState):
    tier: ParticleDerivativeTier = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    jvp_error: Array
    vjp_error: Array
    tangent_residual: Array
    adjoint_residual: Array


class ParticleConstraintResiduals(StrictModule):
    relative_density_linf: Array
    relative_density_l2: Array
    relative_divergence_linf: Array
    relative_divergence_l2: Array
    pressure_complementarity: Array
    wall_constraint: Array
    free_surface_dirichlet: Array


class AbstractParticleQualificationProfile(StrictModule, NonTrainableState):
    profile_id: AbstractAttribute[str]


class ParticleQualificationProfile(AbstractParticleQualificationProfile):
    density_linf_tolerance: float = eqx.field(static=True)
    density_l2_tolerance: float = eqx.field(static=True)
    divergence_linf_tolerance: float = eqx.field(static=True)
    divergence_l2_tolerance: float = eqx.field(static=True)
    complementarity_tolerance: float = eqx.field(static=True)
    boundary_tolerance: float = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        density_linf_tolerance: float = 1e-3,
        density_l2_tolerance: float = 1e-3,
        divergence_linf_tolerance: float = 1e-3,
        divergence_l2_tolerance: float = 1e-3,
        complementarity_tolerance: float = 1e-6,
        boundary_tolerance: float = 1e-6,
    ):
        values = tuple(
            float(value)
            for value in (
                density_linf_tolerance,
                density_l2_tolerance,
                divergence_linf_tolerance,
                divergence_l2_tolerance,
                complementarity_tolerance,
                boundary_tolerance,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "Particle qualification tolerances must be finite and positive."
            )
        (
            self.density_linf_tolerance,
            self.density_l2_tolerance,
            self.divergence_linf_tolerance,
            self.divergence_l2_tolerance,
            self.complementarity_tolerance,
            self.boundary_tolerance,
        ) = values
        self.profile_id = canonical_fingerprint(
            {
                "kind": "particle-qualification-profile",
                "tolerances": list(values),
            }
        )

    def constraints_satisfied(self, residuals: ParticleConstraintResiduals, /) -> Array:
        return (
            (residuals.relative_density_linf <= self.density_linf_tolerance)
            & (residuals.relative_density_l2 <= self.density_l2_tolerance)
            & (residuals.relative_divergence_linf <= self.divergence_linf_tolerance)
            & (residuals.relative_divergence_l2 <= self.divergence_l2_tolerance)
            & (residuals.pressure_complementarity <= self.complementarity_tolerance)
            & (residuals.wall_constraint <= self.boundary_tolerance)
            & (residuals.free_surface_dirichlet <= self.boundary_tolerance)
        )


class ParticleQualificationResult(StrictModule, NonTrainableState):
    maturity: ParticleMethodMaturity = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    evidence: tuple[ParticleClaimEvidence, ...]
    execution_successful: Array
    numerical_constraints_satisfied: Array
    qualification_claims_satisfied: Array
    production_gate_satisfied: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        maturity: ParticleMethodMaturity,
        profile: AbstractParticleQualificationProfile,
        evidence: tuple[ParticleClaimEvidence, ...],
        execution_successful: ArrayLike,
        numerical_constraints_satisfied: ArrayLike,
        /,
    ):
        if not isinstance(maturity, ParticleMethodMaturity):
            raise TypeError("maturity must be a ParticleMethodMaturity.")
        if not isinstance(profile, AbstractParticleQualificationProfile):
            raise TypeError("profile must be an AbstractParticleQualificationProfile.")
        if any(not isinstance(value, ParticleClaimEvidence) for value in evidence):
            raise TypeError("evidence must contain ParticleClaimEvidence values.")
        claim_satisfied = (
            jnp.all(jnp.stack(tuple(value.satisfied for value in evidence)))
            if evidence
            else jnp.asarray(maturity is ParticleMethodMaturity.EXPERIMENTAL)
        )
        execution = jnp.asarray(execution_successful, dtype=bool)
        constraints = jnp.asarray(numerical_constraints_satisfied, dtype=bool)
        production = (
            execution
            & constraints
            & claim_satisfied
            & jnp.asarray(
                maturity
                in (
                    ParticleMethodMaturity.PRODUCTION,
                    ParticleMethodMaturity.CERTIFIED,
                )
            )
        )
        self.maturity = maturity
        self.profile_id = profile.profile_id
        self.evidence = evidence
        self.execution_successful = execution
        self.numerical_constraints_satisfied = constraints
        self.qualification_claims_satisfied = claim_satisfied
        self.production_gate_satisfied = production
        self.result_id = canonical_fingerprint(
            {
                "kind": "particle-qualification-result",
                "maturity": maturity.value,
                "profile": profile.profile_id,
                "evidence": [
                    {"claim": value.claim.value, "id": value.evidence_id}
                    for value in evidence
                ],
            }
        )


def particle_constraint_residuals(
    density: ArrayLike,
    reference_density: ArrayLike,
    volumes: ArrayLike,
    /,
    *,
    density_rate: ArrayLike | None = None,
    step_size: ArrayLike = 1.0,
    pressure: ArrayLike | None = None,
    atmospheric_pressure: ArrayLike = 0.0,
    wall_constraint: ArrayLike = 0.0,
    free_surface_dirichlet: ArrayLike = 0.0,
    active_mask: ArrayLike | None = None,
) -> ParticleConstraintResiduals:
    density_ = jnp.asarray(density)
    reference = jnp.asarray(reference_density, dtype=density_.dtype)
    volume = jnp.asarray(volumes, dtype=density_.dtype)
    active = (
        jnp.ones(density_.shape, dtype=bool)
        if active_mask is None
        else jnp.asarray(active_mask, bool)
    )
    relative = jnp.where(active, (density_ - reference) / reference, 0.0)
    weight = jnp.where(active, volume, 0.0)
    weight_sum = jnp.maximum(compensated_sum(weight), jnp.finfo(density_.dtype).tiny)
    density_l2 = jnp.sqrt(compensated_sum(weight * relative**2) / weight_sum)
    if density_rate is None:
        divergence_relative = jnp.zeros_like(relative)
    else:
        divergence_relative = jnp.where(
            active,
            jnp.asarray(step_size, density_.dtype)
            * jnp.maximum(jnp.asarray(density_rate), 0.0)
            / reference,
            0.0,
        )
    divergence_l2 = jnp.sqrt(
        compensated_sum(weight * divergence_relative**2) / weight_sum
    )
    if pressure is None:
        complementarity = jnp.zeros((), dtype=density_.dtype)
    else:
        pressure_gap = jnp.asarray(pressure) - jnp.asarray(
            atmospheric_pressure, density_.dtype
        )
        compression = jnp.maximum(relative, 0.0)
        complementarity = jnp.max(
            jnp.where(
                active,
                jnp.abs(jnp.minimum(pressure_gap, 0.0))
                + jnp.abs(pressure_gap * compression),
                0.0,
            )
        )
    return ParticleConstraintResiduals(
        jnp.max(jnp.abs(relative)),
        density_l2,
        jnp.max(jnp.abs(divergence_relative)),
        divergence_l2,
        complementarity,
        jnp.asarray(wall_constraint, density_.dtype),
        jnp.asarray(free_surface_dirichlet, density_.dtype),
    )


__all__ = [
    "AbstractParticleQualificationProfile",
    "ParticleBenchmarkIdentity",
    "ParticleClaimEvidence",
    "ParticleConstraintResiduals",
    "ParticleDerivativeQualification",
    "ParticleDerivativeTier",
    "ParticleMethodMaturity",
    "ParticleQualificationClaim",
    "ParticleQualificationProfile",
    "ParticleQualificationResult",
    "particle_constraint_residuals",
]
