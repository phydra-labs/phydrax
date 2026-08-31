#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleMethodMaturity


class AtomisticDynamicsQualificationClaim(StrEnum):
    FINITE_EXECUTION = "finite-execution"
    CONSERVATIVE_FORCE = "conservative-force"
    RIGID_MOTION_EQUIVARIANT = "rigid-motion-equivariant"
    PERMUTATION_EQUIVARIANT = "permutation-equivariant"
    PERIODIC_IMAGE_CORRECT = "periodic-image-correct"
    NVE_INVARIANT = "nve-invariant"
    CANONICAL_SAMPLING = "canonical-sampling"
    CONSTRAINT_SATISFACTION = "constraint-satisfaction"
    ELECTROSTATIC_ACCURACY = "electrostatic-accuracy"
    STRESS_ACCURACY = "stress-accuracy"
    RESTART_EQUIVALENT = "restart-equivalent"
    REPLAY_EQUIVALENT = "replay-equivalent"
    DETERMINISTIC_REDUCTION = "deterministic-reduction"
    DIFFERENTIABLE_ROLLOUT = "differentiable-rollout"
    CROSS_ENGINE_PARITY = "cross-engine-parity"


class AtomisticDynamicsClaimEvidence(StrictModule, NonTrainableState):
    claim: AtomisticDynamicsQualificationClaim = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    satisfied: Array
    residual: Array

    def __init__(
        self,
        claim: AtomisticDynamicsQualificationClaim,
        evidence_id: str,
        satisfied: ArrayLike,
        /,
        *,
        residual: ArrayLike = 0.0,
    ):
        if not isinstance(claim, AtomisticDynamicsQualificationClaim):
            raise TypeError("claim must be AtomisticDynamicsQualificationClaim.")
        identifier = str(evidence_id)
        if not identifier:
            raise ValueError("evidence_id must be non-empty.")
        self.claim = claim
        self.evidence_id = identifier
        self.satisfied = jnp.asarray(satisfied, dtype=bool).reshape(())
        self.residual = jnp.asarray(residual).reshape(())


class AtomisticDynamicsQualificationProfile(StrictModule, NonTrainableState):
    energy_drift_tolerance: float = eqx.field(static=True)
    force_gradient_tolerance: float = eqx.field(static=True)
    momentum_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    ensemble_tolerance: float = eqx.field(static=True)
    stress_tolerance: float = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        energy_drift_tolerance: float = 1.0e-4,
        force_gradient_tolerance: float = 1.0e-6,
        momentum_tolerance: float = 1.0e-10,
        constraint_tolerance: float = 1.0e-8,
        ensemble_tolerance: float = 5.0e-2,
        stress_tolerance: float = 1.0e-5,
    ):
        values = tuple(
            float(value)
            for value in (
                energy_drift_tolerance,
                force_gradient_tolerance,
                momentum_tolerance,
                constraint_tolerance,
                ensemble_tolerance,
                stress_tolerance,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Qualification tolerances must be finite and positive.")
        (
            self.energy_drift_tolerance,
            self.force_gradient_tolerance,
            self.momentum_tolerance,
            self.constraint_tolerance,
            self.ensemble_tolerance,
            self.stress_tolerance,
        ) = values
        self.profile_id = canonical_fingerprint(
            {
                "kind": "atomistic-dynamics-qualification-profile",
                "tolerances": list(values),
            }
        )


class AtomisticDynamicsQualificationResult(StrictModule, NonTrainableState):
    maturity: ParticleMethodMaturity = eqx.field(static=True)
    evidence: tuple[AtomisticDynamicsClaimEvidence, ...]
    execution_successful: Array
    claims_satisfied: Array
    production_gate_satisfied: Array
    profile_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        maturity: ParticleMethodMaturity,
        profile: AtomisticDynamicsQualificationProfile,
        evidence: tuple[AtomisticDynamicsClaimEvidence, ...],
        execution_successful: ArrayLike,
        /,
    ):
        if not isinstance(maturity, ParticleMethodMaturity):
            raise TypeError("maturity must be ParticleMethodMaturity.")
        if not isinstance(profile, AtomisticDynamicsQualificationProfile):
            raise TypeError("profile must be AtomisticDynamicsQualificationProfile.")
        if any(
            not isinstance(value, AtomisticDynamicsClaimEvidence) for value in evidence
        ):
            raise TypeError("evidence must contain AtomisticDynamicsClaimEvidence.")
        execution = jnp.asarray(execution_successful, dtype=bool).reshape(())
        satisfied = (
            jnp.all(jnp.stack(tuple(value.satisfied for value in evidence)))
            if evidence
            else jnp.asarray(maturity is ParticleMethodMaturity.EXPERIMENTAL)
        )
        production = (
            execution
            & satisfied
            & jnp.asarray(
                maturity
                in (ParticleMethodMaturity.PRODUCTION, ParticleMethodMaturity.CERTIFIED)
            )
        )
        self.maturity = maturity
        self.evidence = evidence
        self.execution_successful = execution
        self.claims_satisfied = satisfied
        self.production_gate_satisfied = production
        self.profile_id = profile.profile_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "atomistic-dynamics-qualification-result",
                "maturity": maturity.value,
                "profile": profile.profile_id,
                "evidence": [
                    {"claim": value.claim.value, "id": value.evidence_id}
                    for value in evidence
                ],
            }
        )


__all__ = [
    "AtomisticDynamicsClaimEvidence",
    "AtomisticDynamicsQualificationClaim",
    "AtomisticDynamicsQualificationProfile",
    "AtomisticDynamicsQualificationResult",
]
