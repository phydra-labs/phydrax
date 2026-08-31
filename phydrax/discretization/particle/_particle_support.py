#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ParticlePhysicsSupportStatus(StrEnum):
    EXPERIMENTAL = "experimental"
    QUALIFIED = "qualified"
    PRODUCTION = "production"
    UNSUPPORTED = "unsupported"


class ParticlePhysicsSupportClaim(StrictModule, NonTrainableState):
    mechanics: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    internal_geometry: str = eqx.field(static=True)
    thermodynamics: str = eqx.field(static=True)
    transport: str = eqx.field(static=True)
    reaction: str = eqx.field(static=True)
    phase_change: str = eqx.field(static=True)
    morphology: str = eqx.field(static=True)
    surface_exchange: str = eqx.field(static=True)
    continuum_exchange: str = eqx.field(static=True)
    coupling_schedule: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    sensitivity: str = eqx.field(static=True)
    status: ParticlePhysicsSupportStatus = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    claim_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        mechanics: str = "none",
        geometry: str = "none",
        internal_geometry: str = "none",
        thermodynamics: str = "none",
        transport: str = "none",
        reaction: str = "none",
        phase_change: str = "none",
        morphology: str = "fixed",
        surface_exchange: str = "none",
        continuum_exchange: str = "none",
        coupling_schedule: str = "standalone",
        backend: str = "reference",
        precision: str = "float64",
        sensitivity: str = "forward",
        status: ParticlePhysicsSupportStatus = ParticlePhysicsSupportStatus.EXPERIMENTAL,
        evidence_ids: Sequence[str] = (),
    ):
        if not isinstance(status, ParticlePhysicsSupportStatus):
            raise TypeError("status must be ParticlePhysicsSupportStatus.")
        labels = tuple(
            str(value)
            for value in (
                mechanics,
                geometry,
                internal_geometry,
                thermodynamics,
                transport,
                reaction,
                phase_change,
                morphology,
                surface_exchange,
                continuum_exchange,
                coupling_schedule,
                backend,
                precision,
                sensitivity,
            )
        )
        if any(not value for value in labels):
            raise ValueError("Particle physics support labels must be nonempty.")
        evidence = tuple(str(value) for value in evidence_ids)
        if any(not value for value in evidence) or len(set(evidence)) != len(evidence):
            raise ValueError("evidence_ids must be unique nonempty strings.")
        if (
            status
            in (
                ParticlePhysicsSupportStatus.QUALIFIED,
                ParticlePhysicsSupportStatus.PRODUCTION,
            )
            and not evidence
        ):
            raise ValueError("Qualified and production claims require evidence.")
        (
            self.mechanics,
            self.geometry,
            self.internal_geometry,
            self.thermodynamics,
            self.transport,
            self.reaction,
            self.phase_change,
            self.morphology,
            self.surface_exchange,
            self.continuum_exchange,
            self.coupling_schedule,
            self.backend,
            self.precision,
            self.sensitivity,
        ) = labels
        self.status = status
        self.evidence_ids = evidence
        self.claim_id = canonical_fingerprint(
            {
                "kind": "particle-physics-support-claim",
                "configuration": list(labels),
                "status": status.value,
                "evidence": list(evidence),
            }
        )

    @property
    def configuration(self):
        return (
            self.mechanics,
            self.geometry,
            self.internal_geometry,
            self.thermodynamics,
            self.transport,
            self.reaction,
            self.phase_change,
            self.morphology,
            self.surface_exchange,
            self.continuum_exchange,
            self.coupling_schedule,
            self.backend,
            self.precision,
            self.sensitivity,
        )


class ParticlePhysicsSupportMatrix(StrictModule, NonTrainableState):
    claims: tuple[ParticlePhysicsSupportClaim, ...]
    matrix_id: str = eqx.field(static=True)

    def __init__(self, claims: Sequence[ParticlePhysicsSupportClaim], /):
        values = tuple(claims)
        if not values or any(
            not isinstance(value, ParticlePhysicsSupportClaim) for value in values
        ):
            raise TypeError("claims must contain ParticlePhysicsSupportClaim values.")
        configurations = tuple(value.configuration for value in values)
        if len(set(configurations)) != len(configurations):
            raise ValueError("Particle physics support configurations must be unique.")
        self.claims = values
        self.matrix_id = canonical_fingerprint(
            {
                "kind": "particle-physics-support-matrix",
                "claims": [value.claim_id for value in values],
            }
        )

    @property
    def production_ready(self):
        return all(
            value.status is ParticlePhysicsSupportStatus.PRODUCTION
            for value in self.claims
        )

    def claims_with_status(self, status, /):
        if not isinstance(status, ParticlePhysicsSupportStatus):
            raise TypeError("status must be ParticlePhysicsSupportStatus.")
        return tuple(value for value in self.claims if value.status is status)


__all__ = [
    "ParticlePhysicsSupportClaim",
    "ParticlePhysicsSupportMatrix",
    "ParticlePhysicsSupportStatus",
]
