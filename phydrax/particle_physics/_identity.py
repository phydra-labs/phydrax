#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum, StrEnum

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class ParticleRole(IntEnum):
    """Small normalized particle role; provider status remains separately namespaced."""

    UNKNOWN = 0
    BEAM = 1
    INCOMING = 2
    INTERMEDIATE = 3
    OUTGOING = 4


class ReproducibilityGrade(StrEnum):
    """Reproducibility actually supplied by a native or external provider."""

    EXACT_SEMANTIC_RNG = "exact-semantic-rng"
    EVENT_STABLE = "event-stable"
    SEED_REPLAYABLE = "seed-replayable"
    STATISTICAL = "statistical"
    UNCONTROLLED = "uncontrolled"


class DerivativeMode(StrEnum):
    UNSUPPORTED = "unsupported"
    PATHWISE = "pathwise"
    SCORE = "score"
    SURROGATE = "surrogate"


class ParticleCatalogueReference(StrictModule, NonTrainableState):
    """Immutable authority and checksum for integer particle identities."""

    source_id: str = eqx.field(static=True)
    provider_release: str = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    citation_url: str = eqx.field(static=True)
    catalogue_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_id: str,
        provider_release: str,
        checksum: str,
        citation_url: str,
    ):
        self.source_id = _identifier(source_id, "Source ID")
        self.provider_release = _identifier(provider_release, "Provider release")
        self.checksum = _identifier(checksum, "Checksum")
        self.citation_url = _identifier(citation_url, "Citation URL")
        self.catalogue_id = canonical_fingerprint(
            {
                "kind": "particle-catalogue-reference",
                "source": self.source_id,
                "release": self.provider_release,
                "checksum": self.checksum,
                "citation": self.citation_url,
            }
        )


class DerivativeContract(StrictModule, NonTrainableState):
    """Parameter-level derivative claim for one fixed support program."""

    differentiable_parameters: tuple[str, ...] = eqx.field(static=True)
    discrete_parameters: tuple[str, ...] = eqx.field(static=True)
    stopped_events: tuple[str, ...] = eqx.field(static=True)
    mode: DerivativeMode = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        differentiable_parameters: Sequence[str] = (),
        discrete_parameters: Sequence[str] = (),
        stopped_events: Sequence[str] = (),
        mode: DerivativeMode = DerivativeMode.UNSUPPORTED,
        support_id: str,
        evidence_ids: Sequence[str] = (),
    ):
        if not isinstance(mode, DerivativeMode):
            raise TypeError("mode must be DerivativeMode.")
        differentiable = tuple(
            _identifier(v, "Differentiable parameter") for v in differentiable_parameters
        )
        discrete = tuple(
            _identifier(v, "Discrete parameter") for v in discrete_parameters
        )
        stopped = tuple(_identifier(v, "Stopped event") for v in stopped_events)
        evidence = tuple(_identifier(v, "Evidence ID") for v in evidence_ids)
        for values, name in (
            (differentiable, "differentiable_parameters"),
            (discrete, "discrete_parameters"),
            (stopped, "stopped_events"),
            (evidence, "evidence_ids"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique values.")
        if set(differentiable) & set(discrete):
            raise ValueError("A parameter cannot be both differentiable and discrete.")
        if mode is not DerivativeMode.UNSUPPORTED and not differentiable:
            raise ValueError(
                "A supported derivative mode requires differentiable parameters."
            )
        if mode is not DerivativeMode.UNSUPPORTED and not evidence:
            raise ValueError("A supported derivative mode requires evidence.")
        self.differentiable_parameters = differentiable
        self.discrete_parameters = discrete
        self.stopped_events = stopped
        self.mode = mode
        self.support_id = _identifier(support_id, "Support ID")
        self.evidence_ids = evidence
        self.contract_id = canonical_fingerprint(
            {
                "kind": "hep-derivative-contract",
                "differentiable": list(differentiable),
                "discrete": list(discrete),
                "stopped_events": list(stopped),
                "mode": mode.value,
                "support": self.support_id,
                "evidence": list(evidence),
            }
        )


__all__ = [
    "DerivativeContract",
    "DerivativeMode",
    "ParticleCatalogueReference",
    "ParticleRole",
    "ReproducibilityGrade",
]
