#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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


__all__ = [
    "ParticleCatalogueReference",
    "ParticleRole",
    "ReproducibilityGrade",
]
