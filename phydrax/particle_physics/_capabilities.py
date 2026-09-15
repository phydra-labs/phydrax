#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._identity import ReproducibilityGrade


class HEPCapabilityContract(StrictModule, NonTrainableState):
    """Admitted capabilities and support of one pinned HEP provider configuration."""

    provider_id: str = eqx.field(static=True)
    provider_release: str = eqx.field(static=True)
    configuration_checksum: str = eqx.field(static=True)
    data_checksums: tuple[str, ...] = eqx.field(static=True)
    capabilities: tuple[str, ...] = eqx.field(static=True)
    support_domain_id: str = eqx.field(static=True)
    reproducibility: ReproducibilityGrade = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        provider_id: str,
        provider_release: str,
        configuration_checksum: str,
        data_checksums: Sequence[str] = (),
        capabilities: Sequence[str],
        support_domain_id: str,
        reproducibility: ReproducibilityGrade,
        evidence_ids: Sequence[str] = (),
    ):
        def identifiers(values: Sequence[str], name: str) -> tuple[str, ...]:
            result = tuple(str(value).strip() for value in values)
            if any(not value for value in result) or len(set(result)) != len(result):
                raise ValueError(f"{name} must contain distinct non-empty values.")
            return result

        if not isinstance(reproducibility, ReproducibilityGrade):
            raise TypeError("reproducibility must be ReproducibilityGrade.")
        provider = str(provider_id).strip()
        release = str(provider_release).strip()
        configuration = str(configuration_checksum).strip()
        domain = str(support_domain_id).strip()
        capability_values = identifiers(capabilities, "capabilities")
        data = identifiers(data_checksums, "data_checksums")
        evidence = identifiers(evidence_ids, "evidence_ids")
        if (
            not provider
            or not release
            or not configuration
            or not domain
            or not capability_values
        ):
            raise ValueError(
                "Provider identity, release, configuration, domain, and capabilities are required."
            )
        self.provider_id = provider
        self.provider_release = release
        self.configuration_checksum = configuration
        self.data_checksums = data
        self.capabilities = capability_values
        self.support_domain_id = domain
        self.reproducibility = reproducibility
        self.evidence_ids = evidence
        self.capability_id = canonical_fingerprint(
            {
                "kind": "hep-provider-capability",
                "provider": provider,
                "release": release,
                "configuration": configuration,
                "data": list(data),
                "capabilities": list(capability_values),
                "support": domain,
                "reproducibility": reproducibility.value,
                "evidence": list(evidence),
            }
        )

    def supports(self, capability: str, /) -> bool:
        return str(capability) in self.capabilities


__all__ = ["HEPCapabilityContract"]
