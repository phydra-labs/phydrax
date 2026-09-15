#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _identifiers(
    values: Sequence[str], name: str, /, *, required: bool = True
) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if (
        (required and not result)
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain distinct non-empty values.")
    return tuple(sorted(result))


class SystematicKind(StrEnum):
    NORMALIZATION = "normalization"
    EVENT_WEIGHT = "event-weight"
    KINEMATIC = "kinematic"
    MULTIPLICITY = "multiplicity"
    REPLICA = "replica"
    HESSIAN = "hessian"
    ENVELOPE = "envelope"
    ALTERNATIVE_PROVIDER = "alternative-provider"
    CALIBRATION = "calibration"


class SystematicSource(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    kind: SystematicKind = eqx.field(static=True)
    variation_names: tuple[str, ...] = eqx.field(static=True)
    affected_collections: tuple[str, ...] = eqx.field(static=True)
    correlation_scopes: tuple[str, ...] = eqx.field(static=True)
    mutually_exclusive_group: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: SystematicKind,
        variation_names: Sequence[str],
        /,
        *,
        affected_collections: Sequence[str],
        correlation_scopes: Sequence[str],
        provider_id: str,
        mutually_exclusive_group: str = "none",
    ):
        name_ = str(name).strip()
        provider = str(provider_id).strip()
        exclusive = str(mutually_exclusive_group).strip()
        if (
            not name_
            or not provider
            or not exclusive
            or not isinstance(kind, SystematicKind)
        ):
            raise ValueError(
                "Systematic identity, kind, provider, and exclusivity group are required."
            )
        variations = _identifiers(variation_names, "variation_names")
        collections = _identifiers(affected_collections, "affected_collections")
        scopes = _identifiers(correlation_scopes, "correlation_scopes")
        self.name = name_
        self.kind = kind
        self.variation_names = variations
        self.affected_collections = collections
        self.correlation_scopes = scopes
        self.mutually_exclusive_group = exclusive
        self.provider_id = provider
        self.source_id = canonical_fingerprint(
            {
                "kind": "hep-systematic-source",
                "name": name_,
                "systematic_kind": kind.value,
                "variations": list(variations),
                "collections": list(collections),
                "correlation_scopes": list(scopes),
                "exclusive_group": exclusive,
                "provider": provider,
            }
        )


class SystematicConfiguration(StrictModule, NonTrainableState):
    sources: tuple[SystematicSource, ...]
    configuration_id: str = eqx.field(static=True)

    def __init__(self, sources: Sequence[SystematicSource], /):
        sources_ = tuple(sources)
        if not sources_ or any(
            not isinstance(value, SystematicSource) for value in sources_
        ):
            raise TypeError("sources must contain typed non-empty systematic sources.")
        names = tuple(value.name for value in sources_)
        ids = tuple(value.source_id for value in sources_)
        if len(set(names)) != len(names) or len(set(ids)) != len(ids):
            raise ValueError("Systematic source names and identities must be unique.")
        self.sources = tuple(sorted(sources_, key=lambda value: value.name))
        self.configuration_id = canonical_fingerprint(
            {
                "kind": "hep-systematic-configuration",
                "sources": [value.source_id for value in self.sources],
            }
        )

    def by_name(self, name: str, /) -> SystematicSource:
        matches = tuple(value for value in self.sources if value.name == str(name))
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]


__all__ = ["SystematicConfiguration", "SystematicKind", "SystematicSource"]
