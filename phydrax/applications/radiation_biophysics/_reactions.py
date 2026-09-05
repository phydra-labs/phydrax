#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""External chemical reaction records, separate from deposited energy and lesions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..._fingerprint import canonical_fingerprint
from ._interactions import (
    _canonical_events,
    _nonnegative,
    _point,
    _text,
    RadiationEventKey,
    RadiationSource,
)


@dataclass(frozen=True, slots=True)
class ChemicalReaction:
    key: RadiationEventKey
    position: tuple[float, float, float]
    channel: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    source_site_id: str | None = None
    time: float | None = None
    track_ids: tuple[str, ...] = ()
    parent_event_keys: tuple[RadiationEventKey, ...] = ()
    material: str | None = None

    def __post_init__(self):
        if self.key.stage != "chemical":
            raise ValueError("Chemical reactions require chemical event keys.")
        _point(self.position)
        _text(self.channel, "reaction channel")
        for values in (
            self.reactants,
            self.products,
            self.track_ids,
            self.parent_event_keys,
        ):
            if not isinstance(values, tuple):
                raise TypeError("Reaction identities must be immutable tuples.")
        if not self.reactants:
            raise ValueError("Chemical reaction requires declared reactants.")
        for species in (*self.reactants, *self.products):
            _text(species, "species")
        if self.time is not None:
            _nonnegative(self.time, "reaction time")
        if any(parent.history != self.key.history for parent in self.parent_event_keys):
            raise ValueError("Reaction parents cannot cross primary histories.")


@dataclass(frozen=True, slots=True)
class ReactionLedger:
    source: RadiationSource
    records: tuple[ChemicalReaction, ...]

    def __post_init__(self):
        if not all(isinstance(item, ChemicalReaction) for item in self.records):
            raise TypeError("Reaction ledger requires chemical reactions.")
        object.__setattr__(self, "records", _canonical_events(self.records, self.source))
        endpoint = self.source.chemistry_endpoint
        if endpoint is not None and any(
            item.time is not None and item.time > endpoint for item in self.records
        ):
            raise ValueError("Reaction lies beyond the declared chemistry endpoint.")

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "source": self.source.fingerprint(),
                "chemical": [asdict(item) for item in self.records],
            }
        )
