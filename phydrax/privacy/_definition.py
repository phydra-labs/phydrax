#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .._fingerprint import canonical_fingerprint
from .._validation import canonical_identifier


def _require_record(
    record: dict[str, Any] | Mapping[str, Any],
    kind: str,
    required: frozenset[str],
    /,
    *,
    optional: frozenset[str] = frozenset(),
) -> None:
    if record.get("kind") != kind:
        raise ValueError(f"Serialized record must have kind {kind!r}.")
    expected = required | {"kind"}
    missing = expected - set(record)
    unknown = set(record) - expected - optional
    if missing or unknown:
        raise ValueError(
            f"Serialized {kind} fields differ from the canonical contract; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )


class NeighboringRelation(StrEnum):
    """Dataset relation protected by a differential-privacy mechanism."""

    ADD_OR_REMOVE_ONE = "add-or-remove-one"
    REPLACE_ONE = "replace-one"
    REPLACE_SPECIAL = "replace-special"


class TrustModel(StrEnum):
    """Party trusted to observe unprotected data and mechanism state."""

    CENTRAL = "central"
    LOCAL = "local"
    SHUFFLE = "shuffle"
    DISTRIBUTED = "distributed"


class RandomnessAssurance(StrEnum):
    """Security status of the randomness used by a mechanism."""

    RESEARCH_PRNG = "research-prng"
    SECURE_HOST = "secure-host"
    SECURE_DISTRIBUTED = "secure-distributed"

    @property
    def permits_public_release(self) -> bool:
        return self is not RandomnessAssurance.RESEARCH_PRNG


@dataclass(frozen=True, slots=True)
class PrivacyUnit:
    """One semantic entity whose presence or value may differ between neighbors."""

    name: str
    unit_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = canonical_identifier(self.name, "privacy unit")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "unit_id", canonical_fingerprint(self._content_record()))

    def _content_record(self) -> dict[str, object]:
        return {"kind": "privacy-unit", "name": self.name}

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "unit_id": self.unit_id}

    @classmethod
    def from_record(cls, record: dict[str, Any], /) -> PrivacyUnit:
        _require_record(
            record,
            "privacy-unit",
            frozenset(("name",)),
            optional=frozenset(("unit_id",)),
        )
        value = cls(str(record["name"]))
        recorded_id = record.get("unit_id")
        if recorded_id is not None and str(recorded_id) != value.unit_id:
            raise ValueError("Serialized privacy unit has an invalid content address.")
        return value


@dataclass(frozen=True, slots=True)
class PrivacyDefinition:
    """Privacy unit, adjacency, and trust assumptions for one data scope."""

    unit: PrivacyUnit
    neighboring_relation: NeighboringRelation
    trust_model: TrustModel = TrustModel.CENTRAL
    population_size_public: bool = False
    definition_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.unit, PrivacyUnit):
            raise TypeError("unit must be a PrivacyUnit.")
        if not isinstance(self.neighboring_relation, NeighboringRelation):
            raise TypeError("neighboring_relation must be a NeighboringRelation.")
        if not isinstance(self.trust_model, TrustModel):
            raise TypeError("trust_model must be a TrustModel.")
        object.__setattr__(
            self, "definition_id", canonical_fingerprint(self._content_record())
        )

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "privacy-definition",
            "unit": self.unit.to_record(),
            "neighboring_relation": self.neighboring_relation.value,
            "trust_model": self.trust_model.value,
            "population_size_public": self.population_size_public,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "definition_id": self.definition_id}

    @classmethod
    def from_record(cls, record: dict[str, Any], /) -> PrivacyDefinition:
        _require_record(
            record,
            "privacy-definition",
            frozenset(
                (
                    "unit",
                    "neighboring_relation",
                    "trust_model",
                    "population_size_public",
                )
            ),
            optional=frozenset(("definition_id",)),
        )
        unit = record["unit"]
        if not isinstance(unit, dict):
            raise TypeError("Serialized privacy definition unit must be a mapping.")
        value = cls(
            PrivacyUnit.from_record(unit),
            NeighboringRelation(str(record["neighboring_relation"])),
            TrustModel(str(record["trust_model"])),
            bool(record["population_size_public"]),
        )
        recorded_id = record.get("definition_id")
        if recorded_id is not None and str(recorded_id) != value.definition_id:
            raise ValueError(
                "Serialized privacy definition has an invalid content address."
            )
        return value


@dataclass(frozen=True, slots=True)
class PrivateDataScope:
    """Opaque release scope for one private dataset and privacy definition."""

    scope_id: str
    definition: PrivacyDefinition
    scope_contract_id: str = field(init=False)

    def __post_init__(self) -> None:
        scope_id = canonical_identifier(self.scope_id, "private data scope ID")
        if not isinstance(self.definition, PrivacyDefinition):
            raise TypeError("definition must be a PrivacyDefinition.")
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(
            self, "scope_contract_id", canonical_fingerprint(self._content_record())
        )

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "private-data-scope",
            "scope_id": self.scope_id,
            "definition": self.definition.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "scope_contract_id": self.scope_contract_id}

    @classmethod
    def from_record(cls, record: dict[str, Any], /) -> PrivateDataScope:
        _require_record(
            record,
            "private-data-scope",
            frozenset(("scope_id", "definition")),
            optional=frozenset(("scope_contract_id",)),
        )
        definition = record["definition"]
        if not isinstance(definition, dict):
            raise TypeError("Serialized scope definition must be a mapping.")
        value = cls(str(record["scope_id"]), PrivacyDefinition.from_record(definition))
        recorded_id = record.get("scope_contract_id")
        if recorded_id is not None and str(recorded_id) != value.scope_contract_id:
            raise ValueError(
                "Serialized private data scope has an invalid content address."
            )
        return value


__all__ = [
    "NeighboringRelation",
    "PrivateDataScope",
    "PrivacyDefinition",
    "PrivacyUnit",
    "RandomnessAssurance",
    "TrustModel",
]
