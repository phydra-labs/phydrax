#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Heterogeneous measurement collections with explicit typed relations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from .._fingerprint import canonical_fingerprint, canonical_mapping
from ._asset import MeasurementAsset
from ._quantity import canonical_quantity_text


class MeasurementRole(StrEnum):
    RAW_ACQUISITION = "raw_acquisition"
    CALIBRATION = "calibration"
    REFERENCE = "reference"
    OBSERVATION = "observation"
    DERIVED_PRODUCT = "derived_product"
    RECONSTRUCTION = "reconstruction"
    INFERRED_RESULT = "inferred_result"
    SYNTHETIC_PREDICTION = "synthetic_prediction"


class MeasurementRelationKind(StrEnum):
    SAME_ACQUISITION = "same_acquisition"
    CALIBRATION_APPLIES_TO = "calibration_applies_to"
    REFERENCE_FOR = "reference_for"
    REGISTERED_TO = "registered_to"
    DERIVED_FROM = "derived_from"
    SYNCHRONIZED_WITH = "synchronized_with"


@dataclass(frozen=True, slots=True)
class MeasurementRelation:
    source_asset_id: str
    target_asset_id: str
    kind: MeasurementRelationKind
    evidence_id: str
    relation_id: str = field(init=False)

    def __post_init__(self) -> None:
        source = canonical_quantity_text(self.source_asset_id, "source_asset_id")
        target = canonical_quantity_text(self.target_asset_id, "target_asset_id")
        if source == target:
            raise ValueError("Measurement relations require distinct assets.")
        if not isinstance(self.kind, MeasurementRelationKind):
            raise TypeError("kind must be MeasurementRelationKind.")
        evidence = canonical_quantity_text(self.evidence_id, "evidence_id")
        object.__setattr__(self, "source_asset_id", source)
        object.__setattr__(self, "target_asset_id", target)
        object.__setattr__(self, "evidence_id", evidence)
        object.__setattr__(
            self,
            "relation_id",
            canonical_fingerprint(
                {
                    "kind": "measurement-relation",
                    "source": source,
                    "target": target,
                    "relation_kind": self.kind.value,
                    "evidence": evidence,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class MeasurementRoleAssignment:
    asset_id: str
    role: MeasurementRole
    assignment_id: str = field(init=False)

    def __post_init__(self) -> None:
        asset = canonical_quantity_text(self.asset_id, "asset_id")
        if not isinstance(self.role, MeasurementRole):
            raise TypeError("role must be MeasurementRole.")
        object.__setattr__(self, "asset_id", asset)
        object.__setattr__(
            self,
            "assignment_id",
            canonical_fingerprint(
                {
                    "kind": "measurement-role-assignment",
                    "asset": asset,
                    "role": self.role.value,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class MeasurementCollection:
    collection_id: str
    campaign_id: str
    assets: tuple[MeasurementAsset, ...]
    roles: tuple[MeasurementRoleAssignment, ...]
    relations: tuple[MeasurementRelation, ...] = ()
    parent_collection_ids: tuple[str, ...] = ()
    platform_id: str | None = None
    metadata: Mapping[str, Any] | None = None
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        collection = canonical_quantity_text(self.collection_id, "collection_id")
        campaign = canonical_quantity_text(self.campaign_id, "campaign_id")
        assets = tuple(self.assets)
        if not assets or any(not isinstance(value, MeasurementAsset) for value in assets):
            raise TypeError("assets must contain at least one MeasurementAsset.")
        asset_ids = tuple(value.asset_id for value in assets)
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("Collection asset IDs must be unique.")
        roles = tuple(self.roles)
        if any(not isinstance(value, MeasurementRoleAssignment) for value in roles):
            raise TypeError("roles must contain MeasurementRoleAssignment values.")
        if {value.asset_id for value in roles} != set(asset_ids):
            raise ValueError(
                "Every collection asset requires exactly one role assignment."
            )
        if len(roles) != len(assets):
            raise ValueError(
                "Every collection asset requires one unique role assignment."
            )
        relations = tuple(self.relations)
        if any(not isinstance(value, MeasurementRelation) for value in relations):
            raise TypeError("relations must contain MeasurementRelation values.")
        if any(
            value.source_asset_id not in asset_ids
            or value.target_asset_id not in asset_ids
            for value in relations
        ):
            raise ValueError("Every relation endpoint must belong to the collection.")
        if len({value.relation_id for value in relations}) != len(relations):
            raise ValueError("Collection relations must be unique.")
        parents = tuple(
            canonical_quantity_text(value, "parent_collection_id")
            for value in self.parent_collection_ids
        )
        if collection in parents or len(parents) != len(set(parents)):
            raise ValueError(
                "Parent collection IDs must be unique and non-self-referential."
            )
        platform = (
            None
            if self.platform_id is None
            else canonical_quantity_text(self.platform_id, "platform_id")
        )
        metadata = canonical_mapping({} if self.metadata is None else self.metadata)
        object.__setattr__(self, "collection_id", collection)
        object.__setattr__(self, "campaign_id", campaign)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "parent_collection_ids", parents)
        object.__setattr__(self, "platform_id", platform)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(
            self,
            "content_id",
            canonical_fingerprint(
                {
                    "kind": "measurement-collection",
                    "collection": collection,
                    "campaign": campaign,
                    "assets": [value.content_id for value in assets],
                    "roles": [value.assignment_id for value in roles],
                    "relations": [value.relation_id for value in relations],
                    "parents": list(parents),
                    "platform": platform,
                    "metadata": metadata,
                }
            ),
        )

    def asset(self, asset_id: str, /) -> MeasurementAsset:
        identifier = canonical_quantity_text(asset_id, "asset_id")
        matches = tuple(value for value in self.assets if value.asset_id == identifier)
        if len(matches) != 1:
            raise KeyError(identifier)
        return matches[0]


__all__ = [
    "MeasurementCollection",
    "MeasurementRelation",
    "MeasurementRelationKind",
    "MeasurementRole",
    "MeasurementRoleAssignment",
]
