#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._core import DiscretizationKey, nonempty_identifier, resolved_identifier
from ._transfer import FieldTransfer


class DiscretizationRecord(StrictModule, NonTrainableState):
    """Typed provenance record for one approximation artifact or realization."""

    key: DiscretizationKey
    artifact_kind: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    numeric_version: str | None = eqx.field(static=True)
    dependency_key_ids: tuple[str, ...] = eqx.field(static=True)
    realization_id: str | None = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        key: DiscretizationKey,
        artifact_kind: str,
        artifact_id: str,
        /,
        *,
        numeric_version: str | None = None,
        dependency_key_ids: Sequence[str] = (),
        realization_id: str | None = None,
        record_id: str | None = None,
    ):
        if not isinstance(key, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        kind = nonempty_identifier("artifact_kind", artifact_kind)
        artifact = nonempty_identifier("artifact_id", artifact_id)
        version = (
            None
            if numeric_version is None
            else nonempty_identifier("numeric_version", numeric_version)
        )
        dependencies = tuple(str(value) for value in dependency_key_ids)
        if any(not value for value in dependencies) or len(set(dependencies)) != len(
            dependencies
        ):
            raise ValueError("dependency_key_ids must be unique non-empty strings.")
        if key.key_id in dependencies:
            raise ValueError("A discretization record cannot depend on itself.")
        realization = (
            None
            if realization_id is None
            else nonempty_identifier("realization_id", realization_id)
        )
        self.key = key
        self.artifact_kind = kind
        self.artifact_id = artifact
        self.numeric_version = version
        self.dependency_key_ids = dependencies
        self.realization_id = realization
        self.record_id = resolved_identifier(
            "record_id",
            record_id,
            {
                "kind": "discretization-record",
                "key": key.key_id,
                "artifact_kind": kind,
                "artifact": artifact,
                "numeric_version": version,
                "dependencies": list(dependencies),
                "realization": realization,
            },
        )


class DiscretizationBundle(StrictModule, NonTrainableState):
    """Acyclic collection of every approximation used by one computation."""

    records: tuple[DiscretizationRecord, ...]
    transfers: tuple[FieldTransfer, ...]
    stochastic_coupling_ids: tuple[str, ...] = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)

    def __init__(
        self,
        records: Sequence[DiscretizationRecord],
        /,
        *,
        transfers: Sequence[FieldTransfer] = (),
        stochastic_coupling_ids: Sequence[str] = (),
        bundle_id: str | None = None,
    ):
        records_ = tuple(records)
        if not records_ or not all(
            isinstance(record, DiscretizationRecord) for record in records_
        ):
            raise TypeError(
                "records must contain one or more DiscretizationRecord values."
            )
        key_ids = tuple(record.key.key_id for record in records_)
        key_names = tuple((record.key.role, record.key.name) for record in records_)
        if len(set(key_ids)) != len(key_ids) or len(set(key_names)) != len(key_names):
            raise ValueError("Bundle discretization keys must be unique.")
        known = set(key_ids)
        for record in records_:
            unknown = tuple(
                dependency
                for dependency in record.dependency_key_ids
                if dependency not in known
            )
            if unknown:
                raise ValueError(
                    f"Record {record.key.name!r} has unknown dependencies {unknown}."
                )
        self._validate_acyclic(records_)
        transfers_ = tuple(transfers)
        if not all(isinstance(transfer, FieldTransfer) for transfer in transfers_):
            raise TypeError("transfers must contain FieldTransfer values.")
        transfer_ids = tuple(transfer.transfer_id for transfer in transfers_)
        if len(set(transfer_ids)) != len(transfer_ids):
            raise ValueError("Bundle transfer IDs must be unique.")
        couplings = tuple(str(value) for value in stochastic_coupling_ids)
        if any(not value for value in couplings) or len(set(couplings)) != len(couplings):
            raise ValueError("stochastic_coupling_ids must be unique non-empty strings.")
        self.records = records_
        self.transfers = transfers_
        self.stochastic_coupling_ids = couplings
        self.bundle_id = resolved_identifier(
            "bundle_id",
            bundle_id,
            {
                "kind": "discretization-bundle",
                "records": [record.record_id for record in records_],
                "transfers": list(transfer_ids),
                "stochastic_couplings": list(couplings),
            },
        )

    @staticmethod
    def _validate_acyclic(records: tuple[DiscretizationRecord, ...], /) -> None:
        dependencies = {
            record.key.key_id: set(record.dependency_key_ids) for record in records
        }
        ready = [key for key, values in dependencies.items() if not values]
        visited = 0
        while ready:
            key = ready.pop()
            visited += 1
            for candidate, values in dependencies.items():
                if key in values:
                    values.remove(key)
                    if not values:
                        ready.append(candidate)
        if visited != len(records):
            raise ValueError("Discretization bundle dependencies must be acyclic.")

    def record(self, key: DiscretizationKey | str, /) -> DiscretizationRecord:
        key_id = key.key_id if isinstance(key, DiscretizationKey) else str(key)
        for record in self.records:
            if record.key.key_id == key_id:
                return record
        raise KeyError(f"Unknown discretization key {key_id!r}.")


class DiscretizationLevel(StrictModule, NonTrainableState):
    """One complete approximation bundle in a refinement hierarchy."""

    bundle: DiscretizationBundle
    parent_level_id: str | None = eqx.field(static=True)
    transfers: tuple[FieldTransfer, ...]
    refinements: tuple[str, ...] = eqx.field(static=True)
    level_id: str = eqx.field(static=True)

    def __init__(
        self,
        bundle: DiscretizationBundle,
        /,
        *,
        parent_level_id: str | None = None,
        transfers: Sequence[FieldTransfer] = (),
        refinements: Sequence[str] = (),
        level_id: str | None = None,
    ):
        if not isinstance(bundle, DiscretizationBundle):
            raise TypeError("bundle must be a DiscretizationBundle.")
        parent = (
            None
            if parent_level_id is None
            else nonempty_identifier("parent_level_id", parent_level_id)
        )
        transfers_ = tuple(transfers)
        if not all(isinstance(transfer, FieldTransfer) for transfer in transfers_):
            raise TypeError("transfers must contain FieldTransfer values.")
        refinements_ = tuple(str(value) for value in refinements)
        if any(not value for value in refinements_) or len(set(refinements_)) != len(
            refinements_
        ):
            raise ValueError("refinements must be unique non-empty strings.")
        self.bundle = bundle
        self.parent_level_id = parent
        self.transfers = transfers_
        self.refinements = refinements_
        self.level_id = resolved_identifier(
            "level_id",
            level_id,
            {
                "kind": "discretization-level",
                "bundle": bundle.bundle_id,
                "parent": parent,
                "transfers": [transfer.transfer_id for transfer in transfers_],
                "refinements": list(refinements_),
            },
        )
        if self.level_id == parent:
            raise ValueError("A discretization level cannot be its own parent.")


class DiscretizationHierarchy(StrictModule, NonTrainableState):
    """Ordered acyclic hierarchy of complete discretization levels."""

    levels: tuple[DiscretizationLevel, ...]
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        levels: Sequence[DiscretizationLevel],
        /,
        *,
        hierarchy_id: str | None = None,
    ):
        levels_ = tuple(levels)
        if not levels_ or not all(
            isinstance(level, DiscretizationLevel) for level in levels_
        ):
            raise TypeError("levels must contain one or more DiscretizationLevel values.")
        identifiers = tuple(level.level_id for level in levels_)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Discretization level IDs must be unique.")
        if levels_[0].parent_level_id is not None:
            raise ValueError("The first discretization level must be a root.")
        previous: set[str] = {levels_[0].level_id}
        for level in levels_[1:]:
            if level.parent_level_id not in previous:
                raise ValueError(
                    "Each level parent must precede the level in the hierarchy."
                )
            if not level.refinements:
                raise ValueError("Non-root hierarchy levels must declare refinements.")
            previous.add(level.level_id)
        self.levels = levels_
        self.hierarchy_id = resolved_identifier(
            "hierarchy_id",
            hierarchy_id,
            {
                "kind": "discretization-hierarchy",
                "levels": [level.level_id for level in levels_],
            },
        )


__all__ = [
    "DiscretizationBundle",
    "DiscretizationHierarchy",
    "DiscretizationLevel",
    "DiscretizationRecord",
]
