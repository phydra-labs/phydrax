#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Lower accepted stable-ID publication banks to compact owner-local arrays."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_array_hex_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import parse


class PublicationEntityRole(StrEnum):
    """Stable-ID namespace used to lower one accepted numerical bank."""

    CELL = "cell"
    SOURCE_POINT = "source-point"
    VERTEX = "vertex"


class PublicationFieldRole(StrEnum):
    """Scientific meaning retained for each lowered accepted array."""

    FIELD = "field"
    GEOMETRY = "geometry"
    HISTORY = "history"
    ORGANIZATION = "organization"
    SOURCE_SUPPORT = "source-support"


@final
class PublicationField(StrictModule, NonTrainableState):
    """One accepted field and its canonical owner-local ID/value rows."""

    name: str = eqx.field(static=True)
    entity_role: PublicationEntityRole = eqx.field(static=True)
    field_role: PublicationFieldRole = eqx.field(static=True)
    entity_ids: Array
    values: Array
    owned: Array

    def __init__(
        self,
        name: str,
        entity_role: PublicationEntityRole,
        field_role: PublicationFieldRole,
        entity_ids: ArrayLike,
        values: ArrayLike,
        owned: ArrayLike,
        /,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Publication field names must be non-empty strings.")
        entity_role_ = parse(entity_role, PublicationEntityRole, "entity_role")
        field_role_ = parse(field_role, PublicationFieldRole, "field_role")
        ids = np.asarray(entity_ids)
        payload = np.asarray(values)
        owned_rows = np.asarray(owned)
        if (
            ids.dtype.kind not in "iu"
            or ids.ndim != 1
            or not ids.dtype.isnative
            or (ids.size and np.any(ids < 0))
            or (
                ids.dtype.kind == "u"
                and ids.size
                and int(np.max(ids)) > np.iinfo(np.int64).max
            )
            or (ids.size > 1 and np.unique(ids).size != ids.size)
            or payload.ndim == 0
            or payload.shape[0] != ids.shape[0]
            or payload.dtype.hasobject
            or payload.dtype.kind not in "biufc"
            or not payload.dtype.isnative
            or owned_rows.shape != ids.shape
            or owned_rows.dtype != np.dtype(np.bool_)
        ):
            raise ValueError(
                "Publication field IDs, values, or owner mask are malformed."
            )
        order = np.argsort(ids, kind="stable")
        self.name = name.strip()
        self.entity_role = entity_role_
        self.field_role = field_role_
        self.entity_ids = jnp.asarray(ids[order], dtype=jnp.int64)
        self.values = jnp.asarray(payload[order])
        self.owned = jnp.asarray(owned_rows[order], dtype=jnp.bool_)


@final
class LoweredPublication(StrictModule, NonTrainableState):
    """Compact accepted arrays for one current owner rank."""

    accepted_id: str = eqx.field(static=True)
    distribution_id: str = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    fields: tuple[PublicationField, ...]
    source_identity_id: str = eqx.field(static=True)
    lowered_id: str = eqx.field(static=True)

    def __init__(
        self,
        accepted_id: str,
        distribution_id: str,
        rank: int,
        part_count: int,
        fields: tuple[PublicationField, ...],
        source_identity_id: str,
        /,
    ) -> None:
        if (
            not isinstance(accepted_id, str)
            or not accepted_id.strip()
            or not isinstance(distribution_id, str)
            or not distribution_id.strip()
            or isinstance(rank, bool)
            or not isinstance(rank, (int, np.integer))
            or isinstance(part_count, bool)
            or not isinstance(part_count, (int, np.integer))
            or int(part_count) < 1
            or not 0 <= int(rank) < int(part_count)
            or not isinstance(source_identity_id, str)
            or not source_identity_id
        ):
            raise ValueError("Lowered publication identity or owner rank is malformed.")
        if any(not isinstance(field, PublicationField) for field in fields):
            raise TypeError(
                "Lowered publication fields must be PublicationField records."
            )
        names = tuple(field.name for field in fields)
        if len(set(names)) != len(names):
            raise ValueError("Lowered publication field names must be unique.")
        ordered = tuple(sorted(fields, key=lambda field: field.name))
        self.accepted_id = accepted_id.strip()
        self.distribution_id = distribution_id.strip()
        self.rank = int(rank)
        self.part_count = int(part_count)
        self.fields = ordered
        self.source_identity_id = source_identity_id
        self.lowered_id = canonical_fingerprint(
            {
                "kind": "owner-local-publication-lowering",
                "accepted": accepted_id,
                "distribution": distribution_id,
                "rank": int(rank),
                "part_count": int(part_count),
                "source": source_identity_id,
                "fields": tuple(
                    (
                        field.name,
                        field.entity_role.value,
                        field.field_role.value,
                        canonical_array_hex_fingerprint(
                            np.asarray(field.entity_ids, dtype=np.int64)
                        ),
                        canonical_array_hex_fingerprint(np.asarray(field.values)),
                        canonical_array_hex_fingerprint(
                            np.asarray(field.owned, dtype=np.bool_)
                        ),
                    )
                    for field in ordered
                ),
            }
        )

    def arrays(self) -> dict[str, Array]:
        """Return field-role-lowered arrays by stable publication name."""
        return {field.name: field.values for field in self.fields}

    def roles(
        self,
    ) -> tuple[tuple[str, PublicationEntityRole, PublicationFieldRole], ...]:
        """Return canonical role declarations consumed by a compact result view."""
        return tuple(
            (field.name, field.entity_role, field.field_role) for field in self.fields
        )


def lower_accepted_publication(
    accepted_id: str,
    distribution_id: str,
    arrays: Mapping[str, ArrayLike],
    entity_ids: Mapping[str, ArrayLike],
    entity_owner_ranks: Mapping[str, ArrayLike],
    entity_roles: Mapping[str, PublicationEntityRole],
    field_roles: Mapping[str, PublicationFieldRole],
    rank: int,
    part_count: int,
    /,
    *,
    entity_present: Mapping[str, ArrayLike] | None = None,
) -> LoweredPublication:
    """Project role-declared accepted arrays onto one owner using stable IDs.

    Rows are selected by the declared current owner rank and then sorted by
    stable entity ID. The source bank remains untouched; each lowered field
    retains its entity namespace and scientific role as static identity.
    """
    if (
        isinstance(rank, bool)
        or not isinstance(rank, (int, np.integer))
        or isinstance(part_count, bool)
        or not isinstance(part_count, (int, np.integer))
        or int(part_count) < 1
        or not 0 <= int(rank) < int(part_count)
    ):
        raise ValueError("Publication owner rank must lie in the declared part count.")
    names = set(arrays)
    if (
        names != set(entity_ids)
        or names != set(entity_owner_ranks)
        or names != set(entity_roles)
        or names != set(field_roles)
        or (entity_present is not None and names != set(entity_present))
        or any(not isinstance(name, str) or not name.strip() for name in names)
    ):
        raise ValueError(
            "Accepted publication banks and role declarations must align exactly."
        )
    local_fields: list[PublicationField] = []
    source_records: list[tuple[str, str, str, str, str, str]] = []
    for name in sorted(names):
        raw_ids = np.asarray(entity_ids[name])
        owners = np.asarray(entity_owner_ranks[name])
        values = np.asarray(arrays[name])
        present = (
            np.zeros(raw_ids.shape, dtype=np.bool_)
            if entity_present is None
            else np.asarray(entity_present[name])
        )
        entity_role = parse(
            entity_roles[name], PublicationEntityRole, f"{name}.entity_role"
        )
        field_role = parse(field_roles[name], PublicationFieldRole, f"{name}.field_role")
        if (
            raw_ids.dtype.kind not in "iu"
            or raw_ids.ndim != 1
            or not raw_ids.dtype.isnative
            or (raw_ids.size and np.any(raw_ids < 0))
            or (
                raw_ids.dtype.kind == "u"
                and raw_ids.size
                and int(np.max(raw_ids)) > np.iinfo(np.int64).max
            )
            or np.unique(raw_ids).size != raw_ids.size
            or owners.dtype.kind not in "iu"
            or not owners.dtype.isnative
            or owners.shape != raw_ids.shape
            or np.any(owners < 0)
            or np.any(owners >= int(part_count))
            or present.shape != raw_ids.shape
            or present.dtype != np.dtype(np.bool_)
            or values.ndim == 0
            or values.shape[0] != raw_ids.size
            or values.dtype.hasobject
            or values.dtype.kind not in "biufc"
            or not values.dtype.isnative
        ):
            raise ValueError(f"Accepted publication bank {name!r} is malformed.")
        selected = np.flatnonzero((owners == int(rank)) | present)
        order = np.argsort(raw_ids[selected], kind="stable")
        selected = selected[order]
        local_fields.append(
            PublicationField(
                name,
                entity_role,
                field_role,
                raw_ids[selected],
                values[selected],
                owners[selected] == int(rank),
            )
        )
        source_records.append(
            (
                name,
                entity_role.value,
                field_role.value,
                canonical_array_hex_fingerprint(raw_ids.astype(np.int64, copy=False)),
                canonical_array_hex_fingerprint(owners.astype(np.int32, copy=False)),
                canonical_array_hex_fingerprint(values),
            )
        )
    source_identity_id = canonical_fingerprint(
        {
            "accepted": accepted_id,
            "distribution": distribution_id,
            "part_count": int(part_count),
            "field_roles": tuple(source_records),
        }
    )
    return LoweredPublication(
        accepted_id,
        distribution_id,
        int(rank),
        int(part_count),
        tuple(local_fields),
        source_identity_id,
    )


__all__ = [
    "LoweredPublication",
    "PublicationEntityRole",
    "PublicationField",
    "PublicationFieldRole",
    "lower_accepted_publication",
]
