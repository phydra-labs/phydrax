#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Identity of declared source entities, independent of numerical storage rows."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .._fingerprint import canonical_json, plain_json_fingerprint
from .._validation import canonical_identifier, nonnegative_integer


def source_entity_id(
    source_id: str,
    source_revision: str,
    kind: str,
    entity_id: str | int,
    /,
    *,
    occurrence_path: tuple[str, ...] = (),
) -> str:
    """Bind an authored identifier to its source, role, and exact occurrence.

    Strings and integer identifiers remain distinct. Structured encoding avoids
    delimiter ambiguity; geometry values and execution row positions are never
    used to infer identity. Source owners resolve this key through their actual
    entity tables rather than attempting to parse an index from the digest.
    """
    source = canonical_identifier(source_id, "source_id")
    revision = canonical_identifier(source_revision, "source_revision")
    role = canonical_identifier(kind, "kind")
    identifier = (
        canonical_identifier(entity_id, "entity_id")
        if isinstance(entity_id, str)
        else nonnegative_integer(entity_id, "entity_id")
    )
    if isinstance(occurrence_path, (str, bytes)):
        raise TypeError("occurrence_path must contain occurrence identifiers.")
    path = tuple(
        canonical_identifier(value, "occurrence_id") for value in occurrence_path
    )
    return plain_json_fingerprint(
        {
            "kind": "geometry-source-entity",
            "source": source,
            "revision": revision,
            "entity_kind": role,
            "entity": identifier,
            "occurrence": path,
        }
    )


def source_entity_ids(
    source_id: str,
    source_revision: str,
    kind: str,
    entity_ids: Iterable[str | int],
    /,
) -> tuple[str, ...]:
    """``source_entity_id`` of many entities of one source and kind, in order.

    Without occurrence paths the canonical text of each key differs only in
    its entity, so the shared text is validated and encoded once.
    """
    source = canonical_identifier(source_id, "source_id")
    revision = canonical_identifier(source_revision, "source_revision")
    role = canonical_identifier(kind, "kind")
    # Sorted keys: entity, entity_kind, kind, occurrence, revision, source.
    suffix = canonical_json(
        {
            "entity_kind": role,
            "kind": "geometry-source-entity",
            "occurrence": (),
            "revision": revision,
            "source": source,
        }
    )[1:]
    keys = []
    for entity_id in entity_ids:
        # A plain nonnegative int's JSON text is its decimal repr.
        if type(entity_id) is int and entity_id >= 0:
            text = str(entity_id)
        elif isinstance(entity_id, str):
            text = canonical_json(canonical_identifier(entity_id, "entity_id"))
        else:
            text = str(nonnegative_integer(entity_id, "entity_id"))
        keys.append(
            hashlib.sha256(('{"entity":' + text + "," + suffix).encode("ascii")).hexdigest()
        )
    return tuple(keys)
