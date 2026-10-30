#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Identity of declared source entities, independent of numerical storage rows."""

from __future__ import annotations

from .._fingerprint import canonical_fingerprint
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
    return canonical_fingerprint(
        {
            "kind": "geometry-source-entity",
            "source": source,
            "revision": revision,
            "entity_kind": role,
            "entity": identifier,
            "occurrence": path,
        }
    )
