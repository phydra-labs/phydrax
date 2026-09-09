#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _text(value: str, name: str, /, *, optional: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not optional and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    result = tuple(_text(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


class DataLineage(StrictModule, NonTrainableState):
    """Data-only provenance kept separate from model and numerical evidence."""

    source_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    publisher_id: str = eqx.field(static=True)
    upstream_lineage_ids: tuple[str, ...] = eqx.field(static=True)
    transformation_ids: tuple[str, ...] = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        dataset_id: str,
        /,
        *,
        publisher_id: str = "",
        upstream_lineage_ids: Sequence[str] = (),
        transformation_ids: Sequence[str] = (),
    ):
        source = _text(source_id, "source_id")
        dataset = _text(dataset_id, "dataset_id")
        publisher = _text(publisher_id, "publisher_id", optional=True)
        upstream = tuple(
            sorted(_identifiers(upstream_lineage_ids, "upstream_lineage_ids"))
        )
        transformations = _identifiers(transformation_ids, "transformation_ids")
        self.source_id = source
        self.dataset_id = dataset
        self.publisher_id = publisher
        self.upstream_lineage_ids = upstream
        self.transformation_ids = transformations
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "financial-data-lineage",
                "source_id": source,
                "dataset_id": dataset,
                "publisher_id": publisher,
                "upstream_lineage_ids": list(upstream),
                "transformation_ids": list(transformations),
            }
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> DataLineage:
        value = cls(
            str(record["source_id"]),
            str(record["dataset_id"]),
            publisher_id=str(record["publisher_id"]),
            upstream_lineage_ids=tuple(
                str(item) for item in record["upstream_lineage_ids"]
            ),
            transformation_ids=tuple(str(item) for item in record["transformation_ids"]),
        )
        if "lineage_id" in record and str(record["lineage_id"]) != value.lineage_id:
            raise ValueError("Data-lineage record content does not match lineage_id.")
        return value

    @classmethod
    def derived(
        cls,
        transformation_id: str,
        upstream: Sequence[DataLineage],
        /,
        *,
        source_id: str,
        dataset_id: str,
        publisher_id: str = "",
    ) -> DataLineage:
        values = tuple(upstream)
        if not values or not all(isinstance(value, DataLineage) for value in values):
            raise TypeError("upstream must contain at least one DataLineage.")
        transform = _text(transformation_id, "transformation_id")
        unique_ids = tuple(sorted({value.lineage_id for value in values}))
        return cls(
            source_id,
            dataset_id,
            publisher_id=publisher_id,
            upstream_lineage_ids=unique_ids,
            transformation_ids=(transform,),
        )

    def to_record(self) -> Mapping[str, Any]:
        return {
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "publisher_id": self.publisher_id,
            "upstream_lineage_ids": list(self.upstream_lineage_ids),
            "transformation_ids": list(self.transformation_ids),
            "lineage_id": self.lineage_id,
        }


__all__ = ["DataLineage"]
