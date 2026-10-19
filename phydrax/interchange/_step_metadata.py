#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Authored STEP presentation metadata, separate from geometric identity."""

from __future__ import annotations

from typing import assert_never, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..geometry.brep._model import BRepEntityId, BRepModel
from ..typing import checked, parse


if TYPE_CHECKING:
    from ..geometry.brep._selection import BRepSelection

StepMetadataKind: TypeAlias = Literal["name", "color", "layer"]
StepSelectionFamily: TypeAlias = Literal["face", "edge"]


@final
class StepMetadataConflict(ValueError):
    """An authored label identifies multiple native entities."""

    def __init__(self, entities: tuple[BRepEntityId, ...], /) -> None:
        self.entities = entities
        super().__init__(f"STEP metadata label is not unique: {entities!r}")


@final
class StepEntityLabels(StrictModule):
    """Authored presentation labels for one native entity."""

    entity: BRepEntityId = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    colors: tuple[tuple[float, float, float], ...] = eqx.field(static=True)
    layers: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        entity: BRepEntityId,
        /,
        *,
        names: tuple[str, ...] = (),
        colors: tuple[tuple[float, float, float], ...] = (),
        layers: tuple[str, ...] = (),
    ) -> None:
        for labels in (names, layers):
            if any(not isinstance(label, str) or not label for label in labels):
                raise ValueError(
                    f"StepEntityLabels.names must contain nonempty strings, as must layers; got {labels!r}"
                )
        if any(
            len(color) != 3 or any(not np.isfinite(c) or not 0 <= c <= 1 for c in color)
            for color in colors
        ):
            raise ValueError(
                f"StepEntityLabels.colors must contain finite RGB triples in [0, 1]; got {colors!r}"
            )
        self.entity = entity
        self.names = tuple(sorted(set(names)))
        self.colors = tuple(sorted(set(colors)))
        self.layers = tuple(sorted(set(layers)))


@final
class StepEntityMetadata(StrictModule):
    """Entity-keyed labels in model order, outside model identity."""

    model_id: str = eqx.field(static=True)
    rows: tuple[StepEntityLabels, ...] = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)

    @checked
    def __init__(self, model: BRepModel, rows: tuple[StepEntityLabels, ...], /) -> None:
        entities = (*model.solid_ids, *model.face_ids, *model.edge_ids, *model.vertex_ids)
        by_entity = {row.entity: row for row in rows}
        if len(by_entity) != len(rows) or any(
            entity not in entities for entity in by_entity
        ):
            raise ValueError(
                f"StepEntityMetadata.rows must uniquely identify bound model entities; got {[row.entity for row in rows]!r}"
            )
        self.model_id = model.model_id
        self.rows = tuple(by_entity[entity] for entity in entities if entity in by_entity)
        self.metadata_id = canonical_fingerprint(
            {
                "kind": "step-entity-metadata",
                "model": model.model_id,
                "rows": [
                    (row.entity.kind, row.entity.index, row.names, row.colors, row.layers)
                    for row in self.rows
                ],
            }
        )

    def entities(
        self,
        kind: StepMetadataKind,
        label: str | tuple[float, float, float],
        /,
        *,
        unique: bool = False,
    ) -> tuple[BRepEntityId, ...]:
        """Resolve authored labels, optionally refusing a nonunique name."""
        kind = parse(kind, StepMetadataKind, "kind")
        selected = []
        for row in self.rows:
            match kind:
                case "name":
                    matches = label in row.names
                case "color":
                    matches = label in row.colors
                case "layer":
                    matches = label in row.layers
                case _:
                    assert_never(kind)
            if matches:
                selected.append(row.entity)
        entities = tuple(selected)
        if unique and len(entities) > 1:
            raise StepMetadataConflict(entities)
        return entities


def _expand_solid_labels(
    model: BRepModel, entities: tuple[BRepEntityId, ...], /
) -> set[BRepEntityId]:
    expanded = set(entities)
    if model.geometry is None:
        return expanded
    for entity in entities:
        if entity.kind == "solid":
            for shell in model.geometry.solid_shells[entity.index]:
                expanded.update(
                    model.face_ids[index] for index in model.geometry.shell_faces[shell]
                )
    return expanded


def select_step_entities(
    model: BRepModel,
    metadata: StepEntityMetadata,
    kind: StepMetadataKind,
    label: str | tuple[float, float, float],
    /,
    *,
    family: StepSelectionFamily = "face",
    unique: bool | None = None,
) -> BRepSelection:
    """Produce a named-region selection from authored STEP labels.

    Names require uniqueness by default; colors and layers select sets. Solid
    labels expand through explicit solid-face incidence, never proximity.
    """
    from ..geometry.brep._selection import BRepSelection

    if metadata.model_id != model.model_id:
        raise ValueError(
            f"StepEntityMetadata.model_id must equal the selected model ID; got {metadata.model_id!r}, expected={model.model_id!r}"
        )
    kind = parse(kind, StepMetadataKind, "kind")
    family = parse(family, StepSelectionFamily, "family")
    entities = metadata.entities(
        kind, label, unique=kind == "name" if unique is None else unique
    )
    expanded = (
        _expand_solid_labels(model, entities) if family == "face" else set(entities)
    )
    inventory = model.face_ids if family == "face" else model.edge_ids
    return BRepSelection(
        model,
        tuple(entity for entity in inventory if entity in expanded),
        tuple(entity for entity in inventory if entity not in expanded),
        (),
        provenance=(f"STEP exact {kind}: {label!r}",),
    )
