#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Recipe identity transported only by exhaustive native topology lineage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx

from ..._strict import StrictModule
from ..._validation import normalized_identifier
from ...typing import checked, parse
from .._cad_revision import AssociationGraph, CADOccurrence, CADRevision, CADSelectionSet
from ..brep._model import BRepEntityId, BRepModel


if TYPE_CHECKING:
    from ..brep._partition import BRepPartitionResult
    from ..brep._sewing import BRepSewingLineage


RecipeEntityKind: TypeAlias = Literal["face", "edge"]


class RecipeNamingError(ValueError):
    """Recipe identity is unsupported, incomplete, split, merged, or ambiguous."""


class RecipeEntityName(StrictModule):
    """Geometry-independent identity declared by a recipe, not an importer index."""

    kind: RecipeEntityKind = eqx.field(static=True)
    region: str = eqx.field(static=True)
    source_entity: str = eqx.field(static=True)
    lineage_path: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        kind: RecipeEntityKind,
        region: str,
        source_entity: str,
        lineage_path: Sequence[str],
    ) -> None:
        kind_ = parse(kind, RecipeEntityKind, "RecipeEntityName.kind")
        region_ = normalized_identifier(region, "RecipeEntityName.region")
        source_ = normalized_identifier(source_entity, "RecipeEntityName.source_entity")
        path = tuple(
            normalized_identifier(step, "RecipeEntityName.lineage_path")
            for step in lineage_path
        )
        if not path:
            raise ValueError("RecipeEntityName.lineage_path must be non-empty.")
        self.kind = kind_
        self.region = region_
        self.source_entity = source_
        self.lineage_path = path

    @property
    def key(self) -> tuple[str, str, str, tuple[str, ...]]:
        return self.kind, self.region, self.source_entity, self.lineage_path

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RecipeEntityName) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


def _name_family(
    names: tuple[RecipeEntityName, ...] | None,
    kind: RecipeEntityKind,
    count: int,
    /,
) -> tuple[RecipeEntityName, ...] | None:
    if names is None:
        return None
    names_ = tuple(names)
    if len(names_) != count:
        raise RecipeNamingError(
            f"RecipeEntityNames.names must exhaust the model inventory; got {len(names_)} names for {count} {kind} entities."
        )
    if any(
        not isinstance(name, RecipeEntityName) or name.kind != kind for name in names_
    ):
        raise RecipeNamingError(
            f"RecipeEntityNames.names must contain {kind} names; got a mismatched name kind."
        )
    if len(set(names_)) != len(names_):
        raise RecipeNamingError(
            f"RecipeEntityNames.names must identify distinct entities; got ambiguous identity in {kind} names."
        )
    return names_


class RecipeEntityNames(StrictModule):
    """Bind stable recipe values to one model's face and edge entities.

    ``None`` explicitly refuses a whole family whose exhaustive lineage is not
    available. An empty tuple means a supported family with zero entities.
    """

    model_id: str = eqx.field(static=True)
    face_ids: tuple[BRepEntityId, ...] = eqx.field(static=True)
    edge_ids: tuple[BRepEntityId, ...] = eqx.field(static=True)
    face_names: tuple[RecipeEntityName, ...] | None = eqx.field(static=True)
    edge_names: tuple[RecipeEntityName, ...] | None = eqx.field(static=True)
    num_vertices: int = eqx.field(static=True)
    num_coedges: int | None = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: BRepModel,
        face_names: tuple[RecipeEntityName, ...] | None,
        edge_names: tuple[RecipeEntityName, ...] | None,
        /,
    ) -> None:
        faces = _name_family(face_names, "face", model.topology.num_faces)
        edges = _name_family(edge_names, "edge", model.topology.num_edges)
        self.model_id = model.model_id
        self.face_ids = model.face_ids
        self.edge_ids = model.edge_ids
        self.face_names = faces
        self.edge_names = edges
        self.num_vertices = model.topology.num_vertices
        self.num_coedges = (
            None if model.geometry is None else len(model.geometry.coedge_edges)
        )

    def _family(
        self, kind: RecipeEntityKind, /
    ) -> tuple[tuple[BRepEntityId, ...], tuple[RecipeEntityName, ...]]:
        match kind:
            case "face":
                ids, names = self.face_ids, self.face_names
            case "edge":
                ids, names = self.edge_ids, self.edge_names
            case _:
                raise RecipeNamingError(
                    f"RecipeEntityNames.kind must be face or edge; got {kind!r}."
                )
        if names is None:
            raise RecipeNamingError(
                f"RecipeEntityNames.{kind} must have exhaustive lineage; got unsupported or incomplete {kind} lineage."
            )
        return ids, names

    @checked
    def resolve(self, name: RecipeEntityName, /) -> BRepEntityId:
        """Resolve a recipe binding in this realization, without geometric fallback."""
        ids, names = self._family(name.kind)
        for entity, candidate in zip(ids, names, strict=True):
            if candidate == name:
                return entity
        raise KeyError(
            f"RecipeEntityNames.resolve.name must identify a bound entity; got {name.key!r}."
        )

    @checked
    def name(self, entity_id: BRepEntityId, /) -> RecipeEntityName:
        """Read an entity name; foreign revisions never match."""
        kind = parse(
            entity_id.kind,
            RecipeEntityKind,
            "RecipeEntityNames.entity_id.kind",
        )
        ids, names = self._family(kind)
        for entity, candidate in zip(ids, names, strict=True):
            if entity == entity_id:
                return candidate
        raise KeyError(
            "RecipeEntityNames.name.entity_id must identify an entity from this model; got a foreign or unbound entity."
        )

    @checked
    def select_occurrence(
        self,
        name: RecipeEntityName,
        revision: CADRevision,
        occurrence_id: str,
        qualified_entity: BRepEntityId,
        /,
    ) -> CADSelectionSet:
        """Bind a recipe name to one explicitly qualified CAD occurrence.

        The caller supplies the occurrence path and revision inventory; neither a
        display label nor a geometric match can choose an assembly occurrence.
        The resulting set declares descendant-set intent for graph transport.
        """
        definition = self.resolve(name)
        if (
            qualified_entity.source_id != definition.source_id
            or qualified_entity.source_revision != definition.source_revision
            or qualified_entity.kind != definition.kind
            or qualified_entity.index != definition.index
        ):
            raise RecipeNamingError(
                "Qualified occurrence does not belong to the named definition."
            )
        occurrence = revision.occurrence(occurrence_id)
        if occurrence.kind != name.kind or occurrence.entity_id != _entity_text(
            qualified_entity
        ):
            raise RecipeNamingError(
                "Qualified occurrence does not match the CAD revision inventory."
            )
        return CADSelectionSet(revision.revision_id, (revision.select(occurrence_id),))


def _bijective_lineage(graph: AssociationGraph, /) -> dict[str, str]:
    transaction = graph.transaction
    if (
        not transaction.coverage.source_exhaustive
        or not transaction.coverage.target_exhaustive
    ):
        raise RecipeNamingError(
            "RecipeEntityNames.partition.coverage must be exhaustive; got non-exhaustive source or target lineage."
        )
    if transaction.unresolved_source_ids or transaction.unresolved_target_ids:
        raise RecipeNamingError(
            "RecipeEntityNames.partition.lineage must be resolved; got unresolved lineage."
        )
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for pair in transaction.correspondences:
        if pair.source_occurrence_id in forward:
            raise RecipeNamingError(
                "RecipeEntityNames.partition.lineage must be bijective; got split lineage."
            )
        if pair.target_occurrence_id in reverse:
            raise RecipeNamingError(
                "RecipeEntityNames.partition.lineage must be bijective; got merge lineage."
            )
        forward[pair.source_occurrence_id] = pair.target_occurrence_id
        reverse[pair.target_occurrence_id] = pair.source_occurrence_id
    if set(forward) != {
        occurrence.occurrence_id for occurrence in graph.source_revision.occurrences
    }:
        raise RecipeNamingError(
            "RecipeEntityNames.partition.lineage must exhaust sources; got deleted or incomplete source lineage."
        )
    if set(reverse) != {
        occurrence.occurrence_id for occurrence in graph.target_revision.occurrences
    }:
        raise RecipeNamingError(
            "RecipeEntityNames.partition.lineage must exhaust targets; got created or incomplete target lineage."
        )
    return reverse


def _entity_text(entity: BRepEntityId, /) -> str:
    from ..brep._projection_contracts import brep_entity_id, BRepEntityDimension

    match entity.kind:
        case "vertex":
            dimension = BRepEntityDimension.VERTEX
        case "edge":
            dimension = BRepEntityDimension.EDGE
        case "face":
            dimension = BRepEntityDimension.FACE
        case "solid":
            dimension = BRepEntityDimension.SOLID
        case _:
            raise RecipeNamingError("BRep entity kind is unsupported for recipe naming.")
    return brep_entity_id(
        entity.source_id,
        entity.source_revision,
        dimension,
        entity.index,
        occurrence_path=entity.occurrence_path,
    )


def _partition_family(
    model: BRepModel,
    kind: RecipeEntityKind,
    occurrences: tuple[CADOccurrence, ...],
    reverse: Mapping[str, str],
    source_names: Mapping[str, RecipeEntityName],
    /,
) -> tuple[RecipeEntityName, ...]:
    entities = model.face_ids if kind == "face" else model.edge_ids
    names_by_entity: dict[str, RecipeEntityName] = {}
    for occurrence in occurrences:
        if occurrence.kind != kind:
            continue
        source_name = source_names[reverse[occurrence.occurrence_id]]
        transported = RecipeEntityName(
            kind,
            source_name.region,
            source_name.source_entity,
            (*source_name.lineage_path, "partition"),
        )
        previous = names_by_entity.get(occurrence.entity_id)
        if previous is not None and previous != transported:
            raise RecipeNamingError(
                "RecipeEntityNames.partition.occurrence must identify one recipe value; got ambiguous naming on a shared entity."
            )
        names_by_entity[occurrence.entity_id] = transported
    if set(names_by_entity) != {_entity_text(entity) for entity in entities}:
        raise RecipeNamingError(
            f"RecipeEntityNames.partition.{kind} must have exhaustive occurrence inventory; got missing or foreign entity lineage."
        )
    return tuple(names_by_entity[_entity_text(entity)] for entity in entities)


def name_partition_entities(
    partition: BRepPartitionResult,
    source_names: Mapping[str, RecipeEntityName],
    /,
) -> RecipeEntityNames:
    """Name bijective native partition families from authored source occurrences.

    Keys are source CAD occurrence IDs, never inferred from display labels. The
    current 3D partition substrate certifies faces but not edges, so it leaves
    ``edge_names=None``. Planar partition lineage can name both families.
    Split/merge, deletion/creation, missing source names, and shared-entity
    ambiguity are explicit refusals.
    """
    from ..brep._partition import BRepPartitionResult

    if not isinstance(partition, BRepPartitionResult):
        raise TypeError("partition must be a BRepPartitionResult.")
    graph = partition.association_graph
    reverse = _bijective_lineage(graph)
    sources: dict[str, CADOccurrence] = {}
    for occurrence in graph.source_revision.occurrences:
        match occurrence.kind:
            case "face" | "edge":
                sources[occurrence.occurrence_id] = occurrence
            case "solid":
                continue
            case _:
                raise RecipeNamingError(
                    f"RecipeEntityNames.partition.source.kind is unsupported; got {occurrence.kind!r}."
                )
    declared_regions = {region.name for region in partition.regions}
    if set(source_names) != set(sources):
        raise RecipeNamingError(
            "RecipeEntityNames.partition.source_names must exhaust source face and edge occurrences."
        )
    for occurrence_id, name in source_names.items():
        source = sources[occurrence_id]
        if not isinstance(name, RecipeEntityName) or name.kind != source.kind:
            raise RecipeNamingError(
                "RecipeEntityNames.partition.source_names.kind must match its source occurrence."
            )
        if name.region not in declared_regions:
            raise RecipeNamingError(
                "RecipeEntityNames.partition.source_names.region must identify a declared region."
            )
    faces = _partition_family(
        partition.model,
        "face",
        graph.target_revision.occurrences,
        reverse,
        source_names,
    )
    edges = (
        _partition_family(
            partition.model,
            "edge",
            graph.target_revision.occurrences,
            reverse,
            source_names,
        )
        if partition.topological_dimension == 2
        else None
    )
    face_regions = (
        {
            entity: (region.name,)
            for region in partition.regions
            for entity in region.entity_ids
        }
        if partition.topological_dimension == 2
        else {
            entity: patch.adjacent_region_ids
            for patch in partition.patches
            for entity in patch.entity_ids
        }
    )
    for entity, name in zip(partition.model.face_ids, faces, strict=True):
        if name.region not in face_regions[entity]:
            raise RecipeNamingError(
                "RecipeEntityNames.partition.face.region must match target region incidence."
            )
    if edges is not None:
        edge_regions = {
            entity: patch.adjacent_region_ids
            for patch in partition.patches
            for entity in patch.entity_ids
        }
        for entity, name in zip(partition.model.edge_ids, edges, strict=True):
            if name.region not in edge_regions[entity]:
                raise RecipeNamingError(
                    "RecipeEntityNames.partition.edge.region must match target region incidence."
                )
    return RecipeEntityNames(partition.model, faces, edges)


def _sewing_incidence(
    source: RecipeEntityNames,
    target: BRepModel,
    lineage: BRepSewingLineage,
    /,
) -> None:
    if target.geometry is None or source.num_coedges is None:
        raise RecipeNamingError(
            "RecipeEntityNames.sewing.incidence must include source and target coedges."
        )
    if len(lineage.vertex_sources) != target.topology.num_vertices:
        raise RecipeNamingError(
            "RecipeEntityNames.sewing.vertices must have exhaustive target lineage."
        )
    vertices = tuple(vertex for members in lineage.vertex_sources for vertex in members)
    if any(len(members) != 1 for members in lineage.vertex_sources) or sorted(
        vertices
    ) != list(range(source.num_vertices)):
        raise RecipeNamingError(
            "RecipeEntityNames.sewing.vertices must be bijective; got incomplete or merged lineage."
        )
    if len(lineage.coedge_sources) != len(target.geometry.coedge_edges) or sorted(
        lineage.coedge_sources
    ) != list(range(source.num_coedges)):
        raise RecipeNamingError(
            "RecipeEntityNames.sewing.coedges must be bijective; got incomplete or split lineage."
        )


def _sewn_edges(
    source: RecipeEntityNames,
    target: BRepModel,
    lineage: BRepSewingLineage,
    /,
) -> tuple[RecipeEntityName, ...] | None:
    _sewing_incidence(source, target, lineage)
    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for image in lineage.edge_images:
        if image.scale == 0:
            raise RecipeNamingError(
                "RecipeEntityNames.sewing.parameter_map must have nonzero scale."
            )
        if image.source_edge in forward:
            raise RecipeNamingError(
                "RecipeEntityNames.sewing.edges must be bijective; got split lineage."
            )
        if image.target_edge in reverse:
            raise RecipeNamingError(
                "RecipeEntityNames.sewing.edges must be bijective; got merge lineage."
            )
        forward[image.source_edge] = image.target_edge
        reverse[image.target_edge] = image.source_edge
    if set(forward) != set(range(len(source.edge_ids))) or set(reverse) != set(
        range(target.topology.num_edges)
    ):
        raise RecipeNamingError(
            "RecipeEntityNames.sewing.edges must have exhaustive lineage."
        )
    if source.edge_names is None:
        return None
    return tuple(
        RecipeEntityName(
            "edge", name.region, name.source_entity, (*name.lineage_path, "sewing")
        )
        for index in range(target.topology.num_edges)
        for name in (source.edge_names[reverse[index]],)
    )


def transport_recipe_names(
    source: RecipeEntityNames,
    target: BRepModel,
    lineage: BRepSewingLineage,
    /,
) -> RecipeEntityNames:
    """Transport sewing's index-preserving faces and bijective edge lineage."""
    from ..brep._sewing import BRepSewingLineage

    if not isinstance(source, RecipeEntityNames):
        raise TypeError("source must be RecipeEntityNames.")
    if not isinstance(target, BRepModel):
        raise TypeError("target must be a BRepModel.")
    if not isinstance(lineage, BRepSewingLineage):
        raise TypeError("lineage must be a BRepSewingLineage.")
    if len(source.face_ids) != target.topology.num_faces:
        raise RecipeNamingError(
            "RecipeEntityNames.sewing.faces must preserve the face inventory."
        )
    edges = _sewn_edges(source, target, lineage)
    faces = (
        None
        if source.face_names is None
        else tuple(
            RecipeEntityName(
                "face", name.region, name.source_entity, (*name.lineage_path, "sewing")
            )
            for name in source.face_names
        )
    )
    return RecipeEntityNames(target, faces, edges)


__all__ = [
    "RecipeEntityKind",
    "RecipeEntityName",
    "RecipeEntityNames",
    "RecipeNamingError",
    "name_partition_entities",
    "transport_recipe_names",
]
