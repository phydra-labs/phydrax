#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer
from ..discretization import CellMesh, EntitySelection, EntitySet
from ..geometry._cad_revision import CADSelector
from ..geometry._source_identity import source_entity_id
from ..geometry.brep._model import BRepModel
from ..typing import checked, ConvertibleToArray


if TYPE_CHECKING:
    from ..geometry._cad_revision import CADSelectionSet
    from ..geometry.brep._differentiable import FixedTopologyBRepSource
    from ..geometry.brep._model import BRepEntityId
    from ..geometry.brep._source import BRepSource
    from ..geometry.multiregion_surface._label_extraction import (
        LabelFieldSurfaceExtractionResult,
    )
    from ..geometry.surface import SurfaceModel


@runtime_checkable
class _SourceEntityScopes(Protocol):
    """An authoritative geometry source's prepared entity-ID projection."""

    @property
    def source_id(self) -> str: ...

    @property
    def source_revision(self) -> str: ...

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray: ...

    def scope_entity_set_id(self, dimension: int, /) -> str: ...

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]: ...


def _geometry_model(
    source: BRepModel
    | BRepSource
    | FixedTopologyBRepSource
    | LabelFieldSurfaceExtractionResult
    | _SourceEntityScopes,
    /,
) -> BRepModel:
    from ..geometry.brep._differentiable import FixedTopologyBRepSource
    from ..geometry.brep._model import BRepModel
    from ..geometry.brep._source import BRepSource

    if isinstance(source, BRepModel):
        return source
    if isinstance(source, (BRepSource, FixedTopologyBRepSource)):
        return source.model
    raise TypeError("Geometry scopes require a BRep model or source.")


def _source_scope_owner(
    source: BRepModel
    | BRepSource
    | FixedTopologyBRepSource
    | LabelFieldSurfaceExtractionResult
    | _SourceEntityScopes,
    /,
) -> _SourceEntityScopes:
    """Resolve source-owned entity banks without importing an external engine."""
    if isinstance(source, _SourceEntityScopes):
        return source
    from ..geometry.brep._differentiable import FixedTopologyBRepSource
    from ..geometry.brep._model import BRepModel
    from ..geometry.brep._source import BRepSource

    if isinstance(source, (BRepModel, BRepSource, FixedTopologyBRepSource)):
        from ..geometry._meshing_domain import MeshingDomain

        return MeshingDomain.from_brep(_geometry_model(source))
    from ..geometry.multiregion_surface._label_extraction import (
        LabelFieldSurfaceExtractionResult,
    )

    if isinstance(source, LabelFieldSurfaceExtractionResult):
        from .providers._native_compartment import _normalize_label_field_source

        return _normalize_label_field_source(source).source
    raise TypeError("Geometry scopes require a declared source entity owner.")


def _brep_scope_dimension(kind: str, /) -> int:
    match kind:
        case "vertex":
            return 0
        case "edge":
            return 1
        case "face":
            return 2
        case "solid":
            return 3
        case _:
            raise ValueError("Unsupported BRep entity kind.")


def _brep_scope_ids(
    owner: _SourceEntityScopes,
    entities: tuple[BRepEntityId, ...],
    dimension: int | None,
    /,
) -> tuple[int, np.ndarray]:
    """Resolve definition selectors only through unique declared occurrence uses."""
    target = _brep_scope_dimension(entities[0].kind)
    if dimension is not None and nonnegative_integer(dimension, "dimension") != target:
        raise ValueError("The requested dimension disagrees with the BRep entity kind.")
    if any(
        entity.source_id != owner.source_id
        or entity.source_revision != owner.source_revision
        or entity.kind != entities[0].kind
        for entity in entities
    ):
        raise ValueError(
            "Geometry scopes require one entity kind from the actual source owner and revision."
        )
    bank = {
        owner.source_entity_key_for_scope(target, int(identifier)): int(identifier)
        for identifier in owner.scope_entity_ids(target)
    }
    paths = tuple(sorted({path for _, path in bank}))
    selected: list[int] = []
    for entity in entities:
        candidate_paths = (entity.occurrence_path,) if entity.occurrence_path else paths
        matches = []
        for path in candidate_paths:
            key = source_entity_id(
                entity.source_id,
                entity.source_revision,
                entity.kind,
                entity.index,
                occurrence_path=path,
            )
            if (key, path) in bank:
                matches.append(bank[(key, path)])
        if len(matches) != 1:
            raise ValueError(
                "A BRep selector must resolve to exactly one declared occurrence; specify its occurrence path."
            )
        selected.append(matches[0])
    return target, np.asarray(selected, dtype=np.int64)


def _cad_scope_ids(
    source: BRepModel
    | BRepSource
    | FixedTopologyBRepSource
    | LabelFieldSurfaceExtractionResult
    | _SourceEntityScopes,
    owner: _SourceEntityScopes,
    selection: CADSelectionSet,
    dimension: int | None,
    /,
) -> tuple[int, np.ndarray]:
    """Authenticate selectors against the model inventory before projecting them."""
    from ..geometry._meshing_domain import _placed_copy_keys
    from ..geometry.brep._partition import (
        _entity_occurrence_id,
        cad_revision_from_brep_model,
    )

    model = _geometry_model(source)
    revision = cad_revision_from_brep_model(model)
    if selection.revision_id != revision.revision_id:
        raise ValueError("CAD selection belongs to another source revision.")
    if not selection.selectors:
        raise ValueError("Meshing scopes must contain at least one entity.")
    incidence = model.geometry.qualified_entity_incidence(
        model.source_id, model.source_revision
    )
    entities = {
        _entity_occurrence_id(record.member): record.member for record in incidence
    }
    selected = []
    for selector in selection.selectors:
        try:
            stored = revision.select(selector.occurrence_id)
            entity = entities[selector.occurrence_id]
        except KeyError as exc:
            raise ValueError(
                "CAD selector is not a declared BRep entity occurrence."
            ) from exc
        if selector != stored:
            raise ValueError("CAD selector does not match the source revision inventory.")
        selected.append(entity)
    # A domain slot represents the entire shared entity, not one incident
    # solid's side.  Its canonical path is only the first occurrence in the
    # placed copy.  Project a complete set of uses to that slot, but never
    # silently widen a single side to all of them.
    occurrences = tuple(model.geometry.assembly_graph.iter_occurrences())
    copies = _placed_copy_keys(occurrences)
    occurrences_by_path = {occurrence.path: occurrence for occurrence in occurrences}
    placed_members: dict[tuple, dict[int, set[tuple[str, ...]]]] = {}
    for entity in entities.values():
        occurrence = occurrences_by_path.get(entity.occurrence_path)
        if occurrence is not None:
            key = (entity.kind, entity.index, occurrence.placement.matrix)
            placed_members.setdefault(key, {}).setdefault(
                occurrence.solid_index, set()
            ).add(occurrence.path)
    for entity in selected:
        occurrence = occurrences_by_path.get(entity.occurrence_path)
        if occurrence is None:
            continue
        members = placed_members[(entity.kind, entity.index, occurrence.placement.matrix)]
        if len(members) > 1 and any(len(paths) > 1 for paths in members.values()):
            raise ValueError(
                "CAD selection has ambiguous coincident assembly copies; source "
                "slot pairing by occurrence order cannot preserve exact identity."
            )

    def group_key(entity: BRepEntityId) -> tuple:
        return (
            entity.kind,
            entity.index,
            copies.get(entity.occurrence_path, entity.occurrence_path),
        )

    groups: dict[tuple, set[BRepEntityId]] = {}
    for entity in entities.values():
        groups.setdefault(group_key(entity), set()).add(entity)
    selected_set = set(selected)
    for entity in selected:
        if not groups[group_key(entity)] <= selected_set:
            raise ValueError(
                "A meshing scope cannot represent only one occurrence side of a "
                "shared source entity; select all incident occurrences instead."
            )

    target = _brep_scope_dimension(selected[0].kind)
    if dimension is not None and nonnegative_integer(dimension, "dimension") != target:
        raise ValueError("The requested dimension disagrees with the BRep entity kind.")
    if any(entity.kind != selected[0].kind for entity in selected):
        raise ValueError("Geometry scopes require one entity kind.")
    bank = {
        owner.source_entity_key_for_scope(target, int(identifier)): int(identifier)
        for identifier in owner.scope_entity_ids(target)
    }
    identifiers: set[int] = set()
    for entity in selected:
        matches = set()
        for member in groups[group_key(entity)]:
            path = member.occurrence_path
            key = source_entity_id(
                member.source_id,
                member.source_revision,
                member.kind,
                member.index,
                occurrence_path=path,
            )
            if (key, path) in bank:
                matches.add(bank[(key, path)])
        if len(matches) != 1:
            raise ValueError(
                "CAD selection must resolve to one declared source entity slot."
            )
        identifiers.update(matches)
    return target, np.asarray(sorted(identifiers), dtype=np.int64)


def _geometry_scope_ids(
    owner: _SourceEntityScopes,
    entity_ids: Sequence[BRepEntityId] | BRepEntityId | ArrayLike,
    dimension: int | None,
    /,
) -> tuple[int, np.ndarray]:
    from ..geometry.brep._model import BRepEntityId

    if isinstance(entity_ids, BRepEntityId):
        return _brep_scope_ids(owner, (entity_ids,), dimension)
    if isinstance(entity_ids, Sequence) and any(
        isinstance(value, BRepEntityId) for value in entity_ids
    ):
        selectors: list[BRepEntityId] = []
        for value in entity_ids:
            if not isinstance(value, BRepEntityId):
                raise TypeError("BRep selectors cannot be mixed with numeric scope IDs.")
            selectors.append(value)
        return _brep_scope_ids(owner, tuple(selectors), dimension)
    if dimension is None:
        raise ValueError("A source entity bank requires an explicit dimension.")
    target = nonnegative_integer(dimension, "dimension")
    identifiers = np.atleast_1d(np.asarray(entity_ids))
    if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
        raise TypeError("Source entity IDs must form one integer vector.")
    if not np.all(np.isin(identifiers, owner.scope_entity_ids(target))):
        raise ValueError("Scope IDs must belong to the declared source entity bank.")
    return target, identifiers


class MeshingEntityKind(StrEnum):
    GEOMETRY = "geometry"
    MESH = "mesh"
    PART = "part"
    LABEL = "label"
    ZONE = "zone"


@dataclass(frozen=True, slots=True)
class MeshingScopeSide:
    """Exact source face occurrence and its incident solid and orientation."""

    entity_id: int
    face_index: int
    face: CADSelector
    solid_index: int
    solid: CADSelector
    region_path: tuple[str, ...]
    orientation: int

    def __post_init__(self) -> None:
        for name in ("entity_id", "face_index", "solid_index"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer.")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.face, CADSelector) or not isinstance(
            self.solid, CADSelector
        ):
            raise TypeError("Side face and solid must be CADSelector values.")
        if (
            self.face.kind != "face"
            or self.solid.kind != "solid"
            or self.face.revision_id != self.solid.revision_id
        ):
            raise ValueError("Side selectors must be a face and solid of one revision.")
        if not isinstance(self.region_path, tuple) or any(
            not isinstance(part, str) or not part for part in self.region_path
        ):
            raise ValueError(
                "Side region_path must be a tuple of occurrence identifiers."
            )
        if (
            isinstance(self.orientation, bool)
            or not isinstance(self.orientation, (int, np.integer))
            or self.orientation not in (-1, 1)
        ):
            raise ValueError("Side orientation must be -1 or 1.")
        object.__setattr__(self, "orientation", int(self.orientation))


def _side_key(side: MeshingScopeSide) -> tuple:
    return (
        side.entity_id,
        side.face_index,
        side.face.revision_id,
        side.face.occurrence_id,
        side.face.kind,
        side.face.path,
        side.face.entity_id,
        side.solid_index,
        side.solid.revision_id,
        side.solid.occurrence_id,
        side.solid.kind,
        side.solid.path,
        side.solid.entity_id,
        side.region_path,
        side.orientation,
    )


def _cad_face_sides(
    source: BRepModel
    | BRepSource
    | FixedTopologyBRepSource
    | LabelFieldSurfaceExtractionResult
    | _SourceEntityScopes,
    selection: CADSelectionSet,
    /,
) -> tuple[MeshingScopeSide, ...]:
    """Resolve through the domain's authored region/patch incidence, not copy order."""
    from ..geometry._meshing_domain import MeshingDomain
    from ..geometry.brep._partition import (
        _entity_occurrence_id,
        cad_revision_from_brep_model,
    )

    model = _geometry_model(source)
    domain = _source_scope_owner(source)
    if not isinstance(domain, MeshingDomain):
        raise TypeError("CAD sides require the source's BRep meshing domain.")
    revision = cad_revision_from_brep_model(model)
    if selection.revision_id != revision.revision_id or not selection.selectors:
        raise ValueError(
            "CAD selection must be nonempty and belong to the source revision."
        )
    incidence = model.geometry.qualified_entity_incidence(
        model.source_id, model.source_revision
    )
    uses = {
        _entity_occurrence_id(record.member): record
        for record in incidence
        if record.container.kind == "solid" and record.member.kind == "face"
    }
    occurrences = {
        occurrence.path: occurrence
        for occurrence in model.geometry.assembly_graph.iter_occurrences()
    }
    placed_uses: dict[tuple, dict[int, set[tuple[str, ...]]]] = {}
    for use in uses.values():
        occurrence = occurrences[use.container.occurrence_path]
        key = (use.member.index, occurrence.placement.matrix)
        placed_uses.setdefault(key, {}).setdefault(use.container.index, set()).add(
            use.container.occurrence_path
        )
    bank = {
        domain.source_entity_key_for_scope(2, int(identifier)): int(identifier)
        for identifier in domain.scope_entity_ids(2)
    }
    result = []
    for selector in selection.selectors:
        try:
            use = uses[selector.occurrence_id]
            occurrence = occurrences[use.container.occurrence_path]
            peers = placed_uses[(use.member.index, occurrence.placement.matrix)]
            if len(peers) > 1 and any(len(paths) > 1 for paths in peers.values()):
                raise ValueError(
                    "Repeated coincident solid uses do not establish exact shared-face "
                    "copy pairing; source occurrence order cannot authenticate the side."
                )
            if revision.select(selector.occurrence_id) != selector:
                raise ValueError("CAD face selector differs from its source inventory.")
            regions = [
                region
                for region, (solid, path) in enumerate(
                    zip(
                        domain.region_source_indices,
                        domain.region_source_occurrences,
                        strict=True,
                    )
                )
                if solid == use.container.index and path == use.container.occurrence_path
            ]
            if len(regions) != 1:
                raise ValueError(
                    "CAD solid occurrence does not identify one source region."
                )
            patches = [
                patch
                for patch, sign in domain.regions[regions[0]].boundary
                if domain.source_indices[2][patch] == use.member.index
                and sign == use.orientation
            ]
            if len(patches) != 1:
                raise ValueError(
                    "CAD face use does not identify one source region boundary."
                )
            patch = patches[0]
            path = domain.source_occurrences[2][patch]
            key = source_entity_id(
                model.source_id,
                model.source_revision,
                "face",
                use.member.index,
                occurrence_path=path,
            )
            slot = bank[(key, path)]
            solid = revision.select(_entity_occurrence_id(use.container))
        except KeyError as exc:
            raise ValueError(
                "CAD side is absent from the source topology or domain."
            ) from exc
        result.append(
            MeshingScopeSide(
                slot,
                use.member.index,
                selector,
                use.container.index,
                solid,
                use.container.occurrence_path,
                use.orientation,
            )
        )
    return tuple(result)


class MeshingScope(StrictModule, NonTrainableState):
    """Exact entity scope bound to one immutable source revision."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    entity_kind: MeshingEntityKind = eqx.field(static=True)
    entity_dimension: int = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    entity_ids: Array
    unqualified_entity_ids: tuple[int, ...] = eqx.field(static=True)
    side_records: tuple[MeshingScopeSide, ...] = eqx.field(static=True)
    scope_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        entity_kind: MeshingEntityKind,
        entity_dimension: int,
        entity_set_id: str,
        entity_ids: ConvertibleToArray,
        /,
        *,
        side_records: tuple[MeshingScopeSide, ...] = (),
        unqualified_entity_ids: tuple[int, ...] | None = None,
    ) -> None:
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        entity_set = str(entity_set_id).strip()
        if not source or not revision or not entity_set:
            raise ValueError("Meshing scope identities must be non-empty.")
        if not isinstance(entity_kind, MeshingEntityKind):
            raise TypeError("entity_kind must be MeshingEntityKind.")
        dimension = int(entity_dimension)
        if dimension < 0:
            raise ValueError("entity_dimension must be non-negative.")
        identifiers = np.asarray(entity_ids)
        if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
            raise TypeError("Meshing scope entity_ids must be one integer vector.")
        identifiers = identifiers.astype(np.int64, copy=False)
        if identifiers.size == 0:
            raise ValueError("Meshing scopes must contain at least one entity.")
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError("Meshing scope entity_ids must be unique and non-negative.")
        identifiers = np.sort(identifiers, kind="stable")
        sides = tuple(side_records)
        if any(not isinstance(side, MeshingScopeSide) for side in sides):
            raise TypeError("side_records must contain MeshingScopeSide values.")
        if len(set(sides)) != len(sides):
            raise ValueError("Side records must be unique.")
        if sides and (
            entity_kind is not MeshingEntityKind.GEOMETRY
            or dimension != 2
            or any(side.face.revision_id != revision for side in sides)
        ):
            raise ValueError("CAD sides require a matching geometry face revision.")
        unqualified = (
            tuple(int(value) for value in identifiers)
            if unqualified_entity_ids is None
            else tuple(unqualified_entity_ids)
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 0
            for value in unqualified
        ) or len(set(unqualified)) != len(unqualified):
            raise ValueError("Unqualified IDs must be unique non-negative integers.")
        unqualified = tuple(sorted(int(value) for value in unqualified))
        sides = tuple(sorted(sides, key=_side_key))
        if set(map(int, identifiers)) != set(unqualified) | {
            side.entity_id for side in sides
        }:
            raise ValueError("Entity IDs must be exactly the projection of scope parts.")
        self.source_id = source
        self.source_revision = revision
        self.entity_kind = entity_kind
        self.entity_dimension = dimension
        self.entity_set_id = entity_set
        self.entity_ids = jnp.asarray(identifiers, dtype=jnp.int64)
        self.unqualified_entity_ids = unqualified
        self.side_records = sides
        if sides:
            self.scope_id = canonical_fingerprint(
                {
                    "kind": "sided-meshing-scope",
                    "source_id": source,
                    "source_revision": revision,
                    "entity_kind": entity_kind.value,
                    "entity_dimension": dimension,
                    "entity_set_id": entity_set,
                    "unqualified_entity_ids": unqualified,
                    "side_records": tuple(_side_key(side) for side in sides),
                }
            )
        else:
            self.scope_id = canonical_fingerprint(
                {
                    "kind": "meshing-scope",
                    "source_id": source,
                    "source_revision": revision,
                    "entity_kind": entity_kind.value,
                    "entity_dimension": dimension,
                    "entity_set_id": entity_set,
                    "entity_ids": array_tree_fingerprint(identifiers),
                }
            )

    @classmethod
    def _from_parts(
        cls,
        source_id: str,
        source_revision: str,
        entity_kind: MeshingEntityKind,
        entity_dimension: int,
        entity_set_id: str,
        unqualified: tuple[int, ...],
        sides: tuple[MeshingScopeSide, ...],
    ) -> MeshingScope:
        identifiers = sorted(set(unqualified) | {side.entity_id for side in sides})
        if not identifiers:
            raise ValueError("Meshing scope set operation is empty.")
        return cls(
            source_id,
            source_revision,
            entity_kind,
            entity_dimension,
            entity_set_id,
            np.asarray(identifiers, dtype=np.int64),
            side_records=sides,
            unqualified_entity_ids=unqualified,
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[MeshingScope],
        value: MeshingScope,
        /,
        *,
        source_model: BRepModel | None = None,
    ) -> MeshingScope:
        if type(value) is not cls:
            raise TypeError("value must be MeshingScope.")
        if type(value.entity_dimension) is not int:
            raise TypeError("Stored meshing scope dimension must be an int.")
        stored_ids = np.asarray(value.entity_ids)
        if stored_ids.ndim != 1 or stored_ids.dtype != np.dtype(np.int64):
            raise TypeError("Stored meshing scope entity_ids must be int64.")
        sides = tuple(value.side_records)
        unqualified = tuple(value.unqualified_entity_ids)
        if any(not isinstance(side, MeshingScopeSide) for side in sides):
            raise TypeError("Stored side records must be MeshingScopeSide values.")
        if len(set(sides)) != len(sides) or len(set(unqualified)) != len(unqualified):
            raise ValueError("Stored scope parts must be unique.")
        if any(
            isinstance(identifier, bool)
            or not isinstance(identifier, (int, np.integer))
            or identifier < 0
            for identifier in unqualified
        ):
            raise ValueError("Stored unqualified IDs must be non-negative integers.")
        if sides:
            if source_model is None:
                raise ValueError("Stored CAD sides require their source model.")
            from ..geometry._cad_revision import CADSelectionSet

            selection = CADSelectionSet(
                value.source_revision, tuple(side.face for side in sides)
            )
            expected = cls.from_geometry(source_model, selection)
            if (
                expected.source_id != value.source_id
                or expected.source_revision != value.source_revision
                or expected.entity_set_id != value.entity_set_id
                or value.entity_kind is not MeshingEntityKind.GEOMETRY
                or value.entity_dimension != 2
                or expected.side_records != tuple(sorted(sides, key=_side_key))
            ):
                raise ValueError("Stored CAD sides do not match the source incidence.")
            owner = _source_scope_owner(source_model)
            if not set(unqualified).issubset(set(map(int, owner.scope_entity_ids(2)))):
                raise ValueError("Stored scope contains undeclared unqualified IDs.")
        rebuilt = cls._from_parts(
            value.source_id,
            value.source_revision,
            value.entity_kind,
            value.entity_dimension,
            value.entity_set_id,
            unqualified,
            sides,
        )
        if (
            rebuilt.scope_id != value.scope_id
            or rebuilt.source_id != value.source_id
            or rebuilt.source_revision != value.source_revision
            or rebuilt.entity_kind is not value.entity_kind
            or rebuilt.entity_dimension != value.entity_dimension
            or rebuilt.entity_set_id != value.entity_set_id
            or rebuilt.unqualified_entity_ids != value.unqualified_entity_ids
            or rebuilt.side_records != value.side_records
            or not np.array_equal(
                np.asarray(rebuilt.entity_ids), np.asarray(value.entity_ids)
            )
        ):
            raise ValueError("Stored meshing scope differs from its exact entity bank.")
        return rebuilt

    @classmethod
    def from_geometry(
        cls,
        source: BRepModel
        | BRepSource
        | FixedTopologyBRepSource
        | LabelFieldSurfaceExtractionResult
        | _SourceEntityScopes,
        entity_ids: Sequence[BRepEntityId] | BRepEntityId | CADSelectionSet | ArrayLike,
        /,
        *,
        dimension: int | None = None,
    ) -> MeshingScope:
        """Bind exact source entities, including authenticated CAD occurrence sets.

        CAD selections require their target BRep model or source and one nonempty
        entity dimension. Transport across revisions must be performed explicitly
        with an association graph before constructing the scope.

        Face occurrences on solids retain their exact region side and orientation;
        the entity IDs are only the projection onto source facet slots.
        """
        from ..geometry._cad_revision import CADSelectionSet

        owner = _source_scope_owner(source)
        if (
            isinstance(entity_ids, CADSelectionSet)
            and entity_ids.selectors
            and all(selector.kind == "face" for selector in entity_ids.selectors)
        ):
            model = _geometry_model(source)
            from ..geometry.brep._partition import _entity_occurrence_id

            incident = {
                _entity_occurrence_id(record.member)
                for record in model.geometry.qualified_entity_incidence(
                    model.source_id, model.source_revision
                )
                if record.container.kind == "solid" and record.member.kind == "face"
            }
            sided = tuple(s for s in entity_ids.selectors if s.occurrence_id in incident)
            unsided = tuple(
                s for s in entity_ids.selectors if s.occurrence_id not in incident
            )
            if dimension is not None and nonnegative_integer(dimension, "dimension") != 2:
                raise ValueError(
                    "The requested dimension disagrees with the BRep entity kind."
                )
            sides = (
                _cad_face_sides(source, CADSelectionSet(entity_ids.revision_id, sided))
                if sided
                else ()
            )
            if unsided:
                _, ids = _cad_scope_ids(
                    source,
                    owner,
                    CADSelectionSet(entity_ids.revision_id, unsided),
                    dimension,
                )
                unqualified = tuple(int(value) for value in ids)
            else:
                unqualified = ()
            return cls._from_parts(
                owner.source_id,
                owner.source_revision,
                MeshingEntityKind.GEOMETRY,
                2,
                owner.scope_entity_set_id(2),
                unqualified,
                sides,
            )
        if isinstance(entity_ids, CADSelectionSet):
            target, identifiers = _cad_scope_ids(source, owner, entity_ids, dimension)
        else:
            target, identifiers = _geometry_scope_ids(owner, entity_ids, dimension)
        return cls(
            owner.source_id,
            owner.source_revision,
            MeshingEntityKind.GEOMETRY,
            target,
            owner.scope_entity_set_id(target),
            identifiers,
        )

    @classmethod
    def whole(
        cls,
        source: (
            BRepModel
            | BRepSource
            | FixedTopologyBRepSource
            | SurfaceModel
            | CellMesh
            | _SourceEntityScopes
            | LabelFieldSurfaceExtractionResult
        ),
        dimension: int,
        /,
    ) -> MeshingScope:
        """Select all active entities of an explicitly declared dimension."""
        from ..geometry.surface import SurfaceModel

        target = nonnegative_integer(dimension, "dimension")
        if isinstance(source, (SurfaceModel, CellMesh)):
            mesh = source.mesh if isinstance(source, SurfaceModel) else source
            if target > mesh.topological_dimension:
                raise ValueError(
                    "The requested entity dimension is absent from the mesh."
                )
            entities = mesh.entity_set(target)
            return cls(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingEntityKind.MESH,
                target,
                entities.entity_set_id,
                np.asarray(entities.entity_ids)[np.asarray(entities.active_mask)],
            )
        owner = _source_scope_owner(source)
        return cls.from_geometry(owner, owner.scope_entity_ids(target), dimension=target)

    @classmethod
    @checked
    def from_selection(
        cls,
        source_id: str,
        source_revision: str,
        entities: EntitySet,
        selection: EntitySelection,
        /,
    ) -> MeshingScope:
        """Convert a positional selection using its exact owning entity set."""
        if selection.entity_set_id != entities.entity_set_id:
            raise ValueError("Selection must belong to the supplied entity set.")
        mask = np.asarray(selection.mask, dtype=np.bool_)
        if mask.shape != (entities.count,) or not np.array_equal(
            selection.active_mask, entities.active_mask
        ):
            raise ValueError(
                "Selection must match the entity set's capacity and active mask."
            )
        return cls(
            source_id,
            source_revision,
            MeshingEntityKind.MESH,
            entities.intrinsic_dimension,
            entities.entity_set_id,
            np.asarray(entities.entity_ids)[mask],
        )

    @checked
    def _check_compatible(self, other: MeshingScope, /) -> None:
        binding = (
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
        )
        other_binding = (
            other.source_id,
            other.source_revision,
            other.entity_kind,
            other.entity_dimension,
            other.entity_set_id,
        )
        if binding != other_binding:
            raise ValueError("Meshing scope set operations require one exact binding.")

    def _set_operation(self, other: MeshingScope, operation: str, /) -> MeshingScope:
        self._check_compatible(other)
        left_ids, right_ids = (
            set(self.unqualified_entity_ids),
            set(other.unqualified_entity_ids),
        )
        left_sides, right_sides = set(self.side_records), set(other.side_records)
        if operation == "union":
            identifiers, sides = left_ids | right_ids, left_sides | right_sides
        elif operation == "intersection":
            identifiers = left_ids & right_ids
            sides = (
                (left_sides & right_sides)
                | {side for side in left_sides if side.entity_id in right_ids}
                | {side for side in right_sides if side.entity_id in left_ids}
            )
        else:
            partial = {
                side.entity_id
                for side in right_sides
                if side.entity_id in left_ids and side.entity_id not in right_ids
            }
            if partial:
                raise ValueError(
                    "Subtracting a CAD side from an unqualified entity requires "
                    "the authoritative complete side inventory."
                )
            identifiers = left_ids - right_ids
            sides = {
                side
                for side in left_sides
                if side not in right_sides and side.entity_id not in right_ids
            }
        return self._from_parts(
            self.source_id,
            self.source_revision,
            self.entity_kind,
            self.entity_dimension,
            self.entity_set_id,
            tuple(identifiers),
            tuple(sides),
        )

    def union(self, other: MeshingScope, /) -> MeshingScope:
        return self._set_operation(other, "union")

    def intersection(self, other: MeshingScope, /) -> MeshingScope:
        return self._set_operation(other, "intersection")

    def difference(self, other: MeshingScope, /) -> MeshingScope:
        return self._set_operation(other, "difference")

    def __or__(self, other: MeshingScope) -> MeshingScope:
        return self.union(other)

    def __and__(self, other: MeshingScope) -> MeshingScope:
        return self.intersection(other)

    def __sub__(self, other: MeshingScope) -> MeshingScope:
        return self.difference(other)


def resolve_mesh_scope(mesh: CellMesh, scope: MeshingScope, /) -> EntitySelection:
    """Resolve persistent mesh-entity IDs onto one exact mesh topology."""

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(scope, MeshingScope):
        raise TypeError("scope must be MeshingScope.")
    if scope.source_id != mesh.mesh_id:
        raise ValueError("Meshing scope belongs to another mesh source.")
    if scope.source_revision != mesh.numeric_version:
        raise ValueError("Meshing scope belongs to another mesh revision.")
    if scope.entity_kind is not MeshingEntityKind.MESH:
        raise ValueError("Meshing scope must select mesh entities.")
    if scope.entity_dimension < 0 or scope.entity_dimension > mesh.topological_dimension:
        raise ValueError("Meshing scope entity dimension is not present on the mesh.")
    entities = mesh.entity_set(scope.entity_dimension)
    if scope.entity_set_id != entities.entity_set_id:
        raise ValueError("Meshing scope does not match the mesh entity set.")
    entity_ids = np.asarray(entities.entity_ids, dtype=np.int64)
    active = np.asarray(entities.active_mask, dtype=np.bool_)
    requested = np.asarray(scope.entity_ids, dtype=np.int64)
    mask = active & np.isin(entity_ids, requested)
    if np.count_nonzero(mask) != requested.size:
        raise ValueError("Meshing scope contains undeclared mesh entity IDs.")
    return EntitySelection(entities, mask)


class ScopeResolutionReport(StrictModule, NonTrainableState):
    query: str = eqx.field(static=True)
    matched_names: tuple[str, ...] = eqx.field(static=True)
    unmatched_names: tuple[str, ...] = eqx.field(static=True)
    scope: MeshingScope
    report_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        query: str,
        scope: MeshingScope,
        /,
        *,
        matched_names: tuple[str, ...] = (),
        unmatched_names: tuple[str, ...] = (),
    ) -> None:
        expression = str(query).strip()
        if not expression:
            raise ValueError("Scope resolution query must be non-empty.")
        matched = tuple(str(value).strip() for value in matched_names)
        unmatched = tuple(str(value).strip() for value in unmatched_names)
        if any(not value for value in (*matched, *unmatched)):
            raise ValueError("Scope resolution names must be non-empty.")
        self.query = expression
        self.matched_names = matched
        self.unmatched_names = unmatched
        self.scope = scope
        self.report_id = canonical_fingerprint(
            {
                "kind": "scope-resolution-report",
                "query": expression,
                "matched_names": matched,
                "unmatched_names": unmatched,
                "scope": scope.scope_id,
            }
        )


__all__ = [
    "MeshingEntityKind",
    "MeshingScope",
    "MeshingScopeSide",
    "ScopeResolutionReport",
    "resolve_mesh_scope",
]
