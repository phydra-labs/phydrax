#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from ..units import UnitDefinition


def _integer_array(values: ArrayLike, owner: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{owner} must contain integers.")
    if np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{owner} must contain non-negative signed 64-bit integers.")
    result = np.array(raw, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _entity_ids(values: ArrayLike, owner: str) -> np.ndarray:
    result = _integer_array(values, owner)
    if result.ndim != 1 or not result.size or np.unique(result).size != result.size:
        raise ValueError(f"{owner} must be one non-empty unique integer vector.")
    return result


class MeshArrayAssociation(StrEnum):
    POINT = "point"
    CELL = "cell"
    ENTITY = "entity"


@dataclass(frozen=True, slots=True)
class MeshArrayBlock:
    name: str
    cell_kind: str
    source_cell_type: str
    geometry_order: int
    connectivity: np.ndarray
    global_ids: np.ndarray
    block_id: str

    def __init__(
        self,
        name: str,
        cell_kind: str,
        source_cell_type: str,
        geometry_order: int,
        connectivity: ArrayLike,
        global_ids: ArrayLike,
        /,
    ):
        name_ = str(name).strip()
        kind = str(cell_kind).strip()
        source_type = str(source_cell_type).strip()
        if isinstance(geometry_order, (bool, np.bool_)) or not isinstance(
            geometry_order, (int, np.integer)
        ):
            raise TypeError("Mesh array geometry_order must be an integer.")
        order = int(geometry_order)
        cells = _integer_array(connectivity, "Mesh array connectivity")
        identifiers = _entity_ids(global_ids, "Mesh array cell IDs")
        if not name_ or not kind or not source_type:
            raise ValueError("Mesh array block identities must be non-empty.")
        if order <= 0:
            raise ValueError("Mesh array geometry_order must be positive.")
        if cells.ndim != 2 or 0 in cells.shape:
            raise ValueError(
                "Mesh array connectivity must be a non-empty integer matrix."
            )
        if identifiers.shape != (cells.shape[0],):
            raise ValueError("Mesh array cell IDs must match connectivity rows.")
        object.__setattr__(self, "name", name_)
        object.__setattr__(self, "cell_kind", kind)
        object.__setattr__(self, "source_cell_type", source_type)
        object.__setattr__(self, "geometry_order", order)
        object.__setattr__(self, "connectivity", cells)
        object.__setattr__(self, "global_ids", identifiers)
        object.__setattr__(
            self,
            "block_id",
            canonical_fingerprint(
                {
                    "kind": "mesh-array-block",
                    "name": name_,
                    "cell_kind": kind,
                    "source_cell_type": source_type,
                    "geometry_order": order,
                    "connectivity": array_tree_fingerprint(cells),
                    "global_ids": array_tree_fingerprint(identifiers),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class MeshArrayField:
    name: str
    association: MeshArrayAssociation
    block_name: str | None
    values: np.ndarray
    role: str
    unit: UnitDefinition | None
    entity_dimension: int | None
    entity_ids: np.ndarray | None
    entity_kind: str
    field_id: str

    def __init__(
        self,
        name: str,
        association: MeshArrayAssociation,
        values: ArrayLike,
        /,
        *,
        block_name: str | None = None,
        role: str = "user",
        unit: UnitDefinition | None = None,
        entity_dimension: int | None = None,
        entity_ids: ArrayLike | None = None,
        entity_kind: str = "mesh",
    ):
        name_ = str(name).strip()
        block = None if block_name is None else str(block_name).strip()
        role_ = str(role).strip()
        kind = str(entity_kind).strip()
        array = np.array(values, copy=True)
        if not name_ or not role_ or not kind:
            raise ValueError(
                "Mesh array field name, role, and entity kind must be non-empty."
            )
        if not isinstance(association, MeshArrayAssociation):
            raise TypeError("association must be MeshArrayAssociation.")
        if association is MeshArrayAssociation.CELL and not block:
            raise ValueError("Cell fields require a block_name.")
        if association is not MeshArrayAssociation.CELL and block is not None:
            raise ValueError("Only cell fields can name one cell block.")
        identifiers = None
        if association is MeshArrayAssociation.ENTITY:
            if (
                isinstance(entity_dimension, (bool, np.bool_))
                or not isinstance(entity_dimension, (int, np.integer))
                or entity_dimension < 0
            ):
                raise ValueError(
                    "Entity fields require a non-negative integer dimension."
                )
            if entity_ids is None:
                raise ValueError("Entity fields require scoped entity IDs.")
            identifiers = _entity_ids(entity_ids, "Mesh array field entity IDs")
            if array.ndim == 0 or array.shape[0] != identifiers.size:
                raise ValueError("Entity field values must match their scoped IDs.")
        elif entity_dimension is not None or entity_ids is not None:
            raise ValueError("Only entity fields can provide scoped entity IDs.")
        if unit is not None and not isinstance(unit, UnitDefinition):
            raise TypeError("Mesh array field unit must be UnitDefinition or None.")
        if array.ndim == 0 or array.shape[0] == 0:
            raise ValueError("Mesh array fields require one non-empty entity axis.")
        if array.dtype.kind not in "biuf" or (
            array.dtype.kind == "f" and not np.all(np.isfinite(array))
        ):
            raise ValueError("Mesh array fields must contain finite real numeric data.")
        array.setflags(write=False)
        for key, value in (
            ("name", name_),
            ("association", association),
            ("block_name", block),
            ("values", array),
            ("role", role_),
            ("unit", unit),
            (
                "entity_dimension",
                None if entity_dimension is None else int(entity_dimension),
            ),
            ("entity_ids", identifiers),
            ("entity_kind", kind),
        ):
            object.__setattr__(self, key, value)
        object.__setattr__(
            self,
            "field_id",
            canonical_fingerprint(
                {
                    "kind": "mesh-array-field",
                    "name": name_,
                    "association": association.value,
                    "block_name": block,
                    "values": array_tree_fingerprint(array),
                    "role": role_,
                    "unit": None if unit is None else unit.unit_id,
                    "entity_dimension": self.entity_dimension,
                    "entity_ids": None
                    if identifiers is None
                    else array_tree_fingerprint(identifiers),
                    "entity_kind": kind,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class MeshArraySelection:
    """Neutral named entity selection; zones have a role, labels do not."""

    name: str
    entity_dimension: int
    entity_ids: np.ndarray
    role: str | None
    entity_kind: str
    selection_id: str

    def __init__(
        self,
        name: str,
        entity_dimension: int,
        entity_ids: ArrayLike,
        /,
        *,
        role: str | None = None,
        entity_kind: str = "mesh",
    ):
        name_ = str(name).strip()
        kind = str(entity_kind).strip()
        role_ = None if role is None else str(role).strip()
        if not name_ or not kind or role_ == "":
            raise ValueError("Mesh array selection identities must be non-empty.")
        if (
            isinstance(entity_dimension, (bool, np.bool_))
            or not isinstance(entity_dimension, (int, np.integer))
            or entity_dimension < 0
        ):
            raise ValueError("Selection entity_dimension must be a non-negative integer.")
        identifiers = _entity_ids(entity_ids, "Mesh array selection entity IDs")
        object.__setattr__(self, "name", name_)
        object.__setattr__(self, "entity_dimension", int(entity_dimension))
        object.__setattr__(self, "entity_ids", identifiers)
        object.__setattr__(self, "role", role_)
        object.__setattr__(self, "entity_kind", kind)
        object.__setattr__(
            self,
            "selection_id",
            canonical_fingerprint(
                {
                    "kind": "mesh-array-selection",
                    "name": name_,
                    "entity_dimension": int(entity_dimension),
                    "entity_ids": array_tree_fingerprint(identifiers),
                    "role": role_,
                    "entity_kind": kind,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class MeshArrayArtifact:
    points: np.ndarray
    point_global_ids: np.ndarray
    blocks: tuple[MeshArrayBlock, ...]
    fields: tuple[MeshArrayField, ...]
    coordinate_contract: SpatialCoordinateContract
    source_id: str
    source_format: str
    numeric_version: str
    vertex_point_indices: np.ndarray | None
    vertex_coordinates: np.ndarray | None
    entity_global_ids: tuple[tuple[int, np.ndarray], ...]
    zones: tuple[MeshArraySelection, ...]
    labels: tuple[MeshArraySelection, ...]
    artifact_id: str

    def __init__(
        self,
        points: ArrayLike,
        point_global_ids: ArrayLike,
        blocks: tuple[MeshArrayBlock, ...],
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        source_id: str,
        source_format: str,
        fields: tuple[MeshArrayField, ...] = (),
        numeric_version: str = "0",
        vertex_point_indices: ArrayLike | None = None,
        vertex_coordinates: ArrayLike | None = None,
        entity_global_ids: tuple[tuple[int, ArrayLike], ...] = (),
        zones: tuple[MeshArraySelection, ...] = (),
        labels: tuple[MeshArraySelection, ...] = (),
    ):
        if np.asarray(points).dtype.kind not in "iuf":
            raise TypeError("Mesh array points must contain real numeric coordinates.")
        if (
            vertex_coordinates is not None
            and np.asarray(vertex_coordinates).dtype.kind not in "iuf"
        ):
            raise TypeError("Vertex coordinates must contain real numeric coordinates.")
        coordinates = np.array(points, dtype=float, copy=True)
        identifiers = _entity_ids(point_global_ids, "Point IDs")
        blocks_ = tuple(blocks)
        fields_ = tuple(fields)
        zones_, labels_ = tuple(zones), tuple(labels)
        source, format_, version = (
            str(source_id).strip(),
            str(source_format).strip(),
            str(numeric_version).strip(),
        )
        if (
            coordinates.ndim != 2
            or 0 in coordinates.shape
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError("Mesh array points must be one non-empty finite matrix.")
        if identifiers.shape != (coordinates.shape[0],):
            raise ValueError("Point IDs must match the mesh array points.")
        if not blocks_ or not all(isinstance(block, MeshArrayBlock) for block in blocks_):
            raise ValueError("Mesh array artifacts require at least one block.")
        block_by_name = {block.name: block for block in blocks_}
        if len(block_by_name) != len(blocks_):
            raise ValueError("Mesh array block names must be unique.")
        cell_ids = np.concatenate([block.global_ids for block in blocks_])
        if np.unique(cell_ids).size != cell_ids.size:
            raise ValueError("Mesh array cell IDs must be unique across blocks.")
        if not all(isinstance(field, MeshArrayField) for field in fields_):
            raise TypeError("fields must contain MeshArrayField values.")
        field_keys = [
            (
                field.name,
                field.association,
                field.block_name,
                field.entity_dimension,
                None
                if field.entity_ids is None
                else canonical_fingerprint(
                    array_tree_fingerprint(np.sort(field.entity_ids))
                ),
            )
            for field in fields_
        ]
        if len(set(field_keys)) != len(field_keys):
            raise ValueError(
                "Mesh array field names must be unique within each association."
            )
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if not source or not format_ or not version:
            raise ValueError(
                "Mesh array source identity, format, and numeric version must be non-empty."
            )
        for block in blocks_:
            if np.any(block.connectivity >= coordinates.shape[0]):
                raise ValueError("Mesh array connectivity indexes undeclared points.")
        vertices = (
            None
            if vertex_point_indices is None
            else _entity_ids(vertex_point_indices, "Vertex point indices")
        )
        if vertices is not None and np.any(vertices >= coordinates.shape[0]):
            raise ValueError("Vertex point indices index undeclared points.")
        entity_sets = []
        for dimension, values in entity_global_ids:
            if (
                isinstance(dimension, (bool, np.bool_))
                or not isinstance(dimension, (int, np.integer))
                or dimension < 0
            ):
                raise ValueError(
                    "Entity global IDs require non-negative integer dimensions."
                )
            entity_sets.append((int(dimension), _entity_ids(values, "Entity global IDs")))
        entity_sets = tuple(sorted(entity_sets, key=lambda item: item[0]))
        if len(dict(entity_sets)) != len(entity_sets):
            raise ValueError("Entity global ID dimensions must be unique.")
        by_dimension = dict(entity_sets)
        for field in fields_:
            if field.association is MeshArrayAssociation.ENTITY:
                field_dimension, field_ids = field.entity_dimension, field.entity_ids
                if field_dimension is None or field_ids is None:
                    raise ValueError(
                        "Mesh array entity fields require a dimension and scoped IDs."
                    )
                declared_ids = by_dimension.get(field_dimension)
                if declared_ids is None or not np.all(np.isin(field_ids, declared_ids)):
                    raise ValueError(
                        "Mesh array entity field indexes undeclared entity IDs."
                    )
                continue
            if field.association is MeshArrayAssociation.POINT:
                expected = coordinates.shape[0]
            else:
                block_name = field.block_name
                if block_name is None or block_name not in block_by_name:
                    raise ValueError("Mesh array cell field names an undeclared block.")
                expected = block_by_name[block_name].connectivity.shape[0]
            if field.values.shape[0] != expected:
                raise ValueError(
                    "Mesh array field cardinality does not match its association."
                )
        for selections, is_zone in ((zones_, True), (labels_, False)):
            if not all(
                isinstance(selection, MeshArraySelection) for selection in selections
            ):
                raise TypeError(
                    "Mesh array organization must contain MeshArraySelection values."
                )
            if len({selection.name for selection in selections}) != len(selections):
                raise ValueError("Mesh array organization names must be unique.")
            for selection in selections:
                if (selection.role is not None) != is_zone:
                    raise ValueError(
                        "Mesh array zones require roles; labels cannot have roles."
                    )
                if selection.entity_dimension not in by_dimension or not np.all(
                    np.isin(
                        selection.entity_ids, by_dimension[selection.entity_dimension]
                    )
                ):
                    raise ValueError(
                        "Mesh array selection indexes undeclared entity IDs."
                    )
        vertex_points = (
            None
            if vertex_coordinates is None
            else np.array(vertex_coordinates, dtype=float, copy=True)
        )
        if vertex_points is not None:
            if (
                vertices is None
                or vertex_points.shape != (vertices.size, coordinates.shape[1])
                or not np.all(np.isfinite(vertex_points))
            ):
                raise ValueError(
                    "Vertex coordinates must match the vertex point mapping."
                )
            vertex_points.setflags(write=False)
        coordinates.setflags(write=False)
        for key, value in (
            ("points", coordinates),
            ("point_global_ids", identifiers),
            ("blocks", blocks_),
            ("fields", fields_),
            ("coordinate_contract", coordinate_contract),
            ("source_id", source),
            ("source_format", format_),
            ("numeric_version", version),
            ("vertex_point_indices", vertices),
            ("entity_global_ids", entity_sets),
            ("zones", zones_),
            ("labels", labels_),
            ("vertex_coordinates", vertex_points),
        ):
            object.__setattr__(self, key, value)
        object.__setattr__(
            self,
            "artifact_id",
            canonical_fingerprint(
                {
                    "kind": "mesh-array-artifact",
                    "points": array_tree_fingerprint(coordinates),
                    "point_global_ids": array_tree_fingerprint(identifiers),
                    "blocks": [block.block_id for block in blocks_],
                    "fields": [field.field_id for field in fields_],
                    "coordinate_contract": coordinate_contract.spatial_id,
                    "source_id": source,
                    "source_format": format_,
                    "numeric_version": version,
                    "vertex_point_indices": None
                    if vertices is None
                    else array_tree_fingerprint(vertices),
                    "vertex_coordinates": None
                    if vertex_points is None
                    else array_tree_fingerprint(vertex_points),
                    "entity_global_ids": [
                        [dimension, array_tree_fingerprint(values)]
                        for dimension, values in entity_sets
                    ],
                    "zones": [selection.selection_id for selection in zones_],
                    "labels": [selection.selection_id for selection in labels_],
                }
            ),
        )


__all__ = [
    "MeshArrayArtifact",
    "MeshArrayAssociation",
    "MeshArrayBlock",
    "MeshArrayField",
    "MeshArraySelection",
]
