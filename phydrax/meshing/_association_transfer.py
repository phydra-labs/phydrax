from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, TYPE_CHECKING


if TYPE_CHECKING:
    from ..discretization._cell_mesh import CellMesh
    from ._lineage import MeshLineage

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._volume_material_map import (
    EulerianVolumeOverlapMap,
)


class MaterialAssociationSide(NamedTuple):
    """Stable source-cell association side preserved by one physical overlap."""

    cell_global_id: int
    entity_id: str
    region_id: str
    occurrence_path: tuple[str, ...]
    overlap_measure: float


class TargetMaterialAssociationSide(NamedTuple):
    """Target material/occurrence side with its physical denominator measure."""

    cell_global_id: int
    entity_id: str
    region_id: str
    occurrence_path: tuple[str, ...]
    physical_measure: float


class MaterialAssociationRows(NamedTuple):
    """Target cell sides resolved from exact cell-lineage contributors."""

    target_cell_global_ids: tuple[int, ...]
    cell_entity_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    occurrence_paths: tuple[tuple[str, ...], ...]
    lineage_id: str


class MaterialAssociationTransferError(ValueError):
    """Source occurrence, region, or coverage relations are not transferable."""

    def __init__(
        self,
        message: str,
        /,
        *,
        source_unmatched_rows: np.ndarray | None = None,
        target_unmatched_rows: np.ndarray | None = None,
        source_region_ids: tuple[str, ...] = (),
        target_region_ids: tuple[str, ...] = (),
        source_occurrence_paths: tuple[tuple[str, ...], ...] = (),
        target_occurrence_paths: tuple[tuple[str, ...], ...] = (),
        target_conflicting_rows: np.ndarray | None = None,
    ) -> None:
        super().__init__(message)
        self.source_unmatched_rows = (
            None
            if source_unmatched_rows is None
            else jnp.asarray(source_unmatched_rows, dtype=jnp.int64)
        )
        self.target_unmatched_rows = (
            None
            if target_unmatched_rows is None
            else jnp.asarray(target_unmatched_rows, dtype=jnp.int64)
        )
        self.source_region_ids = source_region_ids
        self.target_region_ids = target_region_ids
        self.source_occurrence_paths = source_occurrence_paths
        self.target_occurrence_paths = target_occurrence_paths
        self.target_conflicting_rows = (
            None
            if target_conflicting_rows is None
            else jnp.asarray(target_conflicting_rows, dtype=jnp.int64)
        )


def propagate_material_association_rows(
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    lineage: MeshLineage,
    source_cell_entity_ids: Sequence[str],
    source_region_ids: Sequence[str],
    source_occurrence_paths: Sequence[Sequence[str]],
    target_region_ids: Sequence[str],
    /,
) -> MaterialAssociationRows:
    """Propagate cell material sides only when every exact lineage parent agrees."""
    from ..discretization._cell_mesh import CellMesh as CellMeshOwner
    from ._lineage import EntityLineageKind, MeshLineage as MeshLineageOwner

    if not isinstance(source_mesh, CellMeshOwner) or not isinstance(
        target_mesh, CellMeshOwner
    ):
        raise TypeError(
            "Material lineage propagation requires canonical CellMesh endpoints."
        )
    if not isinstance(lineage, MeshLineageOwner):
        raise TypeError("lineage must be MeshLineage.")
    if (
        source_mesh.topological_dimension != 3
        or target_mesh.topological_dimension != 3
        or lineage.source_topology_id != source_mesh.topology_id
        or lineage.target_topology_id != target_mesh.topology_id
    ):
        raise MaterialAssociationTransferError(
            "Material cell lineage endpoints do not match the three-dimensional mesh epochs."
        )
    try:
        cell_lineage = lineage.entity_lineage(3)
    except KeyError as error:
        raise MaterialAssociationTransferError(
            "Material association propagation requires explicit three-dimensional cell lineage."
        ) from error
    if (
        cell_lineage.source_entity_set_id != source_mesh.entity_set(3).entity_set_id
        or cell_lineage.target_entity_set_id != target_mesh.entity_set(3).entity_set_id
    ):
        raise MaterialAssociationTransferError(
            "Cell lineage entity-set identities do not match material endpoints."
        )

    source_global_ids = tuple(
        int(value)
        for block in source_mesh.blocks
        for value in np.asarray(block.global_ids, dtype=np.int64)
    )
    target_global_ids = tuple(
        int(value)
        for block in target_mesh.blocks
        for value in np.asarray(block.global_ids, dtype=np.int64)
    )
    source_entities = tuple(str(value).strip() for value in source_cell_entity_ids)
    source_regions = tuple(str(value).strip() for value in source_region_ids)
    source_paths = tuple(tuple(path) for path in source_occurrence_paths)
    target_regions = tuple(str(value).strip() for value in target_region_ids)
    if (
        len(set(source_global_ids)) != len(source_global_ids)
        or len(set(target_global_ids)) != len(target_global_ids)
        or len(source_entities) != len(source_global_ids)
        or len(source_regions) != len(source_global_ids)
        or len(source_paths) != len(source_global_ids)
        or len(target_regions) != len(target_global_ids)
        or any(
            not value for value in (*source_entities, *source_regions, *target_regions)
        )
        or any(
            any(
                not isinstance(part, str) or not part or part != part.strip()
                for part in path
            )
            for path in source_paths
        )
    ):
        raise MaterialAssociationTransferError(
            "Material source/target side rows must align unique endpoint cell global IDs."
        )

    source_side_by_id = {
        identifier: (entity, region, path)
        for identifier, entity, region, path in zip(
            source_global_ids, source_entities, source_regions, source_paths, strict=True
        )
    }
    target_id_set = set(target_global_ids)
    relation_sources = tuple(
        int(value) for value in np.asarray(cell_lineage.source_global_ids, dtype=np.int64)
    )
    relation_targets = tuple(
        int(value) for value in np.asarray(cell_lineage.target_global_ids, dtype=np.int64)
    )
    relation_kinds = tuple(
        int(value) for value in np.asarray(cell_lineage.relation_kinds, dtype=np.int32)
    )
    if len(set(zip(relation_sources, relation_targets, strict=True))) != len(
        relation_sources
    ):
        raise MaterialAssociationTransferError(
            "Cell lineage repeats a source/target contributor pair."
        )
    created_ids = tuple(
        int(value)
        for value in np.asarray(cell_lineage.created_target_ids, dtype=np.int64)
    )
    if set(relation_targets).union(created_ids) != target_id_set:
        missing_rows = np.asarray(
            [
                row
                for row, identifier in enumerate(target_global_ids)
                if identifier not in set(relation_targets).union(created_ids)
            ],
            dtype=np.int64,
        )
        raise MaterialAssociationTransferError(
            "Cell lineage does not account for every target material cell.",
            target_unmatched_rows=missing_rows,
        )
    supported_kinds = {
        EntityLineageKind.PRESERVED,
        EntityLineageKind.REFINED_FROM,
        EntityLineageKind.COARSENED_INTO,
        EntityLineageKind.SPLIT_FROM,
        EntityLineageKind.MERGED_INTO,
        EntityLineageKind.COLLAPSED_INTO,
        EntityLineageKind.SWAPPED_FROM,
        EntityLineageKind.RELOCATED,
    }
    parents: dict[int, list[tuple[str, str, tuple[str, ...]]]] = {}
    for source_id, target_id, kind_value in zip(
        relation_sources, relation_targets, relation_kinds, strict=True
    ):
        if target_id not in target_id_set:
            raise MaterialAssociationTransferError(
                "Cell lineage names a target cell outside the accepted target mesh."
            )
        try:
            kind = EntityLineageKind(kind_value)
        except ValueError as error:
            raise MaterialAssociationTransferError(
                "Cell lineage contains an unsupported material relation kind."
            ) from error
        if kind not in supported_kinds:
            raise MaterialAssociationTransferError(
                "Created-on-geometry or unknown cell lineage cannot inherit material identity."
            )
        source_side = source_side_by_id.get(source_id)
        if source_side is None:
            raise MaterialAssociationTransferError(
                "Cell lineage names a source cell without retained material association."
            )
        parents.setdefault(target_id, []).append(source_side)

    target_entities = []
    target_paths = []
    conflicting_rows = []
    unmatched_rows = []
    for row, (identifier, region) in enumerate(
        zip(target_global_ids, target_regions, strict=True)
    ):
        contributors = parents.get(identifier, ())
        if not contributors:
            unmatched_rows.append(row)
            continue
        side = contributors[0]
        if any(value != side for value in contributors[1:]) or region != side[1]:
            conflicting_rows.append(row)
            continue
        target_entities.append(side[0])
        target_paths.append(side[2])
    if unmatched_rows or conflicting_rows:
        raise MaterialAssociationTransferError(
            "Target material sides are not uniquely inherited from agreeing cell-lineage contributors.",
            target_unmatched_rows=np.asarray(unmatched_rows, dtype=np.int64),
            target_conflicting_rows=np.asarray(conflicting_rows, dtype=np.int64),
            source_region_ids=source_regions,
            target_region_ids=target_regions,
            source_occurrence_paths=source_paths,
            target_occurrence_paths=tuple(target_paths),
        )
    return MaterialAssociationRows(
        target_global_ids,
        tuple(target_entities),
        target_regions,
        tuple(target_paths),
        lineage.lineage_id,
    )


class EulerianMaterialSourceAssociationTransfer(StrictModule, NonTrainableState):
    """Sparse, overlap-weighted source-side relations for every target material cell.

    The rows retain entity, material-region, and occurrence identity independently.
    ``overlap_measures`` are physical overlap measures, not normalized weights.
    """

    volume_overlap: EulerianVolumeOverlapMap
    target_offsets: Array
    source_cells: Array
    overlap_measures: Array
    source_cell_global_ids: Array
    target_cell_global_ids: Array
    source_entity_ids: tuple[str, ...] = eqx.field(static=True)
    target_entity_ids: tuple[str, ...] = eqx.field(static=True)
    source_region_ids: tuple[str, ...] = eqx.field(static=True)
    target_region_ids: tuple[str, ...] = eqx.field(static=True)
    source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    target_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_coverage_defects: Array
    target_coverage_defects: Array
    unmatched_source_rows: Array
    unmatched_target_rows: Array
    transfer_id: str = eqx.field(static=True)

    def __init__(self, volume_overlap: EulerianVolumeOverlapMap, /) -> None:
        if not isinstance(volume_overlap, EulerianVolumeOverlapMap):
            raise TypeError("volume_overlap must be EulerianVolumeOverlapMap.")
        refinement = volume_overlap.refinement
        source_rows = np.asarray(refinement.source_cells, dtype=np.int64)
        target_rows = np.asarray(refinement.target_cells, dtype=np.int64)
        offsets = np.asarray(refinement.target_offsets, dtype=np.int64)
        overlap = np.asarray(refinement.volumes, dtype=np.float64)
        source_count = len(volume_overlap.source_cell_entity_ids)
        target_count = len(volume_overlap.target_cell_entity_ids)
        source_unmatched = np.flatnonzero(
            np.bincount(source_rows, minlength=source_count) == 0
        )
        target_unmatched = np.flatnonzero(np.diff(offsets) == 0)
        if (
            not refinement.succeeded
            or refinement.policy.coverage.value != "complete"
            or offsets.shape != (target_count + 1,)
            or source_rows.shape != target_rows.shape
            or source_rows.shape != overlap.shape
            or np.any(source_rows < 0)
            or np.any(source_rows >= source_count)
            or np.any(target_rows < 0)
            or np.any(target_rows >= target_count)
            or np.any(overlap <= 0.0)
            or source_unmatched.size
            or target_unmatched.size
        ):
            raise MaterialAssociationTransferError(
                "Material associations require complete positive physical overlap coverage.",
                source_unmatched_rows=source_unmatched,
                target_unmatched_rows=target_unmatched,
                source_region_ids=volume_overlap.source_region_ids,
                target_region_ids=volume_overlap.target_region_ids,
                source_occurrence_paths=volume_overlap.source_occurrence_paths,
                target_occurrence_paths=volume_overlap.target_occurrence_paths,
            )
        source_regions = volume_overlap.source_region_ids
        target_regions = volume_overlap.target_region_ids
        source_paths = volume_overlap.source_occurrence_paths
        target_paths = volume_overlap.target_occurrence_paths
        for source_row, target_row in zip(source_rows, target_rows, strict=True):
            if (
                source_regions[int(source_row)] != target_regions[int(target_row)]
                or source_paths[int(source_row)] != target_paths[int(target_row)]
            ):
                raise MaterialAssociationTransferError(
                    "Overlap crosses an unmatched material region or occurrence side.",
                    source_region_ids=source_regions,
                    target_region_ids=target_regions,
                    source_occurrence_paths=source_paths,
                    target_occurrence_paths=target_paths,
                )
        source_ids = np.concatenate(
            tuple(
                np.asarray(block.global_ids, dtype=np.int64)
                for block in volume_overlap.source_mesh.blocks
            )
        )
        target_ids = np.concatenate(
            tuple(
                np.asarray(block.global_ids, dtype=np.int64)
                for block in volume_overlap.target_mesh.blocks
            )
        )
        if (
            offsets[0] != 0
            or offsets[-1] != source_rows.size
            or np.any(np.diff(offsets) < 0)
            or np.any(np.diff(target_rows) < 0)
            or any(
                np.any(target_rows[int(offsets[row]) : int(offsets[row + 1])] != row)
                for row in range(target_count)
            )
        ):
            raise MaterialAssociationTransferError(
                "Physical overlap CSR rows do not align with target material cells.",
                source_region_ids=source_regions,
                target_region_ids=target_regions,
                source_occurrence_paths=source_paths,
                target_occurrence_paths=target_paths,
            )
        source_defects = np.asarray(
            volume_overlap.source_coverage_defect, dtype=np.float64
        )
        target_defects = np.asarray(
            volume_overlap.target_coverage_defect, dtype=np.float64
        )
        self.volume_overlap = volume_overlap
        self.target_offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.source_cells = jnp.asarray(source_rows, dtype=jnp.int32)
        self.overlap_measures = jnp.asarray(overlap)
        self.source_cell_global_ids = jnp.asarray(source_ids, dtype=jnp.int64)
        self.target_cell_global_ids = jnp.asarray(target_ids, dtype=jnp.int64)
        self.source_entity_ids = volume_overlap.source_cell_entity_ids
        self.target_entity_ids = volume_overlap.target_cell_entity_ids
        self.source_region_ids = source_regions
        self.target_region_ids = target_regions
        self.source_occurrence_paths = source_paths
        self.target_occurrence_paths = target_paths
        self.source_coverage_defects = jnp.asarray(source_defects)
        self.target_coverage_defects = jnp.asarray(target_defects)
        self.unmatched_source_rows = jnp.asarray(source_unmatched, dtype=jnp.int32)
        self.unmatched_target_rows = jnp.asarray(target_unmatched, dtype=jnp.int32)
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "eulerian-material-source-association-transfer",
                "volume_overlap": volume_overlap.map_id,
                "source_global_ids": array_tree_fingerprint(source_ids),
                "target_global_ids": array_tree_fingerprint(target_ids),
                "source_entities": self.source_entity_ids,
                "target_entities": self.target_entity_ids,
                "source_regions": source_regions,
                "target_regions": target_regions,
                "source_occurrences": source_paths,
                "target_occurrences": target_paths,
                "target_offsets": array_tree_fingerprint(offsets),
                "source_cells": array_tree_fingerprint(source_rows),
                "overlap_measures": array_tree_fingerprint(overlap),
            }
        )

    def source_sides_for_target(
        self, target_row: int, /
    ) -> tuple[MaterialAssociationSide, ...]:
        row = int(target_row)
        if row < 0 or row >= len(self.target_entity_ids):
            raise IndexError(
                "target_row is outside the Eulerian material association transfer."
            )
        offsets = np.asarray(self.target_offsets, dtype=np.int64)
        source_rows = np.asarray(self.source_cells, dtype=np.int64)
        measures = np.asarray(self.overlap_measures, dtype=np.float64)
        return tuple(
            MaterialAssociationSide(
                int(np.asarray(self.source_cell_global_ids)[source_row]),
                self.source_entity_ids[source_row],
                self.source_region_ids[source_row],
                self.source_occurrence_paths[source_row],
                float(measures[entry]),
            )
            for entry, source_row in zip(
                range(int(offsets[row]), int(offsets[row + 1])),
                source_rows[int(offsets[row]) : int(offsets[row + 1])],
                strict=True,
            )
        )

    def target_side(self, target_row: int, /) -> TargetMaterialAssociationSide:
        row = int(target_row)
        if row < 0 or row >= len(self.target_entity_ids):
            raise IndexError(
                "target_row is outside the Eulerian material association transfer."
            )
        return TargetMaterialAssociationSide(
            int(np.asarray(self.target_cell_global_ids)[row]),
            self.target_entity_ids[row],
            self.target_region_ids[row],
            self.target_occurrence_paths[row],
            float(np.asarray(self.volume_overlap.refinement.target_measures)[row]),
        )


def prepare_eulerian_material_source_association_transfer(
    volume_overlap: EulerianVolumeOverlapMap, /
) -> EulerianMaterialSourceAssociationTransfer:
    """Bind material identity sides to an Eulerian physical-overlap CSR."""
    return EulerianMaterialSourceAssociationTransfer(volume_overlap)


__all__ = [
    "MaterialAssociationSide",
    "MaterialAssociationRows",
    "MaterialAssociationTransferError",
    "EulerianMaterialSourceAssociationTransfer",
    "TargetMaterialAssociationSide",
    "prepare_eulerian_material_source_association_transfer",
    "propagate_material_association_rows",
]
