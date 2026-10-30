#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact combinatorial source ancestry for mapped PLC carrier simplices."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ._volume_generation import VolumeConstruction


@dataclass(frozen=True, slots=True, eq=False)
class PlcCarrierAncestry:
    """Stable source stratum, occurrence and deviation for each power carrier."""

    dimensions: np.ndarray
    source_entity_ids: tuple[str, ...]
    source_occurrence_paths: tuple[tuple[str, ...], ...]
    deviations: np.ndarray


def _row_key(vertices: tuple[int, ...], /) -> tuple[int, ...]:
    return tuple(sorted(vertices))


def _containing_region(
    carrier: tuple[int, ...],
    simplex_regions: dict[tuple[int, ...], set[int]],
    /,
) -> int | None:
    regions = simplex_regions.get(_row_key(carrier), set())
    if len(regions) != 1:
        return None
    return next(iter(regions))


def classify_plc_carriers(
    construction: VolumeConstruction, vertex_carriers: object, /
) -> PlcCarrierAncestry:
    """Resolve native carrier simplices by source incidence, never proximity.

    Carrier values are native PLC-recovery vertex IDs; ``-1`` is padding. A
    singleton uses its authored vertex/edge/facet witness. Higher-dimensional
    carriers first match constrained segments and faces, then edges of exactly
    one constrained facet (facet diagonals), then the unique region of their
    containing tetrahedra. Ambiguous material junctions fail closed.
    """
    carriers = np.asarray(vertex_carriers)
    if not np.issubdtype(carriers.dtype, np.integer):
        raise TypeError("vertex_carriers must contain integer native vertex IDs.")
    if carriers.ndim != 2 or carriers.shape[1] != 4:
        raise ValueError("vertex_carriers must have shape (vertex_count, 4).")
    if np.any(carriers < -1):
        raise ValueError("vertex_carriers may use only -1 for an empty carrier slot.")

    native_vertices = np.asarray(construction.mesh.vertex_global_ids, dtype=np.int64)
    native_to_row = {int(native): row for row, native in enumerate(native_vertices)}
    vertex_dimensions = np.asarray(construction.vertex_dimension, dtype=np.int64)
    vertex_source_ids = construction.vertex_source_ids
    witness_deviations = np.asarray(construction.witness_deviations, dtype=np.float64)
    boundary_faces = np.asarray(construction.boundary_faces, dtype=np.int64)
    face_source_ids = construction.face_source_ids
    constrained_segments = np.asarray(construction.segments, dtype=np.int64)
    tetrahedra = np.asarray(construction.mesh.blocks[0].vertices, dtype=np.int64)
    cell_regions = np.asarray(construction.cell_regions, dtype=np.int64)

    segment_ids: dict[tuple[int, ...], tuple[str, tuple[str, ...]]] = {}
    for edge, source_id, occurrence_path in zip(
        constrained_segments.tolist(),
        construction.segment_source_ids,
        construction.segment_source_occurrence_paths,
        strict=True,
    ):
        segment_ids[_row_key(tuple(edge))] = (source_id, occurrence_path)
    face_ids: dict[tuple[int, ...], tuple[str, tuple[str, ...]]] = {}
    face_edge_sources: dict[tuple[int, ...], set[tuple[str, tuple[str, ...]]]] = {}
    for face, source_id, occurrence_path in zip(
        boundary_faces.tolist(),
        face_source_ids,
        construction.face_source_occurrence_paths,
        strict=True,
    ):
        face_ids[_row_key(tuple(face))] = (source_id, occurrence_path)
        for edge in combinations(face, 2):
            face_edge_sources.setdefault(_row_key(edge), set()).add(
                (source_id, occurrence_path)
            )

    simplex_regions: dict[tuple[int, ...], set[int]] = {}
    for cell, region in zip(tetrahedra.tolist(), cell_regions.tolist(), strict=True):
        vertices = tuple(cell)
        for dimension in (1, 2, 3, 4):
            for simplex in combinations(vertices, dimension):
                simplex_regions.setdefault(_row_key(simplex), set()).add(region)

    dimensions = np.empty((carriers.shape[0],), dtype=np.int8)
    deviations = np.zeros((carriers.shape[0],), dtype=np.float64)
    entities: list[str] = []
    occurrence_paths: list[tuple[str, ...]] = []
    for vertex, raw in enumerate(carriers.tolist()):
        native = tuple(sorted(value for value in raw if value >= 0))
        if any(first == second for first, second in zip(native, native[1:])):
            raise ValueError("A carrier simplex cannot repeat a native PLC vertex.")
        if not native:
            raise ValueError(
                "Every power vertex must have a nonempty PLC carrier simplex."
            )
        try:
            rows = tuple(native_to_row[value] for value in native)
        except KeyError as error:
            raise ValueError(
                "A power vertex carrier references a vertex outside the PLC mesh."
            ) from error

        if len(rows) == 1:
            row = rows[0]
            source_id = vertex_source_ids[row]
            dimension = int(vertex_dimensions[row])
            if source_id and 0 <= dimension <= 2:
                entities.append(source_id)
                occurrence_paths.append(construction.vertex_source_occurrence_paths[row])
                dimensions[vertex] = dimension
                deviations[vertex] = witness_deviations[row]
                continue
        source = segment_ids.get(_row_key(rows)) if len(rows) == 2 else None
        if source is not None:
            source_id, occurrence_path = source
            entities.append(source_id)
            occurrence_paths.append(occurrence_path)
            dimensions[vertex] = 1
            deviations[vertex] = max(witness_deviations[row] for row in rows)
            continue
        source = face_ids.get(_row_key(rows)) if len(rows) == 3 else None
        if source is not None:
            source_id, occurrence_path = source
            entities.append(source_id)
            occurrence_paths.append(occurrence_path)
            dimensions[vertex] = 2
            deviations[vertex] = max(witness_deviations[row] for row in rows)
            continue
        # A constrained-facet edge that is not an authored segment is a facet
        # diagonal; its unique incident facet is the carrier's source, even on a
        # two-region interface where the containing tetrahedra are ambiguous.
        facet_sources = (
            face_edge_sources.get(_row_key(rows), set()) if len(rows) == 2 else set()
        )
        if len(facet_sources) == 1:
            source_id, occurrence_path = next(iter(facet_sources))
            entities.append(source_id)
            occurrence_paths.append(occurrence_path)
            dimensions[vertex] = 2
            deviations[vertex] = max(witness_deviations[row] for row in rows)
            continue
        region = _containing_region(rows, simplex_regions)
        if region is None:
            raise ValueError(
                "A PLC carrier simplex has ambiguous or missing source ancestry."
            )
        dimensions[vertex] = 3
        entities.append(construction.region_ids[region])
        occurrence_paths.append(construction.region_occurrence_paths[region])
    dimensions.setflags(write=False)
    deviations.setflags(write=False)
    return PlcCarrierAncestry(
        dimensions,
        tuple(entities),
        tuple(occurrence_paths),
        deviations,
    )


__all__ = ["PlcCarrierAncestry", "classify_plc_carriers"]
