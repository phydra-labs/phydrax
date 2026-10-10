#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reserve accepted layer storage before allocating a native fixed-PLC core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._meshcore import charge_native_geometry_queries
from .._validation import nonnegative_integer, positive_integer
from ..discretization import CellMesh
from ._boundary_layer import BoundaryLayerMesh
from ._canonical import _entity_vertex_keys
from ._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    VolumeMeshingSpec,
)
from ._trace import MeshingStageKind
from ._volume_generation import PiecewiseLinearComplex


@dataclass(slots=True)
class LayerCoreSourceWork:
    """Debit actual source records and consistency tests before dispatch."""

    maximum: int
    work_units: int = 0

    def __post_init__(self) -> None:
        self.maximum = positive_integer(self.maximum, "maximum_work_units")
        self.work_units = nonnegative_integer(self.work_units, "source_work_units")
        self.charge(0)

    def charge(self, count: int, /) -> None:
        count = nonnegative_integer(count, "source_work_units")
        if count > self.maximum - self.work_units:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Layer/core source preparation exhausted its original work budget.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
                requested=(("maximum_work_units", self.maximum),),
                achieved=(
                    ("work_units", self.work_units),
                    ("next_source_records", count),
                ),
            )
        charge_native_geometry_queries(0, work_units=count)
        self.work_units += count


def _row_entity_vertex_keys(
    mesh: CellMesh, dimension: int, /
) -> tuple[tuple[int, ...], ...]:
    """Lower authoritative global vertex keys onto the carrier's execution rows."""
    rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(mesh.vertex_global_ids))
    }
    if dimension == 0:
        return tuple(
            (rows[int(identifier)],)
            for identifier in np.asarray(mesh.entity_set(0).entity_ids)
        )
    return tuple(
        tuple(sorted(rows[identifier] for identifier in key))
        for key in _entity_vertex_keys(mesh, dimension)
    )


def reserve_layer_storage(
    layers: BoundaryLayerMesh,
    complex_: PiecewiseLinearComplex,
    specification: VolumeMeshingSpec,
    vertex_layer_ids: np.ndarray,
    /,
) -> VolumeMeshingSpec:
    limits = specification.limits
    points = layers.mesh.coordinates.shape[0]
    shared = nonnegative_integer(
        np.count_nonzero(vertex_layer_ids >= 0), "shared_vertex_count"
    )
    polygons = complex_.polygon_vertices.reshape(-1, 3)
    mapped = vertex_layer_ids[polygons]
    plc_edges = {
        tuple(sorted((int(row[a]), int(row[b]))))
        for row in mapped
        for a, b in ((0, 1), (1, 2), (2, 0))
        if row[a] >= 0 and row[b] >= 0
    }
    layer_edges = set(_row_entity_vertex_keys(layers.mesh, 1))
    layer_faces = set(_row_entity_vertex_keys(layers.mesh, 2))
    shared_faces = {
        tuple(sorted(row.tolist())) for row in mapped if np.all(row >= 0)
    } & layer_faces
    cells = sum(block.cell_count for block in layers.mesh.blocks)
    entries = sum(block.vertices.size for block in layers.mesh.blocks)
    reserved = {
        "vertices": points - shared,
        "edges": len(layer_edges - plc_edges),
        "faces": len(layer_faces - shared_faces),
        "cells": cells,
        "connectivity_entries": entries,
        "data_bytes": (points - shared) * 24 + entries * 4,
    }
    bounds = {
        "vertices": limits.maximum_vertices,
        "edges": limits.maximum_edges,
        "faces": limits.maximum_faces,
        "cells": limits.maximum_cells,
        "connectivity_entries": limits.maximum_connectivity_entries,
        "data_bytes": limits.maximum_data_bytes,
    }
    remaining = {name: bounds[name] - used for name, used in reserved.items()}
    if (
        any(value <= 0 for value in remaining.values())
        or remaining["vertices"] < complex_.vertices.shape[0]
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Accepted layers and immutable PLC consume the combined allocation budget before core fill.",
            stage=MeshingStageKind.VOLUME_FILL.value,
            requested=tuple((f"maximum_{name}", value) for name, value in bounds.items()),
            achieved=tuple(
                (f"reserved_layer_{name}", value) for name, value in reserved.items()
            ),
        )
    core_limits = MeshingLimits(
        maximum_vertices=remaining["vertices"],
        maximum_edges=remaining["edges"],
        maximum_faces=remaining["faces"],
        maximum_cells=remaining["cells"],
        maximum_connectivity_entries=remaining["connectivity_entries"],
        maximum_data_bytes=remaining["data_bytes"],
        maximum_work_units=limits.maximum_work_units,
        maximum_cavity_cells=limits.maximum_cavity_cells,
        maximum_geometry_queries=limits.maximum_geometry_queries,
        maximum_scratch_bytes=limits.maximum_scratch_bytes,
        maximum_wall_seconds=limits.maximum_wall_seconds,
    )
    return VolumeMeshingSpec(
        specification.target,
        specification.boundary_scope,
        specification.fill_strategy,
        size_controls=specification.size_controls,
        protected_features=specification.protected_features,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        region_seeds=specification.region_seeds,
        hole_seeds=specification.hole_seeds,
        layer_controls=specification.layer_controls,
        periodic_constraints=specification.periodic_constraints,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=core_limits,
        deterministic=specification.deterministic,
    )


__all__ = ["reserve_layer_storage"]
