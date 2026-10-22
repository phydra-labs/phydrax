#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bound layer source phases and reserve accepted storage before native PLC fill."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

import phydrax.meshing._boundary_layer as _dependency_phydrax_meshing__boundary_layer
import phydrax.meshing._canonical as _dependency_phydrax_meshing__canonical
import phydrax.meshing._contracts as _dependency_phydrax_meshing__contracts
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._trace as _dependency_phydrax_meshing__trace

from .._meshcore import NativeBatchTicket, NativeExecutionScope
from .._resource_ticket import BatchTicket
from .._validation import nonnegative_integer, positive_integer
from ..discretization import CellMesh
from ..typing import checked
from ._family_execution import family_route_refusals
from .providers import _native_sources as sources_owner


# Logical host payload retained by one charged composition record: an int64
# identifier plus at most eight int64 vertex IDs (the hexahedron row). Every
# host array, key, and numeric evidence row of a metered layer phase is created
# by a charged record, so this bounds the phase's live array payload. Python
# object headers and allocator metadata are not managed bytes.
HOST_RECORD_BYTES = 72


@checked
def layer_source_work_upper(
    source: sources_owner.NativeLayerCoreSource,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    /,
) -> int:
    """Bound identity, orbit, seam, and full requested-pair source visits."""
    periodic = source.layers.mesh.periodic_topology
    rank = 0 if periodic is None else periodic.cell.rank
    pairs = source.core_seam_polygon_pairs
    seam_count = 0 if pairs is None else pairs.shape[0]
    claims = sum(
        np.asarray(constraint.target_scope.entity_ids).size
        for constraint in specification.periodic_constraints
    )
    vertices = source.complex.vertices.shape[0]
    entries = source.complex.polygon_vertices.size
    polygons = source.complex.polygon_facets.size
    return (
        (8 + 2 * rank) * vertices
        + source.layers.mesh.coordinates.shape[0]
        + 3 * entries
        + 5 * seam_count
        + source.cap_polygon_ids.size
        + claims * (2 * polygons + 2 * seam_count)
    )


@dataclass(slots=True)
class LayerCoreSourceWork:
    """Count source records locally on the request's admitted phase ticket."""

    maximum: int
    work_units: int = 0
    ticket: BatchTicket = field(kw_only=True)
    _storage_ticket: BatchTicket | None = field(default=None, init=False)
    _storage_bytes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.maximum = positive_integer(self.maximum, "maximum_work_units")
        self.work_units = nonnegative_integer(self.work_units, "source_work_units")
        self.charge(0)

    def charge(self, count: int, /) -> None:
        count = nonnegative_integer(count, "source_work_units")
        if count > self.maximum - self.work_units:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"LayerCoreSourceWork.maximum must admit the next source records; got {self.work_units} consumed and {count} requested with maximum {self.maximum}",
                stage=_dependency_phydrax_meshing__trace.MeshingStageKind.SOURCE_INSPECTION.value,
                requested=(("maximum_work_units", self.maximum),),
                achieved=(
                    ("work_units", self.work_units),
                    ("next_source_records", count),
                ),
            )
        if count > self.ticket.remaining_work_units:
            # A request-wide shortfall is a native refusal; a fitting
            # increment is an estimate defect of the admitted phase bound.
            self.ticket.refuse(count)
        self.ticket.consume(count)
        self.work_units += count

    def admit_storage(self, bytes_upper: int, /) -> None:
        """Declare a phase's bounded live host arrays before allocating them."""
        bytes_upper = nonnegative_integer(bytes_upper, "host_bytes_upper")
        ticket = self.ticket
        if not isinstance(ticket, NativeBatchTicket):
            raise TypeError(
                "LayerCoreSourceWork.ticket must be a native request reservation for host arrays; got "
                f"{type(ticket).__name__}"
            )
        current = self._storage_bytes if self._storage_ticket is ticket else 0
        available = ticket.remaining_managed_bytes + current
        if bytes_upper > available:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "LayerCoreSourceWork.ticket must reserve the phase's host arrays before allocation; "
                f"got {bytes_upper} bytes requested with {available} reserved",
                stage=_dependency_phydrax_meshing__trace.MeshingStageKind.CANONICALIZATION.value,
                requested=(("reserved_scratch_bytes", available),),
                achieved=(("host_array_bytes_upper", bytes_upper),),
            )
        ticket.consume(managed_bytes=bytes_upper)
        self._storage_ticket = ticket
        self._storage_bytes = bytes_upper


@contextmanager
@checked
def admit_layer_phase(
    scope: NativeExecutionScope,
    limits: _dependency_phydrax_meshing__contracts.MeshingLimits,
    work: LayerCoreSourceWork,
    stage: _dependency_phydrax_meshing__trace.MeshingStageKind,
    /,
    *,
    work_upper: int,
) -> Iterator[NativeBatchTicket]:
    """Admit one host phase's work and array payload before it allocates.

    The record bound covers every array the phase creates; the reservation is
    a native request ticket, so the request's work and managed-byte allowance
    refuses the phase before any of its storage exists.
    """
    work_upper = nonnegative_integer(work_upper, "work_upper")
    bytes_upper = HOST_RECORD_BYTES * work_upper
    if (
        work_upper > limits.maximum_work_units - work.work_units
        or bytes_upper > limits.maximum_scratch_bytes
    ):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"{stage.value}.admission must fit the request's work and scratch limits before allocation; "
            f"got {work_upper} work units and {bytes_upper} bytes after {work.work_units} consumed",
            stage=stage.value,
            requested=(
                ("maximum_work_units", limits.maximum_work_units),
                ("maximum_scratch_bytes", limits.maximum_scratch_bytes),
            ),
            achieved=(
                ("work_units", work.work_units),
                ("phase_work_units_upper", work_upper),
                ("phase_host_bytes_upper", bytes_upper),
            ),
        )
    with family_route_refusals(limits, stage.value):
        ticket = scope.batch(work_upper, bytes_upper=bytes_upper)
    with ticket:
        work.ticket = ticket
        work.admit_storage(bytes_upper)
        yield ticket


@checked
def row_entity_vertex_keys(
    mesh: CellMesh, dimension: int, /
) -> tuple[tuple[int, ...], ...]:
    """Lower declared global vertex keys onto the carrier's execution rows."""
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
        for key in _dependency_phydrax_meshing__canonical.entity_vertex_keys(
            mesh, dimension
        )
    )


@checked
def reserve_layer_storage(
    layers: _dependency_phydrax_meshing__boundary_layer.BoundaryLayerMesh,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    specification: _dependency_phydrax_meshing__contracts.VolumeMeshingSpec,
    vertex_layer_ids: np.ndarray,
    /,
) -> _dependency_phydrax_meshing__contracts.VolumeMeshingSpec:
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
    layer_edges = set(row_entity_vertex_keys(layers.mesh, 1))
    layer_faces = set(row_entity_vertex_keys(layers.mesh, 2))
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "VolumeMeshingSpec.limits must admit accepted layers and the remaining fixed PLC storage before core fill; got accepted layers and immutable PLC consume the combined allocation budget before core fill",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
            requested=tuple((f"maximum_{name}", value) for name, value in bounds.items()),
            achieved=tuple(
                (f"reserved_layer_{name}", value) for name, value in reserved.items()
            ),
        )
    core_limits = _dependency_phydrax_meshing__contracts.MeshingLimits(
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
    return _dependency_phydrax_meshing__contracts.VolumeMeshingSpec(
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
