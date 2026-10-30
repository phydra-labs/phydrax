#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from ...discretization._cell_mesh import CellMesh
from ...meshing._contracts import SurfaceMeshingSpec, VolumeMeshingSpec
from ...meshing._sizing import UniformSizeControl
from ...typing import checked
from ._native_periodic_algebraic import periodic_quotient_edge_squared_lengths


@checked
def periodic_size_compliance(
    mesh: CellMesh,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    source_id: str,
    source_revision: str,
    source_entity_ids: np.ndarray,
    /,
    *,
    source_entity_set_id: str,
    target_scope_entity_ids: tuple[np.ndarray, ...] | None = None,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...], tuple[str, ...]]:
    """Measure declared uniform targets on exact quotient-edge incidence rows."""
    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if not controls or len(controls) != len(specification.size_controls):
        raise ValueError("Periodic Delaunay requires explicit uniform quotient-edge size controls.")
    edge_ids, endpoints, squared = periodic_quotient_edge_squared_lengths(mesh)
    lengths = np.asarray([np.sqrt(float(value)) for value in squared], dtype=np.float64)
    source_ids = set(map(int, np.asarray(source_entity_ids, dtype=np.int64)))
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    statistics = tuple(specification.size_compliance.target_statistics)
    tolerance = specification.size_compliance
    if target_scope_entity_ids is not None and len(target_scope_entity_ids) != len(controls):
        raise ValueError("Periodic target scope rows must align with the size controls.")
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError("Periodic size compliance requires attached quotient topology.")
    target_vertex_ids = set(
        map(int, np.asarray(topology.quotient.entity_sets[0].entity_ids))
    )
    for control_index, control in enumerate(controls):
        scope = control.scope
        if (
            scope.source_id != source_id
            or scope.source_revision != source_revision
            or scope.entity_dimension != 0
            or scope.entity_set_id != source_entity_set_id
        ):
            raise ValueError("Periodic uniform size scopes must select vertices of the bound source revision.")
        selected = set(map(int, np.asarray(scope.entity_ids, dtype=np.int64)))
        if not selected or not selected <= source_ids:
            raise ValueError("Periodic uniform size scope contains no valid source vertex IDs.")
        selected_target = selected if target_scope_entity_ids is None else set(
            map(int, np.asarray(target_scope_entity_ids[control_index], dtype=np.int64))
        )
        if not selected_target or not selected_target <= target_vertex_ids:
            raise ValueError("Periodic target size scope contains no valid quotient vertices.")
        mask = np.asarray(
            [set(map(int, pair)) <= selected_target for pair in endpoints], dtype=np.bool_
        )
        values = lengths[mask]
        if values.size == 0:
            raise ValueError("Periodic uniform size scope selects no quotient edges.")
        for statistic in statistics:
            quantile = 0.50 if statistic == "p50" else 0.95
            actual = float(np.quantile(values, quantile, method="linear"))
            name = f"periodic-size:{control.control_id}:{statistic}"
            requested.append((name, control.target_size))
            achieved.append((name, actual))
            allowed = tolerance.absolute_tolerance + tolerance.relative_tolerance * control.target_size
            if abs(actual - control.target_size) > allowed:
                issues.append(
                    f"Periodic quotient-edge {statistic} size {actual:g} differs from target {control.target_size:g} beyond the declared compliance tolerance."
                )
        if control.minimum_size is not None and np.any(values < control.minimum_size):
            issues.append("Periodic quotient edges violate a declared minimum size.")
        if control.maximum_size is not None and np.any(values > control.maximum_size):
            issues.append("Periodic quotient edges violate a declared maximum size.")
    return tuple(requested), tuple(achieved), tuple(issues)

@checked
def periodic_metric_from_uniform_controls(
    mesh: CellMesh,
    specification: SurfaceMeshingSpec | VolumeMeshingSpec,
    source_id: str,
    source_revision: str,
    source_entity_ids: np.ndarray,
    source_entity_set_id: str,
) -> np.ndarray:
    """Build lifted isotropic metric tensors from complete quotient vertex controls."""
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError("Periodic metric assembly requires periodic mesh topology.")
    topology.require_lifted(mesh.topology, mesh.lifted_topology_id, mesh.blocks)
    controls = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    if not controls or len(controls) != len(specification.size_controls):
        raise ValueError("Periodic metric adaptation requires uniform quotient-vertex size controls.")
    source_ids = np.asarray(source_entity_ids, dtype=np.int64)
    quotient_ids = np.asarray(
        topology.quotient.entity_sets[0].entity_ids, dtype=np.int64
    )
    if (
        source_ids.ndim != 1
        or np.unique(source_ids).size != source_ids.size
        or not np.array_equal(np.sort(source_ids), np.sort(quotient_ids))
    ):
        raise ValueError("Periodic size controls must cover the authored quotient vertex identities.")
    target_by_id: dict[int, float] = {}
    for control in controls:
        scope = control.scope
        if (
            scope.source_id != source_id
            or scope.source_revision != source_revision
            or scope.entity_dimension != 0
            or scope.entity_set_id != source_entity_set_id
        ):
            raise ValueError("Periodic size controls must select source vertices at the bound revision.")
        selected = np.asarray(scope.entity_ids, dtype=np.int64)
        if selected.size == 0 or np.setdiff1d(selected, source_ids).size:
            raise ValueError("Periodic size controls must select authored quotient vertices.")
        for identifier in selected:
            key = int(identifier)
            previous = target_by_id.get(key)
            if previous is not None and previous != control.target_size:
                raise ValueError("Overlapping periodic size controls prescribe incompatible target sizes.")
            target_by_id[key] = float(control.target_size)
    if set(target_by_id) != set(map(int, source_ids)):
        raise ValueError("Periodic uniform size controls must cover every quotient vertex.")
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    lifted_begin, lifted_end = map(int, lifted_offsets[:2])
    quotient_begin = int(quotient_offsets[0])
    orbit_rows = np.asarray(topology.orbit_indices, dtype=np.int64)[
        lifted_begin:lifted_end
    ] - quotient_begin
    lifted_quotient_ids = quotient_ids[orbit_rows]
    lengths = np.asarray(
        [target_by_id[int(identifier)] for identifier in lifted_quotient_ids],
        dtype=np.float64,
    )
    return np.eye(mesh.ambient_dimension, dtype=np.float64)[None, :, :] / lengths[:, None, None] ** 2


__all__ = ["periodic_metric_from_uniform_controls", "periodic_size_compliance"]
