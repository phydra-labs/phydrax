#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-authored seam anatomy for fixed-PLC layer/core assembly."""

from __future__ import annotations

import numpy as np

import phydrax.discretization._periodic_topology as periodic_topology_owner
import phydrax.meshing._layer_core_controls as controls_owner
import phydrax.meshing._layer_core_topology as layer_topology_owner
import phydrax.meshing.providers._native_sources as sources_owner
from phydrax.discretization._periodic_exact import (
    exact_periodic_element,
    identification_orders,
    reduce_periodic_exponents,
)

from ..discretization import CellMesh
from ..typing import checked
from ._contracts import VolumeMeshingSpec


def _require_seams(
    source: sources_owner.NativeLayerCoreSource,
    polygons: np.ndarray,
    roots: np.ndarray,
    shifts: np.ndarray,
    /,
) -> None:
    """Prove source-declared seam pairs exactly cover oriented periodic facets."""
    periodic = source.layers.mesh.periodic_topology
    pairs = source.core_seam_polygon_pairs
    if periodic is None or pairs is None:
        raise ValueError(
            "Periodic PLC seams require retained layer identification and explicit seam anatomy."
        )
    pairs = np.asarray(pairs, dtype=np.int64)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or np.any(pairs < 0)
        or np.any(pairs >= polygons.shape[0])
    ):
        raise ValueError("Core seam pairs must index two PLC triangles per seam.")
    if (
        np.unique(pairs).size != pairs.size
        or np.intersect1d(pairs, np.asarray(source.cap_polygon_ids, dtype=np.int64)).size
    ):
        raise ValueError("Core seam triangles must be distinct non-cap PLC polygons.")
    orders = identification_orders(periodic.cell)
    fixed = periodic_topology_owner.fixed_generators(
        periodic.cell, source.complex.vertices, roots
    )
    keys, signs, anchors = periodic_topology_owner.oriented_forms(
        roots[polygons], shifts[polygons], fixed[polygons], orders
    )
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(tuple(key.tolist()), []).append(index)
    repeated = {tuple(sorted(group)) for group in groups.values() if len(group) > 1}
    declared = {tuple(sorted(pair.tolist())) for pair in pairs}
    if repeated != declared:
        raise ValueError(
            "Explicit seam pairs must cover every periodic polygon orbit exactly once."
        )
    complex_ = source.complex
    for first, second in pairs:
        delta = reduce_periodic_exponents(anchors[second] - anchors[first], orders)
        if signs[first] * signs[second] != -1 or not np.any(delta):
            raise ValueError(
                "Periodic seam pairs must have opposite orientation and a nontrivial group image."
            )
        incidence = complex_.facet_regions[
            complex_.polygon_facets[np.asarray((first, second), dtype=np.int64)]
        ]
        if np.any(np.count_nonzero(incidence >= 0, axis=1) != 1):
            raise ValueError("Periodic seam polygons must be exterior core facets.")
        if not np.array_equal(incidence[0], incidence[1]):
            raise ValueError(
                "Periodic seam copies must preserve identical occupied-region incidence."
            )


@checked
def prepare_core_periodic(
    source: sources_owner.NativeLayerCoreSource,
    mapping: np.ndarray,
    polygons: np.ndarray,
    points: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compile all core boundary orbits before fill; never pair generated nodes."""
    periodic = source.layers.mesh.periodic_topology
    supplied = source.core_vertex_representatives
    exponents = source.core_vertex_shifts
    if periodic is None:
        if (
            supplied is not None
            or exponents is not None
            or source.core_seam_polygon_pairs is not None
        ):
            raise ValueError(
                "Periodic core ancestry requires an accepted periodic layer mesh."
            )
        return None
    if supplied is None or exponents is None:
        raise ValueError(
            "Periodic core construction requires source-authored vertex orbits and seam anatomy."
        )
    roots, shifts = periodic_topology_owner.validate_periodic_vertex_orbits(
        source.complex.vertices,
        periodic.cell,
        supplied,
        exponents,
    )
    _require_seams(source, polygons, roots, shifts)
    layer_count = source.layers.mesh.coordinates.shape[0]
    layer_roots = np.asarray(periodic.vertex_representatives, dtype=np.int64)
    layer_shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
    combined_roots = np.arange(points.shape[0], dtype=np.int64)
    combined_shifts = np.zeros((points.shape[0], periodic.cell.rank), dtype=np.int64)
    combined_roots[:layer_count] = layer_roots
    combined_shifts[:layer_count] = layer_shifts
    orders = identification_orders(periodic.cell)
    member_order = np.argsort(roots, kind="stable")
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.diff(roots[member_order])) + 1,
            np.asarray([roots.size], dtype=np.int64),
        )
    )
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        members = member_order[start:end]
        root = int(roots[members[0]])
        shared = members[
            np.asarray(source.vertex_layer_ids, dtype=np.int64)[members] >= 0
        ]
        if shared.size:
            layer_rows = np.asarray(source.vertex_layer_ids, dtype=np.int64)[shared]
            target_root = int(layer_roots[layer_rows[0]])
            offsets = reduce_periodic_exponents(
                layer_shifts[layer_rows] - shifts[shared], orders
            )
            if np.any(layer_roots[layer_rows] != target_root) or np.any(
                offsets != offsets[0]
            ):
                raise ValueError(
                    "Core/cap periodic cycles disagree with accepted layer construction ancestry."
                )
            offset = offsets[0]
        else:
            target_root = int(mapping[root])
            offset = np.zeros(periodic.cell.rank, dtype=np.int64)
        combined_roots[mapping[members]] = target_root
        combined_shifts[mapping[members]] = reduce_periodic_exponents(
            shifts[members] + offset, orders
        )
    return periodic_topology_owner.validate_periodic_vertex_orbits(
        points, periodic.cell, combined_roots, combined_shifts
    )


@checked
def prepare_periodic_layer_core_source(
    source: sources_owner.NativeLayerCoreSource,
    core_vertex_representatives: np.ndarray,
    core_vertex_shifts: np.ndarray,
    core_seam_polygon_pairs: np.ndarray,
    /,
) -> sources_owner.NativeLayerCoreSource:
    """Bind source-authored core orbits to a periodic layer and its source charts."""
    if not isinstance(source, sources_owner.NativeLayerCoreSource):
        raise TypeError("source must be a NativeLayerCoreSource.")
    periodic = source.layers.mesh.periodic_topology
    if periodic is None or source.periodic_topology_id != periodic.periodic_topology_id:
        raise ValueError(
            "Periodic layer/core preparation requires the exact accepted layer orbit identity."
        )
    if source.source_surface_charts is None or source.fidelity_source is None:
        raise ValueError(
            "Periodic layer/core preparation requires source-bound surface charts and their fidelity query."
        )
    if any(
        value is not None
        for value in (
            source.core_vertex_representatives,
            source.core_vertex_shifts,
            source.core_seam_polygon_pairs,
        )
    ):
        raise ValueError(
            "Periodic layer/core preparation requires an unbound source-core ancestry precursor."
        )
    charts = source.source_surface_charts
    charts.require_bound(
        source.layers.source_wall, charts.geometry, source.fidelity_source
    )
    bound = sources_owner.NativeLayerCoreSource(
        source.layers,
        source.complex,
        source.source_id,
        source.source_revision,
        vertex_layer_ids=source.vertex_layer_ids,
        cap_polygon_ids=source.cap_polygon_ids,
        layer_regions=source.layer_regions,
        region_entity_ids=source.region_entity_ids,
        region_occurrence_paths=source.region_occurrence_paths,
        region_ids=source.region_ids,
        core_region_map=source.core_region_map,
        source_boundary_scope=source.source_boundary_scope,
        source_domain=source.source_domain,
        fidelity_source=source.fidelity_source,
        source_surface_charts=charts,
        maximum_deviation=source.maximum_deviation,
        core_facet_source_ids=source.core_facet_source_ids,
        core_facet_source_occurrence_paths=source.core_facet_source_occurrence_paths,
        core_vertex_representatives=core_vertex_representatives,
        core_vertex_shifts=core_vertex_shifts,
        core_seam_polygon_pairs=core_seam_polygon_pairs,
        reference_source=source.reference_source,
        mapped_domain=source.mapped_domain,
        core_power_source=source.core_power_source,
    )
    mapping, _, polygons, points = layer_topology_owner.prepare_identity(
        bound.layers,
        bound.complex,
        bound.vertex_layer_ids,
        bound.cap_polygon_ids,
    )
    prepare_core_periodic(bound, mapping, polygons, points)
    return bound


@checked
def core_periodic_constraint_evidence(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    /,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    """Check requested source face pairs against authored seam permutations."""
    if not specification.periodic_constraints:
        return (), ()
    periodic = source.layers.mesh.periodic_topology
    pairs = source.core_seam_polygon_pairs
    roots = source.core_vertex_representatives
    shifts = source.core_vertex_shifts
    if periodic is None or pairs is None or roots is None or shifts is None:
        raise ValueError(
            "Periodic constraints require full source-authored core seam ancestry."
        )
    complex_ = source.complex
    polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    owners = controls_owner.core_facet_scope_ids(source)[
        np.asarray(complex_.polygon_facets, dtype=np.int64)
    ]
    orders = identification_orders(periodic.cell)
    fixed = periodic_topology_owner.fixed_generators(
        periodic.cell, complex_.vertices, roots
    )
    _, signs, anchors = periodic_topology_owner.oriented_forms(
        np.asarray(roots)[polygons], np.asarray(shifts)[polygons], fixed[polygons], orders
    )
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    for constraint in specification.periodic_constraints:
        source_scope = constraint.source_scope
        target_scope = constraint.target_scope
        if (
            source_scope.entity_dimension != 2
            or target_scope.entity_dimension != 2
            or source_scope.entity_kind is not target_scope.entity_kind
            or source_scope.entity_set_id != target_scope.entity_set_id
            or source_scope.source_id != source.source_id
            or source_scope.source_revision != source.source_revision
            or target_scope.source_id != source.source_id
            or target_scope.source_revision != source.source_revision
        ):
            raise ValueError("Periodic controls must bind explicit source face pairs.")
        source_ids = np.asarray(source_scope.entity_ids, dtype=np.int64)
        target_ids = np.asarray(target_scope.entity_ids, dtype=np.int64)
        source_set = set(source_ids.tolist())
        target_set = set(target_ids.tolist())
        if source_set & target_set:
            raise ValueError("Periodic source and target face scopes must be disjoint.")
        source_rows = np.flatnonzero(np.isin(owners, source_ids))
        target_rows = np.flatnonzero(np.isin(owners, target_ids))
        if (
            not source_rows.size
            or source_rows.size != target_rows.size
            or set(np.unique(owners[source_rows]).tolist()) != source_set
            or set(np.unique(owners[target_rows]).tolist()) != target_set
        ):
            raise ValueError(
                "Periodic controls must retain full opposite-oriented seam triangulations."
            )
        selected = [
            (int(a), int(b)) if int(owners[a]) in source_set else (int(b), int(a))
            for a, b in np.asarray(pairs, dtype=np.int64)
            if (int(owners[a]) in source_set and int(owners[b]) in target_set)
            or (int(owners[a]) in target_set and int(owners[b]) in source_set)
        ]
        selected_source = [a for a, _ in selected]
        selected_target = [b for _, b in selected]
        if (
            set(selected_source) != set(source_rows.tolist())
            or set(selected_target) != set(target_rows.tolist())
            or len(selected) != source_rows.size
        ):
            raise ValueError(
                "Periodic face controls are not exhaustively represented by source seam anatomy."
            )
        residual = 0.0
        covered: set[int] = set()
        for a_index, b_index in selected:
            if signs[a_index] * signs[b_index] != -1:
                raise ValueError(
                    "A periodic face control contradicts its construction permutation."
                )
            delta = reduce_periodic_exponents(anchors[b_index] - anchors[a_index], orders)
            expected = np.asarray(
                exact_periodic_element(periodic.cell, delta), dtype=np.float64
            )
            residual = max(
                residual,
                float(np.max(np.abs(expected - np.asarray(constraint.transform)))),
            )
            covered.update((a_index, b_index))
        if residual > constraint.tolerance:
            raise ValueError(
                "A periodic control transform contradicts the source-authored seam cycle."
            )
        requested.append(
            (f"periodic:{constraint.constraint_id}:tolerance", constraint.tolerance)
        )
        achieved.extend(
            (
                (f"periodic:{constraint.constraint_id}:transform_residual", residual),
                (
                    f"periodic:{constraint.constraint_id}:seam_triangles",
                    float(len(covered)),
                ),
            )
        )
    return tuple(requested), tuple(achieved)


@checked
def bind_core_periodic(
    source: sources_owner.NativeLayerCoreSource,
    mesh: CellMesh,
    ancestry: tuple[np.ndarray, np.ndarray] | None,
    /,
) -> periodic_topology_owner.PeriodicMeshTopology | None:
    """Build the quotient descriptor with authored winding and singleton core nodes."""
    if ancestry is None:
        return None
    periodic = source.layers.mesh.periodic_topology
    if periodic is None:
        raise ValueError(
            "Compiled periodic core ancestry lost the accepted identification."
        )
    roots, shifts = ancestry
    count = mesh.coordinates.shape[0]
    rank = periodic.cell.rank
    if count < roots.size:
        raise ValueError("Combined mesh lost source-authored periodic boundary vertices.")
    extra = count - roots.size
    roots = np.concatenate((roots, roots.size + np.arange(extra, dtype=np.int64)))
    shifts = np.concatenate((shifts, np.zeros((extra, rank), dtype=np.int64)), axis=0)
    return periodic_topology_owner.PeriodicMeshTopology(
        mesh, periodic.cell, roots, shifts
    )


__all__ = [
    "bind_core_periodic",
    "core_periodic_constraint_evidence",
    "prepare_core_periodic",
    "prepare_periodic_layer_core_source",
]
