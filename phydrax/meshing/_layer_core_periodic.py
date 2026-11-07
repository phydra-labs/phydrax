#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-authored seam anatomy for fixed-PLC layer/core assembly."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

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
from ._layer_core_subdivision import NestedLayerPartition


_Stencil = tuple[tuple[int, Fraction], ...]


@dataclass(frozen=True, slots=True)
class PeriodicSeamPartition:
    """Source-polygon descendants paired by the authored deck action."""

    source_polygons: tuple[tuple[int, int], tuple[int, int]]
    deck_shift: tuple[int, ...]
    first: tuple[tuple[_Stencil, _Stencil, _Stencil], ...]
    second: tuple[tuple[_Stencil, _Stencil, _Stencil], ...]


def partition_periodic_seams(
    source: sources_owner.NativeLayerCoreSource, partition: NestedLayerPartition, /
) -> tuple[PeriodicSeamPartition, ...]:
    """Refine paired source seam quads using accepted-cap edge stencils.

    Exact coplanarity is required before removing the internal source diagonal.
    """
    periodic = source.layers.mesh.periodic_topology
    pairs = source.core_seam_polygon_pairs
    roots = source.core_vertex_representatives
    shifts = source.core_vertex_shifts
    if periodic is None or pairs is None or roots is None or shifts is None:
        raise ValueError(
            "Periodic seam refinement requires authored seam and vertex orbits."
        )
    if not isinstance(partition, NestedLayerPartition):
        raise TypeError("partition must be NestedLayerPartition.")
    _refined_layer_orbits(source, partition)
    complex_ = source.complex
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    vertices = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    polygons = tuple(
        tuple(int(v) for v in vertices[a:b])
        for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    )
    if any(len(polygon) != 3 for polygon in polygons):
        raise ValueError("Periodic seam refinement requires triangular source polygons.")
    polygons_array = np.asarray(polygons, dtype=np.int64)
    roots, shifts = periodic_topology_owner.validate_periodic_vertex_orbits(
        complex_.vertices, periodic.cell, roots, shifts
    )
    _require_seams(source, polygons_array, roots, shifts)
    pairs = np.asarray(pairs, dtype=np.int64)
    cap_edges: set[tuple[int, int]] = set()
    for index in np.asarray(source.cap_polygon_ids, dtype=np.int64):
        polygon = polygons[int(index)]
        cap_edges.update(
            (min(polygon[i], polygon[(i + 1) % 3]), max(polygon[i], polygon[(i + 1) % 3]))
            for i in range(3)
        )
    layer_ids = np.asarray(source.layers.mesh.vertex_global_ids, dtype=np.int64)
    source_layers = np.asarray(source.vertex_layer_ids, dtype=np.int64)
    edge_points: dict[tuple[int, int], dict[Fraction, _Stencil]] = {}
    for edge in cap_edges:
        a, b = edge
        if source_layers[a] < 0 or source_layers[b] < 0:
            raise ValueError("A cap edge lost its accepted layer vertex ancestry.")
        id_a, id_b = int(layer_ids[source_layers[a]]), int(layer_ids[source_layers[b]])
        edge_stencils: dict[Fraction, _Stencil] = {}
        for stencil in partition.stencils:
            weights = dict(stencil)
            if set(weights).issubset((id_a, id_b)):
                t = weights.get(id_b, Fraction(0))
                if 0 < t < 1:
                    edge_stencils[t] = ((a, 1 - t), (b, t))
        edge_points[edge] = edge_stencils

    seam_rows = {int(index) for pair in pairs for index in pair}
    seam_edges = {
        row: {
            (
                min(polygons[row][i], polygons[row][(i + 1) % 3]),
                max(polygons[row][i], polygons[row][(i + 1) % 3]),
            )
            for i in range(3)
        }
        for row in seam_rows
    }
    quads: dict[int, tuple[int, tuple[int, int, int, int]]] = {}
    exact = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(complex_.vertices)
    )

    opposite = {int(a): int(b) for a, b in pairs}
    opposite.update({int(b): int(a) for a, b in pairs})
    cap_rows = {
        row: next(iter(seam_edges[row] & cap_edges))
        for row in seam_rows
        if len(seam_edges[row] & cap_edges) == 1
    }
    if any(len(seam_edges[row] & cap_edges) > 1 for row in seam_rows):
        raise ValueError("A periodic seam triangle touches multiple cap edges.")
    options: dict[int, dict[int, set[tuple[int, int]]]] = {}
    for row, edge in cap_rows.items():
        options[row] = {}
        for other in seam_rows - set(cap_rows):
            if len(seam_edges[row] & seam_edges[other]) != 1:
                continue
            boundary = seam_edges[row] ^ seam_edges[other]
            if len(
                set(polygons[row]) | set(polygons[other])
            ) == 4 and boundary & cap_edges == {edge}:
                options[row][other] = boundary
    representatives = sorted(row for row in cap_rows if row < opposite[row])
    if len(representatives) * 2 != len(cap_rows):
        raise ValueError("Periodic seam cap triangles do not have paired cap copies.")
    solutions: list[dict[int, int]] = []

    def match(index: int, used: set[int], chosen: dict[int, int]) -> None:
        if len(solutions) > 1:
            return
        if index == len(representatives):
            if len(used) == len(seam_rows):
                solutions.append(chosen.copy())
            return
        row = representatives[index]
        mate = opposite[row]
        for other in sorted(options[row]):
            other_mate = opposite[other]
            selected = {row, mate, other, other_mate}
            if len(selected) != 4 or selected & used or other_mate not in options[mate]:
                continue
            chosen[row], chosen[mate] = other, other_mate
            match(index + 1, used | selected, chosen)
            del chosen[row], chosen[mate]

    match(0, set(), {})
    if len(solutions) != 1:
        raise ValueError(
            "Periodic source seams do not determine one disjoint paired quad partition."
        )
    for row, other in sorted(solutions[0].items()):
        edge = cap_rows[row]
        boundary = options[row][other]
        a, b = edge
        a_next = [v for e in boundary if a in e and b not in e for v in e if v != a]
        b_next = [v for e in boundary if b in e and a not in e for v in e if v != b]
        if len(a_next) != 1 or len(b_next) != 1:
            raise ValueError("Periodic seam quad has invalid boundary incidence.")
        upper_a, upper_b = a_next[0], b_next[0]
        upper = (min(upper_a, upper_b), max(upper_a, upper_b))
        if upper not in boundary:
            raise ValueError("Periodic seam quad lost its opposite upper edge.")
        points = [exact[v] for v in (a, b, upper_a, upper_b)]
        u = tuple(points[1][i] - points[0][i] for i in range(3))
        v = tuple(points[2][i] - points[0][i] for i in range(3))
        w = tuple(points[3][i] - points[0][i] for i in range(3))
        determinant = (
            (u[1] * v[2] - u[2] * v[1]) * w[0]
            + (u[2] * v[0] - u[0] * v[2]) * w[1]
            + (u[0] * v[1] - u[1] * v[0]) * w[2]
        )
        if determinant:
            raise ValueError(
                "Periodic source seam triangles are not exactly coplanar; "
                "their authored diagonal cannot be removed."
            )
        oriented = polygons[row]
        if (a, b) not in tuple((oriented[i], oriented[(i + 1) % 3]) for i in range(3)):
            a, b, upper_a, upper_b = b, a, upper_b, upper_a
        quads[row] = (other, (a, b, upper_b, upper_a))

    result = []
    orders = identification_orders(periodic.cell)
    opposite = {int(a): int(b) for a, b in pairs}
    opposite.update({int(b): int(a) for a, b in pairs})
    for first_row in sorted(quads):
        other, first = quads[first_row]
        second_row = opposite[first_row]
        if second_row not in quads:
            continue
        second_other, second = quads[second_row]
        if opposite.get(other) != second_other:
            raise ValueError(
                "Periodic seam quad source triangle pairing is inconsistent."
            )
        if first_row > second_row:
            continue
        matching = {}
        for vertex in first:
            matches = [
                candidate for candidate in second if roots[candidate] == roots[vertex]
            ]
            if len(matches) != 1:
                raise ValueError("Periodic seam quad vertex pairing is ambiguous.")
            matching[vertex] = matches[0]
        delta = reduce_periodic_exponents(
            shifts[np.asarray([matching[v] for v in first], dtype=np.int64)]
            - shifts[np.asarray(first, dtype=np.int64)],
            orders,
        )
        if np.any(delta != delta[0]) or not np.any(delta[0]):
            raise ValueError(
                "Periodic seam vertices do not share one nontrivial deck action."
            )
        matrix = exact_periodic_element(periodic.cell, delta[0])
        for vertex in first:
            point = exact[vertex] + (Fraction(1),)
            image = tuple(
                sum(
                    (matrix[axis][column] * point[column] for column in range(4)),
                    Fraction(),
                )
                for axis in range(3)
            )
            if image != exact[matching[vertex]]:
                raise ValueError(
                    "Periodic seam source vertices do not satisfy the exact deck action."
                )

        def mapped(stencil: _Stencil) -> _Stencil:
            return tuple(sorted((matching[vertex], weight) for vertex, weight in stencil))

        a, b, upper_b, upper_a = first
        edge = (min(a, b), max(a, b))
        other_edge = (min(matching[a], matching[b]), max(matching[a], matching[b]))
        if {mapped(value) for value in edge_points[edge].values()} != set(
            edge_points[other_edge].values()
        ):
            raise ValueError("Paired periodic seam cap subdivisions disagree.")
        fractions = [Fraction(0)]
        fractions.extend(sorted(t if a < b else 1 - t for t in edge_points[edge]))
        fractions.append(Fraction(1))

        def interpolate(x: int, y: int, t: Fraction) -> _Stencil:
            return tuple(
                sorted(
                    (vertex, weight) for vertex, weight in ((x, 1 - t), (y, t)) if weight
                )
            )

        first_triangles = []
        second_triangles = []
        for left, right in zip(fractions[:-1], fractions[1:], strict=True):
            low_a, low_b = interpolate(a, b, left), interpolate(a, b, right)
            high_a = interpolate(upper_a, upper_b, left)
            high_b = interpolate(upper_a, upper_b, right)
            for triangle in ((low_a, low_b, high_b), (low_a, high_b, high_a)):
                first_triangles.append(triangle)
                second_triangles.append(
                    (mapped(triangle[2]), mapped(triangle[1]), mapped(triangle[0]))
                )
        result.append(
            PeriodicSeamPartition(
                ((first_row, other), (second_row, second_other)),
                tuple(int(value) for value in delta[0]),
                tuple(first_triangles),
                tuple(second_triangles),
            )
        )
    if len(result) * 2 != len(quads):
        raise ValueError("Periodic seam quads were not completely paired.")
    return tuple(result)


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
    periodic = source.layers.mesh.periodic_topology
    if periodic is None or source.periodic_topology_id != periodic.periodic_topology_id:
        raise ValueError(
            "Periodic layer/core preparation requires the exact accepted layer orbit identity."
        )
    if (
        source.source_wall_charts is None
        or source.source_boundary_charts is None
        or source.fidelity_source is None
    ):
        raise ValueError(
            "Periodic layer/core preparation requires wall and boundary source charts and their fidelity query."
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
    wall_charts = source.source_wall_charts
    boundary_charts = source.source_boundary_charts
    wall_charts.require_root(source.layers.source_wall, wall_charts.geometry)
    boundary_charts.require_bound(
        boundary_charts.mesh, boundary_charts.geometry, source.fidelity_source
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
        source_wall_charts=wall_charts,
        source_wall_projection=source.source_wall_projection,
        source_wall_association_policy=source.source_wall_association_policy,
        source_boundary_charts=boundary_charts,
        source_boundary_schedule=source.source_boundary_schedule,
        cap_refinement=source.cap_refinement,
        noncap_refinement=source.noncap_refinement,
        layer_control=source.layer_control,
        layer_policy=source.layer_policy,
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


def _refined_layer_orbits(
    source: sources_owner.NativeLayerCoreSource, partition: NestedLayerPartition, /
) -> tuple[np.ndarray, np.ndarray]:
    layer = source.layers.mesh
    periodic = layer.periodic_topology
    if periodic is None:
        raise ValueError("Refined periodic layers require authored layer orbits.")
    orders = identification_orders(periodic.cell)
    layer_ids = np.asarray(layer.vertex_global_ids, dtype=np.int64)
    source_by_id = {int(identifier): row for row, identifier in enumerate(layer_ids)}
    if (
        len(partition.stencils) != len(partition.vertex_ids)
        or len(partition.exact_coordinates) != len(partition.stencils)
        or np.asarray(partition.coordinates).shape != (len(partition.stencils), 3)
        or len(set(int(value) for value in partition.vertex_ids))
        != len(partition.vertex_ids)
    ):
        raise ValueError("Refined layer stencil and coordinate banks do not align.")
    if not np.array_equal(partition.vertex_ids[: layer_ids.size], layer_ids) or any(
        partition.stencils[row] != ((int(identifier), Fraction(1)),)
        for row, identifier in enumerate(layer_ids)
    ):
        raise ValueError(
            "Refined layers must retain every original vertex and singleton stencil."
        )
    source_points = np.asarray(layer.coordinates, dtype=np.float64)
    for row, stencil in enumerate(partition.stencils):
        identifiers = tuple(int(identifier) for identifier, _ in stencil)
        if (
            not stencil
            or identifiers != tuple(sorted(set(identifiers)))
            or any(identifier not in source_by_id for identifier in identifiers)
            or any(
                not isinstance(weight, Fraction) or weight <= 0 for _, weight in stencil
            )
            or sum((weight for _, weight in stencil), Fraction(0)) != 1
        ):
            raise ValueError(
                "Refined layer stencils must be canonical convex source rows."
            )
        exact = tuple(
            sum(
                (
                    weight
                    * Fraction(float(source_points[source_by_id[int(identifier)], axis]))
                    for identifier, weight in stencil
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        if exact != partition.exact_coordinates[row] or not np.array_equal(
            np.asarray([float(value) for value in exact]), partition.coordinates[row]
        ):
            raise ValueError(
                "Refined layer coordinates do not reproduce their exact source stencils."
            )
    layer_roots = np.asarray(periodic.vertex_representatives, dtype=np.int64)
    layer_shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
    fixed = np.asarray(periodic.vertex_fixed_generators, dtype=np.bool_)
    roots = np.arange(len(partition.stencils), dtype=np.int64)
    shifts = np.zeros((len(partition.stencils), len(orders)), dtype=np.int64)
    groups: dict[tuple[object, ...], tuple[int, np.ndarray]] = {}
    for row, stencil in enumerate(partition.stencils):
        terms = [source_by_id[int(identifier)] for identifier, _ in stencil]
        exponents = layer_shifts[terms].copy()
        fixed_rows = fixed[terms]
        exponents[fixed_rows] = 0
        movable = np.flatnonzero(~np.any(fixed_rows, axis=1))
        choices = movable if movable.size else np.arange(len(terms))
        variants = []
        for index in choices:
            anchor = exponents[index].copy()
            relative = reduce_periodic_exponents(exponents - anchor, orders)
            relative[fixed_rows] = 0
            key = tuple(
                sorted(
                    (
                        int(layer_ids[layer_roots[term]]),
                        tuple(int(x) for x in relative[i]),
                        weight,
                    )
                    for i, (term, (_, weight)) in enumerate(
                        zip(terms, stencil, strict=True)
                    )
                )
            )
            variants.append((key, tuple(int(x) for x in anchor), anchor))
        key, _, anchor = min(variants, key=lambda value: (value[0], value[1]))
        if key in groups:
            root_row, root_anchor = groups[key]
            roots[row] = root_row
            shifts[row] = reduce_periodic_exponents(anchor - root_anchor, orders)
        else:
            groups[key] = (row, anchor)
    return roots, shifts


def bind_refined_core_periodic(
    source: sources_owner.NativeLayerCoreSource,
    partition: NestedLayerPartition,
    execution_stencils: tuple[_Stencil, ...],
    input_vertices: np.ndarray,
    core_vertex_map: np.ndarray,
    mesh: CellMesh,
    exact_coordinate_source: periodic_topology_owner.ExactPeriodicCoordinateSource | None,
    seam_records: tuple[PeriodicSeamPartition, ...],
    /,
) -> periodic_topology_owner.PeriodicMeshTopology:
    """Replay authored and refined seam orbits onto the composed target mesh."""
    periodic = source.layers.mesh.periodic_topology
    if (
        periodic is None
        or source.core_vertex_representatives is None
        or source.core_vertex_shifts is None
    ):
        raise ValueError(
            "Refined periodic binding requires authored layer and core orbits."
        )
    if seam_records != partition_periodic_seams(source, partition):
        raise ValueError(
            "Refined periodic seam records differ from the current source partition."
        )
    count = int(mesh.coordinates.shape[0])
    layer_count = int(partition.vertex_ids.size)
    if count < layer_count or not np.array_equal(
        np.asarray(mesh.vertex_global_ids, dtype=np.int64)[:layer_count],
        partition.vertex_ids,
    ):
        raise ValueError(
            "Combined target does not retain refined layer vertex identities."
        )
    input_rows = np.asarray(input_vertices, dtype=np.int64)
    target_rows = np.asarray(core_vertex_map, dtype=np.int64)
    if input_rows.ndim != 1 or target_rows.shape != input_rows.shape:
        raise ValueError("Core input and combined vertex maps must align.")
    if np.any(target_rows < 0) or np.any(target_rows >= count):
        raise ValueError("Core vertex map escapes the combined target.")
    if np.any(input_rows < -1) or np.any(input_rows >= len(execution_stencils)):
        raise ValueError("Core input ancestry escapes the execution PLC.")
    if len(set(execution_stencils)) != len(execution_stencils):
        raise ValueError("Execution PLC stencils must identify unique vertices.")
    execution_targets: dict[_Stencil, int] = {}
    for input_row, target_row in zip(input_rows, target_rows, strict=True):
        if input_row < 0:
            continue
        stencil = execution_stencils[int(input_row)]
        prior = execution_targets.setdefault(stencil, int(target_row))
        if prior != int(target_row):
            raise ValueError("An execution PLC vertex maps to multiple target rows.")

    orders = identification_orders(periodic.cell)
    rank = len(orders)
    graph: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(count)]

    def link(first: int, second: int, delta: np.ndarray) -> None:
        delta = reduce_periodic_exponents(np.asarray(delta, dtype=np.int64), orders)
        graph[first].append((second, delta))
        graph[second].append((first, reduce_periodic_exponents(-delta, orders)))

    layer_ids = np.asarray(source.layers.mesh.vertex_global_ids, dtype=np.int64)
    layer_roots, layer_shifts = _refined_layer_orbits(source, partition)
    for row, root in enumerate(layer_roots):
        link(int(root), row, layer_shifts[row])

    plc_roots = np.asarray(source.core_vertex_representatives, dtype=np.int64)
    plc_shifts = np.asarray(source.core_vertex_shifts, dtype=np.int64)
    for row, root in enumerate(plc_roots):
        first = execution_targets.get(((int(root), Fraction(1)),))
        second = execution_targets.get(((row, Fraction(1)),))
        if first is None or second is None:
            raise ValueError("Execution PLC lost an authored source vertex.")
        link(first, second, plc_shifts[row])

    source_layer_rows = np.asarray(source.vertex_layer_ids, dtype=np.int64)
    plc_by_layer: dict[int, int] = {}
    for row, layer_row in enumerate(source_layer_rows):
        if layer_row >= 0:
            identifier = int(layer_ids[layer_row])
            if identifier in plc_by_layer:
                raise ValueError("A source cap vertex has ambiguous PLC ancestry.")
            plc_by_layer[identifier] = row
    for row, stencil in enumerate(partition.stencils):
        if all(int(identifier) in plc_by_layer for identifier, _ in stencil):
            plc_stencil = tuple(
                sorted(
                    (plc_by_layer[int(identifier)], weight)
                    for identifier, weight in stencil
                )
            )
            target = execution_targets.get(plc_stencil)
            if target is not None:
                link(row, target, np.zeros(rank, dtype=np.int64))

    for record in seam_records:
        for first_triangle, second_triangle in zip(
            record.first, record.second, strict=True
        ):
            for first_stencil, second_stencil in zip(
                first_triangle, reversed(second_triangle), strict=True
            ):
                first = execution_targets.get(first_stencil)
                second = execution_targets.get(second_stencil)
                if first is None or second is None:
                    raise ValueError(
                        "Protected seam partition lost its source vertex ancestry."
                    )
                link(first, second, np.asarray(record.deck_shift, dtype=np.int64))

    roots = np.full(count, -1, dtype=np.int64)
    shifts = np.zeros((count, rank), dtype=np.int64)
    for root in range(count):
        if roots[root] >= 0:
            continue
        roots[root] = root
        stack = [root]
        while stack:
            row = stack.pop()
            for target, delta in graph[row]:
                winding = reduce_periodic_exponents(shifts[row] + delta, orders)
                if roots[target] < 0:
                    roots[target] = root
                    shifts[target] = winding
                    stack.append(target)
                elif roots[target] != root:
                    raise ValueError(
                        "Refined periodic ancestry joins incompatible orbit roots."
                    )
                elif np.any(shifts[target] != winding):
                    cycle = reduce_periodic_exponents(winding - shifts[target], orders)
                    matrix = exact_periodic_element(periodic.cell, cycle)
                    point = (
                        tuple(
                            Fraction(float(value))
                            for value in np.asarray(mesh.coordinates[target])
                        )
                        if exact_coordinate_source is None
                        else tuple(
                            exact_coordinate_source.source_coordinates(
                                np.asarray([target])
                            )[0]
                        )
                    ) + (Fraction(1),)
                    image = tuple(
                        sum(
                            (matrix[axis][column] * point[column] for column in range(4)),
                            Fraction(),
                        )
                        for axis in range(3)
                    )
                    if image != point[:3]:
                        raise ValueError(
                            "Refined periodic ancestry has an inconsistent deck cycle."
                        )
    return periodic_topology_owner.PeriodicMeshTopology(
        mesh,
        periodic.cell,
        roots,
        shifts,
        exact_coordinate_source=exact_coordinate_source,
    )


__all__ = [
    "bind_core_periodic",
    "bind_refined_core_periodic",
    "core_periodic_constraint_evidence",
    "partition_periodic_seams",
    "prepare_core_periodic",
    "prepare_periodic_layer_core_source",
]
