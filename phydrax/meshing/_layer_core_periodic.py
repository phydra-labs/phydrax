#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-authored seam anatomy for immutable fixed-PLC layer/core assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..discretization import CellMesh, PeriodicCell
from ..discretization._periodic_topology import (
    _fixed_generators,
    _identification_orders,
    _oriented_forms,
    _reduced,
    PeriodicMeshTopology,
    validate_periodic_vertex_orbits,
)
from ._layer_core_resources import LayerCoreSourceWork


if TYPE_CHECKING:
    from ._contracts import VolumeMeshingSpec
    from .providers._native_sources import NativeLayerCoreSource


def _require_seams(
    source: NativeLayerCoreSource,
    polygons: np.ndarray,
    roots: np.ndarray,
    shifts: np.ndarray,
    work: LayerCoreSourceWork,
    /,
) -> None:
    periodic = source.layers.mesh.periodic_topology
    pairs = source.core_seam_polygon_pairs
    if periodic is None or pairs is None:
        raise ValueError(
            "Periodic core construction requires retained layer identification and seam anatomy."
        )
    work.charge(pairs.size)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or np.any(pairs < 0)
        or np.any(pairs >= polygons.shape[0])
    ):
        raise ValueError(
            "Core seam polygon pairs must index two exact PLC triangles per seam."
        )
    if (
        np.unique(pairs).size != pairs.size
        or np.intersect1d(pairs, source.cap_polygon_ids).size
    ):
        raise ValueError(
            "Seam triangles must be distinct remaining PLC polygons, not cap triangles."
        )
    orders = _identification_orders(periodic.cell)
    work.charge(source.complex.vertices.shape[0] * periodic.cell.rank)
    fixed = _fixed_generators(periodic.cell, source.complex.vertices, roots)
    work.charge(polygons.size)
    keys, signs, anchors = _oriented_forms(
        roots[polygons], shifts[polygons], fixed[polygons], orders
    )
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, key in enumerate(keys):
        work.charge(1)
        groups.setdefault(tuple(key.tolist()), []).append(index)
    repeated = {tuple(sorted(group)) for group in groups.values() if len(group) > 1}
    declared = {tuple(sorted(pair.tolist())) for pair in pairs}
    if repeated != declared:
        raise ValueError(
            "Explicit seam anatomy must cover every paired polygon orbit with equivariant boundary triangulation."
        )
    complex_ = source.complex
    for first, second in pairs:
        work.charge(1)
        delta = _reduced(anchors[second] - anchors[first], orders)
        if signs[first] * signs[second] != -1 or not np.any(delta):
            raise ValueError(
                "A seam pair must retain a unique opposite oriented facet permutation and nontrivial group image."
            )
        work.charge(1)
        incidence = complex_.facet_regions[complex_.polygon_facets[[first, second]]]
        if np.any(np.count_nonzero(incidence >= 0, axis=1) != 1):
            raise ValueError("Seam strata must be external core facets.")
        # Opposite exterior normals retain the same occupied PLC side on both copies.
        work.charge(1)
        if not np.array_equal(incidence[0], incidence[1]):
            raise ValueError(
                "Periodic seam facets must preserve authoritative material incidence and orientation."
            )


def prepare_core_periodic(
    source: NativeLayerCoreSource,
    mapping: np.ndarray,
    polygons: np.ndarray,
    points: np.ndarray,
    /,
    *,
    work: LayerCoreSourceWork,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compile all boundary orbits before native fill; never search generated nodes."""
    periodic = source.layers.mesh.periodic_topology
    supplied = source.core_vertex_representatives
    exponents = source.core_vertex_shifts
    if periodic is None:
        if supplied is not None:
            raise ValueError(
                "Core periodic ancestry requires a periodic accepted layer realization."
            )
        return None
    if supplied is None or exponents is None:
        raise ValueError(
            "Periodic layers require source-authored core vertex orbits and explicit seam anatomy before fill."
        )
    work.charge(source.complex.vertices.shape[0])
    roots, shifts = validate_periodic_vertex_orbits(
        source.complex.vertices,
        periodic.cell,
        supplied,
        exponents,
    )
    _require_seams(source, polygons, roots, shifts, work)
    count = source.layers.mesh.coordinates.shape[0]
    layer_roots = np.asarray(periodic.vertex_representatives, dtype=np.int64)
    layer_shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
    combined_roots = np.arange(points.shape[0], dtype=np.int64)
    combined_shifts = np.zeros((points.shape[0], periodic.cell.rank), dtype=np.int64)
    combined_roots[:count] = layer_roots
    combined_shifts[:count] = layer_shifts
    orders = _identification_orders(periodic.cell)
    member_order = np.argsort(roots, kind="stable")
    boundaries = np.concatenate(
        (
            np.asarray([0]),
            np.flatnonzero(np.diff(roots[member_order])) + 1,
            np.asarray([roots.size]),
        )
    )
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        members = member_order[start:end]
        work.charge(members.size)
        root = int(roots[members[0]])
        shared = members[source.vertex_layer_ids[members] >= 0]
        if shared.size:
            work.charge(shared.size)
            layer_rows = source.vertex_layer_ids[shared]
            target_root = int(layer_roots[layer_rows[0]])
            offsets = _reduced(layer_shifts[layer_rows] - shifts[shared], orders)
            if np.any(layer_roots[layer_rows] != target_root) or np.any(
                offsets != offsets[0]
            ):
                raise ValueError(
                    "Core/cap periodic cycles disagree with exact retained layer construction ancestry."
                )
            offset = offsets[0]
        else:
            target_root = int(mapping[root])
            offset = np.zeros(periodic.cell.rank, dtype=np.int64)
        combined_roots[mapping[members]] = target_root
        combined_shifts[mapping[members]] = _reduced(shifts[members] + offset, orders)
    work.charge(points.shape[0])
    return validate_periodic_vertex_orbits(
        points, periodic.cell, combined_roots, combined_shifts
    )


def core_periodic_constraint_evidence(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    /,
    *,
    work: LayerCoreSourceWork,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    """Bind requested source face pairs to the pre-authored seam permutations."""
    if not specification.periodic_constraints:
        return (), ()
    periodic = source.layers.mesh.periodic_topology
    pairs = source.core_seam_polygon_pairs
    roots, shifts = source.core_vertex_representatives, source.core_vertex_shifts
    if periodic is None or pairs is None or roots is None or shifts is None:
        raise ValueError(
            "Periodic controls require complete source-authored core seam ancestry."
        )
    complex_ = source.complex
    polygons = complex_.polygon_vertices.reshape(-1, 3)
    owners = complex_.polygon_facets
    if source.core_facet_source_ids is not None:
        owners = source.core_facet_source_ids[owners]
    orders = _identification_orders(periodic.cell)
    work.charge(complex_.vertices.shape[0] * periodic.cell.rank)
    fixed = _fixed_generators(periodic.cell, complex_.vertices, roots)
    work.charge(polygons.size)
    _, signs, anchors = _oriented_forms(
        roots[polygons], shifts[polygons], fixed[polygons], orders
    )
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    for constraint in specification.periodic_constraints:
        if (
            constraint.source_scope.entity_dimension != 2
            or constraint.source_entity_ids is None
        ):
            raise ValueError(
                "Core periodic controls must bind explicit original face pairs."
            )
        if (
            constraint.source_scope.source_id,
            constraint.source_scope.source_revision,
        ) != (source.source_id, source.source_revision):
            raise ValueError(
                "Core periodic controls must bind the authoritative source revision."
            )
        residual = 0.0
        covered: set[int] = set()
        for first, second, sign in zip(
            np.asarray(constraint.source_entity_ids),
            np.asarray(constraint.target_scope.entity_ids),
            np.asarray(constraint.orientations),
            strict=True,
        ):
            work.charge(2 * owners.size + pairs.shape[0])
            a_indices, b_indices = (
                np.flatnonzero(owners == first),
                np.flatnonzero(owners == second),
            )
            if not a_indices.size or a_indices.size != b_indices.size or sign != -1:
                raise ValueError(
                    "Periodic face controls must retain complete opposite oriented seam triangulations."
                )
            selected = [
                (int(a), int(b)) if owners[a] == first else (int(b), int(a))
                for a, b in pairs
                if {int(owners[a]), int(owners[b])} == {int(first), int(second)}
            ]
            if {a for a, _ in selected} != set(a_indices.tolist()) or {
                b for _, b in selected
            } != set(b_indices.tolist()):
                raise ValueError(
                    "Requested periodic faces are not exhaustively paired by source-authored seam anatomy."
                )
            for a_index, b_index in selected:
                work.charge(1)
                if signs[a_index] * signs[b_index] != sign:
                    raise ValueError(
                        "Requested periodic face contradicts its construction permutation."
                    )
                delta = _reduced(anchors[b_index] - anchors[a_index], orders)
                if isinstance(periodic.cell, PeriodicCell):
                    expected = np.eye(4, dtype=np.float64)
                    expected[:3, 3] = np.asarray(periodic.cell.vectors) @ delta
                else:
                    expected = periodic.cell.element(delta)
                residual = max(
                    residual,
                    float(np.max(np.abs(expected - np.asarray(constraint.transform)))),
                )
                covered.update((a_index, b_index))
        if residual > constraint.tolerance:
            raise ValueError(
                "Periodic control transform contradicts the source-authored seam cycle."
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


def bind_core_periodic(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    ancestry: tuple[np.ndarray, np.ndarray] | None,
    /,
) -> CellMesh:
    if ancestry is None:
        return mesh
    periodic = source.layers.mesh.periodic_topology
    if periodic is None:
        raise ValueError(
            "Compiled periodic core ancestry lost its accepted identification."
        )
    roots, shifts = ancestry
    # Fixed PLC recovery forbids boundary Steiner vertices. Generated points
    # therefore belong to ordinary interior orbits with their own representatives.
    extra = mesh.coordinates.shape[0] - roots.size
    roots = np.concatenate((roots, roots.size + np.arange(extra, dtype=np.int64)))
    shifts = np.concatenate(
        (shifts, np.zeros((extra, periodic.cell.rank), dtype=np.int64))
    )
    topology = PeriodicMeshTopology(mesh, periodic.cell, roots, shifts)
    return CellMesh(
        mesh.coordinates,
        mesh.blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        numeric_version=mesh.numeric_version,
        periodic_topology=topology,
    )


__all__ = [
    "prepare_core_periodic",
    "bind_core_periodic",
    "core_periodic_constraint_evidence",
]
