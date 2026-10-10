#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ._exact_polyhedral_geometry import (
    exact_vertices,
    star_tetrahedra,
    tetrahedron_overlap,
    triangle_contact,
)
from ._mesh_certificates import (
    _boxes,
    _candidate_pairs,
    _EmbeddingState,
    _facet_entity_ids,
    _Facets,
    _loop_triangles,
    MeshCertificateLimits,
)


def certify_exact_power_embedding(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    facets: _Facets,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    /,
) -> None:
    """Prove the ideal polyhedral source, not floating carrier planarity."""
    points = exact_vertices(mesh, geometry)
    triangles, owners, failed, unknown = _loop_triangles(
        points, facets.rows, limits.maximum_candidate_pairs
    )
    state.checks.extend(
        (
            "exact_power_source_witnesses",
            "exact_polyhedral_facets",
            "exact_polyhedral_contacts",
            "exact_power_cell_nonoverlap",
        )
    )
    if np.any(failed):
        state.add(
            "exact_polyhedral_facet",
            "violated",
            "facet",
            _facet_entity_ids(mesh, facets.rows[failed]),
        )
    if np.any(unknown):
        state.add(
            "exact_polyhedral_facet_work_capacity",
            "unresolved",
            "facet",
            _facet_entity_ids(mesh, facets.rows[unknown]),
        )
    if np.any(failed | unknown):
        return
    first, second, exceeded = _candidate_pairs(
        *_boxes(points, triangles), limits.maximum_candidate_pairs
    )
    state.candidate_pairs += first.size
    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        if (
            facets.cells[owners[a]] == facets.cells[owners[b]]
            or facets.group[owners[a]] == facets.group[owners[b]]
        ):
            continue
        if triangle_contact(
            points, tuple(triangles[a].tolist()), tuple(triangles[b].tolist())
        ):
            state.add(
                "exact_polyhedral_contact",
                "violated",
                "facet",
                _facet_entity_ids(mesh, facets.rows[[owners[a], owners[b]]]),
            )
    if exceeded:
        state.add("exact_polyhedral_contact_capacity", "unresolved", "mesh")
        return
    cells = star_tetrahedra(mesh, points)
    from ..discretization._exact_power_geometry import (
        ExactPowerCellGeometryLinearActionSource,
        ExactPowerCellGeometryRestrictionSource,
        ExactPowerCellGeometrySource,
    )

    source = geometry.exact_source
    if not isinstance(
        source,
        (
            ExactPowerCellGeometrySource,
            ExactPowerCellGeometryRestrictionSource,
            ExactPowerCellGeometryLinearActionSource,
        ),
    ):
        raise ValueError(
            "Exact power embedding requires its current power source construction."
        )

    cell_sites: list[int | None] = [None] * len(cells)
    if isinstance(source, ExactPowerCellGeometrySource):
        preparation = source._periodic_owner()
        offsets = np.asarray(source.vertex_site_offsets)
        sites = np.asarray(source.vertex_sites)
        cell = 0
        site_count = (
            source.site_points.shape[0]
            if preparation is None
            else len(preparation.image_sites)
        )
        for block in mesh.blocks:
            for row, valid in zip(
                np.asarray(block.vertices),
                np.asarray(block.vertex_valid),
                strict=True,
            ):
                common = set(range(site_count))
                for vertex in row[valid]:
                    common.intersection_update(
                        sites[offsets[vertex] : offsets[vertex + 1]].tolist()
                    )
                cell_sites[cell] = min(common) if common else None
                cell += 1
    work = source.prepare(geometry.coordinates).operation_count
    for first_cell in range(len(cells)):
        for second_cell in range(first_cell + 1, len(cells)):
            # Distinct site's interiors are disjoint by the independently proven
            # original power inequalities at every vertex of every convex piece.
            if (
                cell_sites[first_cell] is not None
                and cell_sites[second_cell] is not None
                and cell_sites[first_cell] != cell_sites[second_cell]
            ):
                continue
            for first_tet in cells[first_cell]:
                for second_tet in cells[second_cell]:
                    if any(
                        max(
                            min(point[axis] for point in first_tet),
                            min(point[axis] for point in second_tet),
                        )
                        >= min(
                            max(point[axis] for point in first_tet),
                            max(point[axis] for point in second_tet),
                        )
                        for axis in range(3)
                    ):
                        continue
                    from ..discretization._coordinate_enclosure import (
                        CoordinateEnclosureResourceError,
                    )

                    try:
                        overlap = tetrahedron_overlap(
                            first_tet, second_tet, maximum_work=source.maximum_work - work
                        )
                    except CoordinateEnclosureResourceError:
                        state.add(
                            "exact_power_nonoverlap_work_capacity", "unresolved", "mesh"
                        )
                        return
                    work += overlap.operation_count
                    if work > source.maximum_work:
                        state.add(
                            "exact_power_nonoverlap_work_capacity", "unresolved", "mesh"
                        )
                        return
                    if overlap.volume > Fraction(0):
                        state.add(
                            "exact_power_cell_overlap",
                            "violated",
                            "cell",
                            cell_ids[[first_cell, second_cell]],
                        )
                        return
    state.source_expression_work_units += work
