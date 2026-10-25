#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certify power-cell facets and interior disjointness.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "exact_power_source_witnesses"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The current power source prepares rational coordinates and verifies their defining site inequalities before cell comparisons."
      }
    ],
    "conclusion": "The prepared vertices belong to the current declared power-source construction.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.discretization._exact_power_geometry.ExactPowerCellGeometrySource.prepare",
      "locator": "phydrax.discretization._exact_power_geometry.ExactPowerCellGeometrySource.prepare",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "exact_polyhedral_facets"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Facet loop triangulation checks distinct vertices, plane rank, planarity, simple edges and nonzero signed area in rational arithmetic."
      }
    ],
    "conclusion": "Each admitted facet has a simple planar triangulation; invalid loops produce a violated finding.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._exact_polyhedral_geometry.triangulate_loop",
      "locator": "phydrax.geometry._exact_polyhedral_geometry.triangulate_loop",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "exact_polyhedral_facet"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Facet loop triangulation checks distinct vertices, plane rank, planarity, simple edges and nonzero signed area in rational arithmetic."
      }
    ],
    "conclusion": "Each admitted facet has a simple planar triangulation; invalid loops produce a violated finding.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._exact_polyhedral_geometry.triangulate_loop",
      "locator": "phydrax.geometry._exact_polyhedral_geometry.triangulate_loop",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "exact_polyhedral_contacts"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "All bounding-box candidate triangle pairs outside a common cell or facet group are checked for contact outside their shared combinatorial simplex."
      }
    ],
    "conclusion": "A detected contact outside shared incidence violates the facet-contact obligation.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._exact_polyhedral_geometry.triangle_contact",
      "locator": "phydrax.geometry._exact_polyhedral_geometry.triangle_contact",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "exact_polyhedral_contact"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "All bounding-box candidate triangle pairs outside a common cell or facet group are checked for contact outside their shared combinatorial simplex."
      }
    ],
    "conclusion": "A detected contact outside shared incidence violates the facet-contact obligation.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._exact_polyhedral_geometry.triangle_contact",
      "locator": "phydrax.geometry._exact_polyhedral_geometry.triangle_contact",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "exact_power_cell_nonoverlap"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Positive star tetrahedra represent every power cell; distinct-site inequalities separate interiors, and every remaining tetrahedron intersection is integrated in rational arithmetic."
      },
      {
        "tag": "X",
        "statement": "Tetrahedron face planes are oriented by rational opposite-vertex signs, and their intersection vertices satisfy every halfspace."
      }
    ],
    "conclusion": "Positive intersection volume violates cell interior disjointness; zero intersections for every candidate prove nonoverlap.",
    "citation": {
      "kind": "mathematical",
      "reference": "Boyd and Vandenberghe, Convex Optimization (2004)",
      "locator": "Section 2.2.4, polyhedra as intersections of halfspaces",
      "url": "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf"
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "exact_power_cell_overlap"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Positive star tetrahedra represent every power cell; distinct-site inequalities separate interiors, and every remaining tetrahedron intersection is integrated in rational arithmetic."
      },
      {
        "tag": "X",
        "statement": "Tetrahedron face planes are oriented by rational opposite-vertex signs, and their intersection vertices satisfy every halfspace."
      }
    ],
    "conclusion": "Positive intersection volume violates cell interior disjointness; zero intersections for every candidate prove nonoverlap.",
    "citation": {
      "kind": "mathematical",
      "reference": "Boyd and Vandenberghe, Convex Optimization (2004)",
      "locator": "Section 2.2.4, polyhedra as intersections of halfspaces",
      "url": "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf"
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "exact_polyhedral_facet_work_capacity",
      "exact_polyhedral_contact_capacity",
      "exact_power_nonoverlap_work_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Triangulation, candidate enumeration and intersection work remain within their declared capacities."
      }
    ],
    "conclusion": "Exhausted capacity produces unresolved evidence without a geometric disjointness conclusion.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._exact_power_certificates.certify_exact_power_embedding",
      "locator": "phydrax.geometry._exact_power_certificates.certify_exact_power_embedding",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  }
]
End mesh certificate checks
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from heapq import merge
from typing import assert_never

import numpy as np

import phydrax.discretization._exact_power_geometry as exact_power_geometry_owner
import phydrax.geometry._polygon_triangulation as polygon_triangulation_owner
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._embedding_state import (
    boxes,
    candidate_pairs,
    EmbeddingState,
    facet_entity_ids,
    Facets,
    MeshCertificateLimits,
)

from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..typing import checked
from ._exact_polyhedral_geometry import (
    convex_cell_overlap,
    convex_cells_disjoint,
    exact_convex_cells,
    exact_vertices,
    ExactConvexCell,
    ExactOverlapRefusal,
    ExactTetrahedronOverlap,
    star_tetrahedra,
    tetrahedra_disjoint,
    Tetrahedron,
    tetrahedron_overlap,
    TetrahedronPlanePreparation,
    triangle_contact,
    TriangleContactPreparation,
)


@checked
def certify_exact_power_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    facets: Facets,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Certify facet contacts and interior disjointness of a power source."""
    points = exact_vertices(mesh, geometry, budget=budget)
    if not _certify_power_facets(state, mesh, points, facets, limits, budget=budget):
        return
    cells = star_tetrahedra(mesh, points, budget=budget)
    source = geometry.exact_source
    match source:
        case (
            exact_power_geometry_owner.ExactPowerCellGeometrySource()
            | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
            | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
        ):
            cell_sites = _power_cell_sites(mesh, source, len(cells))
            work = source.prepare(geometry.coordinates, budget=budget).operation_count
            multiplicity = Counter(cell_sites)
            unseparated = (
                None
                if None in multiplicity
                else frozenset(
                    cell
                    for cell, site in enumerate(cell_sites)
                    if multiplicity[site] > 1
                )
            )
            # Positive stars about each vertex mean make supporting faces a hull proof.
            convex = exact_convex_cells(
                mesh, points, budget=budget, selected=unseparated
            )
            _certify_power_nonoverlap(
                state,
                cells,
                convex,
                cell_sites,
                cell_ids,
                work,
                source.maximum_work,
                budget=budget,
            )
        case _:
            raise ValueError(
                "exact_power_embedding.geometry.exact_source must be a power source; got another source"
            )


def _certify_power_facets(
    state: EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: Facets,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Triangulate each facet loop once and test each distinct triangle pair once.

    Both occurrences of an interior facet bound the same planar region; when
    their vertex loops are one cycle up to rotation and reversal, a single
    loop proof and triangulation serve both. Occurrence pairs are tested
    unless they share a cell or a facet group, so a triangle pair is tested
    exactly when some occurrence pair of its two classes has distinct cells.
    """
    classes = _facet_loop_classes(facets, budget=budget)
    rows = facets.rows[classes.representatives]
    triangles, owners, failed, unknown = polygon_triangulation_owner.loop_triangles(
        points, rows, limits.maximum_candidate_pairs, budget=budget
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
            facet_entity_ids(mesh, facets.rows[failed[classes.occurrence_class]]),
        )
    if np.any(unknown):
        state.add(
            "exact_polyhedral_facet_work_capacity",
            "unresolved",
            "facet",
            facet_entity_ids(mesh, facets.rows[unknown[classes.occurrence_class]]),
        )
    if np.any(failed | unknown):
        return False
    first, second, exceeded = candidate_pairs(
        *boxes(points, triangles), limits.maximum_candidate_pairs
    )
    state.candidate_pairs += first.size
    groups, cells = facets.group[classes.representatives], classes.cells
    preparation = budget.exact_triangle_contact_preparation
    if preparation is None:
        preparation = TriangleContactPreparation(budget)
        budget.exact_triangle_contact_preparation = preparation
    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        left, right = owners[a], owners[b]
        if groups[left] == groups[right] or (
            len(cells[left]) == 1 and cells[left] == cells[right]
        ):
            continue
        # The contact verdict is retained; its rational temporaries are not.
        with budget.temporary_scope():
            contact = triangle_contact(
                points,
                tuple(triangles[a].tolist()),
                tuple(triangles[b].tolist()),
                budget=budget,
                preparation=preparation,
            )
        if contact:
            state.add(
                "exact_polyhedral_contact",
                "violated",
                "facet",
                facet_entity_ids(mesh, rows[[left, right]]),
            )
    if exceeded:
        state.add("exact_polyhedral_contact_capacity", "unresolved", "mesh")
        return False
    return True


@dataclass(frozen=True, slots=True)
class _FacetLoopClasses:
    representatives: np.ndarray
    occurrence_class: np.ndarray
    cells: tuple[frozenset[int], ...]


def _facet_loop_classes(
    facets: Facets, /, *, budget: CoordinateEnclosureBudget
) -> _FacetLoopClasses:
    """Partition facet occurrences into equal vertex cycles within each group.

    Integer rows only: a loop matches a class when it is the class cycle up to
    rotation and reversal, which preserves the planar region and every loop
    proof.
    """
    representatives: list[int] = []
    loops: list[tuple[int, ...]] = []
    cells: list[set[int]] = []
    occurrence_class = np.empty(facets.rows.shape[0], dtype=np.int64)
    for members in facets.groups:
        start = len(representatives)
        for occurrence in members.tolist():
            row = facets.rows[occurrence]
            loop = tuple(row[row >= 0].tolist())
            match = None
            for index in range(start, len(representatives)):
                if _same_cycle(loops[index], loop):
                    match = index
                    break
            if match is None:
                match = len(representatives)
                representatives.append(occurrence)
                loops.append(loop)
                cells.append(set())
            occurrence_class[occurrence] = match
            cells[match].add(int(facets.cells[occurrence]))
    return _FacetLoopClasses(
        np.asarray(representatives, dtype=np.int64),
        occurrence_class,
        tuple(frozenset(value) for value in cells),
    )


def _same_cycle(first: tuple[int, ...], second: tuple[int, ...], /) -> bool:
    if len(first) != len(second) or not second or second[0] not in first:
        return False
    start = first.index(second[0])
    rotated = first[start:] + first[:start]
    return rotated == second or (rotated[0], *rotated[:0:-1]) == second


def _power_cell_sites(
    mesh: CellMesh,
    source: (
        exact_power_geometry_owner.ExactPowerCellGeometrySource
        | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource
        | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource
    ),
    count: int,
    /,
) -> list[int | None]:
    cell_sites: list[int | None] = [None] * count
    match source:
        case exact_power_geometry_owner.ExactPowerCellGeometrySource():
            pass
        case (
            exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
            | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
        ):
            return cell_sites
        case invalid:
            assert_never(invalid)
    preparation = source.periodic_preparation
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
    return cell_sites


def _certify_power_nonoverlap(
    state: EmbeddingState,
    cells: tuple[tuple[Tetrahedron, ...], ...],
    convex: tuple[ExactConvexCell | None, ...],
    cell_sites: list[int | None],
    cell_ids: np.ndarray,
    work: int,
    maximum_work: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove pairwise interior disjointness of cells not separated by site powers.

    Verified convex cells use one exact convex intersection; other pairs test
    every star tetrahedron pair with request-shared face planes.
    """
    preparation = TetrahedronPlanePreparation(budget)
    # Vertex power inequalities separate interiors belonging to distinct
    # sites, so only same-site pairs and pairs with an unknown site remain,
    # visited in the same ascending order as the full pair table.
    by_site: dict[int, list[int]] = {}
    unknown: list[int] = []
    for cell, site in enumerate(cell_sites):
        (unknown if site is None else by_site.setdefault(site, [])).append(cell)
    for first_cell in range(len(cells)):
        site = cell_sites[first_cell]
        partners = (
            range(first_cell + 1, len(cells))
            if site is None
            else merge(
                by_site[site][bisect_right(by_site[site], first_cell) :],
                unknown[bisect_right(unknown, first_cell) :],
            )
        )
        for second_cell in partners:
            first_convex, second_convex = convex[first_cell], convex[second_cell]
            hulls = first_convex is not None and second_convex is not None
            pairs = (
                ((first_convex, second_convex),)
                if hulls
                else (
                    (first_tet, second_tet)
                    for first_tet in cells[first_cell]
                    for second_tet in cells[second_cell]
                )
            )
            for first_piece, second_piece in pairs:
                if isinstance(first_piece, ExactConvexCell) and isinstance(second_piece, ExactConvexCell):
                    if convex_cells_disjoint(first_piece, second_piece):
                        continue
                    overlap = convex_cell_overlap(
                        first_piece,
                        second_piece,
                        maximum_work=maximum_work - work,
                        budget=budget,
                    )
                elif isinstance(first_piece, tuple) and isinstance(second_piece, tuple):
                    if tetrahedra_disjoint(first_piece, second_piece):
                        continue
                    overlap = tetrahedron_overlap(
                        first_piece,
                        second_piece,
                        maximum_work=maximum_work - work,
                        budget=budget,
                        preparation=preparation,
                    )
                else:
                    raise RuntimeError("Power certificate pieces must use the same exact cell representation.")
                match overlap:
                    case ExactOverlapRefusal():
                        state.add(
                            "exact_power_nonoverlap_work_capacity",
                            "unresolved",
                            "mesh",
                        )
                        return
                    case ExactTetrahedronOverlap():
                        pass
                    case invalid:
                        assert_never(invalid)
                work += overlap.operation_count
                if work > maximum_work:
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
