# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact degree-one admission for closed affine translational triangle quotients.

A coherent positive PL map of a closed oriented surface is a branched covering.
The signed pullback area divided by lattice covolume is its degree. Degree one
therefore excludes overlapping sheets and vertex branching. Edge pairing alone
is not a manifold proof: each quotient vertex link is separately checked below.

Mesh certificate checks
-----------------------
[
  {
    "owner": "phydrax.geometry._periodic_affine_degree.certify_affine_torus_degree",
    "checks": ["periodic_affine_torus_manifold", "periodic_affine_torus_trace", "periodic_affine_torus_orientation", "periodic_affine_torus_degree_one"],
    "premises": [
      {"tag": "X", "statement": "The current quotient is a closed connected coherently oriented triangle two-manifold; every vertex link is one cycle and its reconstructed incidence and winding cocycle match the declaration."},
      {"tag": "X", "statement": "The current full coordinate bases and their reference actions are affine, and all lifted vertex images agree exactly under the same rank-two translation lattice."},
      {"tag": "X", "statement": "Every authenticated triangle has strictly positive signed determinant, and the signed pullback area with multiplicity is exactly one lattice covolume."}
    ],
    "conclusion": "Every local covering degree is positive and their global sum is one, so the quotient map is a homeomorphism onto the flat torus; no image-pair enumeration is needed.",
    "citation": {"kind": "mathematical", "reference": "Outerelo and Ruiz, Mapping Degree Theory: degree as the sum of local degrees for an orientation-preserving PL branched covering", "locator": "closed oriented two-manifold degree theorem", "url": "https://doi.org/10.1090/gsm/108"},
    "failure_statuses": ["violated", "unresolved"]
  }
]
End mesh certificate checks
"""

from __future__ import annotations

from fractions import Fraction
from typing import TypeAlias

import numpy as np

from phydrax._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._coordinate_budget import (
    coordinate_retention_preparation_bounds,
    CoordinateEnclosureBudget,
)
from phydrax.discretization._coordinate_enclosure import (
    affine_simplex_corner_images,
    affine_simplex_source,
    prepared_coordinate_source_bank,
)
from phydrax.discretization._exact_algebraic_geometry import (
    RadicalElement,
    rational_coefficient,
    reserve_algebraic_work,
)
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.geometry._embedding_state import (
    EmbeddingState,
    PeriodicAffineTorusDegreeEvidence,
)


Scalar: TypeAlias = Fraction | RadicalElement
Point: TypeAlias = tuple[Scalar, Scalar]
Germ: TypeAlias = tuple[int, int, int]
Edge: TypeAlias = tuple[int, int, int, int]


def _scalar_record(value: Scalar) -> tuple[tuple[int, int, int], ...]:
    terms = value.terms if isinstance(value, RadicalElement) else ((1, value),)
    return tuple((root, coefficient.numerator, coefficient.denominator) for root, coefficient in terms if coefficient)


def _edge(first: int, second: int, shifts: np.ndarray, roots: np.ndarray) -> tuple[Edge, int]:
    a, b = int(roots[first]), int(roots[second])
    sx, sy = (int(value) for value in shifts[second] - shifts[first])
    forward, backward = (a, b, sx, sy), (b, a, -sx, -sy)
    return (forward, 1) if forward < backward else (backward, -1)


def _one_cycle(links: dict[Germ, Germ]) -> bool:
    if not links or set(links) != set(links.values()) or len(set(links.values())) != len(links):
        return False
    start = next(iter(links))
    current, seen = start, set()
    while current not in seen:
        seen.add(current)
        current = links[current]
    return current == start and len(seen) == len(links)


def certify_affine_torus_degree(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> PeriodicAffineTorusDegreeEvidence | bool:
    """Return False only outside admission, True for refusal, or earned evidence.

    A claimed premise failure never falls through to another accepting route.
    Rebuilding the quotient checks its current incidence and winding cocycle;
    the independent directed corner-link cycles establish the manifold premise.
    """
    topology = mesh.periodic_topology
    if (
        mesh.topological_dimension != 2
        or mesh.ambient_dimension != 2
        or mesh.storage is not None
        or topology is None
        or not isinstance(topology.cell, PeriodicCell)
        or topology.cell.rank != 2
        or not topology.cell.fully_periodic
        or any(block.cell_kind != "triangle" for block in mesh.blocks)
    ):
        return False
    if not all(affine_simplex_source(element, budget=budget) for element in geometry.elements):
        return False
    state.checks.extend((
        "periodic_affine_torus_manifold",
        "periodic_affine_torus_trace",
        "periodic_affine_torus_orientation",
        "periodic_affine_torus_degree_one",
    ))
    with budget.temporary_scope(), budget.polynomial_preparation_scope(), budget.live_storage() as area_storage:
        count = sum(block.cell_count for block in mesh.blocks)
        vertices = len(mesh.coordinates)
        budget.reserve(64 * (vertices + count), 4096 + 1024 * vertices + 3072 * count)
        actual_ids = np.concatenate(tuple(np.asarray(block.global_ids) for block in mesh.blocks))
        if actual_ids.size != np.unique(actual_ids).size or not np.array_equal(np.sort(actual_ids), np.sort(cell_ids)):
            state.add("periodic_affine_torus_manifold", "violated", "mesh")
            return True
        topology.cell.validate_restored()
        lifted = CellMesh.lifted_from_stored_fields(mesh)
        renewed = topology.rebuilt(lifted, budget=budget)
        if not tree_content_equal(renewed, topology):
            state.add("periodic_affine_torus_manifold", "violated", "mesh")
            return True
        lattice = tuple(tuple(Fraction(float(value)) for value in row) for row in np.asarray(topology.cell.vectors))
        determinant = lattice[0][0] * lattice[1][1] - lattice[0][1] * lattice[1][0]
        if not determinant:
            state.add("periodic_affine_torus_trace", "violated", "mesh")
            return True
        roots = np.asarray(topology.vertex_representatives, dtype=np.int64)
        shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
        if roots.shape != (vertices,) or shifts.shape != (vertices, 2) or np.any(roots < 0) or np.any(roots >= vertices):
            state.add("periodic_affine_torus_manifold", "violated", "mesh")
            return True
        bank = prepared_coordinate_source_bank(geometry, budget=budget)
        numerical = np.asarray(geometry.coordinates, dtype=np.float64)
        if numerical.shape != (len(bank), 2):
            state.add("periodic_affine_torus_trace", "violated", "mesh")
            return True
        for row, values in enumerate(bank):
            with budget.temporary_scope():
                reserve_algebraic_work(values, 2, 4, 1, budget=budget)
                rounded = np.asarray(
                    [value.rounded(budget=budget)[0] if isinstance(value, RadicalElement) else float(value) for value in values],
                    dtype=np.float64,
                )
                if not np.array_equal(rounded.view(np.uint64), numerical[row].view(np.uint64)):
                    state.add("periodic_affine_torus_trace", "violated", "mesh")
                    return True
        elements, routes, _ = geometry._resolve(mesh, exact_source_prepared=True, budget=budget)
        points: dict[int, Point] = {}
        representatives: dict[int, Point] = {}
        edges: dict[Edge, list[int]] = {}
        links: dict[int, dict[Germ, Germ]] = {}
        neighbors: dict[int, set[int]] = {}
        corner_ids: list[str] = []
        doubled_area: Scalar = Fraction(0)
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            for identifier, corners, row in zip(np.asarray(block.global_ids), np.asarray(block.vertices), np.asarray(route), strict=True):
                local = tuple(bank[int(index)] for index in row if index >= 0)
                exact = affine_simplex_corner_images(element, local, budget=budget)
                if exact is None or len(exact) != 3 or any(len(point) != 2 for point in exact):
                    state.add("periodic_affine_torus_orientation", "violated", "cell", (int(identifier),))
                    return True
                with budget.temporary_scope():
                    reserve_algebraic_work((*exact, doubled_area, *lattice), 64, 32, 4, budget=budget)
                    a, b, c = exact
                    signed = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                    if signed <= 0:
                        state.add("periodic_affine_torus_orientation", "violated", "cell", (int(identifier),))
                        return True
                    doubled_area += signed
                    visits, storage = coordinate_retention_preparation_bounds((doubled_area,))
                    area_storage.set_bound(storage, work=visits)
                    corner_ids.append(canonical_fingerprint(tuple(tuple(_scalar_record(value) for value in point) for point in exact)))
                    for vertex, image in zip(corners, exact, strict=True):
                        index = int(vertex)
                        point: Point = (image[0], image[1])
                        old = points.get(index)
                        if old is not None:
                            if old != point:
                                state.add("periodic_affine_torus_trace", "violated", "cell", (int(identifier),))
                                return True
                            continue
                        rounded = tuple(value.rounded(budget=budget)[0] if isinstance(value, RadicalElement) else float(value) for value in point)
                        if not np.array_equal(np.asarray(rounded, dtype=np.float64).view(np.uint64), np.asarray(mesh.coordinates[index], dtype=np.float64).view(np.uint64)):
                            state.add("periodic_affine_torus_trace", "violated", "cell", (int(identifier),))
                            return True
                        points[index] = point
                        root = int(roots[index])
                        base: Point = (
                            point[0] - sum(int(shifts[index, j]) * lattice[j][0] for j in range(2)),
                            point[1] - sum(int(shifts[index, j]) * lattice[j][1] for j in range(2)),
                        )
                        previous = representatives.get(root)
                        if previous is not None and previous != base:
                            state.add("periodic_affine_torus_trace", "violated", "cell", (int(identifier),))
                            return True
                        if previous is None:
                            budget.retain_polynomial_preparation((root, base))
                            representatives[root] = base
                    for slot in range(3):
                        first, second, third = (int(corners[(slot + offset) % 3]) for offset in range(3))
                        key, sign = _edge(first, second, shifts, roots)
                        edges.setdefault(key, []).append(sign)
                        root = int(roots[first])
                        germ_a: Germ = (int(roots[second]), int(shifts[second, 0] - shifts[first, 0]), int(shifts[second, 1] - shifts[first, 1]))
                        germ_b: Germ = (int(roots[third]), int(shifts[third, 0] - shifts[first, 0]), int(shifts[third, 1] - shifts[first, 1]))
                        link = links.setdefault(root, {})
                        if germ_a in link:
                            state.add("periodic_affine_torus_manifold", "violated", "cell", (int(identifier),))
                            return True
                        link[germ_a] = germ_b
                        neighbors.setdefault(root, set()).add(int(roots[second]))
        if len(points) != vertices or set(representatives) != set(np.unique(roots).tolist()) or any(len(sides) != 2 or sum(sides) for sides in edges.values()) or any(not _one_cycle(link) for link in links.values()) or len(representatives) - len(edges) + count != 0:
            state.add("periodic_affine_torus_manifold", "violated", "mesh")
            return True
        pending = [next(iter(representatives))]
        seen: set[int] = set()
        while pending:
            root = pending.pop()
            if root not in seen:
                seen.add(root)
                pending.extend(neighbors[root] - seen)
        if seen != set(representatives):
            state.add("periodic_affine_torus_manifold", "violated", "mesh")
            return True
        if doubled_area != 2 * abs(determinant):
            state.add("periodic_affine_torus_degree_one", "violated", "mesh")
            return True
        area = abs(determinant)
        integral = rational_coefficient(
            doubled_area, role="periodic_affine_torus.signed_pullback_integral"
        ) / 2
        input_binding = canonical_fingerprint(array_tree_fingerprint((mesh, geometry)))
        manifold_id = canonical_fingerprint({"kind": "closed-connected-oriented-torus-links-and-cocycle", "topology": array_tree_fingerprint(renewed), "links": tuple((root, tuple(sorted(link.items()))) for root, link in sorted(links.items()))})
        trace_id = canonical_fingerprint({"kind": "exact-affine-vertex-and-edge-equivariance", "lattice": tuple(tuple((x.numerator, x.denominator) for x in row) for row in lattice), "corners": tuple(corner_ids), "cocycle": array_tree_fingerprint((roots, shifts, topology.corner_shifts))})
        return PeriodicAffineTorusDegreeEvidence(
            mesh.mesh_id, cell_geometry_id(geometry), topology.periodic_topology_id,
            input_binding,
            canonical_fingerprint({"kind": "full-source-affine-triangle-actions", "elements": tuple(element.element_id for element in elements), "corners": tuple(corner_ids)}),
            manifold_id, trace_id,
            canonical_fingerprint({"kind": "strict-positive-oriented-exact-triangle-determinants", "corners": tuple(corner_ids), "area": (area.numerator, area.denominator), "covering_degree": 1}),
            (integral.numerator, integral.denominator),
            (area.numerator, area.denominator),
        )
