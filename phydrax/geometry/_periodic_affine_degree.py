#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact degree-one admission for closed affine translational triangle quotients.

A coherently oriented, positive piecewise-linear map from a closed connected
oriented surface to a flat torus is a branched covering. Its signed pullback
area divided by lattice covolume is its degree. Degree one therefore excludes
multiple sheets and branching; the edge and vertex-link checks below establish
the required source manifold independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_algebraic_geometry import (
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from ..discretization._periodic_cell import PeriodicCell
from ._embedding_cells import EmbeddingCell
from ._embedding_state import (
    MeshCertificateEntityKind,
    MeshCertificateFinding,
    MeshFindingStatus,
    PeriodicAffineTorusDegreeEvidence,
)


type Scalar = ExactScalar
type Point = tuple[Scalar, Scalar]
type Germ = tuple[int, int, int]
type Edge = tuple[int, int, int, int]
type Lattice = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
type _ScalarRecord = tuple[tuple[int, int, int], ...]
type _PointRecord = tuple[_ScalarRecord, ...]
type _TriangleRecord = tuple[_PointRecord, ...]


@dataclass(frozen=True, slots=True)
class PeriodicAffineDegreeResult:
    """Applicability, exact theorem checks, and evidence for one quotient."""

    applicable: bool
    checks: tuple[str, ...] = ()
    findings: tuple[MeshCertificateFinding, ...] = ()
    evidence: PeriodicAffineTorusDegreeEvidence | None = None


def _scalar_record(value: Scalar, /) -> tuple[tuple[int, int, int], ...]:
    terms = value.terms if isinstance(value, RadicalElement) else ((1, value),)
    return tuple(
        (root, coefficient.numerator, coefficient.denominator)
        for root, coefficient in terms
        if coefficient
    )


def _edge(
    first: int,
    second: int,
    shifts: np.ndarray,
    roots: np.ndarray,
    /,
) -> tuple[Edge, int]:
    first_root, second_root = int(roots[first]), int(roots[second])
    delta = tuple(int(value) for value in shifts[second] - shifts[first])
    forward: Edge = (first_root, second_root, delta[0], delta[1])
    backward: Edge = (second_root, first_root, -delta[0], -delta[1])
    return (forward, 1) if forward < backward else (backward, -1)


def _one_cycle(links: dict[Germ, Germ], /) -> bool:
    if (
        not links
        or set(links) != set(links.values())
        or len(set(links.values())) != len(links)
    ):
        return False
    start = next(iter(links))
    current, seen = start, set()
    while current not in seen:
        seen.add(current)
        current = links[current]
    return current == start and len(seen) == len(links)


def _finding(
    check: str,
    status: MeshFindingStatus,
    entity_kind: MeshCertificateEntityKind,
    entity_ids: tuple[int, ...] = (),
    /,
    *,
    detail: str,
) -> MeshCertificateFinding:
    return MeshCertificateFinding(check, status, entity_kind, entity_ids, detail=detail)


def _translated_point(
    point: Point,
    shift: tuple[int, int],
    lattice: Lattice,
    /,
) -> Point:
    x: ExactScalar = point[0] + sum(
        (shift[generator] * lattice[generator][0] for generator in range(2)),
        Fraction(0),
    )
    y: ExactScalar = point[1] + sum(
        (shift[generator] * lattice[generator][1] for generator in range(2)),
        Fraction(0),
    )
    return x, y


def _torus_manifold(
    cells: tuple[EmbeddingCell, ...],
    roots: np.ndarray,
    shifts: np.ndarray,
    /,
) -> tuple[str | None, tuple[int, ...]]:
    edges: dict[Edge, list[int]] = {}
    links: dict[int, dict[Germ, Germ]] = {}
    neighbors: dict[int, set[int]] = {}
    used_vertices: set[int] = set()
    for cell in cells:
        if len(cell.vertex_indices) != 3:
            return "periodic_affine_torus_manifold", (cell.global_id,)
        corners = cell.vertex_indices
        for slot in range(3):
            first = corners[slot]
            second = corners[(slot + 1) % 3]
            third = corners[(slot + 2) % 3]
            key, sign = _edge(first, second, shifts, roots)
            edges.setdefault(key, []).append(sign)
            root = int(roots[first])
            germ_a: Germ = (
                int(roots[second]),
                int(shifts[second, 0] - shifts[first, 0]),
                int(shifts[second, 1] - shifts[first, 1]),
            )
            germ_b: Germ = (
                int(roots[third]),
                int(shifts[third, 0] - shifts[first, 0]),
                int(shifts[third, 1] - shifts[first, 1]),
            )
            link = links.setdefault(root, {})
            if germ_a in link:
                return "periodic_affine_torus_manifold", (cell.global_id,)
            link[germ_a] = germ_b
            neighbors.setdefault(root, set()).add(int(roots[second]))
            used_vertices.add(root)
    if any(len(sides) != 2 or sum(sides) != 0 for sides in edges.values()):
        return "periodic_affine_torus_manifold", ()
    if any(not _one_cycle(link) for link in links.values()):
        return "periodic_affine_torus_manifold", ()
    if len(used_vertices) != len(links) or not links:
        return "periodic_affine_torus_manifold", ()
    pending = [next(iter(links))]
    seen: set[int] = set()
    while pending:
        vertex = pending.pop()
        if vertex not in seen:
            seen.add(vertex)
            pending.extend(neighbors.get(vertex, set()) - seen)
    if seen != set(links) or len(links) - len(edges) + len(cells) != 0:
        return "periodic_affine_torus_manifold", ()
    return None, ()


def certify_affine_torus_degree(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    cells: tuple[EmbeddingCell, ...],
    binding_id: str,
    /,
) -> PeriodicAffineDegreeResult:
    """Prove a closed rank-two translation quotient has covering degree one.

    ``applicable`` is false only when this exact theorem does not cover the
    input representation. Once admitted, any failed premise is a finding and
    must not fall through to a weaker contact route.
    """
    topology = mesh.periodic_topology
    if (
        topology is None
        or not isinstance(topology.cell, PeriodicCell)
        or mesh.topological_dimension != 2
        or mesh.ambient_dimension != 2
        or topology.cell.rank != 2
        or not topology.cell.fully_periodic
        or not cells
        or any(block.cell_kind != "triangle" for block in mesh.blocks)
        or any(cell.kind != "triangle" or not cell.affine for cell in cells)
    ):
        return PeriodicAffineDegreeResult(False)

    checks = (
        "periodic_affine_torus_manifold",
        "periodic_affine_torus_trace",
        "periodic_affine_torus_orientation",
        "periodic_affine_torus_degree_one",
    )
    findings: list[MeshCertificateFinding] = []
    if (
        not validity.all_certified
        or validity.geometry_id != cell_geometry_id(geometry)
        or not binding_id
        or len(cells) != sum(block.cell_count for block in mesh.blocks)
    ):
        findings.append(
            _finding(
                checks[0],
                "unresolved",
                "mesh",
                detail="The exact torus proof is not bound to complete positive current geometry.",
            )
        )
        return PeriodicAffineDegreeResult(True, checks, tuple(findings))

    try:
        renewed = CellMesh.from_stored_fields(mesh, actual_geometry=geometry)
    except (ArithmeticError, IndexError, TypeError, ValueError) as error:
        findings.append(
            _finding(
                checks[0],
                "unresolved",
                "mesh",
                detail=f"The current periodic quotient could not be reconstructed: {error}",
            )
        )
        return PeriodicAffineDegreeResult(True, checks, tuple(findings))
    if (
        renewed.periodic_topology is None
        or renewed.periodic_topology.periodic_topology_id != topology.periodic_topology_id
    ):
        findings.append(
            _finding(
                checks[0],
                "violated",
                "mesh",
                detail="Reconstructed periodic incidence and winding cocycle differ from the declared quotient.",
            )
        )
        return PeriodicAffineDegreeResult(True, checks, tuple(findings))

    raw_vectors = np.asarray(topology.cell.vectors, dtype=np.float64)
    vectors: Lattice = (
        (
            Fraction(float(raw_vectors[0, 0])),
            Fraction(float(raw_vectors[0, 1])),
        ),
        (
            Fraction(float(raw_vectors[1, 0])),
            Fraction(float(raw_vectors[1, 1])),
        ),
    )
    determinant = vectors[0][0] * vectors[1][1] - vectors[0][1] * vectors[1][0]
    if not determinant:
        findings.append(
            _finding(
                checks[1],
                "violated",
                "mesh",
                detail="The exact rank-two translation lattice has zero covolume.",
            )
        )
        return PeriodicAffineDegreeResult(True, checks, tuple(findings))

    vertex_count = mesh.coordinates.shape[0]
    roots = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    fixed = np.asarray(topology.vertex_fixed_generators, dtype=np.bool_)
    if (
        roots.shape != (vertex_count,)
        or shifts.shape != (vertex_count, 2)
        or fixed.shape != shifts.shape
        or np.any(roots < 0)
        or np.any(roots >= vertex_count)
        or np.any(roots[roots] != roots)
        or np.any(shifts[roots] != 0)
        or np.any(fixed)
    ):
        findings.append(
            _finding(
                checks[0],
                "violated",
                "mesh",
                detail="The quotient has malformed vertex orbits or a fixed translation generator.",
            )
        )
        return PeriodicAffineDegreeResult(True, checks, tuple(findings))

    actual_rows: dict[tuple[str, int], int] = {}
    for row, cell in enumerate(cells):
        key = (cell.block_name, cell.local_index)
        if (
            key in actual_rows
            or len(cell.corner_images) != 3
            or any(len(point) != 2 for point in cell.corner_images)
        ):
            findings.append(
                _finding(
                    checks[0],
                    "violated",
                    "cell",
                    (cell.global_id,),
                    detail="The affine triangle source rows are incomplete or duplicated.",
                )
            )
            return PeriodicAffineDegreeResult(True, checks, tuple(findings))
        actual_rows[key] = row
    expected_rows = {
        (block.name, local_index)
        for block in mesh.blocks
        for local_index in range(block.cell_count)
    }
    if set(actual_rows) != expected_rows:
        findings.append(
            _finding(
                checks[0],
                "violated",
                "mesh",
                detail="The exact affine source bank does not cover every current triangle row exactly once.",
            )
        )
        return PeriodicAffineDegreeResult(True, checks, tuple(findings))
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    points: dict[int, Point] = {}
    representatives: dict[int, Point] = {}
    area: ExactScalar = Fraction(0)
    corner_records: list[_TriangleRecord] = []
    orientation_records: list[tuple[int, _TriangleRecord]] = []
    trace_records: list[tuple[int, int, tuple[int, int], _PointRecord]] = []
    for block in mesh.blocks:
        block_identifiers = np.asarray(block.global_ids, dtype=np.int64).tolist()
        block_vertices = np.asarray(block.vertices, dtype=np.int64).tolist()
        for local_index, (identifier, vertices) in enumerate(
            zip(block_identifiers, block_vertices, strict=True)
        ):
            cell = cells[actual_rows[(block.name, local_index)]]
            if tuple(int(value) for value in vertices) != cell.vertex_indices:
                findings.append(
                    _finding(
                        checks[1],
                        "violated",
                        "cell",
                        (int(identifier),),
                        detail="The exact affine source corners do not follow current triangle incidence.",
                    )
                )
                return PeriodicAffineDegreeResult(True, checks, tuple(findings))
            exact = tuple((point[0], point[1]) for point in cell.corner_images)
            corner_records.append(
                tuple(tuple(_scalar_record(value) for value in point) for point in exact)
            )
            a, b, c = exact
            signed = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if signed <= 0:
                findings.append(
                    _finding(
                        checks[2],
                        "violated",
                        "cell",
                        (int(identifier),),
                        detail="An exact affine quotient triangle has nonpositive signed determinant.",
                    )
                )
                return PeriodicAffineDegreeResult(True, checks, tuple(findings))
            area += signed
            orientation_records.append((int(identifier), corner_records[-1]))
            for vertex, image in zip(vertices, exact, strict=True):
                index = int(vertex)
                previous = points.get(index)
                if previous is not None and previous != image:
                    findings.append(
                        _finding(
                            checks[1],
                            "violated",
                            "cell",
                            (int(identifier),),
                            detail="The exact affine source map disagrees at a shared lifted vertex.",
                        )
                    )
                    return PeriodicAffineDegreeResult(True, checks, tuple(findings))
                points[index] = image
                root = int(roots[index])
                shift = (int(shifts[index, 0]), int(shifts[index, 1]))
                base = _translated_point(
                    (image[0], image[1]),
                    (-shift[0], -shift[1]),
                    vectors,
                )
                prior = representatives.get(root)
                if prior is not None and prior != base:
                    findings.append(
                        _finding(
                            checks[1],
                            "violated",
                            "cell",
                            (int(identifier),),
                            detail="The affine source map is not equivariant with its declared lattice winding.",
                        )
                    )
                    return PeriodicAffineDegreeResult(True, checks, tuple(findings))
                representatives[root] = (base[0], base[1])
                trace_records.append(
                    (
                        int(vertex_ids[index]),
                        int(vertex_ids[root]),
                        shift,
                        tuple(_scalar_record(value) for value in base),
                    )
                )
    if len(points) != vertex_count or set(representatives) != set(
        np.unique(roots).tolist()
    ):
        findings.append(
            _finding(
                checks[1],
                "violated",
                "mesh",
                detail="The exact affine source map does not cover every current lifted vertex orbit.",
            )
        )
        return PeriodicAffineDegreeResult(True, checks, tuple(findings))

    reason, bad_cells = _torus_manifold(cells, roots, shifts)
    if reason is not None:
        findings.append(
            _finding(
                reason,
                "violated",
                "cell" if bad_cells else "mesh",
                bad_cells,
                detail="The quotient triangles do not form one connected, closed oriented two-manifold with cyclic vertex links and torus Euler characteristic.",
            )
        )
        return PeriodicAffineDegreeResult(True, checks, tuple(findings))

    lattice_area = abs(determinant)
    if area != 2 * lattice_area:
        findings.append(
            _finding(
                checks[3],
                "violated",
                "mesh",
                detail="The exact positive pullback area is not one lattice covolume.",
            )
        )
        return PeriodicAffineDegreeResult(True, checks, tuple(findings))

    pullback_area = rational_coefficient(
        area / 2, role="periodic_affine_torus.signed_pullback_area"
    )
    affinity_id = canonical_fingerprint(
        {
            "kind": "full-source-affine-periodic-triangles",
            "geometry": geometry.geometry_layout_id,
            "corners": tuple(corner_records),
            "cells": tuple(cell.global_id for cell in cells),
        }
    )
    manifold_id = canonical_fingerprint(
        {
            "kind": "connected-oriented-torus-triangle-links",
            "topology": array_tree_fingerprint(renewed.periodic_topology),
            "vertices": tuple(
                sorted(
                    (root, tuple(sorted(link.items())))
                    for root, link in _links(cells, roots, shifts).items()
                )
            ),
        }
    )
    trace_id = canonical_fingerprint(
        {
            "kind": "exact-affine-lattice-equivariance",
            "lattice": tuple(
                tuple((value.numerator, value.denominator) for value in row)
                for row in vectors
            ),
            "traces": tuple(trace_records),
            "corners": tuple(corner_records),
        }
    )
    orientation_id = canonical_fingerprint(
        {
            "kind": "positive-affine-triangle-determinants",
            "cells": tuple(orientation_records),
            "pullback_area": (pullback_area.numerator, pullback_area.denominator),
        }
    )
    evidence = PeriodicAffineTorusDegreeEvidence(
        mesh.mesh_id,
        cell_geometry_id(geometry),
        topology.periodic_topology_id,
        canonical_fingerprint(
            {
                "kind": "periodic-affine-degree-input-binding",
                "binding": binding_id,
                "inputs": array_tree_fingerprint((mesh, geometry, validity)),
            }
        ),
        affinity_id,
        manifold_id,
        trace_id,
        orientation_id,
        (pullback_area.numerator, pullback_area.denominator),
        (lattice_area.numerator, lattice_area.denominator),
    )
    return PeriodicAffineDegreeResult(True, checks, evidence=evidence)


def _links(
    cells: tuple[EmbeddingCell, ...], roots: np.ndarray, shifts: np.ndarray, /
) -> dict[int, dict[Germ, Germ]]:
    links: dict[int, dict[Germ, Germ]] = {}
    for cell in cells:
        corners = cell.vertex_indices
        for slot in range(3):
            first = corners[slot]
            second = corners[(slot + 1) % 3]
            third = corners[(slot + 2) % 3]
            root = int(roots[first])
            first_germ: Germ = (
                int(roots[second]),
                int(shifts[second, 0] - shifts[first, 0]),
                int(shifts[second, 1] - shifts[first, 1]),
            )
            second_germ: Germ = (
                int(roots[third]),
                int(shifts[third, 0] - shifts[first, 0]),
                int(shifts[third, 1] - shifts[first, 1]),
            )
            links.setdefault(root, {})[first_germ] = second_germ
    return links


__all__ = ["PeriodicAffineDegreeResult", "certify_affine_torus_degree"]
