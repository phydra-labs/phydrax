#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Build a conforming execution PLC without replacing its source authority."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ._layer_core_periodic import partition_periodic_seams, PeriodicSeamPartition
from ._layer_core_subdivision import NestedLayerPartition
from ._plc_source import PiecewiseLinearComplex
from .providers._native_sources import NativeLayerCoreSource


Stencil = tuple[tuple[int, Fraction], ...]


@dataclass(frozen=True, slots=True)
class LayerCoreExecution:
    """Transient carrier and exact relations back to the authored PLC.

    Polygon source rows are candidates, not an assertion that the entire
    execution triangle belongs to every candidate.  Exact geometric overlap
    determines the many-to-many relation.
    """

    complex: PiecewiseLinearComplex
    vertex_layer_ids: np.ndarray
    cap_polygon_ids: np.ndarray
    vertex_stencils: tuple[Stencil, ...]
    source_polygon_candidates: tuple[tuple[int, ...], ...]
    periodic_seams: tuple[PeriodicSeamPartition, ...]
    protected_facets: np.ndarray


def build_layer_core_execution(
    source: NativeLayerCoreSource,
    partition: NestedLayerPartition,
    /,
    *,
    source_authority: NativeLayerCoreSource | None = None,
) -> LayerCoreExecution:
    """Restrict cap triangles to actual child prisms and conform adjoining faces.

    All stencils refer to original PLC vertex rows.  Original vertices and
    source incidence remain available even where a coplanar periodic quad is
    retriangulated across its former diagonal.
    """
    if source_authority is not None and source_authority.reference_source is not source:
        raise ValueError("Mapped execution must retain its original reference source.")
    complex_ = source.complex
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    flat = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    polygon_rows = tuple(
        tuple(int(v) for v in flat[a:b])
        for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    )
    if any(len(polygon) != 3 for polygon in polygon_rows):
        raise ValueError(
            "Conforming layer execution requires triangular source polygons."
        )
    polygons = tuple((row[0], row[1], row[2]) for row in polygon_rows)
    original_points = np.asarray(complex_.vertices, dtype=np.float64)
    original_exact = tuple(
        (Fraction(float(point[0])), Fraction(float(point[1])), Fraction(float(point[2])))
        for point in original_points
    )
    layer_rows = np.asarray(source.vertex_layer_ids, dtype=np.int64)
    layer_ids = np.asarray(source.layers.mesh.vertex_global_ids, dtype=np.int64)
    source_by_layer: dict[int, int] = {}
    for row, layer_row in enumerate(layer_rows.tolist()):
        if layer_row >= 0:
            identifier = int(layer_ids[layer_row])
            if identifier in source_by_layer:
                raise ValueError("A layer vertex has ambiguous PLC ancestry.")
            source_by_layer[identifier] = row
    stencils: list[Stencil] = [
        ((row, Fraction(1)),) for row in range(len(original_points))
    ]
    lookup = {stencil: row for row, stencil in enumerate(stencils)}
    target_rows = [-1] * len(stencils)
    for row, layer_row in enumerate(layer_rows.tolist()):
        if layer_row >= 0:
            target_rows[row] = layer_row

    def add(stencil: Stencil, layer_row: int = -1) -> int:
        stencil = tuple(sorted((int(row), weight) for row, weight in stencil if weight))
        if not stencil or sum((weight for _, weight in stencil), Fraction()) != 1:
            raise ValueError("An execution vertex must have an affine source stencil.")
        row = lookup.get(stencil)
        if row is None:
            row = len(stencils)
            lookup[stencil] = row
            stencils.append(stencil)
            target_rows.append(layer_row)
        elif layer_row >= 0:
            if target_rows[row] not in (-1, layer_row):
                raise ValueError("An execution stencil names different layer vertices.")
            target_rows[row] = layer_row
        return row

    def add_partition_vertex(row: int) -> int:
        stencil = partition.stencils[row]
        if not all(identifier in source_by_layer for identifier, _ in stencil):
            raise ValueError("A refined cap vertex has no original PLC ancestry.")
        converted = tuple(
            sorted(
                (source_by_layer[identifier], weight) for identifier, weight in stencil
            )
        )
        return add(converted, row)

    cap_ids = tuple(
        int(row) for row in np.asarray(source.cap_polygon_ids, dtype=np.int64)
    )
    cap_by_vertices = {frozenset(polygons[row]): row for row in cap_ids}
    if len(cap_by_vertices) != len(cap_ids):
        raise ValueError("Source cap triangles must have distinct vertex sets.")
    parent_caps: dict[tuple[int, int], int] = {}
    for block in source.layers.mesh.blocks:
        for identifier, vertices in zip(
            block.global_ids.tolist(), block.vertices.tolist(), strict=True
        ):
            for side, local in enumerate(((0, 1, 2), (3, 4, 5))):
                ids = [int(layer_ids[int(vertices[index])]) for index in local]
                if all(identifier_ in source_by_layer for identifier_ in ids):
                    polygon = cap_by_vertices.get(
                        frozenset(source_by_layer[identifier_] for identifier_ in ids)
                    )
                    if polygon is not None:
                        parent_caps[(int(identifier), side)] = polygon

    def exact(row: int) -> tuple[Fraction, Fraction, Fraction]:
        coordinates = tuple(
            sum(
                (weight * original_exact[index][axis] for index, weight in stencils[row]),
                Fraction(),
            )
            for axis in range(3)
        )
        return coordinates[0], coordinates[1], coordinates[2]

    def normal(
        vertices: tuple[int, int, int],
        points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    ) -> tuple[Fraction, Fraction, Fraction]:
        a, b, c = (points[index] for index in vertices)
        u = tuple(b[i] - a[i] for i in range(3))
        v = tuple(c[i] - a[i] for i in range(3))
        return (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )

    cap_children: dict[int, list[tuple[int, int, int]]] = {row: [] for row in cap_ids}
    for parent, vertices in zip(
        partition.parent_cell_ids, partition.prism_vertices, strict=True
    ):
        for side, local in enumerate(((0, 1, 2), (3, 4, 5))):
            source_row = parent_caps.get((int(parent), side))
            if source_row is None:
                continue
            child = (
                add_partition_vertex(int(vertices[local[0]])),
                add_partition_vertex(int(vertices[local[1]])),
                add_partition_vertex(int(vertices[local[2]])),
            )
            source_normal = normal(polygons[source_row], original_exact)
            child_points = tuple(exact(row) for row in child)
            child_normal = normal((0, 1, 2), child_points)
            orientation = sum(
                (a * b for a, b in zip(source_normal, child_normal, strict=True)),
                Fraction(),
            )
            if not orientation:
                raise ValueError("A refined cap triangle is degenerate.")
            if orientation < 0:
                child = (child[0], child[2], child[1])
            cap_children[source_row].append(child)
    if any(not children for children in cap_children.values()):
        raise ValueError("A source cap triangle has no refined prism descendants.")

    removable_edges: frozenset[tuple[int, int]] = frozenset()
    if source.noncap_refinement == "conforming":
        from ._layer_core_edge_policy import noncap_source_edge_classification

        classification = (
            noncap_source_edge_classification(source)
            if source_authority is None
            else noncap_source_edge_classification(source_authority, partition=partition)
        )
        removable_edges = frozenset() if classification is None else classification[0]

    seams = (
        ()
        if source.core_seam_polygon_pairs is None
        else partition_periodic_seams(source, partition)
    )
    seam_polygons: set[int] = set()
    seam_triangles: dict[int, list[tuple[tuple[int, int, int], tuple[int, ...]]]] = {}
    for seam in seams:
        for source_pair, triangles in zip(
            seam.source_polygons, (seam.first, seam.second), strict=True
        ):
            first, second = source_pair
            shared = set(polygons[first]) & set(polygons[second])
            explicit_segments = {
                frozenset((int(a), int(b)))
                for a, b in np.asarray(complex_.segments, dtype=np.int64)
            }
            if len(shared) != 2 or frozenset(shared) in explicit_segments:
                raise ValueError(
                    "A periodic seam cannot remove an explicit source segment."
                )
            if (
                source_authority is not None
                and tuple(sorted(shared)) not in removable_edges
            ):
                raise ValueError(
                    "A mapped periodic seam diagonal lacks exact removable-edge authority."
                )
            if first in seam_polygons or second in seam_polygons:
                raise ValueError("A source seam polygon belongs to multiple strips.")
            if not np.array_equal(
                np.asarray(complex_.facet_regions)[int(complex_.polygon_facets[first])],
                np.asarray(complex_.facet_regions)[int(complex_.polygon_facets[second])],
            ):
                raise ValueError(
                    "A retriangulated seam quad has incompatible material sides."
                )
            seam_polygons.update(source_pair)
            seam_triangles[first] = [
                (
                    (add(triangle[0]), add(triangle[1]), add(triangle[2])),
                    tuple(source_pair),
                )
                for triangle in triangles
            ]

    # Retriangulate coplanar side quads across their synthetic diagonals.
    cap_edges = {
        tuple(sorted((polygon[i], polygon[(i + 1) % 3])))
        for row in cap_ids
        for polygon in (polygons[row],)
        for i in range(3)
    }
    polygon_edges = tuple(
        {tuple(sorted((polygon[i], polygon[(i + 1) % 3]))) for i in range(3)}
        for polygon in polygons
    )
    explicit_edges = {
        tuple(sorted((int(a), int(b))))
        for a, b in np.asarray(complex_.segments, dtype=np.int64)
    }
    strip_rows: set[int] = set()
    strips: dict[int, list[tuple[tuple[int, int, int], tuple[int, ...]]]] = {}
    for row, polygon in enumerate(polygons):
        if row in cap_children or row in seam_polygons or row in strip_rows:
            continue
        for edge in sorted(polygon_edges[row] & cap_edges):
            a, b = edge
            divisions = sorted(
                weight
                for stencil in stencils
                if len(stencil) == 2
                for (first, _), (second, weight) in (stencil,)
                if (first, second) == edge and 0 < weight < 1
            )
            if not divisions:
                continue
            for other in range(len(polygons)):
                if (
                    other == row
                    or other in cap_children
                    or other in seam_polygons
                    or other in strip_rows
                ):
                    continue
                shared = polygon_edges[row] & polygon_edges[other]
                boundary = polygon_edges[row] ^ polygon_edges[other]
                if len(shared) != 1 or next(iter(shared)) in explicit_edges:
                    continue
                if (
                    source_authority is not None
                    and next(iter(shared)) not in removable_edges
                ):
                    continue
                if len(
                    set(polygon) | set(polygons[other])
                ) != 4 or boundary & cap_edges != {edge}:
                    continue
                first_facet = int(complex_.polygon_facets[row])
                second_facet = int(complex_.polygon_facets[other])
                if not np.array_equal(
                    complex_.facet_regions[first_facet],
                    complex_.facet_regions[second_facet],
                ):
                    continue
                a_next = [
                    v for e in boundary if a in e and b not in e for v in e if v != a
                ]
                b_next = [
                    v for e in boundary if b in e and a not in e for v in e if v != b
                ]
                if len(a_next) != 1 or len(b_next) != 1:
                    continue
                upper_a, upper_b = a_next[0], b_next[0]
                if tuple(sorted((upper_a, upper_b))) not in boundary:
                    continue
                n = normal((a, b, upper_a), original_exact)
                delta = tuple(
                    original_exact[upper_b][i] - original_exact[a][i] for i in range(3)
                )
                if sum((n[i] * delta[i] for i in range(3)), Fraction()):
                    continue
                if (a, b) not in tuple(
                    (polygon[i], polygon[(i + 1) % 3]) for i in range(3)
                ):
                    a, b, upper_a, upper_b = b, a, upper_b, upper_a
                    divisions = sorted(1 - t for t in divisions)

                def interpolate(x: int, y: int, t: Fraction) -> int:
                    return add(
                        tuple(
                            sorted(
                                (vertex, weight)
                                for vertex, weight in ((x, 1 - t), (y, t))
                                if weight
                            )
                        )
                    )

                fractions = [Fraction(), *divisions, Fraction(1)]
                triangles = []
                for left, right in zip(fractions[:-1], fractions[1:], strict=True):
                    low_a, low_b = interpolate(a, b, left), interpolate(a, b, right)
                    high_a, high_b = (
                        interpolate(upper_a, upper_b, left),
                        interpolate(upper_a, upper_b, right),
                    )
                    triangles.extend(((low_a, low_b, high_b), (low_a, high_b, high_a)))
                strips[row] = [(triangle, (row, other)) for triangle in triangles]
                strip_rows.update((row, other))
                break
            if row in strip_rows:
                break
    # Recover cap-to-opposite vertex incidence from the proved lateral quads.
    # The top grid is only installed when it has exactly the same triangular
    # source partition; no coordinate proximity or inferred source ID is used.
    opposite: dict[int, int] = {}
    quad_pairs = [
        ancestry for triangles in strips.values() for _, ancestry in triangles[:1]
    ]
    quad_pairs.extend(pair for seam in seams for pair in seam.source_polygons)
    for first, second in quad_pairs:
        boundary = polygon_edges[first] ^ polygon_edges[second]
        lower = boundary & cap_edges
        if len(lower) != 1:
            raise ValueError("A lateral quad must have exactly one source cap edge.")
        a, b = next(iter(lower))
        for vertex, other_vertex in ((a, b), (b, a)):
            neighbors = [
                v
                for edge in boundary
                if vertex in edge and other_vertex not in edge
                for v in edge
                if v != vertex
            ]
            if len(neighbors) != 1:
                raise ValueError("A lateral quad has ambiguous opposite incidence.")
            previous = opposite.setdefault(vertex, neighbors[0])
            if previous != neighbors[0]:
                raise ValueError("Lateral quads disagree on a cap vertex's opposite.")
    top_by_vertices: dict[frozenset[int], list[int]] = {}
    for row, polygon in enumerate(polygons):
        if row not in cap_children and row not in seam_polygons and row not in strip_rows:
            top_by_vertices.setdefault(frozenset(polygon), []).append(row)
    top_children: dict[int, list[tuple[int, int, int]]] = {}
    for cap, children in cap_children.items():
        if not all(vertex in opposite for vertex in polygons[cap]):
            continue
        top_vertices = frozenset(opposite[vertex] for vertex in polygons[cap])
        matches = top_by_vertices.get(top_vertices, [])
        if len(matches) != 1:
            continue
        top = matches[0]
        if top in top_children:
            raise ValueError("Two cap triangles have the same opposite source triangle.")
        top_normal = normal(polygons[top], original_exact)
        target = []
        for child in children:
            mapped = (
                add(
                    tuple(
                        sorted(
                            (opposite[vertex], weight)
                            for vertex, weight in stencils[child[0]]
                        )
                    )
                ),
                add(
                    tuple(
                        sorted(
                            (opposite[vertex], weight)
                            for vertex, weight in stencils[child[1]]
                        )
                    )
                ),
                add(
                    tuple(
                        sorted(
                            (opposite[vertex], weight)
                            for vertex, weight in stencils[child[2]]
                        )
                    )
                ),
            )
            mapped_points = tuple(exact(row) for row in mapped)
            mapped_normal = normal((0, 1, 2), mapped_points)
            orientation = sum(
                (a * b for a, b in zip(top_normal, mapped_normal, strict=True)),
                Fraction(),
            )
            if not orientation:
                raise ValueError("An opposite execution triangle is degenerate.")
            target.append(
                mapped if orientation > 0 else (mapped[0], mapped[2], mapped[1])
            )
        top_children[top] = target

    # Split all remaining source edges at every introduced point.  This also
    # conforms the opposite seam edge to its neighboring, unprotected face.
    edge_points: dict[tuple[int, int], list[tuple[Fraction, int]]] = {}
    for row, stencil in enumerate(stencils):
        if len(stencil) == 2 and all(weight > 0 for _, weight in stencil):
            (a, _), (b, t) = stencil
            edge_points.setdefault((a, b), []).append((t, row))
    for points in edge_points.values():
        points.sort()

    def split_triangle(triangle: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        pieces = [triangle]
        for i in range(3):
            a, b = triangle[i], triangle[(i + 1) % 3]
            key = (min(a, b), max(a, b))
            points = edge_points.get(key, [])
            ordered = points if a < b else list(reversed(points))
            previous = a
            for _, vertex in ordered:
                for index, face in enumerate(pieces):
                    for j in range(3):
                        if face[j] == previous and face[(j + 1) % 3] == b:
                            other = face[(j + 2) % 3]
                            pieces[index : index + 1] = [
                                (previous, vertex, other),
                                (vertex, b, other),
                            ]
                            break
                    else:
                        continue
                    break
                else:
                    raise ValueError("Cannot conform an execution PLC edge.")
                previous = vertex
        return pieces

    output: list[tuple[int, int, int]] = []
    sources: list[tuple[int, ...]] = []
    owners: list[int] = []
    protected: list[bool] = []
    output_caps: list[int] = []
    source_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
    for row, polygon in enumerate(polygons):
        if row in cap_children:
            triangles = [(triangle, (row,)) for triangle in cap_children[row]]
            is_protected = True
        elif row in seam_polygons:
            triangles = seam_triangles.get(row, [])
            is_protected = True
        elif row in strip_rows:
            triangles = strips.get(row, [])
            is_protected = False
        elif row in top_children:
            triangles = [(triangle, (row,)) for triangle in top_children[row]]
            is_protected = False
        else:
            triangles = [(triangle, (row,)) for triangle in split_triangle(polygon)]
            is_protected = False
        for triangle, ancestry in triangles:
            if row in cap_children:
                output_caps.append(len(output))
            output.append(triangle)
            sources.append(ancestry)
            owners.append(int(source_facets[row]))
            protected.append(is_protected)

    # Only remove synthetic triangulation locks when both incident execution
    # triangles have the same exact oriented plane, material sides, and
    # authoritative source face occurrence. Explicit PLC segments remain
    # constraints even when both sides belong to that face.
    components = list(range(len(output)))
    if source.noncap_refinement == "conforming":

        def find(row: int) -> int:
            while components[row] != row:
                components[row] = components[components[row]]
                row = components[row]
            return row

        def signature(row: int) -> tuple | None:
            if protected[row]:
                return None
            facets = tuple(int(source_facets[index]) for index in sources[row])
            identities = {
                (
                    source.core_facet_source_ids[facet],
                    source.core_facet_source_occurrence_paths[facet],
                )
                for facet in facets
            }
            if len(identities) != 1:
                return None
            identity = next(iter(identities))
            if identity[0] is None or identity[1] is None:
                return None
            sides = tuple(int(value) for value in complex_.facet_regions[facets[0]])
            if any(
                tuple(int(value) for value in complex_.facet_regions[facet]) != sides
                for facet in facets
            ):
                return None
            points = tuple(exact(vertex) for vertex in output[row])
            n = normal((0, 1, 2), points)
            pivot = next((abs(value) for value in n if value), None)
            if pivot is None:
                raise ValueError("An execution triangle is degenerate.")
            plane = tuple(value / pivot for value in n)
            offset = sum((plane[axis] * points[0][axis] for axis in range(3)), Fraction())
            plane_metadata = tuple(
                None
                if values is None
                else tuple(np.asarray(values[facet]).tobytes() for facet in facets)
                for values in (complex_.facet_planes, complex_.facet_plane_tolerances)
            )
            if any(
                values is not None and len(set(values)) != 1 for values in plane_metadata
            ):
                return None
            # For mapped sources the classifier proves the common actual root
            # face across each removable edge; original per-facet identities
            # remain in the source-polygon overlap relation.
            return (
                (None if source_authority is not None else identity),
                sides,
                plane,
                offset,
                tuple(None if values is None else values[0] for values in plane_metadata),
            )

        signatures = tuple(signature(row) for row in range(len(output)))
        edge_rows: dict[tuple[int, int], list[int]] = {}
        for row, triangle in enumerate(output):
            for index in range(3):
                edge = (
                    min(triangle[index], triangle[(index + 1) % 3]),
                    max(triangle[index], triangle[(index + 1) % 3]),
                )
                edge_rows.setdefault(edge, []).append(row)
        for edge, rows in edge_rows.items():
            if len(rows) != 2:
                continue
            roots = {
                root for vertex in edge for root, weight in stencils[vertex] if weight
            }
            if len(roots) != 2 or tuple(sorted(roots)) not in removable_edges:
                continue
            first, second = rows
            if signatures[first] is None or signatures[first] != signatures[second]:
                continue
            left, right = find(first), find(second)
            if left != right:
                components[max(left, right)] = min(left, right)
        components = [find(row) for row in range(len(output))]

    # Facet groups must not mix protected and refinable triangles.
    groups: dict[tuple[int, bool, int], int] = {}
    polygon_facets = []
    for output_row, (facet, guard, ancestry) in enumerate(
        zip(owners, protected, sources, strict=True)
    ):
        # Protected triangles are individual facets; strip facets can merge.
        discriminator = (
            output_row + 1 if guard else (-min(ancestry) - 1 if len(ancestry) > 1 else 0)
        )
        if source.noncap_refinement == "conforming" and not guard:
            representative = components[output_row]
            key = (owners[representative], False, -representative - 1)
        else:
            key = (facet, guard, discriminator)
        if key not in groups:
            groups[key] = len(groups)
        polygon_facets.append(groups[key])
    ordered_groups = tuple(groups)
    facet_ids = tuple(
        f"execution-{source.binding_id}-facet-{row}" for row in range(len(ordered_groups))
    )
    execution_path = ("execution", source.binding_id)
    facet_paths = (execution_path,) * len(ordered_groups)
    candidates: list[set[int]] = [set() for _ in ordered_groups]
    for facet, ancestry in zip(polygon_facets, sources, strict=True):
        candidates[facet].update(ancestry)
    points = np.asarray(
        [[float(value) for value in exact(row)] for row in range(len(stencils))],
        dtype=np.float64,
    )
    points[: len(original_points)] = original_points
    for row, target in enumerate(target_rows):
        if target >= 0:
            points[row] = partition.coordinates[target]
    vertex_ids = complex_.vertex_ids + tuple(
        f"execution-vertex-{row}" for row in range(len(original_points), len(stencils))
    )
    vertex_paths = complex_.vertex_occurrence_paths + tuple(
        ("execution", str(row)) for row in range(len(original_points), len(stencils))
    )
    segments: list[tuple[int, int]] = []
    segment_ids: list[str] = []
    segment_paths: list[tuple[str, ...]] = []
    for index, (a_, b_) in enumerate(
        np.asarray(complex_.segments, dtype=np.int64).tolist()
    ):
        a, b = int(a_), int(b_)
        points_on_edge = edge_points.get(tuple(sorted((a, b))), [])
        ordered = points_on_edge if a < b else list(reversed(points_on_edge))
        chain = [a] + [row for _, row in ordered] + [b]
        for part, (start, end) in enumerate(zip(chain[:-1], chain[1:], strict=True)):
            segments.append((start, end))
            segment_ids.append(f"execution-{source.binding_id}-segment-{index}-{part}")
            segment_paths.append(execution_path)
    if source.noncap_refinement == "conforming":
        # The classifier's complement is deliberately conservative: every
        # unproved edge stays a hard native segment, even if the execution
        # facet plane/material test alone would call it coplanar.
        for index, edge in enumerate(complex_.plc_edge_vertices):
            a, b = (int(value) for value in edge)
            key = tuple(sorted((a, b)))
            if key in removable_edges or key in explicit_edges or key in cap_edges:
                continue
            points_on_edge = edge_points.get(key, [])
            ordered = points_on_edge if a < b else list(reversed(points_on_edge))
            chain = [a] + [row for _, row in ordered] + [b]
            for part, (start, end) in enumerate(zip(chain[:-1], chain[1:], strict=True)):
                segments.append((start, end))
                # Carrier labels share the execution facet occurrence; original
                # source identity and exact intervals are replayed from PLC roots.
                segment_ids.append(
                    f"execution-{source.binding_id}-source-edge-{index}-{part}"
                )
                segment_paths.append(execution_path)
    facet_indices = np.asarray([facet for facet, _, _ in ordered_groups], dtype=np.int64)
    execution = PiecewiseLinearComplex(
        points,
        tuple(np.asarray(triangle, dtype=np.int64) for triangle in output),
        np.asarray(polygon_facets, dtype=np.int64),
        np.asarray(complex_.facet_regions)[facet_indices],
        complex_.region_ids,
        segments=np.asarray(segments, dtype=np.int64).reshape(-1, 2),
        vertex_ids=vertex_ids,
        vertex_occurrence_paths=vertex_paths,
        facet_ids=facet_ids,
        facet_occurrence_paths=facet_paths,
        segment_ids=tuple(segment_ids),
        segment_occurrence_paths=tuple(segment_paths),
        region_occurrence_paths=complex_.region_occurrence_paths,
        boundary="conforming",
        facet_planes=None
        if complex_.facet_planes is None
        else np.asarray(complex_.facet_planes)[facet_indices],
        facet_plane_tolerances=None
        if complex_.facet_plane_tolerances is None
        else np.asarray(complex_.facet_plane_tolerances)[facet_indices],
    )
    return LayerCoreExecution(
        execution,
        np.asarray(target_rows, dtype=np.int64),
        np.asarray(output_caps, dtype=np.int64),
        tuple(stencils),
        tuple(tuple(sorted(rows)) for rows in candidates),
        seams,
        np.asarray([guard for _, guard, _ in ordered_groups], dtype=np.bool_),
    )
