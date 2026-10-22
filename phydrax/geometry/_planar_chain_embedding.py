#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Fresh exact planar-complex nonoverlap proofs, never cached verdicts."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from itertools import accumulate
from operator import index as integer_index
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from .._host_storage import immutable_host_array_layout, retained_host_storage_profile
from ._planar_primitives import prepared_signed_measure, signed_measure


if TYPE_CHECKING:
    from ._affine_chart_coverage import _PlaneKey, _Point, _Preparation, _Triangle


_IdentityKey = tuple[int, ...]


def _compare_identity_coordinates(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], preparation: _Preparation,
) -> int:
    for left, right in zip(first, second, strict=True):
        preparation.budget.reserve(1)
        if left != right:
            preparation.budget.reserve(1)
            return -1 if left < right else 1
    return 0


def _compile_source_identity_keys(
    records: tuple[tuple[_Triangle, _PlaneKey | None], ...], preparation: _Preparation,
) -> Mapping[_IdentityKey, tuple[int, ...]]:
    """Canonical key-to-row incidence, never a matching or uniqueness verdict."""
    budget = preparation.budget
    incidence: dict[_IdentityKey, list[int]] = {}
    for index, (points, plane) in enumerate(records):
        assert plane is not None
        ordered = sorted(
            points, key=cmp_to_key(lambda first, second: _compare_identity_coordinates(
                first, second, preparation,
            )),
        )
        # Flatten the already-earned numerator/denominator owners directly.
        # No old Fraction records, per-scalar pair tuples or ratio memo survive.
        budget.reserve(26)
        key = tuple(
            component for point in (*ordered, plane[0]) for value in point
            for component in (value.numerator, value.denominator)
        )
        budget.reserve(1)
        incidence.setdefault(key, []).append(index)
    budget.reserve(len(records))
    return MappingProxyType({key: tuple(rows) for key, rows in incidence.items()})


@dataclass(frozen=True, slots=True)
class PlanarEmbedding:
    complete: bool
    obligations: int
    components: int
    holes: int
    boundary_edges: int


@dataclass(frozen=True, slots=True)
class _PreparedTriangle:
    """Authenticated numeric operands, never a nonoverlap verdict."""

    corners: _Triangle
    plane: _PlaneKey | None
    projected: _Triangle
    measure: Fraction
    group: _Point


class _PlanarOperation:
    """One closed chain-proof invocation; no token survives its caller."""

    def __init__(self) -> None:
        self.pending: _PreparedPlanarOperands | None = None
        self.proved: tuple[_PreparedPlanarOperands, PlanarEmbedding] | None = None
        self.closed = False

    def close(self) -> None:
        self.pending = None
        self.proved = None
        self.closed = True


@dataclass(frozen=True, slots=True)
class _PreparedPlanarOperands:
    """One-shot capability bound to the exact preparation and ordered role."""

    operation: _PlanarOperation
    preparation: _Preparation
    rows: tuple[np.ndarray, ...]
    records: tuple[tuple[_Triangle, _PlaneKey | None], ...]
    owners: dict[int, tuple[_Triangle, _PlaneKey | None]]
    triangles: tuple[_PreparedTriangle, ...]
    groups: dict[_Point, _PreparedTriangle | list[_PreparedTriangle]] | None = None
    topology: _PlanarTopology | None = None
    source_mapping: tuple[int, ...] | None = None

    def consume(
        self,
        rows: tuple[np.ndarray, ...],
        records: tuple[tuple[_Triangle, _PlaneKey | None], ...],
        owners: dict[int, tuple[_Triangle, _PlaneKey | None]],
        preparation: _Preparation,
    ) -> tuple[_PreparedTriangle, ...]:
        pending = self.operation.pending
        closed = self.operation.closed
        # Refusal consumes the capability too. A wrong owner must not leave a
        # live token available for a subsequent correctly shaped replay.
        self.operation.close()
        if (
            pending is not self
            or closed
            or self.preparation is not preparation
            or self.rows is not rows
            or self.records is not records
            or self.owners is not owners
        ):
            raise ValueError("PlanarEmbedding.prepared must bind the current operation and ordered owners")
        return self.triangles


def _integer_bank(values: Iterable[int], maximum: int) -> np.ndarray:
    """Pack validated nonnegative IDs without an intermediate dense ndarray."""
    if type(maximum) is not int or not 0 <= maximum <= np.iinfo(np.int64).max:
        raise ValueError("PlanarEmbedding.topology integer bounds must fit int64")
    if isinstance(values, np.ndarray) and values.dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise ValueError("PlanarEmbedding.topology integer bank must have int32/int64 dtype")
    dtype = np.dtype("int32") if maximum <= np.iinfo(np.int32).max else np.dtype("int64")
    packed = array("i" if dtype.itemsize == 4 else "q")
    if packed.itemsize != dtype.itemsize:
        raise ValueError("PlanarEmbedding.topology integer bank must use canonical widths")
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("PlanarEmbedding.topology integer bank must contain integer IDs")
        try:
            current = integer_index(value)
        except TypeError as error:
            raise ValueError("PlanarEmbedding.topology integer bank must contain integer IDs") from error
        if not 0 <= current <= maximum:
            raise ValueError("PlanarEmbedding.topology integer ID exceeds validated bounds")
        packed.append(current)
    return np.frombuffer(packed.tobytes(), dtype=dtype)


@dataclass(frozen=True, slots=True)
class _IntegerRows:
    """Ragged incidence in offset/flat banks, preserving insertion order."""

    offsets: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __iter__(self) -> Iterator[np.ndarray]:
        for start, stop in zip(self.offsets[:-1], self.offsets[1:], strict=True):
            yield self.values[start:stop]

    def banks(self) -> tuple[np.ndarray, ...]:
        return self.offsets, self.values


def _integer_rows(rows: Sequence[Sequence[int]], maximum: int, *, width: int = 1) -> _IntegerRows:
    offsets = [0]
    for row in rows:
        if len(row) % width:
            raise ValueError("PlanarEmbedding.topology incidence must have complete rows")
        offsets.append(offsets[-1] + len(row) // width)
    values = _integer_bank((value for row in rows for value in row), maximum)
    if width != 1:
        values = values.reshape((-1, width))
    return _IntegerRows(_integer_bank(offsets, offsets[-1]), values)


@dataclass(frozen=True, slots=True)
class _VertexLinks:
    offsets: np.ndarray
    nodes: np.ndarray
    neighbors: _IntegerRows

    def banks(self) -> tuple[np.ndarray, ...]:
        return self.offsets, self.nodes, *self.neighbors.banks()

    def __iter__(self) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        # Only one vertex lookup is live at a time; the theorem's admitted graph
        # scratch owns it. No dense copy of the retained schedule is rebuilt.
        neighbors = self.neighbors
        for vertex, (start, stop) in enumerate(zip(self.offsets[:-1], self.offsets[1:], strict=True)):
            yield vertex, {
                int(self.nodes[node]): neighbors.values[neighbors.offsets[node]:neighbors.offsets[node + 1]]
                for node in range(int(start), int(stop))
            }


@dataclass(frozen=True, slots=True)
class _PlaneTopology:
    """Exact incidence and traversal operands, with no validity decision."""

    plane: tuple[_Point, tuple[int, ...], int]
    rows: np.ndarray
    points: tuple[_Point, ...]
    faces: np.ndarray
    edges: _IntegerRows
    links: _VertexLinks
    adjacency: _IntegerRows
    labels: np.ndarray
    components: _IntegerRows
    boundary: np.ndarray
    segments: np.ndarray
    coefficients: tuple[tuple[Fraction, int, int], ...]

    def banks(self) -> tuple[np.ndarray, ...]:
        return (
            self.rows, self.faces, *self.edges.banks(), *self.links.banks(),
            *self.adjacency.banks(), self.labels, *self.components.banks(),
            self.boundary, self.segments,
        )

    def owners(self) -> tuple[object, ...]:
        return (
            self, self.plane, self.rows, self.points, self.faces, self.edges, self.links,
            self.adjacency, self.labels, self.components, self.boundary,
            self.segments, self.coefficients,
        )


def _sort_boundary_segments(
    segments: list[tuple[int, int]],
    points: Sequence[_Point],
    preparation: _Preparation,
) -> None:
    """Canonical numeric sweep order, shared by construction and raw proof."""
    def compare(first: tuple[int, int], second: tuple[int, int]) -> int:
        preparation.budget.reserve(4)
        left = min(points[first[0]][0], points[first[1]][0])
        right = min(points[second[0]][0], points[second[1]][0])
        return (left > right) - (left < right)

    segments.sort(key=cmp_to_key(compare))


@dataclass(frozen=True, slots=True)
class _PlanarTopology:
    """Immutable source numeric schedule, never an embedding certificate."""

    row_count: int
    planes: tuple[_PlaneTopology, ...]
    identity_keys: Mapping[_IdentityKey, tuple[int, ...]]
    bindings: tuple[tuple[object, ...], ...]
    storage_bytes: int


def prepare_planar_topology(
    records: tuple[tuple[_Triangle, _PlaneKey | None], ...],
    preparation: _Preparation,
) -> _PlanarTopology | None:
    """Compile incidence once at genuine source construction, without proving it.

    Nonmanifold incidences, pinches, folds and intersecting boundaries are
    represented rather than accepted. Only the consuming fresh theorem can
    certify them. Singleton planes need no graph schedule.
    """
    budget = preparation.budget
    with budget.temporary_scope():
        budget.reserve(len(records), 4096 + 16384 * len(records))
        groups: dict[_Point, list[int]] = {}
        for index, (points, plane) in enumerate(records):
            if plane is None or len(points) != 3:
                return None
            budget.reserve(1)
            groups.setdefault(plane[0], []).append(index)
        planes = []
        for rows in groups.values():
            if len(rows) == 1:
                continue
            vertices: dict[_Point, int] = {}
            key = records[rows[0]][1]
            assert key is not None
            plane_key = key[:3]
            flat: list[_Point] = []
            faces = []
            edges: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
            links: dict[int, dict[int, list[int]]] = {}
            coefficients = [
                (value, value.numerator, value.denominator) for value in plane_key[0]
            ]
            for row in rows:
                points, plane = records[row]
                assert plane is not None
                projected = preparation.projected(points, plane[1])
                measure = preparation.signed_measure(projected)
                budget.reserve(2)
                if not measure:
                    return None
                oriented = projected if measure > 0 else (projected[0], projected[2], projected[1])
                ids = []
                for point in oriented:
                    budget.reserve(3)
                    vertex = vertices.get(point)
                    if vertex is None:
                        vertex = len(flat)
                        vertices[point] = vertex
                        flat.append(point)
                        # Only projected operands and the exact supporting
                        # plane survive; unprojected source records do not.
                        budget.reserve(1 + len(point))
                        coefficients.extend(
                            (value, value.numerator, value.denominator) for value in point
                        )
                    ids.append(vertex)
                face = len(faces)
                faces.append(tuple(ids))
                for index in range(3):
                    a, b, c = ids[index], ids[(index + 1) % 3], ids[(index + 2) % 3]
                    budget.reserve(9)
                    edges.setdefault((min(a, b), max(a, b)), []).append((a, b, face))
                    link = links.setdefault(a, {})
                    link.setdefault(b, []).append(c)
                    link.setdefault(c, []).append(b)
            adjacency: list[list[int]] = [[] for _ in faces]
            boundary = []
            for incidence in edges.values():
                budget.reserve(3)
                if len(incidence) == 2:
                    first, second = incidence[0][2], incidence[1][2]
                    adjacency[first].append(second)
                    adjacency[second].append(first)
                elif len(incidence) == 1:
                    boundary.append(incidence[0])
            labels = [-1] * len(faces)
            components = []
            for face in range(len(faces)):
                budget.reserve(1)
                if labels[face] >= 0:
                    continue
                component = []
                pending = [face]
                label = len(components)
                while pending:
                    budget.reserve(1)
                    current = pending.pop()
                    if labels[current] < 0:
                        budget.reserve(2 + len(adjacency[current]))
                        labels[current] = label
                        component.append(current)
                        pending.extend(adjacency[current])
                components.append(tuple(component))
            budget.reserve(len(boundary))
            segments = [(a, b) for a, b, _ in boundary]
            _sort_boundary_segments(segments, flat, preparation)
            # The existing construction envelope covers the mutable graph plus
            # one packed buffer and its immutable bytes snapshot. Integer payload
            # is linear in incidences, including invalid/nonmanifold schedules.
            maximum = max(len(records), len(flat), len(faces))
            link_offsets = list(accumulate((len(link) for link in links.values()), initial=0))
            edge_offsets = list(accumulate((len(incidence) for incidence in edges.values()), initial=0))
            node_neighbors = [neighbors for link in links.values() for neighbors in link.values()]
            budget.reserve(
                len(rows) + len(flat) + 3 * len(faces) + len(edge_offsets)
                + 3 * edge_offsets[-1] + len(link_offsets) + link_offsets[-1]
                + sum(map(len, node_neighbors)) + len(node_neighbors) + 1
                + len(adjacency) + 1 + sum(map(len, adjacency))
                + len(labels) + len(components) + 1 + sum(map(len, components))
                + 3 * len(boundary) + 2 * len(segments) + len(coefficients),
            )
            planes.append(_PlaneTopology(
                plane_key, _integer_bank(rows, len(records)), tuple(flat),
                _integer_bank((vertex for face in faces for vertex in face), maximum).reshape((-1, 3)),
                _IntegerRows(
                    _integer_bank(edge_offsets, edge_offsets[-1]),
                    _integer_bank(
                        (value for incidence in edges.values() for edge in incidence for value in edge),
                        maximum,
                    ).reshape((-1, 3)),
                ),
                _VertexLinks(
                    _integer_bank(link_offsets, link_offsets[-1]),
                    _integer_bank((node for link in links.values() for node in link), maximum),
                    _integer_rows(node_neighbors, maximum),
                ),
                _integer_rows(adjacency, maximum), _integer_bank(labels, len(components)),
                _integer_rows(components, maximum),
                _integer_bank((value for edge in boundary for value in edge), maximum).reshape((-1, 3)),
                _integer_bank((value for edge in segments for value in edge), maximum).reshape((-1, 2)),
                tuple(coefficients),
            ))
            # Do not carry the last plane's mutable graph into profiling or the
            # subsequent target preparation. Canonical arrays are the sole IDs.
            del vertices, flat, faces, edges, links, adjacency, labels, components
            del boundary, segments, coefficients, node_neighbors
            del link_offsets, edge_offsets, ids, link, incidence, pending, component
        budget.reserve(2 * len(planes) + 1)
        numeric_planes = tuple(planes)
        del groups, planes
        row_count = len(records)
        identity_keys = _compile_source_identity_keys(records, preparation)
        bindings = ((row_count, numeric_planes, identity_keys),) + tuple(
            plane.owners() + tuple((bank, immutable_host_array_layout(bank)) for bank in plane.banks())
            for plane in numeric_planes
        )
        _, storage, _, _ = retained_host_storage_profile(
            (row_count, numeric_planes, identity_keys, bindings), {}, lambda _buffer: False,
            admit_owner=lambda _owner: budget.reserve(1),
        )
        storage += 1024
        bindings = ((row_count, numeric_planes, identity_keys, storage),) + bindings[1:]
        return _PlanarTopology(row_count, numeric_planes, identity_keys, bindings, storage)


def _import_planar_topology(
    token: _PreparedPlanarOperands,
    preparation: _Preparation,
) -> dict[_Point, tuple[_PlaneTopology, list[_PreparedTriangle]]]:
    """Bind compiled source incidence to the current complete target bijection."""
    topology, mapping = token.topology, token.source_mapping
    if topology is None or mapping is None:
        raise ValueError("PlanarEmbedding.topology must bind the current source permutation")
    budget = preparation.budget
    count = len(token.rows)
    budget.reserve(2 * count, 4096 + 1024 * count)
    if (
        topology.row_count != count or len(mapping) != count
        or len(token.triangles) != count or len(token.records) != count
        or set(mapping) != set(range(count))
        or len(topology.bindings) != len(topology.planes) + 1
        or len(topology.bindings[0]) != 4
        or topology.bindings[0][0] is not topology.row_count
        or topology.bindings[0][1] is not topology.planes
        or topology.bindings[0][2] is not topology.identity_keys
        or topology.bindings[0][3] is not topology.storage_bytes
    ):
        raise ValueError("PlanarEmbedding.topology must bind complete unchanged numeric owners")
    budget.reserve(0, topology.storage_bytes)
    inverse = [0] * count
    for target, source in enumerate(mapping):
        inverse[source] = target
    result = {}
    for plane, binding in zip(topology.planes, topology.bindings[1:], strict=True):
        owners = plane.owners()
        budget.reserve(len(owners) + 2 * len(plane.coefficients))
        if (
            len(binding) < len(owners)
            or any(first is not second for first, second in zip(owners, binding[:len(owners)], strict=True))
        ):
            raise ValueError("PlanarEmbedding.topology must bind unchanged incidence owners")
        if len(binding) != len(owners) + len(plane.banks()):
            raise ValueError("PlanarEmbedding.topology must bind complete integer bank owners")
        for bank, bound in zip(plane.banks(), binding[len(owners):], strict=True):
            budget.reserve(2)
            if (
                not isinstance(bound, tuple) or len(bound) != 2 or bank is not bound[0]
                or bank.dtype not in (np.dtype("int32"), np.dtype("int64"))
                or immutable_host_array_layout(bank) != bound[1]
            ):
                raise ValueError("PlanarEmbedding.topology must bind unchanged integer bank owners")
        if any(
            value.numerator is not numerator or value.denominator is not denominator
            for value, numerator, denominator in plane.coefficients
        ):
            raise ValueError("PlanarEmbedding.topology must bind unchanged coefficient owners")
        triangles = []
        for local, source in enumerate(plane.rows):
            target = inverse[source]
            row, record, operand = token.rows[target], token.records[target], token.triangles[target]
            source_plane = plane.plane
            budget.reserve(12)
            if (
                token.owners.get(id(row)) is not record or record[0] is not operand.corners
                or record[1] is not operand.plane or row.shape != (3, 3)
                or operand.plane is None
                or operand.plane[:3] != source_plane
            ):
                raise ValueError("PlanarEmbedding.topology must bind current row and plane owners")
            for point, current in zip(operand.corners, row, strict=True):
                for value, coordinate in zip(point, current, strict=True):
                    budget.reserve(1)
                    if value != float(coordinate):
                        raise ValueError("PlanarEmbedding.topology must bind unchanged row owners")
            oriented = operand.projected if operand.measure > 0 else (
                operand.projected[0], operand.projected[2], operand.projected[1],
            )
            face = plane.faces[local]
            matched = False
            for offset in range(3):
                budget.reserve(2)
                if oriented[0] != plane.points[face[offset]]:
                    continue
                budget.reserve(4)
                matched = (
                    oriented[1] == plane.points[face[(offset + 1) % 3]]
                    and oriented[2] == plane.points[face[(offset + 2) % 3]]
                )
                break
            if not matched:
                raise ValueError("PlanarEmbedding.topology must preserve oriented vertex incidence")
            budget.reserve(1)
            triangles.append(operand)
        budget.reserve(1)
        result[triangles[0].group] = plane, triangles
    return result


def _group_prepared_triangle(
    groups: dict[_Point, _PreparedTriangle | list[_PreparedTriangle]],
    triangle: _PreparedTriangle,
    preparation: _Preparation,
) -> None:
    """Retain the first member directly; allocate lists only for real groups.

    The caller admits the canonical-plane lookup. Only a repeated plane needs
    a second membership insertion, admitted here before promotion or append.
    """
    previous = groups.setdefault(triangle.group, triangle)
    if previous is not triangle:
        preparation.budget.reserve(1)
        if isinstance(previous, _PreparedTriangle):
            groups[triangle.group] = [previous, triangle]
        else:
            previous.append(triangle)


class _Proof:
    def __init__(self, preparation: _Preparation, capacity: int, bits: int) -> None:
        self.preparation = preparation
        self.capacity = capacity
        self.bits = bits
        self.obligations = 0
        self.components = 0
        self.holes = 0
        self.boundary_edges = 0

    def obligation(self) -> bool:
        if self.obligations >= self.capacity:
            return False
        self.preparation.budget.reserve(1)
        self.obligations += 1
        return True

    def orientation(self, a: _Point, b: _Point, c: _Point) -> Fraction:
        with self.preparation.budget.temporary_scope():
            return self.preparation.stage(
                13, 16, 16 * self.bits + 64,
                lambda: (prepared_signed_measure((a, b, c)), 13),
            )

    def simplex(self, measure: Fraction) -> bool:
        """A nonzero affine simplex is an embedded disk without graph work."""
        if not self.obligation() or not measure:
            return False
        self.components += 1
        self.boundary_edges += 3
        return True

    def plane(
        self,
        triangles: list[_Triangle] | list[_PreparedTriangle],
        *,
        topology: _PlaneTopology | None = None,
    ) -> bool:
        budget = self.preparation.budget
        if len(triangles) == 1:
            triangle = triangles[0]
            return self.simplex(
                triangle.measure if isinstance(triangle, _PreparedTriangle)
                else self.preparation.signed_measure(triangle),
            )
        # Every coordinate is already authenticated and retained by preparation.
        # This bound covers incidence dictionaries, links, DFS, cycles, sorting
        # scratch and IDs; arithmetic temporaries have their own admitted bound.
        budget.reserve(0, 4096 + 12288 * len(triangles))
        if topology is None:
            vertices: dict[_Point, int] = {}
            raw_points: list[_Point] = []
            raw_faces: list[tuple[int, ...]] = []
            raw_edges: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
            raw_links: dict[int, dict[int, list[int]]] = {}
            for operand in triangles:
                if not self.obligation():
                    return False
                triangle = operand.projected if isinstance(operand, _PreparedTriangle) else operand
                measure = operand.measure if isinstance(operand, _PreparedTriangle) else self.preparation.signed_measure(triangle)
                budget.reserve(2)
                if not measure:
                    return False
                oriented = triangle if measure > 0 else (triangle[0], triangle[2], triangle[1])
                ids = []
                for point in oriented:
                    budget.reserve(3)
                    vertex = vertices.get(point)
                    if vertex is None:
                        vertex = len(raw_points)
                        vertices[point] = vertex
                        raw_points.append(point)
                    ids.append(vertex)
                face = len(raw_faces)
                raw_faces.append(tuple(ids))
                for index in range(3):
                    a, b, c = ids[index], ids[(index + 1) % 3], ids[(index + 2) % 3]
                    budget.reserve(9)
                    edge = (min(a, b), max(a, b))
                    raw_edges.setdefault(edge, []).append((a, b, face))
                    link = raw_links.setdefault(a, {})
                    link.setdefault(b, []).append(c)
                    link.setdefault(c, []).append(b)
            points, faces = raw_points, raw_faces
            edge_incidence = raw_edges.values()
            vertex_links = raw_links.items()
        else:
            # The importer has reauthenticated every scheduled oriented face.
            # Incidence is numeric input; every scientific check below is fresh.
            points, faces = topology.points, topology.faces
            edge_incidence = topology.edges
            budget.reserve(len(topology.links.nodes))
            vertex_links = topology.links
            for operand in triangles:
                if not self.obligation():
                    return False
                budget.reserve(2)
                if not isinstance(operand, _PreparedTriangle) or not operand.measure:
                    return False
        adjacency: list[list[int]] = [[] for _ in faces] if topology is None else []
        boundary: dict[int, tuple[int, int]] = {}
        incoming: set[int] = set()
        for incidence in edge_incidence:
            if not self.obligation():
                return False
            budget.reserve(8)
            if len(incidence) == 2:
                a, b, first = incidence[0]
                c, d, second = incidence[1]
                if a != d or b != c or first == second:
                    return False
                if topology is None:
                    adjacency[first].append(second)
                    adjacency[second].append(first)
            elif len(incidence) == 1:
                a, b, face = incidence[0]
                if a in boundary or b in incoming:
                    return False
                boundary[a] = b, face
                incoming.add(b)
            else:
                return False
        budget.reserve(len(boundary) + len(incoming))
        if not boundary or boundary.keys() != incoming:
            return False
        # A manifold vertex link is one circle (interior) or one interval
        # (boundary). Edge incidence alone would admit pinched vertices.
        for vertex, link in vertex_links:
            if not self.obligation():
                return False
            endpoints = 0
            for neighbors in link.values():
                budget.reserve(2)
                if len(neighbors) == 1:
                    endpoints += 1
                elif len(neighbors) != 2:
                    return False
            budget.reserve(1)
            if endpoints != (2 if vertex in boundary else 0):
                return False
            reached: set[int] = set()
            pending = [next(iter(link))]
            while pending:
                budget.reserve(1)
                node = pending.pop()
                if node not in reached:
                    budget.reserve(1 + len(link[node]))
                    reached.add(node)
                    pending.extend(link[node])
            if len(reached) != len(link):
                return False
        if topology is None:
            raw_labels = [-1] * len(faces)
            raw_components: list[list[int]] = []
            for face in range(len(faces)):
                budget.reserve(1)
                if raw_labels[face] >= 0:
                    continue
                members: list[int] = []
                pending = [face]
                label = len(raw_components)
                while pending:
                    budget.reserve(1)
                    current = pending.pop()
                    if raw_labels[current] < 0:
                        budget.reserve(2 + len(adjacency[current]))
                        raw_labels[current] = label
                        members.append(current)
                        pending.extend(adjacency[current])
                raw_components.append(members)
            labels, components = raw_labels, raw_components
        else:
            labels, components = topology.labels, topology.components
        cycles: list[tuple[list[int], int, int]] = []
        unseen = set(boundary)
        while unseen:
            if not self.obligation():
                return False
            start = next(iter(unseen))
            cycle: list[int] = []
            current = start
            label = labels[boundary[start][1]]
            while current in unseen:
                budget.reserve(5)
                unseen.remove(current)
                cycle.append(current)
                following, face = boundary[current]
                if labels[face] != label:
                    return False
                current = following
            if current != start or len(cycle) < 3:
                return False
            polygon = tuple(points[vertex] for vertex in cycle)
            with budget.temporary_scope():
                measure = signed_measure(polygon, budget=budget)
                budget.reserve(2)
                if not measure:
                    return False
                cycles.append((cycle, label, 1 if measure > 0 else -1))
        for label, component in enumerate(components):
            if not self.obligation():
                return False
            component_vertices: set[int] = set()
            edge_count = 0
            for face in component:
                budget.reserve(6)
                component_vertices.update(faces[face])
                edge_count += 3
            signs = []
            for cycle, owner, sign in cycles:
                budget.reserve(1)
                if owner == label:
                    budget.reserve(1)
                    edge_count += len(cycle)
                    signs.append(sign)
            budget.reserve(len(signs) + 4)
            if signs.count(1) != 1 or len(component_vertices) - edge_count // 2 + len(component) != 2 - len(signs):
                return False
        if topology is None:
            raw_segments = [(a, boundary[a][0]) for a in boundary]
            _sort_boundary_segments(raw_segments, points, self.preparation)
            segments = raw_segments
        else:
            segments = topology.segments
        for index, (a, b) in enumerate(segments):
            pa, pb = points[a], points[b]
            for other in range(index + 1, len(segments)):
                c, d = segments[other]
                pc, pd = points[c], points[d]
                budget.reserve(6)
                if min(pc[0], pd[0]) > max(pa[0], pb[0]):
                    break
                if max(pc[1], pd[1]) < min(pa[1], pb[1]) or max(pa[1], pb[1]) < min(pc[1], pd[1]):
                    continue
                if not self.obligation():
                    return False
                if a == d or b == c:
                    shared, first, second = (pa, pb, pc) if a == d else (pb, pa, pd)
                    if self.orientation(shared, first, second) == 0:
                        # Collinear adjacent segments may continue, not retrace.
                        budget.reserve(6)
                        axis = 0 if first[0] != shared[0] else 1
                        if (first[axis] > shared[axis]) == (second[axis] > shared[axis]):
                            return False
                    continue
                ab_c, ab_d = self.orientation(pa, pb, pc), self.orientation(pa, pb, pd)
                cd_a, cd_b = self.orientation(pc, pd, pa), self.orientation(pc, pd, pb)
                budget.reserve(8)
                if (ab_c <= 0 <= ab_d or ab_d <= 0 <= ab_c) and (cd_a <= 0 <= cd_b or cd_b <= 0 <= cd_a):
                    return False
        # Disjoint simple cycles have constant integer winding on each other's
        # images. Positive cycles start in degree zero; negative cycles start
        # in degree one. This rejects nested filled components/holes and permits
        # a separate component inside a genuine hole.
        for index, (cycle, label, sign) in enumerate(cycles):
            point = points[cycle[0]]
            surrounding = 0
            own_outer = 0
            for other, (loop, owner, other_sign) in enumerate(cycles):
                if index == other:
                    continue
                if not self.obligation():
                    return False
                winding = 0
                for offset, a in enumerate(loop):
                    pa, pb = points[a], points[loop[(offset + 1) % len(loop)]]
                    budget.reserve(4)
                    if pa[1] <= point[1] < pb[1]:
                        winding += self.orientation(pa, pb, point) > 0
                    elif pb[1] <= point[1] < pa[1]:
                        winding -= self.orientation(pa, pb, point) < 0
                surrounding += winding
                if owner == label and other_sign == 1:
                    own_outer += winding
            budget.reserve(3)
            if surrounding != (0 if sign == 1 else 1) or (sign == -1 and own_outer != 1):
                return False
        self.components += len(components)
        self.holes += sum(sign == -1 for _, _, sign in cycles)
        self.boundary_edges += len(boundary)
        return True


def prove_planar_embedding(
    rows: tuple[np.ndarray, ...],
    records: tuple[tuple[_Triangle, _PlaneKey | None], ...],
    owners: dict[int, tuple[_Triangle, _PlaneKey | None]],
    preparation: _Preparation,
    *,
    capacity: int,
    point_bits: int,
) -> PlanarEmbedding:
    """Authenticate general operands and prove a fresh planar embedding."""
    return _prove_planar_embedding(
        rows, records, owners, preparation, capacity=capacity, point_bits=point_bits,
    )


def _prove_planar_embedding(
    rows: tuple[np.ndarray, ...],
    records: tuple[tuple[_Triangle, _PlaneKey | None], ...],
    owners: dict[int, tuple[_Triangle, _PlaneKey | None]],
    preparation: _Preparation,
    *,
    capacity: int,
    point_bits: int,
    _prepared: _PreparedPlanarOperands | None = None,
) -> PlanarEmbedding:
    """Prove positive-area nonoverlap, not coverage or physical fidelity.

    An oriented PL manifold with positive triangle Jacobians has multiplicity
    equal to its boundary winding off its edges. The admitted Jordan boundaries
    have degree zero or one. Thus no two open triangles overlap. Vertex links,
    Euler data and boundary injectivity additionally exclude pinches and folds.
    Different exact planes have no positive-area intersection. Existing chain
    tiling must still prove that this boundary is the required source boundary.
    """
    proof = _Proof(preparation, capacity, point_bits)
    budget = preparation.budget
    with budget.temporary_scope():
        operands = None if _prepared is None else _prepared.consume(rows, records, owners, preparation)
        budget.reserve(len(rows), 2048 + 1024 * len(rows))
        groups: dict[_Point, list[_Triangle]] = {}
        prepared_groups = None if _prepared is None else _prepared.groups
        complete = bool(rows) and len(rows) == len(records)
        if operands is not None:
            # Issuance and consumption are adjacent inside the closed chain
            # operation. No caller can mutate a Fraction/ndarray owner between
            # them. Numeric authentication and plane membership are reused;
            # topology, Jacobians, links and Jordan obligations remain fresh.
            complete = complete and len(operands) == len(rows)
            if complete and prepared_groups is None:
                prepared_groups = {}
                for operand in operands:
                    if operand.plane is None:
                        complete = False
                        break
                    budget.reserve(1)
                    _group_prepared_triangle(prepared_groups, operand, preparation)
        elif complete:
            for row, record in zip(rows, records, strict=True):
                points, plane = record
                budget.reserve(1)
                if owners.get(id(row)) is not record:
                    budget.reserve(17)
                    if owners.get(id(row)) != record:
                        raise ValueError("PlanarEmbedding.records must bind current row owners")
                if plane is None:
                    complete = False
                    break
                budget.reserve(4 + sum(map(len, points)) + len(plane[0]))
                if preparation._point_profile(points) is None or preparation._point_profile(plane[0]) is None:
                    complete = False
                    break
                if row.shape != (3, 3) or len(points) != 3 or any(len(point) != 3 for point in points):
                    complete = False
                    break
                for point, current in zip(points, row, strict=True):
                    for value, coordinate in zip(point, current, strict=True):
                        budget.reserve(1)
                        if value != float(coordinate):
                            raise ValueError("PlanarEmbedding.rows must bind unchanged row owners")
                budget.reserve(5)
                groups.setdefault(plane[0], []).append(preparation.projected(points, plane[1]))
        if complete and operands is not None:
            assert prepared_groups is not None
            schedules = (
                _import_planar_topology(_prepared, preparation)
                if _prepared is not None and _prepared.topology is not None else {}
            )
            for triangles in prepared_groups.values():
                with budget.temporary_scope():
                    if isinstance(triangles, _PreparedTriangle):
                        admitted = proof.simplex(triangles.measure)
                    else:
                        scheduled = schedules.get(triangles[0].group)
                        admitted = proof.plane(triangles) if scheduled is None else proof.plane(
                            scheduled[1], topology=scheduled[0],
                        )
                    if not admitted:
                        complete = False
                        break
        elif complete:
            for triangles in groups.values():
                with budget.temporary_scope():
                    if not proof.plane(triangles):
                        complete = False
                        break
    result = PlanarEmbedding(complete, proof.obligations, proof.components, proof.holes, proof.boundary_edges)
    if _prepared is not None:
        _prepared.operation.proved = (_prepared, result)
    return result


def _transport_planar_embedding(
    rows: tuple[np.ndarray, ...],
    records: tuple[tuple[_Triangle, _PlaneKey | None], ...],
    owners: dict[int, tuple[_Triangle, _PlaneKey | None]],
    preparation: _Preparation,
    *,
    capacity: int,
    source: _PreparedPlanarOperands,
    target: _PreparedPlanarOperands,
    mapping: tuple[int, ...],
    evidence: PlanarEmbedding,
) -> PlanarEmbedding:
    """Transport a fresh theorem through a complete simplicial isomorphism.

    All six triangle vertex permutations preserve its unoriented incidence.
    Matching exact corners on the same canonical plane therefore transports
    edges, vertex links, components and Jordan boundaries, not merely areas.
    The source consumes its own ordered-role capability and rechecks current
    owners. No topology decision survives the enclosing operation.
    """
    operands = source.consume(rows, records, owners, preparation)
    proved = target.operation.proved
    if (
        proved is None or proved[0] is not target or proved[1] is not evidence
        or not evidence.complete or target.preparation is not preparation
        or target.owners is not owners or target is source
    ):
        raise ValueError("PlanarEmbedding.transport must bind a fresh target theorem and source role")
    budget = preparation.budget
    with budget.temporary_scope():
        count = len(rows)
        # Both permutation sets, range integers and their resize scratch
        # coexist with the constant-size per-triangle incidence map below.
        budget.reserve(2 * count, 1024 + 512 * count)
        if (
            not count or len(mapping) != count or len(operands) != count
            or len(records) != count or len(target.rows) != count
            or len(target.records) != count or len(target.triangles) != count
            or set(mapping) != set(range(count))
        ):
            raise ValueError("PlanarEmbedding.transport must bind a complete permutation")
        if capacity < count:
            return PlanarEmbedding(False, 0, 0, 0, 0)
        for first, second in enumerate(mapping):
            left, right = target.triangles[first], operands[second]
            target_record = target.records[first]
            budget.reserve(3)
            if (
                owners.get(id(target.rows[first])) is not target_record
                or target_record[0] is not left.corners or target_record[1] is not left.plane
            ):
                raise ValueError("PlanarEmbedding.transport must bind fresh target record owners")
            # The just-completed target theorem owns these immutable exact
            # operands. Reauthenticate the complete current source role, not
            # the target's already proved arithmetic a second time.
            row, record = rows[second], records[second]
            budget.reserve(5)
            if (
                owners.get(id(row)) is not record or record[0] is not right.corners
                or record[1] is not right.plane or row.shape != (3, 3)
                or right.plane is None
            ):
                raise ValueError("PlanarEmbedding.transport must bind current record owners")
            budget.reserve(16)
            if (
                preparation._point_profile(right.corners) is None
                or preparation._point_profile(right.plane[0]) is None
            ):
                raise ValueError("PlanarEmbedding.transport must bind current exact owners")
            for point, current in zip(right.corners, row, strict=True):
                for value, coordinate in zip(point, current, strict=True):
                    budget.reserve(1)
                    if value != float(coordinate):
                        raise ValueError("PlanarEmbedding.transport must bind unchanged row owners")
            budget.reserve(16)
            if (
                left.plane is None or right.plane is None
                or left.plane[:3] != right.plane[:3] or left.group != right.group
                or len(left.corners) != 3 or len(right.corners) != 3
                or len(left.projected) != 3 or len(right.projected) != 3
            ):
                raise ValueError("PlanarEmbedding.transport must preserve plane incidence")
            # Unique corners and the complete triangle permutation make this
            # an exact one-to-one incidence map, including shared vertices.
            permutation = []
            for index, point in enumerate(right.corners):
                for vertex, candidate in enumerate(left.corners):
                    budget.reserve(1)
                    same = point is candidate
                    if not same:
                        budget.reserve(3)
                        same = point == candidate
                    if same:
                        budget.reserve(3)
                        if right.projected[index] != left.projected[vertex]:
                            raise ValueError("PlanarEmbedding.transport must preserve vertex incidence")
                        permutation.append(vertex)
                        break
            budget.reserve(3)
            if set(permutation) != {0, 1, 2}:
                raise ValueError("PlanarEmbedding.transport must preserve vertex incidence")
        return PlanarEmbedding(
            True, count, evidence.components, evidence.holes, evidence.boundary_edges,
        )
