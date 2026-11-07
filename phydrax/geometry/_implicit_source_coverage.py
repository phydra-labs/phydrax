#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent implicit source coverage and root-ancestry proof."""

from __future__ import annotations

import math
from collections.abc import Iterator
from fractions import Fraction
from typing import Any, NoReturn

import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ._approximate_source_coverage import (
    _fraction,
    _interval_dot,
    _strict_sign,
    SourceCoverageUnavailable,
)
from ._source_boundary_records import SourceBoundaryChartWitness
from .implicit._adaptive_discovery import _FieldEnclosures, AdaptiveImplicitSurface


def _integer_rows(rows: np.ndarray, /) -> np.ndarray:
    """View fixed-width integer tuples as lexicographically ordered records."""
    values = np.ascontiguousarray(rows, dtype=np.int64)
    width = values.shape[-1]
    dtype = np.dtype([(f"f{axis}", np.int64) for axis in range(width)])
    return values.reshape(-1, width).view(dtype).reshape(-1)


def _same_bits(first: np.ndarray, second: np.ndarray, /) -> bool:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    if first.dtype == np.dtype(np.float64):
        return bool(np.array_equal(first.view(np.uint64), second.view(np.uint64)))
    return bool(np.array_equal(first, second))


def _validate_oriented_implicit_triangles(faces: np.ndarray, /) -> None:
    """Check an oriented closed chain using compact sorted integer records."""
    face_rows = np.asarray(faces)
    if face_rows.ndim != 2 or face_rows.shape[1] != 3:
        raise SourceCoverageUnavailable(
            "Implicit source boundary is not triangular.", status="unresolved"
        )
    compact = (
        face_rows.size == 0
        or (np.min(face_rows) >= 0 and np.max(face_rows) <= np.iinfo(np.uint32).max)
    )
    index_dtype = np.dtype(np.uint32 if compact else np.int64)
    triangles = np.empty(face_rows.shape, dtype=index_dtype)
    for start in range(0, len(face_rows), 65536):
        part = triangles[start : start + 65536]
        part[:] = face_rows[start : start + 65536]
        part.sort(axis=1)
    if np.any(triangles[:, 1:] == triangles[:, :-1]):
        raise SourceCoverageUnavailable(
            "Implicit source boundary repeats or degenerates a triangle.",
            status="violated",
        )
    triangle_dtype = np.dtype([(f"f{axis}", index_dtype) for axis in range(3)])
    triangle_keys = triangles.view(triangle_dtype).reshape(-1)
    triangle_keys.sort()
    if np.any(triangle_keys[1:] == triangle_keys[:-1]):
        raise SourceCoverageUnavailable(
            "Implicit source boundary repeats or degenerates a triangle.",
            status="violated",
        )
    del triangles, triangle_keys

    count = len(face_rows)
    edges = np.empty(3 * count, dtype=[("low", index_dtype), ("high", index_dtype), ("sign", "i1")])
    for axis in range(3):
        for start in range(0, count, 65536):
            stop = min(start + 65536, count)
            first = face_rows[start:stop, axis]
            second = face_rows[start:stop, (axis + 1) % 3]
            part = edges[axis * count + start : axis * count + stop]
            part["low"] = np.minimum(first, second)
            part["high"] = np.maximum(first, second)
            part["sign"] = np.where(first < second, 1, -1)
    edges.sort(order=("low", "high", "sign"))
    if (
        not len(edges)
        or len(edges) % 2
        or np.any(edges["low"][0::2] != edges["low"][1::2])
        or np.any(edges["high"][0::2] != edges["high"][1::2])
        or np.any(edges["sign"][0::2] != -1)
        or np.any(edges["sign"][1::2] != 1)
        or np.any(
            (edges["low"][1:-1:2] == edges["low"][2::2])
            & (edges["high"][1:-1:2] == edges["high"][2::2])
        )
    ):
        raise SourceCoverageUnavailable(
            "Implicit source root triangles are not a closed oriented 2-chain.",
            status="violated",
        )


class _SortedIntegerKeys:
    """Compact lookup for canonical lexicographically sorted integer tuples."""

    def __init__(self, rows: np.ndarray) -> None:
        self.rows = np.asarray(rows, dtype=np.int64).reshape(len(rows), math.prod(rows.shape[1:]))
        self.keys = _integer_rows(self.rows)

    def get(self, key: Any, default: Any = None) -> Any:
        probe = _integer_rows(np.asarray(key, dtype=np.int64).reshape(1, -1))
        row = int(np.searchsorted(self.keys, probe)[0])
        return row if row < len(self.keys) and self.keys[row] == probe[0] else default

    def __contains__(self, key: Any) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            row = self.rows[key].reshape(-1, 3)
            return tuple(tuple(int(value) for value in point) for point in row)
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __iter__(self) -> Iterator[Any]:
        return (self[row] for row in range(len(self.rows)))


class _SourceTriangles:
    """Exact coordinate-ranked triangle keys and source row ancestry."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        index_dtype = np.uint32 if max(len(vertices), len(faces)) <= np.iinfo(np.uint32).max else np.uint64
        self.vertices = vertices
        self.vertex_order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0])).astype(
            index_dtype, copy=False
        )
        vertex_ranks = np.empty(len(vertices), dtype=index_dtype)
        previous_rank = 0
        for start in range(0, len(vertices), 65536):
            stop = min(start + 65536, len(vertices))
            rows = self.vertex_order[start:stop]
            ranks = np.arange(start, stop, dtype=index_dtype)
            if start:
                previous = vertices[self.vertex_order[start - 1:stop - 1]]
                equal = np.all(vertices[rows] == previous, axis=1)
                ranks[equal] = previous_rank
            elif stop > 1:
                equal = np.all(vertices[rows[1:]] == vertices[rows[:-1]], axis=1)
                ranks[1:][equal] = 0
            np.maximum.accumulate(ranks, out=ranks)
            vertex_ranks[rows] = ranks
            previous_rank = int(ranks[-1])

        self.records = np.empty(
            len(faces), dtype=[("a", index_dtype), ("b", index_dtype),
                               ("c", index_dtype), ("row", index_dtype)]
        )
        for start in range(0, len(faces), 4096):
            stop = min(start + 4096, len(faces))
            keys = np.sort(vertex_ranks[faces[start:stop]], axis=1)
            for axis, name in enumerate(("a", "b", "c")):
                self.records[name][start:stop] = keys[:, axis]
            self.records["row"][start:stop] = np.arange(start, stop, dtype=index_dtype)
        del vertex_ranks
        self.records.sort(order=("a", "b", "c", "row"))
        for start in range(0, max(0, len(faces) - 1), 65536):
            stop = min(start + 65536, len(faces) - 1)
            duplicate = np.ones(stop - start, dtype=np.bool_)
            for name in ("a", "b", "c"):
                duplicate &= self.records[name][start:stop] == self.records[name][start + 1:stop + 1]
            for offset in np.flatnonzero(duplicate):
                row = int(self.records["row"][start + offset])
                # The former coordinate comparison never equated NaNs.
                if not np.isnan(vertices[faces[row]]).any():
                    raise SourceCoverageUnavailable(
                        "Implicit source scaffold repeats a target boundary triangle.",
                        status="violated",
                    )

    def get(self, triangle: Any, default: Any = None) -> Any:
        points = sorted(triangle)
        try:
            values = [float(value) for point in points for value in point]
        except (OverflowError, ValueError):
            return default
        if any(not math.isfinite(value) or Fraction(value) != exact
               for value, exact in zip(values, (x for point in points for x in point), strict=True)):
            return default
        ranks = []
        for offset in range(0, 9, 3):
            target = values[offset:offset + 3]
            low, high = 0, len(self.vertex_order)
            while low < high:
                middle = (low + high) // 2
                point = self.vertices[self.vertex_order[middle]]
                # NumPy's lexsort places NaNs after every non-NaN value.
                before = False
                for value, wanted in zip(point, target, strict=True):
                    if math.isnan(value) or value > wanted:
                        break
                    if value < wanted:
                        before = True
                        break
                if before:
                    low = middle + 1
                else:
                    high = middle
            if low == len(self.vertex_order) or not np.array_equal(
                self.vertices[self.vertex_order[low]], target
            ):
                return default
            ranks.append(low)
        probe = np.zeros(1, dtype=self.records.dtype)
        for name, value in zip(("a", "b", "c"), sorted(ranks), strict=True):
            probe[name] = value
        row = int(np.searchsorted(self.records, probe)[0])
        if row >= len(self.records) or any(
            self.records[name][row] != value
            for name, value in zip(("a", "b", "c"), sorted(ranks), strict=True)
        ):
            return default
        return int(self.records["row"][row])


class _DyadicPartition:
    """Exact Morton-interval tiling and outward-box containment proof."""

    def __init__(self, domain: np.ndarray, target_level: int, count: int) -> None:
        if target_level < 0 or count < 0:
            self._fail()
        self.target_level = target_level
        self.position = 0
        self.width = max(1, (3 * target_level + 8) // 8)
        self.intervals = np.empty(
            count, dtype=[("start", f"V{self.width}"), ("stop", f"V{self.width}")]
        )
        self.domain = []
        for axis in range(3):
            lower_num, lower_den = float(domain[0, axis]).as_integer_ratio()
            upper_num, upper_den = float(domain[1, axis]).as_integer_ratio()
            exponent = max(lower_den.bit_length(), upper_den.bit_length()) - 1
            lower_num <<= exponent - (lower_den.bit_length() - 1)
            upper_num <<= exponent - (upper_den.bit_length() - 1)
            self.domain.append((lower_num, upper_num - lower_num, exponent))

    @staticmethod
    def _fail() -> None:
        raise SourceCoverageUnavailable(
            "Implicit interval leaves do not form an exhaustive dyadic partition.",
            status="violated",
        )

    def add(self, levels: np.ndarray, indices: np.ndarray, boxes: np.ndarray) -> None:
        for level_, index_, box in zip(levels, indices, boxes, strict=True):
            level = int(level_)
            index = tuple(int(value) for value in index_)
            if (level < 0 or level > self.target_level
                or any(value < 0 or value >= (1 << level) for value in index)
                or self.position >= len(self.intervals)):
                self._fail()
            for axis, coordinate in enumerate(index):
                lower_num, extent_num, domain_exponent = self.domain[axis]
                exact_exponent = domain_exponent + level
                for side in (0, 1):
                    value = float(box[side, axis])
                    if not math.isfinite(value):
                        self._fail()
                    numerator, denominator = value.as_integer_ratio()
                    exponent = denominator.bit_length() - 1
                    exact = (lower_num << level) + (coordinate + side) * extent_num
                    shift = max(exponent, exact_exponent)
                    observed = numerator << (shift - exponent)
                    expected = exact << (shift - exact_exponent)
                    if (side == 0 and observed > expected) or (side == 1 and observed < expected):
                        self._fail()
            morton = 0
            for bit in range(level - 1, -1, -1):
                morton = (morton << 3) | (((index[0] >> bit) & 1) << 2) | (
                    ((index[1] >> bit) & 1) << 1
                ) | ((index[2] >> bit) & 1)
            remaining = 3 * (self.target_level - level)
            first = morton << remaining
            last = first + (1 << remaining)
            self.intervals["start"][self.position] = first.to_bytes(self.width, "big")
            self.intervals["stop"][self.position] = last.to_bytes(self.width, "big")
            self.position += 1

    def finish(self) -> None:
        if self.position != len(self.intervals) or not self.position:
            self._fail()
        self.intervals.sort(order="start")
        zero = np.void(bytes(self.width))
        end = np.void((1 << (3 * self.target_level)).to_bytes(self.width, "big"))
        if (
            self.intervals["start"][0] != zero
            or self.intervals["stop"][-1] != end
            or np.any(self.intervals["stop"][:-1] != self.intervals["start"][1:])
        ):
            self._fail()


def _check_leaf_replay(surface: AdaptiveImplicitSurface, bounds: _FieldEnclosures, /) -> None:
    """Replay every octree leaf against both retained cover and root cells."""
    from .implicit._adaptive_packed_frontier import discover_leaves

    cover = surface.cover
    scaffold = surface.scaffold
    if cover is None:
        raise SourceCoverageUnavailable(
            "Implicit discovery does not cold-reproduce its retained roots and intervals.",
            status="violated",
        )
    leaf_row = 0
    cell_row = 0
    partition = _DyadicPartition(
        np.asarray(scaffold.domain), int(scaffold.target_level), len(cover.boxes)
    )

    same = _same_bits

    def fail() -> None:
        raise SourceCoverageUnavailable(
            "Implicit discovery does not cold-reproduce its retained roots and intervals.",
            status="violated",
        )

    def consume(
        level: np.ndarray, index: np.ndarray, boxes: np.ndarray,
        lower: np.ndarray, upper: np.ndarray, grad_lower: np.ndarray,
        grad_upper: np.ndarray, issue: np.ndarray, mask: np.ndarray,
    ) -> None:
        nonlocal leaf_row, cell_row
        count = len(level)
        stop = leaf_row + count
        if stop > len(cover.boxes) or any(
            not same(actual, retained[leaf_row:stop])
            for actual, retained in (
                (boxes, cover.boxes),
                (lower, cover.value_lower),
                (upper, cover.value_upper),
                (grad_lower, cover.gradient_lower),
                (grad_upper, cover.gradient_upper),
            )
        ):
            fail()
        partition.add(level, index, boxes)
        if not np.array_equal(mask, (lower <= 0.0) & (upper >= 0.0)):
            fail()
        selected = np.flatnonzero(mask)
        end = cell_row + len(selected)
        if end > len(scaffold.cell_indices):
            fail()
        keys = np.column_stack((level[selected], index[selected]))
        if (
            np.any(issue[selected] != 0)
            or not same(keys, scaffold.cell_indices[cell_row:end])
            or not same(boxes[selected], scaffold.cell_boxes[cell_row:end])
            or not same(lower[selected], scaffold.cell_value_lower[cell_row:end])
            or not same(upper[selected], scaffold.cell_value_upper[cell_row:end])
            or not same(grad_lower[selected], scaffold.cell_gradient_lower[cell_row:end])
            or not same(grad_upper[selected], scaffold.cell_gradient_upper[cell_row:end])
        ):
            fail()
        leaf_row, cell_row = stop, end

    deepest, status = discover_leaves(
        bounds, np.asarray(scaffold.domain), int(scaffold.target_level), sink=consume
    )
    if (
        leaf_row != len(cover.boxes)
        or cell_row != len(scaffold.cell_indices)
        or deepest != surface.evidence.maximum_level_reached
        or status != int(surface.evidence.status)
    ):
        fail()
    partition.finish()


def _check_scaffold_lattice(surface: AdaptiveImplicitSurface, bounds: _FieldEnclosures, /) -> None:
    """Re-evaluate lattice, point enclosures, roots and tetrahedral incidence."""
    from .implicit._adaptive_discovery import (
        _CORNER_OFFSETS,
        _root_on_edges,
        _TETRAHEDRA,
    )
    from .implicit._adaptive_packed_frontier import _Coordinates

    scaffold = surface.scaffold
    ids = np.asarray(scaffold.corner_ids, dtype=np.int64)
    cell_ids = np.asarray(scaffold.cell_indices, dtype=np.int64)
    edges = np.asarray(scaffold.root_edge_ids, dtype=np.int64)
    corners = _integer_rows(ids)

    def fail() -> None:
        raise SourceCoverageUnavailable(
            "Implicit discovery does not cold-reproduce its retained roots and intervals.",
            status="violated",
        )
    def strictly_sorted(rows: np.ndarray) -> bool:
        before, after = rows[:-1], rows[1:]
        less = np.zeros(len(before), dtype=np.bool_)
        equal = np.ones(len(before), dtype=np.bool_)
        for axis in range(rows.shape[1]):
            less |= equal & (before[:, axis] < after[:, axis])
            equal &= before[:, axis] == after[:, axis]
        return bool(np.all(less))


    if (
        ids.ndim != 2 or ids.shape[1:] != (3,)
        or cell_ids.ndim != 2 or cell_ids.shape[1:] != (4,)
        or edges.ndim != 3 or edges.shape[1:] != (2, 3)
        or not strictly_sorted(ids)
        or not strictly_sorted(edges.reshape(-1, 6))
        or np.any(cell_ids[:, 0] != scaffold.target_level)
        or np.any(cell_ids[:, 1:] < 0)
    ):
        fail()
    level = int(scaffold.target_level)
    if level < 0 or np.any(cell_ids[:, 1:] >= (1 << level)):
        fail()
    coordinates = _Coordinates(np.asarray(scaffold.domain), level, len(ids))
    seen_corners = np.zeros(len(ids), dtype=np.bool_)
    tetra_pattern = np.asarray(_TETRAHEDRA, dtype=np.int64)
    tetra_count = len(tetra_pattern)
    if (scaffold.tetrahedra.shape != (len(cell_ids) * tetra_count, 4)
        or scaffold.tetrahedron_cell_rows.shape != (len(cell_ids) * tetra_count,)):
        fail()

    def corner_lookup(query: np.ndarray) -> np.ndarray:
        keys = _integer_rows(query)
        position = np.searchsorted(corners, keys)
        if np.any(position >= len(corners)):
            fail()
        if np.any(corners[position] != keys):
            fail()
        return position

    for start in range(0, len(ids), 4096):
        stop = min(start + 4096, len(ids))
        points = coordinates.values(ids[start:stop], 2)
        lower, upper = bounds.point_values(points)
        values = bounds.field_values(points)
        signs = np.where(upper < 0.0, -1, np.where(lower > 0.0, 1, 0))
        if (
            not _same_bits(points, scaffold.corner_coordinates[start:stop])
            or not _same_bits(lower, scaffold.corner_value_lower[start:stop])
            or not _same_bits(upper, scaffold.corner_value_upper[start:stop])
            or not _same_bits(values, scaffold.corner_sampled_values[start:stop])
            or not np.array_equal(signs, scaffold.corner_signs[start:stop])
        ):
            fail()

    for start in range(0, len(cell_ids), 1024):
        stop = min(start + 1024, len(cell_ids))
        expected = cell_ids[start:stop, None, 1:] + _CORNER_OFFSETS[None, :, :]
        if not np.array_equal(expected, scaffold.cell_corner_ids[start:stop]):
            fail()
        rows = corner_lookup(expected.reshape(-1, 3)).reshape(-1, 8)
        seen_corners[rows] = True
        tetra = rows[:, tetra_pattern].reshape(-1, 4)
        lo, hi = start * tetra_count, stop * tetra_count
        if (
            not np.array_equal(tetra, scaffold.tetrahedra[lo:hi])
            or not np.array_equal(
                np.repeat(np.arange(start, stop), tetra_count),
                scaffold.tetrahedron_cell_rows[lo:hi],
            )
        ):
            fail()
    if not np.all(seen_corners):
        fail()

    for start in range(0, len(edges), 256):
        stop = min(start + 256, len(edges))
        edge = edges[start:stop]
        rows = corner_lookup(edge.reshape(-1, 3)).reshape(-1, 2)
        if np.any(rows[:, 0] >= rows[:, 1]):
            fail()
        endpoint_signs = np.asarray(scaffold.corner_signs)[rows]
        if np.any(endpoint_signs[:, 0] * endpoint_signs[:, 1] != -1):
            fail()
        if (
            not _same_bits(scaffold.corner_value_lower[rows], scaffold.root_endpoint_value_lower[start:stop])
            or not _same_bits(scaffold.corner_value_upper[rows], scaffold.root_endpoint_value_upper[start:stop])
        ):
            fail()
        endpoints = np.asarray(scaffold.corner_coordinates)[rows]
        records = _root_on_edges(bounds, endpoints, endpoint_signs, surface.policy.root_tolerance)
        for offset, record in enumerate(records):
            row = start + offset
            if (
                not _same_bits(record[0], scaffold.root_points[row])
                or not _same_bits(np.asarray(record[1]), np.asarray(scaffold.root_errors[row]))
                or record[2] != scaffold.root_issues[row]
                or not _same_bits(record[3], scaffold.root_brackets[row, 0])
                or not _same_bits(record[4], scaffold.root_brackets[row, 1])
            ):
                fail()


def _check_discovery_metadata(surface: AdaptiveImplicitSurface, bounds: _FieldEnclosures, /) -> None:
    """Bind retained identities and evidence to the replayed source content."""
    from .implicit._adaptive_discovery import (
        _cover_identity,
        _norm_upper_rows,
        _state_identity,
    )

    cover = surface.cover
    evidence = surface.evidence
    scaffold = surface.scaffold
    volume = surface.volume

    def fail() -> NoReturn:
        raise SourceCoverageUnavailable(
            "Implicit discovery does not cold-reproduce its retained roots and intervals.",
            status="violated",
        )

    if cover is None:
        fail()
    state_id = _state_identity(surface.geometry)
    domain = np.asarray(scaffold.domain, dtype=np.float64)
    geometry_bounds = np.asarray(surface.geometry.bounds, dtype=np.float64)
    source_in_domain = (
        geometry_bounds.shape == (2, 3)
        and np.all(np.isfinite(geometry_bounds))
        and np.all(domain[0] <= geometry_bounds[0])
        and np.all(domain[1] >= geometry_bounds[1])
    )
    cover_id = _cover_identity(
        cover.boxes, cover.value_lower, cover.value_upper, domain,
        surface.source_id, state_id, surface.policy.enclosure, bool(source_in_domain),
    )
    query_id = canonical_fingerprint(
        {
            "kind": "implicit-volume-query",
            "state": state_id,
            "domain": array_tree_fingerprint(domain),
            "enclosure": surface.policy.enclosure,
        }
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "adaptive-implicit-evidence",
            "accuracy": "enclosed",
            "enclosure": surface.policy.enclosure,
            "status": evidence.status,
            "unresolved_boxes": array_tree_fingerprint(evidence.unresolved_boxes),
            "unresolved_issues": array_tree_fingerprint(evidence.unresolved_issues),
            "cover": cover_id,
            "leaf_count": len(cover.boxes),
            **(
                {}
                if evidence.lattice_anchor == 0
                else {
                    "lattice_anchor": evidence.lattice_anchor,
                    "declared_domain": array_tree_fingerprint(evidence.declared_domain),
                }
            ),
        }
    )
    max_diagonal = 0.0
    for start in range(0, len(scaffold.cell_boxes), 4096):
        boxes = scaffold.cell_boxes[start : start + 4096]
        max_diagonal = max(
            max_diagonal,
            float(np.max(_norm_upper_rows(boxes[:, 1] - boxes[:, 0]), initial=0.0)),
        )
    if (
        not source_in_domain
        or not bool(np.asarray(surface.geometry.validity().accepted))
        or not cover.complete or not cover.certified
        or cover.source_id != surface.source_id
        or cover.state_id != state_id
        or cover.enclosure != surface.policy.enclosure
        or cover.cover_id != cover_id
        or not np.array_equal(cover.domain, domain)
        or not np.array_equal(volume.domain, domain)
        or volume.source_id != surface.source_id
        or not volume.certified
        or array_tree_fingerprint(volume.geometry) != array_tree_fingerprint(surface.geometry)
        or volume.query_id != query_id
        or evidence.accuracy != "enclosed"
        or evidence.enclosure != surface.policy.enclosure
        or evidence.leaf_count != len(cover.boxes)
        or evidence.surface_leaf_count != len(scaffold.cell_boxes)
        or evidence.maximum_surface_box_diagonal != max_diagonal
        or evidence.evidence_id != evidence_id
        or not bounds.value_rigorous
    ):
        fail()


def _implicit_root_derivation(
    surface: AdaptiveImplicitSurface,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_regions: np.ndarray,
    witness: SourceBoundaryChartWitness,
    /,
) -> tuple[str, int]:
    from ._certificate import FieldRegularity
    from ._contracts import GeometryKind
    from .implicit._adaptive_discovery import _FieldEnclosures, verify_discovery_history
    from .implicit._adaptive_polygon_replay import replay_polygon

    if not isinstance(surface, AdaptiveImplicitSurface):
        raise SourceCoverageUnavailable(
            "Implicit coverage requires AdaptiveImplicitSurface.", status="unresolved"
        )

    if (
        surface.geometry.kind is not GeometryKind.REGION
        or surface.policy.allow_nonsmooth_field
        or surface.geometry.field_certificate.regularity is not FieldRegularity.SMOOTH
    ):
        raise SourceCoverageUnavailable(
            "Implicit coverage requires a smooth region-valued source.",
            status="unresolved",
        )
    cover = surface.cover
    scaffold = surface.scaffold
    source_mesh = surface.mesh
    if cover is None or source_mesh is None:
        raise SourceCoverageUnavailable(
            "Implicit discovery does not cold-reproduce its retained roots and intervals.",
            status="violated",
        )
    bounds = _FieldEnclosures(surface.geometry, surface.policy.enclosure)
    if (
        not bounds.value_rigorous
        or not bounds.gradient_rigorous
        or surface.evidence.unresolved_issues.size
    ):
        raise SourceCoverageUnavailable(
            "Implicit source has unresolved value, gradient, root, or topology premises.",
            status="unresolved",
        )
    _validate_oriented_implicit_triangles(np.asarray(source_mesh.faces))
    if (
        scaffold.cell_indices.shape[0] != scaffold.cell_boxes.shape[0]
        or scaffold.triangle_cell_rows.shape[0] != source_mesh.faces.shape[0]
        or scaffold.triangle_root_edge_rows.shape[0] != source_mesh.faces.shape[0]
        or np.any(scaffold.cell_issues != 0)
        or np.any(scaffold.root_issues != 0)
        or np.any(scaffold.corner_signs == 0)
        or np.any(scaffold.root_errors < 0.0)
    ):
        raise SourceCoverageUnavailable(
            "Implicit source root ancestry contains unresolved strata.",
            status="unresolved",
        )

    # Reprove exact dyadic tiling and the outward value enclosures of its cells.
    boxes = np.asarray(cover.boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        raise SourceCoverageUnavailable(
            "Implicit source has no interval cover leaves.", status="unresolved"
        )
    domain_box = np.asarray(cover.domain, dtype=np.float64)
    if not np.array_equal(domain_box, scaffold.domain):
        raise SourceCoverageUnavailable(
            "Implicit discovery does not cold-reproduce its retained roots and intervals.",
            status="violated",
        )
    _check_leaf_replay(surface, bounds)
    for start in range(0, boxes.shape[0], 256):
        stop = min(start + 256, boxes.shape[0])
        evaluated = bounds.boxes(boxes[start:stop, 0], boxes[start:stop, 1])
        if np.any(cover.value_lower[start:stop] > evaluated.value_lower) or np.any(
            cover.value_upper[start:stop] < evaluated.value_upper
        ):
            raise SourceCoverageUnavailable(
                "Implicit cover does not retain valid source value enclosures.",
                status="violated",
            )

    _check_scaffold_lattice(surface, bounds)
    if not bounds.gradient_rigorous:
        raise SourceCoverageUnavailable(
            "Implicit source has unresolved value, gradient, root, or topology premises.",
            status="unresolved",
        )
    _check_discovery_metadata(surface, bounds)
    if not replay_polygon(surface, bounds) or not verify_discovery_history(surface):
        raise SourceCoverageUnavailable(
            "Implicit discovery does not cold-reproduce its retained roots and intervals.",
            status="violated",
        )

    corner_rows_array = np.asarray(scaffold.corner_ids, dtype=np.int64)
    if corner_rows_array.ndim != 2 or corner_rows_array.shape[1] != 3:
        raise SourceCoverageUnavailable(
            "Implicit grid corner IDs are not three-dimensional.", status="violated"
        )
    corner_rows = _SortedIntegerKeys(corner_rows_array)
    root_edge_rows = np.asarray(scaffold.root_edge_ids, dtype=np.int64)
    if root_edge_rows.ndim != 3 or root_edge_rows.shape[1:] != (2, 3):
        raise SourceCoverageUnavailable(
            "Implicit root edge IDs omit endpoint coordinates.", status="violated"
        )
    root_keys = _SortedIntegerKeys(root_edge_rows)
    root_point_rows = np.asarray(scaffold.root_points, dtype=np.float64)
    order = np.lexsort(root_point_rows.T[::-1])
    for start in range(0, max(0, len(order) - 1), 65536):
        stop = min(start + 65536, len(order) - 1)
        if np.any(np.all(
            root_point_rows[order[start:stop]] == root_point_rows[order[start + 1:stop + 1]],
            axis=1,
        )):
            raise SourceCoverageUnavailable(
                "Implicit scaffold merges distinct source root identities.",
                status="violated",
            )
    del order

    def certify_zero_free_segments(endpoints: np.ndarray, brackets: np.ndarray) -> np.ndarray:
        # Each stack is bounded to one root; only the current frontier of this
        # fixed-size batch is passed to the interval evaluator.
        stacks = [
            [(endpoints[row, 0], brackets[row, 0]),
             (brackets[row, 1], endpoints[row, 1])]
            for row in range(len(endpoints))
        ]
        certified = np.ones(len(stacks), dtype=np.bool_)
        while True:
            rows = [row for row, stack in enumerate(stacks) if stack]
            if not rows:
                return certified
            segments = [stacks[row].pop() for row in rows]
            low = np.asarray([segment[0] for segment in segments])
            high = np.asarray([segment[1] for segment in segments])
            enclosure = bounds.boxes(np.minimum(low, high), np.maximum(low, high))
            excluded = (enclosure.value_lower > 0.0) | (enclosure.value_upper < 0.0)
            for local in np.flatnonzero(~excluded):
                row = rows[int(local)]
                first, second = low[local], high[local]
                midpoint = first + 0.5 * (second - first)
                if np.array_equal(midpoint, first) or np.array_equal(midpoint, second):
                    certified[row] = False
                    stacks[row].clear()
                else:
                    stacks[row].append((midpoint, second))
                    stacks[row].append((first, midpoint))

    # Each retained root is the unique simple root on its complete source edge.
    for start in range(0, len(root_edge_rows), 256):
        stop = min(start + 256, len(root_edge_rows))
        edge_records = _integer_rows(root_edge_rows[start:stop].reshape(-1, 3))
        corner_positions = np.searchsorted(corner_rows.keys, edge_records)
        if (
            np.any(corner_positions >= len(corner_rows.keys))
            or np.any(corner_rows.keys[corner_positions] != edge_records)
        ):
            raise SourceCoverageUnavailable(
                "Implicit root ancestry references absent grid corners.",
                status="violated",
            )
        endpoints = np.asarray(scaffold.corner_coordinates)[corner_positions].reshape(-1, 2, 3)
        brackets = np.asarray(scaffold.root_brackets[start:stop], dtype=np.float64)
        endpoint_values = bounds.point_values(endpoints.reshape(-1, 3))
        bracket_values = bounds.point_values(brackets.reshape(-1, 3))
        endpoint_signs = np.where(
            endpoint_values[1] < 0.0, -1, np.where(endpoint_values[0] > 0.0, 1, 0)
        ).reshape(-1, 2)
        bracket_signs = np.where(
            bracket_values[1] < 0.0, -1, np.where(bracket_values[0] > 0.0, 1, 0)
        ).reshape(-1, 2)
        derivatives = bounds.boxes(np.min(brackets, axis=1), np.max(brackets, axis=1))
        excluded = certify_zero_free_segments(endpoints, brackets)
        for local in range(stop - start):
            point = np.asarray(scaffold.root_points[start + local], dtype=np.float64)
            bracket = brackets[local]
            first, second = endpoints[local]
            if np.any(point < np.minimum(bracket[0], bracket[1])) or np.any(
                point > np.maximum(bracket[0], bracket[1])
            ):
                raise SourceCoverageUnavailable(
                    "Implicit root point escapes its retained bracket.", status="violated"
                )
            signs = endpoint_signs[local]
            bracket_pair = bracket_signs[local]
            if (not signs[0] or not signs[1] or signs[0] == signs[1]
                or not bracket_pair[0] or not bracket_pair[1]
                or bracket_pair[0] == bracket_pair[1]):
                raise SourceCoverageUnavailable(
                    "Implicit root endpoints lack opposite strict interval signs.",
                    status="unresolved",
                )
            direction = (
                _fraction(second[0]) - _fraction(first[0]),
                _fraction(second[1]) - _fraction(first[1]),
                _fraction(second[2]) - _fraction(first[2]),
            )
            derivative_sign = _strict_sign(
                _interval_dot(
                    derivatives.gradient_lower[local], derivatives.gradient_upper[local], direction
                )
            )
            if derivative_sign == 0:
                raise SourceCoverageUnavailable(
                    "Implicit root bracket is not proven simple and unique.",
                    status="unresolved",
                )
            if not excluded[local]:
                raise SourceCoverageUnavailable(
                    "Implicit root edge has unexcluded additional zero roots.",
                    status="unresolved",
                )

    from ._implicit_cube_topology import prove_implicit_cube_topology

    prove_implicit_cube_topology(surface, bounds)
    triangle_cells = np.asarray(scaffold.triangle_cell_rows, dtype=np.int64)
    triangle_roots = np.asarray(scaffold.triangle_root_edge_rows, dtype=np.int64)

    # The root-point carrier must be an embedded, oriented subdivision of the
    # independently reconstructed source scaffold. Exact coordinates and root
    # ancestry, not source identity labels, establish the relation.
    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    source_face_rows = np.asarray(source_mesh.faces)
    for start in range(0, source_face_rows.shape[0], 4096):
        stop = min(start + 4096, source_face_rows.shape[0])
        if not np.array_equal(
            source_vertices[source_face_rows[start:stop]],
            root_point_rows[triangle_roots[start:stop]],
        ):
            raise SourceCoverageUnavailable(
                "Implicit triangle ancestry does not reproduce its source root-edge vertices.",
                status="violated",
            )
    from ._implicit_target_coverage import prove_implicit_target_coverage

    prove_implicit_target_coverage(
        mesh, geometry, cell_regions, witness, bounds, source_vertices,
        source_face_rows, triangle_cells, triangle_roots, root_keys, scaffold,
    )

    degree = canonical_fingerprint(
        {
            "kind": "implicit-source-stratified-boundary-degree",
            "domain": surface.domain_id,
            "source": (surface.source_id, surface.volume.query_id),
            "region_id": surface.region_ids[0],
            "root_cells": array_tree_fingerprint(
                (scaffold.cell_indices, scaffold.tetrahedra)
            ),
            "root_intervals": array_tree_fingerprint(
                (
                    scaffold.root_edge_ids,
                    scaffold.root_brackets,
                    scaffold.root_endpoint_value_lower,
                    scaffold.root_endpoint_value_upper,
                )
            ),
            "triangle_ancestry": array_tree_fingerprint(
                (scaffold.triangle_cell_rows, scaffold.triangle_root_edge_rows)
            ),
            "witness": witness.witness_id,
            "orientation": "gradient-outward-negative-interior",
        }
    )
    return degree, source_mesh.faces.shape[0]

