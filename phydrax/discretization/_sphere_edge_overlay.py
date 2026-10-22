#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Candidate-bound exact spherical edge intervals and occurrence cycles.

Only actual candidate incidences introduce edge pairs. A real edge is named
by its *signed* primitive endpoints, not merely its supporting great circle:
opposite hemispheres must never acquire each other's crossings. Each pair
owns its actual finite crossing/overlap rays and each edge their exact arc
parameters. Queries construct convex boundary intervals from these numeric
operands and freshly classified endpoints, never fragment midpoints or saved
polygon verdicts. Whole-reference coverage remains the caller's obligation.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Protocol

import numpy as np

from . import _chart_geometry as chart_geometry
from ._coordinate_budget import CoordinateEnclosureBudget, CoordinateLiveStorage


type _Ray = tuple[int, int, int]
type _Point = tuple[Fraction, Fraction]
type _EdgeKey = tuple[_Ray, _Ray]


class _Cell(Protocol):
    @property
    def rays(self) -> tuple[_Ray, ...]: ...
    @property
    def scales(self) -> tuple[Fraction, ...]: ...
    @property
    def normals(self) -> tuple[_Ray, ...]: ...
    @property
    def orientations(self) -> tuple[Fraction, ...]: ...


class _Operands(Protocol):
    budget: CoordinateEnclosureBudget
    storage: CoordinateLiveStorage
    _work_start: int
    _work_limit: int
    _memory_limit: int

    def _cell(self, target: bool, row: int) -> _Cell: ...
    def _meet(self, first: _Ray, second: _Ray) -> _Ray: ...
    def _dot(self, first: _Ray, second: _Ray) -> int: ...
    def _release_prepare_products(self) -> None: ...
    def _coordinates(self, row: int, ray: _Ray) -> tuple[_Point, Fraction]: ...
    def _admit(self, work: int, count: int = 0, bits: int = 1) -> None: ...
    def _retain(self, count: int, bits: int, metadata: int = 512) -> None: ...
    def _retain_metadata(self, size: int) -> None: ...
    def _bind_operation_rays(
        self, source: object, target: object, source_id: str, target_id: str,
        source_rays: np.ndarray, target_rays: np.ndarray,
    ) -> None: ...


class _SphereCandidateOrientations:
    """Owned exact native numeric results for one actual candidate operation."""

    def __init__(
        self, source: object, target: object, source_id: str, target_id: str,
        banks: tuple[np.ndarray, np.ndarray], count: int,
        *, budget: CoordinateEnclosureBudget,
    ) -> None:
        self.source, self.target = source, target
        self.source_id, self.target_id = source_id, target_id
        self.budget = budget
        size = sum(bank.nbytes for bank in banks)
        budget.reserve((size + 63) // 64, size + 36 * count + 1024)
        self.rays = tuple(
            np.frombuffer(bank.tobytes(order="C"), dtype=np.float64).reshape(bank.shape)
            for bank in banks
        )
        self._full_signs = np.empty((count, 2, 3, 3), dtype=np.int8)
        self._full_certain = np.empty((count, 2, 3, 3), dtype=np.bool_)
        self.signs: np.ndarray | None = None
        self.certain: np.ndarray | None = None
        self.rows: dict[tuple[int, int], int] | None = None
        self._consumer: weakref.ReferenceType[_Operands] | None = None

    def record(
        self, owner: int, start: int, stop: int,
        signs: np.ndarray, certain: np.ndarray,
    ) -> None:
        self.budget.reserve((18 * (stop - start) + 63) // 64)
        self._full_signs[start:stop, owner] = signs
        self._full_certain[start:stop, owner] = certain

    def finish(
        self, sources: np.ndarray, targets: np.ndarray, kept: np.ndarray
    ) -> None:
        count = kept.size
        self.budget.reserve(
            2 * count + (36 * count + 63) // 64,
            1024 + 4 * 36 * count + 128 * count,
        )
        shape = (count, 2, 3, 3)
        self.signs = np.frombuffer(
            self._full_signs[kept].tobytes(), dtype=np.int8
        ).reshape(shape)
        self.certain = np.frombuffer(
            self._full_certain[kept].tobytes(), dtype=np.bool_
        ).reshape(shape)
        self.rows = {
            (int(source), int(target)): index
            for index, (source, target) in enumerate(zip(sources, targets, strict=True))
        }
        del self._full_signs, self._full_certain

    def bind(self, primitive: _Operands) -> None:
        if (
            self.budget is not primitive.budget or self.rows is None
            or self._consumer is not None
        ):
            raise ValueError("sphere_chart.overlay.orientations must belong to this completed numeric operation.")
        self._consumer = weakref.ref(primitive)
        primitive._bind_operation_rays(
            self.source, self.target, self.source_id, self.target_id,
            self.rays[0], self.rays[1],
        )

    def edge_values(
        self, source: int, target: int, source_opposite: int, target_opposite: int
    ) -> tuple[int, int, int, int] | None:
        if self.rows is None or self.signs is None or self.certain is None:
            raise ValueError("sphere_chart.overlay.orientations must contain actual native entries.")
        row = self.rows.get((source, target))
        if row is None:
            raise ValueError("sphere_chart.overlay.orientations must bind the actual candidate occurrence.")
        # Native rows name cyclic (i,i+1) edges; overlay slots are opposite i.
        # Products of both endpoint signs are invariant under edge reversal.
        result = []
        for owner, edge, opposite in (
            (0, (source_opposite + 1) % 3, target_opposite),
            (1, (target_opposite + 1) % 3, source_opposite),
        ):
            first, last = (opposite + 1) % 3, (opposite + 2) % 3
            if not (
                self.certain[row, owner, edge, first]
                and self.certain[row, owner, edge, last]
            ):
                return None
            result.extend((
                int(self.signs[row, owner, edge, first]),
                int(self.signs[row, owner, edge, last]),
            ))
        return result[0], result[1], result[2], result[3]

    def endpoint_values(
        self, source: int, target: int, owner: int, corner: int,
    ) -> tuple[int, int, int] | None:
        """Return actual opposite-cell edge numerics for one owned endpoint."""
        if self.rows is None or self.signs is None or self.certain is None:
            raise ValueError("sphere_chart.overlay.orientations must contain actual native entries.")
        row = self.rows.get((source, target))
        if row is None:
            raise ValueError("sphere_chart.overlay.orientations must bind the actual candidate occurrence.")
        # Native owner names the tested edge, not the tested endpoint. Its
        # cyclic edge (i,i+1) is opposite (i+2), hence opposite slot + 1.
        other = 1 - owner
        result = []
        for opposite in range(3):
            edge = (opposite + 1) % 3
            value = int(self.signs[row, other, edge, corner])
            if not self.certain[row, other, edge, corner] or value == 0:
                return None
            result.append(value)
        return result[0], result[1], result[2]


@dataclass(slots=True)
class _Edge:
    first: _Ray
    last: _Ray
    normal: _Ray
    axes: tuple[int, int]
    determinant: int
    events: dict[_Ray, Fraction]


@dataclass(frozen=True, slots=True)
class _Occurrence:
    cell: _Cell
    signed: tuple[_Ray, ...]
    edges: tuple[_EdgeKey, ...]
    determinant: int


def _negate(ray: _Ray) -> _Ray:
    return -ray[0], -ray[1], -ray[2]


def _canonical(ray: _Ray) -> _Ray:
    return _negate(ray) if next(value for value in ray if value) < 0 else ray


def _bits(*rays: _Ray) -> int:
    return max(abs(value).bit_length() for ray in rays for value in ray) + 2


class _SphereEdgeOverlay:
    """One prepared candidate bank, with independently reconstructed cycles.

    ``prepare`` binds copies of the actual source/target occurrence row IDs.
    Its returned work is additional to each ``polygon`` call's returned work;
    both are measured on the supplied primitive's real caller budget.  Limits
    bound the complete live operand/overlay storage, not a private extra pool.
    The primitive owns authenticated operands only; this object owns numeric
    incidence, edge parameters, finite crossings and candidate identities.
    """

    def __init__(
        self, primitive: _Operands,
        orientations: _SphereCandidateOrientations | None = None,
    ) -> None:
        self.primitive = primitive
        self.orientations = orientations
        self._pairs: set[tuple[int, int]] | None = None
        self._edges: dict[_EdgeKey, _Edge] | None = None
        self._occurrences: dict[tuple[bool, int], _Occurrence] | None = None
        self._crossings: dict[tuple[_EdgeKey, _EdgeKey], tuple[_Ray, ...]] | None = None

    def _begin(self, maximum_work: int, maximum_memory: int) -> int:
        primitive = self.primitive
        primitive._work_start = primitive.budget.work_units
        primitive._work_limit = maximum_work
        primitive._memory_limit = maximum_memory
        if primitive.storage.bound > maximum_memory:
            raise ValueError(
                "sphere_chart.overlay.budget must satisfy the admitted resource limits; "
                "got Retained exact edge operands exceed the resident memory limit."
            )
        primitive._admit(0)
        return primitive.budget.work_units

    def _occurrence(self, target: bool, row: int) -> _Occurrence:
        primitive = self.primitive
        occurrences, edges = self._occurrences, self._edges
        if occurrences is None or edges is None:
            raise ValueError("sphere_chart.overlay must be prepared before occurrence access.")
        primitive._admit(1)
        key = target, row
        known = occurrences.get(key)
        if known is not None:
            return known
        cell = primitive._cell(target, row)
        primitive._retain(0, 1, 1024)
        primitive._admit(6, 12, _bits(*cell.rays))
        signed = tuple(
            ray if scale > 0 else _negate(ray)
            for ray, scale in zip(cell.rays, cell.scales, strict=True)
        )
        edge_keys: list[_EdgeKey] = []
        for slot in range(3):
            first, last = signed[(slot + 1) % 3], signed[(slot + 2) % 3]
            edge_key = (first, last) if first < last else (last, first)
            primitive._admit(1)
            if edge_key not in edges:
                normal = cell.normals[slot]
                # A nonzero normal component supplies an invertible 2x2
                # coordinate minor of the two actual, signed endpoints.
                axis = next(index for index, value in enumerate(normal) if value)
                i, j = (axis + 1) % 3, (axis + 2) % 3
                a, b = edge_key
                bits = 2 * _bits(a, b)
                primitive._admit(3, 6, bits)
                primitive._retain(3, bits, 1536)
                determinant = a[i] * b[j] - a[j] * b[i]
                if determinant == 0:
                    raise ValueError("sphere_chart.overlay.edges must have rank two.")
                edges[edge_key] = _Edge(
                    a, b, normal, (i, j), determinant,
                    {a: Fraction(0), b: Fraction(1)},
                )
            edge_keys.append(edge_key)
        # The physical signed-ray determinant fixes boundary orientation
        # independently of arbitrary primitive-normal sign conventions.
        a, b, c = signed
        primitive._admit(14, 18, 3 * _bits(a, b, c))
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if determinant == 0:
            raise ValueError("sphere_chart.overlay.cells must have rank three.")
        # Rank three makes the positive endpoint cone full-dimensional and
        # pointed: the dual covector taking value one on all three signed
        # generators is strictly positive on every nonzero positive sum.
        # Thus all its finite edges are short positive arcs in one open
        # hemisphere, and intersecting another such convex cone gives one
        # closed convex edge interval (possibly empty or a singleton).
        known = _Occurrence(cell, signed, tuple(edge_keys), determinant)
        occurrences[key] = known
        return known

    def _locate(self, edge: _Edge, ray: _Ray) -> tuple[_Ray, Fraction] | None:
        """Select the physical antipode and its exact actual-edge parameter."""
        primitive = self.primitive
        # An existing event stores this exact actual-edge numeric parameter,
        # not an inside/accepted predicate. Do not solve the same two minors
        # and normalize the same rational again for a shared endpoint/event.
        primitive._admit(1)
        known = edge.events.get(ray)
        if known is not None:
            return ray, known
        bits = 2 * _bits(edge.first, edge.last, ray) + 2
        primitive._admit(12, 16, bits)
        i, j = edge.axes
        first = ray[i] * edge.last[j] - ray[j] * edge.last[i]
        last = edge.first[i] * ray[j] - edge.first[j] * ray[i]
        if first == 0 and last == 0:
            raise ValueError("sphere_chart.overlay.incidence must name a nonzero edge ray.")
        # Coefficients have the common denominator edge.determinant.  Both
        # must have the same sign to lie on one of this edge's two antipodes.
        if (first < 0 < last) or (last < 0 < first):
            return None
        positive = (first or last) > 0
        selected = ray if positive == (edge.determinant > 0) else _negate(ray)
        primitive._admit(3, 3, bits + 2)
        parameter = Fraction(abs(last), abs(first) + abs(last))
        return selected, parameter

    def _event(self, edge: _Edge, ray: _Ray, parameter: Fraction) -> None:
        primitive = self.primitive
        primitive._admit(1)
        known = edge.events.get(ray)
        if known is not None:
            if known != parameter:
                raise ValueError("sphere_chart.overlay.events must have one exact edge parameter.")
            return
        metadata = 192 + 3 * ((_bits(ray) + 7) // 8 + 32)
        primitive._retain(
            1, abs(parameter.numerator).bit_length() + parameter.denominator.bit_length(),
            metadata,
        )
        # Numeric parameters remain live for exact per-occurrence extrema.
        edge.events[ray] = parameter

    def _inspect(
        self, first: _Edge, last: _Edge,
        native_sides: tuple[int, int, int, int] | None = None,
    ) -> tuple[_Ray, ...]:
        primitive = self.primitive
        primitive._admit(1)
        if first.normal == last.normal:
            # Coincident circles require actual finite-arc overlap, not a
            # zero meet treated as an ordinary crossing.  Insert every shared
            # endpoint; these delimit partial, complete and singleton overlap.
            overlap: list[_Ray] = []
            for source, destination in ((first, last), (last, first)):
                for ray in (source.first, source.last):
                    located = self._locate(destination, ray)
                    if located is not None and located[0] == ray:
                        self._event(destination, ray, located[1])
                        if ray not in overlap:
                            overlap.append(ray)
            if len(overlap) > 2:
                raise ValueError("sphere_chart.overlay.crossings must delimit one finite coincident interval.")
            return tuple(overlap)
        # Distinct circles sharing an actual signed endpoint have no other
        # intersection on either positive endpoint cone: their second meet is
        # antipodal to that endpoint, which rank-two positive arcs exclude.
        # Both endpoint events already exist, so there is no numeric work to
        # repeat and no cached acceptance flag to consult.
        primitive._admit(4)
        shared = (
            first.first == last.first or first.first == last.last,
            first.last == last.first or first.last == last.last,
        )
        if all(shared):
            raise ValueError(
                "sphere_chart.overlay.edges must have consistent exact supporting "
                "circles; got distinct normals for the same two endpoint rays."
            )
        if any(shared):
            return (first.first if shared[0] else first.last,)
        if native_sides is None:
            # Zero/uncertain/collinear native entries take the original exact
            # numeric event path, including actual side reconstruction.
            for edge, normal in ((first, last.normal), (last, first.normal)):
                a = primitive._dot(normal, edge.first)
                b = primitive._dot(normal, edge.last)
                primitive._admit(2)
                if (a > 0 and b > 0) or (a < 0 and b < 0):
                    return ()
        else:
            primitive._admit(2)
            if not (
                native_sides[0] * native_sides[1] < 0
                and native_sides[2] * native_sides[3] < 0
            ):
                raise ValueError("sphere_chart.overlay.native_sides must be the actual strict opposite endpoint signs.")
        ray = primitive._meet(first.normal, last.normal)
        if not any(ray):
            raise ValueError("sphere_chart.overlay.crossings must have distinct supports.")
        a, b = self._locate(first, ray), self._locate(last, ray)
        primitive._admit(1)
        if a is not None and b is not None and a[0] == b[0]:
            self._event(first, *a)
            self._event(last, *b)
            return (a[0],)
        return ()

    def prepare(
        self,
        source_rows: Iterable[int],
        target_rows: Iterable[int],
        *,
        maximum_work: int,
        maximum_memory: int,
    ) -> int:
        """Bind actual candidate pairs and construct their finite crossings."""
        if self._pairs is not None:
            raise ValueError("sphere_chart.overlay.prepare must be called exactly once.")
        start = self._begin(maximum_work, maximum_memory)
        primitive = self.primitive
        if self.orientations is not None:
            self.orientations.bind(primitive)
        primitive._retain(0, 1, 8192)
        self._pairs, self._edges, self._occurrences = set(), {}, {}
        self._crossings = {}
        for source_value, target_value in zip(source_rows, target_rows, strict=True):
            # Exact arithmetic scratch belongs to this actual incidence.
            # Persistent atoms/events promote their own LiveStorage bounds.
            with primitive.budget.temporary_scope():
                primitive._admit(3)
                source_row, target_row = int(source_value), int(target_value)
                if source_row < 0 or target_row < 0:
                    raise ValueError("sphere_chart.overlay.candidates must name nonnegative actual rows.")
                pair = source_row, target_row
                if pair in self._pairs:
                    raise ValueError("sphere_chart.overlay.candidates must contain distinct actual pairs.")
                primitive._retain(0, 1, 160)
                self._pairs.add(pair)
                source = self._occurrence(False, source_row)
                target = self._occurrence(True, target_row)
                for source_slot, source_edge in enumerate(source.edges):
                    for target_slot, target_edge in enumerate(target.edges):
                        primitive._admit(1)
                        edge_pair = (
                            (source_edge, target_edge) if source_edge < target_edge
                            else (target_edge, source_edge)
                        )
                        if edge_pair not in self._crossings:
                            # Retain the real signed edgepair key and numeric
                            # crossing tuple, including an empty intersection.
                            primitive._retain_metadata(384)
                            values = None if self.orientations is None else (
                                self.orientations.edge_values(
                                    source_row, target_row, source_slot, target_slot
                                )
                            )
                            if values is not None and (
                                values[0] * values[1] > 0 or values[2] * values[3] > 0
                            ):
                                self._crossings[edge_pair] = ()
                                continue
                            self._crossings[edge_pair] = self._inspect(
                                self._edges[source_edge], self._edges[target_edge],
                                values if values is not None and all(values) else None,
                            )
        primitive._release_prepare_products()
        return primitive.budget.work_units - start

    def _inside(self, cell: _Cell, ray: _Ray) -> bool:
        primitive = self.primitive
        for normal, orientation in zip(cell.normals, cell.orientations, strict=True):
            value = primitive._dot(normal, ray)
            primitive._admit(1)
            if value and (value > 0) != (orientation > 0):
                return False
        return True

    def _endpoint_inside(
        self, source_row: int, target_row: int, owner: int,
        corner: int, occurrence: _Occurrence, other: _Occurrence,
    ) -> bool:
        primitive = self.primitive
        primitive._admit(1)
        values = None if self.orientations is None else (
            self.orientations.endpoint_values(source_row, target_row, owner, corner)
        )
        if values is None:
            # Zero, collinear and unresolved native numerics retain the full
            # exact side proof, using the current actual signed endpoint.
            return self._inside(other.cell, occurrence.signed[corner])
        primitive._admit(3)
        return all((value > 0) == (other.determinant > 0) for value in values)

    def _interval(
        self, edge: _Edge, key: _EdgeKey, owner: _Occurrence,
        other: _Occurrence, slot: int, inside: tuple[bool, ...],
    ) -> tuple[_Ray, _Ray] | None:
        """Construct the one convex interval from actual endpoint geometry."""
        primitive = self.primitive
        crossings = self._crossings
        if crossings is None:
            raise ValueError("sphere_chart.overlay must own its actual finite crossings.")
        candidates: list[_Ray] = []
        for corner in ((slot + 1) % 3, (slot + 2) % 3):
            primitive._admit(1)
            if inside[corner]:
                candidates.append(owner.signed[corner])
        for other_key in other.edges:
            primitive._admit(1)
            pair = (key, other_key) if key < other_key else (other_key, key)
            # Every ray is proved on BOTH actual finite positive arcs by
            # _inspect/_locate. A finite boundary arc of a rank-three positive
            # triangular cone lies in that cone, including coincident overlap
            # endpoints. No middle-point side predicate is necessary.
            candidates.extend(crossings[pair])
        if not candidates:
            return None
        first = last = candidates[0]
        first_parameter = last_parameter = edge.events[first]
        for ray in candidates[1:]:
            parameter = edge.events[ray]
            bits = max(
                abs(value.numerator).bit_length() + value.denominator.bit_length()
                for value in (parameter, first_parameter, last_parameter)
            )
            primitive._admit(2, 4, 2 * bits + 2)
            if parameter < first_parameter:
                first, first_parameter = ray, parameter
            if parameter > last_parameter:
                last, last_parameter = ray, parameter
        if first_parameter == last_parameter:
            return None
        return first, last

    def polygon(
        self,
        source_row: int,
        target_row: int,
        *,
        maximum_work: int,
        maximum_memory: int,
    ) -> tuple[
        tuple[_Point, ...], tuple[int, ...], tuple[_Ray, ...],
        tuple[int | None, ...], int,
    ]:
        """Reconstruct this actual occurrence's exact positive halfedge cycle."""
        primitive = self.primitive
        start = self._begin(maximum_work, maximum_memory)
        pairs, edges, occurrences = self._pairs, self._edges, self._occurrences
        primitive._admit(1)
        if pairs is None or edges is None or occurrences is None or self._crossings is None:
            raise ValueError("sphere_chart.overlay must be prepared before polygon queries.")
        if (source_row, target_row) not in pairs:
            raise ValueError("sphere_chart.overlay.query must name an admitted actual candidate pair.")
        source, target = occurrences[False, source_row], occurrences[True, target_row]
        # Each support has at most one boundary interval in this convex
        # intersection. Merge coincident selected supports, not subdivisions
        # introduced by unrelated neighbouring candidate occurrences.
        # Values are directed extreme source-chart endpoints and their label.
        primitive._admit(0, 192, _bits(*source.signed, *target.signed))
        supports: dict[_Ray, tuple[_Ray, _Ray, int, int, tuple[int, int]]] = {}
        for is_target, owner, other in ((False, source, target), (True, target, source)):
            inside = tuple(
                self._endpoint_inside(
                    source_row, target_row, int(is_target), corner, owner, other
                )
                for corner in range(3)
            )
            reverse = is_target and ((source.determinant > 0) != (target.determinant > 0))
            for slot, key in enumerate(owner.edges):
                edge = edges[key]
                forward = edge.first == owner.signed[(slot + 1) % 3]
                if reverse:
                    forward = not forward
                label = 3 + slot if is_target else (2, 0, 1)[slot]
                interval = self._interval(edge, key, owner, other, slot, inside)
                if interval is None:
                    continue
                first, last = interval if forward else (interval[1], interval[0])
                known = supports.get(edge.normal)
                i, j = edge.axes
                if known is None:
                    primitive._admit(3, 4, 2 * _bits(first, last))
                    direction = first[i] * last[j] - first[j] * last[i]
                    if direction == 0:
                        raise ValueError("sphere_chart.overlay.intervals must have distinct nonantipodal endpoints.")
                    supports[edge.normal] = first, last, label, direction, edge.axes
                    continue
                old_first, old_last, old_label, direction, axes = known
                i, j = axes
                primitive._admit(6, 8, 2 * _bits(old_first, old_last, first, last))
                # All selected rays lie in the same convex physical source
                # cone, hence an oriented 2x2 minor orders its short arc.
                before = old_first[i] * first[j] - old_first[j] * first[i]
                after = old_last[i] * last[j] - old_last[j] * last[i]
                if before and ((before > 0) != (direction > 0)):
                    old_first = first
                if after and ((after > 0) == (direction > 0)):
                    old_last = last
                supports[edge.normal] = (
                    old_first, old_last, min(old_label, label), direction, axes
                )
        work = lambda: primitive.budget.work_units - start
        if len(supports) < 3:
            return (), (), (), (), work()
        # Join the actual oriented halfedges, rather than angle-sorting
        # approximate points or selecting cached polygon cycles.
        outgoing: dict[_Ray, tuple[_Ray, int]] = {}
        for first, last, label, _, _ in supports.values():
            primitive._admit(2)
            if first in outgoing:
                raise ValueError("sphere_chart.overlay.cycles must have one outgoing boundary halfedge.")
            outgoing[first] = last, label
        first_ray = min(outgoing)
        ray = first_ray
        polygon: list[_Point] = []
        rays: list[_Ray] = []
        labels: list[int] = []
        corners: list[int | None] = []
        for _ in range(len(outgoing)):
            primitive._admit(4)
            following = outgoing.get(ray)
            if following is None or ray in rays:
                raise ValueError("sphere_chart.overlay.cycles must form one closed actual boundary.")
            last, label = following
            canonical = _canonical(ray)
            point, scale = primitive._coordinates(source_row, canonical)
            primitive._admit(2)
            if (scale > 0) != (canonical == ray):
                raise ValueError("sphere_chart.overlay.vertices must lie in the actual source hemisphere.")
            polygon.append(point)
            rays.append(ray)
            labels.append(label)
            corners.append(next((slot for slot, corner in enumerate(target.signed) if corner == ray), None))
            ray = last
        if ray != first_ray:
            raise ValueError("sphere_chart.overlay.cycles must close after every boundary halfedge.")
        exact_polygon = tuple(polygon)
        bits = max(
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for point in polygon for value in point
        )
        primitive._admit(12 * len(polygon) + 1, 32 * len(polygon), 8 * bits + 16)
        area = chart_geometry.area(exact_polygon)
        if area == 0:
            return (), (), (), (), work()
        if area < 0 or len(set(polygon)) != len(polygon) or any(
            chart_geometry.orientation(polygon[slot - 1], point, polygon[(slot + 1) % len(polygon)]) <= 0
            for slot, point in enumerate(polygon)
        ):
            raise ValueError(
                "sphere_chart.overlay.polygon must be distinct, strictly convex and positively oriented; "
                "got inconsistent exact edge-interval topology."
            )
        return exact_polygon, tuple(labels), tuple(_canonical(value) for value in rays), tuple(corners), work()
