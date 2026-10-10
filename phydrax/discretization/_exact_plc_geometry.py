#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact source coordinates of native PLC carriers with ancestry-backed vertices.

A native PLC/tetrahedral construction names, for every constrained vertex, a
source row (a PLC edge or an input triangle) and binary64 parameters ``t`` or
``(l1, l2)`` of the exact source point ``S = A + t (B - A)`` or
``S = A + l1 (B - A) + l2 (C - A)``. The authoritative coordinate is a pure
function of (carrier, row, parameters): the binary64 carrier itself when it
lies exactly in the closed row entity (exact rational membership, no
tolerance), otherwise ``S``, whose correctly rounded (RNE) value must equal
the carrier coordinate-wise. Any other carrier is refused, never repaired.
Unconstrained vertices (stratum 0) are their own exact carrier.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier


if TYPE_CHECKING:
    from .._bvh import PackedBVH
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh


type ExactPoint = tuple[Fraction, Fraction, Fraction]
type ExactMatrix = tuple[tuple[Fraction, ...], ...]

_NONE, _SEGMENT, _TRIANGLE = 0, 1, 2

from weakref import WeakValueDictionary


_WITNESS_LOCATORS = WeakValueDictionary()


def _witness_storage(size: int) -> None:
    """Reserve host proof storage before allocating it."""
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    if ledger is not None:
        ledger.reserve(0, size)
        return
    from .._meshcore import current_native_execution_budget, current_native_host_workspace

    workspace = current_native_host_workspace()
    if workspace is not None:
        workspace.set_bound(workspace.bound + size)
    elif current_native_execution_budget() is not None:
        raise RuntimeError(
            "PLC witness preparation requires an active native host workspace."
        )


def _witness_source_leaves(
    source: ExactPlcCellGeometrySource,
) -> tuple[Array, ...]:
    return (
        source.source_points,
        source.source_triangles,
        source.source_segments,
        source.source_triangle_ids,
        source.source_triangle_bounds,
        source.source_segment_ids,
        source.source_segment_bounds,
        source.vertex_strata,
        source.vertex_rows,
        source.vertex_parameters,
    )


def _witness_packed_snapshot(
    hierarchy: PackedBVH, budget: _ExactPlcBudget
) -> tuple[np.ndarray, ...]:
    """Immutable host views of the actual canonical packed hierarchy."""
    snapshots: list[np.ndarray] = []
    for name in (
        "bbox_min",
        "bbox_max",
        "left",
        "right",
        "leaf_id",
        "leaf_items",
        "item_bbox_min",
        "item_bbox_max",
    ):
        value = hierarchy.__getattribute__(name)
        budget.charge(value.size)
        _witness_storage(256 + 2 * value.size * value.dtype.itemsize)
        host = np.asarray(value)
        snapshots.append(
            np.frombuffer(host.tobytes(), dtype=host.dtype).reshape(host.shape)
        )
    return tuple(snapshots)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _PreparedWitnessLocator:
    """Ephemeral closed AABB proof; exact source rows remain authoritative."""

    owner: ExactPlcCellGeometrySource
    leaves: tuple[Array, ...]
    metadata: tuple[str, str, int, int]
    source_id: str
    rows: tuple[tuple[int, int, tuple[ExactPoint, ...]], ...]
    packed: tuple[np.ndarray, ...] | None
    hierarchy: PackedBVH | None

    def candidates(
        self,
        source: ExactPlcCellGeometrySource,
        point: ExactPoint,
        budget: _ExactPlcBudget,
    ) -> tuple[tuple[int, int, tuple[ExactPoint, ...]], ...]:
        budget.charge(16, point)
        if (
            _WITNESS_LOCATORS.get(id(self)) is not self
            or source is not self.owner
            or any(
                a is not b for a, b in zip(self.leaves, _witness_source_leaves(source))
            )
            or self.metadata
            != (
                source.domain_source_id,
                source.domain_source_revision,
                source.maximum_work,
                source.maximum_bits,
            )
        ):
            raise ValueError(
                "Prepared PLC witness locator does not bind the original source."
            )
        lower, upper = [], []
        for value in point:
            budget.charge(4, (value,))
            rounded = float(value)
            lower.append(
                float(np.nextafter(rounded, -np.inf))
                if Fraction(rounded) > value
                else rounded
            )
            upper.append(
                float(np.nextafter(rounded, np.inf))
                if Fraction(rounded) < value
                else rounded
            )
        _witness_storage(128)
        pending = [] if self.packed is None else [0]
        candidates = []
        while pending:
            budget.charge(8)
            node = pending.pop()
            (
                node_lower,
                node_upper,
                left,
                right,
                leaf_id,
                leaf_items,
                item_lower,
                item_upper,
            ) = self.packed
            if any(
                node_upper[node, axis] < lower[axis]
                or node_lower[node, axis] > upper[axis]
                for axis in range(3)
            ):
                continue
            leaf = int(leaf_id[node])
            if leaf >= 0:
                for item in leaf_items[leaf]:
                    budget.charge(8)
                    row = int(item)
                    if row < 0:
                        continue
                    if any(
                        item_upper[row, axis] < lower[axis]
                        or item_lower[row, axis] > upper[axis]
                        for axis in range(3)
                    ):
                        continue
                    _witness_storage(32)
                    candidates.append(row)
            else:
                _witness_storage(16)
                pending.extend((int(left[node]), int(right[node])))
        # Original segment rows precede original triangle rows, independent of tree order.
        from functools import cmp_to_key

        def compare(a: int, b: int) -> int:
            budget.charge(1)
            return (a > b) - (a < b)

        _witness_storage(64 * len(candidates))
        candidates.sort(key=cmp_to_key(compare))
        return tuple(self.rows[index] for index in candidates)


def _prepare_source_witness_locator(
    source: ExactPlcCellGeometrySource, budget: _ExactPlcBudget
) -> _PreparedWitnessLocator:
    leaves = _witness_source_leaves(source)
    for leaf in leaves:
        budget.charge(int(leaf.size))
        _witness_storage(int(leaf.size) * int(leaf.dtype.itemsize))
    identity = source.source_id
    points = np.asarray(source.source_points)
    rows, boxes = [], []
    from math import frexp

    for stratum, table in (
        (_SEGMENT, source.source_segments),
        (_TRIANGLE, source.source_triangles),
    ):
        for row, indices in enumerate(np.asarray(table)):
            budget.charge(16)
            # Admit scan scratch before inspecting the original float exponents;
            # then use the canonical PLC rational-bank bound before conversion.
            _witness_storage(512)
            maximum_ratio_bits = 1
            for index in indices:
                for coordinate in points[int(index)]:
                    budget.charge(1)
                    _, exponent = frexp(float(coordinate))
                    maximum_ratio_bits = max(
                        maximum_ratio_bits, 53, exponent, 54 - exponent
                    )
            bank_upper, _ = _plc_storage_bounds(len(indices), maximum_ratio_bits)
            _witness_storage(bank_upper)
            corners = tuple(_point(points[int(index)]) for index in indices)
            budget.charge(0, tuple(value for corner in corners for value in corner))
            lower = tuple(
                min(float(corner[axis]) for corner in corners) for axis in range(3)
            )
            upper = tuple(
                max(float(corner[axis]) for corner in corners) for axis in range(3)
            )
            rows.append((stratum, row, corners))
            boxes.append((lower, upper))
    from .._bvh import prepare_bvh

    hierarchy, packed = None, None
    if rows:
        budget.charge(6 * len(rows))
        _witness_storage(256 + 64 * len(rows))
        lower = np.asarray([box[0] for box in boxes], dtype=np.float64)
        upper = np.asarray([box[1] for box in boxes], dtype=np.float64)
        hierarchy = prepare_bvh(
            lower,
            upper,
            dtype=np.float64,
            _charge_work=budget.charge,
            _reserve_storage=_witness_storage,
        )
        packed = _witness_packed_snapshot(hierarchy, budget)
    _witness_storage(256 + 32 * len(rows))
    locator = _PreparedWitnessLocator(
        source,
        leaves,
        (
            source.domain_source_id,
            source.domain_source_revision,
            source.maximum_work,
            source.maximum_bits,
        ),
        identity,
        tuple(rows),
        packed,
        hierarchy,
    )
    _WITNESS_LOCATORS[id(locator)] = locator
    return locator


@dataclass(frozen=True, slots=True)
class PreparedExactPlcGeometry:
    """Bounded host proof of the current source, never an alternate authority.

    ``on_entity`` marks vertices whose carrier is the authority (exactly in
    its row entity or unconstrained); the others are their exact source point.
    ``maximum_rounding_error`` is the largest coordinate distance between an
    authoritative coordinate and its carrier.
    """

    vertices: tuple[ExactPoint, ...]
    on_entity: tuple[bool, ...]
    source_id: str
    operation_count: int
    maximum_integer_bits: int
    maximum_rounding_error: Fraction


@final
class ExactPlcCellGeometrySource(StrictModule, NonTrainableState):
    """PLC source rows and per-vertex ancestry witnesses of a native carrier.

    ``source_points`` (P, 3) and ``source_triangles`` (R, 3) are the declared
    PLC domain vertices and input triangles; ``source_segments`` (Q, 2) its
    PLC edges. Per carrier vertex, ``vertex_strata`` (0 unconstrained, 1
    segment row, 2 triangle row), ``vertex_rows`` and ``vertex_parameters``
    (N, 2; a segment's second parameter is exactly zero). ``domain_source_id``
    and ``domain_source_revision`` bind the declared original source identity.
    Preparation is host-only and charged to the active coordinate-enclosure
    and native execution budgets.
    """

    source_points: Array
    source_triangles: Array
    source_segments: Array
    source_triangle_ids: Array
    source_triangle_bounds: Array
    source_segment_ids: Array
    source_segment_bounds: Array
    vertex_strata: Array
    vertex_rows: Array
    vertex_parameters: Array
    domain_source_id: str = eqx.field(static=True)
    domain_source_revision: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)

    def __init__(
        self,
        source_points: ArrayLike,
        source_triangles: ArrayLike,
        source_segments: ArrayLike,
        vertex_strata: ArrayLike,
        vertex_rows: ArrayLike,
        vertex_parameters: ArrayLike,
        /,
        *,
        domain_source_id: str,
        domain_source_revision: str,
        source_triangle_ids: ArrayLike,
        source_triangle_bounds: ArrayLike,
        source_segment_ids: ArrayLike,
        source_segment_bounds: ArrayLike,
        maximum_work: int = 10_000_000,
        maximum_bits: int = 4096,
    ) -> None:
        identity = canonical_identifier(domain_source_id, "domain_source_id")
        revision = canonical_identifier(domain_source_revision, "domain_source_revision")
        points = np.asarray(source_points, dtype=np.float64)
        parameters = np.asarray(vertex_parameters, dtype=np.float64)
        raw = tuple(
            np.asarray(value)
            for value in (
                source_triangles,
                source_segments,
                vertex_strata,
                vertex_rows,
                source_triangle_ids,
                source_segment_ids,
            )
        )
        if any(value.dtype.kind not in "iu" for value in raw):
            raise TypeError("Exact PLC rows and witnesses require integer arrays.")
        if any(np.any(value > np.iinfo(np.int64).max) for value in raw):
            raise ValueError(
                "Exact PLC source indices and scientific IDs must fit signed int64."
            )
        triangles, segments, strata, rows, triangle_ids, segment_ids = (
            np.asarray(value, dtype=np.int64) for value in raw
        )
        triangle_bounds = np.asarray(source_triangle_bounds, dtype=np.float64)
        segment_bounds = np.asarray(source_segment_bounds, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not points.shape[0]:
            raise ValueError("Exact PLC sources require nonempty 3-D source points.")
        if (
            triangles.ndim != 2
            or triangles.shape[1] != 3
            or segments.ndim != 2
            or segments.shape[1] != 2
        ):
            raise ValueError(
                "Exact PLC source rows must be triangles (R, 3) and segments (Q, 2)."
            )
        points = points.reshape((points.shape[0], 3))
        triangles = triangles.reshape((triangles.shape[0], 3))
        segments = segments.reshape((segments.shape[0], 2))
        for table in (triangles, segments):
            if np.any(table < 0) or np.any(table >= points.shape[0]):
                raise ValueError("Exact PLC source rows index undeclared source points.")
        for table, identifiers, bounds in (
            (triangles, triangle_ids, triangle_bounds),
            (segments, segment_ids, segment_bounds),
        ):
            if (
                identifiers.shape != (table.shape[0],)
                or bounds.shape != identifiers.shape
            ):
                raise ValueError(
                    "Exact PLC scientific row IDs and bounds must align with source rows."
                )
            if (
                np.any(identifiers < 0)
                or not np.all(np.isfinite(bounds))
                or np.any(bounds < 0)
            ):
                raise ValueError(
                    "Exact PLC row IDs and bounds must be nonnegative and finite."
                )
        if strata.ndim != 1:
            raise ValueError("Exact PLC vertex strata require a rank-one bank.")
        count = strata.shape[0]
        if rows.shape != (count,) or parameters.shape != (count, 2):
            raise ValueError("Exact PLC vertex witnesses must align per carrier vertex.")
        strata = strata.reshape((count,))
        rows = rows.reshape((count,))
        parameters = parameters.reshape((count, 2))
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(parameters)):
            raise ValueError("Exact PLC source leaves must be finite.")
        limits = np.where(
            strata == _SEGMENT,
            segments.shape[0],
            np.where(strata == _TRIANGLE, triangles.shape[0], 0),
        )
        unconstrained = strata == _NONE
        if np.any((strata < _NONE) | (strata > _TRIANGLE)):
            raise ValueError("Exact PLC witness strata must be 0, 1 or 2.")
        if np.any(unconstrained & ((rows != -1) | np.any(parameters != 0.0, axis=1))):
            raise ValueError("Unconstrained vertices name no source row or parameters.")
        if np.any(~unconstrained & ((rows < 0) | (rows >= limits))):
            raise ValueError("Exact PLC witnesses index undeclared source rows.")
        if np.any((strata == _SEGMENT) & (parameters[:, 1] != 0.0)):
            raise ValueError("A segment witness has exactly one parameter.")
        for vertex in range(count):
            if strata[vertex] != _NONE:
                parameter_row = parameters[vertex]
                weights = tuple(Fraction(float(parameter_row[axis])) for axis in range(2))
                if min(weights) < 0 or sum(weights, Fraction(0)) > 1:
                    raise ValueError(
                        "Exact PLC witness parameters must lie in the closed source entity."
                    )
        for name, value in (
            ("maximum_work", maximum_work),
            ("maximum_bits", maximum_bits),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        budget = _ExactPlcBudget(maximum_work, maximum_bits)
        for table in (triangles, segments):
            for row_index in range(table.shape[0]):
                row = table[row_index]
                corners = tuple(_point(points[row[axis]]) for axis in range(row.shape[0]))
                budget.charge(64, tuple(value for corner in corners for value in corner))
                _in_entity(corners[0], corners)
        self.source_points = jnp.asarray(points)
        self.source_triangles = jnp.asarray(triangles)
        self.source_segments = jnp.asarray(segments)
        self.source_triangle_ids = jnp.asarray(triangle_ids)
        self.source_triangle_bounds = jnp.asarray(triangle_bounds)
        self.source_segment_ids = jnp.asarray(segment_ids)
        self.source_segment_bounds = jnp.asarray(segment_bounds)
        self.vertex_strata = jnp.asarray(strata.astype(np.int8))
        self.vertex_rows = jnp.asarray(rows)
        self.vertex_parameters = jnp.asarray(parameters)
        self.domain_source_id, self.domain_source_revision = identity, revision
        self.maximum_work, self.maximum_bits = maximum_work, maximum_bits

    def _with_witnesses(
        self,
        strata: ArrayLike,
        rows: ArrayLike,
        parameters: ArrayLike,
        /,
    ) -> ExactPlcCellGeometrySource:
        """Renew vertex ancestry without changing any original row authority."""
        return ExactPlcCellGeometrySource(
            self.source_points,
            self.source_triangles,
            self.source_segments,
            strata,
            rows,
            parameters,
            domain_source_id=self.domain_source_id,
            domain_source_revision=self.domain_source_revision,
            source_triangle_ids=self.source_triangle_ids,
            source_triangle_bounds=self.source_triangle_bounds,
            source_segment_ids=self.source_segment_ids,
            source_segment_bounds=self.source_segment_bounds,
            maximum_work=self.maximum_work,
            maximum_bits=self.maximum_bits,
        )

    def _prepare_witness_locator(
        self, budget: _ExactPlcBudget, /
    ) -> _PreparedWitnessLocator:
        """Prepare one bounded immutable index of the complete original source."""
        return _prepare_source_witness_locator(self, budget)

    def _locate_witness(
        self,
        point: ExactPoint,
        budget: _ExactPlcBudget,
        /,
        *,
        locator: _PreparedWitnessLocator | None = None,
    ) -> tuple[int, int, tuple[float, float], ExactPoint]:
        """Construct a closed row witness and its canonical source coordinate."""
        from .._meshcore import charge_native_geometry_queries

        charge_native_geometry_queries(1, work_units=0)
        if locator is not None and type(locator) is not _PreparedWitnessLocator:
            raise ValueError(
                "PLC witness lookup requires its source-owned prepared locator."
            )
        if locator is None:

            def rows() -> Iterator[tuple[int, int, tuple[ExactPoint, ...]]]:
                points = np.asarray(self.source_points)
                for stratum, table in (
                    (_SEGMENT, self.source_segments),
                    (_TRIANGLE, self.source_triangles),
                ):
                    for row, indices in enumerate(np.asarray(table)):
                        budget.charge(16)
                        yield (
                            stratum,
                            row,
                            tuple(_point(points[int(index)]) for index in indices),
                        )

            candidates = rows()
        else:
            candidates = locator.candidates(self, point, budget)
        for stratum, row, corners in candidates:
            budget.charge(
                128, point + tuple(value for corner in corners for value in corner)
            )
            if not _in_entity(point, corners):
                continue
            weights = _row_parameters(point, corners)
            parameters = (
                float(weights[0]),
                0.0 if len(weights) == 1 else float(weights[1]),
            )
            exact = _witness_point(
                corners, tuple(Fraction(value) for value in parameters[: len(weights)])
            )
            carrier = (
                Fraction(float(exact[0])),
                Fraction(float(exact[1])),
                Fraction(float(exact[2])),
            )
            authority = carrier if _in_entity(carrier, corners) else exact
            return stratum, row, parameters, (authority[0], authority[1], authority[2])
        carrier = (
            Fraction(float(point[0])),
            Fraction(float(point[1])),
            Fraction(float(point[2])),
        )
        return _NONE, -1, (0.0, 0.0), carrier

    @property
    def source_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "exact-plc-cell-source",
                "domain_source_id": self.domain_source_id,
                "domain_source_revision": self.domain_source_revision,
                "source_arrays": array_tree_fingerprint(
                    (
                        self.source_points,
                        self.source_triangles,
                        self.source_segments,
                        self.source_triangle_ids,
                        self.source_triangle_bounds,
                        self.source_segment_ids,
                        self.source_segment_bounds,
                        self.vertex_strata,
                        self.vertex_rows,
                        self.vertex_parameters,
                    )
                ),
                "maximum_work": self.maximum_work,
                "maximum_bits": self.maximum_bits,
            }
        )

    def require_domain(
        self,
        vertices: ArrayLike,
        facets: ArrayLike,
        source_id: str,
        source_revision: str,
        /,
    ) -> None:
        """Bind original PLC identity and revision, bitwise points and facets."""
        points = np.asarray(vertices, dtype=np.float64)
        declared = {
            tuple(sorted(row)) for row in np.asarray(facets, dtype=np.int64).tolist()
        }
        rows = {tuple(sorted(row)) for row in np.asarray(self.source_triangles).tolist()}
        own = np.asarray(self.source_points)
        if (
            source_id != self.domain_source_id
            or source_revision != self.domain_source_revision
            or points.shape != own.shape
            or not np.array_equal(points.view(np.uint64), own.view(np.uint64))
            or not rows <= declared
        ):
            raise ValueError("The exact PLC source does not bind the declared domain.")

    def prepare(self, rounded_vertices: ArrayLike, /) -> PreparedExactPlcGeometry:
        if any(
            isinstance(value, core.Tracer) for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError(
                "Exact PLC preparation is host-only; no source derivative is claimed."
            )
        rounded = np.asarray(rounded_vertices, dtype=np.float64)
        if rounded.shape != (self.vertex_strata.shape[0], 3) or not np.all(
            np.isfinite(rounded)
        ):
            raise ValueError(
                "Exact PLC RNE carrier coordinates must align with vertex witnesses."
            )
        vertices, on_entity, work, bits, error = self._prepare_witnesses(
            rounded,
            np.asarray(self.vertex_strata),
            np.asarray(self.vertex_rows),
            np.asarray(self.vertex_parameters),
        )
        return PreparedExactPlcGeometry(
            vertices, on_entity, self.source_id, work, bits, error
        )

    def _prepare_witnesses(
        self,
        rounded: np.ndarray,
        strata: np.ndarray,
        rows: np.ndarray,
        parameter_rows: np.ndarray,
        /,
    ) -> tuple[tuple[ExactPoint, ...], tuple[bool, ...], int, int, Fraction]:
        """Prove native construction witnesses with the same coordinate authority."""
        if (
            strata.shape != (rounded.shape[0],)
            or rows.shape != strata.shape
            or parameter_rows.shape != (rounded.shape[0], 2)
        ):
            raise ValueError(
                "Exact PLC construction witnesses must align with their carriers."
            )
        budget = _ExactPlcBudget(self.maximum_work, self.maximum_bits)
        from ._coordinate_enclosure import (
            _COORDINATE_BUDGET,
            CoordinateEnclosureResourceError,
        )

        minimum_work = sum(1 if stratum == _NONE else 64 for stratum in strata)
        if minimum_work > self.maximum_work:
            raise CoordinateEnclosureResourceError(
                "coefficient_work", self.maximum_work, minimum_work, 0
            )

        ledger = _COORDINATE_BUDGET.get()
        if ledger is not None:
            ledger.admit_work_bound(minimum_work)
            bank_upper, arithmetic_scratch_upper = _plc_storage_bounds(
                rounded.shape[0], self.maximum_bits
            )
            with ledger.temporary_scope():
                ledger.reserve(0, bank_upper + arithmetic_scratch_upper)
                prepared = self._build_witnesses(
                    rounded, strata, rows, parameter_rows, budget
                )
                # The pre-admitted bank stays live while retention establishes
                # ownership through the rest of the request's theorem.
                ledger.retain_basis((prepared[0], prepared[1]))
                return prepared
        return self._build_witnesses(rounded, strata, rows, parameter_rows, budget)

    def _build_witnesses(
        self,
        rounded: np.ndarray,
        strata: np.ndarray,
        rows: np.ndarray,
        parameter_rows: np.ndarray,
        budget: _ExactPlcBudget,
        /,
    ) -> tuple[tuple[ExactPoint, ...], tuple[bool, ...], int, int, Fraction]:
        points = np.asarray(self.source_points)
        tables = {
            _SEGMENT: np.asarray(self.source_segments),
            _TRIANGLE: np.asarray(self.source_triangles),
        }
        vertices: list[ExactPoint] = []
        on_entity: list[bool] = []
        rounding_error = Fraction(0)
        for vertex in range(rounded.shape[0]):
            carrier_row = rounded[vertex : vertex + 1].reshape((3,))
            stratum = strata[vertex]
            row = rows[vertex]
            parameters = parameter_rows[vertex : vertex + 1].reshape((2,))
            carrier = _point(carrier_row)
            if stratum == _NONE:
                budget.charge(1, carrier)
                vertices.append(carrier)
                on_entity.append(True)
                continue
            source_indices = tables[stratum][row : row + 1].reshape((-1,))
            corners = tuple(
                _point(points[index : index + 1].reshape((3,)))
                for index in source_indices
            )
            weights = tuple(
                Fraction(float(value)) for value in parameters[: len(corners) - 1]
            )
            if any(value < 0 for value in weights) or sum(weights, Fraction(0)) > 1:
                raise ValueError(
                    "Exact PLC witness parameters leave their closed source entity."
                )
            budget.charge(
                64, carrier + tuple(value for corner in corners for value in corner)
            )
            if _in_entity(carrier, corners):
                vertices.append(carrier)
                on_entity.append(True)
                continue
            exact = _witness_point(corners, weights)
            budget.charge(16, exact)
            # Fraction -> float is the correctly rounded (ties-to-even) quotient.
            if any(
                float(value) != float(rne)
                for value, rne in zip(exact, carrier_row, strict=True)
            ):
                raise ValueError(
                    "A PLC carrier is neither on its source entity nor the RNE of its witness."
                )
            declared_bound = (
                self.source_segment_bounds[row]
                if stratum == _SEGMENT
                else self.source_triangle_bounds[row]
            )
            error_squared = sum(
                ((value - rne) ** 2 for value, rne in zip(exact, carrier, strict=True)),
                Fraction(0),
            )
            if error_squared > Fraction(float(declared_bound)) ** 2:
                raise ValueError(
                    "An exact PLC carrier exceeds its declared source row bound."
                )
            rounding_error = max(
                rounding_error,
                *(abs(value - rne) for value, rne in zip(exact, carrier, strict=True)),
            )
            vertices.append(exact)
            on_entity.append(False)
        prepared_vertices = tuple(vertices)
        prepared_on_entity = tuple(on_entity)
        # The caller owns admission and the complete returned bank lifetime.
        return (
            prepared_vertices,
            prepared_on_entity,
            budget.work,
            budget.bits,
            rounding_error,
        )


class ExactPlcCellGeometryConvexSource(StrictModule, NonTrainableState):
    """Exact convex images of original PLC P1 corners, with original SCI binding."""

    parent_mesh: object
    parent_geometry: object
    target_mesh: object
    support_offsets: Array
    support_vertices: Array
    cell_parent_ids: Array
    support_coefficients: tuple[Fraction, ...] = eqx.field(static=True)
    original_binding_id: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)

    def __init__(
        self,
        parent_mesh: CellMesh,
        parent_geometry: CellGeometrySpec,
        support_offsets: ArrayLike,
        support_vertices: ArrayLike,
        support_coefficients: tuple[Fraction, ...],
        target_mesh: CellMesh,
        cell_parent_ids: ArrayLike,
        /,
    ) -> None:
        from ._cell_geometry import _require_full_p1_source, CellGeometrySpec
        from ._cell_mesh import CellMesh

        if not isinstance(parent_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
            raise TypeError(
                "Convex PLC sources require actual parent and target CellMesh owners."
            )
        if not isinstance(parent_geometry, CellGeometrySpec) or not isinstance(
            parent_geometry.exact_source, ExactPlcCellGeometrySource
        ):
            raise TypeError(
                "Convex PLC sources require the original exact PLC geometry binding."
            )
        if (
            parent_geometry.periodic_source is not None
            or parent_mesh.periodic_topology is not None
        ):
            raise ValueError(
                "Convex PLC ancestry requires an original nonperiodic P1 source."
            )
        mapping = dict(
            zip(parent_geometry.block_names, parent_geometry.elements, strict=True)
        )
        if set(mapping) != {block.name for block in parent_mesh.blocks}:
            raise ValueError(
                "Convex PLC parent geometry blocks must bind the original mesh."
            )
        elements = tuple(mapping[block.name] for block in parent_mesh.blocks)
        for block, element in zip(parent_mesh.blocks, elements, strict=True):
            if (
                block.cell_kind != "tetrahedron"
                or _require_full_p1_source(element).cell_kind != "tetrahedron"
            ):
                raise ValueError(
                    "Convex PLC parents require original full P1 tetrahedra."
                )
        raw = tuple(
            np.asarray(value)
            for value in (support_offsets, support_vertices, cell_parent_ids)
        )
        if any(value.ndim != 1 or value.dtype.kind not in "iu" for value in raw):
            raise TypeError(
                "Convex PLC supports and scientific parent IDs require integer vectors."
            )
        if any(np.any(value > np.iinfo(np.int64).max) for value in raw):
            raise ValueError("Convex PLC support indices must fit signed int64.")
        offsets, vertices, parents = (np.asarray(value, dtype=np.int64) for value in raw)
        count = target_mesh.coordinates.shape[0]
        if (
            offsets.shape != (count + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(vertices)
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError(
                "Convex PLC CSR must give every target vertex a nonempty support."
            )
        if np.any(vertices < 0) or np.any(vertices >= parent_mesh.coordinates.shape[0]):
            raise ValueError(
                "Convex PLC supports must index original parent mesh vertex rows."
            )
        if (
            not isinstance(support_coefficients, tuple)
            or len(support_coefficients) != len(vertices)
            or any(type(value) is not Fraction for value in support_coefficients)
        ):
            raise TypeError(
                "Convex PLC coefficients require an aligned exact Fraction tuple."
            )
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            weights = support_coefficients[int(start) : int(stop)]
            if (
                min(weights) <= 0
                or sum(weights, Fraction(0)) != 1
                or len(set(vertices[start:stop].tolist())) != stop - start
            ):
                raise ValueError(
                    "Convex PLC supports require unique rows and strictly positive normalized coefficients."
                )
        if parents.shape != (sum(block.cell_count for block in target_mesh.blocks),):
            raise ValueError(
                "Convex PLC original cell SCI must align with every target cell."
            )
        if any(
            block.cell_kind not in ("tetrahedron", "hexahedron", "pyramid", "polyhedron")
            for block in target_mesh.blocks
        ):
            raise ValueError(
                "Convex PLC targets require canonical degree-one maps or an authenticated cut-polyhedron vertex bank."
            )
        root = parent_geometry.exact_source
        binding = canonical_fingerprint(
            (
                self._binding(parent_mesh, parent_geometry, target_mesh),
                array_tree_fingerprint(
                    (jnp.asarray(offsets), jnp.asarray(vertices), jnp.asarray(parents))
                ),
                tuple(
                    (value.numerator, value.denominator) for value in support_coefficients
                ),
            )
        )
        # Validate all scientific and exact-coordinate invariants before assigning leaves.
        self._prove(
            parent_mesh,
            parent_geometry,
            target_mesh,
            offsets,
            vertices,
            support_coefficients,
            parents,
            root.maximum_work,
            root.maximum_bits,
        )
        self.parent_mesh, self.parent_geometry, self.target_mesh = (
            parent_mesh,
            parent_geometry,
            target_mesh,
        )
        self.support_offsets, self.support_vertices, self.cell_parent_ids = map(
            jnp.asarray, (offsets, vertices, parents)
        )
        self.support_coefficients = support_coefficients
        self.original_binding_id = binding
        self.maximum_work, self.maximum_bits = root.maximum_work, root.maximum_bits

    @staticmethod
    def _binding(
        parent_mesh: CellMesh,
        parent_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
    ) -> str:
        source = parent_geometry.exact_source
        if not isinstance(source, ExactPlcCellGeometrySource):
            raise RuntimeError("Convex PLC parent source changed type.")
        return canonical_fingerprint(
            {
                "kind": "original-plc-parent-cell-reference-authority",
                "arrays": array_tree_fingerprint(
                    (parent_mesh, parent_geometry, target_mesh)
                ),
                "parent_source": source.source_id,
                "parent_geometry_layout": parent_geometry.geometry_layout_id,
                "parent_topology": parent_mesh.topology_id,
                "target_topology": target_mesh.topology_id,
                "parent_blocks": tuple(
                    (block.name, block.cell_kind, block.block_id)
                    for block in parent_mesh.blocks
                ),
                "target_blocks": tuple(
                    (block.name, block.cell_kind, block.block_id)
                    for block in target_mesh.blocks
                ),
            }
        )

    def _owners(self) -> tuple[CellMesh, CellGeometrySpec, CellMesh]:
        from ._cell_geometry import CellGeometrySpec
        from ._cell_mesh import CellMesh

        if (
            not isinstance(self.parent_mesh, CellMesh)
            or not isinstance(self.parent_geometry, CellGeometrySpec)
            or not isinstance(self.target_mesh, CellMesh)
            or not isinstance(
                self.parent_geometry.exact_source, ExactPlcCellGeometrySource
            )
        ):
            raise RuntimeError("Convex PLC source owners changed type.")
        return self.parent_mesh, self.parent_geometry, self.target_mesh

    @property
    def source_id(self) -> str:
        parent_mesh, parent_geometry, target_mesh = self._owners()
        return canonical_fingerprint(
            {
                "kind": "exact-plc-convex-source",
                "graph": self._binding(parent_mesh, parent_geometry, target_mesh),
                "supports": array_tree_fingerprint(
                    (
                        self.support_offsets,
                        self.support_vertices,
                        self.cell_parent_ids,
                    )
                ),
                "coefficients": tuple(
                    (value.numerator, value.denominator)
                    for value in self.support_coefficients
                ),
                "maximum_work": self.maximum_work,
                "maximum_bits": self.maximum_bits,
            }
        )

    @staticmethod
    def _prove(
        parent_mesh: CellMesh,
        parent_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        offsets: np.ndarray,
        vertices: np.ndarray,
        weights: tuple[Fraction, ...],
        parents: np.ndarray,
        maximum_work: int,
        maximum_bits: int,
    ) -> tuple[
        tuple[ExactPoint, ...],
        tuple[tuple[ExactPoint, ...], ...],
        int,
        int,
        Fraction,
    ]:
        from ._coordinate_enclosure import (
            _COORDINATE_BUDGET,
            coordinate_corner_images,
        )

        mapping = dict(
            zip(parent_geometry.block_names, parent_geometry.elements, strict=True)
        )
        route_mapping = dict(
            zip(parent_geometry.block_names, parent_geometry.geometry_dofs, strict=True)
        )
        elements = tuple(mapping[block.name] for block in parent_mesh.blocks)
        routes = tuple(route_mapping[block.name] for block in parent_mesh.blocks)
        root = parent_geometry.exact_source
        if not isinstance(root, ExactPlcCellGeometrySource):
            raise RuntimeError("Convex PLC parent source changed type.")
        parent_proof = root.prepare(parent_geometry.coordinates)
        bank = parent_proof.vertices
        budget = _ExactPlcBudget(maximum_work, maximum_bits)
        # The actual owning root receipt is adopted, never estimated from strata
        # or charged again. This proof invocation owns its real root evaluation.
        budget.work = parent_proof.operation_count
        budget.bits = max(
            (
                max(value.numerator.bit_length(), value.denominator.bit_length())
                for point in bank
                for value in point
            ),
            default=0,
        )
        original = {}
        cells = {}
        cell_vertices = {}
        for block, element, route in zip(
            parent_mesh.blocks, elements, routes, strict=True
        ):
            for sci, row, coefficients in zip(
                np.asarray(block.global_ids),
                np.asarray(block.vertices),
                np.asarray(route),
                strict=True,
            ):
                images = coordinate_corner_images(
                    element, tuple(bank[int(index)] for index in coefficients)
                )
                if images is None or len(images) != 4:
                    raise ValueError(
                        "Original PLC geometry has no complete exact P1 corner law."
                    )
                carrier = np.asarray(parent_mesh.coordinates, dtype=np.float64)
                for vertex, image in zip(row, images, strict=True):
                    if not np.array_equal(
                        carrier[int(vertex)].view(np.uint64),
                        np.asarray(
                            [float(value) for value in image], dtype=np.float64
                        ).view(np.uint64),
                    ):
                        raise ValueError(
                            "Original PLC corner law does not bind the original mesh RNE carrier."
                        )
                cells[int(sci)] = images
                cell_vertices[int(sci)] = tuple(int(vertex) for vertex in row)
                for vertex, image in zip(row, images, strict=True):
                    point = tuple(image)
                    if int(vertex) in original and original[int(vertex)] != point:
                        raise ValueError(
                            "Original PLC corner laws disagree at a shared vertex."
                        )
                    original[int(vertex)] = point
        ledger = _COORDINATE_BUDGET.get()

        def build() -> tuple[
            tuple[ExactPoint, ...],
            tuple[tuple[ExactPoint, ...], ...],
            int,
            int,
            Fraction,
        ]:
            from .._meshcore import charge_native_geometry_queries

            parent_frames: dict[
                int, tuple[ExactMatrix, dict[int, int], ExactMatrix | None]
            ] = {}
            reference_cache: dict[tuple[int, int], ExactPoint] = {}
            points: list[ExactPoint] = []
            for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
                charge_native_geometry_queries(1, work_units=0)
                indices = vertices[int(start) : int(stop)]
                coefficients = weights[int(start) : int(stop)]
                if any(int(index) not in original for index in indices):
                    raise ValueError(
                        "Convex PLC support names an unused original vertex."
                    )
                operands = coefficients + tuple(
                    value for index in indices for value in original[int(index)]
                )
                budget.charge(12 * len(indices), operands)
                point: ExactPoint = (
                    sum(
                        (
                            coefficient * original[int(index)][0]
                            for index, coefficient in zip(
                                indices, coefficients, strict=True
                            )
                        ),
                        Fraction(),
                    ),
                    sum(
                        (
                            coefficient * original[int(index)][1]
                            for index, coefficient in zip(
                                indices, coefficients, strict=True
                            )
                        ),
                        Fraction(),
                    ),
                    sum(
                        (
                            coefficient * original[int(index)][2]
                            for index, coefficient in zip(
                                indices, coefficients, strict=True
                            )
                        ),
                        Fraction(),
                    ),
                )
                budget.charge(3, point)
                points.append(point)
            numeric = np.asarray(target_mesh.coordinates, dtype=np.float64)
            rne = np.asarray(
                [[float(value) for value in point] for point in points], dtype=np.float64
            )
            if not np.array_equal(numeric.view(np.uint64), rne.view(np.uint64)):
                raise ValueError(
                    "Convex PLC target coordinates are not bitwise RNE of original exact images."
                )
            references: list[tuple[ExactPoint, ...]] = []
            cursor = 0
            for block in target_mesh.blocks:
                for row, active in zip(
                    np.asarray(block.vertices),
                    np.asarray(block.vertex_valid),
                    strict=True,
                ):
                    row = row[active]
                    sci = int(parents[cursor])
                    cursor += 1
                    if sci not in cells:
                        raise ValueError(
                            "Convex PLC target names an absent original global cell SCI."
                        )
                    corners = cells[sci]
                    if sci not in parent_frames:
                        budget.charge(
                            9, tuple(value for corner in corners for value in corner)
                        )
                        matrix = tuple(
                            tuple(
                                corners[column + 1][axis] - corners[0][axis]
                                for column in range(3)
                            )
                            for axis in range(3)
                        )
                        a, b, c = tuple(
                            tuple(matrix[axis][column] for axis in range(3))
                            for column in range(3)
                        )
                        budget.charge(14, a + b + c)
                        determinant = (
                            a[0] * (b[1] * c[2] - b[2] * c[1])
                            - a[1] * (b[0] * c[2] - b[2] * c[0])
                            + a[2] * (b[0] * c[1] - b[1] * c[0])
                        )
                        budget.charge(0, (determinant,))
                        if not determinant:
                            raise ValueError(
                                "Convex PLC original source cell has deficient exact rank."
                            )
                        parent_frames[sci] = (
                            matrix,
                            {
                                vertex: index
                                for index, vertex in enumerate(cell_vertices[sci])
                            },
                            None,
                        )
                    matrix, parent_rows, inverse = parent_frames[sci]
                    cell_refs: list[ExactPoint] = []
                    for vertex in row:
                        vertex = int(vertex)
                        key = (vertex, sci)
                        budget.charge(1)
                        reference = reference_cache.get(key)
                        if reference is None:
                            charge_native_geometry_queries(1, work_units=0)
                            start, stop = int(offsets[vertex]), int(offsets[vertex + 1])
                            supports, coefficients = (
                                vertices[start:stop],
                                weights[start:stop],
                            )
                            budget.charge(len(supports))
                            ordinals = tuple(
                                parent_rows.get(int(index)) for index in supports
                            )
                            if all(index is not None for index in ordinals):
                                values = [Fraction(0)] * 3
                                for ordinal, coefficient in zip(
                                    ordinals, coefficients, strict=True
                                ):
                                    if ordinal:
                                        budget.charge(1)
                                        values[ordinal - 1] += coefficient
                                reference = (values[0], values[1], values[2])
                            else:
                                if inverse is None:
                                    inverse = _convex_parent_inverse(matrix, budget)
                                    parent_frames[sci] = (matrix, parent_rows, inverse)
                                point = points[vertex]
                                budget.charge(3, point)
                                delta = tuple(
                                    point[axis] - corners[0][axis] for axis in range(3)
                                )
                                budget.charge(15, delta)
                                reference = (
                                    sum(
                                        (
                                            entry * value
                                            for entry, value in zip(
                                                inverse[0], delta, strict=True
                                            )
                                        ),
                                        Fraction(),
                                    ),
                                    sum(
                                        (
                                            entry * value
                                            for entry, value in zip(
                                                inverse[1], delta, strict=True
                                            )
                                        ),
                                        Fraction(),
                                    ),
                                    sum(
                                        (
                                            entry * value
                                            for entry, value in zip(
                                                inverse[2], delta, strict=True
                                            )
                                        ),
                                        Fraction(),
                                    ),
                                )
                            budget.charge(3, reference)
                            if min(reference) < 0 or sum(reference, Fraction(0)) > 1:
                                raise ValueError(
                                    "Convex PLC target full map leaves its declared original P1 simplex."
                                )
                            reference_cache[key] = reference
                        cell_refs.append(reference)
                    # Canonical Q1/pyramid maps and the convex hull of a clipped
                    # polyhedron use nonnegative unit vertex weights. These
                    # original references authenticate the whole source range;
                    # polyhedron topology/convexity is independently certified.
                    from itertools import combinations

                    rank = False
                    for indices in combinations(range(1, len(cell_refs)), 3):
                        a, b, c = (
                            tuple(
                                cell_refs[index][axis] - cell_refs[0][axis]
                                for axis in range(3)
                            )
                            for index in indices
                        )
                        budget.charge(32, a + b + c)
                        determinant = (
                            a[0] * (b[1] * c[2] - b[2] * c[1])
                            - a[1] * (b[0] * c[2] - b[2] * c[0])
                            + a[2] * (b[0] * c[1] - b[1] * c[0])
                        )
                        if determinant:
                            rank = True
                            break
                    if not rank:
                        raise ValueError(
                            "Convex PLC target corner law has deficient exact rank."
                        )
                    rounded_corners = tuple(
                        _point(numeric[int(vertex)]) for vertex in row
                    )
                    rounded_rank = False
                    for indices in combinations(range(1, len(rounded_corners)), 3):
                        a, b, c = (
                            tuple(
                                rounded_corners[index][axis] - rounded_corners[0][axis]
                                for axis in range(3)
                            )
                            for index in indices
                        )
                        budget.charge(32, a + b + c)
                        determinant = (
                            a[0] * (b[1] * c[2] - b[2] * c[1])
                            - a[1] * (b[0] * c[2] - b[2] * c[0])
                            + a[2] * (b[0] * c[1] - b[1] * c[0])
                        )
                        if determinant:
                            rounded_rank = True
                            break
                    if not rounded_rank:
                        raise ValueError(
                            "Convex PLC RNE target carrier has deficient rank."
                        )
                    references.append(tuple(cell_refs))
            error = max(
                (
                    abs(value - Fraction(float(value)))
                    for point in points
                    for value in point
                ),
                default=Fraction(0),
            )
            return tuple(points), tuple(references), budget.work, budget.bits, error

        if ledger is None:
            return build()
        with ledger.temporary_scope():
            bank_upper, scratch = _plc_storage_bounds(
                len(vertices) + len(parents) * 8, maximum_bits
            )
            ledger.reserve(0, bank_upper + scratch)
            result = build()
            ledger.retain_basis((result[0], result[1]))
            return result

    def _proof(
        self,
    ) -> tuple[
        tuple[ExactPoint, ...],
        tuple[tuple[ExactPoint, ...], ...],
        int,
        int,
        Fraction,
    ]:
        from ._cell_geometry import CellGeometrySpec
        from ._cell_mesh import CellMesh

        parent_mesh = self.parent_mesh
        parent_geometry = self.parent_geometry
        target_mesh = self.target_mesh
        if (
            not isinstance(parent_mesh, CellMesh)
            or not isinstance(target_mesh, CellMesh)
            or not isinstance(parent_geometry, CellGeometrySpec)
            or not isinstance(parent_geometry.exact_source, ExactPlcCellGeometrySource)
        ):
            raise RuntimeError("Convex PLC source owners changed type.")
        root = parent_geometry.exact_source
        if (self.maximum_work, self.maximum_bits) != (
            root.maximum_work,
            root.maximum_bits,
        ):
            raise ValueError(
                "Convex PLC source cannot reset its original resource allowance."
            )
        current = canonical_fingerprint(
            (
                self._binding(parent_mesh, parent_geometry, target_mesh),
                array_tree_fingerprint(
                    (
                        self.support_offsets,
                        self.support_vertices,
                        self.cell_parent_ids,
                    )
                ),
                tuple(
                    (value.numerator, value.denominator)
                    for value in self.support_coefficients
                ),
            )
        )
        if current != self.original_binding_id:
            raise ValueError(
                "Convex PLC original geometry or carrier topology binding changed."
            )
        return self._prove(
            parent_mesh,
            parent_geometry,
            target_mesh,
            np.asarray(self.support_offsets),
            np.asarray(self.support_vertices),
            self.support_coefficients,
            np.asarray(self.cell_parent_ids),
            self.maximum_work,
            self.maximum_bits,
        )

    @property
    def cell_parent_reference_corners(
        self,
    ) -> tuple[tuple[ExactPoint, ...], ...]:
        return self._proof()[1]

    def prepare(self, rounded_vertices: ArrayLike, /) -> PreparedExactPlcGeometry:
        from ._cell_mesh import CellMesh

        if not isinstance(self.target_mesh, CellMesh):
            raise RuntimeError("Convex PLC target mesh owner changed type.")
        rounded = np.asarray(rounded_vertices, dtype=np.float64)
        target = np.asarray(self.target_mesh.coordinates, dtype=np.float64)
        if rounded.shape != target.shape or not np.array_equal(
            rounded.view(np.uint64), target.view(np.uint64)
        ):
            raise ValueError(
                "Convex PLC preparation requires its bound target RNE carrier."
            )
        points, _, work, bits, error = self._proof()
        return PreparedExactPlcGeometry(
            points,
            tuple(False for _ in points),
            self.source_id,
            work,
            bits,
            error,
        )


def _plc_storage_bounds(point_count: int, maximum_bits: int, /) -> tuple[int, int]:
    """Live 3-D rational-bank and arithmetic upper bounds before construction."""
    # Binary64 rationals can require 1075 bits before the source bit limit is
    # checked; determinants and squared errors can grow to eight times that.
    allocation_bits = max(1075, maximum_bits)
    integer_upper = 64 + 4 * ((allocation_bits + 29) // 30)
    fraction_upper = 128 + 2 * integer_upper
    bank_upper = 512 + point_count * (192 + 3 * fraction_upper)
    scratch_integer_upper = 64 + 4 * ((8 * allocation_bits + 29) // 30)
    return bank_upper, 64 * (128 + 2 * scratch_integer_upper)


def _point(row: np.ndarray) -> ExactPoint:
    return (Fraction(float(row[0])), Fraction(float(row[1])), Fraction(float(row[2])))


def _witness_point(
    corners: tuple[ExactPoint, ...], weights: tuple[Fraction, ...], /
) -> ExactPoint:
    def coordinate(axis: int) -> Fraction:
        return corners[0][axis] + sum(
            (
                weight * (corner[axis] - corners[0][axis])
                for weight, corner in zip(weights, corners[1:], strict=True)
            ),
            Fraction(0),
        )

    return (coordinate(0), coordinate(1), coordinate(2))


def _row_parameters(
    point: ExactPoint, corners: tuple[ExactPoint, ...], /
) -> tuple[Fraction, ...]:
    from ._coordinate_enclosure import _solve_exact

    edges = tuple(
        tuple(corner[axis] - corners[0][axis] for axis in range(3))
        for corner in corners[1:]
    )
    delta = tuple(point[axis] - corners[0][axis] for axis in range(3))
    matrix = [
        [
            sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))
            for second in edges
        ]
        for first in edges
    ]
    right = [
        [sum((a * b for a, b in zip(edge, delta, strict=True)), Fraction(0))]
        for edge in edges
    ]
    return tuple(row[0] for row in _solve_exact(matrix, right))


def _in_entity(point: ExactPoint, corners: tuple[ExactPoint, ...], /) -> bool:
    """Exact rational membership in a closed segment or triangle (no tolerance)."""
    origin = corners[0]
    offset = tuple(point[axis] - origin[axis] for axis in range(3))
    edges = tuple(
        tuple(corner[axis] - origin[axis] for axis in range(3)) for corner in corners[1:]
    )
    if len(edges) == 1:
        (direction,) = edges
        if not any(direction):
            raise ValueError("An exact PLC source segment is degenerate.")
        cross = (
            offset[1] * direction[2] - offset[2] * direction[1],
            offset[2] * direction[0] - offset[0] * direction[2],
            offset[0] * direction[1] - offset[1] * direction[0],
        )
        if any(cross):
            return False
        along = sum(a * b for a, b in zip(offset, direction, strict=True))
        return 0 <= along <= sum(value * value for value in direction)
    first, second = edges
    normal = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    if not any(normal):
        raise ValueError("An exact PLC source triangle is degenerate.")
    if sum(a * b for a, b in zip(offset, normal, strict=True)) != 0:
        return False
    # Coplanar: exact 2-D orientations in the projection dropping the dominant
    # normal axis; inside the closed triangle iff no edge sees it on the
    # opposite side of the triangle's own orientation.
    axis = max(range(3), key=lambda k: abs(normal[k]))
    x, y = (axis + 1) % 3, (axis + 2) % 3
    sign = 1 if normal[axis] > 0 else -1
    for start, stop in (
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[0]),
    ):
        turn = (stop[x] - start[x]) * (point[y] - start[y]) - (stop[y] - start[y]) * (
            point[x] - start[x]
        )
        if turn * sign < 0:
            return False
    return True


class _ExactPlcBudget:
    def __init__(self, maximum_work: int, maximum_bits: int, /) -> None:
        self.maximum_work, self.maximum_bits = maximum_work, maximum_bits
        self.work, self.bits = 0, 0

    def charge(self, work: int, values: tuple[Fraction, ...] = (), /) -> None:
        self.work += work
        if values:
            self.bits = max(
                self.bits,
                *(
                    max(value.numerator.bit_length(), value.denominator.bit_length())
                    for value in values
                ),
            )
        from ._coordinate_enclosure import CoordinateEnclosureResourceError

        if self.work > self.maximum_work:
            raise CoordinateEnclosureResourceError(
                "coefficient_work",
                self.maximum_work,
                self.work,
                self.work - work,
            )
        if self.bits > self.maximum_bits:
            raise ValueError(
                "Exact PLC source exceeds its declared maximum integer bits."
            )
        from ._coordinate_enclosure import _COORDINATE_BUDGET

        ledger = _COORDINATE_BUDGET.get()
        if ledger is not None:
            ledger.reserve(work)
            ledger.charge_native_work(work)
            return
        from .._meshcore import current_native_execution_budget

        execution = current_native_execution_budget()
        if execution is not None:
            execution.charge(work=work)


__all__ = [
    "ExactPlcCellGeometrySource",
    "ExactPlcCellGeometryConvexSource",
    "PreparedExactPlcGeometry",
]


def _convex_parent_inverse(
    matrix: tuple[tuple[Fraction, ...], ...], budget: _ExactPlcBudget
) -> tuple[tuple[Fraction, ...], ...]:
    """Prepare one original-parent inverse with its actual canonical work receipt."""
    import sys

    from ..linalg._small_batched import prepare_exact_small_linear_actions
    from ._coordinate_enclosure import coordinate_enclosure_budget

    ledger = coordinate_enclosure_budget(
        max(0, budget.maximum_work - budget.work), sys.maxsize
    )
    start = ledger.work_units
    try:
        with (
            ledger.activate(),
            ledger.bound_stage(
                max(0, budget.maximum_work - budget.work),
                ledger.maximum_memory_bytes,
                starting_work_units=start,
            ),
            ledger.temporary_scope(),
        ):
            prepared = prepare_exact_small_linear_actions(
                matrix,
                tuple(
                    tuple(Fraction(int(row == column)) for column in range(3))
                    for row in range(3)
                ),
                coordinate_budget=ledger,
            )
    finally:
        work = ledger.work_units - start
        budget.work += work
        ledger.charge_native_work(work)
    if prepared.actions is None:
        raise ValueError(
            "Convex PLC original parent reference action has deficient exact rank."
        )
    budget.charge(0, tuple(value for row in prepared.actions for value in row))
    return prepared.actions
