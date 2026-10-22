#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Define native PLC source coordinates from closed row constructions.

A native PLC construction names a source edge or triangle and two parameters
for every constrained vertex in ambient dimension two or three. Optional
reduced integer ratios define rational parameters; binary64 parameters are
their rounded display. Ratio-backed constructed points define the coordinates.
Without ratios, a carrier in its closed row entity defines its coordinates;
otherwise it must equal the coordinate-wise RNE of the constructed point.
Ratio-backed carriers need not be RNE, but their rational squared Euclidean
deviation must not exceed the declared row bound squared. A zero bound
requires equality with the source coordinates. Unconstrained vertices define
their own coordinates.
"""

from __future__ import annotations

import functools as functools_owner
import itertools as itertools_owner
import math as math_owner
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from typing import final, TYPE_CHECKING
from weakref import ReferenceType, WeakKeyDictionary, WeakValueDictionary, ref

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core
from jax.typing import ArrayLike

from phydrax.linalg._exact import solve_exact

from .. import _bvh as bvh_owner
from .. import _geometry_predicates as geometry_predicates_owner
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._issued_proofs import IssuedProofs
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, positive_integer
from ..geometry._rational import rational_vector
from ..linalg import _small_batched as small_batched_owner
from ..linalg._small_batched import exact_cofactor_determinant_3
from ..typing import checked
from . import (
    _cell_geometry as cell_geometry_owner,
    _cell_mesh as cell_mesh_owner,
    _coordinate_budget as coordinate_budget_owner,
    _coordinate_enclosure as coordinate_enclosure_owner,
)
from ._coordinate_budget import CoordinateEnclosureBudget
from ._exact_algebraic_geometry import rational_coefficient


if TYPE_CHECKING:
    from .._bvh import PackedBVH
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh


type ExactPoint = tuple[Fraction, ...]
type ExactMatrix = tuple[tuple[Fraction, ...], ...]

_NONE, _SEGMENT, _TRIANGLE = 0, 1, 2

@dataclass(frozen=True, slots=True, eq=False)
class _PlcEntityFrame:
    """Actual exact edge/normal operands of one source entity, not a verdict."""

    corners: tuple[ExactPoint, ...]
    edges: tuple[ExactPoint, ...]
    normal: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class _PreparedPlcEntityFrames:
    """Nonportable numerical rows; caller-bound issued operands authorize reuse."""

    triangles: tuple[_PlcEntityFrame, ...]
    segments: tuple[_PlcEntityFrame, ...]
    storage_bytes_upper: int

    def __reduce__(self) -> tuple[object, ...]:
        raise TypeError("PLC entity frames must re-earn at archive and cold boundaries.")


# Verification-only weak producer seals. There is no source-ID lookup or
# cross-request numerical reuse here, and a seal never retains its root key.
@dataclass(frozen=True, slots=True)
class _PlcEntityFrameSeal:
    source: ReferenceType[ExactPlcCellGeometrySource]
    arrays: tuple[Array, ...]
    policy: tuple[str, str, int, int]
    numeric_fields: tuple[object, ...]


_ENTITY_FRAME_SEALS: WeakKeyDictionary[
    _PreparedPlcEntityFrames, _PlcEntityFrameSeal
] = WeakKeyDictionary()

_ENTITY_SOURCE_ARRAY_NAMES = (
    "source_points", "source_triangles", "source_segments",
    "source_triangle_ids", "source_triangle_bounds",
    "source_segment_ids", "source_segment_bounds",
)


def _entity_frame_fields(frames: _PreparedPlcEntityFrames, /) -> tuple[object, ...]:
    fields: list[object] = [
        frames.triangles, frames.segments, frames.storage_bytes_upper
    ]
    for frame in (*frames.triangles, *frames.segments):
        if type(frame) is not _PlcEntityFrame:
            raise ValueError("PLC entity frames must retain their actual numeric row type.")
        fields.extend((frame, frame.corners, frame.edges, frame.normal))
    return tuple(fields)


def _plc_entity_frame_basis(
    frames: _PreparedPlcEntityFrames, /, *,
    source: ExactPlcCellGeometrySource | None = None,
) -> tuple[object, ...]:
    """Authenticate actual numeric rows and their original constructor source."""
    seal = _ENTITY_FRAME_SEALS.get(frames) if type(frames) is _PreparedPlcEntityFrames else None
    if seal is None or any(
        current is not expected
        for current, expected in zip(
            (frames.triangles, frames.segments, frames.storage_bytes_upper),
            seal.numeric_fields[:3], strict=True,
        )
    ):
        raise ValueError("PLC entity preparation must retain its actual producer binding.")
    actual_source = seal.source()
    if (
        actual_source is None or (source is not None and actual_source is not source)
        or any(
            getattr(actual_source, name) is not expected or expected.is_deleted()
            for name, expected in zip(_ENTITY_SOURCE_ARRAY_NAMES, seal.arrays, strict=True)
        )
        or (
            actual_source.domain_source_id, actual_source.domain_source_revision,
            actual_source.maximum_work, actual_source.maximum_bits,
        ) != seal.policy
    ):
        raise ValueError("PLC entity preparation must bind its actual original source and policy.")
    actual = _entity_frame_fields(frames)
    if len(actual) != len(seal.numeric_fields) or any(
        current is not expected
        for current, expected in zip(actual, seal.numeric_fields, strict=True)
    ):
        raise ValueError("PLC entity preparation numeric rows changed after their earning operation.")
    return actual


def _require_plc_entity_frame_source(
    frames: _PreparedPlcEntityFrames, source: ExactPlcCellGeometrySource, /
) -> None:
    _plc_entity_frame_basis(frames, source=source)


def _issue_plc_entity_frames(
    source: ExactPlcCellGeometrySource,
    triangles: tuple[_PlcEntityFrame, ...],
    segments: tuple[_PlcEntityFrame, ...],
    storage_bytes_upper: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> _PreparedPlcEntityFrames:
    # The graph and seal storage were admitted before the earning row
    # operations. Only the actual small producer-index work is added here.
    budget.reserve(3 + 4 * (len(triangles) + len(segments)))
    frames = _PreparedPlcEntityFrames(triangles, segments, storage_bytes_upper)
    _ENTITY_FRAME_SEALS[frames] = _PlcEntityFrameSeal(
        ref(source),
        tuple(getattr(source, name) for name in _ENTITY_SOURCE_ARRAY_NAMES),
        (
            source.domain_source_id, source.domain_source_revision,
            source.maximum_work, source.maximum_bits,
        ),
        _entity_frame_fields(frames),
    )
    return frames


def _plc_entity_frame_wrapper_storage_upper(row_count: int, /) -> int:
    """Bound new wrapper/source seal/index metadata around already-owned rows."""
    return 4096 + 128 * row_count


def _plc_entity_frame_storage_upper(
    triangle_count: int, segment_count: int, dimension: int, maximum_bits: int, /
) -> int:
    # Inputs are normalized binary64 before row preparation. Only valid rows
    # survive their unchanged maximum_bits guard; differences and normal minors
    # need at most eight input numerator/denominator bit extents plus carries.
    allocation_bits = 8 * min(1075, maximum_bits) + 16
    integer_upper = 64 + 4 * ((allocation_bits + 29) // 30)
    fraction_upper = 128 + 2 * integer_upper
    triangle_scalars = 5 * dimension + (1 if dimension == 2 else 3)
    segment_scalars = 3 * dimension
    return (
        4096
        + triangle_count * (1024 + triangle_scalars * fraction_upper)
        + segment_count * (1024 + segment_scalars * fraction_upper)
    )


def _plc_entity_numeric_frame(
    corners: tuple[ExactPoint, ...], dimension: int, /
) -> _PlcEntityFrame:
    origin = corners[0]
    edges = tuple(
        tuple(corner[axis] - origin[axis] for axis in range(dimension))
        for corner in corners[1:]
    )
    if len(edges) == 1:
        normal: tuple[Fraction, ...] = ()
    else:
        first, second = edges
        normal = (
            (first[0] * second[1] - first[1] * second[0],)
            if dimension == 2 else (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            )
        )
    return _PlcEntityFrame(corners, edges, normal)




_WITNESS_LOCATORS: WeakValueDictionary[int, _PreparedWitnessLocator] = (
    WeakValueDictionary()
)
# Issued witness banks, weakly keyed by their source object (see ``IssuedProofs``).
_ISSUED_WITNESSES: IssuedProofs[
    ExactPlcCellGeometrySource, PreparedExactPlcGeometry
] = IssuedProofs()


def _witness_storage(size: int, *, budget: CoordinateEnclosureBudget) -> None:
    """Reserve host proof storage before allocating it."""
    budget.reserve(0, size)


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


@dataclass(frozen=True, slots=True)
class _WitnessPackedSnapshot:
    """Immutable host banks of one prepared PLC row hierarchy."""

    bbox_min: np.ndarray
    bbox_max: np.ndarray
    left: np.ndarray
    right: np.ndarray
    leaf_id: np.ndarray
    leaf_items: np.ndarray
    item_bbox_min: np.ndarray
    item_bbox_max: np.ndarray


def _witness_packed_snapshot(
    hierarchy: PackedBVH,
    exact_work: ExactPlcBudget,
    *,
    budget: CoordinateEnclosureBudget,
) -> _WitnessPackedSnapshot:
    """Return immutable host views of the packed hierarchy."""
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
        exact_work.charge(value.size)
        _witness_storage(256 + 2 * value.size * value.dtype.itemsize, budget=budget)
        host = np.asarray(value)
        snapshots.append(
            np.frombuffer(host.tobytes(), dtype=host.dtype).reshape(host.shape)
        )
    return _WitnessPackedSnapshot(
        snapshots[0],
        snapshots[1],
        snapshots[2],
        snapshots[3],
        snapshots[4],
        snapshots[5],
        snapshots[6],
        snapshots[7],
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _PreparedWitnessLocator:
    """Index closed source-row bounds for request-scoped PLC point construction."""

    owner: ExactPlcCellGeometrySource
    leaves: tuple[Array, ...]
    metadata: tuple[str, str, int, int]
    source_id: str
    rows: tuple[tuple[int, int, tuple[ExactPoint, ...]], ...]
    normals: tuple[ExactPoint | None, ...]
    packed: _WitnessPackedSnapshot | None
    hierarchy: PackedBVH | None

    def candidates(
        self,
        source: ExactPlcCellGeometrySource,
        point: ExactPoint,
        exact_work: ExactPlcBudget,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> Iterator[tuple[int, int, tuple[ExactPoint, ...]]]:
        exact_work.charge(16, point)
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
                "ExactPlcCellGeometrySource.locator must bind the current source leaves and metadata; got a mismatched locator"
            )
        lower, upper = [], []
        for value in point:
            exact_work.charge(4, (value,))
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
        _witness_storage(128, budget=budget)
        packed = self.packed
        if packed is None:
            _witness_storage(0, budget=budget)
            return
        pending = [0]
        candidates = []
        while pending:
            exact_work.charge(8)
            node = pending.pop()
            node_lower = packed.bbox_min
            node_upper = packed.bbox_max
            left = packed.left
            right = packed.right
            leaf_id = packed.leaf_id
            leaf_items = packed.leaf_items
            item_lower = packed.item_bbox_min
            item_upper = packed.item_bbox_max
            if any(
                node_upper[node, axis] < lower[axis]
                or node_lower[node, axis] > upper[axis]
                for axis in range(len(point))
            ):
                continue
            leaf = int(leaf_id[node])
            if leaf >= 0:
                for item in leaf_items[leaf]:
                    exact_work.charge(8)
                    row = int(item)
                    if row < 0:
                        continue
                    if any(
                        item_upper[row, axis] < lower[axis]
                        or item_lower[row, axis] > upper[axis]
                        for axis in range(len(point))
                    ):
                        continue
                    _witness_storage(32, budget=budget)
                    candidates.append(row)
            else:
                _witness_storage(16, budget=budget)
                pending.extend((int(left[node]), int(right[node])))
        # Segment rows precede triangle rows, independent of tree order.

        def compare(a: int, b: int) -> int:
            exact_work.charge(1)
            return (a > b) - (a < b)

        _witness_storage(64 * len(candidates), budget=budget)
        candidates.sort(key=functools_owner.cmp_to_key(compare))
        for index in candidates:
            normal = self.normals[index]
            if normal is not None:
                # A noncoplanar point cannot belong to this closed triangle.
                # The source-owned normal was prepared once, rather than
                # rebuilt by every full triangle membership predicate.
                exact_work.charge(9, point + normal)
                origin = self.rows[index][2][0]
                if sum(
                    (
                        normal[axis] * (point[axis] - origin[axis])
                        for axis in range(3)
                    ),
                    Fraction(0),
                ):
                    continue
            yield self.rows[index]


def _prepare_witness_normal(
    corners: tuple[ExactPoint, ...],
    maximum_ratio_bits: int,
    exact_work: ExactPlcBudget,
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactPoint | None:
    """Prepare an exact plane only when the original rational-bit cap admits it."""
    # Each normal component subtracts products of differences of source
    # rationals. This bound also covers arbitrary denominator products, not
    # only the dyadic denominators of the binary64 source.
    bits = 8 * maximum_ratio_bits + 16
    if len(corners) != 3 or len(corners[0]) != 3 or bits > exact_work.maximum_bits:
        return None
    storage, _ = plc_storage_bounds(3, bits)
    _witness_storage(storage, budget=budget)
    # Six edge differences, six products, three differences, three zero tests.
    exact_work.charge(18)
    origin, first, second = corners
    a = tuple(first[axis] - origin[axis] for axis in range(3))
    b = tuple(second[axis] - origin[axis] for axis in range(3))
    normal = (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
    exact_work.charge(0, normal)
    # Preserve the existing membership predicate's degeneracy refusal point.
    return normal if any(normal) else None


def _prepare_source_witness_locator(
    source: ExactPlcCellGeometrySource,
    exact_work: ExactPlcBudget,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PreparedWitnessLocator:
    leaves = _witness_source_leaves(source)
    for leaf in leaves:
        exact_work.charge(int(leaf.size))
        _witness_storage(int(leaf.size) * int(leaf.dtype.itemsize), budget=budget)
    identity = source.source_id
    points = np.asarray(source.source_points)
    rows, normals, boxes = [], [], []

    for stratum, table in (
        (_SEGMENT, source.source_segments),
        (_TRIANGLE, source.source_triangles),
    ):
        for row, indices in enumerate(np.asarray(table)):
            exact_work.charge(16)
            # Admit scan scratch before inspecting the float exponents;
            # then use the PLC rational-bank bound before conversion.
            _witness_storage(512, budget=budget)
            maximum_ratio_bits = 1
            for index in indices:
                for coordinate in points[int(index)]:
                    exact_work.charge(1)
                    _, exponent = math_owner.frexp(float(coordinate))
                    maximum_ratio_bits = max(
                        maximum_ratio_bits, 53, exponent, 54 - exponent
                    )
            bank_upper, _ = plc_storage_bounds(len(indices), maximum_ratio_bits)
            _witness_storage(bank_upper, budget=budget)
            corners = tuple(rational_vector(points[int(index)]) for index in indices)
            exact_work.charge(0, tuple(value for corner in corners for value in corner))
            normals.append(
                _prepare_witness_normal(
                    corners, maximum_ratio_bits, exact_work, budget=budget
                )
                if stratum == _TRIANGLE
                else None
            )
            lower = tuple(
                min(float(corner[axis]) for corner in corners)
                for axis in range(points.shape[1])
            )
            upper = tuple(
                max(float(corner[axis]) for corner in corners)
                for axis in range(points.shape[1])
            )
            rows.append((stratum, row, corners))
            boxes.append((lower, upper))

    hierarchy, packed = None, None
    if rows:
        exact_work.charge(6 * len(rows))
        _witness_storage(256 + 64 * len(rows), budget=budget)
        lower = np.asarray([box[0] for box in boxes], dtype=np.float64)
        upper = np.asarray([box[1] for box in boxes], dtype=np.float64)

        def reserve_storage(size: int, /) -> None:
            _witness_storage(size, budget=budget)

        hierarchy = bvh_owner.prepare_bvh(
            lower,
            upper,
            dtype=np.float64,
            _charge_work=exact_work.charge,
            _reserve_storage=reserve_storage,
        )
        packed = _witness_packed_snapshot(hierarchy, exact_work, budget=budget)
    _witness_storage(256 + 64 * len(rows), budget=budget)
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
        tuple(normals),
        packed,
        hierarchy,
    )
    _WITNESS_LOCATORS[id(locator)] = locator
    return locator


def _canonical_parameter_ratios(
    ratios: tuple[tuple[tuple[int, int], tuple[int, int]], ...] | None,
    parameters: np.ndarray,
    /,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...] | None:
    """Validate rounded display weights and reduce host integer parameter ratios."""
    if ratios is None:
        return None
    if len(ratios) != parameters.shape[0]:
        raise ValueError(
            f"ExactPlcCellGeometrySource.vertex_parameter_ratios must align per vertex; got {len(ratios)} rows"
        )
    result: list[tuple[tuple[int, int], tuple[int, int]]] = []
    records: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ] = {}
    for vertex, row in enumerate(ratios):
        if len(row) != 2:
            raise ValueError(
                f"ExactPlcCellGeometrySource.vertex_parameter_ratios must contain two weights per row; got {len(row)}"
            )
        canonical: list[tuple[int, int]] = []
        for axis, pair in enumerate(row):
            if len(pair) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) for value in pair
            ):
                raise ValueError(
                    f"ExactPlcCellGeometrySource.vertex_parameter_ratios must contain integer numerator/denominator pairs; got {pair}"
                )
            numerator, denominator = pair
            if denominator == 0:
                raise ValueError(
                    f"ExactPlcCellGeometrySource.vertex_parameter_ratios must have nonzero denominators; got {pair}"
                )
            value = Fraction(numerator, denominator)
            if float(value) != float(parameters[vertex, axis]):
                raise ValueError(
                    f"ExactPlcCellGeometrySource.vertex_parameter_ratios must round to vertex_parameters; got row {vertex}, weight {axis}"
                )
            canonical.append((value.numerator, value.denominator))
        normalized = canonical[0], canonical[1]
        # Integer ratio records are immutable metadata. Share identical rows
        # after every row's original type, reduction and display checks.
        result.append(records.setdefault(normalized, normalized))
    return tuple(result)


def _validate_plc_witness_rows(
    strata: np.ndarray,
    rows: np.ndarray,
    parameters: np.ndarray,
    segment_count: int,
    triangle_count: int,
    vertex_parameter_ratios: tuple[tuple[tuple[int, int], tuple[int, int]], ...] | None,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...] | None:
    """Validate stratum routing and closed-entity barycentric weights."""
    count = strata.shape[0]
    ratios = _canonical_parameter_ratios(vertex_parameter_ratios, parameters)
    limits = np.where(
        strata == _SEGMENT,
        segment_count,
        np.where(strata == _TRIANGLE, triangle_count, 0),
    )
    unconstrained = strata == _NONE
    if np.any((strata < _NONE) | (strata > _TRIANGLE)):
        raise ValueError(
            "ExactPlcCellGeometrySource.vertex_strata must contain only 0, 1, or 2; got an unsupported stratum"
        )
    if np.any(unconstrained & ((rows != -1) | np.any(parameters != 0.0, axis=1))):
        raise ValueError(
            "ExactPlcCellGeometrySource.vertex_rows must be -1 with zero parameters for unconstrained vertices; got a named row or nonzero weight"
        )
    if ratios is not None and any(
        strata[vertex] == _NONE and any(numerator != 0 for numerator, _ in pair)
        for vertex, pair in enumerate(ratios)
    ):
        raise ValueError(
            "ExactPlcCellGeometrySource.vertex_parameter_ratios must be zero for unconstrained vertices; got nonzero weight"
        )
    if np.any(~unconstrained & ((rows < 0) | (rows >= limits))):
        raise ValueError(
            "ExactPlcCellGeometrySource.vertex_rows must index declared source rows; got an undeclared row"
        )
    if np.any((strata == _SEGMENT) & (parameters[:, 1] != 0.0)):
        raise ValueError(
            "ExactPlcCellGeometrySource.vertex_parameters must have zero second segment weight; got a nonzero second weight"
        )
    for vertex in range(count):
        if strata[vertex] != _NONE:
            parameter_row = parameters[vertex]
            weights = (
                tuple(Fraction(*ratio) for ratio in ratios[vertex])
                if ratios is not None
                else tuple(Fraction(float(parameter_row[axis])) for axis in range(2))
            )
            if min(weights) < 0 or sum(weights, Fraction(0)) > 1:
                raise ValueError(
                    "ExactPlcCellGeometrySource.vertex_parameters must lie in the closed source entity; got weights outside its barycentric domain"
                )
            if strata[vertex] == _SEGMENT and weights[1] != 0:
                raise ValueError(
                    f"ExactPlcCellGeometrySource.vertex_parameter_ratios must have zero second segment weight; got {weights[1]}"
                )
    return ratios


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PreparedExactPlcGeometry:
    """Return PLC rational coordinates and bounded preparation evidence.

    ``on_entity`` marks vertices whose carrier defines their source coordinates
    (on-entity carriers without ratio parameters, or unconstrained vertices).
    ``maximum_rounding_error`` is the largest coordinate distance between a
    source coordinate and its carrier.
    """

    vertices: tuple[ExactPoint, ...]
    on_entity: tuple[bool, ...]
    source_id: str
    operation_count: int
    maximum_integer_bits: int
    maximum_rounding_error: Fraction


@final
class ExactPlcWitnessProof(StrictModule, NonTrainableState):
    """Return rational carrier coordinates and measured preparation consumption."""

    vertices: tuple[ExactPoint, ...] = eqx.field(static=True)
    on_entity: tuple[bool, ...] = eqx.field(static=True)
    operation_count: int = eqx.field(static=True)
    maximum_integer_bits: int = eqx.field(static=True)
    maximum_rounding_error: Fraction = eqx.field(static=True)


@final
class ConvexPlcProof(StrictModule, NonTrainableState):
    """Return rational convex vertices, parent references, and proof consumption."""

    vertices: tuple[ExactPoint, ...] = eqx.field(static=True)
    parent_reference_corners: tuple[tuple[ExactPoint, ...], ...] = eqx.field(static=True)
    operation_count: int = eqx.field(static=True)
    maximum_integer_bits: int = eqx.field(static=True)
    maximum_rounding_error: Fraction = eqx.field(static=True)


@final
class ExactPlcCellGeometrySource(StrictModule, NonTrainableState):
    """Bind PLC source rows and per-vertex constructions to a native carrier.

    ``source_points`` (P, D), D in {2, 3}, and ``source_triangles`` (R, 3)
    describe the PLC domain vertices and triangles; ``source_segments`` (Q, 2)
    describes its edges. ``vertex_strata`` (0 unconstrained, 1 segment row,
    2 triangle row), ``vertex_rows``, and ``vertex_parameters`` (N, 2) bind
    each carrier vertex; a segment's second parameter is zero.
    ``domain_source_id`` and ``domain_source_revision`` bind the source identity.
    Optional ``vertex_parameter_ratios`` stores two reduced integer ratios per
    vertex as host-static metadata; its constructed point takes precedence
    over carrier membership. Carriers satisfy the Euclidean row bound using
    exact arithmetic; a zero bound requires equality with the source coordinate.
    Preparation is host-only and charged to the coordinate-enclosure and
    native execution budgets.
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
    vertex_parameter_ratios: (
        tuple[tuple[tuple[int, int], tuple[int, int]], ...] | None
    ) = eqx.field(static=True)
    domain_source_id: str = eqx.field(static=True)
    domain_source_revision: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)

    @checked
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
        vertex_parameter_ratios: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
        | None = None,
        maximum_work: int = 10_000_000,
        maximum_bits: int = 4096,
        budget: CoordinateEnclosureBudget,
        _entity_origin: ExactPlcCellGeometrySource | None = None,
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
            raise TypeError(
                "ExactPlcCellGeometrySource.indices must have integer dtype; got a noninteger array"
            )
        if any(np.any(value > np.iinfo(np.int64).max) for value in raw):
            raise ValueError(
                "ExactPlcCellGeometrySource.indices must fit signed int64; got an out-of-range index or scientific ID"
            )
        triangles, segments, strata, rows, triangle_ids, segment_ids = (
            np.asarray(value, dtype=np.int64) for value in raw
        )
        triangle_bounds = np.asarray(source_triangle_bounds, dtype=np.float64)
        segment_bounds = np.asarray(source_segment_bounds, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (2, 3) or not points.shape[0]:
            raise ValueError(
                f"ExactPlcCellGeometrySource.source_points must be nonempty (P, 2) or (P, 3); got {points.shape}"
            )
        if (
            triangles.ndim != 2
            or triangles.shape[1] != 3
            or segments.ndim != 2
            or segments.shape[1] != 2
        ):
            raise ValueError(
                "ExactPlcCellGeometrySource.source_rows must contain triangles (R, 3) and segments (Q, 2); got incompatible row shapes"
            )
        points = points.reshape((points.shape[0], points.shape[1]))
        triangles = triangles.reshape((triangles.shape[0], 3))
        segments = segments.reshape((segments.shape[0], 2))
        for table in (triangles, segments):
            if np.any(table < 0) or np.any(table >= points.shape[0]):
                raise ValueError(
                    "ExactPlcCellGeometrySource.source_rows must index declared source points; got an undeclared point index"
                )
        for table, identifiers, bounds in (
            (triangles, triangle_ids, triangle_bounds),
            (segments, segment_ids, segment_bounds),
        ):
            if (
                identifiers.shape != (table.shape[0],)
                or bounds.shape != identifiers.shape
            ):
                raise ValueError(
                    "ExactPlcCellGeometrySource.source_row_ids must align with row bounds and source rows; got incompatible bank shapes"
                )
            if (
                np.any(identifiers < 0)
                or not np.all(np.isfinite(bounds))
                or np.any(bounds < 0)
            ):
                raise ValueError(
                    "ExactPlcCellGeometrySource.source_row_bounds must be finite and nonnegative with nonnegative row IDs; got a negative ID or invalid bound"
                )
        if strata.ndim != 1:
            raise ValueError(
                "ExactPlcCellGeometrySource.vertex_strata must have rank one; got another rank"
            )
        count = strata.shape[0]
        if rows.shape != (count,) or parameters.shape != (count, 2):
            raise ValueError(
                "ExactPlcCellGeometrySource.vertex_rows must align with (N, 2) parameters per carrier vertex; got incompatible bank shapes"
            )
        strata = strata.reshape((count,))
        rows = rows.reshape((count,))
        parameters = parameters.reshape((count, 2))
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(parameters)):
            raise ValueError(
                "ExactPlcCellGeometrySource.source_points must be finite with finite vertex parameters; got a nonfinite value"
            )
        ratios = _validate_plc_witness_rows(
            strata,
            rows,
            parameters,
            segments.shape[0],
            triangles.shape[0],
            vertex_parameter_ratios,
        )
        maximum_work = positive_integer(maximum_work, "maximum_work")
        maximum_bits = positive_integer(maximum_bits, "maximum_bits")
        from ._coordinate_operands import (
            _plc_entity_frames_for,
            _record_plc_entity_frames,
        )

        prepared_entities = _plc_entity_frames_for(
            _entity_origin,
            (points, triangles, segments, triangle_ids, triangle_bounds, segment_ids, segment_bounds),
            (identity, revision, maximum_work, maximum_bits),
            budget,
        )
        capture = budget.capture_coordinate_operands
        exact_work = ExactPlcBudget(maximum_work, maximum_bits, budget=budget)
        with budget.live_storage() as credit, budget.temporary_scope():
            if capture:
                if prepared_entities is None:
                    storage_upper = _plc_entity_frame_storage_upper(
                        triangles.shape[0], segments.shape[0], points.shape[1], maximum_bits
                    )
                    credit.set_bound(storage_upper)
                    # Temporary input ratios/rank arithmetic may reach binary64's
                    # full exponent range before the unchanged operand-bit guard.
                    row_storage, row_scratch = plc_storage_bounds(3, 1075)
                    budget.reserve(0, row_storage + row_scratch)
                else:
                    storage_upper = prepared_entities.storage_bytes_upper
                    credit.set_bound(_plc_entity_frame_wrapper_storage_upper(
                        triangles.shape[0] + segments.shape[0]
                    ))
            triangle_frames: list[_PlcEntityFrame] = []
            segment_frames: list[_PlcEntityFrame] = []
            if prepared_entities is None:
                for table, target in (
                    (triangles, triangle_frames), (segments, segment_frames)
                ):
                    for row_index in range(table.shape[0]):
                        row = table[row_index]
                        corners = tuple(
                            rational_vector(points[row[axis]]) for axis in range(row.shape[0])
                        )
                        # This remains the original WHOLE-ROW tariff, including
                        # local/enclosure work and every original operand-bit guard.
                        exact_work.charge(
                            64, tuple(value for corner in corners for value in corner)
                        )
                        frame = _plc_entity_numeric_frame(corners, points.shape[1])
                        _in_entity_frame(corners[0], frame)
                        if capture:
                            target.append(frame)
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
            ) = jax.device_put(
                (
                    points, triangles, segments, triangle_ids, triangle_bounds,
                    segment_ids, segment_bounds, strata.astype(np.int8), rows, parameters,
                )
            )
            self.vertex_parameter_ratios = ratios
            self.domain_source_id, self.domain_source_revision = identity, revision
            self.maximum_work, self.maximum_bits = maximum_work, maximum_bits
            if capture:
                # A new source receives a new constructor-bound wrapper, even
                # when its immutable numerical entity rows are already owned.
                frames = _issue_plc_entity_frames(
                    self,
                    tuple(triangle_frames) if prepared_entities is None else prepared_entities.triangles,
                    tuple(segment_frames) if prepared_entities is None else prepared_entities.segments,
                    storage_upper,
                    budget,
                )
                _record_plc_entity_frames(
                    self, frames, budget, temporary_storage_credit=credit
                )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ExactPlcCellGeometrySource],
        value: ExactPlcCellGeometrySource,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> ExactPlcCellGeometrySource:
        """Rebuild PLC source fields under their bounded constructor admission."""
        points = np.asarray(value.source_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(
                "ExactPlcCellGeometrySource.source_points must be nonempty "
                f"(P, 2) or (P, 3); got {points.shape}"
            )
        if budget is None:
            work_upper, bytes_upper = cell_geometry_owner.exact_source_preparation_bounds(
                value, value.vertex_strata.shape[0], points.shape[1]
            )
            preparation = cell_geometry_owner.owned_coordinate_preparation(
                work_upper,
                bytes_upper,
                maximum_geometry_queries=value.preparation_query_upper(),
            )
        else:
            preparation = nullcontext(budget)
        with preparation as admission:
            return cls(
                value.source_points,
                value.source_triangles,
                value.source_segments,
                value.vertex_strata,
                value.vertex_rows,
                value.vertex_parameters,
                domain_source_id=value.domain_source_id,
                domain_source_revision=value.domain_source_revision,
                source_triangle_ids=value.source_triangle_ids,
                source_triangle_bounds=value.source_triangle_bounds,
                source_segment_ids=value.source_segment_ids,
                source_segment_bounds=value.source_segment_bounds,
                vertex_parameter_ratios=value.vertex_parameter_ratios,
                maximum_work=value.maximum_work,
                maximum_bits=value.maximum_bits,
                budget=admission,
            )

    def exact_vertex_parameters(self) -> tuple[tuple[Fraction, Fraction], ...]:
        """Return two rational source-row weights per carrier vertex."""
        if self.vertex_parameter_ratios is not None:
            return tuple(
                (Fraction(*first), Fraction(*second))
                for first, second in self.vertex_parameter_ratios
            )
        return tuple(
            (Fraction(float(row[0])), Fraction(float(row[1])))
            for row in np.asarray(self.vertex_parameters)
        )

    @checked
    def with_witnesses(
        self,
        strata: ArrayLike,
        rows: ArrayLike,
        parameters: ArrayLike,
        /,
        *,
        vertex_parameter_ratios: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
        | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> ExactPlcCellGeometrySource:
        """Renew vertex ancestry while retaining source row identities and bounds."""
        if vertex_parameter_ratios is None and self.vertex_parameter_ratios is not None:
            if not (
                np.array_equal(np.asarray(strata), np.asarray(self.vertex_strata))
                and np.array_equal(np.asarray(rows), np.asarray(self.vertex_rows))
            ):
                raise ValueError(
                    "ExactPlcCellGeometrySource.vertex_parameter_ratios must follow changed ancestry explicitly; got a renewal without rational ratios."
                )
            vertex_parameter_ratios = self.vertex_parameter_ratios
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
            vertex_parameter_ratios=vertex_parameter_ratios,
            maximum_work=self.maximum_work,
            maximum_bits=self.maximum_bits,
            budget=budget,
            _entity_origin=self,
        )

    @checked
    def prepare_witness_locator(
        self,
        exact_work: ExactPlcBudget,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> _PreparedWitnessLocator:
        """Prepare a bounded immutable index of every segment and triangle row."""
        return _prepare_source_witness_locator(self, exact_work, budget=budget)

    @checked
    def locate_witness(
        self,
        point: ExactPoint,
        exact_work: ExactPlcBudget,
        /,
        *,
        locator: _PreparedWitnessLocator | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[
        int,
        int,
        tuple[float, float],
        ExactPoint,
        tuple[tuple[int, int], tuple[int, int]] | None,
    ]:
        """Return a closed source-row construction and its rational coordinates."""

        budget.reserve_geometry_queries(1)
        if locator is not None and type(locator) is not _PreparedWitnessLocator:
            raise ValueError(
                "ExactPlcCellGeometrySource.locator must be prepared by this source; got a mismatched locator"
            )
        if locator is None:

            def rows() -> Iterator[tuple[int, int, tuple[ExactPoint, ...]]]:
                points = np.asarray(self.source_points)
                for stratum, table in (
                    (_SEGMENT, self.source_segments),
                    (_TRIANGLE, self.source_triangles),
                ):
                    for row, indices in enumerate(np.asarray(table)):
                        exact_work.charge(16)
                        yield (
                            stratum,
                            row,
                            tuple(
                                rational_vector(points[int(index)]) for index in indices
                            ),
                        )

            candidates = rows()
        else:
            candidates = locator.candidates(self, point, exact_work, budget=budget)
        for stratum, row, corners in candidates:
            exact_work.charge(
                128, point + tuple(value for corner in corners for value in corner)
            )
            if not in_entity(point, corners):
                continue
            weights = _row_parameters(point, corners)
            parameters = (
                float(weights[0]),
                0.0 if len(weights) == 1 else float(weights[1]),
            )
            if self.vertex_parameter_ratios is not None:
                second = weights[1] if len(weights) == 2 else Fraction(0)
                ratios = (
                    (weights[0].numerator, weights[0].denominator),
                    (second.numerator, second.denominator),
                )
                exact_work.charge(16, weights)
                return stratum, row, parameters, point, ratios
            exact = _witness_point(
                corners, tuple(Fraction(value) for value in parameters[: len(weights)])
            )
            carrier = tuple(Fraction(float(value)) for value in exact)
            authority = carrier if in_entity(carrier, corners) else exact
            return stratum, row, parameters, authority, None
        carrier = tuple(Fraction(float(value)) for value in point)
        return (
            _NONE,
            -1,
            (0.0, 0.0),
            carrier,
            ((0, 1), (0, 1)) if self.vertex_parameter_ratios is not None else None,
        )

    @checked
    def admits_located_witness(
        self,
        point: ExactPoint,
        exact_work: ExactPlcBudget,
        /,
        *,
        locator: _PreparedWitnessLocator,
        budget: CoordinateEnclosureBudget,
    ) -> bool:
        """Whether ``point`` binds a source row whose original bound admits its carrier.

        Applies the nested producer's own row location, round-to-nearest carrier
        of the located authority and declared row-bound admission. A point
        outside every segment and triangle binds no source row and is refused.
        """
        stratum, row, parameters, authority, ratios = self.locate_witness(
            point, exact_work, locator=locator, budget=budget
        )
        if stratum == _NONE:
            return False
        if stratum == _SEGMENT:
            indices, bound = self.source_segments[row], self.source_segment_bounds[row]
        else:
            indices, bound = (
                self.source_triangles[row],
                self.source_triangle_bounds[row],
            )
        points = np.asarray(self.source_points)
        corners = tuple(
            rational_vector(points[int(index)]) for index in np.asarray(indices)
        )
        weights = (
            tuple(Fraction(*ratio) for ratio in ratios[: len(corners) - 1])
            if ratios is not None
            else tuple(Fraction(value) for value in parameters[: len(corners) - 1])
        )
        carrier = tuple(Fraction(float(value)) for value in authority)
        exact_work.charge(
            64, carrier + tuple(value for corner in corners for value in corner)
        )
        return not isinstance(
            _row_witness(
                carrier,
                corners,
                weights,
                float(bound),
                exact_work,
                rational_weights=ratios is not None,
            ),
            str,
        )

    @property
    def source_id(self) -> str:
        payload = {
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
        if self.vertex_parameter_ratios is not None:
            return canonical_fingerprint(
                {**payload, "vertex_parameter_ratios": self.vertex_parameter_ratios}
            )
        return canonical_fingerprint(payload)

    def require_domain(
        self,
        vertices: ArrayLike,
        facets: ArrayLike,
        source_id: str,
        source_revision: str,
        /,
    ) -> None:
        """Bind PLC identity, revision, bitwise points, and declared facets."""
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
            raise ValueError(
                "ExactPlcCellGeometrySource.domain_source_id must bind the declared revision, points, and facets; got a mismatched domain binding"
            )

    def preparation_query_upper(self) -> int:
        """Return the row preparation query bound; locator queries are separate."""
        return 0

    @checked
    def prepare(
        self, rounded_vertices: ArrayLike, /, *, budget: CoordinateEnclosureBudget
    ) -> PreparedExactPlcGeometry:
        """Prepare source-row coordinates once per request and carrier contents.

        A bank already issued in this process for this source object and these
        carrier coordinates is reused while another owner still retains it.
        """
        if any(
            isinstance(value, core.Tracer) for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError(
                "ExactPlcCellGeometrySource.prepare must receive host source leaves; got a tracer"
            )
        rounded = np.asarray(rounded_vertices, dtype=np.float64)
        if rounded.shape != (
            self.vertex_strata.shape[0],
            self.source_points.shape[1],
        ) or not np.all(np.isfinite(rounded)):
            raise ValueError(
                "ExactPlcCellGeometrySource.rounded_vertices must be finite and align with vertex rows; got incompatible coordinates"
            )

        ledger = budget
        identity = self.source_id
        key = (identity, canonical_fingerprint(array_tree_fingerprint(rounded)))
        cached = ledger.plc_witness_cache.get(key)
        if cached is not None:
            ledger.reserve(0)
            return cached
        issued_key = "/".join(key)
        prepared = _ISSUED_WITNESSES.get(self, issued_key)
        if prepared is None:
            proof = self.prepare_witnesses(
                rounded,
                np.asarray(self.vertex_strata),
                np.asarray(self.vertex_rows),
                np.asarray(self.vertex_parameters),
                vertex_parameter_ratios=self.vertex_parameter_ratios,
                budget=budget,
            )
            prepared = PreparedExactPlcGeometry(
                proof.vertices,
                proof.on_entity,
                identity,
                proof.operation_count,
                proof.maximum_integer_bits,
                proof.maximum_rounding_error,
            )
            _ISSUED_WITNESSES.record(self, issued_key, prepared)
        # A bank issued for this source object and still owned elsewhere is not
        # proved again; every request admits and retains the bank it holds.
        ledger.retain_basis((key, prepared))
        ledger.plc_witness_cache[key] = prepared
        return prepared

    def _binary64_memberships(
        self, rounded: np.ndarray, strata: np.ndarray, rows: np.ndarray,
        exact_work: ExactPlcBudget,
    ) -> np.ndarray:
        """Prove closed row membership by the canonical exact binary predicates."""
        budget = exact_work.enclosure
        budget.reserve(0, 1024 * rounded.shape[0] + 256)
        proved = np.zeros((rounded.shape[0],), dtype=np.bool_)
        points = np.asarray(self.source_points)
        if rounded.dtype != np.float64 or points.dtype != np.float64:
            return proved
        mode = geometry_predicates_owner.PredicateMode.EXACT
        ambient = rounded.shape[1]
        if ambient not in (2, 3):
            return proved
        axes = ((0, 1),) if ambient == 2 else ((0, 1), (1, 2), (2, 0))
        for stratum, table in (
            (_SEGMENT, np.asarray(self.source_segments)),
            (_TRIANGLE, np.asarray(self.source_triangles)),
        ):
            selected = np.flatnonzero(
                (strata == stratum) & (rows >= 0) & (rows < table.shape[0])
            )
            if not selected.size:
                continue
            corners = points[table[rows[selected]]]
            carriers = rounded[selected]
            if stratum == _SEGMENT:
                inside = (
                    np.any(corners[:, 0] != corners[:, 1], axis=1)
                    & np.all(carriers >= np.minimum(corners[:, 0], corners[:, 1]), axis=1)
                    & np.all(carriers <= np.maximum(corners[:, 0], corners[:, 1]), axis=1)
                )
                for projection in axes:
                    exact_work.charge(selected.size)
                    signs = geometry_predicates_owner.orient2d_host(
                        corners[:, 0, projection], corners[:, 1, projection],
                        carriers[:, projection], mode=mode, budget=budget,
                    )
                    inside &= signs.certain & (signs.signs == 0)
                proved[selected] = inside
                continue
            coplanar = np.ones((selected.size,), dtype=np.bool_)
            if ambient == 3:
                exact_work.charge(selected.size)
                signs = geometry_predicates_owner.orient3d_host(
                    corners[:, 0], corners[:, 1], corners[:, 2], carriers,
                    mode=mode, budget=budget,
                )
                coplanar = signs.certain & (signs.signs == 0)
            pending = np.flatnonzero(coplanar)
            for projection in axes:
                if not pending.size:
                    break
                projected = corners[pending][:, :, projection]
                exact_work.charge(pending.size)
                orientation = geometry_predicates_owner.orient2d_host(
                    projected[:, 0], projected[:, 1], projected[:, 2],
                    mode=mode, budget=budget,
                )
                usable = orientation.certain & (orientation.signs != 0)
                active = pending[usable]
                if not active.size:
                    continue
                triangle = projected[usable]
                point = carriers[active][:, projection]
                inside = np.ones((active.size,), dtype=np.bool_)
                for first, second in ((0, 1), (1, 2), (2, 0)):
                    exact_work.charge(active.size)
                    side = geometry_predicates_owner.orient2d_host(
                        triangle[:, first], triangle[:, second], point,
                        mode=mode, budget=budget,
                    )
                    inside &= side.certain & (
                        side.signs * orientation.signs[usable] >= 0
                    )
                proved[selected[active]] = inside
                pending = pending[~usable]
        return proved

    @checked
    def prepare_witnesses(
        self,
        rounded: np.ndarray,
        strata: np.ndarray,
        rows: np.ndarray,
        parameter_rows: np.ndarray,
        /,
        *,
        vertex_parameter_ratios: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
        | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> ExactPlcWitnessProof:
        """Validate row constructions and return their rational coordinates."""
        if (
            rounded.ndim != 2
            or rounded.shape[1] != self.source_points.shape[1]
            or strata.shape != (rounded.shape[0],)
            or rows.shape != strata.shape
            or parameter_rows.shape != (rounded.shape[0], 2)
        ):
            raise ValueError(
                "ExactPlcCellGeometrySource.vertex_rows must align with construction carrier coordinates and parameters; got incompatible bank shapes"
            )
        vertex_parameter_ratios = _canonical_parameter_ratios(
            vertex_parameter_ratios, parameter_rows
        )
        exact_work = ExactPlcBudget(self.maximum_work, self.maximum_bits, budget=budget)

        ledger = budget

        bank_upper, arithmetic_scratch_upper = plc_storage_bounds(
            rounded.shape[0], self.maximum_bits
        )
        with ledger.temporary_scope(), ledger.live_storage() as scalar_storage:
            ledger.reserve(
                0, bank_upper + arithmetic_scratch_upper + rounded.shape[0] + 256
            )
            with ledger.temporary_scope():
                on_entity = (
                    self._binary64_memberships(rounded, strata, rows, exact_work)
                    if vertex_parameter_ratios is None
                    else np.zeros((rounded.shape[0],), dtype=np.bool_)
                )
            minimum_work = sum(
                1 if stratum == _NONE or proved else 64
                for stratum, proved in zip(strata, on_entity, strict=True)
            )
            if exact_work.work + minimum_work > self.maximum_work:
                raise coordinate_budget_owner.CoordinateEnclosureResourceError(
                    "coefficient_work", self.maximum_work,
                    exact_work.work + minimum_work, exact_work.work,
                )
            ledger.admit_work_bound(minimum_work)
            prepared = self._build_witnesses(
                rounded, strata, rows, parameter_rows, exact_work, scalar_storage,
                vertex_parameter_ratios, on_entity,
            )
            # The pre-admitted bank stays live while retention establishes
            # ownership through the rest of the request's theorem.
            ledger.retain_basis(
                (prepared.vertices, prepared.on_entity), operand=True,
                _kernel_operands=True,
            )
            return prepared

    def _build_witnesses(
        self,
        rounded: np.ndarray,
        strata: np.ndarray,
        rows: np.ndarray,
        parameter_rows: np.ndarray,
        exact_work: ExactPlcBudget,
        /,
        scalar_storage: coordinate_budget_owner.CoordinateLiveStorage,
        vertex_parameter_ratios: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
        | None = None,
        on_entity_proof: np.ndarray | None = None,
    ) -> ExactPlcWitnessProof:
        points = np.asarray(self.source_points)
        tables = {
            _SEGMENT: np.asarray(self.source_segments),
            _TRIANGLE: np.asarray(self.source_triangles),
        }
        bounds = {
            _SEGMENT: np.asarray(self.source_segment_bounds, dtype=np.float64),
            _TRIANGLE: np.asarray(self.source_triangle_bounds, dtype=np.float64),
        }
        vertices: list[ExactPoint] = []
        on_entity: list[bool] = []
        rounding_error = Fraction(0)
        # The kernel's exact scalar bank factors repeated lattice coordinates.
        # Only immutable Fraction values are shared; carriers and source IDs,
        # row witnesses and their original arithmetic charges remain unchanged.
        float_scalars: dict[float, Fraction] = {}
        exact_scalars: dict[tuple[int, int], Fraction] = {}
        source_corners: dict[int, ExactPoint] = {}
        scalar_storage.set_bound(1024)

        def reserve_scalar_slots() -> None:
            scalar_storage.set_bound(
                1024
                + 384 * (len(float_scalars) + len(exact_scalars) + 2)
                + 256 * (len(source_corners) + 1)
            )

        def exact_scalar(value: Fraction) -> Fraction:
            exact_work.enclosure.reserve(1)
            key = value.numerator, value.denominator
            cached = exact_scalars.get(key)
            if cached is None:
                reserve_scalar_slots()
                exact_scalars[key] = value
                return value
            return cached

        def lift(values: np.ndarray) -> ExactPoint:
            lifted = []
            for value in values:
                represented = float(value)
                exact_work.enclosure.reserve(1)
                cached = float_scalars.get(represented)
                if cached is None:
                    reserve_scalar_slots()
                    cached = exact_scalar(Fraction(represented))
                    float_scalars[represented] = cached
                lifted.append(cached)
            return tuple(lifted)

        def source_corner(index: int) -> ExactPoint:
            exact_work.enclosure.reserve(1)
            cached = source_corners.get(index)
            if cached is None:
                reserve_scalar_slots()
                cached = lift(points[index])
                source_corners[index] = cached
            return cached

        for vertex in range(rounded.shape[0]):
            carrier_row = rounded[vertex]
            stratum = strata[vertex]
            row = rows[vertex]
            parameters = parameter_rows[vertex : vertex + 1].reshape((2,))
            carrier = lift(carrier_row)
            if stratum == _NONE:
                exact_work.charge(1, carrier)
                vertices.append(carrier)
                on_entity.append(True)
                continue
            source_indices = tables[stratum][row : row + 1].reshape((-1,))
            corners = tuple(source_corner(int(index)) for index in source_indices)
            weights = (
                tuple(
                    Fraction(*ratio)
                    for ratio in vertex_parameter_ratios[vertex][: len(corners) - 1]
                )
                if vertex_parameter_ratios is not None
                else tuple(
                    Fraction(float(value)) for value in parameters[: len(corners) - 1]
                )
            )
            if any(value < 0 for value in weights) or sum(weights, Fraction(0)) > 1:
                raise ValueError(
                    "ExactPlcCellGeometrySource.vertex_parameters must lie in the closed source entity; got weights outside its barycentric domain"
                )
            if on_entity_proof is not None and on_entity_proof[vertex]:
                exact_work.charge(1, carrier)
                vertices.append(carrier)
                on_entity.append(True)
                continue
            exact_work.charge(
                64, carrier + tuple(value for corner in corners for value in corner)
            )
            witness = _row_witness(
                carrier,
                corners,
                weights,
                float(bounds[stratum][row]),
                exact_work,
                rational_weights=vertex_parameter_ratios is not None,
            )
            if isinstance(witness, str):
                raise ValueError(witness)
            exact, carrier_on_entity = witness
            if not carrier_on_entity:
                rounding_error = max(
                    rounding_error,
                    *(abs(value - rne) for value, rne in zip(exact, carrier, strict=True)),
                )
            vertices.append(tuple(exact_scalar(value) for value in exact))
            on_entity.append(carrier_on_entity)
        prepared_vertices = tuple(vertices)
        prepared_on_entity = tuple(on_entity)
        # The caller owns admission and the returned bank lifetime.
        return ExactPlcWitnessProof(
            vertices=prepared_vertices,
            on_entity=prepared_on_entity,
            operation_count=exact_work.work,
            maximum_integer_bits=exact_work.bits,
            maximum_rounding_error=rounding_error,
        )


@checked
def translated_plc_layout_matches(
    before: cell_geometry_owner.CellGeometrySpec,
    after: cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    translation: np.ndarray | None = None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove unchanged PLC coordinate structure under a certified source translation."""
    match before.exact_source, after.exact_source:
        case (
            ExactPlcCellGeometrySource() as original,
            ExactPlcCellGeometrySource() as actual,
        ):
            pass
        case _:
            return False
    if (
        before.block_names != after.block_names
        or tuple(element.element_id for element in before.elements)
        != tuple(element.element_id for element in after.elements)
        or before.coordinates.shape != after.coordinates.shape
        or original.domain_source_id != actual.domain_source_id
        or original.vertex_parameter_ratios != actual.vertex_parameter_ratios
        or any(
            not np.array_equal(first, last)
            for first, last in zip(before.geometry_dofs, after.geometry_dofs, strict=True)
        )
        or any(
            not np.array_equal(first, last)
            for first, last in (
                (original.source_triangles, actual.source_triangles),
                (original.source_segments, actual.source_segments),
                (original.source_triangle_ids, actual.source_triangle_ids),
                (original.source_segment_ids, actual.source_segment_ids),
                (original.vertex_strata, actual.vertex_strata),
                (original.vertex_rows, actual.vertex_rows),
                (original.vertex_parameters, actual.vertex_parameters),
            )
        )
    ):
        return False
    old_points = np.asarray(original.source_points, dtype=np.float64)
    new_points = np.asarray(actual.source_points, dtype=np.float64)
    if old_points.shape != new_points.shape:
        return False
    if translation is None:
        shift = np.zeros(old_points.shape[1], dtype=np.float64)
        for axis in range(old_points.shape[1]):
            lower: Fraction | None = None
            upper: Fraction | None = None
            for old, new in zip(old_points[:, axis], new_points[:, axis], strict=True):
                predecessor = float(np.nextafter(new, -np.inf))
                successor = float(np.nextafter(new, np.inf))
                if not np.isfinite(predecessor) or not np.isfinite(successor):
                    return False
                point, origin = Fraction(float(new)), Fraction(float(old))
                row_lower = (Fraction(predecessor) + point) / 2 - origin
                row_upper = (point + Fraction(successor)) / 2 - origin
                lower = row_lower if lower is None else max(lower, row_lower)
                upper = row_upper if upper is None else min(upper, row_upper)
            if lower is None or upper is None or lower > upper:
                return False
            shift[axis] = float((lower + upper) / 2)
    else:
        shift = translation
    if (
        shift.shape != (old_points.shape[1],)
        or not np.all(np.isfinite(shift))
        or not np.array_equal(new_points, old_points + shift)
        or (
            np.any(shift != 0.0)
            and original.domain_source_revision == actual.domain_source_revision
        )
    ):
        return False
    actual.prepare(after.coordinates, budget=budget)
    return True


@final
class _ConvexParentBank(StrictModule):
    """Authenticated parent corner banks and their active arithmetic ledger."""

    vertices: dict[int, ExactPoint] = eqx.field(static=True)
    cells: dict[int, tuple[ExactPoint, ...]] = eqx.field(static=True)
    cell_vertices: dict[int, tuple[int, ...]] = eqx.field(static=True)
    exact_work: ExactPlcBudget = eqx.field(static=True)


@final
class _ConvexParentFrame(StrictModule):
    """One parent frame with an optional admitted rational inverse action."""

    matrix: ExactMatrix = eqx.field(static=True)
    vertex_rows: dict[int, int] = eqx.field(static=True)
    inverse: ExactMatrix | None = eqx.field(static=True)


def _prepare_convex_parent_bank(
    parent_mesh: cell_mesh_owner.CellMesh,
    parent_geometry: cell_geometry_owner.CellGeometrySpec,
    maximum_work: int,
    maximum_bits: int,
    *,
    root: ExactPlcCellGeometrySource,
    budget: CoordinateEnclosureBudget,
) -> _ConvexParentBank:
    """Authenticate parent corner laws and adopt their proof consumption."""
    mapping = dict(
        zip(parent_geometry.block_names, parent_geometry.elements, strict=True)
    )
    route_mapping = dict(
        zip(parent_geometry.block_names, parent_geometry.geometry_dofs, strict=True)
    )
    elements = tuple(mapping[block.name] for block in parent_mesh.blocks)
    routes = tuple(route_mapping[block.name] for block in parent_mesh.blocks)
    parent_proof = root.prepare(parent_geometry.coordinates, budget=budget)
    bank = parent_proof.vertices
    exact_work = ExactPlcBudget(maximum_work, maximum_bits, budget=budget)
    # Adopt measured root preparation work without estimating from strata
    # or charging it again; this invocation evaluates the root coordinates.
    exact_work.work = parent_proof.operation_count
    exact_work.bits = max(
        (
            max(value.numerator.bit_length(), value.denominator.bit_length())
            for point in bank
            for value in point
        ),
        default=0,
    )
    original = {}
    cells: dict[int, tuple[ExactPoint, ...]] = {}
    cell_vertices = {}
    carrier = np.asarray(jax.device_get(parent_mesh.coordinates), dtype=np.float64)
    for block, element, route in zip(parent_mesh.blocks, elements, routes, strict=True):
        for sci, row, coefficients in zip(
            np.asarray(block.global_ids),
            np.asarray(block.vertices),
            np.asarray(route),
            strict=True,
        ):
            images = coordinate_enclosure_owner.coordinate_corner_images(
                element,
                tuple(bank[int(index)] for index in coefficients),
                budget=budget,
            )
            if images is None or len(images) != 4:
                raise ValueError(
                    "ExactPlcCellGeometryConvexSource.parent_geometry must provide four P1 corner images per parent cell; got an unsupported corner law"
                )
            images = tuple(
                tuple(
                    rational_coefficient(
                        value,
                        role="ExactPlcCellGeometryConvexSource.parent_geometry.corner",
                    )
                    for value in image
                )
                for image in images
            )
            for vertex, image in zip(row, images, strict=True):
                if not np.array_equal(
                    carrier[int(vertex)].view(np.uint64),
                    np.asarray([float(value) for value in image], dtype=np.float64).view(
                        np.uint64
                    ),
                ):
                    raise ValueError(
                        "ExactPlcCellGeometryConvexSource.parent_mesh must contain the bitwise rounded parent corner images; got a mismatched carrier"
                    )
            cells[int(sci)] = images
            cell_vertices[int(sci)] = tuple(int(vertex) for vertex in row)
            for vertex, image in zip(row, images, strict=True):
                point = tuple(image)
                if int(vertex) in original and original[int(vertex)] != point:
                    raise ValueError(
                        "ExactPlcCellGeometryConvexSource.parent_geometry must agree at shared vertex rows; got conflicting corner images"
                    )
                original[int(vertex)] = point
    return _ConvexParentBank(original, cells, cell_vertices, exact_work)


def _convex_target_points(
    original: dict[int, ExactPoint],
    target_mesh: cell_mesh_owner.CellMesh,
    offsets: np.ndarray,
    vertices: np.ndarray,
    weights: tuple[Fraction, ...],
    exact_work: ExactPlcBudget,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[list[ExactPoint], np.ndarray]:
    """Build convex points and verify the bitwise rounded target carrier."""
    points: list[ExactPoint] = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        budget.reserve_geometry_queries(1)
        indices = vertices[int(start) : int(stop)]
        coefficients = weights[int(start) : int(stop)]
        if any(int(index) not in original for index in indices):
            raise ValueError(
                "ExactPlcCellGeometryConvexSource.support_vertices must name used parent vertex rows; got an unused parent vertex"
            )
        operands = coefficients + tuple(
            value for index in indices for value in original[int(index)]
        )
        exact_work.charge(12 * len(indices), operands)
        point: ExactPoint = (
            sum(
                (
                    coefficient * original[int(index)][0]
                    for index, coefficient in zip(indices, coefficients, strict=True)
                ),
                Fraction(),
            ),
            sum(
                (
                    coefficient * original[int(index)][1]
                    for index, coefficient in zip(indices, coefficients, strict=True)
                ),
                Fraction(),
            ),
            sum(
                (
                    coefficient * original[int(index)][2]
                    for index, coefficient in zip(indices, coefficients, strict=True)
                ),
                Fraction(),
            ),
        )
        exact_work.charge(3, point)
        points.append(point)
    numeric = np.asarray(target_mesh.coordinates, dtype=np.float64)
    rne = np.asarray(
        [[float(value) for value in point] for point in points], dtype=np.float64
    )
    if not np.array_equal(numeric.view(np.uint64), rne.view(np.uint64)):
        raise ValueError(
            "ExactPlcCellGeometryConvexSource.target_mesh must contain bitwise rounded convex images; got a mismatched target carrier"
        )
    return points, numeric


def _prepare_convex_parent_frame(
    corners: tuple[ExactPoint, ...],
    cell_vertex_rows: tuple[int, ...],
    exact_work: ExactPlcBudget,
) -> _ConvexParentFrame:
    """Admit parent frame rank before preparing rational reference actions."""
    exact_work.charge(9, tuple(value for corner in corners for value in corner))
    matrix = tuple(
        tuple(corners[column + 1][axis] - corners[0][axis] for column in range(3))
        for axis in range(3)
    )
    a, b, c = tuple(
        (matrix[0][column], matrix[1][column], matrix[2][column]) for column in range(3)
    )
    exact_work.charge(14, a + b + c)
    determinant = exact_cofactor_determinant_3(a, b, c)
    exact_work.charge(0, (determinant,))
    if not determinant:
        raise ValueError(
            "ExactPlcCellGeometryConvexSource.parent_mesh must have rank-three parent corner frames; got a deficient rational frame"
        )
    return _ConvexParentFrame(
        matrix,
        {vertex: index for index, vertex in enumerate(cell_vertex_rows)},
        None,
    )


def _convex_parent_reference(
    vertex: int,
    sci: int,
    corners: tuple[ExactPoint, ...],
    points: list[ExactPoint],
    offsets: np.ndarray,
    vertices: np.ndarray,
    weights: tuple[Fraction, ...],
    parent_frames: dict[int, _ConvexParentFrame],
    exact_work: ExactPlcBudget,
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactPoint:
    """Prove one target point lies in its parent simplex, reusing frame actions."""
    frame = parent_frames[sci]
    matrix, inverse = frame.matrix, frame.inverse
    budget.reserve_geometry_queries(1)
    start, stop = int(offsets[vertex]), int(offsets[vertex + 1])
    supports, coefficients = (
        vertices[start:stop],
        weights[start:stop],
    )
    exact_work.charge(len(supports))
    ordinals = tuple(frame.vertex_rows.get(int(index)) for index in supports)
    if all(index is not None for index in ordinals):
        values = [Fraction(0)] * 3
        for ordinal, coefficient in zip(ordinals, coefficients, strict=True):
            if ordinal:
                exact_work.charge(1)
                values[ordinal - 1] += coefficient
        reference = (values[0], values[1], values[2])
    else:
        if inverse is None:
            inverse = convex_parent_inverse(matrix, exact_work, budget=budget)
            parent_frames[sci] = _ConvexParentFrame(matrix, frame.vertex_rows, inverse)
        point = points[vertex]
        exact_work.charge(3, point)
        delta = tuple(point[axis] - corners[0][axis] for axis in range(3))
        exact_work.charge(15, delta)
        reference = (
            sum(
                (entry * value for entry, value in zip(inverse[0], delta, strict=True)),
                Fraction(),
            ),
            sum(
                (entry * value for entry, value in zip(inverse[1], delta, strict=True)),
                Fraction(),
            ),
            sum(
                (entry * value for entry, value in zip(inverse[2], delta, strict=True)),
                Fraction(),
            ),
        )
    exact_work.charge(3, reference)
    if min(reference) < 0 or sum(reference, Fraction(0)) > 1:
        raise ValueError(
            "ExactPlcCellGeometryConvexSource.target_mesh must stay within its declared parent simplex; got a reference point outside that simplex"
        )
    return reference


def _require_convex_corner_rank(
    corners: tuple[ExactPoint, ...],
    exact_work: ExactPlcBudget,
    message: str,
) -> None:
    """Refuse corner banks whose rational affine span has deficient rank."""
    rank = False
    for indices in itertools_owner.combinations(range(1, len(corners)), 3):
        a, b, c = (
            (
                corners[index][0] - corners[0][0],
                corners[index][1] - corners[0][1],
                corners[index][2] - corners[0][2],
            )
            for index in indices
        )
        exact_work.charge(32, a + b + c)
        determinant = exact_cofactor_determinant_3(a, b, c)
        if determinant:
            rank = True
            break
    if not rank:
        raise ValueError(message)


def _convex_target_references(
    bank: _ConvexParentBank,
    target_mesh: cell_mesh_owner.CellMesh,
    points: list[ExactPoint],
    numeric: np.ndarray,
    offsets: np.ndarray,
    vertices: np.ndarray,
    weights: tuple[Fraction, ...],
    parents: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[ExactPoint, ...], ...]:
    """Authenticate target cells against their parent simplex and rounded rank."""
    parent_frames: dict[int, _ConvexParentFrame] = {}
    reference_cache: dict[tuple[int, int], ExactPoint] = {}
    references: list[tuple[ExactPoint, ...]] = []
    cursor = 0
    for block in target_mesh.blocks:
        for row, active in zip(
            np.asarray(block.vertices), np.asarray(block.vertex_valid), strict=True
        ):
            row = row[active]
            sci = int(parents[cursor])
            cursor += 1
            if sci not in bank.cells:
                raise ValueError(
                    "ExactPlcCellGeometryConvexSource.cell_parent_ids must name parent global cell IDs; got an absent parent cell ID"
                )
            corners = bank.cells[sci]
            if sci not in parent_frames:
                parent_frames[sci] = _prepare_convex_parent_frame(
                    corners, bank.cell_vertices[sci], bank.exact_work
                )
            cell_refs: list[ExactPoint] = []
            for vertex in row:
                vertex = int(vertex)
                key = (vertex, sci)
                bank.exact_work.charge(1)
                reference = reference_cache.get(key)
                if reference is None:
                    reference = _convex_parent_reference(
                        vertex,
                        sci,
                        corners,
                        points,
                        offsets,
                        vertices,
                        weights,
                        parent_frames,
                        bank.exact_work,
                        budget=budget,
                    )
                    reference_cache[key] = reference
                cell_refs.append(reference)
            # Nonnegative unit vertex weights authenticate the whole source range.
            # Polyhedron topology and convexity are certified separately.
            _require_convex_corner_rank(
                tuple(cell_refs),
                bank.exact_work,
                "ExactPlcCellGeometryConvexSource.target_mesh must have rank-three rational corner frames; got a deficient frame",
            )
            rounded_corners = tuple(
                rational_vector(numeric[int(vertex)]) for vertex in row
            )
            _require_convex_corner_rank(
                rounded_corners,
                bank.exact_work,
                "ExactPlcCellGeometryConvexSource.target_mesh must have rank-three rounded corner frames; got a deficient carrier frame",
            )
            references.append(tuple(cell_refs))
    return tuple(references)


@checked
def require_plc_parent_source(
    geometry: cell_geometry_owner.CellGeometrySpec, /
) -> ExactPlcCellGeometrySource:
    """Return the PLC root admitted by a convex parent geometry binding."""
    match geometry.exact_source:
        case ExactPlcCellGeometrySource() as source:
            return source
        case _:
            raise TypeError(
                "ExactPlcCellGeometryConvexSource.parent_geometry must retain an ExactPlcCellGeometrySource; got another source binding"
            )


class ExactPlcCellGeometryConvexSource(StrictModule, NonTrainableState):
    """Bind convex PLC P1 corner images to parent scientific cell IDs."""

    parent_mesh: cell_mesh_owner.CellMesh
    parent_geometry: cell_geometry_owner.CellGeometrySpec
    target_mesh: cell_mesh_owner.CellMesh
    support_offsets: Array
    support_vertices: Array
    cell_parent_ids: Array
    support_coefficients: tuple[Fraction, ...] = eqx.field(static=True)
    original_binding_id: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        parent_mesh: cell_mesh_owner.CellMesh,
        parent_geometry: cell_geometry_owner.CellGeometrySpec,
        support_offsets: ArrayLike,
        support_vertices: ArrayLike,
        support_coefficients: tuple[Fraction, ...],
        target_mesh: cell_mesh_owner.CellMesh,
        cell_parent_ids: ArrayLike,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:

        root = require_plc_parent_source(parent_geometry)
        if (
            parent_geometry.periodic_source is not None
            or parent_mesh.periodic_topology is not None
        ):
            raise ValueError(
                "ExactPlcCellGeometryConvexSource.parent_geometry must bind a nonperiodic P1 source; got a periodic binding"
            )
        mapping = dict(
            zip(parent_geometry.block_names, parent_geometry.elements, strict=True)
        )
        if set(mapping) != {block.name for block in parent_mesh.blocks}:
            raise ValueError(
                "ExactPlcCellGeometryConvexSource.parent_geometry must bind every parent mesh block; got mismatched block names"
            )
        elements = tuple(mapping[block.name] for block in parent_mesh.blocks)
        for block, element in zip(parent_mesh.blocks, elements, strict=True):
            if (
                block.cell_kind != "tetrahedron"
                or cell_geometry_owner.require_full_p1_source(element).cell_kind
                != "tetrahedron"
            ):
                raise ValueError(
                    "ExactPlcCellGeometryConvexSource.parent_mesh must use full P1 tetrahedron maps; got another parent cell map"
                )
        raw = tuple(
            np.asarray(value)
            for value in (support_offsets, support_vertices, cell_parent_ids)
        )
        if any(value.ndim != 1 or value.dtype.kind not in "iu" for value in raw):
            raise TypeError(
                "ExactPlcCellGeometryConvexSource.support_vertices must use integer vectors with integer offsets and parent IDs; got another dtype or rank"
            )
        if any(np.any(value > np.iinfo(np.int64).max) for value in raw):
            raise ValueError(
                "ExactPlcCellGeometryConvexSource.support_vertices must fit signed int64 with offsets and parent IDs; got an out-of-range value"
            )
        offsets, vertices, parents = (np.asarray(value, dtype=np.int64) for value in raw)
        count = target_mesh.coordinates.shape[0]
        if (
            offsets.shape != (count + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(vertices)
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError(
                "ExactPlcCellGeometryConvexSource.support_offsets must define one nonempty support per target vertex; got invalid CSR offsets"
            )
        if np.any(vertices < 0) or np.any(vertices >= parent_mesh.coordinates.shape[0]):
            raise ValueError(
                "ExactPlcCellGeometryConvexSource.support_vertices must index parent mesh vertex rows; got an undeclared vertex row"
            )
        if (
            not isinstance(support_coefficients, tuple)
            or len(support_coefficients) != len(vertices)
            or any(type(value) is not Fraction for value in support_coefficients)
        ):
            raise TypeError(
                "ExactPlcCellGeometryConvexSource.support_coefficients must be an aligned Fraction tuple; got another coefficient representation"
            )
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            weights = support_coefficients[int(start) : int(stop)]
            if (
                min(weights) <= 0
                or sum(weights, Fraction(0)) != 1
                or len(set(vertices[start:stop].tolist())) != stop - start
            ):
                raise ValueError(
                    "ExactPlcCellGeometryConvexSource.support_coefficients must be positive and sum to one on unique support rows; got a repeated row or unnormalized weights"
                )
        if parents.shape != (sum(block.cell_count for block in target_mesh.blocks),):
            raise ValueError(
                "ExactPlcCellGeometryConvexSource.cell_parent_ids must align with every target cell; got another bank shape"
            )
        if any(
            block.cell_kind not in ("tetrahedron", "hexahedron", "pyramid", "polyhedron")
            for block in target_mesh.blocks
        ):
            raise ValueError(
                "ExactPlcCellGeometryConvexSource.target_mesh must use degree-one tetrahedron, hexahedron, pyramid, or cut-polyhedron maps; got another cell kind"
            )
        binding = canonical_fingerprint(
            (
                self._binding(parent_mesh, parent_geometry, target_mesh, root),
                array_tree_fingerprint(
                    (jnp.asarray(offsets), jnp.asarray(vertices), jnp.asarray(parents))
                ),
                tuple(
                    (value.numerator, value.denominator) for value in support_coefficients
                ),
            )
        )
        # Validate scientific bindings and rational-coordinate invariants before assignment.
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
            root=root,
            budget=budget,
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ExactPlcCellGeometryConvexSource],
        value: ExactPlcCellGeometryConvexSource,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> ExactPlcCellGeometryConvexSource:
        """Rebuild convex support fields under their parent proof admission."""
        if budget is None:
            work_upper, bytes_upper = cell_geometry_owner.exact_source_preparation_bounds(
                value,
                value.support_vertices.shape[0],
                value.parent_geometry.coordinates.shape[1],
            )
            preparation = cell_geometry_owner.owned_coordinate_preparation(
                work_upper,
                bytes_upper,
                maximum_geometry_queries=value.preparation_query_upper(),
            )
        else:
            preparation = nullcontext(budget)
        with preparation as admission:
            return cls(
                value.parent_mesh,
                value.parent_geometry,
                value.support_offsets,
                value.support_vertices,
                value.support_coefficients,
                value.target_mesh,
                value.cell_parent_ids,
                budget=admission,
            )

    @staticmethod
    def _binding(
        parent_mesh: CellMesh,
        parent_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        source: ExactPlcCellGeometrySource,
    ) -> str:
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

    def owners(
        self,
    ) -> tuple[
        cell_mesh_owner.CellMesh,
        cell_geometry_owner.CellGeometrySpec,
        cell_mesh_owner.CellMesh,
    ]:
        """Return the parent mesh, parent geometry, and target mesh bindings."""
        return self.parent_mesh, self.parent_geometry, self.target_mesh

    @property
    def source_id(self) -> str:
        parent_mesh, parent_geometry, target_mesh = self.owners()
        root = require_plc_parent_source(parent_geometry)
        return canonical_fingerprint(
            {
                "kind": "exact-plc-convex-source",
                "graph": self._binding(parent_mesh, parent_geometry, target_mesh, root),
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
        parent_mesh: cell_mesh_owner.CellMesh,
        parent_geometry: cell_geometry_owner.CellGeometrySpec,
        target_mesh: cell_mesh_owner.CellMesh,
        offsets: np.ndarray,
        vertices: np.ndarray,
        weights: tuple[Fraction, ...],
        parents: np.ndarray,
        maximum_work: int,
        maximum_bits: int,
        *,
        root: ExactPlcCellGeometrySource,
        budget: CoordinateEnclosureBudget,
    ) -> ConvexPlcProof:
        bank = _prepare_convex_parent_bank(
            parent_mesh,
            parent_geometry,
            maximum_work,
            maximum_bits,
            root=root,
            budget=budget,
        )
        with budget.temporary_scope():
            bank_upper, scratch = plc_storage_bounds(
                len(vertices) + len(parents) * 8, maximum_bits
            )
            budget.reserve(0, bank_upper + scratch)
            points, numeric = _convex_target_points(
                bank.vertices,
                target_mesh,
                offsets,
                vertices,
                weights,
                bank.exact_work,
                budget=budget,
            )
            references = _convex_target_references(
                bank,
                target_mesh,
                points,
                numeric,
                offsets,
                vertices,
                weights,
                parents,
                budget=budget,
            )
            error = max(
                (
                    abs(value - Fraction(float(value)))
                    for point in points
                    for value in point
                ),
                default=Fraction(0),
            )
            result = ConvexPlcProof(
                vertices=tuple(points),
                parent_reference_corners=references,
                operation_count=bank.exact_work.work,
                maximum_integer_bits=bank.exact_work.bits,
                maximum_rounding_error=error,
            )
            budget.retain_basis((result.vertices, result.parent_reference_corners))
            return result

    def _proof(
        self,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> ConvexPlcProof:

        parent_mesh = self.parent_mesh
        parent_geometry = self.parent_geometry
        target_mesh = self.target_mesh
        root = require_plc_parent_source(parent_geometry)
        if (self.maximum_work, self.maximum_bits) != (
            root.maximum_work,
            root.maximum_bits,
        ):
            raise ValueError(
                "ExactPlcCellGeometryConvexSource.maximum_work must retain the parent source allowance with maximum_bits; got changed controls"
            )
        current = canonical_fingerprint(
            (
                self._binding(parent_mesh, parent_geometry, target_mesh, root),
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
                "ExactPlcCellGeometryConvexSource.original_binding_id must bind current support, geometry, and topology; got a changed binding"
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
            root=root,
            budget=budget,
        )

    @checked
    def cell_parent_reference_corners(
        self,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[tuple[ExactPoint, ...], ...]:
        return self._proof(budget=budget).parent_reference_corners

    def preparation_query_upper(self) -> int:
        """Bound support-point and distinct target-corner reference queries."""
        _, parent_geometry, target = self.owners()
        source = require_plc_parent_source(parent_geometry)
        return (
            source.preparation_query_upper()
            + self.support_offsets.shape[0]
            - 1
            + sum(block.vertices.size for block in target.blocks)
        )

    @checked
    def prepare(
        self, rounded_vertices: ArrayLike, /, *, budget: CoordinateEnclosureBudget
    ) -> PreparedExactPlcGeometry:

        rounded = np.asarray(rounded_vertices, dtype=np.float64)
        target = np.asarray(self.target_mesh.coordinates, dtype=np.float64)
        if rounded.shape != target.shape or not np.array_equal(
            rounded.view(np.uint64), target.view(np.uint64)
        ):
            raise ValueError(
                "ExactPlcCellGeometryConvexSource.rounded_vertices must equal the bound target carrier bitwise; got another carrier"
            )
        proof = self._proof(budget=budget)
        return PreparedExactPlcGeometry(
            proof.vertices,
            tuple(False for _ in proof.vertices),
            self.source_id,
            proof.operation_count,
            proof.maximum_integer_bits,
            proof.maximum_rounding_error,
        )


def plc_storage_bounds(point_count: int, maximum_bits: int, /) -> tuple[int, int]:
    """Live 3-D rational-bank and arithmetic upper bounds before construction."""
    # Binary64 rationals can require 1075 bits before the source bit limit is
    # checked; determinants and squared errors can grow to eight times that.
    allocation_bits = max(1075, maximum_bits)
    integer_upper = 64 + 4 * ((allocation_bits + 29) // 30)
    fraction_upper = 128 + 2 * integer_upper
    bank_upper = 512 + point_count * (192 + 3 * fraction_upper)
    scratch_integer_upper = 64 + 4 * ((8 * allocation_bits + 29) // 30)
    return bank_upper, 64 * (128 + 2 * scratch_integer_upper)


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

    return tuple(coordinate(axis) for axis in range(len(corners[0])))


def _row_witness(
    carrier: ExactPoint,
    corners: tuple[ExactPoint, ...],
    weights: tuple[Fraction, ...],
    declared_bound: float,
    exact_work: ExactPlcBudget,
    /,
    *,
    rational_weights: bool,
) -> tuple[ExactPoint, bool] | str:
    """Return a row carrier's source vertex and on-entity flag, or its refusal.

    Binary64 weights let an on-entity carrier stand for itself; otherwise the
    carrier must be the rounded construction. Any off-entity carrier must lie
    within the row's declared bound of the exact row point.
    """
    if not rational_weights and in_entity(carrier, corners):
        return carrier, True
    exact = _witness_point(corners, weights)
    exact_work.charge(16, exact)
    if not rational_weights and any(
        float(value) != float(coordinate)
        for value, coordinate in zip(exact, carrier, strict=True)
    ):
        return "ExactPlcCellGeometrySource.rounded_vertices must lie on their source entity or equal rounded constructed coordinates; got another carrier point"
    error_squared = sum(
        ((value - rne) ** 2 for value, rne in zip(exact, carrier, strict=True)),
        Fraction(0),
    )
    if error_squared > Fraction(declared_bound) ** 2:
        return "ExactPlcCellGeometrySource.source_row_bounds must cover the carrier displacement; got a carrier outside its declared row bound"
    return exact, False


def _row_parameters(
    point: ExactPoint, corners: tuple[ExactPoint, ...], /
) -> tuple[Fraction, ...]:

    edges = tuple(
        tuple(corner[axis] - corners[0][axis] for axis in range(len(point)))
        for corner in corners[1:]
    )
    delta = tuple(point[axis] - corners[0][axis] for axis in range(len(point)))
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
    return tuple(row[0] for row in solve_exact(matrix, right))


def in_entity(point: ExactPoint, corners: tuple[ExactPoint, ...], /) -> bool:
    """Return closed segment or triangle membership using rational predicates."""
    return _in_entity_frame(point, _plc_entity_numeric_frame(corners, len(point)))


def _in_entity_frame(point: ExactPoint, frame: _PlcEntityFrame, /) -> bool:
    corners, edges = frame.corners, frame.edges
    origin = corners[0]
    offset = tuple(point[axis] - origin[axis] for axis in range(len(point)))
    if len(edges) == 1:
        (direction,) = edges
        if not any(direction):
            raise ValueError(
                "in_entity.corners must span a segment; got coincident corners"
            )
        cross = tuple(
            offset[a] * direction[b] - offset[b] * direction[a]
            for a in range(len(point))
            for b in range(a + 1, len(point))
        )
        if any(cross):
            return False
        along = sum(a * b for a, b in zip(offset, direction, strict=True))
        return 0 <= along <= sum(value * value for value in direction)
    first, second = edges
    if len(point) == 2:
        (determinant,) = frame.normal
        if determinant == 0:
            raise ValueError(
                "in_entity.corners must span a triangle; got collinear corners"
            )
        weights = _row_parameters(point, corners)
        return min(weights) >= 0 and sum(weights, Fraction(0)) <= 1
    normal = frame.normal
    if not any(normal):
        raise ValueError("in_entity.corners must span a triangle; got collinear corners")
    if sum(a * b for a, b in zip(offset, normal, strict=True)) != 0:
        return False
    # Coplanar: rational 2-D orientations in the projection dropping the dominant
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


class ExactPlcBudget:
    @checked
    def __init__(
        self,
        maximum_work: int,
        maximum_bits: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        self.maximum_work, self.maximum_bits = maximum_work, maximum_bits
        self.work, self.bits = 0, 0
        self.enclosure = budget

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

        if self.work > self.maximum_work:
            raise coordinate_budget_owner.CoordinateEnclosureResourceError(
                "coefficient_work",
                self.maximum_work,
                self.work,
                self.work - work,
            )
        if self.bits > self.maximum_bits:
            raise ValueError(
                "ExactPlcBudget.maximum_bits must cover every numerator and denominator; got a larger rational operand"
            )
        self.enclosure.reserve(work)


__all__ = [
    "ExactPlcCellGeometrySource",
    "ExactPlcCellGeometryConvexSource",
    "PreparedExactPlcGeometry",
]


@checked
def convex_parent_inverse(
    matrix: tuple[tuple[Fraction, ...], ...],
    exact_work: ExactPlcBudget,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, ...], ...]:
    """Prepare a parent reference action and settle its measured work."""

    ledger = budget
    start = ledger.work_units
    try:
        with (
            ledger.activate(),
            ledger.bound_stage(
                max(0, exact_work.maximum_work - exact_work.work),
                ledger.maximum_memory_bytes,
                starting_work_units=start,
            ),
            ledger.temporary_scope(),
        ):
            prepared = small_batched_owner.prepare_exact_small_linear_actions(
                matrix,
                tuple(
                    tuple(Fraction(int(row == column)) for column in range(3))
                    for row in range(3)
                ),
                budget=budget,
            )
    finally:
        work = ledger.work_units - start
        exact_work.work += work

    if prepared.actions is None:
        raise ValueError(
            "convex_parent_inverse.matrix must have rank three; got a deficient reference action"
        )
    exact_work.charge(0, tuple(value for row in prepared.actions for value in row))
    return prepared.actions
