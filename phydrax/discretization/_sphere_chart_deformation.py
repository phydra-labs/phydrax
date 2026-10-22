#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepare exact projective correspondence between sphere material cells.

The admitted BVH/cone workset bounds a shared exact great-circle edge overlay.
Actual edge fragments reconstruct occurrence-labelled halfedge cycles. Their
polygons equal independent native intersections up to cyclic canonicalization;
exact boundary, feasibility and BOTH whole-reference partitions remain binding.
Projective images factor through actual source/target cone dual actions, and
published maps/points are interned once in the domain-owned sphere piece bank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

import phydrax.discretization._cell_geometry_transition_types as transition_types_owner
import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._rational import fraction_endpoint

from .. import _geometry_predicates as imported_geometry_predicates, _meshcore
from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import NativeBatchTicket, NativeExecutionScope
from .._strict import StrictModule
from .._validation import positive_integer
from ..geometry import (
    _domain_numerics as domain_numerics,
    _meshing_domain as imported_geometry__meshing_domain,
    _sphere_material_atlas as imported_geometry__sphere_material_atlas,
)
from ..geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from ..geometry._interval_enclosure import interval_divide
from ..geometry._sphere_material_atlas import (
    SphereMaterialCellAtlas,
    SphereProjectiveActionBank,
    SphereProjectiveReferenceMap,
    SphereProjectiveTriangleBounds,
)
from ..linalg import _hermitian_spectral as imported_linalg__hermitian_spectral
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ..typing import checked, ConvertibleToArray
from . import (
    _cell_geometry_transfer as imported_cell_geometry_transfer,
    _chart_execution as chart_execution,
    _chart_geometry as chart_geometry,
    _coordinate_enclosure as imported_coordinate_enclosure,
    _sphere_chart_records as sphere_chart_records,
)
from ._cell_geometry import CellGeometrySpec
from ._cell_geometry_transition_types import CellGeometryTransitionPolicy
from ._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ._cell_mesh import CellMesh
from ._chart_certification import certify_chart_endpoint
from ._coordinate_budget import CoordinateEnclosureBudget, CoordinateLiveStorage
from ._coordinate_enclosure import Expression, ExpressionComposition
from ._exact_algebraic_geometry import rational_coefficient
from ._sphere_chart_piece_bank import _SphereKernelMap, _SphereKernelPoint
from ._sphere_edge_overlay import _SphereCandidateOrientations, _SphereEdgeOverlay


type _Point = tuple[Fraction, Fraction]
type _Triangle = tuple[_Point, _Point, _Point]
type _Plane = tuple[Fraction, Fraction, Fraction]
type _FloatArray = NDArray[np.float64]
type _IntArray = NDArray[np.int64]


def _tokens(vertices: tuple[_Point, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(
        tuple((value.numerator, value.denominator) for value in point)
        for point in vertices
    )


def _direction_boxes(atlas: SphereMaterialCellAtlas) -> tuple[_FloatArray, _FloatArray]:
    directions = np.asarray(atlas.directions, dtype=np.float64)
    low, high = interval_divide(
        (np.min(directions, axis=1), np.max(directions, axis=1)),
        (
            np.asarray(atlas.hemisphere_lower)[:, None],
            np.asarray(atlas.direction_norm_upper)[:, None],
        ),
    )
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


@checked
def cone_candidates(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    source_rows: _IntArray,
    target_rows: _IntArray,
    memory: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_IntArray, _IntArray, _SphereCandidateOrientations | None]:
    """Discard only certified halfspace-separated or zero-area cone pairs."""

    budget.reserve(0, source_rows.size)
    keep = np.ones(source_rows.shape, dtype=np.bool_)
    banks = (np.asarray(source.directions), np.asarray(target.directions))
    intermediates = None
    if all(bank.dtype == np.dtype(np.float64) for bank in banks):
        intermediates = _SphereCandidateOrientations(
            source, target, source.atlas_id, target.atlas_id,
            banks, source_rows.size, budget=budget,
        )
        banks = intermediates.rays
    orientations = tuple(
        np.asarray(
            [1 if value > 0 else -1 for value in atlas.exact_determinants],
            dtype=np.int8,
        )
        for atlas in (source, target)
    )
    chunk = min(4096, memory // 2048)
    if chunk < 1:
        raise ValueError(
            "sphere_chart.cone_candidates.budget must satisfy the admitted resource limits; got Sphere cone candidate filtering exceeds its source resident memory limit."
        )
    for start in range(0, source_rows.size, chunk):
        stop = min(start + chunk, source_rows.size)
        with budget.temporary_scope():
            budget.reserve(0, (stop - start) * 2048)
            rows = (source_rows[start:stop], target_rows[start:stop])
            rays = (banks[0][rows[0]], banks[1][rows[1]])
            for owner, other in ((0, 1), (1, 0)):
                predicate = imported_geometry_predicates.orient3d(
                    np.zeros((3,), dtype=np.float64),
                    rays[owner][:, (0, 1, 2), None, :],
                    rays[owner][:, (1, 2, 0), None, :],
                    rays[other][:, None, :, :],
                    mode=imported_geometry_predicates.PredicateMode.EXACT,
                    budget=budget,
                )
                if intermediates is not None:
                    intermediates.record(
                        owner, start, stop,
                        np.asarray(predicate.signs), np.asarray(predicate.certain),
                    )
                signs = (
                    np.asarray(predicate.signs)
                    * orientations[owner][rows[owner], None, None]
                )
                # Resolve uncertain filtered signs before forming projective
                # coefficient maps. A closed exterior halfspace admits only
                # zero-area contact, which cannot contribute reference coverage.
                separated = np.any(
                    np.all(np.asarray(predicate.certain) & (signs <= 0), axis=-1), axis=-1
                )
                keep[start:stop] &= ~separated
    budget.reserve(0, source_rows.size * np.dtype(np.int64).itemsize)
    kept = np.flatnonzero(keep)
    budget.reserve(0, 2 * kept.size * np.dtype(np.int64).itemsize)
    sources, targets = source_rows[kept], target_rows[kept]
    if intermediates is not None:
        intermediates.finish(sources, targets, kept)
    return sources, targets, intermediates


def _candidates(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    limit: int,
    memory: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_IntArray, _IntArray, _SphereCandidateOrientations | None]:
    source_low, source_high = _direction_boxes(source)
    target_low, target_high = _direction_boxes(target)
    source_bvh = prepare_bvh(source_low, source_high, dtype=jnp.float64)
    target_bvh = prepare_bvh(target_low, target_high, dtype=jnp.float64)
    targets: list[_IntArray] = []
    sources: list[_IntArray] = []
    count = 0
    for target_rows, source_rows in bvh_overlap_pair_blocks(target_bvh, source_bvh):
        count += target_rows.size
        if count > limit or count * 32 > memory:
            raise ValueError(
                "sphere_chart.candidates.budget must satisfy the admitted resource limits; got Sphere native candidate pairs exceed their work or resident memory limit."
            )
        targets.append(np.asarray(target_rows, dtype=np.int64))
        sources.append(np.asarray(source_rows, dtype=np.int64))
    if not sources:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), None
    source_rows, target_rows = np.concatenate(sources), np.concatenate(targets)
    order = np.lexsort((source_rows, target_rows))
    return cone_candidates(
        source, target, source_rows[order], target_rows[order], memory, budget=budget
    )


def _planes(mapping: SphereProjectiveReferenceMap) -> tuple[_Plane, ...]:
    return (
        (Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(-1), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(1)),
    ) + tuple((-row[1], -row[2], row[0]) for row in mapping.exact_coefficients)


def _support(label: int, planes: tuple[_Plane, ...]) -> _Plane:
    if label < 0 or label >= len(planes):
        raise ValueError(
            "sphere_chart.support.pieces must satisfy the material reference contract; got Native polygon names an unknown source rational halfplane."
        )
    return planes[label]


type _SupportKey = frozenset[frozenset[bytes]]
type _Integer3 = tuple[int, int, int]
type _VertexKey = _SupportKey | _Integer3


@dataclass(frozen=True, slots=True)
class _SphereArrangementCell:
    rays: tuple[_Integer3, ...]
    scales: tuple[Fraction, ...]
    normals: tuple[_Integer3, ...]
    orientations: tuple[Fraction, ...]


class _SphereArrangementOperands:
    """Private exact operands for the candidate-bound shared edge overlay.

    Primitive integer triples name unoriented rays and great circles; actual
    chart dual scales distinguish the physical antipodes. Only immutable
    numerical atoms are retained. The overlay reconstructs occurrence-bound
    edge fragments and halfedge cycles, with no clipping or verdict cache here.
    """

    def __init__(
        self,
        actions: SphereProjectiveActionBank,
        storage: CoordinateLiveStorage,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        self.actions = actions
        self.storage = storage
        self.budget = budget
        storage.set_bound(storage.bound + 2048)
        self.cells: dict[tuple[bool, int], _SphereArrangementCell] = {}
        self.rays: dict[tuple[Fraction, ...], tuple[_Integer3, Fraction]] = {}
        self.meets: dict[frozenset[_Integer3], _Integer3] = {}
        self.cross_zeros: set[frozenset[_Integer3]] = set()
        self.products: dict[frozenset[_Integer3], int] = {}
        self._product_storage = 0
        self.coordinates: dict[tuple[bool, int, _Integer3], tuple[_Point, Fraction]] = {}
        self._kernel_builder: sphere_chart_records.SphereChartPieceBankBuilder | None = None
        self._kernel_points: dict[tuple[bool, int, _Integer3], _SphereKernelPoint] = {}
        self._work_start = 0
        self._work_limit = 0
        self._memory_limit = 0

    def _bind_operation_rays(
        self, source: object, target: object, source_id: str, target_id: str,
        source_rays: np.ndarray, target_rays: np.ndarray,
    ) -> None:
        """Bind native intermediates to owned current rays before any event."""
        if (
            source is not self.actions.source or target is not self.actions.target
            or source_id != self.actions.source.atlas_id
            or target_id != self.actions.target.atlas_id
            or self.cells or self.actions.corners or self.actions.inverses
            or self.actions.columns
        ):
            raise ValueError("sphere_chart.overlay.rays must bind this fresh actual source/target operation.")
        banks = (source_rays, target_rays)
        size = source_rays.nbytes + target_rays.nbytes
        with self.budget.temporary_scope():
            self.budget.reserve(2 * ((size + 63) // 64) + 4, size + 1024)
            for atlas, snapshot in zip(
                (self.actions.source, self.actions.target), banks, strict=True
            ):
                root: object = snapshot
                while isinstance(root, np.ndarray):
                    root = root.base
                actual = np.asarray(atlas.directions)
                if (
                    type(root) is not bytes or snapshot.dtype != np.dtype(np.float64)
                    or not snapshot.flags.c_contiguous
                    or actual.dtype != snapshot.dtype or actual.shape != snapshot.shape
                    or actual.tobytes(order="C") != root
                ):
                    raise ValueError("sphere_chart.overlay.rays must retain complete current owned binary64 bytes; got stale or mismatched rays.")
        # These exact Fractions are created only from the owned immutable
        # binary64 snapshots, not from caller/decoded Fraction cache entries.
        # The positive scale law follows the same primitive-ray construction.
        for target_bank, values in enumerate(banks):
            for row, corners in enumerate(values):
                self._retain(9, 2 * 1100, 512)
                self.actions.corners[(bool(target_bank), row)] = tuple(
                    tuple(Fraction(float(value)) for value in corner) for corner in corners
                )

    def _check_work(self, work: int) -> None:
        if self.budget.work_units - self._work_start + work > self._work_limit:
            raise ValueError(
                "sphere_chart.arrangement.budget must satisfy the admitted resource limits; got Exact edge arrangement exceeds its declared work budget."
            )

    def _admit(self, work: int, count: int = 0, bits: int = 1) -> None:
        self._check_work(work)
        if count:
            imported_linalg__hermitian_spectral.reserve_fraction_work(
                self.budget, work, count, bits
            )
        else:
            self.budget.reserve(work)

    def _retain(self, count: int, bits: int, metadata: int = 512) -> None:
        size = metadata + imported_linalg__hermitian_spectral.fraction_storage_upper(
            count, bits
        )
        self._retain_metadata(size)

    def _retain_metadata(self, size: int) -> None:
        """Admit actual containers without inventing an empty Fraction bank."""
        if self.storage.bound + size > self._memory_limit:
            raise ValueError(
                "sphere_chart.arrangement.budget must satisfy the admitted resource limits; got Exact edge arrangement exceeds its resident memory limit."
            )
        self.storage.set_bound(self.storage.bound + size)

    def _primitive(self, values: _Integer3) -> _Integer3:
        self._admit(9, 8, max(abs(value).bit_length() for value in values) + 2)
        common = math.gcd(*values)
        if common == 0:
            return (0, 0, 0)
        if next(value for value in values if value) < 0:
            common = -common
        return values[0] // common, values[1] // common, values[2] // common

    def _ray(self, values: tuple[Fraction, ...]) -> tuple[_Integer3, Fraction]:
        self._admit(1)
        known = self.rays.get(values)
        if known is not None:
            return known
        bits = max(
            abs(value.numerator).bit_length() + 3 * value.denominator.bit_length()
            for value in values
        ) + 8
        self._admit(16, 24, bits)
        self._retain(4, bits)
        denominator = math.lcm(*(value.denominator for value in values))
        integers = (
            values[0].numerator * (denominator // values[0].denominator),
            values[1].numerator * (denominator // values[1].denominator),
            values[2].numerator * (denominator // values[2].denominator),
        )
        ray = self._primitive(integers)
        if not any(ray):
            raise ValueError(
                "sphere_chart.arrangement.rays must be nonzero; got a zero authenticated corner ray."
            )
        axis = next(index for index, value in enumerate(ray) if value)
        scale = values[axis] / ray[axis]
        known = ray, scale
        self.rays[values] = known
        return known

    def _meet(self, first: _Integer3, second: _Integer3) -> _Integer3:
        self._admit(1)
        key = frozenset((first, second))
        known = self.meets.get(key)
        if known is not None:
            return known
        bits = 2 * max(abs(value).bit_length() for value in (*first, *second)) + 2
        self._admit(9, 16, bits)
        self._retain(3, bits)
        value = self._primitive(
            (
                first[1] * second[2] - first[2] * second[1],
                first[2] * second[0] - first[0] * second[2],
                first[0] * second[1] - first[1] * second[0],
            )
        )
        self.meets[key] = value
        # The exact constructed cross product is perpendicular to both actual
        # operands, including a zero cross product. Normalization is a scalar
        # division and preserves these two numeric zeros. Keep that algebraic
        # operand identity, not a cone-containment or clipping verdict.
        for operand in (first, second):
            self._admit(1)
            zero_key = frozenset((value, operand))
            if zero_key not in self.cross_zeros:
                self._retain_metadata(512)
                self.cross_zeros.add(zero_key)
        return value

    def _dot(self, first: _Integer3, second: _Integer3) -> int:
        self._admit(1)
        key = frozenset((first, second))
        if key in self.cross_zeros:
            return 0
        known = self.products.get(key)
        if known is not None:
            return known
        bits = 2 * max(abs(value).bit_length() for value in (*first, *second)) + 3
        self._admit(5, 8, bits)
        before = self.storage.bound
        self._retain(1, bits, 256)
        self._product_storage += self.storage.bound - before
        value = first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
        self.products[key] = value
        return value

    def _release_prepare_products(self) -> None:
        """Discard construction-only scalar memo owners before query work."""
        self.products.clear()
        self.storage.set_bound(self.storage.bound - self._product_storage)
        self._product_storage = 0

    def _cell(self, target: bool, row: int) -> _SphereArrangementCell:
        self._admit(1)
        key = target, row
        known = self.cells.get(key)
        if known is not None:
            return known
        self._retain(0, 1, 1024)
        corners = tuple(self._ray(value) for value in self.actions.cell_corners(target, row))
        rays = tuple(value[0] for value in corners)
        scales = tuple(value[1] for value in corners)
        normals = tuple(
            self._meet(rays[(slot + 1) % 3], rays[(slot + 2) % 3])
            for slot in range(3)
        )
        orientations: list[Fraction] = []
        for slot, normal in enumerate(normals):
            value = self._dot(normal, rays[slot])
            bits = abs(value).bit_length() + max(
                abs(scales[slot].numerator).bit_length(),
                scales[slot].denominator.bit_length(),
            ) + 2
            self._admit(2, 4, bits)
            self._retain(1, bits, 128)
            if value == 0:
                raise ValueError(
                    "sphere_chart.arrangement.cells must have rank three; got a degenerate authenticated cone."
                )
            # Retain the actual normal/corner scalar, not its predicate sign.
            orientations.append(value * scales[slot])
        known = _SphereArrangementCell(rays, scales, normals, tuple(orientations))
        self.cells[key] = known
        bits = max(
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for value in scales
        ) + 2
        self._admit(3, 3, bits)
        self._retain(3, bits, 384)
        for slot, ray in enumerate(rays):
            # Authored corner = scale * primitive ray; its homogeneous dual
            # sum is reciprocal scale, in the actual source OR target chart.
            self.coordinates[(target, row, ray)] = (
                _TARGET_CORNERS[slot], Fraction(1) / scales[slot]
            )
        return known

    def _coordinates(
        self, row: int, ray: _Integer3, *, target: bool = False
    ) -> tuple[_Point, Fraction]:
        self._admit(1)
        key = target, row, ray
        known = self.coordinates.get(key)
        if known is not None:
            return known
        cell = self._cell(target, row)
        products = tuple(self._dot(normal, ray) for normal in cell.normals)
        self._admit(6)
        bits = max(abs(value).bit_length() for value in products) + max(
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for value in cell.orientations
        ) + 4
        self._admit(11, 32, 8 * bits + 16)
        self._retain(3, 8 * bits + 16)
        # The i-th great-circle normal vanishes on the other two authored
        # corners. Dividing its ray product by its *actual* corner product
        # is therefore the i-th barycentric dual action, with all authored
        # ray magnitudes retained. This uses the shared integer arrangement
        # operands directly, not a second inverse/elimination per chart.
        homogeneous = tuple(
            Fraction(value) / orientation
            for value, orientation in zip(products, cell.orientations, strict=True)
        )
        total = homogeneous[0] + homogeneous[1] + homogeneous[2]
        if total == 0:
            raise ValueError(
                "sphere_chart.arrangement.vertices must have a finite source pullback; got a ray at the source chart horizon."
            )
        point = homogeneous[1] / total, homogeneous[2] / total
        known = point, total
        self.coordinates[key] = known
        return known

    def _prepare_kernel_points(
        self,
        pieces: sphere_chart_records.SphereChartPieceBankBuilder,
        row: int,
        rays: tuple[_VertexKey, ...],
        *,
        target: bool = False,
    ) -> tuple[_SphereKernelPoint, ...]:
        """Publish each owned exact chart/ray operand once in this live builder."""
        if pieces._budget is not self.budget or (
            self._kernel_builder is not None and self._kernel_builder is not pieces
        ):
            raise ValueError("sphere_chart.arrangement.points must use its actual live builder and request.")
        self._kernel_builder = pieces
        operands: list[_Point | _SphereKernelPoint] = []
        keys: list[tuple[bool, int, _Integer3]] = []
        self._admit(1, 0)
        with self.budget.temporary_scope():
            self.budget.reserve(0, 256 + 128 * len(rays))
            for ray in rays:
                if not isinstance(ray, tuple):
                    raise ValueError("sphere_chart.arrangement.points must name actual primitive rays.")
                key = target, row, ray
                self._admit(1)
                token = self._kernel_points.get(key)
                operands.append(
                    self._coordinates(row, ray, target=target)[0]
                    if token is None else token
                )
                keys.append(key)
            result = pieces._prepare_kernel_points(operands)
            for key, token in zip(keys, result, strict=True):
                self._admit(1)
                if key not in self._kernel_points:
                    self._retain(3, max(abs(value).bit_length() for value in key[2]), 256)
                    self._kernel_points[key] = token
            return result





def _support_edges(
    actions: SphereProjectiveActionBank, source_row: int, target_row: int, /
) -> tuple[frozenset[bytes], ...]:
    """Name each clipping halfplane by its exact great-circle corner rays.

    Labels 0, 1 and 2 are the source edges ``x = 0``, ``y = 0`` and
    ``x + y = 1``; label ``3 + i`` is the target edge opposite corner ``i``.
    Two distinct labelled lines meet in one exact point of the source chart.
    """
    source = tuple(actions.corner_key(False, source_row, slot) for slot in range(3))
    target = tuple(actions.corner_key(True, target_row, slot) for slot in range(3))
    return (
        frozenset((source[0], source[2])),
        frozenset((source[0], source[1])),
        frozenset((source[1], source[2])),
        frozenset((target[1], target[2])),
        frozenset((target[0], target[2])),
        frozenset((target[0], target[1])),
    )


def _native_polygon(
    mapping: SphereProjectiveReferenceMap,
    memory: int,
    maximum_work: int,
    edges: tuple[frozenset[bytes], ...],
    constructed: dict[_SupportKey, _Point],
    storage: CoordinateLiveStorage,
    *,
    source_row: int,
    target_corner_keys: tuple[bytes, ...],
    target_corner_points: dict[tuple[int, bytes], _Point],
    budget: CoordinateEnclosureBudget,
) -> tuple[
    tuple[_Point, ...],
    tuple[int, ...],
    tuple[_SupportKey, ...],
    tuple[int | None, ...],
    int,
]:
    """Clip natively, then reconstruct every vertex from its two exact supports.

    ``target_corner_keys`` must come from the authenticated action bank for
    this target row. ``target_corner_points`` belongs to this request only;
    source rows and exact binary rays name its entries. Reused points still
    undergo every polygon validation.

    Also returns each vertex's exact support key and, for a vertex on two
    target edges, the target corner it names.
    """
    planes = _planes(mapping)

    supports, labels, work = _meshcore.clip_reference_triangle_exact(
        planes,
        maximum_scratch_bytes=memory,
        maximum_work_units=maximum_work,
    )
    count = supports.shape[0]
    if count < 3:
        return (), (), (), (), work
    edge_labels = tuple(int(value) for value in labels)
    vertices: list[_Point] = []
    keys: list[_SupportKey] = []
    target_corners: list[int | None] = []
    for slot in range(count):
        first_label, second_label = map(int, supports[slot])
        first = _support(first_label, planes)
        second = _support(second_label, planes)
        # Native support identities, not coordinate coincidence, name the
        # already-authored corners of the source reference triangle.
        point = _SOURCE_CORNERS.get(
            (min(first_label, second_label), max(first_label, second_label))
        )
        key = frozenset((edges[first_label], edges[second_label]))
        keys.append(key)
        # Label 3 + i is the target edge opposite corner i: two of them meet
        # exactly at the remaining target corner ray.
        target_corner = (
            3 - (first_label - 3) - (second_label - 3)
            if first_label >= 3 and second_label >= 3
            else None
        )
        target_corners.append(target_corner)
        corner_key = (
            (source_row, target_corner_keys[target_corner])
            if target_corner is not None
            else None
        )
        if point is None and corner_key is not None:
            # Adjacent target cones use different great-circle pairs for
            # this same authenticated binary ray. Reuse the actual source
            # point, but retain every polygon-specific check below.
            point = target_corner_points.get(corner_key)
        if point is None:
            # The same two exact great circles meet in one source-chart
            # point for every target cone of this source cell; the checks
            # below re-prove each reuse against this polygon's halfplanes.
            point = constructed.get(key)
        if point is None:
            if min(first_label, second_label) < 3:
                # One support is a source reference edge: substitute its
                # exact parametrization into the other (target) line.
                edge, line = (
                    (first_label, second) if first_label < 3 else (second_label, first)
                )
                point, operations = _source_edge_vertex(edge, line, budget=budget)
                work += operations
            else:
                with budget.temporary_scope():
                    construction = prepare_exact_small_linear_actions(
                        ((first[0], first[1]), (second[0], second[1])),
                        ((first[2],), (second[2],)),
                        budget=budget,
                    )
                work += construction.operation_count
                if construction.actions is None:
                    raise ValueError(
                        "sphere_chart.native_polygon.reference_map must satisfy the regular reference-map contract; got Native boundary cannot reconstruct a nonsingular source rational vertex."
                    )
                point = construction.actions[0][0], construction.actions[1][0]
            storage.set_bound(
                storage.bound
                + 512
                + sum(
                    (value.numerator.bit_length() + value.denominator.bit_length() + 7)
                    // 8
                    + 128
                    for value in point
                )
            )
            constructed[key] = point
        if any(
            normal[0] * point[0] + normal[1] * point[1] > normal[2] for normal in planes
        ) or any(value < 0 or value > 1 for value in point):
            raise ValueError(
                "sphere_chart.native_polygon.pieces must satisfy the material reference contract; got Native topology fails source rational halfplane feasibility."
            )
        vertices.append(point)
    polygon = tuple(vertices)
    if len(set(polygon)) != len(polygon) or chart_geometry.area(polygon) <= 0:
        raise ValueError(
            "sphere_chart.native_polygon.pieces must satisfy the material reference contract; got Native polygon fails positive distinct-vertex construction."
        )
    for slot in range(count):
        first, second, third = (
            polygon[slot - 1],
            polygon[slot],
            polygon[(slot + 1) % count],
        )
        if (second[0] - first[0]) * (third[1] - second[1]) - (second[1] - first[1]) * (
            third[0] - second[0]
        ) <= 0:
            raise ValueError(
                "sphere_chart.native_polygon.pieces must satisfy the material reference contract; got Native polygon is not the strictly convex source rational intersection."
            )
        support = _support(edge_labels[slot], planes)
        if any(
            support[0] * point[0] + support[1] * point[1] != support[2]
            for point in (second, third)
        ):
            raise ValueError(
                "sphere_chart.native_polygon.pieces must satisfy the material reference contract; got Native edge does not retain its source rational boundary support."
            )
    for point, corner in zip(polygon, target_corners, strict=True):
        if corner is not None:
            corner_key = (source_row, target_corner_keys[corner])
            if corner_key not in target_corner_points:
                storage.set_bound(storage.bound + 256)
                target_corner_points[corner_key] = point
    return polygon, edge_labels, tuple(keys), tuple(target_corners), work


def _source_edge_vertex(
    edge: int,
    line: _Plane,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_Point, int]:
    """Meet one source reference edge with a rational line ``a x + b y = c``.

    Edges 0, 1 and 2 are ``x = 0``, ``y = 0`` and ``x + y = 1``; the unique
    intersection follows by substitution, without a generic elimination.
    """
    a, b, c = line
    pivot = (b, a, a - b)[edge]
    if pivot == 0:
        raise ValueError(
            "sphere_chart.native_polygon.reference_map must satisfy the regular reference-map contract; got Native boundary cannot reconstruct a nonsingular source rational vertex."
        )
    # One quotient on an axis edge; two differences, one quotient and the
    # complementary coordinate on the diagonal edge.
    operations = 1 if edge < 2 else 4
    numerator_bits, denominator_bits = (
        imported_linalg__hermitian_spectral.fraction_matrix_profile(((line,),), budget)
    )
    imported_linalg__hermitian_spectral.reserve_fraction_work(
        budget, operations, 4, 4 * (numerator_bits + denominator_bits) + 64
    )
    if edge == 0:
        return (Fraction(0), c / b), operations
    if edge == 1:
        return (c / a, Fraction(0)), operations
    x = (c - b) / pivot
    return (x, 1 - x), operations


class _SphereEndpointCertificates(StrictModule):
    validities: list[CellValidityCertificate]
    embeddings: list[GlobalEmbeddingCertificate]
    measures: list[_FloatArray]
    errors: list[_FloatArray]


def _certify_sphere_endpoints(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None,
    prepared_target_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None,
    validity_policy: CellValidityPolicy | None,
    certificate_limits: MeshCertificateLimits | None,
    candidates_limit: int,
    work_limit: int,
    memory_limit: int,
    measure_absolute_tolerance: float,
    measure_relative_tolerance: float,
    maximum_measure_work: int,
    maximum_measure_subcells: int,
    maximum_binomial_terms: int,
    budget: CoordinateEnclosureBudget,
) -> _SphereEndpointCertificates:
    "Certify sphere endpoints and enclose their physical cell measures."
    validity_policy = (
        CellValidityPolicy(
            maximum_piece_count=min(work_limit, memory_limit // 1024),
            maximum_bernstein_nodes=min(work_limit, memory_limit // 1024),
        )
        if validity_policy is None
        else validity_policy
    )
    limits = (
        MeshCertificateLimits(
            maximum_candidate_pairs=candidates_limit,
            maximum_source_samples=work_limit,
            maximum_distance_evaluations=work_limit,
            maximum_subdivision_pieces=min(work_limit, memory_limit // 1024),
            maximum_bernstein_nodes=min(work_limit, memory_limit // 1024),
        )
        if certificate_limits is None
        else certificate_limits
    )
    validities: list[CellValidityCertificate] = []
    embeddings: list[GlobalEmbeddingCertificate] = []
    measures: list[_FloatArray] = []
    errors: list[_FloatArray] = []
    with budget.activate():
        for atlas, prepared in (
            (source_atlas, prepared_source_certificates),
            (target_atlas, prepared_target_certificates),
        ):
            _endpoint = certify_chart_endpoint(
                atlas.source_mesh,
                atlas.source_geometry,
                validity_policy,
                limits,
                prepared,
                budget=budget,
                stale_message="sphere_chart.certify_sphere_endpoints.pieces must satisfy the material reference contract; got Prepared sphere certificates change the source proof controls.",
            )
            validity, embedding = _endpoint.validity, _endpoint.embedding
            if not validity.all_certified or embedding.status != "certified":
                raise ValueError(
                    "sphere_chart.certify_sphere_endpoints.validity must satisfy the positive cell validity contract; got Sphere deformation requires independent source/target validity and global embedding."
                )
            values, bounds, _ = mapped_measure_owner.certified_cell_measures(
                atlas.source_mesh,
                atlas.source_geometry,
                absolute_tolerance=measure_absolute_tolerance,
                relative_tolerance=measure_relative_tolerance,
                maximum_work=maximum_measure_work,
                maximum_subcells=maximum_measure_subcells,
                maximum_binomial_terms=maximum_binomial_terms,
                budget=budget,
            )
            if (
                np.any(~np.isfinite(values))
                or np.any(~np.isfinite(bounds))
                or np.any(bounds < 0)
                or np.any(values <= bounds)
            ):
                raise ValueError(
                    "sphere_chart.certify_sphere_endpoints.measures must satisfy the positive physical measure contract; got Sphere physical cell measures require quantitative strictly positive enclosures."
                )
            validities.append(validity)
            embeddings.append(embedding)
            measures.append(values)
            errors.append(bounds)
    return _SphereEndpointCertificates(
        validities=validities, embeddings=embeddings, measures=measures, errors=errors
    )


class _SphereMaterialOverlaps(StrictModule):
    pieces: sphere_chart_records.SphereChartPieceBank
    source_coverage: list[Fraction]
    target_coverage: list[Fraction]
    maximum_bound: float
    retained: int
    work: int


class _SphereOverlapPublication(StrictModule):
    retained: int


def _publish_sphere_overlap_fan(
    source_points: tuple[_SphereKernelPoint, ...],
    target_points: tuple[_SphereKernelPoint, ...],
    bounds: tuple[SphereProjectiveTriangleBounds, ...],
    source_areas: tuple[Fraction, ...],
    target_areas: tuple[Fraction, ...],
    retained: int,
    memory_limit: int,
    pieces_limit: int,
    pieces: sphere_chart_records.SphereChartPieceBankBuilder,
    kernel_map: _SphereKernelMap,
    clip_id: str,
    source_id: int,
    target_id: int,
    physical_source: int,
    physical_target: int,
    source_row: int,
    target_row: int,
    group: tuple[str, tuple[str, ...]],
    labels: tuple[int, ...],
    displacements: tuple[float, ...],
) -> _SphereOverlapPublication:
    """Publish the complete actual fan without repeating its fixed metadata."""
    count = len(source_points) - 2
    if any(value <= 0 for value in (*source_areas, *target_areas)):
        raise ValueError(
            "sphere_chart.publish_sphere_overlap_fan.pieces must retain positive "
            "source and target reference areas; got a degenerate triangle"
        )
    previous_bytes = pieces.retained_bytes_upper
    if retained > memory_limit or len(pieces) + count > pieces_limit:
        raise ValueError(
            "sphere_chart.publish_sphere_overlap_fan.budget must satisfy the "
            "admitted resource limits; got excess retained memory or fan pieces"
        )
    piece_ids = tuple(
        canonical_fingerprint(
            {
                "kind": "sphere-exact-reference-piece",
                "clip": clip_id,
                "source": tuple(
                    source_points[index].exact for index in (0, slot, slot + 1)
                ),
                "target": tuple(
                    target_points[index].exact for index in (0, slot, slot + 1)
                ),
                "indices": (0, slot, slot + 1),
            }
        )
        for slot in range(1, len(source_points) - 1)
    )
    pieces._append_kernel_fan(
        kernel_map, source_points, target_points,
        source_cell_global_id=source_id,
        target_cell_global_id=target_id,
        source_row=physical_source,
        target_row=physical_target,
        source_atlas_row=source_row,
        target_atlas_row=target_row,
        geometry_entity_id=group[0],
        occurrence_path=group[1],
        projective_bounds=bounds,
        source_reference_areas=source_areas,
        target_reference_areas=target_areas,
        native_boundary_labels=labels,
        native_clip_id=clip_id,
        displacement_bounds=displacements,
        piece_ids=piece_ids,
    )
    retained += pieces.retained_bytes_upper - previous_bytes
    if retained > memory_limit:
        raise ValueError(
            "sphere_chart.publish_sphere_overlap_fan.budget must satisfy the "
            "admitted resource limits; got excess canonical bank memory"
        )
    return _SphereOverlapPublication(retained=retained)


class _SphereRadialDisplacementTerms(StrictModule):
    """Atlas-row terms of one same-ray chord displacement bound."""

    floor: _FloatArray
    ceiling: _FloatArray
    unit_error: _FloatArray
    polynomial: _FloatArray
    corners: _FloatArray


def _sphere_radial_displacement_terms(
    atlas: SphereMaterialCellAtlas,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _SphereRadialDisplacementTerms:
    """Collect the certified radial and chart terms already owned by ``atlas``.

    ``floor``/``ceiling`` enclose the ray-combination norm on each cone and
    the polynomial/corner terms are the same parts of the fidelity bound.
    Only the ray unit defect is re-enclosed from the retained directions.
    """
    with budget.temporary_scope():
        count = int(atlas.directions.shape[0])
        # Preserve the original norm-DAG scratch bound: absolute-value
        # arrays and outward sum/sqrt temporaries coexist with a possible
        # direction host copy. Only the returned row arrays outlive this call.
        budget.reserve(count * 3, 5 * count * 3 * 3 * 8 + 4096)
        directions = np.asarray(atlas.directions, dtype=np.float64)
        lows, highs = domain_numerics.interval_vector_norm((directions, directions))
        unit_error = np.max(
            np.maximum(
                np.maximum(
                    np.nextafter(highs - 1.0, np.inf),
                    np.nextafter(1.0 - lows, np.inf),
                ),
                0.0,
            ),
            axis=1,
        )
        result = _SphereRadialDisplacementTerms(
            floor=np.asarray(atlas.hemisphere_lower, dtype=np.float64),
            ceiling=np.asarray(atlas.direction_norm_upper, dtype=np.float64),
            unit_error=unit_error,
            polynomial=np.asarray(atlas.source_polynomial_chord_bounds, dtype=np.float64),
            corners=np.asarray(atlas.source_corner_enclosure_bounds, dtype=np.float64),
        )
        budget.retain_basis((result,))
        return result


def _sphere_same_ray_displacement(
    source: _SphereRadialDisplacementTerms,
    target: _SphereRadialDisplacementTerms,
    source_row: int,
    target_row: int,
    source_norm: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Bound both charts at one projectively shared material ray.

    Each chart differs from its corner chord by its polynomial term and its
    corners differ from the exact source images by its corner term. Both
    exact chords are ``A + B * sum(weights * unit rays)`` for the same source
    map, and the projective action makes their ray combinations positive
    multiples of one direction. Hence their difference is ``B`` applied to a
    norm difference enclosed by the two cone floors/ceilings plus both unit
    defects; ``source_norm`` is the certified upper norm of ``B``.
    """
    # Eight rational sums/products and one outward endpoint.
    budget.reserve(9, 512)
    floor = (
        Fraction(float(source.floor[source_row])),
        Fraction(float(target.floor[target_row])),
    )
    ceiling = (
        Fraction(float(source.ceiling[source_row])),
        Fraction(float(target.ceiling[target_row])),
    )
    spread = max(ceiling[0] - floor[1], ceiling[1] - floor[0], Fraction(0))
    return (
        Fraction(float(source.polynomial[source_row]))
        + Fraction(float(source.corners[source_row]))
        + Fraction(float(target.polynomial[target_row]))
        + Fraction(float(target.corners[target_row]))
        + source_norm
        * (
            spread
            + Fraction(float(source.unit_error[source_row]))
            + Fraction(float(target.unit_error[target_row]))
        )
    )


def _sphere_material_overlaps(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    source_candidates: _IntArray,
    target_candidates: _IntArray,
    source_bounds: _FloatArray,
    target_bounds: _FloatArray,
    maximum_displacement: float,
    pieces_limit: int,
    work_limit: int,
    memory_limit: int,
    retained: int,
    work: int,
    budget: CoordinateEnclosureBudget,
    actions: SphereProjectiveActionBank,
    storage: CoordinateLiveStorage,
    orientations: _SphereCandidateOrientations | None,
) -> _SphereMaterialOverlaps:
    "Assemble the shared edge overlay and certify both reference partitions."
    pieces = sphere_chart_records.SphereChartPieceBankBuilder(
        budget=budget, maximum_memory_bytes=memory_limit - retained
    )
    try:
        base_retained = retained
        source_coverage = [Fraction(0) for _ in range(source_atlas.num_charts)]
        target_coverage = [Fraction(0) for _ in range(target_atlas.num_charts)]
        maximum_bound = 0.0
        coordinate_maps: (
            tuple[
                tuple[tuple[Expression, ...], ...],
                tuple[tuple[Expression, ...], ...],
            ]
            | None
        ) = None
        radial_terms: (
            tuple[_SphereRadialDisplacementTerms, _SphereRadialDisplacementTerms] | None
        ) = None
        source_norms: dict[int, Fraction] = {}
        arrangement = _SphereArrangementOperands(actions, storage, budget=budget)
        # Bind only actual same-occurrence cone pairs before preparing shared
        # edge incidences. Geometrically coincident scientific occurrences
        # remain distinct query identities.
        count = source_candidates.size
        budget.reserve(4 * count, 17 * count + 4096)
        source_groups = tuple(
            zip(source_atlas.geometry_entity_ids, source_atlas.occurrence_paths, strict=True)
        )
        target_groups = tuple(
            zip(target_atlas.geometry_entity_ids, target_atlas.occurrence_paths, strict=True)
        )
        compatible = np.fromiter(
            (
                source_groups[int(source)] == target_groups[int(target)]
                for source, target in zip(
                    source_candidates, target_candidates, strict=True
                )
            ),
            dtype=np.bool_,
            count=count,
        )
        source_candidates = source_candidates[compatible]
        target_candidates = target_candidates[compatible]
        overlay = _SphereEdgeOverlay(arrangement, orientations)
        work += overlay.prepare(
            source_candidates, target_candidates,
            maximum_work=work_limit - work,
            maximum_memory=memory_limit - retained,
        )
        for source_index, target_index in zip(
            source_candidates, target_candidates, strict=True
        ):
            with budget.temporary_scope():
                source_row, target_row = int(source_index), int(target_index)
                group = (
                    source_atlas.geometry_entity_ids[source_row],
                    source_atlas.occurrence_paths[source_row],
                )
                if group != (
                    target_atlas.geometry_entity_ids[target_row],
                    target_atlas.occurrence_paths[target_row],
                ):
                    continue
                source_id = int(np.asarray(source_atlas.cell_global_ids)[source_row])
                target_id = int(np.asarray(target_atlas.cell_global_ids)[target_row])
                if work_limit - work - 64 <= 0:
                    raise ValueError(
                        "sphere_chart.sphere_material_overlaps.budget must satisfy the admitted resource limits; got Sphere correspondence exhausts work before exact arrangement."
                    )
                polygon, labels, keys, _, intersection_work = overlay.polygon(
                    source_row,
                    target_row,
                    maximum_work=work_limit - work - 64,
                    maximum_memory=memory_limit - retained,
                )
                work += intersection_work + 64
                if work > work_limit:
                    raise ValueError(
                        "sphere_chart.sphere_material_overlaps.budget must satisfy the admitted resource limits; got Shared sphere edge reconstruction exceeds its declared work budget."
                    )
                if not polygon:
                    continue
                mapping = source_atlas.projective_reference_map(
                    source_id, target_atlas, target_id, budget=budget, actions=actions
                )
                clip_id = canonical_fingerprint(
                    {
                        "kind": "sphere-exact-great-circle-arrangement-boundary",
                        "map": mapping.map_id,
                        "vertices": _tokens(polygon),
                        "support_labels": labels,
                    }
                )
                physical_source = int(np.asarray(source_atlas.physical_rows)[source_row])
                physical_target = int(np.asarray(target_atlas.physical_rows)[target_row])
                # Both coordinate carriers approximate the same radial
                # point selected by the projective map. Their certified
                # source-fidelity radii therefore give a whole triangle-inequality
                # displacement bound without expanding a removable rational
                # difference. When it is too coarse, the two exact chords on the
                # shared ray give an independently certified same-ray bound. Fall
                # back to the coefficient proof only when both are too coarse.
                budget.reserve(1, 512)
                displacement = fraction_endpoint(
                    Fraction(float(source_bounds[physical_source]))
                    + Fraction(float(target_bounds[physical_target])),
                    upper=True,
                    saturate=True,
                )
                patch = int(np.asarray(source_atlas.patches)[source_row])
                if displacement > maximum_displacement and patch == int(
                    np.asarray(target_atlas.patches)[target_row]
                ):
                    if radial_terms is None:
                        radial_terms = (
                            _sphere_radial_displacement_terms(source_atlas, budget=budget),
                            _sphere_radial_displacement_terms(target_atlas, budget=budget),
                        )
                    if patch not in source_norms:
                        source_norms[patch] = Fraction(
                            imported_geometry__sphere_material_atlas.sphere_radial_source_norm(
                                source_atlas.domain.patches[patch].surface, budget=budget
                            )
                        )
                    displacement = min(
                        displacement,
                        fraction_endpoint(
                            _sphere_same_ray_displacement(
                                radial_terms[0],
                                radial_terms[1],
                                source_row,
                                target_row,
                                source_norms[patch],
                                budget=budget,
                            ),
                            upper=True,
                            saturate=True,
                        ),
                    )
                displacement_map: tuple[Expression, ...] | None = None
                if displacement > maximum_displacement:
                    if coordinate_maps is None:
                        coordinate_maps = (
                            _sphere_full_coordinate_expressions(source_atlas, budget=budget),
                            _sphere_full_coordinate_expressions(target_atlas, budget=budget),
                        )
                    displacement_map = _sphere_displacement_map(
                        coordinate_maps[0][physical_source],
                        coordinate_maps[1][physical_target],
                        mapping,
                        budget=budget,
                    )
                images = _sphere_arrangement_images(
                    arrangement, source_row, target_row, polygon, keys
                )
                source_areas = _fan_areas(polygon)
                target_areas = _projective_fan_areas(mapping, images, source_areas)
                pieces.set_memory_limit(
                    memory_limit - base_retained - storage.bound - 4096
                )
                kernel_map = pieces._prepare_kernel_map(mapping)
                source_points = arrangement._prepare_kernel_points(pieces, source_row, keys)
                target_points = arrangement._prepare_kernel_points(
                    pieces, target_row, keys, target=True
                )
                retained = base_retained + pieces.retained_bytes_upper
                # The entire fan's actual bounds/IDs coexist before its single
                # bulk append; reserve the same scalar visits ahead of them.
                budget.reserve(2 * len(source_areas), 4096 * len(source_areas))
                fan_bounds = []
                fan_displacements = []
                for slot in range(1, len(polygon) - 1):
                    vertices: _Triangle = polygon[0], polygon[slot], polygon[slot + 1]
                    bounds, _ = _sphere_piece_geometry(
                        mapping, (images[0], images[slot], images[slot + 1])
                    )
                    source_area = source_areas[slot - 1]
                    target_area = target_areas[slot - 1]
                    if displacement_map is not None:
                        displacement = sphere_full_map_displacement(
                            displacement_map, vertices, budget=budget
                        )
                    fan_bounds.append(bounds)
                    fan_displacements.append(displacement)
                    maximum_bound = max(maximum_bound, displacement)
                    if displacement > maximum_displacement:
                        raise ValueError(
                            "sphere_chart.sphere_material_overlaps.displacement must satisfy the whole-piece displacement bound; got full-coordinate sphere deformation exceeds its whole-piece displacement bound."
                        )
                    source_coverage[source_row] += source_area
                    target_coverage[target_row] += target_area
                pieces.set_memory_limit(
                    memory_limit - base_retained - storage.bound - 4096
                )
                _phase = _publish_sphere_overlap_fan(
                    source_points=source_points, target_points=target_points,
                    bounds=tuple(fan_bounds),
                    source_areas=source_areas, target_areas=target_areas,
                    retained=retained, memory_limit=memory_limit,
                    pieces_limit=pieces_limit, pieces=pieces, kernel_map=kernel_map,
                    clip_id=clip_id, source_id=source_id, target_id=target_id,
                    physical_source=physical_source, physical_target=physical_target,
                    source_row=source_row, target_row=target_row,
                    group=group, labels=labels, displacements=tuple(fan_displacements),
                )
                retained = _phase.retained
        pieces.set_memory_limit(memory_limit - base_retained - storage.bound)
        bank = pieces.finish()
        return _SphereMaterialOverlaps(
            pieces=bank,
            source_coverage=source_coverage,
            target_coverage=target_coverage,
            maximum_bound=maximum_bound,
            retained=base_retained + bank.retained_bytes_upper,
            work=work,
        )
    finally:
        pieces.close()


class _SphereOccurrences(StrictModule):
    occurrences: list[sphere_chart_records.PreparedSphereChartOccurrence]


def _group_sphere_occurrences(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    source_groups: set[tuple[str, tuple[str, ...]]],
    piece_metadata: tuple[tuple[str, str, tuple[str, ...]], ...],
) -> _SphereOccurrences:
    "Group overlap pieces without merging scientific source occurrences."
    occurrences: list[sphere_chart_records.PreparedSphereChartOccurrence] = []
    for entity, path in sorted(source_groups):
        indices = tuple(
            index
            for index, (_, piece_entity, piece_path) in enumerate(piece_metadata)
            if (piece_entity, piece_path) == (entity, path)
        )
        source_ids = tuple(
            int(identifier)
            for identifier, group in zip(
                np.asarray(source_atlas.cell_global_ids),
                zip(
                    source_atlas.geometry_entity_ids,
                    source_atlas.occurrence_paths,
                    strict=True,
                ),
                strict=True,
            )
            if group == (entity, path)
        )
        target_ids = tuple(
            int(identifier)
            for identifier, group in zip(
                np.asarray(target_atlas.cell_global_ids),
                zip(
                    target_atlas.geometry_entity_ids,
                    target_atlas.occurrence_paths,
                    strict=True,
                ),
                strict=True,
            )
            if group == (entity, path)
        )
        occurrences.append(
            sphere_chart_records.PreparedSphereChartOccurrence(
                geometry_entity_id=entity,
                occurrence_path=path,
                piece_indices=indices,
                source_cell_global_ids=source_ids,
                target_cell_global_ids=target_ids,
                coverage_status="certified",
            )
        )
    return _SphereOccurrences(occurrences=occurrences)


# This bounded host coordinator retains admission, overlap, certification, and publication order; numerical phases already have owners.
def _prepare_sphere_chart_deformation(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    /,
    *,
    maximum_fidelity: float,
    maximum_displacement: float,
    maximum_candidate_pairs: int = 1_000_000,
    maximum_pieces: int = 1_000_000,
    maximum_work_units: int = 10_000_000,
    maximum_memory_bytes: int = 256 * 1024**2,
    measure_absolute_tolerance: float = 1e-10,
    measure_relative_tolerance: float = 1e-10,
    maximum_measure_work: int = 100_000_000,
    maximum_measure_subcells: int = 10_000,
    maximum_binomial_terms: int = 32,
    budget: CoordinateEnclosureBudget,
    certificate_limits: MeshCertificateLimits | None = None,
    validity_policy: CellValidityPolicy | None = None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
    prepared_target_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
) -> sphere_chart_records.PreparedSphereChartDeformation:
    "Certify closed material cells and retained projective overlap pieces."

    candidates_limit = positive_integer(
        maximum_candidate_pairs, "maximum_candidate_pairs"
    )
    pieces_limit = positive_integer(maximum_pieces, "maximum_pieces")
    work_limit = positive_integer(maximum_work_units, "maximum_work_units")
    memory_limit = positive_integer(maximum_memory_bytes, "maximum_memory_bytes")
    if any(
        not math.isfinite(value) or value < 0
        for value in (maximum_fidelity, maximum_displacement)
    ):
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.fidelity must satisfy the declared fidelity bound; got Sphere deformation fidelity/displacement limits must be finite and nonnegative."
        )
    for atlas in (source_atlas, target_atlas):
        atlas.require_bound(
            atlas.domain,
            atlas.source_mesh,
            atlas.source_geometry,
            atlas.coordinate_contract,
            budget=budget,
        )
        atlas.source_mesh.require_dense("Sphere material deformation")
    if (
        source_atlas.domain_id,
        source_atlas.source_id,
        source_atlas.source_revision,
        source_atlas.coordinate_contract.spatial_id,
    ) != (
        target_atlas.domain_id,
        target_atlas.source_id,
        target_atlas.source_revision,
        target_atlas.coordinate_contract.spatial_id,
    ):
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.binding must satisfy the declared geometry and cell identities; got Sphere source and target atlases must retain one revision, domain and spatial contract."
        )
    source_groups = set(
        zip(source_atlas.geometry_entity_ids, source_atlas.occurrence_paths, strict=True)
    )
    target_groups = set(
        zip(target_atlas.geometry_entity_ids, target_atlas.occurrence_paths, strict=True)
    )
    if source_groups != target_groups:
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.pieces must satisfy the material reference contract; got Sphere correspondence cannot omit or merge source occurrences."
        )
    count = source_atlas.num_charts + target_atlas.num_charts
    budget.reserve(count * 64, count * 1024)
    retained = count * 1024
    work = count * 64
    if retained >= memory_limit or work > work_limit:
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.pieces must satisfy the material reference contract; got Sphere correspondence exceeds its bounded preparation footprint."
        )
    source_bounds = sphere_chart_records.physical_order(
        source_atlas, source_atlas.source_fidelity_bounds
    )
    target_bounds = sphere_chart_records.physical_order(
        target_atlas, target_atlas.source_fidelity_bounds
    )
    if max(float(np.max(source_bounds)), float(np.max(target_bounds))) > maximum_fidelity:
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.fidelity must satisfy the declared fidelity bound; got source or target polynomial coordinate fidelity exceeds the sphere bound."
        )
    source_candidates, target_candidates, orientations = _candidates(
        source_atlas,
        target_atlas,
        candidates_limit,
        memory_limit - retained,
        budget=budget,
    )
    retained += source_candidates.nbytes + target_candidates.nbytes
    if work + source_candidates.size * 64 > work_limit:
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.budget must satisfy the admitted resource limits; got Sphere coefficient candidates exceed their preparation work budget."
        )
    # One authenticated action bank owns every source-target map of this
    # request; its exact cone inverses and shared vertex constructions live
    # until the overlap partition has been published.
    with budget.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(
            source_atlas, target_atlas, budget=budget, storage=storage
        )
        _phase = _sphere_material_overlaps(
            source_atlas=source_atlas,
            target_atlas=target_atlas,
            source_candidates=source_candidates,
            target_candidates=target_candidates,
            source_bounds=source_bounds,
            target_bounds=target_bounds,
            maximum_displacement=maximum_displacement,
            pieces_limit=pieces_limit,
            work_limit=work_limit,
            memory_limit=memory_limit,
            retained=retained,
            work=work,
            budget=budget,
            actions=actions,
            storage=storage,
            orientations=orientations,
        )
    pieces = _phase.pieces
    source_coverage = _phase.source_coverage
    target_coverage = _phase.target_coverage
    maximum_bound = _phase.maximum_bound
    retained = _phase.retained
    work = _phase.work
    if any(value != Fraction(1, 2) for value in (*source_coverage, *target_coverage)):
        raise ValueError(
            "sphere_chart.prepare_sphere_chart_deformation.pieces must satisfy the material reference contract; got Native sphere correspondence has a source/target gap or double cover; whole rational reference coverage required."
        )
    _phase = _certify_sphere_endpoints(
        source_atlas=source_atlas,
        target_atlas=target_atlas,
        prepared_source_certificates=prepared_source_certificates,
        prepared_target_certificates=prepared_target_certificates,
        validity_policy=validity_policy,
        certificate_limits=certificate_limits,
        candidates_limit=candidates_limit,
        work_limit=work_limit,
        memory_limit=memory_limit,
        measure_absolute_tolerance=measure_absolute_tolerance,
        measure_relative_tolerance=measure_relative_tolerance,
        maximum_measure_work=maximum_measure_work,
        maximum_measure_subcells=maximum_measure_subcells,
        maximum_binomial_terms=maximum_binomial_terms,
        budget=budget,
    )
    validities = _phase.validities
    embeddings = _phase.embeddings
    measures = _phase.measures
    errors = _phase.errors
    piece_metadata = pieces.metadata_rows(budget=budget)
    _phase = _group_sphere_occurrences(
        source_atlas=source_atlas,
        target_atlas=target_atlas,
        source_groups=source_groups,
        piece_metadata=piece_metadata,
    )
    occurrences = _phase.occurrences
    identity = canonical_fingerprint(
        {
            "kind": "prepared-sphere-projective-deformation",
            "source": source_atlas.atlas_id,
            "target": target_atlas.atlas_id,
            "pieces": tuple(item[0] for item in piece_metadata),
            "piece_bank": pieces.bank_id,
            "source_areas": _tokens(
                tuple((area, Fraction(0)) for area in source_coverage)
            ),
            "target_areas": _tokens(
                tuple((area, Fraction(0)) for area in target_coverage)
            ),
            "measure_values": array_tree_fingerprint(tuple(measures)),
            "measure_errors": array_tree_fingerprint(tuple(errors)),
            "embedding": tuple(value.certificate_id for value in embeddings),
        }
    )
    deformation = sphere_chart_records.PreparedSphereChartDeformation(
        source_atlas=source_atlas,
        target_atlas=target_atlas,
        source_mesh=source_atlas.source_mesh,
        target_mesh=target_atlas.source_mesh,
        source_geometry=source_atlas.source_geometry,
        target_geometry=target_atlas.source_geometry,
        source_geometry_id=source_atlas.source_geometry_id,
        target_geometry_id=target_atlas.source_geometry_id,
        source_topology_id=source_atlas.source_topology_id,
        target_topology_id=target_atlas.source_topology_id,
        domain_id=source_atlas.domain_id,
        pieces=pieces,
        occurrences=tuple(occurrences),
        source_reference_coverage=tuple(source_coverage),
        target_reference_coverage=tuple(target_coverage),
        source_fidelity_bounds=jnp.asarray(source_bounds, dtype=jnp.float64),
        target_fidelity_bounds=jnp.asarray(target_bounds, dtype=jnp.float64),
        source_cell_measures=jnp.asarray(measures[0], dtype=jnp.float64),
        target_cell_measures=jnp.asarray(measures[1], dtype=jnp.float64),
        source_measure_errors=jnp.asarray(errors[0], dtype=jnp.float64),
        target_measure_errors=jnp.asarray(errors[1], dtype=jnp.float64),
        source_validity=validities[0],
        target_validity=validities[1],
        source_embedding=embeddings[0],
        target_embedding=embeddings[1],
        maximum_displacement_bound=maximum_bound,
        candidate_pair_count=int(source_candidates.size),
        work_units=work,
        retained_bytes_upper=retained,
        deformation_id=identity,
    )
    # This request has just proved every piece, map, coverage and endpoint
    # certificate; its later consumers reuse that proof under this ledger.
    sphere_chart_records.issue_sphere_chart_validation(deformation, budget)
    return deformation


@checked
def prepare_sphere_chart_deformation(
    source_atlas: SphereMaterialCellAtlas,
    target_atlas: SphereMaterialCellAtlas,
    /,
    *,
    maximum_fidelity: float,
    maximum_displacement: float,
    maximum_candidate_pairs: int = 1_000_000,
    maximum_pieces: int = 1_000_000,
    maximum_work_units: int = 10_000_000,
    maximum_memory_bytes: int = 256 * 1024**2,
    measure_absolute_tolerance: float = 1e-10,
    measure_relative_tolerance: float = 1e-10,
    maximum_measure_work: int = 100_000_000,
    maximum_measure_subcells: int = 10_000,
    maximum_binomial_terms: int = 32,
    budget: CoordinateEnclosureBudget | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
    validity_policy: CellValidityPolicy | None = None,
    prepared_source_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
    prepared_target_certificates: tuple[
        CellValidityCertificate, GlobalEmbeddingCertificate
    ]
    | None = None,
    scope: NativeExecutionScope | None = None,
) -> sphere_chart_records.PreparedSphereChartDeformation:
    """Prepare bounded chart correspondence with an explicit request ledger."""
    work = positive_integer(maximum_work_units, "maximum_work_units")
    memory = positive_integer(maximum_memory_bytes, "maximum_memory_bytes")
    with chart_execution.chart_native_execution(
        budget,
        scope,
        maximum_work=work,
        maximum_memory=memory,
        maximum_queries=work,
    ) as (budget, _native_ticket):
        return _prepare_sphere_chart_deformation(
            source_atlas,
            target_atlas,
            maximum_fidelity=maximum_fidelity,
            maximum_displacement=maximum_displacement,
            maximum_candidate_pairs=maximum_candidate_pairs,
            maximum_pieces=maximum_pieces,
            maximum_work_units=maximum_work_units,
            maximum_memory_bytes=maximum_memory_bytes,
            measure_absolute_tolerance=measure_absolute_tolerance,
            measure_relative_tolerance=measure_relative_tolerance,
            maximum_measure_work=maximum_measure_work,
            maximum_measure_subcells=maximum_measure_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
            certificate_limits=certificate_limits,
            validity_policy=validity_policy,
            prepared_source_certificates=prepared_source_certificates,
            prepared_target_certificates=prepared_target_certificates,
            budget=budget,
        )


__all__ = [
    "prepare_sphere_chart_deformation",
    "reconstruct_sphere_material_cell_geometry",
]


class _SphereRadialNodeEnclosure(StrictModule):
    error: float


def _sphere_radial_node_enclosure(
    source_atlas: SphereMaterialCellAtlas,
    domain: imported_geometry__meshing_domain.MeshingDomain,
    patch: int,
    slot: int,
    physical: _FloatArray,
    local: int,
    nodes: _FloatArray,
    rays: _FloatArray,
    directions: _FloatArray,
    row: int,
    budget: CoordinateEnclosureBudget,
) -> _SphereRadialNodeEnclosure:
    "Bound radial node realization and barycentric ray rounding errors."
    low, high = imported_geometry__sphere_material_atlas.source_image_enclosure(
        domain.patches[patch].surface, rays, budget=budget
    )
    point_error = max(
        sum(
            (
                max(
                    abs(
                        Fraction(float(physical[local, node, axis]))
                        - Fraction(float(low[node, axis]))
                    ),
                    abs(
                        Fraction(float(physical[local, node, axis]))
                        - Fraction(float(high[node, axis]))
                    ),
                )
                for axis in range(3)
            ),
            Fraction(0),
        )
        for node in range(nodes.shape[0])
    )
    # barycentric ray arithmetic separately encloses host dot-product
    # rounding; no fitted radial polynomial is used as the source.
    ray_error = Fraction(0)
    for node in range(nodes.shape[0]):
        weights = (
            1 - Fraction(float(nodes[node, 0])) - Fraction(float(nodes[node, 1])),
            Fraction(float(nodes[node, 0])),
            Fraction(float(nodes[node, 1])),
        )
        exact = tuple(
            sum(
                (
                    weights[corner] * Fraction(float(directions[row, corner, axis]))
                    for corner in range(3)
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        ray_error = max(
            ray_error,
            sum(
                (
                    abs(exact[axis] - Fraction(float(rays[node, axis])))
                    for axis in range(3)
                ),
                Fraction(0),
            ),
        )
    floor = float(np.min(domain_numerics.interval_vector_norm((rays, rays))[0]))
    if Fraction(floor) <= ray_error:
        raise ValueError(
            "sphere_chart.sphere_radial_node_enclosure.pieces must satisfy the material reference contract; got Sphere coordinate reconstruction has an unresolved radial normalization."
        )
    radius = sum(
        (
            abs(Fraction(float(term)))
            for term in np.asarray(source_atlas.radius_terms)[slot]
        ),
        Fraction(0),
    )
    amplification = (
        radius
        * Fraction(domain_numerics.matrix_norm_upper(np.asarray(source_atlas.axes)[slot]))
        * Fraction(
            domain_numerics.matrix_norm_upper(np.asarray(source_atlas.rotations)[slot])
        )
    )
    error = fraction_endpoint(
        point_error + 6 * amplification * ray_error / Fraction(floor),
        upper=True,
        saturate=True,
    )
    return _SphereRadialNodeEnclosure(error=error)


class _SphereReconstructedNodes(StrictModule):
    prescribed_routes: list[_IntArray] | None
    values: list[_FloatArray]
    errors: list[float]
    evaluations: int


def _sphere_prescribed_corner_routes(
    blocks: tuple[imported_cell_geometry_transfer.SimplexGeometryBlock, ...],
) -> list[_IntArray] | None:
    """Identify layouts whose every coordinate node is already prescribed."""
    corner_reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    routes: list[_IntArray] = []
    for block in blocks:
        nodes = np.asarray(block.element.reference_nodes, dtype=np.float64)
        matches = np.all(nodes[:, None, :] == corner_reference[None, :, :], axis=-1)
        if not np.all(np.sum(matches, axis=1) == 1):
            return None
        routes.append(np.asarray(np.argmax(matches, axis=1), dtype=np.int64))
    return routes


def _sphere_reconstructed_node_values(
    source_atlas: SphereMaterialCellAtlas,
    target_mesh: CellMesh,
    blocks: tuple[imported_cell_geometry_transfer.SimplexGeometryBlock, ...],
    cells: _IntArray,
    count: int,
    patches: np.ndarray,
    directions: _FloatArray | None,
    prescribed_routes: list[_IntArray] | None,
    domain: imported_geometry__meshing_domain.MeshingDomain,
    budget: CoordinateEnclosureBudget,
) -> _SphereReconstructedNodes:
    "Realize target coordinate nodes while preserving prescribed corners."
    values: list[_FloatArray] = []
    errors: list[float] = []
    first, evaluations = 0, 0 if directions is None else 3 * count
    if prescribed_routes is not None:
        for block, route in zip(blocks, prescribed_routes, strict=True):
            rows = block.cell_ids.size
            budget.reserve(rows * route.size * 3, rows * route.size * 3 * 32)
            values.append(
                np.asarray(target_mesh.coordinates)[cells[first : first + rows][:, route]]
            )
            evaluations += rows * route.size
            first += rows
    else:
        if directions is None:
            raise ValueError(
                "sphere_chart.reconstruction.directions must be realized for interior coordinate nodes; got absent directions"
            )
        for block in blocks:
            nodes = np.asarray(block.element.reference_nodes, dtype=np.float64)
            barycentric = np.column_stack((1 - np.sum(nodes, axis=1), nodes))
            rows = block.cell_ids.size
            budget.reserve(rows * nodes.shape[0] * 128, rows * nodes.shape[0] * 3 * 32)
            physical = np.empty((rows, nodes.shape[0], 3), dtype=np.float64)
            for local in range(rows):
                row = first + local
                patch = int(patches[row])
                slot = int(
                    np.asarray(source_atlas.source_slots)[
                        np.flatnonzero(np.asarray(source_atlas.patches) == patch)[0]
                    ]
                )
                rays = barycentric @ directions[row]
                normalized = rays / np.linalg.norm(rays, axis=-1)[:, None]
                unit = normalized @ np.asarray(source_atlas.axes)[slot].T
                points = np.broadcast_to(
                    np.asarray(source_atlas.centers)[slot], unit.shape
                ).copy()
                for radius in np.asarray(source_atlas.radius_terms)[slot]:
                    points += radius * unit
                physical[local] = (
                    points @ np.asarray(source_atlas.rotations)[slot].T
                    + np.asarray(source_atlas.translations)[slot]
                )
                # Prescribed vertices are scientific edit coordinates. Retaining
                # them exactly avoids a second rounding epoch at shared corners.
                for corner, reference in enumerate(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))):
                    selected = np.flatnonzero(np.all(nodes == reference, axis=1))
                    physical[local, selected] = np.asarray(target_mesh.coordinates)[
                        cells[row, corner]
                    ]
                _phase = _sphere_radial_node_enclosure(
                    source_atlas=source_atlas,
                    domain=domain,
                    patch=patch,
                    slot=slot,
                    physical=physical,
                    local=local,
                    nodes=nodes,
                    rays=rays,
                    directions=directions,
                    row=row,
                    budget=budget,
                )
                error = _phase.error
                errors.append(error)
                evaluations += nodes.shape[0]
            values.append(physical)
            first += rows
    return _SphereReconstructedNodes(
        prescribed_routes=prescribed_routes,
        values=values,
        errors=errors,
        evaluations=evaluations,
    )


class _SphereReconstructionInputs(StrictModule):
    blocks: tuple[imported_cell_geometry_transfer.SimplexGeometryBlock, ...]
    cells: _IntArray
    count: int
    patches: np.ndarray
    dimensions: np.ndarray
    indices: np.ndarray
    parameters: _FloatArray
    domain: imported_geometry__meshing_domain.MeshingDomain
    before: int
    planned_evaluations: int
    directions: _FloatArray | None
    prescribed_routes: list[_IntArray] | None


def _prepare_sphere_reconstruction_nodes(
    source_atlas: SphereMaterialCellAtlas,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    cell_patches: ConvertibleToArray,
    corner_source_dimensions: ConvertibleToArray,
    corner_source_indices: ConvertibleToArray,
    corner_source_parameters: ConvertibleToArray,
    corner_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
    corner_source_entity_ids: tuple[tuple[str, ...], ...],
    maximum_fidelity: float,
    policy: transition_types_owner.CellGeometryTransitionPolicy,
    certificate_limits: MeshCertificateLimits,
    validity_policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
    native_ticket: NativeBatchTicket,
) -> _SphereReconstructionInputs:
    "Validate sphere reconstruction strata before admitting native queries."
    source_atlas.require_bound(
        source_atlas.domain,
        source_atlas.source_mesh,
        source_atlas.source_geometry,
        source_atlas.coordinate_contract,
        budget=budget,
    )
    if policy.reconstruction != "bounded_chart_deformation":
        raise ValueError(
            "sphere_chart.prepare_sphere_reconstruction_nodes.pieces must satisfy the material reference contract; got Sphere reconstruction requires the explicit bounded chart-deformation policy."
        )
    if not np.isfinite(maximum_fidelity) or maximum_fidelity < 0:
        raise ValueError(
            "sphere_chart.prepare_sphere_reconstruction_nodes.fidelity must satisfy the declared fidelity bound; got Sphere reconstruction requires finite nonnegative authored fidelity."
        )
    degree = imported_cell_geometry_transfer.nested_geometry_degree(
        source_atlas.source_mesh, source_atlas.source_geometry, budget=budget
    )
    blocks = imported_cell_geometry_transfer.simplex_blocks(
        target_mesh, target_layout, "sphere target", budget=budget
    )
    if any(
        block.element.cell_kind != "triangle" or block.degree != degree
        for block in blocks
    ):
        raise ValueError(
            "sphere_chart.prepare_sphere_reconstruction_nodes.pieces must satisfy the material reference contract; got Sphere reconstruction must retain the whole source coordinate degree."
        )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in target_mesh.blocks]
    )
    count = cells.shape[0]
    patches = np.asarray(cell_patches)
    dimensions, indices = (
        np.asarray(corner_source_dimensions),
        np.asarray(corner_source_indices),
    )
    parameters = np.asarray(corner_source_parameters, dtype=np.float64)
    if (
        patches.shape != (count,)
        or not np.issubdtype(patches.dtype, np.integer)
        or np.any(patches < 0)
        or np.any(patches >= len(source_atlas.domain.patches))
        or dimensions.shape != (count, 3)
        or not np.issubdtype(dimensions.dtype, np.integer)
        or indices.shape != (count, 3)
        or not np.issubdtype(indices.dtype, np.integer)
        or parameters.shape != (count, 3, 2)
        or len(corner_source_occurrence_paths) != count
        or len(corner_source_entity_ids) != count
        or any(
            len(row) != 3
            for row in (*corner_source_occurrence_paths, *corner_source_entity_ids)
        )
    ):
        raise ValueError(
            "sphere_chart.prepare_sphere_reconstruction_nodes.binding must satisfy the declared geometry and cell identities; got Sphere reconstruction strata must follow real target ordered corners."
        )
    domain = source_atlas.domain
    lookup = tuple(
        {
            (path, index): row
            for row, (path, index) in enumerate(zip(paths, ids, strict=True))
        }
        for paths, ids in zip(
            domain.source_occurrences, domain.source_indices, strict=True
        )
    )
    before = budget.work_units
    prescribed_routes = _sphere_prescribed_corner_routes(blocks)
    # Corner-only layouts copy prescribed edit coordinates. Realizing source
    # directions here would be unused work: the final target atlas below still
    # independently proves every source occurrence and the full coordinate map.
    budget.reserve(count * 64 if prescribed_routes is None else 0, count * 2048)
    planned_evaluations = (3 * count if prescribed_routes is None else 0) + sum(
        block.cell_ids.size * block.element.reference_nodes.shape[0] for block in blocks
    )
    if planned_evaluations > policy.maximum_evaluations:
        raise ValueError(
            "sphere_chart.prepare_sphere_reconstruction_nodes.budget must satisfy the admitted resource limits; got Sphere coordinate reconstruction exceeds source evaluation allowance."
        )
    native_ticket.consume(geometry_queries=planned_evaluations)
    directions = (
        np.empty((count, 3, 3), dtype=np.float64) if prescribed_routes is None else None
    )
    for row in range(count):
        patch = int(patches[row])
        for corner in range(3):
            dimension = int(dimensions[row, corner])
            if not 0 <= dimension <= 2:
                raise ValueError(
                    "sphere_chart.prepare_sphere_reconstruction_nodes.binding must satisfy the declared geometry and cell identities; got Sphere reconstruction requires source strata."
                )
            index = lookup[dimension].get(
                (corner_source_occurrence_paths[row][corner], int(indices[row, corner]))
            )
            if (
                index is None
                or domain.entity_id(dimension, index)
                != corner_source_entity_ids[row][corner]
            ):
                raise ValueError(
                    "sphere_chart.prepare_sphere_reconstruction_nodes.binding must satisfy the declared geometry and cell identities; got Sphere reconstruction changes source entity or occurrence identity."
                )
            if directions is not None:
                directions[row, corner], _ = (
                    imported_geometry__sphere_material_atlas.corner_direction(
                        domain, patch, dimension, index, parameters[row, corner],
                        budget=budget,
                    )
                )
    return _SphereReconstructionInputs(
        blocks=blocks,
        cells=cells,
        count=count,
        patches=patches,
        dimensions=dimensions,
        indices=indices,
        parameters=parameters,
        domain=domain,
        before=before,
        planned_evaluations=planned_evaluations,
        directions=directions,
        prescribed_routes=prescribed_routes,
    )


def _reconstruct_sphere_material_cell_geometry(
    source_atlas: SphereMaterialCellAtlas,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    /,
    *,
    cell_patches: ConvertibleToArray,
    corner_source_dimensions: ConvertibleToArray,
    corner_source_indices: ConvertibleToArray,
    corner_source_parameters: ConvertibleToArray,
    corner_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
    corner_source_entity_ids: tuple[tuple[str, ...], ...],
    maximum_fidelity: float,
    policy: transition_types_owner.CellGeometryTransitionPolicy,
    budget: CoordinateEnclosureBudget,
    native_ticket: NativeBatchTicket,
    certificate_limits: MeshCertificateLimits,
    validity_policy: CellValidityPolicy,
) -> sphere_chart_records.SphereGeometryReconstruction:
    "Reconstruct the source radial source at real target coordinate nodes.\n\n source expressions and coordinate banks remain bound in source_atlas.\n The successor's whole coordinate degree is retained, while its new full\n polynomial map receives independent fidelity, validity and embedding proofs."

    _phase = _prepare_sphere_reconstruction_nodes(
        source_atlas=source_atlas,
        target_mesh=target_mesh,
        target_layout=target_layout,
        cell_patches=cell_patches,
        corner_source_dimensions=corner_source_dimensions,
        corner_source_indices=corner_source_indices,
        corner_source_parameters=corner_source_parameters,
        corner_source_occurrence_paths=corner_source_occurrence_paths,
        corner_source_entity_ids=corner_source_entity_ids,
        maximum_fidelity=maximum_fidelity,
        policy=policy,
        certificate_limits=certificate_limits,
        validity_policy=validity_policy,
        budget=budget,
        native_ticket=native_ticket,
    )
    blocks = _phase.blocks
    cells = _phase.cells
    count = _phase.count
    patches = _phase.patches
    dimensions = _phase.dimensions
    indices = _phase.indices
    parameters = _phase.parameters
    domain = _phase.domain
    before = _phase.before
    planned_evaluations = _phase.planned_evaluations
    directions = _phase.directions
    prescribed_routes = _phase.prescribed_routes
    _phase = _sphere_reconstructed_node_values(
        source_atlas=source_atlas,
        target_mesh=target_mesh,
        blocks=blocks,
        cells=cells,
        count=count,
        patches=patches,
        directions=directions,
        prescribed_routes=prescribed_routes,
        domain=domain,
        budget=budget,
    )
    prescribed_routes = _phase.prescribed_routes
    values = _phase.values
    errors = _phase.errors
    evaluations = _phase.evaluations
    if evaluations > policy.maximum_evaluations:
        raise ValueError(
            "sphere_chart.reconstruct_sphere_material_cell_geometry.budget must satisfy the admitted resource limits; got Sphere coordinate reconstruction exceeds source evaluation allowance."
        )
    if evaluations != planned_evaluations:
        raise ValueError(
            "sphere_chart.reconstruct_sphere_material_cell_geometry.pieces must satisfy the material reference contract; got Sphere reconstruction evaluation accounting disagrees with the admitted node batch."
        )
    placed = imported_cell_geometry_transfer.place(
        blocks, values, target_layout.coordinates.shape[0]
    )
    extent = max(1.0, float(np.max(np.abs(placed.coordinates), initial=0.0)))
    if placed.continuity > policy.continuity_tolerance * extent:
        raise ValueError(
            "sphere_chart.reconstruct_sphere_material_cell_geometry.pieces must satisfy the material reference contract; got source sphere source realizations disagree at shared coordinate nodes."
        )
    elements, routes, _ = target_layout.resolve(target_mesh, budget=budget)
    names = tuple(block.name for block in target_mesh.blocks)
    geometry = CellGeometrySpec(
        dict(zip(names, elements, strict=True)),
        dict(zip(names, routes, strict=True)),
        placed.coordinates,
        budget=budget,
    )
    if target_mesh.periodic_topology is not None:
        geometry = geometry.with_periodic_source(target_mesh, budget=budget)
    with budget.activate(), budget.temporary_scope():
        atlas = SphereMaterialCellAtlas(
            domain,
            target_mesh,
            geometry,
            source_atlas.coordinate_contract,
            cell_patches=patches,
            corner_source_dimensions=dimensions,
            corner_source_indices=indices,
            corner_source_parameters=parameters,
            corner_source_occurrence_paths=corner_source_occurrence_paths,
            corner_source_entity_ids=corner_source_entity_ids,
            maximum_fidelity=maximum_fidelity,
            maximum_cells=count,
            maximum_work_units=budget.maximum_work_units,
            maximum_coordinate_memory_bytes=budget.maximum_memory_bytes,
            budget=budget,
        )
    if prescribed_routes is not None:
        # Corner-only layouts already retain the prescribed native edit points.
        # The fresh atlas above earned their full source-source enclosure;
        # there is no interior barycentric-ray rounding term at these nodes.
        errors.extend(
            float(value)
            for value in sphere_chart_records.physical_order(
                atlas, atlas.source_corner_enclosure_bounds
            )
        )
    with budget.activate(), budget.temporary_scope():
        validity = certify_cell_geometry_validity(
            geometry, mesh=target_mesh, policy=validity_policy, budget=budget
        )
        embedding = certify_global_embedding(
            target_mesh, geometry, validity, limits=certificate_limits, budget=budget
        )
    if not validity.all_certified or embedding.status != "certified":
        raise ValueError(
            "sphere_chart.reconstruct_sphere_material_cell_geometry.validity must satisfy the positive cell validity contract; got Sphere reconstruction fails full-map validity or global embedding."
        )
    bounds = sphere_chart_records.physical_order(atlas, atlas.source_fidelity_bounds)
    identity = canonical_fingerprint(
        {
            "kind": "sphere-radial-coordinate-reconstruction",
            "source": source_atlas.atlas_id,
            "target": atlas.atlas_id,
            "policy": policy.policy_id,
            "node_errors": array_tree_fingerprint(np.asarray(errors, dtype=np.float64)),
            "validity": validity.certificate_id,
            "embedding": embedding.certificate_id,
        }
    )
    return sphere_chart_records.SphereGeometryReconstruction(
        source_geometry_id=source_atlas.source_geometry_id,
        target_geometry_id=cell_geometry_id(geometry),
        source_topology_id=source_atlas.source_topology_id,
        target_topology_id=target_mesh.topology_id,
        domain_id=domain.domain_id,
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        geometry=geometry,
        vertex_coordinates=jnp.asarray(
            imported_cell_geometry_transfer.vertex_rows(
                target_mesh, blocks, placed.coordinates
            )
        ),
        target_atlas=atlas,
        target_validity=validity,
        target_embedding=embedding,
        cell_global_ids=jnp.asarray(np.concatenate([block.cell_ids for block in blocks])),
        cell_patches=jnp.asarray(patches),
        fidelity_bounds=jnp.asarray(bounds),
        node_enclosure_bounds=jnp.asarray(np.asarray(errors, dtype=np.float64)),
        node_owner_cells=jnp.asarray(placed.owner_cells),
        node_owner_locals=jnp.asarray(placed.owner_locals),
        continuity_residual=placed.continuity,
        evaluation_count=evaluations,
        preparation_work_units=budget.work_units - before,
        reconstruction_id=identity,
    )


@checked
def reconstruct_sphere_material_cell_geometry(
    source_atlas: SphereMaterialCellAtlas,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    /,
    *,
    cell_patches: ConvertibleToArray,
    corner_source_dimensions: ConvertibleToArray,
    corner_source_indices: ConvertibleToArray,
    corner_source_parameters: ConvertibleToArray,
    corner_source_occurrence_paths: tuple[tuple[tuple[str, ...], ...], ...],
    corner_source_entity_ids: tuple[tuple[str, ...], ...],
    maximum_fidelity: float,
    policy: CellGeometryTransitionPolicy,
    budget: CoordinateEnclosureBudget,
    certificate_limits: MeshCertificateLimits,
    validity_policy: CellValidityPolicy,
    scope: NativeExecutionScope | None = None,
) -> sphere_chart_records.SphereGeometryReconstruction:
    """Reconstruct sphere coordinates within the admitted native batch.

    Corner-only layouts retain prescribed edit coordinates without an unused
    preliminary source-direction realization. Interior nodes still realize
    their complete radial maps. Both routes earn a fresh source-bound target
    atlas, whole-map validity and global embedding before publication.
    """
    with chart_execution.chart_native_execution(
        budget,
        scope,
        maximum_work=budget.maximum_work_units,
        maximum_memory=budget.maximum_memory_bytes,
        maximum_queries=policy.maximum_evaluations,
    ) as (budget, native_ticket):
        return _reconstruct_sphere_material_cell_geometry(
            source_atlas,
            target_mesh,
            target_layout,
            cell_patches=cell_patches,
            corner_source_dimensions=corner_source_dimensions,
            corner_source_indices=corner_source_indices,
            corner_source_parameters=corner_source_parameters,
            corner_source_occurrence_paths=corner_source_occurrence_paths,
            corner_source_entity_ids=corner_source_entity_ids,
            maximum_fidelity=maximum_fidelity,
            policy=policy,
            budget=budget,
            native_ticket=native_ticket,
            certificate_limits=certificate_limits,
            validity_policy=validity_policy,
        )


type _ReferenceImage = tuple[Fraction, _Point]

# Target reference corners 0, 1 and 2: the homogeneous image is a positive
# multiple of one barycentric unit vector.
_TARGET_CORNERS: tuple[_Point, _Point, _Point] = (
    (Fraction(0), Fraction(0)),
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
)
# Source reference corners named by their two native source-edge labels.
_SOURCE_CORNERS: dict[tuple[int, int], _Point] = {
    (0, 1): _TARGET_CORNERS[0],
    (0, 2): _TARGET_CORNERS[2],
    (1, 2): _TARGET_CORNERS[1],
}


def _sphere_fraction_add(
    left: Fraction,
    right: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Omit addition only when an exact operand is zero."""
    if not left:
        return right
    if not right:
        return left
    budget.reserve(1)
    return left + right


def _sphere_affine_row(
    row: tuple[Fraction, ...],
    point: _Point,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Evaluate one exact affine row, charging only executed arithmetic.

    The caller admits the coefficient-derived storage bound before expansion.
    Zero coefficients/coordinates and unit factors return existing immutable
    operands; in particular the source origin returns the stored constant.
    """
    value = row[0]
    for index in range(2):
        coefficient, coordinate = row[index + 1], point[index]
        if not coefficient or not coordinate:
            continue
        if coefficient == 1:
            term = coordinate
        elif coordinate == 1:
            term = coefficient
        else:
            budget.reserve(1)
            term = coefficient * coordinate
        value = _sphere_fraction_add(value, term, budget=budget)
    return value


def _sphere_image_quotient(
    numerator: Fraction,
    denominator: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    """Divide by an already checked positive denominator without identities."""
    if not numerator:
        return _TARGET_CORNERS[0][0]
    if numerator == denominator:
        return _TARGET_CORNERS[1][0]
    if denominator == 1:
        return numerator
    budget.reserve(1)
    return numerator / denominator


def _sphere_reference_image(
    mapping: SphereProjectiveReferenceMap,
    point: _Point,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _ReferenceImage:
    """Retain the exact source/cone checks while omitting identity arithmetic."""
    if len(point) != 2 or any(not isinstance(value, Fraction) for value in point):
        raise TypeError(
            "sphere_projective.vertices must contain three pairs of Fraction coordinates; got another representation"
        )
    if (
        point[0] < 0
        or point[1] < 0
        or _sphere_fraction_add(point[0], point[1], budget=budget) > 1
    ):
        raise ValueError(
            "sphere_projective.vertices must lie in the source reference triangle; got exterior coordinate"
        )
    homogeneous = tuple(
        _sphere_affine_row(row, point, budget=budget)
        for row in mapping.exact_coefficients
    )
    # Preserve the existing three homogeneous-sign visits and denominator-sign
    # visit, independently of the arithmetic actually needed by these rows.
    budget.reserve(3)
    if any(value < 0 for value in homogeneous):
        raise ValueError(
            "sphere_projective.vertices must lie in the target direction cone; got negative barycentric coordinate"
        )
    denominator = _sphere_fraction_add(
        _sphere_fraction_add(homogeneous[0], homogeneous[1], budget=budget),
        homogeneous[2],
        budget=budget,
    )
    budget.reserve(1)
    if denominator <= 0:
        raise ValueError(
            "sphere_projective.reference must have positive denominator and orientation; got singular or reversed map"
        )
    return (
        denominator,
        (
            _sphere_image_quotient(homogeneous[1], denominator, budget=budget),
            _sphere_image_quotient(homogeneous[2], denominator, budget=budget),
        ),
    )


def _sphere_arrangement_images(
    arrangement: _SphereArrangementOperands,
    source_row: int,
    target_row: int,
    polygon: tuple[_Point, ...],
    keys: tuple[_VertexKey, ...],
) -> tuple[_ReferenceImage, ...]:
    """Factor the same projective map through both actual cone dual actions.

    For one canonical ray r, the two barycentric homogeneous sums are s and t.
    The source chart's physical affine image is r/s, so its target denominator
    is exactly t/s. Target reference coordinates use the same target dual
    actions, independent of which incident source cone names the ray.
    """
    images = []
    for point, ray in zip(polygon, keys, strict=True):
        if not isinstance(ray, tuple):
            raise ValueError(
                "sphere_chart.arrangement.images must name actual primitive rays; "
                "got a nonarrangement support key"
            )
        actual_source, source_sum = arrangement._coordinates(source_row, ray)
        target_point, target_sum = arrangement._coordinates(target_row, ray, target=True)
        bits = max(
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for value in (*point, *target_point, source_sum, target_sum)
        ) + 8
        arrangement._admit(8, 4, 4 * bits)
        denominator = target_sum / source_sum
        if (
            actual_source != point
            or denominator <= 0
            or min(target_point) < 0
            or target_point[0] + target_point[1] > 1
        ):
            raise ValueError(
                "sphere_chart.arrangement.images must retain actual source/target "
                "cone coordinates and positive denominator; got a foreign ray"
            )
        images.append((denominator, target_point))
    return tuple(images)


def _sphere_polygon_images(
    mapping: SphereProjectiveReferenceMap,
    polygon: tuple[_Point, ...],
    keys: tuple[_VertexKey, ...],
    corners: tuple[int | None, ...],
    target_row: int,
    images: dict[tuple[_VertexKey, int], _Point],
    storage: CoordinateLiveStorage,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_ReferenceImage, ...]:
    """Independent matrix-action image oracle with exact target-ray reuse.

    Keys may be canonical primitive rays or the native oracle's two-circle
    support keys; production publication instead uses the two cone duals.
    Inside one positive target cone either key names a unique physical ray,
    shared by all incident source cells. A vertex on two target edges is the
    remaining target corner. Only its source-chart denominator
    ``d . (1, x, y)`` is recomputed for this map.
    """
    count = len(polygon)
    # Image arithmetic is admitted at each executed operation, not at the
    # dense-row worst case: exact zero/unit operands genuinely skip work.
    # Source fan vertex after the apex: its two offsets (2). Per fan triangle:
    # denominator extrema (4), two cubic Jacobian quotients (6), source
    # cross-product area (4), and the exact projective area identity (4).
    # These downstream operations remain precharged. No target offsets arise.
    # Profile all actual inputs, including the stored determinant and the
    # denominator coefficients used by shared images. If every input numerator
    # and denominator has at most B bits, a homogeneous row needs at most
    # 5B+2, its three-row sum 15B+6, and an image quotient 20B+8.
    # A fan source area needs 8B+3. The largest subsequent expression is
    # det(H)*A/(d0*d1*d2), bounded by 54B+19 bits; the spare 13 bits cover
    # signed sums and division by two. This follows the actual expression DAG
    # rather than multiplying an already-expanded denominator bound again.
    work = 2 * (count - 1) + 18 * (count - 2)
    numerator_bits, denominator_bits = (
        imported_linalg__hermitian_spectral.fraction_matrix_profile(
            (
                mapping.exact_coefficients,
                polygon,
                (mapping.exact_denominator,),
                ((mapping.orientation_ratio,),),
            ),
            budget,
        )
    )
    bits = max(numerator_bits, denominator_bits)
    imported_linalg__hermitian_spectral.reserve_fraction_work(
        budget,
        work,
        256 * (count - 2),
        54 * bits + 32,
    )
    denominator = mapping.exact_denominator
    values: list[_ReferenceImage] = []
    for point, key, corner in zip(polygon, keys, corners, strict=True):
        shared = (
            _TARGET_CORNERS[corner]
            if corner is not None
            else images.get((key, target_row))
        )
        if shared is None:
            # Admit retained-image expansion before constructing any Fraction.
            # Each quotient's numerator and denominator need at most 20B+8
            # bits, including the exact homogeneous three-row sum.
            previous_bound = storage.bound
            storage.set_bound(
                previous_bound + 512 + 2 * ((40 * bits + 16 + 7) // 8 + 128)
            )
            value = _sphere_reference_image(mapping, point, budget=budget)
            storage.set_bound(
                previous_bound
                + 512
                + sum(
                    (part.numerator.bit_length() + part.denominator.bit_length() + 7) // 8
                    + 128
                    for part in value[1]
                )
            )
            images[(key, target_row)] = value[1]
            values.append(value)
            continue
        scale = _sphere_affine_row(denominator, point, budget=budget)
        budget.reserve(1)
        if scale <= 0:
            raise ValueError(
                "sphere_projective.reference must have positive denominator and orientation; got singular or reversed map"
            )
        values.append((scale, shared))
    return tuple(values)


def _fan_areas(points: tuple[_Point, ...], /) -> tuple[Fraction, ...]:
    """Signed areas of the apex fan ``(p0, p_i, p_i+1)`` from shared offsets."""
    offsets = tuple(
        (point[0] - points[0][0], point[1] - points[0][1]) for point in points[1:]
    )
    return tuple(
        (first[0] * second[1] - first[1] * second[0]) / 2
        for first, second in zip(offsets[:-1], offsets[1:], strict=True)
    )


def _projective_fan_areas(
    mapping: SphereProjectiveReferenceMap,
    images: tuple[_ReferenceImage, ...],
    source_areas: tuple[Fraction, ...],
    /,
) -> tuple[Fraction, ...]:
    """Exact target areas without constructing a second set of fan offsets.

    For homogeneous barycentric columns H and source triangle vertices p_i,
    det(H [p_0 p_1 p_2]) equals det(H) det([p_0 p_1 p_2]). Dividing each image
    column by its positive projective denominator d_i gives
    A_target = det(H) A_source / (d_0 d_1 d_2). The authenticated map's
    orientation_ratio is det(H), and its vertex images supply those d_i.
    """
    apex = images[0][0]
    return tuple(
        mapping.orientation_ratio * area / (apex * first[0] * second[0])
        for area, first, second in zip(
            source_areas, images[1:-1], images[2:], strict=True
        )
    )


def _sphere_piece_geometry(
    mapping: SphereProjectiveReferenceMap,
    images: tuple[_ReferenceImage, _ReferenceImage, _ReferenceImage],
    /,
) -> tuple[SphereProjectiveTriangleBounds, _Triangle]:
    "Assemble one fan triangle from its already admitted vertex images."

    bounds = mapping.triangle_bounds((images[0][0], images[1][0], images[2][0]))
    target: _Triangle = images[0][1], images[1][1], images[2][1]
    return bounds, target


def _sphere_full_coordinate_expressions(
    atlas: SphereMaterialCellAtlas,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Expression, ...], ...]:
    "Retain the immutable coefficient maps in scientific physical-cell order."

    with budget.activate():
        cells = mapped_measure_owner.mapped_geometry_cells(
            atlas.source_mesh, atlas.source_geometry, budget=budget
        )
        expressions: list[tuple[Expression, ...]] = []
        for element, bank in cells:
            with budget.temporary_scope():
                values = mapped_measure_owner.mapped_coordinate_expressions(
                    element, bank, budget=budget
                )
            expressions.append(values)
    return tuple(expressions)


def _sphere_displacement_map(
    source: tuple[Expression, ...],
    target: tuple[Expression, ...],
    mapping: SphereProjectiveReferenceMap,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...]:
    "Subtract the whole coordinate maps once in the source chart."

    with budget.activate():
        composition = ExpressionComposition(
            mapping.reference_expressions(), budget=budget
        )
        differences: list[Expression] = []
        for old, new in zip(source, target, strict=True):
            if (
                composition.composition is not None
                or isinstance(old, imported_coordinate_enclosure.RationalPolynomial)
                or isinstance(new, imported_coordinate_enclosure.RationalPolynomial)
            ):
                difference = imported_coordinate_enclosure.expression_add(
                    composition(new, budget=budget),
                    imported_coordinate_enclosure.expression_scale(
                        old, -1, budget=budget
                    ),
                    budget=budget,
                )
            else:
                numerator, denominator = composition._polynomial(new, budget=budget)
                numerator = imported_coordinate_enclosure.add(
                    numerator,
                    imported_coordinate_enclosure.scale(
                        imported_coordinate_enclosure.multiply(
                            old, denominator, budget=budget
                        ),
                        -1,
                        budget=budget,
                    ),
                    budget=budget,
                )
                # The source homogeneous denominator is already certified by
                # the projective owner. An enclosure needs its ratio, not
                # an expensive removable-factor search on every coordinate.
                difference = (
                    imported_coordinate_enclosure.RationalPolynomial(
                        numerator, denominator
                    )
                    if numerator
                    else {}
                )
            differences.append(difference)
        return tuple(differences)


@checked
def sphere_full_map_displacement(
    displacement_map: tuple[Expression, ...],
    vertices: _Triangle,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    "Enclose the whole displacement on an entire material piece."

    with budget.activate(), budget.temporary_scope():
        budget.reserve(len(displacement_map))
        if all(
            not isinstance(value, imported_coordinate_enclosure.RationalPolynomial)
            and not value
            for value in displacement_map
        ):
            # A ordered empty coefficient support is the zero polynomial on
            # every affine subpiece, independently of mesh or chart identity.
            return 0.0
        variables = imported_coordinate_enclosure.axes(2, budget=budget)
        reference = tuple(
            imported_coordinate_enclosure.add(
                imported_coordinate_enclosure.constant(
                    vertices[0][axis], 2, budget=budget
                ),
                imported_coordinate_enclosure.add(
                    imported_coordinate_enclosure.scale(
                        variables[0], vertices[1][axis] - vertices[0][axis], budget=budget
                    ),
                    imported_coordinate_enclosure.scale(
                        variables[1], vertices[2][axis] - vertices[0][axis], budget=budget
                    ),
                    budget=budget,
                ),
                budget=budget,
            )
            for axis in range(2)
        )
        # affine substitution commutes with subtraction. The source-chart
        # difference retains both full coordinate maps and is shared by all
        # triangles of this projective overlap, including cancellation.
        at_source = ExpressionComposition(reference, budget=budget)
        bound = Fraction(0)
        for difference in displacement_map:
            if (
                not isinstance(
                    difference, imported_coordinate_enclosure.RationalPolynomial
                )
                and not difference
            ):
                continue
            if isinstance(difference, imported_coordinate_enclosure.RationalPolynomial):
                polynomial_action = at_source.composition
                if polynomial_action is None:
                    raise RuntimeError(
                        "sphere_chart.sphere_full_map_displacement.pieces must satisfy the material reference contract; got A material piece lost its affine polynomial action."
                    )
                image = imported_coordinate_enclosure.RationalPolynomial(
                    polynomial_action(difference.numerator, budget=budget),
                    polynomial_action(difference.denominator, budget=budget),
                )
            else:
                image = at_source(difference, budget=budget)
            controls = imported_coordinate_enclosure.expression_bernstein_coefficients(
                image, "simplex", 2, budget=budget
            )
            bound += max(abs(value) for value in controls)
        # coefficient cancellation, not an identity-case bypass. The
        # ordered realization enclosure preserves a zero.
        return imported_geometry__sphere_material_atlas.float_enclosure(
            rational_coefficient(
                bound, role="sphere_chart.sphere_full_map_displacement.bound"
            )
        )[1]
