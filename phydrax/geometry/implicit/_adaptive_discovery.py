#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Adaptive error-controlled implicit surface discovery on a balanced octree.

The domain box is refined by a worklist over a 2:1-balanced linear octree
(`refined_octree_leaves`). Every leaf carries value and gradient bounds from the
selected enclosure. A leaf whose value enclosure excludes zero is certainly
inside or outside; no leaf is treated as empty merely because its samples do
not change sign.

With rigorous enclosures the final octree certifies a regular CW decomposition
of the zero set ``S``:

- every minimal edge either excludes zero or is monotone along its axis, so its
  endpoint signs give its exact root count (zero or one);
- every minimal face either excludes zero, or is monotone along an in-plane axis
  (no closed loop) with zero or two boundary crossings (at most one arc);
- every leaf meeting ``S`` is monotone along some axis, so ``S`` in the leaf is a
  graph over the orthogonal face, and its boundary carries exactly one cycle, so
  that graph is a closed disk.

Dual contouring then places one vertex per (leaf, cycle), one polygon per
crossing minimal edge, and one dual edge per face arc: the mesh is the dual cell
complex of that decomposition and is homeomorphic to ``S``. Level transitions
are crack free because polygons are generated from minimal edges shared by all
incident leaves. Exact-zero vertices are resolved by the symbolic perturbation
``f + eta`` with ``eta -> 0+`` (a zero vertex counts as outside) and are counted
in the evidence. Failed checks refine the incident leaves within the level and
budget limits; whatever remains is reported as unresolved boxes, never dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array
from jax.typing import ArrayLike
from scipy.sparse.csgraph import connected_components

from ..._bvh import bvh_overlap_pairs_host, prepare_bvh
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._meshcore import charge_native_geometry_queries
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.spatial._level_octree import refined_octree_leaves
from ...discretization.spatial._morton import (
    morton_decode_integer_host,
    morton_encode_integer_host,
    MortonAddressPlan,
)
from ...ein import contract
from ...typing import Dim, HostFloat64, HostInt64, parse, Scope
from .._certificate import SignReliability, ZeroSetAccuracy
from .._certified_implicit import (
    _established_implicit_cover,
    CertifiedImplicitCover,
    CertifiedImplicitTopology,
    establish_implicit_cover,
    implicit_state_id,
)
from .._contracts import CompiledGeometry, GeometryKind
from .._mesh_certificates import SourceBoundaryDistance, SourceBoundarySamples
from ..simplicial import TriangleMesh
from ._discovery import (
    _CORNER_OFFSETS,
    _EDGE_AXIS,
    _EDGE_LOWER,
    _INCIDENT_CELL_OFFSETS,
    _isolate_roots,
    _qef_vertices,
)
from ._enclosure import (
    _AbstractFieldBounds,
    _BoxBounds,
    _field_bounds,
    DISCOVERY_DIRECTIONS,
    ENCLOSURE_ROUNDING_MODEL,
)
from ._policy import (
    AdaptiveImplicitBoxIssue,
    AdaptiveImplicitSurfacePolicy,
    AdaptiveImplicitSurfaceStatus,
    ImplicitDiscoveryAccuracy,
    ImplicitDiscoveryEnclosure,
)
from ._projection import _field_and_gradient
from ._realization import _triangles_intersect


_DEFAULT_ADAPTIVE_POLICY = AdaptiveImplicitSurfacePolicy()
_DIMENSION = 3
_UNITS = np.eye(_DIMENSION, dtype=np.int64)
_PLANE_AXES = np.asarray(((1, 2), (0, 2), (0, 1)), dtype=np.int64)
# In-plane `DISCOVERY_DIRECTIONS` of faces with normal 0, 1, 2: both in-plane
# axes and both in-plane diagonals.
_PLANE_DIRECTIONS = np.asarray(((1, 2, 7, 8), (0, 2, 5, 6), (0, 1, 3, 4)), dtype=np.int64)
# Upper bound on new box and point evaluations one new leaf can introduce: its
# box and center, 8 vertices, 24 minimal edges, 24 minimal faces and 24 face
# centers. Refinement is refused before evaluation when it could exceed budget.
_EVALUATIONS_PER_NEW_LEAF = 82
_SIZE_BITS = 32
_PAIR_CHUNK = 8192


class _UnresolvedDim(Dim):
    """Unresolved octree leaves."""


class _LeafDim(Dim, minimum=1):
    """Final octree leaves."""


class _QueryDim(Dim):
    """Queried points or boxes."""


class _FeatureEdgeDim(Dim):
    """Mesh edges on sharp features."""


class ImplicitVolumeClass(IntEnum):
    """Inside/outside classification of a point or box of the implicit volume."""

    OUTSIDE = 0
    INSIDE = 1
    UNKNOWN = 2


# ---------------------------------------------------------------------------
# Integer entity keys.


def _vertex_keys(coordinates: np.ndarray, modulus: int) -> np.ndarray:
    return coordinates[..., 0] + modulus * (
        coordinates[..., 1] + modulus * coordinates[..., 2]
    )


def _entity_keys(
    coordinates: np.ndarray, axis: np.ndarray, exponent: np.ndarray, modulus: int
) -> np.ndarray:
    return (
        _vertex_keys(coordinates, modulus) * _DIMENSION + axis
    ) * _SIZE_BITS + exponent


def _lookup(sorted_keys: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions of ``keys`` in ``sorted_keys`` and whether each is present."""
    if not sorted_keys.size:
        return np.zeros(keys.shape, dtype=np.int64), np.zeros(keys.shape, dtype=np.bool_)
    position = np.minimum(np.searchsorted(sorted_keys, keys), sorted_keys.size - 1)
    return position, sorted_keys[position] == keys


def _exponent(sizes: np.ndarray) -> np.ndarray:
    return np.round(np.log2(sizes)).astype(np.int64)


# ---------------------------------------------------------------------------
# Working state.


@dataclass(frozen=True, slots=True)
class _Leaves:
    prefixes: np.ndarray
    levels: np.ndarray
    corners: np.ndarray
    sizes: np.ndarray
    code_starts: np.ndarray


class _NormalFiberMacrochartUnresolved(ValueError):
    """A candidate union lacks complete regular normal-fiber premises."""


def _normal_half_edge_keys(
    lower: np.ndarray, size: int, normal: int, modulus: int, /
) -> np.ndarray:
    plane = _PLANE_AXES[normal]
    offsets = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int64)
    starts = np.repeat(lower[None], 4, axis=0)
    starts[:, plane] += offsets * size
    halves = np.stack((starts, starts + _UNITS[normal] * size), axis=1)
    return _entity_keys(
        halves,
        np.full((4, 2), normal, dtype=np.int64),
        np.full((4, 2), size.bit_length() - 1, dtype=np.int64),
        modulus,
    )


@final
class ImplicitNormalFiberMacrochart(StrictModule, NonTrainableState):
    """Complete regular graph across two incident discovery leaves.

    Opposite signs on the two *outer* normal faces and a uniform derivative
    sign establish one source root on every normal fiber. The internal face
    may contain a tangent point or a closed critical contour; it is retained
    inside this chart, never declared empty from samples.
    """

    __strict_contract__ = True
    bounds: _AbstractFieldBounds
    domain: HostFloat64[Literal[2], Literal[3]]
    integer_lower: HostInt64[Literal[3]]
    macro_box: HostFloat64[Literal[2], Literal[3]]
    normal_edge_keys: HostInt64[Literal[4], Literal[2]]
    root_edge_keys: HostInt64[Literal[4]]
    incident_leaf_codes: HostInt64[Literal[2]]
    outer_value_lower: HostFloat64[Literal[2]]
    outer_value_upper: HostFloat64[Literal[2]]
    directional_lower: HostFloat64[Literal[13]]
    directional_upper: HostFloat64[Literal[13]]
    source_id: str = eqx.field(static=True)
    maximum_level: int = eqx.field(static=True)
    normal_axis: int = eqx.field(static=True)
    integer_size: int = eqx.field(static=True)
    face_key: int = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)

    def __init__(
        self,
        bounds: _AbstractFieldBounds,
        domain: ArrayLike,
        maximum_level: int,
        source_id: str,
        normal_axis: int,
        integer_lower: ArrayLike,
        integer_size: int,
        face_key: int,
        /,
    ) -> None:
        box = parse(
            np.array(domain, dtype=np.float64, copy=True),
            HostFloat64[Literal[2], Literal[3]],
            "domain",
        )
        lower = parse(
            np.array(integer_lower, dtype=np.int64, copy=True),
            HostInt64[Literal[3]],
            "integer_lower",
        )
        if (
            not 0 <= normal_axis < 3
            or not 1 <= maximum_level <= 16
            or integer_size < 1
            or integer_size & (integer_size - 1)
            or not np.all(np.isfinite(box))
            or np.any(box[0] >= box[1])
        ):
            raise ValueError(
                "A normal-fiber macrochart requires valid axis, dyadic size and domain."
            )
        resolution = 1 << maximum_level
        extent = integer_size * (np.ones(3, dtype=np.int64) + _UNITS[normal_axis])
        if np.any(lower < 0) or np.any(lower + extent > resolution):
            raise ValueError(
                "A normal-fiber macrochart must lie inside the declared carrier."
            )
        if not bounds.continuity_rigorous or not bounds.gradient_rigorous:
            raise _NormalFiberMacrochartUnresolved(
                "The source has no established continuous regular graph premises."
            )
        spacing = (box[1] - box[0]) / resolution

        def coordinates(integer: np.ndarray) -> np.ndarray:
            return np.where(integer == resolution, box[1], box[0] + integer * spacing)

        macro = np.stack((coordinates(lower), coordinates(lower + extent)))
        face_integer = lower + _UNITS[normal_axis] * integer_size
        expected = _entity_keys(
            face_integer[None],
            np.asarray((normal_axis,), dtype=np.int64),
            np.asarray((integer_size.bit_length() - 1,), dtype=np.int64),
            resolution + 1,
        )[0]
        if face_key != expected:
            raise ValueError(
                "A macrochart must retain its exact owning internal-face key."
            )
        outer_lower = np.repeat(macro[:1], 2, axis=0)
        outer_upper = np.repeat(macro[1:], 2, axis=0)
        outer_upper[0, normal_axis] = macro[0, normal_axis]
        outer_lower[1, normal_axis] = macro[1, normal_axis]
        result = bounds.boxes(
            np.concatenate((macro[:1], outer_lower)),
            np.concatenate((macro[1:], outer_upper)),
        )
        plane = _PLANE_AXES[normal_axis]
        offsets = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int64)
        middle_integer = np.repeat(face_integer[None], 4, axis=0)
        middle_integer[:, plane] += offsets * integer_size
        _, middle_upper = bounds.point_values(coordinates(middle_integer))
        derivative_lower = result.directional_lower[0, normal_axis]
        derivative_upper = result.directional_upper[0, normal_axis]
        increasing = derivative_lower > 0.0
        decreasing = derivative_upper < 0.0
        bracketed = (
            increasing and result.value_upper[1] < 0.0 and result.value_lower[2] > 0.0
        ) or (decreasing and result.value_lower[1] > 0.0 and result.value_upper[2] < 0.0)
        if (
            not bracketed
            or not np.all(np.isfinite(result.directional_lower[0]))
            or not np.all(np.isfinite(result.directional_upper[0]))
        ):
            raise _NormalFiberMacrochartUnresolved(
                "The complete macrochart lacks a uniform outer-face source bracket."
            )
        halves = _normal_half_edge_keys(lower, integer_size, normal_axis, resolution + 1)
        middle_negative = middle_upper < 0.0
        first_half = middle_negative != increasing
        roots = halves[np.arange(4), np.where(first_half, 0, 1)]
        codes = morton_encode_integer_host(
            np.stack((lower, face_integer)),
            maximum_level,
        ).astype(np.int64)
        binding = canonical_fingerprint(
            {
                "kind": "implicit-normal-fiber-source",
                "kernel": str(jax.tree_util.tree_structure(bounds.kernel)),
                "state": array_tree_fingerprint(bounds.state),
                "source": source_id,
            }
        )
        self.bounds, self.domain, self.integer_lower = bounds, box, lower
        self.maximum_level, self.source_id = maximum_level, source_id
        self.normal_axis, self.integer_size, self.face_key = (
            normal_axis,
            integer_size,
            face_key,
        )
        self.macro_box, self.normal_edge_keys, self.root_edge_keys = macro, halves, roots
        self.incident_leaf_codes = codes
        self.outer_value_lower, self.outer_value_upper = (
            result.value_lower[1:],
            result.value_upper[1:],
        )
        self.directional_lower, self.directional_upper = (
            result.directional_lower[0],
            result.directional_upper[0],
        )
        self.source_binding_id = binding
        self.chart_id = canonical_fingerprint(
            {
                "kind": "implicit-normal-fiber-macrochart",
                "source": binding,
                "domain": array_tree_fingerprint(box),
                "level": maximum_level,
                "face": face_key,
                "edges": array_tree_fingerprint(halves),
                "roots": array_tree_fingerprint(roots),
                "values": array_tree_fingerprint(
                    (self.outer_value_lower, self.outer_value_upper)
                ),
                "derivative": array_tree_fingerprint(
                    (self.directional_lower, self.directional_upper)
                ),
            }
        )
        for value in (
            box,
            lower,
            macro,
            halves,
            roots,
            codes,
            self.outer_value_lower,
            self.outer_value_upper,
            self.directional_lower,
            self.directional_upper,
        ):
            value.setflags(write=False)


@dataclass(slots=True)
class _Work:
    """Mutable host working state of one discovery; never published."""

    bounds: _AbstractFieldBounds
    policy: AdaptiveImplicitSurfacePolicy
    address: MortonAddressPlan
    lower: np.ndarray
    upper: np.ndarray
    spacing: np.ndarray
    depth: int
    modulus: int
    box_evaluations: int
    point_evaluations: int
    isolation_exhausted: bool
    leaf_bounds: dict[int, tuple[np.ndarray, _BoxBounds]]
    vertex_keys: np.ndarray
    vertex_signs: np.ndarray
    vertex_zero: np.ndarray
    checked_keys: np.ndarray
    checked_ok: np.ndarray

    source_id: str
    macrocharts: dict[int, ImplicitNormalFiberMacrochart]
    maximum_root_solves: int

    def coordinates(self, integer: np.ndarray) -> np.ndarray:
        # The last lattice plane is the domain corner itself, so leaves tile the
        # domain exactly and shared lattice points have one float value.
        return np.where(
            integer == self.modulus - 1,
            self.upper,
            self.lower + integer.astype(np.float64) * self.spacing,
        )

    def evaluate_boxes(self, lower: np.ndarray, upper: np.ndarray) -> _BoxBounds:
        # Each box costs its enclosure and its center (mean-value form).
        remaining = (
            self.policy.maximum_evaluations
            - self.box_evaluations
            - self.point_evaluations
        )
        count = min(lower.shape[0], max(0, remaining // 2))
        self.box_evaluations += 2 * count
        if count == lower.shape[0]:
            return self.bounds.boxes(lower, upper)
        self.isolation_exhausted = True
        missing = lower.shape[0] - count
        directions = DISCOVERY_DIRECTIONS.shape[0]
        unknown = _BoxBounds(
            np.full(missing, -np.inf, dtype=np.float64),
            np.full(missing, np.inf, dtype=np.float64),
            np.full((missing, directions), -np.inf, dtype=np.float64),
            np.full((missing, directions), np.inf, dtype=np.float64),
        )
        if not count:
            return unknown
        fresh = self.bounds.boxes(lower[:count], upper[:count])
        return _merged(fresh, unknown, np.arange(lower.shape[0], dtype=np.int64))

    def point_signs(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signs with the zero-as-outside perturbation and exact-zero flags."""
        remaining = (
            self.policy.maximum_evaluations
            - self.box_evaluations
            - self.point_evaluations
        )
        count = min(points.shape[0], max(0, remaining))
        self.point_evaluations += count
        lower = np.full(points.shape[0], -np.inf, dtype=np.float64)
        upper = np.full(points.shape[0], np.inf, dtype=np.float64)
        if count:
            lower[:count], upper[:count] = self.bounds.point_values(points[:count])
        if count < points.shape[0]:
            self.isolation_exhausted = True
        negative = upper < 0.0
        zero = ~negative & ~(lower > 0.0)
        return np.where(negative, -1, 1).astype(np.int8), zero


def _leaves(work: _Work, refined: list[np.ndarray]) -> _Leaves:
    prefixes, levels, corners = refined_octree_leaves(
        work.address, refined, balanced=True
    )
    levels_ = levels.astype(np.int64)
    shifts = (_DIMENSION * (work.depth - levels_)).astype(np.uint64)
    return _Leaves(
        prefixes=prefixes,
        levels=levels_,
        corners=corners.astype(np.int64),
        sizes=np.left_shift(np.int64(1), work.depth - levels_),
        code_starts=prefixes << shifts,
    )


def _locate(work: _Work, leaves: _Leaves, cells: np.ndarray) -> np.ndarray:
    """Leaf containing each finest cell, or ``-1`` outside the domain."""
    inside = np.all((cells >= 0) & (cells < work.modulus - 1), axis=-1)
    safe = np.where(inside[..., None], cells, 0)
    codes = morton_encode_integer_host(safe, work.depth)
    slot = np.searchsorted(leaves.code_starts, codes, side="right") - 1
    return np.where(inside, slot, -1).astype(np.int64)


def _empty_bounds() -> _BoxBounds:
    directions = DISCOVERY_DIRECTIONS.shape[0]
    return _BoxBounds(
        np.zeros((0,)),
        np.zeros((0,)),
        np.zeros((0, directions)),
        np.zeros((0, directions)),
    )


def _merged(first: _BoxBounds, second: _BoxBounds, order: np.ndarray) -> _BoxBounds:
    return _BoxBounds(
        np.concatenate((first.value_lower, second.value_lower))[order],
        np.concatenate((first.value_upper, second.value_upper))[order],
        np.concatenate((first.directional_lower, second.directional_lower))[order],
        np.concatenate((first.directional_upper, second.directional_upper))[order],
    )


def _rows(bounds: _BoxBounds, rows: np.ndarray) -> _BoxBounds:
    return _BoxBounds(
        bounds.value_lower[rows],
        bounds.value_upper[rows],
        bounds.directional_lower[rows],
        bounds.directional_upper[rows],
    )


def _leaf_bounds(work: _Work, leaves: _Leaves) -> _BoxBounds:
    """Bounds of every leaf, evaluating only leaves new to this discovery."""
    order = np.lexsort((leaves.prefixes, leaves.levels))
    parts: list[_BoxBounds] = []
    for level in np.unique(leaves.levels).tolist():
        prefixes = np.sort(leaves.prefixes[leaves.levels == level])
        known, bounds = work.leaf_bounds.get(
            level, (np.zeros((0,), dtype=np.uint64), _empty_bounds())
        )
        _, present = _lookup(known, prefixes)
        missing = prefixes[~present]
        if missing.size:
            shift = np.uint64(_DIMENSION * (work.depth - level))
            corners = morton_decode_integer_host(missing << shift, _DIMENSION, work.depth)
            size = 1 << (work.depth - level)
            fresh = work.evaluate_boxes(
                work.coordinates(corners), work.coordinates(corners + size)
            )
            merged_keys = np.concatenate((known, missing))
            merged_order = np.argsort(merged_keys, kind="stable")
            known = merged_keys[merged_order]
            bounds = _merged(bounds, fresh, merged_order)
            work.leaf_bounds[level] = (known, bounds)
        position, _ = _lookup(known, prefixes)
        parts.append(_rows(bounds, position))
    stacked = _BoxBounds(
        np.concatenate([part.value_lower for part in parts]),
        np.concatenate([part.value_upper for part in parts]),
        np.concatenate([part.directional_lower for part in parts]),
        np.concatenate([part.directional_upper for part in parts]),
    )
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return _rows(stacked, inverse)


# ---------------------------------------------------------------------------
# One analysis of the current tree.


@dataclass(frozen=True, slots=True)
class _Edges:
    keys: np.ndarray
    start: np.ndarray
    axis: np.ndarray
    length: np.ndarray
    start_vertex: np.ndarray
    end_vertex: np.ndarray
    crossing: np.ndarray
    incident: np.ndarray
    certified: np.ndarray


@dataclass(frozen=True, slots=True)
class _Faces:
    keys: np.ndarray
    incident: np.ndarray
    certified: np.ndarray
    boundary_excluded: np.ndarray
    arc_faces: np.ndarray
    arc_edges: np.ndarray
    macrocharts: tuple[ImplicitNormalFiberMacrochart, ...]


@dataclass(frozen=True, slots=True)
class _Analysis:
    leaves: _Leaves
    bounds: _BoxBounds
    excluded: np.ndarray
    monotone: np.ndarray
    edges: _Edges
    faces: _Faces
    node_leaf: np.ndarray
    node_edge: np.ndarray
    node_cycle: np.ndarray
    cycle_leaf: np.ndarray
    cycles: np.ndarray
    issues: np.ndarray
    refine: np.ndarray
    flatness: np.ndarray


def _vertex_table(
    work: _Work, leaves: _Leaves, bounds: _BoxBounds, excluded: np.ndarray
) -> None:
    """Extend the vertex sign cache with every leaf corner of the tree."""
    corners = (
        leaves.corners[:, None, :] + _CORNER_OFFSETS[None] * leaves.sizes[:, None, None]
    )
    keys = _vertex_keys(corners, work.modulus)
    unique = np.unique(keys)
    _, known = _lookup(work.vertex_keys, unique)
    fresh = unique[~known]
    if not fresh.size:
        return
    signs = np.zeros((fresh.size,), dtype=np.int8)
    zero = np.zeros((fresh.size,), dtype=np.bool_)
    # A vertex of a leaf whose value enclosure excludes zero has that sign.
    leaf_sign = np.where(bounds.value_upper < 0.0, -1, 1).astype(np.int8)
    excluded_keys = keys[excluded].reshape((-1,))
    excluded_signs = np.repeat(leaf_sign[excluded], 8)
    position, present = _lookup(fresh, excluded_keys)
    signs[position[present]] = excluded_signs[present]
    pending = np.flatnonzero(signs == 0)
    if pending.size:
        modulus = work.modulus
        key = fresh[pending]
        integer = np.stack(
            (key % modulus, (key // modulus) % modulus, key // (modulus * modulus)),
            axis=-1,
        )
        pending_signs, pending_zero = work.point_signs(work.coordinates(integer))
        signs[pending] = pending_signs
        zero[pending] = pending_zero
    merged = np.concatenate((work.vertex_keys, fresh))
    order = np.argsort(merged, kind="stable")
    work.vertex_keys = merged[order]
    work.vertex_signs = np.concatenate((work.vertex_signs, signs))[order]
    work.vertex_zero = np.concatenate((work.vertex_zero, zero))[order]


def _vertex_sign(work: _Work, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position, present = _lookup(work.vertex_keys, _vertex_keys(coordinates, work.modulus))
    return np.where(present, work.vertex_signs[position], 0), present


def _checked(
    work: _Work,
    keys: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """Cached degenerate-box checks: value excludes zero or strictly monotone
    along one of the row's admissible ``DISCOVERY_DIRECTIONS`` indices."""
    _, known = _lookup(work.checked_keys, keys)
    fresh = np.flatnonzero(~known)
    if fresh.size:
        bounds = work.evaluate_boxes(lower[fresh], upper[fresh])
        ok = (bounds.value_lower > 0.0) | (bounds.value_upper < 0.0)
        monotone = (bounds.directional_lower > 0.0) | (bounds.directional_upper < 0.0)
        ok = ok | np.any(np.take_along_axis(monotone, directions[fresh], axis=1), axis=1)
        merged = np.concatenate((work.checked_keys, keys[fresh]))
        order = np.argsort(merged, kind="stable")
        work.checked_keys = merged[order]
        work.checked_ok = np.concatenate((work.checked_ok, ok))[order]
    position, _ = _lookup(work.checked_keys, keys)
    return work.checked_ok[position]


def _minimal_edges(
    work: _Work, leaves: _Leaves, excluded: np.ndarray, monotone: np.ndarray
) -> _Edges:
    size = leaves.sizes[:, None]
    start = leaves.corners[:, None, :] + _EDGE_LOWER[None] * size[..., None]
    axis = np.broadcast_to(_EDGE_AXIS[None], start.shape[:2])
    length = np.broadcast_to(size, start.shape[:2])
    half = length // 2
    middle = start + _UNITS[axis] * half[..., None]
    _, middle_vertex = _vertex_sign(work, middle)
    split = (length >= 2) & middle_vertex
    whole = ~split
    candidates_start = np.concatenate((start[whole], start[split], middle[split]), axis=0)
    candidates_axis = np.concatenate((axis[whole], axis[split], axis[split]))
    candidates_length = np.concatenate((length[whole], half[split], half[split]))
    keys = _entity_keys(
        candidates_start, candidates_axis, _exponent(candidates_length), work.modulus
    )
    keys, first = np.unique(keys, return_index=True)
    start_ = candidates_start[first]
    axis_ = candidates_axis[first]
    length_ = candidates_length[first]
    end = start_ + _UNITS[axis_] * length_[:, None]
    start_sign, start_present = _vertex_sign(work, start_)
    end_sign, end_present = _vertex_sign(work, end)
    if not (np.all(start_present) and np.all(end_present)):
        raise RuntimeError("A minimal octree edge lost an endpoint vertex.")
    start_vertex, _ = _lookup(work.vertex_keys, _vertex_keys(start_, work.modulus))
    end_vertex, _ = _lookup(work.vertex_keys, _vertex_keys(end, work.modulus))
    probes = (
        start_[:, None, :]
        + _INCIDENT_CELL_OFFSETS[axis_]
        + _UNITS[axis_][:, None, :] * (length_ // 2)[:, None, None]
    )
    incident = _locate(work, leaves, probes)
    valid = incident >= 0
    safe = np.where(valid, incident, 0)
    by_leaf = np.any(valid & (excluded[safe] | monotone[safe, axis_[:, None]]), axis=1)
    certified = by_leaf.copy()
    pending = np.flatnonzero(~by_leaf)
    if pending.size:
        lower = work.coordinates(start_[pending])
        upper = work.coordinates(end[pending])
        # Edge and face keys share one cache; the low bit separates them.
        certified[pending] = _checked(
            work, 2 * keys[pending], lower, upper, axis_[pending, None]
        )
        isolate = pending[~certified[pending]]
        if isolate.size:
            certified[isolate] = _edge_root_isolation(
                work,
                work.coordinates(start_[isolate]),
                work.coordinates(end[isolate]),
                axis_[isolate],
                start_sign[isolate],
                end_sign[isolate],
            )
            position, _ = _lookup(work.checked_keys, 2 * keys[isolate])
            work.checked_ok[position] = certified[isolate]
    return _Edges(
        keys=keys,
        start=start_,
        axis=axis_,
        length=length_,
        start_vertex=start_vertex,
        end_vertex=end_vertex,
        crossing=start_sign != end_sign,
        incident=incident,
        certified=certified,
    )


def _edge_root_isolation(
    work: _Work,
    start: np.ndarray,
    end: np.ndarray,
    axis: np.ndarray,
    start_sign: np.ndarray,
    end_sign: np.ndarray,
) -> np.ndarray:
    """Certify at most one root per edge by interval bisection along the edge.

    Segments are split until each excludes zero or is strictly monotone along
    the edge; the root count is then the number of monotone segments whose
    endpoint signs differ. An edge is certified when that count is at most one
    (its parity always matches the endpoint signs); breakpoints whose point
    enclosure contains zero, unresolved segments at ``edge_isolation_depth`` and
    two or more roots leave the edge uncertified so its leaves refine.
    """
    policy = work.policy
    count = start.shape[0]
    edge = np.arange(count)
    parameter_lower = np.zeros((count,), dtype=np.float64)
    parameter_upper = np.ones((count,), dtype=np.float64)
    failed = np.zeros((count,), dtype=np.bool_)
    resolved_edge: list[np.ndarray] = []
    resolved_lower: list[np.ndarray] = []
    resolved_monotone: list[np.ndarray] = []
    direction = end - start
    for depth in range(policy.edge_isolation_depth + 1):
        if not edge.size:
            break
        used = work.box_evaluations + work.point_evaluations
        if used + 3 * edge.size > policy.maximum_evaluations:
            work.isolation_exhausted = True
            failed[edge] = True
            break
        lower_points = start[edge] + parameter_lower[:, None] * direction[edge]
        upper_points = start[edge] + parameter_upper[:, None] * direction[edge]
        bounds = work.evaluate_boxes(
            np.minimum(lower_points, upper_points), np.maximum(lower_points, upper_points)
        )
        excluded = (bounds.value_lower > 0.0) | (bounds.value_upper < 0.0)
        monotone = (bounds.directional_lower[np.arange(edge.size), axis[edge]] > 0.0) | (
            bounds.directional_upper[np.arange(edge.size), axis[edge]] < 0.0
        )
        done = excluded | monotone
        resolved_edge.append(edge[done])
        resolved_lower.append(parameter_lower[done])
        resolved_monotone.append(monotone[done] & ~excluded[done])
        open_ = ~done
        if depth == policy.edge_isolation_depth:
            failed[edge[open_]] = True
            break
        middle = 0.5 * (parameter_lower[open_] + parameter_upper[open_])
        edge = np.repeat(edge[open_], 2)
        parameter_lower = np.stack((parameter_lower[open_], middle), axis=1).reshape(
            (-1,)
        )
        parameter_upper = np.stack((middle, parameter_upper[open_]), axis=1).reshape(
            (-1,)
        )
    segment_edge = np.concatenate(resolved_edge)
    segment_lower = np.concatenate(resolved_lower)
    segment_monotone = np.concatenate(resolved_monotone)
    interior = segment_lower > 0.0
    signs_lower = np.where(segment_lower == 0.0, start_sign[segment_edge], 0).astype(
        np.int8
    )
    ambiguous = np.zeros((segment_edge.size,), dtype=np.bool_)
    if np.any(interior):
        points = (
            start[segment_edge[interior]]
            + segment_lower[interior, None] * direction[segment_edge[interior]]
        )
        point_signs, zero = work.point_signs(points)
        signs_lower[interior] = point_signs
        ambiguous[interior] = zero
    failed[segment_edge[ambiguous]] = True
    # The upper sign of a segment is the lower sign of the next one on its edge.
    order = np.lexsort((segment_lower, segment_edge))
    ordered_edge = segment_edge[order]
    ordered_sign = signs_lower[order]
    last = np.append(ordered_edge[1:] != ordered_edge[:-1], True)
    upper_sign = np.where(last, end_sign[ordered_edge], np.roll(ordered_sign, -1))
    roots = segment_monotone[order] & (ordered_sign != upper_sign)
    root_count = np.bincount(ordered_edge[roots], minlength=count)
    return ~failed & (root_count <= 1)


def _face_cycles(
    work: _Work, lower: np.ndarray, normal: np.ndarray, size: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cyclic boundary vertices (8 slots, midpoints optional) of minimal faces."""
    plane = _PLANE_AXES[normal]
    first = _UNITS[plane[:, 0]] * size[:, None]
    second = _UNITS[plane[:, 1]] * size[:, None]
    slots = np.stack(
        (
            lower,
            lower + first // 2,
            lower + first,
            lower + first + second // 2,
            lower + first + second,
            lower + first // 2 + second,
            lower + second,
            lower + second // 2,
        ),
        axis=1,
    )
    _, present = _vertex_sign(work, slots)
    present[:, 1::2] &= (size >= 2)[:, None]
    present[:, 0::2] = True
    return slots, present


@dataclass(frozen=True, slots=True)
class _Tiles:
    """Minimal faces tiling the six faces of every leaf, laid out (leaf, 6, 4)."""

    lower: np.ndarray
    normal: np.ndarray
    size: np.ndarray
    valid: np.ndarray
    keys: np.ndarray


def _leaf_face_tiles(work: _Work, leaves: _Leaves) -> _Tiles:
    """One minimal face per leaf face, or its four quarters when a finer
    neighbor subdivides it (its center is then an octree vertex)."""
    size = leaves.sizes
    normal = np.repeat(np.arange(_DIMENSION), 2)
    side = np.tile(np.arange(2), _DIMENSION)
    face_lower = leaves.corners[:, None, :] + _UNITS[normal][None] * (
        side[None, :, None] * size[:, None, None]
    )
    plane = _PLANE_AXES[normal]
    offsets = _UNITS[plane[:, 0]] + _UNITS[plane[:, 1]]
    center = face_lower + offsets[None] * (size[:, None, None] // 2)
    _, center_vertex = _vertex_sign(work, center)
    subdivided = (size[:, None] >= 2) & center_vertex
    quarter = (size // 2)[:, None, None, None]
    quadrant = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), dtype=np.int64)
    quarter_lower = (
        face_lower[:, :, None, :]
        + (
            _UNITS[plane[:, 0]][None, :, None, :] * quadrant[None, None, :, 0, None]
            + _UNITS[plane[:, 1]][None, :, None, :] * quadrant[None, None, :, 1, None]
        )
        * quarter
    )
    face_size = np.broadcast_to(size[:, None], subdivided.shape)
    lower = np.where(
        subdivided[..., None, None], quarter_lower, face_lower[:, :, None, :]
    )
    valid = np.where(subdivided[..., None], True, np.arange(4)[None, None, :] == 0)
    tile_size = np.broadcast_to(
        np.where(subdivided, face_size // 2, face_size)[..., None], valid.shape
    )
    tile_normal = np.broadcast_to(normal[None, :, None], valid.shape)
    return _Tiles(
        lower=lower,
        normal=tile_normal,
        size=tile_size,
        valid=valid,
        keys=_entity_keys(lower, tile_normal, _exponent(tile_size), work.modulus),
    )


def _normal_macrocharts(
    work: _Work,
    leaves: _Leaves,
    keys: np.ndarray,
    normal: np.ndarray,
    size: np.ndarray,
    incident: np.ndarray,
    certified: np.ndarray,
    crossing_count: np.ndarray,
    edges: _Edges,
    /,
) -> tuple[ImplicitNormalFiberMacrochart, ...]:
    """Join a pending internal face only under a complete bounded graph proof."""
    if not work.bounds.continuity_rigorous:
        return ()
    candidates = np.flatnonzero(
        ~certified & (crossing_count == 0) & np.all(incident >= 0, axis=1)
    )
    charts: list[ImplicitNormalFiberMacrochart] = []
    domain = np.stack((work.lower, work.upper))
    for face in candidates:
        first, second = incident[face]
        axis, width = int(normal[face]), int(size[face])
        lower = leaves.corners[first]
        if (
            leaves.sizes[first] != width
            or leaves.sizes[second] != width
            or not np.array_equal(leaves.corners[second], lower + _UNITS[axis] * width)
        ):
            continue
        boundary = np.any(np.isin(incident, (first, second)), axis=1)
        boundary[face] = False
        if not np.all(certified[boundary]):
            continue
        half_keys = _normal_half_edge_keys(lower, width, axis, work.modulus)
        slots, found = _lookup(edges.keys, half_keys)
        if not np.all(found) or not np.all(edges.certified[slots]):
            continue
        if not np.all(np.sum(edges.crossing[slots], axis=1) == 1):
            continue
        key = int(keys[face])
        chart = work.macrocharts.get(key)
        if chart is None:
            if (
                work.box_evaluations + work.point_evaluations + 10
                > work.policy.maximum_evaluations
            ):
                work.isolation_exhausted = True
                continue
            try:
                chart = ImplicitNormalFiberMacrochart(
                    work.bounds,
                    domain,
                    work.depth,
                    work.source_id,
                    axis,
                    lower,
                    width,
                    key,
                )
            except _NormalFiberMacrochartUnresolved:
                work.box_evaluations += 6
                work.point_evaluations += 4
                continue
            work.box_evaluations += 6
            work.point_evaluations += 4
            work.macrocharts[key] = chart
        actual = np.sort(edges.keys[slots][edges.crossing[slots]])
        if not np.array_equal(actual, np.sort(chart.root_edge_keys)):
            continue
        certified[face] = True
        charts.append(chart)
    return tuple(charts)


def _minimal_faces(
    work: _Work,
    leaves: _Leaves,
    excluded: np.ndarray,
    monotone: np.ndarray,
    edges: _Edges,
) -> tuple[_Faces, np.ndarray, np.ndarray]:
    tiles = _leaf_face_tiles(work, leaves)
    keys, first = np.unique(tiles.keys[tiles.valid], return_index=True)
    lower_ = tiles.lower[tiles.valid][first]
    normal_ = tiles.normal[tiles.valid][first]
    size_ = tiles.size[tiles.valid][first]
    slots, present = _face_cycles(work, lower_, normal_, size_)
    signs, _ = _vertex_sign(work, slots)
    # Compact present slots into cyclic order.
    order = np.argsort(~present, axis=1, kind="stable")
    ring = np.take_along_axis(slots, order[..., None], axis=1)
    ring_signs = np.take_along_axis(signs, order, axis=1)
    ring_size = np.sum(present, axis=1)
    index = np.arange(8)[None, :]
    following = np.where(index + 1 < ring_size[:, None], index + 1, 0)
    ring_next = np.take_along_axis(ring, following[..., None], axis=1)
    next_signs = np.take_along_axis(ring_signs, following, axis=1)
    segment_valid = index < ring_size[:, None]
    crossing = segment_valid & (ring_signs != next_signs)
    crossing_count = np.sum(crossing, axis=1)
    segment_start = np.minimum(ring, ring_next)
    segment_axis = np.argmax(ring != ring_next, axis=-1)
    segment_length = np.max(np.abs(ring_next - ring), axis=-1)
    segment_keys = _entity_keys(
        segment_start,
        segment_axis,
        _exponent(np.maximum(segment_length, 1)),
        work.modulus,
    )
    segment_edge, found = _lookup(edges.keys, segment_keys)
    if np.any(crossing & ~found):
        raise RuntimeError("A crossing face segment is not a minimal octree edge.")
    probe = (
        lower_
        + (_UNITS[_PLANE_AXES[normal_][:, 0]] + _UNITS[_PLANE_AXES[normal_][:, 1]])
        * (size_ // 2)[:, None]
    )
    incident = _locate(
        work,
        leaves,
        np.stack((probe - _UNITS[normal_], probe), axis=1),
    )
    valid = incident >= 0
    safe = np.where(valid, incident, 0)
    plane_ = _PLANE_AXES[normal_]
    plane_directions = _PLANE_DIRECTIONS[normal_]
    excluded_by_leaf = np.any(valid & excluded[safe], axis=1)
    in_plane = np.any(
        valid & np.any(monotone[safe[:, :, None], plane_directions[:, None, :]], axis=2),
        axis=1,
    )
    admissible = (crossing_count == 0) | (crossing_count == 2)
    certified = excluded_by_leaf | (admissible & in_plane)
    pending = np.flatnonzero(~certified & admissible)
    upper_ = lower_ + (_UNITS[plane_[:, 0]] + _UNITS[plane_[:, 1]]) * size_[:, None]
    if pending.size:
        certified[pending] = _checked(
            work,
            2 * keys[pending] + 1,
            work.coordinates(lower_[pending]),
            work.coordinates(upper_[pending]),
            plane_directions[pending],
        )
    boundary = ~np.all(valid, axis=1)
    boundary_excluded = excluded_by_leaf.copy()
    boundary_pending = np.flatnonzero(boundary & ~excluded_by_leaf)
    if boundary_pending.size:
        bounds = work.evaluate_boxes(
            work.coordinates(lower_[boundary_pending]),
            work.coordinates(upper_[boundary_pending]),
        )
        boundary_excluded[boundary_pending] = (bounds.value_lower > 0.0) | (
            bounds.value_upper < 0.0
        )
    macrocharts = _normal_macrocharts(
        work,
        leaves,
        keys,
        normal_,
        size_,
        incident,
        certified,
        crossing_count,
        edges,
    )
    arc_faces, arc_edges = _face_arcs(
        work,
        lower_,
        normal_,
        size_,
        crossing,
        crossing_count,
        ring_signs,
        ring_size,
        segment_edge,
    )
    faces = _Faces(
        keys=keys,
        incident=incident,
        certified=certified | (boundary & boundary_excluded),
        boundary_excluded=np.where(boundary, boundary_excluded, True),
        arc_faces=arc_faces,
        arc_edges=arc_edges,
        macrocharts=macrocharts,
    )
    tile_face, _ = _lookup(keys, tiles.keys)
    return faces, np.where(tiles.valid, tile_face, -1), tiles.valid


def _face_arcs(
    work: _Work,
    lower: np.ndarray,
    normal: np.ndarray,
    size: np.ndarray,
    crossing: np.ndarray,
    crossing_count: np.ndarray,
    ring_signs: np.ndarray,
    ring_size: np.ndarray,
    segment_edge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair the crossing segments of every face into non-crossing arcs.

    With two crossings the arc is unique. With more, the face-center sign
    decides: the region of the center sign is connected through the center, so
    each run of the opposite sign is cut off by one arc between its bounding
    crossings.
    """
    faces: list[np.ndarray] = []
    pairs: list[np.ndarray] = []
    two = np.flatnonzero(crossing_count == 2)
    if two.size:
        positions = np.argsort(~crossing[two], axis=1, kind="stable")[:, :2]
        faces.append(two)
        pairs.append(np.take_along_axis(segment_edge[two], positions, axis=1))
    many = np.flatnonzero(crossing_count >= 4)
    if many.size:
        plane = _PLANE_AXES[normal[many]]
        span = (_UNITS[plane[:, 0]] + _UNITS[plane[:, 1]]) * size[many][:, None]
        center_signs, _ = work.point_signs(
            work.lower + (lower[many] + 0.5 * span) * work.spacing
        )
        for row, face in enumerate(many.tolist()):
            positions = np.flatnonzero(crossing[face])
            for index, position in enumerate(positions.tolist()):
                following = positions[(index + 1) % positions.size]
                run_sign = ring_signs[face, (position + 1) % ring_size[face]]
                if run_sign != center_signs[row]:
                    faces.append(np.asarray([face]))
                    pairs.append(
                        np.asarray(
                            [
                                [
                                    segment_edge[face, position],
                                    segment_edge[face, following],
                                ]
                            ]
                        )
                    )
    if not faces:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(faces), np.concatenate(pairs, axis=0).astype(np.int64)


def _link_cycles(
    leaves: _Leaves,
    edges: _Edges,
    faces: _Faces,
    tile_face: np.ndarray,
    tile_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-set cycles on every leaf boundary from the arcs of its minimal faces.

    Nodes are (leaf, crossing minimal edge on the leaf boundary). Every such edge
    lies on exactly two minimal faces of the leaf boundary and each face pairs
    all of its crossings, so every node has degree two and components are
    cycles. Cycles are numbered by leaf, then by their smallest edge.
    """
    edge_count = edges.keys.shape[0]
    leaf_count = leaves.levels.shape[0]
    tile_leaf = np.broadcast_to(
        np.arange(leaf_count, dtype=np.int64)[:, None, None], tile_face.shape
    )[tile_valid]
    tile_face_ = tile_face[tile_valid]
    order = np.argsort(faces.arc_faces, kind="stable")
    arc_faces = faces.arc_faces[order]
    arc_edges = faces.arc_edges[order]
    face_count = faces.keys.shape[0]
    arc_start = np.searchsorted(arc_faces, np.arange(face_count))
    arc_stop = np.searchsorted(arc_faces, np.arange(face_count), side="right")
    counts = (arc_stop - arc_start)[tile_face_]
    total = int(np.sum(counts))
    if not total:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, empty, empty, np.zeros((leaf_count,), dtype=np.int64)
    owner = np.repeat(tile_leaf, counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    arcs = np.repeat(arc_start[tile_face_], counts) + np.arange(total) - offsets
    first = owner * edge_count + arc_edges[arcs, 0]
    second = owner * edge_count + arc_edges[arcs, 1]
    nodes, inverse = np.unique(np.concatenate((first, second)), return_inverse=True)
    degree = np.bincount(inverse, minlength=nodes.size)
    if np.any(degree != 2):
        raise RuntimeError("An octree leaf boundary produced an open zero-set link.")
    adjacency = sp.csr_matrix(
        (
            np.ones((total,), dtype=np.int8),
            (inverse[:total], inverse[total:]),
        ),
        shape=(nodes.size, nodes.size),
    )
    component_count, labels = connected_components(adjacency, directed=False)
    _, smallest = np.unique(labels, return_index=True)
    relabel = np.empty((component_count,), dtype=np.int64)
    relabel[np.argsort(smallest, kind="stable")] = np.arange(component_count)
    node_cycle = relabel[labels]
    node_leaf = nodes // edge_count
    node_edge = nodes % edge_count
    cycle_leaf = np.zeros((component_count,), dtype=np.int64)
    cycle_leaf[node_cycle] = node_leaf
    cycles = np.bincount(cycle_leaf, minlength=leaf_count).astype(np.int64)
    return nodes, node_edge, node_cycle, cycle_leaf, cycles


def _flatness(work: _Work, leaves: _Leaves, bounds: _BoxBounds) -> np.ndarray:
    """Leaf diagonal times the direction spread of the gradient enclosure."""
    with np.errstate(divide="ignore", invalid="ignore"):
        middle = 0.5 * (bounds.gradient_lower + bounds.gradient_upper)
        radius = 0.5 * (bounds.gradient_upper - bounds.gradient_lower)
        middle_norm = np.linalg.norm(middle, axis=1)
        radius_norm = np.linalg.norm(radius, axis=1)
        spread = np.where(
            np.isfinite(radius_norm) & (middle_norm > radius_norm),
            radius_norm / middle_norm,
            1.0,
        )
    diagonal = np.linalg.norm(leaves.sizes[:, None] * work.spacing[None], axis=1)
    return diagonal * spread


def _incident_mask(
    incident: np.ndarray, selected: np.ndarray, leaf_count: int
) -> np.ndarray:
    rows = incident[selected].reshape((-1,))
    mask = np.zeros((leaf_count,), dtype=np.bool_)
    mask[rows[rows >= 0]] = True
    return mask


def _analyze(work: _Work, leaves: _Leaves) -> _Analysis:
    policy = work.policy
    bounds = _leaf_bounds(work, leaves)
    excluded = (bounds.value_lower > 0.0) | (bounds.value_upper < 0.0)
    monotone = (bounds.directional_lower > 0.0) | (bounds.directional_upper < 0.0)
    _vertex_table(work, leaves, bounds, excluded)
    edges = _minimal_edges(work, leaves, excluded, monotone)
    faces, tile_face, tile_valid = _minimal_faces(work, leaves, excluded, monotone, edges)
    nodes, node_edge, node_cycle, cycle_leaf, cycles = _link_cycles(
        leaves, edges, faces, tile_face, tile_valid
    )
    admitted_charts: list[ImplicitNormalFiberMacrochart] = []
    for chart in faces.macrocharts:
        participants = np.flatnonzero(
            np.isin(leaves.code_starts, chart.incident_leaf_codes)
        )
        local_nodes = np.isin(nodes // max(edges.keys.shape[0], 1), participants)
        roots = np.unique(edges.keys[node_edge[local_nodes]])
        if np.sum(cycles[participants]) == 1 and np.array_equal(
            np.sort(roots), np.sort(chart.root_edge_keys)
        ):
            admitted_charts.append(chart)
        else:
            faces.certified[faces.keys == chart.face_key] = False
    faces = _Faces(
        faces.keys,
        faces.incident,
        faces.certified,
        faces.boundary_excluded,
        faces.arc_faces,
        faces.arc_edges,
        tuple(admitted_charts),
    )
    leaf_count = leaves.levels.shape[0]
    flatness = np.where(excluded, 0.0, _flatness(work, leaves, bounds))
    for chart in faces.macrocharts:
        middle = 0.5 * (chart.directional_lower[:3] + chart.directional_upper[:3])
        radius = 0.5 * (chart.directional_upper[:3] - chart.directional_lower[:3])
        norm, variation = np.linalg.norm(middle), np.linalg.norm(radius)
        spread = variation / norm if norm > variation else 1.0
        bound = np.linalg.norm(chart.macro_box[1] - chart.macro_box[0]) * spread
        participants = np.isin(leaves.code_starts, chart.incident_leaf_codes)
        flatness[participants] = np.maximum(flatness[participants], bound)
    issues = np.zeros((leaf_count,), dtype=np.int64)
    issues |= np.where(
        ~excluded & ~np.any(monotone, axis=1), int(AdaptiveImplicitBoxIssue.SINGULAR), 0
    )
    issues |= np.where(cycles >= 2, int(AdaptiveImplicitBoxIssue.SHEETS), 0)
    issues |= np.where(
        _incident_mask(edges.incident, ~edges.certified, leaf_count),
        int(AdaptiveImplicitBoxIssue.EDGE),
        0,
    )
    issues |= np.where(
        _incident_mask(faces.incident, ~faces.certified, leaf_count),
        int(AdaptiveImplicitBoxIssue.FACE),
        0,
    )
    issues |= np.where(
        _incident_mask(faces.incident, ~faces.boundary_excluded, leaf_count),
        int(AdaptiveImplicitBoxIssue.DOMAIN_BOUNDARY),
        0,
    )
    if policy.flatness_tolerance is not None:
        issues |= np.where(
            (cycles >= 1) & (flatness > policy.flatness_tolerance),
            int(AdaptiveImplicitBoxIssue.FLATNESS),
            0,
        )
    refine = (issues != 0) | (~excluded & (leaves.levels < policy.minimum_surface_level))
    refine &= leaves.levels < work.depth
    return _Analysis(
        leaves=leaves,
        bounds=bounds,
        excluded=excluded,
        monotone=monotone,
        edges=edges,
        faces=faces,
        node_leaf=nodes // max(edges.keys.shape[0], 1),
        node_edge=node_edge,
        node_cycle=node_cycle,
        cycle_leaf=cycle_leaf,
        cycles=cycles,
        issues=issues,
        refine=refine,
        flatness=flatness,
    )


def _refined_with(
    refined: list[np.ndarray], leaves: _Leaves, selected: np.ndarray
) -> list[np.ndarray]:
    updated = list(refined)
    for level in np.unique(leaves.levels[selected]).tolist():
        prefixes = leaves.prefixes[selected & (leaves.levels == level)]
        # Balancing may have refined ancestors the worklist never listed.
        for ancestor in range(level, -1, -1):
            updated[ancestor] = np.union1d(updated[ancestor], prefixes)
            prefixes = np.unique(prefixes >> np.uint64(_DIMENSION))
    return updated


def _refine(work: _Work) -> tuple[_Analysis, int, int]:
    """Refinement worklist; returns the last analysis, its status flags, and rounds."""
    policy = work.policy
    refined = [
        np.arange(1 << (_DIMENSION * level), dtype=np.uint64)
        if level < policy.initial_level
        else np.zeros((0,), dtype=np.uint64)
        for level in range(work.depth)
    ]
    leaves = _leaves(work, refined)
    status = 0
    rounds = 0
    while True:
        analysis = _analyze(work, leaves)
        rounds += 1
        if np.any((analysis.issues != 0) & (leaves.levels == work.depth)):
            status |= int(AdaptiveImplicitSurfaceStatus.MAXIMUM_LEVEL_REACHED)
        if not np.any(analysis.refine):
            return analysis, status, rounds
        candidate = _refined_with(refined, leaves, analysis.refine)
        trial = _leaves(work, candidate)
        added = trial.levels.shape[0] - leaves.levels.shape[0]
        if trial.levels.shape[0] > policy.maximum_boxes:
            status |= int(AdaptiveImplicitSurfaceStatus.BOX_BUDGET_EXHAUSTED)
            return analysis, status, rounds
        used = work.box_evaluations + work.point_evaluations
        if used + _EVALUATIONS_PER_NEW_LEAF * added > policy.maximum_evaluations:
            status |= int(AdaptiveImplicitSurfaceStatus.EVALUATION_BUDGET_EXHAUSTED)
            return analysis, status, rounds
        refined = candidate
        leaves = trial


# ---------------------------------------------------------------------------
# Extraction.


@dataclass(frozen=True, slots=True)
class _Extraction:
    vertices: np.ndarray
    faces: np.ndarray
    status: int
    anchor_residual: float
    vertex_residual: float
    zero_gradient_anchors: int
    feature_vertices: np.ndarray
    feature_edges: np.ndarray
    intersection_checked: bool
    intersection_free: bool
    extraction_evaluations: int
    root_solves: int


def _empty_extraction(status: int) -> _Extraction:
    return _Extraction(
        vertices=np.zeros((0, _DIMENSION), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
        status=status,
        anchor_residual=0.0,
        vertex_residual=0.0,
        zero_gradient_anchors=0,
        feature_vertices=np.zeros((0,), dtype=np.bool_),
        feature_edges=np.zeros((0, 2), dtype=np.int64),
        intersection_checked=False,
        intersection_free=False,
        extraction_evaluations=0,
        root_solves=0,
    )


def _anchors(
    work: _Work, geometry: CompiledGeometry, analysis: _Analysis, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    """Edge roots, unit anchor normals, residual, zero-gradient count, evaluations."""
    edges = analysis.edges
    start = edges.start[active]
    end = start + _UNITS[edges.axis[active]] * edges.length[active, None]
    lower_points = work.coordinates(start)
    upper_points = work.coordinates(end)
    start_sign = work.vertex_signs[edges.start_vertex[active]]
    end_sign = work.vertex_signs[edges.end_vertex[active]]
    for chart in analysis.faces.macrocharts:
        selected = np.isin(edges.keys[active], chart.root_edge_keys)
        lower_points[selected, chart.normal_axis] = chart.macro_box[0, chart.normal_axis]
        upper_points[selected, chart.normal_axis] = chart.macro_box[1, chart.normal_axis]
        increasing = chart.directional_lower[chart.normal_axis] > 0.0
        start_sign[selected] = -1 if increasing else 1
        end_sign[selected] = 1 if increasing else -1
    tiny = np.finfo(np.float64).tiny

    def consistent(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
        # ITP brackets by sign; align float samples with the perturbed signs.
        return np.where(signs < 0, np.minimum(values, -tiny), np.maximum(values, 0.0))

    lower_values = consistent(work.bounds.field_values(lower_points), start_sign)
    upper_values = consistent(work.bounds.field_values(upper_points), end_sign)
    # The fixed solver bound is charged before its one device batch. Endpoint
    # samples are charged by their owner; this covers the root/first-jet batch.
    charge_native_geometry_queries(56 * active.size, work_units=active.size)
    roots, residuals = _isolate_roots(
        geometry.kernel,
        geometry.state,
        jnp.asarray(lower_points),
        jnp.asarray(upper_points),
        jnp.asarray(lower_values),
        jnp.asarray(upper_values),
        jnp.asarray(work.policy.root_tolerance, dtype=jnp.float64),
    )
    roots_ = np.asarray(roots, dtype=np.float64)
    _, gradients = _field_and_gradient(
        geometry.kernel, geometry.state, jnp.asarray(roots_)
    )
    gradients_ = np.asarray(gradients, dtype=np.float64)
    norms = np.linalg.norm(gradients_, axis=1)
    regular = np.isfinite(norms) & (norms > 0.0)
    # A vanishing sampled gradient keeps the Hermite datum of the sign change:
    # the edge direction oriented from inside to outside.
    fallback = _UNITS[edges.axis[active]] * np.where(start_sign < 0, 1.0, -1.0)[:, None]
    normals = np.where(
        regular[:, None], gradients_ / np.where(regular, norms, 1.0)[:, None], fallback
    )
    evaluations = 2 * active.size + (52 + 1 + _DIMENSION) * active.size
    return (
        roots_,
        normals,
        float(np.max(np.asarray(residuals))) if active.size else 0.0,
        int(np.sum(~regular)),
        evaluations,
    )


def _polygon_triangles(
    polygon: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of oriented dual polygons (quads split on the shorter diagonal)."""
    same = polygon == np.roll(polygon, -1, axis=1)
    distinct = 4 - np.sum(same, axis=1)
    if np.any(distinct < 3):
        raise RuntimeError("A dual polygon collapsed below three octree leaves.")
    triangle_rows = np.flatnonzero(distinct == 3)
    keep = np.argsort(same[triangle_rows], axis=1, kind="stable")[:, :3]
    keep = np.sort(keep, axis=1)
    triangles = np.take_along_axis(polygon[triangle_rows], keep, axis=1)
    quads = polygon[distinct == 4]
    first_diagonal = np.linalg.norm(vertices[quads[:, 0]] - vertices[quads[:, 2]], axis=1)
    second_diagonal = np.linalg.norm(
        vertices[quads[:, 1]] - vertices[quads[:, 3]], axis=1
    )
    use_first = first_diagonal <= second_diagonal
    boundary = np.concatenate(
        [
            np.sort(np.stack((row, np.roll(row, -1, axis=1)), axis=-1), axis=-1).reshape(
                (-1, 2)
            )
            for row in (triangles, quads)
        ],
        axis=0,
    )
    modulus = max(vertices.shape[0], 1)
    boundary_keys = np.unique(boundary[:, 0] * modulus + boundary[:, 1])
    first_keys = np.minimum(quads[:, 0], quads[:, 2]) * modulus + np.maximum(
        quads[:, 0], quads[:, 2]
    )
    second_keys = np.minimum(quads[:, 1], quads[:, 3]) * modulus + np.maximum(
        quads[:, 1], quads[:, 3]
    )
    _, first_taken = _lookup(boundary_keys, first_keys)
    _, second_taken = _lookup(boundary_keys, second_keys)
    # A diagonal that duplicates a dual edge would make a nonmanifold edge.
    use_first = np.where(first_taken, False, np.where(second_taken, True, use_first))
    split = np.where(
        use_first[:, None, None],
        np.stack((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]), axis=1),
        np.stack((quads[:, (0, 1, 3)], quads[:, (1, 2, 3)]), axis=1),
    ).reshape((-1, 3))
    dual_edges = np.unique(boundary, axis=0)
    return np.concatenate((triangles, split), axis=0), dual_edges


def _closed_manifold(faces: np.ndarray, vertex_count: int) -> bool:
    halfedges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    keys = np.min(halfedges, axis=1) * vertex_count + np.max(halfedges, axis=1)
    _, counts = np.unique(keys, return_counts=True)
    directed = halfedges[:, 0] * vertex_count + halfedges[:, 1]
    return bool(np.all(counts == 2)) and np.unique(directed).size == directed.size


def _intersections(
    vertices: np.ndarray, faces: np.ndarray, policy: AdaptiveImplicitSurfacePolicy
) -> tuple[bool, bool]:
    """Whether the self-intersection check ran and found no intersecting pair."""
    triangles = vertices[faces]
    hierarchy = prepare_bvh(
        np.min(triangles, axis=1), np.max(triangles, axis=1), dtype=jnp.float64
    )
    first, second = bvh_overlap_pairs_host(hierarchy, hierarchy, include_touching=True)
    ordered = first < second
    first, second = first[ordered], second[ordered]
    shares = np.any(faces[first][:, :, None] == faces[second][:, None, :], axis=(1, 2))
    first, second = first[~shares], second[~shares]
    if first.size > policy.maximum_intersection_pairs:
        return False, False
    for start in range(0, first.size, _PAIR_CHUNK):
        rows = np.arange(start, start + _PAIR_CHUNK)
        # Pad with the last pair so one compiled chunk serves every call.
        rows = np.minimum(rows, first.size - 1)
        hits = _intersecting_pairs(
            jnp.asarray(triangles[first[rows]]),
            jnp.asarray(triangles[second[rows]]),
            tolerance=policy.root_tolerance,
        )
        if bool(np.any(np.asarray(hits))):
            return True, False
    return True, True


@partial(jax.jit, static_argnames=("tolerance",))
def _intersecting_pairs(first: Array, second: Array, *, tolerance: float) -> Array:
    return jax.vmap(lambda one, two: _triangles_intersect(one, two, tolerance))(
        first, second
    )


def _features(
    normals: np.ndarray, anchor_indices: np.ndarray, anchor_mask: np.ndarray, angle: float
) -> np.ndarray:
    selected = normals[anchor_indices]
    cosine = np.asarray(contract("vai,vbi->vab", selected, selected))
    valid = anchor_mask[:, :, None] & anchor_mask[:, None, :]
    return np.any(valid & (cosine < math.cos(angle)), axis=(1, 2))


def _source_cycle_boxes(work: _Work, analysis: _Analysis, /) -> np.ndarray:
    leaves = analysis.leaves
    boxes = np.stack(
        (
            work.coordinates(leaves.corners[analysis.cycle_leaf]),
            work.coordinates(
                leaves.corners[analysis.cycle_leaf]
                + leaves.sizes[analysis.cycle_leaf, None]
            ),
        ),
        axis=1,
    )
    cycle_codes = leaves.code_starts[analysis.cycle_leaf]
    for chart in analysis.faces.macrocharts:
        selected = np.isin(cycle_codes, chart.incident_leaf_codes)
        boxes[selected] = chart.macro_box
    return boxes


def _extract(work: _Work, geometry: CompiledGeometry, analysis: _Analysis) -> _Extraction:
    """Dual contouring of the certified decomposition: one vertex per leaf cycle."""
    policy = work.policy
    edges = analysis.edges
    interior = np.all(edges.incident >= 0, axis=1)
    if np.any(edges.crossing & ~interior):
        return _empty_extraction(
            int(AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING)
        )
    active = np.flatnonzero(edges.crossing)
    if not active.size:
        return _empty_extraction(int(AdaptiveImplicitSurfaceStatus.NO_SURFACE))
    used = work.box_evaluations + work.point_evaluations
    extraction_bound = 58 * active.size + analysis.cycle_leaf.shape[0]
    if used + extraction_bound > policy.maximum_evaluations:
        return _empty_extraction(
            int(AdaptiveImplicitSurfaceStatus.EVALUATION_BUDGET_EXHAUSTED)
        )
    if active.size > work.maximum_root_solves:
        return _empty_extraction(
            int(AdaptiveImplicitSurfaceStatus.EVALUATION_BUDGET_EXHAUSTED)
        )
    anchors, normals, anchor_residual, zero_gradients, evaluations = _anchors(
        work, geometry, analysis, active
    )
    edge_anchor = np.full((edges.keys.shape[0],), -1, dtype=np.int64)
    edge_anchor[active] = np.arange(active.size)
    node_anchor = edge_anchor[analysis.node_edge]
    if np.any(node_anchor < 0):
        raise RuntimeError("A leaf cycle references a non-crossing edge.")
    cycle_count = analysis.cycle_leaf.shape[0]
    order = np.lexsort((node_anchor, analysis.node_cycle))
    grouped_cycle = analysis.node_cycle[order]
    width_counts = np.bincount(grouped_cycle, minlength=cycle_count)
    width = int(np.max(width_counts))
    slot = np.arange(order.size) - np.repeat(
        np.cumsum(width_counts) - width_counts, width_counts
    )
    anchor_indices = np.zeros((cycle_count, width), dtype=np.int32)
    anchor_mask = np.zeros((cycle_count, width), dtype=np.bool_)
    anchor_indices[grouped_cycle, slot] = node_anchor[order]
    anchor_mask[grouped_cycle, slot] = True
    cycle_boxes = _source_cycle_boxes(work, analysis)
    cell_lower, cell_upper = cycle_boxes[:, 0], cycle_boxes[:, 1]
    vertices, _ = _qef_vertices(
        anchors,
        normals,
        anchor_indices,
        anchor_mask,
        cell_lower,
        cell_upper,
        policy.qef_regularization,
        policy.root_tolerance,
    )
    # Polygon corners: the cycle of each incident leaf containing the edge.
    node_keys = analysis.node_leaf * edges.keys.shape[0] + analysis.node_edge
    node_order = np.argsort(node_keys, kind="stable")
    probe_keys = edges.incident[active] * edges.keys.shape[0] + active[:, None]
    position, found = _lookup(node_keys[node_order], probe_keys)
    if not np.all(found):
        raise RuntimeError("A crossing edge is missing from an incident leaf cycle.")
    polygon = analysis.node_cycle[node_order][position]
    # The incident-cell order turns positively about +axis; the outward normal
    # points from the inside endpoint toward the outside endpoint.
    inside_first = work.vertex_signs[edges.start_vertex[active]] < 0
    polygon = np.where(inside_first[:, None], polygon, polygon[:, ::-1])
    faces, dual_edges = _polygon_triangles(polygon, vertices)
    if faces.shape[0] > policy.maximum_faces:
        raise ValueError(
            f"Adaptive implicit surface requires {faces.shape[0]} faces; "
            f"maximum_faces is {policy.maximum_faces}."
        )
    status = 0
    if not _closed_manifold(faces, vertices.shape[0]):
        status |= int(AdaptiveImplicitSurfaceStatus.NONMANIFOLD_EXTRACTION)
    triangles = vertices[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.all(area > policy.minimum_face_area):
        status |= int(AdaptiveImplicitSurfaceStatus.DEGENERATE_FACE)
    checked, free = (False, False) if status else _intersections(vertices, faces, policy)
    if checked and not free:
        status |= int(AdaptiveImplicitSurfaceStatus.SELF_INTERSECTION)
    vertex_residual = float(np.max(np.abs(work.bounds.field_values(vertices))))
    feature = _features(normals, anchor_indices, anchor_mask, policy.feature_angle)
    feature_edges = dual_edges[feature[dual_edges[:, 0]] & feature[dual_edges[:, 1]]]
    return _Extraction(
        vertices=vertices,
        faces=faces.astype(np.int64),
        status=status,
        anchor_residual=anchor_residual,
        vertex_residual=vertex_residual,
        zero_gradient_anchors=zero_gradients,
        feature_vertices=feature,
        feature_edges=feature_edges.astype(np.int64),
        intersection_checked=checked,
        intersection_free=free,
        extraction_evaluations=evaluations + vertices.shape[0],
        root_solves=active.size,
    )


# ---------------------------------------------------------------------------
# Published products.


@final
class ImplicitVolumeClassification(StrictModule, NonTrainableState):
    """Inside/outside/unknown classes of queried points or boxes with bounds.

    ``classes`` holds `ImplicitVolumeClass` values. ``certified`` is true only
    when the bounds are rigorous enclosures; sampled classes are heuristics.
    """

    __strict_contract__ = True

    classes: HostInt64[_QueryDim]
    value_lower: HostFloat64[_QueryDim]
    value_upper: HostFloat64[_QueryDim]
    certified: bool = eqx.field(static=True)

    def __init__(
        self,
        classes: ArrayLike,
        value_lower: ArrayLike,
        value_upper: ArrayLike,
        /,
        *,
        certified: bool,
    ) -> None:
        scope = Scope()
        self.classes = parse(
            np.asarray(classes, dtype=np.int64),
            HostInt64[_QueryDim],
            "classes",
            scope=scope,
        )
        self.value_lower = parse(
            np.asarray(value_lower, dtype=np.float64),
            HostFloat64[_QueryDim],
            "value_lower",
            scope=scope,
        )
        self.value_upper = parse(
            np.asarray(value_upper, dtype=np.float64),
            HostFloat64[_QueryDim],
            "value_upper",
            scope=scope,
        )
        self.certified = bool(certified)


def _classes(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.where(
        upper < 0.0,
        int(ImplicitVolumeClass.INSIDE),
        np.where(
            lower > 0.0,
            int(ImplicitVolumeClass.OUTSIDE),
            int(ImplicitVolumeClass.UNKNOWN),
        ),
    ).astype(np.int64)


@final
class ImplicitVolumeQuery(StrictModule, NonTrainableState):
    """Volume-domain inside/outside/unknown queries over a discovered octree.

    Leaves whose value enclosure excludes zero classify every contained point
    without evaluation; other points and all boxes are classified by the same
    bound source that drove discovery. Classes are certified only for rigorous
    value enclosures.
    """

    __strict_contract__ = True

    bounds: _AbstractFieldBounds
    leaf_lower: HostFloat64[_LeafDim, Literal[3]]
    leaf_upper: HostFloat64[_LeafDim, Literal[3]]
    leaf_classes: HostInt64[_LeafDim]
    leaf_value_lower: HostFloat64[_LeafDim]
    leaf_value_upper: HostFloat64[_LeafDim]
    leaf_code_starts: HostInt64[_LeafDim]
    domain: HostFloat64[Literal[2], Literal[3]]
    maximum_level: int = eqx.field(static=True)
    certified: bool = eqx.field(static=True)
    query_id: str = eqx.field(static=True)

    def __init__(
        self,
        bounds: _AbstractFieldBounds,
        leaf_lower: ArrayLike,
        leaf_upper: ArrayLike,
        leaf_value_lower: ArrayLike,
        leaf_value_upper: ArrayLike,
        leaf_code_starts: ArrayLike,
        domain: ArrayLike,
        /,
        *,
        maximum_level: int,
    ) -> None:
        if not isinstance(bounds, _AbstractFieldBounds):
            raise TypeError("bounds must be implicit field bounds.")
        scope = Scope()
        lower = parse(
            np.asarray(leaf_lower, dtype=np.float64),
            HostFloat64[_LeafDim, Literal[3]],
            "leaf_lower",
            scope=scope,
        )
        upper = parse(
            np.asarray(leaf_upper, dtype=np.float64),
            HostFloat64[_LeafDim, Literal[3]],
            "leaf_upper",
            scope=scope,
        )
        value_lower = parse(
            np.asarray(leaf_value_lower, dtype=np.float64),
            HostFloat64[_LeafDim],
            "leaf_value_lower",
            scope=scope,
        )
        value_upper = parse(
            np.asarray(leaf_value_upper, dtype=np.float64),
            HostFloat64[_LeafDim],
            "leaf_value_upper",
            scope=scope,
        )
        codes = parse(
            np.asarray(leaf_code_starts, dtype=np.int64),
            HostInt64[_LeafDim],
            "leaf_code_starts",
            scope=scope,
        )
        domain_ = parse(
            np.asarray(domain, dtype=np.float64),
            HostFloat64[Literal[2], Literal[3]],
            "domain",
        )
        self.bounds = bounds
        self.leaf_lower = lower
        self.leaf_upper = upper
        self.leaf_classes = _classes(value_lower, value_upper)
        self.leaf_value_lower = value_lower
        self.leaf_value_upper = value_upper
        self.leaf_code_starts = codes
        self.domain = domain_
        self.maximum_level = int(maximum_level)
        self.certified = bounds.value_rigorous
        self.query_id = canonical_fingerprint(
            {
                "kind": "implicit-volume-query",
                "enclosure": bounds.enclosure,
                "leaf_lower": array_tree_fingerprint(lower),
                "leaf_upper": array_tree_fingerprint(upper),
                "leaf_value_lower": array_tree_fingerprint(value_lower),
                "leaf_value_upper": array_tree_fingerprint(value_upper),
            }
        )

    def classify_points(self, points: ArrayLike, /) -> ImplicitVolumeClassification:
        """Classify points; excluded leaves answer without field evaluation."""
        points_ = parse(
            np.asarray(points, dtype=np.float64),
            HostFloat64[_QueryDim, Literal[3]],
            "points",
        )
        if not np.all(np.isfinite(points_)):
            raise ValueError("points must be finite.")
        resolution = 1 << self.maximum_level
        extent = self.domain[1] - self.domain[0]
        cells = np.floor((points_ - self.domain[0]) / extent * resolution).astype(
            np.int64
        )
        inside = np.all((cells >= 0) & (cells < resolution), axis=1)
        codes = morton_encode_integer_host(
            np.where(inside[:, None], cells, 0), self.maximum_level
        ).astype(np.int64)
        leaf = np.searchsorted(self.leaf_code_starts, codes, side="right") - 1
        leaf = np.where(inside, leaf, -1)
        known = (leaf >= 0) & (
            self.leaf_classes[np.maximum(leaf, 0)] != int(ImplicitVolumeClass.UNKNOWN)
        )
        value_lower = np.where(known, self.leaf_value_lower[np.maximum(leaf, 0)], 0.0)
        value_upper = np.where(known, self.leaf_value_upper[np.maximum(leaf, 0)], 0.0)
        pending = np.flatnonzero(~known)
        if pending.size:
            lower, upper = self.bounds.point_values(points_[pending])
            value_lower[pending] = lower
            value_upper[pending] = upper
        return ImplicitVolumeClassification(
            _classes(value_lower, value_upper),
            value_lower,
            value_upper,
            certified=self.certified,
        )

    def classify_boxes(
        self, lower: ArrayLike, upper: ArrayLike, /
    ) -> ImplicitVolumeClassification:
        """Classify axis-aligned boxes by their value enclosures."""
        scope = Scope()
        lower_ = parse(
            np.asarray(lower, dtype=np.float64),
            HostFloat64[_QueryDim, Literal[3]],
            "lower",
            scope=scope,
        )
        upper_ = parse(
            np.asarray(upper, dtype=np.float64),
            HostFloat64[_QueryDim, Literal[3]],
            "upper",
            scope=scope,
        )
        if not (np.all(np.isfinite(lower_)) and np.all(np.isfinite(upper_))) or np.any(
            lower_ > upper_
        ):
            raise ValueError("Query boxes require finite lower <= upper corners.")
        bounds = self.bounds.boxes(lower_, upper_)
        return ImplicitVolumeClassification(
            _classes(bounds.value_lower, bounds.value_upper),
            bounds.value_lower,
            bounds.value_upper,
            certified=self.certified,
        )


@final
class AdaptiveImplicitSurfaceEvidence(StrictModule, NonTrainableState):
    """Approximation, certification, and resource evidence of adaptive discovery.

    ``unresolved_boxes[i]`` holds the lower/upper corners of an unresolved final
    leaf and ``unresolved_issues[i]`` its `AdaptiveImplicitBoxIssue` flags.
    ``maximum_vertex_residual`` is ``max |f|`` at mesh vertices;
    ``maximum_surface_box_diagonal`` bounds the distance from every mesh vertex to
    the zero set patch in its leaf.
    """

    __strict_contract__ = True

    unresolved_boxes: HostFloat64[_UnresolvedDim, Literal[2], Literal[3]]
    unresolved_issues: HostInt64[_UnresolvedDim]
    accuracy: ImplicitDiscoveryAccuracy = eqx.field(static=True)
    enclosure: ImplicitDiscoveryEnclosure = eqx.field(static=True)
    status: int = eqx.field(static=True)
    leaf_count: int = eqx.field(static=True)
    surface_leaf_count: int = eqx.field(static=True)
    singular_leaf_count: int = eqx.field(static=True)
    root_solves: int = eqx.field(static=True)
    maximum_level_reached: int = eqx.field(static=True)
    refinement_rounds: int = eqx.field(static=True)
    box_evaluations: int = eqx.field(static=True)
    point_evaluations: int = eqx.field(static=True)
    extraction_evaluations: int = eqx.field(static=True)
    zero_vertex_count: int = eqx.field(static=True)
    zero_gradient_anchor_count: int = eqx.field(static=True)
    feature_vertex_count: int = eqx.field(static=True)
    maximum_anchor_residual: float = eqx.field(static=True)
    maximum_vertex_residual: float = eqx.field(static=True)
    maximum_surface_box_diagonal: float = eqx.field(static=True)
    maximum_flatness_bound: float = eqx.field(static=True)
    intersection_checked: bool = eqx.field(static=True)
    intersection_free: bool = eqx.field(static=True)
    rounding_model: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        unresolved_boxes: ArrayLike,
        unresolved_issues: ArrayLike,
        /,
        *,
        accuracy: ImplicitDiscoveryAccuracy,
        enclosure: ImplicitDiscoveryEnclosure,
        status: int,
        leaf_count: int,
        surface_leaf_count: int,
        singular_leaf_count: int,
        maximum_level_reached: int,
        refinement_rounds: int,
        box_evaluations: int,
        point_evaluations: int,
        extraction_evaluations: int,
        zero_vertex_count: int,
        zero_gradient_anchor_count: int,
        feature_vertex_count: int,
        maximum_anchor_residual: float,
        maximum_vertex_residual: float,
        maximum_surface_box_diagonal: float,
        maximum_flatness_bound: float,
        intersection_checked: bool,
        intersection_free: bool,
        root_solves: int,
    ) -> None:
        scope = Scope()
        boxes = parse(
            np.asarray(unresolved_boxes, dtype=np.float64).reshape((-1, 2, _DIMENSION)),
            HostFloat64[_UnresolvedDim, Literal[2], Literal[3]],
            "unresolved_boxes",
            scope=scope,
        )
        issues = parse(
            np.asarray(unresolved_issues, dtype=np.int64),
            HostInt64[_UnresolvedDim],
            "unresolved_issues",
            scope=scope,
        )
        self.unresolved_boxes = boxes
        self.unresolved_issues = issues
        self.accuracy = parse(accuracy, ImplicitDiscoveryAccuracy, "accuracy")
        self.enclosure = parse(enclosure, ImplicitDiscoveryEnclosure, "enclosure")
        self.status = int(status)
        self.leaf_count = int(leaf_count)
        self.surface_leaf_count = int(surface_leaf_count)
        self.singular_leaf_count = int(singular_leaf_count)
        self.maximum_level_reached = int(maximum_level_reached)
        self.refinement_rounds = int(refinement_rounds)
        self.box_evaluations = int(box_evaluations)
        self.point_evaluations = int(point_evaluations)
        self.extraction_evaluations = int(extraction_evaluations)
        self.root_solves = root_solves
        self.zero_vertex_count = int(zero_vertex_count)
        self.zero_gradient_anchor_count = int(zero_gradient_anchor_count)
        self.feature_vertex_count = int(feature_vertex_count)
        self.maximum_anchor_residual = float(maximum_anchor_residual)
        self.maximum_vertex_residual = float(maximum_vertex_residual)
        self.maximum_surface_box_diagonal = float(maximum_surface_box_diagonal)
        self.maximum_flatness_bound = float(maximum_flatness_bound)
        self.intersection_checked = bool(intersection_checked)
        self.intersection_free = bool(intersection_free)
        self.rounding_model = ENCLOSURE_ROUNDING_MODEL
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "adaptive-implicit-surface-evidence",
                "accuracy": self.accuracy,
                "enclosure": self.enclosure,
                "status": self.status,
                "unresolved_boxes": array_tree_fingerprint(boxes),
                "unresolved_issues": array_tree_fingerprint(issues),
                "leaf_count": self.leaf_count,
                "box_evaluations": self.box_evaluations,
                "point_evaluations": self.point_evaluations,
            }
        )

    @property
    def certified(self) -> bool:
        return self.accuracy == "certified"

    @property
    def unresolved_count(self) -> int:
        return self.unresolved_issues.shape[0]

    @property
    def status_flags(self) -> AdaptiveImplicitSurfaceStatus:
        return AdaptiveImplicitSurfaceStatus(self.status)


@final
class AdaptiveImplicitSurface(StrictModule, NonTrainableState):
    """Adaptive discovery product: surface, evidence, cover, and volume queries.

    ``mesh`` is ``None`` when the zero set is empty in the domain or when a
    closed embedded extraction could not be published; ``evidence.status`` says
    which. ``cover`` exists for rigorous value enclosures; ``topology`` exists
    only for a certified extraction. ``feature_edges`` index ``mesh`` vertices
    joined across sharp features (anchor normals beyond ``feature_angle``).
    """

    __strict_contract__ = True

    mesh: TriangleMesh | None
    evidence: AdaptiveImplicitSurfaceEvidence
    cover: CertifiedImplicitCover | None
    topology: CertifiedImplicitTopology | None
    volume: ImplicitVolumeQuery
    feature_edges: HostInt64[_FeatureEdgeDim, Literal[2]]
    boundary_witness_boxes: HostFloat64[_QueryDim, Literal[2], Literal[3]]
    source_id: str = eqx.field(static=True)
    macrocharts: tuple[ImplicitNormalFiberMacrochart, ...]
    geometry: CompiledGeometry
    policy: AdaptiveImplicitSurfacePolicy = eqx.field(static=True)
    root_solve_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        mesh: TriangleMesh | None,
        evidence: AdaptiveImplicitSurfaceEvidence,
        cover: CertifiedImplicitCover | None,
        topology: CertifiedImplicitTopology | None,
        volume: ImplicitVolumeQuery,
        feature_edges: ArrayLike,
        boundary_witness_boxes: ArrayLike,
        macrocharts: tuple[ImplicitNormalFiberMacrochart, ...],
        /,
        *,
        source_id: str,
        geometry: CompiledGeometry,
        policy: AdaptiveImplicitSurfacePolicy,
        root_solve_capacity: int,
    ) -> None:
        if mesh is not None and not isinstance(mesh, TriangleMesh):
            raise TypeError("mesh must be a TriangleMesh or None.")
        if not isinstance(evidence, AdaptiveImplicitSurfaceEvidence):
            raise TypeError("evidence must be AdaptiveImplicitSurfaceEvidence.")
        if topology is not None and (mesh is None or cover is None):
            raise ValueError("A certified topology requires a mesh and a cover.")
        self.mesh = mesh
        self.evidence = evidence
        self.cover = cover
        self.topology = topology
        self.volume = volume
        self.feature_edges = parse(
            np.asarray(feature_edges, dtype=np.int64).reshape((-1, 2)),
            HostInt64[_FeatureEdgeDim, Literal[2]],
            "feature_edges",
        )
        witness_boxes = parse(
            np.array(boundary_witness_boxes, dtype=np.float64, copy=True).reshape(
                (-1, 2, 3)
            ),
            HostFloat64[_QueryDim, Literal[2], Literal[3]],
            "boundary_witness_boxes",
        )
        if not np.all(np.isfinite(witness_boxes)) or np.any(
            witness_boxes[:, 0] > witness_boxes[:, 1]
        ):
            raise ValueError(
                "Cycle-root witness boxes require finite lower <= upper bounds."
            )
        if mesh is not None and witness_boxes.shape[0] != mesh.vertices.shape[0]:
            raise ValueError("Cycle-root witness boxes must name every extracted vertex.")
        witness_boxes.setflags(write=False)
        self.boundary_witness_boxes = witness_boxes
        self.macrocharts = macrocharts
        self.geometry, self.policy, self.root_solve_capacity = (
            geometry,
            policy,
            root_solve_capacity,
        )
        self.source_id = str(source_id)


@final
class AdaptiveImplicitBoundarySource(StrictModule, NonTrainableState):
    """Continuous zero-set bounds from an exhaustive cover and root witnesses.

    Possible-zero boxes enclose every boundary point. A cycle's box contains a
    certified sign-changing edge, so it contains an actual zero even when the
    extracted QEF vertex is not on the zero set. No reach or signed-distance
    interpretation is inferred from the scalar field.
    """

    geometry: CompiledGeometry
    surface: AdaptiveImplicitSurface
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    surface_numeric_id: str = eqx.field(static=True)
    maximum_distance_pairs: int = eqx.field(static=True)
    maximum_scratch_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        geometry: CompiledGeometry,
        surface: AdaptiveImplicitSurface,
        source_revision: str,
        /,
        *,
        maximum_distance_pairs: int = 4_000_000,
        maximum_scratch_bytes: int = 64 * 1024**2,
    ) -> None:
        if surface.cover is None or surface.topology is None or surface.mesh is None:
            raise ValueError(
                "Adaptive boundary queries require a certified complete extraction."
            )
        surface.cover.require_bound(surface.source_id, implicit_state_id(geometry))
        if (
            not surface.cover.complete
            or not surface.cover.established
            or not surface.topology.established
        ):
            raise ValueError(
                "Adaptive boundary queries require independently established whole-source premises."
            )
        if maximum_distance_pairs < 1 or maximum_scratch_bytes < 256:
            raise ValueError(
                "Adaptive boundary distance work and scratch budgets must be positive."
            )
        if (
            not surface.evidence.certified
            or surface.evidence.unresolved_count
            or not surface.evidence.intersection_free
            or surface.boundary_witness_boxes.shape[0] != surface.mesh.vertices.shape[0]
        ):
            raise ValueError(
                "Adaptive boundary queries require complete cycle-root witnesses."
            )
        self.geometry = geometry
        self.surface = surface
        self.surface_numeric_id = canonical_fingerprint(array_tree_fingerprint(surface))
        self.source_id = surface.source_id
        self.source_revision = source_revision
        self.maximum_distance_pairs = maximum_distance_pairs
        self.maximum_scratch_bytes = maximum_scratch_bytes

    @property
    def ambient_dimension(self) -> int:
        return 3

    def _boxes(self) -> tuple[np.ndarray, np.ndarray]:
        cover = self.surface.cover
        if cover is None:
            raise RuntimeError("A certified adaptive source lost its enclosure cover.")
        cover.require_bound(self.source_id, implicit_state_id(self.geometry))
        if (
            canonical_fingerprint(array_tree_fingerprint(self.surface))
            != self.surface_numeric_id
        ):
            raise ValueError(
                "Adaptive boundary source enclosures or cycle witnesses are stale."
            )
        possible = (cover.value_lower <= 0.0) & (cover.value_upper >= 0.0)
        return np.asarray(cover.boxes)[possible], self.surface.boundary_witness_boxes

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        charge_native_geometry_queries(points.shape[0])
        carrier_bytes = (
            self.surface.volume.leaf_lower.shape[0] * 64 + points.shape[0] * 64
        )
        if carrier_bytes + 256 * points.shape[0] > self.maximum_scratch_bytes:
            return SourceBoundaryDistance(
                np.zeros(points.shape[0], dtype=np.float64),
                np.full(points.shape[0], np.inf, dtype=np.float64),
                "certified",
            )
        possible, witnessed = self._boxes()
        if (
            points.shape[0] * (possible.shape[0] + witnessed.shape[0])
            > self.maximum_distance_pairs
        ):
            return SourceBoundaryDistance(
                np.zeros(points.shape[0], dtype=np.float64),
                np.full(points.shape[0], np.inf, dtype=np.float64),
                "certified",
            )
        block_size = min(
            256,
            (self.maximum_scratch_bytes - carrier_bytes) // max(256 * points.shape[0], 1),
        )
        if block_size < 1:
            return SourceBoundaryDistance(
                np.zeros(points.shape[0], dtype=np.float64),
                np.full(points.shape[0], np.inf, dtype=np.float64),
                "certified",
            )
        lower = np.full(points.shape[0], np.inf, dtype=np.float64)
        upper = np.full(points.shape[0], np.inf, dtype=np.float64)
        for boxes, nearest, farthest in (
            (possible, lower, False),
            (witnessed, upper, True),
        ):
            for start in range(0, boxes.shape[0], block_size):
                block = boxes[start : start + block_size]
                low = np.nextafter(block[None, :, 0] - points[:, None], -np.inf)
                high = np.nextafter(points[:, None] - block[None, :, 1], -np.inf)
                if farthest:
                    displacement = np.nextafter(
                        np.maximum(
                            np.abs(points[:, None] - block[None, :, 0]),
                            np.abs(points[:, None] - block[None, :, 1]),
                        ),
                        np.inf,
                    )
                    squared = np.nextafter(displacement * displacement, np.inf)
                    distance = np.nextafter(
                        np.sqrt(np.nextafter(np.sum(squared, axis=2), np.inf)), np.inf
                    )
                else:
                    displacement = np.maximum(np.maximum(high, low), 0.0)
                    squared = np.nextafter(displacement * displacement, 0.0)
                    distance = np.nextafter(
                        np.sqrt(
                            np.maximum(np.nextafter(np.sum(squared, axis=2), 0.0), 0.0)
                        ),
                        0.0,
                    )
                nearest[:] = np.minimum(nearest, np.min(distance, axis=1))
        return SourceBoundaryDistance(lower, upper, "certified")

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        if self.surface.volume.leaf_lower.shape[0] * 256 > self.maximum_scratch_bytes:
            return SourceBoundarySamples(
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                "certified",
                complete=False,
            )
        possible, _ = self._boxes()
        if possible.shape[0] > maximum_samples:
            return SourceBoundarySamples(
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                "certified",
                complete=False,
            )
        centers = 0.5 * possible[:, 0] + 0.5 * possible[:, 1]
        displacement = np.nextafter(
            np.maximum(
                np.abs(np.nextafter(centers - possible[:, 0], np.inf)),
                np.abs(np.nextafter(centers - possible[:, 1], -np.inf)),
            ),
            np.inf,
        )
        squared = np.nextafter(displacement * displacement, np.inf)
        radius = np.nextafter(
            np.sqrt(np.nextafter(np.sum(squared, axis=1), np.inf)), np.inf
        )
        # Some possible-zero boxes can be empty; actual cycle witnesses bound
        # their centers' error instead of treating an interval overlap as a root.
        error = self.boundary_distance(centers).upper
        return SourceBoundarySamples(centers, radius, error, "certified", complete=True)


# ---------------------------------------------------------------------------
# Entry point.


def _validate(
    geometry: CompiledGeometry,
    domain: np.ndarray,
    policy: AdaptiveImplicitSurfacePolicy,
    source_id: str,
) -> None:
    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be CompiledGeometry.")
    if not isinstance(policy, AdaptiveImplicitSurfacePolicy):
        raise TypeError("policy must be AdaptiveImplicitSurfacePolicy.")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty string.")
    if (
        geometry.ambient_dimension != _DIMENSION
        or geometry.kind is not GeometryKind.REGION
    ):
        raise ValueError(
            "Adaptive implicit discovery requires a three-dimensional region."
        )
    if not np.all(np.isfinite(domain)) or np.any(domain[1] <= domain[0]):
        raise ValueError("domain must hold finite lower < upper corners.")
    initial = 1 << (_DIMENSION * policy.initial_level)
    if initial > policy.maximum_boxes:
        raise ValueError(
            f"initial_level {policy.initial_level} needs {initial} boxes; "
            f"maximum_boxes is {policy.maximum_boxes}."
        )
    if _EVALUATIONS_PER_NEW_LEAF * initial > policy.maximum_evaluations:
        raise ValueError(
            f"initial_level {policy.initial_level} can need "
            f"{_EVALUATIONS_PER_NEW_LEAF * initial} evaluations; "
            f"maximum_evaluations is {policy.maximum_evaluations}."
        )
    if not bool(np.asarray(geometry.validity().accepted)):
        raise ValueError("Adaptive implicit discovery geometry must be valid.")
    certificate = geometry.field_certificate
    if certificate.sign_reliability is SignReliability.UNRELIABLE:
        raise ValueError("Adaptive implicit discovery requires a meaningful field sign.")
    if (
        certificate.sign_reliability is not SignReliability.RELIABLE
        and policy.enclosure != "sampled"
    ):
        raise ValueError(
            "Enclosed discovery requires a reliable field sign; the source certifies "
            f"{certificate.sign_reliability.value!r} sign, so select the 'sampled' enclosure."
        )
    if (
        certificate.zero_set_accuracy is ZeroSetAccuracy.APPROXIMATE
        and not policy.allow_approximate_zero_set
    ):
        raise ValueError("Approximate zero sets require explicit policy approval.")


def _accuracy(
    bounds: _AbstractFieldBounds,
    unresolved: bool,
    status: int,
    extraction: _Extraction,
) -> ImplicitDiscoveryAccuracy:
    if not bounds.value_rigorous:
        return "sampled"
    blocking = int(
        AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING
        | AdaptiveImplicitSurfaceStatus.DEGENERATE_FACE
        | AdaptiveImplicitSurfaceStatus.SELF_INTERSECTION
        | AdaptiveImplicitSurfaceStatus.NONMANIFOLD_EXTRACTION
    )
    empty = bool(status & int(AdaptiveImplicitSurfaceStatus.NO_SURFACE))
    embedded = empty or extraction.intersection_free
    if bounds.gradient_rigorous and not unresolved and not status & blocking and embedded:
        return "certified"
    return "enclosed"


def _new_work(
    bounds: _AbstractFieldBounds,
    policy: AdaptiveImplicitSurfacePolicy,
    domain: np.ndarray,
    source_id: str,
    maximum_root_solves: int,
) -> _Work:
    depth = policy.maximum_level
    return _Work(
        bounds=bounds,
        policy=policy,
        address=MortonAddressPlan(tuple(domain[0]), tuple(domain[1]), depth),
        lower=domain[0].copy(),
        upper=domain[1].copy(),
        spacing=(domain[1] - domain[0]) / float(1 << depth),
        depth=depth,
        modulus=(1 << depth) + 1,
        box_evaluations=0,
        point_evaluations=0,
        isolation_exhausted=False,
        leaf_bounds={},
        vertex_keys=np.zeros((0,), dtype=np.int64),
        vertex_signs=np.zeros((0,), dtype=np.int8),
        vertex_zero=np.zeros((0,), dtype=np.bool_),
        checked_keys=np.zeros((0,), dtype=np.int64),
        checked_ok=np.zeros((0,), dtype=np.bool_),
        source_id=source_id,
        macrocharts={},
        maximum_root_solves=maximum_root_solves,
    )


def _cover(
    geometry: CompiledGeometry,
    bounds: _AbstractFieldBounds,
    leaf_bounds: _BoxBounds,
    boxes: np.ndarray,
    domain: np.ndarray,
    source_id: str,
) -> CertifiedImplicitCover | None:
    """The final leaves as a cover, bound to the source state, when rigorous."""
    match bounds.enclosure:
        case "interval":
            # Unbounded enclosure endpoints become the largest finite float: the
            # field and its derivatives are finite, so the enclosure stays valid.
            largest = float(np.finfo(np.float64).max)

            def finite(values: np.ndarray) -> np.ndarray:
                return np.clip(values, -largest, largest)

            return _established_implicit_cover(
                boxes,
                finite(leaf_bounds.value_lower),
                finite(leaf_bounds.value_upper),
                finite(leaf_bounds.gradient_lower),
                finite(leaf_bounds.gradient_upper),
                domain=domain,
                source_id=source_id,
                state_id=implicit_state_id(geometry),
                bound_origin="interval_arithmetic",
                directions=DISCOVERY_DIRECTIONS,
                directional_lower=finite(leaf_bounds.directional_lower),
                directional_upper=finite(leaf_bounds.directional_upper),
            )
        case "lipschitz":
            return establish_implicit_cover(
                geometry, boxes, domain=domain, source_id=source_id
            )
        case "sampled":
            return None
        case _:
            raise ValueError(
                f"Unknown implicit discovery enclosure {bounds.enclosure!r}."
            )


def _evidence(
    work: _Work,
    analysis: _Analysis,
    extraction: _Extraction,
    boxes: np.ndarray,
    *,
    accuracy: ImplicitDiscoveryAccuracy,
    status: int,
    rounds: int,
) -> AdaptiveImplicitSurfaceEvidence:
    unresolved = np.flatnonzero(analysis.issues != 0)
    surface = analysis.cycles >= 1
    return AdaptiveImplicitSurfaceEvidence(
        boxes[unresolved],
        analysis.issues[unresolved],
        accuracy=accuracy,
        enclosure=work.bounds.enclosure,
        status=status,
        leaf_count=analysis.leaves.levels.shape[0],
        surface_leaf_count=int(np.sum(surface)),
        singular_leaf_count=int(
            np.sum((analysis.issues & int(AdaptiveImplicitBoxIssue.SINGULAR)) != 0)
        ),
        maximum_level_reached=int(np.max(analysis.leaves.levels)),
        refinement_rounds=rounds,
        box_evaluations=work.box_evaluations,
        point_evaluations=work.point_evaluations,
        extraction_evaluations=extraction.extraction_evaluations,
        zero_vertex_count=int(np.sum(work.vertex_zero)),
        zero_gradient_anchor_count=extraction.zero_gradient_anchors,
        feature_vertex_count=int(np.sum(extraction.feature_vertices)),
        maximum_anchor_residual=extraction.anchor_residual,
        maximum_vertex_residual=extraction.vertex_residual,
        maximum_surface_box_diagonal=float(
            np.max(
                np.linalg.norm(
                    np.diff(_source_cycle_boxes(work, analysis), axis=1)[:, 0], axis=1
                ),
                initial=0.0,
            )
        ),
        maximum_flatness_bound=float(np.max(analysis.flatness[surface], initial=0.0)),
        intersection_checked=extraction.intersection_checked,
        intersection_free=extraction.intersection_free,
        root_solves=extraction.root_solves,
    )


def discover_adaptive_implicit_surface(
    geometry: CompiledGeometry,
    /,
    *,
    domain: ArrayLike,
    policy: AdaptiveImplicitSurfacePolicy = _DEFAULT_ADAPTIVE_POLICY,
    source_id: str,
    maximum_root_solves: int | None = None,
) -> AdaptiveImplicitSurface:
    """Discover the zero set of an implicit region by adaptive octree refinement.

    ``domain`` is the ``(2, 3)`` lower/upper box searched; the zero set must not
    reach its boundary. The result's evidence states what was achieved
    (``certified``, ``enclosed``, or ``sampled``), every unresolved leaf with its
    reasons, and the resource use. Budget exhaustion never raises: it stops
    refinement and reports the remaining failures as unresolved boxes. The
    result is not differentiable; derivatives belong to a fixed-topology route
    such as `discover_implicit_surface` plans.
    """
    domain_ = np.asarray(domain, dtype=np.float64)
    if domain_.shape != (2, _DIMENSION):
        raise ValueError("domain must have shape (2, 3).")
    _validate(geometry, domain_, policy, source_id)
    bounds = _field_bounds(geometry, policy.enclosure)
    root_capacity = (
        policy.maximum_evaluations if maximum_root_solves is None else maximum_root_solves
    )
    if (
        isinstance(root_capacity, bool)
        or not isinstance(root_capacity, int)
        or root_capacity < 0
    ):
        raise ValueError("maximum_root_solves must be a nonnegative integer.")
    work = _new_work(bounds, policy, domain_, source_id, root_capacity)
    analysis, status, rounds = _refine(work)
    if work.isolation_exhausted:
        status |= int(AdaptiveImplicitSurfaceStatus.EVALUATION_BUDGET_EXHAUSTED)
    extraction = _extract(work, geometry, analysis)
    status |= extraction.status
    if np.any(analysis.issues != 0):
        status |= int(AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES)
    leaves = analysis.leaves
    leaf_lower = work.coordinates(leaves.corners)
    leaf_upper = work.coordinates(leaves.corners + leaves.sizes[:, None])
    boxes = np.stack((leaf_lower, leaf_upper), axis=1)
    accuracy = _accuracy(bounds, bool(np.any(analysis.issues != 0)), status, extraction)
    unpublishable = int(
        AdaptiveImplicitSurfaceStatus.NO_SURFACE
        | AdaptiveImplicitSurfaceStatus.DOMAIN_BOUNDARY_CROSSING
        | AdaptiveImplicitSurfaceStatus.NONMANIFOLD_EXTRACTION
        | AdaptiveImplicitSurfaceStatus.DEGENERATE_FACE
    )
    # A closed, nondegenerate extraction is published even when it intersects
    # itself; the evidence then withholds certification.
    mesh = (
        None
        if extraction.status & unpublishable
        else TriangleMesh(extraction.vertices, extraction.faces, source_id=source_id)
    )
    cover = _cover(geometry, bounds, analysis.bounds, boxes, domain_, source_id)
    topology = (
        CertifiedImplicitTopology(
            cover, mesh.topology.cell_complex_topology(), premise="directional_monotone"
        )
        if cover is not None and mesh is not None and accuracy == "certified"
        else None
    )
    evidence = _evidence(
        work,
        analysis,
        extraction,
        boxes,
        accuracy=accuracy,
        status=status,
        rounds=rounds,
    )
    volume = ImplicitVolumeQuery(
        bounds,
        leaf_lower,
        leaf_upper,
        analysis.bounds.value_lower,
        analysis.bounds.value_upper,
        leaves.code_starts.astype(np.int64),
        domain_,
        maximum_level=policy.maximum_level,
    )
    feature_edges = (
        extraction.feature_edges if mesh is not None else np.zeros((0, 2), np.int64)
    )
    return AdaptiveImplicitSurface(
        mesh,
        evidence,
        cover,
        topology,
        volume,
        feature_edges,
        _source_cycle_boxes(work, analysis)
        if mesh is not None
        else np.empty((0, 2, 3), dtype=np.float64),
        analysis.faces.macrocharts,
        source_id=source_id,
        geometry=geometry,
        policy=policy,
        root_solve_capacity=root_capacity,
    )


__all__ = [
    "ImplicitNormalFiberMacrochart",
    "AdaptiveImplicitBoundarySource",
    "AdaptiveImplicitSurface",
    "AdaptiveImplicitSurfaceEvidence",
    "ImplicitVolumeClass",
    "ImplicitVolumeClassification",
    "ImplicitVolumeQuery",
    "discover_adaptive_implicit_surface",
]
