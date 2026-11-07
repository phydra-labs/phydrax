# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Immutable evidence for feature-aligned surface quad layouts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from fractions import Fraction
from functools import reduce
from typing import final, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._exact_kernels import float64_interval_elementary
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import SimplicialConnectivity
from ..discretization._coordinate_enclosure import CoordinateSourceBank
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._interval_enclosure import (
    Interval,
    interval_absolute,
    interval_add,
    interval_cos,
    interval_divide,
    interval_minimum,
    interval_multiply,
    interval_sin,
    interval_sqrt,
    interval_square,
    interval_subtract,
)
from ..geometry._rational import (
    fraction_down,
    fraction_up,
    rational_cross,
    rational_dot,
    rational_vector,
    RationalVector,
)
from ..typing import ConvertibleToArray


def _is_triangle_cell_kind(cell_kind: str, /) -> bool:
    topology = reference_cell_topology(cell_kind)
    return topology.family == "triangle" and topology.dimension == 2


@final
class QuadLayoutObstruction(ValueError):
    """A source-identifiable obstruction to a certified quad patch layout."""

    def __init__(
        self,
        category: str,
        message: str,
        /,
        *,
        source_cell_rows: Iterable[int] = (),
        source_edge_rows: Iterable[int] = (),
        source_vertex_rows: Iterable[int] = (),
    ) -> None:
        self.category = str(category)
        self.source_cell_rows = tuple(sorted({int(value) for value in source_cell_rows}))
        self.source_edge_rows = tuple(sorted({int(value) for value in source_edge_rows}))
        self.source_vertex_rows = tuple(
            sorted({int(value) for value in source_vertex_rows})
        )
        super().__init__(str(message))


type _IntervalPair = tuple[Interval, Interval]
type _IntervalVector = tuple[Interval, ...]

# Acceptance constants of a source-bound cross field. Generation
# (``_quad_generation``) declares them and restore reuses them; this module owns
# them because generation imports the field contract from here.
# Relative residual a sparse connection solve must certify.
CROSS_FIELD_RELATIVE_RESIDUAL_LIMIT = 1.0e-10
# Absolute agreement of field evidence with its declared source formula:
# frames, edge angles, feature residuals, and the absolute energy floor.
CROSS_FIELD_EVIDENCE_ATOL = 1.0e-12
# Relative agreement of the field energy with its declared formula.
CROSS_FIELD_ENERGY_RTOL = 1.0e-10
# Outward binary64 enclosure of pi: math.pi < pi < nextafter(math.pi, inf).
_PI: Interval = (np.float64(math.pi), np.float64(np.nextafter(math.pi, math.inf)))
# Reference rays for winding counts; the count is independent of the ray, so
# the first ray whose crossings are all certified decides it exactly.
_WINDING_RAYS = (
    (1, 0),
    (0, 1),
    (1, 1),
    (1, -1),
    (2, 1),
    (1, 2),
    (2, -1),
    (1, -2),
    (3, 1),
    (1, 3),
    (3, -1),
    (1, -3),
)


def _enclose(lower: Fraction, upper: Fraction, /) -> Interval:
    return (np.float64(fraction_down(lower)), np.float64(fraction_up(upper)))


def _enclose_point(value: Fraction, /) -> Interval:
    return _enclose(value, value)


def _enclose_vector(vector: RationalVector, /) -> _IntervalVector:
    return tuple(_enclose_point(value) for value in vector)


def _scaled(value: Interval, factor: float, /) -> Interval:
    return interval_multiply(value, (np.float64(factor), np.float64(factor)))


def _certified_sign(value: Interval, /) -> int | None:
    """Return the sign of every enclosed value, or None if undecided."""
    if float(value[0]) > 0.0:
        return 1
    if float(value[1]) < 0.0:
        return -1
    return None


def _certified_at_most(value: Interval, bound: float, /) -> bool:
    """Decide that every enclosed value is at most ``bound``."""
    return float(value[1]) <= bound


def _certified_within(stored: float, enclosure: Interval, tolerance: float, /) -> bool:
    """Decide ``|stored - x| <= tolerance`` for every ``x`` in the enclosure."""
    value = Fraction(stored)
    return max(
        value - Fraction(float(enclosure[0])), Fraction(float(enclosure[1])) - value
    ) <= Fraction(tolerance)


def _interval_dot(first: _IntervalVector, second: _IntervalVector, /) -> Interval:
    products = [
        interval_multiply(left, right) for left, right in zip(first, second, strict=True)
    ]
    return reduce(interval_add, products)


def _interval_cross(
    first: _IntervalVector, second: _IntervalVector, /
) -> _IntervalVector:
    return (
        interval_subtract(
            interval_multiply(first[1], second[2]), interval_multiply(first[2], second[1])
        ),
        interval_subtract(
            interval_multiply(first[2], second[0]), interval_multiply(first[0], second[2])
        ),
        interval_subtract(
            interval_multiply(first[0], second[1]), interval_multiply(first[1], second[0])
        ),
    )


def _interval_atan(value: Interval, /) -> Interval:
    lower, upper = float64_interval_elementary("atan", value[0], value[1])
    return lower, upper


def _edge_angle_enclosure(local: tuple[Fraction, Fraction], /) -> Interval:
    """Enclose ``atan2(y, x)`` of an exact frame-local edge direction."""
    first, second = local
    if first > 0:
        return _interval_atan(_enclose_point(second / first))
    if first == 0 and second == 0:
        raise ValueError("A cross-field edge direction must be finite and nonzero.")
    half_turn = _scaled(_PI, 0.5)
    if second > 0:
        return interval_subtract(
            half_turn, _interval_atan(_enclose_point(first / second))
        )
    if second < 0:
        return interval_subtract(
            _scaled(half_turn, -1.0), _interval_atan(_enclose_point(first / second))
        )
    return _PI


def _periodic_distance(value: Interval, /) -> Interval:
    """Enclose the distance of every enclosed value to the nearest multiple of 2 pi.

    This is ``|atan2(sin x, cos x)|`` of the generation formulas and the
    distance between two angles that denote the same direction.
    """
    full_turn = _scaled(_PI, 2.0)
    # Candidate turn counts only select terms; the true nearest multiple is
    # always among the three around the midpoint estimate.
    center = np.rint(
        0.5 * (np.asarray(value[0]) + np.asarray(value[1])) / (2.0 * math.pi)
    )
    distances = [
        interval_absolute(
            interval_subtract(value, interval_multiply(full_turn, (turns, turns)))
        )
        for turns in (center - 1.0, center, center + 1.0)
    ]
    return reduce(interval_minimum, distances)


def _rotate_between_normals(
    direction: _IntervalVector,
    face_normal: RationalVector,
    vertex_normal: RationalVector,
    vertex_normal_square: Fraction,
    vertex: int,
    /,
) -> _IntervalVector:
    """Rotate ``direction`` by the minimal rotation taking one normal to the other.

    Returns a positive multiple of the Rodrigues image: with ``q = a.b``,
    ``K = a x b`` and ``m = |a||b|``, ``m (m + q) R d`` equals
    ``(m + q)(q d + K x d) + K (K.d)``, which avoids normalization.
    """
    axis = rational_cross(face_normal, vertex_normal)
    alignment = rational_dot(face_normal, vertex_normal)
    if not any(axis):
        if alignment < 0:
            raise QuadLayoutObstruction(
                "singularity_transport",
                "Cross-field index transport is ambiguous across opposed source normals.",
                source_vertex_rows=(vertex,),
            )
        return direction
    length_square = rational_dot(face_normal, face_normal) * vertex_normal_square
    shifted = interval_add(
        interval_sqrt(_enclose(length_square, length_square)),
        _enclose(alignment, alignment),
    )
    if _certified_sign(shifted) != 1:
        raise QuadLayoutObstruction(
            "singularity_transport",
            "Cross-field index transport across nearly opposed source normals is not certified.",
            source_vertex_rows=(vertex,),
        )
    enclosed_axis = _enclose_vector(axis)
    enclosed_alignment = _enclose(alignment, alignment)
    swept = _interval_cross(enclosed_axis, direction)
    along = _interval_dot(enclosed_axis, direction)
    return tuple(
        interval_add(
            interval_multiply(
                shifted,
                interval_add(
                    interval_multiply(component, enclosed_alignment), swept_component
                ),
            ),
            interval_multiply(axis_component, along),
        )
        for component, swept_component, axis_component in zip(
            direction, swept, enclosed_axis, strict=True
        )
    )


def _certified_winding(points: Sequence[_IntervalPair], vertex: int, /) -> int:
    """Count the exact winding of a closed polygon of enclosed points about zero.

    Consecutive points must not be antipodal, so each straight edge sweeps the
    principal argument between its ends. A crossing of the reference ray is
    decided only from certified signs.
    """
    count = len(points)
    for first, second in _WINDING_RAYS:
        rotated = [
            (
                interval_add(_scaled(real, first), _scaled(imaginary, second)),
                interval_subtract(_scaled(imaginary, first), _scaled(real, second)),
            )
            for real, imaginary in points
        ]
        sides = [_certified_sign(imaginary) for _, imaginary in rotated]
        if any(side is None for side in sides):
            continue
        winding = 0
        certified = True
        for index, current in enumerate(rotated):
            following = rotated[(index + 1) % count]
            start, end = sides[index], sides[(index + 1) % count]
            if start == end:
                continue
            turn = _certified_sign(
                interval_subtract(
                    interval_multiply(current[0], following[1]),
                    interval_multiply(current[1], following[0]),
                )
            )
            if turn is None:
                certified = False
                break
            if start == -1 and turn == 1:
                winding += 1
            elif start == 1 and turn == -1:
                winding -= 1
        if certified:
            return winding
    raise QuadLayoutObstruction(
        "singularity_index",
        "The cross-field vertex circulation cannot be certified as a fourfold index.",
        source_vertex_rows=(vertex,),
    )


@final
class SurfaceCrossField(StrictModule, NonTrainableState):
    """A sparse-solved fourfold tangent field bound to one triangle surface."""

    source: CellMesh
    frames: Array
    angles: Array
    transport_edges: Array
    patch_ids: Array
    feature_edges: Array
    transport_angles: Array
    feature_rows: Array
    feature_angles: Array
    guide_angles: Array
    feature_residuals: Array
    field_solve_residual: Array
    field_solve_relative_residual: Array
    field_energy: Array
    field_solver: str = eqx.field(static=True)
    singular_vertices: Array
    singularity_indices: Array
    source_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        frames: ConvertibleToArray,
        angles: ConvertibleToArray,
        transport_edges: ConvertibleToArray,
        patch_ids: ConvertibleToArray,
        feature_edges: ConvertibleToArray,
        transport_angles: ConvertibleToArray,
        feature_rows: ConvertibleToArray,
        feature_angles: ConvertibleToArray,
        guide_angles: ConvertibleToArray,
        feature_residuals: ConvertibleToArray,
        field_solve_residual: ConvertibleToArray,
        field_solve_relative_residual: ConvertibleToArray,
        field_energy: ConvertibleToArray,
        singular_vertices: ConvertibleToArray,
        singularity_indices: ConvertibleToArray,
        source: CellMesh,
        /,
    ) -> None:
        frames_ = np.asarray(frames, dtype=np.float64)
        angles_ = np.asarray(angles, dtype=np.float64)
        links_ = np.asarray(transport_edges, dtype=np.int32)
        patches_ = np.asarray(patch_ids, dtype=np.int32)
        edges_ = np.asarray(feature_edges, dtype=np.int64)
        transport_ = np.asarray(transport_angles, dtype=np.float64)
        rows_ = np.asarray(feature_rows, dtype=np.int32)
        directions_ = np.asarray(feature_angles, dtype=np.float64)
        guide_angles_ = np.asarray(guide_angles, dtype=np.float64)
        residuals_ = np.asarray(feature_residuals, dtype=np.float64)
        solve_residual_ = np.asarray(field_solve_residual, dtype=np.float64)
        relative_residual_ = np.asarray(field_solve_relative_residual, dtype=np.float64)
        energy_ = np.asarray(field_energy, dtype=np.float64)
        singular_vertices_ = np.asarray(singular_vertices, dtype=np.int64)
        singularity_indices_ = np.asarray(singularity_indices, dtype=np.int32)
        if len(source.blocks) != 1:
            raise TypeError("Cross-field evidence requires one triangle block.")
        block = source.blocks[0]
        if not _is_triangle_cell_kind(block.cell_kind):
            raise TypeError("Cross-field evidence requires a triangle reference cell.")
        connectivity = source.connectivity
        if isinstance(connectivity, PolygonalConnectivity):
            edge_count = connectivity.edges.shape[0]
        elif (
            isinstance(connectivity, SimplicialConnectivity)
            and connectivity.dimension == 2
        ):
            edge_count = connectivity.entities[1].shape[0]
        else:
            raise TypeError(
                "Cross-field evidence requires two-dimensional edge connectivity."
            )
        face_count = block.cell_count
        vertex_count = source.coordinates.shape[0]
        if frames_.shape != (face_count, 2, 3):
            raise ValueError("Cross-field frames must have shape (triangles, 2, 3).")
        if (
            angles_.shape != (face_count,)
            or patches_.shape != (face_count,)
            or np.any(patches_ < 0)
        ):
            raise ValueError(
                "Cross-field angles and non-negative patches must cover every triangle."
            )
        if guide_angles_.shape not in ((0,), (face_count,)):
            raise ValueError(
                "Cross-field guidance must be empty or cover every triangle."
            )
        if links_.ndim != 2 or links_.shape[1] != 2:
            raise ValueError("Cross-field transport edges must have shape (links, 2).")
        if transport_.shape != (links_.shape[0],):
            raise ValueError("Cross-field transport angles must match transport edges.")
        if (
            rows_.ndim != 1
            or directions_.shape != rows_.shape
            or residuals_.shape != rows_.shape
        ):
            raise ValueError("Cross-field feature evidence must use aligned vectors.")
        if edges_.ndim != 1 or np.any(edges_ < 0) or np.any(edges_[1:] <= edges_[:-1]):
            raise ValueError(
                "Cross-field feature edges must be sorted unique non-negative edge rows."
            )
        if (
            singular_vertices_.ndim != 1
            or singularity_indices_.shape != singular_vertices_.shape
            or np.any(singular_vertices_ < 0)
            or np.any(singular_vertices_ >= vertex_count)
            or np.any(singularity_indices_ == 0)
            or len(set(singular_vertices_.tolist())) != singular_vertices_.size
        ):
            raise ValueError(
                "Cross-field singularity evidence must identify unique nonzero vertex indices."
            )
        if any(
            value.shape != () for value in (solve_residual_, relative_residual_, energy_)
        ):
            raise ValueError("Cross-field solve residuals and energy must be scalars.")
        if (
            not all(
                np.all(np.isfinite(value))
                for value in (
                    frames_,
                    angles_,
                    transport_,
                    directions_,
                    guide_angles_,
                    residuals_,
                    solve_residual_,
                    relative_residual_,
                    energy_,
                )
            )
            or float(solve_residual_) < 0.0
            or float(relative_residual_) < 0.0
            or float(energy_) < 0.0
            or float(relative_residual_) > CROSS_FIELD_RELATIVE_RESIDUAL_LIMIT
        ):
            raise ValueError(
                "Cross-field numerical evidence must be finite, non-negative, and residual-certified."
            )
        if links_.size and (np.any(links_ < 0) or np.any(links_ >= face_count)):
            raise ValueError("Cross-field transport edges index declared triangles.")
        if rows_.size and (np.any(rows_ < 0) or np.any(rows_ >= face_count)):
            raise ValueError("Cross-field feature rows index declared triangles.")
        if edges_.size and np.any(edges_ >= edge_count):
            raise ValueError("Cross-field feature edges index declared source edges.")
        self.frames = jnp.asarray(frames_)
        self.angles = jnp.asarray(angles_)
        self.transport_edges = jnp.asarray(links_)
        self.patch_ids = jnp.asarray(patches_)
        self.feature_edges = jnp.asarray(edges_)
        self.transport_angles = jnp.asarray(transport_)
        self.feature_rows = jnp.asarray(rows_)
        self.feature_angles = jnp.asarray(directions_)
        self.guide_angles = jnp.asarray(guide_angles_)
        self.feature_residuals = jnp.asarray(residuals_)
        self.field_solve_residual = jnp.asarray(solve_residual_)
        self.field_solve_relative_residual = jnp.asarray(relative_residual_)
        self.field_energy = jnp.asarray(energy_)
        self.singular_vertices = jnp.asarray(singular_vertices_)
        self.singularity_indices = jnp.asarray(singularity_indices_)
        self.source = source
        self.field_solver = "scipy.sparse.linalg.spsolve"
        self.source_topology_id = source.topology_id
        self.source_geometry_id = source.geometry_id
        self.field_id = self._scientific_id()

    def _scientific_id(self, /) -> str:
        return canonical_fingerprint(
            {
                "kind": "source-surface-cross-field",
                "source_mesh": self.source.mesh_id,
                "source_topology": self.source_topology_id,
                "source_geometry": self.source_geometry_id,
                "angles": np.asarray(self.angles),
                "features": np.asarray(self.feature_edges),
                "guide_angles": np.asarray(self.guide_angles),
                "singular_vertices": np.asarray(self.singular_vertices),
                "singularity_indices": np.asarray(self.singularity_indices),
            }
        )

    @classmethod
    def from_stored_fields(
        cls: type[SurfaceCrossField], value: SurfaceCrossField, /
    ) -> SurfaceCrossField:
        """Certify the stored field against its restored source without re-solving.

        The sparse connection solve is deterministic per platform but not
        bit-identical across platforms, so restore never repeats it. Identity,
        the source edge graph, feature rows, patch labels, frames, and the
        canonical angle range are decided exactly; edge directions, feature
        residuals, and field energy are certified against the stored angles
        with rational enclosures; singularity indices are recomputed from the
        stored frames and angles with certified sign decisions and must match.
        Solve residuals remain validated stored evidence.
        """
        if type(value) is not cls:
            raise TypeError("value must be SurfaceCrossField.")
        source = CellMesh.from_stored_fields(value.source)
        rebuilt = cls(
            value.frames,
            value.angles,
            value.transport_edges,
            value.patch_ids,
            value.feature_edges,
            value.transport_angles,
            value.feature_rows,
            value.feature_angles,
            value.guide_angles,
            value.feature_residuals,
            value.field_solve_residual,
            value.field_solve_relative_residual,
            value.field_energy,
            value.singular_vertices,
            value.singularity_indices,
            source,
        )
        if (
            rebuilt.field_id != value.field_id
            or rebuilt.source_topology_id != value.source_topology_id
            or rebuilt.source_geometry_id != value.source_geometry_id
            or rebuilt.field_solver != value.field_solver
        ):
            raise ValueError(
                "Stored source-bound cross field does not reproduce its identity."
            )
        _certify_stored_cross_field(rebuilt)
        return rebuilt


def _enclose_all(values: Sequence[Fraction], /) -> Interval:
    """Enclose rationals as one vector interval of outward binary64 endpoints."""
    return (
        np.asarray([fraction_down(value) for value in values], dtype=np.float64),
        np.asarray([fraction_up(value) for value in values], dtype=np.float64),
    )


def _row(value: Interval, index: int, /) -> Interval:
    return np.asarray(value[0])[index], np.asarray(value[1])[index]


def _certify_stored_cross_field(field: SurfaceCrossField, /) -> None:
    """Decide stored field evidence against generation's declared acceptance.

    Every gate reuses the constant with which generation accepts a field and
    is decided exactly on the stored arrays: exact rationals and outward
    binary64 intervals with the rigorous native elementary kernels. The solve
    is never repeated; angles and solve residuals beyond their generation
    limits are stored evidence bound by the field identity and the archive.
    """
    from ._quad_generation import _edge_key, _surface_patch_ids, _triangle_data

    source = field.source
    data = _triangle_data(source)
    face_count = data.cells.shape[0]
    frames = np.asarray(field.frames, dtype=np.float64)
    angles = np.asarray(field.angles, dtype=np.float64)
    guidance = np.asarray(field.guide_angles, dtype=np.float64)
    points = np.asarray(source.coordinates, dtype=np.float64)
    padded = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    exact_points = [rational_vector(row) for row in padded.tolist()]
    exact_frames = [
        (rational_vector(face[0]), rational_vector(face[1])) for face in frames.tolist()
    ]
    # Frames: the exact unit tangent e1/|e1| and in-plane normal
    # (n x e1)/(|n||e1|) of each source triangle.
    numerators: list[Fraction] = []
    denominators: list[Fraction] = []
    for triangle in data.cells.tolist():
        origin = exact_points[triangle[0]]
        first = tuple(
            b - a for a, b in zip(origin, exact_points[triangle[1]], strict=True)
        )
        second = tuple(
            b - a for a, b in zip(origin, exact_points[triangle[2]], strict=True)
        )
        normal = rational_cross(first, second)
        first_square = rational_dot(first, first)
        normal_square = rational_dot(normal, normal)
        if first_square == 0 or normal_square == 0:
            raise ValueError("Cross-field frames require nondegenerate source triangles.")
        numerators.extend((*first, *rational_cross(normal, first)))
        denominators.extend((first_square,) * 3 + (normal_square * first_square,) * 3)
    exact_frame_values = interval_divide(
        _enclose_all(numerators), interval_sqrt(_enclose_all(denominators))
    )
    if not all(
        _certified_within(
            stored, _row(exact_frame_values, index), CROSS_FIELD_EVIDENCE_ATOL
        )
        for index, stored in enumerate(frames.reshape(-1).tolist())
    ):
        raise ValueError(
            "Stored source-bound cross field does not reproduce its source frames."
        )
    features = {int(row) for row in np.asarray(field.feature_edges).tolist()}
    links: list[tuple[int, int]] = []
    link_directions: list[tuple[Interval, Interval]] = []
    feature_rows: list[int] = []
    feature_directions: list[Interval] = []
    for edge_row, edge_values in enumerate(data.edges.tolist()):
        key = _edge_key(int(edge_values[0]), int(edge_values[1]))
        faces = data.edge_owners[key]
        direction = tuple(
            end - start
            for start, end in zip(exact_points[key[0]], exact_points[key[1]], strict=True)
        )
        local = [
            _edge_angle_enclosure(
                (
                    rational_dot(exact_frames[face][0], direction),
                    rational_dot(exact_frames[face][1], direction),
                )
            )
            for face in faces
        ]
        if edge_row in features:
            feature_rows.extend(faces)
            feature_directions.extend(local)
        elif len(faces) == 2:
            links.append((faces[0], faces[1]))
            link_directions.append((local[0], local[1]))
        else:
            raise ValueError(
                "Stored source-bound cross field does not retain every boundary edge as a feature."
            )
    link_array = np.asarray(links, dtype=np.int32).reshape((-1, 2))
    if (
        not np.array_equal(np.asarray(field.transport_edges), link_array)
        or not np.array_equal(
            np.asarray(field.feature_rows), np.asarray(feature_rows, dtype=np.int32)
        )
        or not np.array_equal(
            np.asarray(field.patch_ids), _surface_patch_ids(face_count, link_array)
        )
    ):
        raise ValueError(
            "Stored source-bound cross field does not reproduce its source edge graph."
        )
    # Edge angles are fixed source geometry, never solver unknowns; they denote
    # directions, so agreement is measured modulo a full turn.
    feature_angles = np.asarray(field.feature_angles, dtype=np.float64).tolist()
    transport_angles = np.asarray(field.transport_angles, dtype=np.float64).tolist()
    if not all(
        _certified_at_most(
            _periodic_distance(interval_subtract((stored, stored), exact)),
            CROSS_FIELD_EVIDENCE_ATOL,
        )
        for stored, exact in zip(
            np.asarray(feature_angles, dtype=np.float64),
            feature_directions,
            strict=True,
        )
    ) or not all(
        _certified_at_most(
            _periodic_distance(
                interval_subtract((stored, stored), interval_subtract(second, first))
            ),
            CROSS_FIELD_EVIDENCE_ATOL,
        )
        for stored, (first, second) in zip(
            np.asarray(transport_angles, dtype=np.float64), link_directions, strict=True
        )
    ):
        raise ValueError(
            "Stored source-bound cross field does not reproduce its source edge directions."
        )
    exact_angles = [Fraction(value) for value in angles.tolist()]
    feature_arguments = [
        4 * (exact_angles[row] - Fraction(angle))
        for row, angle in zip(feature_rows, feature_angles, strict=True)
    ]
    # Feature residual |atan2(sin 4d, cos 4d)| / 4 of the stored angles.
    residuals = _scaled(_periodic_distance(_enclose_all(feature_arguments)), 0.25)
    if not all(
        _certified_within(stored, _row(residuals, index), CROSS_FIELD_EVIDENCE_ATOL)
        for index, stored in enumerate(
            np.asarray(field.feature_residuals, dtype=np.float64).tolist()
        )
    ):
        raise ValueError(
            "Stored source-bound cross field does not reproduce its feature-alignment residuals."
        )
    # Energy of the stored angles: link mismatches, and feature plus guidance
    # rows weighted four, exactly as the solve assembles them.
    link_arguments = [
        4 * (exact_angles[second] - exact_angles[first] - Fraction(transport))
        for (first, second), transport in zip(links, transport_angles, strict=True)
    ]
    guidance_arguments = [
        4 * (exact_angles[face] - Fraction(value))
        for face, value in enumerate(guidance.tolist())
    ]
    weights = np.asarray(
        [1.0] * len(link_arguments)
        + [4.0] * (len(feature_arguments) + len(guidance_arguments)),
        dtype=np.float64,
    )
    arguments = link_arguments + feature_arguments + guidance_arguments
    energy: Interval = (np.float64(0.0), np.float64(0.0))
    if arguments:
        terms = interval_multiply(
            (weights, weights),
            interval_subtract(
                (np.float64(1.0), np.float64(1.0)), interval_cos(_enclose_all(arguments))
            ),
        )
        energy = reduce(
            interval_add, (_row(terms, index) for index in range(len(arguments)))
        )
    stored_energy = Fraction(float(field.field_energy))
    energy_lower = Fraction(float(energy[0]))
    energy_upper = Fraction(float(energy[1]))
    smallest = min(abs(energy_lower), abs(energy_upper))
    if energy_lower <= 0 <= energy_upper:
        smallest = Fraction(0)
    # math.isclose(stored, exact, rel_tol, abs_tol) for every enclosed exact value.
    if max(stored_energy - energy_lower, energy_upper - stored_energy) > max(
        Fraction(CROSS_FIELD_ENERGY_RTOL) * max(abs(stored_energy), smallest),
        Fraction(CROSS_FIELD_EVIDENCE_ATOL),
    ):
        raise ValueError(
            "Stored source-bound cross field does not reproduce its field energy."
        )
    angle_cosines = interval_cos((angles, angles))
    angle_sines = interval_sin((angles, angles))
    singular_vertices, singularity_indices = _certified_singularity_indices(
        data.cells.tolist(),
        data.edges,
        data.edge_owners,
        exact_points,
        exact_frames,
        [
            (_row(angle_cosines, face), _row(angle_sines, face))
            for face in range(face_count)
        ],
        features,
    )
    if not np.array_equal(
        np.asarray(field.singular_vertices), np.asarray(singular_vertices, dtype=np.int64)
    ) or not np.array_equal(
        np.asarray(field.singularity_indices),
        np.asarray(singularity_indices, dtype=np.int32),
    ):
        raise ValueError(
            "Stored source-bound cross field does not reproduce its certified singularity indices."
        )


def _certified_singularity_indices(
    cells: Sequence[Sequence[int]],
    edges: np.ndarray,
    edge_owners: dict[tuple[int, int], tuple[int, ...]],
    points: Sequence[RationalVector],
    frames: Sequence[tuple[RationalVector, RationalVector]],
    angles: Sequence[_IntervalPair],
    feature_edges: set[int],
    /,
) -> tuple[list[int], list[int]]:
    """Decide interior fourfold vertex indices exactly from stored frames and angles.

    Each face direction is transported into its vertex tangent plane by the
    minimal rotation between the exact face normal and the exact area-weighted
    vertex normal. The index is the exact winding number of the fourth powers
    of those directions around the vertex star.
    """
    normals: list[RationalVector] = []
    for triangle in cells:
        origin = points[triangle[0]]
        first = tuple(b - a for a, b in zip(origin, points[triangle[1]], strict=True))
        second = tuple(b - a for a, b in zip(origin, points[triangle[2]], strict=True))
        normal = rational_cross(first, second)
        if not any(normal):
            raise QuadLayoutObstruction(
                "singularity_transport",
                "Cross-field vertex indices require nondegenerate oriented source triangles.",
            )
        normals.append(normal)
    vertex_faces: list[list[int]] = [[] for _ in range(len(points))]
    for face, triangle in enumerate(cells):
        for vertex in triangle:
            vertex_faces[vertex].append(face)
    boundary_vertices = {
        vertex
        for key, owners in edge_owners.items()
        if len(owners) == 1
        for vertex in key
    }
    feature_vertices = {
        int(vertex) for row in feature_edges for vertex in edges[row].tolist()
    }
    indexed_vertices: list[int] = []
    indices: list[int] = []
    for vertex, star in enumerate(vertex_faces):
        if vertex in boundary_vertices or vertex in feature_vertices or not star:
            continue
        next_face: dict[int, int] = {}
        for face in star:
            triangle = list(cells[face])
            following = triangle[(triangle.index(vertex) + 2) % 3]
            incident = edge_owners[(min(vertex, following), max(vertex, following))]
            if len(incident) != 2:
                raise QuadLayoutObstruction(
                    "vertex_link",
                    "An interior cross-field vertex link is not a closed manifold cycle.",
                    source_vertex_rows=(vertex,),
                )
            next_face[face] = incident[0] if incident[1] == face else incident[1]
        start = min(star)
        order: list[int] = []
        current = start
        while current not in order:
            order.append(current)
            current = next_face.get(current, -1)
            if current < 0:
                break
        if current != start or len(order) != len(star):
            raise QuadLayoutObstruction(
                "vertex_link",
                "An interior cross-field vertex link is disconnected or inconsistently oriented.",
                source_vertex_rows=(vertex,),
            )
        vertex_normal = tuple(
            sum((normals[face][axis] for face in star), Fraction(0)) for axis in range(3)
        )
        normal_square = rational_dot(vertex_normal, vertex_normal)
        if normal_square == 0:
            raise QuadLayoutObstruction(
                "singularity_transport",
                "A source vertex has no stable tangent plane for cross-field index transport.",
                source_vertex_rows=(vertex,),
            )
        tangent: RationalVector | None = None
        for face in order:
            for other in cells[face]:
                if other == vertex:
                    continue
                edge = tuple(
                    b - a for a, b in zip(points[vertex], points[other], strict=True)
                )
                height = rational_dot(edge, vertex_normal)
                projected = tuple(
                    normal_square * component - height * normal_component
                    for component, normal_component in zip(
                        edge, vertex_normal, strict=True
                    )
                )
                if any(projected):
                    tangent = projected
                    break
            if tangent is not None:
                break
        if tangent is None:
            raise QuadLayoutObstruction(
                "singularity_transport",
                "A source vertex has no stable local tangent direction.",
                source_vertex_rows=(vertex,),
            )
        # The second axis has length |n| |t|; scaling the first coordinate by
        # |n| keeps both coordinates proportional to the unit-frame phase.
        enclosed_tangent = _enclose_vector(tangent)
        enclosed_second = _enclose_vector(rational_cross(vertex_normal, tangent))
        normal_length = interval_sqrt(_enclose(normal_square, normal_square))
        fourth_powers: list[_IntervalPair] = []
        for face in order:
            cosine, sine = angles[face]
            direction = tuple(
                interval_add(
                    interval_multiply(cosine, _enclose(first, first)),
                    interval_multiply(sine, _enclose(second, second)),
                )
                for first, second in zip(frames[face][0], frames[face][1], strict=True)
            )
            transported = _rotate_between_normals(
                direction, normals[face], vertex_normal, normal_square, vertex
            )
            real = interval_multiply(
                normal_length, _interval_dot(transported, enclosed_tangent)
            )
            imaginary = _interval_dot(transported, enclosed_second)
            real_square = interval_square(real)
            imaginary_square = interval_square(imaginary)
            fourth_powers.append(
                (
                    interval_add(
                        interval_subtract(
                            interval_square(real_square),
                            _scaled(interval_multiply(real_square, imaginary_square), 6),
                        ),
                        interval_square(imaginary_square),
                    ),
                    _scaled(
                        interval_multiply(
                            interval_multiply(real, imaginary),
                            interval_subtract(real_square, imaginary_square),
                        ),
                        4,
                    ),
                )
            )
        charge = _certified_winding(fourth_powers, vertex)
        if charge != 0:
            indexed_vertices.append(vertex)
            indices.append(charge)
    return indexed_vertices, indices


@final
class SurfaceQuadLayout(StrictModule, NonTrainableState):
    """Feature-guided triangle connectivity and its source-cell ancestry."""

    mesh: CellMesh
    parent_offsets: Array
    parent_cells: Array
    paired_edge_rows: Array
    source_topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        parent_offsets: ConvertibleToArray,
        parent_cells: ConvertibleToArray,
        paired_edge_rows: ConvertibleToArray,
        source_topology_id: str,
        layout_id: str,
        /,
    ) -> None:
        if mesh.topological_dimension != 2 or len(mesh.blocks) != 1:
            raise ValueError("A surface quad layout must retain one triangle block.")
        block = mesh.blocks[0]
        if not _is_triangle_cell_kind(block.cell_kind):
            raise ValueError(
                "A surface quad layout must retain triangle reference cells."
            )
        connectivity = mesh.connectivity
        if not (
            isinstance(connectivity, PolygonalConnectivity)
            or (
                isinstance(connectivity, SimplicialConnectivity)
                and connectivity.dimension == 2
            )
        ):
            raise TypeError(
                "A surface quad layout requires two-dimensional edge connectivity."
            )
        offsets = np.asarray(parent_offsets, dtype=np.int64)
        parents = np.asarray(parent_cells, dtype=np.int64)
        paired = np.asarray(paired_edge_rows, dtype=np.int64)
        cell_count = mesh.blocks[0].cell_count
        if (
            offsets.shape != (cell_count + 1,)
            or offsets[0] != 0
            or offsets[-1] != parents.size
            or np.any(offsets[1:] <= offsets[:-1])
            or np.any(parents < 0)
        ):
            raise ValueError(
                "Quad-layout ancestry must name one or more source parents per triangle."
            )
        if paired.ndim != 1 or np.any(paired < 0):
            raise ValueError("Paired source-edge rows must be non-negative.")
        if not source_topology_id or not layout_id:
            raise ValueError("Quad-layout identities must be non-empty.")
        self.mesh = mesh
        self.parent_offsets = jnp.asarray(offsets)
        self.parent_cells = jnp.asarray(parents)
        self.paired_edge_rows = jnp.asarray(paired)
        self.source_topology_id = str(source_topology_id)
        self.layout_id = str(layout_id)


class SurfaceQuadFieldArchiveRecipe(NamedTuple):
    """Source and solver inputs needed to reconstruct one field-layout event."""

    source: CellMesh
    source_geometry: CellGeometrySpec
    field: SurfaceCrossField
    paired_edge_rows: Array
    maximum_field_error: float
    complete: bool
    field_id: str
    layout_id: str
    event_id: str


@final
class SurfaceQuadFieldEvent(StrictModule, NonTrainableState):
    """A revalidated source-bound cross field and quad-layout decision."""

    source: CellMesh
    source_geometry: CellGeometrySpec
    field: SurfaceCrossField
    layout: SurfaceQuadLayout
    maximum_field_error: float = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    event_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CellMesh,
        source_geometry: CellGeometrySpec,
        field: SurfaceCrossField,
        layout: SurfaceQuadLayout,
        /,
        *,
        maximum_field_error: float,
        complete: bool,
    ) -> None:
        from ..geometry._mesh_certificates import _same_tree_content
        from ._quad_generation import verify_surface_quad_layout

        if type(complete) is not bool:
            raise TypeError("complete must be bool.")
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("source_geometry must be CellGeometrySpec.")
        validity = certify_cell_geometry_validity(source_geometry, mesh=source)
        if validity.invalid_count or validity.unresolved_count:
            raise ValueError("A quad field event requires certified source geometry.")
        rebuilt = verify_surface_quad_layout(
            source,
            field,
            layout.paired_edge_rows,
            maximum_field_error=maximum_field_error,
            source_geometry=source_geometry,
            complete=complete,
        )
        if (
            layout.source_topology_id != source.topology_id
            or layout.mesh.topology_id != source.topology_id
            or layout.mesh.geometry_id != source.geometry_id
            or not _same_tree_content(layout.mesh, source)
            or layout.layout_id != rebuilt.layout_id
            or not np.array_equal(layout.parent_offsets, rebuilt.parent_offsets)
            or not np.array_equal(layout.parent_cells, rebuilt.parent_cells)
            or not np.array_equal(layout.paired_edge_rows, rebuilt.paired_edge_rows)
        ):
            raise ValueError(
                "Quad field event layout does not reproduce from its inputs."
            )
        self.source = source
        self.source_geometry = source_geometry
        self.field = field
        self.layout = rebuilt
        self.maximum_field_error = float(maximum_field_error)
        self.complete = complete
        self.event_id = canonical_fingerprint(
            {
                "kind": "surface-quad-field-event",
                "source": source.mesh_id,
                "source_geometry": cell_geometry_id(source_geometry),
                "field": field.field_id,
                "layout": rebuilt.layout_id,
                "maximum_field_error": self.maximum_field_error,
                "complete": complete,
            }
        )

    def archive_recipe(self, /) -> SurfaceQuadFieldArchiveRecipe:
        checked = SurfaceQuadFieldEvent(
            self.source,
            self.source_geometry,
            self.field,
            self.layout,
            maximum_field_error=self.maximum_field_error,
            complete=self.complete,
        )
        if checked.event_id != self.event_id:
            raise ValueError(
                "Quad field event no longer matches its scientific identity."
            )
        return SurfaceQuadFieldArchiveRecipe(
            self.source,
            self.source_geometry,
            self.field,
            self.layout.paired_edge_rows,
            self.maximum_field_error,
            self.complete,
            self.field.field_id,
            self.layout.layout_id,
            self.event_id,
        )

    def require_target(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        """Reconstruct this decision's actual target and coordinate map."""
        from ..geometry._mesh_certificates import _same_tree_content
        from ._quad_generation import _build_quad_mesh, _compose_quad_source_geometry

        self.archive_recipe()
        target, supports, offsets, parents, *_ = _build_quad_mesh(
            self.source, self.layout
        )
        if _affine_geometry_matches(self.source, self.source_geometry):
            target_geometry = CellGeometrySpec.affine(target)
        else:
            target, target_geometry = _compose_quad_source_geometry(
                self.source,
                target,
                supports,
                offsets,
                parents,
                self.source_geometry,
            )
        if (
            target.mesh_id != mesh.mesh_id
            or cell_geometry_id(target_geometry) != cell_geometry_id(geometry)
            or not _same_tree_content(target, mesh)
            or not _same_tree_content(target_geometry, geometry)
        ):
            raise ValueError(
                "Quad field event does not reproduce the target mesh and geometry."
            )


def restore_surface_quad_field_event(
    recipe: SurfaceQuadFieldArchiveRecipe,
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    /,
) -> SurfaceQuadFieldEvent:
    """Certify the stored field and decide its stored layout; never solve or restore by ID."""
    from ..geometry._mesh_certificates import _same_tree_content

    if type(recipe) is not SurfaceQuadFieldArchiveRecipe:
        raise TypeError("recipe must be SurfaceQuadFieldArchiveRecipe.")
    field = SurfaceCrossField.from_stored_fields(recipe.field)
    if (
        field.source.mesh_id != source.mesh_id
        or not _same_tree_content(field.source, source)
        or field.field_id != recipe.field_id
    ):
        raise ValueError("Stored source-bound cross field does not reproduce.")
    cell_count = source.blocks[0].cell_count
    layout = SurfaceQuadLayout(
        source,
        np.arange(cell_count + 1, dtype=np.int64),
        np.arange(cell_count, dtype=np.int64),
        recipe.paired_edge_rows,
        source.topology_id,
        recipe.layout_id,
    )
    # The event decides the stored pairing exactly and requires its identity.
    event = SurfaceQuadFieldEvent(
        source,
        source_geometry,
        field,
        layout,
        maximum_field_error=recipe.maximum_field_error,
        complete=recipe.complete,
    )
    if event.event_id != recipe.event_id:
        raise ValueError("Stored source-bound quad field event does not reproduce.")
    return event


@final
class DualExtraction(StrictModule, NonTrainableState):
    """Locally valid surface-quad or volume-hex dual candidate with source support."""

    mesh: CellMesh
    source: CellMesh
    vertex_supports: Array
    parent_offsets: Array
    parent_cells: Array
    validity: CellValidityCertificate
    valences: Array
    boundary_vertices: Array
    singular_vertices: Array
    cross_field: SurfaceCrossField | None
    field_event: SurfaceQuadFieldEvent | None
    geometry: CellGeometrySpec | None
    source_geometry: CellGeometrySpec | None
    source_edge_ids: Array
    source_edge_parameter_numerators: Array
    source_edge_parameter_denominators: Array

    def __init__(
        self,
        mesh: CellMesh,
        source: CellMesh,
        vertex_supports: ConvertibleToArray,
        parent_offsets: ConvertibleToArray,
        parent_cells: ConvertibleToArray,
        validity: CellValidityCertificate,
        valences: ConvertibleToArray,
        boundary_vertices: ConvertibleToArray,
        singular_vertices: ConvertibleToArray,
        cross_field: SurfaceCrossField | None,
        geometry: CellGeometrySpec | None,
        source_geometry: CellGeometrySpec | None,
        /,
        *,
        source_edge_ids: ConvertibleToArray | None = None,
        source_edge_parameter_numerators: ConvertibleToArray | None = None,
        source_edge_parameter_denominators: ConvertibleToArray | None = None,
        field_event: SurfaceQuadFieldEvent | None = None,
    ) -> None:
        supports = np.asarray(vertex_supports, dtype=np.int64)
        offsets = np.asarray(parent_offsets, dtype=np.int64)
        parents = np.asarray(parent_cells, dtype=np.int64)
        valence = np.asarray(valences, dtype=np.int64)
        boundary = np.asarray(boundary_vertices, dtype=np.bool_)
        singular = np.asarray(singular_vertices, dtype=np.int64)
        cell_count = mesh.connectivity.cell_count
        target_kinds = {block.cell_kind for block in mesh.blocks}
        source_kinds = {block.cell_kind for block in source.blocks}
        target_families = {reference_cell_topology(kind).family for kind in target_kinds}
        source_families = {reference_cell_topology(kind).family for kind in source_kinds}
        supported_pair = (
            source.topological_dimension == 2
            and mesh.topological_dimension == 2
            and target_families == {"quadrilateral"}
            and source_families == {"triangle"}
        ) or (
            source.topological_dimension == 3
            and mesh.topological_dimension == 3
            and target_families == {"hexahedron"}
            and source_families == {"tetrahedron"}
        )
        if not supported_pair:
            raise ValueError(
                "Dual extraction must map triangle-to-quad or tetrahedron-to-hex families."
            )
        if (
            supports.ndim != 2
            or supports.shape[0] != mesh.coordinates.shape[0]
            or supports.shape[1] != max(source.topological_dimension + 1, 4)
            or np.any(supports < -1)
            or np.any(supports >= source.coordinates.shape[0])
            or np.any(np.all(supports < 0, axis=1))
        ):
            raise ValueError("Dual vertex supports must name padded source-vertex rows.")
        target_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
        edge_ancestry_values = (
            source_edge_ids,
            source_edge_parameter_numerators,
            source_edge_parameter_denominators,
        )
        if all(value is None for value in edge_ancestry_values):
            edge_source_ids = np.full(target_edge_ids.shape, -1, dtype=np.int64)
            edge_parameter_numerators = np.zeros(
                (target_edge_ids.size, 2), dtype=np.int64
            )
            edge_parameter_denominators = np.ones(
                (target_edge_ids.size, 2), dtype=np.int64
            )
        elif any(value is None for value in edge_ancestry_values):
            raise ValueError(
                "Dual source-edge ancestry arrays must be supplied together."
            )
        else:
            raw_edge_ids = np.asarray(source_edge_ids)
            raw_numerators = np.asarray(source_edge_parameter_numerators)
            raw_denominators = np.asarray(source_edge_parameter_denominators)
            if any(
                value.dtype.kind not in "iu"
                for value in (raw_edge_ids, raw_numerators, raw_denominators)
            ):
                raise TypeError(
                    "Dual edge ancestry identities and fractions must be integers."
                )
            if any(
                value.dtype.kind == "u"
                and value.size
                and np.max(value) > np.iinfo(np.int64).max
                for value in (raw_edge_ids, raw_numerators, raw_denominators)
            ):
                raise ValueError("Dual edge ancestry values must fit int64.")
            edge_source_ids = np.asarray(raw_edge_ids, dtype=np.int64)
            edge_parameter_numerators = np.asarray(raw_numerators, dtype=np.int64)
            edge_parameter_denominators = np.asarray(raw_denominators, dtype=np.int64)
            if (
                edge_source_ids.shape != target_edge_ids.shape
                or edge_parameter_numerators.shape != (target_edge_ids.size, 2)
                or edge_parameter_denominators.shape != (target_edge_ids.size, 2)
                or np.any(edge_source_ids < -1)
                or np.any(edge_parameter_numerators < 0)
                or np.any(edge_parameter_denominators <= 0)
                or np.any(edge_parameter_numerators > edge_parameter_denominators)
            ):
                raise ValueError(
                    "Dual edge ancestry arrays do not match target edge identities."
                )
            source_connectivity = source.connectivity
            target_connectivity = mesh.connectivity
            if isinstance(source_connectivity, PolygonalConnectivity) and isinstance(
                target_connectivity, PolygonalConnectivity
            ):
                source_edge_rows = np.asarray(source_connectivity.edges, dtype=np.int64)
                target_edge_rows = np.asarray(target_connectivity.edges, dtype=np.int64)
            elif (
                isinstance(source_connectivity, SimplicialConnectivity)
                and (source_connectivity.dimension == 2)
                and isinstance(target_connectivity, PolygonalConnectivity)
            ):
                source_edge_rows = np.asarray(
                    source_connectivity.entities[1], dtype=np.int64
                )
                target_edge_rows = np.asarray(target_connectivity.edges, dtype=np.int64)
            elif isinstance(source_connectivity, PolyhedralConnectivity) and isinstance(
                target_connectivity, (HexahedralConnectivity, PolyhedralConnectivity)
            ):
                source_edge_rows = np.asarray(source_connectivity.edges, dtype=np.int64)
                target_edge_rows = np.asarray(target_connectivity.edges, dtype=np.int64)
            elif isinstance(source_connectivity, TetrahedralConnectivity) and isinstance(
                target_connectivity, (HexahedralConnectivity, PolyhedralConnectivity)
            ):
                source_edge_rows = np.asarray(source_connectivity.edges, dtype=np.int64)
                target_edge_rows = np.asarray(target_connectivity.edges, dtype=np.int64)
            elif (
                isinstance(source_connectivity, SimplicialConnectivity)
                and source_connectivity.dimension == 3
                and isinstance(
                    target_connectivity,
                    (HexahedralConnectivity, PolyhedralConnectivity),
                )
            ):
                source_edge_rows = np.asarray(
                    source_connectivity.entities[1], dtype=np.int64
                )
                target_edge_rows = np.asarray(target_connectivity.edges, dtype=np.int64)
            else:
                raise TypeError(
                    "Dual edge ancestry requires supported source and target connectivity."
                )
            source_edge_entity_ids = np.asarray(
                source.entity_set(1).entity_ids, dtype=np.int64
            )
            if source_edge_rows.shape != (
                source_edge_entity_ids.size,
                2,
            ) or target_edge_rows.shape != (target_edge_ids.size, 2):
                raise ValueError(
                    "Dual source and target edge rows are not aligned with identities."
                )
            source_edge_row_by_id = {
                int(identifier): row
                for row, identifier in enumerate(source_edge_entity_ids.tolist())
            }
            for target_row, source_edge_id in enumerate(edge_source_ids.tolist()):
                if source_edge_id < 0:
                    continue
                source_row = source_edge_row_by_id.get(source_edge_id)
                if source_row is None:
                    raise ValueError(
                        "Dual source-edge ancestry names an absent source edge."
                    )
                endpoints = tuple(int(value) for value in source_edge_rows[source_row])
                endpoint_set = set(endpoints)
                for local_endpoint, target_vertex in enumerate(
                    target_edge_rows[target_row]
                ):
                    support = tuple(
                        int(value) for value in supports[int(target_vertex)] if value >= 0
                    )
                    if len(support) == 1 and support[0] in endpoint_set:
                        expected = (endpoints.index(support[0]), 1)
                    elif len(support) == 2 and set(support) == endpoint_set:
                        expected = (1, 2)
                    else:
                        raise ValueError(
                            "Dual edge ancestry changes its exact source-edge support."
                        )
                    if (
                        edge_parameter_numerators[target_row, local_endpoint]
                        * expected[1]
                        != expected[0]
                        * edge_parameter_denominators[target_row, local_endpoint]
                    ):
                        raise ValueError(
                            "Dual edge ancestry changes its exact source-edge parameter."
                        )
        if (
            offsets.shape != (cell_count + 1,)
            or offsets[0] != 0
            or offsets[-1] != parents.size
            or np.any(offsets[1:] <= offsets[:-1])
            or np.any(parents < 0)
            or np.any(parents >= source.connectivity.cell_count)
        ):
            raise ValueError("Dual cells must retain one or more valid source parents.")
        if (
            valence.shape != (mesh.coordinates.shape[0],)
            or boundary.shape != valence.shape
        ):
            raise ValueError(
                "Dual valence and boundary evidence must cover target vertices."
            )
        if singular.ndim != 1 or np.any(singular < 0) or np.any(singular >= valence.size):
            raise ValueError("Dual singular vertices must index target vertices.")
        from ..geometry._mesh_certificates import _same_tree_content

        if cross_field is not None and (
            source.topological_dimension != 2
            or cross_field.source_topology_id != source.topology_id
            or cross_field.source_geometry_id != source.geometry_id
            or not _same_tree_content(cross_field.source, source)
        ):
            raise ValueError("The cross field must bind the surface extraction source.")
        if field_event is not None and (
            cross_field is None
            or source_geometry is None
            or field_event.source.mesh_id != source.mesh_id
            or field_event.field.field_id != cross_field.field_id
            or cell_geometry_id(field_event.source_geometry)
            != cell_geometry_id(source_geometry)
            or not _same_tree_content(field_event.source, source)
            or not _same_tree_content(field_event.field, cross_field)
            or not _same_tree_content(field_event.source_geometry, source_geometry)
        ):
            raise ValueError(
                "The field event must bind the extraction's source and field."
            )
        self.mesh = mesh
        self.source = source
        self.vertex_supports = jnp.asarray(supports)
        self.parent_offsets = jnp.asarray(offsets)
        self.parent_cells = jnp.asarray(parents)
        self.validity = validity
        self.valences = jnp.asarray(valence)
        self.boundary_vertices = jnp.asarray(boundary)
        self.singular_vertices = jnp.asarray(singular)
        self.cross_field = cross_field
        self.field_event = field_event
        self.geometry = geometry
        self.source_geometry = source_geometry
        self.source_edge_ids = jnp.asarray(edge_source_ids, dtype=jnp.int64)
        self.source_edge_parameter_numerators = jnp.asarray(
            edge_parameter_numerators, dtype=jnp.int64
        )
        self.source_edge_parameter_denominators = jnp.asarray(
            edge_parameter_denominators, dtype=jnp.int64
        )


@final
class _RestrictedCoordinateMap(StrictModule, NonTrainableState):
    """Bind a restricted target chart to its exact root mesh and coordinate map."""

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    rounded_vertices: Array
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        /,
    ) -> None:
        if not isinstance(source_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
            raise TypeError(
                "Restricted coordinate maps require source and target CellMesh values."
            )
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("source_geometry must be a CellGeometrySpec.")
        source_geometry.resolve(source_mesh)
        source_map = source_geometry.source_map
        if source_map is None:
            raise ValueError(
                "A restricted source-map adapter requires an exact source map."
            )
        self.source_mesh = source_mesh
        self.source_geometry = source_geometry
        self.rounded_vertices = jnp.asarray(target_mesh.coordinates)
        self.source_mesh_id = source_mesh.mesh_id
        self.source_topology_id = source_mesh.topology_id
        self.target_mesh_id = target_mesh.mesh_id
        self.target_topology_id = target_mesh.topology_id
        self.target_geometry_id = target_mesh.geometry_id
        self.source_id = source_map.source_id
        self.source_revision = source_map.source_revision
        self.source_map_id = canonical_fingerprint(
            {
                "kind": "restricted-source-coordinate-map",
                "source_map": source_map.source_map_id,
                "source_mesh": source_mesh.mesh_id,
                "source_topology": source_mesh.topology_id,
                "source_geometry": cell_geometry_id(source_geometry),
                "target_mesh": target_mesh.mesh_id,
                "target_topology": target_mesh.topology_id,
                "target_geometry": target_mesh.geometry_id,
                "source": (source_map.source_id, source_map.source_revision),
            }
        )

    @classmethod
    def from_stored_fields(
        cls,
        value: _RestrictedCoordinateMap,
        /,
        *,
        target_mesh: CellMesh,
    ) -> _RestrictedCoordinateMap:
        if type(value) is not cls:
            raise TypeError(
                "Stored restricted coordinates must be _RestrictedCoordinateMap."
            )
        if (
            target_mesh.mesh_id != value.target_mesh_id
            or target_mesh.topology_id != value.target_topology_id
            or target_mesh.geometry_id != value.target_geometry_id
        ):
            raise ValueError(
                "Stored restricted coordinates bind a different target mesh."
            )
        renewed = cls(value.source_mesh, value.source_geometry, target_mesh)
        if (
            renewed.source_mesh_id != value.source_mesh_id
            or renewed.source_topology_id != value.source_topology_id
            or renewed.source_id != value.source_id
            or renewed.source_revision != value.source_revision
            or renewed.source_map_id != value.source_map_id
            or not np.array_equal(
                np.asarray(renewed.rounded_vertices),
                np.asarray(value.rounded_vertices),
            )
        ):
            raise ValueError(
                "Stored restricted coordinate inputs do not renew their exact identity."
            )
        return renewed

    def source_coordinates(
        self, indices: ConvertibleToArray | None = None, /
    ) -> CoordinateSourceBank:
        normalized_indices = None if indices is None else np.asarray(indices)
        return self.source_geometry.source_coordinates(normalized_indices)


def _affine_geometry_matches(source: CellMesh, geometry: CellGeometrySpec, /) -> bool:
    affine = CellGeometrySpec.affine(source)
    if geometry.geometry_layout_id != affine.geometry_layout_id:
        return False
    if not np.array_equal(
        np.asarray(geometry.coordinates), np.asarray(source.coordinates)
    ):
        return False
    return all(
        np.array_equal(np.asarray(route), np.asarray(block.vertices))
        for route, block in zip(geometry.geometry_dofs, source.blocks, strict=True)
    )


def fresh_vertex_global_ids(source: CellMesh, added: int, /) -> np.ndarray:
    """Preserve source vertex identities and allocate distinct generated IDs."""
    source_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    count = int(added)
    if count < 0:
        raise ValueError("added must be non-negative.")
    if count == 0:
        return source_ids.copy()
    maximum = int(np.max(source_ids))
    limit = int(np.iinfo(np.int64).max)
    if maximum + count <= limit:
        created = np.arange(maximum + 1, maximum + count + 1, dtype=np.int64)
    else:
        used = set(source_ids.tolist())
        allocated: list[int] = []
        candidate = 0
        while len(allocated) < count:
            if candidate not in used:
                allocated.append(candidate)
                used.add(candidate)
            candidate += 1
        created = np.asarray(allocated, dtype=np.int64)
    return np.concatenate((source_ids, created))


__all__ = [
    "QuadLayoutObstruction",
    "SurfaceCrossField",
    "SurfaceQuadLayout",
    "SurfaceQuadFieldArchiveRecipe",
    "SurfaceQuadFieldEvent",
    "restore_surface_quad_field_event",
    "DualExtraction",
    "fresh_vertex_global_ids",
]
