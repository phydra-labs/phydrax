# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Exact lattice admission for orthogonal PLC hexahedral grids."""

from __future__ import annotations

from bisect import bisect_left
from fractions import Fraction
from itertools import combinations
from typing import final, Protocol, runtime_checkable

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar, normalized_identifier
from ..discretization import CellMesh
from ..discretization._cell_mesh import PolyhedralBlock
from ..discretization._reference_cell import reference_cell_topology
from ..typing import checked, ConvertibleToArray
from ._hex_schedule import NativeHexGridSchedule
from ._plc_source import PiecewiseLinearComplex


@runtime_checkable
class _HexFrameIntegerLayout(Protocol):
    @property
    def cell_global_ids(self) -> np.ndarray: ...

    @property
    def cell_block_indices(self) -> np.ndarray: ...

    @property
    def block_axes(
        self,
    ) -> tuple[tuple[tuple[Fraction, Fraction, Fraction], ...], ...]: ...

    @property
    def block_origins(
        self,
    ) -> tuple[tuple[Fraction, Fraction, Fraction], ...]: ...

    @property
    def block_axis_knots(
        self,
    ) -> tuple[
        tuple[
            tuple[Fraction, ...],
            tuple[Fraction, ...],
            tuple[Fraction, ...],
        ],
        ...,
    ]: ...

    @property
    def source_complex_id(self) -> str: ...

    @property
    def source_mesh_id(self) -> str: ...

    @property
    def source_topology_id(self) -> str: ...

    @property
    def schedule_id(self) -> str: ...

    @property
    def frame_id(self) -> str: ...

    @property
    def layout_id(self) -> str: ...


def _exact_point3(values: np.ndarray, /) -> tuple[Fraction, Fraction, Fraction]:
    return (
        Fraction.from_float(float(values[0])),
        Fraction.from_float(float(values[1])),
        Fraction.from_float(float(values[2])),
    )


def _exact_matrix3(
    values: np.ndarray, /
) -> tuple[
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
]:
    return (
        _exact_point3(values[0]),
        _exact_point3(values[1]),
        _exact_point3(values[2]),
    )


def _as_fraction(value: float, /) -> Fraction:
    return Fraction.from_float(float(value))


def _lattice_index(value: Fraction, knots: tuple[Fraction, ...], /) -> int:
    row = bisect_left(knots, value)
    if row >= len(knots) or knots[row] != value:
        raise ValueError("Every PLC feature coordinate must lie on an exact grid plane.")
    return row


@final
class OrthogonalPlcGridPlan(StrictModule, NonTrainableState):
    """Exact axis-aligned lattice and source-facet plane correspondence."""

    axis_knots: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    cell_counts: tuple[int, int, int] = eqx.field(static=True)
    facet_planes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    source_complex_id: str = eqx.field(static=True)
    source_vertex_ids: tuple[str, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    target_size: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_knots: tuple[tuple[Fraction, ...], ...],
        facet_planes: tuple[tuple[int, int], ...],
        source: PiecewiseLinearComplex,
        schedule: NativeHexGridSchedule,
        target_size: float,
        /,
    ) -> None:
        if len(axis_knots) != 3 or any(len(axis) < 2 for axis in axis_knots):
            raise ValueError(
                "Orthogonal grid plan requires three nonempty coordinate axes."
            )
        if any(
            first >= second
            for axis in axis_knots
            for first, second in zip(axis[:-1], axis[1:], strict=True)
        ):
            raise ValueError("Orthogonal grid knots must be strictly increasing.")
        counts = (
            len(axis_knots[0]) - 1,
            len(axis_knots[1]) - 1,
            len(axis_knots[2]) - 1,
        )
        planes = tuple((int(axis), int(index)) for axis, index in facet_planes)
        if any(
            axis not in (0, 1, 2) or not 0 <= index <= counts[axis]
            for axis, index in planes
        ):
            raise ValueError(
                "Orthogonal PLC facet planes must index declared grid knots."
            )
        target = float(target_size)
        self.axis_knots = axis_knots
        self.cell_counts = counts
        self.facet_planes = planes
        self.source_complex_id = source.complex_id
        self.source_vertex_ids = source.vertex_ids
        self.schedule_id = schedule.schedule_id
        self.target_size = target
        self.plan_id = canonical_fingerprint(
            {
                "kind": "orthogonal-plc-grid-plan",
                "complex": source.complex_id,
                "vertices": source.vertex_ids,
                "schedule": schedule.schedule_id,
                "axis_knots": tuple(
                    tuple((value.numerator, value.denominator) for value in axis)
                    for axis in axis_knots
                ),
                "cell_counts": counts,
                "facet_planes": planes,
                "target_size": target,
            }
        )


@checked
def prepare_orthogonal_plc_grid(
    source: PiecewiseLinearComplex,
    schedule: NativeHexGridSchedule,
    target_size: float,
    /,
) -> OrthogonalPlcGridPlan:
    """Admit a conforming grid only when every PLC face is an exact lattice union.

    This route handles orthogonal, grid-aligned PLCs without cut cells. It does
    not fit non-axis-aligned or curved boundaries. Every source vertex, material
    interface, polygon edge, and declared feature segment must lie on the exact
    grid; a partial cell is never classified by a sample-point fallback.
    """
    if schedule.route != "balanced_grid":
        raise ValueError("Orthogonal PLC grid preparation requires balanced_grid.")
    if source.boundary != "conforming":
        raise ValueError(
            "Orthogonal PLC grid requires conforming source-facet subdivision."
        )
    if source.geometry is not None:
        raise ValueError(
            "Orthogonal PLC grid requires the represented piecewise-linear source map."
        )
    target = finite_real_scalar(target_size, "target_size")
    if target <= 0.0:
        raise ValueError("target_size must be positive.")

    vertices = np.asarray(source.vertices, dtype=np.float64)
    exact_vertices = tuple(
        tuple(_as_fraction(value) for value in point) for point in vertices
    )
    target_fraction = _as_fraction(target)
    axis_knots = []
    for axis in range(3):
        origin = min(point[axis] for point in exact_vertices)
        end = max(point[axis] for point in exact_vertices)
        span = end - origin
        if span <= 0:
            raise ValueError(
                "Orthogonal PLC source must have positive extent on every axis."
            )
        full_intervals = (span / target_fraction).numerator // (
            span / target_fraction
        ).denominator
        knots = [origin + target_fraction * index for index in range(full_intervals + 1)]
        if knots[-1] < end:
            knots.append(end)
        axis = tuple(knots)
        for value in axis:
            represented = float(value)
            if not np.isfinite(represented) or _as_fraction(represented) != value:
                raise ValueError(
                    "Grid coordinates must be exactly representable in binary64."
                )
        axis_knots.append(axis)
    knots_by_axis = tuple(axis_knots)
    for point in exact_vertices:
        for axis in range(3):
            _lattice_index(point[axis], knots_by_axis[axis])

    offsets = np.asarray(source.polygon_offsets, dtype=np.int64)
    polygon_vertices = np.asarray(source.polygon_vertices, dtype=np.int64)
    polygon_facets = np.asarray(source.polygon_facets, dtype=np.int64)
    facet_planes: list[tuple[int, int] | None] = [None] * len(source.facet_ids)
    for polygon, facet in enumerate(polygon_facets.tolist()):
        start, stop = offsets[polygon : polygon + 2]
        loop = polygon_vertices[start:stop]
        coordinates = vertices[loop]
        constant_axes = [
            axis
            for axis in range(3)
            if np.all(coordinates[:, axis] == coordinates[0, axis])
        ]
        if len(constant_axes) != 1:
            raise ValueError(
                "Every orthogonal PLC polygon must lie in exactly one coordinate plane."
            )
        axis = constant_axes[0]
        plane_index = _lattice_index(
            _as_fraction(coordinates[0, axis]), knots_by_axis[axis]
        )
        for row in range(len(loop)):
            first = coordinates[row]
            second = coordinates[(row + 1) % len(loop)]
            if np.count_nonzero(second != first) != 1:
                raise ValueError("Orthogonal PLC polygon edges must follow grid lines.")
        previous = facet_planes[facet]
        if previous is not None and previous != (axis, plane_index):
            raise ValueError("Every polygon of one PLC facet must share one grid plane.")
        facet_planes[facet] = (axis, plane_index)
    if any(value is None for value in facet_planes):
        raise ValueError("Every PLC facet must have an oriented source polygon.")

    segments = np.asarray(source.segments, dtype=np.int64)
    for first, second in segments:
        a, b = exact_vertices[int(first)], exact_vertices[int(second)]
        if sum(a[axis] != b[axis] for axis in range(3)) != 1:
            raise ValueError("Orthogonal PLC feature segments must follow grid lines.")
        for axis in range(3):
            _lattice_index(a[axis], knots_by_axis[axis])
            _lattice_index(b[axis], knots_by_axis[axis])

    return OrthogonalPlcGridPlan(
        knots_by_axis,
        tuple(value for value in facet_planes if value is not None),
        source,
        schedule,
        target,
    )


@final
class BalancedCutGridPlan(StrictModule, NonTrainableState):
    """Exact lattice and solved affine frame for partial-cell PLC clipping."""

    axis_knots: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    cell_counts: tuple[int, int, int] = eqx.field(static=True)
    source_complex_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_vertex_ids: tuple[int, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    target_size: float = eqx.field(static=True)
    frame_axes: tuple[tuple[Fraction, Fraction, Fraction], ...] = eqx.field(static=True)
    frame_origin: tuple[Fraction, Fraction, Fraction] = eqx.field(static=True)
    frame_id: str | None = eqx.field(static=True)
    frame_layout_id: str | None = eqx.field(static=True)
    frame_layout: _HexFrameIntegerLayout | None = eqx.field(static=True)
    frame_block_indices: tuple[int, ...] = eqx.field(static=True)
    block_axis_knots: tuple[tuple[tuple[Fraction, ...], ...], ...] = eqx.field(
        static=True
    )
    block_frame_axes: tuple[tuple[tuple[Fraction, Fraction, Fraction], ...], ...] = (
        eqx.field(static=True)
    )
    block_frame_origins: tuple[tuple[Fraction, Fraction, Fraction], ...] = eqx.field(
        static=True
    )
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_knots: tuple[tuple[Fraction, ...], ...],
        source: PiecewiseLinearComplex,
        source_mesh: CellMesh,
        schedule: NativeHexGridSchedule,
        target_size: float,
        /,
        *,
        source_id: str,
        source_revision: str,
        frame_axes: ConvertibleToArray | None = None,
        frame_origin: ConvertibleToArray | None = None,
        frame_id: str | None = None,
        frame_layout: _HexFrameIntegerLayout | None = None,
    ) -> None:
        if len(axis_knots) != 3 or any(len(axis) < 2 for axis in axis_knots):
            raise ValueError("Balanced cut grid requires three exact nonempty axes.")
        if any(
            first >= second
            for axis in axis_knots
            for first, second in zip(axis[:-1], axis[1:], strict=True)
        ):
            raise ValueError("Balanced cut grid knots must be strictly increasing.")
        if schedule.route not in ("balanced_grid", "frame_grid"):
            raise ValueError(
                "Partial PLC clipping requires a balanced or frame grid schedule."
            )
        layout_id: str | None = None
        layout_cell_blocks: tuple[int, ...] = ()
        layout_knots: tuple[tuple[tuple[Fraction, ...], ...], ...] = ()
        layout_axes: tuple[tuple[tuple[Fraction, Fraction, Fraction], ...], ...] = ()
        layout_origins: tuple[tuple[Fraction, Fraction, Fraction], ...] = ()
        exact_layout_axes: (
            tuple[
                tuple[Fraction, Fraction, Fraction],
                tuple[Fraction, Fraction, Fraction],
                tuple[Fraction, Fraction, Fraction],
            ]
            | None
        ) = None
        exact_layout_origin: tuple[Fraction, Fraction, Fraction] | None = None
        if frame_layout is not None:
            if schedule.route != "frame_grid":
                raise ValueError(
                    "A field-block integer layout requires a frame_grid schedule."
                )
            if not isinstance(frame_layout, _HexFrameIntegerLayout):
                raise TypeError(
                    "frame_layout must be an exact HexFrameIntegerLayout record."
                )
            source_cell_ids = tuple(
                int(value)
                for block in source_mesh.blocks
                for value in np.asarray(block.global_ids, dtype=np.int64).tolist()
            )
            layout_cell_ids = tuple(
                int(value) for value in frame_layout.cell_global_ids.tolist()
            )
            raw_blocks = frame_layout.cell_block_indices
            raw_axes = frame_layout.block_axes
            raw_origins = frame_layout.block_origins
            raw_knots = frame_layout.block_axis_knots
            if (
                frame_layout.source_complex_id != source.complex_id
                or frame_layout.source_mesh_id != source_mesh.mesh_id
                or frame_layout.source_topology_id != source_mesh.topology_id
                or frame_layout.schedule_id != schedule.schedule_id
                or not frame_layout.frame_id
                or layout_cell_ids != source_cell_ids
                or raw_blocks.shape != (len(source_cell_ids),)
                or not raw_axes
                or len(raw_origins) != len(raw_axes)
                or len(raw_knots) != len(raw_axes)
                or np.any((raw_blocks < 0) | (raw_blocks >= len(raw_axes)))
            ):
                raise ValueError(
                    "Frame layout must bind every source tetrahedron in exact mesh order."
                )
            layout_axes = raw_axes
            layout_origins = raw_origins
            layout_knots = raw_knots
            if any(
                len(axes) != 3
                or any(len(row) != 3 for row in axes)
                or len(origin) != 3
                or len(chart) != 3
                or any(
                    len(axis) < 2
                    or any(
                        first >= second
                        for first, second in zip(axis[:-1], axis[1:], strict=True)
                    )
                    for axis in chart
                )
                for axes, origin, chart in zip(
                    layout_axes, layout_origins, layout_knots, strict=True
                )
            ):
                raise ValueError(
                    "Frame layout block charts and exact knot axes must be complete."
                )
            expected_frame_id = normalized_identifier(frame_layout.frame_id, "frame_id")
            if frame_id is not None and frame_id != expected_frame_id:
                raise ValueError(
                    "Balanced cut plan frame identity differs from its field layout."
                )
            frame_id = expected_frame_id
            layout_id = normalized_identifier(frame_layout.layout_id, "layout_id")
            layout_cell_blocks = tuple(int(value) for value in raw_blocks.tolist())
            if axis_knots != raw_knots[0]:
                raise ValueError(
                    "Balanced cut plan's representative knots differ from its first frame block."
                )
            if frame_axes is not None and not np.array_equal(
                np.asarray(frame_axes, dtype=np.float64),
                np.asarray(raw_axes[0], dtype=np.float64),
            ):
                raise ValueError(
                    "Balanced cut plan's representative frame differs from its first block."
                )
            if frame_origin is not None and not np.array_equal(
                np.asarray(frame_origin, dtype=np.float64),
                np.asarray(raw_origins[0], dtype=np.float64),
            ):
                raise ValueError(
                    "Balanced cut plan's representative origin differs from its first block."
                )
            raw_representative_axes = raw_axes[0]
            if len(raw_representative_axes) != 3 or len(raw_origins[0]) != 3:
                raise ValueError(
                    "Frame layout representative coordinates must be three-dimensional."
                )
            exact_layout_axes = (
                raw_representative_axes[0],
                raw_representative_axes[1],
                raw_representative_axes[2],
            )
            exact_layout_origin = raw_origins[0]
            frame_axes = np.asarray(raw_representative_axes, dtype=np.float64)
            frame_origin = np.asarray(exact_layout_origin, dtype=np.float64)
        elif schedule.route == "frame_grid" and (
            frame_axes is None or frame_origin is None or not frame_id
        ):
            raise ValueError(
                "A frame-grid plan requires its solved orthonormal frame and identity."
            )
        if schedule.route == "balanced_grid" and frame_id is not None:
            raise ValueError(
                "A balanced-grid plan cannot claim a solved frame-field identity."
            )
        if frame_layout is not None and schedule.route != "frame_grid":
            raise ValueError("Field-block frame layouts are valid only for frame_grid.")
        target = finite_real_scalar(target_size, "target_size")
        if target <= 0.0:
            raise ValueError("target_size must be positive.")
        axes_host = (
            np.eye(3, dtype=np.float64)
            if frame_axes is None
            else np.asarray(frame_axes, dtype=np.float64)
        )
        origin_host = (
            np.zeros((3,), dtype=np.float64)
            if frame_origin is None
            else np.asarray(frame_origin, dtype=np.float64)
        )
        if (
            axes_host.shape != (3, 3)
            or origin_host.shape != (3,)
            or not np.all(np.isfinite(axes_host))
            or not np.all(np.isfinite(origin_host))
            or not np.allclose(axes_host @ axes_host.T, np.eye(3), rtol=0.0, atol=1.0e-10)
            or float(np.linalg.det(axes_host)) <= 0.0
        ):
            raise ValueError(
                "Cut grid frame must be a finite right-handed orthonormal basis."
            )
        exact_axes = (
            exact_layout_axes
            if exact_layout_axes is not None
            else _exact_matrix3(axes_host)
        )
        exact_origin = (
            exact_layout_origin
            if exact_layout_origin is not None
            else _exact_point3(origin_host)
        )
        frame_identity = (
            None if frame_id is None else normalized_identifier(frame_id, "frame_id")
        )
        counts = (
            len(axis_knots[0]) - 1,
            len(axis_knots[1]) - 1,
            len(axis_knots[2]) - 1,
        )
        vertex_ids = tuple(
            int(value) for value in np.asarray(source_mesh.vertex_global_ids).tolist()
        )
        source_identifier = normalized_identifier(source_id, "source_id")
        revision_identifier = normalized_identifier(source_revision, "source_revision")
        self.axis_knots = axis_knots
        self.cell_counts = counts
        self.source_complex_id = source.complex_id
        self.source_id = source_identifier
        self.source_revision = revision_identifier
        self.source_mesh_id = source_mesh.mesh_id
        self.source_topology_id = source_mesh.topology_id
        self.source_vertex_ids = vertex_ids
        self.schedule_id = schedule.schedule_id
        self.target_size = target
        self.frame_axes = exact_axes
        self.frame_origin = exact_origin
        self.frame_id = frame_identity
        self.frame_layout_id = layout_id
        self.frame_layout = frame_layout
        self.frame_block_indices = layout_cell_blocks
        self.block_axis_knots = layout_knots
        self.block_frame_axes = layout_axes
        self.block_frame_origins = layout_origins
        self.plan_id = canonical_fingerprint(
            {
                "kind": "balanced-partial-cut-grid-plan",
                "source_complex": source.complex_id,
                "source_id": source_identifier,
                "source_revision": revision_identifier,
                "source_mesh": source_mesh.mesh_id,
                "source_topology": source_mesh.topology_id,
                "source_vertices": vertex_ids,
                "schedule": schedule.schedule_id,
                "target_size": target,
                "axis_knots": tuple(
                    tuple((value.numerator, value.denominator) for value in axis)
                    for axis in axis_knots
                ),
                "frame": {
                    "id": frame_identity,
                    "axes": tuple(
                        tuple((value.numerator, value.denominator) for value in row)
                        for row in exact_axes
                    ),
                    "origin": tuple(
                        (value.numerator, value.denominator) for value in exact_origin
                    ),
                    "layout_id": layout_id,
                    "block_indices": layout_cell_blocks,
                    "block_axes": tuple(
                        tuple(
                            tuple((value.numerator, value.denominator) for value in row)
                            for row in axes
                        )
                        for axes in layout_axes
                    ),
                    "block_origins": tuple(
                        tuple((value.numerator, value.denominator) for value in origin)
                        for origin in layout_origins
                    ),
                    "block_axis_knots": tuple(
                        tuple(
                            tuple((value.numerator, value.denominator) for value in axis)
                            for axis in chart
                        )
                        for chart in layout_knots
                    ),
                },
                "cell_counts": counts,
            }
        )


def _generic_axis_knots(
    values: tuple[Fraction, ...],
    spacing: Fraction,
    offset_exponent: int,
    /,
) -> tuple[Fraction, ...]:
    lower, upper = min(values), max(values)
    if lower >= upper:
        raise ValueError(
            "Balanced cut grid source fill must have positive extent on each axis."
        )
    exponent = int(offset_exponent)
    while True:
        offset = spacing / (1 << exponent)
        origin = lower - offset
        if all(((value - origin) / spacing).denominator != 1 for value in values):
            break
        exponent += 1
    interval_count = (
        int(
            ((upper - origin) / spacing).numerator
            // ((upper - origin) / spacing).denominator
        )
        + 1
    )
    return tuple(origin + index * spacing for index in range(interval_count + 1))


def _simplest_fraction_between(lower: Fraction, upper: Fraction, /) -> Fraction:
    """Smallest-denominator rational in the closed interval ``[lower, upper]``."""
    if lower > upper:
        raise ValueError("Simplest-rational search requires an ordered interval.")
    floor = lower.numerator // lower.denominator
    if floor == lower or floor + 1 <= upper:
        return Fraction(floor if floor == lower else floor + 1)
    return floor + 1 / _simplest_fraction_between(
        1 / (upper - floor), 1 / (lower - floor)
    )


def _separated_lattice(
    events: list[Fraction],
    lower: Fraction,
    upper: Fraction,
    spacing: Fraction,
    /,
) -> tuple[Fraction, ...]:
    """Place the fewest knots covering ``[lower, upper]`` far from every event.

    The origin ranges over the window that keeps that knot count; it is put in
    the middle of the widest gap between window-shifted events, so every event
    lies at least a quarter of that gap from every knot. The separation is then
    a positive fraction of the spacing rather than a binary64 accident. A
    window shorter than a quarter spacing takes one more interval.
    """
    intervals = int((upper - lower) // spacing) + 1
    if intervals * spacing - (upper - lower) < spacing / 4:
        intervals += 1
    window_low = upper - intervals * spacing
    forbidden = {window_low, lower}
    for event in events:
        shifted = event - spacing * ((event - window_low) // spacing)
        while shifted <= lower:
            forbidden.add(shifted)
            shifted += spacing
    ordered = sorted(forbidden)
    width, start = max(
        (
            (second - first, first)
            for first, second in zip(ordered[:-1], ordered[1:], strict=True)
        ),
        key=lambda gap: gap[0],
    )
    origin = _simplest_fraction_between(start + width / 4, start + 3 * width / 4)
    return tuple(origin + index * spacing for index in range(intervals + 1))


def _separated_axis_knots(
    points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    cells: np.ndarray,
    spacing: Fraction,
    /,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Choose cut-grid knots robustly separated from the source arrangement.

    Exact genericity is insufficient: a grid plane within roundoff of a source
    vertex, a grid line within roundoff of a source edge, or a grid vertex
    within roundoff of a source face yields clipped cells of exactly positive
    but numerically zero volume. Axes are placed in order; each later axis
    avoids the coordinates where earlier grid planes and lines cross source
    edges and faces, which bounds every clipped-vertex separation below.
    """
    edges = sorted(
        {
            (min(first, second), max(first, second))
            for cell in cells.tolist()
            for first, second in combinations(map(int, cell), 2)
        }
    )
    faces = sorted(
        {
            tuple(sorted(face))
            for cell in cells.tolist()
            for face in combinations(map(int, cell), 3)
        }
    )
    knots: list[tuple[Fraction, ...]] = []
    for axis in range(3):
        values = [point[axis] for point in points]
        events = list(values)
        for crossing_axis, crossing_knots in enumerate(knots):
            for first, second in edges:
                start, end = points[first], points[second]
                low, high = sorted((start[crossing_axis], end[crossing_axis]))
                for knot in crossing_knots:
                    if low < knot < high:
                        ratio = (knot - start[crossing_axis]) / (
                            end[crossing_axis] - start[crossing_axis]
                        )
                        events.append(start[axis] + ratio * (end[axis] - start[axis]))
        if axis == 2:
            for face in faces:
                corners = tuple(points[row] for row in face)
                determinant = (corners[1][0] - corners[0][0]) * (
                    corners[2][1] - corners[0][1]
                ) - (corners[2][0] - corners[0][0]) * (corners[1][1] - corners[0][1])
                if determinant == 0:
                    continue
                x_low = min(corner[0] for corner in corners)
                x_high = max(corner[0] for corner in corners)
                y_low = min(corner[1] for corner in corners)
                y_high = max(corner[1] for corner in corners)
                for x_knot in knots[0]:
                    if not x_low < x_knot < x_high:
                        continue
                    for y_knot in knots[1]:
                        if not y_low < y_knot < y_high:
                            continue
                        weights = tuple(
                            (
                                (corners[(row + 1) % 3][0] - x_knot)
                                * (corners[(row + 2) % 3][1] - y_knot)
                                - (corners[(row + 2) % 3][0] - x_knot)
                                * (corners[(row + 1) % 3][1] - y_knot)
                            )
                            / determinant
                            for row in range(3)
                        )
                        if all(weight >= 0 for weight in weights):
                            events.append(
                                sum(
                                    (
                                        weight * corner[2]
                                        for weight, corner in zip(
                                            weights, corners, strict=True
                                        )
                                    ),
                                    Fraction(0),
                                )
                            )
        lower, upper = min(values), max(values)
        if lower >= upper:
            raise ValueError(
                "Balanced cut grid source fill must have positive extent on each axis."
            )
        knots.append(_separated_lattice(events, lower, upper, spacing))
    return knots[0], knots[1], knots[2]


@checked
def prepare_balanced_cut_grid(
    source: PiecewiseLinearComplex,
    source_mesh: CellMesh,
    schedule: NativeHexGridSchedule,
    target_size: float,
    /,
    *,
    source_id: str,
    source_revision: str,
    frame_axes: ConvertibleToArray | None = None,
    frame_origin: ConvertibleToArray | None = None,
    frame_id: str | None = None,
    frame_layout: _HexFrameIntegerLayout | None = None,
) -> BalancedCutGridPlan:
    """Choose an exact generic grid or bind an exact field-block integer layout."""
    if not isinstance(source, PiecewiseLinearComplex):
        raise TypeError("source must be PiecewiseLinearComplex.")
    if not isinstance(source_mesh, CellMesh):
        raise TypeError("source_mesh must be CellMesh.")
    if not isinstance(schedule, NativeHexGridSchedule):
        raise TypeError("schedule must be NativeHexGridSchedule.")
    if (
        source.boundary != "conforming"
        or source_mesh.topological_dimension != 3
        or source_mesh.ambient_dimension != 3
        or not source_mesh.blocks
        or any(
            isinstance(block, PolyhedralBlock)
            or reference_cell_topology(block.cell_kind).family != "tetrahedron"
            for block in source_mesh.blocks
        )
    ):
        raise ValueError(
            "Balanced cut clipping requires a conforming PLC and tetrahedral fill."
        )
    target = finite_real_scalar(target_size, "target_size")
    if target <= 0.0:
        raise ValueError("target_size must be positive.")
    knots: tuple[tuple[Fraction, ...], ...] = ()
    if frame_layout is not None:
        if not isinstance(frame_layout, _HexFrameIntegerLayout):
            raise TypeError("frame_layout must be an exact HexFrameIntegerLayout record.")
        if schedule.route != "frame_grid":
            raise ValueError(
                "A field-block integer layout requires a frame_grid schedule."
            )
        layout_knots = frame_layout.block_axis_knots
        layout_axes = frame_layout.block_axes
        layout_origins = frame_layout.block_origins
        if (
            not layout_knots
            or len(layout_knots) != len(layout_axes)
            or len(layout_axes) != len(layout_origins)
        ):
            raise ValueError(
                "A frame integer layout must retain every exact block chart."
            )
        if frame_id is not None and frame_id != frame_layout.frame_id:
            raise ValueError(
                "frame_id differs from the exact integer layout's frame identity."
            )
        frame_id = frame_layout.frame_id
        representative_axes = np.asarray(layout_axes[0], dtype=np.float64)
        representative_origin = np.asarray(layout_origins[0], dtype=np.float64)
        if frame_axes is not None and not np.array_equal(
            np.asarray(frame_axes, dtype=np.float64), representative_axes
        ):
            raise ValueError(
                "frame_axes differs from the field layout's first block chart."
            )
        if frame_origin is not None and not np.array_equal(
            np.asarray(frame_origin, dtype=np.float64), representative_origin
        ):
            raise ValueError(
                "frame_origin differs from the field layout's first block chart."
            )
        frame_axes = representative_axes
        frame_origin = representative_origin
        knots = layout_knots[0]
    axes_host = (
        np.eye(3, dtype=np.float64)
        if frame_axes is None
        else np.asarray(frame_axes, dtype=np.float64)
    )
    origin_host = (
        np.zeros((3,), dtype=np.float64)
        if frame_origin is None
        else np.asarray(frame_origin, dtype=np.float64)
    )
    if schedule.route == "balanced_grid" and (
        frame_id is not None
        or not np.array_equal(axes_host, np.eye(3, dtype=np.float64))
        or np.any(origin_host != 0.0)
    ):
        raise ValueError("A balanced_grid plan uses the world coordinate frame.")
    if frame_layout is None:
        exact_axes = _exact_matrix3(axes_host)
        exact_origin = _exact_point3(origin_host)
        exact_points: tuple[tuple[Fraction, Fraction, Fraction], ...] = tuple(
            _exact_point3(point)
            for point in np.asarray(source_mesh.coordinates, dtype=np.float64)
        )
        local_points: tuple[tuple[Fraction, Fraction, Fraction], ...] = tuple(
            (
                sum(
                    (
                        exact_axes[0][coordinate]
                        * (point[coordinate] - exact_origin[coordinate])
                        for coordinate in range(3)
                    ),
                    Fraction(0),
                ),
                sum(
                    (
                        exact_axes[1][coordinate]
                        * (point[coordinate] - exact_origin[coordinate])
                        for coordinate in range(3)
                    ),
                    Fraction(0),
                ),
                sum(
                    (
                        exact_axes[2][coordinate]
                        * (point[coordinate] - exact_origin[coordinate])
                        for coordinate in range(3)
                    ),
                    Fraction(0),
                ),
            )
            for point in exact_points
        )
        rounded_spacing = Fraction.from_float(target)
        spacing = _simplest_fraction_between(
            rounded_spacing * (1 - Fraction(1, 1 << 40)),
            rounded_spacing * (1 + Fraction(1, 1 << 40)),
        )
        knots = _separated_axis_knots(
            local_points,
            np.concatenate(
                [
                    np.asarray(block.vertices, dtype=np.int64)
                    for block in source_mesh.blocks
                ]
            ),
            spacing,
        )
    return BalancedCutGridPlan(
        knots,
        source,
        source_mesh,
        schedule,
        target,
        source_id=source_id,
        source_revision=source_revision,
        frame_axes=axes_host,
        frame_origin=origin_host,
        frame_id=frame_id,
        frame_layout=frame_layout,
    )


__all__ = [
    "BalancedCutGridPlan",
    "OrthogonalPlcGridPlan",
    "prepare_balanced_cut_grid",
    "prepare_orthogonal_plc_grid",
]
