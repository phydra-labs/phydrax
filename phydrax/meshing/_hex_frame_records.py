# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Immutable source-bound frame-field and integer-layout evidence."""

from __future__ import annotations

from fractions import Fraction
from typing import final, TypeAlias

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..optim import MinimizationResult
from ..typing import ConvertibleToArray
from ._hex_cut_preparation import BalancedCutGridPlan


_ExactAxis3: TypeAlias = tuple[Fraction, Fraction, Fraction]
_ExactFrame3: TypeAlias = tuple[_ExactAxis3, _ExactAxis3, _ExactAxis3]
_ExactKnotAxis: TypeAlias = tuple[Fraction, ...]
_ExactKnots3: TypeAlias = tuple[_ExactKnotAxis, _ExactKnotAxis, _ExactKnotAxis]


class FrameFieldRefusal(ValueError):
    """Exact obstruction preventing a frame field from defining a valid layout."""

    def __init__(
        self,
        kind: str,
        message: str,
        entity_ids: tuple[int, ...],
        /,
    ) -> None:
        super().__init__(message)
        self.kind = str(kind)
        self.entity_ids = tuple(int(value) for value in entity_ids)


@final
class FrameFieldObstructionRecord(StrictModule, NonTrainableState):
    """Source-bound exact singularity, holonomy, parity, or compatibility refusal."""

    kind: str = eqx.field(static=True)
    entity_ids: tuple[int, ...] = eqx.field(static=True)
    message: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    obstruction_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: str,
        entity_ids: tuple[int, ...],
        message: str,
        frame_id: str,
        /,
    ) -> None:
        kind_ = str(kind)
        identifiers = tuple(int(value) for value in entity_ids)
        message_ = str(message)
        if not kind_ or not identifiers or not message_ or not frame_id:
            raise ValueError(
                "Frame-field obstructions require exact source entities and a field identity."
            )
        self.kind = kind_
        self.entity_ids = identifiers
        self.message = message_
        self.frame_id = str(frame_id)
        self.obstruction_id = canonical_fingerprint(
            {
                "kind": "native-hex-frame-field-obstruction",
                "obstruction": kind_,
                "entity_ids": identifiers,
                "message": message_,
                "frame": self.frame_id,
            }
        )


def _exact_frame_transition(
    left_axes: _ExactFrame3,
    right_axes: _ExactFrame3,
    left_origin: _ExactAxis3,
    right_origin: _ExactAxis3,
    left_knots: _ExactKnots3,
    right_knots: _ExactKnots3,
    entity_ids: tuple[int, int],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive an exact integer chart map from two rational affine frame charts."""
    a, b, c = left_axes[0]
    d, e, f = left_axes[1]
    g, h, i = left_axes[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if determinant == 0:
        raise FrameFieldRefusal(
            "singular-frame-transition",
            "An integer frame chart has a singular exact basis.",
            entity_ids,
        )
    inverse_left = (
        (
            (e * i - f * h) / determinant,
            (c * h - b * i) / determinant,
            (b * f - c * e) / determinant,
        ),
        (
            (f * g - d * i) / determinant,
            (a * i - c * g) / determinant,
            (c * d - a * f) / determinant,
        ),
        (
            (d * h - e * g) / determinant,
            (b * g - a * h) / determinant,
            (a * e - b * d) / determinant,
        ),
    )
    relative = tuple(
        tuple(
            sum(
                (
                    right_axes[row][inner] * inverse_left[inner][column]
                    for inner in range(3)
                ),
                Fraction(0),
            )
            for column in range(3)
        )
        for row in range(3)
    )
    if any(value.denominator != 1 for row in relative for value in row):
        raise FrameFieldRefusal(
            "nonintegrable-face-transition",
            "Adjacent per-cell charts do not have an exact integer-axis transition.",
            entity_ids,
        )
    transition = np.asarray(
        [[int(value) for value in row] for row in relative],
        dtype=np.int64,
    )
    if (
        not np.array_equal(np.sum(np.abs(transition), axis=0), np.ones(3, dtype=np.int64))
        or not np.array_equal(
            np.sum(np.abs(transition), axis=1), np.ones(3, dtype=np.int64)
        )
        or int(round(np.linalg.det(transition))) != 1
    ):
        raise FrameFieldRefusal(
            "singular-frame-transition",
            "Adjacent cell frames induce a singular or orientation-reversing integer transition.",
            entity_ids,
        )
    physical_translation = tuple(
        sum(
            (
                right_axes[axis][coordinate]
                * (left_origin[coordinate] - right_origin[coordinate])
                for coordinate in range(3)
            ),
            Fraction(0),
        )
        for axis in range(3)
    )
    offsets = np.empty((3,), dtype=np.int64)
    for right_axis in range(3):
        left_axis = int(np.flatnonzero(transition[right_axis])[0])
        sign = int(transition[right_axis, left_axis])
        left_spacing = left_knots[left_axis][1] - left_knots[left_axis][0]
        right_spacing = right_knots[right_axis][1] - right_knots[right_axis][0]
        if left_spacing != right_spacing:
            raise FrameFieldRefusal(
                "nonintegrable-block-spacing",
                "Adjacent frame blocks use incompatible exact lattice spacing.",
                entity_ids,
            )
        index_translation = (
            sign * left_knots[left_axis][0]
            + physical_translation[right_axis]
            - right_knots[right_axis][0]
        ) / right_spacing
        if index_translation.denominator != 1:
            raise FrameFieldRefusal(
                "boundary-parity",
                "Adjacent frame charts have a nonintegral exact boundary-knot transition.",
                entity_ids,
            )
        offsets[right_axis] = int(index_translation)
    return transition, offsets


@final
class HexFrameFieldSolution(StrictModule, NonTrainableState):
    """Solved per-root-tetrahedron frames for the general PLC meshing route."""

    cell_frames: np.ndarray
    cell_origins: np.ndarray
    cell_global_ids: np.ndarray
    cell_block_indices: np.ndarray
    cell_integer_coordinates: np.ndarray
    objective_value: float = eqx.field(static=True)
    optimization: MinimizationResult
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_complex_id: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    source_vertex_identities: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(
        static=True
    )
    source_facet_identities: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(
        static=True
    )
    source_feature_identities: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(
        static=True
    )
    source_region_identities: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(
        static=True
    )
    source_facet_regions: tuple[tuple[int, int], ...] = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_frames: ConvertibleToArray,
        cell_origins: ConvertibleToArray,
        cell_global_ids: ConvertibleToArray,
        cell_block_indices: ConvertibleToArray,
        cell_integer_coordinates: ConvertibleToArray,
        objective_value: float,
        optimization: MinimizationResult,
        source_id: str,
        source_revision: str,
        source_complex_id: str,
        source_mesh_id: str,
        source_topology_id: str,
        schedule_id: str,
        source_vertex_identities: tuple[tuple[str, tuple[str, ...]], ...],
        source_facet_identities: tuple[tuple[str, tuple[str, ...]], ...],
        source_feature_identities: tuple[tuple[str, tuple[str, ...]], ...],
        source_region_identities: tuple[tuple[str, tuple[str, ...]], ...],
        source_facet_regions: ConvertibleToArray,
        /,
    ) -> None:
        frames = np.asarray(cell_frames, dtype=np.float64)
        origins = np.asarray(cell_origins, dtype=np.float64)
        identifiers = np.asarray(cell_global_ids, dtype=np.int64)
        blocks = np.asarray(cell_block_indices, dtype=np.int64)
        integer_coordinates = np.asarray(cell_integer_coordinates, dtype=np.int64)
        facet_regions = np.asarray(source_facet_regions, dtype=np.int64)
        objective = float(objective_value)
        count = identifiers.size
        if (
            frames.shape != (count, 3, 3)
            or origins.shape != (count, 3)
            or blocks.shape != (count,)
            or integer_coordinates.shape != (count, 3)
            or facet_regions.ndim != 2
            or facet_regions.shape[1] != 2
            or count == 0
            or np.unique(identifiers).size != count
            or np.any(blocks < 0)
            or not all(np.all(np.isfinite(value)) for value in (frames, origins))
            or not np.isfinite(objective)
            or objective < 0.0
            or any(
                not np.allclose(
                    frame @ frame.T,
                    np.eye(3),
                    rtol=0.0,
                    atol=1.0e-9,
                )
                or float(np.linalg.det(frame)) <= 0.0
                for frame in frames
            )
        ):
            raise ValueError(
                "Hex frame fields must bind finite right-handed per-cell frames and integer roots."
            )
        if not all(
            (
                source_id,
                source_revision,
                source_complex_id,
                source_mesh_id,
                source_topology_id,
                schedule_id,
            )
        ):
            raise ValueError(
                "Hex frame fields require full source topology and schedule identities."
            )
        arrays = (
            frames,
            origins,
            identifiers,
            blocks,
            integer_coordinates,
            facet_regions,
        )
        frozen = []
        for value in arrays:
            copy = np.array(value, copy=True)
            copy.setflags(write=False)
            frozen.append(copy)
        (
            self.cell_frames,
            self.cell_origins,
            self.cell_global_ids,
            self.cell_block_indices,
            self.cell_integer_coordinates,
            facet_region_array,
        ) = frozen
        self.objective_value = objective
        self.optimization = optimization
        self.source_id = str(source_id)
        self.source_revision = str(source_revision)
        self.source_complex_id = str(source_complex_id)
        self.source_mesh_id = str(source_mesh_id)
        self.source_topology_id = str(source_topology_id)
        self.schedule_id = str(schedule_id)
        self.source_vertex_identities = tuple(
            (str(identifier), tuple(path))
            for identifier, path in source_vertex_identities
        )
        self.source_facet_identities = tuple(
            (str(identifier), tuple(path)) for identifier, path in source_facet_identities
        )
        self.source_feature_identities = tuple(
            (str(identifier), tuple(path))
            for identifier, path in source_feature_identities
        )
        self.source_region_identities = tuple(
            (str(identifier), tuple(path))
            for identifier, path in source_region_identities
        )
        self.source_facet_regions = tuple(
            (
                int(facet_region_array[row, 0]),
                int(facet_region_array[row, 1]),
            )
            for row in range(facet_region_array.shape[0])
        )
        self.frame_id = canonical_fingerprint(
            {
                "kind": "native-source-bound-hex-frame-field",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "source_complex": self.source_complex_id,
                "source_mesh": self.source_mesh_id,
                "source_topology": self.source_topology_id,
                "schedule": self.schedule_id,
                "cell_global_ids": self.cell_global_ids,
                "cell_frames": self.cell_frames,
                "cell_origins": self.cell_origins,
                "cell_block_indices": self.cell_block_indices,
                "cell_integer_coordinates": self.cell_integer_coordinates,
                "objective": objective,
                "optimization_status": np.asarray(optimization.status),
                "source_vertex_identities": self.source_vertex_identities,
                "source_facet_identities": self.source_facet_identities,
                "source_feature_identities": self.source_feature_identities,
                "source_region_identities": self.source_region_identities,
                "source_facet_regions": self.source_facet_regions,
            }
        )


@final
class HexFrameIntegerLayout(StrictModule, NonTrainableState):
    """Exact per-tetrahedron block charts and integer face transitions."""

    cell_global_ids: np.ndarray
    cell_block_indices: np.ndarray
    cell_integer_coordinates: np.ndarray
    block_axes: tuple[tuple[tuple[Fraction, Fraction, Fraction], ...], ...] = eqx.field(
        static=True
    )
    block_origins: tuple[tuple[Fraction, Fraction, Fraction], ...] = eqx.field(
        static=True
    )
    block_axis_knots: tuple[_ExactKnots3, ...] = eqx.field(static=True)
    face_connections: np.ndarray
    face_transitions: np.ndarray
    face_offsets: np.ndarray
    boundary_face_parity: np.ndarray
    source_complex_id: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_global_ids: ConvertibleToArray,
        cell_block_indices: ConvertibleToArray,
        cell_integer_coordinates: ConvertibleToArray,
        block_axes: ConvertibleToArray,
        block_origins: ConvertibleToArray,
        block_axis_knots: tuple[_ExactKnots3, ...],
        face_connections: ConvertibleToArray,
        face_transitions: ConvertibleToArray,
        face_offsets: ConvertibleToArray,
        boundary_face_parity: ConvertibleToArray,
        source_complex_id: str,
        source_mesh_id: str,
        source_topology_id: str,
        schedule_id: str,
        frame_id: str,
        /,
    ) -> None:
        identifiers = np.asarray(cell_global_ids, dtype=np.int64)
        blocks = np.asarray(cell_block_indices, dtype=np.int64)
        integer_coordinates = np.asarray(cell_integer_coordinates, dtype=np.int64)
        axes_host = np.asarray(block_axes, dtype=np.float64)
        origins_host = np.asarray(block_origins, dtype=np.float64)
        connections = np.asarray(face_connections, dtype=np.int64)
        transitions = np.asarray(face_transitions, dtype=np.int64)
        offsets = np.asarray(face_offsets, dtype=np.int64)
        parity = np.asarray(boundary_face_parity, dtype=np.int8)
        block_count = axes_host.shape[0] if axes_host.ndim == 3 else 0
        cell_count = identifiers.size
        if (
            cell_count == 0
            or np.unique(identifiers).size != cell_count
            or blocks.shape != (cell_count,)
            or integer_coordinates.shape != (cell_count, 3)
            or axes_host.shape != (block_count, 3, 3)
            or origins_host.shape != (block_count, 3)
            or len(block_axis_knots) != block_count
            or connections.ndim != 2
            or connections.shape[1] != 4
            or transitions.shape != (connections.shape[0], 3, 3)
            or offsets.shape != (connections.shape[0], 3)
            or parity.shape != (block_count,)
            or np.any(blocks < 0)
            or (
                blocks.size
                and not np.array_equal(np.unique(blocks), np.arange(block_count))
            )
            or not all(np.all(np.isfinite(value)) for value in (axes_host, origins_host))
            or any(
                not np.allclose(
                    axes @ axes.T,
                    np.eye(3),
                    rtol=0.0,
                    atol=1.0e-10,
                )
                or float(np.linalg.det(axes)) <= 0.0
                for axes in axes_host
            )
            or np.any(~np.isin(parity, (-1, 1)))
            or any(
                not np.array_equal(
                    np.sum(np.abs(matrix), axis=0), np.ones(3, dtype=np.int64)
                )
                or not np.array_equal(
                    np.sum(np.abs(matrix), axis=1), np.ones(3, dtype=np.int64)
                )
                or int(round(np.linalg.det(matrix))) != 1
                for matrix in transitions
            )
            or (
                connections.size
                and np.any(
                    (connections[:, (0, 2)] < 0) | (connections[:, (0, 2)] >= cell_count)
                )
            )
            or (
                connections.size
                and np.any((connections[:, (1, 3)] < 0) | (connections[:, (1, 3)] >= 4))
            )
            or not all(
                (
                    source_complex_id,
                    source_mesh_id,
                    source_topology_id,
                    schedule_id,
                    frame_id,
                )
            )
        ):
            raise ValueError(
                "Hex integer layouts require exact source-bound block charts and face transitions."
            )
        if np.any(parity != 1):
            block = int(np.flatnonzero(parity != 1)[0])
            rows = np.flatnonzero(blocks == block)
            raise FrameFieldRefusal(
                "boundary-parity",
                "A frame block has incompatible boundary parity.",
                tuple(int(identifiers[int(row)]) for row in rows.tolist()),
            )
        exact_axes: tuple[_ExactFrame3, ...] = tuple(
            (
                (
                    Fraction.from_float(float(axes[0, 0])),
                    Fraction.from_float(float(axes[0, 1])),
                    Fraction.from_float(float(axes[0, 2])),
                ),
                (
                    Fraction.from_float(float(axes[1, 0])),
                    Fraction.from_float(float(axes[1, 1])),
                    Fraction.from_float(float(axes[1, 2])),
                ),
                (
                    Fraction.from_float(float(axes[2, 0])),
                    Fraction.from_float(float(axes[2, 1])),
                    Fraction.from_float(float(axes[2, 2])),
                ),
            )
            for axes in axes_host
        )
        exact_origins: tuple[_ExactAxis3, ...] = tuple(
            (
                Fraction.from_float(float(origin[0])),
                Fraction.from_float(float(origin[1])),
                Fraction.from_float(float(origin[2])),
            )
            for origin in origins_host
        )
        knots: tuple[_ExactKnots3, ...] = tuple(
            (
                tuple(chart[0]),
                tuple(chart[1]),
                tuple(chart[2]),
            )
            for chart in block_axis_knots
        )
        if any(
            len(chart) != 3
            or any(
                len(axis) < 2
                or any(
                    first >= second
                    for first, second in zip(axis[:-1], axis[1:], strict=True)
                )
                or any(
                    second - first != axis[1] - axis[0]
                    for first, second in zip(axis[:-1], axis[1:], strict=True)
                )
                for axis in chart
            )
            for chart in knots
        ):
            raise ValueError(
                "Each frame block requires three exact uniform-spacing knot axes."
            )
        charts = np.zeros((cell_count, 3, 3), dtype=np.int64)
        chart_offsets = np.zeros((cell_count, 3), dtype=np.int64)
        assigned = np.zeros((cell_count,), dtype=np.bool_)
        adjacency: list[list[tuple[int, np.ndarray, np.ndarray, int]]] = [
            [] for _ in range(cell_count)
        ]
        for connection_row, (left, _, right, _) in enumerate(connections.tolist()):
            matrix = transitions[connection_row]
            translation = offsets[connection_row]
            adjacency[int(left)].append((int(right), matrix, translation, connection_row))
            adjacency[int(right)].append(
                (int(left), matrix.T, -matrix.T @ translation, connection_row)
            )
        for root in range(cell_count):
            if assigned[root]:
                continue
            charts[root] = np.eye(3, dtype=np.int64)
            assigned[root] = True
            stack = [root]
            while stack:
                current = stack.pop()
                for neighbor, matrix, translation, connection_row in adjacency[current]:
                    expected_chart = matrix @ charts[current]
                    expected_offset = matrix @ chart_offsets[current] + translation
                    if not assigned[neighbor]:
                        charts[neighbor] = expected_chart
                        chart_offsets[neighbor] = expected_offset
                        assigned[neighbor] = True
                        stack.append(neighbor)
                    elif not (
                        np.array_equal(charts[neighbor], expected_chart)
                        and np.array_equal(chart_offsets[neighbor], expected_offset)
                    ):
                        left_row, _, right_row, _ = connections[connection_row]
                        raise FrameFieldRefusal(
                            "cross-block-holonomy",
                            "Frame integer transitions have nonidentity cycle holonomy.",
                            (
                                int(identifiers[int(left_row)]),
                                int(identifiers[int(right_row)]),
                            ),
                        )
        for connection_row, (left, _, right, _) in enumerate(connections.tolist()):
            left_row, right_row = int(left), int(right)
            left_block, right_block = int(blocks[left_row]), int(blocks[right_row])
            expected_matrix, expected_offsets = _exact_frame_transition(
                exact_axes[left_block],
                exact_axes[right_block],
                exact_origins[left_block],
                exact_origins[right_block],
                knots[left_block],
                knots[right_block],
                (int(identifiers[left_row]), int(identifiers[right_row])),
            )
            if not np.array_equal(transitions[connection_row], expected_matrix):
                raise FrameFieldRefusal(
                    "nonintegrable-face-transition",
                    "Recorded face transport differs from the exact block-chart map.",
                    (int(identifiers[left_row]), int(identifiers[right_row])),
                )
            if not np.array_equal(offsets[connection_row], expected_offsets):
                raise FrameFieldRefusal(
                    "boundary-parity",
                    "Recorded face index offset differs from the exact block-knot transition.",
                    (int(identifiers[left_row]), int(identifiers[right_row])),
                )
        arrays = (
            identifiers,
            blocks,
            integer_coordinates,
            connections,
            transitions,
            offsets,
            parity,
        )
        frozen = []
        for value in arrays:
            copy = np.array(value, copy=True)
            copy.setflags(write=False)
            frozen.append(copy)
        (
            self.cell_global_ids,
            self.cell_block_indices,
            self.cell_integer_coordinates,
            self.face_connections,
            self.face_transitions,
            self.face_offsets,
            self.boundary_face_parity,
        ) = frozen
        self.block_axes = exact_axes
        self.block_origins = exact_origins
        self.block_axis_knots = knots
        self.source_complex_id = str(source_complex_id)
        self.source_mesh_id = str(source_mesh_id)
        self.source_topology_id = str(source_topology_id)
        self.schedule_id = str(schedule_id)
        self.frame_id = str(frame_id)
        self.layout_id = canonical_fingerprint(
            {
                "kind": "native-hex-frame-integer-block-layout",
                "source_complex": self.source_complex_id,
                "source_mesh": self.source_mesh_id,
                "source_topology": self.source_topology_id,
                "schedule": self.schedule_id,
                "frame": self.frame_id,
                "cell_global_ids": self.cell_global_ids,
                "cell_block_indices": self.cell_block_indices,
                "cell_integer_coordinates": self.cell_integer_coordinates,
                "block_axes": tuple(
                    tuple(
                        tuple((value.numerator, value.denominator) for value in row)
                        for row in axes
                    )
                    for axes in self.block_axes
                ),
                "block_origins": tuple(
                    tuple((value.numerator, value.denominator) for value in origin)
                    for origin in self.block_origins
                ),
                "block_axis_knots": tuple(
                    tuple(
                        tuple((value.numerator, value.denominator) for value in axis)
                        for axis in chart
                    )
                    for chart in self.block_axis_knots
                ),
                "face_connections": self.face_connections,
                "face_transitions": self.face_transitions,
                "face_offsets": self.face_offsets,
                "boundary_face_parity": self.boundary_face_parity,
            }
        )


@final
class PreparedFrameHexGrid(StrictModule, NonTrainableState):
    """Source-bound PLC tetrahedral field, integer layout, or typed obstruction."""

    frame: HexFrameFieldSolution
    layout: HexFrameIntegerLayout | None
    grid: BalancedCutGridPlan | None
    obstruction: FrameFieldObstructionRecord | None
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        frame: HexFrameFieldSolution,
        layout: HexFrameIntegerLayout | None,
        grid: BalancedCutGridPlan | None,
        obstruction: FrameFieldObstructionRecord | None,
        /,
    ) -> None:
        if not isinstance(frame, HexFrameFieldSolution):
            raise TypeError("frame must be HexFrameFieldSolution.")
        if layout is not None and (
            not isinstance(layout, HexFrameIntegerLayout)
            or layout.frame_id != frame.frame_id
            or layout.source_complex_id != frame.source_complex_id
            or layout.source_mesh_id != frame.source_mesh_id
            or layout.source_topology_id != frame.source_topology_id
            or layout.schedule_id != frame.schedule_id
            or not np.array_equal(layout.cell_global_ids, frame.cell_global_ids)
            or not np.array_equal(layout.cell_block_indices, frame.cell_block_indices)
            or not np.array_equal(
                layout.cell_integer_coordinates, frame.cell_integer_coordinates
            )
            or not np.array_equal(
                np.asarray(layout.block_axes, dtype=np.float64)[
                    layout.cell_block_indices
                ],
                frame.cell_frames,
            )
            or not np.array_equal(
                np.asarray(layout.block_origins, dtype=np.float64)[
                    layout.cell_block_indices
                ],
                frame.cell_origins,
            )
        ):
            raise ValueError(
                "Frame integer layout must bind the exact solved field and source roots."
            )
        if grid is not None and (
            grid.frame_id != frame.frame_id
            or grid.source_complex_id != frame.source_complex_id
            or grid.source_mesh_id != frame.source_mesh_id
            or grid.source_id != frame.source_id
            or grid.source_revision != frame.source_revision
            or grid.schedule_id != frame.schedule_id
            or (layout is not None and grid.frame_layout_id != layout.layout_id)
        ):
            raise ValueError(
                "Frame cut grid must bind the exact solved field, block layout, and source revision."
            )
        if obstruction is not None and (
            not isinstance(obstruction, FrameFieldObstructionRecord)
            or obstruction.frame_id != frame.frame_id
            or layout is not None
            or grid is not None
        ):
            raise ValueError(
                "An obstruction must bind this frame field and cannot claim a layout or cut plan."
            )
        if layout is None and grid is None and obstruction is None:
            raise ValueError(
                "Prepared frame fields require a layout, cut plan, or exact obstruction."
            )
        self.frame = frame
        self.layout = layout
        self.grid = grid
        self.obstruction = obstruction
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-frame-field",
                "frame": frame.frame_id,
                "layout": None if layout is None else layout.layout_id,
                "grid": None if grid is None else grid.plan_id,
                "obstruction": None
                if obstruction is None
                else obstruction.obstruction_id,
            }
        )


@final
class PreparedSourceHexFrame(StrictModule, NonTrainableState):
    """Source-map Jacobians and root-face incidence for mapped tensor grids."""

    reference_mesh: CellMesh
    geometry: CellGeometrySpec
    root_cells: np.ndarray
    root_block_indices: np.ndarray
    root_block_rows: np.ndarray
    root_cell_ids: np.ndarray
    corner_jacobians: np.ndarray
    center_frames: np.ndarray
    axis_jacobian_bounds: np.ndarray
    face_connections: np.ndarray
    face_permutations: np.ndarray
    boundary_faces: np.ndarray
    source_geometry_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_mesh: CellMesh,
        geometry: CellGeometrySpec,
        root_cells: object,
        root_block_indices: object,
        root_block_rows: object,
        root_cell_ids: object,
        corner_jacobians: object,
        center_frames: object,
        axis_jacobian_bounds: object,
        face_connections: object,
        face_permutations: object,
        boundary_faces: object,
        /,
    ) -> None:
        cells = np.asarray(root_cells, dtype=np.int64)
        blocks = np.asarray(root_block_indices, dtype=np.int64)
        rows = np.asarray(root_block_rows, dtype=np.int64)
        identifiers = np.asarray(root_cell_ids, dtype=np.int64)
        corners = np.asarray(corner_jacobians, dtype=np.float64)
        frames = np.asarray(center_frames, dtype=np.float64)
        bounds = np.asarray(axis_jacobian_bounds, dtype=np.float64)
        connections = np.asarray(face_connections, dtype=np.int64)
        permutations_ = np.asarray(face_permutations, dtype=np.int64)
        boundary = np.asarray(boundary_faces, dtype=np.int64)
        roots = sum(block.cell_count for block in reference_mesh.blocks)
        if (
            not isinstance(reference_mesh, CellMesh)
            or not isinstance(geometry, CellGeometrySpec)
            or cells.shape != (roots, 8)
            or blocks.shape != (roots,)
            or rows.shape != (roots,)
            or identifiers.shape != (roots,)
            or corners.shape != (roots, 8, 3, 3)
            or frames.shape != (roots, 3, 3)
            or bounds.shape != (roots, 3)
            or connections.ndim != 2
            or connections.shape[1] != 4
            or permutations_.shape != (connections.shape[0], 4)
            or boundary.ndim != 2
            or boundary.shape[1] != 2
            or not all(np.all(np.isfinite(value)) for value in (corners, frames, bounds))
            or np.any(bounds <= 0.0)
        ):
            raise ValueError(
                "Prepared mapped source frame banks must cover exact hexahedral roots."
            )
        arrays = (
            cells,
            blocks,
            rows,
            identifiers,
            corners,
            frames,
            bounds,
            connections,
            permutations_,
            boundary,
        )
        frozen = []
        for value in arrays:
            copy = np.array(value, copy=True)
            copy.setflags(write=False)
            frozen.append(copy)
        (
            self.root_cells,
            self.root_block_indices,
            self.root_block_rows,
            self.root_cell_ids,
            self.corner_jacobians,
            self.center_frames,
            self.axis_jacobian_bounds,
            self.face_connections,
            self.face_permutations,
            self.boundary_faces,
        ) = frozen
        self.reference_mesh = reference_mesh
        self.geometry = geometry
        self.source_geometry_id = cell_geometry_id(geometry)
        self.frame_id = canonical_fingerprint(
            {
                "kind": "prepared-mapped-source-hex-frame",
                "mesh": reference_mesh.mesh_id,
                "geometry": self.source_geometry_id,
                "roots": self.root_cell_ids,
                "corner_jacobians": self.corner_jacobians,
                "center_frames": self.center_frames,
                "axis_jacobian_bounds": self.axis_jacobian_bounds,
                "face_connections": self.face_connections,
                "face_permutations": self.face_permutations,
                "boundary_faces": self.boundary_faces,
            }
        )


@final
class SourceBlockConnection(StrictModule, NonTrainableState):
    """Exact signed-permutation transport between shared root tensor faces."""

    transports: np.ndarray
    offsets: np.ndarray
    face_connections: np.ndarray
    frame_id: str = eqx.field(static=True)
    connection_id: str = eqx.field(static=True)

    def __init__(
        self,
        transports: object,
        offsets: object,
        face_connections: object,
        frame_id: str,
        /,
    ) -> None:
        maps = np.asarray(transports, dtype=np.int64)
        offsets_ = np.asarray(offsets, dtype=np.int64)
        connections = np.asarray(face_connections, dtype=np.int64)
        if (
            maps.ndim != 3
            or maps.shape[1:] != (3, 3)
            or offsets_.shape != (maps.shape[0], 3)
            or connections.shape != (maps.shape[0], 4)
            or not frame_id
            or any(
                not np.array_equal(
                    np.sum(np.abs(matrix), axis=0), np.ones((3,), dtype=np.int64)
                )
                or not np.array_equal(
                    np.sum(np.abs(matrix), axis=1), np.ones((3,), dtype=np.int64)
                )
                for matrix in maps
            )
        ):
            raise ValueError(
                "Source block connections must be signed tensor-axis permutations."
            )
        self.transports = np.array(maps, copy=True)
        self.offsets = np.array(offsets_, copy=True)
        self.face_connections = np.array(connections, copy=True)
        self.transports.setflags(write=False)
        self.offsets.setflags(write=False)
        self.face_connections.setflags(write=False)
        self.frame_id = str(frame_id)
        self.connection_id = canonical_fingerprint(
            {
                "kind": "mapped-source-block-connection",
                "frame": self.frame_id,
                "faces": self.face_connections,
                "transports": self.transports,
                "offsets": self.offsets,
            }
        )


__all__ = [
    "FrameFieldObstructionRecord",
    "FrameFieldRefusal",
    "HexFrameFieldSolution",
    "HexFrameIntegerLayout",
    "PreparedFrameHexGrid",
    "PreparedSourceHexFrame",
    "SourceBlockConnection",
]
