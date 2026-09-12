#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import combinations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, PolyhedralConnectivity
from ..discretization._reference_cell import reference_cell_topology


class CellQualityEvaluation(StrictModule):
    cell_global_ids: Array
    measures: Array
    mean_ratios: Array
    aspect_ratios: Array
    valid: Array
    block_names: tuple[str, ...] = eqx.field(static=True)
    block_offsets: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)


class CellQualityReport(StrictModule, NonTrainableState):
    evaluation: CellQualityEvaluation
    minimum_measure: float = eqx.field(static=True)
    maximum_measure: float = eqx.field(static=True)
    minimum_mean_ratio: float = eqx.field(static=True)
    maximum_aspect_ratio: float = eqx.field(static=True)
    invalid_count: int = eqx.field(static=True)
    worst_cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(self, evaluation: CellQualityEvaluation, /):
        if not isinstance(evaluation, CellQualityEvaluation):
            raise TypeError("evaluation must be CellQualityEvaluation.")
        measures = np.asarray(evaluation.measures, dtype=float)
        ratios = np.asarray(evaluation.mean_ratios, dtype=float)
        aspects = np.asarray(evaluation.aspect_ratios, dtype=float)
        valid = np.asarray(evaluation.valid, dtype=bool)
        identifiers = np.asarray(evaluation.cell_global_ids, dtype=np.int64)
        invalid = int(np.count_nonzero(~valid))
        order = np.argsort(np.where(valid, ratios, -np.inf), kind="stable")
        worst = tuple(int(identifiers[index]) for index in order[: min(10, order.size)])
        self.evaluation = evaluation
        self.minimum_measure = float(np.min(measures))
        self.maximum_measure = float(np.max(measures))
        self.minimum_mean_ratio = float(np.min(ratios))
        self.maximum_aspect_ratio = float(np.max(aspects))
        self.invalid_count = invalid
        self.worst_cell_global_ids = worst
        self.report_id = canonical_fingerprint(
            {
                "kind": "cell-quality-report",
                "evaluation": evaluation.evaluation_id,
                "measures": array_tree_fingerprint(measures),
                "mean_ratios": array_tree_fingerprint(ratios),
                "aspect_ratios": array_tree_fingerprint(aspects),
                "valid": array_tree_fingerprint(valid),
            }
        )


def _lengths(points: Array, edges: tuple[tuple[int, int], ...], /) -> Array:
    return jnp.stack(
        tuple(
            jnp.linalg.norm(points[:, stop] - points[:, start], axis=-1)
            for start, stop in edges
        ),
        axis=1,
    )


def _triangle_quality(points: Array, /) -> tuple[Array, Array, Array, Array]:
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    if points.shape[-1] == 2:
        signed_double_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        double_area = jnp.abs(signed_double_area)
        orientation = signed_double_area > 0.0
    else:
        double_area = jnp.linalg.norm(jnp.cross(first, second), axis=-1)
        orientation = jnp.ones(double_area.shape, dtype=bool)
    lengths = _lengths(points, ((0, 1), (1, 2), (2, 0)))
    squared_sum = jnp.sum(lengths * lengths, axis=1)
    mean_ratio = (
        2.0
        * jnp.sqrt(3.0)
        * double_area
        / jnp.maximum(squared_sum, jnp.finfo(points.dtype).tiny)
    )
    aspect = jnp.max(lengths, axis=1) / jnp.maximum(
        jnp.min(lengths, axis=1), jnp.finfo(points.dtype).tiny
    )
    measure = 0.5 * double_area
    valid = orientation & jnp.isfinite(measure) & (measure > 0.0)
    return measure, mean_ratio, aspect, valid


def _tetra_quality(points: Array, /) -> tuple[Array, Array, Array, Array]:
    matrix = jnp.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=-1,
    )
    determinant = jnp.linalg.det(matrix)
    volume = determinant / 6.0
    edges = tuple(combinations(range(4), 2))
    lengths = _lengths(points, edges)
    squared_sum = jnp.sum(lengths * lengths, axis=1)
    positive_volume = jnp.maximum(volume, 0.0)
    mean_ratio = (
        12.0
        * jnp.power(3.0 * positive_volume, 2.0 / 3.0)
        / jnp.maximum(squared_sum, jnp.finfo(points.dtype).tiny)
    )
    aspect = jnp.max(lengths, axis=1) / jnp.maximum(
        jnp.min(lengths, axis=1), jnp.finfo(points.dtype).tiny
    )
    valid = jnp.isfinite(volume) & (volume > 0.0)
    return volume, mean_ratio, aspect, valid


def _composite_volume_quality(
    points: Array,
    tetrahedra: tuple[tuple[int, int, int, int], ...],
    edges: tuple[tuple[int, int], ...],
    /,
) -> tuple[Array, Array, Array, Array]:
    measures = []
    ratios = []
    valid = []
    for route in tetrahedra:
        measure, ratio, _, valid_ = _tetra_quality(points[:, route, :])
        measures.append(measure)
        ratios.append(ratio)
        valid.append(valid_)
    edge_lengths = _lengths(points, edges)
    volume = jnp.sum(jnp.stack(measures, axis=1), axis=1)
    mean_ratio = jnp.min(jnp.stack(ratios, axis=1), axis=1)
    aspect = jnp.max(edge_lengths, axis=1) / jnp.maximum(
        jnp.min(edge_lengths, axis=1), jnp.finfo(points.dtype).tiny
    )
    validity = jnp.all(jnp.stack(valid, axis=1), axis=1) & (volume > 0.0)
    return volume, mean_ratio, aspect, validity


def _standard_block_quality(kind: str, points: Array, /):
    if kind == "interval":
        length = jnp.linalg.norm(points[:, 1] - points[:, 0], axis=-1)
        return length, jnp.ones_like(length), jnp.ones_like(length), length > 0.0
    if kind == "triangle":
        return _triangle_quality(points)
    if kind in ("quadrilateral", "polygon"):
        triangle_measures = []
        triangle_ratios = []
        triangle_valid = []
        for index in range(1, points.shape[1] - 1):
            measure, ratio, _, valid = _triangle_quality(
                points[:, (0, index, index + 1), :]
            )
            triangle_measures.append(measure)
            triangle_ratios.append(ratio)
            triangle_valid.append(valid)
        edges = tuple(
            (index, (index + 1) % points.shape[1]) for index in range(points.shape[1])
        )
        lengths = _lengths(points, edges)
        measure = jnp.sum(jnp.stack(triangle_measures, axis=1), axis=1)
        ratio = jnp.min(jnp.stack(triangle_ratios, axis=1), axis=1)
        aspect = jnp.max(lengths, axis=1) / jnp.maximum(
            jnp.min(lengths, axis=1), jnp.finfo(points.dtype).tiny
        )
        valid = jnp.all(jnp.stack(triangle_valid, axis=1), axis=1)
        return measure, ratio, aspect, valid
    if kind == "tetrahedron":
        return _tetra_quality(points)
    topology = reference_cell_topology(kind)
    edges = tuple((int(start), int(stop)) for start, stop in topology.entities[1])
    if kind == "pyramid":
        tetrahedra = ((0, 1, 2, 4), (0, 2, 3, 4))
    elif kind == "prism":
        tetrahedra = ((0, 1, 2, 3), (1, 4, 2, 3), (2, 4, 5, 3))
    elif kind == "hexahedron":
        tetrahedra = (
            (0, 1, 2, 6),
            (0, 2, 3, 6),
            (0, 3, 7, 6),
            (0, 7, 4, 6),
            (0, 4, 5, 6),
            (0, 5, 1, 6),
        )
    else:
        raise ValueError(f"No native quality implementation for {kind!r}.")
    return _composite_volume_quality(points, tetrahedra, edges)


def _polyhedral_quality(mesh: CellMesh, coordinates: Array, /):
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral quality requires PolyhedralConnectivity.")
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int32)
    face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int32)
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int32)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int32)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=float)
    cell_vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int32)
    cell_vertices = np.asarray(connectivity.cell_vertex_values, dtype=np.int32)
    measures = []
    ratios = []
    aspects = []
    validity = []
    for cell in range(connectivity.cell_count):
        vertex_start, vertex_stop = cell_vertex_offsets[cell : cell + 2]
        vertices = cell_vertices[int(vertex_start) : int(vertex_stop)]
        star = jnp.mean(coordinates[vertices], axis=0)
        face_start, face_stop = cell_face_offsets[cell : cell + 2]
        cell_measures = []
        cell_ratios = []
        cell_valid = []
        cell_edges: set[tuple[int, int]] = set()
        for face_index, sign in zip(
            cell_faces[int(face_start) : int(face_stop)],
            cell_signs[int(face_start) : int(face_stop)],
            strict=True,
        ):
            start, stop = face_offsets[int(face_index) : int(face_index) + 2]
            face = face_values[int(start) : int(stop)]
            oriented = face if sign > 0.0 else face[::-1]
            for local, first in enumerate(oriented):
                second = int(oriented[(local + 1) % oriented.size])
                cell_edges.add((min(int(first), second), max(int(first), second)))
            for local in range(1, oriented.size - 1):
                tetra = jnp.stack(
                    (
                        star,
                        coordinates[int(oriented[0])],
                        coordinates[int(oriented[local])],
                        coordinates[int(oriented[local + 1])],
                    )
                )[None, :, :]
                measure, ratio, _, valid = _tetra_quality(tetra)
                cell_measures.append(measure[0])
                cell_ratios.append(ratio[0])
                cell_valid.append(valid[0])
        lengths = jnp.stack(
            tuple(
                jnp.linalg.norm(coordinates[stop] - coordinates[start])
                for start, stop in sorted(cell_edges)
            )
        )
        measures.append(jnp.sum(jnp.stack(cell_measures)))
        ratios.append(jnp.min(jnp.stack(cell_ratios)))
        aspects.append(
            jnp.max(lengths)
            / jnp.maximum(jnp.min(lengths), jnp.finfo(coordinates.dtype).tiny)
        )
        validity.append(jnp.all(jnp.stack(cell_valid)))
    return (
        jnp.stack(measures),
        jnp.stack(ratios),
        jnp.stack(aspects),
        jnp.stack(validity),
    )


class SweptLayerQualityEvaluation(StrictModule, NonTrainableState):
    """Measured straight-sweep quality for one exact layer schedule."""

    measured_thicknesses: Array
    measured_growth_rates: Array
    maximum_thickness_residual: float = eqx.field(static=True)
    maximum_alignment_residual: float = eqx.field(static=True)
    maximum_interface_residual: float = eqx.field(static=True)
    valid: bool = eqx.field(static=True)


def evaluate_swept_layer_quality(
    points: ArrayLike,
    prisms: ArrayLike,
    layer_indices: ArrayLike,
    origin: ArrayLike,
    unit_direction: ArrayLike,
    requested_thicknesses: ArrayLike,
    /,
) -> SweptLayerQualityEvaluation:
    """Measure thickness, growth, axial alignment, and interface placement."""

    coordinates = np.asarray(points, dtype=float)
    cells = np.asarray(prisms, dtype=np.int32)
    indices = np.asarray(layer_indices, dtype=np.int32)
    anchor = np.asarray(origin, dtype=float)
    direction = np.asarray(unit_direction, dtype=float)
    requested = np.asarray(requested_thicknesses, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Sweep quality points must have shape (n, 3).")
    if (
        cells.ndim != 2
        or cells.shape[1] != 6
        or indices.shape != cells.shape[:1]
        or np.any(cells < 0)
        or np.any(cells >= coordinates.shape[0])
    ):
        raise ValueError(
            "Sweep quality requires one valid six-node prism index per cell."
        )
    if requested.ndim != 1 or requested.size == 0 or np.any(requested <= 0.0):
        raise ValueError("Requested sweep thicknesses must be one positive vector.")
    if np.any(indices < 0) or np.any(indices >= requested.size):
        raise ValueError("Sweep layer indices are outside the requested schedule.")
    norm = float(np.linalg.norm(direction))
    if (
        anchor.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite(norm)
        or norm <= 0.0
    ):
        raise ValueError(
            "Sweep quality requires a finite nonzero three-vector direction."
        )
    unit = direction / norm
    values = coordinates[cells]
    axial_edges = values[:, 3:] - values[:, :3]
    projected_edges = axial_edges @ unit
    edge_lengths = np.linalg.norm(axial_edges, axis=2)
    transverse = axial_edges - projected_edges[:, :, None] * unit
    alignment = np.linalg.norm(transverse, axis=2) / np.maximum(
        edge_lengths, np.finfo(float).tiny
    )
    levels = np.concatenate(([0.0], np.cumsum(requested)))
    projected_vertices = (values - anchor) @ unit
    expected_lower = levels[indices]
    expected_upper = levels[indices + 1]
    interface_residual = float(
        max(
            np.max(np.abs(projected_vertices[:, :3] - expected_lower[:, None])),
            np.max(np.abs(projected_vertices[:, 3:] - expected_upper[:, None])),
        )
    )
    measured = np.full(requested.shape, np.nan, dtype=float)
    for layer in range(requested.size):
        selected = indices == layer
        if np.any(selected):
            measured[layer] = float(np.mean(projected_edges[selected]))
    growth = (
        measured[1:] / measured[:-1] if measured.size > 1 else np.empty((0,), dtype=float)
    )
    thickness_residual = float(np.max(np.abs(projected_edges - requested[indices, None])))
    alignment_residual = float(np.max(alignment))
    valid = bool(
        np.all(np.isfinite(measured))
        and np.all(measured > 0.0)
        and np.isfinite(thickness_residual)
        and np.isfinite(alignment_residual)
        and np.isfinite(interface_residual)
    )
    return SweptLayerQualityEvaluation(
        measured_thicknesses=jnp.asarray(measured),
        measured_growth_rates=jnp.asarray(growth),
        maximum_thickness_residual=thickness_residual,
        maximum_alignment_residual=alignment_residual,
        maximum_interface_residual=interface_residual,
        valid=valid,
    )


def evaluate_cell_quality(
    mesh: CellMesh,
    coordinates: ArrayLike | None = None,
    /,
) -> CellQualityEvaluation:
    """Evaluate differentiable fixed-topology quality for every cell block."""

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    points = mesh.coordinates if coordinates is None else jnp.asarray(coordinates)
    if points.shape != mesh.coordinates.shape:
        raise ValueError("Quality coordinates must preserve the mesh coordinate shape.")
    measures = []
    ratios = []
    aspects = []
    valid = []
    offsets = [0]
    if any(block.cell_kind == "polyhedron" for block in mesh.blocks):
        poly_measure, poly_ratio, poly_aspect, poly_valid = _polyhedral_quality(
            mesh, points
        )
        cursor = 0
        for block in mesh.blocks:
            stop = cursor + block.cell_count
            if block.cell_kind == "polyhedron":
                measures.append(poly_measure[cursor:stop])
                ratios.append(poly_ratio[cursor:stop])
                aspects.append(poly_aspect[cursor:stop])
                valid.append(poly_valid[cursor:stop])
            else:
                values = points[jnp.asarray(block.vertices, dtype=jnp.int32)]
                measure, ratio, aspect, valid_ = _standard_block_quality(
                    block.cell_kind, values
                )
                measures.append(measure)
                ratios.append(ratio)
                aspects.append(aspect)
                valid.append(valid_)
            cursor = stop
            offsets.append(stop)
    else:
        for block in mesh.blocks:
            values = points[jnp.asarray(block.vertices, dtype=jnp.int32)]
            measure, ratio, aspect, valid_ = _standard_block_quality(
                block.cell_kind, values
            )
            measures.append(measure)
            ratios.append(ratio)
            aspects.append(aspect)
            valid.append(valid_)
            offsets.append(offsets[-1] + block.cell_count)
    evaluation_id = canonical_fingerprint(
        {
            "kind": "cell-quality-evaluation",
            "topology": mesh.topology_id,
            "block_ids": [block.block_id for block in mesh.blocks],
        }
    )
    return CellQualityEvaluation(
        cell_global_ids=jnp.concatenate(tuple(block.global_ids for block in mesh.blocks)),
        measures=jnp.concatenate(tuple(measures)),
        mean_ratios=jnp.concatenate(tuple(ratios)),
        aspect_ratios=jnp.concatenate(tuple(aspects)),
        valid=jnp.concatenate(tuple(valid)),
        block_names=tuple(block.name for block in mesh.blocks),
        block_offsets=tuple(offsets),
        topology_id=mesh.topology_id,
        evaluation_id=evaluation_id,
    )


def summarize_cell_quality(evaluation: CellQualityEvaluation, /) -> CellQualityReport:
    return CellQualityReport(evaluation)


__all__ = [
    "CellQualityEvaluation",
    "CellQualityReport",
    "evaluate_cell_quality",
    "summarize_cell_quality",
]
