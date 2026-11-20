#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable sampled cell quality and finite-volume face quality.

Quality is sampled at cell corners (and star simplices for polygons and
polyhedra); ``sampled_valid`` is therefore evidence, not a validity proof. The
Bernstein certificate in ``phydrax.discretization._cell_geometry_validity`` is
the validity owner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import cache
from math import prod
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._host_table_cache import host_table_cache, logical_table_bytes
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, PolyhedralConnectivity
from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    CellValidityCertificate,
    polyhedral_star_tables,
    PolyhedralStarTables,
)
from ..discretization._cell_mesh import SimplicialConnectivity
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._motion_validity import reference_corner_frames
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._interval_enclosure import (
    Interval,
    interval_add,
    interval_divide,
    interval_multiply,
    interval_square,
    interval_subtract,
)
from ..geometry._rational import (
    binary64_arctangent2,
    binary64_cube_root,
    binary64_dot,
    binary64_matmul,
    binary64_norm,
    fraction_down,
    fraction_up,
    rational_cross,
    rational_dot,
    rational_vector,
)
from ..typing import checked
from ._metric import interpolate_mesh_metric, MeshMetricField


class CellQualityEvaluation(StrictModule):
    """Per-cell quality in mesh block order.

    Angles are radians: interior corner angles for 2-cells and interior
    dihedral angles for 3-cells (above pi at reflex corners/edges). Mean ratio,
    scaled Jacobian, condition number, and metric quality are corner samples of
    the Jacobian relative to the ideal cell (equilateral simplex, unit square or
    cube, regular prism, pyramid, or polygon). ``radius_ratios`` (simplices) and
    ``sliver_measures`` (tetrahedra) are NaN for other kinds; ``metric_quality``
    is NaN without a metric. ``warpage`` is the largest angle between a face's
    centroid-fan triangle normals and its vector-area normal.
    Audits of higher-order or nonvertex coordinate elements replace
    corner-carrier measures with certified physical measures and retain their
    absolute error bounds, exactness status, and geometry ID.
    """

    cell_global_ids: Array
    measures: Array
    mean_ratios: Array
    aspect_ratios: Array
    radius_ratios: Array
    scaled_jacobian: Array
    minimum_angle: Array
    maximum_angle: Array
    condition_number: Array
    sliver_measures: Array
    warpage: Array
    metric_quality: Array
    sampled_valid: Array
    block_names: tuple[str, ...] = eqx.field(static=True)
    block_offsets: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)
    measure_absolute_errors: Array | None = None
    measure_geometry_id: str | None = eqx.field(static=True, default=None)
    measure_exact: bool | None = eqx.field(static=True, default=None)


def _nan_extreme(values: np.ndarray, reducer: Any, /) -> float:
    finite = values[~np.isnan(values)]
    return float(reducer(finite)) if finite.size else float("nan")


def _triangle_minimum_sine_violations(
    coordinates: np.ndarray, triangles: np.ndarray, sine_upper: Fraction, /
) -> np.ndarray:
    """Decide physical triangle minimum-angle bounds on represented coordinates.

    Directed interval arithmetic decides easy faces in bounded batches. An
    undecided face is resolved over exact binary64 rationals; no uncertain
    screen result is accepted as a proof.
    """
    if (
        coordinates.ndim != 2
        or coordinates.dtype != np.dtype(np.float64)
        or coordinates.shape[1] != 3
        or triangles.ndim != 2
        or not np.issubdtype(triangles.dtype, np.integer)
        or triangles.shape[1] != 3
        or not np.all(np.isfinite(coordinates))
        or sine_upper < 0
        or sine_upper > 1
        or np.any(triangles < 0)
        or np.any(triangles >= len(coordinates))
    ):
        raise ValueError(
            "Triangle quality requires finite represented 3D points and a sine bound."
        )
    result = np.empty((len(triangles),), dtype=np.bool_)

    def arrays(value: Interval, /) -> tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(value[0], dtype=np.float64)
        upper = np.asarray(value[1], dtype=np.float64)
        if lower.shape != upper.shape:
            raise ValueError(
                "Triangle quality interval endpoints must have matching shapes."
            )
        return lower, upper

    def norm_squared(
        value: tuple[np.ndarray, np.ndarray], /
    ) -> tuple[np.ndarray, np.ndarray]:
        square = arrays(interval_square(value))
        if square[0].ndim != 2 or square[0].shape[1] != 3:
            raise ValueError("Triangle quality vectors must have three coordinates.")
        return arrays(
            interval_add(
                interval_add(
                    (square[0][:, 0], square[1][:, 0]),
                    (square[0][:, 1], square[1][:, 1]),
                ),
                (square[0][:, 2], square[1][:, 2]),
            )
        )

    sine = (
        np.asarray(fraction_down(sine_upper), dtype=np.float64),
        np.asarray(fraction_up(sine_upper), dtype=np.float64),
    )
    sine_squared = interval_square(sine)
    for start in range(0, len(triangles), 4096):
        stop = min(start + 4096, len(triangles))
        points = coordinates[triangles[start:stop]]
        first = arrays(
            interval_subtract((points[:, 1], points[:, 1]), (points[:, 0], points[:, 0]))
        )
        second = arrays(
            interval_subtract((points[:, 2], points[:, 2]), (points[:, 0], points[:, 0]))
        )
        third = arrays(
            interval_subtract((points[:, 2], points[:, 2]), (points[:, 1], points[:, 1]))
        )
        lengths = tuple(norm_squared(edge) for edge in (first, second, third))
        cross = arrays(
            interval_subtract(
                interval_multiply(
                    (first[0][:, (1, 2, 0)], first[1][:, (1, 2, 0)]),
                    (second[0][:, (2, 0, 1)], second[1][:, (2, 0, 1)]),
                ),
                interval_multiply(
                    (first[0][:, (2, 0, 1)], first[1][:, (2, 0, 1)]),
                    (second[0][:, (1, 2, 0)], second[1][:, (1, 2, 0)]),
                ),
            )
        )
        cross_squared = norm_squared(cross)
        minimum = (
            np.minimum.reduce(tuple(length[0] for length in lengths)),
            np.minimum.reduce(tuple(length[1] for length in lengths)),
        )
        left = arrays(
            interval_multiply(
                interval_multiply(interval_multiply(lengths[0], lengths[1]), lengths[2]),
                sine_squared,
            )
        )
        right = arrays(interval_multiply(minimum, cross_squared))
        valid = (left[1] <= right[0]) & (cross_squared[0] > 0) & (minimum[0] > 0)
        invalid = (left[0] > right[1]) | (cross_squared[1] <= 0) | (minimum[1] <= 0)
        result[start:stop] = invalid
        for row in np.flatnonzero(~(valid | invalid)):
            a, b, c = (rational_vector(point) for point in points[row])
            first_exact = tuple(y - x for x, y in zip(a, b, strict=True))
            second_exact = tuple(y - x for x, y in zip(a, c, strict=True))
            third_exact = tuple(y - x for x, y in zip(b, c, strict=True))
            exact_lengths = tuple(
                rational_dot(edge, edge)
                for edge in (first_exact, second_exact, third_exact)
            )
            normal = rational_cross(first_exact, second_exact)
            area_squared = rational_dot(normal, normal)
            result[start + row] = (
                area_squared <= 0
                or exact_lengths[0] * exact_lengths[1] * exact_lengths[2] * sine_upper**2
                > min(exact_lengths) * area_squared
            )
    return result


class CellQualityReport(StrictModule, NonTrainableState):
    """Sampled quality diagnostics with exact threshold decisions.

    Concrete evaluation arrays and float summaries are diagnostics; traced
    evaluations use JAX arithmetic and may round differently. ``report_id``
    hashes the mesh, the evaluation request, the gate thresholds, and
    the exactly decided failing cell IDs of each gate (see
    :func:`summarize_cell_quality`). ``gate_thresholds`` is
    ``(minimum_measure, minimum_mean_ratio, maximum_aspect_ratio)``.
    """

    evaluation: CellQualityEvaluation
    minimum_measure: float = eqx.field(static=True)
    maximum_measure: float = eqx.field(static=True)
    minimum_mean_ratio: float = eqx.field(static=True)
    maximum_aspect_ratio: float = eqx.field(static=True)
    minimum_scaled_jacobian: float = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    minimum_angle: float = eqx.field(static=True)
    maximum_angle: float = eqx.field(static=True)
    maximum_warpage: float = eqx.field(static=True)
    minimum_metric_quality: float = eqx.field(static=True)
    sampled_invalid_count: int = eqx.field(static=True)
    worst_cell_global_ids: tuple[int, ...] = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    gate_thresholds: tuple[float, float, float] = eqx.field(static=True)
    measure_failure_ids: tuple[int, ...] = eqx.field(static=True)
    mean_ratio_failure_ids: tuple[int, ...] = eqx.field(static=True)
    aspect_ratio_failure_ids: tuple[int, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        evaluation: CellQualityEvaluation,
        /,
        *,
        mesh_id: str,
        gate_thresholds: tuple[float, float, float],
        measure_failure_ids: tuple[int, ...],
        mean_ratio_failure_ids: tuple[int, ...],
        aspect_ratio_failure_ids: tuple[int, ...],
    ) -> None:
        measures = np.asarray(evaluation.measures, dtype=np.float64)
        ratios = np.asarray(evaluation.mean_ratios, dtype=np.float64)
        aspects = np.asarray(evaluation.aspect_ratios, dtype=np.float64)
        scaled = np.asarray(evaluation.scaled_jacobian, dtype=np.float64)
        condition = np.asarray(evaluation.condition_number, dtype=np.float64)
        minimum_angle = np.asarray(evaluation.minimum_angle, dtype=np.float64)
        maximum_angle = np.asarray(evaluation.maximum_angle, dtype=np.float64)
        warpage = np.asarray(evaluation.warpage, dtype=np.float64)
        metric = np.asarray(evaluation.metric_quality, dtype=np.float64)
        valid = np.asarray(evaluation.sampled_valid, dtype=np.bool_)
        identifiers = np.asarray(evaluation.cell_global_ids, dtype=np.int64)
        order = np.argsort(np.where(valid, ratios, -np.inf), kind="stable")
        self.evaluation = evaluation
        errors = (
            np.zeros_like(measures)
            if evaluation.measure_absolute_errors is None
            else np.asarray(evaluation.measure_absolute_errors, dtype=np.float64)
        )
        if (evaluation.measure_absolute_errors is None) != (
            evaluation.measure_geometry_id is None
        ):
            raise ValueError("Mapped quality measures require their geometry identity.")
        if (evaluation.measure_geometry_id is None) != (
            evaluation.measure_exact is None
        ) or (
            evaluation.measure_exact is not None
            and type(evaluation.measure_exact) is not bool
        ):
            raise ValueError("Mapped quality measures require their exactness status.")
        if (
            errors.shape != measures.shape
            or np.any(errors < 0.0)
            or not np.all(np.isfinite(errors))
        ):
            raise ValueError(
                "Mapped quality measures require finite aligned error bounds."
            )
        self.minimum_measure = float(np.min(measures - errors))
        self.maximum_measure = float(np.max(measures + errors))
        self.minimum_mean_ratio = float(np.min(ratios))
        self.maximum_aspect_ratio = float(np.max(aspects))
        self.minimum_scaled_jacobian = float(np.min(scaled))
        self.maximum_condition_number = float(np.max(condition))
        self.minimum_angle = _nan_extreme(minimum_angle, np.min)
        self.maximum_angle = _nan_extreme(maximum_angle, np.max)
        self.maximum_warpage = float(np.max(warpage))
        self.minimum_metric_quality = _nan_extreme(metric, np.min)
        self.sampled_invalid_count = int(np.count_nonzero(~valid))
        self.worst_cell_global_ids = tuple(
            int(identifiers[index]) for index in order[: min(10, order.size)]
        )
        thresholds = tuple(float(value) for value in gate_thresholds)
        if len(thresholds) != 3:
            raise ValueError("Quality gate thresholds are (measure, mean ratio, aspect).")
        _require_quality_thresholds(_GateThresholds(*thresholds))
        if not isinstance(mesh_id, str) or not mesh_id:
            raise ValueError("Quality report requires the summarized mesh identity.")
        known = set(identifiers.tolist())
        failure_sets = tuple(
            tuple(int(value) for value in values)
            for values in (
                measure_failure_ids,
                mean_ratio_failure_ids,
                aspect_ratio_failure_ids,
            )
        )
        if any(
            list(values) != sorted(set(values)) or not set(values) <= known
            for values in failure_sets
        ):
            raise ValueError(
                "Quality gate failures must be sorted unique evaluated cell IDs."
            )
        self.mesh_id = mesh_id
        self.gate_thresholds = (thresholds[0], thresholds[1], thresholds[2])
        (
            self.measure_failure_ids,
            self.mean_ratio_failure_ids,
            self.aspect_ratio_failure_ids,
        ) = failure_sets
        # Diagnostics are excluded: only the request and exact decisions identify.
        self.report_id = canonical_fingerprint(
            {
                "kind": "cell-quality-report",
                "mesh": mesh_id,
                "evaluation": evaluation.evaluation_id,
                **(
                    {"measure_exact": evaluation.measure_exact}
                    if evaluation.measure_exact is not None
                    else {}
                ),
                "thresholds": {
                    "minimum_measure": thresholds[0],
                    "minimum_mean_ratio": thresholds[1],
                    "maximum_aspect_ratio": thresholds[2],
                },
                "failures": {
                    "minimum_measure": failure_sets[0],
                    "minimum_mean_ratio": failure_sets[1],
                    "maximum_aspect_ratio": failure_sets[2],
                },
            }
        )

    @property
    def failed_gates(self) -> tuple[str, ...]:
        """Names of the threshold gates with at least one exactly failing cell."""
        return tuple(
            name
            for name, failures in (
                ("minimum_measure", self.measure_failure_ids),
                ("minimum_mean_ratio", self.mean_ratio_failure_ids),
                ("maximum_aspect_ratio", self.aspect_ratio_failure_ids),
            )
            if failures
        )

    @classmethod
    def from_stored_fields(
        cls, value: CellQualityReport | CellQualityReportArchiveRecipe, /
    ) -> CellQualityReport:
        if type(value) is CellQualityReport:
            stored = CellQualityReportArchiveRecipe.from_report(value)
        elif type(value) is CellQualityReportArchiveRecipe:
            stored = value
        else:
            raise TypeError(
                "value must be CellQualityReport or CellQualityReportArchiveRecipe."
            )
        report = cls(
            stored.evaluation,
            mesh_id=stored.mesh_id,
            gate_thresholds=stored.gate_thresholds,
            measure_failure_ids=stored.measure_failure_ids,
            mean_ratio_failure_ids=stored.mean_ratio_failure_ids,
            aspect_ratio_failure_ids=stored.aspect_ratio_failure_ids,
        )
        expected = CellQualityReportArchiveRecipe.from_report(report)
        expected_summary = (
            expected.minimum_measure,
            expected.maximum_measure,
            expected.minimum_mean_ratio,
            expected.maximum_aspect_ratio,
            expected.minimum_scaled_jacobian,
            expected.maximum_condition_number,
            expected.minimum_angle,
            expected.maximum_angle,
            expected.maximum_warpage,
            expected.minimum_metric_quality,
            expected.sampled_invalid_count,
            expected.worst_cell_global_ids,
            expected.report_id,
        )
        stored_summary = (
            stored.minimum_measure,
            stored.maximum_measure,
            stored.minimum_mean_ratio,
            stored.maximum_aspect_ratio,
            stored.minimum_scaled_jacobian,
            stored.maximum_condition_number,
            stored.minimum_angle,
            stored.maximum_angle,
            stored.maximum_warpage,
            stored.minimum_metric_quality,
            stored.sampled_invalid_count,
            stored.worst_cell_global_ids,
            stored.report_id,
        )
        if (
            array_tree_fingerprint(stored.evaluation)
            != array_tree_fingerprint(report.evaluation)
            or expected_summary != stored_summary
        ):
            raise ValueError(
                "Stored quality report differs from recomputed evaluation evidence."
            )
        return report


def _finite_quality_summary(value: float, name: str, /) -> float:
    if not np.isfinite(value):
        raise ValueError(f"CellQualityReport.{name} must be finite.")
    return value


def _nonapplicable_quality_summary(
    value: float, samples: Array, name: str, /
) -> float | None:
    if np.isnan(value):
        sample_values = np.asarray(samples, dtype=np.float64)
        if sample_values.size and np.all(np.isnan(sample_values)):
            return None
        raise ValueError(
            f"CellQualityReport.{name} is not a canonical non-applicable summary."
        )
    if not np.isfinite(value):
        raise ValueError(f"CellQualityReport.{name} must be finite.")
    return value


class CellQualityReportArchiveRecipe(NamedTuple):
    """Quality summaries with None reserved for all-NaN non-applicable samples.

    As a NamedTuple the recipe is a PyTree, so archives inventory and restore
    the evaluation arrays as dynamic leaves.
    """

    evaluation: CellQualityEvaluation
    minimum_measure: float
    maximum_measure: float
    minimum_mean_ratio: float
    maximum_aspect_ratio: float
    minimum_scaled_jacobian: float
    maximum_condition_number: float
    minimum_angle: float | None
    maximum_angle: float | None
    maximum_warpage: float
    minimum_metric_quality: float | None
    sampled_invalid_count: int
    worst_cell_global_ids: tuple[int, ...]
    mesh_id: str
    gate_thresholds: tuple[float, float, float]
    measure_failure_ids: tuple[int, ...]
    mean_ratio_failure_ids: tuple[int, ...]
    aspect_ratio_failure_ids: tuple[int, ...]
    report_id: str

    @classmethod
    def from_report(cls, value: CellQualityReport, /) -> CellQualityReportArchiveRecipe:
        if type(value) is not CellQualityReport:
            raise TypeError("value must be CellQualityReport.")
        return cls(
            value.evaluation,
            _finite_quality_summary(value.minimum_measure, "minimum_measure"),
            _finite_quality_summary(value.maximum_measure, "maximum_measure"),
            _finite_quality_summary(value.minimum_mean_ratio, "minimum_mean_ratio"),
            _finite_quality_summary(value.maximum_aspect_ratio, "maximum_aspect_ratio"),
            _finite_quality_summary(
                value.minimum_scaled_jacobian, "minimum_scaled_jacobian"
            ),
            _finite_quality_summary(
                value.maximum_condition_number, "maximum_condition_number"
            ),
            _nonapplicable_quality_summary(
                value.minimum_angle, value.evaluation.minimum_angle, "minimum_angle"
            ),
            _nonapplicable_quality_summary(
                value.maximum_angle, value.evaluation.maximum_angle, "maximum_angle"
            ),
            _finite_quality_summary(value.maximum_warpage, "maximum_warpage"),
            _nonapplicable_quality_summary(
                value.minimum_metric_quality,
                value.evaluation.metric_quality,
                "minimum_metric_quality",
            ),
            value.sampled_invalid_count,
            value.worst_cell_global_ids,
            value.mesh_id,
            value.gate_thresholds,
            value.measure_failure_ids,
            value.mean_ratio_failure_ids,
            value.aspect_ratio_failure_ids,
            value.report_id,
        )


class _BlockQuality(NamedTuple):
    measure: Array | np.ndarray
    mean_ratio: Array | np.ndarray
    aspect: Array | np.ndarray
    radius_ratio: Array | np.ndarray
    scaled_jacobian: Array | np.ndarray
    minimum_angle: Array | np.ndarray
    maximum_angle: Array | np.ndarray
    condition: Array | np.ndarray
    sliver: Array | np.ndarray
    warpage: Array | np.ndarray
    metric_quality: Array | np.ndarray
    sampled_valid: Array | np.ndarray


# Static reference tables -----------------------------------------------------------


def _ideal_vertices(kind: str, arity: int, /) -> np.ndarray:
    root3 = np.sqrt(3.0)
    match kind:
        case "triangle":
            return np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, 0.5 * root3)))
        case "polygon":
            return np.asarray(
                [
                    (
                        _polygon_coordinate(arity, index, False),
                        _polygon_coordinate(arity, index, True),
                    )
                    for index in range(arity)
                ],
                dtype=np.float64,
            )
        case "tetrahedron":
            return np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.5, 0.5 * root3, 0.0),
                    (0.5, root3 / 6.0, np.sqrt(2.0 / 3.0)),
                )
            )
        case "prism":
            base = _ideal_vertices("triangle", 3)
            return np.concatenate(
                (
                    np.pad(base, ((0, 0), (0, 1))),
                    np.pad(base, ((0, 0), (0, 1)), constant_values=1.0),
                )
            )
        case "pyramid":
            vertices = np.asarray(reference_cell_topology("pyramid").vertices)
            vertices[4, 2] = np.sqrt(0.5)
            return vertices
        case "quadrilateral" | "hexahedron":
            return np.asarray(reference_cell_topology(kind).vertices)
        case _:
            raise ValueError(f"No ideal reference shape for {kind!r}.")


def _faces(kind: str, arity: int, /) -> tuple[tuple[int, ...], ...]:
    if kind == "polygon":
        return (tuple(range(arity)),)
    return tuple(reference_cell_topology(kind).entities[2])


def _edges(kind: str, arity: int, /) -> tuple[tuple[int, int], ...]:
    if kind == "polygon":
        return tuple((index, (index + 1) % arity) for index in range(arity))
    return tuple(
        (int(start), int(stop))
        for start, stop in reference_cell_topology(kind).entities[1]
    )


def _square_determinant(matrix: Any, /) -> Any:
    size = matrix.shape[-1]
    if size == 1:
        return matrix[..., 0, 0]
    if size == 2:
        return (
            matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
        )
    return (
        matrix[..., 0, 0]
        * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1]
        * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2]
        * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )


def _adjugate_trace(matrix: Any, /) -> Any:
    if matrix.shape[-1] == 2:
        return matrix[..., 0, 0] + matrix[..., 1, 1]
    return (
        matrix[..., 0, 0] * matrix[..., 1, 1]
        - matrix[..., 0, 1] * matrix[..., 1, 0]
        + matrix[..., 0, 0] * matrix[..., 2, 2]
        - matrix[..., 0, 2] * matrix[..., 2, 0]
        + matrix[..., 1, 1] * matrix[..., 2, 2]
        - matrix[..., 1, 2] * matrix[..., 2, 1]
    )


@dataclass(frozen=True)
class _CornerTable:
    """Corner Jacobian frames ordered to be positive on the reference cell."""

    vertices: np.ndarray
    neighbors: np.ndarray
    ideal_inverse: np.ndarray
    ideal_determinant: np.ndarray
    ideal_scaled: np.ndarray


def _polygon_coordinate(arity: int, index: int, sine: bool, /) -> float:
    """Correctly rounded regular-polygon coordinate, from rational enclosures."""
    multiple = Fraction(2 * index, arity)
    if sine:
        multiple = Fraction(1, 2) - multiple
    multiple = abs(multiple) % 2
    if multiple > 1:
        multiple = 2 - multiple
    special = {
        Fraction(0): 1.0,
        Fraction(1, 3): 0.5,
        Fraction(1, 2): 0.0,
        Fraction(2, 3): -0.5,
        Fraction(1): -1.0,
    }
    if multiple in special:
        return special[multiple]
    bits = 96
    while True:
        fifth = _inverse_arctangent(5, bits + 8)
        large = _inverse_arctangent(239, bits + 8)
        pi_lower = 16 * fifth[0] - 4 * large[1]
        pi_upper = 16 * fifth[1] - 4 * large[0]
        lower = _cosine_point(multiple * pi_upper, bits + 8)[0]
        upper = _cosine_point(multiple * pi_lower, bits + 8)[1]
        if float(lower) == float(upper):
            return float(lower)
        bits *= 2


def _rational_solve(matrix: np.ndarray, right: np.ndarray, /) -> np.ndarray:
    """Solve a small represented system exactly and round each output once."""
    size = matrix.shape[0]
    width = right.shape[1]
    rows = [
        [Fraction(float(value)) for value in matrix[row]]
        + [Fraction(float(value)) for value in right[row]]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next(row for row in range(column, size) if rows[row][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for row in range(size):
            if row != column:
                factor = rows[row][column]
                rows[row] = [
                    a - factor * b for a, b in zip(rows[row], rows[column], strict=True)
                ]
    return np.asarray(
        [[float(value) for value in row[size : size + width]] for row in rows],
        dtype=np.float64,
    )


@host_table_cache(
    max_entries=256,
    max_bytes=1048576,
    weight=lambda value: logical_table_bytes(
        (
            value.vertices,
            value.neighbors,
            value.ideal_inverse,
            value.ideal_determinant,
            value.ideal_scaled,
        )
    ),
)
def _corner_table(kind: str, arity: int, /) -> _CornerTable:
    ideal = _ideal_vertices(kind, arity)
    if kind == "polygon":
        vertices = np.arange(arity)
        neighbors = np.stack(((vertices + 1) % arity, (vertices - 1) % arity), axis=1)
    else:
        vertices, neighbors = reference_corner_frames(kind)
    frames = np.swapaxes(ideal[neighbors] - ideal[vertices][:, None, :], -1, -2)
    determinant = _square_determinant(frames)
    if np.any(determinant <= 0.0):
        raise ValueError(f"Reference corner frames of {kind!r} are not positive.")
    identity = np.eye(frames.shape[-1], dtype=np.float64)
    inverse = np.stack([_rational_solve(frame, identity) for frame in frames])
    scaled = determinant / _quality_product(
        binary64_norm(np.swapaxes(frames, -1, -2)), -1, True
    )
    return _CornerTable(vertices, neighbors, inverse, determinant, scaled)


@dataclass(frozen=True)
class _DihedralTable:
    """Edge samples ``(i, j, k, l)``: interior angle about i->j from k to l."""

    samples: np.ndarray


@cache
def _dihedral_table(kind: str, /) -> _DihedralTable:
    ideal = _ideal_vertices(kind, 0)
    faces = _faces(kind, 0)
    samples = []
    for start, stop in _edges(kind, 0):
        incident = [face for face in faces if start in face and stop in face]
        for origin, target in ((start, stop), (stop, start)):
            others = []
            for face in incident:
                position = face.index(origin)
                before = face[position - 1]
                after = face[(position + 1) % len(face)]
                others.append(after if before == target else before)
            first, second = others
            edge = ideal[target] - ideal[origin]
            cross = np.cross(ideal[first] - ideal[origin], ideal[second] - ideal[origin])
            if binary64_dot(cross, edge) < 0.0:
                first, second = second, first
            samples.append((origin, target, first, second))
    return _DihedralTable(np.asarray(samples))


def _first_order_space(kind: str, points: np.ndarray, /) -> Any:
    """Values and gradients of a spanning set of the first-order geometry space."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    one, zero = np.ones_like(x), np.zeros_like(x)
    match kind:
        case "hexahedron":
            values = (one, x, y, z, x * y, y * z, x * z, x * y * z)
            gradients = (
                (zero, zero, zero),
                (one, zero, zero),
                (zero, one, zero),
                (zero, zero, one),
                (y, x, zero),
                (zero, z, y),
                (z, zero, x),
                (y * z, x * z, x * y),
            )
        case "prism":
            values = (one, x, y, z, x * z, y * z)
            gradients = (
                (zero, zero, zero),
                (one, zero, zero),
                (zero, one, zero),
                (zero, zero, one),
                (z, zero, x),
                (zero, z, y),
            )
        case "pyramid":
            # Rational pyramid space: the apex term is (x - 1/2)(y - 1/2) / (1 - z),
            # which vanishes at the apex (x = y = 1/2, z = 1).
            scale = np.where(z < 1.0, 1.0 - z, 1.0)
            first, second = x - 0.5, y - 0.5
            values = (one, x, y, z, first * second / scale)
            gradients = (
                (zero, zero, zero),
                (one, zero, zero),
                (zero, one, zero),
                (zero, zero, one),
                (second / scale, first / scale, first * second / scale**2),
            )
        case _:
            raise ValueError(f"No first-order geometry space for {kind!r}.")
    return np.stack(values, axis=1), np.stack(
        [np.stack(row, axis=-1) for row in gradients], axis=1
    )


@dataclass(frozen=True)
class _MeasureRule:
    gradients: np.ndarray
    weights: np.ndarray


@cache
def _measure_rule(kind: str, /) -> _MeasureRule:
    """Exact quadrature of the multilinear Jacobian determinant.

    The nodal gradients are host NumPy data: the rule may first be requested
    under a JAX transformation, where a JAX tabulation would be traced.
    """

    gauss = np.asarray((0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)))
    gauss_weights = np.asarray((0.5, 0.5))
    match kind:
        case "hexahedron":
            points = np.stack(np.meshgrid(gauss, gauss, gauss, indexing="ij"), -1)
            points = points.reshape(-1, 3)
            weights = _quality_product(
                np.stack(
                    np.meshgrid(
                        gauss_weights, gauss_weights, gauss_weights, indexing="ij"
                    ),
                    -1,
                ),
                -1,
                True,
            ).reshape(-1)
        case "prism":
            points = np.stack(
                (np.full(2, 1.0 / 3.0), np.full(2, 1.0 / 3.0), gauss), axis=1
            )
            weights = 0.5 * gauss_weights
        case "pyramid":
            collapsed = np.stack(np.meshgrid(gauss, gauss, gauss, indexing="ij"), -1)
            collapsed = collapsed.reshape(-1, 3)
            height = collapsed[:, 2:3]
            points = np.concatenate(
                (collapsed[:, :2] * (1.0 - height) + 0.5 * height, height), axis=1
            )
            weights = np.full(8, 0.125) * (1.0 - collapsed[:, 2]) ** 2
        case _:
            raise ValueError(f"No quadrature measure rule for {kind!r}.")
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    vandermonde, _ = _first_order_space(kind, vertices)
    _, spanning = _first_order_space(kind, points)
    count = vandermonde.shape[0]
    # Nodal gradients solve V^T G_nodal = G_spanning for every point and axis.
    nodal = _rational_solve(
        vandermonde.T, np.moveaxis(spanning, 1, 0).reshape((count, -1))
    )
    gradients = np.moveaxis(nodal.reshape((count, points.shape[0], 3)), 0, 1)
    return _MeasureRule(gradients, weights)


# Quality kernels -----------------------------------------------------------------


def _quality_sum(values: Any, axis: int, keepdims: bool, host: bool, /) -> Any:
    """Sum host entries in increasing index order, without contraction."""
    if not host:
        return jnp.sum(values, axis=axis, keepdims=keepdims)
    moved = np.moveaxis(values, axis, -1)
    result = np.zeros(moved.shape[:-1], dtype=np.float64)
    for index in range(moved.shape[-1]):
        result = result + moved[..., index]
    return np.expand_dims(result, axis) if keepdims else result


def _quality_product(values: Any, axis: int, host: bool, /) -> Any:
    """Multiply host entries in increasing index order, without reassociation."""
    if not host:
        return jnp.prod(values, axis=axis)
    moved = np.moveaxis(values, axis, -1)
    result = np.ones(moved.shape[:-1], dtype=np.float64)
    for index in range(moved.shape[-1]):
        result = result * moved[..., index]
    return result


def _quality_trace(values: Any, host: bool, /) -> Any:
    if not host:
        return jnp.trace(values, axis1=-2, axis2=-1)
    return _quality_sum(np.diagonal(values, axis1=-2, axis2=-1), -1, False, True)


def _quality_norm(values: Any, axis: int, keepdims: bool, host: bool, /) -> Any:
    if not host:
        return jnp.linalg.norm(values, axis=axis, keepdims=keepdims)
    result = binary64_norm(np.moveaxis(values, axis, -1))
    return np.expand_dims(result, axis) if keepdims else result


def _quality_mean(values: Any, axis: int, keepdims: bool, host: bool, /) -> Any:
    if not host:
        return jnp.mean(values, axis=axis, keepdims=keepdims)
    return _quality_sum(values, axis, keepdims, True) / values.shape[axis]


def _quality_atan2(first: Any, second: Any, host: bool, /) -> Any:
    return binary64_arctangent2(first, second) if host else jnp.arctan2(first, second)


def _quality_root(values: Any, dimension: int, host: bool, /) -> Any:
    if not host:
        return values ** (1.0 / dimension)
    return np.sqrt(values) if dimension == 2 else binary64_cube_root(values)


def _quality_segment(
    values: Any, owners: Any, count: int, operation: str, host: bool, /
) -> Any:
    """Reduce in input order for repeated segment indices."""
    if not host:
        if operation == "sum":
            return jax.ops.segment_sum(values, owners, count)
        if operation == "min":
            return jax.ops.segment_min(values, owners, count)
        return jax.ops.segment_max(values, owners, count)
    if operation == "sum":
        result = np.zeros((count, *values.shape[1:]), dtype=values.dtype)
        np.add.at(result, owners, values)
    elif operation == "min":
        initial = (
            np.iinfo(values.dtype).max
            if np.issubdtype(values.dtype, np.integer)
            else np.inf
        )
        result = np.full((count, *values.shape[1:]), initial, dtype=values.dtype)
        np.minimum.at(result, owners, values)
    else:
        result = np.full((count, *values.shape[1:]), -np.inf, dtype=values.dtype)
        np.maximum.at(result, owners, values)
    return result


def _tiny(values: Any, /) -> Any:
    return np.finfo(values.dtype).tiny


def _angle(sine: Any, cosine: Any, host: bool, /) -> Any:
    xp = np if host else jnp
    angle = _quality_atan2(sine, cosine, host)
    return xp.where(angle < 0.0, angle + 2.0 * xp.pi, angle)


def _vector_area(points: Any, host: bool, /) -> Any:
    """Vector area (Newell) of cyclic 2-cells, axis -2 cycling vertices."""
    xp = np if host else jnp
    center = _quality_mean(points, -2, True, host)
    shifted = points - center
    return 0.5 * _quality_sum(
        xp.cross(shifted, xp.roll(shifted, -1, axis=-2)), -2, False, host
    )


def _edge_aspect(
    points: Any, edges: tuple[tuple[int, int], ...], host: bool, /
) -> tuple[Any, Any]:
    xp = np if host else jnp
    index = np.asarray(edges)
    lengths = _quality_norm(
        points[:, index[:, 1]] - points[:, index[:, 0]], -1, False, host
    )
    return xp.max(lengths, axis=1) / xp.maximum(
        xp.min(lengths, axis=1), _tiny(points)
    ), lengths


def _fan_warpage(points: Any, normal: Any, host: bool, /) -> Any:
    """Largest angle between centroid-fan triangle normals and ``normal``."""
    xp = np if host else jnp
    center = _quality_mean(points, -2, True, host)
    shifted = points - center
    fan = xp.cross(shifted, xp.roll(shifted, -1, axis=-2))
    unit = normal[..., None, :]
    sine = _quality_norm(xp.cross(fan, unit), -1, False, host)
    cosine = _quality_sum(fan * unit, -1, False, host)
    return xp.max(_quality_atan2(sine, cosine, host), axis=-1)


def _host_solid_corner_measures(
    points: np.ndarray, table: _CornerTable, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host ``_corner_measures`` of three-frame solid corners without a metric.

    Evaluated on corner-major component arrays with the vector form's rounded
    operations in the same order (determinant cofactors, norms and matrix
    entries from the first product, traces and products from zero and one), so
    every value is bitwise identical; NumPy's loops run over all cells.
    """
    corners = np.ascontiguousarray(np.transpose(points, (2, 1, 0)))
    apex = corners[:, table.vertices]
    # frame[a][d]: component a of the edge to neighbor d, one row per corner.
    frame = [
        [corners[axis, table.neighbors[:, side]] - apex[axis] for side in range(3)]
        for axis in range(3)
    ]
    determinant = _component_determinant(frame)
    lengths = [
        np.sqrt(
            frame[0][side] * frame[0][side]
            + frame[1][side] * frame[1][side]
            + frame[2][side] * frame[2][side]
        )
        for side in range(3)
    ]
    tiny = _tiny(points)
    scaled = (
        determinant
        / np.maximum(1.0 * lengths[0] * lengths[1] * lengths[2], tiny)
        / table.ideal_scaled[:, None]
    )
    inverse = table.ideal_inverse
    target = [
        [
            frame[axis][0] * inverse[:, 0, column, None]
            + frame[axis][1] * inverse[:, 1, column, None]
            + frame[axis][2] * inverse[:, 2, column, None]
            for column in range(3)
        ]
        for axis in range(3)
    ]
    gram = [
        [
            target[0][row] * target[0][column]
            + target[1][row] * target[1][column]
            + target[2][row] * target[2][column]
            for column in range(3)
        ]
        for row in range(3)
    ]
    positive = determinant > 0.0
    gram_determinant = np.maximum(_component_determinant(gram), tiny)
    trace = 0.0 + gram[0][0] + gram[1][1] + gram[2][2]
    adjugate_trace = (
        gram[0][0] * gram[1][1]
        - gram[0][1] * gram[1][0]
        + gram[0][0] * gram[2][2]
        - gram[0][2] * gram[2][0]
        + gram[1][1] * gram[2][2]
        - gram[1][2] * gram[2][1]
    )
    ratio = np.where(
        positive,
        3 * binary64_cube_root(gram_determinant) / np.maximum(trace, tiny),
        0.0,
    )
    condition = np.where(
        positive, np.sqrt(trace * adjugate_trace / gram_determinant) / 3, np.inf
    )
    metric_ratio = np.full(ratio.shape, np.nan, dtype=points.dtype)
    return determinant.T, scaled.T, ratio.T, condition.T, metric_ratio.T


def _component_determinant(matrix: list[list[np.ndarray]], /) -> np.ndarray:
    """``_square_determinant`` of a 3x3 matrix of component arrays."""
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _corner_measures(
    points: Any,
    table: _CornerTable,
    metric: Any,
    unit_normal: Any,
    host: bool,
    /,
) -> tuple[Any, Any, Any, Any, Any]:
    """Corner signed determinant, scaled Jacobian, shape and metric quality."""
    if (
        host
        and metric is None
        and unit_normal is None
        and points.shape[-1] == 3
        and table.neighbors.shape[1] == 3
    ):
        return _host_solid_corner_measures(points, table)
    xp = np if host else jnp
    frames = xp.swapaxes(
        points[:, table.neighbors] - points[:, table.vertices][:, :, None, :], -1, -2
    )
    dimension = frames.shape[-1]
    if unit_normal is None:
        determinant = _square_determinant(frames)
    else:
        cross = xp.cross(frames[..., 0], frames[..., 1])
        determinant = _quality_sum(cross * unit_normal[:, None, :], -1, False, host)
    lengths = _quality_norm(frames, -2, False, host)
    scaled = (
        determinant
        / xp.maximum(_quality_product(lengths, -1, host), _tiny(points))
        / table.ideal_scaled[None, :]
    )
    if host:
        target = binary64_matmul(frames, table.ideal_inverse[None, ...])
        gram = binary64_matmul(np.swapaxes(target, -1, -2), target)
    else:
        target = ein.contract(
            "ckad,kde->ckae", frames, table.ideal_inverse, backend="jax"
        )
        gram = ein.contract("ckad,ckae->ckde", target, target, backend="jax")
    positive = determinant > 0.0
    gram_determinant = xp.maximum(_square_determinant(gram), _tiny(points))
    trace = _quality_trace(gram, host)
    ratio = xp.where(
        positive,
        dimension
        * _quality_root(gram_determinant, dimension, host)
        / xp.maximum(trace, _tiny(points)),
        0.0,
    )
    condition = xp.where(
        positive,
        xp.sqrt(trace * _adjugate_trace(gram) / gram_determinant) / dimension,
        xp.inf,
    )
    if metric is None:
        metric_ratio = xp.full(ratio.shape, xp.nan, dtype=points.dtype)
    else:
        if host:
            metric_target = binary64_matmul(metric[:, None, :, :], target)
            metric_gram = binary64_matmul(np.swapaxes(target, -1, -2), metric_target)
        else:
            metric_gram = ein.contract(
                "ckad,cab,ckbe->ckde", target, metric, target, backend="jax"
            )
        metric_determinant = xp.maximum(_square_determinant(metric_gram), _tiny(points))
        metric_ratio = xp.where(
            positive,
            dimension
            * _quality_root(metric_determinant, dimension, host)
            / xp.maximum(_quality_trace(metric_gram, host), _tiny(points)),
            0.0,
        )
    return determinant, scaled, ratio, condition, metric_ratio


def _metric_quality(metric_ratio: Any, metric: Any, host: bool, /) -> Any:
    """Per-cell minimum metric ratio, or canonical NaN when no metric applies."""
    xp = np if host else jnp
    if metric is None:
        return xp.full(metric_ratio.shape[:1], xp.nan, dtype=metric_ratio.dtype)
    return xp.min(metric_ratio, axis=1)


def _interval_quality(points: Any, metric: Any, host: bool, /) -> _BlockQuality:
    xp = np if host else jnp
    length = _quality_norm(points[:, 1] - points[:, 0], -1, False, host)
    one = xp.ones_like(length)
    nan = xp.full_like(length, xp.nan)
    return _BlockQuality(
        length,
        one,
        one,
        nan,
        one,
        nan,
        nan,
        one,
        nan,
        xp.zeros_like(length),
        nan if metric is None else one,
        xp.isfinite(length) & (length > 0.0),
    )


def _surface_quality(kind: str, points: Any, metric: Any, host: bool, /) -> _BlockQuality:
    """Triangles, quadrilaterals, and polygons in 2-D or embedded in 3-D."""
    xp = np if host else jnp
    arity = points.shape[1]
    table = _corner_table(kind, arity)
    embedded = points.shape[-1] == 3
    if embedded:
        area_vector = _vector_area(points, host)
        area = _quality_norm(area_vector, -1, False, host)
        unit_normal = area_vector / xp.maximum(area, _tiny(points))[:, None]
        measure = area
        warpage = (
            xp.zeros_like(area)
            if kind == "triangle"
            else _fan_warpage(points, unit_normal, host)
        )
    else:
        unit_normal = None
        shifted = points - _quality_mean(points, 1, True, host)
        following = xp.roll(shifted, -1, axis=1)
        fan = shifted[..., 0] * following[..., 1] - shifted[..., 1] * following[..., 0]
        measure = 0.5 * _quality_sum(fan, 1, False, host)
        warpage = xp.zeros_like(measure)
    determinant, scaled, ratio, condition, metric_ratio = _corner_measures(
        points, table, metric, unit_normal, host
    )
    frames = points[:, table.neighbors] - points[:, table.vertices][:, :, None, :]
    cosine = _quality_sum(frames[:, :, 0] * frames[:, :, 1], -1, False, host)
    angles = _angle(determinant, cosine, host)
    aspect, lengths = _edge_aspect(points, _edges(kind, arity), host)
    if kind == "triangle":
        magnitude = xp.abs(measure)
        perimeter = _quality_sum(lengths, 1, False, host)
        radius = (
            16.0
            * magnitude**2
            / xp.maximum(perimeter * _quality_product(lengths, 1, host), _tiny(points))
        )
        valid = measure > 0.0
    elif kind == "quadrilateral":
        radius = xp.full_like(measure, xp.nan)
        valid = xp.all(determinant > 0.0, axis=1)
    else:
        radius = xp.full_like(measure, xp.nan)
        center = _quality_mean(points, 1, True, host)
        shifted = points - center
        following = xp.roll(shifted, -1, axis=1)
        if embedded and unit_normal is not None:
            star = _quality_sum(
                xp.cross(shifted, following) * unit_normal[:, None, :], -1, False, host
            )
        else:
            star = (
                shifted[..., 0] * following[..., 1] - shifted[..., 1] * following[..., 0]
            )
        valid = xp.all(star > 0.0, axis=1)
    return _BlockQuality(
        measure,
        xp.min(ratio, axis=1),
        aspect,
        radius,
        xp.min(scaled, axis=1),
        xp.min(angles, axis=1),
        xp.max(angles, axis=1),
        xp.max(condition, axis=1),
        xp.full_like(measure, xp.nan),
        warpage,
        _metric_quality(metric_ratio, metric, host),
        valid & xp.isfinite(measure),
    )


def _component_sum(*terms: np.ndarray) -> np.ndarray:
    """``_quality_sum`` of component terms: from zero, in increasing order."""
    total = 0.0 + terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def _host_dihedral_angles(points: np.ndarray, samples: np.ndarray, /) -> np.ndarray:
    """Host ``_dihedral_angles`` over contiguous coordinate components.

    The same rounded products, sums (norms from the first square, other sums
    from zero) and cross products as the vector form, so every angle is
    bitwise identical; component arrays keep NumPy's loops long.
    """
    components = np.moveaxis(points, -1, 0)
    origin = components[:, :, samples[:, 0]]
    edge = components[:, :, samples[:, 1]] - origin
    first = components[:, :, samples[:, 2]] - origin
    second = components[:, :, samples[:, 3]] - origin
    length = np.sqrt(edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2])
    unit = edge / np.maximum(length, _tiny(points))
    sine = _component_sum(
        (first[1] * second[2] - first[2] * second[1]) * unit[0],
        (first[2] * second[0] - first[0] * second[2]) * unit[1],
        (first[0] * second[1] - first[1] * second[0]) * unit[2],
    )
    cosine = _component_sum(
        first[0] * second[0], first[1] * second[1], first[2] * second[2]
    ) - _component_sum(
        first[0] * unit[0], first[1] * unit[1], first[2] * unit[2]
    ) * _component_sum(second[0] * unit[0], second[1] * unit[1], second[2] * unit[2])
    return _angle(sine, cosine, True)


def _dihedral_angles(points: Any, samples: np.ndarray, host: bool, /) -> Any:
    if host and points.shape[-1] == 3:
        return _host_dihedral_angles(points, samples)
    xp = np if host else jnp
    origin = points[:, samples[:, 0]]
    edge = points[:, samples[:, 1]] - origin
    first = points[:, samples[:, 2]] - origin
    second = points[:, samples[:, 3]] - origin
    unit = edge / xp.maximum(_quality_norm(edge, -1, True, host), _tiny(points))
    sine = _quality_sum(xp.cross(first, second) * unit, -1, False, host)
    cosine = _quality_sum(first * second, -1, False, host) - _quality_sum(
        first * unit, -1, False, host
    ) * _quality_sum(second * unit, -1, False, host)
    return _angle(sine, cosine, host)


def _volume_measure(kind: str, points: Any, host: bool, /) -> Any:
    xp = np if host else jnp
    if kind == "tetrahedron":
        frames = xp.swapaxes(points[:, 1:] - points[:, :1], -1, -2)
        return _square_determinant(frames) / 6.0
    rule = _measure_rule(kind)
    if host:
        jacobian = binary64_matmul(
            np.swapaxes(points, -1, -2)[:, None, :, :], rule.gradients[None, :, :, :]
        )
    else:
        jacobian = ein.contract("qnk,cna->cqak", rule.gradients, points, backend="jax")
    return _quality_sum(
        _square_determinant(jacobian) * rule.weights[None, :], 1, False, host
    )


def _tetrahedron_shape(points: Any, volume: Any, lengths: Any, host: bool, /) -> Any:
    xp = np if host else jnp
    faces = np.asarray(reference_cell_topology("tetrahedron").entities[2])
    face_areas = 0.5 * _quality_norm(
        xp.cross(
            points[:, faces[:, 1]] - points[:, faces[:, 0]],
            points[:, faces[:, 2]] - points[:, faces[:, 0]],
        ),
        -1,
        False,
        host,
    )
    first = lengths[:, 0] * lengths[:, 5]
    second = lengths[:, 1] * lengths[:, 3]
    third = lengths[:, 2] * lengths[:, 4]
    product = (
        (first + second + third)
        * (first + second - third)
        * (first - second + third)
        * (-first + second + third)
    )
    radius = (
        216.0
        * volume**2
        / xp.maximum(
            _quality_sum(face_areas, 1, False, host)
            * xp.sqrt(xp.maximum(product, _tiny(points))),
            _tiny(points),
        )
    )
    shortest = xp.min(lengths, axis=1)
    sliver = 6.0 * xp.sqrt(2.0) * volume / xp.maximum(shortest**3, _tiny(points))
    return radius, sliver


def _solid_quality(kind: str, points: Any, metric: Any, host: bool, /) -> _BlockQuality:
    xp = np if host else jnp
    table = _corner_table(kind, points.shape[1])
    determinant, scaled, ratio, condition, metric_ratio = _corner_measures(
        points, table, metric, None, host
    )
    measure = _volume_measure(kind, points, host)
    aspect, lengths = _edge_aspect(points, _edges(kind, points.shape[1]), host)
    angles = _dihedral_angles(points, _dihedral_table(kind).samples, host)
    quadrilaterals = tuple(face for face in _faces(kind, 0) if len(face) == 4)
    if quadrilaterals:
        face_points = points[:, np.asarray(quadrilaterals)]
        area = _vector_area(face_points, host)
        unit = area / xp.maximum(_quality_norm(area, -1, True, host), _tiny(points))
        warpage = xp.max(_fan_warpage(face_points, unit, host), axis=1)
    else:
        warpage = xp.zeros_like(measure)
    if kind == "tetrahedron":
        radius, sliver = _tetrahedron_shape(points, measure, lengths, host)
    else:
        radius = xp.full_like(measure, xp.nan)
        sliver = xp.full_like(measure, xp.nan)
    return _BlockQuality(
        measure,
        xp.min(ratio, axis=1),
        aspect,
        radius,
        xp.min(scaled, axis=1),
        xp.min(angles, axis=1),
        xp.max(angles, axis=1),
        xp.max(condition, axis=1),
        sliver,
        warpage,
        _metric_quality(metric_ratio, metric, host),
        xp.all(determinant > 0.0, axis=1) & (measure > 0.0) & xp.isfinite(measure),
    )


def _standard_block_quality(
    kind: str, points: Any, metric: Any, host: bool, /
) -> _BlockQuality:
    kind = kind if kind == "polygon" else reference_cell_topology(kind).family
    match kind:
        case "interval":
            return _interval_quality(points, metric, host)
        case "triangle" | "quadrilateral" | "polygon":
            return _surface_quality(kind, points, metric, host)
        case "tetrahedron" | "hexahedron" | "prism" | "pyramid":
            return _solid_quality(kind, points, metric, host)
        case _:
            raise ValueError(f"No native quality implementation for {kind!r}.")


def _polyhedral_quality(
    tables: PolyhedralStarTables,
    cells: np.ndarray,
    coordinates: Any,
    metric: Any,
    host: bool,
    /,
) -> _BlockQuality:
    """Segment-reduced polyhedral quality in input order on the host."""
    xp = np if host else jnp
    count = cells.size
    cell_total = tables.cell_vertex_counts.size
    face_total = tables.face_sizes.size
    center = (
        _quality_segment(
            coordinates[tables.cell_vertex_values],
            tables.cell_vertex_cell,
            cell_total,
            "sum",
            host,
        )
        / xp.asarray(tables.cell_vertex_counts, dtype=coordinates.dtype)[:, None]
    )
    corner_points = coordinates[tables.face_corner_vertex]
    face_center = (
        _quality_segment(corner_points, tables.face_corner_face, face_total, "sum", host)
        / xp.asarray(tables.face_sizes, dtype=coordinates.dtype)[:, None]
    )
    face_area = 0.5 * _quality_segment(
        xp.cross(
            corner_points - face_center[tables.face_corner_face],
            coordinates[tables.face_corner_next] - face_center[tables.face_corner_face],
        ),
        tables.face_corner_face,
        face_total,
        "sum",
        host,
    )
    selected = np.flatnonzero(np.isin(tables.star_cell, cells))
    local = np.searchsorted(cells, tables.star_cell[selected])
    star_face = tables.star_face[selected]
    first = coordinates[tables.star_first[selected]]
    second = coordinates[tables.star_second[selected]]
    tetrahedra = xp.stack(
        (center[tables.star_cell[selected]], face_center[star_face], first, second),
        axis=1,
    )
    star_metric = None if metric is None else metric[local]
    tetra = _solid_quality("tetrahedron", tetrahedra, star_metric, host)
    measure = _quality_segment(tetra.measure, local, count, "sum", host)
    edge_lengths = _quality_norm(second - first, -1, False, host)
    aspect = _quality_segment(edge_lengths, local, count, "max", host) / xp.maximum(
        _quality_segment(edge_lengths, local, count, "min", host), _tiny(coordinates)
    )
    outward = xp.asarray(tables.star_sign[selected], dtype=coordinates.dtype)[:, None]
    unit = face_area[star_face] * outward
    unit = unit / xp.maximum(_quality_norm(unit, -1, True, host), _tiny(coordinates))
    fan = xp.cross(first - face_center[star_face], second - face_center[star_face])
    fan_angle = _quality_atan2(
        _quality_norm(xp.cross(fan, unit), -1, False, host),
        _quality_sum(fan * unit, -1, False, host),
        host,
    )
    pairs = tables.edge_pairs[np.all(np.isin(tables.edge_pairs, selected), axis=1)]
    pair_first = np.searchsorted(selected, pairs[:, 0])
    pair_second = np.searchsorted(selected, pairs[:, 1])
    direction = second[pair_first] - first[pair_first]
    direction = direction / xp.maximum(
        _quality_norm(direction, -1, True, host), _tiny(coordinates)
    )
    normal_first = unit[pair_first]
    normal_second = unit[pair_second]
    dihedral = xp.pi - _quality_atan2(
        _quality_sum(xp.cross(normal_first, normal_second) * direction, -1, False, host),
        _quality_sum(normal_first * normal_second, -1, False, host),
        host,
    )
    pair_cell = local[pair_first]
    return _BlockQuality(
        measure,
        _quality_segment(tetra.mean_ratio, local, count, "min", host),
        aspect,
        xp.full((count,), xp.nan, dtype=coordinates.dtype),
        _quality_segment(tetra.scaled_jacobian, local, count, "min", host),
        _quality_segment(dihedral, pair_cell, count, "min", host),
        _quality_segment(dihedral, pair_cell, count, "max", host),
        _quality_segment(tetra.condition, local, count, "max", host),
        xp.full((count,), xp.nan, dtype=coordinates.dtype),
        _quality_segment(fan_angle, local, count, "max", host),
        xp.full((count,), xp.nan, dtype=coordinates.dtype)
        if metric is None
        else _quality_segment(tetra.metric_quality, local, count, "min", host),
        (
            _quality_segment(
                tetra.sampled_valid.astype(xp.int32), local, count, "min", host
            )
            > 0
        )
        & (measure > 0.0),
    )


def _vertex_metric(mesh: CellMesh, metric: MeshMetricField | None, /) -> Array | None:
    if metric is None:
        return None
    if not isinstance(metric, MeshMetricField):
        raise TypeError("metric must be MeshMetricField or None.")
    dimension = mesh.ambient_dimension
    if metric.values.shape != (mesh.coordinates.shape[0], dimension, dimension) or not (
        np.array_equal(
            np.asarray(metric.scope.entity_ids), np.asarray(mesh.vertex_global_ids)
        )
    ):
        raise ValueError("Quality metric must be one ambient tensor per mesh vertex.")
    return metric.values


def _cell_metric(values: Array | None, rows: np.ndarray, /) -> Array | None:
    if values is None:
        return None
    weights = jnp.full(rows.shape, 1.0 / rows.shape[1], dtype=values.dtype)
    return interpolate_mesh_metric(values[rows], weights)


# Cells per host quality evaluation: temporaries stay cache resident.
_HOST_QUALITY_CHUNK = 8192


def _host_block_quality(
    kind: str, points: np.ndarray, rows: np.ndarray, metric: np.ndarray | None, /
) -> _BlockQuality:
    """Host standard-block quality, evaluated over consecutive cell chunks.

    Every quality formula is evaluated per cell, so chunking changes only the
    memory traffic: each value is bitwise the one of a single evaluation.
    """
    parts = [
        _standard_block_quality(
            kind,
            points[rows[start : start + _HOST_QUALITY_CHUNK]],
            None if metric is None else metric[start : start + _HOST_QUALITY_CHUNK],
            True,
        )
        for start in range(0, max(rows.shape[0], 1), _HOST_QUALITY_CHUNK)
    ]
    return _BlockQuality(*(np.concatenate(values) for values in zip(*parts, strict=True)))


def evaluate_cell_quality(
    mesh: CellMesh,
    coordinates: ArrayLike | None = None,
    /,
    *,
    metric: MeshMetricField | None = None,
) -> CellQualityEvaluation:
    """Evaluate fixed-topology quality on the host, or trace its JAX formulas."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    vertex_metric = _vertex_metric(mesh, metric)
    source = mesh.coordinates if coordinates is None else coordinates
    traced = any(
        isinstance(value, Tracer)
        for value in jax.tree_util.tree_leaves((source, vertex_metric))
    )
    host = not traced
    points = np.asarray(source, dtype=np.float64) if host else jnp.asarray(source)
    if points.shape != mesh.coordinates.shape:
        raise ValueError("Quality coordinates must preserve the mesh coordinate shape.")
    tables = (
        # ty: ignore[invalid-argument-type]
        polyhedral_star_tables(mesh.connectivity)
        if any(block.cell_kind == "polyhedron" for block in mesh.blocks)
        else None
    )
    blocks = []
    offsets = [0]
    for block in mesh.blocks:
        start = offsets[-1]
        stop = start + block.cell_count
        if block.cell_kind == "polyhedron":
            cells = np.arange(start, stop)
            cell_metric = (
                None
                if vertex_metric is None
                # ty: ignore[invalid-argument-type]
                else _polyhedral_cell_metric(vertex_metric, tables, cells)
            )
            if host and cell_metric is not None:
                cell_metric = np.asarray(cell_metric, dtype=np.float64)
            # ty: ignore[invalid-argument-type]
            blocks.append(_polyhedral_quality(tables, cells, points, cell_metric, host))
        else:
            rows = np.asarray(block.vertices, dtype=np.int32)
            cell_metric = _cell_metric(vertex_metric, rows)
            if host:
                blocks.append(
                    _host_block_quality(
                        block.cell_kind,
                        np.asarray(points, dtype=np.float64),
                        rows,
                        None
                        if cell_metric is None
                        else np.asarray(cell_metric, dtype=np.float64),
                    )
                )
            else:
                blocks.append(
                    _standard_block_quality(
                        block.cell_kind, points[rows], cell_metric, host
                    )
                )
        offsets.append(stop)
    fields = _BlockQuality(
        *(
            jnp.asarray(np.concatenate(values)) if host else jnp.concatenate(values)
            for values in zip(*blocks, strict=True)
        )
    )
    evaluation_id = canonical_fingerprint(
        {
            "kind": "cell-quality-evaluation",
            "topology": mesh.topology_id,
            "block_ids": [block.block_id for block in mesh.blocks],
            "metric": None if metric is None else metric.metric_id,
        }
    )
    return CellQualityEvaluation(
        cell_global_ids=jnp.concatenate(tuple(block.global_ids for block in mesh.blocks)),
        measures=jnp.asarray(fields.measure),
        mean_ratios=jnp.asarray(fields.mean_ratio),
        aspect_ratios=jnp.asarray(fields.aspect),
        radius_ratios=jnp.asarray(fields.radius_ratio),
        scaled_jacobian=jnp.asarray(fields.scaled_jacobian),
        minimum_angle=jnp.asarray(fields.minimum_angle),
        maximum_angle=jnp.asarray(fields.maximum_angle),
        condition_number=jnp.asarray(fields.condition),
        sliver_measures=jnp.asarray(fields.sliver),
        warpage=jnp.asarray(fields.warpage),
        metric_quality=jnp.asarray(fields.metric_quality),
        sampled_valid=jnp.asarray(fields.sampled_valid),
        block_names=tuple(block.name for block in mesh.blocks),
        block_offsets=tuple(offsets),
        topology_id=mesh.topology_id,
        evaluation_id=evaluation_id,
    )


def _with_certified_mapped_measures(
    evaluation: CellQualityEvaluation,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    /,
) -> CellQualityEvaluation:
    """Bind exact-map measure enclosures to the existing sampled quality rows."""
    from ..discretization._cell_geometry_validity import cell_geometry_id
    from ..discretization._mapped_geometry_measure import certified_cell_measures

    measures = certified_cell_measures(mesh, geometry, validity=validity)
    geometry_id = cell_geometry_id(geometry)
    identifier = canonical_fingerprint(
        {
            "kind": "mapped-cell-quality-evaluation",
            "sampled_evaluation": evaluation.evaluation_id,
            "geometry": geometry_id,
            "values": array_tree_fingerprint(measures.values),
            "absolute_errors": array_tree_fingerprint(measures.absolute_errors),
            "exact": measures.exact,
        }
    )
    return replace(
        evaluation,
        measures=jnp.asarray(measures.values),
        measure_absolute_errors=jnp.asarray(measures.absolute_errors),
        measure_geometry_id=geometry_id,
        measure_exact=measures.exact,
        evaluation_id=identifier,
    )


def _polyhedral_cell_metric(
    values: Array, tables: PolyhedralStarTables, cells: np.ndarray, /
) -> Array:
    selected = np.isin(tables.cell_vertex_cell, cells)
    owners = np.searchsorted(cells, tables.cell_vertex_cell[selected])
    counts = np.bincount(owners, minlength=cells.size)
    width = int(np.max(counts))
    rank = np.arange(owners.size) - np.repeat(np.cumsum(counts) - counts, counts)
    rows = np.zeros((cells.size, width), dtype=np.int64)
    weights = np.zeros((cells.size, width), dtype=np.float64)
    rows[owners, rank] = tables.cell_vertex_values[selected]
    weights[owners, rank] = 1.0 / counts[owners]
    return interpolate_mesh_metric(values[rows], jnp.asarray(weights, dtype=values.dtype))


# Exact quality gates ---------------------------------------------------------------
#
# Acceptance thresholds are decided on the represented binary64 coordinates, never
# on the sampled XLA diagnostics. Every gate is the sign of a polynomial in the
# coordinates (and, for polygons, in the algebraic cosine of the ideal corner). An
# outward-rounded host enclosure decides almost every cell; a cell it cannot
# decide is evaluated over exact binary64 rationals with the same formula.

_GATE_BATCH = 32768


@dataclass(frozen=True)
class _Enclosure:
    """Outward-rounded binary64 enclosure of one real per row."""

    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def full(cls, value: float, shape: tuple[int, ...], /) -> _Enclosure:
        array = np.full(shape, value, dtype=np.float64)
        return cls(array, array)

    def _operand(self, other: _Enclosure | int | float, /) -> Interval:
        if isinstance(other, _Enclosure):
            return other.lower, other.upper
        value = np.full(self.lower.shape, float(other), dtype=np.float64)
        return value, value

    def __add__(self, other: _Enclosure | int | float, /) -> _Enclosure:
        lower, upper = interval_add((self.lower, self.upper), self._operand(other))
        return _Enclosure(np.asarray(lower), np.asarray(upper))

    def __radd__(self, other: int | float, /) -> _Enclosure:
        return self + other

    def __sub__(self, other: _Enclosure | int | float, /) -> _Enclosure:
        lower, upper = interval_subtract((self.lower, self.upper), self._operand(other))
        return _Enclosure(np.asarray(lower), np.asarray(upper))

    def __rsub__(self, other: int | float, /) -> _Enclosure:
        lower, upper = interval_subtract(self._operand(other), (self.lower, self.upper))
        return _Enclosure(np.asarray(lower), np.asarray(upper))

    def __mul__(self, other: _Enclosure | int | float, /) -> _Enclosure:
        lower, upper = interval_multiply((self.lower, self.upper), self._operand(other))
        return _Enclosure(np.asarray(lower), np.asarray(upper))

    def __rmul__(self, other: int | float, /) -> _Enclosure:
        return self * other

    def __neg__(self) -> _Enclosure:
        return _Enclosure(-self.upper, -self.lower)

    def take(self, index: np.ndarray, /) -> _Enclosure:
        return _Enclosure(self.lower[index], self.upper[index])


def _total(values: Any, /) -> Any:
    iterator = iter(values)
    total = next(iterator)
    for value in iterator:
        total = total + value
    return total


def _power(value: Any, exponent: int, /) -> Any:
    result = value
    for _ in range(exponent - 1):
        result = result * value
    return result


def _difference(first: Any, second: Any, /) -> tuple[Any, ...]:
    return tuple(a - b for a, b in zip(first, second, strict=True))


def _dot(first: Any, second: Any, /) -> Any:
    return _total(a * b for a, b in zip(first, second, strict=True))


def _cross(first: Any, second: Any, /) -> tuple[Any, Any, Any]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _determinant(frames: Any, /) -> Any:
    if len(frames) == 2:
        return frames[0][0] * frames[1][1] - frames[0][1] * frames[1][0]
    return _dot(frames[0], _cross(frames[1], frames[2]))


@dataclass(frozen=True)
class _GateTable:
    """Exact corner, edge, and boundary data of one standard cell kind.

    ``grams`` are twice the ideal corner Gram matrices: every non-polygon ideal
    cell has integral squared vertex distances, so they are integral. Simplices
    keep one corner: their Jacobian relative to the regular simplex is the same
    affine map at every corner. ``faces`` are the outward boundary cycles of
    the multilinear solids. ``frames`` keep every sampled corner, because the
    scaled Jacobian differs between simplex corners; ``scaled_ideals`` are the
    squared ideal scaled Jacobians ``det(K) / prod(diag K)`` of their ideal
    Gram matrices ``K``.
    """

    corners: tuple[tuple[int, tuple[int, ...]], ...]
    grams: tuple[tuple[tuple[int, ...], ...], ...]
    edges: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, ...], ...]
    frames: tuple[tuple[int, tuple[int, ...]], ...]
    scaled_ideals: tuple[Fraction, ...]


def _ideal_corner_grams(
    kind: str, arity: int, corners: tuple[tuple[int, tuple[int, ...]], ...], /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    ideal = _ideal_vertices(kind, arity)
    squared = np.sum((ideal[:, None, :] - ideal[None, :, :]) ** 2, axis=-1)
    distances = np.rint(squared)
    if np.any(np.abs(squared - distances) > 1.0e-12):
        raise ValueError(f"Ideal {kind!r} vertex distances are not integral.")
    integral = distances.astype(np.int64)
    return tuple(
        tuple(
            tuple(
                int(
                    integral[vertex, first]
                    + integral[vertex, second]
                    - integral[first, second]
                )
                for second in neighbors
            )
            for first in neighbors
        )
        for vertex, neighbors in corners
    )


@cache
def _gate_table(kind: str, arity: int, /) -> _GateTable:
    if kind == "interval":
        return _GateTable((), (), _edges(kind, arity), (), (), ())
    table = _corner_table(kind, arity)
    frames = tuple(
        (int(vertex), tuple(int(value) for value in neighbors))
        for vertex, neighbors in zip(table.vertices, table.neighbors, strict=True)
    )
    corners = frames[:1] if kind in ("triangle", "tetrahedron") else frames
    grams: tuple[tuple[tuple[int, ...], ...], ...] = ()
    scaled_ideals: tuple[Fraction, ...] = ()
    if kind != "polygon":
        frame_grams = _ideal_corner_grams(kind, arity, frames)
        grams = frame_grams[: len(corners)]
        scaled_ideals = tuple(
            Fraction(
                _determinant(gram), prod(gram[axis][axis] for axis in range(len(gram)))
            )
            for gram in frame_grams
        )
    faces: tuple[tuple[int, ...], ...] = ()
    if kind in ("hexahedron", "prism", "pyramid"):
        reference = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
        centroid = np.mean(reference, axis=0)
        oriented = []
        for face in _faces(kind, arity):
            points = reference[np.asarray(face)]
            normal = np.sum(
                np.cross(points[1:-1] - points[0], points[2:] - points[0]), axis=0
            )
            outward = float(np.dot(normal, np.mean(points, axis=0) - centroid)) > 0.0
            cycle = tuple(int(value) for value in face)
            oriented.append(cycle if outward else (cycle[0], *cycle[:0:-1]))
        faces = tuple(oriented)
    return _GateTable(corners, grams, _edges(kind, arity), faces, frames, scaled_ideals)


def _surface_normal(q: Any, /) -> tuple[Any, ...]:
    """Doubled Newell vector area of a cyclic 2-cell translated by vertex 0."""
    terms = [_cross(q[index], q[index + 1]) for index in range(1, len(q) - 1)]
    return tuple(_total(term[axis] for term in terms) for axis in range(3))


def _face_flux(points: tuple[Any, ...], /) -> Any:
    """Twelve times the flux of ``x`` through a planar triangle or bilinear quad.

    The bilinear patch ``a + b u + c v + d u v`` over the unit square carries
    ``a.(B + P/2 + Q/2) + det(b, d, c)/4`` with ``B = b x c``, ``P = b x d`` and
    ``Q = d x c``; a triangle carries ``det(p0, p1, p2)/2``.
    """
    if len(points) == 3:
        return 6 * _determinant(points)
    a = points[0]
    b = _difference(points[1], points[0])
    c = _difference(points[3], points[0])
    d = _difference(_difference(points[0], points[1]), _difference(points[3], points[2]))
    dc = _cross(d, c)
    return (
        12 * _dot(a, _cross(b, c))
        + 6 * (_dot(a, _cross(b, d)) + _dot(a, dc))
        + 3 * _dot(b, dc)
    )


def _scaled_measure(kind: str, ambient: int, table: _GateTable, q: Any, /) -> Any:
    """Measure times 2 (2-D cells), 6 (tetrahedra), or 36 (multilinear solids).

    Intervals return the squared length and cells embedded in 3-D the squared
    doubled vector area. ``q`` holds the vertices translated by vertex 0, which
    leaves every closed-boundary flux unchanged.
    """
    match kind:
        case "interval":
            return _dot(q[1], q[1])
        case "triangle" | "quadrilateral" | "polygon":
            if ambient == 2:
                return _total(
                    _determinant((q[index], q[index + 1]))
                    for index in range(1, len(q) - 1)
                )
            area = _surface_normal(q)
            return _dot(area, area)
        case "tetrahedron":
            return _determinant((q[1], q[2], q[3]))
        case _:
            return _total(
                _face_flux(tuple(q[index] for index in face)) for face in table.faces
            )


def _corner_terms(
    q: Any, corner: tuple[int, tuple[int, ...]], ambient: int, normal: Any, /
) -> tuple[Any, tuple[tuple[Any, ...], ...], Any]:
    """Orientation, physical frame Gram matrix, and its determinant at a corner."""
    vertex, neighbors = corner
    frames = tuple(_difference(q[index], q[vertex]) for index in neighbors)
    dimension = len(frames)
    rows: list[list[Any]] = [[None] * dimension for _ in range(dimension)]
    for first in range(dimension):
        for second in range(first, dimension):
            value = _dot(frames[first], frames[second])
            rows[first][second] = value
            rows[second][first] = value
    gram = tuple(tuple(row) for row in rows)
    if ambient == dimension:
        determinant = _determinant(frames)
        return determinant, gram, determinant * determinant
    cross = _cross(frames[0], frames[1])
    return _dot(cross, normal), gram, _dot(cross, cross)


def _ratio_margin(
    gram: tuple[tuple[Any, ...], ...], determinant: Any, ideal: Any, threshold: Any, /
) -> Any:
    """``r^d tr(C adj K)^d - d^d det(C) det(K)^(d-1)``: positive iff ratio < r.

    At a positive corner the mean ratio relative to the ideal Gram matrix ``K``
    is ``d (det C / det K)^(1/d) / tr(C K^-1)`` for the physical frame Gram
    matrix ``C``; it is invariant under scaling ``K``.
    """
    dimension = len(gram)
    if dimension == 2:
        adjugate = ((ideal[1][1], -ideal[0][1]), (-ideal[1][0], ideal[0][0]))
    else:
        adjugate = tuple(
            tuple(
                ideal[(column + 1) % 3][(row + 1) % 3]
                * ideal[(column + 2) % 3][(row + 2) % 3]
                - ideal[(column + 1) % 3][(row + 2) % 3]
                * ideal[(column + 2) % 3][(row + 1) % 3]
                for column in range(3)
            )
            for row in range(3)
        )
    trace = _total(
        gram[row][column] * adjugate[column][row]
        for row in range(dimension)
        for column in range(dimension)
        if not (type(adjugate[column][row]) is int and adjugate[column][row] == 0)
    )
    return _power(threshold, dimension) * _power(trace, dimension) - (
        dimension**dimension
    ) * determinant * _power(_determinant(ideal), dimension - 1)


# Algebraic ideal polygon corners ----------------------------------------------------

# Minimal polynomials (c^2, c, 1) of c = -cos(2 pi / n) where c is a quadratic
# irrational (Euler phi(n) = 4); c is rational exactly for phi(n) = 2.
_QUADRATIC_POLYGON_COSINES = {
    5: (4, -2, -1),
    8: (2, 0, -1),
    10: (4, 2, -1),
    12: (4, 0, -3),
}
_RATIONAL_POLYGON_COSINES = {3: Fraction(1, 2), 4: Fraction(0), 6: Fraction(-1, 2)}


def _inverse_arctangent(denominator: int, bits: int, /) -> tuple[Fraction, Fraction]:
    """Rational enclosure of ``atan(1 / denominator)`` by its alternating series."""
    square = Fraction(1, denominator * denominator)
    tolerance = Fraction(1, 1 << bits)
    power = Fraction(1, denominator)
    total = Fraction(0)
    order = 0
    while True:
        term = power / (2 * order + 1)
        following = total + term if order % 2 == 0 else total - term
        if term < tolerance:
            return min(total, following), max(total, following)
        total = following
        power *= square
        order += 1


def _cosine_point(value: Fraction, bits: int, /) -> tuple[Fraction, Fraction]:
    """Rational enclosure of ``cos(value)`` with the Lagrange Taylor remainder."""
    square = value * value
    tolerance = Fraction(1, 1 << bits)
    term = Fraction(1)
    total = Fraction(0)
    order = 0
    while True:
        total += term if order % 2 == 0 else -term
        term = term * square / ((2 * order + 1) * (2 * order + 2))
        order += 1
        if term < tolerance:
            return total - term, total + term


@cache
def _polygon_cosine(arity: int, bits: int, /) -> tuple[Fraction, Fraction]:
    """Rational enclosure of ``c = -cos(2 pi / n)``, the ideal corner cosine."""
    if arity in _RATIONAL_POLYGON_COSINES:
        value = _RATIONAL_POLYGON_COSINES[arity]
        return value, value
    # Machin: pi = 16 atan(1/5) - 4 atan(1/239).
    fifth = _inverse_arctangent(5, bits + 8)
    large = _inverse_arctangent(239, bits + 8)
    pi_lower = 16 * fifth[0] - 4 * large[1]
    pi_upper = 16 * fifth[1] - 4 * large[0]
    # cos decreases on [0, pi], which contains 2 pi / n for n >= 2.
    upper_cosine = _cosine_point(2 * pi_lower / arity, bits + 8)[1]
    lower_cosine = _cosine_point(2 * pi_upper / arity, bits + 8)[0]
    return -upper_cosine, -lower_cosine


def _euler_phi(value: int, /) -> int:
    result = value
    remaining = value
    factor = 2
    while factor * factor <= remaining:
        if remaining % factor == 0:
            while remaining % factor == 0:
                remaining //= factor
            result -= result // factor
        factor += 1
    if remaining > 1:
        result -= result // remaining
    return result


def _polygon_ratio_fails(
    gram: tuple[tuple[Fraction, ...], ...],
    determinant: Fraction,
    threshold: Fraction,
    arity: int,
    /,
) -> bool:
    """Exact ``ratio < r`` at a positive corner of an ``n``-gon.

    With ideal corner cosine ``c`` the margin is the quadratic
    ``P(c) = r^2 (C00 + C11 - 2 c C01)^2 - 4 det(C) (1 - c^2)`` with rational
    coefficients. ``c`` has degree ``phi(n)/2``, so a zero of ``P`` is decided
    algebraically; a nonzero sign by rational enclosures refined until they
    exclude zero.
    """
    trace = gram[0][0] + gram[1][1]
    mixed = gram[0][1]
    squared = threshold * threshold
    coefficients = (
        4 * squared * mixed * mixed + 4 * determinant,
        -4 * squared * trace * mixed,
        squared * trace * trace - 4 * determinant,
    )
    if arity in _RATIONAL_POLYGON_COSINES:
        cosine = _RATIONAL_POLYGON_COSINES[arity]
        return (
            coefficients[0] * cosine * cosine + coefficients[1] * cosine + coefficients[2]
            > 0
        )
    if arity in _QUADRATIC_POLYGON_COSINES:
        minimal = _QUADRATIC_POLYGON_COSINES[arity]
        if all(
            coefficients[first] * minimal[second] == coefficients[second] * minimal[first]
            for first, second in ((0, 1), (0, 2), (1, 2))
        ):
            return False
    elif _euler_phi(arity) <= 4:
        raise ValueError(f"Unclassified ideal polygon corner of arity {arity}.")
    elif not any(coefficients):
        return False
    bits = 64
    while True:
        lower, upper = _polygon_cosine(arity, bits)
        squares = (lower * lower, upper * upper)
        square = (
            Fraction(0) if lower <= 0 <= upper else min(squares),
            max(squares),
        )
        quadratic = sorted(coefficients[0] * value for value in square)
        linear = sorted(coefficients[1] * value for value in (lower, upper))
        if quadratic[0] + linear[0] + coefficients[2] > 0:
            return True
        if quadratic[1] + linear[1] + coefficients[2] < 0:
            return False
        bits *= 2


# Gate evaluation --------------------------------------------------------------------


class _GateThresholds(NamedTuple):
    measure: float | None
    mean_ratio: float
    aspect_ratio: float


def _require_quality_thresholds(thresholds: _GateThresholds, /) -> None:
    measure, ratio, aspect = thresholds
    if measure is None or not np.isfinite(measure) or measure < 0.0:
        raise ValueError("minimum_measure must be finite and non-negative.")
    if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValueError("minimum_mean_ratio must lie in [0, 1].")
    if np.isnan(aspect) or aspect < 1.0:
        raise ValueError("maximum_aspect_ratio must be at least one.")


_MEASURE_SCALES = {
    "interval": 1,
    "tetrahedron": 6,
    "hexahedron": 36,
    "prism": 36,
    "pyramid": 36,
}


def _standard_gate_failures(
    kind: str,
    arity: int,
    ambient: int,
    points: np.ndarray,
    thresholds: _GateThresholds,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact per-cell gate failures of one standard block of vertex coordinates."""
    table = _gate_table(kind, arity)
    count = points.shape[0]
    measure_active = thresholds.measure is not None
    ratio_active = thresholds.mean_ratio > 0.0 and bool(table.corners)
    aspect_active = bool(np.isfinite(thresholds.aspect_ratio)) and len(table.edges) > 1
    embedded = ambient == 3 and kind in ("triangle", "quadrilateral", "polygon")
    squared_measure = kind == "interval" or embedded
    scale = _MEASURE_SCALES.get(kind, 2)
    failures = (
        np.zeros((count,), dtype=np.bool_),
        np.zeros((count,), dtype=np.bool_),
        np.zeros((count,), dtype=np.bool_),
    )

    def margins(
        q: Any, measure: Any, ratio: Any, cosine: Any, /
    ) -> tuple[Any, list[tuple[Any, Any]], Any]:
        measure_margin = None
        if measure_active:
            value = _scaled_measure(kind, ambient, table, q)
            bound = scale * measure
            measure_margin = value - bound * bound if squared_measure else value - bound
        corners: list[tuple[Any, Any]] = []
        if ratio_active:
            normal = _surface_normal(q) if embedded else None
            for index, corner in enumerate(table.corners):
                orientation, gram, determinant = _corner_terms(q, corner, ambient, normal)
                if kind != "polygon":
                    ideal = table.grams[index]
                elif cosine is None:
                    corners.append((orientation, (gram, determinant)))
                    continue
                else:
                    ideal = ((1, cosine), (cosine, 1))
                corners.append(
                    (orientation, _ratio_margin(gram, determinant, ideal, ratio))
                )
        lengths = None
        if aspect_active:
            lengths = [
                _dot(edge, edge)
                for edge in (
                    _difference(q[stop], q[start]) for start, stop in table.edges
                )
            ]
        return measure_margin, corners, lengths

    polygon_cosine = None
    if kind == "polygon" and ratio_active:
        cosine = _polygon_cosine(arity, 64)
        polygon_cosine = (fraction_down(cosine[0]), fraction_up(cosine[1]))
    for start in range(0, count, _GATE_BATCH):
        stop = min(start + _GATE_BATCH, count)
        block = points[start:stop]
        shape = (stop - start,)
        lower, upper = interval_subtract((block, block), (block[:, :1], block[:, :1]))
        lower, upper = np.asarray(lower), np.asarray(upper)
        q = [
            tuple(
                _Enclosure(lower[:, vertex, axis], upper[:, vertex, axis])
                for axis in range(ambient)
            )
            for vertex in range(arity)
        ]
        aspect_bound = _Enclosure.full(thresholds.aspect_ratio, shape)
        measure_margin, corners, lengths = margins(
            q,
            _Enclosure.full(thresholds.measure, shape) if measure_active else None,
            _Enclosure.full(thresholds.mean_ratio, shape),
            None
            if polygon_cosine is None
            else _Enclosure(
                np.full(shape, polygon_cosine[0]), np.full(shape, polygon_cosine[1])
            ),
        )
        undecided = np.zeros(shape, dtype=np.bool_)
        if measure_margin is not None:
            fail = measure_margin.upper <= 0.0
            failures[0][start:stop] = fail
            undecided |= ~(fail | (measure_margin.lower > 0.0))
        if corners:
            fail = np.zeros(shape, dtype=np.bool_)
            passed = np.ones(shape, dtype=np.bool_)
            for orientation, margin in corners:
                positive = orientation.lower > 0.0
                fail |= (orientation.upper <= 0.0) | (positive & (margin.lower > 0.0))
                passed &= positive & (margin.upper <= 0.0)
            failures[1][start:stop] = fail
            undecided |= ~(fail | passed)
        if lengths is not None:
            longest = (
                np.maximum.reduce([value.lower for value in lengths]),
                np.maximum.reduce([value.upper for value in lengths]),
            )
            bound = (
                aspect_bound
                * aspect_bound
                * _Enclosure(
                    np.minimum.reduce([value.lower for value in lengths]),
                    np.minimum.reduce([value.upper for value in lengths]),
                )
            )
            fail = longest[0] > bound.upper
            failures[2][start:stop] = fail
            undecided |= ~(fail | (longest[1] <= bound.lower))
        ratio = Fraction(thresholds.mean_ratio)
        for row in np.flatnonzero(undecided):
            origin = block[row, 0]
            exact = [
                tuple(
                    Fraction(float(value)) - Fraction(float(base))
                    for value, base in zip(vertex, origin, strict=True)
                )
                for vertex in block[row]
            ]
            measure_exact, corners_exact, lengths_exact = margins(
                exact,
                None if thresholds.measure is None else Fraction(thresholds.measure),
                ratio,
                None,
            )
            index = start + row
            if measure_exact is not None:
                failures[0][index] = measure_exact <= 0
            if corners_exact:
                failures[1][index] = any(
                    orientation <= 0
                    or (
                        _polygon_ratio_fails(margin[0], margin[1], ratio, arity)
                        if kind == "polygon"
                        else margin > 0
                    )
                    for orientation, margin in corners_exact
                )
            if lengths_exact is not None:
                failures[2][index] = max(lengths_exact) > Fraction(
                    thresholds.aspect_ratio
                ) ** 2 * min(lengths_exact)
    return failures


def _jacobian_gate_failures(
    kind: str, arity: int, points: np.ndarray, threshold: float, /
) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-cell nonpositive corner Jacobians and scaled-Jacobian failures.

    The sampled corner scaled Jacobian is ``det(F) / (|f_1| ... |f_d| s_K)`` for
    the ideal corner value ``s_K``, ``s_K^2 = a / b``. For a positive threshold
    ``s`` it is below ``s`` iff ``det(F) <= 0`` or
    ``s^2 a |f_1|^2 ... |f_d|^2 - b det(F)^2 > 0``.
    """
    table = _gate_table(kind, arity)
    dimension = points.shape[-1]
    if not table.scaled_ideals or any(
        len(neighbors) != dimension for _, neighbors in table.frames
    ):
        raise ValueError(
            f"Exact scaled-Jacobian gates require full-dimensional {kind!r} cells."
        )
    count = points.shape[0]
    nonpositive = np.zeros((count,), dtype=np.bool_)
    below = np.zeros((count,), dtype=np.bool_)
    square = Fraction(threshold) ** 2
    factors = tuple(square * ideal.numerator for ideal in table.scaled_ideals)

    def margins(q: Any, weights: tuple[Any, ...], /) -> list[tuple[Any, Any]]:
        result = []
        for (vertex, neighbors), weight, ideal in zip(
            table.frames, weights, table.scaled_ideals, strict=True
        ):
            frames = tuple(_difference(q[index], q[vertex]) for index in neighbors)
            determinant = _determinant(frames)
            lengths = _dot(frames[0], frames[0])
            for frame in frames[1:]:
                lengths = lengths * _dot(frame, frame)
            result.append(
                (
                    determinant,
                    weight * lengths - ideal.denominator * (determinant * determinant),
                )
            )
        return result

    for start in range(0, count, _GATE_BATCH):
        stop = min(start + _GATE_BATCH, count)
        block = points[start:stop]
        shape = (stop - start,)
        lower, upper = interval_subtract((block, block), (block[:, :1], block[:, :1]))
        lower, upper = np.asarray(lower), np.asarray(upper)
        q = [
            tuple(
                _Enclosure(lower[:, vertex, axis], upper[:, vertex, axis])
                for axis in range(dimension)
            )
            for vertex in range(arity)
        ]
        corners = margins(
            q,
            tuple(
                _Enclosure(
                    np.full(shape, fraction_down(factor)),
                    np.full(shape, fraction_up(factor)),
                )
                for factor in factors
            ),
        )
        sign_fail = np.zeros(shape, dtype=np.bool_)
        sign_pass = np.ones(shape, dtype=np.bool_)
        scaled_fail = np.zeros(shape, dtype=np.bool_)
        scaled_pass = np.ones(shape, dtype=np.bool_)
        for determinant, margin in corners:
            positive = determinant.lower > 0.0
            negative = determinant.upper <= 0.0
            sign_fail |= negative
            sign_pass &= positive
            scaled_fail |= negative | (positive & (margin.lower > 0.0))
            scaled_pass &= positive & (margin.upper <= 0.0)
        nonpositive[start:stop] = sign_fail
        below[start:stop] = scaled_fail
        undecided = ~(sign_fail | sign_pass) | ~(scaled_fail | scaled_pass)
        for row in np.flatnonzero(undecided):
            origin = block[row, 0]
            exact = [
                tuple(
                    Fraction(float(value)) - Fraction(float(base))
                    for value, base in zip(vertex, origin, strict=True)
                )
                for vertex in block[row]
            ]
            exact_corners = margins(exact, factors)
            nonpositive[start + row] = any(value <= 0 for value, _ in exact_corners)
            below[start + row] = any(
                value <= 0 or margin > 0 for value, margin in exact_corners
            )
    return nonpositive, below


class ExactCellGateFailures(NamedTuple):
    """Per-cell exact gate failures of one standard cell block, in row order."""

    nonpositive_corner: np.ndarray
    measure: np.ndarray
    scaled_jacobian: np.ndarray
    mean_ratio: np.ndarray
    aspect_ratio: np.ndarray


def exact_cell_gate_failures(
    kind: str,
    points: np.ndarray,
    /,
    *,
    minimum_measure: float,
    minimum_scaled_jacobian: float,
    minimum_mean_ratio: float,
    maximum_aspect_ratio: float,
) -> ExactCellGateFailures:
    """Decide the sampled corner-quality gates exactly on binary64 vertices.

    ``points`` holds one row of reference-ordered vertices per full-dimensional
    standard cell. Every gate keeps the sampled definition of
    `evaluate_cell_quality`, decided by the outward-rounded screen with an exact
    binary64-rational fallback: ``nonpositive_corner`` is a corner Jacobian
    determinant ``<= 0``; ``measure`` is ``measure <= minimum_measure``;
    ``scaled_jacobian`` and ``mean_ratio`` are corner minima below their
    thresholds; ``aspect_ratio`` is a longest-to-shortest edge ratio above its
    bound.
    """
    family = reference_cell_topology(kind).family
    values = np.asarray(points, dtype=np.float64)
    arity = len(reference_cell_topology(family).vertices)
    if values.ndim != 3 or values.shape[1] != arity:
        raise ValueError(f"Exact {family!r} gates require (cells, {arity}, d) vertices.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Exact quality gates require finite cell vertices.")
    thresholds = _GateThresholds(
        float(minimum_measure), float(minimum_mean_ratio), float(maximum_aspect_ratio)
    )
    _require_quality_thresholds(thresholds)
    scaled = float(minimum_scaled_jacobian)
    if not np.isfinite(scaled) or scaled <= 0.0:
        raise ValueError("minimum_scaled_jacobian must be finite and positive.")
    nonpositive, scaled_failures = _jacobian_gate_failures(family, arity, values, scaled)
    measure, ratio, aspect = _standard_gate_failures(
        family, arity, values.shape[-1], values, thresholds
    )
    return ExactCellGateFailures(nonpositive, measure, scaled_failures, ratio, aspect)


def _padded_sum(values: _Enclosure, owners: np.ndarray, count: int, /) -> _Enclosure:
    """Enclosure of per-owner sums accumulated in stable row order."""
    order = np.argsort(owners, kind="stable")
    sorted_owners = owners[order]
    counts = np.bincount(sorted_owners, minlength=count)
    rank = np.arange(sorted_owners.size) - np.repeat(np.cumsum(counts) - counts, counts)
    width = int(np.max(counts, initial=0))
    lower = np.zeros((count, width), dtype=np.float64)
    upper = np.zeros((count, width), dtype=np.float64)
    lower[sorted_owners, rank] = values.lower[order]
    upper[sorted_owners, rank] = values.upper[order]
    total = _Enclosure.full(0.0, (count,))
    for column in range(width):
        total = total + _Enclosure(lower[:, column], upper[:, column])
    return total


def _polyhedral_gate_failures(
    tables: PolyhedralStarTables,
    cells: np.ndarray,
    coordinates: np.ndarray,
    thresholds: _GateThresholds,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact gate failures of polyhedra through their star tetrahedra.

    Each star tetrahedron joins the cell vertex mean, a face vertex mean, and
    one oriented face edge, as in the sampled evaluation.
    """
    count = cells.size
    tetrahedron = _gate_table("tetrahedron", 4)
    corner, ideal = tetrahedron.corners[0], tetrahedron.grams[0]
    ratio_active = thresholds.mean_ratio > 0.0
    aspect_active = bool(np.isfinite(thresholds.aspect_ratio))
    selected = np.flatnonzero(np.isin(tables.star_cell, cells))
    stars = selected.size
    local = np.searchsorted(cells, tables.star_cell[selected])
    member = np.isin(tables.cell_vertex_cell, cells)
    vertex_owner = np.searchsorted(cells, tables.cell_vertex_cell[member])
    vertex_rows = tables.cell_vertex_values[member]
    faces = np.unique(tables.star_face[selected])
    corner_member = np.isin(tables.face_corner_face, faces)
    corner_owner = np.searchsorted(faces, tables.face_corner_face[corner_member])
    corner_rows = tables.face_corner_vertex[corner_member]
    star_face = np.searchsorted(faces, tables.star_face[selected])
    first = coordinates[tables.star_first[selected]]
    second = coordinates[tables.star_second[selected]]

    def mean(rows: np.ndarray, owners: np.ndarray, total: int, /) -> tuple[Any, ...]:
        divisor = np.asarray(np.bincount(owners, minlength=total), dtype=np.float64)
        means = []
        for axis in range(3):
            values = coordinates[rows, axis]
            sums = _padded_sum(_Enclosure(values, values), owners, total)
            lower, upper = interval_divide((sums.lower, sums.upper), (divisor, divisor))
            means.append(_Enclosure(np.asarray(lower), np.asarray(upper)))
        return tuple(means)

    def star_margins(
        zero: Any, center: Any, face: Any, first_: Any, second_: Any, ratio: Any, /
    ) -> tuple[Any, Any, Any]:
        q = (
            (zero, zero, zero),
            _difference(face, center),
            _difference(first_, center),
            _difference(second_, center),
        )
        volume = _determinant(q[1:])
        if not ratio_active:
            return volume, None, None
        orientation, gram, determinant = _corner_terms(q, corner, 3, None)
        return volume, orientation, _ratio_margin(gram, determinant, ideal, ratio)

    center = mean(vertex_rows, vertex_owner, count)
    face_center = mean(corner_rows, corner_owner, faces.size)
    volume, orientation, ratio_margin = star_margins(
        _Enclosure.full(0.0, (stars,)),
        tuple(axis.take(local) for axis in center),
        tuple(axis.take(star_face) for axis in face_center),
        tuple(_Enclosure(first[:, axis], first[:, axis]) for axis in range(3)),
        tuple(_Enclosure(second[:, axis], second[:, axis]) for axis in range(3)),
        _Enclosure.full(thresholds.mean_ratio, (stars,)),
    )
    undecided = np.zeros((count,), dtype=np.bool_)
    measure_failures = np.zeros((count,), dtype=np.bool_)
    if thresholds.measure is not None:
        measure = _padded_sum(volume, local, count) - 6 * _Enclosure.full(
            thresholds.measure, (count,)
        )
        measure_failures = measure.upper <= 0.0
        undecided |= ~(measure_failures | (measure.lower > 0.0))
    ratio_failures = np.zeros((count,), dtype=np.bool_)
    if ratio_active:
        positive = orientation.lower > 0.0
        star_fail = (orientation.upper <= 0.0) | (positive & (ratio_margin.lower > 0.0))
        star_pass = positive & (ratio_margin.upper <= 0.0)
        ratio_failures = np.bincount(local[star_fail], minlength=count) > 0
        passed = np.bincount(local[~star_pass], minlength=count) == 0
        undecided |= ~(ratio_failures | passed)
    aspect_failures = np.zeros((count,), dtype=np.bool_)
    if aspect_active:
        lower, upper = interval_subtract((second, second), (first, first))
        lower, upper = np.asarray(lower), np.asarray(upper)
        edge = tuple(_Enclosure(lower[:, axis], upper[:, axis]) for axis in range(3))
        lengths = _dot(edge, edge)
        longest = (np.full((count,), -np.inf), np.full((count,), -np.inf))
        shortest = (np.full((count,), np.inf), np.full((count,), np.inf))
        np.maximum.at(longest[0], local, lengths.lower)
        np.maximum.at(longest[1], local, lengths.upper)
        np.minimum.at(shortest[0], local, lengths.lower)
        np.minimum.at(shortest[1], local, lengths.upper)
        bound = _Enclosure.full(thresholds.aspect_ratio, (count,))
        bound = bound * bound * _Enclosure(*shortest)
        aspect_failures = longest[0] > bound.upper
        undecided |= ~(aspect_failures | (longest[1] <= bound.lower))

    def exact_mean(rows: np.ndarray, /) -> tuple[Fraction, ...]:
        return tuple(
            sum(
                (Fraction(float(value)) for value in coordinates[rows, axis]), Fraction(0)
            )
            / rows.size
            for axis in range(3)
        )

    ratio = Fraction(thresholds.mean_ratio)
    for row in np.flatnonzero(undecided):
        exact_center = exact_mean(vertex_rows[vertex_owner == row])
        volume_total = Fraction(0)
        ratio_fail = False
        lengths_exact = []
        for star in np.flatnonzero(local == row):
            first_exact = tuple(Fraction(float(value)) for value in first[star])
            second_exact = tuple(Fraction(float(value)) for value in second[star])
            star_volume, star_orientation, star_margin = star_margins(
                Fraction(0),
                exact_center,
                exact_mean(corner_rows[corner_owner == star_face[star]]),
                first_exact,
                second_exact,
                ratio,
            )
            volume_total += star_volume
            if ratio_active:
                ratio_fail = ratio_fail or star_orientation <= 0 or star_margin > 0
            edge_exact = _difference(second_exact, first_exact)
            lengths_exact.append(_dot(edge_exact, edge_exact))
        if thresholds.measure is not None:
            measure_failures[row] = volume_total <= 6 * Fraction(thresholds.measure)
        ratio_failures[row] = ratio_fail
        if aspect_active:
            aspect_failures[row] = max(lengths_exact) > Fraction(
                thresholds.aspect_ratio
            ) ** 2 * min(lengths_exact)
    return measure_failures, ratio_failures, aspect_failures


def _certified_measure_failures(
    evaluation: CellQualityEvaluation, threshold: float, /
) -> np.ndarray:
    """Exact ``value - error <= threshold`` on certified mapped measure enclosures."""
    values = np.asarray(evaluation.measures, dtype=np.float64)
    errors = np.asarray(evaluation.measure_absolute_errors, dtype=np.float64)
    bound = np.full(values.shape, threshold, dtype=np.float64)
    lower, upper = interval_subtract((values, values), (errors, errors))
    lower, upper = interval_subtract(
        (np.asarray(lower), np.asarray(upper)), (bound, bound)
    )
    lower, upper = np.asarray(lower), np.asarray(upper)
    failures = upper <= 0.0
    for row in np.flatnonzero(~(failures | (lower > 0.0))):
        failures[row] = Fraction(float(values[row])) - Fraction(
            float(errors[row])
        ) <= Fraction(threshold)
    return failures


def _decide_quality_gates(
    mesh: CellMesh, evaluation: CellQualityEvaluation, thresholds: _GateThresholds, /
) -> tuple[np.ndarray, ...]:
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Exact quality gates require finite mesh coordinates.")
    mapped = evaluation.measure_absolute_errors is not None
    block_thresholds = thresholds._replace(measure=None) if mapped else thresholds
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    start = 0
    for block in mesh.blocks:
        stop = start + block.cell_count
        if block.cell_kind == "polyhedron":
            parts.append(
                _polyhedral_gate_failures(
                    # ty: ignore[invalid-argument-type]
                    polyhedral_star_tables(mesh.connectivity),
                    np.arange(start, stop),
                    coordinates,
                    block_thresholds,
                )
            )
        else:
            rows = np.asarray(block.vertices, dtype=np.int64)
            kind = (
                "polygon"
                if block.cell_kind == "polygon"
                else reference_cell_topology(block.cell_kind).family
            )
            parts.append(
                _standard_gate_failures(
                    kind,
                    rows.shape[1],
                    coordinates.shape[1],
                    coordinates[rows],
                    block_thresholds,
                )
            )
        start = stop
    measure, ratio, aspect = (
        np.concatenate(values) for values in zip(*parts, strict=True)
    )
    if mapped:
        measure = _certified_measure_failures(
            evaluation,
            thresholds.measure,  # ty: ignore[invalid-argument-type]
        )
    return measure, ratio, aspect


# A global embedding certified by the native tetrahedral degree theorem has
# positively oriented cells and every coordinate zero or of magnitude in
# [2^-120, 2^120] (its exact predicate domain): distinct vertices are then at
# least 2^-172 and less than 2^122 apart, so no cell fails a zero measure or
# mean-ratio threshold, nor an aspect threshold of at least 2^294.
_DEGREE_DECIDED_ASPECT_RATIO = float(2**294)


def summarize_cell_quality(
    mesh: CellMesh,
    evaluation: CellQualityEvaluation,
    /,
    *,
    minimum_measure: float = 0.0,
    minimum_mean_ratio: float = 0.0,
    maximum_aspect_ratio: float = 1.0e300,
) -> CellQualityReport:
    """Summarize sampled diagnostics and decide the quality thresholds exactly.

    A cell fails ``minimum_measure`` when its measure (the certified enclosure
    lower bound for mapped evaluations) is at most the threshold,
    ``minimum_mean_ratio`` when a corner mean ratio is below it (zero at a
    non-positive corner), and ``maximum_aspect_ratio`` when its longest edge
    exceeds the threshold times its shortest. Decisions are exact on the
    represented binary64 mesh coordinates and never read the sampled floats.
    """
    return _summarized_cell_quality(
        mesh,
        evaluation,
        minimum_measure,
        minimum_mean_ratio,
        maximum_aspect_ratio,
        degree_certified=False,
    )


def _summarized_cell_quality(
    mesh: CellMesh,
    evaluation: CellQualityEvaluation,
    minimum_measure: float,
    minimum_mean_ratio: float,
    maximum_aspect_ratio: float,
    /,
    *,
    degree_certified: bool,
) -> CellQualityReport:
    """``summarize_cell_quality``; ``degree_certified`` states that the native
    tetrahedral degree theorem certified the global embedding of exactly this
    mesh, which decides the thresholds it implies on exact (unmapped) measures:
    no cell fails them."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(evaluation, CellQualityEvaluation):
        raise TypeError("evaluation must be CellQualityEvaluation.")
    identifiers = np.asarray(evaluation.cell_global_ids, dtype=np.int64)
    if (
        evaluation.topology_id != mesh.topology_id
        or evaluation.block_names != tuple(block.name for block in mesh.blocks)
        or not np.array_equal(
            identifiers,
            np.concatenate(
                [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
            ),
        )
    ):
        raise ValueError("Quality evaluation is not bound to the summarized mesh.")
    thresholds = _GateThresholds(
        float(minimum_measure), float(minimum_mean_ratio), float(maximum_aspect_ratio)
    )
    _require_quality_thresholds(thresholds)
    if (
        degree_certified
        and evaluation.measure_absolute_errors is None
        and thresholds.measure == 0.0
        and thresholds.mean_ratio == 0.0
        and thresholds.aspect_ratio >= _DEGREE_DECIDED_ASPECT_RATIO
    ):
        measure_ids: tuple[int, ...] = ()
        ratio_ids: tuple[int, ...] = ()
        aspect_ids: tuple[int, ...] = ()
    else:
        measure_ids, ratio_ids, aspect_ids = (
            tuple(int(value) for value in np.sort(identifiers[mask]))
            for mask in _decide_quality_gates(mesh, evaluation, thresholds)
        )
    return CellQualityReport(
        evaluation,
        mesh_id=mesh.mesh_id,
        gate_thresholds=(
            float(minimum_measure),
            thresholds.mean_ratio,
            thresholds.aspect_ratio,
        ),
        measure_failure_ids=measure_ids,
        mean_ratio_failure_ids=ratio_ids,
        aspect_ratio_failure_ids=aspect_ids,
    )


# Straight sweep layers -------------------------------------------------------------


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

    coordinates = np.asarray(points, dtype=np.float64)
    cells = np.asarray(prisms, dtype=np.int32)
    indices = np.asarray(layer_indices, dtype=np.int32)
    anchor = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(unit_direction, dtype=np.float64)
    requested = np.asarray(requested_thicknesses, dtype=np.float64)
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
        edge_lengths, np.finfo(np.float64).tiny
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
    measured = np.full(requested.shape, np.nan, dtype=np.float64)
    for layer in range(requested.size):
        selected = indices == layer
        if np.any(selected):
            measured[layer] = float(np.mean(projected_edges[selected]))
    growth = (
        measured[1:] / measured[:-1]
        if measured.size > 1
        else np.empty((0,), dtype=np.float64)
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


# Finite-volume face quality ------------------------------------------------------


class FiniteVolumeQualityEvaluation(StrictModule, NonTrainableState):
    """Interior-face skewness and non-orthogonality of a full-dimensional mesh.

    ``non_orthogonality`` is the angle (radians) between the owner-to-neighbor
    centroid vector and the face normal oriented out of the owner; values above
    pi/2 mean the neighbor centroid lies behind the face. ``skewness`` is the
    distance from the face centroid to the centroid line's face intersection,
    divided by the centroid distance. Owners are the lower cell entity rows.
    """

    face_global_ids: Array
    owner_cell_global_ids: Array
    neighbor_cell_global_ids: Array
    skewness: Array
    non_orthogonality: Array
    topology_id: str = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)


def _block_centroids(kind: str, points: Array, /) -> tuple[Array, Array]:
    """Measure-weighted centroids of standard cells via star decomposition."""
    kind = kind if kind == "polygon" else reference_cell_topology(kind).family

    if kind in ("triangle", "quadrilateral", "polygon"):
        center = jnp.mean(points, axis=1, keepdims=True)
        following = jnp.roll(points, -1, axis=1)
        first = points - center
        second = following - center
        areas = 0.5 * (first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])
        centroids = (center + points + following) / 3.0
        return jnp.sum(areas[..., None] * centroids, axis=1), jnp.sum(areas, axis=1)
    center = jnp.mean(points, axis=1)
    volumes = []
    moments = []
    for face in _faces(kind, 0):
        loop = points[:, np.asarray(face)]
        face_center = jnp.mean(loop, axis=1)
        following = jnp.roll(loop, -1, axis=1)
        frames = jnp.stack(
            (
                jnp.broadcast_to((face_center - center)[:, None], loop.shape),
                loop - center[:, None],
                following - center[:, None],
            ),
            axis=-1,
        )
        volume = _square_determinant(frames) / 6.0
        centroid = (center[:, None] + face_center[:, None] + loop + following) / 4.0
        volumes.append(volume)
        moments.append(jnp.sum(volume[..., None] * centroid, axis=1))
    total = jnp.sum(jnp.concatenate(volumes, axis=1), axis=1)
    # ty: ignore[invalid-return-type]
    return sum(moments), total


def _polyhedral_centroids(
    tables: PolyhedralStarTables, cells: np.ndarray, coordinates: Array, /
) -> tuple[Array, Array]:
    cell_total = tables.cell_vertex_counts.size
    center = (
        jax.ops.segment_sum(
            coordinates[tables.cell_vertex_values], tables.cell_vertex_cell, cell_total
        )
        / jnp.asarray(tables.cell_vertex_counts, dtype=coordinates.dtype)[:, None]
    )
    face_center = (
        jax.ops.segment_sum(
            coordinates[tables.face_corner_vertex],
            tables.face_corner_face,
            tables.face_sizes.size,
        )
        / jnp.asarray(tables.face_sizes, dtype=coordinates.dtype)[:, None]
    )
    selected = np.flatnonzero(np.isin(tables.star_cell, cells))
    local = np.searchsorted(cells, tables.star_cell[selected])
    apex = center[tables.star_cell[selected]]
    base = face_center[tables.star_face[selected]]
    first = coordinates[tables.star_first[selected]]
    second = coordinates[tables.star_second[selected]]
    volume = (
        _square_determinant(
            jnp.stack((base - apex, first - apex, second - apex), axis=-1)
        )
        / 6.0
    )
    centroid = (apex + base + first + second) / 4.0
    return (
        jax.ops.segment_sum(volume[:, None] * centroid, local, cells.size),
        jax.ops.segment_sum(volume, local, cells.size),
    )


def _facet_geometry(mesh: CellMesh, points: Array, /) -> tuple[Array, Array]:
    """Facet centroids and area-weighted normals in facet entity row order."""

    connectivity = mesh.connectivity
    if mesh.topological_dimension == 2:
        if isinstance(connectivity, PolygonalConnectivity):
            edges = np.asarray(connectivity.edges, dtype=np.int64)
        elif isinstance(connectivity, SimplicialConnectivity):
            edges = np.asarray(connectivity.entities[1], dtype=np.int64)
        else:
            raise TypeError(
                "Planar finite-volume quality requires polygonal or simplex connectivity."
            )
        start = points[edges[:, 0]]
        stop = points[edges[:, 1]]
        tangent = stop - start
        return 0.5 * (start + stop), jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=1)
    if isinstance(connectivity, PolyhedralConnectivity):
        tables = polyhedral_star_tables(connectivity)
        face_count = tables.face_sizes.size
        corners = points[tables.face_corner_vertex]
        following = points[tables.face_corner_next]
        center = (
            jax.ops.segment_sum(corners, tables.face_corner_face, face_count)
            / (jnp.asarray(tables.face_sizes, dtype=points.dtype)[:, None])
        )
        fan = 0.5 * jnp.cross(
            corners - center[tables.face_corner_face],
            following - center[tables.face_corner_face],
        )
        fan_area = jnp.linalg.norm(fan, axis=-1)
        fan_center = (center[tables.face_corner_face] + corners + following) / 3.0
        area = jax.ops.segment_sum(fan_area, tables.face_corner_face, face_count)
        centroid = (
            jax.ops.segment_sum(
                fan_area[:, None] * fan_center, tables.face_corner_face, face_count
            )
            / jnp.maximum(area, _tiny(points))[:, None]
        )
        return centroid, jax.ops.segment_sum(fan, tables.face_corner_face, face_count)
    if isinstance(connectivity, SimplicialConnectivity):
        face_rows = connectivity.entities[2]
    elif isinstance(connectivity, (TetrahedralConnectivity, HexahedralConnectivity)):
        face_rows = connectivity.faces
    else:
        raise TypeError("Finite-volume quality requires volume connectivity.")
    loops = points[np.asarray(face_rows, dtype=np.int64)]
    center = jnp.mean(loops, axis=1, keepdims=True)
    following = jnp.roll(loops, -1, axis=1)
    fan = 0.5 * jnp.cross(loops - center, following - center)
    fan_area = jnp.linalg.norm(fan, axis=-1)
    fan_center = (center + loops + following) / 3.0
    centroid = (
        jnp.sum(fan_area[..., None] * fan_center, axis=1)
        / jnp.maximum(jnp.sum(fan_area, axis=1), _tiny(points))[:, None]
    )
    return centroid, jnp.sum(fan, axis=1)


def evaluate_finite_volume_quality(mesh: CellMesh, /) -> FiniteVolumeQualityEvaluation:
    """Evaluate skewness and non-orthogonality on every interior face."""

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    dimension = mesh.topological_dimension
    if dimension not in (2, 3) or mesh.ambient_dimension != dimension:
        raise ValueError(
            "Finite-volume quality requires full-dimensional 2-D or 3-D cells."
        )
    points = mesh.coordinates
    tables = (
        polyhedral_star_tables(mesh.connectivity)
        if isinstance(mesh.connectivity, PolyhedralConnectivity)
        else None
    )
    moments = []
    volumes = []
    cursor = 0
    for block in mesh.blocks:
        if block.cell_kind == "polyhedron":
            moment, volume = _polyhedral_centroids(
                # ty: ignore[invalid-argument-type]
                tables,
                np.arange(cursor, cursor + block.cell_count),
                points,
            )
        else:
            moment, volume = _block_centroids(
                block.cell_kind, points[np.asarray(block.vertices, dtype=np.int64)]
            )
        moments.append(moment)
        volumes.append(volume)
        cursor += block.cell_count
    volume = jnp.concatenate(volumes)
    centroids = jnp.concatenate(moments) / jnp.where(volume == 0.0, 1.0, volume)[:, None]
    face_centroids, face_normals = _facet_geometry(mesh, points)
    relation = mesh.topology.incidences[-1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    facets = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    cells = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    order = np.lexsort((cells, facets))
    facets = facets[order]
    cells = cells[order]
    paired = np.flatnonzero(facets[1:] == facets[:-1])
    face = facets[paired]
    owner = cells[paired]
    neighbor = cells[paired + 1]
    delta = centroids[neighbor] - centroids[owner]
    normal = face_normals[face]
    offset = face_centroids[face] - centroids[owner]
    normal = jnp.where(
        jnp.sum(normal * offset, axis=-1, keepdims=True) < 0.0, -normal, normal
    )
    projection = jnp.sum(delta * normal, axis=-1)
    if dimension == 2:
        sine = jnp.abs(delta[:, 0] * normal[:, 1] - delta[:, 1] * normal[:, 0])
    else:
        sine = jnp.linalg.norm(jnp.cross(delta, normal), axis=-1)
    non_orthogonality = jnp.arctan2(sine, projection)
    parameter = jnp.sum(offset * normal, axis=-1) / jnp.where(
        projection == 0.0, jnp.inf, projection
    )
    intersection = centroids[owner] + parameter[:, None] * delta
    distance = jnp.linalg.norm(delta, axis=-1)
    skewness = jnp.linalg.norm(
        face_centroids[face] - intersection, axis=-1
    ) / jnp.maximum(distance, _tiny(points))
    skewness = jnp.where(projection == 0.0, jnp.inf, skewness)
    facet_ids = np.asarray(mesh.entity_set(dimension - 1).entity_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    return FiniteVolumeQualityEvaluation(
        face_global_ids=jnp.asarray(facet_ids[face]),
        owner_cell_global_ids=jnp.asarray(cell_ids[owner]),
        neighbor_cell_global_ids=jnp.asarray(cell_ids[neighbor]),
        skewness=skewness,
        non_orthogonality=non_orthogonality,
        topology_id=mesh.topology_id,
        evaluation_id=canonical_fingerprint(
            {
                "kind": "finite-volume-quality-evaluation",
                "mesh": mesh.mesh_id,
            }
        ),
    )


__all__ = [
    "CellQualityEvaluation",
    "CellQualityReport",
    "FiniteVolumeQualityEvaluation",
    "evaluate_cell_quality",
    "evaluate_finite_volume_quality",
    "summarize_cell_quality",
]
