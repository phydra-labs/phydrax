#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Immutable source-boundary query results and continuous chart covers."""

from __future__ import annotations

from collections.abc import Iterable
from enum import IntEnum
from fractions import Fraction
from itertools import groupby
from typing import final, Literal, Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..typing import ConvertibleToArray, Dim, HostFloat64, parse, Scope
from ._embedding_state import MeshCertificateFinding


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh
    from .reconstruction._envelope import SurfaceEnvelopeReproduction


type SourceBoundSemantics = Literal["certified", "sampled"]


class SourceProjectionStatus(IntEnum):
    """UNIQUE and SEAM are resolved; every other outcome is non-resolved evidence."""

    UNIQUE = 0
    SEAM = 1
    AMBIGUOUS = 2
    FAILED = 3
    UNRESOLVED = 4
    NON_ISOLATED = 5
    SINGULAR = 6


class _SourcePointDim(Dim):
    """Queries or representatives of one source boundary."""


class _SourceChartCornerDim(Dim, minimum=3):
    """Three vertices of one affine chart simplex."""


class SourceAmbientDim(Dim, minimum=1):
    """Physical coordinate dimension of a source query."""


class SourceBoundaryCellDim(Dim, minimum=1):
    """Global cells carrying oriented source-boundary sides."""


class SourceBoundarySideDim(Dim, minimum=3):
    """Oriented local sides of a triangle."""


class SourceBoundaryOccurrenceDim(Dim, minimum=3):
    """Loop, use ordinal, and authored traversal orientation."""


class SourceBoundaryDistance(StrictModule, NonTrainableState):
    """Lower and upper Euclidean-distance bounds in query-point order."""

    __strict_contract__ = True

    lower: HostFloat64[_SourcePointDim]
    upper: HostFloat64[_SourcePointDim]
    semantics: SourceBoundSemantics

    def __init__(
        self,
        lower: ConvertibleToArray,
        upper: ConvertibleToArray,
        semantics: SourceBoundSemantics,
        /,
    ) -> None:
        scope = Scope()
        low = parse(
            np.asarray(lower, dtype=np.float64),
            HostFloat64[_SourcePointDim],
            "lower",
            scope=scope,
        )
        high = parse(
            np.asarray(upper, dtype=np.float64),
            HostFloat64[_SourcePointDim],
            "upper",
            scope=scope,
        )
        if np.any(np.isnan(low)) or np.any(np.isnan(high)) or np.any(low > high):
            raise ValueError(
                "SourceBoundaryDistance requires non-NaN lower <= upper bounds."
            )
        low_ = np.array(low, copy=True)
        high_ = np.array(high, copy=True)
        low_.setflags(write=False)
        high_.setflags(write=False)
        self.lower = low_
        self.upper = high_
        self.semantics = parse(semantics, SourceBoundSemantics, "semantics")


class SourceBoundarySamples(StrictModule, NonTrainableState):
    """Source representatives with point-error and continuous cover radii."""

    __strict_contract__ = True

    points: HostFloat64[_SourcePointDim, SourceAmbientDim]
    covering_radius: HostFloat64[_SourcePointDim]
    point_error: HostFloat64[_SourcePointDim]
    semantics: SourceBoundSemantics
    complete: bool

    def __init__(
        self,
        points: ConvertibleToArray,
        covering_radius: ConvertibleToArray,
        point_error: ConvertibleToArray,
        semantics: SourceBoundSemantics,
        /,
        *,
        complete: bool,
    ) -> None:
        scope = Scope()
        values = parse(
            np.asarray(points, dtype=np.float64),
            HostFloat64[_SourcePointDim, SourceAmbientDim],
            "points",
            scope=scope,
        )
        radius = parse(
            np.asarray(covering_radius, dtype=np.float64),
            HostFloat64[_SourcePointDim],
            "covering_radius",
            scope=scope,
        )
        error = parse(
            np.asarray(point_error, dtype=np.float64),
            HostFloat64[_SourcePointDim],
            "point_error",
            scope=scope,
        )
        if (
            not np.all(np.isfinite(values))
            or np.any(~(radius >= 0.0))
            or np.any(~(error >= 0.0))
            or not isinstance(complete, (bool, np.bool_))
        ):
            raise ValueError(
                "SourceBoundarySamples requires finite points and nonnegative radii/errors."
            )
        points_ = np.array(values, copy=True)
        radius_ = np.array(radius, copy=True)
        error_ = np.array(error, copy=True)
        for array in (points_, radius_, error_):
            array.setflags(write=False)
        self.points = points_
        self.covering_radius = radius_
        self.point_error = error_
        self.semantics = parse(semantics, SourceBoundSemantics, "semantics")
        self.complete = bool(complete)


class SourceBoundaryChartCover(StrictModule, NonTrainableState):
    """Independent continuous source cover by affine physical chart triangles."""

    __strict_contract__ = True

    simplices: HostFloat64[_SourcePointDim, _SourceChartCornerDim, SourceAmbientDim]
    deviation_bounds: HostFloat64[_SourcePointDim]
    semantics: SourceBoundSemantics
    complete: bool
    source_id: str
    source_revision: str
    findings: tuple[MeshCertificateFinding, ...]
    restriction_ids: tuple[str, ...]
    cover_id: str

    def __init__(
        self,
        simplices: ConvertibleToArray,
        deviation_bounds: ConvertibleToArray,
        semantics: SourceBoundSemantics,
        complete: bool,
        source_id: str,
        source_revision: str,
        /,
        *,
        findings: tuple[MeshCertificateFinding, ...] = (),
        restriction_ids: tuple[str, ...] = (),
    ) -> None:
        scope = Scope()
        values = parse(
            np.asarray(simplices, dtype=np.float64),
            HostFloat64[_SourcePointDim, _SourceChartCornerDim, SourceAmbientDim],
            "simplices",
            scope=scope,
        )
        bounds = parse(
            np.asarray(deviation_bounds, dtype=np.float64),
            HostFloat64[_SourcePointDim],
            "deviation_bounds",
            scope=scope,
        )
        if values.shape[1] != 3:
            raise ValueError("SourceBoundaryChartCover.simplices must contain triangles.")
        if not np.all(np.isfinite(values)) or np.any(~(bounds >= 0.0)):
            raise ValueError(
                "SourceBoundaryChartCover requires finite charts and nonnegative bounds."
            )
        if not isinstance(complete, (bool, np.bool_)):
            raise TypeError("SourceBoundaryChartCover.complete must be boolean.")
        if not all(isinstance(finding, MeshCertificateFinding) for finding in findings):
            raise TypeError(
                "SourceBoundaryChartCover.findings must contain certificate findings."
            )
        source = canonical_identifier(source_id, "source_id")
        revision = canonical_identifier(source_revision, "source_revision")
        restrictions = tuple(
            canonical_identifier(value, "restriction_id") for value in restriction_ids
        )
        if len(set(restrictions)) != len(restrictions):
            raise ValueError("SourceBoundaryChartCover.restriction_ids must be unique.")
        simplices_ = np.array(values, copy=True)
        bounds_ = np.array(bounds, copy=True)
        simplices_.setflags(write=False)
        bounds_.setflags(write=False)
        self.simplices = simplices_
        self.deviation_bounds = bounds_
        self.semantics = parse(semantics, SourceBoundSemantics, "semantics")
        self.complete = bool(complete)
        self.source_id = source
        self.source_revision = revision
        self.findings = tuple(findings)
        self.restriction_ids = restrictions
        self.cover_id = canonical_fingerprint(
            {
                "kind": "source-boundary-chart-cover",
                "source": source,
                "revision": revision,
                "arrays": array_tree_fingerprint((simplices_, bounds_)),
                "semantics": self.semantics,
                "complete": self.complete,
                "findings": tuple(finding.finding_id for finding in self.findings),
                "restrictions": restrictions,
            }
        )


type _RationalToken = tuple[int, int]
type SourceBoundaryRationalChartBank = tuple[tuple[tuple[_RationalToken, ...], ...], ...]


def rational_chart_tokens(
    values: Iterable[Iterable[Iterable[Fraction | int | np.integer]]],
    /,
) -> SourceBoundaryRationalChartBank:
    """Encode exact chart coordinates as reduced numerator/denominator pairs."""
    rows: list[tuple[tuple[_RationalToken, ...], ...]] = []
    distinct: dict[
        tuple[tuple[_RationalToken, ...], ...], tuple[tuple[_RationalToken, ...], ...]
    ] = {}
    for row in values:
        points: list[tuple[_RationalToken, ...]] = []
        for point in row:
            coordinates: list[_RationalToken] = []
            for value in point:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (Fraction, int, np.integer)
                ):
                    raise TypeError(
                        "Exact chart coordinates must be Fraction or integer values."
                    )
                exact = value if isinstance(value, Fraction) else Fraction(int(value))
                coordinates.append((exact.numerator, exact.denominator))
            points.append(tuple(coordinates))
        canonical_row = tuple(points)
        rows.append(distinct.setdefault(canonical_row, canonical_row))
    return tuple(rows)


def _validated_rational_chart_tokens(
    supplied: SourceBoundaryRationalChartBank | None,
    realization: np.ndarray,
    count: int,
    corners: int,
    dimension: int,
    name: str,
    /,
) -> SourceBoundaryRationalChartBank:
    if supplied is None:
        exact = (
            tuple(
                tuple(Fraction.from_float(float(value)) for value in point)
                for point in row
            )
            for row in realization
        )
        return rational_chart_tokens(exact)
    if len(supplied) != count:
        raise ValueError(f"{name} must contain one exact chart row per witness row.")
    result: list[tuple[tuple[_RationalToken, ...], ...]] = []
    distinct: dict[
        tuple[tuple[_RationalToken, ...], ...], tuple[tuple[_RationalToken, ...], ...]
    ] = {}
    for row_index, row in enumerate(supplied):
        if len(row) != corners:
            raise ValueError(f"{name} chart rows have the wrong corner count.")
        points: list[tuple[_RationalToken, ...]] = []
        for corner_index, point in enumerate(row):
            if len(point) != dimension:
                raise ValueError(
                    f"{name} chart points have the wrong coordinate dimension."
                )
            coordinates: list[_RationalToken] = []
            for axis, token in enumerate(point):
                if (
                    not isinstance(token, tuple)
                    or len(token) != 2
                    or any(
                        isinstance(value, (bool, np.bool_))
                        or not isinstance(value, (int, np.integer))
                        for value in token
                    )
                ):
                    raise TypeError(
                        f"{name} coordinates must be integer numerator/denominator pairs."
                    )
                numerator, denominator = (int(value) for value in token)
                if denominator <= 0:
                    raise ValueError(f"{name} denominators must be positive.")
                exact = Fraction(numerator, denominator)
                canonical = (exact.numerator, exact.denominator)
                represented = float(realization[row_index, corner_index, axis])
                if (
                    canonical != (numerator, denominator)
                    or not np.isfinite(represented)
                    or float(exact) != represented
                ):
                    raise ValueError(
                        f"{name} exact coordinate differs from its binary64 realization."
                    )
                coordinates.append(canonical)
            points.append(tuple(coordinates))
        canonical_row = tuple(points)
        result.append(distinct.setdefault(canonical_row, canonical_row))
    return tuple(result)


def _fractions_from_chart_tokens(
    bank: SourceBoundaryRationalChartBank, row: int, /
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(numerator, denominator) for numerator, denominator in point)
        for point in bank[row]
    )


class SourceBoundaryChartWitness(StrictModule, NonTrainableState):
    """Cell-bound source chart pieces for curves and oriented surface boundaries.

    Standard targets use parent-cell reference charts. Polyhedral targets carry
    actual face-triangle vertex IDs; their 2D reference rows may be virtual
    subtriangles of that actual target triangle and never invent polyhedron
    volume-reference coordinates. Exact rational chart-token banks are the
    topology authority; float arrays are checked rounded realizations and never
    substitute for exact partition or ancestry validation.
    """

    mesh_id: str
    geometry_id: str
    topology_id: str
    source_id: str
    source_revision: str
    source_query_id: str
    chart_dimension: int
    reference_dimension: int
    cell_global_ids: np.ndarray
    entity_global_ids: np.ndarray
    local_entity_indices: np.ndarray
    target_reference_vertices: np.ndarray
    target_reference_rational_tokens: SourceBoundaryRationalChartBank = eqx.field(
        static=True
    )
    target_face_vertex_global_ids: np.ndarray
    source_chart_ids: tuple[str, ...]
    source_entity_ids: tuple[str, ...]
    occurrence_paths: tuple[tuple[str, ...], ...]
    source_parameters: np.ndarray
    source_parameter_rational_tokens: SourceBoundaryRationalChartBank = eqx.field(
        static=True
    )
    orientations: np.ndarray
    implicit_cell_keys: np.ndarray
    implicit_root_edge_keys: np.ndarray
    witness_id: str

    def __init__(
        self,
        *,
        mesh_id: str,
        geometry_id: str,
        topology_id: str,
        source_id: str,
        source_revision: str,
        source_query_id: str,
        chart_dimension: int,
        reference_dimension: int,
        cell_global_ids: ConvertibleToArray,
        entity_global_ids: ConvertibleToArray,
        local_entity_indices: ConvertibleToArray,
        target_reference_vertices: ConvertibleToArray,
        target_reference_rational_tokens: SourceBoundaryRationalChartBank | None = None,
        source_chart_ids: tuple[str, ...],
        source_entity_ids: tuple[str, ...],
        occurrence_paths: tuple[tuple[str, ...], ...],
        source_parameters: ConvertibleToArray,
        source_parameter_rational_tokens: SourceBoundaryRationalChartBank | None = None,
        orientations: ConvertibleToArray,
        implicit_cell_keys: ConvertibleToArray | None = None,
        implicit_root_edge_keys: ConvertibleToArray | None = None,
        target_face_vertex_global_ids: ConvertibleToArray | None = None,
    ) -> None:
        identity_values = tuple(
            canonical_identifier(value, name)
            for value, name in (
                (mesh_id, "mesh_id"),
                (geometry_id, "geometry_id"),
                (topology_id, "topology_id"),
                (source_id, "source_id"),
                (source_revision, "source_revision"),
                (source_query_id, "source_query_id"),
            )
        )
        if type(chart_dimension) is not int or chart_dimension not in (1, 2):
            raise ValueError("chart_dimension must be one or two.")
        if (
            type(reference_dimension) is not int
            or reference_dimension < chart_dimension
            or reference_dimension > 3
        ):
            raise ValueError(
                "reference_dimension must contain the source chart and be at most three."
            )
        cell_ids = np.asarray(cell_global_ids)
        entity_ids = np.asarray(entity_global_ids)
        local_entity_values = np.asarray(local_entity_indices)
        if any(
            value.dtype.kind not in "iu"
            for value in (cell_ids, entity_ids, local_entity_values)
        ):
            raise TypeError(
                "Chart witness cell, entity, and local-entity IDs must be integers."
            )
        if (
            cell_ids.dtype.kind == "u"
            and cell_ids.size
            and np.max(cell_ids) > np.iinfo(np.int64).max
        ) or (
            entity_ids.dtype.kind == "u"
            and entity_ids.size
            and np.max(entity_ids) > np.iinfo(np.int64).max
        ):
            raise ValueError(
                "Chart witness IDs must fit the canonical nonnegative int64 range."
            )
        if local_entity_values.size and (
            np.min(local_entity_values) < -1
            or np.max(local_entity_values) > np.iinfo(np.int32).max
        ):
            raise ValueError(
                "Chart witness local-entity indices exceed the supported range."
            )
        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        entity_ids = np.asarray(entity_ids, dtype=np.int64)
        local_entities = np.asarray(local_entity_values, dtype=np.int32)
        count = cell_ids.size
        corners = chart_dimension + 1
        target = np.asarray(target_reference_vertices, dtype=np.float64)
        parameters = np.asarray(source_parameters, dtype=np.float64)
        if target_face_vertex_global_ids is None:
            face_vertices = np.empty((0, 3), dtype=np.int64)
        else:
            raw_face_vertices = np.asarray(target_face_vertex_global_ids)
            if raw_face_vertices.size == 0:
                face_vertices = np.empty((0, 3), dtype=np.int64)
            elif (
                raw_face_vertices.shape != (count, 3)
                or raw_face_vertices.dtype.kind not in "iu"
            ):
                raise TypeError(
                    "target_face_vertex_global_ids must be an integer (rows, 3) bank."
                )
            elif (
                raw_face_vertices.dtype.kind == "u"
                and np.max(raw_face_vertices) > np.iinfo(np.int64).max
            ):
                raise ValueError("Polyhedral target vertex IDs must fit canonical int64.")
            else:
                face_vertices = np.asarray(raw_face_vertices, dtype=np.int64)
                if (
                    np.any(face_vertices < 0)
                    or np.any(face_vertices[:, 0] == face_vertices[:, 1])
                    or np.any(face_vertices[:, 1] == face_vertices[:, 2])
                    or np.any(face_vertices[:, 2] == face_vertices[:, 0])
                ):
                    raise ValueError(
                        "Polyhedral target triangles require distinct nonnegative vertex IDs."
                    )
        orientation_values = np.asarray(orientations)
        if orientation_values.dtype.kind not in "iu" or np.any(
            ~np.isin(orientation_values, (-1, 1))
        ):
            raise TypeError("Chart witness orientations must be integer signs ±1.")
        orientation = np.asarray(orientation_values, dtype=np.int8)

        if (implicit_cell_keys is None) != (implicit_root_edge_keys is None):
            raise ValueError(
                "Implicit scaffold cell and root-edge keys must be supplied together."
            )
        if implicit_cell_keys is None:
            root_cells = np.empty((0, 4), dtype=np.int64)
            root_edges = np.empty((0, 3, 2, 3), dtype=np.int64)
        else:
            cell_values = np.asarray(implicit_cell_keys)
            edge_values = np.asarray(implicit_root_edge_keys)
            if cell_values.dtype.kind not in "iu" or edge_values.dtype.kind not in "iu":
                raise TypeError("Implicit scaffold keys must be integer arrays.")
            if (
                cell_values.dtype.kind == "u"
                and cell_values.size
                and np.max(cell_values) > np.iinfo(np.int64).max
            ) or (
                edge_values.dtype.kind == "u"
                and edge_values.size
                and np.max(edge_values) > np.iinfo(np.int64).max
            ):
                raise ValueError(
                    "Implicit scaffold keys must fit canonical int64 values."
                )
            root_cells = np.asarray(cell_values, dtype=np.int64)
            root_edges = np.asarray(edge_values, dtype=np.int64)
        charts = tuple(
            canonical_identifier(value, "source_chart_id") for value in source_chart_ids
        )
        entities = tuple(
            canonical_identifier(value, "source_entity_id") for value in source_entity_ids
        )
        paths = tuple(
            tuple(canonical_identifier(part, "occurrence_path") for part in path)
            for path in occurrence_paths
        )
        if (
            cell_ids.ndim != 1
            or count == 0
            or entity_ids.shape != (count,)
            or local_entities.shape != (count,)
            or target.shape != (count, corners, reference_dimension)
            or parameters.shape != (count, corners, chart_dimension)
            or orientation.shape != (count,)
            or len(charts) != count
            or len(entities) != count
            or len(paths) != count
            or root_cells.shape not in ((0, 4), (count, 4))
            or root_edges.shape not in ((0, 3, 2, 3), (count, 3, 2, 3))
            or root_cells.shape[0] != root_edges.shape[0]
            or np.any(root_cells < 0)
            or np.any(root_edges < 0)
            or np.any(cell_ids < 0)
            or np.any(entity_ids < 0)
            or np.any(~np.isfinite(target))
            or np.any(~np.isfinite(parameters))
            or np.any(~np.isin(orientation, (-1, 1)))
            or (
                face_vertices.shape[0] == 0
                and (
                    (
                        chart_dimension == reference_dimension
                        and np.any(local_entities != -1)
                    )
                    or (
                        chart_dimension < reference_dimension
                        and np.any(local_entities < 0)
                    )
                )
            )
            or (
                face_vertices.shape[0] != 0
                and (
                    face_vertices.shape != (count, 3)
                    or chart_dimension != 2
                    or reference_dimension != 2
                    or np.any(local_entities < 0)
                )
            )
        ):
            raise ValueError(
                "SourceBoundaryChartWitness rows and source identities must align."
            )
        target_rational_tokens = _validated_rational_chart_tokens(
            target_reference_rational_tokens,
            target,
            count,
            corners,
            reference_dimension,
            "target_reference_rational_tokens",
        )
        source_rational_tokens = _validated_rational_chart_tokens(
            source_parameter_rational_tokens,
            parameters,
            count,
            corners,
            chart_dimension,
            "source_parameter_rational_tokens",
        )
        # Validate each distinct exact chart pair without retaining two full
        # Fraction banks alongside the compact tokens and binary64 arrays.
        for target_tokens, source_tokens in dict.fromkeys(
            zip(target_rational_tokens, source_rational_tokens, strict=True)
        ):
            target_exact = _fractions_from_chart_tokens((target_tokens,), 0)
            source_exact = _fractions_from_chart_tokens((source_tokens,), 0)
            if face_vertices.shape[0] != 0:
                for point in target_exact:
                    if point[0] < 0 or point[1] < 0 or point[0] + point[1] > 1:
                        raise ValueError(
                            "Polyhedral target face chart points must stay in the parent triangle."
                        )
            if chart_dimension == 1:
                target_nondegenerate = target_exact[0] != target_exact[1]
                source_nondegenerate = source_exact[0] != source_exact[1]
            else:
                target_first = tuple(
                    target_exact[1][axis] - target_exact[0][axis]
                    for axis in range(reference_dimension)
                )
                target_second = tuple(
                    target_exact[2][axis] - target_exact[0][axis]
                    for axis in range(reference_dimension)
                )
                source_first = tuple(
                    source_exact[1][axis] - source_exact[0][axis] for axis in range(2)
                )
                source_second = tuple(
                    source_exact[2][axis] - source_exact[0][axis] for axis in range(2)
                )
                target_nonzero = any(
                    target_first[left] * target_second[right]
                    - target_first[right] * target_second[left]
                    for left in range(reference_dimension)
                    for right in range(left + 1, reference_dimension)
                )
                source_nonzero = (
                    source_first[0] * source_second[1]
                    - source_first[1] * source_second[0]
                ) != 0
                target_nondegenerate = bool(target_nonzero)
                source_nondegenerate = source_nonzero
            if not target_nondegenerate or not source_nondegenerate:
                raise ValueError(
                    "SourceBoundaryChartWitness contains a degenerate exact chart piece."
                )
        # Most witnesses have one row per cell. Resolve later sort keys only
        # for ties instead of retaining expanded float chart tuples per row.
        order: list[int] = []
        for _, group in groupby(
            sorted(range(count), key=lambda row: int(cell_ids[row])),
            key=lambda row: int(cell_ids[row]),
        ):
            rows = list(group)
            if len(rows) > 1:
                rows.sort(
                    key=lambda row: (
                        int(local_entities[row]),
                        int(entity_ids[row]),
                        charts[row],
                        target_rational_tokens[row],
                        tuple(float(value) for value in target[row].reshape(-1)),
                        tuple(int(value) for value in face_vertices[row])
                        if face_vertices.shape[0]
                        else (),
                        tuple(float(value) for value in parameters[row].reshape(-1)),
                        source_rational_tokens[row],
                        entities[row],
                        paths[row],
                    )
                )
            order.extend(rows)
        arrays = (
            cell_ids[order],
            entity_ids[order],
            local_entities[order],
            target[order],
            face_vertices if face_vertices.shape[0] == 0 else face_vertices[order],
            parameters[order],
            orientation[order],
            root_cells if root_cells.shape[0] == 0 else root_cells[order],
            root_edges if root_edges.shape[0] == 0 else root_edges[order],
        )
        immutable: list[np.ndarray] = []
        for value in arrays:
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            immutable.append(copied)
        self.mesh_id, self.geometry_id, self.topology_id = identity_values[:3]
        self.source_id, self.source_revision, self.source_query_id = identity_values[3:]
        self.chart_dimension = chart_dimension
        self.reference_dimension = reference_dimension
        (
            self.cell_global_ids,
            self.entity_global_ids,
            self.local_entity_indices,
            self.target_reference_vertices,
            self.target_face_vertex_global_ids,
            self.source_parameters,
            self.orientations,
            self.implicit_cell_keys,
            self.implicit_root_edge_keys,
        ) = immutable
        self.source_chart_ids = tuple(charts[row] for row in order)
        self.source_entity_ids = tuple(entities[row] for row in order)
        self.occurrence_paths = tuple(paths[row] for row in order)
        self.target_reference_rational_tokens = tuple(
            target_rational_tokens[row] for row in order
        )
        self.source_parameter_rational_tokens = tuple(
            source_rational_tokens[row] for row in order
        )
        self.witness_id = canonical_fingerprint(
            {
                "kind": "source-boundary-chart-witness",
                "identities": identity_values,
                "dimensions": (chart_dimension, reference_dimension),
                "arrays": array_tree_fingerprint(
                    (
                        self.cell_global_ids,
                        self.entity_global_ids,
                        self.local_entity_indices,
                        self.target_reference_vertices,
                        self.target_face_vertex_global_ids,
                        self.source_parameters,
                        self.orientations,
                        self.implicit_cell_keys,
                        self.implicit_root_edge_keys,
                    )
                ),
                "charts": self.source_chart_ids,
                "entities": self.source_entity_ids,
                "occurrences": self.occurrence_paths,
                "exact_rational_charts": (
                    self.target_reference_rational_tokens,
                    self.source_parameter_rational_tokens,
                ),
            }
        )

    def target_reference_chart(self, row: int, /) -> tuple[tuple[Fraction, ...], ...]:
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
            raise IndexError("source-bound target chart row must be an integer.")
        row = int(row)
        if not 0 <= row < len(self.source_chart_ids):
            raise IndexError("source-bound target chart row is out of range.")
        return _fractions_from_chart_tokens(self.target_reference_rational_tokens, row)

    def source_parameter_chart(self, row: int, /) -> tuple[tuple[Fraction, ...], ...]:
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
            raise IndexError("source-bound source chart row must be an integer.")
        row = int(row)
        if not 0 <= row < len(self.source_chart_ids):
            raise IndexError("source-bound source chart row is out of range.")
        return _fractions_from_chart_tokens(self.source_parameter_rational_tokens, row)

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SourceBoundaryQuery,
        /,
    ) -> None:
        """Revalidate source chart ancestry against its current scientific operands."""
        from ._source_fidelity import validate_source_fidelity_witness

        validate_source_fidelity_witness(mesh, geometry, source, self)


class SourceBoundaryPointWitness(StrictModule, NonTrainableState):
    """Input-bound source point to target vertex correspondence for 0D scopes."""

    mesh_id: str
    geometry_id: str
    topology_id: str
    source_id: str
    source_revision: str
    source_query_id: str
    vertex_global_ids: np.ndarray
    source_entity_ids: tuple[str, ...]
    occurrence_paths: tuple[tuple[str, ...], ...]
    witness_id: str

    def __init__(
        self,
        *,
        mesh_id: str,
        geometry_id: str,
        topology_id: str,
        source_id: str,
        source_revision: str,
        source_query_id: str,
        vertex_global_ids: ConvertibleToArray,
        source_entity_ids: tuple[str, ...],
        occurrence_paths: tuple[tuple[str, ...], ...],
    ) -> None:
        identities = tuple(
            canonical_identifier(value, name)
            for value, name in (
                (mesh_id, "mesh_id"),
                (geometry_id, "geometry_id"),
                (topology_id, "topology_id"),
                (source_id, "source_id"),
                (source_revision, "source_revision"),
                (source_query_id, "source_query_id"),
            )
        )
        raw_ids = np.asarray(vertex_global_ids)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu" or raw_ids.size == 0:
            raise TypeError(
                "Point witness vertex_global_ids must be a nonempty integer vector."
            )
        if raw_ids.dtype.kind == "u" and np.max(raw_ids) > np.iinfo(np.int64).max:
            raise ValueError("Point witness vertex IDs must fit canonical int64 values.")
        ids = np.asarray(raw_ids, dtype=np.int64)
        entities = tuple(
            canonical_identifier(value, "source_entity_id") for value in source_entity_ids
        )
        paths = tuple(
            tuple(canonical_identifier(part, "occurrence_path") for part in path)
            for path in occurrence_paths
        )
        if (
            np.any(ids < 0)
            or len(entities) != ids.size
            or len(paths) != ids.size
            or np.unique(ids).size != ids.size
            or len(set(zip(entities, paths, strict=True))) != ids.size
        ):
            raise ValueError(
                "Point witness requires a one-to-one target vertex/source occurrence map."
            )
        order = sorted(
            range(ids.size),
            key=lambda row: (int(ids[row]), entities[row], paths[row]),
        )
        ids = np.array(ids[order], dtype=np.int64, copy=True)
        ids.setflags(write=False)
        self.mesh_id, self.geometry_id, self.topology_id = identities[:3]
        self.source_id, self.source_revision, self.source_query_id = identities[3:]
        self.vertex_global_ids = ids
        self.source_entity_ids = tuple(entities[row] for row in order)
        self.occurrence_paths = tuple(paths[row] for row in order)
        self.witness_id = canonical_fingerprint(
            {
                "kind": "source-boundary-point-witness",
                "identities": identities,
                "vertices": array_tree_fingerprint(ids),
                "entities": self.source_entity_ids,
                "occurrences": self.occurrence_paths,
            }
        )

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SourceBoundaryQuery,
        /,
    ) -> None:
        from ._source_fidelity import validate_source_fidelity_witness

        validate_source_fidelity_witness(mesh, geometry, source, self)


@final
class SurfaceEnvelopeDistanceWitness(StrictModule, NonTrainableState):
    """Continuous distance evidence for a distinct PWL surface envelope.

    ``reproduction`` optionally carries the record of the kernel run that
    produced the envelope; validation trusts it only when it binds exactly the
    witnessed source triangles, controls, and envelope, and otherwise reruns
    the kernel. It is evidence about the bound content, not part of its identity.
    """

    mesh_id: str
    geometry_id: str
    topology_id: str
    source_id: str
    source_revision: str
    source_query_id: str
    offset: float = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    certificate_spacing: float = eqx.field(static=True)
    bounds: np.ndarray
    envelope_vertices: np.ndarray
    envelope_triangles: np.ndarray
    target_vertex_global_ids: np.ndarray
    target_facet_global_ids: np.ndarray
    witness_id: str
    reproduction: SurfaceEnvelopeReproduction | None

    def __init__(
        self,
        *,
        mesh_id: str,
        geometry_id: str,
        topology_id: str,
        source_id: str,
        source_revision: str,
        source_query_id: str,
        offset: float,
        spacing: float,
        certificate_spacing: float,
        bounds: ConvertibleToArray,
        envelope_vertices: ConvertibleToArray,
        envelope_triangles: ConvertibleToArray,
        target_vertex_global_ids: ConvertibleToArray,
        target_facet_global_ids: ConvertibleToArray,
        reproduction: SurfaceEnvelopeReproduction | None = None,
    ) -> None:
        from .reconstruction._envelope import SurfaceEnvelopeReproduction

        if reproduction is not None and not isinstance(
            reproduction, SurfaceEnvelopeReproduction
        ):
            raise TypeError(
                "reproduction must be a SurfaceEnvelopeReproduction when provided."
            )
        identities = tuple(
            canonical_identifier(value, name)
            for value, name in (
                (mesh_id, "mesh_id"),
                (geometry_id, "geometry_id"),
                (topology_id, "topology_id"),
                (source_id, "source_id"),
                (source_revision, "source_revision"),
                (source_query_id, "source_query_id"),
            )
        )
        controls = tuple(float(value) for value in (offset, spacing, certificate_spacing))
        if any(not np.isfinite(value) or value <= 0.0 for value in controls):
            raise ValueError(
                "Surface envelope offset and spacings must be finite and positive."
            )
        bound_values = np.asarray(bounds, dtype=np.float64)
        points = np.asarray(envelope_vertices, dtype=np.float64)
        raw_triangles = np.asarray(envelope_triangles)
        raw_vertex_ids = np.asarray(target_vertex_global_ids)
        raw_facet_ids = np.asarray(target_facet_global_ids)
        if (
            bound_values.shape != (4,)
            or np.any(~np.isfinite(bound_values))
            or bound_values[0] < 0.0
            or bound_values[1] < 0.0
            or bound_values[2] <= 0.0
            or bound_values[3] < 0.0
            or bound_values[2] > controls[0]
            or points.ndim != 2
            or points.shape[1] != 3
            or points.shape[0] == 0
            or np.any(~np.isfinite(points))
            or raw_triangles.ndim != 2
            or raw_triangles.shape[1] != 3
            or raw_triangles.shape[0] == 0
            or raw_triangles.dtype.kind not in "iu"
            or np.any(raw_triangles < 0)
            or np.any(raw_triangles >= points.shape[0])
            or np.any(raw_triangles[:, 0] == raw_triangles[:, 1])
            or np.any(raw_triangles[:, 1] == raw_triangles[:, 2])
            or np.any(raw_triangles[:, 2] == raw_triangles[:, 0])
            or raw_vertex_ids.shape != (points.shape[0],)
            or raw_vertex_ids.dtype.kind not in "iu"
            or raw_facet_ids.shape != (raw_triangles.shape[0],)
            or raw_facet_ids.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Surface envelope witness bounds, triangles, or target identities are malformed."
            )
        if (
            raw_vertex_ids.dtype.kind == "u"
            and raw_vertex_ids.size
            and np.max(raw_vertex_ids) > np.iinfo(np.int64).max
        ) or (
            raw_facet_ids.dtype.kind == "u"
            and raw_facet_ids.size
            and np.max(raw_facet_ids) > np.iinfo(np.int64).max
        ):
            raise ValueError(
                "Surface envelope target identities must fit canonical int64 values."
            )
        triangles = np.asarray(raw_triangles, dtype=np.int64)
        vertex_ids = np.asarray(raw_vertex_ids, dtype=np.int64)
        facet_ids = np.asarray(raw_facet_ids, dtype=np.int64)
        if (
            np.any(vertex_ids < 0)
            or np.unique(vertex_ids).size != vertex_ids.size
            or np.any(facet_ids < 0)
            or np.unique(facet_ids).size != facet_ids.size
        ):
            raise ValueError(
                "Surface envelope target vertex/facet identities must be unique and nonnegative."
            )
        arrays = tuple(
            np.array(value, copy=True)
            for value in (bound_values, points, triangles, vertex_ids, facet_ids)
        )
        for value in arrays:
            value.setflags(write=False)
        self.mesh_id, self.geometry_id, self.topology_id = identities[:3]
        self.source_id, self.source_revision, self.source_query_id = identities[3:]
        self.offset, self.spacing, self.certificate_spacing = controls
        (
            self.bounds,
            self.envelope_vertices,
            self.envelope_triangles,
            self.target_vertex_global_ids,
            self.target_facet_global_ids,
        ) = arrays
        self.reproduction = reproduction
        self.witness_id = canonical_fingerprint(
            {
                "kind": "surface-envelope-distance-witness",
                "identities": identities,
                "controls": controls,
                "bounds": array_tree_fingerprint(self.bounds),
                "envelope": array_tree_fingerprint(
                    (self.envelope_vertices, self.envelope_triangles)
                ),
                "target": array_tree_fingerprint(
                    (self.target_vertex_global_ids, self.target_facet_global_ids)
                ),
            }
        )

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SourceBoundaryQuery,
        /,
    ) -> None:
        from ._source_fidelity import validate_source_fidelity_witness

        validate_source_fidelity_witness(mesh, geometry, source, self)


@final
class SourceBoundaryDistanceWitness(StrictModule, NonTrainableState):
    """Continuous two-directed distance evidence without chart ancestry."""

    mesh_id: str
    geometry_id: str
    topology_id: str
    source_id: str
    source_revision: str
    source_query_id: str
    sample_count: int = eqx.field(static=True)
    cover_radius: float = eqx.field(static=True)
    target_facet_global_ids: np.ndarray
    witness_id: str

    def __init__(
        self,
        *,
        mesh_id: str,
        geometry_id: str,
        topology_id: str,
        source_id: str,
        source_revision: str,
        source_query_id: str,
        sample_count: int,
        cover_radius: float,
        target_facet_global_ids: ConvertibleToArray,
    ) -> None:
        identities = tuple(
            canonical_identifier(value, name)
            for value, name in (
                (mesh_id, "mesh_id"),
                (geometry_id, "geometry_id"),
                (topology_id, "topology_id"),
                (source_id, "source_id"),
                (source_revision, "source_revision"),
                (source_query_id, "source_query_id"),
            )
        )
        if (
            isinstance(sample_count, (bool, np.bool_))
            or not isinstance(sample_count, (int, np.integer))
            or int(sample_count) <= 0
        ):
            raise ValueError(
                "sample_count must be the positive integer replayed source-point count."
            )
        if isinstance(cover_radius, (bool, np.bool_)):
            raise TypeError(
                "cover_radius must be a real positive distance, not a boolean."
            )
        cover_radius_ = float(cover_radius)
        if not np.isfinite(cover_radius_) or cover_radius_ <= 0.0:
            raise ValueError("cover_radius must be finite and positive.")
        raw_ids = np.asarray(target_facet_global_ids)
        if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu" or raw_ids.size == 0:
            raise TypeError("target_facet_global_ids must be a nonempty integer vector.")
        if raw_ids.dtype.kind == "u" and np.max(raw_ids) > np.iinfo(np.int64).max:
            raise ValueError("Target facet global IDs must fit canonical int64 values.")
        facet_ids = np.asarray(raw_ids, dtype=np.int64)
        if np.any(facet_ids < 0) or np.unique(facet_ids).size != facet_ids.size:
            raise ValueError("Target facet global IDs must be unique and nonnegative.")
        facet_ids = np.array(facet_ids, copy=True)
        facet_ids.setflags(write=False)
        self.mesh_id, self.geometry_id, self.topology_id = identities[:3]
        self.source_id, self.source_revision, self.source_query_id = identities[3:]
        self.sample_count = int(sample_count)
        self.cover_radius = cover_radius_
        self.target_facet_global_ids = facet_ids
        self.witness_id = canonical_fingerprint(
            {
                "kind": "source-bound-distance-witness",
                "identities": identities,
                "sample_count": self.sample_count,
                "cover_radius": self.cover_radius,
                "target_facets": array_tree_fingerprint(facet_ids),
            }
        )

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: SourceBoundaryCoverQuery,
        /,
    ) -> None:
        from ._source_fidelity import validate_source_fidelity_witness

        validate_source_fidelity_witness(mesh, geometry, source, self)


@runtime_checkable
class SourceBoundaryQuery(Protocol):
    """Geometry-owned batched boundary queries for one source revision."""

    @property
    def source_id(self) -> str: ...

    @property
    def source_revision(self) -> str: ...

    @property
    def ambient_dimension(self) -> int: ...

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance: ...


@runtime_checkable
class SourceBoundaryProjectionResult(Protocol):
    """Source/frame-bound continuous projection evidence for batched points.

    distance_lower_bounds are certified lower distances and residuals are
    certified upper distances to the returned points. Only UNIQUE and SEAM
    outcomes are resolved for projection-specific acceptance.
    """

    source_id: str
    source_revision: str
    frame_id: str
    queries: ArrayLike
    points: ArrayLike
    parameters: ArrayLike
    residuals: ArrayLike
    distance_lower_bounds: ArrayLike
    status: ArrayLike
    normals: ArrayLike
    tangents: ArrayLike
    occurrence_paths: tuple[tuple[str, ...], ...]

    def entity_ids(self, /) -> tuple[str, ...]:
        """Return each known classified source ID; failures return an empty ID."""
        ...


@runtime_checkable
class SourceBoundaryProjectionQuery(SourceBoundaryQuery, Protocol):
    """Optional isolated projection capability with explicit source scopes."""

    @property
    def frame_id(self) -> str: ...

    def project_boundary(
        self,
        points: np.ndarray,
        /,
        *,
        entity_ids: tuple[str, ...] | None = None,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> SourceBoundaryProjectionResult: ...


@runtime_checkable
class SourceBoundaryCoverQuery(SourceBoundaryQuery, Protocol):
    """Optional source cover refined by a requested physical radius."""

    def boundary_cover(self, maximum_radius: float, /) -> SourceBoundarySamples: ...


@runtime_checkable
class SourceBoundaryChartQuery(Protocol):
    """A source's independent complete trim/chart cover capability."""

    def boundary_chart_cover(self, /) -> SourceBoundaryChartCover: ...


__all__ = [
    "SourceAmbientDim",
    "SourceBoundSemantics",
    "SourceBoundaryCellDim",
    "SurfaceEnvelopeDistanceWitness",
    "SourceBoundaryDistanceWitness",
    "SourceBoundaryChartWitness",
    "SourceBoundaryRationalChartBank",
    "SourceBoundaryChartCover",
    "SourceBoundaryPointWitness",
    "SourceBoundaryChartQuery",
    "SourceBoundaryDistance",
    "SourceBoundaryOccurrenceDim",
    "SourceBoundaryQuery",
    "SourceBoundaryProjectionQuery",
    "SourceBoundaryProjectionResult",
    "SourceBoundarySamples",
    "SourceProjectionStatus",
    "SourceBoundaryCoverQuery",
    "rational_chart_tokens",
    "SourceBoundarySideDim",
]
