"""H1, discontinuous-form, and conservative FV surface chart transfer."""

from __future__ import annotations

from fractions import Fraction
from typing import assert_never, Literal, NamedTuple, Protocol, TypeAlias

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...exterior._form_type import FormTwist
from ...linalg import ComplexMap
from ...sparse import EdgeRelation, SparseLinearMap
from ...typing import parse
from .._cell_geometry import (
    CellGeometrySpec,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from .._cell_mesh import CellMesh
from .._surface_chart_records import PreparedSurfaceChartDeformation
from ..finite_volume._unstructured import UnstructuredFiniteVolumeDiscretization
from ._de_rham import _ReferenceChartPiece, FiniteElementDeRhamComplex
from ._surface_chart_compatible import (
    prepare_surface_chart_compatible_transfer,
    PreparedSurfaceChartCompatibleTransfer,
)


SurfaceChartFieldSemantics: TypeAlias = Literal["h1", "hcurl", "hdiv", "dg"]


class _ChartDeformation(Protocol):
    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    source_geometry_id: str
    target_geometry_id: str
    source_cell_measures: Array
    target_cell_measures: Array
    source_measure_errors: Array
    target_measure_errors: Array
    deformation_id: str


class PreparedSurfaceChartFieldTransfer(NamedTuple):
    complex_map: ComplexMap
    degree: int
    commuting_defects: Array
    form_twist: FormTwist
    deformation_id: str


class PreparedSurfaceChartFiniteVolumeContents(NamedTuple):
    """Conservative cell-average map and its volume-paired adjoint.

    ``primal`` maps old cell-average densities to target cell averages. Its raw
    transpose is the unweighted dual pullback. ``hilbert_adjoint`` instead uses
    source and target physical cell measures.
    """

    primal: SparseLinearMap
    hilbert_adjoint: SparseLinearMap
    source_measures: Array
    target_measures: Array
    source_coverage_defects: Array
    conservation_defect: Array
    deformation_id: str


def _triangle_quadrature(
    triangle: tuple[tuple[Fraction, Fraction], ...], order: int, /
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    first, second = np.meshgrid(nodes, nodes, indexing="ij")
    parameters = np.column_stack(
        (first.reshape(-1), ((1.0 - first) * second).reshape(-1))
    )
    origin = np.asarray(triangle[0], dtype=np.float64)
    frame = np.asarray(
        (
            (triangle[1][0] - triangle[0][0], triangle[2][0] - triangle[0][0]),
            (triangle[1][1] - triangle[0][1], triangle[2][1] - triangle[0][1]),
        ),
        dtype=np.float64,
    )
    determinant = abs(
        float(
            (triangle[1][0] - triangle[0][0]) * (triangle[2][1] - triangle[0][1])
            - (triangle[2][0] - triangle[0][0]) * (triangle[1][1] - triangle[0][1])
        )
    )
    quadrature = (weights[:, None] * weights[None, :] * (1.0 - first)).reshape(-1)
    return origin + parameters @ frame.T, quadrature * determinant


def _cell_row_lookup(mesh: CellMesh, /) -> dict[int, tuple[int, int]]:
    rows: dict[int, tuple[int, int]] = {}
    for block_index, block in enumerate(mesh.blocks):
        for cell_row, identity in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            if int(identity) in rows:
                raise ValueError("Surface chart FV cell IDs must be unique.")
            rows[int(identity)] = (block_index, cell_row)
    return rows


def _physical_piece_measure(
    geometry: CellGeometrySpec,
    mesh: CellMesh,
    cell_rows: dict[int, tuple[int, int]],
    cell_global_id: int,
    triangle: tuple[tuple[Fraction, Fraction], ...],
    order: int,
    /,
) -> float:
    block_index, local_cell = cell_rows[cell_global_id]
    elements, routes, coordinates = geometry.resolve(mesh)
    element = elements[block_index]
    if not isinstance(element, ScalarCoordinateElement):
        raise ValueError("Surface FV chart integration requires a scalar coordinate map.")
    points, weights = _triangle_quadrature(triangle, order)
    values_and_gradients = (
        element.tabulate_cell(local_cell, points)
        if isinstance(element, PerCellRestrictedCellGeometryElement)
        else element.tabulate(points)
    )
    gradients = np.asarray(values_and_gradients[1], dtype=np.float64)
    coordinate_values = np.asarray(coordinates, dtype=np.float64)
    cell_routes = np.asarray(routes[block_index][local_cell], dtype=np.int64)
    if gradients.shape != (points.shape[0], cell_routes.size, 2):
        raise ValueError("Surface FV coordinate gradients have inconsistent extents.")
    jacobians = np.einsum(
        "qvd,va->qad", gradients, coordinate_values[cell_routes], optimize=True
    )
    metric_00 = np.sum(jacobians[:, :, 0] ** 2, axis=1)
    metric_11 = np.sum(jacobians[:, :, 1] ** 2, axis=1)
    metric_01 = np.sum(jacobians[:, :, 0] * jacobians[:, :, 1], axis=1)
    squared_density = metric_00 * metric_11 - metric_01**2
    if np.any(~np.isfinite(squared_density) | (squared_density <= 0.0)):
        raise ValueError("Surface FV chart integration encountered a singular metric.")
    measure = float(np.dot(weights, np.sqrt(squared_density)))
    if not np.isfinite(measure) or measure <= 0.0:
        raise ValueError("Surface FV chart piece has no positive physical measure.")
    return measure


def _piece_measure_with_error(
    geometry: CellGeometrySpec,
    mesh: CellMesh,
    cell_rows: dict[int, tuple[int, int]],
    cell_global_id: int,
    triangle: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> tuple[float, float]:
    coarse = _physical_piece_measure(
        geometry, mesh, cell_rows, cell_global_id, triangle, 12
    )
    fine = _physical_piece_measure(
        geometry, mesh, cell_rows, cell_global_id, triangle, 20
    )
    roundoff = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(fine))
    return fine, abs(fine - coarse) + roundoff


def _validate_fv_endpoint(
    discretization: UnstructuredFiniteVolumeDiscretization,
    expected_mesh: CellMesh,
    expected_geometry: CellGeometrySpec,
    expected_measures: ArrayLike,
    expected_measure_errors: ArrayLike,
    role: str,
    /,
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    if (
        discretization.mesh is None
        or discretization.mesh.mesh_id != expected_mesh.mesh_id
    ):
        raise ValueError(f"Surface chart FV {role} mesh differs from its deformation.")
    expected_coordinates = np.asarray(
        expected_geometry.resolve(expected_mesh)[2], dtype=np.float64
    )
    actual_coordinates = np.asarray(discretization.vertices, dtype=np.float64)
    if not np.array_equal(actual_coordinates, expected_coordinates):
        raise ValueError(f"Surface chart FV {role} geometry differs from its map.")
    rows = _cell_row_lookup(expected_mesh)
    expected_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in expected_mesh.blocks]
    )
    if not np.array_equal(np.asarray(discretization.cell_global_ids), expected_ids):
        raise ValueError(
            f"Surface chart FV {role} rows differ from its scientific cells."
        )
    measures = np.asarray(discretization.cell_volumes, dtype=np.float64)
    reference = np.asarray(expected_measures, dtype=np.float64)
    errors = np.asarray(expected_measure_errors, dtype=np.float64)
    if (
        measures.shape != expected_ids.shape
        or reference.shape != measures.shape
        or errors.shape != measures.shape
        or not np.all(np.isfinite(measures))
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(errors))
        or np.any(measures <= 0.0)
        or np.any(errors < 0.0)
    ):
        raise ValueError(f"Surface chart FV {role} physical measures are malformed.")
    tolerance = errors + 64.0 * np.finfo(np.float64).eps * np.maximum(
        1.0, np.abs(reference)
    )
    if np.any(np.abs(measures - reference) > tolerance):
        raise ValueError(f"Surface chart FV {role} measures disagree with its map.")
    return measures, rows


def _prepare_finite_volume_transfer(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    prepared: _ChartDeformation,
    pieces: tuple[_ReferenceChartPiece, ...],
    /,
) -> PreparedSurfaceChartFiniteVolumeContents:
    if source.cell_dimension != 2 or target.cell_dimension != 2:
        raise ValueError("Surface chart FV transfer requires two-dimensional cells.")
    if source.mesh is None or target.mesh is None:
        raise ValueError("Surface chart FV transfer requires resident endpoint meshes.")
    source_measures, source_cell_rows = _validate_fv_endpoint(
        source,
        prepared.source_mesh,
        prepared.source_geometry,
        prepared.source_cell_measures,
        prepared.source_measure_errors,
        "source",
    )
    target_measures, _ = _validate_fv_endpoint(
        target,
        prepared.target_mesh,
        prepared.target_geometry,
        prepared.target_cell_measures,
        prepared.target_measure_errors,
        "target",
    )
    source_ids = np.asarray(source.cell_global_ids, dtype=np.int64)
    target_ids = np.asarray(target.cell_global_ids, dtype=np.int64)
    source_index = {int(identity): row for row, identity in enumerate(source_ids)}
    target_index = {int(identity): row for row, identity in enumerate(target_ids)}
    source_piece_totals = np.zeros(source.cell_count, dtype=np.float64)
    source_piece_errors = np.zeros(source.cell_count, dtype=np.float64)
    pair_values: dict[tuple[int, int], float] = {}
    for piece in pieces:
        source_global_id = int(source_ids[piece.source_row])
        target_global_id = int(target_ids[piece.target_row])
        source_row = source_index[source_global_id]
        target_row = target_index[target_global_id]
        area, error = _piece_measure_with_error(
            prepared.source_geometry,
            prepared.source_mesh,
            source_cell_rows,
            source_global_id,
            piece.source_vertices,
        )
        source_piece_totals[source_row] += area
        source_piece_errors[source_row] += error
        key = (target_row, source_row)
        pair_values[key] = pair_values.get(key, 0.0) + area / target_measures[target_row]
    defects = source_piece_totals - source_measures
    accepted = source_piece_errors + np.asarray(
        prepared.source_measure_errors, dtype=np.float64
    )
    accepted += 128.0 * np.finfo(np.float64).eps * np.maximum(1.0, source_measures)
    if np.any(np.abs(defects) > accepted):
        raise ValueError("Surface FV source material pieces fail physical area coverage.")
    ordered = sorted(pair_values.items())
    rows = np.asarray([key[0] for key, _ in ordered], dtype=np.int32)
    columns = np.asarray([key[1] for key, _ in ordered], dtype=np.int32)
    coefficients = np.asarray([value for _, value in ordered], dtype=np.float64)
    relation = EdgeRelation(
        columns,
        rows,
        source_size=source.cell_count,
        target_size=target.cell_count,
    )
    transfer_id = canonical_fingerprint(
        {
            "kind": "surface-chart-finite-volume-content-transfer",
            "deformation": prepared.deformation_id,
            "source_topology": source.topology_id,
            "target_topology": target.topology_id,
            "source_measures": source_measures,
            "target_measures": target_measures,
            "rows": rows,
            "columns": columns,
            "coefficients": coefficients,
        }
    )
    primal = SparseLinearMap(
        relation,
        jnp.asarray(coefficients, dtype=source.cell_volumes.dtype),
        operator_id=transfer_id,
    )
    adjoint_coefficients = coefficients * target_measures[rows] / source_measures[columns]
    adjoint_relation = EdgeRelation(
        rows,
        columns,
        source_size=target.cell_count,
        target_size=source.cell_count,
    )
    hilbert_adjoint = SparseLinearMap(
        adjoint_relation,
        jnp.asarray(adjoint_coefficients, dtype=source.cell_volumes.dtype),
        operator_id=f"{transfer_id}:hilbert-adjoint",
    )
    observed = np.asarray(
        primal.transpose_mv(jnp.asarray(target_measures)), dtype=np.float64
    )
    conservation_vector = observed - source_measures
    conservation_defect = float(np.max(np.abs(conservation_vector), initial=0.0))
    if np.any(np.abs(conservation_vector) > accepted):
        raise ValueError(
            "Surface FV content action fails its physical conservation check."
        )
    return PreparedSurfaceChartFiniteVolumeContents(
        primal,
        hilbert_adjoint,
        jnp.asarray(source_measures, dtype=source.cell_volumes.dtype),
        jnp.asarray(target_measures, dtype=target.cell_volumes.dtype),
        jnp.asarray(defects, dtype=jnp.float64),
        jnp.asarray(conservation_defect, dtype=jnp.float64),
        prepared.deformation_id,
    )


def prepare_surface_chart_field_transfer(
    source: FiniteElementDeRhamComplex,
    target: FiniteElementDeRhamComplex,
    prepared: PreparedSurfaceChartDeformation,
    /,
    *,
    semantics: SurfaceChartFieldSemantics,
    commuting_tolerance: float = 2.0e-8,
) -> PreparedSurfaceChartFieldTransfer:
    selected = parse(semantics, SurfaceChartFieldSemantics, "semantics")
    match selected:
        case "h1":
            degree, transfer_twist = 0, source.primal_twist
        case "hcurl":
            degree, transfer_twist = 1, "untwisted"
        case "hdiv":
            degree, transfer_twist = 1, "twisted"
        case "dg":
            degree, transfer_twist = source.dimension, source.primal_twist
        case other:
            assert_never(other)
    prepared_transfer: PreparedSurfaceChartCompatibleTransfer = (
        prepare_surface_chart_compatible_transfer(
            source,
            target,
            prepared,
            commuting_tolerance=commuting_tolerance,
            form_twist=transfer_twist,
        )
    )
    return PreparedSurfaceChartFieldTransfer(
        prepared_transfer.complex_map,
        degree,
        prepared_transfer.commuting_defects,
        prepared_transfer.form_twist,
        prepared_transfer.deformation_id,
    )


def prepare_surface_chart_finite_volume_contents(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    prepared: PreparedSurfaceChartDeformation,
    /,
) -> PreparedSurfaceChartFiniteVolumeContents:
    _validate_fv_endpoint(
        source,
        prepared.source_mesh,
        prepared.source_geometry,
        prepared.source_cell_measures,
        prepared.source_measure_errors,
        "source",
    )
    _validate_fv_endpoint(
        target,
        prepared.target_mesh,
        prepared.target_geometry,
        prepared.target_cell_measures,
        prepared.target_measure_errors,
        "target",
    )
    pieces = tuple(
        _ReferenceChartPiece(
            piece.source_row,
            piece.target_row,
            piece.exact_target_reference_vertices,
            piece.exact_source_reference_vertices,
            None,
        )
        for occurrence in prepared.occurrences
        for piece in occurrence.pieces
    )
    return _prepare_finite_volume_transfer(source, target, prepared, pieces)


__all__ = [
    "PreparedSurfaceChartFieldTransfer",
    "PreparedSurfaceChartFiniteVolumeContents",
    "SurfaceChartFieldSemantics",
    "prepare_surface_chart_field_transfer",
    "prepare_surface_chart_finite_volume_contents",
]
