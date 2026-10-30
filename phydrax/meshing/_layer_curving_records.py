from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Protocol, runtime_checkable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry import (
    _coordinate_coefficient_identity,
    CellGeometryElement,
    ScalarCoordinateElement,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import (
    coordinate_map_expressions,
    derivative,
    evaluate_expression,
    Expression,
    RationalPolynomial,
)
from ..discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
    RadicalElement,
)
from ..discretization._nested_reference import (
    nested_cell_reference_maps,
    NestedCellReferenceMap,
)
from ..discretization._nested_reference_types import (
    ExactReferenceChartExpression,
    reference_chart_arguments_identity,
    ReferenceCellMapWitnesses,
    validate_reference_cell_map_witnesses,
)
from ..discretization.fem import lagrange_element
from ..discretization.fem._reference import (
    FiniteElementSpec,
    lagrange_reference_node_data,
)
from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ..geometry._rational import fraction_endpoint
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryProjectionQuery,
    SourceBoundaryQuery,
    SourceProjectionStatus,
)
from ..geometry._source_fidelity import (
    SourceFidelityCertificate,
    validate_source_fidelity_witness,
)
from ..geometry._source_identity import source_entity_id
from ..geometry._source_interpolation import exact_surface_coordinate_expressions
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.brep._patches import surface_differential
from ..geometry.implicit._adaptive_discovery import AdaptiveImplicitBoundarySource
from ..geometry.implicit._source_charts import (
    _AdaptiveImplicitBoundaryCarrier,
    AdaptiveImplicitSourceQuery,
    ImplicitRootChartEvidence,
)
from ..linalg._exact import (
    ExactRationalAlgebraicFactorization,
    factor_exact_rational_algebraic,
)
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ._boundary_layer import BoundaryLayerEvidence


@runtime_checkable
class _NodalReferenceElement(Protocol):
    @property
    def reference_nodes(self) -> ArrayLike: ...

    @property
    def entity_dofs(self) -> tuple[tuple[tuple[int, ...], ...], ...]: ...


def _adaptive_implicit_source_query(
    source_query: SourceBoundaryProjectionQuery, /
) -> AdaptiveImplicitSourceQuery:
    if isinstance(source_query, AdaptiveImplicitBoundarySource):
        return source_query
    if isinstance(source_query, _AdaptiveImplicitBoundaryCarrier):
        return source_query
    raise TypeError("Adaptive layer source query must retain its native adaptive chart authority.")

def _fraction(value: object, /) -> Fraction:
    if isinstance(value, Fraction):
        return value
    raise ValueError("Exact source-reference maps must retain rational coordinates.")


def _fraction2(point: Sequence[object], /) -> tuple[Fraction, Fraction]:
    if len(point) != 2:
        raise ValueError("An exact source chart point must have two coordinates.")
    return (_fraction(point[0]), _fraction(point[1]))


def _fraction3(point: Sequence[object], /) -> tuple[Fraction, Fraction, Fraction]:
    if len(point) != 3:
        raise ValueError("An exact target reference point must have three coordinates.")
    return (_fraction(point[0]), _fraction(point[1]), _fraction(point[2]))
type _ExactReferencePoint2 = tuple[ExactScalar, ExactScalar]
type _ExactReferencePoint3 = tuple[ExactScalar, ExactScalar, ExactScalar]
type _BarycentricSolver2 = ExactRationalAlgebraicFactorization[RadicalElement]
type _BarycentricSolver3 = tuple[tuple[int, int], _BarycentricSolver2]


def _exact_scalar(value: object, /) -> ExactScalar:
    if isinstance(value, (Fraction, RadicalElement)):
        return value
    raise ValueError("Exact layer reference coordinates require canonical exact scalars.")


def _exact2(point: Sequence[object], /) -> _ExactReferencePoint2:
    if len(point) != 2:
        raise ValueError("An exact layer parameter must have two coordinates.")
    return (_exact_scalar(point[0]), _exact_scalar(point[1]))


def _exact3(point: Sequence[object], /) -> _ExactReferencePoint3:
    if len(point) != 3:
        raise ValueError("An exact target reference point must have three coordinates.")
    return (_exact_scalar(point[0]), _exact_scalar(point[1]), _exact_scalar(point[2]))


@lru_cache(maxsize=None)
def _exact_reference_cell_vertices(
    cell_kind: str, /
) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    element = lagrange_element(cell_kind, 1)
    if not isinstance(element, FiniteElementSpec):
        raise ValueError("Exact layer maps require a canonical degree-one reference element.")
    node_data = lagrange_reference_node_data(element)
    vertices: list[tuple[Fraction, Fraction, Fraction]] = []
    for dofs in element.entity_dofs[0]:
        if len(dofs) != 1:
            raise ValueError("A degree-one reference vertex must have one exact nodal DOF.")
        local_dof = int(dofs[0])
        node = node_data[local_dof]
        if node.local_dof != local_dof or node.exact_point is None:
            raise ValueError("A reference vertex has no canonical exact nodal coordinate.")
        vertices.append(_fraction3(node.exact_point))
    return tuple(vertices)


def _exact_scalar_identity(value: ExactScalar, /) -> object:
    return _coordinate_coefficient_identity(value)




def _fraction_chart2(
    points: Sequence[Sequence[object]], /
) -> tuple[
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
    tuple[Fraction, Fraction],
]:
    if len(points) != 3:
        raise ValueError("An exact source chart must retain three corners.")
    return (_fraction2(points[0]), _fraction2(points[1]), _fraction2(points[2]))


def _fraction_chart3(
    points: Sequence[Sequence[object]], /
) -> tuple[
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
    tuple[Fraction, Fraction, Fraction],
]:
    if len(points) != 3:
        raise ValueError("An exact target chart must retain three corners.")
    return (_fraction3(points[0]), _fraction3(points[1]), _fraction3(points[2]))


def _reference_expression(
    expression: ExactReferenceChartExpression, /
) -> Expression:
    return expression if isinstance(expression, RationalPolynomial) else dict(expression)


type _ImplicitCellKey = tuple[int, int, int, int]
type _ImplicitRootEdge = tuple[tuple[int, int, int], tuple[int, int, int]]
type _ImplicitRootEdgeRow = tuple[
    _ImplicitRootEdge, _ImplicitRootEdge, _ImplicitRootEdge
]


def _implicit_cell_key(values: Sequence[int], /) -> _ImplicitCellKey:
    if len(values) != 4:
        raise ValueError("Adaptive implicit source-cell keys must retain four coordinates.")
    return (int(values[0]), int(values[1]), int(values[2]), int(values[3]))


def _int3(values: Sequence[int], /) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("Adaptive implicit root vertices must have three coordinates.")
    return (int(values[0]), int(values[1]), int(values[2]))


def _implicit_root_edge(values: ArrayLike, /) -> _ImplicitRootEdge:
    edge = np.asarray(values, dtype=np.int64)
    if edge.shape != (2, 3):
        raise ValueError("Adaptive implicit root edges must retain two 3D endpoints.")
    return (_int3(edge[0]), _int3(edge[1]))


def _implicit_root_edge_row(
    values: ArrayLike | Sequence[Sequence[Sequence[int]]], /
) -> _ImplicitRootEdgeRow:
    edges = np.asarray(values, dtype=np.int64)
    if edges.shape != (3, 2, 3):
        raise ValueError("Adaptive implicit source triangles must retain three root edges.")
    return (
        _implicit_root_edge(edges[0]),
        _implicit_root_edge(edges[1]),
        _implicit_root_edge(edges[2]),
    )


def _implicit_root_witness_row(
    root_witness: SourceBoundaryChartWitness,
    target_witness: SourceBoundaryChartWitness,
    target_row: int,
    /,
) -> int:
    cell_key = tuple(int(value) for value in target_witness.implicit_cell_keys[target_row])
    root_edges = tuple(
        tuple(tuple(int(value) for value in endpoint) for endpoint in edge)
        for edge in target_witness.implicit_root_edge_keys[target_row]
    )
    matches = tuple(
        row
        for row in range(len(root_witness.source_chart_ids))
        if root_witness.source_chart_ids[row] == target_witness.source_chart_ids[target_row]
        and tuple(int(value) for value in root_witness.implicit_cell_keys[row]) == cell_key
        and tuple(
            tuple(tuple(int(value) for value in endpoint) for endpoint in edge)
            for edge in root_witness.implicit_root_edge_keys[row]
        )
        == root_edges
    )
    if len(matches) != 1:
        raise ValueError("Adaptive target chart has no unique retained source-wall root witness.")
    return matches[0]
def _mapped_reference_chart_rows(
    query: MappedReferenceBoundarySource,
    witness: SourceBoundaryChartWitness,
    reference_maps: Sequence[ReferenceCellMapWitnesses],
    /,
) -> tuple[tuple[int, int, int, int, NestedCellReferenceMap], ...]:
    query.require_current()
    query.validate_chart_witness(witness)
    by_pair: dict[tuple[int, int], NestedCellReferenceMap] = {}
    signatures: dict[tuple[int, int], tuple[object, ...] | None] = {}
    for reference_map in reference_maps:
        validate_reference_cell_map_witnesses(reference_map)
        for mapping in nested_cell_reference_maps(reference_map):
            arguments = mapping.arguments
            if arguments is None or len(arguments) != 3:
                raise ValueError("Mapped layer witness requires an exact target-to-root expression bank.")
            key = (mapping.target_cell_id, mapping.source_cell_id)
            identity = reference_chart_arguments_identity((arguments,))
            if key in signatures and signatures[key] != identity:
                raise ValueError("Mapped layer cell reference map repeats conflicting source charts.")
            by_pair[key] = mapping
            signatures[key] = identity
    result = []
    for row, chart_id in enumerate(witness.source_chart_ids):
        query_row = query.chart_row(2, chart_id)
        entity_key = query.chart_source_key(2, chart_id)
        if entity_key != (witness.source_entity_ids[row], witness.occurrence_paths[row]):
            raise ValueError("Mapped layer witness changed its source entity/occurrence.")
        root_face = int(query.polygon_root_face_global_ids[query_row])
        root_cell = int(query.polygon_root_cell_global_ids[query_row])
        root_local = int(query.polygon_root_local_face_indices[query_row])
        target_cell = int(witness.cell_global_ids[row])
        try:
            mapping = by_pair[(target_cell, root_cell)]
        except KeyError as error:
            raise ValueError("Mapped layer target chart has no exact root-cell reference map.") from error
        arguments = mapping.arguments
        if arguments is None or len(arguments) != 3:
            raise ValueError("Mapped layer witness requires its exact target-to-root expression bank.")
        target_points = _fraction_chart3(witness.target_reference_chart(row))
        exact_arguments = tuple(_reference_expression(value) for value in arguments)
        if _determinant_3x3(
            _exact_map_jacobian(exact_arguments, target_points[0])
        ) == 0:
            raise ValueError("Mapped layer target-to-root reference map is singular.")
        root_points = tuple(
            _fraction3(mapping.evaluate(point)) for point in target_points
        )
        source_parameters = _fraction_chart2(
            query.root_face_uv(
                root_face,
                root_cell,
                root_local,
                root_points,
            )
        )
        if source_parameters != _fraction_chart2(witness.source_parameter_chart(row)):
            raise ValueError("Mapped layer witness source parameters differ from its exact root map.")
        result.append((query_row, root_face, root_cell, root_local, mapping))
    return tuple(result)
def _exact_map_jacobian(
    arguments: Sequence[Expression],
    point: _ExactReferencePoint3,
    /,
) -> tuple[
    tuple[ExactScalar, ExactScalar, ExactScalar],
    tuple[ExactScalar, ExactScalar, ExactScalar],
    tuple[ExactScalar, ExactScalar, ExactScalar],
]:
    if len(arguments) != 3:
        raise ValueError("Mapped root cell chart must retain three exact coordinate expressions.")
    rows = tuple(
        _exact3(
            tuple(
                evaluate_expression(derivative(expression, axis), point)
                for axis in range(3)
            )
        )
        for expression in arguments
    )
    return (rows[0], rows[1], rows[2])


def _determinant_3x3(
    matrix: tuple[
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
    ],
    /,
) -> ExactScalar:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )

def _query_frame_id(query: SourceBoundaryQuery, /) -> str | None:
    frame = getattr(query, "frame_id", None)
    if frame:
        return str(frame)
    contract = getattr(query, "coordinate_contract", None)
    if contract is None:
        contract = getattr(getattr(query, "domain", None), "coordinate_contract", None)
    spatial_id = getattr(contract, "spatial_id", None)
    return None if not spatial_id else str(spatial_id)


def _prepare_barycentric_solver_2d(
    corners: Sequence[tuple[Fraction, Fraction]], /
) -> _BarycentricSolver2:
    if len(corners) != 3:
        raise ValueError("Source parameter charts require exactly three corners.")
    first = (corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
    second = (corners[2][0] - corners[0][0], corners[2][1] - corners[0][1])
    matrix = ((first[0], second[0]), (first[1], second[1]))
    try:
        return factor_exact_rational_algebraic(
            matrix,
            RadicalElement,
            singular_message="Source parameter chart is degenerate.",
        )
    except ValueError as error:
        raise ValueError("Source parameter chart is degenerate.") from error


def _prepare_barycentric_solver_3d(
    corners: Sequence[tuple[Fraction, Fraction, Fraction]], /
) -> _BarycentricSolver3:
    if len(corners) != 3:
        raise ValueError("Target reference charts require exactly three corners.")
    first = tuple(corners[1][axis] - corners[0][axis] for axis in range(3))
    second = tuple(corners[2][axis] - corners[0][axis] for axis in range(3))
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        matrix = (
            (first[axis_a], second[axis_a]),
            (first[axis_b], second[axis_b]),
        )
        try:
            return (
                (axis_a, axis_b),
                factor_exact_rational_algebraic(matrix, RadicalElement),
            )
        except ValueError:
            continue
    raise ValueError("Target reference facet chart is degenerate.")


def _barycentric_2d(
    point: _ExactReferencePoint2,
    corners: Sequence[tuple[Fraction, Fraction]],
    /,
    *,
    solver: _BarycentricSolver2 | None = None,
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    if len(corners) != 3:
        raise ValueError("Source parameter charts require exactly three corners.")
    if solver is None:
        solver = _prepare_barycentric_solver_2d(corners)
    right_hand_side = (
        (point[0] - corners[0][0],),
        (point[1] - corners[0][1],),
    )
    solved = solver.solve(right_hand_side)
    alpha, beta = solved[0][0], solved[1][0]
    return Fraction(1) - alpha - beta, alpha, beta


def _barycentric_3d(
    point: _ExactReferencePoint3,
    corners: Sequence[tuple[Fraction, Fraction, Fraction]],
    /,
    *,
    solver: _BarycentricSolver3 | None = None,
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    if len(corners) != 3:
        raise ValueError("Target reference charts require exactly three corners.")
    if solver is None:
        solver = _prepare_barycentric_solver_3d(corners)
    axes, factorization = solver
    right_hand_side = (
        (point[axes[0]] - corners[0][axes[0]],),
        (point[axes[1]] - corners[0][axes[1]],),
    )
    solved = factorization.solve(right_hand_side)
    alpha, beta = solved[0][0], solved[1][0]
    first = tuple(corners[1][axis] - corners[0][axis] for axis in range(3))
    second = tuple(corners[2][axis] - corners[0][axis] for axis in range(3))
    if any(
        corners[0][axis] + alpha * first[axis] + beta * second[axis]
        != point[axis]
        for axis in range(3)
    ):
        raise ValueError("Reference chart point is outside its exact target facet plane.")
    return Fraction(1) - alpha - beta, alpha, beta


def _mapped_chart_reference_contexts(
    query: MappedReferenceBoundarySource,
    witness: SourceBoundaryChartWitness,
    chart_row: int,
    mapping: NestedCellReferenceMap,
    target_references: Sequence[_ExactReferencePoint3],
    /,
) -> tuple[
    tuple[
        _ExactReferencePoint3,
        _ExactReferencePoint2,
        tuple[tuple[Fraction, Fraction], ...],
        _BarycentricSolver2,
    ],
    ...,
]:
    if not target_references:
        return ()
    query_row = query.chart_row(2, witness.source_chart_ids[chart_row])
    root_face = int(query.polygon_root_face_global_ids[query_row])
    root_cell = int(query.polygon_root_cell_global_ids[query_row])
    root_local = int(query.polygon_root_local_face_indices[query_row])
    root_references = tuple(
        _exact3(mapping.evaluate(point)) for point in target_references
    )
    source_uvs = query.root_face_uv(
        root_face, root_cell, root_local, root_references
    )
    if len(source_uvs) != len(root_references):
        raise ValueError("Mapped root-face chart returned misaligned exact UV rows.")
    source_chart = _fraction_chart2(
        query.chart_face_uv(witness.source_chart_ids[chart_row])
    )
    source_chart_solver = _prepare_barycentric_solver_2d(source_chart)
    return tuple(
        (
            _exact3(root_reference),
            _exact2(source_uv),
            source_chart,
            source_chart_solver,
        )
        for root_reference, source_uv in zip(
            root_references, source_uvs, strict=True
        )
    )


def _evaluate_mapped_chart_point(
    query: MappedReferenceBoundarySource,
    witness: SourceBoundaryChartWitness,
    chart_row: int,
    target_reference: _ExactReferencePoint3,
    mapping: NestedCellReferenceMap,
    root_reference: _ExactReferencePoint3,
    source_uv: _ExactReferencePoint2,
    source_chart_uv: tuple[tuple[Fraction, Fraction], ...],
    source_chart_solver: _BarycentricSolver2,
    /,
) -> tuple[
    _ExactReferencePoint3,
    _ExactReferencePoint2,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    arguments = mapping.arguments
    if arguments is None or len(arguments) != 3:
        raise ValueError("Mapped layer source chart has no retained exact root reference expressions.")
    exact_arguments = tuple(_reference_expression(value) for value in arguments)
    if (
        _exact3(mapping.evaluate(target_reference)) != root_reference
        or _determinant_3x3(
            _exact_map_jacobian(exact_arguments, target_reference)
        )
        == 0
    ):
        raise ValueError("Mapped layer target-to-root reference map is singular or stale.")
    chart_weights = _barycentric_2d(
        source_uv, source_chart_uv, solver=source_chart_solver
    )
    if any(value < 0 for value in chart_weights):
        raise ValueError("Mapped target reference point leaves its exact source chart.")
    chart_parameter = _exact2((chart_weights[1], chart_weights[2]))
    expressions = tuple(
        _reference_expression(expression)
        for expression in query.chart_coordinate_expressions(
            2, witness.source_chart_ids[chart_row]
        )
    )
    source_exact = tuple(
        evaluate_expression(expression, chart_parameter)
        for expression in expressions
    )
    tangents = np.asarray(
        [
            [
                float(evaluate_expression(derivative(expression, axis), chart_parameter))
                for expression in expressions
            ]
            for axis in range(2)
        ],
        dtype=np.float64,
    )
    normal_raw = np.cross(tangents[0], tangents[1])
    normal_norm = float(np.linalg.norm(normal_raw))
    if (
        len(source_exact) != 3
        or tangents.shape != (2, 3)
        or not np.all(np.isfinite(tangents))
        or not np.isfinite(normal_norm)
        or normal_norm <= 0.0
    ):
        raise ValueError("Mapped root source-chart derivative is physically singular.")
    return (
        root_reference,
        chart_parameter,
        np.asarray([float(value) for value in source_exact], dtype=np.float64),
        tangents,
        normal_raw / normal_norm,
    )


def _mapped_physical_distance_bounds(
    query: MappedReferenceBoundarySource,
    chart_id: str,
    source_parameter: _ExactReferencePoint2,
    target_point: ArrayLike,
    /,
) -> tuple[float, float]:
    expressions = tuple(
        _reference_expression(expression)
        for expression in query.chart_coordinate_expressions(2, chart_id)
    )
    source_exact = tuple(
        evaluate_expression(expression, source_parameter)
        for expression in expressions
    )
    target_exact = tuple(
        Fraction(float(value))
        for value in np.asarray(target_point, dtype=np.float64)
    )
    if len(source_exact) != 3:
        raise ValueError("Mapped source physical distance requires three root-chart coordinates.")
    squared_lower = Fraction(0)
    squared_upper = Fraction(0)
    for axis, value in enumerate(source_exact):
        lower, upper = coefficient_interval(value)
        delta_lower = lower - target_exact[axis]
        delta_upper = upper - target_exact[axis]
        minimum = (
            Fraction(0)
            if delta_lower <= 0 <= delta_upper
            else min(abs(delta_lower), abs(delta_upper))
        )
        maximum = max(abs(delta_lower), abs(delta_upper))
        squared_lower += minimum * minimum
        squared_upper += maximum * maximum
    lower, _ = fraction_sqrt_interval(squared_lower)
    _, upper = fraction_sqrt_interval(squared_upper)
    return (
        fraction_endpoint(lower, upper=False, saturate=True),
        fraction_endpoint(upper, upper=True, saturate=True),
    )


def _face_reference_element(
    element: object, cell_kind: str, /
) -> _NodalReferenceElement:
    if isinstance(element, _NodalReferenceElement):
        return element
    degree = getattr(element, "degree", None)
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise ValueError("Layer source trace has no supported nodal reference order.")
    return lagrange_element(cell_kind, int(degree))



def _reference_face_dofs(
    element: _NodalReferenceElement,
    target_reference_corners: Sequence[tuple[Fraction, Fraction, Fraction]],
    /,
    *,
    solver: _BarycentricSolver3 | None = None,
) -> tuple[tuple[int, _ExactReferencePoint3], ...]:
    corners = tuple(_fraction3(point) for point in target_reference_corners)
    if len(corners) != 3:
        raise ValueError("Layer source charts require exact triangular target facets.")
    barycentric_solver = solver or _prepare_barycentric_solver_3d(corners)
    if not isinstance(element, FiniteElementSpec):
        raise ValueError("Layer source charts require a canonical exact nodal element.")
    node_data = lagrange_reference_node_data(element)
    result: list[tuple[int, _ExactReferencePoint3]] = []
    for local_dof, node in enumerate(node_data):
        if node.local_dof != local_dof:
            raise ValueError("Canonical nodal reference data changed local DOF order.")
        if node.exact_point is None:
            raise ValueError(
                "Layer source face has a nodal reference point without exact authority."
            )
        reference = _exact3(node.exact_point)
        try:
            weights = _barycentric_3d(
                reference, corners, solver=barycentric_solver
            )
        except ValueError:
            continue
        if all(value >= 0 for value in weights):
            result.append((local_dof, reference))
    return tuple(result)
def _evaluate_cell_geometry_point(
    elements: Sequence[CellGeometryElement],
    routes: Sequence[ArrayLike],
    coordinates: ArrayLike,
    block_row: int,
    cell_row: int,
    reference: _ExactReferencePoint3,
    /,
) -> NDArray[np.float64]:
    route = np.asarray(routes[block_row], dtype=np.int64)
    coordinate_bank = np.asarray(coordinates, dtype=np.float64)
    local = tuple(coordinate_bank[int(index)] for index in route[cell_row])
    element = elements[block_row]
    if not isinstance(element, ScalarCoordinateElement):
        raise TypeError("Target geometry element has no exact scalar coordinate map.")
    expressions = coordinate_map_expressions(element, local, cell_row=cell_row)
    point = np.asarray(
        [float(evaluate_expression(value, reference)) for value in expressions],
        dtype=np.float64,
    )
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise ValueError("Exact target coordinate map produced an invalid source-bound point.")
    return point


def _validate_parametric_chart_geometry(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    charts: SurfaceSourceCharts,
    witness: SourceBoundaryChartWitness,
    column_identity: LayerColumnProjectionIdentity,
    tolerance: float,
    /,
) -> None:
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Source-chart physical tolerance must be finite and nonnegative.")
    charts.require_current()
    charts.require_root(charts.mesh, charts.geometry)
    charts.require_bound(charts.mesh, charts.geometry, charts.boundary_source)
    if (
        charts.source_id != witness.source_id
        or charts.source_revision != witness.source_revision
        or charts.boundary_source.source_scope_id != witness.source_query_id
        or charts.coordinate_contract.spatial_id == ""
        or witness.chart_dimension != 2
        or witness.reference_dimension != 3
        or column_identity.row_count != len(witness.source_chart_ids)
    ):
        raise ValueError("Parametric source chart, frame, or column identity is stale.")
    elements, routes, coordinates = geometry.resolve(mesh)
    cell_rows = {
        int(cell_id): (block_row, cell_row)
        for block_row, block in enumerate(mesh.blocks)
        for cell_row, cell_id in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    source_chart_rows = {
        int(cell_id): row for row, cell_id in enumerate(charts.cell_global_ids)
    }
    expected_dofs: dict[tuple[int, int], set[int]] = {}
    covered_dofs: dict[tuple[int, int], set[int]] = {}
    patches: list[int] = []
    parameters: list[tuple[float, float]] = []
    exact_parameters: list[_ExactReferencePoint2] = []
    target_points: list[NDArray[np.float64]] = []
    for chart_row, source_cell_id in enumerate(
        column_identity.source_wall_cell_ids
    ):
        source_chart_row = source_chart_rows.get(int(source_cell_id))
        if source_chart_row is None:
            raise ValueError("Layer column ancestry names an absent source-wall chart cell.")
        patch = int(charts.root_patches[source_chart_row])
        source_entity = charts.domain.entity_id(2, patch)
        occurrence = tuple(charts.domain.source_occurrences[2][patch])
        source_parameters = _fraction_chart2(
            witness.source_parameter_chart(chart_row)
        )
        root_chart = _fraction_chart2(
            charts.root_parameter_chart(source_chart_row)
        )
        root_chart_solver = _prepare_barycentric_solver_2d(root_chart)
        if (
            witness.source_entity_ids[chart_row] != source_entity
            or witness.source_chart_ids[chart_row] != source_entity
            or witness.occurrence_paths[chart_row] != occurrence
            or any(
                any(
                    value < 0
                    for value in _barycentric_2d(
                        point, root_chart, solver=root_chart_solver
                    )
                )
                for point in source_parameters
            )
        ):
            raise ValueError("Layer source chart changed its exact entity, occurrence, or root patch.")
        cell_id = int(witness.cell_global_ids[chart_row])
        try:
            block_row, cell_row = cell_rows[cell_id]
        except KeyError as error:
            raise ValueError("Layer source chart names an absent target cell.") from error
        block = mesh.blocks[block_row]
        local_face = int(witness.local_entity_indices[chart_row])
        element = _face_reference_element(elements[block_row], block.cell_kind)
        if local_face < 0 or local_face >= len(element.entity_dofs[2]):
            raise ValueError("Layer source chart names an absent target face.")
        face_key = (cell_id, local_face)
        target_corners = _fraction_chart3(
            witness.target_reference_chart(chart_row)
        )
        target_chart_solver = _prepare_barycentric_solver_3d(target_corners)
        face_nodes = _reference_face_dofs(
            element, target_corners, solver=target_chart_solver
        )
        face_dofs = {local_dof for local_dof, _ in face_nodes}
        expected_dofs.setdefault(face_key, set()).update(face_dofs)
        for local_dof, reference in face_nodes:
            weights = _barycentric_3d(
                reference, target_corners, solver=target_chart_solver
            )
            source_uv = _exact2(
                tuple(
                    sum(
                        (weights[corner] * source_parameters[corner][axis] for corner in range(3)),
                        Fraction(0),
                    )
                    for axis in range(2)
                )
            )
            if any(
                value < 0
                for value in _barycentric_2d(
                    source_uv, root_chart, solver=root_chart_solver
                )
            ):
                raise ValueError("A target boundary DOF leaves its exact source root chart.")
            route = np.asarray(routes[block_row], dtype=np.int64)
            geometry_row = int(route[cell_row, local_dof])
            if geometry_row >= 0:
                target_point = np.asarray(coordinates[geometry_row], dtype=np.float64)
            elif geometry_row == -1:
                target_point = _evaluate_cell_geometry_point(
                    elements,
                    routes,
                    coordinates,
                    block_row,
                    cell_row,
                    reference,
                )
            else:
                raise ValueError("A source-chart boundary DOF has an invalid map-node sentinel.")
            patches.append(patch)
            parameters.append((float(source_uv[0]), float(source_uv[1])))
            exact_parameters.append(source_uv)
            target_points.append(target_point)
            covered_dofs.setdefault(face_key, set()).add(local_dof)
    if not patches or any(
        covered_dofs.get(key, set()) != value for key, value in expected_dofs.items()
    ):
        raise ValueError("Exact source charts do not cover every target boundary geometry DOF.")
    targets = np.asarray(target_points, dtype=np.float64)
    source_points = np.empty_like(targets)
    exact_surface_maps: dict[int, tuple[Expression, ...]] = {}
    for row, (patch, parameter) in enumerate(
        zip(patches, exact_parameters, strict=True)
    ):
        expressions = exact_surface_maps.get(patch)
        if expressions is None:
            exact = exact_surface_coordinate_expressions(
                charts.domain.patches[patch].surface
            )
            if exact is None:
                raise ValueError(
                    "Parametric cold validation requires an exact source-surface map."
                )
            expressions = tuple(exact)
            exact_surface_maps[patch] = expressions
        if len(expressions) != targets.shape[1]:
            raise ValueError("Exact source-surface map has the wrong ambient dimension.")
        source_points[row] = tuple(
            float(evaluate_expression(expression, parameter))
            for expression in expressions
        )
    patch_rows = np.asarray(patches, dtype=np.int64)
    uv = np.asarray(parameters, dtype=np.float64)
    normals, regular = charts.domain.oriented_normals(patch_rows, uv)
    normals = np.asarray(normals, dtype=np.float64)
    regular = np.asarray(regular, dtype=np.bool_)
    if (
        source_points.shape != targets.shape
        or normals.shape != targets.shape
        or regular.shape != (len(patches),)
        or not np.all(np.isfinite(source_points))
        or not np.all(np.isfinite(normals))
        or np.any(~regular)
    ):
        raise ValueError("Exact source-chart evaluation or physical frame is singular.")
    residuals = np.linalg.norm(targets - source_points, axis=1)
    if np.any(~np.isfinite(residuals)) or np.any(residuals > tolerance):
        raise ValueError("Target coordinate map is outside its exact source-chart tolerance.")
    for patch in sorted(set(patches)):
        selected = np.flatnonzero(patch_rows == patch)
        differential = np.asarray(
            surface_differential(
                charts.domain.patches[patch].surface,
                jnp.asarray(uv[selected], dtype=jnp.float64),
            ),
            dtype=np.float64,
        )
        if differential.shape != (selected.size, 3, 2) or not np.all(
            np.isfinite(differential)
        ):
            raise ValueError("Exact source-chart derivative returned an invalid physical frame.")
        cross = np.cross(differential[:, :, 0], differential[:, :, 1])
        lengths = np.linalg.norm(cross, axis=1)
        if np.any(~np.isfinite(lengths) | (lengths <= 0.0)):
            raise ValueError("Exact source-chart derivative is physically singular.")
        oriented = int(charts.domain.patches[patch].orientation) * cross / lengths[:, None]
        if np.any(np.sum(oriented * normals[selected], axis=1) <= 0.0):
            raise ValueError("Exact source-chart derivative changed its physical frame orientation.")


class LayerColumnProjectionIdentity(StrictModule, NonTrainableState):
    """Stable layer schedule, front-row, column, and source-root ancestry per node."""

    layer_indices: NDArray[np.int64]
    source_wall_cell_ids: NDArray[np.int64]
    front_row_ids: NDArray[np.int64]
    front_row_identities: tuple[str, ...] = eqx.field(static=True)
    source_column_ids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    source_column_identities: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_column_root_vertex_ids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)

    def __init__(
        self,
        layer_indices: ArrayLike,
        source_wall_cell_ids: ArrayLike,
        front_row_ids: ArrayLike,
        front_row_identities: Sequence[str],
        source_column_ids: Sequence[Sequence[int]],
        source_column_identities: Sequence[Sequence[str]],
        source_column_root_vertex_ids: Sequence[Sequence[int]],
        /,
    ) -> None:
        layers = np.asarray(layer_indices, dtype=np.int64)
        wall_cells = np.asarray(source_wall_cell_ids, dtype=np.int64)
        front_rows = np.asarray(front_row_ids, dtype=np.int64)
        identities = tuple(str(value) for value in front_row_identities)
        columns = tuple(tuple(int(value) for value in row) for row in source_column_ids)
        column_identities = tuple(
            tuple(str(value) for value in row) for row in source_column_identities
        )
        roots = tuple(tuple(int(value) for value in row) for row in source_column_root_vertex_ids)
        count = layers.size
        if (
            layers.ndim != 1
            or wall_cells.shape != (count,)
            or front_rows.shape != (count,)
            or len(identities) != count
            or len(columns) != count
            or len(column_identities) != count
            or len(roots) != count
            or np.any(layers < 0)
            or np.any(wall_cells < 0)
            or np.any(front_rows < 0)
            or any(not value for value in identities)
            or any(
                not column_row
                or len(column_row) != len(name_row)
                or len(column_row) != len(root_row)
                or any(value < 0 for value in column_row)
                or any(not value for value in name_row)
                or any(value < 0 for value in root_row)
                for column_row, name_row, root_row in zip(
                    columns, column_identities, roots, strict=True
                )
            )
        ):
            raise ValueError("Layer column/root ancestry rows are incomplete or misaligned.")
        self.layer_indices = layers
        self.source_wall_cell_ids = wall_cells
        self.front_row_ids = front_rows
        self.front_row_identities = identities
        self.source_column_ids = columns
        self.source_column_identities = column_identities
        self.source_column_root_vertex_ids = roots
        self.identity_id = canonical_fingerprint(
            {
                "kind": "layer-column-projection-identity",
                "layer_indices": array_tree_fingerprint(layers),
                "source_wall_cell_ids": array_tree_fingerprint(wall_cells),
                "front_row_ids": array_tree_fingerprint(front_rows),
                "front_row_identities": identities,
                "source_column_ids": columns,
                "source_column_identities": column_identities,
                "source_column_root_vertices": roots,
            }
        )

    @property
    def row_count(self) -> int:
        return int(self.layer_indices.size)


def _column_identity_row(
    identity: LayerColumnProjectionIdentity,
    row: int,
    /,
) -> tuple[object, ...]:
    return (
        int(identity.layer_indices[row]),
        int(identity.source_wall_cell_ids[row]),
        int(identity.front_row_ids[row]),
        identity.front_row_identities[row],
        identity.source_column_ids[row],
        identity.source_column_identities[row],
        identity.source_column_root_vertex_ids[row],
    )
class LayerCurvingProjectionTrace(StrictModule, NonTrainableState):
    """Row-aligned source projection results and failures at exact target chart nodes.

    ``geometry_node_rows`` is ``-1`` when a per-cell restricted coordinate map
    has no globally stored target-node row; ``target_reference_points`` then
    binds the exact full-map evaluation used by the projection.
    """

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    geometry_node_rows: NDArray[np.int64]
    target_cell_global_ids: NDArray[np.int64]
    target_local_dofs: NDArray[np.int32]
    chart_witness_rows: NDArray[np.int64]
    target_reference_points: tuple[_ExactReferencePoint3, ...] = eqx.field(static=True)
    source_root_cell_ids: NDArray[np.int64]
    source_piece_ids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    column_identity: LayerColumnProjectionIdentity
    requested_entity_ids: tuple[str, ...] = eqx.field(static=True)
    returned_entity_ids: tuple[str, ...] = eqx.field(static=True)
    requested_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    returned_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    queries: NDArray[np.float64]
    parameters: NDArray[np.float64]
    projected_points: NDArray[np.float64]
    distance_lower_bounds: NDArray[np.float64]
    residuals: NDArray[np.float64]
    status: NDArray[np.int8]
    normals: NDArray[np.float64]
    tangents: NDArray[np.float64]
    returned_source_ids: tuple[str, ...] = eqx.field(static=True)
    returned_source_revisions: tuple[str, ...] = eqx.field(static=True)
    returned_frame_ids: tuple[str, ...] = eqx.field(static=True)
    failures: tuple[str, ...] = eqx.field(static=True)
    trace_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        frame_id: str,
        geometry_node_rows: ArrayLike,
        target_cell_global_ids: ArrayLike,
        target_local_dofs: ArrayLike,
        source_root_cell_ids: ArrayLike,
        source_piece_ids: Sequence[Sequence[int]],
        column_identity: LayerColumnProjectionIdentity,
        requested_entity_ids: Sequence[str],
        returned_entity_ids: Sequence[str],
        requested_occurrence_paths: Sequence[tuple[str, ...]],
        returned_occurrence_paths: Sequence[tuple[str, ...]],
        queries: ArrayLike,
        parameters: ArrayLike,
        projected_points: ArrayLike,
        distance_lower_bounds: ArrayLike,
        residuals: ArrayLike,
        status: ArrayLike,
        returned_source_ids: Sequence[str],
        returned_source_revisions: Sequence[str],
        returned_frame_ids: Sequence[str],
        failures: Sequence[str],
        /,
        *,
        normals: ArrayLike,
        tangents: ArrayLike,
        chart_witness_rows: ArrayLike,
        target_reference_points: Sequence[Sequence[ExactScalar]],
    ) -> None:
        rows = np.asarray(geometry_node_rows, dtype=np.int64)
        cells = np.asarray(target_cell_global_ids, dtype=np.int64)
        dofs = np.asarray(target_local_dofs, dtype=np.int32)
        chart_rows = np.asarray(chart_witness_rows, dtype=np.int64)
        reference = tuple(_exact3(point) for point in target_reference_points)
        roots = np.asarray(source_root_cell_ids, dtype=np.int64)
        query = np.asarray(queries, dtype=np.float64)
        params = np.asarray(parameters, dtype=np.float64)
        points = np.asarray(projected_points, dtype=np.float64)
        lower = np.asarray(distance_lower_bounds, dtype=np.float64)
        upper = np.asarray(residuals, dtype=np.float64)
        states = np.asarray(status, dtype=np.int8)
        normal = np.asarray(normals, dtype=np.float64)
        tangent = np.asarray(tangents, dtype=np.float64)
        count = rows.size
        pieces = tuple(tuple(int(value) for value in row) for row in source_piece_ids)
        requested_entities = tuple(str(value) for value in requested_entity_ids)
        returned_entities = tuple(str(value) for value in returned_entity_ids)
        requested_paths = tuple(tuple(path) for path in requested_occurrence_paths)
        returned_paths = tuple(tuple(path) for path in returned_occurrence_paths)
        source_ids = tuple(str(value) for value in returned_source_ids)
        revisions = tuple(str(value) for value in returned_source_revisions)
        frames = tuple(str(value) for value in returned_frame_ids)
        failures_ = tuple(str(value) for value in failures)
        if (
            any(not value for value in (source_id, source_revision, frame_id))
            or not isinstance(column_identity, LayerColumnProjectionIdentity)
            or column_identity.row_count != count
            or any(value.shape != (count,) for value in (cells, dofs, roots, lower, upper, states))
            or chart_rows.shape != (count,)
            or np.any(rows < -1)
            or np.any(chart_rows < 0)
            or len(reference) != count
            or any(len(point) != 3 for point in reference)
            or query.ndim != 2
            or query.shape[0] != count
            or params.ndim != 2
            or params.shape[0] != count
            or points.shape != query.shape
            or len(pieces) != count
            or any(len(value) != count for value in (requested_entities, returned_entities, requested_paths, returned_paths, source_ids, revisions, frames))
            or normal.shape != (count, 3)
            or tangent.shape != (count, 2, 3)
        ):
            raise ValueError("Layer source projection trace rows do not align.")
        self.source_id = str(source_id)
        self.source_revision = str(source_revision)
        self.frame_id = str(frame_id)
        self.geometry_node_rows = rows
        self.target_cell_global_ids = cells
        self.target_local_dofs = dofs
        self.chart_witness_rows = chart_rows
        self.target_reference_points = reference
        self.source_root_cell_ids = roots
        self.source_piece_ids = pieces
        self.column_identity = column_identity
        self.requested_entity_ids = requested_entities
        self.returned_entity_ids = returned_entities
        self.requested_occurrence_paths = requested_paths
        self.returned_occurrence_paths = returned_paths
        self.queries = query
        self.parameters = params
        self.projected_points = points
        self.distance_lower_bounds = lower
        self.residuals = upper
        self.status = states
        self.normals = normal
        self.tangents = tangent
        self.returned_source_ids = source_ids
        self.returned_source_revisions = revisions
        self.returned_frame_ids = frames
        self.failures = failures_
        self.trace_id = canonical_fingerprint(
            {
                "kind": "layer-curving-projection-trace",
                "source": (self.source_id, self.source_revision, self.frame_id),
                "geometry_nodes": array_tree_fingerprint(rows),
                "target_cells": array_tree_fingerprint(cells),
                "target_dofs": array_tree_fingerprint(dofs),
                "chart_witness_rows": array_tree_fingerprint(chart_rows),
                "target_reference_points": tuple(
                    tuple(_exact_scalar_identity(value) for value in point)
                    for point in reference
                ),
                "source_roots": array_tree_fingerprint(roots),
                "source_pieces": pieces,
                "column_identity": column_identity.identity_id,
                "requested_entities": requested_entities,
                "returned_entities": returned_entities,
                "requested_occurrences": requested_paths,
                "returned_occurrences": returned_paths,
                "queries": array_tree_fingerprint(query),
                "parameters": array_tree_fingerprint(params),
                "points": array_tree_fingerprint(points),
                "lower": array_tree_fingerprint(lower),
                "upper": array_tree_fingerprint(upper),
                "normals": array_tree_fingerprint(normal),
                "tangents": array_tree_fingerprint(tangent),
                "status": array_tree_fingerprint(states),
                "returned_sources": (source_ids, revisions, frames),
                "failures": failures_,
            }
        )

    @property
    def resolved(self) -> bool:
        return bool(
            not self.failures
            and np.all(np.isin(self.status, (int(SourceProjectionStatus.UNIQUE), int(SourceProjectionStatus.SEAM))))
            and self.returned_entity_ids == self.requested_entity_ids
            and self.returned_occurrence_paths == self.requested_occurrence_paths
            and all(value == self.source_id for value in self.returned_source_ids)
            and all(value == self.source_revision for value in self.returned_source_revisions)
            and all(value == self.frame_id for value in self.returned_frame_ids)
            and np.all(np.isfinite(self.parameters))
            and np.all(np.isfinite(self.projected_points))
            and np.all(np.isfinite(self.normals))
            and np.all(np.isfinite(self.tangents))
            and np.all(np.isfinite(self.distance_lower_bounds))
            and np.all(np.isfinite(self.residuals))
            and np.all((self.distance_lower_bounds >= 0.0) & (self.distance_lower_bounds <= self.residuals))
            and np.all(np.linalg.norm(np.cross(self.tangents[:, 0, :], self.tangents[:, 1, :]), axis=1) > 0.0)
        )


class LayerCurvingImplicitRootTrace(StrictModule, NonTrainableState):
    """Per-node certified local adaptive-implicit root lifts; no global projection claim."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    map_witness_id: str = eqx.field(static=True)
    root_chart_witness_id: str = eqx.field(static=True)
    geometry_node_rows: NDArray[np.int64]
    target_cell_global_ids: NDArray[np.int64]
    target_local_dofs: NDArray[np.int32]
    chart_rows: NDArray[np.int64]
    target_reference_points: tuple[_ExactReferencePoint3, ...] = eqx.field(static=True)
    target_points: NDArray[np.float64]
    source_points: NDArray[np.float64]
    displacement_upper: NDArray[np.float64]
    root_error: NDArray[np.float64]
    source_cell_keys: tuple[_ImplicitCellKey, ...] = eqx.field(static=True)
    root_edge_keys: tuple[_ImplicitRootEdgeRow, ...] = eqx.field(static=True)
    statuses: tuple[str, ...] = eqx.field(static=True)
    chart_evidence: tuple[ImplicitRootChartEvidence, ...] = eqx.field(static=True)
    column_identity: LayerColumnProjectionIdentity
    trace_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        frame_id: str,
        map_witness_id: str,
        root_chart_witness_id: str,
        geometry_node_rows: ArrayLike,
        target_cell_global_ids: ArrayLike,
        target_local_dofs: ArrayLike,
        chart_rows: ArrayLike,
        target_reference_points: Sequence[Sequence[ExactScalar]],
        target_points: ArrayLike,
        source_points: ArrayLike,
        displacement_upper: ArrayLike,
        root_error: ArrayLike,
        source_cell_keys: Sequence[Sequence[int]],
        root_edge_keys: Sequence[Sequence[Sequence[Sequence[int]]]],
        statuses: Sequence[str],
        chart_evidence: Sequence[ImplicitRootChartEvidence],
        column_identity: LayerColumnProjectionIdentity,
        /,
    ) -> None:
        nodes = np.asarray(geometry_node_rows, dtype=np.int64)
        cells = np.asarray(target_cell_global_ids, dtype=np.int64)
        dofs = np.asarray(target_local_dofs, dtype=np.int32)
        rows = np.asarray(chart_rows, dtype=np.int64)
        target = np.asarray(target_points, dtype=np.float64)
        source = np.asarray(source_points, dtype=np.float64)
        displacement = np.asarray(displacement_upper, dtype=np.float64)
        error = np.asarray(root_error, dtype=np.float64)
        reference = tuple(_exact3(row) for row in target_reference_points)
        keys = tuple(_implicit_cell_key(row) for row in source_cell_keys)
        edges = tuple(_implicit_root_edge_row(row) for row in root_edge_keys)
        states = tuple(str(value) for value in statuses)
        evidence = tuple(chart_evidence)
        count = nodes.size
        if (
            not all((source_id, source_revision, frame_id, map_witness_id, root_chart_witness_id))
            or nodes.ndim != 1
            or np.any(nodes < -1)
            or cells.shape != (count,)
            or dofs.shape != (count,)
            or rows.shape != (count,)
            or len(reference) != count
            or target.shape != (count, 3)
            or source.shape != (count, 3)
            or displacement.shape != (count,)
            or error.shape != (count,)
            or len(keys) != count
            or len(edges) != count
            or len(states) != count
            or np.any(~np.isfinite(target))
            or np.any(~np.isfinite(source))
            or np.any(~np.isfinite(displacement) | (displacement < 0.0))
            or np.any(~np.isfinite(error) | (error < 0.0))
            or any(value != "certified_root_chart" for value in states)
            or any(len(key) != 4 for key in keys)
            or any(len(row) != 3 for row in edges)
            or any(not isinstance(value, ImplicitRootChartEvidence) for value in evidence)
            or not isinstance(column_identity, LayerColumnProjectionIdentity)
            or column_identity.row_count != count
        ):
            raise ValueError("Implicit layer root-lift rows are unresolved or misaligned.")
        if np.any(rows < 0) or (rows.size and np.max(rows) >= len(evidence)):
            raise ValueError("Implicit layer root-lift chart row is absent from its evidence bank.")
        self.source_id = str(source_id)
        self.source_revision = str(source_revision)
        self.frame_id = str(frame_id)
        self.map_witness_id = str(map_witness_id)
        self.root_chart_witness_id = str(root_chart_witness_id)
        self.geometry_node_rows = nodes
        self.target_cell_global_ids = cells
        self.target_local_dofs = dofs
        self.chart_rows = rows
        self.target_reference_points = reference
        self.target_points = target
        self.source_points = source
        self.displacement_upper = displacement
        self.root_error = error
        self.source_cell_keys = keys
        self.root_edge_keys = edges
        self.statuses = states
        self.chart_evidence = evidence
        self.column_identity = column_identity
        self.trace_id = canonical_fingerprint(
            {
                "kind": "layer-curving-adaptive-implicit-root-trace",
                "source": (self.source_id, self.source_revision, self.frame_id),
                "map": self.map_witness_id,
                "root_chart_witness": self.root_chart_witness_id,
                "geometry_nodes": array_tree_fingerprint(nodes),
                "target_cells": array_tree_fingerprint(cells),
                "target_dofs": array_tree_fingerprint(dofs),
                "chart_rows": array_tree_fingerprint(rows),
                "reference_points": tuple(
                    tuple(_exact_scalar_identity(value) for value in row)
                    for row in reference
                ),
                "target_points": array_tree_fingerprint(target),
                "source_points": array_tree_fingerprint(source),
                "displacement": array_tree_fingerprint(displacement),
                "root_error": array_tree_fingerprint(error),
                "source_cell_keys": keys,
                "root_edges": edges,
                "statuses": states,
                "chart_evidence": tuple(
                    (
                        row.source_cell_key,
                        row.root_edge_keys,
                        row.field_residual_bound,
                        row.target_to_root_bound,
                        row.source_normal_deviation_bounds,
                        row.chart_jacobian_lower,
                        row.chart_injectivity_margin,
                        row.source_gradient_lower,
                        row.status,
                    )
                    for row in evidence
                ),
                "column_identity": column_identity.identity_id,
            }
        )

    @property
    def resolved(self) -> bool:
        return all(value == "certified_root_chart" for value in self.statuses) and all(
            row.status == "certified" for row in self.chart_evidence
        )



class LayerCurvingMappedChartTrace(StrictModule, NonTrainableState):
    """Exact mapped-root chart evaluations with physical distance/frame evidence."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    query_id: str = eqx.field(static=True)
    geometry_node_rows: NDArray[np.int64]
    target_cell_global_ids: NDArray[np.int64]
    target_local_dofs: NDArray[np.int32]
    chart_witness_rows: NDArray[np.int64]
    target_reference_points: tuple[_ExactReferencePoint3, ...] = eqx.field(static=True)
    root_reference_points: tuple[_ExactReferencePoint3, ...] = eqx.field(static=True)
    source_parameters: tuple[_ExactReferencePoint2, ...] = eqx.field(static=True)
    source_chart_ids: tuple[str, ...] = eqx.field(static=True)
    source_entity_ids: tuple[str, ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    root_face_global_ids: NDArray[np.int64]
    root_cell_global_ids: NDArray[np.int64]
    root_local_face_indices: NDArray[np.int32]
    target_points: NDArray[np.float64]
    source_points: NDArray[np.float64]
    distance_lower_bounds: NDArray[np.float64]
    distance_upper_bounds: NDArray[np.float64]
    normals: NDArray[np.float64]
    tangents: NDArray[np.float64]
    column_identity: LayerColumnProjectionIdentity
    trace_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        frame_id: str,
        query_id: str,
        geometry_node_rows: ArrayLike,
        target_cell_global_ids: ArrayLike,
        target_local_dofs: ArrayLike,
        chart_witness_rows: ArrayLike,
        target_reference_points: Sequence[Sequence[ExactScalar]],
        root_reference_points: Sequence[Sequence[ExactScalar]],
        source_parameters: Sequence[Sequence[ExactScalar]],
        source_chart_ids: Sequence[str],
        source_entity_ids: Sequence[str],
        occurrence_paths: Sequence[tuple[str, ...]],
        root_face_global_ids: ArrayLike,
        root_cell_global_ids: ArrayLike,
        root_local_face_indices: ArrayLike,
        target_points: ArrayLike,
        source_points: ArrayLike,
        distance_lower_bounds: ArrayLike,
        distance_upper_bounds: ArrayLike,
        normals: ArrayLike,
        tangents: ArrayLike,
        column_identity: LayerColumnProjectionIdentity,
        /,
    ) -> None:
        nodes = np.asarray(geometry_node_rows, dtype=np.int64)
        cells = np.asarray(target_cell_global_ids, dtype=np.int64)
        dofs = np.asarray(target_local_dofs, dtype=np.int32)
        chart_rows = np.asarray(chart_witness_rows, dtype=np.int64)
        root_faces = np.asarray(root_face_global_ids, dtype=np.int64)
        root_cells = np.asarray(root_cell_global_ids, dtype=np.int64)
        root_local = np.asarray(root_local_face_indices, dtype=np.int32)
        target = np.asarray(target_points, dtype=np.float64)
        source = np.asarray(source_points, dtype=np.float64)
        lower = np.asarray(distance_lower_bounds, dtype=np.float64)
        upper = np.asarray(distance_upper_bounds, dtype=np.float64)
        normal = np.asarray(normals, dtype=np.float64)
        tangent = np.asarray(tangents, dtype=np.float64)
        target_reference = tuple(_exact3(row) for row in target_reference_points)
        root_reference = tuple(_exact3(row) for row in root_reference_points)
        parameters = tuple(_exact2(row) for row in source_parameters)
        charts = tuple(str(value) for value in source_chart_ids)
        entities = tuple(str(value) for value in source_entity_ids)
        paths = tuple(tuple(path) for path in occurrence_paths)
        count = nodes.size
        if (
            not all((source_id, source_revision, frame_id, query_id))
            or count == 0
            or cells.shape != (count,)
            or dofs.shape != (count,)
            or chart_rows.shape != (count,)
            or root_faces.shape != (count,)
            or root_cells.shape != (count,)
            or root_local.shape != (count,)
            or target.shape != (count, 3)
            or source.shape != (count, 3)
            or lower.shape != (count,)
            or upper.shape != (count,)
            or normal.shape != (count, 3)
            or tangent.shape != (count, 2, 3)
            or len(target_reference) != count
            or len(root_reference) != count
            or len(parameters) != count
            or any(len(value) != count for value in (charts, entities, paths))
            or np.any(nodes < -1)
            or np.any(chart_rows < 0)
            or np.any(root_faces < 0)
            or np.any(root_cells < 0)
            or np.any(root_local < 0)
            or np.any(~np.isfinite(target))
            or np.any(~np.isfinite(source))
            or np.any(~np.isfinite(lower) | (lower < 0.0))
            or np.any(~np.isfinite(upper) | (upper < lower))
            or np.any(~np.isfinite(normal))
            or np.any(~np.isfinite(tangent))
            or not isinstance(column_identity, LayerColumnProjectionIdentity)
            or column_identity.row_count != count
            or any(len(point) != 3 for point in target_reference)
            or any(len(point) != 3 for point in root_reference)
            or any(len(point) != 2 for point in parameters)
            or any(not value for value in charts)
            or any(not value for value in entities)
        ):
            raise ValueError("Mapped layer source-chart trace rows are malformed or incomplete.")
        if np.any(
            np.linalg.norm(np.cross(tangent[:, 0, :], tangent[:, 1, :]), axis=1) <= 0.0
        ) or np.any(np.linalg.norm(normal, axis=1) <= 0.0):
            raise ValueError("Mapped layer source chart has a singular physical frame.")
        self.source_id = str(source_id)
        self.source_revision = str(source_revision)
        self.frame_id = str(frame_id)
        self.query_id = str(query_id)
        self.geometry_node_rows = nodes
        self.target_cell_global_ids = cells
        self.target_local_dofs = dofs
        self.chart_witness_rows = chart_rows
        self.target_reference_points = target_reference
        self.root_reference_points = root_reference
        self.source_parameters = parameters
        self.source_chart_ids = charts
        self.source_entity_ids = entities
        self.occurrence_paths = paths
        self.root_face_global_ids = root_faces
        self.root_cell_global_ids = root_cells
        self.root_local_face_indices = root_local
        self.target_points = target
        self.source_points = source
        self.distance_lower_bounds = lower
        self.distance_upper_bounds = upper
        self.normals = normal
        self.tangents = tangent
        self.column_identity = column_identity
        self.trace_id = canonical_fingerprint(
            {
                "kind": "layer-curving-mapped-chart-trace",
                "source": (self.source_id, self.source_revision, self.frame_id, self.query_id),
                "geometry_nodes": array_tree_fingerprint(nodes),
                "target_cells": array_tree_fingerprint(cells),
                "target_dofs": array_tree_fingerprint(dofs),
                "chart_rows": array_tree_fingerprint(chart_rows),
                "target_reference": tuple(
                    tuple(_exact_scalar_identity(value) for value in row)
                    for row in target_reference
                ),
                "root_reference": tuple(
                    tuple(_exact_scalar_identity(value) for value in row)
                    for row in root_reference
                ),
                "parameters": tuple(
                    tuple(_exact_scalar_identity(value) for value in row)
                    for row in parameters
                ),
                "source_charts": (charts, entities, paths),
                "root_entities": (
                    array_tree_fingerprint(root_faces),
                    array_tree_fingerprint(root_cells),
                    array_tree_fingerprint(root_local),
                ),
                "target_points": array_tree_fingerprint(target),
                "source_points": array_tree_fingerprint(source),
                "distance_bounds": (array_tree_fingerprint(lower), array_tree_fingerprint(upper)),
                "normals": array_tree_fingerprint(normal),
                "tangents": array_tree_fingerprint(tangent),
                "column_identity": column_identity.identity_id,
            }
        )

    @property
    def resolved(self) -> bool:
        return bool(
            np.all(np.isfinite(self.target_points))
            and np.all(np.isfinite(self.source_points))
            and np.all(self.distance_lower_bounds <= self.distance_upper_bounds)
        )


class LayerCurvingSourceEvidence(StrictModule, NonTrainableState):
    """Revalidatable parametric, mapped, or adaptive-implicit source and layer ancestry."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    source_surface_charts: SurfaceSourceCharts | None
    source_query: SourceBoundaryQuery
    layer_result_id: str = eqx.field(static=True)
    layer_evidence: BoundaryLayerEvidence
    chart_column_identity: LayerColumnProjectionIdentity
    projection_trace: LayerCurvingProjectionTrace | None
    implicit_root_trace: LayerCurvingImplicitRootTrace | None
    mapped_chart_trace: LayerCurvingMappedChartTrace | None
    mapped_reference_maps: tuple[ReferenceCellMapWitnesses, ...]
    implicit_map_witness_id: str | None = eqx.field(static=True)
    implicit_chart_evidence: tuple[ImplicitRootChartEvidence, ...] = eqx.field(static=True)
    implicit_root_chart_witness: SourceBoundaryChartWitness | None
    implicit_root_cell_vertex_global_ids: NDArray[np.int64] | None
    chart_witness: SourceBoundaryChartWitness
    fidelity: SourceFidelityCertificate
    minimum_scaled_jacobian: float = eqx.field(static=True)
    maximum_distortion: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_surface_charts: SurfaceSourceCharts | None,
        source_query: SourceBoundaryQuery,
        layer_result_id: str,
        layer_evidence: BoundaryLayerEvidence,
        chart_witness: SourceBoundaryChartWitness,
        fidelity: SourceFidelityCertificate,
        minimum_scaled_jacobian: float,
        maximum_distortion: float,
        /,
        *,
        projection_trace: LayerCurvingProjectionTrace | None = None,
        implicit_root_trace: LayerCurvingImplicitRootTrace | None = None,
        implicit_map_witness_id: str | None = None,
        implicit_chart_evidence: Sequence[ImplicitRootChartEvidence] = (),
        implicit_root_chart_witness: SourceBoundaryChartWitness | None = None,
        implicit_root_cell_vertex_global_ids: ArrayLike | None = None,
        chart_column_identity: LayerColumnProjectionIdentity,
        mapped_chart_trace: LayerCurvingMappedChartTrace | None = None,
        mapped_reference_maps: Sequence[ReferenceCellMapWitnesses] = (),
        frame_id: str | None = None,
    ) -> None:
        if not isinstance(source_query, SourceBoundaryQuery):
            raise TypeError("source_query must implement SourceBoundaryQuery.")
        if isinstance(source_query, MappedReferenceBoundarySource):
            source_query.require_current()
        query_frame = _query_frame_id(source_query)
        selected_frame = frame_id or query_frame
        if selected_frame is None and isinstance(
            source_surface_charts, SurfaceSourceCharts
        ):
            selected_frame = source_surface_charts.coordinate_contract.spatial_id
        if not selected_frame or (query_frame and selected_frame != query_frame):
            raise ValueError(
                "Layer source query requires the exact current physical frame."
            )
        if not isinstance(layer_evidence, BoundaryLayerEvidence):
            raise TypeError("layer_evidence must be BoundaryLayerEvidence.")
        if not isinstance(chart_witness, SourceBoundaryChartWitness):
            raise TypeError("chart_witness must be SourceBoundaryChartWitness.")
        if (
            not isinstance(chart_column_identity, LayerColumnProjectionIdentity)
            or chart_column_identity.row_count != len(chart_witness.source_chart_ids)
        ):
            raise ValueError("Source-chart column ancestry must align with the retained chart witness.")
        if not isinstance(fidelity, SourceFidelityCertificate) or fidelity.status != "certified":
            raise ValueError("Accepted layer source evidence requires certified fidelity.")
        chart_rows = tuple(implicit_chart_evidence)
        root_cell_vertices = (
            None
            if implicit_root_cell_vertex_global_ids is None
            else np.array(implicit_root_cell_vertex_global_ids, dtype=np.int64, copy=True)
        )
        mapped_maps = tuple(mapped_reference_maps)
        root_query_id = getattr(
            source_query, "boundary_source_revision", source_query.source_revision
        )
        if source_surface_charts is not None:
            if not isinstance(source_surface_charts, SurfaceSourceCharts):
                raise TypeError("source_surface_charts must be SurfaceSourceCharts or None.")
            if (
                projection_trace is not None
                or implicit_root_trace is not None
                or implicit_map_witness_id is not None
                or chart_rows
                or implicit_root_chart_witness is not None
                or root_cell_vertices is not None
                or mapped_chart_trace is not None
                or mapped_maps
                or source_surface_charts.boundary_source is not source_query
            ):
                raise ValueError("Parametric evidence requires its retained exact source-chart authority.")
            if (
                source_surface_charts.source_id != source_query.source_id
                or source_surface_charts.source_revision != source_query.source_revision
                or source_surface_charts.coordinate_contract.spatial_id != selected_frame
            ):
                raise ValueError("Parametric source charts and query change source identity or frame.")
            source_surface_charts.require_current()
            source_surface_charts.require_bound(
                source_surface_charts.mesh,
                source_surface_charts.geometry,
                source_query,
            )
            source_cell_rows = {
                int(cell_id): tuple(
                    int(value) for value in np.asarray(block.vertices[row], dtype=np.int64)
                )
                for block in source_surface_charts.mesh.blocks
                for row, cell_id in enumerate(np.asarray(block.global_ids, dtype=np.int64))
            }
            source_vertex_ids = np.asarray(
                source_surface_charts.mesh.vertex_global_ids, dtype=np.int64
            )
            chart_cells = {
                int(cell_id): row
                for row, cell_id in enumerate(source_surface_charts.cell_global_ids)
            }
            for chart_row, source_cell_id in enumerate(
                chart_column_identity.source_wall_cell_ids
            ):
                source_cell = int(source_cell_id)
                source_chart_row = chart_cells.get(source_cell)
                vertices = source_cell_rows.get(source_cell)
                if source_chart_row is None or vertices is None:
                    raise ValueError("Parametric chart column ancestry names an absent source-wall cell.")
                patch = int(source_surface_charts.root_patches[source_chart_row])
                root_chart = _fraction_chart2(
                    source_surface_charts.root_parameter_chart(source_chart_row)
                )
                chart_parameters = _fraction_chart2(
                    chart_witness.source_parameter_chart(chart_row)
                )
                root_chart_solver = _prepare_barycentric_solver_2d(root_chart)
                if (
                    chart_witness.source_entity_ids[chart_row]
                    != source_surface_charts.domain.entity_id(2, patch)
                    or chart_witness.occurrence_paths[chart_row]
                    != tuple(source_surface_charts.domain.source_occurrences[2][patch])
                    or any(
                        any(
                            value < 0
                            for value in _barycentric_2d(
                                point, root_chart, solver=root_chart_solver
                            )
                        )
                        for point in chart_parameters
                    )
                    or not set(chart_column_identity.source_column_root_vertex_ids[chart_row])
                    <= {int(value) for value in source_vertex_ids}
                ):
                    raise ValueError("Parametric chart column ancestry changed its source face or root vertices.")
        elif isinstance(source_query, MappedReferenceBoundarySource):
            if (
                source_surface_charts is not None
                or projection_trace is not None
                or implicit_root_trace is not None
                or implicit_map_witness_id is not None
                or chart_rows
                or implicit_root_chart_witness is not None
                or root_cell_vertices is not None
                or not isinstance(mapped_chart_trace, LayerCurvingMappedChartTrace)
                or not mapped_chart_trace.resolved
                or not mapped_maps
                or chart_witness.chart_dimension != 2
                or chart_witness.reference_dimension != 3
                or chart_witness.source_query_id != source_query.source_query_id
            ):
                raise ValueError("Mapped layer evidence requires its exact root chart maps and source trace.")
            mapped_rows = _mapped_reference_chart_rows(
                source_query, chart_witness, mapped_maps
            )
            if (
                mapped_chart_trace.source_id != source_query.source_id
                or mapped_chart_trace.source_revision != source_query.source_revision
                or mapped_chart_trace.frame_id != selected_frame
                or mapped_chart_trace.query_id != source_query.source_query_id
                or np.any(mapped_chart_trace.chart_witness_rows >= len(mapped_rows))
            ):
                raise ValueError("Mapped layer source trace changes its source, frame, or chart scope.")
            for row, chart_row in enumerate(mapped_chart_trace.chart_witness_rows):
                query_row, root_face, root_cell, root_local, _ = mapped_rows[int(chart_row)]
                if (
                    mapped_chart_trace.source_chart_ids[row]
                    != chart_witness.source_chart_ids[int(chart_row)]
                    or mapped_chart_trace.source_entity_ids[row]
                    != chart_witness.source_entity_ids[int(chart_row)]
                    or mapped_chart_trace.occurrence_paths[row]
                    != chart_witness.occurrence_paths[int(chart_row)]
                    or int(mapped_chart_trace.root_face_global_ids[row]) != root_face
                    or int(mapped_chart_trace.root_cell_global_ids[row]) != root_cell
                    or int(mapped_chart_trace.root_local_face_indices[row]) != root_local
                    or int(mapped_chart_trace.target_cell_global_ids[row])
                    != int(chart_witness.cell_global_ids[int(chart_row)])
                    or _column_identity_row(
                        mapped_chart_trace.column_identity, row
                    )
                    != _column_identity_row(chart_column_identity, int(chart_row))
                    or source_query.chart_row(2, mapped_chart_trace.source_chart_ids[row])
                    != query_row
                ):
                    raise ValueError("Mapped layer source trace changes exact root, source, or column ancestry.")
        elif source_surface_charts is None:
            root_witness = implicit_root_chart_witness
            implicit_entity = source_entity_id(
                source_query.source_id,
                source_query.source_revision,
                "implicit-surface",
                "zero-set",
            )
            if (
                projection_trace is not None
                or not isinstance(implicit_root_trace, LayerCurvingImplicitRootTrace)
                or not implicit_root_trace.resolved
                or not implicit_map_witness_id
                or not chart_rows
                or any(not isinstance(row, ImplicitRootChartEvidence) for row in chart_rows)
                or implicit_root_trace.map_witness_id != implicit_map_witness_id
                or implicit_root_trace.chart_evidence != chart_rows
                or not isinstance(root_witness, SourceBoundaryChartWitness)
                or root_cell_vertices is None
                or root_cell_vertices.shape != (len(root_witness.source_chart_ids), 3)
                or np.any(root_cell_vertices < 0)
                or any(np.unique(row).size != 3 for row in root_cell_vertices)
                or root_witness.chart_dimension != 2
                or root_witness.reference_dimension != 2
                or root_witness.source_id != source_query.source_id
                or root_witness.source_revision != source_query.source_revision
                or root_witness.source_query_id != root_query_id
                or root_witness.source_entity_ids
                != tuple(implicit_entity for _ in root_witness.source_chart_ids)
                or chart_witness.source_entity_ids
                != tuple(implicit_entity for _ in chart_witness.source_chart_ids)
                or np.unique(root_witness.cell_global_ids).size
                != len(root_witness.source_chart_ids)
                or not np.array_equal(root_witness.entity_global_ids, root_witness.cell_global_ids)
                or np.any(root_witness.local_entity_indices != -1)
            ):
                raise ValueError("Adaptive implicit evidence requires its current source-wall root chart authority.")
            root_chart_rows = tuple(
                _implicit_root_witness_row(root_witness, chart_witness, row)
                for row in range(len(chart_witness.source_chart_ids))
            )
            root_vertex_ids = set(int(value) for value in root_cell_vertices.reshape(-1))
            for row, root_row in enumerate(root_chart_rows):
                if (
                    int(chart_column_identity.source_wall_cell_ids[row])
                    != int(root_witness.cell_global_ids[root_row])
                    or not set(chart_column_identity.source_column_root_vertex_ids[row])
                    <= root_vertex_ids
                ):
                    raise ValueError("Implicit chart column ancestry changed its source-wall root cell.")
            standard_chart = (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
            if tuple(
                root_witness.source_parameter_chart(row)
                for row in range(len(root_witness.source_chart_ids))
            ) != tuple(standard_chart for _ in root_witness.source_chart_ids):
                raise ValueError("Adaptive root witness does not retain canonical source parameters.")
        if (fidelity.source_id, fidelity.source_revision) != (
            source_query.source_id,
            source_query.source_revision,
        ):
            raise ValueError("Layer fidelity and source query change source identity.")
        if not isinstance(fidelity.witness, SourceBoundaryChartWitness):
            raise ValueError("Accepted layer source fidelity must retain an exact source chart witness.")
        if (
            chart_witness.source_id != source_query.source_id
            or chart_witness.source_revision != source_query.source_revision
            or chart_witness.witness_id != fidelity.witness.witness_id
        ):
            raise ValueError("Layer chart witness differs from its continuous source-fidelity proof.")
        if not layer_result_id:
            raise ValueError("Layer source evidence requires the immutable layer result identity.")
        if not np.isfinite(minimum_scaled_jacobian) or not np.isfinite(maximum_distortion):
            raise ValueError("Layer curving quality measurements must be finite.")
        self.source_id = source_query.source_id
        self.source_revision = source_query.source_revision
        self.frame_id = str(selected_frame)
        self.source_surface_charts = source_surface_charts
        self.source_query = source_query
        self.layer_result_id = str(layer_result_id)
        self.layer_evidence = layer_evidence
        self.chart_column_identity = chart_column_identity
        self.projection_trace = projection_trace
        self.implicit_root_trace = implicit_root_trace
        self.mapped_chart_trace = mapped_chart_trace
        self.mapped_reference_maps = mapped_maps
        self.implicit_map_witness_id = implicit_map_witness_id
        self.implicit_chart_evidence = chart_rows
        if root_cell_vertices is not None:
            root_cell_vertices.setflags(write=False)
        self.implicit_root_chart_witness = implicit_root_chart_witness
        self.implicit_root_cell_vertex_global_ids = root_cell_vertices
        self.chart_witness = chart_witness
        self.fidelity = fidelity
        self.minimum_scaled_jacobian = float(minimum_scaled_jacobian)
        self.maximum_distortion = float(maximum_distortion)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "layer-curving-source-evidence",
                "source": (self.source_id, self.source_revision, self.frame_id),
                "surface_chart": (
                    None
                    if source_surface_charts is None
                    else source_surface_charts.source_chart_id
                ),
                "layer_result": self.layer_result_id,
                "layer_evidence": layer_evidence.evidence_id,
                "chart_column_identity": chart_column_identity.identity_id,
                "projection_trace": (
                    None if projection_trace is None else projection_trace.trace_id
                ),
                "implicit_root_trace": (
                    None if implicit_root_trace is None else implicit_root_trace.trace_id
                ),
                "mapped_chart_trace": (
                    None if mapped_chart_trace is None else mapped_chart_trace.trace_id
                ),
                "mapped_reference_maps": tuple(
                    tuple(
                        (
                            mapping.target_cell_id,
                            mapping.source_cell_id,
                            reference_chart_arguments_identity((mapping.arguments,))
                            if mapping.arguments is not None
                            else None,
                        )
                        for mapping in nested_cell_reference_maps(reference_map)
                    )
                    for reference_map in mapped_maps
                ),
                "implicit_map": self.implicit_map_witness_id,
                "implicit_root_chart_witness": (
                    None
                    if implicit_root_chart_witness is None
                    else implicit_root_chart_witness.witness_id
                ),
                "implicit_root_cell_vertices": (
                    None
                    if root_cell_vertices is None
                    else array_tree_fingerprint(root_cell_vertices)
                ),
                "implicit_chart_evidence": tuple(
                    (
                        row.source_cell_key,
                        row.root_edge_keys,
                        row.status,
                        row.field_residual_bound,
                        row.target_to_root_bound,
                        row.source_normal_deviation_bounds,
                        row.chart_jacobian_lower,
                        row.chart_injectivity_margin,
                        row.source_gradient_lower,
                    )
                    for row in chart_rows
                ),
                "chart_witness": chart_witness.witness_id,
                "fidelity": fidelity.certificate_id,
                "quality": (self.minimum_scaled_jacobian, self.maximum_distortion),
            }
        )

    @property
    def column_identity(self) -> LayerColumnProjectionIdentity:
        return self.chart_column_identity

    @property
    def source_root_cell_ids(self) -> NDArray[np.int64]:
        if self.source_surface_charts is not None:
            chart_rows = {
                int(cell_id): row
                for row, cell_id in enumerate(self.source_surface_charts.cell_global_ids)
            }
            return np.asarray(
                [
                    self.source_surface_charts.source_root_cell_ids[
                        chart_rows[int(cell_id)]
                    ]
                    for cell_id in self.chart_column_identity.source_wall_cell_ids
                ],
                dtype=np.int64,
            )
        if self.projection_trace is not None:
            return self.projection_trace.source_root_cell_ids
        if self.mapped_chart_trace is not None:
            return self.mapped_chart_trace.root_cell_global_ids
        return self.chart_column_identity.source_wall_cell_ids


    @property
    def source_piece_ids(self) -> tuple[tuple[int, ...], ...]:
        if self.source_surface_charts is not None:
            chart_rows = {
                int(cell_id): row
                for row, cell_id in enumerate(self.source_surface_charts.cell_global_ids)
            }
            return tuple(
                tuple(
                    int(value)
                    for value in self.source_surface_charts.source_piece_ids[
                        chart_rows[int(cell_id)]
                    ]
                )
                for cell_id in self.chart_column_identity.source_wall_cell_ids
            )
        if self.projection_trace is not None:
            return self.projection_trace.source_piece_ids
        if self.mapped_chart_trace is not None:
            trace = self.mapped_chart_trace
            return tuple(
                (
                    int(trace.root_face_global_ids[row]),
                    int(trace.root_cell_global_ids[row]),
                    int(trace.root_local_face_indices[row]),
                )
                for row in range(trace.root_face_global_ids.size)
            )
        assert self.implicit_root_trace is not None
        return tuple(tuple(key) for key in self.implicit_root_trace.source_cell_keys)

    @property
    def source_entity_ids(self) -> tuple[str, ...]:
        if self.source_surface_charts is not None:
            return self.chart_witness.source_entity_ids
        if self.projection_trace is not None:
            return self.projection_trace.returned_entity_ids
        if self.mapped_chart_trace is not None:
            return self.mapped_chart_trace.source_entity_ids
        assert self.implicit_root_trace is not None
        return self.chart_witness.source_entity_ids

    @property
    def occurrence_paths(self) -> tuple[tuple[str, ...], ...]:
        if self.source_surface_charts is not None:
            return self.chart_witness.occurrence_paths
        if self.projection_trace is not None:
            return self.projection_trace.returned_occurrence_paths
        if self.mapped_chart_trace is not None:
            return self.mapped_chart_trace.occurrence_paths
        assert self.implicit_root_trace is not None
        return tuple(() for _ in self.implicit_root_trace.source_cell_keys)

    @property
    def source_chart_id(self) -> str:
        if self.source_surface_charts is not None:
            return self.source_surface_charts.source_chart_id
        if self.mapped_chart_trace is not None:
            return self.mapped_chart_trace.query_id
        assert self.implicit_root_trace is not None
        return self.implicit_root_trace.root_chart_witness_id

    def require_current(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError("require_current requires CellMesh and CellGeometrySpec.")
        query = self.source_query
        query_frame = _query_frame_id(query)
        current_frame = (
            query_frame
            if self.source_surface_charts is None
            else self.source_surface_charts.coordinate_contract.spatial_id
        )
        if (
            self.source_id != query.source_id
            or self.source_revision != query.source_revision
            or self.frame_id != current_frame
            or (query_frame is not None and self.frame_id != query_frame)
        ):
            raise ValueError(
                "Layer source evidence changed source revision or physical frame."
            )
        if (
            mesh.mesh_id != self.fidelity.mesh_id
            or mesh.topology_id != self.fidelity.topology_id
            or cell_geometry_id(geometry) != self.fidelity.geometry_id
            or self.chart_witness.mesh_id != mesh.mesh_id
            or self.chart_witness.topology_id != mesh.topology_id
            or self.chart_witness.geometry_id != cell_geometry_id(geometry)
            or not isinstance(self.fidelity.witness, SourceBoundaryChartWitness)
            or self.fidelity.witness.witness_id != self.chart_witness.witness_id
        ):
            raise ValueError("Layer source evidence is bound to different source or target operands.")
        self.fidelity.require_current(mesh, geometry, query)
        validate_source_fidelity_witness(mesh, geometry, query, self.chart_witness)
        if self.source_surface_charts is not None:
            if self.projection_trace is not None:
                raise ValueError("Exact parametric chart evidence cannot claim point projection.")
            charts = self.source_surface_charts
            charts.require_current()
            charts.require_root(charts.mesh, charts.geometry)
            charts.require_bound(charts.mesh, charts.geometry, query)
            source_chart_rows = {
                int(cell_id): row
                for row, cell_id in enumerate(charts.cell_global_ids)
            }
            source_wall_vertex_ids = {
                int(value) for value in np.asarray(charts.mesh.vertex_global_ids)
            }
            for chart_row, source_cell_id in enumerate(
                self.chart_column_identity.source_wall_cell_ids
            ):
                source_chart_row = source_chart_rows.get(int(source_cell_id))
                if source_chart_row is None:
                    raise ValueError("Layer column ancestry names an absent source-wall chart cell.")
                patch = int(charts.root_patches[source_chart_row])
                if (
                    self.chart_witness.source_entity_ids[chart_row]
                    != charts.domain.entity_id(2, patch)
                    or self.chart_witness.occurrence_paths[chart_row]
                    != tuple(charts.domain.source_occurrences[2][patch])
                    or not set(
                        self.chart_column_identity.source_column_root_vertex_ids[chart_row]
                    )
                    <= source_wall_vertex_ids
                ):
                    raise ValueError("Layer chart changed source entity, occurrence, or root-column identity.")
            _validate_parametric_chart_geometry(
                mesh,
                geometry,
                charts,
                self.chart_witness,
                self.chart_column_identity,
                self.fidelity.tolerance,
            )
            return
        elements, routes, coordinates = geometry.resolve(mesh)
        cell_rows = {
            int(cell_id): (block_row, cell_row)
            for block_row, block in enumerate(mesh.blocks)
            for cell_row, cell_id in enumerate(np.asarray(block.global_ids, dtype=np.int64))
        }
        if self.mapped_chart_trace is not None:
            if (
                not isinstance(query, MappedReferenceBoundarySource)
                or self.source_surface_charts is not None
                or self.implicit_root_trace is not None
                or not self.mapped_reference_maps
            ):
                raise ValueError("Mapped layer trace lost its exact original-source query or root maps.")
            trace = self.mapped_chart_trace
            mapped_rows = _mapped_reference_chart_rows(
                query, self.chart_witness, self.mapped_reference_maps
            )
            if (
                trace.source_id != self.source_id
                or trace.source_revision != self.source_revision
                or trace.frame_id != self.frame_id
                or trace.query_id != query.source_query_id
                or trace.column_identity.row_count != trace.geometry_node_rows.size
            ):
                raise ValueError("Mapped layer trace changed its source/frame/root-query identity.")
            chart_rows = tuple(int(value) for value in trace.chart_witness_rows)
            if any(
                chart_row < 0 or chart_row >= len(mapped_rows)
                for chart_row in chart_rows
            ):
                raise ValueError("Mapped layer trace names an absent target source chart.")
            contexts_by_trace_row = {}
            target_face_contexts = {}
            for chart_row in sorted(set(chart_rows)):
                trace_rows = tuple(
                    row
                    for row, candidate in enumerate(chart_rows)
                    if candidate == chart_row
                )
                mapping = mapped_rows[chart_row][4]
                contexts = _mapped_chart_reference_contexts(
                    query,
                    self.chart_witness,
                    chart_row,
                    mapping,
                    tuple(trace.target_reference_points[row] for row in trace_rows),
                )
                for row, context in zip(trace_rows, contexts, strict=True):
                    contexts_by_trace_row[row] = context
                block_row, _ = cell_rows[
                    int(self.chart_witness.cell_global_ids[chart_row])
                ]
                face_element = _face_reference_element(
                    elements[block_row],
                    mesh.blocks[block_row].cell_kind,
                )
                target_corners = _fraction_chart3(
                    self.chart_witness.target_reference_chart(chart_row)
                )
                target_solver = _prepare_barycentric_solver_3d(target_corners)
                target_face_contexts[chart_row] = (
                    target_corners,
                    target_solver,
                    dict(
                        _reference_face_dofs(
                            face_element,
                            target_corners,
                            solver=target_solver,
                        )
                    ),
                )
            for row, chart_row in enumerate(chart_rows):
                if chart_row < 0 or chart_row >= len(mapped_rows):
                    raise ValueError("Mapped layer trace names an absent target source chart.")
                query_row, root_face, root_cell, root_local, mapping = mapped_rows[chart_row]
                chart_id = self.chart_witness.source_chart_ids[chart_row]
                if (
                    trace.source_chart_ids[row] != chart_id
                    or trace.source_entity_ids[row]
                    != self.chart_witness.source_entity_ids[chart_row]
                    or trace.occurrence_paths[row]
                    != self.chart_witness.occurrence_paths[chart_row]
                    or int(trace.target_cell_global_ids[row])
                    != int(self.chart_witness.cell_global_ids[chart_row])
                    or int(trace.root_face_global_ids[row]) != root_face
                    or int(trace.root_cell_global_ids[row]) != root_cell
                    or int(trace.root_local_face_indices[row]) != root_local
                    or query.chart_row(2, chart_id) != query_row
                    or _column_identity_row(trace.column_identity, row)
                    != _column_identity_row(self.chart_column_identity, chart_row)
                ):
                    raise ValueError("Mapped layer trace changed source occurrence or layer-column/root identity.")
                target_corners, target_solver, face_reference_by_dof = (
                    target_face_contexts[chart_row]
                )
                local_dof = int(trace.target_local_dofs[row])
                try:
                    target_reference = face_reference_by_dof[local_dof]
                except KeyError as error:
                    raise ValueError(
                        "Mapped layer trace DOF is outside its target source face."
                    ) from error
                if any(
                    value < 0
                    for value in _barycentric_3d(
                        target_reference, target_corners, solver=target_solver
                    )
                ):
                    raise ValueError("Mapped layer trace DOF is outside its exact target chart.")
                (
                    root_reference,
                    source_uv,
                    source_chart_uv,
                    source_chart_solver,
                ) = contexts_by_trace_row[row]
                root_reference, source_parameter, source_point, tangents, normal = (
                    _evaluate_mapped_chart_point(
                        query,
                        self.chart_witness,
                        chart_row,
                        target_reference,
                        mapping,
                        root_reference,
                        source_uv,
                        source_chart_uv,
                        source_chart_solver,
                    )
                )
                distance_lower, distance_upper = _mapped_physical_distance_bounds(
                    query,
                    trace.source_chart_ids[row],
                    source_parameter,
                    trace.target_points[row],
                )
                if (
                    trace.target_reference_points[row] != target_reference
                    or trace.root_reference_points[row] != root_reference
                    or trace.source_parameters[row] != source_parameter
                    or not np.array_equal(source_point.view(np.uint64), trace.source_points[row].view(np.uint64))
                    or not np.array_equal(tangents.view(np.uint64), trace.tangents[row].view(np.uint64))
                    or not np.array_equal(normal.view(np.uint64), trace.normals[row].view(np.uint64))
                    or trace.distance_lower_bounds[row] != distance_lower
                    or trace.distance_upper_bounds[row] != distance_upper
                ):
                    raise ValueError("Mapped root source chart point or physical derivative evidence does not replay.")
            rows = trace.geometry_node_rows
            target_cells = trace.target_cell_global_ids
            local_dofs = trace.target_local_dofs
            target_points = trace.target_points
            reference_points = trace.target_reference_points
        else:
            if self.source_surface_charts is not None or self.implicit_root_trace is None:
                raise ValueError("Implicit root trace/source authority is incomplete.")
            trace = self.implicit_root_trace
            root_witness = self.implicit_root_chart_witness
            root_vertices = self.implicit_root_cell_vertex_global_ids
            root_scope = getattr(
                query, "boundary_source_revision", query.source_revision
            )
            implicit_entity = source_entity_id(
                self.source_id,
                self.source_revision,
                "implicit-surface",
                "zero-set",
            )
            if (
                not isinstance(root_witness, SourceBoundaryChartWitness)
                or root_vertices is None
                or root_vertices.shape != (len(root_witness.source_chart_ids), 3)
                or root_witness.source_id != self.source_id
                or root_witness.source_revision != self.source_revision
                or root_witness.source_query_id != root_scope
                or root_witness.source_entity_ids
                != tuple(implicit_entity for _ in root_witness.source_chart_ids)
                or self.chart_witness.source_entity_ids
                != tuple(implicit_entity for _ in self.chart_witness.source_chart_ids)
                or np.unique(root_witness.cell_global_ids).size
                != len(root_witness.source_chart_ids)
                or not np.array_equal(root_witness.entity_global_ids, root_witness.cell_global_ids)
                or np.any(root_witness.local_entity_indices != -1)
            ):
                raise ValueError("Implicit layer evidence changed its source-wall root-chart authority.")
            root_chart_rows = tuple(
                _implicit_root_witness_row(root_witness, self.chart_witness, row)
                for row in range(len(self.chart_witness.source_chart_ids))
            )
            root_cell_rows = {
                int(cell_id): row
                for row, cell_id in enumerate(root_witness.cell_global_ids)
            }
            if len(root_cell_rows) != len(root_witness.source_chart_ids):
                raise ValueError("Implicit source-wall root-cell IDs are not unique.")
            root_vertex_ids = set(int(value) for value in root_vertices.reshape(-1))
            for row, chart_row in enumerate(trace.chart_rows):
                chart_index = int(chart_row)
                if chart_index < 0 or chart_index >= len(root_chart_rows):
                    raise ValueError("Implicit trace chart row is absent from the target witness.")
                root_index = root_chart_rows[chart_index]
                if (
                    int(self.chart_witness.cell_global_ids[chart_index])
                    != int(trace.target_cell_global_ids[row])
                    or _column_identity_row(
                        trace.column_identity, row
                    )
                    != _column_identity_row(self.chart_column_identity, chart_index)
                    or int(self.chart_column_identity.source_wall_cell_ids[chart_index])
                    != int(root_witness.cell_global_ids[root_index])
                    or trace.source_cell_keys[row]
                    != tuple(int(value) for value in root_witness.implicit_cell_keys[root_index])
                    or trace.root_edge_keys[row]
                    != tuple(
                        tuple(
                            tuple(int(value) for value in endpoint)
                            for endpoint in edge
                        )
                        for edge in root_witness.implicit_root_edge_keys[root_index]
                    )
                    or not set(
                        self.chart_column_identity.source_column_root_vertex_ids[chart_index]
                    )
                    <= root_vertex_ids
                ):
                    raise ValueError("Implicit layer column/root ancestry does not match its retained source chart.")
            if not isinstance(query, SourceBoundaryProjectionQuery):
                raise ValueError("Adaptive implicit evidence requires its native projection query.")
            adaptive_query = _adaptive_implicit_source_query(query)
            from ..geometry.implicit._source_charts import (
                prepare_adaptive_implicit_root_charts,
            )

            root_map = prepare_adaptive_implicit_root_charts(
                mesh,
                geometry,
                adaptive_query,
                self.chart_witness,
                frame_id=self.frame_id,
                tolerance=self.fidelity.tolerance,
                maximum_normal_deviation=self.fidelity.maximum_normal_deviation,
            )
            if (
                root_map.witness_id != self.implicit_map_witness_id
                or tuple(root_map.chart_evidence) != self.implicit_chart_evidence
                or root_map.status != "certified"
            ):
                raise ValueError("Adaptive implicit root-map evidence no longer revalidates.")
            for row, chart_row in enumerate(trace.chart_rows):
                point = root_map.project_target(
                    int(chart_row), trace.target_reference_points[row]
                )
                if (
                    point.status != trace.statuses[row]
                    or point.source_cell_key != trace.source_cell_keys[row]
                    or point.root_edge_keys != trace.root_edge_keys[row]
                    or not np.array_equal(point.point, trace.source_points[row])
                    or point.displacement_upper != trace.displacement_upper[row]
                    or point.root_error != trace.root_error[row]
                ):
                    raise ValueError("Adaptive implicit root lift does not replay for this target map.")
            rows = trace.geometry_node_rows
            target_cells = trace.target_cell_global_ids
            local_dofs = trace.target_local_dofs
            target_points = trace.target_points
            reference_points = trace.target_reference_points
        for row, (node_row, cell_id, local_dof, reference) in enumerate(
            zip(rows, target_cells, local_dofs, reference_points, strict=True)
        ):
            try:
                block_row, cell_row = cell_rows[int(cell_id)]
            except KeyError as error:
                raise ValueError("Layer source map names an absent target cell.") from error
            reference_element = _face_reference_element(
                elements[block_row], mesh.blocks[block_row].cell_kind
            )
            if not isinstance(reference_element, FiniteElementSpec):
                raise ValueError(
                    "Layer source map requires a canonical exact nodal element."
                )
            reference_nodes = lagrange_reference_node_data(reference_element)
            if local_dof < 0 or local_dof >= len(reference_nodes):
                raise ValueError("Layer source map changed its target reference DOF identity.")
            node = reference_nodes[local_dof]
            if node.local_dof != local_dof or node.exact_point is None:
                raise ValueError(
                    "Layer source map target DOF has no canonical exact reference point."
                )
            expected_reference = _exact3(node.exact_point)
            if expected_reference != reference:
                raise ValueError("Layer source map changed its exact target reference point.")
            if node_row >= 0:
                route = np.asarray(routes[block_row], dtype=np.int64)
                if (
                    local_dof >= route.shape[1]
                    or int(route[cell_row, local_dof]) != int(node_row)
                ):
                    raise ValueError("Layer source map changed exact cell/dof/node identity.")
                actual_point = np.asarray(coordinates[int(node_row)], dtype=np.float64)
            elif node_row == -1:
                actual_point = _evaluate_cell_geometry_point(
                    elements,
                    routes,
                    coordinates,
                    block_row,
                    cell_row,
                    reference,
                )
            else:
                raise ValueError("Layer source map has an invalid virtual geometry-node sentinel.")
            if not np.array_equal(
                np.asarray(actual_point, dtype=np.float64).view(np.uint64),
                np.asarray(target_points[row], dtype=np.float64).view(np.uint64),
            ):
                raise ValueError("Layer source map queries no longer equal accepted map nodes.")

__all__ = [
    "LayerColumnProjectionIdentity",
    "LayerCurvingImplicitRootTrace",
    "LayerCurvingMappedChartTrace",
    "LayerCurvingProjectionTrace",
    "LayerCurvingSourceEvidence",
]

