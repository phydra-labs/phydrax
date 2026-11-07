from __future__ import annotations

from dataclasses import fields
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
from ..geometry._domain_boundary import MeshingDomainBoundarySource
from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ..geometry._mesh_certificates import _same_tree_content
from ..geometry._rational import fraction_endpoint
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryProjectionQuery,
    SourceBoundaryQuery,
    SourceProjectionStatus,
)
from ..geometry._source_fidelity import (
    certify_source_fidelity,
    validate_source_fidelity_witness,
)
from ..geometry._source_fidelity_certificate import SourceFidelityCertificate
from ..geometry._source_identity import source_entity_id
from ..geometry._source_interpolation import exact_surface_coordinate_expressions
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..geometry.brep._patches import surface_differential
from ..geometry.implicit._adaptive_discovery import AdaptiveImplicitBoundarySource
from ..geometry.implicit._source_charts import (
    _AdaptiveImplicitBoundaryCarrier,
    AdaptiveImplicitRootChartMap,
    AdaptiveImplicitSourceQuery,
    ImplicitRootChartEvidence,
)
from ..linalg._exact import (
    ExactRationalAlgebraicFactorization,
    factor_exact_rational_algebraic,
)
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ._boundary_layer import BoundaryLayerEvidence, BoundaryLayerMesh, BoundaryLayerPolicy
from ._controls import BoundaryLayerControl


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
    raise TypeError(
        "Adaptive layer source query must retain its native adaptive chart authority."
    )


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
        raise ValueError(
            "Exact layer maps require a canonical degree-one reference element."
        )
    node_data = lagrange_reference_node_data(element)
    vertices: list[tuple[Fraction, Fraction, Fraction]] = []
    for dofs in element.entity_dofs[0]:
        if len(dofs) != 1:
            raise ValueError(
                "A degree-one reference vertex must have one exact nodal DOF."
            )
        local_dof = int(dofs[0])
        node = node_data[local_dof]
        if node.local_dof != local_dof or node.exact_point is None:
            raise ValueError(
                "A reference vertex has no canonical exact nodal coordinate."
            )
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


def _reference_expression(expression: ExactReferenceChartExpression, /) -> Expression:
    return expression if isinstance(expression, RationalPolynomial) else dict(expression)


type _ImplicitCellKey = tuple[int, int, int, int]
type _ImplicitRootEdge = tuple[tuple[int, int, int], tuple[int, int, int]]
type _ImplicitRootEdgeRow = tuple[_ImplicitRootEdge, _ImplicitRootEdge, _ImplicitRootEdge]


def _implicit_cell_key(values: Sequence[int], /) -> _ImplicitCellKey:
    if len(values) != 4:
        raise ValueError(
            "Adaptive implicit source-cell keys must retain four coordinates."
        )
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
        raise ValueError(
            "Adaptive implicit source triangles must retain three root edges."
        )
    return (
        _implicit_root_edge(edges[0]),
        _implicit_root_edge(edges[1]),
        _implicit_root_edge(edges[2]),
    )


def _implicit_root_witness_rows(
    root_witness: SourceBoundaryChartWitness,
    target_witness: SourceBoundaryChartWitness,
    source_wall_cell_ids: Sequence[int],
    /,
) -> tuple[int, ...]:
    """Bind each target chart to the root-witness row of its own source-wall cell.

    A remeshed wall seeds several cells with one discovery triangle, so the
    chart, cell key, and root edges alone do not name a root row. The layer
    column of every target chart retains its exact source-wall cell; that cell's
    row is the target's root piece, and it must carry the target's root chart.
    """
    if len(source_wall_cell_ids) != len(target_witness.source_chart_ids):
        raise ValueError(
            "Adaptive target charts and their source-wall cells are not aligned."
        )
    root_rows = {
        int(cell_id): row for row, cell_id in enumerate(root_witness.cell_global_ids)
    }
    if len(root_rows) != len(root_witness.source_chart_ids):
        raise ValueError("Implicit source-wall root-cell IDs are not unique.")
    result: list[int] = []
    for target_row, cell_id in enumerate(source_wall_cell_ids):
        row = root_rows.get(int(cell_id))
        if (
            row is None
            or root_witness.source_chart_ids[row]
            != target_witness.source_chart_ids[target_row]
            or not np.array_equal(
                root_witness.implicit_cell_keys[row],
                target_witness.implicit_cell_keys[target_row],
            )
            or not np.array_equal(
                root_witness.implicit_root_edge_keys[row],
                target_witness.implicit_root_edge_keys[target_row],
            )
        ):
            raise ValueError(
                "Adaptive target chart differs from its source-wall cell's retained root witness."
            )
        result.append(row)
    return tuple(result)


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
                raise ValueError(
                    "Mapped layer witness requires an exact target-to-root expression bank."
                )
            key = (mapping.target_cell_id, mapping.source_cell_id)
            identity = reference_chart_arguments_identity((arguments,))
            if key in signatures and signatures[key] != identity:
                raise ValueError(
                    "Mapped layer cell reference map repeats conflicting source charts."
                )
            by_pair[key] = mapping
            signatures[key] = identity
    relation = query.authority.root_relation
    root_cells = np.asarray(relation.polygon_root_cell_global_ids, dtype=np.int64)
    root_locals = np.asarray(relation.polygon_root_local_face_indices, dtype=np.int64)
    if root_cells.shape != (len(query.face_chart_ids), 2) or root_locals.shape != root_cells.shape:
        raise ValueError("Mapped layer source root-side incidence is malformed.")
    result = []
    for row, chart_id in enumerate(witness.source_chart_ids):
        query_row = query.chart_row(2, chart_id)
        entity_key = query.chart_source_key(2, chart_id)
        if entity_key != (witness.source_entity_ids[row], witness.occurrence_paths[row]):
            raise ValueError("Mapped layer witness changed its source entity/occurrence.")
        root_face = int(query.polygon_root_face_global_ids[query_row])
        target_cell = int(witness.cell_global_ids[row])
        sides = [
            (int(root_cells[query_row, side]), int(root_locals[query_row, side]))
            for side in range(2)
            if int(root_cells[query_row, side]) >= 0
            and (target_cell, int(root_cells[query_row, side])) in by_pair
        ]
        if len(sides) != 1:
            raise ValueError(
                "Mapped layer target chart must bind exactly one retained root-cell side."
            )
        root_cell, root_local = sides[0]
        mapping = by_pair[(target_cell, root_cell)]
        arguments = mapping.arguments
        if arguments is None or len(arguments) != 3:
            raise ValueError(
                "Mapped layer witness requires its exact target-to-root expression bank."
            )
        target_points = _fraction_chart3(witness.target_reference_chart(row))
        exact_arguments = tuple(_reference_expression(value) for value in arguments)
        if _determinant_3x3(_exact_map_jacobian(exact_arguments, target_points[0])) == 0:
            raise ValueError("Mapped layer target-to-root reference map is singular.")
        root_points = tuple(
            _fraction3(mapping.evaluate(point)) for point in target_points
        )
        root_uvs = _fraction_chart2(
            query.root_face_uv(
                root_face,
                root_cell,
                root_local,
                root_points,
            )
        )
        # Witness parameters are chart-local (barycentric in the source chart's
        # root-face polygon), as _evaluate_mapped_chart_point consumes them.
        source_chart = _fraction_chart2(query.chart_face_uv(chart_id))
        solver = _prepare_barycentric_solver_2d(source_chart)
        source_parameters = tuple(
            _exact2(weights[1:])
            for weights in (
                _barycentric_2d(uv, source_chart, solver=solver) for uv in root_uvs
            )
        )
        if source_parameters != _fraction_chart2(witness.source_parameter_chart(row)):
            raise ValueError(
                "Mapped layer witness source parameters differ from its exact root map."
            )
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
        raise ValueError(
            "Mapped root cell chart must retain three exact coordinate expressions."
        )
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
        corners[0][axis] + alpha * first[axis] + beta * second[axis] != point[axis]
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
    relation = query.authority.root_relation
    side_cells = np.asarray(relation.polygon_root_cell_global_ids, dtype=np.int64)[query_row]
    side_locals = np.asarray(relation.polygon_root_local_face_indices, dtype=np.int64)[query_row]
    sides = np.flatnonzero(side_cells == mapping.source_cell_id)
    if sides.size != 1:
        raise ValueError("Mapped chart nodes do not bind a unique root-cell side.")
    root_cell = int(mapping.source_cell_id)
    root_local = int(side_locals[int(sides[0])])
    root_references = tuple(
        _exact3(mapping.evaluate(point)) for point in target_references
    )
    source_uvs = query.root_face_uv(root_face, root_cell, root_local, root_references)
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
        for root_reference, source_uv in zip(root_references, source_uvs, strict=True)
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
        raise ValueError(
            "Mapped layer source chart has no retained exact root reference expressions."
        )
    exact_arguments = tuple(_reference_expression(value) for value in arguments)
    if (
        _exact3(mapping.evaluate(target_reference)) != root_reference
        or _determinant_3x3(_exact_map_jacobian(exact_arguments, target_reference)) == 0
    ):
        raise ValueError(
            "Mapped layer target-to-root reference map is singular or stale."
        )
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
        evaluate_expression(expression, chart_parameter) for expression in expressions
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
        evaluate_expression(expression, source_parameter) for expression in expressions
    )
    target_exact = tuple(
        Fraction(float(value)) for value in np.asarray(target_point, dtype=np.float64)
    )
    if len(source_exact) != 3:
        raise ValueError(
            "Mapped source physical distance requires three root-chart coordinates."
        )
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


def _face_reference_element(element: object, cell_kind: str, /) -> _NodalReferenceElement:
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
            weights = _barycentric_3d(reference, corners, solver=barycentric_solver)
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
        raise ValueError(
            "Exact target coordinate map produced an invalid source-bound point."
        )
    return point


def _require_reoriented_wall_side(
    charts: SurfaceSourceCharts,
    chart_row: int,
    actual_chart: Sequence[Sequence[Fraction]],
    target_area: Fraction,
    /,
) -> None:
    """Bind a changed, reversed atlas row to the actual outward region side."""
    patch = int(charts.root_patches[chart_row])
    original = charts.root_parameter_exact[chart_row]
    original_area = (
        (original[1][0] - original[0][0]) * (original[2][1] - original[0][1])
        - (original[1][1] - original[0][1]) * (original[2][0] - original[0][0])
    )
    orientation = charts.domain.patches[patch].orientation
    if tuple(tuple(point) for point in actual_chart) == original or original_area * orientation >= 0:
        return
    uses = tuple(
        sign for region in charts.domain.regions
        for region_patch, sign in region.boundary if region_patch == patch
    )
    if len(uses) != 1 or target_area * orientation * uses[0] <= 0:
        raise ValueError("Reoriented wall chart does not match the authored region side.")


def parametric_wall_query(
    charts: SurfaceSourceCharts, source_cell_ids: Sequence[int], /
) -> MeshingDomainBoundarySource:
    """Bind complete grown wall patches to the retained full source atlas."""
    charts.require_current()
    charts.require_root(charts.mesh, charts.geometry)
    parent = charts.boundary_source
    charts.require_bound(charts.mesh, charts.geometry, parent)
    selected_cells = {int(value) for value in source_cell_ids}
    cell_patches = {
        int(identifier): int(patch)
        for identifier, patch in zip(charts.cell_global_ids, charts.root_patches, strict=True)
    }
    if not selected_cells or not selected_cells <= set(cell_patches):
        raise ValueError("Selected layer wall cells do not bind the retained source atlas.")
    patches = tuple(sorted({cell_patches[cell] for cell in selected_cells}))
    if not set(patches) <= set(parent.patches):
        raise ValueError("Selected layer wall patches leave the original source query.")
    complete = {cell for cell, patch in cell_patches.items() if patch in patches}
    if selected_cells != complete:
        raise ValueError("A selected layer wall must cover each original source patch completely.")
    if patches == parent.patches:
        return parent
    cells = parent.boundary_cell_ids
    owners = parent.boundary_patch_ids
    occurrences = parent.boundary_occurrences
    if cells is None or owners is None or occurrences is None:
        if cells is not None or owners is not None or occurrences is not None:
            raise ValueError("Original source boundary occurrence columns are incomplete.")
        return MeshingDomainBoundarySource(
            charts.domain, patches, resolution=parent.resolution,
            validated=parent.validated_domain,
        )
    selected = np.isin(np.asarray(owners, dtype=np.int64), patches)
    return MeshingDomainBoundarySource(
        charts.domain, patches, resolution=parent.resolution,
        boundary_cell_ids=np.asarray(cells)[selected],
        boundary_patch_ids=np.asarray(owners)[selected],
        boundary_occurrences=np.asarray(occurrences)[selected],
        validated=parent.validated_domain,
    )


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
        raise ValueError(
            "Source-chart physical tolerance must be finite and nonnegative."
        )
    charts.require_current()
    charts.require_root(charts.mesh, charts.geometry)
    charts.require_bound(charts.mesh, charts.geometry, charts.boundary_source)
    from ..geometry._source_fidelity import reprove_actual_planar_chart_rows

    actual_chart_rows = reprove_actual_planar_chart_rows(charts)
    if (
        charts.source_id != witness.source_id
        or charts.source_revision != witness.source_revision
        or parametric_wall_query(
            charts, tuple(int(value) for value in column_identity.source_wall_cell_ids)
        ).source_scope_id != witness.source_query_id
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
    for chart_row, source_cell_id in enumerate(column_identity.source_wall_cell_ids):
        source_chart_row = source_chart_rows.get(int(source_cell_id))
        if source_chart_row is None:
            raise ValueError(
                "Layer column ancestry names an absent source-wall chart cell."
            )
        patch = int(charts.root_patches[source_chart_row])
        source_entity = charts.domain.entity_id(2, patch)
        occurrence = tuple(charts.domain.source_occurrences[2][patch])
        source_parameters = _fraction_chart2(witness.source_parameter_chart(chart_row))
        root_chart = _fraction_chart2(
            charts.root_parameter_chart(source_chart_row)
            if actual_chart_rows is None
            else actual_chart_rows[source_chart_row]
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
            raise ValueError(
                "Layer source chart changed its exact entity, occurrence, or root patch."
            )
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
        target_corners = _fraction_chart3(witness.target_reference_chart(chart_row))
        original_chart = _fraction_chart2(charts.root_parameter_chart(source_chart_row))
        original_area = (
            (original_chart[1][0] - original_chart[0][0])
            * (original_chart[2][1] - original_chart[0][1])
            - (original_chart[1][1] - original_chart[0][1])
            * (original_chart[2][0] - original_chart[0][0])
        )
        if (
            actual_chart_rows is not None
            and root_chart != original_chart
            and original_area * charts.domain.patches[patch].orientation < 0
        ):
            from ..discretization._reference_cell import reference_cell_topology
            from ._boundary_layer import _FACE_ROUTES

            topology = reference_cell_topology(block.cell_kind)
            slots = _FACE_ROUTES[block.cell_kind][local_face]
            face = tuple(
                _fraction3(tuple(Fraction(float(value)) for value in topology.vertices[slot]))
                for slot in slots[:3]
            )
            face_solver = _prepare_barycentric_solver_3d(face)
            weights = tuple(
                _barycentric_3d(point, face, solver=face_solver)
                for point in target_corners
            )
            reference_area = _fraction(
                (weights[1][1] - weights[0][1]) * (weights[2][2] - weights[0][2])
                - (weights[1][2] - weights[0][2]) * (weights[2][1] - weights[0][1])
            )
            source_area = (
                (source_parameters[1][0] - source_parameters[0][0])
                * (source_parameters[2][1] - source_parameters[0][1])
                - (source_parameters[1][1] - source_parameters[0][1])
                * (source_parameters[2][0] - source_parameters[0][0])
            )
            _require_reoriented_wall_side(
                charts, source_chart_row, root_chart, reference_area * source_area
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
                        (
                            weights[corner] * source_parameters[corner][axis]
                            for corner in range(3)
                        ),
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
                raise ValueError(
                    "A target boundary DOF leaves its exact source root chart."
                )
            actual_element = elements[block_row]
            if not isinstance(actual_element, _NodalReferenceElement):
                # The synthesized target reference nodes do not index the
                # coefficient bank of a composed/restricted coordinate map.
                target_point = _evaluate_cell_geometry_point(
                    elements, routes, coordinates, block_row, cell_row, reference
                )
            else:
                route = np.asarray(routes[block_row], dtype=np.int64)
                geometry_row = int(route[cell_row, local_dof])
                if geometry_row >= 0:
                    target_point = np.asarray(coordinates[geometry_row], dtype=np.float64)
                elif geometry_row == -1:
                    target_point = _evaluate_cell_geometry_point(
                        elements, routes, coordinates, block_row, cell_row, reference
                    )
                else:
                    raise ValueError(
                        "A source-chart boundary DOF has an invalid map-node sentinel."
                    )
            patches.append(patch)
            parameters.append((float(source_uv[0]), float(source_uv[1])))
            exact_parameters.append(source_uv)
            target_points.append(target_point)
            covered_dofs.setdefault(face_key, set()).add(local_dof)
    if not patches or any(
        covered_dofs.get(key, set()) != value for key, value in expected_dofs.items()
    ):
        raise ValueError(
            "Exact source charts do not cover every target boundary geometry DOF."
        )
    targets = np.asarray(target_points, dtype=np.float64)
    source_points = np.empty_like(targets)
    exact_surface_maps: dict[int, tuple[Expression, ...]] = {}
    for row, (patch, parameter) in enumerate(zip(patches, exact_parameters, strict=True)):
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
        raise ValueError(
            "Target coordinate map is outside its exact source-chart tolerance."
        )
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
            raise ValueError(
                "Exact source-chart derivative returned an invalid physical frame."
            )
        cross = np.cross(differential[:, :, 0], differential[:, :, 1])
        lengths = np.linalg.norm(cross, axis=1)
        if np.any(~np.isfinite(lengths) | (lengths <= 0.0)):
            raise ValueError("Exact source-chart derivative is physically singular.")
        oriented = (
            int(charts.domain.patches[patch].orientation) * cross / lengths[:, None]
        )
        if np.any(np.sum(oriented * normals[selected], axis=1) <= 0.0):
            raise ValueError(
                "Exact source-chart derivative changed its physical frame orientation."
            )


class LayerColumnProjectionIdentity(StrictModule, NonTrainableState):
    """Stable layer schedule, front-row, column, and source-root ancestry per node."""

    layer_indices: NDArray[np.int64]
    source_wall_cell_ids: NDArray[np.int64]
    front_row_ids: NDArray[np.int64]
    front_row_identities: tuple[str, ...] = eqx.field(static=True)
    source_column_ids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    source_column_identities: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_column_root_vertex_ids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    allow_unbound: bool = eqx.field(static=True)
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
        *,
        allow_unbound: bool = False,
    ) -> None:
        layers = np.asarray(layer_indices, dtype=np.int64)
        wall_cells = np.asarray(source_wall_cell_ids, dtype=np.int64)
        front_rows = np.asarray(front_row_ids, dtype=np.int64)
        identities = tuple(str(value) for value in front_row_identities)
        columns = tuple(tuple(int(value) for value in row) for row in source_column_ids)
        column_identities = tuple(
            tuple(str(value) for value in row) for row in source_column_identities
        )
        roots = tuple(
            tuple(int(value) for value in row) for row in source_column_root_vertex_ids
        )
        count = layers.size
        if (
            layers.ndim != 1
            or wall_cells.shape != (count,)
            or front_rows.shape != (count,)
            or len(identities) != count
            or len(columns) != count
            or len(column_identities) != count
            or len(roots) != count
            or not isinstance(allow_unbound, bool)
            or any(
                not (
                    layer >= 0 and wall >= 0 and front >= 0 and name
                    and column_row
                    and len(column_row) == len(name_row) == len(root_row)
                    and all(value >= 0 for value in column_row)
                    and all(name_row)
                    and all(value >= 0 for value in root_row)
                )
                and not (
                    allow_unbound and layer == wall == front == -1 and not name
                    and not column_row and not name_row and not root_row
                )
                for layer, wall, front, name, column_row, name_row, root_row in zip(
                    layers, wall_cells, front_rows, identities, columns,
                    column_identities, roots, strict=True
                )
            )
        ):
            raise ValueError(
                "Layer column/root ancestry rows are incomplete or misaligned."
            )
        self.layer_indices = layers
        self.source_wall_cell_ids = wall_cells
        self.front_row_ids = front_rows
        self.front_row_identities = identities
        self.source_column_ids = columns
        self.source_column_identities = column_identities
        self.source_column_root_vertex_ids = roots
        self.allow_unbound = allow_unbound
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
                "allow_unbound": allow_unbound,
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
            or any(
                value.shape != (count,)
                for value in (cells, dofs, roots, lower, upper, states)
            )
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
            or any(
                len(value) != count
                for value in (
                    requested_entities,
                    returned_entities,
                    requested_paths,
                    returned_paths,
                    source_ids,
                    revisions,
                    frames,
                )
            )
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
            and np.all(
                np.isin(
                    self.status,
                    (
                        int(SourceProjectionStatus.UNIQUE),
                        int(SourceProjectionStatus.SEAM),
                    ),
                )
            )
            and self.returned_entity_ids == self.requested_entity_ids
            and self.returned_occurrence_paths == self.requested_occurrence_paths
            and all(value == self.source_id for value in self.returned_source_ids)
            and all(
                value == self.source_revision for value in self.returned_source_revisions
            )
            and all(value == self.frame_id for value in self.returned_frame_ids)
            and np.all(np.isfinite(self.parameters))
            and np.all(np.isfinite(self.projected_points))
            and np.all(np.isfinite(self.normals))
            and np.all(np.isfinite(self.tangents))
            and np.all(np.isfinite(self.distance_lower_bounds))
            and np.all(np.isfinite(self.residuals))
            and np.all(
                (self.distance_lower_bounds >= 0.0)
                & (self.distance_lower_bounds <= self.residuals)
            )
            and np.all(
                np.linalg.norm(
                    np.cross(self.tangents[:, 0, :], self.tangents[:, 1, :]), axis=1
                )
                > 0.0
            )
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
            not all(
                (
                    source_id,
                    source_revision,
                    frame_id,
                    map_witness_id,
                    root_chart_witness_id,
                )
            )
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
            raise ValueError(
                "Implicit layer root-lift rows are unresolved or misaligned."
            )
        if np.any(rows < 0) or (rows.size and np.max(rows) >= len(evidence)):
            raise ValueError(
                "Implicit layer root-lift chart row is absent from its evidence bank."
            )
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
            raise ValueError(
                "Mapped layer source-chart trace rows are malformed or incomplete."
            )
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
                "source": (
                    self.source_id,
                    self.source_revision,
                    self.frame_id,
                    self.query_id,
                ),
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
                "distance_bounds": (
                    array_tree_fingerprint(lower),
                    array_tree_fingerprint(upper),
                ),
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
    implicit_chart_evidence: tuple[ImplicitRootChartEvidence, ...] = eqx.field(
        static=True
    )
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
        if not isinstance(
            chart_column_identity, LayerColumnProjectionIdentity
        ) or chart_column_identity.row_count != len(chart_witness.source_chart_ids):
            raise ValueError(
                "Source-chart column ancestry must align with the retained chart witness."
            )
        if chart_column_identity.allow_unbound and not isinstance(
            source_query, MappedReferenceBoundarySource
        ):
            raise ValueError("Only mapped core charts may lack layer-column ancestry.")
        if (
            not isinstance(fidelity, SourceFidelityCertificate)
            or fidelity.status != "certified"
        ):
            raise ValueError(
                "Accepted layer source evidence requires certified fidelity."
            )
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
                raise TypeError(
                    "source_surface_charts must be SurfaceSourceCharts or None."
                )
            if not isinstance(source_query, MeshingDomainBoundarySource):
                raise TypeError("Parametric layer evidence requires the original domain boundary query.")
            source_query.require_current()
            if (
                projection_trace is not None
                or implicit_root_trace is not None
                or implicit_map_witness_id is not None
                or chart_rows
                or implicit_root_chart_witness is not None
                or root_cell_vertices is not None
                or mapped_chart_trace is not None
                or mapped_maps
            ):
                raise ValueError(
                    "Parametric evidence requires its retained exact source-chart authority."
                )
            if (
                source_surface_charts.source_id != source_query.source_id
                or source_surface_charts.source_revision != source_query.source_revision
                or source_surface_charts.coordinate_contract.spatial_id != selected_frame
            ):
                raise ValueError(
                    "Parametric source charts and query change source identity or frame."
                )
            expected_query = parametric_wall_query(
                source_surface_charts,
                tuple(int(value) for value in chart_column_identity.source_wall_cell_ids),
            )
            if not _same_tree_content(source_query, expected_query):
                raise ValueError("Parametric layer query changes the actual selected source patches.")
            from ..geometry._source_fidelity import reprove_actual_planar_chart_rows

            actual_chart_rows = reprove_actual_planar_chart_rows(source_surface_charts)
            source_cell_rows = {
                int(cell_id): tuple(
                    int(value)
                    for value in np.asarray(block.vertices[row], dtype=np.int64)
                )
                for block in source_surface_charts.mesh.blocks
                for row, cell_id in enumerate(
                    np.asarray(block.global_ids, dtype=np.int64)
                )
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
                    raise ValueError(
                        "Parametric chart column ancestry names an absent source-wall cell."
                    )
                patch = int(source_surface_charts.root_patches[source_chart_row])
                root_chart = _fraction_chart2(
                    source_surface_charts.root_parameter_chart(source_chart_row)
                    if actual_chart_rows is None
                    else actual_chart_rows[source_chart_row]
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
                    or not set(
                        chart_column_identity.source_column_root_vertex_ids[chart_row]
                    )
                    <= {int(value) for value in source_vertex_ids}
                ):
                    raise ValueError(
                        "Parametric chart column ancestry changed its source face or root vertices."
                    )
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
                raise ValueError(
                    "Mapped layer evidence requires its exact root chart maps and source trace."
                )
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
                raise ValueError(
                    "Mapped layer source trace changes its source, frame, or chart scope."
                )
            for row, chart_row in enumerate(mapped_chart_trace.chart_witness_rows):
                query_row, root_face, root_cell, root_local, _ = mapped_rows[
                    int(chart_row)
                ]
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
                    or _column_identity_row(mapped_chart_trace.column_identity, row)
                    != _column_identity_row(chart_column_identity, int(chart_row))
                    or source_query.chart_row(2, mapped_chart_trace.source_chart_ids[row])
                    != query_row
                ):
                    raise ValueError(
                        "Mapped layer source trace changes exact root, source, or column ancestry."
                    )
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
                or any(
                    not isinstance(row, ImplicitRootChartEvidence) for row in chart_rows
                )
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
                or not np.array_equal(
                    root_witness.entity_global_ids, root_witness.cell_global_ids
                )
                or np.any(root_witness.local_entity_indices != -1)
            ):
                raise ValueError(
                    "Adaptive implicit evidence requires its current source-wall root chart authority."
                )
            _implicit_root_witness_rows(
                root_witness,
                chart_witness,
                tuple(int(value) for value in chart_column_identity.source_wall_cell_ids),
            )
            root_vertex_ids = set(int(value) for value in root_cell_vertices.reshape(-1))
            if any(
                not set(column_roots) <= root_vertex_ids
                for column_roots in chart_column_identity.source_column_root_vertex_ids
            ):
                raise ValueError(
                    "Implicit chart column ancestry changed its source-wall root cell."
                )
            standard_chart = (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
            if tuple(
                root_witness.source_parameter_chart(row)
                for row in range(len(root_witness.source_chart_ids))
            ) != tuple(standard_chart for _ in root_witness.source_chart_ids):
                raise ValueError(
                    "Adaptive root witness does not retain canonical source parameters."
                )
        if (fidelity.source_id, fidelity.source_revision) != (
            source_query.source_id,
            source_query.source_revision,
        ):
            raise ValueError("Layer fidelity and source query change source identity.")
        if not isinstance(fidelity.witness, SourceBoundaryChartWitness):
            raise ValueError(
                "Accepted layer source fidelity must retain an exact source chart witness."
            )
        if (
            chart_witness.source_id != source_query.source_id
            or chart_witness.source_revision != source_query.source_revision
            or chart_witness.witness_id != fidelity.witness.witness_id
        ):
            raise ValueError(
                "Layer chart witness differs from its continuous source-fidelity proof."
            )
        if not layer_result_id:
            raise ValueError(
                "Layer source evidence requires the immutable layer result identity."
            )
        if not np.isfinite(minimum_scaled_jacobian) or not np.isfinite(
            maximum_distortion
        ):
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

    @classmethod
    def from_stored_fields(
        cls,
        stored: LayerCurvingSourceEvidence,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        source: object,
        restored_query: SourceBoundaryQuery,
        restored_charts: SurfaceSourceCharts | None,
        /,
        *,
        layer_control: BoundaryLayerControl | None = None,
        layer_policy: BoundaryLayerPolicy | None = None,
    ) -> LayerCurvingSourceEvidence:
        """Reprove actual evidence from independently restored scientific owners."""
        if type(source) is BoundaryLayerMesh:
            if layer_control is None or layer_policy is None:
                raise TypeError("Standalone evidence requires its authored layer controls.")
            evidence, _, _, _ = cls._restore_standalone(
                stored, mesh, geometry, source, restored_query, restored_charts,
                layer_control, layer_policy,
            )
            return evidence
        layers, charts = _reprove_layer_evidence_owner(
            source, restored_query, restored_charts, layer_control, layer_policy
        )
        if charts is not None:
            return cls._reprove_parametric(stored, mesh, geometry, layers, restored_query, charts)
        return _reprove_root_chart(cls, stored, mesh, geometry, layers, restored_query)

    @classmethod
    def _restore_standalone(
        cls,
        stored: LayerCurvingSourceEvidence,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        layers: BoundaryLayerMesh,
        query: SourceBoundaryQuery,
        charts: SurfaceSourceCharts | None,
        control: BoundaryLayerControl,
        policy: BoundaryLayerPolicy,
        /,
    ) -> tuple[
        LayerCurvingSourceEvidence, BoundaryLayerMesh, BoundaryLayerControl,
        BoundaryLayerPolicy,
    ]:
        """Restore standalone authored operands and evidence in one proof pass."""
        from ._model_codec import restore_boundary_layer_state

        layers, control, policy = restore_boundary_layer_state(layers, control, policy)
        if charts is not None:
            charts.require_current()
            charts.require_root(layers.source_wall, charts.geometry)
            evidence = cls._reprove_parametric(stored, mesh, geometry, layers, query, charts)
        else:
            evidence = _reprove_root_chart(cls, stored, mesh, geometry, layers, query)
        return evidence, layers, control, policy

    @classmethod
    def _reprove_parametric(
        cls,
        stored: LayerCurvingSourceEvidence,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        layers: BoundaryLayerMesh,
        restored_query: SourceBoundaryQuery,
        charts: SurfaceSourceCharts,
        /,
    ) -> LayerCurvingSourceEvidence:
        """Reconstruct and reprove actual parametric evidence from restored owners."""
        from ._curving import _distortion_blocks, _sampled_quality
        from ._model_codec import _same_model_content
        from ._layer_curving import _rebind_chart_witness_geometry
        if type(stored) is not cls:
            raise TypeError("Parametric layer evidence requires its canonical retained record.")
        if (
            stored.source_surface_charts is None
            or charts is None
            or stored.projection_trace is not None
            or stored.implicit_root_trace is not None
            or stored.mapped_chart_trace is not None
            or stored.mapped_reference_maps
            or stored.implicit_map_witness_id is not None
            or stored.implicit_chart_evidence
            or stored.implicit_root_chart_witness is not None
            or stored.implicit_root_cell_vertex_global_ids is not None
        ):
            raise ValueError("Stored evidence is not an exact parametric layer-chart record.")
        if (
            stored.layer_result_id != layers.result_id
            or not _same_tree_content(stored.layer_evidence, layers.evidence)
        ):
            raise ValueError("Stored layer evidence changes its actual source atlas or layer owner.")
        old_columns = stored.chart_column_identity
        columns = LayerColumnProjectionIdentity(
            old_columns.layer_indices,
            old_columns.source_wall_cell_ids,
            old_columns.front_row_ids,
            old_columns.front_row_identities,
            old_columns.source_column_ids,
            old_columns.source_column_identities,
            old_columns.source_column_root_vertex_ids,
            allow_unbound=old_columns.allow_unbound,
        )
        if not _same_tree_content(columns, old_columns):
            raise ValueError("Stored layer column ancestry does not reproduce its fields.")
        query = parametric_wall_query(
            charts, tuple(int(value) for value in columns.source_wall_cell_ids)
        )
        if not _same_tree_content(restored_query, query):
            raise ValueError("Stored layer query changes its independently selected source patches.")
        witness = _rebind_chart_witness_geometry(stored.chart_witness, geometry)
        if not _same_tree_content(witness, stored.chart_witness):
            raise ValueError("Stored layer source chart changes its actual target geometry.")
        fidelity = certify_source_fidelity(
            mesh,
            geometry,
            query,
            tolerance=stored.fidelity.tolerance,
            maximum_normal_deviation=stored.fidelity.maximum_normal_deviation,
            witness=witness,
        )
        if not _same_model_content(fidelity, stored.fidelity):
            raise ValueError("Stored layer source fidelity does not reproduce its exact proof.")
        elements, _, _ = geometry.resolve(mesh)
        if not elements:
            raise ValueError("Layer evidence requires actual target coordinate elements.")
        degree = max(int(getattr(element, "degree", 1)) for element in elements)
        minimum, maximum = _sampled_quality(
            _distortion_blocks(mesh, geometry, degree),
            np.asarray(geometry.coordinates, dtype=np.float64),
        )
        evidence = cls(
            charts,
            query,
            layers.result_id,
            layers.evidence,
            witness,
            fidelity,
            minimum,
            maximum,
            chart_column_identity=columns,
            frame_id=stored.frame_id,
        )
        if not _same_layer_evidence_fields(evidence, stored):
            raise ValueError("Stored parametric layer evidence does not reproduce its actual fields.")
        evidence.revalidate(mesh, geometry)
        return evidence

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
        """Require this computed evidence to answer ``mesh`` and ``geometry``.

        Nothing is recomputed: source revision, frame and operand identities
        must be the bound ones, and the retained fidelity proof must be the one
        these operands determine (that proof validated the chart witness on
        them). :meth:`revalidate` replays the witness, chart and root proofs.
        """
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
            raise ValueError(
                "Layer source evidence is bound to different source or target operands."
            )
        self.fidelity.require_inputs(
            mesh,
            geometry,
            query,
            tolerance=self.fidelity.tolerance,
            maximum_normal_deviation=self.fidelity.maximum_normal_deviation,
            witness=self.fidelity.witness,
        )
        if self.fidelity.status != "certified":
            raise ValueError(
                "Layer source evidence retains an uncertified fidelity proof."
            )

    def _replay_fidelity(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        root_charts: AdaptiveImplicitRootChartMap | None,
        /,
    ) -> None:
        fresh = certify_source_fidelity(
            mesh,
            geometry,
            self.source_query,
            tolerance=self.fidelity.tolerance,
            maximum_normal_deviation=self.fidelity.maximum_normal_deviation,
            witness=self.fidelity.witness,
            root_charts=root_charts,
        )
        if fresh.certificate_id != self.fidelity.certificate_id:
            raise ValueError(
                "Source fidelity proof does not reproduce its current source and map inputs."
            )

    def revalidate(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        """Replay every retained source proof; stored identities are not proof authority."""
        self.require_current(mesh, geometry)
        query = self.source_query
        if self.source_surface_charts is not None or self.mapped_chart_trace is not None:
            # Adaptive implicit evidence replays its fidelity proof below, on the
            # root charts it prepares for its own revalidation.
            self._replay_fidelity(mesh, geometry, None)
        validate_source_fidelity_witness(mesh, geometry, query, self.chart_witness)
        if self.source_surface_charts is not None:
            if self.projection_trace is not None:
                raise ValueError(
                    "Exact parametric chart evidence cannot claim point projection."
                )
            charts = self.source_surface_charts
            charts.require_current()
            charts.require_root(charts.mesh, charts.geometry)
            expected_query = parametric_wall_query(
                charts,
                tuple(int(value) for value in self.chart_column_identity.source_wall_cell_ids),
            )
            if not _same_tree_content(query, expected_query):
                raise ValueError("Layer source query no longer matches its selected atlas patches.")
            source_chart_rows = {
                int(cell_id): row for row, cell_id in enumerate(charts.cell_global_ids)
            }
            source_wall_vertex_ids = {
                int(value) for value in np.asarray(charts.mesh.vertex_global_ids)
            }
            for chart_row, source_cell_id in enumerate(
                self.chart_column_identity.source_wall_cell_ids
            ):
                source_chart_row = source_chart_rows.get(int(source_cell_id))
                if source_chart_row is None:
                    raise ValueError(
                        "Layer column ancestry names an absent source-wall chart cell."
                    )
                patch = int(charts.root_patches[source_chart_row])
                if (
                    self.chart_witness.source_entity_ids[chart_row]
                    != charts.domain.entity_id(2, patch)
                    or self.chart_witness.occurrence_paths[chart_row]
                    != tuple(charts.domain.source_occurrences[2][patch])
                    or not set(
                        self.chart_column_identity.source_column_root_vertex_ids[
                            chart_row
                        ]
                    )
                    <= source_wall_vertex_ids
                ):
                    raise ValueError(
                        "Layer chart changed source entity, occurrence, or root-column identity."
                    )
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
            for cell_row, cell_id in enumerate(
                np.asarray(block.global_ids, dtype=np.int64)
            )
        }
        if self.mapped_chart_trace is not None:
            if (
                not isinstance(query, MappedReferenceBoundarySource)
                or self.source_surface_charts is not None
                or self.implicit_root_trace is not None
                or not self.mapped_reference_maps
            ):
                raise ValueError(
                    "Mapped layer trace lost its exact original-source query or root maps."
                )
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
                raise ValueError(
                    "Mapped layer trace changed its source/frame/root-query identity."
                )
            chart_rows = tuple(int(value) for value in trace.chart_witness_rows)
            if any(
                chart_row < 0 or chart_row >= len(mapped_rows) for chart_row in chart_rows
            ):
                raise ValueError(
                    "Mapped layer trace names an absent target source chart."
                )
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
                    raise ValueError(
                        "Mapped layer trace names an absent target source chart."
                    )
                query_row, root_face, root_cell, root_local, mapping = mapped_rows[
                    chart_row
                ]
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
                    raise ValueError(
                        "Mapped layer trace changed source occurrence or layer-column/root identity."
                    )
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
                    raise ValueError(
                        "Mapped layer trace DOF is outside its exact target chart."
                    )
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
                    or not np.array_equal(
                        source_point.view(np.uint64),
                        trace.source_points[row].view(np.uint64),
                    )
                    or not np.array_equal(
                        tangents.view(np.uint64), trace.tangents[row].view(np.uint64)
                    )
                    or not np.array_equal(
                        normal.view(np.uint64), trace.normals[row].view(np.uint64)
                    )
                    or trace.distance_lower_bounds[row] != distance_lower
                    or trace.distance_upper_bounds[row] != distance_upper
                ):
                    raise ValueError(
                        "Mapped root source chart point or physical derivative evidence does not replay."
                    )
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
            root_scope = getattr(query, "boundary_source_revision", query.source_revision)
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
                or not np.array_equal(
                    root_witness.entity_global_ids, root_witness.cell_global_ids
                )
                or np.any(root_witness.local_entity_indices != -1)
            ):
                raise ValueError(
                    "Implicit layer evidence changed its source-wall root-chart authority."
                )
            root_chart_rows = _implicit_root_witness_rows(
                root_witness,
                self.chart_witness,
                tuple(
                    int(value)
                    for value in self.chart_column_identity.source_wall_cell_ids
                ),
            )
            root_vertex_ids = set(int(value) for value in root_vertices.reshape(-1))
            for row, chart_row in enumerate(trace.chart_rows):
                chart_index = int(chart_row)
                if chart_index < 0 or chart_index >= len(root_chart_rows):
                    raise ValueError(
                        "Implicit trace chart row is absent from the target witness."
                    )
                root_index = root_chart_rows[chart_index]
                if (
                    int(self.chart_witness.cell_global_ids[chart_index])
                    != int(trace.target_cell_global_ids[row])
                    or _column_identity_row(trace.column_identity, row)
                    != _column_identity_row(self.chart_column_identity, chart_index)
                    or trace.source_cell_keys[row]
                    != tuple(
                        int(value)
                        for value in root_witness.implicit_cell_keys[root_index]
                    )
                    or trace.root_edge_keys[row]
                    != tuple(
                        tuple(
                            tuple(int(value) for value in endpoint) for endpoint in edge
                        )
                        for edge in root_witness.implicit_root_edge_keys[root_index]
                    )
                    or not set(
                        self.chart_column_identity.source_column_root_vertex_ids[
                            chart_index
                        ]
                    )
                    <= root_vertex_ids
                ):
                    raise ValueError(
                        "Implicit layer column/root ancestry does not match its retained source chart."
                    )
            if not isinstance(query, SourceBoundaryProjectionQuery):
                raise ValueError(
                    "Adaptive implicit evidence requires its native projection query."
                )
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
                raise ValueError(
                    "Adaptive implicit root-map evidence no longer revalidates."
                )
            # The fidelity proof replays on the root charts just prepared for
            # its own source and controls; they are never prepared twice.
            self._replay_fidelity(mesh, geometry, root_map)
            replayed = root_map.project_targets(
                tuple(int(chart_row) for chart_row in trace.chart_rows),
                trace.target_reference_points,
            )
            for row, point in enumerate(replayed):
                if (
                    point.status != trace.statuses[row]
                    or point.source_cell_key != trace.source_cell_keys[row]
                    or point.root_edge_keys != trace.root_edge_keys[row]
                    or not np.array_equal(point.point, trace.source_points[row])
                    or point.displacement_upper != trace.displacement_upper[row]
                    or point.root_error != trace.root_error[row]
                ):
                    raise ValueError(
                        "Adaptive implicit root lift does not replay for this target map."
                    )
            rows = trace.geometry_node_rows
            target_cells = trace.target_cell_global_ids
            local_dofs = trace.target_local_dofs
            target_points = trace.target_points
            reference_points = trace.target_reference_points
        reference_nodes_by_block = {}
        for row, (node_row, cell_id, local_dof, reference) in enumerate(
            zip(rows, target_cells, local_dofs, reference_points, strict=True)
        ):
            try:
                block_row, cell_row = cell_rows[int(cell_id)]
            except KeyError as error:
                raise ValueError(
                    "Layer source map names an absent target cell."
                ) from error
            reference_nodes = reference_nodes_by_block.get(block_row)
            if reference_nodes is None:
                reference_element = _face_reference_element(
                    elements[block_row], mesh.blocks[block_row].cell_kind
                )
                if not isinstance(reference_element, FiniteElementSpec):
                    raise ValueError(
                        "Layer source map requires a canonical exact nodal element."
                    )
                reference_nodes = lagrange_reference_node_data(reference_element)
                reference_nodes_by_block[block_row] = reference_nodes
            if local_dof < 0 or local_dof >= len(reference_nodes):
                raise ValueError(
                    "Layer source map changed its target reference DOF identity."
                )
            node = reference_nodes[local_dof]
            if node.local_dof != local_dof or node.exact_point is None:
                raise ValueError(
                    "Layer source map target DOF has no canonical exact reference point."
                )
            expected_reference = _exact3(node.exact_point)
            if expected_reference != reference:
                raise ValueError(
                    "Layer source map changed its exact target reference point."
                )
            if node_row >= 0:
                route = np.asarray(routes[block_row], dtype=np.int64)
                if local_dof >= route.shape[1] or int(route[cell_row, local_dof]) != int(
                    node_row
                ):
                    raise ValueError(
                        "Layer source map changed exact cell/dof/node identity."
                    )
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
                raise ValueError(
                    "Layer source map has an invalid virtual geometry-node sentinel."
                )
            if not np.array_equal(
                np.asarray(actual_point, dtype=np.float64).view(np.uint64),
                np.asarray(target_points[row], dtype=np.float64).view(np.uint64),
            ):
                raise ValueError(
                    "Layer source map queries no longer equal accepted map nodes."
                )


__all__ = [
    "LayerColumnProjectionIdentity",
    "LayerCurvingImplicitRootTrace",
    "LayerCurvingMappedChartTrace",
    "LayerCurvingProjectionTrace",
    "LayerCurvingSourceEvidence",
]


def _reprove_layer_evidence_owner(
    owner: object,
    query: SourceBoundaryQuery,
    charts: SurfaceSourceCharts | None,
    control: BoundaryLayerControl | None,
    policy: BoundaryLayerPolicy | None,
    /,
) -> tuple[BoundaryLayerMesh, SurfaceSourceCharts | None]:
    """Reprove the actual native source and its source-chart owner."""
    from .providers._native_sources import NativeLayerCoreSource

    if type(owner) is NativeLayerCoreSource:
        if control is not None or policy is not None:
            raise ValueError("Native layer source already owns its original controls.")
        owner.validate_restored()
        wall_charts = owner.source_wall_charts
        if charts is not None:
            if (
                wall_charts is None
                or owner.mapped_domain is not None
                or not _same_tree_content(charts, wall_charts)
            ):
                raise ValueError("Stored curving atlas changes its actual source-wall owner.")
        elif (
            wall_charts is not None
            or owner.fidelity_source is None
            or not _same_tree_content(query, owner.fidelity_source)
        ):
            raise ValueError("Root-chart evidence changes its actual source query.")
        return owner.layers, wall_charts
    raise TypeError("Layer evidence requires its canonical native source.")


def _same_layer_evidence_fields(
    rebuilt: LayerCurvingSourceEvidence, stored: LayerCurvingSourceEvidence, /
) -> bool:
    # These owners are independently restored from their archive recipes.
    # Implicit rows are independently materialized and compared by the implicit
    # branch, since their portable representation is a primitive mapping.
    excluded = {
        "source_query", "source_surface_charts", "implicit_root_trace",
        "implicit_chart_evidence", "fidelity",
    }
    return all(
        _same_tree_content(getattr(rebuilt, field.name), getattr(stored, field.name))
        for field in fields(LayerCurvingSourceEvidence)
        if field.name not in excluded
    )


def _restored_column_identity(
    value: LayerColumnProjectionIdentity, /
) -> LayerColumnProjectionIdentity:
    if type(value) is not LayerColumnProjectionIdentity:
        raise TypeError("Stored layer column ancestry has the wrong concrete owner.")
    result = LayerColumnProjectionIdentity(
        value.layer_indices, value.source_wall_cell_ids, value.front_row_ids,
        value.front_row_identities, value.source_column_ids,
        value.source_column_identities, value.source_column_root_vertex_ids,
        allow_unbound=value.allow_unbound,
    )
    if not _same_tree_content(result, value):
        raise ValueError("Stored layer column ancestry does not reproduce its fields.")
    return result


def _restored_mapped_trace(
    value: LayerCurvingMappedChartTrace, /
) -> LayerCurvingMappedChartTrace:
    if type(value) is not LayerCurvingMappedChartTrace:
        raise TypeError("Stored mapped trace has the wrong concrete owner.")
    trace = LayerCurvingMappedChartTrace(
        value.source_id, value.source_revision, value.frame_id, value.query_id,
        value.geometry_node_rows, value.target_cell_global_ids, value.target_local_dofs,
        value.chart_witness_rows, value.target_reference_points,
        value.root_reference_points, value.source_parameters, value.source_chart_ids,
        value.source_entity_ids, value.occurrence_paths, value.root_face_global_ids,
        value.root_cell_global_ids, value.root_local_face_indices, value.target_points,
        value.source_points, value.distance_lower_bounds, value.distance_upper_bounds,
        value.normals, value.tangents, _restored_column_identity(value.column_identity),
    )
    if not _same_tree_content(trace, value):
        raise ValueError("Stored mapped trace does not reproduce its fields.")
    return trace


def _restored_implicit_chart_row(value: object, /) -> ImplicitRootChartEvidence:
    if type(value) is ImplicitRootChartEvidence:
        return value
    if not isinstance(value, dict):
        raise TypeError("Stored implicit chart evidence must be a primitive mapping.")
    row: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Implicit chart field names must be strings.")
        row[key] = item
    expected = {
        "row", "target_cell_global_id", "target_entity_global_id",
        "target_local_entity_index", "source_cell_key", "root_edge_keys", "status",
        "field_residual_bound", "target_to_root_lower_bound", "target_to_root_bound",
        "source_normal_deviation_bounds", "chart_jacobian_lower",
        "chart_injectivity_margin", "source_gradient_lower", "detail",
    }
    if set(row) != expected:
        raise TypeError("Stored implicit chart fields differ from the canonical record.")

    def integer(name: str, /) -> int:
        result = row[name]
        if type(result) is not int:
            raise TypeError(f"Implicit chart {name} must be an integer.")
        return result

    def real(name: str, /) -> float:
        result = row[name]
        if type(result) is not float:
            raise TypeError(f"Implicit chart {name} must be a float.")
        return result

    def text(name: str, /) -> str:
        result = row[name]
        if not isinstance(result, str):
            raise TypeError(f"Implicit chart {name} must be text.")
        return result

    def corner(value: object, /) -> tuple[int, int, int]:
        if not isinstance(value, tuple) or len(value) != 3:
            raise TypeError("Implicit root-edge corners must be integer triples.")
        x, y, z = value
        if type(x) is not int or type(y) is not int or type(z) is not int:
            raise TypeError("Implicit root-edge corners must contain exact integers.")
        return x, y, z

    def edge(value: object, /) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("Implicit root edges must retain two endpoint corners.")
        return corner(value[0]), corner(value[1])

    key = row["source_cell_key"]
    if not isinstance(key, tuple) or len(key) != 4:
        raise TypeError("Implicit source cell keys must be integer quadruples.")
    a, b, c, d = key
    if type(a) is not int or type(b) is not int or type(c) is not int or type(d) is not int:
        raise TypeError("Implicit source cell keys must contain exact integers.")
    edges = row["root_edge_keys"]
    if not isinstance(edges, tuple) or len(edges) != 3:
        raise TypeError("Implicit root-edge triangles must retain three edges.")
    normal = row["source_normal_deviation_bounds"]
    if not isinstance(normal, tuple) or len(normal) != 2:
        raise TypeError("Implicit normal deviation bounds must be an interval.")
    lower, upper = normal
    if type(lower) is not float or type(upper) is not float:
        raise TypeError("Implicit normal deviation endpoints must be floats.")
    return ImplicitRootChartEvidence(
        row=integer("row"),
        target_cell_global_id=integer("target_cell_global_id"),
        target_entity_global_id=integer("target_entity_global_id"),
        target_local_entity_index=integer("target_local_entity_index"),
        source_cell_key=(a, b, c, d),
        root_edge_keys=(edge(edges[0]), edge(edges[1]), edge(edges[2])),
        status=text("status"),
        field_residual_bound=real("field_residual_bound"),
        target_to_root_lower_bound=real("target_to_root_lower_bound"),
        target_to_root_bound=real("target_to_root_bound"),
        source_normal_deviation_bounds=(lower, upper),
        chart_jacobian_lower=real("chart_jacobian_lower"),
        chart_injectivity_margin=real("chart_injectivity_margin"),
        source_gradient_lower=real("source_gradient_lower"),
        detail=text("detail"),
    )


def _restored_implicit_trace(
    value: LayerCurvingImplicitRootTrace, /
) -> LayerCurvingImplicitRootTrace:
    if type(value) is not LayerCurvingImplicitRootTrace:
        raise TypeError("Stored implicit trace has the wrong concrete owner.")
    chart_rows = tuple(_restored_implicit_chart_row(row) for row in value.chart_evidence)
    trace = LayerCurvingImplicitRootTrace(
        value.source_id, value.source_revision, value.frame_id, value.map_witness_id,
        value.root_chart_witness_id, value.geometry_node_rows,
        value.target_cell_global_ids, value.target_local_dofs, value.chart_rows,
        value.target_reference_points, value.target_points, value.source_points,
        value.displacement_upper, value.root_error, value.source_cell_keys,
        value.root_edge_keys, value.statuses, chart_rows,
        _restored_column_identity(value.column_identity),
    )
    if any(
        not _same_tree_content(getattr(trace, field.name), getattr(value, field.name))
        for field in fields(LayerCurvingImplicitRootTrace)
        if field.name != "chart_evidence"
    ):
        raise ValueError("Stored implicit trace does not reproduce its fields.")
    return trace


def _restored_root_vertices(
    witness: SourceBoundaryChartWitness, wall: CellMesh, /
) -> NDArray[np.int64]:
    from ._layer_curving import _implicit_root_corner_rows

    vertex_ids = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    cells = {
        int(cell_id): tuple(int(vertex) for vertex in block.vertices[row])
        for block in wall.blocks
        for row, cell_id in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    rows = []
    for row, cell_id in enumerate(witness.cell_global_ids):
        vertices = cells.get(int(cell_id))
        if vertices is None or len(vertices) != 3:
            raise ValueError("Implicit root chart names an absent source-wall triangle.")
        corners = _implicit_root_corner_rows(witness, row, vertices)
        rows.append(tuple(int(vertex_ids[corner]) for corner in corners))
    return np.asarray(rows, dtype=np.int64).reshape((-1, 3))


def _reprove_root_chart(
    cls: type[LayerCurvingSourceEvidence],
    stored: LayerCurvingSourceEvidence,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    layers: BoundaryLayerMesh,
    query: SourceBoundaryQuery,
    /,
) -> LayerCurvingSourceEvidence:
    from ._curving import _distortion_blocks, _sampled_quality
    from ._layer_curving import _rebind_chart_witness_geometry
    from ._model_codec import _same_model_content
    if type(stored) is not cls:
        raise TypeError("Layer evidence requires its canonical retained record.")
    if stored.source_surface_charts is not None:
        raise ValueError("Root-chart evidence cannot replace a retained parametric atlas.")
    if (
        stored.layer_result_id != layers.result_id
        or not _same_tree_content(stored.layer_evidence, layers.evidence)
    ):
        raise ValueError("Root-chart evidence changes its actual source or layer owner.")
    columns = _restored_column_identity(stored.chart_column_identity)
    witness = _rebind_chart_witness_geometry(stored.chart_witness, geometry)
    if not _same_tree_content(witness, stored.chart_witness):
        raise ValueError("Stored source chart changes its actual target geometry.")
    root_map = None
    implicit_trace = None
    mapped_trace = None
    root_witness = None
    root_vertices = None
    implicit_rows: tuple[ImplicitRootChartEvidence, ...] = ()
    if isinstance(query, MappedReferenceBoundarySource):
        if (
            stored.mapped_chart_trace is None
            or stored.implicit_root_trace is not None
            or stored.implicit_map_witness_id is not None
            or stored.implicit_chart_evidence
            or stored.implicit_root_chart_witness is not None
            or stored.implicit_root_cell_vertex_global_ids is not None
            or stored.projection_trace is not None
        ):
            raise ValueError("Mapped evidence requires its actual mapped trace.")
        mapped_trace = _restored_mapped_trace(stored.mapped_chart_trace)
        for mapping in stored.mapped_reference_maps:
            validate_reference_cell_map_witnesses(mapping)
    else:
        from ..geometry.implicit._source_charts import prepare_adaptive_implicit_root_charts
        from .providers._native_layer import _implicit_layer_root_witness

        if (
            not isinstance(query, SourceBoundaryProjectionQuery)
            or stored.implicit_root_trace is None
            or stored.mapped_chart_trace is not None
            or stored.projection_trace is not None
            or stored.mapped_reference_maps
        ):
            raise ValueError("Implicit evidence requires its actual adaptive root trace.")
        adaptive = _adaptive_implicit_source_query(query)
        root_witness = _implicit_layer_root_witness(
            query.source_id, query.source_revision, query, layers.source_wall,
            CellGeometrySpec.affine(layers.source_wall),
        )
        authored_vertices = _restored_root_vertices(root_witness, layers.source_wall)
        stored_vertices = stored.implicit_root_cell_vertex_global_ids
        if (
            not _same_tree_content(root_witness, stored.implicit_root_chart_witness)
            or stored_vertices is None
            or stored_vertices.dtype != np.dtype(np.int64)
            or stored_vertices.shape != authored_vertices.shape
            or sorted(map(tuple, stored_vertices.tolist()))
            != sorted(map(tuple, authored_vertices.tolist()))
        ):
            raise ValueError("Implicit root witness changes its actual source-wall cells.")
        # The bank was emitted in canonical *target-face* order, which need
        # not be the source root-witness row order. Its full actual content is
        # still rederived above; retain its authored row order for identity.
        root_vertices = np.array(stored_vertices, copy=True)
        implicit_trace = _restored_implicit_trace(stored.implicit_root_trace)
        implicit_rows = tuple(_restored_implicit_chart_row(row) for row in stored.implicit_chart_evidence)
        root_map = prepare_adaptive_implicit_root_charts(
            mesh, geometry, adaptive, witness, frame_id=stored.frame_id,
            tolerance=stored.fidelity.tolerance,
            maximum_normal_deviation=stored.fidelity.maximum_normal_deviation,
        )
        if (
            root_map.status != "certified"
            or root_map.witness_id != stored.implicit_map_witness_id
            or tuple(root_map.chart_evidence) != implicit_rows
            or implicit_trace.chart_evidence != implicit_rows
        ):
            raise ValueError("Implicit root charts do not reproduce their actual source proof.")
    fidelity = certify_source_fidelity(
        mesh, geometry, query, tolerance=stored.fidelity.tolerance,
        maximum_normal_deviation=stored.fidelity.maximum_normal_deviation,
        witness=witness, root_charts=root_map,
    )
    if not _same_model_content(fidelity, stored.fidelity):
        raise ValueError("Stored source fidelity does not reproduce its actual proof.")
    elements, _, _ = geometry.resolve(mesh)
    if not elements:
        raise ValueError("Layer evidence requires actual target coordinate elements.")
    degree = max(int(getattr(element, "degree", 1)) for element in elements)
    minimum, maximum = _sampled_quality(
        _distortion_blocks(mesh, geometry, degree),
        np.asarray(geometry.coordinates, dtype=np.float64),
    )
    evidence = cls(
        None, query, layers.result_id, layers.evidence, witness, fidelity,
        minimum, maximum, chart_column_identity=columns, frame_id=stored.frame_id,
        mapped_chart_trace=mapped_trace, mapped_reference_maps=stored.mapped_reference_maps,
        implicit_root_trace=implicit_trace,
        implicit_map_witness_id=stored.implicit_map_witness_id,
        implicit_chart_evidence=implicit_rows,
        implicit_root_chart_witness=root_witness,
        implicit_root_cell_vertex_global_ids=root_vertices,
    )
    if not _same_layer_evidence_fields(evidence, stored):
        raise ValueError("Stored root-chart evidence does not reproduce its actual fields.")
    evidence.revalidate(mesh, geometry)
    return evidence
