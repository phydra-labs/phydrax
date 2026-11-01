from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Literal, NamedTuple, Protocol, runtime_checkable

import equinox as eqx
import numpy as np
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..typing import parse
from ._cell_geometry import (
    _quotient_chart_lift,
    CellGeometryElement,
    CellGeometrySpec,
    CellVertexGeometryElement,
    coordinate_effective_basis,
    coordinate_reference_arguments,
    PerCellRestrictedCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ._cell_geometry_validity import cell_geometry_id
from ._cell_mesh import CellBlock, CellMesh
from ._coordinate_enclosure import (
    add,
    axes,
    BernsteinDomain,
    compose_expression,
    constant,
    ExactScalar,
    Expression,
    expression_equal,
    linear_combinations,
    multiply,
    polynomial_bounds,
    RationalPolynomial,
    scale,
)
from ._nested_reference_types import (
    evaluate_reference_chart_arguments,
    exact_reference_cell_vertices,
    exact_reference_vertices,
    ExactReferenceChartExpression,
    NestedReferenceWitnesses,
    reference_chart_arguments_identity,
    ReferenceCellMapWitnesses,
    validate_nested_reference_witnesses,
    validate_reference_cell_map_witnesses,
)
from ._reference_cell import reference_cell_topology


CellGeometryTransitionKind = Literal[
    "nested_restriction",
    "coarsening_interpolation",
    "parent_restoration",
    "vertex_displacement",
    "source_realization",
    "bounded_reconstruction",
]


@runtime_checkable
class MaterialChartDeformation(Protocol):
    """Validated physical material correspondence over two geometry epochs."""

    source_topology_id: str
    target_topology_id: str
    source_geometry_id: str
    target_geometry_id: str
    deformation_id: str
    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    domain_id: str
    source_domain_coverage: str
    target_domain_coverage: str
    maximum_displacement_bound: float

    def require_bound(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
    ) -> None: ...


class CellGeometryTransitionEvidence(StrictModule, NonTrainableState):
    """Measured and validated findings for one coordinate-map transition."""

    kind: str = eqx.field(static=True)
    reference_map_count: int = eqx.field(static=True)
    refinement_map_count: int = eqx.field(static=True)
    coarsening_map_count: int = eqx.field(static=True)
    continuity_residual: float = eqx.field(static=True)
    approximation_bound: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: CellGeometryTransitionKind,
        /,
        *,
        reference_map_count: int = 0,
        refinement_map_count: int = 0,
        coarsening_map_count: int = 0,
        continuity_residual: float = 0.0,
        approximation_bound: float = 0.0,
    ) -> None:
        values = (reference_map_count, refinement_map_count, coarsening_map_count)
        residuals = (float(continuity_residual), float(approximation_bound))
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in values
        ):
            raise ValueError(
                "Geometry transition map counts must be nonnegative integers."
            )
        kind = parse(kind, CellGeometryTransitionKind, "kind")
        if any(not np.isfinite(value) or value < 0.0 for value in residuals):
            raise ValueError(
                "Geometry transition error bounds must be finite and nonnegative."
            )
        self.kind = kind
        self.reference_map_count = int(reference_map_count)
        self.refinement_map_count = int(refinement_map_count)
        self.coarsening_map_count = int(coarsening_map_count)
        self.continuity_residual = residuals[0]
        self.approximation_bound = residuals[1]
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "cell-geometry-transition-evidence",
                "transition_kind": self.kind,
                "reference_maps": self.reference_map_count,
                "refinement_maps": self.refinement_map_count,
                "coarsening_maps": self.coarsening_map_count,
                "continuity_residual": self.continuity_residual,
                "approximation_bound": self.approximation_bound,
            }
        )


def _cell_kinds(mesh: CellMesh) -> dict[int, str]:
    result: dict[int, str] = {}
    for block in mesh.blocks:
        for identifier in np.asarray(block.global_ids, dtype=np.int64).tolist():
            result[int(identifier)] = block.cell_kind
    return result


def _require_geometry_realizes_mesh(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> None:
    elements, routes, coefficients = geometry.resolve(mesh)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    coefficient_values = np.asarray(coefficients, dtype=np.float64)
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        cells = np.asarray(block.vertices, dtype=np.int64)
        if isinstance(element, CellVertexGeometryElement):
            local = coefficient_values[np.asarray(route, dtype=np.int64)]
            expected = points[cells]
            valid = np.asarray(block.vertex_valid, dtype=np.bool_)
            if not np.array_equal(local[valid], expected[valid]):
                raise ValueError(
                    "Variable-topology coordinate elements must realize mesh vertices."
                )
            continue
        reference = reference_cell_topology(block.cell_kind).vertices
        expected = points[cells]
        if isinstance(
            element,
            PerCellRestrictedCellGeometryElement
            | RestrictedCellGeometryElement
            | PolynomialComposedCellGeometryElement
            | RationalComposedCellGeometryElement,
        ):
            # Binary64 tabulation of a composed chart rounds every basis term; the
            # realized corner is the correctly rounded exact image of the chart.
            exact_corners = tuple(
                tuple(Fraction(float(value)) for value in point) for point in reference
            )
            images = geometry.evaluate_exact(
                mesh,
                np.asarray(block.global_ids, dtype=np.int64),
                (exact_corners,) * block.cell_count,
            )
            actual = np.asarray(
                [
                    [[float(value) for value in point] for point in cell]
                    for cell in images
                ],
                dtype=np.float64,
            ).reshape(expected.shape)
        else:
            reference_rows = np.broadcast_to(
                np.asarray(reference, dtype=np.float64),
                (block.cell_count, len(reference), len(reference[0])),
            )
            actual = geometry.evaluate(mesh, np.asarray(block.global_ids), reference_rows)
        if not np.array_equal(actual, expected):
            raise ValueError(
                "Coordinate element corner images must equal mesh physical vertices."
            )


def _reference_map_groups(
    value: ReferenceCellMapWitnesses | tuple[ReferenceCellMapWitnesses, ...] | None,
    /,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    if value is None:
        return ()
    groups = (value,) if isinstance(value, ReferenceCellMapWitnesses) else value
    if (
        not isinstance(groups, tuple)
        or not groups
        or any(not isinstance(group, ReferenceCellMapWitnesses) for group in groups)
    ):
        raise TypeError(
            "reference_map must be one witness or a nonempty tuple of witnesses."
        )
    return groups


def _reference_shape_polynomials(cell_kind: str, /) -> tuple[Expression, ...]:
    topology = reference_cell_topology(cell_kind)
    dimension = topology.dimension
    variables = axes(dimension)
    one = constant(Fraction(1), dimension)
    if cell_kind in ("interval", "triangle", "tetrahedron") or cell_kind.startswith(
        "simplex:"
    ):
        first = one
        for variable in variables:
            first = add(first, scale(variable, -1))
        return (first, *variables)
    if cell_kind in ("quadrilateral", "hexahedron", "tensor:1", "tensor:2", "tensor:3"):
        weights = []
        for vertex in topology.vertices:
            value = one
            for axis, variable in enumerate(variables):
                factor = variable if int(vertex[axis]) else add(one, scale(variable, -1))
                value = multiply(value, factor)
            weights.append(value)
        return tuple(weights)
    if cell_kind == "prism":
        x, y, z = variables
        barycentric = (
            add(add(one, scale(x, -1)), scale(y, -1)),
            x,
            y,
        )
        lower = add(one, scale(z, -1))
        return tuple(
            multiply(barycentric[index % 3], z if index >= 3 else lower)
            for index in range(6)
        )
    if cell_kind == "pyramid":
        x, y, z = variables
        lower = add(one, scale(z, -1))
        left = add(one, scale(x, -1))
        back = add(one, scale(y, -1))
        return (
            multiply(multiply(lower, left), back),
            multiply(multiply(lower, x), back),
            multiply(multiply(lower, x), y),
            multiply(multiply(lower, left), y),
            z,
        )
    raise ValueError(f"Exact reference-map proof does not support {cell_kind!r}.")


type _CellMapParts = tuple[
    tuple[Expression, ...] | None,
    tuple[Expression, ...],
    tuple[tuple[ExactScalar, ...], ...],
]


class _ExactCellMaps(NamedTuple):
    """Memoized exact physical maps of one resolved coordinate geometry, by cell ID.

    ``expressions`` composes each cell's basis into its physical map;
    ``parts`` returns the uncomposed ``(chart arguments or None, basis,
    coefficients)`` whose value at ``x`` is ``sum_i c_i b_i(chart(x))``.
    """

    expressions: Callable[[int], tuple[Expression, ...]]
    parts: Callable[[int], _CellMapParts]


def _geometry_map_expressions(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    resolved: tuple[tuple[CellGeometryElement, ...], tuple[Array, ...]],
    /,
) -> _ExactCellMaps:
    """Exact physical maps by cell ID, memoized over one resolved map.

    The quotient-chart lift, exact source bank, host route tables and cell rows
    are prepared once, so each cell costs only its own basis work.
    """
    elements, routes = resolved
    lift = _quotient_chart_lift(geometry, mesh, routes)
    source_bank = geometry.source_coordinates() if lift is None else ()
    route_tables = tuple(np.asarray(route, dtype=np.int64) for route in routes)
    # A cell resolves in the first block naming it; a repeated ID there is ambiguous.
    owners: dict[int, tuple[int, int]] = {}
    repeated: set[int] = set()
    for block_index, block in enumerate(mesh.blocks):
        identifiers = np.asarray(block.global_ids, dtype=np.int64).tolist()
        first_rows: dict[int, int] = {}
        for local_row, identifier in enumerate(identifiers):
            if identifier in owners:
                continue
            if identifier in first_rows:
                repeated.add(identifier)
            else:
                first_rows[identifier] = local_row
        owners.update(
            (identifier, (block_index, local_row))
            for identifier, local_row in first_rows.items()
        )

    def cell_row(
        identifier: int, /
    ) -> tuple[ScalarCoordinateElement, int, int, tuple[tuple[ExactScalar, ...], ...]]:
        owner = owners.get(identifier)
        if owner is None:
            raise ValueError(
                f"Coordinate geometry does not contain cell ID {identifier}."
            )
        block_index, local_row = owner
        element = elements[block_index]
        if identifier in repeated or not isinstance(element, ScalarCoordinateElement):
            raise ValueError(
                "Exact same-domain map proof requires one fixed-family coordinate cell."
            )
        route_rows = route_tables[block_index][local_row]
        if lift is None:
            if np.any(route_rows < 0) or np.any(route_rows >= len(source_bank)):
                raise IndexError(
                    "Coordinate source index lies outside the coefficient bank."
                )
            coefficients = tuple(source_bank[row] for row in route_rows.tolist())
        else:
            coefficients = lift(route_rows)
        return element, block_index, local_row, coefficients

    expression_cache: dict[int, tuple[Expression, ...]] = {}

    def expressions(cell_global_id: int, /) -> tuple[Expression, ...]:
        identifier = int(cell_global_id)
        value = expression_cache.get(identifier)
        if value is not None:
            return value
        element, _, local_row, coefficients = cell_row(identifier)
        basis = coordinate_effective_basis(element, cell_row=local_row)
        if len(basis) != len(coefficients):
            raise ValueError(
                "Coordinate basis and exact source coefficients are misaligned."
            )
        value = linear_combinations(
            basis,
            tuple(
                tuple(coefficients[dof][axis] for dof in range(len(basis)))
                for axis in range(mesh.ambient_dimension)
            ),
        )
        expression_cache[identifier] = value
        return value

    # Root (composed elements) or own (other elements) bases depend only on the block.
    block_bases: dict[int, tuple[Expression, ...]] = {}

    def parts(cell_global_id: int, /) -> _CellMapParts:
        element, block_index, local_row, coefficients = cell_row(int(cell_global_id))
        arguments: tuple[Expression, ...] | None
        root: ScalarCoordinateElement = element
        if isinstance(element, PerCellRestrictedCellGeometryElement):
            arguments = tuple(element.chart_arguments(local_row))
            root = element.source_element
        elif isinstance(
            element,
            RestrictedCellGeometryElement
            | PolynomialComposedCellGeometryElement
            | RationalComposedCellGeometryElement,
        ):
            chart = coordinate_reference_arguments(element, cell_row=local_row)
            if chart is None:
                raise ValueError(
                    "Composed coordinate element must retain its exact chart."
                )
            arguments = tuple(chart)
            root = element.source_element
        else:
            arguments = None
        basis = block_bases.get(block_index)
        if basis is None:
            basis = (
                coordinate_effective_basis(element, cell_row=local_row)
                if arguments is None
                else coordinate_effective_basis(root)
            )
            block_bases[block_index] = basis
        if len(basis) != len(coefficients):
            raise ValueError(
                "Coordinate basis and exact source coefficients are misaligned."
            )
        return arguments, basis, coefficients

    return _ExactCellMaps(expressions, parts)


type _RationalData = dict[tuple[int, ...], Fraction]
type _RationalMap = tuple[
    tuple[_RationalData, ...] | None,
    tuple[_RationalData, ...],
    tuple[tuple[Fraction, ...], ...],
]


def _rational_data(
    expression: Mapping[tuple[int, ...], ExactScalar] | RationalPolynomial, /
) -> _RationalData | None:
    """A polynomial with rational coefficients, or ``None`` for any other datum."""
    if isinstance(expression, RationalPolynomial):
        return None
    data = {
        exponent: coefficient
        for exponent, coefficient in expression.items()
        if isinstance(coefficient, Fraction)
    }
    return data if len(data) == len(expression) else None


def _rational_polynomials(
    expressions: Sequence[Mapping[tuple[int, ...], ExactScalar] | RationalPolynomial],
    /,
) -> tuple[_RationalData, ...] | None:
    result = []
    for expression in expressions:
        data = _rational_data(expression)
        if data is None:
            return None
        result.append(data)
    return tuple(result)


def _rational_map(parts: _CellMapParts, /) -> _RationalMap | None:
    chart, basis, coefficients = parts
    rational_chart = None if chart is None else _rational_polynomials(chart)
    rational_basis = _rational_polynomials(basis)
    rows = tuple(
        tuple(value for value in row if isinstance(value, Fraction))
        for row in coefficients
    )
    if (
        (chart is not None and rational_chart is None)
        or rational_basis is None
        or any(len(row) != len(source) for row, source in zip(rows, coefficients))
    ):
        return None
    return rational_chart, rational_basis, rows


def _polynomial_degree(expression: _RationalData, /) -> int:
    return max((sum(exponent) for exponent in expression), default=0)


def _polynomial_value(
    expression: _RationalData, point: tuple[Fraction, ...], /
) -> Fraction:
    total = Fraction(0)
    for exponent, coefficient in expression.items():
        term = coefficient
        for coordinate, power in zip(point, exponent, strict=True):
            if power:
                term *= coordinate**power
        total += term
    return total


def _unisolvent_lattice(
    cell_kind: str, degree: int, /
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Reference points on which polynomials of total degree ``degree`` agree iff equal.

    The principal simplex lattice is unisolvent for total degree on simplices;
    the tensor grid (and the simplex-by-interval grid on prisms) is unisolvent
    for a larger space containing it.
    """
    order = max(degree, 1)
    dimension = reference_cell_topology(cell_kind).dimension
    if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        indices = (
            index
            for index in itertools.product(range(order + 1), repeat=dimension)
            if sum(index) <= order
        )
    elif cell_kind in ("interval", "quadrilateral", "hexahedron") or cell_kind.startswith(
        "tensor:"
    ):
        indices = itertools.product(range(order + 1), repeat=dimension)
    elif cell_kind == "prism":
        indices = (
            index
            for index in itertools.product(range(order + 1), repeat=3)
            if index[0] + index[1] <= order
        )
    else:
        return None
    return tuple(tuple(Fraction(value, order) for value in index) for index in indices)


def _lattice_map_equality(
    input_parts: _CellMapParts,
    mapped_parts: _CellMapParts,
    arguments: Sequence[Mapping[tuple[int, ...], ExactScalar] | RationalPolynomial],
    cell_kind: str,
    /,
) -> bool | None:
    """Decide ``input map == mapped map o arguments`` exactly without composing.

    Both sides are polynomials of total degree at most the composed degree
    bound, so they are identical iff they agree on a unisolvent lattice of that
    order. ``None`` when a chart, basis or coefficient is not a rational
    polynomial datum; the symbolic proof decides those maps.
    """
    input_map = _rational_map(input_parts)
    mapped_map = _rational_map(mapped_parts)
    reference_chart = _rational_polynomials(arguments)
    if input_map is None or mapped_map is None or reference_chart is None:
        return None

    def chart_degree(chart: tuple[_RationalData, ...] | None, /) -> int:
        if chart is None:
            return 1
        return max((_polynomial_degree(value) for value in chart), default=0)

    degree = max(
        chart_degree(input_map[1]) * chart_degree(input_map[0]),
        chart_degree(mapped_map[1])
        * chart_degree(mapped_map[0])
        * chart_degree(reference_chart),
    )
    lattice = _unisolvent_lattice(cell_kind, degree)
    if lattice is None:
        return None

    def value(
        parts: _RationalMap, point: tuple[Fraction, ...], /
    ) -> tuple[Fraction, ...]:
        chart, basis, coefficients = parts
        image = (
            point
            if chart is None
            else tuple(_polynomial_value(component, point) for component in chart)
        )
        weights = tuple(_polynomial_value(function, image) for function in basis)
        return tuple(
            sum(
                (coefficients[dof][axis] * weights[dof] for dof in range(len(weights))),
                Fraction(0),
            )
            for axis in range(len(coefficients[0]))
        )

    return all(
        value(input_map, point)
        == value(
            mapped_map,
            tuple(_polynomial_value(component, point) for component in reference_chart),
        )
        for point in lattice
    )


def _require_exact_reference_group(
    source_maps: _ExactCellMaps,
    target_maps: _ExactCellMaps,
    witnesses: ReferenceCellMapWitnesses,
    exact: tuple[tuple[tuple[Fraction, ...], ...], ...],
    /,
    *,
    reverse: bool = False,
) -> None:
    """Prove each witnessed reference map composes one endpoint map into the other.

    ``exact`` is the validated exact corner bank of ``witnesses``.
    """
    shapes = _reference_shape_polynomials(witnesses.target_cell_kind)
    target_ids = np.asarray(witnesses.target_cell_ids, dtype=np.int64).tolist()
    source_ids = np.asarray(witnesses.source_cell_ids, dtype=np.int64).tolist()
    if reverse:
        input_maps, mapped_maps = source_maps, target_maps
    else:
        input_maps, mapped_maps = target_maps, source_maps
    source_dimension = reference_cell_topology(witnesses.source_cell_kind).dimension
    argument_bank = witnesses.source_reference_chart_arguments
    for row, (input_id, mapped_id, reference_vertices) in enumerate(
        zip(target_ids, source_ids, exact, strict=True)
    ):
        if argument_bank is not None:
            arguments = tuple(argument_bank[row])
        else:
            arguments = tuple(
                linear_combinations(
                    shapes,
                    (
                        tuple(
                            reference_vertices[corner][axis]
                            for corner in range(len(shapes))
                        ),
                    ),
                )[0]
                for axis in range(source_dimension)
            )
        mapped_parts = mapped_maps.parts(mapped_id)
        input_parts = input_maps.parts(input_id)
        equal = (
            _lattice_map_equality(
                input_parts, mapped_parts, arguments, witnesses.target_cell_kind
            )
            if len(input_parts[2][0]) == len(mapped_parts[2][0])
            else None
        )
        if equal is None:
            mapped = tuple(
                compose_expression(expression, arguments)
                for expression in mapped_maps.expressions(mapped_id)
            )
            actual = input_maps.expressions(input_id)
            equal = len(mapped) == len(actual) and all(
                expression_equal(first, second)
                for first, second in zip(mapped, actual, strict=True)
            )
        if not equal:
            raise ValueError(
                "Exact reference maps do not prove source/target coordinate-map equality."
            )


def _nested_reference_groups(
    witnesses: NestedReferenceWitnesses,
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    /,
    *,
    reverse: bool = False,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    exact = exact_reference_vertices(witnesses)
    fine_ids = np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    coarse_ids = np.asarray(witnesses.coarse_cell_ids, dtype=np.int64)
    source_kinds = _cell_kinds(source_mesh)
    target_kinds = _cell_kinds(target_mesh)
    grouped: dict[
        tuple[str, str],
        list[tuple[int, int, tuple[tuple[Fraction, ...], ...]]],
    ] = {}
    for fine_id, coarse_id, vertices in zip(fine_ids, coarse_ids, exact, strict=True):
        if reverse:
            target_id, source_id = int(fine_id), int(coarse_id)
            target_kind = source_kinds.get(target_id)
            source_kind = target_kinds.get(source_id)
        else:
            target_id, source_id = int(fine_id), int(coarse_id)
            target_kind = target_kinds.get(target_id)
            source_kind = source_kinds.get(source_id)
        if target_kind is None or source_kind is None:
            raise ValueError("Nested reference maps must name endpoint global cell IDs.")
        grouped.setdefault((target_kind, source_kind), []).append(
            (target_id, source_id, vertices)
        )
    results = []
    for (target_kind, source_kind), rows in sorted(grouped.items()):
        rows.sort(key=lambda value: value[0])
        exact_vertices = tuple(value[2] for value in rows)
        execution = np.asarray(
            [
                [[float(value) for value in point] for point in cell]
                for cell in exact_vertices
            ],
            dtype=np.float64,
        )
        results.append(
            ReferenceCellMapWitnesses(
                np.asarray([value[0] for value in rows], dtype=np.int64),
                np.asarray([value[1] for value in rows], dtype=np.int64),
                target_kind,
                source_kind,
                execution,
                exact_vertices,
            )
        )
    return tuple(results)


def _reference_map_identity(
    value: ReferenceCellMapWitnesses | tuple[ReferenceCellMapWitnesses, ...] | None,
    /,
) -> str | None:
    if value is None:
        return None
    groups = _reference_map_groups(value)
    return canonical_fingerprint(
        {
            "kind": "reference-cell-map-witness-groups",
            "maps": tuple(
                {
                    "target_cells": array_tree_fingerprint(group.target_cell_ids),
                    "source_cells": array_tree_fingerprint(group.source_cell_ids),
                    "target_kind": group.target_cell_kind,
                    "source_kind": group.source_cell_kind,
                    "exact_vertices": tuple(
                        tuple(
                            tuple(str(coordinate) for coordinate in point)
                            for point in cell
                        )
                        for cell in exact_reference_cell_vertices(group)
                    ),
                    "chart_arguments": reference_chart_arguments_identity(
                        group.source_reference_chart_arguments
                    ),
                }
                for group in groups
            ),
        }
    )


def _nested_map_identity(value: NestedReferenceWitnesses | None) -> str | None:
    if value is None:
        return None
    exact = exact_reference_vertices(value)
    return canonical_fingerprint(
        {
            "kind": "nested-reference-witnesses",
            "fine_cells": array_tree_fingerprint(value.fine_cell_ids),
            "coarse_cells": array_tree_fingerprint(value.coarse_cell_ids),
            "exact_vertices": tuple(
                tuple(tuple(str(coordinate) for coordinate in point) for point in cell)
                for cell in exact
            ),
        }
    )


type _SurfaceReconstructionChart = tuple[
    int, str, tuple[ExactReferenceChartExpression, ...]
]


def _canonical_surface_reconstruction_maps(
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    value: ReferenceCellMapWitnesses | tuple[ReferenceCellMapWitnesses, ...] | None,
    /,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    if not isinstance(source_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
        raise TypeError("Surface reconstruction endpoints must be CellMesh owners.")
    if (
        source_mesh.topological_dimension != 2
        or target_mesh.topological_dimension != 2
        or source_mesh.ambient_dimension != 3
        or target_mesh.ambient_dimension != 3
        or any(not isinstance(block, CellBlock) for block in source_mesh.blocks)
        or any(not isinstance(block, CellBlock) for block in target_mesh.blocks)
    ):
        raise ValueError(
            "Bounded surface reconstruction requires fixed-family two-dimensional cells in three-space."
        )
    if any(
        block.cell_kind not in ("triangle", "quadrilateral")
        for block in target_mesh.blocks
    ):
        raise ValueError(
            "Bounded surface reconstruction supports triangle and quadrilateral target cells."
        )
    groups = _reference_map_groups(value)
    source_kinds = _cell_kinds(source_mesh)
    target_kinds = _cell_kinds(target_mesh)
    records: dict[
        int,
        tuple[
            str,
            str,
            int,
            tuple[tuple[Fraction, ...], ...],
            tuple[ExactReferenceChartExpression, ...],
        ],
    ] = {}
    grouped: dict[
        tuple[str, str],
        list[
            tuple[
                int,
                int,
                tuple[tuple[Fraction, ...], ...],
                tuple[ExactReferenceChartExpression, ...],
            ]
        ],
    ] = {}
    for group in groups:
        validate_reference_cell_map_witnesses(group)
        arguments = group.source_reference_chart_arguments
        if arguments is None:
            raise ValueError(
                "Bounded surface reconstruction is unresolved without explicit target-to-source reference chart arguments."
            )
        if (
            reference_cell_topology(group.target_cell_kind).dimension != 2
            or reference_cell_topology(group.source_cell_kind).dimension != 2
        ):
            raise ValueError(
                "Bounded surface reconstruction charts must map two-dimensional reference cells."
            )
        target_ids = np.asarray(group.target_cell_ids, dtype=np.int64)
        source_ids = np.asarray(group.source_cell_ids, dtype=np.int64)
        exact = exact_reference_cell_vertices(group)
        for row, (target_id_value, source_id_value, corners, chart) in enumerate(
            zip(target_ids, source_ids, exact, arguments, strict=True)
        ):
            target_id = int(target_id_value)
            source_id = int(source_id_value)
            target_kind = target_kinds.get(target_id)
            source_kind = source_kinds.get(source_id)
            if target_kind is None or source_kind is None:
                raise ValueError(
                    f"Surface reconstruction chart row {row} names an absent endpoint cell."
                )
            if (
                target_kind != group.target_cell_kind
                or source_kind != group.source_cell_kind
            ):
                raise ValueError(
                    f"Surface reconstruction chart row {row} changes an endpoint cell family."
                )
            if target_id in records:
                raise ValueError(
                    f"Surface reconstruction charts duplicate target cell ID {target_id}."
                )
            records[target_id] = (
                group.target_cell_kind,
                group.source_cell_kind,
                source_id,
                corners,
                chart,
            )
            grouped.setdefault(
                (group.target_cell_kind, group.source_cell_kind), []
            ).append((target_id, source_id, corners, chart))
    if set(records) != set(target_kinds):
        missing = sorted(set(target_kinds).difference(records))
        raise ValueError(
            f"Bounded surface reconstruction charts do not cover target cell IDs {missing}."
        )
    canonical: list[ReferenceCellMapWitnesses] = []
    for (target_kind, source_kind), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row[0])
        exact_rows = tuple(row[2] for row in rows)
        execution_rows = np.asarray(
            [
                [[float(value) for value in point] for point in cell]
                for cell in exact_rows
            ],
            dtype=np.float64,
        )
        witness = ReferenceCellMapWitnesses(
            np.asarray([row[0] for row in rows], dtype=np.int64),
            np.asarray([row[1] for row in rows], dtype=np.int64),
            target_kind,
            source_kind,
            execution_rows,
            exact_rows,
            tuple(row[3] for row in rows),
        )
        validate_reference_cell_map_witnesses(witness)
        canonical.append(witness)
    if not canonical:
        raise ValueError("Bounded surface reconstruction needs at least one chart map.")
    return tuple(canonical)


def _surface_reconstruction_rows(
    reference_map: ReferenceCellMapWitnesses | tuple[ReferenceCellMapWitnesses, ...],
    /,
) -> dict[int, _SurfaceReconstructionChart]:
    result: dict[int, _SurfaceReconstructionChart] = {}
    for group in _reference_map_groups(reference_map):
        arguments = group.source_reference_chart_arguments
        if arguments is None:
            raise ValueError(
                "Bounded surface reconstruction is unresolved without explicit target-to-source reference chart arguments."
            )
        for target_id, source_id, chart in zip(
            np.asarray(group.target_cell_ids, dtype=np.int64),
            np.asarray(group.source_cell_ids, dtype=np.int64),
            arguments,
            strict=True,
        ):
            result[int(target_id)] = (
                int(source_id),
                group.target_cell_kind,
                chart,
            )
    return result


def _surface_vector_difference_bound(
    differences: tuple[Expression, ...],
    domain: BernsteinDomain,
    dimension: int,
    evidence_label: str,
    /,
) -> float:
    component_bounds: list[Fraction] = []
    for difference in differences:
        lower, upper = polynomial_bounds(difference, domain, dimension)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(
                f"{evidence_label} is unresolved because no finite outward enclosure is available."
            )
        component_bounds.append(
            Fraction.from_float(max(abs(float(lower)), abs(float(upper))))
        )
    squared_bound = sum(
        (component * component for component in component_bounds), Fraction(0)
    )
    exact_bound = fraction_sqrt_interval(squared_bound)[1]
    try:
        bound = float(exact_bound)
    except OverflowError as error:
        raise ValueError(f"{evidence_label} exceeds finite binary64 range.") from error
    if not np.isfinite(bound):
        raise ValueError(f"{evidence_label} exceeds finite binary64 range.")
    while Fraction.from_float(bound) < exact_bound:
        bound = float(np.nextafter(bound, np.inf))
    return bound


def _reconstructed_surface_endpoints(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_topology: CellMesh,
    reference_map: ReferenceCellMapWitnesses | tuple[ReferenceCellMapWitnesses, ...],
    coordinate_degree: int,
    /,
) -> tuple[CellMesh, CellGeometrySpec]:
    from .fem._reference import lagrange_element

    if not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError(
            "Surface reconstruction source geometry must be CellGeometrySpec."
        )
    if (
        isinstance(coordinate_degree, (bool, np.bool_))
        or not isinstance(coordinate_degree, (int, np.integer))
        or int(coordinate_degree) < 1
    ):
        raise ValueError("coordinate_degree must be a positive integer.")
    chart_rows = _surface_reconstruction_rows(reference_map)
    vertex_values: list[tuple[ExactScalar, ...] | None] = [
        None for _ in range(np.asarray(target_topology.vertex_global_ids).size)
    ]
    nodal_values: dict[tuple[str, int], tuple[tuple[ExactScalar, ...], ...]] = {}
    elements: dict[str, CellGeometryElement] = {}
    vertex_dofs: dict[str, dict[int, int]] = {}
    for block in target_topology.blocks:
        if not isinstance(block, CellBlock):
            raise ValueError("Surface reconstruction requires fixed-family target cells.")
        if block.cell_kind not in ("triangle", "quadrilateral"):
            raise ValueError(
                "Surface reconstruction supports triangle and quadrilateral target cells."
            )
        element = lagrange_element(block.cell_kind, int(coordinate_degree))
        nodes = np.asarray(element.reference_nodes, dtype=np.float64)
        if (
            nodes.ndim != 2
            or nodes.shape != (element.local_dof_count, 2)
            or not np.all(np.isfinite(nodes))
        ):
            raise ValueError(
                "Target coordinate element has unresolved reference-node layout."
            )
        node_by_point = {
            tuple(Fraction(value) for value in point): row
            for row, point in enumerate(nodes.tolist())
        }
        topology = reference_cell_topology(block.cell_kind)
        local_vertex_by_dof: dict[int, int] = {}
        for local_vertex, point in enumerate(topology.vertices):
            key = tuple(Fraction(float(value)) for value in point)
            node = node_by_point.get(key)
            if node is None:
                raise ValueError(
                    "Target coordinate element does not contain every reference-cell vertex."
                )
            local_vertex_by_dof[node] = local_vertex
        vertex_dofs[block.name] = local_vertex_by_dof
        elements[block.name] = element
        for cell_row, cell_id_value in enumerate(
            np.asarray(block.global_ids, dtype=np.int64)
        ):
            cell_id = int(cell_id_value)
            chart_row = chart_rows.get(cell_id)
            if chart_row is None or chart_row[1] != block.cell_kind:
                raise ValueError(
                    f"Surface reconstruction has no matching explicit chart for target cell ID {cell_id}."
                )
            source_id, _, arguments = chart_row
            mapped_nodes = tuple(
                evaluate_reference_chart_arguments(
                    block.cell_kind,
                    arguments,
                    tuple(Fraction(float(value)) for value in point),
                )
                for point in nodes
            )
            exact_values = source_geometry.evaluate_exact(
                source_mesh,
                np.asarray((source_id,), dtype=np.int64),
                (mapped_nodes,),
            )[0]
            nodal_values[(block.name, cell_row)] = exact_values
            cell_vertices = np.asarray(block.vertices, dtype=np.int64)[cell_row]
            for dof, local_vertex in local_vertex_by_dof.items():
                vertex_row = int(cell_vertices[local_vertex])
                physical = exact_values[dof]
                previous = vertex_values[vertex_row]
                if previous is None:
                    vertex_values[vertex_row] = physical
                elif previous != physical:
                    vertex_id = int(
                        np.asarray(target_topology.vertex_global_ids, dtype=np.int64)[
                            vertex_row
                        ]
                    )
                    raise ValueError(
                        f"Explicit source charts disagree at shared target vertex ID {vertex_id}."
                    )
    coordinates = np.empty(
        (
            len(vertex_values),
            target_topology.ambient_dimension,
        ),
        dtype=np.float64,
    )
    for row, exact_values in enumerate(vertex_values):
        if exact_values is None:
            vertex_id = int(
                np.asarray(target_topology.vertex_global_ids, dtype=np.int64)[row]
            )
            raise ValueError(
                f"Surface reconstruction has no source chart for target vertex ID {vertex_id}."
            )
        coordinates[row] = tuple(float(value) for value in exact_values)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Reconstructed source-chart vertex coordinates are not finite.")
    entity_ids = {
        entity.intrinsic_dimension: np.asarray(entity.entity_ids, dtype=np.int64)
        for entity in target_topology.topology.entity_sets
        if entity.intrinsic_dimension > 0
    }
    target_mesh = CellMesh(
        coordinates,
        target_topology.blocks,
        vertex_global_ids=np.asarray(target_topology.vertex_global_ids, dtype=np.int64),
        entity_global_ids=entity_ids,
        numeric_version=target_topology.numeric_version,
        periodic_topology=target_topology.periodic_topology,
    )
    coordinate_rows = coordinates.tolist()
    routes: dict[str, np.ndarray] = {}
    for block in sorted(target_mesh.blocks, key=lambda item: item.name):
        if not isinstance(block, CellBlock):
            raise ValueError("Surface reconstruction requires fixed-family target cells.")
        element = elements[block.name]
        local_vertex_by_dof = vertex_dofs[block.name]
        route = np.empty((block.cell_count, element.local_dof_count), dtype=np.int32)
        row_order = np.argsort(
            np.asarray(block.global_ids, dtype=np.int64), kind="stable"
        )
        for row_value in row_order:
            row = int(row_value)
            cell_vertices = np.asarray(block.vertices, dtype=np.int64)[row]
            values = nodal_values[(block.name, row)]
            for dof, value in enumerate(values):
                local_vertex = local_vertex_by_dof.get(dof)
                if local_vertex is not None:
                    route[row, dof] = int(cell_vertices[local_vertex])
                    continue
                physical = [float(component) for component in value]
                if not np.all(np.isfinite(physical)):
                    raise ValueError(
                        f"Source chart coordinates are not finite on target cell ID {int(np.asarray(block.global_ids)[row])}."
                    )
                route[row, dof] = len(coordinate_rows)
                coordinate_rows.append(physical)
        routes[block.name] = route
    geometry = CellGeometrySpec(
        elements,
        routes,
        np.asarray(coordinate_rows, dtype=np.float64),
    )
    return target_mesh, geometry


def _surface_reconstruction_bounds(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    reference_map: ReferenceCellMapWitnesses | tuple[ReferenceCellMapWitnesses, ...],
    /,
) -> tuple[float, float]:
    chart_rows = _surface_reconstruction_rows(reference_map)
    source_kinds = _cell_kinds(source_mesh)
    target_kinds = _cell_kinds(target_mesh)
    source_resolved = source_geometry.resolve(source_mesh)
    target_resolved = target_geometry.resolve(target_mesh)
    source_expressions = _geometry_map_expressions(
        source_mesh, source_geometry, (source_resolved[0], source_resolved[1])
    )
    target_expressions = _geometry_map_expressions(
        target_mesh, target_geometry, (target_resolved[0], target_resolved[1])
    )
    maximum_displacement = 0.0
    for target_id, (source_id, target_kind, arguments) in sorted(chart_rows.items()):
        if target_kinds.get(target_id) != target_kind or source_id not in source_kinds:
            raise ValueError(
                f"Bounded source reconstruction chart for target cell ID {target_id} is stale."
            )
        source_map = tuple(
            compose_expression(expression, arguments)
            for expression in source_expressions.expressions(source_id)
        )
        target_map = target_expressions.expressions(target_id)
        if len(source_map) != len(target_map):
            raise ValueError(
                f"Source and target physical dimensions disagree on target cell ID {target_id}."
            )
        differences = tuple(
            add(source_component, scale(target_component, -1))
            for source_component, target_component in zip(
                source_map, target_map, strict=True
            )
        )
        domain: BernsteinDomain = "simplex" if target_kind == "triangle" else "box"
        displacement = _surface_vector_difference_bound(
            differences,
            domain,
            2,
            f"Source-target displacement on target cell ID {target_id}",
        )
        maximum_displacement = max(maximum_displacement, displacement)
    maximum_continuity = _surface_shared_edge_continuity_bound(
        target_mesh, target_geometry, target_resolved
    )
    return maximum_displacement, maximum_continuity


def _surface_edge_reference_arguments(
    cell_kind: str,
    first_vertex: int,
    second_vertex: int,
    /,
) -> tuple[Expression, ...]:
    topology = reference_cell_topology(cell_kind)
    first = topology.vertices[first_vertex]
    second = topology.vertices[second_vertex]
    parameter = axes(1)[0]
    return tuple(
        add(
            constant(Fraction(float(first[axis])), 1),
            scale(
                parameter,
                Fraction(float(second[axis])) - Fraction(float(first[axis])),
            ),
        )
        for axis in range(2)
    )


def _surface_shared_edge_continuity_bound(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    resolved: tuple[tuple[CellGeometryElement, ...], tuple[Array, ...], Array],
    /,
) -> float:
    edge_uses: dict[tuple[int, int], list[tuple[int, tuple[Expression, ...]]]] = {}
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_expressions: dict[int, tuple[Expression, ...]] = {}
    expressions_by_cell = _geometry_map_expressions(
        mesh, geometry, (resolved[0], resolved[1])
    )
    for block in mesh.blocks:
        if not isinstance(block, CellBlock):
            raise ValueError(
                "Shared-edge continuity requires fixed-family surface cells."
            )
        cells = np.asarray(block.vertices, dtype=np.int64)
        for row, cell_id_value in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            cell_id = int(cell_id_value)
            cell_expressions[cell_id] = expressions_by_cell.expressions(cell_id)
            cell_vertices = cells[row]
            topology = reference_cell_topology(block.cell_kind)
            for edge in topology.entities[1]:
                first_local, second_local = int(edge[0]), int(edge[1])
                first_id = int(vertex_ids[int(cell_vertices[first_local])])
                second_id = int(vertex_ids[int(cell_vertices[second_local])])
                if first_id <= second_id:
                    ordered_local = (first_local, second_local)
                    key = (first_id, second_id)
                else:
                    ordered_local = (second_local, first_local)
                    key = (second_id, first_id)
                arguments = _surface_edge_reference_arguments(
                    block.cell_kind, ordered_local[0], ordered_local[1]
                )
                edge_uses.setdefault(key, []).append((cell_id, arguments))
    maximum = 0.0
    for key, uses in sorted(edge_uses.items()):
        if len(uses) < 2:
            continue
        uses.sort(key=lambda use: use[0])
        for first in range(len(uses)):
            for second in range(first + 1, len(uses)):
                first_id, first_arguments = uses[first]
                second_id, second_arguments = uses[second]
                first_trace = tuple(
                    compose_expression(component, first_arguments)
                    for component in cell_expressions[first_id]
                )
                second_trace = tuple(
                    compose_expression(component, second_arguments)
                    for component in cell_expressions[second_id]
                )
                if len(first_trace) != len(second_trace):
                    raise ValueError(
                        f"Shared edge {key} has inconsistent ambient coordinate dimensions."
                    )
                differences = tuple(
                    add(first_component, scale(second_component, -1))
                    for first_component, second_component in zip(
                        first_trace, second_trace, strict=True
                    )
                )
                residual = _surface_vector_difference_bound(
                    differences,
                    "box",
                    1,
                    f"Shared-edge continuity for target cell IDs {first_id} and {second_id}",
                )
                maximum = max(maximum, residual)
    return maximum


class CellGeometryTransition(StrictModule, NonTrainableState):
    """Endpoint-bound exact coordinate ancestry and bounded surface reconstruction.

    Physical material correspondence is a separate `MaterialChartDeformation`;
    a reference map alone never implies conservation or same-domain equivalence.
    """

    source_mesh: CellMesh
    target_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_geometry: CellGeometrySpec
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    reference_map: (
        ReferenceCellMapWitnesses | tuple[ReferenceCellMapWitnesses, ...] | None
    )
    refinement: NestedReferenceWitnesses | None
    coarsening: NestedReferenceWitnesses | None
    reference_map_id: str | None = eqx.field(static=True)
    refinement_id: str | None = eqx.field(static=True)
    coarsening_id: str | None = eqx.field(static=True)
    evidence: CellGeometryTransitionEvidence
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
        *,
        evidence: CellGeometryTransitionEvidence,
        reference_map: ReferenceCellMapWitnesses
        | tuple[ReferenceCellMapWitnesses, ...]
        | None = None,
        refinement: NestedReferenceWitnesses | None = None,
        coarsening: NestedReferenceWitnesses | None = None,
    ) -> None:
        if not isinstance(source_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
            raise TypeError("Geometry transition endpoints must be CellMesh owners.")
        if not isinstance(source_geometry, CellGeometrySpec) or not isinstance(
            target_geometry, CellGeometrySpec
        ):
            raise TypeError("Geometry transition endpoints need CellGeometrySpec maps.")
        if source_mesh.topological_dimension != target_mesh.topological_dimension:
            raise ValueError(
                "CellGeometryTransition is same-domain; dimensional construction ancestry is separate."
            )
        if not isinstance(evidence, CellGeometryTransitionEvidence):
            raise TypeError("evidence must be CellGeometryTransitionEvidence.")
        bounded_reconstruction = evidence.kind == "bounded_reconstruction"
        if bounded_reconstruction:
            if refinement is not None or coarsening is not None:
                raise ValueError(
                    "Bounded surface reconstruction does not accept nested ancestry witnesses."
                )
            reference_map = _canonical_surface_reconstruction_maps(
                source_mesh, target_mesh, reference_map
            )
        _require_geometry_realizes_mesh(source_mesh, source_geometry)
        _require_geometry_realizes_mesh(target_mesh, target_geometry)
        source_kinds, target_kinds = _cell_kinds(source_mesh), _cell_kinds(target_mesh)
        source_resolved = source_geometry.resolve(source_mesh)
        target_resolved = target_geometry.resolve(target_mesh)
        source_maps = _geometry_map_expressions(
            source_mesh, source_geometry, (source_resolved[0], source_resolved[1])
        )
        target_maps = _geometry_map_expressions(
            target_mesh, target_geometry, (target_resolved[0], target_resolved[1])
        )
        map_count = 0
        seen_target_ids: set[int] = set()
        for group in _reference_map_groups(reference_map):
            exact = exact_reference_cell_vertices(group)
            source_ids = np.asarray(group.source_cell_ids, dtype=np.int64)
            target_ids = np.asarray(group.target_cell_ids, dtype=np.int64)
            if any(
                int(identifier) not in source_kinds for identifier in source_ids
            ) or any(int(identifier) not in target_kinds for identifier in target_ids):
                raise ValueError(
                    "Reference maps must bind present source and target cells."
                )
            if (
                any(
                    source_kinds[int(identifier)] != group.source_cell_kind
                    for identifier in source_ids
                )
                or any(
                    target_kinds[int(identifier)] != group.target_cell_kind
                    for identifier in target_ids
                )
                or seen_target_ids.intersection(map(int, target_ids))
            ):
                raise ValueError(
                    "Reference maps must match unique endpoint cell families."
                )
            seen_target_ids.update(map(int, target_ids))
            if not bounded_reconstruction:
                _require_exact_reference_group(source_maps, target_maps, group, exact)
            map_count += target_ids.size
        if refinement is not None:
            validate_nested_reference_witnesses(refinement)
        if coarsening is not None:
            validate_nested_reference_witnesses(coarsening)
        if refinement is not None:
            fine_ids = np.asarray(refinement.fine_cell_ids, dtype=np.int64)
            coarse_ids = np.asarray(refinement.coarse_cell_ids, dtype=np.int64)
            if not set(fine_ids.tolist()) <= set(_cell_kinds(target_mesh)) or not set(
                coarse_ids.tolist()
            ) <= set(_cell_kinds(source_mesh)):
                raise ValueError(
                    "Refinement witnesses must bind target fine/source coarse cells."
                )
        if coarsening is not None:
            fine_ids = np.asarray(coarsening.fine_cell_ids, dtype=np.int64)
            coarse_ids = np.asarray(coarsening.coarse_cell_ids, dtype=np.int64)
            if not set(fine_ids.tolist()) <= set(_cell_kinds(source_mesh)) or not set(
                coarse_ids.tolist()
            ) <= set(_cell_kinds(target_mesh)):
                raise ValueError(
                    "Coarsening witnesses must bind source fine/target coarse cells."
                )
        if reference_map is not None and refinement is not None:
            parent_by_target = {
                int(target_id): int(source_id)
                for group in _reference_map_groups(reference_map)
                for target_id, source_id in zip(
                    np.asarray(group.target_cell_ids, dtype=np.int64),
                    np.asarray(group.source_cell_ids, dtype=np.int64),
                    strict=True,
                )
            }
            if any(
                parent_by_target.get(int(fine_id)) != int(coarse_id)
                for fine_id, coarse_id in zip(
                    refinement.fine_cell_ids,
                    refinement.coarse_cell_ids,
                    strict=True,
                )
            ):
                raise ValueError(
                    "Refinement and reference maps disagree on stable cell parents."
                )
        if reference_map is None and refinement is not None:
            for group in _nested_reference_groups(refinement, source_mesh, target_mesh):
                _require_exact_reference_group(
                    source_maps,
                    target_maps,
                    group,
                    exact_reference_cell_vertices(group),
                )
        if reference_map is None and coarsening is not None:
            for group in _nested_reference_groups(
                coarsening, source_mesh, target_mesh, reverse=True
            ):
                _require_exact_reference_group(
                    source_maps,
                    target_maps,
                    group,
                    exact_reference_cell_vertices(group),
                    reverse=True,
                )
        if bounded_reconstruction:
            if reference_map is None:
                raise ValueError(
                    "Bounded surface reconstruction requires retained explicit chart maps."
                )
            displacement, continuity = _surface_reconstruction_bounds(
                source_mesh,
                source_geometry,
                target_mesh,
                target_geometry,
                reference_map,
            )
            if (
                evidence.approximation_bound != displacement
                or evidence.continuity_residual != continuity
            ):
                raise ValueError(
                    "Bounded reconstruction evidence differs from its retained endpoint-map bounds."
                )
        reference_map_id = _reference_map_identity(reference_map)
        refinement_id = _nested_map_identity(refinement)
        coarsening_id = _nested_map_identity(coarsening)
        if (
            evidence.reference_map_count != map_count
            or evidence.refinement_map_count
            != (0 if refinement is None else len(refinement.fine_cell_ids))
            or evidence.coarsening_map_count
            != (0 if coarsening is None else len(coarsening.fine_cell_ids))
        ):
            raise ValueError(
                "Geometry transition evidence counts must match retained maps."
            )
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.source_geometry = source_geometry
        self.target_geometry = target_geometry
        self.source_topology_id = source_mesh.topology_id
        self.target_topology_id = target_mesh.topology_id
        self.source_geometry_id = cell_geometry_id(source_geometry)
        self.target_geometry_id = cell_geometry_id(target_geometry)
        self.reference_map = reference_map
        self.refinement = refinement
        self.coarsening = coarsening
        self.reference_map_id = reference_map_id
        self.refinement_id = refinement_id
        self.coarsening_id = coarsening_id
        self.evidence = evidence
        self.transition_id = canonical_fingerprint(
            {
                "kind": "cell-geometry-transition",
                "source_mesh": source_mesh.mesh_id,
                "source_topology": source_mesh.topology_id,
                "source_geometry": self.source_geometry_id,
                "target_mesh": target_mesh.mesh_id,
                "target_topology": target_mesh.topology_id,
                "target_geometry": self.target_geometry_id,
                "reference_map": reference_map_id,
                "refinement": refinement_id,
                "coarsening": coarsening_id,
                "evidence": evidence.evidence_id,
            }
        )

    @classmethod
    def reconstruct_surface(
        cls,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_topology: CellMesh,
        /,
        *,
        reference_map: ReferenceCellMapWitnesses | tuple[ReferenceCellMapWitnesses, ...],
        maximum_displacement: float,
        maximum_continuity_residual: float,
        coordinate_degree: int = 1,
    ) -> CellGeometryTransition:
        """Reconstruct target surface maps from explicit source charts.

        The input mesh contributes its stable topology and cell/vertex identities;
        its coordinates are replaced by samples of the explicit source geometry.
        The retained transition certifies the requested displacement and
        shared-edge continuity bounds.
        """
        maps = _canonical_surface_reconstruction_maps(
            source_mesh, target_topology, reference_map
        )
        displacement_limit = finite_real_scalar(
            maximum_displacement, "maximum_displacement"
        )
        continuity_limit = finite_real_scalar(
            maximum_continuity_residual, "maximum_continuity_residual"
        )
        if displacement_limit < 0.0 or continuity_limit < 0.0:
            raise ValueError("Surface reconstruction bounds must be nonnegative.")
        target_mesh, target_geometry = _reconstructed_surface_endpoints(
            source_mesh,
            source_geometry,
            target_topology,
            maps,
            coordinate_degree,
        )
        displacement, continuity = _surface_reconstruction_bounds(
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            maps,
        )
        if displacement > displacement_limit:
            raise ValueError(
                f"Certified source-target displacement bound {displacement} exceeds requested maximum_displacement {displacement_limit}."
            )
        if continuity > continuity_limit:
            raise ValueError(
                f"Certified shared-edge continuity residual {continuity} exceeds requested maximum_continuity_residual {continuity_limit}."
            )
        evidence = CellGeometryTransitionEvidence(
            "bounded_reconstruction",
            reference_map_count=sum(
                np.asarray(group.target_cell_ids).size for group in maps
            ),
            continuity_residual=continuity,
            approximation_bound=displacement,
        )
        return cls(
            source_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            evidence=evidence,
            reference_map=maps,
        )

    def require_bound(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
    ) -> None:
        if (
            source_mesh.mesh_id != self.source_mesh.mesh_id
            or source_mesh.topology_id != self.source_topology_id
            or cell_geometry_id(source_geometry) != self.source_geometry_id
            or target_mesh.mesh_id != self.target_mesh.mesh_id
            or target_mesh.topology_id != self.target_topology_id
            or cell_geometry_id(target_geometry) != self.target_geometry_id
        ):
            raise ValueError("Cell geometry transition is stale for these endpoint maps.")
        if (
            _reference_map_identity(self.reference_map) != self.reference_map_id
            or _nested_map_identity(self.refinement) != self.refinement_id
            or _nested_map_identity(self.coarsening) != self.coarsening_id
        ):
            raise ValueError("Retained exact reference-map witnesses have changed.")
        # Exact nested equality or bounded reconstruction evidence is established
        # in construction and cold reconstruction. Ordinary consumers authenticate
        # endpoint/witness identity without replaying symbolic composition.

    @classmethod
    def from_stored_fields(
        cls, value: CellGeometryTransition, /
    ) -> CellGeometryTransition:
        """Reconstruct and re-prove retained transition evidence from stored fields."""
        if not isinstance(value, CellGeometryTransition):
            raise TypeError("value must be CellGeometryTransition.")
        reconstructed = cls(
            value.source_mesh,
            value.source_geometry,
            value.target_mesh,
            value.target_geometry,
            evidence=value.evidence,
            reference_map=value.reference_map,
            refinement=value.refinement,
            coarsening=value.coarsening,
        )
        if reconstructed.transition_id != value.transition_id:
            raise ValueError("Stored geometry transition identity does not reproduce.")
        return reconstructed


__all__ = [
    "CellGeometryTransition",
    "CellGeometryTransitionEvidence",
    "CellGeometryTransitionKind",
    "MaterialChartDeformation",
    "cell_geometry_id",
]
