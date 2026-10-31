from __future__ import annotations

from fractions import Fraction
from typing import Literal, Protocol, runtime_checkable

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
    CellGeometryElement,
    CellGeometrySpec,
    CellVertexGeometryElement,
    coordinate_effective_basis,
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


def _geometry_map_expressions(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_global_id: int,
    /,
    *,
    resolved: tuple[tuple[CellGeometryElement, ...], tuple[Array, ...]] | None = None,
) -> tuple[Expression, ...]:
    if resolved is None:
        elements, routes, _ = geometry.resolve(mesh)
    else:
        elements, routes = resolved
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        rows = np.flatnonzero(
            np.asarray(block.global_ids, dtype=np.int64) == int(cell_global_id)
        )
        if rows.size == 0:
            continue
        if rows.size != 1 or not isinstance(element, ScalarCoordinateElement):
            raise ValueError(
                "Exact same-domain map proof requires one fixed-family coordinate cell."
            )
        local_row = int(rows[0])
        basis = coordinate_effective_basis(element, cell_row=local_row)
        coefficients = geometry.source_coordinates(np.asarray(route)[local_row])
        if len(basis) != len(coefficients):
            raise ValueError(
                "Coordinate basis and exact source coefficients are misaligned."
            )
        return linear_combinations(
            basis,
            tuple(
                tuple(coefficients[dof][axis] for dof in range(len(basis)))
                for axis in range(mesh.ambient_dimension)
            ),
        )
    raise ValueError(f"Coordinate geometry does not contain cell ID {cell_global_id}.")


def _require_exact_reference_group(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    witnesses: ReferenceCellMapWitnesses,
    /,
    *,
    reverse: bool = False,
) -> None:
    validate_reference_cell_map_witnesses(witnesses)
    shapes = _reference_shape_polynomials(witnesses.target_cell_kind)
    exact = exact_reference_cell_vertices(witnesses)
    target_ids = np.asarray(witnesses.target_cell_ids, dtype=np.int64)
    source_ids = np.asarray(witnesses.source_cell_ids, dtype=np.int64)
    source_resolved = source_geometry.resolve(source_mesh)
    target_resolved = target_geometry.resolve(target_mesh)
    source_cache: dict[int, tuple[Expression, ...]] = {}
    target_cache: dict[int, tuple[Expression, ...]] = {}

    def expressions(
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        identifier: int,
        /,
        *,
        endpoint: str,
    ) -> tuple[Expression, ...]:
        cache = source_cache if endpoint == "source" else target_cache
        value = cache.get(identifier)
        if value is None:
            resolved = source_resolved if endpoint == "source" else target_resolved
            value = _geometry_map_expressions(
                mesh, geometry, identifier, resolved=(resolved[0], resolved[1])
            )
            cache[identifier] = value
        return value

    for row, (target_id, source_id, reference_vertices) in enumerate(
        zip(target_ids, source_ids, exact, strict=True)
    ):
        if reverse:
            input_mesh, input_geometry = source_mesh, source_geometry
            mapped_mesh, mapped_geometry = target_mesh, target_geometry
            input_id, mapped_id = int(target_id), int(source_id)
            input_endpoint, mapped_endpoint = "source", "target"
        else:
            input_mesh, input_geometry = target_mesh, target_geometry
            mapped_mesh, mapped_geometry = source_mesh, source_geometry
            input_id, mapped_id = int(target_id), int(source_id)
            input_endpoint, mapped_endpoint = "target", "source"
        source_dimension = reference_cell_topology(witnesses.source_cell_kind).dimension
        argument_bank = witnesses.source_reference_chart_arguments
        if argument_bank is not None:
            arguments = argument_bank[row]
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
        mapped = tuple(
            compose_expression(expression, arguments)
            for expression in expressions(
                mapped_mesh,
                mapped_geometry,
                mapped_id,
                endpoint=mapped_endpoint,
            )
        )
        actual = expressions(
            input_mesh,
            input_geometry,
            input_id,
            endpoint=input_endpoint,
        )
        if len(mapped) != len(actual) or any(
            not expression_equal(first, second)
            for first, second in zip(mapped, actual, strict=True)
        ):
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
    source_cache: dict[int, tuple[Expression, ...]] = {}
    target_cache: dict[int, tuple[Expression, ...]] = {}
    maximum_displacement = 0.0
    for target_id, (source_id, target_kind, arguments) in sorted(chart_rows.items()):
        if target_kinds.get(target_id) != target_kind or source_id not in source_kinds:
            raise ValueError(
                f"Bounded source reconstruction chart for target cell ID {target_id} is stale."
            )
        if source_id not in source_cache:
            source_cache[source_id] = _geometry_map_expressions(
                source_mesh,
                source_geometry,
                source_id,
                resolved=(source_resolved[0], source_resolved[1]),
            )
        if target_id not in target_cache:
            target_cache[target_id] = _geometry_map_expressions(
                target_mesh,
                target_geometry,
                target_id,
                resolved=(target_resolved[0], target_resolved[1]),
            )
        source_map = tuple(
            compose_expression(expression, arguments)
            for expression in source_cache[source_id]
        )
        target_map = target_cache[target_id]
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
    for block in mesh.blocks:
        if not isinstance(block, CellBlock):
            raise ValueError(
                "Shared-edge continuity requires fixed-family surface cells."
            )
        for row, cell_id_value in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            cell_id = int(cell_id_value)
            expressions = _geometry_map_expressions(
                mesh,
                geometry,
                cell_id,
                resolved=(resolved[0], resolved[1]),
            )
            cell_expressions[cell_id] = expressions
            cell_vertices = np.asarray(block.vertices, dtype=np.int64)[row]
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
        map_count = 0
        seen_target_ids: set[int] = set()
        for group in _reference_map_groups(reference_map):
            validate_reference_cell_map_witnesses(group)
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
                _require_exact_reference_group(
                    source_mesh,
                    source_geometry,
                    target_mesh,
                    target_geometry,
                    group,
                )
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
                validate_reference_cell_map_witnesses(group)
                _require_exact_reference_group(
                    source_mesh, source_geometry, target_mesh, target_geometry, group
                )
        if reference_map is None and coarsening is not None:
            for group in _nested_reference_groups(
                coarsening, source_mesh, target_mesh, reverse=True
            ):
                validate_reference_cell_map_witnesses(group)
                _require_exact_reference_group(
                    source_mesh,
                    source_geometry,
                    target_mesh,
                    target_geometry,
                    group,
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
