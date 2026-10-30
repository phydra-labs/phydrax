from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Sequence, SupportsFloat, TYPE_CHECKING

import numpy as np

from ._coordinate_enclosure import (
    add,
    affine_map,
    axes,
    compose_expression,
    constant,
    CoordinateCoefficients,
    Expression,
    expression_equal,
    multiply,
    normalize_pyramid_reference_expression,
    RationalPolynomial,
    scale,
)
from ._exact_algebraic_geometry import ExactScalar


if TYPE_CHECKING:
    from ..geometry._mapped_reference_domain import MappedReferenceDomain
    from ._cell_geometry import CellGeometrySpec, ScalarCoordinateElement
    from ._cell_mesh import CellMesh
    from ._reference_cell import ReferenceCellTopology

from ._nested_reference_types import (
    evaluate_reference_chart_arguments,
    exact_reference_cell_vertices,
    exact_reference_vertices,
    ExactReferenceChartExpression,
    freeze_reference_chart_arguments,
    NestedReferenceWitnesses,
    reference_cell_strictly_contains,
    reference_chart_arguments_contain_source_cell,
    reference_chart_arguments_have_nonvanishing_denominators,
    ReferenceCellMapWitnesses,
    validate_nested_reference_witnesses,
    validate_reference_cell_map_witnesses,
)


def _fraction(value: Fraction | SupportsFloat, /) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction.from_float(float(value))


@dataclass(frozen=True, slots=True)
class NestedSimplexMap:
    """Exact affine target-reference to source-reference map for one simplex."""

    fine_cell_id: int
    coarse_cell_id: int
    vertices: tuple[tuple[Fraction, ...], ...]

    def __post_init__(self) -> None:
        dimension = len(self.vertices[0]) if self.vertices else 0
        if (
            self.fine_cell_id < 0
            or self.coarse_cell_id < 0
            or dimension < 1
            or len(self.vertices) != dimension + 1
            or any(len(vertex) != dimension for vertex in self.vertices)
            or not all(isinstance(value, Fraction) for vertex in self.vertices for value in vertex)
        ):
            raise ValueError("A nested simplex map needs exact, aligned reference corners.")
        if any(
            any(value < 0 for value in (*vertex, Fraction(1) - sum(vertex)))
            for vertex in self.vertices
        ):
            raise ValueError("Nested reference corners must remain inside the parent simplex.")

    @property
    def dimension(self) -> int:
        return len(self.vertices[0])

    @property
    def offset(self) -> tuple[Fraction, ...]:
        return self.vertices[0]

    @property
    def matrix(self) -> tuple[tuple[Fraction, ...], ...]:
        return tuple(
            tuple(self.vertices[column + 1][row] - self.vertices[0][row] for column in range(self.dimension))
            for row in range(self.dimension)
        )

    def evaluate(
        self, point: Sequence[Fraction | SupportsFloat], /
    ) -> tuple[Fraction, ...]:
        coordinate = tuple(_fraction(value) for value in point)
        if len(coordinate) != self.dimension:
            raise ValueError("Reference points must match the nested-map dimension.")
        barycentric = (Fraction(1) - sum(coordinate), *coordinate)
        return tuple(
            sum(
                (barycentric[corner] * self.vertices[corner][axis] for corner in range(self.dimension + 1)),
                Fraction(0),
            )
            for axis in range(self.dimension)
        )


def nested_simplex_maps(
    witnesses: NestedReferenceWitnesses, /
) -> tuple[NestedSimplexMap, ...]:
    validate_nested_reference_witnesses(witnesses)
    exact = exact_reference_vertices(witnesses)
    fine_ids = np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    coarse_ids = np.asarray(witnesses.coarse_cell_ids, dtype=np.int64)
    return tuple(
        NestedSimplexMap(int(fine), int(coarse), vertices)
        for fine, coarse, vertices in zip(fine_ids, coarse_ids, exact, strict=True)
    )


def nested_reference_vertices(
    fine_sources: np.ndarray,
    fine_weights: np.ndarray,
    coarse_rows: np.ndarray,
    /,
) -> np.ndarray:
    """Coarse-simplex reference coordinates of fine vertices from global-ID stencils."""
    sources = np.asarray(fine_sources, dtype=np.int64)
    weights = np.asarray(fine_weights, dtype=np.float64)
    coarse = np.asarray(coarse_rows, dtype=np.int64)
    if (
        sources.ndim != 3
        or weights.shape != sources.shape
        or coarse.ndim != 2
        or sources.shape[:2] != coarse.shape
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError("Fine stencils and coarse simplex rows have incompatible shapes.")
    matches = (sources[..., None] == coarse[:, None, None, :]) & (sources[..., None] >= 0)
    if np.any(np.any(matches, axis=-1) != (sources >= 0)):
        raise ValueError("Every fine stencil source must be a corner of its coarse cell.")
    barycentric = np.sum(np.where(matches, weights[..., None], 0.0), axis=2)
    return barycentric[..., 1:]


def exact_nested_reference_vertices(
    fine_sources: np.ndarray,
    fine_homogeneous: np.ndarray,
    coarse_rows: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate exact binary64 homogeneous rows into coarse barycentrics.

    Returns the round-to-nearest reference execution view and a source-owned
    homogeneous bank over coarse corners. Sums and quotients are formed in exact
    rational arithmetic; no intermediate coordinate is promoted from a rounded
    vertex value.
    """
    sources = np.asarray(fine_sources, dtype=np.int64)
    homogeneous = np.asarray(fine_homogeneous, dtype=np.float64)
    coarse = np.asarray(coarse_rows, dtype=np.int64)
    if (
        sources.ndim != 3
        or homogeneous.shape != sources.shape
        or coarse.ndim != 2
        or sources.shape[:2] != coarse.shape
        or not np.all(np.isfinite(homogeneous))
        or np.any(homogeneous < 0.0)
    ):
        raise ValueError("Homogeneous stencils and coarse simplex rows have incompatible shapes.")
    matches = (sources[..., None] == coarse[:, None, None, :]) & (sources[..., None] >= 0)
    if np.any(np.any(matches, axis=-1) != (sources >= 0)):
        raise ValueError("Every homogeneous stencil source must be a coarse-cell corner.")
    cells, corners, _ = sources.shape
    width = coarse.shape[1]
    result = np.zeros((cells, corners, width), dtype=np.float64)
    bank = np.zeros_like(result)
    for cell in range(cells):
        for vertex in range(corners):
            accumulated = [Fraction(0) for _ in range(width)]
            for source_slot in range(sources.shape[2]):
                source_id = int(sources[cell, vertex, source_slot])
                if source_id < 0:
                    continue
                corner = int(np.flatnonzero(coarse[cell] == source_id)[0])
                accumulated[corner] += Fraction(float(homogeneous[cell, vertex, source_slot]))
            total = sum(accumulated, Fraction(0))
            if total < 0:
                raise ValueError("Nested homogeneous barycentrics must have nonnegative sum.")
            bank[cell, vertex] = [float(value) for value in accumulated]
            if total:
                result[cell, vertex] = [float(value / total) for value in accumulated]
            elif any(accumulated):
                raise ValueError("A nonempty homogeneous row must have positive total weight.")
    return result[..., 1:], bank


def _reference_shape_weights(
    cell_kind: str, point: tuple[Fraction, ...], /
) -> tuple[Fraction, ...]:
    from ._reference_cell import reference_cell_topology

    topology = reference_cell_topology(cell_kind)
    if len(point) != topology.dimension:
        raise ValueError("Target reference coordinates must match their cell dimension.")
    if cell_kind in ("interval", "triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        weights = (Fraction(1) - sum(point), *point)
        if len(weights) != len(topology.vertices):
            raise ValueError("Simplex reference corner count is inconsistent.")
        return weights
    if cell_kind in ("quadrilateral", "hexahedron") or cell_kind.startswith("tensor:"):
        return tuple(
            math.prod(
                (
                    point[axis]
                    if int(vertex[axis])
                    else Fraction(1) - point[axis]
                    for axis in range(topology.dimension)
                ),
                start=Fraction(1),
            )
            for vertex in topology.vertices
        )
    if cell_kind == "prism":
        x, y, z = point
        barycentric = (Fraction(1) - x - y, x, y)
        return tuple(
            barycentric[index % 3] * (z if index >= 3 else Fraction(1) - z)
            for index in range(6)
        )
    if cell_kind == "pyramid":
        x, y, z = point
        if z == 1:
            if (x, y) != (Fraction(1, 2), Fraction(1, 2)):
                raise ValueError("Pyramid apex coordinates must be centered.")
            return (Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1))
        denominator = Fraction(1) - z
        half_z = z * Fraction(1, 2)
        upper_x = x - half_z
        upper_y = y - half_z
        lower_x = Fraction(1) - half_z - x
        lower_y = Fraction(1) - half_z - y
        return (
            lower_x * lower_y / denominator,
            upper_x * lower_y / denominator,
            upper_x * upper_y / denominator,
            lower_x * upper_y / denominator,
            z,
        )
    raise ValueError(f"Reference map does not support {cell_kind!r} target cells.")


@dataclass(frozen=True, slots=True)
class NestedCellReferenceMap:
    """Exact target-cell reference map into one source reference cell."""

    target_cell_id: int
    source_cell_id: int
    target_cell_kind: str
    source_cell_kind: str
    source_reference_vertices: tuple[tuple[Fraction, ...], ...]
    arguments: tuple[ExactReferenceChartExpression, ...] | None = None

    def __post_init__(self) -> None:
        from ._reference_cell import reference_cell_topology

        target_topology = reference_cell_topology(self.target_cell_kind)
        source_topology = reference_cell_topology(self.source_cell_kind)
        if (
            self.target_cell_id < 0
            or self.source_cell_id < 0
            or len(self.source_reference_vertices) != len(target_topology.vertices)
            or any(
                len(point) != source_topology.dimension
                or any(not isinstance(value, Fraction) for value in point)
                for point in self.source_reference_vertices
            )
            or (
                self.arguments is not None
                and len(self.arguments) != source_topology.dimension
            )
        ):
            raise ValueError("Nested reference-cell map records are misaligned.")

    def evaluate(
        self, point: Sequence[Fraction | SupportsFloat], /
    ) -> tuple[ExactScalar, ...]:
        coordinates = tuple(_fraction(value) for value in point)
        if self.arguments is not None:
            return evaluate_reference_chart_arguments(
                self.target_cell_kind, self.arguments, coordinates
            )
        weights = _reference_shape_weights(self.target_cell_kind, coordinates)
        return tuple(
            sum(
                (
                    weights[corner] * self.source_reference_vertices[corner][axis]
                    for corner in range(len(weights))
                ),
                Fraction(0),
            )
            for axis in range(len(self.source_reference_vertices[0]))
        )


def nested_cell_reference_maps(
    witnesses: ReferenceCellMapWitnesses, /
) -> tuple[NestedCellReferenceMap, ...]:
    validate_reference_cell_map_witnesses(witnesses)
    exact = exact_reference_cell_vertices(witnesses)
    targets = np.asarray(witnesses.target_cell_ids, dtype=np.int64)
    sources = np.asarray(witnesses.source_cell_ids, dtype=np.int64)
    argument_rows = witnesses.source_reference_chart_arguments
    return tuple(
        NestedCellReferenceMap(
            int(target),
            int(source),
            witnesses.target_cell_kind,
            witnesses.source_cell_kind,
            vertices,
            freeze_reference_chart_arguments(
                (_chart_from_corners(witnesses.target_cell_kind, vertices),)
            )[0]
            if argument_rows is None
            else argument_rows[row],
        )
        for row, (target, source, vertices) in enumerate(
            zip(targets, sources, exact, strict=True)
        )
    )


def coordinate_reference_arguments(
    witness: ReferenceCellMapWitnesses, target_cell_id: int, /
) -> NestedCellReferenceMap:
    for value in nested_cell_reference_maps(witness):
        if value.target_cell_id == int(target_cell_id):
            return value
    raise KeyError(f"No nested reference map for target cell {target_cell_id}.")



def _reference_map_index_by_intermediate_id(
    witness: ReferenceCellMapWitnesses,
    intermediate_cell_global_ids: Sequence[int] | np.ndarray | None,
    /,
) -> dict[int, NestedCellReferenceMap]:
    maps = nested_cell_reference_maps(witness)
    if intermediate_cell_global_ids is None:
        return {value.target_cell_id: value for value in maps}
    local_ids = np.asarray(witness.target_cell_ids, dtype=np.int64)
    global_ids = np.asarray(intermediate_cell_global_ids)
    if (
        local_ids.shape != (len(maps),)
        or not np.array_equal(local_ids, np.arange(len(maps)))
        or global_ids.ndim != 1
        or global_ids.shape != local_ids.shape
        or global_ids.dtype.kind not in "iu"
        or (global_ids.dtype.kind == "u" and np.any(global_ids > np.iinfo(np.int64).max))
        or np.unique(global_ids).size != global_ids.size
        or np.any(global_ids < 0)
    ):
        raise ValueError("Intermediate global cell IDs must align unique block-local target rows.")
    return {
        int(global_id): reference_map
        for global_id, reference_map in zip(global_ids.tolist(), maps, strict=True)
    }


def compose_reference_cell_maps(
    target_to_intermediate: ReferenceCellMapWitnesses,
    intermediate_to_source: ReferenceCellMapWitnesses,
    /,
    *,
    intermediate_cell_global_ids: Sequence[int] | np.ndarray | None = None,
) -> ReferenceCellMapWitnesses:
    """Compose exact charts by intermediate global IDs, or explicit local-row IDs."""
    validate_reference_cell_map_witnesses(target_to_intermediate)
    validate_reference_cell_map_witnesses(intermediate_to_source)
    if target_to_intermediate.source_cell_kind != intermediate_to_source.target_cell_kind:
        raise ValueError("Reference-map composition requires matching intermediate cell kinds.")
    source_maps = _reference_map_index_by_intermediate_id(
        intermediate_to_source, intermediate_cell_global_ids
    )
    targets = nested_cell_reference_maps(target_to_intermediate)
    target_corners = _fraction_corners(target_to_intermediate.target_cell_kind)
    output_source_ids = []
    output_exact_vertices = []
    output_arguments = []
    for target_map in targets:
        intermediate_map = source_maps.get(target_map.source_cell_id)
        if intermediate_map is None:
            raise ValueError(
                f"Reference-map composition names absent intermediate cell {target_map.source_cell_id}."
            )
        if target_map.arguments is None or intermediate_map.arguments is None:
            raise ValueError("Exact reference-map composition requires complete chart arguments.")
        arguments = tuple(
            compose_expression(expression, target_map.arguments)
            for expression in intermediate_map.arguments
        )
        corners = tuple(
            evaluate_reference_chart_arguments(
                target_to_intermediate.target_cell_kind, arguments, point
            )
            for point in target_corners
        )
        if any(
            not isinstance(value, Fraction)
            for point in corners
            for value in point
        ):
            raise ValueError("Composed reference-map corners must be exact rationals.")
        output_source_ids.append(intermediate_map.source_cell_id)
        output_exact_vertices.append(
            tuple(
                tuple(value for value in point if isinstance(value, Fraction))
                for point in corners
            )
        )
        output_arguments.append(arguments)
    exact_bank = tuple(output_exact_vertices)
    execution = np.asarray(
        [[[float(value) for value in point] for point in cell] for cell in exact_bank],
        dtype=np.float64,
    )
    target_ids = np.asarray(
        [value.target_cell_id for value in targets], dtype=np.int64
    )
    source_ids = np.asarray(output_source_ids, dtype=np.int64)
    for value in (execution, target_ids, source_ids):
        value.setflags(write=False)
    result = ReferenceCellMapWitnesses(
        target_ids,
        source_ids,
        target_to_intermediate.target_cell_kind,
        intermediate_to_source.source_cell_kind,
        execution,
        exact_bank,
        freeze_reference_chart_arguments(tuple(output_arguments)),
    )
    validate_reference_cell_map_witnesses(result)
    return result


def _mesh_cell_kinds(mesh: CellMesh, /) -> dict[int, tuple[str, int, str]]:
    result: dict[int, tuple[str, int, str]] = {}
    for block in mesh.blocks:
        for row, raw_cell_id in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            cell_id = int(raw_cell_id)
            if cell_id in result:
                raise ValueError("Coarsening cell global IDs must be unique across blocks.")
            result[cell_id] = (block.name, row, block.cell_kind)
    return result


def coarsen_reference_cell_maps(
    coarsening: NestedReferenceWitnesses,
    fine_to_root_maps: Mapping[str, ReferenceCellMapWitnesses],
    fine_mesh: CellMesh,
    coarse_mesh: CellMesh,
    /,
) -> Mapping[str, ReferenceCellMapWitnesses]:
    """Reconstruct one exact parent→root chart from complete child ancestry.

    Fine-to-root maps use block-local target rows. Returned maps also use the
    restored coarse block's local row IDs and include only parents named by the
    coarsening witness; callers merge untouched rows before common-root use.
    Every child of a restored parent must identify one root cell and reproduce
    the same exact parent chart.
    """
    from ..linalg._exact import determinant_exact, solve_exact
    from ._cell_mesh import CellMesh
    from ._reference_cell import reference_cell_topology

    if not isinstance(coarsening, NestedReferenceWitnesses):
        raise TypeError("coarsening must be NestedReferenceWitnesses.")
    if not isinstance(fine_mesh, CellMesh) or not isinstance(coarse_mesh, CellMesh):
        raise TypeError("fine_mesh and coarse_mesh must be canonical CellMesh owners.")
    dimension = coarse_mesh.topological_dimension
    if fine_mesh.topological_dimension != dimension:
        raise ValueError("Nested coarsening must preserve mesh dimension.")
    validate_nested_reference_witnesses(coarsening, dimension=dimension)
    if not isinstance(fine_to_root_maps, Mapping) or set(fine_to_root_maps) != {
        block.name for block in fine_mesh.blocks
    }:
        raise ValueError("Fine-to-root maps must name every fine mesh block.")

    fine_kinds = _mesh_cell_kinds(fine_mesh)
    coarse_kinds = _mesh_cell_kinds(coarse_mesh)
    fine_to_root: dict[int, NestedCellReferenceMap] = {}
    for block in fine_mesh.blocks:
        witness = fine_to_root_maps[block.name]
        validate_reference_cell_map_witnesses(witness)
        local_target_ids = np.asarray(witness.target_cell_ids, dtype=np.int64)
        if (
            witness.target_cell_kind != block.cell_kind
            or not np.array_equal(local_target_ids, np.arange(block.cell_count))
            or np.asarray(witness.source_cell_ids).shape != (block.cell_count,)
        ):
            raise ValueError("Fine-to-root witness rows must align block-local fine cells.")
        global_ids = np.asarray(block.global_ids, dtype=np.int64)
        for reference_map in nested_cell_reference_maps(witness):
            fine_id = int(global_ids[reference_map.target_cell_id])
            if fine_id in fine_to_root:
                raise ValueError("Fine-to-root maps assign a fine cell more than once.")
            fine_to_root[fine_id] = reference_map

    parent_charts: dict[
        int,
        tuple[str, int, str, int, str, tuple[Expression, ...]],
    ] = {}
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )
    parent_axes = axes(dimension)
    for child in nested_simplex_maps(coarsening):
        fine_id = child.fine_cell_id
        parent_id = child.coarse_cell_id
        fine_kind_record = fine_kinds.get(fine_id)
        parent_kind_record = coarse_kinds.get(parent_id)
        root_map = fine_to_root.get(fine_id)
        if fine_kind_record is None or parent_kind_record is None or root_map is None:
            raise ValueError("Coarsening ancestry must bind present fine, parent, and root cells.")
        fine_kind = fine_kind_record[2]
        parent_block_name, parent_row, parent_kind = parent_kind_record
        if not (
            fine_kind in ("interval", "triangle", "tetrahedron")
            or fine_kind.startswith("simplex:")
        ) or not (
            parent_kind in ("interval", "triangle", "tetrahedron")
            or parent_kind.startswith("simplex:")
        ):
            raise ValueError("Nested coarsening witnesses require simplex child and parent cells.")
        if (
            reference_cell_topology(fine_kind).dimension != dimension
            or reference_cell_topology(parent_kind).dimension != dimension
            or root_map.target_cell_kind != fine_kind
            or reference_cell_topology(root_map.source_cell_kind).dimension != dimension
            or root_map.arguments is None
        ):
            raise ValueError("Coarsening chart kinds must preserve one exact simplex dimension.")
        matrix = child.matrix
        if determinant_exact(matrix) == 0:
            raise ValueError("A coarsening child-to-parent reference map must be nonsingular.")
        inverse = tuple(tuple(row) for row in solve_exact(matrix, identity))
        child_from_parent = tuple(
            _linear_expression(
                parent_axes,
                tuple(inverse[axis][column] for column in range(dimension)),
                -sum(
                    (
                        inverse[axis][column] * child.offset[column]
                        for column in range(dimension)
                    ),
                    Fraction(0),
                ),
                dimension,
            )
            for axis in range(dimension)
        )
        parent_arguments = tuple(
            compose_expression(expression, child_from_parent)
            for expression in root_map.arguments
        )
        previous = parent_charts.get(parent_id)
        if previous is None:
            parent_charts[parent_id] = (
                parent_block_name,
                parent_row,
                parent_kind,
                root_map.source_cell_id,
                root_map.source_cell_kind,
                parent_arguments,
            )
        elif (
            previous[2] != parent_kind
            or previous[3] != root_map.source_cell_id
            or previous[4] != root_map.source_cell_kind
            or any(
                not expression_equal(first, second)
                for first, second in zip(previous[5], parent_arguments, strict=True)
            )
        ):
            raise ValueError(
                f"Coarsened cell {parent_id} spans multiple root cells or exact root charts."
            )

    grouped: dict[str, list[tuple[int, int, str, tuple[Expression, ...]]]] = {}
    root_kind_by_block: dict[str, str] = {}
    for chart in parent_charts.values():
        block_name, local_row, target_kind, root_id, root_kind, arguments = chart
        previous_kind = root_kind_by_block.setdefault(block_name, root_kind)
        if previous_kind != root_kind:
            raise ValueError(
                f"Coarsened block {block_name} must be split by source root cell kind."
            )
        grouped.setdefault(block_name, []).append(
            (local_row, root_id, root_kind, arguments)
        )

    result: dict[str, ReferenceCellMapWitnesses] = {}
    for block_name, rows in grouped.items():
        block = next(block for block in coarse_mesh.blocks if block.name == block_name)
        rows.sort(key=lambda value: value[0])
        target_ids = np.asarray([value[0] for value in rows], dtype=np.int64)
        source_ids = np.asarray([value[1] for value in rows], dtype=np.int64)
        target_corners = _fraction_corners(block.cell_kind)
        chart_rows = tuple(value[3] for value in rows)
        exact_cells = tuple(
            tuple(
                evaluate_reference_chart_arguments(
                    block.cell_kind, cell_arguments, point
                )
                for point in target_corners
            )
            for cell_arguments in chart_rows
        )
        if any(
            not isinstance(value, Fraction)
            for cell in exact_cells
            for point in cell
            for value in point
        ):
            raise ValueError("Coarsened root-chart corners must be exact rationals.")
        exact_bank = tuple(
            tuple(
                tuple(value for value in point if isinstance(value, Fraction))
                for point in cell
            )
            for cell in exact_cells
        )
        execution = np.asarray(
            [[[float(value) for value in point] for point in cell] for cell in exact_bank],
            dtype=np.float64,
        )
        for value in (execution, target_ids, source_ids):
            value.setflags(write=False)
        witness = ReferenceCellMapWitnesses(
            target_ids,
            source_ids,
            block.cell_kind,
            rows[0][2],
            execution,
            exact_bank,
            freeze_reference_chart_arguments(chart_rows),
        )
        validate_reference_cell_map_witnesses(witness)
        result[block_name] = witness
    return MappingProxyType(result)

@dataclass(frozen=True, slots=True)
class ExactReferenceChartControl:
    """Exact target-reference chart controls and source-root identity."""

    target_block_name: str
    target_block_index: int
    target_cell_row: int
    target_cell_id: int
    target_cell_kind: str
    source_block_name: str
    source_block_index: int
    source_cell_row: int
    source_cell_id: int
    source_cell_kind: str
    source_region: int
    source_region_id: str
    source_id: str
    source_revision: str
    mapped_domain_id: str
    source_geometry_id: str
    source_topology_id: str
    source_occurrence_path: tuple[str, ...]
    source_reference_vertices: tuple[tuple[Fraction, ...], ...]
    arguments: tuple[ExactReferenceChartExpression, ...]

    def _witness(self, target_id: int, /) -> ReferenceCellMapWitnesses:
        points = np.asarray(
            [[tuple(float(value) for value in point) for point in self.source_reference_vertices]],
            dtype=np.float64,
        )
        target_ids = np.asarray((target_id,), dtype=np.int64)
        source_ids = np.asarray((self.source_cell_id,), dtype=np.int64)
        for value in (points, target_ids, source_ids):
            value.setflags(write=False)
        return ReferenceCellMapWitnesses(
            target_ids,
            source_ids,
            self.target_cell_kind,
            self.source_cell_kind,
            points,
            (self.source_reference_vertices,),
            freeze_reference_chart_arguments((self.arguments,)),
        )

    @property
    def witness(self) -> ReferenceCellMapWitnesses:
        """Return this control as a witness keyed by target global cell ID."""
        return self._witness(self.target_cell_id)

    @property
    def restriction_witness(self) -> ReferenceCellMapWitnesses:
        """Return a witness keyed by its local target-block row for restrictions."""
        return self._witness(self.target_cell_row)


def _reference_shape_polynomials(cell_kind: str, /) -> tuple[Expression, ...]:
    from ._reference_cell import reference_cell_topology

    topology = reference_cell_topology(cell_kind)
    dimension = topology.dimension
    variables = axes(dimension)
    one = constant(1, dimension)
    if cell_kind in ("interval", "triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        total: Expression = constant(0, dimension)
        for variable in variables:
            total = add(total, variable)
        return (add(one, scale(total, -1)), *variables)
    if cell_kind in ("quadrilateral", "hexahedron") or cell_kind.startswith("tensor:"):
        weights = []
        for vertex in topology.vertices:
            weight: Expression = constant(1, dimension)
            for axis in range(dimension):
                factor = (
                    variables[axis]
                    if int(vertex[axis])
                    else add(one, scale(variables[axis], -1))
                )
                weight = multiply(weight, factor)
            weights.append(weight)
        return tuple(weights)
    if cell_kind == "prism":
        x, y, z = variables
        lower = add(add(one, scale(x, -1)), scale(y, -1))
        triangle = (lower, x, y)
        lower_z = add(one, scale(z, -1))
        return tuple(
            multiply(triangle[index % 3], z if index >= 3 else lower_z)
            for index in range(6)
        )
    if cell_kind == "pyramid":
        x, y, z = variables
        half_z = scale(z, Fraction(1, 2))
        upper_x = add(x, scale(half_z, -1))
        upper_y = add(y, scale(half_z, -1))
        lower_x = add(add(one, scale(half_z, -1)), scale(x, -1))
        lower_y = add(add(one, scale(half_z, -1)), scale(y, -1))
        denominator = add(one, scale(z, -1))
        return (
            RationalPolynomial(multiply(lower_x, lower_y), denominator),
            RationalPolynomial(multiply(upper_x, lower_y), denominator),
            RationalPolynomial(multiply(upper_x, upper_y), denominator),
            RationalPolynomial(multiply(lower_x, upper_y), denominator),
            z,
        )
    raise ValueError(f"Exact reference charts do not support {cell_kind!r} cells.")


def _chart_from_corners(
    cell_kind: str,
    corners: tuple[tuple[ExactScalar, ...], ...],
    /,
) -> tuple[Expression, ...]:
    from ._reference_cell import reference_cell_topology

    weights = _reference_shape_polynomials(cell_kind)
    if len(corners) != len(weights) or not corners:
        raise ValueError("Reference-chart corner count disagrees with its cell kind.")
    ambient_dimension = len(corners[0])
    if any(len(point) != ambient_dimension for point in corners):
        raise ValueError("Reference-chart corner coordinates must share one dimension.")
    maps: list[Expression] = []
    for axis in range(ambient_dimension):
        coordinate: Expression = constant(
            0, reference_cell_topology(cell_kind).dimension
        )
        for weight, point in zip(weights, corners, strict=True):
            coordinate = add(coordinate, scale(weight, point[axis]))
        maps.append(coordinate)
    return tuple(maps)


def _freeze_expression(
    value: Expression, /
) -> Mapping[tuple[int, ...], ExactScalar] | RationalPolynomial:
    if isinstance(value, RationalPolynomial):
        return RationalPolynomial(dict(value.numerator), dict(value.denominator))
    return MappingProxyType(dict(value))




def _reference_cell_topology_signature(
    topology: ReferenceCellTopology, /
) -> tuple[object, ...]:
    vertices = tuple(sorted(topology.vertices))
    entities = tuple(
        tuple(
            sorted(
                tuple(sorted(topology.vertices[index] for index in entity))
                for entity in level
            )
        )
        for level in topology.entities
    )
    return topology.dimension, vertices, entities


def _canonical_mapped_reference_cell_kind(cell_kind: str, /) -> str | None:
    from ._reference_cell import reference_cell_topology

    actual = _reference_cell_topology_signature(
        reference_cell_topology(cell_kind)
    )
    for canonical_kind in ("tetrahedron", "prism", "hexahedron", "pyramid"):
        canonical = _reference_cell_topology_signature(
            reference_cell_topology(canonical_kind)
        )
        if actual == canonical:
            return canonical_kind
    return None


def _reference_interior_point(cell_kind: str, /) -> tuple[Fraction, ...]:
    match _canonical_mapped_reference_cell_kind(cell_kind):
        case "tetrahedron":
            return (Fraction(1, 4),) * 3
        case "prism":
            return (Fraction(1, 3), Fraction(1, 3), Fraction(1, 2))
        case "hexahedron":
            return (Fraction(1, 2),) * 3
        case "pyramid":
            return (Fraction(1, 2), Fraction(1, 2), Fraction(1, 5))
        case _:
            raise ValueError(f"Exact reference-cell interiors do not support {cell_kind!r}.")


def _fraction_corners(cell_kind: str, /) -> tuple[tuple[Fraction, ...], ...]:
    from ._reference_cell import reference_cell_topology

    return tuple(
        tuple(_fraction(value) for value in point)
        for point in reference_cell_topology(cell_kind).vertices
    )


def _pyramid_affine_chart_from_corners(
    corners: tuple[tuple[Fraction, ...], ...], /
) -> tuple[Expression, ...] | None:
    from ._reference_cell import reference_cell_topology

    topology = reference_cell_topology("pyramid")
    reference_vertices = tuple(
        tuple(_fraction(value) for value in point) for point in topology.vertices
    )
    vertex_rows = {point: row for row, point in enumerate(reference_vertices)}
    try:
        origin_index = vertex_rows[(Fraction(0), Fraction(0), Fraction(0))]
        x_index = vertex_rows[(Fraction(1), Fraction(0), Fraction(0))]
        y_index = vertex_rows[(Fraction(0), Fraction(1), Fraction(0))]
        apex_index = vertex_rows[(Fraction(1, 2), Fraction(1, 2), Fraction(1))]
    except KeyError:
        return None
    if len(corners) != len(reference_vertices) or any(
        len(point) != 3 for point in corners
    ):
        return None
    origin = corners[origin_index]
    x_axis = tuple(
        corners[x_index][axis] - origin[axis] for axis in range(3)
    )
    y_axis = tuple(
        corners[y_index][axis] - origin[axis] for axis in range(3)
    )
    z_axis = tuple(
        corners[apex_index][axis]
        - origin[axis]
        - Fraction(1, 2) * (x_axis[axis] + y_axis[axis])
        for axis in range(3)
    )
    if any(
        tuple(
            origin[axis]
            + reference[0] * x_axis[axis]
            + reference[1] * y_axis[axis]
            + reference[2] * z_axis[axis]
            for axis in range(3)
        )
        != point
        for reference, point in zip(reference_vertices, corners, strict=True)
    ):
        return None
    variables = axes(3)
    return tuple(
        _linear_expression(
            variables,
            (x_axis[axis], y_axis[axis], z_axis[axis]),
            origin[axis],
            3,
        )
        for axis in range(3)
    )


def _root_affine_inverse(
    cell_kind: str,
    corners: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], tuple[tuple[Fraction, ...], ...]]:
    from ..linalg._exact import determinant_exact, solve_exact

    expression = _chart_from_corners(cell_kind, corners)
    affine = affine_map(expression, 3)
    if affine is None and cell_kind == "pyramid":
        affine_expression = _pyramid_affine_chart_from_corners(corners)
        if affine_expression is not None and all(
            expression_equal(actual, expected)
            for actual, expected in zip(expression, affine_expression, strict=True)
        ):
            affine = affine_map(affine_expression, 3)
    if affine is None:
        raise ValueError(
            "Reference-root localization requires an exact affine chart in reference-domain coordinates."
        )
    offset, matrix = affine
    if any(
        not isinstance(value, Fraction)
        for value in (*offset, *(entry for row in matrix for entry in row))
    ):
        raise ValueError("Reference-root chart coefficients must be exact rationals.")
    exact_offset = tuple(value for value in offset if isinstance(value, Fraction))
    exact_matrix = tuple(
        tuple(value for value in row if isinstance(value, Fraction))
        for row in matrix
    )
    if determinant_exact(exact_matrix) <= 0:
        raise ValueError("Reference-root charts must have positive exact orientation.")
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(3))
        for row in range(3)
    )
    inverse_values = tuple(tuple(row) for row in solve_exact(exact_matrix, identity))
    if any(
        not isinstance(value, Fraction)
        for row in inverse_values
        for value in row
    ):
        raise ValueError("Reference-root inverse coefficients must be exact rationals.")
    inverse = tuple(
        tuple(value for value in row if isinstance(value, Fraction))
        for row in inverse_values
    )
    return exact_offset, inverse


def _linear_expression(
    expressions: tuple[Expression, ...],
    coefficients: tuple[Fraction, ...],
    offset: Fraction,
    dimension: int,
    /,
) -> Expression:
    result: Expression = constant(offset, dimension)
    for expression, coefficient in zip(expressions, coefficients, strict=True):
        result = add(result, scale(expression, coefficient))
    return result


@dataclass(frozen=True, slots=True)
class _MappedReferenceRoot:
    block_index: int
    block_name: str
    cell_row: int
    cell_id: int
    cell_kind: str
    region: int
    region_id: str
    occurrence_path: tuple[str, ...]
    offset: tuple[Fraction, ...]
    inverse: tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True, slots=True)
class _ReferenceRootCandidate:
    root: _MappedReferenceRoot
    arguments: tuple[Expression, ...]
    source_reference_vertices: tuple[tuple[Fraction, ...], ...]
    interior_owned: bool


def _validate_reference_chart_target(
    domain: MappedReferenceDomain,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    target_cell_regions: np.ndarray,
    /,
) -> np.ndarray:
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh

    if not isinstance(target_mesh, CellMesh) or not isinstance(
        target_geometry, CellGeometrySpec
    ):
        raise TypeError("target_mesh and target_geometry must be canonical cell owners.")
    if target_mesh.topological_dimension != 3 or target_mesh.ambient_dimension != 3:
        raise ValueError("Exact mapped-root localization requires three-dimensional target cells.")
    regions = np.asarray(target_cell_regions)
    cell_count = sum(block.cell_count for block in target_mesh.blocks)
    if (
        regions.ndim != 1
        or regions.shape != (cell_count,)
        or regions.dtype.kind not in "iu"
        or (regions.dtype.kind == "u" and np.any(regions > np.iinfo(np.int64).max))
    ):
        raise ValueError(
            "target_cell_regions must assign an integer reference region to every target cell."
        )
    regions = regions.astype(np.int64, copy=False)
    if np.any(regions < 0) or np.any(regions >= len(domain.region_ids)):
        raise ValueError("target_cell_regions must name declared mapped reference regions.")
    return regions


def _mapped_reference_roots(
    domain: MappedReferenceDomain, /
) -> tuple[_MappedReferenceRoot, ...]:
    root_mesh = domain.reference_mesh
    root_coordinates = np.asarray(root_mesh.coordinates)
    roots = []
    root_cursor = 0
    root_ids_seen: set[int] = set()
    for block_index, block in enumerate(root_mesh.blocks):
        if _canonical_mapped_reference_cell_kind(block.cell_kind) is None:
            raise ValueError(
                f"Exact mapped-root localization does not support {block.cell_kind!r} reference roots."
            )
        for cell_row, raw_cell_id in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            cell_id = int(raw_cell_id)
            if cell_id in root_ids_seen:
                raise ValueError("Mapped reference-root global IDs must be unique.")
            root_ids_seen.add(cell_id)
            vertex_rows = np.asarray(block.vertices[cell_row], dtype=np.int64)
            corners = tuple(
                tuple(_fraction(value) for value in point)
                for point in root_coordinates[vertex_rows]
            )
            offset, inverse = _root_affine_inverse(block.cell_kind, corners)
            region = int(domain.cell_regions[root_cursor])
            roots.append(
                _MappedReferenceRoot(
                    block_index,
                    block.name,
                    cell_row,
                    cell_id,
                    block.cell_kind,
                    region,
                    domain.region_ids[region],
                    domain.root_cell_occurrence_paths[root_cursor],
                    offset,
                    inverse,
                )
            )
            root_cursor += 1
    return tuple(roots)


def _exact_degree_one_target_chart(
    element: ScalarCoordinateElement,
    local_coefficients: CoordinateCoefficients,
    target_cell_kind: str,
    target_cell_row: int,
    /,
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Expression, ...]]:
    from ._coordinate_enclosure import coordinate_map_expressions

    reference_corners = _fraction_corners(target_cell_kind)
    expressions = coordinate_map_expressions(
        element, local_coefficients, cell_row=target_cell_row
    )
    if len(expressions) != 3:
        raise ValueError("Target reference charts must have three coordinate components.")
    corner_values = tuple(
        evaluate_reference_chart_arguments(target_cell_kind, expressions, point)
        for point in reference_corners
    )
    if any(
        not isinstance(value, Fraction)
        for point in corner_values
        for value in point
    ):
        raise ValueError("Reference-chart corner controls must be exact rational coordinates.")
    exact_corners = tuple(
        tuple(value for value in point if isinstance(value, Fraction))
        for point in corner_values
    )
    degree_one = _chart_from_corners(target_cell_kind, exact_corners)
    comparison_expressions = expressions
    comparison_degree_one = degree_one
    if target_cell_kind == "pyramid":
        comparison_expressions = tuple(
            normalize_pyramid_reference_expression(value).expression
            for value in expressions
        )
        comparison_degree_one = tuple(
            normalize_pyramid_reference_expression(value).expression
            for value in degree_one
        )
    if not all(
        expression_equal(actual, expected)
        for actual, expected in zip(
            comparison_expressions, comparison_degree_one, strict=True
        )
    ):
        raise ValueError(
            "Target geometry must be an exact degree-one chart in reference-domain coordinates."
        )
    if not reference_chart_arguments_have_nonvanishing_denominators(
        target_cell_kind, expressions
    ):
        raise ValueError(
            "Target reference-chart denominators must be proved nonzero over the whole cell."
        )
    return reference_corners, expressions if target_cell_kind == "pyramid" else degree_one


def _reference_root_candidate(
    root: _MappedReferenceRoot,
    target_cell_kind: str,
    target_reference_corners: tuple[tuple[Fraction, ...], ...],
    target_chart: tuple[Expression, ...],
    /,
) -> _ReferenceRootCandidate | None:
    arguments = tuple(
        _linear_expression(
            target_chart,
            tuple(root.inverse[parameter][axis] for axis in range(3)),
            -sum(
                (
                    root.inverse[parameter][axis] * root.offset[axis]
                    for axis in range(3)
                ),
                Fraction(0),
            ),
            3,
        )
        for parameter in range(3)
    )
    if not reference_chart_arguments_contain_source_cell(
        target_cell_kind, root.cell_kind, arguments
    ):
        return None
    corner_images = tuple(
        evaluate_reference_chart_arguments(target_cell_kind, arguments, point)
        for point in target_reference_corners
    )
    if any(
        not isinstance(value, Fraction)
        for point in corner_images
        for value in point
    ):
        raise ValueError("Exact target-to-root chart controls must be rational.")
    exact_corner_images = tuple(
        tuple(value for value in point if isinstance(value, Fraction))
        for point in corner_images
    )
    interior = _reference_interior_point(target_cell_kind)
    interior_image = evaluate_reference_chart_arguments(
        target_cell_kind, arguments, interior
    )
    interior_owned = reference_cell_strictly_contains(root.cell_kind, interior_image)
    return _ReferenceRootCandidate(
        root, arguments, exact_corner_images, interior_owned
    )


def _unique_reference_root(
    roots: tuple[_MappedReferenceRoot, ...],
    target_cell_id: int,
    target_cell_kind: str,
    target_region: int,
    target_reference_corners: tuple[tuple[Fraction, ...], ...],
    target_chart: tuple[Expression, ...],
    /,
) -> _ReferenceRootCandidate:
    candidates = []
    for root in roots:
        candidate = _reference_root_candidate(
            root,
            target_cell_kind,
            target_reference_corners,
            target_chart,
        )
        if candidate is not None:
            candidates.append(candidate)
    if len(candidates) != 1 or not candidates[0].interior_owned:
        if len(candidates) > 1 or candidates:
            raise ValueError(
                f"Target cell {target_cell_id} has ambiguous shared-boundary or overlapping mapped-root ownership."
            )
        raise ValueError(
            f"Target cell {target_cell_id} is outside or crosses every exact mapped reference root."
        )
    candidate = candidates[0]
    if target_region != candidate.root.region:
        raise ValueError(
            f"Target cell {target_cell_id} region disagrees with its unique source root {candidate.root.cell_id}."
        )
    return candidate


def exact_reference_chart_controls(
    domain: MappedReferenceDomain,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    target_cell_regions: np.ndarray,
    /,
) -> tuple[ExactReferenceChartControl, ...]:
    """Locate exact degree-one target charts in unique mapped reference roots.

    Localization uses only the represented reference_mesh coordinates. It does
    not invert source_geometry's potentially curved or rational physical maps;
    the returned arguments compose with those original maps verbatim.
    A rationally represented target chart is admitted only when exact
    expression equality proves it is the same canonical target-cell
    interpolation, so corner witnesses retain the complete map.
    """
    from ..geometry._mapped_reference_domain import MappedReferenceDomain
    from ._cell_geometry import ScalarCoordinateElement

    if not isinstance(domain, MappedReferenceDomain):
        raise TypeError("domain must be MappedReferenceDomain.")
    target_regions = _validate_reference_chart_target(
        domain, target_mesh, target_geometry, target_cell_regions
    )
    roots = _mapped_reference_roots(domain)
    target_elements, target_routes, _ = target_geometry.resolve(target_mesh)
    target_coefficients = target_geometry.source_coordinates()
    controls: list[ExactReferenceChartControl] = []
    target_ids_seen: set[int] = set()
    target_cursor = 0
    for block_index, (block, element, route) in enumerate(
        zip(target_mesh.blocks, target_elements, target_routes, strict=True)
    ):
        if not isinstance(element, ScalarCoordinateElement):
            raise TypeError(
                "Exact mapped-root localization requires a scalar coordinate element."
            )
        if (
            _canonical_mapped_reference_cell_kind(block.cell_kind) is None
            or element.cell_kind != block.cell_kind
        ):
            raise ValueError(
                f"Exact mapped-root localization does not support target kind {block.cell_kind!r}."
            )
        for target_row, raw_target_id in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            target_cell_id = int(raw_target_id)
            if target_cell_id in target_ids_seen:
                raise ValueError("Target cell global IDs must be unique across blocks.")
            target_ids_seen.add(target_cell_id)
            local_coefficients = tuple(
                target_coefficients[int(index)]
                for index in np.asarray(route[target_row], dtype=np.int64)
            )
            reference_corners, target_chart = _exact_degree_one_target_chart(
                element,
                local_coefficients,
                block.cell_kind,
                target_row,
            )
            target_region = int(target_regions[target_cursor])
            candidate = _unique_reference_root(
                roots,
                target_cell_id,
                block.cell_kind,
                target_region,
                reference_corners,
                target_chart,
            )
            root = candidate.root
            controls.append(
                ExactReferenceChartControl(
                    block.name,
                    block_index,
                    target_row,
                    target_cell_id,
                    block.cell_kind,
                    root.block_name,
                    root.block_index,
                    root.cell_row,
                    root.cell_id,
                    root.cell_kind,
                    root.region,
                    root.region_id,
                    domain.source_id,
                    domain.source_revision,
                    domain.domain_id,
                    domain.reference_geometry_id,
                    domain.reference_mesh.topology_id,
                    root.occurrence_path,
                    candidate.source_reference_vertices,
                    tuple(_freeze_expression(value) for value in candidate.arguments),
                )
            )
            target_cursor += 1
    return tuple(controls)



def reference_chart_controls_by_target_id(
    controls: Sequence[ExactReferenceChartControl], /
) -> Mapping[int, ExactReferenceChartControl]:
    """Build an immutable global-ID lookup without changing control row order."""
    if not isinstance(controls, Sequence) or any(
        not isinstance(value, ExactReferenceChartControl) for value in controls
    ):
        raise TypeError("controls must be a sequence of exact reference chart records.")
    indexed: dict[int, ExactReferenceChartControl] = {}
    for control in controls:
        if control.target_cell_id in indexed:
            raise ValueError("Reference chart controls must have unique target global IDs.")
        indexed[control.target_cell_id] = control
    return MappingProxyType(indexed)

__all__ = [
    "ExactReferenceChartControl",
    "NestedCellReferenceMap",
    "NestedSimplexMap",
    "coarsen_reference_cell_maps",
    "compose_reference_cell_maps",
    "coordinate_reference_arguments",
    "exact_nested_reference_vertices",
    "exact_reference_chart_controls",
    "nested_cell_reference_maps",
    "nested_reference_vertices",
    "nested_simplex_maps",
    "reference_chart_controls_by_target_id",
]
