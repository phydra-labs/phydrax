# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from math import prod
from types import MappingProxyType
from typing import (
    final,
    Literal,
    NamedTuple,
    NoReturn,
    Protocol,
    runtime_checkable,
    SupportsIndex,
    TYPE_CHECKING,
)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike
from scipy.sparse import csr_matrix

from ..._coordinate_measure import CoordinateMeasurePlan
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...ein import contract
from ...exterior._algebra import pullback, vector_to_form
from ...exterior._complex import ComplexBoundary, DiscreteForm
from ...exterior._form_type import FormProxy, FormTwist, FormType
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    ComplexMap,
    HilbertComplex,
    LinearSolvePolicy,
    OperatorProperties,
    SolveResourcePolicy,
    SparseFactorizationPolicy,
)
from ...linalg._algebra_operators import apply_real_map_componentwise
from ...linalg._complexes import (
    coordinate_space,
    hodge_laplacian,
    HodgeLaplacianPart,
)
from ...linalg._exact import matmul_exact, solve_exact
from ...sparse import EdgeRelation, RowRelation, SparseCoordinateOperator
from ...typing import checked, ConvertibleToArray, parse
from .._cell_de_rham import AbstractCellDeRhamComplex
from .._cell_geometry import CellGeometrySpec
from .._cell_geometry_transition_types import CellGeometryTransition
from .._cell_mesh import CellBlock, CellMesh, PolyhedralBlock
from .._cochain_hodge import _admission_policy, SparseHodge
from .._distributed_field import DistributedSparseTransfer
from .._gram import gram_solve_policy
from .._nested_reference import (
    nested_cell_reference_maps,
    nested_simplex_maps,
    NestedCellReferenceMap,
    NestedSimplexMap,
)
from .._nested_reference_types import (
    freeze_reference_chart_arguments,
    NestedReferenceWitnesses,
    reference_chart_arguments_identity,
    ReferenceCellMapWitnesses,
    ReferenceCellPartition,
    validate_reference_cell_map_witnesses,
)
from .._reference_cell import reference_cell_topology
from .._side_actions import SideTraceProvider
from .._topology import CellComplexTopology
from .._views import PreparedFieldReconstruction
from ._cell_map import PreparedFiniteElementCellMap
from ._form_elements import (
    _interior_dof_indices,
    form_element,
    FormBasis,
    FormElementFamily,
)
from ._generic import (
    _exact_prism_canonical_form_cell,
    _form_cell_orientation_sign,
    _linear_reference_element,
    _local_mass_tensor,
    _MappedBasisTable,
    _prism_canonical_form_coordinates,
    _prism_form_entity_chart_vertices,
    FiniteElementBlockGeometry,
    FiniteElementDiscretization,
    FiniteElementDofMap,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from ._point_interpolation import prepare_finite_element_field_reconstruction
from ._reference import FiniteElementSpec


@runtime_checkable
class _CurvedIntegrationPolicy(Protocol):
    relative_tolerance: float
    absolute_tolerance: float
    policy_id: str


if TYPE_CHECKING:
    from ...geometry._common_root_volume_overlap import CurvedOverlapIntegrationPolicy
    from ...linalg._sparse_rank import SparseSupportRank
    from ._exact_form_moments import _PreparedReferenceChartOrientation
    from ._partition_form_moments import PartitionCell
else:
    CurvedOverlapIntegrationPolicy = _CurvedIntegrationPolicy


def _index_nested_reference_maps(
    witnesses: NestedReferenceWitnesses
    | ReferenceCellMapWitnesses
    | Sequence[NestedReferenceWitnesses | ReferenceCellMapWitnesses],
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    /,
    *,
    partition_target_ids: frozenset[int] = frozenset(),
) -> tuple[dict[int, NestedSimplexMap | NestedCellReferenceMap], np.ndarray]:
    """Bind exact local maps to globally concatenated source and target rows."""
    source_cell_rows: dict[int, int] = {}
    source_cell_kinds: dict[int, str] = {}
    target_cell_rows: dict[int, int] = {}
    target_cell_kinds: dict[int, str] = {}
    source_offset = 0
    for block in source_mesh.blocks:
        for local, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            cell_id = int(identifier)
            if cell_id in source_cell_rows:
                raise ValueError("Source global cell IDs must be unique across blocks.")
            source_cell_rows[cell_id] = source_offset + local
            source_cell_kinds[cell_id] = block.cell_kind
        source_offset += block.cell_count
    target_offset = 0
    for block in target_mesh.blocks:
        for local, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64)):
            cell_id = int(identifier)
            if cell_id in target_cell_rows:
                raise ValueError("Target global cell IDs must be unique across blocks.")
            target_cell_rows[cell_id] = target_offset + local
            target_cell_kinds[cell_id] = block.cell_kind
        target_offset += block.cell_count
    if not partition_target_ids <= target_cell_rows.keys():
        raise ValueError("Coarsening partitions reference absent target cells.")
    partition_rows = {target_cell_rows[value] for value in partition_target_ids}

    records = (
        (witnesses,)
        if isinstance(witnesses, (NestedReferenceWitnesses, ReferenceCellMapWitnesses))
        else witnesses
    )
    indexed: dict[int, NestedSimplexMap | NestedCellReferenceMap] = {}
    from .._coordinate_enclosure import (
        add,
        axes,
        constant,
        Expression,
        expression_sequence_equal,
        RationalPolynomial,
        scale,
    )

    def exact_arguments(
        reference_map: NestedSimplexMap | NestedCellReferenceMap, /
    ) -> tuple[Expression, ...]:
        if isinstance(reference_map, NestedCellReferenceMap):
            if reference_map.arguments is None:
                raise ValueError("Reference-cell maps must retain exact chart arguments.")
            return tuple(
                value if isinstance(value, RationalPolynomial) else dict(value)
                for value in reference_map.arguments
            )
        variables = axes(reference_map.dimension)
        arguments: list[Expression] = []
        for axis in range(reference_map.dimension):
            expression = constant(
                reference_map.vertices[0][axis], reference_map.dimension
            )
            for column in range(reference_map.dimension):
                expression = add(
                    expression,
                    scale(
                        variables[column],
                        reference_map.vertices[column + 1][axis]
                        - reference_map.vertices[0][axis],
                    ),
                )
            arguments.append(expression)
        return tuple(arguments)

    for witness in records:
        if isinstance(witness, NestedReferenceWitnesses):
            maps = nested_simplex_maps(witness)
        elif isinstance(witness, ReferenceCellMapWitnesses):
            maps = nested_cell_reference_maps(witness)
        else:
            raise TypeError(
                "reference_maps entries must be exact nested-reference witness records."
            )
        for reference_map in maps:
            if isinstance(reference_map, NestedCellReferenceMap):
                target_id = int(reference_map.target_cell_id)
                source_id = int(reference_map.source_cell_id)
            else:
                target_id = int(reference_map.fine_cell_id)
                source_id = int(reference_map.coarse_cell_id)
            target_row = target_cell_rows.get(target_id)
            source_row = source_cell_rows.get(source_id)
            if target_row is None or source_row is None:
                raise ValueError(
                    "Exact reference maps must cover present global cell IDs."
                )
            if target_row in partition_rows:
                raise ValueError(
                    "A target cell cannot have both direct and coarsening maps."
                )
            target_kind = target_cell_kinds[target_id]
            source_kind = source_cell_kinds[source_id]
            if isinstance(reference_map, NestedCellReferenceMap):
                if (
                    target_kind != reference_map.target_cell_kind
                    or source_kind != reference_map.source_cell_kind
                ):
                    raise ValueError(
                        "Reference-map cell kinds disagree with their mesh rows."
                    )
            else:
                dimension = reference_map.dimension
                kind_dimensions = {
                    "interval": 1,
                    "triangle": 2,
                    "tetrahedron": 3,
                }
                target_dimension = kind_dimensions.get(target_kind)
                source_dimension = kind_dimensions.get(source_kind)
                if target_kind.startswith("simplex:"):
                    target_dimension = int(target_kind.split(":", 1)[1])
                if source_kind.startswith("simplex:"):
                    source_dimension = int(source_kind.split(":", 1)[1])
                if target_dimension != dimension or source_dimension != dimension:
                    raise ValueError(
                        "Exact simplex witnesses require simplex source and target cells."
                    )
            previous = indexed.get(target_row)
            if previous is not None:
                previous_source_id = (
                    previous.source_cell_id
                    if isinstance(previous, NestedCellReferenceMap)
                    else previous.coarse_cell_id
                )
                if int(previous_source_id) == source_id and expression_sequence_equal(
                    exact_arguments(previous), exact_arguments(reference_map)
                ):
                    continue
                raise ValueError("Exact reference maps disagree for a target cell ID.")
            indexed[target_row] = reference_map
    if set(indexed) | partition_rows != set(range(target_offset)):
        raise ValueError(
            "Exact reference maps must cover every target cell exactly once."
        )
    parents_by_target = []
    for index in range(target_offset):
        if index in partition_rows:
            parents_by_target.append(-1)
            continue
        reference_map = indexed[index]
        if isinstance(reference_map, NestedCellReferenceMap):
            source_id = reference_map.source_cell_id
        else:
            source_id = reference_map.coarse_cell_id
        parents_by_target.append(source_cell_rows[int(source_id)])
    parents = np.asarray(parents_by_target, dtype=np.int64)

    return indexed, parents


def _reference_chart_argument_ids(
    witnesses: NestedReferenceWitnesses
    | ReferenceCellMapWitnesses
    | Sequence[NestedReferenceWitnesses | ReferenceCellMapWitnesses]
    | None,
    /,
) -> dict[int, tuple[object, ...]]:
    if witnesses is None:
        return {}
    records = (
        (witnesses,)
        if isinstance(witnesses, (NestedReferenceWitnesses, ReferenceCellMapWitnesses))
        else tuple(witnesses)
    )
    result: dict[int, tuple[object, ...]] = {}
    for record in records:
        if not isinstance(record, ReferenceCellMapWitnesses):
            continue
        bank = record.source_reference_chart_arguments
        if bank is None:
            continue
        for target_id, arguments in zip(
            np.asarray(record.target_cell_ids, dtype=np.int64), bank, strict=True
        ):
            identifier = int(target_id)
            if identifier in result:
                raise ValueError("Exact chart arguments repeat a target cell ID.")
            identity = reference_chart_arguments_identity((arguments,))
            if identity is None:
                raise ValueError("Retained exact chart arguments need a stable identity.")
            result[identifier] = identity
    return result


def _require_p1_simplex_coordinates(
    discretization: FiniteElementDiscretization, side: str, /
) -> None:
    mesh = discretization.mesh
    for block_index, (block, element, routes) in enumerate(
        zip(
            mesh.blocks,
            discretization.coordinate_elements,
            discretization.coordinate_dofs,
            strict=True,
        )
    ):
        if not (
            block.cell_kind in ("interval", "triangle", "tetrahedron")
            or block.cell_kind.startswith("simplex:")
        ):
            raise ValueError(
                f"{side} parent-cell derivation requires linear simplex geometry."
            )
        topology = reference_cell_topology(block.cell_kind)
        if (
            topology.dimension != mesh.topological_dimension
            or not isinstance(element, FiniteElementSpec)
            or element.degree != 1
            or element.local_dof_count != len(topology.vertices)
            or not np.array_equal(
                np.asarray(element.reference_nodes, dtype=np.float64),
                np.asarray(topology.vertices, dtype=np.float64),
            )
            or not np.array_equal(
                np.asarray(routes, dtype=np.int64),
                np.asarray(block.vertices, dtype=np.int64),
            )
        ):
            raise ValueError(
                f"{side} parent-cell derivation requires verified P1 vertex charts."
            )


def _exact_affine_simplex_reference_vertex(
    physical_vertices: np.ndarray,
    physical_point: np.ndarray,
    reference_vertices: tuple[tuple[float, ...], ...],
    /,
) -> tuple[Fraction, ...]:
    source = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(physical_vertices, dtype=np.float64)
    )
    point = tuple(Fraction(float(value)) for value in np.asarray(physical_point))
    dimension = len(source) - 1
    ambient = len(source[0])
    if (
        dimension < 1
        or len(source) != dimension + 1
        or len(point) != ambient
        or len(reference_vertices) != dimension + 1
        or any(len(row) != ambient for row in source)
    ):
        raise ValueError("Affine simplex coordinates have inconsistent extents.")
    augmented = [
        [
            *(source[column + 1][axis] - source[0][axis] for column in range(dimension)),
            point[axis] - source[0][axis],
        ]
        for axis in range(ambient)
    ]
    pivot_rows = []
    pivot_row = 0
    for column in range(dimension):
        pivot = next(
            (row for row in range(pivot_row, ambient) if augmented[row][column] != 0),
            None,
        )
        if pivot is None:
            raise ValueError("The declared source parent simplex is singular.")
        augmented[pivot_row], augmented[pivot] = (
            augmented[pivot],
            augmented[pivot_row],
        )
        divisor = augmented[pivot_row][column]
        augmented[pivot_row] = [value / divisor for value in augmented[pivot_row]]
        for row in range(ambient):
            if row == pivot_row:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        augmented[row], augmented[pivot_row], strict=True
                    )
                ]
        pivot_rows.append(pivot_row)
        pivot_row += 1
    if any(
        not any(row[column] for column in range(dimension)) and row[dimension] != 0
        for row in augmented
    ):
        raise ValueError("A target corner is outside its declared parent affine span.")
    coordinates = tuple(augmented[row][dimension] for row in pivot_rows)
    barycentric = (Fraction(1) - sum(coordinates, Fraction(0)), *coordinates)
    return tuple(
        sum(
            (
                barycentric[corner] * Fraction(float(reference_vertices[corner][axis]))
                for corner in range(dimension + 1)
            ),
            Fraction(0),
        )
        for axis in range(dimension)
    )


def _exact_p1_simplex_reference_maps(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    parents: np.ndarray,
    /,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    source_mesh, target_mesh = source.mesh, target.mesh
    if source_mesh.topological_dimension != target_mesh.topological_dimension:
        raise ValueError("Nested parent maps require equal simplex dimensions.")
    _require_p1_simplex_coordinates(source, "Source")
    _require_p1_simplex_coordinates(target, "Target")
    source_cells = tuple(
        (block_index, cell, int(identifier))
        for block_index, block in enumerate(source_mesh.blocks)
        for cell, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    )
    target_cells = tuple(
        (block_index, cell, int(identifier))
        for block_index, block in enumerate(target_mesh.blocks)
        for cell, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    )
    if parents.shape != (len(target_cells),):
        raise ValueError("Parent indices must align with target cell block order.")
    source_coordinates = np.asarray(source.default_runtime.coordinates, dtype=np.float64)
    target_coordinates = np.asarray(target.default_runtime.coordinates, dtype=np.float64)
    grouped: dict[
        tuple[str, str],
        list[tuple[int, int, tuple[tuple[Fraction, ...], ...]]],
    ] = {}
    for target_row, (target_block_index, target_cell, target_id) in enumerate(
        target_cells
    ):
        source_row = int(parents[target_row])
        source_block_index, source_cell, source_id = source_cells[source_row]
        source_block = source_mesh.blocks[source_block_index]
        target_block = target_mesh.blocks[target_block_index]
        source_kind, target_kind = source_block.cell_kind, target_block.cell_kind
        if len(source_block.vertices[source_cell]) != len(
            target_block.vertices[target_cell]
        ):
            raise ValueError("P1 simplex parent maps require matching simplex arity.")
        source_topology = reference_cell_topology(source_kind)
        source_vertices = source_coordinates[
            np.asarray(source_block.vertices[source_cell], dtype=np.int64)
        ]
        target_vertices = target_coordinates[
            np.asarray(target_block.vertices[target_cell], dtype=np.int64)
        ]
        exact_vertices = tuple(
            _exact_affine_simplex_reference_vertex(
                source_vertices,
                target_vertex,
                source_topology.vertices,
            )
            for target_vertex in target_vertices
        )
        grouped.setdefault((target_kind, source_kind), []).append(
            (target_id, source_id, exact_vertices)
        )
    witnesses = []
    for (target_kind, source_kind), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row[0])
        target_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
        source_ids = np.asarray([row[1] for row in rows], dtype=np.int64)
        exact = tuple(row[2] for row in rows)
        execution = np.asarray(
            [[[float(value) for value in point] for point in cell] for cell in exact],
            dtype=np.float64,
        )
        witness = ReferenceCellMapWitnesses(
            target_ids,
            source_ids,
            target_kind,
            source_kind,
            execution,
            exact,
        )
        validate_reference_cell_map_witnesses(witness)
        witnesses.append(witness)
    return tuple(witnesses)


def _retained_identity_reference_maps(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    covered_target_ids: set[int],
    /,
) -> tuple[ReferenceCellMapWitnesses, ...]:
    """Witness exact identity maps for physically unchanged, retained cells."""
    from .._cell_geometry import ScalarCoordinateElement
    from .._coordinate_enclosure import (
        axes,
        coordinate_map_expressions,
        expression_sequence_equal,
    )

    source_cells: dict[int, tuple[int, int, str]] = {}
    target_cells: dict[int, tuple[int, int, str]] = {}
    for destination, mesh in (
        (source_cells, source.mesh),
        (target_cells, target.mesh),
    ):
        for block_index, block in enumerate(mesh.blocks):
            for row, identifier in enumerate(
                np.asarray(block.global_ids, dtype=np.int64)
            ):
                cell_id = int(identifier)
                if cell_id in destination:
                    raise ValueError(
                        "Retained-cell identity maps require unique global IDs."
                    )
                destination[cell_id] = (block_index, row, block.cell_kind)
    target_ids = set(target_cells)
    if not covered_target_ids <= target_ids:
        raise ValueError("Existing exact maps reference absent target FE cells.")
    source_vertex_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(target.mesh.vertex_global_ids, dtype=np.int64)
    source_coordinates = np.asarray(source.default_runtime.coordinates, dtype=np.float64)
    target_coordinates = np.asarray(target.default_runtime.coordinates, dtype=np.float64)
    grouped: dict[str, list[int]] = {}
    for cell_id in sorted(target_ids - covered_target_ids):
        source_record = source_cells.get(cell_id)
        target_record = target_cells[cell_id]
        if source_record is None:
            raise ValueError(f"Target FE cell {cell_id} has no exact source ancestry.")
        source_block_index, source_row, source_kind = source_record
        target_block_index, target_row, target_kind = target_record
        if source_kind != target_kind:
            raise ValueError(f"Retained FE cell {cell_id} changed reference-cell kind.")
        source_block = source.mesh.blocks[source_block_index]
        target_block = target.mesh.blocks[target_block_index]
        source_connectivity = source_vertex_ids[
            np.asarray(source_block.vertices[source_row], dtype=np.int64)
        ]
        target_connectivity = target_vertex_ids[
            np.asarray(target_block.vertices[target_row], dtype=np.int64)
        ]
        if not np.array_equal(source_connectivity, target_connectivity):
            raise ValueError(
                f"Retained FE cell {cell_id} changed ordered global vertex identity."
            )
        source_routes = np.asarray(
            source.coordinate_dofs[source_block_index], dtype=np.int64
        )[source_row]
        target_routes = np.asarray(
            target.coordinate_dofs[target_block_index], dtype=np.int64
        )[target_row]
        source_element = source.coordinate_elements[source_block_index]
        target_element = target.coordinate_elements[target_block_index]
        if not isinstance(source_element, ScalarCoordinateElement) or not isinstance(
            target_element, ScalarCoordinateElement
        ):
            raise TypeError("Retained FE cells require exact scalar coordinate maps.")
        source_map = coordinate_map_expressions(
            source_element, source_coordinates[source_routes], cell_row=source_row
        )
        target_map = coordinate_map_expressions(
            target_element, target_coordinates[target_routes], cell_row=target_row
        )
        if not expression_sequence_equal(source_map, target_map):
            raise ValueError(
                f"Retained FE cell {cell_id} changed its exact represented physical map."
            )
        grouped.setdefault(target_kind, []).append(cell_id)
    witnesses: list[ReferenceCellMapWitnesses] = []
    for cell_kind, identifiers in sorted(grouped.items()):
        topology = reference_cell_topology(cell_kind)
        exact_corners = tuple(
            tuple(Fraction(float(value)) for value in point)
            for point in np.asarray(topology.vertices, dtype=np.float64)
        )
        ordered_ids = np.asarray(identifiers, dtype=np.int64)
        source_ids = ordered_ids.copy()
        execution_corners = np.asarray(
            [
                [[float(value) for value in point] for point in exact_corners]
                for _ in identifiers
            ],
            dtype=np.float64,
        )
        identity_arguments = axes(topology.dimension)
        argument_bank = freeze_reference_chart_arguments(
            tuple(identity_arguments for _ in identifiers)
        )
        for bank in (ordered_ids, source_ids, execution_corners):
            bank.setflags(write=False)
        witness = ReferenceCellMapWitnesses(
            ordered_ids,
            source_ids,
            cell_kind,
            cell_kind,
            execution_corners,
            tuple(exact_corners for _ in identifiers),
            argument_bank,
        )
        validate_reference_cell_map_witnesses(witness)
        witnesses.append(witness)
    return tuple(witnesses)


def _require_array_space(space: AbstractVectorSpace, /) -> ArraySpace:
    if not isinstance(space, ArraySpace):
        raise TypeError("Finite-element moments require an array coefficient space.")
    return space


def _sparse_operator(
    rows: np.ndarray,
    columns: np.ndarray,
    values: ArrayLike,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    identifier: str,
    /,
    *,
    properties: OperatorProperties | None = None,
) -> SparseCoordinateOperator:
    relation = EdgeRelation(
        columns, rows, source_size=source.size, target_size=target.size
    )
    return SparseCoordinateOperator(
        relation,
        jnp.asarray(values, dtype=_require_array_space(target).dtype),
        source=source,
        target=target,
        operator_id=identifier,
        properties=properties,
    )


def _periodic_matrix(operator: AbstractLinearOperator, /) -> csr_matrix:
    from scipy import sparse as scipy_sparse

    if not isinstance(operator, SparseCoordinateOperator):
        raise TypeError("Periodic finite-element actions require sparse coordinates.")
    if operator.block_shape is not None:
        raise TypeError("Periodic de Rham coordinates require scalar sparse routes.")
    storage = operator.sparse_storage()
    if storage.format != "csr" or storage.batch_shape:
        raise TypeError("Periodic de Rham conversion requires unbatched canonical CSR.")
    return scipy_sparse.csr_matrix(
        (
            np.asarray(storage.values),
            np.asarray(storage.indices),
            np.asarray(storage.indptr),
        ),
        shape=storage.shape,
    )


def _sparse_matrix_operator(
    matrix: csr_matrix,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    identifier: str,
    /,
) -> SparseCoordinateOperator:
    coo = matrix.tocoo()
    keep = np.asarray(coo.data) != 0
    return _sparse_operator(
        np.asarray(coo.row[keep], dtype=np.int32),
        np.asarray(coo.col[keep], dtype=np.int32),
        np.asarray(coo.data[keep]),
        source,
        target,
        identifier,
    )


def _exact_matrix_product(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("Exact orientation matrix product has incompatible extents.")
    if any(len(row) != len(left[0]) for row in left) or any(
        len(row) != len(right[0]) for row in right
    ):
        raise ValueError("Exact orientation matrices must be rectangular.")
    return matmul_exact(left, right)


def _exact_matrix_subspace(
    matrix: tuple[tuple[Fraction, ...], ...],
    indices: tuple[int, ...],
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    if len(matrix) != len(matrix[0]) or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("Exact form orientation matrices must be square.")
    if any(index < 0 or index >= len(matrix) for index in indices):
        raise ValueError("Exact form orientation indices are out of range.")
    return tuple(tuple(matrix[row][column] for column in indices) for row in indices)


def _exact_form_cell_transform(
    mesh: CellMesh,
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    cell_row: int,
    vertex_ids: np.ndarray,
    cache: dict[
        tuple[str, tuple[int, ...], int],
        tuple[tuple[Fraction, ...], ...],
    ]
    | None = None,
    reference_gradient_cache: dict[str, np.ndarray] | None = None,
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    basis = element.form_basis
    if basis is None:
        raise ValueError("Exact cell transforms require canonical form-basis metadata.")
    cell_vertices = np.asarray(block.vertices[cell_row], dtype=np.int64)
    global_vertices = tuple(vertex_ids[cell_vertices].tolist())
    permutation = basis.canonical_permutation(global_vertices)
    sign = _form_cell_orientation_sign(
        mesh, block, element, cell_vertices, global_vertices, reference_gradient_cache
    )
    if sign not in (-1, 1):
        raise ValueError("Exact cell orientation must be nonzero and representable.")
    normalized_prism = basis.family == "prism-trimmed"
    if normalized_prism:
        rank = {
            identifier: index for index, identifier in enumerate(sorted(global_vertices))
        }
        permutation = tuple(rank[identifier] for identifier in global_vertices)
    key = (
        basis.basis_id + (":normalized-prism" if normalized_prism else ""),
        permutation,
        sign,
    )
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    if normalized_prism:
        transform = _exact_prism_canonical_form_cell(
            mesh,
            block,
            element,
            cell_vertices,
            *_prism_canonical_form_coordinates(element),
        )
    else:
        transform = basis.exact_entity_permutation_matrix(global_vertices)
    if sign != 1 and not normalized_prism:
        transform = tuple(
            tuple(sign * coefficient for coefficient in row) for row in transform
        )
    if cache is not None:
        cache[key] = transform
    return transform


def _relative_transfer_with_evidence(
    operator: SparseCoordinateOperator,
    evidence: SparseTransferEntryEvidence,
    source_indices: tuple[int, ...],
    target_indices: tuple[int, ...],
    source_space: ArraySpace,
    target_space: ArraySpace,
    identifier: str,
    /,
) -> tuple[
    SparseCoordinateOperator,
    SparseTransferEntryEvidence,
    SparseCommutingEvidence | None,
]:
    relation = operator.relation
    if operator.block_shape is not None or not isinstance(relation, EdgeRelation):
        raise TypeError("Relative de Rham transfer requires scalar edge routes.")
    if (
        evidence.source_size != operator.source.size
        or evidence.target_size != operator.target.size
    ):
        raise ValueError(
            "Relative FE action and integration evidence endpoints disagree."
        )
    source_lookup = np.full((operator.source.size,), -1, dtype=np.int32)
    target_lookup = np.full((operator.target.size,), -1, dtype=np.int32)
    source_lookup[np.asarray(source_indices, dtype=np.int64)] = np.arange(
        len(source_indices), dtype=np.int32
    )
    target_lookup[np.asarray(target_indices, dtype=np.int64)] = np.arange(
        len(target_indices), dtype=np.int32
    )
    valid = np.asarray(relation.valid, dtype=np.bool_)
    rows = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    columns = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    values = np.asarray(operator.coefficients)[valid]
    evidence_keys = np.asarray(evidence.target_indices, dtype=np.int64) * max(
        operator.source.size, 1
    ) + np.asarray(evidence.source_indices, dtype=np.int64)
    error_by_key = dict(
        zip(
            evidence_keys.tolist(),
            np.asarray(evidence.absolute_error_bounds),
            strict=True,
        )
    )
    exact_by_key = dict(
        zip(
            evidence_keys.tolist(),
            np.asarray(evidence.exact_integral_counts),
            strict=True,
        )
    )
    enclosed_by_key = dict(
        zip(
            evidence_keys.tolist(),
            np.asarray(evidence.enclosed_integral_counts),
            strict=True,
        )
    )
    keys = rows * max(operator.source.size, 1) + columns
    errors = np.asarray([error_by_key[int(key)] for key in keys], dtype=np.float64)
    exact_counts = np.asarray([exact_by_key[int(key)] for key in keys], dtype=np.int64)
    enclosed_counts = np.asarray(
        [enclosed_by_key[int(key)] for key in keys], dtype=np.int64
    )
    source_rows = source_lookup[columns]
    target_rows = target_lookup[rows]
    leaks = (source_rows >= 0) & (target_rows < 0)
    if np.any(leaks) and np.any(np.abs(values[leaks]) > errors[leaks]):
        raise ValueError(
            "FE transfer's certified intervals do not preserve the target relative boundary."
        )
    constraint_evidence = None
    if np.any(leaks):
        leak_order = np.flatnonzero(leaks)
        keep_leaks = (values[leak_order] != 0.0) | (errors[leak_order] != 0.0)
        selected = leak_order[keep_leaks]
        if selected.size:
            constraint_evidence = SparseCommutingEvidence(
                "relative-boundary",
                evidence.degree,
                operator.source.size,
                operator.target.size,
                rows[selected],
                columns[selected],
                values[selected],
                errors[selected],
                np.zeros((selected.size,), dtype=np.float64),
            )
    keep = (source_rows >= 0) & (target_rows >= 0)
    restricted = _sparse_operator(
        target_rows[keep],
        source_rows[keep],
        values[keep],
        source_space,
        target_space,
        identifier,
    )
    result_evidence = SparseTransferEntryEvidence(
        evidence.degree,
        source_space.size,
        target_space.size,
        target_rows[keep],
        source_rows[keep],
        errors[keep],
        exact_counts[keep],
        enclosed_counts[keep],
    )
    return restricted, result_evidence, constraint_evidence


def _matrix_entry_error(evidence: SparseTransferEntryEvidence, /) -> csr_matrix:
    from scipy import sparse as scipy_sparse

    return scipy_sparse.coo_matrix(
        (
            np.asarray(evidence.absolute_error_bounds),
            (
                np.asarray(evidence.target_indices, dtype=np.int64),
                np.asarray(evidence.source_indices, dtype=np.int64),
            ),
        ),
        shape=(evidence.target_size, evidence.source_size),
        dtype=np.float64,
    ).tocsr()


def _gamma_bound(operation_count: int, /) -> float:
    if operation_count <= 0:
        return 0.0
    product = float(operation_count) * np.finfo(np.float64).eps
    return np.inf if product >= 1.0 else product / (1.0 - product)


def _maximum_row_terms(matrix: csr_matrix, /) -> int:
    return int(np.max(np.diff(matrix.indptr), initial=0))


_SMALLEST_SUBNORMAL = float(np.ldexp(1.0, -1074))


def _nonzero_pattern(matrix: csr_matrix, /) -> csr_matrix:
    pattern = matrix.copy()
    pattern.data = (pattern.data != 0.0).astype(np.float64)
    pattern.eliminate_zeros()
    return pattern


def _product_counts(first: csr_matrix, second: csr_matrix, /) -> csr_matrix:
    """Per-entry count of nonzero scalar products formed by ``first @ second``."""
    return (_nonzero_pattern(first) @ _nonzero_pattern(second)).tocsr()


def _require_complex_map_commutes(
    map: ComplexMap,
    entry_evidence: Sequence[SparseTransferEntryEvidence],
    integration_policy: CurvedOverlapIntegrationPolicy | None,
    /,
) -> tuple[SparseCommutingEvidence, ...]:

    if len(entry_evidence) != len(map.maps):
        raise ValueError("Commuting checks require integration evidence in every degree.")
    epsilon = np.finfo(np.float64).eps

    def check_degree(degree: int, /) -> SparseCommutingEvidence:
        target_differential = _periodic_matrix(map.target.differential(degree))
        target_map = _periodic_matrix(map.map(degree))
        next_map = _periodic_matrix(map.map(degree + 1))
        source_differential = _periodic_matrix(map.source.differential(degree))
        left = (target_differential @ target_map).tocsr()
        right = (next_map @ source_differential).tocsr()
        difference = (left - right).tocsr()
        del left, right
        difference.eliminate_zeros()
        observed = abs(difference)
        current_error = _matrix_entry_error(entry_evidence[degree])
        next_error = _matrix_entry_error(entry_evidence[degree + 1])
        # Gradual underflow adds an absolute error of at most 2^-1075 to every
        # rounded product and none to additions; relative gamma terms miss it.
        product_counts = (
            _product_counts(target_differential, target_map)
            + _product_counts(next_map, source_differential)
            + _product_counts(target_differential, current_error)
            + _product_counts(next_error, source_differential)
        ).tocsr()
        left_integral_error = abs(target_differential) @ current_error
        right_integral_error = next_error @ abs(source_differential)
        integration_error = (left_integral_error + right_integral_error).tocsr()
        del current_error, next_error, left_integral_error, right_integral_error
        integration_error.eliminate_zeros()
        left_magnitude = abs(target_differential) @ abs(target_map)
        right_magnitude = abs(next_map) @ abs(source_differential)
        left_terms = _maximum_row_terms(target_differential) * _maximum_row_terms(
            target_map
        )
        right_terms = _maximum_row_terms(next_map) * _maximum_row_terms(
            source_differential
        )
        scale_matrix = (left_magnitude + right_magnitude + integration_error).tocsr()
        roundoff = (
            _gamma_bound(2 * left_terms) * left_magnitude
            + _gamma_bound(2 * right_terms) * right_magnitude
        ).tocsr()
        del left_magnitude, right_magnitude
        scale = max(
            1.0,
            float(np.max(scale_matrix.data, initial=0.0)),
        )
        roundoff = (roundoff + scale_matrix.multiply(64.0 * epsilon * scale)).tocsr()
        integration_roundoff = (
            _gamma_bound(
                2
                * max(
                    _maximum_row_terms(target_differential),
                    _maximum_row_terms(source_differential),
                    1,
                )
            )
            * integration_error
        )
        roundoff = (roundoff + integration_roundoff).tocsr()
        # Each counted product is charged 2^-1074 (twice its underflow loss, which
        # also covers the gamma terms evaluated on underflowed magnitudes); two
        # more units cover the four rounded scalar products assembling roundoff.
        product_counts.data += 2.0
        roundoff = (roundoff + product_counts.multiply(_SMALLEST_SUBNORMAL)).tocsr()
        del product_counts
        del (
            integration_roundoff,
            target_differential,
            source_differential,
            target_map,
            next_map,
        )
        if integration_error.nnz:
            integration_error.data = np.nextafter(
                np.maximum(integration_error.data, 0.0), np.inf
            )
        if roundoff.nnz:
            roundoff.data = np.nextafter(np.maximum(roundoff.data, 0.0), np.inf)
        allowed = None
        if integration_policy is not None:
            allowed = scale_matrix.copy()
            allowed.data = (
                integration_policy.absolute_tolerance
                + integration_policy.relative_tolerance * allowed.data
            )
        del scale_matrix
        total_bound = (integration_error + roundoff).tocsr()
        excess = (observed - total_bound).tocsr()
        excess.eliminate_zeros()
        if excess.nnz and float(np.max(excess.data, initial=0.0)) > 0.0:
            raise ValueError(
                f"FE transfer does not commute within its certified moment intervals in degree {degree}."
            )
        del excess
        if allowed is not None:
            tolerance_excess = (total_bound - allowed).tocsr()
            tolerance_excess.eliminate_zeros()
            if (
                tolerance_excess.nnz
                and float(np.max(tolerance_excess.data, initial=0.0)) > 0.0
            ):
                raise ValueError(
                    f"FE commuting enclosure exceeds integration_policy in degree {degree}."
                )
            del tolerance_excess
        union = (observed + total_bound).tocsr()
        union.eliminate_zeros()
        del observed, total_bound, allowed
        coo = union.tocoo()
        del union
        if coo.nnz:
            observed_values = np.asarray(difference.tocsr()[coo.row, coo.col]).reshape(-1)
            del difference
            integration_values = np.asarray(
                integration_error.tocsr()[coo.row, coo.col]
            ).reshape(-1)
            del integration_error
            roundoff_values = np.asarray(roundoff.tocsr()[coo.row, coo.col]).reshape(-1)
            del roundoff
        else:
            observed_values = integration_values = roundoff_values = np.zeros(
                (0,), dtype=np.float64
            )
            del difference, integration_error, roundoff
        return SparseCommutingEvidence(
            "commuting",
            degree,
            map.source.space(degree).size,
            map.target.space(degree + 1).size,
            np.asarray(coo.row, dtype=np.int32),
            np.asarray(coo.col, dtype=np.int32),
            observed_values,
            integration_values,
            roundoff_values,
        )

    return tuple(check_degree(degree) for degree in range(map.source.top_degree))


def _periodic_quotient_transfer_action(
    lifted_result: FiniteElementTransferResult,
    degree: int,
    source_lift: SparseCoordinateOperator,
    target_collapse: SparseCoordinateOperator,
    quotient_action: csr_matrix,
    identifier: str,
    /,
) -> tuple[SparseCoordinateOperator, SparseTransferEntryEvidence]:
    from scipy import sparse as scipy_sparse

    lifted_operator = lifted_result.complex_map.map(degree)
    if not isinstance(lifted_operator, SparseCoordinateOperator):
        raise TypeError("Periodic FE quotient actions require sparse lifted maps.")
    evidence = lifted_result.entry_evidence[degree]
    source_lift_matrix = _periodic_matrix(source_lift)
    target_collapse_matrix = _periodic_matrix(target_collapse)
    lifted_matrix = _periodic_matrix(lifted_operator)
    source_abs = abs(source_lift_matrix)
    target_abs = abs(target_collapse_matrix)
    lifted_abs = abs(lifted_matrix)
    error_matrix = _matrix_entry_error(evidence)
    left_integral_error = target_abs @ error_matrix
    integral_error = left_integral_error @ source_abs
    magnitude = target_abs @ lifted_abs @ source_abs
    left_stage_magnitude = target_abs @ lifted_abs
    left_terms = _maximum_row_terms(target_abs) * _maximum_row_terms(lifted_abs)
    right_terms = _maximum_row_terms(left_stage_magnitude) * _maximum_row_terms(
        source_abs
    )
    roundoff = (
        _gamma_bound(2 * left_terms) * (left_stage_magnitude @ source_abs)
        + _gamma_bound(2 * right_terms) * magnitude
        + 64.0 * np.finfo(np.float64).eps * magnitude
    ).tocsr()
    integral_roundoff = (
        _gamma_bound(
            2
            * max(
                _maximum_row_terms(target_abs),
                _maximum_row_terms(source_abs),
                1,
            )
        )
        * integral_error
    )
    # Gradual underflow adds an absolute error of at most 2^-1075 to every rounded
    # product and none to additions; relative gamma terms miss it. Losses of the
    # first stage reach the result through the source lift, whose entries are
    # bounded by their integer ceilings. Each counted product is charged 2^-1074
    # (twice its loss, which also covers the gamma terms evaluated on underflowed
    # magnitudes); two more units cover the four rounded scalar products that
    # assemble roundoff and integral_roundoff.
    source_ceiling = source_abs.copy()
    source_ceiling.data = np.ceil(source_ceiling.data)
    product_counts = (
        _product_counts(target_collapse_matrix, lifted_matrix) @ source_ceiling
        + _product_counts(left_stage_magnitude, source_abs)
        + _product_counts(target_abs, error_matrix) @ source_ceiling
        + _product_counts(left_integral_error, source_abs)
    ).tocsr()
    del source_ceiling, left_integral_error
    product_counts.data += 2.0
    roundoff = (roundoff + product_counts.multiply(_SMALLEST_SUBNORMAL)).tocsr()
    del product_counts
    error_matrix = (integral_error + roundoff + integral_roundoff).tocsr()
    error_matrix.eliminate_zeros()
    if error_matrix.nnz:
        error_matrix.data = np.nextafter(np.maximum(error_matrix.data, 0.0), np.inf)
    exact_input = scipy_sparse.coo_matrix(
        (
            np.asarray(evidence.exact_integral_counts, dtype=np.int64),
            (
                np.asarray(evidence.target_indices, dtype=np.int64),
                np.asarray(evidence.source_indices, dtype=np.int64),
            ),
        ),
        shape=(evidence.target_size, evidence.source_size),
        dtype=np.int64,
    ).tocsr()
    enclosed_input = scipy_sparse.coo_matrix(
        (
            np.asarray(evidence.enclosed_integral_counts, dtype=np.int64),
            (
                np.asarray(evidence.target_indices, dtype=np.int64),
                np.asarray(evidence.source_indices, dtype=np.int64),
            ),
        ),
        shape=(evidence.target_size, evidence.source_size),
        dtype=np.int64,
    ).tocsr()
    exact_output = (
        target_abs.astype(np.int64) @ exact_input @ source_abs.astype(np.int64)
    ).tocsr()
    enclosed_output = (
        target_abs.astype(np.int64) @ enclosed_input @ source_abs.astype(np.int64)
    ).tocsr()
    quotient = quotient_action.tocsr()
    union = (abs(quotient) + error_matrix).tocoo()
    keep = np.asarray(union.data) != 0.0
    rows = np.asarray(union.row[keep], dtype=np.int64)
    columns = np.asarray(union.col[keep], dtype=np.int64)
    order = np.lexsort((columns, rows))
    rows, columns = rows[order], columns[order]
    values = np.asarray(quotient[rows, columns]).reshape(-1)
    errors = np.asarray(error_matrix[rows, columns]).reshape(-1)
    exact_counts = np.asarray(exact_output[rows, columns]).reshape(-1)
    enclosed_counts = np.asarray(enclosed_output[rows, columns]).reshape(-1)
    operator = _sparse_operator(
        rows.astype(np.int32),
        columns.astype(np.int32),
        values,
        source_lift.source,
        target_collapse.target,
        identifier,
    )
    result_evidence = SparseTransferEntryEvidence(
        degree,
        source_lift.source.size,
        target_collapse.target.size,
        rows,
        columns,
        errors,
        exact_counts,
        enclosed_counts,
    )
    return operator, result_evidence


def _sparse_defect(left: csr_matrix, right: csr_matrix, /) -> tuple[float, float]:
    difference = (left - right).tocsr()
    difference.eliminate_zeros()
    scale = max(
        1.0,
        0.0 if left.nnz == 0 else float(np.max(np.abs(left.data))),
        0.0 if right.nnz == 0 else float(np.max(np.abs(right.data))),
    )
    defect = 0.0 if difference.nnz == 0 else float(np.max(np.abs(difference.data)))
    return defect, scale


def _require_sparse_identity(
    left: csr_matrix, right: csr_matrix, context: str, /
) -> float:
    defect, scale = _sparse_defect(left, right)
    if not np.isfinite(defect) or defect > 1.0e-10 * scale:
        raise ValueError(
            f"{context} failed: coefficient defect {defect:.6g} at scale {scale:.6g}."
        )
    return defect


class _PeriodicFormTopology:
    """Call-local incidence and orbit preparation for one actual periodic mesh.

    These tables depend only on mesh connectivity and periodic topology. Form
    degree, twist, coordinate geometry and field orientation remain checked by
    each consumer, and are not inferred from this preparation.
    """

    __slots__ = (
        "mesh",
        "dimension",
        "ordered_vertices",
        "vertex_sets",
        "lookups",
        "orbit_maps",
    )

    def __init__(self, mesh: CellMesh, dimension: int, /) -> None:
        from ...meshing._periodic import periodic_entity_orbit_map
        from .._periodic_topology import _entity_vertex_rows
        from ._generic import _topology_vertex_sets

        if mesh.periodic_topology is None or dimension != mesh.topological_dimension:
            raise ValueError("Periodic form topology requires its actual periodic mesh.")
        self.mesh = mesh
        self.dimension = dimension
        self.ordered_vertices = _entity_vertex_rows(mesh)
        self.vertex_sets = _topology_vertex_sets(mesh)
        if len(self.ordered_vertices) != dimension + 1 or len(self.vertex_sets) != len(
            self.ordered_vertices
        ):
            raise ValueError(
                "Periodic entity corner tables do not span the cell complex."
            )
        self.lookups = tuple(
            MappingProxyType(
                {tuple(sorted(vertices)): row for row, vertices in enumerate(level)}
            )
            for level in self.vertex_sets
        )
        self.orbit_maps = tuple(
            periodic_entity_orbit_map(mesh, degree) for degree in range(dimension + 1)
        )

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError("Periodic form topology preparation is immutable.")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Periodic form topology preparation is immutable.")

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> NoReturn:
        raise TypeError("Periodic form topology preparation cannot be serialized.")

    def require(self, mesh: CellMesh, dimension: int, /) -> None:
        if self.mesh is not mesh or self.dimension != dimension:
            raise ValueError("Periodic form topology binds a different actual mesh.")
        for orbit in self.orbit_maps:
            orbit.require_source(mesh)


def _vertex_id_orientation(vertices: tuple[int, ...], vertex_ids: np.ndarray, /) -> int:
    """Sign taking an entity's vertex-row orientation to its ascending-ID one.

    Orbit signs orient entities by topology vertex rows, while form moments
    orient them by ascending global vertex ID.
    """
    ids = [int(vertex_ids[vertex]) for vertex in vertices]
    inversions = sum(
        ids[first] > ids[second]
        for first in range(len(ids))
        for second in range(first + 1, len(ids))
    )
    return -1 if inversions % 2 else 1


def _periodic_entity_coordinate_maps(
    complex: FiniteElementDeRhamComplex,
    degree: int,
    /,
    *,
    _topology: _PeriodicFormTopology | None = None,
) -> tuple[
    SparseCoordinateOperator,
    SparseCoordinateOperator,
    tuple[str, ...],
    np.ndarray,
]:
    """Build exact-form DOF lifts and representative collapses for every orbit."""
    from scipy import sparse as scipy_sparse

    mesh = complex.mesh
    periodic = mesh.periodic_topology
    if periodic is None:
        raise ValueError("Quotient transfer requires periodic endpoint meshes.")
    if degree < 0 or degree > complex.dimension:
        raise ValueError("Finite-element degree is outside its de Rham complex.")
    dof_map = complex.discretization.dof_maps[degree]
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    elements = complex.discretization.elements[degree]
    if len(elements) != len(mesh.blocks):
        raise ValueError("Periodic form elements do not align with cell blocks.")
    topology = (
        _PeriodicFormTopology(mesh, complex.dimension) if _topology is None else _topology
    )
    if type(topology) is not _PeriodicFormTopology:
        raise TypeError("Periodic form topology must be an operation-local preparation.")
    topology.require(mesh, complex.dimension)
    ordered_vertices = topology.ordered_vertices
    vertex_sets = topology.vertex_sets
    lookups = topology.lookups
    orbit_maps = topology.orbit_maps
    for orbit in orbit_maps:
        signs = np.asarray(orbit.orientation_signs, dtype=np.int32)
        shifts = np.asarray(orbit.winding_shifts, dtype=np.int64)
        representatives = np.asarray(orbit.representative_lifted_rows, dtype=np.int64)
        if (
            signs.shape != orbit.lifted_to_quotient.shape
            or np.any(np.abs(signs) != 1)
            or shifts.ndim != 2
            or shifts.shape[0] != signs.size
            or representatives.shape != orbit.quotient_entity_ids.shape
            or np.any(shifts[representatives] != 0)
        ):
            raise ValueError("Periodic orientation and winding tables are inconsistent.")
    orbit_lifted_rows = tuple(
        np.asarray(orbit.lifted_to_quotient, dtype=np.int64) for orbit in orbit_maps
    )
    orbit_representatives = tuple(
        np.asarray(orbit.representative_lifted_rows, dtype=np.int64)
        for orbit in orbit_maps
    )
    orbit_signs = tuple(
        np.asarray(orbit.orientation_signs, dtype=np.int32) for orbit in orbit_maps
    )
    entity_charts: list[list[tuple[int, ...] | None]] = [
        [None] * len(vertex_sets[entity_degree])
        for entity_degree in range(complex.dimension + 1)
    ]
    entity_chart_owners: list[list[tuple[int, int, tuple[int, ...]] | None]] = [
        [None] * len(vertex_sets[entity_degree])
        for entity_degree in range(complex.dimension + 1)
    ]
    for block_index, (block, element) in enumerate(
        zip(mesh.blocks, elements, strict=True)
    ):
        basis = element.form_basis
        if basis is None:
            raise ValueError("Periodic transfer requires canonical form-basis metadata.")
        cells = np.asarray(block.vertices, dtype=np.int64)
        for cell_row in range(cells.shape[0]):
            cell = cells[cell_row]
            for entity_degree, (faces, dof_groups) in enumerate(
                zip(basis.entity_vertices, element.entity_dofs, strict=True)
            ):
                for face, local_dofs in zip(faces, dof_groups, strict=True):
                    chart = (
                        _prism_form_entity_chart_vertices(face)
                        if basis.family == "prism-trimmed"
                        else face
                    )
                    local_vertices = tuple(int(cell[index]) for index in chart)
                    entity_row = lookups[entity_degree].get(tuple(sorted(local_vertices)))
                    if entity_row is None:
                        raise ValueError("A form chart is absent from mesh incidence.")
                    if entity_charts[entity_degree][entity_row] is None:
                        entity_charts[entity_degree][entity_row] = local_vertices
                        entity_chart_owners[entity_degree][entity_row] = (
                            block_index,
                            cell_row,
                            tuple(int(value) for value in local_dofs),
                        )
    entity_dof_ids: list[list[np.ndarray | None]] = [
        [None] * len(vertex_sets[entity_degree])
        for entity_degree in range(complex.dimension + 1)
    ]
    entity_actions: list[list[np.ndarray | None]] = [
        [None] * len(vertex_sets[entity_degree])
        for entity_degree in range(complex.dimension + 1)
    ]
    entity_exact_actions: list[list[tuple[tuple[Fraction, ...], ...] | None]] = [
        [None] * len(vertex_sets[entity_degree])
        for entity_degree in range(complex.dimension + 1)
    ]
    basis_cache: dict[tuple[int, int, int, str, str], FormBasis] = {}
    cell_transform_cache: dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]] = {}
    cell_sign_cache: dict[tuple[int, int], int] = {}
    prism_normalizations: dict[str, tuple[tuple[Fraction, ...], ...]] = {}
    orientation_transform_cache: dict[
        tuple[str, tuple[int, ...], int],
        tuple[tuple[Fraction, ...], ...],
    ] = {}
    reference_gradient_cache: dict[str, np.ndarray] = {}

    def exact_cell_transform(
        block_index: int, cell_row: int, /
    ) -> tuple[tuple[Fraction, ...], ...]:
        nonlocal cell_transform_cache, orientation_transform_cache
        nonlocal reference_gradient_cache, prism_normalizations
        key = (block_index, cell_row)
        cached = cell_transform_cache.get(key)
        if cached is not None:
            return cached
        block_ = mesh.blocks[block_index]
        element_ = elements[block_index]
        result = _exact_form_cell_transform(
            mesh,
            block_,
            element_,
            cell_row,
            vertex_ids,
            orientation_transform_cache,
            reference_gradient_cache,
        )
        if (
            element_.form_basis is not None
            and element_.form_basis.family == "prism-trimmed"
        ):
            normalization = prism_normalizations.get(element_.element_id)
            if normalization is None:
                normalization = _prism_canonical_form_coordinates(element_)[0]
                prism_normalizations[element_.element_id] = normalization
            result = tuple(tuple(row) for row in solve_exact(normalization, result))
        cell_transform_cache[key] = result
        return result

    def cell_coorientation(block_index: int, cell_row: int, /) -> int:
        nonlocal cell_sign_cache, reference_gradient_cache
        key = (block_index, cell_row)
        sign = cell_sign_cache.get(key)
        if sign is None:
            block_ = mesh.blocks[block_index]
            cell_vertices = np.asarray(block_.vertices[cell_row], dtype=np.int64)
            sign = _form_cell_orientation_sign(
                mesh,
                block_,
                elements[block_index],
                cell_vertices,
                tuple(vertex_ids[cell_vertices].tolist()),
                reference_gradient_cache,
            )
            if sign not in (-1, 1):
                raise ValueError("Periodic cell coorientation must be nonzero.")
            cell_sign_cache[key] = sign
        return sign

    for block_index, (block, element) in enumerate(
        zip(mesh.blocks, elements, strict=True)
    ):
        cell_basis = element.form_basis
        if cell_basis is None or cell_basis.form_degree != degree:
            raise ValueError("Periodic transfer requires canonical form-basis metadata.")
        cells = np.asarray(block.vertices, dtype=np.int64)
        cell_dofs = np.asarray(dof_map.cell_dofs[block_index], dtype=np.int64)
        cell_transforms = np.asarray(dof_map.cell_transforms[block_index])
        if (
            cells.ndim != 2
            or cell_dofs.ndim != 2
            or cell_transforms.shape
            != (cells.shape[0], cell_dofs.shape[1], cell_dofs.shape[1])
        ):
            raise ValueError(
                "Periodic finite-element cell tables have inconsistent shapes."
            )
        if len(cell_basis.entity_vertices) != complex.dimension + 1:
            raise ValueError("Form entity charts do not span the cell dimension.")
        for entity_degree, (local_entities, local_dof_entities) in enumerate(
            zip(
                cell_basis.entity_vertices,
                element.entity_dofs,
                strict=True,
            )
        ):
            if len(local_entities) != len(local_dof_entities):
                raise ValueError("Form entity charts and moment groups disagree.")
            orbit = orbit_maps[entity_degree]
            ordered_level = ordered_vertices[entity_degree]
            corner_offsets = np.asarray(orbit.corner_permutation_offsets, dtype=np.int64)
            if corner_offsets.shape != (len(ordered_level) + 1,):
                raise ValueError("Periodic corner permutations do not match entity rows.")
            for cell_row in range(cells.shape[0]):
                cell = tuple(
                    int(cells[cell_row, index]) for index in range(cells.shape[1])
                )
                for local_entity, (face, local_dofs) in enumerate(
                    zip(local_entities, local_dof_entities, strict=True)
                ):
                    if not local_dofs:
                        continue
                    chart = (
                        _prism_form_entity_chart_vertices(face)
                        if cell_basis.family == "prism-trimmed"
                        else face
                    )
                    local_vertex_rows = tuple(int(cell[index]) for index in chart)
                    entity_key = tuple(sorted(local_vertex_rows))
                    entity_row = lookups[entity_degree].get(entity_key)
                    if entity_row is None:
                        raise ValueError("A form entity is absent from mesh incidence.")
                    entity_vertices = tuple(
                        int(value) for value in ordered_level[entity_row]
                    )
                    entity_permutation = np.asarray(
                        orbit.corner_permutation_row(entity_row), dtype=np.int64
                    )
                    if (
                        entity_permutation.shape != (len(entity_vertices),)
                        or set(entity_vertices) != set(local_vertex_rows)
                        or sorted(entity_permutation.tolist())
                        != list(range(len(entity_vertices)))
                    ):
                        raise ValueError(
                            "Periodic corner action disagrees with the FE entity."
                        )
                    q_label_by_vertex = {
                        entity_vertices[int(position)]: quotient_corner
                        for quotient_corner, position in enumerate(entity_permutation)
                    }
                    quotient_row = int(orbit_lifted_rows[entity_degree][entity_row])
                    representative_row = int(
                        orbit_representatives[entity_degree][quotient_row]
                    )
                    representative_vertices = tuple(
                        int(value) for value in ordered_level[representative_row]
                    )
                    representative_permutation = np.asarray(
                        orbit.corner_permutation_row(representative_row),
                        dtype=np.int64,
                    )
                    if representative_permutation.shape != (
                        len(representative_vertices),
                    ):
                        raise ValueError(
                            "Periodic representative corner map is malformed."
                        )
                    representative_labels_by_vertex = {
                        representative_vertices[int(position)]: quotient_corner
                        for quotient_corner, position in enumerate(
                            representative_permutation
                        )
                    }
                    representative_owner = entity_chart_owners[entity_degree][
                        representative_row
                    ]
                    if representative_owner is None:
                        raise ValueError(
                            "Periodic representative has no incident FE cell chart."
                        )
                    (
                        representative_block,
                        representative_cell,
                        representative_local_dofs,
                    ) = representative_owner
                    representative_block = int(representative_block)
                    representative_cell = int(representative_cell)
                    representative_chart = entity_charts[entity_degree][
                        representative_row
                    ]
                    if representative_chart is None:
                        raise ValueError(
                            "Periodic representative has no FE reference chart."
                        )
                    representative_chart_labels = tuple(
                        representative_labels_by_vertex[vertex]
                        for vertex in representative_chart
                    )
                    representative_index_by_label = {
                        label: index
                        for index, label in enumerate(representative_chart_labels)
                    }
                    local_q_vertices = tuple(
                        representative_index_by_label[q_label_by_vertex[vertex]]
                        for vertex in local_vertex_rows
                    )
                    local_dofs_tuple = tuple(local_dofs)
                    local_dofs_array = np.asarray(local_dofs_tuple, dtype=np.int64)
                    transform = _exact_matrix_subspace(
                        exact_cell_transform(block_index, cell_row),
                        local_dofs_tuple,
                    )
                    if entity_degree == 0:
                        local_action_exact = tuple(
                            tuple(
                                Fraction(int(row == column))
                                for column in range(local_dofs_array.size)
                            )
                            for row in range(local_dofs_array.size)
                        )
                    else:
                        entity_family = (
                            (
                                "tensor-trimmed"
                                if entity_degree == 2 and len(face) == 4
                                else "trimmed"
                                if entity_degree < complex.dimension
                                else cell_basis.family
                            )
                            if cell_basis.family == "prism-trimmed"
                            else cell_basis.family
                        )
                        # Cell transforms orient entity moments as ordinary traces;
                        # the ambient cell coorientation owns the twist character.
                        # Applying the twist again to an entity chart permutation
                        # would introduce a spurious sign on reversed traces.
                        entity_twist = (
                            "untwisted"
                            if entity_degree < complex.dimension
                            else cell_basis.twist
                        )
                        basis_key = (
                            entity_degree,
                            degree,
                            cell_basis.order,
                            entity_family,
                            entity_twist,
                        )
                        entity_basis = basis_cache.get(basis_key)
                        if entity_basis is None:
                            entity_basis = FormBasis(
                                entity_degree,
                                degree,
                                cell_basis.order,
                                entity_family,
                                entity_twist,
                            )
                            basis_cache[basis_key] = entity_basis
                        full_action = entity_basis.exact_permutation_matrix(
                            local_q_vertices
                        )
                        vertex_count = len(face)
                        interior = tuple(
                            _interior_dof_indices(entity_basis.dof_labels, vertex_count)
                        )
                        local_action_exact = _exact_matrix_subspace(full_action, interior)
                    if len(local_action_exact) != local_dofs_array.size or any(
                        len(row) != local_dofs_array.size for row in local_action_exact
                    ):
                        raise ValueError(
                            "Periodic entity action does not match its moment space."
                        )
                    representative_local_dofs_tuple = tuple(representative_local_dofs)
                    if len(representative_local_dofs_tuple) != local_dofs_array.size:
                        raise ValueError(
                            "Periodic representative and member moment spaces disagree."
                        )
                    if (
                        cell_basis.twist == "twisted"
                        and entity_degree < complex.dimension
                    ):
                        # The entity chart action is untwisted, while local
                        # coefficients carry each ambient cell's coorientation.
                        local_sign = cell_coorientation(block_index, cell_row)
                        representative_cell_sign = cell_coorientation(
                            representative_block, representative_cell
                        )
                        local_action_exact = tuple(
                            tuple(
                                local_sign * representative_cell_sign * value
                                for value in row
                            )
                            for row in local_action_exact
                        )
                    representative_transform = _exact_matrix_subspace(
                        exact_cell_transform(representative_block, representative_cell),
                        representative_local_dofs_tuple,
                    )
                    if len(representative_transform) != len(local_action_exact):
                        raise ValueError(
                            "Periodic representative orientation map has the wrong shape."
                        )
                    global_action_exact = tuple(
                        tuple(row)
                        for row in solve_exact(
                            transform,
                            _exact_matrix_product(
                                local_action_exact, representative_transform
                            ),
                        )
                    )
                    global_action = np.asarray(global_action_exact, dtype=np.float64)
                    if not np.all(np.isfinite(global_action)):
                        raise ValueError("Periodic entity action is not finite.")
                    orientation_sign = int(orbit_signs[entity_degree][entity_row])
                    representative_sign = int(
                        orbit_signs[entity_degree][representative_row]
                    )
                    if entity_degree == 0:
                        if orientation_sign != 1:
                            raise ValueError(
                                "Scalar vertex orbits cannot reverse orientation."
                            )
                    elif (
                        entity_degree == degree
                        and entity_degree < complex.dimension
                        and global_action.shape == (1, 1)
                        and cell_basis.twist == "untwisted"
                        and global_action_exact[0][0]
                        != Fraction(
                            orientation_sign
                            * representative_sign
                            * _vertex_id_orientation(entity_vertices, vertex_ids)
                            * _vertex_id_orientation(representative_vertices, vertex_ids)
                        )
                    ):
                        raise ValueError(
                            "Periodic oriented scalar moment disagrees with its orbit sign."
                        )
                    global_ids = cell_dofs[cell_row, local_dofs_array]
                    previous_ids = entity_dof_ids[entity_degree][entity_row]
                    previous_action = entity_actions[entity_degree][entity_row]
                    previous_exact_action = entity_exact_actions[entity_degree][
                        entity_row
                    ]
                    if previous_ids is None:
                        entity_dof_ids[entity_degree][entity_row] = global_ids.copy()
                        entity_actions[entity_degree][entity_row] = global_action
                        entity_exact_actions[entity_degree][entity_row] = (
                            global_action_exact
                        )
                    elif (
                        previous_action is None
                        or previous_exact_action is None
                        or not np.array_equal(previous_ids, global_ids)
                        or not np.array_equal(previous_action, global_action)
                        or previous_exact_action != global_action_exact
                    ):
                        raise ValueError(
                            "Incident cells disagree on a periodic entity moment action."
                        )

    del exact_cell_transform, cell_coorientation
    del entity_charts, entity_chart_owners, cell_transform_cache, cell_sign_cache
    del (
        basis_cache,
        prism_normalizations,
        orientation_transform_cache,
        reference_gradient_cache,
    )

    rows_by_entity: list[list[np.ndarray | None]] = []
    actions_by_entity: list[list[np.ndarray | None]] = []
    exact_actions_by_entity: list[list[tuple[tuple[Fraction, ...], ...] | None]] = []
    quotient_widths: list[np.ndarray] = []
    quotient_ids: list[np.ndarray] = []
    for entity_degree, orbit in enumerate(orbit_maps):
        rows = entity_dof_ids[entity_degree]
        actions = entity_actions[entity_degree]
        exact_actions = entity_exact_actions[entity_degree]
        if len(rows) != orbit.lifted_entity_ids.shape[0]:
            raise ValueError("Periodic DOF rows do not match lifted entity identities.")
        widths = np.full(orbit.quotient_entity_ids.shape[0], -1, dtype=np.int64)
        orbit_rows = np.asarray(orbit.lifted_to_quotient, dtype=np.int64)
        for entity_row, (global_ids, action, exact_action) in enumerate(
            zip(rows, actions, exact_actions, strict=True)
        ):
            width = 0 if global_ids is None else int(global_ids.size)
            quotient_row = int(orbit_rows[entity_row])
            if widths[quotient_row] not in (-1, width):
                raise ValueError("Periodic images have different FE moment-space widths.")
            widths[quotient_row] = width
            if global_ids is not None and (
                action is None
                or action.shape != (width, width)
                or exact_action is None
                or len(exact_action) != width
                or any(len(row) != width for row in exact_action)
            ):
                raise ValueError("Periodic entity moment action has an invalid extent.")
        widths[widths < 0] = 0
        rows_by_entity.append(rows)
        actions_by_entity.append(actions)
        exact_actions_by_entity.append(exact_actions)
        quotient_widths.append(widths)
        quotient_ids.append(np.asarray(orbit.quotient_entity_ids, dtype=np.int64))

    all_global_ids = [
        int(value)
        for level in rows_by_entity
        for entity in level
        if entity is not None
        for value in entity
    ]
    global_count = int(dof_map.global_dof_count)
    if (
        len(all_global_ids) != global_count
        or len(set(all_global_ids)) != global_count
        or sorted(all_global_ids) != list(range(global_count))
    ):
        raise ValueError(
            "Periodic entity actions do not partition the FE coordinate space."
        )
    del all_global_ids

    lift_rows: list[np.ndarray] = []
    lift_columns: list[np.ndarray] = []
    lift_values: list[np.ndarray] = []
    collapse_rows: list[np.ndarray] = []
    collapse_columns: list[np.ndarray] = []
    collapse_values: list[np.ndarray] = []
    quotient_counts: list[int] = []
    for entity_degree, orbit in enumerate(orbit_maps):
        widths = quotient_widths[entity_degree]
        offsets = np.concatenate(
            (np.asarray((0,), dtype=np.int64), np.cumsum(widths, dtype=np.int64))
        )
        degree_offset = sum(quotient_counts)
        quotient_counts.append(int(offsets[-1]))
        orbit_rows = np.asarray(orbit.lifted_to_quotient, dtype=np.int64)
        for entity_row, (global_ids, action, exact_action) in enumerate(
            zip(
                rows_by_entity[entity_degree],
                actions_by_entity[entity_degree],
                exact_actions_by_entity[entity_degree],
                strict=True,
            )
        ):
            if global_ids is None or action is None or exact_action is None:
                continue
            if any(
                coefficient != 0 and action[row, column] == 0.0
                for row, values in enumerate(exact_action)
                for column, coefficient in enumerate(values)
            ):
                raise ValueError(
                    "Periodic exact entity action is not representable in binary64."
                )
            quotient_row = int(orbit_rows[entity_row])
            local_row, local_column = np.nonzero(action)
            if local_row.size:
                lift_rows.append(global_ids[local_row])
                lift_columns.append(degree_offset + offsets[quotient_row] + local_column)
                lift_values.append(action[local_row, local_column])
        representatives = np.asarray(orbit.representative_lifted_rows, dtype=np.int64)
        for quotient_row, representative in enumerate(representatives):
            width = int(widths[quotient_row])
            if width == 0:
                continue
            if int(orbit_rows[representative]) != quotient_row:
                raise ValueError(
                    "Periodic representative does not belong to its quotient entity."
                )
            global_ids = rows_by_entity[entity_degree][int(representative)]
            exact_action = exact_actions_by_entity[entity_degree][int(representative)]
            if global_ids is None or exact_action is None:
                raise ValueError(
                    "Periodic representative has no exact FE entity coordinates."
                )
            identity = tuple(
                tuple(Fraction(int(row == column)) for column in range(width))
                for row in range(width)
            )
            inverse_exact = solve_exact(exact_action, identity)
            inverse_action = np.asarray(inverse_exact, dtype=np.float64)
            if any(
                coefficient != 0 and inverse_action[row, column] == 0.0
                for row, values in enumerate(inverse_exact)
                for column, coefficient in enumerate(values)
            ):
                raise ValueError(
                    "Periodic exact collapse action is not representable in binary64."
                )
            local_row, local_column = np.nonzero(inverse_action)
            if local_row.size:
                collapse_rows.append(degree_offset + offsets[quotient_row] + local_row)
                collapse_columns.append(global_ids[local_column])
                collapse_values.append(inverse_action[local_row, local_column])

    quotient_count = sum(quotient_counts)
    cell_orbit = orbit_maps[complex.dimension]
    cell_orbit_rows = np.asarray(cell_orbit.lifted_to_quotient, dtype=np.int64)
    covering_counts = np.bincount(
        cell_orbit_rows,
        minlength=cell_orbit.quotient_entity_ids.shape[0],
    )
    if (
        covering_counts.size == 0
        or np.any(covering_counts == 0)
        or np.unique(covering_counts).size != 1
    ):
        raise ValueError(
            "Periodic FE quotient requires a uniformly covered cell complex."
        )
    covering_degree = int(covering_counts[0])
    quotient_space_id = canonical_fingerprint(
        {
            "kind": "periodic-quotient-fe-coordinate-space",
            "periodic_topology": periodic.periodic_topology_id,
            "degree": degree,
            "orbit_maps": tuple(orbit.orbit_map_id for orbit in orbit_maps),
            "quotient_entity_ids": tuple(
                array_tree_fingerprint(values) for values in quotient_ids
            ),
            "entity_widths": tuple(
                array_tree_fingerprint(values) for values in quotient_widths
            ),
            "covering_degree": covering_degree,
            "elements": tuple(element.element_id for element in elements),
            "exact_entity_actions": tuple(
                tuple(
                    None
                    if action is None
                    else tuple(
                        tuple((value.numerator, value.denominator) for value in row)
                        for row in action
                    )
                    for action in level
                )
                for level in exact_actions_by_entity
            ),
        }
    )
    del rows_by_entity, actions_by_entity, exact_actions_by_entity
    del entity_dof_ids, entity_actions, entity_exact_actions, rows, actions, exact_actions
    lift_target_space = complex._absolute.space(degree)
    lift_matrix = scipy_sparse.coo_matrix(
        (
            np.concatenate(lift_values) if lift_values else np.zeros((0,)),
            (
                np.concatenate(lift_rows)
                if lift_rows
                else np.zeros((0,), dtype=np.int64),
                np.concatenate(lift_columns)
                if lift_columns
                else np.zeros((0,), dtype=np.int64),
            ),
        ),
        shape=(global_count, quotient_count),
    ).tocsr()
    collapse_matrix = scipy_sparse.coo_matrix(
        (
            np.concatenate(collapse_values) if collapse_values else np.zeros((0,)),
            (
                np.concatenate(collapse_rows)
                if collapse_rows
                else np.zeros((0,), dtype=np.int64),
                np.concatenate(collapse_columns)
                if collapse_columns
                else np.zeros((0,), dtype=np.int64),
            ),
        ),
        shape=(quotient_count, global_count),
    ).tocsr()
    del lift_rows, lift_columns, lift_values
    del collapse_rows, collapse_columns, collapse_values
    from .._gram import sparse_gram_space

    lifted_space = _require_array_space(lift_target_space)
    lifted_gram = getattr(lifted_space.pairing, "operator", None)
    if not isinstance(lifted_gram, SparseCoordinateOperator):
        raise TypeError(
            "Periodic FE quotients require the endpoint's physical Hodge metric."
        )
    quotient_gram = (lift_matrix.T @ _periodic_matrix(lifted_gram) @ lift_matrix).tocsr()
    quotient_gram = ((quotient_gram + quotient_gram.T) * 0.5).tocsr()
    quotient_gram /= covering_degree
    gram_coo = quotient_gram.tocoo()
    gram_keep = np.asarray(gram_coo.data) != 0
    gram_rows = np.asarray(gram_coo.row[gram_keep], dtype=np.int32)
    gram_columns = np.asarray(gram_coo.col[gram_keep], dtype=np.int32)
    gram_values = np.asarray(gram_coo.data[gram_keep])
    del quotient_gram, gram_coo, gram_keep
    lift_source_space, _ = sparse_gram_space(
        gram_rows,
        gram_columns,
        gram_values,
        size=quotient_count,
        dtype=lifted_space.dtype,
        space_id=quotient_space_id,
        gram_tolerance=1.0e-12,
    )
    _require_sparse_identity(
        collapse_matrix @ lift_matrix,
        scipy_sparse.identity(quotient_count, format="csr"),
        "Periodic quotient lift/collapse identity",
    )
    boundary_parts = []
    for entity_degree, widths in enumerate(quotient_widths):
        entity_boundary = np.asarray(
            periodic.quotient.entity_sets[entity_degree].subset("boundary").mask,
            dtype=np.bool_,
        )
        if entity_boundary.shape != widths.shape:
            raise ValueError(
                "Quotient boundary masks do not align with FE entity orbits."
            )
        boundary_parts.append(np.repeat(entity_boundary, widths))
    quotient_boundary = (
        np.concatenate(boundary_parts)
        if boundary_parts
        else np.zeros((0,), dtype=np.bool_)
    )
    if quotient_boundary.shape != (quotient_count,):
        raise ValueError("Quotient FE boundary mask does not align with its DOF space.")
    orbit_id = canonical_fingerprint(
        {
            "kind": "periodic-finite-element-coordinate-actions",
            "mesh": mesh.mesh_id,
            "degree": degree,
            "orbit_maps": tuple(orbit.orbit_map_id for orbit in orbit_maps),
        }
    )
    lift = _sparse_matrix_operator(
        lift_matrix,
        lift_source_space,
        lift_target_space,
        f"{orbit_id}:lift",
    )
    collapse = _sparse_matrix_operator(
        collapse_matrix,
        lift_target_space,
        lift_source_space,
        f"{orbit_id}:representative-collapse",
    )
    return (
        lift,
        collapse,
        tuple(orbit.orbit_map_id for orbit in orbit_maps),
        quotient_boundary,
    )


def _periodic_quotient_complex(
    complex: FiniteElementDeRhamComplex,
    lifts: tuple[SparseCoordinateOperator, ...],
    collapses: tuple[SparseCoordinateOperator, ...],
    identifier: str,
    /,
) -> tuple[HilbertComplex, tuple[float, ...], tuple[float, ...]]:
    from scipy import sparse as scipy_sparse

    spaces = tuple(lift.source for lift in lifts)
    differentials = []
    closure_defects = []
    quotient_matrices = []
    for degree in range(complex.dimension):
        lifted_differential = _periodic_matrix(complex._absolute.differential(degree))
        lifted_action = _periodic_matrix(lifts[degree])
        collapse = _periodic_matrix(collapses[degree + 1])
        quotient_differential = collapse @ lifted_differential @ lifted_action
        closure = (
            lifted_differential @ lifted_action
            - _periodic_matrix(lifts[degree + 1]) @ quotient_differential
        )
        defect, scale = _sparse_defect(
            closure,
            scipy_sparse.csr_matrix(closure.shape, dtype=closure.dtype),
        )
        if not np.isfinite(defect) or defect > 1.0e-10 * scale:
            raise ValueError(
                "Periodic quotient is not a subcomplex of its lifted de Rham complex."
            )
        closure_defects.append(defect)
        quotient_matrices.append(quotient_differential.tocsr())
        differentials.append(
            _sparse_matrix_operator(
                quotient_differential,
                spaces[degree],
                spaces[degree + 1],
                f"{identifier}:d:{degree}",
            )
        )
    nilpotency_defects = []
    for degree in range(max(0, complex.dimension - 1)):
        product = quotient_matrices[degree + 1] @ quotient_matrices[degree]
        defect, scale = _sparse_defect(
            product,
            scipy_sparse.csr_matrix(product.shape, dtype=product.dtype),
        )
        if not np.isfinite(defect) or defect > 1.0e-10 * scale:
            raise ValueError("Periodic quotient differentials do not form a complex.")
        nilpotency_defects.append(defect)
    quotient = HilbertComplex(
        spaces,
        tuple(differentials),
        complex_id=identifier,
    )
    return quotient, tuple(closure_defects), tuple(nilpotency_defects)


def _relative_quotient_complex(
    full: HilbertComplex,
    boundary_masks: tuple[np.ndarray, ...],
    identifier: str,
    /,
) -> tuple[HilbertComplex, tuple[tuple[int, ...], ...]]:
    from scipy import sparse as scipy_sparse

    from ...linalg._operator_pairing import OperatorPairing
    from .._gram import sparse_gram_space

    if len(boundary_masks) != len(full.spaces):
        raise ValueError("Periodic quotient boundary masks must cover every form degree.")
    active = []
    spaces = []
    for degree, (space, mask) in enumerate(zip(full.spaces, boundary_masks, strict=True)):
        if not isinstance(space, ArraySpace):
            raise TypeError("Periodic quotient forms require array coefficient spaces.")
        host_mask = np.asarray(mask, dtype=np.bool_)
        if host_mask.shape != (space.size,):
            raise ValueError("Periodic quotient boundary mask has the wrong DOF extent.")
        indices = np.flatnonzero(~host_mask).astype(np.int32)
        active.append(tuple(int(value) for value in indices))
        pairing = space.pairing
        if not isinstance(pairing, OperatorPairing):
            raise TypeError("Periodic quotient restriction requires operator pairings.")
        gram = _periodic_matrix(pairing.operator)
        restricted = gram[indices][:, indices].tocsr()
        coo = restricted.tocoo()
        keep = np.asarray(coo.data) != 0
        restricted_space, _ = sparse_gram_space(
            np.asarray(coo.row[keep], dtype=np.int32),
            np.asarray(coo.col[keep], dtype=np.int32),
            np.asarray(coo.data[keep]),
            size=indices.size,
            dtype=space.dtype,
            space_id=f"{identifier}:relative:{degree}",
        )
        spaces.append(restricted_space)
    derivatives = []
    derivative_matrices = []
    for degree in range(full.top_degree):
        operator = _periodic_matrix(full.differential(degree))
        source_indices = np.asarray(active[degree], dtype=np.int64)
        target_indices = np.asarray(active[degree + 1], dtype=np.int64)
        source_active = np.zeros((full.space(degree).size,), dtype=np.bool_)
        target_active = np.zeros((full.space(degree + 1).size,), dtype=np.bool_)
        source_active[source_indices] = True
        target_active[target_indices] = True
        boundary_rows = np.flatnonzero(~target_active)
        leak = operator[boundary_rows][:, source_indices].tocsr()
        leak.eliminate_zeros()
        scale = max(
            1.0,
            0.0 if operator.nnz == 0 else float(np.max(np.abs(operator.data))),
        )
        leak_defect = 0.0 if leak.nnz == 0 else float(np.max(np.abs(leak.data)))
        if not np.isfinite(leak_defect) or leak_defect > 1.0e-10 * scale:
            raise ValueError(
                "Periodic quotient differential does not preserve relative boundary constraints."
            )
        restricted = operator[target_indices][:, source_indices].tocsr()
        derivatives.append(
            _sparse_matrix_operator(
                restricted,
                spaces[degree],
                spaces[degree + 1],
                f"{identifier}:relative:d:{degree}",
            )
        )
        derivative_matrices.append(restricted)
    for degree in range(max(0, full.top_degree - 1)):
        product = derivative_matrices[degree + 1] @ derivative_matrices[degree]
        _require_sparse_identity(
            product,
            scipy_sparse.csr_matrix(product.shape, dtype=product.dtype),
            "Periodic relative quotient nilpotency",
        )
    return (
        HilbertComplex(
            tuple(spaces),
            tuple(derivatives),
            complex_id=f"{identifier}:relative",
        ),
        tuple(active),
    )


def _apply_coordinate_payload(
    operator: SparseCoordinateOperator, values: ArrayLike, /
) -> Array:
    array = jnp.asarray(values)
    if array.ndim == 0 or array.shape[0] != operator.source.size:
        raise ValueError("Coordinate-map payloads must align with source coefficients.")
    if array.ndim == 1:
        return operator.mv(array)
    mapped = operator.mv_block(array.reshape((array.shape[0], -1)))
    return mapped.reshape((operator.target.size,) + array.shape[1:])


@final
class SparseTransferEntryEvidence(StrictModule):
    """Certified per-entry integration bounds for one FE degree map."""

    degree: int = eqx.field(static=True)
    source_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    target_indices: Array
    source_indices: Array
    absolute_error_bounds: Array
    exact_integral_counts: Array
    enclosed_integral_counts: Array

    def __init__(
        self,
        degree: int,
        source_size: int,
        target_size: int,
        target_indices: ConvertibleToArray,
        source_indices: ConvertibleToArray,
        absolute_error_bounds: ConvertibleToArray,
        exact_integral_counts: ConvertibleToArray,
        enclosed_integral_counts: ConvertibleToArray,
        /,
    ) -> None:
        rows = np.asarray(target_indices, dtype=np.int64)
        columns = np.asarray(source_indices, dtype=np.int64)
        errors = np.asarray(absolute_error_bounds, dtype=np.float64)
        exact = np.asarray(exact_integral_counts, dtype=np.int64)
        enclosed = np.asarray(enclosed_integral_counts, dtype=np.int64)
        if (
            isinstance(degree, bool)
            or degree < 0
            or source_size < 0
            or target_size < 0
            or rows.ndim != 1
            or columns.shape != rows.shape
            or errors.shape != rows.shape
            or exact.shape != rows.shape
            or enclosed.shape != rows.shape
            or np.any((rows < 0) | (rows >= target_size))
            or np.any((columns < 0) | (columns >= source_size))
            or np.any(errors < 0.0)
            or np.any(exact < 0)
            or np.any(enclosed < 0)
            or not np.all(np.isfinite(errors))
            or (rows.size and np.any(np.diff(rows * max(source_size, 1) + columns) <= 0))
        ):
            raise ValueError(
                "Sparse FE transfer-entry evidence is malformed or unordered."
            )
        self.degree, self.source_size, self.target_size = (
            int(degree),
            int(source_size),
            int(target_size),
        )
        self.target_indices = jnp.asarray(rows, dtype=jnp.int32)
        self.source_indices = jnp.asarray(columns, dtype=jnp.int32)
        self.absolute_error_bounds = jnp.asarray(errors, dtype=jnp.float64)
        self.exact_integral_counts = jnp.asarray(exact, dtype=jnp.int64)
        self.enclosed_integral_counts = jnp.asarray(enclosed, dtype=jnp.int64)


@final
class SparseCommutingEvidence(StrictModule):
    """Sparse coefficientwise residual intervals for commuting or constraints."""

    kind: Literal["commuting", "relative-boundary", "periodic-descent"] = eqx.field(
        static=True
    )
    degree: int = eqx.field(static=True)
    source_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    target_indices: Array
    source_indices: Array
    observed_defects: Array
    integration_error_bounds: Array
    roundoff_error_bounds: Array

    def __init__(
        self,
        kind: Literal["commuting", "relative-boundary", "periodic-descent"],
        degree: int,
        source_size: int,
        target_size: int,
        target_indices: ConvertibleToArray,
        source_indices: ConvertibleToArray,
        observed_defects: ConvertibleToArray,
        integration_error_bounds: ConvertibleToArray,
        roundoff_error_bounds: ConvertibleToArray,
        /,
    ) -> None:
        rows = np.asarray(target_indices, dtype=np.int64)
        columns = np.asarray(source_indices, dtype=np.int64)
        defects = np.asarray(observed_defects)
        if defects.dtype.kind not in "fc":
            raise TypeError("Observed FE residuals must be real or complex numbers.")
        integration = np.asarray(integration_error_bounds, dtype=np.float64)
        roundoff = np.asarray(roundoff_error_bounds, dtype=np.float64)
        if (
            rows.ndim == 1
            and columns.shape == rows.shape
            and defects.shape == rows.shape
            and integration.shape == rows.shape
            and roundoff.shape == rows.shape
        ):
            order = np.lexsort((columns, rows))
            rows, columns = rows[order], columns[order]
            defects, integration, roundoff = (
                value[order] for value in (defects, integration, roundoff)
            )
        if (
            kind not in ("commuting", "relative-boundary", "periodic-descent")
            or isinstance(degree, bool)
            or degree < 0
            or source_size < 0
            or target_size < 0
            or rows.ndim != 1
            or columns.shape != rows.shape
            or defects.shape != rows.shape
            or integration.shape != rows.shape
            or roundoff.shape != rows.shape
            or np.any((rows < 0) | (rows >= target_size))
            or np.any((columns < 0) | (columns >= source_size))
            or np.any(integration < 0.0)
            or np.any(roundoff < 0.0)
            or not all(
                np.all(np.isfinite(value)) for value in (defects, integration, roundoff)
            )
            or (rows.size and np.any(np.diff(rows * max(source_size, 1) + columns) <= 0))
        ):
            raise ValueError("Sparse FE transfer residual evidence is malformed.")
        self.kind, self.degree, self.source_size, self.target_size = (
            kind,
            int(degree),
            int(source_size),
            int(target_size),
        )
        self.target_indices = jnp.asarray(rows, dtype=jnp.int32)
        self.source_indices = jnp.asarray(columns, dtype=jnp.int32)
        self.observed_defects = jnp.asarray(defects)
        self.integration_error_bounds = jnp.asarray(integration, dtype=jnp.float64)
        self.roundoff_error_bounds = jnp.asarray(roundoff, dtype=jnp.float64)


@final
class FiniteElementTransferResult(StrictModule):
    """A FE chain action with sparse integration and commuting evidence.

    Rational chart integrals are bounded per sparse map entry. Commuting
    residuals retain their observed center, propagated integral interval, and
    floating arithmetic bound; the wrapped ``ComplexMap`` alone does not claim
    exact commutation.
    """

    complex_map: ComplexMap
    entry_evidence: tuple[SparseTransferEntryEvidence, ...]
    commuting_evidence: tuple[SparseCommutingEvidence, ...]
    constraint_evidence: tuple[SparseCommutingEvidence, ...]
    integration_policy: CurvedOverlapIntegrationPolicy | None
    exact_integral_count: int = eqx.field(static=True)
    enclosed_integral_count: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex_map: ComplexMap,
        entry_evidence: Sequence[SparseTransferEntryEvidence],
        commuting_evidence: Sequence[SparseCommutingEvidence],
        integration_policy: CurvedOverlapIntegrationPolicy | None,
        exact_integral_count: int,
        enclosed_integral_count: int,
        /,
        *,
        constraint_evidence: Sequence[SparseCommutingEvidence] = (),
        expected_evidence_id: str | None = None,
    ) -> None:
        if not isinstance(complex_map, ComplexMap):
            raise TypeError("complex_map must be a ComplexMap.")
        entries, commuting, constraints = (
            tuple(entry_evidence),
            tuple(commuting_evidence),
            tuple(constraint_evidence),
        )
        if (
            len(entries) != len(complex_map.maps)
            or len(commuting) != max(0, len(complex_map.maps) - 1)
            or min(exact_integral_count, enclosed_integral_count) < 0
            or any(
                not isinstance(value, SparseTransferEntryEvidence) for value in entries
            )
            or any(
                not isinstance(value, SparseCommutingEvidence)
                or value.kind != "commuting"
                for value in commuting
            )
            or any(
                not isinstance(value, SparseCommutingEvidence)
                or value.kind not in ("relative-boundary", "periodic-descent")
                for value in constraints
            )
        ):
            raise ValueError("Finite-element transfer result evidence is incomplete.")
        if integration_policy is not None:
            from ...geometry._common_root_volume_overlap import (
                CurvedOverlapIntegrationPolicy,
            )

            if not isinstance(integration_policy, CurvedOverlapIntegrationPolicy):
                raise TypeError(
                    "integration_policy must be CurvedOverlapIntegrationPolicy or None."
                )
        sparse_operators: list[SparseCoordinateOperator] = []
        for index, (operator, evidence) in enumerate(
            zip(complex_map.maps, entries, strict=True)
        ):
            degree = complex_map.first_degree + index
            if not isinstance(operator, SparseCoordinateOperator):
                raise TypeError("FE transfer maps must use sparse coordinate operators.")
            if operator.block_shape is not None:
                raise TypeError("FE transfer maps require scalar sparse coordinates.")
            relation = operator.relation
            if not isinstance(relation, EdgeRelation):
                raise TypeError("FE transfer maps require edge-relation storage.")
            source_space = _require_array_space(operator.source)
            target_space = _require_array_space(operator.target)
            if (
                evidence.degree != degree
                or evidence.source_size != source_space.size
                or evidence.target_size != target_space.size
            ):
                raise ValueError(
                    "FE transfer-entry evidence does not match its degree map."
                )
            valid = np.asarray(relation.valid, dtype=np.bool_)
            keys = (
                np.asarray(relation.target_indices, dtype=np.int64)[valid]
                * max(source_space.size, 1)
                + np.asarray(relation.source_indices, dtype=np.int64)[valid]
            )
            expected = np.asarray(evidence.target_indices, dtype=np.int64) * max(
                source_space.size, 1
            ) + np.asarray(evidence.source_indices, dtype=np.int64)
            if not np.array_equal(np.sort(keys), expected):
                raise ValueError(
                    "FE transfer-entry evidence does not align sparse map storage."
                )
            sparse_operators.append(operator)
        if any(
            value.degree != complex_map.first_degree + index
            for index, value in enumerate(commuting)
        ):
            raise ValueError(
                "FE commuting evidence degrees do not match the complex map."
            )
        for value in (*commuting, *constraints):
            if value.degree < 0:
                raise ValueError("FE residual evidence degree must be nonnegative.")
        policy_id = None if integration_policy is None else integration_policy.policy_id
        entry_payload = tuple(
            (
                value.degree,
                array_tree_fingerprint(np.asarray(value.target_indices)),
                array_tree_fingerprint(np.asarray(value.source_indices)),
                array_tree_fingerprint(np.asarray(value.absolute_error_bounds)),
                array_tree_fingerprint(np.asarray(value.exact_integral_counts)),
                array_tree_fingerprint(np.asarray(value.enclosed_integral_counts)),
            )
            for value in entries
        )

        def operator_properties_payload(operator: SparseCoordinateOperator, /) -> str:
            properties = operator.properties
            return canonical_fingerprint(
                {
                    "diagonal": properties.diagonal,
                    "triangular": properties.triangular,
                    "self_adjoint": properties.self_adjoint,
                    "positive_definite": properties.positive_definite,
                    "positive_semidefinite": properties.positive_semidefinite,
                    "block_diagonal": properties.block_diagonal,
                    "rank": properties.rank,
                    "evidence": properties.evidence,
                }
            )

        def pairing_payload(
            space: ArraySpace,
            visited_operators: frozenset[int] = frozenset(),
            /,
        ) -> tuple[str, ...]:
            from ...linalg._distributed import DistributedPairing
            from ...linalg._operator_pairing import OperatorPairing
            from ...linalg._pairings import DiagonalPairing, EuclideanPairing

            pairing = space.pairing
            payload = [type(pairing).__qualname__, pairing.pairing_id]
            if isinstance(pairing, DiagonalPairing):
                payload.append(
                    canonical_fingerprint(array_tree_fingerprint(pairing.weights))
                )
            elif isinstance(pairing, OperatorPairing):
                if not isinstance(pairing.operator, SparseCoordinateOperator):
                    raise TypeError(
                        "FE transfer operator pairings require sparse coordinate operators."
                    )
                payload.extend(
                    sparse_operator_payload(pairing.operator, visited_operators)
                )
            elif isinstance(pairing, DistributedPairing):
                payload.append(
                    canonical_fingerprint(array_tree_fingerprint(pairing.owned_mask))
                )
                if pairing.weights is not None:
                    payload.append(
                        canonical_fingerprint(array_tree_fingerprint(pairing.weights))
                    )
            elif not isinstance(pairing, EuclideanPairing):
                raise TypeError(
                    "FE transfer identity does not support this vector pairing type."
                )
            return tuple(payload)

        def sparse_operator_payload(
            operator: SparseCoordinateOperator,
            visited_operators: frozenset[int] = frozenset(),
            /,
        ) -> tuple[str, ...]:
            if not isinstance(operator, SparseCoordinateOperator):
                raise TypeError(
                    "FE transfer identity requires authored sparse coordinate operators."
                )
            if id(operator) in visited_operators:
                return ("operator-reference", operator.operator_id)
            nested_operators = visited_operators | {id(operator)}
            source_space = _require_array_space(operator.source)
            target_space = _require_array_space(operator.target)
            relation = operator.relation
            if isinstance(relation, RowRelation):
                relation = relation.as_edge_relation()
            return (
                operator.operator_id,
                source_space.space_id,
                target_space.space_id,
                canonical_fingerprint(
                    array_tree_fingerprint(np.asarray(relation.source_indices))
                ),
                canonical_fingerprint(
                    array_tree_fingerprint(np.asarray(relation.target_indices))
                ),
                canonical_fingerprint(array_tree_fingerprint(np.asarray(relation.valid))),
                canonical_fingerprint(
                    array_tree_fingerprint(np.asarray(operator.coefficients))
                ),
                operator_properties_payload(operator),
                np.dtype(operator.accumulation_dtype).str,
                canonical_fingerprint(operator.block_shape),
                *pairing_payload(source_space, nested_operators),
                *pairing_payload(target_space, nested_operators),
            )

        def sparse_map_payload(operator: SparseCoordinateOperator, /) -> tuple[str, ...]:
            if not isinstance(operator.relation, EdgeRelation):
                raise TypeError("FE transfer evidence requires edge-relation routes.")
            return sparse_operator_payload(operator)

        map_payload = tuple(sparse_map_payload(operator) for operator in sparse_operators)
        residual_payload = tuple(
            (
                value.kind,
                value.degree,
                array_tree_fingerprint(np.asarray(value.target_indices)),
                array_tree_fingerprint(np.asarray(value.source_indices)),
                array_tree_fingerprint(np.asarray(value.observed_defects)),
                array_tree_fingerprint(np.asarray(value.integration_error_bounds)),
                array_tree_fingerprint(np.asarray(value.roundoff_error_bounds)),
            )
            for value in (*commuting, *constraints)
        )
        identity = canonical_fingerprint(
            {
                "kind": "finite-element-transfer-result",
                "complex_map": complex_map.map_id,
                "sparse_map_payload": map_payload,
                "entry_evidence": entry_payload,
                "residual_evidence": residual_payload,
                "integration_policy": policy_id,
                "integral_counts": (
                    int(exact_integral_count),
                    int(enclosed_integral_count),
                ),
            }
        )
        if expected_evidence_id is not None and expected_evidence_id != identity:
            raise ValueError(
                "Stored FE transfer evidence identity does not match its contents."
            )
        self.complex_map = complex_map
        self.entry_evidence = entries
        self.commuting_evidence = commuting
        self.constraint_evidence = constraints
        self.integration_policy = integration_policy
        self.exact_integral_count = int(exact_integral_count)
        self.enclosed_integral_count = int(enclosed_integral_count)
        self.transfer_id = complex_map.map_id
        self.evidence_id = identity

    @property
    def commuting_defects(self) -> Array:
        return jnp.asarray(
            [
                float(np.max(np.abs(np.asarray(value.observed_defects)), initial=0.0))
                for value in self.commuting_evidence
            ],
            dtype=jnp.float64,
        )

    @property
    def commuting_error_bounds(self) -> Array:
        return jnp.asarray(
            [
                float(
                    np.max(
                        np.asarray(value.integration_error_bounds)
                        + np.asarray(value.roundoff_error_bounds),
                        initial=0.0,
                    )
                )
                for value in self.commuting_evidence
            ],
            dtype=jnp.float64,
        )

    def support_report(self, /) -> tuple[SparseSupportRank, ...]:
        """Report execution support and rigorous sparse rank bounds by form degree."""
        from ._topology_transfer import _sparse_transfer_support

        return tuple(
            _sparse_transfer_support(operator) for operator in self.complex_map.maps
        )


@final
class PeriodicFiniteElementTransfer(StrictModule):
    """Validated de Rham transfer on the actual periodic quotient complexes."""

    complex_map: ComplexMap
    lifted_transfer_result: FiniteElementTransferResult
    quotient_transfer_result: FiniteElementTransferResult
    source_lifts: tuple[SparseCoordinateOperator, ...]
    source_collapses: tuple[SparseCoordinateOperator, ...]
    target_lifts: tuple[SparseCoordinateOperator, ...]
    target_collapses: tuple[SparseCoordinateOperator, ...]
    source_orbit_map_ids: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    target_orbit_map_ids: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_active_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    target_active_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    boundary: ComplexBoundary = eqx.field(static=True)
    source_closure_defects: Array
    target_closure_defects: Array
    source_nilpotency_defects: Array
    target_nilpotency_defects: Array
    deck_descent_defects: Array
    deck_descent_error_bounds: Array
    commuting_error_bounds: Array
    commuting_defects: Array
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex_map: ComplexMap,
        lifted_transfer_result: FiniteElementTransferResult,
        quotient_transfer_result: FiniteElementTransferResult,
        source_lifts: tuple[SparseCoordinateOperator, ...],
        source_collapses: tuple[SparseCoordinateOperator, ...],
        target_lifts: tuple[SparseCoordinateOperator, ...],
        target_collapses: tuple[SparseCoordinateOperator, ...],
        source_orbit_map_ids: tuple[tuple[str, ...], ...],
        target_orbit_map_ids: tuple[tuple[str, ...], ...],
        source_active_indices: tuple[tuple[int, ...], ...],
        target_active_indices: tuple[tuple[int, ...], ...],
        boundary: ComplexBoundary,
        source_closure_defects: ConvertibleToArray,
        target_closure_defects: ConvertibleToArray,
        source_nilpotency_defects: ConvertibleToArray,
        target_nilpotency_defects: ConvertibleToArray,
        deck_descent_defects: ConvertibleToArray,
        deck_descent_error_bounds: ConvertibleToArray,
        commuting_defects: ConvertibleToArray,
        commuting_error_bounds: ConvertibleToArray,
        transfer_id: str,
        /,
    ) -> None:
        if (
            not isinstance(lifted_transfer_result, FiniteElementTransferResult)
            or not isinstance(quotient_transfer_result, FiniteElementTransferResult)
            or quotient_transfer_result.complex_map.map_id != complex_map.map_id
        ):
            raise ValueError(
                "Periodic FE transfer must retain lifted and quotient evidence."
            )
        self.complex_map = complex_map
        self.lifted_transfer_result = lifted_transfer_result
        self.quotient_transfer_result = quotient_transfer_result
        self.source_lifts, self.source_collapses = source_lifts, source_collapses
        self.target_lifts, self.target_collapses = target_lifts, target_collapses
        self.source_orbit_map_ids, self.target_orbit_map_ids = (
            source_orbit_map_ids,
            target_orbit_map_ids,
        )
        self.source_active_indices = source_active_indices
        self.target_active_indices = target_active_indices
        self.boundary = boundary
        self.source_closure_defects = jnp.asarray(
            source_closure_defects, dtype=jnp.float64
        )
        self.target_closure_defects = jnp.asarray(
            target_closure_defects, dtype=jnp.float64
        )
        self.source_nilpotency_defects = jnp.asarray(
            source_nilpotency_defects, dtype=jnp.float64
        )
        self.target_nilpotency_defects = jnp.asarray(
            target_nilpotency_defects, dtype=jnp.float64
        )
        self.deck_descent_defects = jnp.asarray(deck_descent_defects, dtype=jnp.float64)
        self.commuting_defects = jnp.asarray(commuting_defects, dtype=jnp.float64)
        self.transfer_id = transfer_id
        self.deck_descent_error_bounds = jnp.asarray(
            deck_descent_error_bounds, dtype=jnp.float64
        )

        self.commuting_error_bounds = jnp.asarray(
            commuting_error_bounds, dtype=jnp.float64
        )

    def map(self, degree: int, /) -> AbstractLinearOperator:
        return self.complex_map.map(degree)

    def support_report(self, /) -> tuple[SparseSupportRank, ...]:
        """Report sparse rank bounds and support on the periodic quotient."""
        from ._topology_transfer import _sparse_transfer_support

        return tuple(
            _sparse_transfer_support(operator) for operator in self.complex_map.maps
        )

    def lift_source(self, degree: int, values: ArrayLike, /) -> Array:
        operator = self.source_lifts[degree]
        array = jnp.asarray(values)
        if self.boundary == "relative":
            active = jnp.asarray(self.source_active_indices[degree], dtype=jnp.int32)
            if array.ndim == 0 or array.shape[0] != active.size:
                raise ValueError("Relative quotient source values have the wrong extent.")
            full = jnp.zeros((operator.source.size,) + array.shape[1:], dtype=array.dtype)
            full = full.at[active].set(array)
        else:
            full = array
        return _apply_coordinate_payload(operator, full)

    def lift_target(self, degree: int, values: ArrayLike, /) -> Array:
        operator = self.target_lifts[degree]
        array = jnp.asarray(values)
        if self.boundary == "relative":
            active = jnp.asarray(self.target_active_indices[degree], dtype=jnp.int32)
            if array.ndim == 0 or array.shape[0] != active.size:
                raise ValueError("Relative quotient target values have the wrong extent.")
            full = jnp.zeros((operator.source.size,) + array.shape[1:], dtype=array.dtype)
            full = full.at[active].set(array)
        else:
            full = array
        return _apply_coordinate_payload(operator, full)

    def collapse_source(self, degree: int, values: ArrayLike, /) -> Array:
        result = _apply_coordinate_payload(
            self.source_collapses[degree], jnp.asarray(values)
        )
        if self.boundary == "relative":
            return result[
                jnp.asarray(self.source_active_indices[degree], dtype=jnp.int32)
            ]
        return result

    def collapse_target(self, degree: int, values: ArrayLike, /) -> Array:
        result = _apply_coordinate_payload(
            self.target_collapses[degree], jnp.asarray(values)
        )
        if self.boundary == "relative":
            return result[
                jnp.asarray(self.target_active_indices[degree], dtype=jnp.int32)
            ]
        return result

    def distributed(
        self,
        source_owners: Sequence[ConvertibleToArray],
        target_owners: Sequence[ConvertibleToArray],
        part_count: int,
        /,
    ) -> DistributedFiniteElementTransfer:
        return _distribute_transfer_result(
            self.quotient_transfer_result, source_owners, target_owners, part_count
        )


@final
class DistributedFiniteElementTransfer(StrictModule):
    """Owner-local sparse actions retaining the FE transfer evidence."""

    transfer_result: FiniteElementTransferResult
    degree_maps: tuple[DistributedSparseTransfer, ...]
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer_result: FiniteElementTransferResult,
        degree_maps: tuple[DistributedSparseTransfer, ...],
        transfer_id: str,
        /,
    ) -> None:
        if not isinstance(transfer_result, FiniteElementTransferResult):
            raise TypeError("transfer_result must be FiniteElementTransferResult.")
        complex_map = transfer_result.complex_map
        if len(degree_maps) != len(complex_map.maps):
            raise ValueError(
                "Distributed degree maps must cover the complete FE complex."
            )
        if any(not isinstance(value, DistributedSparseTransfer) for value in degree_maps):
            raise TypeError(
                "Distributed FE degree actions require sparse transfer plans."
            )
        self.transfer_result = transfer_result
        self.degree_maps = degree_maps
        self.transfer_id = transfer_id

    @property
    def complex_map(self) -> ComplexMap:
        return self.transfer_result.complex_map

    def map(self, degree: int, /) -> DistributedSparseTransfer:
        index = degree - self.complex_map.first_degree
        if index < 0 or index >= len(self.degree_maps):
            raise ValueError("Requested degree is outside the distributed FE transfer.")
        return self.degree_maps[index]


def _distribute_transfer_result(
    transfer_result: FiniteElementTransferResult,
    source_owners: Sequence[ConvertibleToArray],
    target_owners: Sequence[ConvertibleToArray],
    part_count: int,
    /,
) -> DistributedFiniteElementTransfer:
    if not isinstance(transfer_result, FiniteElementTransferResult):
        raise TypeError("Distributed FE conversion requires FiniteElementTransferResult.")
    complex_map = transfer_result.complex_map
    if complex_map.degree_offset != 0 or complex_map.first_degree != 0:
        raise ValueError("Distributed de Rham transfer requires aligned degree maps.")
    source_layouts, target_layouts = tuple(source_owners), tuple(target_owners)
    if len(source_layouts) != len(complex_map.maps) or len(target_layouts) != len(
        complex_map.maps
    ):
        raise ValueError("Distributed DOF ownership must be supplied per FE degree.")
    maps = []
    for degree, operator in enumerate(complex_map.maps):
        if (
            not isinstance(operator, SparseCoordinateOperator)
            or operator.block_shape is not None
        ):
            raise TypeError(
                "Distributed FE transfer requires scalar sparse coordinate maps in every degree."
            )
        maps.append(
            DistributedSparseTransfer(
                operator.relation,
                operator.coefficients,
                source_layouts[degree],
                target_layouts[degree],
                part_count,
                operator_id=operator.operator_id,
            )
        )
    transfer_id = canonical_fingerprint(
        {
            "kind": "distributed-finite-element-complex-transfer",
            "finite_element_transfer": transfer_result.evidence_id,
            "degree_maps": tuple(value.transfer_id for value in maps),
        }
    )
    return DistributedFiniteElementTransfer(
        transfer_result,
        tuple(maps),
        transfer_id,
    )


def _coalesce_entries(
    rows: np.ndarray, columns: np.ndarray, values: Array, size: int, /
) -> tuple[np.ndarray, np.ndarray, Array]:
    keys = rows.astype(np.int64) * size + columns
    unique, inverse = np.unique(keys, return_inverse=True)
    summed = (
        jnp.zeros((unique.size,), dtype=values.dtype).at[jnp.asarray(inverse)].add(values)
    )
    return (unique // size).astype(np.int32), (unique % size).astype(np.int32), summed


def _coalesce_transfer_entry_evidence(
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    exact: np.ndarray,
    enclosed: np.ndarray,
    size: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = rows.astype(np.int64) * size + columns.astype(np.int64)
    unique, inverse = np.unique(keys, return_inverse=True)
    if not (
        columns.shape
        == values.shape
        == errors.shape
        == exact.shape
        == enclosed.shape
        == rows.shape
    ):
        raise ValueError("FE sparse values and integration evidence are misaligned.")
    error_sums = np.zeros((unique.size,), dtype=np.float64)
    exact_sums = np.zeros((unique.size,), dtype=np.int64)
    enclosed_sums = np.zeros((unique.size,), dtype=np.int64)
    magnitude_sums = np.zeros((unique.size,), dtype=np.float64)
    multiplicities = np.zeros((unique.size,), dtype=np.int64)
    np.add.at(error_sums, inverse, errors)
    np.add.at(exact_sums, inverse, exact)
    np.add.at(enclosed_sums, inverse, enclosed)
    np.add.at(magnitude_sums, inverse, np.abs(values) + errors)
    np.add.at(multiplicities, inverse, 1)
    epsilon = np.finfo(np.float64).eps
    operations = np.maximum(multiplicities - 1, 0).astype(np.float64) * epsilon
    gamma = np.where(operations < 1.0, operations / (1.0 - operations), np.inf)
    bounds = error_sums + gamma * magnitude_sums
    positive = bounds > 0.0
    bounds[positive] = np.nextafter(bounds[positive], np.inf)
    return (
        (unique // size).astype(np.int32) if size else np.zeros((unique.size,), np.int32),
        (unique % size).astype(np.int32) if size else np.zeros((unique.size,), np.int32),
        bounds,
        exact_sums,
        enclosed_sums,
    )


def _metric_hodge(
    discretization: FiniteElementDiscretization,
    degree: int,
    policy: LinearSolvePolicy | None,
    resources: SolveResourcePolicy | None,
    admission: SparseFactorizationPolicy,
    /,
) -> SparseHodge:
    dofs = discretization.dof_maps[degree]
    rows, columns, values = [], [], []
    for routes, transform, geometry in zip(
        dofs.cell_dofs,
        dofs.cell_transforms,
        discretization.block_geometries[degree],
        strict=True,
    ):
        local = _local_mass_tensor(geometry)
        local = contract("cai,cab,cbj->cij", transform, local, transform)
        indices = np.asarray(routes)
        first = np.broadcast_to(indices[:, :, None], local.shape)
        second = np.broadcast_to(indices[:, None, :], local.shape)
        keep = first <= second
        rows.append(first[keep])
        columns.append(second[keep])
        values.append(local[jnp.asarray(keep)])
    row, column, data = _coalesce_entries(
        np.concatenate(rows),
        np.concatenate(columns),
        jnp.concatenate(values),
        dofs.global_dof_count,
    )
    solve_policy = (
        policy
        if resources is None
        else gram_solve_policy(dofs.global_dof_count, resources=resources)
    )
    return SparseHodge(
        row,
        column,
        data,
        dofs.global_dof_count,
        policy=solve_policy,
        admission_policy=admission,
    )


def _local_derivatives(
    discretization: FiniteElementDiscretization, degree: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, target = discretization.dof_maps[degree : degree + 2]
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    owners: set[int] = set()
    vertex_ids = np.asarray(discretization.mesh.vertex_global_ids, dtype=np.int64)
    transform_cache: dict[
        tuple[str, tuple[int, ...], int],
        tuple[tuple[Fraction, ...], ...],
    ] = {}
    reference_gradient_cache: dict[str, np.ndarray] = {}
    local_cache: dict[
        tuple[
            str,
            str,
            tuple[tuple[Fraction, ...], ...],
            tuple[tuple[Fraction, ...], ...],
        ],
        tuple[tuple[Fraction, ...], ...],
    ] = {}
    for block, first, second, source_routes, target_routes in zip(
        discretization.mesh.blocks,
        discretization.elements[degree],
        discretization.elements[degree + 1],
        source.cell_dofs,
        target.cell_dofs,
        strict=True,
    ):
        first_basis, second_basis = first.form_basis, second.form_basis
        if first_basis is None or second_basis is None:
            raise ValueError(
                "An FE de Rham differential requires canonical polynomial forms."
            )
        exact_reference = first_basis.exact_exterior_derivative_matrix(second_basis)
        for cell in range(block.cell_count):
            source_transform = _exact_form_cell_transform(
                discretization.mesh,
                block,
                first,
                cell,
                vertex_ids,
                transform_cache,
                reference_gradient_cache,
            )
            target_transform = _exact_form_cell_transform(
                discretization.mesh,
                block,
                second,
                cell,
                vertex_ids,
                transform_cache,
                reference_gradient_cache,
            )
            cache_key = (
                first_basis.basis_id,
                second_basis.basis_id,
                source_transform,
                target_transform,
            )
            local = local_cache.get(cache_key)
            if local is None:
                local = tuple(
                    tuple(row)
                    for row in solve_exact(
                        target_transform,
                        _exact_matrix_product(exact_reference, source_transform),
                    )
                )
                local_cache[cache_key] = local
            source_cell_routes = np.asarray(source_routes[cell], dtype=np.int64)
            target_cell_routes = np.asarray(target_routes[cell], dtype=np.int64)
            for target_local, global_row in enumerate(target_cell_routes):
                if int(global_row) in owners:
                    continue
                owners.add(int(global_row))
                for source_local, coefficient in enumerate(local[target_local]):
                    if coefficient == 0:
                        continue
                    value = float(coefficient)
                    if not np.isfinite(value) or value == 0.0:
                        raise ValueError(
                            "Exact FE differential is not representable in binary64."
                        )
                    rows.append(int(global_row))
                    columns.append(int(source_cell_routes[source_local]))
                    values.append(value)
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
    )


def _complex_proxy(dimension: int, degree: int, /) -> FormProxy:
    """Choose the realization's physical proxy, including ambiguous endpoint degrees."""
    if degree == 0:
        return "scalar"
    if degree == dimension:
        return "density"
    if degree == 1:
        return "circulation"
    if degree == dimension - 1:
        return "flux"
    return "components"


def _complex_fields(
    mesh: CellMesh, family: FormElementFamily, order: int, twist: FormTwist, /
) -> tuple[FiniteElementFieldSpec, ...]:
    fields = []
    kinds = {block.cell_kind for block in mesh.blocks}
    mixed_trimmed = (
        family == "trimmed"
        and "prism" in kinds
        and bool(kinds & {"tetrahedron", "simplex:3"})
        and kinds <= {"prism", "tetrahedron", "simplex:3"}
    )
    for degree in range(mesh.topological_dimension + 1):
        polynomial_order = (
            order + mesh.topological_dimension - degree if family == "full" else order
        )
        proxy = _complex_proxy(mesh.topological_dimension, degree)
        elements = {
            block.name: form_element(
                block.cell_kind,
                degree,
                polynomial_order,
                family="prism-trimmed"
                if mixed_trimmed and block.cell_kind == "prism"
                else family,
                twist=twist,
                proxy=proxy,
            )
            for block in mesh.blocks
        }
        fields.append(FiniteElementFieldSpec(f"form_{degree}", elements))
    return tuple(fields)


def _absolute_fe_complex(
    discretization: FiniteElementDiscretization,
    hodges: tuple[SparseHodge, ...],
    coefficient_dtype: DTypeLike,
    family: FormElementFamily,
    order: int,
    identifier: str,
    /,
) -> HilbertComplex:
    spaces = tuple(
        hodge.make_space(
            space_id=f"{identifier}:absolute:{degree}", dtype=coefficient_dtype
        )[0]
        for degree, hodge in enumerate(hodges)
    )
    differentials = []
    # Topology incidences orient entities by vertex row while form DOFs orient by
    # ascending vertex global ID; the two agree only when IDs ascend with rows
    # (periodic rebinding, for one, reorders rows against their stable IDs).
    row_ordered_ids = bool(
        np.all(np.diff(np.asarray(discretization.mesh.vertex_global_ids)) > 0)
    )
    for degree in range(discretization.mesh.topological_dimension):
        if family in ("trimmed", "tensor-trimmed") and order == 1 and row_ordered_ids:
            incidence = discretization.mesh.topology.incidences[degree]
            relation = incidence.relation
            valid = np.asarray(relation.valid, dtype=np.bool_)
            rows = np.asarray(relation.target_indices)[valid]
            columns = np.asarray(relation.source_indices)[valid]
            values = incidence.signs[jnp.asarray(valid)]
        else:
            rows, columns, values = _local_derivatives(discretization, degree)
        differentials.append(
            _sparse_operator(
                rows,
                columns,
                values,
                spaces[degree],
                spaces[degree + 1],
                f"{identifier}:d:{degree}",
            )
        )
    return HilbertComplex(
        spaces, tuple(differentials), complex_id=f"{identifier}:absolute"
    )


def _boundary_entity_closure(
    topology: CellComplexTopology, boundary_mask: ArrayLike | None, /
) -> list[np.ndarray]:
    n = topology.dimension
    facet_boundary = np.asarray(topology.entity_sets[n - 1].subset("boundary").mask)
    selected = facet_boundary if boundary_mask is None else np.asarray(boundary_mask)
    if (
        selected.dtype != np.bool_
        or selected.shape != facet_boundary.shape
        or np.any(selected & ~facet_boundary)
    ):
        raise ValueError("boundary_mask must select topological boundary facets.")
    closure = [
        np.zeros((entities.count,), dtype=np.bool_) for entities in topology.entity_sets
    ]
    closure[n - 1] = selected
    for degree in range(n - 1, 0, -1):
        relation = topology.incidences[degree - 1].relation
        keep = (
            np.asarray(relation.valid)
            & closure[degree][np.asarray(relation.target_indices)]
        )
        closure[degree - 1][np.asarray(relation.source_indices)[keep]] = True
    return closure


def _record_boundary_cell(
    basis: FormBasis,
    entity_dofs: tuple[tuple[tuple[int, ...], ...], ...],
    cell_vertices: np.ndarray,
    cell_dofs: Array,
    cell: int,
    lookups: tuple[dict[tuple[int, ...], int], ...],
    closure: list[np.ndarray],
    mask: np.ndarray,
    orientation: np.ndarray,
    facet_signs: np.ndarray,
    degree: int,
    dimension: int,
    /,
) -> None:
    for entity_dimension, faces in enumerate(basis.entity_vertices):
        for face, local_dofs in zip(faces, entity_dofs[entity_dimension], strict=True):
            entity = lookups[entity_dimension][tuple(sorted(cell_vertices[list(face)]))]
            if closure[entity_dimension][entity]:
                mask[np.asarray(cell_dofs[cell])[list(local_dofs)]] = True
                if 0 < degree == dimension - 1 and entity_dimension == dimension - 1:
                    orientation[np.asarray(cell_dofs[cell])[list(local_dofs)]] = (
                        facet_signs[entity]
                    )


def _boundary_moment_selection(
    discretization: FiniteElementDiscretization,
    closure: list[np.ndarray],
    counts: tuple[int, ...],
    /,
) -> tuple[tuple[tuple[int, ...], ...], list[np.ndarray]]:
    from ._generic import _topology_vertex_sets

    mesh = discretization.mesh
    vertices = _topology_vertex_sets(mesh)
    lookups = tuple({face: i for i, face in enumerate(level)} for level in vertices)
    masks = [np.zeros((count,), dtype=np.bool_) for count in counts]
    orientation = [np.ones((count,), dtype=np.float64) for count in counts]
    incidence = mesh.topology.incidences[-1]
    facet_signs = np.zeros_like(closure[mesh.topological_dimension - 1], dtype=np.float64)
    valid = np.asarray(incidence.relation.valid)
    np.add.at(
        facet_signs,
        np.asarray(incidence.relation.source_indices)[valid],
        np.asarray(incidence.signs)[valid],
    )
    for degree, dofs in enumerate(discretization.dof_maps):
        for block_index, (block, element) in enumerate(
            zip(mesh.blocks, discretization.elements[degree], strict=True)
        ):
            basis = element.form_basis
            if basis is None:
                raise ValueError("FE trace requires canonical entity functionals.")
            for cell, cell_vertices in enumerate(np.asarray(block.vertices)):
                _record_boundary_cell(
                    basis,
                    element.entity_dofs,
                    cell_vertices,
                    dofs.cell_dofs[block_index],
                    cell,
                    lookups,
                    closure,
                    masks[degree],
                    orientation[degree],
                    facet_signs,
                    degree,
                    mesh.topological_dimension,
                )
    active = tuple(tuple(np.flatnonzero(mask).tolist()) for mask in masks)
    return active, orientation


def _oriented_boundary_complex(
    source: HilbertComplex,
    restricted: HilbertComplex,
    hodges: tuple[SparseHodge, ...],
    active: tuple[tuple[int, ...], ...],
    signs: tuple[Array, ...],
    identifier: str,
    /,
) -> HilbertComplex:
    spaces = []
    for degree, indices in enumerate(active[:-1]):
        hodge = hodges[degree].restrict(np.asarray(indices, dtype=np.int32))
        data = (
            hodge.upper_values
            * signs[degree][jnp.asarray(hodge.rows, dtype=jnp.int32)]
            * signs[degree][jnp.asarray(hodge.columns, dtype=jnp.int32)]
        )
        spaces.append(
            hodge.refresh(data).make_space(
                space_id=f"{identifier}:boundary:{degree}",
                dtype=_require_array_space(source.space(degree)).dtype,
            )[0]
        )
    derivatives = []
    for degree, derivative in enumerate(restricted.differentials[:-1]):
        if not isinstance(derivative, SparseCoordinateOperator):
            raise TypeError("FE boundary derivatives must be sparse coordinate maps.")
        relation = derivative.relation
        if not isinstance(relation, EdgeRelation):
            raise TypeError("FE boundary differentials require explicit edge relations.")
        data = (
            derivative.coefficients
            * signs[degree + 1][relation.target_indices]
            * signs[degree][relation.source_indices]
        )
        derivatives.append(
            _sparse_operator(
                np.asarray(relation.target_indices),
                np.asarray(relation.source_indices),
                data,
                spaces[degree],
                spaces[degree + 1],
                f"{identifier}:boundary:d:{degree}",
            )
        )
    return HilbertComplex(
        tuple(spaces), tuple(derivatives), complex_id=f"{identifier}:boundary"
    )


def _constitutive_admission(
    data: Array, cells: int, components: int, /
) -> tuple[Array, OperatorProperties]:
    allowed = ((), (cells,), (components, components), (cells, components, components))
    if data.shape not in allowed:
        raise ValueError(
            "Constitutive coefficient must be scalar, cell scalar, tensor, or cell tensor."
        )
    properties = OperatorProperties()
    if not jnp.issubdtype(data.dtype, jnp.complexfloating):
        if data.ndim <= 1:
            invalid = jnp.any(~jnp.isfinite(data) | (data <= 0))
        else:
            invalid = jnp.any(~jnp.isfinite(data)) | jnp.any(
                jnp.abs(data - jnp.swapaxes(data, -1, -2)) > 1e-12
            )
            invalid |= jnp.any(jnp.linalg.eigvalsh(data) <= 0)
        data = eqx.error_if(
            data,
            invalid,
            "Real constitutive coefficients must be symmetric positive definite.",
        )
        properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        )
    return data, properties


def _weighted_local_mass(
    geometry: FiniteElementBlockGeometry,
    data: Array,
    cells: int,
    components: int,
    offset: int,
    /,
) -> Array:
    table = geometry.basis_table
    if isinstance(table, _MappedBasisTable):
        # One-pass assembly of a deferred table in bounded cell blocks; the
        # weighted and tensor-applied rows are the row-sized intermediates.
        return jnp.concatenate(
            tuple(
                _weighted_mass_rows(
                    geometry.physical_weights[start:stop],
                    table.rows(start, stop),
                    data,
                    stop - start,
                    components,
                    offset + start,
                )
                for start, stop in table.row_blocks(2)
            ),
            axis=0,
        )
    return _weighted_mass_rows(
        geometry.physical_weights, table(), data, cells, components, offset
    )


def _weighted_mass_rows(
    physical_weights: Array,
    basis: Array,
    data: Array,
    cells: int,
    components: int,
    offset: int,
    /,
) -> Array:
    if basis.ndim == 2:
        basis = jnp.broadcast_to(basis[None, :, :, None], (cells, *basis.shape, 1))
    elif basis.ndim == 3:
        basis = basis[..., None]
    if data.ndim in (0, 1):
        weight = data if data.ndim == 0 else data[offset : offset + cells, None]
        return contract("cq,cqiv,cqjv->cij", physical_weights * weight, basis, basis)
    tensor = (
        jnp.broadcast_to(data, (cells, components, components))
        if data.ndim == 2
        else data[offset : offset + cells]
    )
    return contract("cq,cqiv,cvw,cqjw->cij", physical_weights, basis, tensor, basis)


def _constitutive_entries(
    dofs: FiniteElementDofMap,
    geometries: tuple[FiniteElementBlockGeometry, ...],
    data: Array,
    components: int,
    /,
) -> tuple[np.ndarray, np.ndarray, Array]:
    rows, columns, values = [], [], []
    offset = 0
    for routes, transform, geometry in zip(
        dofs.cell_dofs, dofs.cell_transforms, geometries, strict=True
    ):
        count = routes.shape[0]
        local = _weighted_local_mass(geometry, data, count, components, offset)
        local = contract("cai,cab,cbj->cij", transform, local, transform)
        indices = np.asarray(routes)
        rows.append(np.broadcast_to(indices[:, :, None], local.shape).reshape((-1,)))
        columns.append(np.broadcast_to(indices[:, None, :], local.shape).reshape((-1,)))
        values.append(local.reshape((-1,)))
        offset += count
    return _coalesce_entries(
        np.concatenate(rows),
        np.concatenate(columns),
        jnp.concatenate(values),
        dofs.global_dof_count,
    )


@final
class FiniteElementDeRhamComplex(AbstractCellDeRhamComplex):
    """Exact conforming polynomial forms with metric-only quadrature Riesz maps.

    Trimmed and tensor-trimmed sequences use ``order`` in every degree.
    Full sequences use polynomial order ``order + dimension - degree``:
    ``order`` specifies the top-form polynomial order.
    ``coordinate_spec`` binds the exact source/reference map used for Piola
    actions; nonlinear nested transfers require a bound CellGeometryTransition.
    ``hodge_admission`` declares the native complete-Cholesky policy for SPD
    admission, independently of the metric Riesz solve. Its byte cap includes
    factor storage and multifrontal workspace; ordering, work and pivot checks
    remain native policy. ``hodge_resources`` budgets default Riesz solves and
    uses ``workspace_bytes`` for bounded geometry/Hodge assembly. A supplied
    ``hodge_solve`` carries its own solve budgets, so the two remain exclusive.
    Mapped basis tables are deferred (per-point Jacobians retained): the metric
    Hodges and constitutive Grams form them once, in bounded cell blocks.
    """

    mesh: CellMesh
    discretization: FiniteElementDiscretization
    topology: CellComplexTopology
    boundary_masks: tuple[Array, ...]
    hodges: tuple[SparseHodge, ...]
    _absolute: HilbertComplex
    _relative: HilbertComplex
    _active_absolute: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    _active_relative: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    family: FormElementFamily = eqx.field(static=True)
    order: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    primal_twist: FormTwist = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        /,
        *,
        family: FormElementFamily,
        order: int,
        twist: FormTwist = "untwisted",
        coordinate_spec: CellGeometrySpec | None = None,
        coordinate_measure: CoordinateMeasurePlan | None = None,
        hodge_solve: LinearSolvePolicy | None = None,
        hodge_resources: SolveResourcePolicy | None = None,
        hodge_admission: SparseFactorizationPolicy | None = None,
        coefficient_dtype: DTypeLike = jnp.float64,
    ) -> None:
        selected = parse(family, FormElementFamily, "family")
        twist_ = parse(twist, FormTwist, "twist")
        minimum_order = 0 if selected == "full" else 1
        if isinstance(order, bool) or not isinstance(order, int) or order < minimum_order:
            raise ValueError(
                f"FE complex order must be an integer at least {minimum_order} for {selected!r}."
            )
        if hodge_solve is not None and not isinstance(hodge_solve, LinearSolvePolicy):
            raise TypeError("hodge_solve must be an owning LinearSolvePolicy or None.")
        if hodge_resources is not None and not isinstance(
            hodge_resources, SolveResourcePolicy
        ):
            raise TypeError("hodge_resources must be a SolveResourcePolicy or None.")
        if hodge_solve is not None and hodge_resources is not None:
            raise ValueError(
                "hodge_resources budgets the default Hodge solve; a supplied "
                "hodge_solve declares its own resources."
            )
        admission = _admission_policy(hodge_admission)
        dimension = mesh.topological_dimension
        fields = _complex_fields(mesh, selected, order, twist_)
        discretization = FiniteElementDiscretization(
            FiniteElementPlan(
                mesh,
                fields,
                coordinate_spec=coordinate_spec,
                coordinate_measure=coordinate_measure,
                coefficient_dtype=coefficient_dtype,
            ),
            basis_tables="deferred",
            geometry_workset_bytes=(
                None if hodge_resources is None else hodge_resources.workspace_bytes
            ),
        )
        hodges = tuple(
            _metric_hodge(discretization, degree, hodge_solve, hodge_resources, admission)
            for degree in range(dimension + 1)
        )
        identifier = canonical_fingerprint(
            {
                "kind": "finite-element-de-rham",
                "mesh": mesh.mesh_id,
                "family": selected,
                "order": order,
                "twist": twist_,
                "dtype": np.dtype(coefficient_dtype).str,
                "geometry_layout": discretization.default_runtime.geometry_layout_id,
                "geometry_coordinates": array_tree_fingerprint(
                    np.asarray(discretization.default_runtime.coordinates)
                ),
                "coordinate_frame": discretization.default_runtime.coordinate_frame,
                "coordinate_measure": (
                    None
                    if discretization.coordinate_measure is None
                    else discretization.coordinate_measure.plan_id
                ),
            }
        )
        absolute = _absolute_fe_complex(
            discretization, hodges, coefficient_dtype, selected, order, identifier
        )
        masks = tuple(dofs.boundary_dof_mask for dofs in discretization.dof_maps)
        active = tuple(
            tuple(np.flatnonzero(~np.asarray(mask)).tolist()) for mask in masks
        )
        relative = self._restricted_complex(
            absolute, hodges, active, f"{identifier}:relative"
        )
        self.mesh, self.discretization, self.topology = (
            mesh,
            discretization,
            mesh.topology,
        )
        self.hodges, self.boundary_masks = hodges, masks
        self._absolute, self._relative = absolute, relative
        self._active_absolute = tuple(
            tuple(range(space.size)) for space in absolute.spaces
        )
        self._active_relative = active
        self.family, self.order, self.dimension = selected, order, dimension
        self.primal_twist, self.realization_id = twist_, identifier

    @staticmethod
    def _restricted_complex(
        complex: HilbertComplex,
        hodges: tuple[SparseHodge, ...],
        active: tuple[tuple[int, ...], ...],
        identifier: str,
        /,
    ) -> HilbertComplex:
        identifier = canonical_fingerprint(
            {
                "kind": "finite-element-restriction",
                "owner": identifier,
                "source": complex.complex_id,
                "active_indices": active,
            }
        )
        spaces = tuple(
            hodge.restrict(np.asarray(indices, dtype=np.int32)).make_space(
                space_id=f"{identifier}:{degree}",
                dtype=_require_array_space(complex.space(degree)).dtype,
            )[0]
            for degree, (hodge, indices) in enumerate(zip(hodges, active, strict=True))
        )
        maps = []
        for degree, differential in enumerate(complex.differentials):
            if not isinstance(differential, SparseCoordinateOperator):
                raise TypeError(
                    "FE restriction requires sparse coordinate differentials."
                )
            relation = differential.relation
            if not isinstance(relation, EdgeRelation):
                raise TypeError(
                    "FE polynomial differentials require explicit edge relations."
                )
            row_lookup = np.full((complex.space(degree + 1).size,), -1, dtype=np.int32)
            col_lookup = np.full((complex.space(degree).size,), -1, dtype=np.int32)
            row_lookup[list(active[degree + 1])] = np.arange(
                len(active[degree + 1]), dtype=np.int32
            )
            col_lookup[list(active[degree])] = np.arange(
                len(active[degree]), dtype=np.int32
            )
            rows = row_lookup[np.asarray(relation.target_indices)]
            columns = col_lookup[np.asarray(relation.source_indices)]
            keep = np.asarray(relation.valid) & (rows >= 0) & (columns >= 0)
            maps.append(
                _sparse_operator(
                    rows[keep],
                    columns[keep],
                    differential.coefficients[jnp.asarray(keep)],
                    spaces[degree],
                    spaces[degree + 1],
                    f"{identifier}:d:{degree}",
                )
            )
        return HilbertComplex(spaces, tuple(maps), complex_id=identifier)

    def hilbert_complex(
        self, /, *, boundary: ComplexBoundary = "absolute"
    ) -> HilbertComplex:
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        return self._absolute if boundary_ == "absolute" else self._relative

    def active_indices(
        self, degree: int, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        self.form_type(degree)
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        indices = (
            self._active_absolute if boundary_ == "absolute" else self._active_relative
        )
        return jnp.asarray(indices[degree], dtype=jnp.int32)

    def _values(self, degree: int, values: ArrayLike, /) -> Array:
        self.form_type(degree)
        value = jnp.asarray(values)
        count = self._absolute.space(degree).size
        if value.shape != (count,):
            raise ValueError(
                f"Form coefficients require shape {(count,)}; got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(jnp.float64)
        return value

    def _compact(self, degree: int, values: Array, boundary: ComplexBoundary, /) -> Array:
        if self.hilbert_complex(boundary=boundary).space(degree).size == values.size:
            return values
        return values[self.active_indices(degree, boundary=boundary)]

    def _extend(self, degree: int, values: Array, boundary: ComplexBoundary, /) -> Array:
        if values.size == self._absolute.space(degree).size:
            return values
        return (
            jnp.zeros((self._absolute.space(degree).size,), dtype=values.dtype)
            .at[self.active_indices(degree, boundary=boundary)]
            .set(values)
        )

    def exterior_derivative(
        self, degree: int, values: ArrayLike, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        """Differentiate full moment coordinates, zero extending boundary constraints."""
        value = self._values(degree, values)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        complex_ = self.hilbert_complex(boundary=boundary)
        if degree >= self.dimension:
            raise ValueError("Exterior derivative degree must be below dimension.")
        active = self._compact(degree, value, boundary)
        output = apply_real_map_componentwise(complex_.differential(degree).mv, active)
        return self._extend(degree + 1, output, boundary)

    def codifferential(
        self, degree: int, values: ArrayLike, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        """Apply the metric adjoint in the declared full moment coordinates."""
        value = self._values(degree, values)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        complex_ = self.hilbert_complex(boundary=boundary)
        if degree <= 0:
            raise ValueError("Codifferential degree must be positive.")
        active = self._compact(degree, value, boundary)
        weighted = apply_real_map_componentwise(complex_.space(degree).riesz, active)
        transposed = apply_real_map_componentwise(
            complex_.differential(degree - 1).transpose_mv, weighted
        )
        output = apply_real_map_componentwise(
            complex_.space(degree - 1).inverse_riesz, transposed
        )
        return self._extend(degree - 1, output, boundary)

    def hodge_laplacian(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        part: HodgeLaplacianPart = "complete",
    ) -> Array:
        """Apply the requested metric Laplacian in full moment coordinates."""
        value = self._values(degree, values)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        operator = hodge_laplacian(
            self.hilbert_complex(boundary=boundary), degree, part=part
        )
        active = self._compact(degree, value, boundary)
        output = apply_real_map_componentwise(operator.mv, active)
        return self._extend(degree, output, boundary)

    def hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        space = _require_array_space(self._absolute.space(degree))
        if jnp.issubdtype(space.dtype, jnp.complexfloating):
            return space.riesz(values)
        return apply_real_map_componentwise(space.riesz, values)

    def inverse_hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        space = _require_array_space(self._absolute.space(degree))
        if jnp.issubdtype(space.dtype, jnp.complexfloating):
            return space.inverse_riesz(values)
        return apply_real_map_componentwise(space.inverse_riesz, values)

    def interpolant(
        self, degree: int, form: Callable[[Array], ArrayLike], /
    ) -> DiscreteForm:
        form_type = self.form_type(degree)
        if not callable(form):
            raise TypeError("form must evaluate canonical physical form components.")
        dofs = self.discretization.dof_maps[degree]
        result = jnp.zeros(
            (dofs.global_dof_count,),
            dtype=_require_array_space(self._absolute.space(degree)).dtype,
        )
        counts = jnp.zeros((dofs.global_dof_count,), dtype=jnp.int32)
        for block_index, element in enumerate(self.discretization.elements[degree]):
            basis = element.form_basis
            if basis is None:
                raise ValueError("Interpolation requires canonical form functionals.")
            cell_map = PreparedFiniteElementCellMap(self.discretization, block_index)
            for cell in range(cell_map.cell_count):
                points = basis.functional_points
                cells = jnp.full((points.shape[0],), cell, dtype=jnp.int32)
                geometry = cell_map.evaluate(self.mesh.coordinates, cells, points)
                values = jnp.asarray(form(geometry.physical_points))
                expected = (points.shape[0], form_type.component_count)
                if values.shape != expected:
                    raise ValueError(
                        f"Form interpolation requires canonical component shape {expected}."
                    )
                reference = pullback(values, form_type, geometry.jacobian)
                local = basis.interpolate(reference)
                canonical = jnp.linalg.solve(
                    dofs.cell_transforms[block_index][cell], local
                )
                routes = dofs.cell_dofs[block_index][cell]
                result = result.at[routes].add(canonical)
                counts = counts.at[routes].add(1)
        return DiscreteForm(self.realization_id, form_type, result / counts)

    def reconstruction(self, degree: int, /) -> PreparedFieldReconstruction:
        self.form_type(degree)
        return prepare_finite_element_field_reconstruction(
            self.discretization, f"form_{degree}"
        )

    def side_traces(self, degree: int, /) -> SideTraceProvider:
        self.form_type(degree)
        return self.discretization

    def constitutive_operator(
        self,
        degree: int,
        coefficient: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
    ) -> AbstractLinearOperator:
        """Weighted coordinate Gram, separate from the metric Hilbert pairing."""
        self.form_type(degree)
        data = jnp.asarray(coefficient)
        cells = sum(block.cell_count for block in self.mesh.blocks)
        component_count = prod(self.discretization.elements[degree][0].value_shape) or 1
        data, properties = _constitutive_admission(data, cells, component_count)
        dofs = self.discretization.dof_maps[degree]
        row, column, payload = _constitutive_entries(
            dofs, self.discretization.block_geometries[degree], data, component_count
        )
        active = np.asarray(self.active_indices(degree, boundary=boundary))
        lookup = np.full((dofs.global_dof_count,), -1, dtype=np.int32)
        lookup[active] = np.arange(active.size, dtype=np.int32)
        keep = (lookup[row] >= 0) & (lookup[column] >= 0)
        space = coordinate_space(self.hilbert_complex(boundary=boundary).space(degree))
        if not isinstance(space, ArraySpace):
            raise TypeError("FE coordinate spaces must be array spaces.")
        promoted = jnp.result_type(space.dtype, data.dtype)
        if promoted != space.dtype:
            space = ArraySpace(
                space.shape,
                dtype=promoted,
                space_id=f"{space.space_id}:complexification:{np.dtype(promoted).str}",
            )
        coefficient_id = array_tree_fingerprint(coefficient)
        return _sparse_operator(
            lookup[row[keep]],
            lookup[column[keep]],
            payload[jnp.asarray(keep)],
            space,
            space,
            f"{self.realization_id}:constitutive:{degree}:{boundary}:{coefficient_id}",
            properties=properties,
        )

    def trace_complex_map(
        self, /, *, boundary_mask: ArrayLike | None = None
    ) -> ComplexMap:
        """Outward-oriented restriction to the selected boundary facet closure."""
        closure = _boundary_entity_closure(self.topology, boundary_mask)
        active, orientation = _boundary_moment_selection(
            self.discretization, closure, self.cell_counts
        )
        selected_signs = tuple(
            tuple(orientation[degree][list(indices)].tolist())
            for degree, indices in enumerate(active[:-1])
        )
        identifier = canonical_fingerprint(
            {
                "kind": "finite-element-boundary-trace",
                "realization": self.realization_id,
                "entity_closure": tuple(
                    tuple(np.flatnonzero(mask).tolist()) for mask in closure
                ),
                "active_indices": active,
                "orientation": selected_signs,
            }
        )
        restricted = self._restricted_complex(
            self._absolute, self.hodges, active, f"{identifier}:restriction"
        )
        signs = tuple(jnp.asarray(values) for values in selected_signs)
        target = _oriented_boundary_complex(
            self._absolute, restricted, self.hodges, active, signs, identifier
        )
        maps = tuple(
            _sparse_operator(
                np.arange(len(indices), dtype=np.int32),
                np.asarray(indices, dtype=np.int32),
                signs[degree],
                self._absolute.space(degree),
                target.space(degree),
                f"{identifier}:trace:{degree}",
            )
            for degree, indices in enumerate(active[:-1])
        )
        return ComplexMap(self._absolute, target, maps, map_id=f"{identifier}:trace")

    def vector_interpolation(
        self, degree: int, /, *, boundary: ComplexBoundary = "absolute"
    ) -> AbstractLinearOperator:
        """Canonical moments of a physical piecewise linear nodal field."""
        self.form_type(degree)
        boundary = parse(boundary, ComplexBoundary, "boundary")
        components = prod(self.discretization.elements[degree][0].value_shape) or 1
        vertices = self.mesh.coordinates.shape[0]
        nodal_mask = np.asarray(self.mesh.topology.entity_sets[0].subset("boundary").mask)
        selected = (
            np.arange(vertices, dtype=np.int32)
            if boundary == "absolute"
            else np.flatnonzero(~nodal_mask).astype(np.int32)
        )
        source = ArraySpace(
            (selected.size, components),
            dtype=_require_array_space(self._absolute.space(degree)).dtype,
            space_id=f"{self.realization_id}:nodal-vector:{components}:{boundary}",
        )
        target = self.hilbert_complex(boundary=boundary).space(degree)
        if not isinstance(target, ArraySpace):
            raise TypeError("FE vector interpolation requires array coefficient spaces.")
        vertex_lookup = np.full((vertices,), -1, dtype=np.int32)
        vertex_lookup[selected] = np.arange(selected.size, dtype=np.int32)
        active = np.asarray(self.active_indices(degree, boundary=boundary))
        target_lookup = np.full((self.cell_counts[degree],), -1, dtype=np.int32)
        target_lookup[active] = np.arange(active.size, dtype=np.int32)
        rows, columns, payload = [], [], []
        owners: set[int] = set()
        dofs = self.discretization.dof_maps[degree]
        for block_index, (block, element) in enumerate(
            zip(self.mesh.blocks, self.discretization.elements[degree], strict=True)
        ):
            basis = element.form_basis
            if basis is None:
                raise ValueError("Vector interpolation requires form functionals.")
            points = basis.functional_points
            nodal = _linear_reference_element(block.cell_kind).tabulate(points)[0]
            cell_map = PreparedFiniteElementCellMap(self.discretization, block_index)
            for cell, cell_vertices in enumerate(np.asarray(block.vertices)):
                geometry = cell_map.evaluate(
                    self.discretization.default_runtime.coordinates,
                    jnp.full((points.shape[0],), cell, dtype=jnp.int32),
                    points,
                )
                physical = (
                    nodal[:, :, None, None]
                    * jnp.eye(components, dtype=nodal.dtype)[None, None]
                )
                physical = physical.reshape((points.shape[0], -1, *element.value_shape))
                reference = pullback(
                    vector_to_form(physical, element.value_spec),
                    self.form_type(degree),
                    geometry.jacobian[:, None],
                )
                moments = contract("dqc,qbc->db", basis.functional_weights, reference)
                local = jnp.linalg.solve(dofs.cell_transforms[block_index][cell], moments)
                source_columns = (
                    vertex_lookup[cell_vertices, None] * components
                    + np.arange(components)[None]
                ).reshape((-1,))
                for dof, global_row in enumerate(
                    np.asarray(dofs.cell_dofs[block_index][cell])
                ):
                    row = target_lookup[global_row]
                    if row < 0 or int(global_row) in owners:
                        continue
                    owners.add(int(global_row))
                    keep = np.repeat(vertex_lookup[cell_vertices] >= 0, components)
                    rows.extend([int(row)] * int(np.sum(keep)))
                    columns.extend(source_columns[keep].tolist())
                    payload.append(local[dof, jnp.asarray(keep)])
        return _sparse_operator(
            np.asarray(rows, dtype=np.int32),
            np.asarray(columns, dtype=np.int32),
            jnp.concatenate(payload) if payload else jnp.zeros((0,), dtype=source.dtype),
            source,
            target,
            f"{self.realization_id}:vector-interpolation:{degree}:{boundary}",
        )

    @checked
    def transfer(
        self,
        target: FiniteElementDeRhamComplex,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        parent_cells: ConvertibleToArray | None = None,
        reference_maps: NestedReferenceWitnesses
        | ReferenceCellMapWitnesses
        | Sequence[NestedReferenceWitnesses | ReferenceCellMapWitnesses]
        | None = None,
        geometry_transition: CellGeometryTransition | None = None,
        integration_policy: CurvedOverlapIntegrationPolicy | None = None,
    ) -> FiniteElementTransferResult:
        """Prepare a same-mesh p or exact nested h/Piola form transfer.

        ``parent_cells`` uses concatenated source block order and derives exact
        reference maps only for verified P1 simplex coordinate charts.
        ``reference_maps`` supplies exact target-to-source charts by stable cell
        IDs. Nonlinear/higher-order geometry requires its bound
        ``geometry_transition``; no sampled physical inversion admits ancestry.
        Coarsening requires the transition's complete ``coarsening_partition``
        or, for nested simplex coarsening, its exact fine-to-coarse
        ``coarsening`` witness, which is proved to be a complete partition;
        target moments are integrated over uniquely owned fine-cell entity pieces.
        Nonconstant rational reference charts require an explicit
        ``integration_policy``. The result retains entrywise integral enclosures
        and coefficientwise commuting residual intervals; it carries no
        whole-domain coverage or conservative claim.
        """
        boundary_ = parse(boundary, ComplexBoundary, "boundary")
        if integration_policy is not None:
            from ...geometry._common_root_volume_overlap import (
                CurvedOverlapIntegrationPolicy,
            )

            if not isinstance(integration_policy, CurvedOverlapIntegrationPolicy):
                raise TypeError(
                    "integration_policy must be CurvedOverlapIntegrationPolicy or None."
                )
        if target.dimension != self.dimension or target.primal_twist != self.primal_twist:
            raise ValueError("FE transfer requires the same dimension and twist.")
        source_measure = self.discretization.coordinate_measure
        target_measure = target.discretization.coordinate_measure
        source_measure_key = (
            ("cartesian", None, "reference")
            if source_measure is None
            else (source_measure.kind, source_measure.radial_axis, source_measure.frame)
        )
        target_measure_key = (
            ("cartesian", None, "reference")
            if target_measure is None
            else (target_measure.kind, target_measure.radial_axis, target_measure.frame)
        )
        if (
            self.discretization.default_runtime.coordinate_frame
            != target.discretization.default_runtime.coordinate_frame
            or source_measure_key != target_measure_key
        ):
            raise ValueError(
                "FE transfer endpoints must use the same coordinate frame and measure."
            )
        geometry_map_transition = None
        coarsening_partition = None
        partition_target_ids: frozenset[int] = frozenset()
        if geometry_transition is not None:
            if not isinstance(geometry_transition, CellGeometryTransition):
                raise TypeError(
                    "geometry_transition must be a bound CellGeometryTransition."
                )
            geometry_transition.require_bound(
                self.mesh,
                geometry_transition.source_geometry,
                target.mesh,
                geometry_transition.target_geometry,
            )
            source_geometry = geometry_transition.source_geometry
            target_geometry = geometry_transition.target_geometry
            if (
                source_geometry.geometry_layout_id
                != self.discretization.default_runtime.geometry_layout_id
                or target_geometry.geometry_layout_id
                != target.discretization.default_runtime.geometry_layout_id
                or not np.array_equal(
                    np.asarray(source_geometry.resolve(self.mesh)[2], dtype=np.float64),
                    np.asarray(
                        self.discretization.default_runtime.coordinates, dtype=np.float64
                    ),
                )
                or not np.array_equal(
                    np.asarray(target_geometry.resolve(target.mesh)[2], dtype=np.float64),
                    np.asarray(
                        target.discretization.default_runtime.coordinates,
                        dtype=np.float64,
                    ),
                )
            ):
                raise ValueError(
                    "FE transfer geometry does not match its accepted coordinate transition."
                )
            direct_maps = geometry_transition.reference_map
            if isinstance(direct_maps, ReferenceCellMapWitnesses):
                direct_records = (direct_maps,)
            elif direct_maps is None:
                direct_records = ()
            else:
                direct_records = tuple(direct_maps)
            refinement_record = geometry_transition.refinement
            coarsening_partition = geometry_transition.coarsening_partition
            if (
                coarsening_partition is None
                and geometry_transition.coarsening is not None
            ):
                from .._cell_geometry_transition_types import _nested_reference_groups

                # Nested fine→coarse rows are fine-target/coarse-source maps:
                # exactly the partition orientation, proved complete below.
                coarsening_partition = ReferenceCellPartition(
                    _nested_reference_groups(
                        geometry_transition.coarsening,
                        self.mesh,
                        target.mesh,
                        reverse=True,
                    )
                )
            if coarsening_partition is not None:
                if parent_cells is not None:
                    raise ValueError(
                        "Coarsening requires partition witnesses, not one parent per target."
                    )
                partition_target_ids = frozenset(
                    int(identifier)
                    for group in coarsening_partition.maps
                    for identifier in np.asarray(group.source_cell_ids, dtype=np.int64)
                )
            retained_records = (
                *direct_records,
                *((refinement_record,) if refinement_record is not None else ()),
            )
            if not retained_records and reference_maps is not None:
                raise ValueError(
                    "FE transfer maps must be retained by the geometry transition."
                )
            geometry_map_transition = geometry_transition

            def target_ids_for_record(
                record: NestedReferenceWitnesses | ReferenceCellMapWitnesses, /
            ) -> tuple[int, ...]:
                ids = (
                    record.target_cell_ids
                    if isinstance(record, ReferenceCellMapWitnesses)
                    else record.fine_cell_ids
                )
                return tuple(int(value) for value in np.asarray(ids, dtype=np.int64))

            if reference_maps is not None:
                provided_records = (
                    (reference_maps,)
                    if isinstance(
                        reference_maps,
                        (NestedReferenceWitnesses, ReferenceCellMapWitnesses),
                    )
                    else tuple(reference_maps)
                )
                if any(
                    not any(record is retained for retained in retained_records)
                    for record in provided_records
                ):
                    raise ValueError(
                        "FE transfer maps must be retained by the geometry transition."
                    )
                selected_records = provided_records
            else:
                target_ids = {
                    int(identifier)
                    for block in target.mesh.blocks
                    for identifier in np.asarray(block.global_ids, dtype=np.int64)
                }
                direct_ids = tuple(
                    identifier
                    for record in direct_records
                    for identifier in target_ids_for_record(record)
                )
                direct_target_ids = set(direct_ids)
                if len(direct_target_ids) != len(direct_ids):
                    raise ValueError(
                        "Geometry transition direct maps repeat target cell IDs."
                    )
                if not direct_target_ids <= target_ids:
                    raise ValueError(
                        "Geometry transition direct maps reference absent target cells."
                    )
                selected: list[NestedReferenceWitnesses | ReferenceCellMapWitnesses] = (
                    list(direct_records)
                )
                covered_target_ids = set(direct_target_ids) | set(partition_target_ids)
                if refinement_record is not None:
                    refined_ids = target_ids_for_record(refinement_record)
                    refined_target_ids = set(refined_ids)
                    if (
                        len(refined_target_ids) != len(refined_ids)
                        or not refined_target_ids <= target_ids
                    ):
                        raise ValueError(
                            "Geometry transition refinement maps repeat or reference absent target cells."
                        )
                    selected.append(refinement_record)
                    covered_target_ids.update(refined_target_ids)
                selected.extend(
                    _retained_identity_reference_maps(
                        self.discretization,
                        target.discretization,
                        covered_target_ids,
                    )
                )
                selected_records = tuple(selected)
            reference_maps = selected_records
        reference_by_target = None
        declared_parents = None
        if reference_maps is not None:
            reference_by_target, declared_parents = _index_nested_reference_maps(
                reference_maps,
                self.mesh,
                target.mesh,
                partition_target_ids=partition_target_ids,
            )
        parents = None if parent_cells is None else np.asarray(parent_cells)
        if declared_parents is not None:
            if parents is not None and (
                parents.shape != declared_parents.shape
                or not np.issubdtype(parents.dtype, np.integer)
                or not np.array_equal(parents, declared_parents)
            ):
                raise ValueError("parent_cells disagrees with the exact reference maps.")
            parents = declared_parents
        if parents is None and target.mesh.mesh_id != self.mesh.mesh_id:
            raise ValueError(
                "Distinct meshes require exact reference maps or explicit parent cells."
            )
        if parents is None and (
            target.discretization.default_runtime.geometry_layout_id
            != self.discretization.default_runtime.geometry_layout_id
            or not np.array_equal(
                np.asarray(target.discretization.default_runtime.coordinates),
                np.asarray(self.discretization.default_runtime.coordinates),
            )
        ):
            raise ValueError(
                "Same-cell p transfer requires identical represented coordinate maps."
            )
        if parents is not None:
            shape = (sum(block.cell_count for block in target.mesh.blocks),)
            count = sum(block.cell_count for block in self.mesh.blocks)
            if (
                parents.shape != shape
                or not np.issubdtype(parents.dtype, np.integer)
                or np.any((parents < -1) | (parents >= count))
                or (np.any(parents == -1) and coarsening_partition is None)
            ):
                raise ValueError(
                    "parent_cells must assign each target child one valid source cell."
                )
            if reference_by_target is None:
                if geometry_transition is not None:
                    raise ValueError(
                        "A geometry transition must retain exact reference-map witnesses for FE transfer."
                    )
                derived_maps = _exact_p1_simplex_reference_maps(
                    self.discretization,
                    target.discretization,
                    parents,
                )
                reference_by_target, derived_parents = _index_nested_reference_maps(
                    derived_maps,
                    self.mesh,
                    target.mesh,
                )
                reference_maps = derived_maps
                if not np.array_equal(parents, derived_parents):
                    raise ValueError("Derived exact maps disagree with parent_cells.")
                parents = derived_parents
            self._admit_nested_parents(
                target,
                parents,
                reference_by_target,
                geometry_map_transition,
            )

        prepared_orientations: dict[int, _PreparedReferenceChartOrientation] | None = None
        if reference_by_target is not None:
            from ._exact_form_moments import _prepare_reference_chart_orientation

            prepared_orientations = {}
            orientation_cache: dict[
                tuple[str, tuple[object, ...]], _PreparedReferenceChartOrientation
            ] = {}
            target_offset = 0
            for block_index, block in enumerate(target.mesh.blocks):
                basis = target.discretization.elements[0][block_index].form_basis
                if basis is None:
                    raise ValueError(
                        "Prepared FE chart orientation requires canonical target form bases."
                    )
                for cell in range(block.cell_count):
                    target_row = target_offset + cell
                    if target_row not in reference_by_target:
                        continue
                    prepared_orientations[target_row] = (
                        _prepare_reference_chart_orientation(
                            reference_by_target[target_row],
                            basis.dimension,
                            basis.family,
                            orientation_cache,
                        )
                    )
                target_offset += block.cell_count
        partition_cells = None
        if coarsening_partition is not None:
            from ._partition_form_moments import index_reference_cell_partition

            partition_cells = index_reference_cell_partition(coarsening_partition)
        degree_results = tuple(
            self._transfer_degree(
                target,
                degree,
                parents,
                partition_cells,
                reference_by_target,
                prepared_orientations,
                integration_policy,
            )
            for degree in range(self.dimension + 1)
        )
        absolute_maps = tuple(value[0] for value in degree_results)
        absolute_entry_evidence = tuple(value[1] for value in degree_results)
        exact_integral_count = sum(value[2] for value in degree_results)
        enclosed_integral_count = sum(value[3] for value in degree_results)
        chart_argument_ids = _reference_chart_argument_ids(reference_maps)
        if parents is None:
            lineage = "p"
        elif reference_by_target is None:
            lineage = array_tree_fingerprint(parents)
        else:
            exact_payload = []
            for target_row, reference_map in sorted(reference_by_target.items()):
                target_id = (
                    reference_map.target_cell_id
                    if isinstance(reference_map, NestedCellReferenceMap)
                    else reference_map.fine_cell_id
                )
                source_id = (
                    reference_map.source_cell_id
                    if isinstance(reference_map, NestedCellReferenceMap)
                    else reference_map.coarse_cell_id
                )
                vertices = (
                    reference_map.source_reference_vertices
                    if isinstance(reference_map, NestedCellReferenceMap)
                    else reference_map.vertices
                )
                arguments = chart_argument_ids.get(int(target_id))
                exact_payload.append(
                    (
                        int(target_id),
                        int(source_id),
                        tuple(
                            tuple((value.numerator, value.denominator) for value in point)
                            for point in vertices
                        ),
                        arguments,
                    )
                )
            lineage = canonical_fingerprint(
                {
                    "kind": "nested-exact-reference-maps",
                    "parent_cells": array_tree_fingerprint(parents),
                    "maps": tuple(exact_payload),
                }
            )
        source_complex = self.hilbert_complex(boundary=boundary_)
        target_complex = target.hilbert_complex(boundary=boundary_)
        policy_id = None if integration_policy is None else integration_policy.policy_id
        map_identifier = canonical_fingerprint(
            {
                "kind": "finite-element-reference-form-transfer",
                "source": self.realization_id,
                "target": target.realization_id,
                "lineage": lineage,
                "geometry_transition": (
                    None
                    if geometry_map_transition is None
                    else geometry_map_transition.transition_id
                ),
                "boundary": boundary_,
                "integration_policy": policy_id,
            }
        )
        constraints = []
        if boundary_ == "absolute":
            maps = absolute_maps
            entry_evidence = absolute_entry_evidence
        else:
            restricted = tuple(
                _relative_transfer_with_evidence(
                    absolute_maps[degree],
                    absolute_entry_evidence[degree],
                    self._active_relative[degree],
                    target._active_relative[degree],
                    _require_array_space(source_complex.space(degree)),
                    _require_array_space(target_complex.space(degree)),
                    f"{map_identifier}:relative:{degree}",
                )
                for degree in range(self.dimension + 1)
            )
            maps = tuple(value[0] for value in restricted)
            entry_evidence = tuple(value[1] for value in restricted)
            constraints = [value[2] for value in restricted if value[2] is not None]
        complex_map = ComplexMap(
            source_complex,
            target_complex,
            maps,
            map_id=map_identifier,
        )
        commuting_evidence = _require_complex_map_commutes(
            complex_map, entry_evidence, integration_policy
        )
        return FiniteElementTransferResult(
            complex_map,
            entry_evidence,
            commuting_evidence,
            integration_policy,
            exact_integral_count,
            enclosed_integral_count,
            constraint_evidence=tuple(constraints),
        )

    @checked
    def transfer_periodic_quotient(
        self,
        target: FiniteElementDeRhamComplex,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        geometry_transition: CellGeometryTransition | None = None,
        parent_cells: ConvertibleToArray | None = None,
        reference_maps: NestedReferenceWitnesses
        | ReferenceCellMapWitnesses
        | Sequence[NestedReferenceWitnesses | ReferenceCellMapWitnesses]
        | None = None,
        integration_policy: CurvedOverlapIntegrationPolicy | None = None,
    ) -> PeriodicFiniteElementTransfer:
        """Descend the full de Rham transfer to validated periodic quotients.

        Every form degree is lifted through its oriented, higher-order entity
        action.  The induced source and target differentials and transfer are
        admitted only when the lifted actions descend to those quotient
        subcomplexes.
        Rational chart moments use ``integration_policy``; the result retains
        both lifted and quotient entrywise integration/commuting evidence.
        """
        boundary_ = parse(boundary, ComplexBoundary, "boundary")

        from .._periodic_exact import identification_id

        source_periodic = self.mesh.periodic_topology
        target_periodic = target.mesh.periodic_topology
        if source_periodic is None or target_periodic is None:
            raise ValueError("Periodic quotient transfer requires two periodic meshes.")
        if identification_id(source_periodic.cell) != identification_id(
            target_periodic.cell
        ):
            raise ValueError("Periodic endpoints must use the same exact identification.")
        lifted_transfer_result = self.transfer(
            target,
            boundary="absolute",
            parent_cells=parent_cells,
            reference_maps=reference_maps,
            geometry_transition=geometry_transition,
            integration_policy=integration_policy,
        )
        lifted_map = lifted_transfer_result.complex_map
        source_topology = _PeriodicFormTopology(self.mesh, self.dimension)
        target_topology = (
            source_topology
            if target.mesh is self.mesh
            else _PeriodicFormTopology(target.mesh, target.dimension)
        )
        source_actions = tuple(
            _periodic_entity_coordinate_maps(self, degree, _topology=source_topology)
            for degree in range(self.dimension + 1)
        )
        del source_topology
        target_actions = tuple(
            _periodic_entity_coordinate_maps(target, degree, _topology=target_topology)
            for degree in range(self.dimension + 1)
        )
        del target_topology
        source_lifts = tuple(action[0] for action in source_actions)
        source_collapses = tuple(action[1] for action in source_actions)
        target_lifts = tuple(action[0] for action in target_actions)
        target_collapses = tuple(action[1] for action in target_actions)
        source_orbit_ids = tuple(action[2] for action in source_actions)
        target_orbit_ids = tuple(action[2] for action in target_actions)
        source_boundary_masks = tuple(action[3] for action in source_actions)
        target_boundary_masks = tuple(action[3] for action in target_actions)
        source_id = canonical_fingerprint(
            {
                "kind": "periodic-quotient-finite-element-complex",
                "owner": self.realization_id,
                "periodic_topology": source_periodic.periodic_topology_id,
                "orbit_maps": source_orbit_ids,
            }
        )
        target_id = canonical_fingerprint(
            {
                "kind": "periodic-quotient-finite-element-complex",
                "owner": target.realization_id,
                "periodic_topology": target_periodic.periodic_topology_id,
                "orbit_maps": target_orbit_ids,
            }
        )
        source_full_quotient, source_closure, source_nilpotency = (
            _periodic_quotient_complex(
                self,
                source_lifts,
                source_collapses,
                source_id,
            )
        )
        target_full_quotient, target_closure, target_nilpotency = (
            _periodic_quotient_complex(
                target,
                target_lifts,
                target_collapses,
                target_id,
            )
        )
        transfer_id = canonical_fingerprint(
            {
                "kind": "periodic-finite-element-transfer",
                "lifted_transfer_evidence": lifted_transfer_result.evidence_id,
                "source_periodic_topology": source_periodic.periodic_topology_id,
                "target_periodic_topology": target_periodic.periodic_topology_id,
                "source_orbit_maps": source_orbit_ids,
                "target_orbit_maps": target_orbit_ids,
                "boundary": boundary_,
            }
        )
        full_maps = []
        full_entry_evidence = []
        deck_evidence = []
        deck_defects = []
        deck_error_bounds = []
        for degree in range(self.dimension + 1):
            lifted_action = _periodic_matrix(lifted_map.map(degree))
            source_lift = _periodic_matrix(source_lifts[degree])
            target_lift = _periodic_matrix(target_lifts[degree])
            target_collapse = _periodic_matrix(target_collapses[degree])
            quotient_action = target_collapse @ lifted_action @ source_lift
            quotient_operator, quotient_entries = _periodic_quotient_transfer_action(
                lifted_transfer_result,
                degree,
                source_lifts[degree],
                target_collapses[degree],
                quotient_action,
                f"{transfer_id}:quotient:{degree}",
            )
            left = lifted_action @ source_lift
            right = target_lift @ quotient_action
            difference = (left - right).tocsr()
            difference.eliminate_zeros()
            observed = abs(difference)
            lifted_error = _matrix_entry_error(
                lifted_transfer_result.entry_evidence[degree]
            )
            quotient_error = _matrix_entry_error(quotient_entries)
            integration_error = (
                lifted_error @ abs(source_lift) + abs(target_lift) @ quotient_error
            ).tocsr()
            left_magnitude = abs(lifted_action) @ abs(source_lift)
            right_magnitude = abs(target_lift) @ abs(quotient_action)
            left_terms = _maximum_row_terms(lifted_action) * _maximum_row_terms(
                source_lift
            )
            right_terms = _maximum_row_terms(target_lift) * _maximum_row_terms(
                quotient_action
            )
            roundoff = (
                _gamma_bound(2 * left_terms) * left_magnitude
                + _gamma_bound(2 * right_terms) * right_magnitude
                + 64.0 * np.finfo(np.float64).eps * (left_magnitude + right_magnitude)
            ).tocsr()
            roundoff = (
                roundoff
                + _gamma_bound(
                    2
                    * max(
                        _maximum_row_terms(source_lift),
                        _maximum_row_terms(target_lift),
                        1,
                    )
                )
                * integration_error
            ).tocsr()
            if integration_error.nnz:
                integration_error.data = np.nextafter(
                    np.maximum(integration_error.data, 0.0), np.inf
                )
            if roundoff.nnz:
                roundoff.data = np.nextafter(np.maximum(roundoff.data, 0.0), np.inf)
            total_bound = (integration_error + roundoff).tocsr()
            excess = (observed - total_bound).tocsr()
            excess.eliminate_zeros()
            if excess.nnz and float(np.max(excess.data, initial=0.0)) > 0.0:
                raise ValueError(
                    f"Finite-element transfer does not descend within its certified intervals in form degree {degree}."
                )
            union = (observed + total_bound).tocsr()
            union.eliminate_zeros()
            coo = union.tocoo()
            if coo.nnz:
                observed_values = np.asarray(
                    difference.tocsr()[coo.row, coo.col]
                ).reshape(-1)
                integration_values = np.asarray(
                    integration_error.tocsr()[coo.row, coo.col]
                ).reshape(-1)
                roundoff_values = np.asarray(roundoff.tocsr()[coo.row, coo.col]).reshape(
                    -1
                )
            else:
                observed_values = integration_values = roundoff_values = np.zeros(
                    (0,), dtype=np.float64
                )
            deck = SparseCommutingEvidence(
                "periodic-descent",
                degree,
                source_lift.shape[1],
                target_lift.shape[0],
                np.asarray(coo.row, dtype=np.int32),
                np.asarray(coo.col, dtype=np.int32),
                observed_values,
                integration_values,
                roundoff_values,
            )
            deck_evidence.append(deck)
            deck_defects.append(float(np.max(np.abs(observed_values), initial=0.0)))
            deck_error_bounds.append(
                float(
                    np.max(
                        integration_values + roundoff_values,
                        initial=0.0,
                    )
                )
            )
            full_maps.append(quotient_operator)
            full_entry_evidence.append(quotient_entries)
        if boundary_ == "relative":
            source_quotient, source_active = _relative_quotient_complex(
                source_full_quotient,
                source_boundary_masks,
                f"{source_id}:relative",
            )
            target_quotient, target_active = _relative_quotient_complex(
                target_full_quotient,
                target_boundary_masks,
                f"{target_id}:relative",
            )
            restricted = tuple(
                _relative_transfer_with_evidence(
                    full_maps[degree],
                    full_entry_evidence[degree],
                    source_active[degree],
                    target_active[degree],
                    _require_array_space(source_quotient.space(degree)),
                    _require_array_space(target_quotient.space(degree)),
                    f"{transfer_id}:relative:{degree}",
                )
                for degree in range(self.dimension + 1)
            )
            quotient_maps = tuple(value[0] for value in restricted)
            quotient_entries = tuple(value[1] for value in restricted)
            boundary_evidence = tuple(
                value[2] for value in restricted if value[2] is not None
            )
        else:
            source_quotient, target_quotient = (
                source_full_quotient,
                target_full_quotient,
            )
            source_active = tuple(
                tuple(range(space.size)) for space in source_quotient.spaces
            )
            target_active = tuple(
                tuple(range(space.size)) for space in target_quotient.spaces
            )
            quotient_maps = tuple(full_maps)
            quotient_entries = tuple(full_entry_evidence)
            boundary_evidence = ()
        quotient_map = ComplexMap(
            source_quotient,
            target_quotient,
            quotient_maps,
            map_id=transfer_id,
        )
        quotient_commuting_evidence = _require_complex_map_commutes(
            quotient_map,
            quotient_entries,
            lifted_transfer_result.integration_policy,
        )
        quotient_transfer_result = FiniteElementTransferResult(
            quotient_map,
            quotient_entries,
            quotient_commuting_evidence,
            lifted_transfer_result.integration_policy,
            lifted_transfer_result.exact_integral_count,
            lifted_transfer_result.enclosed_integral_count,
            constraint_evidence=(*deck_evidence, *boundary_evidence),
        )
        return PeriodicFiniteElementTransfer(
            quotient_map,
            lifted_transfer_result,
            quotient_transfer_result,
            source_lifts,
            source_collapses,
            target_lifts,
            target_collapses,
            source_orbit_ids,
            target_orbit_ids,
            source_active,
            target_active,
            boundary_,
            source_closure,
            target_closure,
            source_nilpotency,
            target_nilpotency,
            deck_defects,
            deck_error_bounds,
            quotient_transfer_result.commuting_defects,
            quotient_transfer_result.commuting_error_bounds,
            transfer_id,
        )

    @checked
    def distributed_transfer(
        self,
        target: FiniteElementDeRhamComplex,
        source_owners: Sequence[ConvertibleToArray],
        target_owners: Sequence[ConvertibleToArray],
        part_count: int,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        parent_cells: ConvertibleToArray | None = None,
        reference_maps: NestedReferenceWitnesses
        | ReferenceCellMapWitnesses
        | Sequence[NestedReferenceWitnesses | ReferenceCellMapWitnesses]
        | None = None,
        geometry_transition: CellGeometryTransition | None = None,
        integration_policy: CurvedOverlapIntegrationPolicy | None = None,
    ) -> DistributedFiniteElementTransfer:
        """Distribute one admitted FE result without losing its error evidence."""
        transfer_result = self.transfer(
            target,
            boundary=boundary,
            parent_cells=parent_cells,
            reference_maps=reference_maps,
            geometry_transition=geometry_transition,
            integration_policy=integration_policy,
        )
        return _distribute_transfer_result(
            transfer_result,
            source_owners,
            target_owners,
            part_count,
        )

    def _admit_nested_parents(
        self,
        target: FiniteElementDeRhamComplex,
        parents: np.ndarray,
        reference_by_target: dict[int, NestedSimplexMap | NestedCellReferenceMap],
        geometry_transition: CellGeometryTransition | None,
        /,
    ) -> None:
        if reference_by_target is None:
            raise ValueError("Nested FE transfer requires exact reference-map witnesses.")
        if geometry_transition is None:
            _require_p1_simplex_coordinates(self.discretization, "Source")
            _require_p1_simplex_coordinates(target.discretization, "Target")
        offsets = np.cumsum((0, *(block.cell_count for block in self.mesh.blocks)))
        source_cell_global_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in self.mesh.blocks]
        )
        source_coordinates = np.asarray(
            self.discretization.default_runtime.coordinates, dtype=np.float64
        )
        target_coordinates = np.asarray(
            target.discretization.default_runtime.coordinates, dtype=np.float64
        )
        target_offset = 0
        for block in target.mesh.blocks:
            target_vertices = np.asarray(block.vertices, dtype=np.int64)
            target_ids = np.asarray(block.global_ids, dtype=np.int64)
            target_topology = reference_cell_topology(block.cell_kind)
            for cell in range(block.cell_count):
                target_row = target_offset + cell
                parent = int(parents[target_row])
                if parent == -1:
                    if geometry_transition is None or (
                        geometry_transition.coarsening_partition is None
                        and geometry_transition.coarsening is None
                    ):
                        raise ValueError("Coarsened cells require a bound partition.")
                    continue
                source_block_index = int(
                    np.searchsorted(offsets[1:], parent, side="right")
                )
                source_cell = parent - int(offsets[source_block_index])
                source_block = self.mesh.blocks[source_block_index]
                source_global_id = int(source_cell_global_ids[parent])
                target_global_id = int(target_ids[cell])
                exact_map = reference_by_target.get(target_row)
                if exact_map is None:
                    raise ValueError(
                        "Exact reference maps must cover every target FE cell."
                    )
                mapped_parent = (
                    exact_map.source_cell_id
                    if isinstance(exact_map, NestedCellReferenceMap)
                    else exact_map.coarse_cell_id
                )
                if (
                    mapped_parent != source_global_id
                    or (
                        exact_map.target_cell_id
                        if isinstance(exact_map, NestedCellReferenceMap)
                        else exact_map.fine_cell_id
                    )
                    != target_global_id
                ):
                    raise ValueError(
                        "Exact reference-map cell IDs disagree with parent_cells."
                    )
                if geometry_transition is not None:
                    continue
                if isinstance(exact_map, NestedCellReferenceMap):
                    from .._coordinate_enclosure import (
                        affine_map,
                        RationalPolynomial,
                    )

                    arguments = exact_map.arguments
                    if arguments is None:
                        raise ValueError(
                            "Exact reference-map chart arguments are unavailable."
                        )
                    affine_arguments = tuple(
                        value if isinstance(value, RationalPolynomial) else dict(value)
                        for value in arguments
                    )
                    if affine_map(affine_arguments, target_topology.dimension) is None:
                        raise ValueError(
                            "Non-affine reference maps require their bound geometry transition."
                        )
                source_topology = reference_cell_topology(source_block.cell_kind)
                source_vertex_rows = np.asarray(
                    source_block.vertices[source_cell], dtype=np.int64
                )
                target_vertex_rows = target_vertices[cell]
                parent_physical_vertices = source_coordinates[source_vertex_rows]
                for corner, target_reference in enumerate(target_topology.vertices):
                    source_reference = exact_map.evaluate(
                        tuple(Fraction(float(value)) for value in target_reference)
                    )
                    barycentric = (
                        Fraction(1) - sum(source_reference, Fraction(0)),
                        *source_reference,
                    )
                    if len(barycentric) != len(source_topology.vertices):
                        raise ValueError(
                            "Exact source reference map has the wrong dimension."
                        )
                    physical = tuple(
                        sum(
                            (
                                barycentric[source_corner]
                                * Fraction(
                                    float(parent_physical_vertices[source_corner, axis])
                                )
                                for source_corner in range(len(barycentric))
                            ),
                            Fraction(0),
                        )
                        for axis in range(self.mesh.ambient_dimension)
                    )
                    target_physical = tuple(
                        Fraction(float(value))
                        for value in target_coordinates[target_vertex_rows[corner]]
                    )
                    if physical != target_physical:
                        raise ValueError(
                            "Exact reference maps do not identify the represented P1 geometry."
                        )
            target_offset += block.cell_count

    def _transfer_degree(
        self,
        target: FiniteElementDeRhamComplex,
        degree: int,
        parents: np.ndarray | None,
        partition_cells: Mapping[int, tuple[PartitionCell, ...]] | None,
        reference_by_target: dict[int, NestedSimplexMap | NestedCellReferenceMap] | None,
        prepared_orientations: (dict[int, _PreparedReferenceChartOrientation] | None),
        integration_policy: CurvedOverlapIntegrationPolicy | None,
        /,
    ) -> tuple[
        SparseCoordinateOperator,
        SparseTransferEntryEvidence,
        int,
        int,
    ]:
        from ._exact_form_moments import (
            _reference_arguments,
            exact_reference_form_moments,
            MomentIntegralMatrix,
        )

        source_dofs, target_dofs = (
            self.discretization.dof_maps[degree],
            target.discretization.dof_maps[degree],
        )
        rows, columns, payload = [], [], []
        error_values, exact_counts, enclosed_counts = [], [], []
        owners: set[int] = set()
        exact_integral_count = enclosed_integral_count = 0
        offsets = np.cumsum((0, *(block.cell_count for block in self.mesh.blocks)))
        source_cell_global_ids = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in self.mesh.blocks]
        )
        source_cell_lookup = (
            {
                int(identifier): (block_index, row)
                for block_index, source_block in enumerate(self.mesh.blocks)
                for row, identifier in enumerate(np.asarray(source_block.global_ids))
            }
            if partition_cells is not None
            else {}
        )
        target_offset = 0
        source_vertex_id_bank = np.asarray(self.mesh.vertex_global_ids, dtype=np.int64)
        target_vertex_id_bank = np.asarray(target.mesh.vertex_global_ids, dtype=np.int64)
        cell_transform_cache: dict[
            tuple[str, tuple[int, ...], int],
            tuple[tuple[Fraction, ...], ...],
        ] = {}
        reference_gradient_cache: dict[str, np.ndarray] = {}
        moment_cache: dict[
            tuple[object, ...],
            tuple[FiniteElementSpec, FiniteElementSpec, object, MomentIntegralMatrix],
        ] = {}
        for block_index, (block, element) in enumerate(
            zip(target.mesh.blocks, target.discretization.elements[degree], strict=True)
        ):
            target_cell_dofs = np.asarray(
                target_dofs.cell_dofs[block_index], dtype=np.int64
            )
            for cell in range(block.cell_count):
                target_id = target_offset + cell
                target_global_rows = target_cell_dofs[cell]
                if all(int(row) in owners for row in target_global_rows):
                    continue
                if parents is not None and parents[target_id] == -1:
                    if partition_cells is None or element.form_basis is None:
                        raise ValueError(
                            "Coarsened FE cells require partitioned form moments."
                        )
                    from ._exact_form_moments import (
                        exact_partitioned_reference_form_moments,
                    )
                    from ._partition_form_moments import (
                        prepare_indexed_partitioned_reference_entities,
                    )

                    coarse_id = int(np.asarray(block.global_ids)[cell])
                    records = prepare_indexed_partitioned_reference_entities(
                        partition_cells.get(coarse_id, ()),
                        coarse_id,
                        element.form_basis,
                    )
                    if not records:
                        raise ValueError("A coarsened FE cell has no source partition.")
                    target_transform = _exact_form_cell_transform(
                        target.mesh,
                        block,
                        element,
                        cell,
                        target_vertex_id_bank,
                        cell_transform_cache,
                        reference_gradient_cache,
                    )
                    pending_rows = tuple(
                        local
                        for local, global_row in enumerate(target_global_rows)
                        if int(global_row) not in owners
                    )
                    for record in records:
                        source_location = source_cell_lookup.get(int(record.fine_cell_id))
                        if (
                            source_location is None
                            or int(record.coarse_cell_id) != coarse_id
                        ):
                            raise ValueError(
                                "Partitioned moments reference absent or mismatched cells."
                            )
                        source_block, source_cell = source_location
                        source_mesh_block = self.mesh.blocks[source_block]
                        source_element = self.discretization.elements[degree][
                            source_block
                        ]
                        if (
                            source_mesh_block.cell_kind != record.fine_cell_kind
                            or block.cell_kind != record.coarse_cell_kind
                        ):
                            raise ValueError(
                                "Partitioned moment cell kinds disagree with mesh blocks."
                            )
                        source_transform = _exact_form_cell_transform(
                            self.mesh,
                            source_mesh_block,
                            source_element,
                            source_cell,
                            source_vertex_id_bank,
                            cell_transform_cache,
                            reference_gradient_cache,
                        )
                        moments = exact_partitioned_reference_form_moments(
                            source_element,
                            element,
                            record,
                            source_transform,
                            target_transform,
                            integration_policy,
                        )
                        source_columns = np.asarray(
                            source_dofs.cell_dofs[source_block][source_cell],
                            dtype=np.int64,
                        )
                        for local_dof in pending_rows:
                            global_row = int(target_global_rows[local_dof])
                            local_values = moments.values[local_dof]
                            local_errors = moments.errors[local_dof]
                            keep = (local_values != 0.0) | (local_errors != 0.0)
                            rows.extend([global_row] * int(np.count_nonzero(keep)))
                            columns.extend(source_columns[keep].tolist())
                            payload.append(local_values[keep])
                            error_values.extend(local_errors[keep].tolist())
                            exact_counts.extend(
                                moments.exact[local_dof, keep].astype(np.int64).tolist()
                            )
                            enclosed_counts.extend(
                                moments.enclosed[local_dof, keep]
                                .astype(np.int64)
                                .tolist()
                            )
                            exact_integral_count += int(
                                np.count_nonzero(moments.exact[local_dof])
                            )
                            enclosed_integral_count += int(
                                np.count_nonzero(moments.enclosed[local_dof])
                            )
                    owners.update(int(target_global_rows[row]) for row in pending_rows)
                    continue
                reference_map = None
                if parents is None:
                    source_block = self.discretization.dof_maps[degree].block_names.index(
                        block.name
                    )
                    source_cell = cell
                else:
                    parent = int(parents[target_id])
                    source_block = int(np.searchsorted(offsets[1:], parent, side="right"))
                    source_cell = parent - int(offsets[source_block])
                    if reference_by_target is None:
                        raise ValueError(
                            "Nested FE transfer requires exact reference-map witnesses."
                        )
                    reference_map = reference_by_target[target_id]
                    mapped_parent = (
                        reference_map.source_cell_id
                        if isinstance(reference_map, NestedCellReferenceMap)
                        else reference_map.coarse_cell_id
                    )
                    if mapped_parent != int(source_cell_global_ids[parent]):
                        raise ValueError(
                            "Exact reference-map parent disagrees with parent_cells."
                        )
                prepared_orientation = (
                    None
                    if prepared_orientations is None
                    else prepared_orientations.get(target_id)
                )
                if reference_map is not None and prepared_orientation is None:
                    raise ValueError(
                        "An exact FE reference map lacks prepared orientation evidence."
                    )
                source_element = self.discretization.elements[degree][source_block]
                source_basis = source_element.form_basis
                target_basis = element.form_basis
                if source_basis is None or target_basis is None:
                    raise ValueError("Nested FE transfer requires canonical form bases.")
                source_block_mesh = self.mesh.blocks[source_block]
                source_transform = _exact_form_cell_transform(
                    self.mesh,
                    source_block_mesh,
                    source_element,
                    source_cell,
                    source_vertex_id_bank,
                    cell_transform_cache,
                    reference_gradient_cache,
                )
                target_transform = _exact_form_cell_transform(
                    target.mesh,
                    block,
                    element,
                    cell,
                    target_vertex_id_bank,
                    cell_transform_cache,
                    reference_gradient_cache,
                )
                if prepared_orientation is not None:
                    prepared_orientation.require(
                        _reference_arguments(reference_map, target_basis.dimension),
                        target_basis.family,
                    )
                moment_key = (
                    id(source_element),
                    id(element),
                    source_transform,
                    target_transform,
                    prepared_orientation,
                )
                cached = moment_cache.get(moment_key)
                if cached is None:
                    moments = exact_reference_form_moments(
                        source_element,
                        element,
                        reference_map,
                        source_transform,
                        target_transform,
                        integration_policy,
                        prepared_orientation=prepared_orientation,
                    )
                    moment_cache[moment_key] = (
                        source_element,
                        element,
                        integration_policy,
                        moments,
                    )
                else:
                    cached_source, cached_target, cached_policy, moments = cached
                    if (
                        cached_source is not source_element
                        or cached_target is not element
                        or cached_policy is not integration_policy
                    ):
                        raise RuntimeError(
                            "Exact form moment owners changed during transfer."
                        )
                source_columns = np.asarray(
                    source_dofs.cell_dofs[source_block][source_cell], dtype=np.int64
                )
                for local_dof, global_row in enumerate(target_global_rows):
                    if int(global_row) in owners:
                        continue
                    owners.add(int(global_row))
                    local_values = moments.values[local_dof]
                    local_errors = moments.errors[local_dof]
                    keep = (local_values != 0.0) | (local_errors != 0.0)
                    rows.extend([int(global_row)] * int(np.count_nonzero(keep)))
                    columns.extend(source_columns[keep].tolist())
                    payload.append(local_values[keep])
                    error_values.extend(local_errors[keep].tolist())
                    exact_counts.extend(
                        moments.exact[local_dof, keep].astype(np.int64).tolist()
                    )
                    enclosed_counts.extend(
                        moments.enclosed[local_dof, keep].astype(np.int64).tolist()
                    )
                    exact_integral_count += int(
                        np.count_nonzero(moments.exact[local_dof])
                    )
                    enclosed_integral_count += int(
                        np.count_nonzero(moments.enclosed[local_dof])
                    )
            target_offset += block.cell_count
        raw_rows = np.asarray(rows, dtype=np.int32)
        raw_columns = np.asarray(columns, dtype=np.int32)
        raw_values = (
            jnp.asarray(np.concatenate(payload), dtype=jnp.float64)
            if payload
            else jnp.zeros(
                (0,), dtype=_require_array_space(self._absolute.space(degree)).dtype
            )
        )
        return self._finish_transfer_degree(
            target,
            degree,
            raw_rows,
            raw_columns,
            raw_values,
            error_values,
            exact_counts,
            enclosed_counts,
            exact_integral_count,
            enclosed_integral_count,
        )

    def _finish_transfer_degree(
        self,
        target: FiniteElementDeRhamComplex,
        degree: int,
        raw_rows: np.ndarray,
        raw_columns: np.ndarray,
        raw_values: Array,
        error_values: Sequence[float],
        exact_counts: Sequence[int],
        enclosed_counts: Sequence[int],
        exact_integral_count: int,
        enclosed_integral_count: int,
        /,
    ) -> tuple[SparseCoordinateOperator, SparseTransferEntryEvidence, int, int]:
        """Finalize already prepared exact moment entries and their evidence."""
        source_dofs = self.discretization.dof_maps[degree]
        target_dofs = target.discretization.dof_maps[degree]
        row, column, values = _coalesce_entries(
            raw_rows,
            raw_columns,
            raw_values,
            source_dofs.global_dof_count,
        )
        evidence_rows, evidence_columns, errors, exact, enclosed = (
            _coalesce_transfer_entry_evidence(
                raw_rows,
                raw_columns,
                np.asarray(raw_values, dtype=np.float64),
                np.asarray(error_values, dtype=np.float64),
                np.asarray(exact_counts, dtype=np.int64),
                np.asarray(enclosed_counts, dtype=np.int64),
                source_dofs.global_dof_count,
            )
        )
        if not np.array_equal(row, evidence_rows) or not np.array_equal(
            column, evidence_columns
        ):
            raise RuntimeError(
                "FE transfer values and integration evidence coalesced differently."
            )
        operator = _sparse_operator(
            row,
            column,
            values,
            self._absolute.space(degree),
            target._absolute.space(degree),
            f"{self.realization_id}:transfer:{target.realization_id}:{degree}",
        )
        return (
            operator,
            SparseTransferEntryEvidence(
                degree,
                source_dofs.global_dof_count,
                target_dofs.global_dof_count,
                evidence_rows,
                evidence_columns,
                errors,
                exact,
                enclosed,
            ),
            exact_integral_count,
            enclosed_integral_count,
        )


class _ReferenceChartPiece(NamedTuple):
    source_row: int
    target_row: int
    target_vertices: tuple[tuple[Fraction, Fraction], ...]
    source_vertices: tuple[tuple[Fraction, Fraction], ...]
    projective_coefficients: tuple[tuple[Fraction, ...], ...] | None


def _chart_cell_location(mesh: CellMesh, row: int, /) -> tuple[int, int]:
    offset = 0
    for block_index, block in enumerate(mesh.blocks):
        if row < offset + block.cell_count:
            return block_index, row - offset
        offset += block.cell_count
    raise IndexError("Material chart cell row is outside its mesh.")


def _chart_vertex_permutation(
    chart_mesh: CellMesh,
    physical_mesh: CellMesh,
    cell_row: int,
    /,
) -> tuple[int, int, int]:
    chart_block_index, chart_cell = _chart_cell_location(chart_mesh, cell_row)
    physical_block_index, physical_cell = _chart_cell_location(physical_mesh, cell_row)
    chart_block = chart_mesh.blocks[chart_block_index]
    physical_block = physical_mesh.blocks[physical_block_index]
    if (
        chart_block.cell_kind != "triangle"
        or physical_block.cell_kind != "triangle"
        or int(np.asarray(chart_block.global_ids)[chart_cell])
        != int(np.asarray(physical_block.global_ids)[physical_cell])
    ):
        raise ValueError(
            "Material chart cell rows do not preserve triangular identities."
        )
    chart_vertices = np.asarray(chart_block.vertices[chart_cell], dtype=np.int64)
    physical_vertices = np.asarray(physical_block.vertices[physical_cell], dtype=np.int64)
    chart_ids = np.asarray(chart_mesh.vertex_global_ids, dtype=np.int64)[chart_vertices]
    physical_ids = np.asarray(physical_mesh.vertex_global_ids, dtype=np.int64)[
        physical_vertices
    ]
    if set(chart_ids.tolist()) != set(physical_ids.tolist()):
        raise ValueError("Material chart cells do not preserve their vertex identities.")
    return (
        int(np.flatnonzero(physical_ids == chart_ids[0])[0]),
        int(np.flatnonzero(physical_ids == chart_ids[1])[0]),
        int(np.flatnonzero(physical_ids == chart_ids[2])[0]),
    )


def _rechart_reference_triangle(
    vertices: tuple[tuple[Fraction, Fraction], ...],
    permutation: tuple[int, int, int],
    /,
) -> tuple[tuple[Fraction, Fraction], ...]:
    result = []
    for x, y in vertices:
        physical_barycentric = (Fraction(1) - x - y, x, y)
        chart_barycentric = tuple(physical_barycentric[index] for index in permutation)
        result.append((chart_barycentric[1], chart_barycentric[2]))
    return tuple(result)


def _rechart_reference_piece(
    piece: _ReferenceChartPiece,
    source_chart_mesh: CellMesh,
    target_chart_mesh: CellMesh,
    source_physical_mesh: CellMesh,
    target_physical_mesh: CellMesh,
    /,
) -> _ReferenceChartPiece:
    source_permutation = _chart_vertex_permutation(
        source_chart_mesh, source_physical_mesh, piece.source_row
    )
    target_permutation = _chart_vertex_permutation(
        target_chart_mesh, target_physical_mesh, piece.target_row
    )
    coefficients = piece.projective_coefficients
    if coefficients is not None:
        coefficients = tuple(
            tuple(
                coefficients[source_permutation[row]][target_permutation[column]]
                for column in range(3)
            )
            for row in range(3)
        )
    return _ReferenceChartPiece(
        piece.source_row,
        piece.target_row,
        _rechart_reference_triangle(piece.target_vertices, target_permutation),
        _rechart_reference_triangle(piece.source_vertices, source_permutation),
        coefficients,
    )


def _chart_element_for_twist(
    element: FiniteElementSpec, twist: FormTwist, /
) -> FiniteElementSpec:
    basis = element.form_basis
    if basis is None:
        raise ValueError("Chart form transfer requires canonical form elements.")
    if basis.twist == twist:
        return element
    selected = form_element(
        element.cell_kind,
        basis.form_degree,
        basis.order,
        family=basis.family,
        twist=twist,
        proxy=element.value_spec.proxy,
    )
    if selected.local_dof_count != element.local_dof_count:
        raise ValueError("Twist changes the canonical chart functional layout.")
    return selected


def _chart_cross(
    first: tuple[Fraction, Fraction],
    second: tuple[Fraction, Fraction],
    point: tuple[Fraction, Fraction],
    /,
) -> Fraction:
    return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (
        point[0] - first[0]
    )


def _chart_contains(
    triangle: tuple[tuple[Fraction, Fraction], ...],
    point: tuple[Fraction, Fraction],
    /,
) -> bool:
    signed_area = _chart_cross(triangle[0], triangle[1], triangle[2])
    if signed_area == 0:
        raise ValueError("A reference chart piece has a singular target triangle.")
    sign = 1 if signed_area > 0 else -1
    return all(
        sign * _chart_cross(first, second, point) >= 0
        for first, second in zip(triangle, (*triangle[1:], triangle[0]), strict=True)
    )


def _chart_segment_interval(
    first: tuple[Fraction, Fraction],
    last: tuple[Fraction, Fraction],
    triangle: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> tuple[Fraction, Fraction] | None:
    orientation = _chart_cross(triangle[0], triangle[1], triangle[2])
    if orientation == 0:
        raise ValueError("A reference chart piece has a singular target triangle.")
    sign = 1 if orientation > 0 else -1
    lower, upper = Fraction(0), Fraction(1)
    for edge_first, edge_last in zip(triangle, (*triangle[1:], triangle[0]), strict=True):
        left = sign * _chart_cross(edge_first, edge_last, first)
        right = sign * _chart_cross(edge_first, edge_last, last)
        if left < 0 and right < 0:
            return None
        if left < 0:
            lower = max(lower, left / (left - right))
        elif right < 0:
            upper = min(upper, left / (left - right))
        if lower >= upper:
            return None
    return lower, upper


def _unit_gauss(order: int, /) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (points + 1.0), 0.5 * weights


def _entity_piece_quadrature(
    basis: FormBasis,
    entity: tuple[int, ...],
    piece: _ReferenceChartPiece,
    order: int,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Cubature points in the declared entity chart on its piece intersection."""
    topology = reference_cell_topology("triangle")
    vertices = tuple(
        (Fraction(float(point[0])), Fraction(float(point[1])))
        for point in topology.vertices
    )
    entity_vertices = tuple(vertices[index] for index in entity)
    dimension = len(entity) - 1
    triangle = piece.target_vertices
    if dimension == 0:
        if not _chart_contains(triangle, entity_vertices[0]):
            return None
        return np.empty((1, 0), dtype=np.float64), np.ones((1,), dtype=np.float64)
    points_1d, weights_1d = _unit_gauss(order)
    if dimension == 1:
        interval = _chart_segment_interval(
            entity_vertices[0], entity_vertices[1], triangle
        )
        if interval is None:
            return None
        lower, upper = interval
        span = float(upper - lower)
        points = np.asarray(
            [float(lower) + span * point for point in points_1d],
            dtype=np.float64,
        )
        return points[:, None], weights_1d * span
    if dimension != 2:
        raise ValueError("Reference chart transfer requires simplex entities.")
    face = entity_vertices
    face_frame = (
        (face[1][0] - face[0][0], face[2][0] - face[0][0]),
        (face[1][1] - face[0][1], face[2][1] - face[0][1]),
    )
    face_transpose = tuple(zip(*face_frame, strict=True))
    exact_points = tuple(
        tuple(
            row[0]
            for row in solve_exact(
                face_transpose,
                tuple((vertex[axis] - face[0][axis],) for axis in range(2)),
            )
        )
        for vertex in triangle
    )
    origin = exact_points[0]
    frame = (
        (exact_points[1][0] - origin[0], exact_points[2][0] - origin[0]),
        (exact_points[1][1] - origin[1], exact_points[2][1] - origin[1]),
    )
    determinant = abs(frame[0][0] * frame[1][1] - frame[0][1] * frame[1][0])
    if determinant == 0:
        raise ValueError("A reference chart piece has a singular target triangle.")
    second_points, second_weights = _unit_gauss(order)
    first_grid, second_grid = np.meshgrid(points_1d, second_points, indexing="ij")
    reference_points = np.column_stack(
        (
            first_grid.reshape(-1),
            ((1.0 - first_grid) * second_grid).reshape(-1),
        )
    )
    affine = np.asarray(frame, dtype=np.float64)
    chart_points = (
        np.asarray(origin, dtype=np.float64)[None, :] + reference_points @ affine.T
    )
    quadrature = (
        weights_1d[:, None] * second_weights[None, :] * (1.0 - first_grid)
    ).reshape(-1)
    return chart_points, quadrature * float(determinant)


def _reference_chart_map(
    piece: _ReferenceChartPiece,
    points: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if piece.projective_coefficients is None:
        target = piece.target_vertices
        source = piece.source_vertices
        target_frame = (
            (target[1][0] - target[0][0], target[2][0] - target[0][0]),
            (target[1][1] - target[0][1], target[2][1] - target[0][1]),
        )
        source_frame = (
            (source[1][0] - source[0][0], source[2][0] - source[0][0]),
            (source[1][1] - source[0][1], source[2][1] - source[0][1]),
        )
        exact_jacobian_transpose = solve_exact(
            tuple(zip(*target_frame, strict=True)),
            tuple(zip(*source_frame, strict=True)),
        )
        jacobian = np.asarray(exact_jacobian_transpose, dtype=np.float64).T
        target_origin = np.asarray(target[0], dtype=np.float64)
        source_origin = np.asarray(source[0], dtype=np.float64)
        source_points = source_origin + (points - target_origin) @ jacobian.T
        jacobians = np.broadcast_to(jacobian, (points.shape[0], 2, 2))
        return source_points, jacobians
    coefficients = np.asarray(piece.projective_coefficients, dtype=np.float64)
    barycentric = np.column_stack((1.0 - points[:, 0] - points[:, 1], points))
    homogeneous = barycentric @ coefficients.T
    denominator = np.sum(homogeneous, axis=1)
    if np.any(~np.isfinite(denominator) | (denominator <= 0.0)):
        raise ValueError("A projective reference map has a nonpositive denominator.")
    source_points = homogeneous[:, 1:] / denominator[:, None]
    denominator_coefficients = np.sum(coefficients, axis=0)
    denominator_gradient = denominator_coefficients[1:] - denominator_coefficients[:1]
    jacobians = np.empty((points.shape[0], 2, 2), dtype=np.float64)
    for coordinate in range(2):
        numerator = homogeneous[:, coordinate + 1]
        numerator_gradient = (
            coefficients[coordinate + 1, 1:] - coefficients[coordinate + 1, :1]
        )
        jacobians[:, coordinate, :] = (
            numerator_gradient[None, :] * denominator[:, None]
            - numerator[:, None] * denominator_gradient[None, :]
        ) / denominator[:, None] ** 2
    if not np.all(np.isfinite(source_points)) or not np.all(np.isfinite(jacobians)):
        raise ValueError("A projective reference map has no finite differential.")
    return source_points, jacobians


def _reference_chart_form_action(
    source: FiniteElementDeRhamComplex,
    target: FiniteElementDeRhamComplex,
    degree: int,
    pieces: Sequence[_ReferenceChartPiece],
    identifier: str,
    /,
    *,
    form_twist: FormTwist | None = None,
) -> SparseCoordinateOperator:
    """Interpolate exact chart correspondences by canonical form functionals."""
    if (
        source.dimension != 2
        or target.dimension != 2
        or source.dimension != target.dimension
        or source.primal_twist != target.primal_twist
        or degree < 0
        or degree > 2
    ):
        raise ValueError(
            "Chart form transfer requires matching two-dimensional complexes."
        )
    selected_twist = (
        source.primal_twist
        if form_twist is None
        else parse(form_twist, FormTwist, "form_twist")
    )
    source_elements = tuple(
        _chart_element_for_twist(element, selected_twist)
        for element in source.discretization.elements[degree]
    )
    target_elements = tuple(
        _chart_element_for_twist(element, selected_twist)
        for element in target.discretization.elements[degree]
    )
    source_dofs = source.discretization.dof_maps[degree]
    target_dofs = target.discretization.dof_maps[degree]
    source_offsets = np.cumsum(
        (0, *(block.cell_count for block in source.mesh.blocks)), dtype=np.int64
    )
    target_offsets = np.cumsum(
        (0, *(block.cell_count for block in target.mesh.blocks)), dtype=np.int64
    )
    source_rows = int(source_offsets[-1])
    target_rows = int(target_offsets[-1])
    if len(pieces) == 0:
        raise ValueError("Chart form transfer requires material overlap pieces.")
    by_target: dict[int, list[_ReferenceChartPiece]] = {}
    for piece in pieces:
        if (
            piece.source_row < 0
            or piece.source_row >= source_rows
            or piece.target_row < 0
            or piece.target_row >= target_rows
            or len(piece.target_vertices) != 3
            or len(piece.source_vertices) != 3
        ):
            raise ValueError(
                "Chart form pieces do not bind their source and target cells."
            )
        by_target.setdefault(piece.target_row, []).append(piece)
    if set(by_target) != set(range(target_rows)):
        raise ValueError("Chart form pieces must cover every target cell.")
    source_vertex_ids = np.asarray(source.mesh.vertex_global_ids, dtype=np.int64)
    target_vertex_ids = np.asarray(target.mesh.vertex_global_ids, dtype=np.int64)
    source_transform_cache: dict[
        tuple[str, tuple[int, ...], int], tuple[tuple[Fraction, ...], ...]
    ] = {}
    target_transform_cache: dict[
        tuple[str, tuple[int, ...], int], tuple[tuple[Fraction, ...], ...]
    ] = {}
    reference_gradient_cache: dict[str, np.ndarray] = {}
    source_routes_by_row: dict[int, np.ndarray] = {}
    source_cell_by_row: dict[int, tuple[int, int]] = {}
    for block_index, (block, element, routes) in enumerate(
        zip(
            source.mesh.blocks,
            source_elements,
            source_dofs.cell_dofs,
            strict=True,
        )
    ):
        if block.cell_kind != "triangle" or element.form_basis is None:
            raise ValueError(
                "Chart form transfer requires canonical triangular elements."
            )
        for cell in range(block.cell_count):
            row = int(source_offsets[block_index]) + cell
            source_routes_by_row[row] = np.asarray(routes[cell], dtype=np.int64)
            source_cell_by_row[row] = (block_index, cell)
    resolved_rows: dict[int, dict[int, float]] = {}
    for target_block_index, (target_block, target_element, target_routes) in enumerate(
        zip(
            target.mesh.blocks,
            target_elements,
            target_dofs.cell_dofs,
            strict=True,
        )
    ):
        basis = target_element.form_basis
        if target_block.cell_kind != "triangle" or basis is None:
            raise ValueError(
                "Chart form transfer requires canonical triangular elements."
            )
        for target_cell in range(target_block.cell_count):
            target_row = int(target_offsets[target_block_index]) + target_cell
            selected = sorted(
                by_target[target_row],
                key=lambda value: (
                    value.source_row,
                    value.target_vertices,
                    value.source_vertices,
                ),
            )
            source_columns = np.unique(
                np.concatenate(
                    [source_routes_by_row[piece.source_row] for piece in selected]
                )
            )
            column_positions = {
                int(column): position for position, column in enumerate(source_columns)
            }
            topology_vertices = tuple(
                (Fraction(float(point[0])), Fraction(float(point[1])))
                for point in reference_cell_topology("triangle").vertices
            )
            point_piece_owner: dict[tuple[int, ...], _ReferenceChartPiece] = {}
            for entity in dict.fromkeys(label[0] for label in basis.dof_labels):
                if len(entity) != 1:
                    continue
                point = topology_vertices[entity[0]]
                owner = next(
                    (
                        piece
                        for piece in selected
                        if _chart_contains(piece.target_vertices, point)
                    ),
                    None,
                )
                if owner is None:
                    raise ValueError("Chart form pieces omit a target vertex functional.")
                point_piece_owner[entity] = owner
            local_moments = np.zeros(
                (basis.local_dof_count, source_columns.size), dtype=np.float64
            )
            target_block_rows = np.asarray(target_routes[target_cell], dtype=np.int64)
            for piece in selected:
                source_block_index, source_cell = source_cell_by_row[piece.source_row]
                source_element = source_elements[source_block_index]
                source_basis = source_element.form_basis
                if source_basis is None:
                    raise ValueError(
                        "Chart source elements require canonical form bases."
                    )
                minimum_order = 24 if piece.projective_coefficients is not None else 12
                quadrature_order = max(
                    minimum_order, 2 * (source_basis.order + basis.order) + 8
                )
                source_transform_exact = _exact_form_cell_transform(
                    source.mesh,
                    source.mesh.blocks[source_block_index],
                    source_element,
                    source_cell,
                    source_vertex_ids,
                    source_transform_cache,
                    reference_gradient_cache,
                )
                source_transform = np.asarray(source_transform_exact, dtype=np.float64)
                source_routes = source_routes_by_row[piece.source_row]
                route_positions = np.asarray(
                    [column_positions[int(route)] for route in source_routes],
                    dtype=np.int64,
                )
                for entity in dict.fromkeys(label[0] for label in basis.dof_labels):
                    if len(entity) == 1 and piece != point_piece_owner[entity]:
                        continue
                    quadrature = _entity_piece_quadrature(
                        basis, entity, piece, quadrature_order
                    )
                    if quadrature is None:
                        continue
                    entity_points, entity_weights = quadrature
                    target_points, test_weights = basis.functional_weights_at(
                        entity, entity_points, entity_weights
                    )
                    target_points_host = np.asarray(target_points, dtype=np.float64)
                    source_points, jacobians = _reference_chart_map(
                        piece, target_points_host
                    )
                    source_values = np.asarray(
                        source_basis.tabulate_components(source_points)[0],
                        dtype=np.float64,
                    )
                    pulled = np.asarray(
                        pullback(
                            source_values,
                            FormType(source.dimension, degree, twist=selected_twist),
                            np.expand_dims(jacobians, axis=1),
                        ),
                        dtype=np.float64,
                    )
                    weights = np.asarray(test_weights, dtype=np.float64)
                    action = np.einsum("dqc,qsc->ds", weights, pulled, optimize=True)
                    action = action @ source_transform
                    target_entity_dofs = np.asarray(
                        [
                            local
                            for local, label in enumerate(basis.dof_labels)
                            if label[0] == entity
                        ],
                        dtype=np.int64,
                    )
                    local_moments[np.ix_(target_entity_dofs, route_positions)] += action[
                        target_entity_dofs
                    ]
            target_transform_exact = _exact_form_cell_transform(
                target.mesh,
                target_block,
                target_element,
                target_cell,
                target_vertex_ids,
                target_transform_cache,
                reference_gradient_cache,
            )
            target_transform_inverse = np.asarray(
                solve_exact(
                    target_transform_exact,
                    tuple(
                        tuple(
                            Fraction(int(row == column))
                            for column in range(basis.local_dof_count)
                        )
                        for row in range(basis.local_dof_count)
                    ),
                ),
                dtype=np.float64,
            )
            global_action = target_transform_inverse @ local_moments
            for local_row, target_global_row in enumerate(target_block_rows):
                candidate = {
                    int(column): float(value)
                    for column, value in zip(
                        source_columns, global_action[local_row], strict=True
                    )
                    if value != 0.0
                }
                global_row = int(target_global_row)
                previous = resolved_rows.get(global_row)
                if previous is None:
                    resolved_rows[global_row] = candidate
                    continue
                columns = set(previous) | set(candidate)
                if any(
                    not np.isclose(
                        previous.get(column, 0.0),
                        candidate.get(column, 0.0),
                        rtol=2.0e-9,
                        atol=2.0e-11,
                    )
                    for column in columns
                ):
                    raise ValueError(
                        "Adjacent target charts disagree on a shared form functional."
                    )
    if len(resolved_rows) != target_dofs.global_dof_count:
        raise ValueError("Chart form transfer did not define every target functional.")
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row in sorted(resolved_rows):
        for column, value in sorted(resolved_rows[row].items()):
            rows.append(row)
            columns.append(column)
            values.append(value)
    return _sparse_operator(
        np.asarray(rows, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
        np.asarray(values, dtype=np.float64),
        source._absolute.space(degree),
        target._absolute.space(degree),
        identifier,
    )


__all__ = [
    "DistributedFiniteElementTransfer",
    "FiniteElementDeRhamComplex",
    "FiniteElementTransferResult",
    "PeriodicFiniteElementTransfer",
    "SparseCommutingEvidence",
    "SparseTransferEntryEvidence",
]
