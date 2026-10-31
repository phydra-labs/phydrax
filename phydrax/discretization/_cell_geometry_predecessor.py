from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._cell_complex import PolyhedralConnectivity
from ._cell_geometry import (
    CellGeometryElement,
    CellGeometrySpec,
    CellVertexGeometryElement,
    coordinate_effective_basis,
    coordinate_execution_basis,
    coordinate_source_basis,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from ._cell_geometry_validity import (
    _BlockCertificate,
    _certify_polygon_block,
    _certify_polyhedral_cells,
    _certify_source_block,
    _element_basis_rounding_error,
    _expression_sequence_evidence_equal,
    _polyhedral_coordinates,
    _source_proof_domain,
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityStatus,
    polyhedral_star_tables,
)
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import (
    affine_map,
    coordinate_execution_map_expressions,
    coordinate_map_expressions,
    coordinate_point_rounding_error,
    CoordinateSourceBank,
    ExactScalar,
    Expression,
    RationalPolynomial,
)
from ._exact_algebraic_geometry import coefficient_endpoint, coefficient_payload


_PREDECESSOR_TOKEN = object()


def _polynomial_payload(
    value: Mapping[tuple[int, ...], ExactScalar], /
) -> tuple[tuple[tuple[int, ...], object], ...]:
    return tuple(
        (exponent, coefficient_payload(value[exponent])) for exponent in sorted(value)
    )


def _expression_payload(value: Expression, /) -> tuple[str, object]:
    if isinstance(value, RationalPolynomial):
        return "rational", (
            _polynomial_payload(value.numerator),
            _polynomial_payload(value.denominator),
        )
    return "polynomial", _polynomial_payload(value)


def _cell_keys(mesh: CellMesh, geometry: CellGeometrySpec, /) -> tuple[str, ...]:
    elements, routes, _ = geometry.resolve(mesh)
    coefficients = geometry.source_coordinates()
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    keys: list[str] = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        route_rows = np.asarray(route, dtype=np.int64)
        cell_vertices = np.asarray(block.vertices, dtype=np.int64)
        cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        for row in range(block.cell_count):
            dofs = route_rows[row]
            local = tuple(coefficients[int(index)] for index in dofs)
            if isinstance(element, CellVertexGeometryElement):
                basis_payload: object = None
                degree: int | None = None
            else:
                if not isinstance(element, ScalarCoordinateElement):
                    raise TypeError("Cell validity predecessor requires scalar geometry.")
                scalar = element
                basis_payload = tuple(
                    _expression_payload(value)
                    for value in coordinate_effective_basis(scalar, cell_row=row)
                )
                degree = scalar.degree
            keys.append(
                canonical_fingerprint(
                    {
                        "block": block.name,
                        "kind": block.cell_kind,
                        "cell_id": int(cell_ids[row]),
                        "vertex_ids": vertex_ids[cell_vertices[row]],
                        "vertex_coordinates": coordinates[cell_vertices[row]],
                        "element_type": type(element).__qualname__,
                        "element_degree": degree,
                        "basis": basis_payload,
                        "local_coefficients": tuple(
                            tuple(coefficient_payload(value) for value in row)
                            for row in local
                        ),
                        "source_map": geometry.source_map_id,
                    }
                )
            )
    return tuple(keys)


def _predecessor_content_id(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_validity: CellValidityCertificate,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    source_rows: np.ndarray,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "cell-geometry-predecessor",
            "source_mesh": array_tree_fingerprint(source_mesh),
            "source_geometry": array_tree_fingerprint(source_geometry),
            "source_validity": array_tree_fingerprint(source_validity),
            "target_mesh": array_tree_fingerprint(target_mesh),
            "target_geometry": array_tree_fingerprint(target_geometry),
            "source_rows": source_rows,
        }
    )


@dataclass(frozen=True)
class _TargetProofContext:
    execution_coordinates: CoordinateSourceBank
    source_coordinates: CoordinateSourceBank | None
    distinct_source: bool
    maximum_coefficient_rounding_error: float
    maximum_element_coefficient_rounding_error: float
    unsupported_block_names: frozenset[str]


def _target_proof_context(
    geometry: CellGeometrySpec,
    names: tuple[str, ...],
    elements: tuple[CellGeometryElement, ...],
    routes: tuple[np.ndarray, ...],
    /,
) -> _TargetProofContext:
    values = np.asarray(geometry.coordinates, dtype=np.float64)
    execution_coordinates: CoordinateSourceBank = tuple(
        tuple(Fraction(float(value)) for value in row) for row in values
    )
    source_coordinates = (
        geometry.source_coordinates()
        if geometry.exact_coordinate_bank is not None
        else None
    )
    distinct_source = (
        source_coordinates is not None and source_coordinates != execution_coordinates
    )
    coefficient_rounding_error: ExactScalar = (
        max(
            (
                coordinate_point_rounding_error(point, values[row])
                for row, point in enumerate(source_coordinates)
            ),
            default=Fraction(0),
        )
        if source_coordinates is not None
        else Fraction(0)
    )
    element_rounding_error: ExactScalar = Fraction(0)
    unsupported: set[str] = set()
    for name, element, route in zip(names, elements, routes, strict=True):
        if not isinstance(element, ScalarCoordinateElement):
            continue
        for row in range(route.shape[0]):
            try:
                source_basis = coordinate_source_basis(element, cell_row=row)
                execution_basis = coordinate_execution_basis(element, cell_row=row)
            except (OverflowError, TypeError, ValueError):
                distinct_source = True
                unsupported.add(name)
                continue
            distinct_source = distinct_source or not _expression_sequence_evidence_equal(
                source_basis, execution_basis
            )
            element_rounding_error = max(
                element_rounding_error,
                _element_basis_rounding_error(
                    source_basis,
                    execution_basis,
                    element.topological_dimension,
                ),
            )
    return _TargetProofContext(
        execution_coordinates,
        source_coordinates,
        distinct_source,
        coefficient_endpoint(coefficient_rounding_error, upper=True, saturate=True),
        coefficient_endpoint(element_rounding_error, upper=True, saturate=True),
        frozenset(unsupported),
    )


def _unresolved_proof(count: int, /) -> _BlockCertificate:
    return _BlockCertificate(
        np.full((count,), CellValidityStatus.UNRESOLVED, dtype=np.int32),
        np.full((count,), np.nan, dtype=np.float64),
        np.full((count,), np.nan, dtype=np.float64),
        np.zeros((count,), dtype=np.int64),
    )


def _selected_element(
    element: ScalarCoordinateElement, rows: np.ndarray, /
) -> ScalarCoordinateElement:
    if not isinstance(element, PerCellRestrictedCellGeometryElement):
        return element
    return PerCellRestrictedCellGeometryElement(
        element.source_element,
        element.cell_kind,
        tuple(element.target_cell_ids[int(row)] for row in rows),
        tuple(element.source_cell_ids[int(row)] for row in rows),
        tuple(element.chart_arguments(int(row)) for row in rows),
    )


def _map_is_unsupported(
    element: ScalarCoordinateElement,
    cell_kind: str,
    coordinates: CoordinateSourceBank,
    route: np.ndarray,
    cell_row: int,
    /,
    *,
    execution_map: bool,
) -> bool:
    if _source_proof_domain(cell_kind) is not None:
        return False
    local = tuple(coordinates[int(index)] for index in route[cell_row])
    try:
        expressions = (
            coordinate_execution_map_expressions(element, local, cell_row=cell_row)
            if execution_map
            else coordinate_map_expressions(element, local, cell_row=cell_row)
        )
        return affine_map(expressions, element.topological_dimension) is None
    except (OverflowError, TypeError, ValueError):
        return True


class CellGeometryPredecessor(StrictModule, NonTrainableState):
    """Bind exact unchanged target cells to predecessor validity rows."""

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_validity: CellValidityCertificate
    target_mesh: CellMesh
    target_geometry: CellGeometrySpec
    source_rows: Array
    reused_cell_ids: tuple[int, ...] = eqx.field(static=True)
    predecessor_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        source_validity: CellValidityCertificate,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        source_rows: ArrayLike,
        reused_cell_ids: tuple[int, ...],
        predecessor_id: str,
        /,
        *,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _PREDECESSOR_TOKEN:
            raise TypeError(
                "CellGeometryPredecessor must be issued by its preparation owner."
            )
        self.source_mesh = source_mesh
        self.source_geometry = source_geometry
        self.source_validity = source_validity
        self.target_mesh = target_mesh
        self.target_geometry = target_geometry
        self.source_rows = jnp.asarray(source_rows, dtype=jnp.int64)
        self.reused_cell_ids = reused_cell_ids
        self.predecessor_id = predecessor_id
        self.require_current()

    def require_current(self, /) -> None:
        self.source_validity.require_bound(self.source_geometry, mesh=self.source_mesh)
        source_count = sum(block.cell_count for block in self.source_mesh.blocks)
        target_count = sum(block.cell_count for block in self.target_mesh.blocks)
        rows = np.asarray(self.source_rows, dtype=np.int64)
        if (
            rows.shape != (target_count,)
            or np.any(rows < -1)
            or np.any(rows >= source_count)
        ):
            raise ValueError(
                "Predecessor source rows must bind each target cell or use -1."
            )
        expected = _predecessor_content_id(
            self.source_mesh,
            self.source_geometry,
            self.source_validity,
            self.target_mesh,
            self.target_geometry,
            rows,
        )
        target_ids = tuple(
            int(identifier)
            for block in self.target_mesh.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64)
        )
        expected_reused = tuple(
            target_ids[row] for row in range(target_count) if rows[row] >= 0
        )
        if expected_reused != self.reused_cell_ids:
            raise ValueError("Predecessor reused cell IDs disagree with its source rows.")
        if expected != self.predecessor_id:
            raise ValueError(
                "Cell geometry predecessor content changed after preparation."
            )


@checked
def prepare_cell_geometry_predecessor(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_validity: CellValidityCertificate,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
) -> CellGeometryPredecessor:
    """Match target cells only when their complete exact map and carrier agree."""
    if not isinstance(source_validity, CellValidityCertificate):
        raise TypeError("source_validity must be a CellValidityCertificate.")
    source_validity.require_bound(source_geometry, mesh=source_mesh)
    source_keys = _cell_keys(source_mesh, source_geometry)
    target_keys = _cell_keys(target_mesh, target_geometry)
    source_row_by_key = {key: row for row, key in enumerate(source_keys)}
    source_rows = np.asarray(
        [source_row_by_key.get(key, -1) for key in target_keys], dtype=np.int64
    )
    target_ids = tuple(
        int(identifier)
        for block in target_mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    )
    reused = tuple(
        target_ids[row] for row in range(len(target_ids)) if source_rows[row] >= 0
    )
    predecessor_id = _predecessor_content_id(
        source_mesh,
        source_geometry,
        source_validity,
        target_mesh,
        target_geometry,
        source_rows,
    )
    return CellGeometryPredecessor(
        source_mesh,
        source_geometry,
        source_validity,
        target_mesh,
        target_geometry,
        source_rows,
        reused,
        predecessor_id,
        _construction_token=_PREDECESSOR_TOKEN,
    )


@checked
def reuse_cell_geometry_validity(
    predecessor: CellGeometryPredecessor, /
) -> CellValidityCertificate:
    """Build target evidence from predecessor rows and fresh changed-cell proofs."""
    if not isinstance(predecessor, CellGeometryPredecessor):
        raise TypeError("predecessor must be a CellGeometryPredecessor.")
    predecessor.require_current()
    source_validity = predecessor.source_validity
    source_keys = _cell_keys(predecessor.source_mesh, predecessor.source_geometry)
    target_keys = _cell_keys(predecessor.target_mesh, predecessor.target_geometry)
    rows = np.asarray(predecessor.source_rows, dtype=np.int64)
    for target_row, source_row in enumerate(rows.tolist()):
        if source_row >= 0 and source_keys[source_row] != target_keys[target_row]:
            raise ValueError(
                "Predecessor source row does not authenticate the exact target cell."
            )

    mesh = predecessor.target_mesh
    geometry = predecessor.target_geometry
    policy = source_validity.policy
    resolved_elements, resolved_routes, _ = geometry.resolve(mesh)
    elements = tuple(resolved_elements)
    routes = tuple(np.asarray(route, dtype=np.int64) for route in resolved_routes)
    names = tuple(block.name for block in mesh.blocks)
    kinds = tuple(block.cell_kind for block in mesh.blocks)
    context = _target_proof_context(geometry, names, elements, routes)
    values = np.asarray(geometry.coordinates, dtype=np.float64)
    source_bank = (
        context.execution_coordinates
        if context.source_coordinates is None
        else context.source_coordinates
    )
    exact_source_differs = (
        context.source_coordinates is not None
        and context.source_coordinates != context.execution_coordinates
    )
    count = rows.size
    execution_status = np.empty((count,), dtype=np.int32)
    determinant_lower = np.empty((count,), dtype=np.float64)
    determinant_upper = np.empty((count,), dtype=np.float64)
    depth = np.empty((count,), dtype=np.int64)
    source_status = np.empty((count,), dtype=np.int32)
    source_lower = np.empty((count,), dtype=np.float64)
    source_upper = np.empty((count,), dtype=np.float64)
    source_depth = np.empty((count,), dtype=np.int64)
    apex_unique = np.ones((count,), dtype=np.bool_)
    prior_apex_unique = np.asarray(source_validity.apex_derivative_unique)
    prior_execution_status = np.asarray(source_validity.execution_status)
    prior_lower = np.asarray(source_validity.determinant_lower)
    prior_upper = np.asarray(source_validity.determinant_upper)
    prior_depth = np.asarray(source_validity.depth)
    prior_source_status = np.asarray(source_validity.source_status)
    prior_source_lower = np.asarray(source_validity.source_determinant_lower)
    prior_source_upper = np.asarray(source_validity.source_determinant_upper)
    prior_source_depth = np.asarray(source_validity.source_depth)
    for target_row, source_row in enumerate(rows.tolist()):
        if source_row < 0:
            continue
        execution_status[target_row] = prior_execution_status[source_row]
        determinant_lower[target_row] = prior_lower[source_row]
        determinant_upper[target_row] = prior_upper[source_row]
        depth[target_row] = prior_depth[source_row]
        apex_unique[target_row] = prior_apex_unique[source_row]
        if context.distinct_source:
            source_status[target_row] = prior_source_status[source_row]
            source_lower[target_row] = prior_source_lower[source_row]
            source_upper[target_row] = prior_source_upper[source_row]
            source_depth[target_row] = prior_source_depth[source_row]

    unsupported = set(context.unsupported_block_names)
    cursor = 0
    offsets = [0]
    polyhedral_tables = None
    polyhedral_points = None
    source_offsets: dict[str, tuple[int, int]] = {}
    source_cursor = 0
    for source_block in predecessor.source_mesh.blocks:
        source_stop = source_cursor + source_block.cell_count
        source_offsets[source_block.name] = (source_cursor, source_stop)
        source_cursor = source_stop
    for name, kind, element, route in zip(names, kinds, elements, routes, strict=True):
        cell_count = route.shape[0]
        stop = cursor + cell_count
        changed_rows = np.flatnonzero(rows[cursor:stop] < 0)
        original_range = source_offsets.get(name)
        if (
            name in source_validity.unsupported_block_names
            and original_range is not None
            and stop - cursor == original_range[1] - original_range[0]
            and np.array_equal(
                np.sort(rows[cursor:stop]),
                np.arange(original_range[0], original_range[1], dtype=np.int64),
            )
        ):
            unsupported.add(name)
        if not isinstance(element, ScalarCoordinateElement | CellVertexGeometryElement):
            unsupported.add(name)
        if (
            isinstance(element, ScalarCoordinateElement)
            and _source_proof_domain(kind) is None
        ):
            for local_row in range(cell_count):
                if rows[cursor + local_row] < 0:
                    continue
                if _map_is_unsupported(
                    element,
                    kind,
                    context.execution_coordinates,
                    route,
                    local_row,
                    execution_map=True,
                ) or (
                    context.distinct_source
                    and _map_is_unsupported(
                        element,
                        kind,
                        source_bank,
                        route,
                        local_row,
                        execution_map=False,
                    )
                ):
                    unsupported.add(name)

        if changed_rows.size:
            changed_offset = cursor + changed_rows
            if isinstance(element, ScalarCoordinateElement):
                selected_element = _selected_element(element, changed_rows)
                selected_routes = route[changed_rows]
                try:
                    execution_proof, execution_unsupported = _certify_source_block(
                        selected_element,
                        kind,
                        context.execution_coordinates,
                        selected_routes,
                        policy,
                        execution_map=True,
                    )
                except (OverflowError, TypeError, ValueError):
                    execution_proof = _unresolved_proof(changed_rows.size)
                    execution_unsupported = True
                if execution_unsupported:
                    unsupported.add(name)
                execution_status[changed_offset] = execution_proof.status
                determinant_lower[changed_offset] = execution_proof.lower
                determinant_upper[changed_offset] = execution_proof.upper
                depth[changed_offset] = execution_proof.depth
                apex_unique[changed_offset] = execution_proof.apex_flags()
                if context.distinct_source:
                    try:
                        source_proof, source_unsupported = _certify_source_block(
                            selected_element,
                            kind,
                            source_bank,
                            selected_routes,
                            policy,
                        )
                    except (OverflowError, TypeError, ValueError):
                        source_proof = _unresolved_proof(changed_rows.size)
                        source_unsupported = True
                    if source_unsupported:
                        unsupported.add(name)
                    source_status[changed_offset] = source_proof.status
                    source_lower[changed_offset] = source_proof.lower
                    source_upper[changed_offset] = source_proof.upper
                    source_depth[changed_offset] = source_proof.depth
                    apex_unique[changed_offset] &= source_proof.apex_flags()
            elif isinstance(element, CellVertexGeometryElement):
                if kind == "polygon":
                    execution_proof = _certify_polygon_block(
                        values[route[changed_rows]], policy
                    )
                else:
                    if polyhedral_tables is None:
                        connectivity = mesh.connectivity
                        if not isinstance(connectivity, PolyhedralConnectivity):
                            raise TypeError(
                                "Polyhedral geometry requires polyhedral mesh connectivity."
                            )
                        polyhedral_tables = polyhedral_star_tables(connectivity)
                        polyhedral_points = _polyhedral_coordinates(mesh, routes, values)
                    if polyhedral_points is None:
                        raise RuntimeError(
                            "Polyhedral coordinate preparation did not produce coordinates."
                        )
                    execution_proof = _certify_polyhedral_cells(
                        polyhedral_tables,
                        polyhedral_points,
                        changed_offset,
                        policy,
                    )
                execution_status[changed_offset] = execution_proof.status
                determinant_lower[changed_offset] = execution_proof.lower
                determinant_upper[changed_offset] = execution_proof.upper
                depth[changed_offset] = execution_proof.depth
                if context.distinct_source:
                    if exact_source_differs:
                        from ._cell_geometry_source_validity import (
                            certify_exact_variable_topology_block,
                        )

                        try:
                            source_proof = certify_exact_variable_topology_block(
                                mesh,
                                dict(zip(names, routes, strict=True)),
                                source_bank,
                                policy,
                                block_name=name,
                                cell_rows=changed_rows,
                            )
                        except (OverflowError, TypeError, ValueError):
                            source_proof = _unresolved_proof(changed_rows.size)
                            unsupported.add(name)
                    else:
                        source_proof = execution_proof
                    source_status[changed_offset] = source_proof.status
                    source_lower[changed_offset] = source_proof.lower
                    source_upper[changed_offset] = source_proof.upper
                    source_depth[changed_offset] = source_proof.depth
            else:
                unsupported.add(name)
                execution_proof = _unresolved_proof(changed_rows.size)
                execution_status[changed_offset] = execution_proof.status
                determinant_lower[changed_offset] = execution_proof.lower
                determinant_upper[changed_offset] = execution_proof.upper
                depth[changed_offset] = execution_proof.depth
                if context.distinct_source:
                    source_status[changed_offset] = execution_proof.status
                    source_lower[changed_offset] = execution_proof.lower
                    source_upper[changed_offset] = execution_proof.upper
                    source_depth[changed_offset] = execution_proof.depth
        cursor = stop
        offsets.append(cursor)

    if not context.distinct_source:
        source_status = execution_status.copy()
        source_lower = determinant_lower.copy()
        source_upper = determinant_upper.copy()
        source_depth = depth.copy()
    return CellValidityCertificate(
        execution_status,
        determinant_lower,
        determinant_upper,
        depth,
        source_proof=_BlockCertificate(
            source_status, source_lower, source_upper, source_depth
        ),
        maximum_coefficient_rounding_error=(context.maximum_coefficient_rounding_error),
        maximum_element_coefficient_rounding_error=(
            context.maximum_element_coefficient_rounding_error
        ),
        apex_derivative_unique=apex_unique,
        geometry=geometry,
        mesh=mesh,
        policy=policy,
        block_names=names,
        block_offsets=tuple(offsets),
        unsupported_block_names=tuple(name for name in names if name in unsupported),
        geometry_id=cell_geometry_id(geometry),
        geometry_layout_id=geometry.geometry_layout_id,
        policy_id=policy.policy_id,
    )


__all__ = [
    "CellGeometryPredecessor",
    "prepare_cell_geometry_predecessor",
    "reuse_cell_geometry_validity",
]
