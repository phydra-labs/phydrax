#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

import phydrax.ein as ein

from ..._coordinate_measure import CoordinateFrame, CoordinateMeasurePlan
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._owned_programs import run_owned_program
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...exterior._algebra import map_reference_values, pullback, vector_to_form
from ...exterior._form_type import FormType
from ...linalg import (
    ArraySpace,
    BlockSpace,
    OperatorProperties,
)
from ...sparse import (
    EdgeRelation,
    gather_routes,
    route_reduce,
    RowRelation,
    SparseLinearMap,
)
from ...typing import checked
from .._adaptive_simplex import MaskedSimplexMesh
from .._cell_complex import (
    IntervalConnectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from .._cell_geometry import (
    CellGeometryElement,
    CellGeometrySpec,
    coordinate_tabulation,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from .._cell_geometry_validity import cell_jacobian_inverse_and_measure
from .._cell_mesh import CellBlock, CellMesh, PolyhedralBlock, SimplicialConnectivity
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._hexahedral import (
    _EDGES as _HEXAHEDRAL_EDGES,
    _FACES as _HEXAHEDRAL_FACES,
    _quadrilateral_tensor_permutation,
    HexahedralConnectivity,
)
from .._integration_domain import IntegrationDomain
from .._lifecycle import (
    AbstractDiscretizationPlan,
    validate_prepared_metadata,
)
from .._local_variational import (
    AbstractPreparedLocalDiscretization,
    LocalFieldBinding,
    LocalVariationalCapabilities,
    PreparedLocalRegion,
)
from .._measure import DiscreteMeasure
from .._reference_cell import reference_cell_topology
from .._side_actions import (
    FacetTraceRule,
    PreparedTraceAction,
    SideTraceQuantity,
)
from .._spaces import BlockDofLayout, DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import EntitySelection
from .._views import FieldTraceSide
from ._precision import FiniteElementPrecisionPolicy
from ._reference import FiniteElementSpec, lagrange_element


def _linear_reference_element(cell_kind: str, /) -> FiniteElementSpec:
    """Select the owning scalar coordinate chart without a generic Lagrange alias."""
    if cell_kind.startswith(("simplex:", "tensor:")):
        from ._form_elements import form_element

        family = "tensor-trimmed" if cell_kind.startswith("tensor:") else "trimmed"
        return form_element(
            cell_kind, 0, 1, family=family, twist="untwisted", proxy="scalar"
        )
    return lagrange_element(cell_kind, 1)


def _field_element_assignments(
    elements: FiniteElementSpec | Mapping[str, FiniteElementSpec],
    block_names: Sequence[str] | None,
    /,
) -> tuple[tuple[str, ...], tuple[FiniteElementSpec, ...]]:
    if isinstance(elements, FiniteElementSpec):
        names = () if block_names is None else tuple(str(block) for block in block_names)
        element_values = (elements,) if not names else (elements,) * len(names)
    else:
        items = tuple(
            sorted((str(block), element) for block, element in elements.items())
        )
        if not items:
            raise ValueError("Field element mapping must be non-empty.")
        names = tuple(block for block, _ in items)
        element_values = tuple(element for _, element in items)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Field block names must be unique and non-empty.")
    if not all(isinstance(element, FiniteElementSpec) for element in element_values):
        raise TypeError("Field elements must be FiniteElementSpec instances.")
    return names, element_values


def _resolve_field_assignments(
    mesh: CellMesh, names: tuple[str, ...], elements: tuple[FiniteElementSpec, ...], /
) -> tuple[FiniteElementSpec, ...]:
    if not names:
        if len(elements) != 1:
            raise ValueError("Implicit field element assignment requires one element.")
        return (elements[0],) * len(mesh.blocks)
    assignments = dict(zip(names, elements, strict=True))
    if set(assignments) != {block.name for block in mesh.blocks}:
        raise ValueError(
            "Field element assignments must match the mesh block names exactly."
        )
    return tuple(assignments[block.name] for block in mesh.blocks)


def _validate_resolved_elements(
    mesh: CellMesh, elements: tuple[FiniteElementSpec, ...], /
) -> None:
    for block, element in zip(mesh.blocks, elements, strict=True):
        if block.cell_kind != element.cell_kind:
            raise ValueError(
                f"Element {element.cell_kind!r} is incompatible with block {block.name!r} ({block.cell_kind!r})."
            )
    if len({element.conformity for element in elements}) != 1:
        raise ValueError("One field must use one conformity across mesh blocks.")
    if len({element.representation for element in elements}) != 1:
        raise ValueError(
            "One field must use one coefficient representation across mesh blocks."
        )
    if len({element.mapping for element in elements}) != 1:
        raise ValueError("One field must use one mapping across mesh blocks.")
    if len({element.value_shape for element in elements}) != 1:
        raise ValueError("One field must use one value shape across mesh blocks.")


@final
class FiniteElementFieldSpec(StrictModule, NonTrainableState):
    """One named field and its reference element on every mesh block."""

    name: str = eqx.field(static=True)
    block_names: tuple[str, ...] = eqx.field(static=True)
    elements: tuple[FiniteElementSpec, ...]
    component_shape: tuple[int, ...] = eqx.field(static=True)
    field_spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        elements: FiniteElementSpec | Mapping[str, FiniteElementSpec],
        /,
        *,
        block_names: Sequence[str] | None = None,
        component_shape: Sequence[int] = (),
    ) -> None:
        field_name = str(name)
        if not field_name:
            raise ValueError("Finite-element field name must be non-empty.")
        components = tuple(component_shape)
        if any(size <= 0 for size in components):
            raise ValueError("Field component dimensions must be positive.")
        names, element_values = _field_element_assignments(elements, block_names)
        self.name = field_name
        self.block_names = names
        self.elements = element_values
        self.component_shape = components
        self.field_spec_id = canonical_fingerprint(
            {
                "kind": "finite-element-field-spec",
                "name": field_name,
                "blocks": list(names),
                "elements": [element.element_id for element in element_values],
                "component_shape": list(components),
            }
        )

    def resolve(self, mesh: CellMesh, /) -> tuple[FiniteElementSpec, ...]:
        resolved = _resolve_field_assignments(mesh, self.block_names, self.elements)
        _validate_resolved_elements(mesh, resolved)
        return resolved


_HEXAHEDRAL_EDGE_BY_VERTICES = {
    frozenset(edge): index for index, edge in enumerate(_HEXAHEDRAL_EDGES)
}

_TETRAHEDRAL_TOPOLOGY = reference_cell_topology("tetrahedron")
_TETRAHEDRAL_EDGES = _TETRAHEDRAL_TOPOLOGY.entities[1]
_TETRAHEDRAL_FACES = _TETRAHEDRAL_TOPOLOGY.entities[2]
_TRIANGULAR_EDGES = reference_cell_topology("triangle").entities[1]


def _simplex_entity_routes(
    edges: np.ndarray,
    faces: np.ndarray | None,
    cells: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reference-ordered edge routes, signs and face routes of triangles or tetrahedra.

    ``edges`` and ``faces`` hold canonical (ascending) vertex rows; triangles have
    no face routes.
    """
    edge_by_vertices = {tuple(edge): index for index, edge in enumerate(edges)}
    reference_edges = _TETRAHEDRAL_EDGES if faces is not None else _TRIANGULAR_EDGES
    edge_routes = np.empty((cells.shape[0], len(reference_edges)), dtype=np.int32)
    edge_signs = np.empty_like(edge_routes, dtype=np.float64)
    face_routes = (
        None
        if faces is None
        else np.empty((cells.shape[0], len(_TETRAHEDRAL_FACES)), dtype=np.int32)
    )
    face_by_vertices = (
        {} if faces is None else {tuple(face): index for index, face in enumerate(faces)}
    )
    for cell, vertices in enumerate(cells):
        for local_edge, (start, stop) in enumerate(reference_edges):
            oriented = (int(vertices[start]), int(vertices[stop]))
            canonical = tuple(sorted(oriented))
            edge_routes[cell, local_edge] = edge_by_vertices[canonical]
            edge_signs[cell, local_edge] = 1.0 if oriented == canonical else -1.0
        if face_routes is not None:
            for local_face, local_vertices in enumerate(_TETRAHEDRAL_FACES):
                canonical = tuple(
                    sorted(int(vertices[index]) for index in local_vertices)
                )
                face_routes[cell, local_face] = face_by_vertices[canonical]
    return edge_routes, edge_signs, face_routes


def _tetrahedral_face_dof_positions(
    element: FiniteElementSpec,
    local_face: int,
    local_vertices: tuple[int, ...],
    canonical_vertices: tuple[int, ...],
    /,
) -> np.ndarray:
    face_dofs = element.entity_dofs[2][local_face]
    if not face_dofs:
        return np.empty((0,), dtype=np.int32)
    reference_vertices = _TETRAHEDRAL_FACES[local_face]
    vertex_dofs = element.entity_dofs[0]
    if any(len(vertex_dofs[vertex]) != 1 for vertex in reference_vertices):
        raise ValueError("H1 nodal vertices require one DOF per vertex.")
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    corners = nodes[
        np.asarray(
            [vertex_dofs[vertex][0] for vertex in reference_vertices],
            dtype=np.int32,
        )
    ]
    augmented_corners = np.concatenate(
        (corners.T, np.ones((1, len(reference_vertices)))),
        axis=0,
    )
    face_nodes = nodes[np.asarray(face_dofs, dtype=np.int32)]
    augmented_nodes = np.concatenate(
        (face_nodes.T, np.ones((1, len(face_dofs)))),
        axis=0,
    )
    barycentric = np.linalg.lstsq(augmented_corners, augmented_nodes, rcond=None)[0].T
    if not np.allclose(
        barycentric @ corners,
        face_nodes,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("Tetrahedral face DOFs are not on their declared face.")
    canonical_barycentric = np.empty_like(barycentric)
    for local_position, vertex in enumerate(local_vertices):
        canonical_position = canonical_vertices.index(vertex)
        canonical_barycentric[:, canonical_position] = barycentric[:, local_position]
    keys = tuple(
        tuple(float(value) for value in np.round(row, decimals=12))
        for row in canonical_barycentric
    )
    order = sorted(range(len(keys)), key=keys.__getitem__)
    positions = np.empty((len(keys),), dtype=np.int32)
    positions[np.asarray(order, dtype=np.int32)] = np.arange(len(keys), dtype=np.int32)
    return positions


def _has_nonvertex_dofs(element: FiniteElementSpec, /) -> bool:
    return any(
        entity
        for dimension_entities in element.entity_dofs[1:]
        for entity in dimension_entities
    )


def _hexahedral_face_shape(
    element: FiniteElementSpec,
    local_face: int,
    /,
) -> tuple[int, int]:
    vertices = _HEXAHEDRAL_FACES[local_face]
    side_edges = tuple(
        _HEXAHEDRAL_EDGE_BY_VERTICES[
            frozenset((vertices[position], vertices[(position + 1) % 4]))
        ]
        for position in range(4)
    )
    side_widths = tuple(
        len(element.entity_dofs[1][local_edge]) for local_edge in side_edges
    )
    if side_widths[0] != side_widths[2] or side_widths[1] != side_widths[3]:
        raise ValueError(
            "Hexahedral tensor faces require equal widths on opposite edges."
        )
    shape = (side_widths[0], side_widths[3])
    if len(element.entity_dofs[2][local_face]) != prod(shape):
        raise ValueError(
            "Hexahedral face-interior DOFs do not match its tensor trace shape."
        )
    return shape


def _canonical_face_shape(
    vertex_permutation: ArrayLike,
    local_shape: tuple[int, int],
    /,
) -> tuple[int, int]:
    permutation = np.asarray(vertex_permutation, dtype=np.int32)
    corners = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.int32)
    direction_u = corners[permutation[1]] - corners[permutation[0]]
    if direction_u[0] != 0:
        return local_shape
    return local_shape[1], local_shape[0]


def _hexahedral_face_grid_positions(
    element: FiniteElementSpec,
    local_face: int,
    shape: tuple[int, int],
    /,
) -> np.ndarray:
    face_dofs = element.entity_dofs[2][local_face]
    if not face_dofs:
        return np.empty((0,), dtype=np.int32)
    face_vertices = _HEXAHEDRAL_FACES[local_face]
    vertex_dofs = element.entity_dofs[0]
    if any(len(vertex_dofs[vertex]) != 1 for vertex in face_vertices):
        raise ValueError("H1 nodal vertices require one DOF per vertex.")
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    corners = nodes[
        np.asarray(
            [vertex_dofs[vertex][0] for vertex in face_vertices],
            dtype=np.int32,
        )
    ]
    origin = corners[0]
    axes = np.stack((corners[1] - origin, corners[3] - origin), axis=1)
    parameters = np.linalg.lstsq(
        axes,
        nodes[np.asarray(face_dofs, dtype=np.int32)].T - origin[:, None],
        rcond=None,
    )[0].T
    reconstructed = origin + parameters @ axes.T
    if not np.allclose(
        reconstructed,
        nodes[np.asarray(face_dofs, dtype=np.int32)],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("Hexahedral face DOFs are not on their declared face.")
    parameters = np.round(parameters, decimals=12)
    levels_u = np.unique(parameters[:, 0])
    levels_v = np.unique(parameters[:, 1])
    if (len(levels_u), len(levels_v)) != shape:
        raise ValueError(
            "Hexahedral face DOF coordinates do not match its tensor trace shape."
        )
    positions = np.empty((len(face_dofs),), dtype=np.int32)
    for position, (parameter_u, parameter_v) in enumerate(parameters):
        index_u = int(np.argmin(np.abs(levels_u - parameter_u)))
        index_v = int(np.argmin(np.abs(levels_v - parameter_v)))
        if not np.isclose(levels_u[index_u], parameter_u) or not np.isclose(
            levels_v[index_v], parameter_v
        ):
            raise ValueError("Hexahedral face tensor coordinates are inconsistent.")
        positions[position] = index_u * shape[1] + index_v
    if np.unique(positions).size != len(face_dofs):
        raise ValueError("Hexahedral face tensor positions must be unique.")
    return positions


def _uniform_entity_width(widths: np.ndarray, /) -> int:
    unique = np.unique(widths)
    return int(unique[0]) if unique.size == 1 and unique[0] > 0 else 1


@dataclass(frozen=True, slots=True)
class _FiniteElementDofLayout:
    conformity: str
    association: str
    global_count: int
    entity_dof_counts: tuple[int, ...]
    entity_dofs_per_entity: tuple[int, ...]
    edge_widths: np.ndarray | None = None
    edge_starts: np.ndarray | None = None
    face_widths: np.ndarray | None = None
    face_starts: np.ndarray | None = None
    cell_starts: np.ndarray | None = None
    canonical_routes: tuple[np.ndarray, ...] | None = None
    canonical_transforms: tuple[np.ndarray, ...] | None = None
    canonical_boundary: np.ndarray | None = None


def _topology_vertex_sets(mesh: CellMesh, /) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Recover canonical entity vertex sets from the owning incidence."""
    result = [tuple((vertex,) for vertex in range(mesh.coordinates.shape[0]))]
    for degree, incidence in enumerate(mesh.topology.incidences, start=1):
        relation = incidence.relation
        valid = np.asarray(relation.valid, dtype=np.bool_)
        source = np.asarray(relation.source_indices)[valid]
        target = np.asarray(relation.target_indices)[valid]
        vertices = [set() for _ in range(mesh.topology.entity_sets[degree].count)]
        for lower, upper in zip(source, target, strict=True):
            vertices[upper].update(result[-1][lower])
        result.append(tuple(tuple(sorted(entity)) for entity in vertices))
    return tuple(result)


def _record_form_entity_widths(
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    lookups: tuple[dict[tuple[int, ...], int], ...],
    widths: list[np.ndarray],
    /,
) -> None:
    basis = element.form_basis
    if basis is None:
        raise ValueError("Canonical form routing requires form-basis metadata.")
    for cell in np.asarray(block.vertices):
        for dimension, faces in enumerate(basis.entity_vertices):
            for face, dofs in zip(faces, element.entity_dofs[dimension], strict=True):
                entity = lookups[dimension][tuple(sorted(cell[list(face)]))]
                old = widths[dimension][entity]
                if old not in (0, len(dofs)):
                    raise ValueError("Shared form entities require equal moment spaces.")
                widths[dimension][entity] = len(dofs)


def _form_cell_orientation_sign(
    mesh: CellMesh,
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    cell: np.ndarray,
    global_vertices: tuple[int, ...],
    /,
) -> int:
    """Return the cell-chart and physical coorientation character."""
    basis = element.form_basis
    if basis is None:
        raise ValueError("Canonical form routing requires form-basis metadata.")
    sign = 1
    if basis.form_degree == mesh.topological_dimension:
        permutation = basis.canonical_permutation(global_vertices)
        if basis.family == "tensor-trimmed":
            chart = _linear_reference_element(block.cell_kind)
            reference_vertices = np.asarray(chart.reference_nodes)
            affine = np.linalg.lstsq(
                np.c_[
                    reference_vertices,
                    np.ones((len(cell),), dtype=np.float64),
                ],
                reference_vertices[list(permutation)],
                rcond=None,
            )[0]
            sign = int(np.sign(np.linalg.det(affine[:-1].T)))
        else:
            sign = (-1) ** sum(
                permutation[i] > permutation[j]
                for i in range(len(permutation))
                for j in range(i + 1, len(permutation))
            )
    if element.value_spec.form_type.twist == "twisted":
        chart = _linear_reference_element(block.cell_kind)
        gradients = np.asarray(
            chart.tabulate(np.mean(np.asarray(chart.reference_nodes), axis=0)[None])[1][0]
        )
        jacobian = np.asarray(mesh.coordinates)[cell].T @ gradients
        if jacobian.shape[0] != jacobian.shape[1]:
            raise ValueError(
                "Twisted embedded FE moments require explicit ambient coorientation."
            )
        sign *= int(np.sign(np.linalg.det(jacobian)))
    return sign


def _orient_form_cell_matrix(
    matrix: np.ndarray,
    mesh: CellMesh,
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    cell: np.ndarray,
    /,
) -> None:
    basis = element.form_basis
    if basis is None:
        raise ValueError("Canonical form routing requires form-basis metadata.")
    global_vertices = tuple(
        np.asarray(mesh.vertex_global_ids, dtype=np.int64)[cell].tolist()
    )
    matrix[:] = np.asarray(
        basis.exact_entity_permutation_matrix(global_vertices), dtype=np.float64
    )
    matrix *= _form_cell_orientation_sign(mesh, block, element, cell, global_vertices)


def _canonical_form_dof_layout(
    mesh: CellMesh, elements: tuple[FiniteElementSpec, ...], /
) -> _FiniteElementDofLayout:
    """Prepare permutation-covariant entity moment coordinates."""
    entities = _topology_vertex_sets(mesh)
    lookups = tuple(
        {vertices: index for index, vertices in enumerate(level)} for level in entities
    )
    widths = [np.zeros((len(level),), dtype=np.int32) for level in entities]
    for block, element in zip(mesh.blocks, elements, strict=True):
        _record_form_entity_widths(block, element, lookups, widths)
    counts = tuple(int(np.sum(level)) for level in widths)
    offsets = np.cumsum((0, *counts), dtype=np.int64)
    starts = tuple(
        offsets[degree] + np.cumsum(np.r_[0, level[:-1]], dtype=np.int64)
        for degree, level in enumerate(widths)
    )
    boundary = np.zeros((int(offsets[-1]),), dtype=np.bool_)
    for degree, level in enumerate(widths):
        mask = np.asarray(mesh.topology.entity_sets[degree].subset("boundary").mask)
        for entity in np.flatnonzero(mask):
            start = starts[degree][entity]
            boundary[start : start + level[entity]] = True
    routes, transforms = _canonical_form_block_routes(mesh, elements, lookups, starts)
    return _FiniteElementDofLayout(
        conformity=elements[0].conformity,
        association="form_entity",
        global_count=int(offsets[-1]),
        entity_dof_counts=counts,
        entity_dofs_per_entity=tuple(_uniform_entity_width(level) for level in widths),
        canonical_routes=routes,
        canonical_transforms=transforms,
        canonical_boundary=boundary,
    )


def _canonical_form_block_routes(
    mesh: CellMesh,
    elements: tuple[FiniteElementSpec, ...],
    lookups: tuple[dict[tuple[int, ...], int], ...],
    starts: tuple[np.ndarray, ...],
    /,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    routes, transforms = [], []
    for block, element in zip(mesh.blocks, elements, strict=True):
        basis = element.form_basis
        if basis is None:
            raise ValueError("Canonical form routing requires form-basis metadata.")
        local = np.empty((block.cell_count, element.local_dof_count), dtype=np.int32)
        matrices = np.empty(
            (block.cell_count, element.local_dof_count, element.local_dof_count),
            dtype=np.float64,
        )
        for row, cell in enumerate(np.asarray(block.vertices)):
            _orient_form_cell_matrix(matrices[row], mesh, block, element, cell)
            for dimension, faces in enumerate(basis.entity_vertices):
                for face, dofs in zip(faces, element.entity_dofs[dimension], strict=True):
                    # Vertex-set matching, never positional reference-face matching.
                    entity = lookups[dimension][tuple(sorted(cell[list(face)]))]
                    local[row, list(dofs)] = starts[dimension][entity] + np.arange(
                        len(dofs)
                    )
        routes.append(local)
        transforms.append(matrices)
    return tuple(routes), tuple(transforms)


_HIGH_ORDER_H1_CONNECTIVITIES = (
    PolygonalConnectivity,
    TetrahedralConnectivity,
    HexahedralConnectivity,
    SimplicialConnectivity,
)
type _HighOrderH1Connectivity = (
    PolygonalConnectivity
    | TetrahedralConnectivity
    | HexahedralConnectivity
    | SimplicialConnectivity
)


def _h1_simplex_entities(
    connectivity: TetrahedralConnectivity | SimplicialConnectivity, /
) -> tuple[np.ndarray, np.ndarray | None]:
    """Canonical edge and (tetrahedral) face rows of a simplex connectivity."""
    if isinstance(connectivity, TetrahedralConnectivity):
        return np.asarray(connectivity.edges), np.asarray(connectivity.faces)
    if connectivity.dimension == 2:
        return np.asarray(connectivity.entities[1]), None
    if connectivity.dimension == 3:
        return np.asarray(connectivity.entities[1]), np.asarray(connectivity.entities[2])
    raise ValueError(
        "High-order H1 routing on simplicial complexes requires triangles or tetrahedra."
    )


def _h1_edge_count(connectivity: _HighOrderH1Connectivity, /) -> int:
    if isinstance(connectivity, (PolygonalConnectivity, HexahedralConnectivity)):
        return int(connectivity.edges.shape[0])
    return int(_h1_simplex_entities(connectivity)[0].shape[0])


def _h1_cell_entity_routes(
    connectivity: _HighOrderH1Connectivity,
    vertices: np.ndarray,
    rows: slice,
    local_edge_count: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Reference-ordered edge routes and signs, then face routes and canonical faces.

    Face entries are present only for tetrahedral simplex routing; polygonal and
    hexahedral connectivities publish their own cell-edge routes.
    """
    if isinstance(connectivity, (PolygonalConnectivity, HexahedralConnectivity)):
        edges = np.asarray(connectivity.cell_edges, dtype=np.int32)[
            rows, :local_edge_count
        ]
        signs = np.asarray(connectivity.cell_edge_signs, dtype=np.float64)[
            rows, :local_edge_count
        ]
        return edges, signs, None, None
    canonical_edges, canonical_faces = _h1_simplex_entities(connectivity)
    edges, signs, faces = _simplex_entity_routes(
        canonical_edges, canonical_faces, vertices
    )
    return edges, signs, faces, canonical_faces


def _prepare_finite_element_dof_layout(
    mesh: CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    components: tuple[int, ...],
    /,
) -> _FiniteElementDofLayout:
    if all(element.form_basis is not None for element in resolved):
        if components:
            raise ValueError(
                "Canonical form elements do not replicate coefficient fibers."
            )
        return _canonical_form_dof_layout(mesh, resolved)
    conformities = {element.conformity for element in resolved}
    if len(conformities) != 1:
        raise ValueError("One field must use one conformity across cell blocks.")
    conformity = conformities.pop()
    match conformity:
        case "L2":
            return _single_association_dof_layout(
                mesh,
                conformity,
                "cell",
                sum(
                    block.cell_count * element.local_dof_count
                    for block, element in zip(mesh.blocks, resolved, strict=True)
                ),
            )
        case "Hdiv" | "Hcurl" | "HLambda":
            raise ValueError("Compatible fields require canonical form-element metadata.")
        case "H1":
            if not any(_has_nonvertex_dofs(element) for element in resolved):
                return _single_association_dof_layout(
                    mesh, conformity, "vertex", mesh.coordinates.shape[0]
                )
            return _high_order_h1_dof_layout(mesh, resolved)
        case _:
            raise ValueError(f"Unsupported finite-element conformity {conformity!r}.")


def _single_association_dof_layout(
    mesh: CellMesh,
    conformity: str,
    association: str,
    global_count: int,
    /,
) -> _FiniteElementDofLayout:
    # One association owns every DOF, so no per-entity offsets are published.
    entity_count = mesh.topological_dimension + 1
    return _FiniteElementDofLayout(
        conformity=conformity,
        association=association,
        global_count=global_count,
        entity_dof_counts=(0,) * entity_count,
        entity_dofs_per_entity=(1,) * entity_count,
    )


def _high_order_h1_dof_layout(
    mesh: CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    /,
) -> _FiniteElementDofLayout:
    edge_widths, face_widths, cell_widths = _high_order_h1_entity_widths(mesh, resolved)
    # Global numbering: vertices, then edge, face, and cell interiors.
    vertex_count = mesh.coordinates.shape[0]
    cursor = vertex_count
    edge_starts = np.empty_like(edge_widths)
    for edge, width in enumerate(edge_widths):
        edge_starts[edge] = cursor
        cursor += int(width)
    edge_dof_count = cursor - vertex_count

    face_starts = None
    face_dof_count = 0
    if face_widths is not None:
        face_starts = np.empty((len(face_widths),), dtype=np.int32)
        for face, width in enumerate(face_widths):
            face_starts[face] = cursor
            cursor += int(width)
        face_dof_count = cursor - vertex_count - edge_dof_count

    # Cell interiors follow global cell identity, independent of block order.
    cell_starts = np.empty_like(cell_widths)
    cell_global_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    for cell in np.argsort(cell_global_ids, kind="stable"):
        cell_starts[cell] = cursor
        cursor += int(cell_widths[cell])
    cell_dof_count = cursor - (vertex_count + edge_dof_count + face_dof_count)

    counts = [vertex_count, edge_dof_count]
    per_entity = [1, _uniform_entity_width(edge_widths)]
    if mesh.topological_dimension == 3:
        if face_widths is None:
            raise TypeError("Three-dimensional H1 routing requires face widths.")
        counts.append(face_dof_count)
        per_entity.append(_uniform_entity_width(face_widths))
    counts.append(cell_dof_count)
    per_entity.append(_uniform_entity_width(cell_widths))
    return _FiniteElementDofLayout(
        conformity="H1",
        association="entity",
        global_count=cursor,
        entity_dof_counts=tuple(counts),
        entity_dofs_per_entity=tuple(per_entity),
        edge_widths=edge_widths,
        edge_starts=edge_starts,
        face_widths=face_widths,
        face_starts=face_starts,
        cell_starts=cell_starts,
    )


def _high_order_h1_entity_widths(
    mesh: CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    /,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return conforming edge, face, and cell-interior DOF widths per entity."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, _HIGH_ORDER_H1_CONNECTIVITIES):
        raise ValueError(
            "High-order H1 entity routing requires polygonal, tetrahedral, hexahedral, "
            "or simplicial connectivity."
        )
    edge_widths = np.full((_h1_edge_count(connectivity),), -1, dtype=np.int32)
    total_cell_count = sum(block.cell_count for block in mesh.blocks)
    cell_widths = np.empty((total_cell_count,), dtype=np.int32)
    face_widths = None
    face_shapes = None
    if isinstance(connectivity, HexahedralConnectivity):
        face_count = connectivity.faces.shape[0]
        face_widths = np.full((face_count,), -1, dtype=np.int32)
        face_shapes = np.full((face_count, 2), -1, dtype=np.int32)
    elif not isinstance(connectivity, PolygonalConnectivity):
        canonical_faces = _h1_simplex_entities(connectivity)[1]
        if canonical_faces is not None:
            face_widths = np.full((canonical_faces.shape[0],), -1, dtype=np.int32)

    cell_offset = 0
    for block, element in zip(mesh.blocks, resolved, strict=True):
        if len(element.entity_dofs[0]) != block.arity:
            raise ValueError("H1 nodal vertex entities must match the cell vertices.")
        _record_high_order_h1_trace_widths(
            connectivity,
            block,
            element,
            cell_offset,
            edge_widths,
            face_widths,
            face_shapes,
        )
        top_entities = element.entity_dofs[mesh.topological_dimension]
        if len(top_entities) != 1:
            raise ValueError("H1 cell interiors require one top-dimensional entity.")
        cell_widths[cell_offset : cell_offset + block.cell_count] = len(top_entities[0])
        cell_offset += block.cell_count

    if np.any(edge_widths < 0):
        raise ValueError("High-order H1 routing left unassigned edges.")
    if face_widths is not None and np.any(face_widths < 0):
        raise ValueError("High-order H1 routing left unassigned faces.")
    return edge_widths, face_widths, cell_widths


def _record_high_order_h1_trace_widths(
    connectivity: _HighOrderH1Connectivity,
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    cell_offset: int,
    edge_widths: np.ndarray,
    face_widths: np.ndarray | None,
    face_shapes: np.ndarray | None,
    /,
) -> None:
    """Record one block's edge and face trace widths on the shared entities."""
    local_edge_count = len(element.entity_dofs[1])
    if isinstance(connectivity, PolygonalConnectivity):
        expected_edge_count = block.arity
    elif isinstance(connectivity, HexahedralConnectivity):
        expected_edge_count = len(_HEXAHEDRAL_EDGES)
    else:
        expected_edge_count = len(
            _TRIANGULAR_EDGES
            if _h1_simplex_entities(connectivity)[1] is None
            else _TETRAHEDRAL_EDGES
        )
    if local_edge_count != expected_edge_count:
        raise ValueError("H1 edge entities must match the reference cell edges.")
    block_cell_edges, _, block_cell_faces, _ = _h1_cell_entity_routes(
        connectivity,
        np.asarray(block.vertices, dtype=np.int32),
        slice(cell_offset, cell_offset + block.cell_count),
        local_edge_count,
    )
    _record_shared_entity_widths(
        edge_widths,
        block_cell_edges,
        element.entity_dofs[1],
        "Shared H1 edge trace widths are incompatible; a mortar is required.",
    )

    if isinstance(connectivity, HexahedralConnectivity):
        _record_hexahedral_face_widths(
            connectivity, block, element, cell_offset, face_shapes, face_widths
        )
    elif block_cell_faces is not None:
        if face_widths is None:
            raise RuntimeError("Tetrahedral H1 routing requires allocated face widths.")
        if len(element.entity_dofs[2]) != len(_TETRAHEDRAL_FACES):
            raise ValueError("H1 face entities must match the tetrahedron faces.")
        _record_shared_entity_widths(
            face_widths,
            block_cell_faces,
            element.entity_dofs[2],
            "Shared H1 triangular trace widths are incompatible; a mortar is required.",
        )


def _record_shared_entity_widths(
    widths: np.ndarray,
    block_routes: np.ndarray,
    local_entity_dofs: tuple[tuple[int, ...], ...],
    conflict_message: str,
    /,
) -> None:
    """Assign each shared entity one DOF width; unequal traces need a mortar."""
    for local_entity, entity_dofs in enumerate(local_entity_dofs):
        width = len(entity_dofs)
        for entity in np.unique(block_routes[:, local_entity]):
            existing = widths[int(entity)]
            if existing >= 0 and existing != width:
                raise ValueError(conflict_message)
            widths[int(entity)] = width


def _record_hexahedral_face_widths(
    connectivity: HexahedralConnectivity,
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    cell_offset: int,
    face_shapes: np.ndarray | None,
    face_widths: np.ndarray | None,
    /,
) -> None:
    """Assign each shared quadrilateral face one canonical tensor trace shape."""
    if face_shapes is None:
        raise RuntimeError("Hexahedral H1 routing requires allocated face shapes.")
    if len(element.entity_dofs[2]) != len(_HEXAHEDRAL_FACES):
        raise ValueError("H1 face entities must match the hexahedron faces.")
    block_cell_faces = np.asarray(
        connectivity.cell_faces,
        dtype=np.int32,
    )[cell_offset : cell_offset + block.cell_count]
    block_face_permutations = np.asarray(
        connectivity.cell_face_vertex_permutations,
        dtype=np.int32,
    )[cell_offset : cell_offset + block.cell_count]
    for local_face in range(len(_HEXAHEDRAL_FACES)):
        local_shape = _hexahedral_face_shape(element, local_face)
        for cell in range(block.cell_count):
            face = int(block_cell_faces[cell, local_face])
            canonical_shape = _canonical_face_shape(
                block_face_permutations[cell, local_face],
                local_shape,
            )
            existing = tuple(face_shapes[face])
            if existing[0] >= 0 and existing != canonical_shape:
                raise ValueError(
                    "Shared H1 quadrilateral trace shapes are incompatible; a mortar is required."
                )
            face_shapes[face] = canonical_shape
            if face_widths is None:
                raise RuntimeError("Hexahedral H1 routing requires face widths.")
            face_widths[face] = prod(canonical_shape)


def _build_finite_element_dof_routes(
    mesh: CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    layout: _FiniteElementDofLayout,
    /,
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[RowRelation, ...]]:
    connectivity = mesh.connectivity
    block_dofs = []
    orientations = []
    relations = []
    cell_offset = 0
    dof_offset = 0
    for block, element in zip(mesh.blocks, resolved, strict=True):
        vertices = np.asarray(block.vertices, dtype=np.int32)
        match layout.association:
            case "cell":
                width = element.local_dof_count
                local = np.arange(
                    dof_offset,
                    dof_offset + block.cell_count * width,
                    dtype=np.int32,
                ).reshape((block.cell_count, width))
                dof_offset += block.cell_count * width
                orientation = np.ones_like(local, dtype=np.float64)
            case "edge":
                if not isinstance(connectivity, PolygonalConnectivity):
                    raise TypeError(
                        "Compatible edge map requires polygonal connectivity."
                    )
                rows = slice(cell_offset, cell_offset + block.cell_count)
                local = np.asarray(connectivity.cell_edges, dtype=np.int32)[
                    rows, : element.local_dof_count
                ]
                orientation = np.asarray(
                    connectivity.cell_edge_signs,
                    dtype=np.float64,
                )[rows, : element.local_dof_count]
            case "form_entity":
                if layout.canonical_routes is None:
                    raise RuntimeError("Canonical form DOF routes were not prepared.")
                local = layout.canonical_routes[len(block_dofs)]
                orientation = np.ones_like(local, dtype=np.float64)
            case "entity":
                local = _high_order_h1_block_routes(
                    mesh, layout, block, element, vertices, cell_offset
                )
                orientation = np.ones_like(local, dtype=np.float64)
            case "vertex":
                if element.local_dof_count != vertices.shape[1]:
                    raise ValueError(
                        "Vertex-associated H1 elements require one DOF per vertex."
                    )
                local = vertices
                orientation = np.ones_like(local, dtype=np.float64)
            case _:
                raise ValueError("Unsupported finite-element DOF map.")
        block_dofs.append(jnp.asarray(local))
        orientations.append(jnp.asarray(orientation))
        relations.append(RowRelation(local, source_size=layout.global_count))
        cell_offset += block.cell_count
    return tuple(block_dofs), tuple(orientations), tuple(relations)


def _high_order_h1_block_routes(
    mesh: CellMesh,
    layout: _FiniteElementDofLayout,
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    vertices: np.ndarray,
    cell_offset: int,
    /,
) -> np.ndarray:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, _HIGH_ORDER_H1_CONNECTIVITIES):
        raise RuntimeError("High-order H1 routing lost compatible connectivity.")
    edge_starts = layout.edge_starts
    cell_starts = layout.cell_starts
    if layout.edge_widths is None or edge_starts is None or cell_starts is None:
        raise TypeError("High-order H1 entity offsets are unavailable.")
    local = np.full(
        (block.cell_count, element.local_dof_count),
        -1,
        dtype=np.int32,
    )
    for local_vertex, entity_dofs in enumerate(element.entity_dofs[0]):
        if len(entity_dofs) != 1:
            raise ValueError("H1 nodal vertices require one DOF per vertex.")
        local[:, entity_dofs[0]] = vertices[:, local_vertex]

    local_edge_count = len(element.entity_dofs[1])
    (
        block_cell_edges,
        block_cell_signs,
        block_cell_faces,
        canonical_faces,
    ) = _h1_cell_entity_routes(
        connectivity,
        vertices,
        slice(cell_offset, cell_offset + block.cell_count),
        local_edge_count,
    )
    # Edge-interior DOFs are stored along the canonical edge direction.
    for local_edge, entity_dofs in enumerate(element.entity_dofs[1]):
        width = len(entity_dofs)
        if width == 0:
            continue
        positions = np.arange(width, dtype=np.int32)
        canonical_positions = np.where(
            block_cell_signs[:, local_edge, None] > 0.0,
            positions,
            positions[::-1],
        )
        local[:, np.asarray(entity_dofs, dtype=np.int32)] = (
            edge_starts[block_cell_edges[:, local_edge], None] + canonical_positions
        )
    if isinstance(connectivity, HexahedralConnectivity):
        _assign_hexahedral_face_dofs(
            local, connectivity, element, layout.face_starts, cell_offset
        )
    elif block_cell_faces is not None and canonical_faces is not None:
        _assign_tetrahedral_face_dofs(
            local,
            canonical_faces,
            element,
            layout.face_starts,
            vertices,
            block_cell_faces,
        )

    interior_dofs = element.entity_dofs[mesh.topological_dimension][0]
    if interior_dofs:
        for cell in range(block.cell_count):
            local[
                cell,
                np.asarray(interior_dofs, dtype=np.int32),
            ] = cell_starts[cell_offset + cell] + np.arange(
                len(interior_dofs), dtype=np.int32
            )
    if np.any(local < 0):
        raise ValueError("High-order H1 entity map left unassigned DOFs.")
    return local


def _assign_hexahedral_face_dofs(
    local: np.ndarray,
    connectivity: HexahedralConnectivity,
    element: FiniteElementSpec,
    face_starts: np.ndarray | None,
    cell_offset: int,
    /,
) -> None:
    """Route quadrilateral face-interior DOFs through each face's tensor permutation."""
    if face_starts is None:
        raise TypeError("Hexahedral H1 face offsets are unavailable.")
    cell_count = local.shape[0]
    block_cell_faces = np.asarray(
        connectivity.cell_faces,
        dtype=np.int32,
    )[cell_offset : cell_offset + cell_count]
    block_face_permutations = np.asarray(
        connectivity.cell_face_vertex_permutations,
        dtype=np.int32,
    )[cell_offset : cell_offset + cell_count]
    for local_face, face_dofs in enumerate(element.entity_dofs[2]):
        if not face_dofs:
            continue
        shape = _hexahedral_face_shape(element, local_face)
        grid_positions = _hexahedral_face_grid_positions(element, local_face, shape)
        for cell in range(cell_count):
            tensor_permutation = _quadrilateral_tensor_permutation(
                block_face_permutations[cell, local_face],
                *shape,
            )
            face = block_cell_faces[cell, local_face]
            local[
                cell,
                np.asarray(face_dofs, dtype=np.int32),
            ] = face_starts[face] + tensor_permutation[grid_positions]


def _assign_tetrahedral_face_dofs(
    local: np.ndarray,
    faces: np.ndarray,
    element: FiniteElementSpec,
    face_starts: np.ndarray | None,
    vertices: np.ndarray,
    block_cell_faces: np.ndarray,
    /,
) -> None:
    """Route triangular face-interior DOFs by canonical face barycentric order."""
    if face_starts is None:
        raise TypeError("Tetrahedral H1 face offsets are unavailable.")
    canonical_faces = np.asarray(faces, dtype=np.int32)
    for local_face, face_dofs in enumerate(element.entity_dofs[2]):
        if not face_dofs:
            continue
        reference_vertices = _TETRAHEDRAL_FACES[local_face]
        for cell in range(local.shape[0]):
            face = int(block_cell_faces[cell, local_face])
            local_vertices = tuple(
                int(vertices[cell, vertex]) for vertex in reference_vertices
            )
            canonical_vertices = tuple(canonical_faces[face])
            positions = _tetrahedral_face_dof_positions(
                element,
                local_face,
                local_vertices,
                canonical_vertices,
            )
            local[
                cell,
                np.asarray(face_dofs, dtype=np.int32),
            ] = face_starts[face] + positions


def _coordinate_node_weights(
    element: FiniteElementSpec,
    coordinate_element: CellGeometryElement,
    cell_count: int,
    /,
) -> np.ndarray:
    """Coordinate-map basis at the field's reference nodes.

    Shared charts give ``(node, coordinate dof)``; per-cell charts give
    ``(cell, node, coordinate dof)``.
    """
    if not isinstance(coordinate_element, ScalarCoordinateElement):
        raise TypeError("Finite-element DOF points require a scalar coordinate chart.")
    nodes = np.asarray(element.reference_nodes)
    if isinstance(coordinate_element, PerCellRestrictedCellGeometryElement):
        return np.stack(
            tuple(
                np.asarray(
                    coordinate_tabulation(coordinate_element, nodes, cell_row=row)[0]
                )
                for row in range(cell_count)
            )
        )
    return np.asarray(coordinate_tabulation(coordinate_element, nodes)[0])


def _mapped_reference_nodes(weights: ArrayLike, cell_coordinates: ArrayLike, /) -> Any:
    """Physical images ``(cell, node, ambient)`` of the field's reference nodes."""
    specification = "ia,cad->cid" if np.ndim(weights) == 2 else "cia,cad->cid"
    return ein.contract(specification, weights, cell_coordinates)


def _dof_coordinate_owners(
    block_dofs: tuple[Array, ...], global_count: int, /
) -> np.ndarray:
    """First routed ``(block, cell, node)`` slot of every global DOF.

    The coordinate map is continuous across shared entities, so one deterministic
    owner gives the DOF point without averaging away exact interpolatory nodes.
    """
    routed = np.concatenate(
        tuple(np.asarray(routes).reshape((-1,)) for routes in block_dofs)
    )
    dofs, owners = np.unique(routed, return_index=True)
    if not np.array_equal(dofs, np.arange(global_count)):
        raise ValueError("Finite-element DOF points contain unassigned DOFs.")
    return owners.astype(np.int32)


def _build_finite_element_dof_coordinates(
    mesh: CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    layout: _FiniteElementDofLayout,
    block_dofs: tuple[Array, ...],
    coordinate_spec: CellGeometrySpec,
    /,
) -> tuple[
    tuple[np.ndarray, ...], tuple[Array, ...], np.ndarray, int, np.ndarray, np.ndarray
]:
    """DOF points are the cell coordinate map evaluated at the field's nodes."""
    coordinate_elements, coordinate_routes, coordinate_values = coordinate_spec.resolve(
        mesh
    )
    coordinate_weights = tuple(
        _coordinate_node_weights(element, coordinate_element, block.cell_count)
        for block, element, coordinate_element in zip(
            mesh.blocks, resolved, coordinate_elements, strict=True
        )
    )
    match layout.association:
        case "cell":
            boundary = np.zeros((layout.global_count,), dtype=np.bool_)
        case "form_entity":
            if layout.canonical_boundary is None:
                raise RuntimeError("Canonical form boundary was not prepared.")
            boundary = layout.canonical_boundary
        case "entity":
            boundary = _high_order_h1_boundary_mask(mesh, layout)
        case "vertex":
            boundary = _vertex_boundary_mask(mesh, layout.global_count)
        case _:
            raise ValueError("Unsupported finite-element DOF map.")
    owners = _dof_coordinate_owners(block_dofs, layout.global_count)
    values = np.asarray(coordinate_values)
    mapped = np.concatenate(
        tuple(
            np.asarray(
                _mapped_reference_nodes(weights, values[np.asarray(routes)])
            ).reshape((-1, values.shape[1]))
            for weights, routes in zip(coordinate_weights, coordinate_routes, strict=True)
        )
    )
    return (
        coordinate_weights,
        coordinate_routes,
        owners,
        values.shape[0],
        boundary,
        mapped[owners],
    )


def _vertex_boundary_mask(mesh: CellMesh, global_count: int, /) -> np.ndarray:
    boundary = np.zeros((global_count,), dtype=np.bool_)
    boundary[: mesh.coordinates.shape[0]] = np.asarray(
        mesh.topology.entity_sets[0].subset("boundary").mask,
        dtype=np.bool_,
    )
    return boundary


def _high_order_h1_boundary_mask(
    mesh: CellMesh,
    layout: _FiniteElementDofLayout,
    /,
) -> np.ndarray:
    boundary = _vertex_boundary_mask(mesh, layout.global_count)
    connectivity = mesh.connectivity
    if not isinstance(connectivity, _HIGH_ORDER_H1_CONNECTIVITIES):
        raise TypeError("High-order H1 boundary routing requires edge connectivity.")
    edge_starts = layout.edge_starts
    edge_widths = layout.edge_widths
    if edge_starts is None or edge_widths is None:
        raise TypeError("High-order H1 edge offsets are unavailable.")
    for edge in np.flatnonzero(np.asarray(connectivity.boundary_edges, dtype=np.bool_)):
        start = int(edge_starts[edge])
        boundary[start : start + int(edge_widths[edge])] = True
    if isinstance(connectivity, (TetrahedralConnectivity, HexahedralConnectivity)) or (
        isinstance(connectivity, SimplicialConnectivity) and connectivity.dimension == 3
    ):
        face_starts = layout.face_starts
        face_widths = layout.face_widths
        if face_starts is None or face_widths is None:
            raise TypeError("High-order H1 face offsets are unavailable.")
        for face in np.flatnonzero(
            np.asarray(connectivity.boundary_faces, dtype=np.bool_)
        ):
            start = int(face_starts[face])
            boundary[start : start + int(face_widths[face])] = True
    return boundary


def _canonical_finite_element_routes(
    mesh: CellMesh,
    block_dofs: tuple[Array, ...],
    orientations: tuple[Array, ...],
    /,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    canonical_routes = []
    canonical_orientations = []
    for block, routes, orientation in zip(
        mesh.blocks,
        block_dofs,
        orientations,
        strict=True,
    ):
        order = np.argsort(np.asarray(block.global_ids), kind="stable")
        canonical_routes.append(np.asarray(routes)[order])
        canonical_orientations.append(np.asarray(orientation)[order])
    return tuple(canonical_routes), tuple(canonical_orientations)


@final
class FiniteElementDofMap(StrictModule, NonTrainableState):
    """Per-block FE local gathers into one global field coordinate array.

    DOF points are the bound cell coordinate map (affine by default) evaluated at
    each element's reference nodes, so iso- and superparametric curved cells
    place every node on the represented geometry.
    """

    block_names: tuple[str, ...] = eqx.field(static=True)
    cell_dofs: tuple[Array, ...]
    relations: tuple[RowRelation, ...]
    orientations: tuple[Array, ...]
    cell_transforms: tuple[Array, ...]
    cell_coordinate_weights: tuple[Array, ...]
    coordinate_routes: tuple[Array, ...]
    coordinate_owners: Array
    coordinate_count: int = eqx.field(static=True)
    global_dof_count: int = eqx.field(static=True)
    entity_dof_counts: tuple[int, ...] = eqx.field(static=True)
    entity_dofs_per_entity: tuple[int, ...] = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    association: str = eqx.field(static=True)
    boundary_dof_mask: Array
    dof_coordinates: Array
    dof_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        elements: Sequence[FiniteElementSpec],
        /,
        *,
        component_shape: Sequence[int] = (),
        coordinate_spec: CellGeometrySpec | None = None,
    ) -> None:
        resolved = tuple(elements)
        if len(resolved) != len(mesh.blocks):
            raise ValueError("One finite element is required per mesh block.")
        components = tuple(component_shape)
        if any(size <= 0 for size in components):
            raise ValueError("DOF component dimensions must be positive.")
        layout = _prepare_finite_element_dof_layout(mesh, resolved, components)
        association = layout.association
        global_count = layout.global_count
        entity_dof_counts = layout.entity_dof_counts
        entity_dofs_per_entity = layout.entity_dofs_per_entity

        block_dofs, orientations, relations = _build_finite_element_dof_routes(
            mesh, resolved, layout
        )

        geometry = (
            CellGeometrySpec.affine(mesh) if coordinate_spec is None else coordinate_spec
        )
        if not isinstance(geometry, CellGeometrySpec):
            raise TypeError("coordinate_spec must be CellGeometrySpec or None.")
        (
            coordinate_weights,
            coordinate_routes,
            coordinate_owners,
            coordinate_count,
            boundary,
            dof_coordinates,
        ) = _build_finite_element_dof_coordinates(
            mesh, resolved, layout, block_dofs, geometry
        )

        canonical_routes, canonical_orientations = _canonical_finite_element_routes(
            mesh, block_dofs, orientations
        )
        self.block_names = tuple(block.name for block in mesh.blocks)
        self.cell_dofs = tuple(block_dofs)
        self.orientations = tuple(orientations)
        self.cell_transforms = (
            tuple(jnp.asarray(value) for value in layout.canonical_transforms)
            if layout.canonical_transforms is not None
            else tuple(
                jnp.asarray(np.eye(routes.shape[1])[None] * np.asarray(signs)[:, None, :])
                for routes, signs in zip(block_dofs, orientations, strict=True)
            )
        )
        self.cell_coordinate_weights = tuple(
            jnp.asarray(value) for value in coordinate_weights
        )
        self.coordinate_routes = tuple(jnp.asarray(value) for value in coordinate_routes)
        self.coordinate_owners = jnp.asarray(coordinate_owners)
        self.coordinate_count = coordinate_count
        self.relations = tuple(relations)
        self.global_dof_count = global_count
        self.entity_dof_counts = entity_dof_counts
        self.entity_dofs_per_entity = entity_dofs_per_entity
        self.component_shape = components
        self.association = association
        self.boundary_dof_mask = jnp.asarray(boundary)
        self.dof_coordinates = jnp.asarray(dof_coordinates)
        self.dof_map_id = canonical_fingerprint(
            {
                "kind": "finite-element-dof-map",
                "mesh": mesh.topology_id,
                "elements": [element.element_id for element in resolved],
                "global_dof_count": global_count,
                "entity_dof_counts": list(entity_dof_counts),
                "entity_dofs_per_entity": list(entity_dofs_per_entity),
                "component_shape": list(components),
                "association": association,
                "cell_dofs": [
                    array_tree_fingerprint(value) for value in canonical_routes
                ],
                "orientations": [
                    array_tree_fingerprint(value) for value in canonical_orientations
                ],
                "cell_coordinate_weights": [
                    array_tree_fingerprint(np.asarray(value))
                    for value in coordinate_weights
                ],
            }
        )

    def evaluate_coordinates(self, coordinates: ArrayLike, /) -> Array:
        """DOF points for runtime coordinate-map values of the bound geometry."""
        points = jnp.asarray(coordinates)
        if points.ndim != 2 or points.shape[0] != self.coordinate_count:
            raise ValueError(
                "DOF coordinate evaluation requires the bound coordinate-map values."
            )
        mapped = jnp.concatenate(
            tuple(
                _mapped_reference_nodes(weights, points[routes]).reshape(
                    (-1, points.shape[1])
                )
                for weights, routes in zip(
                    self.cell_coordinate_weights, self.coordinate_routes, strict=True
                )
            ),
            axis=0,
        )
        return mapped[self.coordinate_owners]


class FiniteElementBlockGeometry(StrictModule, NonTrainableState):
    block_name: str = eqx.field(static=True)
    reference_points: Array
    reference_weights: Array
    basis_values: Array
    reference_gradients: Array
    physical_points: Array
    physical_gradients: Array
    physical_weights: Array
    measure: Array
    coordinate_measure: CoordinateMeasurePlan | None = None


class FiniteElementRuntimeData(StrictModule, NonTrainableState):
    """Dynamic fixed-topology geometry realization for FE execution."""

    coordinates: Array
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    coordinate_frame: CoordinateFrame = eqx.field(static=True, default="reference")

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        coordinates: ArrayLike,
        /,
        *,
        numeric_version: str,
        geometry_layout_id: str | None = None,
        coordinate_frame: CoordinateFrame = "reference",
    ) -> None:
        points = jnp.asarray(coordinates)
        if points.ndim != 2 or points.shape[1] != mesh.ambient_dimension:
            raise ValueError(
                "Finite-element runtime coordinates must preserve ambient dimension."
            )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        self.coordinates = points
        from ...typing import parse

        self.coordinate_frame = parse(
            coordinate_frame, CoordinateFrame, "coordinate_frame"
        )
        self.topology_id = mesh.topology_id
        layout_id = (
            mesh.geometry_layout_id
            if geometry_layout_id is None
            else str(geometry_layout_id)
        )
        if not layout_id:
            raise ValueError("geometry_layout_id must be non-empty.")
        self.geometry_layout_id = layout_id
        self.numeric_version = version
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "finite-element-runtime",
                "topology": mesh.topology_id,
                "geometry_layout": layout_id,
                "numeric_version": version,
                **(
                    {"coordinate_frame": self.coordinate_frame}
                    if self.coordinate_frame != "reference"
                    else {}
                ),
            }
        )


def _facet_routes(mesh: CellMesh, /) -> tuple[np.ndarray, ...]:
    connectivity = mesh.connectivity
    if isinstance(connectivity, IntervalConnectivity):
        cell_facets = np.asarray(connectivity.cell_vertices, dtype=np.int32)
        valid = np.ones_like(cell_facets, dtype=np.bool_)
        facet_count = connectivity.vertex_count
    elif isinstance(connectivity, PolygonalConnectivity):
        cell_facets = np.asarray(connectivity.cell_edges, dtype=np.int32)
        valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
        facet_count = connectivity.edges.shape[0]
    elif isinstance(connectivity, PolyhedralConnectivity):
        return (
            np.asarray(connectivity.face_owner, dtype=np.int32),
            np.asarray(connectivity.face_neighbor, dtype=np.int32),
            np.asarray(connectivity.face_owner_local, dtype=np.int32),
            np.asarray(connectivity.face_neighbor_local, dtype=np.int32),
        )
    elif isinstance(connectivity, SimplicialConnectivity):
        cell_facets = np.asarray(
            connectivity.cell_entities[mesh.topological_dimension - 1]
        )
        valid = np.ones_like(cell_facets, dtype=np.bool_)
        facet_count = connectivity.entities[mesh.topological_dimension - 1].shape[0]
    else:
        cell_facets = np.asarray(connectivity.cell_faces, dtype=np.int32)
        valid = np.ones_like(cell_facets, dtype=np.bool_)
        facet_count = connectivity.faces.shape[0]
    owner = np.full((facet_count,), -1, dtype=np.int32)
    neighbor = np.full((facet_count,), -1, dtype=np.int32)
    owner_local = np.full((facet_count,), -1, dtype=np.int32)
    neighbor_local = np.full((facet_count,), -1, dtype=np.int32)
    for cell in range(cell_facets.shape[0]):
        for local in range(cell_facets.shape[1]):
            if not valid[cell, local]:
                continue
            facet = int(cell_facets[cell, local])
            if owner[facet] < 0:
                owner[facet] = cell
                owner_local[facet] = local
            else:
                neighbor[facet] = cell
                neighbor_local[facet] = local
    if np.any(owner < 0):
        raise ValueError("Every finite-element facet requires an owner cell.")
    return owner, neighbor, owner_local, neighbor_local


def _cell_metric_determinants(
    cell_kind: str, points: np.ndarray, /
) -> tuple[np.ndarray, str]:
    """Evaluate each admitted cell's metric determinant det(J^T J).

    The value is the square of the measure from
    ``cell_jacobian_inverse_and_measure`` (0 where its solve fails): full-
    dimensional cells square their evaluated det(J), whose binary64 sign is
    reliable, and embedded cells keep the Gram determinant.
    """

    def metric_determinant(edge_matrix: ArrayLike) -> np.ndarray:
        _, _, measure, _, successful = cell_jacobian_inverse_and_measure(
            jnp.asarray(edge_matrix, dtype=jnp.float64)
        )
        return np.asarray(jnp.where(successful, measure * measure, 0.0))

    def edges(origin: int, *others: int) -> np.ndarray:
        return np.stack(
            tuple(points[:, other] - points[:, origin] for other in others), axis=-1
        )

    if cell_kind.startswith("simplex:"):
        return metric_determinant(
            np.swapaxes(points[:, 1:] - points[:, :1], -1, -2)
        ), "Finite-element simplices require positive finite metric determinant."
    if cell_kind.startswith("tensor:"):
        chart = _linear_reference_element(cell_kind)
        gradients = np.asarray(chart.tabulate(chart.reference_nodes)[1])
        jacobian = np.asarray(ein.contract("qir,cid->cqdr", gradients, points))
        return metric_determinant(
            jacobian
        ), "Finite-element tensor cells require positive finite metric determinant."
    if cell_kind == "interval":
        determinant = metric_determinant(edges(0, 1))
    elif cell_kind == "triangle":
        determinant = metric_determinant(edges(0, 1, 2))
    elif cell_kind == "quadrilateral":
        determinant = metric_determinant(edges(0, 1, 3))
    elif cell_kind == "tetrahedron":
        determinant = metric_determinant(edges(0, 1, 2, 3))
    elif cell_kind == "hexahedron":
        determinant = metric_determinant(edges(0, 1, 3, 4))
    elif cell_kind in ("prism", "pyramid"):
        tetrahedra = (
            ((0, 1, 2, 3), (1, 2, 4, 3), (2, 4, 5, 3))
            if cell_kind == "prism"
            else ((0, 1, 2, 4), (0, 2, 3, 4))
        )
        determinant = np.min(
            np.stack(
                tuple(
                    metric_determinant(edges(*tetrahedron)) for tetrahedron in tetrahedra
                ),
                axis=-1,
            ),
            axis=-1,
        )
    else:
        raise ValueError("Unsupported finite-element cell kind.")
    return np.asarray(
        determinant
    ), "Finite-element cells require positive finite metric determinant."


def _validate_mesh_geometry(mesh: CellMesh, /) -> None:
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    for block in mesh.blocks:
        cells = np.asarray(block.vertices, dtype=np.int32)
        determinant, failure = _cell_metric_determinants(
            block.cell_kind, coordinates[cells]
        )
        if np.any(~np.isfinite(determinant)) or np.any(determinant <= 0.0):
            raise ValueError(failure)


def _validate_mixed_prism_tetrahedron_admission(
    mesh: CellMesh,
    fields: tuple[FiniteElementFieldSpec, ...],
    resolved_fields: tuple[tuple[FiniteElementSpec, ...], ...],
    coordinate_spec: CellGeometrySpec,
    /,
) -> None:
    if {block.cell_kind for block in mesh.blocks} != {"prism", "tetrahedron"}:
        return
    if not isinstance(mesh.connectivity, PolyhedralConnectivity):
        raise ValueError(
            "Mixed prism/tetrahedron finite elements require PolyhedralConnectivity."
        )
    canonical_elements = tuple(
        _linear_reference_element(block.cell_kind) for block in mesh.blocks
    )
    if any(
        element.conformity == "H1" and element.degree > 1
        for elements in resolved_fields
        for element in elements
    ):
        raise ValueError(
            "Mixed prism/tetrahedron conforming finite elements support degree 1 only."
        )
    for field, elements in zip(fields, resolved_fields, strict=True):
        if field.component_shape or any(
            element.element_id != canonical.element_id
            for element, canonical in zip(elements, canonical_elements, strict=True)
        ):
            raise ValueError(
                "Mixed prism/tetrahedron finite elements admit only scalar P1 H1 Lagrange fields."
            )
    coordinate_elements, coordinate_dofs, coordinate_values = coordinate_spec.resolve(
        mesh
    )
    if coordinate_values.shape != mesh.coordinates.shape or any(
        not isinstance(element, FiniteElementSpec)
        or element.element_id != canonical.element_id
        or not np.array_equal(
            np.asarray(routes, dtype=np.int32),
            np.asarray(block.vertices, dtype=np.int32),
        )
        for block, element, routes, canonical in zip(
            mesh.blocks,
            coordinate_elements,
            coordinate_dofs,
            canonical_elements,
            strict=True,
        )
    ):
        raise ValueError(
            "Mixed prism/tetrahedron finite elements require affine P1 vertex geometry."
        )


def _coefficient_dtype(value: DTypeLike, /) -> np.dtype:
    dtype = jnp.dtype(jax.dtypes.canonicalize_dtype(np.dtype(value)))
    if not jnp.issubdtype(dtype, jnp.inexact):
        raise TypeError(
            "Finite-element coefficient dtype must be real or complex floating point."
        )
    return dtype


def _validate_vertex_usage(mesh: CellMesh, /) -> None:
    used_vertices = np.unique(
        np.concatenate(
            tuple(
                np.asarray(block.vertices, dtype=np.int32).reshape((-1,))
                for block in mesh.blocks
            )
        )
    )
    if used_vertices.size != mesh.coordinates.shape[0]:
        raise ValueError("FiniteElementPlan requires every mesh vertex to be used.")


@final
class FiniteElementPlan(AbstractDiscretizationPlan):
    mesh: CellMesh
    fields: tuple[FiniteElementFieldSpec, ...]
    coordinate_spec: CellGeometrySpec
    precision_policy: FiniteElementPrecisionPolicy
    coefficient_dtype: np.dtype = eqx.field(static=True)
    coordinate_measure: CoordinateMeasurePlan | None
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        fields: FiniteElementFieldSpec | Sequence[FiniteElementFieldSpec],
        /,
        *,
        precision_policy: FiniteElementPrecisionPolicy | None = None,
        coordinate_spec: CellGeometrySpec | None = None,
        coefficient_dtype: DTypeLike = jnp.float64,
        coordinate_measure: CoordinateMeasurePlan | None = None,
    ) -> None:
        dtype = _coefficient_dtype(coefficient_dtype)
        _validate_mesh_geometry(mesh)
        _validate_vertex_usage(mesh)
        field_specs = (
            (fields,) if isinstance(fields, FiniteElementFieldSpec) else tuple(fields)
        )
        if not field_specs or not all(
            isinstance(field, FiniteElementFieldSpec) for field in field_specs
        ):
            raise TypeError("fields must contain FiniteElementFieldSpec instances.")
        names = tuple(field.name for field in field_specs)
        if len(set(names)) != len(names):
            raise ValueError("Finite-element field names must be unique.")
        resolved_fields = tuple(field.resolve(mesh) for field in field_specs)
        coordinates = (
            CellGeometrySpec.affine(mesh) if coordinate_spec is None else coordinate_spec
        )
        if not isinstance(coordinates, CellGeometrySpec):
            raise TypeError("coordinate_spec must be CellGeometrySpec or None.")
        _validate_mixed_prism_tetrahedron_admission(
            mesh,
            field_specs,
            resolved_fields,
            coordinates,
        )
        if coordinate_measure is not None:
            if not isinstance(coordinate_measure, CoordinateMeasurePlan):
                raise TypeError(
                    "coordinate_measure must be CoordinateMeasurePlan or None."
                )
            _, _, coordinate_values = coordinates.resolve(mesh)
            coordinate_measure.validate_coordinates(coordinate_values)
            if coordinate_measure.kind == "axisymmetric-rz":
                if mesh.topological_dimension != 2 or mesh.ambient_dimension != 2:
                    raise ValueError(
                        "Axisymmetric RZ requires a two-dimensional RZ mesh."
                    )
            elif coordinate_measure.kind != "cartesian":
                if mesh.topological_dimension != 1 or mesh.ambient_dimension != 1:
                    raise ValueError("Radial reductions require a one-dimensional mesh.")
        precision = (
            FiniteElementPrecisionPolicy()
            if precision_policy is None
            else precision_policy
        )
        if not isinstance(precision, FiniteElementPrecisionPolicy):
            raise TypeError(
                "precision_policy must be FiniteElementPrecisionPolicy or None."
            )
        self.mesh = mesh
        self.fields = field_specs
        self.coordinate_spec = coordinates
        self.precision_policy = precision
        self.coefficient_dtype = dtype
        self.coordinate_measure = coordinate_measure
        self.key = DiscretizationKey("finite_element", DiscretizationRole.PHYSICAL)
        self.capabilities = (
            DiscretizationCapability.PROJECTION,
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.BOUNDARY_INTEGRAL,
            DiscretizationCapability.VARIATIONAL_ASSEMBLY,
            DiscretizationCapability.SPARSE_ASSEMBLY,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-plan",
                "mesh": mesh.mesh_id,
                "fields": [field.field_spec_id for field in field_specs],
                "coordinate_spec": coordinates.geometry_layout_id,
                "precision_policy": precision.policy_id,
                "coefficient_dtype": dtype.str,
                **(
                    {"coordinate_measure": coordinate_measure.plan_id}
                    if coordinate_measure is not None
                    and coordinate_measure.kind != "cartesian"
                    else {}
                ),
            }
        )

    def prepare(self, /, *, numeric_version: str = "0") -> Any:
        return FiniteElementDiscretization(self, numeric_version=numeric_version)


def _finite_element_field_layout(
    mesh: CellMesh, dof_map: FiniteElementDofMap, components: tuple[int, ...], /
) -> EntityDofLayout | BlockDofLayout:
    vertex_count = mesh.coordinates.shape[0]
    if dof_map.association == "vertex":
        return EntityDofLayout(
            mesh.topology.entity_sets[0].entity_set_id,
            vertex_count,
            vertex_count,
            component_shape=components,
        )
    if dof_map.association in ("entity", "form_entity"):
        entity_names = (
            ("vertices", "edges", "faces", "cells")
            if mesh.topological_dimension <= 3
            else tuple(
                f"entities_{degree}" for degree in range(mesh.topological_dimension + 1)
            )
        )
        block_names, layouts = [], []
        for dimension, (entity_dof_count, dofs_per_entity) in enumerate(
            zip(dof_map.entity_dof_counts, dof_map.entity_dofs_per_entity, strict=True)
        ):
            if entity_dof_count == 0:
                continue
            entities = mesh.topology.entity_sets[dimension]
            block_names.append(entity_names[dimension])
            layouts.append(
                EntityDofLayout(
                    entities.entity_set_id,
                    entities.count,
                    entity_dof_count,
                    dofs_per_entity=dofs_per_entity,
                    component_shape=components,
                )
            )
        return BlockDofLayout(tuple(block_names), tuple(layouts))
    if dof_map.association == "edge":
        edge_count = dof_map.global_dof_count
        return EntityDofLayout(
            mesh.topology.entity_sets[1].entity_set_id,
            edge_count,
            edge_count,
            component_shape=components,
        )
    if dof_map.association == "cell":
        cell_dof_count = dof_map.global_dof_count
        return EntityDofLayout(
            mesh.topology.entity_sets[mesh.topological_dimension].entity_set_id,
            cell_dof_count,
            cell_dof_count,
            component_shape=components,
        )
    raise ValueError("Unknown finite-element DOF association.")


def _prepare_finite_element_field(
    mesh: CellMesh,
    field: FiniteElementFieldSpec,
    coefficient_dtype: np.dtype,
    coordinate_spec: CellGeometrySpec,
    precision_policy: FiniteElementPrecisionPolicy,
    /,
    *,
    coordinate_measure: CoordinateMeasurePlan | None = None,
) -> tuple[
    tuple[FiniteElementSpec, ...],
    FiniteElementDofMap,
    DiscreteFieldSpace,
    tuple[FiniteElementBlockGeometry, ...],
]:
    elements = field.resolve(mesh)
    coordinate_elements, coordinate_dofs, coordinate_values = coordinate_spec.resolve(
        mesh
    )
    dof_map = FiniteElementDofMap(
        mesh,
        elements,
        component_shape=field.component_shape,
        coordinate_spec=coordinate_spec,
    )
    vector_shape = (dof_map.global_dof_count,) + field.component_shape
    vector_space = ArraySpace(vector_shape, dtype=coefficient_dtype)
    layout = _finite_element_field_layout(mesh, dof_map, field.component_shape)
    conformity = elements[0].conformity
    representation = elements[0].representation
    space = DiscreteFieldSpace(
        field.name,
        mesh.support.support_id,
        layout,
        vector_space,
        representation=representation,
        conformity=conformity,
        form_type=FormType(
            elements[0].value_spec.form_type.dimension,
            elements[0].value_spec.form_type.degree,
            twist=elements[0].value_spec.form_type.twist,
            fiber_shape=elements[0].value_spec.form_type.fiber_shape
            + field.component_shape,
            ambient_dimension=mesh.ambient_dimension,
        ),
        projection_id=canonical_fingerprint(
            {"kind": "finite-element-projection", "field": field.field_spec_id}
        ),
        reconstruction_id=canonical_fingerprint(
            {"kind": "finite-element-reconstruction", "field": field.field_spec_id}
        ),
    )
    geometries = tuple(
        _prepare_block_geometry(
            mesh,
            block,
            element,
            coordinates=coordinate_values,
            coordinate_element=coordinate_element,
            geometry_dofs=geometry_dofs,
            precision_policy=precision_policy,
            coordinate_measure=coordinate_measure,
        )
        for block, element, coordinate_element, geometry_dofs in zip(
            mesh.blocks, elements, coordinate_elements, coordinate_dofs, strict=True
        )
    )
    return elements, dof_map, space, geometries


def _finite_element_domains(
    mesh: CellMesh, routes: tuple[np.ndarray, ...], /
) -> tuple[IntegrationDomain, IntegrationDomain, IntegrationDomain]:
    owner, neighbor, owner_local, neighbor_local = routes
    exterior = np.flatnonzero(neighbor < 0)
    interior = np.flatnonzero(neighbor >= 0)
    cell_count = sum(block.cell_count for block in mesh.blocks)
    cell_entities = mesh.topology.entity_sets[mesh.topological_dimension]
    facet_entities = mesh.topology.entity_sets[mesh.topological_dimension - 1]
    cell_domain = IntegrationDomain(
        "cell",
        np.arange(cell_count, dtype=np.int32),
        mesh.support.support_id,
        cell_entities.entity_set_id,
        owner_cells=np.arange(cell_count, dtype=np.int32),
    )
    exterior_domain = IntegrationDomain(
        "exterior_facet",
        exterior,
        mesh.support.support_id,
        facet_entities.entity_set_id,
        owner_cells=owner[exterior],
        neighbor_cells=neighbor[exterior],
        owner_local_entities=owner_local[exterior],
        neighbor_local_entities=neighbor_local[exterior],
    )
    interior_domain = IntegrationDomain(
        "interior_facet",
        interior,
        mesh.support.support_id,
        facet_entities.entity_set_id,
        owner_cells=owner[interior],
        neighbor_cells=neighbor[interior],
        owner_local_entities=owner_local[interior],
        neighbor_local_entities=neighbor_local[interior],
    )
    return cell_domain, exterior_domain, interior_domain


@final
class FiniteElementDiscretization(AbstractPreparedLocalDiscretization):
    mesh: CellMesh
    dof_maps: tuple[FiniteElementDofMap, ...]
    default_runtime: FiniteElementRuntimeData
    elements: tuple[tuple[FiniteElementSpec, ...], ...]
    coordinate_elements: tuple[CellGeometryElement, ...]
    coordinate_dofs: tuple[Array, ...]
    block_geometries: tuple[tuple[FiniteElementBlockGeometry, ...], ...]
    cell_domain: IntegrationDomain
    exterior_facet_domain: IntegrationDomain
    interior_facet_domain: IntegrationDomain
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    block_space: BlockSpace
    measures: tuple[DiscreteMeasure, ...]
    precision_policy: FiniteElementPrecisionPolicy
    coefficient_dtype: np.dtype = eqx.field(static=True)
    coordinate_measure: CoordinateMeasurePlan | None
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport

    @checked
    def __init__(self, plan: FiniteElementPlan, /, *, numeric_version: str = "0") -> None:
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        mesh = plan.mesh
        coordinate_elements, coordinate_dofs, coordinate_values = (
            plan.coordinate_spec.resolve(mesh)
        )
        field_spaces = []
        dof_maps = []
        all_elements = []
        all_geometries = []
        cell_measures = None
        for field in plan.fields:
            elements, dof_map, field_space, geometries = _prepare_finite_element_field(
                mesh,
                field,
                plan.coefficient_dtype,
                plan.coordinate_spec,
                plan.precision_policy,
                coordinate_measure=plan.coordinate_measure,
            )
            all_elements.append(elements)
            dof_maps.append(dof_map)
            field_spaces.append(field_space)
            if (
                plan.coordinate_measure is not None
                and plan.coordinate_measure.kind != "cartesian"
            ):
                for geometry in geometries:
                    plan.coordinate_measure.validate_coordinates(geometry.physical_points)
                    if np.any(np.asarray(geometry.physical_weights) <= 0.0):
                        raise ValueError(
                            "Rotational volume quadrature must lie strictly off the axis."
                        )
            all_geometries.append(geometries)
            if cell_measures is None:
                cell_measures = jnp.concatenate(
                    tuple(geometry.measure for geometry in geometries),
                    axis=0,
                )
        if cell_measures is None:
            raise ValueError("Finite-element preparation produced no cell measures.")
        top_entities = mesh.topology.entity_sets[mesh.topological_dimension]
        measure_metadata = (
            DiscreteMeasure(
                "finite_element_cell_measure",
                mesh.support.support_id,
                top_entities.entity_set_id,
                cell_measures,
            ),
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "reference elements are compatible with mesh blocks",
                "cell geometry measures are positive",
                "local-to-global DOF routes are fixed",
            ),
            resource_counts={
                "vertices": mesh.coordinates.shape[0],
                "cells": sum(block.cell_count for block in mesh.blocks),
                "fields": len(plan.fields),
                "global_dofs": sum(dof_map.global_dof_count for dof_map in dof_maps),
            },
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=mesh.support,
            field_spaces=tuple(field_spaces),
            measures=measure_metadata,
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        facet_routes = _facet_routes(mesh)
        self.mesh = mesh
        self.default_runtime = FiniteElementRuntimeData(
            mesh,
            coordinate_values,
            numeric_version=version,
            geometry_layout_id=plan.coordinate_spec.geometry_layout_id,
            coordinate_frame=(
                plan.coordinate_measure.frame
                if plan.coordinate_measure is not None
                else "reference"
            ),
        )
        self.dof_maps = tuple(dof_maps)
        self.elements = tuple(all_elements)
        self.coordinate_elements = coordinate_elements
        self.coordinate_dofs = coordinate_dofs
        self.block_geometries = tuple(all_geometries)
        self.cell_domain, self.exterior_facet_domain, self.interior_facet_domain = (
            _finite_element_domains(mesh, facet_routes)
        )
        self.key = plan.key
        self.precision_policy = plan.precision_policy
        self.coefficient_dtype = plan.coefficient_dtype
        self.coordinate_measure = plan.coordinate_measure
        self.support = mesh.support
        self.block_space = BlockSpace(
            tuple(space.vector_space for space in spaces),
            names=tuple(space.name for space in spaces),
        )
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.numeric_version = version
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-element",
                "plan": plan.plan_id,
                "mesh": mesh.mesh_id,
                "numeric_version": version,
            }
        )

    @property
    def precision_evidence(self) -> Any:
        return self.precision_policy.evidence()

    def prepare_runtime(
        self,
        coordinates: ArrayLike | None = None,
        /,
        *,
        numeric_version: str,
        coordinate_frame: CoordinateFrame | None = None,
    ) -> FiniteElementRuntimeData:
        points = jnp.asarray(
            self.default_runtime.coordinates if coordinates is None else coordinates
        )
        if points.shape != self.default_runtime.coordinates.shape:
            raise ValueError(
                "Fixed-topology FE runtime coordinates must preserve coordinate shape."
            )
        frame = (
            self.default_runtime.coordinate_frame
            if coordinate_frame is None
            else coordinate_frame
        )
        if self.coordinate_measure is not None and frame != self.coordinate_measure.frame:
            raise ValueError(
                "Runtime coordinate frame must match the coordinate measure."
            )
        return FiniteElementRuntimeData(
            self.mesh,
            points,
            numeric_version=numeric_version,
            geometry_layout_id=self.default_runtime.geometry_layout_id,
            coordinate_frame=frame,
        )

    def local_variational_capabilities(self, /) -> LocalVariationalCapabilities:
        from ._local_provider import FiniteElementLocalProvider

        return FiniteElementLocalProvider(self).local_variational_capabilities()

    def local_field_binding(self, name: str, /) -> LocalFieldBinding:
        from ._local_provider import FiniteElementLocalProvider

        return FiniteElementLocalProvider(self).local_field_binding(name)

    def prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
    ) -> tuple[PreparedLocalRegion, ...]:
        from ._local_provider import FiniteElementLocalProvider

        return FiniteElementLocalProvider(self).prepare_local_regions(
            domain,
            field_names=field_names,
            maximum_derivative_order=maximum_derivative_order,
            kernel_mode=kernel_mode,
        )

    def validate_local_runtime(self, runtime: object, /) -> None:
        from ._local_provider import FiniteElementLocalProvider

        FiniteElementLocalProvider(self).validate_local_runtime(runtime)

    @property
    def mass(self) -> SparseLinearMap:
        if len(self.field_spaces) != 1:
            raise ValueError("mass is only unambiguous for one field.")
        return self._field_operators(
            self.field_spaces[0].name, self.default_runtime, (True,)
        )[0]

    @property
    def stiffness(self) -> SparseLinearMap:
        if len(self.field_spaces) != 1:
            raise ValueError("stiffness is only unambiguous for one field.")
        return self._field_operators(
            self.field_spaces[0].name, self.default_runtime, (False,)
        )[0]

    @property
    def vertices(self) -> Array:
        return self.mesh.coordinates

    @property
    def boundary_dof_mask(self) -> Array:
        if len(self.dof_maps) != 1:
            raise ValueError("boundary_dof_mask is only unambiguous for one field.")
        return self.dof_maps[0].boundary_dof_mask

    def axis_dof_mask(self, field_name: str, /) -> Array:
        """Identify regular-axis DOFs for the owner's radial displacement condition.

        Radial displacement must vanish here, unlike temperature or axial motion.
        """
        measure = self.coordinate_measure
        if measure is None or measure.radial_axis is None:
            raise ValueError(
                "Axis DOFs require a declared rotational coordinate measure."
            )
        dof_map = self.dof_maps[self._field_index(field_name)]
        return dof_map.dof_coordinates[..., measure.radial_axis] == 0.0

    @checked
    def dof_indices(
        self,
        field_name: str,
        selection: EntitySelection,
        /,
    ) -> Array:
        """Return sorted P1 DOFs incident to selected cells or facets."""

        field_index = self._field_index(field_name)
        dof_map = self.dof_maps[field_index]
        field_elements = self.elements[field_index]
        if dof_map.association != "vertex" or any(
            element.conformity != "H1" or element.degree != 1
            for element in field_elements
        ):
            raise ValueError(
                "Entity selection to DOFs supports only vertex-associated P1 H1 fields."
            )
        selected_entities = tuple(
            entities
            for entities in self.mesh.topology.entity_sets
            if entities.entity_set_id == selection.entity_set_id
        )
        if len(selected_entities) != 1:
            raise ValueError("Entity selection does not belong to this mesh topology.")
        entities = selected_entities[0]
        dimension = entities.intrinsic_dimension
        if dimension not in (
            self.mesh.topological_dimension,
            self.mesh.topological_dimension - 1,
        ):
            raise ValueError("P1 DOFs may be selected only from mesh cells or facets.")
        entity_mask = np.asarray(selection.mask, dtype=np.bool_)
        active_mask = np.asarray(selection.active_mask, dtype=np.bool_)
        expected_active = np.asarray(entities.active_mask, dtype=np.bool_)
        if (
            entity_mask.shape != (entities.count,)
            or active_mask.shape != (entities.count,)
            or not np.array_equal(active_mask, expected_active)
            or np.any(entity_mask & ~expected_active)
        ):
            raise ValueError(
                "Entity selection must match the mesh entity capacity and active mask."
            )
        for degree in range(dimension, 0, -1):
            incidence = self.mesh.topology.incidences[degree - 1]
            relation = incidence.relation
            valid = np.asarray(relation.valid, dtype=np.bool_)
            source = np.asarray(relation.source_indices, dtype=np.int32)
            target = np.asarray(relation.target_indices, dtype=np.int32)
            routes = valid & entity_mask[target]
            lower_entities = self.mesh.topology.entity_sets[degree - 1]
            lower_mask = np.zeros((lower_entities.count,), dtype=np.bool_)
            lower_mask[source[routes]] = True
            entity_mask = lower_mask & np.asarray(
                lower_entities.active_mask,
                dtype=np.bool_,
            )
        indices = np.flatnonzero(entity_mask).astype(np.int32)
        if indices.size and int(indices[-1]) >= dof_map.global_dof_count:
            raise ValueError("P1 topology and field DOF map are inconsistent.")
        return jnp.asarray(indices, dtype=jnp.int32)

    def dof_mask(
        self,
        field_name: str,
        selection: EntitySelection,
        /,
    ) -> Array:
        """Project a supported cell/facet selection onto the P1 DOF axis."""

        field_index = self._field_index(field_name)
        mask = np.zeros((self.dof_maps[field_index].global_dof_count,), dtype=np.bool_)
        mask[np.asarray(self.dof_indices(field_name, selection), dtype=np.int32)] = True
        return jnp.asarray(mask)

    def project(
        self,
        field_name: str,
        function: Callable[[Array, object], ArrayLike],
        /,
        *,
        runtime: FiniteElementRuntimeData | None = None,
        args: object = None,
    ) -> Array:
        if not callable(function):
            raise TypeError("function must be callable.")
        field_index = self._field_index(field_name)
        realized = self.default_runtime if runtime is None else runtime
        if self.elements[field_index][0].form_basis is not None:
            from ._cell_map import PreparedFiniteElementCellMap

            dofs = self.dof_maps[field_index]
            moments = jnp.zeros((dofs.global_dof_count,), dtype=self.coefficient_dtype)
            counts = jnp.zeros((dofs.global_dof_count,), dtype=jnp.int32)
            for block_index, element in enumerate(self.elements[field_index]):
                basis = element.form_basis
                if basis is None:
                    raise ValueError(
                        "Canonical projection requires complete form-basis metadata."
                    )
                cell_map = PreparedFiniteElementCellMap(self, block_index)
                for cell in range(cell_map.cell_count):
                    cells = jnp.full(
                        (basis.functional_points.shape[0],), cell, dtype=jnp.int32
                    )
                    geometry = cell_map.evaluate(
                        realized.coordinates, cells, basis.functional_points
                    )
                    values = vector_to_form(
                        jnp.asarray(function(geometry.physical_points, args)),
                        element.value_spec,
                    )
                    reference = pullback(
                        values, element.value_spec.form_type, geometry.jacobian
                    )
                    local = jnp.linalg.solve(
                        dofs.cell_transforms[block_index][cell],
                        basis.interpolate(reference),
                    )
                    routes = dofs.cell_dofs[block_index][cell]
                    moments = moments.at[routes].add(local)
                    counts = counts.at[routes].add(1)
            return moments / counts
        coordinates = self.dof_maps[field_index].evaluate_coordinates(
            realized.coordinates
        )
        values = jnp.asarray(function(coordinates, args))
        return self.field_spaces[field_index].vector_space.validate(values)

    def integration_domain(
        self,
        kind: str,
        selection: EntitySelection | None = None,
        /,
    ) -> IntegrationDomain:
        domains = {
            "cell": self.cell_domain,
            "exterior_facet": self.exterior_facet_domain,
            "interior_facet": self.interior_facet_domain,
        }
        if kind not in domains:
            raise ValueError("Unknown finite-element integration-domain kind.")
        base = domains[kind]
        if selection is None:
            return base
        if not isinstance(selection, EntitySelection):
            raise TypeError("selection must be EntitySelection or None.")
        if selection.entity_set_id != base.entity_set_id:
            raise ValueError("Entity selection does not match the domain entity set.")
        entity_mask = np.asarray(selection.mask, dtype=np.bool_)
        base_entities = np.asarray(base.entity_indices, dtype=np.int32)
        selected_rows = np.flatnonzero(entity_mask[base_entities])
        return IntegrationDomain(
            base.kind,
            base_entities[selected_rows],
            base.support_id,
            base.entity_set_id,
            owner_cells=np.asarray(base.owner_cells)[selected_rows],
            neighbor_cells=np.asarray(base.neighbor_cells)[selected_rows],
            owner_local_entities=np.asarray(base.owner_local_entities)[selected_rows],
            neighbor_local_entities=np.asarray(base.neighbor_local_entities)[
                selected_rows
            ],
            neighbor_trace_permutations=np.asarray(base.neighbor_trace_permutations)[
                selected_rows
            ],
            periodic_face_mask=np.asarray(base.periodic_face_mask)[selected_rows],
            selection_id=selection.selection_id,
        )

    def cell_block_domain(self, block_name: str, /) -> IntegrationDomain:
        """Return the exact global cell domain owned by one named mesh block."""

        name = str(block_name)
        names = tuple(block.name for block in self.mesh.blocks)
        if name not in names:
            raise ValueError(f"Unknown finite-element cell block {name!r}.")
        index = names.index(name)
        start = sum(block.cell_count for block in self.mesh.blocks[:index])
        stop = start + self.mesh.blocks[index].cell_count
        entities = np.arange(start, stop, dtype=np.int32)
        return IntegrationDomain(
            "cell",
            entities,
            self.cell_domain.support_id,
            self.cell_domain.entity_set_id,
            owner_cells=entities,
            selection_id=canonical_fingerprint(
                {
                    "kind": "finite-element-cell-block-domain",
                    "prepared": self.prepared_id,
                    "block": name,
                }
            ),
        )

    def trace(
        self,
        field_name: str,
        coefficients: ArrayLike,
        /,
        *,
        runtime: FiniteElementRuntimeData | None = None,
    ) -> tuple[Array, Array]:
        field_index = self._field_index(field_name)
        values = self.field_spaces[field_index].vector_space.validate(coefficients)
        realized = self.default_runtime if runtime is None else runtime
        coordinates = self.dof_maps[field_index].evaluate_coordinates(
            realized.coordinates
        )
        mask = self.dof_maps[field_index].boundary_dof_mask
        return coordinates[mask], values[mask]

    def prepare_side_trace(
        self,
        field_name: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
        runtime: FiniteElementRuntimeData | None = None,
    ) -> PreparedTraceAction:
        """Prepare the exact trace of one field on selected facets.

        See `prepare_finite_element_side_trace`.
        """
        from ._point_interpolation import prepare_finite_element_side_trace

        return prepare_finite_element_side_trace(
            self,
            field_name,
            domain,
            rule=rule,
            quantity=quantity,
            side=side,
            runtime=runtime,
        )

    def reconstruct(
        self,
        field_name: str,
        coefficients: ArrayLike,
        block_name: str,
        reference_points: ArrayLike,
        /,
        *,
        runtime: FiniteElementRuntimeData | None = None,
    ) -> Array:
        field_index = self._field_index(field_name)
        block_index = self.dof_maps[field_index].block_names.index(str(block_name))
        points = jnp.asarray(reference_points)
        realized = self.default_runtime if runtime is None else runtime
        geometry = self.evaluate_block_geometry(
            field_name,
            block_index,
            realized.coordinates,
            points,
            jnp.ones((points.shape[0],), dtype=points.dtype),
        )
        local = jnp.asarray(coefficients)[
            self.dof_maps[field_index].cell_dofs[block_index]
        ]
        local = ein.contract(
            "cij,cj...->ci...",
            self.dof_maps[field_index].cell_transforms[block_index],
            local,
        )
        if geometry.basis_values.ndim == 2:
            return ein.contract("qi,ci...->cq...", geometry.basis_values, local)
        if geometry.basis_values.ndim == 3:
            return ein.contract("cqi,ci...->cq...", geometry.basis_values, local)
        return ein.contract("cqiv,ci->cqv", geometry.basis_values, local)

    def _field_index(self, field_name: str, /) -> int:
        requested = str(field_name)
        for index, field_space in enumerate(self.field_spaces):
            if field_space.name == requested:
                return index
        raise KeyError(f"Unknown finite-element field {requested!r}.")

    def _field_elements(self, field_index: int, /) -> tuple[FiniteElementSpec, ...]:
        return self.elements[field_index]

    def evaluate_geometry(
        self,
        field_name: str,
        coordinates: ArrayLike,
        /,
    ) -> tuple[FiniteElementBlockGeometry, ...]:
        field_index = self._field_index(field_name)
        points = jnp.asarray(coordinates)
        if points.shape != self.default_runtime.coordinates.shape:
            raise ValueError(
                "Fixed-topology FE geometry evaluation must preserve coordinate shape."
            )
        return tuple(
            _prepare_block_geometry(
                self.mesh,
                block,
                element,
                coordinates=points,
                coordinate_element=coordinate_element,
                geometry_dofs=geometry_dofs,
                precision_policy=self.precision_policy,
                coordinate_measure=self.coordinate_measure,
            )
            for block, element, coordinate_element, geometry_dofs in zip(
                self.mesh.blocks,
                self.elements[field_index],
                self.coordinate_elements,
                self.coordinate_dofs,
                strict=True,
            )
        )

    def evaluate_block_geometry(
        self,
        field_name: str,
        block_index: int,
        coordinates: ArrayLike,
        reference_points: ArrayLike,
        reference_weights: ArrayLike,
        /,
    ) -> FiniteElementBlockGeometry:
        field_index = self._field_index(field_name)
        index = int(block_index)
        if index < 0 or index >= len(self.mesh.blocks):
            raise IndexError("block_index is outside the finite-element mesh.")
        return _prepare_block_geometry(
            self.mesh,
            self.mesh.blocks[index],
            self.elements[field_index][index],
            coordinate_element=self.coordinate_elements[index],
            geometry_dofs=self.coordinate_dofs[index],
            coordinates=coordinates,
            reference_points=reference_points,
            reference_weights=reference_weights,
            precision_policy=self.precision_policy,
            coordinate_measure=self.coordinate_measure,
        )

    def _field_operator_geometry(
        self, field_name: str, runtime: FiniteElementRuntimeData, /
    ) -> tuple[FiniteElementDofMap, tuple[FiniteElementBlockGeometry, ...]]:
        if (
            runtime.topology_id != self.mesh.topology_id
            or runtime.geometry_layout_id != self.default_runtime.geometry_layout_id
        ):
            raise ValueError("Finite-element runtime does not match this discretization.")
        if (
            self.coordinate_measure is not None
            and runtime.coordinate_frame != self.coordinate_measure.frame
        ):
            raise ValueError(
                "Runtime coordinate frame must match the coordinate measure."
            )
        field_index = self._field_index(field_name)
        geometries = self.evaluate_geometry(field_name, runtime.coordinates)
        return self.dof_maps[field_index], geometries

    def _assemble_geometry_operator(
        self,
        dof_map: FiniteElementDofMap,
        geometries: tuple[FiniteElementBlockGeometry, ...],
        /,
        *,
        mass: bool,
    ) -> SparseLinearMap:
        tensors = tuple(
            _local_mass_tensor(geometry) if mass else _local_stiffness_tensor(geometry)
            for geometry in geometries
        )
        return _assemble_local_operator(
            dof_map,
            tensors,
            "finite-element-mass" if mass else "finite-element-stiffness",
            positive_definite=mass,
            component_shape=dof_map.component_shape,
            coefficient_dtype=self.coefficient_dtype,
        )

    @checked
    def assemble_field_operators(
        self, field_name: str, runtime: FiniteElementRuntimeData, /
    ) -> tuple[SparseLinearMap, SparseLinearMap]:
        mass, stiffness = self._field_operators(field_name, runtime, (True, False))
        return mass, stiffness

    def _field_operators(
        self,
        field_name: str,
        runtime: FiniteElementRuntimeData,
        kinds: tuple[bool, ...],
        /,
    ) -> tuple[SparseLinearMap, ...]:
        """Assemble mass (``True``) and stiffness (``False``) operators.

        Assembly runs as one compiled program per discretization. The default
        runtime reads the prepared block geometry; any other runtime evaluates
        its geometry anew inside the program.
        """
        if runtime is self.default_runtime:
            return run_owned_program(
                self,
                "prepared-field-operators",
                FiniteElementDiscretization._prepared_field_operators,
                str(field_name),
                kinds,
            )
        return run_owned_program(
            self,
            "runtime-field-operators",
            FiniteElementDiscretization._runtime_field_operators,
            str(field_name),
            runtime,
            kinds,
        )

    def _prepared_field_operators(
        self, field_name: str, kinds: tuple[bool, ...], /
    ) -> tuple[SparseLinearMap, ...]:
        field_index = self._field_index(field_name)
        return tuple(
            self._assemble_geometry_operator(
                self.dof_maps[field_index], self.block_geometries[field_index], mass=mass
            )
            for mass in kinds
        )

    def _runtime_field_operators(
        self,
        field_name: str,
        runtime: FiniteElementRuntimeData,
        kinds: tuple[bool, ...],
        /,
    ) -> tuple[SparseLinearMap, ...]:
        dof_map, geometries = self._field_operator_geometry(field_name, runtime)
        return tuple(
            self._assemble_geometry_operator(dof_map, geometries, mass=mass)
            for mass in kinds
        )

    def assemble_cell_operator(
        self,
        field_name: str,
        local_values: Sequence[ArrayLike],
        /,
        *,
        operator_id: str,
        properties: OperatorProperties | None = None,
    ) -> SparseLinearMap:
        """Assemble fixed-topology cell matrices without owning equation semantics."""
        field_index = self._field_index(field_name)
        values = tuple(jnp.asarray(value) for value in local_values)
        identifier = str(operator_id)
        if not identifier:
            raise ValueError("Cell operator_id must be non-empty.")
        if len(values) != len(self.mesh.blocks):
            raise ValueError("Cell operator requires one local tensor per mesh block.")
        dof_map = self.dof_maps[field_index]
        for block_index, (block, value) in enumerate(
            zip(self.mesh.blocks, values, strict=True)
        ):
            width = dof_map.cell_dofs[block_index].shape[1]
            expected = (block.cell_count, width, width)
            if value.shape != expected:
                raise ValueError(
                    f"Cell operator tensor shape must be {expected!r}; got {value.shape!r}."
                )
        properties_ = OperatorProperties() if properties is None else properties
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties or None.")
        return _assemble_local_operator(
            dof_map,
            values,
            identifier,
            positive_definite=False,
            component_shape=dof_map.component_shape,
            properties=properties_,
            coefficient_dtype=self.coefficient_dtype,
        )


def _local_mass_tensor(geometry: FiniteElementBlockGeometry, /) -> Array:
    weights = geometry.physical_weights
    if (
        geometry.coordinate_measure is not None
        and geometry.coordinate_measure.kind != "cartesian"
    ):
        # Nodal quadrature at the axis cannot certify positive capacity.
        valid = jnp.all(jnp.isfinite(weights) & (weights > 0.0), axis=-1)
        weights = jnp.where(valid[:, None], weights, jnp.nan)
    if geometry.basis_values.ndim == 2:
        return ein.contract(
            "cq,qi,qj->cij",
            weights,
            geometry.basis_values,
            geometry.basis_values,
        )
    if geometry.basis_values.ndim == 3:
        return ein.contract(
            "cq,cqi,cqj->cij",
            geometry.physical_weights,
            geometry.basis_values,
            geometry.basis_values,
        )
    return ein.contract(
        "cq,cqiv,cqjv->cij",
        weights,
        geometry.basis_values,
        geometry.basis_values,
    )


def _local_stiffness_tensor(geometry: FiniteElementBlockGeometry, /) -> Array:
    if geometry.physical_gradients.ndim == 4:
        return ein.contract(
            "cq,cqid,cqjd->cij",
            geometry.physical_weights,
            geometry.physical_gradients,
            geometry.physical_gradients,
        )
    return ein.contract(
        "cq,cqivd,cqjvd->cij",
        geometry.physical_weights,
        geometry.physical_gradients,
        geometry.physical_gradients,
    )


_NAMED_REFERENCE_LAYOUTS: dict[str, tuple[str, int]] = {
    "interval": ("tensor", 1),
    "quadrilateral": ("tensor", 2),
    "hexahedron": ("tensor", 3),
    "triangle": ("simplex", 2),
    "tetrahedron": ("simplex", 3),
    "prism": ("prism", 3),
    "pyramid": ("pyramid", 3),
}


def _reference_rule_layout(cell_kind: str, /) -> tuple[str, int]:
    if cell_kind.startswith(("simplex:", "tensor:")):
        kind, dimension_text = cell_kind.split(":")
        return kind, int(dimension_text)
    layout = _NAMED_REFERENCE_LAYOUTS.get(cell_kind)
    if layout is None:
        raise ValueError("Unsupported finite-element cell kind.")
    return layout


def _collapse_weight_degrees(kind: str, dimension: int, /) -> tuple[int, ...]:
    """Degree of the collapsed-coordinate Jacobian on each Gauss axis."""
    if kind == "simplex":
        return tuple(dimension - axis - 1 for axis in range(dimension))
    if kind == "prism":
        return (1, 0, 0)
    if kind == "pyramid":
        return (0, 0, 2)
    return (0,) * dimension


def _reference_rule_axis_counts(
    cell_kind: str, polynomial_degree: int, geometry_degree: int | None, /
) -> tuple[int, ...]:
    """Smallest Gauss-Legendre count per collapsed axis for the required integrand.

    The integrand pairs two basis functions of ``polynomial_degree`` (total
    degree ``2 * polynomial_degree``) with the metric determinant of the
    geometry map: none for an affine map (``geometry_degree`` None), total
    degree ``d (g - 1)`` for a degree-``g`` simplex map, and degree ``d g - 1``
    per axis for any other degree-``g`` map. Each collapsed axis adds its
    collapse-Jacobian degree; ``n`` points integrate degree ``2 n - 1``.
    """
    kind, dimension = _reference_rule_layout(cell_kind)
    if polynomial_degree < 0:
        raise ValueError("Reference rule polynomial degree must be nonnegative.")
    if geometry_degree is None:
        measure_degree = 0
    elif kind == "simplex":
        measure_degree = dimension * (geometry_degree - 1)
    else:
        measure_degree = dimension * geometry_degree - 1
    integrand = 2 * int(polynomial_degree) + measure_degree
    return tuple(
        (integrand + weight) // 2 + 1
        for weight in _collapse_weight_degrees(kind, dimension)
    )


def _reference_rule_exactness(
    cell_kind: str, polynomial_degree: int, geometry_degree: int | None = None, /
) -> int:
    """Total polynomial degree the reference rule integrates exactly on its cell."""
    kind, dimension = _reference_rule_layout(cell_kind)
    counts = _reference_rule_axis_counts(cell_kind, polynomial_degree, geometry_degree)
    return min(
        2 * count - 1 - weight
        for count, weight in zip(
            counts, _collapse_weight_degrees(kind, dimension), strict=True
        )
    )


def _degree_aware_reference_rule(
    cell_kind: str,
    polynomial_degree: int,
    /,
    *,
    geometry_degree: int | None = None,
) -> tuple[Array, Array]:
    """Collapsed Gauss-Legendre rule on the reference cell, exactness-derived.

    Each axis uses the smallest count integrating the pairing of two basis
    functions of ``polynomial_degree`` times the geometry measure (see
    ``_reference_rule_axis_counts``); ``_reference_rule_exactness`` states the
    total degree integrated exactly.
    """
    kind, dimension = _reference_rule_layout(cell_kind)
    counts = _reference_rule_axis_counts(cell_kind, polynomial_degree, geometry_degree)
    axes = []
    axis_weights = []
    for count in counts:
        nodes, weights = np.polynomial.legendre.leggauss(count)
        axes.append(0.5 * (nodes + 1.0))
        axis_weights.append(0.5 * weights)
    grids = np.meshgrid(*axes, indexing="ij")
    weight_grids = np.meshgrid(*axis_weights, indexing="ij")
    points = np.stack(grids, axis=-1)
    combined = np.prod(np.stack(weight_grids, axis=-1), axis=-1)
    if kind == "simplex":
        scale = np.ones_like(grids[0])
        for coordinate in range(dimension):
            points[..., coordinate] = scale * grids[coordinate]
            combined *= (1.0 - grids[coordinate]) ** (dimension - coordinate - 1)
            scale *= 1.0 - grids[coordinate]
    elif kind == "prism":
        points[..., 1] = (1.0 - grids[0]) * grids[1]
        combined *= 1.0 - grids[0]
    elif kind == "pyramid":
        scale = 1.0 - grids[2]
        points[..., 0] = scale * grids[0] + 0.5 * grids[2]
        points[..., 1] = scale * grids[1] + 0.5 * grids[2]
        combined *= scale**2
    return jnp.asarray(points.reshape((-1, dimension))), jnp.asarray(
        combined.reshape((-1,))
    )


# Output elements of one row chunk of mapped basis gradients: forward-mode
# gradients materialize per-row products of the full local basis, so rows are
# processed in chunks whose output stays within this many elements.
_MAPPED_GRADIENT_ELEMENT_BUDGET = 1 << 19


def _row_chunked_map(function: Callable[..., Array], *rows: Array) -> Array:
    """Apply a row-vectorized ``function`` in chunks bounded by output size."""
    row_count = rows[0].shape[0]
    per_row = prod(jax.eval_shape(function, *(row[:1] for row in rows)).shape[1:])
    chunk = max(1, _MAPPED_GRADIENT_ELEMENT_BUDGET // max(per_row, 1))
    parts = [
        function(*(row[start : start + chunk] for row in rows))
        for start in range(0, row_count, chunk)
    ]
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)


def _evaluate_coordinate_map(
    coordinate_element: ScalarCoordinateElement,
    coordinate_routes: ArrayLike,
    coordinates: ArrayLike,
    reference_points: ArrayLike,
    /,
    *,
    precision_policy: FiniteElementPrecisionPolicy,
    paired: bool,
    cell_rows: ArrayLike | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Evaluate a shared or exact cellwise coordinate map."""

    points = precision_policy.geometry(reference_points)
    coordinate_values = precision_policy.geometry(coordinates)
    routes = jnp.asarray(coordinate_routes)
    cell_coordinates = coordinate_values[routes]
    if isinstance(coordinate_element, PerCellRestrictedCellGeometryElement):
        if paired:
            raise ValueError(
                "Per-cell coordinate maps require a prepared cell-row workset."
            )
        if cell_rows is None:
            raise ValueError(
                "Per-cell coordinate tabulation requires explicit block rows."
            )
        rows = np.asarray(cell_rows)
        if (
            rows.ndim != 1
            or rows.dtype.kind not in "iu"
            or rows.shape[0] != cell_coordinates.shape[0]
        ):
            raise ValueError("Per-cell coordinate rows must align with the cell workset.")
        tabulations = tuple(
            coordinate_tabulation(coordinate_element, points, cell_row=row)
            for row in rows.tolist()
        )
        geometry_values = precision_policy.geometry(
            jnp.stack(tuple(value for value, _ in tabulations), axis=0)
        )
        geometry_gradients = precision_policy.geometry(
            jnp.stack(tuple(gradient for _, gradient in tabulations), axis=0)
        )
    else:
        if cell_rows is not None:
            raise ValueError(
                "Shared coordinate charts do not accept per-cell row indices."
            )
        values, gradients = coordinate_element.tabulate(points)
        geometry_values = precision_policy.geometry(values)
        geometry_gradients = precision_policy.geometry(gradients)
    if paired:
        if cell_coordinates.shape[0] != points.shape[0]:
            raise ValueError(
                "Paired cell-map evaluation requires one reference point per cell index."
            )
        return _coordinate_map_values(
            ein.contract("qi,qid->qd", geometry_values, cell_coordinates),
            ein.contract("qir,qid->qdr", geometry_gradients, cell_coordinates),
        )
    if isinstance(coordinate_element, PerCellRestrictedCellGeometryElement):
        return _coordinate_map_values(
            ein.contract("cqi,cid->cqd", geometry_values, cell_coordinates),
            ein.contract("cqir,cid->cqdr", geometry_gradients, cell_coordinates),
        )

    # One shared chart: evaluate cell by cell so the Jacobian inversion's
    # intermediates stay one cell's quadrature wide; only the outputs span the
    # block. The scanned body is one module-level function, so executing it op
    # by op reuses one compiled loop instead of compiling a fresh closure.
    _, values = jax.lax.scan(
        _shared_chart_cell_map, (geometry_values, geometry_gradients), cell_coordinates
    )
    return values


def _shared_chart_cell_map(
    charts: tuple[Array, Array], nodes: Array, /
) -> tuple[
    tuple[Array, Array], tuple[Array, Array, Array, Array, Array, Array, Array, Array]
]:
    geometry_values, geometry_gradients = charts
    return charts, _coordinate_map_values(
        ein.contract("qi,id->qd", geometry_values, nodes),
        ein.contract("qir,id->qdr", geometry_gradients, nodes),
    )


def _shared_chart_identity_cell(
    charts: tuple[Array, Array, Array], nodes: Array, /
) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array]]:
    geometry_values, geometry_gradients, reference_gradients = charts
    points, _, _, _, inverse_jacobian, _, measure, _ = _coordinate_map_values(
        ein.contract("qi,id->qd", geometry_values, nodes),
        ein.contract("qir,id->qdr", geometry_gradients, nodes),
    )
    # Physical gradients are J^+T grad = J (J^T J)^-1 grad, formed from the
    # (pseudo-)inverse Jacobian rather than J times the inverse metric: the
    # metric is conditioned like r^-2, the Jacobian like r^-1.
    gradients = ein.contract("qid,qki->qkd", inverse_jacobian, reference_gradients)
    return charts, (points, gradients, measure)


def _shared_chart_identity_geometry(
    coordinate_element: ScalarCoordinateElement,
    coordinate_routes: ArrayLike,
    coordinates: ArrayLike,
    reference_points: Array,
    reference_gradients: Array,
    /,
    *,
    precision_policy: FiniteElementPrecisionPolicy,
) -> tuple[Array, Array, Array]:
    """Points, identity-mapped gradients and measure of one shared-chart block.

    Only these retained outputs span the block; the Jacobian, its inverse and
    the metric stay one cell's quadrature wide inside the scanned body.
    """
    values, gradients = coordinate_element.tabulate(reference_points)
    cell_coordinates = precision_policy.geometry(coordinates)[
        jnp.asarray(coordinate_routes)
    ]
    _, outputs = jax.lax.scan(
        _shared_chart_identity_cell,
        (
            precision_policy.geometry(values),
            precision_policy.geometry(gradients),
            reference_gradients,
        ),
        cell_coordinates,
    )
    return outputs


def _coordinate_map_values(
    physical_points: Array, jacobian: Array, /
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    metric = ein.contract("...di,...dj->...ij", jacobian, jacobian)
    # Full-dimensional maps evaluate det(J) itself (its sign survives binary64
    # down to a few u relative to the column-norm product, where the Gram
    # determinant's is lost below sqrt(u)); embedded maps keep the Gram
    # determinant. Both solve column-normalized Jacobians.
    inverse_jacobian, inverse_metric, measure, determinant, successful = (
        cell_jacobian_inverse_and_measure(jacobian)
    )
    inverse_metric = jnp.where(successful[..., None, None], inverse_metric, 0.0)
    inverse_jacobian = jnp.where(successful[..., None, None], inverse_jacobian, 0.0)
    measure = jnp.where(successful, measure, 0.0)
    determinant = jnp.where(successful, determinant, 0.0)
    gram_determinant = measure * measure
    return (
        physical_points,
        jacobian,
        metric,
        inverse_metric,
        inverse_jacobian,
        gram_determinant,
        measure,
        determinant,
    )


def _prepare_block_geometry(
    mesh: CellMesh,
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    /,
    *,
    coordinates: ArrayLike | None = None,
    reference_points: ArrayLike | None = None,
    reference_weights: ArrayLike | None = None,
    coordinate_element: CellGeometryElement | None = None,
    geometry_dofs: ArrayLike | None = None,
    precision_policy: FiniteElementPrecisionPolicy,
    coordinate_measure: CoordinateMeasurePlan | None = None,
) -> FiniteElementBlockGeometry:
    if (reference_points is None) != (reference_weights is None):
        raise ValueError(
            "reference_points and reference_weights must be supplied together."
        )
    if not isinstance(block, CellBlock):
        raise TypeError("Finite-element block geometry requires a fixed-cell block.")
    if coordinate_element is not None and not isinstance(
        coordinate_element, ScalarCoordinateElement
    ):
        raise TypeError("Finite-element coordinate geometry requires a scalar chart.")
    geometry_element = (
        _linear_reference_element(block.cell_kind)
        if coordinate_element is None
        else coordinate_element
    )
    if reference_points is None:
        points_, weights_ = _degree_aware_reference_rule(
            block.cell_kind,
            element.degree
            + (
                1
                if coordinate_measure is not None
                and coordinate_measure.kind == "spherical-radial"
                else 0
            ),
            geometry_degree=geometry_element.degree,
        )
    else:
        points_ = jnp.asarray(reference_points)
        weights_ = jnp.asarray(reference_weights)
    reference_points = precision_policy.geometry(points_)
    reference_weights = precision_policy.accumulation(weights_)
    basis_values, reference_gradients = element.tabulate(reference_points)
    basis_values = precision_policy.evaluation(basis_values)
    reference_gradients = precision_policy.evaluation(reference_gradients)
    identity_mapped = (
        element.mapping == "identity"
        and element.value_spec.form_type.twist == "untwisted"
    )
    routes = block.vertices if geometry_dofs is None else jnp.asarray(geometry_dofs)
    coordinate_values = mesh.coordinates if coordinates is None else coordinates
    if identity_mapped and not isinstance(
        geometry_element, PerCellRestrictedCellGeometryElement
    ):
        physical_points, physical_gradients, measure_factor = (
            _shared_chart_identity_geometry(
                geometry_element,
                routes,
                coordinate_values,
                reference_points,
                reference_gradients,
                precision_policy=precision_policy,
            )
        )
    else:
        (
            physical_points,
            jacobian,
            _,
            _,
            inverse_jacobian,
            _,
            measure_factor,
            _,
        ) = _evaluate_coordinate_map(
            geometry_element,
            routes,
            coordinate_values,
            reference_points,
            precision_policy=precision_policy,
            paired=False,
            cell_rows=(
                np.arange(block.cell_count, dtype=np.int32)
                if isinstance(geometry_element, PerCellRestrictedCellGeometryElement)
                else None
            ),
        )
        if identity_mapped:
            # Physical gradients are J^+T grad = J (J^T J)^-1 grad, formed from
            # the (pseudo-)inverse Jacobian rather than J times the inverse
            # metric: the metric is conditioned like r^-2, the Jacobian like r^-1.
            physical_gradients = ein.contract(
                "cqid,qki->cqkd", inverse_jacobian, reference_gradients
            )
        else:
            physical_basis = map_reference_values(
                basis_values[None], element.value_spec, jacobian[:, :, None]
            )
    measure_factor = eqx.error_if(
        measure_factor,
        jnp.any(~jnp.isfinite(measure_factor) | (measure_factor <= 0.0)),
        "Finite-element geometry requires positive finite metric determinant.",
    )
    if identity_mapped:
        physical_basis = basis_values
    else:
        routes_ = block.vertices if geometry_dofs is None else jnp.asarray(geometry_dofs)
        coordinates_ = (
            mesh.coordinates if coordinates is None else jnp.asarray(coordinates)
        )
        if isinstance(geometry_element, PerCellRestrictedCellGeometryElement):
            mapped_gradients = []
            for cell_row in range(block.cell_count):
                cell_coordinates = coordinates_[routes_[cell_row]]

                def mapped_reference(reference: Array, /) -> Array:
                    _, geometry_gradients = coordinate_tabulation(
                        geometry_element, reference[None], cell_row=cell_row
                    )
                    jacobian_ = cell_coordinates.T @ geometry_gradients[0]
                    values_ = element.tabulate(reference[None])[0][0]
                    return map_reference_values(values_, element.value_spec, jacobian_)

                mapped_gradients.append(
                    _row_chunked_map(
                        jax.vmap(jax.jacfwd(mapped_reference)), reference_points
                    )
                )
            mapped_gradient = jnp.stack(mapped_gradients, axis=0)
        else:

            def mapped_reference(reference: Array, cell_coordinates: Array, /) -> Array:
                gradients = geometry_element.tabulate(reference[None])[1][0]
                jacobian_ = cell_coordinates.T @ gradients
                values_ = element.tabulate(reference[None])[0][0]
                return map_reference_values(values_, element.value_spec, jacobian_)

            # Forward-mode gradients of every basis function at every (cell,
            # point) row are formed for a bounded number of rows at a time.
            cell_coordinates = coordinates_[routes_]
            point_count = reference_points.shape[0]
            row_cells = jnp.repeat(
                jnp.arange(cell_coordinates.shape[0], dtype=jnp.int32), point_count
            )
            row_points = jnp.tile(
                jnp.arange(point_count, dtype=jnp.int32), cell_coordinates.shape[0]
            )
            mapped = jax.vmap(jax.jacfwd(mapped_reference, argnums=0))

            def mapped_rows(cells_: Array, points_: Array, /) -> Array:
                return mapped(reference_points[points_], cell_coordinates[cells_])

            mapped_gradient = _row_chunked_map(mapped_rows, row_cells, row_points)
            mapped_gradient = mapped_gradient.reshape(
                (cell_coordinates.shape[0], point_count, *mapped_gradient.shape[1:])
            )
        physical_gradients = mapped_gradient
    physical_weights = precision_policy.accumulation(
        measure_factor * reference_weights[None, :]
    )
    if coordinate_measure is not None:
        physical_weights = coordinate_measure.weight(
            physical_points, physical_weights, frame=coordinate_measure.frame
        )
    if coordinate_measure is not None and coordinate_measure.radial_axis is not None:
        coordinate_values = jnp.asarray(
            mesh.coordinates if coordinates is None else coordinates
        )
        routes = block.vertices if geometry_dofs is None else jnp.asarray(geometry_dofs)
        nodes = coordinate_values[routes]
        valid = jnp.all(jnp.isfinite(nodes), axis=(-2, -1)) & jnp.all(
            nodes[..., coordinate_measure.radial_axis] >= 0.0, axis=-1
        )
        physical_weights = jnp.where(valid[:, None], physical_weights, jnp.nan)
    return FiniteElementBlockGeometry(
        block_name=block.name,
        reference_points=reference_points,
        reference_weights=reference_weights,
        basis_values=physical_basis,
        reference_gradients=reference_gradients,
        physical_points=physical_points,
        physical_gradients=physical_gradients,
        physical_weights=physical_weights,
        measure=precision_policy.output(jnp.sum(physical_weights, axis=1)),
        coordinate_measure=coordinate_measure,
    )


def _assemble_local_operator(
    dof_map: FiniteElementDofMap,
    local_values: Sequence[Array],
    kind: str,
    /,
    *,
    positive_definite: bool,
    component_shape: Sequence[int] = (),
    properties: OperatorProperties | None = None,
    coefficient_dtype: DTypeLike = jnp.float64,
) -> SparseLinearMap:
    source_parts = []
    target_parts = []
    coefficient_parts = []
    component_count = prod(tuple(component_shape)) if component_shape else 1
    for cell_dofs, values in zip(dof_map.cell_dofs, local_values, strict=True):
        block_index = len(source_parts)
        transform = dof_map.cell_transforms[block_index]
        values = ein.contract("cai,cab,cbj->cij", transform, values, transform)
        indices = jnp.asarray(cell_dofs, dtype=jnp.int32)
        width = indices.shape[1]
        components = jnp.arange(component_count, dtype=jnp.int32)
        flat = indices[..., None] * component_count + components
        source_parts.append(
            jnp.broadcast_to(
                flat[:, None, :, :],
                (indices.shape[0], width, width, component_count),
            ).reshape((-1,))
        )
        target_parts.append(
            jnp.broadcast_to(
                flat[:, :, None, :],
                (indices.shape[0], width, width, component_count),
            ).reshape((-1,))
        )
        coefficient_parts.append(
            jnp.broadcast_to(
                jnp.asarray(values)[..., None],
                values.shape + (component_count,),
            ).reshape((-1,))
        )
    relation = EdgeRelation(
        jnp.concatenate(source_parts),
        jnp.concatenate(target_parts),
        source_size=dof_map.global_dof_count * component_count,
        target_size=dof_map.global_dof_count * component_count,
    )
    properties_ = (
        OperatorProperties(
            self_adjoint=True,
            positive_definite=positive_definite,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
                **({"positive_definite": "construction"} if positive_definite else {}),
            },
        )
        if properties is None
        else properties
    )
    if not isinstance(properties_, OperatorProperties):
        raise TypeError("properties must be OperatorProperties or None.")
    return SparseLinearMap(
        relation,
        jnp.concatenate(tuple(coefficient_parts)).astype(coefficient_dtype),
        properties=properties_,
        operator_id=canonical_fingerprint(
            {
                "kind": kind,
                "dof_map": dof_map.dof_map_id,
                "coefficient_dtype": np.dtype(coefficient_dtype).str,
            }
        ),
    )


@final
class MaskedFiniteElementPlan(StrictModule, NonTrainableState):
    """Static Lagrange finite-element plan for one `MaskedSimplexMesh` bucket.

    Every field is static: cell kind, ambient dimension, vertex and cell
    capacities, the layout ``signature_id``, family, degree, and precision policy.
    ``plan_id`` fingerprints the family, degree, layout signature, and precision
    policy only, so `assemble_masked_finite_element` compiles once per capacity
    bucket and serves every layout of that signature regardless of its active
    counts or topology values.

    Masked capacity layouts carry vertex DOFs only: DOF ``i`` is vertex slot
    ``i``, so the only admissible Lagrange degree is 1. Higher degrees need edge,
    face, and cell DOF slots the layout does not allocate and raise
    ``ValueError``.
    """

    cell_kind: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    vertex_capacity: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)
    family: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    precision_policy: FiniteElementPrecisionPolicy
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: MaskedSimplexMesh,
        /,
        *,
        degree: int = 1,
        precision_policy: FiniteElementPrecisionPolicy | None = None,
    ) -> None:
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
            raise TypeError("degree must be an integer.")
        if degree != 1:
            raise ValueError(
                "Masked capacity layouts carry vertex DOFs only; the Lagrange "
                "degree must be 1."
            )
        precision = (
            FiniteElementPrecisionPolicy()
            if precision_policy is None
            else precision_policy
        )
        if not isinstance(precision, FiniteElementPrecisionPolicy):
            raise TypeError(
                "precision_policy must be FiniteElementPrecisionPolicy or None."
            )
        self.cell_kind = mesh.cell_kind
        self.ambient_dimension = mesh.ambient_dimension
        self.vertex_capacity = mesh.vertex_capacity
        self.cell_capacity = mesh.cell_capacity
        self.signature_id = mesh.signature_id
        self.family = "lagrange"
        self.degree = 1
        self.precision_policy = precision
        self.plan_id = canonical_fingerprint(
            {
                "kind": "masked-finite-element-plan",
                "family": "lagrange",
                "degree": 1,
                "mesh_signature": mesh.signature_id,
                "precision_policy": precision.policy_id,
            }
        )


@final
class MaskedFiniteElementSystem(StrictModule, NonTrainableState):
    """Capacity-shaped P1 mass and stiffness operators of one masked layout.

    Both operators act on ``(vertex_capacity,)`` vectors and share one
    `EdgeRelation`: ``cell_capacity * (d + 1) ** 2`` cell routes valid only on
    active cells, then ``vertex_capacity`` diagonal routes valid only on inactive
    vertex slots with coefficient 1. Inactive DOFs are therefore pinned through
    identity rows: each operator stays square, is the identity on padding, and
    never couples padding to active DOFs, so solve, transpose, and adjoint
    actions keep the capacity shape and map zero padding to zero padding.
    ``mass`` is symmetric positive definite; ``stiffness`` is symmetric positive
    semidefinite with the constants of each active component as its kernel.
    ``boundary_dofs`` marks active vertices lying on a boundary facet. Operator
    IDs derive from ``plan_id`` alone, never from topology values.
    """

    plan_id: str = eqx.field(static=True)
    mass: SparseLinearMap
    stiffness: SparseLinearMap
    dof_active: Array
    boundary_dofs: Array

    def __init__(
        self,
        plan_id: str,
        mass: SparseLinearMap,
        stiffness: SparseLinearMap,
        dof_active: Array,
        boundary_dofs: Array,
        /,
    ) -> None:
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("plan_id must be a non-empty string.")
        if not isinstance(mass, SparseLinearMap) or not isinstance(
            stiffness, SparseLinearMap
        ):
            raise TypeError("mass and stiffness must be SparseLinearMap operators.")
        shape = mass.input_shape
        if (
            mass.output_shape != shape
            or stiffness.input_shape != shape
            or stiffness.output_shape != shape
        ):
            raise ValueError("Masked operators must share one square DOF space.")
        for name, mask in (("dof_active", dof_active), ("boundary_dofs", boundary_dofs)):
            if mask.dtype != jnp.bool_ or mask.shape != shape:
                raise TypeError(f"{name} must be a boolean {shape} array.")
        self.plan_id = plan_id
        self.mass = mass
        self.stiffness = stiffness
        self.dof_active = dof_active
        self.boundary_dofs = boundary_dofs


def _masked_local_tensors(
    plan: MaskedFiniteElementPlan, mesh: MaskedSimplexMesh, /
) -> tuple[Array, Array]:
    """P1 local mass and stiffness on every cell lane, exactly zero on padding.

    Inactive lanes are mapped onto the reference simplex, appended after the
    vertex slots, so the metric inverse and the positive-measure check never see
    padding rows; their finite tensors are then replaced by exact zeros.
    """

    element = lagrange_element(plan.cell_kind, plan.degree)
    policy = plan.precision_policy
    width = mesh.dimension + 1
    reference_vertices = jnp.pad(
        jnp.asarray(element.reference_nodes, dtype=mesh.coordinates.dtype),
        ((0, 0), (0, plan.ambient_dimension - mesh.dimension)),
    )
    reference_slots = jnp.arange(
        plan.vertex_capacity, plan.vertex_capacity + width, dtype=jnp.int32
    )
    routes = jnp.where(mesh.cell_active[:, None], mesh.cells, reference_slots[None, :])
    points_, weights_ = _degree_aware_reference_rule(plan.cell_kind, plan.degree)
    reference_points = policy.geometry(points_)
    reference_weights = policy.accumulation(weights_)
    physical_points, _, _, _, inverse_jacobian, _, measure_factor, _ = (
        _evaluate_coordinate_map(
            element,
            routes,
            jnp.concatenate((mesh.coordinates, reference_vertices)),
            reference_points,
            precision_policy=policy,
            paired=False,
        )
    )
    measure_factor = eqx.error_if(
        measure_factor,
        jnp.any(~jnp.isfinite(measure_factor) | (measure_factor <= 0.0)),
        "Masked finite-element geometry requires positive finite metric "
        "determinants on active cells.",
    )
    basis_values, reference_gradients = element.tabulate(reference_points)
    reference_gradients = policy.evaluation(reference_gradients)
    physical_weights = policy.accumulation(measure_factor * reference_weights[None, :])
    geometry = FiniteElementBlockGeometry(
        block_name=plan.cell_kind,
        reference_points=reference_points,
        reference_weights=reference_weights,
        basis_values=policy.evaluation(basis_values),
        reference_gradients=reference_gradients,
        physical_points=physical_points,
        physical_gradients=ein.contract(
            "cqid,qki->cqkd", inverse_jacobian, reference_gradients
        ),
        physical_weights=physical_weights,
        measure=policy.output(jnp.sum(physical_weights, axis=1)),
    )
    active = mesh.cell_active[:, None, None]
    zero = jnp.zeros((), dtype=physical_weights.dtype)
    return (
        jnp.where(active, _local_mass_tensor(geometry), zero),
        jnp.where(active, _local_stiffness_tensor(geometry), zero),
    )


def _masked_boundary_dofs(mesh: MaskedSimplexMesh, /) -> Array:
    """Active vertex slots on a boundary facet.

    Local vertex ``j`` lies on the facet opposite local vertex ``i`` iff
    ``i != j``; per-cell flags reduce onto vertex slots over the active
    cell-vertex incidence relation.
    """

    width = mesh.dimension + 1
    opposite = jnp.asarray(1 - np.eye(width, dtype=np.int32))
    on_boundary = mesh.boundary_facets.astype(jnp.int32) @ opposite
    incidence = EdgeRelation(
        jnp.arange(mesh.cell_capacity * width, dtype=jnp.int32),
        mesh.cells.reshape((-1,)),
        source_size=mesh.cell_capacity * width,
        target_size=mesh.vertex_capacity,
        valid=jnp.repeat(mesh.cell_active, width),
    )
    counts = route_reduce(incidence, on_boundary.reshape((-1,)))
    return (counts > 0) & mesh.vertex_active


@eqx.filter_jit
def assemble_masked_finite_element(
    plan: MaskedFiniteElementPlan, mesh: MaskedSimplexMesh, /
) -> MaskedFiniteElementSystem:
    """Assemble capacity-shaped P1 mass and stiffness on one masked layout.

    Module-level compiled entry. Its compile key is the plan's static identity
    plus the layout signature (cell kind, ambient dimension, capacities,
    coordinate dtype); coordinates, IDs, masks, cells, and adjacency are traced,
    so layouts of one bucket with different active counts or topology reuse one
    executable. See `MaskedFiniteElementSystem` for the pinned-padding layout.
    """

    if not isinstance(plan, MaskedFiniteElementPlan):
        raise TypeError("plan must be a MaskedFiniteElementPlan.")
    if not isinstance(mesh, MaskedSimplexMesh):
        raise TypeError("mesh must be a MaskedSimplexMesh.")
    if mesh.signature_id != plan.signature_id:
        raise ValueError("mesh does not belong to the plan's capacity bucket.")
    width = mesh.dimension + 1
    mass_local, stiffness_local = _masked_local_tensors(plan, mesh)
    columns = jnp.broadcast_to(mesh.cells[:, None, :], (plan.cell_capacity, width, width))
    slots = jnp.arange(plan.vertex_capacity, dtype=jnp.int32)
    relation = EdgeRelation(
        jnp.concatenate((columns.reshape((-1,)), slots)),
        jnp.concatenate((jnp.swapaxes(columns, 1, 2).reshape((-1,)), slots)),
        source_size=plan.vertex_capacity,
        target_size=plan.vertex_capacity,
        valid=jnp.concatenate(
            (jnp.repeat(mesh.cell_active, width * width), ~mesh.vertex_active)
        ),
    )
    pins = jnp.ones((plan.vertex_capacity,), dtype=mass_local.dtype)
    mass = SparseLinearMap(
        relation,
        jnp.concatenate((mass_local.reshape((-1,)), pins)),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=canonical_fingerprint(
            {"kind": "masked-finite-element-mass", "plan": plan.plan_id}
        ),
    )
    stiffness = SparseLinearMap(
        relation,
        jnp.concatenate((stiffness_local.reshape((-1,)), pins)),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=canonical_fingerprint(
            {"kind": "masked-finite-element-stiffness", "plan": plan.plan_id}
        ),
    )
    return MaskedFiniteElementSystem(
        plan.plan_id,
        mass,
        stiffness,
        mesh.vertex_active,
        _masked_boundary_dofs(mesh),
    )


def constrain_masked_dofs(
    operator: SparseLinearMap, pinned: ArrayLike, /
) -> SparseLinearMap:
    """Pin DOFs of a square edge-relation operator through identity rows.

    Returns ``P_free A P_free + P_pinned``: routes touching a pinned source or
    target are invalidated and one appended diagonal route per DOF, valid only
    where ``pinned``, carries 1. The route capacity is static (operator routes
    plus one per DOF), so a traced ``pinned`` mask never changes the compiled
    shape. With ``pinned = system.boundary_dofs`` a masked stiffness becomes the
    homogeneous Dirichlet operator; padding stays pinned whether or not it is
    included in ``pinned``. Self-adjointness and (semi)definiteness carry over
    as transformed evidence; the operator ID derives from the input operator ID.
    """

    if not isinstance(operator, SparseLinearMap) or not isinstance(
        operator.relation, EdgeRelation
    ):
        raise TypeError("operator must be a SparseLinearMap over an EdgeRelation.")
    relation = operator.relation
    if operator.batch_shape or relation.source_size != relation.target_size:
        raise ValueError("constrain_masked_dofs requires an unbatched square operator.")
    mask = jnp.asarray(pinned)
    if mask.dtype != jnp.bool_:
        raise TypeError("pinned must be a boolean array.")
    if mask.shape != (relation.source_size,):
        raise ValueError("pinned must hold one flag per DOF.")
    slots = jnp.arange(relation.source_size, dtype=relation.source_indices.dtype)
    free = (
        relation.valid
        & ~gather_routes(relation, mask)
        & ~gather_routes(relation.transpose(), mask)
    )
    constrained = EdgeRelation(
        jnp.concatenate((relation.source_indices, slots)),
        jnp.concatenate((relation.target_indices, slots)),
        source_size=relation.source_size,
        target_size=relation.target_size,
        valid=jnp.concatenate((free, mask)),
    )
    source = operator.properties
    claims = {
        "self_adjoint": source.self_adjoint,
        "positive_definite": source.positive_definite,
        "positive_semidefinite": source.positive_semidefinite,
    }
    return SparseLinearMap(
        constrained,
        jnp.concatenate(
            (
                operator.coefficients,
                jnp.ones((relation.source_size,), dtype=operator.coefficients.dtype),
            )
        ),
        properties=OperatorProperties(
            **claims,
            evidence={
                name: "transformed"
                for name, claimed in claims.items()
                if claimed and source.evidence_for(name) != "unknown"
            },
        ),
        operator_id=canonical_fingerprint(
            {"kind": "masked-dof-constraint", "operator": operator.operator_id}
        ),
    )


__all__ = [
    "FiniteElementDiscretization",
    "FiniteElementDofMap",
    "FiniteElementFieldSpec",
    "FiniteElementPlan",
    "FiniteElementRuntimeData",
    "IntegrationDomain",
    "MaskedFiniteElementPlan",
    "MaskedFiniteElementSystem",
    "assemble_masked_finite_element",
    "constrain_masked_dofs",
]
