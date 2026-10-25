from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import prod
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike, DTypeLike

import phydrax.ein as ein
from phydrax.discretization._reference_cell_types import (
    CellBlock,
    CellGeometryElement,
    PolyhedralBlock,
)
from phydrax.discretization.fem._reference_types import FiniteElementSpec

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    inverse_small_linear,
    OperatorProperties,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ...sparse import EdgeRelation, RowRelation, SparseLinearMap
from ...typing import checked
from .. import _cell_geometry as cell_geometry_owner, _cell_mesh as cell_mesh_owner
from .._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from .._coordinate_budget import CoordinateEnclosureBudget
from .._hexahedral import (
    HEXAHEDRAL_EDGES as _HEXAHEDRAL_EDGES,
    HEXAHEDRAL_FACES as _HEXAHEDRAL_FACES,
    HexahedralConnectivity,
    quadrilateral_tensor_permutation,
)
from .._periodic_topology import lifted_loops, PeriodicMeshTopology
from .._reference_cell import reference_cell_topology
from . import _form_elements as form_elements_owner, _nodal_orbits as nodal_orbits_owner
from ._form_basis_types import AbstractReferenceFormBasis, EntityMomentCache
from ._precision import FiniteElementPrecisionPolicy
from ._reference import lagrange_element


_HEXAHEDRAL_EDGE_BY_VERTICES = {
    frozenset(edge): index for index, edge in enumerate(_HEXAHEDRAL_EDGES)
}

_TETRAHEDRAL_TOPOLOGY = reference_cell_topology("tetrahedron")
_TETRAHEDRAL_EDGES = _TETRAHEDRAL_TOPOLOGY.entities[1]
_TETRAHEDRAL_FACES = _TETRAHEDRAL_TOPOLOGY.entities[2]
_HIGH_ORDER_H1_CONNECTIVITIES = (
    PolygonalConnectivity,
    TetrahedralConnectivity,
    HexahedralConnectivity,
    PolyhedralConnectivity,
)


if TYPE_CHECKING:
    from ._generic import FiniteElementBlockGeometry, FiniteElementDofMap


@dataclass(frozen=True)
class _PreparedCoordinateMaps:
    """Resolved maps bound to one preparation's immutable mesh and geometry."""

    mesh: cell_mesh_owner.CellMesh
    geometry: cell_geometry_owner.CellGeometrySpec
    elements: tuple[CellGeometryElement, ...]
    routes: tuple[Array, ...]
    coordinates: Array

    @classmethod
    def prepare(
        cls,
        mesh: cell_mesh_owner.CellMesh,
        geometry: cell_geometry_owner.CellGeometrySpec,
        *,
        budget: CoordinateEnclosureBudget | None,
    ) -> _PreparedCoordinateMaps:
        elements, routes, coordinates = geometry.resolve(mesh, budget=budget)
        return cls(mesh, geometry, elements, routes, coordinates)

    def require(
        self,
        mesh: cell_mesh_owner.CellMesh,
        geometry: cell_geometry_owner.CellGeometrySpec,
    ) -> None:
        if mesh is not self.mesh or geometry is not self.geometry:
            raise ValueError(
                "finite_element.coordinate_preparation.binding must retain the same immutable mesh and geometry; "
                f"got mesh={mesh.mesh_id!r}, geometry={geometry.geometry_layout_id!r}"
            )


def linear_reference_element(cell_kind: str, /) -> FiniteElementSpec:
    """Select the scalar coordinate chart for the declared reference cell."""
    if cell_kind.startswith(("simplex:", "tensor:")):
        family = "tensor-trimmed" if cell_kind.startswith("tensor:") else "trimmed"
        return form_elements_owner.form_element(
            cell_kind, 0, 1, family=family, twist="untwisted", proxy="scalar"
        )
    return lagrange_element(cell_kind, 1)


def tetrahedral_face_vertices(
    connectivity: TetrahedralConnectivity | PolyhedralConnectivity,
) -> npt.NDArray[np.int32]:
    if isinstance(connectivity, TetrahedralConnectivity):
        return np.asarray(connectivity.faces, dtype=np.int32)
    if isinstance(connectivity, PolyhedralConnectivity):
        offsets = np.asarray(connectivity.face_vertex_offsets)
        if np.any(np.diff(offsets) != 3):
            raise ValueError(
                "finite_element.connectivity.faces must be triangular; got non-triangular facets"
            )
        return np.asarray(connectivity.face_vertex_values, dtype=np.int32).reshape(
            (-1, 3)
        )
    raise TypeError(
        "finite_element.connectivity must provide oriented tetrahedral entities; got unsupported connectivity"
    )


def tetrahedral_entity_routes(
    connectivity: TetrahedralConnectivity | PolyhedralConnectivity,
    cells: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_by_vertices = {
        tuple(sorted(edge)): index
        for index, edge in enumerate(np.asarray(connectivity.edges))
    }
    faces = tetrahedral_face_vertices(connectivity)
    face_by_vertices = {
        tuple(sorted(int(faces[row, local]) for local in range(3))): row
        for row in range(faces.shape[0])
    }
    edge_routes = np.empty((cells.shape[0], len(_TETRAHEDRAL_EDGES)), dtype=np.int32)
    edge_signs = np.empty_like(edge_routes, dtype=np.float64)
    face_routes = np.empty((cells.shape[0], len(_TETRAHEDRAL_FACES)), dtype=np.int32)
    for cell, vertices in enumerate(cells):
        for local_edge, (start, stop) in enumerate(_TETRAHEDRAL_EDGES):
            oriented = (int(vertices[start]), int(vertices[stop]))
            canonical = tuple(sorted(oriented))
            edge_routes[cell, local_edge] = edge_by_vertices[canonical]
            edge_signs[cell, local_edge] = (
                1.0
                if oriented
                == tuple(np.asarray(connectivity.edges)[edge_routes[cell, local_edge]])
                else -1.0
            )
        for local_face, local_vertices in enumerate(_TETRAHEDRAL_FACES):
            canonical = tuple(sorted(int(vertices[index]) for index in local_vertices))
            face_routes[cell, local_face] = face_by_vertices[canonical]
    return edge_routes, edge_signs, face_routes


def tetrahedral_face_dof_positions(
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
        raise ValueError(
            "finite_element.element.entity_dofs.vertices must contain one DOF per vertex; got incompatible vertex DOFs"
        )
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
        raise ValueError(
            "finite_element.element.reference_nodes must lie on the declared tetrahedral face; got off-face DOFs"
        )
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


def has_nonvertex_dofs(element: FiniteElementSpec, /) -> bool:
    return any(
        entity
        for dimension_entities in element.entity_dofs[1:]
        for entity in dimension_entities
    )


def hexahedral_face_shape(
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
            "finite_element.element.entity_dofs.edges must have equal widths on opposite hexahedral edges; got unequal widths"
        )
    shape = (side_widths[0], side_widths[3])
    if len(element.entity_dofs[2][local_face]) != prod(shape):
        raise ValueError(
            "finite_element.element.entity_dofs.faces must match the hexahedral tensor trace shape; got incompatible face-interior DOFs"
        )
    return shape


def canonical_face_shape(
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


def hexahedral_face_grid_positions(
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
        raise ValueError(
            "finite_element.element.entity_dofs.vertices must contain one DOF per vertex; got incompatible vertex DOFs"
        )
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
        raise ValueError(
            "finite_element.element.reference_nodes must lie on the declared hexahedral face; got off-face DOFs"
        )
    parameters = np.round(parameters, decimals=12)
    levels_u = np.unique(parameters[:, 0])
    levels_v = np.unique(parameters[:, 1])
    if (len(levels_u), len(levels_v)) != shape:
        raise ValueError(
            "finite_element.element.reference_nodes.faces must match the hexahedral tensor trace shape; got incompatible coordinate levels"
        )
    positions = np.empty((len(face_dofs),), dtype=np.int32)
    for position, (parameter_u, parameter_v) in enumerate(parameters):
        index_u = int(np.argmin(np.abs(levels_u - parameter_u)))
        index_v = int(np.argmin(np.abs(levels_v - parameter_v)))
        if not np.isclose(levels_u[index_u], parameter_u) or not np.isclose(
            levels_v[index_v], parameter_v
        ):
            raise ValueError(
                "finite_element.element.reference_nodes.faces must match tensor coordinate levels; got inconsistent face coordinates"
            )
        positions[position] = index_u * shape[1] + index_v
    if np.unique(positions).size != len(face_dofs):
        raise ValueError(
            "finite_element.element.reference_nodes.faces must have unique tensor positions; got duplicate positions"
        )
    return positions


def uniform_entity_width(widths: np.ndarray, /) -> int:
    unique = np.unique(widths)
    return int(unique[0]) if unique.size == 1 and unique[0] > 0 else 1


class FiniteElementDofLayout(StrictModule):
    """Store shared DOF counts for one preparation route."""

    __strict_abstract__ = True
    conformity: str = eqx.field(static=True)
    association: str = eqx.field(static=True)
    global_count: int = eqx.field(static=True)
    entity_dof_counts: tuple[int, ...] = eqx.field(static=True)
    entity_dofs_per_entity: tuple[int, ...] = eqx.field(static=True)


class SingleAssociationDofLayout(FiniteElementDofLayout):
    """Represent a homogeneous vertex, edge, or cell DOF association."""


class FormDofLayout(FiniteElementDofLayout):
    canonical_routes: tuple[np.ndarray, ...]
    canonical_transforms: tuple[np.ndarray, ...]
    canonical_boundary: np.ndarray
    canonical_widths: tuple[np.ndarray, ...]
    canonical_bases: dict[tuple[int, int], AbstractReferenceFormBasis]


class NodalDofLayout(FiniteElementDofLayout):
    edge_widths: np.ndarray
    edge_starts: np.ndarray
    face_widths: np.ndarray
    face_starts: np.ndarray
    cell_starts: np.ndarray
    cell_widths: np.ndarray
    nodal_orbit_keys: dict[tuple[int, int], tuple[tuple[Fraction, ...], ...]]
    nodal_orbit_families: dict[tuple[int, int], str]


class QuotientDofLayout(FiniteElementDofLayout):
    quotient: QuotientDofMap


class QuotientDofMap(StrictModule):
    """Lifted entity moments mapped into periodic quotient coordinates."""

    lifted: FiniteElementDofLayout
    dofs: np.ndarray
    signs: np.ndarray
    representatives: np.ndarray
    boundary: np.ndarray
    entity_transforms: tuple[tuple[np.ndarray, np.ndarray], ...]


def topology_vertex_sets(
    mesh: cell_mesh_owner.CellMesh, /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Recover entity vertex sets from oriented incidence."""
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


def record_form_entity_widths(
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    lookups: tuple[dict[tuple[int, ...], int], ...],
    widths: list[np.ndarray],
    /,
) -> None:
    basis = element.form_basis
    if basis is None:
        raise ValueError(
            "finite_element.element.form_basis must provide routing metadata; got None"
        )
    for cell in np.asarray(block.vertices):
        for dimension, faces in enumerate(basis.entity_vertices):
            for face, dofs in zip(faces, element.entity_dofs[dimension], strict=True):
                entity = lookups[dimension][tuple(sorted(cell[list(face)]))]
                old = widths[dimension][entity]
                if old not in (0, len(dofs)):
                    raise ValueError(
                        "finite_element.element.entity_dofs.moments must agree on shared entities; got unequal moment spaces"
                    )
                widths[dimension][entity] = len(dofs)


def canonical_form_cell_orientation(
    block: CellBlock | PolyhedralBlock,
    basis: AbstractReferenceFormBasis,
    cell: np.ndarray,
    /,
) -> float:
    """Orient a local reference cell relative to its global vertex chart."""
    permutation = basis.canonical_permutation(tuple(cell.tolist()))
    sign = float(
        (-1)
        ** sum(
            permutation[i] > permutation[j]
            for i in range(len(permutation))
            for j in range(i + 1, len(permutation))
        )
    )
    if basis.family in ("tensor-trimmed", "prism-trimmed", "pyramid-trimmed"):
        chart = linear_reference_element(block.cell_kind)
        reference_vertices = np.asarray(chart.reference_nodes)
        affine = np.linalg.lstsq(
            np.c_[reference_vertices, np.ones((len(cell),), dtype=np.float64)],
            reference_vertices[list(permutation)],
            rcond=None,
        )[0]
        sign = float(np.sign(np.linalg.det(affine[:-1].T)))
    if abs(sign) != 1.0:
        raise ValueError(
            "finite_element.element.cell_orientation must be plus or minus one; got an invalid orientation sign"
        )
    return sign


def canonical_form_dof_layout(
    mesh: cell_mesh_owner.CellMesh, elements: tuple[FiniteElementSpec, ...], /
) -> FiniteElementDofLayout:
    """Prepare permutation-covariant entity moment coordinates."""
    entities = topology_vertex_sets(mesh)
    lookups = tuple(
        {vertices: index for index, vertices in enumerate(level)} for level in entities
    )
    widths = [np.zeros((len(level),), dtype=np.int32) for level in entities]
    for block, element in zip(mesh.blocks, elements, strict=True):
        record_form_entity_widths(block, element, lookups, widths)
    counts = tuple(int(np.sum(level)) for level in widths)
    offsets = np.cumsum((0, *counts), dtype=np.int64)
    starts = tuple(
        offsets[degree] + np.cumsum(np.r_[0, level[:-1]], dtype=np.int64)
        for degree, level in enumerate(widths)
    )
    boundary = np.zeros((int(offsets[-1]),), dtype=np.bool_)
    physical = _physical_boundary_entity_masks(mesh)
    for degree, level in enumerate(widths):
        mask = physical[degree]
        for entity in np.flatnonzero(mask):
            start = starts[degree][entity]
            boundary[start : start + level[entity]] = True
    bases = canonical_form_entity_bases(mesh, elements, lookups)
    routes, transforms = canonical_form_block_routes(
        mesh, elements, lookups, starts, bases
    )
    return FormDofLayout(
        conformity=elements[0].conformity,
        association="form_entity",
        global_count=int(offsets[-1]),
        entity_dof_counts=counts,
        entity_dofs_per_entity=tuple(uniform_entity_width(level) for level in widths),
        canonical_routes=routes,
        canonical_transforms=transforms,
        canonical_boundary=boundary,
        canonical_widths=tuple(widths),
        canonical_bases={
            (degree, entity): basis
            for degree, level in enumerate(bases)
            for entity, basis in enumerate(level)
            if basis is not None
        },
    )


def canonical_form_block_routes(
    mesh: cell_mesh_owner.CellMesh,
    elements: tuple[FiniteElementSpec, ...],
    lookups: tuple[dict[tuple[int, ...], int], ...],
    starts: tuple[np.ndarray, ...],
    entity_bases: tuple[tuple[AbstractReferenceFormBasis | None, ...], ...],
    /,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    routes, transforms = [], []
    for block, element in zip(mesh.blocks, elements, strict=True):
        basis = element.form_basis
        if basis is None:
            raise ValueError(
                "finite_element.element.form_basis must provide routing metadata; got None"
            )
        local = np.empty((block.cell_count, element.local_dof_count), dtype=np.int32)
        matrices = np.empty(
            (block.cell_count, element.local_dof_count, element.local_dof_count),
            dtype=np.float64,
        )
        # Entity moment charts depend on relative vertex ordering, not IDs.
        # Keep the cache scoped to this block's basis and preparation.
        permutations: dict[
            tuple[tuple[int, ...], tuple[str | None, ...]], np.ndarray
        ] = {}
        orientations: dict[tuple[int, ...], float] = {}
        entity_moments = EntityMomentCache()
        for row, cell in enumerate(np.asarray(block.vertices)):
            local_bases = tuple(
                tuple(
                    entity_bases[dimension][
                        lookups[dimension][tuple(sorted(cell[list(face)]))]
                    ]
                    for face in faces
                )
                for dimension, faces in enumerate(basis.entity_vertices)
            )
            relative_order = tuple(np.argsort(np.argsort(cell)).tolist())
            key = (
                relative_order,
                tuple(
                    None if value is None else value.basis_id
                    for level in local_bases
                    for value in level
                ),
            )
            canonical = permutations.get(key)
            if canonical is None:
                canonical = np.asarray(
                    basis.entity_permutation_matrix(
                        tuple(cell.tolist()),
                        entity_bases=local_bases,
                        cache=entity_moments,
                    )
                )
                permutations[key] = canonical
            matrices[row] = canonical
            needs_cell_orientation = basis.form_degree == mesh.topological_dimension
            cell_orientation = 1.0
            if needs_cell_orientation:
                cell_orientation = orientations.get(relative_order, 0.0)
                if cell_orientation == 0.0:
                    cell_orientation = canonical_form_cell_orientation(block, basis, cell)
                    orientations[relative_order] = cell_orientation
            if basis.form_degree == mesh.topological_dimension:
                matrices[row] *= cell_orientation
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


def canonical_form_entity_bases(
    mesh: cell_mesh_owner.CellMesh,
    elements: tuple[FiniteElementSpec, ...],
    lookups: tuple[dict[tuple[int, ...], int], ...],
    /,
) -> tuple[tuple[AbstractReferenceFormBasis | None, ...], ...]:
    """Reuse immutable local-face traces and match their entity owners."""

    result: list[list[AbstractReferenceFormBasis | None]] = [
        [None] * len(level) for level in lookups
    ]
    trace_cache: dict[tuple[str, int, int], AbstractReferenceFormBasis] = {}
    for block, element in zip(mesh.blocks, elements, strict=True):
        basis = element.form_basis
        if basis is None:
            raise ValueError(
                "finite_element.element.form_basis must provide trace metadata; got None"
            )
        for dimension, faces in enumerate(basis.entity_vertices):
            if dimension == 0 or dimension == mesh.topological_dimension:
                continue
            if not any(element.entity_dofs[dimension]):
                continue
            traces: list[tuple[tuple[int, ...], AbstractReferenceFormBasis]] = []
            if block.cell_count:
                for face, dofs in zip(faces, element.entity_dofs[dimension], strict=True):
                    if not dofs:
                        continue
                    key = (basis.entity_kind(face), basis.form_degree, basis.order)
                    trace = trace_cache.get(key)
                    if trace is None:
                        trace = basis.entity_basis(face)
                        trace_cache[key] = trace
                    traces.append((face, trace))
            for cell in np.asarray(block.vertices):
                for face, trace in traces:
                    entity = lookups[dimension][tuple(sorted(cell[list(face)]))]
                    previous = result[dimension][entity]
                    if previous is None or trace.basis_id < previous.basis_id:
                        result[dimension][entity] = trace
    return tuple(tuple(level) for level in result)


@checked
def lifted_finite_element_dof_layout(
    mesh: cell_mesh_owner.CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    components: tuple[int, ...],
    /,
) -> FiniteElementDofLayout:
    if all(element.form_basis is not None for element in resolved):
        if components:
            raise ValueError(
                "finite_element.field.component_shape must be empty for form elements; got replicated coefficient fibers"
            )
        return canonical_form_dof_layout(mesh, resolved)
    conformities = {element.conformity for element in resolved}
    if len(conformities) != 1:
        raise ValueError(
            "finite_element.field.elements.conformity must agree across cell blocks; got mixed conformities"
        )
    conformity = conformities.pop()
    match conformity:
        case "L2":
            return single_association_dof_layout(
                mesh,
                conformity,
                "cell",
                sum(
                    block.cell_count * element.local_dof_count
                    for block, element in zip(mesh.blocks, resolved, strict=True)
                ),
            )
        case "Hdiv" | "Hcurl" | "HLambda":
            raise ValueError(
                "finite_element.field.elements.form_basis must provide compatible moment metadata; got an unsupported field"
            )
        case "H1":
            if not any(has_nonvertex_dofs(element) for element in resolved):
                return single_association_dof_layout(
                    mesh, conformity, "vertex", mesh.coordinates.shape[0]
                )
            return high_order_h1_dof_layout(mesh, resolved)
        case _:
            raise ValueError(
                f"finite_element.field.elements.conformity must select a supported conformity; got {conformity!r}"
            )


def entity_dof_ranges(
    starts: np.ndarray, widths: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return global indices, entity indices, and entity-interior DOF positions."""

    entity = np.repeat(np.arange(widths.shape[0], dtype=np.int64), widths)
    first = np.cumsum(widths, dtype=np.int64) - widths
    position = np.arange(entity.shape[0], dtype=np.int64) - first[entity]
    return starts.astype(np.int64)[entity] + position, entity, position


def lifted_dof_entities(
    mesh: cell_mesh_owner.CellMesh, layout: FiniteElementDofLayout, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Entity degree, entity index and position of every lifted DOF.

    Also returns, per degree, the DOF width of every lifted entity.
    """

    dimension = mesh.topological_dimension
    count = layout.global_count
    degree = np.full((count,), -1, dtype=np.int64)
    entity = np.zeros((count,), dtype=np.int64)
    position = np.zeros((count,), dtype=np.int64)
    widths = [
        np.zeros((entities.count,), dtype=np.int64)
        for entities in mesh.topology.entity_sets
    ]
    ranges: list[tuple[int, np.ndarray, np.ndarray]] = []
    match layout:
        case FormDofLayout():
            offset = 0
            for entity_degree, values in enumerate(layout.canonical_widths):
                starts = offset + np.cumsum(np.r_[0, values[:-1]], dtype=np.int64)
                ranges.append((entity_degree, starts, values))
                offset += int(np.sum(values))
        case NodalDofLayout():
            vertices = mesh.coordinates.shape[0]
            ranges.append((0, np.arange(vertices), np.ones((vertices,), dtype=np.int64)))
            ranges.append((1, layout.edge_starts, layout.edge_widths))
            if dimension == 3:
                ranges.append((2, layout.face_starts, layout.face_widths))
            ranges.append((dimension, layout.cell_starts, layout.cell_widths))
        case FiniteElementDofLayout(association="vertex"):
            ranges.append((0, np.arange(count), np.ones((count,), dtype=np.int64)))
        case FiniteElementDofLayout(association="edge"):
            ranges.append((1, np.arange(count), np.ones((count,), dtype=np.int64)))
        case FiniteElementDofLayout(association="cell"):
            # Cell-major routes (``build_finite_element_dof_routes``): every cell
            # owns one contiguous run of its local DOFs in mesh cell order.
            cells = mesh.topology.entity_sets[dimension].count
            if cells == 0 or count % cells:
                raise ValueError(
                    "finite_element.cell_layout must give every cell one uniform local DOF width; "
                    f"got {count} DOFs on {cells} cells"
                )
            width = count // cells
            ranges.append((
                dimension,
                np.arange(cells, dtype=np.int64) * width,
                np.full((cells,), width, dtype=np.int64),
            ))
        case _:
            raise ValueError(
                "finite_element.quotient_layout.association must number vertex, edge, face, "
                f"or cell entity DOFs; got {layout.association!r}"
            )
    for entity_degree, starts, entity_widths in ranges:
        values = np.asarray(entity_widths, dtype=np.int64)
        index, owner, local = entity_dof_ranges(np.asarray(starts), values)
        degree[index] = entity_degree
        entity[index] = owner
        position[index] = local
        widths[entity_degree] = values
    if np.any(degree < 0):
        raise RuntimeError(
            "finite_element.quotient_layout.lifted_dofs must all have entity assignments; got unassigned lifted DOFs"
        )
    return degree, entity, position, tuple(widths)


def periodic_form_trace_transform(
    trace: AbstractReferenceFormBasis,
    corners: np.ndarray,
    base_corners: np.ndarray,
    permutation: np.ndarray,
    width: int,
    /,
) -> np.ndarray:
    """Pull moments between entity charts rather than oriented incidence loops."""
    copy_chart = np.asarray(trace.canonical_permutation(tuple(corners.tolist())))
    base_chart = np.asarray(trace.canonical_permutation(tuple(base_corners.tolist())))
    canonical = base_chart[permutation[np.argsort(copy_chart)]]
    interior = [
        i for i, label in enumerate(trace.dof_labels) if len(label[0]) == permutation.size
    ]
    transform = np.asarray(trace.permutation_matrix(tuple(canonical.tolist())))[
        np.ix_(interior, interior)
    ]
    if transform.shape != (width, width):
        raise ValueError(
            "finite_element.quotient_layout.trace_transform must match the entity moment width; got incompatible transform dimensions"
        )
    return transform


def periodic_nodal_trace_ordering(
    lifted: NodalDofLayout,
    entity_degree: int,
    base: int,
    copy: int,
    permutation: np.ndarray,
    width: int,
    /,
) -> np.ndarray:
    """Match nodal labels under the authored corner permutation."""
    if (
        lifted.nodal_orbit_families[entity_degree, base]
        != lifted.nodal_orbit_families[entity_degree, copy]
    ):
        raise ValueError(
            "finite_element.quotient_layout.nodal_families must agree across periodic traces; got different scientific reference-source conventions"
        )
    source_keys = lifted.nodal_orbit_keys[entity_degree, base]
    target_keys = lifted.nodal_orbit_keys[entity_degree, copy]
    source_positions = {key: index for index, key in enumerate(source_keys)}
    inverse = np.argsort(permutation)
    mapped_keys = tuple(
        tuple(key[int(index)] for index in inverse) for key in target_keys
    )
    if len(source_positions) != width or any(
        key not in source_positions for key in mapped_keys
    ):
        raise ValueError(
            "finite_element.quotient_layout.nodal_keys must agree under the authored permutation; got incompatible reference traces"
        )
    ordering = np.asarray([source_positions[key] for key in mapped_keys], dtype=np.int64)
    if np.unique(ordering).size != width:
        raise ValueError(
            "finite_element.quotient_layout.nodal_keys must define a bijective orbit; got repeated nodal positions"
        )
    return ordering


class _QuotientRepresentatives(StrictModule):
    dofs: np.ndarray
    boundary: np.ndarray


def _prepare_quotient_representatives(
    periodic: PeriodicMeshTopology,
    dimension: int,
    degree: np.ndarray,
    entity: np.ndarray,
    quotient_index: np.ndarray,
    global_count: int,
    /,
) -> _QuotientRepresentatives:
    """Select lifted representatives and quotient boundary flags for every DOF."""
    representative_dofs = np.full((global_count,), -1, dtype=np.int64)
    on_representative = np.zeros_like(degree, dtype=np.bool_)
    boundary = np.zeros((global_count,), dtype=np.bool_)
    for entity_degree in range(dimension + 1):
        members = degree == entity_degree
        entity_representatives = np.asarray(
            periodic.orbit_representatives(entity_degree), dtype=np.int64
        )
        orbit = np.asarray(periodic.orbits(entity_degree)[0], dtype=np.int64)
        on_representative[members] = (
            entity_representatives[orbit[entity[members]]] == entity[members]
        )
        if entity_degree < dimension:
            quotient_boundary = np.asarray(
                periodic.quotient.entities(entity_degree).subset("boundary").mask,
                dtype=np.bool_,
            )
            boundary[quotient_index[members]] = quotient_boundary[orbit[entity[members]]]
    representative_dofs[quotient_index[on_representative]] = np.flatnonzero(
        on_representative
    )
    if np.any(representative_dofs < 0):
        raise RuntimeError(
            "finite_element.quotient_dofs must have lifted representatives; "
            f"got {np.count_nonzero(representative_dofs < 0)} unassigned DOFs"
        )
    return _QuotientRepresentatives(representative_dofs, boundary)


@checked
def quotient_dof_layout(
    mesh: cell_mesh_owner.CellMesh,
    lifted: FiniteElementDofLayout,
    periodic: PeriodicMeshTopology,
    /,
) -> QuotientDofLayout:
    """Number lifted entity DOFs through periodic orbit identities.

    Entity orientations are relative to the selected lifted representative.
    Nodal traces match under its explicit isometry; compatible moments use
    corner permutations and, for tangential face moments, a dense dual map.
    """

    dimension = mesh.topological_dimension
    degree, entity, position, widths = lifted_dof_entities(mesh, lifted)
    oriented = lifted.conformity in ("Hdiv", "Hcurl")
    quotient_index = np.zeros_like(entity)
    witness = np.ones_like(entity)
    quotient_widths: list[np.ndarray] = []
    entity_transforms: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = 0
    for entity_degree in range(dimension + 1):
        orbit, orientation, _ = (
            np.asarray(value, dtype=np.int64) for value in periodic.orbits(entity_degree)
        )
        representatives = np.asarray(
            periodic.orbit_representatives(entity_degree), dtype=np.int64
        )
        lifted_widths = widths[entity_degree]
        entity_widths = lifted_widths[representatives]
        if np.any(lifted_widths != entity_widths[orbit]):
            raise ValueError(
                "finite_element.quotient_layout.entity_widths must agree across seam copies; "
                "got different DOF widths, a periodic mortar is required"
            )
        permutations = (
            periodic.entity_vertex_permutations(mesh, entity_degree)
            if 0 < entity_degree < dimension and np.any(entity_widths)
            else ()
        )
        loops = (
            {
                int(row): corners
                for rows, group in lifted_loops(mesh, entity_degree)
                for row, corners in zip(rows, group, strict=True)
            }
            if permutations and lifted.association == "form_entity"
            else {}
        )
        entity_starts = cursor + np.cumsum(entity_widths) - entity_widths
        cursor += int(np.sum(entity_widths))
        quotient_widths.append(entity_widths)
        members = degree == entity_degree
        member_orbit = orbit[entity[members]]
        member_witness = (
            orientation[entity[members]] * orientation[representatives[member_orbit]]
        )
        local = position[members].copy()
        quotient_index[members] = entity_starts[member_orbit] + local
        if oriented and entity_degree < dimension:
            witness[members] = member_witness
        if 0 < entity_degree < dimension and np.any(entity_widths):
            for copy in np.flatnonzero(lifted_widths):
                indices = np.flatnonzero(members & (entity == copy))
                base = int(representatives[orbit[copy]])
                width = indices.size
                permutation = permutations[copy]
                match lifted:
                    case FormDofLayout():
                        trace = lifted.canonical_bases[entity_degree, int(copy)]
                        transform = periodic_form_trace_transform(
                            trace, loops[int(copy)], loops[base], permutation, width
                        )
                        entity_transforms.append((indices, transform))
                        witness[indices] = 1
                    case NodalDofLayout():
                        ordering = periodic_nodal_trace_ordering(
                            lifted, entity_degree, base, int(copy), permutation, width
                        )
                        quotient_index[indices] = entity_starts[orbit[copy]] + ordering
                    case FiniteElementDofLayout(association="vertex" | "edge"):
                        pass
                    case _:
                        raise ValueError(
                            f"finite_element.quotient_layout.association must describe supported entity DOFs; got {lifted.association!r}"
                        )
    representatives = _prepare_quotient_representatives(
        periodic, dimension, degree, entity, quotient_index, cursor
    )
    counts = tuple(int(np.sum(value)) for value in quotient_widths)
    per_entity = tuple(uniform_entity_width(value) for value in quotient_widths)
    single = lifted.association in ("vertex", "edge")
    return QuotientDofLayout(
        conformity=lifted.conformity,
        association=f"quotient_{lifted.association}",
        global_count=cursor,
        entity_dof_counts=(0,) * (dimension + 1) if single else counts,
        entity_dofs_per_entity=(1,) * (dimension + 1) if single else per_entity,
        quotient=QuotientDofMap(
            lifted,
            quotient_index.astype(np.int32),
            witness.astype(np.float64),
            representatives.dofs.astype(np.int32),
            representatives.boundary,
            tuple(entity_transforms),
        ),
    )


def quotient_dof_routes(
    quotient: QuotientDofMap,
    global_count: int,
    block_dofs: tuple[Array, ...],
    orientations: tuple[Array, ...],
    /,
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[RowRelation, ...], np.ndarray]:
    """Compose lifted routes with the quotient map.

    Also returns, per quotient DOF, the flat (block, cell, local DOF) route
    position of its representative lifted DOF, which locates its coordinate.
    """

    routes = []
    signs = []
    relations = []
    gather = np.full((global_count,), -1, dtype=np.int64)
    offset = 0
    for lifted_routes, orientation in zip(block_dofs, orientations, strict=True):
        lifted_local = np.asarray(lifted_routes, dtype=np.int64)
        local = quotient.dofs[lifted_local]
        flat = lifted_local.reshape(-1)
        hits = np.flatnonzero(quotient.representatives[quotient.dofs[flat]] == flat)
        targets = quotient.dofs[flat[hits]]
        unset = gather[targets] < 0
        gather[targets[unset]] = offset + hits[unset]
        offset += flat.shape[0]
        routes.append(jnp.asarray(local))
        signs.append(jnp.asarray(np.asarray(orientation) * quotient.signs[lifted_local]))
        relations.append(RowRelation(local, source_size=global_count))
    if np.any(gather < 0):
        raise RuntimeError(
            "finite_element.quotient_layout.representative_routes must cover every DOF; got an unrouted representative"
        )
    return tuple(routes), tuple(signs), tuple(relations), gather


def quotient_cell_transforms(
    quotient: QuotientDofMap,
    lifted_routes: tuple[Array, ...],
    transformations: tuple[Array, ...],
    /,
) -> tuple[Array, ...]:
    """Compose local-from-entity moments with lifted-from-quotient traces."""

    if not quotient.entity_transforms:
        return transformations
    result = []
    for block, routes in enumerate(lifted_routes):
        local = np.asarray(routes)
        count, width = local.shape
        seam = np.broadcast_to(np.eye(width), (count, width, width)).copy()
        for indices, transform in quotient.entity_transforms:
            matching = np.isin(local, indices)
            for cell in np.flatnonzero(np.any(matching, axis=1)):
                positions = np.asarray(
                    [np.flatnonzero(local[cell] == index)[0] for index in indices]
                )
                seam[cell][np.ix_(positions, positions)] = transform
        seam = np.asarray(transformations[block]) @ seam
        result.append(jnp.asarray(seam))
    return tuple(result)


def single_association_dof_layout(
    mesh: cell_mesh_owner.CellMesh,
    conformity: str,
    association: str,
    global_count: int,
    /,
) -> FiniteElementDofLayout:
    # One association owns every DOF, so no per-entity offsets are published.
    entity_count = mesh.topological_dimension + 1
    return SingleAssociationDofLayout(
        conformity=conformity,
        association=association,
        global_count=global_count,
        entity_dof_counts=(0,) * entity_count,
        entity_dofs_per_entity=(1,) * entity_count,
    )


def high_order_h1_dof_layout(
    mesh: cell_mesh_owner.CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    /,
) -> FiniteElementDofLayout:
    edge_widths, face_widths, cell_widths = high_order_h1_entity_widths(mesh, resolved)
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
    per_entity = [1, uniform_entity_width(edge_widths)]
    if mesh.topological_dimension == 3:
        if face_widths is None:
            raise TypeError(
                "finite_element.nodal_layout.face_widths must be allocated in three dimensions; got None"
            )
        counts.append(face_dof_count)
        per_entity.append(uniform_entity_width(face_widths))
    counts.append(cell_dof_count)
    per_entity.append(uniform_entity_width(cell_widths))
    nodal_keys = {}
    nodal_families = {}
    if mesh.periodic_topology is not None:
        nodal_keys, nodal_families = nodal_orbits_owner.nodal_orbit_keys(mesh, resolved)
    return NodalDofLayout(
        conformity="H1",
        association="entity",
        global_count=cursor,
        entity_dof_counts=tuple(counts),
        entity_dofs_per_entity=tuple(per_entity),
        edge_widths=edge_widths,
        edge_starts=edge_starts,
        face_widths=np.empty((0,), dtype=np.int32)
        if face_widths is None
        else face_widths,
        face_starts=np.empty((0,), dtype=np.int32)
        if face_starts is None
        else face_starts,
        cell_starts=cell_starts,
        cell_widths=cell_widths,
        nodal_orbit_keys=nodal_keys,
        nodal_orbit_families=nodal_families,
    )


def polyhedral_h1_entity_routes(
    connectivity: PolyhedralConnectivity, kind: str, vertices: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Lower reference entities through packed vertex incidence."""
    topology = reference_cell_topology(kind)
    edge_lookup = {
        tuple(sorted(row.tolist())): index
        for index, row in enumerate(np.asarray(connectivity.edges))
    }
    offsets = np.asarray(connectivity.face_vertex_offsets)
    values = np.asarray(connectivity.face_vertex_values)
    face_lookup = {
        tuple(sorted(values[start:stop].tolist())): face
        for face, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
    }
    edges = np.asarray(
        [
            [
                edge_lookup[tuple(sorted(cell[list(entity)].tolist()))]
                for entity in topology.entities[1]
            ]
            for cell in vertices
        ],
        dtype=np.int32,
    )
    faces = np.asarray(
        [
            [
                face_lookup[tuple(sorted(cell[list(entity)].tolist()))]
                for entity in topology.entities[2]
            ]
            for cell in vertices
        ],
        dtype=np.int32,
    )
    return edges, faces


def h1_trace_positions(
    mesh: cell_mesh_owner.CellMesh,
    element: FiniteElementSpec,
    vertices: np.ndarray,
    dimension: int,
    entities: np.ndarray,
    signatures: dict[int, np.ndarray] | None = None,
    /,
) -> tuple[np.ndarray, ...]:
    """Prepare nodal permutations and compatibility of shared edge and face traces."""
    if (
        element.representation != "point_value"
        or element.mapping != "identity"
        or element.value_shape
    ):
        raise ValueError(
            "finite_element.element.values must be scalar point_value nodes for hybrid H1 routing; got incompatible representation or mapping"
        )
    topology = reference_cell_topology(element.cell_kind)
    weights = np.asarray(
        cell_geometry_owner.coordinate_lagrange_element(element.cell_kind, 1).tabulate(
            element.reference_nodes
        )[0],
        dtype=np.float64,
    )
    identifiers = np.asarray(mesh.vertex_global_ids)
    tables = []
    for local, dofs in enumerate(element.entity_dofs[dimension]):
        corners = list(topology.entities[dimension][local])
        positions = np.empty((vertices.shape[0], len(dofs)), dtype=np.int32)
        for cell, row in enumerate(vertices):
            order = np.argsort(identifiers[row[corners]], kind="stable")
            trace = weights[np.asarray(dofs, dtype=np.int32)][:, corners][:, order]
            quantized = np.rint(trace * (1 << 40)).astype(np.int64)
            permutation = np.lexsort(
                tuple(quantized[:, axis] for axis in range(len(corners) - 1, -1, -1))
            )
            # On an edge, increasing weight of the high-ID endpoint is the
            # shared edge direction, irrespective of reference node storage order.
            if dimension == 1:
                permutation = np.argsort(quantized[:, 1], kind="stable")
            positions[cell, permutation] = np.arange(len(dofs), dtype=np.int32)
            canonical = quantized[permutation]
            if np.unique(canonical, axis=0).shape[0] != len(dofs):
                raise ValueError(
                    "finite_element.element.reference_nodes.trace must have unique nodal positions; got duplicate positions"
                )
            if signatures is not None:
                entity = int(entities[cell, local])
                previous = signatures.get(entity)
                if previous is not None and not np.array_equal(previous, canonical):
                    raise ValueError(
                        "finite_element.element.reference_nodes.trace must agree on shared hybrid H1 entities; got incompatible traces, a mortar is required"
                    )
                signatures[entity] = canonical
        tables.append(positions)
    return tuple(tables)


def high_order_h1_entity_widths(
    mesh: cell_mesh_owner.CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    /,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return conforming edge, face, and cell-interior DOF widths per entity."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, _HIGH_ORDER_H1_CONNECTIVITIES):
        raise ValueError(
            "finite_element.mesh.connectivity must provide edge and face incidence for high-order H1 routing; got unsupported connectivity"
        )
    edge_widths = np.full((connectivity.edges.shape[0],), -1, dtype=np.int32)
    total_cell_count = sum(block.cell_count for block in mesh.blocks)
    cell_widths = np.empty((total_cell_count,), dtype=np.int32)
    face_widths = None
    face_shapes = None
    if isinstance(
        connectivity,
        (TetrahedralConnectivity, HexahedralConnectivity, PolyhedralConnectivity),
    ):
        face_count = connectivity.boundary_faces.shape[0]
        face_widths = np.full((face_count,), -1, dtype=np.int32)
    if isinstance(connectivity, HexahedralConnectivity):
        face_shapes = np.full((face_count, 2), -1, dtype=np.int32)

    edge_signatures: dict[int, np.ndarray] = {}
    face_signatures: dict[int, np.ndarray] = {}
    cell_offset = 0
    for block, element in zip(mesh.blocks, resolved, strict=True):
        if len(element.entity_dofs[0]) != block.arity:
            raise ValueError(
                "finite_element.element.entity_dofs.vertices must match the cell vertices; got incompatible vertex entities"
            )
        if isinstance(connectivity, PolyhedralConnectivity):
            vertices = np.asarray(block.vertices, dtype=np.int32)
            edges, faces = polyhedral_h1_entity_routes(
                connectivity, block.cell_kind, vertices
            )
            record_shared_entity_widths(
                edge_widths,
                edges,
                element.entity_dofs[1],
                "finite_element.element.entity_dofs.edges must have matching shared H1 widths; got incompatible widths",
            )
            if face_widths is None:
                raise RuntimeError(
                    "finite_element.nodal_layout.face_widths must be allocated for hybrid H1 routing; got None"
                )
            record_shared_entity_widths(
                face_widths,
                faces,
                element.entity_dofs[2],
                "finite_element.element.entity_dofs.faces must have matching shared H1 widths; got incompatible widths",
            )
            h1_trace_positions(mesh, element, vertices, 1, edges, edge_signatures)
            h1_trace_positions(mesh, element, vertices, 2, faces, face_signatures)
        else:
            record_high_order_h1_trace_widths(
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
            raise ValueError(
                "finite_element.element.entity_dofs.cells must contain one top-dimensional entity; got incompatible cell-interior entities"
            )
        cell_widths[cell_offset : cell_offset + block.cell_count] = len(top_entities[0])
        cell_offset += block.cell_count

    if np.any(edge_widths < 0):
        raise ValueError(
            "finite_element.nodal_layout.edge_widths must cover every edge; got unassigned edges"
        )
    if face_widths is not None and np.any(face_widths < 0):
        raise ValueError(
            "finite_element.nodal_layout.face_widths must cover every face; got unassigned faces"
        )
    return edge_widths, face_widths, cell_widths


def record_high_order_h1_trace_widths(
    connectivity: PolygonalConnectivity
    | TetrahedralConnectivity
    | HexahedralConnectivity,
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
    match connectivity:
        case PolygonalConnectivity():
            expected_edge_count = block.arity
        case TetrahedralConnectivity():
            expected_edge_count = len(_TETRAHEDRAL_EDGES)
        case HexahedralConnectivity():
            expected_edge_count = len(_HEXAHEDRAL_EDGES)
        case _:
            raise TypeError(
                "finite_element.mesh.connectivity must provide H1 trace incidence; "
                f"got {type(connectivity).__name__}"
            )
    if local_edge_count != expected_edge_count:
        raise ValueError(
            "finite_element.element.entity_dofs.edges must match reference-cell edges; got incompatible edge entities"
        )
    match connectivity:
        case TetrahedralConnectivity():
            block_cell_edges, _, block_cell_faces = tetrahedral_entity_routes(
                connectivity,
                np.asarray(block.vertices, dtype=np.int32),
            )
        case PolygonalConnectivity() | HexahedralConnectivity():
            block_cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)[
                cell_offset : cell_offset + block.cell_count,
                :local_edge_count,
            ]
        case _:
            raise TypeError(
                "finite_element.mesh.connectivity must provide H1 edge incidence; "
                f"got {type(connectivity).__name__}"
            )
    record_shared_entity_widths(
        edge_widths,
        block_cell_edges,
        element.entity_dofs[1],
        "finite_element.element.entity_dofs.edges must have matching shared trace widths; got incompatible widths, a mortar is required",
    )

    match connectivity:
        case HexahedralConnectivity():
            record_hexahedral_face_widths(
                connectivity, block, element, cell_offset, face_shapes, face_widths
            )
        case TetrahedralConnectivity():
            if face_widths is None:
                raise RuntimeError(
                    "finite_element.nodal_layout.face_widths must be allocated for tetrahedral H1 routing; got None"
                )
            if len(element.entity_dofs[2]) != len(_TETRAHEDRAL_FACES):
                raise ValueError(
                    "finite_element.element.entity_dofs.faces must match tetrahedral faces; got incompatible face entities"
                )
            record_shared_entity_widths(
                face_widths,
                block_cell_faces,
                element.entity_dofs[2],
                "finite_element.element.entity_dofs.faces must have matching shared triangular widths; got incompatible widths, a mortar is required",
            )
        case PolygonalConnectivity():
            pass
        case _:
            raise TypeError(
                "finite_element.mesh.connectivity must provide H1 face incidence; "
                f"got {type(connectivity).__name__}"
            )


def record_shared_entity_widths(
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


def record_hexahedral_face_widths(
    connectivity: HexahedralConnectivity,
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    cell_offset: int,
    face_shapes: np.ndarray | None,
    face_widths: np.ndarray | None,
    /,
) -> None:
    """Assign each shared quadrilateral face one tensor trace shape."""
    if face_shapes is None:
        raise RuntimeError(
            "finite_element.nodal_layout.face_shapes must be allocated for hexahedral H1 routing; got None"
        )
    if len(element.entity_dofs[2]) != len(_HEXAHEDRAL_FACES):
        raise ValueError(
            "finite_element.element.entity_dofs.faces must match hexahedral faces; got incompatible face entities"
        )
    block_cell_faces = np.asarray(
        connectivity.cell_faces,
        dtype=np.int32,
    )[cell_offset : cell_offset + block.cell_count]
    block_face_permutations = np.asarray(
        connectivity.cell_face_vertex_permutations,
        dtype=np.int32,
    )[cell_offset : cell_offset + block.cell_count]
    for local_face in range(len(_HEXAHEDRAL_FACES)):
        local_shape = hexahedral_face_shape(element, local_face)
        for cell in range(block.cell_count):
            face = int(block_cell_faces[cell, local_face])
            canonical_shape = canonical_face_shape(
                block_face_permutations[cell, local_face],
                local_shape,
            )
            existing = tuple(face_shapes[face])
            if existing[0] >= 0 and existing != canonical_shape:
                raise ValueError(
                    "finite_element.element.entity_dofs.faces must have matching shared quadrilateral trace shapes; got incompatible shapes, a mortar is required"
                )
            face_shapes[face] = canonical_shape
            if face_widths is None:
                raise RuntimeError(
                    "finite_element.nodal_layout.face_widths must be allocated for hexahedral H1 routing; got None"
                )
            face_widths[face] = prod(canonical_shape)


@checked
def build_finite_element_dof_routes(
    mesh: cell_mesh_owner.CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    layout: FiniteElementDofLayout,
    /,
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[RowRelation, ...]]:
    block_dofs = []
    orientations = []
    relations = []
    cell_offset = 0
    dof_offset = 0
    for block, element in zip(mesh.blocks, resolved, strict=True):
        vertices = np.asarray(block.vertices, dtype=np.int32)
        match layout:
            case FiniteElementDofLayout(association="cell"):
                width = element.local_dof_count
                local = np.arange(
                    dof_offset,
                    dof_offset + block.cell_count * width,
                    dtype=np.int32,
                ).reshape((block.cell_count, width))
                dof_offset += block.cell_count * width
                orientation = np.ones_like(local, dtype=np.float64)
            case FormDofLayout():
                local = layout.canonical_routes[len(block_dofs)]
                orientation = np.ones_like(local, dtype=np.float64)
            case NodalDofLayout():
                local = high_order_h1_block_routes(
                    mesh, layout, block, element, vertices, cell_offset
                )
                orientation = np.ones_like(local, dtype=np.float64)
            case FiniteElementDofLayout(association="vertex"):
                if element.local_dof_count != vertices.shape[1]:
                    raise ValueError(
                        "finite_element.element.entity_dofs.vertices must contain one DOF per vertex; got incompatible vertex-associated DOFs"
                    )
                local = vertices
                orientation = np.ones_like(local, dtype=np.float64)
            case _:
                raise ValueError(
                    "finite_element.dof_layout must describe supported DOF associations; got an unsupported layout"
                )
        block_dofs.append(jax.device_put(local))
        orientations.append(jax.device_put(orientation))
        relations.append(RowRelation(local, source_size=layout.global_count))
        cell_offset += block.cell_count
    return tuple(block_dofs), tuple(orientations), tuple(relations)


def high_order_h1_block_routes(
    mesh: cell_mesh_owner.CellMesh,
    layout: NodalDofLayout,
    block: CellBlock | PolyhedralBlock,
    element: FiniteElementSpec,
    vertices: np.ndarray,
    cell_offset: int,
    /,
) -> np.ndarray:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, _HIGH_ORDER_H1_CONNECTIVITIES):
        raise RuntimeError(
            "finite_element.mesh.connectivity must support high-order H1 routing; got unsupported connectivity"
        )
    edge_starts = layout.edge_starts
    cell_starts = layout.cell_starts
    local = np.full(
        (block.cell_count, element.local_dof_count),
        -1,
        dtype=np.int32,
    )
    for local_vertex, entity_dofs in enumerate(element.entity_dofs[0]):
        if len(entity_dofs) != 1:
            raise ValueError(
                "finite_element.element.entity_dofs.vertices must contain one DOF per vertex; got incompatible vertex DOFs"
            )
        local[:, entity_dofs[0]] = vertices[:, local_vertex]

    local_edge_count = len(element.entity_dofs[1])
    match connectivity:
        case PolyhedralConnectivity():
            block_cell_edges, block_cell_faces = polyhedral_h1_entity_routes(
                connectivity, block.cell_kind, vertices
            )
        case TetrahedralConnectivity():
            (
                block_cell_edges,
                _,
                block_cell_faces,
            ) = tetrahedral_entity_routes(connectivity, vertices)
        case PolygonalConnectivity() | HexahedralConnectivity():
            rows = slice(cell_offset, cell_offset + block.cell_count)
            block_cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)[
                rows, :local_edge_count
            ]
        case _:
            raise RuntimeError(
                "finite_element.mesh.connectivity must support high-order H1 routing; got unsupported connectivity"
            )
    edge_positions = h1_trace_positions(mesh, element, vertices, 1, block_cell_edges)
    # Edge-interior DOFs follow the shared edge direction.
    for local_edge, entity_dofs in enumerate(element.entity_dofs[1]):
        width = len(entity_dofs)
        if width == 0:
            continue
        canonical_positions = edge_positions[local_edge]
        local[:, np.asarray(entity_dofs, dtype=np.int32)] = (
            edge_starts[block_cell_edges[:, local_edge], None] + canonical_positions
        )

    match connectivity:
        case PolyhedralConnectivity():
            face_positions = h1_trace_positions(
                mesh, element, vertices, 2, block_cell_faces
            )
            for face, dofs in enumerate(element.entity_dofs[2]):
                if dofs:
                    local[:, np.asarray(dofs, dtype=np.int32)] = (
                        layout.face_starts[block_cell_faces[:, face], None]
                        + face_positions[face]
                    )
        case HexahedralConnectivity():
            assign_hexahedral_face_dofs(
                local, connectivity, element, layout.face_starts, cell_offset
            )
        case TetrahedralConnectivity():
            assign_tetrahedral_face_dofs(
                local,
                connectivity,
                element,
                layout.face_starts,
                vertices,
                block_cell_faces,
            )
        case PolygonalConnectivity():
            pass
        case _:
            raise RuntimeError(
                "finite_element.mesh.connectivity must support high-order H1 routing; got unsupported connectivity"
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
        raise ValueError(
            "finite_element.nodal_layout.cell_dofs must assign every local DOF; got unassigned DOFs"
        )
    return local


def assign_hexahedral_face_dofs(
    local: np.ndarray,
    connectivity: HexahedralConnectivity,
    element: FiniteElementSpec,
    face_starts: np.ndarray,
    cell_offset: int,
    /,
) -> None:
    """Route quadrilateral face-interior DOFs through each face's tensor permutation."""
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
        shape = hexahedral_face_shape(element, local_face)
        grid_positions = hexahedral_face_grid_positions(element, local_face, shape)
        for cell in range(cell_count):
            tensor_permutation = quadrilateral_tensor_permutation(
                block_face_permutations[cell, local_face],
                *shape,
            )
            face = block_cell_faces[cell, local_face]
            local[
                cell,
                np.asarray(face_dofs, dtype=np.int32),
            ] = face_starts[face] + tensor_permutation[grid_positions]


def assign_tetrahedral_face_dofs(
    local: np.ndarray,
    connectivity: TetrahedralConnectivity,
    element: FiniteElementSpec,
    face_starts: np.ndarray,
    vertices: np.ndarray,
    block_cell_faces: np.ndarray,
    /,
) -> None:
    """Route triangular face-interior DOFs in shared-face barycentric order."""
    canonical_faces = np.asarray(connectivity.faces, dtype=np.int32)
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
            positions = tetrahedral_face_dof_positions(
                element,
                local_face,
                local_vertices,
                canonical_vertices,
            )
            local[
                cell,
                np.asarray(face_dofs, dtype=np.int32),
            ] = face_starts[face] + positions


@checked
def build_finite_element_dof_coordinates(
    mesh: cell_mesh_owner.CellMesh,
    resolved: tuple[FiniteElementSpec, ...],
    layout: FiniteElementDofLayout,
    block_dofs: tuple[Array, ...],
    coordinate_spec: cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
    coordinate_preparation: _PreparedCoordinateMaps | None = None,
) -> tuple[tuple[Array, ...], tuple[Array, ...], Array, np.ndarray]:
    if coordinate_preparation is None:
        coordinate_preparation = _PreparedCoordinateMaps.prepare(
            mesh, coordinate_spec, budget=budget
        )
    coordinate_preparation.require(mesh, coordinate_spec)
    elements, routes, coordinates = (
        coordinate_preparation.elements,
        coordinate_preparation.routes,
        coordinate_preparation.coordinates,
    )
    weights = []
    for element, field in zip(elements, resolved, strict=True):
        if not isinstance(
            element,
            (
                FiniteElementSpec,
                cell_geometry_owner.BarycentricCellGeometryElement,
                cell_geometry_owner.RestrictedCellGeometryElement,
                cell_geometry_owner.PolynomialComposedCellGeometryElement,
                cell_geometry_owner.RationalComposedCellGeometryElement,
                cell_geometry_owner.SplineCellGeometryElement,
                cell_geometry_owner.LayerColumnCellGeometryElement,
            ),
        ):
            raise TypeError(
                "finite_element.coordinate_elements must provide supported coordinate tabulation; got an unsupported coordinate map"
            )
        weights.append(
            cell_geometry_owner.coordinate_tabulation(element).tabulate(
                field.reference_nodes
            )[0]
        )
    coordinate_weights = tuple(weights)
    match layout:
        case FiniteElementDofLayout(association="cell"):
            boundary = np.zeros((layout.global_count,), dtype=np.bool_)
        case FiniteElementDofLayout(association="edge"):
            connectivity = mesh.connectivity
            if not isinstance(connectivity, _HIGH_ORDER_H1_CONNECTIVITIES):
                raise TypeError(
                    "finite_element.mesh.connectivity must provide edge incidence for compatible coordinates; got unsupported connectivity"
                )
            boundary = np.asarray(connectivity.boundary_edges, dtype=np.bool_)
        case FormDofLayout():
            boundary = layout.canonical_boundary
        case NodalDofLayout():
            boundary = high_order_h1_boundary_mask(mesh, layout)
        case FiniteElementDofLayout(association="vertex"):
            boundary = vertex_boundary_mask(mesh, layout.global_count)
        case _:
            raise ValueError(
                "finite_element.dof_layout must describe supported DOF associations; got an unsupported layout"
            )
    dof_coordinates = averaged_dof_coordinates(
        mesh,
        coordinate_weights,
        routes,
        coordinates,
        block_dofs,
        layout.global_count,
        all(
            field.conformity == "H1" and field.representation == "point_value"
            for field in resolved
        ),
    )
    return coordinate_weights, routes, jax.device_put(boundary), dof_coordinates


def averaged_dof_coordinates(
    mesh: cell_mesh_owner.CellMesh,
    coordinate_weights: tuple[Array, ...],
    coordinate_dofs: tuple[Array, ...],
    coordinates: Array,
    block_dofs: tuple[Array, ...],
    global_count: int,
    require_consistent: bool,
    /,
) -> np.ndarray:
    """Map reference dual points with the coordinate element and routes."""
    values = np.asarray(coordinates)
    accumulated = np.zeros((global_count, mesh.ambient_dimension), dtype=np.float64)
    counts = np.zeros((global_count,), dtype=np.int32)
    lower = np.full_like(accumulated, np.inf) if require_consistent else None
    upper = np.full_like(accumulated, -np.inf) if require_consistent else None
    for weights, geometry_route, field_route in zip(
        coordinate_weights, coordinate_dofs, block_dofs, strict=True
    ):
        mapped = np.asarray(weights) @ values[np.asarray(geometry_route)]
        rows = np.asarray(field_route).reshape(-1)
        np.add.at(accumulated, rows, mapped.reshape((-1, mesh.ambient_dimension)))
        np.add.at(counts, rows, 1)
        if lower is not None and upper is not None:
            np.minimum.at(lower, rows, mapped.reshape((-1, mesh.ambient_dimension)))
            np.maximum.at(upper, rows, mapped.reshape((-1, mesh.ambient_dimension)))
    if np.any(counts == 0):
        raise ValueError(
            "finite_element.dof_coordinates must cover every DOF; got unassigned DOFs"
        )
    averaged = accumulated / counts[:, None]
    if lower is not None and upper is not None:
        tolerance = (
            1024
            * np.finfo(np.float64).eps
            * max(1.0, float(np.max(np.abs(values), initial=0.0)))
        )
        if np.max(upper - lower, initial=0.0) > tolerance:
            raise ValueError(
                "finite_element.dof_coordinates must agree on shared H1 nodes; got inconsistent mapped physical coordinates"
            )
        # Summing k identical contributions and dividing by k can round in
        # binary64; retain the common coordinate when every contribution agrees.
        return np.where(lower == upper, lower, averaged)
    return averaged


def _physical_boundary_entity_masks(
    mesh: cell_mesh_owner.CellMesh, /
) -> tuple[np.ndarray, ...]:
    """Restrict physical boundary receipts, not artificial closure cut facets."""
    topology = mesh.topology
    storage = mesh.storage
    if storage is not None and storage.local_physical_boundary_entities is not None:
        return tuple(
            np.asarray(mask, dtype=np.bool_)
            for mask in storage.local_physical_boundary_entities
        )
    if storage is not None:
        raise ValueError(
            "finite_element.mesh.storage.local_physical_boundary_entities must retain the scientific global boundary at every degree; got missing boundary projection receipts"
        )
    return tuple(
        np.asarray(entities.subset("boundary").mask, dtype=np.bool_)
        for entities in topology.entity_sets
    )


def vertex_boundary_mask(
    mesh: cell_mesh_owner.CellMesh, global_count: int, /
) -> np.ndarray:
    boundary = np.zeros((global_count,), dtype=np.bool_)
    boundary[: mesh.coordinates.shape[0]] = _physical_boundary_entity_masks(mesh)[0]
    return boundary


def high_order_h1_boundary_mask(
    mesh: cell_mesh_owner.CellMesh,
    layout: NodalDofLayout,
    /,
) -> np.ndarray:
    physical = _physical_boundary_entity_masks(mesh)
    boundary = np.zeros((layout.global_count,), dtype=np.bool_)
    boundary[: mesh.coordinates.shape[0]] = physical[0]
    connectivity = mesh.connectivity
    if not isinstance(connectivity, _HIGH_ORDER_H1_CONNECTIVITIES):
        raise TypeError(
            "finite_element.mesh.connectivity must provide edge incidence for boundary H1 routing; got unsupported connectivity"
        )
    edge_starts = layout.edge_starts
    edge_widths = layout.edge_widths
    for edge in np.flatnonzero(physical[1]):
        start = int(edge_starts[edge])
        boundary[start : start + int(edge_widths[edge])] = True
    if isinstance(
        connectivity,
        (TetrahedralConnectivity, HexahedralConnectivity, PolyhedralConnectivity),
    ):
        face_starts = layout.face_starts
        face_widths = layout.face_widths
        for face in np.flatnonzero(physical[2]):
            start = int(face_starts[face])
            boundary[start : start + int(face_widths[face])] = True
    return boundary


def canonical_finite_element_routes(
    mesh: cell_mesh_owner.CellMesh,
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


def validate_restored_fem_state(restored: StrictModule, rebuilt: StrictModule, /) -> None:
    """Compare constructor replay, including scientific source bytes."""
    if (
        type(restored) is not type(rebuilt)
        or array_tree_fingerprint(restored) != array_tree_fingerprint(rebuilt)
        or not eqx.tree_equal(
            eqx.filter(restored, lambda value: not eqx.is_array(value)),
            eqx.filter(rebuilt, lambda value: not eqx.is_array(value)),
            typematch=True,
        )
    ):
        restored_paths = {
            path for path, _ in jax.tree_util.tree_flatten_with_path(restored)[0]
        }
        missing_component = next(
            (
                jax.tree_util.keystr(path)
                for path, _ in jax.tree_util.tree_flatten_with_path(rebuilt)[0]
                if path not in restored_paths
            ),
            None,
        )
        detail = (
            "a type, field, or source mismatch"
            if missing_component is None
            else f"a missing component {missing_component}"
        )
        raise ValueError(
            "finite_element.restored_state must agree with constructor replay; "
            f"got {detail}"
        )


def local_mass_tensor(geometry: FiniteElementBlockGeometry, /) -> Array:
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


def local_stiffness_tensor(geometry: FiniteElementBlockGeometry, /) -> Array:
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


@eqx.filter_jit
def _geometry_tensor_workset(
    weights: Array, basis: Array, /, *, mass: bool, shared_basis: bool
) -> Array:
    """Contract one homogeneous cell workset without block metadata."""
    if shared_basis:
        # Retain the shared-reference contraction's multiply-then-quadrature order.
        weighted = ein.contract("cqi,cq->cqi", basis, weights)
        return ein.contract("cqi,cqj->cij", weighted, basis)
    if mass:
        expression = "cq,cqi,cqj->cij" if basis.ndim == 3 else "cq,cqiv,cqjv->cij"
    else:
        expression = "cq,cqid,cqjd->cij" if basis.ndim == 4 else "cq,cqivd,cqjvd->cij"
    return ein.contract(expression, weights, basis, basis)


type _NumericalWorksetSignature = tuple[tuple[tuple[int, ...], str], ...]


def geometry_local_tensors(
    geometries: tuple[FiniteElementBlockGeometry, ...], /, *, mass: bool
) -> tuple[Array, ...]:
    """Group numerical cell lanes and restore the declared block order."""
    worksets: dict[
        tuple[bool, _NumericalWorksetSignature], list[tuple[int, Array, Array]]
    ] = {}
    for index, geometry in enumerate(geometries):
        weights = geometry.physical_weights
        basis = geometry.basis_values if mass else geometry.physical_gradients
        shared_basis = mass and basis.ndim == 2
        if mass:
            if (
                geometry.coordinate_measure is not None
                and geometry.coordinate_measure.kind != "cartesian"
                and basis.ndim != 3
            ):
                valid = jnp.all(jnp.isfinite(weights) & (weights > 0.0), axis=-1)
                weights = jnp.where(valid[:, None], weights, jnp.nan)
            if basis.ndim == 2:
                basis = jnp.broadcast_to(basis, (weights.shape[0],) + basis.shape)
        signature = tuple(
            (array.shape[1:], np.dtype(array.dtype).str) for array in (weights, basis)
        )
        worksets.setdefault((shared_basis, signature), []).append((index, weights, basis))
    result: dict[int, Array] = {}
    for (shared_basis, _), lanes in worksets.items():
        values = _geometry_tensor_workset(
            jnp.concatenate(tuple(weights for _, weights, _ in lanes)),
            jnp.concatenate(tuple(basis for _, _, basis in lanes)),
            mass=mass,
            shared_basis=shared_basis,
        )
        offset = 0
        for index, weights, _ in lanes:
            stop = offset + weights.shape[0]
            result[index] = values[offset:stop]
            offset = stop
    return tuple(result[index] for index in range(len(geometries)))


@eqx.filter_jit
def _transform_tensor_workset(transform: Array, values: Array, /) -> Array:
    return ein.contract("cai,cab,cbj->cij", transform, values, transform)


def _transformed_local_values(
    transforms: tuple[Array, ...], local_values: Sequence[Array], /
) -> tuple[Array, ...]:
    worksets: dict[_NumericalWorksetSignature, list[tuple[int, Array, Array]]] = {}
    for index, (transform, values) in enumerate(
        zip(transforms, local_values, strict=True)
    ):
        signature = tuple(
            (array.shape[1:], np.dtype(array.dtype).str) for array in (transform, values)
        )
        worksets.setdefault(signature, []).append((index, transform, values))
    result: dict[int, Array] = {}
    for lanes in worksets.values():
        values = _transform_tensor_workset(
            jnp.concatenate(tuple(transform for _, transform, _ in lanes)),
            jnp.concatenate(tuple(values for _, _, values in lanes)),
        )
        offset = 0
        for index, transform, _ in lanes:
            stop = offset + transform.shape[0]
            result[index] = values[offset:stop]
            offset = stop
    return tuple(result[index] for index in range(len(transforms)))


def evaluate_coordinate_map(
    coordinate_element: cell_geometry_owner.ScalarCoordinateTabulation,
    coordinate_routes: ArrayLike,
    coordinates: ArrayLike,
    reference_points: ArrayLike,
    /,
    *,
    precision_policy: FiniteElementPrecisionPolicy,
    paired: bool,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Evaluate the FE coordinate map without changing topology."""

    points = precision_policy.geometry(reference_points)
    geometry_values, geometry_gradients = coordinate_element.tabulate(points)
    geometry_values = precision_policy.geometry(geometry_values)
    geometry_gradients = precision_policy.geometry(geometry_gradients)
    coordinate_values = precision_policy.geometry(coordinates)
    routes = jnp.asarray(coordinate_routes)
    cell_coordinates = coordinate_values[routes]
    if paired:
        if cell_coordinates.shape[0] != points.shape[0]:
            raise ValueError(
                "finite_element.coordinate_map.reference_points must contain one point per cell index; got incompatible point count"
            )
        physical_points = ein.contract("qi,qid->qd", geometry_values, cell_coordinates)
        jacobian = ein.contract("qir,qid->qdr", geometry_gradients, cell_coordinates)
    else:
        physical_points = ein.contract("qi,cid->cqd", geometry_values, cell_coordinates)
        jacobian = ein.contract("qir,cid->cqdr", geometry_gradients, cell_coordinates)
    metric = ein.contract("...di,...dj->...ij", jacobian, jacobian)
    if jacobian.shape[-2] == jacobian.shape[-1]:
        # Rank, condition and measure belong to J itself, not squared J^T J.
        dimension = jacobian.shape[-1]
        if dimension <= 4:
            canonical_columns = jnp.broadcast_to(
                jnp.eye(dimension, dtype=jacobian.dtype), jacobian.shape
            )
            solved = solve_small_linear(
                SmallLinearSolvePlan(dimension), jacobian, canonical_columns
            )
            determinant = solved.determinant
        else:
            tangent = ArraySpace(
                (dimension,),
                dtype=jacobian.dtype,
                space_id=f"finite-element-reference-tangent:{dimension}",
            )
            factors = factorize(
                DenseLinearOperator(
                    jacobian,
                    source=tangent,
                    target=tangent,
                    operator_id=f"finite-element-geometry-jacobian:{dimension}",
                ),
                FactorizationPolicy("lu"),
            )
            solved = factors.materialize_inverse()
            determinant = factors.determinant_sign() * jnp.exp(
                factors.log_abs_determinant()
            )
        inverse_jacobian = solved.value
        determinant = jnp.where(
            solved.successful,
            determinant,
            0.0,
        )
        measure = jnp.abs(determinant)
        gram_determinant = measure * measure
        inverse_metric = ein.contract(
            "...id,...jd->...ij", inverse_jacobian, inverse_jacobian
        )
    else:
        # Embedded cells retain the tangent metric and its native rank evidence.
        dimension = metric.shape[-1]
        if dimension <= 4:
            inverse_result = inverse_small_linear(SmallLinearSolvePlan(dimension), metric)
            gram_determinant = inverse_result.determinant
        else:
            tangent = ArraySpace(
                (dimension,),
                dtype=metric.dtype,
                space_id=f"finite-element-reference-tangent:{dimension}",
            )
            factors = factorize(
                DenseLinearOperator(
                    metric,
                    source=tangent,
                    target=tangent,
                    operator_id=f"finite-element-geometry-gram:{dimension}",
                ),
                FactorizationPolicy("lu"),
            )
            inverse_result = factors.materialize_inverse()
            gram_determinant = jnp.exp(factors.log_abs_determinant())
        inverse_metric = inverse_result.value
        gram_determinant = jnp.where(
            inverse_result.successful,
            gram_determinant,
            0.0,
        )
        measure = jnp.sqrt(gram_determinant)
        inverse_jacobian = ein.contract("...ij,...dj->...id", inverse_metric, jacobian)
        determinant = measure
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


def prepare_local_assembly_relation(
    cell_dofs: Sequence[ArrayLike],
    global_dof_count: int,
    component_shape: tuple[int, ...],
    /,
) -> EdgeRelation:
    """Prepare the fixed COO incidence once, after quotient-aware DOF routing."""
    source_parts: list[npt.NDArray[np.int32]] = []
    target_parts: list[npt.NDArray[np.int32]] = []
    component_count = prod(component_shape) if component_shape else 1
    for cell_routes in cell_dofs:
        indices = np.asarray(cell_routes, dtype=np.int32)
        width = indices.shape[1]
        components = np.arange(component_count, dtype=np.int32)
        flat = indices[..., None] * component_count + components
        source_parts.append(
            np.broadcast_to(
                flat[:, None, :, :],
                (indices.shape[0], width, width, component_count),
            ).reshape((-1,))
        )
        target_parts.append(
            np.broadcast_to(
                flat[:, :, None, :],
                (indices.shape[0], width, width, component_count),
            ).reshape((-1,))
        )
    return EdgeRelation(
        np.concatenate(source_parts),
        np.concatenate(target_parts),
        source_size=global_dof_count * component_count,
        target_size=global_dof_count * component_count,
    )


def assemble_local_operator(
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
    coefficient_parts = []
    component_count = prod(dof_map.component_shape) if dof_map.component_shape else 1
    for values in _transformed_local_values(dof_map.cell_transforms, local_values):
        coefficient_parts.append(
            jnp.broadcast_to(
                jnp.asarray(values)[..., None],
                values.shape + (component_count,),
            ).reshape((-1,))
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
        raise TypeError(
            "finite_element.local_operator.properties must be OperatorProperties or None; got an incompatible object"
        )
    return SparseLinearMap(
        dof_map.assembly_relation,
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
